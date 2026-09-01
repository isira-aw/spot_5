"""One cycle: price in, four brains, one answer, at most one order.

    price -> protective exits -> engine_1 ‖ engine_2 -> engine_3 -> Agent
          -> constitution -> risk guard -> broker -> books

Order of operations is deliberate. Stops are checked **before** anything is asked
of any model: a stop is arithmetic, and waiting on a language model to agree with
arithmetic is how accounts die. The two forecasting engines run concurrently
because neither depends on the other. The risk engine runs after them because it
scores their agreement. The Agent runs last because it is the only one allowed to
decide.

Every cycle is written to the database as a unit — the signals, the risk assessment,
the decision, the order — so any decision can be reconstructed months later with
the exact inputs that produced it.
"""
from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

from adapters.engine_one import EngineOneAdapter
from adapters.engine_two import EngineTwoAdapter
from core import repository
from core.config import get_settings
from core.contracts import (BUY, SELL, AgentDecision, CycleResult, EngineSignal,
                            sane_levels, utcnow)
from core.market import get_ohlcv, get_quote
from engine_3.service import get_risk_engine
from execution import risk_guard
from execution.broker import Broker, make_broker
from execution.portfolio import PortfolioStore
from execution.trader import Trader
from llm_agent.agent import TradingAgent, get_agent

log = logging.getLogger("pipeline.orchestrator")


class Orchestrator:
    def __init__(self, mode: str | None = None, broker: Broker | None = None,
                 agent: TradingAgent | None = None):
        s = get_settings()
        self.settings = s
        self.mode = (mode or s.execution.mode).upper()
        self.symbol = s.execution.symbol
        self.broker = broker or make_broker(self.mode)
        self.store = PortfolioStore(self.mode)
        self.trader = Trader(self.broker, self.store)
        self.agent = agent or get_agent()
        self.risk_engine = get_risk_engine()
        self.engine_1 = EngineOneAdapter()
        self.engine_2 = EngineTwoAdapter()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cycle")

    # ── the cycle ───────────────────────────────────────────────────────────
    def run_cycle(self, *, autotrade: bool | None = None,
                  force: bool = False) -> CycleResult:
        autotrade = self.settings.autotrade if autotrade is None else autotrade
        cycle_id = f"{self.mode.lower()}-{utcnow():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        result = CycleResult(cycle_id=cycle_id, mode=self.mode, symbol=self.symbol,
                             started_at=utcnow())
        log.info("── cycle %s (%s %s) ──", cycle_id, self.mode, self.symbol)

        try:
            price = self._price()
            result.price = price
        except Exception as exc:
            result.status, result.error = "failed", f"no price: {exc}"
            result.finished_at = utcnow()
            log.error("cycle aborted: %s", result.error)
            repository.save_cycle(result)
            return result

        restrictions = repository.active_admin_rules()
        killed = risk_guard.is_killed() or restrictions.kill_switch
        portfolio = self.store.state(self.symbol, price, kill_switch=killed)

        # 1. Stops and targets first, before any model is consulted.
        protective = risk_guard.protective_exit(portfolio, price)
        if protective:
            log.warning("%s hit at %.2f — exiting before consulting the engines",
                        protective, price)
            exit_decision = self._protective_decision(protective, portfolio, price)
            decision_id = self._persist(result, decision=exit_decision, status="protective_exit")
            order = self.trader.execute(exit_decision, symbol=self.symbol, price=price,
                                        portfolio=portfolio, restrictions=restrictions,
                                        cycle_id=cycle_id, decision_id=decision_id,
                                        reason=protective)
            result.order = order
            result.decision = exit_decision
            result.portfolio = self.store.state(self.symbol, price, kill_switch=killed)
            result.status = "protective_exit"
            result.finished_at = utcnow()
            self._persist(result, decision=exit_decision, status="protective_exit")
            return result

        # 2. The two forecasting engines, concurrently.
        context = {"position": (portfolio.position.to_dict() if portfolio.position else {}),
                   "force": force}
        futures = {"engine_1": self._pool.submit(self.engine_1.run, self.symbol, context),
                   "engine_2": self._pool.submit(self.engine_2.run, self.symbol, context)}
        signals: list[EngineSignal] = []
        for name, fut in futures.items():
            try:
                signals.append(fut.result(timeout=self._engine_timeout(name)))
            except Exception as exc:
                log.error("%s did not return: %s", name, exc)
                signals.append(EngineSignal.failed(name, self.symbol, str(exc)))
        result.signals = signals
        for sig in signals:
            log.info("  %s: %s conf=%.2f %s", sig.engine, sig.direction, sig.confidence,
                     "" if sig.ok else f"(DOWN: {sig.error})")

        # 3. The risk engine scores the shape of this setup against our own history.
        candles = get_ohlcv(self.symbol, "1h", 48)
        intent = self._intent(signals, price)
        risk = self.risk_engine.assess(signals=signals, portfolio=portfolio,
                                       candles=candles, intent=intent,
                                       restrictions=restrictions)
        result.risk = risk
        log.info("  engine_3: p(win)=%.2f ev=%+.2fR size=%.2f regime=%s%s",
                 risk.win_probability, risk.expected_r, risk.size_multiplier,
                 risk.regime, " VETO" if risk.veto else "")

        # 4. The Agent decides, in words.
        decision = self.agent.decide(
            symbol=self.symbol, mode=self.mode, price=price, signals=signals, risk=risk,
            portfolio=portfolio, restrictions=restrictions,
            extra_context={"recent_decisions": self._recent_decision_lines()})
        result.decision = decision
        result.portfolio = portfolio
        log.info("  AGENT: %s conf=%.2f size=%.2f%% via %s", decision.action,
                 decision.confidence, decision.size_pct, decision.source)

        decision_id = self._persist(result, decision=decision, status="ok")

        # 5. Execution, if the operator has left it switched on.
        if autotrade and decision.action in (BUY, SELL):
            order = self.trader.execute(
                decision, symbol=self.symbol, price=price, portfolio=portfolio,
                restrictions=restrictions, cycle_id=cycle_id, decision_id=decision_id,
                context={"features": risk.features, "decision": decision.to_dict(),
                         "signals": [s.to_dict() for s in signals]},
                reason="agent")
            result.order = order
            if not order.get("executed"):
                result.blocked_by = order.get("reasons", [])
        elif decision.action in (BUY, SELL):
            result.blocked_by = ["autotrade is off; decision recorded but not executed"]

        # 6. Housekeeping: mark another cycle held, snapshot the curve.
        self.store.tick_bars_held(self.symbol)
        self.store.record_equity(self.symbol, price, cycle_id=cycle_id)
        result.portfolio = self.store.state(self.symbol, price, kill_switch=killed)
        result.finished_at = utcnow()
        self._persist(result, decision=decision, status=result.status)
        log.info("── cycle %s done in %dms ──", cycle_id,
                 int((result.finished_at - result.started_at).total_seconds() * 1000))
        return result

    # ── helpers ─────────────────────────────────────────────────────────────
    def _price(self) -> float:
        try:
            return self.broker.price(self.symbol)
        except Exception as exc:
            log.warning("broker price failed (%s); using the public feed", exc)
            quote = get_quote(self.symbol)
            if not quote.ok:
                raise RuntimeError(quote.error or "no price source answered") from exc
            return quote.price

    def _engine_timeout(self, name: str) -> int:
        e = self.settings.engines
        return (e.engine_1_timeout_s if name == "engine_1" else e.engine_2_timeout_s) + 15

    def _intent(self, signals: list[EngineSignal], price: float) -> dict:
        """The setup engine_3 is being asked about, before the Agent has spoken."""
        stop = target = 0.0
        conf = 0.0
        for s in signals:
            if not s.ok:
                continue
            lv = s.levels or {}
            stop = stop or float(lv.get("stop_loss") or lv.get("reference_stop") or 0.0)
            target = target or float(lv.get("take_profit_1") or lv.get("reference_target") or 0.0)
            conf = max(conf, s.confidence)
        caps = self.settings.caps
        stop, target = sane_levels(price, stop, target,
                                   default_stop_pct=caps.stop_loss_pct,
                                   default_target_pct=caps.take_profit_pct)
        return {"price": price, "stop_price": stop, "target_price": target,
                "confidence": conf}

    def _protective_decision(self, reason: str, portfolio, price: float) -> AgentDecision:
        pos = portfolio.position
        level = pos.stop_price if reason == "stop_loss" else pos.target_price
        word = "stop" if reason == "stop_loss" else "target"
        return AgentDecision(
            action=SELL, confidence=1.0, size_pct=0.0,
            entry_price=price, stop_price=pos.stop_price, target_price=pos.target_price,
            engine_agreement="n/a",
            rationale=(f"The {word} at {level:,.2f} was reached with price at {price:,.2f}, "
                       f"so I closed the position without waiting for the engines. That level "
                       f"was set when the trade was opened as the point where the reasoning "
                       f"would be wrong ({word} hit after {pos.bars_held} cycles, entry "
                       f"{pos.avg_entry_price:,.2f}). Risk exits are arithmetic, not opinion."),
            key_risks=["A stop can be hit by a wick and reverse; that is the cost of "
                       "having one."],
            change_my_mind=["Nothing on this cycle — the exit is already done. A fresh setup "
                            "is judged from scratch next cycle."],
            used_theories=["What Would Change the View", "Risk of Ruin and Position Sizing"],
            source="risk_guard", compliance_notes=[f"protective exit: {reason}"])

    def _recent_decision_lines(self, limit: int = 4) -> list[str]:
        try:
            rows = repository.recent_cycles(self.mode, limit=limit + 1)[1:]
        except Exception:
            return []
        out = []
        for r in rows:
            if not r.get("action"):
                continue
            out.append(f"{r['started_at']:%H:%M} — {r['action']} at "
                       f"{r.get('confidence') or 0:.2f} confidence "
                       f"(price {r.get('price') or 0:,.0f})")
        return out

    def _persist(self, result: CycleResult, *, decision, status: str) -> int | None:
        result.status = status
        try:
            return repository.save_cycle(result)
        except Exception as exc:
            log.error("cycle not persisted: %s", exc)
            return None


_orchestrator: Orchestrator | None = None


def get_orchestrator(mode: str | None = None) -> Orchestrator:
    global _orchestrator
    if _orchestrator is None or (mode and _orchestrator.mode != mode.upper()):
        _orchestrator = Orchestrator(mode)
    return _orchestrator


def reset_orchestrator() -> None:
    global _orchestrator
    _orchestrator = None
