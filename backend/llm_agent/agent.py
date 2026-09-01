"""The Agent — four brains, one voice.

This is the only component allowed to produce something that becomes an order.
It takes the two forecasting engines, the risk engine, the book, the operator's
restrictions and the knowledge base, and returns one answer in plain English with
the numbers attached.

Three things are non-negotiable in here:

1. **The restrictions are enforced after the model speaks, not merely described to
   it.** The prompt tells the Agent what it may not do; :func:`enforce` then checks
   the answer against the same rules and rewrites it if it strayed. Every
   correction is recorded in ``compliance_notes``, so a model that keeps trying to
   exceed a cap is visible rather than silently clipped.
2. **A missing LLM is not a missing decision.** Anything unusable — no key, a
   timeout, malformed JSON, a hallucinated action — falls through to the
   deterministic policy in :mod:`llm_agent.fallback`.
3. **The knowledge base is read at the moment of the decision.** An edit made a
   minute ago is in this prompt; no restart, no redeploy.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

from core.contracts import (ACTIONS, BUY, HOLD, SELL, AdminRestrictions, AgentDecision,
                            EngineSignal, PortfolioState, RiskAssessment, clamp, utcnow)

from . import fallback, prompt as P
from .client import LLMClient, get_client
from .knowledge_base import KnowledgeBaseStore, get_store

log = logging.getLogger("llm_agent.agent")


class TradingAgent:
    def __init__(self, client: LLMClient | None = None,
                 kb_store: KnowledgeBaseStore | None = None):
        self.client = client or get_client()
        self.kb = kb_store or get_store()

    # ── the decision ────────────────────────────────────────────────────────
    def decide(self, *, symbol: str, mode: str, price: float,
               signals: Sequence[EngineSignal], risk: RiskAssessment | None,
               portfolio: PortfolioState, restrictions: AdminRestrictions,
               extra_context: dict | None = None) -> AgentDecision:
        kb = self.kb.get()
        sections = kb.retrieve(
            P.situation_terms(signals=signals, risk=risk, portfolio=portfolio),
            max_sections=self.kb_limit_sections(), max_chars=self.kb_limit_chars())
        knowledge = kb.render(sections)

        max_quote = self._max_quote(portfolio, restrictions)
        user = P.build_user_prompt(
            symbol=symbol, mode=mode, price=price, signals=signals, risk=risk,
            portfolio=portfolio, restrictions=restrictions, knowledge=knowledge,
            max_size_quote=max_quote,
            extra_context={**(extra_context or {}), "now": utcnow().strftime("%Y-%m-%d %H:%M UTC")})

        response = self.client.complete(P.SYSTEM_PROMPT, user)
        if response.ok and response.parsed:
            decision = self._from_llm(response.parsed, price=price, portfolio=portfolio,
                                      signals=signals)
            decision.source = response.source
            decision.raw_llm = {"latency_ms": response.latency_ms,
                                "usage": response.usage or {},
                                "parsed": response.parsed}
        else:
            log.warning("no usable LLM answer (%s) — deterministic policy takes over",
                        response.error)
            decision = fallback.decide(signals=signals, risk=risk, portfolio=portfolio,
                                       restrictions=restrictions, price=price)
            decision.raw_llm = {"error": response.error, "source": response.source}

        decision.kb_version = kb.version
        decision.admin_version = restrictions.version
        decision.degraded = any(not s.ok for s in signals) or not kb.ok
        decision.used_theories = decision.used_theories or [s.title for s in sections[:3]]
        return enforce(decision, signals=signals, risk=risk, portfolio=portfolio,
                       restrictions=restrictions, price=price, symbol=symbol)

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def kb_limit_sections() -> int:
        from core.config import get_settings
        return get_settings().kb.max_sections_in_prompt

    @staticmethod
    def kb_limit_chars() -> int:
        from core.config import get_settings
        return get_settings().kb.max_chars_in_prompt

    @staticmethod
    def _max_quote(portfolio: PortfolioState, restrictions: AdminRestrictions) -> float:
        return min(portfolio.cash, portfolio.equity * restrictions.max_position_pct / 100.0)

    @staticmethod
    def _from_llm(parsed: dict, *, price: float, portfolio: PortfolioState,
                  signals: Sequence[EngineSignal]) -> AgentDecision:
        def _num(key: str, default: float | None = None) -> float | None:
            v = parsed.get(key)
            try:
                f = float(v)
            except (TypeError, ValueError):
                return default
            return f if f > 0 else default

        def _list(key: str) -> list[str]:
            v = parsed.get(key)
            if isinstance(v, str):
                return [v]
            return [str(x) for x in v][:6] if isinstance(v, list) else []

        action = str(parsed.get("action", HOLD)).strip().upper()
        return AgentDecision(
            action=action if action in ACTIONS else HOLD,
            confidence=clamp(parsed.get("confidence", 0.0), 0.0, 1.0),
            size_pct=clamp(parsed.get("size_pct", 0.0), 0.0, 100.0),
            entry_price=_num("entry_price", price),
            stop_price=_num("stop_price"),
            target_price=_num("target_price"),
            time_horizon=str(parsed.get("time_horizon", ""))[:64],
            rationale=str(parsed.get("rationale", "")).strip()[:4000],
            key_risks=_list("key_risks"),
            change_my_mind=_list("change_my_mind"),
            used_theories=_list("used_theories"),
            engine_agreement=str(parsed.get("engine_agreement", ""))[:16].lower())


# ── the constitution ────────────────────────────────────────────────────────
def enforce(decision: AgentDecision, *, signals: Sequence[EngineSignal],
            risk: RiskAssessment | None, portfolio: PortfolioState,
            restrictions: AdminRestrictions, price: float,
            symbol: str, now: datetime | None = None) -> AgentDecision:
    """Rewrite anything the Agent proposed that it is not allowed to do.

    Every rewrite appends a line to ``compliance_notes``. Nothing is silently
    clipped: an operator reading the decision can see exactly which limit bit and
    what the Agent wanted to do instead.
    """
    now = now or utcnow()
    notes: list[str] = list(decision.compliance_notes)

    def block(reason: str, to_action: str = HOLD) -> None:
        notes.append(f"{decision.action} -> {to_action}: {reason}")
        decision.action = to_action
        decision.size_pct = 0.0
        decision.size_quote = 0.0

    # symbol and action permissions
    if restrictions.allowed_symbols and symbol not in restrictions.allowed_symbols:
        block(f"{symbol} is not in the allowed symbol list")
    if symbol in (restrictions.blocked_symbols or []):
        block(f"{symbol} is on the blocked list")
    if decision.action not in (restrictions.allowed_actions or list(ACTIONS)):
        block(f"{decision.action} is not an allowed action right now")

    # entry gates
    if decision.action == BUY:
        if restrictions.kill_switch:
            block("kill switch is on; exits only")
        elif not restrictions.allow_new_entries:
            block("new entries are administratively paused")
        elif risk is not None and risk.veto:
            block("risk engine vetoed a new entry: " +
                  (risk.veto_reasons[0] if risk.veto_reasons else "no reason given"))
        elif decision.confidence < restrictions.min_confidence:
            block(f"confidence {decision.confidence:.2f} is below the "
                  f"{restrictions.min_confidence:.2f} floor")
        elif portfolio.in_position and restrictions.max_open_positions <= 1:
            block("already holding the maximum number of positions")
        elif portfolio.trades_today >= restrictions.max_trades_per_day:
            block(f"daily trade cap reached ({portfolio.trades_today}/"
                  f"{restrictions.max_trades_per_day})")
        elif restrictions.trading_hours_utc and now.hour not in restrictions.trading_hours_utc:
            block(f"{now:%H:%M} UTC is outside the permitted trading hours")
        elif now.strftime("%Y-%m-%d") in (restrictions.blackout_dates or []):
            block(f"{now:%Y-%m-%d} is a blackout date")

    # exits must have something to exit
    if decision.action == SELL and not portfolio.in_position:
        block("nothing is held, so there is nothing to sell (spot book: bearish = cash)")

    # stop / target sanity for a live entry
    if decision.action == BUY:
        entry = decision.entry_price or price
        stop = decision.stop_price
        if restrictions.require_stop_loss and (not stop or stop <= 0):
            stop = round(entry * (1 - restrictions.max_stop_distance_pct / 200.0), 2)
            notes.append(f"no stop was supplied; defaulted to {stop:,.2f}")
        if stop and stop >= entry:
            stop = round(entry * 0.985, 2)
            notes.append(f"stop was at or above entry; moved to {stop:,.2f}")
        widest = entry * (1 - restrictions.max_stop_distance_pct / 100.0)
        if stop and stop < widest:
            notes.append(f"stop {stop:,.2f} was wider than the "
                         f"{restrictions.max_stop_distance_pct:.1f}% limit; tightened to "
                         f"{widest:,.2f}")
            stop = round(widest, 2)
        if decision.target_price and decision.target_price <= entry:
            decision.target_price = round(entry * 1.02, 2)
            notes.append("target was at or below entry; set to +2%")
        decision.entry_price, decision.stop_price = entry, stop

        # size: the smaller of the position cap, the risk budget and the risk
        # engine's own multiplier. The model may ask for less, never for more.
        stop_distance_pct = max(0.05, (entry - (stop or entry * 0.985)) / entry * 100.0)
        risk_cap = restrictions.max_capital_at_risk_pct / stop_distance_pct * 100.0
        ceiling = min(restrictions.max_position_pct, risk_cap)
        if risk is not None:
            ceiling *= max(0.0, min(1.0, risk.size_multiplier))
        if decision.size_pct <= 0:
            decision.size_pct = round(ceiling, 3)
            notes.append(f"no size was supplied; sized to {decision.size_pct:.2f}% from the "
                         f"risk budget")
        elif decision.size_pct > ceiling:
            notes.append(f"requested {decision.size_pct:.2f}% of equity; capped to "
                         f"{ceiling:.2f}% ({restrictions.max_capital_at_risk_pct:.2f}% risk "
                         f"over a {stop_distance_pct:.2f}% stop)")
            decision.size_pct = round(ceiling, 3)
        decision.size_quote = round(min(portfolio.cash,
                                        portfolio.equity * decision.size_pct / 100.0), 2)
        if decision.size_quote < restrictions.min_order_quote:
            block(f"size {decision.size_quote:,.2f} is below the minimum order of "
                  f"{restrictions.min_order_quote:,.2f}")
    else:
        decision.size_pct, decision.size_quote = 0.0, 0.0
        if decision.action == HOLD:
            decision.entry_price = decision.entry_price or price

    if not decision.rationale:
        decision.rationale = ("No rationale was produced; the decision stands on the "
                              "engines' numbers and the operator's limits alone.")
    if not decision.change_my_mind:
        decision.change_my_mind = ["A change in either engine's direction, or a new "
                                   "risk-model version, would be reviewed next cycle."]
    decision.compliance_notes = notes[:10]
    return decision


_agent: TradingAgent | None = None


def get_agent() -> TradingAgent:
    global _agent
    if _agent is None:
        _agent = TradingAgent()
    return _agent
