"""Where a decision becomes an order, a fill, a position and eventually a trade.

Two properties are load-bearing:

**Idempotency.** The client order id is derived from the cycle id and the action,
so replaying a cycle after a crash finds the order already there and returns it
instead of buying twice. The ``(mode, client_order_id)`` unique constraint is the
backstop if two processes ever race.

**Every entry carries its own training data.** The feature snapshot that engine_3
computed for the cycle is written onto the position and copied onto the trade when
it closes. That is the whole reason the risk engine has anything to learn from:
the X was recorded at the moment of the decision, not reconstructed afterwards
from data that already knows how it ended.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from core.contracts import BUY, HOLD, AdminRestrictions, AgentDecision, PortfolioState, utcnow
from core.db import session_scope
from core.repository import _aware, record_event
from core.tables import Order, Trade

from . import risk_guard
from .broker import Broker, BrokerError, Fill
from .portfolio import PortfolioStore

log = logging.getLogger("execution.trader")


class Trader:
    def __init__(self, broker: Broker, store: PortfolioStore | None = None):
        self.broker = broker
        self.store = store or PortfolioStore(broker.mode)

    # ── public ──────────────────────────────────────────────────────────────
    def execute(self, decision: AgentDecision, *, symbol: str, price: float,
                portfolio: PortfolioState, restrictions: AdminRestrictions,
                cycle_id: str, decision_id: int | None = None,
                context: dict | None = None, reason: str = "agent") -> dict[str, Any]:
        """Run the guard, place the order, write the books. Returns a report."""
        if decision.action == HOLD:
            return {"executed": False, "action": HOLD, "reasons": ["decision is HOLD"]}

        guard = risk_guard.check(decision, portfolio=portfolio, restrictions=restrictions,
                                 price=price, symbol=symbol)
        if not guard.allowed:
            log.info("order blocked (%s): %s", decision.action, "; ".join(guard.reasons))
            record_event(f"{decision.action} blocked: {'; '.join(guard.reasons)}",
                         category="execution", mode=self.broker.mode,
                         payload={"cycle_id": cycle_id, **guard.to_dict()})
            return {"executed": False, "action": decision.action,
                    "reasons": guard.reasons, "blocked": True}

        client_order_id = f"{cycle_id}-{decision.action.lower()}"
        existing = self._existing_order(client_order_id)
        if existing:
            log.info("order %s already filled; not repeating it", client_order_id)
            return {"executed": True, "idempotent": True, "order_id": existing["id"],
                    "action": decision.action, "reasons": ["already executed"]}

        try:
            if decision.action == BUY:
                fill = self.broker.buy(symbol, guard.quote, client_order_id=client_order_id,
                                       reference_price=price)
            else:
                fill = self.broker.sell(symbol, guard.quantity, client_order_id=client_order_id,
                                        reference_price=price)
        except BrokerError as exc:
            log.error("broker refused the order: %s", exc)
            record_event(f"broker error on {decision.action}: {exc}", level="error",
                         category="execution", mode=self.broker.mode,
                         payload={"cycle_id": cycle_id})
            return {"executed": False, "action": decision.action, "error": str(exc)}
        except Exception as exc:
            log.exception("order failed")
            record_event(f"order failed on {decision.action}: {exc}", level="error",
                         category="execution", mode=self.broker.mode,
                         payload={"cycle_id": cycle_id})
            return {"executed": False, "action": decision.action, "error": str(exc)}

        order_id = self._record_order(fill, cycle_id=cycle_id, decision_id=decision_id,
                                      reason=reason)
        if fill.side == BUY:
            report = self._apply_buy(fill, decision, cycle_id, decision_id, order_id, context)
        else:
            report = self._apply_sell(fill, cycle_id, decision_id, order_id, reason)

        self.store.record_equity(symbol, price, cycle_id=cycle_id)
        report.update({"executed": True, "action": decision.action, "order_id": order_id,
                       "fill": fill.to_dict(), "reasons": guard.reasons})
        record_event(f"{fill.side} {fill.quantity:.8f} {symbol} @ {fill.price:,.2f} "
                     f"({self.broker.mode})", category="execution", mode=self.broker.mode,
                     payload={"cycle_id": cycle_id, "order_id": order_id,
                              "quote": fill.quote_amount, "fee": fill.fee, "reason": reason})
        return report

    # ── bookkeeping ─────────────────────────────────────────────────────────
    def _existing_order(self, client_order_id: str) -> dict | None:
        with session_scope() as s:
            row = s.execute(select(Order).where(Order.mode == self.broker.mode,
                                                Order.client_order_id == client_order_id)
                            ).scalar_one_or_none()
            return {"id": row.id} if row else None

    def _record_order(self, fill: Fill, *, cycle_id: str, decision_id: int | None,
                      reason: str) -> int:
        with session_scope() as s:
            row = Order(mode=self.broker.mode, cycle_id=cycle_id, decision_id=decision_id,
                        symbol=fill.symbol, side=fill.side, quantity=fill.quantity,
                        price=fill.price, quote_amount=fill.quote_amount, fee=fill.fee,
                        slippage=fill.slippage, status=fill.status, broker=fill.broker,
                        client_order_id=fill.client_order_id,
                        exchange_order_id=fill.exchange_order_id, reason=reason,
                        filled_at=utcnow(), raw=fill.raw)
            s.add(row)
            s.flush()
            return row.id

    def _apply_buy(self, fill: Fill, decision: AgentDecision, cycle_id: str,
                   decision_id: int | None, order_id: int, context: dict | None) -> dict:
        self.store.add_ledger("buy", fill.quote_amount, ref=str(order_id),
                              note=f"{fill.quantity:.8f} {fill.symbol} @ {fill.price:.2f}")
        existing = self.store.position_row(fill.symbol) or {}
        held = float(existing.get("quantity") or 0.0)
        old_cost = held * float(existing.get("avg_entry_price") or 0.0)
        new_qty = held + fill.quantity
        avg = (old_cost + fill.quantity * fill.price) / new_qty if new_qty > 0 else fill.price
        self.store.upsert_position(
            fill.symbol, quantity=new_qty, avg_entry_price=avg,
            opened_at=existing.get("opened_at") or utcnow(),
            stop_price=decision.stop_price, target_price=decision.target_price,
            bars_held=int(existing.get("bars_held") or 0),
            entry_cycle_id=cycle_id, entry_decision_id=decision_id,
            entry_confidence=decision.confidence, context=context or {})
        return {"position_quantity": new_qty, "avg_entry_price": avg}

    def _apply_sell(self, fill: Fill, cycle_id: str, decision_id: int | None,
                    order_id: int, reason: str) -> dict:
        self.store.add_ledger("sell", fill.quote_amount, ref=str(order_id),
                              note=f"{fill.quantity:.8f} {fill.symbol} @ {fill.price:.2f}")
        pos = self.store.position_row(fill.symbol) or {}
        entry_price = float(pos.get("avg_entry_price") or fill.price)
        entry_at = _aware(pos.get("opened_at"))
        stop = pos.get("stop_price")
        gross_entry = entry_price * fill.quantity
        pnl = fill.quote_amount - gross_entry          # quote_amount is net of fees
        pnl_pct = (pnl / gross_entry * 100.0) if gross_entry > 0 else 0.0
        risk_per_unit = (entry_price - float(stop)) if stop else 0.0
        r_multiple = (pnl / (risk_per_unit * fill.quantity)) if risk_per_unit > 0 else 0.0
        holding_min = ((utcnow() - entry_at).total_seconds() / 60.0) if entry_at else 0.0

        with session_scope() as s:
            trade = Trade(
                mode=self.broker.mode, symbol=fill.symbol,
                entry_order_id=None, exit_order_id=order_id, decision_id=decision_id,
                entry_cycle_id=pos.get("entry_cycle_id"), exit_cycle_id=cycle_id,
                quantity=fill.quantity, entry_price=entry_price, exit_price=fill.price,
                entry_at=entry_at, exit_at=utcnow(), holding_minutes=round(holding_min, 2),
                fees=fill.fee, pnl_quote=round(pnl, 8), pnl_pct=round(pnl_pct, 6),
                r_multiple=round(r_multiple, 4), exit_reason=reason,
                stop_price=stop, target_price=pos.get("target_price"),
                entry_confidence=float(pos.get("entry_confidence") or 0.0),
                context=pos.get("context") or {})
            s.add(trade)
            s.flush()
            trade_id = trade.id

        remaining = max(0.0, float(pos.get("quantity") or 0.0) - fill.quantity)
        if remaining <= 1e-12:
            self.store.clear_position(fill.symbol)
        else:
            self.store.upsert_position(fill.symbol, quantity=remaining)
        log.info("closed %s: pnl %.2f (%.2f%%, %.2fR) after %.0f minutes",
                 fill.symbol, pnl, pnl_pct, r_multiple, holding_min)
        return {"trade_id": trade_id, "pnl_quote": round(pnl, 8),
                "pnl_pct": round(pnl_pct, 4), "r_multiple": round(r_multiple, 4),
                "position_quantity": remaining}
