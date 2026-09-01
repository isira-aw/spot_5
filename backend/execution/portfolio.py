"""The books — one per mode, and never the two shall meet.

PAPER and REAL are the same code path with a different broker and a different
``mode`` value on every row. Cash, positions, trades, the equity curve and the
ledger are all filtered by mode, so switching modes shows you a different account,
not a merged one, and nothing that happens in PAPER can touch REAL history.

Cash is derived from the ledger, not stored as a mutable number, so the balance is
always reconstructible from its entries — which is what makes a crash mid-fill
recoverable rather than a mystery.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from core.config import PAPER, get_settings
from core.contracts import PortfolioState, Position, utcnow
from core.db import session_scope
from core.repository import _aware, trade_stats
from core.tables import EquityPoint, LedgerEntry, Order, PositionRow, Trade

log = logging.getLogger("execution.portfolio")


class PortfolioStore:
    """Reads and writes one mode's book."""

    def __init__(self, mode: str | None = None):
        self.mode = (mode or get_settings().execution.mode).upper()

    # ── cash ────────────────────────────────────────────────────────────────
    def cash(self) -> float:
        with session_scope() as s:
            return float(s.execute(select(func.coalesce(func.sum(LedgerEntry.amount), 0.0))
                                   .where(LedgerEntry.mode == self.mode)).scalar() or 0.0)

    def ensure_funded(self, starting_cash: float | None = None) -> float:
        """Seed a paper account once. Real accounts are funded on the exchange."""
        balance = self.cash()
        if balance > 0 or self.mode != PAPER:
            return balance
        amount = float(starting_cash if starting_cash is not None
                       else get_settings().execution.paper_starting_cash)
        self.add_ledger("deposit", amount, note="paper account opening balance")
        log.info("paper account funded with %.2f", amount)
        return amount

    def add_ledger(self, kind: str, amount: float, *, ref: str = "", note: str = "") -> float:
        with session_scope() as s:
            current = float(s.execute(select(func.coalesce(func.sum(LedgerEntry.amount), 0.0))
                                      .where(LedgerEntry.mode == self.mode)).scalar() or 0.0)
            balance = current + float(amount)
            s.add(LedgerEntry(mode=self.mode, ts=utcnow(), kind=kind, amount=float(amount),
                              balance_after=balance, ref=ref, note=note))
            return balance

    # ── position ────────────────────────────────────────────────────────────
    def position(self, symbol: str) -> Position | None:
        with session_scope() as s:
            row = s.execute(select(PositionRow).where(PositionRow.mode == self.mode,
                                                      PositionRow.symbol == symbol)).scalar_one_or_none()
            if row is None or row.quantity <= 1e-12:
                return None
            return Position(symbol=row.symbol, quantity=float(row.quantity),
                            avg_entry_price=float(row.avg_entry_price),
                            opened_at=_aware(row.opened_at), stop_price=row.stop_price,
                            target_price=row.target_price, bars_held=int(row.bars_held or 0))

    def position_row(self, symbol: str) -> dict | None:
        with session_scope() as s:
            row = s.execute(select(PositionRow).where(PositionRow.mode == self.mode,
                                                      PositionRow.symbol == symbol)).scalar_one_or_none()
            if row is None:
                return None
            return {c.name: getattr(row, c.name) for c in row.__table__.columns}

    def upsert_position(self, symbol: str, **fields: Any) -> None:
        with session_scope() as s:
            row = s.execute(select(PositionRow).where(PositionRow.mode == self.mode,
                                                      PositionRow.symbol == symbol)).scalar_one_or_none()
            if row is None:
                row = PositionRow(mode=self.mode, symbol=symbol)
                s.add(row)
            for k, v in fields.items():
                setattr(row, k, v)
            row.updated_at = utcnow()

    def clear_position(self, symbol: str) -> None:
        self.upsert_position(symbol, quantity=0.0, avg_entry_price=0.0, opened_at=None,
                             stop_price=None, target_price=None, bars_held=0,
                             entry_cycle_id=None, entry_decision_id=None,
                             entry_confidence=0.0, context={})

    def tick_bars_held(self, symbol: str) -> None:
        row = self.position_row(symbol)
        if row and float(row["quantity"] or 0) > 0:
            self.upsert_position(symbol, bars_held=int(row["bars_held"] or 0) + 1)

    # ── aggregate state ─────────────────────────────────────────────────────
    def state(self, symbol: str, price: float, *, kill_switch: bool = False) -> PortfolioState:
        cash = self.cash()
        pos = self.position(symbol)
        position_value = (pos.quantity * price) if pos else 0.0
        equity = cash + position_value
        stats = trade_stats(self.mode)

        with session_scope() as s:
            day_start = datetime.combine(utcnow().date(), datetime.min.time(),
                                         tzinfo=timezone.utc)
            trades_today = int(s.execute(
                select(func.count(Trade.id)).where(Trade.mode == self.mode,
                                                   Trade.entry_at >= day_start)).scalar() or 0)
            pnl_today = float(s.execute(
                select(func.coalesce(func.sum(Trade.pnl_quote), 0.0))
                .where(Trade.mode == self.mode, Trade.exit_at >= day_start)).scalar() or 0.0)
            realized = float(s.execute(
                select(func.coalesce(func.sum(Trade.pnl_quote), 0.0))
                .where(Trade.mode == self.mode)).scalar() or 0.0)
            peak = float(s.execute(select(func.coalesce(func.max(EquityPoint.equity), 0.0))
                                   .where(EquityPoint.mode == self.mode)).scalar() or 0.0)

        peak = max(peak, equity)
        drawdown = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
        return PortfolioState(
            mode=self.mode, cash=round(cash, 8), equity=round(equity, 8), position=pos,
            last_price=price, realized_pnl=round(realized, 8),
            unrealized_pnl=round(pos.quantity * (price - pos.avg_entry_price), 8) if pos else 0.0,
            trades_today=trades_today, realized_pnl_today=round(pnl_today, 8),
            peak_equity=round(peak, 8), max_drawdown_pct=round(drawdown, 4),
            open_risk_pct=self._open_risk_pct(pos, equity),
            win_rate=stats.get("win_rate", 0.0), total_trades=stats.get("trades", 0),
            kill_switch=kill_switch, updated_at=utcnow())

    @staticmethod
    def _open_risk_pct(pos: Position | None, equity: float) -> float:
        if not pos or not pos.stop_price or equity <= 0:
            return 0.0
        return round(max(0.0, (pos.avg_entry_price - pos.stop_price)) * pos.quantity
                     / equity * 100.0, 4)

    def record_equity(self, symbol: str, price: float, cycle_id: str | None = None) -> None:
        st = self.state(symbol, price)
        with session_scope() as s:
            s.add(EquityPoint(mode=self.mode, ts=utcnow(), equity=st.equity, cash=st.cash,
                              position_value=st.equity - st.cash, price=price,
                              drawdown_pct=st.max_drawdown_pct, cycle_id=cycle_id))

    # ── history ─────────────────────────────────────────────────────────────
    def equity_curve(self, limit: int = 500) -> list[dict]:
        with session_scope() as s:
            rows = s.execute(select(EquityPoint).where(EquityPoint.mode == self.mode)
                             .order_by(EquityPoint.ts.desc()).limit(limit)).scalars().all()
            return [{"ts": _aware(r.ts), "equity": r.equity, "cash": r.cash,
                     "position_value": r.position_value, "price": r.price,
                     "drawdown_pct": r.drawdown_pct} for r in reversed(rows)]

    def recent_orders(self, limit: int = 50) -> list[dict]:
        with session_scope() as s:
            rows = s.execute(select(Order).where(Order.mode == self.mode)
                             .order_by(Order.created_at.desc()).limit(limit)).scalars().all()
            return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in rows]

    def recent_trades(self, limit: int = 50) -> list[dict]:
        with session_scope() as s:
            rows = s.execute(select(Trade).where(Trade.mode == self.mode)
                             .order_by(Trade.exit_at.desc().nullslast()).limit(limit)).scalars().all()
            return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in rows]

    def stats(self) -> dict:
        base = trade_stats(self.mode)
        curve = self.equity_curve(limit=5000)
        if curve:
            equities = [p["equity"] for p in curve]
            peak, max_dd = equities[0], 0.0
            for e in equities:
                peak = max(peak, e)
                max_dd = max(max_dd, (peak - e) / peak * 100.0 if peak > 0 else 0.0)
            base.update({"equity": round(equities[-1], 2), "peak_equity": round(peak, 2),
                         "max_drawdown_pct": round(max_dd, 3),
                         "return_pct": round((equities[-1] / equities[0] - 1) * 100, 3)
                         if equities[0] > 0 else 0.0,
                         "points": len(equities)})
        base["mode"] = self.mode
        return base
