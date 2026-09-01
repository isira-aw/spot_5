"""The last check before money moves.

The Agent already knows the restrictions and is told to respect them; the
constitution in :mod:`llm_agent.agent` already rewrites answers that stray. This
module checks a third time, immediately before the order, because the two earlier
layers are software that can be changed by editing a prompt and this one is not.

The kill switch lives here. It blocks new entries and nothing else: an operator
hitting it in a panic wants the system to stop *buying*, not to be trapped holding
a position it can no longer sell.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

from core.config import REAL, get_settings
from core.contracts import (BUY, HOLD, SELL, AdminRestrictions, AgentDecision,
                            PortfolioState, utcnow)
from core.repository import get_state, record_event, set_state

log = logging.getLogger("execution.risk_guard")

KILL_SWITCH_KEY = "kill_switch"


# ── kill switch ─────────────────────────────────────────────────────────────
def is_killed() -> bool:
    try:
        return bool((get_state(KILL_SWITCH_KEY) or {}).get("enabled"))
    except Exception as exc:
        # Fail closed: if we cannot read the switch we assume the operator meant
        # to stop, because the cost of a wrong "on" is a missed trade and the cost
        # of a wrong "off" is an unwanted one.
        log.error("kill switch unreadable (%s) — treating as ON", exc)
        return True


def set_kill_switch(enabled: bool, *, by: str = "admin", reason: str = "") -> dict:
    payload = {"enabled": bool(enabled), "reason": reason, "by": by,
               "at": utcnow().isoformat()}
    set_state(KILL_SWITCH_KEY, payload, updated_by=by)
    record_event(f"kill switch {'ENGAGED' if enabled else 'released'} by {by}"
                 + (f": {reason}" if reason else ""),
                 level="warning" if enabled else "info",
                 category="risk", payload=payload)
    log.warning("kill switch %s by %s%s", "ENGAGED" if enabled else "released", by,
                f" ({reason})" if reason else "")
    return payload


# ── pre-trade checks ────────────────────────────────────────────────────────
class GuardResult:
    def __init__(self, allowed: bool, reasons: Iterable[str] = (), quote: float = 0.0,
                 quantity: float = 0.0):
        self.allowed = allowed
        self.reasons = list(reasons)
        self.quote = float(quote)
        self.quantity = float(quantity)

    def __bool__(self) -> bool:
        return self.allowed

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "reasons": self.reasons,
                "quote": round(self.quote, 8), "quantity": round(self.quantity, 10)}


def check(decision: AgentDecision, *, portfolio: PortfolioState,
          restrictions: AdminRestrictions, price: float, symbol: str,
          now: datetime | None = None) -> GuardResult:
    now = now or utcnow()
    caps = get_settings().caps
    reasons: list[str] = []

    if decision.action == HOLD:
        return GuardResult(False, ["decision is HOLD"], 0.0)

    if price <= 0:
        return GuardResult(False, ["no valid price available"], 0.0)

    # ── exits: permitted in almost every state, on purpose ──────────────────
    if decision.action == SELL:
        pos = portfolio.position
        if not pos or not pos.is_open:
            return GuardResult(False, ["nothing held to sell"], 0.0)
        return GuardResult(True, ["exit permitted"], quote=pos.quantity * price,
                           quantity=pos.quantity)

    # ── entries ─────────────────────────────────────────────────────────────
    if is_killed() or restrictions.kill_switch:
        reasons.append("kill switch is engaged: no new entries")
    if not restrictions.allow_new_entries:
        reasons.append("new entries are administratively paused")
    if restrictions.allowed_symbols and symbol not in restrictions.allowed_symbols:
        reasons.append(f"{symbol} is not on the allowed list")
    if symbol in (restrictions.blocked_symbols or []):
        reasons.append(f"{symbol} is blocked")
    if BUY not in (restrictions.allowed_actions or [BUY, SELL, HOLD]):
        reasons.append("BUY is not currently an allowed action")
    if portfolio.in_position and restrictions.max_open_positions <= 1:
        reasons.append("maximum open positions reached")
    if portfolio.trades_today >= restrictions.max_trades_per_day:
        reasons.append(f"daily trade cap reached "
                       f"({portfolio.trades_today}/{restrictions.max_trades_per_day})")
    if portfolio.equity > 0:
        loss_pct = -portfolio.realized_pnl_today / portfolio.equity * 100.0
        if loss_pct >= restrictions.max_daily_loss_pct:
            reasons.append(f"daily loss limit hit ({loss_pct:.2f}% >= "
                           f"{restrictions.max_daily_loss_pct:.2f}%)")
    if decision.confidence < restrictions.min_confidence:
        reasons.append(f"confidence {decision.confidence:.2f} below the "
                       f"{restrictions.min_confidence:.2f} floor")
    if restrictions.trading_hours_utc and now.hour not in restrictions.trading_hours_utc:
        reasons.append(f"{now:%H:%M} UTC is outside permitted trading hours")
    if now.strftime("%Y-%m-%d") in (restrictions.blackout_dates or []):
        reasons.append(f"{now:%Y-%m-%d} is a blackout date")
    if restrictions.require_stop_loss and not decision.stop_price:
        reasons.append("no stop loss on a BUY")

    # size: the guard recomputes it rather than trusting the number it was handed
    ceiling_pct = min(restrictions.max_position_pct, caps.max_position_pct)
    quote = min(decision.size_quote or (portfolio.equity * decision.size_pct / 100.0),
                portfolio.equity * ceiling_pct / 100.0,
                portfolio.cash)
    if decision.stop_price and decision.entry_price and decision.entry_price > 0:
        stop_distance = (decision.entry_price - decision.stop_price) / decision.entry_price
        if stop_distance > 0:
            risk_budget = portfolio.equity * restrictions.max_capital_at_risk_pct / 100.0
            quote = min(quote, risk_budget / stop_distance)
    quote = max(0.0, round(quote, 8))

    if quote < restrictions.min_order_quote:
        reasons.append(f"order size {quote:,.2f} is below the minimum "
                       f"{restrictions.min_order_quote:,.2f}")

    if reasons:
        return GuardResult(False, reasons, quote)
    return GuardResult(True, ["entry permitted"], quote=quote, quantity=quote / price)


# ── between-cycle protection ────────────────────────────────────────────────
def protective_exit(portfolio: PortfolioState, price: float) -> str | None:
    """Stops and targets are checked every cycle, before anyone is asked anything.

    The Agent is not consulted about a stop. A stop is the price at which the
    thesis was wrong, and waiting for a language model to agree with arithmetic is
    not risk management.
    """
    pos = portfolio.position
    if not pos or not pos.is_open or price <= 0:
        return None
    if pos.stop_price and price <= pos.stop_price:
        return "stop_loss"
    if pos.target_price and price >= pos.target_price:
        return "take_profit"
    return None


def live_mode_preflight() -> list[str]:
    """Everything that must be true before REAL mode is allowed to start."""
    return live_mode_preflight_for(get_settings().execution.mode)


def live_mode_preflight_for(mode: str) -> list[str]:
    """The same checks, for a mode we are considering switching *into*."""
    s = get_settings()
    problems: list[str] = []
    if mode.upper() != REAL:
        return problems
    if not s.execution.live_confirmed:
        problems.append("LIVE_TRADING_CONFIRMED is not set")
    if not (s.execution.exchange_api_key and s.execution.exchange_api_secret):
        problems.append("exchange API credentials are missing")
    if s.caps.max_capital_at_risk_pct > 5:
        problems.append(f"MAX_CAPITAL_AT_RISK_PCT is {s.caps.max_capital_at_risk_pct}%, "
                        f"which is above the 5% sanity ceiling for real funds")
    if not s.admin_token:
        problems.append("ADMIN_TOKEN is not set: the admin API would be unauthenticated")
    return problems
