"""The Agent's voice when there is no model to speak with.

A trading system whose decisions stop when an API key expires is not a trading
system. This module produces the same :class:`AgentDecision` the LLM would — same
fields, same constraints, same plain-English rationale — from the engines' numbers
alone. It is deliberately conservative: it will refuse far more often than the LLM
does, because a deterministic rule set has no way to notice that this particular
setup is the exception.

It is also the reference implementation of the decision policy. If the LLM's
answer disagrees with what this would do, that difference is visible in the logs
and is the thing worth reviewing.
"""
from __future__ import annotations

from typing import Sequence

from core.contracts import (BUY, DOWN, HOLD, NEUTRAL, SELL, UP, AdminRestrictions,
                            AgentDecision, EngineSignal, PortfolioState, RiskAssessment,
                            sane_levels)

# engine_2 is the trained specialist on this timeframe; engine_1 is the wider
# context. Neither is allowed to act alone at full strength.
ENGINE_WEIGHTS = {"engine_1": 0.45, "engine_2": 0.55}


def consensus(signals: Sequence[EngineSignal]) -> tuple[str, float, str]:
    """-> (direction, strength 0..1, agreement label)."""
    live = [s for s in signals if s.ok]
    if not live:
        return NEUTRAL, 0.0, "no_engines"

    total_w = sum(ENGINE_WEIGHTS.get(s.engine, 0.5) for s in live)
    score = sum(ENGINE_WEIGHTS.get(s.engine, 0.5) * s.signed_confidence()
                for s in live) / max(total_w, 1e-9)

    directions = {s.direction for s in live if s.direction != NEUTRAL}
    if len(live) == 1:
        agreement = "single_engine"
    elif len(directions) > 1:
        agreement = "conflicted"
    elif directions:
        agreement = "aligned"
    else:
        agreement = "split"

    direction = UP if score > 0.12 else DOWN if score < -0.12 else NEUTRAL
    strength = min(1.0, abs(score))
    if agreement == "conflicted":
        strength *= 0.5
    elif agreement == "single_engine":
        strength *= 0.7
    return direction, strength, agreement


def _levels(signals: Sequence[EngineSignal], price: float,
            restrictions: AdminRestrictions) -> tuple[float, float]:
    """Prefer a real engine level; fall back to a percentage of price."""
    stop = target = 0.0
    for s in signals:
        if not s.ok:
            continue
        lv = s.levels or {}
        stop = stop or float(lv.get("stop_loss") or lv.get("reference_stop") or 0.0)
        target = target or float(lv.get("take_profit_1") or lv.get("reference_target") or 0.0)
    stop, target = sane_levels(price, stop, target)
    max_stop = price * (1 - restrictions.max_stop_distance_pct / 100.0)
    return max(stop, max_stop), target


def decide(*, signals: Sequence[EngineSignal], risk: RiskAssessment | None,
           portfolio: PortfolioState, restrictions: AdminRestrictions,
           price: float, reason_prefix: str = "") -> AgentDecision:
    direction, strength, agreement = consensus(signals)
    stop, target = _levels(signals, price, restrictions)
    in_position = portfolio.in_position
    degraded = any(not s.ok for s in signals)

    win_p = risk.win_probability if risk else 0.5
    size_mult = risk.size_multiplier if risk else 0.5
    vetoed = bool(risk and risk.veto)

    # ── the decision ────────────────────────────────────────────────────────
    if in_position and (direction == DOWN or (risk and risk.risk_score > 0.7)):
        action = SELL
        confidence = max(strength, 0.5 if vetoed else strength)
    elif (not in_position and direction == UP and not vetoed
          and restrictions.allow_new_entries and not restrictions.kill_switch):
        action = BUY
        # Confidence in the *decision*: the engines' combined strength and the risk
        # model's own probability, weighted. Neither gets to speak alone.
        confidence = 0.55 * strength + 0.45 * win_p
    else:
        action = HOLD
        confidence = max(0.3, 1.0 - strength) if direction == NEUTRAL else strength * 0.6

    if action == BUY and confidence < restrictions.min_confidence:
        action, confidence = HOLD, confidence

    # ── size, from risk, not from enthusiasm ────────────────────────────────
    size_pct = 0.0
    if action == BUY:
        stop_distance_pct = max(0.05, (price - stop) / price * 100.0)
        risk_sized = restrictions.max_capital_at_risk_pct / stop_distance_pct * 100.0
        size_pct = min(restrictions.max_position_pct, risk_sized) * size_mult
        size_pct = max(0.0, round(size_pct, 3))

    decision = AgentDecision(
        action=action, confidence=round(min(confidence, 0.85), 3), size_pct=size_pct,
        entry_price=price if action == BUY else None,
        stop_price=round(stop, 2) if action == BUY else (
            portfolio.position.stop_price if in_position and portfolio.position else None),
        target_price=round(target, 2) if action == BUY else None,
        time_horizon="4-12 hours", engine_agreement=agreement,
        rationale=_rationale(action, direction, strength, agreement, signals, risk,
                             portfolio, restrictions, price, stop, target, size_pct,
                             reason_prefix),
        key_risks=_risks(signals, risk, portfolio, degraded),
        change_my_mind=_change_my_mind(action, direction, signals, price, stop, target),
        used_theories=["Model Confluence and Disagreement", "Risk of Ruin and Position Sizing",
                       "Decision Discipline for This System"],
        source="fallback", degraded=degraded, admin_version=restrictions.version)
    return decision


# ── prose ───────────────────────────────────────────────────────────────────
def _describe_engines(signals: Sequence[EngineSignal]) -> str:
    bits = []
    for s in signals:
        label = {"engine_1": "the context engine", "engine_2": "the quant engine",
                 "engine_3": "the risk engine"}.get(s.engine, s.engine)
        if not s.ok:
            bits.append(f"{label} is down")
        elif s.stale:
            bits.append(f"{label} is running on a {s.age_seconds / 60:.0f}-minute-old reading "
                        f"({s.direction.lower()}, {s.confidence:.0%})")
        else:
            bits.append(f"{label} reads {s.direction.lower()} at {s.confidence:.0%}")
    return "; ".join(bits) if bits else "no engine reported"


def _rationale(action, direction, strength, agreement, signals, risk, portfolio,
               restrictions, price, stop, target, size_pct, prefix) -> str:
    lines = []
    if prefix:
        lines.append(prefix)
    lines.append(f"No language model was reachable this cycle, so this is the desk's "
                 f"deterministic policy speaking. {_describe_engines(signals).capitalize()}.")

    if agreement == "conflicted":
        lines.append("The two engines are pointing in opposite directions, which historically "
                     "is the worst configuration to size into, so I am treating the combined "
                     "signal as roughly half of what either claims on its own.")
    elif agreement == "single_engine":
        lines.append("Only one engine is reporting, so I am running on a single opinion and "
                     "have discounted it accordingly.")
    elif agreement == "aligned" and direction != NEUTRAL:
        lines.append(f"Both engines agree on {direction.lower()}, which is the strongest "
                     f"evidence this desk gets, and the combined strength is {strength:.0%}.")

    if risk is not None:
        lines.append(f"The risk model puts the win probability at {risk.win_probability:.0%} "
                     f"with an expected value of {risk.expected_r:+.2f}R in a "
                     f"{risk.regime.replace('_', ' ')} regime.")
        if risk.veto:
            lines.append("It is vetoing a new entry: " + risk.veto_reasons[0])

    if action == BUY:
        lines.append(f"So I am buying {size_pct:.2f}% of equity at about {price:,.2f}, with the "
                     f"stop at {stop:,.2f} ({(price - stop) / price * 100:.2f}% away) and the "
                     f"first target at {target:,.2f}. The size comes from the stop distance and "
                     f"the {restrictions.max_capital_at_risk_pct:.2f}% risk budget, not from how "
                     f"the setup feels.")
    elif action == SELL:
        pos = portfolio.position
        held = f" from {pos.avg_entry_price:,.2f}" if pos and pos.avg_entry_price else ""
        lines.append(f"I am closing the position{held} at about {price:,.2f}. The reason to hold "
                     f"it is gone, and in a spot book the way to express a bearish view is to "
                     f"be in cash.")
    else:
        if portfolio.in_position:
            lines.append("I am holding what we have: nothing in the evidence justifies adding, "
                         "and nothing justifies giving up the position either.")
        else:
            lines.append("I am staying in cash. There is no configuration here worth paying the "
                         "round-trip cost for, and cash is a position.")
    return " ".join(lines)


def _risks(signals, risk, portfolio, degraded) -> list[str]:
    out = []
    if degraded:
        out.append("Running degraded — at least one engine did not report this cycle.")
    if risk is not None and risk.regime == "high_volatility":
        out.append("High-volatility regime: stops are hit more often than the backtest implies.")
    if portfolio.max_drawdown_pct > 5:
        out.append(f"Account is {portfolio.max_drawdown_pct:.1f}% below its equity peak.")
    if portfolio.trades_today >= 3:
        out.append(f"{portfolio.trades_today} trades already today; fees compound quickly.")
    for s in signals:
        if s.ok and s.stale:
            out.append(f"{s.engine} data is {s.age_seconds / 60:.0f} minutes stale.")
    return out[:5] or ["Ordinary market risk: the next move can invalidate any of this."]


def _change_my_mind(action, direction, signals, price, stop, target) -> list[str]:
    out = []
    if action == BUY:
        out.append(f"A close below {stop:,.2f} — that is the level that says the entry was wrong.")
        out.append(f"A tag of {target:,.2f} without follow-through would have me taking profit "
                   f"rather than pressing.")
        out.append("Either engine flipping to a down reading would cut the size immediately.")
    elif action == SELL:
        out.append("Both engines turning up again, with the daily structure intact, would put "
                   "me back on the long side.")
        out.append(f"A reclaim of {price * 1.01:,.2f} on volume would say the exit was early.")
    else:
        out.append("Both engines agreeing on a direction with above-threshold confidence would "
                   "get me off the fence.")
        out.append("A volatility contraction with a clean higher low would make an entry worth "
                   "the round-trip cost.")
    out.append("A restored engine or a fresh risk-model version can change this within one cycle.")
    return out[:4]
