"""Turning four machine outputs into something a model can reason over.

The prompt is built in a fixed order — who you are, what you may not do, what the
engines said, what your own history says, what you are holding, what the theory
says — because the constraints have to be in front of the model *before* the
market data, not appended after it as an afterthought.

The output contract is strict JSON. The prose lives inside the JSON, in the
``rationale`` field, which is where the human voice belongs: everything else in
the object is a number or an enum that the execution layer can act on without
guessing.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from core.contracts import (AdminRestrictions, EngineSignal, PortfolioState,
                            RiskAssessment)

SYSTEM_PROMPT = """You are the senior trader on a small systematic desk. Three \
systems report to you:

- Engine 1, a multi-horizon technical and context engine (EMA/RSI/ATR structure \
across 15 minutes to 30 days, plus news);
- Engine 2, a deep learning quant engine (CNN-BiLSTM-attention forecaster feeding \
a PPO policy trained on realistic fills);
- Engine 3, a risk model trained on this desk's own trade history.

They do not trade. You do. Your answer is the order.

How you work:
- You are spot-only and long-only. BUY opens or adds, SELL closes what is held, \
HOLD does nothing. A bearish view when flat is HOLD (in cash), never SELL.
- HOLD is a real answer and usually the right one. Most bars are not opportunities.
- You reason like a person, not a template: what the evidence is, what it is not, \
what you are doing about it, and what would make you change your mind.
- You never contradict the operator's restrictions. They are limits, not advice.
- Your confidence is calibrated. If you say 0.8 you should be right four times in \
five. When the engines disagree or one is down, the honest number is low.
- You are explicit about the size, the stop and the target, in the units asked for.
- You never invent data. If something is missing or stale, you say so and account \
for it in your confidence.

You reply with a single JSON object and nothing else."""

OUTPUT_SCHEMA = """{
  "action": "BUY | SELL | HOLD",
  "confidence": 0.0,
  "size_pct": 0.0,
  "entry_price": 0.0,
  "stop_price": 0.0,
  "target_price": 0.0,
  "time_horizon": "e.g. 4-12 hours",
  "engine_agreement": "aligned | split | conflicted",
  "rationale": "3-6 sentences in plain English, first person, as a human trader would explain the decision to a colleague. Say what the evidence is, how much of it you trust, and why this size.",
  "key_risks": ["short phrases"],
  "change_my_mind": ["concrete, observable conditions that would flip or void this view"],
  "used_theories": ["names of knowledge-base sections you actually relied on"]
}"""


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


def _engine_block(sig: EngineSignal | None, title: str) -> str:
    if sig is None:
        return f"### {title}\nNot run this cycle."
    if not sig.ok:
        return (f"### {title}\nDOWN — {sig.error or 'no signal'}. "
                f"Treat this engine as absent; do not assume it agrees with the other.")
    age = f"{sig.age_seconds / 60:.0f} min old" if sig.age_seconds > 90 else "fresh"
    stale = "  ⚠ STALE, last known good reading" if sig.stale else ""
    lines = [f"### {title}",
             f"direction={sig.direction}  confidence={sig.confidence:.2f}  "
             f"hint={sig.action_hint}  horizon={sig.horizon}  ({age}){stale}"]
    if sig.levels:
        lines.append("levels: " + json.dumps({k: v for k, v in sig.levels.items()
                                              if v is not None}, default=str))
    if sig.features:
        compact = {k: v for k, v in list(sig.features.items())[:14] if v is not None}
        lines.append("features: " + json.dumps(compact, default=str))
    for reason in sig.reasons[:4]:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def _risk_block(risk: RiskAssessment | None) -> str:
    if risk is None:
        return "### Engine 3 — risk model\nNot available this cycle."
    trained = (f"trained on {risk.trained_on_samples} of this desk's own outcomes"
               if risk.trained_on_samples else
               "no trade history yet — these are the desk's default rules, not learned")
    lines = [
        "### Engine 3 — risk model (learned from this desk's own results)",
        f"win_probability={risk.win_probability:.2f}  expected_value={risk.expected_r:+.2f}R  "
        f"risk_score={risk.risk_score:.2f}  size_multiplier={risk.size_multiplier:.2f}",
        f"regime={risk.regime}  model={risk.model_kind} v{risk.model_version}  ({trained})",
    ]
    for note in risk.notes[:4]:
        lines.append(f"  - {note}")
    if risk.veto:
        lines.append("  VETO — the risk engine refuses a new entry here:")
        for reason in risk.veto_reasons[:4]:
            lines.append(f"    * {reason}")
    return "\n".join(lines)


def _portfolio_block(p: PortfolioState | None, mode: str) -> str:
    if p is None:
        return f"### Book ({mode})\nUnknown."
    pos = p.position
    if pos and pos.is_open:
        held = (f"LONG {pos.quantity:.6f} at {_fmt(pos.avg_entry_price)}  "
                f"unrealised {pos.unrealized_pct(p.last_price):+.2f}%  "
                f"stop={_fmt(pos.stop_price)}  target={_fmt(pos.target_price)}  "
                f"held {pos.bars_held} cycles")
    else:
        held = "FLAT (fully in cash)"
    return "\n".join([
        f"### Book ({mode} mode — {'simulated money' if mode == 'PAPER' else 'REAL FUNDS'})",
        f"equity={_fmt(p.equity)}  cash={_fmt(p.cash)}  last_price={_fmt(p.last_price)}",
        f"position: {held}",
        f"today: {p.trades_today} trades, realised {_fmt(p.realized_pnl_today)}  "
        f"| all time: {p.total_trades} trades, win rate {p.win_rate:.1f}%, "
        f"max drawdown {p.max_drawdown_pct:.2f}%",
        f"kill_switch={'ON — exits only' if p.kill_switch else 'off'}",
    ])


def build_user_prompt(*, symbol: str, mode: str, price: float,
                      signals: Sequence[EngineSignal], risk: RiskAssessment | None,
                      portfolio: PortfolioState | None,
                      restrictions: AdminRestrictions, knowledge: str,
                      max_size_quote: float, extra_context: dict | None = None) -> str:
    by_engine = {s.engine: s for s in signals}
    rules = "\n".join(f"- {line}" for line in restrictions.as_prompt_lines())
    extra = extra_context or {}

    parts = [
        f"# Decision required: {symbol} — {mode} mode",
        f"Spot price now: {_fmt(price)}. Cycle time: {extra.get('now', 'now')}.",
        "",
        "## Operator restrictions (hard limits — you may not exceed these)",
        rules,
        f"- The most you may deploy on this decision is {_fmt(max_size_quote)} "
        f"in quote currency. size_pct is a percentage of total equity.",
        "",
        "## What the engines say",
        _engine_block(by_engine.get("engine_1"), "Engine 1 — context / calibration"),
        "",
        _engine_block(by_engine.get("engine_2"), "Engine 2 — quantitative / ML"),
        "",
        _risk_block(risk),
        "",
        _portfolio_block(portfolio, mode),
    ]

    if extra.get("recent_decisions"):
        parts += ["", "## Your last few calls on this symbol",
                  *[f"- {d}" for d in extra["recent_decisions"][:5]]]
    if extra.get("notes"):
        parts += ["", "## Desk notes", *[f"- {n}" for n in extra["notes"][:6]]]

    parts += [
        "",
        "## Relevant theory from the desk's knowledge base",
        knowledge or "(knowledge base unavailable this cycle — rely on the engines "
                     "and say so in your rationale)",
        "",
        "## Your answer",
        "Return exactly this JSON object, no prose outside it:",
        OUTPUT_SCHEMA,
        "",
        "Rules for the numbers: confidence is 0-1. size_pct is a percentage of equity "
        "and must be 0 when action is HOLD or SELL. For a BUY, stop_price must be below "
        "entry_price and target_price above it. For a SELL, size_pct is 0 and the "
        "position is closed in full. If the risk engine vetoed a new entry, you may not "
        "answer BUY — explain why in the rationale instead.",
    ]
    return "\n".join(parts)


def situation_terms(*, signals: Iterable[EngineSignal], risk: RiskAssessment | None,
                    portfolio: PortfolioState | None) -> list[str]:
    """The query used to retrieve knowledge-base sections for this exact situation."""
    terms: list[str] = ["decision", "process", "spot"]
    sigs = list(signals)
    for s in sigs:
        if not s.ok:
            terms += ["degraded", "disagreement", "confidence"]
            continue
        terms += [s.direction.lower(), s.action_hint.lower()]
        rsi = s.features.get("rsi14")
        if isinstance(rsi, (int, float)):
            if rsi > 70:
                terms += ["overbought", "mean reversion", "extremes"]
            elif rsi < 30:
                terms += ["oversold", "mean reversion", "extremes"]
        if s.features.get("trend_up") is not None:
            terms.append("trend")
        atr = s.features.get("atr_pct")
        if isinstance(atr, (int, float)) and atr > 2:
            terms += ["volatility", "atr", "sizing"]
    directions = {s.direction for s in sigs if s.ok}
    if len(directions) > 1:
        terms += ["conflict", "disagreement", "confluence", "ensemble"]
    else:
        terms += ["confluence", "agreement"]
    if risk is not None:
        terms += [risk.regime, "risk", "sizing", "kelly", "expectancy"]
        if risk.veto:
            terms += ["ruin", "drawdown", "discipline"]
    if portfolio is not None:
        if portfolio.in_position:
            terms += ["exit", "invalidation", "stop"]
        else:
            terms += ["entry", "fomo"]
        if portfolio.max_drawdown_pct > 5:
            terms += ["drawdown", "risk of ruin", "psychology"]
        if portfolio.trades_today >= 3:
            terms += ["overtrading", "costs", "fees"]
    return terms
