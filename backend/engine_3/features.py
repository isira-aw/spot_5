"""ONE definition of engine_3's feature vector, used by training and by serving.

The single most common way a system like this quietly breaks is that the training
job and the live path compute features slightly differently, so the model is
served inputs it never saw. So there is exactly one builder here, it is called
from both sides, and the feature *order* is pinned by ``FEATURE_NAMES``.

Every model version stores the feature list it was trained on. When this file
grows a new feature, old models keep scoring correctly because
:func:`vectorize` reads each model's own name list and fills anything it does not
recognise with a neutral default.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from core.contracts import DOWN, NEUTRAL, UP

# Order matters. Append new names at the end; never reorder or remove.
FEATURE_NAMES: tuple[str, ...] = (
    # engine_1 — context / calibration
    "e1_ok", "e1_signed_conf", "e1_confidence", "e1_agreement_pct",
    "e1_rsi14", "e1_atr_pct", "e1_trend_up", "e1_range_pos_pct", "e1_vol_ratio",
    "e1_ret_3", "e1_ret_10", "e1_horizons_up", "e1_horizons_down", "e1_horizons_neutral",
    # engine_2 — quantitative / ML
    "e2_ok", "e2_signed_conf", "e2_confidence", "e2_decisiveness",
    "e2_forecast_edge", "e2_p_up_mean", "e2_volatility",
    # cross-engine
    "engines_agree", "engine_conflict", "engine_conf_gap", "both_engines_ok",
    # portfolio / account state
    "drawdown_pct", "trades_today", "pnl_today_pct", "in_position",
    "equity_to_peak", "recent_win_rate", "recent_trade_count",
    # market timing and regime
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "range_pos_24h", "realized_vol_24h", "trend_slope_24h",
    # the setup being proposed
    "stop_distance_pct", "target_distance_pct", "risk_reward", "intent_confidence",
)

NEUTRAL_DEFAULTS: dict[str, float] = {
    "e1_ok": 0.0, "e2_ok": 0.0, "both_engines_ok": 0.0,
    "e1_agreement_pct": 50.0, "e1_rsi14": 50.0, "e1_atr_pct": 1.0,
    "e1_range_pos_pct": 50.0, "e1_vol_ratio": 1.0,
    "e2_p_up_mean": 0.5, "equity_to_peak": 1.0, "recent_win_rate": 0.5,
    "range_pos_24h": 50.0, "risk_reward": 1.5, "intent_confidence": 0.5,
}

_DIR_SIGN = {UP: 1.0, DOWN: -1.0, NEUTRAL: 0.0}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _signal_map(signals: Iterable[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in signals or []:
        name = s.get("engine") if isinstance(s, Mapping) else getattr(s, "engine", "")
        if name:
            out[name] = s
    return out


def _get(sig: Any, key: str, default: Any = None) -> Any:
    if sig is None:
        return default
    if isinstance(sig, Mapping):
        return sig.get(key, default)
    return getattr(sig, key, default)


def build_features(*, signals: Iterable[Any], portfolio: Any = None,
                   candles: Sequence[Sequence[float]] | None = None,
                   intent: Mapping[str, Any] | None = None,
                   now: datetime | None = None,
                   history: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Build the full feature dict from one cycle's raw material.

    ``intent`` is the setup under consideration (stop, target, confidence). At
    training time it is replayed from the trade that was actually taken, so the
    model learns "setups shaped like this one won or lost", not merely "the market
    looked like this".
    """
    now = now or datetime.now(timezone.utc)
    by_engine = _signal_map(signals)
    e1, e2 = by_engine.get("engine_1"), by_engine.get("engine_2")
    f: dict[str, float] = {}

    # ── engine_1 ────────────────────────────────────────────────────────────
    e1_ok = bool(_get(e1, "ok", False))
    e1_dir = str(_get(e1, "direction", NEUTRAL) or NEUTRAL).upper()
    e1_conf = _f(_get(e1, "confidence", 0.0))
    e1_feat = _get(e1, "features", {}) or {}
    f["e1_ok"] = 1.0 if e1_ok else 0.0
    f["e1_signed_conf"] = _DIR_SIGN.get(e1_dir, 0.0) * e1_conf if e1_ok else 0.0
    f["e1_confidence"] = e1_conf if e1_ok else 0.0
    f["e1_agreement_pct"] = _f(e1_feat.get("agreement_pct"), 50.0)
    f["e1_rsi14"] = _f(e1_feat.get("rsi14"), 50.0)
    f["e1_atr_pct"] = _f(e1_feat.get("atr_pct"), 1.0)
    f["e1_trend_up"] = 1.0 if e1_feat.get("trend_up") else 0.0
    f["e1_range_pos_pct"] = _f(e1_feat.get("range_pos_pct"), 50.0)
    f["e1_vol_ratio"] = _f(e1_feat.get("vol_ratio"), 1.0)
    f["e1_ret_3"] = _f(e1_feat.get("ret_3"))
    f["e1_ret_10"] = _f(e1_feat.get("ret_10"))
    f["e1_horizons_up"] = _f(e1_feat.get("horizons_up"))
    f["e1_horizons_down"] = _f(e1_feat.get("horizons_down"))
    f["e1_horizons_neutral"] = _f(e1_feat.get("horizons_neutral"))

    # ── engine_2 ────────────────────────────────────────────────────────────
    e2_ok = bool(_get(e2, "ok", False))
    e2_dir = str(_get(e2, "direction", NEUTRAL) or NEUTRAL).upper()
    e2_conf = _f(_get(e2, "confidence", 0.0))
    e2_feat = _get(e2, "features", {}) or {}
    p_up = [_f(p, 0.5) for p in (e2_feat.get("p_up") or [])]
    f["e2_ok"] = 1.0 if e2_ok else 0.0
    f["e2_signed_conf"] = _DIR_SIGN.get(e2_dir, 0.0) * e2_conf if e2_ok else 0.0
    f["e2_confidence"] = e2_conf if e2_ok else 0.0
    f["e2_decisiveness"] = _f(e2_feat.get("decisiveness"))
    f["e2_forecast_edge"] = _f(e2_feat.get("forecast_edge"))
    f["e2_p_up_mean"] = sum(p_up) / len(p_up) if p_up else 0.5
    f["e2_volatility"] = _f(e2_feat.get("volatility"))

    # ── cross-engine ────────────────────────────────────────────────────────
    s1, s2 = f["e1_signed_conf"], f["e2_signed_conf"]
    f["engines_agree"] = 1.0 if (s1 > 0 and s2 > 0) or (s1 < 0 and s2 < 0) else 0.0
    f["engine_conflict"] = 1.0 if (s1 > 0 > s2) or (s2 > 0 > s1) else 0.0
    f["engine_conf_gap"] = abs(s1 - s2)
    f["both_engines_ok"] = 1.0 if (e1_ok and e2_ok) else 0.0

    # ── portfolio ───────────────────────────────────────────────────────────
    p = portfolio if isinstance(portfolio, Mapping) else (
        portfolio.to_dict() if hasattr(portfolio, "to_dict") else {})
    equity = _f(p.get("equity"))
    peak = _f(p.get("peak_equity"), equity)
    f["drawdown_pct"] = _f(p.get("max_drawdown_pct"))
    f["trades_today"] = _f(p.get("trades_today"))
    f["pnl_today_pct"] = (_f(p.get("realized_pnl_today")) / equity * 100.0) if equity > 0 else 0.0
    f["in_position"] = 1.0 if p.get("in_position") or _f(
        (p.get("position") or {}).get("quantity") if isinstance(p.get("position"), Mapping) else 0) > 0 else 0.0
    f["equity_to_peak"] = (equity / peak) if peak > 0 else 1.0
    h = history or {}
    f["recent_win_rate"] = _f(h.get("win_rate"), 50.0) / 100.0
    f["recent_trade_count"] = _f(h.get("trades"))

    # ── timing and regime ───────────────────────────────────────────────────
    hour = now.hour + now.minute / 60.0
    f["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    f["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    f["dow_sin"] = math.sin(2 * math.pi * now.weekday() / 7.0)
    f["dow_cos"] = math.cos(2 * math.pi * now.weekday() / 7.0)
    f.update(regime_features(candles))

    # ── the proposed setup ──────────────────────────────────────────────────
    it = dict(intent or {})
    price = _f(it.get("price")) or _f(e1_feat.get("price")) or _f(e2_feat.get("close"))
    stop, target = _f(it.get("stop_price")), _f(it.get("target_price"))
    f["stop_distance_pct"] = abs(price - stop) / price * 100.0 if price > 0 and stop > 0 else 0.0
    f["target_distance_pct"] = abs(target - price) / price * 100.0 if price > 0 and target > 0 else 0.0
    risk = abs(price - stop)
    f["risk_reward"] = (abs(target - price) / risk) if risk > 1e-9 and target > 0 else 1.5
    f["intent_confidence"] = _f(it.get("confidence"), 0.5)

    return {k: round(_f(f.get(k, NEUTRAL_DEFAULTS.get(k, 0.0))), 6) for k in FEATURE_NAMES}


def regime_features(candles: Sequence[Sequence[float]] | None) -> dict[str, float]:
    """Where price sits in its recent range, how violent the tape is, which way it leans."""
    out = {"range_pos_24h": 50.0, "realized_vol_24h": 0.0, "trend_slope_24h": 0.0}
    rows = [c for c in (candles or []) if c and len(c) >= 5][-24:]
    if len(rows) < 3:
        return out
    highs = [_f(c[2]) for c in rows]
    lows = [_f(c[3]) for c in rows]
    closes = [_f(c[4]) for c in rows]
    hi, lo, last = max(highs), min(lows), closes[-1]
    if hi > lo:
        out["range_pos_24h"] = max(0.0, min(100.0, (last - lo) / (hi - lo) * 100.0))
    rets = [(closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if rets:
        mean = sum(rets) / len(rets)
        out["realized_vol_24h"] = round(
            math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets)) * 100.0, 6)
    n = len(closes)
    xbar, ybar = (n - 1) / 2.0, sum(closes) / n
    denom = sum((i - xbar) ** 2 for i in range(n))
    if denom > 0 and ybar > 0:
        slope = sum((i - xbar) * (closes[i] - ybar) for i in range(n)) / denom
        out["trend_slope_24h"] = round(slope / ybar * 100.0, 6)
    return out


def vectorize(features: Mapping[str, Any], names: Sequence[str] | None = None) -> list[float]:
    """Feature dict -> ordered vector, using the *model's own* name list."""
    names = list(names or FEATURE_NAMES)
    return [_f(features.get(n, NEUTRAL_DEFAULTS.get(n, 0.0)),
              NEUTRAL_DEFAULTS.get(n, 0.0)) for n in names]
