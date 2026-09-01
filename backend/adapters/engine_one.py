"""Engine 1 — the context / calibration brain.

``engine_1/btc_multi_horizon.py`` already produces exactly what the Agent needs:
eleven horizons from 15 minutes to 30 days, each with an EMA/RSI/ATR/structure
score, a spot label and an ATR-derived plan. This adapter imports that module
(rather than shelling out to its CLI) so the results arrive as objects, and
collapses the eleven horizons into one normalized :class:`EngineSignal`.

How the collapse works, and why: the cross-horizon *consensus* gives the
direction, and confidence is the average confidence of the horizons that agree,
scaled by how much of the weighted signal they represent. Eleven horizons all
pointing up is a different animal from six up and five down at the same average
confidence, and the Agent is told which one it is looking at.
"""
from __future__ import annotations

import logging
import os
import sys

from core.config import BASE_DIR, get_settings
from core.contracts import (DOWN, NEUTRAL, SPOT_LABEL_TO_DIRECTION, UP,
                            EngineSignal, utcnow)

from .base import EngineAdapter

log = logging.getLogger("adapters.engine_1")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Which horizon supplies the concrete entry/stop/target the Agent will consider.
PRIMARY_HORIZON = os.environ.get("ENGINE_1_PRIMARY_HORIZON", "4h")
USE_NEWS = os.environ.get("ENGINE_1_USE_NEWS", "1").lower() in {"1", "true", "yes"}
USE_INNER_LLM = os.environ.get("ENGINE_1_USE_LLM", "0").lower() in {"1", "true", "yes"}
OFFLINE = os.environ.get("ENGINE_1_OFFLINE", "0").lower() in {"1", "true", "yes"}


def _load_module():
    import importlib
    return importlib.import_module("engine_1.btc_multi_horizon")


class EngineOneAdapter(EngineAdapter):
    name = "engine_1"

    def __init__(self, enabled: bool | None = None):
        s = get_settings()
        super().__init__(s.engines.engine_1_enabled if enabled is None else enabled)
        self.timeout_s = s.engines.engine_1_timeout_s
        self.cache_s = s.engines.engine_1_cache_s
        self.pair = s.engines.engine_1_pair

    def _compute(self, symbol: str, context: dict) -> EngineSignal:
        m = _load_module()
        offline = bool(context.get("offline", OFFLINE))
        native = m.load_native_series(self.pair, offline)

        results = []
        for h in m.HORIZONS:
            key = h.source if h.source in native else next(iter(native))
            resampled = m.resample_ohlc(native[key], h.rule, m.NATIVE_INTERVALS[key])
            if len(resampled) < 20:
                continue
            results.append(m.analyse_horizon(h, resampled))
        if not results:
            raise RuntimeError("no horizon had enough data")

        llm_meta = {"source": "baseline", "applied": 0,
                    "market_summary": None, "key_risk": None}
        news = {"crypto": [], "macro": []}
        if USE_NEWS and not offline:
            try:
                news = {"crypto": m.fetch_news("crypto"), "macro": m.fetch_news("macro")}
            except Exception as exc:                       # news is strictly optional
                log.info("engine_1 news skipped: %s", exc)
        if USE_INNER_LLM and not offline:
            try:
                llm_meta = m.run_llm(results, news["crypto"], news["macro"])
            except Exception as exc:
                log.info("engine_1 inner LLM skipped: %s", exc)

        consensus = m.consensus(results)
        finest = min(native, key=lambda k: m.NATIVE_INTERVALS[k])
        price = round(float(native[finest]["close"].iloc[-1]), 2)
        return self._normalize(symbol, results, consensus, llm_meta, price, news)

    # ── eleven horizons -> one opinion ──────────────────────────────────────
    def _normalize(self, symbol, results, consensus, llm_meta, price, news) -> EngineSignal:
        by_name = {r["horizon"]: r for r in results}
        primary = by_name.get(PRIMARY_HORIZON) or results[len(results) // 2]

        direction = consensus["bias"]
        agreeing = [r for r in results if r["final"]["bias"] == direction]
        avg_conf = (sum(r["final"]["confidence"] for r in agreeing) / len(agreeing) / 100.0
                    if agreeing else 0.0)
        confidence = avg_conf * (consensus["agreement_pct"] / 100.0)
        if direction == NEUTRAL:
            confidence = min(confidence, 0.4)

        plan = primary["plan"]
        f = primary["features"]
        # engine_1 only fills a plan when its label is a buy. The Agent still needs
        # structural levels to reason about when the answer is HOLD or SELL, so an
        # ATR-derived reference is always supplied alongside the (possibly empty) plan.
        atr = float(plan.get("atr_used") or f.get("atr") or max(price * 0.005, 1e-9))
        k = 1.5 if primary["minutes"] <= 240 else (2.0 if primary["minutes"] <= 1440 else 2.5)
        levels = {
            "entry_zone": plan.get("entry_zone"),
            "stop_loss": plan.get("stop_loss"),
            "take_profit_1": plan.get("take_profit_1"),
            "take_profit_2": plan.get("take_profit_2"),
            "risk_reward": plan.get("risk_reward"),
            "invalidation": plan.get("invalidation"),
            "atr": round(atr, 2),
            "reference_stop": round(price - k * atr, 2),
            "reference_target": round(price + 1.5 * k * atr, 2),
        }
        features = {
            "price": price,
            "rsi14": f.get("rsi14"),
            "atr_pct": f.get("atr_pct"),
            "ema_fast": f.get("ema_fast"),
            "ema_slow": f.get("ema_slow"),
            "trend_up": bool(f.get("ema_fast", 0) > f.get("ema_slow", 0)),
            "range_pos_pct": f.get("range_pos_pct"),
            "vol_ratio": f.get("vol_ratio"),
            "ret_3": f.get("ret_3"),
            "ret_10": f.get("ret_10"),
            "agreement_pct": consensus["agreement_pct"],
            "horizons_up": sum(1 for r in results if r["final"]["bias"] == UP),
            "horizons_down": sum(1 for r in results if r["final"]["bias"] == DOWN),
            "horizons_neutral": sum(1 for r in results if r["final"]["bias"] == NEUTRAL),
            "primary_horizon": primary["horizon"],
            "primary_label": primary["final"]["label"],
        }

        reasons = [
            f"Cross-horizon consensus is {direction} at {consensus['agreement_pct']:.0f}% "
            f"of the weighted signal ({features['horizons_up']} up / "
            f"{features['horizons_down']} down / {features['horizons_neutral']} neutral).",
            f"On the {primary['horizon']} the label is {primary['final']['label']} "
            f"({primary['final']['confidence']}% confidence).",
        ] + list(primary.get("reasons", []))[:2]
        if llm_meta.get("market_summary"):
            reasons.append(f"Engine-1 narrative: {llm_meta['market_summary']}")

        action_hint = primary["final"]["label"]
        return EngineSignal(
            engine=self.name, ok=True, symbol=symbol, generated_at=utcnow(),
            direction=direction, action_hint=action_hint, confidence=confidence,
            horizon=primary["horizon"], levels=levels, features=features,
            reasons=reasons, source="live",
            raw={"consensus": consensus, "llm": llm_meta, "price": price,
                 "news_counts": {k: len(v) for k, v in news.items()},
                 "horizons": [{"horizon": r["horizon"], "bias": r["final"]["bias"],
                               "confidence": r["final"]["confidence"],
                               "label": r["final"]["label"],
                               "score": r["baseline"]["score"],
                               "reasons": r.get("reasons", [])[:2],
                               "plan": r["plan"]} for r in results]})


def label_direction(label: str) -> str:
    return SPOT_LABEL_TO_DIRECTION.get(str(label).upper(), NEUTRAL)
