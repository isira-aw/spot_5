"""Engine 2 — the quantitative / ML brain.

``engine_2`` is the CNN-BiLSTM-MHA forecaster plus the PPO policy trained against
it. It can be consumed two ways, and this adapter supports both because they fail
differently:

``inline``
    Import ``engine_2.inference.Runner`` and score a bar in-process. Needs
    TensorFlow and a promoted model bundle. Highest fidelity: the adapter feeds
    the *real* portfolio position into the state vector, so the policy sees the
    same position the broker is actually holding rather than its own private
    guess.

``file``
    Read the last line of the JSONL the engine's own live loop writes. Works when
    the model runs as a separate service (its own box, its own GPU, its own
    schedule) and keeps this process free of a TensorFlow dependency.

``auto`` (the default) tries inline, falls back to file, and finally reports the
engine as down — which the Agent then says out loud instead of pretending it has
two opinions when it has one.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

from core.config import BASE_DIR, get_settings
from core.contracts import BUY, DOWN, HOLD, NEUTRAL, SELL, UP, EngineSignal, utcnow

from .base import EngineAdapter

log = logging.getLogger("adapters.engine_2")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SOURCE = os.environ.get("ENGINE_2_SOURCE", "auto").strip().lower()   # auto|inline|file
DECISION_LOG = os.environ.get(
    "ENGINE_2_DECISION_LOG", os.path.join(BASE_DIR, "engine_2", "reports", "live_decisions.jsonl"))
MAX_FILE_AGE_S = int(os.environ.get("ENGINE_2_MAX_FILE_AGE_S", "3600"))

ACTION_TO_DIRECTION = {BUY: UP, SELL: DOWN, HOLD: NEUTRAL}


class EngineTwoAdapter(EngineAdapter):
    name = "engine_2"

    def __init__(self, enabled: bool | None = None):
        s = get_settings()
        super().__init__(s.engines.engine_2_enabled if enabled is None else enabled)
        self.timeout_s = s.engines.engine_2_timeout_s
        self.cache_s = s.engines.engine_2_cache_s
        self.models_dir = s.engines.engine_2_models_dir
        self._runner = None
        self._runner_error: str | None = None

    # ── inline ──────────────────────────────────────────────────────────────
    def _get_runner(self):
        if self._runner is not None:
            return self._runner
        from engine_2.inference import Runner            # imports TF lazily
        self._runner = Runner(models_dir=self.models_dir,
                              symbol=get_settings().execution.symbol)
        return self._runner

    def _from_inline(self, symbol: str, context: dict) -> EngineSignal:
        runner = self._get_runner()
        # The policy's state vector contains position/PnL/bars-held. Feed it the
        # broker's real position so it reasons about the book we actually have.
        pos = context.get("position") or {}
        runner.position = 1 if float(pos.get("quantity") or 0) > 0 else 0
        runner.entry_px = float(pos.get("avg_entry_price") or 0.0)
        runner.bars_in = int(pos.get("bars_held") or 0)
        return self._normalize(symbol, runner.step(), "inline")

    # ── file ────────────────────────────────────────────────────────────────
    def _from_file(self, symbol: str) -> EngineSignal:
        if not os.path.exists(DECISION_LOG):
            raise FileNotFoundError(f"no engine_2 decision log at {DECISION_LOG}")
        with open(DECISION_LOG, "rb") as fh:               # tail without reading it all
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            chunk = min(size, 8192)
            fh.seek(size - chunk)
            lines = [ln for ln in fh.read().decode("utf-8", "replace").splitlines() if ln.strip()]
        if not lines:
            raise ValueError("engine_2 decision log is empty")
        payload = json.loads(lines[-1])
        age = time.time() - float(payload.get("ts", 0)) / 1000.0
        if age > MAX_FILE_AGE_S:
            raise ValueError(f"engine_2 decision log is stale ({age / 60:.0f} min old)")
        signal = self._normalize(symbol, payload, "file")
        signal.stale = age > 2 * 900
        return signal

    def _compute(self, symbol: str, context: dict) -> EngineSignal:
        errors = []
        if SOURCE in ("auto", "inline"):
            try:
                return self._from_inline(symbol, context)
            except Exception as exc:
                errors.append(f"inline: {type(exc).__name__}: {exc}")
                log.info("engine_2 inline unavailable (%s)", errors[-1])
                if SOURCE == "inline":
                    raise
        if SOURCE in ("auto", "file"):
            try:
                return self._from_file(symbol)
            except Exception as exc:
                errors.append(f"file: {type(exc).__name__}: {exc}")
                if SOURCE == "file":
                    raise
        raise RuntimeError(" | ".join(errors) or "engine_2 produced nothing")

    # ── normalization ───────────────────────────────────────────────────────
    def _normalize(self, symbol: str, d: dict, source: str) -> EngineSignal:
        """The PPO action is the opinion; the forecaster head is the evidence."""
        from engine_2 import config as C2                  # constants only, no TF

        action = str(d.get("action", HOLD)).upper()
        probs = [float(p) for p in (d.get("probs") or [])]
        p_up = [float(p) for p in (d.get("p_up") or [])]
        close = float(d.get("close") or 0.0)

        direction = ACTION_TO_DIRECTION.get(action, NEUTRAL)
        # Confidence = how decisive the policy was, tempered by the forecaster's
        # own dispersion. A 0.34/0.33/0.33 softmax is a shrug, not a signal.
        policy_conf = max(probs) if probs else 0.34
        decisiveness = max(0.0, (policy_conf - 1.0 / 3.0) / (1.0 - 1.0 / 3.0))
        forecast_edge = abs((sum(p_up) / len(p_up)) - 0.5) * 2 if p_up else 0.0
        # Four horizons pointing the same way is a different animal from a 2-2
        # split with the same mean, so agreement gets its own weight rather than
        # being averaged away.
        agreement = float(d.get("horizon_agreement") or 0.0)
        confidence = 0.5 * decisiveness + 0.3 * forecast_edge + 0.2 * abs(agreement)
        if direction == NEUTRAL:
            confidence = min(confidence, 0.5)

        stop = close * (1 - C2.STOP_LOSS_PCT) if close else None
        target = close * (1 + C2.TAKE_PROFIT_PCT) if close else None
        reasons = [
            f"PPO policy chose {action} with probabilities "
            f"{[round(p, 3) for p in probs]} over (HOLD, BUY, SELL).",
            f"Forecaster P(up) over the next {len(p_up) or C2.HORIZON} bars: "
            f"{[round(p, 3) for p in p_up]}.",
        ]
        if p_up:
            reasons.append(f"Horizon agreement {agreement:+.2f} "
                           f"({'unanimous' if abs(agreement) == 1 else 'split'} across "
                           f"h1..h{len(p_up)}).")
        decay = (d.get("drift") or {})
        if decay.get("verdict") == "degraded":
            reasons.append(
                f"Model decay warning: live directional accuracy "
                f"{decay.get('dir_acc', 0):.3f} over {decay.get('n', 0)} matured "
                f"predictions — discount this engine until it is retrained.")
        if d.get("volatility") is not None:
            reasons.append(f"Realized volatility {float(d['volatility']):.4f}; "
                           f"model position flag {d.get('position', 0)}.")

        return EngineSignal(
            engine=self.name, ok=True, symbol=symbol, generated_at=utcnow(),
            direction=direction, action_hint=action, confidence=confidence,
            horizon=f"{C2.HORIZON} x {C2.TIMEFRAME}",
            levels={"entry": close, "stop_loss": round(stop, 2) if stop else None,
                    "take_profit_1": round(target, 2) if target else None,
                    "stop_pct": C2.STOP_LOSS_PCT * 100, "target_pct": C2.TAKE_PROFIT_PCT * 100},
            features={"close": close, "p_up": p_up, "probs": probs,
                      "policy_confidence": round(policy_conf, 4),
                      "decisiveness": round(decisiveness, 4),
                      "forecast_edge": round(forecast_edge, 4),
                      "volatility": d.get("volatility"),
                      "model_position": d.get("position"),
                      "model_pnl": d.get("pnl"), "bars_in": d.get("bars_in"),
                      "horizon_agreement": d.get("horizon_agreement"),
                      "model_version": d.get("model_version"),
                      "drift": d.get("drift")},
            reasons=reasons, raw=d, source=source)
