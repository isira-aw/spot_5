"""Live inference loop — decoupled from training by design.

Wakes a few seconds after each 15m candle closes, pulls the last
WINDOW_SIZE + INDICATOR_WARMUP closed candles, rebuilds the same 16 features with
the same code path used in training, applies the scaler saved alongside the
model, and asks the actor for an action. It never trains, never blocks on the
retrain job, and hot-reloads models only when their mtime changes — so a
promotion takes effect on the next bar without a restart.

Every decision is also fed to `drift.observe`, which matches it against what
the market actually did HORIZON bars later and keeps a rolling scorecard of live
directionalAccuracy / predStd. That is the only measurement that can tell you the
model has decayed since it was promoted.

    python -m engine_2.inference --paper
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import config as C
from .features import build_features, realized_volatility
from . import drift, registry
from .fetch import fetch_live_window, _exchange

ACTIONS = ["HOLD", "BUY", "SELL"]


class Runner:
    def __init__(self, models_dir=C.MODELS_DIR, symbol=C.SYMBOL,
                 timeframe=C.TIMEFRAME, greedy=True):
        self.models_dir, self.symbol, self.timeframe = models_dir, symbol, timeframe
        self.greedy = greedy
        self._stamp = None
        self.position, self.entry_px, self.bars_in = 0, 0.0, 0
        self.exchange = _exchange()
        self._load()

    # ── model hot-reload ────────────────────────────────────────────────────
    def _mtimes(self):
        return tuple(os.path.getmtime(f"{self.models_dir}/{s}/model.keras")
                     for s in ("forecaster", "ppo/policy"))

    def _load(self):
        from tensorflow import keras
        from . import models as M
        load = lambda p: keras.models.load_model(p, custom_objects=M.CUSTOM_OBJECTS,
                                                 compile=False)
        self.forecaster = load(f"{self.models_dir}/forecaster/model.keras")
        self.actor = load(f"{self.models_dir}/ppo/policy/model.keras")
        s = np.load(f"{self.models_dir}/scaler.npz")
        self.mu, self.sd = s["mean"], s["std"]
        self.assemble_state = M.assemble_state
        self.horizon_agreement = M.horizon_agreement
        self.model_version = registry.current_version()
        self._stamp = self._mtimes()
        print(f"[{time.strftime('%H:%M:%S')}] models loaded from {self.models_dir} "
              f"(version {self.model_version or 'unversioned'})")

    def _maybe_reload(self):
        if self._mtimes() != self._stamp:
            print("model files changed — reloading")
            self._load()

    # ── one decision ────────────────────────────────────────────────────────
    def step(self) -> dict:
        self._maybe_reload()
        candles = fetch_live_window(self.symbol, self.timeframe,
                                    C.WINDOW_SIZE + C.INDICATOR_WARMUP, self.exchange)
        feats = build_features(candles)[-C.WINDOW_SIZE:]
        x = ((feats - self.mu) / self.sd).astype(np.float32)[None, ...]
        forecast = self.forecaster(x, training=False).numpy()[0]

        i = len(candles) - 1
        vol = float(realized_volatility(candles[:, 4])[i])
        state = self.assemble_state(candles, i, forecast, self.position,
                                    self.entry_px, self.bars_in, vol)
        probs = self.actor(state[None, :], training=False).numpy()[0]
        action = int(np.argmax(probs)) if self.greedy else \
            int(np.random.choice(3, p=probs))

        close = float(candles[i, 4])
        pnl = (close / self.entry_px - 1.0) if self.position == 1 else 0.0
        # risk exits override the policy, exactly as in the backtest
        if self.position == 1 and (pnl <= -C.STOP_LOSS_PCT or pnl >= C.TAKE_PROFIT_PCT):
            action = 2

        # Score the previous predictions that have now matured, and file this one.
        # Never let monitoring take the decision loop down with it.
        drift_status = {}
        try:
            drift_status = drift.observe(int(candles[i, 0]), close,
                                         [float(p) for p in forecast],
                                         self.model_version)
        except Exception as exc:
            print(f"drift monitor unavailable: {type(exc).__name__}: {exc}")

        return {"ts": int(candles[i, 0]), "close": close, "action": ACTIONS[action],
                "action_id": action, "probs": [round(float(p), 4) for p in probs],
                "p_up": [round(float(p), 4) for p in forecast],
                "horizon_agreement": self.horizon_agreement(forecast),
                "position": self.position, "pnl": round(pnl, 5),
                "bars_in": self.bars_in, "volatility": round(vol, 6),
                "slippage_assumed": round(float(C.slippage_for_vol(vol)), 6),
                "model_version": self.model_version,
                "drift": {k: drift_status.get(k) for k in
                          ("verdict", "dir_acc", "pred_std", "n",
                           "retrain_recommended")} if drift_status else None}

    def apply(self, d: dict):
        """Paper-trade bookkeeping. Swap for real order placement when live."""
        if self.position == 0 and d["action_id"] == 1:
            self.position, self.entry_px, self.bars_in = 1, d["close"], 0
        elif self.position == 1 and d["action_id"] == 2:
            self.position, self.entry_px, self.bars_in = 0, 0.0, 0
        elif self.position == 1:
            self.bars_in += 1


def sleep_to_next_bar(offset_s: float = 5.0):
    now = time.time()
    period = C.BAR_MS / 1000.0
    time.sleep(max(1.0, period - (now % period) + offset_s))


def main(paper=True, once=False, log_path=None):
    r = Runner()
    log_path = log_path or os.path.join(C.REPORTS_DIR, "live_decisions.jsonl")
    while True:
        if not once:
            sleep_to_next_bar()
        try:
            d = r.step()
            if paper:
                r.apply(d)
            print(f"[{time.strftime('%H:%M:%S')}] {d['action']:<4} "
                  f"px={d['close']:.2f} p_up={d['p_up'][0]:.3f} "
                  f"probs={d['probs']} pos={d['position']}")
            with open(log_path, "a") as f:
                f.write(json.dumps(d) + "\n")
        except Exception as e:                      # network blips must not kill the loop
            print(f"[{time.strftime('%H:%M:%S')}] step failed: {type(e).__name__}: {e}")
        if once:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true", default=True)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    main(paper=a.paper, once=a.once)
