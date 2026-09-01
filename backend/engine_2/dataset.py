"""Windows, soft labels, and a strict chronological split (point 2).

Three leaks are closed here, in order of how much damage they do:

1. Split order. Train is the OLDEST slice, val the middle, test the newest.
   Nothing is shuffled across the boundary. (Shuffling WITHIN train, as
   model.fit(shuffle=True) does, is fine and stays.)
2. Embargo. Window t spans bars [t, t+128) and its label reads to t+128+4.
   Adjacent windows across a boundary therefore share candles, so a gap of
   WINDOW_SIZE + HORIZON bars is dropped at each seam.
3. Scaling. Mean/std come from the training slice only and are stored in the
   npz, so live inference applies the exact same transform.

CLI:  python -m trader.dataset
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C
from .features import build_features, realized_volatility
from .fetch import load_cache


def soft_labels(close: np.ndarray, anchor_idx: np.ndarray) -> np.ndarray:
    """sigmoid(future_return * SCALE) for h = 1..HORIZON, clipped."""
    out = np.zeros((len(anchor_idx), C.HORIZON), dtype=np.float32)
    anchor = close[anchor_idx]
    for h in range(1, C.HORIZON + 1):
        fut = close[anchor_idx + h]
        ret = np.clip((fut - anchor) / anchor, -C.RETURN_CLIP, C.RETURN_CLIP)
        out[:, h - 1] = 1.0 / (1.0 + np.exp(-ret * C.SOFT_LABEL_SCALE))
    return np.clip(out, *C.SOFT_LABEL_CLIP).astype(np.float32)


def make_windows(candles: np.ndarray):
    """-> (X, anchor_idx). X[i] covers candles[i0 : i0+WINDOW], anchor = last bar."""
    feats = build_features(candles)
    n = len(candles)
    first = C.INDICATOR_WARMUP
    last = n - C.HORIZON - C.WINDOW_SIZE      # exclusive: label must exist
    starts = np.arange(first, last, dtype=np.int64)
    X = np.lib.stride_tricks.sliding_window_view(
        feats, (C.WINDOW_SIZE, C.NUM_FEATURES))[starts, 0]
    anchor = starts + C.WINDOW_SIZE - 1
    return np.ascontiguousarray(X, dtype=np.float32), anchor


def chronological_split(n: int, val_frac=C.VAL_FRACTION, test_frac=C.TEST_FRACTION,
                        embargo=C.EMBARGO_BARS):
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_lo = n - n_test
    val_lo = test_lo - n_val
    tr = slice(0, max(0, val_lo - embargo))
    va = slice(val_lo, max(val_lo, test_lo - embargo))
    te = slice(test_lo, n)
    return tr, va, te


def build(path_csv=C.RAW_CSV, out=C.DATASET_NPZ, verbose=True):
    candles = load_cache(path_csv)
    close = candles[:, 4]
    X, anchor = make_windows(candles)
    y = soft_labels(close, anchor)
    vol = realized_volatility(close)[anchor]

    tr, va, te = chronological_split(len(X))

    mu = X[tr].reshape(-1, C.NUM_FEATURES).mean(axis=0)
    sd = X[tr].reshape(-1, C.NUM_FEATURES).std(axis=0) + 1e-8
    Xs = ((X - mu) / sd).astype(np.float32)

    meta = {
        "symbol": C.SYMBOL, "timeframe": C.TIMEFRAME,
        "feature_count": C.NUM_FEATURES, "window": C.WINDOW_SIZE,
        "horizon": C.HORIZON, "embargo": C.EMBARGO_BARS,
        "ts_train": [int(candles[anchor[tr][0], 0]), int(candles[anchor[tr][-1], 0])],
        "ts_val": [int(candles[anchor[va][0], 0]), int(candles[anchor[va][-1], 0])],
        "ts_test": [int(candles[anchor[te][0], 0]), int(candles[anchor[te][-1], 0])],
    }

    np.savez_compressed(
        out,
        X_train=Xs[tr], y_train=y[tr],
        X_val=Xs[va], y_val=y[va],
        X_test=Xs[te], y_test=y[te],
        # index-aligned extras for the backtester / PPO env
        anchor_train=anchor[tr], anchor_val=anchor[va], anchor_test=anchor[te],
        vol_train=vol[tr], vol_val=vol[va], vol_test=vol[te],
        candles=candles, feat_mean=mu.astype(np.float32), feat_std=sd.astype(np.float32),
        meta=json.dumps(meta),
    )

    if verbose:
        import pandas as pd
        fmt = lambda ms: str(pd.to_datetime(ms, unit="ms").date())
        print(f"dataset -> {out}")
        for name, s in (("train", tr), ("val", va), ("test", te)):
            k = meta[f"ts_{name}"]
            print(f"  {name:5s} {X[s].shape[0]:>7,} windows  "
                  f"{fmt(k[0])} .. {fmt(k[1])}  "
                  f"up-rate={float((y[s][:, 0] > 0.5).mean()):.3f}")
        print(f"  embargo {C.EMBARGO_BARS} bars at each seam")
    return out


def load(path=C.DATASET_NPZ):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files if k != "meta"} | {"meta": json.loads(str(d["meta"]))}


if __name__ == "__main__":
    build()
