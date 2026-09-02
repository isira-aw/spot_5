"""Windows, volatility-scaled soft labels, and a strict chronological split.

Four leaks are closed here, in order of how much damage they do:

1. Split order. Train is the OLDEST slice, val the middle, test the newest.
   Nothing is shuffled across the boundary. (Shuffling WITHIN train, as
   model.fit(shuffle=True) does, is fine and stays.)
2. Embargo. Window t spans bars [t, t+128) and its label reads to t+128+4.
   Adjacent windows across a boundary therefore share candles, so a gap of
   WINDOW_SIZE + HORIZON bars is dropped at each seam.
3. Scaling. Mean/std come from the training slice only and are stored in the
   npz, so live inference applies the exact same transform.
4. Holdout. A final block after `test` is written but never read by training,
   early stopping or the promotion gate — only by the final backtest. Every
   other slice has had at least one decision made on it.

CLI:  python -m engine_2.dataset
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C
from .features import build_features, realized_volatility
from .fetch import load_cache


def label_sigma(close: np.ndarray, window: int = C.LABEL_VOL_WINDOW) -> np.ndarray:
    """Trailing per-bar return sigma at each bar, floored. Causal by construction:
    sigma[t] is computed from returns up to and including t."""
    lr = np.diff(np.log(close), prepend=np.log(close[0]))
    pad = np.concatenate([np.full(window - 1, lr[0]), lr])
    sig = np.lib.stride_tricks.sliding_window_view(pad, window).std(axis=1)
    return np.maximum(sig, C.LABEL_VOL_FLOOR).astype(np.float64)


def soft_labels(close: np.ndarray, anchor_idx: np.ndarray,
                sigma: np.ndarray | None = None) -> np.ndarray:
    """P(up) for h = 1..HORIZON, scaled by how unusual the move is.

    The notebook used sigmoid(return * 400). At 15m on BTC a typical move is
    0.05-0.3%, which lands at 0.51-0.56: the label barely moves off 0.5, so the
    loss is dominated by the handful of >1% bars and the network has almost no
    gradient to learn the ordinary case from. Worse, the same 0.3% move means
    "violent" in a calm regime and "noise" in a volatile one, and a fixed scale
    cannot tell them apart.

    Here the h-bar return is divided by the trailing sigma of 1-bar returns,
    grown as sqrt(h) because that is how a random walk's dispersion grows. The
    label is then sigmoid(z * LABEL_Z_SCALE): a +1 sigma move is 0.73, a -2 sigma
    move is 0.12, in every regime. The clip on |z| keeps a flash crash from
    saturating a whole batch.
    """
    if sigma is None:
        sigma = label_sigma(close)
    out = np.zeros((len(anchor_idx), C.HORIZON), dtype=np.float32)
    anchor = close[anchor_idx]
    sig = sigma[anchor_idx]
    for h in range(1, C.HORIZON + 1):
        fut = close[anchor_idx + h]
        ret = np.clip((fut - anchor) / anchor, -C.RETURN_CLIP, C.RETURN_CLIP)
        z = np.clip(ret / (sig * np.sqrt(h)), -C.LABEL_Z_CLIP, C.LABEL_Z_CLIP)
        out[:, h - 1] = 1.0 / (1.0 + np.exp(-z * C.LABEL_Z_SCALE))
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
                        holdout_frac=C.BACKTEST_HOLDOUT, embargo=C.EMBARGO_BARS):
    """-> (train, val, test, holdout) slices, oldest first, embargoed at each seam.

    Nothing is shuffled across a boundary and every seam drops `embargo` windows,
    because window t spans bars [t, t+WINDOW) and its label reads to t+WINDOW+H:
    adjacent windows across a naive boundary literally share candles.
    """
    n_hold = int(n * holdout_frac)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    hold_lo = n - n_hold
    test_lo = hold_lo - n_test
    val_lo = test_lo - n_val
    tr = slice(0, max(0, val_lo - embargo))
    va = slice(val_lo, max(val_lo, test_lo - embargo))
    te = slice(test_lo, max(test_lo, hold_lo - embargo))
    ho = slice(hold_lo, n)
    return tr, va, te, ho


def build(path_csv=C.RAW_CSV, out=C.DATASET_NPZ, verbose=True):
    candles = load_cache(path_csv)
    close = candles[:, 4]
    X, anchor = make_windows(candles)
    y = soft_labels(close, anchor)
    vol = realized_volatility(close)[anchor]

    tr, va, te, ho = chronological_split(len(X))

    mu = X[tr].reshape(-1, C.NUM_FEATURES).mean(axis=0)
    sd = X[tr].reshape(-1, C.NUM_FEATURES).std(axis=0) + 1e-8
    Xs = ((X - mu) / sd).astype(np.float32)

    slices = {"train": tr, "val": va, "test": te, "holdout": ho}
    meta = {
        "symbol": C.SYMBOL, "timeframe": C.TIMEFRAME,
        "feature_count": C.NUM_FEATURES, "window": C.WINDOW_SIZE,
        "horizon": C.HORIZON, "embargo": C.EMBARGO_BARS,
        "label": {"kind": "vol_scaled_sigmoid", "vol_window": C.LABEL_VOL_WINDOW,
                  "z_scale": C.LABEL_Z_SCALE, "z_clip": C.LABEL_Z_CLIP},
    }
    for name, s in slices.items():
        meta[f"ts_{name}"] = [int(candles[anchor[s][0], 0]),
                              int(candles[anchor[s][-1], 0])] if X[s].shape[0] else [0, 0]

    arrays = {}
    for name, s in slices.items():
        arrays[f"X_{name}"] = Xs[s]
        arrays[f"y_{name}"] = y[s]
        arrays[f"anchor_{name}"] = anchor[s]
        arrays[f"vol_{name}"] = vol[s]

    np.savez_compressed(
        out, **arrays,
        candles=candles, feat_mean=mu.astype(np.float32), feat_std=sd.astype(np.float32),
        meta=json.dumps(meta),
    )

    if verbose:
        import pandas as pd
        fmt = lambda ms: str(pd.to_datetime(ms, unit="ms").date())
        print(f"dataset -> {out}")
        for name, s in slices.items():
            k = meta[f"ts_{name}"]
            print(f"  {name:8s} {X[s].shape[0]:>7,} windows  "
                  f"{fmt(k[0])} .. {fmt(k[1])}  "
                  f"up-rate={float((y[s][:, 0] > 0.5).mean()) if X[s].shape[0] else 0:.3f}  "
                  f"label-sd={float(y[s][:, 0].std()) if X[s].shape[0] else 0:.3f}")
        print(f"  embargo {C.EMBARGO_BARS} bars at each seam; "
              f"holdout is never read before the final backtest")
    return out


def load(path=C.DATASET_NPZ):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files if k != "meta"} | {"meta": json.loads(str(d["meta"]))}


if __name__ == "__main__":
    build()
