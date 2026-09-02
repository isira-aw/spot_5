"""The 16 features, computed once, used everywhere.

Every feature here is CAUSAL: the value at bar t is a function of bars <= t only.
That is what makes the chronological split in dataset.py meaningful — a single
centred rolling window or a full-series z-score would leak the future into the
training set and inflate every metric downstream.

Contract, enforced by tests:
    build_features(candles)[t] depends only on candles[:t+1]

Feature 0 stays the standardized close so existing plotting code that reads
X[:, :, 0] as "price" keeps working.

IMPORTANT: if your original dataset notebook used a different set of 10 base
indicators, port them into _base_features() rather than retraining against these.
A model trained on one feature set and served another produces confident garbage,
and nothing in the stack will warn you.
"""
from __future__ import annotations

import numpy as np

from . import config as C

FEATURE_NAMES = [
    "close_z", "log_return", "rsi_centered", "macd_hist_norm", "bb_pctb_centered",
    "atr_norm", "volume_z", "ema_ratio", "realized_vol", "body_ratio",
    "wv_trend_dev", "wv_mid_osc", "wv_hf_noise",
    "wv_trend_slope", "wv_noise_ratio", "wv_spec_entropy",
]
assert len(FEATURE_NAMES) == C.NUM_FEATURES


# ── small causal helpers ─────────────────────────────────────────────────────
def _rolling(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing windows, left-padded with the first value (n, w)."""
    pad = np.concatenate([np.full(w - 1, a[0]), a])
    return np.lib.stride_tricks.sliding_window_view(pad, w)


def _sma(a, w):
    return _rolling(a, w).mean(axis=1)


def _std(a, w):
    return _rolling(a, w).std(axis=1)


def _ema(a: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(a)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def _rma(a: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing, used by RSI and ATR."""
    alpha = 1.0 / period
    out = np.empty_like(a)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    gain, loss = np.maximum(d, 0.0), np.maximum(-d, 0.0)
    rs = _rma(gain, period) / (_rma(loss, period) + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high, low, close, period: int = 14) -> np.ndarray:
    pc = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return _rma(tr, period)


# ── base block: 10 features ──────────────────────────────────────────────────
def _base_features(o, h, l, c, v) -> np.ndarray:
    eps = 1e-12
    n = len(c)
    f = np.zeros((n, 10), dtype=np.float32)

    # 0 close, z-scored against its own trailing 200 bars (scale-free, causal)
    f[:, 0] = (c - _sma(c, 200)) / (_std(c, 200) + eps)
    # 1 clipped log return
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    f[:, 1] = np.clip(lr, -C.RETURN_CLIP, C.RETURN_CLIP)
    # 2 RSI(14), centred to [-0.5, 0.5]
    f[:, 2] = _rsi(c, 14) / 100.0 - 0.5
    # 3 MACD histogram, normalized by price
    macd = _ema(c, 12) - _ema(c, 26)
    f[:, 3] = (macd - _ema(macd, 9)) / (c + eps)
    # 4 Bollinger %B, centred
    mid, sd = _sma(c, 20), _std(c, 20)
    f[:, 4] = (c - mid) / (4.0 * sd + eps)
    # 5 ATR as a fraction of price (the SL/TP unit)
    f[:, 5] = _atr(h, l, c, 14) / (c + eps)
    # 6 volume z-score
    f[:, 6] = (v - _sma(v, 100)) / (_std(v, 100) + eps)
    # 7 fast/slow EMA ratio
    f[:, 7] = _ema(c, 12) / (_ema(c, 48) + eps) - 1.0
    # 8 realized vol (20-bar std of returns)
    f[:, 8] = _std(lr, C.__dict__.get("VOLATILITY_WINDOW", 20))
    # 9 candle body as a share of its range
    f[:, 9] = (c - o) / (h - l + eps)

    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


# ── wavelet block: 6 features ────────────────────────────────────────────────
def _wavelet_features(close: np.ndarray, w: int = C.WAVELET_WINDOW) -> np.ndarray:
    """db4 level-2 DWT over the trailing w bars, recomputed at every bar.

    Recomputing per bar is the only causal way to do this. Decomposing the whole
    series once and slicing it is a classic leak: db4 is a symmetric-ish filter,
    so coefficient k is contaminated by bars after k.
    """
    try:
        import pywt
    except ModuleNotFoundError as exc:                  # pragma: no cover
        raise ModuleNotFoundError(
            "engine_2 needs PyWavelets for the 6 wavelet features. Install this "
            "package's dependencies:\n"
            "    pip install -r backend/engine_2/requirements.txt") from exc

    n = len(close)
    out = np.zeros((n, 6), dtype=np.float32)
    wavelet = pywt.Wavelet("db4")
    pad = np.concatenate([np.full(w - 1, close[0]), close])
    windows = np.lib.stride_tricks.sliding_window_view(pad, w)
    eps = 1e-12
    t = np.arange(0, dtype=np.float64)

    for i in range(n):
        seg = np.array(windows[i], dtype=np.float64)   # sliding views are read-only
        px = seg[-1]
        cA2, cD2, cD1 = pywt.wavedec(seg, wavelet, level=2, mode="periodization")

        trend = pywt.waverec([cA2, np.zeros_like(cD2), np.zeros_like(cD1)],
                             wavelet, mode="periodization")[:w]
        e2, e1, ea = cD2 @ cD2, cD1 @ cD1, cA2 @ cA2
        tot = ea + e2 + e1 + eps
        p = np.array([ea, e2, e1]) / tot

        if len(t) != len(cA2):
            t = np.arange(len(cA2), dtype=np.float64)
        slope = np.polyfit(t, cA2, 1)[0] if len(cA2) > 1 else 0.0

        out[i, 0] = (px - trend[-1]) / (px + eps)          # trend deviation
        out[i, 1] = cD2.std() / (px + eps)                 # mid-frequency swing
        out[i, 2] = cD1.std() / (px + eps)                 # high-frequency noise
        out[i, 3] = slope / (px + eps)                     # trend slope
        out[i, 4] = (e2 + e1) / tot                        # noise / total energy
        out[i, 5] = -(p * np.log(p + eps)).sum() / np.log(3.0)   # spectral entropy

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_features(candles: np.ndarray) -> np.ndarray:
    """candles: (n, >=6) [ts, o, h, l, c, v]  ->  (n, 16) float32, causal."""
    candles = np.asarray(candles, dtype=np.float64)
    o, h, l, c, v = (candles[:, 1], candles[:, 2], candles[:, 3],
                     candles[:, 4], candles[:, 5])
    return np.concatenate([_base_features(o, h, l, c, v),
                           _wavelet_features(c)], axis=1).astype(np.float32)


def realized_volatility(close: np.ndarray, window: int = 20) -> np.ndarray:
    """The `volatility` slot of the PPO state vector (state index 8)."""
    lr = np.diff(np.log(close), prepend=np.log(close[0]))
    return _std(lr, window).astype(np.float32)
