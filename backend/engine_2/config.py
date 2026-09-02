"""Single source of truth for every stage: fetch -> dataset -> train -> gate -> live.

If a constant lives here, no other module is allowed to redefine it. Feature drift
between training and inference is the most common silent killer in this kind of
system, so the feature builder, the label maker and the live loop all read from
this file.

Every constant that an operator might want to move between environments is
env-overridable with the same ``ENGINE_2_*`` prefix the rest of the backend uses
(see ``backend/.env.example``). Defaults are the values this pipeline was
validated with.
"""
from __future__ import annotations

import os


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _num(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except ValueError:
        return default


def _flag(name: str, default: bool = False) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


# ── Market ───────────────────────────────────────────────────────────────────
EXCHANGE_ID = _env("EXCHANGE_ID", "binance")
SYMBOL = _env("ENGINE_2_SYMBOL", "BTC/USDT")
# Deliberately ONE pair. The notebook trained the forecaster on 7 symbols and the
# agent on BTC alone, which conflates "does this generalise across assets" with
# "does it work at all". This is a documented BTC/USDT-only policy; see README.
TIMEFRAME = _env("ENGINE_2_TIMEFRAME", "15m")
TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
         "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
BAR_MS = TF_MS[TIMEFRAME]
BARS_PER_YEAR = int(365 * 24 * 60 * 60 * 1000 / BAR_MS)   # 35_040 for 15m

HISTORY_YEARS = _num("ENGINE_2_HISTORY_YEARS", 3)   # ~105k 15m bars: bear, chop, bull

# ── Read-only market-data credentials ────────────────────────────────────────
# OPTIONAL. Binance serves public OHLCV unauthenticated; a key only buys a higher
# rate limit. The key MUST be read-only (no trading, no withdrawals) — see
# `fetch.assert_read_only`, which refuses to run if the key can trade.
BINANCE_API_KEY = _env("BINANCE_API_KEY")
BINANCE_API_SECRET = _env("BINANCE_API_SECRET")

# ── Windowing / labels ───────────────────────────────────────────────────────
WINDOW_SIZE = 128
HORIZON = 4                   # h1..h4 = 15..60 min ahead; all four are consumed
                              # by the state vector AND the reward (agreement bonus)
NUM_FEATURES = 16             # 10 price/indicator + 6 wavelet
INDICATOR_WARMUP = 200        # bars discarded at the head so no indicator is
                              # computed from a partially-filled buffer
WAVELET_WINDOW = 64           # trailing bars fed to each per-bar DWT

# Soft labels are scaled by ROLLING VOLATILITY, not a constant. A fixed
# SOFT_LABEL_SCALE=400 maps a typical 15m move (0.05-0.3%) to 0.51-0.56 — a label
# that is almost indistinguishable from "no information", so the network spends
# its capacity on the tails and learns nothing about the body of the
# distribution. Dividing by the trailing sigma of returns asks the only question
# worth asking: how unusual is this move for this regime.
LABEL_VOL_WINDOW = _int("ENGINE_2_LABEL_VOL_WINDOW", 96)     # 24h of 15m bars
LABEL_VOL_FLOOR = _num("ENGINE_2_LABEL_VOL_FLOOR", 2.5e-4)   # ~2.5 bps/bar
LABEL_Z_SCALE = _num("ENGINE_2_LABEL_Z_SCALE", 1.0)          # logit = z * scale
LABEL_Z_CLIP = _num("ENGINE_2_LABEL_Z_CLIP", 4.0)            # |z| cap before sigmoid
SOFT_LABEL_CLIP = (0.02, 0.98)
RETURN_CLIP = 0.10

# ── Split: strict chronological, embargoed, with a final untouched holdout ───
# train | emb | val | emb | test | emb | holdout
# `test` gates promotion. `holdout` is touched exactly once, by the final
# backtest, and never by any tuning decision — it is the only slice whose numbers
# were not selected on.
TEST_FRACTION = _num("ENGINE_2_TEST_FRACTION", 0.15)
VAL_FRACTION = _num("ENGINE_2_VAL_FRACTION", 0.15)
BACKTEST_HOLDOUT = _num("ENGINE_2_BACKTEST_HOLDOUT", 0.10)
EMBARGO_BARS = WINDOW_SIZE + HORIZON

# ── Execution model ──────────────────────────────────────────────────────────
FEE_RATE = _num("ENGINE_2_FEE_RATE", 0.00075)        # taker, BNB discount
SLIPPAGE_PCT = _num("ENGINE_2_SLIPPAGE_PCT", 0.0005) # slippage at REFERENCE_VOL
# Slippage is not a constant. Spreads widen and books thin exactly when the model
# most wants to trade, so cost is modelled as a function of the same realized-vol
# feature the policy sees: slip = base * clip(vol/ref, lo, hi).
REFERENCE_VOL = _num("ENGINE_2_REFERENCE_VOL", 0.0025)
SLIPPAGE_VOL_MIN = _num("ENGINE_2_SLIPPAGE_VOL_MIN", 0.6)
SLIPPAGE_VOL_MAX = _num("ENGINE_2_SLIPPAGE_VOL_MAX", 4.0)
STOP_LOSS_PCT = _num("ENGINE_2_STOP_LOSS_PCT", 0.0030)
TAKE_PROFIT_PCT = _num("ENGINE_2_TAKE_PROFIT_PCT", 0.0050)

# ── Hard gates (they raise; they do not print) ───────────────────────────────
GATE_MIN_PRED_STD = _num("ENGINE_2_GATE_MIN_PRED_STD", 0.02)
GATE_MIN_DIR_ACC = _num("ENGINE_2_GATE_MIN_DIR_ACC", 0.51)
GATE_MAX_PRED_MEAN_DEV = _num("ENGINE_2_GATE_MAX_PRED_MEAN_DEV", 0.20)  # |mean-0.5|
GATE_MIN_POLICY_SPREAD = _num("ENGINE_2_GATE_MIN_POLICY_SPREAD", 0.05)  # see gates.py

# ── Drift monitoring in production ───────────────────────────────────────────
DRIFT_WINDOW = _int("ENGINE_2_DRIFT_WINDOW", 480)          # live predictions kept
DRIFT_MIN_SAMPLES = _int("ENGINE_2_DRIFT_MIN_SAMPLES", 96) # before judging
DRIFT_MIN_DIR_ACC = _num("ENGINE_2_DRIFT_MIN_DIR_ACC", 0.51)
DRIFT_MIN_PRED_STD = _num("ENGINE_2_DRIFT_MIN_PRED_STD", 0.02)
DRIFT_BREACHES_TO_ALERT = _int("ENGINE_2_DRIFT_BREACHES", 3)  # consecutive checks

# ── Model registry ───────────────────────────────────────────────────────────
MODEL_RETENTION = _int("ENGINE_2_MODEL_RETENTION", 5)      # versions kept on disk

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.environ.get("ENGINE_2_ROOT",
                      os.environ.get("TRADER_ROOT",
                                     os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
RAW_CSV = os.path.join(DATA_DIR, f"{SYMBOL.replace('/', '')}_{TIMEFRAME}.csv")
DATASET_NPZ = os.path.join(DATA_DIR, "dataset.npz")
MODELS_DIR = os.environ.get("ENGINE_2_MODELS_DIR", os.path.join(ROOT, "models"))
VERSIONS_DIR = os.path.join(ROOT, "models_versions")
CANDIDATE_DIR = os.path.join(ROOT, "models_candidate")
REPORTS_DIR = os.path.join(ROOT, "reports")

for _d in (DATA_DIR, MODELS_DIR, VERSIONS_DIR, CANDIDATE_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)


def slippage_for_vol(vol):
    """Slippage at a given realized volatility. Used by the backtester, the PPO
    reward and the live loop so all three price a fill the same way."""
    import numpy as np
    ratio = np.clip(np.asarray(vol, dtype=float) / REFERENCE_VOL,
                    SLIPPAGE_VOL_MIN, SLIPPAGE_VOL_MAX)
    return SLIPPAGE_PCT * ratio
