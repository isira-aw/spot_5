"""Single source of truth for every stage: fetch -> dataset -> train -> backtest -> live.

If a constant lives here, no other module is allowed to redefine it. Feature drift
between training and inference is the most common silent killer in this kind of
system, so the feature builder, the label maker and the live loop all read from
this file.
"""
from __future__ import annotations

import os

# ── Market ───────────────────────────────────────────────────────────────────
EXCHANGE_ID = "binance"
SYMBOL = "BTC/USDT"          # start with ONE pair (point 3). Add more only after
                             # a single-pair edge survives walk-forward.
TIMEFRAME = "15m"
TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
         "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
BAR_MS = TF_MS[TIMEFRAME]
BARS_PER_YEAR = int(365 * 24 * 60 * 60 * 1000 / BAR_MS)   # 35_040 for 15m

HISTORY_YEARS = 3             # ~105k bars of 15m -> covers 2022 bear, 2023 chop,
                              # 2024 bull, 2025. Multiple volatility regimes.

# ── Windowing / labels ───────────────────────────────────────────────────────
WINDOW_SIZE = 128
HORIZON = 4
NUM_FEATURES = 16             # 10 price/indicator + 6 wavelet
INDICATOR_WARMUP = 200        # bars discarded at the head so no indicator is
                              # computed from a partially-filled buffer
WAVELET_WINDOW = 64           # trailing bars fed to each per-bar DWT
SOFT_LABEL_SCALE = 400.0
SOFT_LABEL_CLIP = (0.02, 0.98)
RETURN_CLIP = 0.10

# ── Split (point 2): strict chronological, with an embargo gap ───────────────
# The last training window overlaps the first validation window by WINDOW_SIZE
# bars, and its label peeks HORIZON bars ahead. Without an embargo of at least
# WINDOW_SIZE + HORIZON bars the two sets literally share candles.
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
EMBARGO_BARS = WINDOW_SIZE + HORIZON

# ── Execution model ──────────────────────────────────────────────────────────
FEE_RATE = 0.00075            # taker, BNB discount
SLIPPAGE_PCT = 0.0005
STOP_LOSS_PCT = 0.0030
TAKE_PROFIT_PCT = 0.0050

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.environ.get("TRADER_ROOT", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_CSV = os.path.join(DATA_DIR, f"{SYMBOL.replace('/', '')}_{TIMEFRAME}.csv")
DATASET_NPZ = os.path.join(DATA_DIR, "dataset.npz")
MODELS_DIR = os.path.join(ROOT, "models")
CANDIDATE_DIR = os.path.join(ROOT, "models_candidate")
REPORTS_DIR = os.path.join(ROOT, "reports")

for _d in (DATA_DIR, MODELS_DIR, CANDIDATE_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)
