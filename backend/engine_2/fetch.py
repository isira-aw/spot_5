"""Paginated OHLCV fetch with an incremental on-disk cache (point 1 + point 3).

Replaces the static dataset.npz upload. First run pulls HISTORY_YEARS of candles
one page at a time; every later run only asks for what is missing since the last
cached bar, so a weekly retrain costs a handful of requests instead of 400.

CLI:
    python -m trader.fetch                    # top up the cache to "now"
    python -m trader.fetch --years 3 --force  # rebuild from scratch
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config as C

COLS = ["timestamp", "open", "high", "low", "close", "volume"]
PAGE_LIMIT = 1000


def _exchange():
    import ccxt  # imported lazily so backtests run without ccxt installed
    ex = getattr(ccxt, C.EXCHANGE_ID)({"enableRateLimit": True})
    ex.load_markets()
    return ex


def fetch_range(symbol: str, timeframe: str, since_ms: int,
                until_ms: int | None = None, exchange=None) -> pd.DataFrame:
    """Page forward from since_ms until the exchange stops returning new bars."""
    ex = exchange or _exchange()
    until_ms = until_ms or ex.milliseconds()
    step = C.TF_MS[timeframe]
    cursor, rows = since_ms, []

    while cursor < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=PAGE_LIMIT)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if last <= cursor:          # exchange refused to advance: stop, not spin
            break
        cursor = last + step
        time.sleep(max(ex.rateLimit, 200) / 1000.0)
        print(f"  {pd.to_datetime(last, unit='ms')}  ({len(rows)} bars)", end="\r")

    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(rows, columns=COLS)
    return df[df.timestamp < until_ms]


def _sanitize(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Deduplicate, sort, and report gaps. Gaps are reported, never interpolated:
    a synthetic candle is a fabricated training example."""
    df = (df.drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True))
    step = C.TF_MS[timeframe]
    gaps = np.diff(df.timestamp.values)
    n_missing = int(((gaps - step) // step).clip(min=0).sum())
    if n_missing:
        worst = int(gaps.max() // step) - 1
        print(f"  warning: {n_missing} missing bars, largest gap {worst} bars "
              f"(exchange downtime / delisting windows) — left as-is")
    return df


def update_cache(symbol=C.SYMBOL, timeframe=C.TIMEFRAME, years=C.HISTORY_YEARS,
                 path=C.RAW_CSV, force=False) -> pd.DataFrame:
    ex = _exchange()
    now = ex.milliseconds()
    want_since = now - int(years * 365.25 * 24 * 3600 * 1000)

    old = pd.DataFrame(columns=COLS)
    if not force:
        try:
            old = pd.read_csv(path)
        except FileNotFoundError:
            pass

    frames = []
    if len(old):
        # backfill anything older than the cache, then top up the head
        if old.timestamp.min() > want_since + C.TF_MS[timeframe]:
            print(f"Backfilling {pd.to_datetime(want_since, unit='ms').date()} -> "
                  f"{pd.to_datetime(old.timestamp.min(), unit='ms').date()}")
            frames.append(fetch_range(symbol, timeframe, want_since,
                                      int(old.timestamp.min()), ex))
        frames.append(old)
        head = int(old.timestamp.max()) + C.TF_MS[timeframe]
    else:
        head = want_since

    print(f"Fetching {pd.to_datetime(head, unit='ms')} -> now")
    frames.append(fetch_range(symbol, timeframe, head, now, ex))

    df = _sanitize(pd.concat(frames, ignore_index=True), timeframe)
    df = df[df.timestamp >= want_since].reset_index(drop=True)

    # Drop the in-progress candle: it has not closed and its OHLC will change.
    if len(df) and df.timestamp.iloc[-1] + C.TF_MS[timeframe] > now:
        df = df.iloc[:-1]

    df.to_csv(path, index=False)
    print(f"\n{len(df)} bars cached -> {path}")
    print(f"  {pd.to_datetime(df.timestamp.iloc[0], unit='ms')} .. "
          f"{pd.to_datetime(df.timestamp.iloc[-1], unit='ms')}")
    return df


def load_cache(path=C.RAW_CSV) -> np.ndarray:
    """-> float64 array (n, 6): timestamp, open, high, low, close, volume."""
    return pd.read_csv(path)[COLS].to_numpy(dtype=np.float64)


def fetch_live_window(symbol=C.SYMBOL, timeframe=C.TIMEFRAME,
                      n=C.WINDOW_SIZE + C.INDICATOR_WARMUP, exchange=None) -> np.ndarray:
    """Latest n CLOSED candles for live inference."""
    ex = exchange or _exchange()
    raw = ex.fetch_ohlcv(symbol, timeframe, limit=n + 1)
    arr = np.asarray(raw, dtype=np.float64)
    if arr[-1, 0] + C.TF_MS[timeframe] > ex.milliseconds():
        arr = arr[:-1]
    return arr[-n:]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default=C.SYMBOL)
    p.add_argument("--timeframe", default=C.TIMEFRAME)
    p.add_argument("--years", type=float, default=C.HISTORY_YEARS)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    update_cache(a.symbol, a.timeframe, a.years, force=a.force)
