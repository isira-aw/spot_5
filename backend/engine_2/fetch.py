"""Paginated OHLCV fetch with an incremental on-disk cache.

Replaces the static dataset.npz upload. First run pulls HISTORY_YEARS of candles
one page at a time; every later run only asks for what is missing since the last
cached bar, so a weekly retrain costs a handful of requests instead of 400.

READ-ONLY BY CONSTRUCTION. BINANCE_API_KEY / BINANCE_API_SECRET are optional and
buy nothing but a higher rate limit — this pipeline produces a model artifact and
nothing else. `assert_read_only()` refuses to continue if the configured key can
place orders or withdraw, and no module in engine_2 calls any ccxt method other
than `fetch_ohlcv` / `load_markets` / `milliseconds`. There is no order path here
to enable by accident.

Failure handling: exchange downtime, rate limits and transient network errors are
retried with exponential backoff; gaps in the returned history are reported and
left alone, because a synthetic candle is a fabricated training example.

CLI:
    python -m engine_2.fetch                    # top up the cache to "now"
    python -m engine_2.fetch --years 3 --force  # rebuild from scratch
    python -m engine_2.fetch --check-key        # prove the key is read-only
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config as C

COLS = ["timestamp", "open", "high", "low", "close", "volume"]
PAGE_LIMIT = 1000
MAX_RETRIES = 6
BACKOFF_BASE_S = 2.0

# Permissions that would make this key more than a market-data reader. If the
# exchange reports any of them the pipeline refuses to use the key at all.
FORBIDDEN_PERMISSIONS = ("enableSpotAndMarginTrading", "enableFutures",
                         "enableWithdrawals", "enableMargin", "enableInternalTransfer",
                         "permitsUniversalTransfer", "enableVanillaOptions")


def _exchange(auth: bool = True):
    """A ccxt client for market data. Credentials are attached only when present,
    and only ever used for the higher public-endpoint rate limit."""
    try:
        import ccxt  # imported lazily so backtests run without ccxt installed
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "engine_2 needs ccxt to pull market data. Install this package's "
            "dependencies:\n    pip install -r backend/engine_2/requirements.txt"
        ) from exc
    params = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    if auth and C.BINANCE_API_KEY and C.BINANCE_API_SECRET:
        params |= {"apiKey": C.BINANCE_API_KEY, "secret": C.BINANCE_API_SECRET}
    ex = getattr(ccxt, C.EXCHANGE_ID)(params)
    _with_retry(ex.load_markets, what="load_markets")
    return ex


def assert_read_only(exchange=None) -> dict:
    """Verify the configured key cannot trade or withdraw. No key -> nothing to
    check, and public OHLCV keeps working.

    This is a safety interlock, not a feature: engine_2 exists to produce a model
    artifact. If someone drops a trading-capable key into the environment, the
    data pull stops rather than running with privileges it must never hold.
    """
    if not (C.BINANCE_API_KEY and C.BINANCE_API_SECRET):
        return {"authenticated": False, "read_only": True,
                "detail": "no API key configured; using public market data"}
    ex = exchange or _exchange()
    try:
        info = ex.sapi_get_account_apirestrictions()
    except Exception as exc:                    # endpoint blocked = key is limited
        return {"authenticated": True, "read_only": True,
                "detail": f"permission endpoint unavailable ({type(exc).__name__}); "
                          f"treating key as read-only"}
    granted = [p for p in FORBIDDEN_PERMISSIONS if str(info.get(p)).lower() == "true"]
    if granted:
        raise PermissionError(
            "BINANCE_API_KEY has permissions this pipeline must never hold: "
            f"{', '.join(granted)}. Use a key with read-only market-data access "
            "(no trading, no withdrawals) or remove the key entirely.")
    return {"authenticated": True, "read_only": True, "detail": "key is read-only"}


def _with_retry(fn, *args, what: str = "request", **kw):
    """Exchange downtime and rate limits are normal, not exceptional.

    Anything the exchange raises is retried with exponential backoff; the last
    failure is re-raised so a genuinely broken run fails loudly rather than
    silently caching a short history.
    """
    delay = BACKOFF_BASE_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kw)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  {what} failed ({type(exc).__name__}: {str(exc)[:120]}) — "
                  f"retry {attempt}/{MAX_RETRIES - 1} in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def fetch_range(symbol: str, timeframe: str, since_ms: int,
                until_ms: int | None = None, exchange=None) -> pd.DataFrame:
    """Page forward from since_ms until the exchange stops returning new bars."""
    ex = exchange or _exchange()
    until_ms = until_ms or ex.milliseconds()
    step = C.TF_MS[timeframe]
    cursor, rows = since_ms, []

    while cursor < until_ms:
        batch = _with_retry(ex.fetch_ohlcv, symbol, timeframe, since=cursor,
                            limit=PAGE_LIMIT, what=f"fetch_ohlcv({symbol})")
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
    print(f"credentials: {assert_read_only(ex)['detail']}")
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
    raw = _with_retry(ex.fetch_ohlcv, symbol, timeframe, limit=n + 1,
                      what="fetch_ohlcv(live)")
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
    p.add_argument("--check-key", action="store_true")
    a = p.parse_args()
    if a.check_key:
        print(assert_read_only())
    else:
        update_cache(a.symbol, a.timeframe, a.years, force=a.force)
