"""Live market data with more than one way to get it.

PAPER mode is defined as *real* market data against simulated money, so the price
feed is not an optional component — a stale or missing price silently turns paper
results into fiction. Three public endpoints are tried in order; the first one
that answers wins, and every quote carries the venue it came from and its age.
"""
from __future__ import annotations

import logging
import math
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Any, Callable

import requests

from .config import REAL, get_settings

log = logging.getLogger("core.market")

_TIMEOUT = 12
_cache: dict[str, tuple[float, Any]] = {}
_lock = threading.Lock()


@dataclass
class Quote:
    symbol: str
    price: float
    source: str
    ts: float
    ok: bool = True
    error: str | None = None

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.ts)

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "price": self.price, "source": self.source,
                "age_s": round(self.age_s, 2), "ok": self.ok, "error": self.error}


def _cached(key: str, ttl: float, producer: Callable[[], Any]) -> Any:
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    value = producer()
    with _lock:
        _cache[key] = (time.time(), value)
    return value


# ── venue adapters ───────────────────────────────────────────────────────────
def _binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("-", "").upper()


def _kraken_pair(symbol: str) -> str:
    base, _, quote = symbol.partition("/")
    base = {"BTC": "XBT"}.get(base.upper(), base.upper())
    return f"{base}{quote.upper()}"


def _from_binance(symbol: str) -> float:
    r = requests.get("https://api.binance.com/api/v3/ticker/price",
                     params={"symbol": _binance_symbol(symbol)}, timeout=_TIMEOUT)
    r.raise_for_status()
    return float(r.json()["price"])


def _from_kraken(symbol: str) -> float:
    r = requests.get("https://api.kraken.com/0/public/Ticker",
                     params={"pair": _kraken_pair(symbol)}, timeout=_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    result = payload["result"]
    return float(next(iter(result.values()))["c"][0])


def _from_coinbase(symbol: str) -> float:
    base, _, quote = symbol.partition("/")
    quote = {"USDT": "USD"}.get(quote.upper(), quote.upper())
    r = requests.get(f"https://api.coinbase.com/v2/prices/{base.upper()}-{quote}/spot",
                     timeout=_TIMEOUT)
    r.raise_for_status()
    return float(r.json()["data"]["amount"])


VENUES: tuple[tuple[str, Callable[[str], float]], ...] = (
    ("binance", _from_binance), ("kraken", _from_kraken), ("coinbase", _from_coinbase))


# ── offline feed ─────────────────────────────────────────────────────────────
def _offline_allowed() -> bool:
    """``MARKET_OFFLINE=1`` runs the whole pipeline with no network — a synthetic
    but deterministic price walk. It is refused in REAL mode: simulated prices and
    real funds must never meet."""
    s = get_settings()
    if not s.execution.market_offline:
        return False
    if s.execution.mode == REAL:
        log.error("MARKET_OFFLINE is set but the mode is REAL — ignoring it. "
                  "Real funds are never traded against synthetic prices.")
        return False
    return True


def _synthetic_price(symbol: str, ts: float | None = None) -> float:
    """A smooth, seeded walk so repeated runs are comparable but not constant."""
    ts = time.time() if ts is None else ts
    base = 20_000.0 + (zlib.crc32(symbol.encode()) % 60_000)
    minutes = ts / 60.0
    wave = (math.sin(minutes / 97.0) * 0.03 + math.sin(minutes / 13.0) * 0.008
            + math.sin(minutes / 3.1) * 0.002)
    return round(base * (1.0 + wave), 2)


def get_quote(symbol: str | None = None, ttl: float = 5.0) -> Quote:
    symbol = symbol or get_settings().execution.symbol
    if _offline_allowed():
        return Quote(symbol=symbol, price=_synthetic_price(symbol), source="synthetic",
                     ts=time.time())

    def _fetch() -> Quote:
        errors = []
        for name, fn in VENUES:
            try:
                price = fn(symbol)
                if price > 0:
                    return Quote(symbol=symbol, price=price, source=name, ts=time.time())
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}")
        return Quote(symbol=symbol, price=0.0, source="none", ts=time.time(), ok=False,
                     error="; ".join(errors) or "no venue answered")

    quote: Quote = _cached(f"quote:{symbol}", ttl, _fetch)
    if not quote.ok:
        stale = _cache.get(f"last_good:{symbol}")
        if stale:
            last: Quote = stale[1]
            if last.age_s < 900:
                log.warning("price feed down; reusing %.2f from %s (%.0fs old)",
                            last.price, last.source, last.age_s)
                return Quote(symbol=symbol, price=last.price, source=f"{last.source}:stale",
                             ts=last.ts, ok=True, error=quote.error)
    else:
        with _lock:
            _cache[f"last_good:{symbol}"] = (time.time(), quote)
    return quote


def get_price(symbol: str | None = None) -> float:
    return get_quote(symbol).price


# ── candles (used by engine_3's regime features) ─────────────────────────────
def get_ohlcv(symbol: str | None = None, interval: str = "1h", limit: int = 200,
              ttl: float = 60.0) -> list[list[float]]:
    """-> [[ts_ms, open, high, low, close, volume], ...] oldest first."""
    symbol = symbol or get_settings().execution.symbol
    if _offline_allowed():
        step = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                "1h": 3600, "4h": 14400, "1d": 86400}.get(interval, 3600)
        now = time.time()
        out = []
        for i in range(limit, 0, -1):
            ts = now - i * step
            c = _synthetic_price(symbol, ts)
            o = _synthetic_price(symbol, ts - step)
            out.append([ts * 1000, o, max(o, c) * 1.001, min(o, c) * 0.999, c, 100.0])
        return out

    def _fetch() -> list[list[float]]:
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params={"symbol": _binance_symbol(symbol),
                                     "interval": interval, "limit": min(limit, 1000)},
                             timeout=_TIMEOUT)
            r.raise_for_status()
            return [[float(k[0]), float(k[1]), float(k[2]), float(k[3]),
                     float(k[4]), float(k[5])] for k in r.json()]
        except Exception as exc:
            log.warning("binance klines failed (%s); trying kraken", type(exc).__name__)
        try:
            minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
                       "4h": 240, "1d": 1440}.get(interval, 60)
            r = requests.get("https://api.kraken.com/0/public/OHLC",
                             params={"pair": _kraken_pair(symbol), "interval": minutes},
                             timeout=_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            series = next(v for k, v in payload["result"].items() if k != "last")
            return [[float(c[0]) * 1000, float(c[1]), float(c[2]), float(c[3]),
                     float(c[4]), float(c[6])] for c in series][-limit:]
        except Exception as exc:
            log.error("no candle source available: %s", exc)
            return []

    return _cached(f"ohlcv:{symbol}:{interval}:{limit}", ttl, _fetch)


def clear_cache() -> None:
    with _lock:
        _cache.clear()
