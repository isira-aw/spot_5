#!/usr/bin/env python3
"""
BTC/USDT multi-horizon spot-trading analyser.

Pipeline
--------
1. Market data  : Kraken public OHLC (no key, no geo-block).
2. Derived TFs  : 15m / 30m / 1h / 2h / 4h / 12h / 24h / 2d / 7d / 14d / 30d
                  (resampled from 3 native Kraken series).
3. Indicators   : EMA, RSI, ATR, structure, volume -> deterministic baseline score.
4. News         : Alpha Vantage (optional, degrades silently).
5. LLM layer    : Groq (OpenAI-compatible API) -> falls back to local Ollama
                  (qwen2.5:0.5b) -> falls back to the deterministic baseline.
6. Output       : per-horizon bias + spot-trade LABEL + entry/stop/targets,
                  printed as a table and written to JSON.

The LLM can only *adjust* the numeric baseline. If every model is down, the
script still produces a full result set. That is what makes it stable.

NOT financial advice. Educational tooling only.

Usage
-----
    pip install -r requirements.txt
    cp .env.example .env      # then edit
    python btc_multi_horizon.py
    python btc_multi_horizon.py --self-test     # offline, synthetic data
    python btc_multi_horizon.py --no-news --no-llm
    python btc_multi_horizon.py --pair XBTUSD --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


# ======================================================================
# Config
# ======================================================================

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"
ALPHA_URL = "https://www.alphavantage.co/query"

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "25"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))

# Models tried on Groq if GROQ_MODEL is not served by the account.
GROQ_FALLBACK_MODELS = [
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]

# Native Kraken intervals we actually download (minutes).
# 720 candles each: 15m -> 7.5d, 4h -> 120d, 1d -> ~2y.
NATIVE_INTERVALS = {"m15": 15, "h4": 240, "d1": 1440}


@dataclass(frozen=True)
class Horizon:
    name: str          # label shown to the user
    rule: str          # pandas resample rule (minute-based = version safe)
    source: str        # which native series to resample from
    minutes: int       # horizon length in minutes


HORIZONS: list[Horizon] = [
    Horizon("15m", "15min", "m15", 15),
    Horizon("30m", "30min", "m15", 30),
    Horizon("1h", "60min", "m15", 60),
    Horizon("2h", "120min", "m15", 120),
    Horizon("4h", "240min", "h4", 240),
    Horizon("12h", "720min", "h4", 720),
    Horizon("24h", "1440min", "d1", 1440),
    Horizon("2d", "2880min", "d1", 2880),
    Horizon("7d", "10080min", "d1", 10080),
    Horizon("14d", "20160min", "d1", 20160),
    Horizon("30d", "43200min", "d1", 43200),
]

# Spot-only label vocabulary (no shorting).
LABELS = {
    "STRONG_BUY": "open / add spot position aggressively",
    "BUY": "scale into spot on dips",
    "ACCUMULATE": "small starter position only",
    "HOLD": "keep existing bags, no new entry",
    "WAIT": "no edge, stay in cash",
    "REDUCE": "trim into strength / take partials",
    "EXIT": "close spot exposure, stand aside",
}


# ======================================================================
# Small utilities
# ======================================================================


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def http_get(url: str, params: dict, tries: int = 3) -> dict:
    """GET with linear backoff. Raises RuntimeError after the last try."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < tries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"GET {url} failed after {tries} tries: {last}")


# ======================================================================
# 1. Market data
# ======================================================================


def fetch_kraken_ohlc(pair: str, interval: int) -> pd.DataFrame:
    data = http_get(KRAKEN_URL, {"pair": pair, "interval": interval})
    if data.get("error"):
        raise RuntimeError(f"Kraken error for {pair}@{interval}m: {data['error']}")
    result = data.get("result") or {}
    keys = [k for k in result if k != "last"]
    if not keys:
        raise RuntimeError(f"Kraken returned no candles for {pair}@{interval}m")
    cols = ["open_time", "open", "high", "low", "close", "vwap", "volume", "trades"]
    df = pd.DataFrame(result[keys[0]], columns=cols)
    for c in ["open", "high", "low", "close", "vwap", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="s", utc=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df.set_index("open_time").sort_index()


def synthetic_ohlc(interval: int, n: int = 700, seed: int = 7) -> pd.DataFrame:
    """Offline random walk, used by --self-test so the pipeline is testable."""
    rng = np.random.default_rng(seed + interval)
    step = np.sqrt(interval / 1440.0) * 0.02
    ret = rng.normal(0.0002, step, n)
    close = 65000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, step / 2, n)))
    low = close * (1 - np.abs(rng.normal(0, step / 2, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    end = pd.Timestamp.now(tz="UTC").floor("min")
    idx = pd.date_range(end=end, periods=n, freq=f"{interval}min")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "vwap": close,
            "volume": np.abs(rng.normal(120, 40, n)),
            "trades": rng.integers(50, 900, n),
        },
        index=idx,
    )


def load_native_series(pair: str, offline: bool) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, interval in NATIVE_INTERVALS.items():
        if offline:
            out[key] = synthetic_ohlc(interval)
            continue
        try:
            out[key] = fetch_kraken_ohlc(pair, interval)
            log(f"Kraken {interval:>4}m : {len(out[key])} candles")
        except Exception as exc:  # noqa: BLE001
            log(f"Kraken {interval}m FAILED: {exc}")
    if not out:
        raise RuntimeError("No market data available from Kraken.")
    return out


def resample_ohlc(df: pd.DataFrame, rule: str, base_minutes: int) -> pd.DataFrame:
    """Aggregate to `rule`, epoch-anchored, with the in-progress bar removed."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "trades": "sum",
    }
    out = (
        df.resample(rule, origin="epoch", label="left", closed="left")
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
    )
    if out.empty:
        return out
    # Drop the final bar if it is not yet closed.
    data_end = df.index[-1] + pd.Timedelta(minutes=base_minutes)
    if out.index[-1] + pd.Timedelta(rule) > data_end and len(out) > 30:
        out = out.iloc[:-1]
    return out


# ======================================================================
# 2. Indicators
# ======================================================================


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=min(n, len(s))).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    ag = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def build_features(df: pd.DataFrame) -> dict[str, float]:
    n = len(df)
    close = df["close"]
    price = safe_float(close.iloc[-1])

    e_fast = safe_float(ema(close, min(20, max(3, n // 3))).iloc[-1], price)
    e_slow = safe_float(ema(close, min(50, max(5, n // 2))).iloc[-1], price)
    r = safe_float(rsi(close).iloc[-1], 50.0)
    a = safe_float(atr(df).iloc[-1], price * 0.01) or price * 0.01

    def ret(k: int) -> float:
        if n <= k:
            return 0.0
        p = safe_float(close.iloc[-1 - k], price)
        return 0.0 if p == 0 else (price / p - 1) * 100

    look = min(20, n)
    hi = safe_float(df["high"].iloc[-look:].max(), price)
    lo = safe_float(df["low"].iloc[-look:].min(), price)
    rng = hi - lo
    pos = 50.0 if rng <= 0 else clamp((price - lo) / rng * 100, 0, 100)

    v = df["volume"]
    v_recent = safe_float(v.iloc[-min(5, n):].mean())
    v_base = safe_float(v.iloc[-min(50, n):].mean())
    v_ratio = 1.0 if v_base <= 0 else clamp(v_recent / v_base, 0, 5)

    return {
        "price": round(price, 2),
        "ema_fast": round(e_fast, 2),
        "ema_slow": round(e_slow, 2),
        "rsi14": round(r, 1),
        "atr": round(a, 2),
        "atr_pct": round(a / price * 100, 2) if price else 0.0,
        "ret_1": round(ret(1), 2),
        "ret_3": round(ret(3), 2),
        "ret_10": round(ret(10), 2),
        "dist_ema_fast_atr": round((price - e_fast) / a, 2) if a else 0.0,
        "range_pos_pct": round(pos, 1),
        "range_high": round(hi, 2),
        "range_low": round(lo, 2),
        "vol_ratio": round(v_ratio, 2),
        "bars": n,
    }


# ======================================================================
# 3. Deterministic baseline
# ======================================================================


def baseline_score(f: dict[str, float]) -> float:
    """Weighted score in roughly [-1, 1]. Positive = bullish."""
    s = 0.0
    s += 0.30 * (1.0 if f["ema_fast"] > f["ema_slow"] else -1.0)
    s += 0.20 * clamp((f["rsi14"] - 50) / 25, -1, 1)
    s += 0.20 * clamp(f["dist_ema_fast_atr"] / 1.5, -1, 1)
    s += 0.15 * clamp(f["ret_3"] / max(f["atr_pct"] * 1.5, 0.3), -1, 1)
    s += 0.10 * clamp((f["range_pos_pct"] - 50) / 40, -1, 1)
    s += 0.05 * clamp((f["vol_ratio"] - 1) / 1.5, -1, 1)
    # Fade blow-off extremes rather than chase them.
    if f["rsi14"] > 78:
        s -= 0.18
    if f["rsi14"] < 22:
        s += 0.18
    return clamp(s, -1, 1)


def score_to_bias(score: float) -> str:
    if score >= 0.18:
        return "UP"
    if score <= -0.18:
        return "DOWN"
    return "NEUTRAL"


def baseline_confidence(score: float, f: dict[str, float], minutes: int) -> int:
    conf = 30 + abs(score) * 55
    if f["bars"] < 60:
        conf -= 8
    if f["atr_pct"] > 4:            # very noisy regime
        conf -= 6
    if minutes <= 60:               # short TFs are mostly noise
        conf -= 6
    if minutes >= 20160:            # very long TFs are macro-driven
        conf -= 4
    return int(clamp(round(conf), 5, 88))


def make_label(bias: str, conf: int, f: dict[str, float]) -> str:
    """Spot-only label. Short side means 'do not hold', never 'go short'."""
    overbought = f["rsi14"] > 75
    oversold = f["rsi14"] < 28
    if bias == "UP":
        if conf >= 70 and not overbought:
            return "STRONG_BUY"
        if conf >= 55:
            return "BUY"
        return "ACCUMULATE"
    if bias == "DOWN":
        if conf >= 70:
            return "EXIT"
        if conf >= 55:
            return "REDUCE"
        return "HOLD"
    if oversold:
        return "ACCUMULATE"
    if overbought:
        return "REDUCE"
    return "WAIT" if conf < 45 else "HOLD"


def trade_plan(f: dict[str, float], label: str, minutes: int) -> dict[str, Any]:
    """ATR-derived levels. Deterministic — the LLM never sets your stop."""
    price, a = f["price"], max(f["atr"], f["price"] * 0.001)
    # Wider structure on higher timeframes.
    k = 1.5 if minutes <= 240 else (2.0 if minutes <= 1440 else 2.5)
    buying = label in ("STRONG_BUY", "BUY", "ACCUMULATE")

    entry_lo, entry_hi = price - 0.35 * a, price + 0.15 * a
    stop = price - k * a
    tp1, tp2 = price + 1.5 * k * a, price + 3.0 * k * a
    rr = round((tp1 - price) / max(price - stop, 1e-9), 2)

    plan = {
        "entry_zone": [round(entry_lo, 2), round(entry_hi, 2)] if buying else None,
        "stop_loss": round(stop, 2) if buying else None,
        "take_profit_1": round(tp1, 2) if buying else None,
        "take_profit_2": round(tp2, 2) if buying else None,
        "risk_reward": rr if buying else None,
        "invalidation": round(min(stop, f["range_low"]), 2),
        "atr_used": round(a, 2),
    }
    if not buying:
        plan["note"] = "No spot entry for this horizon; manage or avoid exposure."
    return plan


def analyse_horizon(h: Horizon, df: pd.DataFrame) -> dict[str, Any]:
    f = build_features(df)
    score = baseline_score(f)
    bias = score_to_bias(score)
    conf = baseline_confidence(score, f, h.minutes)
    label = make_label(bias, conf, f)
    return {
        "horizon": h.name,
        "minutes": h.minutes,
        "features": f,
        "baseline": {
            "score": round(score, 3),
            "bias": bias,
            "confidence": conf,
            "label": label,
        },
        "final": {"bias": bias, "confidence": conf, "label": label, "source": "baseline"},
        "reasons": baseline_reasons(f, bias),
        "plan": trade_plan(f, label, h.minutes),
    }


def baseline_reasons(f: dict[str, float], bias: str) -> list[str]:
    out = []
    trend = "above" if f["ema_fast"] > f["ema_slow"] else "below"
    out.append(f"Fast EMA is {trend} slow EMA ({f['ema_fast']} vs {f['ema_slow']}).")
    out.append(f"RSI(14) at {f['rsi14']}, price sits at {f['range_pos_pct']}% of the recent range.")
    out.append(
        f"3-bar move {f['ret_3']:+.2f}% against ATR of {f['atr_pct']:.2f}%; "
        f"volume {f['vol_ratio']}x its average."
    )
    if bias == "NEUTRAL":
        out.append("Signals conflict, so no directional edge is assumed.")
    return out


# ======================================================================
# 4. News (optional)
# ======================================================================


def fetch_news(kind: str, limit: int = 15) -> list[dict]:
    if not ALPHA_VANTAGE_API_KEY:
        return []
    params = {"function": "NEWS_SENTIMENT", "apikey": ALPHA_VANTAGE_API_KEY, "limit": limit}
    if kind == "crypto":
        params["tickers"] = "CRYPTO:BTC"
    else:
        params["topics"] = "economy_macro,economy_monetary,financial_markets"
    try:
        data = http_get(ALPHA_URL, params, tries=2)
    except Exception as exc:  # noqa: BLE001
        log(f"News ({kind}) unavailable: {exc}")
        return []
    if "feed" not in data:
        msg = data.get("Information") or data.get("Note") or data.get("Error Message") or data
        log(f"News ({kind}) skipped: {str(msg)[:140]}")
        return []
    feed = data["feed"][:limit]
    return [
        {
            "title": a.get("title", "")[:200],
            "source": a.get("source", ""),
            "time": a.get("time_published", ""),
            "sentiment": a.get("overall_sentiment_label", ""),
            "summary": (a.get("summary") or "")[:320],
        }
        for a in feed
    ]


# ======================================================================
# 5. LLM layer
# ======================================================================


def extract_json(text: str) -> dict | None:
    """Tolerant JSON extraction — small local models wrap output in junk."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    candidates = []
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : i + 1])
    for chunk in sorted(candidates, key=len, reverse=True):
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def resolve_groq_model(preferred: str) -> str | None:
    """Return a model id the account can actually call, or None if Groq is down."""
    try:
        r = requests.get(
            f"{GROQ_BASE_URL}/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        ids = {m.get("id") for m in r.json().get("data", [])}
    except Exception as exc:  # noqa: BLE001
        log(f"Groq model list unavailable ({exc}); trying '{preferred}' blind.")
        return preferred
    if preferred in ids:
        return preferred
    for cand in GROQ_FALLBACK_MODELS:
        if cand in ids:
            log(f"Groq model '{preferred}' not served; using '{cand}'.")
            return cand
    log(f"Groq reachable but no known model available (saw {len(ids)}).")
    return None


def call_groq(prompt: str, system: str) -> tuple[str | None, str]:
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY not set"
    model = resolve_groq_model(GROQ_MODEL)
    if not model:
        return None, "no usable Groq model"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2400,
        "response_format": {"type": "json_object"},
    }
    for attempt in (1, 2):
        try:
            r = requests.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=LLM_TIMEOUT,
            )
            if r.status_code == 400 and "response_format" in r.text:
                body.pop("response_format", None)  # model lacks JSON mode
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"], f"groq:{model}"
        except Exception as exc:  # noqa: BLE001
            log(f"Groq attempt {attempt} failed: {str(exc)[:160]}")
            time.sleep(2)
    return None, "groq failed"


def call_ollama(prompt: str, system: str) -> tuple[str | None, str]:
    try:
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=8).json()
        have = {m.get("name", "").split(":")[0] for m in tags.get("models", [])}
        if OLLAMA_MODEL.split(":")[0] not in have:
            log(f"Ollama is up but '{OLLAMA_MODEL}' is missing. Run: ollama pull {OLLAMA_MODEL}")
            return None, "ollama model missing"
    except Exception as exc:  # noqa: BLE001
        return None, f"ollama unreachable ({str(exc)[:80]})"
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": 1600},
            },
            timeout=LLM_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content"), f"ollama:{OLLAMA_MODEL}"
    except Exception as exc:  # noqa: BLE001
        log(f"Ollama call failed: {str(exc)[:160]}")
        return None, "ollama failed"


SYSTEM_PROMPT = (
    "You are a market-analysis component inside a trading research tool. "
    "You never invent news or price levels. You never claim certainty about the future. "
    "You output strictly valid JSON and nothing else. No prose, no markdown."
)


def build_prompt(results: list[dict], crypto_news: list[dict], macro_news: list[dict]) -> str:
    keep = ("price", "rsi14", "ema_fast", "ema_slow", "atr_pct",
            "ret_3", "ret_10", "range_pos_pct", "vol_ratio")
    compact = [
        {
            "horizon": r["horizon"],
            **{k: r["features"].get(k) for k in keep},
            "baseline_bias": r["baseline"]["bias"],
            "baseline_confidence": r["baseline"]["confidence"],
        }
        for r in results
    ]
    news_block = json.dumps(
        {"btc_news": crypto_news[:8], "macro_news": macro_news[:8]}, ensure_ascii=False
    )
    horizons = ", ".join(r["horizon"] for r in results)
    return f"""Analyse BTC/USDT for spot trading across these horizons: {horizons}.

TECHNICAL SNAPSHOT (computed, trustworthy):
{json.dumps(compact, ensure_ascii=False)}

RECENT NEWS (may be empty; do not use anything not listed here):
{news_block}

Rules:
- Start from baseline_bias/baseline_confidence and only deviate when the news or
  the cross-horizon picture justifies it. Do not flip a bias without a reason.
- confidence is 0-100 and must stay below 90.
- label must be exactly one of: STRONG_BUY, BUY, ACCUMULATE, HOLD, WAIT, REDUCE, EXIT.
  These are SPOT labels: "DOWN" means reduce or stay out, never short.
- Give 2-3 short, concrete reasons per horizon.

Return ONLY this JSON:
{{
  "live_investigation": true,
  "overall_bias": "UP|DOWN|NEUTRAL",
  "market_summary": "two sentences max",
  "key_risk": "one sentence",
  "horizons": [
    {{"horizon": "15m", "bias": "UP|DOWN|NEUTRAL", "confidence": 0,
      "label": "WAIT", "reasons": ["...", "..."]}}
  ]
}}"""


VALID_BIAS = {"UP", "DOWN", "NEUTRAL"}


def merge_llm(results: list[dict], parsed: dict, source: str) -> dict:
    """Apply LLM output per horizon. Anything malformed keeps the baseline."""
    by_name = {r["horizon"]: r for r in results}
    applied = 0
    for item in parsed.get("horizons") or []:
        if not isinstance(item, dict):
            continue
        r = by_name.get(str(item.get("horizon", "")).strip())
        if r is None:
            continue
        bias = str(item.get("bias", "")).strip().upper()
        label = str(item.get("label", "")).strip().upper().replace(" ", "_")
        if bias not in VALID_BIAS or label not in LABELS:
            continue
        conf = int(clamp(safe_float(item.get("confidence"), r["baseline"]["confidence"]), 0, 89))
        # Blend with the baseline so one bad LLM call cannot dominate.
        blended = int(round(0.5 * conf + 0.5 * r["baseline"]["confidence"]))
        r["final"] = {
            "bias": bias,
            "confidence": blended,
            "label": label,
            "source": source,
        }
        reasons = [str(x)[:220] for x in (item.get("reasons") or []) if str(x).strip()]
        if reasons:
            r["reasons"] = reasons[:3]
        r["plan"] = trade_plan(r["features"], label, r["minutes"])
        applied += 1
    log(f"LLM adjusted {applied}/{len(results)} horizons ({source}).")
    return {
        "overall_bias": str(parsed.get("overall_bias", "")).upper()
        if str(parsed.get("overall_bias", "")).upper() in VALID_BIAS
        else None,
        "market_summary": str(parsed.get("market_summary", ""))[:600] or None,
        "key_risk": str(parsed.get("key_risk", ""))[:400] or None,
        "applied": applied,
        "source": source,
    }


def run_llm(results: list[dict], crypto_news: list[dict], macro_news: list[dict]) -> dict:
    prompt = build_prompt(results, crypto_news, macro_news)
    for caller, name in ((call_groq, "Groq"), (call_ollama, "Ollama")):
        log(f"Calling {name}...")
        text, source = caller(prompt, SYSTEM_PROMPT)
        if not text:
            log(f"{name} unavailable: {source}")
            continue
        parsed = extract_json(text)
        if not parsed:
            log(f"{name} returned unparseable output; falling through.")
            continue
        return merge_llm(results, parsed, source)
    log("No LLM available — using the deterministic baseline for every horizon.")
    return {"overall_bias": None, "market_summary": None, "key_risk": None,
            "applied": 0, "source": "baseline"}


# ======================================================================
# 6. Reporting
# ======================================================================


def build_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        p, f, fin = r["plan"], r["features"], r["final"]
        rows.append(
            {
                "horizon": r["horizon"],
                "bias": fin["bias"],
                "conf": fin["confidence"],
                "label": fin["label"],
                "price": f["price"],
                "rsi": f["rsi14"],
                "atr%": f["atr_pct"],
                "entry": f"{p['entry_zone'][0]:.0f}-{p['entry_zone'][1]:.0f}" if p["entry_zone"] else "-",
                "stop": f"{p['stop_loss']:.0f}" if p["stop_loss"] else "-",
                "tp1": f"{p['take_profit_1']:.0f}" if p["take_profit_1"] else "-",
                "tp2": f"{p['take_profit_2']:.0f}" if p["take_profit_2"] else "-",
                "r:r": p["risk_reward"] or "-",
            }
        )
    return pd.DataFrame(rows)


def consensus(results: list[dict]) -> dict:
    w = {"UP": 0.0, "DOWN": 0.0, "NEUTRAL": 0.0}
    for r in results:
        w[r["final"]["bias"]] += r["final"]["confidence"] / 100.0
    total = sum(w.values()) or 1.0
    bias = max(w, key=w.get)
    return {
        "bias": bias,
        "agreement_pct": round(w[bias] / total * 100, 1),
        "weights": {k: round(v, 2) for k, v in w.items()},
    }


def print_report(results: list[dict], meta: dict, llm_meta: dict) -> None:
    table = build_table(results)
    print("\n" + "=" * 100)
    print(f"BTC MULTI-HORIZON SPOT VIEW  |  {meta['pair']}  |  {meta['generated_at']}")
    print(f"Spot price: {meta['price']}   |   analysis source: {llm_meta['source']}")
    print("=" * 100)
    print(table.to_string(index=False))

    c = meta["consensus"]
    print(f"\nCross-horizon consensus: {c['bias']} ({c['agreement_pct']}% of weighted signal)")
    if llm_meta.get("market_summary"):
        print(f"Summary: {llm_meta['market_summary']}")
    if llm_meta.get("key_risk"):
        print(f"Key risk: {llm_meta['key_risk']}")

    print("\nReasoning per horizon")
    print("-" * 100)
    for r in results:
        print(f"{r['horizon']:>4}  {r['final']['label']:<12} ({r['final']['bias']}, {r['final']['confidence']}%)")
        for reason in r["reasons"]:
            print(f"      - {reason}")
    print("\nLabel key")
    print("-" * 100)
    used = {r["final"]["label"] for r in results}
    for k in LABELS:
        if k in used:
            print(f"  {k:<12} {LABELS[k]}")
    print("\nEducational output, not financial advice. Levels are ATR-derived, not predictions.\n")


# ======================================================================
# Main
# ======================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description="BTC multi-horizon spot analyser")
    ap.add_argument("--pair", default=os.getenv("PAIR", "XBTUSDT"))
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="offline synthetic data, no network")
    ap.add_argument("--json", default="btc_multi_horizon.json")
    args = ap.parse_args()

    offline = args.self_test
    log(f"Loading market data ({'synthetic' if offline else args.pair})...")
    native = load_native_series(args.pair, offline)

    results: list[dict] = []
    for h in HORIZONS:
        source_key = h.source if h.source in native else next(iter(native))
        src = native[source_key]
        res = resample_ohlc(src, h.rule, NATIVE_INTERVALS[source_key])
        if len(res) < 20:
            log(f"Skipping {h.name}: only {len(res)} bars available.")
            continue
        results.append(analyse_horizon(h, res))

    if not results:
        log("No horizon had enough data. Aborting.")
        return 1

    crypto_news: list[dict] = []
    macro_news: list[dict] = []
    if not (args.no_news or offline):
        crypto_news = fetch_news("crypto")
        macro_news = fetch_news("macro")
        log(f"News: {len(crypto_news)} BTC, {len(macro_news)} macro headlines.")

    llm_meta = {"source": "baseline", "applied": 0, "market_summary": None, "key_risk": None}
    if not (args.no_llm or offline):
        llm_meta = run_llm(results, crypto_news, macro_news)

    # Spot price comes from the finest series available, not a resampled bar.
    finest = min(native, key=lambda k: NATIVE_INTERVALS[k])
    live_price = round(float(native[finest]["close"].iloc[-1]), 2)
    meta = {
        "pair": args.pair if not offline else "SYNTHETIC",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "price": live_price,
        "consensus": consensus(results),
    }
    print_report(results, meta, llm_meta)

    payload = {
        "meta": meta,
        "llm": llm_meta,
        "news_counts": {"crypto": len(crypto_news), "macro": len(macro_news)},
        "horizons": results,
        "label_key": LABELS,
        "disclaimer": "Educational analysis. Not financial advice.",
    }
    try:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        log(f"Saved {args.json}")
    except OSError as exc:
        log(f"Could not write JSON: {exc}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        log(f"Fatal: {exc}")
        sys.exit(1)
