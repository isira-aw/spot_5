"""Turn the system's own history into a supervised learning problem.

Two label sources, in order of trustworthiness:

``trade``
    A round trip that actually happened. The feature snapshot taken at entry is
    the X; whether it closed green is the y. This is ground truth and it is what
    the model is judged on.

``shadow``
    Every cycle stores a feature snapshot and the price at that moment, so the
    forward return over the next few cycles answers the counterfactual "would a
    long opened here have paid?". It costs nothing, it exists from the first hour
    of running, and it is what makes the risk engine trainable before the first
    trade ever closes. Shadow rows are used only to top up a thin trade set, and
    the split between the two is reported in the model's metrics.

Both sources are ordered by time and split chronologically with an embargo, for
the same reason engine_2 does it: a random split lets the model read the future.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from core import repository
from core.config import get_settings

from .features import FEATURE_NAMES, vectorize

log = logging.getLogger("engine_3.dataset")

SHADOW_HORIZON_CYCLES = 4        # ~1 hour at a 15-minute cadence
SHADOW_COST_PCT = 0.15           # a move has to clear fees + slippage to count as a win
EMBARGO_ROWS = 2                 # dropped at the train/test seam


def _label_from_trade(trade: dict) -> int:
    return 1 if float(trade.get("pnl_quote") or 0.0) > 0 else 0


def rows_from_trades(mode: str, limit: int = 5000) -> list[dict]:
    out = []
    for t in repository.closed_trades(mode, limit=limit):
        ctx = t.get("context") or {}
        feats = ctx.get("features") if isinstance(ctx, dict) else None
        if not feats:
            continue
        out.append({"features": feats, "y": _label_from_trade(t),
                    "r": float(t.get("r_multiple") or 0.0),
                    "ts": t.get("exit_at") or t.get("entry_at"), "source": "trade"})
    out.sort(key=lambda r: r["ts"] or datetime.min)
    return out


def rows_from_shadow(mode: str, horizon: int = SHADOW_HORIZON_CYCLES,
                     limit: int = 5000) -> list[dict]:
    cycles = repository.risk_feature_rows(mode, limit=limit)
    out = []
    for i, row in enumerate(cycles):
        j = i + horizon
        if j >= len(cycles) or not row["features"] or row["price"] <= 0:
            continue
        fwd_pct = (cycles[j]["price"] / row["price"] - 1.0) * 100.0
        out.append({"features": row["features"], "y": 1 if fwd_pct > SHADOW_COST_PCT else 0,
                    "r": round(fwd_pct, 4), "ts": row["ts"], "source": "shadow"})
    return out


def build(mode: str, *, min_samples: int | None = None,
          allow_shadow: bool = True) -> dict[str, Any]:
    """-> {X_train, y_train, X_test, y_test, rows, counts, window, feature_names}."""
    min_samples = min_samples or get_settings().engines.engine_3_min_samples
    trades = rows_from_trades(mode)
    shadow = rows_from_shadow(mode) if allow_shadow and len(trades) < min_samples else []
    rows = sorted(trades + shadow, key=lambda r: r["ts"] or datetime.min)

    X = [vectorize(r["features"], FEATURE_NAMES) for r in rows]
    y = [int(r["y"]) for r in rows]
    n = len(rows)
    cut = int(n * 0.7)
    train_slice = slice(0, max(0, cut - EMBARGO_ROWS))
    test_slice = slice(cut, n)

    counts = {"total": n, "trades": len(trades), "shadow": len(shadow),
              "positives": sum(y), "negatives": n - sum(y)}
    window = (rows[0]["ts"] if rows else None, rows[-1]["ts"] if rows else None)
    log.info("engine_3 dataset for %s: %s", mode, counts)
    return {"X_train": X[train_slice], "y_train": y[train_slice],
            "X_test": X[test_slice], "y_test": y[test_slice],
            "rows": rows, "counts": counts, "window": window,
            "feature_names": list(FEATURE_NAMES)}


def is_trainable(data: dict, min_samples: int) -> tuple[bool, str]:
    """Refuse to train on a set that cannot produce an honest evaluation."""
    c = data["counts"]
    if c["total"] < min_samples:
        return False, (f"only {c['total']} labelled samples, need {min_samples} "
                       f"(trades={c['trades']}, shadow={c['shadow']})")
    if c["positives"] < 5 or c["negatives"] < 5:
        return False, (f"one-sided labels: {c['positives']} wins / {c['negatives']} losses — "
                       f"nothing to learn yet")
    if len(data["y_test"]) < 10:
        return False, f"holdout too small ({len(data['y_test'])} rows) to judge a model"
    if len(set(data["y_test"])) < 2:
        return False, "holdout has a single class, AUC would be meaningless"
    return True, "ok"
