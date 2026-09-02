"""Walk-forward validation.

One train/test split tells you how the model did in one regime. Walk-forward
retrains on a rolling window and tests on the next unseen block, so you get K
independent out-of-sample results instead of one. The question it answers is not
"is the backtest good" but "is it good in most periods, or did one 2024 melt-up
carry the whole curve".

Fold layout (anchored=False, the default rolling form):

    |<-- train -->|emb|<- val ->|emb|<- test ->|
                        |<-- train -->|emb|<- val ->|emb|<- test ->|

train_fn is injected so this module never imports TensorFlow. It receives a fold
dict and must return a policy callable `f(bar_index, state) -> 0|1|2`.
"""
from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np

from . import config as C
from . import backtest as bt
from .dataset import make_windows, soft_labels
from .features import realized_volatility


def make_folds(n_windows: int, n_folds: int = 6, test_frac: float = 0.12,
               val_frac: float = 0.10, embargo: int = C.EMBARGO_BARS,
               anchored: bool = False):
    """-> list of dicts with train/val/test slices over WINDOW indices."""
    test_len = int(n_windows * test_frac)
    val_len = int(n_windows * val_frac)
    folds = []
    # last fold's test block ends at the very last window; walk backwards
    for k in range(n_folds):
        test_hi = n_windows - k * test_len
        test_lo = test_hi - test_len
        val_hi = test_lo - embargo
        val_lo = val_hi - val_len
        train_hi = val_lo - embargo
        train_lo = 0 if anchored else max(0, train_hi - (n_windows - test_len * n_folds))
        if train_hi - train_lo < 2000:
            break
        folds.append({"train": slice(train_lo, train_hi),
                      "val": slice(val_lo, val_hi),
                      "test": slice(test_lo, test_hi)})
    return list(reversed(folds))


def run(candles: np.ndarray,
        train_fn: Callable[[dict], Callable],
        n_folds: int = 6, anchored: bool = False,
        cfg: bt.ExecConfig = bt.ExecConfig(),
        out_json: str | None = None, verbose: bool = True) -> dict:
    X_raw, anchors = make_windows(candles)
    y = soft_labels(candles[:, 4], anchors)
    vol = realized_volatility(candles[:, 4])[anchors]
    folds = make_folds(len(X_raw), n_folds=n_folds, anchored=anchored)

    results = []
    for fi, f in enumerate(folds, 1):
        tr, va, te = f["train"], f["val"], f["test"]
        # scaling fitted on this fold's TRAIN block only
        mu = X_raw[tr].reshape(-1, C.NUM_FEATURES).mean(axis=0)
        sd = X_raw[tr].reshape(-1, C.NUM_FEATURES).std(axis=0) + 1e-8
        scale = lambda s: ((X_raw[s] - mu) / sd).astype(np.float32)

        fold = {"index": fi, "candles": candles,
                "X_train": scale(tr), "y_train": y[tr],
                "X_val": scale(va), "y_val": y[va],
                "X_test": scale(te), "y_test": y[te],
                "anchors_train": anchors[tr], "anchors_val": anchors[va],
                "anchors_test": anchors[te],
                "vol_train": vol[tr], "vol_val": vol[va], "vol_test": vol[te],
                "feat_mean": mu, "feat_std": sd}

        if verbose:
            print(f"\n=== fold {fi}/{len(folds)}  "
                  f"train={X_raw[tr].shape[0]:,}  test={X_raw[te].shape[0]:,} "
                  f"bars {anchors[te][0]}..{anchors[te][-1]} ===")

        policy = train_fn(fold)

        lo, hi = int(anchors[te][0]), int(anchors[te][-1]) + 2
        # same volatility-scaled costs the promotion backtest and the PPO
        # reward use, so a fold's Sharpe is comparable to the gate's
        res = bt.run(candles[:hi], policy, start=lo, cfg=cfg,
                     vol=realized_volatility(candles[:hi, 4]))
        m = bt.metrics(res, cfg)
        m["fold"] = fi
        m["test_bars"] = [lo, hi]
        results.append(m)
        if verbose:
            print(bt.report(m, f"fold {fi} out-of-sample"))

    summary = aggregate(results)
    if verbose:
        print("\n" + summarize(summary, results))
    payload = {"folds": results, "summary": summary}
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump(payload, fh, indent=2, default=float)
    return payload


def aggregate(results: list[dict]) -> dict:
    g = lambda k: np.array([r[k] for r in results], dtype=float)
    if not results:
        return {}
    return {
        "n_folds": len(results),
        "total_trades": int(g("n_trades").sum()),
        "mean_sharpe": float(g("sharpe").mean()),
        "median_sharpe": float(np.median(g("sharpe"))),
        "worst_sharpe": float(g("sharpe").min()),
        "sharpe_std": float(g("sharpe").std()),
        "mean_expectancy_bps": float(g("expectancy_bps").mean()),
        "profitable_folds": int((g("total_return") > 0).sum()),
        "mean_win_rate": float(g("win_rate").mean()),
        "worst_drawdown": float(g("max_drawdown").min()),
        "beat_buy_hold_folds": int((g("total_return") > g("buy_hold_return")).sum()),
        # the gate: consistency, not the best fold
        "consistent_edge": bool(len(results) >= 4
                                and (g("total_return") > 0).mean() >= 0.75
                                and g("expectancy_bps").mean() > 0
                                and np.median(g("sharpe")) > 0.5),
    }


def summarize(s: dict, results: list[dict]) -> str:
    rows = [f"  fold {r['fold']}: ret={r['total_return']:+7.2%}  "
            f"sharpe={r['sharpe']:+5.2f}  trades={r['n_trades']:>5,}  "
            f"exp={r['expectancy_bps']:+6.2f}bps  dd={r['max_drawdown']:.2%}"
            for r in results]
    return "\n".join([
        "── walk-forward summary " + "─" * 42, *rows,
        f"  profitable folds  {s['profitable_folds']}/{s['n_folds']}   "
        f"beat buy&hold {s['beat_buy_hold_folds']}/{s['n_folds']}",
        f"  sharpe            mean {s['mean_sharpe']:+.2f}  "
        f"median {s['median_sharpe']:+.2f}  worst {s['worst_sharpe']:+.2f}  "
        f"sd {s['sharpe_std']:.2f}",
        f"  expectancy        {s['mean_expectancy_bps']:+.2f} bps/trade over "
        f"{s['total_trades']:,} trades",
        f"  verdict           {'CONSISTENT edge' if s['consistent_edge'] else 'NOT consistent — do not deploy'}",
    ])
