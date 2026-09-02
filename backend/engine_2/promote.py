"""Score a candidate on unseen bars and decide whether it replaces what is live.

The retrain job must never overwrite a working model just because it finished.
This is the gate: candidate and incumbent are scored on the SAME unseen bars,
against the same baselines, and the candidate ships only if it clears absolute
floors, beats a random trader, survives the never-tuned-on holdout, and beats the
incumbent by a margin wider than fold-to-fold noise.

Two slices matter and they are not the same thing:

  test     the model selection slice. The gate reads it, so by the time a model
           has been promoted, decisions have been made on these numbers.
  holdout  read exactly once, by `final_backtest`, after everything else is
           decided. Nothing was ever selected on it, which is the only reason its
           Sharpe / max drawdown / win rate mean anything.

Promotion goes through registry.py, so `models/` is always a copy of an immutable
version directory and rollback is one command.

    python -m engine_2.promote --evaluate models_candidate
    python -m engine_2.promote --gate      # compare vs live, maybe promote
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import backtest as bt
from . import config as C
from . import gates
from . import registry
from .dataset import load
from .features import realized_volatility

# absolute floors — a candidate below any of these does not ship even if the
# incumbent is worse
FLOORS = {"n_trades": 50, "sharpe": 0.5, "expectancy_bps": 0.0,
          "max_drawdown": -0.25, "profit_factor": 1.05}
# the holdout is noisier (it is smaller), so it gets its own, looser floors — but
# it must still be positive and not fall off a cliff
HOLDOUT_FLOORS = {"n_trades": 20, "sharpe": 0.0, "expectancy_bps": 0.0,
                  "max_drawdown": -0.30}
IMPROVEMENT_MARGIN = 0.10        # candidate Sharpe must beat incumbent by 10%


def _load_pair(bundle_dir: str):
    from tensorflow import keras
    from . import models as M
    load_m = lambda p: keras.models.load_model(p, custom_objects=M.CUSTOM_OBJECTS,
                                               compile=False)
    return (load_m(f"{bundle_dir}/forecaster/model.keras"),
            load_m(f"{bundle_dir}/ppo/policy/model.keras"))


def evaluate(bundle_dir: str, dataset_path: str = C.DATASET_NPZ,
             split: str = "test", cfg: bt.ExecConfig = bt.ExecConfig(),
             with_baselines: bool = True, verbose=True) -> dict:
    """Backtest one bundle on one slice. `split` is 'test' or 'holdout'."""
    from . import models as M
    d = load(dataset_path)
    forecaster, actor = _load_pair(bundle_dir)

    candles = d["candles"]
    X, anchors, vols = d[f"X_{split}"], d[f"anchor_{split}"], d[f"vol_{split}"]
    policy = M.make_policy(forecaster, actor, X, anchors, vols, candles)
    lo, hi = int(anchors[0]), int(anchors[-1]) + 2

    # Per-bar volatility for cost scaling: the same causal series the state
    # vector and the PPO reward use, computed over the same candles.
    vol_by_bar = realized_volatility(candles[:hi, 4])
    m = bt.metrics(bt.run(candles[:hi], policy, start=lo, cfg=cfg, vol=vol_by_bar), cfg)

    if with_baselines:
        # baselines on identical bars — a strategy that cannot beat a coin flip
        # with the same trade count is a fee-paying random number generator
        base = {}
        for name, pol in (("random", bt.random_policy(p_buy=max(m["n_trades"], 1)
                                                      / max(hi - lo, 1))),
                          ("buy_and_hold", bt.always_long)):
            base[name] = bt.metrics(bt.run(candles[:hi], pol, start=lo, cfg=cfg), cfg)
        m["baselines"] = {k: {"sharpe": v["sharpe"], "total_return": v["total_return"],
                              "expectancy_bps": v["expectancy_bps"]}
                          for k, v in base.items()}

    p = policy.predictions
    yt = (d[f"y_{split}"] > 0.5).astype(int)
    m["forecaster"] = {
        "pred_std": float(p[:, 0].std()), "pred_mean": float(p[:, 0].mean()),
        "dir_acc": float(((p[:, 0] > 0.5).astype(int) == yt[:, 0]).mean()),
        "per_horizon_dir_acc": [float(((p[:, h] > 0.5).astype(int) == yt[:, h]).mean())
                                for h in range(p.shape[1])],
        "horizon_agreement": float(np.mean(np.abs(np.sign(p - 0.5).mean(axis=1))))}
    m["split"] = split
    m["bundle"] = bundle_dir
    m["evaluated_at"] = int(time.time())

    if verbose:
        print(bt.report(m, f"{os.path.basename(bundle_dir)} — {split}"))
        print(f"  forecaster        dir_acc={m['forecaster']['dir_acc']:.4f} "
              f"predStd={m['forecaster']['pred_std']:.4f} "
              f"per-horizon={[round(x, 3) for x in m['forecaster']['per_horizon_dir_acc']]}")
        for k, v in m.get("baselines", {}).items():
            print(f"  baseline {k:<13} sharpe={v['sharpe']:+.2f} "
                  f"return={v['total_return']:+.2%}")
    return m


def final_backtest(bundle_dir: str, dataset_path: str = C.DATASET_NPZ,
                   verbose=True) -> dict:
    """The required validation step: the holdout slice, read once, judged hard.

    Raises GateFailed if the model does not survive data no decision was ever
    made on. Reports Sharpe, max drawdown and win rate, which is what a human
    actually needs to see before anything ships.
    """
    m = evaluate(bundle_dir, dataset_path, split="holdout", verbose=verbose)
    gates.check_backtest(m, HOLDOUT_FLOORS, stage="holdout_backtest")
    if verbose:
        print(f"  HOLDOUT PASSED  sharpe={m['sharpe']:.2f} "
              f"maxDD={m['max_drawdown']:.2%} winRate={m['win_rate']:.1%}")
    return m


def passes_floors(m: dict) -> tuple[bool, list[str]]:
    fails = [f"{k}={m[k]:.4g} vs floor {floor}" for k, floor in FLOORS.items()
             if not m[k] >= floor]
    if m.get("baselines") and m["sharpe"] <= m["baselines"]["random"]["sharpe"]:
        fails.append("does not beat the random-trader baseline")
    return (not fails), fails


def gate(candidate_dir=C.CANDIDATE_DIR, live_dir=C.MODELS_DIR,
         report_dir=C.REPORTS_DIR, register: bool = True) -> dict:
    """Full promotion decision. Order: floors -> holdout -> beat the incumbent."""
    cand = evaluate(candidate_dir, split="test")
    ok, fails = passes_floors(cand)
    decision = {"promoted": False, "reasons": fails, "candidate": cand,
                "version": None}

    holdout = None
    if ok:
        try:
            holdout = final_backtest(candidate_dir)
            decision["holdout"] = holdout
        except gates.GateFailed as exc:
            ok = False
            decision["reasons"] = list(exc.reasons)
            decision["holdout"] = exc.metrics

    incumbent = None
    if os.path.exists(f"{live_dir}/forecaster/model.keras"):
        incumbent = evaluate(live_dir, split="test")
        decision["incumbent"] = {k: incumbent[k] for k in
                                 ("sharpe", "total_return", "expectancy_bps", "n_trades")}

    if ok:
        if incumbent is None:
            decision["promoted"] = True
            decision["reasons"] = ["no incumbent; candidate clears floors and holdout"]
        else:
            need = incumbent["sharpe"] * (1 + IMPROVEMENT_MARGIN) \
                if incumbent["sharpe"] > 0 else FLOORS["sharpe"]
            if cand["sharpe"] > need:
                decision["promoted"] = True
                decision["reasons"] = [f"sharpe {cand['sharpe']:.2f} > required {need:.2f}"]
            else:
                decision["reasons"] = [f"sharpe {cand['sharpe']:.2f} <= required "
                                       f"{need:.2f} (incumbent {incumbent['sharpe']:.2f})"]

    if decision["promoted"] and register:
        version = registry.register(candidate_dir, meta={
            "metrics": {k: cand[k] for k in ("sharpe", "total_return", "win_rate",
                                             "max_drawdown", "expectancy_bps", "n_trades")},
            "holdout": {k: (holdout or {}).get(k) for k in
                        ("sharpe", "max_drawdown", "win_rate", "n_trades")},
            "forecaster": cand["forecaster"]})
        registry.promote(version, reason="; ".join(decision["reasons"]),
                         metrics={"test": cand["sharpe"],
                                  "holdout": (holdout or {}).get("sharpe")})
        pruned = registry.prune()
        decision["version"] = version
        decision["pruned"] = pruned
        print(f"\nPROMOTED version {version} -> {live_dir}"
              + (f" (pruned {len(pruned)} old version(s))" if pruned else ""))
    elif decision["promoted"]:
        print("\nPROMOTED (registry write skipped by caller)")
    else:
        print("\nNOT promoted. Live models untouched.")
        for r in decision["reasons"]:
            print(f"  - {r}")

    path = f"{report_dir}/promotion_{int(time.time())}.json"
    with open(path, "w") as f:
        json.dump(decision, f, indent=2, default=float)
    print(f"report -> {path}")
    return decision


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", metavar="DIR")
    ap.add_argument("--split", default="test", choices=("test", "holdout"))
    ap.add_argument("--final-backtest", metavar="DIR")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    if a.evaluate:
        evaluate(a.evaluate, split=a.split)
    elif a.final_backtest:
        final_backtest(a.final_backtest)
    elif a.gate:
        gate()
    else:
        evaluate(C.CANDIDATE_DIR)
