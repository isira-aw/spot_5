"""Evaluate a model bundle on the held-out test slice and decide whether it
replaces what is live.

The retrain job must never overwrite a working model just because it finished.
This is the gate: candidate and incumbent are scored on the SAME unseen bars,
against the same baselines, and the candidate ships only if it clears absolute
floors and beats the incumbent by a margin wider than fold-to-fold noise.

    python -m trader.promote --evaluate models_candidate
    python -m trader.promote --gate      # compare candidate vs models/, maybe swap
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np

from . import config as C
from . import backtest as bt
from .dataset import load

# absolute floors — a candidate below any of these does not ship even if the
# incumbent is worse
FLOORS = {"n_trades": 50, "sharpe": 0.5, "expectancy_bps": 0.0,
          "max_drawdown": -0.25, "profit_factor": 1.05}
IMPROVEMENT_MARGIN = 0.10        # candidate Sharpe must beat incumbent by 10%


def evaluate(bundle_dir: str, dataset_path: str = C.DATASET_NPZ,
             cfg: bt.ExecConfig = bt.ExecConfig(), verbose=True) -> dict:
    from . import models as M
    d = load(dataset_path)
    forecaster = __import__("tensorflow").keras.models.load_model(
        f"{bundle_dir}/forecaster/model.keras", custom_objects=M.CUSTOM_OBJECTS,
        compile=False)
    actor = __import__("tensorflow").keras.models.load_model(
        f"{bundle_dir}/ppo/policy/model.keras", custom_objects=M.CUSTOM_OBJECTS,
        compile=False)

    candles = d["candles"]
    policy = M.make_policy(forecaster, actor, d["X_test"], d["anchor_test"],
                           d["vol_test"], candles)
    lo, hi = int(d["anchor_test"][0]), int(d["anchor_test"][-1]) + 2
    m = bt.metrics(bt.run(candles[:hi], policy, start=lo, cfg=cfg), cfg)

    # baselines on identical bars — a strategy that cannot beat a coin flip with
    # the same trade count is a fee-paying random number generator
    base = {}
    for name, pol in (("random", bt.random_policy(p_buy=max(m["n_trades"], 1)
                                                  / max(hi - lo, 1))),
                      ("buy_and_hold", bt.always_long)):
        base[name] = bt.metrics(bt.run(candles[:hi], pol, start=lo, cfg=cfg), cfg)

    p = policy.predictions[:, 0]
    m["forecaster"] = {"pred_std": float(p.std()), "pred_mean": float(p.mean()),
                       "dir_acc": float(((p > 0.5).astype(int) ==
                                         (d["y_test"][:, 0] > 0.5).astype(int)).mean())}
    m["baselines"] = {k: {"sharpe": v["sharpe"], "total_return": v["total_return"],
                          "expectancy_bps": v["expectancy_bps"]} for k, v in base.items()}
    m["bundle"] = bundle_dir
    m["evaluated_at"] = int(time.time())

    if verbose:
        print(bt.report(m, f"{os.path.basename(bundle_dir)} — out-of-sample test"))
        print(f"  forecaster        dir_acc={m['forecaster']['dir_acc']:.4f} "
              f"predStd={m['forecaster']['pred_std']:.4f}")
        for k, v in m["baselines"].items():
            print(f"  baseline {k:<13} sharpe={v['sharpe']:+.2f} "
                  f"return={v['total_return']:+.2%}")
    return m


def passes_floors(m: dict) -> tuple[bool, list[str]]:
    fails = []
    for k, floor in FLOORS.items():
        v = m[k]
        ok = v >= floor if k != "max_drawdown" else v >= floor
        if not ok:
            fails.append(f"{k}={v:.4g} vs floor {floor}")
    if m["sharpe"] <= m["baselines"]["random"]["sharpe"]:
        fails.append("does not beat the random-trader baseline")
    return (not fails), fails


def gate(candidate_dir=C.CANDIDATE_DIR, live_dir=C.MODELS_DIR,
         report_dir=C.REPORTS_DIR) -> dict:
    cand = evaluate(candidate_dir)
    ok, fails = passes_floors(cand)
    decision = {"promoted": False, "reasons": fails, "candidate": cand}

    incumbent = None
    if os.path.exists(f"{live_dir}/forecaster/model.keras"):
        incumbent = evaluate(live_dir)
        decision["incumbent"] = incumbent

    if ok:
        if incumbent is None:
            decision["promoted"] = True
            decision["reasons"] = ["no incumbent; candidate clears floors"]
        else:
            need = incumbent["sharpe"] * (1 + IMPROVEMENT_MARGIN) \
                if incumbent["sharpe"] > 0 else FLOORS["sharpe"]
            if cand["sharpe"] > need:
                decision["promoted"] = True
                decision["reasons"] = [f"sharpe {cand['sharpe']:.2f} > "
                                       f"required {need:.2f}"]
            else:
                decision["reasons"] = [f"sharpe {cand['sharpe']:.2f} <= required "
                                       f"{need:.2f} (incumbent "
                                       f"{incumbent['sharpe']:.2f})"]

    if decision["promoted"]:
        backup = f"{live_dir}_prev"
        if os.path.exists(live_dir):
            shutil.rmtree(backup, ignore_errors=True)
            shutil.copytree(live_dir, backup)
        for sub in ("forecaster", "ppo/policy", "ppo/value"):
            os.makedirs(f"{live_dir}/{sub}", exist_ok=True)
            shutil.copy(f"{candidate_dir}/{sub}/model.keras",
                        f"{live_dir}/{sub}/model.keras")
        if os.path.exists(f"{candidate_dir}/scaler.npz"):
            shutil.copy(f"{candidate_dir}/scaler.npz", f"{live_dir}/scaler.npz")
        print(f"\nPROMOTED -> {live_dir} (previous kept at {backup})")
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
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    if a.evaluate:
        evaluate(a.evaluate)
    elif a.gate:
        gate()
    else:
        evaluate(C.CANDIDATE_DIR)
