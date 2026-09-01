"""The weekly/monthly job, end to end, in one command.

    python -m trader.retrain               # full cycle, gated promotion
    python -m trader.retrain --walkforward # add a K-fold walk-forward check first

Order matters: walk-forward runs BEFORE the promotion gate, because a candidate
that only works in the most recent block is exactly what the gate alone would
wave through. If --walkforward is on and the folds are not consistent, the
candidate is not even offered for promotion.

Nothing here touches the live inference process. It writes to models_candidate/,
and promotion copies into models/, which inference.py picks up on the next bar
via mtime.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from . import config as C


def main(years=C.HISTORY_YEARS, folds=5, do_walkforward=False,
         forecaster_epochs=60, ppo_updates=200, skip_fetch=False):
    t0 = time.time()

    if not skip_fetch:
        from .fetch import update_cache
        update_cache(years=years)

    from .dataset import build
    build()

    if do_walkforward:
        import numpy as np
        from . import walkforward as wf
        from .fetch import load_cache
        from .train import train_fold
        res = wf.run(load_cache(), lambda f: train_fold(f, forecaster_epochs,
                                                        ppo_updates, verbose=False),
                     n_folds=folds,
                     out_json=os.path.join(C.REPORTS_DIR, f"walkforward_{int(t0)}.json"))
        if not res["summary"].get("consistent_edge"):
            print("\nWalk-forward says the edge is not consistent. "
                  "Candidate not built, live models untouched.")
            return {"promoted": False, "reason": "walk_forward_failed",
                    "walkforward": res["summary"]}

    # final candidate: fitted on train (validated on val), scored on the
    # untouched test slice by the gate below
    import numpy as np
    from . import train as T
    from .dataset import load

    d = load()
    fc = T.train_forecaster(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                            epochs=forecaster_epochs, verbose=2)
    health = T.forecaster_health(fc, d["X_val"], d["y_val"])
    print(f"forecaster val: {health}")
    if health["pred_std"] < T.DEGENERATE_STD:
        print("Forecaster collapsed — aborting before PPO. Live models untouched.")
        return {"promoted": False, "reason": "forecaster_collapsed", "health": health}

    preds = np.concatenate([fc(d["X_train"][i:i + 1024], training=False).numpy()
                            for i in range(0, len(d["X_train"]), 1024)], axis=0)
    actor, critic = T.train_ppo(d["candles"], d["anchor_train"], preds,
                                d["vol_train"], updates=ppo_updates)
    for sub, m in (("forecaster", fc), ("ppo/policy", actor), ("ppo/value", critic)):
        os.makedirs(f"{C.CANDIDATE_DIR}/{sub}", exist_ok=True)
        m.save(f"{C.CANDIDATE_DIR}/{sub}/model.keras")
    np.savez(f"{C.CANDIDATE_DIR}/scaler.npz", mean=d["feat_mean"], std=d["feat_std"])

    from .promote import gate
    decision = gate()
    decision["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(C.REPORTS_DIR, f"retrain_{int(t0)}.json"), "w") as f:
        json.dump({k: v for k, v in decision.items() if k != "candidate"},
                  f, indent=2, default=float)
    print(f"\nretrain cycle finished in {decision['elapsed_s']}s "
          f"(promoted={decision['promoted']})")
    return decision


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=C.HISTORY_YEARS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ppo-updates", type=int, default=200)
    a = ap.parse_args()
    main(a.years, a.folds, a.walkforward, a.epochs, a.ppo_updates, a.skip_fetch)
