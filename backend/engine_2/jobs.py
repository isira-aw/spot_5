"""Every stage of the pipeline as a callable job, plus one CLI to run them.

This is the seam between engine_2 and whatever schedules it. Each job returns a
JSON-serializable dict, records a `SystemEvent` when the backend database is
reachable (the same audit trail engine_3 and the trader write to), and raises on
a hard gate failure so a scheduler sees a failed task rather than a silent
no-op — `pipeline.scheduler.PeriodicTask` already logs, records and keeps going.

    python -m engine_2.jobs pull        # fresh OHLCV -> csv cache -> dataset.npz
    python -m engine_2.jobs forecaster  # train + gate the forecaster only
    python -m engine_2.jobs ppo         # train the agent (warm start by default)
    python -m engine_2.jobs promote     # gate + holdout backtest + promotion
    python -m engine_2.jobs cycle       # all of the above, in order
    python -m engine_2.jobs drift       # score the live model against reality

Jobs never place orders. engine_2 produces a model artifact; execution lives in
backend/execution and is driven by the Agent, not by this package.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from . import config as C
from . import gates
from . import runner

log = logging.getLogger("engine_2.jobs")

DEPS_HINT = ("Install this package's dependencies:\n"
             "    pip install -r backend/engine_2/requirements.txt")


def _require(module: str, what: str):
    """Fail with the command to run, not a bare ModuleNotFoundError.

    TensorFlow is the big one: it is not in backend/requirements.txt because the
    API process does not need it, so a fresh checkout hits this on the first
    training run.
    """
    import importlib
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"engine_2 needs {module} to {what}. "
                                  f"{DEPS_HINT}") from exc


def _event(message: str, *, level="info", payload: dict | None = None) -> None:
    """Audit trail if the backend is importable; a log line if it is not."""
    try:
        from core import repository
        repository.record_event(message, level=level, category="engine_2",
                                payload=payload or {})
    except Exception:
        log.log(logging.ERROR if level == "error" else logging.INFO,
                "%s %s", message, payload or "")


def pull(years: float = C.HISTORY_YEARS, force: bool = False) -> dict:
    """Fresh candles from the exchange, then rebuild the windowed dataset."""
    from .dataset import build
    from .fetch import update_cache

    df = update_cache(years=years, force=force)
    build()
    out = {"job": "pull", "bars": int(len(df)),
           "first_ts": int(df.timestamp.iloc[0]) if len(df) else None,
           "last_ts": int(df.timestamp.iloc[-1]) if len(df) else None,
           "dataset": C.DATASET_NPZ}
    _event(f"engine_2 data pull: {out['bars']} bars", payload=out)
    return out


def train_forecaster(epochs: int = 60, warm_start: bool = True) -> dict:
    """Forecaster only, with the hard gate. Used to fail fast before PPO."""
    _require("tensorflow", "train the forecaster")
    from . import train as T
    from .dataset import load

    d = load()
    prev = C.MODELS_DIR if warm_start else None
    fc = T.train_forecaster(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                            epochs=epochs, verbose=2,
                            warm_start=f"{prev}/forecaster/model.keras" if prev else None)
    health = T.forecaster_health(fc, d["X_val"], d["y_val"])
    gates.check_forecaster(health)
    _event("engine_2 forecaster trained and gated", payload=health)
    return {"job": "forecaster", "health": health}


def train_models(epochs: int = 60, ppo_updates: int = 200,
                 warm_start: bool = True) -> dict:
    """Forecaster + PPO -> models_candidate/. Raises GateFailed on a bad model."""
    _require("tensorflow", "train the forecaster and the PPO agent")
    from . import train as T
    from .dataset import load

    d = load()
    out = T.train_bundle(d, forecaster_epochs=epochs, ppo_updates=ppo_updates,
                         warm_start_dir=C.MODELS_DIR if warm_start else None)
    T.save_bundle(C.CANDIDATE_DIR, out["forecaster"], out["actor"], out["critic"],
                  d["feat_mean"], d["feat_std"],
                  {"health": out["health"], "ppo": out["ppo"], "dataset": d["meta"]})
    payload = {"health": out["health"],
               "policy_spread": out["ppo"]["final_policy_spread"],
               "warm_started": out["ppo"]["warm_started"]}
    _event("engine_2 candidate trained", payload=payload)
    return {"job": "train", "candidate": C.CANDIDATE_DIR, **payload}


def promote(register: bool = True) -> dict:
    """Gate the candidate on `test`, then on the untouched `holdout`, then swap."""
    _require("tensorflow", "score the candidate model")
    from .promote import gate

    decision = gate(register=register)
    _event(f"engine_2 promotion decision: promoted={decision['promoted']}",
           level="info" if decision["promoted"] else "warning",
           payload={"promoted": decision["promoted"], "version": decision.get("version"),
                    "reasons": decision["reasons"]})
    return {"job": "promote", "promoted": decision["promoted"],
            "version": decision.get("version"), "reasons": decision["reasons"],
            "test_sharpe": decision["candidate"]["sharpe"],
            "holdout_sharpe": (decision.get("holdout") or {}).get("sharpe")}


def check_drift() -> dict:
    """Rolling live-vs-realized scorecard. Returns retrain_recommended."""
    from . import drift

    st = drift.status()
    if st.get("retrain_recommended"):
        _event("engine_2 forecaster has drifted; retrain recommended",
               level="warning", payload=st)
    return {"job": "drift", **st}


def cycle(years: float = C.HISTORY_YEARS, epochs: int = 60, ppo_updates: int = 200,
          walkforward: bool = False, folds: int = 5, warm_start: bool = True,
          skip_fetch: bool = False) -> dict:
    """The whole retraining cycle. Safe to call from a scheduler: any gate
    failure leaves the live model exactly where it was."""
    t0 = time.time()
    result: dict = {"job": "cycle", "started_at": int(t0)}
    try:
        if not skip_fetch:
            runner.progress("pull", "fetching candles and rebuilding the dataset")
            result["pull"] = pull(years=years)
        if walkforward:
            runner.progress("walkforward", f"{folds} rolling folds, retrained from scratch")
            result["walkforward"] = walk_forward(folds=folds, epochs=epochs,
                                                 ppo_updates=ppo_updates)
            if not result["walkforward"]["consistent_edge"]:
                raise gates.GateFailed("walk_forward", [
                    "the edge is not consistent across folds; a candidate that only "
                    "works in the most recent block is what the single-split gate "
                    "would wave through"], result["walkforward"])
        runner.progress("train", "forecaster, then PPO, both gated")
        result["train"] = train_models(epochs, ppo_updates, warm_start)
        runner.progress("promote", "test backtest, holdout backtest, promotion")
        result["promote"] = promote()
        result["ok"] = True
        runner.progress("done", "promoted" if result["promote"]["promoted"]
                        else "not promoted; live model untouched")
    except gates.GateFailed as exc:
        result |= {"ok": False, "gate": exc.stage, "reasons": exc.reasons,
                   "promoted": False}
        runner.progress("gated", f"{exc.stage}: {'; '.join(exc.reasons)[:200]}")
        _event(f"engine_2 cycle stopped at the {exc.stage} gate", level="error",
               payload={"reasons": exc.reasons})
    result["elapsed_s"] = round(time.time() - t0, 1)
    with open(f"{C.REPORTS_DIR}/cycle_{int(t0)}.json", "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    return result


def walk_forward(folds: int = 5, epochs: int = 60, ppo_updates: int = 200) -> dict:
    from . import walkforward as wf
    from .fetch import load_cache
    from .train import train_fold

    res = wf.run(load_cache(),
                 lambda f: train_fold(f, epochs, ppo_updates, verbose=False),
                 n_folds=folds,
                 out_json=f"{C.REPORTS_DIR}/walkforward_{int(time.time())}.json")
    return res["summary"]


JOBS = {"pull": pull, "forecaster": train_forecaster, "ppo": train_models,
        "train": train_models, "promote": promote, "drift": check_drift,
        "cycle": cycle, "walkforward": walk_forward}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job", choices=sorted(JOBS))
    ap.add_argument("--years", type=float, default=C.HISTORY_YEARS)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ppo-updates", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--no-warm-start", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    kw = {}
    if a.job in ("cycle",):
        kw = dict(years=a.years, epochs=a.epochs, ppo_updates=a.ppo_updates,
                  walkforward=a.walkforward, folds=a.folds,
                  warm_start=not a.no_warm_start, skip_fetch=a.skip_fetch)
    elif a.job in ("ppo", "train"):
        kw = dict(epochs=a.epochs, ppo_updates=a.ppo_updates,
                  warm_start=not a.no_warm_start)
    elif a.job == "forecaster":
        kw = dict(epochs=a.epochs, warm_start=not a.no_warm_start)
    elif a.job == "pull":
        kw = dict(years=a.years)
    elif a.job == "walkforward":
        kw = dict(folds=a.folds, epochs=a.epochs, ppo_updates=a.ppo_updates)

    print(json.dumps(JOBS[a.job](**kw), indent=2, default=float))
