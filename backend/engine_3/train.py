"""The auto-training cycle for engine_3.

    build dataset -> train candidates -> evaluate on an unseen tail ->
    save every candidate -> promote only if it beats the incumbent AND the
    heuristic floor -> prune to the newest N versions

The gate is the point. A risk model that is worse than the rules it replaced is
worse than useless, because the Agent trusts it when sizing. So the heuristic is
scored on the same holdout as the learned models and a candidate that cannot beat
it stays a candidate forever — recorded, inspectable, never served.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core import repository
from core.config import get_settings
from core.contracts import utcnow

from . import dataset as D
from . import registry
from .model import HAVE_SKLEARN, HeuristicRiskModel, SklearnRiskModel, evaluate

log = logging.getLogger("engine_3.train")

MIN_AUC = 0.53              # below this the model is not distinguishable from a coin
GBM_MIN_SAMPLES = 150       # boosting on less than this memorises


def _score(model: Any, X, y, feature_names) -> dict:
    if hasattr(model, "predict_batch"):
        probs = model.predict_batch(X)
    else:
        probs = [model.predict_proba(dict(zip(feature_names, row))) for row in X]
    return evaluate(y, probs)


def _better(candidate: dict, incumbent: dict | None) -> tuple[bool, str]:
    if candidate["auc"] < MIN_AUC:
        return False, f"AUC {candidate['auc']} below the {MIN_AUC} floor"
    if incumbent is None:
        return True, "no incumbent to beat"
    if candidate["auc"] > incumbent.get("auc", 0.5) + 0.01:
        return True, (f"AUC {candidate['auc']} beats incumbent "
                      f"{incumbent.get('auc')}")
    if (abs(candidate["auc"] - incumbent.get("auc", 0.5)) <= 0.01
            and candidate["brier"] < incumbent.get("brier", 1.0) - 0.005):
        return True, (f"AUC tied, Brier {candidate['brier']} better than "
                      f"{incumbent.get('brier')}")
    return False, (f"AUC {candidate['auc']} does not beat incumbent "
                   f"{incumbent.get('auc')}")


def run(mode: str | None = None, *, min_samples: int | None = None,
        keep: int | None = None, force_promote: bool = False) -> dict[str, Any]:
    """One full auto-training cycle. Safe to call on a timer, and idempotent."""
    settings = get_settings()
    mode = mode or settings.execution.mode
    min_samples = min_samples or settings.engines.engine_3_min_samples
    started = time.perf_counter()

    data = D.build(mode, min_samples=min_samples)
    ok, why = D.is_trainable(data, min_samples)
    if not ok:
        log.info("engine_3 training skipped: %s", why)
        repository.record_event(f"engine_3 training skipped: {why}",
                                category="engine_3", mode=mode,
                                payload={"counts": data["counts"]})
        return {"trained": False, "reason": why, "counts": data["counts"]}

    X_tr, y_tr, X_te, y_te = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    names = data["feature_names"]

    # The floor: the same rules the system uses on day one, scored on this holdout.
    floor = _score(HeuristicRiskModel(), X_te, y_te, names)
    log.info("heuristic floor on holdout: %s", floor)

    candidates: list[tuple[Any, dict]] = []
    if HAVE_SKLEARN:
        kinds = ["logistic"] + (["gbm"] if len(y_tr) >= GBM_MIN_SAMPLES else [])
        for kind in kinds:
            try:
                model = SklearnRiskModel(kind=kind, feature_names=names).fit(X_tr, y_tr)
                metrics = _score(model, X_te, y_te, names)
                metrics["kind"] = kind
                candidates.append((model, metrics))
                log.info("candidate %s: %s", kind, metrics)
            except Exception as exc:
                log.warning("candidate %s failed to train: %s", kind, exc)
    else:
        log.warning("scikit-learn missing — engine_3 stays on the heuristic floor")

    if not candidates:
        repository.record_event("engine_3 produced no trainable candidate",
                                level="warning", category="engine_3", mode=mode)
        return {"trained": False, "reason": "no candidate trained",
                "counts": data["counts"], "floor": floor}

    # best by AUC, ties broken by Brier
    model, metrics = min(candidates, key=lambda t: (-t[1]["auc"], t[1]["brier"]))
    metrics = {**metrics, "floor": floor, "mode": mode,
               "trained_at": utcnow().isoformat(),
               "elapsed_s": round(time.perf_counter() - started, 2)}

    version = registry.save_candidate(
        model, metrics=metrics, counts=data["counts"], window=data["window"],
        feature_names=names, note=f"auto-train {mode}")

    incumbent_row = repository.active_risk_model()
    incumbent = (incumbent_row or {}).get("metrics") if incumbent_row else None
    beats_incumbent, why_inc = _better(metrics, incumbent)
    beats_floor = metrics["auc"] >= floor["auc"] - 0.005
    promoted = force_promote or (beats_incumbent and beats_floor)

    if promoted:
        registry.promote(version, note=why_inc)
    else:
        reason = why_inc if not beats_incumbent else (
            f"AUC {metrics['auc']} does not clear the heuristic floor {floor['auc']}")
        log.info("engine_3 candidate v%d not promoted: %s", version, reason)
        repository.record_event(f"engine_3 candidate v{version} not promoted: {reason}",
                                category="engine_3", mode=mode, payload=metrics)

    # Retention runs last, after the promotion decision.
    removed = registry.prune(keep=keep)

    return {"trained": True, "version": version, "kind": model.kind, "promoted": promoted,
            "reason": why_inc, "metrics": metrics, "floor": floor,
            "counts": data["counts"], "pruned": removed,
            "elapsed_s": metrics["elapsed_s"]}


if __name__ == "__main__":                                        # pragma: no cover
    import argparse
    import json
    from core.db import init_db
    from core.logging_setup import setup_logging

    ap = argparse.ArgumentParser(description="engine_3 auto-training cycle")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--keep", type=int, default=None)
    ap.add_argument("--force-promote", action="store_true")
    a = ap.parse_args()

    setup_logging()
    init_db()
    print(json.dumps(run(a.mode, min_samples=a.min_samples, keep=a.keep,
                         force_promote=a.force_promote), indent=2, default=str))
