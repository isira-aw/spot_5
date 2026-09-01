"""Model storage that survives the machine.

Trained risk models are bytes in a database column, not files on a disk. Point a
new host at the same database and it loads the same model at boot, mid-cycle,
with no export step and no shared volume.

Retention: after every evaluation the registry keeps the newest ``keep``
versions (default 10) plus whatever is currently active, and deletes the rest.
Pruning runs *after* the promotion decision, never before, so a candidate is
never deleted in the same breath that it earns its place.
"""
from __future__ import annotations

import logging
from typing import Any

from core import repository
from core.config import get_settings

from .model import HeuristicRiskModel, load_model

log = logging.getLogger("engine_3.registry")


def save_candidate(model: Any, *, metrics: dict, counts: dict, window: tuple,
                   feature_names: list[str], note: str = "") -> int:
    blob, fmt = model.serialize()
    version = repository.next_risk_model_version()
    repository.save_risk_model(
        version=version, kind=model.kind, artifact=blob, artifact_format=fmt,
        params=getattr(model, "params", {}) or {},
        metrics={**metrics, "counts": counts}, feature_names=feature_names,
        trained_on_samples=int(counts.get("total", 0)), window=window,
        status="candidate", note=note)
    log.info("saved risk model candidate v%d (%s) %s", version, model.kind, metrics)
    return version


def promote(version: int, note: str = "") -> None:
    repository.promote_risk_model(version, note=note)
    repository.record_event(f"engine_3 model v{version} promoted to active",
                            category="engine_3", payload={"version": version, "note": note})


def prune(keep: int | None = None) -> list[int]:
    keep = keep or get_settings().engines.engine_3_retention
    removed = repository.prune_risk_models(keep=keep)
    if removed:
        log.info("pruned risk model versions %s (keeping newest %d + active)", removed, keep)
        repository.record_event(f"pruned {len(removed)} old risk model version(s)",
                                category="engine_3",
                                payload={"removed": removed, "keep": keep})
    return removed


def load_active() -> tuple[Any, dict]:
    """The active model rebuilt from the database, or the heuristic floor."""
    try:
        row = repository.active_risk_model()
    except Exception as exc:
        log.error("registry unreachable (%s) — using heuristic floor", exc)
        return HeuristicRiskModel(), {"version": None, "kind": "heuristic",
                                      "trained_on_samples": 0, "source": "fallback"}
    if not row:
        return HeuristicRiskModel(), {"version": None, "kind": "heuristic",
                                      "trained_on_samples": 0, "source": "cold_start"}
    model = load_model(row["kind"], row["artifact"], row["artifact_format"])
    meta = {"version": row["version"], "kind": getattr(model, "kind", row["kind"]),
            "trained_on_samples": row["trained_on_samples"], "metrics": row["metrics"],
            "trained_at": row["trained_at"], "source": "db"}
    log.info("loaded risk model v%s (%s) trained on %s samples",
             meta["version"], meta["kind"], meta["trained_on_samples"])
    return model, meta


def active_version() -> int | None:
    try:
        row = repository.active_risk_model()
        return row["version"] if row else None
    except Exception:
        return None
