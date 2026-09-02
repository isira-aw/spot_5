"""The operator's controls.

Everything here changes what the system is allowed to do, so everything here is
behind the admin token and everything here writes an audit event. The two levers
that matter in a hurry are the kill switch (stops new entries immediately,
everywhere, including mid-cycle) and the restriction set (tightens caps for every
subsequent decision without a restart).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from core import repository
from core.config import MODES, PAPER, REAL, get_settings
from core.contracts import ACTIONS
from engine_3 import train as engine3_train
from engine_3.service import get_risk_engine
from execution import risk_guard
from llm_agent.knowledge_base import get_store as get_kb_store

from .deps import require_admin

log = logging.getLogger("api.admin")
router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


# ── payloads ────────────────────────────────────────────────────────────────
class RulesPayload(BaseModel):
    max_position_pct: float | None = Field(None, ge=0, le=100)
    max_capital_at_risk_pct: float | None = Field(None, ge=0, le=100)
    max_trades_per_day: int | None = Field(None, ge=0, le=1000)
    max_daily_loss_pct: float | None = Field(None, ge=0, le=100)
    max_open_positions: int | None = Field(None, ge=0, le=10)
    min_confidence: float | None = Field(None, ge=0, le=1)
    min_order_quote: float | None = Field(None, ge=0)
    allowed_symbols: list[str] | None = None
    blocked_symbols: list[str] | None = None
    allowed_actions: list[str] | None = None
    trading_hours_utc: list[int] | None = None
    blackout_dates: list[str] | None = None
    kill_switch: bool | None = None
    allow_new_entries: bool | None = None
    require_stop_loss: bool | None = None
    max_stop_distance_pct: float | None = Field(None, ge=0.1, le=50)
    notes: list[str] | None = None
    note: str = ""


class KillSwitchPayload(BaseModel):
    enabled: bool
    reason: str = ""


class ModePayload(BaseModel):
    mode: str
    confirm: bool = False


class KnowledgePayload(BaseModel):
    content: str = Field(min_length=50)
    label: str = ""


class CyclePayload(BaseModel):
    autotrade: bool | None = None
    force: bool = False


# ── restrictions ────────────────────────────────────────────────────────────
@router.put("/rules", summary="Replace the operator restriction set")
def put_rules(payload: RulesPayload, who: str = Depends(require_admin)) -> dict[str, Any]:
    data = {k: v for k, v in payload.model_dump(exclude={"note"}).items() if v is not None}
    bad = set(data.get("allowed_actions") or []) - set(ACTIONS)
    if bad:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"unknown actions: {sorted(bad)}")
    version = repository.save_admin_rules(data, updated_by=who, note=payload.note)
    effective = repository.active_admin_rules()
    repository.record_event(f"admin restrictions updated to v{version} by {who}",
                            category="admin", payload=data)
    return {"version": version, "submitted": data, "effective": effective.to_dict(),
            "note": "Environment hard caps still apply: a rule may tighten a cap, never "
                    "loosen it."}


@router.post("/kill-switch", summary="Block new entries immediately (exits stay open)")
def kill_switch(payload: KillSwitchPayload, who: str = Depends(require_admin)) -> dict[str, Any]:
    return risk_guard.set_kill_switch(payload.enabled, by=who, reason=payload.reason)


@router.post("/mode", summary="Switch between PAPER and REAL")
def switch_mode(payload: ModePayload, who: str = Depends(require_admin)) -> dict[str, Any]:
    mode = payload.mode.upper()
    if mode not in MODES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"mode must be one of {list(MODES)}")
    if mode == REAL:
        if not payload.confirm:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "switching to REAL requires confirm=true")
        problems = risk_guard.live_mode_preflight_for(mode)
        if problems:
            raise HTTPException(status.HTTP_412_PRECONDITION_FAILED,
                                {"message": "REAL mode preflight failed", "problems": problems})

    os.environ["TRADING_MODE"] = mode
    get_settings(refresh=True)
    from pipeline.orchestrator import reset_orchestrator
    reset_orchestrator()
    repository.record_event(f"trading mode switched to {mode} by {who}", level="warning",
                            category="admin", mode=mode)
    log.warning("trading mode is now %s", mode)
    return {"mode": mode, "note": "Both books are preserved; only execution changed. "
                                  "Set TRADING_MODE in the environment to make this "
                                  "survive a restart."}


# ── knowledge base ──────────────────────────────────────────────────────────
@router.post("/knowledge-base", summary="Publish a new knowledge base with no downtime")
def put_knowledge(payload: KnowledgePayload, who: str = Depends(require_admin)) -> dict[str, Any]:
    store = get_kb_store()
    from llm_agent.knowledge_base import parse
    try:
        parse(payload.content, label=payload.label or "uploaded")
    except Exception as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"knowledge base rejected: {exc}") from exc

    path = store.path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:      # atomic: never a half-written file
        fh.write(payload.content)
    os.replace(tmp, path)

    kb = store.refresh(force=True)
    repository.record_event(f"knowledge base published by {who}: {kb.version}",
                            category="knowledge_base", payload=kb.summary())
    return {"published": kb.summary(),
            "note": "Live from the next decision onward; nothing restarted."}


@router.post("/knowledge-base/refresh", summary="Re-read the knowledge base now")
def refresh_knowledge() -> dict[str, Any]:
    return get_kb_store().refresh(force=True).summary()


# ── engine_3 ────────────────────────────────────────────────────────────────
@router.post("/engine3/train", summary="Run an auto-training cycle now")
def train_now(mode: str | None = None, keep: int | None = Query(None, ge=1, le=100),
              force_promote: bool = False) -> dict[str, Any]:
    result = engine3_train.run(mode, keep=keep, force_promote=force_promote)
    if result.get("promoted"):
        get_risk_engine().load(force=True)
    return result


@router.post("/engine3/reload", summary="Reload the active risk model from the database")
def reload_risk_model() -> dict[str, Any]:
    return get_risk_engine().load(force=True)


# ── engine_2 ────────────────────────────────────────────────────────────────
# Training and rollback only. engine_2 produces a model artifact; it has no order
# path, and these endpoints deliberately expose none.
@router.post("/engine2/retrain", summary="Run an engine_2 retraining cycle now")
def engine_two_retrain(walkforward: bool = False, skip_fetch: bool = False,
                       warm_start: bool = True,
                       who: str = Depends(require_admin)) -> dict[str, Any]:
    """Long-running (hours on CPU). Any gate failure leaves the live model as-is."""
    from engine_2 import jobs as engine2_jobs
    s = get_settings().engines
    return engine2_jobs.cycle(epochs=s.engine_2_epochs, ppo_updates=s.engine_2_ppo_updates,
                              walkforward=walkforward, warm_start=warm_start,
                              skip_fetch=skip_fetch)


@router.post("/engine2/rollback", summary="Serve a previous engine_2 model version")
def engine_two_rollback(version: str | None = None,
                        who: str = Depends(require_admin)) -> dict[str, Any]:
    """Omit `version` to go back to whatever was live before the last promotion."""
    from engine_2 import registry as engine2_registry
    info = engine2_registry.rollback(version)
    repository.record_event(f"engine_2 rolled back to {info['version']}",
                            level="warning", category="engine_2", payload=info)
    return info


@router.get("/engine2/drift", summary="Live forecaster decay scorecard")
def engine_two_drift(who: str = Depends(require_admin)) -> dict[str, Any]:
    from engine_2 import drift
    return drift.status()


# ── cycles and scheduler ────────────────────────────────────────────────────
@router.post("/cycle/run", summary="Run one decision cycle now")
def run_cycle(payload: CyclePayload | None = None) -> dict[str, Any]:
    from pipeline.orchestrator import get_orchestrator
    payload = payload or CyclePayload()
    result = get_orchestrator().run_cycle(autotrade=payload.autotrade, force=payload.force)
    return result.to_dict()


@router.post("/scheduler/{action}", summary="Start or stop the background loops")
def scheduler(action: str, who: str = Depends(require_admin)) -> dict[str, Any]:
    from pipeline.scheduler import get_scheduler
    sched = get_scheduler()
    if action == "start":
        sched.start()
    elif action == "stop":
        sched.stop()
    else:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "action must be start or stop")
    repository.record_event(f"scheduler {action} by {who}", category="admin")
    return sched.status()


@router.post("/paper/reset", summary="Wipe and re-fund the PAPER book (never touches REAL)")
def reset_paper(starting_cash: float = Query(None, gt=0),
                confirm: bool = False, who: str = Depends(require_admin)) -> dict[str, Any]:
    if not confirm:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "pass confirm=true to reset")
    from sqlalchemy import delete

    from core.db import session_scope
    from core.tables import EquityPoint, LedgerEntry, Order, PositionRow, Trade
    with session_scope() as s:
        for table in (Order, Trade, EquityPoint, LedgerEntry, PositionRow):
            s.execute(delete(table).where(table.mode == PAPER))
    from execution.portfolio import PortfolioStore
    cash = PortfolioStore(PAPER).ensure_funded(starting_cash)
    repository.record_event(f"paper book reset by {who} (funded {cash})", level="warning",
                            category="admin", mode=PAPER)
    return {"mode": PAPER, "cash": cash,
            "note": "REAL history is untouched — the two books share no rows."}
