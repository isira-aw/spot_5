"""Read-only views of everything the system knows.

Every execution endpoint takes a ``mode`` parameter and defaults to the configured
one, because the two books are separate and asking for "the trades" without saying
which account is a question with two different right answers.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from core import repository
from core.config import MODES, get_settings
from core.db import healthcheck, outbox_size
from core.market import get_quote
from engine_3.service import get_risk_engine
from execution import risk_guard
from execution.portfolio import PortfolioStore
from llm_agent.knowledge_base import get_store as get_kb_store

log = logging.getLogger("api.routes")
router = APIRouter()


def _mode(mode: str | None) -> str:
    m = (mode or get_settings().execution.mode).upper()
    return m if m in MODES else get_settings().execution.mode


@router.get("/health", summary="Liveness, dependencies and versions")
def health() -> dict[str, Any]:
    s = get_settings()
    db = healthcheck()
    kb = get_kb_store().status()
    risk = get_risk_engine().info
    from pipeline.scheduler import get_scheduler
    return {
        "ok": db["ok"],
        "mode": s.execution.mode,
        "symbol": s.execution.symbol,
        "database": db,
        "knowledge_base": {"version": kb.get("version"), "sections": kb.get("sections"),
                           "source": kb.get("source"), "reloads": kb.get("reloads"),
                           "last_error": kb.get("last_error")},
        "risk_model": {"version": risk.get("version"), "kind": risk.get("kind"),
                       "trained_on_samples": risk.get("trained_on_samples")},
        "kill_switch": risk_guard.is_killed(),
        "scheduler": get_scheduler().status(),
        "outbox_pending": outbox_size(),
    }


@router.get("/config", summary="Effective configuration, secrets redacted")
def config() -> dict[str, Any]:
    return get_settings().public_dict()


@router.get("/price", summary="Current spot quote and its venue")
def price(symbol: str | None = None) -> dict[str, Any]:
    return get_quote(symbol or get_settings().execution.symbol).to_dict()


@router.get("/state", summary="The book, the position and the last decision")
def state(mode: str | None = None) -> dict[str, Any]:
    m = _mode(mode)
    s = get_settings()
    store = PortfolioStore(m)
    quote = get_quote(s.execution.symbol)
    portfolio = store.state(s.execution.symbol, quote.price,
                            kill_switch=risk_guard.is_killed())
    return {"mode": m, "symbol": s.execution.symbol, "price": quote.to_dict(),
            "portfolio": portfolio.to_dict(),
            "latest_decision": repository.latest_decision(m),
            "restrictions": repository.active_admin_rules().to_dict(),
            "stats": store.stats()}


@router.get("/decisions", summary="Recent Agent answers, newest first")
def decisions(mode: str | None = None, limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    return repository.recent_cycles(_mode(mode), limit=limit)


@router.get("/cycles/{cycle_id}", summary="Everything one cycle saw and decided")
def cycle_detail(cycle_id: str) -> dict[str, Any]:
    signals = repository.signals_by_cycle([cycle_id]).get(cycle_id, [])
    return {"cycle_id": cycle_id, "signals": signals}


@router.get("/trades", summary="Closed trades for one book")
def trades(mode: str | None = None, limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    return PortfolioStore(_mode(mode)).recent_trades(limit=limit)


@router.get("/orders", summary="Order history for one book")
def orders(mode: str | None = None, limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    return PortfolioStore(_mode(mode)).recent_orders(limit=limit)


@router.get("/equity", summary="Equity curve for one book")
def equity(mode: str | None = None, limit: int = Query(200, ge=1, le=5000)) -> list[dict]:
    return PortfolioStore(_mode(mode)).equity_curve(limit=limit)


@router.get("/stats", summary="Performance statistics for one book")
def stats(mode: str | None = None) -> dict[str, Any]:
    return PortfolioStore(_mode(mode)).stats()


@router.get("/events", summary="System audit trail")
def events(limit: int = Query(50, ge=1, le=500), category: str | None = None) -> list[dict]:
    return repository.recent_events(limit=limit, category=category)


@router.get("/knowledge-base", summary="Which knowledge base is live right now")
def knowledge_base(include_content: bool = False) -> dict[str, Any]:
    store = get_kb_store()
    kb = store.get()
    out: dict[str, Any] = {**store.status(), "history": repository.kb_history(limit=10)}
    if include_content:
        out["content"] = kb.raw
    return out


@router.get("/engine3/models", summary="Risk model versions, newest first")
def risk_models(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    return {"active": get_risk_engine().info,
            "history": repository.risk_model_history(limit=limit),
            "retention": get_settings().engines.engine_3_retention}


@router.get("/admin/rules", summary="Active operator restrictions (read-only)")
def rules() -> dict[str, Any]:
    r = repository.active_admin_rules()
    return {"active": r.to_dict(), "as_briefed_to_the_agent": r.as_prompt_lines(),
            "history": repository.admin_rule_history(limit=10)}
