"""Every read and write the application performs, in one auditable place.

Two invariants are enforced here rather than trusted to callers:

* every execution query is filtered by ``mode``, so PAPER and REAL histories can
  never contaminate each other;
* every write that matters is idempotent — re-running a cycle after a crash
  updates rows instead of duplicating them.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from sqlalchemy import delete, desc, func, select, update

from .config import get_settings
from .contracts import AdminRestrictions, CycleResult, EngineSignal, _clean, utcnow
from .db import DatabaseUnavailable, session_scope
from .tables import (AdminRuleSet, AgentDecisionRow, AppState, Cycle, EngineSignalRow,
                     KnowledgeBaseVersion, RiskAssessmentRow, RiskModelVersion,
                     SystemEvent, Trade)

log = logging.getLogger("core.repository")


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


# ── key/value runtime state ─────────────────────────────────────────────────
def get_state(key: str, default: Any = None) -> Any:
    with session_scope() as s:
        row = s.get(AppState, key)
        return row.value if row else default


def set_state(key: str, value: Any, updated_by: str = "system") -> None:
    with session_scope() as s:
        row = s.get(AppState, key)
        if row is None:
            s.add(AppState(key=key, value=value, updated_by=updated_by))
        else:
            row.value, row.updated_by, row.updated_at = value, updated_by, utcnow()


def all_state() -> dict[str, Any]:
    with session_scope() as s:
        return {r.key: r.value for r in s.execute(select(AppState)).scalars()}


# ── audit trail ─────────────────────────────────────────────────────────────
def record_event(message: str, *, level: str = "info", category: str = "general",
                 mode: str = "", payload: dict | None = None) -> None:
    """Never let an audit write take the trading loop down with it."""
    payload = _clean(payload or {})
    try:
        with session_scope(spool_on_failure={"kind": "system_event",
                                             "payload": {"message": message, "level": level,
                                                         "category": category, "mode": mode,
                                                         "payload": payload}}) as s:
            s.add(SystemEvent(ts=utcnow(), level=level, category=category, mode=mode,
                              message=str(message)[:4000], payload=payload))
    except DatabaseUnavailable:
        pass
    except Exception as exc:                                     # pragma: no cover
        log.warning("event not recorded: %s", exc)


def recent_events(limit: int = 50, category: str | None = None) -> list[dict]:
    with session_scope() as s:
        q = select(SystemEvent).order_by(desc(SystemEvent.ts)).limit(limit)
        if category:
            q = select(SystemEvent).where(SystemEvent.category == category)\
                .order_by(desc(SystemEvent.ts)).limit(limit)
        return [{"ts": _aware(r.ts), "level": r.level, "category": r.category,
                 "mode": r.mode, "message": r.message, "payload": r.payload}
                for r in s.execute(q).scalars()]


# ── the decision cycle ──────────────────────────────────────────────────────
def save_cycle(result: CycleResult) -> int | None:
    """Persist a whole cycle atomically. Returns the decision row id, if any."""
    spool = {"kind": "cycle", "payload": result.to_dict()}
    with session_scope(spool_on_failure=spool) as s:
        cycle = s.get(Cycle, result.cycle_id)
        started, finished = _aware(result.started_at), _aware(result.finished_at)
        duration = int(((finished or utcnow()) - started).total_seconds() * 1000)
        if cycle is None:
            cycle = Cycle(id=result.cycle_id, mode=result.mode, symbol=result.symbol,
                          started_at=started, instance_id=get_settings().instance_id)
            s.add(cycle)
        cycle.finished_at = finished
        cycle.duration_ms = duration
        cycle.price = float(result.price or 0.0)
        cycle.status = result.status
        cycle.error = result.error
        cycle.blocked_by = list(result.blocked_by or [])

        s.execute(delete(EngineSignalRow).where(EngineSignalRow.cycle_id == result.cycle_id))
        for sig in result.signals:
            s.add(EngineSignalRow(
                cycle_id=result.cycle_id, engine=sig.engine, ok=sig.ok, symbol=sig.symbol,
                generated_at=_aware(sig.generated_at), direction=sig.direction,
                action_hint=sig.action_hint, confidence=sig.confidence, horizon=sig.horizon,
                latency_ms=sig.latency_ms, stale=sig.stale, source=sig.source,
                error=sig.error, levels=sig.levels, features=sig.features,
                reasons=sig.reasons, raw=sig.raw))

        if result.risk is not None:
            s.execute(delete(RiskAssessmentRow).where(RiskAssessmentRow.cycle_id == result.cycle_id))
            r = result.risk
            s.add(RiskAssessmentRow(
                cycle_id=result.cycle_id, mode=result.mode, symbol=result.symbol,
                win_probability=r.win_probability, risk_score=r.risk_score,
                expected_r=r.expected_r, size_multiplier=r.size_multiplier, veto=r.veto,
                veto_reasons=r.veto_reasons, regime=r.regime, notes=r.notes,
                model_version=r.model_version, model_kind=r.model_kind,
                trained_on_samples=r.trained_on_samples, features=r.features, error=r.error))

        decision_id = None
        if result.decision is not None:
            d = result.decision
            row = s.execute(select(AgentDecisionRow)
                            .where(AgentDecisionRow.cycle_id == result.cycle_id)).scalar_one_or_none()
            if row is None:
                row = AgentDecisionRow(cycle_id=result.cycle_id)
                s.add(row)
            row.mode, row.symbol = result.mode, result.symbol
            row.action, row.confidence = d.action, d.confidence
            row.size_pct, row.size_quote = d.size_pct, d.size_quote
            row.entry_price, row.stop_price, row.target_price = (
                d.entry_price, d.stop_price, d.target_price)
            row.time_horizon, row.rationale = d.time_horizon, d.rationale
            row.change_my_mind, row.key_risks = d.change_my_mind, d.key_risks
            row.engine_agreement, row.used_theories = d.engine_agreement, d.used_theories
            row.compliance_notes, row.source = d.compliance_notes, d.source
            row.kb_version, row.admin_version = d.kb_version, d.admin_version
            row.degraded, row.raw_llm = d.degraded, d.raw_llm
            row.executed = bool(result.order)
            s.flush()
            decision_id = row.id
        return decision_id


def recent_cycles(mode: str, limit: int = 20) -> list[dict]:
    with session_scope() as s:
        rows = s.execute(select(Cycle).where(Cycle.mode == mode)
                         .order_by(desc(Cycle.started_at)).limit(limit)).scalars().all()
        out = []
        for c in rows:
            d = s.execute(select(AgentDecisionRow)
                          .where(AgentDecisionRow.cycle_id == c.id)).scalar_one_or_none()
            out.append({"cycle_id": c.id, "started_at": _aware(c.started_at),
                        "duration_ms": c.duration_ms, "price": c.price, "status": c.status,
                        "blocked_by": c.blocked_by, "error": c.error,
                        "action": d.action if d else None,
                        "confidence": d.confidence if d else None,
                        "rationale": d.rationale if d else None,
                        "source": d.source if d else None})
        return out


def latest_decision(mode: str) -> dict | None:
    with session_scope() as s:
        row = s.execute(select(AgentDecisionRow).where(AgentDecisionRow.mode == mode)
                        .order_by(desc(AgentDecisionRow.created_at)).limit(1)).scalar_one_or_none()
        if row is None:
            return None
        return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def last_signal(engine: str, symbol: str, max_age_s: int) -> EngineSignal | None:
    """The freshest cached signal for an engine — used when a live call fails."""
    cutoff = utcnow() - timedelta(seconds=max_age_s)
    with session_scope() as s:
        row = s.execute(select(EngineSignalRow)
                        .where(EngineSignalRow.engine == engine,
                               EngineSignalRow.symbol == symbol,
                               EngineSignalRow.ok.is_(True),
                               EngineSignalRow.generated_at >= cutoff)
                        .order_by(desc(EngineSignalRow.generated_at)).limit(1)).scalar_one_or_none()
        if row is None:
            return None
        return EngineSignal(engine=row.engine, ok=True, symbol=row.symbol,
                            generated_at=_aware(row.generated_at), direction=row.direction,
                            action_hint=row.action_hint, confidence=row.confidence,
                            horizon=row.horizon, levels=row.levels or {},
                            features=row.features or {}, reasons=list(row.reasons or []),
                            raw=row.raw or {}, latency_ms=row.latency_ms, stale=True,
                            source="db_cache")


# ── admin restrictions ──────────────────────────────────────────────────────
def active_admin_rules() -> AdminRestrictions:
    """Database policy merged over the env-var hard caps. Caps always win."""
    caps = get_settings().caps
    base = AdminRestrictions(
        max_position_pct=caps.max_position_pct,
        max_capital_at_risk_pct=caps.max_capital_at_risk_pct,
        max_trades_per_day=caps.max_trades_per_day,
        max_daily_loss_pct=caps.max_daily_loss_pct,
        max_open_positions=caps.max_open_positions,
        min_confidence=caps.min_confidence,
        min_order_quote=caps.min_order_quote,
    )
    try:
        with session_scope() as s:
            row = s.execute(select(AdminRuleSet).where(AdminRuleSet.active.is_(True))
                            .order_by(desc(AdminRuleSet.version)).limit(1)).scalar_one_or_none()
            payload = dict(row.payload or {}) if row else {}
            version = row.version if row else 0
            updated_by = row.updated_by if row else "default"
            updated_at = _aware(row.created_at) if row else utcnow()
    except Exception as exc:
        log.warning("admin rules unreadable (%s); falling back to env caps", exc)
        return base

    fields = {f for f in AdminRestrictions.__dataclass_fields__}
    for key, value in payload.items():
        if key in fields and value is not None:
            setattr(base, key, value)

    # An operator may tighten a hard cap but never loosen it.
    base.max_position_pct = min(float(base.max_position_pct), caps.max_position_pct)
    base.max_capital_at_risk_pct = min(float(base.max_capital_at_risk_pct),
                                       caps.max_capital_at_risk_pct)
    base.max_trades_per_day = min(int(base.max_trades_per_day), caps.max_trades_per_day)
    base.max_daily_loss_pct = min(float(base.max_daily_loss_pct), caps.max_daily_loss_pct)
    base.max_open_positions = min(int(base.max_open_positions), caps.max_open_positions)
    base.min_confidence = max(float(base.min_confidence), caps.min_confidence)
    base.version, base.updated_by, base.updated_at = version, updated_by, updated_at

    # The kill switch can also be flipped from runtime state (API/CLI), and either
    # source being on means on.
    try:
        if bool(get_state("kill_switch", {}).get("enabled")):
            base.kill_switch = True
    except Exception:
        pass
    return base


def save_admin_rules(payload: dict, updated_by: str = "admin", note: str = "") -> int:
    with session_scope() as s:
        current = s.execute(select(func.max(AdminRuleSet.version))).scalar() or 0
        version = int(current) + 1
        s.execute(update(AdminRuleSet).values(active=False))
        s.add(AdminRuleSet(version=version, payload=payload, active=True,
                           updated_by=updated_by, note=note))
        return version


def admin_rule_history(limit: int = 20) -> list[dict]:
    with session_scope() as s:
        rows = s.execute(select(AdminRuleSet).order_by(desc(AdminRuleSet.version))
                         .limit(limit)).scalars().all()
        return [{"version": r.version, "active": r.active, "updated_by": r.updated_by,
                 "note": r.note, "created_at": _aware(r.created_at), "payload": r.payload}
                for r in rows]


# ── knowledge base ──────────────────────────────────────────────────────────
def checksum(text_: str) -> str:
    return hashlib.sha256(text_.encode("utf-8")).hexdigest()


def active_kb_version() -> dict | None:
    with session_scope() as s:
        row = s.execute(select(KnowledgeBaseVersion)
                        .where(KnowledgeBaseVersion.active.is_(True))
                        .order_by(desc(KnowledgeBaseVersion.id)).limit(1)).scalar_one_or_none()
        if row is None:
            return None
        return {"id": row.id, "checksum": row.checksum, "label": row.label,
                "content": row.content, "source": row.source,
                "char_count": row.char_count, "section_count": row.section_count,
                "created_at": _aware(row.created_at)}


def save_kb_version(content: str, *, label: str, source: str = "file",
                    section_count: int = 0, note: str = "") -> dict:
    """Content-addressed: re-saving identical text just reactivates that row."""
    digest = checksum(content)
    with session_scope() as s:
        existing = s.execute(select(KnowledgeBaseVersion)
                             .where(KnowledgeBaseVersion.checksum == digest)).scalar_one_or_none()
        s.execute(update(KnowledgeBaseVersion).values(active=False))
        if existing is not None:
            existing.active = True
            existing.label = label
            row = existing
        else:
            row = KnowledgeBaseVersion(checksum=digest, label=label, content=content,
                                       source=source, char_count=len(content),
                                       section_count=section_count, active=True, note=note)
            s.add(row)
        s.flush()
        return {"id": row.id, "checksum": row.checksum, "label": row.label,
                "char_count": row.char_count, "section_count": row.section_count}


def kb_history(limit: int = 20) -> list[dict]:
    with session_scope() as s:
        rows = s.execute(select(KnowledgeBaseVersion)
                         .order_by(desc(KnowledgeBaseVersion.id)).limit(limit)).scalars().all()
        return [{"id": r.id, "checksum": r.checksum[:12], "label": r.label,
                 "active": r.active, "char_count": r.char_count,
                 "section_count": r.section_count, "created_at": _aware(r.created_at)}
                for r in rows]


# ── engine_3 model registry ─────────────────────────────────────────────────
def next_risk_model_version() -> int:
    with session_scope() as s:
        return int(s.execute(select(func.max(RiskModelVersion.version))).scalar() or 0) + 1


def save_risk_model(*, version: int, kind: str, artifact: bytes, artifact_format: str,
                    params: dict, metrics: dict, feature_names: Sequence[str],
                    trained_on_samples: int, window: tuple[datetime | None, datetime | None],
                    status: str = "candidate", note: str = "") -> int:
    with session_scope() as s:
        row = RiskModelVersion(
            version=version, kind=kind, artifact=artifact, artifact_format=artifact_format,
            artifact_sha256=hashlib.sha256(artifact or b"").hexdigest(),
            params=params, metrics=metrics, feature_names=list(feature_names),
            trained_at=utcnow(), trained_on_samples=trained_on_samples,
            train_window_start=_aware(window[0]), train_window_end=_aware(window[1]),
            status=status, note=note)
        s.add(row)
        s.flush()
        return row.id


def promote_risk_model(version: int, note: str = "") -> None:
    with session_scope() as s:
        s.execute(update(RiskModelVersion)
                  .where(RiskModelVersion.status == "active")
                  .values(status="retired"))
        s.execute(update(RiskModelVersion)
                  .where(RiskModelVersion.version == version)
                  .values(status="active", promoted_at=utcnow(), note=note))


def active_risk_model() -> dict | None:
    with session_scope() as s:
        row = s.execute(select(RiskModelVersion)
                        .where(RiskModelVersion.status == "active")
                        .order_by(desc(RiskModelVersion.version)).limit(1)).scalar_one_or_none()
        if row is None:
            return None
        return {"id": row.id, "version": row.version, "kind": row.kind,
                "artifact": row.artifact, "artifact_format": row.artifact_format,
                "params": row.params or {}, "metrics": row.metrics or {},
                "feature_names": list(row.feature_names or []),
                "trained_on_samples": row.trained_on_samples,
                "trained_at": _aware(row.trained_at)}


def risk_model_history(limit: int = 20) -> list[dict]:
    with session_scope() as s:
        rows = s.execute(select(RiskModelVersion)
                         .order_by(desc(RiskModelVersion.version)).limit(limit)).scalars().all()
        return [{"version": r.version, "kind": r.kind, "status": r.status,
                 "metrics": r.metrics, "trained_on_samples": r.trained_on_samples,
                 "trained_at": _aware(r.trained_at), "note": r.note} for r in rows]


def prune_risk_models(keep: int = 10) -> list[int]:
    """Keep the newest ``keep`` versions plus whatever is active; delete the rest.

    Runs *after* evaluation, so a freshly trained model is never pruned before it
    has had the chance to be promoted.
    """
    with session_scope() as s:
        versions = [v for (v,) in s.execute(
            select(RiskModelVersion.version).order_by(desc(RiskModelVersion.version)))]
        active = s.execute(select(RiskModelVersion.version)
                           .where(RiskModelVersion.status == "active")).scalars().all()
        doomed = [v for v in versions[keep:] if v not in set(active)]
        if doomed:
            s.execute(delete(RiskModelVersion).where(RiskModelVersion.version.in_(doomed)))
        return doomed


# ── history the risk engine learns from ─────────────────────────────────────
def closed_trades(mode: str, limit: int = 5000, since: datetime | None = None) -> list[dict]:
    with session_scope() as s:
        q = select(Trade).where(Trade.mode == mode, Trade.exit_at.isnot(None))
        if since:
            q = q.where(Trade.exit_at >= _aware(since))
        rows = s.execute(q.order_by(desc(Trade.exit_at)).limit(limit)).scalars().all()
        return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in rows]


def decisions_for_training(mode: str, limit: int = 5000) -> list[dict]:
    with session_scope() as s:
        rows = s.execute(select(AgentDecisionRow).where(AgentDecisionRow.mode == mode)
                         .order_by(desc(AgentDecisionRow.created_at)).limit(limit)).scalars().all()
        return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in rows]


def signals_by_cycle(cycle_ids: Iterable[str]) -> dict[str, list[dict]]:
    ids = [c for c in cycle_ids if c]
    if not ids:
        return {}
    out: dict[str, list[dict]] = {}
    with session_scope() as s:
        rows = s.execute(select(EngineSignalRow)
                         .where(EngineSignalRow.cycle_id.in_(ids))).scalars().all()
        for r in rows:
            out.setdefault(r.cycle_id, []).append(
                {"engine": r.engine, "ok": r.ok, "direction": r.direction,
                 "confidence": r.confidence, "features": r.features or {},
                 "levels": r.levels or {}, "horizon": r.horizon})
    return out


def trade_stats(mode: str) -> dict:
    with session_scope() as s:
        rows = s.execute(select(Trade).where(Trade.mode == mode,
                                             Trade.exit_at.isnot(None))).scalars().all()
        n = len(rows)
        if not n:
            return {"trades": 0, "win_rate": 0.0, "expectancy": 0.0, "profit_factor": 0.0,
                    "gross_profit": 0.0, "gross_loss": 0.0, "avg_holding_minutes": 0.0}
        wins = [r for r in rows if r.pnl_quote > 0]
        gp = sum(r.pnl_quote for r in wins)
        gl = abs(sum(r.pnl_quote for r in rows if r.pnl_quote <= 0))
        return {"trades": n, "win_rate": round(len(wins) / n * 100, 2),
                "expectancy": round(sum(r.pnl_quote for r in rows) / n, 4),
                "profit_factor": round(gp / gl, 3) if gl > 0 else float("inf") if gp else 0.0,
                "gross_profit": round(gp, 4), "gross_loss": round(gl, 4),
                "avg_holding_minutes": round(sum(r.holding_minutes for r in rows) / n, 1)}


def risk_feature_rows(mode: str, limit: int = 5000) -> list[dict]:
    """Per-cycle risk features joined to that cycle's price and the action taken.

    This is what lets engine_3 learn before a single trade has closed: every cycle
    leaves behind a feature snapshot and a price, so forward returns supply
    labels for the counterfactual "would a long taken here have worked".
    """
    with session_scope() as s:
        rows = s.execute(
            select(RiskAssessmentRow, Cycle, AgentDecisionRow)
            .join(Cycle, Cycle.id == RiskAssessmentRow.cycle_id)
            .outerjoin(AgentDecisionRow, AgentDecisionRow.cycle_id == RiskAssessmentRow.cycle_id)
            .where(RiskAssessmentRow.mode == mode, Cycle.price > 0)
            .order_by(Cycle.started_at.asc()).limit(limit)).all()
        return [{"cycle_id": r.cycle_id, "features": r.features or {},
                 "ts": _aware(c.started_at), "price": float(c.price or 0.0),
                 "action": d.action if d else None,
                 "confidence": float(d.confidence) if d else 0.0}
                for r, c, d in rows]


def count_trades_since(mode: str, since: datetime | None) -> int:
    with session_scope() as s:
        q = select(func.count(Trade.id)).where(Trade.mode == mode, Trade.exit_at.isnot(None))
        if since:
            q = q.where(Trade.exit_at >= _aware(since))
        return int(s.execute(q).scalar() or 0)


# ── outbox recovery ─────────────────────────────────────────────────────────
def _parse_ts(value) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return _aware(value)
    try:
        return _aware(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except ValueError:
        return None


def rebuild_cycle(payload: dict) -> None:
    """Re-insert a cycle that was spooled while Postgres was unreachable.

    Written straight from the serialized dict rather than by rehydrating the
    dataclasses, so a schema change in the contracts cannot make an old spooled
    record unreplayable.
    """
    cycle_id = payload.get("cycle_id")
    if not cycle_id:
        raise ValueError("spooled cycle has no cycle_id")
    with session_scope() as s:
        if s.get(Cycle, cycle_id) is not None:
            return                                   # already landed; nothing to do
        started = _parse_ts(payload.get("started_at")) or utcnow()
        finished = _parse_ts(payload.get("finished_at"))
        s.add(Cycle(id=cycle_id, mode=payload.get("mode", ""), symbol=payload.get("symbol", ""),
                    started_at=started, finished_at=finished,
                    duration_ms=int(((finished or started) - started).total_seconds() * 1000),
                    price=float(payload.get("price") or 0.0),
                    status=payload.get("status", "recovered"), error=payload.get("error"),
                    blocked_by=list(payload.get("blocked_by") or [])))
        for sig in payload.get("signals") or []:
            s.add(EngineSignalRow(
                cycle_id=cycle_id, engine=sig.get("engine", "?"), ok=bool(sig.get("ok")),
                symbol=sig.get("symbol", ""),
                generated_at=_parse_ts(sig.get("generated_at")) or started,
                direction=sig.get("direction", "NEUTRAL"),
                action_hint=sig.get("action_hint", "HOLD"),
                confidence=float(sig.get("confidence") or 0.0),
                horizon=sig.get("horizon", ""), latency_ms=int(sig.get("latency_ms") or 0),
                stale=bool(sig.get("stale")), source=sig.get("source", ""),
                error=sig.get("error"), levels=sig.get("levels") or {},
                features=sig.get("features") or {}, reasons=sig.get("reasons") or [],
                raw=sig.get("raw") or {}))
        r = payload.get("risk")
        if r:
            s.add(RiskAssessmentRow(
                cycle_id=cycle_id, mode=payload.get("mode", ""), symbol=payload.get("symbol", ""),
                win_probability=float(r.get("win_probability") or 0.5),
                risk_score=float(r.get("risk_score") or 0.5),
                expected_r=float(r.get("expected_r") or 0.0),
                size_multiplier=float(r.get("size_multiplier") or 0.0),
                veto=bool(r.get("veto")), veto_reasons=r.get("veto_reasons") or [],
                regime=r.get("regime", "unknown"), notes=r.get("notes") or [],
                model_version=r.get("model_version"), model_kind=r.get("model_kind", "unknown"),
                trained_on_samples=int(r.get("trained_on_samples") or 0),
                features=r.get("features") or {}, error=r.get("error")))
        d = payload.get("decision")
        if d:
            s.add(AgentDecisionRow(
                cycle_id=cycle_id, mode=payload.get("mode", ""),
                symbol=payload.get("symbol", ""), action=d.get("action", "HOLD"),
                confidence=float(d.get("confidence") or 0.0),
                size_pct=float(d.get("size_pct") or 0.0),
                size_quote=float(d.get("size_quote") or 0.0),
                entry_price=d.get("entry_price"), stop_price=d.get("stop_price"),
                target_price=d.get("target_price"), time_horizon=d.get("time_horizon", ""),
                rationale=d.get("rationale", ""), change_my_mind=d.get("change_my_mind") or [],
                key_risks=d.get("key_risks") or [],
                engine_agreement=d.get("engine_agreement", ""),
                used_theories=d.get("used_theories") or [],
                compliance_notes=d.get("compliance_notes") or [],
                source=d.get("source", "recovered"), kb_version=d.get("kb_version"),
                admin_version=int(d.get("admin_version") or 0),
                degraded=bool(d.get("degraded")), executed=bool(payload.get("order")),
                raw_llm=d.get("raw_llm") or {}))


def replay_record(record: dict) -> None:
    """Handler for :func:`core.db.replay_outbox`."""
    kind = record.get("kind")
    payload = record.get("payload") or {}
    if kind == "cycle":
        rebuild_cycle(payload)
    elif kind == "system_event":
        with session_scope() as s:
            s.add(SystemEvent(ts=utcnow(), level=payload.get("level", "info"),
                              category=payload.get("category", "general"),
                              mode=payload.get("mode", ""),
                              message=f"[replayed] {payload.get('message', '')}"[:4000],
                              payload=payload.get("payload") or {}))
    else:
        raise ValueError(f"unknown outbox record kind: {kind!r}")
