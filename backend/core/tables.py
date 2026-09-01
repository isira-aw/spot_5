"""The database schema — the single place this system's memory lives.

Design rule: **nothing important is kept on local disk.** Engine outputs, the
Agent's decisions, orders, trades, both equity curves, the knowledge base, the
admin restrictions and the trained risk models (as raw bytes) are all rows here.
Moving the deployment to another machine is therefore: install the code, point
``DATABASE_URL`` at the same Postgres, start. Nothing is lost because nothing
was local in the first place.

Second rule: **PAPER and REAL never mix.** Every execution table carries a
``mode`` column and every read filters on it, so switching modes cannot corrupt
or overwrite the other mode's history.

The column types are chosen to work on Postgres (production) *and* SQLite (the
test suite), so the schema is exercised by tests without a live server.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Index,
                        Integer, LargeBinary, String, Text, UniqueConstraint,
                        JSON, func)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship

# JSONB on Postgres, plain JSON on SQLite.
JSONType = JSONB().with_variant(JSON(), "sqlite")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at = Column(DateTime(timezone=True), default=utcnow,
                        server_default=func.now(), nullable=False, index=True)


# ── runtime key/value: kill switch, active mode, scheduler heartbeat ─────────
class AppState(Base, TimestampMixin):
    __tablename__ = "app_state"
    key = Column(String(96), primary_key=True)
    value = Column(JSONType, nullable=False, default=dict)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow,
                        nullable=False)
    updated_by = Column(String(96), default="system")


# ── one pass of the pipeline ────────────────────────────────────────────────
class Cycle(Base, TimestampMixin):
    __tablename__ = "cycles"
    id = Column(String(48), primary_key=True)
    mode = Column(String(8), nullable=False, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), nullable=False)
    finished_at = Column(DateTime(timezone=True))
    duration_ms = Column(Integer, default=0)
    price = Column(Float, default=0.0)
    status = Column(String(24), default="ok", index=True)
    error = Column(Text)
    blocked_by = Column(JSONType, default=list)
    instance_id = Column(String(96), default="")

    signals = relationship("EngineSignalRow", back_populates="cycle",
                           cascade="all, delete-orphan")
    decision = relationship("AgentDecisionRow", back_populates="cycle",
                            uselist=False, cascade="all, delete-orphan")


class EngineSignalRow(Base, TimestampMixin):
    __tablename__ = "engine_signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(String(48), ForeignKey("cycles.id", ondelete="CASCADE"), index=True)
    engine = Column(String(24), nullable=False, index=True)
    ok = Column(Boolean, default=True, nullable=False)
    symbol = Column(String(32), nullable=False, index=True)
    generated_at = Column(DateTime(timezone=True), nullable=False, index=True)
    direction = Column(String(12), default="NEUTRAL")
    action_hint = Column(String(16), default="HOLD")
    confidence = Column(Float, default=0.0)
    horizon = Column(String(16), default="")
    latency_ms = Column(Integer, default=0)
    stale = Column(Boolean, default=False)
    source = Column(String(32), default="")
    error = Column(Text)
    levels = Column(JSONType, default=dict)
    features = Column(JSONType, default=dict)
    reasons = Column(JSONType, default=list)
    raw = Column(JSONType, default=dict)

    cycle = relationship("Cycle", back_populates="signals")


Index("ix_engine_signals_engine_time", EngineSignalRow.engine, EngineSignalRow.generated_at)


class RiskAssessmentRow(Base, TimestampMixin):
    __tablename__ = "risk_assessments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(String(48), index=True)
    mode = Column(String(8), nullable=False, index=True)
    symbol = Column(String(32), nullable=False)
    win_probability = Column(Float, default=0.5)
    risk_score = Column(Float, default=0.5)
    expected_r = Column(Float, default=0.0)
    size_multiplier = Column(Float, default=1.0)
    veto = Column(Boolean, default=False)
    veto_reasons = Column(JSONType, default=list)
    regime = Column(String(32), default="unknown")
    notes = Column(JSONType, default=list)
    model_version = Column(Integer)
    model_kind = Column(String(32), default="cold_start")
    trained_on_samples = Column(Integer, default=0)
    features = Column(JSONType, default=dict)
    error = Column(Text)


class AgentDecisionRow(Base, TimestampMixin):
    __tablename__ = "agent_decisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(String(48), ForeignKey("cycles.id", ondelete="CASCADE"),
                      unique=True, index=True)
    mode = Column(String(8), nullable=False, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    action = Column(String(8), nullable=False, index=True)
    confidence = Column(Float, default=0.0)
    size_pct = Column(Float, default=0.0)
    size_quote = Column(Float, default=0.0)
    entry_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    time_horizon = Column(String(32), default="")
    rationale = Column(Text, default="")
    change_my_mind = Column(JSONType, default=list)
    key_risks = Column(JSONType, default=list)
    engine_agreement = Column(String(16), default="")
    used_theories = Column(JSONType, default=list)
    compliance_notes = Column(JSONType, default=list)
    source = Column(String(64), default="fallback")
    kb_version = Column(String(64))
    admin_version = Column(Integer, default=0)
    degraded = Column(Boolean, default=False)
    executed = Column(Boolean, default=False, index=True)
    raw_llm = Column(JSONType, default=dict)

    cycle = relationship("Cycle", back_populates="decision")


# ── execution, strictly partitioned by mode ─────────────────────────────────
class Order(Base, TimestampMixin):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mode = Column(String(8), nullable=False, index=True)
    cycle_id = Column(String(48), index=True)
    decision_id = Column(Integer, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    quote_amount = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    status = Column(String(16), default="filled", index=True)
    broker = Column(String(24), default="paper")
    client_order_id = Column(String(96), nullable=False)
    exchange_order_id = Column(String(96))
    reason = Column(String(64), default="")
    filled_at = Column(DateTime(timezone=True))
    raw = Column(JSONType, default=dict)

    __table_args__ = (UniqueConstraint("mode", "client_order_id",
                                       name="uq_orders_mode_client_id"),)


class Trade(Base, TimestampMixin):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mode = Column(String(8), nullable=False, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    entry_order_id = Column(Integer)
    exit_order_id = Column(Integer)
    decision_id = Column(Integer, index=True)
    entry_cycle_id = Column(String(48), index=True)
    exit_cycle_id = Column(String(48))
    quantity = Column(Float, default=0.0)
    entry_price = Column(Float, default=0.0)
    exit_price = Column(Float, default=0.0)
    entry_at = Column(DateTime(timezone=True), index=True)
    exit_at = Column(DateTime(timezone=True), index=True)
    holding_minutes = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
    pnl_quote = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    r_multiple = Column(Float, default=0.0)
    exit_reason = Column(String(32), default="")
    stop_price = Column(Float)
    target_price = Column(Float)
    entry_confidence = Column(Float, default=0.0)
    context = Column(JSONType, default=dict)     # the snapshot engine_3 learns from


class EquityPoint(Base, TimestampMixin):
    __tablename__ = "equity_points"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mode = Column(String(8), nullable=False, index=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    equity = Column(Float, nullable=False)
    cash = Column(Float, default=0.0)
    position_value = Column(Float, default=0.0)
    price = Column(Float, default=0.0)
    drawdown_pct = Column(Float, default=0.0)
    cycle_id = Column(String(48))


class LedgerEntry(Base, TimestampMixin):
    __tablename__ = "ledger_entries"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mode = Column(String(8), nullable=False, index=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    kind = Column(String(24), nullable=False)      # deposit/buy/sell/fee/adjustment
    amount = Column(Float, nullable=False)
    balance_after = Column(Float, default=0.0)
    ref = Column(String(96), default="")
    note = Column(Text, default="")


class PositionRow(Base, TimestampMixin):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mode = Column(String(8), nullable=False)
    symbol = Column(String(32), nullable=False)
    quantity = Column(Float, default=0.0)
    avg_entry_price = Column(Float, default=0.0)
    opened_at = Column(DateTime(timezone=True))
    stop_price = Column(Float)
    target_price = Column(Float)
    bars_held = Column(Integer, default=0)
    entry_cycle_id = Column(String(48))
    entry_decision_id = Column(Integer)
    entry_confidence = Column(Float, default=0.0)
    context = Column(JSONType, default=dict)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (UniqueConstraint("mode", "symbol", name="uq_positions_mode_symbol"),)


# ── governance and learning ─────────────────────────────────────────────────
class AdminRuleSet(Base, TimestampMixin):
    __tablename__ = "admin_rules"
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, nullable=False, unique=True)
    payload = Column(JSONType, nullable=False, default=dict)
    active = Column(Boolean, default=True, index=True)
    updated_by = Column(String(96), default="admin")
    note = Column(Text, default="")


class KnowledgeBaseVersion(Base, TimestampMixin):
    __tablename__ = "kb_versions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    checksum = Column(String(64), nullable=False, unique=True, index=True)
    label = Column(String(64), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(32), default="file")
    char_count = Column(Integer, default=0)
    section_count = Column(Integer, default=0)
    active = Column(Boolean, default=True, index=True)
    note = Column(Text, default="")


class RiskModelVersion(Base, TimestampMixin):
    """engine_3's trained models, stored as bytes so they survive a host change."""
    __tablename__ = "risk_models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, nullable=False, unique=True, index=True)
    kind = Column(String(32), nullable=False)          # gbm | logistic | heuristic
    artifact = Column(LargeBinary)                     # joblib/json bytes
    artifact_format = Column(String(16), default="joblib")
    artifact_sha256 = Column(String(64), default="")
    params = Column(JSONType, default=dict)
    metrics = Column(JSONType, default=dict)
    feature_names = Column(JSONType, default=list)
    trained_at = Column(DateTime(timezone=True), default=utcnow, index=True)
    trained_on_samples = Column(Integer, default=0)
    train_window_start = Column(DateTime(timezone=True))
    train_window_end = Column(DateTime(timezone=True))
    status = Column(String(16), default="candidate", index=True)  # candidate|active|retired
    promoted_at = Column(DateTime(timezone=True))
    note = Column(Text, default="")


class SystemEvent(Base, TimestampMixin):
    __tablename__ = "system_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    level = Column(String(12), default="info", index=True)
    category = Column(String(32), default="general", index=True)
    mode = Column(String(8), default="")
    message = Column(Text, default="")
    payload = Column(JSONType, default=dict)


ALL_TABLES = (AppState, Cycle, EngineSignalRow, RiskAssessmentRow, AgentDecisionRow,
              Order, Trade, EquityPoint, LedgerEntry, PositionRow, AdminRuleSet,
              KnowledgeBaseVersion, RiskModelVersion, SystemEvent)
