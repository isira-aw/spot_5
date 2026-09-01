"""The data contracts every part of the system speaks.

Four brains, one voice — but only if they all hand over the same shape of thing.
These dataclasses are that shape. Engines produce :class:`EngineSignal`, the risk
engine produces :class:`RiskAssessment`, the execution layer produces
:class:`PortfolioState`, and the Agent consumes all of them and emits a single
:class:`AgentDecision`.

Plain dataclasses on purpose: the engines must stay importable in a bare
NumPy-only environment, so nothing here may depend on pydantic, SQLAlchemy or
FastAPI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

# ── vocabulary ───────────────────────────────────────────────────────────────
BUY, SELL, HOLD = "BUY", "SELL", "HOLD"
ACTIONS = (BUY, SELL, HOLD)

UP, DOWN, NEUTRAL = "UP", "DOWN", "NEUTRAL"
DIRECTIONS = (UP, DOWN, NEUTRAL)

# engine_1 speaks in spot labels; this is how they map onto the three actions the
# Agent is allowed to emit.
SPOT_LABEL_TO_DIRECTION = {
    "STRONG_BUY": UP, "BUY": UP, "ACCUMULATE": UP,
    "HOLD": NEUTRAL, "WAIT": NEUTRAL,
    "REDUCE": DOWN, "EXIT": DOWN,
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clean(value: Any) -> Any:
    """JSON-safe: datetimes to ISO strings, NaN/Inf to None, dataclasses to dicts."""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 8)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        return {k: _clean(v) for k, v in asdict(self).items()}


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(x):
        return lo
    return max(lo, min(hi, x))


# ── engine output ────────────────────────────────────────────────────────────
@dataclass
class EngineSignal(_Serializable):
    """One engine's opinion, normalized.

    ``ok=False`` is a first-class outcome: an engine that times out, crashes or
    has no model yet still returns a signal, flagged, so the Agent can say "the
    quant engine is down" instead of silently deciding with one eye closed.
    """
    engine: str
    ok: bool = True
    symbol: str = ""
    generated_at: datetime = field(default_factory=utcnow)
    direction: str = NEUTRAL
    action_hint: str = HOLD
    confidence: float = 0.0                 # 0..1
    horizon: str = ""
    levels: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    error: str | None = None
    stale: bool = False
    source: str = ""                        # e.g. "live", "cache", "fallback"

    def __post_init__(self):
        self.direction = str(self.direction or NEUTRAL).upper()
        if self.direction not in DIRECTIONS:
            self.direction = NEUTRAL
        self.confidence = clamp(self.confidence, 0.0, 1.0)
        self.reasons = [str(r)[:400] for r in (self.reasons or [])][:6]

    @property
    def age_seconds(self) -> float:
        ts = self.generated_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (utcnow() - ts).total_seconds())

    def signed_confidence(self) -> float:
        """+conf for UP, -conf for DOWN, 0 for NEUTRAL or a failed engine."""
        if not self.ok:
            return 0.0
        return {UP: 1.0, DOWN: -1.0, NEUTRAL: 0.0}[self.direction] * self.confidence

    @classmethod
    def failed(cls, engine: str, symbol: str, error: str, latency_ms: int = 0) -> "EngineSignal":
        return cls(engine=engine, ok=False, symbol=symbol, error=str(error)[:1000],
                   latency_ms=latency_ms, source="error",
                   reasons=[f"{engine} did not produce a signal: {str(error)[:200]}"])


# ── engine_3 output ──────────────────────────────────────────────────────────
@dataclass
class RiskAssessment(_Serializable):
    """What the system's own history says about a setup that looks like this one."""
    ok: bool = True
    win_probability: float = 0.5            # P(this trade closes green)
    risk_score: float = 0.5                 # 0 = benign, 1 = the shape of past losers
    expected_r: float = 0.0                 # expected return in R multiples
    size_multiplier: float = 1.0            # 0..1 scaling advice for the Agent
    veto: bool = False
    veto_reasons: list[str] = field(default_factory=list)
    regime: str = "unknown"
    notes: list[str] = field(default_factory=list)
    model_version: int | None = None
    model_kind: str = "cold_start"
    trained_on_samples: int = 0
    features: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self):
        self.win_probability = clamp(self.win_probability, 0.0, 1.0)
        self.risk_score = clamp(self.risk_score, 0.0, 1.0)
        self.size_multiplier = clamp(self.size_multiplier, 0.0, 1.0)


# ── execution state ──────────────────────────────────────────────────────────
@dataclass
class Position(_Serializable):
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    opened_at: datetime | None = None
    stop_price: float | None = None
    target_price: float | None = None
    bars_held: int = 0

    @property
    def is_open(self) -> bool:
        return self.quantity > 1e-12

    def unrealized_pct(self, price: float) -> float:
        if not self.is_open or self.avg_entry_price <= 0:
            return 0.0
        return (price / self.avg_entry_price - 1.0) * 100.0


@dataclass
class PortfolioState(_Serializable):
    mode: str
    cash: float = 0.0
    equity: float = 0.0
    position: Position | None = None
    last_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_today: int = 0
    realized_pnl_today: float = 0.0
    peak_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    open_risk_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    kill_switch: bool = False
    updated_at: datetime = field(default_factory=utcnow)

    @property
    def in_position(self) -> bool:
        return bool(self.position and self.position.is_open)


# ── admin restrictions ───────────────────────────────────────────────────────
@dataclass
class AdminRestrictions(_Serializable):
    """Operator policy. Loaded from the database, merged over the env-var hard caps.

    Two things happen with these: they are *enforced* mechanically in
    :mod:`execution.risk_guard`, and they are *shown to the Agent* in plain
    English so its rationale never proposes something it is not allowed to do.
    """
    max_position_pct: float = 25.0
    max_capital_at_risk_pct: float = 2.0
    max_trades_per_day: int = 12
    max_daily_loss_pct: float = 4.0
    max_open_positions: int = 1
    min_confidence: float = 0.55
    min_order_quote: float = 10.0
    allowed_symbols: list[str] = field(default_factory=list)     # empty = all
    blocked_symbols: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=lambda: list(ACTIONS))
    trading_hours_utc: list[int] = field(default_factory=list)   # empty = 24/7
    blackout_dates: list[str] = field(default_factory=list)      # ISO yyyy-mm-dd
    kill_switch: bool = False
    allow_new_entries: bool = True
    require_stop_loss: bool = True
    max_stop_distance_pct: float = 8.0
    notes: list[str] = field(default_factory=list)               # free text, verbatim to the LLM
    version: int = 0
    updated_by: str = "default"
    updated_at: datetime = field(default_factory=utcnow)

    def as_prompt_lines(self) -> list[str]:
        """The restrictions, written the way you would brief a human trader."""
        out = [
            f"Never risk more than {self.max_capital_at_risk_pct:.2f}% of equity on one trade.",
            f"A single position may not exceed {self.max_position_pct:.1f}% of equity.",
            f"At most {self.max_trades_per_day} trades per day; at most "
            f"{self.max_open_positions} position(s) open at once.",
            f"Stop trading for the day after a {self.max_daily_loss_pct:.1f}% equity loss.",
            f"Do not act below {self.min_confidence * 100:.0f}% confidence — say HOLD instead.",
            f"Minimum order size is {self.min_order_quote:g} in quote currency.",
        ]
        if self.require_stop_loss:
            out.append(f"Every BUY must carry a stop loss, no wider than "
                       f"{self.max_stop_distance_pct:.1f}% below entry.")
        if self.allowed_symbols:
            out.append(f"Only these symbols may be traded: {', '.join(self.allowed_symbols)}.")
        if self.blocked_symbols:
            out.append(f"These symbols are blocked: {', '.join(self.blocked_symbols)}.")
        if sorted(self.allowed_actions) != sorted(ACTIONS):
            out.append(f"Only these actions are permitted right now: "
                       f"{', '.join(self.allowed_actions)}.")
        if self.trading_hours_utc:
            hours = ", ".join(f"{h:02d}:00" for h in sorted(self.trading_hours_utc))
            out.append(f"New entries are allowed only during these UTC hours: {hours}.")
        if self.blackout_dates:
            out.append(f"No new entries on: {', '.join(self.blackout_dates)}.")
        if self.kill_switch:
            out.append("KILL SWITCH IS ON: no new entries under any circumstances. "
                       "Exits are still allowed and encouraged if risk is rising.")
        if not self.allow_new_entries:
            out.append("New entries are administratively paused. Manage existing exposure only.")
        out.extend(str(n) for n in self.notes)
        return out


# ── the Agent's answer ───────────────────────────────────────────────────────
@dataclass
class AgentDecision(_Serializable):
    """The only thing in this system that is allowed to become an order."""
    action: str = HOLD
    confidence: float = 0.0                 # 0..1
    size_pct: float = 0.0                   # % of equity to deploy
    size_quote: float = 0.0                 # absolute quote amount, after sizing rules
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    time_horizon: str = ""
    rationale: str = ""                     # human voice, several sentences
    change_my_mind: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    engine_agreement: str = ""              # "aligned" | "split" | "conflicted"
    used_theories: list[str] = field(default_factory=list)
    compliance_notes: list[str] = field(default_factory=list)
    source: str = "fallback"                # "groq:model" | "ollama:model" | "fallback"
    kb_version: str | None = None
    admin_version: int = 0
    degraded: bool = False                  # true if any engine was down
    raw_llm: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self):
        self.action = str(self.action or HOLD).upper()
        if self.action not in ACTIONS:
            self.action = HOLD
        self.confidence = clamp(self.confidence, 0.0, 1.0)
        self.size_pct = clamp(self.size_pct, 0.0, 100.0)
        self.change_my_mind = [str(x)[:400] for x in (self.change_my_mind or [])][:6]
        self.key_risks = [str(x)[:400] for x in (self.key_risks or [])][:6]

    def headline(self) -> str:
        return f"{self.action} @ {self.confidence * 100:.0f}% confidence"


@dataclass
class CycleResult(_Serializable):
    """Everything one pass of the pipeline produced — the audit unit."""
    cycle_id: str
    mode: str
    symbol: str
    started_at: datetime
    finished_at: datetime | None = None
    price: float = 0.0
    signals: list[EngineSignal] = field(default_factory=list)
    risk: RiskAssessment | None = None
    portfolio: PortfolioState | None = None
    decision: AgentDecision | None = None
    order: dict[str, Any] | None = None
    blocked_by: list[str] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None


# ── level sanity ────────────────────────────────────────────────────────────
MAX_STOP_DISTANCE_PCT = 20.0        # beyond this a "stop" is not a stop
MIN_STOP_DISTANCE_PCT = 0.05        # below this it is inside the spread
MAX_TARGET_DISTANCE_PCT = 50.0


def sane_levels(price: float, stop: float | None, target: float | None, *,
                default_stop_pct: float = 1.5,
                default_target_pct: float = 3.0) -> tuple[float, float]:
    """Discard engine levels that do not belong to the price in front of us.

    An engine can hand back a level computed from a stale bar, a different feed or
    a different symbol. Sizing and reward-to-risk are both derived from these two
    numbers, so one nonsense level quietly poisons the risk assessment and the
    position size.

    The pair is validated as a **unit**: if either member is implausible, both are
    replaced by the percentage defaults. Half an engine's plan stitched to half a
    default is not a plan — a 6% stop paired with a 3% default target invents a
    0.5:1 setup that neither the engine nor the configuration ever proposed. A
    coherent pair the risk engine can judge is worth more than one salvaged level.
    """
    price = float(price or 0.0)
    if price <= 0:
        return 0.0, 0.0

    stop, target = float(stop or 0.0), float(target or 0.0)
    stop_ok = (0 < stop < price) and (
        MIN_STOP_DISTANCE_PCT <= (price - stop) / price * 100.0 <= MAX_STOP_DISTANCE_PCT)
    target_ok = target > price and (target - price) / price * 100.0 <= MAX_TARGET_DISTANCE_PCT

    if stop_ok and target_ok:
        return round(stop, 8), round(target, 8)
    return (round(price * (1 - default_stop_pct / 100.0), 8),
            round(price * (1 + default_target_pct / 100.0), 8))
