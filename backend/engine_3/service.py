"""Engine 3 at runtime — the third brain, speaking in probabilities and size.

It answers one question: *given everything this system has already lived through,
how does a setup shaped like this one usually end?* The answer is a win
probability, an expected R, a size multiplier and, when things are bad enough, a
veto.

Two properties matter operationally:

* **Loads from the database at start.** No model file, no warm-up, no shared
  volume. Whatever version is marked active in the database is what scores the next
  cycle, on any machine.
* **Hot-swaps without a restart.** A training job that promotes a new version is
  picked up on the next cycle: the service re-checks the active version number on
  a short interval and rebuilds only when it actually changed.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterable, Sequence

from core import repository
from core.config import get_settings
from core.contracts import RiskAssessment, utcnow

from . import registry
from .features import build_features
from .model import HeuristicRiskModel

log = logging.getLogger("engine_3.service")

VERSION_CHECK_S = 30
KELLY_FRACTION = 0.25          # quarter-Kelly: full Kelly is a drawdown machine
FULL_SIZE_KELLY = 0.10         # the quarter-Kelly stake that earns the full allowed size


class RiskEngine:
    """Thread-safe, hot-reloading wrapper around the active risk model."""

    def __init__(self):
        self._lock = threading.RLock()
        self._model: Any = HeuristicRiskModel()
        self._meta: dict = {"version": None, "kind": "heuristic", "trained_on_samples": 0,
                            "source": "cold_start"}
        self._checked_at = 0.0
        self._loaded = False

    # ── lifecycle ───────────────────────────────────────────────────────────
    def load(self, force: bool = False) -> dict:
        with self._lock:
            if self._loaded and not force:
                return self._meta
            model, meta = registry.load_active()
            self._model, self._meta, self._loaded = model, meta, True
            self._checked_at = time.time()
            return meta

    def maybe_reload(self) -> None:
        """Cheap version check; a full rebuild only when the active version moved."""
        if time.time() - self._checked_at < VERSION_CHECK_S:
            return
        self._checked_at = time.time()
        try:
            latest = registry.active_version()
        except Exception:
            return
        if latest != self._meta.get("version"):
            log.info("risk model changed (%s -> %s), reloading",
                     self._meta.get("version"), latest)
            self.load(force=True)

    @property
    def info(self) -> dict:
        with self._lock:
            return dict(self._meta)

    # ── scoring ─────────────────────────────────────────────────────────────
    def assess(self, *, signals: Iterable[Any], portfolio: Any,
               candles: Sequence[Sequence[float]] | None = None,
               intent: dict | None = None, restrictions: Any = None) -> RiskAssessment:
        self.load()
        self.maybe_reload()

        try:
            history = repository.trade_stats(getattr(portfolio, "mode", "") or "PAPER")
        except Exception:
            history = {}
        features = build_features(signals=signals, portfolio=portfolio, candles=candles,
                                  intent=intent or {}, now=utcnow(), history=history)

        with self._lock:
            model, meta = self._model, dict(self._meta)
        try:
            win_p = float(model.predict_proba(features))
        except Exception as exc:
            log.error("risk model scoring failed (%s) — falling back to heuristic", exc)
            win_p = HeuristicRiskModel().predict_proba(features)
            meta = {**meta, "kind": "heuristic", "version": None, "degraded": True}

        rr = max(0.2, float(features.get("risk_reward") or 1.5))
        expected_r = win_p * rr - (1.0 - win_p)

        # Fractional Kelly on the model's own numbers, then cut for regime.
        # kelly is the optimal fraction of capital; a quarter of it is the stake we
        # would actually take, and FULL_SIZE_KELLY is the stake that earns full size.
        kelly = win_p - (1.0 - win_p) / rr
        staked = max(0.0, kelly) * KELLY_FRACTION
        size_mult = max(0.0, min(1.0, staked / FULL_SIZE_KELLY))

        notes, vetoes = [], []
        dd = float(features.get("drawdown_pct") or 0.0)
        vol = float(features.get("realized_vol_24h") or 0.0)
        trades_today = float(features.get("trades_today") or 0.0)

        if features.get("engine_conflict"):
            size_mult *= 0.5
            notes.append("The two engines disagree, so any size here is halved.")
        if dd > 5:
            size_mult *= max(0.3, 1.0 - dd / 20.0)
            notes.append(f"Account is {dd:.1f}% below its peak; size cut to protect capital.")
        if vol > 1.2:
            size_mult *= 0.7
            notes.append(f"Realized volatility is elevated ({vol:.2f}%/bar); stops get hit "
                         f"more often in this regime.")
        if not features.get("both_engines_ok"):
            size_mult *= 0.6
            notes.append("Running on one engine only — size reduced while degraded.")

        settings = get_settings()
        max_trades = getattr(restrictions, "max_trades_per_day", settings.caps.max_trades_per_day)
        max_dd = getattr(restrictions, "max_daily_loss_pct", settings.caps.max_daily_loss_pct)

        if expected_r <= 0:
            vetoes.append(f"Expected value is negative ({expected_r:+.2f}R at "
                          f"{win_p * 100:.0f}% win probability and {rr:.1f}:1 reward-to-risk).")
        if win_p < 0.42:
            vetoes.append(f"Win probability {win_p * 100:.0f}% is below the 42% floor for "
                          f"a new entry.")
        if trades_today >= max_trades:
            vetoes.append(f"Daily trade cap reached ({int(trades_today)}/{int(max_trades)}).")
        if float(features.get("pnl_today_pct") or 0.0) <= -abs(max_dd):
            vetoes.append(f"Daily loss limit hit ({features['pnl_today_pct']:.2f}%).")

        regime = _regime(features)
        assessment = RiskAssessment(
            ok=True, win_probability=win_p,
            risk_score=max(0.0, min(1.0, 1.0 - win_p)),
            expected_r=round(expected_r, 4), size_multiplier=round(size_mult, 4),
            veto=bool(vetoes), veto_reasons=vetoes, regime=regime, notes=notes,
            model_version=meta.get("version"), model_kind=meta.get("kind", "heuristic"),
            trained_on_samples=int(meta.get("trained_on_samples") or 0),
            features=features)
        return assessment


def _regime(f: dict) -> str:
    vol = float(f.get("realized_vol_24h") or 0.0)
    slope = float(f.get("trend_slope_24h") or 0.0)
    if vol > 1.5:
        return "high_volatility"
    if slope > 0.4:
        return "trending_up"
    if slope < -0.4:
        return "trending_down"
    if vol < 0.3:
        return "quiet_range"
    return "chop"


_engine: RiskEngine | None = None


def get_risk_engine() -> RiskEngine:
    global _engine
    if _engine is None:
        _engine = RiskEngine()
    return _engine
