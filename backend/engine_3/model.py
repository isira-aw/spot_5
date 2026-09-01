"""The risk model itself, in three flavours that share one interface.

``heuristic``
    No training data required. Encodes the rules a risk manager would apply on
    day one — engine agreement, volatility regime, drawdown state, overtrading,
    reward-to-risk. This is what runs before the system has any history of its
    own, and it is also the permanent floor: if a learned model cannot beat it on
    a holdout, it is not promoted.

``logistic``
    L2 logistic regression on standardized features. The right choice on a few
    hundred trades — it cannot memorise, and its probabilities are well behaved.

``gbm``
    Histogram gradient boosting, once there are enough trades to justify it.

Every flavour serializes to bytes so the whole thing can live in a database
column and follow the deployment to a new machine.
"""
from __future__ import annotations

import io
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .features import FEATURE_NAMES, vectorize

log = logging.getLogger("engine_3.model")

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:                                                # pragma: no cover
    HAVE_NUMPY = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except Exception:                                                # pragma: no cover
    HAVE_SKLEARN = False


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    e = math.exp(max(x, -60.0))
    return e / (1.0 + e)


# ── the cold-start / floor model ─────────────────────────────────────────────
@dataclass
class HeuristicRiskModel:
    """Trading-desk rules as log-odds adjustments on a 50% prior.

    The weights are not fitted; they are the shape of the prior every trading
    book teaches. A learned model has to beat this to be allowed near a decision.
    """
    kind: str = "heuristic"
    feature_names: tuple[str, ...] = FEATURE_NAMES
    weights: dict[str, float] = field(default_factory=lambda: {
        "engines_agree": 0.55,          # the two brains pointing the same way
        "engine_conflict": -0.75,       # ... and the cost when they do not
        "both_engines_ok": 0.20,        # a full deck beats a degraded one
        "e1_agreement_pct": 0.010,      # cross-horizon alignment, per point over 50
        "e2_decisiveness": 0.45,
        "risk_reward": 0.22,            # asymmetric payoffs carry losing hit rates
        "drawdown_pct": -0.045,         # trade smaller into a drawdown
        "trades_today": -0.09,          # overtrading is the classic account killer
        "realized_vol_24h": -0.30,
        "e1_atr_pct": -0.10,
    })

    def predict_proba(self, features: dict[str, Any]) -> float:
        f = {k: float(features.get(k, 0.0) or 0.0) for k in FEATURE_NAMES}
        z = 0.0
        z += self.weights["engines_agree"] * f["engines_agree"]
        z += self.weights["engine_conflict"] * f["engine_conflict"]
        z += self.weights["both_engines_ok"] * f["both_engines_ok"]
        z += self.weights["e1_agreement_pct"] * (f["e1_agreement_pct"] - 50.0)
        z += self.weights["e2_decisiveness"] * f["e2_decisiveness"]
        z += self.weights["risk_reward"] * (min(f["risk_reward"], 5.0) - 1.5)
        z += self.weights["drawdown_pct"] * f["drawdown_pct"]
        z += self.weights["trades_today"] * max(0.0, f["trades_today"] - 2.0)
        z += self.weights["realized_vol_24h"] * f["realized_vol_24h"]
        z += self.weights["e1_atr_pct"] * max(0.0, f["e1_atr_pct"] - 2.0)
        # Buying a blow-off top or selling a capitulation low: fade the extreme.
        if f["e1_rsi14"] > 78 and f["e1_signed_conf"] > 0:
            z -= 0.5
        if f["e1_rsi14"] < 22 and f["e1_signed_conf"] < 0:
            z -= 0.5
        return max(0.02, min(0.98, sigmoid(z)))

    def serialize(self) -> tuple[bytes, str]:
        return json.dumps({"kind": self.kind, "weights": self.weights,
                           "feature_names": list(self.feature_names)}).encode(), "json"

    @classmethod
    def deserialize(cls, blob: bytes) -> "HeuristicRiskModel":
        d = json.loads(blob.decode())
        m = cls()
        m.weights.update(d.get("weights") or {})
        return m


# ── learned models ───────────────────────────────────────────────────────────
class SklearnRiskModel:
    def __init__(self, kind: str = "logistic", feature_names: Sequence[str] = FEATURE_NAMES,
                 estimator: Any = None, params: dict | None = None):
        if not HAVE_SKLEARN:
            raise RuntimeError("scikit-learn is not installed")
        self.kind = kind
        self.feature_names = list(feature_names)
        self.params = params or {}
        self.estimator = estimator or self._build(kind, self.params)

    @staticmethod
    def _build(kind: str, params: dict):
        if kind == "gbm":
            return HistGradientBoostingClassifier(
                max_depth=params.get("max_depth", 3),
                max_iter=params.get("max_iter", 200),
                learning_rate=params.get("learning_rate", 0.06),
                min_samples_leaf=params.get("min_samples_leaf", 12),
                l2_regularization=params.get("l2", 1.0),
                random_state=7)
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=params.get("C", 0.5), max_iter=2000,
                                       class_weight="balanced", random_state=7)),
        ])

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "SklearnRiskModel":
        self.estimator.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, features: dict[str, Any]) -> float:
        x = np.asarray([vectorize(features, self.feature_names)], dtype=float)
        p = self.estimator.predict_proba(x)[0]
        classes = list(getattr(self.estimator, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else len(p) - 1
        return float(max(0.02, min(0.98, p[idx])))

    def predict_batch(self, X: Sequence[Sequence[float]]) -> list[float]:
        p = self.estimator.predict_proba(np.asarray(X, dtype=float))
        classes = list(getattr(self.estimator, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else p.shape[1] - 1
        return [float(v) for v in p[:, idx]]

    def serialize(self) -> tuple[bytes, str]:
        import joblib
        buf = io.BytesIO()
        joblib.dump({"kind": self.kind, "feature_names": self.feature_names,
                     "params": self.params, "estimator": self.estimator}, buf, compress=3)
        return buf.getvalue(), "joblib"

    @classmethod
    def deserialize(cls, blob: bytes) -> "SklearnRiskModel":
        import joblib
        d = joblib.load(io.BytesIO(blob))
        return cls(kind=d["kind"], feature_names=d["feature_names"],
                   estimator=d["estimator"], params=d.get("params") or {})


def load_model(kind: str, blob: bytes | None, fmt: str) -> Any:
    """Rebuild a model from its stored bytes; fall back to the heuristic floor."""
    try:
        if not blob:
            return HeuristicRiskModel()
        if fmt == "json" or kind == "heuristic":
            return HeuristicRiskModel.deserialize(blob)
        return SklearnRiskModel.deserialize(blob)
    except Exception as exc:
        log.error("could not load risk model %s/%s (%s) — using heuristic floor",
                  kind, fmt, exc)
        return HeuristicRiskModel()


# ── evaluation ───────────────────────────────────────────────────────────────
def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUC with tie handling. No sklearn needed, so tests stay light."""
    pairs = sorted(zip(scores, y_true), key=lambda t: t[0])
    n = len(pairs)
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = [r for r, (_, y) in zip(ranks, pairs) if y == 1]
    n_pos, n_neg = len(pos), n - len(pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (sum(pos) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(y_true: Sequence[int], probs: Sequence[float]) -> dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {"n": 0, "auc": 0.5, "brier": 0.25, "accuracy": 0.0,
                "log_loss": 0.693, "base_rate": 0.0}
    brier = sum((p - y) ** 2 for p, y in zip(probs, y_true)) / n
    acc = sum(1 for p, y in zip(probs, y_true) if (p >= 0.5) == bool(y)) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
              for p, y in zip(probs, y_true)) / n
    return {"n": n, "auc": round(roc_auc(y_true, probs), 4), "brier": round(brier, 4),
            "accuracy": round(acc, 4), "log_loss": round(ll, 4),
            "base_rate": round(sum(y_true) / n, 4)}
