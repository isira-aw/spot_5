"""engine_3: one feature definition, an honest gate, and a bounded model shelf."""
import math
from datetime import datetime, timedelta, timezone

import pytest

from core.contracts import EngineSignal, PortfolioState


def _seed_history(n=300, edge=True, seed=5):
    """Write n cycles with a feature snapshot and a price the label can be read from."""
    import random

    from core.db import session_scope
    from core.tables import AgentDecisionRow, Cycle, RiskAssessmentRow
    from engine_3.features import FEATURE_NAMES

    rnd = random.Random(seed)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = 60000.0
    with session_scope() as s:
        for i in range(n):
            e1 = rnd.uniform(-0.9, 0.9)
            e2 = rnd.uniform(-0.9, 0.9)
            agree = 1.0 if (e1 > 0 and e2 > 0) or (e1 < 0 and e2 < 0) else 0.0
            cid = f"cyc-{i:05d}"
            ts = t0 + timedelta(minutes=15 * i)
            s.add(Cycle(id=cid, mode="PAPER", symbol="BTC/USDT", started_at=ts,
                        finished_at=ts, price=round(price, 2), status="ok"))
            feats = {k: 0.0 for k in FEATURE_NAMES}
            feats.update({"e1_signed_conf": e1, "e2_signed_conf": e2,
                          "e1_confidence": abs(e1), "e2_confidence": abs(e2),
                          "engines_agree": agree, "engine_conflict": 1.0 - agree,
                          "both_engines_ok": 1.0, "e1_ok": 1.0, "e2_ok": 1.0,
                          "risk_reward": 2.0, "e1_agreement_pct": 50 + 40 * agree,
                          "e1_rsi14": 50 + 20 * e1, "e1_atr_pct": 1.0,
                          "realized_vol_24h": 0.4, "intent_confidence": abs(e1)})
            s.add(RiskAssessmentRow(cycle_id=cid, mode="PAPER", symbol="BTC/USDT",
                                    features=feats))
            s.add(AgentDecisionRow(cycle_id=cid, mode="PAPER", symbol="BTC/USDT",
                                   action="BUY" if e1 + e2 > 0.4 else "HOLD",
                                   confidence=abs(e1 + e2) / 2))
            # the price moves AFTER the snapshot, so the forward return is a label
            drift = (0.006 * (e1 + e2) * (1.6 if agree else 0.3)) if edge else 0.0
            price *= math.exp(drift + rnd.gauss(0, 0.004))


# ── features ────────────────────────────────────────────────────────────────
def test_the_feature_vector_is_stable_and_ordered(env):
    from engine_3.features import FEATURE_NAMES, build_features, vectorize
    f = build_features(signals=[], portfolio=None, candles=None, intent={})
    assert list(f) == list(FEATURE_NAMES)
    assert len(vectorize(f)) == len(FEATURE_NAMES)


def test_an_old_model_keeps_scoring_when_new_features_are_added(env):
    """vectorize follows the model's own name list, not today's global list."""
    from engine_3.features import build_features, vectorize
    f = build_features(signals=[], portfolio=None, candles=None, intent={})
    old_names = ["e1_confidence", "e2_confidence", "a_feature_that_no_longer_exists"]
    assert vectorize(f, old_names) == [0.0, 0.0, 0.0]


def test_conflicting_engines_are_encoded_as_conflict(env):
    from engine_3.features import build_features
    up = EngineSignal(engine="engine_1", direction="UP", confidence=0.6)
    down = EngineSignal(engine="engine_2", direction="DOWN", confidence=0.5)
    f = build_features(signals=[up, down], portfolio=None, candles=None, intent={})
    assert f["engine_conflict"] == 1.0 and f["engines_agree"] == 0.0
    assert f["e1_signed_conf"] == 0.6 and f["e2_signed_conf"] == -0.5


def test_a_dead_engine_contributes_zero_not_a_guess(env):
    from engine_3.features import build_features
    ok = EngineSignal(engine="engine_1", direction="UP", confidence=0.6)
    dead = EngineSignal.failed("engine_2", "BTC/USDT", "down")
    f = build_features(signals=[ok, dead], portfolio=None, candles=None, intent={})
    assert f["e2_ok"] == 0.0 and f["e2_signed_conf"] == 0.0 and f["both_engines_ok"] == 0.0


# ── the heuristic floor ─────────────────────────────────────────────────────
def test_the_heuristic_prefers_agreement_and_punishes_conflict(env):
    from engine_3.model import HeuristicRiskModel
    m = HeuristicRiskModel()
    agree = {"engines_agree": 1, "both_engines_ok": 1, "e1_agreement_pct": 85,
             "e2_decisiveness": 0.7, "risk_reward": 2.5, "e1_rsi14": 55}
    conflict = {**agree, "engines_agree": 0, "engine_conflict": 1, "e1_agreement_pct": 40,
                "drawdown_pct": 8, "trades_today": 6, "realized_vol_24h": 1.4}
    assert m.predict_proba(agree) > 0.7
    assert m.predict_proba(conflict) < 0.35


def test_the_heuristic_round_trips_through_bytes(env):
    from engine_3.model import HeuristicRiskModel
    blob, fmt = HeuristicRiskModel().serialize()
    assert fmt == "json"
    back = HeuristicRiskModel.deserialize(blob)
    assert back.predict_proba({"engines_agree": 1}) == pytest.approx(
        HeuristicRiskModel().predict_proba({"engines_agree": 1}))


def test_auc_is_computed_correctly_including_ties(env):
    from engine_3.model import roc_auc
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


# ── dataset and training ────────────────────────────────────────────────────
def test_shadow_labels_make_the_engine_trainable_before_any_trade(env):
    from engine_3 import dataset as D
    _seed_history(200)
    data = D.build("PAPER")
    assert data["counts"]["trades"] == 0
    assert data["counts"]["shadow"] > 150
    ok, why = D.is_trainable(data, 40)
    assert ok, why


def test_training_refuses_a_dataset_it_cannot_judge(env):
    from engine_3 import train
    result = train.run("PAPER")
    assert result["trained"] is False and "need" in result["reason"]


def test_a_model_with_an_edge_is_trained_promoted_and_reloadable(env):
    from engine_3 import registry, train
    _seed_history(400)
    result = train.run("PAPER")
    assert result["trained"] and result["promoted"], result
    assert result["metrics"]["auc"] > result["floor"]["auc"]

    model, meta = registry.load_active()
    assert meta["version"] == result["version"] and meta["source"] == "db"
    from engine_3.features import FEATURE_NAMES
    base = {k: 0.0 for k in FEATURE_NAMES}
    strong = {**base, "e1_signed_conf": 0.8, "e2_signed_conf": 0.7, "engines_agree": 1.0,
              "both_engines_ok": 1.0, "risk_reward": 2.0}
    weak = {**base, "e1_signed_conf": -0.8, "e2_signed_conf": -0.7, "engines_agree": 1.0,
            "both_engines_ok": 1.0, "risk_reward": 2.0}
    assert model.predict_proba(strong) > model.predict_proba(weak)


def test_a_model_that_cannot_beat_the_floor_is_kept_but_never_served(env):
    from core.repository import risk_model_history
    from engine_3 import train
    _seed_history(400, edge=False)          # pure noise: nothing to learn
    result = train.run("PAPER")
    assert result["trained"] is True
    assert result["promoted"] is False
    history = risk_model_history()
    assert history[0]["status"] == "candidate"
    assert not any(h["status"] == "active" for h in history)


def test_retention_keeps_ten_versions_plus_whatever_is_active(env):
    from core.repository import risk_model_history
    from engine_3 import train
    _seed_history(400)
    for _ in range(13):
        train.run("PAPER")
    history = risk_model_history(limit=100)
    active = [h for h in history if h["status"] == "active"]
    assert len(active) == 1
    assert len(history) <= 11                       # newest ten, plus the active one
    versions = sorted(h["version"] for h in history)
    assert versions[-1] == 13


def test_the_risk_engine_scores_a_live_setup_and_sizes_it(env):
    from engine_3.service import RiskEngine
    engine = RiskEngine()
    up1 = EngineSignal(engine="engine_1", direction="UP", confidence=0.7,
                       features={"rsi14": 55, "atr_pct": 1.0, "agreement_pct": 85})
    up2 = EngineSignal(engine="engine_2", direction="UP", confidence=0.65,
                       features={"decisiveness": 0.6, "p_up": [0.62]})
    pf = PortfolioState(mode="PAPER", cash=10000, equity=10000, last_price=65000)
    good = engine.assess(signals=[up1, up2], portfolio=pf,
                         candles=[[i, 100, 101, 99, 100 + i * 0.1, 5] for i in range(30)],
                         intent={"price": 65000, "stop_price": 64000, "target_price": 67000})
    assert good.ok and good.size_multiplier > 0 and not good.veto

    down = EngineSignal(engine="engine_2", direction="DOWN", confidence=0.8)
    bad = engine.assess(signals=[up1, down], portfolio=pf, candles=None,
                        intent={"price": 65000, "stop_price": 64000, "target_price": 65200})
    assert bad.size_multiplier < good.size_multiplier
    assert "engines disagree" in " ".join(bad.notes).lower() or bad.veto


def test_the_risk_engine_vetoes_a_negative_expected_value(env):
    from engine_3.service import RiskEngine
    conflict = [EngineSignal(engine="engine_1", direction="UP", confidence=0.3),
                EngineSignal(engine="engine_2", direction="DOWN", confidence=0.9)]
    pf = PortfolioState(mode="PAPER", cash=10000, equity=10000, last_price=65000,
                        max_drawdown_pct=12.0, trades_today=8)
    a = RiskEngine().assess(signals=conflict, portfolio=pf, candles=None,
                            intent={"price": 65000, "stop_price": 64000,
                                    "target_price": 65300})
    assert a.veto and a.veto_reasons
