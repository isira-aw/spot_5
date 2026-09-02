"""engine_2 properties that silently break a trading pipeline.

No TensorFlow, no network, no database: these test the numpy half — features,
labels, splits, execution costs, gates, the model registry and the drift monitor.
The half that needs a GPU is checked by the gates themselves at training time.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from engine_2 import backtest as bt
from engine_2 import config as C
from engine_2 import drift, gates
from engine_2.dataset import chronological_split, label_sigma, make_windows, soft_labels
from engine_2.features import build_features


def synth(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.lognormal(3, 0.5, n)
    ts = np.arange(n) * C.BAR_MS
    return np.column_stack([ts, open_, high, low, close, vol])


# ── features and labels ──────────────────────────────────────────────────────
def test_features_are_causal():
    """Appending future bars must not change any past feature value."""
    c = synth()
    a = build_features(c[:1000])
    b = build_features(c)[:1000]
    assert np.allclose(a, b, atol=1e-5), "feature at t changes when future bars arrive"


def test_feature_shape_and_finiteness():
    f = build_features(synth())
    assert f.shape[1] == C.NUM_FEATURES
    assert np.isfinite(f).all()


def test_label_sigma_is_causal():
    c = synth()
    a = label_sigma(c[:1000, 4])
    b = label_sigma(c[:, 4])[:1000]
    assert np.allclose(a, b, atol=1e-9)


def test_labels_separate_typical_moves():
    """The point of volatility scaling: an ordinary move must not land on 0.5.

    With the notebook's fixed scale of 400, a 0.1% move produced 0.60 in a calm
    regime and 0.60 in a violent one, and the bulk of the distribution sat inside
    0.48-0.56. Scaled by trailing sigma, a one-sigma move is ~0.73 whatever the
    regime, so the label carries information about the body of the distribution.
    """
    c = synth(4000, seed=3)
    close = c[:, 4]
    _, anchor = make_windows(c)
    y = soft_labels(close, anchor)
    assert 0.05 < y[:, 0].std(), "labels are still bunched around 0.5"
    # a move of exactly one trailing sigma maps to sigmoid(1) regardless of regime
    sigma = label_sigma(close)
    i = anchor[len(anchor) // 2]
    bumped = close.copy()
    bumped[i + 1] = close[i] * np.exp(sigma[i])
    y1 = soft_labels(bumped, np.array([i]), sigma)
    assert abs(float(y1[0, 0]) - 1 / (1 + np.exp(-1.0))) < 0.02


def test_labels_are_regime_invariant():
    """The same raw return in a calm and a violent regime gets different labels."""
    n = 3000
    calm = np.full(n, 100.0) * np.exp(np.cumsum(np.full(n, 0.0)) + 0.0)
    calm = calm + np.tile([0.0, 0.01], n // 2)[:n]          # near-zero vol
    wild = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, n)))
    idx = np.array([n - 20])
    move = 0.002
    for series, name in ((calm, "calm"), (wild, "wild")):
        s = series.copy()
        s[idx[0] + 1:] = s[idx[0]] * (1 + move)
        globals()[f"_y_{name}"] = float(soft_labels(s, idx)[0, 0])
    assert _y_calm > _y_wild, "a fixed-scale label cannot tell the regimes apart"


def test_windows_align_with_labels():
    c = synth(2000)
    X, anchor = make_windows(c)
    y = soft_labels(c[:, 4], anchor)
    assert len(X) == len(y) == len(anchor)
    assert anchor[-1] + C.HORIZON < len(c)          # label never reads past the end
    i = 10
    up = c[anchor[i] + 1, 4] > c[anchor[i], 4]
    assert (y[i, 0] > 0.5) == up                    # label sign matches the move


# ── split ────────────────────────────────────────────────────────────────────
def test_split_is_chronological_embargoed_and_keeps_a_holdout():
    n = 10_000
    tr, va, te, ho = chronological_split(n)
    assert tr.start < tr.stop <= va.start < va.stop <= te.start < te.stop <= ho.start
    assert va.start - tr.stop >= C.EMBARGO_BARS
    assert te.start - va.stop >= C.EMBARGO_BARS
    assert ho.start - te.stop >= C.EMBARGO_BARS
    assert ho.stop == n
    # the embargo must cover a whole window plus its label horizon, or the last
    # training window and the first validation window literally share candles
    assert C.EMBARGO_BARS >= C.WINDOW_SIZE + C.HORIZON
    assert ho.stop - ho.start > 0, "BACKTEST_HOLDOUT produced no holdout bars"


def test_holdout_is_the_newest_data():
    c = synth(4000)
    _, anchor = make_windows(c)
    tr, va, te, ho = chronological_split(len(anchor))
    assert anchor[ho][0] > anchor[te][-1] > anchor[va][-1] > anchor[tr][-1]


# ── execution costs ──────────────────────────────────────────────────────────
def test_backtest_charges_costs():
    """A flat market must lose exactly the round-trip cost on every trade."""
    n = 50
    px = np.full(n, 100.0)
    c = np.column_stack([np.arange(n) * C.BAR_MS, px, px, px, px, np.ones(n)])
    actions = np.zeros(n, int)
    actions[5] = bt.BUY
    actions[10] = bt.SELL
    cfg = bt.ExecConfig(stop_loss=1.0, take_profit=1.0)
    r = bt.run(c, actions, cfg=cfg)
    assert len(r["trades"]) == 1
    expected = (1 - cfg.slippage) * (1 - cfg.fee) / ((1 + cfg.slippage) * (1 + cfg.fee)) - 1
    assert abs(r["trades"][0].ret - expected) < 1e-12
    assert r["trades"][0].ret < 0


def test_slippage_rises_with_volatility():
    """The same trade must cost more when the market is violent."""
    assert C.slippage_for_vol(C.REFERENCE_VOL) == pytest.approx(C.SLIPPAGE_PCT)
    assert C.slippage_for_vol(C.REFERENCE_VOL * 3) > C.slippage_for_vol(C.REFERENCE_VOL)
    assert C.slippage_for_vol(1.0) <= C.SLIPPAGE_PCT * C.SLIPPAGE_VOL_MAX + 1e-12

    n = 50
    px = np.full(n, 100.0)
    c = np.column_stack([np.arange(n) * C.BAR_MS, px, px, px, px, np.ones(n)])
    actions = np.zeros(n, int)
    actions[5], actions[10] = bt.BUY, bt.SELL
    cfg = bt.ExecConfig(stop_loss=1.0, take_profit=1.0)
    calm = bt.run(c, actions, cfg=cfg, vol=np.full(n, C.REFERENCE_VOL))
    wild = bt.run(c, actions, cfg=cfg, vol=np.full(n, C.REFERENCE_VOL * 4))
    assert wild["trades"][0].ret < calm["trades"][0].ret


def test_stop_loss_fires_before_signal():
    n = 30
    px = np.full(n, 100.0)
    low = px.copy()
    low[7] = 100.0 * (1 - C.STOP_LOSS_PCT * 2)      # deep wick on bar 7
    c = np.column_stack([np.arange(n) * C.BAR_MS, px, px, low, px, np.ones(n)])
    actions = np.zeros(n, int)
    actions[5] = bt.BUY
    r = bt.run(c, actions)
    assert r["trades"][0].reason == "stop"
    assert r["trades"][0].ret < -C.STOP_LOSS_PCT + 0.001


def test_metrics_on_a_known_winner():
    n = 400
    close = 100 * np.exp(np.linspace(0, 0.5, n))
    c = np.column_stack([np.arange(n) * C.BAR_MS, close, close * 1.0001,
                         close * 0.9999, close, np.ones(n)])
    cfg = bt.ExecConfig(stop_loss=1.0, take_profit=1.0)
    m = bt.metrics(bt.run(c, bt.always_long, cfg=cfg), cfg)
    assert m["n_trades"] >= 1
    assert m["total_return"] > 0.4
    assert m["max_drawdown"] <= 0.0
    assert m["sharpe"] > 0


def test_random_policy_loses_to_costs():
    """Sanity floor: random trading on a random walk must have negative expectancy."""
    c = synth(6000, seed=7)
    cfg = bt.ExecConfig(stop_loss=1.0, take_profit=1.0)
    m = bt.metrics(bt.run(c, bt.random_policy(0.05, seed=3), cfg=cfg), cfg)
    assert m["n_trades"] > 50
    assert m["expectancy"] < 0, "costs are not being applied"
    assert not m["edge_after_costs"]


# ── gates: these must RAISE, not print ───────────────────────────────────────
def test_collapsed_forecaster_stops_the_pipeline():
    with pytest.raises(gates.GateFailed):
        gates.check_forecaster({"pred_std": 0.0001, "dir_acc": 0.60, "pred_mean": 0.5})


def test_coin_flip_forecaster_stops_the_pipeline():
    with pytest.raises(gates.GateFailed):
        gates.check_forecaster({"pred_std": 0.2, "dir_acc": 0.499, "pred_mean": 0.5})


def test_one_sided_forecaster_stops_the_pipeline():
    with pytest.raises(gates.GateFailed):
        gates.check_forecaster({"pred_std": 0.2, "dir_acc": 0.6, "pred_mean": 0.95})


def test_healthy_forecaster_passes():
    h = {"pred_std": 0.12, "dir_acc": 0.54, "pred_mean": 0.51}
    assert gates.check_forecaster(h) is h


def test_constant_policy_is_rejected():
    """The 'why does it always say HOLD' failure, as a gate."""
    with pytest.raises(gates.GateFailed, match="near-constant"):
        gates.check_policy({"final_policy_spread": 0.001})
    assert gates.check_policy({"final_policy_spread": 0.4})


def test_backtest_floors_raise():
    with pytest.raises(gates.GateFailed):
        gates.check_backtest({"sharpe": -0.2, "n_trades": 500},
                             {"sharpe": 0.5, "n_trades": 50})


# ── drift monitor ────────────────────────────────────────────────────────────
def _feed(state, prices, p_up):
    """Replay a price path, filing prediction p_up(i) at each bar.

    A prediction filed at bar i is judged against prices[i + HORIZON], which is
    what the live monitor does, so p_up must be a function of the future to score
    well — exactly the thing being measured.
    """
    ts = 1_700_000_000_000
    for i, px in enumerate(prices):
        state = drift.resolve(ts + i * C.BAR_MS, float(px), state)
        state = drift.record_prediction(ts + i * C.BAR_MS, float(px),
                                        p_up(i), "vtest", state)
    return state


def _random_walk(n, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))


def test_drift_needs_a_sustained_window_before_judging():
    st = _feed(drift._empty(), _random_walk(20), lambda i: [0.6] * C.HORIZON)
    v = drift.evaluate(st)
    assert v["verdict"] == "warming_up" and not v["retrain_recommended"]


def test_drift_flags_a_decayed_forecaster():
    """Predict up every bar while the market alternates: accuracy ~0.5, and the
    prediction spread is zero. Both breach; three consecutive checks escalate."""
    px = _random_walk(400, seed=5)
    st = _feed(drift._empty(), px, lambda i: [0.55 + 0.001 * (i % 2)] * C.HORIZON)
    for _ in range(C.DRIFT_BREACHES_TO_ALERT):
        v = drift.evaluate(st)
    assert v["verdict"] == "degraded"
    assert v["breaches"] >= C.DRIFT_BREACHES_TO_ALERT
    assert v["retrain_recommended"]
    assert any("directionalAccuracy" in r or "predStd" in r for r in v["reasons"])


def test_drift_is_happy_with_a_working_forecaster():
    px = _random_walk(600, seed=2)
    up = lambda i: px[min(i + C.HORIZON, len(px) - 1)] > px[i]     # an oracle
    st = _feed(drift._empty(), px,
               lambda i: [0.8 if up(i) else 0.2] * C.HORIZON)
    v = drift.evaluate(st)
    assert v["verdict"] == "healthy" and not v["retrain_recommended"]
    assert v["dir_acc"] > 0.9 and v["pred_std"] > C.DRIFT_MIN_PRED_STD


def test_promotion_resets_the_drift_window():
    st = _feed(drift._empty(), _random_walk(300), lambda i: [0.6] * C.HORIZON)
    assert st["observations"]
    st = drift.record_prediction(1, 100.0, [0.6] * C.HORIZON, "vnew", st)
    assert st["model_version"] == "vnew"
    assert len(st["observations"]) == 0, "old model's scores blamed on the new one"


# ── model registry ───────────────────────────────────────────────────────────
@pytest.fixture
def registry(tmp_path, monkeypatch):
    import importlib

    from engine_2 import config as cfg
    monkeypatch.setattr(cfg, "VERSIONS_DIR", str(tmp_path / "versions"))
    monkeypatch.setattr(cfg, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(cfg, "CANDIDATE_DIR", str(tmp_path / "candidate"))
    (tmp_path / "versions").mkdir(); (tmp_path / "models").mkdir()
    mod = importlib.import_module("engine_2.registry")
    return mod


def _fake_bundle(root, tag="x"):
    import os
    for part in ("forecaster", "ppo/policy", "ppo/value"):
        os.makedirs(f"{root}/{part}", exist_ok=True)
        with open(f"{root}/{part}/model.keras", "w") as fh:
            fh.write(tag)
    with open(f"{root}/meta.json", "w") as fh:
        json.dump({"tag": tag}, fh)
    return root


def test_registry_versions_promotes_and_rolls_back(registry, tmp_path):
    v1 = registry.register(_fake_bundle(str(tmp_path / "c1"), "one"), version="v1")
    registry.promote(v1, reason="first")
    assert registry.current_version() == "v1"

    v2 = registry.register(_fake_bundle(str(tmp_path / "c2"), "two"), version="v2")
    registry.promote(v2, reason="better")
    assert registry.current_version() == "v2"
    with open(f"{registry.C.MODELS_DIR}/forecaster/model.keras") as fh:
        assert fh.read() == "two"

    # the whole point of versioning: the previous bundle is still on disk
    registry.rollback()
    assert registry.current_version() == "v1"
    with open(f"{registry.C.MODELS_DIR}/forecaster/model.keras") as fh:
        assert fh.read() == "one"


def test_registry_prune_keeps_current_and_the_newest(registry, tmp_path):
    for i in range(6):
        registry.register(_fake_bundle(str(tmp_path / f"c{i}"), str(i)), version=f"v{i}")
    registry.promote("v0", reason="old but live")
    removed = registry.prune(keep=2)
    kept = {v["version"] for v in registry.list_versions()}
    assert "v0" in kept, "pruned the version we are actually serving"
    assert {"v5", "v4"} <= kept
    assert removed and "v0" not in removed


def test_incomplete_bundle_is_not_registerable(registry, tmp_path):
    import os
    os.makedirs(tmp_path / "broken" / "forecaster")
    with open(tmp_path / "broken" / "forecaster" / "model.keras", "w") as fh:
        fh.write("only half a bundle")
    with pytest.raises(FileNotFoundError):
        registry.register(str(tmp_path / "broken"))


# ── the hard constraint ──────────────────────────────────────────────────────
def test_engine_2_contains_no_order_placement():
    """engine_2 produces a model artifact. It must never grow an execution path.

    ccxt's order methods are the ones that move money; none of them may appear
    anywhere in this package.
    """
    import os
    import ast
    forbidden = ("create_order", "createOrder", "create_market_buy", "create_market_sell",
                 "create_limit_buy", "create_limit_sell", "cancel_order", "withdraw",
                 "sapi_post", "transfer", "sapi_post_capital_withdraw_apply")
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "engine_2")
    hits = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(root, name)) as fh:
            tree = ast.parse(fh.read(), filename=name)
        # Attribute/name lookups only: prose in a docstring saying "no
        # withdrawals" is the documentation, not an order path.
        for node in ast.walk(tree):
            called = (node.attr if isinstance(node, ast.Attribute)
                      else node.id if isinstance(node, ast.Name) else None)
            if called and any(f == called for f in forbidden):
                hits.append(f"{name}:{called}")
    assert not hits, f"engine_2 must contain no order path, found {hits}"
