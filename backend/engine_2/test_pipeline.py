"""Run with:  python -m pytest trader/tests -q   (or python trader/tests/test_pipeline.py)

These check the properties that silently break a trading pipeline: look-ahead in
the features, overlap across the split boundary, and a backtester that quietly
pays no fees.
"""
import numpy as np

from trader import config as C
from trader import backtest as bt
from trader.features import build_features
from trader.dataset import chronological_split, make_windows, soft_labels


def synth(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.lognormal(3, 0.5, n)
    ts = np.arange(n) * C.BAR_MS
    return np.column_stack([ts, open_, high, low, close, vol])


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


def test_split_has_no_overlap_and_is_ordered():
    n = 10_000
    tr, va, te = chronological_split(n)
    assert tr.stop <= va.start - C.EMBARGO_BARS + 1
    assert va.stop <= te.start - C.EMBARGO_BARS + 1
    assert te.stop == n
    # every train window (plus its label horizon) ends before validation starts
    assert tr.stop + C.WINDOW_SIZE + C.HORIZON <= va.start + C.WINDOW_SIZE


def test_windows_align_with_labels():
    c = synth(2000)
    X, anchor = make_windows(c)
    y = soft_labels(c[:, 4], anchor)
    assert len(X) == len(y) == len(anchor)
    assert anchor[-1] + C.HORIZON < len(c)          # label never reads past the end
    i = 10
    up = c[anchor[i] + 1, 4] > c[anchor[i], 4]
    assert (y[i, 0] > 0.5) == up                    # label sign matches the move


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
    """Uptrend + always-long should beat zero and report one trade."""
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all tests passed")
