import pytest

from core.contracts import (BUY, DOWN, HOLD, NEUTRAL, UP, AdminRestrictions,
                            AgentDecision, EngineSignal, Position, clamp)


def test_engine_signal_normalizes_direction_and_confidence():
    s = EngineSignal(engine="engine_1", direction="up", confidence=1.7)
    assert s.direction == UP and s.confidence == 1.0
    assert EngineSignal(engine="x", direction="sideways").direction == NEUTRAL


def test_signed_confidence_encodes_direction_and_zeroes_a_dead_engine():
    assert EngineSignal(engine="a", direction=UP, confidence=0.6).signed_confidence() == 0.6
    assert EngineSignal(engine="a", direction=DOWN, confidence=0.6).signed_confidence() == -0.6
    dead = EngineSignal.failed("a", "BTC/USDT", "boom")
    assert dead.signed_confidence() == 0.0 and dead.ok is False


def test_failed_signal_carries_a_readable_reason():
    s = EngineSignal.failed("engine_2", "BTC/USDT", "ccxt missing")
    assert "engine_2" in s.reasons[0] and "ccxt missing" in s.reasons[0]


def test_agent_decision_rejects_an_unknown_action():
    assert AgentDecision(action="YOLO").action == HOLD
    assert AgentDecision(action="buy", confidence=2).action == BUY


def test_restrictions_read_as_english_for_the_prompt():
    lines = AdminRestrictions(kill_switch=True, max_trades_per_day=3,
                              notes=["No CPI days."]).as_prompt_lines()
    joined = " ".join(lines)
    assert "KILL SWITCH IS ON" in joined
    assert "3 trades per day" in joined
    assert "No CPI days." in joined


def test_position_pnl_and_open_flag():
    p = Position(symbol="BTC/USDT", quantity=0.5, avg_entry_price=100.0)
    assert p.is_open and abs(p.unrealized_pct(110.0) - 10.0) < 1e-9
    assert not Position(symbol="BTC/USDT").is_open


def test_clamp_survives_nonsense():
    assert clamp("abc", 0, 1) == 0
    assert clamp(float("nan"), 0, 1) == 0
    assert clamp(5, 0, 1) == 1


def test_implausible_engine_levels_are_replaced_as_a_pair():
    """A stop from a stale bar must not be stitched to a default target."""
    from core.contracts import sane_levels
    price = 65000.0

    # a coherent pair survives untouched, good reward-to-risk or bad
    assert sane_levels(price, 64000.0, 67000.0) == (64000.0, 67000.0)
    assert sane_levels(price, 61000.0, 65500.0) == (61000.0, 65500.0)

    # a "stop" 40% away is not a stop, and the target goes with it
    stop, target = sane_levels(price, 39000.0, 67000.0,
                               default_stop_pct=1.5, default_target_pct=3.0)
    assert stop == pytest.approx(price * 0.985) and target == pytest.approx(price * 1.03)

    # a target below the price invalidates the pair, not just itself
    stop, target = sane_levels(price, 61000.0, 64000.0,
                               default_stop_pct=1.5, default_target_pct=3.0)
    assert stop == pytest.approx(price * 0.985) and target == pytest.approx(price * 1.03)
    assert (target - price) / (price - stop) == pytest.approx(2.0)   # coherent 2:1

    # a stop inside the spread is not a stop either
    stop, _ = sane_levels(price, price * 0.9999, 67000.0)
    assert stop == pytest.approx(price * 0.985)

    # no price, nothing to say
    assert sane_levels(0.0, 1.0, 2.0) == (0.0, 0.0)
