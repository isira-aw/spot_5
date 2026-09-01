"""Persistence: the part that has to still be true after the power goes out."""
import pytest

from core.contracts import (AgentDecision, CycleResult, EngineSignal, PortfolioState,
                            RiskAssessment, utcnow)


def _cycle(cycle_id="cyc-1", mode="PAPER", action="BUY"):
    return CycleResult(
        cycle_id=cycle_id, mode=mode, symbol="BTC/USDT", started_at=utcnow(),
        finished_at=utcnow(), price=65000.0,
        signals=[EngineSignal(engine="engine_1", direction="UP", confidence=0.6,
                              symbol="BTC/USDT", features={"rsi14": 55}),
                 EngineSignal.failed("engine_2", "BTC/USDT", "no bundle")],
        risk=RiskAssessment(win_probability=0.58, expected_r=0.4, regime="trending_up"),
        portfolio=PortfolioState(mode=mode, cash=10000, equity=10000),
        decision=AgentDecision(action=action, confidence=0.7, size_pct=10,
                               rationale="Because the engines agree.",
                               change_my_mind=["A close under 63,700"], kb_version="kb:abc"))


def test_a_cycle_saves_and_reads_back_whole(env):
    from core import repository
    decision_id = repository.save_cycle(_cycle())
    assert decision_id

    cycles = repository.recent_cycles("PAPER")
    assert cycles[0]["cycle_id"] == "cyc-1" and cycles[0]["action"] == "BUY"
    signals = repository.signals_by_cycle(["cyc-1"])["cyc-1"]
    assert {s["engine"] for s in signals} == {"engine_1", "engine_2"}
    assert repository.latest_decision("PAPER")["rationale"].startswith("Because")


def test_saving_the_same_cycle_twice_updates_instead_of_duplicating(env):
    from core import repository
    repository.save_cycle(_cycle())
    repository.save_cycle(_cycle(action="HOLD"))
    cycles = repository.recent_cycles("PAPER")
    assert len(cycles) == 1 and cycles[0]["action"] == "HOLD"
    assert len(repository.signals_by_cycle(["cyc-1"])["cyc-1"]) == 2


def test_the_two_modes_keep_separate_histories(env):
    from core import repository
    repository.save_cycle(_cycle("p-1", "PAPER", "BUY"))
    repository.save_cycle(_cycle("r-1", "REAL", "SELL"))
    assert [c["cycle_id"] for c in repository.recent_cycles("PAPER")] == ["p-1"]
    assert [c["cycle_id"] for c in repository.recent_cycles("REAL")] == ["r-1"]
    assert repository.latest_decision("REAL")["action"] == "SELL"


def test_admin_rules_may_tighten_a_cap_but_never_loosen_it(env):
    from core import repository
    repository.save_admin_rules({"max_position_pct": 5, "max_trades_per_day": 999,
                                 "min_confidence": 0.1, "notes": ["No weekends."]},
                                updated_by="isira")
    r = repository.active_admin_rules()
    assert r.max_position_pct == 5.0            # tighter than the 25% env cap: accepted
    assert r.max_trades_per_day == 12           # looser than the env cap: clipped back
    assert r.min_confidence == 0.55             # a lower floor is not allowed
    assert "No weekends." in r.notes and r.version == 1


def test_the_runtime_kill_switch_shows_up_in_the_restrictions(env):
    from core import repository
    from execution import risk_guard
    assert repository.active_admin_rules().kill_switch is False
    risk_guard.set_kill_switch(True, by="test", reason="drill")
    assert repository.active_admin_rules().kill_switch is True


def test_a_stale_signal_is_returned_when_a_live_one_is_missing(env):
    from core import repository
    repository.save_cycle(_cycle())
    cached = repository.last_signal("engine_1", "BTC/USDT", max_age_s=3600)
    assert cached and cached.stale is True and cached.source == "db_cache"
    assert repository.last_signal("engine_2", "BTC/USDT", 3600) is None   # it failed
    assert repository.last_signal("engine_1", "BTC/USDT", max_age_s=0) is None


def test_a_write_that_cannot_reach_the_database_is_spooled_and_replayed(env, monkeypatch):
    """The outbox: a network blip must not lose a decision."""
    import core.db as db
    from core import repository
    from sqlalchemy.exc import OperationalError

    def unreachable():
        raise OperationalError("SELECT 1", {}, Exception("connection lost"))

    monkeypatch.setattr(db, "get_sessionmaker", unreachable)
    with pytest.raises(db.DatabaseUnavailable):
        repository.save_cycle(_cycle("spooled-1"))
    assert db.outbox_size() == 1

    monkeypatch.undo()                               # the database comes back
    assert db.replay_outbox(repository.replay_record) == 1
    assert db.outbox_size() == 0

    recovered = repository.recent_cycles("PAPER")
    assert recovered and recovered[0]["cycle_id"] == "spooled-1"
    assert recovered[0]["action"] == "BUY"
    assert len(repository.signals_by_cycle(["spooled-1"])["spooled-1"]) == 2
    assert repository.latest_decision("PAPER")["rationale"].startswith("Because")


def test_events_survive_a_datetime_in_the_payload(env):
    from core import repository
    repository.record_event("something happened", category="test",
                            payload={"when": utcnow(), "n": 1})
    events = repository.recent_events(5, category="test")
    assert events and events[0]["message"] == "something happened"
    assert isinstance(events[0]["payload"]["when"], str)


def test_state_is_a_durable_key_value_store(env):
    from core import repository
    repository.set_state("mode_note", {"a": 1}, updated_by="test")
    assert repository.get_state("mode_note") == {"a": 1}
    repository.set_state("mode_note", {"a": 2})
    assert repository.get_state("mode_note")["a"] == 2
    assert repository.get_state("missing", "default") == "default"
