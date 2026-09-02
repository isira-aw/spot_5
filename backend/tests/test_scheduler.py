"""The background loops: they must start, survive a failure, and stop cleanly."""
import threading
import time


def test_a_periodic_task_keeps_running_after_it_throws(env):
    from pipeline.scheduler import PeriodicTask

    calls = {"n": 0}
    stop = threading.Event()

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first run explodes")

    task = PeriodicTask("flaky", 5, flaky, stop, run_at_start=True)
    task.state.interval_s = 0.05          # tighten for the test
    task.start()
    deadline = time.time() + 3
    while calls["n"] < 3 and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    task.join(timeout=2)

    assert calls["n"] >= 3                 # it kept going
    assert task.state.failures == 1        # and it remembered the failure
    assert task.state.runs >= 3


def test_the_scheduler_starts_its_loops_and_stops_them(env, fake_market, monkeypatch):
    from pipeline import scheduler as sched_mod

    ran = {"cycles": 0}
    monkeypatch.setattr(sched_mod.Scheduler, "_cycle",
                        lambda self: ran.__setitem__("cycles", ran["cycles"] + 1))

    s = sched_mod.Scheduler(mode="PAPER")
    s.start()
    try:
        names = {t.state.name for t in s.tasks}
        # engine_2's drift monitor rides along whenever engine_2 is enabled; its
        # heavy training loop only appears with ENGINE_2_AUTO_TRAIN=1.
        assert names == {"decision_cycle", "maintenance", "engine_3_training",
                         "engine_2_drift"}
        assert "engine_2_training" not in names
        assert s.status()["running"] is True
        deadline = time.time() + 3
        while ran["cycles"] == 0 and time.time() < deadline:
            time.sleep(0.05)
        assert ran["cycles"] >= 1          # the first cycle runs immediately
    finally:
        s.stop()
    assert s.status()["running"] is False


def test_maintenance_writes_a_heartbeat_and_drains_the_outbox(env, fake_market):
    import core.db as db
    from core import repository
    from pipeline.scheduler import Scheduler

    db.spool({"kind": "system_event",
              "payload": {"message": "stranded", "level": "info", "category": "test"}})
    assert db.outbox_size() == 1

    Scheduler(mode="PAPER")._maintenance()

    assert db.outbox_size() == 0
    beat = repository.get_state("heartbeat")
    assert beat and beat["mode"] == "PAPER" and beat["db_ok"] is True
    assert any("stranded" in e["message"] for e in repository.recent_events(20))
