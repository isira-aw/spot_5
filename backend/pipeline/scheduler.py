"""The clock. Four loops, all of them survivable.

* **decision** — a cycle every ``CYCLE_SECONDS``.
* **training** — engine_3 retrains itself on the system's own history, evaluates,
  promotes only if it wins, and prunes to the newest ten versions.
* **maintenance** — database health, outbox replay, knowledge-base refresh,
  heartbeat.
* **the trading lock** — only the instance holding the Postgres advisory lock
  trades; the others keep serving the API. That is what makes "start the new box
  before shutting down the old one" safe.

Every loop catches its own exceptions. A task that throws logs, records an event
and runs again next tick; it never takes the process with it.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from core import repository
from core.config import get_settings
from core.contracts import utcnow
from core.db import healthcheck, outbox_size, replay_outbox
from core.locks import AdvisoryLock
from engine_3 import train as engine3_train
from engine_3.service import get_risk_engine
from llm_agent.knowledge_base import get_store as get_kb_store

from .orchestrator import get_orchestrator

log = logging.getLogger("pipeline.scheduler")


@dataclass
class TaskState:
    name: str
    interval_s: int
    runs: int = 0
    failures: int = 0
    last_run: Any = None
    last_error: str | None = None
    last_duration_ms: int = 0
    enabled: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "interval_s": self.interval_s, "runs": self.runs,
                "failures": self.failures, "last_run": self.last_run,
                "last_error": self.last_error, "last_duration_ms": self.last_duration_ms,
                "enabled": self.enabled}


class PeriodicTask(threading.Thread):
    def __init__(self, name: str, interval_s: int, fn: Callable[[], Any],
                 stop_event: threading.Event, run_at_start: bool = False):
        # NB: the attribute is `_stop_event`, not `_stop` — threading.Thread uses
        # `_stop` internally and shadowing it breaks join().
        super().__init__(name=f"task-{name}", daemon=True)
        self.state = TaskState(name=name, interval_s=max(5, int(interval_s)))
        self.fn = fn
        self._stop_event = stop_event
        self.run_at_start = run_at_start

    def run(self) -> None:
        if not self.run_at_start and self._stop_event.wait(self.state.interval_s):
            return
        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                self.fn()
                self.state.last_error = None
            except Exception as exc:
                self.state.failures += 1
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("scheduled task %s failed", self.state.name)
                repository.record_event(f"scheduled task {self.state.name} failed: {exc}",
                                        level="error", category="scheduler")
            finally:
                self.state.runs += 1
                self.state.last_run = utcnow()
                self.state.last_duration_ms = int((time.perf_counter() - started) * 1000)
            if self._stop_event.wait(self.state.interval_s):
                return


class Scheduler:
    def __init__(self, mode: str | None = None):
        self.settings = get_settings()
        self.mode = (mode or self.settings.execution.mode).upper()
        self._stop = threading.Event()
        self.tasks: list[PeriodicTask] = []
        self.lock = AdvisoryLock()
        self.started_at = None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if self.tasks:
            return
        self.started_at = utcnow()
        is_trader = self.lock.acquire()
        get_kb_store().start_watcher()
        try:
            get_risk_engine().load()
        except Exception as exc:
            log.warning("risk model not loaded at boot: %s", exc)

        s = self.settings
        specs = [("maintenance", 300, self._maintenance, True),
                 ("engine_3_training", s.engines.engine_3_train_interval_s, self._train, False)]
        if is_trader:
            specs.insert(0, ("decision_cycle", s.cycle_seconds, self._cycle, True))
        else:
            log.warning("this instance is an observer: no decision cycles will run here")

        for name, interval, fn, at_start in specs:
            task = PeriodicTask(name, interval, fn, self._stop, run_at_start=at_start)
            self.tasks.append(task)
            task.start()

        repository.record_event(
            f"scheduler started ({'trader' if is_trader else 'observer'}) in {self.mode} mode",
            category="scheduler", mode=self.mode,
            payload={"tasks": [t.state.name for t in self.tasks],
                     "cycle_seconds": s.cycle_seconds})
        log.info("scheduler running: %s", ", ".join(
            f"{t.state.name} every {t.state.interval_s}s" for t in self.tasks))

    def stop(self) -> None:
        self._stop.set()
        get_kb_store().stop_watcher()
        self.lock.release()
        for t in self.tasks:
            t.join(timeout=2)
        self.tasks.clear()
        log.info("scheduler stopped")

    # ── the tasks ───────────────────────────────────────────────────────────
    def _cycle(self) -> None:
        get_orchestrator(self.mode).run_cycle()

    def _train(self) -> None:
        result = engine3_train.run(self.mode)
        if result.get("trained"):
            log.info("engine_3 retrained: v%s %s (promoted=%s)", result.get("version"),
                     result.get("kind"), result.get("promoted"))
            if result.get("promoted"):
                get_risk_engine().load(force=True)

    def _maintenance(self) -> None:
        health = healthcheck()
        if health["ok"] and outbox_size():
            replay_outbox(repository.replay_record)
        get_kb_store().refresh()
        repository.set_state("heartbeat", {
            "instance": self.settings.instance_id or "default",
            "mode": self.mode, "at": utcnow().isoformat(),
            "trader": self.lock.held, "db_ok": health["ok"],
            "outbox": health.get("outbox", 0)}, updated_by="scheduler")

    # ── introspection ───────────────────────────────────────────────────────
    def status(self) -> dict:
        return {"running": bool(self.tasks), "mode": self.mode,
                "started_at": self.started_at, "lock": self.lock.status(),
                "tasks": [t.state.to_dict() for t in self.tasks]}


_scheduler: Scheduler | None = None


def get_scheduler(mode: str | None = None) -> Scheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler(mode)
    return _scheduler
