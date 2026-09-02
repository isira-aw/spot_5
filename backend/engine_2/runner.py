"""Run engine_2's long jobs in the background so the UI can drive them.

A full cycle is hours of GPU. An HTTP request cannot hold that open, so the API
starts a job here and returns immediately; the dashboard polls `status()` and
watches it progress. One job at a time — training twice concurrently would
thrash the same candidate directory and produce a bundle that is half of each.

Status lives in two places on purpose:

* in memory, so polling costs nothing and the running thread can update it every
  step;
* in `app_state.engine_2_job` via the repository, so it survives a page reload,
  a second API worker, and a restart — after which a job left `running` by a
  killed process is reported as `interrupted` rather than lying about progress.

Nothing here can place an order. The jobs it runs are data pull, training,
promotion and rollback; that is the whole surface.
"""
from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime, timezone

STATE_KEY = "engine_2_job"

_lock = threading.Lock()
_thread: threading.Thread | None = None
_status: dict = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _persist(status: dict) -> None:
    try:
        from core import repository
        repository.set_state(STATE_KEY, status, updated_by="engine_2.runner")
    except Exception:
        pass          # the in-memory copy is still authoritative for this process


def _load_persisted() -> dict:
    try:
        from core import repository
        return repository.get_state(STATE_KEY, None) or {}
    except Exception:
        return {}


def _set(**fields) -> dict:
    global _status
    with _lock:
        _status = {**_status, **fields, "updated_at": _now()}
        snapshot = dict(_status)
    _persist(snapshot)
    return snapshot


def is_running() -> bool:
    return _thread is not None and _thread.is_alive()


def status() -> dict:
    """The live job, or the last one that finished."""
    if _status:
        current = dict(_status)
        # a job that this process is no longer running but never finished died
        # with whatever killed the worker
        if current.get("state") == "running" and not is_running():
            current = _set(state="interrupted", finished_at=_now(),
                           error="the worker process stopped before the job finished")
        return current
    persisted = _load_persisted()
    if persisted.get("state") == "running":
        persisted = {**persisted, "state": "interrupted",
                     "error": "started by a process that is no longer running"}
    return persisted


def progress(step: str, detail: str = "") -> None:
    """Called from inside a job to say where it has got to."""
    steps = list(_status.get("steps", []))
    steps.append({"step": step, "detail": detail, "at": _now()})
    _set(step=step, detail=detail, steps=steps[-40:])


def start(job: str, fn, kwargs: dict | None = None, requested_by: str = "api") -> dict:
    """Launch `fn(**kwargs)` on a daemon thread. Raises if one is already running."""
    global _thread
    if is_running():
        raise RuntimeError(f"engine_2 job '{_status.get('job')}' is already running "
                           f"(started {_status.get('started_at')})")
    kwargs = kwargs or {}
    _set(job=job, state="running", started_at=_now(), finished_at=None,
         requested_by=requested_by, kwargs=kwargs, steps=[], step="starting",
         detail="", result=None, error=None)

    def _run():
        try:
            result = fn(**kwargs)
            # A gate failure is a legitimate outcome, not a crash: the cycle ran
            # and refused to promote. Say so distinctly.
            gated = isinstance(result, dict) and result.get("ok") is False
            _set(state="gated" if gated else "succeeded", finished_at=_now(),
                 result=result, step="finished")
        except Exception as exc:
            _set(state="failed", finished_at=_now(),
                 error=f"{type(exc).__name__}: {exc}",
                 traceback=traceback.format_exc()[-4000:], step="failed")
        finally:
            try:
                from core import repository
                st = status()
                repository.record_event(
                    f"engine_2 job '{job}' {st.get('state')}",
                    level="info" if st.get("state") == "succeeded" else "warning",
                    category="engine_2",
                    payload={k: st.get(k) for k in ("job", "state", "error", "step")})
            except Exception:
                pass

    _thread = threading.Thread(target=_run, name=f"engine2-{job}", daemon=True)
    _thread.start()
    time.sleep(0.05)          # let the thread mark itself started before we answer
    return status()
