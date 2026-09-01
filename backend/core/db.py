"""Database connection management. One SQLite file, no server, no network.

Three things matter here and nothing else:

* **Contention is handled.** WAL plus ``busy_timeout`` lets the scheduler write
  while the API reads, and a bounded retry with exponential backoff turns a
  momentarily locked file into a pause rather than a crash.
* **Writes are never silently lost.** :func:`session_scope` can spool a failed
  unit of work to a local JSONL outbox, which :func:`replay_outbox` drains once
  the database accepts writes again.
* **Schema creation is idempotent.** ``init_db()`` is safe to run on every boot.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from .config import BASE_DIR, get_settings
from .tables import Base

log = logging.getLogger("core.db")

_engine: Engine | None = None
_Session: sessionmaker | None = None
_lock = threading.Lock()
_initialized = False
_last_wait_error = ""

OUTBOX_PATH = os.environ.get("DB_OUTBOX_PATH", os.path.join(BASE_DIR, "var", "outbox.jsonl"))
RETRYABLE = (OperationalError, DBAPIError)


# ── engine ───────────────────────────────────────────────────────────────────
def _tune_sqlite(engine: Engine, busy_timeout_ms: int) -> None:
    """Make the single file safe for the scheduler's background threads.

    Out of the box SQLite serialises everything and raises *database is locked*
    the moment the scheduler writes while a request reads. WAL lets readers and
    the writer proceed at once, and ``busy_timeout`` makes a contended write wait
    its turn instead of failing instantly. ``synchronous=NORMAL`` is the standard
    companion to WAL: durable across process crashes, which is the case that
    matters here.
    """
    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):                       # noqa: ANN001
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            cur.execute("PRAGMA foreign_keys=ON")
        finally:
            cur.close()


def get_engine() -> Engine:
    global _engine, _Session
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        s = get_settings()
        _engine = create_engine(s.db.url, future=True, echo=s.db.echo,
                                pool_pre_ping=True,
                                connect_args={"check_same_thread": False})
        _tune_sqlite(_engine, s.db.busy_timeout_ms)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        log.info("database ready: %s (%s)", s.db.safe_url(), s.db.source)
        return _engine


def get_sessionmaker() -> sessionmaker:
    get_engine()
    assert _Session is not None
    return _Session


def reset_engine() -> None:
    """Drop the pool — used by tests and after a credentials change."""
    global _engine, _Session, _initialized
    with _lock:
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:                                   # pragma: no cover
                pass
        _engine, _Session, _initialized = None, None, False


# ── retry helper ─────────────────────────────────────────────────────────────
def with_retry(fn: Callable[[], Any], *, attempts: int | None = None,
               base_delay: float = 1.0, what: str = "database call") -> Any:
    """Run ``fn`` with exponential backoff on transient connection errors."""
    attempts = attempts or get_settings().db.connect_retries
    last: Exception | None = None
    for i in range(max(1, attempts)):
        try:
            return fn()
        except RETRYABLE as exc:
            last = exc
            if i == attempts - 1:
                break
            delay = base_delay * (2 ** i)
            log.warning("%s failed (%s); retry %d/%d in %.1fs",
                        what, type(exc).__name__, i + 1, attempts, delay)
            reset_engine()
            time.sleep(delay)
    raise last if last else RuntimeError(f"{what} failed")


# ── sessions ─────────────────────────────────────────────────────────────────
@contextmanager
def session_scope(*, spool_on_failure: dict[str, Any] | None = None) -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception.

    Pass ``spool_on_failure={"kind": ..., "payload": ...}`` for writes that must
    not evaporate when the database is unreachable: the record lands in the local
    outbox and is replayed on the next successful connection.
    """
    try:
        session = get_sessionmaker()()
    except Exception as exc:
        if spool_on_failure is not None:
            spool(spool_on_failure, str(exc))
            log.error("database unavailable, spooled %s to outbox", spool_on_failure.get("kind"))
            raise DatabaseUnavailable(str(exc)) from exc
        raise
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        if spool_on_failure is not None and isinstance(exc, RETRYABLE):
            spool(spool_on_failure, str(exc))
            log.error("write failed, spooled %s to outbox", spool_on_failure.get("kind"))
            raise DatabaseUnavailable(str(exc)) from exc
        raise
    finally:
        session.close()


class DatabaseUnavailable(RuntimeError):
    """Raised when a write could not reach the database and was spooled instead."""


# ── outbox ───────────────────────────────────────────────────────────────────
def spool(record: dict[str, Any], error: str = "") -> None:
    try:
        os.makedirs(os.path.dirname(OUTBOX_PATH), exist_ok=True)
        with open(OUTBOX_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "error": error[:500], **record},
                                default=str) + "\n")
    except OSError as exc:                                       # pragma: no cover
        log.error("could not spool to outbox: %s", exc)


def outbox_size() -> int:
    try:
        with open(OUTBOX_PATH, encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def replay_outbox(handler: Callable[[dict[str, Any]], None]) -> int:
    """Feed every spooled record to ``handler``; keep whatever still fails.

    The file is rewritten with only the leftovers, so replay is safe to call on
    a timer and will never lose a record it could not apply.
    """
    if not os.path.exists(OUTBOX_PATH):
        return 0
    with open(OUTBOX_PATH, encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    if not lines:
        return 0

    leftovers, replayed = [], 0
    for line in lines:
        try:
            handler(json.loads(line))
            replayed += 1
        except Exception as exc:
            log.warning("outbox record still failing: %s", exc)
            leftovers.append(line)

    tmp = OUTBOX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(leftovers) + ("\n" if leftovers else ""))
    os.replace(tmp, OUTBOX_PATH)
    if replayed:
        log.info("replayed %d outbox record(s), %d still pending", replayed, len(leftovers))
    return replayed


# ── lifecycle ────────────────────────────────────────────────────────────────
def init_db(create: bool = True) -> bool:
    """Connect (with retries) and make sure the schema exists. Idempotent."""
    global _initialized
    if _initialized:
        return True

    def _do():
        engine = get_engine()
        if create:
            Base.metadata.create_all(engine)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    with_retry(_do, what="schema bootstrap")
    _initialized = True
    log.info("schema ready (%d tables)", len(Base.metadata.tables))
    return True


def healthcheck() -> dict[str, Any]:
    """Ping the database."""
    started = time.perf_counter()
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "url": get_settings().db.safe_url(), "outbox": outbox_size()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "url": get_settings().db.safe_url(), "outbox": outbox_size()}


def wait_for_db(timeout_s: int = 120) -> bool:
    """Make sure the database is usable before boot continues.

    A local file is either openable or it is not, so there is nothing to wait
    for in the common case — this returns on the first attempt. The retry loop
    is kept for the one case that is genuinely transient: another process
    holding the write lock while it checkpoints WAL. The last error is kept in
    :func:`last_wait_error` so the caller can say *why* it gave up.
    """
    global _last_wait_error
    _last_wait_error = ""
    deadline, delay, attempt = time.time() + timeout_s, 0.5, 0
    while True:
        attempt += 1
        result = healthcheck()
        if result["ok"]:
            if attempt > 1:
                log.info("database answered on attempt %d", attempt)
            return True
        _last_wait_error = str(result.get("error", ""))
        remaining = deadline - time.time()
        if remaining <= 0:
            log.error("database not ready after %ds: %s", timeout_s, _last_wait_error)
            return False
        pause = min(delay, remaining)
        log.warning("database not ready (%s); retrying in %.1fs", _last_wait_error, pause)
        reset_engine()
        time.sleep(pause)
        delay = min(delay * 2, 5.0)


def last_wait_error() -> str:
    """The last error :func:`wait_for_db` saw, for the caller's own message."""
    return _last_wait_error
