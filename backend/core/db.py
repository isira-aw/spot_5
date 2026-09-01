"""Postgres connection management built for a box that will lose its network.

Three things matter here and nothing else:

* **Reconnects are automatic.** ``pool_pre_ping`` plus a bounded retry with
  exponential backoff means a Railway restart or a dropped Wi-Fi link produces a
  pause, not a crash.
* **Writes are never silently lost.** :func:`session_scope` can spool a failed
  unit of work to a local JSONL outbox, which :func:`replay_outbox` drains once
  the database answers again.
* **Schema creation is idempotent.** ``init_db()`` is safe to run on every boot
  from every instance.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import struct
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
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
def _connect_args(url: str, connect_timeout_s: int | None = None) -> dict[str, Any]:
    s = get_settings()
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    args: dict[str, Any] = {
        "connect_timeout": min(connect_timeout_s or s.db.connect_timeout_s,
                               s.db.connect_timeout_s),
        "application_name": f"spot5-{s.env}",
    }
    # Railway's managed Postgres speaks TLS; "prefer" keeps local dev working.
    if s.db.sslmode:
        args["sslmode"] = s.db.sslmode
    if s.db.statement_timeout_ms:
        args["options"] = f"-c statement_timeout={s.db.statement_timeout_ms}"
    return args


def get_engine(connect_timeout_s: int | None = None) -> Engine:
    global _engine, _Session
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        s = get_settings()
        url = s.db.url
        if not url:
            from .config import database_hint
            raise RuntimeError(f"No database configured: {database_hint()}")
        kwargs: dict[str, Any] = {
            "future": True,
            "echo": s.db.echo,
            "pool_pre_ping": True,
            "connect_args": _connect_args(url, connect_timeout_s),
        }
        if not url.startswith("sqlite"):
            kwargs.update(pool_size=s.db.pool_size, max_overflow=s.db.max_overflow,
                          pool_recycle=s.db.pool_recycle_s, pool_timeout=30)
        _engine = create_engine(url, **kwargs)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        log.info("database engine ready: %s (from %s)", s.db.safe_url(), s.db.source)
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
    """Raised when a write could not reach Postgres and was spooled instead."""


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


def healthcheck(connect_timeout_s: int | None = None) -> dict[str, Any]:
    """Ping the database. ``connect_timeout_s`` clips the per-connection timeout
    for this call only — used by :func:`wait_for_db` to fit its remaining budget."""
    started = time.perf_counter()
    if connect_timeout_s is not None:
        reset_engine()
    try:
        with get_engine(connect_timeout_s=connect_timeout_s).connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "url": get_settings().db.safe_url(), "outbox": outbox_size()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "url": get_settings().db.safe_url(), "outbox": outbox_size()}


def probe_endpoint(timeout_s: float = 5.0) -> str:
    """Say *where* the connection dies, below the driver.

    ``connect_timeout expired`` is the same message whether DNS is wrong, the
    port is shut, or the port answers but nothing behind it speaks Postgres —
    and the third case is the one that looks like a working endpoint to
    ``Test-NetConnection``, because a TCP proxy completes the handshake for
    ports it has no service mapped to. So: resolve, connect, then send an 8-byte
    SSLRequest and see whether a Postgres server is actually on the far end.
    Read-only, no credentials sent, no bytes beyond the protocol preamble.
    """
    url = get_settings().db.url
    if not url or url.startswith("sqlite"):
        return ""
    u = urlparse(url)
    host, port = u.hostname, u.port or 5432
    if not host:
        return ""

    try:
        socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        return f"DNS lookup for {host} failed ({exc.__class__.__name__}) — check the host in DATABASE_URL"

    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
    except OSError as exc:
        return (f"TCP connect to {host}:{port} failed ({exc.__class__.__name__}) — "
                f"the port is closed or blocked upstream")

    try:
        sock.settimeout(timeout_s)
        sock.sendall(struct.pack("!ii", 8, 80877103))       # postgres SSLRequest
        reply = sock.recv(1)
    except OSError:
        return (f"TCP to {host}:{port} connects but the server never answers the Postgres "
                f"handshake — that port is not mapped to the database (a proxy accepts the "
                f"connection for any port). Re-copy DATABASE_URL from the Railway service's "
                f"Connect tab; the public proxy port changes when the service is redeployed.")
    finally:
        try:
            sock.close()
        except OSError:                                     # pragma: no cover
            pass

    if reply in (b"S", b"N"):
        return (f"{host}:{port} is a live Postgres endpoint — the handshake gets through, so "
                f"the timeout is in authentication or TLS, not in reaching the server")
    return f"{host}:{port} answered the Postgres handshake with {reply!r}, which is not a Postgres server"


def wait_for_db(timeout_s: int = 120) -> bool:
    """Block until Postgres answers or the timeout expires. Used at boot.

    Every failed attempt is logged with the driver's own error, and the last one
    is kept in :func:`last_wait_error` so the caller can say *why* it gave up
    instead of only that it did. Attempts are also budgeted: the per-connection
    timeout is clipped to the time actually left, so a 30s wait makes several
    attempts rather than two 15s ones, and the call returns at the deadline
    instead of overshooting it by a whole backoff.
    """
    global _last_wait_error
    _last_wait_error = ""

    # A missing URL is not a transient condition — waiting cannot fix it.
    if not get_settings().db.url:
        from .config import database_hint
        _last_wait_error = database_hint()
        log.error("no database configured: %s", _last_wait_error)
        return False

    deadline, delay, attempt = time.time() + timeout_s, 1.0, 0
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        attempt += 1
        result = healthcheck(connect_timeout_s=max(1, int(remaining)))
        if result["ok"]:
            if attempt > 1:
                log.info("database answered on attempt %d", attempt)
            reset_engine()          # rebuild with the configured timeout, not the clipped one
            return True
        _last_wait_error = str(result.get("error", ""))
        remaining = deadline - time.time()
        if remaining <= 0:
            log.error("database not ready after %ds: %s", timeout_s, _last_wait_error)
            break
        pause = min(delay, remaining)
        log.warning("database not ready (%s); retrying in %.0fs", _last_wait_error, pause)
        reset_engine()
        time.sleep(pause)
        delay = min(delay * 2, 15.0)

    # Gave up. Probe once — cheap, and it turns "timeout expired" into a reason.
    hint = probe_endpoint()
    if hint:
        log.error("endpoint probe: %s", hint)
        _last_wait_error = f"{_last_wait_error} | {hint}" if _last_wait_error else hint
    return False


def last_wait_error() -> str:
    """The last error :func:`wait_for_db` saw, for the caller's own message."""
    return _last_wait_error
