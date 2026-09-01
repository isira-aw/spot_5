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
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from .config import BASE_DIR, get_settings, is_internal_host, running_inside_railway
from .tables import Base

log = logging.getLogger("core.db")

_engine: Engine | None = None
_Session: sessionmaker | None = None
_lock = threading.Lock()
_initialized = False

OUTBOX_PATH = os.environ.get("DB_OUTBOX_PATH", os.path.join(BASE_DIR, "var", "outbox.jsonl"))
RETRYABLE = (OperationalError, DBAPIError)


# ── engine ───────────────────────────────────────────────────────────────────
def _connect_args(url: str) -> dict[str, Any]:
    s = get_settings()
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    args: dict[str, Any] = {
        "connect_timeout": s.db.connect_timeout_s,
        "application_name": f"spot5-{s.env}",
    }
    # Railway's managed Postgres speaks TLS; "prefer" keeps local dev working.
    if s.db.sslmode:
        args["sslmode"] = s.db.sslmode
    if s.db.statement_timeout_ms:
        args["options"] = f"-c statement_timeout={s.db.statement_timeout_ms}"
    return args


def get_engine() -> Engine:
    global _engine, _Session
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        s = get_settings()
        url = s.db.url
        if not url:
            raise RuntimeError("No database configured: set DATABASE_URL or the PG* variables.")
        kwargs: dict[str, Any] = {
            "future": True,
            "echo": s.db.echo,
            "pool_pre_ping": True,
            "connect_args": _connect_args(url),
        }
        if not url.startswith("sqlite"):
            kwargs.update(pool_size=s.db.pool_size, max_overflow=s.db.max_overflow,
                          pool_recycle=s.db.pool_recycle_s, pool_timeout=30)
        _engine = create_engine(url, **kwargs)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        log.info("database engine ready: %s", s.db.safe_url())
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


def healthcheck() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "url": get_settings().db.safe_url(), "outbox": outbox_size()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "url": get_settings().db.safe_url(), "outbox": outbox_size()}


def probe_host(host: str, port: int, timeout: float = 5.0) -> tuple[str, str]:
    """-> (verdict, detail). Separates "wrong name" from "unreachable port".

    A DNS failure and a refused connection produce the same "did not answer" in a
    connection pool, but they have completely different fixes, so they are worth
    telling apart before guessing.
    """
    import socket
    try:
        addrs = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return "dns_failed", str(exc)
    ip = addrs[0][4][0]
    sock = socket.socket(addrs[0][0], socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(addrs[0][4])
        return "open", ip
    except socket.timeout:
        return "timeout", ip
    except OSError as exc:
        return "refused", f"{ip}: {exc}"
    finally:
        sock.close()


def connection_hint(url: str | None = None, driver_error: str = "") -> str:
    """Turn "the database did not answer" into something actionable.

    A connection timeout has half a dozen causes that look identical in the log,
    so this probes DNS and TCP directly and names the one that actually applies.
    """
    url = url or get_settings().db.url
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    port = parsed.port or 5432

    if url.startswith("sqlite"):
        return f"sqlite file at {parsed.path or url} could not be opened."
    if not host:
        return ("no hostname could be parsed out of the connection URL. Check "
                "DATABASE_URL in backend/.env — and note PGHOST wants a hostname, not "
                "the PGDATA directory path.")

    if is_internal_host(url) and not running_inside_railway():
        return (f"'{host}' is Railway's PRIVATE hostname: it only resolves from inside "
                f"Railway, never from your machine. Use the public URL instead — in the "
                f"Railway dashboard open the Postgres service, Variables tab, and copy "
                f"DATABASE_PUBLIC_URL (host looks like <name>.proxy.rlwy.net with a high "
                f"port). Set it as DATABASE_URL in backend/.env, or leave both and this "
                f"process will pick the public one automatically when it is off-platform.")

    verdict, detail = probe_host(host, port)

    if verdict == "dns_failed":
        extra = ""
        if host.endswith(".proxy.rlwy.net"):
            extra = (" Railway's proxy hostnames are randomly assigned railway words "
                     "(monorail, shortline, viaduct, roundhouse, junction...) — copy the "
                     "exact host AND port out of DATABASE_PUBLIC_URL in the Railway "
                     "dashboard rather than typing one that looks plausible.")
        return (f"'{host}' does not resolve — this hostname does not exist ({detail})."
                f"{extra}")
    if verdict == "refused":
        if host in ("localhost", "127.0.0.1", "::1"):
            return (f"nothing is listening on {host}:{port}. Start a local Postgres, point "
                    f"DATABASE_URL at a remote one, or use DATABASE_URL=sqlite:///./spot5.db "
                    f"to try the system without a database server.")
        return (f"'{host}' resolves but nothing accepted a connection on port {port} "
                f"({detail}). Check the port matches the one the provider shows — for "
                f"Railway the public port is a random high port, never 5432 — and that "
                f"TCP proxying is enabled for the service.")
    if verdict == "timeout":
        return (f"'{host}' ({detail}) resolves but port {port} did not respond in time. "
                f"Usually a firewall, VPN or corporate network blocking outbound traffic "
                f"on a non-standard port.")

    # The socket opened, so this is Postgres itself refusing: credentials, TLS or database.
    detail = driver_error or ("Check the username, password and database name, and that "
                              "PGSSLMODE=require matches what the server expects.")
    return (f"'{host}:{port}' is reachable, so the network is fine and Postgres itself "
            f"rejected the connection. {detail}")


def wait_for_db(timeout_s: int = 120) -> bool:
    """Block until Postgres answers or the timeout expires. Used at boot."""
    deadline, delay = time.time() + timeout_s, 1.0
    hinted = False
    while time.time() < deadline:
        health = healthcheck()
        if health["ok"]:
            return True
        if not hinted:
            log.error("database error: %s", health.get("error", "unknown"))
            log.error("diagnosis: %s", connection_hint(driver_error=health.get("error", "")))
            hinted = True
        log.warning("database not ready, retrying in %.0fs", delay)
        reset_engine()
        time.sleep(delay)
        delay = min(delay * 2, 15.0)
    return False


if __name__ == "__main__":                                        # pragma: no cover
    # A five-second connection diagnosis, without booting the rest of the system:
    #     python -m core.db
    from .logging_setup import setup_logging

    setup_logging()
    settings = get_settings()
    health = healthcheck()
    print(json.dumps({
        "url": settings.db.safe_url(),
        "reachable": health["ok"],
        "driver_error": health.get("error"),
        "diagnosis": "connected" if health["ok"] else connection_hint(
            driver_error=health.get("error", "")),
    }, indent=2))
    raise SystemExit(0 if health["ok"] else 1)
