"""A single-writer guarantee that works across machines.

The migration story for this system is "point the new host at the same database
and start it". That is only safe if two hosts cannot both be trading the same
account at the same time — otherwise the day the old box is finally shut down is
the day it placed one last order.

A Postgres session-level advisory lock does the job: it is held on one dedicated
connection, it is released automatically if that connection dies (so a hard
power-off frees it), and it costs nothing. An instance that cannot get the lock
keeps running in observer mode — it serves the API and reads the books, it just
does not trade.
"""
from __future__ import annotations

import logging
import zlib
from typing import Any

from .db import get_engine

log = logging.getLogger("core.locks")

TRADING_LOCK = "spot5:trading"


def _key(name: str) -> int:
    """Stable 63-bit key derived from the lock name."""
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


class AdvisoryLock:
    def __init__(self, name: str = TRADING_LOCK):
        self.name = name
        self.key = _key(name)
        self._conn: Any = None
        self.held = False
        self.supported = True

    def acquire(self) -> bool:
        if self.held:
            return True
        engine = get_engine()
        if engine.dialect.name != "postgresql":
            self.supported = False
            self.held = True                      # nothing to coordinate on sqlite
            return True
        from sqlalchemy import text
        try:
            self._conn = engine.connect()
            got = bool(self._conn.execute(
                text("SELECT pg_try_advisory_lock(:k)"), {"k": self.key}).scalar())
            if not got:
                self._conn.close()
                self._conn = None
                log.warning("another instance holds the %s lock — running as observer "
                            "(no trading from this process)", self.name)
            else:
                log.info("acquired the %s lock; this instance is the trader", self.name)
            self.held = got
            return got
        except Exception as exc:
            log.error("advisory lock unavailable (%s); continuing without it", exc)
            self.supported = False
            self.held = True
            return True

    def release(self) -> None:
        if self._conn is not None:
            from sqlalchemy import text
            try:
                self._conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": self.key})
            except Exception:                                     # pragma: no cover
                pass
            finally:
                self._conn.close()
                self._conn = None
        self.held = False

    def status(self) -> dict:
        return {"name": self.name, "key": self.key, "held": self.held,
                "supported": self.supported}
