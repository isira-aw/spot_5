"""A single-writer guarantee for the trading loop.

Two processes must never trade the same account at once — otherwise the day the
old one is finally shut down is the day it placed one last order. The database
is a single file on this machine, so the lock is a file lock next to it: the OS
holds it, and it is released automatically if the process dies, including a hard
power-off.

An instance that cannot get the lock keeps running in observer mode — it serves
the API and reads the books, it just does not trade.
"""
from __future__ import annotations

import logging
import os
import zlib
from typing import Any

from .config import BASE_DIR, get_settings

log = logging.getLogger("core.locks")

TRADING_LOCK = "spot5:trading"


def _key(name: str) -> int:
    """Stable 63-bit key derived from the lock name. Kept for the status view."""
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def _lock_path(name: str) -> str:
    safe = name.replace(":", "-")
    return os.path.join(BASE_DIR, "var", f"{safe}.lock")


class AdvisoryLock:
    """Exclusive, non-blocking, auto-released on process exit."""

    def __init__(self, name: str = TRADING_LOCK):
        self.name = name
        self.key = _key(name)
        self.path = _lock_path(name)
        self._fh: Any = None
        self.held = False
        self.supported = True

    def acquire(self) -> bool:
        if self.held:
            return True
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            fh = open(self.path, "a+")
            _lock_file(fh)
        except OSError as exc:
            try:
                fh.close()                                    # noqa: F821
            except Exception:                                 # pragma: no cover
                pass
            log.warning("another instance holds the %s lock — running as observer "
                        "(no trading from this process): %s", self.name, exc)
            self.held = False
            return False
        except Exception as exc:                              # pragma: no cover
            log.error("lock unavailable (%s); continuing without it", exc)
            self.supported = False
            self.held = True
            return True

        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()} {get_settings().instance_id}\n")
        fh.flush()
        self._fh = fh
        self.held = True
        log.info("acquired the %s lock; this instance is the trader", self.name)
        return True

    def release(self) -> None:
        if self._fh is not None:
            try:
                _unlock_file(self._fh)
            except Exception:                                 # pragma: no cover
                pass
            finally:
                try:
                    self._fh.close()
                except Exception:                             # pragma: no cover
                    pass
                self._fh = None
        self.held = False

    def status(self) -> dict:
        return {"name": self.name, "key": self.key, "held": self.held,
                "supported": self.supported, "path": self.path}


# ── platform primitives ──────────────────────────────────────────────────────
if os.name == "nt":                                           # pragma: no cover
    import msvcrt

    def _lock_file(fh) -> None:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock_file(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
else:                                                         # pragma: no cover
    import fcntl

    def _lock_file(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_file(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
