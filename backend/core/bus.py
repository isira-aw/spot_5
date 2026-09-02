"""A tiny in-process broadcast bus.

The decision loop runs in scheduler threads; the API runs in an asyncio loop.
This is the seam between them. Producers call :func:`publish` from whatever
thread they happen to be on and never wait; each subscriber holds its own
bounded queue and a subscriber that cannot keep up is dropped rather than
allowed to slow the publisher or its siblings down.

Publishing is deliberately best-effort and silent: a websocket nobody is
listening to must never be able to break a trading cycle. When no event loop is
attached — the CLI, the tests — :func:`publish` is a no-op.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from core.contracts import utcnow

log = logging.getLogger("core.bus")

#: How many messages a slow subscriber may fall behind before it is dropped.
QUEUE_MAX = 64


class Subscriber:
    """One consumer's view of the bus: a bounded queue plus an overflow flag."""

    def __init__(self, maxsize: int = QUEUE_MAX):
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=maxsize)
        self.dropped = False

    def offer(self, message: dict) -> None:
        if self.dropped:
            return
        try:
            self.queue.put_nowait(message)
        except asyncio.QueueFull:
            # Mark and wake the consumer so it can close the connection itself.
            self.dropped = True
            try:
                self.queue.get_nowait()
                self.queue.put_nowait({"type": "_overflow"})
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass


class Bus:
    def __init__(self) -> None:
        self._subs: set[Subscriber] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── wiring ──────────────────────────────────────────────────────────────
    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the bus to the API's event loop. Called once, at startup."""
        self._loop = loop

    def detach(self) -> None:
        """Release the loop and forget every subscriber: the app is going down."""
        self._loop = None
        with self._lock:
            self._subs.clear()

    def subscribe(self) -> Subscriber:
        sub = Subscriber()
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    @property
    def subscribers(self) -> int:
        with self._lock:
            return len(self._subs)

    # ── publishing ──────────────────────────────────────────────────────────
    def publish(self, type_: str, data: Any) -> None:
        """Fan ``data`` out to every subscriber. Safe from any thread, never raises."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        message = {"type": type_, "ts": utcnow(), "data": data}
        try:
            if _on_loop(loop):
                self._fanout(message)
            else:
                loop.call_soon_threadsafe(self._fanout, message)
        except RuntimeError:
            pass                                    # loop shut down mid-publish
        except Exception:                           # pragma: no cover - never fatal
            log.exception("bus publish failed for %s", type_)

    def _fanout(self, message: dict) -> None:
        with self._lock:
            subs = tuple(self._subs)
        for sub in subs:
            sub.offer(message)


def _on_loop(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


_bus = Bus()


def get_bus() -> Bus:
    return _bus


def publish(type_: str, data: Any) -> None:
    """Module-level shorthand — this is what the pipeline calls."""
    _bus.publish(type_, data)
