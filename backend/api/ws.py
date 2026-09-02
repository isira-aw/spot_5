"""The live feed.

One endpoint, ``GET /ws``. On connect a client receives a single ``snapshot``
message carrying everything ``/state`` would have told it, so the UI can paint
without a REST round-trip, and after that only what changed. Every message uses
the same envelope::

    {"type": ..., "ts": "2026-09-02T01:15:39.000Z", "seq": 1234, "data": {...}}

``seq`` counts from 1 **per connection**, so a client that sees a gap knows it
missed something and can reconnect for a fresh snapshot.

Payloads are built by the same functions that serve the REST routes and encoded
with FastAPI's own encoder, so a ``portfolio`` frame and ``GET /state`` cannot
drift apart.

Everything the socket touches is read-only. It never triggers a cycle, an order
or a write.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder

from api import routes
from core.bus import Subscriber, get_bus
from core.config import get_settings
from core.contracts import utcnow

log = logging.getLogger("api.ws")
router = APIRouter()

PING_INTERVAL_S = 20        # how often we ask the client to prove it is there
MAX_MISSED_PONGS = 2        # ... and how many silences we tolerate before closing
PRICE_INTERVAL_S = 5
HEALTH_INTERVAL_S = 30
SNAPSHOT_DECISIONS = 20
SNAPSHOT_EQUITY = 200


# ── payloads (the same serializers the REST routes use) ─────────────────────
def _snapshot() -> dict[str, Any]:
    """Blocking; called in a worker thread. Shape = /state + /health + history."""
    state = routes.state()
    return {**state,
            "health": routes.health(),
            "decisions": routes.decisions(limit=SNAPSHOT_DECISIONS),
            "equity": routes.equity(limit=SNAPSHOT_EQUITY),
            "cycle_seconds": get_settings().cycle_seconds}


def _encode(type_: str, seq: int, data: Any, ts=None) -> dict:
    return jsonable_encoder({"type": type_, "ts": ts or utcnow(), "seq": seq, "data": data})


# ── the endpoint ────────────────────────────────────────────────────────────
@router.websocket("/ws")
async def feed(websocket: WebSocket) -> None:
    await websocket.accept()
    bus = get_bus()
    sub = bus.subscribe()
    counter = _Seq()
    pending_pongs = 0
    log.info("ws client connected (%d total)", bus.subscribers)

    try:
        snapshot = await asyncio.to_thread(_snapshot)
        await websocket.send_json(_encode("snapshot", counter.next(), snapshot))
    except WebSocketDisconnect:
        bus.unsubscribe(sub)
        return
    except Exception as exc:
        log.warning("could not build the opening snapshot: %s", exc)
        await _close(websocket, sub, code=1011)
        return

    async def pump() -> None:
        """Bus -> client."""
        while True:
            message = await sub.queue.get()
            if message.get("type") == "_overflow":
                raise _SlowClient()
            await websocket.send_json(
                _encode(message["type"], counter.next(), message["data"], ts=message["ts"]))

    async def listen() -> None:
        """Client -> us. The only thing we care about is the pong."""
        nonlocal pending_pongs
        while True:
            raw = await websocket.receive_json()
            if isinstance(raw, dict) and raw.get("type") == "pong":
                pending_pongs = 0

    async def heartbeat() -> None:
        nonlocal pending_pongs
        while True:
            await asyncio.sleep(PING_INTERVAL_S)
            if pending_pongs >= MAX_MISSED_PONGS:
                raise _Unresponsive()
            pending_pongs += 1
            await websocket.send_json(_encode("ping", counter.next(), {}))

    tasks = [asyncio.create_task(c) for c in (pump(), listen(), heartbeat())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()                       # re-raise whatever ended the connection
    except WebSocketDisconnect:
        pass
    except _SlowClient:
        log.info("dropping a websocket client that fell behind")
    except _Unresponsive:
        log.info("dropping a websocket client that missed %d pings", MAX_MISSED_PONGS)
    except Exception as exc:
        log.warning("websocket closed on error: %s", exc)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _close(websocket, sub)
        log.info("ws client gone (%d left)", bus.subscribers)


class _SlowClient(Exception):
    """The client's queue overflowed: it is not keeping up and gets dropped."""


class _Unresponsive(Exception):
    """The client stopped answering pings."""


class _Seq:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


async def _close(websocket: WebSocket, sub: Subscriber, code: int = 1000) -> None:
    get_bus().unsubscribe(sub)
    try:
        await websocket.close(code=code)
    except Exception:
        pass


# ── background publishers ───────────────────────────────────────────────────
async def _every(interval_s: int, type_: str, build) -> None:
    """Publish ``build()`` every ``interval_s`` — but only while someone listens."""
    bus = get_bus()
    while True:
        await asyncio.sleep(interval_s)
        if not bus.subscribers:
            continue                                # don't poll an exchange for nobody
        try:
            bus.publish(type_, await asyncio.to_thread(build))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("%s tick failed: %s", type_, exc)


def start_publishers() -> list[asyncio.Task]:
    return [asyncio.create_task(_every(PRICE_INTERVAL_S, "price", routes.price),
                                name="ws-price"),
            asyncio.create_task(_every(HEALTH_INTERVAL_S, "health", routes.health),
                                name="ws-health")]


async def stop_publishers(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
