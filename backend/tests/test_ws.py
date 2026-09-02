"""The live feed: the envelope, the opening snapshot, fan-out and back-pressure."""
import asyncio

import pytest


@pytest.fixture
def client(env, fake_market):
    import main
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c


def _envelope(msg, type_=None, seq=None):
    assert set(msg) == {"type", "ts", "seq", "data"}
    assert isinstance(msg["seq"], int) and msg["seq"] >= 1
    assert msg["ts"]
    if type_:
        assert msg["type"] == type_
    if seq:
        assert msg["seq"] == seq
    return msg["data"]


def test_snapshot_arrives_first_and_matches_state(client):
    rest = client.get("/state").json()
    with client.websocket_connect("/ws") as ws:
        data = _envelope(ws.receive_json(), type_="snapshot", seq=1)
    assert data["mode"] == rest["mode"] and data["symbol"] == rest["symbol"]
    assert data["portfolio"].keys() == rest["portfolio"].keys()
    assert data["restrictions"].keys() == rest["restrictions"].keys()
    assert data["restrictions"]["min_confidence"] == rest["restrictions"]["min_confidence"]
    # ... plus the extras a fresh UI needs to paint without further calls.
    assert data["health"]["ok"] is True
    assert isinstance(data["decisions"], list) and isinstance(data["equity"], list)
    assert data["cycle_seconds"] > 0


def test_published_messages_reach_the_client_in_sequence(client):
    from core.bus import publish
    with client.websocket_connect("/ws") as ws:
        _envelope(ws.receive_json(), type_="snapshot", seq=1)
        publish("price", {"price": 77296.42, "ok": True})
        publish("decision", {"action": "HOLD", "confidence": 0.25})
        assert _envelope(ws.receive_json(), type_="price", seq=2)["price"] == 77296.42
        assert _envelope(ws.receive_json(), type_="decision", seq=3)["action"] == "HOLD"


def test_every_client_gets_every_message(client):
    from core.bus import publish
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        _envelope(a.receive_json(), type_="snapshot")
        _envelope(b.receive_json(), type_="snapshot")
        publish("cycle_start", {"cycle_id": "paper-1"})
        for ws in (a, b):
            assert _envelope(ws.receive_json(), type_="cycle_start")["cycle_id"] == "paper-1"


def test_a_cycle_publishes_its_progress(client):
    from pipeline.orchestrator import get_orchestrator, reset_orchestrator
    reset_orchestrator()
    with client.websocket_connect("/ws") as ws:
        _envelope(ws.receive_json(), type_="snapshot")
        get_orchestrator("PAPER").run_cycle(autotrade=False)
        seen = []
        for _ in range(6):
            seen.append(ws.receive_json()["type"])
            if seen[-1] in ("portfolio", "event"):
                break
    assert seen[0] == "cycle_start"
    assert "decision" in seen and "portfolio" in seen


def test_disconnecting_unsubscribes(client):
    import time

    from core.bus import get_bus

    def settled(expected):
        for _ in range(100):                    # the server closes on its own thread
            if get_bus().subscribers == expected:
                return True
            time.sleep(0.02)
        return False

    assert settled(0)
    with client.websocket_connect("/ws") as ws:
        _envelope(ws.receive_json(), type_="snapshot")
        assert settled(1)
    assert settled(0)


# ── the bus itself ──────────────────────────────────────────────────────────
def test_a_subscriber_that_falls_behind_is_dropped_not_waited_for():
    from core.bus import Bus

    async def scenario():
        bus = Bus()
        bus.attach(asyncio.get_running_loop())
        slow, fast = bus.subscribe(), bus.subscribe()
        for i in range(bus.subscribe().queue.maxsize * 3):
            bus.publish("price", {"i": i})
            await asyncio.sleep(0)
        drained = [fast.queue.get_nowait() for _ in range(fast.queue.qsize())]
        return slow.dropped, fast.dropped, drained

    slow_dropped, fast_dropped, drained = asyncio.run(scenario())
    assert slow_dropped and fast_dropped        # neither drained: both fall behind
    assert drained[-1]["type"] == "_overflow"   # ... and are told so, not blocked


def test_publishing_without_a_loop_is_a_no_op():
    from core.bus import Bus
    Bus().publish("price", {"price": 1.0})      # the CLI path: must not raise
