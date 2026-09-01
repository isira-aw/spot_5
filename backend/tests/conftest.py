"""Test fixtures: a throwaway database, a fake market, no network, no LLM.

Every test runs against a real SQLAlchemy schema (SQLite) rather than mocks of the
repository, because the thing most worth testing about persistence is that the
schema and the queries agree with each other.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A clean process-level environment pointing at a fresh database."""
    db_path = tmp_path / "spot5.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("DB_OUTBOX_PATH", str(tmp_path / "outbox.jsonl"))
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LLM_ENABLED", "0")
    monkeypatch.setenv("ENGINE_1_OFFLINE", "1")
    monkeypatch.setenv("AUTOSTART_SCHEDULER", "0")
    monkeypatch.setenv("PAPER_STARTING_CASH", "10000")
    monkeypatch.setenv("KB_REFRESH_SECONDS", "0")
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")

    # Tests publish knowledge bases, so they get a throwaway copy of the real one.
    # Nothing in the suite is allowed to write to backend/llm_agent/.
    kb_src = os.path.join(BASE_DIR, "llm_agent", "trading_theories_knowledge_base.md")
    kb_copy = tmp_path / "knowledge_base.md"
    if os.path.exists(kb_src):
        with open(kb_src, encoding="utf-8") as fh:
            kb_copy.write_text(fh.read(), encoding="utf-8")
    monkeypatch.setenv("KB_PATH", str(kb_copy))

    import core.config as config
    import core.db as db
    config.get_settings(refresh=True)
    db.reset_engine()
    db.OUTBOX_PATH = str(tmp_path / "outbox.jsonl")
    db.init_db()

    # every module-level singleton has to forget the previous test
    import engine_3.service as e3s
    import llm_agent.agent as agent_mod
    import llm_agent.knowledge_base as kb_mod
    import pipeline.orchestrator as orch
    import pipeline.scheduler as sched
    e3s._engine = None
    agent_mod._agent = None
    kb_mod._store = None
    orch._orchestrator = None
    sched._scheduler = None

    yield config.get_settings()

    db.reset_engine()


@pytest.fixture
def fake_market(monkeypatch):
    """A deterministic price feed. Set ``fake_market.price`` to move the market."""
    import core.market as market

    class Feed:
        price = 65000.0

        def quote(self, symbol=None, ttl=0.0):
            return market.Quote(symbol=symbol or "BTC/USDT", price=self.price,
                                source="test", ts=__import__("time").time())

        def candles(self, symbol=None, interval="1h", limit=200, ttl=0.0):
            base = self.price * 0.98
            return [[i * 3_600_000, base, base * 1.005, base * 0.995,
                     base * (1 + 0.0004 * i), 100.0] for i in range(limit)]

    feed = Feed()
    monkeypatch.setattr(market, "get_quote", feed.quote)
    monkeypatch.setattr(market, "get_ohlcv", feed.candles)
    import execution.broker as broker_mod
    import pipeline.orchestrator as orch_mod
    monkeypatch.setattr(broker_mod, "get_quote", feed.quote)
    monkeypatch.setattr(orch_mod, "get_quote", feed.quote)
    monkeypatch.setattr(orch_mod, "get_ohlcv", feed.candles)
    return feed


@pytest.fixture
def signals():
    from core.contracts import EngineSignal
    return [
        EngineSignal(engine="engine_1", direction="UP", confidence=0.66, action_hint="BUY",
                     symbol="BTC/USDT",
                     features={"rsi14": 58, "atr_pct": 1.1, "agreement_pct": 80,
                               "trend_up": True, "price": 65000},
                     levels={"reference_stop": 63900, "reference_target": 67500},
                     reasons=["Higher lows on the 4h."]),
        EngineSignal(engine="engine_2", direction="UP", confidence=0.6, action_hint="BUY",
                     symbol="BTC/USDT",
                     features={"p_up": [0.62, 0.58], "decisiveness": 0.55, "close": 65000},
                     levels={"stop_loss": 64805, "take_profit_1": 65325}),
    ]


@pytest.fixture
def restrictions():
    from core.contracts import AdminRestrictions
    return AdminRestrictions()


@pytest.fixture
def utc_now():
    return datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


class StubLLM:
    """Stands in for :class:`llm_agent.client.LLMClient`."""

    def __init__(self, payload=None, ok=True):
        self.payload, self.ok = payload, ok
        self.last_system = self.last_user = None

    def complete(self, system, user):
        from llm_agent.client import LLMResponse
        self.last_system, self.last_user = system, user
        if not self.ok:
            return LLMResponse(False, error="stub offline", source="none")
        import json
        return LLMResponse(True, text=json.dumps(self.payload), parsed=self.payload,
                           source="groq:stub", latency_ms=10)
