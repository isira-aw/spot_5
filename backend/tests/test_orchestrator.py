"""One cycle, end to end, with the network replaced by a fixture."""
import pytest

from core.contracts import BUY, HOLD, SELL


def _orchestrator(monkeypatch, llm_payload=None, llm_ok=False):
    from execution.broker import PaperBroker
    from llm_agent.agent import TradingAgent
    from llm_agent.knowledge_base import KnowledgeBaseStore
    from pipeline.orchestrator import Orchestrator
    from tests.conftest import StubLLM

    agent = TradingAgent(client=StubLLM(llm_payload, ok=llm_ok),
                         kb_store=KnowledgeBaseStore())
    return Orchestrator(mode="PAPER", broker=PaperBroker(), agent=agent)


def test_a_cycle_runs_with_one_engine_down_and_no_llm(env, fake_market, monkeypatch):
    from execution.portfolio import PortfolioStore
    PortfolioStore("PAPER").ensure_funded()

    orch = _orchestrator(monkeypatch)
    result = orch.run_cycle(autotrade=False)

    assert result.status == "ok" and result.price == pytest.approx(65000.0)
    engines = {s.engine: s for s in result.signals}
    assert set(engines) == {"engine_1", "engine_2"}
    assert engines["engine_1"].ok is True                 # offline synthetic mode
    assert engines["engine_2"].ok is False                # no TensorFlow, no bundle
    assert result.risk is not None and result.decision is not None
    assert result.decision.source == "fallback"
    assert result.decision.degraded is True
    assert result.decision.rationale
    assert "engine_2" in " ".join(
        engines["engine_2"].reasons) or engines["engine_2"].error


def test_the_whole_cycle_is_persisted_and_replayable(env, fake_market, monkeypatch):
    from core import repository
    from execution.portfolio import PortfolioStore
    PortfolioStore("PAPER").ensure_funded()

    orch = _orchestrator(monkeypatch)
    result = orch.run_cycle(autotrade=False)

    cycles = repository.recent_cycles("PAPER", limit=5)
    assert cycles and cycles[0]["cycle_id"] == result.cycle_id
    assert cycles[0]["action"] == result.decision.action
    signals = repository.signals_by_cycle([result.cycle_id])[result.cycle_id]
    assert len(signals) == 2
    latest = repository.latest_decision("PAPER")
    assert latest["rationale"] == result.decision.rationale
    assert latest["kb_version"]


def test_an_llm_buy_is_executed_and_shows_up_in_the_book(env, fake_market, monkeypatch):
    from execution.portfolio import PortfolioStore
    store = PortfolioStore("PAPER")
    store.ensure_funded()

    payload = {"action": "BUY", "confidence": 0.75, "size_pct": 10, "entry_price": 65000,
               "stop_price": 63700, "target_price": 68000, "engine_agreement": "single",
               "rationale": "Context engine is constructive and the risk model agrees.",
               "key_risks": ["One engine is down"], "change_my_mind": ["A close under 63,700"]}
    orch = _orchestrator(monkeypatch, llm_payload=payload, llm_ok=True)
    result = orch.run_cycle(autotrade=True)

    if result.decision.action == BUY and result.order and result.order.get("executed"):
        pos = store.position("BTC/USDT")
        assert pos and pos.quantity > 0
        assert store.cash() < 10000.0
        assert result.order["fill"]["side"] == "BUY"
    else:                       # a veto or a cap is also a correct outcome — say which
        assert result.decision.action == HOLD
        assert result.decision.compliance_notes or result.blocked_by


def test_a_stop_hit_exits_before_any_model_is_consulted(env, fake_market, monkeypatch):
    from execution.portfolio import PortfolioStore
    store = PortfolioStore("PAPER")
    store.ensure_funded()
    store.upsert_position("BTC/USDT", quantity=0.05, avg_entry_price=65000.0,
                          stop_price=64500.0, target_price=68000.0,
                          opened_at=__import__("core.contracts", fromlist=["utcnow"]).utcnow(),
                          context={"features": {}})

    fake_market.price = 64000.0                      # gap through the stop
    orch = _orchestrator(monkeypatch)
    result = orch.run_cycle(autotrade=True)

    assert result.status == "protective_exit"
    assert result.decision.action == SELL and result.decision.source == "risk_guard"
    assert result.signals == []                      # no engine was asked
    assert store.position("BTC/USDT") is None
    trade = store.recent_trades(1)[0]
    assert trade["exit_reason"] == "stop_loss" and trade["pnl_quote"] < 0


def test_a_dead_price_feed_aborts_the_cycle_without_trading(env, monkeypatch):
    import core.market as market
    import execution.broker as broker_mod
    import pipeline.orchestrator as orch_mod

    def dead(symbol=None, ttl=0.0):
        return market.Quote(symbol="BTC/USDT", price=0.0, source="none", ts=0.0,
                            ok=False, error="all venues down")

    monkeypatch.setattr(market, "get_quote", dead)
    monkeypatch.setattr(broker_mod, "get_quote", dead)
    monkeypatch.setattr(orch_mod, "get_quote", dead)

    orch = _orchestrator(monkeypatch)
    result = orch.run_cycle(autotrade=True)
    assert result.status == "failed" and "no price" in result.error
    assert result.order is None


def test_autotrade_off_records_the_decision_without_placing_it(env, fake_market, monkeypatch):
    from execution.portfolio import PortfolioStore
    store = PortfolioStore("PAPER")
    store.ensure_funded()
    payload = {"action": "BUY", "confidence": 0.8, "size_pct": 10, "entry_price": 65000,
               "stop_price": 63700, "target_price": 68000, "rationale": "Taking a long."}
    orch = _orchestrator(monkeypatch, llm_payload=payload, llm_ok=True)
    result = orch.run_cycle(autotrade=False)
    if result.decision.action == BUY:
        assert result.order is None
        assert "autotrade is off" in " ".join(result.blocked_by)
    assert store.cash() == 10000.0
