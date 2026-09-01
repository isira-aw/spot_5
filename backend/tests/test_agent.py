"""The Agent's job is to be persuasive; the constitution's job is to be unmoved."""
from core.contracts import (BUY, HOLD, SELL, AdminRestrictions, PortfolioState, Position,
                            RiskAssessment)
from tests.conftest import StubLLM

PRICE = 65000.0


def _portfolio(in_position=False, **kw):
    pos = Position(symbol="BTC/USDT", quantity=0.03, avg_entry_price=64000,
                   stop_price=63000, target_price=68000) if in_position else None
    defaults = dict(mode="PAPER", cash=10000.0, equity=10000.0, position=pos,
                    last_price=PRICE, total_trades=20, win_rate=55.0)
    defaults.update(kw)
    return PortfolioState(**defaults)


def _agent(payload, ok=True):
    from llm_agent.agent import TradingAgent
    from llm_agent.knowledge_base import KnowledgeBaseStore
    stub = StubLLM(payload, ok=ok)
    return TradingAgent(client=stub, kb_store=KnowledgeBaseStore()), stub


GOOD_BUY = {"action": "BUY", "confidence": 0.72, "size_pct": 12, "entry_price": PRICE,
            "stop_price": 63700, "target_price": 68000, "time_horizon": "6-18 hours",
            "engine_agreement": "aligned", "rationale": "Both engines agree and the risk "
            "model is comfortable, so I am taking a measured long.",
            "key_risks": ["A failed breakout"], "change_my_mind": ["A close under 63,700"],
            "used_theories": ["Trend Following and Momentum"]}


def test_the_prompt_carries_the_restrictions_the_engines_and_the_theory(env, signals):
    agent, stub = _agent(GOOD_BUY)
    rules = AdminRestrictions(version=7, notes=["No trading during CPI releases."])
    agent.decide(symbol="BTC/USDT", mode="PAPER", price=PRICE, signals=signals,
                 risk=RiskAssessment(win_probability=0.6, expected_r=0.5, size_multiplier=0.9),
                 portfolio=_portfolio(), restrictions=rules)
    prompt = stub.last_user
    assert "Never risk more than 2.00% of equity" in prompt
    assert "No trading during CPI releases." in prompt
    assert "### Engine 1" in prompt and "### Engine 2" in prompt
    assert "Engine 3 — risk model" in prompt
    assert "## Decision Discipline for This System" in prompt   # pinned rules section
    assert "PAPER mode — simulated money" in prompt


def test_an_oversized_request_is_capped_and_the_cap_is_recorded(env, signals):
    agent, _ = _agent({**GOOD_BUY, "size_pct": 90})
    d = agent.decide(symbol="BTC/USDT", mode="PAPER", price=PRICE, signals=signals,
                     risk=RiskAssessment(win_probability=0.6, expected_r=0.5,
                                         size_multiplier=1.0),
                     portfolio=_portfolio(), restrictions=AdminRestrictions())
    assert d.action == BUY
    assert d.size_pct <= 25.0
    assert any("capped to" in n for n in d.compliance_notes)
    assert d.size_quote <= 10000.0


def test_an_inverted_stop_and_target_are_repaired(env, signals):
    agent, _ = _agent({**GOOD_BUY, "stop_price": 66000, "target_price": 64000})
    d = agent.decide(symbol="BTC/USDT", mode="PAPER", price=PRICE, signals=signals,
                     risk=RiskAssessment(win_probability=0.6, expected_r=0.4),
                     portfolio=_portfolio(), restrictions=AdminRestrictions())
    assert d.stop_price < d.entry_price < d.target_price
    assert len(d.compliance_notes) >= 2


def test_a_risk_engine_veto_overrules_the_model(env, signals):
    agent, _ = _agent(GOOD_BUY)
    veto = RiskAssessment(win_probability=0.41, expected_r=-0.2, veto=True,
                          veto_reasons=["Expected value is negative."])
    d = agent.decide(symbol="BTC/USDT", mode="PAPER", price=PRICE, signals=signals,
                     risk=veto, portfolio=_portfolio(), restrictions=AdminRestrictions())
    assert d.action == HOLD and d.size_quote == 0.0
    assert "vetoed" in " ".join(d.compliance_notes)


def test_the_kill_switch_blocks_entries_but_not_exits(env, signals):
    rules = AdminRestrictions(kill_switch=True)
    agent, _ = _agent(GOOD_BUY)
    blocked = agent.decide(symbol="BTC/USDT", mode="PAPER", price=PRICE, signals=signals,
                           risk=RiskAssessment(win_probability=0.6), portfolio=_portfolio(),
                           restrictions=rules)
    assert blocked.action == HOLD and "kill switch" in " ".join(blocked.compliance_notes)

    agent2, _ = _agent({"action": "SELL", "confidence": 0.8, "rationale": "Out."})
    allowed = agent2.decide(symbol="BTC/USDT", mode="PAPER", price=PRICE, signals=signals,
                            risk=RiskAssessment(win_probability=0.4),
                            portfolio=_portfolio(in_position=True), restrictions=rules)
    assert allowed.action == SELL


def test_selling_with_nothing_held_becomes_hold(env, signals):
    agent, _ = _agent({"action": "SELL", "confidence": 0.9, "rationale": "Bearish."})
    d = agent.decide(symbol="BTC/USDT", mode="PAPER", price=PRICE, signals=signals,
                     risk=RiskAssessment(), portfolio=_portfolio(), restrictions=AdminRestrictions())
    assert d.action == HOLD and "nothing is held" in " ".join(d.compliance_notes)


def test_confidence_below_the_floor_is_not_traded(env, signals):
    agent, _ = _agent({**GOOD_BUY, "confidence": 0.4})
    d = agent.decide(symbol="BTC/USDT", mode="PAPER", price=PRICE, signals=signals,
                     risk=RiskAssessment(win_probability=0.6), portfolio=_portfolio(),
                     restrictions=AdminRestrictions(min_confidence=0.55))
    assert d.action == HOLD and "below the" in " ".join(d.compliance_notes)


def test_a_dead_llm_still_produces_a_full_answer(env, signals):
    agent, _ = _agent(None, ok=False)
    d = agent.decide(symbol="BTC/USDT", mode="PAPER", price=PRICE, signals=signals,
                     risk=RiskAssessment(win_probability=0.6, expected_r=0.5,
                                         size_multiplier=0.9),
                     portfolio=_portfolio(), restrictions=AdminRestrictions())
    assert d.source == "fallback"
    assert d.rationale and d.change_my_mind and d.key_risks
    assert d.action in (BUY, HOLD, SELL)


def test_a_degraded_engine_is_flagged_and_named_in_the_prompt(env, signals):
    from core.contracts import EngineSignal
    broken = [signals[0], EngineSignal.failed("engine_2", "BTC/USDT", "no model bundle")]
    agent, stub = _agent(GOOD_BUY)
    d = agent.decide(symbol="BTC/USDT", mode="PAPER", price=PRICE, signals=broken,
                     risk=RiskAssessment(win_probability=0.55), portfolio=_portfolio(),
                     restrictions=AdminRestrictions())
    assert d.degraded is True
    assert "DOWN — no model bundle" in stub.last_user


def test_the_decision_records_which_knowledge_base_and_rules_produced_it(env, signals):
    agent, _ = _agent(GOOD_BUY)
    d = agent.decide(symbol="BTC/USDT", mode="PAPER", price=PRICE, signals=signals,
                     risk=RiskAssessment(win_probability=0.6), portfolio=_portfolio(),
                     restrictions=AdminRestrictions(version=11))
    assert d.kb_version and d.admin_version == 11 and d.used_theories
