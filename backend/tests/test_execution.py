"""Brokers, books and the guard. The parts where a bug costs money."""
import pytest

from core.contracts import AdminRestrictions, AgentDecision


# ── paper broker ────────────────────────────────────────────────────────────
def test_paper_fills_pay_fees_and_slippage_on_both_legs(env):
    from execution.broker import PaperBroker
    b = PaperBroker(fee_rate=0.00075, slippage_pct=0.0005)
    buy = b.buy("BTC/USDT", 1000.0, client_order_id="a", reference_price=65000.0)
    sell = b.sell("BTC/USDT", buy.quantity, client_order_id="b", reference_price=65000.0)
    round_trip = buy.quote_amount + sell.quote_amount
    assert buy.price > 65000.0 > sell.price          # slippage cuts both ways
    assert -2.6 < round_trip < -2.4                  # ~0.25% on a flat market
    assert buy.broker == "paper" and buy.raw["simulated"] is True


def test_live_broker_refuses_to_exist_without_an_explicit_confirmation(env, monkeypatch):
    from execution.broker import BrokerError, LiveBroker
    with pytest.raises(BrokerError, match="LIVE_TRADING_CONFIRMED"):
        LiveBroker()
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "1")
    from core.config import get_settings
    get_settings(refresh=True)
    with pytest.raises(BrokerError, match="EXCHANGE_API_KEY"):
        LiveBroker()


# ── the books ───────────────────────────────────────────────────────────────
def test_a_full_round_trip_updates_ledger_position_and_trade(env):
    from execution.broker import PaperBroker
    from execution.portfolio import PortfolioStore
    from execution.trader import Trader

    store = PortfolioStore("PAPER")
    assert store.ensure_funded() == 10000.0
    trader = Trader(PaperBroker(), store)
    rules = AdminRestrictions()

    buy = AgentDecision(action="BUY", confidence=0.7, size_pct=20, size_quote=2000,
                        entry_price=65000.0, stop_price=64000.0, target_price=67000.0)
    report = trader.execute(buy, symbol="BTC/USDT", price=65000.0,
                            portfolio=store.state("BTC/USDT", 65000.0), restrictions=rules,
                            cycle_id="c1", decision_id=1,
                            context={"features": {"e1_signed_conf": 0.6}})
    assert report["executed"]
    held = store.position("BTC/USDT")
    assert held and held.quantity > 0 and held.stop_price == 64000.0
    assert store.cash() == pytest.approx(8000.0)

    after = store.state("BTC/USDT", 67000.0)
    sell = AgentDecision(action="SELL", confidence=0.8)
    closed = trader.execute(sell, symbol="BTC/USDT", price=67000.0, portfolio=after,
                            restrictions=rules, cycle_id="c2", decision_id=2,
                            reason="take_profit")
    assert closed["executed"] and closed["pnl_quote"] > 0
    assert closed["r_multiple"] > 1.0
    assert store.position("BTC/USDT") is None

    trade = store.recent_trades(1)[0]
    assert trade["exit_reason"] == "take_profit"
    assert trade["context"]["features"]["e1_signed_conf"] == 0.6   # engine_3's training row


def test_a_replayed_cycle_does_not_fill_twice(env):
    from core.db import session_scope
    from core.tables import Order
    from execution.broker import PaperBroker
    from execution.portfolio import PortfolioStore
    from execution.trader import Trader

    store = PortfolioStore("PAPER")
    store.ensure_funded()
    trader = Trader(PaperBroker(), store)
    with session_scope() as s:                       # an order that landed before a crash
        s.add(Order(mode="PAPER", cycle_id="c9", symbol="BTC/USDT", side="BUY",
                    quantity=0.01, price=65000, quote_amount=-650,
                    client_order_id="c9-buy", status="filled"))
    buy = AgentDecision(action="BUY", confidence=0.7, size_pct=10, size_quote=1000,
                        entry_price=65000.0, stop_price=64000.0)
    report = trader.execute(buy, symbol="BTC/USDT", price=65000.0,
                            portfolio=store.state("BTC/USDT", 65000.0),
                            restrictions=AdminRestrictions(), cycle_id="c9")
    assert report["idempotent"] is True
    assert store.cash() == pytest.approx(10000.0)    # no second fill


def test_the_two_books_never_touch(env):
    from execution.broker import PaperBroker
    from execution.portfolio import PortfolioStore
    from execution.trader import Trader

    paper = PortfolioStore("PAPER")
    paper.ensure_funded()
    Trader(PaperBroker(), paper).execute(
        AgentDecision(action="BUY", confidence=0.7, size_pct=10, size_quote=1000,
                      entry_price=65000.0, stop_price=64000.0),
        symbol="BTC/USDT", price=65000.0, portfolio=paper.state("BTC/USDT", 65000.0),
        restrictions=AdminRestrictions(), cycle_id="c1")

    real = PortfolioStore("REAL")
    assert real.cash() == 0.0
    assert real.position("BTC/USDT") is None
    assert real.stats()["trades"] == 0
    assert paper.cash() == pytest.approx(9000.0)


# ── the guard ───────────────────────────────────────────────────────────────
def test_the_guard_sizes_from_whichever_limit_binds_first(env):
    """Position cap and risk budget are both ceilings; the smaller one wins."""
    from core.contracts import PortfolioState
    from execution import risk_guard
    pf = PortfolioState(mode="PAPER", cash=10000, equity=10000, last_price=65000)

    # A tight 1% stop: 2% of equity at risk would allow 20,000, so the 25%
    # position cap is what actually limits the order.
    tight = AgentDecision(action="BUY", confidence=0.9, size_pct=25, size_quote=2500,
                          entry_price=65000.0, stop_price=64350.0)
    g = risk_guard.check(tight, portfolio=pf, restrictions=AdminRestrictions(),
                         price=65000.0, symbol="BTC/USDT")
    assert g.allowed and g.quote == pytest.approx(2500.0)
    assert g.quantity == pytest.approx(2500.0 / 65000.0)

    # A wide 5% stop with a 1% risk budget: now the risk budget binds at 2,000.
    wide = AgentDecision(action="BUY", confidence=0.9, size_pct=25, size_quote=2500,
                         entry_price=65000.0, stop_price=61750.0)
    g2 = risk_guard.check(wide, portfolio=pf,
                          restrictions=AdminRestrictions(max_capital_at_risk_pct=1.0),
                          price=65000.0, symbol="BTC/USDT")
    assert g2.allowed and g2.quote == pytest.approx(2000.0)


def test_the_guard_blocks_every_way_an_entry_can_be_wrong(env):
    from core.contracts import PortfolioState
    from execution import risk_guard
    pf = PortfolioState(mode="PAPER", cash=10000, equity=10000, last_price=65000,
                        trades_today=99)
    d = AgentDecision(action="BUY", confidence=0.2, size_pct=25, size_quote=2500,
                      entry_price=65000.0, stop_price=64000.0)
    g = risk_guard.check(d, portfolio=pf, restrictions=AdminRestrictions(),
                         price=65000.0, symbol="BTC/USDT")
    assert not g.allowed
    joined = " ".join(g.reasons)
    assert "daily trade cap" in joined and "confidence" in joined


def test_the_kill_switch_stops_entries_and_leaves_exits_alone(env):
    from core.contracts import PortfolioState, Position
    from execution import risk_guard
    risk_guard.set_kill_switch(True, by="test", reason="drill")
    assert risk_guard.is_killed()

    pos = Position(symbol="BTC/USDT", quantity=0.03, avg_entry_price=64000)
    pf = PortfolioState(mode="PAPER", cash=1000, equity=3000, position=pos, last_price=65000)
    entry = risk_guard.check(
        AgentDecision(action="BUY", confidence=0.9, size_quote=500, entry_price=65000,
                      stop_price=64000),
        portfolio=pf, restrictions=AdminRestrictions(), price=65000.0, symbol="BTC/USDT")
    assert not entry.allowed and "kill switch" in " ".join(entry.reasons)

    exit_ = risk_guard.check(AgentDecision(action="SELL", confidence=0.9), portfolio=pf,
                             restrictions=AdminRestrictions(), price=65000.0, symbol="BTC/USDT")
    assert exit_.allowed and exit_.quantity == pytest.approx(0.03)


def test_protective_exit_fires_on_the_stop_and_on_the_target(env):
    from core.contracts import PortfolioState, Position
    from execution import risk_guard
    pos = Position(symbol="BTC/USDT", quantity=0.03, avg_entry_price=65000,
                   stop_price=64000, target_price=67000)
    pf = PortfolioState(mode="PAPER", position=pos, equity=10000, last_price=65000)
    assert risk_guard.protective_exit(pf, 63900) == "stop_loss"
    assert risk_guard.protective_exit(pf, 67100) == "take_profit"
    assert risk_guard.protective_exit(pf, 65500) is None
