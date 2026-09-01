"""One interface, two realities.

Everything above this line — engines, Agent, risk guard, ledgers, statistics — is
identical in PAPER and REAL. The only thing that differs is which object fills the
order, and that difference is exactly one class.

``PaperBroker`` prices fills against the same live feed the engines see and
subtracts the same fees and slippage the backtest assumes. ``LiveBroker`` sends
the order to the exchange. Neither is allowed to touch the other's ledger,
because the ledger is keyed by mode and each broker only ever reports its own.
"""
from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.config import PAPER, REAL, get_settings
from core.contracts import utcnow
from core.market import get_quote

log = logging.getLogger("execution.broker")


@dataclass
class Fill:
    symbol: str
    side: str                    # BUY | SELL
    quantity: float
    price: float                 # the price actually paid/received, after slippage
    quote_amount: float          # signed by side: what left or entered the cash balance
    fee: float = 0.0
    slippage: float = 0.0
    status: str = "filled"
    broker: str = "paper"
    client_order_id: str = ""
    exchange_order_id: str | None = None
    reference_price: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    ts: Any = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ts"] = self.ts.isoformat() if hasattr(self.ts, "isoformat") else str(self.ts)
        return d


class BrokerError(RuntimeError):
    pass


class Broker(ABC):
    mode: str = PAPER
    name: str = "abstract"

    @abstractmethod
    def price(self, symbol: str) -> float: ...

    @abstractmethod
    def balances(self) -> dict[str, float]: ...

    @abstractmethod
    def buy(self, symbol: str, quote_amount: float, *, client_order_id: str,
            reference_price: float | None = None) -> Fill: ...

    @abstractmethod
    def sell(self, symbol: str, quantity: float, *, client_order_id: str,
             reference_price: float | None = None) -> Fill: ...

    @staticmethod
    def new_client_order_id(prefix: str = "spot5") -> str:
        return f"{prefix}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

    def describe(self) -> dict:
        return {"mode": self.mode, "broker": self.name}


# ── paper ────────────────────────────────────────────────────────────────────
class PaperBroker(Broker):
    """Real prices, simulated money. Never authenticates, never can trade real funds.

    Fills are deliberately pessimistic in the same way the engine_2 backtest is:
    a buy pays slippage up, a sell takes slippage down, and both pay the fee. A
    paper curve built on mid prices with no costs is a sales brochure, not a test.
    """
    mode = PAPER
    name = "paper"

    def __init__(self, fee_rate: float | None = None, slippage_pct: float | None = None):
        ex = get_settings().execution
        self.fee_rate = ex.fee_rate if fee_rate is None else fee_rate
        self.slippage_pct = ex.slippage_pct if slippage_pct is None else slippage_pct

    def price(self, symbol: str) -> float:
        q = get_quote(symbol)
        if not q.ok or q.price <= 0:
            raise BrokerError(f"no price for {symbol}: {q.error}")
        return q.price

    def balances(self) -> dict[str, float]:
        return {}                       # the ledger is the source of truth in paper mode

    def buy(self, symbol: str, quote_amount: float, *, client_order_id: str,
            reference_price: float | None = None) -> Fill:
        ref = reference_price or self.price(symbol)
        fill_price = ref * (1 + self.slippage_pct)
        fee = quote_amount * self.fee_rate
        quantity = (quote_amount - fee) / fill_price
        return Fill(symbol=symbol, side="BUY", quantity=quantity, price=fill_price,
                    quote_amount=-abs(quote_amount), fee=fee,
                    slippage=(fill_price - ref) * quantity, broker=self.name,
                    client_order_id=client_order_id, reference_price=ref,
                    raw={"simulated": True, "fee_rate": self.fee_rate,
                         "slippage_pct": self.slippage_pct})

    def sell(self, symbol: str, quantity: float, *, client_order_id: str,
             reference_price: float | None = None) -> Fill:
        ref = reference_price or self.price(symbol)
        fill_price = ref * (1 - self.slippage_pct)
        gross = quantity * fill_price
        fee = gross * self.fee_rate
        return Fill(symbol=symbol, side="SELL", quantity=quantity, price=fill_price,
                    quote_amount=gross - fee, fee=fee,
                    slippage=(ref - fill_price) * quantity, broker=self.name,
                    client_order_id=client_order_id, reference_price=ref,
                    raw={"simulated": True, "fee_rate": self.fee_rate,
                         "slippage_pct": self.slippage_pct})


# ── live ─────────────────────────────────────────────────────────────────────
class LiveBroker(Broker):
    """Real money. Three separate keys have to be turned before this can trade.

    ``TRADING_MODE=REAL`` selects it, API credentials let it authenticate, and
    ``LIVE_TRADING_CONFIRMED=1`` is the explicit acknowledgement that real funds
    are in play. Missing any one of them raises at construction rather than at the
    moment of the first order.
    """
    mode = REAL
    name = "live"

    def __init__(self, exchange=None):
        ex = get_settings().execution
        if not ex.live_confirmed:
            raise BrokerError(
                "REAL mode requires LIVE_TRADING_CONFIRMED=1. Refusing to trade real "
                "funds without an explicit acknowledgement.")
        if not (ex.exchange_api_key and ex.exchange_api_secret):
            raise BrokerError("REAL mode requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET.")
        self.settings = ex
        self._exchange = exchange
        self.max_order_quote = float(get_settings().caps.max_position_pct)  # sanity, see router

    @property
    def exchange(self):
        if self._exchange is None:
            try:
                import ccxt
            except ImportError as exc:                          # pragma: no cover
                raise BrokerError("ccxt is required for REAL mode: pip install ccxt") from exc
            klass = getattr(ccxt, self.settings.exchange_id)
            cfg = {"apiKey": self.settings.exchange_api_key,
                   "secret": self.settings.exchange_api_secret,
                   "enableRateLimit": True, "options": {"defaultType": "spot"}}
            if self.settings.exchange_password:
                cfg["password"] = self.settings.exchange_password
            self._exchange = klass(cfg)
            self._exchange.load_markets()
            log.warning("LIVE broker connected to %s — real funds are in play",
                        self.settings.exchange_id)
        return self._exchange

    def price(self, symbol: str) -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = float(ticker.get("last") or ticker.get("close") or 0.0)
            if price > 0:
                return price
        except Exception as exc:
            log.warning("exchange ticker failed (%s); using the public feed", exc)
        q = get_quote(symbol)
        if not q.ok:
            raise BrokerError(f"no price for {symbol}: {q.error}")
        return q.price

    def balances(self) -> dict[str, float]:
        try:
            free = self.exchange.fetch_balance().get("free", {})
            return {k: float(v) for k, v in free.items() if float(v or 0) > 0}
        except Exception as exc:
            raise BrokerError(f"could not read balances: {exc}") from exc

    def _order(self, symbol: str, side: str, amount: float,
               client_order_id: str, reference_price: float) -> dict:
        params = {"clientOrderId": client_order_id[:36]}
        return self.exchange.create_order(symbol, "market", side.lower(), amount, None, params)

    @staticmethod
    def _fill_from_order(order: dict, symbol: str, side: str, reference_price: float,
                         client_order_id: str) -> Fill:
        filled = float(order.get("filled") or order.get("amount") or 0.0)
        avg = float(order.get("average") or order.get("price") or reference_price or 0.0)
        cost = float(order.get("cost") or (filled * avg))
        fee_obj = order.get("fee") or {}
        fee = float(fee_obj.get("cost") or 0.0)
        if not fee:
            fee = sum(float(f.get("cost") or 0.0) for f in (order.get("fees") or []))
        signed = -abs(cost) if side == "BUY" else (cost - fee)
        return Fill(symbol=symbol, side=side, quantity=filled, price=avg,
                    quote_amount=signed, fee=fee,
                    slippage=abs(avg - reference_price) * filled if reference_price else 0.0,
                    status=str(order.get("status") or "filled"), broker="live",
                    client_order_id=client_order_id,
                    exchange_order_id=str(order.get("id") or ""),
                    reference_price=reference_price, raw={"order": order})

    def buy(self, symbol: str, quote_amount: float, *, client_order_id: str,
            reference_price: float | None = None) -> Fill:
        ref = reference_price or self.price(symbol)
        amount = quote_amount / ref
        try:
            amount = float(self.exchange.amount_to_precision(symbol, amount))
        except Exception:
            pass
        order = self._order(symbol, "BUY", amount, client_order_id, ref)
        fill = self._fill_from_order(order, symbol, "BUY", ref, client_order_id)
        log.warning("LIVE BUY %s %.8f @ ~%.2f (order %s)", symbol, fill.quantity,
                    fill.price, fill.exchange_order_id)
        return fill

    def sell(self, symbol: str, quantity: float, *, client_order_id: str,
             reference_price: float | None = None) -> Fill:
        ref = reference_price or self.price(symbol)
        try:
            quantity = float(self.exchange.amount_to_precision(symbol, quantity))
        except Exception:
            pass
        order = self._order(symbol, "SELL", quantity, client_order_id, ref)
        fill = self._fill_from_order(order, symbol, "SELL", ref, client_order_id)
        log.warning("LIVE SELL %s %.8f @ ~%.2f (order %s)", symbol, fill.quantity,
                    fill.price, fill.exchange_order_id)
        return fill


def make_broker(mode: str | None = None) -> Broker:
    mode = (mode or get_settings().execution.mode).upper()
    return LiveBroker() if mode == REAL else PaperBroker()
