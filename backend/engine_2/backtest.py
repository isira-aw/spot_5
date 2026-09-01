"""Event-driven backtest over the full test slice (point 4).

Replaces the 20-case eyeball demo. Pure numpy — no TensorFlow import — so it can
also score a random or buy-and-hold policy, which is the comparison that actually
tells you whether the model earned its keep.

Execution assumptions, all pessimistic on purpose:
  * decisions are made on the CLOSE of bar t and filled at the OPEN of bar t+1
    (the notebook's runner rewarded a fill at the current close, which is a fill
    at a price you cannot get);
  * fees and slippage are charged on both legs;
  * stop-loss and take-profit are checked intrabar against low/high, and when
    both are touched in the same bar the STOP is assumed to fill first;
  * a gap through the stop fills at the open, not at the stop price.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

from . import config as C

HOLD, BUY, SELL = 0, 1, 2


@dataclass
class ExecConfig:
    fee: float = C.FEE_RATE
    slippage: float = C.SLIPPAGE_PCT
    stop_loss: float = C.STOP_LOSS_PCT
    take_profit: float = C.TAKE_PROFIT_PCT
    bars_per_year: int = C.BARS_PER_YEAR
    max_bars_in_trade: int = 0        # 0 = no time stop


@dataclass
class Trade:
    entry_i: int
    exit_i: int
    entry_px: float
    exit_px: float
    ret: float          # net of fees + slippage
    bars: int
    reason: str


def run(candles: np.ndarray,
        actions: np.ndarray | Callable[[int, dict], int],
        start: int = 0,
        cfg: ExecConfig = ExecConfig()) -> dict:
    """candles: (n,>=6). actions: array aligned to bars, or fn(i, state)->action."""
    o, h, l, c = candles[:, 1], candles[:, 2], candles[:, 3], candles[:, 4]
    n = len(candles)
    callable_policy = callable(actions)

    equity = np.ones(n, dtype=np.float64)
    pos, entry_px, entry_i, bars_in = 0, 0.0, -1, 0
    trades: list[Trade] = []
    eq = 1.0
    slip, fee = cfg.slippage, cfg.fee

    def close_trade(i, px, reason):
        nonlocal eq, pos, entry_px, entry_i, bars_in
        # buy filled slightly above, sell slightly below, fee on both legs
        net = (px * (1 - slip) * (1 - fee)) / (entry_px * (1 + slip) * (1 + fee)) - 1.0
        eq *= (1 + net)
        trades.append(Trade(entry_i, i, entry_px, px, net, i - entry_i, reason))
        pos, entry_px, entry_i, bars_in = 0, 0.0, -1, 0

    for i in range(start, n - 1):
        # ── intrabar risk exits, checked before any new decision ─────────────
        if pos == 1:
            stop_px = entry_px * (1 - cfg.stop_loss)
            take_px = entry_px * (1 + cfg.take_profit)
            if l[i] <= stop_px:
                close_trade(i, min(stop_px, o[i]), "stop")
            elif h[i] >= take_px:
                close_trade(i, max(take_px, o[i]), "target")
            elif cfg.max_bars_in_trade and bars_in >= cfg.max_bars_in_trade:
                close_trade(i, c[i], "time")

        # ── policy decision on this close, filled next open ──────────────────
        if callable_policy:
            state = {"position": pos, "entry_px": entry_px, "bars_in": bars_in,
                     "pnl": (c[i] / entry_px - 1.0) if pos == 1 else 0.0}
            a = int(actions(i, state))
        else:
            a = int(actions[i])

        if pos == 0 and a == BUY:
            pos, entry_px, entry_i, bars_in = 1, o[i + 1], i + 1, 0
        elif pos == 1 and a == SELL:
            close_trade(i + 1, o[i + 1], "signal")
        elif pos == 1:
            bars_in += 1

        # mark to market, net of the entry cost already paid
        if pos == 1:
            equity[i] = eq * (c[i] * (1 - slip) * (1 - fee)) / (entry_px * (1 + slip) * (1 + fee))
        else:
            equity[i] = eq

    if pos == 1:
        close_trade(n - 1, c[-1], "eod")
    equity[-1] = eq
    equity[:start] = 1.0

    return {"trades": trades, "equity": equity, "config": asdict(cfg),
            "buy_hold": float(c[-1] / c[start] - 1.0)}


# ── statistics ───────────────────────────────────────────────────────────────
def _max_drawdown(equity: np.ndarray):
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    i = int(dd.argmin())
    j = int(equity[:i + 1].argmax()) if i else 0
    # longest time under water, in bars
    under = equity < peak
    longest, run_len = 0, 0
    for u in under:
        run_len = run_len + 1 if u else 0
        longest = max(longest, run_len)
    return float(dd.min()), (j, i), longest


def metrics(result: dict, cfg: ExecConfig = ExecConfig()) -> dict:
    trades, equity = result["trades"], result["equity"]
    r = np.array([t.ret for t in trades], dtype=np.float64)
    n = len(r)
    wins, losses = r[r > 0], r[r <= 0]

    bar_ret = np.diff(np.log(equity))
    bar_ret = bar_ret[np.isfinite(bar_ret)]
    ann = cfg.bars_per_year
    sharpe = (bar_ret.mean() / (bar_ret.std() + 1e-12)) * np.sqrt(ann) if len(bar_ret) else 0.0
    downside = bar_ret[bar_ret < 0]
    sortino = (bar_ret.mean() / (downside.std() + 1e-12)) * np.sqrt(ann) if len(downside) else 0.0

    mdd, (peak_i, trough_i), uw = _max_drawdown(equity)
    total = float(equity[-1] - 1.0)
    years = len(equity) / ann
    cagr = (equity[-1] ** (1 / years) - 1.0) if years > 0 and equity[-1] > 0 else 0.0
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    bars_held = sum(t.bars for t in trades)

    m = {
        "n_trades": n,
        "win_rate": float((r > 0).mean()) if n else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(r.mean()) if n else 0.0,
        "expectancy_bps": float(r.mean() * 1e4) if n else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "total_return": total,
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": mdd,
        "underwater_bars": int(uw),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("inf"),
        "exposure": float(bars_held / max(len(equity), 1)),
        "trades_per_year": float(n / years) if years > 0 else 0.0,
        "avg_bars_held": float(np.mean([t.bars for t in trades])) if n else 0.0,
        "buy_hold_return": result["buy_hold"],
        "exit_mix": {k: sum(1 for t in trades if t.reason == k)
                     for k in ("signal", "stop", "target", "time", "eod")},
    }
    m["expectancy_ci95"] = bootstrap_ci(r) if n >= 30 else None
    m["edge_after_costs"] = bool(m["expectancy"] > 0 and n >= 30
                                 and (m["expectancy_ci95"] or [0])[0] > 0)
    return m


def bootstrap_ci(r: np.ndarray, iters: int = 5000, alpha: float = 0.05, seed: int = 0):
    """95% CI on mean trade return. If the lower bound is <= 0 you do not have a
    demonstrated edge, whatever the headline expectancy says."""
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(iters, len(r)), replace=True).mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def report(m: dict, title: str = "Backtest") -> str:
    ci = m["expectancy_ci95"]
    ci_s = f"[{ci[0]*1e4:+.2f}, {ci[1]*1e4:+.2f}] bps" if ci else "n/a (<30 trades)"
    return "\n".join([
        f"── {title} " + "─" * max(0, 58 - len(title)),
        f"  trades            {m['n_trades']:,}  ({m['trades_per_year']:.0f}/yr, "
        f"avg {m['avg_bars_held']:.1f} bars)",
        f"  win rate          {m['win_rate']:.1%}",
        f"  avg win / loss    {m['avg_win']*1e4:+.1f} / {m['avg_loss']*1e4:+.1f} bps",
        f"  expectancy        {m['expectancy_bps']:+.2f} bps/trade   95% CI {ci_s}",
        f"  profit factor     {m['profit_factor']:.3f}",
        f"  total return      {m['total_return']:+.2%}   (buy & hold "
        f"{m['buy_hold_return']:+.2%})",
        f"  CAGR              {m['cagr']:+.2%}",
        f"  Sharpe / Sortino  {m['sharpe']:.2f} / {m['sortino']:.2f}",
        f"  max drawdown      {m['max_drawdown']:.2%}  (underwater "
        f"{m['underwater_bars']:,} bars)",
        f"  Calmar            {m['calmar']:.2f}",
        f"  exposure          {m['exposure']:.1%}",
        f"  exits             {m['exit_mix']}",
        f"  verdict           {'edge survives costs' if m['edge_after_costs'] else 'NO demonstrated edge'}",
    ])


# ── reference policies to beat ───────────────────────────────────────────────
def random_policy(p_buy=0.05, seed=0):
    rng = np.random.default_rng(seed)
    def f(i, s):
        if s["position"] == 0:
            return BUY if rng.random() < p_buy else HOLD
        return SELL if rng.random() < p_buy else HOLD
    return f


def always_long(i, s):
    return BUY if s["position"] == 0 else HOLD
