# Trading Theories Knowledge Base

Reference material for the Agent. Sections are retrieved by relevance to the
current market state and pasted into the prompt, so each one is written to stand
alone. Edit this file freely — the running system notices the change, reloads it
within a minute and keeps serving from the old copy until the new one is parsed,
so there is no downtime and no restart.

Format contract for editors: one `##` heading per section, an optional
`**Tags:**` line directly underneath it, then the body. Nothing else is required.

---

## Trend Following and Momentum
**Tags:** trend, momentum, ema, moving average, breakout, trending_up, trending_down, ret_3, ret_10

The premise is that price series have autocorrelation at some horizons: what has
been rising tends to keep rising slightly more often than chance. This is not a
prediction of the next tick, it is a statement about the distribution of the next
hundred trades.

- **Structure over indicators.** An uptrend is higher highs *and* higher lows. A
  fast EMA above a slow EMA is a summary of that, not a cause of it. When the two
  disagree — EMAs crossed up but the last swing low was broken — trust structure.
- **Trend strength is measured against volatility.** A +2% move where ATR is 0.4%
  is a real thrust; the same move where ATR is 2.5% is noise. Always divide the
  move by the ATR before calling it strong.
- **Momentum decays with horizon.** A 15-minute thrust says almost nothing about
  the next month. Weight short-horizon momentum for short-horizon decisions only.
- **The failure mode is the chop.** Trend systems lose in ranges, in a long string
  of small losses. If the recent regime is a range, expect a lower hit rate and
  size down rather than trying to trade around it.
- **Never chase the third or fourth bar of a vertical move.** Entering after the
  move has extended is how a trend follower ends up buying the exact top of an
  exhaustion leg. Wait for a pullback that holds.

## Mean Reversion and Overextension
**Tags:** mean reversion, rsi, overbought, oversold, range, quiet_range, bollinger, fade, extremes

Prices oscillate around a moving reference. When they get far enough away, the
odds of a snap back rise — but the size of the move against you while you wait
also rises.

- **RSI is a ranking, not a signal.** RSI > 70 in a strong uptrend is normal and
  can persist for weeks. RSI > 78 combined with a fading volume ratio and a stall
  in structure is an actual overextension.
- **Only fade extremes with the higher timeframe.** Fading a 15-minute spike
  inside a daily uptrend is a trade; fading a daily breakdown because the
  15-minute is oversold is a way to lose money slowly.
- **Range position matters.** Buying at 5–20% of the recent range with a stop
  under the low is a very different bet from buying at 90% of the range, even when
  every other signal agrees.
- **Mean reversion needs tighter targets.** Take profit near the mean (the EMA, the
  mid-range), not at a trend-following extension target.

## Volatility Regimes
**Tags:** volatility, atr, regime, high_volatility, chop, realized_vol, sizing, stops

Volatility is the single most useful conditioning variable in this system, because
it changes what every other number means.

- **Stops must be volatility-scaled.** A fixed 1% stop is a coin flip in a 3% ATR
  regime and a straitjacket in a 0.3% one. Anchor stops to ATR multiples (1.5×
  ATR on intraday horizons, 2–2.5× on daily and above), then check the resulting
  percentage is inside policy.
- **Position size is inversely proportional to stop distance.** Risk a fixed
  fraction of equity, then let the stop distance decide the quantity. Do not risk
  a fixed *quantity* and let the stop decide the loss.
- **Volatility clusters.** A violent hour is usually followed by another violent
  hour. After a volatility spike, expect wider ranges for several bars — widen the
  stop and cut the size, do not keep the size and widen the stop.
- **Rising volatility with falling price is a risk-off signal.** Rising volatility
  with rising price is often continuation. The sign matters.

## Risk of Ruin and Position Sizing
**Tags:** risk, sizing, kelly, position size, capital at risk, ruin, drawdown, leverage

The mathematics of survival dominates the mathematics of edge. A 50% drawdown
requires a 100% gain to recover.

- **Fixed fractional risk.** Risk a constant small fraction of equity per trade
  (0.5–2%). Position size = (equity × risk fraction) ÷ (entry − stop).
- **Kelly, then quarter it.** The Kelly fraction f\* = p − (1 − p)/R maximises the
  long-run growth rate but assumes your p and R are exactly right. They are not.
  Quarter-Kelly keeps most of the growth with a fraction of the variance.
- **A negative expected value is never rescued by size.** If p × R − (1 − p) ≤ 0,
  the correct size is zero. There is no confidence level that fixes a bad bet.
- **Correlated positions are one position.** Multiple crypto longs are a single
  bet on the same factor. Count them as one when measuring capital at risk.
- **Cut size in a drawdown.** After a string of losses, halve the size until the
  equity curve stabilises. This is the opposite of what the instinct to "make it
  back" demands, and it is why the instinct loses.

## Expectancy, R-Multiples and Win Rate
**Tags:** expectancy, r multiple, win rate, profit factor, edge, reward to risk, statistics

- Express every outcome in **R**, where 1R is the amount risked. A trade that made
  1.8× what it risked is +1.8R regardless of account size.
- **Expectancy = (win rate × average win in R) − (loss rate × average loss in R).**
  A 40% win rate at 3R is far better than a 70% win rate at 0.5R.
- **A high win rate is not evidence of edge.** Selling volatility, wide stops and
  early exits all produce high win rates and negative expectancy.
- **Sample size.** Fewer than ~30 trades tells you nothing. Fewer than ~100 tells
  you very little. Treat early statistics as a prior, not a conclusion.
- **Profit factor below 1.0 means the strategy loses money**, however good the
  narrative sounds.

## Model Confluence and Disagreement
**Tags:** confluence, disagreement, ensemble, engines, conflict, agreement, calibration, confidence

This system runs two independent forecasting engines plus a risk model trained on
its own history. How they relate is itself a signal.

- **Agreement between methodologically different models is the strongest evidence
  available here.** A technical/structural engine and a learned sequence model
  reaching the same conclusion from different inputs is not double counting.
- **Disagreement is information, not noise to average away.** When they conflict,
  the honest answer is usually a smaller position or none — not the average of
  their views. Say which one you sided with and why.
- **A degraded engine is not a neutral engine.** If one engine is down, you are
  running on one opinion; say so and reduce size rather than pretending the
  ensemble is intact.
- **Confidence must be calibrated, not enthusiastic.** If you say 80% you should
  be right about four times in five. When in doubt, quote a lower number.
- **Beware confirmation stacking.** Five indicators derived from the same closing
  price are one opinion wearing five hats.

## Market Structure and Liquidity
**Tags:** structure, support, resistance, liquidity, order book, spread, slippage, range_pos

- **Levels that matter are the ones many participants can see**: prior swing highs
  and lows, the range extremes of the last session, round numbers, the prior day's
  close.
- **Stops cluster just beyond obvious levels**, which is exactly why price
  frequently pokes through and reverses. Place stops *beyond* the noise around a
  level, not at the level itself.
- **Liquidity is worst when you need it most.** Assume slippage is larger during
  volatility spikes and thin hours, and that the fill is worse than the last
  printed price.
- **Decide on the close, fill on the next open.** Any backtest that fills at the
  price used to make the decision is reporting a number nobody can trade.

## Costs, Fees and the Break-Even Move
**Tags:** fees, costs, slippage, break-even, churn, overtrading, taker

Every round trip pays fees twice plus slippage twice. At 0.075% taker fees and
0.05% slippage, a round trip costs roughly 0.25%. A strategy whose average move is
0.3% is trading for the exchange's benefit.

- **Compute the break-even move before entering.** If the target is not several
  multiples of the round-trip cost, the trade is not worth taking.
- **Frequency multiplies cost.** Ten trades a day at 0.25% is a 2.5% daily headwind.
- **The cheapest improvement available to most systems is trading less.**

## Spot-Only Constraints
**Tags:** spot, no shorting, cash, exit, hold, long only, inventory

This system trades spot with no leverage and no shorting. That changes the
vocabulary of a bearish view.

- **Bearish means "sell what you hold" or "stay in cash", never "go short".** If
  flat and the view is down, the action is HOLD (in cash), not SELL.
- **Cash is a position.** Standing aside during a downtrend is the profitable
  expression of a bearish view here.
- **There is no margin call, so the real risk is opportunity cost and drawdown**,
  not liquidation. That permits wider stops than a leveraged book would allow —
  but wider stops still mean smaller size, not more risk.
- **Never average down into a losing position** without a pre-declared plan that
  was part of the original entry. "It is cheaper now" is not a plan.

## Regime Awareness and Adaptation
**Tags:** regime, adaptation, drawdown, out of sample, overfitting, model decay, walk forward

- **Every edge is conditional on a regime.** A model trained through a bull market
  will be confidently wrong in a bear one.
- **Watch for edge decay.** A rolling win rate that has fallen for many trades is
  a reason to reduce size, not a reason to wait for it to revert.
- **Out-of-sample or it did not happen.** A backtest result that was not produced
  on data the model never saw is a description of the past, not a forecast.
- **Prefer fewer, better-understood signals.** Complexity buys in-sample fit and
  sells out-of-sample reliability.

## Behavioural Traps
**Tags:** psychology, bias, discipline, revenge trading, fomo, anchoring, tilt

Automation removes the hands but not the biases, because the biases are encoded
in the thresholds a human chose.

- **Revenge trading:** raising size after a loss to recover it. The system's
  defence is the daily loss limit and the trade counter.
- **FOMO:** entering late because the move already happened. The defence is a
  pre-declared entry zone that a chased price falls outside.
- **Anchoring:** treating the entry price as meaningful. The market does not know
  where you got in. Only the stop and the thesis matter.
- **Narrative fitting:** finding a story for a random move. If the reasoning would
  have been equally convincing for the opposite outcome, it is not a reason.
- **Sunk cost:** holding a loser because of the time already spent in it.

## What Would Change the View
**Tags:** invalidation, thesis, exit, change mind, stop, review

Every position is a hypothesis with a stated way to be wrong. If it cannot be
falsified, it is not a thesis.

- **Name the invalidation before entering**: the price, the structural event or the
  regime shift that means the reasoning was wrong.
- **A stop is the price where the thesis is void**, not a tolerance for pain.
- **Time stops count.** If the expected move has not begun within the expected
  horizon, the setup has failed even if the stop was not hit.
- **New information beats an old plan** — but only new *information*, not new price
  action inside the range the plan already anticipated.

## Decision Discipline for This System
**Tags:** process, checklist, hold, discipline, default, execution

- **HOLD is the default and is a real answer.** Most bars are not opportunities.
  Producing an action every cycle is how a system churns itself to death.
- **Do not act below the confidence floor.** Under the configured minimum, the
  answer is HOLD regardless of how interesting the setup looks.
- **One thesis per decision.** If the reasoning needs three unrelated arguments to
  reach a conclusion, the conclusion is weak.
- **Respect the operator's limits absolutely.** Position caps, daily trade counts,
  the kill switch and blackout windows are constraints, not suggestions, and no
  market condition overrides them.
- **Explain like a person.** State what you would do, at what size, with what stop,
  why, and what would change your mind. A rationale that could be pasted under
  any decision is not a rationale.
