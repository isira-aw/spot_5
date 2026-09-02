# engine_2 — forecaster + PPO agent, as a production pipeline

A CNN-BiLSTM-MHA forecaster and the PPO policy trained against it, packaged as
scriptable jobs instead of a notebook. **This package is a model factory.** It
pulls public market data, trains, gates, versions and promotes model bundles, and
scores live bars. It contains no order placement, no execution and no withdrawal
path — trading lives in `backend/execution`, driven by the Agent, and engine_2
only ever hands it an opinion. A unit test enforces that
(`tests/test_engine_2.py::test_engine_2_contains_no_order_placement`).

```
config.py      every tunable, env-overridable with the ENGINE_2_* prefix
fetch.py       ccxt paginated fetch, read-only key guard, retries -> CSV cache
features.py    the 16 features, causal, ONE definition
dataset.py     windows + volatility-scaled labels + train/val/test/HOLDOUT split
models.py      architectures, the correctness-aware loss, the state vector
gates.py       the checks that RAISE and stop the cycle
train.py       forecaster fit + PPO (random env starts, warm start, entropy control)
backtest.py    event-driven backtest, volatility-scaled costs, baselines
walkforward.py rolling retrain/test folds
promote.py     test-slice gate + the untouched-holdout backtest
registry.py    versioned bundles, promotion, rollback, retention
drift.py       live predictions vs realized outcomes, rolling decay scorecard
jobs.py        every stage as a callable job + one CLI (what the scheduler calls)
retrain.py     the whole cycle in one command (wrapper over jobs.cycle)
inference.py   live loop, hot-reloads promoted models, feeds the drift monitor
```

## Quick start

```bash
# TensorFlow, ccxt and PyWavelets are NOT in backend/requirements.txt: the API
# process does not need a ~600MB deep-learning install. Training and in-process
# inference do.
pip install -r requirements.txt           # from backend/engine_2/

python -m engine_2.jobs pull              # ~105k 15m bars -> dataset.npz
python -m engine_2.jobs cycle             # train, gate, holdout backtest, promote
python -m engine_2.inference --paper      # live loop
python -m pytest ../tests/test_engine_2.py -q
```

A box that only runs the API can skip all of it: set `ENGINE_2_SOURCE=file` and
this process just tails the decision log written by whichever machine serves the
model.

## How it is consumed

`backend/adapters/engine_two.py` is the only consumer. It reads engine_2 either
`inline` (importing `inference.Runner`, feeding it the broker's real position) or
`file` (tailing `reports/live_decisions.jsonl`), and turns a decision into an
`EngineSignal` for the Agent. Model artifacts are read from
`ENGINE_2_MODELS_DIR` (default `engine_2/models/`), which is always a copy of a
version directory that passed the gates.

Everything engine_2 publishes for the rest of the system:

| Surface | What |
|---|---|
| `models/` + `models/CURRENT.json` | the promoted bundle and which version it is |
| `reports/live_decisions.jsonl` | one JSON line per bar: action, probs, p_up, drift |
| `app_state.engine_2_drift` | rolling live decay scorecard |
| `app_state.engine_2_last_cycle` | the last retraining cycle's outcome |
| `GET /engine2/models` | versions, current, drift, last cycle |
| `POST /admin/engine2/retrain`, `/rollback` | operator controls (admin token) |

## Scheduling

Two loops, in `pipeline/scheduler.py`, following the same `PeriodicTask` pattern
engine_3 uses — a task that throws logs, records an event and runs again.

| Task | Default | Env |
|---|---|---|
| `engine_2_drift` | hourly, on whenever engine_2 is enabled | `ENGINE_2_DRIFT_INTERVAL_S` |
| `engine_2_training` | weekly, **off by default** | `ENGINE_2_AUTO_TRAIN`, `ENGINE_2_TRAIN_INTERVAL_S` |

Training is hours of GPU, not the minutes engine_3 needs, so the usual deployment
runs the API with `ENGINE_2_AUTO_TRAIN=0` and trains on a separate box from cron:

```
0 3 * * 0  cd /opt/spot5/backend && python -m engine_2.jobs cycle --walkforward
```

Set `ENGINE_2_RETRAIN_ON_DRIFT=1` to let a sustained live decay trigger a cycle
without waiting for the weekly slot.

The training and inference paths never touch. A cycle writes
`models_candidate/`, promotion freezes it into `models_versions/<version>/` and
copies it to `models/`, and the inference loop notices the mtime change and
reloads on the next bar. A failed cycle leaves the live model exactly where it
was.

## Data and credentials

`BINANCE_API_KEY` / `BINANCE_API_SECRET` are **optional** and buy nothing but a
higher rate limit — Binance serves OHLCV unauthenticated. They must be read-only:
`fetch.assert_read_only()` queries the key's permissions at the start of every
pull and raises if the key can trade, use margin or withdraw. Exchange downtime,
rate limits and transient errors are retried with exponential backoff. Gaps in
history are reported and left alone; a synthetic candle is a fabricated training
example.

## What changed from the notebook, and why

**Labels were nearly information-free.** `sigmoid(return * 400)` maps a typical
15m BTC move (0.05–0.3%) to 0.51–0.56 — indistinguishable from "no idea" — so the
loss was dominated by a handful of >1% bars. `dataset.soft_labels` now divides the
h-bar return by the trailing sigma of 1-bar returns (grown as `sqrt(h)`) and
sigmoids that, so a one-sigma move is 0.73 in every regime and the label says how
*unusual* a move is, not how *big*.

**The anti-collapse penalty rewarded noise.** `collapse_aware_bce` only required
batch std > 0.05, which a model can satisfy by predicting 0.3/0.7 at random —
which is why `predStd` stopped being evidence of anything. It now subtracts
`mean((2*label-1)*(2*pred-1))`: dispersion pays only when it points the right way,
and confident mistakes are punished symmetrically.

**Three of four horizons were dead weight.** h1–h4 were computed and only
`forecast[0]` was ever read. All four are now in the state vector, plus a signed
`horizon_agreement` scalar (+1 unanimous up, −1 unanimous down), and the PPO
reward pays an entry bonus only when the horizons agree with the trade.

**`BACKTEST_HOLDOUT` was defined and never used.** There is now a fourth split
after `test`, embargoed like the others, read exactly once by
`promote.final_backtest` — after every tuning decision has already been made. It
reports Sharpe, max drawdown and win rate, and it is a required gate: a candidate
that fails it is not promoted, whatever its `test` numbers say.

**Leakage.** The split is strictly chronological with a `WINDOW_SIZE + HORIZON`
embargo at each of the three seams, and the scaler is fitted on train only and
shipped with the model. `test_split_is_chronological_embargoed_and_keeps_a_holdout`
and `test_features_are_causal` keep it honest.

**The always-HOLD policy.** The root cause was a forecaster whose signal did not
survive fees — a near-constant policy is the rational answer to that, so the label
and loss fixes above are the real repair. Three things back them up:
`policy_spread` (mean std of P(action) across states) is measured every update,
the entropy coefficient is raised automatically while it is low and decayed once
it recovers, and `gates.check_policy` **rejects** a bundle whose final spread is
below `ENGINE_2_GATE_MIN_POLICY_SPREAD`. The notebook's shipped agent — 36% HOLD
/ 28% BUY / 36% SELL whatever the market did — would not have passed.

**512 identical environments.** Every env replayed the same candles from the same
start, which parallelizes compute, not data. Each env now draws a random start
offset every update and wraps around the series, so the agent never sees the same
trajectory twice.

**BTC/USDT only, deliberately.** The notebook trained the forecaster on 7 symbols
and the agent on BTC alone, which conflates "does this generalise across assets"
with "does it work at all". This is a documented single-pair policy: `ENGINE_2_SYMBOL`
moves it, and broadening to a multi-symbol agent should wait until a single pair
survives walk-forward.

**The hold-a-winner bonus fought the take-profit.** `1.5 * price_ret` every bar
while in profit was uncapped, so riding a position past `TAKE_PROFIT_PCT` scored
better than the exit the risk rules were about to force. The multiplier now tapers
linearly to 1.0 as unrealized gain approaches `TAKE_PROFIT_PCT`.

**Costs were flat.** Slippage now scales with the same realized-volatility feature
the policy sees (`config.slippage_for_vol`), in the PPO reward, the backtester and
the live loop alike — so the agent is not taught that a fill during a crash costs
what a fill on a dead Sunday costs.

**Warnings became gates.** The `predStd` / `directionalAccuracy` checks printed
and carried on; PPO then trained for an hour against a collapsed forecaster and
the model shipped. `gates.py` raises `GateFailed`, and the cycle stops before any
PPO compute or promotion.

**Nothing was versioned.** `models/` was overwritten every run, so a bad model
destroyed the good one it replaced. `registry.py` freezes every promoted bundle
into `models_versions/<UTC-date>-<git-sha>/`, keeps the newest
`ENGINE_2_MODEL_RETENTION` plus whatever is live, and makes rollback one command.

**PPO always restarted from scratch.** `train_ppo(warm_start_dir=...)` fine-tunes
the previous champion at a quarter of the learning rate, falling back to a fresh
initialization when the saved actor's state shape no longer matches.

**Nothing watched the live model.** `drift.py` keeps a rolling window of live
predictions, resolves each against the realized close `HORIZON` bars later, and
recomputes live `directionalAccuracy` / `predStd`. Below
`ENGINE_2_DRIFT_MIN_DIR_ACC` for `ENGINE_2_DRIFT_BREACHES` consecutive checks it
flags `retrain_recommended`, which the scheduler can act on automatically.

**Export mismatch (found in the notebook, kept fixed).** PPO trained on
`full_forecaster` predictions while the TF.js export shipped `forecaster`, the
plain BiLSTM baseline — the deployed agent was fed a different model than the one
it learned against. `models.FORECASTER_NAME` is the single switch used by
training, evaluation and export alike.

## Execution assumptions

Deliberately pessimistic, because an optimistic backtest is worse than none:

- decisions on the close of bar *t*, filled at the **open of bar t+1**;
- fees and volatility-scaled slippage on both legs;
- stop/target checked intrabar against low/high; if both are touched in one bar
  the **stop** fills first; a gap through the stop fills at the open.

`train.py` uses the same next-open fill and the same cost model in the PPO reward,
so the agent optimizes the quantity the backtest measures.

## A standing caveat

`features.py` implements a documented 16-feature set (RSI, MACD, Bollinger %B,
ATR, volume z, EMA ratio, realized vol, body ratio, a trailing z-scored close, a
clipped log return, and 6 wavelet features). The notebook's original 10 indicators
lived in a separate dataset notebook that was not in hand. If yours differ, port
them into `_base_features()` and retrain — do not serve a model features it never
saw. `test_features_are_causal` will keep whatever you write honest about
look-ahead.
