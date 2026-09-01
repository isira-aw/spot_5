# Trading pipeline — automated data, honest evaluation

Replaces the notebook's static `dataset.npz` upload and 20-case eyeball demo with
a pipeline that can be run on a schedule and produces numbers you can act on.

```
fetch.py      ccxt paginated fetch -> incremental CSV cache
features.py   the 16 features, causal, ONE definition
dataset.py    windows + soft labels + chronological split + embargo + scaler
models.py     architectures, custom objects, state vector, policy adapter
train.py      forecaster fit + PPO, as a function (walk-forward calls it K times)
backtest.py   event-driven backtest, full trade statistics, baselines
walkforward.py rolling retrain/test folds
promote.py    scores candidate vs incumbent on unseen bars, gates the swap
retrain.py    the whole cycle in one command
inference.py  live loop, decoupled, hot-reloads promoted models
```

## Quick start

```bash
pip install ccxt PyWavelets pandas numpy tensorflow
export TRADER_ROOT=$PWD/trader

python -m trader.fetch --years 3          # ~105k 15m bars of BTC/USDT
python -m trader.dataset                  # -> data/dataset.npz
python -m trader.train                    # -> models_candidate/
python -m trader.promote --gate           # scores it, promotes only if better
python -m trader.inference --paper        # live loop
python trader/tests/test_pipeline.py      # 8 property tests, no TF needed
```

Walk-forward (slow — trains K times, run it on the GPU box):

```python
from trader.fetch import load_cache
from trader.train import train_fold
from trader import walkforward as wf
wf.run(load_cache(), train_fold, n_folds=6, out_json="reports/wf.json")
```

## Scheduling

| Job | Cadence | How |
|---|---|---|
| inference | every bar | `systemd` service or Docker; the loop self-schedules to candle close +5s |
| retrain | weekly/monthly | cron: `0 3 * * 0 cd /opt/bot && python -m trader.retrain --walkforward` |

The two never touch. Retrain writes `models_candidate/`; the gate copies into
`models/` and keeps the previous bundle in `models_prev/`; the inference loop
notices the mtime change and reloads on the next bar. A retrain that fails, or a
candidate that loses to the incumbent, leaves the live model exactly where it was.

## The five changes, and where they live

1. **Automated data** — `fetch.py`. First run pages back 3 years, later runs only
   fetch the missing head. Gaps are reported, never interpolated. The in-progress
   candle is dropped (its OHLC is still changing).
2. **Chronological split** — `dataset.chronological_split`. Train is oldest, test
   is newest, with a `WINDOW_SIZE + HORIZON` embargo at each seam. Without the
   embargo the last training window and the first validation window literally
   share candles. The scaler is fitted on train only and saved with the model, so
   live inference applies the identical transform.
3. **More data, one pair** — `config.HISTORY_YEARS = 3`, `SYMBOL = BTC/USDT`.
   Three years spans a bear leg, a chop regime, and a bull leg. The 7-symbol
   forecaster + BTC agent split in the notebook mixes the question "does this
   generalise across assets" with "does it work at all" — answer the second first.
4. **Real backtest** — `backtest.py` + `promote.evaluate`. Thousands of trades
   with win rate, expectancy (with a bootstrap CI), profit factor, Sharpe,
   Sortino, max drawdown, Calmar, exposure, exit mix, and — the part that matters —
   the same metrics for a random trader and buy-and-hold on identical bars.
   `edge_after_costs` is only true when the 95% CI on expectancy excludes zero.
5. **Walk-forward** — `walkforward.py`. K rolling folds, each retrained from
   scratch, each tested on unseen bars. `consistent_edge` requires ≥75% of folds
   profitable and median Sharpe > 0.5. One good fold is a coincidence.

## Execution assumptions

Deliberately pessimistic, because an optimistic backtest is worse than none:

- decisions on the close of bar *t*, filled at the **open of bar t+1** (the
  notebook's reward filled at the current close — a price you cannot get);
- fees and slippage on both legs;
- stop/target checked intrabar against low/high; if both are touched in one bar,
  the **stop** is assumed to fill first; a gap through the stop fills at the open.

`train.py` uses the same next-open fill in the PPO reward, so the agent optimizes
the quantity the backtest measures.

## Things found in the notebook worth fixing regardless

- **Export mismatch (serious).** The PPO trained on `full_forecaster`
  (CNN-BiLSTM-MHA) predictions, but the TF.js export converted `forecaster` (the
  plain BiLSTM baseline) to `models/bilstm/model.json`. The deployed agent was
  fed a different forecaster than the one it learned against. `models.py` has one
  switch, `FORECASTER_NAME`, used by training, evaluation and export alike.
- **The collapse penalty is treating a symptom.** `collapse_aware_bce` pushes
  batch std above 0.05 whether or not the model has learned anything, so `predStd`
  stops being evidence of signal. It is kept for compatibility, but the gate in
  `promote.py` now judges on out-of-sample P&L, not on dispersion.
- **512 identical environments.** The parallel envs share one price series and one
  start index, so they differ only by action sampling — that is variance
  reduction, not data diversity. Randomizing start offsets per env would give the
  agent more regimes per update.
- **`RETURN_CLIP = 0.1` with `SOFT_LABEL_SCALE = 400`** saturates any move beyond
  roughly ±1% to the clip bounds, so a 1% and a 9% move carry the same label.
- **Feature parity is the biggest unverified risk.** The original 10 indicators
  live in a separate dataset notebook that was not in hand, so `features.py`
  implements a documented set (RSI, MACD, Bollinger %B, ATR, volume z, EMA ratio,
  realized vol, body ratio, plus a trailing z-scored close and clipped log return).
  If your dataset notebook used different ones, port them into `_base_features()`
  and retrain — do not serve a model features it never saw. `test_features_are_causal`
  will keep whatever you write honest about look-ahead.

## Tests

`python trader/tests/test_pipeline.py` — no TensorFlow required.

- appending future bars does not change any past feature (look-ahead check);
- split ordering and embargo hold;
- windows and labels stay aligned, labels never read past the end;
- a round trip in a flat market loses exactly the round-trip cost;
- a stop wick exits at the stop, not at the signal;
- random trading on a random walk has negative expectancy (proves costs bite).
