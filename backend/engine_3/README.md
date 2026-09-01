# engine_3 — the risk engine that trains itself on this desk

The other two engines look at the market. This one looks at **us**: every decision
this system has made, every trade it has closed, and how those turned out. It
answers one question — *given everything we have already lived through, how does a
setup shaped like this one usually end?* — as a win probability, an expected value
in R, a size multiplier and, when warranted, a veto.

```
history in the database ──▶ dataset ──▶ train candidates ──▶ evaluate on an unseen tail
                                                              │
                            heuristic floor ──────────────────┤
                                                              ▼
                                             promote only if it beats BOTH
                                                              │
                                              store as bytes in the database
                                                              │
                                              prune to the newest 10 + active
                                                              │
                            service loads the active version at start,
                            and hot-swaps when a new one is promoted
```

## Where the labels come from

| source | what it is | trust |
|---|---|---|
| `trade` | a round trip that actually happened: the feature snapshot taken **at entry** is the X, whether it closed green is the y | ground truth |
| `shadow` | every cycle stores a feature snapshot and a price, so the forward return over the next few cycles answers "would a long opened here have paid?" | supporting |

Shadow labels are the reason the engine is trainable from the first hour instead
of after the hundredth trade, and they are the counterfactual that teaches it when
*not* to buy. They are used only to top up a thin trade set, and the split between
the two is recorded in every model's metrics.

Both are ordered by time and split chronologically with an embargo at the seam —
a random split would let the model read the future, which is the same mistake
`engine_2/dataset.py` documents in its own README.

## One feature definition

`features.py` is called from both training and serving. That is the whole point:
the most common way a system like this quietly breaks is that the training job and
the live path compute features slightly differently, so the model is served inputs
it never saw.

Forty-three features: each engine's signed confidence and its own indicators,
whether the engines agree or conflict, the book (drawdown, trades today, P&L
today, in position or flat), timing as sine/cosine, the regime (range position,
realized volatility, trend slope), and the setup being proposed (stop distance,
target distance, reward-to-risk, confidence).

Every model stores the feature list it was trained on, and `vectorize()` reads
*that* list, so adding a feature here never breaks an older model — it fills what
it does not recognise with a neutral default.

## Three model flavours, one interface

* **`heuristic`** — no training needed. Trading-desk rules as log-odds adjustments
  on a 50% prior: agreement helps, conflict hurts, reward-to-risk helps, drawdown
  and overtrading and high volatility hurt, and buying a blow-off top is faded.
  This runs on day one, and it is also the permanent floor.
* **`logistic`** — L2 logistic regression on standardized features. The right
  choice on a few hundred trades: it cannot memorise and its probabilities behave.
* **`gbm`** — histogram gradient boosting, only once there are ≥150 training rows.

All three serialize to bytes (`joblib` or JSON), which is what lets them live in a
database column.

## The gate

A candidate is promoted only if it clears **both**:

* an AUC floor of 0.53 — below that it is not distinguishable from a coin;
* the incumbent, by more than 0.01 AUC (or a tied AUC with a materially better
  Brier score);

and separately must not be worse than the heuristic floor on the same holdout. A
risk model that is worse than the rules it replaces is worse than useless, because
the Agent trusts it when sizing.

Candidates that fail are **kept, not deleted** — they are visible in
`GET /engine3/models` with `status=candidate`, which is how you see that the
engine has been trying and failing rather than not running.

## Retention

After every evaluation — never before it — the registry keeps the newest ten
versions plus whatever is currently active and deletes the rest. So the shelf
stays bounded, the active model is never pruned out from under the service, and a
freshly trained candidate is never deleted in the same breath that it earns its
place.

Change it with `ENGINE_3_MODEL_RETENTION`.

## Sizing

Expected value is `p × R − (1 − p)`. The stake is quarter-Kelly on the model's own
numbers — `f* = p − (1 − p)/R`, quartered — mapped onto a 0–1 multiplier that the
Agent and the risk guard both apply to the position cap. Full Kelly is a drawdown
machine and assumes your p and R are exactly right; they are not.

The multiplier is then cut further for engine conflict (×0.5), drawdown, elevated
volatility (×0.7) and running degraded on one engine (×0.6).

A veto is issued when expected value is negative, when the win probability is
below 42%, when the daily trade cap is reached or when the daily loss limit is
hit. The Agent may not answer BUY through a veto — the constitution rewrites it.

## Running it

```bash
python backend/main.py --train                    # one cycle: train, gate, prune
python -m engine_3.train --mode PAPER --keep 10   # the same, directly
curl -XPOST localhost:8000/admin/engine3/train -H "X-Admin-Token: ..."
```

It also runs on a timer (`ENGINE_3_TRAIN_INTERVAL_S`, default six hours), and a
promotion is picked up by the live service within 30 seconds without a restart.

`ENGINE_3_MIN_SAMPLES` (default 40) is the point below which it refuses to train
at all, and it will also refuse a one-sided label set or a holdout too small to
judge — those refusals are recorded as events rather than swallowed.
