# spot_5 — four brains, one voice

An automated spot-trading desk. Two independent forecasting engines and a
self-training risk engine report to a single LLM Agent, which is the only thing in
the system allowed to produce an order and does so in plain English.

```
                  ┌──────────────────────────────────────────────┐
   market data ──▶│ engine_1   context / calibration             │──┐
                  │  11 horizons, EMA/RSI/ATR structure, news    │  │
                  └──────────────────────────────────────────────┘  │
                  ┌──────────────────────────────────────────────┐  │   ┌─────────┐
   market data ──▶│ engine_2   quantitative / ML                 │──┼──▶│  AGENT  │──▶ BUY/SELL/HOLD
                  │  CNN-BiLSTM-attention forecaster + PPO       │  │   │  (LLM)  │    size, stop,
                  └──────────────────────────────────────────────┘  │   └─────────┘    target, and
                  ┌──────────────────────────────────────────────┐  │        ▲         why, in words
   own history ──▶│ engine_3   risk, trained on this desk        │──┘        │
                  │  win probability, expected R, size, veto     │           │
                  └──────────────────────────────────────────────┘   knowledge base
                                                                   + admin restrictions
                                                                   + portfolio state
                                        │
                                        ▼
                          constitution → risk guard → Broker
                                                     ├─ PaperBroker  (real prices, simulated money)
                                                     └─ LiveBroker   (real prices, real funds)
```

The three engines never trade. Only the Agent's answer becomes an order, and even
then it passes the constitution (in the Agent) and the risk guard (in execution)
before a broker sees it.

---

## The four brains

| | what it is | what it contributes | when it is down |
|---|---|---|---|
| **engine_1** | `engine_1/btc_multi_horizon.py`, wrapped by `adapters/engine_one.py` | cross-horizon consensus from 15 m to 30 d, ATR-derived levels, optional news | the Agent is told, and says so |
| **engine_2** | `engine_2/`, wrapped by `adapters/engine_two.py` | PPO action + forecaster P(up) on the trading timeframe | the Agent is told, and says so |
| **engine_3** | `engine_3/` (new) | win probability, expected R, size multiplier, veto — learned from this desk's own outcomes | falls back to the heuristic floor |
| **the Agent** | `llm_agent/` | the decision, the size, the stop, the target, the rationale, and what would change its mind | deterministic policy in `llm_agent/fallback.py` |

**Every engine failure is a first-class outcome.** A dead engine returns
`EngineSignal(ok=False)`, which is shown to the Agent as "this engine is down —
do not assume it agrees with the other", and reduces size through engine_3. The
alternative — dropping the engine silently — produces confident answers from half
a system.

**engine_2 can be consumed two ways.** `ENGINE_2_SOURCE=inline` scores in-process
(needs TensorFlow and a promoted bundle) and is fed the *real* broker position so
the PPO state vector matches the book. `ENGINE_2_SOURCE=file` reads the JSONL its
own live loop writes, which keeps TensorFlow off this host entirely. `auto` tries
inline, then file, then reports the engine as down.

---

## The Agent

`llm_agent/agent.py` assembles the prompt in a fixed order — who you are, what you
may not do, what the engines said, what your own history says, what you hold, what
the theory says — then requires a strict JSON answer whose `rationale` field is
where the human voice lives.

Three properties matter:

1. **Restrictions are enforced, not merely described.** `enforce()` re-checks the
   answer against the same rules the prompt described and rewrites anything that
   strayed — an oversized position, a stop above entry, a BUY after a veto, a SELL
   with nothing held. Every rewrite is recorded in `compliance_notes`, so an
   operator can see what the model *wanted* to do.
2. **A missing LLM is not a missing decision.** No key, a timeout, malformed JSON
   or a hallucinated action all fall through to `llm_agent/fallback.py`, which
   produces the same structure with the same constraints and a rationale that
   explains it is the deterministic policy speaking.
3. **The knowledge base is read at the moment of the decision** (see below).

Provider order is Groq → local Ollama → deterministic. `LLM_ENABLED=0` skips
straight to deterministic.

---

## The knowledge base, refreshed without downtime

`llm_agent/trading_theories_knowledge_base.md` is thirteen `##` sections of
trading theory, each with a `**Tags:**` line. The whole file is never pasted into
a prompt — sections are *retrieved* by relevance to the situation (regime,
RSI extremes, engine conflict, drawdown, in/out of position), with the
rule-bearing sections always pinned.

**Editing it is live.** The store holds one immutable snapshot behind a single
attribute. A refresh parses the new text into a *new* snapshot and only then
rebinds the attribute, so a reload mid-cycle can never hand anyone a half-parsed
file. Three consequences:

* edit the file → the next decision uses it, within `KB_REFRESH_SECONDS`;
* a broken edit (no `##` headings) is **rejected** and the last good version keeps
  serving, with the parse error recorded;
* every good version is written to Postgres content-addressed by SHA-256, so a
  fresh host with no local file loads the knowledge base from the database.

`POST /admin/knowledge-base` publishes a new version atomically (validate → write
to a temp file → `os.replace` → reload) and returns the version now live.

---

## engine_3, the self-training risk engine

Full detail in [`engine_3/README.md`](engine_3/README.md). In short:

* **it learns from this system's own records** — closed trades (ground truth) and,
  before there are enough of those, "shadow" labels from every cycle's forward
  return, which makes it trainable from the first hour;
* **it trains itself on a schedule** (`ENGINE_3_TRAIN_INTERVAL_S`, default 6 h);
* **it is gated** — a candidate must beat both the incumbent *and* the heuristic
  floor on a chronological holdout, or it is stored and never served;
* **models live in Postgres as bytes** and load at start on any host;
* **retention keeps the newest ten versions plus whatever is active**, pruned after
  each evaluation.

---

## PAPER and REAL

Identical engines, Agent, constitution, risk guard, ledgers and statistics. The
only difference is which `Broker` fills the order.

* **PAPER** — real live prices, simulated money. Fills pay the same fee and
  slippage the backtest assumes (a round trip on an unchanged price costs exactly
  0.25% at the defaults). It cannot authenticate and cannot touch an exchange.
* **REAL** — real funds via ccxt. Three separate keys have to be turned:
  `TRADING_MODE=REAL`, API credentials, and `LIVE_TRADING_CONFIRMED=1`. Missing
  any one raises at construction, not at the first order. `ADMIN_TOKEN` is
  mandatory, and the admin API refuses to serve without it in REAL mode.

**The books never mix.** Every execution row carries a `mode` column and every
query filters on it. Switching modes shows a different account; it never merges,
overwrites or destroys the other mode's history.

**Hard caps and the kill switch.** `MAX_POSITION_PCT`, `MAX_CAPITAL_AT_RISK_PCT`,
`MAX_TRADES_PER_DAY`, `MAX_DAILY_LOSS_PCT` are environment ceilings; an admin rule
in the database may *tighten* any of them and can never loosen one. The kill
switch blocks new entries everywhere — immediately, mid-cycle — and deliberately
leaves exits open, because an operator hitting it wants the system to stop buying,
not to be trapped holding.

Stops and targets are checked **before** any model is consulted each cycle. A stop
is arithmetic; waiting for a language model to agree with arithmetic is not risk
management.

---

## Reliability

| failure | what happens |
|---|---|
| an engine times out or crashes | flagged `ok=False`, or served from the last known-good reading marked `stale`; the Agent is told and discounts it |
| the LLM is unreachable | deterministic policy produces the same answer shape |
| the knowledge base is broken | last good version keeps serving; the error is recorded |
| Postgres blips mid-write | the unit of work is spooled to a local JSONL outbox and replayed on reconnect (`replay_outbox`) |
| Postgres is down at boot | the process waits for it instead of trading without its memory |
| the price feed dies | three venues are tried, then the last good quote (if recent); if none, the cycle aborts without trading |
| the process dies mid-cycle | client order ids are derived from the cycle id, so a replay finds the order and does not fill twice |
| two hosts point at one database | a Postgres advisory lock elects one trader; the others serve the API in observer mode |

---

## Moving to another machine

Everything that matters is a row in Postgres: ledgers, positions, trades, both
equity curves, the knowledge base, the admin restrictions, the trained risk models
(as bytes) and the full decision audit trail. Local disk holds only caches.

So the migration is:

```bash
# on the new machine
git clone <repo> && cd spot_5
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env      # same DATABASE_URL as the old host
python backend/main.py
```

Start the new host before stopping the old one if you like — the advisory lock
makes sure only one of them trades.

To verify, or to move to a *different* database:

```bash
python backend/tools/migrate.py verify                     # what this database holds
python backend/tools/migrate.py export --out desk.json     # every table, models included
DATABASE_URL=<new> python backend/tools/migrate.py import --in desk.json
```

The import is idempotent, so it can be re-run after an interruption.

---

## Running it

```bash
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env        # then edit

python backend/main.py --check              # boot checks: db, kb, risk model, mode
python backend/main.py --once --no-trade    # one full cycle, decide but do not execute
python backend/main.py --once               # one cycle, execute if allowed
python backend/main.py --train              # one engine_3 auto-training cycle
python backend/main.py                      # API on :8000 plus the background loops
```

Set `ENGINE_1_OFFLINE=1` and `LLM_ENABLED=0` to exercise the whole pipeline with
no network at all.

### API

| | |
|---|---|
| `GET /health` | every dependency, the live KB version, the active risk model, the kill switch |
| `GET /state` | the book, the position, the last decision, the restrictions as briefed |
| `GET /decisions` `/trades` `/orders` `/equity` `/stats` | history, `?mode=PAPER\|REAL` |
| `GET /events` | the audit trail |
| `GET /knowledge-base` `/engine3/models` `/admin/rules` | what the Agent is reading and obeying |
| `PUT /admin/rules` | tighten the restrictions (versioned, audited) |
| `POST /admin/kill-switch` | stop new entries now |
| `POST /admin/knowledge-base` | publish new theory, live on the next decision |
| `POST /admin/engine3/train` | train, evaluate, promote if it wins, prune to ten |
| `POST /admin/cycle/run` | run one cycle on demand |
| `POST /admin/mode` | switch PAPER ⇄ REAL (REAL runs a preflight first) |
| `POST /admin/paper/reset` | wipe and re-fund the paper book; REAL is untouched |

Admin routes need `X-Admin-Token`.

### Tests

```bash
cd backend && python -m pytest tests -q      # 74 tests, no network, no LLM, no TensorFlow
```

They run against a real SQLAlchemy schema on SQLite, so the queries and the schema
are tested against each other rather than against mocks.

---

## Layout

```
backend/
  core/          config, Postgres schema and access, contracts, market data, locks
  adapters/      engine_1 and engine_2 normalized into one EngineSignal
  engine_1/      the context engine (unchanged)
  engine_2/      the quant engine (unchanged)
  engine_3/      the risk engine: features, dataset, model, registry, training, service
  llm_agent/     knowledge base, prompt, client, fallback, agent, the constitution
  execution/     Broker interface, PaperBroker, LiveBroker, portfolio, risk guard, trader
  pipeline/      the decision cycle and the scheduler
  api/           read routes and admin routes
  tools/         migrate.py — export, import, verify
  tests/         the suite
```

---

Educational tooling. Nothing here is financial advice, and a backtest is not a
forecast.
