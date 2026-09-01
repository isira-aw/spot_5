# llm_agent — the voice

Four brains, one voice. This package is the voice: it takes both forecasting
engines, the risk engine, the book and the operator's restrictions, and produces
one answer — BUY / SELL / HOLD, confidence, size, stop, target — with a rationale
that reads like a person explaining themselves, including what would change their
mind.

```
knowledge_base.py   the theory, retrieved by relevance, hot-reloadable
prompt.py           the prompt, assembled in a fixed order
client.py           Groq → Ollama → nothing, with a forgiving JSON extractor
fallback.py         the deterministic policy when there is no model to speak with
agent.py            the Agent, and the constitution that binds it
```

## The order the prompt is built in

Who you are → **what you may not do** → what the engines said → what your own
history says → what you are holding → the relevant theory → the output contract.

The restrictions come *before* the market data on purpose. A constraint appended
after the evidence reads as an afterthought; a constraint stated before it frames
everything that follows.

## The constitution

The prompt tells the Agent the rules. `enforce()` then checks the answer against
the same rules and rewrites anything that strayed:

| the Agent proposed | what happens |
|---|---|
| 90% of equity | capped to the smaller of the position cap and the risk budget over the stop distance |
| a stop above entry | moved below entry |
| a target below entry | moved above entry |
| BUY after a risk-engine veto | becomes HOLD |
| BUY under the confidence floor | becomes HOLD |
| BUY with the kill switch on | becomes HOLD |
| SELL with nothing held | becomes HOLD (spot book: bearish means cash) |
| an order under the minimum size | becomes HOLD |
| no stop at all | one is supplied |

Every rewrite is appended to `compliance_notes` and stored with the decision.
Nothing is silently clipped, so a model that keeps trying to exceed a cap is
visible in the record rather than quietly trimmed.

## The knowledge base

`trading_theories_knowledge_base.md` — thirteen sections, each with a `**Tags:**`
line. Format contract for editors: one `##` heading per section, an optional
`**Tags:**` line under it, then the body. Nothing else is required.

Retrieval scores sections against the current situation (regime, RSI extremes,
whether the engines conflict, drawdown, in position or flat) and always pins the
rule-bearing sections, which do not consume the size budget.

**Refresh has no downtime.** The live snapshot is one immutable object behind one
attribute; a refresh parses the new text into a new snapshot and only then rebinds
the attribute. A reader holding a reference for the length of a prompt build can
never see a half-parsed file. A broken edit is rejected and the previous version
keeps serving. Every good version is stored in the database by SHA-256, so a host with
no local file loads the knowledge base from the database.

```bash
# edit the file — live within KB_REFRESH_SECONDS (0 = every decision)
$EDITOR backend/llm_agent/trading_theories_knowledge_base.md

# or publish atomically over HTTP
curl -XPOST localhost:8000/admin/knowledge-base \
     -H "X-Admin-Token: ..." -H 'Content-Type: application/json' \
     -d '{"content": "# KB\n\n## New Section\n**Tags:** regime\n\n..."}'

curl localhost:8000/knowledge-base        # which version is live right now
```

## Admin restrictions

`AdminRestrictions.as_prompt_lines()` renders the operator's policy the way you
would brief a human trader — "Never risk more than 2.00% of equity on one trade",
"At most 12 trades per day", "KILL SWITCH IS ON: no new entries under any
circumstances. Exits are still allowed" — and free-text `notes` are passed through
verbatim, so "No trading during CPI releases" reaches the Agent as written.

The same object is what the constitution and the risk guard enforce, so what the
Agent is told and what the system permits cannot drift apart.

## When there is no LLM

`fallback.py` produces the same structure from the engines' numbers alone:
weighted consensus (engine_2 0.55, engine_1 0.45), halved on conflict, discounted
when running on one engine, sized from the stop distance and the risk budget, with
a rationale that says plainly that it is the deterministic policy speaking. It is
also the reference implementation of the decision policy — if the LLM's answer
disagrees with what this would do, that difference is in the logs and is worth
reviewing.
