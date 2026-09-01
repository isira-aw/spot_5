"""Entry point: the API, the scheduler and a one-shot CLI.

    python backend/main.py                 # serve the API and run the loops
    python backend/main.py --once          # run exactly one decision cycle, print it
    python backend/main.py --once --no-trade   # ... and do not place the order
    python backend/main.py --train         # run an engine_3 auto-training cycle
    python backend/main.py --check         # boot checks only: db, kb, engines, mode

Boot order matters: the database has to answer before anything else is worth
starting, because the knowledge base, the risk model, the ledgers and the
restrictions all live in it. The database is a local SQLite file created on
first run, so this normally succeeds immediately; if it cannot be opened the
process stops rather than starting in a state where it would trade without its
own memory.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from contextlib import asynccontextmanager                       # noqa: E402

from fastapi import FastAPI                                      # noqa: E402
from fastapi.middleware.cors import CORSMiddleware               # noqa: E402
from fastapi.responses import JSONResponse                       # noqa: E402

from api.admin import router as admin_router                     # noqa: E402
from api.routes import router as read_router                     # noqa: E402
from core import repository                                      # noqa: E402
from core.config import PAPER, get_settings                      # noqa: E402
from core.db import init_db, last_wait_error, wait_for_db                         # noqa: E402
from core.logging_setup import setup_logging                     # noqa: E402
from engine_3.service import get_risk_engine                     # noqa: E402
from execution import risk_guard                                 # noqa: E402
from execution.portfolio import PortfolioStore                   # noqa: E402
from llm_agent.knowledge_base import get_store as get_kb_store   # noqa: E402
from pipeline.scheduler import get_scheduler                     # noqa: E402

log = logging.getLogger("main")


def boot(wait_seconds: int = 120) -> dict:
    """Bring the system up in dependency order and report what happened."""
    setup_logging()
    settings = get_settings()
    log.info("spot_5 starting: mode=%s symbol=%s env=%s",
             settings.execution.mode, settings.execution.symbol, settings.env)

    if not wait_for_db(wait_seconds):
        raise RuntimeError(f"database unusable after {wait_seconds}s: {settings.db.safe_url()}"
                           + (f" — {last_wait_error()}" if last_wait_error() else ""))
    init_db()

    problems = risk_guard.live_mode_preflight()
    if problems:
        for p in problems:
            log.error("REAL mode preflight: %s", p)
        raise RuntimeError("REAL mode preflight failed: " + "; ".join(problems))

    kb = get_kb_store().get()
    if not kb.ok:
        log.warning("no knowledge base loaded — the Agent will run without theory")
    else:
        log.info("knowledge base %s (%d sections)", kb.version, len(kb.sections))

    risk = get_risk_engine().load()
    log.info("risk model: %s v%s (%s samples)", risk.get("kind"), risk.get("version"),
             risk.get("trained_on_samples"))

    store = PortfolioStore(settings.execution.mode)
    if settings.execution.mode == PAPER:
        store.ensure_funded()

    repository.record_event(f"process started in {settings.execution.mode} mode",
                            category="lifecycle", mode=settings.execution.mode,
                            payload={"instance": settings.instance_id,
                                     "kb": kb.version, "risk_model": risk.get("version")})
    return {"mode": settings.execution.mode, "kb": kb.version, "risk_model": risk,
            "cash": store.cash(), "kill_switch": risk_guard.is_killed()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = boot()
    settings = get_settings()
    if settings.autostart_scheduler:
        get_scheduler().start()
    else:
        log.info("scheduler autostart is off; POST /admin/scheduler/start to run it")
    app.state.boot = state
    try:
        yield
    finally:
        get_scheduler().stop()
        repository.record_event("process stopping", category="lifecycle",
                                mode=settings.execution.mode)


app = FastAPI(
    title="spot_5 — autonomous spot trading desk",
    version="1.0.0",
    lifespan=lifespan,
    description=(
        "Two forecasting engines, a self-training risk engine and one LLM Agent that "
        "reasons over all three and answers in plain English. The Agent's answer is the "
        "only thing that becomes an order. PAPER and REAL share every component except "
        "the broker, and keep entirely separate books."),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
app.include_router(read_router, tags=["read"])
app.include_router(admin_router, tags=["admin"])


@app.get("/", include_in_schema=False)
def root():
    s = get_settings()
    return JSONResponse({
        "name": "spot_5", "mode": s.execution.mode, "symbol": s.execution.symbol,
        "docs": "/docs", "health": "/health", "state": "/state",
        "brains": ["engine_1 (context/calibration)", "engine_2 (quant/ML)",
                   "engine_3 (risk, self-trained)", "llm_agent (the voice)"],
    })


# ── CLI ─────────────────────────────────────────────────────────────────────
def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description="spot_5 trading desk")
    ap.add_argument("--once", action="store_true", help="run one decision cycle and exit")
    ap.add_argument("--no-trade", action="store_true", help="with --once: decide, do not execute")
    ap.add_argument("--train", action="store_true", help="run an engine_3 training cycle and exit")
    ap.add_argument("--check", action="store_true", help="boot checks only")
    ap.add_argument("--mode", default=None, help="PAPER or REAL (overrides TRADING_MODE)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    if args.mode:
        os.environ["TRADING_MODE"] = args.mode.upper()
        get_settings(refresh=True)

    if args.check:
        _print(boot(wait_seconds=30))
        return 0

    if args.train:
        boot(wait_seconds=30)
        from engine_3 import train
        _print(train.run())
        return 0

    if args.once:
        boot(wait_seconds=30)
        from pipeline.orchestrator import get_orchestrator
        result = get_orchestrator().run_cycle(autotrade=not args.no_trade)
        _print(result.to_dict())
        if result.decision:
            print("\n" + "=" * 78)
            print(f"{result.decision.headline()}   [{result.decision.source}]")
            print("=" * 78)
            print(result.decision.rationale)
            if result.decision.change_my_mind:
                print("\nWhat would change my mind:")
                for item in result.decision.change_my_mind:
                    print(f"  - {item}")
        return 0

    import uvicorn
    s = get_settings()
    uvicorn.run("main:app", host=args.host or s.api_host, port=args.port or s.api_port,
                reload=False, log_config=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
