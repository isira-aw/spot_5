"""The scheduled retraining job, end to end, in one command.

    python -m engine_2.retrain                 # full cycle, gated promotion
    python -m engine_2.retrain --walkforward   # K-fold walk-forward check first

Kept as a thin wrapper over `jobs.cycle` so existing crontabs and runbooks keep
working; `jobs.py` is where the stages live and is what the in-process scheduler
calls.

Order matters, and every step is a hard gate:

  fetch -> dataset -> [walk-forward] -> forecaster (gate) -> PPO (gate)
        -> test backtest (floors + beat random) -> HOLDOUT backtest -> promote

A failure anywhere leaves the live model untouched, which is always the safe
outcome: the previous champion keeps serving and `models_versions/` still holds
every bundle needed to roll back further.
"""
from __future__ import annotations

import argparse
import json

from . import config as C
from .jobs import cycle


def main(years=C.HISTORY_YEARS, folds=5, do_walkforward=False,
         forecaster_epochs=60, ppo_updates=200, skip_fetch=False,
         warm_start=True) -> dict:
    result = cycle(years=years, epochs=forecaster_epochs, ppo_updates=ppo_updates,
                   walkforward=do_walkforward, folds=folds, warm_start=warm_start,
                   skip_fetch=skip_fetch)
    print(f"\nretrain cycle finished in {result['elapsed_s']}s "
          f"(ok={result.get('ok')}, promoted="
          f"{(result.get('promote') or {}).get('promoted', False)})")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=C.HISTORY_YEARS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ppo-updates", type=int, default=200)
    ap.add_argument("--no-warm-start", action="store_true")
    a = ap.parse_args()
    print(json.dumps(main(a.years, a.folds, a.walkforward, a.epochs, a.ppo_updates,
                          a.skip_fetch, not a.no_warm_start), indent=2, default=float))
