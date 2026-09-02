"""The checks that stop the pipeline, as opposed to the checks that print.

In the notebook the sanity thresholds (`predStd`, `directionalAccuracy`) printed
a warning and execution carried on: PPO trained for an hour against a forecaster
that had collapsed to a constant, the export ran, and the model shipped. A
warning nobody reads is not a control.

Everything here raises `GateFailed`. The callers (retrain, jobs, the scheduler
task) treat that as "this cycle produced nothing; the live model is untouched",
which is always a safe outcome — the previous champion keeps serving.

Thresholds live in config.py so they can be tightened per environment without a
code change.
"""
from __future__ import annotations

from . import config as C


class GateFailed(RuntimeError):
    """A quality gate rejected the artifact. Nothing downstream may run."""

    def __init__(self, stage: str, reasons: list[str], metrics: dict | None = None):
        self.stage, self.reasons, self.metrics = stage, reasons, metrics or {}
        super().__init__(f"{stage} gate failed: " + "; ".join(reasons))


def check_forecaster(health: dict, *, stage: str = "forecaster") -> dict:
    """Runs on VALIDATION metrics, before a single PPO update is spent."""
    reasons = []
    if health["pred_std"] < C.GATE_MIN_PRED_STD:
        reasons.append(f"predStd {health['pred_std']:.4f} < {C.GATE_MIN_PRED_STD} "
                       f"(forecaster has collapsed toward a constant)")
    if health["dir_acc"] < C.GATE_MIN_DIR_ACC:
        reasons.append(f"directionalAccuracy {health['dir_acc']:.4f} < "
                       f"{C.GATE_MIN_DIR_ACC} (no better than a coin flip after costs)")
    if abs(health["pred_mean"] - 0.5) > C.GATE_MAX_PRED_MEAN_DEV:
        reasons.append(f"predMean {health['pred_mean']:.3f} is "
                       f"{abs(health['pred_mean'] - 0.5):.3f} off 0.5 "
                       f"(the model is predicting one direction regardless of input)")
    if reasons:
        raise GateFailed(stage, reasons, health)
    return health


def check_policy(diag: dict, *, stage: str = "policy") -> dict:
    """A policy that answers the same thing to every state is not a policy.

    `policy_spread` is the mean, over the three actions, of the standard
    deviation of P(action) across sampled states. The notebook's shipped agent
    scored near zero: ~36/28/36 whatever the market did.
    """
    spread = diag.get("final_policy_spread", 0.0)
    if spread < C.GATE_MIN_POLICY_SPREAD:
        raise GateFailed(stage, [
            f"policy spread {spread:.4f} < {C.GATE_MIN_POLICY_SPREAD}: the policy "
            f"outputs near-identical action probabilities across unrelated market "
            f"states (collapsed to a near-constant policy)"], diag)
    return diag


def check_backtest(metrics: dict, floors: dict, *, stage: str = "backtest") -> dict:
    """Absolute floors on out-of-sample trading performance."""
    reasons = [f"{k}={metrics.get(k, float('nan')):.4g} below floor {floor}"
               for k, floor in floors.items()
               if not (metrics.get(k) is not None and metrics[k] >= floor)]
    if reasons:
        raise GateFailed(stage, reasons, {k: metrics.get(k) for k in floors})
    return metrics
