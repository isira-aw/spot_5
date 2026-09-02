"""Does the live forecaster still work?

Validation metrics answer "was this model good on data from before it shipped".
They say nothing about last Tuesday. This module keeps a rolling window of live
predictions, matches each one against what the market actually did HORIZON bars
later, and recomputes the same two numbers the promotion gate used —
`directionalAccuracy` and `predStd` — on production data.

When directional accuracy sits below DRIFT_MIN_DIR_ACC for
DRIFT_BREACHES_TO_ALERT consecutive checks (not one bad hour — a sustained
period), the monitor flags `retrain_recommended`. The scheduler turns that into
an actual retrain; a human reading /health sees the same flag.

State lives in the backend database (`app_state` key `engine_2_drift`) when it is
reachable, exactly like every other piece of cross-process state in this repo,
and falls back to a JSONL file under reports/ when engine_2 is being run
standalone on a GPU box with no database.

    python -m engine_2.drift --report
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque

import numpy as np

from . import config as C

STATE_KEY = "engine_2_drift"
FALLBACK_PATH = os.path.join(C.REPORTS_DIR, "drift_state.json")


# ── storage: database if this is running inside the backend, file if not ─────
def _repository():
    try:
        from core import repository            # noqa: PLC0415 - optional dependency
        return repository
    except Exception:
        return None


def _load_state() -> dict:
    repo = _repository()
    if repo is not None:
        try:
            return repo.get_state(STATE_KEY, None) or _empty()
        except Exception:
            pass
    try:
        with open(FALLBACK_PATH) as fh:
            return json.load(fh)
    except Exception:
        return _empty()


def _save_state(state: dict) -> None:
    repo = _repository()
    if repo is not None:
        try:
            repo.set_state(STATE_KEY, state, updated_by="engine_2.drift")
            return
        except Exception:
            pass
    os.makedirs(os.path.dirname(FALLBACK_PATH), exist_ok=True)
    with open(FALLBACK_PATH, "w") as fh:
        json.dump(state, fh, default=float)


def _empty() -> dict:
    return {"pending": [], "observations": [], "breaches": 0,
            "model_version": None, "last_check": None}


# ── recording ────────────────────────────────────────────────────────────────
def record_prediction(ts_ms: int, close: float, p_up, model_version: str | None = None,
                      state: dict | None = None) -> dict:
    """Called once per live decision. Cheap: it only appends."""
    st = state if state is not None else _load_state()
    if model_version and st.get("model_version") != model_version:
        # a promotion resets the window; metrics from the old model would
        # otherwise be blamed on the new one for the next few hours
        st = _empty() | {"model_version": model_version}
    p = [float(x) for x in (p_up or [])]
    st["pending"].append({"ts": int(ts_ms), "close": float(close), "p_up": p})
    st["pending"] = st["pending"][-(C.DRIFT_WINDOW + C.HORIZON * 4):]
    if state is None:
        _save_state(st)
    return st


def resolve(ts_ms: int, close: float, state: dict | None = None) -> dict:
    """Match matured predictions against the realized close.

    A prediction made at bar t is judged at bar t+HORIZON, which for 15m bars is
    an hour later — so `pending` always holds the last few unresolved ones.
    """
    st = state if state is not None else _load_state()
    horizon_ms = C.HORIZON * C.BAR_MS
    still_pending = []
    for row in st.get("pending", []):
        age = int(ts_ms) - int(row["ts"])
        if age < horizon_ms:
            still_pending.append(row)
            continue
        if age > horizon_ms + 2 * C.BAR_MS:      # a gap swallowed its maturity bar
            continue
        realized_up = int(float(close) > float(row["close"]))
        p1 = float(row["p_up"][0]) if row["p_up"] else 0.5
        st.setdefault("observations", []).append({
            "ts": row["ts"], "p_up": p1, "realized_up": realized_up,
            "correct": int((p1 > 0.5) == bool(realized_up)),
            "ret": float(close) / float(row["close"]) - 1.0})
    st["pending"] = still_pending
    st["observations"] = st.get("observations", [])[-C.DRIFT_WINDOW:]
    if state is None:
        _save_state(st)
    return st


def observe(ts_ms: int, close: float, p_up, model_version: str | None = None) -> dict:
    """One live decision in, current drift status out. The only call site needs."""
    st = _load_state()
    st = resolve(ts_ms, close, st)
    st = record_prediction(ts_ms, close, p_up, model_version, st)
    status = evaluate(st)
    st["last_check"] = status
    _save_state(st)
    return status


# ── judgement ────────────────────────────────────────────────────────────────
def evaluate(state: dict | None = None) -> dict:
    st = state if state is not None else _load_state()
    obs = st.get("observations", [])
    n = len(obs)
    if n < C.DRIFT_MIN_SAMPLES:
        return {"ok": True, "verdict": "warming_up", "n": n,
                "needed": C.DRIFT_MIN_SAMPLES, "retrain_recommended": False,
                "model_version": st.get("model_version")}

    p = np.array([o["p_up"] for o in obs], dtype=float)
    correct = np.array([o["correct"] for o in obs], dtype=float)
    dir_acc = float(correct.mean())
    pred_std = float(p.std())
    # recent half vs older half: a slow decay shows up here before the mean moves
    half = n // 2
    recent_acc = float(correct[half:].mean())

    breaches = int(st.get("breaches", 0))
    failing = dir_acc < C.DRIFT_MIN_DIR_ACC or pred_std < C.DRIFT_MIN_PRED_STD
    breaches = breaches + 1 if failing else 0
    st["breaches"] = breaches

    reasons = []
    if dir_acc < C.DRIFT_MIN_DIR_ACC:
        reasons.append(f"live directionalAccuracy {dir_acc:.4f} < {C.DRIFT_MIN_DIR_ACC}")
    if pred_std < C.DRIFT_MIN_PRED_STD:
        reasons.append(f"live predStd {pred_std:.4f} < {C.DRIFT_MIN_PRED_STD} "
                       f"(forecaster has gone flat in production)")

    return {"ok": not failing, "verdict": "degraded" if failing else "healthy",
            "n": n, "dir_acc": dir_acc, "recent_dir_acc": recent_acc,
            "pred_std": pred_std, "pred_mean": float(p.mean()),
            "breaches": breaches, "reasons": reasons,
            "retrain_recommended": breaches >= C.DRIFT_BREACHES_TO_ALERT,
            "model_version": st.get("model_version"),
            "checked_at": int(time.time())}


def status() -> dict:
    """Read-only view for /health and the dashboard."""
    st = _load_state()
    return {"pending": len(st.get("pending", [])), **evaluate(st)}


def reset() -> None:
    _save_state(_empty())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.reset:
        reset(); print("drift window cleared")
    else:
        print(json.dumps(status(), indent=2, default=float))
