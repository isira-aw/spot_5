"""Versioned model bundles on disk, with promotion and rollback.

The notebook overwrote `models/` on every run: if a bad model shipped, the good
one it replaced no longer existed. Here every training cycle writes a NEW
immutable version directory named by UTC date and the git commit that produced
it, and `models/` is only ever a copy of a version that passed the gates:

    models_versions/20260902T0310Z-a1b2c3d/    forecaster/ ppo/ scaler.npz meta.json
    models_versions/20260901T0309Z-9f8e7d6/
    models/                                    <- copy of the promoted version
    models/CURRENT.json                        <- which version, when, and why

`prune` keeps the newest MODEL_RETENTION versions plus whatever is current, and
runs only AFTER a promotion decision, so a candidate is never deleted in the same
breath that it earns its place (the same rule engine_3's registry follows).

Rollback is `promote(previous_version)`: the bundle is still there, byte for
byte, with the metrics it was judged on.

    python -m engine_2.registry --list
    python -m engine_2.registry --rollback 20260901T0309Z-9f8e7d6
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

from . import config as C

BUNDLE_PARTS = ("forecaster", "ppo/policy", "ppo/value")
CURRENT_FILE = "CURRENT.json"


def git_sha(short: bool = True) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
                             cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def new_version_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%MZ}-{git_sha()}"


def version_dir(version: str) -> str:
    return os.path.join(C.VERSIONS_DIR, version)


def is_bundle(path: str) -> bool:
    return all(os.path.exists(os.path.join(path, p, "model.keras")) for p in BUNDLE_PARTS)


def list_versions() -> list[dict]:
    """Newest first."""
    out = []
    for name in sorted(os.listdir(C.VERSIONS_DIR), reverse=True):
        path = version_dir(name)
        if not os.path.isdir(path) or not is_bundle(path):
            continue
        meta = {}
        try:
            with open(os.path.join(path, "meta.json")) as fh:
                meta = json.load(fh)
        except Exception:
            pass
        out.append({"version": name, "path": path, "meta": meta,
                    "created_at": os.path.getmtime(path),
                    "current": name == current_version()})
    return out


def current_version() -> str | None:
    try:
        with open(os.path.join(C.MODELS_DIR, CURRENT_FILE)) as fh:
            return json.load(fh).get("version")
    except Exception:
        return None


def current_info() -> dict:
    try:
        with open(os.path.join(C.MODELS_DIR, CURRENT_FILE)) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _copy_bundle(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for part in BUNDLE_PARTS:
        os.makedirs(os.path.join(dst, part), exist_ok=True)
        shutil.copy(os.path.join(src, part, "model.keras"),
                    os.path.join(dst, part, "model.keras"))
    for extra in ("scaler.npz", "meta.json"):
        if os.path.exists(os.path.join(src, extra)):
            shutil.copy(os.path.join(src, extra), os.path.join(dst, extra))


def register(candidate_dir: str = C.CANDIDATE_DIR, version: str | None = None,
             meta: dict | None = None) -> str:
    """Freeze a freshly trained candidate as an immutable version. Not live yet."""
    if not is_bundle(candidate_dir):
        raise FileNotFoundError(f"{candidate_dir} is not a complete model bundle")
    version = version or new_version_id()
    dst = version_dir(version)
    if os.path.exists(dst):
        version = f"{version}-{int(time.time()) % 10000}"
        dst = version_dir(version)
    _copy_bundle(candidate_dir, dst)
    payload = {"version": version, "registered_at": datetime.now(timezone.utc).isoformat(),
               "git": git_sha(), "symbol": C.SYMBOL, "timeframe": C.TIMEFRAME,
               **(meta or {})}
    existing = {}
    if os.path.exists(os.path.join(dst, "meta.json")):
        with open(os.path.join(dst, "meta.json")) as fh:
            existing = json.load(fh)
    with open(os.path.join(dst, "meta.json"), "w") as fh:
        json.dump({**existing, **payload}, fh, indent=2, default=float)
    return version


def promote(version: str, reason: str = "", metrics: dict | None = None) -> dict:
    """Make a registered version the one `models/` serves. Also the rollback path."""
    src = version_dir(version)
    if not is_bundle(src):
        raise FileNotFoundError(f"no complete bundle at {src}")
    previous = current_version()
    _copy_bundle(src, C.MODELS_DIR)
    info = {"version": version, "previous": previous, "reason": reason,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {}}
    with open(os.path.join(C.MODELS_DIR, CURRENT_FILE), "w") as fh:
        json.dump(info, fh, indent=2, default=float)
    return info


def rollback(version: str | None = None) -> dict:
    """Back to the version we were serving before the last promotion, or to a
    named one. Everything needed is already on disk."""
    target = version or current_info().get("previous")
    if not target:
        raise RuntimeError("no previous version recorded — nothing to roll back to")
    return promote(target, reason=f"manual rollback from {current_version()}")


def prune(keep: int = C.MODEL_RETENTION) -> list[str]:
    """Delete all but the newest `keep` versions and the current one."""
    versions = list_versions()
    protected = {current_version()} | {v["version"] for v in versions[:keep]}
    removed = []
    for v in versions:
        if v["version"] not in protected:
            shutil.rmtree(v["path"], ignore_errors=True)
            removed.append(v["version"])
    return removed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--promote", metavar="VERSION")
    ap.add_argument("--rollback", nargs="?", const="", metavar="VERSION")
    ap.add_argument("--prune", type=int, nargs="?", const=C.MODEL_RETENTION)
    a = ap.parse_args()
    if a.promote:
        print(promote(a.promote, reason="manual promotion"))
    elif a.rollback is not None:
        print(rollback(a.rollback or None))
    elif a.prune is not None:
        print("pruned:", prune(a.prune) or "nothing")
    else:
        cur = current_version()
        for v in list_versions():
            mark = " <- current" if v["version"] == cur else ""
            sharpe = (v["meta"].get("metrics") or {}).get("sharpe")
            print(f"{v['version']}  sharpe={sharpe if sharpe is not None else 'n/a'}{mark}")
