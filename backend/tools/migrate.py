"""Moving the desk to another machine, and proving nothing was left behind.

The normal path needs no tool at all: every piece of state — ledgers, trades,
equity curves, the knowledge base, the admin restrictions and the trained risk
models — is a row in ``backend/var/spot5.db``, so copying that one file to a new
host makes it the same desk. Start it, and it picks up where the old one stopped.

This tool is for the other two cases:

* merging or reshaping a desk, where a readable JSON export/import beats copying
  an opaque binary file;
* proving the move worked, with ``verify`` — row counts, the active knowledge base,
  the active risk model and both books' statistics, on either side.

    python backend/tools/migrate.py verify
    python backend/tools/migrate.py export --out desk.json
    DB_PATH=<new.db> python backend/tools/migrate.py import --in desk.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sqlalchemy import func, select                              # noqa: E402

from core.config import MODES, get_settings                      # noqa: E402
from core.db import init_db, session_scope                       # noqa: E402
from core.logging_setup import setup_logging                     # noqa: E402
from core.tables import ALL_TABLES                              # noqa: E402

FORMAT_VERSION = 1


def _encode(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode()}
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if "__bytes__" in value:
            return base64.b64decode(value["__bytes__"])
        if "__datetime__" in value:
            return datetime.fromisoformat(value["__datetime__"])
    return value


def export(path: str) -> dict:
    counts: dict[str, int] = {}
    payload: dict[str, Any] = {"format": FORMAT_VERSION,
                               "exported_at": datetime.utcnow().isoformat(),
                               "source": get_settings().db.safe_url(), "tables": {}}
    with session_scope() as s:
        for table in ALL_TABLES:
            rows = s.execute(select(table)).scalars().all()
            payload["tables"][table.__tablename__] = [
                {c.name: _encode(getattr(r, c.name)) for c in r.__table__.columns}
                for r in rows]
            counts[table.__tablename__] = len(rows)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    size_mb = os.path.getsize(path) / 1e6
    print(f"exported {sum(counts.values())} rows to {path} ({size_mb:.2f} MB)")
    for name, n in counts.items():
        if n:
            print(f"  {name:<20} {n}")
    return counts


def import_(path: str, *, wipe: bool = False) -> dict:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("format") != FORMAT_VERSION:
        raise SystemExit(f"unsupported export format: {payload.get('format')}")

    by_name = {t.__tablename__: t for t in ALL_TABLES}
    counts: dict[str, int] = {}
    with session_scope() as s:
        if wipe:
            for table in reversed(ALL_TABLES):
                s.execute(table.__table__.delete())
            print("existing rows deleted")
        for name, rows in payload["tables"].items():
            table = by_name.get(name)
            if table is None:
                print(f"  skipping unknown table {name}")
                continue
            existing = {r[0] for r in s.execute(
                select(list(table.__table__.primary_key.columns)[0]))}
            pk = list(table.__table__.primary_key.columns)[0].name
            added = 0
            for row in rows:
                if row.get(pk) in existing:
                    continue                      # import is resumable and idempotent
                s.add(table(**{k: _decode(v) for k, v in row.items()}))
                added += 1
            counts[name] = added
    print(f"imported {sum(counts.values())} rows from {path}")
    for name, n in counts.items():
        if n:
            print(f"  {name:<20} {n}")
    return counts


def verify() -> dict:
    from core.repository import active_admin_rules, active_kb_version, active_risk_model
    from execution.portfolio import PortfolioStore

    report: dict[str, Any] = {"database": get_settings().db.safe_url(), "tables": {}}
    with session_scope() as s:
        for table in ALL_TABLES:
            report["tables"][table.__tablename__] = int(
                s.execute(select(func.count()).select_from(table)).scalar() or 0)

    kb = active_kb_version()
    report["knowledge_base"] = {"label": kb["label"], "checksum": kb["checksum"][:12],
                                "sections": kb["section_count"],
                                "chars": kb["char_count"]} if kb else None
    model = active_risk_model()
    report["risk_model"] = {"version": model["version"], "kind": model["kind"],
                            "samples": model["trained_on_samples"],
                            "bytes": len(model["artifact"] or b"")} if model else None
    rules = active_admin_rules()
    report["admin_rules"] = {"version": rules.version, "updated_by": rules.updated_by,
                             "kill_switch": rules.kill_switch}
    report["books"] = {m: PortfolioStore(m).stats() for m in MODES}

    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="spot_5 database portability tool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="dump every table to JSON")
    e.add_argument("--out", default="spot5_export.json")
    i = sub.add_parser("import", help="load a JSON dump into the current database")
    i.add_argument("--in", dest="src", required=True)
    i.add_argument("--wipe", action="store_true", help="delete existing rows first")
    sub.add_parser("verify", help="report what this database currently holds")

    args = ap.parse_args()
    setup_logging()
    init_db()
    if args.cmd == "export":
        export(args.out)
    elif args.cmd == "import":
        import_(args.src, wipe=args.wipe)
    else:
        verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
