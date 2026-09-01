"""Logging that is readable in a terminal and greppable in a log aggregator."""
from __future__ import annotations

import logging
import sys

from .config import get_settings

_configured = False


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.short = record.name.replace("backend.", "")
        return super().format(record)


def setup_logging(level: str | None = None) -> None:
    global _configured
    if _configured:
        return
    level = (level or get_settings().log_level).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_Formatter(
        fmt="%(asctime)s %(levelname)-7s %(short)-22s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    for noisy in ("urllib3", "httpx", "httpcore", "sqlalchemy.engine.Engine", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
