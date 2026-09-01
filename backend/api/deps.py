"""Admin authentication.

One shared token in a header. Deliberately simple, and deliberately strict where
it matters: in REAL mode an unset ``ADMIN_TOKEN`` disables the admin surface
entirely rather than leaving the kill switch and the position caps open to
anyone who can reach the port.
"""
from __future__ import annotations

import hmac
import logging

from fastapi import Header, HTTPException, status

from core.config import PAPER, get_settings

log = logging.getLogger("api.auth")
_warned = False


def require_admin(x_admin_token: str | None = Header(default=None)) -> str:
    global _warned
    settings = get_settings()
    expected = settings.admin_token

    if not expected:
        if settings.execution.mode != PAPER:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "ADMIN_TOKEN is not set. The admin API is disabled in REAL mode until "
                "it is configured.")
        if not _warned:
            log.warning("ADMIN_TOKEN is not set — the admin API is open in PAPER mode")
            _warned = True
        return "anonymous-paper"

    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-Admin-Token")
    return "admin"
