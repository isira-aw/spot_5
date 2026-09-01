"""Shared plumbing so one sick engine cannot take the pipeline down.

Every adapter gets, for free:

* a hard timeout (the engine runs on a worker thread; a hung network call cannot
  block the cycle),
* exception capture — a crash becomes ``EngineSignal(ok=False)``, not a traceback
  in the trading loop,
* a short in-process cache so a 15-minute cycle does not re-run a 60-second model
  when nothing has changed,
* a last-known-good fallback read from Postgres, clearly marked ``stale`` so the
  Agent can discount it in its own words.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from core.contracts import EngineSignal, utcnow

log = logging.getLogger("adapters")

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="engine")


class EngineAdapter:
    name = "engine"
    timeout_s = 120
    cache_s = 60

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._cache: tuple[float, EngineSignal] | None = None

    # subclasses implement this and nothing else
    def _compute(self, symbol: str, context: dict) -> EngineSignal:
        raise NotImplementedError

    def run(self, symbol: str, context: dict | None = None) -> EngineSignal:
        context = context or {}
        if not self.enabled:
            sig = EngineSignal.failed(self.name, symbol, "engine disabled by configuration")
            sig.source = "disabled"
            return sig

        with self._lock:
            hit = self._cache
        if hit and time.time() - hit[0] < self.cache_s and not context.get("force"):
            cached = hit[1]
            cached.source = cached.source or "cache"
            return cached

        started = time.perf_counter()
        try:
            future = _pool.submit(self._compute, symbol, context)
            signal = future.result(timeout=self.timeout_s)
            signal.latency_ms = int((time.perf_counter() - started) * 1000)
            signal.generated_at = signal.generated_at or utcnow()
            with self._lock:
                self._cache = (time.time(), signal)
            return signal
        except FutureTimeout:
            log.error("%s timed out after %ss", self.name, self.timeout_s)
            return self._degraded(symbol, f"timed out after {self.timeout_s}s",
                                  int((time.perf_counter() - started) * 1000))
        except Exception as exc:
            log.warning("%s failed: %s: %s", self.name, type(exc).__name__, exc,
                        exc_info=log.isEnabledFor(logging.DEBUG))
            return self._degraded(symbol, f"{type(exc).__name__}: {exc}",
                                  int((time.perf_counter() - started) * 1000))

    def _degraded(self, symbol: str, error: str, latency_ms: int) -> EngineSignal:
        """Prefer a stale-but-real opinion over no opinion, and label it as such."""
        try:
            from core.config import get_settings
            from core.repository import last_signal
            cached = last_signal(self.name, symbol, get_settings().engines.max_signal_age_s)
        except Exception:
            cached = None
        if cached is not None:
            cached.error = error
            cached.stale = True
            cached.latency_ms = latency_ms
            cached.reasons = list(cached.reasons) + [
                f"Live run failed ({error[:120]}); this is the last good reading, "
                f"{cached.age_seconds / 60:.0f} minutes old."]
            return cached
        return EngineSignal.failed(self.name, symbol, error, latency_ms)
