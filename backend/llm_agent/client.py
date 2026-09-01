"""Getting words out of a model, with somewhere to fall when it will not answer.

Order: Groq (fast, hosted, OpenAI-compatible) -> local Ollama -> nothing. The
caller decides what "nothing" means; :mod:`llm_agent.fallback` turns it into a
deterministic answer in the same shape, so a dead API degrades the *prose*, never
the pipeline.

The JSON extractor is deliberately forgiving. Models wrap JSON in prose, in code
fences, and — with reasoning models — after a ``<think>`` block. All three are
handled, because a decision thrown away over a stray backtick is a decision the
system did not make.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from core.config import get_settings

log = logging.getLogger("llm_agent.client")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class LLMResponse:
    ok: bool
    text: str = ""
    parsed: dict | None = None
    source: str = "none"
    latency_ms: int = 0
    error: str | None = None
    usage: dict[str, Any] | None = None


def extract_json(text: str) -> dict | None:
    """Pull the first complete JSON object out of whatever the model produced."""
    if not text:
        return None
    cleaned = _THINK_RE.sub("", text).strip()
    for fence in _FENCE_RE.findall(cleaned):
        try:
            obj = json.loads(fence.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    start = cleaned.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)
    return None


class LLMClient:
    def __init__(self, settings=None):
        self.s = (settings or get_settings()).llm
        self._groq_model: str | None = None
        self._groq_checked = 0.0

    # ── Groq ────────────────────────────────────────────────────────────────
    def _resolve_groq_model(self) -> str | None:
        """Pick a model the account is actually served, once every ten minutes."""
        if self._groq_model and time.time() - self._groq_checked < 600:
            return self._groq_model
        try:
            r = requests.get(f"{self.s.groq_base_url}/models",
                             headers={"Authorization": f"Bearer {self.s.groq_api_key}"},
                             timeout=15)
            r.raise_for_status()
            served = {m.get("id") for m in r.json().get("data", [])}
        except Exception as exc:
            log.info("groq model list unavailable (%s); trying configured model", exc)
            served = set()
        for candidate in [self.s.groq_model, *self.s.groq_fallback_models]:
            if not served or candidate in served:
                self._groq_model, self._groq_checked = candidate, time.time()
                return candidate
        self._groq_model = next(iter(served), None)
        self._groq_checked = time.time()
        return self._groq_model

    def _call_groq(self, system: str, user: str) -> LLMResponse:
        if not self.s.groq_api_key:
            return LLMResponse(False, error="no GROQ_API_KEY configured", source="groq")
        model = self._resolve_groq_model()
        if not model:
            return LLMResponse(False, error="no groq model available", source="groq")
        started = time.perf_counter()
        try:
            r = requests.post(
                f"{self.s.groq_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.s.groq_api_key}",
                         "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                      "temperature": self.s.temperature,
                      "max_tokens": self.s.max_tokens,
                      "response_format": {"type": "json_object"}},
                timeout=self.s.timeout_s)
            latency = int((time.perf_counter() - started) * 1000)
            if r.status_code >= 400:
                return LLMResponse(False, source=f"groq:{model}", latency_ms=latency,
                                   error=f"HTTP {r.status_code}: {r.text[:300]}")
            body = r.json()
            text = body["choices"][0]["message"]["content"]
            return LLMResponse(True, text=text, parsed=extract_json(text),
                               source=f"groq:{model}", latency_ms=latency,
                               usage=body.get("usage"))
        except Exception as exc:
            return LLMResponse(False, source=f"groq:{model}",
                               latency_ms=int((time.perf_counter() - started) * 1000),
                               error=f"{type(exc).__name__}: {exc}")

    # ── Ollama ──────────────────────────────────────────────────────────────
    def _call_ollama(self, system: str, user: str) -> LLMResponse:
        started = time.perf_counter()
        try:
            r = requests.post(
                f"{self.s.ollama_host}/api/chat",
                json={"model": self.s.ollama_model, "stream": False, "format": "json",
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                      "options": {"temperature": self.s.temperature}},
                timeout=self.s.timeout_s)
            latency = int((time.perf_counter() - started) * 1000)
            if r.status_code >= 400:
                return LLMResponse(False, source=f"ollama:{self.s.ollama_model}",
                                   latency_ms=latency,
                                   error=f"HTTP {r.status_code}: {r.text[:300]}")
            text = r.json().get("message", {}).get("content", "")
            return LLMResponse(True, text=text, parsed=extract_json(text),
                               source=f"ollama:{self.s.ollama_model}", latency_ms=latency)
        except Exception as exc:
            return LLMResponse(False, source=f"ollama:{self.s.ollama_model}",
                               latency_ms=int((time.perf_counter() - started) * 1000),
                               error=f"{type(exc).__name__}: {exc}")

    # ── public ──────────────────────────────────────────────────────────────
    def complete(self, system: str, user: str) -> LLMResponse:
        if not self.s.enabled:
            return LLMResponse(False, error="LLM disabled by configuration", source="disabled")
        errors = []
        for name, fn in (("groq", self._call_groq), ("ollama", self._call_ollama)):
            resp = fn(system, user)
            if resp.ok and resp.parsed:
                log.info("%s answered in %dms", resp.source, resp.latency_ms)
                return resp
            reason = resp.error or "returned no parseable JSON"
            errors.append(f"{name}: {reason}")
            log.info("%s unusable (%s)", name, reason)
        return LLMResponse(False, error=" | ".join(errors), source="none")


_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
