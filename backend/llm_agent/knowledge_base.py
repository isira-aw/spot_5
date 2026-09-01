"""The Agent's reading material, reloadable while the system is trading.

The knowledge base is a markdown file split into ``##`` sections. The whole file
is never pasted into a prompt — it would crowd out the actual market state and
cost tokens on every cycle — so sections are *retrieved* by relevance to the
situation in front of the Agent, with the rule-bearing sections always pinned.

**Refresh without downtime.** The live snapshot is a single immutable object held
behind one attribute. A refresh parses the new text into a *new* snapshot and only
then rebinds the attribute; readers hold their reference for the length of a
prompt build, so a reload mid-cycle can never hand anyone a half-parsed file. If
parsing fails, the old snapshot stays exactly where it is and the failure is
logged and recorded — a broken edit degrades to "still using the last good
version", never to an outage.

**Portability.** Every version that loads successfully is written to the database,
content-addressed by SHA-256. A fresh machine with no local file loads the last
active version straight from the database, so moving hosts does not lose the
knowledge base.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

from core.config import get_settings
from core.contracts import utcnow

log = logging.getLogger("llm_agent.kb")

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_TAGS_RE = re.compile(r"^\s*\*\*Tags:\*\*\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9_]+")

# Sections whose tags match any of these are always included: they carry the
# rules, and a rule the Agent did not read is a rule it will break.
PINNED_TAGS = ("process", "spot", "discipline")


@dataclass(frozen=True)
class Section:
    title: str
    body: str
    tags: tuple[str, ...] = ()
    index: int = 0

    @property
    def char_count(self) -> int:
        return len(self.body) + len(self.title)

    def render(self) -> str:
        return f"## {self.title}\n{self.body.strip()}"

    def terms(self) -> set[str]:
        return set(_WORD_RE.findall(f"{self.title} {' '.join(self.tags)} {self.body}".lower()))


@dataclass(frozen=True)
class KnowledgeBase:
    """An immutable snapshot. Swapped by reference, never mutated in place."""
    checksum: str
    label: str
    sections: tuple[Section, ...]
    source: str
    raw: str = ""
    loaded_at: datetime = field(default_factory=utcnow)

    @property
    def version(self) -> str:
        return f"{self.label}:{self.checksum[:10]}"

    @property
    def ok(self) -> bool:
        return bool(self.sections)

    def summary(self) -> dict:
        return {"version": self.version, "checksum": self.checksum, "label": self.label,
                "sections": len(self.sections), "chars": len(self.raw),
                "source": self.source, "loaded_at": self.loaded_at,
                "titles": [s.title for s in self.sections]}

    # ── retrieval ───────────────────────────────────────────────────────────
    def retrieve(self, query: Iterable[str], *, max_sections: int = 7,
                 max_chars: int = 9000) -> list[Section]:
        """Rank by tag/title/body overlap with the situation, pinned rules first."""
        terms = {t.lower() for chunk in query for t in _WORD_RE.findall(str(chunk).lower())
                 if len(t) > 2}
        scored: list[tuple[float, Section]] = []
        for s in self.sections:
            tag_set = {t.lower() for t in s.tags}
            title_terms = set(_WORD_RE.findall(s.title.lower()))
            body_terms = s.terms()
            score = (3.0 * len(terms & tag_set)
                     + 2.0 * len(terms & title_terms)
                     + 1.0 * len(terms & body_terms) / max(1, len(body_terms)) * 8.0)
            if tag_set & set(PINNED_TAGS):
                score += 100.0                       # pinned: always in the prompt
            scored.append((score, s))

        scored.sort(key=lambda t: (-t[0], t[1].index))
        # Pinned rule sections are always included and do not consume the budget:
        # the limits exist to keep *retrieved* theory from crowding out market state.
        chosen = [s for score, s in scored if score >= 100.0]
        used = sum(s.char_count for s in chosen)
        picked = 0
        for score, section in scored:
            if score >= 100.0:
                continue
            if picked >= max_sections or used + section.char_count > max_chars:
                continue
            chosen.append(section)
            used += section.char_count
            picked += 1
        chosen.sort(key=lambda s: s.index)
        return chosen

    def render(self, sections: Sequence[Section]) -> str:
        return "\n\n".join(s.render() for s in sections)


def parse(text: str, *, label: str = "kb", source: str = "file") -> KnowledgeBase:
    """Markdown -> snapshot. Raises if the text has no usable sections."""
    if not text or not text.strip():
        raise ValueError("knowledge base is empty")
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        raise ValueError("knowledge base has no '## ' sections")

    sections: list[Section] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        tag_match = _TAGS_RE.search(body)
        tags: tuple[str, ...] = ()
        if tag_match:
            tags = tuple(t.strip().lower() for t in tag_match.group(1).split(",") if t.strip())
            body = body[:tag_match.start()] + body[tag_match.end():]
        sections.append(Section(title=m.group(1).strip(), body=body.strip(),
                                tags=tags, index=i))

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return KnowledgeBase(checksum=digest, label=label, sections=tuple(sections),
                         source=source, raw=text)


EMPTY = KnowledgeBase(checksum="0" * 64, label="empty", sections=(), source="none", raw="")


class KnowledgeBaseStore:
    """Owns the live snapshot and the refresh policy."""

    def __init__(self, path: str | None = None, refresh_seconds: int | None = None):
        s = get_settings()
        self.path = path or s.kb.path
        self.refresh_seconds = (s.kb.refresh_seconds if refresh_seconds is None
                                else refresh_seconds)
        self.persist_to_db = s.kb.persist_to_db
        self._kb: KnowledgeBase = EMPTY
        self._checked_at = 0.0
        self._mtime = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.reload_count = 0

    # ── access ──────────────────────────────────────────────────────────────
    def get(self) -> KnowledgeBase:
        """Never blocks on I/O and never raises. Returns the current snapshot."""
        if self._kb is EMPTY and self._checked_at == 0.0:
            self.refresh(force=True)
        elif (self.refresh_seconds <= 0
              or time.time() - self._checked_at >= self.refresh_seconds):
            # refresh_seconds <= 0 means "check on every read", not "never check"
            self.refresh()
        return self._kb

    @property
    def current(self) -> KnowledgeBase:
        return self._kb

    # ── refresh ─────────────────────────────────────────────────────────────
    def refresh(self, force: bool = False) -> KnowledgeBase:
        with self._lock:
            self._checked_at = time.time()
            try:
                text, source, label = self._read(force)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("knowledge base read failed (%s); keeping %s",
                            self.last_error, self._kb.version)
                return self._kb
            if text is None:                                  # nothing changed
                return self._kb
            try:
                candidate = parse(text, label=label, source=source)
            except Exception as exc:
                self.last_error = f"parse failed: {exc}"
                log.error("knowledge base at %s did not parse (%s); still serving %s",
                          self.path, exc, self._kb.version)
                self._record_event(f"knowledge base parse failed: {exc}", level="error")
                return self._kb

            if candidate.checksum == self._kb.checksum and not force:
                return self._kb

            previous = self._kb
            self._kb = candidate                              # <- the atomic swap
            self.reload_count += 1
            self.last_error = None
            log.info("knowledge base %s -> %s (%d sections, %s)",
                     previous.version, candidate.version, len(candidate.sections), source)
            if previous.checksum != candidate.checksum:
                self._persist(candidate)
                self._record_event(
                    f"knowledge base reloaded: {previous.version} -> {candidate.version}",
                    payload=candidate.summary())
            return self._kb

    def _read(self, force: bool) -> tuple[str | None, str, str]:
        """File first, database second. ``None`` text means 'unchanged'."""
        if os.path.exists(self.path):
            mtime = os.path.getmtime(self.path)
            if not force and mtime == self._mtime:
                return None, "file", self._kb.label
            self._mtime = mtime
            with open(self.path, encoding="utf-8") as fh:
                text = fh.read()
            label = os.path.basename(self.path).replace(".md", "")
            return text, "file", label
        row = self._load_from_db()
        if row:
            if not force and row["checksum"] == self._kb.checksum:
                return None, "database", self._kb.label
            return row["content"], "database", row["label"]
        raise FileNotFoundError(f"no knowledge base at {self.path} and none in the database")

    @staticmethod
    def _load_from_db():
        try:
            from core.repository import active_kb_version
            return active_kb_version()
        except Exception as exc:
            log.debug("kb database read unavailable: %s", exc)
            return None

    def _persist(self, kb: KnowledgeBase) -> None:
        if not self.persist_to_db or kb.source == "database":
            return
        try:
            from core.repository import save_kb_version
            save_kb_version(kb.raw, label=kb.label, source=kb.source,
                            section_count=len(kb.sections))
        except Exception as exc:
            log.warning("knowledge base not saved to database: %s", exc)

    @staticmethod
    def _record_event(message: str, level: str = "info", payload: dict | None = None) -> None:
        try:
            from core.repository import record_event
            record_event(message, level=level, category="knowledge_base",
                         payload=payload or {})
        except Exception:
            pass

    # ── optional background watcher ─────────────────────────────────────────
    def start_watcher(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.wait(max(5, self.refresh_seconds)):
                try:
                    self.refresh()
                except Exception as exc:                       # pragma: no cover
                    log.error("kb watcher error: %s", exc)

        self._thread = threading.Thread(target=loop, name="kb-watcher", daemon=True)
        self._thread.start()
        log.info("knowledge base watcher started (every %ds)", self.refresh_seconds)

    def stop_watcher(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        kb = self._kb
        return {**kb.summary(), "path": self.path, "exists": os.path.exists(self.path),
                "refresh_seconds": self.refresh_seconds, "reloads": self.reload_count,
                "last_error": self.last_error,
                "last_checked": datetime.fromtimestamp(self._checked_at, tz=timezone.utc)
                if self._checked_at else None}


_store: KnowledgeBaseStore | None = None


def get_store() -> KnowledgeBaseStore:
    global _store
    if _store is None:
        _store = KnowledgeBaseStore()
    return _store
