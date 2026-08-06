from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
import re
from pathlib import Path
from typing import Any


class MemoryType(StrEnum):
    USER_PROFILE = "user_profile"
    AGENT_PROFILE = "agent_profile"
    RELATIONSHIP_CONTINUITY = "relationship_continuity"
    RECURRING_PROJECTS = "recurring_projects"
    PREFERENCES = "preferences"
    CANON_FACTS = "canon_facts"
    UNRESOLVED_QUESTIONS = "unresolved_questions"


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    memory_type: str
    text: str
    created_at: str
    updated_at: str
    source: str
    tags: list[str]
    active: bool = True


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def remember(
        self,
        text: str,
        memory_type: MemoryType | str | None = None,
        source: str = "explicit_user_command",
    ) -> MemoryRecord:
        clean_text = _clean_text(text)
        if not clean_text:
            raise ValueError("Cannot store an empty memory.")

        now = _now()
        record = MemoryRecord(
            memory_id=_memory_id(clean_text, now),
            memory_type=str(memory_type or infer_memory_type(clean_text)),
            text=clean_text,
            created_at=now,
            updated_at=now,
            source=source,
            tags=_tags(clean_text),
            active=True,
        )
        self._append(record)
        return record

    def forget(self, keyword: str) -> list[MemoryRecord]:
        keyword_lower = keyword.lower().strip()
        if not keyword_lower:
            return []

        records = self.list_memories(include_inactive=True)
        forgotten: list[MemoryRecord] = []
        rewritten: list[MemoryRecord] = []
        for record in records:
            if record.active and keyword_lower in record.text.lower():
                forgotten_record = _replace_record(record, active=False, updated_at=_now())
                forgotten.append(forgotten_record)
                rewritten.append(forgotten_record)
            else:
                rewritten.append(record)

        self._rewrite(rewritten)
        return forgotten

    def list_memories(self, include_inactive: bool = False) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            record = MemoryRecord(**data)
            if include_inactive or record.active:
                records.append(record)
        return records

    def retrieve(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        query_terms = _expanded_query_terms(query)
        if not query_terms:
            return []

        scored: list[tuple[int, MemoryRecord]] = []
        for record in self.list_memories():
            record_terms = set(record.tags) | set(_tokens(record.text))
            score = len(query_terms & record_terms)
            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda pair: (pair[0], pair[1].updated_at), reverse=True)
        return [record for _score, record in scored[:limit]]

    def search(self, keyword: str, limit: int | None = None) -> list[MemoryRecord]:
        keyword_clean = keyword.lower().strip()
        if not keyword_clean:
            return []

        keyword_terms = _expanded_query_terms(keyword_clean)
        keyword_variants = _query_variants(keyword_clean)
        matches: list[tuple[int, MemoryRecord]] = []
        for record in self.list_memories():
            text_lower = record.text.lower()
            text_normalized = _normalize_search_text(record.text)
            tags = set(record.tags)
            score = 0
            if any(variant in text_lower or _normalize_search_text(variant) in text_normalized for variant in keyword_variants):
                score += 10
            score += len(keyword_terms & (tags | set(_tokens(record.text))))
            if score > 0:
                matches.append((score, record))

        matches.sort(key=lambda pair: (pair[0], pair[1].updated_at), reverse=True)
        records = [record for _score, record in matches]
        return records if limit is None else records[:limit]

    def _append(self, record: MemoryRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

    def _rewrite(self, records: list[MemoryRecord]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


def infer_memory_type(text: str) -> MemoryType:
    text_lower = text.lower()
    if text_lower.endswith("?") or "unresolved" in text_lower:
        return MemoryType.UNRESOLVED_QUESTIONS
    if any(term in text_lower for term in ("prefer", "likes", "dislikes", "wants", "style", "call me")):
        return MemoryType.PREFERENCES
    if any(term in text_lower for term in ("project", "build", "working on", "sarah v2", "novel", "repo")):
        return MemoryType.RECURRING_PROJECTS
    if any(term in text_lower for term in ("canon", "sarah", "darwin", "tak", "paul", "mars", "war & peace")):
        return MemoryType.CANON_FACTS
    if any(term in text_lower for term in ("joke", "banter", "relationship", "warmth", "continuity", "alex and sarah")):
        return MemoryType.RELATIONSHIP_CONTINUITY
    return MemoryType.USER_PROFILE


def build_memory_context(memories: list[MemoryRecord]) -> str:
    lines = [
        "Relevant remembered context from FDR memory:",
        "Use these only as durable continuity hints. Do not overstate them, and do not imply surveillance.",
    ]
    if not memories:
        lines.append("No relevant FDR memories injected for this turn.")
        return "\n".join(lines)

    for memory in memories:
        lines.append(f"* {memory.text}")
    lines.extend(
        [
            "",
            "Use these memories as background context. Do not quote them unless useful. Do not expose raw JSONL.",
        ]
    )
    return "\n".join(lines)


def format_memories(memories: list[MemoryRecord]) -> str:
    if not memories:
        return "No active memories stored."

    lines: list[str] = []
    for memory in memories:
        lines.append(f"- {memory.memory_type}: {memory.text} ({memory.memory_id})")
    return "\n".join(lines)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2]


def _expanded_query_terms(text: str) -> set[str]:
    terms = set(_tokens(text)) | set(_tokens(_normalize_search_text(text)))
    if _is_debate_alex_query(text):
        terms.update(
            {
                "debate_alex",
                "debate",
                "alex",
                "proxy",
                "alex_proxy",
                "judge",
                "crew",
                "turn",
                "order",
                "first",
                "speaker",
                "max",
                "words",
                "dinner",
                "style",
            }
        )
    return terms


def _query_variants(text: str) -> set[str]:
    cleaned = text.lower().strip()
    variants = {cleaned, cleaned.replace("_", " "), cleaned.replace("-", " "), cleaned.replace("_", "-")}
    if _is_debate_alex_query(text):
        variants.update(
            {
                "debate_alex",
                "debate alex",
                "debate",
                "alex-proxy",
                "alex proxy",
                "judge",
                "crew",
                "turn order",
                "first speaker",
                "max words",
                "dinner style",
            }
        )
    return {variant for variant in variants if variant}


def _is_debate_alex_query(text: str) -> bool:
    normalized = _normalize_search_text(text)
    raw = text.lower()
    triggers = (
        "debate_alex",
        "debate alex",
        "alex-proxy",
        "alex proxy",
        "turn order",
        "first speaker",
        "dinner style",
    )
    return any(trigger in raw or trigger in normalized for trigger in triggers)


def _normalize_search_text(text: str) -> str:
    return re.sub(r"[_-]+", " ", text.lower())


def _tags(text: str) -> list[str]:
    return sorted(set(_tokens(text)) - {"the", "and", "for", "with", "that", "this"})


def _memory_id(text: str, timestamp: str) -> str:
    return hashlib.sha1(f"{timestamp}:{text}".encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _replace_record(record: MemoryRecord, **changes: Any) -> MemoryRecord:
    data = asdict(record)
    data.update(changes)
    return MemoryRecord(**data)
