from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from app.rag.document_loader import DocumentSection


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_filename: str
    source_path: str
    location: str
    text: str
    token_count: int


def chunk_sections(
    sections: list[DocumentSection],
    target_tokens: int = 1000,
    min_tokens: int = 800,
    overlap_tokens: int = 150,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for section in sections:
        chunks.extend(
            _chunk_section(
                section=section,
                target_tokens=target_tokens,
                min_tokens=min_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return chunks


def estimate_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _chunk_section(
    section: DocumentSection,
    target_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
) -> list[DocumentChunk]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section.text) if part.strip()]
    if not paragraphs:
        return []

    units = _split_large_paragraphs(paragraphs, target_tokens)
    chunk_texts: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        should_flush = current and current_tokens + unit_tokens > target_tokens
        if should_flush and current_tokens >= min_tokens:
            chunk_texts.append("\n\n".join(current))
            current = _overlap_tail(current, overlap_tokens)
            current_tokens = estimate_tokens("\n\n".join(current))

        current.append(unit)
        current_tokens += unit_tokens

        if current_tokens >= target_tokens:
            chunk_texts.append("\n\n".join(current))
            current = _overlap_tail(current, overlap_tokens)
            current_tokens = estimate_tokens("\n\n".join(current))

    if current:
        chunk_text = "\n\n".join(current)
        if chunk_text not in chunk_texts:
            chunk_texts.append(chunk_text)

    chunks: list[DocumentChunk] = []
    for index, text in enumerate(chunk_texts, start=1):
        chunks.append(
            DocumentChunk(
                chunk_id=_chunk_id(section, index),
                source_filename=section.source_filename,
                source_path=str(section.source_path),
                location=section.location,
                text=text,
                token_count=estimate_tokens(text),
            )
        )
    return chunks


def _split_large_paragraphs(paragraphs: list[str], target_tokens: int) -> list[str]:
    units: list[str] = []
    for paragraph in paragraphs:
        if estimate_tokens(paragraph) <= target_tokens:
            units.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            if current and current_tokens + sentence_tokens > target_tokens:
                units.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sentence)
            current_tokens += sentence_tokens
        if current:
            units.append(" ".join(current))
    return units


def _overlap_tail(parts: list[str], overlap_tokens: int) -> list[str]:
    tail: list[str] = []
    total = 0
    for part in reversed(parts):
        tail.insert(0, part)
        total += estimate_tokens(part)
        if total >= overlap_tokens:
            break
    return tail


def _chunk_id(section: DocumentSection, index: int) -> str:
    raw = f"{section.source_path}|{section.location}|{index}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]

