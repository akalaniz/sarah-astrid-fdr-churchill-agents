from __future__ import annotations

from dataclasses import asdict

from app.rag.retriever import RetrievalResponse, RetrievalResult


MAX_CONTEXT_CHUNKS = 5
MAX_EXCERPT_CHARS = 1200


def build_context_packet(
    retrieval: RetrievalResponse | None,
    retrieval_error: str | None = None,
) -> str:
    lines = [
        "LOCAL RAG CONTEXT FOR THE LATEST USER MESSAGE:",
        "Use this material as grounding, not as text to dump into the answer.",
        "When making canon claims from retrieved local documents, cite source filenames inline near the claim.",
        "If a canon claim is not supported by retrieved local evidence, label it as persona premise, inference, or speculation.",
        "If evidence is weak, absent, or unrelated, say so instead of inventing canon facts.",
    ]

    if retrieval_error:
        lines.append(f"Retrieval status: unavailable ({retrieval_error})")
        return "\n".join(lines)

    if retrieval is None or not retrieval.results:
        lines.append("Retrieval status: no local chunks found.")
        return "\n".join(lines)

    status = "sufficient" if retrieval.has_sufficient_evidence else "insufficient"
    lines.append(f"Retrieval status: {status}.")
    lines.append(f"Retriever message: {retrieval.message}")

    for index, result in enumerate(retrieval.results[:MAX_CONTEXT_CHUNKS], start=1):
        excerpt = _excerpt(result.text)
        lines.extend(
            [
                "",
                f"[{index}] source: {result.source_filename}",
                f"location: {result.location}",
                f"chunk id: {result.chunk_id}",
                f"similarity score: {result.similarity_score:.4f}",
                f"excerpt: {excerpt}",
            ]
        )

    return "\n".join(lines)


def sources_for_transcript(retrieval: RetrievalResponse | None) -> list[dict[str, object]]:
    if retrieval is None:
        return []
    return [asdict(result) for result in retrieval.results[:MAX_CONTEXT_CHUNKS]]


def format_sources(results: list[RetrievalResult]) -> str:
    if not results:
        return "No sources were used for the last answer."

    lines: list[str] = []
    for index, result in enumerate(results[:MAX_CONTEXT_CHUNKS], start=1):
        lines.extend(
            [
                f"[{index}] {result.source_filename} | {result.location}",
                f"chunk id: {result.chunk_id}",
                f"similarity score: {result.similarity_score:.4f}",
                _excerpt(result.text, max_chars=700),
                "",
            ]
        )
    return "\n".join(lines).strip()


def _excerpt(text: str, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."
