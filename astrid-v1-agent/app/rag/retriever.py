from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import load_settings
from app.rag.embeddings import LOCAL_EMBEDDING_MODEL, cosine_similarity, embed_texts
from app.rag.vector_store import read_vector_store


DEFAULT_MIN_SCORE = 0.15
ASTRID_CANON_PRIORITY_FILES = ("astrid 2_0_a.docx",)
ASTRID_SOURCE_TERMS = ("astrid",)
SARAH_SOURCE_TERMS = ("alex sarah", "darwin sarah", "sarah canon", "sarah_profile", "sarah.txt")


@dataclass(frozen=True)
class RetrievalResult:
    source_filename: str
    location: str
    chunk_id: str
    similarity_score: float
    text: str


@dataclass(frozen=True)
class RetrievalResponse:
    query: str
    has_sufficient_evidence: bool
    results: list[RetrievalResult]
    message: str


def retrieve(
    query: str,
    vector_store_dir: Path | None = None,
    top_k: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
) -> RetrievalResponse:
    settings = load_settings()
    store_dir = vector_store_dir or settings.vector_store_dir
    manifest, stored_chunks = read_vector_store(store_dir)

    stored_model = manifest.get("embedding_model", settings.embedding_model)
    if stored_model != LOCAL_EMBEDDING_MODEL and not settings.openai_api_key:
        raise RuntimeError(
            "This vector store was built with OpenAI embeddings, but OPENAI_API_KEY is not set."
        )

    query_embedding = embed_texts(
        [query],
        openai_api_key=settings.openai_api_key if stored_model != LOCAL_EMBEDDING_MODEL else "",
        model=stored_model,
    )[0]
    origin_warning = (
        None
        if vector_store_dir is not None
        else _vector_store_origin_warning(manifest, stored_chunks, settings.source_docs_dir)
    )

    scored = [
        (_prioritized_score(cosine_similarity(query_embedding, stored.embedding), stored.chunk.source_filename), stored.chunk)
        for stored in stored_chunks
    ]
    scored.sort(key=lambda item: item[0], reverse=True)

    results = [
        RetrievalResult(
            source_filename=chunk.source_filename,
            location=chunk.location,
            chunk_id=chunk.chunk_id,
            similarity_score=score,
            text=chunk.text,
        )
        for score, chunk in scored[:top_k]
    ]

    has_sufficient_evidence = bool(results and results[0].similarity_score >= min_score)
    message = (
        "Retrieved evidence for the query."
        if has_sufficient_evidence
        else "Insufficient evidence in the indexed documents. Astrid should not answer this from the docs."
    )
    if origin_warning:
        has_sufficient_evidence = False
        message = f"{origin_warning} {message}"
    return RetrievalResponse(
        query=query,
        has_sufficient_evidence=has_sufficient_evidence,
        results=results,
        message=message,
    )


def _prioritized_score(score: float, source_filename: str) -> float:
    normalized = source_filename.lower()
    if normalized in ASTRID_CANON_PRIORITY_FILES:
        return score + 0.08
    if any(term in normalized for term in ASTRID_SOURCE_TERMS):
        return score + 0.04
    if any(term in normalized for term in SARAH_SOURCE_TERMS):
        return score - 0.03
    return score


def _vector_store_origin_warning(manifest: dict[str, Any], stored_chunks: list[Any], astrid_source_docs_dir: Path) -> str | None:
    source_dir = manifest.get("source_dir")
    if source_dir:
        try:
            if Path(str(source_dir)).resolve() != astrid_source_docs_dir.resolve():
                return (
                    "WARNING: This vector store was not built from Astrid's data/source_docs. "
                    "Re-ingestion is required before Astrid should trust local canon."
                )
        except OSError:
            return (
                "WARNING: This vector store source path could not be verified against Astrid's data/source_docs. "
                "Re-ingestion is required before Astrid should trust local canon."
            )

    source_files = [str(name).lower() for name in manifest.get("source_files", [])]
    has_astrid_source = any(any(term in filename for term in ASTRID_SOURCE_TERMS) for filename in source_files)
    has_sarah_source = any(any(term in filename for term in SARAH_SOURCE_TERMS) for filename in source_files)
    if has_sarah_source and not has_astrid_source:
        return (
            "WARNING: This vector store appears to contain Sarah-centered source docs and no Astrid canon. "
            "Re-ingestion is required before Astrid should trust local canon."
        )

    chunk_paths = [stored.chunk.source_path.lower() for stored in stored_chunks[:20]]
    if any("sarah-v2-agent" in path or "files-mentioned-by-the-user-pasted" in path for path in chunk_paths):
        return (
            "WARNING: This vector store appears to have been created from Sarah/external source docs. "
            "Re-ingestion is required before Astrid should trust local canon."
        )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks from Astrid's vector store.")
    parser.add_argument("--query", required=True, help="Question or search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum similarity score required before evidence is considered sufficient.",
    )
    parser.add_argument(
        "--vector-store",
        type=Path,
        default=None,
        help="Vector store folder. Defaults to data/vector_store.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    response = retrieve(
        query=args.query,
        vector_store_dir=args.vector_store,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    print(response.message)
    for index, result in enumerate(response.results, start=1):
        print()
        print(f"[{index}] {result.source_filename} | {result.location}")
        print(f"chunk id: {result.chunk_id}")
        print(f"similarity score: {result.similarity_score:.4f}")
        print(result.text)


if __name__ == "__main__":
    main()
