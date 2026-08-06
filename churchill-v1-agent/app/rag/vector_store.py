from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from app.rag.chunker import DocumentChunk


CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class StoredChunk:
    chunk: DocumentChunk
    embedding: list[float]


def write_vector_store(
    vector_store_dir: Path,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
    embedding_model: str,
    source_dir: Path | None = None,
) -> None:
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "agent": "churchill",
        "embedding_model": embedding_model,
        "chunk_count": len(chunks),
        "chunks_file": CHUNKS_FILE,
        "source_files": sorted({chunk.source_filename for chunk in chunks}),
    }
    if source_dir is not None:
        manifest["source_dir"] = str(source_dir.resolve())
        manifest["source_origin"] = "churchill_local_source_docs"
        manifest["source_fingerprints"] = _source_fingerprints(source_dir)
    (vector_store_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with (vector_store_dir / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
        for chunk, embedding in zip(chunks, embeddings):
            record = {"chunk": asdict(chunk), "embedding": embedding}
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_vector_store(vector_store_dir: Path) -> tuple[dict[str, Any], list[StoredChunk]]:
    manifest_path = vector_store_dir / MANIFEST_FILE
    chunks_path = vector_store_dir / CHUNKS_FILE

    if not manifest_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Vector store is missing at {vector_store_dir}. Run python -m app.rag.ingest first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_chunks: list[StoredChunk] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            chunk = DocumentChunk(**record["chunk"])
            stored_chunks.append(StoredChunk(chunk=chunk, embedding=record["embedding"]))

    return manifest, stored_chunks


def check_vector_store_freshness(source_dir: Path, vector_store_dir: Path) -> dict[str, Any]:
    manifest, _stored_chunks = read_vector_store(vector_store_dir)
    indexed = manifest.get("source_fingerprints")
    current = _source_fingerprints(source_dir)
    if not indexed:
        return {
            "fresh": False,
            "reason": "Vector store manifest has no source_fingerprints. Re-ingest source docs.",
            "missing_from_index": sorted(current),
            "changed_since_ingest": [],
            "deleted_since_ingest": [],
        }

    indexed_keys = set(indexed)
    current_keys = set(current)
    missing_from_index = sorted(current_keys - indexed_keys)
    deleted_since_ingest = sorted(indexed_keys - current_keys)
    changed_since_ingest = sorted(
        filename
        for filename in indexed_keys & current_keys
        if indexed[filename] != current[filename]
    )
    stale = bool(missing_from_index or deleted_since_ingest or changed_since_ingest)
    return {
        "fresh": not stale,
        "reason": "Vector store is fresh." if not stale else "Source docs changed after ingestion. Re-ingest source docs.",
        "missing_from_index": missing_from_index,
        "changed_since_ingest": changed_since_ingest,
        "deleted_since_ingest": deleted_since_ingest,
    }


def _source_fingerprints(source_dir: Path) -> dict[str, dict[str, int]]:
    fingerprints: dict[str, dict[str, int]] = {}
    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
            continue
        stat = path.stat()
        fingerprints[path.name] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return fingerprints
