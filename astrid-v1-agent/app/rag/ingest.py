from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.core.config import load_settings
from app.core.logging import configure_logging
from app.rag.chunker import chunk_sections
from app.rag.document_loader import load_documents
from app.rag.embeddings import embed_texts, get_embedding_model_name
from app.rag.vector_store import write_vector_store


logger = logging.getLogger(__name__)


def ingest(source: Path | None = None, vector_store_dir: Path | None = None) -> int:
    settings = load_settings()
    source_dir = source or settings.source_docs_dir
    store_dir = vector_store_dir or settings.vector_store_dir

    sections = load_documents(source_dir)
    chunks = chunk_sections(sections)
    embedding_model = get_embedding_model_name(settings.openai_api_key, settings.embedding_model)
    embeddings = embed_texts(
        [chunk.text for chunk in chunks],
        openai_api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    write_vector_store(store_dir, chunks, embeddings, embedding_model, source_dir=source_dir)

    logger.info("Ingested %s sections into %s chunks", len(sections), len(chunks))
    return len(chunks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest local documents into Astrid's vector store.")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Folder containing .txt, .md, .pdf, and .docx source documents.",
    )
    parser.add_argument(
        "--vector-store",
        type=Path,
        default=None,
        help="Destination vector store folder. Defaults to data/vector_store.",
    )
    return parser


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    args = build_parser().parse_args()
    count = ingest(source=args.source, vector_store_dir=args.vector_store)
    print(f"Ingested {count} chunks into {args.vector_store or settings.vector_store_dir}")


if __name__ == "__main__":
    main()
