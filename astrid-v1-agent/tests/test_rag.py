from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.rag.chunker import chunk_sections
from app.rag.document_loader import load_documents
from app.rag.ingest import ingest
from app.rag.retriever import retrieve
from app.rag.vector_store import check_vector_store_freshness, read_vector_store


class RagTests(unittest.TestCase):
    def test_loads_markdown_sections_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp)
            (source_dir / "profile.md").write_text(
                "# Astrid\nAstrid is a careful local research assistant.\n\n"
                "# Notes\nShe avoids making claims without evidence.\n",
                encoding="utf-8",
            )

            sections = load_documents(source_dir)
            chunks = chunk_sections(sections, target_tokens=40, min_tokens=10, overlap_tokens=5)

        self.assertEqual(len(sections), 2)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_filename, "profile.md")
        self.assertIn("section:", chunks[0].location)

    def test_ingest_and_retrieve_finds_evidence(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self._test_ingest_and_retrieve_finds_evidence()

    def _test_ingest_and_retrieve_finds_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            store_dir = root / "store"
            source_dir.mkdir()
            (source_dir / "astrid.txt").write_text(
                "Astrid is the local conversational agent for Astrid v1.0.",
                encoding="utf-8",
            )

            chunk_count = ingest(source=source_dir, vector_store_dir=store_dir)
            response = retrieve("Who is Astrid?", vector_store_dir=store_dir, top_k=1)

        self.assertGreaterEqual(chunk_count, 1)
        self.assertTrue(response.has_sufficient_evidence)
        self.assertEqual(response.results[0].source_filename, "astrid.txt")
        self.assertGreaterEqual(response.results[0].similarity_score, 0.15)

    def test_ingest_manifest_marks_astrid_source_origin(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source_dir = root / "data" / "source_docs"
                store_dir = root / "data" / "vector_store"
                source_dir.mkdir(parents=True)
                _write_docx(
                    source_dir / "Astrid 2_0_a.docx",
                    "Astrid Bach is a propulsion engineer who refuses sterile futures.",
                )

                ingest(source=source_dir, vector_store_dir=store_dir)
                manifest, _chunks = read_vector_store(store_dir)

        self.assertEqual(manifest["agent"], "astrid")
        self.assertEqual(manifest["source_origin"], "astrid_local_source_docs")
        self.assertEqual(Path(manifest["source_dir"]).name, "source_docs")
        self.assertIn("Astrid 2_0_a.docx", manifest["source_files"])

    def test_retrieval_prioritizes_astrid_canon_over_sarah_voice(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source_dir = root / "source"
                store_dir = root / "store"
                source_dir.mkdir()
                _write_docx(
                    source_dir / "Alex Sarah.docx",
                    "Sarah has command authority voice and command trauma center.",
                )
                _write_docx(
                    source_dir / "Astrid 2_0_a.docx",
                    "Astrid Bach is embodied warmth, propulsion, engines, weather, sex, laughter, Steve, and chosen love.",
                )

                ingest(source=source_dir, vector_store_dir=store_dir)
                response = retrieve("Astrid embodied propulsion warmth chosen love", vector_store_dir=store_dir, top_k=2)

        self.assertEqual(response.results[0].source_filename, "Astrid 2_0_a.docx")

    def test_default_retrieval_warns_when_store_was_built_from_sarah_docs(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                astrid_source_dir = root / "data" / "source_docs"
                sarah_source_dir = root / "sarah-v2-agent" / "data" / "source_docs"
                store_dir = root / "data" / "vector_store"
                astrid_source_dir.mkdir(parents=True)
                sarah_source_dir.mkdir(parents=True)
                _write_docx(sarah_source_dir / "Alex Sarah.docx", "Sarah Nelson has command authority voice.")
                ingest(source=sarah_source_dir, vector_store_dir=store_dir)

                settings = load_settings(env_file=root / ".env")
                settings = settings.__class__(
                    **{
                        **settings.__dict__,
                        "source_docs_dir": astrid_source_dir,
                        "external_source_docs_dir": astrid_source_dir,
                        "vector_store_dir": store_dir,
                    }
                )
                with patch("app.rag.retriever.load_settings", return_value=settings):
                    response = retrieve("Who is Astrid?")

        self.assertFalse(response.has_sufficient_evidence)
        self.assertIn("WARNING: This vector store", response.message)
        self.assertIn("Re-ingestion is required", response.message)

    def test_retrieval_reports_insufficient_evidence(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self._test_retrieval_reports_insufficient_evidence()

    def _test_retrieval_reports_insufficient_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            store_dir = root / "store"
            source_dir.mkdir()
            (source_dir / "weather.txt").write_text(
                "The document only discusses weather station calibration.",
                encoding="utf-8",
            )

            ingest(source=source_dir, vector_store_dir=store_dir)
            response = retrieve("Who is Astrid?", vector_store_dir=store_dir, top_k=1)

        self.assertFalse(response.has_sufficient_evidence)
        self.assertIn("Insufficient evidence", response.message)

    def test_rag_freshness_detects_source_doc_changes_after_ingest(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source_dir = root / "source"
                store_dir = root / "store"
                source_dir.mkdir()
                source_file = source_dir / "astrid.md"
                source_file.write_text("Astrid says embodied love still has bodies.", encoding="utf-8")

                ingest(source=source_dir, vector_store_dir=store_dir)
                fresh = check_vector_store_freshness(source_dir, store_dir)
                source_file.write_text("Astrid says embodied love still has bodies and sensation.", encoding="utf-8")
                stale = check_vector_store_freshness(source_dir, store_dir)

        self.assertTrue(fresh["fresh"])
        self.assertFalse(stale["fresh"])
        self.assertIn("astrid.md", stale["changed_since_ingest"])


def _write_docx(path: Path, text: str) -> None:
    from docx import Document

    document = Document()
    document.add_paragraph(text)
    document.save(path)


if __name__ == "__main__":
    unittest.main()
