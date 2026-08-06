from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from app.rag.chunker import chunk_sections
from app.rag.document_loader import load_documents
from app.rag.ingest import ingest
from app.rag.retriever import retrieve
from app.rag.vector_store import check_vector_store_freshness


class RagTests(unittest.TestCase):
    def test_loads_markdown_sections_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp)
            (source_dir / "profile.md").write_text(
                "# Sarah Nelson\nSarah Nelson is a careful local research assistant.\n\n"
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
            (source_dir / "sarah.txt").write_text(
                "Sarah Nelson is the local conversational agent for Sarah v2.0.",
                encoding="utf-8",
            )

            chunk_count = ingest(source=source_dir, vector_store_dir=store_dir)
            response = retrieve("Who is Sarah Nelson?", vector_store_dir=store_dir, top_k=1)

        self.assertGreaterEqual(chunk_count, 1)
        self.assertTrue(response.has_sufficient_evidence)
        self.assertEqual(response.results[0].source_filename, "sarah.txt")
        self.assertGreaterEqual(response.results[0].similarity_score, 0.15)

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
            response = retrieve("Who is Sarah Nelson?", vector_store_dir=store_dir, top_k=1)

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


if __name__ == "__main__":
    unittest.main()
