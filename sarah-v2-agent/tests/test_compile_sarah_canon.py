import unittest

from app.persona.compile_sarah_canon import (
    CanonSection,
    _build_source_packet,
    _ensure_section_references,
)
from app.rag.retriever import RetrievalResponse, RetrievalResult


class CompileSarahCanonTests(unittest.TestCase):
    def test_source_packet_includes_filename_location_chunk_and_score(self) -> None:
        retrieval = _retrieval()

        packet = _build_source_packet(retrieval)

        self.assertIn("filename: Alex Sarah.docx", packet)
        self.assertIn("location: section: Sarah CV", packet)
        self.assertIn("chunk_id: abc123", packet)
        self.assertIn("similarity_score: 0.4200", packet)

    def test_reference_guard_keeps_cited_draft(self) -> None:
        retrieval = _retrieval()
        draft = "## Sarah Biography\n\nSarah is a Mars commander (Alex Sarah.docx)."

        guarded = _ensure_section_references("Sarah Biography", draft, retrieval)

        self.assertEqual(guarded, draft)

    def test_reference_guard_marks_uncited_draft_unconfirmed(self) -> None:
        retrieval = _retrieval()
        draft = "Sarah is a Mars commander."

        guarded = _ensure_section_references("Sarah Biography", draft, retrieval)

        self.assertTrue(guarded.startswith("## Sarah Biography"))
        self.assertIn("UNCONFIRMED", guarded)

    def test_canon_section_shape(self) -> None:
        section = CanonSection("Title", "query", "instructions")

        self.assertEqual(section.top_k, 8)
        self.assertEqual(section.min_score, 0.12)


def _retrieval() -> RetrievalResponse:
    return RetrievalResponse(
        query="Sarah",
        has_sufficient_evidence=True,
        message="Retrieved evidence for the query.",
        results=[
            RetrievalResult(
                source_filename="Alex Sarah.docx",
                location="section: Sarah CV",
                chunk_id="abc123",
                similarity_score=0.42,
                text="Sarah Nelson is commander of the Discovery Mars Expedition.",
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()

