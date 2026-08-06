from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.core.conversation import ConversationMemory, TranscriptWriter
from app.core.rag_context import build_context_packet, format_sources
from app.rag.retriever import RetrievalResponse, RetrievalResult


class ChatCoreTests(unittest.TestCase):
    def test_memory_keeps_last_20_turns(self) -> None:
        memory = ConversationMemory(max_turns=20)
        for index in range(25):
            memory.add_turn(f"user {index}", f"assistant {index}")

        turns = memory.turns
        self.assertEqual(len(turns), 20)
        self.assertEqual(turns[0].user, "user 5")
        self.assertEqual(turns[-1].assistant, "assistant 24")

    def test_transcript_writer_saves_jsonl_under_date_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = TranscriptWriter(Path(tmp))
            writer.write_turn(
                user="hello",
                assistant="hello back",
                model="gpt-5.2",
                sources=[],
                has_sufficient_evidence=False,
            )

            records = [
                json.loads(line)
                for line in writer.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(writer.path.suffix, ".jsonl")
        self.assertEqual(records[0]["type"], "turn")
        self.assertEqual(records[0]["model"], "gpt-5.2")
        self.assertIn("hello back", records[0]["assistant"])

    def test_context_packet_is_compact_and_source_aware(self) -> None:
        result = RetrievalResult(
            source_filename="canon.md",
            location="section: Sarah",
            chunk_id="abc123",
            similarity_score=0.42,
            text="Sarah Nelson was a Mars commander. " * 80,
        )
        retrieval = RetrievalResponse(
            query="Who is Sarah?",
            has_sufficient_evidence=True,
            results=[result],
            message="Retrieved evidence for the query.",
        )

        packet = build_context_packet(retrieval)
        sources = format_sources([result])

        self.assertIn("cite source filenames inline", packet)
        self.assertIn("canon.md", packet)
        self.assertLess(len(packet), 2200)
        self.assertIn("similarity score: 0.4200", sources)

    def test_default_sarah_model_is_strong_reasoning_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / ".env"
                settings = load_settings(env_file=env_file)

        self.assertEqual(settings.sarah_model, "gpt-5.2")

    def test_placeholder_api_key_is_treated_as_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / ".env"
                env_file.write_text("OPENAI_API_KEY=your_openai_api_key_here\n", encoding="utf-8")
                settings = load_settings(env_file=env_file)

        self.assertEqual(settings.openai_api_key, "")


if __name__ == "__main__":
    unittest.main()
