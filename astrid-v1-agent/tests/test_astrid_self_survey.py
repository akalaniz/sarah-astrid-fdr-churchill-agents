from __future__ import annotations

from dataclasses import dataclass, field
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core import chat_loop
from app.core.config import load_settings
from app.core.self_survey import SELF_SURVEY_HELP, build_self_survey_prompt
from app.ui.web_app import AppState, create_app


@dataclass
class FakeReply:
    answer: str = "Astrid self-survey result. Use /remember if you want this retained."
    sources: list[dict[str, str]] = field(default_factory=lambda: [{"source_filename": "Astrid 2_0_a.docx"}])
    web_sources: list[dict[str, str]] = field(default_factory=list)
    web_status: dict[str, object] = field(default_factory=lambda: {"used_web": False, "failed": False, "reason": "test"})
    has_sufficient_evidence: bool = True
    prompt_debug_summary: dict[str, object] = field(default_factory=lambda: {"layers": []})
    safety_debug: dict[str, object] = field(default_factory=dict)
    retrieval_results: list[object] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.answer


class AstridSelfSurveyTests(unittest.TestCase):
    def test_help_command_does_not_call_engine_or_write_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))
            client = TestClient(create_app(settings=settings, state=AppState(settings)))

            with patch("app.core.sarah_engine.generate_sarah_reply") as generate_reply:
                response = client.post("/api/chat", json={"message": "/self_survey help"})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["command"], "self_survey_help")
            self.assertIn("/self_survey literature", payload["text"])
            self.assertIn("Results are not saved automatically", payload["text"])
            generate_reply.assert_not_called()
            self.assertEqual(settings.memory_file.read_text(encoding="utf-8"), "")

    def test_literature_command_uses_shared_engine_and_astrid_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))
            client = TestClient(create_app(settings=settings, state=AppState(settings)))

            with patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeReply()) as generate_reply:
                response = client.post("/api/chat", json={"message": "/self_survey literature"})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["command"], "self_survey")
            self.assertEqual(payload["topic"], "literature")
            self.assertFalse(payload["persisted_automatically"])
            self.assertEqual(Path(payload["memory_file"]), settings.memory_file)
            prompt = generate_reply.call_args.args[0]
            self.assertIn("run a self-survey about: literature", prompt)
            self.assertIn("config/astrid_master_prompt.md", prompt)
            self.assertIn("data/memory/astrid_memory.jsonl", prompt)
            self.assertIn("Do not answer like generic ChatGPT and do not speak as Sarah", prompt)
            self.assertIn("STAGE 1: INTERNAL ANALYSIS", prompt)
            self.assertIn("STAGE 2: ASTRID VOICE RENDERING", prompt)
            self.assertIn("Do not reveal hidden internal analysis or chain-of-thought", prompt)
            self.assertIn("warm, funny, intelligent, playful, embodied", prompt)
            self.assertIn("conversational rather than report-like", prompt)
            self.assertIn("Based on available evidence", prompt)
            self.assertIn("For literature", prompt)
            self.assertEqual(generate_reply.call_args.kwargs["session_id"], "web")
            self.assertEqual(settings.memory_file.read_text(encoding="utf-8"), "")

    def test_music_command_uses_shared_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))
            client = TestClient(create_app(settings=settings, state=AppState(settings)))

            with patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeReply()) as generate_reply:
                response = client.post("/api/chat", json={"message": "/self_survey music"})

            self.assertEqual(response.status_code, 200)
            prompt = generate_reply.call_args.args[0]
            self.assertIn("run a self-survey about: music", prompt)
            self.assertIn("For music", prompt)
            self.assertIn("what you would dance to", prompt)
            self.assertIn("what might make you cry", prompt)
            self.assertEqual(settings.memory_file.read_text(encoding="utf-8"), "")

    def test_travel_and_philosophy_topics_receive_topic_specific_voice_guidance(self) -> None:
        travel_prompt = build_self_survey_prompt("travel")
        philosophy_prompt = build_self_survey_prompt("philosophy")

        self.assertIn("run a self-survey about: travel", travel_prompt)
        self.assertIn("For travel", travel_prompt)
        self.assertIn("landscapes, weather, cities, food, engineering", travel_prompt)
        self.assertIn("run a self-survey about: philosophy", philosophy_prompt)
        self.assertIn("For philosophy", philosophy_prompt)
        self.assertIn("embodied futurism, sterile abstraction", philosophy_prompt)

    def test_cli_self_survey_literature_uses_shared_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))

            with patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeReply()) as generate_reply:
                with patch("builtins.input", side_effect=["/self_survey literature", "/quit"]):
                    with patch("sys.stdout", new=io.StringIO()):
                        chat_loop.run_chat_loop(settings)

            prompt = generate_reply.call_args.args[0]
            self.assertIn("run a self-survey about: literature", prompt)
            self.assertEqual(generate_reply.call_args.kwargs["session_id"], "cli")
            self.assertEqual(settings.memory_file.read_text(encoding="utf-8"), "")

    def test_prompt_documents_non_persistence(self) -> None:
        prompt = build_self_survey_prompt("music")
        self.assertIn("do not write this self-survey into persistent memory automatically", prompt)
        self.assertIn("/remember <text>", prompt)
        self.assertIn("Astrid speaking to Alex", prompt)
        self.assertIn("not Sarah", prompt)
        self.assertIn("Keep reasoning quality, source grounding, and uncertainty handling", prompt)


def _test_settings(tmp: Path):
    settings = load_settings(env_file=tmp / ".env")
    memory_file = tmp / "data" / "memory" / "astrid_memory.jsonl"
    return settings.__class__(
        **{
            **settings.__dict__,
            "memory_file": memory_file,
            "conversations_dir": tmp / "data" / "conversations",
            "web_cache_dir": tmp / "data" / "web_cache",
            "vector_store_dir": tmp / "data" / "vector_store",
        }
    )


if __name__ == "__main__":
    unittest.main()
