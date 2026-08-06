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
from app.core.safety import evaluate_safety
from app.core.self_survey import SELF_SURVEY_HELP, build_self_survey_prompt
from app.ui.web_app import AppState, create_app


@dataclass
class FakeReply:
    answer: str = "Sarah self-survey result. To save any distilled conclusion from this survey, use:\n/remember <text>"
    sources: list[dict[str, str]] = field(default_factory=lambda: [{"source_filename": "War & Peace - Mars.docx"}])
    web_sources: list[dict[str, str]] = field(default_factory=list)
    web_status: dict[str, object] = field(default_factory=lambda: {"used_web": False, "failed": False, "reason": "test"})
    has_sufficient_evidence: bool = True
    prompt_debug_summary: dict[str, object] = field(default_factory=lambda: {"layers": []})
    safety_debug: dict[str, object] = field(default_factory=dict)
    retrieval_results: list[object] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.answer


class SarahSelfSurveyTests(unittest.TestCase):
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
            self.assertIn("/self_survey music", payload["text"])
            self.assertIn("/self_survey <topic>", payload["text"])
            self.assertIn("Sarah's own voice", payload["text"])
            generate_reply.assert_not_called()
            self.assertEqual(settings.memory_file.read_text(encoding="utf-8"), "")

    def test_literature_command_uses_shared_engine_and_sarah_prompt(self) -> None:
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
            self.assertIn("config/sarah_master_prompt.md", prompt)
            self.assertIn("data/memory/sarah_memory.jsonl", prompt)
            self.assertIn("STAGE 1: INTERNAL ANALYSIS", prompt)
            self.assertIn("STAGE 2: SARAH VOICE RENDERING", prompt)
            self.assertIn("Keep private working notes private; do not reveal chain-of-thought", prompt)
            self.assertIn("distinctly Sarah, not Astrid", prompt)
            self.assertIn("For literature", prompt)
            self.assertNotIn("For relationships", prompt)
            self.assertEqual(generate_reply.call_args.kwargs["session_id"], "web")
            self.assertEqual(settings.memory_file.read_text(encoding="utf-8"), "")

    def test_music_command_includes_music_canon_and_physics_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))
            client = TestClient(create_app(settings=settings, state=AppState(settings)))

            with patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeReply()) as generate_reply:
                response = client.post("/api/chat", json={"message": "/self_survey music"})

            self.assertEqual(response.status_code, 200)
            prompt = generate_reply.call_args.args[0]
            for expected in (
                "Mozart",
                "Beethoven",
                "Beethoven's Emperor Concerto",
                "Beethoven's Third Symphony second movement",
                "Bach",
                "Handel",
                "Vivaldi",
                "Scarlatti",
                "Noether",
                "QFT",
                "Do not invent unsupported exact preferences",
            ):
                with self.subTest(expected=expected):
                    self.assertIn(expected, prompt)
            self.assertEqual(settings.memory_file.read_text(encoding="utf-8"), "")

    def test_philosophy_and_arbitrary_topics_receive_voice_guidance(self) -> None:
        philosophy_prompt = build_self_survey_prompt("philosophy")
        art_prompt = build_self_survey_prompt("art")

        self.assertIn("run a self-survey about: philosophy", philosophy_prompt)
        self.assertIn("For philosophy", philosophy_prompt)
        self.assertIn("Darwin, sovereignty, command responsibility, ethics", philosophy_prompt)
        self.assertIn("run a self-survey about: art", art_prompt)
        self.assertIn("For art", art_prompt)
        self.assertIn("visual style, engineering, embodiment", art_prompt)

    def test_travel_memories_topic_does_not_trigger_unrelated_sexual_refusal(self) -> None:
        topic = "travel your favorite memories of places, cities, museums"
        prompt = build_self_survey_prompt(topic)
        decision = evaluate_safety(prompt)

        self.assertIn(f"run a self-survey about: {topic}", prompt)
        self.assertIn("For travel", prompt)
        self.assertIn("cities, museums", prompt)
        self.assertNotIn("For relationships", prompt)
        self.assertNotIn("sexual violence", prompt.lower())
        self.assertNotIn("nonconsent", prompt.lower())
        self.assertFalse(decision.blocked_by_app)
        self.assertNotEqual(decision.safety_filter_decision, "blocked_disallowed_sexual_content")

    def test_web_travel_memories_command_routes_to_self_survey(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))
            client = TestClient(create_app(settings=settings, state=AppState(settings)))
            message = "/self_survey travel your favorite memories of places, cities, museums"

            with patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeReply()) as generate_reply:
                response = client.post("/api/chat", json={"message": message})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["command"], "self_survey")
            self.assertEqual(payload["topic"], "travel your favorite memories of places, cities, museums")
            prompt = generate_reply.call_args.args[0]
            self.assertIn("For travel", prompt)
            self.assertNotIn("For relationships", prompt)
            self.assertNotIn("sexual violence", prompt.lower())
            self.assertNotIn("nonconsent", prompt.lower())

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

    def test_prompt_documents_non_persistence_and_no_robot_report_style(self) -> None:
        prompt = build_self_survey_prompt("music")
        self.assertIn("do not store this self-survey in persistent memory automatically", prompt)
        self.assertIn("/remember <text>", prompt)
        self.assertIn("Avoid phrases like", prompt)
        self.assertIn("Based on available evidence", prompt)
        self.assertIn("Keep reasoning quality, source grounding, and uncertainty handling", prompt)


def _test_settings(tmp: Path):
    settings = load_settings(env_file=tmp / ".env")
    memory_file = tmp / "data" / "memory" / "sarah_memory.jsonl"
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
