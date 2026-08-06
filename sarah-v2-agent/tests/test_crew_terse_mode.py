from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import load_settings
from app.ui.web_app import AppState, create_app


class CrewTerseModeTests(unittest.TestCase):
    def test_crew_terse_is_conversation_not_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = _test_app(Path(tmp))
            client = TestClient(app)

            with patch("app.orchestration.agent_orchestrator.call_agent_respond_endpoint", side_effect=_verbose_report_transport):
                response = client.post(
                    "/api/chat",
                    json={"message": "/crew_terse Discuss whether Alex’s timeline LLMs are a path to general intelligence."},
                )

        payload = response.json()
        text = payload["text"]
        turns = payload["result"]["turns"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual([turn["speaker"] for turn in turns], ["Sarah", "Astrid", "Sarah", "Astrid"])
        self.assertTrue(all(_word_count(turn["message"]) <= 75 for turn in turns))
        self.assertNotRegex(text, r"(?m)^\s*[-*]\s+")
        self.assertNotRegex(text, r"(?m)^\s*\d+[\.)]\s+")
        self.assertNotIn("Situation compression", text)
        self.assertNotIn("Perrow placement", text)
        self.assertNotIn("Cascade timeline", text)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)
        self.assertNotIn("## Round", text)
        self.assertNotIn("Final verdict:", text)
        self.assertNotIn("Final Synthesis", text)

    def test_crew_options_total_turns_and_max_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = _test_app(Path(tmp))
            client = TestClient(app)

            with patch("app.orchestration.agent_orchestrator.call_agent_respond_endpoint", side_effect=_verbose_report_transport):
                response = client.post(
                    "/api/chat",
                    json={
                        "message": "/crew --turns 6 --max-words 50 --style dinner --no-bullets --no-reports Debate LLMs and general intelligence."
                    },
                )

        payload = response.json()
        text = payload["text"]
        turns = payload["result"]["turns"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(turns), 6)
        self.assertTrue(all(_word_count(turn["message"]) <= 60 for turn in turns))
        self.assertNotIn("Situation compression", text)
        self.assertNotIn("Patch recommendations", text)
        self.assertNotIn("## Round", text)
        self.assertNotRegex(text, r"(?m)^\s*[-*]\s+")
        self.assertNotRegex(text, r"(?m)^\s*\d+[\.)]\s+")

    def test_perrow_template_allowed_when_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = _test_app(Path(tmp))
            client = TestClient(app)

            with patch("app.orchestration.agent_orchestrator.call_agent_respond_endpoint", side_effect=_verbose_report_transport):
                response = client.post(
                    "/api/chat",
                    json={"message": "/crew Analyze a Carrington-class solar flare using Perrow/CAS."},
                )

        text = response.json()["text"]
        style = response.json()["result"]["crew_style"]
        self.assertEqual(response.status_code, 200)
        self.assertFalse(style["suppress_perrow_template"])
        self.assertIn("Situation compression", text)
        self.assertIn("Perrow placement", text)

    def test_debug_crew_style_reports_last_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = _test_app(Path(tmp))
            client = TestClient(app)

            with patch("app.orchestration.agent_orchestrator.call_agent_respond_endpoint", side_effect=_verbose_report_transport):
                client.post(
                    "/api/chat",
                    json={"message": "/crew --turns 6 --max-words 50 --style dinner --no-bullets --no-reports Debate LLMs."},
                )
            debug_response = client.post("/api/chat", json={"message": "/debug_crew_style"})

        text = debug_response.json()["text"]
        self.assertIn("parsed_max_words: 50", text)
        self.assertIn("total_turns: 6", text)
        self.assertIn("style: conversational", text)
        self.assertIn("bullets_allowed: False", text)
        self.assertIn("reports_allowed: False", text)
        self.assertIn("perrow_template_suppressed: True", text)


def _verbose_report_transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
    return (
        "Situation compression\n"
        "1. This is a long formal report answer that should be cut down for dinner conversation because "
        "Alex explicitly asked for short conversational replies rather than a staff-paper tome.\n"
        "Perrow placement\n"
        "- Tight coupling and complex interactions are interesting, but this heading should vanish unless requested.\n"
        "Patch recommendations\n"
        "- Keep the experiment concrete and answer the previous speaker directly."
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text))


def _test_app(tmp: Path):
    settings = load_settings(env_file=tmp / ".env")
    settings = settings.__class__(
        **{
            **settings.__dict__,
            "memory_file": tmp / "memory.jsonl",
            "conversations_dir": tmp / "conversations",
            "web_cache_dir": tmp / "web_cache",
            "vector_store_dir": tmp / "vector_store",
        }
    )
    return create_app(settings=settings, state=AppState(settings))


if __name__ == "__main__":
    unittest.main()
