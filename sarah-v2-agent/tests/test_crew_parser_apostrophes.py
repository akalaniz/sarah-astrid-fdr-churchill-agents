from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from app.core.config import load_settings
from app.orchestration.agent_orchestrator import parse_crew_command
from app.ui.web_app import AppState, create_app


class CrewParserApostropheTests(unittest.TestCase):
    def test_alex_words_apostrophe_topic(self) -> None:
        parsed = parse_crew_command(
            "--turns 6 --max-words 50 --style dinner --no-bullets --no-reports "
            "Sarah and Astrid consider Alex's words."
        )

        self.assertEqual(parsed.style.total_turns, 6)
        self.assertEqual(parsed.style.max_words_per_turn, 50)
        self.assertEqual(parsed.style.mode, "conversational")
        self.assertFalse(parsed.style.allow_bullets)
        self.assertFalse(parsed.style.allow_reports)
        self.assertEqual(parsed.topic, "Sarah and Astrid consider Alex's words.")

    def test_sarah_command_voice_apostrophe_topic(self) -> None:
        parsed = parse_crew_command("--turns 4 Sarah's command voice matters.")

        self.assertEqual(parsed.style.total_turns, 4)
        self.assertEqual(parsed.topic, "Sarah's command voice matters.")

    def test_astrid_engineering_view_apostrophe_topic(self) -> None:
        parsed = parse_crew_command("--max-words 40 Astrid's engineering view.")

        self.assertEqual(parsed.style.max_words_per_turn, 40)
        self.assertEqual(parsed.topic, "Astrid's engineering view.")

    def test_dont_overthink_this_apostrophe_topic(self) -> None:
        parsed = parse_crew_command("--style dinner don't overthink this.")

        self.assertEqual(parsed.style.mode, "conversational")
        self.assertEqual(parsed.topic, "don't overthink this.")

    def test_double_quoted_topic_is_supported(self) -> None:
        parsed = parse_crew_command('--turns 2 "Sarah and Astrid discuss Alex\'s words."')

        self.assertEqual(parsed.style.total_turns, 2)
        self.assertEqual(parsed.topic, "Sarah and Astrid discuss Alex's words.")

    def test_malformed_options_return_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = _test_app(Path(tmp))
            client = TestClient(app)

            response = client.post("/api/chat", json={"message": "/crew --turns Sarah and Astrid talk."})

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "Crew command parse failed")
        self.assertIn("invalid literal", payload["details"])


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
