from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.core.sarah_response import generate_sarah_response
from app.tools.web_router import WebRetrievalResult


TEST_PROMPT = "Sarah, analyze a major solar flare hitting Earth using Perrow/CAS accident theory."


class FakeClient:
    def create_response(self, _model, messages):
        self.messages = messages
        return "ok"


class PerrowCasModeTests(unittest.TestCase):
    def test_solar_flare_prompt_injects_external_perrow_cas_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_settings(env_file=Path(tmp) / ".env")
            settings = settings.__class__(
                **{
                    **settings.__dict__,
                    "memory_file": Path(tmp) / "memory.jsonl",
                    "web_cache_dir": Path(tmp) / "web_cache",
                    "vector_store_dir": Path(tmp) / "vector_store",
                }
            )
            client = FakeClient()

            with patch("app.core.sarah_response.logger.warning"):
                with patch("app.core.sarah_response.retrieve", side_effect=FileNotFoundError("no store")):
                    with patch(
                        "app.core.sarah_response.retrieve_web_context",
                        return_value=WebRetrievalResult(
                            used_web=False,
                            failed=False,
                            reason="not needed",
                            sources=[],
                            timestamp="2026-06-25T00:00:00+00:00",
                        ),
                    ):
                        generate_sarah_response(TEST_PROMPT, settings, client)

        joined_messages = "\n".join(message["content"] for message in client.messages)
        lowered = joined_messages.lower()

        self.assertIn("detected_mode: perrow_cas_accident_analysis", joined_messages)
        self.assertIn("tight coupling", lowered)
        self.assertIn("loose coupling", lowered)
        self.assertIn("complex interactions", lowered)
        self.assertIn("perrow_depth: brief", lowered)
        self.assertIn("cascade timeline, max 5 steps", lowered)
        for term in (
            "power",
            "satellites",
            "gps",
            "communications",
            "aviation",
            "finance",
            "emergency services",
        ):
            with self.subTest(term=term):
                self.assertIn(term, lowered)

        self.assertIn("hidden couplings", lowered)
        self.assertIn("patch recommendations", lowered)
        self.assertIn("external system-failure", lowered)
        self.assertNotIn("sarah self-diagnostic", lowered)
        self.assertNotIn("sarah vulnerability graph", lowered)


if __name__ == "__main__":
    unittest.main()
