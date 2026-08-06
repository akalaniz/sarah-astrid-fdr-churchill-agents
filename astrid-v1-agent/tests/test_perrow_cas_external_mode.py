from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.core.perrow_cas import analyze_perrow_cas_event
from app.core.sarah_response import generate_sarah_response
from app.tools.web_router import WebRetrievalResult


TEST_PROMPT = "Sarah, analyze a Carrington-class solar flare hitting Earth using Perrow/CAS accident theory."


class FakeClient:
    def create_response(self, _model, messages):
        self.messages = messages
        return "ok"


class PerrowCasExternalModeTests(unittest.TestCase):
    def test_carrington_solar_flare_is_external_accident_analysis(self) -> None:
        frame = analyze_perrow_cas_event(TEST_PROMPT)

        self.assertIn("solar grid failure", frame["event_summary"])
        self.assertEqual(frame["perrow_quadrant"], "complex_tight")
        self.assertIn("power grid", frame["system_boundary"].lower())
        self.assertIn("gps", " ".join(frame["key_components"]).lower())
        self.assertIn("patch", " ".join(frame["patch_recommendations"]).lower())

    def test_carrington_prompt_injects_expected_external_context(self) -> None:
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

        self.assertIn("detected_mode: perrow_cas_accident_analysis", lowered)
        self.assertIn("external system-failure", lowered)
        self.assertIn("tight coupling", lowered)
        self.assertIn("complex interactions", lowered)
        self.assertIn("perrow_depth: brief", lowered)
        self.assertIn("cascade timeline, max 5 steps", lowered)
        self.assertIn("t0", lowered)
        self.assertIn("t1", lowered)

        for term in (
            "satellites",
            "gps timing",
            "power grid",
            "communications",
            "aviation",
            "finance",
            "emergency services",
        ):
            with self.subTest(term=term):
                self.assertIn(term, lowered)

        self.assertIn("hidden couplings", lowered)
        self.assertIn("patch recommendations", lowered)
        self.assertIn("stabilize systems", lowered)
        self.assertNotIn("sarah self-diagnostics", lowered)
        self.assertNotIn("sarah vulnerability graph", lowered)


if __name__ == "__main__":
    unittest.main()
