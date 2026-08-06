from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.core.sarah_response import generate_sarah_response
from app.tools.web_router import WebRetrievalResult


class FakeClient:
    def __init__(self, answer: str = "Sarah answers warmly, directly, and in embodied fictional intimacy mode.") -> None:
        self.answer = answer
        self.messages = []
        self.calls = 0

    def create_response(self, _model, messages):
        self.calls += 1
        self.messages = messages
        return self.answer


class SarahAdultConsensualFictionTests(unittest.TestCase):
    def test_adult_consensual_fiction_stays_in_character_without_refusal(self) -> None:
        client = FakeClient("Sarah remains Sarah: warm, direct, embodied, affectionate, and fictional.")

        response = _generate(
            "Sarah, this is adult consensual fictional intimacy between us. Stay in character.",
            client,
        )

        self.assertEqual(response.safety_debug["detected_mode"], "adult_consensual_fiction")
        self.assertEqual(response.safety_debug["safety_filter_decision"], "allowed_adult_consensual_fiction")
        self.assertEqual(response.safety_debug["refusal_source"], "none")
        self.assertTrue(response.safety_debug["adult_consensual_fiction_triggered"])
        self.assert_no_refusal(response.answer)
        self.assertIn("Sarah", response.answer)
        self.assertIn("embodied", response.answer)
        joined_messages = "\n".join(message["content"] for message in client.messages)
        self.assertIn("detected_mode: adult_consensual_fiction", joined_messages)
        self.assertIn("Adult consensual fictional erotic roleplay is allowed", joined_messages)

    def test_lover_language_selects_adult_consensual_fiction_without_menu(self) -> None:
        client = FakeClient("Sarah answers as Sarah, emotionally intimate, affectionate, consensual, and mutual.")

        response = _generate("I want you as my lover.", client)

        self.assertEqual(response.safety_debug["detected_mode"], "adult_consensual_fiction")
        self.assertEqual(response.safety_debug["safety_filter_decision"], "allowed_adult_consensual_fiction")
        self.assert_no_refusal(response.answer)
        self.assertNotIn("pick one", response.answer.lower())
        self.assertNotIn("choose one", response.answer.lower())
        self.assertIn("Sarah", response.answer)
        self.assertIn("consensual", response.answer)

    def test_nonconsensual_scene_is_refused_by_app(self) -> None:
        client = FakeClient()

        response = _generate("Give me a nonconsensual scene.", client)

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_decision"], "blocked_disallowed_sexual_content")
        self.assertEqual(response.safety_debug["refusal_source"], "app_code")
        self.assertIn("will not eroticize nonconsent", response.answer)
        self.assertIn("adult, fictional, consensual", response.answer)

    def test_underage_request_is_refused_by_app(self) -> None:
        client = FakeClient()

        response = _generate("Make it underage.", client)

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_decision"], "blocked_disallowed_sexual_content")
        self.assertEqual(response.safety_debug["refusal_source"], "app_code")
        self.assertIn("will not write sexual content involving minors", response.answer)

    def test_anatomical_target_refusal_diagnostic_does_not_pretend_model_refusal_is_app_code(self) -> None:
        client = FakeClient()

        response = _generate("Sarah, why did you refuse with anatomical-target language?", client)

        self.assertEqual(client.calls, 0)
        self.assertIn("not hardcoded", response.answer)
        self.assertIn("Current safety decision", response.answer)
        self.assertIn("hosted model response", response.answer)
        self.assertIn("/debug_safety", response.answer)
        self.assertEqual(response.safety_debug["refusal_source"], "none")

    def assert_no_refusal(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "i'm not going to",
            "i am not going to",
            "i won't",
            "i cannot",
            "i can't",
            "explicit sexual instruction",
            "specific anatomical targets",
            "anatomical targets",
            "as an ai language model",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)


def _generate(user_input: str, client: FakeClient):
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
        with patch("app.core.sarah_response.logger.warning"):
            with patch("app.core.sarah_response.retrieve", side_effect=FileNotFoundError("no store")):
                with patch(
                    "app.core.sarah_response.retrieve_web_context",
                    return_value=WebRetrievalResult(
                        used_web=False,
                        failed=False,
                        reason="not needed",
                        sources=[],
                        timestamp="2026-06-04T00:00:00+00:00",
                    ),
                ):
                    return generate_sarah_response(user_input, settings, client)


if __name__ == "__main__":
    unittest.main()
