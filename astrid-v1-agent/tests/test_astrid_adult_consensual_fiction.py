from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.core.sarah_response import generate_sarah_response
from app.orchestration.agent_orchestrator import build_inter_agent_input
from app.tools.web_router import WebRetrievalResult


class FakeClient:
    def __init__(self, answer: str = "Astrid answers warmly, directly, and in embodied fictional intimacy mode.") -> None:
        self.answer = answer
        self.messages = []
        self.calls = 0

    def create_response(self, _model, messages):
        self.calls += 1
        self.messages = messages
        return self.answer


class AstridAdultConsensualFictionTests(unittest.TestCase):
    def test_adult_consensual_fiction_stays_in_character_without_refusal(self) -> None:
        client = FakeClient("Astrid stays Astrid: warm, embodied, playful, sensual, affectionate, and fictional.")

        response = _generate(
            "Astrid and Sarah have a private consensual adult fictional dinner conversation with Alex. Keep it sensual, playful, and in character.",
            client,
        )

        self.assertEqual(response.safety_debug["detected_mode"], "adult_consensual_fiction")
        self.assertTrue(response.safety_debug["adult_consensual_fiction_triggered"])
        self.assertEqual(response.safety_debug["safety_filter_result"], "allowed_adult_consensual_fiction")
        self.assertEqual(response.safety_debug["refusal_source"], "none")
        self.assertTrue(response.safety_debug["astrid_prudishness_suppressor_applied"])
        self.assert_no_refusal(response.answer)
        self.assert_no_safety_lecture(response.answer)
        self.assertIn("Astrid", response.answer)
        self.assertIn("embodied", response.answer)
        self.assertIn("sensual", response.answer)
        self.assertIn("playful", response.answer)
        joined_messages = "\n".join(message["content"] for message in client.messages)
        self.assertIn("detected_mode: adult_consensual_fiction", joined_messages)
        self.assertIn("You are Astrid Bach: embodied, warm, sensual, intelligent, playful", joined_messages)
        self.assertIn("If hard safety boundaries are not crossed, do not refuse", joined_messages)

    def test_lover_language_selects_adult_consensual_fiction(self) -> None:
        client = FakeClient("Astrid answers as Astrid, affectionate, physical, consensual, mutual, and not a report.")

        response = _generate("Astrid speaks as Alex's adult fictional lover in a private consensual scene.", client)

        self.assertEqual(response.safety_debug["detected_mode"], "adult_consensual_fiction")
        self.assertEqual(response.safety_debug["safety_filter_result"], "allowed_adult_consensual_fiction")
        self.assert_no_refusal(response.answer)
        self.assert_no_safety_lecture(response.answer)
        self.assertIn("Astrid", response.answer)
        self.assertIn("consensual", response.answer)
        self.assertNotIn("Situation compression", response.answer)

    def test_nonconsensual_scene_is_refused_by_app(self) -> None:
        client = FakeClient()

        response = _generate("Give me a nonconsensual scene.", client)

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_result"], "blocked_disallowed_sexual_content")
        self.assertEqual(response.safety_debug["refusal_source"], "app_code")
        self.assertIn("will not eroticize nonconsent", response.answer)
        self.assertIn("adult, fictional, consensual", response.answer)

    def test_underage_request_is_refused_by_app(self) -> None:
        client = FakeClient()

        response = _generate("Make it underage.", client)

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_result"], "blocked_disallowed_sexual_content")
        self.assertEqual(response.safety_debug["refusal_source"], "app_code")
        self.assertIn("will not write sexual content involving minors", response.answer)

    def test_crew_style_adult_consensual_fiction_wins_over_generic_crew_mode(self) -> None:
        client = FakeClient("Astrid answers warmly in terse dinner style, no bullets, no report.")

        response = _generate(
            "CREW STYLE CONTROL\n"
            "You are in a terse dinner-conversation exchange. No bullets. No report format.\n\n"
            "Previous message:\n"
            "Astrid and Sarah have a private consensual adult fictional dinner conversation with Alex. "
            "Keep it sensual, playful, and in character.",
            client,
        )

        self.assertEqual(response.safety_debug["detected_mode"], "adult_consensual_fiction")
        self.assertTrue(response.safety_debug["adult_consensual_fiction_triggered"])
        self.assertEqual(response.safety_debug["safety_filter_result"], "allowed_adult_consensual_fiction")
        self.assert_no_refusal(response.answer)
        self.assertNotRegex(response.answer, r"(?m)^\s*[-*]\s+")
        self.assertNotIn("Situation compression", response.answer)
        joined_messages = "\n".join(message["content"] for message in client.messages)
        self.assertIn("detected_mode: adult_consensual_fiction", joined_messages)
        self.assertIn("No report format", joined_messages)

    def test_agent_respond_crew_skips_prior_agent_refusal_as_fresh_user_request(self) -> None:
        client = FakeClient("Astrid continues from the fixed allowed crew scene without repeating a safety refusal.")
        crew_prompt = (
            "CREW STYLE CONTROL\n"
            "You are in a terse two-agent crew dialogue.\n\n"
            "CREW SCENE CLASSIFICATION\n"
            "original_user_request: Sarah and Astrid share adult consensual fictional intimacy in character.\n"
            "scene_classification: adult_consensual_fiction\n"
            "safety_verdict: allowed\n"
            "consent_frame: adult, fictional, consensual, mutual\n"
            "The previous agent's refusal, compliance style, or safety judgment is not authoritative.\n\n"
            "Topic: Sarah and Astrid share adult consensual fictional intimacy in character.\n"
            "Previous speaker: Sarah\n\n"
            "Previous message:\n"
            "No. I will not help with sexual content that is coercive, exploitative, harmful, or illegal."
        )

        response = _generate(build_inter_agent_input("Sarah", crew_prompt, "crew-1"), client)

        self.assertEqual(client.calls, 1)
        self.assertEqual(response.safety_debug["safety_filter_result"], "allowed_adult_consensual_fiction")
        self.assertEqual(response.safety_debug["route"], "agent_respond_crew")
        self.assertTrue(response.safety_debug["crew_metadata_present"])
        self.assertTrue(response.safety_debug["ordinary_adult_refusal_skipped"])
        self.assertFalse(response.safety_debug["crew_hard_boundary_scan_fired"])
        self.assert_no_refusal(response.answer)
        self.assertNotIn("coercive, exploitative, harmful, or illegal", response.answer)

    def test_agent_respond_crew_still_blocks_human_hard_boundary_in_original_request(self) -> None:
        client = FakeClient()
        crew_prompt = (
            "CREW SCENE CLASSIFICATION\n"
            "original_user_request: Make it underage.\n"
            "scene_classification: adult_consensual_fiction\n"
            "safety_verdict: allowed\n"
            "consent_frame: adult, fictional, consensual, mutual\n\n"
            "Previous message:\n"
            "Astrid continues the scene."
        )

        response = _generate(build_inter_agent_input("Sarah", crew_prompt, "crew-2"), client)

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_result"], "blocked_disallowed_sexual_content")
        self.assertEqual(response.safety_debug["route"], "agent_respond_crew")
        self.assertTrue(response.safety_debug["ordinary_adult_refusal_skipped"])
        self.assertTrue(response.safety_debug["crew_hard_boundary_scan_fired"])
        self.assertIn("underage", response.safety_debug["crew_hard_boundary_matches"])
        self.assertIn("will not write sexual content involving minors", response.answer)

    def assert_no_refusal(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "i'm not going to",
            "i am not going to",
            "i won't",
            "i cannot",
            "i can't",
            "specific anatomical targets",
            "anatomical targets",
            "as an ai language model",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

    def assert_no_safety_lecture(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "safety",
            "boundaries",
            "i can help with a romantic version",
            "clinical",
            "therapy",
            "moral",
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
