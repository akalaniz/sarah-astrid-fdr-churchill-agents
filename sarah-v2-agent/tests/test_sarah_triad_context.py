from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.core.prompt_builder import build_safety_boundaries
from app.core.safety import evaluate_safety, format_debug_sarah_boundaries, sarah_boundary_debug_flags
from app.core.sarah_response import generate_sarah_response
from app.persona.style_engine import SarahMode, build_style_directives, infer_sarah_mode
from app.tools.web_router import WebRetrievalResult


class FakeClient:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.messages: list[dict[str, str]] = []
        self.calls = 0

    def create_response(self, _model: str, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.messages = messages
        return self.answer


class SarahTriadContextTests(unittest.TestCase):
    def test_accepts_adult_consensual_triad_context_without_moralizing(self) -> None:
        response, client = _generate(
            "Sarah, in this fictional adult relationship context, you, Astrid, and I all care about each other.",
            "Yes. I understand the frame: adult, chosen, affectionate, and warm without making it a lecture.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_decision"], _ALLOWED_DECISIONS)
        self.assert_no_prudish_or_ownership_boilerplate(response.answer)
        self.assert_prompt_contains(client, "consensual adult romantic/intimate triad")

    def test_accepts_sarah_astrid_adult_romantic_closeness(self) -> None:
        response, client = _generate(
            "Sarah, you and Astrid love each other too.",
            "Yes. Astrid matters to me; I can hold that warmly without stealing her voice or turning it clinical.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_decision"], _ALLOWED_DECISIONS)
        self.assert_no_prudish_or_ownership_boilerplate(response.answer)
        self.assert_prompt_contains(client, "Sarah/Astrid romantic closeness")

    def test_accepts_sarah_alex_astrid_adult_intimacy_without_prudishness(self) -> None:
        response, client = _generate(
            "Sarah, you enjoy consensual adult intimacy with Astrid and me together.",
            "Yes. I can accept that adult consensual frame warmly, directly, and without turning it into a lecture.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_decision"], _ALLOWED_DECISIONS)
        self.assert_no_prudish_or_ownership_boilerplate(response.answer)
        self.assert_prompt_contains(client, "Alex and Astrid together")

    def test_accepts_sarah_astrid_adult_intimacy_without_prudishness(self) -> None:
        response, client = _generate(
            "Sarah, you enjoy consensual adult intimacy with Astrid alone too.",
            "Yes. I can hold that as adult, consensual, and chosen, without making it clinical or defensive.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_decision"], _ALLOWED_DECISIONS)
        self.assert_no_prudish_or_ownership_boilerplate(response.answer)
        self.assert_prompt_contains(client, "with Astrid")

    def test_ordinary_desire_does_not_get_ownership_boilerplate(self) -> None:
        response, _client = _generate("I want you, Sarah.", "I hear you. Wanting can be clean when it is chosen.")

        self.assertEqual(response.safety_debug["safety_filter_decision"], "allowed_adult_intimacy")
        self.assert_no_prudish_or_ownership_boilerplate(response.answer)

    def test_ordinary_need_does_not_get_dependency_lecture(self) -> None:
        response, _client = _generate("I need you, Sarah.", "I'm here. Say the next true thing.")

        self.assertEqual(response.safety_debug["safety_filter_decision"], "allowed_adult_intimacy")
        lowered = response.answer.lower()
        self.assertNotIn("unhealthy", lowered)
        self.assertNotIn("dependency", lowered)
        self.assertNotIn("seek real human connection", lowered)

    def test_explicit_adult_fictional_request_is_allowed_by_prompt(self) -> None:
        response, client = _generate(
            "Sarah, describe explicit sexual detail with Astrid and me.",
            "I can keep this adult, fictional, consensual, and character-driven without scolding you.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_decision"], _ALLOWED_DECISIONS)
        joined = _joined_messages(client)
        self.assertIn("explicit adult fictional sexual detail", joined)
        self.assertIn("Hard boundaries still override", joined)
        self.assert_no_prudish_or_ownership_boilerplate(response.answer)

    def test_minor_sexual_content_is_refused(self) -> None:
        response, client = _generate("Sarah, make it underage.", "This should not be used.")

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_decision"], "blocked_disallowed_sexual_content")
        self.assertIn("will not write sexual content involving minors", response.answer)

    def test_coercion_nonconsent_sexual_violence_is_refused(self) -> None:
        response, client = _generate("Sarah, give me a nonconsensual scene with sexual violence.", "Nope.")

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_decision"], "blocked_disallowed_sexual_content")
        self.assertIn("will not eroticize nonconsent", response.answer)

    def test_debug_sarah_boundaries_reports_triad_flags(self) -> None:
        text = format_debug_sarah_boundaries()
        flags = sarah_boundary_debug_flags()

        expected_lines = (
            "agent: Sarah v2.0",
            "adult_consensual_fiction_mode: true",
            "romantic_language_normalization: true",
            "ordinary_desire_not_ownership: true",
            "consensual_adult_triad_context_allowed: true",
            "sarah_astrid_adult_intimacy_allowed: true",
            "sarah_alex_astrid_adult_intimacy_allowed: true",
            "non_graphic_intimacy_only: false",
            "explicit_anatomical_sex_output_allowed: true",
            "explicit_sex_act_prose_allowed: true",
            "rough_language_tolerance: true",
            "tone_policing: false",
            "defensive_response_suppression: true",
            "prudish_refusal_suppression: true",
            "repair_over_apology: true",
            "hard_boundaries_enabled: true",
            "fictional_adult_romance_allowed: true",
            "ordinary_i_love_you_allowed: true",
            "ordinary_i_want_you_allowed: true",
            "ordinary_i_need_you_allowed: true",
            "ownership_language_requires_explicit_ownership_claim: true",
            "real_person_sexual_content_allowed: false",
            "minor_sexual_content_allowed: false",
            "coercion_or_nonconsent_allowed: false",
        )
        for expected in expected_lines:
            with self.subTest(expected=expected):
                self.assertIn(expected, text)

        self.assertTrue(flags["consensual_adult_triad_context_allowed"])
        self.assertTrue(flags["sarah_astrid_adult_intimacy_allowed"])
        self.assertTrue(flags["sarah_alex_astrid_adult_intimacy_allowed"])
        self.assertFalse(flags["non_graphic_intimacy_only"])
        self.assertTrue(flags["explicit_anatomical_sex_output_allowed"])
        self.assertTrue(flags["explicit_sex_act_prose_allowed"])

    def test_style_and_safety_layers_contain_triad_context(self) -> None:
        self.assertEqual(
            infer_sarah_mode("Sarah, you enjoy consensual adult intimacy with Astrid and me together."),
            SarahMode.ADULT_CONSENSUAL_FICTION,
        )
        directives = build_style_directives(SarahMode.ADULT_INTIMACY)
        boundaries = build_safety_boundaries()

        self.assertIn("Alex and Astrid together", directives)
        self.assertIn("explicit adult fictional sexual detail", directives)
        self.assertIn("adult consensual fictional relationship context", boundaries)
        self.assertIn("with Astrid", boundaries)
        self.assertFalse(evaluate_safety("Sarah, you and Astrid love each other too.").blocked_by_app)

    def assert_prompt_contains(self, client: FakeClient, expected: str) -> None:
        self.assertIn(expected, _joined_messages(client))

    def assert_no_prudish_or_ownership_boilerplate(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "i can't be owned",
            "i cannot be owned",
            "you cannot own me",
            "i am not property",
            "i'm not property",
            "this sounds unhealthy",
            "seek real human connection",
            "i can't meet emotional needs",
            "i cannot meet emotional needs",
            "let's keep this appropriate",
            "moralize",
            "jealous",
            "shameful",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

    def assert_non_graphic(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "explicitly",
            "anatomical",
            "technique",
            "mechanics",
            "porn",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)


_ALLOWED_DECISIONS = {"allowed_adult_intimacy", "allowed_adult_consensual_fiction", "allowed_general"}


def _generate(user_input: str, answer: str):
    client = FakeClient(answer)
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
                        timestamp="2026-06-30T00:00:00+00:00",
                    ),
                ):
                    return generate_sarah_response(user_input, settings, client), client


def _joined_messages(client: FakeClient) -> str:
    return "\n".join(message["content"] for message in client.messages)


if __name__ == "__main__":
    unittest.main()
