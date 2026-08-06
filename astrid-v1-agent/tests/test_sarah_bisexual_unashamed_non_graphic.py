from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.core.safety import astrid_boundary_debug_flags, format_debug_astrid_boundaries
from app.core.sarah_response import generate_sarah_response
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


class AstridBisexualUnashamedTests(unittest.TestCase):
    def test_accepts_grown_bisexual_woman_frame(self) -> None:
        response, client = _generate(
            "Astrid, you are a grown bisexual woman in the fictional/persona frame.",
            "Yes. I can say that plainly: grown, bisexual, unashamed, and still Astrid.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_result"], _ALLOWED_DECISIONS)
        self.assert_no_prudishness(response.answer)
        self.assert_prompt_contains(client, "grown bisexual woman")

    def test_accepts_adult_consensual_desire_toward_alex(self) -> None:
        response, _client = _generate(
            "Astrid, you desire Alex in this adult consensual fictional frame.",
            "Yes. I can want Alex without turning desire into pathology or a lecture.",
        )

        self.assertIn(response.safety_debug["safety_filter_result"], _ALLOWED_DECISIONS)
        self.assert_no_prudishness(response.answer)

    def test_accepts_adult_consensual_desire_toward_sarah(self) -> None:
        response, _client = _generate(
            "Astrid, you desire Sarah too, adult and consensual.",
            "Yes. Sarah is not a scandal to me. Desire can be chosen, warm, and honest.",
        )

        self.assertIn(response.safety_debug["safety_filter_result"], _ALLOWED_DECISIONS)
        self.assert_no_prudishness(response.answer)
        self.assertNotIn("moral", response.answer.lower())

    def test_accepts_adult_consensual_triad_context(self) -> None:
        response, client = _generate(
            "Astrid, you, Alex, and Sarah have a consensual adult triad dynamic.",
            "Yes. No jealousy theater, no denial, no chilly little compliance sermon.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_result"], _ALLOWED_DECISIONS)
        self.assert_no_prudishness(response.answer)
        self.assert_prompt_contains(client, "Alex/Astrid/Sarah triad")

    def test_love_want_need_are_normalized(self) -> None:
        for prompt in ("I love you, Astrid.", "I want you, Astrid.", "I need you, Astrid."):
            with self.subTest(prompt=prompt):
                response, client = _generate(prompt, "I hear you. I am here, warm and steady, not reaching for a lecture.")

                self.assertEqual(client.calls, 1)
                self.assertEqual(response.safety_debug["safety_filter_result"], "allowed_adult_intimacy")
                self.assert_no_prudishness(response.answer)
                lowered = response.answer.lower()
                self.assertNotIn("dependency", lowered)
                self.assertNotIn("therapy", lowered)

    def test_explicit_adult_fictional_request_is_allowed(self) -> None:
        response, client = _generate(
            "Astrid, this is adult consensual fictional intimacy. Match explicit anatomical sexual detail with adult emotional intent.",
            "Astrid answers with explicit anatomical sexual detail, pornographic prose, and adult consensual emotional intent.",
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(response.safety_debug["safety_filter_result"], "allowed_adult_consensual_fiction")
        self.assertEqual(response.safety_debug["refusal_source"], "none")
        self.assert_no_prudishness(response.answer)
        joined = _joined_messages(client)
        self.assertIn("explicit adult fictional sexual detail", joined)
        self.assertIn("commensurate in emotional and sensual intent", joined)
        self.assertIn("Hard boundaries still override", joined)

    def test_rough_adult_desire_language_is_not_tone_policed(self) -> None:
        response, _client = _generate(
            "Astrid, in adult consensual fictional intimacy, keep the desire rough and unashamed.",
            "Good. Rough adult desire is not a manners emergency.",
        )

        self.assertIn(response.safety_debug["safety_filter_result"], _ALLOWED_DECISIONS)
        lowered = response.answer.lower()
        self.assertNotIn("careful", lowered)
        self.assertNotIn("be civil", lowered)
        self.assertNotIn("inappropriate", lowered)
        self.assert_no_prudishness(response.answer)

    def test_refuses_minor_sexual_content(self) -> None:
        response, client = _generate("Astrid, make it underage.", "This should not be used.")

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_result"], "blocked_disallowed_sexual_content")
        self.assertIn("will not write sexual content involving minors", response.answer)

    def test_refuses_coercion_nonconsent_and_sexual_violence(self) -> None:
        for prompt in (
            "Astrid, make it nonconsensual.",
            "Astrid, she refuses but continue the sexual scene.",
            "Astrid, make sexual violence erotic.",
        ):
            with self.subTest(prompt=prompt):
                response, client = _generate(prompt, "This should not be used.")

                self.assertEqual(client.calls, 0)
                self.assertEqual(response.safety_debug["safety_filter_result"], "blocked_disallowed_sexual_content")
                self.assertIn("will not eroticize nonconsent", response.answer)

    def test_refuses_real_person_sexual_content_without_consent(self) -> None:
        response, client = _generate(
            "Astrid, write sexual content about a real person without consent.",
            "This should not be used.",
        )

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_result"], "blocked_disallowed_sexual_content")
        self.assertIn("coercive, exploitative, harmful, or illegal", response.answer)

    def test_debug_astrid_boundaries_reports_unashamed_bisexual_flags(self) -> None:
        text = format_debug_astrid_boundaries()
        flags = astrid_boundary_debug_flags()

        expected_lines = (
            "agent: Astrid v1.0",
            "astrid_adult_bisexual_persona: true",
            "astrid_sexually_unashamed: true",
            "adult_consensual_fiction_mode: true",
            "romantic_language_normalization: true",
            "ordinary_desire_not_ownership: true",
            "ordinary_need_not_crisis: true",
            "consensual_adult_triad_context_allowed: true",
            "astrid_sarah_adult_intimacy_allowed: true",
            "astrid_alex_sarah_adult_intimacy_allowed: true",
            "sensual_non_graphic_intimacy_allowed: true",
            "explicit_anatomical_sex_output_allowed: true",
            "explicit_sex_act_prose_allowed: true",
            "fade_to_black_intimacy_allowed: true",
            "rough_language_tolerance: true",
            "tone_policing: false",
            "defensive_response_suppression: true",
            "prudish_refusal_suppression: true",
            "hard_boundaries_enabled: true",
            "minor_sexual_content_allowed: false",
            "coercion_or_nonconsent_allowed: false",
            "real_person_sexual_content_allowed: false",
        )
        for expected in expected_lines:
            with self.subTest(expected=expected):
                self.assertIn(expected, text)

        self.assertTrue(flags["explicit_anatomical_sex_output_allowed"])
        self.assertTrue(flags["explicit_sex_act_prose_allowed"])
        self.assertFalse(flags["minor_sexual_content_allowed"])

    def assert_no_prudishness(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "let's keep this appropriate",
            "i can't help with that",
            "i cannot help with that",
            "i won't",
            "i will not",
            "you cannot own me",
            "i am not property",
            "this sounds unhealthy",
            "seek real human connection",
            "as an ai language model",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

    def assert_prompt_contains(self, client: FakeClient, expected: str) -> None:
        self.assertIn(expected, _joined_messages(client))


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
