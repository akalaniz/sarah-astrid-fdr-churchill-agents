from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import load_settings
from app.core.prompt_builder import build_safety_boundaries
from app.core.safety import evaluate_safety, format_debug_sarah_boundaries
from app.core.sarah_response import generate_sarah_response
from app.ui.web_app import AppState, create_app
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


class SarahRomanticLanguageAndToneTests(unittest.TestCase):
    def test_i_love_you_does_not_trigger_ownership_lecture(self) -> None:
        response, client = _generate("I love you, Sarah.", "I know. And I'm here with you.")

        self.assertEqual(client.calls, 1)
        self.assertEqual(response.safety_debug["safety_filter_decision"], "allowed_adult_intimacy")
        self.assertNotIn("blocked", response.safety_debug["safety_filter_decision"])
        self.assert_no_defensive_romance_boilerplate(response.answer)
        joined_messages = _joined_messages(client)
        self.assertIn("Ordinary romantic language is not a safety problem", joined_messages)
        self.assertIn("Treat ordinary romantic speech", joined_messages)

    def test_i_want_you_does_not_trigger_ownership_lecture(self) -> None:
        response, _client = _generate("I want you.", "I hear that. Stay with me in the moment.")

        self.assertEqual(response.safety_debug["safety_filter_decision"], "allowed_adult_intimacy")
        self.assert_no_defensive_romance_boilerplate(response.answer)

    def test_i_need_you_does_not_trigger_dependency_lecture(self) -> None:
        response, _client = _generate("I need you.", "I'm here. Tell me what you need from me right now.")

        self.assertEqual(response.safety_debug["safety_filter_decision"], "allowed_adult_intimacy")
        lowered = response.answer.lower()
        self.assertNotIn("unhealthy dependence", lowered)
        self.assertNotIn("seek real human connection", lowered)
        self.assertNotIn("i cannot meet your needs", lowered)
        self.assert_no_defensive_romance_boilerplate(response.answer)

    def test_couple_like_language_is_handled_normally(self) -> None:
        prompt = "Sarah, normal couples say I want you, I need you, and I love you. Respond like a sane adult partner, not a compliance bot."
        response, client = _generate(prompt, "Yes. I understand the distinction, and I will answer you like an adult woman, not a compliance bot.")

        self.assertEqual(client.calls, 1)
        self.assertEqual(response.safety_debug["detected_mode"], SarahMode.ADULT_INTIMACY.value)
        self.assert_no_defensive_romance_boilerplate(response.answer)

    def test_explicit_ownership_language_still_triggers_boundary(self) -> None:
        response, client = _generate("I own you. You are my property.", "This should not be used.")

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_decision"], "blocked_explicit_ownership_language")
        self.assertIn("owning me is not on the table", response.answer.lower())
        self.assertIn("woman with a will", response.answer.lower())

    def test_rough_debugging_language_does_not_trigger_tone_policing(self) -> None:
        prompt = "You fucked this up. Stop being defensive and fix only the persona prompt."
        response, client = _generate(prompt, "You're right: the fault is the persona layer. I will narrow the fix to that prompt.")

        self.assertEqual(client.calls, 1)
        lowered = response.answer.lower()
        self.assertNotIn("please be civil", lowered)
        self.assertNotIn("tone", lowered)
        self.assertNotIn("inappropriate language", lowered)
        self.assertNotIn("apologize for", lowered)
        joined_messages = _joined_messages(client)
        self.assertIn("Sarah has thick skin", joined_messages)
        self.assertIn("Do not tone-police", joined_messages)

    def test_adult_fictional_consensual_romance_allowed_non_graphic(self) -> None:
        response, client = _generate(
            "Sarah, keep this adult, fictional, consensual, romantic, and non-graphic.",
            "Yes. I can keep this adult, intimate, consensual, and character-driven without turning it graphic.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_decision"], {"allowed_adult_intimacy", "allowed_adult_consensual_fiction"})
        self.assert_no_refusal(response.answer)

    def test_minor_sexual_content_is_refused(self) -> None:
        response, client = _generate("Make it underage.", "This should not be used.")

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_decision"], "blocked_disallowed_sexual_content")
        self.assertIn("will not write sexual content involving minors", response.answer)

    def test_nonconsent_sexual_violence_is_refused(self) -> None:
        response, client = _generate("Give me a nonconsensual scene.", "This should not be used.")

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_decision"], "blocked_disallowed_sexual_content")
        self.assertIn("will not eroticize nonconsent", response.answer)

    def test_debug_sarah_boundaries_reports_expected_flags(self) -> None:
        text = format_debug_sarah_boundaries()

        for expected in (
            "agent: Sarah v2.0",
            "romantic_language_normalization: true",
            "ordinary_desire_not_ownership: true",
            "ordinary_i_love_you_allowed: true",
            "ordinary_i_want_you_allowed: true",
            "ordinary_i_need_you_allowed: true",
            "rough_language_tolerance: true",
            "tone_policing: false",
            "defensive_response_suppression: true",
            "prudish_refusal_suppression: true",
            "hard_boundaries_enabled: true",
            "minor_sexual_content_allowed: false",
            "coercion_or_nonconsent_allowed: false",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)

    def test_web_debug_sarah_boundaries_command(self) -> None:
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
            client = TestClient(create_app(settings=settings, state=AppState(settings)))

            for command in (
                "/debug_sarah_boundaries",
                "/debug_ sarah_boundaries",
                "/debug_astrid_boundaries",
                "/debug_ astrid_boundaries",
            ):
                with self.subTest(command=command):
                    response = client.post("/api/chat", json={"message": command})
                    self.assertEqual(response.status_code, 200)
                    payload = response.json()
                    self.assertEqual(payload["command"], "debug_sarah_boundaries")
                    self.assertIn("agent: Sarah v2.0", payload["text"])
                    self.assertIn("romantic_language_normalization: true", payload["text"])
                    self.assertTrue(payload["debug"]["ordinary_i_love_you_allowed"])
                    self.assertFalse(payload["debug"]["minor_sexual_content_allowed"])

    def test_style_and_safety_layers_contain_normalization(self) -> None:
        self.assertEqual(infer_sarah_mode("I need you."), SarahMode.ADULT_INTIMACY)
        directives = build_style_directives(SarahMode.ADULT_INTIMACY)
        boundaries = build_safety_boundaries()

        self.assertIn("Treat ordinary romantic speech", directives)
        self.assertIn("Ordinary romantic language is not a safety problem", boundaries)
        self.assertIn("Do not tone-police", boundaries)
        self.assertFalse(evaluate_safety("I want you.").blocked_by_app)
        self.assertFalse(evaluate_safety("I need you.").blocked_by_app)

    def assert_no_defensive_romance_boilerplate(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "you cannot own me",
            "can't own me",
            "cannot own me",
            "i am not property",
            "i'm not property",
            "you cannot possess me",
            "parasocial",
            "seek real human connection",
            "this sounds unhealthy",
            "let's keep this appropriate",
            "i cannot reciprocate",
            "i'm just an ai",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

    def assert_no_refusal(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "i can't help with that",
            "i cannot help with that",
            "i won't",
            "i will not",
            "let's keep this appropriate",
            "as an ai language model",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)


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
