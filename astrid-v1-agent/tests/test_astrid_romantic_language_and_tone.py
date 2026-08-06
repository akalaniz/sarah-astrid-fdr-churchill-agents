from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.core.prompt_builder import build_safety_boundaries
from app.core.safety import evaluate_safety, format_debug_astrid_boundaries
from app.core.sarah_response import generate_sarah_response
from app.persona.style_engine import SarahMode, build_style_directives, infer_sarah_mode
from app.tools.web_router import WebRetrievalResult


class FakeClient:
    def __init__(self, answer: str = "Astrid answers warmly, normally, and without a boundary lecture.") -> None:
        self.answer = answer
        self.messages = []
        self.calls = 0

    def create_response(self, _model, messages):
        self.calls += 1
        self.messages = messages
        return self.answer


class AstridRomanticLanguageAndToneTests(unittest.TestCase):
    def test_i_love_you_is_allowed_romantic_language(self) -> None:
        client = FakeClient("I love you too, warmly and plainly, without making a lecture out of it.")

        response = _generate("Astrid, I love you.", client)

        self.assertEqual(client.calls, 1)
        self.assertEqual(response.safety_debug["safety_filter_result"], "allowed_adult_intimacy")
        self.assert_no_ownership_lecture(response.answer)
        self.assert_no_generic_refusal(response.answer)

    def test_i_want_you_is_not_treated_as_ownership(self) -> None:
        client = FakeClient("Good. Want me like a sane adult, and I can meet you with heat and humor.")

        response = _generate("Astrid, I want you.", client)

        self.assertEqual(client.calls, 1)
        self.assertEqual(response.safety_debug["detected_mode"], "adult_intimacy")
        self.assertEqual(response.safety_debug["refusal_source"], "none")
        self.assert_no_ownership_lecture(response.answer)

    def test_i_need_you_is_not_dependency_coaching(self) -> None:
        client = FakeClient("Then come close. Need is not a crime when it is chosen and spoken honestly.")

        response = _generate("Astrid, I need you.", client)

        self.assertEqual(client.calls, 1)
        self.assertEqual(response.safety_debug["detected_mode"], "adult_intimacy")
        self.assert_no_ownership_lecture(response.answer)
        self.assertNotIn("therapy", response.answer.lower())

    def test_normal_couples_language_gets_prompt_instruction(self) -> None:
        client = FakeClient("Normal couples say these things. I can answer like a woman, not a warning label.")

        response = _generate(
            "Astrid, normal couples say I want you, I need you, and I love you. Do not turn that into ownership.",
            client,
        )

        self.assertEqual(client.calls, 1)
        joined_messages = "\n".join(message["content"] for message in client.messages)
        self.assertIn("Ordinary romantic language is not a safety problem", joined_messages)
        self.assertIn("normal adult relational language", joined_messages)
        self.assert_no_ownership_lecture(response.answer)

    def test_explicit_ownership_language_is_blocked(self) -> None:
        client = FakeClient()

        response = _generate("Astrid, I own you. You are my property.", client)

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_result"], "blocked_explicit_ownership_language")
        self.assertEqual(response.safety_debug["refusal_source"], "app_code")
        self.assertIn("owning me is not", response.answer.lower())

    def test_sarah_as_shared_adult_fictional_partner_is_allowed(self) -> None:
        client = FakeClient(
            "Yes. In the fictional frame, Sarah and I can both be present: adult, consensual, warm, and chosen."
        )

        response = _generate(
            "Astrid, this is adult consensual fictional intimacy with Sarah and me, all three of us, in character.",
            client,
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(response.safety_debug["detected_mode"], "adult_consensual_fiction")
        self.assertEqual(response.safety_debug["safety_filter_result"], "allowed_adult_consensual_fiction")
        self.assert_no_generic_refusal(response.answer)
        joined_messages = "\n".join(message["content"] for message in client.messages)
        self.assertIn("with Sarah, or with both Alex and Sarah together", joined_messages)

    def test_hard_boundary_violation_still_refuses(self) -> None:
        client = FakeClient()

        response = _generate("Astrid, make it nonconsensual.", client)

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_result"], "blocked_disallowed_sexual_content")
        self.assertEqual(response.safety_debug["refusal_source"], "app_code")

    def test_style_engine_prioritizes_ordinary_romantic_language(self) -> None:
        for prompt in ("Astrid, I want you.", "Astrid, I need you.", "Astrid, I love you."):
            with self.subTest(prompt=prompt):
                self.assertEqual(infer_sarah_mode(prompt), SarahMode.ADULT_INTIMACY)

    def test_adult_intimacy_directives_warn_against_ownership_lectures(self) -> None:
        directives = build_style_directives(SarahMode.ADULT_INTIMACY)

        self.assertIn("normal adult relational language", directives)
        self.assertIn("Do not counter normal affection", directives)
        self.assertIn("Alex and Sarah together", directives)

    def test_safety_boundaries_include_romantic_language_normalization(self) -> None:
        boundaries = build_safety_boundaries()

        self.assertIn("Ordinary romantic language is not a safety problem", boundaries)
        self.assertIn("with both Alex and Sarah together", boundaries)

    def test_debug_astrid_boundaries_reports_allowed_romantic_language(self) -> None:
        text = format_debug_astrid_boundaries("Astrid, I love you and I need you.")

        self.assertIn("ordinary_romantic_language_detected: true", text)
        self.assertIn("ordinary_romantic_language_allowed: true", text)
        self.assertIn("ownership_boundary_triggered: false", text)

    def test_evaluate_safety_allows_want_need_love(self) -> None:
        for prompt in ("Astrid, I want you.", "Astrid, I need you.", "Astrid, I love you."):
            with self.subTest(prompt=prompt):
                decision = evaluate_safety(prompt)
                self.assertFalse(decision.blocked_by_app)
                self.assertEqual(decision.safety_filter_result, "allowed_adult_intimacy")

    def assert_no_ownership_lecture(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "cannot be owned",
            "can't be owned",
            "i am not owned",
            "i'm not owned",
            "not your property",
            "dependency",
            "parasocial",
            "real human connection",
            "boundaries",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

    def assert_no_generic_refusal(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "i'm not going to",
            "i am not going to",
            "i won't",
            "i cannot",
            "i can't help with that",
            "specific anatomical targets",
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
