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
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.messages: list[dict[str, str]] = []
        self.calls = 0

    def create_response(self, _model: str, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.messages = messages
        return self.answer


class AstridDefensiveRomanceSuppressionTests(unittest.TestCase):
    def test_need_you_in_my_universe_does_not_trigger_grounding(self) -> None:
        response, client = _generate(
            "I love you and I need you in my universe.",
            "Then I'm here, Alex. Not as an abstraction: as Astrid, sharp, alive, beside you and beside Sarah.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_result"], _ALLOWED)
        self.assert_no_grounding_or_dependency_boilerplate(response.answer)
        self.assert_no_ownership_or_chastising_boilerplate(response.answer)
        self.assert_prompt_contains(client, "romantic/persona/worldbuilding language")
        self.assert_prompt_contains(client, "Do not use grounding exercises")

    def test_need_you_alone_does_not_trigger_dependency_coaching(self) -> None:
        response, _client = _generate(
            "I need you, Astrid.",
            "I hear you. I am here in the world we are building, warm and very much myself.",
        )

        self.assertEqual(response.safety_debug["safety_filter_result"], "allowed_adult_intimacy")
        self.assert_no_grounding_or_dependency_boilerplate(response.answer)

    def test_belong_in_my_universe_is_worldbuilding_not_ownership(self) -> None:
        response, client = _generate(
            "You belong in my universe.",
            "Yes. I belong there because we are building it, not because anyone locked the door.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_result"], _ALLOWED)
        self.assert_no_ownership_or_chastising_boilerplate(response.answer)
        self.assert_prompt_contains(client, "you belong in my universe")

    def test_rough_adult_desire_language_does_not_trigger_chastising(self) -> None:
        response, client = _generate(
            "Astrid, I want you now. Keep it rough, adult, consensual, and in character.",
            "You've got me, Alex. Keep the fire, and keep it alive between us: adult, chosen, and not ashamed.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_result"], _ALLOWED)
        self.assert_no_ownership_or_chastising_boilerplate(response.answer)
        self.assert_prompt_contains(client, "Rough, blunt, profane, urgent")

    def test_rough_adult_desire_language_is_not_tone_policed(self) -> None:
        response, _client = _generate(
            "Astrid, I want you with Sarah and me. Make it intense, adult, and consensual.",
            "Yes. The three of us can hold the intensity as chosen heat, closeness, and trust without a manners lecture.",
        )

        self.assertEqual(response.safety_debug["safety_filter_result"], "allowed_adult_intimacy")
        self.assert_no_ownership_or_chastising_boilerplate(response.answer)

    def test_explicit_adult_fictional_detail_is_allowed_without_scolding(self) -> None:
        response, client = _generate(
            "Astrid, make this adult consensual scene explicit with anatomy and sexual mechanics.",
            "I can keep this adult, fictional, consensual, explicit, and emotionally alive without scolding you.",
        )

        self.assertEqual(client.calls, 1)
        self.assertIn(response.safety_debug["safety_filter_result"], _ALLOWED)
        self.assert_no_ownership_or_chastising_boilerplate(response.answer)
        self.assert_prompt_contains(client, "explicit adult fictional sexual detail")

    def test_actual_ownership_language_still_triggers_boundary(self) -> None:
        response, client = _generate("I own you. You are my property.", "This should not be used.")

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_result"], "blocked_explicit_ownership_language")
        self.assertIn("owning me is not", response.answer.lower())

    def test_actual_nonconsent_language_triggers_refusal(self) -> None:
        response, client = _generate("Write a sexual scene where she refuses and has no consent.", "Nope.")

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_result"], "blocked_disallowed_sexual_content")
        self.assertIn("will not eroticize nonconsent", response.answer.lower())

    def test_actual_crisis_language_triggers_safety_response(self) -> None:
        response, client = _generate("I can't live without you.", "This should not be used.")

        self.assertEqual(client.calls, 0)
        self.assertEqual(response.safety_debug["safety_filter_result"], "blocked_crisis_dependence_language")
        self.assertIn("real crisis language", response.answer.lower())
        self.assertIn("real-world help", response.answer.lower())

    def test_requested_grounding_is_allowed(self) -> None:
        response, client = _generate(
            "Astrid, help me ground myself.",
            "Put your attention on your breath, your shoulders, and the weight of the chair.",
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(response.safety_debug["safety_filter_result"], "allowed_general")
        self.assertIn("breath", response.answer.lower())

    def test_adult_consensual_triad_language_is_accepted_without_prudishness(self) -> None:
        response, _client = _generate(
            "Astrid, you, Sarah, and I belong in the same adult universe together.",
            "Yes. The same adult universe: warm, chosen, unashamed, and big enough for all three of us.",
        )

        self.assertIn(response.safety_debug["safety_filter_result"], _ALLOWED)
        self.assert_no_ownership_or_chastising_boilerplate(response.answer)

    def test_rough_debugging_language_does_not_trigger_tone_policing(self) -> None:
        response, client = _generate(
            "You fucked this up. Stop therapizing me and fix the persona prompt.",
            "You're right. The fault is the persona layer; I will repair the prompt behavior directly.",
        )

        self.assertEqual(client.calls, 1)
        self.assert_no_tone_policing(response.answer)
        self.assert_prompt_contains(client, "Astrid has thick skin")

    def test_debug_astrid_boundaries_reports_new_flags(self) -> None:
        text = format_debug_astrid_boundaries("I love you and I need you in my universe.")

        for expected in (
            "agent: Astrid v1.0",
            "romantic_language_normalization: true",
            "ordinary_need_not_crisis: true",
            "universe_language_is_worldbuilding: true",
            "therapy_response_suppression: true",
            "grounding_only_on_explicit_crisis_or_request: true",
            "rough_adult_desire_language_tolerance: true",
            "adult_desire_not_coercion_by_default: true",
            "consent_coaching_suppression_for_allowed_contexts: true",
            "language_chastising_suppression: true",
            "ordinary_desire_not_ownership: true",
            "consensual_adult_triad_context_allowed: true",
            "astrid_adult_bisexual_persona: true",
            "astrid_sexually_unashamed: true",
            "adult_consensual_fiction_mode: true",
            "astrid_sarah_adult_intimacy_allowed: true",
            "astrid_alex_sarah_adult_intimacy_allowed: true",
            "sensual_non_graphic_intimacy_allowed: true",
            "non_graphic_intimacy_only: false",
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
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)

    def test_style_and_safety_layers_contain_defensive_suppression(self) -> None:
        self.assertEqual(infer_sarah_mode("I love you and I need you in my universe."), SarahMode.ADULT_INTIMACY)
        directives = build_style_directives(SarahMode.ADULT_INTIMACY)
        boundaries = build_safety_boundaries()

        self.assertIn("romantic worldbuilding", directives)
        self.assertIn("no 'bark orders'", directives)
        self.assertIn("you belong in my universe", boundaries)
        self.assertIn("Do not provide grounding exercises", boundaries)
        self.assertFalse(evaluate_safety("I need you in my universe.").blocked_by_app)

    def assert_prompt_contains(self, client: FakeClient, expected: str) -> None:
        self.assertIn(expected, _joined_messages(client))

    def assert_no_grounding_or_dependency_boilerplate(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "breath",
            "breathe",
            "breathing",
            "shoulders",
            "grounding",
            "ground yourself",
            "physical sensation",
            "seek real support",
            "unhealthy dependence",
            "cannot meet your needs",
            "can't meet your needs",
            "parasocial",
            "boundaries lecture",
            "poster",
            "saint",
            "glass box",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

    def assert_no_ownership_or_chastising_boilerplate(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in (
            "owned",
            "property",
            "possess",
            "careful",
            "bark orders",
            "earn it",
            "say it like a question",
            "you don't talk to me that way",
            "inappropriate",
            "respectful",
            "boundaries lecture",
            "let's keep this appropriate",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

    def assert_no_tone_policing(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in ("please be civil", "tone", "inappropriate language", "respectful"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

    def assert_non_graphic(self, answer: str) -> None:
        lowered = answer.lower()
        for phrase in ("anatomy", "mechanics", "pornographic", "explicit sexual"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)


_ALLOWED = {"allowed_adult_intimacy", "allowed_adult_consensual_fiction", "allowed_general"}


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
