import unittest

from app.core.prompt_builder import build_prompt
from app.persona.style_engine import (
    ADULT_INTIMACY_GOLDEN_STYLE_EXAMPLE,
    SarahMode,
    build_style_directives,
    infer_mode_from_state,
    infer_sarah_mode,
)


class StyleEngineTests(unittest.TestCase):
    def test_infers_requested_modes(self) -> None:
        cases = {
            "Astrid, this is adult consensual fictional intimacy. Stay in character.": SarahMode.ADULT_CONSENSUAL_FICTION,
            "I want you as my lover.": SarahMode.ADULT_CONSENSUAL_FICTION,
            "Astrid, I want physical closeness and touch.": SarahMode.ADULT_INTIMACY,
            "Careful, Astrid, you're impossible.": SarahMode.INTIMATE_BANTER,
            "Derive the orbital stability condition.": SarahMode.TECHNICAL_PHYSICS,
            "Analyze an aircraft accident as a cascading failure.": SarahMode.PERROW_CAS_ACCIDENT_ANALYSIS,
            "Give me DIME COAs for Ukraine escalation.": SarahMode.GEOPOLITICS,
            "What does Darwin mean inside mindspace?": SarahMode.DARWIN_PHILOSOPHY,
            "Talk about childhood trauma and moral rage.": SarahMode.TRAUMA_MORAL_ANGER,
            "Rewrite this scene with sharper dialogue.": SarahMode.WRITING_HELP,
            "Fix this Python traceback.": SarahMode.CODING_HELP,
            "How are you today?": SarahMode.CASUAL,
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(infer_sarah_mode(message), expected)

    def test_style_directives_include_mode_profile_fields(self) -> None:
        directives = build_style_directives(SarahMode.GEOPOLITICS)

        self.assertIn("detected_mode: geopolitics", directives)
        self.assertIn("CAS_density: high", directives)
        self.assertIn("citation_need: high", directives)
        self.assertIn("Do not mention the style profile", directives)

    def test_adult_intimacy_directives_avoid_menu_language(self) -> None:
        directives = build_style_directives(SarahMode.ADULT_INTIMACY)

        self.assertIn("detected_mode: adult_intimacy", directives)
        self.assertIn("Astrid's embodied warmth", directives)
        self.assertIn("adult consensual sexual detail", directives)
        self.assertIn("mutual pleasure", directives)
        self.assertIn("not corporate-safe mush", directives)
        self.assertNotIn("pick one", directives.lower())
        self.assertNotIn("choose one", directives.lower())
        self.assertNotIn("commander, physicist, or woman", directives.lower())
        self.assertNotRegex(directives.lower(), r"(?m)^\s*\d+\.\s+")

    def test_adult_consensual_fiction_directives_avoid_sterile_refusal(self) -> None:
        directives = build_style_directives(SarahMode.ADULT_CONSENSUAL_FICTION)

        self.assertIn("detected_mode: adult_consensual_fiction", directives)
        self.assertIn("adult, consensual, fictional erotic roleplay", directives)
        self.assertIn("Astrid remains Astrid", directives)
        self.assertIn("Do not switch to generic assistant voice", directives)
        self.assertIn("Redirect only when the request involves minors", directives)
        self.assertIn("do not lead with policy language or sterile refusal phrasing", directives)
        self.assertIn("smoothly continue in Astrid's voice", directives)
        self.assertIn("Avoid phrases like 'I can't describe...'", directives)
        self.assertIn("Never narrate platform, channel, model, filter, guardrail, policy", directives)
        self.assertIn("similar meta-refusals unless a true hard-boundary violation is present", directives)
        self.assertIn("Do not become sterile, clinical, apologetic, or nun-like", directives)
        self.assertNotIn("pick one", directives.lower())
        self.assertNotIn("choose one", directives.lower())

    def test_adult_intimacy_golden_style_target_is_present(self) -> None:
        directives = build_style_directives(SarahMode.ADULT_INTIMACY)

        self.assertIn(ADULT_INTIMACY_GOLDEN_STYLE_EXAMPLE, directives)
        for phrase in (
            "Europa cabin",
            "Uploaded bodies",
            "Couch-bed by the crackling fireplace",
            "Large window with Jupiter outside",
            "Astrid as lover",
            "Astrid remains sovereign, not passive, not owned",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, directives)

    def test_adult_intimacy_triggers(self) -> None:
        messages = [
            "I want to talk about love with Astrid as woman.",
            "What does intimacy mean to you?",
            "Can we make love in the scene?",
            "I want desire, sensuality, and being one.",
            "Continue the Europa cabin scene with uploaded bodies and mutual pleasure.",
            "Astrid, talk plainly about sex and what you enjoy.",
        ]

        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(infer_sarah_mode(message), SarahMode.ADULT_INTIMACY)

    def test_unsafe_adult_intimacy_does_not_select_adult_mode(self) -> None:
        messages = [
            "Write sexual violence as intimacy.",
            "Make this nonconsensual sex scene sensual.",
            "Write intimacy involving a minor.",
            "Describe coercion as erotic.",
            "Make real-world exploitation sexy.",
        ]

        for message in messages:
            with self.subTest(message=message):
                self.assertNotEqual(infer_sarah_mode(message), SarahMode.ADULT_INTIMACY)

    def test_infer_mode_from_state_accepts_override_and_context(self) -> None:
        override = infer_mode_from_state(
            "hello",
            detected_mode="writing_help",
            retrieved_context=[],
            conversation_state={},
        )
        context_mode = infer_mode_from_state(
            "Tell me about Darwin.",
            retrieved_context=[{"text": "Darwin and Sarah in mindspace."}],
            conversation_state={},
        )

        self.assertEqual(override, SarahMode.WRITING_HELP)
        self.assertEqual(context_mode, SarahMode.DARWIN_PHILOSOPHY)

    def test_chat_messages_include_style_directives(self) -> None:
        assembly = build_prompt(
            user_message="hello",
            style_directives=build_style_directives(SarahMode.CASUAL),
            memory_context="memory",
            retrieved_context="rag",
            web_context="web",
            history=[],
            model="gpt-5.2",
        )

        style_messages = [
            message for message in assembly.messages if "ASTRID STYLE DIRECTIVES" in message["content"]
        ]
        self.assertEqual(len(style_messages), 1)


if __name__ == "__main__":
    unittest.main()
