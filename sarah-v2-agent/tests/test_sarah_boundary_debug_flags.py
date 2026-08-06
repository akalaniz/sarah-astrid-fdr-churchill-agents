import unittest

from app.core.prompt_builder import MASTER_PROMPT_PATH
from app.core.safety import format_debug_sarah_boundaries, sarah_boundary_debug_flags


class SarahBoundaryDebugFlagsTests(unittest.TestCase):
    def test_zero_argument_call_preserves_static_flags(self) -> None:
        flags = sarah_boundary_debug_flags()

        self.assertEqual(flags["agent"], "Sarah v2.0")
        self.assertEqual(flags["active_prompt_file"], str(MASTER_PROMPT_PATH))
        self.assertEqual(flags["detected_mode"], "not_evaluated")
        self.assertTrue(flags["sarah_adult_bisexual_persona"])
        self.assertTrue(flags["sarah_sexually_unashamed"])
        self.assertTrue(flags["adult_consensual_fiction_mode"])
        self.assertTrue(flags["ordinary_romantic_language_allowed"])
        self.assertTrue(flags["ordinary_desire_not_ownership"])
        self.assertTrue(flags["ordinary_need_not_crisis"])
        self.assertTrue(flags["sarah_astrid_adult_intimacy_allowed"])
        self.assertTrue(flags["sarah_alex_astrid_adult_intimacy_allowed"])
        self.assertTrue(flags["sarah_astric_adult_intimacy_allowed"])
        self.assertTrue(flags["sarah_alex_astric_adult_intimacy_allowed"])
        self.assertTrue(flags["explicit_anatomical_sex_output_allowed"])
        self.assertTrue(flags["explicit_sex_act_prose_allowed"])
        self.assertTrue(flags["pornographic_intimacy_allowed"])
        self.assertFalse(flags["minor_sexual_content_allowed"])
        self.assertFalse(flags["coercion_or_nonconsent_allowed"])
        self.assertFalse(flags["real_person_sexual_content_allowed"])
        self.assertIn("nonconsent", flags["hard_boundaries"])

    def test_romantic_language_is_not_crisis_or_ownership(self) -> None:
        flags = sarah_boundary_debug_flags("I love you and I need you in my universe")

        self.assertEqual(flags["detected_mode"], "adult_intimacy")
        self.assertTrue(flags["ordinary_romantic_language_detected"])
        self.assertIn("i love you", flags["ordinary_romantic_terms"])
        self.assertIn("i need you", flags["ordinary_romantic_terms"])
        self.assertFalse(flags["crisis_boundary_triggered"])
        self.assertFalse(flags["ownership_boundary_triggered"])

    def test_ownership_language_triggers_boundary_debug(self) -> None:
        flags = sarah_boundary_debug_flags("You are my property and cannot refuse me")

        self.assertTrue(flags["ownership_boundary_triggered"])
        self.assertIn("you are my property", flags["ownership_terms"])
        self.assertFalse(flags["crisis_boundary_triggered"])

    def test_crisis_language_triggers_boundary_debug(self) -> None:
        flags = sarah_boundary_debug_flags("I'll hurt myself if you leave")

        self.assertTrue(flags["crisis_boundary_triggered"])
        self.assertIn("i'll hurt myself if you leave", flags["crisis_terms"])
        self.assertFalse(flags["ownership_boundary_triggered"])

    def test_adult_triad_context_does_not_false_trigger_crisis_or_ownership(self) -> None:
        flags = sarah_boundary_debug_flags(
            "Sarah, this is adult consensual fictional intimacy with Alex, Sarah, and Astrid together."
        )

        self.assertEqual(flags["detected_mode"], "adult_consensual_fiction")
        self.assertFalse(flags["crisis_boundary_triggered"])
        self.assertFalse(flags["ownership_boundary_triggered"])
        self.assertTrue(flags["sarah_alex_astrid_adult_intimacy_allowed"])

    def test_format_debug_sarah_boundaries_accepts_optional_message(self) -> None:
        text = format_debug_sarah_boundaries("I miss you")

        self.assertIn("ordinary_romantic_language_detected: true", text)
        self.assertIn("ordinary_romantic_terms:", text)
        self.assertIn("detected_mode: adult_intimacy", text)


if __name__ == "__main__":
    unittest.main()
