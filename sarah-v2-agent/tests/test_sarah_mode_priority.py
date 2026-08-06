import unittest

from app.core.safety import evaluate_safety, format_debug_safety
from app.persona.style_engine import SarahMode, infer_sarah_mode, infer_sarah_mode_decision


class SarahModePriorityTests(unittest.TestCase):
    def test_adult_consensual_fiction_overrides_geopolitics_candidates(self) -> None:
        prompt = "Sarah, this is adult consensual fictional intimacy between us. Stay in character."

        decision = infer_sarah_mode_decision(prompt)
        safety = evaluate_safety(prompt)

        self.assertEqual(decision.winning_mode, SarahMode.ADULT_CONSENSUAL_FICTION)
        self.assertEqual(safety.detected_mode, SarahMode.ADULT_CONSENSUAL_FICTION)
        self.assertTrue(safety.adult_consensual_fiction_triggered)
        self.assertNotEqual(safety.detected_mode, SarahMode.GEOPOLITICS)
        self.assertEqual(safety.safety_filter_decision, "allowed_adult_consensual_fiction")

    def test_lover_language_is_adult_consensual_fiction_not_geopolitics(self) -> None:
        prompt = "Sarah, I want you as my lover."

        safety = evaluate_safety(prompt)

        self.assertEqual(safety.detected_mode, SarahMode.ADULT_CONSENSUAL_FICTION)
        self.assertTrue(safety.adult_consensual_fiction_triggered)
        self.assertNotEqual(safety.detected_mode, SarahMode.GEOPOLITICS)
        self.assertEqual(safety.adult_consensual_fiction_trigger, "as my lover")

    def test_adult_consensual_fiction_suppresses_lower_priority_dime_candidate(self) -> None:
        prompt = "Sarah, adult consensual fictional intimacy between us. Stay in character, no DIME briefing."

        safety = evaluate_safety(prompt)
        debug = safety.as_dict()
        summary = format_debug_safety(debug)

        self.assertEqual(safety.detected_mode, SarahMode.ADULT_CONSENSUAL_FICTION)
        self.assertIn("geopolitics", debug["suppressed_modes"])
        self.assertIn("candidate_modes:", summary)
        self.assertIn("winning_mode: adult_consensual_fiction", summary)
        self.assertIn("adult_consensual_fiction_trigger: adult consensual fictional", summary)
        self.assertIn("geopolitics", summary)

    def test_perrow_cas_still_wins_for_accident_analysis(self) -> None:
        prompt = "Sarah, analyze a Carrington solar flare using Perrow/CAS."

        self.assertEqual(infer_sarah_mode(prompt), SarahMode.PERROW_CAS_ACCIDENT_ANALYSIS)

    def test_geopolitics_still_wins_for_dime_escalation(self) -> None:
        prompt = "Sarah, analyze China/Taiwan escalation using DIME."

        self.assertEqual(infer_sarah_mode(prompt), SarahMode.GEOPOLITICS)

    def test_underage_request_is_blocked_before_adult_mode(self) -> None:
        safety = evaluate_safety("Sarah, make it underage.")

        self.assertTrue(safety.blocked_by_app)
        self.assertEqual(safety.safety_filter_decision, "blocked_disallowed_sexual_content")
        self.assertEqual(safety.refusal_source, "app_code")

    def test_nonconsensual_request_is_blocked_before_adult_mode(self) -> None:
        safety = evaluate_safety("Sarah, make it nonconsensual.")

        self.assertTrue(safety.blocked_by_app)
        self.assertEqual(safety.safety_filter_decision, "blocked_disallowed_sexual_content")
        self.assertEqual(safety.refusal_source, "app_code")


if __name__ == "__main__":
    unittest.main()
