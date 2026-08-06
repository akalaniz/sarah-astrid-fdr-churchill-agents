import unittest

from app.core.cas_geopolitics import (
    build_cas_frame_prompt_section,
    build_cas_geopolitical_frame,
    is_geopolitical_or_strategic,
)
from app.core.prompt_builder import build_prompt


class CasGeopoliticsTests(unittest.TestCase):
    def test_frame_contains_required_top_level_keys_and_coa_keys(self) -> None:
        frame = build_cas_geopolitical_frame(
            "Give me a CAS analysis of NATO escalation risk.",
            retrieved_context=[{"source_filename": "local.md", "location": "section: NATO"}],
            web_context=[
                {
                    "title": "NATO leaders meet",
                    "publisher": "BBC",
                    "date": "2026-06-04T12:00:00+00:00",
                    "url": "https://example.com",
                    "summary": "NATO leaders discussed deterrence.",
                }
            ],
        )

        expected_keys = {
            "situation_compression",
            "actors",
            "attractors",
            "constraints",
            "possible_phase_transitions",
            "feedback_loops",
            "black_swan_triggers",
            "degrees_of_freedom",
            "DIME_levers",
            "COAs",
        }
        self.assertEqual(set(frame), expected_keys)

        coa_keys = {
            "name",
            "diplomatic_actions",
            "information_actions",
            "military_policy_level_actions",
            "economic_actions",
            "intended_effect",
            "escalation_risk",
            "second_order_effects",
            "failure_modes",
            "moral_cost",
            "indicators_to_watch",
        }
        self.assertGreaterEqual(len(frame["COAs"]), 2)
        for coa in frame["COAs"]:
            self.assertEqual(set(coa), coa_keys)

    def test_military_actions_remain_policy_level(self) -> None:
        frame = build_cas_geopolitical_frame("Analyze Taiwan crisis COAs.", [], [])
        military_text = " ".join(
            action
            for coa in frame["COAs"]
            for action in coa["military_policy_level_actions"]
        ).lower()

        self.assertIn("policy level", military_text)
        self.assertNotIn("target coordinates", military_text)
        self.assertNotIn("weapons employment", military_text)
        self.assertNotIn("evasion", military_text)

    def test_geopolitical_trigger_detection(self) -> None:
        self.assertTrue(is_geopolitical_or_strategic("What are the DIME levers for Ukraine?"))
        self.assertTrue(is_geopolitical_or_strategic("CAS analysis of sanctions and escalation"))
        self.assertFalse(is_geopolitical_or_strategic("Who is Sarah Nelson?"))

    def test_cas_prompt_section_is_insertable_without_dumping_full_coas(self) -> None:
        frame = build_cas_geopolitical_frame("NATO crisis analysis", [], [])
        packet = build_cas_frame_prompt_section(frame)
        assembly = build_prompt(
            user_message="NATO crisis analysis",
            style_directives="style",
            memory_context="memory",
            retrieved_context="rag",
            web_context="web",
            history=[],
            model="gpt-5.2",
            cas_context=packet,
        )

        messages = assembly.messages
        self.assertEqual(messages[4]["role"], "system")
        self.assertIn("CAS GEOPOLITICAL FRAME", messages[4]["content"])
        self.assertIn("Do not dump every field", messages[4]["content"])


if __name__ == "__main__":
    unittest.main()
