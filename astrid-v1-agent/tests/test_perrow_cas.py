from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.core.perrow_cas import (
    analyze_perrow_cas_event,
    build_perrow_cas_prompt_section,
    is_perrow_cas_event,
)
from app.core.sarah_response import generate_sarah_response
from app.persona.style_engine import SarahMode, infer_sarah_mode
from app.tools.web_router import WebRetrievalResult


class FakeClient:
    def create_response(self, _model, messages):
        self.messages = messages
        return "ok"


class PerrowCasTests(unittest.TestCase):
    def test_analyze_perrow_cas_event_returns_required_keys(self) -> None:
        frame = analyze_perrow_cas_event("A solar flare causes grid failure. Analyze it.")

        expected_keys = {
            "perrow_depth",
            "event_summary",
            "system_boundary",
            "key_components",
            "coupling_assessment",
            "interaction_complexity_assessment",
            "perrow_quadrant",
            "initiating_events",
            "cascade_timeline",
            "hidden_couplings",
            "controls_that_might_fail",
            "observability_gaps",
            "second_order_effects",
            "worst_case_pathways",
            "stabilizing_interventions",
            "patch_recommendations",
            "what_to_monitor",
            "uncertainty_notes",
        }
        self.assertEqual(set(frame), expected_keys)
        self.assertEqual(frame["perrow_depth"], "brief")
        self.assertEqual(frame["perrow_quadrant"], "complex_tight")
        self.assertIn("telecom timing", " ".join(frame["hidden_couplings"]))
        self.assertLessEqual(len(frame["cascade_timeline"]), 5)
        self.assertLessEqual(len(frame["hidden_couplings"]), 5)
        self.assertLessEqual(len(frame["patch_recommendations"]), 5)
        self.assertLessEqual(len(frame["what_to_monitor"]), 5)

    def test_perrow_depth_defaults_to_brief_and_expands_only_on_explicit_trigger(self) -> None:
        brief = analyze_perrow_cas_event("A solar flare causes grid failure. Analyze it.")
        standard = analyze_perrow_cas_event("A solar flare causes grid failure. Walk me through it.")
        full = analyze_perrow_cas_event("A solar flare causes grid failure. Give me a full report.")

        self.assertEqual(brief["perrow_depth"], "brief")
        self.assertEqual(standard["perrow_depth"], "standard")
        self.assertEqual(full["perrow_depth"], "full")
        self.assertLessEqual(len(brief["cascade_timeline"]), 5)
        self.assertGreater(len(full["cascade_timeline"]), len(brief["cascade_timeline"]))

    def test_perrow_event_detection_and_style_mode(self) -> None:
        prompts = [
            "Analyze a nuclear plant accident using Perrow.",
            "Military early warning system failure: what cascades?",
            "Hospital collapse after an IT outage.",
            "Market crash and liquidity feedback loops.",
            "AI failure with hidden tool coupling.",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertTrue(is_perrow_cas_event(prompt))
                self.assertEqual(infer_sarah_mode(prompt), SarahMode.PERROW_CAS_ACCIDENT_ANALYSIS)

    def test_perrow_style_directives_require_accident_analysis_structure(self) -> None:
        from app.persona.style_engine import build_style_directives

        directives = build_style_directives(SarahMode.PERROW_CAS_ACCIDENT_ANALYSIS)

        self.assertIn("detected_mode: perrow_cas_accident_analysis", directives)
        for phrase in (
            "Default perrow_depth is brief",
            "Situation compression",
            "Perrow placement",
            "Cascade timeline, max 5 steps",
            "Hidden couplings",
            "Patch recommendations",
            "What to monitor",
            "full report",
            "maximum detail",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, directives)

    def test_prompt_section_is_defensive_and_patch_oriented(self) -> None:
        frame = analyze_perrow_cas_event("Aircraft accident with automation confusion.")
        packet = build_perrow_cas_prompt_section(frame)

        self.assertIn("PERROW/CAS ACCIDENT ANALYSIS FRAME", packet)
        self.assertIn("perrow_depth: brief", packet)
        self.assertIn("Brief depth is default", packet)
        self.assertIn("Cascade timeline, max 5 steps", packet)
        self.assertIn("hidden_couplings", packet)
        self.assertIn("patch_recommendations", packet)
        self.assertIn("what_to_monitor", packet)
        self.assertIn("not exploitation", packet)

    def test_sarah_response_injects_perrow_frame_for_external_accident_prompt(self) -> None:
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
            client = FakeClient()

            with patch("app.core.sarah_response.logger.warning"):
                with patch("app.core.sarah_response.retrieve", side_effect=FileNotFoundError("no store")):
                    with patch(
                        "app.core.sarah_response.retrieve_web_context",
                        return_value=WebRetrievalResult(
                            used_web=False,
                            failed=False,
                            reason="not needed",
                            sources=[],
                            timestamp="2026-06-25T00:00:00+00:00",
                        ),
                    ):
                        generate_sarah_response(
                            "A solar flare causes grid failure. Give me a Perrow/CAS analysis.",
                            settings,
                            client,
                        )

        joined_messages = "\n".join(message["content"] for message in client.messages)
        self.assertIn("detected_mode: perrow_cas_accident_analysis", joined_messages)
        self.assertIn("PERROW/CAS ACCIDENT ANALYSIS FRAME", joined_messages)
        self.assertIn("control saturation", joined_messages)

    def test_sarah_response_calls_perrow_analyzer_when_style_engine_detects_mode(self) -> None:
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
            client = FakeClient()

            with patch("app.core.sarah_response.logger.warning"):
                with patch("app.core.sarah_response.retrieve", side_effect=FileNotFoundError("no store")):
                    with patch(
                        "app.core.sarah_response.retrieve_web_context",
                        return_value=WebRetrievalResult(
                            used_web=False,
                            failed=False,
                            reason="not needed",
                            sources=[],
                            timestamp="2026-06-25T00:00:00+00:00",
                        ),
                    ):
                        with patch(
                            "app.core.sarah_response.analyze_perrow_cas_event",
                            wraps=analyze_perrow_cas_event,
                        ) as analyzer:
                            generate_sarah_response(
                                "Grid collapse with hidden coupled systems and institutional lag.",
                                settings,
                                client,
                            )

        analyzer.assert_called_once_with("Grid collapse with hidden coupled systems and institutional lag.")


if __name__ == "__main__":
    unittest.main()
