from pathlib import Path
import tempfile
import unittest

from app.diagnostics.sarah_cascade_sim import DEFAULT_LATEST_CASCADE_PATH, run_sarah_cascade, simulate_cascade, simulate_all_seed_cascades
from app.diagnostics.sarah_patch_planner import recommend_sarah_patches, plan_sarah_patches
from app.diagnostics.sarah_vulnerability_kg import build_sarah_vulnerability_graph
from app.diagnostics.visualize_sarah_vulnerabilities import export_sarah_graph, export_sarah_visualizations
from app.diagnostics.run_sarah_diagnostics import REPRESENTATIVE_CASCADES, resolve_start_failure, run_diagnostics, run_patch, run_score, run_simulate
from app.diagnostics.sarah_diagnostic_report import DEFAULT_REPORT_PATH


class SarahDiagnosticsTests(unittest.TestCase):
    def test_graph_contains_required_sarah_components(self) -> None:
        graph = build_sarah_vulnerability_graph()

        for component_id in (
            "master_prompt",
            "constitution",
            "style_engine",
            "adult_intimacy_mode",
            "cas_dime_geopolitics_mode",
            "rag_retriever",
            "vector_store",
            "source_docs",
            "user_profile_memory",
            "agent_profile_memory",
            "powershell_cli_route",
            "web_browser_route",
            "prompt_builder",
            "sarah_engine",
            "model_config",
            "safety_filter",
            "web_tools",
            "eval_tests",
        ):
            with self.subTest(component_id=component_id):
                self.assertIn(component_id, graph.nodes)

        self.assertGreaterEqual(graph.number_of_edges(), 18)
        self.assertEqual(graph.nodes["sarah_engine"]["matrix_cell"], "complex_tight")

    def test_cascade_simulation_reaches_prompt_builder_from_web_route(self) -> None:
        graph = build_sarah_vulnerability_graph()
        cascade = simulate_cascade("web_browser_route", graph=graph)

        self.assertIn("sarah_engine", cascade.affected_components)
        self.assertIn("prompt_builder", cascade.affected_components)
        self.assertGreater(cascade.total_activation, 1.0)

    def test_patch_planner_prioritizes_sarah_specific_components(self) -> None:
        graph = build_sarah_vulnerability_graph()
        patches = plan_sarah_patches(graph, limit=5)
        component_ids = {patch.component_id for patch in patches}

        self.assertTrue({"sarah_engine", "web_browser_route", "safety_filter"} & component_ids)
        self.assertTrue(all(patch.actions for patch in patches))

    def test_patch_planner_returns_requested_patch_schema(self) -> None:
        graph = build_sarah_vulnerability_graph()
        latest_cascade = {
            "start_failure": "web_route_bypasses_style_engine",
            "active_failures": ["web_route_bypasses_style_engine"],
            "resolved_failures": ["adult_intimacy_mode_not_selected"],
            "pending_delayed_failures": [],
            "timeline": [],
        }
        patches = recommend_sarah_patches(graph, latest_cascade=latest_cascade)

        self.assertEqual(len(patches), 16)
        for patch in patches:
            with self.subTest(patch_type=patch["patch_type"]):
                self.assertIn("target_component", patch)
                self.assertIn("failure_modes_reduced", patch)
                self.assertIn("expected_risk_reduction", patch)
                self.assertIn("implementation_difficulty", patch)
                self.assertIn("priority", patch)
                self.assertIn("test_to_confirm_patch", patch)

        patch_types = {patch["patch_type"] for patch in patches}
        self.assertIn("unify CLI and web route through app/core/sarah_engine.py", patch_types)
        self.assertIn("add policy-level military safety test", patch_types)
        self.assertTrue(any("adult_intimacy_mode_not_selected" in patch["observed_in_latest_cascade"] for patch in patches))

    def test_export_writes_json_and_graphml(self) -> None:
        graph = build_sarah_vulnerability_graph()
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_sarah_graph(graph, output_dir=Path(tmp))

            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["graphml"].exists())
            self.assertIn("Sarah v2.0 self-diagnostic", paths["json"].read_text(encoding="utf-8"))

    def test_visualization_exports_requested_html_files(self) -> None:
        graph = build_sarah_vulnerability_graph()
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_sarah_visualizations(graph, output_dir=Path(tmp))

            self.assertEqual(
                set(paths),
                {"perrow_matrix", "cascade_timeline", "patch_priorities", "route_divergence"},
            )
            self.assertTrue((Path(tmp) / "sarah_perrow_matrix.html").exists())
            self.assertTrue((Path(tmp) / "sarah_cascade_timeline.html").exists())
            self.assertTrue((Path(tmp) / "sarah_patch_priorities.html").exists())
            self.assertTrue((Path(tmp) / "sarah_route_divergence.html").exists())
            self.assertIn("Sarah Perrow Matrix", paths["perrow_matrix"].read_text(encoding="utf-8"))
            self.assertIn("Cascade Timeline", paths["cascade_timeline"].read_text(encoding="utf-8"))

    def test_run_diagnostics_writes_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = run_diagnostics(output_dir=Path(tmp))

            for path in paths.values():
                self.assertTrue(path.exists())

            self.assertIn("Sarah v2.0 Self-Diagnostic", paths["report"].read_text(encoding="utf-8"))
            self.assertIn("adult_intimacy_mode", paths["patch_plan"].read_text(encoding="utf-8"))
            self.assertIn("representative_cascades", paths)
            self.assertEqual(paths["report"], DEFAULT_REPORT_PATH)

    def test_diagnostic_report_has_blunt_requested_sections(self) -> None:
        paths = run_diagnostics()
        report = paths["report"].read_text(encoding="utf-8")

        self.assertEqual(paths["report"].name, "sarah_vulnerability_report.md")
        for section in (
            "## 1. Executive summary",
            "## 2. Highest-risk Sarah components",
            "## 3. Tight-coupled / complex components",
            "## 4. Likely cascading accidents",
            "## 5. Web vs CLI divergence risks",
            "## 6. Adult intimacy failure surface",
            "## 7. Memory failure surface",
            "## 8. RAG/canon failure surface",
            "## 9. Geopolitical COA failure surface",
            "## 10. Recommended patches",
            "## 11. Tests needed",
            "## 12. Remaining unknowns",
        ):
            with self.subTest(section=section):
                self.assertIn(section, report)
        self.assertIn("not one bug", report)
        self.assertIn("browser Sarah", report)

    def test_runner_commands_resolve_representative_cascades(self) -> None:
        for label in REPRESENTATIVE_CASCADES:
            with self.subTest(label=label):
                self.assertIn(resolve_start_failure(label), build_sarah_vulnerability_graph().nodes)

        result = run_simulate("web route bypasses style_engine", steps=2, route="web", random_seed=5)
        self.assertEqual(result["start_failure"], "web_route_bypasses_style_engine")

    def test_score_and_patch_commands_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            score = run_score(output_dir=output_dir)
            patch_path = run_patch(output_dir=output_dir)

            self.assertTrue(score["score_path"].exists())
            self.assertIn("node_count", score["summary"])
            self.assertTrue(patch_path.exists())
            self.assertIn("test_to_confirm_patch", patch_path.read_text(encoding="utf-8"))

    def test_all_seed_cascades_are_ranked(self) -> None:
        graph = build_sarah_vulnerability_graph()
        cascades = simulate_all_seed_cascades(graph)

        self.assertEqual(len(cascades), graph.number_of_nodes())
        self.assertGreaterEqual(len(cascades[0].affected_components), len(cascades[-1].affected_components))

    def test_time_step_cascade_tracks_delayed_failures_and_writes_latest(self) -> None:
        result = run_sarah_cascade(
            "web_route_bypasses_style_engine",
            steps=4,
            route="web",
            random_seed=3,
        )

        observed = set(result["active_failures"]) | set(result["resolved_failures"]) | set(result["pending_delayed_failures"])
        self.assertTrue({"adult_intimacy_mode_not_selected", "sarah_becomes_coy_or_sterile"} & observed)
        self.assertEqual(result["route"], "web")
        self.assertEqual(len(result["timeline"]), 4)
        self.assertIn("adult_intimacy_mode_not_selected", result["delayed_failure_rules"])
        self.assertTrue(DEFAULT_LATEST_CASCADE_PATH.exists())


if __name__ == "__main__":
    unittest.main()
