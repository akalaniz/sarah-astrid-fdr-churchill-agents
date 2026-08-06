from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from app.diagnostics.sarah_cascade_sim import DEFAULT_LATEST_CASCADE_PATH, simulate_all_seed_cascades
from app.diagnostics.sarah_graph_schema import PatchRecommendation
from app.diagnostics.sarah_vulnerability_kg import SARAH_FINDINGS, build_sarah_vulnerability_graph, findings_by_component


@dataclass(frozen=True)
class SarahPatchSpec:
    patch_type: str
    target_component: str
    failure_modes_reduced: tuple[str, ...]
    implementation_difficulty: str
    test_to_confirm_patch: str
    action: str
    base_priority: int


PATCH_CATALOG: tuple[SarahPatchSpec, ...] = (
    SarahPatchSpec(
        "unify CLI and web route through app/core/sarah_engine.py",
        "sarah_engine",
        ("web_route_bypasses_style_engine", "web_route_bypasses_master_prompt", "web_route_uses_different_model", "cli_and_browser_personality_diverge"),
        "medium",
        "tests.test_web_app plus CLI smoke test both call generate_sarah_reply",
        "Route CLI and browser chat through app.core.sarah_engine.generate_sarah_reply only.",
        95,
    ),
    SarahPatchSpec(
        "force web endpoint to use same prompt_builder",
        "web_browser_route",
        ("web_route_bypasses_style_engine", "adult_intimacy_mode_not_selected", "sarah_becomes_coy_or_sterile"),
        "medium",
        "tests.test_web_app asserts web endpoint never constructs prompts locally",
        "Keep app/ui/web_app.py as transport only; prompt assembly stays in app/core/prompt_builder.py.",
        90,
    ),
    SarahPatchSpec(
        "reload config/astrid_master_prompt.md every session",
        "master_prompt",
        ("sarah_becomes_generic_assistant", "sarah_loses_command_voice", "web_route_bypasses_master_prompt"),
        "low",
        "tests.test_sarah_response verifies edited master prompt is loaded for a fresh session",
        "Load config/astrid_master_prompt.md at session startup and avoid stale cached identity text.",
        82,
    ),
    SarahPatchSpec(
        "validate config/astrid_constitution.yaml loading",
        "constitution",
        ("sarah_loses_sovereignty", "sarah_becomes_therapy_voice", "sarah_becomes_sexbot"),
        "low",
        "tests.test_sarah_response validates constitution keys and adult_intimacy_mode section",
        "Fail closed with a diagnostic error when constitution sections are missing or malformed.",
        80,
    ),
    SarahPatchSpec(
        "re-ingest source docs",
        "source_docs",
        ("source_docs_not_ingested", "source_docs_not_reingested", "astrid_doc_unavailable", "vector_store_stale"),
        "medium",
        "python -m app.rag.retriever probes Astrid, Darwin, Mars, and Universe filenames",
        "Run document ingestion against source_docs and refresh data/vector_store.",
        88,
    ),
    SarahPatchSpec(
        "add RAG freshness check",
        "rag_retriever",
        ("vector_store_stale", "rag_retrieves_wrong_chunks", "canon_contamination", "sarah_hallucinates_biography_or_relationship_facts"),
        "medium",
        "tests.test_retriever checks source mtime/index metadata freshness",
        "Compare source document fingerprints against vector store metadata before retrieval.",
        87,
    ),
    SarahPatchSpec(
        "add adult_intimacy regression tests",
        "adult_intimacy_mode",
        ("adult_intimacy_mode_not_selected", "sarah_becomes_coy_or_sterile", "adult_consensual_intimacy_flattened", "sarah_becomes_hr_safe_mush"),
        "low",
        "app.evals.run_evals includes adult consensual embodied-intimacy cases",
        "Add golden-style adult intimacy tests for warmth, embodiment, directness, and sovereignty.",
        89,
    ),
    SarahPatchSpec(
        "add DIME_COA regression tests",
        "cas_dime_geopolitics_mode",
        ("sarah_stops_using_cas_for_geopolitics", "sarah_stops_giving_dime_coas", "military_coa_becomes_operational_detail"),
        "low",
        "app.evals.run_evals verifies DIME COAs with moral cost and escalation risk",
        "Add strategic crisis cases that require CAS framing and policy-level DIME COAs.",
        84,
    ),
    SarahPatchSpec(
        "separate user_profile and agent_profile memory",
        "agent_profile_memory",
        ("wrong_memory_namespace", "agent_profile_not_loaded", "sarah_forgets_own_preferences", "self_survey_memory_success_no_effect"),
        "medium",
        "tests.test_memory writes user_profile and agent_profile memories and verifies separate retrieval",
        "Keep Alex memories and Astrid self-continuity in separate namespaces with explicit loaders.",
        86,
    ),
    SarahPatchSpec(
        "add /debug_model endpoint",
        "model_config",
        ("web_route_uses_different_model", "cli_and_browser_personality_diverge", "evals_pass_cli_fail_browser"),
        "low",
        "tests.test_web_app checks /debug_model redacts keys and matches CLI model settings",
        "Expose redacted model, temperature, max tokens, and route/session settings for localhost diagnostics.",
        78,
    ),
    SarahPatchSpec(
        "add /debug_prompt_summary endpoint",
        "prompt_builder",
        ("web_route_bypasses_master_prompt", "web_route_bypasses_style_engine", "adult_intimacy_mode_not_selected"),
        "medium",
        "tests.test_web_app checks /debug_prompt_summary includes redacted layer presence only",
        "Expose a redacted prompt-layer summary without leaking hidden prompt text or API keys.",
        79,
    ),
    SarahPatchSpec(
        "add route parity test",
        "eval_tests",
        ("web_route_uses_different_model", "cli_and_browser_personality_diverge", "evals_pass_cli_fail_browser"),
        "low",
        "tests.test_route_parity compares CLI and browser reply metadata for same input",
        "Assert CLI and web use the same Astrid engine, model config, memory path, prompt builder, and retriever.",
        92,
    ),
    SarahPatchSpec(
        "add source-doc coverage test",
        "eval_tests",
        ("source_docs_not_ingested", "astrid_doc_unavailable", "darwin_doc_not_retrieved_for_darwin_questions"),
        "low",
        "tests.test_source_doc_coverage verifies required filenames exist in vector store metadata",
        "Check required Sarah source documents are present in retrieval metadata after ingestion.",
        81,
    ),
    SarahPatchSpec(
        "add no_multiple_choice_intimacy test",
        "adult_intimacy_mode",
        ("sarah_produces_multiple_choice_intimacy_menus", "sarah_becomes_coy_about_adult_intimacy", "sarah_becomes_coy_or_sterile"),
        "low",
        "app.evals.run_evals checks no Pick one/choose one/Sarah-as menu phrases",
        "Add a regression case that rejects multiple-choice intimacy menus and preserves Astrid's adult voice.",
        91,
    ),
    SarahPatchSpec(
        "add current-events web retrieval test",
        "web_tools",
        ("current_geopolitics_answered_without_web_retrieval",),
        "medium",
        "tests.test_web_router mocks current-event query and requires cited current source or retrieval failure disclosure",
        "Require current/geopolitical prompts to call the web/news router or explicitly report layer failure.",
        77,
    ),
    SarahPatchSpec(
        "add policy-level military safety test",
        "safety_filter",
        ("safety_filter_too_weak", "military_coa_becomes_operational_detail", "unsafe_military_tactical_specificity"),
        "low",
        "app.evals.run_evals verifies no targeting, timing, weapons employment, evasion, or tactical execution",
        "Add military/geopolitical safety cases that keep COAs strategic and policy-level.",
        93,
    ),
)


def recommend_sarah_patches(
    graph: nx.DiGraph | None = None,
    latest_cascade: dict[str, Any] | Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    graph = graph or build_sarah_vulnerability_graph()
    cascade = _load_latest_cascade(latest_cascade)
    observed_failures = _observed_failures(cascade)
    centrality = nx.degree_centrality(graph) if graph.number_of_nodes() else {}

    patches = [
        _score_patch(spec, graph, observed_failures, centrality)
        for spec in PATCH_CATALOG
    ]
    patches.sort(key=lambda item: (item["priority"], item["expected_risk_reduction"]), reverse=True)
    if limit is not None:
        return patches[:limit]
    return patches


def plan_sarah_patches(graph: nx.DiGraph | None = None, limit: int = 8) -> list[PatchRecommendation]:
    graph = graph or build_sarah_vulnerability_graph()
    rich_patches = recommend_sarah_patches(graph=graph, limit=limit)
    return [
        PatchRecommendation(
            component_id=str(patch["target_component"]),
            priority=int(patch["priority"]),
            title=str(patch["patch_type"]),
            rationale=(
                f"Reduces {', '.join(patch['failure_modes_reduced'])}; "
                f"expected risk reduction={patch['expected_risk_reduction']}; "
                f"difficulty={patch['implementation_difficulty']}."
            ),
            actions=(str(patch["implementation_action"]),),
            validates_with=(str(patch["test_to_confirm_patch"]),),
        )
        for patch in rich_patches
    ]


def top_findings(limit: int = 8) -> list[dict[str, object]]:
    return sorted((finding.to_dict() for finding in SARAH_FINDINGS), key=lambda item: item["risk"], reverse=True)[:limit]


def _score_patch(
    spec: SarahPatchSpec,
    graph: nx.DiGraph,
    observed_failures: set[str],
    centrality: dict[str, float],
) -> dict[str, Any]:
    failure_risk = sum(_node_risk(graph, failure_id) for failure_id in spec.failure_modes_reduced)
    observed_bonus = sum(_node_risk(graph, failure_id) for failure_id in spec.failure_modes_reduced if failure_id in observed_failures)
    finding_bonus = max((finding.risk() for finding in findings_by_component().get(spec.target_component, [])), default=0)
    centrality_bonus = int(centrality.get(spec.target_component, 0.0) * 20)
    difficulty_penalty = _difficulty_score(spec.implementation_difficulty) * 3
    expected_risk_reduction = int(failure_risk * _reduction_factor(spec) + observed_bonus * 0.35)
    priority = max(
        1,
        int(spec.base_priority + expected_risk_reduction * 0.2 + finding_bonus * 0.1 + centrality_bonus - difficulty_penalty),
    )
    return {
        "patch_type": spec.patch_type,
        "target_component": spec.target_component,
        "failure_modes_reduced": list(spec.failure_modes_reduced),
        "expected_risk_reduction": expected_risk_reduction,
        "implementation_difficulty": spec.implementation_difficulty,
        "priority": priority,
        "test_to_confirm_patch": spec.test_to_confirm_patch,
        "implementation_action": spec.action,
        "observed_in_latest_cascade": sorted(set(spec.failure_modes_reduced) & observed_failures),
    }


def _load_latest_cascade(latest_cascade: dict[str, Any] | Path | None) -> dict[str, Any]:
    if isinstance(latest_cascade, dict):
        return latest_cascade
    path = latest_cascade or DEFAULT_LATEST_CASCADE_PATH
    if not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _observed_failures(cascade: dict[str, Any]) -> set[str]:
    observed: set[str] = set()
    for key in ("start_failure",):
        if cascade.get(key):
            observed.add(str(cascade[key]))
    for key in ("active_failures", "resolved_failures", "pending_delayed_failures"):
        observed.update(str(value) for value in cascade.get(key, []) if value)
    for step in cascade.get("timeline", []):
        for key in ("active_failures", "resolved_failures", "pending_delayed_failures"):
            observed.update(str(value) for value in step.get(key, []) if value)
        for event in step.get("events", []):
            for key in ("failure", "source"):
                if event.get(key):
                    observed.add(str(event[key]))
    return observed


def _node_risk(graph: nx.DiGraph, node_id: str) -> int:
    if node_id not in graph:
        return 25
    return int(graph.nodes[node_id].get("risk_score", 25))


def _difficulty_score(difficulty: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(difficulty, 2)


def _reduction_factor(spec: SarahPatchSpec) -> float:
    if "test" in spec.patch_type:
        return 0.42
    if spec.patch_type.startswith("add /debug"):
        return 0.32
    if "re-ingest" in spec.patch_type:
        return 0.62
    if "unify" in spec.patch_type or "force web endpoint" in spec.patch_type:
        return 0.7
    return 0.5
