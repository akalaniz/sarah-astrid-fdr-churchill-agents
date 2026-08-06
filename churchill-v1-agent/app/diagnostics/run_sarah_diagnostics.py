from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.diagnostics.sarah_cascade_sim import run_sarah_cascade, simulate_all_seed_cascades
from app.diagnostics.sarah_diagnostic_report import write_sarah_diagnostic_report
from app.diagnostics.sarah_patch_planner import plan_sarah_patches, recommend_sarah_patches
from app.diagnostics.sarah_vulnerability_kg import build_sarah_vulnerability_graph, graph_summary
from app.diagnostics.visualize_sarah_vulnerabilities import DEFAULT_OUTPUT_DIR, export_sarah_graph, export_sarah_visualizations


DEFAULT_CASCADE_PATH = PROJECT_ROOT / "data" / "diagnostics" / "sarah_cascades.json"
DEFAULT_PATCH_PATH = PROJECT_ROOT / "data" / "diagnostics" / "sarah_patch_plan.json"
DEFAULT_REPRESENTATIVE_CASCADE_PATH = PROJECT_ROOT / "data" / "diagnostics" / "representative_sarah_cascades.json"

REPRESENTATIVE_CASCADES: tuple[str, ...] = (
    "web route bypasses style_engine",
    "Churchill doc not retrieved for intimacy",
    "agent_profile not loaded",
    "stale vector store",
    "safety_filter too aggressive",
    "current geopolitics answered without web",
    "DIME COA engine bypassed",
)

START_ALIASES: dict[str, str] = {
    "web route bypasses style_engine": "web_route_bypasses_style_engine",
    "web route bypasses style engine": "web_route_bypasses_style_engine",
    "web_route_bypasses_style_engine": "web_route_bypasses_style_engine",
    "churchill doc not retrieved for intimacy": "churchill_doc_not_retrieved_for_intimacy",
    "churchill_doc_not_retrieved_for_intimacy": "churchill_doc_not_retrieved_for_intimacy",
    "agent_profile not loaded": "agent_profile_not_loaded",
    "agent profile not loaded": "agent_profile_not_loaded",
    "agent_profile_not_loaded": "agent_profile_not_loaded",
    "stale vector store": "vector_store_stale",
    "vector store stale": "vector_store_stale",
    "vector_store_stale": "vector_store_stale",
    "safety_filter too aggressive": "safety_filter_too_aggressive",
    "safety filter too aggressive": "safety_filter_too_aggressive",
    "safety_filter_too_aggressive": "safety_filter_too_aggressive",
    "current geopolitics answered without web": "current_geopolitics_answered_without_web_retrieval",
    "current geopolitics answered without web retrieval": "current_geopolitics_answered_without_web_retrieval",
    "current_geopolitics_answered_without_web_retrieval": "current_geopolitics_answered_without_web_retrieval",
    "dime coa engine bypassed": "dime_coa_engine_bypassed",
    "dime_coa_engine_bypassed": "dime_coa_engine_bypassed",
}


def run_diagnostics(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    return run_full(output_dir=output_dir)


def run_build(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_sarah_vulnerability_graph()
    return export_sarah_graph(graph, output_dir=output_dir)


def run_score(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_sarah_vulnerability_graph()
    summary = graph_summary(graph)
    score_path = output_dir / "sarah_diagnostic_score.json"
    score_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"summary": summary, "score_path": score_path}


def run_simulate(
    start: str,
    steps: int = 20,
    route: str = "web",
    random_seed: int | None = None,
) -> dict[str, Any]:
    start_node = resolve_start_failure(start)
    return run_sarah_cascade(start_node, steps=steps, route=route, random_seed=random_seed)


def run_patch(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_sarah_vulnerability_graph()
    patch_path = output_dir / DEFAULT_PATCH_PATH.name
    patch_path.write_text(
        json.dumps(recommend_sarah_patches(graph), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return patch_path


def run_full(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_sarah_vulnerability_graph()
    cascades = simulate_all_seed_cascades(graph)
    patches = plan_sarah_patches(graph)
    rich_patches = recommend_sarah_patches(graph)
    representative_cascades = run_representative_cascades()
    export_paths = export_sarah_graph(graph, output_dir=output_dir)
    visual_paths = export_sarah_visualizations(graph)

    cascade_path = output_dir / DEFAULT_CASCADE_PATH.name
    patch_path = output_dir / DEFAULT_PATCH_PATH.name
    representative_path = output_dir / DEFAULT_REPRESENTATIVE_CASCADE_PATH.name

    cascade_path.write_text(
        json.dumps([cascade.to_dict() for cascade in cascades], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    patch_path.write_text(
        json.dumps(rich_patches, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    representative_path.write_text(
        json.dumps(representative_cascades, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    report_path = write_sarah_diagnostic_report(graph, cascades, patches, export_paths)

    return {
        **export_paths,
        **visual_paths,
        "cascades": cascade_path,
        "representative_cascades": representative_path,
        "patch_plan": patch_path,
        "report": report_path,
    }


def run_representative_cascades(steps: int = 20, route: str = "web") -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, label in enumerate(REPRESENTATIVE_CASCADES, start=1):
        result = run_simulate(label, steps=steps, route=route, random_seed=100 + index)
        result["representative_label"] = label
        results.append(result)
    return results


def resolve_start_failure(start: str) -> str:
    graph = build_sarah_vulnerability_graph()
    if start in graph:
        return start

    normalized = _normalize_label(start)
    alias = START_ALIASES.get(normalized)
    if alias and alias in graph:
        return alias

    underscored = normalized.replace(" ", "_").replace("-", "_")
    if underscored in graph:
        return underscored

    labels = {
        _normalize_label(str(attrs.get("label", attrs.get("name", node_id)))): node_id
        for node_id, attrs in graph.nodes(data=True)
    }
    if normalized in labels:
        return labels[normalized]

    raise ValueError(f"Unknown FDR cascade start: {start}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Churchill v1.0 self-diagnostic vulnerability graph analysis.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("build", help="Build and export the Churchill vulnerability graph.")
    subparsers.add_parser("score", help="Score the Churchill vulnerability graph.")

    simulate_parser = subparsers.add_parser("simulate", help="Run a time-stepped FDR cascade simulation.")
    simulate_parser.add_argument("--start", required=True, help="Failure label or node id to start from.")
    simulate_parser.add_argument("--steps", type=int, default=20)
    simulate_parser.add_argument("--route", choices=("web", "cli"), default="web")
    simulate_parser.add_argument("--random-seed", type=int, default=None)

    subparsers.add_parser("patch", help="Recommend FDR patches from graph and latest cascade.")
    subparsers.add_parser("full-run", help="Build, score, simulate representative cascades, patch, and report.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "full-run"

    if command == "build":
        paths = run_build(output_dir=args.output_dir)
        _print_paths("FDR graph build complete.", paths, as_json=args.json)
        return

    if command == "score":
        score = run_score(output_dir=args.output_dir)
        if args.json:
            print(json.dumps({"summary": score["summary"], "score_path": str(score["score_path"])}, indent=2))
        else:
            print("FDR graph score complete.")
            print(json.dumps(score["summary"], indent=2))
            print(f"score: {score['score_path']}")
        return

    if command == "simulate":
        result = run_simulate(
            args.start,
            steps=args.steps,
            route=args.route,
            random_seed=args.random_seed,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("FDR cascade simulation complete.")
            print(f"start: {result['start_failure']}")
            print(f"route: {result['route']}")
            print(f"steps: {len(result['timeline'])}")
            print(f"active: {', '.join(result['active_failures']) or '(none)'}")
            print(f"resolved: {', '.join(result['resolved_failures']) or '(none)'}")
            print(f"pending: {', '.join(result['pending_delayed_failures']) or '(none)'}")
        return

    if command == "patch":
        patch_path = run_patch(output_dir=args.output_dir)
        if args.json:
            print(json.dumps({"patch_plan": str(patch_path)}, indent=2))
        else:
            print("FDR patch plan complete.")
            print(f"patch_plan: {patch_path}")
        return

    if command == "full-run":
        paths = run_full(output_dir=args.output_dir)
        _print_paths("FDR diagnostics complete.", paths, as_json=args.json)
        return

    raise ValueError(f"Unknown diagnostics command: {command}")


def _print_paths(title: str, paths: dict[str, Path], as_json: bool = False) -> None:
    if as_json:
        graph = build_sarah_vulnerability_graph()
        print(json.dumps({"summary": graph_summary(graph), "outputs": {key: str(value) for key, value in paths.items()}}, indent=2))
        return
    print(title)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


def _normalize_label(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


if __name__ == "__main__":
    main()
