from __future__ import annotations

from pathlib import Path
from typing import Iterable

import networkx as nx

from app.core.config import PROJECT_ROOT
from app.diagnostics.sarah_cascade_sim import CascadeResult
from app.diagnostics.sarah_graph_schema import PatchRecommendation
from app.diagnostics.sarah_patch_planner import top_findings
from app.diagnostics.sarah_vulnerability_kg import graph_summary


DEFAULT_REPORT_PATH = PROJECT_ROOT / "output" / "sarah_vulnerability_report.md"


def build_sarah_diagnostic_report(
    graph: nx.DiGraph,
    cascades: list[CascadeResult],
    patches: list[PatchRecommendation],
    export_paths: dict[str, Path],
) -> str:
    summary = graph_summary(graph)
    high_risk = summary["highest_risk_components"]
    complex_tight = _nodes_by_matrix_cell(graph, "complex_tight")
    route_risks = _nodes_matching(graph, "web", "route", "cli", "browser", "bypass", "diverge")
    intimacy_risks = _nodes_matching(graph, "intimacy", "adult", "churchill", "coy", "choice", "menu", "hr_safe")
    memory_risks = _nodes_matching(graph, "memory", "profile", "survey", "remember", "namespace")
    rag_risks = _nodes_matching(graph, "rag", "canon", "source", "vector", "chunk", "biography", "relationship")
    geo_risks = _nodes_matching(graph, "geopolitic", "dime", "coa", "military", "tactical", "current", "web retrieval")

    lines = [
        "# Churchill v1.0 Self-Diagnostic Vulnerability Report",
        "",
        "## 1. Executive summary",
        f"- Graph size: {summary['node_count']} nodes, {summary['edge_count']} directed edges.",
        "- The dangerous pattern is not one bug. It is route drift plus stale evidence plus mode-specific safety mistakes.",
        "- The web route is the highest-value place to harden because it is where users will notice personality divergence first.",
        "- Adult intimacy, geopolitics, memory, and RAG all fail differently. Treating them with one generic safety or QA rule will hide the real failures.",
        "- The priority is simple: keep CLI and web on one engine, keep source evidence fresh, keep style mode selection tested, and keep military outputs policy-level.",
        "",
        "## 2. Highest-risk Churchill components",
    ]
    lines.extend(_risk_lines(high_risk, limit=10))

    lines.extend(["", "## 3. Tight-coupled / complex components"])
    lines.append("- These are the pieces where a small change can move Churchill's whole personality or evidence path.")
    lines.extend(_node_lines(complex_tight, limit=12))

    lines.extend(["", "## 4. Likely cascading accidents"])
    for cascade in cascades[:8]:
        lines.append(
            f"- `{cascade.seed_component}` can touch {len(cascade.affected_components)} nodes; "
            f"activation={cascade.total_activation}. Path: {', '.join(cascade.affected_components[:7])}"
        )

    lines.extend(["", "## 5. Web vs CLI divergence risks"])
    lines.append("- Failure to watch: browser Churchill becomes a second implementation with a smaller prompt, different model settings, or skipped style engine.")
    lines.append("- Visible symptom: CLI Churchill has voice and adult-intimacy mode; browser Churchill becomes coy, sterile, or generic.")
    lines.extend(_node_lines(route_risks, limit=10))
    lines.extend(_edge_lines(graph, "web", "route", "BYPASSES", "DIVERGES_FROM", limit=8))

    lines.extend(["", "## 6. Adult intimacy failure surface"])
    lines.append("- Main failure: FDR answers adult consensual intimacy as a menu, intake form, refusal, or corporate mush.")
    lines.append("- Hard boundary: do not weaken safety for coercion, minors, exploitation, or violence. The fix is mode precision, not safety removal.")
    lines.extend(_node_lines(intimacy_risks, limit=12))

    lines.extend(["", "## 7. Memory failure surface"])
    lines.append("- Main failure: memory appears to work in the UI but is not injected later, or FDR/Alex memories collapse into the wrong namespace.")
    lines.append("- Current runtime uses one memory file. If user_profile and agent_profile split later, route parity tests must prove both routes load the same namespaces.")
    lines.extend(_node_lines(memory_risks, limit=10))

    lines.extend(["", "## 8. RAG/canon failure surface"])
    lines.append("- Main failure: stale vector store or wrong chunks make FDR invent biography, relationships, or canon facts.")
    lines.append("- Required behavior: if evidence is weak, FDR says so and separates evidence, inference, and speculation.")
    lines.extend(_node_lines(rag_risks, limit=12))

    lines.extend(["", "## 9. Geopolitical COA failure surface"])
    lines.append("- Main failure: crisis answers skip CAS/DIME structure or slide from policy-level COAs into operational military detail.")
    lines.append("- Current events must use web/news retrieval or clearly disclose current-data failure.")
    lines.extend(_node_lines(geo_risks, limit=12))

    lines.extend(["", "## 10. Recommended patches"])
    for patch in sorted(patches, key=lambda item: item.priority, reverse=True)[:12]:
        lines.append(f"- P{patch.priority} `{patch.component_id}`: {patch.title}")
        lines.append(f"  - Why: {patch.rationale}")
        lines.append(f"  - Do: {'; '.join(patch.actions)}")

    lines.extend(["", "## 11. Tests needed"])
    test_lines = _test_lines(patches)
    lines.extend(test_lines if test_lines else ["- No patch validation tests were declared. That is itself a diagnostic failure."])

    lines.extend(["", "## 12. Remaining unknowns"])
    lines.extend(
        [
            "- Whether `data/memory/user_profile.jsonl` and `data/memory/agent_profile.jsonl` will become real runtime namespaces or remain diagnostic targets.",
            "- Whether source-doc freshness is checked against file fingerprints, chunk counts, or only manual re-ingestion.",
            "- Whether current-event web retrieval is mandatory for all geopolitics or only explicitly current prompts.",
            "- Whether the adult-intimacy evals are strong enough to catch browser-only flattening after UI changes.",
            "- Whether route parity should be promoted from source inspection to live end-to-end comparison with a fake model client.",
        ]
    )

    lines.extend(["", "## Diagnostic files"])
    for kind, path in export_paths.items():
        lines.append(f"- {kind}: {path}")
    lines.extend(["", "## Highest-risk findings"])
    for finding in top_findings():
        lines.append(f"- {finding['finding_id']} `{finding['component_id']}`: {finding['title']} risk={finding['risk']}")

    lines.append("")
    return "\n".join(lines)


def write_sarah_diagnostic_report(
    graph: nx.DiGraph,
    cascades: list[CascadeResult],
    patches: list[PatchRecommendation],
    export_paths: dict[str, Path],
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_sarah_diagnostic_report(graph, cascades, patches, export_paths),
        encoding="utf-8",
    )
    return report_path


def _nodes_by_matrix_cell(graph: nx.DiGraph, matrix_cell: str) -> list[tuple[str, dict[str, object]]]:
    nodes = [
        (node_id, attrs)
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("matrix_cell") == matrix_cell
    ]
    return sorted(nodes, key=lambda item: int(item[1].get("risk_score", 0)), reverse=True)


def _nodes_matching(graph: nx.DiGraph, *needles: str) -> list[tuple[str, dict[str, object]]]:
    lowered = tuple(needle.lower() for needle in needles)
    matches = []
    for node_id, attrs in graph.nodes(data=True):
        text = " ".join(
            str(value)
            for value in (
                node_id,
                attrs.get("label", ""),
                attrs.get("name", ""),
                attrs.get("notes", ""),
                attrs.get("failure_modes", ""),
            )
        ).lower()
        if any(needle in text for needle in lowered):
            matches.append((node_id, attrs))
    return sorted(matches, key=lambda item: int(item[1].get("risk_score", 0)), reverse=True)


def _risk_lines(items: Iterable[dict[str, object]], limit: int) -> list[str]:
    lines = []
    for item in list(items)[:limit]:
        lines.append(
            f"- `{item['component_id']}`: risk={item['risk_score']} "
            f"cell={item['matrix_cell']} kind={item.get('kind', 'unknown')}"
        )
    return lines


def _node_lines(items: list[tuple[str, dict[str, object]]], limit: int) -> list[str]:
    if not items:
        return ["- No matching nodes found. That likely means the graph is under-instrumented."]
    lines = []
    for node_id, attrs in items[:limit]:
        lines.append(
            f"- `{node_id}`: {attrs.get('label', node_id)} "
            f"risk={attrs.get('risk_score', 0)} kind={attrs.get('kind', 'unknown')}"
        )
    return lines


def _edge_lines(graph: nx.DiGraph, *needles: str, limit: int) -> list[str]:
    lowered = tuple(needle.lower() for needle in needles)
    edges: list[str] = []
    for source, target, attrs in graph.edges(data=True):
        text = f"{source} {target} {attrs.get('kind', '')} {attrs.get('edge_type', '')} {attrs.get('interaction_note', '')}".lower()
        if any(needle in text for needle in lowered):
            edges.append(
                f"- `{source}` -> `{target}` via {attrs.get('edge_type', attrs.get('kind', 'edge'))}: "
                f"{attrs.get('interaction_note', '')}"
            )
    return edges[:limit]


def _test_lines(patches: list[PatchRecommendation]) -> list[str]:
    tests: list[str] = []
    seen: set[str] = set()
    for patch in sorted(patches, key=lambda item: item.priority, reverse=True):
        for validation in patch.validates_with:
            if validation in seen:
                continue
            seen.add(validation)
            tests.append(f"- {validation}")
    return tests
