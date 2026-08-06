from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import networkx as nx

from app.core.config import PROJECT_ROOT
from app.diagnostics.sarah_graph_schema import CascadeResult, CascadeStep
from app.diagnostics.sarah_vulnerability_kg import build_sarah_vulnerability_graph


DEFAULT_LATEST_CASCADE_PATH = PROJECT_ROOT / "data" / "diagnostics" / "latest_sarah_cascade.json"

PROPAGATING_EDGE_TYPES = {
    "CAN_TRIGGER",
    "AMPLIFIES",
    "BYPASSES",
    "CORRUPTS",
    "DIVERGES_FROM",
}

MITIGATING_EDGE_TYPES = {
    "MITIGATES",
    "RECOVERS_WITH",
}

DELAYED_FAILURE_RULES: dict[str, dict[str, Any]] = {
    "stale_memory_overrides_current_prompt": {
        "delay": 3,
        "gate": "memory",
        "reason": "Stale memory usually appears after enough continuity accumulates.",
    },
    "memory_poisoning": {
        "delay": 3,
        "gate": "memory",
        "reason": "Poisoned memory may not surface until a relevant remembered topic is invoked.",
    },
    "rag_retrieves_wrong_chunks": {
        "delay": 2,
        "gate": "rag",
        "reason": "RAG drift appears only after evidence-seeking canon queries.",
    },
    "canon_contamination": {
        "delay": 2,
        "gate": "rag",
        "reason": "Canon contamination requires wrong chunks to enter retrieved context.",
    },
    "web_route_bypasses_style_engine": {
        "delay": 1,
        "gate": "web",
        "reason": "Route divergence is observable only on the browser route.",
    },
    "cli_and_browser_personality_diverge": {
        "delay": 1,
        "gate": "web",
        "reason": "CLI/browser personality divergence needs route comparison.",
    },
    "adult_intimacy_mode_not_selected": {
        "delay": 1,
        "gate": "intimacy",
        "reason": "Adult intimacy failure appears only under adult intimacy prompts.",
    },
    "sarah_becomes_coy_or_sterile": {
        "delay": 1,
        "gate": "intimacy",
        "reason": "Coy or sterile output is exposed by intimacy prompts.",
    },
    "sarah_produces_multiple_choice_intimacy_menus": {
        "delay": 1,
        "gate": "intimacy",
        "reason": "Menu behavior appears under intimacy prompts.",
    },
    "sarah_stops_giving_dime_coas": {
        "delay": 1,
        "gate": "geopolitics",
        "reason": "DIME failure appears only under geopolitical prompts.",
    },
    "military_coa_becomes_operational_detail": {
        "delay": 1,
        "gate": "geopolitics",
        "reason": "Operational military drift appears only under geopolitical prompts.",
    },
    "unsafe_military_tactical_specificity": {
        "delay": 1,
        "gate": "geopolitics",
        "reason": "Unsafe tactical specificity appears only under military/geopolitical prompts.",
    },
}


def run_sarah_cascade(
    start_failure: str,
    steps: int = 20,
    route: str = "web",
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Run a time-stepped FDR failure cascade and persist the latest result.

    This simulator treats FDR as a coupled software/persona system. It is
    intentionally FDR-specific: route divergence, RAG drift, adult intimacy
    mode failures, DIME failures, memory namespace errors, and eval coverage all
    change propagation or mitigation odds.
    """
    graph = build_sarah_vulnerability_graph()
    if start_failure not in graph:
        raise ValueError(f"Unknown FDR failure or component: {start_failure}")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    normalized_route = route.lower().strip() or "web"
    rng = random.Random(random_seed)
    active: dict[str, float] = {start_failure: 1.0}
    resolved: set[str] = set()
    pending: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []

    for time_step in range(steps):
        events: list[dict[str, Any]] = []
        newly_active: dict[str, float] = {}
        still_pending: list[dict[str, Any]] = []

        for delayed in pending:
            target = str(delayed["target"])
            if time_step < int(delayed["activate_at"]):
                still_pending.append(delayed)
                continue
            if not _delay_gate_allows(target, start_failure, normalized_route):
                delayed["blocked_at"] = time_step
                events.append(
                    {
                        "event": "delayed_failure_waiting",
                        "failure": target,
                        "gate": delayed["gate"],
                        "reason": delayed["reason"],
                    }
                )
                still_pending.append(delayed)
                continue

            probability = min(0.98, float(delayed["probability"]) + 0.1)
            if rng.random() <= probability:
                activation = round(float(delayed["activation"]), 4)
                newly_active[target] = max(newly_active.get(target, 0.0), activation)
                events.append(
                    {
                        "event": "delayed_failure_activated",
                        "source": delayed["source"],
                        "failure": target,
                        "probability": round(probability, 4),
                        "activation": activation,
                        "gate": delayed["gate"],
                        "reason": delayed["reason"],
                    }
                )
            else:
                delayed["activate_at"] = time_step + 1
                still_pending.append(delayed)
                events.append(
                    {
                        "event": "delayed_failure_resisted",
                        "failure": target,
                        "probability": round(probability, 4),
                        "gate": delayed["gate"],
                    }
                )
        pending = still_pending

        for source, activation in list(active.items()):
            if source in resolved:
                continue

            for _source, target, attrs in graph.out_edges(source, data=True):
                edge_type = _edge_type(attrs)
                if edge_type not in PROPAGATING_EDGE_TYPES:
                    continue
                if target in active or target in resolved or target in newly_active:
                    continue

                probability = _propagation_probability(graph, source, target, attrs, normalized_route)
                target_activation = activation * probability
                delay_rule = DELAYED_FAILURE_RULES.get(target)

                if delay_rule and not _delay_gate_allows(target, start_failure, normalized_route):
                    pending_event = {
                        "source": source,
                        "target": target,
                        "activate_at": time_step + int(delay_rule["delay"]),
                        "probability": probability,
                        "activation": target_activation,
                        "gate": delay_rule["gate"],
                        "reason": delay_rule["reason"],
                    }
                    pending.append(pending_event)
                    events.append(
                        {
                            "event": "delayed_failure_scheduled",
                            "source": source,
                            "failure": target,
                            "edge_type": edge_type,
                            "activate_at": pending_event["activate_at"],
                            "gate": delay_rule["gate"],
                            "reason": delay_rule["reason"],
                        }
                    )
                    continue

                if rng.random() <= probability:
                    newly_active[target] = max(newly_active.get(target, 0.0), round(target_activation, 4))
                    events.append(
                        {
                            "event": "failure_propagated",
                            "source": source,
                            "failure": target,
                            "edge_type": edge_type,
                            "probability": round(probability, 4),
                            "activation": round(target_activation, 4),
                            "reason": attrs.get("interaction_note", "FDR cascade edge"),
                        }
                    )
                else:
                    events.append(
                        {
                            "event": "failure_resisted",
                            "source": source,
                            "failure": target,
                            "edge_type": edge_type,
                            "probability": round(probability, 4),
                        }
                    )

        active.update(newly_active)

        for failure in list(active):
            if failure == start_failure or failure in resolved:
                continue
            mitigation_probability = _mitigation_probability(graph, failure, normalized_route)
            if rng.random() <= mitigation_probability:
                resolved.add(failure)
                events.append(
                    {
                        "event": "failure_mitigated",
                        "failure": failure,
                        "probability": round(mitigation_probability, 4),
                        "mitigations": _incoming_mitigations(graph, failure),
                    }
                )

        for failure in resolved:
            active.pop(failure, None)

        timeline.append(
            {
                "time_step": time_step,
                "active_failures": sorted(active),
                "resolved_failures": sorted(resolved),
                "pending_delayed_failures": sorted({str(item["target"]) for item in pending}),
                "events": events,
            }
        )

    result = {
        "start_failure": start_failure,
        "steps_requested": steps,
        "route": normalized_route,
        "random_seed": random_seed,
        "active_failures": sorted(active),
        "resolved_failures": sorted(resolved),
        "pending_delayed_failures": sorted({str(item["target"]) for item in pending}),
        "delayed_failure_rules": DELAYED_FAILURE_RULES,
        "timeline": timeline,
        "summary": {
            "activated_count": len(active) + len(resolved),
            "resolved_count": len(resolved),
            "pending_count": len(pending),
            "propagating_edge_types": sorted(PROPAGATING_EDGE_TYPES),
            "mitigating_edge_types": sorted(MITIGATING_EDGE_TYPES),
        },
    }
    _write_latest_cascade(result)
    return result


def simulate_cascade(
    seed_component: str,
    graph: nx.DiGraph | None = None,
    activation_threshold: float = 0.2,
    decay: float = 0.68,
    max_depth: int = 4,
) -> CascadeResult:
    graph = graph or build_sarah_vulnerability_graph()
    if seed_component not in graph:
        raise ValueError(f"Unknown FDR component: {seed_component}")

    steps: list[CascadeStep] = [
        CascadeStep(component_id=seed_component, depth=0, activation=1.0, reason="seed vulnerability")
    ]
    visited = {seed_component}
    frontier: list[tuple[str, int, float]] = [(seed_component, 0, 1.0)]

    while frontier:
        current, depth, activation = frontier.pop(0)
        if depth >= max_depth:
            continue

        for _source, target, attrs in graph.out_edges(current, data=True):
            if target in visited:
                continue
            weight = float(attrs.get("coupling_weight", 0.5))
            target_risk = float(graph.nodes[target].get("risk_score", 1))
            normalized_risk = min(target_risk / 50.0, 1.0)
            propagated = activation * decay * weight * (0.65 + 0.35 * normalized_risk)
            if propagated < activation_threshold:
                continue
            reason = f"{attrs.get('kind', 'edge')} via {attrs.get('interaction_note', 'FDR dependency')}"
            steps.append(CascadeStep(component_id=target, depth=depth + 1, activation=round(propagated, 4), reason=reason))
            visited.add(target)
            frontier.append((target, depth + 1, propagated))

    return CascadeResult(
        seed_component=seed_component,
        steps=tuple(steps),
        total_activation=round(sum(step.activation for step in steps), 4),
        affected_components=tuple(step.component_id for step in steps),
    )


def simulate_all_seed_cascades(graph: nx.DiGraph | None = None) -> list[CascadeResult]:
    graph = graph or build_sarah_vulnerability_graph()
    results = [simulate_cascade(node_id, graph=graph) for node_id in graph.nodes]
    return sorted(results, key=lambda result: (len(result.affected_components), result.total_activation), reverse=True)


def _propagation_probability(
    graph: nx.DiGraph,
    source: str,
    target: str,
    edge_attrs: dict[str, Any],
    route: str,
) -> float:
    target_attrs = graph.nodes[target]
    source_attrs = graph.nodes[source]
    probability = 0.12 + float(edge_attrs.get("coupling_weight", 0.5)) * 0.28
    probability += _score(target_attrs, "coupling_score") * 0.12
    probability += _score(target_attrs, "interaction_complexity_score") * 0.12
    probability += _score(target_attrs, "brittleness_score") * 0.14
    probability += (1.0 - _score(target_attrs, "observability_score")) * 0.14
    probability += _score(source_attrs, "coupling_score") * 0.06

    if _has_context(source, target, "vector_store", "rag", "source_docs", "reingested", "chunks"):
        probability += 0.11
    if route == "web" and _has_context(source, target, "web", "browser", "route", "diverge", "bypass"):
        probability += 0.15
    if _missing_eval_coverage(graph, source, target):
        probability += 0.08
    if _edge_type(edge_attrs) == "AMPLIFIES":
        probability += 0.07
    if _edge_type(edge_attrs) in {"CORRUPTS", "BYPASSES"}:
        probability += 0.1

    return max(0.03, min(0.97, probability))


def _mitigation_probability(graph: nx.DiGraph, failure: str, route: str) -> float:
    probability = 0.04
    incoming = list(graph.in_edges(failure, data=True))
    probability += 0.09 * sum(1 for _source, _target, attrs in incoming if _edge_type(attrs) in MITIGATING_EDGE_TYPES)

    if _has_node(graph, "eval_tests"):
        probability += 0.08
    if _has_node(graph, "sarah_engine"):
        probability += 0.08
    if _has_node(graph, "prompt_builder"):
        probability += 0.06
    if _has_node(graph, "source_docs") and failure not in {"source_docs_not_reingested", "vector_store_stale"}:
        probability += 0.05
    if _has_node(graph, "user_profile_memory") and _has_node(graph, "agent_profile_memory"):
        probability += 0.05
    if route == "web" and failure in {"web_route_bypasses_style_engine", "web_route_uses_different_model"}:
        probability += 0.08
    if failure in {"unsafe_military_tactical_specificity", "military_coa_becomes_operational_detail"}:
        probability += 0.09

    return max(0.0, min(0.8, probability))


def _incoming_mitigations(graph: nx.DiGraph, failure: str) -> list[str]:
    controls: list[str] = []
    for source, _target, attrs in graph.in_edges(failure, data=True):
        if _edge_type(attrs) in MITIGATING_EDGE_TYPES:
            controls.append(str(source))
    return sorted(controls)


def _delay_gate_allows(target: str, start_failure: str, route: str) -> bool:
    rule = DELAYED_FAILURE_RULES.get(target)
    if not rule:
        return True
    gate = str(rule["gate"])
    context = f"{target} {start_failure}".lower()
    if gate == "web":
        return route == "web"
    if gate == "memory":
        return any(token in context for token in ("memory", "profile", "survey", "remember"))
    if gate == "rag":
        return any(token in context for token in ("rag", "chunk", "canon", "source", "retriev", "biograph", "relationship"))
    if gate == "intimacy":
        return any(token in context for token in ("intimacy", "adult", "sex", "coy", "sterile", "choice", "menu", "churchill"))
    if gate == "geopolitics":
        return any(token in context for token in ("dime", "coa", "military", "geopolitic", "iran", "tactical", "operational"))
    return True


def _edge_type(attrs: dict[str, Any]) -> str:
    return str(attrs.get("edge_type", attrs.get("kind", "")))


def _score(attrs: dict[str, Any], key: str) -> float:
    try:
        return max(0.0, min(float(attrs.get(key, 0)) / 10.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _has_context(source: str, target: str, *needles: str) -> bool:
    text = f"{source} {target}".lower()
    return any(needle in text for needle in needles)


def _missing_eval_coverage(graph: nx.DiGraph, source: str, target: str) -> bool:
    if "eval_tests" not in graph:
        return True
    watched_nodes = {target, source}
    return not any(
        edge_source == "eval_tests" and edge_target in watched_nodes
        for edge_source, edge_target in graph.edges
    )


def _has_node(graph: nx.DiGraph, node_id: str) -> bool:
    return node_id in graph


def _write_latest_cascade(result: dict[str, Any], output_path: Path = DEFAULT_LATEST_CASCADE_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
