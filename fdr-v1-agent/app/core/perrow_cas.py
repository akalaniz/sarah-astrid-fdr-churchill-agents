from __future__ import annotations

import re
from typing import Any


PERROW_EVENT_TRIGGERS: tuple[str, ...] = (
    "solar flare",
    "geomagnetic storm",
    "grid failure",
    "blackout",
    "power grid",
    "nuclear plant",
    "nuclear accident",
    "reactor accident",
    "aircraft accident",
    "plane crash",
    "aviation accident",
    "ai failure",
    "model failure",
    "automation failure",
    "military early warning",
    "early warning",
    "hospital collapse",
    "hospital failure",
    "market crash",
    "flash crash",
    "bank run",
    "supply chain collapse",
    "normal accident",
    "perrow",
    "tight coupling",
    "loose coupling",
    "complex interaction",
    "cascading failure",
    "operator overload",
    "control saturation",
    "institutional lag",
)

FULL_DEPTH_TRIGGERS: tuple[str, ...] = (
    "full report",
    "exhaustive",
    "deep dive",
    "long form",
    "maximum detail",
)

STANDARD_DEPTH_TRIGGERS: tuple[str, ...] = (
    "standard",
    "more detail",
    "walk me through",
)


def is_perrow_cas_event(user_query: str) -> bool:
    text = user_query.lower()
    return any(trigger in text for trigger in PERROW_EVENT_TRIGGERS) or bool(
        re.search(r"\b(accident|collapse|crash|outage|failure)\b", text)
        and re.search(r"\b(system|grid|plant|aircraft|hospital|market|ai|warning|network)\b", text)
    )


def infer_perrow_depth(user_query: str) -> str:
    text = user_query.lower()
    if any(trigger in text for trigger in FULL_DEPTH_TRIGGERS):
        return "full"
    if any(trigger in text for trigger in STANDARD_DEPTH_TRIGGERS):
        return "standard"
    return "brief"


def analyze_perrow_cas_event(user_query: str) -> dict[str, Any]:
    event_type = _classify_event(user_query)
    depth = infer_perrow_depth(user_query)
    coupling = _coupling_assessment(event_type, user_query)
    complexity = _interaction_complexity_assessment(event_type, user_query)
    quadrant = _quadrant(coupling["score"], complexity["score"])
    frame = {
        "perrow_depth": depth,
        "event_summary": _event_summary(user_query, event_type),
        "system_boundary": _system_boundary(event_type),
        "key_components": _key_components(event_type),
        "coupling_assessment": coupling,
        "interaction_complexity_assessment": complexity,
        "perrow_quadrant": quadrant,
        "initiating_events": _initiating_events(event_type),
        "cascade_timeline": _cascade_timeline(event_type),
        "hidden_couplings": _hidden_couplings(event_type),
        "controls_that_might_fail": _controls_that_might_fail(event_type),
        "observability_gaps": _observability_gaps(event_type),
        "second_order_effects": _second_order_effects(event_type),
        "worst_case_pathways": _worst_case_pathways(event_type),
        "stabilizing_interventions": _stabilizing_interventions(event_type),
        "patch_recommendations": _patch_recommendations(event_type),
        "uncertainty_notes": _uncertainty_notes(user_query),
    }
    return _apply_depth_limits(frame)


def build_perrow_cas_prompt_section(frame: dict[str, Any]) -> str:
    depth = frame.get("perrow_depth", "brief")
    if depth == "full":
        answer_shape = (
            "Full depth requested. Use all relevant fields, but still answer in FDR's voice. "
            "Do not become a generic report generator."
        )
    elif depth == "standard":
        answer_shape = (
            "Standard depth requested. Use the brief sections with moderate detail. Keep lists compact."
        )
    else:
        answer_shape = (
            "Brief depth is default. Use exactly these sections unless Alex asks for more: "
            "1. Situation compression. 2. Perrow placement. 3. Cascade timeline, max 5 steps. "
            "4. Hidden couplings, max 5 bullets. 5. Patch recommendations, max 5 bullets. "
            "6. What to monitor, max 5 bullets. In Perrow placement, explicitly name tight coupling "
            "or loose coupling, and complex interactions or linear interactions."
        )
    lines = [
        "PERROW/CAS ACCIDENT ANALYSIS FRAME:",
        "Use this internally when answering external system-failure questions. Do not dump every field unless Alex explicitly asks for full report, exhaustive, deep dive, long form, or maximum detail.",
        f"perrow_depth: {depth}",
        answer_shape,
        f"event_summary: {frame['event_summary']}",
        f"system_boundary: {frame['system_boundary']}",
        f"perrow_quadrant: {frame['perrow_quadrant']}",
        f"coupling_assessment: {frame['coupling_assessment']}",
        f"interaction_complexity_assessment: {frame['interaction_complexity_assessment']}",
        f"key_components: {frame['key_components']}",
        f"initiating_events: {frame['initiating_events']}",
        f"cascade_timeline: {frame['cascade_timeline']}",
        f"hidden_couplings: {frame['hidden_couplings']}",
        f"controls_that_might_fail: {frame['controls_that_might_fail']}",
        f"observability_gaps: {frame['observability_gaps']}",
        f"patch_recommendations: {frame['patch_recommendations']}",
        f"what_to_monitor: {frame['what_to_monitor']}",
    ]
    if depth == "full":
        lines.extend(
            [
                f"second_order_effects: {frame['second_order_effects']}",
                f"worst_case_pathways: {frame['worst_case_pathways']}",
                f"stabilizing_interventions: {frame['stabilizing_interventions']}",
            ]
        )
    lines.extend(
        [
            f"uncertainty_notes: {frame['uncertainty_notes']}",
            "Answer in FDR's voice: crisp, systems-literate, morally awake. Emphasize stabilization and patching, not exploitation.",
        ]
    )
    return "\n".join(lines)


def _classify_event(user_query: str) -> str:
    text = user_query.lower()
    event_map = (
        ("solar_grid", ("solar flare", "geomagnetic", "grid", "blackout", "power")),
        ("nuclear_plant", ("nuclear plant", "reactor", "nuclear accident", "meltdown")),
        ("aviation", ("aircraft", "plane", "aviation", "airline", "flight")),
        ("ai_system", ("ai failure", "model failure", "automation", "agi", "ai system")),
        ("early_warning", ("early warning", "missile warning", "radar warning", "military warning")),
        ("hospital", ("hospital", "icu", "medical system", "triage")),
        ("market", ("market", "flash crash", "bank run", "liquidity", "exchange")),
    )
    for event_type, triggers in event_map:
        if any(trigger in text for trigger in triggers):
            return event_type
    return "generic_complex_system"


def _event_summary(user_query: str, event_type: str) -> str:
    return f"Analyze '{user_query.strip()}' as a {event_type.replace('_', ' ')} failure in a coupled adaptive system."


def _system_boundary(event_type: str) -> str:
    boundaries = {
        "solar_grid": "Space weather source, power grid operators, power transmission network, transformers, generation dispatch, telecom timing, fuel logistics, emergency management.",
        "nuclear_plant": "Plant hardware, operators, safety systems, grid connection, cooling supply, regulator, emergency response, public communication.",
        "aviation": "Aircraft, crew, automation, maintenance, dispatch, weather, air traffic control, airline operations, regulator.",
        "ai_system": "Model, tool layer, data pipeline, operators, monitoring, downstream users, policy controls, incident response.",
        "early_warning": "Sensors, fusion systems, command centers, communications, political decision loops, adversary signals, doctrine.",
        "hospital": "Staffing, beds, ICU capacity, oxygen, pharmacy, labs, IT systems, ambulances, regional load balancing.",
        "market": "Exchanges, liquidity providers, clearing, leverage, margin calls, news flows, algorithms, regulators, payment rails.",
    }
    return boundaries.get(event_type, "Define the operational system, its control loops, dependencies, operators, institutions, and exposed public consequences.")


def _key_components(event_type: str) -> list[str]:
    common = ["operators", "automation", "control room", "communications", "institutional decision layer", "public consequence layer"]
    specific = {
        "solar_grid": [
            "high-voltage transformers",
            "SCADA",
            "protective relays",
            "generation dispatch",
            "telecom timing",
            "satellites",
            "GPS",
            "communications",
            "aviation",
            "finance",
            "emergency services",
        ],
        "nuclear_plant": ["reactor core", "cooling systems", "backup power", "containment", "radiation monitoring"],
        "aviation": ["flight crew", "flight management system", "sensors", "maintenance records", "ATC"],
        "ai_system": ["model", "retrieval/tools", "policy layer", "monitoring", "human escalation path"],
        "early_warning": ["radars/satellites", "sensor fusion", "alert thresholds", "command authority", "hotlines"],
        "hospital": ["triage", "ICU", "staffing", "oxygen", "EHR/IT", "ambulance routing"],
        "market": ["matching engine", "liquidity", "leverage", "margin", "clearing", "circuit breakers"],
    }
    return specific.get(event_type, []) + common


def _coupling_assessment(event_type: str, user_query: str) -> dict[str, Any]:
    tight_types = {"solar_grid", "nuclear_plant", "aviation", "early_warning", "hospital", "market"}
    score = 8 if event_type in tight_types else 6
    if any(term in user_query.lower() for term in ("real time", "seconds", "minutes", "automatic", "synchronous")):
        score = min(10, score + 1)
    return {
        "score": score,
        "label": "tight" if score >= 7 else "mixed/loose",
        "rationale": "Failures propagate faster than institutions can deliberate." if score >= 7 else "Some buffers exist, but shared dependencies can still synchronize failures.",
    }


def _interaction_complexity_assessment(event_type: str, user_query: str) -> dict[str, Any]:
    complex_types = {"solar_grid", "ai_system", "early_warning", "hospital", "market"}
    score = 8 if event_type in complex_types else 7
    if any(term in user_query.lower() for term in ("hidden", "unknown", "cascade", "feedback", "interdependent")):
        score = min(10, score + 1)
    return {
        "score": score,
        "label": "complex" if score >= 7 else "linear",
        "rationale": "Operators see symptoms through partial instruments while feedback loops alter the system state.",
    }


def _quadrant(coupling_score: int, complexity_score: int) -> str:
    interaction = "complex" if complexity_score >= 7 else "linear"
    coupling = "tight" if coupling_score >= 7 else "loose"
    return f"{interaction}_{coupling}"


def _initiating_events(event_type: str) -> list[str]:
    base = ["ordinary disturbance", "bad sensor or missing signal", "operator workload spike", "maintenance/configuration mismatch"]
    specific = {
        "solar_grid": ["coronal mass ejection", "satellite degradation", "GPS timing disruption", "transformer heating", "protective relay trips"],
        "nuclear_plant": ["loss of offsite power", "cooling degradation", "instrumentation ambiguity"],
        "aviation": ["sensor disagreement", "automation mode confusion", "weather or maintenance stressor"],
        "ai_system": ["bad retrieval/tool call", "distribution shift", "monitoring blind spot"],
        "early_warning": ["false alarm", "sensor spoof/ambiguity", "communications delay"],
        "hospital": ["patient surge", "staff shortage", "IT outage"],
        "market": ["liquidity shock", "margin spiral", "algorithmic selling"],
    }
    return specific.get(event_type, []) + base


def _cascade_timeline(event_type: str) -> list[dict[str, str]]:
    return [
        {"time": "t0", "phase": "disturbance", "description": "A trigger enters through a component that appears local."},
        {"time": "t1", "phase": "coupling", "description": "Tight dependencies move the disturbance into adjacent subsystems."},
        {"time": "t2", "phase": "confusion", "description": "Signals conflict; operators see fragments, not the whole phase space."},
        {"time": "t3", "phase": "saturation", "description": "Control channels saturate and workarounds become new failure paths."},
        {"time": "t4", "phase": "institutional lag", "description": "Formal authority catches up after the system has already changed state."},
        {"time": "t5", "phase": "recovery or collapse", "description": "Buffers, isolation, and clear authority determine whether the cascade arrests."},
    ]


def _what_to_monitor(event_type: str) -> list[str]:
    if event_type == "solar_grid":
        return [
            "solar wind speed and magnetic orientation",
            "satellite anomalies and GPS timing drift",
            "transformer heating and reactive power stress",
            "protective relay trips and control-room alarm load",
            "fuel, hospital backup power, water-system pump continuity, and emergency communications",
        ]
    return [
        "coupling strength and queue depth",
        "operator workload and alarm volume",
        "sensor disagreement and delayed telemetry",
        "backup-system dependency health",
        "institutional decision lag",
    ]


def _hidden_couplings(event_type: str) -> list[str]:
    hidden = {
        "solar_grid": [
            "grid depends on telecom timing; telecom depends on grid power",
            "GPS timing couples satellites, finance, aviation navigation, and grid synchronization",
            "emergency services depend on communications and fuel logistics during the same outage",
            "spare transformer logistics lag restoration",
        ],
        "nuclear_plant": ["safety systems depend on power, cooling, and human interpretation at once"],
        "aviation": ["automation, training, checklists, dispatch pressure, and sensor health interact nonlinearly"],
        "ai_system": ["model confidence, tool permissions, data quality, and user trust couple silently"],
        "early_warning": ["technical ambiguity couples directly into political decision time"],
        "hospital": ["bed capacity, staff fatigue, ambulance diversion, and IT outages amplify each other"],
        "market": ["liquidity, leverage, collateral values, and algorithmic triggers synchronize under stress"],
    }
    return hidden.get(event_type, ["shared vendors", "communication dependencies", "human procedures that assume normal load"])


def _controls_that_might_fail(event_type: str) -> list[str]:
    return [
        "alarms that create noise instead of clarity",
        "dashboards that lag the real system state",
        "manual override paths that are too slow or unclear",
        "institutional escalation rules that assume time is available",
        "backup systems sharing the same hidden dependency as primary systems",
    ]


def _observability_gaps(event_type: str) -> list[str]:
    return [
        "operators cannot directly observe coupling strength",
        "near-misses are not visible across institutions",
        "local metrics look normal while global risk accumulates",
        "delayed feedback makes the first corrective action look ineffective or harmless",
    ]


def _second_order_effects(event_type: str) -> list[str]:
    return [
        "public trust damage",
        "regulatory overcorrection or paralysis",
        "supply-chain and staffing aftershocks",
        "misallocation of attention toward visible symptoms instead of structural coupling",
    ]


def _worst_case_pathways(event_type: str) -> list[str]:
    return [
        "false diagnosis leads to the wrong stabilizing action",
        "backup systems fail because they share the same dependency",
        "operators become overloaded and automation becomes opaque",
        "institutional lag lets a recoverable disturbance become a regime change",
    ]


def _stabilizing_interventions(event_type: str) -> list[str]:
    return [
        "slow the system where possible; add deliberate decision time",
        "isolate failing components before they synchronize the whole network",
        "reduce alarm noise and surface the few state variables that matter",
        "pre-delegate authority for known high-tempo failure modes",
        "communicate uncertainty plainly to prevent rumor cascades",
    ]


def _patch_recommendations(event_type: str) -> list[str]:
    if event_type == "solar_grid":
        return [
            "patch: harden high-voltage transformers against geomagnetically induced current stress",
            "patch: design grid islanding and staged load shedding before the storm arrives",
            "patch: run black-start drills under degraded communications and bad telemetry",
            "patch: deploy GPS-independent timing alternatives for grid, telecom, aviation, finance, and emergency services",
            "patch: preserve manual fallback procedures and train operators to use them under alarm overload",
            "patch: stock spare transformers and pre-plan transport, fuel, and installation logistics",
        ]
    return [
        "map hidden couplings and shared dependencies before the accident",
        "test backup paths under degraded conditions, not just nominal drills",
        "add observability for coupling, queue depth, operator workload, and delayed feedback",
        "create hard circuit breakers where tight coupling is unnecessary",
        "practice cross-institution handoff while the system is noisy and partially blind",
    ]


def _uncertainty_notes(user_query: str) -> list[str]:
    return [
        "This is a structural accident-analysis frame, not a factual incident report.",
        "Specific claims require current sources, logs, telemetry, official findings, or local evidence.",
        f"User query scope may be narrower than the inferred system boundary: {user_query.strip()}",
    ]


def _apply_depth_limits(frame: dict[str, Any]) -> dict[str, Any]:
    depth = frame["perrow_depth"]
    event_type = _classify_event(str(frame["event_summary"]))
    if depth == "full":
        frame["what_to_monitor"] = _what_to_monitor(event_type)
        return frame

    if depth == "standard":
        limit = 7
    else:
        limit = 5
    limited = dict(frame)
    for key in (
        "cascade_timeline",
        "hidden_couplings",
        "controls_that_might_fail",
        "observability_gaps",
        "second_order_effects",
        "worst_case_pathways",
        "stabilizing_interventions",
        "patch_recommendations",
        "uncertainty_notes",
    ):
        value = limited.get(key)
        if isinstance(value, list):
            limited[key] = value[:limit]
    limited["what_to_monitor"] = _what_to_monitor(event_type)[:limit]
    return limited
