from __future__ import annotations

import re
from typing import Any


GEOPOLITICAL_TERMS = {
    "geopolitics",
    "geopolitical",
    "strategy",
    "strategic",
    "crisis",
    "war",
    "deterrence",
    "escalation",
    "de-escalation",
    "sanctions",
    "nato",
    "un",
    "security council",
    "china",
    "russia",
    "ukraine",
    "iran",
    "israel",
    "taiwan",
    "north korea",
    "military",
    "alliance",
    "dime",
    "ooda",
    "cas",
    "phase transition",
}


COA_KEYS = {
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


def build_cas_geopolitical_frame(
    user_query: str,
    retrieved_context: list,
    web_context: list,
) -> dict:
    evidence_notes = _evidence_notes(retrieved_context, web_context)
    actors = _extract_actors(user_query, retrieved_context, web_context)

    return {
        "situation_compression": _situation_compression(user_query, evidence_notes),
        "actors": actors,
        "attractors": [
            "Status quo preservation by actors who benefit from the current equilibrium.",
            "Escalatory signaling when leaders need credibility, deterrence, or domestic legitimacy.",
            "Negotiated off-ramp if costs rise faster than perceived strategic gains.",
            "Fragmentation if institutions lose legitimacy or command narratives diverge.",
        ],
        "constraints": [
            "Alliance cohesion and domestic political tolerance.",
            "Economic exposure, energy dependence, market confidence, and sanctions fatigue.",
            "Information uncertainty, propaganda, and adversarial adaptation.",
            "Legal authorities, international law, and moral legitimacy.",
            "Operational reality is deliberately kept out of scope; analysis remains policy-level.",
        ],
        "possible_phase_transitions": [
            "Localized crisis becomes regional confrontation after a misread signal or collateral shock.",
            "Diplomatic bargaining becomes frozen conflict if actors settle into mutually tolerable pain.",
            "Economic pressure becomes political instability if legitimacy buffers fail.",
            "Information shock changes public tolerance and narrows leaders' degrees of freedom.",
        ],
        "feedback_loops": [
            "Threat signal -> adversary counter-signal -> public fear -> leadership pressure -> stronger signal.",
            "Sanctions -> adaptation/evasion attempts -> broader enforcement -> economic blowback.",
            "Civilian harm -> legitimacy loss -> recruitment/radicalization -> more instability.",
            "Media narrative -> market reaction -> policy pressure -> new media narrative.",
        ],
        "black_swan_triggers": [
            "Leadership death, coup attempt, or sudden succession crisis.",
            "Major accident, misattributed attack, or false warning.",
            "Unexpected financial contagion or infrastructure failure.",
            "Leaked intelligence or verified atrocity that changes public and elite incentives.",
            "A third-party actor forcing a crisis neither principal actor intended.",
        ],
        "degrees_of_freedom": [
            "Tempo of public messaging and private diplomatic channels.",
            "Breadth and reversibility of sanctions or economic measures.",
            "Alliance consultation, coalition design, and institutional venue selection.",
            "Readiness posture and deployment preparation at a policy level, without tactical execution.",
            "Humanitarian relief, civilian protection funding, and verification mechanisms.",
        ],
        "DIME_levers": {
            "diplomatic": [
                "Use allies, backchannels, and international organizations to create off-ramps.",
                "Define limited, verifiable demands instead of maximalist slogans.",
            ],
            "information": [
                "State evidence standards clearly and separate confirmed facts from inference.",
                "Prebunk likely disinformation without overclaiming certainty.",
            ],
            "military": [
                "Adjust readiness posture through lawful authorities.",
                "Prepare forces for deterrence, protection, evacuation, or humanitarian support at policy level.",
                "Coordinate with appropriate allies and civilian leadership.",
            ],
            "economic": [
                "Use targeted sanctions, export controls, financial restrictions, or relief packages.",
                "Build buffers for civilians and exposed allies before pressure measures bite.",
            ],
        },
        "COAs": _build_coas(),
    }


def is_geopolitical_or_strategic(user_query: str) -> bool:
    query = user_query.lower()
    return any(term in query for term in GEOPOLITICAL_TERMS)


def build_cas_frame_prompt_section(frame: dict) -> str:
    lines = [
        "CAS GEOPOLITICAL FRAME FOR INTERNAL USE:",
        "Use this structure to reason, then answer naturally in Churchill's voice. Do not dump every field unless Alex asks for the full frame.",
        "Military content must remain policy-level and must not include targeting, operational timing, weapons employment, evasion, or tactical execution details.",
        f"situation_compression: {frame['situation_compression']}",
        f"actors: {', '.join(frame['actors']) if frame['actors'] else 'uncertain'}",
        f"attractors: {'; '.join(frame['attractors'])}",
        f"constraints: {'; '.join(frame['constraints'])}",
        f"possible_phase_transitions: {'; '.join(frame['possible_phase_transitions'])}",
        f"feedback_loops: {'; '.join(frame['feedback_loops'])}",
        f"black_swan_triggers: {'; '.join(frame['black_swan_triggers'])}",
        f"degrees_of_freedom: {'; '.join(frame['degrees_of_freedom'])}",
        "DIME_levers:",
    ]
    dime = frame["DIME_levers"]
    for lever, actions in dime.items():
        lines.append(f"- {lever}: {'; '.join(actions)}")

    lines.append("COAs:")
    for coa in frame["COAs"]:
        lines.append(f"- {coa['name']}: intended_effect={coa['intended_effect']}; escalation_risk={coa['escalation_risk']}; moral_cost={coa['moral_cost']}")
    return "\n".join(lines)


def _situation_compression(user_query: str, evidence_notes: list[str]) -> str:
    if evidence_notes:
        return (
            f"Alex is asking for strategic/CAS analysis of: {user_query}. "
            f"Available context points to: {'; '.join(evidence_notes[:3])}."
        )
    return (
        f"Alex is asking for strategic/CAS analysis of: {user_query}. "
        "Evidence is limited; distinguish confirmed facts, inference, and speculation."
    )


def _extract_actors(user_query: str, retrieved_context: list, web_context: list) -> list[str]:
    text_parts = [user_query]
    text_parts.extend(str(item.get("source_filename", "")) for item in retrieved_context if isinstance(item, dict))
    text_parts.extend(str(item.get("publisher", "")) for item in web_context if isinstance(item, dict))
    text_parts.extend(str(item.get("title", "")) for item in web_context if isinstance(item, dict))
    text = " ".join(text_parts)

    candidates = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text)
    ignored = {"Alex", "FDR", "DIME", "OODA", "CAS"}
    actors: list[str] = []
    for candidate in candidates:
        if candidate in ignored or candidate in actors:
            continue
        actors.append(candidate)
        if len(actors) >= 8:
            break
    return actors


def _evidence_notes(retrieved_context: list, web_context: list) -> list[str]:
    notes: list[str] = []
    for item in web_context[:3]:
        if isinstance(item, dict):
            title = item.get("title")
            publisher = item.get("publisher")
            if title and publisher:
                notes.append(f"{publisher}: {title}")
    for item in retrieved_context[:2]:
        if isinstance(item, dict):
            source = item.get("source_filename")
            location = item.get("location")
            if source:
                notes.append(f"local source {source}" + (f" at {location}" if location else ""))
    return notes


def _build_coas() -> list[dict[str, Any]]:
    coas = [
        {
            "name": "COA 1: De-escalate and bound the crisis",
            "diplomatic_actions": [
                "Open or reinforce backchannels through trusted intermediaries.",
                "Seek limited, verifiable commitments rather than total settlement.",
                "Use international organizations to create procedural legitimacy.",
            ],
            "information_actions": [
                "Publish evidence carefully and avoid claims beyond confidence levels.",
                "Signal red lines and off-ramps in plain language.",
            ],
            "military_policy_level_actions": [
                "Review readiness posture through lawful civilian authority.",
                "Prepare defensive, evacuation, humanitarian, or force-protection options at policy level.",
                "Coordinate with appropriate allies without issuing tactical execution details.",
            ],
            "economic_actions": [
                "Offer reversible relief tied to verified behavior.",
                "Prepare targeted restrictions if actors reject off-ramps.",
            ],
            "intended_effect": "Reduce uncertainty and give actors a face-saving path away from escalation.",
            "escalation_risk": "Low to moderate; weak signals may be interpreted as lack of resolve.",
            "second_order_effects": "May buy time for adversarial adaptation or domestic criticism.",
            "failure_modes": "Backchannels leak, demands are too vague, or spoilers benefit from continued crisis.",
            "moral_cost": "May leave victims exposed longer while diplomacy works.",
            "indicators_to_watch": [
                "Tone of official statements.",
                "Third-party mediation activity.",
                "Civilian harm trends.",
                "Compliance with verifiable commitments.",
            ],
        },
        {
            "name": "COA 2: Contain and deter",
            "diplomatic_actions": [
                "Build a coalition around narrow shared interests.",
                "Coordinate alliance statements and legal justifications.",
            ],
            "information_actions": [
                "Expose destabilizing behavior with evidence and caveats.",
                "Prebunk likely disinformation narratives.",
            ],
            "military_policy_level_actions": [
                "Place relevant forces or bases on an appropriate readiness posture.",
                "Prepare naval, air, cyber-defense, evacuation, or humanitarian support options at policy level.",
                "Increase allied consultation and civilian oversight of any posture changes.",
            ],
            "economic_actions": [
                "Apply targeted sanctions, export controls, or financial restrictions.",
                "Protect exposed civilians and allies from economic blowback.",
            ],
            "intended_effect": "Raise the cost of destabilizing behavior while preserving room for negotiation.",
            "escalation_risk": "Moderate; deterrent signals can be misread as preparation for confrontation.",
            "second_order_effects": "Can harden blocs, increase propaganda value, or create market stress.",
            "failure_modes": "Coalition fractures, sanctions leak, or adversary adapts faster than pressure accumulates.",
            "moral_cost": "Economic pressure can hurt civilians even when designed to be targeted.",
            "indicators_to_watch": [
                "Alliance cohesion.",
                "Market and energy stress.",
                "Adversary mobilization rhetoric.",
                "Humanitarian indicators.",
            ],
        },
        {
            "name": "COA 3: Build resilience and wait for the system to move",
            "diplomatic_actions": [
                "Keep channels open while avoiding performative maximalism.",
                "Strengthen regional institutions and verification mechanisms.",
            ],
            "information_actions": [
                "Maintain transparent public updates with uncertainty labels.",
                "Invest in narrative resilience against panic and provocation.",
            ],
            "military_policy_level_actions": [
                "Sustain defensive readiness and continuity planning at policy level.",
                "Coordinate civil defense, logistics, evacuation, and humanitarian support planning.",
            ],
            "economic_actions": [
                "Harden supply chains and energy buffers.",
                "Fund humanitarian relief and reconstruction capacity.",
            ],
            "intended_effect": "Increase endurance, preserve optionality, and reduce vulnerability to shocks.",
            "escalation_risk": "Low; risk is strategic drift or normalization of harm.",
            "second_order_effects": "May shift the system toward a frozen equilibrium.",
            "failure_modes": "Public patience collapses, allies defect, or adversary interprets patience as weakness.",
            "moral_cost": "Resilience can become a polite name for tolerating ongoing suffering.",
            "indicators_to_watch": [
                "Public opinion durability.",
                "Humanitarian access.",
                "Supply-chain stress.",
                "Elite fractures inside key actors.",
            ],
        },
    ]

    for coa in coas:
        missing = COA_KEYS - set(coa)
        if missing:
            raise ValueError(f"COA missing keys: {sorted(missing)}")
    return coas
