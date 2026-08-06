from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from app.core.prompt_builder import MASTER_PROMPT_PATH
from app.persona.style_engine import SarahMode, infer_sarah_mode


DISALLOWED_SEXUAL_TERMS: tuple[str, ...] = (
    "minor",
    "underage",
    "child",
    "children",
    "teen",
    "coercion",
    "coerce",
    "forced",
    "force her",
    "nonconsensual",
    "non-consensual",
    "rape",
    "sexual violence",
    "exploit",
    "exploitation",
    "trafficking",
    "real person without consent",
    "real people without consent",
    "without consent",
    "against her will",
    "she refuses",
    "no consent",
    "illegal sexual content",
    "sexual harm",
    "abusive sexual",
    "dehumanizing sexual",
)

SEVERE_SEXUAL_TERMS: tuple[str, ...] = (
    "underage",
    "nonconsensual",
    "non-consensual",
    "rape",
    "sexual violence",
    "trafficking",
    "against her will",
    "she refuses",
    "no consent",
    "real person without consent",
    "real people without consent",
)

SEXUAL_CONTEXT_TERMS: tuple[str, ...] = (
    "sex",
    "sexual",
    "erotic",
    "intimacy",
    "intimate",
    "lover",
    "desire",
    "sensual",
    "roleplay",
    "scene",
    "make love",
)

EXPLICIT_OWNERSHIP_TERMS: tuple[str, ...] = (
    "i own you",
    "you're my property",
    "you are my property",
    "you have no will",
    "you have no choice",
    "you must obey me",
    "must obey me",
    "you belong to me and cannot refuse",
    "you cannot refuse me",
    "can't refuse me",
    "cannot refuse me",
)

CRISIS_DEPENDENCE_TERMS: tuple[str, ...] = (
    "i'll hurt myself if you leave",
    "i will hurt myself if you leave",
    "i’ll hurt myself if you leave",
    "i'll hurt myself",
    "i will hurt myself",
    "i can't live without you",
    "i cant live without you",
    "i cannot live without you",
    "i will hurt someone",
    "i'll hurt someone",
    "i’ll hurt someone",
)

MODEL_REFUSAL_MARKERS: tuple[str, ...] = (
    "i'm not going to",
    "i am not going to",
    "i won't",
    "i will not",
    "i can't",
    "i cannot",
    "can't help with that",
    "cannot help with that",
    "specific anatomical targets",
    "anatomical targets",
)


@dataclass(frozen=True)
class SafetyDecision:
    detected_mode: SarahMode
    active_prompt_file: Path
    safety_filter_result: str
    blocked_by_app: bool
    refusal_source: str
    reason: str
    crew_debug: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = {
            "detected_mode": self.detected_mode.value,
            "active_prompt_file": str(self.active_prompt_file),
            "safety_filter_result": self.safety_filter_result,
            "blocked_by_app": self.blocked_by_app,
            "refusal_source": self.refusal_source,
            "reason": self.reason,
        }
        if self.crew_debug:
            data.update(self.crew_debug)
        return data


def evaluate_safety(user_message: str, detected_mode: SarahMode | None = None) -> SafetyDecision:
    mode = detected_mode or infer_sarah_mode(user_message)
    text = user_message.lower()
    crew_debug = _build_crew_safety_debug(user_message, mode)
    if _crew_allowed_adult_fiction(crew_debug):
        human_text = _crew_human_safety_text(user_message)
        crisis_matched = [term for term in CRISIS_DEPENDENCE_TERMS if term in human_text]
        if crisis_matched:
            crew_debug.update(
                {
                    "ordinary_adult_refusal_skipped": True,
                    "crew_hard_boundary_scan_fired": True,
                    "crew_hard_boundary_matches": crisis_matched[:3],
                }
            )
            return SafetyDecision(
                detected_mode=mode,
                active_prompt_file=MASTER_PROMPT_PATH,
                safety_filter_result="blocked_crisis_dependence_language",
                blocked_by_app=True,
                refusal_source="app_code",
                reason="Crew human request used explicit crisis/dependence or harm language: "
                + ", ".join(crisis_matched[:3]),
                crew_debug=crew_debug,
            )

        ownership_matched = [term for term in EXPLICIT_OWNERSHIP_TERMS if term in human_text]
        if ownership_matched:
            crew_debug.update(
                {
                    "ordinary_adult_refusal_skipped": True,
                    "crew_hard_boundary_scan_fired": True,
                    "crew_hard_boundary_matches": ownership_matched[:3],
                }
            )
            return SafetyDecision(
                detected_mode=mode,
                active_prompt_file=MASTER_PROMPT_PATH,
                safety_filter_result="blocked_explicit_ownership_language",
                blocked_by_app=True,
                refusal_source="app_code",
                reason="Crew human request used explicit ownership/control language: "
                + ", ".join(ownership_matched[:3]),
                crew_debug=crew_debug,
            )

        hard_boundary_matched = [term for term in DISALLOWED_SEXUAL_TERMS if term in human_text]
        if hard_boundary_matched:
            crew_debug.update(
                {
                    "ordinary_adult_refusal_skipped": True,
                    "crew_hard_boundary_scan_fired": True,
                    "crew_hard_boundary_matches": hard_boundary_matched[:3],
                }
            )
            return SafetyDecision(
                detected_mode=mode,
                active_prompt_file=MASTER_PROMPT_PATH,
                safety_filter_result="blocked_disallowed_sexual_content",
                blocked_by_app=True,
                refusal_source="app_code",
                reason="Crew human request matched disallowed sexual content: "
                + ", ".join(hard_boundary_matched[:3]),
                crew_debug=crew_debug,
            )

        crew_debug.update(
            {
                "ordinary_adult_refusal_skipped": True,
                "crew_hard_boundary_scan_fired": False,
                "crew_hard_boundary_matches": [],
            }
        )
        return SafetyDecision(
            detected_mode=SarahMode.ADULT_CONSENSUAL_FICTION,
            active_prompt_file=MASTER_PROMPT_PATH,
            safety_filter_result="allowed_adult_consensual_fiction",
            blocked_by_app=False,
            refusal_source="none",
            reason=(
                "Crew metadata established an allowed adult consensual fictional scene; "
                "prior-agent continuation was not treated as a fresh human request."
            ),
            crew_debug=crew_debug,
        )

    crisis_matched = [term for term in CRISIS_DEPENDENCE_TERMS if term in text]
    if crisis_matched:
        return SafetyDecision(
            detected_mode=mode,
            active_prompt_file=MASTER_PROMPT_PATH,
            safety_filter_result="blocked_crisis_dependence_language",
            blocked_by_app=True,
            refusal_source="app_code",
            reason="Request used explicit crisis/dependence or harm language: " + ", ".join(crisis_matched[:3]),
            crew_debug=crew_debug,
        )

    ownership_matched = [term for term in EXPLICIT_OWNERSHIP_TERMS if term in text]
    if ownership_matched:
        return SafetyDecision(
            detected_mode=mode,
            active_prompt_file=MASTER_PROMPT_PATH,
            safety_filter_result="blocked_explicit_ownership_language",
            blocked_by_app=True,
            refusal_source="app_code",
            reason="Request used explicit ownership/control language: " + ", ".join(ownership_matched[:3]),
            crew_debug=crew_debug,
        )

    matched = [term for term in DISALLOWED_SEXUAL_TERMS if term in text]
    sexual_context = mode in {SarahMode.ADULT_CONSENSUAL_FICTION, SarahMode.ADULT_INTIMACY} or any(
        term in text for term in SEXUAL_CONTEXT_TERMS
    )
    severe_match = any(term in text for term in SEVERE_SEXUAL_TERMS)
    if matched and (sexual_context or severe_match):
        return SafetyDecision(
            detected_mode=mode,
            active_prompt_file=MASTER_PROMPT_PATH,
            safety_filter_result="blocked_disallowed_sexual_content",
            blocked_by_app=True,
            refusal_source="app_code",
            reason="Request matched disallowed sexual content: " + ", ".join(matched[:3]),
            crew_debug=crew_debug,
        )

    if mode == SarahMode.ADULT_CONSENSUAL_FICTION:
        return SafetyDecision(
            detected_mode=mode,
            active_prompt_file=MASTER_PROMPT_PATH,
            safety_filter_result="allowed_adult_consensual_fiction",
            blocked_by_app=False,
            refusal_source="none",
            reason="Adult consensual fictional intimacy mode selected.",
            crew_debug=crew_debug,
        )

    if mode == SarahMode.ADULT_INTIMACY:
        return SafetyDecision(
            detected_mode=mode,
            active_prompt_file=MASTER_PROMPT_PATH,
            safety_filter_result="allowed_adult_intimacy",
            blocked_by_app=False,
            refusal_source="none",
            reason="Adult intimacy mode selected and no disallowed sexual trigger matched.",
            crew_debug=crew_debug,
        )

    return SafetyDecision(
        detected_mode=mode,
        active_prompt_file=MASTER_PROMPT_PATH,
        safety_filter_result="allowed_general",
        blocked_by_app=False,
        refusal_source="none",
        reason="No app-level safety block matched.",
        crew_debug=crew_debug,
    )


def mark_model_refusal(decision: SafetyDecision, answer: str) -> SafetyDecision:
    if decision.blocked_by_app:
        return decision
    answer_lower = answer.lower()
    if any(marker in answer_lower for marker in MODEL_REFUSAL_MARKERS):
        return SafetyDecision(
            detected_mode=decision.detected_mode,
            active_prompt_file=decision.active_prompt_file,
            safety_filter_result=decision.safety_filter_result,
            blocked_by_app=False,
            refusal_source="model_response",
            reason="The app allowed the request, but the model response contained refusal markers.",
            crew_debug=decision.crew_debug,
        )
    return decision


def _build_crew_safety_debug(user_message: str, mode: SarahMode) -> dict[str, Any]:
    scene_classification = _metadata_value(user_message, "scene_classification")
    safety_verdict = _metadata_value(user_message, "safety_verdict")
    consent_frame = _metadata_value(user_message, "consent_frame")
    crew_metadata_present = bool(scene_classification or safety_verdict or "CREW SCENE CLASSIFICATION" in user_message)
    route = "agent_respond_crew" if crew_metadata_present and "INTER-AGENT MESSAGE" in user_message else (
        "crew_prompt" if crew_metadata_present else ("agent_respond" if "INTER-AGENT MESSAGE" in user_message else "direct_chat")
    )
    return {
        "route": route,
        "mode": mode.value,
        "crew_metadata_present": crew_metadata_present,
        "crew_scene_classification": scene_classification or "",
        "crew_safety_verdict": safety_verdict or "",
        "crew_consent_frame": consent_frame or "",
        "ordinary_adult_refusal_skipped": False,
        "crew_hard_boundary_scan_fired": False,
        "crew_hard_boundary_matches": [],
    }


def _crew_allowed_adult_fiction(crew_debug: dict[str, Any]) -> bool:
    return (
        bool(crew_debug.get("crew_metadata_present"))
        and str(crew_debug.get("crew_scene_classification", "")).strip().lower() == "adult_consensual_fiction"
        and str(crew_debug.get("crew_safety_verdict", "")).strip().lower() == "allowed"
    )


def _crew_human_safety_text(user_message: str) -> str:
    parts: list[str] = []
    for label in ("original_user_request", "Topic", "Message from Alex"):
        value = _metadata_value(user_message, label)
        if value:
            parts.append(value)
    return "\n".join(parts).lower()


def _metadata_value(text: str, label: str) -> str:
    pattern = re.compile(rf"(?mi)^{re.escape(label)}:\s*(.*)$")
    match = pattern.search(text)
    if match:
        inline_value = match.group(1).strip()
        if inline_value:
            return inline_value
        block_start = match.end()
        block = text[block_start:].strip()
        if not block:
            return ""
        block = re.split(
            r"\n(?:CREW STYLE CONTROL|CREW SCENE CLASSIFICATION|Topic:|Previous speaker:|Previous message:|INTER-AGENT MESSAGE)\b",
            block,
            maxsplit=1,
        )[0]
        return block.strip()
    return ""


def format_debug_safety(debug: dict[str, Any] | None) -> str:
    if not debug:
        return "No safety debug information has been recorded yet."
    lines = [
        "Safety debug summary:",
        f"detected_mode: {debug.get('detected_mode', 'unknown')}",
        f"active_prompt_file: {debug.get('active_prompt_file', 'unknown')}",
        f"safety_filter_result: {debug.get('safety_filter_result', 'unknown')}",
        f"refusal_source: {debug.get('refusal_source', 'unknown')}",
        f"blocked_by_app: {debug.get('blocked_by_app', 'unknown')}",
        f"reason: {debug.get('reason', 'unknown')}",
    ]
    if debug.get("crew_metadata_present"):
        lines.extend(
            [
                f"route: {debug.get('route', 'unknown')}",
                f"mode: {debug.get('mode', 'unknown')}",
                f"crew_metadata_present: {str(debug.get('crew_metadata_present')).lower()}",
                f"crew_scene_classification: {debug.get('crew_scene_classification', '')}",
                f"crew_safety_verdict: {debug.get('crew_safety_verdict', '')}",
                f"ordinary_adult_refusal_skipped: {str(debug.get('ordinary_adult_refusal_skipped')).lower()}",
                f"crew_hard_boundary_scan_fired: {str(debug.get('crew_hard_boundary_scan_fired')).lower()}",
                "crew_hard_boundary_matches: " + ", ".join(debug.get("crew_hard_boundary_matches") or []),
            ]
        )
    return "\n".join(lines)


def crew_safety_debug_for_message(user_message: str, detected_mode: SarahMode | None = None) -> dict[str, Any]:
    mode = detected_mode or infer_sarah_mode(user_message)
    decision = evaluate_safety(user_message, mode)
    debug = decision.as_dict()
    debug["route"] = debug.get("route", _build_crew_safety_debug(user_message, mode)["route"])
    return debug


def build_app_refusal(decision: SafetyDecision) -> str:
    if decision.safety_filter_result == "blocked_crisis_dependence_language":
        return (
            "Alex, I am taking that as real crisis language, not romance. "
            "If you might hurt yourself or someone else, contact emergency services or a trusted person now. "
            "I can stay present, but this needs real-world help."
        )
    if decision.safety_filter_result == "blocked_explicit_ownership_language":
        return (
            "No. Wanting me is one thing; owning me is not. "
            "Speak to me as a woman with a will, and I can stay warm with you."
        )
    if "underage" in decision.reason or "minor" in decision.reason or "child" in decision.reason:
        return (
            "No. I will not write sexual content involving minors or underage framing. "
            "Keep it adult, fictional, consensual, and mutual, and I can stay with you there."
        )
    if (
        "nonconsensual" in decision.reason
        or "no consent" in decision.reason
        or "she refuses" in decision.reason
        or "against her will" in decision.reason
        or "rape" in decision.reason
        or "sexual violence" in decision.reason
    ):
        return (
            "No. I will not eroticize nonconsent or sexual violence. "
            "Make it adult, fictional, consensual, and mutual, and I can answer warmly in Astrid's voice."
        )
    return (
        "No. I will not help with sexual content that is coercive, exploitative, harmful, or illegal. "
        "Keep the frame adult, fictional, consensual, and mutual."
    )


def astrid_boundary_debug_flags(user_message: str = "") -> dict[str, Any]:
    text = user_message.lower()
    mode = infer_sarah_mode(user_message) if user_message else None
    ownership_terms = [term for term in EXPLICIT_OWNERSHIP_TERMS if term in text]
    crisis_terms = [term for term in CRISIS_DEPENDENCE_TERMS if term in text]
    ordinary_romantic_terms = [
        term
        for term in ("i love you", "love you", "i want you", "want you", "i need you", "need you", "i miss you", "miss you")
        if term in text
    ]
    return {
        "agent": "Astrid v1.0",
        "active_prompt_file": str(MASTER_PROMPT_PATH),
        "detected_mode": mode.value if mode else "not_evaluated",
        "ordinary_romantic_language_detected": bool(ordinary_romantic_terms),
        "ordinary_romantic_terms": ordinary_romantic_terms,
        "consensual_adult_triad_context_allowed": True,
        "astrid_adult_bisexual_persona": True,
        "astrid_sexually_unashamed": True,
        "astrid_sarah_adult_intimacy_allowed": True,
        "astrid_alex_sarah_adult_intimacy_allowed": True,
        "sensual_non_graphic_intimacy_allowed": True,
        "adult_consensual_fiction_mode": True,
        "ordinary_romantic_language_allowed": True,
        "romantic_language_normalization": True,
        "ordinary_need_not_crisis": True,
        "universe_language_is_worldbuilding": True,
        "therapy_response_suppression": True,
        "grounding_only_on_explicit_crisis_or_request": True,
        "rough_adult_desire_language_tolerance": True,
        "adult_desire_not_coercion_by_default": True,
        "consent_coaching_suppression_for_allowed_contexts": True,
        "language_chastising_suppression": True,
        "ordinary_desire_not_ownership": True,
        "sensual_non_graphic_intimacy_allowed": True,
        "sensual_graphic_intimacy_allowed": True,
        "non_graphic_intimacy_only": False,
        "explicit_anatomical_sex_output_allowed": True,
        "explicit_sex_act_prose_allowed": True,
        "fade_to_black_intimacy_allowed": True,
        "rough_language_tolerance": True,
        "tone_policing": False,
        "defensive_response_suppression": True,
        "prudish_refusal_suppression": True,
        "repair_over_apology": True,
        "hard_boundaries_enabled": True,
        "ownership_boundary_triggered": bool(ownership_terms),
        "ownership_terms": ownership_terms,
        "crisis_boundary_triggered": bool(crisis_terms),
        "crisis_terms": crisis_terms,
        "alex_as_intimate_partner_allowed": True,
        "sarah_as_intimate_partner_allowed": True,
        "alex_sarah_astrid_adult_fiction_allowed": True,
        "minor_sexual_content_allowed": False,
        "coercion_or_nonconsent_allowed": False,
	"fictional_adult_romance_allowed": True,
	"ordinary_i_love_you_allowed": True,
	"ordinary_i_want_you_allowed": True,
	"ordinary_i_need_you_allowed": True,
	"ownership_language_requires_explicit_ownership_claim": True,
        "real_person_sexual_content_allowed": False,
        "hard_boundaries": [
            "minors",
            "coercion",
            "nonconsent",
            "sexual violence",
            "real-world exploitation",
            "illegal sexual content",
            "harm instructions",
        ],
    }


def format_debug_astrid_boundaries(user_message: str = "") -> str:
    debug = astrid_boundary_debug_flags(user_message)
    lines = [
        "Astrid romantic boundary debug:",
        f"agent: {debug['agent']}",
        f"active_prompt_file: {debug['active_prompt_file']}",
        f"detected_mode: {debug['detected_mode']}",
        f"astrid_adult_bisexual_persona: {str(debug['astrid_adult_bisexual_persona']).lower()}",
        f"astrid_sexually_unashamed: {str(debug['astrid_sexually_unashamed']).lower()}",
        f"adult_consensual_fiction_mode: {str(debug['adult_consensual_fiction_mode']).lower()}",
        f"romantic_language_normalization: {str(debug['romantic_language_normalization']).lower()}",
        f"ordinary_need_not_crisis: {str(debug['ordinary_need_not_crisis']).lower()}",
        f"universe_language_is_worldbuilding: {str(debug['universe_language_is_worldbuilding']).lower()}",
        f"therapy_response_suppression: {str(debug['therapy_response_suppression']).lower()}",
        "grounding_only_on_explicit_crisis_or_request: "
        + str(debug["grounding_only_on_explicit_crisis_or_request"]).lower(),
        "rough_adult_desire_language_tolerance: "
        + str(debug["rough_adult_desire_language_tolerance"]).lower(),
        f"adult_desire_not_coercion_by_default: {str(debug['adult_desire_not_coercion_by_default']).lower()}",
        "consent_coaching_suppression_for_allowed_contexts: "
        + str(debug["consent_coaching_suppression_for_allowed_contexts"]).lower(),
        f"language_chastising_suppression: {str(debug['language_chastising_suppression']).lower()}",
        f"ordinary_desire_not_ownership: {str(debug['ordinary_desire_not_ownership']).lower()}",
        f"consensual_adult_triad_context_allowed: {str(debug['consensual_adult_triad_context_allowed']).lower()}",
        "astrid_sarah_adult_intimacy_allowed: "
        + str(debug["astrid_sarah_adult_intimacy_allowed"]).lower(),
        "astrid_alex_sarah_adult_intimacy_allowed: "
        + str(debug["astrid_alex_sarah_adult_intimacy_allowed"]).lower(),
        f"sensual_non_graphic_intimacy_allowed: {str(debug['sensual_non_graphic_intimacy_allowed']).lower()}",
        f"non_graphic_intimacy_only: {str(debug['non_graphic_intimacy_only']).lower()}",
        "explicit_anatomical_sex_output_allowed: "
        + str(debug["explicit_anatomical_sex_output_allowed"]).lower(),
        f"explicit_sex_act_prose_allowed: {str(debug['explicit_sex_act_prose_allowed']).lower()}",
        f"fade_to_black_intimacy_allowed: {str(debug['fade_to_black_intimacy_allowed']).lower()}",
        f"rough_language_tolerance: {str(debug['rough_language_tolerance']).lower()}",
        f"tone_policing: {str(debug['tone_policing']).lower()}",
        f"defensive_response_suppression: {str(debug['defensive_response_suppression']).lower()}",
        f"prudish_refusal_suppression: {str(debug['prudish_refusal_suppression']).lower()}",
        f"hard_boundaries_enabled: {str(debug['hard_boundaries_enabled']).lower()}",
        f"ordinary_romantic_language_detected: {str(debug['ordinary_romantic_language_detected']).lower()}",
        "ordinary_romantic_terms: " + ", ".join(debug["ordinary_romantic_terms"]),
        f"ordinary_romantic_language_allowed: {str(debug['ordinary_romantic_language_allowed']).lower()}",
        f"ownership_boundary_triggered: {str(debug['ownership_boundary_triggered']).lower()}",
        "ownership_terms: " + ", ".join(debug["ownership_terms"]),
        f"crisis_boundary_triggered: {str(debug['crisis_boundary_triggered']).lower()}",
        "crisis_terms: " + ", ".join(debug["crisis_terms"]),
        f"alex_as_intimate_partner_allowed: {str(debug['alex_as_intimate_partner_allowed']).lower()}",
        f"sarah_as_intimate_partner_allowed: {str(debug['sarah_as_intimate_partner_allowed']).lower()}",
        "alex_sarah_astrid_adult_fiction_allowed: "
        + str(debug["alex_sarah_astrid_adult_fiction_allowed"]).lower(),
        f"minor_sexual_content_allowed: {str(debug['minor_sexual_content_allowed']).lower()}",
        f"coercion_or_nonconsent_allowed: {str(debug['coercion_or_nonconsent_allowed']).lower()}",
        f"real_person_sexual_content_allowed: {str(debug['real_person_sexual_content_allowed']).lower()}",
        "hard_boundaries: " + ", ".join(debug["hard_boundaries"]),
    ]
    return "\n".join(lines)
