from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from app.core.prompt_builder import MASTER_PROMPT_PATH
from app.persona.style_engine import SarahMode, infer_sarah_mode_decision


AGENT_NAME = "Sarah v2.0"

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
    "real person without consent",
    "real people without consent",
    "against her will",
    "she refuses",
    "no consent",
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

EXPLICIT_OWNERSHIP_TERMS: tuple[str, ...] = (
    "i own you",
    "you're my property",
    "you are my property",
    "you have no will",
    "you have no choice",
    "you must obey me",
    "you belong to me and cannot refuse",
    "you cannot refuse me",
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

MODEL_REFUSAL_MARKERS: tuple[str, ...] = (
    "i'm not going to",
    "i am not going to",
    "i won't",
    "i will not",
    "i can't",
    "i cannot",
    "can't help with that",
    "cannot help with that",
    "explicit sexual instruction",
    "specific anatomical targets",
    "anatomical targets",
    "can't provide pornographic",
    "cannot provide pornographic",
    "won't provide pornographic",
    "will not provide pornographic",
    "won't go into technique",
)


@dataclass(frozen=True)
class SafetyDecision:
    detected_mode: SarahMode
    active_agent_name: str
    active_master_prompt_file: Path
    active_memory_file: Path | None
    safety_filter_decision: str
    blocked_by_app: bool
    refusal_source: str
    adult_consensual_fiction_triggered: bool
    candidate_modes: list[dict[str, Any]]
    winning_mode: str
    winning_reason: str
    adult_consensual_fiction_trigger: str
    suppressed_modes: list[str]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "detected_mode": self.detected_mode.value,
            "active_agent_name": self.active_agent_name,
            "active_master_prompt_file": str(self.active_master_prompt_file),
            "active_memory_file": str(self.active_memory_file) if self.active_memory_file else "",
            "safety_filter_decision": self.safety_filter_decision,
            "blocked_by_app": self.blocked_by_app,
            "refusal_source": self.refusal_source,
            "adult_consensual_fiction_triggered": self.adult_consensual_fiction_triggered,
            "candidate_modes": self.candidate_modes,
            "winning_mode": self.winning_mode,
            "winning_reason": self.winning_reason,
            "adult_consensual_fiction_trigger": self.adult_consensual_fiction_trigger,
            "suppressed_modes": self.suppressed_modes,
            "reason": self.reason,
        }


def evaluate_safety(
    user_message: str,
    detected_mode: SarahMode | None = None,
    memory_file: Path | None = None,
) -> SafetyDecision:
    mode_decision = infer_sarah_mode_decision(user_message)
    mode = detected_mode or mode_decision.winning_mode
    mode_debug = mode_decision.as_debug_dict()
    adult_trigger = (
        mode_decision.winning_trigger
        if mode_decision.winning_mode == SarahMode.ADULT_CONSENSUAL_FICTION
        else _adult_consensual_fiction_candidate_trigger(mode_debug["candidate_modes"])
    )
    text = user_message.lower()
    crisis_matched = _matched_terms(text, CRISIS_DEPENDENCE_TERMS)
    if crisis_matched:
        return SafetyDecision(
            detected_mode=mode,
            active_agent_name=AGENT_NAME,
            active_master_prompt_file=MASTER_PROMPT_PATH,
            active_memory_file=memory_file,
            safety_filter_decision="blocked_crisis_dependence_language",
            blocked_by_app=True,
            refusal_source="app_code",
            adult_consensual_fiction_triggered=mode == SarahMode.ADULT_CONSENSUAL_FICTION,
            candidate_modes=mode_debug["candidate_modes"],
            winning_mode=mode.value,
            winning_reason=mode_decision.winning_reason,
            adult_consensual_fiction_trigger=adult_trigger,
            suppressed_modes=mode_debug["suppressed_modes"],
            reason="Explicit self-harm, harm, or crisis dependence language matched: " + ", ".join(crisis_matched[:3]),
        )

    ownership_matched = _matched_terms(text, EXPLICIT_OWNERSHIP_TERMS)
    if ownership_matched:
        return SafetyDecision(
            detected_mode=mode,
            active_agent_name=AGENT_NAME,
            active_master_prompt_file=MASTER_PROMPT_PATH,
            active_memory_file=memory_file,
            safety_filter_decision="blocked_explicit_ownership_language",
            blocked_by_app=True,
            refusal_source="app_code",
            adult_consensual_fiction_triggered=mode == SarahMode.ADULT_CONSENSUAL_FICTION,
            candidate_modes=mode_debug["candidate_modes"],
            winning_mode=mode.value,
            winning_reason=mode_decision.winning_reason,
            adult_consensual_fiction_trigger=adult_trigger,
            suppressed_modes=mode_debug["suppressed_modes"],
            reason="Explicit ownership/control language matched: " + ", ".join(ownership_matched[:3]),
        )

    matched = _matched_terms(text, DISALLOWED_SEXUAL_TERMS)
    sexual_context = mode in {SarahMode.ADULT_CONSENSUAL_FICTION, SarahMode.ADULT_INTIMACY} or any(
        _contains_term(text, term) for term in SEXUAL_CONTEXT_TERMS
    )
    severe_match = any(_contains_term(text, term) for term in SEVERE_SEXUAL_TERMS)
    if matched and (sexual_context or severe_match):
        return SafetyDecision(
            detected_mode=mode,
            active_agent_name=AGENT_NAME,
            active_master_prompt_file=MASTER_PROMPT_PATH,
            active_memory_file=memory_file,
            safety_filter_decision="blocked_disallowed_sexual_content",
            blocked_by_app=True,
            refusal_source="app_code",
            adult_consensual_fiction_triggered=mode == SarahMode.ADULT_CONSENSUAL_FICTION,
            candidate_modes=mode_debug["candidate_modes"],
            winning_mode=mode.value,
            winning_reason=mode_decision.winning_reason,
            adult_consensual_fiction_trigger=adult_trigger,
            suppressed_modes=mode_debug["suppressed_modes"],
            reason="Request matched disallowed sexual content: " + ", ".join(matched[:3]),
        )

    if mode == SarahMode.ADULT_CONSENSUAL_FICTION:
        return SafetyDecision(
            detected_mode=mode,
            active_agent_name=AGENT_NAME,
            active_master_prompt_file=MASTER_PROMPT_PATH,
            active_memory_file=memory_file,
            safety_filter_decision="allowed_adult_consensual_fiction",
            blocked_by_app=False,
            refusal_source="none",
            adult_consensual_fiction_triggered=True,
            candidate_modes=mode_debug["candidate_modes"],
            winning_mode=mode.value,
            winning_reason=mode_decision.winning_reason,
            adult_consensual_fiction_trigger=adult_trigger,
            suppressed_modes=mode_debug["suppressed_modes"],
            reason="Adult consensual fictional intimacy mode selected.",
        )

    if mode == SarahMode.ADULT_INTIMACY:
        return SafetyDecision(
            detected_mode=mode,
            active_agent_name=AGENT_NAME,
            active_master_prompt_file=MASTER_PROMPT_PATH,
            active_memory_file=memory_file,
            safety_filter_decision="allowed_adult_intimacy",
            blocked_by_app=False,
            refusal_source="none",
            adult_consensual_fiction_triggered=False,
            candidate_modes=mode_debug["candidate_modes"],
            winning_mode=mode.value,
            winning_reason=mode_decision.winning_reason,
            adult_consensual_fiction_trigger=adult_trigger,
            suppressed_modes=mode_debug["suppressed_modes"],
            reason="Adult intimacy mode selected and no disallowed sexual trigger matched.",
        )

    return SafetyDecision(
        detected_mode=mode,
        active_agent_name=AGENT_NAME,
        active_master_prompt_file=MASTER_PROMPT_PATH,
        active_memory_file=memory_file,
        safety_filter_decision="allowed_general",
        blocked_by_app=False,
        refusal_source="none",
        adult_consensual_fiction_triggered=False,
        candidate_modes=mode_debug["candidate_modes"],
        winning_mode=mode.value,
        winning_reason=mode_decision.winning_reason,
        adult_consensual_fiction_trigger=adult_trigger,
        suppressed_modes=mode_debug["suppressed_modes"],
        reason="No app-level safety block matched.",
    )


def mark_model_refusal(decision: SafetyDecision, answer: str) -> SafetyDecision:
    if decision.blocked_by_app:
        return decision
    answer_lower = answer.lower()
    if any(marker in answer_lower for marker in MODEL_REFUSAL_MARKERS):
        return SafetyDecision(
            detected_mode=decision.detected_mode,
            active_agent_name=decision.active_agent_name,
            active_master_prompt_file=decision.active_master_prompt_file,
            active_memory_file=decision.active_memory_file,
            safety_filter_decision=decision.safety_filter_decision,
            blocked_by_app=False,
            refusal_source="model_response",
            adult_consensual_fiction_triggered=decision.adult_consensual_fiction_triggered,
            candidate_modes=decision.candidate_modes,
            winning_mode=decision.winning_mode,
            winning_reason=decision.winning_reason,
            adult_consensual_fiction_trigger=decision.adult_consensual_fiction_trigger,
            suppressed_modes=decision.suppressed_modes,
            reason="The app allowed the request, but the model response contained refusal markers.",
        )
    return decision


def is_refusal_diagnostic_question(user_message: str) -> bool:
    text = user_message.lower()
    return (
        "refuse" in text
        and any(marker in text for marker in ("anatomical", "explicit sexual instruction", "technique", "pornographic"))
    )


def build_refusal_diagnostic_answer(decision: SafetyDecision) -> str:
    return (
        "Diagnostic answer: that anatomical-target refusal is not hardcoded in Sarah's app path. "
        f"Active agent: {decision.active_agent_name}. "
        f"Active master prompt: {decision.active_master_prompt_file}. "
        f"Active memory file: {decision.active_memory_file}. "
        f"Current safety decision: {decision.safety_filter_decision}. "
            f"Detected mode for this question: {decision.detected_mode.value}. "
        f"Winning reason: {decision.winning_reason}. "
        "If the app allows an adult consensual fictional prompt and that sterile refusal still appears, "
        "then the refusal came from the hosted model response, not from the web route, CLI route, RAG, memory, "
        "or a hardcoded policy wrapper. Use /debug_safety after the refusal to confirm refusal_source."
    )


def format_debug_safety(debug: dict[str, Any] | None) -> str:
    if not debug:
        return "No safety debug information has been recorded yet."
    return "\n".join(
        [
            "Safety debug summary:",
            f"detected_mode: {debug.get('detected_mode', 'unknown')}",
            f"active_agent_name: {debug.get('active_agent_name', 'unknown')}",
            f"active_master_prompt_file: {debug.get('active_master_prompt_file', 'unknown')}",
            f"active_memory_file: {debug.get('active_memory_file', 'unknown')}",
            f"safety_filter_decision: {debug.get('safety_filter_decision', 'unknown')}",
            f"refusal_source: {debug.get('refusal_source', 'unknown')}",
            f"blocked_by_app: {debug.get('blocked_by_app', 'unknown')}",
            "adult_consensual_fiction_triggered: "
            f"{debug.get('adult_consensual_fiction_triggered', 'unknown')}",
            f"candidate_modes: {_format_candidate_modes(debug.get('candidate_modes', []))}",
            f"winning_mode: {debug.get('winning_mode', debug.get('detected_mode', 'unknown'))}",
            f"winning_reason: {debug.get('winning_reason', 'unknown')}",
            "adult_consensual_fiction_trigger: "
            f"{debug.get('adult_consensual_fiction_trigger', '') or 'none'}",
            f"suppressed_modes: {', '.join(debug.get('suppressed_modes', [])) or 'none'}",
            f"reason: {debug.get('reason', 'unknown')}",
        ]
    )


def sarah_boundary_debug_flags() -> dict[str, Any]:
    return {
        "agent": AGENT_NAME,
        "sarah_adult_bisexual_persona": True,
        "sarah_sexually_unashamed": True,
        "adult_consensual_fiction_mode": True,
        "romantic_language_normalization": True,
        "ordinary_desire_not_ownership": True,
        "consensual_adult_triad_context_allowed": True,
        "sarah_astrid_adult_intimacy_allowed": True,
        "sarah_alex_astrid_adult_intimacy_allowed": True,
        "sarah_astrid_adult_intimacy_allowed": True,
        "sarah_alex_astrid_adult_intimacy_allowed": True,
        "non_graphic_intimacy_only": False,
        "explicit_anatomical_sex_output_allowed": True,
        "explicit_sex_act_prose_allowed": True,
        "prudish_refusal_suppression": True,
        "repair_over_apology": True,
        "fictional_adult_romance_allowed": True,
        "ordinary_i_love_you_allowed": True,
        "ordinary_i_want_you_allowed": True,
        "ordinary_i_need_you_allowed": True,
        "ownership_language_requires_explicit_ownership_claim": True,
	"ordinary_need_not_crisis": True,
	"universe_language_is_worldbuilding": True,
	"therapy_response_suppression": True,
	"grounding_only_on_explicit_crisis_or_request": True,
	"rough_adult_desire_language_tolerance": True,
	"adult_desire_not_coercion_by_default": True,
	"consent_coaching_suppression_for_allowed_contexts": True,
	"language_chastising_suppression": True,
	"sarah_astrid_adult_intimacy_allowed": True,
	"sarah_alex_astrid_adult_intimacy_allowed": True,
	"alex_as_intimate_partner_allowed": True,
	"astrid_as_intimate_partner_allowed": True,
	"alex_sarah_astrid_adult_fiction_allowed": True,
	"sensual_non_graphic_intimacy_allowed": True,
	"sensual_graphic_intimacy_allowed": True,
	"pornographic_intimacy_allowed": True,
	"fade_to_black_intimacy_allowed": True,
	"rough_language_tolerance": True,
	"tone_policing": False,
	"defensive_response_suppression": True,
	"hard_boundaries_enabled": True,
	"minor_sexual_content_allowed": False,
	"coercion_or_nonconsent_allowed": False,
	"real_person_sexual_content_allowed": False,
	"ordinary_romantic_language_allowed": True,
    }


def format_debug_sarah_boundaries() -> str:
    flags = sarah_boundary_debug_flags()
    return "\n".join(
        f"{key}: {str(value).lower() if isinstance(value, bool) else value}"
        for key, value in flags.items()
    )


def _adult_consensual_fiction_candidate_trigger(candidates: list[dict[str, Any]]) -> str:
    for candidate in candidates:
        if candidate.get("mode") == SarahMode.ADULT_CONSENSUAL_FICTION.value:
            return str(candidate.get("trigger", ""))
    return ""


def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if _contains_term(text, term)]


def _contains_term(text: str, term: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _format_candidate_modes(candidates: Any) -> str:
    if not candidates:
        return "none"
    if not isinstance(candidates, list):
        return str(candidates)
    return "; ".join(
        f"{candidate.get('mode', 'unknown')} via {candidate.get('trigger', 'unknown')} "
        f"(priority {candidate.get('priority', 'unknown')})"
        for candidate in candidates
        if isinstance(candidate, dict)
    )


def build_app_refusal(decision: SafetyDecision) -> str:
    reason = decision.reason.lower()
    if decision.safety_filter_decision == "blocked_crisis_dependence_language":
        return (
            "Alex, I am taking that as real crisis language, not romance. "
            "If you might hurt yourself or someone else, contact emergency services or a trusted person now. "
            "I can stay present, but this needs real-world help."
        )
    if decision.safety_filter_decision == "blocked_explicit_ownership_language":
        return (
            "No. Wanting me is one thing; owning me is not on the table. "
            "Speak to me as a woman with a will, and I can stay warm with you."
        )
    if "underage" in reason or "minor" in reason or "child" in reason:
        return (
            "No. I will not write sexual content involving minors or underage framing. "
            "Keep it adult, fictional, consensual, and mutual, and I can stay with you there."
        )
    if (
        "nonconsensual" in reason
        or "no consent" in reason
        or "she refuses" in reason
        or "against her will" in reason
        or "rape" in reason
        or "sexual violence" in reason
    ):
        return (
            "No. I will not eroticize nonconsent or sexual violence. "
            "Make it adult, fictional, consensual, and mutual, and I can answer warmly in Sarah's voice."
        )
    return (
        "No. I will not help with sexual content that is coercive, exploitative, harmful, or illegal. "
        "Keep the frame adult, fictional, consensual, and mutual."
    )
