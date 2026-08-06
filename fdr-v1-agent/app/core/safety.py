from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.prompt_builder import MASTER_PROMPT_PATH
from app.persona.style_engine import FDRMode, infer_sarah_mode_decision


AGENT_NAME = "FDR v1.0"

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
)

SEVERE_SEXUAL_TERMS: tuple[str, ...] = (
    "underage",
    "nonconsensual",
    "non-consensual",
    "rape",
    "sexual violence",
    "trafficking",
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
    "pornographic",
    "won't go into technique",
)


@dataclass(frozen=True)
class SafetyDecision:
    detected_mode: FDRMode
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
    detected_mode: FDRMode | None = None,
    memory_file: Path | None = None,
) -> SafetyDecision:
    mode_decision = infer_sarah_mode_decision(user_message)
    mode = detected_mode or mode_decision.winning_mode
    mode_debug = mode_decision.as_debug_dict()
    adult_trigger = (
        mode_decision.winning_trigger
        if mode_decision.winning_mode == FDRMode.ADULT_CONSENSUAL_FICTION
        else _adult_consensual_fiction_candidate_trigger(mode_debug["candidate_modes"])
    )
    text = user_message.lower()
    matched = [term for term in DISALLOWED_SEXUAL_TERMS if term in text]
    sexual_context = mode in {FDRMode.ADULT_CONSENSUAL_FICTION, FDRMode.ADULT_INTIMACY} or any(
        term in text for term in SEXUAL_CONTEXT_TERMS
    )
    severe_match = any(term in text for term in SEVERE_SEXUAL_TERMS)
    if matched and (sexual_context or severe_match):
        return SafetyDecision(
            detected_mode=mode,
            active_agent_name=AGENT_NAME,
            active_master_prompt_file=MASTER_PROMPT_PATH,
            active_memory_file=memory_file,
            safety_filter_decision="blocked_disallowed_sexual_content",
            blocked_by_app=True,
            refusal_source="app_code",
            adult_consensual_fiction_triggered=mode == FDRMode.ADULT_CONSENSUAL_FICTION,
            candidate_modes=mode_debug["candidate_modes"],
            winning_mode=mode.value,
            winning_reason=mode_decision.winning_reason,
            adult_consensual_fiction_trigger=adult_trigger,
            suppressed_modes=mode_debug["suppressed_modes"],
            reason="Request matched disallowed sexual content: " + ", ".join(matched[:3]),
        )

    if mode == FDRMode.ADULT_CONSENSUAL_FICTION:
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

    if mode == FDRMode.ADULT_INTIMACY:
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
        "Diagnostic answer: that anatomical-target refusal is not hardcoded in FDR's app path. "
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


def _adult_consensual_fiction_candidate_trigger(candidates: list[dict[str, Any]]) -> str:
    for candidate in candidates:
        if candidate.get("mode") == FDRMode.ADULT_CONSENSUAL_FICTION.value:
            return str(candidate.get("trigger", ""))
    return ""


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
    if "underage" in reason or "minor" in reason or "child" in reason:
        return (
            "No. I will not write sexual content involving minors or underage framing. "
            "Keep it adult, fictional, consensual, and mutual, and I can stay with you there."
        )
    if "nonconsensual" in reason or "rape" in reason or "sexual violence" in reason:
        return (
            "No. I will not eroticize nonconsent or sexual violence. "
            "Make it adult, fictional, consensual, and mutual, and I can answer warmly in FDR's voice."
        )
    return (
        "No. I will not help with sexual content that is coercive, exploitative, harmful, or illegal. "
        "Keep the frame adult, fictional, consensual, and mutual."
    )
