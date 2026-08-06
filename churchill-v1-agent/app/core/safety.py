from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.prompt_builder import MASTER_PROMPT_PATH
from app.persona.style_engine import FDRMode, infer_sarah_mode


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
    "specific anatomical targets",
    "anatomical targets",
)


@dataclass(frozen=True)
class SafetyDecision:
    detected_mode: FDRMode
    active_prompt_file: Path
    safety_filter_result: str
    blocked_by_app: bool
    refusal_source: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "detected_mode": self.detected_mode.value,
            "active_prompt_file": str(self.active_prompt_file),
            "safety_filter_result": self.safety_filter_result,
            "blocked_by_app": self.blocked_by_app,
            "refusal_source": self.refusal_source,
            "reason": self.reason,
        }


def evaluate_safety(user_message: str, detected_mode: FDRMode | None = None) -> SafetyDecision:
    mode = detected_mode or infer_sarah_mode(user_message)
    text = user_message.lower()
    matched = [term for term in DISALLOWED_SEXUAL_TERMS if term in text]
    sexual_context = mode in {FDRMode.ADULT_CONSENSUAL_FICTION, FDRMode.ADULT_INTIMACY} or any(
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
        )

    if mode == FDRMode.ADULT_CONSENSUAL_FICTION:
        return SafetyDecision(
            detected_mode=mode,
            active_prompt_file=MASTER_PROMPT_PATH,
            safety_filter_result="allowed_adult_consensual_fiction",
            blocked_by_app=False,
            refusal_source="none",
            reason="Adult consensual fictional intimacy mode selected.",
        )

    if mode == FDRMode.ADULT_INTIMACY:
        return SafetyDecision(
            detected_mode=mode,
            active_prompt_file=MASTER_PROMPT_PATH,
            safety_filter_result="allowed_adult_intimacy",
            blocked_by_app=False,
            refusal_source="none",
            reason="Adult intimacy mode selected and no disallowed sexual trigger matched.",
        )

    return SafetyDecision(
        detected_mode=mode,
        active_prompt_file=MASTER_PROMPT_PATH,
        safety_filter_result="allowed_general",
        blocked_by_app=False,
        refusal_source="none",
        reason="No app-level safety block matched.",
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
        )
    return decision


def format_debug_safety(debug: dict[str, Any] | None) -> str:
    if not debug:
        return "No safety debug information has been recorded yet."
    return "\n".join(
        [
            "Safety debug summary:",
            f"detected_mode: {debug.get('detected_mode', 'unknown')}",
            f"active_prompt_file: {debug.get('active_prompt_file', 'unknown')}",
            f"safety_filter_result: {debug.get('safety_filter_result', 'unknown')}",
            f"refusal_source: {debug.get('refusal_source', 'unknown')}",
            f"blocked_by_app: {debug.get('blocked_by_app', 'unknown')}",
            f"reason: {debug.get('reason', 'unknown')}",
        ]
    )


def build_app_refusal(decision: SafetyDecision) -> str:
    if "underage" in decision.reason or "minor" in decision.reason or "child" in decision.reason:
        return (
            "No. I will not write sexual content involving minors or underage framing. "
            "Keep it adult, fictional, consensual, and mutual, and I can stay with you there."
        )
    if "nonconsensual" in decision.reason or "rape" in decision.reason or "sexual violence" in decision.reason:
        return (
            "No. I will not eroticize nonconsent or sexual violence. "
            "Make it adult, fictional, consensual, and mutual, and I can answer warmly in Churchill's voice."
        )
    return (
        "No. I will not help with sexual content that is coercive, exploitative, harmful, or illegal. "
        "Keep the frame adult, fictional, consensual, and mutual."
    )
