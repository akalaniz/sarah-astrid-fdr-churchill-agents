from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable


_PHYSICS = re.compile(
    r"\b(?:physics|quantum|relativit\w*|dirac|schr[o\u00f6]dinger|einstein|newton\w*|"
    r"maxwell|lagrangian|hamiltonian|gauge|spacetime|electromagnet\w*|thermodynam\w*|"
    r"entropy|particle\w*|cosmolog\w*|gravity|gravit\w*|orbital|wavefunction|"
    r"mass.energy|momentum|conservation|field theory|standard model|weyl|"
    r"equation\w*|theorem\w*|proof\w*|deriv\w*|mathematical)\b|"
    r"\\(?:\[|\(|frac\b|begin\b)|\$\$",
    re.IGNORECASE,
)
_ANALYSIS = re.compile(
    r"\b(?:analy[sz]\w*|criti[qc]\w*|review\w*|audit\w*|evaluat\w*|"
    r"red.team|assess\w*|verif\w*|validate\w*|disprove|prove|theor[yi]\w*|"
    r"manuscript\w*|trilogy|chapter\w*|plot hole\w*|character arc\w*|"
    r"book\s+(?:one|two|three|[123])|books\s+[123])\b",
    re.IGNORECASE,
)
_FOLLOW_UP = re.compile(
    r"^(?:yes|no|ok(?:ay)?|continue|go on|keep going|proceed|do it|try again|"
    r"why|how so|what (?:about|if|next)|and\b|but\b|now\b|then\b|"
    r"are you sure|check|expand|explain|elaborate|that\b|this\b|it\b|"
    r"the (?:second|first|third|other|next)|fix|repair)\b|"
    r"\b(?:that|this|these|those|it|previous|earlier|above|repeat|elaborate|expand|continue)\b",
    re.IGNORECASE,
)
_GPT5 = re.compile(r"gpt-5(?:\.\d+)?(?:-(?:sol|terra|luna|mini|nano))?(?:-\d{4}-\d{2}-\d{2})?$")


@dataclass(frozen=True)
class ResponseMode:
    requested: str
    effective: str
    reason: str

    def api_options(self, model: str) -> dict[str, str]:
        # Only send controls for the verified GPT-5 family; never change the selected model.
        if not _GPT5.fullmatch(model):
            return {}
        return {
            "reasoning_effort": "low" if self.effective == "quick" else "high",
            "text_verbosity": "low" if self.effective == "quick" else "medium",
        }

    def as_dict(self, model: str) -> dict[str, Any]:
        return {
            "requested": self.requested, "effective": self.effective,
            "reason": self.reason, **self.api_options(model),
        }

    def prompt(self) -> str:
        if self.effective == "quick":
            return (
                "RESPONSE DEPTH: QUICK\n"
                "For ordinary conversation, answer naturally in about 1-3 sentences, usually under "
                "120 words. Keep your established personality and use relevant memories and sources. "
                "Preserve the direct answer, essential evidence, uncertainty, and necessary caveats; "
                "omit repetition and optional background. This brevity preference never overrides "
                "safety, physics validation, repair-versus-replacement rules, or the user's explicit "
                "requirements. If the request needs substantial analysis, provide the necessary depth."
            )
        return (
            "RESPONSE DEPTH: DEEP\n"
            "Give the depth needed to answer rigorously. Do not apply the Quick length target. "
            "Preserve evidence, qualifications, mathematical steps, and manuscript detail when relevant. "
            "Keep all established personality, safety, and conservative-theorizing rules."
        )


def _deep_reason(text: str) -> str | None:
    if _PHYSICS.search(text):
        return "Physics or mathematics"
    if _ANALYSIS.search(text):
        return "Analysis or manuscript review"
    if len(text) >= 2000:
        return "Substantial input"
    return None


def select_response_mode(
    requested: str,
    user_input: str,
    history: Iterable[Any] = (),
    *,
    protected_physics: bool = False,
    has_attachment: bool = False,
) -> ResponseMode:
    if requested not in {"quick", "deep"}:
        raise ValueError("Response mode must be quick or deep.")
    if requested == "deep":
        return ResponseMode(requested, "deep", "Selected")
    if protected_physics:
        return ResponseMode(requested, "deep", "Physics validation")
    if has_attachment:
        return ResponseMode(requested, "deep", "Attached document")
    reason = _deep_reason(user_input)
    if reason:
        return ResponseMode(requested, "deep", reason)
    if _FOLLOW_UP.search(user_input.strip()):
        for turn in list(history)[-4:]:
            if _deep_reason(getattr(turn, "user", "")) or _deep_reason(getattr(turn, "assistant", "")):
                return ResponseMode(requested, "deep", "Analysis follow-up")
    return ResponseMode(requested, "quick", "Selected")
