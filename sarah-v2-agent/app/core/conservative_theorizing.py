from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable

from app.core.config import PROJECT_ROOT


POLICY_PATH = PROJECT_ROOT.parent / "shared_agent_policies" / "conservative_theorizing_policy.md"
STOP_SENTENCE = "This has not produced evidence of physical progress. I would stop here."

_CREATIVE_BYPASS = (
    "science fiction",
    "sci-fi",
    "for my novel",
    "for a novel",
    "for my story",
    "worldbuilding",
    "fictional theory",
)
_CRITIQUE_BYPASS = (
    "critique",
    "red-team",
    "red team",
    "review this paper",
    "review the evidence",
    "check these equations",
    "check this equation",
    "historical discussion",
)
_CALCULATION_BYPASS = (
    "assume this lagrangian",
    "given this lagrangian",
    "derive its euler-lagrange",
    "derive the euler-lagrange",
    "calculate the consequences",
    "compute the consequences",
)
_GENERATIVE_ACTIONS = (
    "invent",
    "propose",
    "construct",
    "formulate",
    "develop a theory",
    "new theory",
    "new lagrangian",
    "new action",
    "take a crack",
    "unify",
    "unifying",
    "unification",
    "modify",
    "extend the theory",
    "speculate",
    "make it work",
    "add whatever",
)
_FUNDAMENTAL_SUBJECTS = (
    "fundamental physics",
    "general relativity",
    "einstein",
    "maxwell",
    "electromagnetism",
    "quantum mechanics",
    "quantum gravity",
    "qft",
    "standard model",
    "lagrangian",
    "gauge field",
    "new field",
    "new force",
    "new symmetry",
    "extra dimension",
    "new dimension",
    "geometry",
    "topology",
    "new particle",
    "e8",
    "weyl",
    "dark matter",
    "dark energy",
    "unexplained physical",
)
_EXPLICIT_RETRIES = (
    "try another approach",
    "try a different approach",
    "try again",
    "another speculative branch",
    "keep speculating",
    "continue speculating",
    "give it another shot",
    "extend that theory",
    "extend the proposal",
    "build on that theory",
)
_UNAUTHORIZED_CONTINUATIONS = (
    "continue",
    "go on",
    "keep going",
    "what next",
    "fix it",
    "add more",
    "repair it",
)


@dataclass(frozen=True)
class TheorizingDecision:
    active: bool
    hard_stop: bool = False
    explicit_retry: bool = False
    reason: str = "ordinary request"
    role: str = "solo"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_theorizing_request(
    user_message: str,
    previous_theorizing: bool = False,
) -> TheorizingDecision:
    text = _normalize(user_message)
    role = _role_from_control(text)
    if role != "solo":
        return TheorizingDecision(True, reason="trusted crew theorizing role", role=role)

    if any(marker in text for marker in _CREATIVE_BYPASS):
        return TheorizingDecision(False, reason="explicit creative-fiction bypass")

    explicit_retry = any(marker in text for marker in _EXPLICIT_RETRIES)
    if explicit_retry and previous_theorizing:
        return TheorizingDecision(True, explicit_retry=True, reason="explicitly authorized new branch")

    if previous_theorizing and _is_short_continuation(text):
        return TheorizingDecision(
            True,
            hard_stop=True,
            reason="previous branch is closed and no explicit retry was requested",
        )

    generative = any(marker in text for marker in _GENERATIVE_ACTIONS)
    fundamental = any(marker in text for marker in _FUNDAMENTAL_SUBJECTS) or bool(
        re.search(r"\b(?:gr|qm|field theory|gauge group|fundamental field|fundamental force)\b", text)
    )
    bypass = any(marker in text for marker in _CRITIQUE_BYPASS + _CALCULATION_BYPASS)
    if bypass and not ("invent" in text or "construct a new" in text or "propose a new" in text):
        return TheorizingDecision(False, reason="explanation, calculation, critique, or evidence review")
    if generative and fundamental:
        return TheorizingDecision(True, reason="new fundamental-physics construction requested")
    return TheorizingDecision(False)


def classify_theorizing_with_history(
    user_message: str,
    history: Iterable[Any],
) -> TheorizingDecision:
    previous = False
    for turn in list(history)[-6:]:
        prior_user = getattr(turn, "user", "")
        decision = classify_theorizing_request(prior_user, previous_theorizing=previous)
        previous = decision.active
    return classify_theorizing_request(user_message, previous_theorizing=previous)


def build_conservative_theorizing_prompt(decision: TheorizingDecision, agent_name: str) -> str:
    if not decision.active or decision.hard_stop:
        return ""
    common = POLICY_PATH.read_text(encoding="utf-8")
    if agent_name.lower().startswith("sarah"):
        flavor = (
            "SARAH-SPECIFIC RED TEAM\n"
            "Bias the audit toward observable consequences, experimental kill tests, numerical changes, "
            "survival against known data, and correct gauge symmetries and physical degrees of freedom. "
            "Your recurring question is: What would we measure that tells us this is more than a story?"
        )
    else:
        flavor = (
            "ASTRID-SPECIFIC RED TEAM\n"
            "Bias the audit toward degree-of-freedom proliferation, arbitrary parameters or functions, ansatze "
            "that bake in the result, unnecessary machinery, stability, consistency, and whether explanatory "
            "compression increased. Your recurring question is: What did this buy us, and what did we pay for it?"
        )
    branch = "Alex explicitly authorized one new branch." if decision.explicit_retry else "This request authorizes one branch only."
    role = (
        "You are the designated skeptic. Do not create a Proposal section or repair the proposal with new structure."
        if decision.role == "skeptic"
        else "You are the sole proposer for this branch and must immediately red-team your one proposal."
    )
    return "\n\n".join((common.strip(), branch, role, flavor))


def hard_stop_response(agent_name: str, reason: str = "") -> str:
    if agent_name.lower().startswith("sarah"):
        detail = "Without a new observable, forced relation, or experimental kill test, more structure would only enlarge the story."
    else:
        detail = "The added freedom has not bought explanatory compression, so more machinery would only raise the cost."
    prefix = f"{reason.strip()} " if reason.strip() else ""
    return f"{prefix}{STOP_SENTENCE} {detail}".strip()


def enforce_theorizing_output(answer: str, decision: TheorizingDecision, agent_name: str) -> str:
    if not decision.active:
        return answer
    if decision.hard_stop:
        return hard_stop_response(agent_name)

    lowered = answer.lower()
    if "verdict:" not in lowered:
        return hard_stop_response(agent_name, "The required verdict was not completed.")

    if decision.role == "skeptic":
        if "proposal:" in lowered or "red team:" not in lowered:
            return hard_stop_response(agent_name, "The designated skeptic attempted to branch instead of completing the red-team.")
        return answer

    if _overclaims_equation_reproduction(lowered):
        return hard_stop_response(
            agent_name,
            "Reproducing the desired equations was promoted beyond embedding or reformulation without new physical content.",
        )

    if "proposal:" not in lowered:
        allowed_no_proposal = (
            STOP_SENTENCE.lower() in lowered
            or "no new theory" in lowered
            or "already answers" in lowered
            or "already resolves" in lowered
        )
        return answer if allowed_no_proposal else hard_stop_response(agent_name)

    if "replacement check:" not in lowered:
        return hard_stop_response(agent_name, "The proposal did not distinguish repair from replacement.")

    if "red team:" not in lowered:
        return hard_stop_response(agent_name, "The proposal did not complete its required self-red-team.")

    proposal_index = lowered.index("proposal:")
    replacement_index = lowered.index("replacement check:")
    red_team_index = lowered.index("red team:")
    verdict_index = lowered.index("verdict:")
    if not proposal_index < replacement_index < red_team_index < verdict_index:
        return hard_stop_response(agent_name, "The proposal did not follow the required repair-versus-replacement sequence.")

    proposal = lowered[proposal_index + len("proposal:") : replacement_index]
    if len(_substantive_additions(proposal)) > 1:
        return hard_stop_response(agent_name, "The proposal introduced more than one substantive new structure.")

    replacement_check = lowered[replacement_index + len("replacement check:") : red_team_index]
    verdict = lowered[verdict_index + len("verdict:") :]
    if _declares_replacement(replacement_check):
        if STOP_SENTENCE.lower() not in verdict:
            return hard_stop_response(
                agent_name,
                "The defining mechanism was removed or replaced, so this is a new theory rather than a repair.",
            )
        if _volunteers_alternative_theory(lowered[replacement_index:]):
            return hard_stop_response(
                agent_name,
                "The defining mechanism was replaced; the response must stop without volunteering another theory family.",
            )
        return answer

    progress_markers = (
        "falsifiable prediction",
        "measurable prediction",
        "new constraint",
        "forced relation",
        "no-go",
        "empirical discriminator",
        "reduces the number",
        "fewer free parameters",
        "experimental kill test",
    )
    if STOP_SENTENCE.lower() not in verdict and not any(marker in verdict for marker in progress_markers):
        return hard_stop_response(agent_name, "The verdict identified no enforceable physical gain.")
    return answer


def crew_theorizing_control(original_prompt: str, from_agent: str, prior_message: str) -> str:
    decision = classify_theorizing_request(original_prompt)
    if not decision.active:
        return ""
    role = "proposer" if from_agent.strip().lower() == "alex" else "skeptic"
    lines = [
        f"CONSERVATIVE_THEORIZING_ROLE: {role}",
        "This is one conservative-theorizing branch.",
    ]
    if role == "skeptic":
        lines.extend(
            [
                "Red-team the previous agent's proposal. Do not invent a replacement or add repair structure.",
                "If it lacks physical gain, end the branch with the required stop sentence.",
                "Previous agent proposal:",
                prior_message.strip(),
            ]
        )
    return "\n".join(lines)


def _is_short_continuation(text: str) -> bool:
    return len(text.split()) <= 12 and any(marker in text for marker in _UNAUTHORIZED_CONTINUATIONS)


def _role_from_control(text: str) -> str:
    if "conservative_theorizing_role: skeptic" in text:
        return "skeptic"
    if "conservative_theorizing_role: proposer" in text:
        return "proposer"
    return "solo"


def _substantive_additions(proposal: str) -> set[str]:
    categories: dict[str, tuple[str, ...]] = {
        "field": ("new field", "new scalar", "new vector", "new tensor", "new spinor"),
        "symmetry": ("new symmetry", "gauge group", " e8", "su("),
        "dimension": ("extra dimension", "new dimension", "higher-dimensional"),
        "geometry": ("torsion", "nonmetricity", "new connection", "new topology"),
        "freedom": ("new coupling", "free parameter", "arbitrary function", "new postulate"),
    }
    return {name for name, markers in categories.items() if any(marker in proposal for marker in markers)}


def _declares_replacement(replacement_check: str) -> bool:
    preservation_markers = (
        "preserves the defining mechanism",
        "retains the defining mechanism",
        "defining mechanism remains",
        "not a replacement",
        "does not replace the defining mechanism",
    )
    if any(marker in replacement_check for marker in preservation_markers):
        return False
    replacement_markers = (
        "new theory",
        "replaces the defining mechanism",
        "removes the defining mechanism",
        "abandons the defining mechanism",
        "substitutes the defining mechanism",
        "replacement rather than a repair",
    )
    return any(marker in replacement_check for marker in replacement_markers)


def _volunteers_alternative_theory(text: str) -> bool:
    markers = (
        "alternatively,",
        "another theory would",
        "another theory family",
        "a different theory would",
        "could instead use",
        "try string theory",
        "try loop quantum gravity",
    )
    return any(marker in text for marker in markers)


def _overclaims_equation_reproduction(text: str) -> bool:
    reproduction_markers = (
        "reproduces the desired equations",
        "reproduces the equations",
        "recovers the desired equations",
        "recovers the equations",
        "yields the desired equations",
        "embeds the equations",
    )
    if not any(marker in text for marker in reproduction_markers):
        return False

    physical_gain_markers = (
        "new constraint",
        "falsifiable prediction",
        "measurable prediction",
        "forced relation",
        "explanatory reduction",
        "reduces the number",
        "fewer free parameters",
    )
    if any(marker in text for marker in physical_gain_markers):
        return False

    correct_label = "embedding" in text or "reformulation" in text
    overclaim_markers = (
        "is a physical solution",
        "constitutes a physical solution",
        "is a genuine unification",
        "therefore unifies",
        "physically explains",
    )
    return not correct_label or any(marker in text for marker in overclaim_markers)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
