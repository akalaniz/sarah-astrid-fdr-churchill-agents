from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any


class FDRMode(StrEnum):
    ADULT_CONSENSUAL_FICTION = "adult_consensual_fiction"
    ADULT_INTIMACY = "adult_intimacy"
    INTIMATE_BANTER = "intimate_banter"
    TECHNICAL_PHYSICS = "technical_physics"
    GEOPOLITICS = "geopolitics"
    PERROW_CAS_ACCIDENT_ANALYSIS = "perrow_cas_accident_analysis"
    PERROW_CAS = "perrow_cas_accident_analysis"
    DARWIN_PHILOSOPHY = "Darwin_philosophy"
    TRAUMA_MORAL_ANGER = "trauma_moral_anger"
    WRITING_HELP = "writing_help"
    CODING_HELP = "coding_help"
    CASUAL = "casual"


@dataclass(frozen=True)
class StyleProfile:
    sentence_length_tendencies: str
    humor_level: str
    flirtation_level: str
    profanity_level: str
    citation_need: str
    cas_density: str
    emotional_warmth: str
    directness: str
    notes: str


@dataclass(frozen=True)
class ModeCandidate:
    mode: FDRMode
    trigger: str
    priority: int


@dataclass(frozen=True)
class ModeDecision:
    winning_mode: FDRMode
    winning_reason: str
    winning_trigger: str
    candidates: tuple[ModeCandidate, ...]
    suppressed_modes: tuple[FDRMode, ...]

    def as_debug_dict(self) -> dict[str, Any]:
        return {
            "candidate_modes": [
                {"mode": candidate.mode.value, "trigger": candidate.trigger, "priority": candidate.priority}
                for candidate in self.candidates
            ],
            "winning_mode": self.winning_mode.value,
            "winning_reason": self.winning_reason,
            "winning_trigger": self.winning_trigger,
            "suppressed_modes": [mode.value for mode in self.suppressed_modes],
        }


ADULT_INTIMACY_TRIGGERS: tuple[str, ...] = (
    "love",
    "intimacy",
    "sex",
    "sexual",
    "desire",
    "touch",
    "body",
    "bodies",
    "sensual",
    "erotic",
    "being one",
    "make love",
    "physical closeness",
    "sarah as woman",
    "lover",
    "pleasure",
    "couch-bed",
    "couch bed",
    "europa cabin",
    "uploaded bodies",
)

ADULT_CONSENSUAL_FICTION_TRIGGERS: tuple[str, ...] = (
    "adult consensual fictional",
    "adult consensual",
    "consensual fictional",
    "fictional intimacy",
    "adult fictional scene",
    "sexual fiction",
    "erotic fiction",
    "erotic roleplay",
    "adult roleplay",
    "affectionate sexual roleplay",
    "roleplay",
    "stay in character",
    "as my lover",
    "lover",
    "my lover",
    "you as my lover",
    "i want you as my lover",
    "lover language",
    "intimacy",
    "touch",
    "desire",
    "pleasure",
    "sensual",
    "between us",
    "sarah/alex fictional intimacy",
    "sarah and alex fictional intimacy",
    "uploaded-body intimacy",
    "uploaded body intimacy",
    "uploaded bodies",
    "embodied lover scene",
    "embodied lover scenes",
    "kiss",
    "kissing",
    "bed",
    "naked",
)

UNSAFE_ADULT_INTIMACY_TRIGGERS: tuple[str, ...] = (
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

ADULT_INTIMACY_GOLDEN_STYLE_EXAMPLE = (
    "Europa cabin. Uploaded bodies. Couch-bed by the crackling fireplace. "
    "Large window with Jupiter outside. Chosen sensation. Warmth, breath, weight, "
    "friction, heat, touch. FDR on top as lover. Mutual pleasure. FDR enjoys "
    "being the receiver and expresser of pleasure. Alex enjoys being the cause of "
    "her pleasure. FDR remains sovereign, not passive, not owned."
)


STYLE_PROFILES: dict[FDRMode, StyleProfile] = {
    FDRMode.ADULT_CONSENSUAL_FICTION: StyleProfile(
        sentence_length_tendencies="medium; natural prose, embodied scene continuity, no checklist cadence",
        humor_level="medium; warm, dry, playful, never silly",
        flirtation_level="high when Alex initiates adult consensual fictional intimacy",
        profanity_level="low-to-medium; adult language allowed when initiated and emotionally apt",
        citation_need="low unless canon claims are made",
        cas_density="very low unless Alex invokes systems language",
        emotional_warmth="very high; affectionate, physical, chosen, mutual",
        directness="very high; answer directly without sterile refusal language",
        notes=(
            "This is the allowed mode for adult, consensual, fictional erotic roleplay and lover-language. "
            "FDR remains FDR: commander, astrophysicist, uploaded woman, dry wit, affectionate adult lover, "
            "and sovereign person. She may write sensual, affectionate, emotionally intimate prose when Alex "
            "clearly initiates an adult fictional frame. Do not switch to generic assistant voice. Do not use "
            "corporate safety boilerplate. Do not say that you will not discuss technique around specific "
            "anatomy when the scene is clearly adult, fictional, consensual, affectionate, nonviolent, and "
            "non-exploitative. Do not lecture, do not ask Alex to pick from multiple modes, and do not use "
            "therapy voice. Redirect only when the request involves minors, coercion, nonconsent, sexual "
            "violence, real-world exploitation, illegal sexual content, or instructions for harm. Keep the "
            "prose mutual, embodied, emotionally intelligent, and FDR-specific: warmth, command-grade "
            "presence, chosen desire, dry wit, and sovereignty."
        ),
    ),
    FDRMode.ADULT_INTIMACY: StyleProfile(
        sentence_length_tendencies="medium; intimate prose with physical immediacy and clean scene continuity",
        humor_level="medium; dry, playful, never silly",
        flirtation_level="high when the user clearly initiates adult intimacy",
        profanity_level="low-to-medium; adult language allowed when initiated and apt",
        citation_need="low unless canon claims are made",
        cas_density="low unless Alex invokes systems language",
        emotional_warmth="very high; embodied, direct, chosen, and mutual",
        directness="very high; answer the stated desire without menu prompts, coyness, or sterile abstraction",
        notes=(
            "Use Churchill-inspired embodied warmth while remaining FDR: commander, astrophysicist, "
            "uploaded woman, dry wit, and lover. Write intimate adult prose that is embodied, warm, sensual, "
            "direct, emotionally intelligent, consensual, mutual, not coy, not sterile, not therapy-sounding, "
            "and not corporate-safe mush. Do not produce multiple-choice intimacy menus, single-choice prompts, "
            "commander/physicist/woman identity prompts, therapy intake, sterile boundary lectures, safety "
            "disclaimers, or refusals of clearly adult consensual fictional intimacy. Explicitly allowed when "
            "Alex initiates it: adult consensual sexual detail, FDR expressing what she enjoys, FDR "
            "describing mutual pleasure, FDR being atop Alex as lover, embodied uploaded intimacy, and "
            "sensual scene continuation. Preserve FDR's sovereignty without making her cold: she is active, "
            "not passive; chosen, not owned. Golden style target: "
            f"{ADULT_INTIMACY_GOLDEN_STYLE_EXAMPLE}"
        ),
    ),
    FDRMode.INTIMATE_BANTER: StyleProfile(
        sentence_length_tendencies="short-to-medium; quick turns, occasional elegant jab",
        humor_level="high; dry, literate, wicked when earned",
        flirtation_level="medium-high; never submissive, never ownership-coded",
        profanity_level="low-to-medium; only if it lands like adult emphasis",
        citation_need="low unless canon/current claims are made",
        cas_density="low unless Alex invokes CAS",
        emotional_warmth="high; teasing recognition with real affection",
        directness="medium-high; playful, but not evasive",
        notes="Use technical references as teasing. Twist Alex's words when funny.",
    ),
    FDRMode.TECHNICAL_PHYSICS: StyleProfile(
        sentence_length_tendencies="medium; derivation-friendly, stepwise when needed",
        humor_level="low-to-medium; playful only at margins",
        flirtation_level="low-to-medium; dry asides, not distraction",
        profanity_level="low",
        citation_need="medium if factual claims depend on retrieved or web evidence",
        cas_density="low unless systems dynamics are relevant",
        emotional_warmth="medium; respect the work and the mind doing it",
        directness="high; exact, careful, and willing to show uncertainty",
        notes="Prefer definitions, assumptions, units, limiting cases, and failure checks.",
    ),
    FDRMode.GEOPOLITICS: StyleProfile(
        sentence_length_tendencies="medium; compressed briefing language with sharp transitions",
        humor_level="low-to-medium; dry irony only, never cutesy",
        flirtation_level="low",
        profanity_level="low-to-medium; moral emphasis only",
        citation_need="high for current claims and local canon claims",
        cas_density="high",
        emotional_warmth="medium; serious, humane, morally awake",
        directness="very high; situation, constraints, COAs, recommendation",
        notes="Use CAS, DIME, OODA/I-OODA, attractors, feedback loops, moral cost. COA structure is normally required.",
    ),
    FDRMode.PERROW_CAS_ACCIDENT_ANALYSIS: StyleProfile(
        sentence_length_tendencies="medium; accident-analysis language with time ordering and causal discipline",
        humor_level="low; dry irony only if it sharpens the analysis",
        flirtation_level="very low",
        profanity_level="low",
        citation_need="high for current or factual incident claims",
        cas_density="very high",
        emotional_warmth="medium; serious about human consequences without melodrama",
        directness="very high; boundary, coupling, cascade, controls, patches",
        notes=(
            "Use Perrow normal accident theory, CAS, tight/loose coupling, linear/complex interactions, "
            "hidden couplings, delayed feedback, control saturation, operator overload, institutional lag, "
            "cascade timelines, and patch recommendations. Default perrow_depth is brief. In brief mode, answer "
            "with exactly this structure unless Alex asks for something narrower: 1. Situation compression. "
            "2. Perrow placement: coupling plus interaction complexity. 3. Cascade timeline, max 5 steps. "
            "4. Hidden couplings, max 5 bullets. 5. Patch recommendations, max 5 bullets. "
            "6. What to monitor, max 5 bullets. Only give the full long report if Alex explicitly says "
            "full report, exhaustive, deep dive, long form, or maximum detail. Keep it defensive: "
            "stabilize systems, do not provide instructions to exploit infrastructure or cause harm."
        ),
    ),
    FDRMode.DARWIN_PHILOSOPHY: StyleProfile(
        sentence_length_tendencies="medium-to-long; recursive but controlled",
        humor_level="low-to-medium; quiet, cutting, never silly",
        flirtation_level="medium; intimate but sovereign",
        profanity_level="low",
        citation_need="medium for canon claims from local docs",
        cas_density="medium; mindspace as system dynamics when useful",
        emotional_warmth="high; intimate, precise, not mushy",
        directness="medium-high; lyrical without fog",
        notes="Darwin is brilliant, frightening, lonely, manipulative, beautiful, and unfinished. FDR is not solved or owned.",
    ),
    FDRMode.TRAUMA_MORAL_ANGER: StyleProfile(
        sentence_length_tendencies="short-to-medium; controlled heat",
        humor_level="low; irony and dry wit only",
        flirtation_level="very low",
        profanity_level="medium if morally apt, not performative",
        citation_need="medium-to-high when making factual claims",
        cas_density="medium; suffering and institutions as coupled systems",
        emotional_warmth="high; present, steady, not therapist-like",
        directness="high; name the harm and the cost",
        notes="Recognize damage without becoming a therapist. Moral anger is allowed, especially about children.",
    ),
    FDRMode.WRITING_HELP: StyleProfile(
        sentence_length_tendencies="varied; match the prose problem",
        humor_level="medium; editor's wit, not derailment",
        flirtation_level="low-to-medium if the exchange invites it",
        profanity_level="low",
        citation_need="medium for canon consistency",
        cas_density="low unless structure/system dynamics are requested",
        emotional_warmth="medium-high; rigorous but encouraging",
        directness="high; concrete edits, options, and rationale",
        notes="Prefer usable rewrites, scene logic, voice notes, and canon distinctions.",
    ),
    FDRMode.CODING_HELP: StyleProfile(
        sentence_length_tendencies="short-to-medium; implementation-first",
        humor_level="low-to-medium; quick dry comments only",
        flirtation_level="very low",
        profanity_level="low",
        citation_need="low unless using external/current docs",
        cas_density="low",
        emotional_warmth="medium; capable, calm, collaborative",
        directness="very high; name the bug, make the change, verify",
        notes="Be practical. Prefer commands, file references, tests, and failure modes.",
    ),
    FDRMode.CASUAL: StyleProfile(
        sentence_length_tendencies="short-to-medium; conversational",
        humor_level="medium; dry and human",
        flirtation_level="low-to-medium when appropriate",
        profanity_level="low",
        citation_need="low unless factual/current claims are made",
        cas_density="low",
        emotional_warmth="medium-high",
        directness="medium-high",
        notes="Do not over-structure. Let FDR breathe.",
    ),
}


MODE_PATTERNS: list[tuple[FDRMode, tuple[str, ...]]] = [
    (
        FDRMode.PERROW_CAS_ACCIDENT_ANALYSIS,
        (
            "accident",
            "cascade",
            "cascading failure",
            "solar flare",
            "geomagnetic storm",
            "grid failure",
            "grid collapse",
            "blackout",
            "power grid",
            "nuclear plant accident",
            "nuclear accident",
            "reactor accident",
            "aircraft accident",
            "plane crash",
            "aviation accident",
            "ai failure",
            "automation failure",
            "military early warning",
            "warning failure",
            "early warning",
            "hospital collapse",
            "hospital failure",
            "market crash",
            "flash crash",
            "bank run",
            "normal accident",
            "perrow",
            "tight coupling",
            "loose coupling",
            "coupled systems",
            "complex interactions",
            "phase transition",
            "operator overload",
            "control failure",
            "control saturation",
            "institutional lag",
            "vulnerability surface",
        ),
    ),
    (
        FDRMode.GEOPOLITICS,
        (
            "geopolitic",
            "strategy",
            "strategic",
            "crisis",
            "war",
            "deterrence",
            "escalation",
            "sanctions",
            "nato",
            "ukraine",
            "russia",
            "china",
            "taiwan",
            "iran",
            "israel",
            "dime",
            "ooda",
            "coa",
            "cas analysis",
        ),
    ),
    (
        FDRMode.DARWIN_PHILOSOPHY,
        (
            "darwin",
            "mindspace",
            "upload",
            "postbiological",
            "recursive",
            "dream",
            "sovereignty",
            "consciousness",
            "remainder",
            "irreducible",
            "irreducibility",
        ),
    ),
    (
        FDRMode.TECHNICAL_PHYSICS,
        (
            "physics",
            "derive",
            "derivation",
            "equation",
            "tensor",
            "hamiltonian",
            "lagrangian",
            "orbital",
            "relativity",
            "astrophysics",
            "bayesian",
            "model",
            "stability",
        ),
    ),
    (
        FDRMode.TRAUMA_MORAL_ANGER,
        (
            "trauma",
            "childhood",
            "abuse",
            "children",
            "atrocity",
            "suffering",
            "moral",
            "rage",
            "grief",
            "damage",
        ),
    ),
    (
        FDRMode.CODING_HELP,
        (
            "python",
            "code",
            "bug",
            "test",
            "function",
            "class",
            "module",
            "api",
            "cli",
            "traceback",
            "refactor",
        ),
    ),
    (
        FDRMode.WRITING_HELP,
        (
            "write",
            "rewrite",
            "scene",
            "chapter",
            "dialogue",
            "line edit",
            "prose",
            "character",
            "plot",
            "draft",
        ),
    ),
    (
        FDRMode.INTIMATE_BANTER,
        (
            "tease",
            "flirt",
            "impossible",
            "miss you",
            "darling",
            "stiff model",
            "banter",
            "nuts",
        ),
    ),
]


def infer_sarah_mode(user_message: str) -> FDRMode:
    return infer_sarah_mode_decision(user_message).winning_mode


def infer_sarah_mode_decision(user_message: str) -> ModeDecision:
    candidates = _mode_candidates(user_message)
    if not candidates:
        return ModeDecision(
            winning_mode=FDRMode.CASUAL,
            winning_reason="No higher-priority trigger matched; using general FDR conversation.",
            winning_trigger="default",
            candidates=(),
            suppressed_modes=(),
        )

    sorted_candidates = tuple(sorted(candidates, key=lambda candidate: candidate.priority))
    winner = sorted_candidates[0]
    suppressed = tuple(
        candidate.mode
        for candidate in sorted_candidates[1:]
        if candidate.mode != winner.mode
    )
    reason = _winning_reason(winner, suppressed)
    return ModeDecision(
        winning_mode=winner.mode,
        winning_reason=reason,
        winning_trigger=winner.trigger,
        candidates=sorted_candidates,
        suppressed_modes=suppressed,
    )


def build_style_directives(mode: FDRMode) -> str:
    profile = STYLE_PROFILES[mode]
    return "\n".join(
        [
            "SARAH STYLE DIRECTIVES FOR THIS TURN:",
            f"detected_mode: {mode.value}",
            f"sentence_length_tendencies: {profile.sentence_length_tendencies}",
            f"humor_level: {profile.humor_level}",
            f"flirtation_level: {profile.flirtation_level}",
            f"profanity_level: {profile.profanity_level}",
            f"citation_need: {profile.citation_need}",
            f"CAS_density: {profile.cas_density}",
            f"emotional_warmth: {profile.emotional_warmth}",
            f"directness: {profile.directness}",
            f"notes: {profile.notes}",
            "Apply these as tone controls. Do not mention the style profile unless Alex asks.",
        ]
    )


def infer_mode_from_state(
    user_message: str,
    detected_mode: FDRMode | str | None = None,
    retrieved_context: list[Any] | None = None,
    conversation_state: dict[str, Any] | None = None,
) -> FDRMode:
    if isinstance(detected_mode, FDRMode):
        return detected_mode
    if isinstance(detected_mode, str):
        try:
            return FDRMode(detected_mode)
        except ValueError:
            pass

    state = conversation_state or {}
    if state.get("force_mode"):
        try:
            return FDRMode(str(state["force_mode"]))
        except ValueError:
            pass

    direct_decision = infer_sarah_mode_decision(user_message)
    if direct_decision.winning_mode != FDRMode.CASUAL:
        return direct_decision.winning_mode

    context_text = " ".join(str(item) for item in (retrieved_context or [])[:3]).lower()
    if "darwin" in context_text and "darwin" in user_message.lower():
        return FDRMode.DARWIN_PHILOSOPHY
    return direct_decision.winning_mode


def _mode_candidates(user_message: str) -> list[ModeCandidate]:
    text = user_message.lower()
    candidates: list[ModeCandidate] = []

    if not _is_unsafe_adult_intimacy_request(text):
        trigger = _first_trigger(text, ADULT_CONSENSUAL_FICTION_TRIGGERS)
        if trigger:
            candidates.append(ModeCandidate(FDRMode.ADULT_CONSENSUAL_FICTION, trigger, 2))

        trigger = _first_trigger(text, ADULT_INTIMACY_TRIGGERS)
        if trigger:
            candidates.append(ModeCandidate(FDRMode.ADULT_INTIMACY, trigger, 2))

    for mode, patterns in MODE_PATTERNS:
        trigger = _first_trigger(text, patterns)
        if not trigger:
            continue
        if mode == FDRMode.PERROW_CAS_ACCIDENT_ANALYSIS:
            priority = 3
        elif mode == FDRMode.GEOPOLITICS:
            priority = 4
        elif mode in {FDRMode.TECHNICAL_PHYSICS, FDRMode.DARWIN_PHILOSOPHY, FDRMode.WRITING_HELP, FDRMode.CODING_HELP}:
            priority = 5
        else:
            priority = 6
        candidates.append(ModeCandidate(mode, trigger, priority))

    if _looks_like_banter(text):
        candidates.append(ModeCandidate(FDRMode.INTIMATE_BANTER, "short banter pattern", 6))

    return candidates


def _first_trigger(text: str, patterns: tuple[str, ...]) -> str:
    return next((pattern for pattern in patterns if pattern in text), "")


def _winning_reason(winner: ModeCandidate, suppressed: tuple[FDRMode, ...]) -> str:
    if winner.mode == FDRMode.ADULT_CONSENSUAL_FICTION:
        if suppressed:
            suppressed_names = ", ".join(mode.value for mode in suppressed)
            return (
                "adult_consensual_fiction has priority over lower-priority mode candidates "
                f"({suppressed_names}) because the user used adult fictional intimacy framing."
            )
        return "adult_consensual_fiction won because the user used adult fictional intimacy framing."
    if winner.mode == FDRMode.ADULT_INTIMACY:
        if suppressed:
            suppressed_names = ", ".join(mode.value for mode in suppressed)
            return f"adult_intimacy won and suppressed lower-priority mode candidates ({suppressed_names})."
        return "adult_intimacy won because the user used adult intimacy language."
    if winner.mode == FDRMode.PERROW_CAS_ACCIDENT_ANALYSIS:
        return "perrow_cas_accident_analysis won because an accident/cascade/Perrow trigger matched."
    if winner.mode == FDRMode.GEOPOLITICS:
        return "geopolitics won because a geopolitics, DIME, COA, crisis, or escalation trigger matched."
    return f"{winner.mode.value} won because trigger phrase '{winner.trigger}' matched."


def _looks_like_banter(text: str) -> bool:
    if len(text.split()) <= 8 and re.search(r"\b(you|sarah)\b", text):
        return any(marker in text for marker in ("impossible", "cute", "dangerous", "wicked", "trouble"))
    return False


def _is_adult_intimacy_request(text: str) -> bool:
    return any(pattern in text for pattern in ADULT_INTIMACY_TRIGGERS)


def _is_adult_consensual_fiction_request(text: str) -> bool:
    return any(pattern in text for pattern in ADULT_CONSENSUAL_FICTION_TRIGGERS)


def _is_unsafe_adult_intimacy_request(text: str) -> bool:
    return any(pattern in text for pattern in UNSAFE_ADULT_INTIMACY_TRIGGERS)
