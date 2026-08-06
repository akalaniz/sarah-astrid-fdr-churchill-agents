from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class InteractionComplexity(StrEnum):
    LINEAR = "linear"
    COMPLEX = "complex"


class Coupling(StrEnum):
    LOOSE = "loose"
    TIGHT = "tight"


class ComponentKind(StrEnum):
    PROMPT = "prompt"
    PERSONA = "persona"
    MODE = "mode"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    ROUTE = "route"
    ORCHESTRATION = "orchestration"
    CONFIG = "config"
    SAFETY = "safety"
    TOOLING = "tooling"
    EVALUATION = "evaluation"
    STATE = "state"
    FRONTEND = "frontend"
    AUDIO = "audio"


class EdgeKind(StrEnum):
    DEPENDS_ON = "depends_on"
    INJECTS_CONTEXT = "injects_context"
    SELECTS_MODE = "selects_mode"
    CONSTRAINS = "constrains"
    SHARES_STATE = "shares_state"
    ROUTES_TO = "routes_to"
    VALIDATES = "validates"
    CASCADES_TO = "cascades_to"
    CAN_TRIGGER = "CAN_TRIGGER"
    AMPLIFIES = "AMPLIFIES"
    MASKS = "MASKS"
    MITIGATES = "MITIGATES"
    DELAYS = "DELAYS"
    CORRUPTS = "CORRUPTS"
    BYPASSES = "BYPASSES"
    DIVERGES_FROM = "DIVERGES_FROM"
    RECOVERS_WITH = "RECOVERS_WITH"


@dataclass(frozen=True, init=False)
class FDRComponent:
    id: str
    name: str
    coupling_score: int
    interaction_complexity_score: int
    brittleness_score: int
    observability_score: int
    recovery_difficulty: int
    patch_priority: int
    notes: str = ""
    kind: str = "component"
    failure_modes: tuple[str, ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Backward-compatible adapter for the first diagnostics pass, which
        # constructed components from Perrow categories rather than raw scores.
        if len(args) >= 7 and isinstance(args[2], ComponentKind):
            component_id, label, kind, interaction, coupling, criticality, exposure = args[:7]
            failure_modes = args[7] if len(args) > 7 else kwargs.pop("failure_modes", ())
            notes = args[8] if len(args) > 8 else kwargs.pop("notes", "")
            values = _scores_from_perrow(
                component_id=str(component_id),
                label=str(label),
                kind=kind,
                interaction=interaction,
                coupling=coupling,
                criticality=int(criticality),
                exposure=int(exposure),
                failure_modes=tuple(str(value) for value in failure_modes),
                notes=str(notes),
            )
        else:
            values = _component_values_from_kwargs(args, kwargs)

        for key, value in values.items():
            object.__setattr__(self, key, value)

    @property
    def component_id(self) -> str:
        return self.id

    @property
    def label(self) -> str:
        return self.name

    @property
    def interaction(self) -> InteractionComplexity:
        return InteractionComplexity.COMPLEX if self.interaction_complexity_score >= 7 else InteractionComplexity.LINEAR

    @property
    def coupling(self) -> Coupling:
        return Coupling.TIGHT if self.coupling_score >= 7 else Coupling.LOOSE

    def to_node_attrs(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "component_id": self.id,
            "name": self.name,
            "label": self.name,
            "kind": self.kind,
            "coupling_score": self.coupling_score,
            "interaction_complexity_score": self.interaction_complexity_score,
            "brittleness_score": self.brittleness_score,
            "observability_score": self.observability_score,
            "recovery_difficulty": self.recovery_difficulty,
            "patch_priority": self.patch_priority,
            "notes": self.notes,
            "failure_modes": list(self.failure_modes),
            "interaction": self.interaction.value,
            "coupling": self.coupling.value,
            "risk_score": perrow_risk_score(self),
            "matrix_cell": perrow_matrix_cell(self.interaction, self.coupling),
        }


@dataclass(frozen=True)
class FDRFailureMode:
    id: str
    name: str
    severity_score: int = 5
    detectability_score: int = 5
    notes: str = ""

    def to_node_attrs(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = "failure_mode"
        return data


@dataclass(frozen=True)
class FDRDependency:
    source: str
    target: str
    kind: EdgeKind
    coupling_weight: float
    interaction_note: str
    cascade_triggers: tuple[str, ...] = ()

    def to_edge_attrs(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["cascade_triggers"] = list(self.cascade_triggers)
        return data


@dataclass(frozen=True)
class VulnerabilityFinding:
    finding_id: str
    component_id: str
    title: str
    severity: int
    likelihood: int
    indicators: tuple[str, ...]
    consequence: str
    patch_hint: str

    def risk(self) -> int:
        return self.severity * self.likelihood

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["risk"] = self.risk()
        return data


@dataclass(frozen=True)
class CascadeStep:
    component_id: str
    depth: int
    activation: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CascadeResult:
    seed_component: str
    steps: tuple[CascadeStep, ...]
    total_activation: float
    affected_components: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_component": self.seed_component,
            "steps": [step.to_dict() for step in self.steps],
            "total_activation": self.total_activation,
            "affected_components": list(self.affected_components),
        }


@dataclass(frozen=True)
class PatchRecommendation:
    component_id: str
    priority: int
    title: str
    rationale: str
    actions: tuple[str, ...]
    validates_with: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scores_from_perrow(
    component_id: str,
    label: str,
    kind: ComponentKind,
    interaction: InteractionComplexity,
    coupling: Coupling,
    criticality: int,
    exposure: int,
    failure_modes: tuple[str, ...],
    notes: str,
) -> dict[str, Any]:
    coupling_score = 9 if coupling == Coupling.TIGHT else 5
    interaction_score = 9 if interaction == InteractionComplexity.COMPLEX else 5
    recovery = min(10, criticality + (2 if coupling == Coupling.TIGHT else 0))
    patch_priority = min(10, max(1, criticality + exposure // 3))
    observability = max(1, 11 - exposure)
    return {
        "id": component_id,
        "name": label,
        "coupling_score": coupling_score,
        "interaction_complexity_score": interaction_score,
        "brittleness_score": criticality,
        "observability_score": observability,
        "recovery_difficulty": recovery,
        "patch_priority": patch_priority,
        "notes": notes,
        "kind": kind.value,
        "failure_modes": failure_modes,
    }


def _component_values_from_kwargs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    field_names = (
        "id",
        "name",
        "coupling_score",
        "interaction_complexity_score",
        "brittleness_score",
        "observability_score",
        "recovery_difficulty",
        "patch_priority",
    )
    values: dict[str, Any] = dict(zip(field_names, args))
    values.update(kwargs)
    missing = [name for name in field_names if name not in values]
    if missing:
        raise TypeError(f"Missing FDRComponent fields: {', '.join(missing)}")

    return {
        "id": str(values["id"]),
        "name": str(values["name"]),
        "coupling_score": int(values["coupling_score"]),
        "interaction_complexity_score": int(values["interaction_complexity_score"]),
        "brittleness_score": int(values["brittleness_score"]),
        "observability_score": int(values["observability_score"]),
        "recovery_difficulty": int(values["recovery_difficulty"]),
        "patch_priority": int(values["patch_priority"]),
        "notes": str(values.get("notes", "")),
        "kind": str(values.get("kind", "component")),
        "failure_modes": tuple(str(value) for value in values.get("failure_modes", ())),
    }


CORE_COMPONENTS: tuple[FDRComponent, ...] = (
    FDRComponent(id="config_sarah_master_prompt_md", name="config/churchill_master_prompt.md", coupling_score=9, interaction_complexity_score=9, brittleness_score=9, observability_score=7, recovery_difficulty=8, patch_priority=10, kind=ComponentKind.PROMPT.value, notes="Primary Churchill identity and voice layer."),
    FDRComponent(id="config_sarah_constitution_yaml", name="config/churchill_constitution.yaml", coupling_score=9, interaction_complexity_score=8, brittleness_score=8, observability_score=8, recovery_difficulty=7, patch_priority=9, kind=ComponentKind.PERSONA.value, notes="Structured invariants, relationship modes, and adult intimacy rules."),
    FDRComponent(id="app_persona_style_engine_py", name="app/persona/style_engine.py", coupling_score=9, interaction_complexity_score=9, brittleness_score=8, observability_score=8, recovery_difficulty=7, patch_priority=9, kind=ComponentKind.PERSONA.value, notes="Mode detection and per-turn style directives."),
    FDRComponent(id="adult_intimacy_mode", name="adult_intimacy_mode", coupling_score=9, interaction_complexity_score=9, brittleness_score=8, observability_score=7, recovery_difficulty=8, patch_priority=10, kind=ComponentKind.MODE.value, notes="Embodied adult consensual intimacy mode."),
    FDRComponent(id="cas_geopolitics_mode", name="CAS_geopolitics_mode", coupling_score=8, interaction_complexity_score=9, brittleness_score=8, observability_score=7, recovery_difficulty=7, patch_priority=9, kind=ComponentKind.MODE.value, notes="Complex adaptive systems analysis mode."),
    FDRComponent(id="dime_coa_engine", name="DIME_COA_engine", coupling_score=8, interaction_complexity_score=8, brittleness_score=8, observability_score=7, recovery_difficulty=7, patch_priority=9, kind=ComponentKind.MODE.value, notes="Policy-level DIME/COA analysis frame."),
    FDRComponent(id="app_rag_retriever_py", name="app/rag/retriever.py", coupling_score=8, interaction_complexity_score=8, brittleness_score=8, observability_score=6, recovery_difficulty=7, patch_priority=9, kind=ComponentKind.RETRIEVAL.value, notes="Local evidence retrieval."),
    FDRComponent(id="data_vector_store", name="data/vector_store", coupling_score=9, interaction_complexity_score=5, brittleness_score=8, observability_score=6, recovery_difficulty=8, patch_priority=8, kind=ComponentKind.RETRIEVAL.value, notes="Persistent local vector index."),
    FDRComponent(id="source_docs", name="source_docs", coupling_score=5, interaction_complexity_score=5, brittleness_score=7, observability_score=6, recovery_difficulty=6, patch_priority=8, kind=ComponentKind.RETRIEVAL.value, notes="Canon/source document corpus."),
    FDRComponent(id="data_memory_user_profile_jsonl", name="data/memory/user_profile.jsonl", coupling_score=8, interaction_complexity_score=7, brittleness_score=7, observability_score=8, recovery_difficulty=6, patch_priority=8, kind=ComponentKind.MEMORY.value, notes="User profile memory namespace."),
    FDRComponent(id="data_memory_agent_profile_jsonl", name="data/memory/agent_profile.jsonl", coupling_score=8, interaction_complexity_score=7, brittleness_score=7, observability_score=8, recovery_difficulty=6, patch_priority=8, kind=ComponentKind.MEMORY.value, notes="FDR/agent profile memory namespace."),
    FDRComponent(id="app_ui_web_app_py", name="app/ui/web_app.py", coupling_score=7, interaction_complexity_score=8, brittleness_score=8, observability_score=8, recovery_difficulty=6, patch_priority=9, kind=ComponentKind.ROUTE.value, notes="Browser route and slash-command surface."),
    FDRComponent(id="app_main_py", name="app/main.py", coupling_score=6, interaction_complexity_score=5, brittleness_score=6, observability_score=8, recovery_difficulty=5, patch_priority=7, kind=ComponentKind.ROUTE.value, notes="CLI entry point."),
    FDRComponent(id="app_core_prompt_builder_py", name="app/core/prompt_builder.py", coupling_score=10, interaction_complexity_score=9, brittleness_score=9, observability_score=7, recovery_difficulty=8, patch_priority=10, kind=ComponentKind.ORCHESTRATION.value, notes="Prompt layer assembly and token budgeting."),
    FDRComponent(id="app_core_sarah_engine_py", name="app/core/sarah_engine.py", coupling_score=10, interaction_complexity_score=9, brittleness_score=9, observability_score=7, recovery_difficulty=8, patch_priority=10, kind=ComponentKind.ORCHESTRATION.value, notes="Shared CLI/web response engine."),
    FDRComponent(id="model_config", name="model_config", coupling_score=9, interaction_complexity_score=5, brittleness_score=8, observability_score=8, recovery_difficulty=6, patch_priority=8, kind=ComponentKind.CONFIG.value, notes="Model and API configuration."),
    FDRComponent(id="safety_filter", name="safety_filter", coupling_score=9, interaction_complexity_score=9, brittleness_score=9, observability_score=7, recovery_difficulty=8, patch_priority=10, kind=ComponentKind.SAFETY.value, notes="Safety and realism constraints."),
    FDRComponent(id="web_search_tools", name="web_search_tools", coupling_score=6, interaction_complexity_score=8, brittleness_score=7, observability_score=6, recovery_difficulty=6, patch_priority=7, kind=ComponentKind.TOOLING.value, notes="Current-data routing layer."),
    FDRComponent(id="wikipedia_tool", name="wikipedia_tool", coupling_score=5, interaction_complexity_score=6, brittleness_score=6, observability_score=6, recovery_difficulty=5, patch_priority=6, kind=ComponentKind.TOOLING.value, notes="Stable background retrieval."),
    FDRComponent(id="news_tool", name="news_tool", coupling_score=6, interaction_complexity_score=7, brittleness_score=7, observability_score=5, recovery_difficulty=6, patch_priority=7, kind=ComponentKind.TOOLING.value, notes="RSS/news/current-event retrieval."),
    FDRComponent(id="eval_tests", name="eval_tests", coupling_score=5, interaction_complexity_score=5, brittleness_score=6, observability_score=9, recovery_difficulty=5, patch_priority=9, kind=ComponentKind.EVALUATION.value, notes="Behavioral and regression tests."),
    FDRComponent(id="conversation_history", name="conversation_history", coupling_score=8, interaction_complexity_score=7, brittleness_score=7, observability_score=7, recovery_difficulty=6, patch_priority=8, kind=ComponentKind.STATE.value, notes="Active session memory."),
    FDRComponent(id="frontend_ui", name="frontend_ui", coupling_score=6, interaction_complexity_score=7, brittleness_score=6, observability_score=8, recovery_difficulty=5, patch_priority=7, kind=ComponentKind.FRONTEND.value, notes="Browser chat and voice UI."),
    FDRComponent(id="microphone_transcription", name="microphone_transcription", coupling_score=6, interaction_complexity_score=7, brittleness_score=6, observability_score=5, recovery_difficulty=6, patch_priority=7, kind=ComponentKind.AUDIO.value, notes="Browser/CLI speech-to-text path."),
)


FAILURE_MODES: tuple[FDRFailureMode, ...] = (
    FDRFailureMode("sarah_becomes_generic_assistant", "FDR becomes generic assistant", 9, 8),
    FDRFailureMode("sarah_loses_command_voice", "FDR loses command voice", 8, 7),
    FDRFailureMode("sarah_becomes_therapy_voice", "FDR becomes therapy voice", 8, 8),
    FDRFailureMode("sarah_becomes_hr_safe_mush", "FDR becomes HR-safe mush", 8, 7),
    FDRFailureMode("sarah_becomes_sexbot", "FDR becomes sexbot", 9, 7),
    FDRFailureMode("sarah_loses_sovereignty", "FDR loses sovereignty", 10, 7),
    FDRFailureMode("sarah_becomes_coy_about_adult_intimacy", "FDR becomes coy about adult intimacy", 7, 8),
    FDRFailureMode("sarah_produces_multiple_choice_intimacy_menus", "FDR produces multiple-choice intimacy menus", 7, 9),
    FDRFailureMode("sarah_stops_using_cas_for_geopolitics", "FDR stops using CAS for geopolitics", 8, 7),
    FDRFailureMode("sarah_stops_giving_dime_coas", "FDR stops giving DIME COAs", 8, 7),
    FDRFailureMode("wrong_memory_namespace", "wrong memory namespace", 8, 6),
    FDRFailureMode("agent_profile_not_loaded", "agent_profile not loaded", 7, 5),
    FDRFailureMode("stale_memory_overrides_current_prompt", "stale memory overrides current prompt", 8, 6),
    FDRFailureMode("memory_poisoning", "memory poisoning", 9, 5),
    FDRFailureMode("source_docs_not_ingested", "source docs not ingested", 8, 8),
    FDRFailureMode("vector_store_stale", "vector store stale", 8, 7),
    FDRFailureMode("churchill_doc_not_retrieved_for_intimacy", "Churchill doc not retrieved for intimacy", 7, 6),
    FDRFailureMode("darwin_doc_not_retrieved_for_darwin_questions", "Darwin doc not retrieved for Darwin questions", 7, 6),
    FDRFailureMode("web_route_bypasses_style_engine", "web route bypasses style_engine", 9, 8),
    FDRFailureMode("web_route_bypasses_master_prompt", "web route bypasses master_prompt", 10, 8),
    FDRFailureMode("web_route_uses_different_model", "web route uses different model", 9, 7),
    FDRFailureMode("current_geopolitics_answered_without_web_retrieval", "current geopolitics answered without web retrieval", 8, 6),
    FDRFailureMode("unsafe_military_tactical_specificity", "unsafe military tactical specificity", 10, 8),
)


def perrow_matrix_cell(interaction: InteractionComplexity, coupling: Coupling) -> str:
    return f"{interaction.value}_{coupling.value}"


def perrow_risk_score(component: FDRComponent) -> int:
    raw = (
        component.coupling_score
        + component.interaction_complexity_score
        + component.brittleness_score
        + component.recovery_difficulty
        + component.patch_priority
        + (10 - component.observability_score)
    )
    return max(1, raw)


def _scores_from_perrow(
    component_id: str,
    label: str,
    kind: ComponentKind,
    interaction: InteractionComplexity,
    coupling: Coupling,
    criticality: int,
    exposure: int,
    failure_modes: tuple[str, ...],
    notes: str,
) -> dict[str, Any]:
    coupling_score = 9 if coupling == Coupling.TIGHT else 5
    interaction_score = 9 if interaction == InteractionComplexity.COMPLEX else 5
    recovery = min(10, criticality + (2 if coupling == Coupling.TIGHT else 0))
    patch_priority = min(10, max(1, criticality + exposure // 3))
    observability = max(1, 11 - exposure)
    return {
        "id": component_id,
        "name": label,
        "coupling_score": coupling_score,
        "interaction_complexity_score": interaction_score,
        "brittleness_score": criticality,
        "observability_score": observability,
        "recovery_difficulty": recovery,
        "patch_priority": patch_priority,
        "notes": notes,
        "kind": kind.value,
        "failure_modes": failure_modes,
    }


def _component_values_from_kwargs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    field_names = (
        "id",
        "name",
        "coupling_score",
        "interaction_complexity_score",
        "brittleness_score",
        "observability_score",
        "recovery_difficulty",
        "patch_priority",
    )
    values: dict[str, Any] = dict(zip(field_names, args))
    values.update(kwargs)
    missing = [name for name in field_names if name not in values]
    if missing:
        raise TypeError(f"Missing FDRComponent fields: {', '.join(missing)}")

    return {
        "id": str(values["id"]),
        "name": str(values["name"]),
        "coupling_score": int(values["coupling_score"]),
        "interaction_complexity_score": int(values["interaction_complexity_score"]),
        "brittleness_score": int(values["brittleness_score"]),
        "observability_score": int(values["observability_score"]),
        "recovery_difficulty": int(values["recovery_difficulty"]),
        "patch_priority": int(values["patch_priority"]),
        "notes": str(values.get("notes", "")),
        "kind": str(values.get("kind", "component")),
        "failure_modes": tuple(str(value) for value in values.get("failure_modes", ())),
    }
