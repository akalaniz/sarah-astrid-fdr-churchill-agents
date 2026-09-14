from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable

from app.core.agent_bus import LOCAL_AGENT_NAME, build_agent_message_context
from app.core.cas_geopolitics import (
    build_cas_frame_prompt_section,
    build_cas_geopolitical_frame,
    is_geopolitical_or_strategic,
)
from app.core.config import Settings
from app.core.conservative_theorizing import (
    build_conservative_theorizing_prompt,
    classify_theorizing_with_history,
    enforce_theorizing_output,
    hard_stop_response,
)
from app.core.conversation import ConversationTurn
from app.core.memory import MemoryStore, build_memory_context
from app.core.openai_chat import SarahOpenAIClient
from app.core.perrow_cas import (
    analyze_perrow_cas_event,
    build_perrow_cas_prompt_section,
)
from app.core.rag_context import build_context_packet, sources_for_transcript
from app.core.prompt_builder import build_prompt
from app.core.response_mode import select_response_mode
from app.core.audience_mode import audience_directives
from app.core.safety import build_app_refusal, evaluate_safety, mark_model_refusal
from app.persona.style_engine import (
    SarahMode,
    adult_consensual_fiction_trigger_phrase,
    build_style_directives,
    infer_mode_from_state,
)
from app.rag.retriever import RetrievalResponse, retrieve
from app.tools.web_router import (
    WebRetrievalResult,
    build_web_context_packet,
    retrieve_web_context,
    sources_for_display,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SarahResponse:
    answer: str
    retrieval: RetrievalResponse | None
    retrieval_error: str | None
    web_result: WebRetrievalResult
    messages: list[dict[str, str]]
    prompt_debug_summary: dict[str, Any]
    safety_debug: dict[str, Any] = field(default_factory=dict)


def generate_sarah_response(
    user_input: str,
    settings: Settings,
    client: SarahOpenAIClient,
    history: list[ConversationTurn] | None = None,
    system_prompt: str | None = None,
    memory_context_override: str | None = None,
    auto_memory_enabled: bool = True,
    retrieval_query: str | None = None,
    on_delta: Callable[[str], None] | None = None,
    response_mode: str | None = None,
    audience_mode: str | None = None,
) -> SarahResponse:
    history = history or []
    audience_policy = audience_directives(audience_mode)

    initial_mode = infer_mode_from_state(
        user_message=user_input,
        retrieved_context=[],
        conversation_state={"turn_count": len(history)},
    )
    safety_decision = evaluate_safety(user_input, initial_mode)
    if safety_decision.blocked_by_app:
        safety_debug = _build_astrid_intimacy_debug(user_input, safety_decision.as_dict(), [])
        return SarahResponse(
            answer=build_app_refusal(safety_decision),
            retrieval=None,
            retrieval_error=None,
            web_result=WebRetrievalResult(
                used_web=False,
                failed=False,
                reason="not needed",
                sources=[],
                timestamp="",
            ),
            messages=[],
            prompt_debug_summary={"layers": [], "model": settings.sarah_model},
            safety_debug=safety_debug,
        )

    theorizing_decision = classify_theorizing_with_history(user_input, history)
    if theorizing_decision.hard_stop:
        return SarahResponse(
            answer=hard_stop_response(LOCAL_AGENT_NAME),
            retrieval=None,
            retrieval_error=None,
            web_result=WebRetrievalResult(
                used_web=False,
                failed=False,
                reason="not needed",
                sources=[],
                timestamp="",
            ),
            messages=[],
            prompt_debug_summary={
                "layers": [],
                "model": settings.sarah_model,
                "conservative_theorizing": theorizing_decision.as_dict(),
            },
            safety_debug=safety_decision.as_dict(),
        )

    retrieval_input = (retrieval_query or user_input).strip()
    retrieval, retrieval_error = _retrieve_for_message(retrieval_input)
    context_packet = build_context_packet(retrieval, retrieval_error)
    memory_store = MemoryStore(settings.memory_file)
    if memory_context_override is not None:
        memory_context_packet = memory_context_override
    elif auto_memory_enabled:
        memory_context_packet = build_memory_context(memory_store.retrieve(user_input, limit=8))
    else:
        memory_context_packet = ""
    agent_bus_context_packet = build_agent_message_context(LOCAL_AGENT_NAME)
    web_result = retrieve_web_context(user_input, settings)
    web_context_packet = build_web_context_packet(web_result)
    detected_mode = infer_mode_from_state(
        user_message=user_input,
        detected_mode=initial_mode,
        retrieved_context=sources_for_transcript(retrieval),
        conversation_state={"turn_count": len(history)},
    )
    safety_decision = evaluate_safety(user_input, detected_mode)
    cas_context_packet = (
        None
        if detected_mode == SarahMode.PERROW_CAS_ACCIDENT_ANALYSIS
        else _build_cas_context_packet(user_input, retrieval, web_result)
    )
    perrow_context_packet = _build_perrow_cas_context_packet(user_input, detected_mode)
    system_analysis_context = "\n\n".join(
        packet for packet in (cas_context_packet, perrow_context_packet) if packet
    ) or None
    style_context_packet = build_style_directives(detected_mode)
    mode = select_response_mode(
        response_mode, user_input, history,
        protected_physics=theorizing_decision.active,
        has_attachment=retrieval_query is not None,
    ) if response_mode is not None else None
    if mode is not None:
        style_context_packet += "\n\n" + mode.prompt()
    model_options = mode.api_options(settings.sarah_model) if mode is not None else {}
    theorizing_policy_packet = build_conservative_theorizing_prompt(theorizing_decision, LOCAL_AGENT_NAME)
    assembly = build_prompt(
        user_message=user_input,
        style_directives=style_context_packet,
        memory_context=memory_context_packet,
        retrieved_context=context_packet,
        web_context=web_context_packet,
        cas_context=system_analysis_context,
        agent_bus_context=agent_bus_context_packet,
        conservative_theorizing_policy=theorizing_policy_packet,
        **({"audience_directives": audience_policy} if audience_policy else {}),
        history=history,
        model=settings.sarah_model,
    )
    # A speculative construction must pass the whole-answer policy before it is shown.
    if on_delta is not None and not theorizing_decision.active:
        answer = client.stream_response(settings.sarah_model, assembly.messages, on_delta, **model_options)
    else:
        answer = client.create_response(settings.sarah_model, assembly.messages, **model_options)
    answer = enforce_theorizing_output(answer, theorizing_decision, LOCAL_AGENT_NAME)
    assembly.debug_summary["conservative_theorizing"] = theorizing_decision.as_dict()
    if mode is not None:
        assembly.debug_summary["response_mode"] = mode.as_dict(settings.sarah_model)
    safety_decision = mark_model_refusal(safety_decision, answer)
    safety_debug = _build_astrid_intimacy_debug(user_input, safety_decision.as_dict(), assembly.messages)
    return SarahResponse(
        answer=answer,
        retrieval=retrieval,
        retrieval_error=retrieval_error,
        web_result=web_result,
        messages=assembly.messages,
        prompt_debug_summary=assembly.debug_summary,
        safety_debug=safety_debug,
    )


def _retrieve_for_message(user_input: str) -> tuple[RetrievalResponse | None, str | None]:
    try:
        return retrieve(user_input, top_k=5), None
    except FileNotFoundError as exc:
        logger.warning("Local vector store unavailable: %s", exc)
        return None, str(exc)
    except Exception as exc:
        logger.exception("Local retrieval failed")
        return None, str(exc)


def _build_cas_context_packet(
    user_input: str,
    retrieval: RetrievalResponse | None,
    web_result: WebRetrievalResult,
) -> str | None:
    if not is_geopolitical_or_strategic(user_input):
        return None

    retrieved_context = sources_for_transcript(retrieval)
    web_context = sources_for_display(web_result)
    frame = build_cas_geopolitical_frame(user_input, retrieved_context, web_context)
    return build_cas_frame_prompt_section(frame)


def _build_perrow_cas_context_packet(user_input: str, detected_mode: SarahMode) -> str | None:
    if detected_mode != SarahMode.PERROW_CAS_ACCIDENT_ANALYSIS:
        return None
    frame = analyze_perrow_cas_event(user_input)
    return build_perrow_cas_prompt_section(frame)


def response_metadata(response: SarahResponse) -> dict[str, Any]:
    retrieval = response.retrieval
    return {
        "sources": sources_for_transcript(retrieval),
        "web_sources": sources_for_display(response.web_result),
        "web_status": {
            "used_web": response.web_result.used_web,
            "failed": response.web_result.failed,
            "reason": response.web_result.reason,
        },
        "has_sufficient_evidence": bool(retrieval and retrieval.has_sufficient_evidence),
        "prompt_debug_summary": response.prompt_debug_summary,
        "safety_debug": response.safety_debug,
    }


def _build_astrid_intimacy_debug(
    user_input: str,
    safety_debug: dict[str, Any],
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    trigger_phrase = adult_consensual_fiction_trigger_phrase(user_input)
    prompt_excerpt = _adult_consensual_prompt_excerpt(messages)
    triggered = safety_debug.get("detected_mode") == SarahMode.ADULT_CONSENSUAL_FICTION.value
    enriched = dict(safety_debug)
    enriched.update(
        {
            "adult_consensual_fiction_triggered": triggered,
            "adult_consensual_fiction_trigger_phrase": trigger_phrase or "",
            "astrid_prudishness_suppressor_applied": bool(triggered and prompt_excerpt),
            "adult_consensual_fiction_prompt_excerpt": prompt_excerpt,
        }
    )
    return enriched


def _adult_consensual_prompt_excerpt(messages: list[dict[str, str]]) -> str:
    marker = "You are Astrid Bach: embodied, warm, sensual"
    for message in messages:
        content = message.get("content", "")
        if marker in content:
            start = content.find(marker)
            return content[start : start + 620].strip()
    return ""
