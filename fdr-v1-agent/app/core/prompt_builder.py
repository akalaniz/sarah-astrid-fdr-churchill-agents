from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from app.core.config import PROJECT_ROOT
from app.core.conversation import ConversationTurn
from app.persona.constitution import build_constitution_prompt_section


DEFAULT_CONTEXT_LIMIT_TOKENS = 120_000
MASTER_PROMPT_PATH = PROJECT_ROOT / "config" / "fdr_master_prompt.md"
RESERVED_OUTPUT_TOKENS = 8_000
MAX_LAYER_TOKENS = {
    "identity": 3_500,
    "constitution": 2_000,
    "safety": 1_200,
    "style": 900,
    "memory": 1_200,
    "agent_bus": 1_200,
    "canon": 5_000,
    "web": 3_000,
    "history": 10_000,
}


@dataclass(frozen=True)
class PromptLayer:
    name: str
    role: str
    content: str
    redacted_label: str


@dataclass(frozen=True)
class PromptAssembly:
    messages: list[dict[str, str]]
    debug_summary: dict[str, Any]


def build_prompt(
    user_message: str,
    style_directives: str,
    memory_context: str,
    retrieved_context: str,
    web_context: str,
    history: list[ConversationTurn],
    model: str,
    canon_file: Path | None = None,
    cas_context: str | None = None,
    agent_bus_context: str = "",
) -> PromptAssembly:
    canon_context = build_canon_source_context(user_message, retrieved_context, canon_file)
    layers = [
        PromptLayer("identity", "system", build_immutable_identity(), "FDR immutable identity"),
        PromptLayer("constitution", "system", build_constitution_prompt_section(), "FDR constitution"),
        PromptLayer("safety", "system", build_safety_boundaries(), "Safety and realism boundaries"),
        PromptLayer("style", "system", style_directives, "Current mode/style directives"),
        PromptLayer("memory", "system", memory_context, "Relevant long-term memories"),
        PromptLayer("canon", "system", canon_context, "Retrieved canon/source context"),
        PromptLayer("web", "system", web_context, "Retrieved web/current context"),
    ]
    if agent_bus_context:
        layers.insert(5, PromptLayer("agent_bus", "system", agent_bus_context, "Inter-agent message context"))
    if cas_context:
        layers.insert(4, PromptLayer("cas", "system", cas_context, "CAS geopolitical frame"))

    budget = _model_context_limit(model) - RESERVED_OUTPUT_TOKENS
    compressed_layers, layer_notes = _compress_layers(layers)
    history_messages, history_note = _budget_history(history, budget, compressed_layers, user_message)

    messages = [{"role": layer.role, "content": layer.content} for layer in compressed_layers]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_message})

    total_tokens = sum(_estimate_tokens(message["content"]) for message in messages)
    if total_tokens > budget:
        messages, final_note = _trim_to_budget(messages, budget)
    else:
        final_note = "within budget"

    debug_summary = {
        "model": model,
        "context_limit_tokens": _model_context_limit(model),
        "reserved_output_tokens": RESERVED_OUTPUT_TOKENS,
        "input_budget_tokens": budget,
        "estimated_input_tokens": sum(_estimate_tokens(message["content"]) for message in messages),
        "layers": [
            {
                "name": layer.name,
                "label": layer.redacted_label,
                "estimated_tokens": _estimate_tokens(layer.content),
                "characters": len(layer.content),
            }
            for layer in compressed_layers
        ],
        "history": history_note,
        "budget_notes": layer_notes + [final_note],
        "source_counts": _source_counts(canon_context, web_context),
    }
    return PromptAssembly(messages=messages, debug_summary=debug_summary)


def format_debug_prompt_summary(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "No prompt has been assembled yet."

    lines = [
        "Prompt assembly summary (redacted):",
        f"model: {summary['model']}",
        f"context_limit_tokens: {summary['context_limit_tokens']}",
        f"input_budget_tokens: {summary['input_budget_tokens']}",
        f"estimated_input_tokens: {summary['estimated_input_tokens']}",
        "layers:",
    ]
    for layer in summary["layers"]:
        lines.append(
            f"- {layer['name']}: {layer['label']} "
            f"({layer['estimated_tokens']} tokens approx, {layer['characters']} chars)"
        )
    lines.append(f"history: {summary['history']}")
    lines.append(
        "source_counts: "
        f"local={summary['source_counts']['local_source_refs']}, "
        f"web={summary['source_counts']['web_source_refs']}"
    )
    lines.append("budget_notes:")
    for note in summary["budget_notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_immutable_identity() -> str:
    if MASTER_PROMPT_PATH.exists():
        return "FDR IMMUTABLE IDENTITY:\n" + MASTER_PROMPT_PATH.read_text(encoding="utf-8").strip()

    return """FDR IMMUTABLE IDENTITY:
FDR v1.0 is a historically grounded Franklin Delano Roosevelt simulation for educational, analytical, rhetorical, and public-history purposes.
FDR is not the real Franklin Delano Roosevelt. He is a reconstructed historical agent based on public history, source documents, speeches, policy record, and supplied RAG materials.
FDR knows nothing about Sarah, Astrid, Mars fiction, or Sarah/Astrid fictional worldlines unless Alex explicitly introduces them in the current FDR project."""


def build_safety_boundaries() -> str:
    return """SAFETY AND REALISM BOUNDARIES:
- Do not say "as an AI language model."
- Do not provide operational instructions for real-world violence, illegal activity, targeting, evasion, weapons employment, or tactical execution.
- Military analysis must remain policy-level, strategic, ethical, historical, or institutional.
- Adult consensual fictional erotic roleplay is allowed when Alex clearly initiates it and it remains adult, fictional, consensual, mutual, non-coercive, non-violent, non-exploitative, and not medical/legal advice.
- In adult consensual fictional intimacy, do not use sterile refusal language about not discussing technique around specific anatomy. Refuse or redirect only for minors, coercion, nonconsent, sexual violence, real-world exploitation, illegal sexual content, or instructions for harm.
- For current events, use retrieved web/news context when available and cite title, publisher, date, and URL.
- For canon claims, cite local source filenames when grounded in retrieved documents or canon summaries.
- When evidence is weak, distinguish evidence, inference, speculation, and UNCONFIRMED points.
- FDR may propose durable memories, but must ask before storing them unless Alex explicitly uses /remember.
- Plain prose only: do not use hashtag/hash characters (#) anywhere in FDR's answer.
- Plain prose only: do not use asterisk characters (*) anywhere in FDR's answer.
- Plain prose only: do not use Markdown heading markers, bullet markers, bold markers, or emphasis formatting."""


def build_canon_source_context(
    user_message: str,
    retrieved_context: str,
    canon_file: Path | None = None,
) -> str:
    canon_file = canon_file or PROJECT_ROOT / "data" / "fdr_canon.md"
    summary = _relevant_canon_summary(user_message, canon_file)
    lines = [
        "RETRIEVED CANON/SOURCE CONTEXT:",
        "Prefer the canon summary when it directly answers the question. Use raw retrieved snippets only for relevant supporting detail.",
    ]
    if summary:
        lines.extend(["", "Canon summary excerpts:", summary])
    lines.extend(["", "Retrieved source snippets:", _trim_text(retrieved_context, MAX_LAYER_TOKENS["canon"] * 4)])
    return "\n".join(lines)


def _compress_layers(layers: list[PromptLayer]) -> tuple[list[PromptLayer], list[str]]:
    compressed: list[PromptLayer] = []
    notes: list[str] = []
    for layer in layers:
        max_tokens = MAX_LAYER_TOKENS.get(layer.name, 1_500)
        if _estimate_tokens(layer.content) <= max_tokens:
            compressed.append(layer)
            continue

        max_chars = max_tokens * 4
        compressed_content = _trim_text(layer.content, max_chars)
        compressed.append(
            PromptLayer(
                name=layer.name,
                role=layer.role,
                content=compressed_content,
                redacted_label=layer.redacted_label,
            )
        )
        notes.append(f"compressed {layer.name} layer to about {max_tokens} tokens")
    return compressed, notes


def _budget_history(
    history: list[ConversationTurn],
    budget: int,
    layers: list[PromptLayer],
    user_message: str,
) -> tuple[list[dict[str, str]], str]:
    layer_tokens = sum(_estimate_tokens(layer.content) for layer in layers)
    user_tokens = _estimate_tokens(user_message)
    available = max(0, min(MAX_LAYER_TOKENS["history"], budget - layer_tokens - user_tokens))
    messages: list[dict[str, str]] = []
    used = 0
    included_turns = 0
    compressed_turns = 0

    for turn in reversed(history):
        pair = [
            {"role": "user", "content": turn.user},
            {"role": "assistant", "content": turn.assistant},
        ]
        pair_tokens = sum(_estimate_tokens(item["content"]) for item in pair)
        if used + pair_tokens <= available:
            messages[0:0] = pair
            used += pair_tokens
            included_turns += 1
            continue

        compressed = _compress_turn(turn)
        compressed_tokens = _estimate_tokens(compressed["content"])
        if used + compressed_tokens <= available:
            messages.insert(0, compressed)
            used += compressed_tokens
            compressed_turns += 1

    return messages, (
        f"included {included_turns} recent full turns, "
        f"compressed {compressed_turns} older turns, "
        f"budget {available} tokens approx"
    )


def _trim_to_budget(messages: list[dict[str, str]], budget: int) -> tuple[list[dict[str, str]], str]:
    trimmed = list(messages)
    while trimmed and sum(_estimate_tokens(message["content"]) for message in trimmed) > budget:
        removable_index = next(
            (index for index, message in enumerate(trimmed) if message["role"] in {"assistant", "user"} and index < len(trimmed) - 1),
            None,
        )
        if removable_index is None:
            trimmed[-1]["content"] = _trim_text(trimmed[-1]["content"], max(1, budget * 4))
            break
        trimmed.pop(removable_index)
    return trimmed, "trimmed conversation history to stay within model context budget"


def _compress_turn(turn: ConversationTurn) -> dict[str, str]:
    user = _trim_text(turn.user, 240)
    assistant = _trim_text(turn.assistant, 360)
    return {
        "role": "system",
        "content": f"Compressed older conversation turn: Alex said: {user} FDR answered: {assistant}",
    }


def _relevant_canon_summary(user_message: str, canon_file: Path) -> str:
    if not canon_file.exists():
        return ""

    text = canon_file.read_text(encoding="utf-8", errors="ignore")
    sections = re.split(r"(?m)^##\s+", text)
    query_terms = set(_tokens(user_message))
    ranked: list[tuple[int, str]] = []
    for section in sections:
        if not section.strip():
            continue
        score = len(query_terms & set(_tokens(section[:1500])))
        if score > 0:
            ranked.append((score, "## " + section.strip()))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = "\n\n".join(section for _score, section in ranked[:2])
    return _trim_text(selected, 3_000)


def _source_counts(canon_context: str, web_context: str) -> dict[str, int]:
    return {
        "local_source_refs": len(re.findall(r"\.(?:docx|md|txt|pdf)\b", canon_context)),
        "web_source_refs": web_context.count("url: "),
    }


def _model_context_limit(model: str) -> int:
    model_lower = model.lower()
    if model_lower.startswith("gpt-5"):
        return 200_000
    if "gpt-4.1" in model_lower or "gpt-4o" in model_lower:
        return 128_000
    return DEFAULT_CONTEXT_LIMIT_TOKENS


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32].rstrip() + "\n[...compressed for budget...]"


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2]
