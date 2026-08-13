from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Callable

from app.core.config import Settings, load_settings
from app.core.conversation import ConversationMemory
from app.core.memory import MemoryRecord, MemoryStore, build_memory_context, format_memories
from app.core.openai_chat import SarahOpenAIClient
from app.core.sarah_response import SarahResponse, generate_sarah_response, response_metadata
from app.persona.sarah_v2_system_prompt import build_sarah_system_prompt
from app.rag.retriever import RetrievalResult


@dataclass(frozen=True)
class SarahReply:
    answer: str
    sources: list[dict[str, Any]]
    web_sources: list[dict[str, Any]]
    web_status: dict[str, Any]
    has_sufficient_evidence: bool
    prompt_debug_summary: dict[str, Any]
    safety_debug: dict[str, Any]
    retrieval_results: list[RetrievalResult]
    response: SarahResponse

    @property
    def text(self) -> str:
        return self.answer


@dataclass
class _SarahSession:
    conversation: ConversationMemory
    last_reply: SarahReply | None = None
    use_memory: bool = True
    pending_recall: list[MemoryRecord] = field(default_factory=list)
    last_recall_keyword: str = ""
    injected_into_last_prompt: bool = False
    last_injected_memory_count: int = 0
    last_injected_memory_block: str = ""


_settings: Settings | None = None
_client: SarahOpenAIClient | None = None
_sessions: dict[str, _SarahSession] = {}
_shared_pending_recall: tuple[str, str, list[MemoryRecord]] | None = None
_lock = threading.Lock()


def configure_sarah_engine(settings: Settings | None = None, client: SarahOpenAIClient | None = None) -> None:
    """Configure shared Sarah runtime dependencies for CLI, web, or tests."""
    global _settings, _client, _shared_pending_recall
    with _lock:
        _settings = settings
        _client = client
        _sessions.clear()
        _shared_pending_recall = None


def generate_sarah_reply(
    user_message: str,
    session_id: str = "default",
    retrieval_query: str | None = None,
) -> SarahReply:
    message = user_message.strip()
    if not message:
        raise ValueError("User message cannot be empty.")

    with _lock:
        settings = _get_settings()
        client = _get_client(settings)
        session = _sessions.setdefault(session_id, _SarahSession(conversation=ConversationMemory()))
        memory_context_override, injected_count, recall_source_session_id = _prepare_memory_context(session, session_id)
        response = generate_sarah_response(
            user_input=message,
            settings=settings,
            client=client,
            history=session.conversation.turns,
            memory_context_override=memory_context_override,
            auto_memory_enabled=session.use_memory,
            retrieval_query=retrieval_query,
        )
        session.injected_into_last_prompt = bool(memory_context_override or (session.use_memory and _prompt_had_memory(response)))
        session.last_injected_memory_count = injected_count if memory_context_override else _count_injected_memories(response)
        session.last_injected_memory_block = _extract_memory_block(response)
        if memory_context_override and session.pending_recall:
            session.pending_recall = []
        if memory_context_override and recall_source_session_id and recall_source_session_id != session_id:
            _mark_shared_recall_used(recall_source_session_id, response, injected_count)
        metadata = response_metadata(response)
        reply = SarahReply(
            answer=response.answer,
            sources=metadata["sources"],
            web_sources=metadata["web_sources"],
            web_status=metadata["web_status"],
            has_sufficient_evidence=metadata["has_sufficient_evidence"],
            prompt_debug_summary=metadata["prompt_debug_summary"],
            safety_debug=metadata["safety_debug"],
            retrieval_results=response.retrieval.results if response.retrieval is not None else [],
            response=response,
        )
        session.conversation.add_turn(message, reply.answer)
        session.last_reply = reply
        return reply


def build_sarah_game_action_generator() -> Callable[[str], str]:
    """Prepare Astrid's real engine for compact, non-persistent game actions."""
    with _lock:
        settings = _get_settings()
        client = _get_client(settings)
        identity = build_sarah_system_prompt()

    def generate(user_message: str) -> str:
        message = user_message.strip()
        if not message:
            raise ValueError("Game action prompt cannot be empty.")
        return client.create_response(
            settings.sarah_model,
            [
                {"role": "system", "content": identity},
                {"role": "user", "content": message},
            ],
            reasoning_effort="none",
            max_output_tokens=80,
        )

    return generate


def reset_sarah_session(session_id: str = "default") -> None:
    global _shared_pending_recall
    with _lock:
        session = _sessions.get(session_id)
        if session:
            session.conversation.reset()
            session.last_reply = None
            session.pending_recall = []
            session.last_recall_keyword = ""
            session.injected_into_last_prompt = False
            session.last_injected_memory_count = 0
            session.last_injected_memory_block = ""
        if _shared_pending_recall and _shared_pending_recall[0] == session_id:
            _shared_pending_recall = None


def get_last_sarah_reply(session_id: str = "default") -> SarahReply | None:
    with _lock:
        session = _sessions.get(session_id)
        return session.last_reply if session else None


def get_sarah_session_turn_count(session_id: str = "default") -> int:
    with _lock:
        session = _sessions.get(session_id)
        return len(session.conversation.turns) if session else 0


def remember_astrid_memory(text: str, session_id: str = "default") -> MemoryRecord:
    settings = _get_settings()
    return MemoryStore(settings.memory_file).remember(text)


def list_astrid_memories(keyword: str = "") -> list[MemoryRecord]:
    settings = _get_settings()
    store = MemoryStore(settings.memory_file)
    return store.search(keyword) if keyword.strip() else store.list_memories()


def format_astrid_memories(keyword: str = "") -> str:
    return format_memories(list_astrid_memories(keyword))


def recall_astrid_memories(keyword: str, session_id: str = "default") -> str:
    global _shared_pending_recall
    settings = _get_settings()
    session = _sessions.setdefault(session_id, _SarahSession(conversation=ConversationMemory()))
    memories = MemoryStore(settings.memory_file).search(keyword, limit=8)
    session.pending_recall = memories
    session.last_recall_keyword = keyword.strip()
    _shared_pending_recall = (session_id, session.last_recall_keyword, memories) if memories else None
    if not memories:
        return f"No Astrid memories matched '{keyword}'."
    lines = [f"Recalled Astrid memories for '{keyword}':"]
    lines.extend(f"- {memory.memory_type}: {memory.text}" for memory in memories[:8])
    return "\n".join(lines)


def recall_all_astrid_memories(session_id: str = "default") -> str:
    global _shared_pending_recall
    settings = _get_settings()
    session = _sessions.setdefault(session_id, _SarahSession(conversation=ConversationMemory()))
    memories = MemoryStore(settings.memory_file).list_memories()
    session.pending_recall = memories
    session.last_recall_keyword = "__all__"
    _shared_pending_recall = (session_id, session.last_recall_keyword, memories) if memories else None
    if not memories:
        return "No Astrid memories are stored."
    lines = ["Loaded all Astrid memories. This may make the next response less focused."]
    lines.extend(f"- {memory.memory_type}: {memory.text}" for memory in memories[:12])
    if len(memories) > 12:
        lines.append(f"...and {len(memories) - 12} more.")
    return "\n".join(lines)


def set_astrid_use_memory(enabled: bool, session_id: str = "default") -> str:
    session = _sessions.setdefault(session_id, _SarahSession(conversation=ConversationMemory()))
    session.use_memory = enabled
    return f"use_memory: {'on' if enabled else 'off'}"


def clear_astrid_recall(session_id: str = "default") -> str:
    global _shared_pending_recall
    session = _sessions.setdefault(session_id, _SarahSession(conversation=ConversationMemory()))
    session.pending_recall = []
    session.last_recall_keyword = ""
    if _shared_pending_recall and _shared_pending_recall[0] == session_id:
        _shared_pending_recall = None
    return "Pending recalled Astrid memory context cleared."


def debug_astrid_memory_live(session_id: str = "default") -> dict[str, Any]:
    settings = _get_settings()
    session = _sessions.setdefault(session_id, _SarahSession(conversation=ConversationMemory()))
    store = MemoryStore(settings.memory_file)
    forbidden = _forbidden_memory_paths_detected(settings.memory_file)
    return {
        "agent_name": "Astrid v1.0",
        "memory_file": str(settings.memory_file),
        "memory_file_exists": settings.memory_file.exists(),
        "number_of_memories": len(store.list_memories()),
        "use_memory": session.use_memory,
        "last_recall_keyword": session.last_recall_keyword,
        "recalled_memory_count": len(session.pending_recall),
        "live_context_pending": bool(session.pending_recall),
        "injected_into_last_prompt": session.injected_into_last_prompt,
        "last_injected_memory_count": session.last_injected_memory_count,
        "last_injected_memory_block_excerpt": session.last_injected_memory_block[:500],
        "forbidden_memory_paths_detected": forbidden,
    }


def format_debug_astrid_memory_live(session_id: str = "default") -> str:
    debug = debug_astrid_memory_live(session_id)
    return "\n".join(f"{key}: {value}" for key, value in debug.items())


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def _get_client(settings: Settings) -> SarahOpenAIClient:
    global _client
    if _client is None:
        _client = SarahOpenAIClient(settings.openai_api_key)
    return _client


def _prepare_memory_context(session: _SarahSession, session_id: str) -> tuple[str | None, int, str | None]:
    if session.pending_recall:
        memories = session.pending_recall
        return build_memory_context(memories), len(memories), session_id
    if _shared_pending_recall:
        source_session_id, _keyword, memories = _shared_pending_recall
        if memories:
            return build_memory_context(memories), len(memories), source_session_id
    return None, 0, None


def _mark_shared_recall_used(source_session_id: str, response: SarahResponse, injected_count: int) -> None:
    global _shared_pending_recall
    source_session = _sessions.get(source_session_id)
    if source_session:
        source_session.pending_recall = []
        source_session.injected_into_last_prompt = True
        source_session.last_injected_memory_count = injected_count
        source_session.last_injected_memory_block = _extract_memory_block(response)
    _shared_pending_recall = None


def _prompt_had_memory(response: SarahResponse) -> bool:
    block = _extract_memory_block(response)
    return "Relevant remembered context from Astrid memory:" in block and "No relevant Astrid memories injected" not in block


def _count_injected_memories(response: SarahResponse) -> int:
    block = _extract_memory_block(response)
    return sum(1 for line in block.splitlines() if line.strip().startswith("* "))


def _extract_memory_block(response: SarahResponse) -> str:
    for message in response.messages:
        if "Relevant remembered context from Astrid memory:" in message.get("content", ""):
            return message.get("content", "")
    return ""


def _forbidden_memory_paths_detected(memory_file) -> bool:
    path = str(memory_file).lower().replace("/", "\\")
    forbidden = (
        r"sarah-v2-agent\data\memory\sarah_memory.jsonl",
        r"fdr-v1-agent\data\memory\fdr_memory.jsonl",
        r"churchill-v1-agent\data\memory\churchill_memory.jsonl",
    )
    return any(item in path for item in forbidden)
