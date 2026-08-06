from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any


MAX_ACTIVE_TURNS = 20


@dataclass(frozen=True)
class ConversationTurn:
    user: str
    assistant: str


class ConversationMemory:
    def __init__(self, max_turns: int = MAX_ACTIVE_TURNS) -> None:
        self.max_turns = max_turns
        self._turns: list[ConversationTurn] = []

    @property
    def turns(self) -> list[ConversationTurn]:
        return list(self._turns)

    def add_turn(self, user: str, assistant: str) -> None:
        self._turns.append(ConversationTurn(user=user, assistant=assistant))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def reset(self) -> None:
        self._turns.clear()


class TranscriptWriter:
    def __init__(self, conversations_dir: Path) -> None:
        now = datetime.now()
        day_dir = conversations_dir / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        self.path = day_dir / f"session-{now.strftime('%Y%m%d-%H%M%S')}.jsonl"

    def write_turn(
        self,
        user: str,
        assistant: str,
        model: str,
        sources: list[dict[str, Any]],
        has_sufficient_evidence: bool,
        web_sources: list[dict[str, Any]] | None = None,
        web_status: dict[str, Any] | None = None,
    ) -> None:
        self._write(
            {
                "type": "turn",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "user": user,
                "assistant": assistant,
                "has_sufficient_evidence": has_sufficient_evidence,
                "sources": sources,
                "web_sources": web_sources or [],
                "web_status": web_status or {},
            }
        )

    def write_event(self, event: str) -> None:
        self._write(
            {
                "type": "event",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "event": event,
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def history_to_messages(turns: list[ConversationTurn]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})
    return messages


def turn_to_dict(turn: ConversationTurn) -> dict[str, str]:
    return asdict(turn)
