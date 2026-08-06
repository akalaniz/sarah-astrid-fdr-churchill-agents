from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import shutil
from typing import Any

from app.core.config import PROJECT_ROOT


SHARED_BUS_DIR = PROJECT_ROOT.parent / "shared_agent_bus"
TRANSCRIPTS_PATH = SHARED_BUS_DIR / "transcripts"
ARCHIVE_PATH = SHARED_BUS_DIR / "transcripts_archive"
LATEST_FILES = {"latest_multi_agent_transcript.md", "latest_multi_agent_transcript.json"}
NORMAL_TRANSCRIPT_PATTERN = re.compile(r"^[A-Za-z0-9_-]+\.(?:json|md)$")


@dataclass(frozen=True)
class TranscriptGroup:
    conversation_id: str
    files: list[Path]
    newest_mtime: float
    oldest_mtime: float


def get_sarah_transcript_paths() -> tuple[Path, Path]:
    return TRANSCRIPTS_PATH, ARCHIVE_PATH


def list_transcript_files(transcripts_path: Path | None = None) -> list[Path]:
    path = transcripts_path or TRANSCRIPTS_PATH
    if not path.exists():
        return []
    return sorted([item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".json", ".md"}])


def group_transcripts_by_conversation_id(files: list[Path]) -> dict[str, TranscriptGroup]:
    groups: dict[str, list[Path]] = {}
    for file_path in files:
        if _is_latest_file(file_path) or not _is_normal_transcript_file(file_path):
            continue
        groups.setdefault(file_path.stem, []).append(file_path)
    return {
        conversation_id: TranscriptGroup(
            conversation_id=conversation_id,
            files=sorted(group_files, key=lambda path: path.name),
            newest_mtime=max(path.stat().st_mtime for path in group_files),
            oldest_mtime=min(path.stat().st_mtime for path in group_files),
        )
        for conversation_id, group_files in groups.items()
    }


def transcript_status(transcripts_path: Path | None = None, archive_path: Path | None = None) -> dict[str, Any]:
    transcripts_path, archive_path = _paths(transcripts_path, archive_path)
    error = _safety_error(transcripts_path, archive_path)
    if error:
        return _with_error({"agent": "Sarah v2.0", "worldline": "sarah_astrid"}, error)
    transcripts_path.mkdir(parents=True, exist_ok=True)
    files = list_transcript_files(transcripts_path)
    groups = group_transcripts_by_conversation_id(files)
    normal_files = [path for path in files if not _is_latest_file(path)]
    sorted_files = sorted(normal_files, key=lambda path: path.stat().st_mtime)
    total_size = sum(path.stat().st_size for path in files)
    return {
        "agent": "Sarah v2.0",
        "worldline": "sarah_astrid",
        "transcripts_path": str(transcripts_path),
        "archive_path": str(archive_path),
        "file_count_total": len(files),
        "conversation_count": len(groups),
        "json_count": len([path for path in files if path.suffix.lower() == ".json"]),
        "md_count": len([path for path in files if path.suffix.lower() == ".md"]),
        "total_size_kb": round(total_size / 1024, 2),
        "oldest_file": sorted_files[0].name if sorted_files else "",
        "oldest_modified": _mtime_iso(sorted_files[0]) if sorted_files else "",
        "newest_file": sorted_files[-1].name if sorted_files else "",
        "newest_modified": _mtime_iso(sorted_files[-1]) if sorted_files else "",
        "latest_transcript_md_exists": (transcripts_path / "latest_multi_agent_transcript.md").exists(),
        "latest_transcript_json_exists": (transcripts_path / "latest_multi_agent_transcript.json").exists(),
        "error": "",
    }


def keep_newest_conversations(transcripts_path: Path | None = None, keep_n: int = 20) -> dict[str, Any]:
    transcripts_path, archive_path = _paths(transcripts_path, None)
    error = _safety_error(transcripts_path, archive_path)
    if error:
        return _with_error(_keep_result(transcripts_path, keep_n), error)
    keep_n = max(0, int(keep_n))
    groups = _groups_sorted_newest(transcripts_path)
    keep_ids = {group.conversation_id for group in groups[:keep_n]}
    removed_files = 0
    removed_conversations = 0
    for group in groups:
        if group.conversation_id in keep_ids:
            continue
        removed_conversations += 1
        for file_path in group.files:
            file_path.unlink(missing_ok=True)
            removed_files += 1
    return {
        "kept_conversation_count": min(len(groups), keep_n),
        "removed_conversation_count": removed_conversations,
        "removed_file_count": removed_files,
        "transcripts_path": str(transcripts_path),
        "error": "",
    }


def archive_older_than(
    transcripts_path: Path | None = None,
    archive_path: Path | None = None,
    days: int = 7,
    keep_min: int | None = None,
) -> dict[str, Any]:
    transcripts_path, archive_path = _paths(transcripts_path, archive_path)
    error = _safety_error(transcripts_path, archive_path)
    if error:
        return _with_error(_archive_result(transcripts_path, archive_path, days), error)
    days = max(0, int(days))
    keep_min = max(0, int(keep_min)) if keep_min is not None else None
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    groups = _groups_sorted_newest(transcripts_path)
    protected = {group.conversation_id for group in groups[:keep_min]} if keep_min is not None else set()
    archive_path.mkdir(parents=True, exist_ok=True)
    archived_conversations = 0
    archived_files = 0
    for group in groups:
        if group.conversation_id in protected:
            continue
        if datetime.fromtimestamp(group.newest_mtime, tz=timezone.utc) >= cutoff:
            continue
        archived_conversations += 1
        for file_path in group.files:
            _move_to_archive(file_path, archive_path)
            archived_files += 1
    return {
        "archived_conversation_count": archived_conversations,
        "archived_file_count": archived_files,
        "cutoff_days": days,
        "archive_path": str(archive_path),
        "error": "",
    }


def prune_transcripts(
    transcripts_path: Path | None = None,
    archive_path: Path | None = None,
    days: int = 7,
    keep: int = 20,
) -> dict[str, Any]:
    transcripts_path, archive_path = _paths(transcripts_path, archive_path)
    error = _safety_error(transcripts_path, archive_path)
    if error:
        return _with_error(_prune_result(transcripts_path, archive_path, days, keep), error)
    archive_result = archive_older_than(transcripts_path, archive_path, days=days, keep_min=keep)
    live_count = len(group_transcripts_by_conversation_id(list_transcript_files(transcripts_path)))
    return {
        "days": int(days),
        "keep": int(keep),
        "archived_conversation_count": archive_result.get("archived_conversation_count", 0),
        "archived_file_count": archive_result.get("archived_file_count", 0),
        "live_conversation_count_after": live_count,
        "transcripts_path": str(transcripts_path),
        "archive_path": str(archive_path),
        "error": archive_result.get("error", ""),
    }


def clear_transcripts(transcripts_path: Path | None = None, preserve_latest: bool = True) -> dict[str, Any]:
    transcripts_path, archive_path = _paths(transcripts_path, None)
    error = _safety_error(transcripts_path, archive_path)
    if error:
        return _with_error(_clear_result(transcripts_path), error)
    deleted = 0
    preserved: list[str] = []
    for file_path in list_transcript_files(transcripts_path):
        if preserve_latest and _is_latest_file(file_path):
            preserved.append(file_path.name)
            continue
        if _is_normal_transcript_file(file_path):
            file_path.unlink(missing_ok=True)
            deleted += 1
    return {
        "cleared": True,
        "deleted_file_count": deleted,
        "preserved_files": preserved,
        "transcripts_path": str(transcripts_path),
        "error": "",
    }


def archive_all_transcripts(
    transcripts_path: Path | None = None,
    archive_path: Path | None = None,
    preserve_latest: bool = True,
) -> dict[str, Any]:
    transcripts_path, archive_path = _paths(transcripts_path, archive_path)
    error = _safety_error(transcripts_path, archive_path)
    if error:
        return _with_error(_archive_result(transcripts_path, archive_path, 0), error)
    archive_path.mkdir(parents=True, exist_ok=True)
    groups = group_transcripts_by_conversation_id(list_transcript_files(transcripts_path))
    archived_files = 0
    for group in groups.values():
        for file_path in group.files:
            if preserve_latest and _is_latest_file(file_path):
                continue
            _move_to_archive(file_path, archive_path)
            archived_files += 1
    return {
        "archived_conversation_count": len(groups),
        "archived_file_count": archived_files,
        "transcripts_path": str(transcripts_path),
        "archive_path": str(archive_path),
        "error": "",
    }


def format_transcript_result(title: str, result: dict[str, Any]) -> str:
    lines = [title]
    for key, value in result.items():
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, list):
            rendered = "[" + ", ".join(str(item) for item in value) + "]"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


def parse_keep_arg(text: str) -> int:
    raw = text.strip()
    return int(raw) if raw else 20


def parse_archive_days_arg(text: str) -> int:
    raw = text.strip()
    return int(raw) if raw else 7


def parse_prune_args(text: str) -> tuple[int, int]:
    raw = text.strip()
    if not raw:
        return 7, 20
    parts = raw.split()
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return int(parts[0]), int(parts[1])
    days = 7
    keep = 20
    index = 0
    while index < len(parts):
        token = parts[index]
        if token == "--days":
            if index + 1 >= len(parts) or not parts[index + 1].isdigit():
                raise ValueError("--days requires an integer.")
            days = int(parts[index + 1])
            index += 2
            continue
        if token == "--keep":
            if index + 1 >= len(parts) or not parts[index + 1].isdigit():
                raise ValueError("--keep requires an integer.")
            keep = int(parts[index + 1])
            index += 2
            continue
        raise ValueError(f"Unknown /transcripts_prune option: {token}")
    return days, keep


def _paths(transcripts_path: Path | None, archive_path: Path | None) -> tuple[Path, Path]:
    return transcripts_path or TRANSCRIPTS_PATH, archive_path or ARCHIVE_PATH


def _groups_sorted_newest(transcripts_path: Path) -> list[TranscriptGroup]:
    return sorted(
        group_transcripts_by_conversation_id(list_transcript_files(transcripts_path)).values(),
        key=lambda group: group.newest_mtime,
        reverse=True,
    )


def _is_latest_file(file_path: Path) -> bool:
    return file_path.name in LATEST_FILES


def _is_normal_transcript_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in {".json", ".md"} and file_path.name not in LATEST_FILES and bool(
        NORMAL_TRANSCRIPT_PATTERN.fullmatch(file_path.name)
    )


def _move_to_archive(file_path: Path, archive_path: Path) -> Path:
    destination = archive_path / file_path.name
    counter = 1
    while destination.exists():
        destination = archive_path / f"{file_path.stem}.{counter}{file_path.suffix}"
        counter += 1
    shutil.move(str(file_path), str(destination))
    return destination


def _safety_error(transcripts_path: Path, archive_path: Path) -> str:
    try:
        transcripts = transcripts_path.resolve()
        archive = archive_path.resolve()
        shared_bus = SHARED_BUS_DIR.resolve()
    except OSError as exc:
        return f"Refusing transcript cleanup: unsafe transcript path detected: {transcripts_path} ({exc})"
    if transcripts.name != "transcripts" or transcripts.parent.name != "shared_agent_bus":
        return f"Refusing transcript cleanup: unsafe transcript path detected: {transcripts_path}"
    if archive.name != "transcripts_archive" or archive.parent.name != "shared_agent_bus":
        return f"Refusing transcript cleanup: unsafe transcript path detected: {archive_path}"
    if not _is_relative_to(transcripts, shared_bus) or not _is_relative_to(archive, shared_bus):
        return f"Refusing transcript cleanup: unsafe transcript path detected: {transcripts_path}"
    forbidden_markers = (
        "shared_fdr_churchill_bus",
        "fdr-v1-agent",
        "churchill-v1-agent",
        "astrid-v1-agent",
        "data\\memory",
        "data\\source_docs",
        "data\\vector_store",
    )
    combined = (str(transcripts) + "\n" + str(archive)).lower()
    for marker in forbidden_markers:
        if marker.lower() in combined:
            return f"Refusing transcript cleanup: unsafe transcript path detected: {transcripts_path}"
    return ""


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _mtime_iso(file_path: Path) -> str:
    return datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()


def _with_error(result: dict[str, Any], error: str) -> dict[str, Any]:
    result["error"] = error
    return result


def _keep_result(transcripts_path: Path, keep_n: int) -> dict[str, Any]:
    return {
        "kept_conversation_count": 0,
        "removed_conversation_count": 0,
        "removed_file_count": 0,
        "transcripts_path": str(transcripts_path),
        "error": "",
    }


def _archive_result(transcripts_path: Path, archive_path: Path, days: int) -> dict[str, Any]:
    return {
        "archived_conversation_count": 0,
        "archived_file_count": 0,
        "cutoff_days": int(days),
        "archive_path": str(archive_path),
        "transcripts_path": str(transcripts_path),
        "error": "",
    }


def _prune_result(transcripts_path: Path, archive_path: Path, days: int, keep: int) -> dict[str, Any]:
    return {
        "days": int(days),
        "keep": int(keep),
        "archived_conversation_count": 0,
        "archived_file_count": 0,
        "live_conversation_count_after": 0,
        "transcripts_path": str(transcripts_path),
        "archive_path": str(archive_path),
        "error": "",
    }


def _clear_result(transcripts_path: Path) -> dict[str, Any]:
    return {
        "cleared": False,
        "deleted_file_count": 0,
        "preserved_files": [],
        "transcripts_path": str(transcripts_path),
        "error": "",
    }
