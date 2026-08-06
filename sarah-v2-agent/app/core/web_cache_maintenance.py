from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from app.core.config import PROJECT_ROOT


CACHE_PATH = PROJECT_ROOT / "data" / "web_cache"
ARCHIVE_PATH = PROJECT_ROOT / "data" / "web_cache_archive"


def get_sarah_web_cache_paths() -> tuple[Path, Path]:
    return CACHE_PATH, ARCHIVE_PATH


def list_web_cache_files(cache_path: Path | None = None) -> list[Path]:
    path = cache_path or CACHE_PATH
    if not path.exists():
        return []
    return sorted([item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".json"])


def web_cache_status(cache_path: Path | None = None, archive_path: Path | None = None) -> dict[str, Any]:
    cache_path, archive_path = _paths(cache_path, archive_path)
    error = _safety_error(cache_path, archive_path)
    if error:
        return _with_error(_base_status(cache_path, archive_path), error)
    cache_path.mkdir(parents=True, exist_ok=True)
    files = list_web_cache_files(cache_path)
    sorted_files = sorted(files, key=lambda path: path.stat().st_mtime)
    validation = validate_web_cache_json(cache_path)
    total_size = sum(path.stat().st_size for path in files)
    return {
        **_base_status(cache_path, archive_path),
        "file_count_total": len(files),
        "json_count": len(files),
        "total_size_kb": round(total_size / 1024, 2),
        "oldest_file": sorted_files[0].name if sorted_files else "",
        "oldest_modified": _mtime_iso(sorted_files[0]) if sorted_files else "",
        "newest_file": sorted_files[-1].name if sorted_files else "",
        "newest_modified": _mtime_iso(sorted_files[-1]) if sorted_files else "",
        "corrupt_json_count": validation.get("corrupt_json_count", 0),
        "corrupt_json_examples": [item["filename"] for item in validation.get("corrupt_files", [])[:5]],
        "cache_exists": cache_path.exists(),
        "archive_exists": archive_path.exists(),
        "error": "",
    }


def validate_web_cache_json(cache_path: Path | None = None) -> dict[str, Any]:
    cache_path, archive_path = _paths(cache_path, None)
    error = _safety_error(cache_path, archive_path)
    if error:
        return _with_error(_validation_result(cache_path), error)
    files = list_web_cache_files(cache_path)
    corrupt: list[dict[str, str]] = []
    for file_path in files:
        try:
            json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            corrupt.append({"filename": file_path.name, "error": str(exc)})
    return {
        "agent": "Sarah v2.0",
        "cache_path": str(cache_path),
        "json_count": len(files),
        "valid_json_count": len(files) - len(corrupt),
        "corrupt_json_count": len(corrupt),
        "corrupt_files": corrupt,
        "status": "OK" if not corrupt else "CORRUPT",
        "error": "",
    }


def keep_newest_cache_files(cache_path: Path | None = None, keep_n: int = 200) -> dict[str, Any]:
    cache_path, archive_path = _paths(cache_path, None)
    error = _safety_error(cache_path, archive_path)
    if error:
        return _with_error(_keep_result(cache_path), error)
    keep_n = max(0, int(keep_n))
    files = _files_sorted_newest(cache_path)
    keep = set(files[:keep_n])
    deleted = 0
    for file_path in files:
        if file_path in keep:
            continue
        file_path.unlink(missing_ok=True)
        deleted += 1
    return {
        "kept_file_count": min(len(files), keep_n),
        "deleted_file_count": deleted,
        "cache_path": str(cache_path),
        "error": "",
    }


def archive_cache_older_than(
    cache_path: Path | None = None,
    archive_path: Path | None = None,
    days: int = 14,
    keep_min: int | None = None,
) -> dict[str, Any]:
    cache_path, archive_path = _paths(cache_path, archive_path)
    error = _safety_error(cache_path, archive_path)
    if error:
        return _with_error(_archive_result(cache_path, archive_path, days), error)
    days = max(0, int(days))
    keep_min = max(0, int(keep_min)) if keep_min is not None else None
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    files = _files_sorted_newest(cache_path)
    protected = set(files[:keep_min]) if keep_min is not None else set()
    archive_path.mkdir(parents=True, exist_ok=True)
    archived = 0
    for file_path in files:
        if file_path in protected:
            continue
        if datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc) >= cutoff:
            continue
        _move_to_archive(file_path, archive_path)
        archived += 1
    return {
        "archived_file_count": archived,
        "cutoff_days": days,
        "cache_path": str(cache_path),
        "archive_path": str(archive_path),
        "error": "",
    }


def prune_web_cache(
    cache_path: Path | None = None,
    archive_path: Path | None = None,
    days: int = 14,
    keep: int = 200,
) -> dict[str, Any]:
    cache_path, archive_path = _paths(cache_path, archive_path)
    error = _safety_error(cache_path, archive_path)
    if error:
        return _with_error(_prune_result(cache_path, archive_path, days, keep), error)
    archive_result = archive_cache_older_than(cache_path, archive_path, days=days, keep_min=keep)
    live_count = len(list_web_cache_files(cache_path))
    return {
        "days": int(days),
        "keep": int(keep),
        "archived_file_count": archive_result.get("archived_file_count", 0),
        "live_file_count_after": live_count,
        "cache_path": str(cache_path),
        "archive_path": str(archive_path),
        "error": archive_result.get("error", ""),
    }


def clear_web_cache(cache_path: Path | None = None) -> dict[str, Any]:
    cache_path, archive_path = _paths(cache_path, None)
    error = _safety_error(cache_path, archive_path)
    if error:
        return _with_error(_clear_result(cache_path), error)
    deleted = 0
    for file_path in list_web_cache_files(cache_path):
        file_path.unlink(missing_ok=True)
        deleted += 1
    return {
        "cleared": True,
        "deleted_file_count": deleted,
        "cache_path": str(cache_path),
        "error": "",
    }


def archive_all_web_cache(cache_path: Path | None = None, archive_path: Path | None = None) -> dict[str, Any]:
    cache_path, archive_path = _paths(cache_path, archive_path)
    error = _safety_error(cache_path, archive_path)
    if error:
        return _with_error(_archive_all_result(cache_path, archive_path), error)
    archive_path.mkdir(parents=True, exist_ok=True)
    archived = 0
    for file_path in list_web_cache_files(cache_path):
        _move_to_archive(file_path, archive_path)
        archived += 1
    return {
        "archived_file_count": archived,
        "cache_path": str(cache_path),
        "archive_path": str(archive_path),
        "error": "",
    }


def delete_corrupt_web_cache(cache_path: Path | None = None) -> dict[str, Any]:
    cache_path, archive_path = _paths(cache_path, None)
    error = _safety_error(cache_path, archive_path)
    if error:
        return _with_error(_delete_corrupt_result(cache_path), error)
    validation = validate_web_cache_json(cache_path)
    deleted: list[str] = []
    corrupt_names = {item["filename"] for item in validation.get("corrupt_files", [])}
    for file_path in list_web_cache_files(cache_path):
        if file_path.name not in corrupt_names:
            continue
        file_path.unlink(missing_ok=True)
        deleted.append(file_path.name)
    return {
        "deleted_corrupt_count": len(deleted),
        "cache_path": str(cache_path),
        "deleted_files": deleted,
        "error": "",
    }


def delete_news_cache_older_than(cache_path: Path | None = None, days: int = 5) -> dict[str, Any]:
    cache_path, archive_path = _paths(cache_path, None)
    error = _safety_error(cache_path, archive_path)
    if error:
        return _with_error(_delete_news_result(cache_path, days), error)
    days = max(0, int(days))
    if not cache_path.exists():
        return {
            **_delete_news_result(cache_path, days),
            "cache_exists": False,
            "error": "",
        }
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    deleted: list[str] = []
    for file_path in list_web_cache_files(cache_path):
        modified_at = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
        if modified_at >= cutoff:
            continue
        if not _is_news_cache_file(file_path):
            continue
        file_path.unlink(missing_ok=True)
        deleted.append(file_path.name)
    return {
        "deleted_news_count": len(deleted),
        "cutoff_days": days,
        "cache_path": str(cache_path),
        "cache_exists": True,
        "deleted_files": deleted,
        "error": "",
    }


def format_web_cache_result(title: str, result: dict[str, Any]) -> str:
    lines = [title]
    corrupt_files = result.get("corrupt_files")
    for key, value in result.items():
        if key == "corrupt_files":
            continue
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, list):
            rendered = "[" + ", ".join(str(item) for item in value) + "]"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    if isinstance(corrupt_files, list):
        lines.append("corrupt_files:")
        if corrupt_files:
            for item in corrupt_files:
                lines.append(f"* filename: {item.get('filename', '')}")
                lines.append(f"  error: {item.get('error', '')}")
        else:
            lines.append("[]")
    return "\n".join(lines)


def parse_keep_arg(text: str) -> int:
    raw = text.strip()
    return int(raw) if raw else 200


def parse_archive_days_arg(text: str) -> int:
    raw = text.strip()
    return int(raw) if raw else 14


def parse_prune_args(text: str) -> tuple[int, int]:
    raw = text.strip()
    if not raw:
        return 14, 200
    parts = raw.split()
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return int(parts[0]), int(parts[1])
    days = 14
    keep = 200
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
        raise ValueError(f"Unknown /web_cache_prune option: {token}")
    return days, keep


def _paths(cache_path: Path | None, archive_path: Path | None) -> tuple[Path, Path]:
    return cache_path or CACHE_PATH, archive_path or ARCHIVE_PATH


def _files_sorted_newest(cache_path: Path) -> list[Path]:
    return sorted(list_web_cache_files(cache_path), key=lambda path: path.stat().st_mtime, reverse=True)


def _move_to_archive(file_path: Path, archive_path: Path) -> Path:
    destination = archive_path / file_path.name
    counter = 1
    while destination.exists():
        destination = archive_path / f"{file_path.stem}.{counter}{file_path.suffix}"
        counter += 1
    shutil.move(str(file_path), str(destination))
    return destination


def _is_news_cache_file(file_path: Path) -> bool:
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if "news" in str(data.get("reason", "")).lower():
        return True
    for source in data.get("sources", []):
        if isinstance(source, dict) and source.get("kind") == "news":
            return True
    return False


def _safety_error(cache_path: Path, archive_path: Path) -> str:
    try:
        cache = cache_path.resolve()
        archive = archive_path.resolve()
        data_dir = (PROJECT_ROOT / "data").resolve()
        expected_cache = (PROJECT_ROOT / "data" / "web_cache").resolve()
        expected_archive = (PROJECT_ROOT / "data" / "web_cache_archive").resolve()
    except OSError as exc:
        return f"Refusing web-cache cleanup: unsafe cache path detected: {cache_path} ({exc})"
    if cache != expected_cache or archive != expected_archive:
        return f"Refusing web-cache cleanup: unsafe cache path detected: {cache_path}"
    if not _is_relative_to(cache, data_dir) or not _is_relative_to(archive, data_dir):
        return f"Refusing web-cache cleanup: unsafe cache path detected: {cache_path}"
    forbidden_markers = (
        "shared_agent_bus",
        "shared_fdr_churchill_bus",
        "astrid-v1-agent",
        "fdr-v1-agent",
        "churchill-v1-agent",
        "data\\memory",
        "data\\source_docs",
        "data\\vector_store",
        "transcripts",
    )
    combined = (str(cache) + "\n" + str(archive)).lower()
    for marker in forbidden_markers:
        if marker.lower() in combined:
            return f"Refusing web-cache cleanup: unsafe cache path detected: {cache_path}"
    return ""


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _mtime_iso(file_path: Path) -> str:
    return datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()


def _base_status(cache_path: Path, archive_path: Path) -> dict[str, Any]:
    return {
        "agent": "Sarah v2.0",
        "cache_path": str(cache_path),
        "archive_path": str(archive_path),
        "file_count_total": 0,
        "json_count": 0,
        "total_size_kb": 0,
        "oldest_file": "",
        "oldest_modified": "",
        "newest_file": "",
        "newest_modified": "",
        "corrupt_json_count": 0,
        "corrupt_json_examples": [],
        "cache_exists": cache_path.exists(),
        "archive_exists": archive_path.exists(),
        "error": "",
    }


def _validation_result(cache_path: Path) -> dict[str, Any]:
    return {
        "agent": "Sarah v2.0",
        "cache_path": str(cache_path),
        "json_count": 0,
        "valid_json_count": 0,
        "corrupt_json_count": 0,
        "corrupt_files": [],
        "status": "",
        "error": "",
    }


def _keep_result(cache_path: Path) -> dict[str, Any]:
    return {"kept_file_count": 0, "deleted_file_count": 0, "cache_path": str(cache_path), "error": ""}


def _archive_result(cache_path: Path, archive_path: Path, days: int) -> dict[str, Any]:
    return {
        "archived_file_count": 0,
        "cutoff_days": int(days),
        "cache_path": str(cache_path),
        "archive_path": str(archive_path),
        "error": "",
    }


def _prune_result(cache_path: Path, archive_path: Path, days: int, keep: int) -> dict[str, Any]:
    return {
        "days": int(days),
        "keep": int(keep),
        "archived_file_count": 0,
        "live_file_count_after": 0,
        "cache_path": str(cache_path),
        "archive_path": str(archive_path),
        "error": "",
    }


def _clear_result(cache_path: Path) -> dict[str, Any]:
    return {"cleared": False, "deleted_file_count": 0, "cache_path": str(cache_path), "error": ""}


def _archive_all_result(cache_path: Path, archive_path: Path) -> dict[str, Any]:
    return {"archived_file_count": 0, "cache_path": str(cache_path), "archive_path": str(archive_path), "error": ""}


def _delete_corrupt_result(cache_path: Path) -> dict[str, Any]:
    return {"deleted_corrupt_count": 0, "cache_path": str(cache_path), "deleted_files": [], "error": ""}


def _delete_news_result(cache_path: Path, days: int) -> dict[str, Any]:
    return {
        "deleted_news_count": 0,
        "cutoff_days": int(days),
        "cache_path": str(cache_path),
        "cache_exists": cache_path.exists(),
        "deleted_files": [],
        "error": "",
    }


def _with_error(result: dict[str, Any], error: str) -> dict[str, Any]:
    result["error"] = error
    return result
