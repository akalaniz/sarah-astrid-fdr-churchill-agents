from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import Settings, load_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_BUS_PATH = PROJECT_ROOT.parent / "shared_fdr_churchill_bus"
PROMPT_PATH = PROJECT_ROOT / "config" / "churchill_master_prompt.md"
FORBIDDEN_PATH_MARKERS = (
    "sarah-v2-agent",
    "astrid-v1-agent",
    "shared_agent_bus",
    "sarah_memory.jsonl",
    "astrid_memory.jsonl",
)


def debug_worldline(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    paths = {
        "memory_path": settings.memory_file,
        "source_docs_path": settings.source_docs_dir,
        "vector_store_path": settings.vector_store_dir,
        "prompt_path": PROMPT_PATH,
        "shared_bus_path": SHARED_BUS_PATH,
    }
    offending = _find_spillover(paths)
    return {
        "agent_name": "Churchill v1.0",
        "worldline": "fdr_churchill_historical",
        "port": 8011,
        "memory_path": str(paths["memory_path"]),
        "source_docs_path": str(paths["source_docs_path"]),
        "vector_store_path": str(paths["vector_store_path"]),
        "prompt_path": str(paths["prompt_path"]),
        "shared_bus_path": str(paths["shared_bus_path"]),
        "paired_agent": "FDR",
        "paired_agent_status": "paired",
        "paired_agent_url": "http://127.0.0.1:8010",
        "spillover_detected": bool(offending),
        "offending_path": offending or "",
    }


def format_debug_worldline(settings: Settings | None = None) -> str:
    return "\n".join(f"{key}: {value}" for key, value in debug_worldline(settings).items())


def list_source_doc_names(settings: Settings | None = None) -> list[str]:
    settings = settings or load_settings()
    if not settings.source_docs_dir.exists():
        return []
    return sorted(path.name for path in settings.source_docs_dir.iterdir() if path.is_file() and path.name != ".gitkeep")


def format_sources_list(settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    names = list_source_doc_names(settings)
    lines = [f"source_docs_path: {settings.source_docs_dir}"]
    lines.extend(names or ["No Churchill source documents found."])
    return "\n".join(lines)


def debug_rag(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    return {
        "source_docs_path": str(settings.source_docs_dir),
        "source_docs_exists": settings.source_docs_dir.exists(),
        "source_doc_count": len(list_source_doc_names(settings)),
        "vector_store_path": str(settings.vector_store_dir),
        "vector_store_exists": settings.vector_store_dir.exists(),
    }


def format_debug_rag(settings: Settings | None = None) -> str:
    return "\n".join(f"{key}: {value}" for key, value in debug_rag(settings).items())


def _find_spillover(paths: dict[str, Path]) -> str:
    for path in paths.values():
        value = str(path).lower()
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker.lower() in value:
                return str(path)
    return ""
