from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
EXTERNAL_SOURCE_DOCS = Path(
    r"C:\Users\akala\Documents\Codex\2026-06-03\files-mentioned-by-the-user-pasted\data\source_docs"
)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    sarah_model: str
    embedding_model: str
    transcription_model: str
    whisper_model_size: str
    whisper_cpp_exe: str
    whisper_cpp_model: str
    news_api_mode: str
    project_root: Path
    data_dir: Path
    source_docs_dir: Path
    external_source_docs_dir: Path
    vector_store_dir: Path
    conversations_dir: Path
    web_cache_dir: Path
    memory_file: Path
    log_level: str = "WARNING"


def load_settings(env_file: Path = ENV_FILE) -> Settings:
    env = _load_env_file(env_file)
    env.update(os.environ)

    return Settings(
        openai_api_key=_clean_api_key(env.get("OPENAI_API_KEY", "")),
        sarah_model=env.get("SARAH_MODEL", "gpt-5.2"),
        embedding_model=env.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        transcription_model=env.get("TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"),
        whisper_model_size=env.get("WHISPER_MODEL_SIZE", "base"),
        whisper_cpp_exe=env.get("WHISPER_CPP_EXE", ""),
        whisper_cpp_model=env.get("WHISPER_CPP_MODEL", ""),
        news_api_mode=env.get("NEWS_API_MODE", "off"),
        project_root=PROJECT_ROOT,
        data_dir=PROJECT_ROOT / "data",
        source_docs_dir=PROJECT_ROOT / "data" / "source_docs",
        external_source_docs_dir=EXTERNAL_SOURCE_DOCS,
        vector_store_dir=PROJECT_ROOT / "data" / "vector_store",
        conversations_dir=PROJECT_ROOT / "data" / "conversations",
        web_cache_dir=PROJECT_ROOT / "data" / "web_cache",
        memory_file=PROJECT_ROOT / "data" / "memory" / "sarah_memory.jsonl",
        log_level=env.get("LOG_LEVEL", "WARNING"),
    )


def _load_env_file(env_file: Path) -> dict[str, str]:
    if not env_file.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")

    return values


def _clean_api_key(value: str) -> str:
    placeholder_values = {
        "",
        "your_openai_api_key_here",
        "replace_me",
        "changeme",
    }
    stripped = value.strip()
    if stripped.lower() in placeholder_values:
        return ""
    return stripped
