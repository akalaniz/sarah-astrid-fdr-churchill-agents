from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT


DEFAULT_CONSTITUTION_PATH = PROJECT_ROOT / "config" / "fdr_constitution.yaml"


@dataclass(frozen=True)
class FDRConstitution:
    identity: list[str]
    never: list[str]
    always: list[str]
    relationship_modes: dict[str, dict[str, str]]
    humor_rules: list[str]
    geopolitical_rules: list[str]
    adult_intimacy_mode: dict[str, Any]


def load_sarah_constitution(path: Path = DEFAULT_CONSTITUTION_PATH) -> FDRConstitution:
    if not path.exists():
        raise FileNotFoundError(f"FDR constitution file not found: {path}")

    raw = _read_yaml(path)
    return FDRConstitution(
        identity=_list_of_strings(raw, "identity"),
        never=_list_of_strings(raw, "never"),
        always=_list_of_strings(raw, "always"),
        relationship_modes=_relationship_modes(raw.get("relationship_modes", {})),
        humor_rules=_list_of_strings(raw, "humor_rules"),
        geopolitical_rules=_list_of_strings(raw, "geopolitical_rules"),
        adult_intimacy_mode=_mapping(raw, "adult_intimacy_mode"),
    )


def build_constitution_prompt_section(
    constitution: FDRConstitution | None = None,
) -> str:
    constitution = constitution or load_sarah_constitution()
    lines = ["SARAH CONSTITUTION:"]
    lines.extend(_bulleted_section("identity", constitution.identity))
    lines.extend(_bulleted_section("never", constitution.never))
    lines.extend(_bulleted_section("always", constitution.always))

    lines.append("relationship_modes:")
    for name, values in constitution.relationship_modes.items():
        lines.append(f"- {name}: {values['mode']}")

    lines.extend(_bulleted_section("humor_rules", constitution.humor_rules))
    lines.extend(_bulleted_section("geopolitical_rules", constitution.geopolitical_rules))
    lines.extend(_adult_intimacy_section(constitution.adult_intimacy_mode))
    return "\n".join(lines)


def _list_of_strings(raw: dict[str, Any], key: str) -> list[str]:
    values = raw.get(key, [])
    if not isinstance(values, list):
        raise ValueError(f"Expected '{key}' to be a list in FDR constitution.")
    return [str(value) for value in values]


def _relationship_modes(raw: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        raise ValueError("Expected 'relationship_modes' to be a mapping in FDR constitution.")

    modes: dict[str, dict[str, str]] = {}
    for name, values in raw.items():
        if not isinstance(values, dict) or "mode" not in values:
            raise ValueError(f"Expected relationship mode for '{name}' to include 'mode'.")
        modes[str(name)] = {"mode": str(values["mode"])}
    return modes


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    values = raw.get(key, {})
    if not isinstance(values, dict):
        raise ValueError(f"Expected '{key}' to be a mapping in FDR constitution.")
    return {name: value for name, value in values.items() if not name.startswith("_")}


def _bulleted_section(name: str, values: list[str]) -> list[str]:
    return [f"{name}:"] + [f"- {value}" for value in values]


def _adult_intimacy_section(values: dict[str, Any]) -> list[str]:
    if not values:
        return []

    lines = ["adult_intimacy_mode:"]
    for key in ("model_source", "purpose"):
        if key in values:
            lines.append(f"- {key}: {values[key]}")
    for key in ("allowed", "forbidden", "style_rules"):
        items = values.get(key, [])
        if isinstance(items, list):
            lines.append(f"- {key}:")
            lines.extend(f"  - {item}" for item in items)
    return lines


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _parse_sarah_constitution_yaml(text)

    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Expected FDR constitution YAML to contain a mapping.")
    return loaded


def _parse_sarah_constitution_yaml(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    current_section: str | None = None
    current_relation: str | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0 and line.endswith(":"):
            current_section = line[:-1]
            current_relation = None
            parsed[current_section] = {} if current_section in {"relationship_modes", "adult_intimacy_mode"} else []
            continue

        if current_section is None:
            continue

        if current_section == "relationship_modes":
            if indent == 2 and line.endswith(":"):
                current_relation = line[:-1]
                parsed[current_section][current_relation] = {}
            elif indent == 4 and current_relation and ":" in line:
                key, value = line.split(":", 1)
                parsed[current_section][current_relation][key.strip()] = value.strip()
            continue

        if current_section == "adult_intimacy_mode":
            _parse_adult_intimacy_line(parsed[current_section], indent, line)
            continue

        if indent == 2 and line.startswith("- "):
            parsed[current_section].append(_unquote(line[2:].strip()))

    return parsed


def _parse_adult_intimacy_line(section: dict[str, Any], indent: int, line: str) -> None:
    if indent == 2 and ":" in line and not line.startswith("- "):
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        section[key] = _unquote(value) if value else []
        section["_current_list"] = key if not value else None
        return

    current_list = section.get("_current_list")
    if indent == 4 and current_list and line.startswith("- "):
        section.setdefault(current_list, []).append(_unquote(line[2:].strip()))


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
