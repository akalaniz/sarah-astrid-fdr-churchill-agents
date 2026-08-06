from __future__ import annotations


def normalize_sarah_output(text: str) -> str:
    text = text.replace("*", "")
    text = text.replace("#", "")
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()
