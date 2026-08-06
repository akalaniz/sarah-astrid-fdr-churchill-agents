from __future__ import annotations

from typing import Any

from app.core.text_output import normalize_sarah_output


class SarahOpenAIClient:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to .env before chatting with Astrid.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI SDK is not installed. Run: python -m pip install -e .") from exc

        self._client = OpenAI(api_key=api_key)

    def create_response(self, model: str, messages: list[dict[str, str]]) -> str:
        if hasattr(self._client, "responses"):
            response = self._client.responses.create(
                model=model,
                input=_responses_input(messages),
            )
            text = getattr(response, "output_text", "")
            if text:
                return normalize_sarah_output(text)
            return normalize_sarah_output(_extract_responses_text(response))

        completion = self._client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return normalize_sarah_output(completion.choices[0].message.content)


def _responses_input(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in messages
    ]


def _extract_responses_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)
