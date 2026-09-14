from __future__ import annotations

from typing import Any, Callable

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

    def create_response(
        self,
        model: str,
        messages: list[dict[str, str]],
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        text_verbosity: str | None = None,
    ) -> str:
        if hasattr(self._client, "responses"):
            request: dict[str, Any] = {
                "model": model,
                "input": _responses_input(messages),
            }
            if reasoning_effort:
                request["reasoning"] = {"effort": reasoning_effort}
            if max_output_tokens is not None:
                request["max_output_tokens"] = max_output_tokens
            if text_verbosity:
                request["text"] = {"verbosity": text_verbosity}
            response = self._client.responses.create(**request)
            text = getattr(response, "output_text", "")
            if text:
                return normalize_sarah_output(text)
            return normalize_sarah_output(_extract_responses_text(response))

        completion = self._client.chat.completions.create(
            model=model,
            messages=messages,
            **({"reasoning_effort": reasoning_effort} if reasoning_effort and text_verbosity else {}),
            **({"verbosity": text_verbosity} if text_verbosity else {}),
        )
        return normalize_sarah_output(completion.choices[0].message.content)


    def stream_response(
        self,
        model: str,
        messages: list[dict[str, str]],
        on_delta: Callable[[str], None],
        reasoning_effort: str | None = None,
        text_verbosity: str | None = None,
    ) -> str:
        parts: list[str] = []
        completed = False
        final_text = ""
        if hasattr(self._client, "responses"):
            with self._client.responses.create(
                model=model, input=_responses_input(messages), stream=True,
                **({"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}),
                **({"text": {"verbosity": text_verbosity}} if text_verbosity else {}),
            ) as stream:
                for event in stream:
                    on_delta("")  # Also check for disconnects during non-text events.
                    if event.type in {"response.output_text.delta", "response.refusal.delta"}:
                        parts.append(event.delta)
                        on_delta(event.delta)
                    elif event.type == "response.completed":
                        completed = True
                        response = event.response
                        final_text = getattr(response, "output_text", "") or _extract_responses_text(response)
                    elif event.type in {"response.failed", "response.incomplete", "error"}:
                        raise RuntimeError("The model response did not complete.")
        else:
            with self._client.chat.completions.create(
                model=model, messages=messages, stream=True,
                **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
                **({"verbosity": text_verbosity} if text_verbosity else {}),
            ) as stream:
                for event in stream:
                    on_delta("")
                    for choice in event.choices:
                        text = getattr(choice.delta, "content", None) or getattr(choice.delta, "refusal", None)
                        if text:
                            parts.append(text)
                            on_delta(text)
                        if choice.finish_reason is not None:
                            if choice.finish_reason != "stop":
                                raise RuntimeError("The model response did not complete.")
                            completed = True
        if not completed:
            raise RuntimeError("The model stream ended before completion.")
        return normalize_sarah_output(final_text or "".join(parts))


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
