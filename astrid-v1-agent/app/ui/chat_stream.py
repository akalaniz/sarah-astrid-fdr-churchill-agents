from __future__ import annotations

import asyncio
import json
import logging
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Callable

from fastapi.responses import StreamingResponse

from app.core.text_output import normalize_sarah_output


logger = logging.getLogger(__name__)


class StreamCancelled(Exception):
    """The browser disconnected before the answer was committed."""


def stream_chat(generate: Callable[[Callable[[str], None]], dict[str, Any]]) -> StreamingResponse:
    events: Queue[tuple[str, dict[str, Any]]] = Queue(maxsize=8)
    cancelled = threading.Event()

    def publish(kind: str, payload: dict[str, Any]) -> None:
        while not cancelled.is_set():
            try:
                events.put((kind, payload), timeout=0.1)
                return
            except Full:
                continue
        raise StreamCancelled()

    def produce() -> None:
        parts: list[str] = []
        last_update = 0.0

        def on_delta(text: str) -> None:
            nonlocal last_update
            if cancelled.is_set():
                raise StreamCancelled()
            if not text:
                return
            parts.append(text)
            now = time.monotonic()
            if now - last_update >= 0.05:
                publish("text", {"text": normalize_sarah_output("".join(parts))})
                last_update = now

        try:
            result = generate(on_delta)
            publish("done", result)
        except StreamCancelled:
            pass
        except Exception as exc:
            logger.warning("Chat stream failed (%s).", type(exc).__name__)
            try:
                publish("error", {"detail": "The response was interrupted. Please try again."})
            except StreamCancelled:
                pass

    async def body():
        worker = threading.Thread(target=produce, name="chat-stream", daemon=True)
        worker.start()
        last_keepalive = time.monotonic()
        try:
            yield ": connected\n\n"
            while True:
                try:
                    kind, payload = events.get_nowait()
                except Empty:
                    if time.monotonic() - last_keepalive >= 10:
                        yield ": keepalive\n\n"
                        last_keepalive = time.monotonic()
                    await asyncio.sleep(0.02)
                    continue
                yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"
                if kind in {"done", "error"}:
                    return
        finally:
            cancelled.set()

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
