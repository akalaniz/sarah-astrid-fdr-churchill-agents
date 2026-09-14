from __future__ import annotations

import atexit
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import logging
import math
import os
import re
import threading
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from openai import OpenAI


LOCAL_EMBEDDING_MODEL = "local-hash-embedding-v1"
LOCAL_EMBEDDING_DIMENSIONS = 384
OPENAI_EMBEDDING_BATCH_SIZE = 64
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(repr=False)
class _EmbeddingClient:
    client: OpenAI
    configuration: tuple[tuple[str, str], ...]
    users: int = 0
    retired: bool = False


_client_lock = threading.Lock()
_cached_client: _EmbeddingClient | None = None
_TRANSPORT_ENV_VARS = {
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR",
}


def _client_configuration(openai_api_key: str) -> tuple[tuple[str, str], ...]:
    # Keep credentials and transport settings isolated; never log this cache key.
    environment = tuple(sorted(
        (name, value) for name, value in os.environ.items()
        if name.upper().startswith("OPENAI_") or name.upper() in _TRANSPORT_ENV_VARS
    ))
    return (("api_key", openai_api_key),) + environment


def _close_client(entry: _EmbeddingClient) -> None:
    try:
        entry.client.close()
    except Exception:
        logging.getLogger(__name__).warning("An unused embedding client could not be closed.")


def close_embedding_client() -> None:
    global _cached_client
    with _client_lock:
        entry = _cached_client
        _cached_client = None
        if entry is not None:
            entry.retired = True
        close_now = entry is not None and entry.users == 0
    if close_now:
        _close_client(entry)


@contextmanager
def _borrow_embedding_client(openai_api_key: str) -> Iterator[OpenAI]:
    global _cached_client
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI embeddings require openai. Run: python -m pip install -e .") from exc

    configuration = _client_configuration(openai_api_key)
    retired = None
    with _client_lock:
        entry = _cached_client
        if entry is None or entry.configuration != configuration or entry.client.is_closed():
            replacement = _EmbeddingClient(OpenAI(api_key=openai_api_key), configuration)
            if entry is not None:
                entry.retired = True
                if entry.users == 0:
                    retired = entry
            _cached_client = entry = replacement
        entry.users += 1
    if retired is not None:
        _close_client(retired)
    try:
        # Requests run outside the lock, so retrieval and ingestion can share the HTTP pool.
        yield entry.client
    finally:
        with _client_lock:
            entry.users -= 1
            close_now = entry.retired and entry.users == 0
        if close_now:
            _close_client(entry)


atexit.register(close_embedding_client)


def get_embedding_model_name(openai_api_key: str, configured_model: str) -> str:
    return configured_model if openai_api_key else LOCAL_EMBEDDING_MODEL


def embed_texts(texts: list[str], openai_api_key: str, model: str) -> list[list[float]]:
    if not texts:
        return []
    if openai_api_key:
        return _openai_embed_texts(texts, openai_api_key, model)
    return [_local_embed_text(text) for text in texts]


def _openai_embed_texts(texts: list[str], openai_api_key: str, model: str) -> list[list[float]]:
    embeddings: list[list[float]] = []
    with _borrow_embedding_client(openai_api_key) as client:
        for start in range(0, len(texts), OPENAI_EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + OPENAI_EMBEDDING_BATCH_SIZE]
            response = client.embeddings.create(model=model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
    return embeddings


def _local_embed_text(text: str) -> list[float]:
    vector = [0.0] * LOCAL_EMBEDDING_DIMENSIONS
    tokens = TOKEN_PATTERN.findall(text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % LOCAL_EMBEDDING_DIMENSIONS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    return _normalize(vector)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))
