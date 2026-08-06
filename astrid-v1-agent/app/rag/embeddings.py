from __future__ import annotations

import hashlib
import math
import re


LOCAL_EMBEDDING_MODEL = "local-hash-embedding-v1"
LOCAL_EMBEDDING_DIMENSIONS = 384
OPENAI_EMBEDDING_BATCH_SIZE = 64
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def get_embedding_model_name(openai_api_key: str, configured_model: str) -> str:
    return configured_model if openai_api_key else LOCAL_EMBEDDING_MODEL


def embed_texts(texts: list[str], openai_api_key: str, model: str) -> list[list[float]]:
    if not texts:
        return []
    if openai_api_key:
        return _openai_embed_texts(texts, openai_api_key, model)
    return [_local_embed_text(text) for text in texts]


def _openai_embed_texts(texts: list[str], openai_api_key: str, model: str) -> list[list[float]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI embeddings require openai. Run: python -m pip install -e .") from exc

    client = OpenAI(api_key=openai_api_key)
    embeddings: list[list[float]] = []
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
