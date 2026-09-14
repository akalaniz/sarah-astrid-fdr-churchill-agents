from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

from app.rag import embeddings


class FakeClient:
    def __init__(self, **options):
        self.options = options
        self.closed = False
        self.close_count = 0
        self.calls = []
        self.action = None
        self.embeddings = self

    def is_closed(self):
        return self.closed

    def close(self):
        self.close_count += 1
        self.closed = True

    def create(self, *, model, input):
        assert not self.closed
        self.calls.append((model, list(input)))
        if self.action:
            self.action()
        return SimpleNamespace(data=[
            SimpleNamespace(embedding=[float(len(text)), 1.0]) for text in input
        ])


@pytest.fixture
def factory(monkeypatch):
    embeddings.close_embedding_client()
    created = []

    def create(**options):
        client = FakeClient(**options)
        created.append(client)
        return client

    constructor = Mock(side_effect=create)
    monkeypatch.setattr(openai, "OpenAI", constructor)
    yield constructor, created
    embeddings.close_embedding_client()


def embed(text="query", key="test-key", model="test-model"):
    return embeddings.embed_texts([text], key, model)


def test_repeated_calls_reuse_one_client_without_reusing_results(factory):
    constructor, clients = factory
    assert embed("one") == [[3.0, 1.0]]
    assert embed("different") == [[9.0, 1.0]]
    assert constructor.call_count == 1
    assert clients[0].calls == [
        ("test-model", ["one"]), ("test-model", ["different"]),
    ]


def test_batching_order_and_model_are_unchanged(factory):
    constructor, clients = factory
    texts = ["x" * i for i in range(131)]
    assert embeddings.embed_texts(texts, "test-key", "first") == [
        [float(i), 1.0] for i in range(131)
    ]
    embed("again", model="second")
    assert constructor.call_count == 1
    assert [len(items) for _, items in clients[0].calls] == [64, 64, 3, 1]
    assert [model for model, _ in clients[0].calls] == ["first", "first", "first", "second"]


def test_empty_and_local_embeddings_never_create_an_api_client(factory):
    constructor, _ = factory
    assert embeddings.embed_texts([], "test-key", "test-model") == []
    actual = embeddings.embed_texts(["Hello", ""], "", "ignored")
    assert actual == [embeddings._local_embed_text("Hello"), [0.0] * 384]
    assert constructor.call_count == 0


def test_key_rotation_closes_idle_old_client(factory):
    _, clients = factory
    embed(key="first-secret")
    embed(key="second-secret")
    assert len(clients) == 2
    assert clients[0].close_count == 1
    assert not clients[1].closed
    assert clients[1].options["api_key"] == "second-secret"
    assert "second-secret" not in repr(embeddings._cached_client)


@pytest.mark.parametrize("name", [
    "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_CUSTOM_HEADERS",
    "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE",
])
def test_connection_setting_change_replaces_client(factory, monkeypatch, name):
    _, clients = factory
    monkeypatch.delenv(name, raising=False)
    embed()
    monkeypatch.setenv(name, "test-new-setting")
    embed()
    assert len(clients) == 2
    assert clients[0].close_count == 1


def test_unrelated_environment_change_does_not_discard_connection(factory, monkeypatch):
    constructor, _ = factory
    embed()
    monkeypatch.setenv("AGENT_MODEL", "unchanged-embedding-transport")
    embed()
    assert constructor.call_count == 1


def test_closed_client_is_recreated(factory):
    _, clients = factory
    embed()
    clients[0].close()
    embed()
    assert len(clients) == 2
    assert not clients[1].closed


def test_simultaneous_requests_share_client_without_serializing_network(factory):
    constructor, clients = factory
    embed()
    rendezvous = threading.Barrier(4)
    clients[0].action = lambda: rendezvous.wait(timeout=5)
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = [executor.submit(embed, str(i)) for i in range(4)]
        assert [job.result(timeout=10) for job in jobs] == [[[1.0, 1.0]]] * 4
    assert constructor.call_count == 1
    assert embeddings._cached_client.users == 0


def test_simultaneous_first_use_constructs_only_one_client(factory):
    constructor, _ = factory
    rendezvous = threading.Barrier(4)

    def run():
        rendezvous.wait(timeout=5)
        return embed()

    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = [executor.submit(run) for _ in range(4)]
        assert all(job.result(timeout=10) == [[5.0, 1.0]] for job in jobs)
    assert constructor.call_count == 1


def test_key_rotation_does_not_close_an_inflight_request(factory):
    _, clients = factory
    embed(key="old")
    entered, release = threading.Event(), threading.Event()

    def hold():
        entered.set()
        assert release.wait(5)
        assert not clients[0].closed

    clients[0].action = hold
    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(embed, "active", "old")
        try:
            assert entered.wait(5)
            embed(key="new")
            assert not clients[0].closed
            assert clients[1].options["api_key"] == "new"
        finally:
            release.set()
        assert active.result(timeout=5) == [[6.0, 1.0]]
    assert clients[0].close_count == 1
    assert not clients[1].closed


def test_explicit_cleanup_waits_for_active_request_and_allows_new_client(factory):
    _, clients = factory
    with embeddings._borrow_embedding_client("test-key") as borrowed:
        embeddings.close_embedding_client()
        assert not borrowed.closed
        embed()
        assert len(clients) == 2
        assert not borrowed.closed
    assert borrowed.close_count == 1
    embeddings.close_embedding_client()
    embeddings.close_embedding_client()
    assert clients[1].close_count == 1
    assert embeddings._cached_client is None


def test_request_failure_propagates_and_releases_lease(factory):
    constructor, clients = factory
    embed()
    clients[0].action = Mock(side_effect=RuntimeError("test failure"))
    with pytest.raises(RuntimeError, match="test failure"):
        embed()
    assert embeddings._cached_client.users == 0
    clients[0].action = None
    assert embed() == [[5.0, 1.0]]
    assert constructor.call_count == 1


def test_failed_replacement_does_not_send_to_old_credentials(factory):
    constructor, clients = factory
    embed(key="old")
    constructor.side_effect = RuntimeError("constructor failed")
    with pytest.raises(RuntimeError, match="constructor failed"):
        embed(key="new")
    assert len(clients[0].calls) == 1
    assert not clients[0].closed


def test_cleanup_failure_does_not_hide_success_or_log_credentials(factory, caplog):
    _, clients = factory
    embed()
    clients[0].close = Mock(side_effect=RuntimeError("do-not-log-this-secret"))
    assert embed(key="new-key") == [[5.0, 1.0]]
    assert "could not be closed" in caplog.text
    assert "do-not-log-this-secret" not in caplog.text


def test_sdk_missing_error_is_preserved(monkeypatch):
    import sys

    embeddings.close_embedding_client()
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(RuntimeError, match="OpenAI embeddings require openai"):
        embed()
    assert embeddings.embed_texts(["local"], "", "unused")


def test_real_sdk_ingestion_and_retrieval_reuse_one_http_connection(tmp_path, monkeypatch):
    from app.rag.ingest import ingest
    from app.rag.retriever import retrieve

    embeddings.close_embedding_client()
    requests = []
    connections = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            connections.append(self.client_address)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            data = json.dumps({
                "object": "list", "model": payload["model"],
                "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.0]}
                         for i, _ in enumerate(payload["input"])],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    real_constructor = openai.OpenAI
    constructor = Mock(side_effect=lambda **options: real_constructor(
        **options, http_client=httpx.Client(trust_env=False),
    ))
    monkeypatch.setattr(openai, "OpenAI", constructor)
    try:
        source = tmp_path / "source"
        source.mkdir()
        (source / "evidence.txt").write_text("Evidence about planets and orbital motion.", encoding="utf-8")
        store = tmp_path / "store"
        assert ingest(source=source, vector_store_dir=store) >= 1
        first = retrieve("What is orbital motion?", vector_store_dir=store)
        second = retrieve("What is orbital motion?", vector_store_dir=store)
        assert first == second
        assert first.has_sufficient_evidence
        assert first.results[0].source_filename == "evidence.txt"
        assert constructor.call_count == 1
        assert len(requests) == 3
        assert len(connections) == 1
    finally:
        embeddings.close_embedding_client()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def test_small_talk_still_retains_retrieval_memory_identity_and_history(tmp_path, monkeypatch):
    from app.core import sarah_response
    from app.core.agent_bus import LOCAL_AGENT_NAME
    from app.core.config import load_settings
    from app.core.conversation import ConversationTurn
    from app.rag.retriever import RetrievalResponse, RetrievalResult
    from app.tools.web_router import WebRetrievalResult

    settings = replace(load_settings(env_file=tmp_path / ".env"), memory_file=tmp_path / "memory.jsonl")
    retrieval = RetrievalResponse("Hello", True, [
        RetrievalResult("context.txt", "test", "test-id", 1.0, "Unique source evidence remains present.")
    ], "Found evidence.")
    retrieve = Mock(return_value=retrieval)
    monkeypatch.setattr(sarah_response, "retrieve", retrieve)
    memory = Mock()
    memory.retrieve.return_value = []
    monkeypatch.setattr(sarah_response, "MemoryStore", Mock(return_value=memory))
    monkeypatch.setattr(sarah_response, "build_memory_context", lambda _: "Unique long-term memory remains present.")
    monkeypatch.setattr(sarah_response, "build_agent_message_context", lambda _: "")
    monkeypatch.setattr(sarah_response, "retrieve_web_context",
                        lambda *_: WebRetrievalResult(False, False, "not needed", [], ""))
    client = Mock()
    client.create_response.return_value = "Hello Alex."
    history = [ConversationTurn(user="Prior user context.", assistant="Prior assistant context.")]
    response = sarah_response.generate_sarah_response("Hello", settings, client, history=history)
    retrieve.assert_called_once_with("Hello", top_k=5)
    memory.retrieve.assert_called_once_with("Hello", limit=8)
    joined = "\n".join(message["content"] for message in response.messages)
    assert f"{LOCAL_AGENT_NAME.upper()} IMMUTABLE IDENTITY" in joined
    assert "Unique long-term memory remains present." in joined
    assert "Unique source evidence remains present." in joined
    assert "Prior user context." in joined
    assert "Prior assistant context." in joined
    assert response.answer == "Hello Alex."

