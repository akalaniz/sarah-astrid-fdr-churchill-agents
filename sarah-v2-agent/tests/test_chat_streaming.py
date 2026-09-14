from dataclasses import replace
import asyncio
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import load_settings
from app.core.conservative_theorizing import STOP_SENTENCE, TheorizingDecision
from app.core.openai_chat import SarahOpenAIClient
from app.core import sarah_engine
from app.tools.web_router import WebRetrievalResult
from app.ui.chat_stream import StreamCancelled, stream_chat
from app.ui.web_app import create_app


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def __iter__(self):
        yield from self.events


class OpenAIStreamingTests(unittest.TestCase):
    def client(self, events):
        self.stream = FakeStream(events)
        self.requests = []
        def create(**kwargs):
            self.requests.append(kwargs)
            return self.stream
        client = SarahOpenAIClient.__new__(SarahOpenAIClient)
        client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
        return client

    def test_only_visible_text_is_streamed_and_final_output_is_normalized(self):
        client = self.client([
            SimpleNamespace(type="response.reasoning_text.delta", delta="private reasoning"),
            SimpleNamespace(type="response.output_text.delta", delta="Hello "),
            SimpleNamespace(type="response.output_text.delta", delta="world"),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(output_text="Hello world")),
        ])
        chunks = []
        answer = client.stream_response("configured-model", [{"role": "user", "content": "Hi"}], chunks.append)
        self.assertEqual(answer, "Hello world")
        self.assertEqual("".join(chunks), "Hello world")
        self.assertTrue(self.requests[0]["stream"])
        self.assertNotIn("reasoning", self.requests[0])
        self.assertTrue(self.stream.closed)

    def test_refusal_text_is_preserved(self):
        client = self.client([
            SimpleNamespace(type="response.refusal.delta", delta="I cannot help with that."),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(output_text="", output=[])),
        ])
        self.assertEqual(client.stream_response("model", [], lambda _text: None), "I cannot help with that.")

    def test_truncation_and_api_errors_do_not_silently_succeed(self):
        for ending in [None, "response.incomplete", "response.failed", "error"]:
            events = [SimpleNamespace(type="response.output_text.delta", delta="partial")]
            if ending:
                events.append(SimpleNamespace(type=ending))
            client = self.client(events)
            with self.assertRaises(RuntimeError):
                client.stream_response("model", [], lambda _text: None)
            self.assertTrue(self.stream.closed)

    def test_cancellation_closes_sdk_stream(self):
        client = self.client([SimpleNamespace(type="response.output_text.delta", delta="partial")])
        def cancel(_text):
            raise StreamCancelled()
        with self.assertRaises(StreamCancelled):
            client.stream_response("model", [], cancel)
        self.assertTrue(self.stream.closed)

    def test_legacy_completion_stream(self):
        stream = FakeStream([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"), finish_reason=None)]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")]),
        ])
        client = SarahOpenAIClient.__new__(SarahOpenAIClient)
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kw: stream)))
        self.assertEqual(client.stream_response("model", [], lambda _text: None), "Hi")
        self.assertTrue(stream.closed)


class StreamTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_text_arrives_while_generation_is_still_waiting(self):
        finish = threading.Event()
        def generate(on_delta):
            on_delta("First words")
            if not finish.wait(3):
                raise RuntimeError("test deadline")
            return {"text": "First words, complete.", "sources": []}
        response = stream_chat(generate)
        iterator = response.body_iterator
        try:
            self.assertIn("connected", await anext(iterator))
            text = await asyncio.wait_for(anext(iterator), 2)
            self.assertIn("event: text", text)
            self.assertFalse(finish.is_set())
            finish.set()
            done = await asyncio.wait_for(anext(iterator), 2)
            self.assertIn("event: done", done)
        finally:
            finish.set()
            await iterator.aclose()

    async def test_disconnect_stops_producer_at_next_callback(self):
        proceed = threading.Event()
        stopped = threading.Event()
        def generate(on_delta):
            on_delta("First")
            proceed.wait(3)
            try:
                on_delta("Second")
            except StreamCancelled:
                stopped.set()
                raise
            return {"text": "should not commit", "sources": []}
        iterator = stream_chat(generate).body_iterator
        await anext(iterator)
        await asyncio.wait_for(anext(iterator), 2)
        await iterator.aclose()
        proceed.set()
        self.assertTrue(await asyncio.to_thread(stopped.wait, 2))

    async def test_errors_do_not_expose_exception_secrets(self):
        def generate(_on_delta):
            raise ValueError("secret-api-key")
        frames = [frame async for frame in stream_chat(generate).body_iterator]
        self.assertIn("event: error", "".join(frames))
        self.assertNotIn("secret-api-key", "".join(frames))
        self.assertNotIn("event: done", "".join(frames))


class EngineAndWebStreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = replace(load_settings(root / ".env"), openai_api_key="test-key",
                                memory_file=root / "memory.jsonl", web_cache_dir=root / "web_cache")
        self.app = create_app(settings=self.settings)
        self.web = TestClient(self.app)

    def tearDown(self):
        sarah_engine.configure_sarah_engine()
        self.temp.cleanup()

    def test_route_delivers_snapshots_then_final_answer_and_sources(self):
        def generate(message, session_id, on_delta, **kwargs):
            on_delta("First\n")
            on_delta(r"\[E=mc^2\]")
            return SimpleNamespace(text=r"First \[E=mc^2\]", sources=[{"source_filename": "test.docx"}])
        with patch.object(sarah_engine, "generate_sarah_reply", side_effect=generate):
            response = self.web.post("/api/chat/stream", json={"message": "physics"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("event: text", response.text)
        done = json.loads(response.text.split("event: done\ndata: ")[1].strip())
        self.assertEqual(done["sources"], [{"source_filename": "test.docx"}])
        self.assertIn(r"\[E=mc^2\]", done["text"])

    def test_empty_message_and_slash_commands_retain_existing_behavior(self):
        self.assertEqual(self.web.post("/api/chat/stream", json={"message": " "}).status_code, 400)
        with patch("app.ui.web_app._handle_web_command", return_value={"text": "command", "sources": []}) as command:
            response = self.web.post("/api/chat/stream", json={"message": "/help"})
        self.assertEqual(response.json()["text"], "command")
        command.assert_called_once()

    def test_temporary_pdf_keeps_separate_retrieval_query(self):
        with patch.object(sarah_engine, "generate_sarah_reply", return_value=SimpleNamespace(text="OK", sources=[])) as generate:
            self.web.post("/api/chat/stream", json={"message": "Summarize it", "temporary_pdf_name": "test.pdf",
                                                  "temporary_pdf_text": "Temporary test content."})
        self.assertIn("Temporary test content.", generate.call_args.args[0])
        self.assertEqual(generate.call_args.kwargs["retrieval_query"], "Summarize it")
        self.assertEqual(generate.call_args.kwargs["session_id"], "web")

    def engine_context(self, client):
        sarah_engine.configure_sarah_engine(settings=self.settings, client=client)
        self.enterContext(patch("app.core.sarah_response._retrieve_for_message", return_value=(None, "Test: no index")))
        self.enterContext(patch("app.core.sarah_response.build_agent_message_context", return_value=""))
        self.enterContext(patch("app.core.sarah_response.retrieve_web_context",
                                return_value=WebRetrievalResult(False, False, "not needed", [], "")))

    def test_consecutive_streams_keep_only_final_answers_in_history(self):
        histories = []
        class Client:
            def stream_response(self, model, messages, on_delta):
                histories.append([dict(message) for message in messages])
                on_delta("preview")
                return "completed answer"
        self.engine_context(Client())
        for message in ["Hello", "And what did you say?"]:
            sarah_engine.generate_sarah_reply(message, session_id="test", on_delta=lambda _text: None)
        self.assertEqual(sarah_engine.get_sarah_session_turn_count("test"), 2)
        self.assertTrue(any(m["role"] == "assistant" and "completed answer" in m["content"] for m in histories[1]))
        self.assertFalse(any(m["role"] == "assistant" and "preview" in m["content"] for m in histories[1]))

    def test_failed_stream_does_not_create_a_history_turn(self):
        class Client:
            def stream_response(self, model, messages, on_delta):
                on_delta("partial")
                raise RuntimeError("disconnected")
        self.engine_context(Client())
        with self.assertRaises(RuntimeError):
            sarah_engine.generate_sarah_reply("Hello", session_id="test", on_delta=lambda _text: None)
        self.assertEqual(sarah_engine.get_sarah_session_turn_count("test"), 0)

    def test_speculative_theory_is_checked_before_any_preview(self):
        class Client:
            def create_response(self, model, messages):
                return "An unchecked proposal without a verdict."
            def stream_response(self, *_args):
                raise AssertionError("Unchecked theory must not stream.")
        self.engine_context(Client())
        chunks = []
        with patch("app.core.sarah_response.classify_theorizing_with_history", return_value=TheorizingDecision(True)):
            reply = sarah_engine.generate_sarah_reply("Hello", session_id="test", on_delta=chunks.append)
        self.assertIn(STOP_SENTENCE, reply.text)
        self.assertEqual("".join(chunks), "")


if __name__ == "__main__":
    unittest.main()

