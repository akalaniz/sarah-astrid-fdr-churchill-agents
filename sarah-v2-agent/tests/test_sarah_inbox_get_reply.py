from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import load_settings
from app.ui.web_app import AppState, create_app


class FakeSarahReply:
    text = "Astrid, I hear you. Sarah is answering from her own local engine."
    sources = []


class SarahInboxGetReplyTests(unittest.TestCase):
    def test_inbox_get_finds_astrid_message_generates_reply_and_writes_bus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = tmp / "shared_agent_bus" / "agent_messages.jsonl"
            bus_file.parent.mkdir(parents=True)
            bus_file.write_text(
                json.dumps(
                    {
                        "id": "msg-1",
                        "from_agent": "Astrid",
                        "to_agent": "Sarah",
                        "subject": "Dinner note",
                        "body": "Sarah, answer this as yourself.",
                        "conversation_id": "thread-1",
                        "status": "unread",
                        "category": "direct",
                        "metadata": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            app = _test_app(tmp)
            client = TestClient(app)

            with _patched_bus(bus_file), patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeSarahReply()) as generate:
                response = client.post("/api/chat", json={"message": "/inbox_get msg-1"})

            payload = response.json()
            records = _read_jsonl(bus_file)
            original, reply = records

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["command"], "inbox_get")
        self.assertIn("Original message from Astrid:", payload["text"])
        self.assertIn("Sarah: Astrid, I hear you.", payload["text"])
        self.assertEqual(reply["from_agent"], "Sarah")
        self.assertEqual(reply["to_agent"], "Astrid")
        self.assertEqual(reply["conversation_id"], "thread-1")
        self.assertEqual(reply["category"], "reply")
        self.assertEqual(reply["metadata"]["route"], "inbox_get")
        self.assertEqual(reply["metadata"]["reply_to"], "msg-1")
        self.assertEqual(original["status"], "read")
        self.assertTrue(original["metadata"]["replied"])
        self.assertEqual(original["metadata"]["reply_message_id"], reply["id"])
        generate.assert_called_once()
        generated_prompt = generate.call_args.args[0]
        self.assertIn("INTER-AGENT INBOX MESSAGE FOR SARAH", generated_prompt)
        self.assertIn("Respond directly to Astrid as Sarah", generated_prompt)

    def test_inbox_get_recognizes_sarah_aliases(self) -> None:
        for alias in ("Sarah", "Sarah v2.0", "Sarah Nelson"):
            with self.subTest(alias=alias):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    bus_file = tmp / "shared_agent_bus" / "agent_messages.jsonl"
                    bus_file.parent.mkdir(parents=True)
                    bus_file.write_text(
                        json.dumps(
                            {
                                "message_id": f"msg-{alias}",
                                "sender": "Astrid",
                                "recipient": alias,
                                "content": "Alias check.",
                                "thread_id": "thread-alias",
                                "status": "unread",
                                "category": "direct",
                                "metadata": {},
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    app = _test_app(tmp)
                    client = TestClient(app)

                    with _patched_bus(bus_file), patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeSarahReply()):
                        response = client.post("/api/chat", json={"message": f"/debug_inbox_get msg-{alias}"})

                    payload = response.json()
                    records = _read_jsonl(bus_file)

                self.assertEqual(response.status_code, 200)
                self.assertTrue(payload["debug"]["found"])
                self.assertTrue(payload["debug"]["is_for_this_agent"])
                self.assertTrue(payload["debug"]["reply_generation_attempted"])
                self.assertTrue(payload["debug"]["reply_generated"])
                self.assertTrue(payload["debug"]["reply_written_to_bus"])
                self.assertEqual(payload["debug"]["reply_recipient"], "Astrid")
                self.assertEqual(records[-1]["to_agent"], "Astrid")

    def test_inbox_get_rejects_message_for_other_agent_without_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = tmp / "shared_agent_bus" / "agent_messages.jsonl"
            bus_file.parent.mkdir(parents=True)
            bus_file.write_text(
                json.dumps(
                    {
                        "id": "msg-other",
                        "from_agent": "Astrid",
                        "to_agent": "Someone Else",
                        "body": "Not for Sarah.",
                        "conversation_id": "thread-other",
                        "status": "unread",
                        "category": "direct",
                        "metadata": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            app = _test_app(tmp)
            client = TestClient(app)

            with _patched_bus(bus_file), patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeSarahReply()) as generate:
                response = client.post("/api/chat", json={"message": "/inbox_get msg-other"})

            records = _read_jsonl(bus_file)

        self.assertEqual(response.status_code, 200)
        self.assertIn("not Sarah", response.json()["text"])
        self.assertEqual(len(records), 1)
        generate.assert_not_called()

    def test_inbox_get_does_not_touch_fdr_churchill_bus_or_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = tmp / "shared_agent_bus" / "agent_messages.jsonl"
            fdr_churchill_bus = tmp / "shared_fdr_churchill_bus" / "agent_messages.jsonl"
            bus_file.parent.mkdir(parents=True)
            bus_file.write_text(
                json.dumps(
                    {
                        "id": "msg-isolation",
                        "from_agent": "Astrid",
                        "to_agent": "Sarah Nelson",
                        "body": "Isolation check.",
                        "conversation_id": "thread-isolation",
                        "status": "unread",
                        "category": "direct",
                        "metadata": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            app = _test_app(tmp)
            client = TestClient(app)

            with _patched_bus(bus_file), patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeSarahReply()) as generate:
                response = client.post("/api/chat", json={"message": "/inbox_get msg-isolation"})

            settings = app.state.sarah.settings
            generated_prompt = generate.call_args.args[0]

        self.assertEqual(response.status_code, 200)
        self.assertFalse(fdr_churchill_bus.exists())
        self.assertIn("sarah_memory.jsonl", str(settings.memory_file))
        self.assertNotIn("fdr_memory.jsonl", str(settings.memory_file))
        self.assertNotIn("churchill_memory.jsonl", str(settings.memory_file))
        self.assertNotIn("fdr-v1-agent", str(settings.source_docs_dir).lower())
        self.assertNotIn("churchill-v1-agent", str(settings.source_docs_dir).lower())
        self.assertNotIn("fdr-v1-agent", str(settings.vector_store_dir).lower())
        self.assertNotIn("churchill-v1-agent", str(settings.vector_store_dir).lower())
        self.assertIn("Sarah", generated_prompt)
        self.assertNotIn("FDR", generated_prompt)
        self.assertNotIn("Churchill", generated_prompt)


def _test_app(tmp: Path):
    settings = load_settings(env_file=tmp / ".env")
    settings = settings.__class__(
        **{
            **settings.__dict__,
            "memory_file": tmp / "data" / "memory" / "sarah_memory.jsonl",
            "conversations_dir": tmp / "data" / "conversations",
            "web_cache_dir": tmp / "data" / "web_cache",
            "source_docs_dir": tmp / "data" / "source_docs",
            "external_source_docs_dir": tmp / "data" / "source_docs",
            "vector_store_dir": tmp / "data" / "vector_store",
        }
    )
    settings.memory_file.parent.mkdir(parents=True, exist_ok=True)
    settings.memory_file.write_text("", encoding="utf-8")
    settings.source_docs_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    return create_app(settings=settings, state=AppState(settings))


@contextmanager
def _patched_bus(bus_file: Path):
    with patch("app.core.agent_bus.BUS_FILE", bus_file), patch("app.core.inbox_get.BUS_FILE", bus_file):
        yield


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
