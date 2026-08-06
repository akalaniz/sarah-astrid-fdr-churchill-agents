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
    text = "Astrid, Sarah received the message."
    sources = []


class SarahInboxReceiveTests(unittest.TestCase):
    def test_astrid_to_sarah_message_appears_in_inbox_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = _write_bus(
                tmp,
                [
                    {
                        "id": "direct-1",
                        "from_agent": "Astrid",
                        "to_agent": "Sarah",
                        "body": "Sarah, can you see me?",
                        "conversation_id": "thread-1",
                        "status": "unread",
                        "category": "direct",
                        "metadata": {},
                    }
                ],
            )
            client = TestClient(_test_app(tmp))

            with _patched_bus(bus_file):
                response = client.get("/agent/inbox")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual([message["id"] for message in payload["messages"]], ["direct-1"])

    def test_sarah_recipient_aliases_are_listed(self) -> None:
        aliases = ("Sarah", "Sarah v2.0", "Sarah Nelson")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = _write_bus(
                tmp,
                [
                    {
                        "id": f"recipient-{index}",
                        "from_agent": "Astrid",
                        "to_agent": alias,
                        "body": f"Message to {alias}",
                        "conversation_id": f"thread-{index}",
                        "status": "unread",
                        "category": "direct",
                        "metadata": {},
                    }
                    for index, alias in enumerate(aliases)
                ],
            )
            client = TestClient(_test_app(tmp))

            with _patched_bus(bus_file):
                response = client.get("/agent/inbox")

        messages = response.json()["messages"]
        self.assertEqual(len(messages), 3)
        self.assertEqual({message["to_agent"] for message in messages}, {"Sarah"})

    def test_astrid_sender_aliases_are_normalized(self) -> None:
        aliases = ("Astrid", "Astrid v1.0", "Astrid Bach")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = _write_bus(
                tmp,
                [
                    {
                        "id": f"sender-{index}",
                        "from_agent": alias,
                        "to_agent": "Sarah",
                        "body": f"Message from {alias}",
                        "conversation_id": f"thread-{index}",
                        "status": "unread",
                        "category": "direct",
                        "metadata": {},
                    }
                    for index, alias in enumerate(aliases)
                ],
            )
            client = TestClient(_test_app(tmp))

            with _patched_bus(bus_file):
                response = client.get("/agent/inbox")

        messages = response.json()["messages"]
        self.assertEqual(len(messages), 3)
        self.assertEqual({message["from_agent"] for message in messages}, {"Astrid"})

    def test_sarah_reads_shared_agent_bus_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = _write_bus(
                tmp,
                [
                    {
                        "id": "shared-bus-message",
                        "from_agent": "Astrid",
                        "to_agent": "Sarah",
                        "body": "Correct bus.",
                        "conversation_id": "thread-shared",
                        "status": "unread",
                        "category": "direct",
                        "metadata": {},
                    }
                ],
            )
            fdr_bus = tmp / "shared_fdr_churchill_bus" / "agent_messages.jsonl"
            fdr_bus.parent.mkdir(parents=True)
            fdr_bus.write_text(
                json.dumps(
                    {
                        "id": "wrong-bus-message",
                        "from_agent": "FDR",
                        "to_agent": "Sarah",
                        "body": "Wrong bus.",
                        "status": "unread",
                        "category": "direct",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            client = TestClient(_test_app(tmp))

            with _patched_bus(bus_file):
                response = client.get("/agent/inbox")
                debug_response = client.post("/api/chat", json={"message": "/debug_inbox_receive"})

        ids = [message["id"] for message in response.json()["messages"]]
        self.assertEqual(ids, ["shared-bus-message"])
        self.assertIn("shared_agent_bus", debug_response.json()["debug"]["bus_path"])
        self.assertNotIn("shared_fdr_churchill_bus", debug_response.json()["debug"]["bus_path"])

    def test_schema_variants_sender_recipient_target_and_content_are_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = _write_bus(
                tmp,
                [
                    {
                        "message_id": "variant-sender-recipient",
                        "sender": "Astrid Bach",
                        "recipient": "Sarah Nelson",
                        "content": "Content field.",
                        "thread_id": "thread-a",
                        "status": "unread",
                        "category": "direct",
                    },
                    {
                        "message_id": "variant-from-to",
                        "from": "Astrid v1.0",
                        "to": "Sarah v2.0",
                        "message": "Message field.",
                        "thread_id": "thread-b",
                        "status": "unread",
                        "category": "reply",
                    },
                    {
                        "message_id": "variant-target",
                        "sender": "Astrid",
                        "target": "Sarah",
                        "body": "Target field.",
                        "thread_id": "thread-c",
                        "status": "unread",
                        "category": "direct",
                    },
                ],
            )
            client = TestClient(_test_app(tmp))

            with _patched_bus(bus_file):
                response = client.get("/agent/inbox")

        messages = response.json()["messages"]
        self.assertEqual({message["id"] for message in messages}, {"variant-sender-recipient", "variant-from-to", "variant-target"})
        self.assertEqual({message["from_agent"] for message in messages}, {"Astrid"})
        self.assertEqual({message["to_agent"] for message in messages}, {"Sarah"})
        self.assertEqual({message["body"] for message in messages}, {"Content field.", "Message field.", "Target field."})

    def test_debug_inbox_receive_reports_sarah_addressed_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = _write_bus(
                tmp,
                [
                    {
                        "id": "debug-msg",
                        "sender": "Astrid Bach",
                        "target": "Sarah Nelson",
                        "content": "Debug receive.",
                        "thread_id": "thread-debug",
                        "status": "unread",
                        "category": "direct",
                    }
                ],
            )
            client = TestClient(_test_app(tmp))

            with _patched_bus(bus_file):
                response = client.post("/api/chat", json={"message": "/debug_inbox_receive"})

        debug = response.json()["debug"]
        self.assertGreater(debug["messages_addressed_to_sarah"], 0)
        self.assertGreater(debug["unread_messages_addressed_to_sarah"], 0)
        self.assertEqual(debug["latest_message_id_for_sarah"], "debug-msg")
        self.assertEqual(debug["latest_sender_normalized"], "Astrid")
        self.assertEqual(debug["latest_recipient_normalized"], "Sarah")

    def test_debug_inbox_get_can_retrieve_listed_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = _write_bus(
                tmp,
                [
                    {
                        "message_id": "listed-get-msg",
                        "from": "Astrid Bach",
                        "to": "Sarah Nelson",
                        "message": "Can inbox_get see this?",
                        "thread_id": "thread-get",
                        "status": "unread",
                        "category": "direct",
                    }
                ],
            )
            client = TestClient(_test_app(tmp))

            with _patched_bus(bus_file), patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeSarahReply()):
                list_response = client.get("/agent/inbox")
                get_response = client.post("/api/chat", json={"message": "/debug_inbox_get listed-get-msg"})

        self.assertEqual([message["id"] for message in list_response.json()["messages"]], ["listed-get-msg"])
        self.assertTrue(get_response.json()["debug"]["found"])
        self.assertTrue(get_response.json()["debug"]["is_for_this_agent"])


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


def _write_bus(tmp: Path, records: list[dict]) -> Path:
    bus_file = tmp / "shared_agent_bus" / "agent_messages.jsonl"
    bus_file.parent.mkdir(parents=True)
    bus_file.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return bus_file


@contextmanager
def _patched_bus(bus_file: Path):
    with (
        patch("app.core.agent_bus.BUS_FILE", bus_file),
        patch("app.core.inbox_get.BUS_FILE", bus_file),
        patch("app.ui.web_app.BUS_FILE", bus_file),
    ):
        yield


if __name__ == "__main__":
    unittest.main()
