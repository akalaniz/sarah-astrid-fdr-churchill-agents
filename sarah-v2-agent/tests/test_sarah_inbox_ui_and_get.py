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
    text = "Astrid, received. I am answering from Sarah's side of the bus."
    sources = []


class SarahInboxUiAndGetTests(unittest.TestCase):
    def test_ui_has_separate_send_and_inbox_sections(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "app" / "ui" / "static" / "index.html").read_text(encoding="utf-8")
        css = (root / "app" / "ui" / "static" / "style.css").read_text(encoding="utf-8")

        self.assertIn('class="panel agent-panel"', html)
        self.assertIn('id="agentSendSection"', html)
        self.assertIn('id="agentInboxPanel"', html)
        self.assertIn("Send a note to Astrid", html)
        self.assertIn("Send to Astrid", html)
        self.assertIn(".agent-panel", css)
        self.assertIn("flex-direction: column", css)
        self.assertIn(".agent-inbox-list", css)
        self.assertIn("overflow: auto", css)

    def test_debug_inbox_ui_reports_panel_and_bus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = _write_bus(
                tmp,
                [
                    {
                        "id": "ui-msg-1",
                        "from_agent": "Astrid",
                        "to_agent": "Sarah",
                        "body": "UI message.",
                        "conversation_id": "ui-thread",
                        "status": "unread",
                        "category": "direct",
                        "metadata": {},
                    }
                ],
            )
            client = TestClient(_test_app(tmp))

            with _patched_bus(bus_file):
                response = client.post("/api/chat", json={"message": "/debug_inbox_ui"})

            debug = response.json()["debug"]

        self.assertTrue(debug["inbox_panel_rendered"])
        self.assertTrue(debug["send_box_present"])
        self.assertTrue(debug["send_button_present"])
        self.assertTrue(debug["message_list_present"])
        self.assertEqual(debug["layout_mode"], "flex-column")
        self.assertEqual(debug["message_list_count"], 1)
        self.assertIn("shared_agent_bus", debug["bus_path"])
        self.assertNotIn("shared_fdr_churchill_bus", debug["bus_path"])
        self.assertEqual(debug["current_agent"], "Sarah")
        self.assertEqual(debug["paired_agent"], "Astrid")
        self.assertIn("style.css", debug["css_files_loaded"])
        self.assertIn("app.js", debug["js_files_loaded"])

    def test_astrid_alias_message_appears_in_sarah_inbox_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = _write_bus(
                tmp,
                [
                    {
                        "message_id": "alias-list-msg",
                        "sender": "Astrid Bach",
                        "recipient": "Sarah v2.0",
                        "content": "Sarah, can you see this from Astrid?",
                        "thread_id": "alias-thread",
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
        self.assertEqual(len(payload["messages"]), 1)
        message = payload["messages"][0]
        self.assertEqual(message["id"], "alias-list-msg")
        self.assertEqual(message["from_agent"], "Astrid")
        self.assertEqual(message["to_agent"], "Sarah")
        self.assertEqual(message["body"], "Sarah, can you see this from Astrid?")

    def test_inbox_get_generates_reply_from_sarah_and_writes_shared_bus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus_file = _write_bus(
                tmp,
                [
                    {
                        "message_id": "get-msg-1",
                        "from": "Astrid v1.0",
                        "to": "Sarah Nelson",
                        "message": "Sarah, reply to this note.",
                        "thread_id": "get-thread",
                        "status": "unread",
                        "category": "direct",
                        "metadata": {},
                    }
                ],
            )
            fdr_churchill_bus = tmp / "shared_fdr_churchill_bus" / "agent_messages.jsonl"
            app = _test_app(tmp)
            client = TestClient(app)

            with _patched_bus(bus_file), patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeSarahReply()) as generate:
                debug_response = client.post("/api/chat", json={"message": "/debug_inbox_get get-msg-1"})

            payload = debug_response.json()
            records = _read_jsonl(bus_file)
            settings = app.state.sarah.settings

        self.assertEqual(debug_response.status_code, 200)
        self.assertTrue(payload["debug"]["found"])
        self.assertTrue(payload["debug"]["is_for_this_agent"])
        self.assertTrue(payload["debug"]["reply_generation_attempted"])
        self.assertTrue(payload["debug"]["reply_generated"])
        self.assertTrue(payload["debug"]["reply_written_to_bus"])
        self.assertEqual(payload["debug"]["reply_recipient"], "Astrid")
        self.assertEqual(records[-1]["from_agent"], "Sarah")
        self.assertEqual(records[-1]["to_agent"], "Astrid")
        self.assertEqual(records[-1]["conversation_id"], "get-thread")
        self.assertFalse(fdr_churchill_bus.exists())
        self.assertIn("sarah_memory.jsonl", str(settings.memory_file))
        self.assertNotIn("fdr_memory.jsonl", str(settings.memory_file))
        self.assertNotIn("churchill_memory.jsonl", str(settings.memory_file))
        self.assertNotIn("fdr-v1-agent", str(settings.source_docs_dir).lower())
        self.assertNotIn("churchill-v1-agent", str(settings.vector_store_dir).lower())
        generate.assert_called_once()


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


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
