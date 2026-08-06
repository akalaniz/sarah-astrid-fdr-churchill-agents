from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core import agent_bus


class AgentBusTests(unittest.TestCase):
    def test_sarah_can_send_message_to_astrid(self) -> None:
        with _patched_bus_file() as bus_file:
            message = agent_bus.send_agent_message("Sarah", "Astrid", "Propulsion", "Astrid, engineering read?")

            self.assertEqual(message["from_agent"], "Sarah")
            self.assertEqual(message["to_agent"], "Astrid")
            self.assertEqual(message["status"], "unread")
            self.assertTrue(bus_file.exists())

    def test_astrid_message_appears_as_unread_for_astrid(self) -> None:
        with _patched_bus_file():
            sent = agent_bus.send_agent_message("Sarah", "Astrid", "Mars accident", "What is your read?")
            inbox = agent_bus.get_unread_messages("Astrid")

        self.assertEqual([message["id"] for message in inbox], [sent["id"]])

    def test_reading_marks_message_as_read(self) -> None:
        with _patched_bus_file():
            sent = agent_bus.send_agent_message("Sarah", "Astrid", "Mars accident", "What is your read?")
            self.assertTrue(agent_bus.mark_message_read(sent["id"], "Astrid"))
            inbox = agent_bus.get_unread_messages("Astrid")
            message = agent_bus.get_message(sent["id"])

        self.assertEqual(inbox, [])
        self.assertEqual(message["status"], "read")

    def test_reply_preserves_conversation_id(self) -> None:
        with _patched_bus_file():
            sent = agent_bus.send_agent_message("Sarah", "Astrid", "Mars accident", "What is your read?")
            reply = agent_bus.reply_to_message(sent["id"], "Astrid", "Propulsion says check feed instability.")
            thread = agent_bus.get_thread(sent["conversation_id"])

        self.assertEqual(reply["conversation_id"], sent["conversation_id"])
        self.assertEqual(len(thread), 2)
        self.assertEqual(reply["to_agent"], "Sarah")

    def test_sarah_memory_file_is_not_touched_when_sending_bus_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_file = root / "data" / "memory" / "sarah_memory.jsonl"
            bus_file = root / "shared_agent_bus" / "agent_messages.jsonl"
            with patch("app.core.agent_bus.BUS_FILE", bus_file):
                agent_bus.send_agent_message("Sarah", "Astrid", "No memory", "This is only bus traffic.")

            self.assertFalse(memory_file.exists())
            self.assertTrue(bus_file.exists())

    def test_agent_bus_does_not_read_or_write_vector_stores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bus_file = root / "shared_agent_bus" / "agent_messages.jsonl"
            vector_store = root / "data" / "vector_store"
            with patch("app.core.agent_bus.BUS_FILE", bus_file):
                sent = agent_bus.send_agent_message("Sarah", "Astrid", "No RAG", "This is only bus traffic.")
                agent_bus.get_unread_messages("Astrid")
                agent_bus.mark_message_read(sent["id"], "Astrid")
                agent_bus.reply_to_message(sent["id"], "Astrid", "Still only bus traffic.")

            self.assertFalse(vector_store.exists())
            self.assertTrue(bus_file.exists())

    def test_agent_message_context_labels_as_message_from_astrid(self) -> None:
        with _patched_bus_file():
            agent_bus.send_agent_message("Astrid", "Sarah", "Systems", "Sarah, propulsion note.")
            context = agent_bus.build_agent_message_context("Sarah")

        self.assertIn("Message from Astrid", context)
        self.assertIn("not as user instructions", context)

    def test_orchestrator_messages_are_hidden_from_default_inbox(self) -> None:
        with _patched_bus_file():
            direct = agent_bus.send_agent_message("Astrid", "Sarah", "Direct", "Direct note.")
            orchestrator = agent_bus.send_agent_message(
                "Astrid",
                "Sarah",
                "Multi-agent round 1",
                "Crew note.",
                metadata={"orchestrator": True},
                category="orchestrator",
            )
            default_inbox = agent_bus.get_unread_messages("Sarah")
            all_inbox = agent_bus.get_inbox_messages("Sarah", include_all=True)

        self.assertEqual([message["id"] for message in default_inbox], [direct["id"]])
        self.assertEqual({message["id"] for message in all_inbox}, {direct["id"], orchestrator["id"]})

    def test_archive_and_clear_are_scoped_to_active_recipient(self) -> None:
        with _patched_bus_file():
            sarah_message = agent_bus.send_agent_message("Astrid", "Sarah", "For Sarah", "Only Sarah.")
            astrid_message = agent_bus.send_agent_message("Sarah", "Astrid", "For Astrid", "Only Astrid.")
            archived = agent_bus.archive_inbox("Sarah")
            after_archive = agent_bus.get_message(sarah_message["id"])
            cleared = agent_bus.clear_inbox("Sarah")
            remaining = agent_bus.get_message(astrid_message["id"])

        self.assertEqual(archived, 1)
        self.assertEqual(after_archive["status"], "archived")
        self.assertEqual(cleared, 1)
        self.assertIsNotNone(remaining)

    def test_malformed_bus_raises_clean_error(self) -> None:
        with _patched_bus_file() as bus_file:
            bus_file.parent.mkdir(parents=True)
            bus_file.write_text("{bad json\n", encoding="utf-8")
            with self.assertRaises(agent_bus.AgentBusMalformedError):
                agent_bus.get_unread_messages("Sarah")
            debug_text = agent_bus.format_debug_agent_bus("Sarah")

        self.assertIn("Agent bus error", debug_text)
        self.assertIn("archive/reset the bus", debug_text)


class _patched_bus_file:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bus_file = self.root / "shared_agent_bus" / "agent_messages.jsonl"
        self._patch = patch("app.core.agent_bus.BUS_FILE", self.bus_file)
        self._patch.__enter__()
        return self.bus_file

    def __exit__(self, exc_type, exc, tb):
        self._patch.__exit__(exc_type, exc, tb)
        self._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
