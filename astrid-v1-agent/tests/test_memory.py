from pathlib import Path
import tempfile
import unittest

from app.core.memory import (
    MemoryStore,
    MemoryType,
    build_memory_context,
    format_memories,
    infer_memory_type,
)


class MemoryTests(unittest.TestCase):
    def test_remember_lists_and_retrieves_durable_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.jsonl")
            record = store.remember("Alex prefers concise CAS briefings with DIME.")

            listed = store.list_memories()
            retrieved = store.retrieve("Give me a CAS DIME briefing.")

        self.assertEqual(record.memory_type, MemoryType.PREFERENCES.value)
        self.assertEqual(len(listed), 1)
        self.assertEqual(retrieved[0].text, "Alex prefers concise CAS briefings with DIME.")

    def test_forget_deactivates_matching_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.jsonl")
            store.remember("Alex likes the Nuts joke.")
            store.remember("Sarah canon includes Darwin.")

            forgotten = store.forget("Nuts")
            active = store.list_memories()
            all_records = store.list_memories(include_inactive=True)

        self.assertEqual(len(forgotten), 1)
        self.assertEqual(len(active), 1)
        self.assertEqual(len(all_records), 2)
        self.assertFalse(forgotten[0].active)

    def test_memory_type_inference(self) -> None:
        self.assertEqual(infer_memory_type("Alex prefers terse answers."), MemoryType.PREFERENCES)
        self.assertEqual(infer_memory_type("Sarah canon says Darwin is unfinished."), MemoryType.CANON_FACTS)
        self.assertEqual(infer_memory_type("Unresolved question: what is Sarah's remainder?"), MemoryType.UNRESOLVED_QUESTIONS)

    def test_memory_context_is_concise_and_non_surveillance_framed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.jsonl")
            store.remember("Alex and Sarah reuse the Nuts joke.", MemoryType.RELATIONSHIP_CONTINUITY)
            context = build_memory_context(store.retrieve("Nuts joke"))
            listing = format_memories(store.list_memories())

        self.assertIn("durable continuity hints", context)
        self.assertIn("not imply surveillance", context)
        self.assertIn("relationship_continuity", listing)

    def test_agent_profile_memory_can_be_kept_separate_from_user_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_store = MemoryStore(Path(tmp) / "user_profile.jsonl")
            agent_store = MemoryStore(Path(tmp) / "agent_profile.jsonl")

            user_store.remember("Alex prefers route-parity checks.", MemoryType.USER_PROFILE)
            agent_record = agent_store.remember(
                "Sarah prefers not to answer adult intimacy as a menu.",
                MemoryType.AGENT_PROFILE,
            )

            user_listing = format_memories(user_store.list_memories())
            agent_listing = format_memories(agent_store.list_memories())

        self.assertEqual(agent_record.memory_type, MemoryType.AGENT_PROFILE.value)
        self.assertIn("user_profile", user_listing)
        self.assertIn("agent_profile", agent_listing)
        self.assertNotIn("adult intimacy as a menu", user_listing)


if __name__ == "__main__":
    unittest.main()
