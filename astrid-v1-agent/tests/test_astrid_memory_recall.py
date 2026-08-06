from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import load_settings
from app.core import sarah_engine
from app.core.sarah_engine import configure_sarah_engine, generate_sarah_reply
from app.tools.web_router import WebRetrievalResult
from app.ui.web_app import AppState, create_app


class CapturingClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def create_response(self, model: str, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return "Astrid memory-aware response."


class AstridMemoryRecallTests(unittest.TestCase):
    def setUp(self) -> None:
        self._retrieve_patcher = patch(
            "app.core.sarah_response._retrieve_for_message",
            return_value=(None, "test retrieval disabled"),
        )
        self._web_patcher = patch(
            "app.core.sarah_response.retrieve_web_context",
            return_value=WebRetrievalResult(
                used_web=False,
                failed=False,
                reason="test disabled",
                sources=[],
                timestamp="2026-06-04T00:00:00+00:00",
            ),
        )
        self._retrieve_patcher.start()
        self._web_patcher.start()

    def tearDown(self) -> None:
        self._retrieve_patcher.stop()
        self._web_patcher.stop()
        configure_sarah_engine()

    def test_remember_saves_to_astrid_memory_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(Path(tmp))
            configure_sarah_engine(settings=settings, client=CapturingClient())

            record = sarah_engine.remember_astrid_memory(
                "Alex prefers Astrid to be warm, embodied, direct, and terse."
            )

            self.assertEqual(settings.memory_file.name, "astrid_memory.jsonl")
            text = settings.memory_file.read_text(encoding="utf-8")
            self.assertIn(record.memory_id, text)
            self.assertIn("warm, embodied, direct", text)

    def test_memory_keyword_displays_matching_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(Path(tmp))
            configure_sarah_engine(settings=settings, client=CapturingClient())
            sarah_engine.remember_astrid_memory("Alex prefers Astrid to be warm, embodied, direct, and terse.")
            sarah_engine.remember_astrid_memory("Alex also likes Romance languages.")

            output = sarah_engine.format_astrid_memories("embodied")

            self.assertIn("warm, embodied", output)
            self.assertNotIn("Romance languages", output)

    def test_recall_keyword_injects_one_shot_live_context_into_next_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CapturingClient()
            settings = _settings(Path(tmp))
            configure_sarah_engine(settings=settings, client=client)
            sarah_engine.remember_astrid_memory("Alex prefers Astrid to be warm, embodied, direct, and terse.")

            recall_text = sarah_engine.recall_astrid_memories("embodied", session_id="test")
            before = sarah_engine.debug_astrid_memory_live("test")
            generate_sarah_reply("Use that remembered preference now.", session_id="test")
            after = sarah_engine.debug_astrid_memory_live("test")

            prompt = _joined_prompt(client.calls[-1])
            self.assertIn("Recalled Astrid memories for 'embodied':", recall_text)
            self.assertTrue(before["live_context_pending"])
            self.assertIn("Relevant remembered context from Astrid memory:", prompt)
            self.assertIn("Alex prefers Astrid to be warm, embodied, direct, and terse.", prompt)
            self.assertFalse(after["live_context_pending"])
            self.assertTrue(after["injected_into_last_prompt"])
            self.assertEqual(after["last_injected_memory_count"], 1)

    def test_use_memory_on_auto_injects_relevant_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CapturingClient()
            settings = _settings(Path(tmp))
            configure_sarah_engine(settings=settings, client=client)
            sarah_engine.remember_astrid_memory("Alex prefers Astrid to be warm, embodied, direct, and terse.")
            sarah_engine.set_astrid_use_memory(True, session_id="test")

            generate_sarah_reply("Please keep Astrid warm and embodied.", session_id="test")

            prompt = _joined_prompt(client.calls[-1])
            self.assertIn("Relevant remembered context from Astrid memory:", prompt)
            self.assertIn("warm, embodied, direct", prompt)

    def test_use_memory_off_suppresses_auto_memory_until_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CapturingClient()
            settings = _settings(Path(tmp))
            configure_sarah_engine(settings=settings, client=client)
            sarah_engine.remember_astrid_memory("Alex prefers Astrid to be warm, embodied, direct, and terse.")
            sarah_engine.set_astrid_use_memory(False, session_id="test")

            generate_sarah_reply("Please keep Astrid warm and embodied.", session_id="test")
            off_prompt = _joined_prompt(client.calls[-1])
            sarah_engine.recall_astrid_memories("embodied", session_id="test")
            generate_sarah_reply("Now use the recalled preference.", session_id="test")
            recall_prompt = _joined_prompt(client.calls[-1])

            self.assertNotIn("Relevant remembered context from Astrid memory:", off_prompt)
            self.assertIn("Relevant remembered context from Astrid memory:", recall_prompt)
            self.assertIn("warm, embodied, direct", recall_prompt)

    def test_crew_uses_astrid_memory_for_astrid_endpoint_turns_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CapturingClient()
            settings = _settings(Path(tmp))
            configure_sarah_engine(settings=settings, client=client)
            sarah_engine.remember_astrid_memory("Astrid crew preference: dinner style, embodied, no report format.")

            generate_sarah_reply(
                "INTER-AGENT MESSAGE\nfrom_agent: Sarah\nmessage:\nPlease answer the crew topic in dinner style.",
                session_id="inter_agent:crew-test",
            )

            prompt = _joined_prompt(client.calls[-1])
            self.assertIn("Relevant remembered context from Astrid memory:", prompt)
            self.assertIn("Astrid crew preference", prompt)
            self.assertNotIn("Sarah memory:", prompt)

    def test_alex_continuation_uses_memory_without_echoing_alex_input_in_debug_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CapturingClient()
            settings = _settings(Path(tmp))
            configure_sarah_engine(settings=settings, client=client)
            sarah_engine.remember_astrid_memory("Alex prefers Astrid to keep dinner conversation short.")

            generate_sarah_reply("Message from Alex to the crew: don't overthink this dinner exchange.", session_id="test")
            debug = sarah_engine.format_debug_astrid_memory_live("test")

            self.assertIn("last_injected_memory_count: 1", debug)
            self.assertNotIn("Message from Alex to the crew", debug)

    def test_debate_alex_uses_memory_without_confusing_it_with_alex_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CapturingClient()
            settings = _settings(Path(tmp))
            configure_sarah_engine(settings=settings, client=client)
            sarah_engine.remember_astrid_memory("Astrid debate preference: do not label memory as Alex-proxy.")

            generate_sarah_reply("Alex-proxy argues LLM timelines matter. Astrid, respond.", session_id="debate")

            prompt = _joined_prompt(client.calls[-1])
            self.assertIn("Relevant remembered context from Astrid memory:", prompt)
            self.assertIn("do not label memory as Alex-proxy", prompt)
            memory_block = _memory_block(prompt)
            self.assertNotIn("Observed Alex-proxy position:", memory_block)

    def test_forbidden_path_check_detects_non_astrid_worldline_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(
                Path(tmp),
                memory_file=Path(tmp) / "sarah-v2-agent" / "data" / "memory" / "sarah_memory.jsonl",
            )
            configure_sarah_engine(settings=settings, client=CapturingClient())

            debug = sarah_engine.debug_astrid_memory_live("test")

            self.assertTrue(debug["forbidden_memory_paths_detected"])

    def test_web_commands_support_recall_use_memory_and_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            app = create_app(settings=settings, state=AppState(settings))
            client = TestClient(app)

            remember_response = client.post(
                "/api/chat",
                json={"message": "/remember Alex prefers Astrid to be warm, embodied, direct, and terse."},
            )
            memory_response = client.post("/api/chat", json={"message": "/memory embodied"})
            recall_response = client.post("/api/chat", json={"message": "/recall embodied"})
            use_off_response = client.post("/api/chat", json={"message": "/use_memory off"})
            clear_response = client.post("/api/chat", json={"message": "/clear_recall"})
            debug_response = client.post("/api/chat", json={"message": "/debug_memory_live"})

        self.assertEqual(remember_response.status_code, 200)
        self.assertIn("I'll remember that", remember_response.json()["text"])
        self.assertIn("warm, embodied", memory_response.json()["text"])
        self.assertIn("Recalled Astrid memories for 'embodied'", recall_response.json()["text"])
        self.assertEqual(use_off_response.json()["text"], "use_memory: off")
        self.assertIn("cleared", clear_response.json()["text"].lower())
        self.assertIn("agent_name: Astrid v1.0", debug_response.json()["text"])
        self.assertIn("astrid_memory.jsonl", debug_response.json()["text"])
        self.assertIn("memory_file_exists: True", debug_response.json()["text"])

    def test_memory_debate_alex_displays_matching_memories_without_injection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CapturingClient()
            settings = _settings(Path(tmp))
            configure_sarah_engine(settings=settings, client=client)
            sarah_engine.remember_astrid_memory(
                "Debate Alex memory: debate alex uses Alex-proxy, judge, crew turn order, max words, and dinner style."
            )
            sarah_engine.remember_astrid_memory("Astrid unrelated memory: Romance languages.")

            output = sarah_engine.format_astrid_memories("debate_alex")
            before = sarah_engine.debug_astrid_memory_live("web")
            generate_sarah_reply("Quick neutral status check.", session_id="web")
            prompt = _joined_prompt(client.calls[-1])
            after = sarah_engine.debug_astrid_memory_live("web")

            self.assertIn("Alex-proxy", output)
            self.assertIn("turn order", output)
            self.assertNotIn("Romance languages", output)
            self.assertFalse(before["live_context_pending"])
            self.assertNotIn("Debate Alex memory", prompt)
            self.assertFalse(after["injected_into_last_prompt"])

    def test_recall_debate_alex_injects_into_next_inter_agent_debate_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CapturingClient()
            settings = _settings(Path(tmp))
            configure_sarah_engine(settings=settings, client=client)
            sarah_engine.remember_astrid_memory(
                "Debate Alex memory: Astrid should preserve Alex-proxy, judge, first speaker, and turn order."
            )

            recall = sarah_engine.recall_astrid_memories("debate_alex", session_id="web")
            generate_sarah_reply(
                "DEBATE_ALEX ORCHESTRATION\nAlex-proxy has spoken. Judge the first speaker and turn order.",
                session_id="inter_agent:debate-1",
            )
            prompt = _joined_prompt(client.calls[-1])
            debug = sarah_engine.debug_astrid_memory_live("web")

            self.assertIn("Recalled Astrid memories for 'debate_alex'", recall)
            self.assertIn("Relevant remembered context from Astrid memory:", prompt)
            self.assertIn("preserve Alex-proxy, judge, first speaker, and turn order", prompt)
            self.assertTrue(debug["injected_into_last_prompt"])
            self.assertFalse(debug["live_context_pending"])

    def test_debate_alex_auto_memory_retrieves_relevant_debate_terms_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CapturingClient()
            settings = _settings(Path(tmp))
            configure_sarah_engine(settings=settings, client=client)
            sarah_engine.remember_astrid_memory(
                "Astrid debate_alex preference: max words, dinner style, Alex proxy, judge, and turn order matter."
            )
            sarah_engine.set_astrid_use_memory(True, session_id="inter_agent:debate-auto")

            generate_sarah_reply(
                "DEBATE_ALEX ORCHESTRATION\nPrevious speaker: Alex-proxy\nMaximum: 50 words.\nJudge the turn order.",
                session_id="inter_agent:debate-auto",
            )

            prompt = _joined_prompt(client.calls[-1])
            self.assertIn("Relevant remembered context from Astrid memory:", prompt)
            self.assertIn("Astrid debate_alex preference", prompt)

    def test_astrid_memory_search_does_not_load_sarah_or_fdr_memory_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            configure_sarah_engine(settings=settings, client=CapturingClient())
            sarah_memory = root / "data" / "memory" / "sarah_memory.jsonl"
            fdr_memory = root / "data" / "memory" / "fdr_memory.jsonl"
            sarah_memory.parent.mkdir(parents=True, exist_ok=True)
            sarah_memory.write_text('{"text":"Sarah-only debate_alex memory should not load."}\n', encoding="utf-8")
            fdr_memory.write_text('{"text":"FDR-only debate_alex memory should not load."}\n', encoding="utf-8")
            sarah_engine.remember_astrid_memory("Astrid-only debate_alex memory should load.")

            output = sarah_engine.format_astrid_memories("debate_alex")

            self.assertIn("Astrid-only debate_alex memory", output)
            self.assertNotIn("Sarah-only", output)
            self.assertNotIn("FDR-only", output)


def _settings(tmp: Path, memory_file: Path | None = None):
    settings = load_settings(env_file=tmp / ".env")
    return settings.__class__(
        **{
            **settings.__dict__,
            "openai_api_key": "test-key",
            "memory_file": memory_file or tmp / "data" / "memory" / "astrid_memory.jsonl",
            "web_cache_dir": tmp / "web_cache",
            "vector_store_dir": tmp / "vector_store",
            "conversations_dir": tmp / "conversations",
        }
    )


def _joined_prompt(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(message["content"] for message in messages)


def _memory_block(prompt: str) -> str:
    marker = "Relevant remembered context from Astrid memory:"
    start = prompt.find(marker)
    if start == -1:
        return ""
    end = prompt.find("\n\nRETRIEVED CANON/SOURCE CONTEXT:", start)
    return prompt[start:] if end == -1 else prompt[start:end]


if __name__ == "__main__":
    unittest.main()
