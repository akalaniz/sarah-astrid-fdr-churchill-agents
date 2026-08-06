from __future__ import annotations

from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, load_settings
from app.orchestration.agent_orchestrator import (
    build_debate_alex_turn_order,
    parse_debate_alex_command,
    parse_crew_command,
    run_debate_alex,
    run_multi_agent_dialogue,
)
from app.ui.web_app import create_app


class ChurchillCloneIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = load_settings()
        self.memory_file = self.settings.memory_file
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text("", encoding="utf-8")
        self.client = TestClient(create_app(self.settings))

    def test_01_status_reports_churchill_worldline(self) -> None:
        data = self.client.get("/api/status").json()

        self.assertEqual(data["agent_name"], "Churchill v1.0")
        self.assertEqual(data["worldline"], "fdr_churchill_historical")

    def test_02_debug_worldline_reports_port_and_pairing(self) -> None:
        data = self.client.post("/api/chat", json={"message": "/debug_worldline"}).json()["debug"]

        self.assertEqual(data["agent_name"], "Churchill v1.0")
        self.assertEqual(data["worldline"], "fdr_churchill_historical")
        self.assertEqual(data["port"], 8011)
        self.assertEqual(data["paired_agent"], "FDR")
        self.assertFalse(data["spillover_detected"])

    def test_03_memory_path_is_churchill_only(self) -> None:
        data = self.client.post("/api/chat", json={"message": "/debug_worldline"}).json()["debug"]
        memory_path = data["memory_path"].lower().replace("/", "\\")

        self.assertIn(r"churchill-v1-agent\data\memory\churchill_memory.jsonl", memory_path)
        self.assertNotIn("sarah_memory.jsonl", memory_path)
        self.assertNotIn("astrid_memory.jsonl", memory_path)

    def test_04_source_docs_path_is_local_churchill(self) -> None:
        data = self.client.post("/api/chat", json={"message": "/debug_worldline"}).json()["debug"]
        source_path = data["source_docs_path"].lower().replace("/", "\\")

        self.assertIn(r"churchill-v1-agent\data\source_docs", source_path)
        self.assertNotIn("sarah-v2-agent", source_path)
        self.assertNotIn("astrid-v1-agent", source_path)

    def test_05_vector_store_path_is_local_churchill(self) -> None:
        data = self.client.post("/api/chat", json={"message": "/debug_worldline"}).json()["debug"]
        vector_path = data["vector_store_path"].lower().replace("/", "\\")

        self.assertIn(r"churchill-v1-agent\data\vector_store", vector_path)
        self.assertNotIn("sarah-v2-agent", vector_path)
        self.assertNotIn("astrid-v1-agent", vector_path)

    def test_06_prompt_path_is_churchill_master_prompt(self) -> None:
        data = self.client.post("/api/chat", json={"message": "/debug_worldline"}).json()["debug"]
        prompt_path = data["prompt_path"].lower().replace("/", "\\")

        self.assertIn(r"churchill-v1-agent\config\churchill_master_prompt.md", prompt_path)
        self.assertNotIn("sarah_master_prompt.md", prompt_path)
        self.assertNotIn("astrid_master_prompt.md", prompt_path)

    def test_07_runtime_paths_do_not_spill_sarah_or_astrid(self) -> None:
        data = self.client.post("/api/chat", json={"message": "/debug_worldline"}).json()["debug"]
        joined = "\n".join(str(value).lower() for value in data.values())

        self.assertNotIn("sarah-v2-agent", joined)
        self.assertNotIn("astrid-v1-agent", joined)
        self.assertNotIn("shared_agent_bus", joined)
        self.assertNotIn("sarah_memory.jsonl", joined)
        self.assertNotIn("astrid_memory.jsonl", joined)

    def test_08_shared_bus_is_fdr_churchill_bus(self) -> None:
        data = self.client.post("/api/chat", json={"message": "/debug_worldline"}).json()["debug"]
        bus_path = data["shared_bus_path"].lower().replace("/", "\\")

        self.assertIn("shared_fdr_churchill_bus", bus_path)
        self.assertNotIn("shared_agent_bus", bus_path)

    def test_09_remember_writes_to_churchill_memory(self) -> None:
        marker = "test_churchill_memory_marker"
        self.client.post("/api/chat", json={"message": f"/remember {marker}"})

        self.assertIn(marker, self.memory_file.read_text(encoding="utf-8"))

    def test_10_memory_reads_from_churchill_memory(self) -> None:
        marker = "test_churchill_memory_marker_readback"
        self.client.post("/api/chat", json={"message": f"/remember {marker}"})
        data = self.client.post("/api/chat", json={"message": "/memory"}).json()

        self.assertIn(marker, data["text"])

    def test_11_sources_list_uses_local_source_docs_only(self) -> None:
        marker_path = self.settings.source_docs_dir / "churchill_test_source.txt"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("Churchill source marker.", encoding="utf-8")
        try:
            data = self.client.post("/api/chat", json={"message": "/sources_list"}).json()
        finally:
            marker_path.unlink(missing_ok=True)

        self.assertIn("churchill_test_source.txt", data["text"])
        self.assertIn(str(PROJECT_ROOT / "data" / "source_docs"), data["text"])

    def test_12_debug_rag_uses_local_source_and_vector_paths(self) -> None:
        data = self.client.post("/api/chat", json={"message": "/debug_rag"}).json()["debug"]

        self.assertEqual(Path(data["source_docs_path"]), PROJECT_ROOT / "data" / "source_docs")
        self.assertEqual(Path(data["vector_store_path"]), PROJECT_ROOT / "data" / "vector_store")

    def test_13_crew_alternates_churchill_and_fdr(self) -> None:
        calls: list[str] = []

        def fake_transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
            calls.append(agent)
            return f"{agent} replies to {from_agent}."

        result = run_multi_agent_dialogue(
            topic="Discuss wartime coalition strategy.",
            agents=["Churchill", "FDR"],
            rounds=2,
            first_speaker="Churchill",
            transport=fake_transport,
        )
        speakers = [turn["speaker"] for turn in result["turns"][:4]]

        self.assertEqual(speakers, ["Churchill", "FDR", "Churchill", "FDR"])
        self.assertEqual(calls[:4], speakers)

    def test_14_debate_alex_uses_churchill_first_and_fdr_judge(self) -> None:
        command = parse_debate_alex_command("--turns 6 Churchill and FDR assess Alex's argument.")
        calls: list[str] = []

        def fake_transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
            calls.append(agent)
            return f"{agent} response."

        result = run_debate_alex(command, transport=fake_transport)
        speakers = [turn["speaker"] for turn in result["turns"]]

        self.assertEqual(command.first, "Churchill")
        self.assertEqual(command.judge, "FDR")
        self.assertEqual(build_debate_alex_turn_order(turns=6), ["Churchill", "Alex-proxy", "Churchill", "FDR", "Churchill", "FDR"])
        self.assertEqual(speakers, ["Churchill", "Alex-proxy", "Churchill", "FDR", "Churchill", "FDR"])
        self.assertEqual(calls, ["Churchill", "Churchill", "FDR", "Churchill", "FDR"])

    def test_15_crew_parser_keeps_fdr_churchill_defaults(self) -> None:
        command = parse_crew_command("--turns 4 --max-words 50 Churchill and FDR consider Alex's words.")

        self.assertEqual(command.agents, ["Churchill", "FDR"])
        self.assertEqual(command.first_speaker, "Churchill")
        self.assertEqual(command.style.total_turns, 4)
        self.assertEqual(command.style.max_words_per_turn, 50)
        self.assertEqual(command.topic, "Churchill and FDR consider Alex's words.")


if __name__ == "__main__":
    unittest.main()
