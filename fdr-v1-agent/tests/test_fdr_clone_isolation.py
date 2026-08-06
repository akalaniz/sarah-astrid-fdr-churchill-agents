from __future__ import annotations

from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from app.core.config import load_settings
from app.core.worldline import debug_rag, debug_worldline, list_source_doc_names
from app.ui.web_app import create_app


FORBIDDEN = (
    "sarah-v2-agent",
    "astrid-v1-agent",
    "shared_agent_bus",
    "sarah_memory.jsonl",
    "astrid_memory.jsonl",
)


class FDRCloneIsolationTests(unittest.TestCase):
    def _client(self) -> TestClient:
        return TestClient(create_app(load_settings()))

    def _chat(self, message: str) -> dict:
        response = self._client().post("/api/chat", json={"message": message})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_fdr_launches_using_cloned_web_app_structure(self) -> None:
        response = self._client().get("/api/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["agent_name"], "FDR v1.0")
        self.assertEqual(payload["worldline"], "fdr_churchill_historical")

    def test_debug_worldline_reports_fdr_identity_worldline_and_port(self) -> None:
        payload = self._chat("/debug_worldline")
        debug = payload["debug"]
        self.assertEqual(debug["agent_name"], "FDR v1.0")
        self.assertEqual(debug["worldline"], "fdr_churchill_historical")
        self.assertEqual(debug["port"], 8010)

    def test_fdr_memory_path_is_fdr_memory_only(self) -> None:
        path = str(load_settings().memory_file)
        self.assertTrue(path.endswith(r"fdr-v1-agent\data\memory\fdr_memory.jsonl"))
        self.assertNotIn("sarah_memory.jsonl", path.lower())
        self.assertNotIn("astrid_memory.jsonl", path.lower())

    def test_fdr_source_docs_path_is_local_fdr_path_only(self) -> None:
        path = str(load_settings().source_docs_dir)
        self.assertTrue(path.endswith(r"fdr-v1-agent\data\source_docs"))
        self.assertNotIn("sarah-v2-agent", path.lower())
        self.assertNotIn("astrid-v1-agent", path.lower())

    def test_fdr_vector_store_path_is_local_fdr_path_only(self) -> None:
        path = str(load_settings().vector_store_dir)
        self.assertTrue(path.endswith(r"fdr-v1-agent\data\vector_store"))
        self.assertNotIn("sarah-v2-agent", path.lower())
        self.assertNotIn("astrid-v1-agent", path.lower())

    def test_fdr_prompt_path_is_fdr_master_prompt(self) -> None:
        debug = debug_worldline(load_settings())
        self.assertTrue(debug["prompt_path"].endswith(r"fdr-v1-agent\config\fdr_master_prompt.md"))

    def test_no_active_runtime_path_points_to_sarah_or_astrid(self) -> None:
        debug = debug_worldline(load_settings())
        active_values = [
            debug["memory_path"],
            debug["source_docs_path"],
            debug["vector_store_path"],
            debug["prompt_path"],
            debug["shared_bus_path"],
        ]
        lowered = "\n".join(active_values).lower()
        self.assertFalse(any(marker in lowered for marker in FORBIDDEN))
        self.assertFalse(debug["spillover_detected"])

    def test_crew_before_churchill_exists_returns_clean_message(self) -> None:
        payload = self._chat("/crew Test FDR and Churchill.")
        self.assertIn("Churchill is not running at http://127.0.0.1:8011/agent/respond", payload["text"])
        self.assertIn("Build and start Churchill on port 8011, then retry.", payload["text"])

    def test_remember_writes_to_fdr_memory_only(self) -> None:
        settings = load_settings()
        settings.memory_file.parent.mkdir(parents=True, exist_ok=True)
        settings.memory_file.write_text("", encoding="utf-8")
        payload = self._chat("/remember test_fdr_memory_marker")
        self.assertEqual(payload["command"], "remember")
        after = settings.memory_file.read_text(encoding="utf-8")
        self.assertIn("test_fdr_memory_marker", after)
        self.assertNotIn("sarah_memory.jsonl", after.lower())
        self.assertNotIn("astrid_memory.jsonl", after.lower())

    def test_memory_reads_from_fdr_memory_only(self) -> None:
        settings = load_settings()
        settings.memory_file.parent.mkdir(parents=True, exist_ok=True)
        settings.memory_file.write_text("", encoding="utf-8")
        self._chat("/remember test_fdr_memory_read_marker")
        payload = self._chat("/memory")
        self.assertIn("test_fdr_memory_read_marker", payload["text"])
        self.assertNotIn("sarah_memory.jsonl", payload["text"].lower())
        self.assertNotIn("astrid_memory.jsonl", payload["text"].lower())

    def test_sources_list_lists_only_fdr_source_docs(self) -> None:
        docs_dir = load_settings().source_docs_dir
        docs_dir.mkdir(parents=True, exist_ok=True)
        marker = docs_dir / "fdr_test_source.txt"
        marker.write_text("FDR local source marker", encoding="utf-8")
        try:
            payload = self._chat("/sources_list")
            self.assertIn("fdr_test_source.txt", payload["text"])
            self.assertIn(str(docs_dir), payload["text"])
            self.assertNotIn("sarah-v2-agent", payload["text"].lower())
            self.assertNotIn("astrid-v1-agent", payload["text"].lower())
            self.assertEqual(list_source_doc_names(load_settings()), ["fdr_test_source.txt"])
        finally:
            marker.unlink(missing_ok=True)

    def test_debug_rag_shows_only_fdr_rag_paths(self) -> None:
        payload = self._chat("/debug_rag")
        debug = payload["debug"]
        self.assertTrue(debug["source_docs_path"].endswith(r"fdr-v1-agent\data\source_docs"))
        self.assertTrue(debug["vector_store_path"].endswith(r"fdr-v1-agent\data\vector_store"))
        text = "\n".join(str(value) for value in debug.values()).lower()
        self.assertNotIn("sarah-v2-agent", text)
        self.assertNotIn("astrid-v1-agent", text)
        self.assertTrue(debug_rag(load_settings())["source_docs_exists"])


if __name__ == "__main__":
    unittest.main()
