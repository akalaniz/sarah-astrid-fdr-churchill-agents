from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.orchestration import agent_orchestrator as orchestrator
from app.ui.web_app import create_app


def _fake_transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
    return f"{agent} answers {from_agent} in one short dinner-conversation paragraph."


class CrewTranscriptPersistenceTests(unittest.TestCase):
    def _patched_paths(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        transcript_dir = root / "transcripts"
        state_file = root / "crew_state.json"
        patches = [
            patch.object(orchestrator, "TRANSCRIPT_DIR", transcript_dir),
            patch.object(orchestrator, "CREW_STATE_FILE", state_file),
            patch("app.ui.web_app.TRANSCRIPT_DIR", transcript_dir),
        ]
        return temp, transcript_dir, state_file, patches

    def _client(self):
        return TestClient(create_app())

    def _start_terse_crew(self, client: TestClient) -> dict:
        response = client.post(
            "/api/chat",
            json={"message": "/crew_terse Sarah and Astrid discuss Alex's LLM timeline."},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["command"], "crew_terse")
        return payload["result"]

    def test_crew_terse_writes_conversation_and_latest_files_and_state(self):
        temp, transcript_dir, state_file, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _fake_transport):
            result = self._start_terse_crew(self._client())
            conversation_id = result["conversation_id"]
            md_path = transcript_dir / f"{conversation_id}.md"
            json_path = transcript_dir / f"{conversation_id}.json"
            latest_md = transcript_dir / "latest_multi_agent_transcript.md"
            latest_json = transcript_dir / "latest_multi_agent_transcript.json"
            state = orchestrator.load_crew_state()

            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())
            self.assertTrue(latest_md.exists())
            self.assertTrue(latest_json.exists())
            self.assertEqual(Path(state["transcript_md_path"]), md_path)
            self.assertEqual(Path(state["transcript_json_path"]), json_path)
            self.assertEqual(state["active_conversation_id"], conversation_id)
            self.assertTrue(state_file.exists())

    def test_alex_continuation_updates_json_and_markdown(self):
        temp, transcript_dir, _state_file, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _fake_transport):
            client = self._client()
            started = self._start_terse_crew(client)
            conversation_id = started["conversation_id"]
            response = client.post("/api/chat", json={"message": "/Alex Can both of you continue?"})
            payload = response.json()
            json_path = transcript_dir / f"{conversation_id}.json"
            md_path = transcript_dir / f"{conversation_id}.md"
            saved = orchestrator.load_crew_payload(conversation_id)

            self.assertEqual(response.status_code, 200)
            self.assertTrue(payload["ok"])
            self.assertNotIn("Crew transcript JSON not found", payload["text"])
            self.assertEqual([turn["speaker"] for turn in payload["result"]["turns"][-3:]], ["Alex", "Sarah", "Astrid"])
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertEqual([turn["speaker"] for turn in saved["turns"][-3:]], ["Alex", "Sarah", "Astrid"])

    def test_alex_reconstructs_missing_conversation_json_from_markdown(self):
        temp, transcript_dir, _state_file, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _fake_transport):
            client = self._client()
            started = self._start_terse_crew(client)
            conversation_id = started["conversation_id"]
            json_path = transcript_dir / f"{conversation_id}.json"
            self.assertTrue(json_path.exists())
            json_path.unlink()

            response = client.post("/api/chat", json={"message": "/Alex Repair this and continue."})
            payload = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertTrue(payload["ok"])
            self.assertTrue(json_path.exists())
            self.assertEqual([turn["speaker"] for turn in payload["result"]["turns"][-3:]], ["Alex", "Sarah", "Astrid"])

    def test_crew_repair_last_reconstructs_from_only_latest_markdown(self):
        temp, transcript_dir, _state_file, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _fake_transport):
            client = self._client()
            started = self._start_terse_crew(client)
            conversation_id = started["conversation_id"]
            conv_md = transcript_dir / f"{conversation_id}.md"
            conv_json = transcript_dir / f"{conversation_id}.json"
            latest_md = transcript_dir / "latest_multi_agent_transcript.md"
            latest_json = transcript_dir / "latest_multi_agent_transcript.json"
            conv_md.unlink()
            conv_json.unlink()
            latest_json.unlink()
            self.assertTrue(latest_md.exists())

            response = client.post("/api/chat", json={"message": "/crew_repair_last"})
            payload = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertTrue(payload["ok"])
            self.assertTrue(conv_md.exists())
            self.assertTrue(conv_json.exists())
            self.assertTrue(latest_json.exists())
            self.assertIn("created", payload["text"])


if __name__ == "__main__":
    unittest.main()
