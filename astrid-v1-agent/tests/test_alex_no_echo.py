from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.orchestration import agent_orchestrator as orchestrator
from app.ui.web_app import create_app


def _brief_transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
    return f"{agent} hears Alex and answers briefly."


class AlexNoEchoTests(unittest.TestCase):
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
        return temp, transcript_dir, patches

    def _client(self) -> TestClient:
        return TestClient(create_app())

    def _start_crew(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"message": "/crew_terse Sarah and Astrid begin."})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json().get("ok", True))

    def test_alex_no_echo_default_but_transcript_saves_input(self):
        temp, transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _brief_transport):
            client = self._client()
            self._start_crew(client)
            response = client.post("/api/chat", json={"message": "/Alex Can both of you hear me?"})
            payload = response.json()
            text = payload["text"]

            transcript_path = Path(payload["result"]["transcript_json_path"])
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)
        self.assertNotIn("Alex:", text)
        self.assertNotIn("Can both of you hear me?", text)
        recent_speakers = [turn["speaker"] for turn in transcript["turns"][-3:]]
        self.assertEqual(recent_speakers, ["Alex", "Sarah", "Astrid"])
        self.assertIn("Can both of you hear me?", transcript["turns"][-3]["message"])

    def test_crew_echo_alex_on_restores_visible_echo(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _brief_transport):
            client = self._client()
            self._start_crew(client)
            client.post("/api/chat", json={"message": "/crew_echo_alex on"})
            response = client.post("/api/chat", json={"message": "/Alex Can both of you hear me?"})
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertIn("Alex:", text)
        self.assertIn("Can both of you hear me?", text)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)

    def test_crew_echo_alex_off_suppresses_visible_echo(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _brief_transport):
            client = self._client()
            self._start_crew(client)
            client.post("/api/chat", json={"message": "/crew_echo_alex on"})
            client.post("/api/chat", json={"message": "/crew_echo_alex off"})
            response = client.post("/api/chat", json={"message": "/Alex Can both of you hear me?"})
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Alex:", text)
        self.assertNotIn("Can both of you hear me?", text)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)

    def test_crew_verbose_off_suppresses_echo_and_metadata(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _brief_transport):
            client = self._client()
            self._start_crew(client)
            client.post("/api/chat", json={"message": "/crew_verbose off"})
            response = client.post("/api/chat", json={"message": "/Alex Message for the crew."})
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Crew dialogue complete", text)
        self.assertNotIn("Transcript:", text)
        self.assertNotIn("Participants:", text)
        self.assertNotIn("Alex:", text)
        self.assertNotIn("Message for the crew.", text)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)

    def test_crew_verbose_on_may_show_alex_and_metadata(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _brief_transport):
            client = self._client()
            self._start_crew(client)
            client.post("/api/chat", json={"message": "/crew_verbose on"})
            response = client.post("/api/chat", json={"message": "/Alex Message for the crew."})
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertIn("Crew dialogue complete", text)
        self.assertIn("Alex:", text)
        self.assertIn("Message for the crew.", text)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)

    def test_no_active_crew_returns_json_error(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _brief_transport):
            response = self._client().post("/api/chat", json={"message": "/Alex Can both of you hear me?"})
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["ok"])
        self.assertIn("No active crew conversation", payload["text"])


if __name__ == "__main__":
    unittest.main()
