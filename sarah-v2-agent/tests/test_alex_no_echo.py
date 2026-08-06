from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.orchestration import agent_orchestrator as orchestrator
from app.ui.web_app import create_app


def _fake_transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
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

    def _start_crew(self, client: TestClient) -> dict:
        response = client.post("/api/chat", json={"message": "/crew_terse Sarah and Astrid begin."})
        self.assertEqual(response.status_code, 200)
        return response.json()["result"]

    def test_alex_message_is_saved_but_not_echoed_by_default(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _fake_transport):
            client = TestClient(create_app())
            self._start_crew(client)
            response = client.post("/api/chat", json={"message": "/Alex Can both of you hear me?"})
            payload = response.json()

        text = payload["text"]
        self.assertEqual(response.status_code, 200)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)
        self.assertNotIn("Alex:", text)
        self.assertNotIn("Can both of you hear me?", text)
        transcript_tail = payload["result"]["turns"][-3:]
        self.assertEqual([turn["speaker"] for turn in transcript_tail], ["Alex", "Sarah", "Astrid"])
        self.assertIn("Can both of you hear me?", transcript_tail[0]["message"])

    def test_crew_echo_alex_on_echoes_alex_message(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _fake_transport):
            client = TestClient(create_app())
            self._start_crew(client)
            client.post("/api/chat", json={"message": "/crew_echo_alex on"})
            response = client.post("/api/chat", json={"message": "/Alex Can both of you hear me?"})
            text = response.json()["text"]

        self.assertIn("Alex:", text)
        self.assertIn("Can both of you hear me?", text)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)

    def test_crew_echo_alex_off_stops_echoing_alex_message(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _fake_transport):
            client = TestClient(create_app())
            self._start_crew(client)
            client.post("/api/chat", json={"message": "/crew_echo_alex on"})
            client.post("/api/chat", json={"message": "/crew_echo_alex off"})
            response = client.post("/api/chat", json={"message": "/Alex Can both of you hear me?"})
            text = response.json()["text"]

        self.assertNotIn("Alex:", text)
        self.assertNotIn("Can both of you hear me?", text)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)

    def test_verbose_off_keeps_no_metadata_and_no_alex_echo(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _fake_transport):
            client = TestClient(create_app())
            self._start_crew(client)
            client.post("/api/chat", json={"message": "/crew_verbose off"})
            response = client.post("/api/chat", json={"message": "/Alex Message for the crew."})
            text = response.json()["text"]

        self.assertNotIn("Crew dialogue complete", text)
        self.assertNotIn("Transcript:", text)
        self.assertNotIn("Participants:", text)
        self.assertNotIn("Alex:", text)
        self.assertNotIn("Message for the crew.", text)

    def test_verbose_on_may_include_alex_message_with_separate_agent_replies(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _fake_transport):
            client = TestClient(create_app())
            self._start_crew(client)
            client.post("/api/chat", json={"message": "/crew_verbose on"})
            response = client.post("/api/chat", json={"message": "/Alex Message for the crew."})
            text = response.json()["text"]

        self.assertIn("Crew dialogue complete", text)
        self.assertIn("Alex", text)
        self.assertIn("Message for the crew.", text)
        self.assertIn("Sarah", text)
        self.assertIn("Astrid", text)


if __name__ == "__main__":
    unittest.main()
