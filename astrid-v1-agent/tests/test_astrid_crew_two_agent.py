from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.orchestration import agent_orchestrator as orchestrator
from app.ui.web_app import create_app


def _transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
    return f"{agent} replies to {from_agent} about the crew topic."


class AstridCrewTwoAgentTests(unittest.TestCase):
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
        return temp, patches

    def _client(self) -> TestClient:
        return TestClient(create_app())

    def test_astrid_launched_crew_alternates_both_agents(self):
        calls: list[str] = []

        def transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
            calls.append(agent)
            return _transport(agent, from_agent, message, conversation_id)

        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", transport):
            response = self._client().post(
                "/api/chat",
                json={
                    "message": "/crew --turns 6 --max-words 50 --style dinner Sarah and Astrid debate Alex's LLM timeline."
                },
            )
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload.get("ok", True))
        speakers = [turn["speaker"] for turn in payload["result"]["turns"]]
        self.assertEqual(speakers, ["Astrid", "Sarah", "Astrid", "Sarah", "Astrid", "Sarah"])
        self.assertIn("Sarah:", payload["text"])
        self.assertIn("Astrid:", payload["text"])
        self.assertEqual(calls, speakers)

    def test_first_astrid_turn_order(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post("/api/chat", json={"message": "/crew --turns 4 --first Astrid topic"})
            payload = response.json()

        speakers = [turn["speaker"] for turn in payload["result"]["turns"]]
        self.assertEqual(speakers, ["Astrid", "Sarah", "Astrid", "Sarah"])

    def test_first_sarah_turn_order_from_astrid_browser(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post("/api/chat", json={"message": "/crew --turns 4 --first Sarah topic"})
            payload = response.json()

        speakers = [turn["speaker"] for turn in payload["result"]["turns"]]
        self.assertEqual(speakers, ["Sarah", "Astrid", "Sarah", "Astrid"])

    def test_sarah_endpoint_unavailable_is_clean_json_error(self):
        def reachable(agent: str) -> bool:
            return agent == "Astrid"

        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "is_agent_endpoint_reachable", reachable):
            response = self._client().post("/api/chat", json={"message": "/crew --turns 4 topic"})
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "Crew orchestration failed")
        self.assertIn("Sarah unavailable at http://127.0.0.1:8000/agent/respond", payload["text"])
        self.assertNotIn("Internal Server Error", payload["text"])

    def test_explicit_agents_calls_both_agents(self):
        calls: list[str] = []

        def transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
            calls.append(agent)
            return _transport(agent, from_agent, message, conversation_id)

        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/crew --agents Sarah,Astrid --turns 4 topic"},
            )
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload.get("ok", True))
        self.assertEqual([turn["speaker"] for turn in payload["result"]["turns"]], ["Astrid", "Sarah", "Astrid", "Sarah"])
        self.assertIn("Sarah", calls)
        self.assertIn("Astrid", calls)

    def test_crew_probe_reports_both_endpoints(self):
        with patch.object(orchestrator, "is_agent_endpoint_reachable", lambda agent: agent == "Astrid"):
            response = self._client().post("/api/chat", json={"message": "/crew_probe"})
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertIn("Sarah endpoint: unreachable", text)
        self.assertIn("Astrid endpoint: reachable", text)
        self.assertIn("Can Sarah respond: false", text)
        self.assertIn("Can Astrid respond: true", text)


if __name__ == "__main__":
    unittest.main()
