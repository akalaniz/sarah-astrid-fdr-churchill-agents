from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.orchestration import agent_orchestrator as orchestrator
from app.ui.web_app import create_app


def _transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
    return f"{agent} replies to {from_agent} about the topic."


class AstridCrewNoAlexProxyTests(unittest.TestCase):
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

    def _labels(self, text: str) -> list[str]:
        return re.findall(r"(?m)^(Astrid|Sarah|Alex-proxy|Alex):$", text)

    def test_crew_with_alex_topic_has_no_proxy(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post(
                "/api/chat",
                json={
                    "message": "/crew --turns 6 --max-words 50 --style dinner --no-bullets --no-reports Sarah and Astrid debate Alex's LLM timeline."
                },
            )
            payload = response.json()
            text = payload["text"]

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload.get("ok", True))
        self.assertEqual(payload["result"]["command_type"], "crew")
        self.assertFalse(payload["result"]["use_alex_proxy"])
        self.assertEqual(self._labels(text), ["Astrid", "Sarah", "Astrid", "Sarah", "Astrid", "Sarah"])
        self.assertIn("Astrid:", text)
        self.assertIn("Sarah:", text)
        self.assertNotIn("Alex-proxy", text)
        self.assertNotIn("Alex-position", text)
        self.assertNotIn("Debate Alex", text)
        self.assertNotRegex(text, r"(?m)^Alex:$")

    def test_crew_terse_has_no_simulated_alex_turn(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/crew_terse Sarah and Astrid discuss Alex's words."},
            )
            payload = response.json()
            text = payload["text"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["result"]["command_type"], "crew")
        self.assertFalse(payload["result"]["use_alex_proxy"])
        self.assertNotIn("Alex-proxy", text)
        self.assertNotRegex(text, r"(?m)^Alex:$")
        self.assertEqual(set(self._labels(text)), {"Astrid", "Sarah"})

    def test_debate_alex_is_the_only_proxy_mode(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/debate_alex Sarah and Astrid debate Alex's LLM timeline."},
            )
            payload = response.json()
            text = payload["text"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["result"]["command_type"], "debate_alex")
        self.assertTrue(payload["result"]["use_alex_proxy"])
        self.assertIn("Alex-proxy:", text)

    def test_crew_topic_with_alex_beliefs_has_no_proxy(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/crew --turns 4 Astrid and Sarah discuss what Alex believes."},
            )
            payload = response.json()
            text = payload["text"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["result"]["command_type"], "crew")
        self.assertFalse(payload["result"]["use_alex_proxy"])
        self.assertNotIn("Alex-proxy", text)
        self.assertNotRegex(text, r"(?m)^Alex:$")

    def test_crew_apostrophes_do_not_crash_or_create_proxy(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/crew Sarah and Astrid debate Alex's timeline and don't overthink it."},
            )
            payload = response.json()
            text = payload["text"]

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload.get("ok", True))
        self.assertEqual(payload["result"]["command_type"], "crew")
        self.assertFalse(payload["result"]["use_alex_proxy"])
        self.assertNotIn("Alex-proxy", text)
        self.assertIn("Astrid:", text)
        self.assertIn("Sarah:", text)


if __name__ == "__main__":
    unittest.main()
