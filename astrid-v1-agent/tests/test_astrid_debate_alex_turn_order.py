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
    return f"{agent} answers {from_agent}: concise dinner-table argument."


class AstridDebateAlexTurnOrderTests(unittest.TestCase):
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

    def test_public_turn_order_helper_defaults(self):
        self.assertEqual(
            orchestrator.build_debate_alex_turn_order(first="Astrid", judge="Sarah", turns=8),
            ["Astrid", "Alex-proxy", "Astrid", "Sarah", "Astrid", "Sarah", "Astrid", "Sarah"],
        )

    def test_public_turn_order_helper_override(self):
        self.assertEqual(
            orchestrator.build_debate_alex_turn_order(first="Sarah", judge="Astrid", turns=8),
            ["Sarah", "Alex-proxy", "Sarah", "Astrid", "Sarah", "Astrid", "Sarah", "Astrid"],
        )

    def test_debate_alex_six_turn_order(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post(
                "/api/chat",
                json={
                    "message": "/debate_alex --turns 6 --max-words 50 --style dinner --no-bullets --no-reports Are Alex's timeline LLMs a path to GI?"
                },
            )
            payload = response.json()
            text = payload["text"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["result"]["command_type"], "debate_alex")
        self.assertTrue(payload["result"]["use_alex_proxy"])
        self.assertEqual(self._labels(text), ["Astrid", "Alex-proxy", "Astrid", "Sarah", "Astrid", "Sarah"])
        self.assertEqual(self._labels(text).count("Alex-proxy"), 1)
        self.assertNotRegex(text, r"(?m)^Alex:$")
        self.assertNotIn("Crew dialogue complete", text)
        self.assertNotIn("Transcript:", text)
        self.assertNotIn("Final Synthesis", text)

    def test_debate_alex_eight_turn_order(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post("/api/chat", json={"message": "/debate_alex --turns 8 topic"})
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [turn["speaker"] for turn in payload["result"]["turns"]],
            ["Astrid", "Alex-proxy", "Astrid", "Sarah", "Astrid", "Sarah", "Astrid", "Sarah"],
        )
        self.assertEqual(self._labels(payload["text"]), ["Astrid", "Alex-proxy", "Astrid", "Sarah", "Astrid", "Sarah", "Astrid", "Sarah"])

    def test_debate_alex_first_sarah_override(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/debate_alex --first Sarah --judge Astrid --turns 6 topic"},
            )
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._labels(payload["text"]), ["Sarah", "Alex-proxy", "Sarah", "Astrid", "Sarah", "Astrid"])

    def test_crew_still_has_no_proxy(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post("/api/chat", json={"message": "/crew --turns 6 topic about Alex"})
            payload = response.json()
            text = payload["text"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["result"]["command_type"], "crew")
        self.assertFalse(payload["result"]["use_alex_proxy"])
        self.assertNotIn("Alex-proxy", text)
        self.assertEqual(self._labels(text), ["Astrid", "Sarah", "Astrid", "Sarah", "Astrid", "Sarah"])

    def test_debate_alex_apostrophes_do_not_crash(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/debate_alex Astrid debates Alex's LLM timeline and doesn't overthink it."},
            )
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload.get("ok", True))
        self.assertEqual([turn["speaker"] for turn in payload["result"]["turns"][:4]], ["Astrid", "Alex-proxy", "Astrid", "Sarah"])
        self.assertIn("Alex-proxy:", payload["text"])

    def test_debug_debate_alex_reports_mode_details(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            client = self._client()
            client.post("/api/chat", json={"message": "/debate_alex --turns 6 topic"})
            response = client.post("/api/chat", json={"message": "/debug_debate_alex"})
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertIn("command_type: debate_alex", text)
        self.assertIn("use_alex_proxy: true", text)
        self.assertIn("first: Astrid", text)
        self.assertIn("judge: Sarah", text)
        self.assertIn("requested_turns: 6", text)
        self.assertIn("planned_turn_order: Astrid, Alex-proxy, Astrid, Sarah, Astrid, Sarah", text)
        self.assertIn("route_handler_used: debate_alex", text)


if __name__ == "__main__":
    unittest.main()
