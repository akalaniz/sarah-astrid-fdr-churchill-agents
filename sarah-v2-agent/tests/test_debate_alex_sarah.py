from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.orchestration import agent_orchestrator as orchestrator
from app.ui.web_app import create_app


def _short_transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
    return f"{agent} answers {from_agent}: plausible, but the mechanism still has to earn the name intelligence."


class SarahDebateAlexTests(unittest.TestCase):
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
        return re.findall(r"(?m)^(Sarah|Astrid|Alex-proxy|Alex):$", text)

    def test_default_debate_alex_visible_labels(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _short_transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/debate_alex Are Alex's timeline LLMs a path to GI?"},
            )
            payload = response.json()
            text = payload["text"]

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload.get("ok", True))
        self.assertEqual(self._labels(text)[:6], ["Sarah", "Alex-proxy", "Sarah", "Astrid", "Sarah", "Astrid"])
        self.assertNotRegex(text, r"(?m)^Alex:$")
        self.assertNotIn("Crew dialogue complete", text)
        self.assertNotIn("Transcript:", text)
        self.assertNotIn("Final Synthesis", text)
        self.assertNotIn("Situation compression", text)

    def test_debate_alex_turns_word_limit_and_no_report_format(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _short_transport):
            response = self._client().post(
                "/api/chat",
                json={
                    "message": "/debate_alex --turns 8 --max-words 50 --style dinner --no-bullets --no-reports topic"
                },
            )
            payload = response.json()
            text = payload["text"]

        self.assertEqual(response.status_code, 200)
        turns = payload["result"]["turns"]
        self.assertEqual(len(turns), 8)
        for turn in turns:
            self.assertLessEqual(len(turn["message"].split()), 60)
        self.assertNotRegex(text, r"(?m)^\s*[-*]\s+")
        self.assertNotRegex(text, r"(?m)^\s*\d+[\.)]\s+")
        self.assertNotIn("Situation compression", text)
        self.assertNotIn("Final Synthesis", text)

    def test_override_first_and_judge(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _short_transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/debate_alex --first Astrid --judge Sarah --turns 8 topic"},
            )
            payload = response.json()

        speakers = [turn["speaker"] for turn in payload["result"]["turns"]]
        self.assertEqual(speakers, ["Astrid", "Alex-proxy", "Astrid", "Sarah", "Astrid", "Sarah", "Astrid", "Sarah"])

    def test_real_alex_continuation_stores_alex_but_visible_does_not_echo(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _short_transport):
            client = self._client()
            client.post("/api/chat", json={"message": "/debate_alex Are timeline LLMs a path to GI?"})
            response = client.post("/api/chat", json={"message": "/Alex Stop. Sarah, what did Astrid miss?"})
            payload = response.json()
            text = payload["text"]
            transcript = json.loads(Path(payload["result"]["transcript_json_path"]).read_text(encoding="utf-8"))

        self.assertEqual(response.status_code, 200)
        self.assertNotRegex(text, r"(?m)^Alex:$")
        self.assertNotIn("Stop. Sarah, what did Astrid miss?", text)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)
        recent = transcript["turns"][-3:]
        self.assertEqual(recent[0]["speaker"], "Alex")
        self.assertIn("Stop. Sarah, what did Astrid miss?", recent[0]["message"])
        self.assertEqual([turn["speaker"] for turn in recent[1:]], ["Sarah", "Astrid"])
        self.assertNotIn("Alex-proxy", [turn["speaker"] for turn in recent])

    def test_astrid_unavailable_returns_clean_json_error(self):
        def reachable(agent: str) -> bool:
            return agent == "Sarah"

        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "is_agent_endpoint_reachable", reachable):
            response = self._client().post("/api/chat", json={"message": "/debate_alex topic"})
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "Debate Alex orchestration failed")
        self.assertIn("Astrid unavailable at http://127.0.0.1:8001/agent/respond", payload["text"])
        self.assertNotIn("Internal Server Error", payload["text"])

    def test_apostrophes_do_not_crash_parser(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _short_transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/debate_alex Sarah and Astrid debate Alex's LLM timeline and don't overthink it."},
            )
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload.get("ok", True))
        self.assertIn("Sarah:", payload["text"])
        self.assertIn("Alex-proxy:", payload["text"])
        self.assertIn("Astrid:", payload["text"])


if __name__ == "__main__":
    unittest.main()
