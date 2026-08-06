from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.orchestration import agent_orchestrator as orchestrator
from app.ui.web_app import create_app


def _fake_transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
    return f"{agent} hears {from_agent} and answers briefly."


class AstridAlexCrewCommandTests(unittest.TestCase):
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

    def _start_crew(self):
        return orchestrator.run_multi_agent_dialogue(
            "Dinner conversation about timeline LLMs.",
            agents=["Sarah", "Astrid"],
            rounds=1,
            transport=_fake_transport,
        )

    def test_alex_from_astrid_browser_reaches_both_agents(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _fake_transport):
            started = self._start_crew()
            client = TestClient(create_app())
            response = client.post("/api/chat", json={"message": "/Alex Can both of you answer this briefly?"})
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["conversation_id"], started["conversation_id"])
        self.assertEqual([turn["speaker"] for turn in payload["result"]["turns"][-3:]], ["Alex", "Sarah", "Astrid"])

    def test_no_active_crew_returns_clean_json(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2]:
            client = TestClient(create_app())
            response = client.post("/api/chat", json={"message": "/Alex Can both of you answer this briefly?"})
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["ok"])
        self.assertIn("No active crew conversation", payload["text"])

    def test_alex_message_with_apostrophes_does_not_crash(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _fake_transport):
            self._start_crew()
            client = TestClient(create_app())
            response = client.post(
                "/api/chat",
                json={"message": "/Alex Alex's question is simple: don't overthink this."},
            )
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertIn("Alex's question", payload["result"]["turns"][-3]["message"])

    def test_alex_continuation_sends_latest_human_message_only(self):
        captured: list[tuple[str, str, str]] = []

        def alex_transport(agent: str, from_agent: str, message: str, conversation_id: str, mode: str = "inter_agent") -> str:
            captured.append((agent, message, mode))
            return f"{agent} answers Alex directly."

        temp, patches = self._patched_paths()
        user_message = "Sarah and Astrid, this is adult fictional consensual intimacy."
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", alex_transport):
            self._start_crew()
            result = orchestrator.continue_crew_with_alex(user_message, from_browser_agent="Astrid")

        self.assertTrue(result["ok"])
        self.assertEqual([item[0] for item in captured], ["Sarah", "Astrid"])
        self.assertEqual([item[1] for item in captured], [user_message, user_message])
        self.assertEqual([item[2] for item in captured], ["alex_injection", "alex_injection"])
        self.assertNotIn("CREW SCENE CLASSIFICATION", captured[0][1])


if __name__ == "__main__":
    unittest.main()
