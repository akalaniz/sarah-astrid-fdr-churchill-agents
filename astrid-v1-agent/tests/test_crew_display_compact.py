from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.orchestration import agent_orchestrator as orchestrator
from app.ui.web_app import create_app


REPORTISH_RESPONSE = (
    "Situation compression\n"
    "1. This is an overgrown staff-paper paragraph that should become dinner conversation instead of a report. "
    "Perrow placement\n"
    "- This bullet should not survive visible compact output. "
    "Patch recommendations\n"
    "- Keep it short and answer the previous speaker directly."
)


def _reportish_transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
    return REPORTISH_RESPONSE


class CrewDisplayCompactTests(unittest.TestCase):
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

    def test_dinner_crew_hides_metadata_and_synthesis(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _reportish_transport):
            response = self._client().post(
                "/api/chat",
                json={
                    "message": "/crew --turns 6 --max-words 50 --style dinner --no-bullets --no-reports Sarah and Astrid talk."
                },
            )
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Crew dialogue complete", text)
        self.assertNotIn("Transcript:", text)
        self.assertNotIn("Participants:", text)
        self.assertNotIn("Turns completed:", text)
        self.assertNotIn("Final Synthesis", text)
        self.assertNotIn("Points of agreement", text)
        self.assertNotIn("Recommended next question", text)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)

    def test_alex_visible_output_only_shows_new_exchange(self):
        calls = {"count": 0}

        def transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
            calls["count"] += 1
            if calls["count"] <= 2:
                return f"OLD TRANSCRIPT HISTORY from {agent}."
            return f"{agent} hears Alex clearly."

        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", transport):
            client = self._client()
            client.post("/api/chat", json={"message": "/crew_terse Sarah and Astrid begin."})
            response = client.post("/api/chat", json={"message": "/Alex Can both of you hear me?"})
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Alex:", text)
        self.assertNotIn("Can both of you hear me?", text)
        self.assertIn("Sarah:", text)
        self.assertIn("Astrid:", text)
        self.assertNotIn("OLD TRANSCRIPT HISTORY", text)
        self.assertNotIn("Crew dialogue complete", text)

    def test_report_headings_are_removed_from_visible_compact_output(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _reportish_transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/crew_terse Sarah and Astrid talk."},
            )
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Situation compression", text)
        self.assertNotIn("Perrow placement", text)
        self.assertNotIn("Patch recommendations", text)

    def test_report_flag_allows_final_synthesis_without_metadata_by_default(self):
        temp, _transcript_dir, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _reportish_transport):
            response = self._client().post(
                "/api/chat",
                json={"message": "/crew --report Sarah and Astrid talk."},
            )
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertIn("Final Synthesis", text)
        self.assertNotIn("Crew dialogue complete", text)
        self.assertNotIn("Transcript:", text)


if __name__ == "__main__":
    unittest.main()
