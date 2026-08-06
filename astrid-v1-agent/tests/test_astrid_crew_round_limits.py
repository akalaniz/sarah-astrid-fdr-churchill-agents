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
    return f"{agent} answers {from_agent}: silly enough to be useful, short enough not to become a theology committee."


def _round_label_transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
    return (
        "Round 1\n"
        f"{agent} says the first useful thing.\n"
        "Round 2\n"
        "This internal round label should vanish from visible output.\n"
        "Round 3: And this one should be folded into one conversational turn."
    )


class AstridCrewRoundLimitTests(unittest.TestCase):
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
        return re.findall(r"(?m)^(Astrid|Sarah|Alex-proxy|Alex|Joint verdict for Alex):$", text)

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\b[\w'-]+\b", text))

    def test_natural_language_rounds_max_words_and_joint_verdict(self):
        command = (
            "/crew God cannot make a taco so hot and spicy that even HE can't eat it.\n"
            "Keep it short:\n\n"
            "* 3 rounds only\n"
            "* Each turn max 90 words\n"
            "* Be silly\n"
            "* No full CAS report\n"
            "* No bullet flood\n"
            "* Actual give-and-take\n"
            "* End with a joint verdict for Alex"
        )
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post("/api/chat", json={"message": command})
            payload = response.json()
            text = payload["text"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self._labels(text),
            ["Astrid", "Sarah", "Astrid", "Sarah", "Astrid", "Sarah", "Joint verdict for Alex"],
        )
        self.assertEqual(payload["result"]["parsed_rounds"], 3)
        self.assertEqual(payload["result"]["parsed_max_words"], 90)
        self.assertEqual(payload["result"]["total_agent_turns"], 6)
        self.assertTrue(payload["result"]["joint_verdict_requested"])
        self.assertNotRegex(text, r"(?i)\bRound\s+[123]\b")
        self.assertNotIn("Final Synthesis", text)
        self.assertNotIn("Situation compression", text)
        self.assertNotIn("Perrow placement", text)
        self.assertNotIn("Alex-proxy", text)
        for turn in payload["result"]["turns"]:
            self.assertLessEqual(self._word_count(turn["message"]), 108)

    def test_rounds_flag_means_exchange_pairs(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post("/api/chat", json={"message": "/crew --rounds 2 --max-words 50 silly topic"})
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual([turn["speaker"] for turn in payload["result"]["turns"]], ["Astrid", "Sarah", "Astrid", "Sarah"])
        self.assertNotIn("Joint verdict for Alex", payload["text"])

    def test_turns_flag_means_total_agent_turns(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post("/api/chat", json={"message": "/crew --turns 6 --max-words 50 topic"})
            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload["result"]["turns"]), 6)
        self.assertEqual([turn["speaker"] for turn in payload["result"]["turns"]], ["Astrid", "Sarah", "Astrid", "Sarah", "Astrid", "Sarah"])

    def test_internal_round_labels_are_removed_from_agent_answer(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _round_label_transport):
            response = self._client().post("/api/chat", json={"message": "/crew --turns 2 --max-words 90 topic"})
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertNotRegex(text, r"(?i)\bRound\s+1\b")
        self.assertNotRegex(text, r"(?i)\bRound\s+2\b")
        self.assertNotRegex(text, r"(?i)\bRound\s+3\b")
        self.assertEqual(self._labels(text), ["Astrid", "Sarah"])

    def test_debate_alex_remains_unchanged(self):
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            response = self._client().post("/api/chat", json={"message": "/debate_alex --turns 6 topic"})
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._labels(text), ["Astrid", "Alex-proxy", "Astrid", "Sarah", "Astrid", "Sarah"])

    def test_debug_crew_turns_reports_plan(self):
        command = "/crew God cannot make the taco.\n3 rounds only\nEach turn max 90 words\nEnd with a joint verdict for Alex"
        temp, patches = self._patched_paths()
        with temp, patches[0], patches[1], patches[2], patch.object(orchestrator, "call_agent_respond_endpoint", _transport):
            client = self._client()
            client.post("/api/chat", json={"message": command})
            response = client.post("/api/chat", json={"message": "/debug_crew_turns"})
            text = response.json()["text"]

        self.assertEqual(response.status_code, 200)
        self.assertIn("command_type: crew", text)
        self.assertIn("use_alex_proxy: false", text)
        self.assertIn("rounds: 3", text)
        self.assertIn("max_words: 90", text)
        self.assertIn("total_agent_turns: 6", text)
        self.assertIn("planned_turn_order: Astrid, Sarah, Astrid, Sarah, Astrid, Sarah", text)
        self.assertIn("joint_verdict_requested: true", text)


if __name__ == "__main__":
    unittest.main()
