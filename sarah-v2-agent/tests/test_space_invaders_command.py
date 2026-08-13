import json
from subprocess import CompletedProcess
import unittest
from unittest.mock import patch

from app.core import chat_loop
from app.ui import web_app


class SpaceInvadersCommandTests(unittest.TestCase):
    def test_web_command_uses_shared_bridge(self) -> None:
        payload = {"ok": True, "started": True, "mode": "sarah", "message": "started"}
        completed = CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
        with patch.object(web_app.subprocess, "run", return_value=completed) as run:
            result = web_app._handle_space_invaders_command("/space_invaders sarah")
        self.assertTrue(result["ok"])
        self.assertEqual(result["text"], "started")
        self.assertIn("games", run.call_args.args[0][2])
        self.assertEqual(run.call_args.args[0][-2:], ["start", "sarah"])

    def test_cli_stop_uses_shared_bridge(self) -> None:
        payload = {"ok": True, "stopped": True, "message": "stopped"}
        completed = CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
        with patch.object(chat_loop.subprocess, "run", return_value=completed) as run:
            result = chat_loop._handle_space_invaders_command("/space_invaders stop")
        self.assertEqual(result, "stopped")
        self.assertEqual(run.call_args.args[0][-1], "stop")

    def test_invalid_mode_returns_usage_without_launching(self) -> None:
        with patch.object(web_app.subprocess, "run") as run:
            result = web_app._handle_space_invaders_command("/space_invaders warp")
        self.assertFalse(result["ok"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
