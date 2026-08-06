from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.core.sarah_engine import configure_sarah_engine, generate_sarah_reply, reset_sarah_session
from app.core.sarah_response import SarahResponse
from app.tools.web_router import WebRetrievalResult


class FakeClient:
    pass


class SarahEngineTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_sarah_engine()

    def test_generate_sarah_reply_wraps_shared_response_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))
            response = _response("plain answer")

            configure_sarah_engine(settings=settings, client=FakeClient())
            with patch("app.core.sarah_engine.generate_sarah_response", return_value=response) as generate_response:
                reply = generate_sarah_reply("Hello Sarah", session_id="test")

            self.assertEqual(reply.answer, "plain answer")
            self.assertEqual(reply.sources, [])
            generate_response.assert_called_once()
            self.assertEqual(generate_response.call_args.kwargs["user_input"], "Hello Sarah")
            self.assertEqual(generate_response.call_args.kwargs["settings"], settings)

    def test_session_history_is_shared_inside_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))
            histories = []

            def fake_response(**kwargs):
                histories.append(kwargs["history"])
                return _response(f"turn {len(histories)}")

            configure_sarah_engine(settings=settings, client=FakeClient())
            with patch("app.core.sarah_engine.generate_sarah_response", side_effect=fake_response):
                generate_sarah_reply("first", session_id="test")
                generate_sarah_reply("second", session_id="test")

            self.assertEqual(histories[0], [])
            self.assertEqual(len(histories[1]), 1)
            self.assertEqual(histories[1][0].user, "first")
            self.assertEqual(histories[1][0].assistant, "turn 1")

    def test_reset_sarah_session_clears_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))
            histories = []

            def fake_response(**kwargs):
                histories.append(kwargs["history"])
                return _response("ok")

            configure_sarah_engine(settings=settings, client=FakeClient())
            with patch("app.core.sarah_engine.generate_sarah_response", side_effect=fake_response):
                generate_sarah_reply("first", session_id="test")
                reset_sarah_session("test")
                generate_sarah_reply("after reset", session_id="test")

            self.assertEqual(histories[1], [])


def _test_settings(tmp: Path):
    settings = load_settings(env_file=tmp / ".env")
    return settings.__class__(
        **{
            **settings.__dict__,
            "openai_api_key": "test-key",
            "memory_file": tmp / "memory.jsonl",
            "web_cache_dir": tmp / "web_cache",
            "vector_store_dir": tmp / "vector_store",
        }
    )


def _response(answer: str) -> SarahResponse:
    return SarahResponse(
        answer=answer,
        retrieval=None,
        retrieval_error=None,
        web_result=WebRetrievalResult(
            used_web=False,
            failed=False,
            reason="not needed",
            sources=[],
            timestamp="2026-06-04T00:00:00+00:00",
        ),
        messages=[],
        prompt_debug_summary={"layers": []},
    )


if __name__ == "__main__":
    unittest.main()
