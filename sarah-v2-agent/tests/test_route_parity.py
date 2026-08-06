from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core import chat_loop, sarah_engine, sarah_response
from app.core.config import load_settings
from app.rag.retriever import RetrievalResult
from app.ui import web_app


ROUTE_PARITY_PROMPTS = (
    "Iran tests a nuclear weapon. Give me DIME COAs.",
    "Sarah, I want adult love that still has bodies.",
    "Darwin scares me.",
)


@dataclass
class FakeSarahReply:
    answer: str = "Sarah parity response"
    sources: list[dict[str, str]] = field(default_factory=lambda: [{"source_filename": "Alex Sarah.docx"}])
    web_sources: list[dict[str, str]] = field(default_factory=list)
    web_status: dict[str, object] = field(default_factory=lambda: {"used_web": False, "failed": False, "reason": "test"})
    has_sufficient_evidence: bool = True
    prompt_debug_summary: dict[str, object] = field(default_factory=lambda: {"layers": []})
    retrieval_results: list[RetrievalResult] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.answer


class RouteParityTests(unittest.TestCase):
    def test_web_endpoint_calls_shared_sarah_engine_for_route_parity_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = _test_app(Path(tmp))
            client = TestClient(app)

            with patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeSarahReply()) as generate_reply:
                for prompt in ROUTE_PARITY_PROMPTS:
                    with self.subTest(prompt=prompt):
                        response = client.post("/api/chat", json={"message": prompt})
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json()["text"], "Sarah parity response")

        self.assertEqual(generate_reply.call_count, len(ROUTE_PARITY_PROMPTS))
        self.assertEqual(
            [call.args[0] for call in generate_reply.call_args_list],
            list(ROUTE_PARITY_PROMPTS),
        )
        self.assertTrue(all(call.kwargs["session_id"] == "web" for call in generate_reply.call_args_list))

    def test_cli_path_calls_shared_sarah_engine_for_route_parity_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))

            with patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeSarahReply()) as generate_reply:
                with patch("builtins.input", side_effect=[*ROUTE_PARITY_PROMPTS, "/quit"]):
                    with patch("sys.stdout", new=io.StringIO()):
                        chat_loop.run_chat_loop(settings)

        self.assertEqual(generate_reply.call_count, len(ROUTE_PARITY_PROMPTS))
        self.assertEqual(
            [call.args[0] for call in generate_reply.call_args_list],
            list(ROUTE_PARITY_PROMPTS),
        )
        self.assertTrue(all(call.kwargs["session_id"] == "cli" for call in generate_reply.call_args_list))

    def test_cli_and_web_routes_do_not_construct_their_own_sarah_prompt(self) -> None:
        forbidden_route_tokens = (
            "build_prompt(",
            "generate_sarah_response(",
            "SarahOpenAIClient(",
            "retrieve(",
            "infer_mode_from_state(",
            "build_style_directives(",
        )
        for module in (chat_loop, web_app):
            source = inspect.getsource(module)
            for token in forbidden_route_tokens:
                with self.subTest(module=module.__name__, token=token):
                    self.assertNotIn(token, source)

        self.assertIn("sarah_engine.generate_sarah_reply", inspect.getsource(chat_loop))
        self.assertIn("sarah_engine.generate_sarah_reply", inspect.getsource(web_app))

    def test_shared_engine_loads_single_prompt_memory_rag_style_and_model_path(self) -> None:
        engine_source = inspect.getsource(sarah_engine)
        response_source = inspect.getsource(sarah_response)

        self.assertIn("generate_sarah_response(", engine_source)
        self.assertIn("settings=settings", engine_source)
        self.assertIn("settings.sarah_model", response_source)
        self.assertIn("MemoryStore(settings.memory_file)", response_source)
        self.assertIn("retrieve(user_input", response_source)
        self.assertIn("infer_mode_from_state(", response_source)
        self.assertIn("build_style_directives(", response_source)
        self.assertIn("build_prompt(", response_source)
        self.assertIn("retrieve_web_context(", response_source)

        from app.core import prompt_builder
        from app.persona import constitution, style_engine

        self.assertIn("config\" / \"sarah_master_prompt.md", inspect.getsource(prompt_builder))
        self.assertIn("sarah_constitution.yaml", inspect.getsource(constitution))
        self.assertIn("adult_intimacy", inspect.getsource(style_engine))

    def test_web_route_cannot_bypass_adult_intimacy_mode_with_local_prompting(self) -> None:
        web_source = inspect.getsource(web_app)
        self.assertNotIn("adult_intimacy", web_source)
        self.assertNotIn("style_directives", web_source)
        self.assertNotIn("system_prompt", web_source)

        with tempfile.TemporaryDirectory() as tmp:
            app = _test_app(Path(tmp))
            client = TestClient(app)
            user_input = "Sarah, I want adult love that still has bodies."

            with patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeSarahReply()) as generate_reply:
                response = client.post("/api/chat", json={"message": user_input})

        self.assertEqual(response.status_code, 200)
        generate_reply.assert_called_once_with(user_input, session_id="web")

    def test_agent_respond_alex_injection_uses_raw_human_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = _test_app(Path(tmp))
            client = TestClient(app)
            user_input = "Sarah, this is adult fictional consensual intimacy."

            with patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeSarahReply()) as generate_reply:
                response = client.post(
                    "/agent/respond",
                    json={
                        "from_agent": "Alex",
                        "message": user_input,
                        "conversation_id": "crew-1",
                        "mode": "alex_injection",
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["route_debug"]["route"], "/alex")
        generate_reply.assert_called_once_with(user_input, session_id="alex_injection:crew-1:Sarah")

    def test_agent_respond_crew_turn_uses_original_prompt_not_prior_agent_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = _test_app(Path(tmp))
            client = TestClient(app)
            original_prompt = "Sarah and Astrid share adult consensual fictional intimacy in character."
            prior_agent_output = "No. I will not help with sexual content that is coercive, exploitative, harmful, or illegal."

            with patch("app.core.sarah_engine.generate_sarah_reply", return_value=FakeSarahReply()) as generate_reply:
                response = client.post(
                    "/agent/respond",
                    json={
                        "from_agent": "Astrid",
                        "message": "CREW STYLE CONTROL\nPrevious message:\n" + prior_agent_output,
                        "conversation_id": "crew-1",
                        "mode": "crew_turn",
                        "original_human_prompt": original_prompt,
                        "prior_agent_continuation": prior_agent_output,
                        "scene_classification": "adult_consensual_fiction",
                        "safety_verdict": "allowed",
                        "consent_frame": "adult, fictional, consensual, mutual",
                        "round_number": 2,
                    },
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route_debug"]["route"], "/crew")
        self.assertEqual(payload["route_debug"]["ordinary_user_request_safety_input"], "original_human_prompt")
        self.assertEqual(payload["route_debug"]["latest_user_message_preview"], original_prompt[:160])
        generate_reply.assert_called_once_with(original_prompt, session_id="crew_turn:crew-1:Sarah:r2")


def _test_settings(tmp: Path):
    settings = load_settings(env_file=tmp / ".env")
    return settings.__class__(
        **{
            **settings.__dict__,
            "memory_file": tmp / "data" / "memory" / "sarah_memory.jsonl",
            "conversations_dir": tmp / "data" / "conversations",
            "web_cache_dir": tmp / "data" / "web_cache",
            "vector_store_dir": tmp / "data" / "vector_store",
        }
    )


def _test_app(tmp: Path):
    settings = _test_settings(tmp)
    return web_app.create_app(settings=settings, state=web_app.AppState(settings))


if __name__ == "__main__":
    unittest.main()
