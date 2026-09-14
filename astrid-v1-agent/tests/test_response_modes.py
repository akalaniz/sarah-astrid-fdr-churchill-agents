from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from app.core.config import load_settings
from app.core.conversation import ConversationTurn
from app.core.openai_chat import SarahOpenAIClient
from app.core.response_mode import select_response_mode
from app.core import sarah_engine, sarah_response
from app.tools.web_router import WebRetrievalResult
from app.ui.web_app import create_app


@pytest.mark.parametrize("message", [
    "Hello Sarah.", "How are you today?", "Thanks!", "Tell me a short joke.",
    "I had a long day.", "What should I cook tonight?", "Good morning, Astrid.",
])
def test_ordinary_quick_conversation(message):
    mode = select_response_mode("quick", message)
    assert mode.effective == "quick"
    assert mode.api_options("gpt-5.6") == {"reasoning_effort": "low", "text_verbosity": "low"}


@pytest.mark.parametrize("message", [
    "Critique this physics theory.", "Check the Dirac equation.",
    "Derive the mass-energy relation.", "Explain quantum mechanics.",
    "Does my proposed theory conserve momentum?", "Is this Lagrangian stable?",
    "Review Book 1 and Book 2.", "Analyze the manuscript's structure.",
    "Check continuity across the trilogy.", "What is missing from chapter 3?",
    "Find plot holes and character arcs.", "Evaluate this argument.",
    "Here is the math: \\[E=mc^2\\]", "Compare the Hamiltonian with Maxwell's equations.",
    "Criticism of my speculative model.", "x" * 2000,
])
def test_quick_promotes_substantial_work(message):
    mode = select_response_mode("quick", message)
    assert mode.effective == "deep"
    assert mode.api_options("gpt-5.6") == {"reasoning_effort": "high", "text_verbosity": "medium"}
    assert "120 words" not in mode.prompt()


@pytest.mark.parametrize("followup", [
    "Continue.", "Why?", "What about the second one?", "That doesn't follow.",
    "Are you sure?", "Now expand the argument.", "Yes, proceed.",
    "Could you expand on that?", "Does that hold in the second case?",
    "What happens if we double it?",
])
def test_short_analysis_followups_stay_deep(followup):
    history = [ConversationTurn(user="Review my manuscript.", assistant="Here are the findings.")]
    assert select_response_mode("quick", followup, history).effective == "deep"


def test_fresh_small_talk_after_analysis_can_be_quick():
    history = [ConversationTurn(user="Review the physics.", assistant="We found a contradiction.")]
    assert select_response_mode("quick", "Good morning!", history).effective == "quick"


def test_explicit_deep_is_available_for_any_reply():
    assert select_response_mode("deep", "Hi").effective == "deep"


def test_attachment_and_validation_cannot_be_downgraded():
    assert select_response_mode("quick", "Hi", has_attachment=True).effective == "deep"
    assert select_response_mode("quick", "Hi", protected_physics=True).effective == "deep"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="quick or deep"):
        select_response_mode("fastest", "Hi")


@pytest.mark.parametrize("model", ["gpt-4.1", "custom-model", "gpt-5-pro", "gpt-5-chat-latest"])
def test_unverified_models_keep_api_defaults(model):
    assert select_response_mode("quick", "Hi").api_options(model) == {}


class FakeStream:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __iter__(self):
        yield SimpleNamespace(type="response.output_text.delta", delta=self.text)
        yield SimpleNamespace(type="response.completed", response=SimpleNamespace(output_text=self.text))


@pytest.fixture
def app_fixture(tmp_path, monkeypatch):
    settings = replace(
        load_settings(env_file=tmp_path / ".env"), sarah_model="gpt-5.6",
        memory_file=tmp_path / "memory.jsonl",
        conversations_dir=tmp_path / "conversations",
        web_cache_dir=tmp_path / "web_cache",
    )
    app = create_app(settings=settings)
    calls = []

    def create(**request):
        calls.append(request)
        text = "First complete sentence. The answer is complete."
        return FakeStream(text) if request.get("stream") else SimpleNamespace(output_text=text)

    client = SarahOpenAIClient.__new__(SarahOpenAIClient)
    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    sarah_engine.configure_sarah_engine(settings=settings, client=client)
    retrieval = Mock(return_value=(None, "No test documents"))
    monkeypatch.setattr(sarah_response, "_retrieve_for_message", retrieval)
    monkeypatch.setattr(sarah_response, "build_memory_context", lambda _: "Retained memory marker.")
    monkeypatch.setattr(sarah_response, "build_agent_message_context", lambda _: "")
    monkeypatch.setattr(sarah_response, "retrieve_web_context",
                        lambda *_: WebRetrievalResult(False, False, "not needed", [], ""))
    with TestClient(app) as browser:
        yield browser, calls, retrieval
    sarah_engine.configure_sarah_engine()


def final_payload(response):
    if "text/event-stream" not in response.headers.get("content-type", ""):
        return response.json()
    frames = [frame for frame in response.text.split("\n\n") if frame.startswith("event: done")]
    assert len(frames) == 1, response.text
    return json.loads(frames[0].split("data: ", 1)[1])


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
@pytest.mark.parametrize("requested,message,effective", [
    ("quick", "Hello Alex.", "quick"),
    ("deep", "Hello Alex.", "deep"),
    ("quick", "Critique the physics in my manuscript.", "deep"),
])
def test_browser_mode_reaches_real_engine_and_sdk(app_fixture, endpoint, requested, message, effective):
    browser, calls, retrieval = app_fixture
    response = browser.post(endpoint, json={"message": message, "response_mode": requested})
    assert response.status_code == 200
    payload = final_payload(response)
    assert payload["response_mode"]["requested"] == requested
    assert payload["response_mode"]["effective"] == effective
    assert calls[-1]["model"] == "gpt-5.6"
    assert calls[-1]["reasoning"] == {"effort": "low" if effective == "quick" else "high"}
    assert calls[-1]["text"] == {"verbosity": "low" if effective == "quick" else "medium"}
    assert "max_output_tokens" not in calls[-1]
    prompt = "\n".join(item["content"] for item in calls[-1]["input"])
    assert "Retained memory marker." in prompt
    assert "IMMUTABLE IDENTITY" in prompt
    assert f"RESPONSE DEPTH: {effective.upper()}" in prompt
    retrieval.assert_called_once_with(message)
    assert sarah_engine.get_sarah_session_turn_count("web") == 1


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
def test_omitted_mode_preserves_legacy_request_options(app_fixture, endpoint):
    browser, calls, _ = app_fixture
    response = browser.post(endpoint, json={"message": "Hello."})
    assert response.status_code == 200
    assert "reasoning" not in calls[-1]
    assert "text" not in calls[-1]
    assert "response_mode" not in final_payload(response)


def test_invalid_http_mode_does_not_reach_model(app_fixture):
    browser, calls, _ = app_fixture
    response = browser.post("/api/chat/stream", json={"message": "Hi", "response_mode": "turbo"})
    assert response.status_code == 422
    assert calls == []


def test_quick_physics_validation_still_holds_all_draft_text(app_fixture):
    browser, calls, _ = app_fixture
    response = browser.post("/api/chat/stream", json={
        "message": "Invent a new theory of quantum gravity.", "response_mode": "quick",
    })
    payload = final_payload(response)
    assert calls[-1]["reasoning"] == {"effort": "high"}
    assert not calls[-1].get("stream")
    assert "event: text" not in response.text
    assert "First complete sentence" not in response.text
    assert "I would stop here" in payload["text"]
    assert payload["response_mode"]["effective"] == "deep"


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
def test_quick_attachment_keeps_pdf_and_retrieval_question_and_promotes(app_fixture, endpoint):
    browser, calls, retrieval = app_fixture
    response = browser.post(endpoint, json={
        "message": "What do you think?", "response_mode": "quick",
        "temporary_pdf_name": "sample.pdf", "temporary_pdf_text": "Unique attached manuscript text.",
    })
    assert final_payload(response)["response_mode"]["effective"] == "deep"
    prompt = "\n".join(item["content"] for item in calls[-1]["input"])
    assert "Unique attached manuscript text." in prompt
    retrieval.assert_called_once_with("What do you think?")


def test_followup_and_mode_changes_use_same_conversation(app_fixture):
    browser, calls, _ = app_fixture
    final_payload(browser.post("/api/chat/stream", json={"message": "Review chapter one.", "response_mode": "quick"}))
    payload = final_payload(browser.post("/api/chat/stream", json={"message": "What about the second one?", "response_mode": "quick"}))
    assert payload["response_mode"]["reason"] == "Analysis follow-up"
    payload = final_payload(browser.post("/api/chat/stream", json={"message": "Hello again.", "response_mode": "quick"}))
    assert payload["response_mode"]["effective"] == "quick"
    assert sarah_engine.get_sarah_session_turn_count("web") == 3
    prompt = "\n".join(item["content"] for item in calls[-1]["input"])
    assert "Review chapter one." in prompt


@pytest.mark.parametrize("streaming", [False, True])
def test_legacy_chat_sdk_receives_explicit_mode_controls(streaming):
    calls = []

    def create(**request):
        calls.append(request)
        choice = SimpleNamespace(delta=SimpleNamespace(content="Hello.", refusal=None), finish_reason="stop")
        if streaming:
            class Stream:
                def __enter__(self): return iter([SimpleNamespace(choices=[choice])])
                def __exit__(self, *_): pass
            return Stream()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Hello."))])

    client = SarahOpenAIClient.__new__(SarahOpenAIClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    messages = [{"role": "user", "content": "Hi"}]
    if streaming:
        client.stream_response("gpt-5.6", messages, lambda _: None, reasoning_effort="low", text_verbosity="low")
    else:
        client.create_response("gpt-5.6", messages, reasoning_effort="low", text_verbosity="low")
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["verbosity"] == "low"
