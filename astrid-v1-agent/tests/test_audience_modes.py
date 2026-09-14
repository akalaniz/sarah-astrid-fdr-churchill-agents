import pytest

from app.core.audience_mode import audience_directives
from app.core import sarah_engine
from app.core.conversation import ConversationTurn
from app.core.prompt_builder import build_prompt
from test_response_modes import app_fixture, final_payload


@pytest.mark.parametrize("mode", [None, "informal"])
def test_informal_adds_no_instruction(mode):
    assert audience_directives(mode) == ""


@pytest.mark.parametrize("mode", ["FORMAL", "quick", "", "strict"])
def test_invalid_internal_audience_rejected(mode):
    with pytest.raises(ValueError, match="formal or informal"):
        audience_directives(mode)


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
@pytest.mark.parametrize("audience", ["formal", "informal"])
@pytest.mark.parametrize("depth", ["quick", "deep"])
def test_independent_modes_reach_model_before_streaming(app_fixture, endpoint, audience, depth):
    browser, calls, retrieval = app_fixture
    response = browser.post(endpoint, json={
        "message": "Hello.", "audience_mode": audience, "response_mode": depth,
    })
    assert response.status_code == 200
    payload = final_payload(response)
    assert payload["audience_mode"] == audience
    assert response.headers["X-Audience-Mode"] == audience
    assert payload["response_mode"]["effective"] == depth
    assert calls[-1]["reasoning"] == {"effort": "low" if depth == "quick" else "high"}
    assert len(calls) == 1
    prompt = calls[-1]["input"]
    layers = [m for m in prompt if m["content"].startswith("CURRENT REPLY AUDIENCE:")]
    assert len(layers) == (1 if audience == "formal" else 0)
    if layers:
        assert layers[0]["role"] == "system"
        for term in ["profanity", "flirting", "pet names", "affection", "first streamed",
                     "physics rigor", "Quick/Deep", "memories", "paraphrases"]:
            assert term in layers[0]["content"]
    assert prompt[-1] == {"role": "user", "content": "Hello."}
    assert any("IMMUTABLE IDENTITY" in m["content"] for m in prompt)
    assert any("Retained memory marker." in m["content"] for m in prompt)
    retrieval.assert_called_once_with("Hello.")
    assert sarah_engine.get_sarah_session_turn_count("web") == 1
    if endpoint.endswith("stream"):
        assert "event: text" in response.text


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
def test_audience_alone_does_not_change_depth_options(app_fixture, endpoint):
    browser, calls, _ = app_fixture
    response = browser.post(endpoint, json={"message": "Hello.", "audience_mode": "formal"})
    assert "response_mode" not in final_payload(response)
    assert "reasoning" not in calls[-1]
    assert "text" not in calls[-1]


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
def test_default_and_explicit_informal_have_identical_prompts(app_fixture, endpoint):
    browser, calls, _ = app_fixture
    response = browser.post(endpoint, json={"message": "Hello."})
    baseline = calls[-1]
    assert "audience_mode" not in final_payload(response)
    assert "X-Audience-Mode" not in response.headers
    browser.post("/api/reset")
    browser.post(endpoint, json={"message": "Hello.", "audience_mode": "informal"})
    assert calls[-1] == baseline


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
@pytest.mark.parametrize("mode", ["deep", "FORMAL", "invalid"])
def test_invalid_http_audience_never_calls_model(app_fixture, endpoint, mode):
    browser, calls, retrieval = app_fixture
    assert browser.post(endpoint, json={"message": "Hello.", "audience_mode": mode}).status_code == 422
    assert calls == []
    retrieval.assert_not_called()


@pytest.mark.parametrize("audience", ["formal", "informal"])
def test_audience_keeps_physics_validation_gate(app_fixture, audience):
    browser, calls, _ = app_fixture
    response = browser.post("/api/chat/stream", json={
        "message": "Invent a new theory of quantum gravity.",
        "response_mode": "quick", "audience_mode": audience,
    })
    payload = final_payload(response)
    assert payload["response_mode"]["effective"] == "deep"
    assert calls[-1]["reasoning"] == {"effort": "high"}
    assert not calls[-1].get("stream")
    assert "event: text" not in response.text
    assert "First complete sentence" not in response.text
    assert "I would stop here" in payload["text"]


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
def test_formal_pdf_keeps_contents_and_retrieval(app_fixture, endpoint):
    browser, calls, retrieval = app_fixture
    response = browser.post(endpoint, json={
        "message": "What do you think?", "response_mode": "quick", "audience_mode": "formal",
        "temporary_pdf_name": "sample.pdf", "temporary_pdf_text": "Unmodified document passage.",
    })
    assert final_payload(response)["response_mode"]["effective"] == "deep"
    assert any("Unmodified document passage." in m["content"] for m in calls[-1]["input"])
    retrieval.assert_called_once_with("What do you think?")


def test_switching_audience_does_not_replace_history_or_leak_to_next_turn(app_fixture):
    browser, calls, _ = app_fixture
    for mode in ["informal", "formal", "informal"]:
        browser.post("/api/chat/stream", json={"message": "Hello.", "audience_mode": mode})
    assert sarah_engine.get_sarah_session_turn_count("web") == 3
    assert any(m["role"] == "assistant" for m in calls[-1]["input"])
    assert not any(m["content"].startswith("CURRENT REPLY AUDIENCE:") for m in calls[-1]["input"])


def test_formal_layer_is_budgeted_without_modifying_other_layers_or_history():
    kwargs = dict(
        user_message="Discuss the result.", style_directives="Existing style.",
        memory_context="Original memory.", retrieved_context="Original source.",
        web_context="", history=[ConversationTurn("Earlier greeting.", "Earlier affectionate reply.")],
        model="gpt-5.6",
    )
    original = build_prompt(**kwargs)
    formal = build_prompt(**kwargs, audience_directives=audience_directives("formal"))
    assert [m for m in formal.messages if not m["content"].startswith("CURRENT REPLY AUDIENCE:")] == original.messages
    assert formal.debug_summary["estimated_input_tokens"] > original.debug_summary["estimated_input_tokens"]
    assert formal.debug_summary["estimated_input_tokens"] < formal.debug_summary["input_budget_tokens"]
    assert any(layer["name"] == "audience" for layer in formal.debug_summary["layers"])

