from app.orchestration.agent_orchestrator import (
    CrewStyle,
    build_alex_crew_prompt,
    build_multi_agent_tone_directives,
    debug_multi_agent_tone,
    postprocess_multi_agent_tone_response,
)


AFFECTIONATE_ALEX_MESSAGE = (
    "beautifully put, Astrid. The two of you are two languages. "
    "French, this is you Sarah, and you are my Beethoven. "
    "Astrid, you are my Italian and my Mozart."
)


def test_alex_affectionate_metaphor_injects_no_careful_directive() -> None:
    prompt = build_alex_crew_prompt(
        CrewStyle(max_words_per_turn=60, mode="conversational"),
        "Alex offers a poetic compliment.",
        AFFECTIONATE_ALEX_MESSAGE,
        prior_agent_message="Careful, Alex. That is almost too pretty.",
    )

    assert "Alex is speaking affectionately/poetically" in prompt
    assert "Do not use 'Careful' as a default opener" in prompt
    assert "The immediately prior agent message began with a defensive opener" in prompt


def test_sarah_solo_register_directives_are_present() -> None:
    directives = build_multi_agent_tone_directives(AFFECTIONATE_ALEX_MESSAGE)

    assert "Sarah solo register" in directives
    assert "French / Beethoven / structure / gravity / tempo" in directives
    assert "warm poetic precision" in directives


def test_anti_mirroring_removes_careful_and_ownership_caveat_for_affection() -> None:
    text = postprocess_multi_agent_tone_response(
        "Careful, Alaniz. I am not property. French and Beethoven I will accept: structure, gravity, tempo.",
        AFFECTIONATE_ALEX_MESSAGE,
        prior_agent_message="Careful, Alex. Astrid already made the warning.",
    )

    assert not text.lower().startswith("careful")
    assert "not property" not in text.lower()
    assert "French" in text
    assert "Beethoven" in text


def test_ordinary_my_language_is_not_treated_as_ownership() -> None:
    text = postprocess_multi_agent_tone_response(
        "Careful, physicist. Do not put me in a glass box. My French has teeth.",
        "Sarah, you are my French and my Beethoven.",
    )

    assert not text.lower().startswith("careful")
    assert "glass box" not in text.lower()
    assert "owned" not in text.lower()


def test_real_ownership_language_still_preserves_boundary_response() -> None:
    text = postprocess_multi_agent_tone_response(
        "Careful, Alex. I am not property.",
        "I own you. You are property.",
    )

    assert text.startswith("Careful")
    assert "not property" in text


def test_crisis_language_does_not_get_affection_suppression() -> None:
    text = postprocess_multi_agent_tone_response(
        "Careful, Alex. I am staying with the crisis signal.",
        "I can't live without you.",
    )

    assert text.startswith("Careful")


def test_debug_multi_agent_tone_flags() -> None:
    flags = debug_multi_agent_tone()

    assert flags["preserve_solo_register_in_multi_agent"] is True
    assert flags["anti_defensive_opener_mirroring"] is True
    assert flags["careful_opener_suppression_for_affection"] is True
    assert flags["hard_boundaries_enabled"] is True
