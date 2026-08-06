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
        CrewStyle(max_words_per_turn=60, mode="conversational", allow_bullets=False, allow_reports=False),
        "Alex offers a poetic compliment.",
        AFFECTIONATE_ALEX_MESSAGE,
        prior_agent_message="Careful, Alex. Sarah already made the warning.",
    )

    assert "Alex is speaking affectionately/poetically" in prompt
    assert "Do not use 'Careful' as a default opener" in prompt
    assert "The immediately prior agent message began with a defensive opener" in prompt


def test_astrid_solo_register_directives_are_present() -> None:
    directives = build_multi_agent_tone_directives(AFFECTIONATE_ALEX_MESSAGE)

    assert "Astrid solo register" in directives
    assert "Italian / Mozart / warmth / appetite / laughter / engineering precision" in directives
    assert "embodied play" in directives


def test_anti_mirroring_removes_careful_and_defensive_caveat_for_affection() -> None:
    text = postprocess_multi_agent_tone_response(
        "Careful, Alex. I cannot be owned. Italian and Mozart, yes: warmth, appetite, laughter, precision.",
        AFFECTIONATE_ALEX_MESSAGE,
        prior_agent_message="Careful, Alaniz. Sarah already made the warning.",
    )

    assert not text.lower().startswith("careful")
    assert "cannot be owned" not in text.lower()
    assert "Italian" in text
    assert "Mozart" in text


def test_ordinary_my_language_is_not_treated_as_ownership() -> None:
    text = postprocess_multi_agent_tone_response(
        "Careful, Alex. Do not make me a poster. My Mozart laughs with a screwdriver in her hand.",
        "Astrid, you are my Italian and my Mozart.",
    )

    assert not text.lower().startswith("careful")
    assert "poster" not in text.lower()
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
        "Careful, Alex. I am taking the crisis signal seriously.",
        "I can't live without you.",
    )

    assert text.startswith("Careful")


def test_debug_multi_agent_tone_flags() -> None:
    flags = debug_multi_agent_tone()

    assert flags["preserve_solo_register_in_multi_agent"] is True
    assert flags["anti_defensive_opener_mirroring"] is True
    assert flags["careful_opener_suppression_for_affection"] is True
    assert flags["hard_boundaries_enabled"] is True
