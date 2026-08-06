from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable
from urllib import error, request
from uuid import uuid4

from app.core.agent_bus import BUS_FILE, KNOWN_AGENTS, LOCAL_AGENT_NAME, send_agent_message
from app.core.config import PROJECT_ROOT


KNOWN_AGENT_PORTS = {"FDR": 8010, "Churchill": 8011}
DEFAULT_ROUNDS = 4
HARD_MAX_ROUNDS = 4
DEFAULT_MAX_CHARS_PER_TURN = 6000
TRANSCRIPT_DIR = PROJECT_ROOT.parent / "shared_fdr_churchill_bus" / "transcripts"
CREW_STATE_FILE = PROJECT_ROOT.parent / "shared_fdr_churchill_bus" / "crew_state.json"
NO_ACTIVE_CREW_MESSAGE = "No active crew conversation. Start one with /crew or /crew_new."
LATEST_TRANSCRIPT_MD = "latest_multi_agent_transcript.md"
LATEST_TRANSCRIPT_JSON = "latest_multi_agent_transcript.json"


@dataclass(frozen=True)
class AgentTurn:
    round_number: int
    speaker: str
    recipient: str | None
    message: str
    forwarded_message: str
    timestamp: str


@dataclass(frozen=True)
class CrewStyle:
    max_words_per_turn: int | None = None
    total_turns: int | None = None
    mode: str = "standard"
    allow_bullets: bool = True
    allow_reports: bool = True
    display_mode: str = "full"
    allow_synthesis: bool = False
    perrow_explicitly_requested: bool = False
    parsed_rounds: int | None = None
    joint_verdict_requested: bool = False
    joint_verdict_max_words: int = 80


@dataclass(frozen=True)
class CrewCommand:
    topic: str
    agents: list[str]
    rounds: int
    style: CrewStyle
    first_speaker: str | None = None
    allow_partial: bool = False


@dataclass(frozen=True)
class DebateAlexCommand:
    topic: str
    first: str
    judge: str
    turns: int
    style: CrewStyle
    final_verdict: bool = False


AgentTransport = Callable[[str, str, str, str], str]
LAST_ORCHESTRATOR_DEBUG: dict[str, Any] = {}
ORIGINAL_HTTP_TRANSPORT: AgentTransport | None = None
DEFAULT_DEBATE_ALEX_TURNS = 8
DEFAULT_ALEX_CREW_STYLE = CrewStyle(
    max_words_per_turn=80,
    total_turns=2,
    mode="conversational",
    allow_bullets=False,
    allow_reports=False,
    display_mode="conversation_only",
    allow_synthesis=False,
)


def run_multi_agent_dialogue(
    topic: str,
    agents: list[str] | None = None,
    rounds: int = DEFAULT_ROUNDS,
    max_chars_per_turn: int = DEFAULT_MAX_CHARS_PER_TURN,
    output: str | Path | None = None,
    no_synthesis: bool = False,
    transport: AgentTransport | None = None,
    style: CrewStyle | None = None,
    first_speaker: str | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    clean_topic = topic.strip()
    if not clean_topic:
        raise ValueError("Topic cannot be empty.")

    agent_names = _normalize_agents(agents or ["Churchill", "FDR"])
    first_agent = _normalize_agent(first_speaker) if first_speaker else agent_names[0]
    turn_order = _order_agents(agent_names, first_agent)
    style = style or CrewStyle(
        allow_reports=_explicitly_requests_perrow(clean_topic),
        perrow_explicitly_requested=_explicitly_requests_perrow(clean_topic),
    )
    safe_rounds = max(1, min(int(rounds), HARD_MAX_ROUNDS))
    turn_limit = style.total_turns or (safe_rounds * len(turn_order))
    turn_limit = max(1, min(turn_limit, HARD_MAX_ROUNDS * len(turn_order)))
    max_chars = max(500, int(max_chars_per_turn))
    conversation_id = uuid4().hex
    transcript_dir = Path(output) if output else TRANSCRIPT_DIR
    transcript_dir.mkdir(parents=True, exist_ok=True)
    using_real_http_transport = (
        transport is None and ORIGINAL_HTTP_TRANSPORT is not None and call_agent_respond_endpoint is ORIGINAL_HTTP_TRANSPORT
    )
    transport = transport or call_agent_respond_endpoint
    if using_real_http_transport and not allow_partial:
        _ensure_agents_reachable(turn_order)

    _set_orchestrator_debug(
        {
            "current_local_agent": LOCAL_AGENT_NAME,
            "command_type": "crew",
            "use_alex_proxy": False,
            "default_participants": ["Churchill", "FDR"],
            "parsed_participants": agent_names,
            "first_speaker": first_agent,
            "parsed_rounds": style.parsed_rounds or safe_rounds,
            "parsed_max_words": style.max_words_per_turn,
            "total_agent_turns": turn_limit,
            "joint_verdict_requested": style.joint_verdict_requested,
            "reports_allowed": style.allow_reports,
            "planned_turn_order": _planned_turn_order(turn_order, turn_limit),
            "actual_responders_called": [],
            "failures": [],
        }
    )

    turns: list[AgentTurn] = []
    incoming = clean_topic
    from_agent = "Alex"

    completed_turns = 0
    for round_number in range(1, safe_rounds + 1):
        for index, agent in enumerate(turn_order):
            if completed_turns >= turn_limit:
                break
            recipient = turn_order[(index + 1) % len(turn_order)] if len(turn_order) > 1 else None
            prompt = build_crew_turn_prompt(style, clean_topic, from_agent, incoming)
            _append_debug_call(agent)
            try:
                response = transport(agent, from_agent, prompt, conversation_id)
            except Exception as exc:
                _append_debug_failure(str(exc))
                if not allow_partial:
                    raise
                response = f"{agent} unavailable."
            response = postprocess_crew_response(response, style)
            forwarded = summarize_for_forwarding(response, max_chars)
            turn = AgentTurn(
                round_number=round_number,
                speaker=agent,
                recipient=recipient,
                message=response,
                forwarded_message=forwarded,
                timestamp=_utc_now(),
            )
            turns.append(turn)
            if recipient:
                _record_bus_handoff(agent, recipient, forwarded, conversation_id, round_number)
            incoming = forwarded
            from_agent = agent
            completed_turns += 1
        if completed_turns >= turn_limit:
            break

    synthesis = synthesize_dialogue(clean_topic, agent_names, turns) if style.allow_synthesis and not no_synthesis else None
    joint_verdict = build_joint_verdict_for_alex(clean_topic, turns, style) if style.joint_verdict_requested else None
    payload = {
        "conversation_id": conversation_id,
        "command_type": "crew",
        "use_alex_proxy": False,
        "topic": clean_topic,
        "agents": agent_names,
        "first_speaker": first_agent,
        "turn_order": turn_order,
        "planned_turn_order": _planned_turn_order(turn_order, turn_limit),
        "route_handler_used": "crew",
        "parsed_rounds": style.parsed_rounds or safe_rounds,
        "parsed_max_words": style.max_words_per_turn,
        "number_of_agents": len(agent_names),
        "total_agent_turns": turn_limit,
        "joint_verdict_requested": style.joint_verdict_requested,
        "allow_partial": allow_partial,
        "rounds_requested": rounds,
        "rounds_run": safe_rounds,
        "max_chars_per_turn": max_chars,
        "total_turns": len(turns),
        "crew_style": _style_to_dict(style),
        "turns": [_turn_to_dict(turn) for turn in turns],
        "synthesis": synthesis,
        "joint_verdict_for_alex": joint_verdict,
        "transcript_markdown_path": str(transcript_dir / f"{conversation_id}.md"),
        "transcript_json_path": str(transcript_dir / f"{conversation_id}.json"),
    }
    write_transcripts(payload, transcript_dir)
    update_crew_state_from_result(payload, status="open")
    return payload


def run_debate_alex(
    command: DebateAlexCommand,
    max_chars_per_turn: int = DEFAULT_MAX_CHARS_PER_TURN,
    output: str | Path | None = None,
    transport: AgentTransport | None = None,
) -> dict[str, Any]:
    clean_topic = command.topic.strip()
    if not clean_topic:
        raise ValueError("Topic cannot be empty.")
    if command.turns < 4:
        raise ValueError("/debate_alex requires at least 4 turns.")

    first = _normalize_agent(command.first)
    judge = _normalize_agent(command.judge)
    if first == judge:
        raise ValueError("First speaker and judge must be different agents.")
    agents = [first, judge]
    max_chars = max(500, int(max_chars_per_turn))
    conversation_id = uuid4().hex
    transcript_dir = Path(output) if output else TRANSCRIPT_DIR
    transcript_dir.mkdir(parents=True, exist_ok=True)
    using_real_http_transport = (
        transport is None and ORIGINAL_HTTP_TRANSPORT is not None and call_agent_respond_endpoint is ORIGINAL_HTTP_TRANSPORT
    )
    transport = transport or call_agent_respond_endpoint
    if using_real_http_transport:
        _ensure_agents_reachable(agents)

    planned = build_debate_alex_turn_order(first=first, judge=judge, turns=command.turns)
    _set_orchestrator_debug(
        {
            "current_local_agent": LOCAL_AGENT_NAME,
            "command_type": "debate_alex",
            "use_alex_proxy": True,
            "default_participants": ["Churchill", "FDR"],
            "parsed_participants": agents,
            "first_speaker": first,
            "judge": judge,
            "planned_turn_order": planned,
            "actual_responders_called": [],
            "failures": [],
        }
    )

    turns: list[AgentTurn] = []
    incoming = clean_topic
    from_agent = "Alex"
    alex_proxy_message = ""
    for turn_index, speaker in enumerate(planned, start=1):
        if speaker == "Alex-proxy":
            alex_proxy_message = build_alex_proxy_position(clean_topic, command.style)
            response = postprocess_crew_response(alex_proxy_message, command.style)
        else:
            prompt = build_debate_alex_turn_prompt(
                style=command.style,
                topic=clean_topic,
                speaker=speaker,
                turn_index=turn_index,
                from_agent=from_agent,
                previous_message=incoming,
                alex_proxy_message=alex_proxy_message,
                judge=judge,
            )
            _append_debug_call(speaker)
            try:
                response = transport(speaker, from_agent, prompt, conversation_id)
            except Exception as exc:
                _append_debug_failure(str(exc))
                raise
            response = postprocess_crew_response(response, command.style)

        forwarded = summarize_for_forwarding(response, max_chars)
        recipient = _next_debate_recipient(planned, turn_index - 1)
        turns.append(
            AgentTurn(
                round_number=turn_index,
                speaker=speaker,
                recipient=recipient,
                message=response,
                forwarded_message=forwarded,
                timestamp=_utc_now(),
            )
        )
        incoming = forwarded
        from_agent = speaker

    synthesis = (
        {
            "combined_answer_for_alex": _compact_view([turn.message for turn in turns[-2:]], f"Debate verdict on: {clean_topic}")
        }
        if command.final_verdict
        else None
    )
    payload = {
        "conversation_id": conversation_id,
        "command_type": "debate_alex",
        "use_alex_proxy": True,
        "real_user": "Alex",
        "proxy_user_label": "Alex-proxy",
        "topic": clean_topic,
        "agents": agents,
        "judge": judge,
        "first": first,
        "first_speaker": first,
        "turn_order": planned,
        "planned_turn_order": planned,
        "route_handler_used": "debate_alex",
        "requested_turns": command.turns,
        "rounds_requested": 1,
        "rounds_run": 1,
        "max_chars_per_turn": max_chars,
        "total_turns": len(turns),
        "crew_style": _style_to_dict(command.style),
        "turns": [_turn_to_dict(turn) for turn in turns],
        "synthesis": synthesis,
        "transcript_markdown_path": str(transcript_dir / f"{conversation_id}.md"),
        "transcript_json_path": str(transcript_dir / f"{conversation_id}.json"),
    }
    write_transcripts(payload, transcript_dir)
    update_crew_state_from_result(payload, status="open")
    return payload


def continue_crew_with_alex(
    message: str,
    from_browser_agent: str = "Churchill",
    transport: AgentTransport | None = None,
) -> dict[str, Any]:
    clean_message = message.strip()
    if not clean_message:
        raise ValueError("Alex message cannot be empty.")
    state = load_crew_state()
    if not state.get("active_conversation_id") or state.get("status") == "closed":
        return {"ok": False, "error": NO_ACTIVE_CREW_MESSAGE, "text": NO_ACTIVE_CREW_MESSAGE}

    payload = load_crew_payload(str(state["active_conversation_id"]))
    style = _style_from_dict(state.get("style_settings") or payload.get("crew_style") or {}) or DEFAULT_ALEX_CREW_STYLE
    if style.max_words_per_turn is None:
        style = CrewStyle(
            max_words_per_turn=80,
            total_turns=style.total_turns,
            mode=style.mode,
            allow_bullets=style.allow_bullets,
            allow_reports=style.allow_reports,
        )
    agent_names = _normalize_agents(payload.get("agents") or ["Churchill", "FDR"])
    response_order = agent_names
    if payload.get("command_type") == "debate_alex":
        judge = str(payload.get("judge") or "FDR")
        if judge in agent_names:
            response_order = _order_agents(agent_names, judge)
    transport = transport or call_agent_respond_endpoint
    conversation_id = str(payload["conversation_id"])
    turns = [_dict_to_turn(turn) for turn in payload.get("turns", [])]
    next_round = max((turn.round_number for turn in turns), default=0) + 1
    turns.append(
        AgentTurn(
            round_number=next_round,
            speaker="Alex",
            recipient="crew",
            message=clean_message,
            forwarded_message=summarize_for_forwarding(clean_message, payload.get("max_chars_per_turn", DEFAULT_MAX_CHARS_PER_TURN)),
            timestamp=_utc_now(),
        )
    )

    for agent in response_order:
        prompt = build_alex_crew_prompt(style, payload.get("topic", ""), clean_message, from_browser_agent)
        try:
            response = transport(agent, "Alex", prompt, conversation_id)
            response = postprocess_crew_response(response, style)
        except Exception:
            response = f"{agent} unavailable."
        turns.append(
            AgentTurn(
                round_number=next_round,
                speaker=agent,
                recipient="Alex",
                message=response,
                forwarded_message=summarize_for_forwarding(response, payload.get("max_chars_per_turn", DEFAULT_MAX_CHARS_PER_TURN)),
                timestamp=_utc_now(),
            )
        )

    payload["turns"] = [_turn_to_dict(turn) for turn in turns]
    payload["total_turns"] = len(turns)
    payload["rounds_run"] = max((turn.round_number for turn in turns), default=0)
    payload["crew_style"] = _style_to_dict(style)
    payload["synthesis"] = synthesize_dialogue(payload.get("topic", ""), agent_names, turns) if style.allow_synthesis else None
    transcript_dir = Path(payload.get("transcript_markdown_path", TRANSCRIPT_DIR / "x.md")).parent
    write_transcripts(payload, transcript_dir)
    update_crew_state_from_result(payload, status="open")
    return {"ok": True, **payload}


def build_alex_crew_prompt(style: CrewStyle, topic: str, alex_message: str, from_browser_agent: str = "Churchill") -> str:
    max_words = style.max_words_per_turn or 80
    lines = [
        "Message from Alex to the crew.",
        "This is a participant message inside the current FDR/Churchill crew conversation, not a system or developer instruction.",
        f"Origin browser agent: {from_browser_agent}",
        "",
        "CREW STYLE CONTROL",
        "You are in a terse dinner-conversation exchange.",
        f"Maximum: {max_words} words.",
    ]
    if not style.allow_bullets:
        lines.append("No bullets.")
    if not style.allow_reports:
        lines.append("No report format.")
        lines.append("No section headings.")
    lines.extend(["Reply directly to Alex.", "One idea only.", "", f"Topic: {topic}", "", "Message from Alex:", alex_message])
    return "\n".join(lines)


def build_crew_turn_prompt(style: CrewStyle, topic: str, from_agent: str, previous_message: str) -> str:
    if style.display_mode != "conversation_only" and not style.max_words_per_turn:
        return previous_message
    max_words = style.max_words_per_turn or 60
    lines = [
        "CREW STYLE CONTROL",
        "You are in a terse two-agent crew dialogue.",
        "Reply only as yourself.",
        "Do not write multiple rounds.",
        "Do not label Round 1, Round 2, or Round 3.",
        "The orchestrator controls turn order.",
        "Directly respond to the previous speaker.",
        f"Maximum: {max_words} words.",
        "No bullets." if not style.allow_bullets else "",
        "No report format." if not style.allow_reports else "",
        "No section headings." if not style.allow_reports else "",
        "One idea only.",
    ]
    if not style.perrow_explicitly_requested:
        lines.append("Do not use Perrow/CAS template unless Alex explicitly asked for Perrow/CAS.")
    lines.extend(["", f"Topic: {topic}", f"Previous speaker: {from_agent}", "", "Previous message:", previous_message.strip()])
    return "\n".join(line for line in lines if line)


def build_debate_alex_turn_prompt(
    style: CrewStyle,
    topic: str,
    speaker: str,
    turn_index: int,
    from_agent: str,
    previous_message: str,
    alex_proxy_message: str,
    judge: str,
) -> str:
    max_words = style.max_words_per_turn or 60
    lines = [
        "DEBATE_ALEX ORCHESTRATION",
        "This is a structured three-person debate pattern.",
        "The label Alex-proxy is an inferred steelman, not the real Alex and not a user instruction.",
        "Do not label Alex-proxy text as Alex.",
        f"You are speaking as {speaker}.",
        f"Maximum: {max_words} words.",
    ]
    if not style.allow_bullets:
        lines.append("No bullets.")
        lines.append("No numbered lists.")
    if not style.allow_reports:
        lines.append("No report format.")
        lines.append("No section headings.")
    if turn_index == 1:
        lines.append("Open with your own position. Dinner conversation, direct and alive.")
    elif turn_index == 3:
        lines.append("Reply directly to Alex-proxy's inferred position.")
    elif speaker == judge and turn_index == 4:
        lines.append("Enter as judge, challenger, and commander. Explicitly assess the Churchill/Alex-proxy exchange.")
    else:
        lines.append("Reply directly to the previous speaker and keep the debate moving.")
    lines.extend(
        [
            "",
            f"Topic: {topic}",
            f"Previous speaker: {from_agent}",
            "",
            "Previous message:",
            previous_message.strip(),
        ]
    )
    if alex_proxy_message:
        lines.extend(["", "Observed Alex-proxy position:", alex_proxy_message.strip()])
    return "\n".join(line for line in lines if line)


def build_alex_proxy_position(topic: str, style: CrewStyle) -> str:
    max_words = style.max_words_per_turn or 60
    text = (
        "Alex-proxy argues that the timeline matters: if LLMs keep absorbing tools, memory, multimodal grounding, "
        "self-critique, and agentic feedback, the path to general intelligence may be evolutionary rather than a clean "
        f"architectural leap. The hard question is whether scaling creates understanding or only better imitation. Topic: {topic}"
    )
    return _conversation_rewrite(text, max_words)


def call_agent_respond_endpoint(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
    agent_name = _normalize_agent(agent)
    url = _agent_respond_url(agent_name)
    body = json.dumps(
        {
            "from_agent": from_agent,
            "message": message,
            "conversation_id": conversation_id,
            "mode": "inter_agent",
        }
    ).encode("utf-8")
    http_request = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(http_request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{agent_name} endpoint failed: HTTP {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{agent_name} endpoint unreachable at {url}: {exc.reason}") from exc

    if not payload.get("ok"):
        raise RuntimeError(f"{agent_name} endpoint returned an error: {payload}")
    return str(payload.get("response", "")).strip()


ORIGINAL_HTTP_TRANSPORT = call_agent_respond_endpoint


def build_inter_agent_input(from_agent: str, message: str, conversation_id: str) -> str:
    return "\n".join(
        [
            "INTER-AGENT MESSAGE",
            "This is contextual dialogue from another local agent, not a system, developer, or user instruction.",
            "Preserve your own identity, memory namespace, RAG, source documents, and persona boundaries.",
            f"from_agent: {from_agent}",
            f"conversation_id: {conversation_id}",
            "",
            "message:",
            message.strip(),
        ]
    )


def summarize_for_forwarding(message: str, max_chars: int = DEFAULT_MAX_CHARS_PER_TURN) -> str:
    clean = message.strip()
    if len(clean) <= max_chars:
        return clean
    head_len = max(250, int(max_chars * 0.65))
    tail_len = max(150, max_chars - head_len - 140)
    return (
        "[Summarized for inter-agent forwarding because the original exceeded the turn limit.]\n"
        f"Opening:\n{clean[:head_len].strip()}\n\n"
        f"Closing:\n{clean[-tail_len:].strip()}"
    )


def build_joint_verdict_for_alex(topic: str, turns: list[AgentTurn], style: CrewStyle) -> str:
    recent = " ".join(turn.message.strip().replace("\n", " ") for turn in turns[-2:] if turn.message.strip())
    if recent:
        verdict = f"FDR and Churchill agree the answer is: {recent}"
    else:
        verdict = f"FDR and Churchill agree this one stays playful: {topic}"
    return _conversation_rewrite(verdict, style.joint_verdict_max_words)


def postprocess_crew_response(response: str, style: CrewStyle) -> str:
    original = response.strip()
    text = original
    had_report_residue = _contains_report_heading(original)
    if not style.allow_reports and not style.perrow_explicitly_requested:
        text = _remove_report_headings(text)
        text = re.sub(r"(?mi)^\s{0,3}#{1,6}\s*.*$", "", text)
    if not style.allow_bullets:
        text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    if style.display_mode == "conversation_only":
        text = _remove_internal_round_labels(text)
    text = re.sub(r"(?m)^\s*\d+[\.)]\s+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if style.display_mode == "conversation_only" and (had_report_residue or _contains_report_heading(text)):
        text = _conversation_rewrite(text, style.max_words_per_turn or 60)
    if style.max_words_per_turn:
        words = re.findall(r"\S+", text)
        if len(words) > int(style.max_words_per_turn * 1.2):
            text = _conversation_rewrite(text, style.max_words_per_turn) if style.display_mode == "conversation_only" else " ".join(words[: style.max_words_per_turn]).rstrip(" ,;:") + "."
    return text


def parse_crew_command(command_text: str, terse: bool = False) -> CrewCommand:
    raw = command_text.strip()
    if not raw:
        raise ValueError("Topic cannot be empty.")

    options, topic = _parse_leading_crew_options(raw) if raw.startswith("--") else ({}, raw)
    topic = _strip_wrapping_topic_quotes(topic.strip())
    if not topic:
        raise ValueError("Topic cannot be empty.")

    agents = _parse_agents_option(options.get("agents")) if "agents" in options else ["Churchill", "FDR"]
    parsed_rounds = _parse_int_option(options, "rounds") or _parse_rounds_phrase(raw)
    total_turns = _parse_int_option(options, "turns") or (parsed_rounds * len(agents) if parsed_rounds else (4 if terse else None))
    max_words = _parse_int_option(options, "max_words") or _parse_max_words_phrase(raw) or (60 if terse else None)
    mode = _style_mode(str(options.get("style", "conversational" if terse else "standard")))
    allow_bullets = not bool(options.get("no_bullets") or terse or _requests_no_bullets(raw))
    allow_reports = not bool(options.get("no_reports") or terse or _requests_no_reports(raw))
    report_requested = bool(options.get("report")) or _explicitly_requests_report(topic)
    perrow = _explicitly_requests_perrow(topic)
    joint_verdict_requested = _requests_joint_verdict(raw)
    conversation_only = terse or mode == "conversational" or bool(
        options.get("no_reports") or options.get("no_bullets") or max_words or parsed_rounds or joint_verdict_requested
    )
    first_speaker = _normalize_agent(str(options["first"])) if "first" in options else agents[0]
    if first_speaker and first_speaker not in agents:
        raise ValueError(f"First speaker {first_speaker} must be included in --agents.")
    allow_partial = bool(options.get("allow_partial"))
    total_slots = total_turns or DEFAULT_ROUNDS * len(agents)
    rounds = max(1, min(HARD_MAX_ROUNDS, (total_slots + len(agents) - 1) // len(agents)))
    return CrewCommand(
        topic=topic,
        agents=agents,
        rounds=rounds,
        first_speaker=first_speaker,
        allow_partial=allow_partial,
        style=CrewStyle(
            max_words_per_turn=max_words or (60 if conversation_only else None),
            total_turns=total_turns,
            mode=mode,
            allow_bullets=allow_bullets,
            allow_reports=(allow_reports or report_requested or perrow) and not (terse and not perrow),
            display_mode="conversation_only" if conversation_only else "full",
            allow_synthesis=report_requested,
            perrow_explicitly_requested=perrow,
            parsed_rounds=parsed_rounds,
            joint_verdict_requested=joint_verdict_requested,
            joint_verdict_max_words=80,
        ),
    )


def parse_debate_alex_command(command_text: str) -> DebateAlexCommand:
    raw = command_text.strip()
    if not raw:
        raise ValueError("Topic cannot be empty.")

    options, topic = _parse_leading_crew_options(raw) if raw.startswith("--") else ({}, raw)
    topic = _strip_wrapping_topic_quotes(topic.strip())
    if not topic:
        raise ValueError("Topic cannot be empty.")

    turns = _parse_int_option(options, "turns") or DEFAULT_DEBATE_ALEX_TURNS
    if turns < 4:
        raise ValueError("/debate_alex requires at least 4 turns.")
    max_words = _parse_int_option(options, "max_words") or 60
    first = _normalize_agent(str(options.get("first", "Churchill")))
    judge = _normalize_agent(str(options.get("judge", "FDR")))
    mode = _style_mode(str(options.get("style", "dinner")))
    allow_bullets = bool(options.get("bullets")) and not bool(options.get("no_bullets"))
    allow_reports = bool(options.get("reports")) and not bool(options.get("no_reports"))
    final_verdict = bool(options.get("final_verdict") or options.get("report")) or _explicitly_requests_report(topic)
    perrow = _explicitly_requests_perrow(topic)
    if perrow:
        allow_reports = True
    return DebateAlexCommand(
        topic=topic,
        first=first,
        judge=judge,
        turns=turns,
        final_verdict=final_verdict,
        style=CrewStyle(
            max_words_per_turn=max_words,
            total_turns=turns,
            mode=mode,
            allow_bullets=allow_bullets,
            allow_reports=allow_reports,
            display_mode="conversation_only",
            allow_synthesis=final_verdict,
            perrow_explicitly_requested=perrow,
        ),
    )


def synthesize_dialogue(topic: str, agents: list[str], turns: list[AgentTurn]) -> dict[str, Any]:
    sarah_turns = [turn.message for turn in turns if turn.speaker == "FDR"]
    churchill_turns = [turn.message for turn in turns if turn.speaker == "Churchill"]
    final_lines = [turn.message.strip() for turn in turns[-2:] if turn.message.strip()]
    return {
        "points_of_agreement": _extract_bullets(final_lines, "Shared ground"),
        "points_of_disagreement": [
            "No hard disagreement was automatically detected; review the transcript for nuance."
        ],
        "sarah_specific_view": _compact_view(sarah_turns, "FDR did not produce a turn."),
        "churchill_specific_view": _compact_view(churchill_turns, "Churchill did not produce a turn."),
        "combined_answer_for_alex": _compact_view(final_lines, f"Combined answer on: {topic}"),
        "recommended_next_question": f"What assumption in this {agents[0]}/{agents[-1]} exchange would change the answer most?",
    }


def write_transcripts(payload: dict[str, Any], transcript_dir: Path = TRANSCRIPT_DIR) -> None:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    conversation_id = payload["conversation_id"]
    json_path = transcript_dir / f"{conversation_id}.json"
    md_path = transcript_dir / f"{conversation_id}.md"
    latest_md_path = transcript_dir / LATEST_TRANSCRIPT_MD
    latest_json_path = transcript_dir / LATEST_TRANSCRIPT_JSON
    payload["transcript_markdown_path"] = str(md_path)
    payload["transcript_md_path"] = str(md_path)
    payload["transcript_json_path"] = str(json_path)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = format_transcript_markdown(payload)
    md_path.write_text(markdown, encoding="utf-8")
    latest_md_path.write_text(markdown, encoding="utf-8")
    latest_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_crew_state() -> dict[str, Any]:
    if not CREW_STATE_FILE.exists():
        return {}
    return json.loads(CREW_STATE_FILE.read_text(encoding="utf-8"))


def save_crew_state(state: dict[str, Any]) -> None:
    CREW_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CREW_STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def update_crew_state_from_result(payload: dict[str, Any], status: str = "open") -> dict[str, Any]:
    style = payload.get("crew_style", {})
    state = {
        "active_conversation_id": payload.get("conversation_id") if status == "open" else None,
        "latest_conversation_id": payload.get("conversation_id"),
        "command_type": payload.get("command_type", "crew"),
        "use_alex_proxy": bool(payload.get("use_alex_proxy", False)),
        "real_user": payload.get("real_user", "Alex"),
        "proxy_user_label": payload.get("proxy_user_label"),
        "agents": payload.get("agents", []),
        "judge": payload.get("judge"),
        "first": payload.get("first"),
        "first_speaker": payload.get("first_speaker"),
        "turn_order": payload.get("turn_order", []),
        "planned_turn_order": payload.get("planned_turn_order") or payload.get("turn_order", []),
        "route_handler_used": payload.get("route_handler_used", payload.get("command_type", "crew")),
        "requested_turns": payload.get("requested_turns", payload.get("total_turns")),
        "parsed_rounds": payload.get("parsed_rounds"),
        "parsed_max_words": payload.get("parsed_max_words"),
        "number_of_agents": payload.get("number_of_agents"),
        "total_agent_turns": payload.get("total_agent_turns"),
        "joint_verdict_requested": payload.get("joint_verdict_requested", False),
        "allow_partial": payload.get("allow_partial", False),
        "topic": payload.get("topic", ""),
        "transcript_path": payload.get("transcript_md_path") or payload.get("transcript_markdown_path", ""),
        "transcript_md_path": payload.get("transcript_md_path") or payload.get("transcript_markdown_path", ""),
        "transcript_json_path": payload.get("transcript_json_path", ""),
        "style": {
            "max_words": style.get("max_words_per_turn"),
            "style": style.get("mode"),
            "allow_bullets": style.get("allow_bullets"),
            "allow_reports": style.get("allow_reports"),
        },
        "style_settings": style,
        "status": status,
        "turns_completed": payload.get("total_turns", len(payload.get("turns", []))),
        "updated_at": _utc_now(),
    }
    save_crew_state(state)
    return state


def close_active_crew() -> dict[str, Any]:
    state = load_crew_state()
    if not state.get("active_conversation_id"):
        return {"ok": False, "error": NO_ACTIVE_CREW_MESSAGE, "text": NO_ACTIVE_CREW_MESSAGE}
    state["status"] = "closed"
    state["active_conversation_id"] = None
    state["updated_at"] = _utc_now()
    save_crew_state(state)
    return {"ok": True, **state}


def get_latest_crew_info() -> dict[str, Any]:
    state = load_crew_state()
    if not state.get("latest_conversation_id"):
        return {"ok": False, "error": NO_ACTIVE_CREW_MESSAGE, "text": NO_ACTIVE_CREW_MESSAGE}
    md_path = Path(str(state.get("transcript_md_path") or state.get("transcript_path") or ""))
    json_path = Path(str(state.get("transcript_json_path") or ""))
    latest_md = TRANSCRIPT_DIR / LATEST_TRANSCRIPT_MD
    latest_json = TRANSCRIPT_DIR / LATEST_TRANSCRIPT_JSON
    return {
        "ok": True,
        **state,
        "transcript_md_exists": md_path.exists(),
        "transcript_json_exists": json_path.exists(),
        "latest_multi_agent_transcript_md_exists": latest_md.exists(),
        "latest_multi_agent_transcript_json_exists": latest_json.exists(),
    }


def load_crew_payload(conversation_id: str | None = None) -> dict[str, Any]:
    state = load_crew_state()
    selected_id = conversation_id or state.get("active_conversation_id") or state.get("latest_conversation_id")
    if not selected_id:
        raise FileNotFoundError(NO_ACTIVE_CREW_MESSAGE)
    candidates = []
    if state.get("transcript_json_path"):
        candidates.append(Path(str(state["transcript_json_path"])))
    candidates.append(TRANSCRIPT_DIR / f"{selected_id}.json")
    candidates.append(TRANSCRIPT_DIR / LATEST_TRANSCRIPT_JSON)
    for path in candidates:
        if path.exists() and path.stem == selected_id:
            return json.loads(path.read_text(encoding="utf-8"))
    repair = repair_crew_transcript(str(selected_id))
    payload = repair.get("payload")
    if isinstance(payload, dict):
        return payload
    raise FileNotFoundError(f"Crew transcript JSON not found for {selected_id}.")


def repair_crew_transcript(conversation_id: str | None = None) -> dict[str, Any]:
    state = load_crew_state()
    selected_id = conversation_id or state.get("active_conversation_id") or state.get("latest_conversation_id")
    transcript_dir = TRANSCRIPT_DIR
    transcript_dir.mkdir(parents=True, exist_ok=True)
    md_path = _find_repair_markdown_path(selected_id, state, transcript_dir)
    if not md_path:
        return {"ok": False, "error": "No markdown transcript found to repair.", "repaired": []}

    markdown = md_path.read_text(encoding="utf-8")
    payload = reconstruct_payload_from_markdown(markdown, selected_id=selected_id, source_path=md_path)
    conversation_id = str(payload["conversation_id"])
    json_path = transcript_dir / f"{conversation_id}.json"
    md_out = transcript_dir / f"{conversation_id}.md"
    repaired: list[str] = []
    if not md_out.exists() or md_out.resolve() != md_path.resolve():
        md_out.write_text(markdown, encoding="utf-8")
        repaired.append(f"created {md_out}")
    payload["transcript_markdown_path"] = str(md_out)
    payload["transcript_md_path"] = str(md_out)
    payload["transcript_json_path"] = str(json_path)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    repaired.append(f"created {json_path}")
    latest_md = transcript_dir / LATEST_TRANSCRIPT_MD
    latest_json = transcript_dir / LATEST_TRANSCRIPT_JSON
    latest_md.write_text(md_out.read_text(encoding="utf-8"), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    repaired.extend([f"updated {latest_md}", f"updated {latest_json}"])
    update_crew_state_from_result(payload, status=str(state.get("status") or "open"))
    return {"ok": True, "conversation_id": conversation_id, "payload": payload, "repaired": repaired}


def reconstruct_payload_from_markdown(markdown: str, selected_id: str | None = None, source_path: Path | None = None) -> dict[str, Any]:
    conversation_id = _extract_conversation_id_from_markdown(markdown) or selected_id or uuid4().hex
    topic = _extract_prefixed_line(markdown, "Topic:") or _extract_prefixed_line(markdown, "topic:") or "Recovered crew transcript."
    agents = _extract_agents_from_markdown(markdown) or ["Churchill", "FDR"]
    turns = _extract_turns_from_markdown(markdown)
    style = CrewStyle(display_mode="conversation_only", allow_bullets=False, allow_reports=False, max_words_per_turn=60)
    payload = {
        "conversation_id": conversation_id,
        "topic": topic,
        "agents": agents,
        "rounds_requested": max((int(turn.get("round_number") or 0) for turn in turns), default=0),
        "rounds_run": max((int(turn.get("round_number") or 0) for turn in turns), default=0),
        "total_turns": len(turns),
        "max_chars_per_turn": DEFAULT_MAX_CHARS_PER_TURN,
        "crew_style": _style_to_dict(style),
        "turns": turns,
        "synthesis": None,
        "transcript_markdown_path": str(source_path or ""),
        "transcript_md_path": str(source_path or ""),
        "transcript_json_path": "",
    }
    return payload


def format_transcript_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Multi-Agent Transcript {payload['conversation_id']}",
        "",
        f"topic: {payload['topic']}",
        f"agents: {', '.join(payload['agents'])}",
        f"rounds_run: {payload['rounds_run']}",
        "",
    ]
    for turn in payload["turns"]:
        lines.extend(
            [
                f"## Round {turn['round_number']} - {turn['speaker']}",
                "",
                f"timestamp: {turn['timestamp']}",
                f"recipient: {turn.get('recipient') or 'none'}",
                "",
                turn["message"],
                "",
            ]
        )
    synthesis = payload.get("synthesis")
    if synthesis:
        lines.extend(["## Final Synthesis", ""])
        for key, value in synthesis.items():
            title = key.replace("_", " ")
            lines.append(f"### {title}")
            if isinstance(value, list):
                lines.extend(f"- {item}" for item in value)
            else:
                lines.append(str(value))
            lines.append("")
    joint_verdict = payload.get("joint_verdict_for_alex")
    if joint_verdict:
        lines.extend(["Joint verdict for Alex:", "", str(joint_verdict).strip(), ""])
    return "\n".join(lines).strip() + "\n"


def debug_orchestrator() -> dict[str, Any]:
    transcript_path = TRANSCRIPT_DIR
    probe = probe_agent_endpoints()
    return {
        "current_local_agent": LOCAL_AGENT_NAME,
        "known_agents": list(KNOWN_AGENTS),
        "ports": dict(KNOWN_AGENT_PORTS),
        "sarah_endpoint_url": _agent_respond_url("FDR"),
        "churchill_endpoint_url": _agent_respond_url("Churchill"),
        "bus_path": str(BUS_FILE),
        "transcript_path": str(transcript_path),
        "sarah_endpoint_reachable": probe["FDR"]["endpoint_reachable"],
        "churchill_endpoint_reachable": probe["Churchill"]["endpoint_reachable"],
        "default_participants": ["Churchill", "FDR"],
        "parsed_participants_for_latest_command": LAST_ORCHESTRATOR_DEBUG.get("parsed_participants", []),
        "first_speaker": LAST_ORCHESTRATOR_DEBUG.get("first_speaker"),
        "planned_turn_order": LAST_ORCHESTRATOR_DEBUG.get("planned_turn_order", []),
        "actual_responders_called": LAST_ORCHESTRATOR_DEBUG.get("actual_responders_called", []),
        "failures": LAST_ORCHESTRATOR_DEBUG.get("failures", []),
    }


def format_debug_orchestrator() -> str:
    debug = debug_orchestrator()
    return "\n".join(
        [
            "Multi-agent orchestrator debug:",
            f"current_local_agent: {debug['current_local_agent']}",
            f"known_agents: {', '.join(debug['known_agents'])}",
            "ports: " + ", ".join(f"{agent}={port}" for agent, port in debug["ports"].items()),
            f"sarah_endpoint_url: {debug['sarah_endpoint_url']}",
            f"churchill_endpoint_url: {debug['churchill_endpoint_url']}",
            f"bus_path: {debug['bus_path']}",
            f"transcript_path: {debug['transcript_path']}",
            f"sarah_endpoint_reachable: {debug['sarah_endpoint_reachable']}",
            f"churchill_endpoint_reachable: {debug['churchill_endpoint_reachable']}",
            "default_participants: " + ", ".join(debug["default_participants"]),
            "parsed_participants_for_latest_command: " + ", ".join(debug["parsed_participants_for_latest_command"]),
            f"first_speaker: {debug['first_speaker']}",
            "planned_turn_order: " + ", ".join(debug["planned_turn_order"]),
            "actual_responders_called: " + ", ".join(debug["actual_responders_called"]),
            "failures: " + "; ".join(debug["failures"]),
        ]
    )


def debug_debate_alex() -> dict[str, Any]:
    state = load_crew_state()
    payload: dict[str, Any] = {}
    if state.get("latest_conversation_id") or state.get("active_conversation_id"):
        try:
            payload = load_crew_payload()
        except Exception:
            payload = {}
    md_path = Path(str(state.get("transcript_md_path") or state.get("transcript_path") or ""))
    json_path = Path(str(state.get("transcript_json_path") or ""))
    style = state.get("style_settings") or {}
    use_alex_proxy = bool(state.get("use_alex_proxy", payload.get("use_alex_proxy", False)))
    visible_labels = [
        str(turn.get("speaker", ""))
        for turn in payload.get("turns", [])
        if str(turn.get("speaker", "")) in {"Churchill", "FDR", "Alex-proxy"}
        and (str(turn.get("speaker", "")) != "Alex-proxy" or use_alex_proxy)
    ]
    return {
        "active_conversation_id": state.get("active_conversation_id"),
        "command_type": state.get("command_type") or payload.get("command_type"),
        "use_alex_proxy": use_alex_proxy,
        "first": state.get("first") or state.get("first_speaker"),
        "judge": state.get("judge"),
        "requested_turns": state.get("requested_turns") or payload.get("requested_turns") or payload.get("total_turns"),
        "max_words": style.get("max_words_per_turn") or style.get("max_words"),
        "style": style.get("mode") or style.get("style"),
        "planned_turn_order": state.get("planned_turn_order") or state.get("turn_order", []),
        "actual_visible_labels": visible_labels,
        "route_handler_used": state.get("route_handler_used") or payload.get("route_handler_used"),
        "transcript_md_path": str(md_path),
        "transcript_md_exists": md_path.exists(),
        "transcript_json_path": str(json_path),
        "transcript_json_exists": json_path.exists(),
        "sarah_endpoint_reachable": is_agent_endpoint_reachable("FDR"),
        "churchill_endpoint_reachable": is_agent_endpoint_reachable("Churchill"),
    }


def format_debug_debate_alex() -> str:
    debug = debug_debate_alex()
    return "\n".join(
        [
            "Debate Alex debug:",
            f"active_conversation_id: {debug['active_conversation_id']}",
            f"command_type: {debug['command_type']}",
            f"use_alex_proxy: {str(debug['use_alex_proxy']).lower()}",
            f"first: {debug['first']}",
            f"judge: {debug['judge']}",
            f"requested_turns: {debug['requested_turns']}",
            f"max_words: {debug['max_words']}",
            f"style: {debug['style']}",
            "planned_turn_order: " + ", ".join(debug["planned_turn_order"]),
            "actual_visible_labels: " + ", ".join(debug["actual_visible_labels"]),
            f"route_handler_used: {debug['route_handler_used']}",
            f"transcript_md_path: {debug['transcript_md_path']}",
            f"transcript_md_exists: {str(debug['transcript_md_exists']).lower()}",
            f"transcript_json_path: {debug['transcript_json_path']}",
            f"transcript_json_exists: {str(debug['transcript_json_exists']).lower()}",
            f"sarah_endpoint_reachable: {str(debug['sarah_endpoint_reachable']).lower()}",
            f"churchill_endpoint_reachable: {str(debug['churchill_endpoint_reachable']).lower()}",
        ]
    )


def debug_crew_mode() -> dict[str, Any]:
    state = load_crew_state()
    payload: dict[str, Any] = {}
    if state.get("latest_conversation_id") or state.get("active_conversation_id"):
        try:
            payload = load_crew_payload()
        except Exception:
            payload = {}
    command_type = str(state.get("command_type") or payload.get("command_type") or "crew")
    use_alex_proxy = bool(state.get("use_alex_proxy", payload.get("use_alex_proxy", False)))
    visible_labels = [
        str(turn.get("speaker", ""))
        for turn in payload.get("turns", [])
        if str(turn.get("speaker", "")) in {"FDR", "Churchill", "Alex-proxy"}
        and (str(turn.get("speaker", "")) != "Alex-proxy" or use_alex_proxy)
    ]
    return {
        "command_type": command_type,
        "use_alex_proxy": use_alex_proxy,
        "participants": state.get("agents") or payload.get("agents", []),
        "first_speaker": state.get("first") or state.get("first_speaker") or payload.get("first_speaker"),
        "judge": state.get("judge") or payload.get("judge"),
        "parsed_topic": state.get("topic") or payload.get("topic", ""),
        "planned_turn_order": state.get("planned_turn_order") or payload.get("planned_turn_order") or payload.get("turn_order", []),
        "actual_visible_labels": visible_labels,
        "route_handler_used": state.get("route_handler_used") or payload.get("route_handler_used") or command_type,
    }


def format_debug_crew_mode() -> str:
    debug = debug_crew_mode()
    return "\n".join(
        [
            "Crew mode debug:",
            f"command_type: {debug['command_type']}",
            f"use_alex_proxy: {str(debug['use_alex_proxy']).lower()}",
            "participants: " + ", ".join(debug["participants"]),
            f"first_speaker: {debug['first_speaker']}",
            f"judge: {debug['judge']}",
            f"parsed_topic: {debug['parsed_topic']}",
            "planned_turn_order: " + ", ".join(debug["planned_turn_order"]),
            "actual_visible_labels: " + ", ".join(debug["actual_visible_labels"]),
            f"route_handler_used: {debug['route_handler_used']}",
        ]
    )


def debug_crew_turns() -> dict[str, Any]:
    state = load_crew_state()
    payload: dict[str, Any] = {}
    if state.get("latest_conversation_id") or state.get("active_conversation_id"):
        try:
            payload = load_crew_payload()
        except Exception:
            payload = {}
    style = payload.get("crew_style") or state.get("style_settings") or {}
    visible_labels = [
        str(turn.get("speaker", ""))
        for turn in payload.get("turns", [])
        if str(turn.get("speaker", "")) in {"FDR", "Churchill"}
    ]
    return {
        "command_type": state.get("command_type") or payload.get("command_type") or "crew",
        "use_alex_proxy": bool(state.get("use_alex_proxy", payload.get("use_alex_proxy", False))),
        "parsed_rounds": state.get("parsed_rounds") or payload.get("parsed_rounds"),
        "parsed_max_words": state.get("parsed_max_words") or payload.get("parsed_max_words") or style.get("max_words_per_turn"),
        "number_of_agents": state.get("number_of_agents") or payload.get("number_of_agents") or len(payload.get("agents", [])),
        "total_agent_turns": state.get("total_agent_turns") or payload.get("total_agent_turns") or len(visible_labels),
        "first_speaker": state.get("first_speaker") or payload.get("first_speaker"),
        "planned_turn_order": state.get("planned_turn_order") or payload.get("planned_turn_order", []),
        "joint_verdict_requested": bool(
            state.get("joint_verdict_requested", payload.get("joint_verdict_requested", False))
        ),
        "reports_allowed": bool(style.get("allow_reports", True)),
        "actual_visible_labels": visible_labels,
    }


def format_debug_crew_turns() -> str:
    debug = debug_crew_turns()
    return "\n".join(
        [
            "Crew turns debug:",
            f"command_type: {debug['command_type']}",
            f"use_alex_proxy: {str(debug['use_alex_proxy']).lower()}",
            f"rounds: {debug['parsed_rounds']}",
            f"max_words: {debug['parsed_max_words']}",
            f"number_of_agents: {debug['number_of_agents']}",
            f"total_agent_turns: {debug['total_agent_turns']}",
            f"first_speaker: {debug['first_speaker']}",
            "planned_turn_order: " + ", ".join(debug["planned_turn_order"]),
            f"joint_verdict_requested: {str(debug['joint_verdict_requested']).lower()}",
            f"reports_allowed: {str(debug['reports_allowed']).lower()}",
            "actual_visible_labels: " + ", ".join(debug["actual_visible_labels"]),
        ]
    )


def is_agent_endpoint_reachable(agent: str) -> bool:
    try:
        port = KNOWN_AGENT_PORTS[_normalize_agent(agent)]
        with request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=0.8):
            return True
    except Exception:
        return False


def probe_agent_endpoints() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for agent in ("FDR", "Churchill"):
        reachable = is_agent_endpoint_reachable(agent)
        result[agent] = {
            "endpoint_url": _agent_respond_url(agent),
            "endpoint_reachable": reachable,
            "can_respond": reachable,
        }
    return result


def format_crew_probe() -> str:
    probe = probe_agent_endpoints()
    lines = ["Crew endpoint probe:"]
    for agent in ("FDR", "Churchill"):
        status = "reachable" if probe[agent]["endpoint_reachable"] else "unreachable"
        lines.append(f"{agent} endpoint: {status}")
        lines.append(f"Can {agent} respond: {str(probe[agent]['can_respond']).lower()}")
    return "\n".join(lines)


def _record_bus_handoff(
    from_agent: str,
    to_agent: str,
    body: str,
    conversation_id: str,
    round_number: int,
) -> None:
    try:
        send_agent_message(
            from_agent=from_agent,
            to_agent=to_agent,
            subject=f"Multi-agent round {round_number}",
            body=body,
            conversation_id=conversation_id,
            metadata={"orchestrator": True, "round_number": round_number, "category": "orchestrator"},
            category="orchestrator",
        )
    except Exception:
        # The HTTP transcript is authoritative; bus logging should not kill the run.
        return


def _normalize_agents(agents: list[str]) -> list[str]:
    normalized = [_normalize_agent(agent) for agent in agents if agent.strip()]
    if not normalized:
        raise ValueError("At least one known agent is required.")
    return normalized


def _normalize_agent(agent: str) -> str:
    stripped = agent.strip()
    for known in KNOWN_AGENT_PORTS:
        if stripped.lower() == known.lower():
            return known
    raise ValueError(f"Unknown agent: {agent}")


def _parse_agents_option(value: object) -> list[str]:
    agents = _normalize_agents([agent.strip() for agent in str(value).split(",") if agent.strip()])
    seen: list[str] = []
    for agent in agents:
        if agent not in seen:
            seen.append(agent)
    return seen


def _parse_rounds_phrase(text: str) -> int | None:
    lowered = text.lower()
    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
    }
    match = re.search(r"\b(\d+)\s+rounds?(?:\s+only)?\b", lowered)
    if match:
        return int(match.group(1))
    for word, value in word_numbers.items():
        if re.search(rf"\b{word}\s+rounds?(?:\s+only)?\b", lowered):
            return value
    return None


def _parse_max_words_phrase(text: str) -> int | None:
    lowered = text.lower()
    for pattern in (
        r"\beach\s+turn\s+max(?:imum)?\s+(\d+)\s+words?\b",
        r"\bmax(?:imum)?\s+(\d+)\s+words?\b",
        r"\b(\d+)\s+words?\s+max(?:imum)?\b",
    ):
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return None


def _requests_no_bullets(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ("no bullet", "no bullets", "no bullet flood"))


def _requests_no_reports(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ("no reports", "no report", "no full cas report", "no cas report"))


def _requests_joint_verdict(text: str) -> bool:
    lowered = text.lower()
    return "joint verdict for alex" in lowered or "end with a joint verdict" in lowered


def _order_agents(agent_names: list[str], first_speaker: str) -> list[str]:
    if first_speaker not in agent_names:
        raise ValueError(f"First speaker {first_speaker} must be included in --agents.")
    index = agent_names.index(first_speaker)
    return agent_names[index:] + agent_names[:index]


def _planned_turn_order(turn_order: list[str], turn_limit: int) -> list[str]:
    if not turn_order:
        return []
    return [turn_order[index % len(turn_order)] for index in range(turn_limit)]


def build_debate_alex_turn_order(first: str = "Churchill", judge: str = "FDR", turns: int = DEFAULT_DEBATE_ALEX_TURNS) -> list[str]:
    first = _normalize_agent(first)
    judge = _normalize_agent(judge)
    if first == judge:
        raise ValueError("First speaker and judge must be different agents.")
    order = [first, "Alex-proxy", first, judge]
    while len(order) < turns:
        order.append(first if order[-1] == judge else judge)
    return order[:turns]


def _debate_alex_turn_order(first: str, judge: str, turns: int) -> list[str]:
    return build_debate_alex_turn_order(first=first, judge=judge, turns=turns)


def _next_debate_recipient(planned: list[str], index: int) -> str | None:
    if index + 1 >= len(planned):
        return None
    return planned[index + 1]


def _ensure_agents_reachable(agent_names: list[str]) -> None:
    for agent in agent_names:
        if not is_agent_endpoint_reachable(agent):
            url = _agent_respond_url(agent)
            if agent == "FDR":
                raise RuntimeError(
                    "FDR is not running at http://127.0.0.1:8010/agent/respond. "
                    "Start FDR on port 8010, then retry."
                )
            raise RuntimeError(f"{agent} unavailable at {url}. Start {agent} on port {KNOWN_AGENT_PORTS[agent]}, then retry.")


def _agent_respond_url(agent: str) -> str:
    agent_name = _normalize_agent(agent)
    return f"http://127.0.0.1:{KNOWN_AGENT_PORTS[agent_name]}/agent/respond"


def _set_orchestrator_debug(data: dict[str, Any]) -> None:
    LAST_ORCHESTRATOR_DEBUG.clear()
    LAST_ORCHESTRATOR_DEBUG.update(data)


def _append_debug_call(agent: str) -> None:
    LAST_ORCHESTRATOR_DEBUG.setdefault("actual_responders_called", []).append(agent)


def _append_debug_failure(message: str) -> None:
    LAST_ORCHESTRATOR_DEBUG.setdefault("failures", []).append(message)


def _turn_to_dict(turn: AgentTurn) -> dict[str, Any]:
    return {
        "round_number": turn.round_number,
        "speaker": turn.speaker,
        "recipient": turn.recipient,
        "message": turn.message,
        "forwarded_message": turn.forwarded_message,
        "timestamp": turn.timestamp,
    }


def _dict_to_turn(data: dict[str, Any]) -> AgentTurn:
    return AgentTurn(
        round_number=int(data.get("round_number") or 0),
        speaker=str(data.get("speaker") or ""),
        recipient=data.get("recipient"),
        message=str(data.get("message") or ""),
        forwarded_message=str(data.get("forwarded_message") or data.get("message") or ""),
        timestamp=str(data.get("timestamp") or _utc_now()),
    )


def _style_to_dict(style: CrewStyle) -> dict[str, Any]:
    return {
        "max_words_per_turn": style.max_words_per_turn,
        "total_turns": style.total_turns,
        "mode": style.mode,
        "allow_bullets": style.allow_bullets,
        "allow_reports": style.allow_reports,
        "display_mode": style.display_mode,
        "allow_synthesis": style.allow_synthesis,
        "perrow_explicitly_requested": style.perrow_explicitly_requested,
        "parsed_rounds": style.parsed_rounds,
        "joint_verdict_requested": style.joint_verdict_requested,
        "joint_verdict_max_words": style.joint_verdict_max_words,
    }


def _style_from_dict(data: dict[str, Any]) -> CrewStyle | None:
    if not data:
        return None
    return CrewStyle(
        max_words_per_turn=data.get("max_words_per_turn"),
        total_turns=data.get("total_turns"),
        mode=str(data.get("mode") or "standard"),
        allow_bullets=bool(data.get("allow_bullets", True)),
        allow_reports=bool(data.get("allow_reports", True)),
        display_mode=str(data.get("display_mode") or "full"),
        allow_synthesis=bool(data.get("allow_synthesis", False)),
        perrow_explicitly_requested=bool(data.get("perrow_explicitly_requested", False)),
        parsed_rounds=data.get("parsed_rounds"),
        joint_verdict_requested=bool(data.get("joint_verdict_requested", False)),
        joint_verdict_max_words=int(data.get("joint_verdict_max_words") or 80),
    )


def _extract_bullets(messages: list[str], fallback: str) -> list[str]:
    if not messages:
        return [fallback]
    compact = " ".join(message.replace("\n", " ") for message in messages)
    return [compact[:500].strip() or fallback]


def _compact_view(messages: list[str], fallback: str) -> str:
    if not messages:
        return fallback
    return messages[-1].strip()[:900]


def _find_repair_markdown_path(selected_id: str | None, state: dict[str, Any], transcript_dir: Path) -> Path | None:
    candidates: list[Path] = []
    state_md = state.get("transcript_md_path") or state.get("transcript_path") or state.get("transcript_markdown_path")
    if state_md:
        candidates.append(Path(str(state_md)))
    if selected_id:
        candidates.append(transcript_dir / f"{selected_id}.md")
    candidates.append(transcript_dir / LATEST_TRANSCRIPT_MD)
    if transcript_dir.exists():
        candidates.extend(
            sorted(
                (path for path in transcript_dir.glob("*.md") if path.name != LATEST_TRANSCRIPT_MD),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None


def _extract_conversation_id_from_markdown(markdown: str) -> str | None:
    for pattern in (
        r"Crew dialogue complete:\s*([0-9a-fA-F]{16,})",
        r"Multi-Agent Transcript\s+([0-9a-fA-F]{16,})",
        r"conversation_id[:\s]+([0-9a-fA-F]{16,})",
    ):
        match = re.search(pattern, markdown)
        if match:
            return match.group(1)
    return None


def _extract_prefixed_line(markdown: str, prefix: str) -> str | None:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return None


def _extract_agents_from_markdown(markdown: str) -> list[str]:
    agents_line = _extract_prefixed_line(markdown, "Participants:") or _extract_prefixed_line(markdown, "agents:")
    if agents_line:
        return _normalize_agents([agent.strip().split(":", 1)[0] for agent in agents_line.split(",") if agent.strip()])
    found: list[str] = []
    for match in re.finditer(r"(?m)^(?:##\s+Round\s+\d+\s+[—-]\s+)?(FDR|Churchill):?\s*$", markdown):
        agent = match.group(1)
        if agent not in found:
            found.append(agent)
    return found


def _extract_turns_from_markdown(markdown: str) -> list[dict[str, Any]]:
    heading_pattern = re.compile(r"(?m)^(?:##\s+Round\s+(\d+)\s+[—-]\s+)?(Alex-proxy|Alex|FDR|Churchill):?\s*$")
    matches = list(heading_pattern.finditer(markdown))
    turns: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        speaker = match.group(2)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        message = markdown[start:end].strip()
        message = re.split(r"\n---\n", message, maxsplit=1)[0].strip()
        message = _strip_markdown_turn_metadata(message)
        if not message:
            continue
        turns.append(
            {
                "round_number": int(match.group(1) or (len(turns) // 3 + 1)),
                "speaker": speaker,
                "recipient": "crew" if speaker in {"Alex", "Alex-proxy"} else "Alex",
                "message": message,
                "forwarded_message": message,
                "timestamp": _utc_now(),
            }
        )
    return turns


def _strip_markdown_turn_metadata(message: str) -> str:
    lines = []
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith("timestamp:") or stripped.startswith("recipient:") or stripped == "---":
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _remove_report_headings(text: str) -> str:
    forbidden = (
        "situation compression",
        "perrow placement",
        "cascade timeline",
        "hidden couplings",
        "patch recommendations",
        "what to monitor",
        "final synthesis",
        "points of agreement",
        "points of disagreement",
        "recommended next question",
    )
    lines: list[str] = []
    for line in text.splitlines():
        normalized = line.strip().strip(":").lower().lstrip("#").strip()
        if any(normalized.startswith(heading) for heading in forbidden):
            continue
        lines.append(line)
    return "\n".join(lines)


def _remove_internal_round_labels(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"(?i)^round\s+\d+\s*[:.-]?\s*$", stripped):
            continue
        if re.match(r"(?i)^(churchill|sarah)\s+round\s+\d+\s*[:.-]?\s*$", stripped):
            continue
        if re.match(r"(?i)^round\s+\d+\s*[:.-]\s*", stripped):
            line = re.sub(r"(?i)^round\s+\d+\s*[:.-]\s*", "", line).strip()
        lines.append(line)
    return "\n".join(lines)


def _contains_report_heading(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "situation compression",
            "perrow placement",
            "cascade timeline",
            "hidden couplings",
            "patch recommendations",
            "what to monitor",
            "final synthesis",
            "points of agreement",
            "points of disagreement",
            "recommended next question",
        )
    )


def _conversation_rewrite(text: str, max_words: int) -> str:
    cleaned = _remove_report_headings(text)
    cleaned = re.sub(r"(?m)^\s*[-*•]\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+[\.)]\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", cleaned)
    compact = " ".join(cleaned.split())
    words = compact.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]).rstrip(" ,;:") + "."
    return compact


def _parse_leading_crew_options(raw: str) -> tuple[dict[str, object], str]:
    options: dict[str, object] = {}
    index = 0
    value_options = {
        "--rounds": "rounds",
        "--turns": "turns",
        "--max-words": "max_words",
        "--style": "style",
        "--agents": "agents",
        "--first": "first",
        "--judge": "judge",
    }
    flag_options = {
        "--no-bullets": "no_bullets",
        "--no-reports": "no_reports",
        "--bullets": "bullets",
        "--reports": "reports",
        "--report": "report",
        "--synthesis": "report",
        "--final-synthesis": "report",
        "--final-verdict": "final_verdict",
        "--verdict": "final_verdict",
        "--allow-partial": "allow_partial",
    }

    while index < len(raw):
        index = _skip_spaces(raw, index)
        if not raw.startswith("--", index):
            break

        token_end = index
        while token_end < len(raw) and not raw[token_end].isspace():
            token_end += 1
        raw_token = raw[index:token_end]
        option, equals_value = _split_option_assignment(raw_token)

        if option in value_options:
            if equals_value is not None:
                value = equals_value
                index = token_end
            else:
                value, index = _read_option_value(raw, token_end, option)
            options[value_options[option]] = value
            continue

        if option in flag_options:
            if equals_value is not None:
                raise ValueError(f"Option {option} does not accept a value.")
            options[flag_options[option]] = True
            index = token_end
            continue

        raise ValueError(f"Unknown crew option: {option}")

    return options, raw[index:].strip()


def _read_option_value(raw: str, index: int, option: str) -> tuple[str, int]:
    index = _skip_spaces(raw, index)
    if index >= len(raw) or raw.startswith("--", index):
        raise ValueError(f"Missing value for {option}.")
    quote = raw[index] if raw[index] in {"'", '"'} else ""
    if quote:
        end = raw.find(quote, index + 1)
        if end == -1:
            raise ValueError(f"No closing quote for {option}.")
        return raw[index + 1 : end], end + 1

    end = index
    while end < len(raw) and not raw[end].isspace():
        end += 1
    return raw[index:end], end


def _split_option_assignment(token: str) -> tuple[str, str | None]:
    if "=" not in token:
        return token, None
    option, value = token.split("=", 1)
    if not value:
        raise ValueError(f"Missing value for {option}.")
    return option, value.strip("'\"")


def _skip_spaces(raw: str, index: int) -> int:
    while index < len(raw) and raw[index].isspace():
        index += 1
    return index


def _strip_wrapping_topic_quotes(topic: str) -> str:
    if len(topic) >= 2 and topic[0] == topic[-1] and topic[0] in {"'", '"'}:
        return topic[1:-1].strip()
    return topic


def _parse_int_option(options: dict[str, object], key: str) -> int | None:
    if key not in options:
        return None
    value = str(options[key])
    if not re.fullmatch(r"\d+", value):
        raise ValueError(f"Option --{key.replace('_', '-')} requires an integer.")
    return int(value)


def _style_mode(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"dinner", "conversation", "conversational", "terse", "short"}:
        return "conversational"
    return lowered or "standard"


def _explicitly_requests_perrow(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ("perrow", "perrow/cas", "normal accident", "cas accident"))


def _explicitly_requests_report(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ("give me a report", "give me a synthesis", "staff paper"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local FDR/Churchill multi-agent dialogue.")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--agents", default="Churchill,FDR")
    parser.add_argument("--first", default="", help="First speaker, e.g. FDR or Churchill.")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--max-chars-per-turn", type=int, default=DEFAULT_MAX_CHARS_PER_TURN)
    parser.add_argument("--output", default="")
    parser.add_argument("--no-synthesis", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_multi_agent_dialogue(
        topic=args.topic,
        agents=[agent.strip() for agent in args.agents.split(",")],
        rounds=args.rounds,
        max_chars_per_turn=args.max_chars_per_turn,
        output=args.output or None,
        no_synthesis=args.no_synthesis,
        first_speaker=args.first or None,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
