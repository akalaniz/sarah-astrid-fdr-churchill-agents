import json
import logging
from pathlib import Path
import subprocess
import sys

from app.core.agent_bus import (
    AgentBusMessageTooLongError,
    LOCAL_AGENT_NAME,
    archive_inbox,
    clear_inbox,
    format_debug_agent_bus,
    format_debug_agent_bus_limits,
    format_inbox,
    format_inbox_view,
    format_message,
    format_thread,
    get_message,
    mark_message_read,
    reply_to_message,
    send_agent_message,
    send_agent_message_chunked,
)
from app.core.config import Settings
from app.core.conversation import TranscriptWriter
from app.core import sarah_engine
from app.core.memory import MemoryStore
from app.core.prompt_builder import format_debug_prompt_summary
from app.core.rag_context import format_sources
from app.core.safety import format_debug_astrid_boundaries, format_debug_safety
from app.core.self_survey import SELF_SURVEY_HELP, build_self_survey_prompt, is_self_survey_command, parse_self_survey_topic
from app.core import transcript_maintenance as transcripts
from app.core import web_cache_maintenance as web_cache
from app.orchestration.agent_orchestrator import format_debug_orchestrator, parse_crew_command, run_multi_agent_dialogue
from app.rag.retriever import RetrievalResult
from app.ui.mic_input import MicrophoneInputError, TranscriptionError, capture_and_transcribe

logger = logging.getLogger(__name__)


def run_chat_loop(settings: Settings) -> None:
    logger.info("Starting Astrid chat loop with model=%s", settings.sarah_model)

    sarah_engine.configure_sarah_engine(settings=settings)
    long_term_memory = MemoryStore(settings.memory_file)
    transcript = TranscriptWriter(settings.conversations_dir)
    session_id = "cli"
    last_sources: list[RetrievalResult] = []
    last_prompt_debug_summary = None
    last_safety_debug = None

    print("Astrid v1.0 chat. Type normally, or press Enter on an empty line to speak.")
    print("Commands: /sources, /self_survey help|<topic>, /transcripts, /transcripts_keep [N], /transcripts_archive [DAYS], /transcripts_prune [--days D --keep N], /transcripts_clear, /transcripts_archive_all, /web_cache, /web_cache_validate, /web_cache_keep [N], /web_cache_archive [DAYS], /web_cache_prune [--days D --keep N], /web_cache_clear, /web_cache_archive_all, /web_cache_delete_corrupt, /debug_prompt, /debug_safety, /debug_astrid_boundaries [text], /debug_memory_live, /debug_agent_bus, /debug_agent_bus_limits, /debug_orchestrator, /crew <topic>, /send Sarah: <text>, /send_chunked Sarah: <text>, /inbox, /inbox all, /archive_inbox, /clear_inbox_confirm, /read <id>, /reply <id>: <text>, /thread <id>, /space_invaders human|sarah|astrid|stop, /reset, /remember <text>, /forget <keyword>, /memory [keyword], /recall <keyword>, /recall_all, /clear_recall, /use_memory on|off, /quit.")
    print(f"Transcript: {transcript.path}")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            logger.info("Chat loop stopped by user")
            break

        command = user_input.lower()
        space_invaders = _handle_space_invaders_command(command)
        if space_invaders is not None:
            print(f"Astrid: {space_invaders}")
            continue
        if command in {"/quit", "quit", "exit"}:
            logger.info("Chat loop exited by command")
            transcript.write_event("quit")
            break

        if command == "/reset":
            sarah_engine.reset_sarah_session(session_id)
            last_sources = []
            last_prompt_debug_summary = None
            last_safety_debug = None
            transcript.write_event("reset")
            print("Astrid: Active conversation history cleared.")
            continue

        if command == "/sources":
            print(format_sources(last_sources))
            continue

        if is_self_survey_command(user_input):
            topic = parse_self_survey_topic(user_input)
            if topic.lower() == "help":
                print(SELF_SURVEY_HELP)
                continue
            try:
                reply = sarah_engine.generate_sarah_reply(build_self_survey_prompt(topic), session_id=session_id)
            except Exception as exc:
                logger.exception("Astrid self-survey request failed")
                print(f"Astrid: Self-survey failed: {exc}")
                continue
            answer = reply.answer
            last_prompt_debug_summary = reply.prompt_debug_summary
            last_safety_debug = getattr(reply, "safety_debug", None)
            last_sources = reply.retrieval_results
            print(f"Astrid: {answer}")
            transcript.write_turn(
                user=user_input,
                assistant=answer,
                model=settings.sarah_model,
                sources=reply.sources,
                web_sources=reply.web_sources,
                web_status=reply.web_status,
                has_sufficient_evidence=reply.has_sufficient_evidence,
            )
            continue

        transcript_text = _handle_transcript_command(user_input)
        if transcript_text is not None:
            print(transcript_text)
            continue

        web_cache_text = _handle_web_cache_command(user_input)
        if web_cache_text is not None:
            print(web_cache_text)
            continue

        if command == "/debug_prompt":
            if last_prompt_debug_summary is None:
                last_reply = sarah_engine.get_last_sarah_reply(session_id)
                last_prompt_debug_summary = last_reply.prompt_debug_summary if last_reply else None
            print(format_debug_prompt_summary(last_prompt_debug_summary))
            continue

        if command == "/debug_safety":
            if last_safety_debug is None:
                last_reply = sarah_engine.get_last_sarah_reply(session_id)
                last_safety_debug = getattr(last_reply, "safety_debug", None) if last_reply else None
            print(format_debug_safety(last_safety_debug))
            continue

        if (
            command == "/debug_astrid_boundaries"
            or command.startswith("/debug_astrid_boundaries ")
            or command == "/debug_sarah_boundaries"
            or command.startswith("/debug_sarah_boundaries ")
            or command == "/debug_ sarah_boundaries"
            or command.startswith("/debug_ sarah_boundaries ")
            or command == "/debug_ astrid_boundaries"
            or command.startswith("/debug_ astrid_boundaries ")
        ):
            if command.startswith("/debug_ sarah_boundaries"):
                sample = user_input[len("/debug_ sarah_boundaries") :].strip()
            elif command.startswith("/debug_sarah_boundaries"):
                sample = user_input[len("/debug_sarah_boundaries") :].strip()
            elif command.startswith("/debug_ astrid_boundaries"):
                sample = user_input[len("/debug_ astrid_boundaries") :].strip()
            else:
                sample = user_input[len("/debug_astrid_boundaries") :].strip()
            print(format_debug_astrid_boundaries(sample))
            continue

        if command == "/debug_memory_live":
            print(sarah_engine.format_debug_astrid_memory_live(session_id))
            continue

        if command == "/debug_agent_bus":
            print(format_debug_agent_bus(LOCAL_AGENT_NAME))
            continue

        if command == "/debug_agent_bus_limits":
            print(format_debug_agent_bus_limits())
            continue

        if command == "/debug_orchestrator":
            print(format_debug_orchestrator())
            continue

        if command.startswith("/crew "):
            try:
                crew_command = parse_crew_command(user_input[len("/crew ") :])
            except ValueError as exc:
                print(f"Crew command parse failed: {exc}")
                continue
            try:
                result = run_multi_agent_dialogue(
                    topic=crew_command.topic,
                    agents=crew_command.agents,
                    rounds=crew_command.rounds,
                )
            except Exception as exc:
                print(f"Crew orchestration failed: {exc}")
                continue
            print(f"Crew dialogue complete: {result['conversation_id']}")
            print(f"Transcript: {result['transcript_markdown_path']}")
            print(_format_synthesis_text(result.get("synthesis")))
            continue

        if command.startswith("/send "):
            try:
                recipient, body = _parse_send_command(user_input)
                message = send_agent_message(
                    from_agent=LOCAL_AGENT_NAME,
                    to_agent=recipient,
                    subject="Message from Astrid",
                    body=body,
                    metadata={"route": "cli"},
                )
            except (AgentBusMessageTooLongError, ValueError) as exc:
                print(f"Agent bus send failed: {exc}")
                continue
            warning = f" Warning: {message['warning']}" if message.get("warning") else ""
            print(f"Astrid: Sent to {message['to_agent']} as {message['id']} in thread {message['conversation_id']}.{warning}")
            continue

        if command.startswith("/send_chunked "):
            try:
                recipient, body = _parse_send_command(user_input, command_name="/send_chunked")
                messages = send_agent_message_chunked(
                    from_agent=LOCAL_AGENT_NAME,
                    to_agent=recipient,
                    subject="Message from Astrid",
                    body=body,
                    metadata={"route": "cli", "chunked": True},
                )
            except (AgentBusMessageTooLongError, ValueError) as exc:
                print(f"Agent bus chunked send failed: {exc}")
                continue
            thread_id = messages[0]["conversation_id"] if messages else ""
            print(f"Astrid: Sent {len(messages)} chunks to {recipient} in thread {thread_id}.")
            continue

        if command == "/inbox":
            print(format_inbox(LOCAL_AGENT_NAME))
            continue

        if command == "/inbox all":
            print(format_inbox_view(LOCAL_AGENT_NAME, include_all=True))
            continue

        if command == "/archive_inbox":
            print(f"Astrid: Archived {archive_inbox(LOCAL_AGENT_NAME)} messages addressed to {LOCAL_AGENT_NAME}.")
            continue

        if command == "/clear_inbox_confirm":
            print(f"Astrid: Cleared {clear_inbox(LOCAL_AGENT_NAME)} messages addressed to {LOCAL_AGENT_NAME}.")
            continue

        if command.startswith("/read "):
            message_id = user_input[len("/read ") :].strip()
            message = get_message(message_id)
            if mark_message_read(message_id, LOCAL_AGENT_NAME):
                message = get_message(message_id)
            print(format_message(message))
            continue

        if command.startswith("/reply "):
            try:
                message_id, body = _parse_reply_command(user_input)
                message = reply_to_message(
                    message_id=message_id,
                    from_agent=LOCAL_AGENT_NAME,
                    body=body,
                    metadata={"route": "cli"},
                )
            except (ValueError, PermissionError) as exc:
                print(f"Agent bus reply failed: {exc}")
                continue
            print(f"Astrid: Replied to {message['to_agent']} in thread {message['conversation_id']} as {message['id']}.")
            continue

        if command.startswith("/thread "):
            conversation_id = user_input[len("/thread ") :].strip()
            print(format_thread(conversation_id))
            continue

        if command.startswith("/remember "):
            memory_text = user_input[len("/remember ") :].strip()
            try:
                record = sarah_engine.remember_astrid_memory(memory_text, session_id=session_id)
            except ValueError as exc:
                print(f"Memory not stored: {exc}")
                continue
            transcript.write_event(f"remember:{record.memory_id}")
            print(f"Astrid: I’ll remember that. ({record.memory_type})")
            continue

        if command.startswith("/recall "):
            keyword = user_input[len("/recall ") :].strip()
            if not keyword:
                print("Astrid: Recall keyword cannot be empty.")
                continue
            print(sarah_engine.recall_astrid_memories(keyword, session_id=session_id))
            continue

        if command == "/recall_all":
            print(sarah_engine.recall_all_astrid_memories(session_id=session_id))
            continue

        if command == "/clear_recall":
            print(sarah_engine.clear_astrid_recall(session_id=session_id))
            continue

        if command in {"/use_memory on", "/use_memory off"}:
            print(sarah_engine.set_astrid_use_memory(command.endswith(" on"), session_id=session_id))
            continue

        if command.startswith("/forget "):
            keyword = user_input[len("/forget ") :].strip()
            forgotten = long_term_memory.forget(keyword)
            transcript.write_event(f"forget:{keyword}:{len(forgotten)}")
            print(f"Astrid: Forgot {len(forgotten)} matching memor{'y' if len(forgotten) == 1 else 'ies'}.")
            continue

        if command == "/memory":
            print(sarah_engine.format_astrid_memories())
            continue

        if command.startswith("/memory "):
            keyword = user_input[len("/memory ") :].strip()
            print(sarah_engine.format_astrid_memories(keyword))
            continue

        if not user_input:
            try:
                user_input = capture_and_transcribe(settings)
            except (MicrophoneInputError, TranscriptionError) as exc:
                print(f"Mic input unavailable: {exc}")
                continue

            if not user_input:
                print("Mic input did not produce a transcript.")
                continue
            print(f"You said: {user_input}")

        try:
            reply = sarah_engine.generate_sarah_reply(user_input, session_id=session_id)
        except Exception as exc:
            logger.exception("OpenAI chat request failed")
            print(f"Astrid: OpenAI request failed: {exc}")
            continue

        answer = reply.answer
        last_prompt_debug_summary = reply.prompt_debug_summary
        last_safety_debug = getattr(reply, "safety_debug", None)
        print(f"Astrid: {answer}")
        last_sources = reply.retrieval_results
        transcript.write_turn(
            user=user_input,
            assistant=answer,
            model=settings.sarah_model,
            sources=reply.sources,
            web_sources=reply.web_sources,
            web_status=reply.web_status,
            has_sufficient_evidence=reply.has_sufficient_evidence,
        )


def _handle_space_invaders_command(message: str) -> str | None:
    parts = message.strip().lower().split()
    if not parts or parts[0] != "/space_invaders":
        return None
    if len(parts) != 2 or parts[1] not in {"human", "sarah", "astrid", "stop"}:
        return "Use /space_invaders human|sarah|astrid|stop"
    bridge = Path(__file__).resolve().parents[3] / "games" / "space_invaders" / "bridge.py"
    action = ["stop"] if parts[1] == "stop" else ["start", parts[1]]
    result = subprocess.run(
        [sys.executable, "-u", str(bridge), *action],
        cwd=str(bridge.parents[2]),
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return result.stderr.strip() or "Space Invaders command failed."
    return str(payload.get("message", "Space Invaders command completed."))


def _parse_send_command(user_input: str, command_name: str = "/send") -> tuple[str, str]:
    payload = user_input[len(command_name) :].strip()
    if ":" not in payload:
        raise ValueError(f'Use {command_name} Sarah: message text')
    recipient, body = payload.split(":", 1)
    recipient = recipient.strip()
    body = body.strip()
    if not recipient or not body:
        raise ValueError(f'Use {command_name} Sarah: message text')
    return recipient, body


def _handle_transcript_command(user_input: str) -> str | None:
    command = user_input.lower().strip()
    try:
        if command == "/transcripts":
            return transcripts.format_transcript_result("Transcript status", transcripts.transcript_status())
        if command == "/transcripts_keep" or command.startswith("/transcripts_keep "):
            keep_n = transcripts.parse_keep_arg(user_input[len("/transcripts_keep") :])
            return transcripts.format_transcript_result(
                "Transcript keep result",
                transcripts.keep_newest_conversations(keep_n=keep_n),
            )
        if command == "/transcripts_archive" or command.startswith("/transcripts_archive "):
            days = transcripts.parse_archive_days_arg(user_input[len("/transcripts_archive") :])
            return transcripts.format_transcript_result(
                "Transcript archive result",
                transcripts.archive_older_than(days=days),
            )
        if command == "/transcripts_prune" or command.startswith("/transcripts_prune "):
            days, keep = transcripts.parse_prune_args(user_input[len("/transcripts_prune") :])
            return transcripts.format_transcript_result(
                "Transcript prune result",
                transcripts.prune_transcripts(days=days, keep=keep),
            )
        if command == "/transcripts_clear":
            return (
                "This will delete all live Sarah/Astrid transcript .md/.json files except latest convenience files "
                "if preserved. To confirm, run: /transcripts_clear confirm"
            )
        if command == "/transcripts_clear confirm":
            return transcripts.format_transcript_result(
                "Transcript clear result",
                transcripts.clear_transcripts(preserve_latest=True),
            )
        if command == "/transcripts_archive_all":
            return "To archive all live Sarah/Astrid transcripts, run: /transcripts_archive_all confirm"
        if command == "/transcripts_archive_all confirm":
            return transcripts.format_transcript_result(
                "Transcript archive-all result",
                transcripts.archive_all_transcripts(preserve_latest=True),
            )
    except ValueError as exc:
        return f"Transcript command failed: {exc}"
    return None


def _handle_web_cache_command(user_input: str) -> str | None:
    command = user_input.lower().strip()
    try:
        if command == "/web_cache":
            return web_cache.format_web_cache_result("Web cache status", web_cache.web_cache_status())
        if command == "/web_cache_validate":
            return web_cache.format_web_cache_result("Web cache validation", web_cache.validate_web_cache_json())
        if command == "/web_cache_keep" or command.startswith("/web_cache_keep "):
            keep_n = web_cache.parse_keep_arg(user_input[len("/web_cache_keep") :])
            return web_cache.format_web_cache_result(
                "Web cache keep result",
                web_cache.keep_newest_cache_files(keep_n=keep_n),
            )
        if command == "/web_cache_archive" or command.startswith("/web_cache_archive "):
            days = web_cache.parse_archive_days_arg(user_input[len("/web_cache_archive") :])
            return web_cache.format_web_cache_result(
                "Web cache archive result",
                web_cache.archive_cache_older_than(days=days),
            )
        if command == "/web_cache_prune" or command.startswith("/web_cache_prune "):
            days, keep = web_cache.parse_prune_args(user_input[len("/web_cache_prune") :])
            return web_cache.format_web_cache_result(
                "Web cache prune result",
                web_cache.prune_web_cache(days=days, keep=keep),
            )
        if command == "/web_cache_clear":
            return "This will delete all Astrid web-cache JSON files from data\\web_cache. To confirm, run: /web_cache_clear confirm"
        if command == "/web_cache_clear confirm":
            return web_cache.format_web_cache_result("Web cache clear result", web_cache.clear_web_cache())
        if command == "/web_cache_archive_all":
            return "To archive all Astrid web-cache JSON files, run: /web_cache_archive_all confirm"
        if command == "/web_cache_archive_all confirm":
            return web_cache.format_web_cache_result("Web cache archive-all result", web_cache.archive_all_web_cache())
        if command == "/web_cache_delete_corrupt":
            return "This will delete corrupt Astrid web-cache JSON files only. To confirm, run: /web_cache_delete_corrupt confirm"
        if command == "/web_cache_delete_corrupt confirm":
            return web_cache.format_web_cache_result(
                "Web cache delete-corrupt result",
                web_cache.delete_corrupt_web_cache(),
            )
    except ValueError as exc:
        return f"Web cache command failed: {exc}"
    return None


def _parse_reply_command(user_input: str) -> tuple[str, str]:
    payload = user_input[len("/reply ") :].strip()
    if ":" not in payload:
        raise ValueError('Use /reply <message_id>: message text')
    message_id, body = payload.split(":", 1)
    message_id = message_id.strip()
    body = body.strip()
    if not message_id or not body:
        raise ValueError('Use /reply <message_id>: message text')
    return message_id, body


def _format_synthesis_text(synthesis) -> str:
    if not synthesis:
        return "Synthesis disabled."
    lines = ["Final synthesis:"]
    for key, value in synthesis.items():
        label = key.replace("_", " ")
        if isinstance(value, list):
            lines.append(f"{label}: " + "; ".join(str(item) for item in value))
        else:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)
