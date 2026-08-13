from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from sarah_astrid_ooda_game_v3_focusfix import ENEMY_MAX_SPEED, ENEMY_MIN_SPEED, INITIAL_AMMO


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
GAME_PATH = HERE / "sarah_astrid_ooda_game_v3_focusfix.py"
RUNTIME_DIR = HERE / "runtime"
STATE_FILE = RUNTIME_DIR / "session.json"
STOP_FILE = RUNTIME_DIR / "stop.request"
SESSION_LOG = RUNTIME_DIR / "session_decisions.jsonl"
BRIDGE_LOG = RUNTIME_DIR / "bridge.log"
VALID_MODES = {"human", "sarah", "astrid"}
AGENT_ENEMY_MIN_SPEED = 20.0
AGENT_ENEMY_MAX_SPEED = 30.0
AGENT_INITIAL_AMMO = 150


def invader_speed_range(mode: str) -> tuple[float, float]:
    if mode in {"sarah", "astrid"}:
        return AGENT_ENEMY_MIN_SPEED, AGENT_ENEMY_MAX_SPEED
    return ENEMY_MIN_SPEED, ENEMY_MAX_SPEED


def invader_speed_description(mode: str) -> str:
    minimum, maximum = invader_speed_range(mode)
    return f"{minimum:g}-{maximum:g} pixels/second"


def initial_ammo_for_mode(mode: str) -> int:
    return AGENT_INITIAL_AMMO if mode in {"sarah", "astrid"} else INITIAL_AMMO


class LatestSnapshot:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._snapshot: dict[str, Any] | None = None

    def update(self, snapshot: dict[str, Any]) -> None:
        with self._condition:
            self._snapshot = snapshot
            self._condition.notify_all()

    def get(self) -> dict[str, Any] | None:
        with self._condition:
            return dict(self._snapshot) if self._snapshot is not None else None

    def wait_after(
        self,
        snapshot_id: int,
        stop_event: threading.Event,
        timeout: float = 0.5,
    ) -> dict[str, Any] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: stop_event.is_set()
                or (
                    self._snapshot is not None
                    and int(self._snapshot.get("snapshot_id", -1)) > snapshot_id
                ),
                timeout=timeout,
            )
            if stop_event.is_set() or self._snapshot is None:
                return None
            if int(self._snapshot.get("snapshot_id", -1)) <= snapshot_id:
                return None
            return dict(self._snapshot)


class DecisionWorker:
    def __init__(
        self,
        agent: str,
        latest_snapshot: LatestSnapshot,
        decide: Callable[[dict[str, Any]], Any],
        dispatch: Callable[[dict[str, Any], dict[str, Any], float], None],
        record_failure: Callable[[dict[str, Any], float, str], None],
        stop_event: threading.Event,
    ) -> None:
        self.agent = agent
        self.latest_snapshot = latest_snapshot
        self.decide = decide
        self.dispatch = dispatch
        self.record_failure = record_failure
        self.stop_event = stop_event

    def run(self) -> None:
        last_snapshot_id = -1
        while not self.stop_event.is_set():
            snapshot = self.latest_snapshot.wait_after(last_snapshot_id, self.stop_event)
            if snapshot is None:
                continue
            last_snapshot_id = int(snapshot["snapshot_id"])
            if not snapshot.get("running", True):
                continue

            started = time.perf_counter()
            try:
                raw_command = self.decide(snapshot)
                command = validate_agent_command(raw_command)
            except Exception as exc:
                self.record_failure(snapshot, time.perf_counter() - started, str(exc))
                continue

            latency = time.perf_counter() - started
            if self.stop_event.is_set():
                return
            newest = self.latest_snapshot.get()
            if newest is not None and not newest.get("running", True):
                self.record_failure(snapshot, latency, "Game ended while the agent was deciding.")
                continue
            command.update(
                {
                    "type": "command",
                    "command_id": uuid4().hex,
                    "based_on_snapshot": snapshot["snapshot_id"],
                }
            )
            try:
                self.dispatch(command, snapshot, latency)
            except Exception as exc:
                self.record_failure(snapshot, latency, str(exc))


def validate_agent_command(raw_command: Any) -> dict[str, Any]:
    payload = raw_command if isinstance(raw_command, dict) else _parse_json_object(str(raw_command))
    movement = payload.get("movement")
    target = payload.get("target")
    fire_rounds = payload.get("fire_rounds")
    if movement not in {"LEFT", "RIGHT", "NONE"}:
        raise ValueError("movement must be LEFT, RIGHT, or NONE")
    if target is not None and not isinstance(target, str):
        raise ValueError("target must be a string or null")
    if isinstance(fire_rounds, bool) or not isinstance(fire_rounds, int) or fire_rounds < 0:
        raise ValueError("fire_rounds must be an integer greater than or equal to zero")
    return {"movement": movement, "target": target, "fire_rounds": fire_rounds}


def request_start(mode: str, python_executable: str | None = None) -> dict[str, Any]:
    normalized = mode.strip().lower()
    if normalized not in VALID_MODES:
        raise ValueError("Mode must be human, sarah, or astrid.")
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    current = _read_json(STATE_FILE)
    if current and _pid_is_running(int(current.get("pid", 0))):
        return {
            "ok": True,
            "started": False,
            "mode": current.get("mode"),
            "pid": current.get("pid"),
            "message": (
                f"Space Invaders is already running in {current.get('mode')} mode "
                f"(invader speed: {invader_speed_description(str(current.get('mode')))}; "
                f"ammunition: {initial_ammo_for_mode(str(current.get('mode')))} rounds)."
            ),
        }
    _remove_if_exists(STATE_FILE)
    _remove_if_exists(STOP_FILE)

    executable = python_executable or sys.executable
    command = [executable, "-u", str(Path(__file__).resolve()), "run", normalized]
    launch_started = time.time()
    with BRIDGE_LOG.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPOSITORY_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=_background_creation_flags(),
        )
    deadline = time.monotonic() + 3.0
    state = None
    while time.monotonic() < deadline:
        state = _read_json(STATE_FILE)
        if (
            state
            and state.get("mode") == normalized
            and float(state.get("started_at", 0)) >= launch_started - 1.0
        ):
            break
        time.sleep(0.05)
    bridge_pid = int((state or {}).get("pid", 0))
    if not state or not _pid_is_running(bridge_pid):
        return {
            "ok": False,
            "started": False,
            "mode": normalized,
            "message": f"Space Invaders failed to start. See {BRIDGE_LOG}.",
        }
    return {
        "ok": True,
        "started": True,
        "mode": normalized,
        "pid": bridge_pid,
        "message": (
            f"Space Invaders started in {normalized} mode "
            f"(invader speed: {invader_speed_description(normalized)}; "
            f"ammunition: {initial_ammo_for_mode(normalized)} rounds)."
        ),
    }


def request_stop(timeout: float = 6.0) -> dict[str, Any]:
    state = _read_json(STATE_FILE)
    if not state or not _pid_is_running(int(state.get("pid", 0))):
        _remove_if_exists(STATE_FILE)
        _remove_if_exists(STOP_FILE)
        return {"ok": True, "stopped": False, "message": "No Space Invaders session is running."}
    _write_json_atomic(STOP_FILE, {"type": "stop", "requested_at": time.time()})
    pid = int(state["pid"])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _pid_is_running(pid):
        time.sleep(0.1)
    stopped = not _pid_is_running(pid)
    return {
        "ok": stopped,
        "stopped": stopped,
        "mode": state.get("mode"),
        "message": "Space Invaders stopped." if stopped else "Space Invaders did not stop cleanly within the timeout.",
    }


def run_bridge(mode: str) -> int:
    normalized = mode.strip().lower()
    if normalized not in VALID_MODES:
        raise ValueError("Unsupported Space Invaders mode.")
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    _remove_if_exists(STOP_FILE)
    _write_json_atomic(
        STATE_FILE,
        {"pid": os.getpid(), "mode": normalized, "started_at": time.time(), "game_path": str(GAME_PATH)},
    )

    stop_event = threading.Event()
    latest = LatestSnapshot()
    game_events: queue.Queue[dict[str, Any]] = queue.Queue()
    pending: dict[str, dict[str, Any]] = {}
    pending_lock = threading.Lock()
    send_lock = threading.Lock()
    game = None
    decide = build_real_agent_decider(normalized) if normalized != "human" else None
    enemy_min_speed, enemy_max_speed = invader_speed_range(normalized)
    initial_ammo = initial_ammo_for_mode(normalized)

    try:
        game = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(GAME_PATH),
                "--agent-mode",
                "--enemy-min-speed",
                str(enemy_min_speed),
                "--enemy-max-speed",
                str(enemy_max_speed),
                "--initial-ammo",
                str(initial_ammo),
            ],
            cwd=str(REPOSITORY_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=_game_creation_flags(),
        )

        def send(payload: dict[str, Any]) -> None:
            if game.stdin is None or game.poll() is not None:
                raise RuntimeError("The game process is not accepting commands.")
            with send_lock:
                game.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                game.stdin.flush()

        def read_game() -> None:
            assert game.stdout is not None
            for line in game.stdout:
                try:
                    payload = json.loads(line)
                except ValueError as exc:
                    print(f"Invalid game telemetry: {exc}", file=sys.stderr, flush=True)
                    continue
                if payload.get("type") == "snapshot":
                    latest.update(payload)
                else:
                    game_events.put(payload)
            stop_event.set()

        threading.Thread(target=read_game, daemon=True).start()
        threading.Thread(target=_drain_stderr, args=(game.stderr,), daemon=True).start()

        if normalized != "human":
            assert decide is not None

            def dispatch(command: dict[str, Any], snapshot: dict[str, Any], latency: float) -> None:
                record = _base_decision_record(normalized, snapshot, latency)
                record.update(
                    {
                        "movement": command["movement"],
                        "target": command["target"],
                        "requested_fire_rounds": command["fire_rounds"],
                    }
                )
                with pending_lock:
                    pending[command["command_id"]] = record
                try:
                    send(command)
                except Exception:
                    with pending_lock:
                        pending.pop(command["command_id"], None)
                    raise

            def record_failure(snapshot: dict[str, Any], latency: float, error: str) -> None:
                record = _base_decision_record(normalized, snapshot, latency)
                record.update(
                    {
                        "command_applied_game_time": None,
                        "movement": None,
                        "target": None,
                        "requested_fire_rounds": 0,
                        "actual_rounds_fired": 0,
                        "error": error,
                    }
                )
                _append_session_record(record)

            worker = DecisionWorker(normalized, latest, decide, dispatch, record_failure, stop_event)
            threading.Thread(target=worker.run, daemon=True).start()

        while not stop_event.is_set():
            if STOP_FILE.exists():
                stop_event.set()
                break
            if game.poll() is not None:
                stop_event.set()
                break
            try:
                event = game_events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event.get("type") != "command_result":
                continue
            with pending_lock:
                record = pending.pop(str(event.get("command_id", "")), None)
            if record is None:
                continue
            record.update(
                {
                    "command_applied_game_time": event.get("command_applied_game_time"),
                    "movement": event.get("movement", record.get("movement")),
                    "target": event.get("target", record.get("target")),
                    "actual_rounds_fired": event.get("actual_rounds_fired", 0),
                    "ammo_remaining": event.get("ammo_remaining"),
                    "score": event.get("score"),
                    "lives": event.get("lives"),
                }
            )
            _append_session_record(record)
    finally:
        stop_event.set()
        if game is not None and game.poll() is None:
            try:
                if game.stdin is not None:
                    game.stdin.write('{"type":"close"}\n')
                    game.stdin.flush()
                game.wait(timeout=3.0)
            except Exception:
                game.terminate()
                try:
                    game.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    game.kill()
        state = _read_json(STATE_FILE)
        if state and int(state.get("pid", 0)) == os.getpid():
            _remove_if_exists(STATE_FILE)
        _remove_if_exists(STOP_FILE)
    return 0


def build_real_agent_decider(agent: str) -> Callable[[dict[str, Any]], str]:
    agent_dir = REPOSITORY_ROOT / ("sarah-v2-agent" if agent == "sarah" else "astrid-v1-agent")
    sys.path.insert(0, str(agent_dir))
    from app.core.sarah_engine import build_sarah_game_action_generator

    generate_action = build_sarah_game_action_generator()

    def decide(snapshot: dict[str, Any]) -> str:
        return generate_action(build_agent_prompt(agent, snapshot))

    return decide


def build_agent_prompt(agent: str, snapshot: dict[str, Any]) -> str:
    name = "Sarah" if agent == "sarah" else "Astrid"
    return (
        f"{name}, you control the blue defender in a continuously running Space Invaders OODA game. "
        "The game continues while you think. Coordinates, velocities, time-to-line, alignment, and simple "
        "clusters are supplied below. Each LEFT or RIGHT command is a short 0.675-second movement pulse, "
        "then blue stops pending reassessment. Select greedily: engage the nearest urgent live target or a nearby "
        "cluster, and move toward its current x position, never away from it. The local game validates movement "
        "against current geometry and corrects a stale direction. Use predicted_x_at_bullet_intercept to establish a firing solution "
        "and prefer fire when fire_solution_aligned is true. Selecting a non-null target authorizes at least one "
        "round even if fire_rounds is zero. A requested burst arms a five-second local firing "
        "window; the game releases rounds only while a live intercept is valid. Use predicted_x_at_defender_line "
        "for defensive movement "
        "rather than chasing current x. "
        "Avoid repeated outward movement near an edge and recover toward center when no urgent intercept requires "
        "otherwise. Ammunition is finite and begins at 150. An isolated target may justify "
        "one round; a tight cluster may justify a burst. Movement and firing are independent. Target records "
        "intent only and does not create auto-aim. Return ONLY one JSON object with exactly these keys, "
        'for example {"movement":"NONE","target":null,"fire_rounds":0}. Movement must be LEFT, RIGHT, '
        "or NONE; target must be a target/cluster ID or JSON null; fire_rounds must be an integer of zero "
        "or more. If target is not null, normally request at least one round; local fire control waits for a valid "
        "intercept rather than wasting the shot. While attackers remain alive, do not return both movement NONE "
        "and fire_rounds 0; "
        "take an observable defensive action. Do not return Markdown or explanation.\n\n"
        f"CURRENT SNAPSHOT:\n{json.dumps(snapshot, separators=(',', ':'))}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except ValueError:
        start = stripped.find("{")
        if start < 0:
            raise ValueError("Agent response did not contain a JSON object.")
        try:
            payload, _end = json.JSONDecoder().raw_decode(stripped[start:])
        except ValueError as exc:
            raise ValueError("Agent response contained malformed JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Agent response must be a JSON object.")
    return payload


def _base_decision_record(agent: str, snapshot: dict[str, Any], latency: float) -> dict[str, Any]:
    blue = snapshot.get("blue") or {}
    return {
        "agent": agent,
        "snapshot_id": snapshot.get("snapshot_id"),
        "snapshot_game_time": snapshot.get("game_time"),
        "command_applied_game_time": None,
        "wall_clock_llm_latency": round(latency, 3),
        "movement": None,
        "target": None,
        "requested_fire_rounds": 0,
        "actual_rounds_fired": 0,
        "ammo_remaining": blue.get("ammo_remaining"),
        "score": snapshot.get("score"),
        "lives": snapshot.get("lives"),
    }


def _append_session_record(record: dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with SESSION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _drain_stderr(stream: Any) -> None:
    if stream is None:
        return
    for line in stream:
        print(line.rstrip(), file=sys.stderr, flush=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        open_process.restype = ctypes.c_void_p
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        handle = open_process(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not get_exit_code(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            close_handle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _background_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW


def _game_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Control the local Sarah/Astrid Space Invaders session.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("mode", choices=sorted(VALID_MODES))
    subparsers.add_parser("stop")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("mode", choices=sorted(VALID_MODES))
    args = parser.parse_args(argv)
    if args.action == "start":
        print(json.dumps(request_start(args.mode), separators=(",", ":")))
        return 0
    if args.action == "stop":
        print(json.dumps(request_stop(), separators=(",", ":")))
        return 0
    return run_bridge(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
