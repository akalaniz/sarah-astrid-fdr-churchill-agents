from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import threading
import time
import tkinter as tk
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bridge
import sarah_astrid_ooda_game_v3_focusfix as game_module


def snapshot(snapshot_id: int, running: bool = True) -> dict:
    return {
        "type": "snapshot",
        "snapshot_id": snapshot_id,
        "game_time": float(snapshot_id),
        "running": running,
        "blue": {"x": 100.0, "y": 485.0, "vx": 0.0, "ammo_remaining": 500},
        "attackers": [],
        "clusters": [],
        "score": 0,
        "lives": 3,
    }


class GameStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_ammo_is_shared_finite_and_reset_restores_500(self) -> None:
        game = game_module.ContinuousOODAGame(self.root)
        self.assertEqual(game.ammo_remaining, 500)
        for _ in range(500):
            game.sim_time += game_module.FIRE_COOLDOWN + 0.001
            self.assertTrue(game.fire())
        game.sim_time += game_module.FIRE_COOLDOWN + 0.001
        self.assertFalse(game.fire())
        self.assertEqual(game.ammo_remaining, 0)
        game.reset()
        self.assertEqual(game.ammo_remaining, 500)

    def test_burst_truncates_at_zero_ammo(self) -> None:
        output = io.StringIO()
        game = game_module.ContinuousOODAGame(self.root, agent_mode=True, input_stream=io.StringIO(), output_stream=output)
        game.ammo_remaining = 3
        game._apply_agent_command(
            {
                "type": "command",
                "command_id": "burst",
                "based_on_snapshot": 1,
                "movement": "RIGHT",
                "target": "C1",
                "fire_rounds": 8,
            }
        )
        for _ in range(5):
            game.sim_time += game_module.FIRE_COOLDOWN
            game._service_bursts()
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        result = next(event for event in events if event.get("type") == "command_result")
        self.assertEqual(result["actual_rounds_fired"], 3)
        self.assertEqual(game.ammo_remaining, 0)
        self.assertEqual(game.agent_movement, "RIGHT")

    def test_space_after_restart_fires_and_does_not_restart(self) -> None:
        game = game_module.ContinuousOODAGame(self.root)
        game.running = False
        game.reset()
        before = len(game.bullets)
        self.assertEqual(game._space_fire(None), "break")
        self.assertEqual(len(game.bullets), before + 1)
        self.assertEqual(game.ammo_remaining, 499)
        self.assertTrue(game.running)


class TrackingTests(unittest.TestCase):
    def test_agent_attackers_spawn_visible_or_immediately_above_canvas(self) -> None:
        game = game_module.ContinuousOODAGame.__new__(game_module.ContinuousOODAGame)
        game.rng = game_module.random.Random(7)
        game.agent_mode = True
        game.enemy_min_speed = 20.0
        game.enemy_max_speed = 30.0
        initial = [game._new_enemy(f"T{i}", start_high=True) for i in range(20)]
        replacements = [game._new_enemy() for _ in range(20)]
        self.assertTrue(all(20 <= enemy.y <= 120 for enemy in initial))
        self.assertTrue(all(-20 <= enemy.y <= 20 for enemy in replacements))

    def test_tracking_uses_enemy_state_and_marks_clusters(self) -> None:
        enemies = [
            game_module.Enemy("T1", 100.0, 100.0, 1.0, 50.0, 30.0),
            game_module.Enemy("T2", 140.0, 120.0, 2.0, 60.0, 30.0),
            game_module.Enemy("T3", 500.0, 100.0, 3.0, 70.0, 30.0),
        ]
        attackers, clusters = game_module.build_attacker_tracking(enemies, player_x=110.0, player_w=44.0)
        by_id = {attacker["id"]: attacker for attacker in attackers}
        self.assertEqual(by_id["T1"]["cluster_id"], "C1")
        self.assertEqual(by_id["T2"]["cluster_id"], "C1")
        self.assertIsNone(by_id["T3"]["cluster_id"])
        self.assertTrue(by_id["T1"]["aligned_with_blue"])
        self.assertEqual(clusters[0]["members"], ["T1", "T2"])
        for field in ("x", "y", "vx", "vy", "time_to_defender_line"):
            self.assertIn(field, by_id["T1"])
        self.assertIn("predicted_x_at_defender_line", by_id["T1"])
        self.assertIn("predicted_x_at_bullet_intercept", by_id["T1"])
        self.assertIn("fire_solution_aligned", by_id["T1"])

    def test_bouncing_x_projection_stays_inside_the_field(self) -> None:
        for duration in (0.0, 1.0, 10.0, 100.0):
            projected = game_module._project_bouncing_x(700.0, 45.0, duration, 30.0)
            self.assertGreaterEqual(projected, 15.0)
            self.assertLessEqual(projected, game_module.WIDTH - 15.0)


class AgentMovementTests(unittest.TestCase):
    def _game(self):
        game = game_module.ContinuousOODAGame.__new__(game_module.ContinuousOODAGame)
        game.left_down = False
        game.right_down = False
        game.agent_mode = True
        game.agent_movement = "NONE"
        game.agent_movement_until = 0.0
        game.sim_time = 10.0
        game.player_x = game_module.WIDTH / 2
        return game

    def test_agent_movement_is_a_short_pulse(self) -> None:
        game = self._game()
        game.agent_movement = "RIGHT"
        game.agent_movement_until = game.sim_time + game_module.AGENT_MOVEMENT_PULSE
        self.assertEqual(game._movement_direction(), 1)
        game.sim_time = game.agent_movement_until
        self.assertEqual(game._movement_direction(), 0)
        self.assertEqual(game.agent_movement, "NONE")

    def test_agent_cannot_remain_pinned_outward_at_edges(self) -> None:
        game = self._game()
        game.agent_movement = "LEFT"
        game.agent_movement_until = game.sim_time + 1.0
        game.player_x = game_module.AGENT_EDGE_MARGIN
        self.assertEqual(game._movement_direction(), 0)
        self.assertEqual(game.agent_movement, "NONE")

    def test_agent_movement_pulse_is_half_the_previous_size(self) -> None:
        self.assertEqual(game_module.AGENT_MOVEMENT_PULSE, 0.675)

    def test_greedy_control_corrects_movement_away_from_live_target(self) -> None:
        game = self._game()
        game.player_w = 44.0
        game.enemies = [
            game_module.Enemy("T1", 120.0, 300.0, 0.0, 8.0, 30.0),
        ]
        movement, target = game._greedy_agent_engagement("T1")
        self.assertEqual(target, "T1")
        self.assertEqual(movement, "LEFT")

    def test_greedy_control_moves_toward_requested_cluster(self) -> None:
        game = self._game()
        game.player_w = 44.0
        game.enemies = [
            game_module.Enemy("T1", 470.0, 300.0, 0.0, 8.0, 30.0),
            game_module.Enemy("T2", 500.0, 310.0, 0.0, 8.0, 30.0),
        ]
        movement, target = game._greedy_agent_engagement("C1")
        self.assertEqual(target, "C1")
        self.assertEqual(movement, "RIGHT")

    def test_stale_target_falls_back_to_live_urgent_target(self) -> None:
        game = self._game()
        game.player_w = 44.0
        game.enemies = [
            game_module.Enemy("T1", 180.0, 420.0, 0.0, 8.0, 30.0),
            game_module.Enemy("T2", 600.0, 100.0, 0.0, 8.0, 30.0),
        ]
        movement, target = game._greedy_agent_engagement("T999")
        self.assertEqual(target, "T1")
        self.assertEqual(movement, "LEFT")


class FireSolutionTests(unittest.TestCase):
    def test_fire_solution_requires_current_bullet_intercept_alignment(self) -> None:
        aligned = game_module.Enemy("T1", 360.0, 200.0, 0.0, 15.0, 30.0)
        unaligned = game_module.Enemy("T2", 100.0, 200.0, 0.0, 15.0, 30.0)
        self.assertTrue(game_module._enemy_has_fire_solution(aligned, 360.0))
        self.assertFalse(game_module._enemy_has_fire_solution(unaligned, 360.0))

    def _game_with_burst(self):
        game = game_module.ContinuousOODAGame.__new__(game_module.ContinuousOODAGame)
        game.active_burst = {
            "command_id": "window",
            "based_on_snapshot": 1,
            "requested_fire_rounds": 1,
            "remaining": 1,
            "actual_rounds_fired": 0,
            "applied_game_time": 1.0,
            "expires_at": 3.0,
            "target": "T1",
            "movement": "NONE",
        }
        game.burst_queue = []
        game.sim_time = 1.5
        game.ammo_remaining = 500
        return game

    def test_fire_window_waits_instead_of_discarding_request(self) -> None:
        game = self._game_with_burst()
        game._target_has_fire_solution = lambda _target: False
        game.fire = lambda: self.fail("fire should wait for a solution")
        game._service_bursts()
        self.assertEqual(game.active_burst["remaining"], 1)

    def test_fire_window_releases_round_when_alignment_appears(self) -> None:
        game = self._game_with_burst()
        game._target_has_fire_solution = lambda _target: True
        game.fire = lambda: True
        game._emit_command_result = lambda _burst: None
        game._service_bursts()
        self.assertIsNone(game.active_burst)

    def test_target_selection_authorizes_default_single_round(self) -> None:
        game = game_module.ContinuousOODAGame.__new__(game_module.ContinuousOODAGame)
        game.agent_mode = True
        game.agent_movement = "NONE"
        game.agent_movement_until = 0.0
        game.sim_time = 1.0
        game.active_burst = None
        game.burst_queue = []
        game.ammo_remaining = 500
        game.running = True
        game.player_x = 360.0
        game.player_w = 44.0
        game.enemies = [
            game_module.Enemy("T1", 360.0, 200.0, 0.0, 8.0, 30.0),
        ]
        game._apply_agent_command(
            {
                "type": "command",
                "command_id": "default-round",
                "based_on_snapshot": 1,
                "movement": "NONE",
                "target": "T1",
                "fire_rounds": 0,
            }
        )
        self.assertEqual(game.burst_queue[0]["requested_fire_rounds"], 1)
        self.assertEqual(game.burst_queue[0]["remaining"], 1)
        self.assertEqual(game.burst_queue[0]["expires_at"], 6.0)


class BridgeTests(unittest.TestCase):
    def test_speed_ranges_preserve_human_and_slow_agents(self) -> None:
        self.assertEqual(bridge.invader_speed_range("human"), (42.0, 90.0))
        self.assertEqual(bridge.invader_speed_range("sarah"), (20.0, 30.0))
        self.assertEqual(bridge.invader_speed_range("astrid"), (20.0, 30.0))
        self.assertEqual(bridge.invader_speed_description("human"), "42-90 pixels/second")
        self.assertEqual(bridge.invader_speed_description("sarah"), "20-30 pixels/second")
        self.assertEqual(bridge.initial_ammo_for_mode("human"), 500)
        self.assertEqual(bridge.initial_ammo_for_mode("sarah"), 150)
        self.assertEqual(bridge.initial_ammo_for_mode("astrid"), 150)

    def test_command_validation(self) -> None:
        valid = bridge.validate_agent_command('{"movement":"LEFT","target":"T1","fire_rounds":1}')
        self.assertEqual(valid, {"movement": "LEFT", "target": "T1", "fire_rounds": 1})
        for invalid in (
            '{"movement":"UP","target":null,"fire_rounds":0}',
            '{"movement":"NONE","target":null,"fire_rounds":-1}',
            '{"movement":"NONE","target":7,"fire_rounds":0}',
        ):
            with self.assertRaises(ValueError):
                bridge.validate_agent_command(invalid)

    def test_slow_decision_keeps_only_latest_snapshot_and_one_call(self) -> None:
        latest = bridge.LatestSnapshot()
        stop_event = threading.Event()
        first_started = threading.Event()
        release_first = threading.Event()
        finished = threading.Event()
        lock = threading.Lock()
        calls = []
        active = 0
        maximum_active = 0

        def decide(state: dict) -> dict:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
                calls.append(state["snapshot_id"])
            try:
                if state["snapshot_id"] == 1:
                    first_started.set()
                    self.assertTrue(release_first.wait(2.0))
                return {"movement": "NONE", "target": None, "fire_rounds": 0}
            finally:
                with lock:
                    active -= 1

        dispatched = []

        def dispatch(command: dict, state: dict, latency: float) -> None:
            dispatched.append((command["based_on_snapshot"], state["snapshot_id"], latency))
            if len(dispatched) == 2:
                stop_event.set()
                finished.set()

        worker = bridge.DecisionWorker("sarah", latest, decide, dispatch, lambda *_args: None, stop_event)
        thread = threading.Thread(target=worker.run)
        thread.start()
        latest.update(snapshot(1))
        self.assertTrue(first_started.wait(1.0))
        latest.update(snapshot(2))
        latest.update(snapshot(3))
        latest.update(snapshot(4))
        time.sleep(0.05)
        self.assertEqual(calls, [1])
        release_first.set()
        self.assertTrue(finished.wait(2.0))
        thread.join(2.0)
        self.assertEqual(calls, [1, 4])
        self.assertEqual(maximum_active, 1)

    def test_malformed_and_api_failures_do_not_stop_next_decision(self) -> None:
        for failure in ("not json", RuntimeError("API unavailable")):
            with self.subTest(failure=failure):
                latest = bridge.LatestSnapshot()
                stop_event = threading.Event()
                failed = threading.Event()
                dispatched = threading.Event()
                attempts = 0

                def decide(_state: dict) -> Any:
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        if isinstance(failure, Exception):
                            raise failure
                        return failure
                    return {"movement": "NONE", "target": None, "fire_rounds": 0}

                worker = bridge.DecisionWorker(
                    "astrid",
                    latest,
                    decide,
                    lambda *_args: (dispatched.set(), stop_event.set()),
                    lambda *_args: failed.set(),
                    stop_event,
                )
                thread = threading.Thread(target=worker.run)
                thread.start()
                latest.update(snapshot(1))
                self.assertTrue(failed.wait(1.0))
                latest.update(snapshot(2))
                self.assertTrue(dispatched.wait(1.0))
                thread.join(2.0)
                self.assertEqual(attempts, 2)

    def test_game_over_pauses_decisions_and_restart_resumes(self) -> None:
        latest = bridge.LatestSnapshot()
        stop_event = threading.Event()
        dispatched = threading.Event()
        calls = []

        def decide(state: dict) -> dict:
            calls.append(state["snapshot_id"])
            return {"movement": "NONE", "target": None, "fire_rounds": 0}

        worker = bridge.DecisionWorker(
            "sarah",
            latest,
            decide,
            lambda *_args: (dispatched.set(), stop_event.set()),
            lambda *_args: None,
            stop_event,
        )
        thread = threading.Thread(target=worker.run)
        thread.start()
        latest.update(snapshot(1, running=False))
        time.sleep(0.05)
        self.assertEqual(calls, [])
        latest.update(snapshot(2, running=True))
        self.assertTrue(dispatched.wait(1.0))
        thread.join(2.0)
        self.assertEqual(calls, [2])


if __name__ == "__main__":
    unittest.main()
