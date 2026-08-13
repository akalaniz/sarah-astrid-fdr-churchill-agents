"""
Sarah/Astrid OODA Lab — Continuous-Time Human Test v3

Controls
--------
LEFT  : Left Arrow or A (hold to move; release to stop)
RIGHT : Right Arrow or D (hold to move; release to stop)
FIRE  : Space bar

CRITICAL DESIGN RULE
--------------------
Simulation time advances continuously and independently of all user input.
Movement and firing NEVER advance, pause, reset, or gate simulation time.
"""

import argparse
from dataclasses import dataclass
import json
import math
import queue
import random
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

WIDTH = 720
HEIGHT = 540
PLAYER_Y = HEIGHT - 55

PLAYER_SPEED = 310.0          # pixels / simulated second
BULLET_SPEED = 560.0          # pixels / simulated second
ENEMY_MIN_SPEED = 42.0
ENEMY_MAX_SPEED = 90.0
FIRE_COOLDOWN = 0.18          # real/sim seconds between shots
INITIAL_LIVES = 3
INITIAL_AMMO = 500
TARGET_COUNT = 4
FRAME_MS = 16                 # ~60 Hz display/update target
SNAPSHOT_INTERVAL = 0.25
CLUSTER_DISTANCE = 90.0
ALIGNMENT_MARGIN = 12.0
AGENT_MOVEMENT_PULSE = 0.675
AGENT_EDGE_MARGIN = 55.0
AGENT_FIRE_WINDOW = 5.0
AGENT_AIM_DEADBAND = 18.0
AGENT_NEARBY_CLUSTER_RANGE = 200.0


@dataclass
class Enemy:
    ident: str
    x: float
    y: float
    vx: float
    vy: float
    size: float


@dataclass
class Bullet:
    x: float
    y: float
    w: float = 6.0
    h: float = 14.0


class ContinuousOODAGame:
    def __init__(
        self,
        root: tk.Tk,
        agent_mode=False,
        input_stream=None,
        output_stream=None,
        enemy_min_speed=ENEMY_MIN_SPEED,
        enemy_max_speed=ENEMY_MAX_SPEED,
        initial_ammo=INITIAL_AMMO,
    ):
        self.root = root
        self.root.title("Sarah/Astrid OODA Lab — Continuous-Time v3")
        self.root.geometry("820x800")
        self.root.minsize(760, 730)

        self.rng = random.Random()
        self.enemy_min_speed = float(enemy_min_speed)
        self.enemy_max_speed = float(enemy_max_speed)
        self.initial_ammo = int(initial_ammo)

        self.left_down = False
        self.right_down = False
        self.agent_mode = bool(agent_mode)
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self.command_queue = queue.Queue()
        self.output_lock = threading.Lock()
        self.agent_movement = "NONE"
        self.agent_movement_until = 0.0
        self.player_vx = 0.0
        self.snapshot_id = 0
        self.last_snapshot_time = -999.0
        self.active_burst = None
        self.burst_queue = []
        self.closed = False

        self.score = 0
        self.lives = INITIAL_LIVES
        self.sim_time = 0.0
        self.running = True

        self.player_x = WIDTH / 2
        self.player_w = 44.0
        self.player_h = 22.0

        self.bullets = []
        self.enemies = []
        self.last_fire_time = -999.0

        self.last_wall_time = time.perf_counter()

        self._build_ui()
        self.reset()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        if self.agent_mode:
            threading.Thread(target=self._read_agent_commands, daemon=True).start()
        self._tick()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Sarah/Astrid OODA Lab — Continuous-Time Human Test",
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text=(
                "Time runs continuously. Hold LEFT/RIGHT to move; release to stop. "
                "SPACE fires. Doing nothing does NOT stop the world."
            ),
            wraplength=780,
        ).pack(anchor="w", pady=(2, 8))

        hud = ttk.Frame(outer)
        hud.pack(fill="x", pady=(0, 8))

        self.score_var = tk.StringVar()
        self.lives_var = tk.StringVar()
        self.ammo_var = tk.StringVar()
        self.time_var = tk.StringVar()
        self.state_var = tk.StringVar()

        for label, var in [
            ("Score", self.score_var),
            ("Lives", self.lives_var),
            ("Ammo", self.ammo_var),
            ("Sim Time", self.time_var),
            ("Motion", self.state_var),
        ]:
            box = ttk.LabelFrame(hud, text=label, padding=6)
            box.pack(side="left", fill="x", expand=True, padx=3)
            ttk.Label(box, textvariable=var, font=("Segoe UI", 11, "bold")).pack()

        self.canvas = tk.Canvas(
            outer,
            width=WIDTH,
            height=HEIGHT,
            bg="#08111d",
            highlightthickness=1,
            highlightbackground="#263047",
        )
        self.canvas.pack()

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(10, 4))

        ttk.Button(
            controls,
            text="◀ LEFT",
            command=lambda: self._run_and_refocus(self._tap_left),
            takefocus=False,
        ).pack(side="left", fill="x", expand=True, padx=4, ipady=10)

        ttk.Button(
            controls,
            text="FIRE",
            command=lambda: self._run_and_refocus(self.fire),
            takefocus=False,
        ).pack(side="left", fill="x", expand=True, padx=4, ipady=10)

        ttk.Button(
            controls,
            text="RIGHT ▶",
            command=lambda: self._run_and_refocus(self._tap_right),
            takefocus=False,
        ).pack(side="left", fill="x", expand=True, padx=4, ipady=10)

        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(4, 0))

        ttk.Button(
            bottom,
            text="Restart",
            command=lambda: self._run_and_refocus(self.reset),
            takefocus=False,
        ).pack(side="left", fill="x", expand=True, padx=4)

        ttk.Button(
            bottom,
            text="Quit",
            command=self.root.destroy,
            takefocus=False,
        ).pack(side="left", fill="x", expand=True, padx=4)

        self.debug_var = tk.StringVar()
        ttk.Label(
            outer,
            textvariable=self.debug_var,
            font=("Consolas", 9),
        ).pack(anchor="w", pady=(8, 0))

        # Press/release bindings. Releasing the key stops lateral movement.
        self.root.bind("<KeyPress-Left>", lambda e: self._set_left(True))
        self.root.bind("<KeyRelease-Left>", lambda e: self._set_left(False))
        self.root.bind("<KeyPress-Right>", lambda e: self._set_right(True))
        self.root.bind("<KeyRelease-Right>", lambda e: self._set_right(False))

        self.root.bind("<KeyPress-a>", lambda e: self._set_left(True))
        self.root.bind("<KeyRelease-a>", lambda e: self._set_left(False))
        self.root.bind("<KeyPress-A>", lambda e: self._set_left(True))
        self.root.bind("<KeyRelease-A>", lambda e: self._set_left(False))

        self.root.bind("<KeyPress-d>", lambda e: self._set_right(True))
        self.root.bind("<KeyRelease-d>", lambda e: self._set_right(False))
        self.root.bind("<KeyPress-D>", lambda e: self._set_right(True))
        self.root.bind("<KeyRelease-D>", lambda e: self._set_right(False))

        self.root.bind("<KeyPress-space>", self._space_fire)

        # Keep gameplay focus on the canvas, never on a button.
        self.root.after_idle(self.canvas.focus_set)

    def _run_and_refocus(self, func):
        """Run a button command, then immediately return keyboard focus to gameplay."""
        func()
        self.root.after_idle(self.canvas.focus_set)

    def _space_fire(self, _event):
        """
        SPACE is exclusively the FIRE key.
        Returning "break" prevents Tk/ttk from treating SPACE as activation
        of any focused button such as Restart.
        """
        self.fire()
        return "break"

    def reset(self):
        if self.active_burst is not None or self.burst_queue:
            self._finish_all_bursts()
        self.score = 0
        self.lives = INITIAL_LIVES
        self.ammo_remaining = self.initial_ammo
        self.sim_time = 0.0
        self.running = True

        self.left_down = False
        self.right_down = False
        self.agent_movement = "NONE"
        self.agent_movement_until = 0.0
        self.player_vx = 0.0

        self.player_x = WIDTH / 2
        self.bullets = []
        self.enemies = []
        self.last_fire_time = -999.0
        self.active_burst = None
        self.burst_queue = []

        for i in range(TARGET_COUNT):
            self.enemies.append(self._new_enemy(f"T{i+1}", start_high=True))

        self.last_wall_time = time.perf_counter()
        self._update_hud()
        self._draw()
        self.root.after_idle(self.canvas.focus_set)
        if self.agent_mode:
            self._emit_snapshot(force=True)

    def _new_enemy(self, ident=None, start_high=False):
        size = self.rng.uniform(25, 38)
        x = self.rng.uniform(size / 2, WIDTH - size / 2)
        if self.agent_mode:
            # Slow agent-mode threats must remain visible to the observer.
            # Replacements enter just above the canvas instead of spending
            # many seconds travelling through an invisible approach.
            y = self.rng.uniform(20, 120) if start_high else self.rng.uniform(-20, 20)
        else:
            y = self.rng.uniform(-180, 120 if start_high else -20)
        vx = self.rng.uniform(-45, 45)
        vy = self.rng.uniform(self.enemy_min_speed, self.enemy_max_speed)
        if ident is None:
            ident = f"T{self.rng.randint(100,999)}"
        return Enemy(ident, x, y, vx, vy, size)

    def _set_left(self, down):
        self.left_down = down

    def _set_right(self, down):
        self.right_down = down

    def _tap_left(self):
        # Button click gives a small impulse for mouse debugging.
        self.player_x -= 35
        self.player_x = max(self.player_w/2, self.player_x)

    def _tap_right(self):
        self.player_x += 35
        self.player_x = min(WIDTH-self.player_w/2, self.player_x)

    def fire(self):
        if not self.running or self.ammo_remaining <= 0:
            return False

        # Fire affects only bullets. It does NOT affect the simulation clock.
        if self.sim_time - self.last_fire_time < FIRE_COOLDOWN:
            return False

        self.last_fire_time = self.sim_time
        self.ammo_remaining -= 1
        self.bullets.append(Bullet(self.player_x - 3, PLAYER_Y - 30))
        self.debug_var.set(
            f"FIRE at sim_time={self.sim_time:.2f}s — clock continues independently."
        )
        return True

    def _tick(self):
        now = time.perf_counter()
        dt = now - self.last_wall_time
        self.last_wall_time = now

        # Avoid giant jumps if the window was dragged/frozen by the OS.
        dt = max(0.0, min(dt, 0.05))

        self._drain_agent_commands()
        if self.closed:
            return
        if self.running:
            self._advance_world(dt)
            self._service_bursts()

        if self.agent_mode:
            self._emit_snapshot()

        self._update_hud()
        self._draw()
        self.root.after(FRAME_MS, self._tick)

    def _advance_world(self, dt):
        # THIS is the only place simulation time advances.
        # It is called every frame, regardless of user input.
        self.sim_time += dt

        direction = self._movement_direction()
        self.player_vx = direction * PLAYER_SPEED

        self.player_x += direction * PLAYER_SPEED * dt
        self.player_x = max(
            self.player_w / 2,
            min(WIDTH - self.player_w / 2, self.player_x)
        )

        # Bullets advance continuously.
        for b in self.bullets:
            b.y -= BULLET_SPEED * dt
        self.bullets = [b for b in self.bullets if b.y + b.h > 0]

        # Enemies advance continuously.
        for e in self.enemies:
            e.x += e.vx * dt
            e.y += e.vy * dt

            if e.x - e.size/2 < 0:
                e.x = e.size/2
                e.vx = abs(e.vx)
            elif e.x + e.size/2 > WIDTH:
                e.x = WIDTH - e.size/2
                e.vx = -abs(e.vx)

        self._resolve_collisions()
        self._resolve_penetrations()

        while len(self.enemies) < TARGET_COUNT:
            self.enemies.append(self._new_enemy())

    @staticmethod
    def _rect_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

    def _resolve_collisions(self):
        dead_bullets = set()
        dead_enemies = set()

        for bi, b in enumerate(self.bullets):
            bx1, by1 = b.x, b.y
            bx2, by2 = b.x + b.w, b.y + b.h

            for ei, e in enumerate(self.enemies):
                if bi in dead_bullets or ei in dead_enemies:
                    continue

                ex1 = e.x - e.size/2
                ey1 = e.y - e.size/2
                ex2 = e.x + e.size/2
                ey2 = e.y + e.size/2

                if self._rect_overlap(
                    bx1, by1, bx2, by2,
                    ex1, ey1, ex2, ey2
                ):
                    dead_bullets.add(bi)
                    dead_enemies.add(ei)
                    self.score += 10

        self.bullets = [
            b for i, b in enumerate(self.bullets)
            if i not in dead_bullets
        ]
        self.enemies = [
            e for i, e in enumerate(self.enemies)
            if i not in dead_enemies
        ]

    def _resolve_penetrations(self):
        survivors = []
        losses = 0

        for e in self.enemies:
            if e.y + e.size/2 >= PLAYER_Y - 10:
                losses += 1
            else:
                survivors.append(e)

        self.enemies = survivors

        if losses:
            self.lives -= losses
            if self.lives <= 0:
                self.lives = 0
                self.running = False
                self.player_vx = 0.0
                self._finish_all_bursts()
                self._emit_snapshot(force=True)
                self.debug_var.set(
                    f"GAME OVER at sim_time={self.sim_time:.2f}s"
                )

    def _update_hud(self):
        self.score_var.set(str(self.score))
        self.lives_var.set(str(self.lives))
        self.ammo_var.set(str(self.ammo_remaining))
        self.time_var.set(f"{self.sim_time:.1f} s")

        direction = self._movement_direction()
        if direction < 0:
            motion = "LEFT"
        elif direction > 0:
            motion = "RIGHT"
        else:
            motion = "STOPPED"
        self.state_var.set(motion)

    def _movement_direction(self):
        direction = 0
        if self.left_down:
            direction -= 1
        if self.right_down:
            direction += 1
        if direction or not self.agent_mode:
            return direction
        if self.sim_time >= self.agent_movement_until:
            self.agent_movement = "NONE"
            return 0
        if self.agent_movement == "LEFT":
            if self.player_x <= AGENT_EDGE_MARGIN:
                self.agent_movement = "NONE"
                return 0
            return -1
        if self.agent_movement == "RIGHT":
            if self.player_x >= WIDTH - AGENT_EDGE_MARGIN:
                self.agent_movement = "NONE"
                return 0
            return 1
        return 0

    def _read_agent_commands(self):
        for line in self.input_stream:
            try:
                payload = json.loads(line)
            except (TypeError, ValueError) as exc:
                print(f"Invalid command JSON: {exc}", file=sys.stderr, flush=True)
                continue
            self.command_queue.put(payload)

    def _drain_agent_commands(self):
        while True:
            try:
                payload = self.command_queue.get_nowait()
            except queue.Empty:
                return
            if payload.get("type") == "close":
                self._close()
                return
            if payload.get("type") == "command":
                self._apply_agent_command(payload)

    def _apply_agent_command(self, payload):
        movement = str(payload.get("movement", "")).upper()
        target = payload.get("target")
        fire_rounds = payload.get("fire_rounds")
        if movement not in {"LEFT", "RIGHT", "NONE"}:
            self._emit_json({"type": "command_rejected", "reason": "invalid movement"})
            return
        if isinstance(fire_rounds, bool) or not isinstance(fire_rounds, int) or fire_rounds < 0:
            self._emit_json({"type": "command_rejected", "reason": "invalid fire_rounds"})
            return
        if target is not None and not isinstance(target, str):
            self._emit_json({"type": "command_rejected", "reason": "invalid target"})
            return

        movement, target = self._greedy_agent_engagement(target)
        self.agent_movement = movement
        self.agent_movement_until = (
            self.sim_time + AGENT_MOVEMENT_PULSE if movement != "NONE" else self.sim_time
        )
        if self.active_burst is not None or self.burst_queue:
            self._finish_all_bursts()
        authorized_fire_rounds = fire_rounds
        if target is not None and authorized_fire_rounds == 0:
            authorized_fire_rounds = 1
        burst = {
            "command_id": str(payload.get("command_id", "")),
            "based_on_snapshot": payload.get("based_on_snapshot"),
            "requested_fire_rounds": authorized_fire_rounds,
            "remaining": min(authorized_fire_rounds, self.ammo_remaining),
            "actual_rounds_fired": 0,
            "applied_game_time": self.sim_time,
            "expires_at": self.sim_time + AGENT_FIRE_WINDOW,
            "target": target,
            "movement": movement,
        }
        if not self.running or burst["remaining"] == 0:
            self._emit_command_result(burst)
            return
        self.burst_queue.append(burst)

    def _greedy_agent_engagement(self, requested_target):
        if not self.enemies:
            return "NONE", None

        attackers, clusters = build_attacker_tracking(
            self.enemies, self.player_x, self.player_w
        )
        enemies_by_id = {enemy.ident: enemy for enemy in self.enemies}
        cluster_by_id = {cluster["id"]: cluster for cluster in clusters}

        engagement_target = None
        aim_x = None
        if requested_target in enemies_by_id:
            engagement_target = requested_target
            aim_x = enemies_by_id[requested_target].x
        elif requested_target in cluster_by_id:
            engagement_target = requested_target
            aim_x = cluster_by_id[requested_target]["center_x"]
        else:
            nearby_clusters = [
                cluster
                for cluster in clusters
                if abs(cluster["center_x"] - self.player_x) <= AGENT_NEARBY_CLUSTER_RANGE
            ]
            individual = min(
                attackers,
                key=lambda attacker: (
                    attacker["time_to_defender_line"]
                    if attacker["time_to_defender_line"] is not None
                    else float("inf"),
                    abs(attacker["x"] - self.player_x),
                ),
            )
            if nearby_clusters:
                cluster = min(
                    nearby_clusters,
                    key=lambda candidate: math.hypot(
                        candidate["center_x"] - self.player_x,
                        PLAYER_Y - candidate["center_y"],
                    ) / candidate["count"],
                )
                cluster_distance = math.hypot(
                    cluster["center_x"] - self.player_x,
                    PLAYER_Y - cluster["center_y"],
                ) / cluster["count"]
                individual_distance = math.hypot(
                    individual["x"] - self.player_x,
                    PLAYER_Y - individual["y"],
                )
                if cluster_distance <= individual_distance:
                    engagement_target = cluster["id"]
                    aim_x = cluster["center_x"]
            if aim_x is None:
                engagement_target = individual["id"]
                aim_x = individual["x"]

        delta = aim_x - self.player_x
        if abs(delta) <= AGENT_AIM_DEADBAND:
            return "NONE", engagement_target
        return ("RIGHT" if delta > 0 else "LEFT"), engagement_target

    def _target_has_fire_solution(self, target):
        candidates = self.enemies
        if target is not None:
            if target.startswith("T"):
                candidates = [enemy for enemy in self.enemies if enemy.ident == target]
            elif target.startswith("C"):
                attackers, _clusters = build_attacker_tracking(
                    self.enemies, self.player_x, self.player_w
                )
                member_ids = {
                    attacker["id"]
                    for attacker in attackers
                    if attacker.get("cluster_id") == target
                }
                candidates = [enemy for enemy in self.enemies if enemy.ident in member_ids]
            else:
                candidates = []
        if any(_enemy_has_fire_solution(enemy, self.player_x) for enemy in candidates):
            return True
        return any(_enemy_has_fire_solution(enemy, self.player_x) for enemy in self.enemies)

    def _service_bursts(self):
        if self.active_burst is None and self.burst_queue:
            self.active_burst = self.burst_queue.pop(0)
        if self.active_burst is None:
            return
        if self.sim_time >= self.active_burst["expires_at"]:
            self._finish_active_burst()
            return
        if self.ammo_remaining <= 0 or self.active_burst["remaining"] <= 0:
            self._finish_active_burst()
            return
        if not self._target_has_fire_solution(self.active_burst["target"]):
            return
        if self.fire():
            self.active_burst["remaining"] -= 1
            self.active_burst["actual_rounds_fired"] += 1
            if self.active_burst["remaining"] <= 0 or self.ammo_remaining <= 0:
                self._finish_active_burst()

    def _finish_active_burst(self):
        if self.active_burst is None:
            return
        self._emit_command_result(self.active_burst)
        self.active_burst = None

    def _finish_all_bursts(self):
        self._finish_active_burst()
        while self.burst_queue:
            self._emit_command_result(self.burst_queue.pop(0))

    def _emit_command_result(self, burst):
        self._emit_json(
            {
                "type": "command_result",
                "command_id": burst["command_id"],
                "based_on_snapshot": burst["based_on_snapshot"],
                "command_applied_game_time": round(burst["applied_game_time"], 3),
                "movement": burst["movement"],
                "target": burst["target"],
                "requested_fire_rounds": burst["requested_fire_rounds"],
                "actual_rounds_fired": burst["actual_rounds_fired"],
                "ammo_remaining": self.ammo_remaining,
                "score": self.score,
                "lives": self.lives,
            }
        )

    def _emit_snapshot(self, force=False):
        if not force and self.sim_time - self.last_snapshot_time < SNAPSHOT_INTERVAL:
            return
        self.last_snapshot_time = self.sim_time
        self.snapshot_id += 1
        attackers, clusters = build_attacker_tracking(self.enemies, self.player_x, self.player_w)
        self._emit_json(
            {
                "type": "snapshot",
                "snapshot_id": self.snapshot_id,
                "game_time": round(self.sim_time, 3),
                "running": self.running,
                "blue": {
                    "x": round(self.player_x, 3),
                    "y": float(PLAYER_Y),
                    "vx": round(self.player_vx, 3),
                    "ammo_remaining": self.ammo_remaining,
                    "center_offset": round(self.player_x - WIDTH / 2, 3),
                    "distance_to_left_edge": round(self.player_x - self.player_w / 2, 3),
                    "distance_to_right_edge": round(WIDTH - (self.player_x + self.player_w / 2), 3),
                },
                "attackers": attackers,
                "clusters": clusters,
                "score": self.score,
                "lives": self.lives,
            }
        )

    def _emit_json(self, payload):
        if not self.agent_mode:
            return
        with self.output_lock:
            print(json.dumps(payload, separators=(",", ":")), file=self.output_stream, flush=True)

    def _close(self):
        if self.closed:
            return
        self.closed = True
        self._finish_all_bursts()
        self.root.destroy()

    def _draw(self):
        c = self.canvas
        c.delete("all")

        # Grid
        for x in range(0, WIDTH, 40):
            c.create_line(x, 0, x, HEIGHT, fill="#142033")
        for y in range(0, HEIGHT, 40):
            c.create_line(0, y, WIDTH, y, fill="#142033")

        c.create_line(
            0, PLAYER_Y - 10, WIDTH, PLAYER_Y - 10,
            fill="#7a3943", dash=(8, 8)
        )

        # Defender
        x = self.player_x
        c.create_polygon(
            x, PLAYER_Y - 22,
            x + 22, PLAYER_Y + 10,
            x - 22, PLAYER_Y + 10,
            fill="#79c7ff",
            outline=""
        )

        # Bullets
        for b in self.bullets:
            c.create_rectangle(
                b.x, b.y, b.x+b.w, b.y+b.h,
                fill="#ffd166", outline=""
            )

        # Targets
        for e in self.enemies:
            s = e.size
            c.create_rectangle(
                e.x-s/2, e.y-s/2,
                e.x+s/2, e.y+s/2,
                fill="#ff7c7c", outline=""
            )
            c.create_text(
                e.x, e.y-s/2-10,
                text=e.ident, fill="white",
                font=("Segoe UI", 8)
            )

        if not self.running:
            c.create_rectangle(
                0, 0, WIDTH, HEIGHT,
                fill="#000000", stipple="gray50", outline=""
            )
            c.create_text(
                WIDTH/2, HEIGHT/2,
                text="GAME OVER",
                fill="white",
                font=("Segoe UI", 28, "bold")
            )


def build_attacker_tracking(enemies, player_x, player_w):
    ordered = sorted(enemies, key=lambda enemy: enemy.ident)
    unvisited = set(range(len(ordered)))
    components = []
    while unvisited:
        seed = unvisited.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            nearby = {
                other
                for other in unvisited
                if math.hypot(
                    ordered[current].x - ordered[other].x,
                    ordered[current].y - ordered[other].y,
                )
                <= CLUSTER_DISTANCE
            }
            component.update(nearby)
            frontier.extend(nearby)
            unvisited.difference_update(nearby)
        components.append(sorted(component))

    cluster_by_enemy = {}
    clusters = []
    cluster_number = 1
    for component in components:
        if len(component) < 2:
            continue
        cluster_id = f"C{cluster_number}"
        cluster_number += 1
        members = [ordered[index] for index in component]
        for enemy in members:
            cluster_by_enemy[enemy.ident] = cluster_id
        clusters.append(
            {
                "id": cluster_id,
                "members": [enemy.ident for enemy in members],
                "count": len(members),
                "center_x": round(sum(enemy.x for enemy in members) / len(members), 3),
                "center_y": round(sum(enemy.y for enemy in members) / len(members), 3),
            }
        )

    attackers = []
    for enemy in ordered:
        distance_to_line = max(0.0, (PLAYER_Y - 10) - (enemy.y + enemy.size / 2))
        time_to_line = distance_to_line / enemy.vy if enemy.vy > 0 else None
        predicted_x = (
            _project_bouncing_x(enemy.x, enemy.vx, time_to_line, enemy.size)
            if time_to_line is not None
            else enemy.x
        )
        bullet_intercept_time = _bullet_intercept_time(enemy)
        bullet_intercept_x = (
            _project_bouncing_x(enemy.x, enemy.vx, bullet_intercept_time, enemy.size)
            if bullet_intercept_time is not None
            else enemy.x
        )
        aligned = abs(enemy.x - player_x) <= max(player_w / 2, enemy.size / 2) + ALIGNMENT_MARGIN
        attackers.append(
            {
                "id": enemy.ident,
                "x": round(enemy.x, 3),
                "y": round(enemy.y, 3),
                "vx": round(enemy.vx, 3),
                "vy": round(enemy.vy, 3),
                "time_to_defender_line": round(time_to_line, 3) if time_to_line is not None else None,
                "predicted_x_at_defender_line": round(predicted_x, 3),
                "predicted_x_at_bullet_intercept": round(bullet_intercept_x, 3),
                "fire_solution_aligned": _enemy_has_fire_solution(enemy, player_x),
                "aligned_with_blue": aligned,
                "cluster_id": cluster_by_enemy.get(enemy.ident),
            }
        )
    return attackers, clusters


def _project_bouncing_x(x, vx, duration, size):
    minimum = size / 2
    maximum = WIDTH - size / 2
    span = maximum - minimum
    if span <= 0:
        return WIDTH / 2
    offset = (x - minimum + vx * duration) % (2 * span)
    if offset > span:
        offset = 2 * span - offset
    return minimum + offset


def _bullet_intercept_time(enemy):
    vertical_distance = (PLAYER_Y - 30) - enemy.y
    closing_speed = BULLET_SPEED + enemy.vy
    if vertical_distance <= 0 or closing_speed <= 0:
        return None
    return vertical_distance / closing_speed


def _enemy_has_fire_solution(enemy, player_x):
    intercept_time = _bullet_intercept_time(enemy)
    if intercept_time is None:
        return False
    intercept_x = _project_bouncing_x(enemy.x, enemy.vx, intercept_time, enemy.size)
    tolerance = enemy.size / 2 + 3.0
    return abs(intercept_x - player_x) <= tolerance


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Sarah/Astrid continuous-time OODA game.")
    parser.add_argument("--agent-mode", action="store_true", help="Enable JSONL telemetry and commands on stdio.")
    parser.add_argument("--enemy-min-speed", type=float, default=ENEMY_MIN_SPEED)
    parser.add_argument("--enemy-max-speed", type=float, default=ENEMY_MAX_SPEED)
    parser.add_argument("--initial-ammo", type=int, default=INITIAL_AMMO)
    args = parser.parse_args(argv)
    root = tk.Tk()
    ContinuousOODAGame(
        root,
        agent_mode=args.agent_mode,
        enemy_min_speed=args.enemy_min_speed,
        enemy_max_speed=args.enemy_max_speed,
        initial_ammo=args.initial_ammo,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
