"""
Maze Runner — Hand Tracking Edition
Move your finger to guide the runner from start to goal.
"""

import sys
import os
import random
import time

import cv2
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_tracker import HandTracker


SCREEN_W, SCREEN_H = 900, 640
FPS = 60
CAM_W, CAM_H = 640, 480

MAZE_Y = 72
LEVEL_CONFIGS = [
    {"grid_w": 19, "grid_h": 13, "tile": 36, "key_speed": 2.0, "hand_speed": 2.6, "deadzone": 34, "time": 52},
    {"grid_w": 21, "grid_h": 15, "tile": 33, "key_speed": 2.15, "hand_speed": 2.75, "deadzone": 32, "time": 50},
    {"grid_w": 23, "grid_h": 15, "tile": 30, "key_speed": 2.3, "hand_speed": 2.95, "deadzone": 30, "time": 48},
    {"grid_w": 25, "grid_h": 17, "tile": 28, "key_speed": 2.45, "hand_speed": 3.1, "deadzone": 28, "time": 46},
    {"grid_w": 27, "grid_h": 17, "tile": 26, "key_speed": 2.6, "hand_speed": 3.25, "deadzone": 26, "time": 44},
]

BG = (20, 20, 34)
WALL = (60, 70, 110)
PATH = (228, 232, 244)
START = (120, 220, 150)
GOAL = (240, 120, 120)
PLAYER = (80, 170, 255)
WHITE = (245, 245, 252)


class MazeRunnerGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🧭 Maze Runner — Hand Tracking")
        self.clock = pygame.time.Clock()

        self.font_big = pygame.font.SysFont("Segoe UI", 44, bold=True)
        self.font_mid = pygame.font.SysFont("Segoe UI", 28, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 20)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ready = self.cap.isOpened()
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)

        self.cam_surface = None
        self.finger_pos = None
        self.max_level = len(LEVEL_CONFIGS)
        self.reset_campaign()

    def setup_level(self, level, time_bonus=0.0):
        cfg = LEVEL_CONFIGS[level - 1]
        self.level = level
        self.grid_w = cfg["grid_w"]
        self.grid_h = cfg["grid_h"]
        self.tile = cfg["tile"]
        self.key_speed = cfg["key_speed"]
        self.hand_speed = cfg["hand_speed"]
        self.hand_deadzone = cfg["deadzone"]
        self.time_limit = float(cfg["time"])
        self.time_left = min(self.time_limit + 12.0, self.time_limit + float(time_bonus))
        self.last_time_tick = time.time()
        self.maze_x = (SCREEN_W - self.grid_w * self.tile) // 2

        self.maze = [[1 for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        self.generate_maze()
        self.start = (1, 1)
        self.goal = (self.grid_h - 2, self.grid_w - 2)
        self.maze[self.start[0]][self.start[1]] = 0
        self.maze[self.goal[0]][self.goal[1]] = 0

        self.player_x = self.start[1] * self.tile + self.tile // 2
        self.player_y = self.start[0] * self.tile + self.tile // 2
        self.player_radius = max(6, self.tile // 3)

        self.level_cleared = False

    def reset_campaign(self):
        self.completed = False
        self.failed = False
        self.failure_reason = ""
        self.transition_until = 0.0
        self.next_level_bonus = 0.0
        self.setup_level(1)

    def generate_maze(self):
        stack = [(1, 1)]
        self.maze[1][1] = 0
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]

        while stack:
            r, c = stack[-1]
            random.shuffle(dirs)
            moved = False
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 1 <= nr < self.grid_h - 1 and 1 <= nc < self.grid_w - 1 and self.maze[nr][nc] == 1:
                    self.maze[nr][nc] = 0
                    self.maze[r + dr // 2][c + dc // 2] = 0
                    stack.append((nr, nc))
                    moved = True
                    break
            if not moved:
                stack.pop()

    def is_walkable(self, x, y):
        gx = int(x // self.tile)
        gy = int(y // self.tile)
        if not (0 <= gy < self.grid_h and 0 <= gx < self.grid_w):
            return False
        return self.maze[gy][gx] == 0

    def is_walkable_circle(self, x, y):
        r = self.player_radius
        return (
            self.is_walkable(x - r, y)
            and self.is_walkable(x + r, y)
            and self.is_walkable(x, y - r)
            and self.is_walkable(x, y + r)
        )

    def move_player(self, vx, vy):
        new_x = self.player_x + vx
        new_y = self.player_y + vy

        if self.is_walkable_circle(new_x, self.player_y):
            self.player_x = new_x
        if self.is_walkable_circle(self.player_x, new_y):
            self.player_y = new_y

        gx = int(self.player_x // self.tile)
        gy = int(self.player_y // self.tile)
        if (gy, gx) == self.goal:
            self.level_cleared = True
            self.next_level_bonus = min(10.0, 2.0 + self.time_left * 0.25)
            self.transition_until = time.time() + 1.1

    def update_timer(self):
        if self.completed or self.failed or self.level_cleared:
            return
        now = time.time()
        dt = now - self.last_time_tick
        self.last_time_tick = now
        self.time_left = max(0.0, self.time_left - dt)
        if self.time_left <= 0:
            self.failed = True
            self.failure_reason = "Time up!"

    def update_level_transition(self):
        if not self.level_cleared or time.time() < self.transition_until:
            return

        if self.level < self.max_level:
            bonus = self.next_level_bonus
            self.next_level_bonus = 0.0
            self.setup_level(self.level + 1, time_bonus=bonus)
            return

        self.level_cleared = False
        self.completed = True

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (130, 96))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw(self):
        self.screen.fill(BG)

        title = self.font_mid.render("Maze Runner", True, WHITE)
        self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 18))

        hud = self.font_small.render(
            f"Level: {self.level}/{self.max_level}   Time: {int(self.time_left)}s",
            True,
            WHITE,
        )
        self.screen.blit(hud, (28, 22))

        for r in range(self.grid_h):
            for c in range(self.grid_w):
                x = self.maze_x + c * self.tile
                y = MAZE_Y + r * self.tile
                color = WALL if self.maze[r][c] == 1 else PATH
                pygame.draw.rect(self.screen, color, (x, y, self.tile, self.tile))

        sx, sy = self.start[1], self.start[0]
        gx, gy = self.goal[1], self.goal[0]
        pygame.draw.rect(
            self.screen,
            START,
            (self.maze_x + sx * self.tile + 4, MAZE_Y + sy * self.tile + 4, self.tile - 8, self.tile - 8),
        )
        pygame.draw.rect(
            self.screen,
            GOAL,
            (self.maze_x + gx * self.tile + 4, MAZE_Y + gy * self.tile + 4, self.tile - 8, self.tile - 8),
        )

        px = self.maze_x + int(self.player_x)
        py = MAZE_Y + int(self.player_y)
        pygame.draw.circle(self.screen, PLAYER, (px, py), self.player_radius)

        if self.finger_pos:
            pygame.draw.circle(self.screen, (150, 220, 255), self.finger_pos, 12, 2)

        if self.completed or self.failed or self.level_cleared:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            self.screen.blit(overlay, (0, 0))
            msg_text = "Level Cleared!"
            msg_color = START
            if self.failed:
                msg_text = self.failure_reason or "Failed"
                msg_color = GOAL
            elif self.completed:
                msg_text = "All Levels Cleared!"

            msg = self.font_big.render(msg_text, True, msg_color)
            self.screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, SCREEN_H // 2 - 70))
            hint_text = "Preparing next level..."
            if self.failed or self.completed:
                hint_text = "Press R to restart  |  ESC to quit"
            hint = self.font_small.render(hint_text, True, WHITE)
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H // 2 + 10))
        else:
            hint = "Move finger to steer  •  Arrow keys also work"
            if not self.camera_ready:
                hint = "Camera unavailable - use arrow keys"
            htxt = self.font_small.render(hint, True, (200, 200, 220))
            self.screen.blit(htxt, (SCREEN_W // 2 - htxt.get_width() // 2, SCREEN_H - 30))

        if self.cam_surface:
            self.screen.blit(self.cam_surface, (SCREEN_W - 142, SCREEN_H - 108))
            pygame.draw.rect(self.screen, (90, 90, 130), (SCREEN_W - 142, SCREEN_H - 108, 130, 96), 2)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            keys = pygame.key.get_pressed()
            vx = vy = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_campaign()

            if keys[pygame.K_LEFT]:
                vx -= self.key_speed
            if keys[pygame.K_RIGHT]:
                vx += self.key_speed
            if keys[pygame.K_UP]:
                vy -= self.key_speed
            if keys[pygame.K_DOWN]:
                vy += self.key_speed

            ret, frame = (False, None)
            if self.camera_ready:
                ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                self.tracker.process_frame(frame)
                self.tracker.draw_landmarks(frame)
                self.draw_cam_overlay(frame)

                self.finger_pos = self.tracker.get_fingertip_pos("index", SCREEN_W, SCREEN_H)
                if self.finger_pos and not (self.completed or self.failed or self.level_cleared):
                    tx = self.finger_pos[0] - (self.maze_x + self.player_x)
                    ty = self.finger_pos[1] - (MAZE_Y + self.player_y)
                    mag = max(1.0, (tx * tx + ty * ty) ** 0.5)
                    if mag > self.hand_deadzone:
                        strength = min(1.0, (mag - self.hand_deadzone) / 130.0)
                        vx += tx / mag * self.hand_speed * strength
                        vy += ty / mag * self.hand_speed * strength
            else:
                self.finger_pos = None
                self.tracker.hand_detected = False

            self.update_timer()
            self.update_level_transition()

            if not (self.completed or self.failed or self.level_cleared):
                self.move_player(vx, vy)

            self.draw()
            self.clock.tick(FPS)

        self.cap.release()
        self.tracker.release()
        pygame.quit()


def main():
    game = MazeRunnerGame()
    game.run()


if __name__ == "__main__":
    main()
