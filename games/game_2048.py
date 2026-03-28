"""
2048 — Hand Tracking Edition
Use swipe gestures (or arrow keys) to move and merge tiles.
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


SCREEN_W, SCREEN_H = 700, 700
FPS = 60
CAM_W, CAM_H = 640, 480

BG = (32, 28, 24)
BOARD_BG = (60, 52, 46)
EMPTY = (92, 84, 78)
WHITE = (250, 250, 245)

GRID = 4
CELL = 125
GAP = 12
BOARD_SIZE = GRID * CELL + (GRID - 1) * GAP
BOARD_X = (SCREEN_W - BOARD_SIZE) // 2
BOARD_Y = 120

TILE_COLORS = {
    0: EMPTY,
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

SWIPE_THRESHOLD = 64
AXIS_LOCK_RATIO = 1.35
MOVE_COOLDOWN = 0.15
LEVEL_TARGETS = [256, 512, 1024, 2048]


class Game2048:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🔢 2048 — Hand Tracking")
        self.clock = pygame.time.Clock()

        self.font_big = pygame.font.SysFont("Segoe UI", 48, bold=True)
        self.font_tile = pygame.font.SysFont("Segoe UI", 38, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 20)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ready = self.cap.isOpened()
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)

        self.cam_surface = None
        self.finger_pos = None
        self.swipe_anchor = None
        self.cooldown_until = 0.0
        self.max_level = len(LEVEL_TARGETS)
        self.reset_campaign()

    def setup_level(self, level):
        self.level = level
        self.target_tile = LEVEL_TARGETS[level - 1]
        self.grid = [[0] * GRID for _ in range(GRID)]
        self.game_over = False
        self.win = False
        self.level_cleared = False
        self.level_transition_until = 0.0
        self.add_tile()
        self.add_tile()

    def reset_campaign(self):
        self.score = 0
        self.campaign_complete = False
        self.setup_level(1)

    def reset(self):
        self.reset_campaign()

    def add_tile(self):
        empty = [(r, c) for r in range(GRID) for c in range(GRID) if self.grid[r][c] == 0]
        if not empty:
            return
        r, c = random.choice(empty)
        self.grid[r][c] = 4 if random.random() < 0.1 else 2

    def compress(self, row):
        nums = [v for v in row if v != 0]
        out = []
        i = 0
        while i < len(nums):
            if i + 1 < len(nums) and nums[i] == nums[i + 1]:
                merged = nums[i] * 2
                self.score += merged
                out.append(merged)
                if merged >= self.target_tile:
                    self.level_cleared = True
                i += 2
            else:
                out.append(nums[i])
                i += 1
        out += [0] * (GRID - len(out))
        return out

    def move_left(self):
        moved = False
        new_grid = []
        for row in self.grid:
            comp = self.compress(row)
            if comp != row:
                moved = True
            new_grid.append(comp)
        self.grid = new_grid
        return moved

    def move_right(self):
        self.grid = [list(reversed(row)) for row in self.grid]
        moved = self.move_left()
        self.grid = [list(reversed(row)) for row in self.grid]
        return moved

    def transpose(self):
        self.grid = [list(row) for row in zip(*self.grid)]

    def move_up(self):
        self.transpose()
        moved = self.move_left()
        self.transpose()
        return moved

    def move_down(self):
        self.transpose()
        moved = self.move_right()
        self.transpose()
        return moved

    def can_move(self):
        for r in range(GRID):
            for c in range(GRID):
                v = self.grid[r][c]
                if v == 0:
                    return True
                if r + 1 < GRID and self.grid[r + 1][c] == v:
                    return True
                if c + 1 < GRID and self.grid[r][c + 1] == v:
                    return True
        return False

    def best_tile(self):
        return max(max(row) for row in self.grid)

    def apply_move(self, direction):
        if self.game_over or self.campaign_complete or self.level_cleared:
            return
        moved = False
        if direction == "left":
            moved = self.move_left()
        elif direction == "right":
            moved = self.move_right()
        elif direction == "up":
            moved = self.move_up()
        elif direction == "down":
            moved = self.move_down()

        if moved:
            self.add_tile()
            if not self.can_move():
                self.game_over = True
            elif self.level_cleared:
                if self.level < self.max_level:
                    self.level_transition_until = time.time() + 1.0
                else:
                    self.campaign_complete = True
                    self.win = True

    def update_level_transition(self):
        if self.level_transition_until and time.time() >= self.level_transition_until:
            self.setup_level(self.level + 1)

    def detect_swipe(self):
        """Detect axis-aligned swipes using hysteresis: swipe needs minimum distance
        and must be clearly in one direction (not diagonal)."""
        now = time.time()
        if now < self.cooldown_until:
            return None
        if self.finger_pos is None:
            self.swipe_anchor = None
            return None

        if self.swipe_anchor is None:
            self.swipe_anchor = self.finger_pos
            return None

        dx = self.finger_pos[0] - self.swipe_anchor[0]
        dy = self.finger_pos[1] - self.swipe_anchor[1]
        dist_x = abs(dx)
        dist_y = abs(dy)
        total_dist = max(dist_x, dist_y)

        if total_dist < SWIPE_THRESHOLD:
            return None

        detected_direction = None
        if dist_x > dist_y:
            if dist_x > dist_y * AXIS_LOCK_RATIO:
                detected_direction = "right" if dx > 0 else "left"
        else:
            if dist_y > dist_x * AXIS_LOCK_RATIO:
                detected_direction = "down" if dy > 0 else "up"

        if detected_direction is None:
            return None

        self.cooldown_until = now + MOVE_COOLDOWN
        self.swipe_anchor = self.finger_pos
        return detected_direction

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (120, 90))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw(self):
        self.screen.fill(BG)
        title = self.font_big.render(f"2048  •  Level {self.level}/{self.max_level}", True, WHITE)
        self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 18))
        score = self.font_small.render(
            f"Score: {self.score}   Target: {self.target_tile}   Best Tile: {self.best_tile()}",
            True,
            WHITE,
        )
        self.screen.blit(score, (BOARD_X, 88))

        pygame.draw.rect(self.screen, BOARD_BG, (BOARD_X - 10, BOARD_Y - 10, BOARD_SIZE + 20, BOARD_SIZE + 20), border_radius=12)

        for r in range(GRID):
            for c in range(GRID):
                x = BOARD_X + c * (CELL + GAP)
                y = BOARD_Y + r * (CELL + GAP)
                val = self.grid[r][c]
                color = TILE_COLORS.get(val, TILE_COLORS[2048])
                pygame.draw.rect(self.screen, color, (x, y, CELL, CELL), border_radius=8)
                if val:
                    text_color = (60, 50, 40) if val <= 4 else WHITE
                    txt = self.font_tile.render(str(val), True, text_color)
                    self.screen.blit(txt, (x + CELL // 2 - txt.get_width() // 2, y + CELL // 2 - txt.get_height() // 2))

        hint = "Swipe with hand (or arrow keys)  •  R restart  •  ESC quit"
        if not self.camera_ready:
            hint = "Camera unavailable - use arrow keys  •  R restart  •  ESC quit"
        if self.game_over:
            hint = "N retry level  •  R restart campaign  •  ESC quit"
        htxt = self.font_small.render(hint, True, (210, 210, 220))
        self.screen.blit(htxt, (SCREEN_W // 2 - htxt.get_width() // 2, SCREEN_H - 30))

        if self.cam_surface:
            self.screen.blit(self.cam_surface, (SCREEN_W - 132, 12))
            pygame.draw.rect(self.screen, (110, 110, 135), (SCREEN_W - 132, 12, 120, 90), 2)

        if self.level_cleared and not self.campaign_complete:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (0, 0))
            txt = self.font_big.render("Level Cleared!", True, WHITE)
            self.screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2, SCREEN_H // 2 - 50))
            hint2 = self.font_small.render("Loading next level...", True, (210, 210, 220))
            self.screen.blit(hint2, (SCREEN_W // 2 - hint2.get_width() // 2, SCREEN_H // 2 + 12))

        if self.game_over or self.campaign_complete:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "All Levels Cleared!" if self.campaign_complete else "No moves left"
            txt = self.font_big.render(msg, True, WHITE)
            self.screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2, SCREEN_H // 2 - 60))
            hint2 = self.font_small.render("Press R to restart campaign", True, (210, 210, 220))
            self.screen.blit(hint2, (SCREEN_W // 2 - hint2.get_width() // 2, SCREEN_H // 2 + 10))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_n and self.game_over:
                        self.setup_level(self.level)
                    elif event.key == pygame.K_LEFT:
                        self.apply_move("left")
                    elif event.key == pygame.K_RIGHT:
                        self.apply_move("right")
                    elif event.key == pygame.K_UP:
                        self.apply_move("up")
                    elif event.key == pygame.K_DOWN:
                        self.apply_move("down")

            ret, frame = (False, None)
            if self.camera_ready:
                ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                self.tracker.process_frame(frame)
                self.tracker.draw_landmarks(frame)
                self.draw_cam_overlay(frame)

                self.finger_pos = self.tracker.get_fingertip_pos("index", SCREEN_W, SCREEN_H)
                swipe = self.detect_swipe()
                if swipe is not None:
                    self.apply_move(swipe)
            else:
                self.finger_pos = None
                self.swipe_anchor = None
                self.tracker.hand_detected = False

            self.update_level_transition()
            self.draw()
            self.clock.tick(FPS)

        self.cap.release()
        self.tracker.release()
        pygame.quit()


def main():
    game = Game2048()
    game.run()


if __name__ == "__main__":
    main()
