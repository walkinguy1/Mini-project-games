"""
Whack-a-Mole — Hand Tracking Edition
Point your index finger at moles to whack them! 60-second timed mode.
"""

import sys
import os
import math
import random
import time

import cv2
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_tracker import HandTracker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 700, 600
FPS = 60
CAM_W, CAM_H = 640, 480

# Colors
BG_COLOR = (45, 120, 60)
BG_DARK = (30, 80, 40)
HOLE_COLOR = (40, 30, 20)
HOLE_RIM = (60, 45, 30)
MOLE_BODY = (100, 70, 40)
MOLE_FACE = (130, 95, 55)
MOLE_NOSE = (220, 100, 100)
MOLE_EYE = (20, 20, 20)
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
RED = (255, 80, 80)
GREEN = (100, 255, 120)
DARK_TEXT = (30, 30, 30)

# Grid
GRID_COLS = 3
GRID_ROWS = 3
HOLE_RADIUS = 45
GRID_OFFSET_X = 130
GRID_OFFSET_Y = 150
GRID_SPACING_X = 180
GRID_SPACING_Y = 140

# Timing
GAME_DURATION = 60  # seconds
MOLE_SHOW_MIN = 0.6
MOLE_SHOW_MAX = 1.8
MOLE_COOLDOWN = 0.3
WHACK_DISTANCE = 55

# Difficulty scaling
INITIAL_ACTIVE = 1
MAX_ACTIVE = 4
PINCH_WHACK_DIST = 45
COMBO_RESET_SECS = 1.2


class Mole:
    """A mole that pops up from a hole."""

    def __init__(self, row, col, cx, cy):
        self.row = row
        self.col = col
        self.cx = cx
        self.cy = cy
        self.active = False
        self.show_time = 0
        self.duration = 0
        self.whacked = False
        self.whack_anim = 0  # 0 to 1 animation
        self.cooldown_until = 0

    def pop_up(self, duration=None):
        if duration is None:
            duration = random.uniform(MOLE_SHOW_MIN, MOLE_SHOW_MAX)
        self.active = True
        self.show_time = time.time()
        self.duration = duration
        self.whacked = False
        self.whack_anim = 0

    def update(self):
        now = time.time()
        if self.active:
            elapsed = now - self.show_time
            if self.whacked:
                self.whack_anim += 0.1
                if self.whack_anim >= 1.0:
                    self.active = False
                    self.cooldown_until = now + MOLE_COOLDOWN
            elif elapsed >= self.duration:
                self.active = False
                self.cooldown_until = now + MOLE_COOLDOWN

    def can_pop(self):
        return not self.active and time.time() >= self.cooldown_until

    def try_whack(self, px, py):
        """Check if position (px, py) hits this mole."""
        if not self.active or self.whacked:
            return False
        dist = math.sqrt((px - self.cx) ** 2 + (py - self.cy) ** 2)
        if dist < WHACK_DISTANCE:
            self.whacked = True
            return True
        return False

    def draw(self, surface):
        # Hole (always visible)
        pygame.draw.ellipse(
            surface, HOLE_RIM,
            (self.cx - HOLE_RADIUS, self.cy - 10, HOLE_RADIUS * 2, 30),
        )
        pygame.draw.ellipse(
            surface, HOLE_COLOR,
            (self.cx - HOLE_RADIUS + 4, self.cy - 6, HOLE_RADIUS * 2 - 8, 22),
        )

        if not self.active:
            return

        # Mole pop-up offset
        if self.whacked:
            offset = int(30 * (1 - self.whack_anim))
        else:
            elapsed = time.time() - self.show_time
            # Pop-up animation (first 0.15s)
            if elapsed < 0.15:
                t = elapsed / 0.15
                offset = int(30 * t)
            else:
                offset = 30

        mole_y = self.cy - offset

        # Body
        pygame.draw.ellipse(
            surface, MOLE_BODY,
            (self.cx - 30, mole_y - 35, 60, 50),
        )
        # Face
        pygame.draw.ellipse(
            surface, MOLE_FACE,
            (self.cx - 22, mole_y - 28, 44, 36),
        )
        # Eyes
        if self.whacked:
            # X eyes
            for ex in [self.cx - 10, self.cx + 10]:
                pygame.draw.line(surface, MOLE_EYE, (ex - 4, mole_y - 18), (ex + 4, mole_y - 12), 2)
                pygame.draw.line(surface, MOLE_EYE, (ex + 4, mole_y - 18), (ex - 4, mole_y - 12), 2)
        else:
            pygame.draw.circle(surface, MOLE_EYE, (self.cx - 10, mole_y - 15), 4)
            pygame.draw.circle(surface, MOLE_EYE, (self.cx + 10, mole_y - 15), 4)
            # Eye highlights
            pygame.draw.circle(surface, WHITE, (self.cx - 8, mole_y - 16), 2)
            pygame.draw.circle(surface, WHITE, (self.cx + 12, mole_y - 16), 2)

        # Nose
        pygame.draw.circle(surface, MOLE_NOSE, (self.cx, mole_y - 5), 5)

        # Whack effect
        if self.whacked and self.whack_anim < 0.5:
            # Stars
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                dist = 30 + self.whack_anim * 40
                sx = self.cx + int(math.cos(rad) * dist)
                sy = mole_y - 20 + int(math.sin(rad) * dist)
                pygame.draw.circle(surface, GOLD, (sx, sy), 4)


class WhackAMoleGame:
    """Main Whack-a-Mole game class."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🔨 Whack-a-Mole — Hand Tracking")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.SysFont("Segoe UI", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Segoe UI", 32, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 22)

        # Camera & tracker
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ready = self.cap.isOpened()
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)

        self.cam_surface = None
        self.grass_blades = [(i, 5 + ((i * 17) % 11)) for i in range(0, SCREEN_W, 30)]

        # Create moles grid
        self.moles = []
        for r in range(GRID_ROWS):
            row = []
            for c in range(GRID_COLS):
                cx = GRID_OFFSET_X + c * GRID_SPACING_X + GRID_SPACING_X // 2
                cy = GRID_OFFSET_Y + r * GRID_SPACING_Y + GRID_SPACING_Y // 2
                row.append(Mole(r, c, cx, cy))
            self.moles.append(row)

        self.reset()

    def reset(self):
        self.score = 0
        self.misses = 0
        self.combo = 0
        self.best_combo = 0
        self.last_hit_time = 0.0
        self.start_time = None
        self.game_over = False
        self.started = False
        self.finger_pos = None
        self.last_pinch = False
        self.whack_texts = []  # (text, x, y, start_time)

        for row in self.moles:
            for mole in row:
                mole.active = False
                mole.cooldown_until = 0

    def get_active_count(self):
        return sum(
            1 for row in self.moles for mole in row if mole.active and not mole.whacked
        )

    def get_max_active(self):
        """Difficulty scaling based on elapsed time."""
        if self.start_time is None:
            return INITIAL_ACTIVE
        elapsed = time.time() - self.start_time
        level = INITIAL_ACTIVE + int(elapsed / 15)
        return min(level, MAX_ACTIVE)

    def get_level(self):
        if self.start_time is None:
            return 1
        elapsed = time.time() - self.start_time
        return min(5, 1 + int(elapsed / 12))

    def whack_at(self, x, y):
        if not self.started or self.game_over:
            return

        hit = False
        now = time.time()
        for row in self.moles:
            for mole in row:
                if mole.try_whack(x, y):
                    hit = True
                    if now - self.last_hit_time <= COMBO_RESET_SECS:
                        self.combo += 1
                    else:
                        self.combo = 1
                    self.best_combo = max(self.best_combo, self.combo)
                    self.last_hit_time = now

                    level_bonus = max(0, self.get_level() - 1)
                    combo_bonus = self.combo // 4
                    points = 1 + level_bonus + combo_bonus
                    self.score += points

                    text = f"+{points}"
                    if self.combo >= 3:
                        text += f" x{self.combo}"
                    self.whack_texts.append((text, mole.cx, mole.cy - 40, now))

        if not hit:
            self.combo = 0

    def spawn_moles(self):
        """Randomly pop up moles based on difficulty."""
        active = self.get_active_count()
        max_active = self.get_max_active()

        if active < max_active:
            available = [
                mole for row in self.moles for mole in row if mole.can_pop()
            ]
            if available:
                mole = random.choice(available)
                # Shorter duration as game progresses
                elapsed = time.time() - self.start_time if self.start_time else 0
                duration_scale = max(0.45, 1.0 - elapsed / 120)
                duration = random.uniform(MOLE_SHOW_MIN, MOLE_SHOW_MAX) * duration_scale
                mole.pop_up(duration)

    def update(self):
        if self.game_over or not self.started:
            return

        # Timer check
        elapsed = time.time() - self.start_time
        if elapsed >= GAME_DURATION:
            self.game_over = True
            return

        # Update moles
        for row in self.moles:
            for mole in row:
                was_active = mole.active and not mole.whacked
                mole.update()
                # Count misses (mole disappeared without being whacked)
                if was_active and not mole.active and not mole.whacked:
                    self.misses += 1

        # Spawn new moles
        self.spawn_moles()

        if self.combo > 0 and time.time() - self.last_hit_time > COMBO_RESET_SECS:
            self.combo = 0

        # Clean up floating texts
        now = time.time()
        self.whack_texts = [(t, x, y, st) for t, x, y, st in self.whack_texts if now - st < 0.8]

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (120, 90))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw(self):
        self.screen.fill(BG_COLOR)

        # Decorative grass pattern
        for i, h in self.grass_blades:
            pygame.draw.line(self.screen, BG_DARK, (i, SCREEN_H), (i + 5, SCREEN_H - h), 2)

        # Draw moles
        for row in self.moles:
            for mole in row:
                mole.draw(self.screen)

        # Draw finger cursor
        if self.finger_pos and self.started and not self.game_over:
            # Mallet cursor
            fx, fy = self.finger_pos
            # Handle
            pygame.draw.line(self.screen, (120, 80, 40), (fx + 15, fy - 15), (fx + 35, fy - 35), 6)
            # Head
            pygame.draw.rect(self.screen, (150, 100, 50), (fx - 12, fy - 8, 30, 18), border_radius=4)
            pygame.draw.rect(self.screen, (180, 130, 70), (fx - 10, fy - 6, 26, 14), border_radius=3)
            # Target circle
            pygame.draw.circle(self.screen, (255, 255, 100, 128), (fx, fy), WHACK_DISTANCE, 2)

        # Floating score texts
        now = time.time()
        for text, x, y, start in self.whack_texts:
            elapsed = now - start
            alpha = max(0, 1 - elapsed / 0.8)
            offset_y = int(elapsed * 40)
            color = (
                int(GOLD[0] * alpha),
                int(GOLD[1] * alpha),
                int(GOLD[2] * alpha),
            )
            txt = self.font_medium.render(text, True, color)
            self.screen.blit(txt, (x - txt.get_width() // 2, y - offset_y))

        # HUD
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (20, 15))

        level_text = self.font_small.render(f"Level: {self.get_level()}", True, (180, 220, 255))
        self.screen.blit(level_text, (20, 54))

        if self.combo > 1:
            combo_text = self.font_small.render(f"Combo x{self.combo}", True, GOLD)
            self.screen.blit(combo_text, (20, 84))

        # Timer
        if self.started and self.start_time:
            remaining = max(0, GAME_DURATION - (time.time() - self.start_time))
            mins = int(remaining) // 60
            secs = int(remaining) % 60
            timer_color = RED if remaining < 10 else WHITE
            timer_text = self.font_medium.render(f"{mins}:{secs:02d}", True, timer_color)
            self.screen.blit(timer_text, (SCREEN_W // 2 - timer_text.get_width() // 2, 15))

        # Accuracy
        total = self.score + self.misses
        accuracy = (self.score / total * 100) if total > 0 else 100
        acc_text = self.font_small.render(f"Accuracy: {accuracy:.0f}%", True, GREEN)
        self.screen.blit(acc_text, (SCREEN_W - acc_text.get_width() - 20, 20))

        # Camera overlay
        if self.cam_surface:
            self.screen.blit(self.cam_surface, (SCREEN_W - 130, SCREEN_H - 100))
            pygame.draw.rect(self.screen, WHITE, (SCREEN_W - 130, SCREEN_H - 100, 120, 90), 2)

        # Start screen
        if not self.started:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (0, 0))

            title = self.font_large.render("🔨 WHACK-A-MOLE", True, GOLD)
            self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 180))

            instr = self.font_small.render("Point your finger at moles to whack them!", True, WHITE)
            self.screen.blit(instr, (SCREEN_W // 2 - instr.get_width() // 2, 260))

            ctl = self.font_small.render("Pinch on mole (or click) to hit", True, (210, 240, 255))
            self.screen.blit(ctl, (SCREEN_W // 2 - ctl.get_width() // 2, 294))

            start = self.font_medium.render("Press SPACE to start", True, GREEN)
            self.screen.blit(start, (SCREEN_W // 2 - start.get_width() // 2, 320))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (0, 0))

            go_text = self.font_large.render("TIME'S UP!", True, GOLD)
            self.screen.blit(go_text, (SCREEN_W // 2 - go_text.get_width() // 2, 150))

            score_display = self.font_large.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(score_display, (SCREEN_W // 2 - score_display.get_width() // 2, 230))

            acc_display = self.font_medium.render(f"Accuracy: {accuracy:.0f}%", True, GREEN)
            self.screen.blit(acc_display, (SCREEN_W // 2 - acc_display.get_width() // 2, 300))

            combo_display = self.font_small.render(f"Best Combo: x{self.best_combo}", True, GOLD)
            self.screen.blit(combo_display, (SCREEN_W // 2 - combo_display.get_width() // 2, 338))

            restart = self.font_small.render("Press R to play again  |  ESC to quit", True, (180, 180, 180))
            self.screen.blit(restart, (SCREEN_W // 2 - restart.get_width() // 2, 386))

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
                    elif event.key == pygame.K_SPACE and not self.started:
                        self.started = True
                        self.start_time = time.time()
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.whack_at(*event.pos)

            # Camera & tracking
            ret, frame = (False, None)
            if self.camera_ready:
                ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                self.tracker.process_frame(frame)
                self.tracker.draw_landmarks(frame)
                self.draw_cam_overlay(frame)

                pos = self.tracker.get_fingertip_pos("index", SCREEN_W, SCREEN_H)
                self.finger_pos = pos
                pinch = False
                pd = self.tracker.get_pinch_distance(SCREEN_W, SCREEN_H)
                if pd is not None:
                    pinch = pd < PINCH_WHACK_DIST

                if pinch and not self.last_pinch and self.finger_pos is not None:
                    self.whack_at(self.finger_pos[0], self.finger_pos[1])
                self.last_pinch = pinch
            else:
                self.finger_pos = None
                self.tracker.hand_detected = False
                self.last_pinch = False

            self.update()
            self.draw()
            self.clock.tick(FPS)

        self.cap.release()
        self.tracker.release()
        pygame.quit()


def main():
    game = WhackAMoleGame()
    game.run()


if __name__ == "__main__":
    main()
