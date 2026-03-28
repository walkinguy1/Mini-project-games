"""
Fruit Ninja — Hand Tracking Edition
Swipe with your index finger to slice fruits rising from the bottom of the screen.

Improvements applied:
  1. Persistent high score (reads/writes scores.json via score_manager.py)
  3. Procedural sound effects (via sound_fx.py — no audio files needed)
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

# ── Improvement 1: persistent high score ────────────────────────────────────
try:
    from games.score_manager import save_score, get_high_score
    _SCORES_AVAILABLE = True
except ImportError:
    # Graceful fallback if score_manager.py hasn't been created yet
    def save_score(name, score): pass
    def get_high_score(name): return 0
    _SCORES_AVAILABLE = False

# ── Improvement 3: procedural sound effects ──────────────────────────────────
try:
    from games.sound_fx import play_slice, play_game_over
    _SOUND_AVAILABLE = True
except ImportError:
    # Graceful fallback if sound_fx.py hasn't been created yet
    def play_slice(): pass
    def play_game_over(): pass
    _SOUND_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 800, 600
FPS = 60
CAM_W, CAM_H = 640, 480

# Colors
BG_COLOR = (15, 15, 30)
WHITE = (255, 255, 255)
RED = (220, 50, 50)
GREEN = (50, 200, 80)
YELLOW = (255, 210, 50)
ORANGE = (255, 140, 30)
PINK = (255, 100, 150)
PURPLE = (180, 60, 220)
TRAIL_COLOR = (100, 200, 255)
DARK_OVERLAY = (0, 0, 0)

# Fruit settings
FRUIT_COLORS = [RED, GREEN, YELLOW, ORANGE, PINK, PURPLE]
FRUIT_NAMES = ["Apple", "Lime", "Banana", "Orange", "Dragonfruit", "Grape"]
FRUIT_RADIUS = 28
GRAVITY = 0.15
SPAWN_INTERVAL_START = 1.2  # seconds
SPAWN_INTERVAL_MIN = 0.4
SLICE_DISTANCE = 45
MAX_LIVES = 3

# Trail settings
TRAIL_LENGTH = 15
MAX_PARTICLES = 250
MAX_LEVEL = 8


class Particle:
    """A small particle for slice effects."""

    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 7)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.life = random.uniform(0.3, 0.8)
        self.max_life = self.life
        self.radius = random.randint(3, 7)

    def update(self, dt):
        self.x += self.vx
        self.vy += 0.15
        self.y += self.vy
        self.life -= dt
        return self.life > 0

    def draw(self, surface):
        alpha = max(0, self.life / self.max_life)
        r = max(1, int(self.radius * alpha))
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), r)


class Fruit:
    """A fruit that rises from the bottom and falls back down."""

    def __init__(self, level=1):
        self.x = random.randint(80, SCREEN_W - 80)
        self.y = SCREEN_H + random.uniform(FRUIT_RADIUS * 0.8, FRUIT_RADIUS * 3.2)
        spread = 1.8 + min(1.2, (level - 1) * 0.16)
        self.vx = random.uniform(-spread, spread)

        # Stronger upward launch so fruits reach upper half consistently.
        min_vy = -12.2 - (level - 1) * 0.22
        max_vy = -10.4 - (level - 1) * 0.18
        self.vy = random.uniform(min_vy, max_vy)
        color_idx = random.randint(0, len(FRUIT_COLORS) - 1)
        self.color = FRUIT_COLORS[color_idx]
        self.name = FRUIT_NAMES[color_idx]
        self.radius = FRUIT_RADIUS + random.randint(-4, 4)
        self.alive = True
        self.sliced = False
        self.rotation = random.uniform(0, 360)
        self.rot_speed = random.uniform(-5, 5)

    def update(self):
        self.x += self.vx
        self.vy += GRAVITY
        self.y += self.vy
        self.rotation += self.rot_speed

    def is_off_screen(self):
        return self.y > SCREEN_H + self.radius * 2

    def draw(self, surface):
        if not self.alive:
            return
        # Draw fruit body
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        # Highlight
        highlight_pos = (int(self.x - self.radius * 0.3), int(self.y - self.radius * 0.3))
        pygame.draw.circle(surface, WHITE, highlight_pos, self.radius // 4)
        # Stem
        stem_end = (int(self.x), int(self.y - self.radius - 5))
        stem_start = (int(self.x + 3), int(self.y - self.radius + 2))
        pygame.draw.line(surface, (80, 50, 20), stem_start, stem_end, 3)


class FruitNinjaGame:
    """Main Fruit Ninja game class."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🍉 Fruit Ninja — Hand Tracking")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.SysFont("Segoe UI", 60, bold=True)
        self.font_medium = pygame.font.SysFont("Segoe UI", 36, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 24)

        # Camera & tracker
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ready = self.cap.isOpened()
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)

        # ── Improvement 1: load persisted high score on startup ──────────────
        self.high_score = get_high_score("Fruit Ninja")

        # Game state
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.lives = MAX_LIVES
        self.fruits = []
        self.particles = []
        self.trail = []
        self.last_spawn = time.time()
        self.spawn_interval = SPAWN_INTERVAL_START
        self.game_over = False
        self.finger_pos = None
        self.prev_finger_pos = None
        self.start_time = time.time()
        self.cam_surface = None
        self.dt = 1.0 / FPS
        self.level = 1
        self.paused = False
        self.mouse_pos = None
        self.prev_mouse_pos = None
        self.mouse_swiping = False

    def spawn_fruit(self):
        """Spawn a new fruit."""
        self.fruits.append(Fruit(level=self.level))

    def check_slice(self, pos, prev_pos):
        """Check if the finger movement slices any fruit."""
        if pos is None or prev_pos is None:
            return

        # Calculate swipe speed
        dx = pos[0] - prev_pos[0]
        dy = pos[1] - prev_pos[1]
        speed = math.sqrt(dx * dx + dy * dy)

        if speed < 8:  # Need minimum swipe speed
            return

        sliced_any = False
        for fruit in self.fruits:
            if not fruit.alive or fruit.sliced:
                continue
            dist = math.sqrt(
                (fruit.x - pos[0]) ** 2 + (fruit.y - pos[1]) ** 2
            )
            if dist < SLICE_DISTANCE:
                fruit.sliced = True
                fruit.alive = False
                sliced_any = True
                self.score += 1
                self.combo += 1
                self.best_combo = max(self.best_combo, self.combo)

                # ── Improvement 3: play slice sound ──────────────────────────
                play_slice()

                # ── Improvement 1: update and save high score live ────────────
                if self.score > self.high_score:
                    self.high_score = self.score
                    save_score("Fruit Ninja", self.high_score)

                # Spawn particles
                for _ in range(12):
                    self.particles.append(Particle(fruit.x, fruit.y, fruit.color))
                if len(self.particles) > MAX_PARTICLES:
                    self.particles = self.particles[-MAX_PARTICLES:]

        if not sliced_any:
            self.combo = 0

    def update(self, dt):
        """Update game state."""
        if self.game_over or self.paused:
            return

        now = time.time()
        elapsed = now - self.start_time
        self.level = min(MAX_LEVEL, 1 + int(elapsed // 20))

        # Spawn fruits
        if now - self.last_spawn > self.spawn_interval:
            # Spawn 1-3 fruits at once
            count = random.choices([1, 2, 3], weights=[max(2, 6 - self.level), 3 + self.level // 2, 1 + self.level // 3])[0]
            for _ in range(count):
                self.spawn_fruit()

            # Occasionally add one high-arc challenge fruit for variety.
            if self.level >= 3 and random.random() < 0.35:
                bonus = Fruit(level=self.level)
                bonus.vy -= random.uniform(0.6, 1.8)
                self.fruits.append(bonus)

            self.last_spawn = now
            # Gradually decrease spawn interval
            self.spawn_interval = max(
                SPAWN_INTERVAL_MIN,
                SPAWN_INTERVAL_START - elapsed * 0.005 - (self.level - 1) * 0.03,
            )

        # Update fruits
        for fruit in self.fruits[:]:
            fruit.update()
            if fruit.is_off_screen():
                if fruit.alive and not fruit.sliced:
                    self.lives -= 1
                    self.combo = 0
                    if self.lives <= 0:
                        self.game_over = True
                        # ── Improvement 3: play game over sound ──────────────
                        play_game_over()
                        # ── Improvement 1: save final score on game over ──────
                        if self.score > self.high_score:
                            self.high_score = self.score
                            save_score("Fruit Ninja", self.high_score)
                self.fruits.remove(fruit)

        # Update particles
        self.particles = [p for p in self.particles if p.update(dt)]

        # Check slice (hand first, then mouse swipe fallback)
        if self.finger_pos and self.prev_finger_pos:
            self.check_slice(self.finger_pos, self.prev_finger_pos)
        elif self.mouse_swiping and self.mouse_pos and self.prev_mouse_pos:
            self.check_slice(self.mouse_pos, self.prev_mouse_pos)

        # Update trail
        active_pos = self.finger_pos if self.finger_pos else (self.mouse_pos if self.mouse_swiping else None)
        if active_pos:
            self.trail.append(active_pos)
            if len(self.trail) > TRAIL_LENGTH:
                self.trail.pop(0)
        else:
            self.trail.clear()

    def draw_cam_overlay(self, frame):
        """Convert camera frame to a small Pygame surface for overlay."""
        frame_small = cv2.resize(frame, (160, 120))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw(self):
        """Draw everything."""
        self.screen.fill(BG_COLOR)

        # Draw trail
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                alpha = i / len(self.trail)
                width = int(3 + alpha * 5)
                color = (
                    int(TRAIL_COLOR[0] * alpha),
                    int(TRAIL_COLOR[1] * alpha),
                    int(TRAIL_COLOR[2] * alpha),
                )
                pygame.draw.line(
                    self.screen, color,
                    self.trail[i - 1], self.trail[i],
                    width,
                )

        # Draw fruits
        for fruit in self.fruits:
            fruit.draw(self.screen)

        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)

        # Draw finger cursor
        if self.finger_pos:
            pygame.draw.circle(self.screen, TRAIL_COLOR, self.finger_pos, 10, 2)

        # HUD - Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (20, 15))

        level_text = self.font_small.render(f"Level: {self.level}", True, (170, 220, 255))
        self.screen.blit(level_text, (20, 52))

        active_text = self.font_small.render(f"Active Fruits: {len(self.fruits)}", True, (160, 170, 210))
        self.screen.blit(active_text, (20, 86))

        # ── Improvement 1: show persistent high score in HUD ─────────────────
        hs_color = (255, 210, 80) if self.score >= self.high_score and self.score > 0 else (180, 180, 200)
        hs_text = self.font_small.render(f"Best: {self.high_score}", True, hs_color)
        self.screen.blit(hs_text, (20, 120))

        # HUD - Combo
        if self.combo > 1:
            combo_text = self.font_small.render(f"Combo x{self.combo}!", True, YELLOW)
            self.screen.blit(combo_text, (160, 52))

        # HUD - Lives
        for i in range(MAX_LIVES):
            color = RED if i < self.lives else (60, 60, 60)
            x = SCREEN_W - 40 - i * 40
            pygame.draw.circle(self.screen, color, (x, 30), 14)
            # Heart shape approximation
            pygame.draw.circle(self.screen, color, (x - 6, 24), 8)
            pygame.draw.circle(self.screen, color, (x + 6, 24), 8)

        # Camera overlay (top-right corner)
        if self.cam_surface:
            self.screen.blit(self.cam_surface, (SCREEN_W - 170, SCREEN_H - 130))
            pygame.draw.rect(
                self.screen, WHITE,
                (SCREEN_W - 170, SCREEN_H - 130, 160, 120), 2,
            )

        # Game Over screen
        if self.game_over:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (0, 0))

            go_text = self.font_large.render("GAME OVER", True, RED)
            go_rect = go_text.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 70))
            self.screen.blit(go_text, go_rect)

            final_text = self.font_medium.render(
                f"Final Score: {self.score}  |  Best Combo: {self.best_combo}", True, WHITE
            )
            final_rect = final_text.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2))
            self.screen.blit(final_text, final_rect)

            # ── Improvement 1: show all-time high score on game over screen ───
            hs_label = "NEW HIGH SCORE!" if self.score >= self.high_score and self.score > 0 else f"High Score: {self.high_score}"
            hs_color_go = YELLOW if self.score >= self.high_score and self.score > 0 else (180, 180, 200)
            hs_go_text = self.font_small.render(hs_label, True, hs_color_go)
            hs_go_rect = hs_go_text.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 45))
            self.screen.blit(hs_go_text, hs_go_rect)

            restart_text = self.font_small.render(
                "Press R to restart  |  ESC to quit", True, (180, 180, 180)
            )
            restart_rect = restart_text.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 85))
            self.screen.blit(restart_text, restart_rect)

        if self.paused and not self.game_over:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (0, 0))
            txt = self.font_large.render("PAUSED", True, WHITE)
            self.screen.blit(txt, txt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 10)))
            hint = self.font_small.render("Press P to resume", True, (210, 210, 230))
            self.screen.blit(hint, hint.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 40)))

        pygame.display.flip()

    def reset(self):
        """Reset game state."""
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.lives = MAX_LIVES
        self.fruits.clear()
        self.particles.clear()
        self.trail.clear()
        self.last_spawn = time.time()
        self.spawn_interval = SPAWN_INTERVAL_START
        self.game_over = False
        self.start_time = time.time()
        self.level = 1
        self.paused = False
        self.mouse_pos = None
        self.prev_mouse_pos = None
        self.mouse_swiping = False
        # ── Improvement 1: re-load high score in case it was updated ─────────
        self.high_score = get_high_score("Fruit Ninja")

    def run(self):
        """Main game loop."""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset()
                    elif event.key == pygame.K_p and not self.game_over:
                        self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.mouse_swiping = True
                    self.prev_mouse_pos = event.pos
                    self.mouse_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.mouse_swiping = False
                    self.prev_mouse_pos = None
                elif event.type == pygame.MOUSEMOTION and self.mouse_swiping:
                    self.prev_mouse_pos = self.mouse_pos
                    self.mouse_pos = event.pos

            # Read camera & track hand
            ret, frame = (False, None)
            if self.camera_ready:
                ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)  # Mirror
                self.tracker.process_frame(frame)
                self.tracker.draw_landmarks(frame)
                self.draw_cam_overlay(frame)

                self.prev_finger_pos = self.finger_pos
                pos = self.tracker.get_fingertip_pos("index", SCREEN_W, SCREEN_H)
                self.finger_pos = pos
            else:
                self.prev_finger_pos = self.finger_pos
                self.finger_pos = None
                self.tracker.hand_detected = False

            self.dt = self.clock.tick(FPS) / 1000.0
            self.update(self.dt)
            self.draw()

        self.cap.release()
        self.tracker.release()
        pygame.quit()


def main():
    game = FruitNinjaGame()
    game.run()


if __name__ == "__main__":
    main()