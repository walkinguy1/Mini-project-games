"""
Flappy Bird — Hand Tracking Edition
Open your hand to flap upward, make a fist to fall.
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 400, 600
FPS = 60
CAM_W, CAM_H = 640, 480

# Colors
SKY_TOP = (25, 25, 60)
SKY_BOTTOM = (70, 130, 180)
PIPE_COLOR = (50, 180, 80)
PIPE_HIGHLIGHT = (80, 220, 110)
PIPE_SHADOW = (30, 120, 50)
BIRD_BODY = (255, 210, 50)
BIRD_WING = (255, 170, 30)
BIRD_EYE = (0, 0, 0)
BIRD_BEAK = (255, 100, 50)
WHITE = (255, 255, 255)
GROUND_COLOR = (100, 70, 40)
GROUND_GRASS = (60, 160, 60)

# Physics
GRAVITY = 0.45
FLAP_FORCE = -7.0
TERMINAL_VELOCITY = 10.0

# Pipe settings
PIPE_W = 60
PIPE_GAP = 160
PIPE_SPEED = 3
PIPE_SPAWN_DIST = 220
MIN_PIPE_GAP = 115
MAX_PIPE_SPEED = 6

# Bird settings
BIRD_X = 80
BIRD_RADIUS = 18


class Bird:
    """The flappy bird."""

    def __init__(self):
        self.x = BIRD_X
        self.y = SCREEN_H // 2
        self.vy = 0
        self.angle = 0

    def flap(self):
        self.vy = FLAP_FORCE

    def update(self):
        self.vy += GRAVITY
        self.vy = min(self.vy, TERMINAL_VELOCITY)
        self.y += self.vy

        # Tilt based on velocity
        self.angle = max(-30, min(60, -self.vy * 4))

    def draw(self, surface):
        x, y = int(self.x), int(self.y)
        r = BIRD_RADIUS

        # Body
        pygame.draw.circle(surface, BIRD_BODY, (x, y), r)
        # Wing
        wing_y_offset = -3 if self.vy < 0 else 3
        pygame.draw.ellipse(
            surface, BIRD_WING,
            (x - r + 2, y + wing_y_offset - 5, r, 12),
        )
        # Eye
        pygame.draw.circle(surface, WHITE, (x + 6, y - 5), 6)
        pygame.draw.circle(surface, BIRD_EYE, (x + 8, y - 5), 3)
        # Beak
        beak_points = [(x + r, y), (x + r + 10, y + 3), (x + r, y + 6)]
        pygame.draw.polygon(surface, BIRD_BEAK, beak_points)

    def get_rect(self):
        return pygame.Rect(
            self.x - BIRD_RADIUS, self.y - BIRD_RADIUS,
            BIRD_RADIUS * 2, BIRD_RADIUS * 2,
        )


class Pipe:
    """A pair of pipes (top and bottom)."""

    def __init__(self, x, gap, speed):
        self.x = x
        self.gap = gap
        self.speed = speed
        self.gap_y = random.randint(120, SCREEN_H - 120 - self.gap)
        self.passed = False

    def update(self):
        self.x -= self.speed

    def is_off_screen(self):
        return self.x + PIPE_W < 0

    def draw(self, surface):
        x = int(self.x)

        # Top pipe
        top_rect = pygame.Rect(x, 0, PIPE_W, self.gap_y)
        pygame.draw.rect(surface, PIPE_COLOR, top_rect)
        pygame.draw.rect(surface, PIPE_HIGHLIGHT, (x, 0, 6, self.gap_y))
        pygame.draw.rect(surface, PIPE_SHADOW, (x + PIPE_W - 6, 0, 6, self.gap_y))
        # Lip
        pygame.draw.rect(
            surface, PIPE_SHADOW,
            (x - 4, self.gap_y - 20, PIPE_W + 8, 20),
        )

        # Bottom pipe
        bottom_y = self.gap_y + self.gap
        bottom_rect = pygame.Rect(x, bottom_y, PIPE_W, SCREEN_H - bottom_y)
        pygame.draw.rect(surface, PIPE_COLOR, bottom_rect)
        pygame.draw.rect(surface, PIPE_HIGHLIGHT, (x, bottom_y, 6, SCREEN_H - bottom_y))
        pygame.draw.rect(surface, PIPE_SHADOW, (x + PIPE_W - 6, bottom_y, 6, SCREEN_H - bottom_y))
        # Lip
        pygame.draw.rect(
            surface, PIPE_SHADOW,
            (x - 4, bottom_y, PIPE_W + 8, 20),
        )

    def collides(self, bird_rect):
        x = int(self.x)
        top_rect = pygame.Rect(x, 0, PIPE_W, self.gap_y)
        bottom_y = self.gap_y + self.gap
        bottom_rect = pygame.Rect(x, bottom_y, PIPE_W, SCREEN_H - bottom_y)
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)


class FlappyBirdGame:
    """Main Flappy Bird game class."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🐦 Flappy Bird — Hand Tracking")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.SysFont("Segoe UI", 60, bold=True)
        self.font_medium = pygame.font.SysFont("Segoe UI", 36, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 22)

        # Camera & tracker
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ready = self.cap.isOpened()
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)

        # Create gradient background
        self.bg_surface = pygame.Surface((SCREEN_W, SCREEN_H))
        for y in range(SCREEN_H):
            t = y / SCREEN_H
            r = int(SKY_TOP[0] + (SKY_BOTTOM[0] - SKY_TOP[0]) * t)
            g = int(SKY_TOP[1] + (SKY_BOTTOM[1] - SKY_TOP[1]) * t)
            b = int(SKY_TOP[2] + (SKY_BOTTOM[2] - SKY_TOP[2]) * t)
            pygame.draw.line(self.bg_surface, (r, g, b), (0, y), (SCREEN_W, y))

        self.reset()
        self.cam_surface = None
        self.high_score = 0
        self.open_frames = 0
        self.flap_cooldown_until = 0.0

    def reset(self):
        self.bird = Bird()
        self.level = 1
        self.pipe_speed = PIPE_SPEED
        self.pipe_gap = PIPE_GAP
        self.pipes = [Pipe(SCREEN_W + 100, self.pipe_gap, self.pipe_speed)]
        self.score = 0
        self.game_over = False
        self.started = False
        self.was_open = False
        self.ground_offset = 0
        self.open_frames = 0
        self.flap_cooldown_until = 0.0

    def update(self):
        if self.game_over or not self.started:
            return

        # Difficulty progression
        self.level = min(10, 1 + self.score // 5)
        self.pipe_speed = min(MAX_PIPE_SPEED, PIPE_SPEED + (self.level - 1) * 0.35)
        self.pipe_gap = max(MIN_PIPE_GAP, PIPE_GAP - (self.level - 1) * 6)

        self.bird.update()
        self.ground_offset = (self.ground_offset + self.pipe_speed) % 40

        # Update pipes
        for pipe in self.pipes[:]:
            pipe.update()

            # Score
            if not pipe.passed and pipe.x + PIPE_W < self.bird.x:
                pipe.passed = True
                self.score += 1
                self.high_score = max(self.high_score, self.score)

            if pipe.is_off_screen():
                self.pipes.remove(pipe)

            # Collision
            if pipe.collides(self.bird.get_rect()):
                self.game_over = True

        # Spawn pipes
        if len(self.pipes) == 0 or self.pipes[-1].x < SCREEN_W - PIPE_SPAWN_DIST:
            self.pipes.append(Pipe(SCREEN_W, self.pipe_gap, self.pipe_speed))

        # Floor / ceiling collision
        if self.bird.y + BIRD_RADIUS > SCREEN_H - 50 or self.bird.y - BIRD_RADIUS < 0:
            self.game_over = True

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (120, 90))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw(self):
        # Background gradient
        self.screen.blit(self.bg_surface, (0, 0))

        # Draw some clouds
        for cx, cy, cw in [(80, 60, 60), (250, 100, 45), (350, 40, 50)]:
            pygame.draw.ellipse(self.screen, (200, 220, 255, 80), (cx, cy, cw, cw // 2))
            pygame.draw.ellipse(self.screen, (200, 220, 255, 80), (cx + 20, cy - 10, cw + 10, cw // 2))

        # Pipes
        for pipe in self.pipes:
            pipe.draw(self.screen)

        # Ground
        ground_y = SCREEN_H - 50
        pygame.draw.rect(self.screen, GROUND_GRASS, (0, ground_y, SCREEN_W, 10))
        pygame.draw.rect(self.screen, GROUND_COLOR, (0, ground_y + 10, SCREEN_W, 40))
        # Ground pattern
        for i in range(-1, SCREEN_W // 40 + 2):
            x = i * 40 - int(self.ground_offset)
            pygame.draw.line(self.screen, (80, 55, 30), (x, ground_y + 15), (x + 20, ground_y + 45), 2)

        # Bird
        self.bird.draw(self.screen)

        # Score
        score_text = self.font_large.render(str(self.score), True, WHITE)
        # Shadow
        shadow_text = self.font_large.render(str(self.score), True, (0, 0, 0))
        self.screen.blit(shadow_text, (SCREEN_W // 2 - score_text.get_width() // 2 + 2, 22))
        self.screen.blit(score_text, (SCREEN_W // 2 - score_text.get_width() // 2, 20))

        level_text = self.font_small.render(f"Level {self.level}", True, (200, 225, 255))
        self.screen.blit(level_text, (15, 16))

        # Hand gesture indicator
        gesture = self.tracker.get_gesture() if self.tracker.hand_detected else None
        indicator_color = (100, 255, 100) if gesture == "open" else (255, 100, 100)
        gesture_label = gesture if gesture else "No hand"
        indicator_text = self.font_small.render(f"✋ {gesture_label}", True, indicator_color)
        self.screen.blit(indicator_text, (10, SCREEN_H - 80))

        # Camera overlay
        if self.cam_surface:
            self.screen.blit(self.cam_surface, (SCREEN_W - 130, SCREEN_H - 145))
            pygame.draw.rect(self.screen, WHITE, (SCREEN_W - 130, SCREEN_H - 145, 120, 90), 2)

        # Start screen
        if not self.started:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            self.screen.blit(overlay, (0, 0))

            title = self.font_large.render("FLAPPY", True, BIRD_BODY)
            self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 120))

            instr = self.font_small.render("Open hand to flap  •  Fist to fall", True, WHITE)
            self.screen.blit(instr, (SCREEN_W // 2 - instr.get_width() // 2, 220))

            start = self.font_small.render("Open your hand to start!", True, (180, 255, 180))
            if not self.camera_ready:
                start = self.font_small.render("Camera unavailable - press SPACE to play", True, (255, 180, 180))
            self.screen.blit(start, (SCREEN_W // 2 - start.get_width() // 2, 280))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (0, 0))

            go_text = self.font_large.render("GAME OVER", True, (255, 80, 80))
            self.screen.blit(go_text, (SCREEN_W // 2 - go_text.get_width() // 2, 150))

            score_display = self.font_medium.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(score_display, (SCREEN_W // 2 - score_display.get_width() // 2, 240))

            best_display = self.font_small.render(f"High Score: {self.high_score}", True, BIRD_BODY)
            self.screen.blit(best_display, (SCREEN_W // 2 - best_display.get_width() // 2, 290))

            restart = self.font_small.render("Press R to restart  |  ESC to quit", True, (180, 180, 180))
            self.screen.blit(restart, (SCREEN_W // 2 - restart.get_width() // 2, 350))

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
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset()
                    elif event.key == pygame.K_SPACE:
                        if not self.started:
                            self.started = True
                        if not self.game_over:
                            self.bird.flap()

            # Camera & hand tracking
            ret, frame = (False, None)
            if self.camera_ready:
                ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                self.tracker.process_frame(frame)
                self.tracker.draw_landmarks(frame)
                self.draw_cam_overlay(frame)

                gesture = self.tracker.get_gesture()
                is_open = gesture == "open"

                now = time.time()
                self.open_frames = self.open_frames + 1 if is_open else 0

                # Flap on debounced open hand
                if self.open_frames >= 2 and now >= self.flap_cooldown_until:
                    if not self.started:
                        self.started = True
                    if not self.game_over:
                        self.bird.flap()
                        self.flap_cooldown_until = now + 0.18
                    self.open_frames = 0

                self.was_open = is_open
            else:
                self.tracker.hand_detected = False

            self.update()
            self.draw()
            self.clock.tick(FPS)

        self.cap.release()
        self.tracker.release()
        pygame.quit()


def main():
    game = FlappyBirdGame()
    game.run()


if __name__ == "__main__":
    main()
