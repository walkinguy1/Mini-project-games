"""
Breakout — Hand Tracking Edition
Move your hand to control the paddle and pinch to trigger a temporary wider paddle boost.
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

try:
    from games.score_manager import save_score, get_high_score
except ImportError:
    def save_score(name, score):
        return False

    def get_high_score(name):
        return 0


SCREEN_W, SCREEN_H = 800, 600
FPS = 60
CAM_W, CAM_H = 640, 480

BG = (14, 14, 28)
WHITE = (245, 245, 255)
PADDLE = (90, 190, 255)
BALL = (255, 220, 110)
BRICK_COLORS = [(255, 120, 120), (255, 170, 100), (120, 220, 140), (120, 160, 255)]

PADDLE_W = 130
PADDLE_H = 16
PADDLE_Y = SCREEN_H - 60
BALL_RADIUS = 9
BALL_SPEED = 5.2
MAX_BALL_SPEED = 7.2

ROWS = 5
COLS = 10
BRICK_W = 68
BRICK_H = 24
BRICK_GAP = 8
GRID_X = (SCREEN_W - (COLS * BRICK_W + (COLS - 1) * BRICK_GAP)) // 2
GRID_Y = 70

BOOST_SECS = 1.8
BOOST_COOLDOWN_SECS = 0.75
LEVEL_ROWS = [4, 5, 6, 7]
KEY_PADDLE_SPEED = 7.5
MIN_BALL_VY = 2.15


class BreakoutGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🧱 Breakout — Hand Tracking")
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.SysFont("Segoe UI", 46, bold=True)
        self.font_medium = pygame.font.SysFont("Segoe UI", 30, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 20)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ready = self.cap.isOpened()
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)

        self.cam_surface = None
        self.last_pinch = False
        self.boost_until = 0.0
        self.boost_cooldown_until = 0.0
        self.high_score = get_high_score("Breakout")
        self._high_score_saved_for_round = False
        self.reset()

    def build_bricks(self, rows):
        bricks = []
        for r in range(rows):
            row = []
            for c in range(COLS):
                x = GRID_X + c * (BRICK_W + BRICK_GAP)
                y = GRID_Y + r * (BRICK_H + BRICK_GAP)
                row.append(pygame.Rect(x, y, BRICK_W, BRICK_H))
            bricks.append(row)
        return bricks

    def setup_level(self, level):
        self.level = level
        self.ball_base_speed = BALL_SPEED + (self.level - 1) * 0.45
        self.paddle_base_width = max(92, PADDLE_W - (self.level - 1) * 8)
        rows = LEVEL_ROWS[min(self.level - 1, len(LEVEL_ROWS) - 1)]
        self.bricks = self.build_bricks(rows)
        self.paddle_x = SCREEN_W // 2 - self.current_paddle_width() // 2
        self.paddle_vx = 0.0
        self.ball_x = SCREEN_W // 2
        self.ball_y = SCREEN_H // 2
        self.ball_vx = random.choice([-1, 1]) * self.ball_base_speed
        self.ball_vy = -self.ball_base_speed
        self.started = False
        self.level_cleared = False
        self.level_transition_until = 0.0

    def reset(self):
        self.score = 0
        self.lives = 4
        self.level = 1
        self.max_level = len(LEVEL_ROWS)
        self.game_over = False
        self.win = False
        self.high_score = get_high_score("Breakout")
        self._high_score_saved_for_round = False
        self.setup_level(1)

    def current_paddle_width(self):
        width = self.paddle_base_width
        return int(width * 1.45) if time.time() < self.boost_until else width

    def activate_boost(self):
        if time.time() < self.boost_cooldown_until:
            return
        self.boost_until = time.time() + BOOST_SECS
        self.boost_cooldown_until = time.time() + BOOST_COOLDOWN_SECS

    def accelerate_ball(self, delta):
        speed = (self.ball_vx * self.ball_vx + self.ball_vy * self.ball_vy) ** 0.5
        if speed >= MAX_BALL_SPEED:
            return
        target = min(MAX_BALL_SPEED, speed + delta)
        scale = target / max(0.001, speed)
        self.ball_vx *= scale
        self.ball_vy *= scale

    def _stabilize_ball_vector(self):
        """Avoid near-horizontal loops while preserving intended speed range."""
        speed = (self.ball_vx * self.ball_vx + self.ball_vy * self.ball_vy) ** 0.5
        min_speed = min(MAX_BALL_SPEED, self.ball_base_speed * 0.92)
        target_speed = max(min_speed, min(MAX_BALL_SPEED, speed))

        vy_sign = -1 if self.ball_vy < 0 else 1
        if abs(self.ball_vy) < MIN_BALL_VY:
            self.ball_vy = vy_sign * MIN_BALL_VY

        vx_abs = max(0.0, target_speed * target_speed - self.ball_vy * self.ball_vy) ** 0.5
        self.ball_vx = vx_abs if self.ball_vx >= 0 else -vx_abs

    def _handle_brick_collision(self, ball_rect):
        """Resolve one brick collision per frame using shallow-overlap axis."""
        for row in self.bricks:
            for brick in row[:]:
                if not ball_rect.colliderect(brick):
                    continue

                row.remove(brick)
                overlap_left = ball_rect.right - brick.left
                overlap_right = brick.right - ball_rect.left
                overlap_top = ball_rect.bottom - brick.top
                overlap_bottom = brick.bottom - ball_rect.top
                overlap_x = min(overlap_left, overlap_right)
                overlap_y = min(overlap_top, overlap_bottom)

                if overlap_x < overlap_y:
                    if self.ball_vx > 0:
                        self.ball_x = brick.left - BALL_RADIUS
                    else:
                        self.ball_x = brick.right + BALL_RADIUS
                    self.ball_vx *= -1
                else:
                    if self.ball_vy > 0:
                        self.ball_y = brick.top - BALL_RADIUS
                    else:
                        self.ball_y = brick.bottom + BALL_RADIUS
                    self.ball_vy *= -1

                self.accelerate_ball(0.05)
                self._stabilize_ball_vector()
                self.score += 10
                return True
        return False

    def update(self):
        if self.level_cleared:
            if time.time() >= self.level_transition_until:
                self.setup_level(self.level + 1)
            return

        if self.game_over or not self.started:
            return

        paddle_w = self.current_paddle_width()
        paddle_rect = pygame.Rect(int(self.paddle_x), PADDLE_Y, paddle_w, PADDLE_H)

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_x - BALL_RADIUS <= 0:
            self.ball_x = BALL_RADIUS
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x + BALL_RADIUS >= SCREEN_W:
            self.ball_x = SCREEN_W - BALL_RADIUS
            self.ball_vx = -abs(self.ball_vx)

        if self.ball_y - BALL_RADIUS <= 0:
            self.ball_y = BALL_RADIUS
            self.ball_vy = abs(self.ball_vy)

        if self.ball_y - BALL_RADIUS > SCREEN_H:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
                self.win = False
                self._persist_high_score_if_needed()
            else:
                self.ball_x = SCREEN_W // 2
                self.ball_y = SCREEN_H // 2
                self.ball_vx = random.choice([-1, 1]) * self.ball_base_speed
                self.ball_vy = -self.ball_base_speed
                self.started = False
            return

        ball_rect = pygame.Rect(int(self.ball_x - BALL_RADIUS), int(self.ball_y - BALL_RADIUS), BALL_RADIUS * 2, BALL_RADIUS * 2)

        if ball_rect.colliderect(paddle_rect) and self.ball_vy > 0:
            self.ball_y = PADDLE_Y - BALL_RADIUS
            hit = (self.ball_x - paddle_rect.x) / max(1, paddle_rect.w) - 0.5
            self.ball_vx = self.ball_base_speed * 2.4 * hit + self.paddle_vx * 0.28
            if abs(self.ball_vx) < 1.1:
                self.ball_vx = 1.1 if self.ball_vx >= 0 else -1.1
            self.ball_vy = -abs(self.ball_base_speed)
            self._stabilize_ball_vector()

        self._handle_brick_collision(ball_rect)

        self.bricks = [row for row in self.bricks if row]
        if not self.bricks:
            if self.level < self.max_level:
                self.lives = min(5, self.lives + 1)
                self.level_cleared = True
                self.level_transition_until = time.time() + 1.1
            else:
                self.game_over = True
                self.win = True
                self._persist_high_score_if_needed()

    def _persist_high_score_if_needed(self):
        """Save score once per finished run and refresh local high score."""
        if self._high_score_saved_for_round:
            return
        save_score("Breakout", self.score)
        self.high_score = max(self.high_score, self.score)
        self._high_score_saved_for_round = True

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (130, 96))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw(self):
        self.screen.fill(BG)

        for r, row in enumerate(self.bricks):
            color = BRICK_COLORS[r % len(BRICK_COLORS)]
            for brick in row:
                pygame.draw.rect(self.screen, color, brick, border_radius=5)
                pygame.draw.rect(self.screen, (30, 30, 50), brick, 1, border_radius=5)

        paddle_w = self.current_paddle_width()
        paddle_rect = pygame.Rect(int(self.paddle_x), PADDLE_Y, paddle_w, PADDLE_H)
        pygame.draw.rect(self.screen, PADDLE, paddle_rect, border_radius=8)

        pygame.draw.circle(self.screen, BALL, (int(self.ball_x), int(self.ball_y)), BALL_RADIUS)

        hud = self.font_small.render(
            f"Level: {self.level}/{self.max_level}   Score: {self.score}   Best: {self.high_score}   Lives: {self.lives}",
            True,
            WHITE,
        )
        self.screen.blit(hud, (16, 14))

        if time.time() < self.boost_until:
            boost = self.font_small.render("Boost active", True, (255, 215, 120))
            self.screen.blit(boost, (16, 40))
        elif time.time() < self.boost_cooldown_until:
            cd = self.font_small.render("Boost cooldown", True, (180, 180, 210))
            self.screen.blit(cd, (16, 40))

        if not self.started and not self.game_over:
            msg = "Pinch to boost paddle  •  Open hand / SPACE to launch"
            if not self.camera_ready:
                msg = "Camera unavailable  •  Press SPACE to launch"
            txt = self.font_small.render(msg, True, WHITE)
            self.screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2, SCREEN_H // 2 + 80))

        if self.game_over:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            title = self.font_large.render("YOU WIN!" if self.win else "GAME OVER", True, WHITE)
            self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, SCREEN_H // 2 - 70))
            sub = self.font_medium.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(sub, (SCREEN_W // 2 - sub.get_width() // 2, SCREEN_H // 2 - 10))
            best = self.font_small.render(f"Best: {self.high_score}", True, (205, 225, 255))
            self.screen.blit(best, (SCREEN_W // 2 - best.get_width() // 2, SCREEN_H // 2 + 24))
            hint = self.font_small.render("Press R to restart  |  ESC to quit", True, (190, 190, 210))
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H // 2 + 58))

        if self.level_cleared:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (0, 0))
            txt = self.font_large.render(f"Level {self.level} Cleared!", True, WHITE)
            self.screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2, SCREEN_H // 2 - 45))
            nxt = self.font_small.render("Loading next level...", True, (210, 210, 225))
            self.screen.blit(nxt, (SCREEN_W // 2 - nxt.get_width() // 2, SCREEN_H // 2 + 12))

        if self.cam_surface:
            self.screen.blit(self.cam_surface, (SCREEN_W - 142, SCREEN_H - 108))
            pygame.draw.rect(self.screen, (90, 90, 130), (SCREEN_W - 142, SCREEN_H - 108, 130, 96), 2)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset()
                    elif event.key == pygame.K_SPACE and not self.game_over:
                        self.started = True

            prev_paddle_x = self.paddle_x

            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.paddle_x -= KEY_PADDLE_SPEED
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.paddle_x += KEY_PADDLE_SPEED

            ret, frame = (False, None)
            if self.camera_ready:
                ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                self.tracker.process_frame(frame)
                self.tracker.draw_landmarks(frame)
                self.draw_cam_overlay(frame)

                pos = self.tracker.get_fingertip_pos("index", SCREEN_W, SCREEN_H)
                if pos:
                    paddle_w = self.current_paddle_width()
                    target_x = pos[0] - paddle_w / 2
                    self.paddle_x += (target_x - self.paddle_x) * 0.35

                pinch = False
                pdist = self.tracker.get_pinch_distance(SCREEN_W, SCREEN_H)
                if pdist is not None:
                    pinch = pdist < 42
                if pinch and not self.last_pinch:
                    self.activate_boost()
                    self.started = True
                self.last_pinch = pinch
            else:
                self.tracker.hand_detected = False

            paddle_w = self.current_paddle_width()
            self.paddle_x = max(0, min(SCREEN_W - paddle_w, self.paddle_x))
            self.paddle_vx = self.paddle_x - prev_paddle_x

            if not self.started and not self.game_over and not self.level_cleared:
                self.ball_x = self.paddle_x + paddle_w / 2
                self.ball_y = PADDLE_Y - BALL_RADIUS - 1

            self.update()
            self.draw()
            self.clock.tick(FPS)

        self.cap.release()
        self.tracker.release()
        pygame.quit()


def main():
    game = BreakoutGame()
    game.run()


if __name__ == "__main__":
    main()
