"""
Pong — Hand Tracking Edition
Move your hand up and down to control the right paddle against an AI opponent.
"""

import sys
import os
import math
import random

import cv2
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_tracker import HandTracker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 800, 500
FPS = 60
CAM_W, CAM_H = 640, 480

# Colors
BG_COLOR = (10, 10, 30)
LINE_COLOR = (40, 40, 80)
PADDLE_PLAYER = (80, 200, 255)
PADDLE_AI = (255, 100, 100)
BALL_COLOR = (255, 255, 255)
BALL_TRAIL = (100, 100, 200)
SCORE_COLOR = (60, 60, 100)
WHITE = (255, 255, 255)
GLOW_PLAYER = (40, 100, 180)
GLOW_AI = (180, 50, 50)

# Paddle
PADDLE_W = 12
PADDLE_H = 90
PADDLE_MARGIN = 30
AI_SPEED = 4.5
AI_DEAD_ZONE = 15

# Ball
BALL_RADIUS = 8
BALL_SPEED_INITIAL = 5
BALL_SPEED_INCREMENT = 0.3
BALL_MAX_SPEED = 12

# Score
WIN_SCORE = 7
LEVEL_UP_SCORE_STEP = 2


class Ball:
    """The pong ball with trail effect."""

    def __init__(self):
        self.reset()
        self.trail = []

    def reset(self):
        self.x = SCREEN_W / 2
        self.y = SCREEN_H / 2
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        direction = random.choice([-1, 1])
        self.speed = BALL_SPEED_INITIAL
        self.vx = direction * self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)
        self.trail = []

    def update(self):
        self.trail.append((self.x, self.y))
        if len(self.trail) > 10:
            self.trail.pop(0)

        self.x += self.vx
        self.y += self.vy

        # Top/bottom bounce
        if self.y - BALL_RADIUS <= 0:
            self.y = BALL_RADIUS
            self.vy = abs(self.vy)
        elif self.y + BALL_RADIUS >= SCREEN_H:
            self.y = SCREEN_H - BALL_RADIUS
            self.vy = -abs(self.vy)

    def speed_up(self):
        self.speed = min(self.speed + BALL_SPEED_INCREMENT, BALL_MAX_SPEED)
        # Preserve direction
        angle = math.atan2(self.vy, self.vx)
        self.vx = self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)

    def draw(self, surface):
        # Trail
        if self.trail:
            for i, (tx, ty) in enumerate(self.trail):
                alpha = (i + 1) / len(self.trail) * 0.4
                r = max(2, int(BALL_RADIUS * alpha))
                color = (
                    int(BALL_TRAIL[0] * alpha),
                    int(BALL_TRAIL[1] * alpha),
                    int(BALL_TRAIL[2] * alpha),
                )
                pygame.draw.circle(surface, color, (int(tx), int(ty)), r)

        # Glow
        glow_surf = pygame.Surface((BALL_RADIUS * 6, BALL_RADIUS * 6), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (255, 255, 255, 30), (BALL_RADIUS * 3, BALL_RADIUS * 3), BALL_RADIUS * 3)
        surface.blit(glow_surf, (int(self.x) - BALL_RADIUS * 3, int(self.y) - BALL_RADIUS * 3))

        # Ball
        pygame.draw.circle(surface, BALL_COLOR, (int(self.x), int(self.y)), BALL_RADIUS)


class Paddle:
    """A pong paddle."""

    def __init__(self, x, color, glow_color):
        self.x = x
        self.y = SCREEN_H / 2 - PADDLE_H / 2
        self.w = PADDLE_W
        self.h = PADDLE_H
        self.color = color
        self.glow_color = glow_color

    def get_rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

    def draw(self, surface):
        rect = self.get_rect()
        # Glow
        glow_rect = rect.inflate(10, 10)
        glow_surf = pygame.Surface((glow_rect.w, glow_rect.h), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.glow_color, 60), (0, 0, glow_rect.w, glow_rect.h), border_radius=6)
        surface.blit(glow_surf, glow_rect.topleft)
        # Paddle
        pygame.draw.rect(surface, self.color, rect, border_radius=4)

    def clamp(self):
        self.y = max(0, min(SCREEN_H - self.h, self.y))


class PongGame:
    """Main Pong game class."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🏓 Pong — Hand Tracking")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_score = pygame.font.SysFont("Segoe UI", 80, bold=True)
        self.font_medium = pygame.font.SysFont("Segoe UI", 36, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 22)

        # Camera & tracker
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ready = self.cap.isOpened()
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)

        self.cam_surface = None
        self.reset()

    def reset(self):
        self.player = Paddle(SCREEN_W - PADDLE_MARGIN - PADDLE_W, PADDLE_PLAYER, GLOW_PLAYER)
        self.ai = Paddle(PADDLE_MARGIN, PADDLE_AI, GLOW_AI)
        self.ball = Ball()
        self.player_score = 0
        self.ai_score = 0
        self.level = 1
        self.ai_speed = AI_SPEED
        self.game_over = False
        self.winner = None
        self.rally = 0

    def update_difficulty(self):
        total_points = self.player_score + self.ai_score
        self.level = min(8, 1 + total_points // LEVEL_UP_SCORE_STEP)
        self.ai_speed = AI_SPEED + (self.level - 1) * 0.28

    def update_ai(self):
        """Simple AI that tracks the ball."""
        target_y = self.ball.y - self.ai.h / 2

        # Add some imperfection
        if abs(self.ai.y + self.ai.h / 2 - self.ball.y) > AI_DEAD_ZONE:
            if self.ai.y < target_y:
                self.ai.y += self.ai_speed
            else:
                self.ai.y -= self.ai_speed
        self.ai.clamp()

    def check_collision(self):
        """Check ball-paddle collisions."""
        ball_rect = pygame.Rect(
            self.ball.x - BALL_RADIUS, self.ball.y - BALL_RADIUS,
            BALL_RADIUS * 2, BALL_RADIUS * 2,
        )

        # Player paddle
        if ball_rect.colliderect(self.player.get_rect()) and self.ball.vx > 0:
            self.ball.x = self.player.x - BALL_RADIUS
            # Angle based on where ball hits paddle
            relative_y = (self.ball.y - self.player.y) / self.player.h - 0.5
            angle = relative_y * math.pi / 3
            self.ball.vx = -self.ball.speed * math.cos(angle)
            self.ball.vy = self.ball.speed * math.sin(angle)
            self.ball.speed_up()
            self.rally += 1

        # AI paddle
        if ball_rect.colliderect(self.ai.get_rect()) and self.ball.vx < 0:
            self.ball.x = self.ai.x + self.ai.w + BALL_RADIUS
            relative_y = (self.ball.y - self.ai.y) / self.ai.h - 0.5
            angle = relative_y * math.pi / 3
            self.ball.vx = self.ball.speed * math.cos(angle)
            self.ball.vy = self.ball.speed * math.sin(angle)
            self.ball.speed_up()
            self.rally += 1

    def check_score(self):
        """Check if ball went past a paddle."""
        if self.ball.x < 0:
            self.player_score += 1
            self.rally = 0
            if self.player_score >= WIN_SCORE:
                self.game_over = True
                self.winner = "You"
            else:
                self.ball.reset()
                self.update_difficulty()

        elif self.ball.x > SCREEN_W:
            self.ai_score += 1
            self.rally = 0
            if self.ai_score >= WIN_SCORE:
                self.game_over = True
                self.winner = "CPU"
            else:
                self.ball.reset()
                self.update_difficulty()

    def update(self):
        if self.game_over:
            return

        self.ball.update()
        self.update_ai()
        self.check_collision()
        self.check_score()

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (120, 90))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw(self):
        self.screen.fill(BG_COLOR)

        # Center line
        for y in range(0, SCREEN_H, 20):
            pygame.draw.rect(self.screen, LINE_COLOR, (SCREEN_W // 2 - 2, y, 4, 10))

        # Scores
        p_score_text = self.font_score.render(str(self.player_score), True, SCORE_COLOR)
        a_score_text = self.font_score.render(str(self.ai_score), True, SCORE_COLOR)
        self.screen.blit(p_score_text, (SCREEN_W * 3 // 4 - p_score_text.get_width() // 2, 20))
        self.screen.blit(a_score_text, (SCREEN_W // 4 - a_score_text.get_width() // 2, 20))

        # Labels
        you_label = self.font_small.render("YOU", True, PADDLE_PLAYER)
        cpu_label = self.font_small.render("CPU", True, PADDLE_AI)
        self.screen.blit(you_label, (SCREEN_W * 3 // 4 - you_label.get_width() // 2, 100))
        self.screen.blit(cpu_label, (SCREEN_W // 4 - cpu_label.get_width() // 2, 100))

        lvl_text = self.font_small.render(f"Level {self.level}", True, (165, 195, 245))
        self.screen.blit(lvl_text, (SCREEN_W // 2 - lvl_text.get_width() // 2, 16))

        # Rally counter
        if self.rally > 2:
            rally_text = self.font_small.render(f"Rally: {self.rally}", True, (150, 150, 200))
            self.screen.blit(rally_text, (SCREEN_W // 2 - rally_text.get_width() // 2, SCREEN_H - 35))

        # Paddles & ball
        self.ai.draw(self.screen)
        self.player.draw(self.screen)
        self.ball.draw(self.screen)

        # Camera overlay
        if self.cam_surface:
            self.screen.blit(self.cam_surface, (SCREEN_W // 2 - 60, SCREEN_H - 100))
            pygame.draw.rect(self.screen, (60, 60, 100), (SCREEN_W // 2 - 60, SCREEN_H - 100, 120, 90), 2)

        # Game over
        if self.game_over:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (0, 0))

            color = PADDLE_PLAYER if self.winner == "You" else PADDLE_AI
            win_text = self.font_medium.render(f"{self.winner} Win{'s' if self.winner == 'CPU' else ''}!", True, color)
            self.screen.blit(win_text, (SCREEN_W // 2 - win_text.get_width() // 2, SCREEN_H // 2 - 50))

            score_text = self.font_small.render(
                f"{self.player_score} - {self.ai_score}", True, WHITE,
            )
            self.screen.blit(score_text, (SCREEN_W // 2 - score_text.get_width() // 2, SCREEN_H // 2 + 10))

            restart = self.font_small.render("Press R to restart  |  ESC to quit", True, (180, 180, 180))
            self.screen.blit(restart, (SCREEN_W // 2 - restart.get_width() // 2, SCREEN_H // 2 + 60))
        else:
            hint = self.font_small.render("Hand tracking or W/S / Up/Down", True, (130, 140, 180))
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 28))

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

            # Camera & hand tracking
            ret, frame = (False, None)
            if self.camera_ready:
                ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                self.tracker.process_frame(frame)
                self.tracker.draw_landmarks(frame)
                self.draw_cam_overlay(frame)

                # Control paddle with index finger Y position
                pos = self.tracker.get_fingertip_pos("index", SCREEN_W, SCREEN_H)
                if pos:
                    target_y = pos[1] - self.player.h / 2
                    # Smooth movement
                    self.player.y += (target_y - self.player.y) * 0.3
                    self.player.clamp()
            else:
                self.tracker.hand_detected = False

            # Keyboard fallback/assist
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.player.y -= 6
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.player.y += 6
            self.player.clamp()

            self.update()
            self.draw()
            self.clock.tick(FPS)

        self.cap.release()
        self.tracker.release()
        pygame.quit()


def main():
    game = PongGame()
    game.run()


if __name__ == "__main__":
    main()
