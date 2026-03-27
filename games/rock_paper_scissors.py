"""
Rock Paper Scissors — Hand Tracking Edition
Uses a TensorFlow model to classify your hand gesture and plays against the CPU.
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
SCREEN_W, SCREEN_H = 700, 500
FPS = 30
CAM_W, CAM_H = 640, 480

# Colors
BG_TOP = (20, 10, 40)
BG_BOTTOM = (50, 20, 80)
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
SILVER = (180, 180, 200)
RED = (255, 80, 80)
GREEN = (80, 255, 120)
BLUE = (80, 150, 255)
PURPLE = (180, 100, 255)
DARK_PANEL = (30, 20, 50)

GESTURE_COLORS = {
    "rock": (180, 100, 80),
    "paper": (80, 180, 120),
    "scissors": (80, 120, 220),
}

GESTURE_EMOJIS = {
    "rock": "✊",
    "paper": "✋",
    "scissors": "✌️",
}

BEST_OF = 5
COUNTDOWN_SECS = 3
RESULT_DISPLAY_SECS = 2.0
MIN_TF_CONFIDENCE = 0.4


def who_wins(player, cpu):
    """Returns 'player', 'cpu', or 'draw'."""
    if player == cpu:
        return "draw"
    wins = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
    if wins[player] == cpu:
        return "player"
    return "cpu"


class RockPaperScissorsGame:
    """Main RPS game class."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("✊✋✌️ Rock Paper Scissors — Hand Tracking")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_huge = pygame.font.SysFont("Segoe UI", 80, bold=True)
        self.font_large = pygame.font.SysFont("Segoe UI", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Segoe UI", 32, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 22)

        # Camera & tracker
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ready = self.cap.isOpened()
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)

        # TF gesture classifier
        self.classifier = None
        self.tf_available = True
        self.tf_error = ""
        try:
            from gesture_model import GestureClassifier
            self.classifier = GestureClassifier()
        except Exception as exc:
            self.tf_available = False
            self.tf_error = str(exc)

        # Background gradient
        self.bg_surface = pygame.Surface((SCREEN_W, SCREEN_H))
        for y in range(SCREEN_H):
            t = y / SCREEN_H
            r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t)
            g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t)
            b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t)
            pygame.draw.line(self.bg_surface, (r, g, b), (0, y), (SCREEN_W, y))

        self.cam_surface = None
        self.reset()

    def reset(self):
        self.player_score = 0
        self.cpu_score = 0
        self.round_num = 0
        self.state = "waiting"  # waiting, countdown, result, game_over
        self.countdown_start = 0
        self.result_start = 0
        self.player_gesture = None
        self.cpu_gesture = None
        self.round_result = None
        self.current_prediction = None
        self.current_confidence = 0.0
        self.match_winner = None

    def start_countdown(self):
        self.state = "countdown"
        self.countdown_start = time.time()
        self.player_gesture = None
        self.cpu_gesture = None
        self.round_result = None

    def capture_gestures(self):
        """Capture player gesture via TF model and generate CPU choice."""
        # Use TF model prediction
        gesture, confidence = (None, 0.0)
        if self.classifier is not None:
            gesture, confidence = self.classifier.predict_from_tracker(self.tracker)

        if gesture and confidence > MIN_TF_CONFIDENCE:
            self.player_gesture = gesture
        else:
            # Fallback to hand_tracker basic gesture
            basic = self.tracker.get_gesture()
            if basic == "fist":
                self.player_gesture = "rock"
            elif basic == "open":
                self.player_gesture = "paper"
            elif basic == "peace":
                self.player_gesture = "scissors"
            else:
                self.player_gesture = random.choice(["rock", "paper", "scissors"])

        self.cpu_gesture = random.choice(["rock", "paper", "scissors"])
        self.round_result = who_wins(self.player_gesture, self.cpu_gesture)

        if self.round_result == "player":
            self.player_score += 1
        elif self.round_result == "cpu":
            self.cpu_score += 1

        self.round_num += 1
        self.state = "result"
        self.result_start = time.time()

    def update(self):
        now = time.time()

        if self.state == "countdown":
            elapsed = now - self.countdown_start
            if elapsed >= COUNTDOWN_SECS:
                self.capture_gestures()

        elif self.state == "result":
            if now - self.result_start >= RESULT_DISPLAY_SECS:
                # Check if game is over
                if self.player_score >= (BEST_OF + 1) // 2 or self.cpu_score >= (BEST_OF + 1) // 2:
                    self.state = "game_over"
                    self.match_winner = "You" if self.player_score > self.cpu_score else "CPU"
                elif self.round_num >= BEST_OF:
                    self.state = "game_over"
                    if self.player_score > self.cpu_score:
                        self.match_winner = "You"
                    elif self.cpu_score > self.player_score:
                        self.match_winner = "CPU"
                    else:
                        self.match_winner = "Draw"
                else:
                    self.state = "waiting"

        # Update TF prediction continuously
        if self.tracker.hand_detected and self.classifier is not None:
            gesture, confidence = self.classifier.predict_from_tracker(self.tracker)
            self.current_prediction = gesture
            self.current_confidence = confidence

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (180, 135))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw_gesture_card(self, x, y, label, gesture, color, is_winner=False):
        """Draw a gesture card."""
        card_w, card_h = 200, 200

        # Card background
        card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        pygame.draw.rect(card_surf, (*DARK_PANEL, 200), (0, 0, card_w, card_h), border_radius=15)
        if is_winner:
            pygame.draw.rect(card_surf, (*GOLD, 200), (0, 0, card_w, card_h), 3, border_radius=15)
        else:
            pygame.draw.rect(card_surf, (*color, 100), (0, 0, card_w, card_h), 2, border_radius=15)
        self.screen.blit(card_surf, (x, y))

        # Label
        label_text = self.font_small.render(label, True, color)
        self.screen.blit(label_text, (x + card_w // 2 - label_text.get_width() // 2, y + 15))

        if gesture:
            # Emoji
            emoji = GESTURE_EMOJIS.get(gesture, "?")
            emoji_text = self.font_huge.render(emoji, True, WHITE)
            self.screen.blit(emoji_text, (x + card_w // 2 - emoji_text.get_width() // 2, y + 50))

            # Gesture name
            name_text = self.font_medium.render(gesture.upper(), True, GESTURE_COLORS.get(gesture, WHITE))
            self.screen.blit(name_text, (x + card_w // 2 - name_text.get_width() // 2, y + 150))

    def draw(self):
        self.screen.blit(self.bg_surface, (0, 0))

        # Title
        title = self.font_medium.render("Rock  Paper  Scissors", True, PURPLE)
        self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 10))

        # Score display
        score_text = self.font_medium.render(
            f"YOU  {self.player_score}  -  {self.cpu_score}  CPU", True, WHITE
        )
        self.screen.blit(score_text, (SCREEN_W // 2 - score_text.get_width() // 2, 50))

        # Round info
        round_text = self.font_small.render(
            f"Best of {BEST_OF}  •  Round {self.round_num + 1}", True, SILVER
        )
        self.screen.blit(round_text, (SCREEN_W // 2 - round_text.get_width() // 2, 90))

        # State-specific drawing
        if self.state == "waiting":
            instr = self.font_medium.render("Press SPACE to play!", True, GREEN)
            self.screen.blit(instr, (SCREEN_W // 2 - instr.get_width() // 2, 250))

            if not self.tf_available:
                warn = self.font_small.render("TensorFlow model unavailable - using basic gesture fallback", True, SILVER)
                self.screen.blit(warn, (SCREEN_W // 2 - warn.get_width() // 2, 345))
                if self.tf_error:
                    err = self.font_small.render(f"Reason: {self.tf_error[:58]}", True, (160, 160, 185))
                    self.screen.blit(err, (SCREEN_W // 2 - err.get_width() // 2, 372))

            if self.current_prediction:
                pred_text = self.font_small.render(
                    f"Detected: {self.current_prediction} ({self.current_confidence:.0%})",
                    True, GESTURE_COLORS.get(self.current_prediction, WHITE),
                )
                self.screen.blit(pred_text, (SCREEN_W // 2 - pred_text.get_width() // 2, 310))

        elif self.state == "countdown":
            elapsed = time.time() - self.countdown_start
            count = COUNTDOWN_SECS - int(elapsed)
            count = max(1, count)

            count_text = self.font_huge.render(str(count), True, GOLD)
            self.screen.blit(count_text, (SCREEN_W // 2 - count_text.get_width() // 2, 200))

            hint = self.font_small.render("Show your gesture!", True, WHITE)
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, 320))

        elif self.state == "result":
            # Draw gesture cards
            player_win = self.round_result == "player"
            cpu_win = self.round_result == "cpu"

            self.draw_gesture_card(100, 150, "YOU", self.player_gesture, BLUE, player_win)
            self.draw_gesture_card(400, 150, "CPU", self.cpu_gesture, RED, cpu_win)

            # VS
            vs_text = self.font_large.render("VS", True, PURPLE)
            self.screen.blit(vs_text, (SCREEN_W // 2 - vs_text.get_width() // 2, 220))

            # Result
            if self.round_result == "player":
                result_text = self.font_medium.render("You win this round!", True, GREEN)
            elif self.round_result == "cpu":
                result_text = self.font_medium.render("CPU wins this round!", True, RED)
            else:
                result_text = self.font_medium.render("It's a draw!", True, SILVER)
            self.screen.blit(result_text, (SCREEN_W // 2 - result_text.get_width() // 2, 380))

        elif self.state == "game_over":
            if self.match_winner == "You":
                color = GREEN
                msg = "🎉 YOU WIN THE MATCH! 🎉"
            elif self.match_winner == "CPU":
                color = RED
                msg = "CPU Wins the Match"
            else:
                color = SILVER
                msg = "Match is a Draw!"

            win_text = self.font_large.render(msg, True, color)
            self.screen.blit(win_text, (SCREEN_W // 2 - win_text.get_width() // 2, 200))

            final = self.font_medium.render(
                f"Final: {self.player_score} - {self.cpu_score}", True, WHITE
            )
            self.screen.blit(final, (SCREEN_W // 2 - final.get_width() // 2, 280))

            restart = self.font_small.render("Press R to play again  |  ESC to quit", True, (180, 180, 180))
            self.screen.blit(restart, (SCREEN_W // 2 - restart.get_width() // 2, 340))

        # Camera overlay
        if self.cam_surface:
            cam_x = SCREEN_W - 190
            cam_y = SCREEN_H - 145
            self.screen.blit(self.cam_surface, (cam_x, cam_y))
            pygame.draw.rect(self.screen, PURPLE, (cam_x, cam_y, 180, 135), 2)

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
                    elif event.key == pygame.K_SPACE and self.state == "waiting":
                        self.start_countdown()
                    elif event.key == pygame.K_r and self.state == "game_over":
                        self.reset()

            # Camera & tracking
            ret, frame = (False, None)
            if self.camera_ready:
                ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                self.tracker.process_frame(frame)
                self.tracker.draw_landmarks(frame)
                self.draw_cam_overlay(frame)
            else:
                self.tracker.hand_detected = False

            self.update()
            self.draw()
            self.clock.tick(FPS)

        self.cap.release()
        self.tracker.release()
        pygame.quit()


def main():
    game = RockPaperScissorsGame()
    game.run()


if __name__ == "__main__":
    main()
