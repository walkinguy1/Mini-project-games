"""
Memory Match — Hand Tracking Edition
Hover over cards and pinch to flip; match all pairs with minimum moves.
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

BG = (22, 22, 38)
WHITE = (250, 250, 255)
ACCENT = (130, 200, 255)
CARD_BACK = (60, 70, 110)
MATCHED = (80, 170, 115)

GRID_Y = 120
CARD_GAP = 14
LEVELS = [(4, 4), (4, 5), (4, 6)]

SYMBOLS = ["🍎", "🍌", "🍇", "🍉", "🍒", "🥝", "🍍", "🥥", "🍋", "🍓", "🥭", "🍑"]
BASE_PREVIEW_SECS = 2.2
BASE_MISMATCH_DELAY = 0.65


class MemoryMatchGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🃏 Memory Match — Hand Tracking")
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.SysFont("Segoe UI", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Segoe UI", 32, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 20)
        self.font_emoji = pygame.font.SysFont("Segoe UI Emoji", 42)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ready = self.cap.isOpened()
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)

        self.cam_surface = None
        self.last_pinch = False
        self.max_level = len(LEVELS)
        self.reset_campaign()

    def setup_level(self, level):
        self.level = level
        self.rows, self.cols = LEVELS[level - 1]

        max_board_w = SCREEN_W - 110
        max_board_h = SCREEN_H - 220
        self.card_w = min(140, (max_board_w - (self.cols - 1) * CARD_GAP) // self.cols)
        self.card_h = min(90, (max_board_h - (self.rows - 1) * CARD_GAP) // self.rows)
        self.grid_w = self.cols * self.card_w + (self.cols - 1) * CARD_GAP
        self.grid_h = self.rows * self.card_h + (self.rows - 1) * CARD_GAP
        self.grid_x = (SCREEN_W - self.grid_w) // 2

        pair_count = (self.rows * self.cols) // 2
        values = SYMBOLS[:pair_count] * 2
        random.shuffle(values)

        self.cards = []
        idx = 0
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                x = self.grid_x + c * (self.card_w + CARD_GAP)
                y = GRID_Y + r * (self.card_h + CARD_GAP)
                row.append({
                    "rect": pygame.Rect(x, y, self.card_w, self.card_h),
                    "value": values[idx],
                    "revealed": False,
                    "matched": False,
                })
                idx += 1
            self.cards.append(row)

        self.first_pick = None
        self.second_pick = None
        self.lock_until = 0.0
        self.preview_secs = max(1.0, BASE_PREVIEW_SECS - (self.level - 1) * 0.35)
        self.mismatch_delay = max(0.35, BASE_MISMATCH_DELAY - (self.level - 1) * 0.08)
        self.preview_until = time.time() + self.preview_secs
        self.moves = 0
        self.matches = 0
        self.target_pairs = pair_count
        self.level_cleared = False
        self.level_clear_until = 0.0
        self.game_over = False
        self.finger_pos = None
        self.hints_left = 1

    def reset_campaign(self):
        self.setup_level(1)

    def reset(self):
        self.reset_campaign()

    def card_at(self, pos):
        if pos is None:
            return None
        for r, row in enumerate(self.cards):
            for c, card in enumerate(row):
                if card["rect"].collidepoint(pos):
                    return r, c
        return None

    def reveal_card(self, rc):
        if rc is None or self.game_over:
            return
        if time.time() < self.preview_until:
            return
        r, c = rc
        card = self.cards[r][c]
        if card["matched"] or card["revealed"]:
            return

        card["revealed"] = True
        if self.first_pick is None:
            self.first_pick = (r, c)
        elif self.second_pick is None:
            self.second_pick = (r, c)
            self.moves += 1
            self.check_pair()

    def check_pair(self):
        if self.first_pick is None or self.second_pick is None:
            return
        r1, c1 = self.first_pick
        r2, c2 = self.second_pick
        a = self.cards[r1][c1]
        b = self.cards[r2][c2]

        if a["value"] == b["value"]:
            a["matched"] = True
            b["matched"] = True
            self.matches += 1
            self.first_pick = None
            self.second_pick = None
            if self.matches == self.target_pairs:
                if self.level < self.max_level:
                    self.level_cleared = True
                    self.level_clear_until = time.time() + 1.1
                else:
                    self.game_over = True
        else:
            self.lock_until = time.time() + self.mismatch_delay

    def update(self):
        if self.level_cleared and time.time() >= self.level_clear_until:
            self.setup_level(self.level + 1)
            return

        if self.preview_until and time.time() >= self.preview_until:
            for row in self.cards:
                for card in row:
                    if not card["matched"]:
                        card["revealed"] = False
            self.preview_until = 0.0

        if self.lock_until and time.time() >= self.lock_until:
            r1, c1 = self.first_pick
            r2, c2 = self.second_pick
            self.cards[r1][c1]["revealed"] = False
            self.cards[r2][c2]["revealed"] = False
            self.first_pick = None
            self.second_pick = None
            self.lock_until = 0.0

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (130, 96))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw(self):
        self.screen.fill(BG)

        title = self.font_medium.render(f"Memory Match  •  Level {self.level}/{self.max_level}", True, ACCENT)
        self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 20))
        hud = self.font_small.render(
            f"Moves: {self.moves}   Pairs: {self.matches}/{self.target_pairs}   Hints: {self.hints_left}",
            True,
            WHITE,
        )
        self.screen.blit(hud, (SCREEN_W // 2 - hud.get_width() // 2, 62))

        hovered = self.card_at(self.finger_pos)
        for r, row in enumerate(self.cards):
            for c, card in enumerate(row):
                rect = card["rect"]
                color = CARD_BACK
                if card["matched"]:
                    color = MATCHED
                show_face = card["revealed"] or card["matched"] or (self.preview_until > time.time())
                if show_face:
                    pygame.draw.rect(self.screen, (230, 230, 245), rect, border_radius=10)
                    emoji = self.font_emoji.render(card["value"], True, (30, 30, 50))
                    self.screen.blit(emoji, (rect.centerx - emoji.get_width() // 2, rect.centery - emoji.get_height() // 2))
                else:
                    pygame.draw.rect(self.screen, color, rect, border_radius=10)
                border = (180, 220, 255) if hovered == (r, c) else (90, 90, 120)
                pygame.draw.rect(self.screen, border, rect, 2, border_radius=10)

        if self.finger_pos:
            pygame.draw.circle(self.screen, ACCENT, self.finger_pos, 12, 2)

        if self.level_cleared:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (0, 0))
            done = self.font_large.render("Level Cleared!", True, WHITE)
            self.screen.blit(done, (SCREEN_W // 2 - done.get_width() // 2, SCREEN_H // 2 - 40))
            nxt = self.font_small.render("Loading next level...", True, (210, 210, 225))
            self.screen.blit(nxt, (SCREEN_W // 2 - nxt.get_width() // 2, SCREEN_H // 2 + 20))

        if self.game_over:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            self.screen.blit(overlay, (0, 0))
            done = self.font_large.render("All Pairs Matched!", True, WHITE)
            self.screen.blit(done, (SCREEN_W // 2 - done.get_width() // 2, SCREEN_H // 2 - 70))
            moves = self.font_medium.render(f"Moves: {self.moves}", True, WHITE)
            self.screen.blit(moves, (SCREEN_W // 2 - moves.get_width() // 2, SCREEN_H // 2 - 10))
            hint = self.font_small.render("Press R to restart  |  ESC to quit", True, (200, 200, 220))
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H // 2 + 46))
        else:
            hint = "Pinch to flip a highlighted card"
            if not self.camera_ready:
                hint = "Camera unavailable - click cards with mouse"
            if self.preview_until > time.time():
                hint = f"Memorize cards... {max(0.0, self.preview_until - time.time()):.1f}s"
            elif self.hints_left > 0:
                hint = "Pinch to flip • Press H for a short hint reveal"
            txt = self.font_small.render(hint, True, (200, 200, 220))
            self.screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2, SCREEN_H - 28))

        if self.cam_surface:
            self.screen.blit(self.cam_surface, (SCREEN_W - 142, SCREEN_H - 108))
            pygame.draw.rect(self.screen, (90, 90, 130), (SCREEN_W - 142, SCREEN_H - 108, 130, 96), 2)

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
                        self.reset_campaign()
                    elif event.key == pygame.K_h and not self.level_cleared and self.hints_left > 0 and not self.game_over:
                        self.preview_until = max(self.preview_until, time.time() + 1.0)
                        self.hints_left -= 1
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.lock_until == 0.0 and not self.level_cleared:
                        self.reveal_card(self.card_at(event.pos))

            ret, frame = (False, None)
            if self.camera_ready:
                ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                self.tracker.process_frame(frame)
                self.tracker.draw_landmarks(frame)
                self.draw_cam_overlay(frame)

                self.finger_pos = self.tracker.get_fingertip_pos("index", SCREEN_W, SCREEN_H)
                pinch = False
                pd = self.tracker.get_pinch_distance(SCREEN_W, SCREEN_H)
                if pd is not None:
                    pinch = pd < 42

                if (
                    pinch
                    and not self.last_pinch
                    and self.lock_until == 0.0
                    and self.preview_until <= time.time()
                    and not self.level_cleared
                ):
                    self.reveal_card(self.card_at(self.finger_pos))
                self.last_pinch = pinch
            else:
                self.finger_pos = None
                self.tracker.hand_detected = False

            self.update()
            self.draw()
            self.clock.tick(FPS)

        self.cap.release()
        self.tracker.release()
        pygame.quit()


def main():
    game = MemoryMatchGame()
    game.run()


if __name__ == "__main__":
    main()
