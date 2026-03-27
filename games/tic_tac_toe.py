"""
Tic-Tac-Toe — Hand Tracking Edition
Pinch on a cell to place your move against a CPU opponent.
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


SCREEN_W, SCREEN_H = 700, 600
FPS = 60
CAM_W, CAM_H = 640, 480

BG = (24, 24, 40)
WHITE = (245, 245, 255)
PLAYER = (100, 210, 255)
CPU = (255, 150, 150)
GRID = (130, 130, 170)

BOARD_X, BOARD_Y = 170, 120
CELL = 120
SIZE = 3
LEVEL_AI = [
    {"think": 0.45, "mistake": 0.30},
    {"think": 0.40, "mistake": 0.24},
    {"think": 0.36, "mistake": 0.19},
    {"think": 0.32, "mistake": 0.14},
    {"think": 0.28, "mistake": 0.08},
]


def winner(board):
    lines = []
    for i in range(3):
        lines.append(board[i])
        lines.append([board[0][i], board[1][i], board[2][i]])
    lines.append([board[0][0], board[1][1], board[2][2]])
    lines.append([board[0][2], board[1][1], board[2][0]])
    for ln in lines:
        if ln[0] and ln[0] == ln[1] == ln[2]:
            return ln[0]
    if all(board[r][c] for r in range(3) for c in range(3)):
        return "draw"
    return None


class TicTacToeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("⭕ Tic-Tac-Toe — Hand Tracking")
        self.clock = pygame.time.Clock()

        self.font_big = pygame.font.SysFont("Segoe UI", 50, bold=True)
        self.font_mark = pygame.font.SysFont("Segoe UI", 84, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 20)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ready = self.cap.isOpened()
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)

        self.cam_surface = None
        self.last_pinch = False
        self.finger_pos = None
        self.max_level = len(LEVEL_AI)
        self.reset_campaign()

    def start_round(self):
        self.board = [[None] * SIZE for _ in range(SIZE)]
        self.turn = "X"
        self.over = False
        self.result = None
        self.cpu_turn_at = 0.0
        self.round_transition_at = 0.0
        self.round_msg = ""
        self.level_failed = False

    def reset_campaign(self):
        self.level = 1
        self.completed = False
        self.start_round()

    def reset(self):
        self.reset_campaign()

    def ai_settings(self):
        return LEVEL_AI[self.level - 1]

    def cpu_move(self):
        empties = [(r, c) for r in range(3) for c in range(3) if self.board[r][c] is None]
        if not empties:
            return

        if random.random() < self.ai_settings()["mistake"]:
            r, c = random.choice(empties)
            self.board[r][c] = "O"
            return

        # try winning move
        for r, c in empties:
            self.board[r][c] = "O"
            if winner(self.board) == "O":
                return
            self.board[r][c] = None

        # block player winning move
        for r, c in empties:
            self.board[r][c] = "X"
            if winner(self.board) == "X":
                self.board[r][c] = "O"
                return
            self.board[r][c] = None

        # center > corners > random
        if self.board[1][1] is None:
            self.board[1][1] = "O"
            return
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        random.shuffle(corners)
        for r, c in corners:
            if self.board[r][c] is None:
                self.board[r][c] = "O"
                return
        r, c = random.choice(empties)
        self.board[r][c] = "O"

    def cell_from_pos(self, pos):
        if pos is None:
            return None
        x, y = pos
        if x < BOARD_X or y < BOARD_Y:
            return None
        col = (x - BOARD_X) // CELL
        row = (y - BOARD_Y) // CELL
        if 0 <= row < 3 and 0 <= col < 3:
            return int(row), int(col)
        return None

    def player_move(self, rc):
        if self.over or self.turn != "X" or rc is None:
            return
        r, c = rc
        if self.board[r][c] is not None:
            return
        self.board[r][c] = "X"
        self.result = winner(self.board)
        if self.result:
            self.over = True
            return

        self.turn = "O"
        self.cpu_turn_at = time.time() + self.ai_settings()["think"]

    def update(self):
        if self.over or self.turn != "O":
            return
        if time.time() < self.cpu_turn_at:
            return

        self.cpu_move()
        self.result = winner(self.board)
        if self.result:
            self.over = True
            if self.result == "X":
                if self.level < self.max_level:
                    self.round_msg = "Level Cleared!"
                    self.round_transition_at = time.time() + 1.0
                else:
                    self.completed = True
            elif self.result in ("O", "draw"):
                self.level_failed = True
            return

        self.turn = "X"
        self.cpu_turn_at = 0.0

    def update_round_transition(self):
        if self.round_transition_at and time.time() >= self.round_transition_at:
            self.level += 1
            self.start_round()

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (120, 90))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw(self):
        self.screen.fill(BG)
        title = self.font_big.render(f"Tic-Tac-Toe  •  Level {self.level}/{self.max_level}", True, WHITE)
        self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 24))

        for i in range(4):
            x = BOARD_X + i * CELL
            y = BOARD_Y + i * CELL
            pygame.draw.line(self.screen, GRID, (x, BOARD_Y), (x, BOARD_Y + 3 * CELL), 3)
            pygame.draw.line(self.screen, GRID, (BOARD_X, y), (BOARD_X + 3 * CELL, y), 3)

        hover = self.cell_from_pos(self.finger_pos)
        if hover and not self.over:
            r, c = hover
            rect = pygame.Rect(BOARD_X + c * CELL + 3, BOARD_Y + r * CELL + 3, CELL - 6, CELL - 6)
            pygame.draw.rect(self.screen, (70, 95, 140), rect, 0, border_radius=8)

        for r in range(3):
            for c in range(3):
                mark = self.board[r][c]
                if mark:
                    color = PLAYER if mark == "X" else CPU
                    txt = self.font_mark.render(mark, True, color)
                    x = BOARD_X + c * CELL + CELL // 2 - txt.get_width() // 2
                    y = BOARD_Y + r * CELL + CELL // 2 - txt.get_height() // 2
                    self.screen.blit(txt, (x, y))

        if self.finger_pos:
            pygame.draw.circle(self.screen, PLAYER, self.finger_pos, 12, 2)

        status = "Your turn (pinch to place)"
        if not self.camera_ready:
            status = "Camera unavailable - click to place"
        if not self.over and self.turn == "O":
            status = "CPU thinking..."
        if self.over:
            if self.result == "X":
                status = "You win!"
            elif self.result == "O":
                status = "CPU wins (retry level)"
            else:
                status = "Draw (retry level)"
        if self.round_transition_at:
            status = self.round_msg or "Level Cleared!"
        if self.completed:
            status = "Campaign complete!"
        stxt = self.font_small.render(status, True, WHITE)
        self.screen.blit(stxt, (SCREEN_W // 2 - stxt.get_width() // 2, SCREEN_H - 36))

        if self.over or self.completed:
            hint_text = "Press R to restart campaign  |  ESC to quit"
            if self.level_failed:
                hint_text = "Press N retry level  |  R restart campaign  |  ESC quit"
            hint = self.font_small.render(hint_text, True, (200, 200, 220))
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 16))

        if self.cam_surface:
            self.screen.blit(self.cam_surface, (SCREEN_W - 132, 12))
            pygame.draw.rect(self.screen, (110, 110, 135), (SCREEN_W - 132, 12, 120, 90), 2)

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
                    elif event.key == pygame.K_n and self.level_failed:
                        self.start_round()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self.round_transition_at and not self.level_failed:
                        self.player_move(self.cell_from_pos(event.pos))

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
                    pinch = pd < 40
                if pinch and not self.last_pinch and not self.round_transition_at and not self.level_failed:
                    self.player_move(self.cell_from_pos(self.finger_pos))
                self.last_pinch = pinch
            else:
                self.finger_pos = None
                self.tracker.hand_detected = False

            self.update()
            self.update_round_transition()
            self.draw()
            self.clock.tick(FPS)

        self.cap.release()
        self.tracker.release()
        pygame.quit()


def main():
    game = TicTacToeGame()
    game.run()


if __name__ == "__main__":
    main()
