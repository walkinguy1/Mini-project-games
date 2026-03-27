"""
Minesweeper Lite — Hand Tracking Edition
Hover your finger over a cell and pinch to reveal it.
"""

import sys
import os
import random
from collections import deque

import cv2
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_tracker import HandTracker


SCREEN_W, SCREEN_H = 760, 680
FPS = 60
CAM_W, CAM_H = 640, 480

GRID_W = 10
GRID_H = 10
LEVEL_MINES = [12, 14, 16, 18]
CELL = 54
BOARD_X = (SCREEN_W - GRID_W * CELL) // 2
BOARD_Y = 90

BG = (24, 24, 36)
PANEL = (40, 40, 58)
CELL_COVER = (72, 76, 104)
CELL_OPEN = (212, 216, 230)
GRID_LINE = (95, 100, 130)
WHITE = (246, 246, 252)
RED = (240, 90, 90)
GREEN = (120, 220, 150)

NUM_COLORS = {
    1: (70, 100, 220),
    2: (60, 150, 80),
    3: (200, 70, 70),
    4: (120, 70, 180),
    5: (160, 80, 60),
    6: (50, 150, 150),
    7: (60, 60, 80),
    8: (100, 100, 100),
}


class MinesweeperLiteGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("💣 Minesweeper Lite — Hand Tracking")
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
        self.last_pinch = False
        self.max_level = len(LEVEL_MINES)
        self.reset_campaign()

    def setup_level(self, level):
        self.level = level
        self.mine_count = LEVEL_MINES[level - 1]
        self.board = [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.revealed = [[False for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.flagged = [[False for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.mines = set()
        self.first_move = True
        self.over = False
        self.win = False
        self.level_cleared = False
        self.transition_until = 0.0

    def reset_campaign(self):
        self.campaign_complete = False
        self.setup_level(1)

    def reset(self):
        self.reset_campaign()

    def in_bounds(self, r, c):
        return 0 <= r < GRID_H and 0 <= c < GRID_W

    def neighbors(self, r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if self.in_bounds(rr, cc):
                    yield rr, cc

    def place_mines(self, safe_r, safe_c):
        banned = {(safe_r, safe_c)}
        banned.update(self.neighbors(safe_r, safe_c))

        cells = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if (r, c) not in banned]
        random.shuffle(cells)
        self.mines = set(cells[:self.mine_count])

        for r in range(GRID_H):
            for c in range(GRID_W):
                if (r, c) in self.mines:
                    self.board[r][c] = -1
                else:
                    self.board[r][c] = sum((rr, cc) in self.mines for rr, cc in self.neighbors(r, c))

    def reveal_cell(self, r, c):
        if not self.in_bounds(r, c) or self.revealed[r][c] or self.flagged[r][c] or self.over:
            return

        if self.first_move:
            self.place_mines(r, c)
            self.first_move = False

        if (r, c) in self.mines:
            self.revealed[r][c] = True
            self.over = True
            self.win = False
            return

        queue = deque([(r, c)])
        while queue:
            rr, cc = queue.popleft()
            if self.revealed[rr][cc]:
                continue
            self.revealed[rr][cc] = True
            if self.board[rr][cc] == 0:
                for nr, nc in self.neighbors(rr, cc):
                    if not self.revealed[nr][nc] and (nr, nc) not in self.mines:
                        queue.append((nr, nc))

        self.check_win()

    def check_win(self):
        safe_total = GRID_W * GRID_H - len(self.mines)
        opened = sum(self.revealed[r][c] for r in range(GRID_H) for c in range(GRID_W) if (r, c) not in self.mines)
        if opened == safe_total:
            if self.level < self.max_level:
                self.level_cleared = True
                self.transition_until = pygame.time.get_ticks() + 1100
            else:
                self.over = True
                self.win = True
                self.campaign_complete = True

    def toggle_flag(self, r, c):
        if not self.in_bounds(r, c) or self.revealed[r][c] or self.over or self.level_cleared:
            return
        self.flagged[r][c] = not self.flagged[r][c]

    def update_level_transition(self):
        if self.level_cleared and pygame.time.get_ticks() >= self.transition_until:
            self.setup_level(self.level + 1)

    def cell_from_pos(self, pos):
        if pos is None:
            return None
        x, y = pos
        if x < BOARD_X or y < BOARD_Y:
            return None
        c = (x - BOARD_X) // CELL
        r = (y - BOARD_Y) // CELL
        if self.in_bounds(r, c):
            return int(r), int(c)
        return None

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (130, 96))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw(self):
        self.screen.fill(BG)

        title = self.font_mid.render(f"Minesweeper Lite  •  Level {self.level}/{self.max_level}", True, WHITE)
        self.screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 20))

        flagged_count = sum(1 for r in range(GRID_H) for c in range(GRID_W) if self.flagged[r][c])
        level_info = self.font_small.render(
            f"Mines: {self.mine_count}   Flags: {flagged_count}/{self.mine_count}",
            True,
            (200, 200, 220),
        )
        self.screen.blit(level_info, (24, 28))

        hovered = self.cell_from_pos(self.finger_pos)

        for r in range(GRID_H):
            for c in range(GRID_W):
                x = BOARD_X + c * CELL
                y = BOARD_Y + r * CELL
                rect = pygame.Rect(x, y, CELL, CELL)

                is_open = self.revealed[r][c] or (self.over and (r, c) in self.mines)
                color = CELL_OPEN if is_open else CELL_COVER
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID_LINE, rect, 1)

                if hovered == (r, c) and not self.over:
                    pygame.draw.rect(self.screen, (160, 210, 255), rect, 3)

                if is_open:
                    val = self.board[r][c]
                    if val == -1:
                        pygame.draw.circle(self.screen, RED, rect.center, CELL // 4)
                    elif val > 0:
                        txt = self.font_mid.render(str(val), True, NUM_COLORS.get(val, (0, 0, 0)))
                        self.screen.blit(txt, (x + CELL // 2 - txt.get_width() // 2, y + CELL // 2 - txt.get_height() // 2))
                elif self.flagged[r][c]:
                    flag = self.font_mid.render("⚑", True, (255, 190, 90))
                    self.screen.blit(flag, (x + CELL // 2 - flag.get_width() // 2, y + CELL // 2 - flag.get_height() // 2))

        if self.finger_pos:
            pygame.draw.circle(self.screen, (120, 210, 255), self.finger_pos, 12, 2)

        if self.level_cleared:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (0, 0))
            txt = self.font_big.render("Level Cleared!", True, GREEN)
            self.screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2, SCREEN_H // 2 - 60))
            hint = self.font_small.render("Loading next level...", True, WHITE)
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H // 2 + 12))

        if self.over:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            self.screen.blit(overlay, (0, 0))
            msg = "You Win!" if self.win else "Boom!"
            color = GREEN if self.win else RED
            txt = self.font_big.render(msg, True, color)
            self.screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2, SCREEN_H // 2 - 70))
            hint = self.font_small.render("Press R to restart campaign  |  ESC to quit", True, WHITE)
            self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H // 2 + 10))
        else:
            hint = "Pinch/left-click reveal • Right-click/F flag"
            if not self.camera_ready:
                hint = "Camera unavailable - left-click reveal, right-click flag"
            htxt = self.font_small.render(hint, True, (200, 200, 220))
            self.screen.blit(htxt, (SCREEN_W // 2 - htxt.get_width() // 2, SCREEN_H - 30))

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
                    elif event.key == pygame.K_f:
                        target = self.cell_from_pos(self.finger_pos)
                        if target is not None:
                            self.toggle_flag(*target)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                    if not self.level_cleared:
                        cell = self.cell_from_pos(event.pos)
                        if cell is not None:
                            if event.button == 1:
                                self.reveal_cell(*cell)
                            else:
                                self.toggle_flag(*cell)

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
                if pinch and not self.last_pinch and not self.level_cleared:
                    cell = self.cell_from_pos(self.finger_pos)
                    if cell is not None:
                        self.reveal_cell(*cell)
                self.last_pinch = pinch
            else:
                self.finger_pos = None
                self.tracker.hand_detected = False
                self.last_pinch = False

            self.update_level_transition()
            self.draw()
            self.clock.tick(FPS)

        self.cap.release()
        self.tracker.release()
        pygame.quit()


def main():
    game = MinesweeperLiteGame()
    game.run()


if __name__ == "__main__":
    main()
