"""
Hand Tracking Games Launcher
Main menu to select and launch any of the 12 hand-tracking mini-games.

Improvements applied:
  5. Live high scores displayed on each game card (reads scores.json)
"""

import sys
import os
import math
import time
import subprocess

import cv2
import numpy as np
import pygame

# Ensure modules can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_tracker import HandTracker

try:
    from games.score_manager import get_high_score as _shared_get_high_score
except ImportError:
    _shared_get_high_score = None


# ---------------------------------------------------------------------------
# Improvement 5: high score helper (via shared score_manager.py)
# ---------------------------------------------------------------------------
def _get_high_score(game_name: str) -> int:
    """Return the saved high score for a game, or 0 if unavailable."""
    if _shared_get_high_score is None:
        return 0
    try:
        return int(_shared_get_high_score(game_name))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 900, 650
FPS = 30
CAM_W, CAM_H = 640, 480

# Colors
BG_TOP = (10, 10, 25)
BG_BOTTOM = (30, 20, 60)
CARD_BG = (35, 30, 55)
CARD_HOVER = (55, 45, 85)
CARD_BORDER = (80, 70, 120)
CARD_HOVER_BORDER = (140, 120, 220)
WHITE = (255, 255, 255)
TITLE_COLOR = (200, 160, 255)
SUBTITLE = (150, 140, 170)
GLOW = (120, 80, 220)

# Game definitions
GAMES = [
    {
        "name": "Fruit Ninja",
        "emoji": "🍉",
        "desc": "Swipe to slice fruits",
        "color": (255, 80, 80),
        "module": "games.fruit_ninja",
        "file": "games/fruit_ninja.py",
    },
    {
        "name": "Flappy Bird",
        "emoji": "🐦",
        "desc": "Open hand to flap",
        "color": (255, 210, 50),
        "module": "games.flappy_bird",
        "file": "games/flappy_bird.py",
    },
    {
        "name": "Pong",
        "emoji": "🏓",
        "desc": "Hand controls paddle",
        "color": (80, 200, 255),
        "module": "games.pong",
        "file": "games/pong.py",
    },
    {
        "name": "Rock Paper Scissors",
        "emoji": "✊",
        "desc": "TF gesture recognition",
        "color": (200, 80, 255),
        "module": "games.rock_paper_scissors",
        "file": "games/rock_paper_scissors.py",
    },
    {
        "name": "Whack-a-Mole",
        "emoji": "🔨",
        "desc": "Poke finger to whack",
        "color": (255, 180, 50),
        "module": "games.whack_a_mole",
        "file": "games/whack_a_mole.py",
    },
    {
        "name": "Drawing Canvas",
        "emoji": "🎨",
        "desc": "Draw with fingertip",
        "color": (80, 255, 150),
        "module": "games.drawing_canvas",
        "file": "games/drawing_canvas.py",
    },
    {
        "name": "Breakout",
        "emoji": "🧱",
        "desc": "Hand paddle + pinch boost",
        "color": (255, 140, 90),
        "module": "games.breakout",
        "file": "games/breakout.py",
    },
    {
        "name": "Memory Match",
        "emoji": "🃏",
        "desc": "Match cards with pinch",
        "color": (140, 170, 255),
        "module": "games.memory_match",
        "file": "games/memory_match.py",
    },
    {
        "name": "2048",
        "emoji": "🔢",
        "desc": "Swipe gestures to merge",
        "color": (255, 210, 120),
        "module": "games.game_2048",
        "file": "games/game_2048.py",
    },
    {
        "name": "Tic-Tac-Toe",
        "emoji": "⭕",
        "desc": "Pinch to place vs CPU",
        "color": (130, 245, 195),
        "module": "games.tic_tac_toe",
        "file": "games/tic_tac_toe.py",
    },
    {
        "name": "Minesweeper Lite",
        "emoji": "💣",
        "desc": "Pinch to reveal safely",
        "color": (255, 125, 125),
        "module": "games.minesweeper_lite",
        "file": "games/minesweeper_lite.py",
    },
    {
        "name": "Maze Runner",
        "emoji": "🧭",
        "desc": "Guide runner to goal",
        "color": (140, 210, 255),
        "module": "games.maze_runner",
        "file": "games/maze_runner.py",
    },
]

# Card layout
CARD_W = 150
CARD_H = 130
CARDS_PER_ROW = 5
CARD_MARGIN = 15
GRID_START_X = (SCREEN_W - (CARD_W * CARDS_PER_ROW + CARD_MARGIN * (CARDS_PER_ROW - 1))) // 2
GRID_START_Y = 180

# Cursor
CURSOR_RADIUS = 18
HOVER_TIME_TO_SELECT = 1.2  # seconds


class Launcher:
    """Main launcher for hand tracking games."""

    def __init__(self):
        self._init_pygame()
        self._init_fonts()
        self._init_camera_tracker()

        # Background gradient
        self.bg_surface = pygame.Surface((SCREEN_W, SCREEN_H))
        for y in range(SCREEN_H):
            t = y / SCREEN_H
            r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t)
            g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t)
            b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t)
            pygame.draw.line(self.bg_surface, (r, g, b), (0, y), (SCREEN_W, y))

        self.cam_surface = None
        self.camera_warning = ""
        self.finger_pos = None
        self.hovered_card = -1
        self.hover_start = 0
        self.hover_progress = 0
        self.selection_mode = False
        self.selected_games = set()
        self.last_pinch = False
        self.animation_time = 0

        # ── Improvement 5: cache scores so we don't read the file every frame ─
        # Scores are refreshed each time we return from a game via _reset_after_game
        self._score_cache = {}
        self._refresh_score_cache()

    # ── Improvement 5: score cache helpers ───────────────────────────────────
    def _refresh_score_cache(self):
        """Re-read scores.json into memory. Called on startup and after each game."""
        self._score_cache = {game["name"]: _get_high_score(game["name"]) for game in GAMES}

    def _cached_score(self, game_name: str) -> int:
        return self._score_cache.get(game_name, 0)

    # -------------------------------------------------------------------------

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🎮 Hand Tracking Games — Launcher")
        self.clock = pygame.time.Clock()

    def _init_fonts(self):
        self.font_title = pygame.font.SysFont("Segoe UI", 42, bold=True)
        self.font_subtitle = pygame.font.SysFont("Segoe UI", 18)
        self.font_card_name = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.font_card_desc = pygame.font.SysFont("Segoe UI", 12)
        self.font_emoji = pygame.font.SysFont("Segoe UI Emoji", 30)
        self.font_small = pygame.font.SysFont("Segoe UI", 14)

    def _init_camera_tracker(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)
        self.camera_warning = ""
        if not self.cap.isOpened():
            self.camera_warning = "Camera unavailable - use mouse/keyboard to launch"

    def _reset_after_game(self):
        self._init_pygame()
        self._init_fonts()
        self._init_camera_tracker()
        self.hovered_card = -1
        self.hover_start = 0
        self.hover_progress = 0
        self.last_pinch = False
        # ── Improvement 5: refresh scores now that a game may have updated them
        self._refresh_score_cache()

    def _release_resources(self):
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        if hasattr(self, "tracker") and self.tracker:
            self.tracker.release()

    def get_card_rect(self, index):
        """Get the rectangle for a game card."""
        row = index // CARDS_PER_ROW
        col = index % CARDS_PER_ROW
        x = GRID_START_X + col * (CARD_W + CARD_MARGIN)
        y = GRID_START_Y + row * (CARD_H + CARD_MARGIN)
        return pygame.Rect(x, y, CARD_W, CARD_H)

    def get_hovered_card(self, pos):
        """Get index of card under the given position, or -1."""
        if pos is None:
            return -1
        for i in range(len(GAMES)):
            rect = self.get_card_rect(i)
            if rect.collidepoint(pos):
                return i
        return -1

    def launch_game(self, index):
        """Launch a game as a subprocess."""
        game = GAMES[index]
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            game["file"],
        )
        self._release_resources()
        pygame.quit()
        print(f"\nLaunching {game['name']}...")
        try:
            subprocess.run([sys.executable, script_path], check=False)
        finally:
            self._reset_after_game()

    def play_selected_games(self):
        """Play selected games sequentially."""
        if not self.selected_games:
            return
        queue = sorted(self.selected_games)
        for game_idx in queue:
            if 0 <= game_idx < len(GAMES):
                self.launch_game(game_idx)

    def toggle_game_selection(self, index):
        if index in self.selected_games:
            self.selected_games.remove(index)
        else:
            self.selected_games.add(index)

    def get_menu_buttons(self):
        y = 145
        return {
            "mode": pygame.Rect(40, y, 170, 34),
            "play_selected": pygame.Rect(220, y, 190, 34),
            "clear": pygame.Rect(420, y, 120, 34),
        }

    def handle_pointer_activate(self, pos):
        if pos is None:
            return

        buttons = self.get_menu_buttons()
        for action, rect in buttons.items():
            if rect.collidepoint(pos):
                if action == "mode":
                    self.selection_mode = not self.selection_mode
                    self.hovered_card = -1
                    self.hover_progress = 0
                elif action == "play_selected":
                    self.play_selected_games()
                elif action == "clear":
                    self.selected_games.clear()
                return

        card_idx = self.get_hovered_card(pos)
        if card_idx < 0:
            return

        if self.selection_mode:
            self.toggle_game_selection(card_idx)
        else:
            self.launch_game(card_idx)

    def quick_key_action(self, idx):
        if not (0 <= idx < len(GAMES)):
            return
        if self.selection_mode:
            self.toggle_game_selection(idx)
        else:
            self.launch_game(idx)

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (140, 105))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw_card(self, index, hovered):
        """Draw a game card."""
        game = GAMES[index]
        rect = self.get_card_rect(index)
        color = game["color"]

        selected = index in self.selected_games

        # Card background
        bg = CARD_HOVER if hovered else CARD_BG
        if selected:
            bg = (70, 55, 100)
        border = CARD_HOVER_BORDER if hovered else CARD_BORDER
        if selected:
            border = (200, 180, 255)

        # Slight floating animation
        y_offset = 0
        if hovered:
            y_offset = -3

        draw_rect = rect.copy()
        draw_rect.y += y_offset

        # Shadow
        shadow_rect = draw_rect.copy()
        shadow_rect.x += 4
        shadow_rect.y += 4
        shadow_surf = pygame.Surface((CARD_W, CARD_H), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (0, 0, 0, 40), (0, 0, CARD_W, CARD_H), border_radius=12)
        self.screen.blit(shadow_surf, shadow_rect.topleft)

        # Card body
        card_surf = pygame.Surface((CARD_W, CARD_H), pygame.SRCALPHA)
        pygame.draw.rect(card_surf, (*bg, 230), (0, 0, CARD_W, CARD_H), border_radius=12)
        pygame.draw.rect(card_surf, border, (0, 0, CARD_W, CARD_H), 2, border_radius=12)

        # Color accent line at top
        pygame.draw.rect(card_surf, color, (0, 0, CARD_W, 4), border_radius=12)

        self.screen.blit(card_surf, draw_rect.topleft)

        # Emoji
        emoji_text = self.font_emoji.render(game["emoji"], True, WHITE)
        self.screen.blit(emoji_text, (draw_rect.x + 14, draw_rect.y + 12))

        # Name
        name_text = self.font_card_name.render(game["name"], True, color)
        self.screen.blit(name_text, (draw_rect.x + 14, draw_rect.y + 58))

        # Description
        desc_text = self.font_card_desc.render(game["desc"], True, SUBTITLE)
        self.screen.blit(desc_text, (draw_rect.x + 14, draw_rect.y + 74))

        # ── Improvement 5: show high score beneath description ────────────────
        hs = self._cached_score(game["name"])
        if hs > 0:
            hs_color = (160, 220, 150)   # soft green so it doesn't clash with the accent color
            hs_text = self.font_card_desc.render(f"Best: {hs}", True, hs_color)
            self.screen.blit(hs_text, (draw_rect.x + 14, draw_rect.y + 88))

        # Hover progress ring
        if hovered and self.hover_progress > 0 and not self.selection_mode:
            center = (draw_rect.right - 25, draw_rect.top + 25)
            pygame.draw.circle(self.screen, (60, 60, 80), center, 14)
            # Progress arc
            angle = self.hover_progress * 360
            if angle > 0:
                start_angle = -90
                for a in range(int(angle)):
                    rad = math.radians(start_angle + a)
                    px = center[0] + int(12 * math.cos(rad))
                    py = center[1] + int(12 * math.sin(rad))
                    pygame.draw.circle(self.screen, color, (px, py), 2)

        # Number badge
        num_text = self.font_small.render(str(index + 1), True, (120, 110, 150))
        self.screen.blit(num_text, (draw_rect.x + CARD_W - 25, draw_rect.y + CARD_H - 25))

        if self.selection_mode:
            tag = "✓" if selected else "+"
            tag_color = (180, 255, 190) if selected else (170, 170, 210)
            tag_text = self.font_small.render(tag, True, tag_color)
            self.screen.blit(tag_text, (draw_rect.x + 8, draw_rect.y + CARD_H - 24))

    def draw_menu_section(self):
        panel = pygame.Rect(20, 138, SCREEN_W - 40, 48)
        panel_surf = pygame.Surface((panel.w, panel.h), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, (30, 25, 50, 210), (0, 0, panel.w, panel.h), border_radius=10)
        pygame.draw.rect(panel_surf, (95, 85, 130), (0, 0, panel.w, panel.h), 1, border_radius=10)
        self.screen.blit(panel_surf, panel.topleft)

        section_title = self.font_small.render("GAME SELECTION MENU", True, (215, 205, 255))
        self.screen.blit(section_title, (28, 122))

        buttons = self.get_menu_buttons()
        mode_text = "Mode: Selection" if self.selection_mode else "Mode: Quick Launch"
        mode_color = (140, 230, 180) if self.selection_mode else (180, 180, 220)

        pygame.draw.rect(self.screen, (55, 50, 85), buttons["mode"], border_radius=8)
        pygame.draw.rect(self.screen, mode_color, buttons["mode"], 2, border_radius=8)
        mode_lbl = self.font_small.render(mode_text, True, mode_color)
        self.screen.blit(mode_lbl, (buttons["mode"].x + 10, buttons["mode"].y + 8))

        pygame.draw.rect(self.screen, (55, 50, 85), buttons["play_selected"], border_radius=8)
        pygame.draw.rect(self.screen, (150, 220, 255), buttons["play_selected"], 2, border_radius=8)
        play_lbl = self.font_small.render("Play Selected Queue", True, (150, 220, 255))
        self.screen.blit(play_lbl, (buttons["play_selected"].x + 16, buttons["play_selected"].y + 8))

        pygame.draw.rect(self.screen, (55, 50, 85), buttons["clear"], border_radius=8)
        pygame.draw.rect(self.screen, (255, 190, 160), buttons["clear"], 2, border_radius=8)
        clear_lbl = self.font_small.render("Clear", True, (255, 190, 160))
        self.screen.blit(clear_lbl, (buttons["clear"].x + 38, buttons["clear"].y + 8))

        selected_lbl = self.font_small.render(
            f"Selected: {len(self.selected_games)}", True, (200, 200, 230)
        )
        self.screen.blit(selected_lbl, (560, 154))

        selected_names = [GAMES[idx]["name"] for idx in sorted(self.selected_games) if 0 <= idx < len(GAMES)]
        queue_text = ", ".join(selected_names) if selected_names else "No games selected"
        if len(queue_text) > 46:
            queue_text = queue_text[:43] + "..."
        queue_lbl = self.font_small.render(f"Queue: {queue_text}", True, (188, 188, 216))
        self.screen.blit(queue_lbl, (640, 154))

    def draw(self):
        """Draw the launcher."""
        self.screen.blit(self.bg_surface, (0, 0))
        self.animation_time += 1 / FPS

        # Title with subtle pulse
        pulse = 1.0 + 0.02 * math.sin(self.animation_time * 2)
        title = self.font_title.render("🎮  Hand Tracking Games", True, TITLE_COLOR)
        title_rect = title.get_rect(center=(SCREEN_W // 2, 60))
        self.screen.blit(title, title_rect)

        # Subtitle
        subtitle = self.font_subtitle.render(
            "Selection Menu: click 'Mode' then pick cards  •  P = Play queue  •  1-9/0/-/= keys",
            True, SUBTITLE,
        )
        self.screen.blit(subtitle, (SCREEN_W // 2 - subtitle.get_width() // 2, 100))

        # Decorative line
        line_w = 300
        pygame.draw.line(
            self.screen, GLOW,
            (SCREEN_W // 2 - line_w // 2, 135),
            (SCREEN_W // 2 + line_w // 2, 135),
            2,
        )

        self.draw_menu_section()

        # Game cards
        for i in range(len(GAMES)):
            self.draw_card(i, i == self.hovered_card)

        # Finger cursor
        if self.finger_pos:
            fx, fy = self.finger_pos
            # Outer ring
            pygame.draw.circle(self.screen, GLOW, (fx, fy), CURSOR_RADIUS, 2)
            # Inner dot
            pygame.draw.circle(self.screen, WHITE, (fx, fy), 4)
            # Glow effect
            glow_surf = pygame.Surface((CURSOR_RADIUS * 4, CURSOR_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surf, (*GLOW, 30),
                (CURSOR_RADIUS * 2, CURSOR_RADIUS * 2),
                CURSOR_RADIUS * 2,
            )
            self.screen.blit(glow_surf, (fx - CURSOR_RADIUS * 2, fy - CURSOR_RADIUS * 2))

        # Camera overlay
        if self.cam_surface:
            cam_x = SCREEN_W - 150
            cam_y = SCREEN_H - 115
            self.screen.blit(self.cam_surface, (cam_x, cam_y))
            pygame.draw.rect(self.screen, CARD_BORDER, (cam_x, cam_y, 140, 105), 2)

        # Bottom hint
        hint_text = "ESC = Quit  •  Built with MediaPipe + TensorFlow + Pygame"
        if self.camera_warning:
            hint_text = self.camera_warning
        hint = self.font_small.render(hint_text, True, (80, 75, 100))
        self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 25))

        pygame.display.flip()

    def run(self):
        """Main launcher loop."""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_m:
                        self.selection_mode = not self.selection_mode
                        self.hovered_card = -1
                        self.hover_progress = 0
                    elif event.key == pygame.K_p:
                        self.play_selected_games()
                    elif event.key == pygame.K_c and self.selection_mode:
                        self.selected_games.clear()
                    # Number keys to quick launch (1-9, then 0 for 10, - for 11, = for 12)
                    elif event.key in [
                        pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,
                        pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0,
                        pygame.K_MINUS, pygame.K_EQUALS,
                    ]:
                        if event.key == pygame.K_0:
                            idx = 9
                        elif event.key == pygame.K_MINUS:
                            idx = 10
                        elif event.key == pygame.K_EQUALS:
                            idx = 11
                        else:
                            idx = event.key - pygame.K_1
                        self.quick_key_action(idx)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_pointer_activate(event.pos)

            # Camera & tracking
            ret, frame = (False, None)
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                self.tracker.process_frame(frame)
                self.tracker.draw_landmarks(frame)
                self.draw_cam_overlay(frame)

                pos = self.tracker.get_fingertip_pos("index", SCREEN_W, SCREEN_H)
                self.finger_pos = pos

                # Hover detection
                new_hovered = self.get_hovered_card(pos)
                if new_hovered != self.hovered_card:
                    self.hovered_card = new_hovered
                    self.hover_start = time.time()
                    self.hover_progress = 0
                elif new_hovered >= 0 and not self.selection_mode:
                    elapsed = time.time() - self.hover_start
                    self.hover_progress = min(1.0, elapsed / HOVER_TIME_TO_SELECT)
                    if self.hover_progress >= 1.0:
                        self.launch_game(new_hovered)
                        self.hovered_card = -1
                        self.hover_progress = 0

                pinch_dist = self.tracker.get_pinch_distance(SCREEN_W, SCREEN_H)
                pinching = pinch_dist is not None and pinch_dist < 40
                if pinching and not self.last_pinch:
                    self.handle_pointer_activate(pos)
                self.last_pinch = pinching
            else:
                self.finger_pos = None
                self.last_pinch = False

            self.draw()
            self.clock.tick(FPS)

        self._release_resources()
        pygame.quit()


def main():
    launcher = Launcher()
    launcher.run()


if __name__ == "__main__":
    main()