"""
Drawing Canvas — Hand Tracking Edition
Draw with your index fingertip. Pinch thumb + index to stop drawing.
Select colors and brush sizes from the toolbar.
"""

import sys
import os
import math
import time
from datetime import datetime

import cv2
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_tracker import HandTracker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 900, 650
FPS = 60
CAM_W, CAM_H = 640, 480

# Canvas
CANVAS_X, CANVAS_Y = 100, 60
CANVAS_W, CANVAS_H = SCREEN_W - 120, SCREEN_H - 80

# Colors
BG_COLOR = (30, 30, 40)
CANVAS_BG = (255, 255, 255)
TOOLBAR_BG = (40, 40, 55)
WHITE = (255, 255, 255)
DARK_TEXT = (200, 200, 210)
HIGHLIGHT = (100, 180, 255)
SELECTED_RING = (255, 220, 50)

# Palette
COLOR_PALETTE = [
    (0, 0, 0),           # Black
    (255, 255, 255),     # White
    (255, 60, 60),       # Red
    (60, 180, 255),      # Blue
    (60, 220, 100),      # Green
    (255, 200, 50),      # Yellow
    (255, 130, 50),      # Orange
    (200, 80, 255),      # Purple
    (255, 100, 180),     # Pink
    (80, 220, 220),      # Cyan
    (120, 80, 40),       # Brown
    (150, 150, 150),     # Gray
]

BRUSH_SIZES = [2, 4, 8, 14, 22]
ERASER_SIZE = 30
PINCH_THRESHOLD = 45  # pixels - below this = pinch detected


class DrawingCanvasGame:
    """Main Drawing Canvas class."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("🎨 Drawing Canvas — Hand Tracking")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_medium = pygame.font.SysFont("Segoe UI", 24, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 18)
        self.font_tiny = pygame.font.SysFont("Segoe UI", 14)

        # Camera & tracker
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ready = self.cap.isOpened()
        self.tracker = HandTracker(max_hands=1, detection_confidence=0.7)

        self.cam_surface = None

        # Canvas
        self.canvas = pygame.Surface((CANVAS_W, CANVAS_H))
        self.canvas.fill(CANVAS_BG)

        # Drawing state
        self.current_color = COLOR_PALETTE[0]  # Black
        self.current_size_idx = 2  # Medium
        self.eraser_mode = False
        self.drawing = False
        self.prev_draw_pos = None
        self.finger_pos = None
        self.is_pinching = False
        self.mouse_drawing = False
        self.mouse_pos = None

        # Undo stack
        self.undo_stack = []
        self.toolbar_regions = self.get_toolbar_regions()
        self.save_snapshot()

    def save_snapshot(self):
        """Save current canvas state for undo."""
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.canvas.copy())

    def undo(self):
        """Undo last action."""
        if len(self.undo_stack) > 1:
            self.undo_stack.pop()
            self.canvas = self.undo_stack[-1].copy()

    def clear_canvas(self):
        """Clear the canvas."""
        self.save_snapshot()
        self.canvas.fill(CANVAS_BG)

    def save_drawing(self):
        """Save drawing to file."""
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "saved_drawings",
        )
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"drawing_{timestamp}.png")
        pygame.image.save(self.canvas, filename)
        print(f"Drawing saved to {filename}")
        return filename

    def get_toolbar_regions(self):
        """Define clickable regions in the toolbar."""
        regions = []
        y = 70

        # Colors
        for i, color in enumerate(COLOR_PALETTE):
            row = i // 2
            col = i % 2
            cx = 15 + col * 35
            cy = y + row * 35
            regions.append(("color", i, pygame.Rect(cx, cy, 28, 28)))
        y += (len(COLOR_PALETTE) // 2 + 1) * 35

        # Brush sizes
        y += 10
        for i, size in enumerate(BRUSH_SIZES):
            cy = y + i * 35
            regions.append(("size", i, pygame.Rect(15, cy, 65, 28)))
        y += len(BRUSH_SIZES) * 35

        # Eraser
        y += 10
        regions.append(("eraser", 0, pygame.Rect(15, y, 65, 30)))
        y += 40

        # Clear
        regions.append(("clear", 0, pygame.Rect(15, y, 65, 30)))
        y += 40

        # Undo
        regions.append(("undo", 0, pygame.Rect(15, y, 65, 30)))
        y += 40

        # Save
        regions.append(("save", 0, pygame.Rect(15, y, 65, 30)))

        return regions

    def handle_toolbar_click(self, pos):
        """Handle clicks/finger touches on toolbar."""
        for action, idx, rect in self.toolbar_regions:
            if rect.collidepoint(pos):
                if action == "color":
                    self.current_color = COLOR_PALETTE[idx]
                    self.eraser_mode = False
                elif action == "size":
                    self.current_size_idx = idx
                elif action == "eraser":
                    self.eraser_mode = not self.eraser_mode
                elif action == "clear":
                    self.clear_canvas()
                elif action == "undo":
                    self.undo()
                elif action == "save":
                    self.save_drawing()
                return True
        return False

    def draw_on_canvas(self, pos):
        """Draw on the canvas at given screen position."""
        # Convert screen pos to canvas pos
        cx = pos[0] - CANVAS_X
        cy = pos[1] - CANVAS_Y

        if 0 <= cx < CANVAS_W and 0 <= cy < CANVAS_H:
            brush_size = BRUSH_SIZES[self.current_size_idx]
            color = CANVAS_BG if self.eraser_mode else self.current_color
            size = ERASER_SIZE if self.eraser_mode else brush_size

            if self.prev_draw_pos:
                prev_cx = self.prev_draw_pos[0] - CANVAS_X
                prev_cy = self.prev_draw_pos[1] - CANVAS_Y
                # Draw line between previous and current position for smooth strokes
                pygame.draw.line(self.canvas, color, (prev_cx, prev_cy), (cx, cy), size)
                # Round caps
                pygame.draw.circle(self.canvas, color, (cx, cy), size // 2)
            else:
                pygame.draw.circle(self.canvas, color, (cx, cy), size // 2)

            self.prev_draw_pos = pos
        else:
            self.prev_draw_pos = None

    def is_in_canvas(self, pos):
        x, y = pos
        return CANVAS_X <= x < CANVAS_X + CANVAS_W and CANVAS_Y <= y < CANVAS_Y + CANVAS_H

    def update(self):
        """Update drawing state based on hand tracking."""
        if self.mouse_drawing:
            return

        if not self.tracker.hand_detected:
            self.drawing = False
            self.prev_draw_pos = None
            return

        # Get finger position
        pos = self.tracker.get_fingertip_pos("index", SCREEN_W, SCREEN_H)
        self.finger_pos = pos

        # Check pinch
        pinch_dist = self.tracker.get_pinch_distance(SCREEN_W, SCREEN_H)
        self.is_pinching = pinch_dist is not None and pinch_dist < PINCH_THRESHOLD

        if pos is None:
            self.drawing = False
            self.prev_draw_pos = None
            return

        # If pinching, don't draw (lift pen)
        if self.is_pinching:
            if self.drawing:
                self.save_snapshot()
            self.drawing = False
            self.prev_draw_pos = None
            return

        # Check if finger is on toolbar
        if pos[0] < CANVAS_X:
            self.handle_toolbar_click(pos)
            self.drawing = False
            self.prev_draw_pos = None
            return

        # Draw on canvas
        if not self.drawing:
            self.drawing = True

        self.draw_on_canvas(pos)

    def draw_cam_overlay(self, frame):
        frame_small = cv2.resize(frame, (120, 90))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_small = np.rot90(frame_small)
        self.cam_surface = pygame.surfarray.make_surface(frame_small)

    def draw_toolbar(self):
        """Draw the left toolbar."""
        # Toolbar background
        pygame.draw.rect(self.screen, TOOLBAR_BG, (0, 0, CANVAS_X - 5, SCREEN_H))

        # Title
        title = self.font_medium.render("Tools", True, WHITE)
        self.screen.blit(title, (20, 10))

        y = 40
        # Label
        label = self.font_tiny.render("Colors:", True, DARK_TEXT)
        self.screen.blit(label, (15, y))
        y += 25

        # Color palette
        for i, color in enumerate(COLOR_PALETTE):
            row = i // 2
            col = i % 2
            cx = 15 + col * 35
            cy = y + row * 35
            rect = pygame.Rect(cx, cy, 28, 28)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if color == self.current_color and not self.eraser_mode:
                pygame.draw.rect(self.screen, SELECTED_RING, rect.inflate(6, 6), 3, border_radius=6)
            pygame.draw.rect(self.screen, (80, 80, 100), rect, 1, border_radius=4)

        y += (len(COLOR_PALETTE) // 2 + 1) * 35

        # Brush sizes label
        label = self.font_tiny.render("Brush:", True, DARK_TEXT)
        self.screen.blit(label, (15, y - 5))
        y += 18

        for i, size in enumerate(BRUSH_SIZES):
            cy = y + i * 35
            rect = pygame.Rect(15, cy, 65, 28)
            is_selected = i == self.current_size_idx and not self.eraser_mode
            bg = (60, 60, 80) if is_selected else TOOLBAR_BG
            pygame.draw.rect(self.screen, bg, rect, border_radius=4)
            if is_selected:
                pygame.draw.rect(self.screen, HIGHLIGHT, rect, 2, border_radius=4)
            # Draw circle to represent size
            pygame.draw.circle(
                self.screen, self.current_color if not self.eraser_mode else WHITE,
                (rect.x + 16, rect.centery), min(size, 10),
            )
            size_label = self.font_tiny.render(f"{size}px", True, DARK_TEXT)
            self.screen.blit(size_label, (rect.x + 32, rect.y + 5))

        y += len(BRUSH_SIZES) * 35 + 10

        # Eraser button
        eraser_rect = pygame.Rect(15, y, 65, 30)
        eraser_bg = (80, 60, 60) if self.eraser_mode else TOOLBAR_BG
        pygame.draw.rect(self.screen, eraser_bg, eraser_rect, border_radius=4)
        if self.eraser_mode:
            pygame.draw.rect(self.screen, (255, 100, 100), eraser_rect, 2, border_radius=4)
        eraser_label = self.font_tiny.render("Eraser", True, WHITE)
        self.screen.blit(eraser_label, (eraser_rect.x + 5, eraser_rect.y + 6))
        y += 40

        # Clear button
        clear_rect = pygame.Rect(15, y, 65, 30)
        pygame.draw.rect(self.screen, (80, 50, 50), clear_rect, border_radius=4)
        clear_label = self.font_tiny.render("Clear", True, WHITE)
        self.screen.blit(clear_label, (clear_rect.x + 12, clear_rect.y + 6))
        y += 40

        # Undo button
        undo_rect = pygame.Rect(15, y, 65, 30)
        pygame.draw.rect(self.screen, (50, 60, 80), undo_rect, border_radius=4)
        undo_label = self.font_tiny.render("Undo", True, WHITE)
        self.screen.blit(undo_label, (undo_rect.x + 13, undo_rect.y + 6))
        y += 40

        # Save button
        save_rect = pygame.Rect(15, y, 65, 30)
        pygame.draw.rect(self.screen, (50, 80, 60), save_rect, border_radius=4)
        save_label = self.font_tiny.render("Save", True, WHITE)
        self.screen.blit(save_label, (save_rect.x + 14, save_rect.y + 6))

    def draw(self):
        """Draw everything."""
        self.screen.fill(BG_COLOR)

        # Toolbar
        self.draw_toolbar()

        # Canvas border
        pygame.draw.rect(
            self.screen, (60, 60, 80),
            (CANVAS_X - 3, CANVAS_Y - 3, CANVAS_W + 6, CANVAS_H + 6),
            border_radius=4,
        )

        # Canvas
        self.screen.blit(self.canvas, (CANVAS_X, CANVAS_Y))

        # Finger cursor on canvas
        if self.finger_pos and not self.is_pinching:
            fx, fy = self.finger_pos
            if fx >= CANVAS_X:
                color = CANVAS_BG if self.eraser_mode else self.current_color
                size = ERASER_SIZE if self.eraser_mode else BRUSH_SIZES[self.current_size_idx]
                pygame.draw.circle(self.screen, color, (fx, fy), size // 2, 2)
                # Cross-hair
                pygame.draw.line(self.screen, (200, 200, 200), (fx - 8, fy), (fx + 8, fy), 1)
                pygame.draw.line(self.screen, (200, 200, 200), (fx, fy - 8), (fx, fy + 8), 1)

        if self.mouse_drawing and self.mouse_pos and self.is_in_canvas(self.mouse_pos):
            mx, my = self.mouse_pos
            color = CANVAS_BG if self.eraser_mode else self.current_color
            size = ERASER_SIZE if self.eraser_mode else BRUSH_SIZES[self.current_size_idx]
            pygame.draw.circle(self.screen, color, (mx, my), size // 2, 2)

        # Drawing mode indicator
        if self.is_pinching:
            mode_text = self.font_small.render("✋ Pen Lifted (Pinch)", True, (255, 200, 100))
        elif self.drawing:
            mode_text = self.font_small.render("✏️ Drawing", True, (100, 255, 150))
        elif not self.camera_ready:
            mode_text = self.font_small.render("Camera unavailable (mouse toolbar still works)", True, (220, 140, 140))
        elif self.tracker.hand_detected:
            mode_text = self.font_small.render("☝️ Move to draw", True, DARK_TEXT)
        else:
            mode_text = self.font_small.render("No hand detected", True, (150, 100, 100))
        self.screen.blit(mode_text, (CANVAS_X, SCREEN_H - 25))

        # Camera overlay
        if self.cam_surface:
            cam_x = SCREEN_W - 130
            cam_y = 10
            self.screen.blit(self.cam_surface, (cam_x, cam_y))
            pygame.draw.rect(self.screen, (60, 60, 80), (cam_x, cam_y, 120, 90), 2)

        # Keyboard shortcuts hint
        hint = self.font_tiny.render("C=Clear  Z=Undo  S=Save  ESC=Quit", True, (100, 100, 120))
        self.screen.blit(hint, (CANVAS_X + CANVAS_W - hint.get_width(), SCREEN_H - 22))

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
                    elif event.key == pygame.K_c:
                        self.clear_canvas()
                    elif event.key == pygame.K_z:
                        self.undo()
                    elif event.key == pygame.K_s:
                        self.save_drawing()
                    elif event.key == pygame.K_e:
                        self.eraser_mode = not self.eraser_mode
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        pos = event.pos
                        if pos[0] < CANVAS_X:
                            self.handle_toolbar_click(pos)
                            self.mouse_drawing = False
                            self.prev_draw_pos = None
                        elif self.is_in_canvas(pos):
                            self.save_snapshot()
                            self.mouse_drawing = True
                            self.mouse_pos = pos
                            self.prev_draw_pos = pos
                            self.draw_on_canvas(pos)
                elif event.type == pygame.MOUSEMOTION:
                    if self.mouse_drawing:
                        self.mouse_pos = event.pos
                        self.draw_on_canvas(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.mouse_drawing:
                        self.mouse_drawing = False
                        self.mouse_pos = None
                        self.prev_draw_pos = None

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
    game = DrawingCanvasGame()
    game.run()


if __name__ == "__main__":
    main()
