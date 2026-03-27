# Hand Tracking Games

A collection of hand-tracking mini-games built with Python, OpenCV, MediaPipe, TensorFlow, and Pygame.

## Features

- Hand-tracking launcher with card-based game selection.
- **Selection Menu** in launcher to queue multiple games and play them sequentially.
- Mouse/keyboard fallback when camera or hand tracking is unavailable.
- Multiple games with level/campaign progression in the newer set.

## Setup

1. Create/activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start launcher:

```bash
python launcher.py
```

## Launcher Controls

- `M`: Toggle **Quick Launch** / **Selection Mode**.
- `P`: Play selected queue.
- `C`: Clear queue (in selection mode).
- `1-9`, `0`, `-`, `=`: Select/launch cards by index.
- Mouse click / pinch: Activate buttons and cards.
- `ESC`: Exit launcher.

### Selection Menu (top control bar)

- **Mode button** switches to Selection Mode.
- In Selection Mode, selecting cards adds/removes games from queue.
- **Play Selected Queue** launches selected games in order.
- **Clear** resets queue.

## Games and Controls

### 1) Fruit Ninja (`games/fruit_ninja.py`)
- Objective: Slice fruit, avoid missing too many.
- Controls:
  - Hand: Move index fingertip to slash.
  - Mouse fallback: move cursor.
  - `R`: restart after game over.
  - `ESC`: quit.

### 2) Flappy Bird (`games/flappy_bird.py`)
- Objective: Survive through pipes.
- Controls:
  - Hand: Open-hand gesture to flap.
  - Keyboard fallback: `SPACE` flap.
  - `R`: restart after game over.
  - `ESC`: quit.

### 3) Pong (`games/pong.py`)
- Objective: Outscore CPU.
- Controls:
  - Hand: Move hand vertically to control paddle.
  - Keyboard fallback: arrow keys.
  - `R`: restart.
  - `ESC`: quit.

### 4) Rock Paper Scissors (`games/rock_paper_scissors.py`)
- Objective: Win best-of-5 rounds.
- Controls:
  - `SPACE`: start round countdown.
  - Hand: show rock/paper/scissors gesture.
  - Uses TensorFlow classifier when available, otherwise falls back to basic hand gesture detection.
  - `R`: restart match after game over.
  - `ESC`: quit.

### 5) Whack-a-Mole (`games/whack_a_mole.py`)
- Objective: Hit appearing moles before timeout.
- Controls:
  - Hand: point/tap with fingertip over moles.
  - Mouse fallback: click moles.
  - `R`: restart after game over.
  - `ESC`: quit.

### 6) Drawing Canvas (`games/drawing_canvas.py`)
- Objective: Free drawing with gesture/mouse controls.
- Controls:
  - Hand: fingertip to draw.
  - Mouse fallback: click/drag.
  - Toolbar buttons for clear/tools.
  - `ESC`: quit.

### 7) Breakout (`games/breakout.py`) — **Level Campaign**
- Objective: Clear brick waves.
- Controls:
  - Hand: move paddle using fingertip x-position.
  - Pinch: temporary paddle boost.
  - Keyboard fallback: `SPACE` launch, `R` restart, `ESC` quit.
- Levels:
  - Multi-level progression with more rows, faster ball, narrower paddle each level.

### 8) Memory Match (`games/memory_match.py`) — **Level Campaign**
- Objective: Match all card pairs.
- Controls:
  - Hand: hover + pinch to flip cards.
  - Mouse fallback: click cards.
  - `R`: restart campaign.
  - `ESC`: quit.
- Levels:
  - Board size increases per level (more pairs).

### 9) 2048 (`games/game_2048.py`) — **Level Campaign**
- Objective: Merge tiles to reach per-level target.
- Controls:
  - Hand: swipe gesture (axis-locked).
  - Keyboard fallback: arrow keys.
  - `R`: restart campaign.
  - `ESC`: quit.
- Levels:
  - Targets progress from 256 → 512 → 1024 → 2048.

### 10) Tic-Tac-Toe (`games/tic_tac_toe.py`) — **Level Campaign**
- Objective: Beat CPU across increasing difficulty levels.
- Controls:
  - Hand: hover cell + pinch to place.
  - Mouse fallback: click cell.
  - `R`: restart campaign.
  - `ESC`: quit.
- Levels:
  - CPU improves each level (lower mistake rate, faster move timing).

### 11) Minesweeper Lite (`games/minesweeper_lite.py`) — **Level Campaign**
- Objective: Reveal all safe cells.
- Controls:
  - Hand: hover + pinch to reveal.
  - Mouse fallback: click cell.
  - `R`: restart campaign.
  - `ESC`: quit.
- Levels:
  - Mine count increases by level.

### 12) Maze Runner (`games/maze_runner.py`) — **Level Campaign**
- Objective: Reach goal before time runs out.
- Controls:
  - Hand: steer by moving fingertip directionally.
  - Keyboard fallback: arrow keys.
  - `R`: restart campaign.
  - `ESC`: quit.
- Levels:
  - Multiple maze sizes/densities.
  - Increasing movement challenge and per-level timer constraints.

## Troubleshooting

- If camera is unavailable, games should still be playable with mouse/keyboard fallbacks.
- If RPS TensorFlow classifier is unavailable, the game now falls back to basic hand gesture rules instead of crashing.
- If dependencies conflict, reinstall from `requirements.txt` inside your project virtual environment.

## Project Structure

- `launcher.py` — main launcher and game queue menu.
- `hand_tracker.py` — shared hand tracking utility.
- `gesture_model.py` — TensorFlow gesture classifier for RPS.
- `games/` — all game modules.
