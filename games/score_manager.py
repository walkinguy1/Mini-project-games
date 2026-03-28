"""Shared score persistence helpers for all games.

This module is intentionally defensive: invalid/missing files are treated as empty
scoreboards, and writes are done atomically to avoid partial-file corruption.
"""

import json
import os
import tempfile
from typing import Dict

SCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scores.json")


def load_scores() -> Dict[str, int]:
    """Load all saved scores.

    Returns an empty mapping if the file is missing, unreadable, or malformed.
    """
    if not os.path.exists(SCORE_FILE):
        return {}

    try:
        with open(SCORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    clean_scores: Dict[str, int] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            clean_scores[key] = int(value)
    return clean_scores


def _write_scores_atomic(scores: Dict[str, int]) -> None:
    """Write scores using a temporary file then atomic replace."""
    score_dir = os.path.dirname(SCORE_FILE)
    os.makedirs(score_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="scores_", suffix=".json", dir=score_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(scores, tmp_file, indent=2)
        os.replace(tmp_path, SCORE_FILE)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def save_score(game_name: str, score: int) -> bool:
    """Save a score if it beats the existing high score.

    Returns True when a new high score is persisted, else False.
    """
    if not game_name:
        return False

    try:
        score_value = int(score)
    except (TypeError, ValueError):
        return False

    scores = load_scores()
    if score_value <= scores.get(game_name, 0):
        return False

    scores[game_name] = score_value
    try:
        _write_scores_atomic(scores)
    except OSError:
        return False
    return True


def get_high_score(game_name: str) -> int:
    return int(load_scores().get(game_name, 0))