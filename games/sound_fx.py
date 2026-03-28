# games/sound_fx.py
"""
Procedural sound effects using only pygame — no audio files required.
"""
import numpy as np
import pygame

_SAMPLE_RATE = 44100
_initialized = False


def _ensure_init():
    global _initialized
    if not _initialized:
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=_SAMPLE_RATE, size=-16, channels=1, buffer=512)
        _initialized = True


def _make_sound(freq: float, duration: float, shape: str = "sine", volume: float = 0.4) -> pygame.mixer.Sound:
    """Generate a tone as a pygame Sound object."""
    _ensure_init()
    n_samples = int(_SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    if shape == "sine":
        wave = np.sin(2 * np.pi * freq * t)
    elif shape == "square":
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    elif shape == "sawtooth":
        wave = 2 * (t * freq - np.floor(t * freq + 0.5))
    else:
        wave = np.sin(2 * np.pi * freq * t)

    # Fade out last 20% to avoid clicks
    fade_start = int(n_samples * 0.8)
    fade = np.ones(n_samples)
    fade[fade_start:] = np.linspace(1, 0, n_samples - fade_start)
    wave = wave * fade * volume

    samples = (wave * 32767).astype(np.int16)
    sound = pygame.sndarray.make_sound(samples)
    return sound


# Pre-built sound effects
def play_score():
    """Short upbeat tone for scoring a point."""
    _make_sound(880, 0.12, "sine", 0.3).play()

def play_hit():
    """Thud/impact sound."""
    _make_sound(120, 0.15, "square", 0.25).play()

def play_game_over():
    """Descending tone for game over."""
    for i, (freq, dur) in enumerate([(440, 0.1), (330, 0.1), (220, 0.25)]):
        s = _make_sound(freq, dur, "sawtooth", 0.2)
        pygame.time.set_timer(pygame.USEREVENT + i, int(i * 120))  # stagger playback
        s.play()

def play_flap():
    """Quick whoosh for Flappy Bird."""
    _make_sound(600, 0.07, "sine", 0.2).play()

def play_slice():
    """Slice sound for Fruit Ninja."""
    _make_sound(1200, 0.05, "sawtooth", 0.15).play()