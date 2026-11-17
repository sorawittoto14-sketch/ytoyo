"""Audio helpers: resampling and fixed-length windows."""

from math import gcd

import numpy as np
from scipy import signal

from .config import CLIP_SAMPLES, FRAME_HOP, N_FRAMES, SAMPLE_RATE


def resample(audio: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Polyphase resample (includes anti-alias filtering)."""
    audio = np.asarray(audio, dtype=np.float64)
    if sr == target_sr:
        return audio
    g = gcd(int(sr), int(target_sr))
    return signal.resample_poly(audio, target_sr // g, int(sr) // g)


def fit_length(audio: np.ndarray, n: int = CLIP_SAMPLES) -> np.ndarray:
    """Crop or zero-pad to exactly ``n`` samples."""
    if len(audio) >= n:
        return audio[:n]
    return np.pad(audio, (0, n - len(audio)))


def frame_labels(segments, n_frames: int = N_FRAMES) -> np.ndarray:
    """0/1 per output frame: 1 where the frame centre lies inside any (start, end) sample segment."""
    centres = np.arange(n_frames) * FRAME_HOP + FRAME_HOP / 2
    lab = np.zeros(n_frames, dtype=np.int8)
    for lo, hi in segments:
        lab[(centres >= lo) & (centres < hi)] = 1
    return lab
