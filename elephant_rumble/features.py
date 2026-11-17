"""Audio -> low-frequency log spectrogram features."""

from math import gcd

import numpy as np
from scipy import signal

from .config import CLIP_SAMPLES, F_MAX, HOP, N_FFT, SAMPLE_RATE

_FREQS = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
_KEEP = _FREQS <= F_MAX
N_FREQ_BINS = int(_KEEP.sum())
N_FRAMES = (CLIP_SAMPLES - N_FFT) // HOP + 1
_WINDOW = signal.get_window("hann", N_FFT)


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


def spectrogram(clip: np.ndarray) -> np.ndarray:
    """Return a normalised (N_FREQ_BINS, N_FRAMES) float32 log-power spectrogram.

    Each frequency row has its median subtracted, which whitens stationary
    background noise (wind floor, constant hum) so the model focuses on
    tonal, time-varying structure such as rumble harmonics.
    """
    clip = fit_length(np.asarray(clip, dtype=np.float64))
    clip = clip - clip.mean()
    frames = np.lib.stride_tricks.sliding_window_view(clip, N_FFT)[::HOP]
    spec = np.fft.rfft(frames * _WINDOW, axis=1)[:, _KEEP]
    power = (spec.real ** 2 + spec.imag ** 2).T  # (freq, time)
    logp = 10.0 * np.log10(power + 1e-12)
    logp -= np.median(logp, axis=1, keepdims=True)
    logp /= 10.0  # ~unit scale (10 dB -> 1.0)
    return np.clip(logp, -3.0, 6.0).astype(np.float32)
