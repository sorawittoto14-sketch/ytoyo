"""Synthetic elephant rumbles and field-like background noise.

This is used to bootstrap / demo the pipeline when no labelled field data is
available. Real rumbles (e.g. Cornell Elephant Listening Project recordings)
should replace it for any real-world use.

Rumble model: harmonic stack with a fundamental of ~10-30 Hz, a curved
(rise-then-fall) pitch contour, 1-6 s duration, soft attack/decay and
random harmonic amplitudes (sometimes with a weak fundamental).

Background: coloured (1/f^a) noise plus optional distractors that also live
in the infrasound band and make the task non-trivial:
  * engine / generator hum - stationary harmonic stack lasting the whole clip
  * aircraft - harmonic stack with a slow Doppler sweep across the clip
  * thunder - impulsive low-frequency broadband burst
  * wind gusts - amplitude-modulated noise below ~20 Hz
  * insects / birds - short tones above the infrasound band (70-120 Hz)
"""

import numpy as np
from scipy import signal

from .config import CLIP_SAMPLES, MIN_RUMBLE_IN_WINDOW, RUMBLE_SECONDS, SAMPLE_RATE, SNR_BAND

SR = SAMPLE_RATE


def _band_power(x: np.ndarray, band=SNR_BAND) -> float:
    sos = signal.butter(4, band, btype="bandpass", fs=SR, output="sos")
    y = signal.sosfiltfilt(sos, x)
    return float(np.mean(y ** 2) + 1e-20)


def colored_noise(rng, n: int) -> np.ndarray:
    """1/f^alpha noise, alpha in [0.5, 2] (pinkish to brown)."""
    alpha = rng.uniform(0.5, 2.0)
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1]
    spec *= f ** (-alpha / 2.0)
    x = np.fft.irfft(spec, n)
    return x / (x.std() + 1e-12)


def _harmonic_stack(rng, f0: np.ndarray, n_harm: int, decay: float) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros_like(f0)
    for h in range(1, n_harm + 1):
        if np.max(f0) * h >= SR / 2 - 20:
            break
        amp = decay ** (h - 1) * rng.uniform(0.4, 1.0)
        out += amp * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
    return out


def rumble(rng, duration: float | None = None) -> np.ndarray:
    """One synthetic rumble (unit RMS)."""
    dur = duration if duration is not None else rng.uniform(*RUMBLE_SECONDS)
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    f_start = rng.uniform(10, 24)
    rise = rng.uniform(0, 9)
    fall = rng.uniform(-3, 4)
    peak_at = rng.uniform(0.2, 0.6)
    contour = np.where(
        t < peak_at,
        f_start + rise * np.sin(0.5 * np.pi * t / peak_at),
        f_start + rise - (rise + fall) * np.clip((t - peak_at) / (1 - peak_at), 0, None) ** 1.5,
    )
    contour = np.maximum(contour, 7.0)
    contour = contour + rng.normal(0, 0.15, n).cumsum() / np.sqrt(n) * 3  # pitch jitter
    x = _harmonic_stack(rng, contour, n_harm=rng.integers(3, 10), decay=rng.uniform(0.5, 0.9))
    if rng.random() < 0.3:  # weak fundamental
        sos = signal.butter(2, contour.mean() * 1.4, btype="highpass", fs=SR, output="sos")
        x = signal.sosfiltfilt(sos, x)
    env = signal.windows.tukey(n, alpha=rng.uniform(0.3, 0.8))
    x *= env * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.3, 2) * t * dur))
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def _engine(rng, n):
    f0 = rng.uniform(12, 60) + 0.3 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * np.arange(n) / SR)
    return _harmonic_stack(rng, f0, rng.integers(3, 12), rng.uniform(0.6, 0.95))


def _aircraft(rng, n):
    f_a, f_b = rng.uniform(20, 90), rng.uniform(15, 80)
    f0 = np.linspace(f_a, f_b, n)
    t = np.arange(n) / n
    env = 0.3 + np.exp(-0.5 * ((t - rng.uniform(0, 1)) / rng.uniform(0.2, 0.6)) ** 2)
    return _harmonic_stack(rng, f0, rng.integers(4, 14), rng.uniform(0.7, 0.95)) * env


def _thunder(rng, n):
    x = np.zeros(n)
    start = rng.integers(0, n - SR // 2)
    length = min(n - start, int(rng.uniform(0.5, 3.0) * SR))
    burst = rng.normal(size=length) * np.exp(-np.linspace(0, rng.uniform(2, 6), length))
    sos = signal.butter(2, rng.uniform(30, 100), btype="lowpass", fs=SR, output="sos")
    x[start:start + length] = signal.sosfilt(sos, burst)
    return x


def _wind(rng, n):
    sos = signal.butter(2, rng.uniform(8, 25), btype="lowpass", fs=SR, output="sos")
    gust = np.abs(signal.sosfilt(signal.butter(1, 0.5, fs=SR, output="sos"), rng.normal(size=n)))
    return signal.sosfilt(sos, rng.normal(size=n)) * (gust / (gust.max() + 1e-12))


def _insects(rng, n):
    x = np.zeros(n)
    for _ in range(rng.integers(1, 6)):
        length = int(rng.uniform(0.1, 1.0) * SR)
        start = rng.integers(0, n - length)
        f = rng.uniform(70, 120)
        tt = np.arange(length) / SR
        x[start:start + length] += np.sin(2 * np.pi * f * tt) * np.hanning(length)
    return x


DISTRACTORS = {
    "engine": (_engine, 0.25),
    "aircraft": (_aircraft, 0.15),
    "thunder": (_thunder, 0.10),
    "wind": (_wind, 0.30),
    "insects": (_insects, 0.30),
}


def background(rng, n: int = CLIP_SAMPLES) -> tuple[np.ndarray, list[str]]:
    """Background noise (unit band power in SNR_BAND) and names of distractors added."""
    x = colored_noise(rng, n)
    used = []
    base = _band_power(x)
    for name, (fn, prob) in DISTRACTORS.items():
        if rng.random() < prob:
            d = fn(rng, n)
            d_db = rng.uniform(-6, 8)  # distractor level relative to base noise
            x = x + d * np.sqrt(base * 10 ** (d_db / 10) / _band_power(d))
            used.append(name)
    return x / np.sqrt(_band_power(x)), used


def make_clip(rng, positive: bool, snr_db_range=(-6.0, 12.0)) -> tuple[np.ndarray, dict]:
    """One CLIP_SAMPLES-long clip at SAMPLE_RATE plus metadata."""
    x, used = background(rng)
    meta = {"label": int(positive), "distractors": used, "snr_db": None, "segments": []}
    if positive:
        r = rumble(rng)
        # at least MIN_RUMBLE_IN_WINDOW s (or the whole rumble) must lie inside the window
        min_in = min(len(r), int(MIN_RUMBLE_IN_WINDOW * SR))
        start = rng.integers(-(len(r) - min_in), CLIP_SAMPLES - min_in + 1)
        snr = rng.uniform(*snr_db_range)
        r = r * np.sqrt(10 ** (snr / 10) / _band_power(r))  # background band power == 1
        lo, hi = max(start, 0), min(start + len(r), CLIP_SAMPLES)
        x[lo:hi] += r[lo - start: hi - start]
        meta["snr_db"] = float(snr)
        meta["segments"] = [(int(lo), int(hi))]  # rumble extent in samples
    x *= rng.uniform(0.01, 1.0)  # random gain
    return x.astype(np.float32), meta


def make_long_recording(seed: int, minutes: float = 10.0, n_rumbles: int = 12,
                        snr_db_range=(0.0, 12.0)):
    """A long recording with known rumble positions, for testing the detector.

    Returns (audio @ SAMPLE_RATE, list of (begin_s, end_s)).
    """
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * SR)
    chunks = [background(rng, CLIP_SAMPLES * 6)[0] for _ in range(int(np.ceil(n / (CLIP_SAMPLES * 6))))]
    x = np.concatenate(chunks)[:n]
    events = []
    slots = np.sort(rng.choice(np.arange(5, int(minutes * 60) - 15, 15), n_rumbles, replace=False))
    for s in slots:
        r = rumble(rng)
        start = int((s + rng.uniform(0, 4)) * SR)
        snr = rng.uniform(*snr_db_range)
        r = r * np.sqrt(10 ** (snr / 10) / _band_power(r))
        x[start:start + len(r)] += r
        events.append((start / SR, (start + len(r)) / SR))
    return (x * 0.1).astype(np.float32), events
