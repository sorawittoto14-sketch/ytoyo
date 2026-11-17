"""Shared constants for the elephant rumble (infrasound) detector.

Elephant rumbles have a fundamental of roughly 8-35 Hz with harmonics that
usually fade out below ~200 Hz, so everything is resampled to 1 kHz and only
the 0-250 Hz band of the spectrogram is kept.
"""

SAMPLE_RATE = 1000          # Hz, model input rate (recordings are resampled to this)
CLIP_SECONDS = 5.0          # length of one classification window
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

N_FFT = 512                 # ~1.95 Hz frequency resolution at 1 kHz
HOP = 50                    # 50 ms hop
F_MAX = 250.0               # highest frequency kept (Hz)

# Band used to measure signal-to-noise ratio of rumbles (Hz)
SNR_BAND = (5.0, 150.0)
