"""Shared constants for the elephant rumble (infrasound) detector.

Elephant rumbles have a fundamental of roughly 8-35 Hz, harmonics that
usually fade out below ~200 Hz, and last about 1-6 s. Audio is resampled to
1 kHz and the model looks at 8-s windows so a whole rumble fits in one window.
"""

SAMPLE_RATE = 1000          # Hz, model input rate (recordings are resampled to this)
CLIP_SECONDS = 8.0          # length of one model window (fits a 6-s rumble + context)
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

FRAME_HOP = 50              # samples per output frame (50 ms -> 20 frames/s)
N_FRAMES = CLIP_SAMPLES // FRAME_HOP

RUMBLE_SECONDS = (1.0, 6.0)  # duration range of synthetic rumbles

# Band used to measure signal-to-noise ratio of rumbles (Hz)
SNR_BAND = (5.0, 150.0)
