"""Shared constants for the elephant rumble (infrasound) detector.

Follows the ELSENSE SincNet-LSTM design:
  raw audio -> sliding window (3 s, 1 s overlap) -> balanced dataset
  -> SincNet restricted to the elephant infrasound band (10-60 Hz) -> LSTM.

Because only 10-60 Hz is used, audio is resampled to 250 Hz (Nyquist 125 Hz).
Elephant rumbles last about 1-6 s, so a rumble can span several windows.
"""

SAMPLE_RATE = 250           # Hz, model input rate (recordings are resampled to this)
CLIP_SECONDS = 3.0          # sliding-window length
HOP_SECONDS = 2.0           # 3-s windows with 1-s overlap
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

BAND_HZ = (10.0, 60.0)      # SincNet band-pass filters are kept inside this band

FRAME_HOP = 50              # samples per LSTM output frame (0.2 s -> 15 frames/window)
N_FRAMES = CLIP_SAMPLES // FRAME_HOP

RUMBLE_SECONDS = (1.0, 6.0)  # duration range of synthetic rumbles
MIN_RUMBLE_IN_WINDOW = 1.0   # a window is "positive" if >= this many seconds of rumble are inside

# Band used to measure signal-to-noise ratio of rumbles (Hz)
SNR_BAND = BAND_HZ

# Long-recording detection defaults. 0.8 gave ~93 % event recall at ~2.5 false
# alarms/hour on three full 24-h ELP test days with models/sincnet_lstm_real_elephant_data.pt;
# re-tune for each deployment site (background noise differs).
DETECT_THRESHOLD = 0.8      # frame probability threshold
DETECT_MIN_DURATION = 1.0   # s; rumbles are >= ~1 s, shorter blips are dropped
