import numpy as np
import soundfile as sf
import torch

from elephant_rumble import synth
from elephant_rumble.config import CLIP_SAMPLES, SAMPLE_RATE
from elephant_rumble.data import clips_from_raven, folder_dataset, read_raven_table
from elephant_rumble.detect import score_file, windows_to_events, write_raven
from elephant_rumble.features import N_FRAMES, N_FREQ_BINS, resample, spectrogram
from elephant_rumble.metrics import roc_auc
from elephant_rumble.model import RumbleCNN


def test_spectrogram_shape_and_band():
    assert spectrogram(np.zeros(CLIP_SAMPLES)).shape == (N_FREQ_BINS, N_FRAMES)
    t = np.arange(CLIP_SAMPLES) / SAMPLE_RATE
    spec = spectrogram(np.sin(2 * np.pi * 20 * t) * (t > 2.5) + 1e-3 * np.random.default_rng(0).normal(size=len(t)))
    row = int(np.argmax(spec[:, -10:].mean(1)))
    assert abs(row * SAMPLE_RATE / 512 - 20) < 3  # energy appears at ~20 Hz


def test_resample_keeps_infrasound_tone():
    sr = 4000
    t = np.arange(sr * 2) / sr
    y = resample(np.sin(2 * np.pi * 15 * t), sr)
    assert len(y) == 2 * SAMPLE_RATE
    f = np.fft.rfftfreq(len(y), 1 / SAMPLE_RATE)[np.argmax(np.abs(np.fft.rfft(y)))]
    assert abs(f - 15) < 1


def test_synthetic_rumble_is_infrasonic():
    r = synth.rumble(np.random.default_rng(0), duration=3.0)
    p = np.abs(np.fft.rfft(r)) ** 2
    f = np.fft.rfftfreq(len(r), 1 / SAMPLE_RATE)
    assert p[f < 150].sum() / p.sum() > 0.9


def test_model_forward():
    assert RumbleCNN()(torch.zeros(3, 1, N_FREQ_BINS, N_FRAMES)).shape == (3,)


def test_roc_auc():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_windows_to_events_merges():
    times = np.arange(20.0)
    probs = np.zeros(20)
    probs[[1, 2, 3, 15]] = [.9, .7, .9, .8]  # windows are 5 s long, so 1-3 overlap
    ev = windows_to_events(times, probs, 0.5, min_windows=1)
    assert [(b, e) for b, e, _ in ev] == [(1.0, 8.0), (15.0, 20.0)]
    assert ev[0][2] == .9
    assert len(windows_to_events(times, probs, 0.5, min_windows=2)) == 1


def test_raven_roundtrip_and_clip_export(tmp_path):
    audio, events = synth.make_long_recording(seed=1, minutes=1.0, n_rumbles=3)
    wav = tmp_path / "rec.wav"
    sf.write(wav, resample(audio, SAMPLE_RATE, 2000), 2000)
    table = tmp_path / "rec.txt"
    write_raven([(b, e, 1.0) for b, e in events], table)
    assert read_raven_table(table) == [(round(b, 3), round(e, 3)) for b, e in events]
    n_pos, n_neg = clips_from_raven(wav, table, tmp_path / "clips")
    assert (n_pos, n_neg) == (3, 3)
    ds = folder_dataset(tmp_path / "clips")
    assert len(ds) == 6 and ds.labels.sum() == 3


def test_score_file_streams_blocks(tmp_path):
    wav = tmp_path / "long.wav"
    sf.write(wav, np.random.default_rng(0).normal(size=SAMPLE_RATE * 130).astype(np.float32) * 0.01, SAMPLE_RATE)
    model = RumbleCNN().eval()
    t, p = score_file(model, wav, hop_s=1.0, block_minutes=1.0)
    assert np.allclose(t, np.arange(126))  # no duplicated / missing windows across blocks
