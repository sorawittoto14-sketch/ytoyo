import numpy as np
import soundfile as sf
import torch

from elephant_rumble import synth
from elephant_rumble.config import BAND_HZ, CLIP_SAMPLES, FRAME_HOP, N_FRAMES, SAMPLE_RATE
from elephant_rumble.data import clips_from_raven, folder_dataset, read_raven_table
from elephant_rumble.detect import FRAME_S, frame_timeline, score_file, timeline_to_events, write_raven
from elephant_rumble.features import frame_labels, resample
from elephant_rumble.metrics import roc_auc
from elephant_rumble.model import SincConv, SincLSTM
from elephant_rumble.train import loss_fn


def test_resample_keeps_infrasound_tone():
    sr = 4000
    t = np.arange(sr * 2) / sr
    y = resample(np.sin(2 * np.pi * 15 * t), sr)
    assert len(y) == 2 * SAMPLE_RATE
    f = np.fft.rfftfreq(len(y), 1 / SAMPLE_RATE)[np.argmax(np.abs(np.fft.rfft(y)))]
    assert abs(f - 15) < 1


def test_synthetic_rumble_is_infrasonic_and_1_to_6_s():
    rng = np.random.default_rng(0)
    fracs = []
    for _ in range(50):
        r = synth.rumble(rng)
        p = np.abs(np.fft.rfft(r)) ** 2
        f = np.fft.rfftfreq(len(r), 1 / SAMPLE_RATE)
        fracs.append(p[(f >= BAND_HZ[0]) & (f <= BAND_HZ[1])].sum() / p.sum())
    assert np.median(fracs) > 0.7  # most rumble energy is in the 10-60 Hz band
    durs = [len(synth.rumble(rng)) / SAMPLE_RATE for _ in range(50)]
    assert 1.0 <= min(durs) and max(durs) <= 6.0


def test_make_clip_segments_match_frame_labels():
    clip, meta = synth.make_clip(np.random.default_rng(5), positive=True)
    (lo, hi), = meta["segments"]
    lab = frame_labels(meta["segments"])
    assert len(clip) == CLIP_SAMPLES and lab.shape == (N_FRAMES,)
    assert abs(lab.sum() * FRAME_HOP - (hi - lo)) <= FRAME_HOP
    assert frame_labels([]).sum() == 0


def test_sinc_filter_passes_its_band():
    conv = SincConv(n_filters=4, kernel_size=251)
    low, high = (v.detach() for v in conv.cutoffs())
    assert low.min() >= BAND_HZ[0] and high.max() <= BAND_HZ[1]
    t = torch.arange(10 * SAMPLE_RATE) / SAMPLE_RATE
    for k in range(4):
        centre = float((low[k] + high[k]) / 2)
        inband = conv(torch.sin(2 * np.pi * centre * t).view(1, 1, -1))[0, k, 300:-300].pow(2).mean()
        outband = conv(torch.sin(2 * np.pi * 100 * t).view(1, 1, -1))[0, k, 300:-300].pow(2).mean()
        assert inband > 20 * outband


def test_sinc_bands_stay_in_infrasound_band_after_drift():
    conv = SincConv(n_filters=8)
    with torch.no_grad():
        conv.low_hz_.add_(500.0)
        conv.band_hz_.add_(500.0)
    low, high = conv.cutoffs()
    assert low.min() >= BAND_HZ[0] and high.max() <= BAND_HZ[1] and (high > low).all()


def test_model_shapes_and_loss():
    model = SincLSTM()
    logits = model(torch.randn(3, CLIP_SAMPLES))
    assert logits.shape == (3, N_FRAMES)
    y = torch.tensor([1.0, 0.0, 1.0])
    fy = torch.zeros(3, N_FRAMES)
    fy[2] = -1  # unknown frames are ignored
    loss = loss_fn(logits, y, fy, 1.0)
    loss.backward()
    assert model.sinc.low_hz_.grad is not None  # sinc cut-offs are learned


def test_roc_auc():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_timeline_to_events():
    f = lambda s: int(round(s / FRAME_S))
    p = np.zeros(f(40))
    p[f(1):f(3)] = 0.9      # 2 s event
    p[f(3.4):f(4)] = 0.8    # 0.4 s gap -> merged with previous
    p[f(20):f(20.2)] = 0.9  # 0.2 s -> too short
    ev = timeline_to_events(p, 0.5, min_dur_s=0.5, max_gap_s=0.5)
    assert [(round(b, 2), round(e, 2)) for b, e, _ in ev] == [(1.0, 4.0)]


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
    pos_frames = ds.frames[ds.labels == 1]
    assert (pos_frames >= 0).all() and (pos_frames.sum(1) > 0).all()  # frame labels from Raven


def test_score_file_blocks_match_single_pass(tmp_path):
    rng = np.random.default_rng(0)
    audio = (rng.normal(size=SAMPLE_RATE * 130) * 0.01).astype(np.float32)
    wav = tmp_path / "long.wav"
    sf.write(wav, audio, SAMPLE_RATE)
    model = SincLSTM().eval()
    blocked = score_file(model, wav, block_minutes=1.0)
    single = frame_timeline(model, audio.astype(np.float64))
    assert blocked.shape == (130 * SAMPLE_RATE // FRAME_HOP,)
    assert np.isfinite(blocked).all() and (blocked > 0).all()
    assert np.abs(blocked - single).max() < 1e-4
