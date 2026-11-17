"""Datasets: synthetic clips, folders of real clips, and Raven selection tables.

Every item is (waveform[CLIP_SAMPLES], clip_label, frame_labels[N_FRAMES]).
Frame labels are -1 where unknown (e.g. real clips labelled only per clip);
those frames are ignored by the frame-level loss.
"""

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import synth
from .config import CLIP_SAMPLES, CLIP_SECONDS, N_FRAMES, SAMPLE_RATE
from .features import fit_length, frame_labels, resample

AUDIO_EXT = {".wav", ".flac", ".aif", ".aiff"}


class WaveDataset(torch.utils.data.Dataset):
    def __init__(self, waves: np.ndarray, labels: np.ndarray, frames: np.ndarray, meta=None, augment=False):
        self.waves = torch.from_numpy(np.ascontiguousarray(waves, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        self.frames = torch.from_numpy(np.asarray(frames, dtype=np.float32))
        self.meta = meta or [{} for _ in range(len(labels))]
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.waves[i]
        if self.augment:
            # gain + polarity are label-preserving; small additive white noise
            x = x * (torch.rand(1) * 1.5 + 0.25) * (1 if torch.rand(1) < 0.5 else -1)
            x = x + 0.02 * x.std() * torch.randn_like(x)
        return x, self.labels[i], self.frames[i]


# ---------------------------------------------------------------- synthetic
def _synth_item(args):
    seed, idx, snr_range = args
    rng = np.random.default_rng([seed, idx])
    clip, meta = synth.make_clip(rng, positive=bool(idx % 2), snr_db_range=snr_range)
    return clip, meta, frame_labels(meta["segments"])


def synthetic_dataset(n: int, seed: int, snr_range=(-6.0, 12.0), augment=False,
                      workers: int | None = None) -> WaveDataset:
    """Balanced synthetic dataset with exact frame labels (deterministic per seed)."""
    jobs = [(seed, i, snr_range) for i in range(n)]
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        items = list(ex.map(_synth_item, jobs, chunksize=64))
    waves = np.stack([c for c, _, _ in items])
    meta = [m for _, m, _ in items]
    frames = np.stack([f for _, _, f in items])
    return WaveDataset(waves, np.array([m["label"] for m in meta]), frames, meta, augment)


# ---------------------------------------------------------------- real data
def load_audio(path, start_s: float = 0.0, dur_s: float | None = None) -> np.ndarray:
    """Read (part of) an audio file, first channel, resampled to SAMPLE_RATE."""
    info = sf.info(str(path))
    start = int(start_s * info.samplerate)
    frames = -1 if dur_s is None else int(dur_s * info.samplerate)
    audio, sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    return resample(audio[:, 0], sr)


def folder_dataset(root, augment=False) -> WaveDataset:
    """``root/pos/*`` (rumble) and ``root/neg/*`` (no rumble) clips.

    If ``root/segments.json`` exists (written by :func:`clips_from_raven`) it
    supplies exact rumble positions per clip -> frame labels; otherwise positive
    clips get unknown (-1) frame labels and negatives get all-zero frames.
    """
    seg_file = Path(root, "segments.json")
    segs = json.loads(seg_file.read_text()) if seg_file.exists() else {}
    waves, labels, frames, meta = [], [], [], []
    for label, sub in ((1, "pos"), (0, "neg")):
        for p in sorted(Path(root, sub).glob("*")):
            if p.suffix.lower() not in AUDIO_EXT:
                continue
            waves.append(fit_length(load_audio(p)).astype(np.float32))
            labels.append(label)
            if label == 0:
                frames.append(np.zeros(N_FRAMES, np.int8))
            elif p.name in segs:
                frames.append(frame_labels(segs[p.name]))
            else:
                frames.append(np.full(N_FRAMES, -1, np.int8))
            meta.append({"label": label, "path": str(p)})
    if not waves:
        raise FileNotFoundError(f"no audio clips found under {root}/pos and {root}/neg")
    return WaveDataset(np.stack(waves), np.array(labels), np.stack(frames), meta, augment)


def read_raven_table(path) -> list[tuple[float, float]]:
    """(begin_s, end_s) of every selection in a Raven Pro selection table."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for r in rows:
        if r.get("View", "Spectrogram 1").startswith("Waveform"):
            continue  # Raven duplicates each selection per view
        out.append((float(r["Begin Time (s)"]), float(r["End Time (s)"])))
    return sorted(set(out))


def clips_from_raven(audio_path, table_path, out_dir, neg_per_pos: float = 1.0, seed: int = 0):
    """Cut CLIP_SECONDS positive clips centred on each Raven selection plus random
    negative clips that overlap no selection, write them as FLAC to
    ``out_dir/pos`` and ``out_dir/neg``, and record the rumble positions inside
    each positive clip in ``out_dir/segments.json`` (for frame labels)."""
    rng = np.random.default_rng(seed)
    events = read_raven_table(table_path)
    total = sf.info(str(audio_path)).duration
    out_dir = Path(out_dir)
    (out_dir / "pos").mkdir(parents=True, exist_ok=True)
    (out_dir / "neg").mkdir(parents=True, exist_ok=True)
    seg_file = out_dir / "segments.json"
    segs = json.loads(seg_file.read_text()) if seg_file.exists() else {}
    stem = Path(audio_path).stem
    for i, (b, e) in enumerate(events):
        start = min(max(0.0, (b + e) / 2 - CLIP_SECONDS / 2), max(0.0, total - CLIP_SECONDS))
        name = f"{stem}_{i:05d}.flac"
        sf.write(out_dir / "pos" / name, load_audio(audio_path, start, CLIP_SECONDS), SAMPLE_RATE)
        # every selection that falls inside this clip, in samples relative to the clip
        segs[name] = [(max(0, int((bb - start) * SAMPLE_RATE)), min(CLIP_SAMPLES, int((ee - start) * SAMPLE_RATE)))
                      for bb, ee in events if bb < start + CLIP_SECONDS and ee > start]
    seg_file.write_text(json.dumps(segs))
    n_neg, tries, written = int(len(events) * neg_per_pos), 0, 0
    while written < n_neg and tries < n_neg * 50:
        tries += 1
        s = rng.uniform(0, max(0.0, total - CLIP_SECONDS))
        if any(s < e + 1 and s + CLIP_SECONDS > b - 1 for b, e in events):
            continue
        sf.write(out_dir / "neg" / f"{stem}_neg{written:05d}.flac", load_audio(audio_path, s, CLIP_SECONDS), SAMPLE_RATE)
        written += 1
    return len(events), written
