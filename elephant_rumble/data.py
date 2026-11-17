"""Datasets: synthetic clips, folders of real clips, and Raven selection tables."""

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import synth
from .config import CLIP_SAMPLES, CLIP_SECONDS, SAMPLE_RATE
from .features import fit_length, resample, spectrogram

AUDIO_EXT = {".wav", ".flac", ".aif", ".aiff"}


class SpecDataset(torch.utils.data.Dataset):
    """Pre-computed spectrograms with optional SpecAugment-style masking."""

    def __init__(self, specs: np.ndarray, labels: np.ndarray, meta=None, augment=False):
        self.specs = torch.from_numpy(specs).unsqueeze(1)  # (N, 1, F, T)
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.meta = meta or [{} for _ in range(len(labels))]
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.specs[i]
        if self.augment:
            x = x.clone()
            _, f, t = x.shape
            if torch.rand(1) < 0.5:  # time mask
                w = int(torch.randint(1, t // 8, (1,)))
                s = int(torch.randint(0, t - w, (1,)))
                x[:, :, s:s + w] = 0
            if torch.rand(1) < 0.5:  # frequency mask
                w = int(torch.randint(1, f // 10, (1,)))
                s = int(torch.randint(0, f - w, (1,)))
                x[:, s:s + w, :] = 0
            x = x + 0.05 * torch.randn_like(x)
        return x, self.labels[i]


# ---------------------------------------------------------------- synthetic
def _synth_item(args):
    seed, idx, snr_range = args
    rng = np.random.default_rng([seed, idx])
    clip, meta = synth.make_clip(rng, positive=bool(idx % 2), snr_db_range=snr_range)
    return spectrogram(clip), meta


def synthetic_dataset(n: int, seed: int, snr_range=(-6.0, 12.0), augment=False,
                      workers: int | None = None) -> SpecDataset:
    """Balanced synthetic dataset (deterministic for a given seed)."""
    jobs = [(seed, i, snr_range) for i in range(n)]
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        items = list(ex.map(_synth_item, jobs, chunksize=64))
    specs = np.stack([s for s, _ in items])
    meta = [m for _, m in items]
    labels = np.array([m["label"] for m in meta])
    return SpecDataset(specs, labels, meta, augment)


# ---------------------------------------------------------------- real data
def load_audio(path, start_s: float = 0.0, dur_s: float | None = None) -> np.ndarray:
    """Read (part of) an audio file, first channel, resampled to SAMPLE_RATE."""
    info = sf.info(str(path))
    start = int(start_s * info.samplerate)
    frames = -1 if dur_s is None else int(dur_s * info.samplerate)
    audio, sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    return resample(audio[:, 0], sr)


def folder_dataset(root, augment=False) -> SpecDataset:
    """``root/pos/*`` (rumble) and ``root/neg/*`` (no rumble) 5-s clips."""
    specs, labels, meta = [], [], []
    for label, sub in ((1, "pos"), (0, "neg")):
        for p in sorted(Path(root, sub).glob("*")):
            if p.suffix.lower() in AUDIO_EXT:
                specs.append(spectrogram(fit_length(load_audio(p), CLIP_SAMPLES)))
                labels.append(label)
                meta.append({"label": label, "path": str(p)})
    if not specs:
        raise FileNotFoundError(f"no audio clips found under {root}/pos and {root}/neg")
    return SpecDataset(np.stack(specs), np.array(labels), meta, augment)


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
    """Cut 5-s positive clips centred on each selection plus random negative
    clips that do not overlap any selection, and write them as FLAC into
    ``out_dir/pos`` and ``out_dir/neg`` (ready for :func:`folder_dataset`)."""
    rng = np.random.default_rng(seed)
    events = read_raven_table(table_path)
    total = sf.info(str(audio_path)).duration
    out_dir = Path(out_dir)
    (out_dir / "pos").mkdir(parents=True, exist_ok=True)
    (out_dir / "neg").mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem
    for i, (b, e) in enumerate(events):
        start = min(max(0.0, (b + e) / 2 - CLIP_SECONDS / 2), max(0.0, total - CLIP_SECONDS))
        sf.write(out_dir / "pos" / f"{stem}_{i:05d}.flac", load_audio(audio_path, start, CLIP_SECONDS), SAMPLE_RATE)
    n_neg, tries = int(len(events) * neg_per_pos), 0
    written = 0
    while written < n_neg and tries < n_neg * 50:
        tries += 1
        s = rng.uniform(0, max(0.0, total - CLIP_SECONDS))
        if any(s < e + 1 and s + CLIP_SECONDS > b - 1 for b, e in events):
            continue
        sf.write(out_dir / "neg" / f"{stem}_neg{written:05d}.flac", load_audio(audio_path, s, CLIP_SECONDS), SAMPLE_RATE)
        written += 1
    return len(events), written
