"""Detect elephant rumbles in long recordings and write Raven selection tables.

    python -m elephant_rumble.detect recording.wav [more.wav ...] \
        --model models/rumble_cnn.pt --out-dir detections/

The file is scanned with 5-s windows (default hop 1 s), windows scoring above
the threshold are merged into events, and each event becomes one row of a
Raven Pro selection table (``<name>.rumbles.txt``), so results can be
reviewed directly in Raven alongside the spectrogram.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import CLIP_SAMPLES, CLIP_SECONDS, SAMPLE_RATE
from .features import resample, spectrogram
from .model import RumbleCNN


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = RumbleCNN(width=ckpt.get("width", 16))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, float(ckpt.get("threshold", 0.5))


@torch.no_grad()
def score_audio(model, audio: np.ndarray, hop_s: float = 1.0, batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Score windows of audio already at SAMPLE_RATE. Returns (start_times_s, probs)."""
    hop = int(hop_s * SAMPLE_RATE)
    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    starts = np.arange(0, len(audio) - CLIP_SAMPLES + 1, hop)
    probs = []
    for i in range(0, len(starts), batch):
        specs = np.stack([spectrogram(audio[s:s + CLIP_SAMPLES]) for s in starts[i:i + batch]])
        probs.append(torch.sigmoid(model(torch.from_numpy(specs).unsqueeze(1))).numpy())
    return starts / SAMPLE_RATE, np.concatenate(probs) if probs else np.array([])


def score_file(model, path, hop_s: float = 1.0, block_minutes: float = 10.0):
    """Stream a (possibly day-long) file in blocks so memory stays bounded."""
    info = sf.info(str(path))
    sr = info.samplerate
    block = int(block_minutes * 60 * sr)
    overlap = int(CLIP_SECONDS * sr)
    times, probs = [], []
    offset = 0
    while offset < info.frames:
        raw, _ = sf.read(str(path), start=offset, frames=block + overlap, dtype="float32", always_2d=True)
        audio = resample(raw[:, 0], sr)
        t, p = score_audio(model, audio, hop_s)
        t = t + offset / sr
        keep = t < (offset + block) / sr  # windows starting in the overlap belong to the next block
        if offset + block >= info.frames:
            keep[:] = True
        times.append(t[keep])
        probs.append(p[keep])
        offset += block
    return np.concatenate(times), np.concatenate(probs)


def windows_to_events(times, probs, threshold, min_windows: int = 1, gap_s: float = 1.0):
    """Merge overlapping/adjacent positive windows into (begin, end, max_score) events."""
    events = []
    cur = None
    for t, p in zip(times, probs):
        if p < threshold:
            continue
        b, e = float(t), float(t) + CLIP_SECONDS
        if cur and b <= cur[1] + gap_s:
            cur[1], cur[2], cur[3] = e, max(cur[2], float(p)), cur[3] + 1
        else:
            if cur:
                events.append(cur)
            cur = [b, e, float(p), 1]
    if cur:
        events.append(cur)
    return [(b, e, s) for b, e, s, n in events if n >= min_windows]


def write_raven(events, path, low_hz=5.0, high_hz=150.0):
    cols = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)", "Score"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for i, (b, e, s) in enumerate(events, 1):
            f.write(f"{i}\tSpectrogram 1\t1\t{b:.3f}\t{e:.3f}\t{low_hz}\t{high_hz}\t{s:.4f}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--model", default="models/rumble_cnn.pt")
    ap.add_argument("--threshold", type=float, help="default: value stored in the checkpoint")
    ap.add_argument("--hop", type=float, default=1.0, help="window hop in seconds")
    ap.add_argument("--min-windows", type=int, default=2, help="min consecutive positive windows per event")
    ap.add_argument("--out-dir", default="detections")
    args = ap.parse_args(argv)

    model, thr = load_model(args.model)
    thr = args.threshold if args.threshold is not None else thr
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.audio:
        t, p = score_file(model, path, args.hop)
        events = windows_to_events(t, p, thr, args.min_windows)
        out = out_dir / (Path(path).stem + ".rumbles.txt")
        write_raven(events, out)
        print(f"{path}: {len(events)} rumble events -> {out}")


if __name__ == "__main__":
    main()
