"""Detect elephant rumbles in long recordings and write Raven selection tables.

    python -m elephant_rumble.detect recording.wav [more.wav ...] \
        --model models/sinc_lstm.pt --out-dir detections/

The file is streamed in blocks and scanned with overlapping 8-s windows
(default hop 4 s). The SincNet+LSTM model outputs a rumble probability every
50 ms; overlapping windows are averaged into one probability timeline, which
is thresholded into events (begin/end at frame resolution). Each event is one
row of a Raven Pro selection table (``<name>.rumbles.txt``).
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import CLIP_SAMPLES, FRAME_HOP, N_FRAMES, SAMPLE_RATE
from .features import fit_length, resample
from .model import SincLSTM

FRAME_S = FRAME_HOP / SAMPLE_RATE


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SincLSTM(**ckpt.get("arch", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, float(ckpt.get("threshold", 0.5))


@torch.no_grad()
def _frame_probs(model, windows: np.ndarray) -> np.ndarray:
    return torch.sigmoid(model(torch.from_numpy(windows.astype(np.float32)))).numpy()


def frame_timeline(model, audio: np.ndarray, hop_s: float = 4.0, batch: int = 64) -> np.ndarray:
    """Per-frame rumble probability for audio already at SAMPLE_RATE."""
    hop = int(round(hop_s / FRAME_S)) * FRAME_HOP
    n_frames = int(np.ceil(len(audio) / FRAME_HOP))
    starts = list(range(0, max(1, len(audio) - CLIP_SAMPLES + 1), hop))
    if starts[-1] + CLIP_SAMPLES < len(audio):
        starts.append(starts[-1] + hop)  # zero-padded tail window
    total = np.zeros(max(n_frames, starts[-1] // FRAME_HOP + N_FRAMES))
    count = np.zeros_like(total)
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        probs = _frame_probs(model, np.stack([fit_length(audio[s:s + CLIP_SAMPLES]) for s in chunk]))
        for s, p in zip(chunk, probs):
            f = s // FRAME_HOP
            total[f:f + N_FRAMES] += p
            count[f:f + N_FRAMES] += 1
    return (total / np.maximum(count, 1))[:n_frames]


def score_file(model, path, hop_s: float = 4.0, block_minutes: float = 10.0) -> np.ndarray:
    """Frame timeline for a (possibly day-long) file, streamed in blocks.

    Blocks overlap by one window so frames near block edges get the same
    context as everywhere else; the overlap is cross-faded away."""
    info = sf.info(str(path))
    sr = info.samplerate
    block_s = block_minutes * 60
    margin_s = CLIP_SAMPLES / SAMPLE_RATE
    n_total = int(np.ceil(info.frames / sr * SAMPLE_RATE / FRAME_HOP))
    out = np.zeros(n_total)
    t = 0.0
    while t * sr < info.frames:
        lo = max(0.0, t - margin_s)
        raw, _ = sf.read(str(path), start=int(lo * sr), frames=int((block_s + 2 * margin_s) * sr),
                         dtype="float32", always_2d=True)
        tl = frame_timeline(model, resample(raw[:, 0], sr), hop_s)
        f0 = int(round(lo / FRAME_S))
        a, b = int(round(t / FRAME_S)), min(n_total, int(round((t + block_s) / FRAME_S)))
        out[a:b] = tl[a - f0:b - f0]
        t += block_s
    return out


def timeline_to_events(probs, threshold, min_dur_s=0.5, max_gap_s=0.5):
    """Threshold a frame timeline into (begin_s, end_s, max_prob) events."""
    on = np.concatenate([[False], probs >= threshold, [False]])
    edges = np.flatnonzero(np.diff(on.astype(int)))
    runs = [[a, b] for a, b in zip(edges[::2], edges[1::2])]
    merged = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) * FRAME_S <= max_gap_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a * FRAME_S, b * FRAME_S, float(probs[a:b].max())) for a, b in merged if (b - a) * FRAME_S >= min_dur_s]


def write_raven(events, path, low_hz=5.0, high_hz=150.0):
    cols = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)", "Score"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for i, (b, e, s) in enumerate(events, 1):
            f.write(f"{i}\tSpectrogram 1\t1\t{b:.3f}\t{e:.3f}\t{low_hz}\t{high_hz}\t{s:.4f}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--model", default="models/sinc_lstm.pt")
    ap.add_argument("--threshold", type=float, help="default: value stored in the checkpoint")
    ap.add_argument("--hop", type=float, default=4.0, help="window hop in seconds")
    ap.add_argument("--min-dur", type=float, default=0.5, help="minimum event duration (s)")
    ap.add_argument("--out-dir", default="detections")
    args = ap.parse_args(argv)

    model, thr = load_model(args.model)
    thr = args.threshold if args.threshold is not None else thr
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.audio:
        events = timeline_to_events(score_file(model, path, args.hop), thr, args.min_dur)
        out = out_dir / (Path(path).stem + ".rumbles.txt")
        write_raven(events, out)
        print(f"{path}: {len(events)} rumble events -> {out}")


if __name__ == "__main__":
    main()
