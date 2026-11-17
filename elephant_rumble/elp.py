"""Real elephant rumbles from the Cornell Elephant Listening Project (ELP).

Audio:  24-h recordings from the Nouabale-Ndoki (PNNN) acoustic grid, Republic of
        Congo, published as AWS Open Data: s3://congo8khz-pnnn (8 kHz, 16-bit,
        free, no account needed). Attribution: "Cornell Elephant Listening Project".
Labels: expert rumble selection tables (Raven: file, start offset, duration)
        for 163 of those 24-h files, published in the GitHub repository
        statlogic/Elephant-Rumble-Detection (Resources/*_rumble_info.txt).

Only the seconds around each rumble (and around randomly chosen negative
times) are fetched with HTTP Range requests, so nothing near the 1.4 GB per
24-h file is downloaded.

    python -m elephant_rumble.elp --out data/elp            # download (~10 min)
    python -m elephant_rumble.train --data elp:data/elp      # train / evaluate

Split: 80/10/10 by *recording file* (whole 24-h days), so windows cut from the
same day never appear in both training and test.
"""

import argparse
import csv
import io
import json
import struct
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .config import CLIP_SAMPLES, MIN_RUMBLE_IN_WINDOW, N_FRAMES, SAMPLE_RATE
from .features import frame_labels, resample

BUCKET = "https://congo8khz-pnnn.s3.us-west-2.amazonaws.com/recordings/wav"
LABELS = "https://raw.githubusercontent.com/statlogic/Elephant-Rumble-Detection/master/Resources/"
LABEL_FILES = ["TrainingSet_rumble_info.txt", "GeneralTest_rumble_info.txt"]
PAD_S = 4.0          # context kept on each side of a rumble
NEG_LEN_S = 12.0     # length of each negative segment
NEG_MARGIN_S = 10.0  # negatives stay this far from any labelled rumble
DAY_S = 24 * 3600


def _get(url, byte_range=None, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {})
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def load_labels() -> dict[str, list[tuple[float, float]]]:
    """{wav filename: sorted [(begin_s, end_s), ...]} from the public selection tables."""
    out = {}
    for name in LABEL_FILES:
        rows = csv.DictReader(io.StringIO(_get(LABELS + name).decode()), delimiter="\t")
        for r in rows:
            b = float(r["File Offset (s)"])
            out.setdefault(r["filename"], []).append((b, b + float(r["duration"])))
    return {k: sorted(set(v)) for k, v in out.items()}


class RemoteWav:
    """Random access to a PCM WAV in the bucket via HTTP Range requests."""

    def __init__(self, filename):
        self.url = f"{BUCKET}/{filename[:5]}/{filename}"
        head = _get(self.url, (0, 4095))
        i = 12
        while i < len(head) - 8:
            cid, size = head[i:i + 4], struct.unpack("<I", head[i + 4:i + 8])[0]
            if cid == b"fmt ":
                _, self.channels, self.sr, _, _, self.bits = struct.unpack("<HHIIHH", head[i + 8:i + 24])
            if cid == b"data":
                self.data0, self.data_size = i + 8, size
                break
            i += 8 + size + (size & 1)
        assert self.bits == 16, "expected 16-bit PCM"
        self.frame_bytes = self.channels * 2
        self.duration = self.data_size / self.frame_bytes / self.sr

    def read(self, start_s, dur_s) -> np.ndarray:
        """float32 audio, first channel, resampled to SAMPLE_RATE."""
        start_s = max(0.0, start_s)
        n = int(min(dur_s, self.duration - start_s) * self.sr)
        a = self.data0 + int(start_s * self.sr) * self.frame_bytes
        raw = np.frombuffer(_get(self.url, (a, a + n * self.frame_bytes - 1)), "<i2")
        x = raw.reshape(-1, self.channels)[:, 0].astype(np.float32) / 32768
        return resample(x, self.sr).astype(np.float32)


def _segments_for_file(events, rng, total_s=DAY_S):
    """Positive segments around (merged) rumbles + the same number of negatives."""
    merged = []
    for b, e in events:
        if merged and b - PAD_S <= merged[-1][1] + PAD_S:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([b, e])
    segs = [("pos", b - PAD_S, e - b + 2 * PAD_S) for b, e in merged]
    n_pos = sum(1 for _ in events)
    negs, tries = [], 0
    while len(negs) < n_pos and tries < 1000 * max(1, n_pos):
        tries += 1
        s = rng.uniform(0, total_s - NEG_LEN_S)
        if any(s < e + NEG_MARGIN_S and s + NEG_LEN_S > b - NEG_MARGIN_S for b, e in events):
            continue
        if any(abs(s - t) < NEG_LEN_S for t in negs):
            continue
        negs.append(s)
    segs += [("neg", s, NEG_LEN_S) for s in negs]
    return segs


def download(out_dir, workers=16, seed=0, max_files=None):
    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    files = sorted(labels)[:max_files] if max_files else sorted(labels)
    rng = np.random.default_rng(seed)
    print(f"{sum(len(labels[f]) for f in files)} labelled rumbles in {len(files)} 24-h files")

    def work(filename):
        try:
            wav = RemoteWav(filename)
        except Exception as e:  # file missing from bucket
            return filename, [], str(e)
        events = labels[filename]
        items = []
        for k, (kind, start, dur) in enumerate(_segments_for_file(events, np.random.default_rng([seed, zlib.crc32(filename.encode())]), wav.duration)):
            start = max(0.0, start)
            path = out_dir / "audio" / f"{filename[:-4]}_{kind}{k:04d}.npy"
            if not path.exists():
                np.save(path, wav.read(start, dur))
            rel = [(round(b - start, 3), round(e - start, 3)) for b, e in events if b < start + dur and e > start]
            items.append({"file": filename, "path": path.name, "kind": kind, "start_s": start, "rumbles_s": rel})
        return filename, items, None

    index, failed = [], []
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, (f, items, err) in enumerate(ex.map(work, files), 1):
            index += items
            if err:
                failed.append(f)
            if i % 10 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files, {len(index)} segments, {time.time() - t0:.0f}s", flush=True)

    # 80/10/10 split by recording file
    order = list(rng.permutation(sorted({it["file"] for it in index})))
    n = len(order)
    split = {f: ("train" if i < 0.8 * n else "val" if i < 0.9 * n else "test") for i, f in enumerate(order)}
    for it in index:
        it["split"] = split[it["file"]]
    meta = {"source_audio": "s3://congo8khz-pnnn (Cornell Elephant Listening Project, AWS Open Data)",
            "source_labels": "github.com/statlogic/Elephant-Rumble-Detection Resources/*_rumble_info.txt",
            "sample_rate": SAMPLE_RATE, "failed_files": failed, "segments": index}
    (out_dir / "index.json").write_text(json.dumps(meta))
    summary(out_dir)


def summary(out_dir):
    idx = json.loads(Path(out_dir, "index.json").read_text())["segments"]
    for sp in ("train", "val", "test"):
        s = [it for it in idx if it["split"] == sp]
        print(f"{sp:5s}: {len({it['file'] for it in s}):3d} files, "
              f"{sum(len(it['rumbles_s']) for it in s if it['kind'] == 'pos'):5d} rumbles in "
              f"{sum(it['kind'] == 'pos' for it in s):5d} positive segments, {sum(it['kind'] == 'neg' for it in s):5d} negative segments")


class ELPWindows(torch.utils.data.Dataset):
    """3-s windows cut from downloaded ELP segments.

    Positive windows contain >= MIN_RUMBLE_IN_WINDOW s of a labelled rumble;
    negative windows come from negative segments (>= 10 s from any label).
    Train mode draws a new random window every time (data augmentation);
    eval mode uses fixed windows. One window per labelled rumble and one per
    negative segment -> balanced classes.
    """

    def __init__(self, root, split, train=False, seed=0):
        self.root = Path(root)
        idx = json.loads((self.root / "index.json").read_text())["segments"]
        self.items = []  # (segment, rumble_index or None)
        for it in idx:
            if it["split"] != split:
                continue
            if it["kind"] == "neg":
                self.items.append((it, None))
            else:
                self.items += [(it, k) for k in range(len(it["rumbles_s"]))]
        self.train = train
        self.seed = seed
        self._cache = {}
        self.labels = torch.tensor([0.0 if k is None else 1.0 for _, k in self.items])
        self.meta = [{"label": int(k is not None), "file": it["file"]} for it, k in self.items]
        if not train:  # materialise fixed windows so metrics are reproducible
            xs, fs = zip(*(self._window(i, np.random.default_rng([seed, i])) for i in range(len(self.items))))
            self.waves, self.frames = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(fs).astype(np.float32))

    def _audio(self, name):
        if name not in self._cache:
            self._cache[name] = np.load(self.root / "audio" / name)
        return self._cache[name]

    def _window(self, i, rng):
        seg, k = self.items[i]
        x = self._audio(seg["path"])
        L = len(x)
        if k is None:
            start = int(rng.integers(0, max(1, L - CLIP_SAMPLES)))
        else:
            b, e = (int(v * SAMPLE_RATE) for v in seg["rumbles_s"][k])
            need = min(e - b, int(MIN_RUMBLE_IN_WINDOW * SAMPLE_RATE))
            lo, hi = max(0, b + need - CLIP_SAMPLES), min(L - CLIP_SAMPLES, e - need)
            start = int(rng.integers(lo, max(lo, hi) + 1))
        w = x[start:start + CLIP_SAMPLES]
        if len(w) < CLIP_SAMPLES:
            w = np.pad(w, (0, CLIP_SAMPLES - len(w)))
        segs = [(int(b * SAMPLE_RATE) - start, int(e * SAMPLE_RATE) - start) for b, e in seg["rumbles_s"]]
        segs = [(max(0, b), min(CLIP_SAMPLES, e)) for b, e in segs if e > 0 and b < CLIP_SAMPLES]
        return w.astype(np.float32), frame_labels(segs)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if not self.train:
            return self.waves[i], self.labels[i], self.frames[i]
        x, f = self._window(i, np.random.default_rng())
        x = torch.from_numpy(x)
        x = x * (torch.rand(1) * 1.5 + 0.25) * (1 if torch.rand(1) < 0.5 else -1)
        return x, self.labels[i], torch.from_numpy(f.astype(np.float32))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/elp")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-files", type=int, help="only the first N 24-h files (quick test)")
    args = ap.parse_args(argv)
    download(args.out, args.workers, max_files=args.max_files)


if __name__ == "__main__":
    main()
