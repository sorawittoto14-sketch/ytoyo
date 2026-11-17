"""Event-level check of the detector on a synthetic long recording with known rumbles.

    python -m elephant_rumble.demo_long --minutes 30 --rumbles 40
"""

import argparse
import json
from pathlib import Path

import soundfile as sf

from .config import SAMPLE_RATE
from .detect import load_model, score_file, windows_to_events, write_raven
from .synth import make_long_recording


def match(truth, found):
    """A true rumble is detected if any event overlaps it; an event is a false
    alarm if it overlaps no true rumble."""
    ov = lambda a, b: a[0] < b[1] and b[0] < a[1]
    hit = sum(any(ov(t, f) for f in found) for t in truth)
    fa = sum(not any(ov(f, t) for t in truth) for f in found)
    return {"true_rumbles": len(truth), "detected": hit, "events": len(found), "false_alarms": fa,
            "event_recall": hit / len(truth) if truth else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/rumble_cnn.pt")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--rumbles", type=int, default=40)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-dir", default="detections/demo")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audio, truth = make_long_recording(args.seed, args.minutes, args.rumbles)
    wav = out / "synthetic_recording.wav"
    sf.write(wav, audio, SAMPLE_RATE)
    write_raven([(b, e, 1.0) for b, e in truth], out / "synthetic_recording.truth.txt")

    model, thr = load_model(args.model)
    t, p = score_file(model, wav)
    found = windows_to_events(t, p, thr, min_windows=2)
    write_raven(found, out / "synthetic_recording.rumbles.txt")
    res = match(truth, [(b, e) for b, e, _ in found])
    res["false_alarms_per_hour"] = res["false_alarms"] / (args.minutes / 60)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
