"""Event-level check of the detector on a synthetic long recording with known rumbles.

    python -m elephant_rumble.demo_long --minutes 30 --rumbles 40
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import DETECT_MIN_DURATION, DETECT_THRESHOLD, SAMPLE_RATE
from .detect import load_model, score_file, timeline_to_events, write_raven
from .synth import make_long_recording


def match(truth, found):
    """A true rumble is detected if any event overlaps it; an event is a false
    alarm if it overlaps no true rumble."""
    ov = lambda a, b: a[0] < b[1] and b[0] < a[1]
    hits = [(t, next(f for f in found if ov(t, f))) for t in truth if any(ov(t, f) for f in found)]
    fa = sum(not any(ov(f, t) for t in truth) for f in found)
    err = lambda i: float(np.median([abs(t[i] - f[i]) for t, f in hits])) if hits else None
    return {"true_rumbles": len(truth), "detected": len(hits), "events": len(found), "false_alarms": fa,
            "event_recall": len(hits) / len(truth) if truth else None,
            "median_onset_error_s": err(0), "median_offset_error_s": err(1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/sinc_lstm.pt")
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

    model, _ = load_model(args.model)
    found = timeline_to_events(score_file(model, wav), DETECT_THRESHOLD, DETECT_MIN_DURATION)
    write_raven(found, out / "synthetic_recording.rumbles.txt")
    res = match(truth, [(b, e) for b, e, _ in found])
    res["false_alarms_per_hour"] = res["false_alarms"] / (args.minutes / 60)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
