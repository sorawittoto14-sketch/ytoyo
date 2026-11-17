"""Train the SincNet + BiLSTM rumble detector.

Synthetic data (default)::

    python -m elephant_rumble.train --out models/sinc_lstm.pt

Real Cornell ELP rumbles (download first with ``python -m elephant_rumble.elp``)::

    python -m elephant_rumble.train --data elp:data/elp --init models/sinc_lstm.pt --lr 1e-3 --out models/sinc_lstm_elp.pt

Real clips in ``DIR/pos`` and ``DIR/neg`` (80/10/10 random split; synthetic
defaults are also 80/10/10: 20,000 / 2,500 / 2,500), optionally
starting from the synthetic model::

    python -m elephant_rumble.train --data DIR --init models/sinc_lstm.pt --lr 5e-4

Loss = BCE on the window score (max over frames) + BCE on every frame whose
label is known (frame labels come from synthetic ground truth or Raven
selection tables).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from . import config
from .data import WaveDataset, folder_dataset, synthetic_dataset
from .metrics import binary_report
from .model import SincLSTM


@torch.no_grad()
def predict(model, ds, batch=128):
    """Returns (clip_probs (N,), frame_probs (N, T))."""
    model.eval()
    frames = []
    for x, _, _ in DataLoader(ds, batch_size=batch):
        frames.append(torch.sigmoid(model(x)).numpy())
    frames = np.concatenate(frames)
    return frames.max(1), frames


def _arrays(ds):
    if isinstance(ds, Subset):
        idx = np.asarray(ds.indices)
        return ds.dataset.labels.numpy()[idx], ds.dataset.frames.numpy()[idx]
    return ds.labels.numpy(), ds.frames.numpy()


def best_f1_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda t: binary_report(y, p, t)["f1"]))


def frame_report(frame_y, frame_p, thr) -> dict:
    known = frame_y >= 0
    if not known.any():
        return {}
    r = binary_report(frame_y[known], frame_p[known], thr)
    return {k: r[k] for k in ("precision", "recall", "f1", "roc_auc")} | {"n_frames": int(known.sum())}


def build_datasets(args):
    if args.data == "synthetic":
        t0 = time.time()
        train = synthetic_dataset(args.n_train, seed=1, augment=True)
        val = synthetic_dataset(args.n_val, seed=2)
        test = synthetic_dataset(args.n_test, seed=3)
        print(f"generated synthetic data in {time.time() - t0:.0f}s")
        return train, val, test
    if args.data.startswith("elp:"):
        from .elp import ELPWindows, summary
        root = args.data[4:]
        summary(root)
        return ELPWindows(root, "train", train=True), ELPWindows(root, "val"), ELPWindows(root, "test")
    full = folder_dataset(args.data)
    aug = WaveDataset(full.waves.numpy(), full.labels.numpy(), full.frames.numpy(), full.meta, augment=True)
    idx = np.random.default_rng(0).permutation(len(full))
    n_val = n_test = max(1, len(full) // 10)
    return (Subset(aug, idx[n_val + n_test:].tolist()),
            Subset(full, idx[:n_val].tolist()),
            Subset(full, idx[n_val:n_val + n_test].tolist()))


def synthetic_breakdown(ds, prob, threshold) -> dict:
    """Recall per SNR / duration bucket and false-positive rate per distractor."""
    out = {"recall_by_snr": {}, "recall_by_duration": {}, "fpr_by_distractor_in_negatives": {}}
    pos = [i for i, m in enumerate(ds.meta) if m["label"] == 1]
    for lo, hi in [(-6, -2), (-2, 2), (2, 6), (6, 12)]:
        sel = [i for i in pos if lo <= ds.meta[i]["snr_db"] < hi]
        if sel:
            out["recall_by_snr"][f"{lo}..{hi} dB"] = {"n": len(sel), "recall": float(np.mean(prob[sel] >= threshold))}
    for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 3.1)]:
        sel = [i for i in pos if lo <= (ds.meta[i]["segments"][0][1] - ds.meta[i]["segments"][0][0]) / config.SAMPLE_RATE < hi]
        if sel:
            out["recall_by_duration"][f"{lo}-{hi:g} s in window"] = {"n": len(sel), "recall": float(np.mean(prob[sel] >= threshold))}
    neg = [i for i, m in enumerate(ds.meta) if m["label"] == 0]
    for name in ["none", "engine", "aircraft", "thunder", "wind", "insects"]:
        sel = [i for i in neg if (name in ds.meta[i]["distractors"]) or (name == "none" and not ds.meta[i]["distractors"])]
        if sel:
            out["fpr_by_distractor_in_negatives"][name] = {"n": len(sel), "false_positive_rate": float(np.mean(prob[sel] >= threshold))}
    return out


def loss_fn(frame_logits, y, fy, frame_weight):
    clip_loss = F.binary_cross_entropy_with_logits(SincLSTM.clip_logit(frame_logits), y)
    known = fy >= 0
    if not known.any():
        return clip_loss
    frame_loss = F.binary_cross_entropy_with_logits(frame_logits[known], fy[known])
    return clip_loss + frame_weight * frame_loss


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="synthetic",
                    help="'synthetic', 'elp:DIR' (real Cornell ELP data from elephant_rumble.elp) or a folder with pos/ and neg/")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2500)
    ap.add_argument("--n-test", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--frame-weight", type=float, default=1.0)
    ap.add_argument("--n-filters", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--init", help="checkpoint to fine-tune from")
    ap.add_argument("--out", default="models/sinc_lstm.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    train, val, test = build_datasets(args)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    arch = {"n_filters": args.n_filters, "hidden": args.hidden}
    model = SincLSTM(**arch)
    if args.init:
        model.load_state_dict(torch.load(args.init, weights_only=False)["state_dict"])
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader))

    y_val, fy_val = _arrays(val)
    best, best_state, history = -1.0, None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y, fy in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y, fy, args.frame_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
        p_val, fp_val = predict(model, val)
        rep = binary_report(y_val, p_val)
        fr = frame_report(fy_val, fp_val, 0.5)
        history.append({"epoch": ep, "train_loss": tot / (len(loader) * args.batch),
                        "val_auc": rep["roc_auc"], "val_f1": rep["f1"], "val_frame_f1": fr.get("f1")})
        print(f"epoch {ep:2d} loss {history[-1]['train_loss']:.4f} val auc {rep['roc_auc']:.4f} "
              f"f1 {rep['f1']:.4f} acc {rep['accuracy']:.4f} frame-f1 {fr.get('f1', float('nan')):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if rep["roc_auc"] > best:
            best, best_state = rep["roc_auc"], {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    p_val, _ = predict(model, val)
    thr = best_f1_threshold(y_val, p_val)
    y_test, fy_test = _arrays(test)
    p_test, fp_test = predict(model, test)
    low, high = (v.detach() for v in model.sinc.cutoffs())
    result = {"test_clip": binary_report(y_test, p_test, thr), "test_frame": frame_report(fy_test, fp_test, thr),
              "val_best_auc": best, "threshold": thr,
              "learned_sinc_bands_hz": [[round(float(a), 1), round(float(b), 1)] for a, b in zip(low[:, 0], high[:, 0])],
              "history": history, "args": vars(args)}
    if args.data == "synthetic":
        result["test_breakdown"] = synthetic_breakdown(test, p_test, thr)
    print(json.dumps({k: result[k] for k in ("test_clip", "test_frame")}, indent=2))
    if "test_breakdown" in result:
        print(json.dumps(result["test_breakdown"], indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "arch": arch, "threshold": thr,
                "config": {k: getattr(config, k) for k in dir(config) if k.isupper()}}, out)
    out.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
