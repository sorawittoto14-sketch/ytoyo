"""Train the rumble CNN.

Synthetic data (default)::

    python -m elephant_rumble.train --out models/rumble_cnn.pt

Real 5-s clips in ``DIR/pos`` and ``DIR/neg`` (80/10/10 random split)::

    python -m elephant_rumble.train --data DIR --out models/rumble_real.pt
    # optionally start from the synthetic model:
    python -m elephant_rumble.train --data DIR --init models/rumble_cnn.pt --lr 5e-4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import config
from .data import SpecDataset, folder_dataset, synthetic_dataset
from .metrics import binary_report
from .model import RumbleCNN


@torch.no_grad()
def predict(model, ds, batch=256) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in DataLoader(ds, batch_size=batch):
        out.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(out)


def _labels(ds) -> np.ndarray:
    if isinstance(ds, Subset):
        return ds.dataset.labels.numpy()[ds.indices]
    return ds.labels.numpy()


def best_f1_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda t: binary_report(y, p, t)["f1"]))


def build_datasets(args):
    if args.data == "synthetic":
        t0 = time.time()
        train = synthetic_dataset(args.n_train, seed=1, augment=True)
        val = synthetic_dataset(args.n_val, seed=2)
        test = synthetic_dataset(args.n_test, seed=3)
        print(f"generated synthetic data in {time.time() - t0:.0f}s")
        return train, val, test
    full = folder_dataset(args.data)
    aug = SpecDataset(full.specs.squeeze(1).numpy(), full.labels.numpy(), full.meta, augment=True)
    idx = np.random.default_rng(0).permutation(len(full))
    n_val = n_test = max(1, len(full) // 10)
    return (Subset(aug, idx[n_val + n_test:].tolist()),
            Subset(full, idx[:n_val].tolist()),
            Subset(full, idx[n_val:n_val + n_test].tolist()))


def snr_breakdown(ds, prob, threshold) -> dict:
    """Recall per SNR bucket (synthetic positives only)."""
    buckets = {}
    for (lo, hi) in [(-6, -2), (-2, 2), (2, 6), (6, 12)]:
        sel = [i for i, m in enumerate(ds.meta) if m.get("snr_db") is not None and lo <= m["snr_db"] < hi]
        if sel:
            buckets[f"{lo}..{hi} dB"] = {"n": len(sel), "recall": float(np.mean(prob[sel] >= threshold))}
    neg = [i for i, m in enumerate(ds.meta) if m.get("label") == 0]
    fp_by = {}
    for name in ["none", "engine", "aircraft", "thunder", "wind", "insects"]:
        sel = [i for i in neg if (name in ds.meta[i]["distractors"]) or (name == "none" and not ds.meta[i]["distractors"])]
        if sel:
            fp_by[name] = {"n": len(sel), "false_positive_rate": float(np.mean(prob[sel] >= threshold))}
    return {"recall_by_snr": buckets, "fpr_by_distractor_in_negatives": fp_by}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="synthetic", help="'synthetic' or a folder with pos/ and neg/")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--init", help="checkpoint to fine-tune from")
    ap.add_argument("--out", default="models/rumble_cnn.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    train, val, test = build_datasets(args)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    model = RumbleCNN(width=args.width)
    if args.init:
        model.load_state_dict(torch.load(args.init, weights_only=False)["state_dict"])
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader))
    lossf = torch.nn.BCEWithLogitsLoss()

    y_val = _labels(val)
    best, best_state, history = -1.0, None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
        rep = binary_report(y_val, predict(model, val))
        history.append({"epoch": ep, "train_loss": tot / (len(loader) * args.batch), **{k: rep[k] for k in ("roc_auc", "f1", "accuracy")}})
        print(f"epoch {ep:2d} loss {history[-1]['train_loss']:.4f} val auc {rep['roc_auc']:.4f} "
              f"f1 {rep['f1']:.4f} acc {rep['accuracy']:.4f} ({time.time() - t0:.0f}s)")
        if rep["roc_auc"] > best:
            best, best_state = rep["roc_auc"], {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    thr = best_f1_threshold(y_val, predict(model, val))
    p_test = predict(model, test)
    result = {"test": binary_report(_labels(test), p_test, thr), "val_best_auc": best,
              "threshold": thr, "history": history, "args": vars(args)}
    if args.data == "synthetic":
        result["test_breakdown"] = snr_breakdown(test, p_test, thr)
    print(json.dumps(result["test"], indent=2))
    if "test_breakdown" in result:
        print(json.dumps(result["test_breakdown"], indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "width": args.width, "threshold": thr,
                "config": {k: getattr(config, k) for k in dir(config) if k.isupper()}}, out)
    out.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
