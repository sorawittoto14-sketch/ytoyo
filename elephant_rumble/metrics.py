import numpy as np


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC (ties averaged)."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = np.asarray(score)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_report(y, prob, threshold=0.5) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(y), "threshold": threshold,
        "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "roc_auc": roc_auc(y, prob), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
