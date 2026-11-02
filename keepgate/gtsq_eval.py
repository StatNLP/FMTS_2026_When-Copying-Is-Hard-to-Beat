#!/usr/bin/env python3
"""Single source of truth for GTSQ decoding + the four evaluation metrics.

Before this module the same logic was copy-pasted across 03_train.py (frozen training path),
05_baselines.py, perfeature_report.py, 06_data_analysis.py, 08_probes.py, fm_baseline.py and
ordinal_metric.py -- which is exactly why the accuracy convention diverged (some files pooled by
cell, some averaged per feature) and had to be fixed in three places. Every analysis script now
imports from here so a convention change is made once.

Conventions (must match 03_train.eval_gtsq, the training-time reference):
  * decode: binary -> logit>0 ; ordinal -> argmax. From RAW class values: round + clip to range.
  * macro-F1: per-feature f1_score(average='macro', labels=range(nclass)), then averaged.
  * accuracy / change-accuracy: CELL-POOLED (every observed cell weighted equally).
  * "changed" = day-D class != day-D+1 class.
"""
import glob
import os
import re

import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score


def seed_of(run_dir):
    """Seed parsed from a run directory name, or None.

    Anchored on '_s<digits>' followed by end-of-name or '_' -- NOT str.split('_s')[-1], which breaks
    on any trailing tag ('_shuf', '_sup', '_genc1', '_cdetf2.0w10.0') and then silently mislabels or
    drops runs (audit 2026-07-27, bug 1.3)."""
    m = re.search(r'_s(\d+)(?:_|$)', os.path.basename(str(run_dir).rstrip('/')))
    return m.group(1) if m else None


def canon_runs(pattern, root='02_models', exclude=('_shuf', '_genc', 'uniform', 'ffilm', 'fcrossattn')):
    """One canonical run per seed for `pattern` (glob, relative to `root`).

    Fixes audit bug 1.3: the old idiom `by[seed] = r` OVERWROTE, so several *variants* matching the
    same glob could be silently merged into one 'seed set'. Here variants are excluded by name and
    any remaining collision is reported instead of hidden; the first (sorted) run wins deterministically."""
    by = {}
    for r in sorted(glob.glob(os.path.join(root, pattern))):
        b = os.path.basename(r)
        if any(t in b for t in exclude):
            continue
        sd = seed_of(r)
        if sd is not None:
            by.setdefault(sd, []).append(r)
    dupes = {k: v for k, v in by.items() if len(v) > 1}
    if dupes:
        k = sorted(dupes)[0]
        print(f"  [canon_runs] WARNING {len(dupes)} seed(s) match >1 run for '{pattern}'; taking the "
              f"first. Tighten the glob if variants are being mixed. seed {k}: "
              f"{[os.path.basename(x)[:58] for x in dupes[k][:2]]}")
    return [v[0] for v in by.values()]


def round_to_class(raw, ftype, nclass):
    """Round a raw (already de-normalised) value/array to a valid class label: clip(round(.), 0, hi),
    hi = 1 for binary else nclass-1. The one place the round+clip rule lives."""
    hi = 1 if str(ftype) == "binary" else int(nclass) - 1
    return np.clip(np.round(raw), 0, hi).astype(int)


def denorm_to_class(arr_norm, means, stds, ftypes, nclasses):
    """Denormalise a z-scored GTSQ array and round to integer class labels (per-feature range)."""
    raw = np.asarray(arr_norm) * stds + means
    cls = np.zeros(raw.shape, dtype=int)
    for g in range(raw.shape[1]):
        cls[:, g] = round_to_class(raw[:, g], ftypes[g], nclasses[g])
    return cls


def decode_feature(lg, ftype):
    """Decode one feature's logit block to a class vector: binary -> logit>0, ordinal -> argmax."""
    return (lg[:, 0] > 0).astype(int) if str(ftype) == "binary" else lg.argmax(1)


def macro_f1(P, T, nclasses, mask=None):
    """Mean per-feature macro-F1 (labels=range(nclass)). `mask` (N,G bool) restricts cells per feature."""
    G = P.shape[1]
    fs = []
    for g in range(G):
        m = np.ones(len(P), bool) if mask is None else mask[:, g]
        if m.sum() < 1:
            continue
        fs.append(f1_score(T[m, g], np.clip(P[m, g], 0, int(nclasses[g]) - 1),
                           average="macro", labels=list(range(int(nclasses[g]))), zero_division=0))
    return float(np.mean(fs)) if fs else float("nan")


def weighted_f1(P, T, nclasses, mask=None):
    """Mean per-feature WEIGHTED F1 -- per-class F1 weighted by that class's support, then averaged
    over features. This is the class-frequency weighting the group asked for.

    Deliberately NOT micro-F1: for a single-label multiclass problem micro-F1 == accuracy exactly
    (every cell contributes one TP or one FP+FN), so a micro column would duplicate `accuracy` and
    tell you nothing new. Weighted-F1 is the variant that actually carries class weights.
    Sits between macro-F1 (rare classes count equally) and accuracy (rare classes vanish)."""
    G = P.shape[1]
    fs = []
    for g in range(G):
        m = np.ones(len(P), bool) if mask is None else mask[:, g]
        if m.sum() < 1:
            continue
        fs.append(f1_score(T[m, g], np.clip(P[m, g], 0, int(nclasses[g]) - 1),
                           average="weighted", labels=list(range(int(nclasses[g]))),
                           zero_division=0))
    return float(np.mean(fs)) if fs else float("nan")


def detector_metrics(y_true, score, threshold=0.5):
    """Change-DETECTOR metrics on pooled cells: y_true = 1 where the cell changes, score = the
    model's change probability (the gate gamma, post-sigmoid).

    Why not AUROC alone (the group's question): AUROC is threshold-free ranking and, with only
    ~3.8% positives, it flatters -- a detector can rank well and still be unusable. The numbers
    that decide the false-alarm tax are AUPRC (average precision, whose baseline is the positive
    rate itself) and the F1/precision/recall at the operating point you would actually deploy.
    'f1_best' is the best achievable F1 over all thresholds -- the honest ceiling for the detector.

    Returns dict(auroc, auprc, prevalence, f1_best, thr_best, precision, recall, f1)."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_recall_curve, precision_score, recall_score)
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(score, dtype=float).ravel()
    out = {"prevalence": float(y.mean()), "n": int(y.size)}
    if y.min() == y.max():                       # degenerate: no positives (or no negatives)
        return {**out, "auroc": float("nan"), "auprc": float("nan"), "f1_best": float("nan"),
                "thr_best": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan")}
    out["auroc"] = float(roc_auc_score(y, s))    # constant score -> exactly 0.5 (persistence)
    out["auprc"] = float(average_precision_score(y, s))
    prec, rec, thr = precision_recall_curve(y, s)
    f1c = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    k = int(np.nanargmax(f1c))
    out["f1_best"] = float(f1c[k])
    out["thr_best"] = float(thr[k]) if k < len(thr) else 1.0
    p = (s >= threshold).astype(int)
    out["precision"] = float(precision_score(y, p, zero_division=0))
    out["recall"] = float(recall_score(y, p, zero_division=0))
    out["f1"] = float(f1_score(y, p, zero_division=0))
    return out


def four_metrics(P, I, T, Mout, nclasses):
    """The four headline metrics on the target-observed mask Mout, CELL-POOLED where they pool:
    (cell-acc, macro-F1, change-acc, F1_change). macro-F1 stays per-feature-averaged (its definition).
    P=prediction, I=day-D (for change), T=target, all (N,G) integer classes; Mout (N,G) bool."""
    G = P.shape[1]
    n_ok = n_all = c_ok = c_all = 0
    f1s, f1cs = [], []
    for g in range(G):
        m = Mout[:, g]
        if m.sum() < 1:
            continue
        t = T[m, g]; p = np.clip(P[m, g], 0, int(nclasses[g]) - 1); i = I[m, g]
        n_ok += int((p == t).sum()); n_all += int(m.sum())
        f1s.append(f1_score(t, p, average="macro", labels=list(range(int(nclasses[g]))), zero_division=0))
        ch = i != t
        if ch.sum():
            c_ok += int((p[ch] == t[ch]).sum()); c_all += int(ch.sum())
            f1cs.append(f1_score(t[ch], p[ch], average="macro",
                                 labels=list(range(int(nclasses[g]))), zero_division=0))
    return (n_ok / n_all if n_all else float("nan"),
            float(np.mean(f1s)) if f1s else float("nan"),
            c_ok / c_all if c_all else float("nan"),
            float(np.mean(f1cs)) if f1cs else float("nan"))


def per_feature_report(pred_cls, in_cls, out_cls, out_mask, ftypes, nclasses):
    """Full evaluation on integer-class arrays: the four aggregate metrics (cell-pooled where they
    pool, §four_metrics) PLUS Cohen's kappa and a per-feature breakdown. Single source used by the
    standalone baselines (05_baselines.evaluate) and the analysis scripts; 03_train.eval_gtsq (which
    additionally handles logit-decode, domains and continuous MSE) is locked to it by test_metric_parity.

    pred_cls/in_cls/out_cls/out_mask: (N, G); continuous features are skipped. Returns a dict with
    f1_macro, kappa, f1_change, accuracy (cell-pooled), acc_change (cell-pooled), per_feat."""
    G = pred_cls.shape[1]
    f1_list, kappa_list, f1_change_list, acc_list = [], [], [], []
    f1w_list = []
    ch_ok = ch_all = n_ok = n_all = 0
    per_feat = {}
    for g in range(G):
        if str(ftypes[g]) == 'continuous':
            continue
        m = out_mask[:, g] == 1
        if m.sum() < 1:
            continue
        nc = int(nclasses[g])
        labels = list(range(nc))
        t = out_cls[m, g]
        p = np.clip(pred_cls[m, g], 0, nc - 1)
        i = in_cls[m, g]
        f1 = float(f1_score(t, p, average='macro', labels=labels, zero_division=0))
        f1w = float(f1_score(t, p, average='weighted', labels=labels, zero_division=0))
        try:
            kappa = float(cohen_kappa_score(t, p, labels=labels))
        except Exception:
            kappa = 0.0
        acc = float((p == t).mean())
        n_ok += int((p == t).sum()); n_all += int(m.sum())
        ch = i != t
        n_ch = int(ch.sum())
        f1_ch = (float(f1_score(t[ch], p[ch], average='macro', labels=labels,
                                zero_division=0)) if n_ch >= 1 else float('nan'))
        acc_ch = float((p[ch] == t[ch]).mean()) if n_ch >= 1 else float('nan')
        if n_ch >= 1:
            ch_ok += int((p[ch] == t[ch]).sum()); ch_all += n_ch
        per_feat[g] = {'f1': f1, 'f1_weighted': f1w, 'kappa': kappa, 'f1_change': f1_ch,
                       'accuracy': acc, 'acc_change': acc_ch,
                       'n_changed': n_ch, 'n_total': int(m.sum())}
        f1_list.append(f1); kappa_list.append(kappa); acc_list.append(acc); f1w_list.append(f1w)
        if not np.isnan(f1_ch):
            f1_change_list.append(f1_ch)
    return {
        'f1_macro': float(np.mean(f1_list)) if f1_list else 0.0,
        'f1_weighted': float(np.mean(f1w_list)) if f1w_list else 0.0,
        'kappa': float(np.mean(kappa_list)) if kappa_list else 0.0,
        'f1_change': float(np.mean(f1_change_list)) if f1_change_list else 0.0,
        'accuracy': float(n_ok / n_all) if n_all else 0.0,
        'acc_change': float(ch_ok / ch_all) if ch_all else 0.0,
        'per_feat': per_feat,
    }
