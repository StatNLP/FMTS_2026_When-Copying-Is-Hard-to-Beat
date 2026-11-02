#!/usr/bin/env python3
"""eICU daily diagnosis preprocessing for the keep-gate companion (Paper 3).

Public, low-circularity analogue of the GTSQ questionnaire: predict day D+1's active
problem list (level-2 diagnosis categories) from day D. Mirrors 02b_preprocess_mimic.py
(same day-pairing, patient-level split, normalisation and npz/norm_stats format, reused by
03_train.py / 07_aggregate.py) but draws targets + clinical window from eICU.

  python 02c_preprocess_eicu.py                       # score-only  -> eicu_dx_pred_noclin
  python 02c_preprocess_eicu.py --use_clinical        # +clinical   -> eicu_dx_pred_clin
"""

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split

from eicu_targets import build_clinical_windows, load_eicu_dx_targets
from mimic_targets import find_repo_root
from pathlib import Path

CLINICAL_CLIP = 10.0     # winsorise clinical z-scores (guard against lab/med outliers); cf. 02b


def build_pairs(df, target_cols, min_targets, K):
    """df: per stay-day target table → list of pair dicts (predict day D+1 from day D)."""
    pairs = []
    for stay_id, grp in df.sort_values(["stay_id", "icu_day"]).groupby("stay_id"):
        subject = grp["subject_id"].iloc[0]
        days = grp["icu_day"].values
        vals = grp[target_cols].values.astype(np.float32)
        day_to_row = {int(d): i for i, d in enumerate(days)}
        for i, d in enumerate(days):
            d = int(d)
            if (d + 1) not in day_to_row:
                continue
            out = vals[day_to_row[d + 1]].copy()
            out_mask = (~np.isnan(out)).astype(np.float32)
            np.nan_to_num(out, copy=False, nan=0.0)
            if (~np.isnan(vals[i])).sum() < min_targets or out_mask.sum() < min_targets:
                continue
            hist_v, hist_m = [], []
            for k in range(K - 1, -1, -1):
                if (d - k) in day_to_row:
                    v = vals[day_to_row[d - k]].copy()
                    m = (~np.isnan(v)).astype(np.float32)
                    np.nan_to_num(v, copy=False, nan=0.0)
                else:
                    v = np.zeros(len(target_cols), np.float32)
                    m = np.zeros(len(target_cols), np.float32)
                hist_v.append(v); hist_m.append(m)
            gin = hist_v[0] if K == 1 else np.stack(hist_v)
            gin_m = hist_m[0] if K == 1 else np.stack(hist_m)
            pairs.append(dict(subject=subject, stay_id=stay_id, day_in=d,
                              gin=gin, gin_m=gin_m, gout=out, gout_m=out_mask))
    return pairs


def main(args):
    repo_root = Path(args.repo_root) if args.repo_root else find_repo_root()
    df = load_eicu_dx_targets(repo_root, horizon_h=args.horizon_h, min_coverage=args.min_coverage)
    target_cols = sorted(c for c in df.columns if c.startswith("dx_"))
    ftypes = np.array(["binary"] * len(target_cols))
    nclasses = np.array([2] * len(target_cols), dtype=np.int32)
    G = len(target_cols); K = args.gtsq_history
    df["subject_id"] = df["subject_id"].astype(str)
    print(f"eICU diagnosis targets: {G} level-2 categories (≥{args.min_coverage:.0%} coverage)")

    pairs = build_pairs(df, target_cols, args.min_targets, K)
    print(f"Day-pairs: {len(pairs)}")
    if not pairs:
        raise SystemExit("No day-pairs — check load_eicu_dx_targets / icu_day binning.")
    for i, p in enumerate(pairs):
        p["idx"] = i

    subs = sorted({p["subject"] for p in pairs})
    tr_sub, te_sub = train_test_split(subs, test_size=args.test_size, random_state=args.random_state)
    tr_sub, va_sub = train_test_split(tr_sub, test_size=args.val_size, random_state=args.random_state)
    split_of = {s: "train" for s in tr_sub}
    split_of.update({s: "val" for s in va_sub}); split_of.update({s: "test" for s in te_sub})
    splits = {s: [p for p in pairs if split_of[p["subject"]] == s] for s in ("train", "val", "test")}
    for s in splits:
        print(f"  {s}: {len(splits[s])} pairs")

    # --- optional clinical windows (day D, or target day D+1 if --oracle_window) ---
    seq_len = args.seq_len
    use_clin = bool(args.use_clinical)
    clin_X, clin_vars = None, []
    clin_means = np.zeros(0, np.float32); clin_stds = np.ones(0, np.float32)
    if use_clin:
        clin_X, clin_vars = build_clinical_windows(
            pairs, repo_root, seq_len, int(args.horizon_h),
            target_day_window=bool(args.oracle_window))
        Vc = len(clin_vars)
        Xtr = clin_X[[p["idx"] for p in splits["train"]]]
        clin_means = np.zeros(Vc, np.float32); clin_stds = np.ones(Vc, np.float32)
        for v in range(Vc):
            obs = Xtr[:, v, :, 0][Xtr[:, v, :, 1] == 1]
            obs = obs[np.isfinite(obs)]            # defensive: never let a stray NaN/inf -> NaN stats
            if obs.size > 1:
                clin_means[v] = obs.mean(); clin_stds[v] = max(obs.std(), 1e-8)
        print(f"Clinical: V={Vc}, train coverage {Xtr[:, :, :, 1].mean():.4f}")

    # --- target normalisation on TRAIN observed only ---
    def stack(split, key):
        return np.array([p[key] for p in splits[split]], dtype=np.float32)

    gin_tr = stack("train", "gin"); ginm_tr = stack("train", "gin_m")
    means = np.zeros(G, np.float32); stds = np.ones(G, np.float32)
    for g in range(G):
        obs = (gin_tr[:, :, g][ginm_tr[:, :, g] == 1] if K > 1
               else gin_tr[:, g][ginm_tr[:, g] == 1])
        if obs.size > 1:
            means[g] = obs.mean(); stds[g] = max(obs.std(), 1e-8)

    os.makedirs(args.save_dir, exist_ok=True)
    for s in ("train", "val", "test"):
        if not splits[s]:
            continue
        gin = stack(s, "gin"); ginm = stack(s, "gin_m")
        gout = stack(s, "gout"); goutm = stack(s, "gout_m")
        gin_n, gout_n = gin.copy(), gout.copy()
        for g in range(G):
            if K > 1:
                for k in range(K):
                    m = ginm[:, k, g] == 1
                    gin_n[m, k, g] = (gin_n[m, k, g] - means[g]) / stds[g]
            else:
                m = ginm[:, g] == 1
                gin_n[m, g] = (gin_n[m, g] - means[g]) / stds[g]
            mo = goutm[:, g] == 1
            gout_n[mo, g] = (gout_n[mo, g] - means[g]) / stds[g]
        N = len(splits[s])
        if use_clin:
            idxs = [p["idx"] for p in splits[s]]
            Xs = clin_X[idxs].copy()
            for v in range(len(clin_vars)):
                mv = Xs[:, v, :, 1] == 1
                Xs[:, v, :, 0][mv] = (Xs[:, v, :, 0][mv] - clin_means[v]) / clin_stds[v]
            np.clip(Xs[:, :, :, 0], -CLINICAL_CLIP, CLINICAL_CLIP, out=Xs[:, :, :, 0])
            np.nan_to_num(Xs[:, :, :, 0], copy=False, nan=0.0)   # unobserved hours → 0 (mask flags)
            vals = Xs[:, :, :, 0].transpose(0, 2, 1); msks = Xs[:, :, :, 1].transpose(0, 2, 1)
            clinical = np.concatenate([vals, msks], axis=2).astype(np.float32)
        else:
            clinical = np.zeros((N, seq_len, 0), np.float32)
        np.savez_compressed(
            os.path.join(args.save_dir, f"{s}.npz"), clinical=clinical,
            gtsq_in=gin_n, gtsq_in_mask=ginm, gtsq_out=gout_n, gtsq_out_mask=goutm,
            enc_ids=np.array([p["stay_id"] for p in splits[s]]),
            dates=np.array([f'{p["stay_id"]}_d{p["day_in"]}' for p in splits[s]]))
        print(f"  saved {s}.npz ({N} samples)")

    np.savez(os.path.join(args.save_dir, "norm_stats.npz"),
             clinical_means=clin_means, clinical_stds=clin_stds,
             clinical_variables=np.array(clin_vars),
             gtsq_means=means, gtsq_stds=stds, gtsq_features=np.array(target_cols),
             gtsq_ftypes=ftypes, gtsq_nclasses=nclasses,
             seq_len=seq_len, gap_hours=0, gtsq_hour=0, gtsq_history=K)
    print(f"Done -> {args.save_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--gtsq_history", type=int, default=1)
    p.add_argument("--min_targets", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--min_coverage", type=float, default=0.02)
    p.add_argument("--use_clinical", action="store_true", default=False)
    p.add_argument("--oracle_window", action="store_true", default=False)
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--val_size", type=float, default=0.20)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--repo_root", type=str, default=None)
    p.add_argument("--horizon_h", type=int, default=168)
    a = p.parse_args()
    if a.oracle_window:
        a.use_clinical = True
    if a.save_dir is None:
        tag = "oracle" if a.oracle_window else "clin" if a.use_clinical else "noclin"
        a.save_dir = f"01_split_sets/eicu_dx_pred_{tag}"
    return a


if __name__ == "__main__":
    main(parse_args())