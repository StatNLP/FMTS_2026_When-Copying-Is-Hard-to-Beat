#!/usr/bin/env python3
"""MIMIC-IV preprocessing for persistence-dominated next-day multi-target prediction.

Companion experiment to the GTSQ keep-gate work. Produces splits in the SAME npz
format as 02_preprocess.py, so 03_train.py and 05_baselines.py run unchanged.

Two target sets (--target):
  sofa     6 daily SOFA subscores (ordinal 0-4) — pure persistence-dominated forecast.
           A SOFA increase >=2 is the Sepsis-3 acute-deterioration event = the rare,
           high-value "override" the keep gate is meant to catch.
  support  daily organ-support status (binary/ordinal): invasive ventilation,
           vasopressor use, RRT, (optional) RASS sedation level. Therapy states are
           highly persistent day-to-day; changes = escalation / weaning.

Design mirrors GTSQ:
  * day-level targets, predict day D+1 from day D ("score-only") and optionally from a
    backward-anchored clinical window (no t+1 leakage).
  * patient-level split (by subject_id, NOT stay_id) to avoid cross-stay leakage.
  * normalisation on TRAIN observed values only.

IMPORTANT — drop-in compatibility:
  03_train.py hard-codes the npz key names 'gtsq_in', 'gtsq_out', 'gtsq_means', ...
  We keep those exact names even though the content is SOFA/support. Semantically they
  are the multi-target score; nothing else needs to change.

What you must fill on the MIMIC cluster (marked TODO):
  * load_daily_targets()  — from mimic-derived tables (sofa / ventilation / rrt /
                            vasoactive_agent) or raw with the cutoffs provided below.
  * load_clinical()       — hourly vitals/labs long table (only for the +clinical probe).
Everything else (day-pairing, splitting, normalisation, npz layout) is complete.
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Target schemas  (name, ftype, nclass)
# ---------------------------------------------------------------------------
# SOFA: 6 ordinal subscores, 0-4 each (Vincent et al. 1996).
SOFA_TARGETS = [
    ("sofa_respiration",     "ordinal", 5),
    ("sofa_coagulation",     "ordinal", 5),
    ("sofa_liver",           "ordinal", 5),
    ("sofa_cardiovascular",  "ordinal", 5),
    ("sofa_cns",             "ordinal", 5),
    ("sofa_renal",           "ordinal", 5),
]

# Organ support: daily status. Keep binary unless you have graded levels.
SUPPORT_TARGETS = [
    ("vent_invasive",  "binary",  2),   # invasive mechanical ventilation that day
    ("vasopressor",    "binary",  2),   # any vasopressor/inotrope that day
    ("rrt",            "binary",  2),   # renal replacement therapy that day
    # optional graded items (uncomment if you extract them):
    # ("vent_status",  "ordinal", 4),   # none / hi-flow / non-invasive / invasive
    # ("rass",         "ordinal", 7),   # -5..+1 mapped to 0..6 (sedation level)
]

# AKI (KDIGO 2012): daily stage 0–3 (ordinal) + any-AKI flag. A consensus override target
# (creatinine/urine/RRT) analogous to Sepsis-3, with the SOFA-like deterministic circularity.
AKI_TARGETS = [
    ("aki_stage", "ordinal", 4),        # KDIGO 0/1/2/3
    ("aki_any",   "binary",  2),        # stage ≥ 1
]

DAY_ANCHOR_HOURS = 24   # one row per 24h of ICU stay (icu_day), measured from intime
CLINICAL_CLIP = 10.0    # winsorise clinical z-scores to ±10 (guards against med/lab outliers)


# ---------------------------------------------------------------------------
# SOFA subscore cutoffs (only needed if you compute SOFA from raw instead of the
# mimic-derived `sofa` table — strongly prefer the derived table).
# Reference: Vincent et al., Intensive Care Med 1996.
# ---------------------------------------------------------------------------
def sofa_respiration(pao2_fio2, on_vent):
    if pd.isna(pao2_fio2): return np.nan
    if pao2_fio2 >= 400: return 0
    if pao2_fio2 >= 300: return 1
    if pao2_fio2 >= 200: return 2
    if pao2_fio2 >= 100: return 3 if on_vent else 2
    return 4 if on_vent else 2

def sofa_coagulation(platelets):  # 10^3/uL
    if pd.isna(platelets): return np.nan
    return 0 if platelets >= 150 else 1 if platelets >= 100 else 2 if platelets >= 50 else 3 if platelets >= 20 else 4

def sofa_liver(bilirubin):  # mg/dL
    if pd.isna(bilirubin): return np.nan
    return 0 if bilirubin < 1.2 else 1 if bilirubin < 2.0 else 2 if bilirubin < 6.0 else 3 if bilirubin < 12.0 else 4

def sofa_cns(gcs):
    if pd.isna(gcs): return np.nan
    return 0 if gcs >= 15 else 1 if gcs >= 13 else 2 if gcs >= 10 else 3 if gcs >= 6 else 4

def sofa_renal(creatinine, urine_ml_day=None):  # mg/dL ; urine mL/24h
    s = np.nan
    if not pd.isna(creatinine):
        s = 0 if creatinine < 1.2 else 1 if creatinine < 2.0 else 2 if creatinine < 3.5 else 3 if creatinine < 5.0 else 4
    if urine_ml_day is not None and not pd.isna(urine_ml_day):
        su = 3 if urine_ml_day < 500 else 0
        su = 4 if urine_ml_day < 200 else su
        s = max(s, su) if not pd.isna(s) else su
    return s

# cardiovascular needs MAP + vasopressor doses; prefer the derived sofa table.


# ---------------------------------------------------------------------------
# MIMIC loaders  — FILL THESE on the cluster
# ---------------------------------------------------------------------------
def load_daily_targets(target_set, args):
    """Return long DataFrame: columns ['subject_id','stay_id','icu_day', <target names>].

    icu_day = floor((charttime - intime) / DAY_ANCHOR_HOURS). One row per stay-day.
    Missing target -> NaN (handled downstream as mask=0).

    Implemented on top of the existing dissertation harmonisation pipeline
    (see mimic_targets.py): SOFA subscores reuse harmonize/scores/sofa.py on the
    harmonised long parquet; support flags (vent_invasive/rrt) come from raw
    procedureevents, vasopressor from the long parquet. Stays on the harmonised
    7-day forecasting horizon (ICU days 0..6).
    """
    from mimic_targets import load_daily_targets as _load
    return _load(target_set, args)


def load_clinical(args):
    """OPTIONAL (only for --use_clinical). Long table of hourly vitals/labs:
       columns ['stay_id','charttime','variableName','value','intime'].
    Used to build the backward-anchored clinical window. Return None to skip.
    TODO: fill from chartevents/labevents (or a curated derived vitals table)."""
    return None


# ---------------------------------------------------------------------------
# Day-pair construction (mirrors GTSQ: predict day D+1 from day D)
# ---------------------------------------------------------------------------
def build_pairs(df, target_cols, min_targets, K):
    """df: per stay-day target table. Returns list of pair dicts."""
    pairs = []
    for stay_id, grp in df.sort_values(["stay_id", "icu_day"]).groupby("stay_id"):
        subject = grp["subject_id"].iloc[0]
        days = grp["icu_day"].values
        vals = grp[target_cols].values.astype(np.float32)   # (n_days, G)
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
            # K days of history ending at day D (chronological: oldest .. D)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    target_cols_info = {"sofa": SOFA_TARGETS, "support": SUPPORT_TARGETS, "aki": AKI_TARGETS}[args.target]
    target_cols = [t[0] for t in target_cols_info]
    ftypes = np.array([t[1] for t in target_cols_info])
    nclasses = np.array([t[2] for t in target_cols_info], dtype=np.int32)
    G = len(target_cols)
    K = args.gtsq_history

    print(f"Target set: {args.target}  ({G} targets)  history K={K}")
    df = load_daily_targets(args.target, args)
    df["subject_id"] = df["subject_id"].astype(str)

    pairs = build_pairs(df, target_cols, args.min_targets, K)
    print(f"Day-pairs: {len(pairs)}")
    if not pairs:
        raise SystemExit("No day-pairs — check load_daily_targets / icu_day binning.")
    for i, p in enumerate(pairs):
        p["idx"] = i               # global index for aligning clinical windows

    # --- patient-level split (by subject_id) ---
    subs = sorted({p["subject"] for p in pairs})
    tr_sub, te_sub = train_test_split(subs, test_size=args.test_size,
                                      random_state=args.random_state)
    tr_sub, va_sub = train_test_split(tr_sub, test_size=args.val_size,
                                      random_state=args.random_state)
    split_of = {s: "train" for s in tr_sub}
    split_of.update({s: "val" for s in va_sub}); split_of.update({s: "test" for s in te_sub})
    splits = {s: [p for p in pairs if split_of[p["subject"]] == s]
              for s in ("train", "val", "test")}
    for s in splits:
        print(f"  {s}: {len(splits[s])} pairs")

    # --- (optional) clinical windows: per-pair day-D 24h window (no t+1 leakage) ---
    seq_len = args.seq_len
    use_clin = bool(args.use_clinical)
    clin_X = None
    clin_vars = []
    clin_means = np.zeros(0, np.float32)
    clin_stds = np.ones(0, np.float32)
    if use_clin:
        from pathlib import Path
        from mimic_targets import build_clinical_windows, find_repo_root
        repo_root = (Path(args.repo_root) if getattr(args, "repo_root", None)
                     else find_repo_root())
        clin_X, clin_vars = build_clinical_windows(
            pairs, repo_root, seq_len, int(getattr(args, "horizon_h", 168)),
            target_day_window=bool(getattr(args, "oracle_window", False)))
        Vc = len(clin_vars)
        tr_idx = [p["idx"] for p in splits["train"]]
        Xtr = clin_X[tr_idx]
        clin_means = np.zeros(Vc, np.float32)
        clin_stds = np.ones(Vc, np.float32)
        for v in range(Vc):
            obs = Xtr[:, v, :, 0][Xtr[:, v, :, 1] == 1]
            obs = obs[np.isfinite(obs)]            # defensive: never let a stray NaN/inf -> NaN stats
            if obs.size > 1:
                clin_means[v] = obs.mean()
                clin_stds[v] = max(obs.std(), 1e-8)
        print(f"Clinical: V={Vc} variables, train coverage "
              f"{Xtr[:, :, :, 1].mean():.4f}")

    # --- GTSQ (=score) normalisation on TRAIN observed only ---
    def stack(split, key):  # key in {'gin','gin_m','gout','gout_m'}
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
            Xs = clin_X[idxs].copy()                          # (N, V, seq_len, 2)
            for v in range(len(clin_vars)):
                mv = Xs[:, v, :, 1] == 1
                Xs[:, v, :, 0][mv] = (Xs[:, v, :, 0][mv] - clin_means[v]) / clin_stds[v]
            # Winsorise z-scores: harmonised meds/labs have extreme outliers (e.g.
            # propofol/heparin rates) that yield huge z-scores → GRU blow-up → NaN
            # (gate BCE then device-asserts). Clip to ±CLINICAL_CLIP keeps it finite.
            np.clip(Xs[:, :, :, 0], -CLINICAL_CLIP, CLINICAL_CLIP, out=Xs[:, :, :, 0])
            # Unobserved hours keep their NaN init (the mask-selective z-score above
            # only touches observed cells, and np.clip leaves NaN as NaN) — zero them
            # so the GRU sees finite values; the mask channel still flags them missing.
            np.nan_to_num(Xs[:, :, :, 0], copy=False, nan=0.0)
            vals = Xs[:, :, :, 0].transpose(0, 2, 1)          # (N, seq_len, V)
            msks = Xs[:, :, :, 1].transpose(0, 2, 1)
            clinical = np.concatenate([vals, msks], axis=2).astype(np.float32)
        else:
            clinical = np.zeros((N, seq_len, 0), np.float32)  # score-only
        np.savez_compressed(
            os.path.join(args.save_dir, f"{s}.npz"),
            clinical=clinical,
            gtsq_in=gin_n, gtsq_in_mask=ginm,
            gtsq_out=gout_n, gtsq_out_mask=goutm,
            enc_ids=np.array([p["stay_id"] for p in splits[s]]),
            dates=np.array([f'{p["stay_id"]}_d{p["day_in"]}' for p in splits[s]]),
        )
        print(f"  saved {s}.npz  ({N} samples)")

    np.savez(os.path.join(args.save_dir, "norm_stats.npz"),
             clinical_means=clin_means, clinical_stds=clin_stds,
             clinical_variables=np.array(clin_vars),
             gtsq_means=means, gtsq_stds=stds,
             gtsq_features=np.array(target_cols),
             gtsq_ftypes=ftypes, gtsq_nclasses=nclasses,
             seq_len=seq_len, gap_hours=args.gap_hours, gtsq_hour=0,
             gtsq_history=K)
    print(f"Done -> {args.save_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["sofa", "support", "aki"], default="sofa")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--gtsq_history", type=int, default=1)
    p.add_argument("--min_targets", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--gap_hours", type=int, default=6)
    p.add_argument("--use_clinical", action="store_true", default=False)
    p.add_argument("--oracle_window", action="store_true", default=False,
                   help="Use the TARGET day's (D+1) 24h clinical window instead of day D's "
                        "— a deliberate leaky upper bound (SOFA is computed from it). "
                        "Implies --use_clinical; split named mimic_<t>_pred_oracle.")
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--val_size", type=float, default=0.20)
    p.add_argument("--random_state", type=int, default=42)  # dissertation-wide seed convention
    p.add_argument("--repo_root", type=str, default=None,
                   help="Dissertation root holding data/long + harmonize (auto-detected if unset)")
    p.add_argument("--dataset", type=str, default="mimic_iv",
                   choices=["mimic_iv", "mimic_iii", "hirid"],
                   help="which harmonised database to build the target from. Splits are named "
                        "<dataset>_<target>_pred_<tag>. Defaults to mimic_iv, which is what the "
                        "existing circularity targets were built from -- the paper called them "
                        "MIMIC-III for a while because the filenames were hardcoded and nothing "
                        "carried the version.")
    p.add_argument("--horizon_h", type=int, default=168,
                   help="ICU horizon in hours (harmonised long parquet is capped at 168h)")
    a = p.parse_args()
    if a.oracle_window:
        a.use_clinical = True            # the oracle IS a clinical window (target day)
    if a.save_dir is None:
        tag = "oracle" if a.oracle_window else "clin" if a.use_clinical else "noclin"
        # name the split for the DATASET, not a generic "mimic" -- the version has to be
        # recoverable from the path or it gets lost, as it did for the circularity targets
        a.save_dir = f"01_split_sets/{a.dataset}_{a.target}_pred_{tag}"
    return a


if __name__ == "__main__":
    main(parse_args())
