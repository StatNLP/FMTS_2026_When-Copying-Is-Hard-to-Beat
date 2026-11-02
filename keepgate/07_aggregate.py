#!/usr/bin/env python3
"""Aggregate Keep-Gate MIMIC results into results_keepgate_mimic.csv.

Race-free: reads each run's own `02_models/<run>/config.json` (one writer per file)
instead of the concurrently-appended `results.csv`, so it is immune to interleaved
writes from parallel SLURM jobs. Also folds in the non-neural baselines from each
split's `baseline_results.json`, producing one unified per-target comparison table.

Outputs (in --out_dir, default the keepgate package root):
  results_keepgate_mimic.csv          one row per neural run (deduped, latest timestamp)
  results_keepgate_mimic_summary.csv  target × method → F1 / kappa / F1_change (mean±std)

Usage:
  python 07_aggregate.py
  python 07_aggregate.py --models_dir 02_models --splits_dir 01_split_sets
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

# Non-neural baselines first, then neural variants — fixed order for the paper table.
METHOD_ORDER = ["persistence", "markov", "hmm", "hmm_smooth", "logreg", "gbt", "twostage", "blup",
                "base", "skip_linear", "skip_copy", "skip_markov",
                "gate_linear", "gate_copy", "gate_markov"]
# Methods supplied by 05_baselines.py; the neural sweep's model_type=persistence run
# duplicates 'persistence', so it is dropped from the summary (kept in the per-run CSV).
BASELINE_METHODS = {"persistence", "markov", "logreg", "gbt", "twostage"}


def target_of(data_dir):
    d = str(data_dir)
    for t in ("support", "sofa", "aki"):
        if f"mimic_{t}" in d:
            return t
    if "gtsq_pred" in d:
        return "gtsq"
    if "eicu_dx" in d:
        return "eicu_dx"
    return "other"


SCOPE_TARGETS = {"mimic": {"support", "sofa", "aki"}, "gtsq": {"gtsq"},
                 "eicu": {"eicu_dx"},
                 "all": {"support", "sofa", "aki", "gtsq", "eicu_dx"}}


def variant_of(cfg):
    if str(cfg.get("model_type")) == "persistence":
        return "persistence"
    pp = str(cfg.get("persist_path", "linear"))
    if cfg.get("use_gate"):
        return f"gate_{pp}"
    if cfg.get("use_skip"):
        return f"skip_{pp}"
    return "base"


def collect_neural(models_dir, targets):
    rows = []
    for cfgpath in glob.glob(os.path.join(models_dir, "*", "config.json")):
        try:
            with open(cfgpath) as f:
                c = json.load(f)
        except Exception:
            continue
        run = os.path.basename(os.path.dirname(cfgpath))
        tgt = target_of(c.get("data_dir", run))
        if tgt not in targets:          # scope filter (mimic / gtsq / all)
            continue
        dd = str(c.get("data_dir", run))
        _hz = re.search(r"_hz(\d+)", dd); horizon = int(_hz.group(1)) if _hz else 1
        # Robust input label. MIMIC dirs read "..._pred_clin"; GTSQ dirs read
        # "..._gap6h_clin_complete" (no "pred_clin"), so match the "clin" token directly.
        # "_clin" never matches "_noclin" (the char before "clin" there is 'o', not '_').
        inp = ("oracle" if "_pred_oracle" in dd
               else "clin_complete" if "clin_complete" in dd
               else "clin_capped" if "clin_capped" in dd
               else "clin" if "_clin" in dd else "noclin")
        rows.append({
            "run": run,
            "timestamp": run[:11],       # YYMMDD_HHMM prefix (chronological sort)
            "target": tgt,
            "input": inp,                # noclin (score-only) vs clin (+clinical)
            "horizon": horizon,          # prediction horizon (days); 1=next-day, hz2/3/7
            "method": variant_of(c),
            "seed": c.get("seed"),
            "model_type": c.get("model_type"),
            "persist_path": c.get("persist_path"),
            "loss_balance": c.get("loss_balance"),
            # Future review-driven training axes (None = the canonical default). Adding them to
            # BOTH the row and the dedup identity (and to is_canonical below) is the lesson from
            # the horizon-collision bug: a new run-type must NOT collapse onto a headline cell.
            "ordinal_loss": c.get("ordinal_loss"),   # Tier B: ordinal-aware objective (corn/cumlink/ordce)
            "change_loss": c.get("change_loss"),     # Tier D: change-focused training (focal/reweight/twostage)
            # Class-balancing ablation: --no_class_weights = plain CE/BCE (no pos_weight/class_weight).
            "no_class_weights": bool(c.get("no_class_weights", False)),
            # Day-of-stay covariate: --use_day_of_stay adds the episode-relative ICU day as input.
            "use_day_of_stay": bool(c.get("use_day_of_stay", False)),
        "use_supervisor": str(c.get("use_supervisor", "none") or "none"),
            "change_detector": bool(c.get("change_detector", False)),  # detection-only runs (garbage F1)
            # ablation axes — part of the run identity so encoder/λ/depth/fusion/input-mode
            # variants are NOT collapsed onto the main-comparison buckets and deduped away.
            "gate_lambda": c.get("gate_lambda"),
            "gtsq_enc_layers": c.get("gtsq_enc_layers"),
            "fusion": c.get("fusion"),
            # NB: 03_train stores the flag as use_gtsq_input (--no_gtsq_input sets it False),
            # so read that, not "no_gtsq_input". clinical_only = predict from the clinical
            # window with no GTSQ-score history (run_name input_mode "onlyclin").
            "clinical_only": not bool(c.get("use_gtsq_input", True)),
            "shuffle_patients": bool(c.get("shuffle_patients")),
            "best_epoch": c.get("best_epoch"),
            "F1_val": c.get("F1_val"),
            "F1_test": c.get("F1_test"),
            "Kappa_test": c.get("Kappa_test"),
            "F1_change_test": c.get("F1_change_test"),
            # accuracy counterparts (03_train >= 2026-07); older runs lack them -> NaN
            "Acc_test": c.get("Acc_test"),
            "Acc_change_test": c.get("Acc_change_test"),
            "MSE_cont_test": c.get("MSE_cont_test"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # --- data-provenance corrections (one-time; keep the committed CSVs honest) -----------
    # (1) 5-seed policy. Seed 2024 was a 6th seed used only in the pre-June-2026 MIMIC batch;
    #     the canonical seed set is {42,123,456,789,1337}. Drop it so every cell is 5 seeds.
    df = df[df["seed"] != 2024]
    # (2) NaN-norm clinical-feature bug, fixed 2026-06-23 (commits 0d089c4 / a93024d / 60fa367):
    #     the clinical-window builder zeroed the value channel for sparse MIMIC/eICU labs, so
    #     +clinical (clin/oracle) runs recorded BEFORE 260623 have corrupted clinical features.
    #     Exclude those pre-fix clinical runs; only the corrected re-runs (base, gate_copy) feed
    #     the +clin/oracle cells. noclin runs use no clinical features and are unaffected; GTSQ's
    #     clinical builder is separate and was not touched, so this is scoped to the lab targets.
    _clin_bug_targets = {"support", "sofa", "aki", "eicu_dx"}
    _clin_inputs = {"clin", "clin_complete", "clin_capped", "oracle"}
    pre_nan = (df["target"].isin(_clin_bug_targets)
               & df["input"].isin(_clin_inputs)
               & (df["timestamp"].str.slice(0, 6) < "260623"))
    df = df[~pre_nan].reset_index(drop=True)
    # Dedup re-runs: keep the latest timestamp per FULL config identity (incl. ablation
    # axes), so a re-run replaces only its own twin — not a different ablation cell.
    ident = ["target", "input", "method", "seed", "model_type", "gate_lambda",
             "gtsq_enc_layers", "fusion", "clinical_only", "shuffle_patients", "horizon",
             "loss_balance", "ordinal_loss", "change_loss", "no_class_weights", "use_day_of_stay", "use_supervisor", "change_detector"]
    df = (df.sort_values("timestamp")
            .drop_duplicates(ident, keep="last")
            .reset_index(drop=True))
    return df


def collect_baselines(splits_dir, targets):
    src = []                                            # (target, baseline_results.json)
    if "support" in targets:
        src.append(("support", os.path.join(splits_dir, "mimic_support_pred_noclin", "baseline_results.json")))
    if "sofa" in targets:
        src.append(("sofa", os.path.join(splits_dir, "mimic_sofa_pred_noclin", "baseline_results.json")))
    if "gtsq" in targets:
        for p in glob.glob(os.path.join(splits_dir, "gtsq_pred_*_noclin", "baseline_results.json")):
            src.append(("gtsq", p))
    if "eicu_dx" in targets:
        src.append(("eicu_dx", os.path.join(splits_dir, "eicu_dx_pred_noclin", "baseline_results.json")))
    rows = []
    seen = set()                                    # (target, method) — original file always wins
    for t, p in src:
        # Tier C new baselines (HMM, smoothed-Markov) land in a sibling *_extra.json so the
        # committed baseline_results.json is never recomputed/overwritten; read both, original
        # first, and let _extra only ADD methods it does not already contain.
        for path in (p, p.replace("baseline_results.json", "baseline_results_extra.json")):
            if not os.path.exists(path):
                continue
            with open(path) as f:
                b = json.load(f)
            for method, r in b.get("results", {}).items():
                if (t, method) in seen:
                    continue
                seen.add((t, method))
                rows.append({"target": t, "method": method, "seed": "—",
                             "F1_test": r.get("f1_macro"), "Kappa_test": r.get("kappa"),
                             "F1_change_test": r.get("f1_change")})
    return pd.DataFrame(rows)


def is_canonical(df):
    """Main-comparison runs only: gru/persistence encoder, default gate-λ/depth/fusion,
    no input-mode/shuffle ablation. Keeps the summary table clean; ablation cells live
    in the per-run CSV (results_keepgate_<scope>.csv)."""
    return (df["model_type"].isin(["gru", "persistence"])
            & df["gate_lambda"].isin([np.nan, None, 0.5])
            & df["gtsq_enc_layers"].isin([np.nan, None, 2])
            & df["fusion"].isin([np.nan, None, "concat"])
            & ~df["clinical_only"].fillna(False)
            & ~df["shuffle_patients"].fillna(False)
            # canonical loss per dataset (GTSQ default 'lw', MIMIC sweep 'uniform'); the
            # other loss is kept as an ablation cell in the per-run CSV, not the summary.
            & (df["loss_balance"].astype(str)
               == df["target"].map({"gtsq": "lw"}).fillna("uniform"))
            & df["ordinal_loss"].isin([np.nan, None, "ce"])    # Tier B: only standard CE in the headline
            & df["change_loss"].isin([np.nan, None])           # Tier D: only standard training in the headline
            & ~df["no_class_weights"].fillna(False)            # class-balancing ablation stays out of the headline
            & (df["use_supervisor"].fillna("none") == "none")  # rater-identity covariate stays out of the canonical summary
        & ~df["use_day_of_stay"].fillna(False)             # day-of-stay covariate stays out of the headline
            & ~df["change_detector"].fillna(False)             # detection-only runs never enter the F1 summary
            & (df["horizon"].fillna(1) == 1))   # main summary = next-day only; hz2/3/7 are a separate sweep


def summarise(neural, baselines):
    parts = []
    if not neural.empty:
        neural = neural[~neural["method"].isin(BASELINE_METHODS) & is_canonical(neural)]
    if not neural.empty:
        g = (neural.groupby(["target", "input", "method"])
                   .agg(n=("seed", "nunique"),
                        F1_test=("F1_test", "mean"), F1_test_std=("F1_test", "std"),
                        Kappa_test=("Kappa_test", "mean"),
                        F1_change_test=("F1_change_test", "mean"),
                        Acc_test=("Acc_test", "mean"),
                        Acc_change_test=("Acc_change_test", "mean"))
                   .reset_index())
        g["source"] = "neural"
        parts.append(g)
    if not baselines.empty:
        b = baselines.copy()
        b["input"] = "noclin"            # baselines are score-only (gtsq-only)
        b["n"] = 1; b["F1_test_std"] = np.nan; b["source"] = "baseline"
        b["Acc_test"] = np.nan; b["Acc_change_test"] = np.nan
        parts.append(b[["target", "input", "method", "n", "F1_test", "F1_test_std",
                        "Kappa_test", "F1_change_test", "Acc_test", "Acc_change_test",
                        "source"]])
    if not parts:
        return pd.DataFrame()
    s = pd.concat(parts, ignore_index=True)
    s["__o"] = s["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    s["__i"] = s["input"].map({"noclin": 0, "clin": 1, "clin_capped": 2,
                               "clin_complete": 3}).fillna(9)
    return (s.sort_values(["target", "__i", "__o"])
             .drop(columns=["__o", "__i"]).reset_index(drop=True))


def main(args):
    targets = SCOPE_TARGETS[args.scope]
    neural = collect_neural(args.models_dir, targets)
    baselines = collect_baselines(args.splits_dir, targets)

    stem = f"results_keepgate_{args.scope}"          # _mimic / _gtsq / _all
    per_run = os.path.join(args.out_dir, f"{stem}.csv")
    if not neural.empty:
        neural.sort_values(["target", "input", "method", "seed"]).to_csv(
            per_run, index=False, float_format="%.6f")
    summary = summarise(neural, baselines)
    summ_path = os.path.join(args.out_dir, f"{stem}_summary.csv")
    if not summary.empty:
        summary.to_csv(summ_path, index=False, float_format="%.6f")

    # --- console report ---
    n_runs = 0 if neural.empty else len(neural)
    print(f"Neural runs: {n_runs}  ({per_run})")
    if not summary.empty:
        for tgt in summary["target"].unique():
          for inp in summary.loc[summary["target"] == tgt, "input"].unique():
            sub = summary[(summary["target"] == tgt) & (summary["input"] == inp)]
            if sub.empty:
                continue
            print(f"\n=== {tgt} / {inp} ===")
            print(f"  {'method':14s}{'n':>3s}{'F1_test':>10s}{'±std':>9s}{'kappa':>9s}{'F1_change':>11s}")
            for _, r in sub.iterrows():
                std = "" if pd.isna(r["F1_test_std"]) else f"{r['F1_test_std']:.4f}"
                f1 = "" if pd.isna(r["F1_test"]) else f"{r['F1_test']:.4f}"
                kp = "" if pd.isna(r["Kappa_test"]) else f"{r['Kappa_test']:.4f}"
                fc = "" if pd.isna(r["F1_change_test"]) else f"{r['F1_change_test']:.4f}"
                print(f"  {r['method']:14s}{int(r['n']):>3d}{f1:>10s}{std:>9s}{kp:>9s}{fc:>11s}")
        print(f"\nSaved summary -> {summ_path}")


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="02_models")
    p.add_argument("--splits_dir", default="01_split_sets")
    p.add_argument("--out_dir", default=here)
    p.add_argument("--scope", choices=["mimic", "gtsq", "eicu", "all"], default="mimic",
                   help="Which experiment's runs to aggregate (-> results_keepgate_<scope>*.csv)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
