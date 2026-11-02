#!/usr/bin/env python3
"""MIMIC-IV daily-target loaders for the keep-gate companion experiment.

This module fills the two data hooks of 02b_preprocess_mimic.py
(load_daily_targets / load_clinical) by tapping the EXISTING dissertation
harmonisation pipeline rather than re-deriving anything from raw MIMIC:

  * SOFA subscores reuse harmonize/scores/sofa.py (Vincent 1996 cutoffs) on the
    harmonised long-format parquet (data/long/mimic_iv_long.parquet), which is
    already in physical units with the canonical variable names sofa.py expects.
  * Organ-support flags (vent_invasive / rrt) come from raw procedureevents
    (invasive-vent and dialysis itemids); vasopressor comes from the same long
    parquet (normalised pressor rates).

Everything is kept on the harmonised forecasting horizon (long parquet is capped
at HORIZON_H = 168 h -> ICU days 0..6), so the two targets share one day grid.

Output contract (consumed by 02b_preprocess_mimic.build_pairs):
    long DataFrame with columns ['subject_id', 'stay_id', 'icu_day', <targets>]
    one row per stay-day. Missing target -> NaN (masked downstream).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repo / data location  (auto-detected; override via --repo_root)
# ---------------------------------------------------------------------------

def find_repo_root(start: str | None = None) -> Path:
    """Walk up from this file until we find the dissertation root (the dir that
    contains both `harmonize/` and `data/long/`)."""
    here = Path(start or __file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "harmonize").is_dir() and (parent / "data" / "long").is_dir():
            return parent
    # Fallback: cluster workspace default.
    return Path(__file__).resolve().parents[1]


def _ensure_harmonize_importable(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZON_H = 168           # harmonised long parquet is capped at 7 ICU days
DAY_HOURS = 24

# RRT delivery procedures (procedureevents). Excludes catheter insertion (224270)
# and filter-change maintenance (225436) — those are not RRT delivery.
RRT_ITEMIDS = [225441, 225802, 225803, 225805, 225809, 225955]
INVASIVE_VENT_ITEMIDS = [225792]   # 225794 = non-invasive, deliberately excluded

# Vasopressors / inotropes present in the harmonised long parquet (ug/kg/min).
PRESSOR_VARS = ["norepinephrine", "epinephrine", "dopamine", "dobutamine", "milrinone"]

# Time-varying clinical signal for the +clinical probe (exclude demographic/outcome/static).
CLINICAL_TYPES = {"vital", "lab", "ventilation", "medication", "fluid"}

# SOFA inputs (canonical names == what harmonize/scores/sofa.py._pick expects),
# with the daily-worst aggregation direction for each.
#   lower-is-worse  -> daily min ;  higher-is-worse -> daily max ;  urine -> daily sum
SOFA_AGG = {
    "pao2":            "min",
    "fio2":            "max",   # higher FiO2 -> worse P/F
    "platelets":       "min",
    "bilirubin_total": "max",
    "mbp_invasive":    "min",
    "mbp_noninvasive": "min",
    "dopamine":        "max",
    "dobutamine":      "max",
    "epinephrine":     "max",
    "norepinephrine":  "max",
    "creatinine":      "max",
    "urine_output":    "sum",
    "gcs_eye":         "min",
    "gcs_motor":       "min",
    "gcs_verbal":      "min",
}

# Feature order handed to sofa_components (gcs_total is summed from the components).
SOFA_FEATURE_NAMES = [
    "pao2", "fio2", "platelets", "bilirubin_total",
    "mbp_invasive", "mbp_noninvasive",
    "dopamine", "dobutamine", "epinephrine", "norepinephrine",
    "creatinine", "urine_output", "gcs_total",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Which harmonised tables a target set is built from. The circularity targets were built for
# MIMIC-IV only and the filenames were hardcoded, which is how the paper came to describe them as
# MIMIC-III. Naming the dataset explicitly makes that impossible to get wrong again, and lets HiRID
# serve as a size-matched control for UMM (similar cohort size, different institution).
DATASETS = {
    "mimic_iv": dict(cohort="mimic_iv_cohort.parquet", long="mimic_iv_long.parquet",
                     intime="intime"),
    "mimic_iii": dict(cohort="mimic_iii_cohort.parquet", long="mimic_iii_long.parquet",
                      intime="intime"),
    "hirid": dict(cohort="hirid_cohort.parquet", long="hirid_long.parquet",
                  intime="admissiontime", stay_is_patient=True),
}


def load_cohort(repo_root: Path, dataset: str = "mimic_iv") -> pd.DataFrame:
    """ICU cohort: stay_id -> subject_id, intime, los_hours, for any harmonised dataset."""
    if dataset not in DATASETS:
        raise SystemExit(f"ERROR: unknown dataset {dataset!r}; known: {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    path = repo_root / "data" / "long" / spec["cohort"]
    coh = pd.read_parquet(path)
    ren = {"patient_id": "subject_id", spec["intime"]: "intime"}
    coh = coh.rename(columns={k: v for k, v in ren.items() if k in coh.columns})
    if "stay_id" not in coh.columns:
        if spec.get("stay_is_patient"):
            # harmonize/hirid.py: "stay_id == patient_id; hospital_admission_id = pd.NA
            # throughout." One stay per patient, so the key is derivable without touching the long
            # table -- which is unnecessary here and is the read that OOMs a login node.
            coh["stay_id"] = coh["subject_id"]
            print(f"[{dataset}] stay_id := subject_id ({len(coh):,} stays, one per patient)")
        else:
            # Recover the stay list from the long table. The cohort was renamed to subject_id
            # above, so rename the long table's key to match -- joining a renamed left frame
            # against an unrenamed right one raises KeyError: 'subject_id'.
            lg = (pd.read_parquet(repo_root / "data" / "long" / spec["long"],
                                  columns=["patient_id", "stay_id"])
                  .drop_duplicates()
                  .rename(columns={"patient_id": "subject_id"}))
            before = len(coh)
            coh = coh.merge(lg, on="subject_id", how="inner")
            print(f"[{dataset}] cohort {before:,} patients -> {len(coh):,} stays "
                  f"({coh['stay_id'].nunique():,} unique)")

    missing = [c for c in ("subject_id", "stay_id", "intime", "los_hours") if c not in coh.columns]
    if missing:
        raise SystemExit(f"ERROR: {dataset} cohort {path.name} lacks {missing}; "
                         f"has {sorted(coh.columns)[:12]}")
    coh["intime"] = pd.to_datetime(coh["intime"])
    return coh[["subject_id", "stay_id", "intime", "los_hours"]]


def _max_icu_day(los_hours: float, horizon_h: int) -> int:
    """Highest valid 0-based ICU day index for a stay within the horizon.

    Day d covers [24d, 24d+24); it exists iff the stay reaches into it."""
    span = min(float(los_hours), float(horizon_h))
    return int(np.floor((span - 1e-9) / DAY_HOURS))


def _read_long(repo_root: Path, variables: list[str], dataset: str = "mimic_iv") -> pd.DataFrame:
    """Read the harmonised long parquet filtered to `variables`."""
    df = pd.read_parquet(
        repo_root / "data" / "long" / DATASETS[dataset]["long"],
        columns=["stay_id", "hour", "variable", "value"],
        filters=[("variable", "in", variables)],
    )
    df = df[df["hour"] < HORIZON_H]            # safety: stay on the 7-day grid
    df["icu_day"] = np.floor(df["hour"] / DAY_HOURS).astype(int)
    return df


# ---------------------------------------------------------------------------
# SOFA target
# ---------------------------------------------------------------------------

def load_sofa_targets(repo_root: Path, horizon_h: int = HORIZON_H, dataset: str = "mimic_iv") -> pd.DataFrame:
    """Six daily SOFA subscores (ordinal 0-4) per stay-day (daily-worst)."""
    _ensure_harmonize_importable(repo_root)
    from harmonize.scores.sofa import sofa_components

    coh = load_cohort(repo_root, dataset)
    df = _read_long(repo_root, list(SOFA_AGG.keys()), dataset)
    df = df[df["icu_day"] <= _max_icu_day(horizon_h, horizon_h)]

    # Daily-worst physical value per (stay, day, variable) using per-var direction.
    by_dir: dict[str, list[str]] = {"min": [], "max": [], "sum": []}
    for var, how in SOFA_AGG.items():
        by_dir[how].append(var)

    parts = []
    for how, vlist in by_dir.items():
        sub = df[df["variable"].isin(vlist)]
        if sub.empty:
            continue
        agg = (sub.groupby(["stay_id", "icu_day", "variable"])["value"]
                  .agg(how).reset_index())
        parts.append(agg)
    long_daily = pd.concat(parts, ignore_index=True)

    wide = long_daily.pivot_table(index=["stay_id", "icu_day"],
                                  columns="variable", values="value").reset_index()

    # GCS total = sum of the three component daily-worsts; NaN if any missing.
    for c in ("gcs_eye", "gcs_motor", "gcs_verbal"):
        if c not in wide.columns:
            wide[c] = np.nan
    wide["gcs_total"] = wide[["gcs_eye", "gcs_motor", "gcs_verbal"]].sum(
        axis=1, min_count=3)

    for name in SOFA_FEATURE_NAMES:                # ensure all inputs exist
        if name not in wide.columns:
            wide[name] = np.nan

    x = wide[SOFA_FEATURE_NAMES].to_numpy(dtype=np.float32)   # (n_days, F)
    # (n_days, F): urine_output is already a 24 h total, so it must not be summed again.
    comp = sofa_components(x, SOFA_FEATURE_NAMES, urine_is_24h_total=True)

    out = wide[["stay_id", "icu_day"]].copy()
    out["sofa_respiration"]    = comp.respiration
    out["sofa_coagulation"]    = comp.coagulation
    out["sofa_liver"]          = comp.liver
    out["sofa_cardiovascular"] = comp.cardiovascular
    out["sofa_cns"]            = comp.cns
    out["sofa_renal"]          = comp.renal

    sub_cols = ["sofa_respiration", "sofa_coagulation", "sofa_liver",
                "sofa_cardiovascular", "sofa_cns", "sofa_renal"]
    out = out.dropna(subset=sub_cols, how="all")             # keep days with >=1 subscore

    out = out.merge(coh[["stay_id", "subject_id"]], on="stay_id", how="left")
    return out[["subject_id", "stay_id", "icu_day", *sub_cols]]


# ---------------------------------------------------------------------------
# Support target
# ---------------------------------------------------------------------------

def _stay_day_skeleton(coh: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """One row per valid (subject, stay, icu_day) within the horizon."""
    rows = []
    for sub, stay, los in coh[["subject_id", "stay_id", "los_hours"]].itertuples(index=False):
        for d in range(_max_icu_day(los, horizon_h) + 1):
            rows.append((sub, stay, d))
    return pd.DataFrame(rows, columns=["subject_id", "stay_id", "icu_day"])


def _procedure_days(repo_root: Path, coh: pd.DataFrame, itemids: list[int],
                    horizon_h: int) -> pd.DataFrame:
    """(stay_id, icu_day) pairs covered by any procedure interval in `itemids`."""
    pe = pd.read_parquet(
        repo_root / "raw_data" / "physionet.org" / "files" / "mimiciv" / "3.1"
        / "icu" / "procedureevents.parquet",
        columns=["stay_id", "starttime", "endtime", "itemid"],
        filters=[("itemid", "in", itemids)],
    )
    pe = pe.merge(coh[["stay_id", "intime", "los_hours"]], on="stay_id", how="inner")
    pe = pe.dropna(subset=["starttime", "endtime", "intime"])
    if pe.empty:
        return pd.DataFrame(columns=["stay_id", "icu_day"])

    start_h = (pd.to_datetime(pe["starttime"]) - pe["intime"]).dt.total_seconds() / 3600.0
    end_h   = (pd.to_datetime(pe["endtime"])   - pe["intime"]).dt.total_seconds() / 3600.0
    pe["start_day"] = np.floor(start_h / DAY_HOURS).astype(int).clip(lower=0)
    pe["end_day"]   = np.floor(end_h / DAY_HOURS).astype(int)
    pe["max_day"]   = pe["los_hours"].apply(lambda l: _max_icu_day(l, horizon_h))
    pe["end_day"]   = np.minimum(pe["end_day"], pe["max_day"])
    pe = pe[pe["end_day"] >= pe["start_day"]]

    pe["icu_day"] = pe.apply(
        lambda r: list(range(int(r["start_day"]), int(r["end_day"]) + 1)), axis=1)
    out = pe.explode("icu_day")[["stay_id", "icu_day"]].dropna()
    out["icu_day"] = out["icu_day"].astype(int)
    return out.drop_duplicates()


def load_support_targets(repo_root: Path, horizon_h: int = HORIZON_H, dataset: str = "mimic_iv") -> pd.DataFrame:
    """Daily organ-support flags: vent_invasive, vasopressor, rrt (binary 0/1)."""
    coh = load_cohort(repo_root, dataset)
    skel = _stay_day_skeleton(coh, horizon_h)

    # vasopressor: any pressor record with value>0 that day (long parquet)
    pres = _read_long(repo_root, PRESSOR_VARS, dataset)
    pres = pres[pres["value"] > 0]
    vaso_days = pres[["stay_id", "icu_day"]].drop_duplicates()
    vaso_days["vasopressor"] = 1.0

    # vent_invasive / rrt: from raw procedureevents intervals
    vent_days = _procedure_days(repo_root, coh, INVASIVE_VENT_ITEMIDS, horizon_h)
    vent_days["vent_invasive"] = 1.0
    rrt_days = _procedure_days(repo_root, coh, RRT_ITEMIDS, horizon_h)
    rrt_days["rrt"] = 1.0

    out = skel.merge(vent_days, on=["stay_id", "icu_day"], how="left")
    out = out.merge(vaso_days, on=["stay_id", "icu_day"], how="left")
    out = out.merge(rrt_days, on=["stay_id", "icu_day"], how="left")
    for c in ("vent_invasive", "vasopressor", "rrt"):
        out[c] = out[c].fillna(0.0)            # absence of evidence = not on support

    return out[["subject_id", "stay_id", "icu_day",
                "vent_invasive", "vasopressor", "rrt"]]


# ---------------------------------------------------------------------------
# AKI / KDIGO target
# ---------------------------------------------------------------------------

def load_aki_targets(repo_root: Path, horizon_h: int = HORIZON_H, dataset: str = "mimic_iv") -> pd.DataFrame:
    """Daily KDIGO AKI stage (ordinal 0–3) + any-AKI flag (binary) per stay-day.

    Creatinine (daily max, mg/dL) vs a per-stay baseline (lowest creatinine in the stay),
    a 48 h absolute-rise criterion, RRT → stage 3, and an optional urine criterion
    (daily total ÷ weight ÷ 24). A consensus override target like Sepsis-3."""
    _ensure_harmonize_importable(repo_root)
    from harmonize.scores.kdigo import kdigo_stage

    coh = load_cohort(repo_root, dataset)
    df = _read_long(repo_root, ["creatinine", "urine_output", "weight", "weight_admission"], dataset)
    df = df[df["icu_day"] <= _max_icu_day(horizon_h, horizon_h)]

    creat = (df[df["variable"] == "creatinine"].groupby(["stay_id", "icu_day"])["value"]
             .max().rename("creatinine").reset_index())
    urine = (df[df["variable"] == "urine_output"].groupby(["stay_id", "icu_day"])
             .agg(urine_24h=("value", "sum"), urine_hours=("hour", "nunique")).reset_index())
    wt = (df[df["variable"].isin(["weight", "weight_admission"])].groupby("stay_id")["value"]
          .median().rename("weight").reset_index())

    g = (creat.merge(urine, on=["stay_id", "icu_day"], how="outer")
              .merge(wt, on="stay_id", how="left")
              .sort_values(["stay_id", "icu_day"]))

    g["baseline"] = g.groupby("stay_id")["creatinine"].transform("min")   # KDIGO reference proxy
    g["rise_48h"] = g["creatinine"] - g.groupby("stay_id")["creatinine"].transform(
        lambda s: s.rolling(window=3, min_periods=1).min())               # day d−2..d minimum
    g["urine_ml_kg_h"] = np.where(g["weight"] > 0, g["urine_24h"] / g["weight"] / g["urine_hours"].clip(lower=1), np.nan)
    g["urine_ml_kg_h"] = g["urine_ml_kg_h"].where(g["urine_hours"] >= 6, np.nan)   # ≥6h coverage

    rrt_days = _procedure_days(repo_root, coh, RRT_ITEMIDS, horizon_h)
    rrt_days["on_rrt"] = 1.0
    g = g.merge(rrt_days, on=["stay_id", "icu_day"], how="left")
    g["on_rrt"] = g["on_rrt"].fillna(0.0)

    # Full KDIGO: creatinine + RRT + urine. The urine rate is over the hours actually charted
    # with a ≥6 h coverage gate (above), so partial charting no longer deflates it; it adds
    # genuine oliguric AKI (any-AKI 0.21 → 0.24) without the earlier stage-3 over-call.
    stage = kdigo_stage(g["creatinine"].to_numpy(), g["baseline"].to_numpy(),
                        creat_rise_48h=g["rise_48h"].to_numpy(),
                        urine_ml_kg_h=g["urine_ml_kg_h"].to_numpy(),
                        on_rrt=g["on_rrt"].to_numpy())
    g["aki_stage"] = stage
    g["aki_any"] = np.where(np.isnan(stage), np.nan, (stage >= 1).astype(np.float32))
    g = g.dropna(subset=["aki_stage"])
    g = g.merge(coh[["stay_id", "subject_id"]], on="stay_id", how="left")
    return g[["subject_id", "stay_id", "icu_day", "aki_stage", "aki_any"]]


# ---------------------------------------------------------------------------
# Clinical window (+clinical probe)
# ---------------------------------------------------------------------------

def build_clinical_windows(pairs, repo_root: Path, seq_len: int = 24,
                           horizon_h: int = HORIZON_H, target_day_window: bool = False):
    """Per-pair clinical window of hourly clinical values (vitals/labs/vent/meds).

    Default (target_day_window=False): the INPUT day's window (icu_day == day_in) —
    day D's 24 h, strictly before day D+1, so no t+1 leakage. This is the honest
    +clinical probe.

    Oracle (target_day_window=True): the TARGET day's window (icu_day == day_in + 1) —
    day D+1's 24 h. This is the exact data from which the day-D+1 daily-worst SOFA
    subscores are computed, so it is a deliberate leaky upper bound: a capable model
    should reconstruct SOFA ~perfectly. Use it only to bound circularity, never as a
    forecasting result.

    Equi-join on (stay_id, icu_day) == (stay_id, anchor_day), so each clinical hour maps
    to at most one pair (no cartesian blow-up). Returns (X, variables): X is
    (N, V, seq_len, 2) [value, mask] aligned to the order of `pairs`; seq_len should be 24.
    """
    day_shift = 1 if target_day_window else 0
    clin = pd.read_parquet(
        repo_root / "data" / "long" / "mimic_iv_long.parquet",
        columns=["stay_id", "hour", "variable", "value", "measurement_type"],
        filters=[("measurement_type", "in", list(CLINICAL_TYPES))],
    )
    clin = clin[clin["hour"] < horizon_h]
    # Drop NaN-valued measurement rows BEFORE aggregation: otherwise `.first()` (unlike the
    # label's groupby-max, which skips NaN) keeps a NaN value while still setting mask=1, which
    # poisons the per-variable normalisation (obs.mean()->NaN) and zeroes the value channel for
    # sparse labs. mask=1 must imply a finite value, consistent with the SOFA-label pipeline.
    clin = clin.dropna(subset=["value"])
    variables = sorted(clin["variable"].unique())
    var_to_idx = {v: i for i, v in enumerate(variables)}
    V = len(variables)
    N = len(pairs)

    clin["icu_day"] = np.floor(clin["hour"] / DAY_HOURS).astype(int)
    clin["hour_bin"] = (np.floor(clin["hour"]).astype(int)
                        - DAY_HOURS * clin["icu_day"]).clip(0, seq_len - 1)
    clin["var_idx"] = clin["variable"].map(var_to_idx).astype(int)

    pdf = pd.DataFrame({"pair_idx": np.arange(N, dtype=int),
                        "stay_id": [p["stay_id"] for p in pairs],
                        "anchor_day": [int(p["day_in"]) + day_shift for p in pairs]})

    m = clin[["stay_id", "icu_day", "hour_bin", "var_idx", "value"]].merge(
        pdf, left_on=["stay_id", "icu_day"], right_on=["stay_id", "anchor_day"], how="inner")
    agg = (m.groupby(["pair_idx", "var_idx", "hour_bin"])["value"]
             .first().reset_index())

    X = np.zeros((N, V, seq_len, 2), dtype=np.float32)
    pi = agg["pair_idx"].to_numpy(); vi = agg["var_idx"].to_numpy()
    hb = agg["hour_bin"].to_numpy()
    X[pi, vi, hb, 0] = agg["value"].to_numpy(dtype=np.float32)
    X[pi, vi, hb, 1] = 1.0
    return X, variables


# ---------------------------------------------------------------------------
# Entry point used by 02b_preprocess_mimic.py
# ---------------------------------------------------------------------------

def load_daily_targets(target_set: str, args) -> pd.DataFrame:
    repo_root = Path(getattr(args, "repo_root", None) or find_repo_root())
    horizon_h = int(getattr(args, "horizon_h", HORIZON_H))
    dataset = str(getattr(args, "dataset", "mimic_iv"))
    if target_set == "sofa":
        return load_sofa_targets(repo_root, horizon_h, dataset)
    if target_set == "support":
        return load_support_targets(repo_root, horizon_h, dataset)
    if target_set == "aki":
        return load_aki_targets(repo_root, horizon_h, dataset)
    raise ValueError(f"Unknown target_set: {target_set!r}")
