#!/usr/bin/env python3
"""eICU daily diagnosis targets for the keep-gate companion (Paper 3).

Public analogue of the GTSQ organ/localisation questionnaire: forecast tomorrow's active
problem list from today's. Targets are the daily presence of each **level-2** diagnosis
category from eICU `diagnosis.diagnosisstring` (e.g. "pulmonary | respiratory failure",
"cardiovascular | shock / hypotension"), low-circularity because they are clinician-coded
rather than derived from labs.

Presence model: an active-problem list is cumulative — a category is "present on day D" if
any diagnosis in it was documented (entered) on day ≤ D. The override the gate must catch is
a NEW problem appearing. eICU offers no per-day resolution, so presence is monotone 0→1.

Output contract (consumed by 02c_preprocess_eicu.build_pairs), one row per stay-day:
    DataFrame ['subject_id', 'stay_id', 'icu_day', <category columns 0/1>]
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_targets import (CLINICAL_TYPES, DAY_HOURS, HORIZON_H,
                           _max_icu_day, find_repo_root)

EICU_DIR = "eicu-collaborative-research-database-2.0"
MIN_COVERAGE = 0.02            # keep level-2 categories present in ≥2 % of stays (→ 39 targets)


def _col(l12: str) -> str:
    """'pulmonary | respiratory failure' -> 'dx_pulmonary__respiratory_failure'."""
    s = l12.lower().replace(" | ", "__")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return "dx_" + s


def load_cohort(repo_root: Path) -> pd.DataFrame:
    """eICU ICU cohort: stay_id (== patientunitstayid) -> patient_id, los_hours."""
    coh = pd.read_parquet(repo_root / "data" / "long" / "eicu_cohort.parquet",
                          columns=["patient_id", "stay_id", "los_hours"])
    return coh.rename(columns={"patient_id": "subject_id"})


def _stay_day_skeleton(coh: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    rows = []
    for sub, stay, los in coh[["subject_id", "stay_id", "los_hours"]].itertuples(index=False):
        for d in range(_max_icu_day(los, horizon_h) + 1):
            rows.append((sub, stay, d))
    return pd.DataFrame(rows, columns=["subject_id", "stay_id", "icu_day"])


def load_eicu_dx_targets(repo_root: Path, horizon_h: int = HORIZON_H,
                         min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """Daily cumulative presence of each level-2 diagnosis category (binary)."""
    coh = load_cohort(repo_root)
    dx = pd.read_csv(repo_root / "raw_data" / EICU_DIR / "diagnosis.csv",
                     usecols=["patientunitstayid", "diagnosisoffset", "diagnosisstring"])
    dx = dx.rename(columns={"patientunitstayid": "stay_id"})
    dx = dx[dx["stay_id"].isin(set(coh["stay_id"]))]

    parts = dx["diagnosisstring"].str.split("|", n=2, expand=True)
    dx["l12"] = (parts[0].str.strip() + " | " + parts[1].str.strip())
    dx = dx.dropna(subset=["l12"])
    dx["doc_day"] = np.floor(dx["diagnosisoffset"] / (60 * DAY_HOURS)).clip(lower=0).astype(int)

    n_stays = coh["stay_id"].nunique()
    cov = dx.groupby("l12")["stay_id"].nunique() / n_stays
    cats = sorted(cov[cov >= min_coverage].index)          # deterministic feature order
    colmap = {c: _col(c) for c in cats}
    dx = dx[dx["l12"].isin(cats)]

    # earliest documentation day per (stay, category); present from that day onward
    first_day = (dx.groupby(["stay_id", "l12"])["doc_day"].min()
                   .unstack("l12"))                        # stay_id × category, NaN if absent
    first_day = first_day.reindex(columns=cats)

    skel = _stay_day_skeleton(coh, horizon_h)
    m = skel.merge(first_day, on="stay_id", how="left")
    day = m["icu_day"].to_numpy()[:, None]
    fd = m[cats].to_numpy()                                # (N, C) first-doc day or NaN
    with np.errstate(invalid="ignore"):
        present = (day >= fd)                              # NaN comparison → False
    out = skel.copy()
    for j, c in enumerate(cats):
        out[colmap[c]] = present[:, j].astype(np.float32)
    return out


def build_clinical_windows(pairs, repo_root: Path, seq_len: int = 24,
                           horizon_h: int = HORIZON_H, target_day_window: bool = False):
    """Per-pair clinical window from eicu_long (day D, or day D+1 if target_day_window).

    Mirrors mimic_targets.build_clinical_windows: equi-join hourly vitals/labs/vent/meds on
    (stay_id, icu_day) == (stay_id, day_in[+1]). Returns (X, variables), X (N, V, seq_len, 2).
    """
    day_shift = 1 if target_day_window else 0
    clin = pd.read_parquet(
        repo_root / "data" / "long" / "eicu_long.parquet",
        columns=["stay_id", "hour", "variable", "value", "measurement_type"],
        filters=[("measurement_type", "in", list(CLINICAL_TYPES))])
    clin = clin[clin["hour"] < horizon_h]
    stays = {int(p["stay_id"]) for p in pairs}
    clin = clin[clin["stay_id"].isin(stays)]
    clin = clin.dropna(subset=["value"])   # mask=1 must imply a finite value (see mimic_targets fix:
                                           # .first() keeps NaN rows -> poisons norm-stats -> value zeroed)
    variables = sorted(clin["variable"].unique())
    var_to_idx = {v: i for i, v in enumerate(variables)}
    V = len(variables); N = len(pairs)

    clin["icu_day"] = np.floor(clin["hour"] / DAY_HOURS).astype(int)
    clin["hour_bin"] = (np.floor(clin["hour"]).astype(int)
                        - DAY_HOURS * clin["icu_day"]).clip(0, seq_len - 1)
    clin["var_idx"] = clin["variable"].map(var_to_idx).astype(int)

    pdf = pd.DataFrame({"pair_idx": np.arange(N, dtype=int),
                        "stay_id": [int(p["stay_id"]) for p in pairs],
                        "anchor_day": [int(p["day_in"]) + day_shift for p in pairs]})
    m = clin[["stay_id", "icu_day", "hour_bin", "var_idx", "value"]].merge(
        pdf, left_on=["stay_id", "icu_day"], right_on=["stay_id", "anchor_day"], how="inner")
    agg = (m.groupby(["pair_idx", "var_idx", "hour_bin"])["value"].first().reset_index())

    X = np.zeros((N, V, seq_len, 2), dtype=np.float32)
    pi = agg["pair_idx"].to_numpy(); vi = agg["var_idx"].to_numpy()
    hb = agg["hour_bin"].to_numpy()
    X[pi, vi, hb, 0] = agg["value"].to_numpy(dtype=np.float32)
    X[pi, vi, hb, 1] = 1.0
    return X, variables