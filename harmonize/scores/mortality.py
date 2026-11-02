"""Horizon-based mortality labels (7-day / 30-day) per ICU stay.

The Stage-3 HDF5 carries only `mortality_hours = los_hours if deceased else NaN` — i.e. the
*stay length* of patients who died, **not** a time-to-death. Thresholding that field therefore
does NOT give a horizon mortality (it gives "died AND had a stay shorter than H"). This module
builds the correct label from the Stage-1 cohort, where real death timestamps live, and keys it
by `stay_id` so a head can look it up for the stays in a window batch.

Definition: `died_within(H) = 1` iff death occurred within H days of ICU admission, else 0.
Patients discharged alive before H are counted as survivors (the standard convention).

Per-dataset mechanism and censoring:
  mimic_iii / mimic_iv / umm : real death timestamp (incl. post-discharge follow-up)
      -> exact:   (death_time - icu_intime) <= H days.
  eicu   : no post-discharge follow-up; in-hospital death + its timing is known
      -> (hospitaldischargestatus == 'expired') AND (hospitaldischargeoffset <= H days).
  hirid  : same shape as eICU -> (discharge_status == 'dead') AND (los_hours <= H*24).
For eICU/HiRID a patient discharged alive who dies at home before H is *missed*, so those two
are mildly right-censored and the rate is a slight UNDER-estimate. Report that caveat with the
numbers; do not silently pool them with the exact-timestamp datasets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# (death-timestamp column, admission-timestamp column) for the exact-timestamp datasets
_EXACT = {
    "mimic_iii": ("death_time", "intime", "DOD"),
    "mimic_iv":  ("deathtime", "intime", "dod"),
    "umm":       ("death_time", "icu_start", None),
}


def _stay_ids(coh: pd.DataFrame) -> pd.Series:
    """Cohort stay identifier (HiRID is 1 stay/patient, so patient_id is the stay key)."""
    for c in ("stay_id", "patientunitstayid", "icustay_id", "patient_id"):
        if c in coh.columns:
            return coh[c]
    raise KeyError(f"no stay-id column in cohort (have {list(coh.columns)[:10]})")


def load_mortality_horizon_map(dataset: str, days: float,
                               long_root: str | Path = "data/long") -> dict[int, float]:
    """Return {stay_id: 1.0 if died within `days` of ICU admission else 0.0}.

    Raises KeyError if the cohort lacks the columns needed for `dataset`.
    """
    coh = pd.read_parquet(Path(long_root) / f"{dataset}_cohort.parquet")
    sid = _stay_ids(coh).astype("int64", errors="ignore")

    if dataset in _EXACT:
        dcol, icol, alt = _EXACT[dataset]
        if dcol not in coh.columns and alt and alt in coh.columns:
            dcol = alt
        death = pd.to_datetime(coh[dcol], errors="coerce")
        adm = pd.to_datetime(coh[icol], errors="coerce")
        # fall back to the alternative death column wherever the primary is missing
        if alt and alt in coh.columns and dcol != alt:
            death = death.fillna(pd.to_datetime(coh[alt], errors="coerce"))
        delta_d = (death - adm).dt.total_seconds() / 86400.0
        y = ((delta_d >= 0) & (delta_d <= days)).astype(float)
    elif dataset == "eicu":
        expired = coh["hospitaldischargestatus"].astype(str).str.lower().eq("expired")
        off_d = pd.to_numeric(coh["hospitaldischargeoffset"], errors="coerce") / 1440.0
        y = (expired & (off_d <= days)).astype(float)
    elif dataset == "hirid":
        dead = coh["discharge_status"].astype(str).str.lower().isin(("dead", "expired"))
        los_d = pd.to_numeric(coh["los_hours"], errors="coerce") / 24.0
        y = (dead & (los_d <= days)).astype(float)
    else:
        raise KeyError(f"no horizon-mortality rule for dataset {dataset!r}")

    return {int(s): float(v) for s, v in zip(sid, y) if pd.notna(s)}


def is_censored(dataset: str) -> bool:
    """True where the label misses post-discharge deaths (eICU / HiRID) -> slight under-estimate."""
    return dataset in ("eicu", "hirid")
