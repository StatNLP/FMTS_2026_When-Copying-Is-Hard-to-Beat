"""
harmonize/common.py
Shared schema, utilities, and feature-catalog loader used by all dataset modules.
"""

from __future__ import annotations
import os

import logging

_log = logging.getLogger(__name__)
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Long-format schema
# ---------------------------------------------------------------------------

LONG_COLUMNS = [
    "patient_id",            # int  — subject_id / patient identifier
    "stay_id",               # int  — ICU-stay identifier (np.nan if unavailable)
    "hospital_admission_id", # int  — hospital admission id (np.nan if unavailable)
    "hour",                  # float — hours since ICU intime (native resolution)
    "variable",              # str  — harmonized feature key matching features.yaml
    "value",                 # float
    "unit",                  # str  — harmonized unit string
    "dataset",               # str  — e.g. "mimic_iv"
    "measurement_type",      # str  — vital | lab | ventilation | medication | fluid | static | derived | demographic
]

LONG_DTYPES: dict[str, Any] = {
    "patient_id":            "int64",
    "stay_id":               "Int64",   # nullable int
    "hospital_admission_id": "Int64",
    "hour":                  "float64",
    "variable":              "object",
    "value":                 "float64",
    "unit":                  "object",
    "dataset":               "object",
    "measurement_type":      "object",
}


def make_long_df(
    patient_id: pd.Series,
    stay_id: pd.Series | None,
    hospital_admission_id: pd.Series | None,
    hour: pd.Series,
    variable: str,
    value: pd.Series,
    unit: str,
    dataset: str,
    measurement_type: str,
) -> pd.DataFrame:
    """Construct a Long-Format DataFrame with the canonical schema."""
    n = len(patient_id)
    df = pd.DataFrame(
        {
            "patient_id": patient_id.values,
            "stay_id": stay_id.values if stay_id is not None else np.full(n, pd.NA, dtype=object),
            "hospital_admission_id": hospital_admission_id.values if hospital_admission_id is not None else np.full(n, pd.NA, dtype=object),
            "hour": hour.values,
            "variable": variable,
            "value": value.values,
            "unit": unit,
            "dataset": dataset,
            "measurement_type": measurement_type,
        }
    )
    return df.astype({k: v for k, v in LONG_DTYPES.items() if k in df.columns})


# ---------------------------------------------------------------------------
# Feature catalog
# ---------------------------------------------------------------------------

class FeatureCatalog:
    """Thin wrapper around a loaded features.yaml for structured access."""

    def __init__(self, catalog: dict[str, dict]):
        self._cat = catalog

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FeatureCatalog":
        with open(path, "r") as fh:
            cat = yaml.safe_load(fh)
        return cls(cat)

    def __contains__(self, key: str) -> bool:
        return key in self._cat

    def __getitem__(self, key: str) -> dict:
        return self._cat[key]

    def items(self):
        return self._cat.items()

    def keys(self):
        return self._cat.keys()

    def value_range(self, key: str) -> tuple[float, float] | None:
        vr = self._cat[key].get("value_range")
        if vr is None:
            return None
        return (float(vr[0]), float(vr[1]))

    def unit(self, key: str) -> str:
        return self._cat[key].get("unit", "")

    def measurement_type(self, key: str) -> str:
        return self._cat[key].get("type", "")

    def source_mapping(self, key: str, dataset: str) -> Any:
        return self._cat[key].get("source_mapping", {}).get(dataset)

    def sofa_relevant(self, key: str) -> bool:
        return bool(self._cat[key].get("sofa_relevant", False))

    def tier(self, key: str) -> str:
        return self._cat[key].get("tier", "")


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def fahrenheit_to_celsius(f: pd.Series) -> pd.Series:
    """Convert Fahrenheit to Celsius."""
    return (f - 32.0) * 5.0 / 9.0


def fio2_percent_to_fraction(v: pd.Series) -> pd.Series:
    """Convert FiO2 from percent (>1.0) to fraction (0.21–1.0)."""
    result = v.copy()
    mask = result > 1.0
    result.loc[mask] = result.loc[mask] / 100.0
    return result


def lbs_to_kg(lbs: pd.Series) -> pd.Series:
    return lbs * 0.453592


def mg_to_mmol_lactate(mg_dl: pd.Series) -> pd.Series:
    """Lactate mg/dL → mmol/L."""
    return mg_dl / 9.01


def hours_since_intime(charttime: pd.Series, intime: pd.Series) -> pd.Series:
    """Compute float hours between intime and charttime.

    Both series must contain timezone-naive datetime64 values (or compatible strings).
    """
    ct = pd.to_datetime(charttime)
    it = pd.to_datetime(intime)
    return (ct - it).dt.total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Outlier filter
# ---------------------------------------------------------------------------

def apply_outlier_filter(
    df: pd.DataFrame,
    value_col: str,
    value_range: tuple[float, float] | None,
) -> pd.DataFrame:
    """Set values outside [lo, hi] to NaN (in-place on copy).

    NaN convention: values outside plausible range are set to NaN,
    never imputed to median or dropped. Downstream Step 2 handles imputation.
    """
    if value_range is None:
        return df
    lo, hi = value_range
    df = df.copy()
    mask = (df[value_col] < lo) | (df[value_col] > hi)
    n_out = mask.sum()
    if n_out > 0:
        _log.debug("Outlier filter: %d values outside [%s, %s] set to NaN.", n_out, lo, hi)
    df.loc[mask, value_col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Interval expansion (for inputevents / procedureevents)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Training-run bookkeeping — checkpoint overwrite guard + central run manifest
# ---------------------------------------------------------------------------
# Run folders are named deterministically (by --run-name), so a direct re-run
# would silently overwrite a previous model. These two helpers make that safe
# and auditable WITHOUT changing the folder-naming scheme (sweeps' skip-existing
# and harvest-by-name stay intact):
#   * archive_existing_run() — move a previous completed run out of the way
#     (to checkpoints_archive/<name>__<finished_at>) before it is overwritten.
#   * append_run_manifest()  — append one audit row per finished run to a central
#     CSV (when it ran, where, git sha, SLURM job, best metric).

_MANIFEST_FIELDS = [
    "logged_at", "run_name", "output_dir", "model", "dataset",
    "started_at", "finished_at", "epochs_run", "best_metric", "best_val",
    "git_sha", "slurm_job",
]


def archive_existing_run(out_dir, archive_root="checkpoints_archive"):
    """If *out_dir* already holds a previous run (``best.pt`` present), MOVE the
    whole folder to ``archive_root/<name>__<finished_at>`` before it is
    overwritten, so a good model is never silently lost. Returns the archive
    Path, or None if there was nothing to archive. Only triggers when a
    checkpoint actually exists (fresh runs and crashed-before-best runs are
    untouched), so it never interferes with the sweep skip-existing guard."""
    import datetime as _dt
    import json as _json
    import shutil as _shutil

    out_dir = Path(out_dir)
    if not (out_dir / "best.pt").exists():
        return None
    stamp = None
    fm = out_dir / "final_metrics.json"
    if fm.exists():
        try:
            stamp = _json.loads(fm.read_text()).get("finished_at")
        except Exception:
            stamp = None
    if not stamp:
        stamp = _dt.datetime.fromtimestamp(
            (out_dir / "best.pt").stat().st_mtime).isoformat(timespec="seconds")
    safe = str(stamp).replace(":", "").replace("-", "").replace("T", "_")
    dest = Path(archive_root) / f"{out_dir.name}__{safe}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    base, i = dest, 0
    while dest.exists():
        i += 1
        dest = Path(f"{base}_{i}")
    _shutil.move(str(out_dir), str(dest))
    _log.warning("Existing run at %s found — archived to %s (NOT overwritten).",
                 out_dir, dest)
    return dest


def _git_sha() -> str:
    import subprocess as _sp
    try:
        return _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                stderr=_sp.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return ""


def append_run_manifest(run_name, out_dir, started_at="", finished_at="", *,
                        model="", dataset="", epochs_run="", best_metric="",
                        best_val="", manifest="checkpoints/_runs_manifest.csv"):
    """Append one audit row per finished run to a central CSV. Fixed schema (so
    columns never drift), flock-guarded append. Folder names stay unchanged; this
    is the at-a-glance "when / which / where" trail the deterministic names lack."""
    import csv as _csv
    import datetime as _dt
    import fcntl as _fcntl
    import os as _os

    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "logged_at":   _dt.datetime.now().isoformat(timespec="seconds"),
        "run_name":    run_name,
        "output_dir":  str(out_dir),
        "model":       model,
        "dataset":     dataset,
        "started_at":  started_at,
        "finished_at": finished_at,
        "epochs_run":  epochs_run,
        "best_metric": best_metric,
        "best_val":    best_val,
        "git_sha":     _git_sha(),
        "slurm_job":   _os.environ.get("SLURM_JOB_ID", ""),
    }
    new = not manifest.exists()
    try:
        with open(manifest, "a", newline="") as fh:
            _fcntl.flock(fh, _fcntl.LOCK_EX)
            w = _csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            _fcntl.flock(fh, _fcntl.LOCK_UN)
    except Exception as e:   # bookkeeping must never crash a finished run
        _log.warning("run-manifest append failed (%s): %s", manifest, e)


def expand_intervals_hourly(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    intime_lookup: pd.Series,  # indexed by stay_id
    stay_id_col: str = "stay_id",
    value_col: str = "rate",
) -> pd.DataFrame:
    """Expand [start, end] infusion intervals to one row per integer hour.

    For each row, emits one record per integer hour h (0-based relative to
    ICU intime) that falls within [starttime, endtime).

    Parameters
    ----------
    df : DataFrame with infusion records (one row per order segment)
    start_col, end_col : column names for start/end timestamps
    intime_lookup : Series mapping stay_id -> ICU intime (datetime)
    value_col : column holding the rate/value to carry forward

    Returns
    -------
    DataFrame with columns: stay_id, hour (int), value_col, plus any
    extra columns from df that are preserved (subject_id, hadm_id, itemid).
    """
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])
    df["_intime"] = df[stay_id_col].map(intime_lookup)
    df["_start_h"] = (df[start_col] - df["_intime"]).dt.total_seconds() / 3600.0
    df["_end_h"] = (df[end_col] - df["_intime"]).dt.total_seconds() / 3600.0

    keep_cols = [c for c in ["subject_id", "hadm_id", stay_id_col, "itemid", value_col]
                 if c in df.columns]

    rows = []
    for _, row in df.iterrows():
        start_h = row["_start_h"]
        end_h = row["_end_h"]
        if pd.isna(start_h) or pd.isna(end_h) or end_h <= start_h:
            continue
        first_h = int(np.floor(start_h)) + 1
        last_h = int(np.floor(end_h))
        if first_h > last_h:
            first_h = last_h = int(np.round((start_h + end_h) / 2))
        for h in range(first_h, last_h + 1):
            r = {c: row[c] for c in keep_cols if c in row.index}
            r["hour"] = float(h)
            rows.append(r)

    if not rows:
        return pd.DataFrame(columns=keep_cols + ["hour"])
    return pd.DataFrame(rows)


def expand_intervals_hourly_vectorized(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    intime_lookup: pd.Series,
    stay_id_col: str = "stay_id",
    value_col: str = "rate",
) -> pd.DataFrame:
    """Vectorized version of expand_intervals_hourly using numpy repeat.

    Substantially faster for large DataFrames.
    """
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])
    df["_intime"] = df[stay_id_col].map(intime_lookup)
    df["_start_h"] = (df[start_col] - df["_intime"]).dt.total_seconds() / 3600.0
    df["_end_h"] = (df[end_col] - df["_intime"]).dt.total_seconds() / 3600.0

    df = df.dropna(subset=["_start_h", "_end_h"])
    df = df.loc[df["_end_h"] > df["_start_h"]]

    first_h = np.floor(df["_start_h"].values).astype(int) + 1
    last_h = np.floor(df["_end_h"].values).astype(int)
    # ensure at least one point per interval
    last_h = np.maximum(last_h, first_h)
    n_hours = last_h - first_h + 1

    # repeat each row n_hours times
    idx = np.repeat(df.index, n_hours)
    hours = np.concatenate([
        np.arange(f, l + 1, dtype=float)
        for f, l in zip(first_h, last_h)
    ])

    keep_cols = [c for c in ["subject_id", "hadm_id", stay_id_col, "itemid", value_col]
                 if c in df.columns]
    out = df.loc[idx, keep_cols].copy().reset_index(drop=True)
    out["hour"] = hours.astype(float)
    return out