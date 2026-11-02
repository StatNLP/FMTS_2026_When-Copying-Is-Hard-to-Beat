"""
harmonize/stage3_make_dataset.py
Stage 3 — Forecast dataset construction.

Reads Stage-2 hourly Z-scored splits and emits per-stay sliding-window forecast
tensors as HDF5. Reproduction target: MLHC 2024 "Early Prediction of Causes
(not Effects)" — see docs/2605_comprehensive_documentation.md for the full design.

Per file (one per dataset × split × config):

    /attrs   window_in, window_out, stride, features_in, features_target,
             features_static, datasets, seed, imputation,
             predict_interventions, version
    /x_in           (N, T_in,  F_in)      float32   normalised inputs (Z-scored, then imputed)
    /mask_in        (N, T_in,  F_in)      bool      True where value was actually observed
    /x_target       (N, T_out, F_target)  float32   target window (Z-scored, NO imputation)
    /mask_target    (N, T_out, F_target)  bool
    /static         (N, F_static)         float32
    /static_mask    (N, F_static)         bool
    /stay_id        (N,)                  int64
    /ds_id          (N,)                  int8
    /t0_hour        (N,)                  float32   window-start hour since intime
    /los_hours      (N,)                  float32   full-stay LOS (metadata only)
    /mortality_hours (N,)                 float32   = los_hours if deceased else NaN (metadata)

Usage:
    python -m harmonize.stage3_make_dataset \
        --window-in 24 --window-out 24 \
        --stride first --features sofa \
        --imputation fwd --predict-interventions none \
        --datasets umm

See `--help` for full arg list.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from harmonize.feature_selection import (
    load_catalog,
    partition_by_role,
    resolve_features,
)

log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"
OUTPUT_DIR = ROOT / "data" / "dataset"
SCHEMA_VERSION = "1.0-forecast"

DATASETS = ["mimic_iii", "mimic_iv", "hirid", "eicu", "umm"]

STRIDE_VALUES = ("first", "4h", "24h")
IMPUTATION_VALUES = ("nan", "zero", "fwd", "mean", "interp", "learned")
PREDICT_INTERVENTIONS_VALUES = ("none", "infusions", "all")

# Modelling horizon: Stage-1 clips eICU/HiRID observations to 168h (7 days).
# Window generation must not exceed this horizon, otherwise long-stay datasets
# (eICU LOS up to 12153h, UMM up to 6187h) generate thousands of empty windows
# beyond the data extent that are then dropped by the quality filter (eICU/HiRID
# drop was ~34-38%, almost entirely this clip artefact). Capping here makes all
# datasets share the same 0-168h horizon and avoids the wasted generate-then-drop.
# With window_in+window_out=48, the last valid t0 is 168-48 = 120 → max 31
# windows/stay at stride=4h.
MAX_WINDOW_HORIZON_H = 168

# Static features always taken from the cohort file directly (not from long).
COHORT_STATIC_FEATURES = ["age", "sex", "weight_admission", "height"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_norm_stats(ds: str, splits_dir: Path) -> pd.DataFrame:
    p = splits_dir / f"{ds}_norm_stats.parquet"
    return pq.read_table(p).to_pandas().set_index("variable")


def load_cohort(ds: str) -> pd.DataFrame:
    p = ROOT / "data" / "long" / f"{ds}_cohort.parquet"
    df = pq.read_table(p).to_pandas()
    if "stay_id" not in df.columns and "patient_id" in df.columns:
        df["stay_id"] = df["patient_id"]
    return df


#: Which mortality endpoint `stay_deceased` returns. See §37.67.
#:   "legacy" — the per-dataset column priority below. THIS IS WHAT EVERY EXISTING H5 WAS BUILT
#:              WITH, so it is the default purely for reproducibility of the current artefacts.
#:              It is NOT a uniform endpoint: ICU death for eICU/HiRID, HOSPITAL death for
#:              MIMIC-III/IV, ALL-CAUSE-any-time for UMM (§37.35 D12).
#:   "icu"    — the uniform ICU definition (death within [intime, outtime+1h]; eICU/HiRID via
#:              discharge_status), identical to `explore_datasets._get_mortality_rate`, which
#:              documents the legacy columns as "deliberately NOT used … over-count ICU mortality".
#: ANY REBUILD OF STAGE 3 SHOULD PASS "icu" (`--mortality-endpoint icu`). The existing H5s are NOT
#: being patched: MF's decision 2026-10-27 — the published results must stay reproducible from the
#: files as they are, and no resources are available to recompute everything.
MORTALITY_ENDPOINT_DEFAULT = "legacy"


def stay_deceased(cohort: pd.DataFrame, endpoint: str = MORTALITY_ENDPOINT_DEFAULT) -> pd.Series:
    """Return Series stay_id -> 1.0 (deceased) / 0.0 (alive) / NaN (unknown).

    `endpoint="legacy"` (default, reproduces every existing H5) uses the per-dataset column
    priority below, which is NOT a uniform endpoint — see MORTALITY_ENDPOINT_DEFAULT and §37.35 D12.
    `endpoint="icu"` returns the uniform ICU definition and is what a rebuild should use.

    Handles different column names across datasets:
      UMM       : deceased (1.0 or NaN)
      MIMIC-III : hospital_mortality (0/1, ICU) or EXPIRE_FLAG (hospital)
      MIMIC-IV  : hospital_expire_flag (0/1)
      eICU/HiRID: mortality_icu not in cohort — return NaN Series (extracted
                  from long-format parquet as a static observation in Stage 1).
    """
    if endpoint not in ("legacy", "icu"):
        raise ValueError(f"endpoint must be 'legacy' or 'icu', got {endpoint!r}")
    cix = cohort.set_index("stay_id")
    if endpoint == "icu":
        return _stay_deceased_icu(cix)
    # Check candidate columns in priority order
    for col in ("deceased", "hospital_mortality", "hospital_expire_flag", "EXPIRE_FLAG"):
        if col in cix.columns:
            s = pd.to_numeric(cix[col], errors="coerce")
            if col == "deceased":
                return s   # 1.0 or NaN (UMM: NaN = unknown, not confirmed alive)
            return s.astype(float)   # 0.0 / 1.0
    # HiRID: discharge_status = "dead" / "alive" / NaN
    # eICU: discharge_status = "Alive" / "Expired" / NaN
    if "discharge_status" in cix.columns:
        ds_col = cix["discharge_status"].astype(str).str.strip().str.lower()
        return ds_col.map({"dead": 1.0, "alive": 0.0,
                           "expired": 1.0}).astype(float)
    # eICU: mortality stored as observation variable in long-format, not in cohort
    return pd.Series(np.nan, index=cix.index, dtype=float)


def _stay_deceased_icu(cix: pd.DataFrame) -> pd.Series:
    """Uniform ICU mortality: death timestamp within [icu_intime, icu_outtime + 1h].

    Mirrors `10_scripts/explore_datasets.py::_get_mortality_rate` exactly, which is the
    definition the whole analysis layer already uses (§37.58, §37.64, §37.66). eICU and HiRID
    carry no ICU timestamps but their `discharge_status` IS a unit-level outcome, so it is used
    directly there. Deliberately ignores `deceased` / `hospital_mortality` /
    `hospital_expire_flag` / `EXPIRE_FLAG`: those are hospital- or all-cause and over-count ICU
    mortality (UMM 27.2 % vs 19.4 %, MIMIC-III 11.8 % vs 8.0 %, MIMIC-IV 11.9 % vs 7.6 %).
    """
    cols = set(cix.columns)
    # eICU: the UNIT discharge status (not hospitaldischargestatus, which is hospital-level)
    if "unitdischargestatus" in cols:
        return cix["unitdischargestatus"].astype(str).str.strip().str.lower().eq("expired").astype(float)
    in_c = next((c for c in ("intime", "icu_start") if c in cols), None)
    out_c = next((c for c in ("outtime", "icu_end") if c in cols), None)
    d_c = next((c for c in ("deathtime", "death_time", "DOD", "dod") if c in cols), None)
    if in_c and out_c and d_c:
        ti, to, td = (pd.to_datetime(cix[c], errors="coerce") for c in (in_c, out_c, d_c))
        return (td.notna() & (td >= ti) & (td <= to + pd.Timedelta(hours=1))).astype(float)
    ds_c = next((c for c in ("discharge_status", "dischargestatus") if c in cols), None)
    if ds_c:
        return (cix[ds_c].astype(str).str.strip().str.lower()
                .isin(["expired", "dead"]).astype(float))
    return pd.Series(np.nan, index=cix.index, dtype=float)


def stay_los_hours(cohort: pd.DataFrame) -> pd.Series:
    """Returns Series stay_id -> LOS in hours."""
    if "los_hours" in cohort.columns:
        s = cohort.set_index("stay_id")["los_hours"]
    elif "los" in cohort.columns:
        s = cohort.set_index("stay_id")["los"] * 24.0
    else:
        raise KeyError(f"cohort has no los_hours or los column. cols: {list(cohort.columns)}")
    return s.astype(float)


def stay_static(ds: str, stay_ids: Sequence[int], cohort: pd.DataFrame,
                static_features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read static features per stay from the cohort table.

    Returns (static, static_mask) of shapes (N, F_static).
    """
    cohort_ix = cohort.set_index("stay_id")
    # `sex` is text "M"/"F" in cohort — encode to 1/0
    if "sex" in cohort_ix.columns and cohort_ix["sex"].dtype == object:
        cohort_ix["sex"] = cohort_ix["sex"].map({"M": 1.0, "F": 0.0})
    # MIMIC-IV cohort: gender column instead of sex
    if "sex" not in cohort_ix.columns and "gender" in cohort_ix.columns:
        cohort_ix["sex"] = cohort_ix["gender"].map({"M": 1.0, "F": 0.0})
    out = np.full((len(stay_ids), len(static_features)), np.nan, dtype=np.float32)
    # Map canonical static feature names to possible cohort column names.
    STATIC_ALIASES = {
        "weight_admission": ["weight_admission", "weight_admission_kg", "weight_kg", "weight"],
        "height":           ["height", "height_cm"],
        "age":              ["age"],
        "sex":              ["sex", "gender"],
    }
    for j, feat in enumerate(static_features):
        candidates = STATIC_ALIASES.get(feat, [feat])
        col = None
        for alt in candidates:
            if alt in cohort_ix.columns:
                col = alt
                break
        if col is None:
            continue
        vals = pd.to_numeric(cohort_ix.loc[stay_ids, col], errors="coerce").astype("float32").values
        out[:, j] = vals
    mask = ~np.isnan(out)
    out_filled = np.nan_to_num(out, nan=0.0, copy=False)
    return out_filled.astype(np.float32), mask


# ---------------------------------------------------------------------------
# Sliding-window logic
# ---------------------------------------------------------------------------

def _window_starts(los_h: float, window_in: int, window_out: int, stride: str) -> list[int]:
    """Return valid t0 hours for a single stay given its LOS.

    The window [t0, t0+window_in+window_out) must fit within both the stay LOS
    AND the global MAX_WINDOW_HORIZON_H (168h) data horizon — Stage-1 clips
    observations to 168h, so windows beyond that would be empty. The effective
    upper bound on t0 is therefore min(LOS, 168) - (window_in+window_out).
    """
    total = window_in + window_out
    horizon = min(los_h, float(MAX_WINDOW_HORIZON_H))
    if horizon < total:
        return []
    last = int(horizon - total)  # inclusive upper bound for t0
    if stride == "first":
        return [0]
    if stride == "4h":
        return list(range(0, last + 1, 4))
    if stride == "24h":
        return list(range(0, last + 1, 24))
    raise ValueError(f"unknown stride: {stride}")


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute(x: np.ndarray, mask: np.ndarray, strategy: str,
           train_mean: np.ndarray | None = None) -> np.ndarray:
    """Apply imputation to (T, F) array. `mask` is True where observed.
    Modifies a copy; returns float32."""
    x = x.astype(np.float32, copy=True)
    if strategy == "nan":
        return x
    if strategy == "zero":
        x[~mask] = 0.0
        return x
    if strategy == "mean":
        if train_mean is None:
            raise ValueError("mean imputation needs train_mean")
        nan_mask = ~mask
        x[nan_mask] = np.broadcast_to(train_mean, x.shape)[nan_mask]
        return x
    if strategy == "fwd":
        # forward-fill along time axis (axis 0), then zero-fill leading NaN
        T, F = x.shape
        for f in range(F):
            last = np.nan
            for t in range(T):
                if mask[t, f]:
                    last = x[t, f]
                elif not np.isnan(last):
                    x[t, f] = last
        x = np.nan_to_num(x, nan=0.0, copy=False)
        return x
    if strategy == "interp":
        T, F = x.shape
        for f in range(F):
            col = x[:, f].copy()
            obs_idx = np.where(mask[:, f])[0]
            if len(obs_idx) == 0:
                col[:] = 0.0
            elif len(obs_idx) == 1:
                col[:] = col[obs_idx[0]]
            else:
                col_interp = np.interp(np.arange(T), obs_idx, col[obs_idx])
                col = col_interp
            x[:, f] = col
        return x
    if strategy == "learned":
        # Leave NaN; downstream model handles missing-as-NaN with mask
        return x
    raise ValueError(f"unknown imputation strategy: {strategy}")


# ---------------------------------------------------------------------------
# Per-dataset pivot + window generation
# ---------------------------------------------------------------------------

def pivot_one_stay(df_stay: pd.DataFrame, features: list[str], hours: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Pivot one stay's long df to (T, F) matrix + mask. `hours` is the matrix
    height (one row per integer hour from 0 to hours-1). value_mean is used.
    Returns (x, mask) — x contains NaN where no observation."""
    feat_idx = {f: i for i, f in enumerate(features)}
    x = np.full((hours, len(features)), np.nan, dtype=np.float32)
    mask = np.zeros_like(x, dtype=bool)
    if df_stay.empty:
        return x, mask
    # df_stay: hour, variable, value_mean
    hour_arr = df_stay["hour"].to_numpy()
    var_arr  = df_stay["variable"].to_numpy()
    val_arr  = df_stay["value_mean"].to_numpy(dtype=np.float32)
    for h, v, val in zip(hour_arr, var_arr, val_arr):
        j = feat_idx.get(v)
        if j is None or h < 0 or h >= hours:
            continue
        if np.isnan(val):
            continue
        x[int(h), j] = val
        mask[int(h), j] = True
    return x, mask


def stays_with_minimum_los(cohort: pd.DataFrame, window_in: int, window_out: int) -> pd.Series:
    s = stay_los_hours(cohort)
    keep = s >= (window_in + window_out)
    return s.loc[keep]


# ---------------------------------------------------------------------------
# Process one (dataset, split) — writes one HDF5
# ---------------------------------------------------------------------------

def process_one(ds: str, split: str, splits_dir: Path,
                output_dir: Path, tag: str,
                window_in: int, window_out: int, stride: str,
                features_in: list[str], features_target: list[str],
                features_static: list[str], imputation: str,
                seed: int, ds_id: int,
                mortality_endpoint: str = MORTALITY_ENDPOINT_DEFAULT) -> int:
    """Return number of windows written."""
    src = splits_dir / f"{ds}_seed{seed}_{split}.parquet"
    cohort = load_cohort(ds)
    norm = load_norm_stats(ds, splits_dir)

    # Per-stay mortality (NaN for survivors/unknown, = LOS hours for deceased)
    dec = stay_deceased(cohort, endpoint=mortality_endpoint)   # stay_id -> 1.0 / 0.0 / NaN
    mort_h = stay_los_hours(cohort).where(dec == 1.0, other=np.nan)
    if mortality_endpoint == "legacy":
        log.warning(
            "[%s/%s] mortality_endpoint=legacy -> `mortality_hours` is NOT a uniform endpoint "
            "(ICU for eicu/hirid, HOSPITAL for mimic_iii/mimic_iv, ALL-CAUSE for umm; §37.35 D12). "
            "This default exists only to reproduce the existing H5s. Pass "
            "--mortality-endpoint icu for a corrected rebuild (§37.67).", ds, split)
    else:
        log.info("[%s/%s] mortality_endpoint=%s (uniform ICU death within [intime, outtime+1h])",
                 ds, split, mortality_endpoint)

    los_h = stays_with_minimum_los(cohort, window_in, window_out)
    if los_h.empty:
        log.warning("[%s/%s] no stays satisfy LOS ≥ %d", ds, split, window_in + window_out)
        return 0

    log.info("[%s/%s] reading %s (%.1f MB)", ds, split,
             src.name, src.stat().st_size / 1e6)
    needed_vars = set(features_in) | set(features_target)
    t0 = time.time()
    tbl = pq.read_table(
        src,
        columns=["stay_id", "hour", "variable", "value_mean"],
        filters=[("variable", "in", list(needed_vars))],
    )
    df = tbl.to_pandas()
    log.info("[%s/%s]   loaded %d rows in %.1fs",
             ds, split, len(df), time.time() - t0)

    # Restrict to stays that pass LOS filter AND are in this split
    split_stays = set(df["stay_id"].unique())
    valid_stays = sorted(set(los_h.index) & split_stays)
    log.info("[%s/%s]   stays with windows: %d", ds, split, len(valid_stays))

    # Train mean per feature (for mean-imputation); Z-scored data has mean ~ 0
    train_mean_in = np.array(
        [norm.loc[f, "train_mean"] if f in norm.index else 0.0 for f in features_in],
        dtype=np.float32,
    )
    # In Z-scored space the mean is 0; treat mean-imp as zero-imp.
    train_mean_in[:] = 0.0

    # Allocate growable buffers (write in one batch at end)
    F_in  = len(features_in)
    F_out = len(features_target)
    F_sta = len(features_static)
    T_in  = window_in
    T_out = window_out

    # First pass: count windows
    n_windows = 0
    for sid in valid_stays:
        n_windows += len(_window_starts(los_h[sid], window_in, window_out, stride))
    if n_windows == 0:
        log.warning("[%s/%s] 0 windows generated", ds, split)
        return 0
    log.info("[%s/%s]   windows total: %d (stride=%s)", ds, split, n_windows, stride)

    x_in        = np.full((n_windows, T_in,  F_in),  np.nan, dtype=np.float32)
    mask_in     = np.zeros((n_windows, T_in,  F_in),  dtype=bool)
    x_target    = np.full((n_windows, T_out, F_out), np.nan, dtype=np.float32)
    mask_target = np.zeros((n_windows, T_out, F_out), dtype=bool)
    stay_id_arr = np.zeros(n_windows, dtype=np.int64)
    ds_id_arr   = np.full(n_windows, ds_id, dtype=np.int8)
    t0_arr      = np.zeros(n_windows, dtype=np.float32)
    los_arr     = np.zeros(n_windows, dtype=np.float32)
    mort_arr    = np.full(n_windows, np.nan, dtype=np.float32)

    feat_in_idx  = {f: i for i, f in enumerate(features_in)}
    feat_out_idx = {f: i for i, f in enumerate(features_target)}

    # Group by stay
    grouped = df.groupby("stay_id", sort=False)
    win_i = 0
    last_log = time.time()
    for s_i, sid in enumerate(valid_stays):
        if sid not in grouped.groups:
            continue
        df_stay = grouped.get_group(sid)
        stay_los = los_h[sid]

        # Build full per-stay matrix at hourly resolution (size = stay LOS + buffer)
        n_hours_stay = int(stay_los) + 1
        # Pivot for input features
        x_full_in, m_full_in = pivot_one_stay(df_stay, features_in, n_hours_stay)
        # Pivot for target features (often overlaps with input but indexed differently)
        if features_target == features_in:
            x_full_out, m_full_out = x_full_in, m_full_in
        else:
            x_full_out, m_full_out = pivot_one_stay(df_stay, features_target, n_hours_stay)

        for t0_h in _window_starts(stay_los, window_in, window_out, stride):
            t1 = t0_h + window_in
            t2 = t1 + window_out
            x_in[win_i]     = x_full_in[t0_h:t1]
            mask_in[win_i]  = m_full_in[t0_h:t1]
            x_target[win_i] = x_full_out[t1:t2]
            mask_target[win_i] = m_full_out[t1:t2]
            stay_id_arr[win_i] = sid
            t0_arr[win_i]      = t0_h
            los_arr[win_i]     = stay_los
            mort_arr[win_i]    = mort_h.get(sid, np.nan)
            win_i += 1

        if time.time() - last_log > 30:
            log.info("[%s/%s]   %d/%d stays processed", ds, split, s_i + 1, len(valid_stays))
            last_log = time.time()

    assert win_i == n_windows, f"window count mismatch: filled {win_i}, expected {n_windows}"

    # ---------------------------------------------------------------------------
    # Window quality filter — applied to both input and target windows.
    # A window is kept only if it satisfies all six criteria:
    #   Target  (1) ≥1 observed value in the last 3 target hours (prevents
    #               tail-truncated windows that contribute nothing at the end;
    #               3h tolerance avoids over-rejection with stride=4h)
    #           (2) ≥24 total observations in the target window
    #           (3) ≥3 distinct variables observed in the target
    #   Input   (4) ≥1 observed value in the last 3 input hours
    #           (5) ≥24 total observations in the input window
    #           (6) ≥3 distinct variables observed in the input
    # ---------------------------------------------------------------------------
    tgt_last3 = mask_target[:, -3:, :].any(axis=(1, 2))      # (N,)
    tgt_obs   = mask_target.sum(axis=(1, 2)) >= 24            # (N,)
    tgt_vars  = mask_target.any(axis=1).sum(axis=1) >= 3      # (N,)
    inp_last3 = mask_in[:, -3:, :].any(axis=(1, 2))           # (N,)
    inp_obs   = mask_in.sum(axis=(1, 2)) >= 24                # (N,)
    inp_vars  = mask_in.any(axis=1).sum(axis=1) >= 3          # (N,)
    keep = tgt_last3 & tgt_obs & tgt_vars & inp_last3 & inp_obs & inp_vars

    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        log.info(
            "[%s/%s]   quality filter: dropped %d / %d windows — "
            "tgt(last3h=%d obs<24=%d vars<3=%d) inp(last3h=%d obs<24=%d vars<3=%d)",
            ds, split, n_dropped, n_windows,
            int((~tgt_last3).sum()), int((~tgt_obs).sum()), int((~tgt_vars).sum()),
            int((~inp_last3).sum()), int((~inp_obs).sum()),  int((~inp_vars).sum()),
        )
        x_in        = x_in[keep]
        mask_in     = mask_in[keep]
        x_target    = x_target[keep]
        mask_target = mask_target[keep]
        stay_id_arr = stay_id_arr[keep]
        ds_id_arr   = ds_id_arr[keep]
        t0_arr      = t0_arr[keep]
        los_arr     = los_arr[keep]
        mort_arr    = mort_arr[keep]
        n_windows   = int(keep.sum())
    else:
        log.info("[%s/%s]   quality filter: all %d windows passed", ds, split, n_windows)

    # Pre-compute SOFA subscores on de-normalised targets.
    # sofa_scores: (N, T_out, 7) — [resp, coag, liver, cardiovascular, cns, renal, total]
    from harmonize.scores.sofa import sofa_components, denormalise as sofa_denorm
    SOFA_SCORE_NAMES = ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal", "total"]
    norm_for_sofa = load_norm_stats(ds, splits_dir)
    x_target_phys = sofa_denorm(x_target, features_target, norm_for_sofa)
    sofa_scores = np.full((n_windows, T_out, 7), np.nan, dtype=np.float32)
    for k in range(n_windows):
        comp = sofa_components(x_target_phys[k], features_target)
        sofa_scores[k, :, 0] = comp.respiration
        sofa_scores[k, :, 1] = comp.coagulation
        sofa_scores[k, :, 2] = comp.liver
        sofa_scores[k, :, 3] = comp.cardiovascular
        sofa_scores[k, :, 4] = comp.cns
        sofa_scores[k, :, 5] = comp.renal
        sofa_scores[k, :, 6] = comp.total()
    del x_target_phys

    # Apply imputation to inputs (target stays NaN — model evaluates on mask)
    if imputation != "nan":
        log.info("[%s/%s]   applying imputation: %s", ds, split, imputation)
        for k in range(n_windows):
            x_in[k] = impute(x_in[k], mask_in[k], imputation, train_mean_in)
    # Target: never imputed (we want masked-MSE on the actually-observed values)

    # Static features
    log.info("[%s/%s]   reading static features for %d windows", ds, split, n_windows)
    static_arr, static_mask = stay_static(ds, stay_id_arr.tolist(), cohort, features_static)

    # Write HDF5
    out_path = output_dir / f"{ds}_seed{seed}_{split}__{tag}.h5"
    log.info("[%s/%s]   writing %s", ds, split, out_path.name)
    with h5py.File(out_path, "w") as h5:
        for arr_name, arr in [
            ("x_in",        x_in),
            ("mask_in",     mask_in),
            ("x_target",    x_target),
            ("mask_target", mask_target),
            ("static",      static_arr),
            ("static_mask", static_mask),
            ("stay_id",     stay_id_arr),
            ("ds_id",       ds_id_arr),
            ("t0_hour",     t0_arr),
            ("los_hours",   los_arr),
            ("mortality_hours", mort_arr),
        ]:
            h5.create_dataset(arr_name, data=arr,
                              compression="gzip", compression_opts=4,
                              chunks=True)
        h5.attrs["window_in"]  = window_in
        h5.attrs["window_out"] = window_out
        h5.attrs["stride"]     = stride
        h5.attrs["features_in"]      = np.array(features_in, dtype=object)
        h5.attrs["features_target"]  = np.array(features_target, dtype=object)
        h5.attrs["features_static"]  = np.array(features_static, dtype=object)
        h5.attrs["dataset"]    = ds
        h5.attrs["split"]      = split
        h5.attrs["seed"]       = seed
        h5.attrs["imputation"] = imputation
        h5.attrs["version"]    = SCHEMA_VERSION

        # Observation count matrices — how many windows have a measurement at each hour.
        # input_obs_count:  (T_in,  F_in)     — coverage in the input window
        # target_obs_count: (T_out, F_target) — coverage in the forecast window
        inp_count = mask_in.sum(axis=0).astype(np.int32)       # (T_in,  F_in)
        obs_count = mask_target.sum(axis=0).astype(np.int32)   # (T_out, F_target)
        h5.create_dataset("input_obs_count", data=inp_count,
                          compression="gzip", compression_opts=1)
        h5.create_dataset("target_obs_count", data=obs_count,
                          compression="gzip", compression_opts=1)
        # SOFA subscores on de-normalised targets: (N, T_out, 7)
        h5.create_dataset("sofa_scores", data=sofa_scores,
                          compression="gzip", compression_opts=4, chunks=True)
        h5.attrs["sofa_score_names"] = np.array(SOFA_SCORE_NAMES, dtype=object)
        h5.attrs["n_windows"] = n_windows

    # Write per-split severity statistics alongside the HDF5
    # Use per-stay deceased flag for mortality rate (not per-window mort_arr,
    # which is NaN-filled and not reliable for datasets without cohort-level mortality).
    stay_dec_flag = dec.reindex(stay_id_arr).values  # shape (N,), 1.0/0.0/NaN per window
    mortality_rate = float(np.nanmean(stay_dec_flag == 1.0))
    _write_severity_stats(
        out_path.with_suffix(".stats.json"),
        ds=ds, split=split, n_windows=n_windows,
        stay_id_arr=stay_id_arr,
        t0_arr=t0_arr,
        los_arr=los_arr,
        mortality_rate=mortality_rate,
        sofa_scores=sofa_scores,
    )

    del x_in, mask_in, x_target, mask_target, sofa_scores
    gc.collect()
    return n_windows


def _write_severity_stats(
    out_path: Path,
    ds: str, split: str, n_windows: int,
    stay_id_arr: np.ndarray,
    t0_arr: np.ndarray,
    los_arr: np.ndarray,
    mortality_rate: float,
    sofa_scores: np.ndarray,   # (N, T_out, 7) — last dim: resp/coag/liver/cv/cns/renal/total
) -> None:
    """Write a small JSON with severity distribution for this split.

    Fields
    ------
    n_windows, n_stays            — counts
    mortality_rate                — fraction of stays (not windows) that died
    los_h_{p25,p50,p75}          — LOS distribution
    t0_h_{p25,p50,p75}           — window-start distribution (day-1 vs later)
    sofa_t0_{mean,p25,p50,p75}   — SOFA total at first target hour (admission-proximate)
    sofa_mean_{mean,p25,p50,p75} — SOFA total averaged across the 24 h target window
    """
    def _pct(a: np.ndarray, q: float) -> float:
        v = np.nanpercentile(a, q * 100)
        return round(float(v), 2)

    def _mean(a: np.ndarray) -> float:
        return round(float(np.nanmean(a)), 3)

    sofa_total = sofa_scores[:, :, 6]            # (N, T_out)
    sofa_t0    = sofa_total[:, 0]                # SOFA at first target hour
    sofa_win   = np.nanmean(sofa_total, axis=1)  # mean over window

    stats: dict = {
        "dataset":          ds,
        "split":            split,
        "n_windows":        int(n_windows),
        "n_stays":          int(np.unique(stay_id_arr).shape[0]),
        "mortality_rate":   round(mortality_rate, 4),
        "los_h_p25":        _pct(los_arr, .25),
        "los_h_p50":        _pct(los_arr, .50),
        "los_h_p75":        _pct(los_arr, .75),
        "t0_h_p25":         _pct(t0_arr,  .25),
        "t0_h_p50":         _pct(t0_arr,  .50),
        "t0_h_p75":         _pct(t0_arr,  .75),
        "sofa_t0_mean":     _mean(sofa_t0),
        "sofa_t0_p25":      _pct(sofa_t0, .25),
        "sofa_t0_p50":      _pct(sofa_t0, .50),
        "sofa_t0_p75":      _pct(sofa_t0, .75),
        "sofa_mean_mean":   _mean(sofa_win),
        "sofa_mean_p25":    _pct(sofa_win, .25),
        "sofa_mean_p50":    _pct(sofa_win, .50),
        "sofa_mean_p75":    _pct(sofa_win, .75),
    }
    out_path.write_text(json.dumps(stats, indent=2))
    log.info("[%s/%s] severity stats → %s", ds, split, out_path.name)


# ---------------------------------------------------------------------------
# Stay-subsampling (controlled dataset-size experiments)
# ---------------------------------------------------------------------------

def subsample_stays(src: Path, dst: Path, n_stays: int, seed: int) -> int:
    """Write a copy of ``src`` restricted to ``n_stays`` randomly chosen stays.

    Used for the controlled size test (§16.6a): training size is the only thing
    that varies, so only the TRAIN split is ever subsampled and val/test are
    left full. Every dataset, attribute and the per-(t, channel) observation
    counts are carried over; the counts are recomputed from the kept rows so the
    sidecar statistics stay truthful.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(src, "r") as fin:
        sid = fin["stay_id"][:]
        uniq = np.unique(sid)
        if n_stays >= uniq.shape[0]:
            raise ValueError(f"{src.name} has only {uniq.shape[0]} stays, cannot take {n_stays}")
        keep_stays = rng.choice(uniq, size=n_stays, replace=False)
        row = np.flatnonzero(np.isin(sid, keep_stays))
        row.sort()
        log.info("%s: %d/%d stays -> %d/%d windows",
                 src.name, n_stays, uniq.shape[0], row.shape[0], sid.shape[0])

        dst.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst, "w") as fout:
            for k, v in fin.attrs.items():
                fout.attrs[k] = v
            fout.attrs["n_windows"] = int(row.shape[0])
            fout.attrs["subsampled_from"] = src.name
            fout.attrs["subsample_n_stays"] = int(n_stays)
            fout.attrs["subsample_seed"] = int(seed)
            for name in fin.keys():
                d = fin[name]
                if name in ("input_obs_count", "target_obs_count"):
                    continue          # recomputed below from the kept rows
                out = fout.create_dataset(
                    name, shape=(row.shape[0],) + d.shape[1:], dtype=d.dtype,
                    chunks=True, compression="gzip", compression_opts=4)
                # copy in blocks: the full arrays are too large to hold at once
                for lo in range(0, row.shape[0], 8192):
                    idx = row[lo:lo + 8192]
                    out[lo:lo + idx.shape[0]] = d[idx]
            for name, src_mask in (("input_obs_count", "mask_in"),
                                   ("target_obs_count", "mask_target")):
                if src_mask not in fin:
                    continue
                acc = np.zeros(fin[src_mask].shape[1:], dtype=np.int64)
                for lo in range(0, row.shape[0], 8192):
                    idx = row[lo:lo + 8192]
                    acc += fin[src_mask][idx].sum(axis=0)
                fout.create_dataset(name, data=acc.astype(np.int32))
    return int(row.shape[0])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Subsample mode: derive a smaller TRAIN split from an existing h5 -----
    if args.subsample_stays:
        if args.subsample_from is None:
            raise SystemExit("--subsample-stays requires --subsample-from <train h5>")
        src = args.subsample_from
        for n in args.subsample_stays:
            stem = src.name
            if "_seed" not in stem:
                raise SystemExit(f"cannot derive an output name from {stem}")
            head, tail = stem.split("_seed", 1)
            dst = args.output_dir / f"{head}_s{n}_seed{tail}"
            subsample_stays(src, dst, n_stays=n, seed=args.seed)
            log.info("wrote %s", dst)
        return

    # ----- Resolve features -----
    catalog = load_catalog()
    all_features = resolve_features(args.features, catalog)
    parts = partition_by_role(all_features, catalog)
    obs_feats        = parts["observation"]
    intervention_feats = parts["intervention"]

    # input = target: all observations + interventions.
    # features_in and features_target are always identical so the model learns
    # to predict every input channel. Teacher forcing uses features_target as
    # the next-step input during training; for inference they are the same.
    # (--predict-interventions is kept for backwards-compat / ablation studies.)
    features_in = sorted(obs_feats + intervention_feats)
    if args.predict_interventions == "none":
        features_target = features_in          # default: full alignment
    elif args.predict_interventions == "infusions":
        med_fluid = [f for f in intervention_feats
                     if catalog.get(f, {}).get("type") in ("medication", "fluid")]
        features_target = sorted(obs_feats + med_fluid)
    elif args.predict_interventions == "all":
        features_target = features_in
    else:
        raise ValueError(f"bad --predict-interventions: {args.predict_interventions}")

    features_static = list(COHORT_STATIC_FEATURES)

    log.info("Feature selection '%s' -> %d input / %d target / %d static",
             args.features, len(features_in), len(features_target), len(features_static))
    log.info("  features_in:     %s", features_in)
    log.info("  features_target: %s", features_target)
    log.info("  features_static: %s", features_static)

    # ----- Output tag -----
    tag = f"{args.features}_{args.stride}_{args.imputation}"
    if args.predict_interventions != "none":
        tag += f"_p{args.predict_interventions}"

    # Write a JSON sidecar with the feature lists for reproducibility
    side = {
        "window_in":  args.window_in,
        "window_out": args.window_out,
        "stride":     args.stride,
        "features_keyword": args.features,
        "features_in":      features_in,
        "features_target":  features_target,
        "features_static":  features_static,
        "imputation": args.imputation,
        "predict_interventions": args.predict_interventions,
        "seed":       args.seed,
        "version":    SCHEMA_VERSION,
        "datasets":   args.datasets,
    }
    (args.output_dir / f"manifest__{tag}.json").write_text(json.dumps(side, indent=2))

    # ----- Per dataset × split -----
    ds_name_to_id = {ds: i for i, ds in enumerate(args.datasets)}
    total = 0
    for ds in args.datasets:
        for split in ("train", "val", "test"):
            n = process_one(
                ds=ds, split=split,
                splits_dir=args.splits_dir, output_dir=args.output_dir,
                tag=tag,
                window_in=args.window_in, window_out=args.window_out,
                stride=args.stride,
                features_in=features_in, features_target=features_target,
                features_static=features_static, imputation=args.imputation,
                seed=args.seed, ds_id=ds_name_to_id[ds],
                mortality_endpoint=args.mortality_endpoint,
            )
            total += n
    log.info("Stage-3 done. Total windows written: %d", total)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 3 — forecast dataset (sliding-window HDF5)")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--window-in",  type=int, default=24)
    p.add_argument("--window-out", type=int, default=24)
    p.add_argument("--stride",     choices=STRIDE_VALUES, default="first")
    p.add_argument("--features",   default="sofa",
                   help="all | sofa | core | covered<P> | custom:<path>")
    p.add_argument("--imputation", choices=IMPUTATION_VALUES, default="fwd")
    p.add_argument("--predict-interventions",
                   choices=PREDICT_INTERVENTIONS_VALUES, default="none")
    p.add_argument("--datasets",   nargs="+", default=DATASETS)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--mortality-endpoint", choices=["legacy", "icu"],
                   default=MORTALITY_ENDPOINT_DEFAULT,
                   help="Endpoint behind `mortality_hours`. 'legacy' (default) reproduces the "
                        "existing H5s but is NOT uniform across databases (§37.35 D12); 'icu' is "
                        "the uniform ICU definition and is what any REBUILD should use (§37.67).")
    p.add_argument("--subsample-stays", type=int, nargs="+", default=None,
                   help="controlled size test: write stay-subsampled copies of "
                        "--subsample-from with these stay counts (train only; "
                        "val/test stay full so only training size varies)")
    p.add_argument("--subsample-from", type=Path, default=None,
                   help="source train h5 for --subsample-stays")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
