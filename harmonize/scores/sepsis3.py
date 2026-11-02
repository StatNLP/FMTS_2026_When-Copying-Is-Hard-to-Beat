"""
harmonize/scores/sepsis3.py
Sepsis-3 binary label from a SOFA trajectory + suspected-infection indicator.

Reference
---------
- Singer M. et al., "The Third International Consensus Definitions for
  Sepsis and Septic Shock (Sepsis-3)", JAMA 2016.

Definition (paraphrased)
------------------------
    Sepsis  =  suspected_infection  AND  organ_dysfunction
where organ_dysfunction = increase in total SOFA ≥ 2 from baseline,
within a ±24 h window around the time of suspected infection.

Baseline SOFA
-------------
Singer-2016 leaves baseline SOFA = 0 in absence of prior data ("baseline
SOFA can be assumed to be 0 in patients not known to have preexisting
organ dysfunction"). In our 24/24 pipeline we have two pragmatic choices:

  (a) baseline = 0                                        — Singer default
  (b) baseline = SOFA at t=0 of the input window          — within-window Δ

Both are exposed via `baseline_sofa`. We default to (a) (Singer-2016).

Suspected infection — approximate (all datasets except UMM)
-------------------
Singer defines susinf via concurrent antibiotic admin + body-fluid culture.
We don't have culture timestamps in most datasets — Stage-1 has the
`had_<antibiotic>` indicators (`had_pip_tazo`, `had_meropenem`, etc.). As a
working approximation:

    susinf_t = any(had_<antibiotic>) at time t

This will over-call susinf (every antibiotic is treated as infection-driven),
but matches what Paper-1 also does.

Suspected infection — full (UMM, 260528 export)
-------------------------------------------------
UMM has antibiotic start timestamps (`abx_pip_tazo`, `abx_meropenem`,
`abx_vancomycin`, `abx_linezolid` at `hour` of first dose) and blood
culture order times (`blood_culture` at `hour` of order). Full Sepsis-3
suspected infection = first antibiotic start and any blood culture within
±72 h of each other. Use `susinf_per_stay_umm()` to extract this from the
Stage-1 long parquet.

Functions
---------
    delta_sofa(sofa, baseline)          — element-wise SOFA gain
    organ_dysfunction(sofa, baseline)   — ΔSOFA ≥ 2
    sepsis3_at_t(sofa, susinf)          — per-timestep binary
    sepsis3_per_stay(sofa, susinf)      — single bool per stay
    susinf_from_matrix(x, names)        — approx. susinf from antibiotic flags
    susinf_per_stay_umm(long_df)        — full UMM susinf from long parquet
    sepsis3_per_stay_umm(long_df, ...)  — UMM Sepsis-3 from long parquet + SOFA
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def delta_sofa(sofa_traj: np.ndarray, baseline_sofa: float = 0.0) -> np.ndarray:
    """ΔSOFA from baseline. `sofa_traj` shape: (..., T) (last axis time)."""
    return sofa_traj - baseline_sofa


def organ_dysfunction(sofa_traj: np.ndarray,
                      baseline_sofa: float = 0.0,
                      threshold: float = 2.0) -> np.ndarray:
    """Boolean: ΔSOFA ≥ threshold. Shape preserved."""
    return delta_sofa(sofa_traj, baseline_sofa) >= threshold


def sepsis3_at_t(sofa_traj: np.ndarray,
                 susinf_traj: np.ndarray,
                 baseline_sofa: float = 0.0,
                 threshold: float = 2.0,
                 window_h: int = 24) -> np.ndarray:
    """Per-timestep Sepsis-3 indicator.

    `sofa_traj`, `susinf_traj` shape (..., T). At timestep t the criterion
    is met if susinf is positive in [t-window_h, t+window_h] AND organ-
    dysfunction is positive in the same window.

    Implementation: forward+backward roll over the time axis with `window_h`
    radius.
    """
    od = organ_dysfunction(sofa_traj, baseline_sofa, threshold)
    susinf = susinf_traj.astype(bool)
    # Boolean expanding window: union over (t-w .. t+w)
    od_w     = _expand_bool(od,     window_h)
    susinf_w = _expand_bool(susinf, window_h)
    return od_w & susinf_w


def sepsis3_per_stay(sofa_traj: np.ndarray,
                     susinf_traj: np.ndarray,
                     baseline_sofa: float = 0.0,
                     threshold: float = 2.0,
                     window_h: int = 24) -> bool | np.ndarray:
    """Returns a single bool per stay: does the patient meet Sepsis-3 at any
    timestep in the trajectory?  Works for 1-D (T,) or batched (..., T) input.
    """
    at_t = sepsis3_at_t(sofa_traj, susinf_traj, baseline_sofa, threshold, window_h)
    return at_t.any(axis=-1)


def susinf_from_matrix(x: np.ndarray, feature_names: Sequence[str],
                       antibiotic_features: Sequence[str] | None = None,
                       ) -> np.ndarray:
    """Build a (..., T) susinf trajectory from antibiotic indicator columns.

    `x` is a physical-unit feature matrix; antibiotic columns are expected
    to be 0/1 indicators. If no antibiotic feature is present, returns all
    False with `x`'s leading shape.
    """
    if antibiotic_features is None:
        antibiotic_features = [
            "had_pip_tazo", "had_meropenem",
            "had_vancomycin", "had_linezolid",
        ]
    present = [f for f in antibiotic_features if f in feature_names]
    if not present:
        return np.zeros(x.shape[:-1], dtype=bool)
    idx = [list(feature_names).index(f) for f in present]
    # Any of them > 0 at this timestep:
    return (x[..., idx] > 0).any(axis=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_bool(a: np.ndarray, radius: int) -> np.ndarray:
    """Boolean roll/union over a window of ±radius along the LAST axis.

    Equivalent to scipy.ndimage.binary_dilation along that axis. Implemented
    with cumulative OR shifts to avoid a scipy dependency.
    """
    if radius <= 0:
        return a.astype(bool)
    out = a.astype(bool).copy()
    for r in range(1, radius + 1):
        out[..., r:]   |= a[..., :-r]
        out[..., :-r]  |= a[..., r:]
    return out


# ---------------------------------------------------------------------------
# Full Sepsis-3 suspected infection — any dataset with timestamped antibiotic
# start events (abx_*) + blood culture orders (UMM, MIMIC-III, MIMIC-IV).
# ---------------------------------------------------------------------------

def susinf_per_stay(
    long_df: pd.DataFrame,
    window_h: float = 72.0,
    abx_prefix: str = "abx_",
    culture_var: str = "blood_culture",
    union: bool = False,
) -> pd.DataFrame:
    """Suspected infection per stay from a Stage-1 long parquet.

    Dataset-agnostic: detects any variable starting with `abx_prefix` as an
    antibiotic start event, and `culture_var` as a blood culture order. Works
    for UMM (abx_pip_tazo/meropenem/vancomycin/linezolid) and MIMIC-III/IV
    (10 abx_* itemid groups), all + blood_culture.

        suspected_infection = an antibiotic start AND a blood culture within
                              ±window_h (default 72 h) of each other.
    Infection onset hour = min(abx_start_h, paired_bc_h) over the closest pair.

    Excludes the `_end` antibiotic stop markers (UMM emits abx_*_end too).

    Returns a per-stay DataFrame: stay_id, has_susinf, susinf_onset_h,
    first_abx_h, first_bc_h, n_abx_events, n_bc_events.
    """
    is_abx = (long_df["variable"].str.startswith(abx_prefix)
              & ~long_df["variable"].str.endswith("_end"))
    abx = long_df[is_abx]
    bc  = long_df[long_df["variable"] == culture_var]

    all_stays = long_df["stay_id"].unique()
    abx_by_stay = abx.groupby("stay_id")["hour"].apply(sorted).to_dict()
    bc_by_stay  = bc.groupby("stay_id")["hour"].apply(sorted).to_dict()

    records: list[dict] = []
    for sid in all_stays:
        abx_hours = abx_by_stay.get(sid, [])
        bc_hours  = bc_by_stay.get(sid, [])
        first_abx = float(abx_hours[0]) if abx_hours else float("nan")
        first_bc  = float(bc_hours[0])  if bc_hours  else float("nan")

        onset_h, has_susinf = float("nan"), False
        if union:
            # MLHC-2024 eICU definition: suspected infection = ANY antibiotic OR ANY culture
            # (no ±window co-occurrence required) — the looser "suspected infection".
            cands = [h for h in (first_abx, first_bc) if not np.isnan(h)]
            if cands:
                has_susinf = True
                onset_h = min(cands)
        else:
            for abx_h in abx_hours:
                for bc_h in bc_hours:
                    if abs(abx_h - bc_h) <= window_h:
                        candidate = min(abx_h, bc_h)
                        if not has_susinf or candidate < onset_h:
                            onset_h = candidate
                        has_susinf = True
        records.append({
            "stay_id": sid, "has_susinf": has_susinf, "susinf_onset_h": onset_h,
            "first_abx_h": first_abx, "first_bc_h": first_bc,
            "n_abx_events": len(abx_hours), "n_bc_events": len(bc_hours),
        })

    result = pd.DataFrame(records)
    n_susinf, n_total = result["has_susinf"].sum(), len(result)
    print(f"susinf_per_stay: {n_susinf}/{n_total} stays "
          f"({100*n_susinf/max(n_total,1):.1f}%) meet suspected-infection "
          f"criterion (window_h={window_h})")
    return result


def susinf_per_stay_umm(long_df: pd.DataFrame, window_h: float = 72.0) -> pd.DataFrame:
    """Backward-compatible alias — susinf_per_stay works for UMM unchanged."""
    return susinf_per_stay(long_df, window_h=window_h)


#: Sepsis-3 requires the SOFA rise within ±24 h of the suspected-infection time (Singer 2016).
SEPSIS3_COINCIDENCE_H = 24.0


def susinf_coincident(onset_h, t0_hour, window_in: int, window_out: int,
                      coincidence_h: float = SEPSIS3_COINCIDENCE_H) -> np.ndarray:
    """Does this window's organ dysfunction coincide with the suspected infection?

    Sepsis-3 requires the SOFA rise to fall within ±`coincidence_h` of the suspected-infection
    time — **not** merely in a stay that had an infection at some point. Using the per-stay
    `has_susinf` flag instead labelled 60–81 % of positive windows against the definition
    (§37.38 D19), with a median lag of +15 h to +46 h, i.e. deterioration a day or two *after*
    the workup.

    ΔSOFA is assessed on the TARGET window, hours ``[t0 + window_in, t0 + window_in + window_out)``,
    so the criterion is an overlap between that interval and ``[onset − c, onset + c]``. A stay
    with no suspected infection (NaN onset) is False.

    Note this is deliberately *not* one window per stay: the band is 2·`coincidence_h` wide, so a
    single infection episode legitimately marks several consecutive windows of the same stay.

    MIMIC-III CAVEAT (§37.62): its prescriptions carry DAY resolution, so every `abx_*` event sits
    at its day boundary and `onset_h` carries up to 24 h of one-sided quantisation error against
    this 48 h band — 100 % of MIMIC-III within-stay abx gaps are exact multiples of 24 h, against
    5.6 % for MIMIC-IV. The ±24 h test is therefore sharper than MIMIC-III supports, and some of its
    calls there are timing noise. MIMIC-IV and UMM have real timestamps. The ±72 h susinf pairing in
    `susinf_per_stay` is wide enough to be unaffected.
    """
    onset = np.asarray(onset_h, dtype=np.float64)
    t0    = np.asarray(t0_hour, dtype=np.float64)
    dys_lo = t0 + float(window_in)
    dys_hi = dys_lo + float(window_out)
    with np.errstate(invalid="ignore"):
        ok = (dys_lo <= onset + coincidence_h) & (dys_hi >= onset - coincidence_h)
    return np.asarray(ok & np.isfinite(onset), dtype=bool)


def sepsis3_per_stay_umm(
    susinf_df: pd.DataFrame,
    sofa_hours: dict[int, np.ndarray],
    sofa_values: dict[int, np.ndarray],
    sofa_window_h: float = 24.0,
    baseline_window_h: float = 48.0,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Sepsis-3 onset per stay for UMM.

    Combines suspected-infection results (from `susinf_per_stay_umm`) with
    SOFA trajectories to determine Sepsis-3 onset.

    Parameters
    ----------
    susinf_df : Output of `susinf_per_stay_umm`.
    sofa_hours : {stay_id → np.ndarray of hours} — hourly grid of SOFA computation.
    sofa_values : {stay_id → np.ndarray of SOFA totals} — same shape as sofa_hours.
    sofa_window_h : Hours around infection onset to check for ΔSOFA ≥ threshold.
    baseline_window_h : Look-back window for baseline SOFA before infection onset.
                        If no data in look-back, baseline = 0 (Singer default).
    threshold : ΔSOFA threshold (default 2.0 per Singer 2016).

    Returns
    -------
    susinf_df extended with:
        has_sepsis3      bool   — True if Sepsis-3 criteria met
        sepsis3_onset_h  float  — hour of first Sepsis-3 criterion fulfilment (NaN otherwise)
        baseline_sofa    float  — SOFA used as baseline (0 if no prior data)
        peak_sofa        float  — peak SOFA within sofa_window around infection onset
        delta_sofa       float  — peak_sofa - baseline_sofa (NaN if no susinf)
    """
    result = susinf_df.copy()
    has_sepsis3      = []
    sepsis3_onset_h  = []
    baseline_sofa_v  = []
    peak_sofa_v      = []
    delta_sofa_v     = []

    for _, row in result.iterrows():
        sid = row["stay_id"]
        if not row["has_susinf"] or np.isnan(row["susinf_onset_h"]):
            has_sepsis3.append(False)
            sepsis3_onset_h.append(float("nan"))
            baseline_sofa_v.append(float("nan"))
            peak_sofa_v.append(float("nan"))
            delta_sofa_v.append(float("nan"))
            continue

        t_inf = row["susinf_onset_h"]
        hours = sofa_hours.get(sid)
        vals  = sofa_values.get(sid)

        if hours is None or vals is None or len(hours) == 0:
            has_sepsis3.append(False)
            sepsis3_onset_h.append(float("nan"))
            baseline_sofa_v.append(0.0)
            peak_sofa_v.append(float("nan"))
            delta_sofa_v.append(float("nan"))
            continue

        hours = np.asarray(hours, dtype=float)
        vals  = np.asarray(vals,  dtype=float)

        # Baseline: median SOFA in [t_inf - baseline_window_h, t_inf)
        baseline_mask = (hours >= t_inf - baseline_window_h) & (hours < t_inf)
        baseline = float(np.nanmedian(vals[baseline_mask])) if baseline_mask.any() else 0.0

        # Peak SOFA within ±sofa_window_h of infection onset
        peak_mask = (hours >= t_inf - sofa_window_h) & (hours <= t_inf + sofa_window_h)
        if not peak_mask.any():
            has_sepsis3.append(False)
            sepsis3_onset_h.append(float("nan"))
            baseline_sofa_v.append(baseline)
            peak_sofa_v.append(float("nan"))
            delta_sofa_v.append(float("nan"))
            continue

        peak = float(np.nanmax(vals[peak_mask]))
        delta = peak - baseline

        if delta >= threshold:
            # First hour where SOFA ≥ baseline + threshold (within window)
            onset_mask = peak_mask & (vals >= baseline + threshold)
            onset_h = float(hours[onset_mask][0]) if onset_mask.any() else t_inf
            has_sepsis3.append(True)
            sepsis3_onset_h.append(onset_h)
        else:
            has_sepsis3.append(False)
            sepsis3_onset_h.append(float("nan"))

        baseline_sofa_v.append(baseline)
        peak_sofa_v.append(peak)
        delta_sofa_v.append(delta)

    result["has_sepsis3"]      = has_sepsis3
    result["sepsis3_onset_h"]  = sepsis3_onset_h
    result["baseline_sofa"]    = baseline_sofa_v
    result["peak_sofa"]        = peak_sofa_v
    result["delta_sofa"]       = delta_sofa_v

    n_sep = sum(has_sepsis3)
    n_sus = result["has_susinf"].sum()
    n_tot = len(result)
    print(f"sepsis3_per_stay_umm: {n_sep}/{n_tot} stays ({100*n_sep/n_tot:.1f}%) "
          f"Sepsis-3 positive | {n_sus} with suspected infection "
          f"| ΔSOFA threshold={threshold}")
    return result
