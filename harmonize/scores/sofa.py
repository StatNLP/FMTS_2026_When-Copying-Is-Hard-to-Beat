"""
harmonize/scores/sofa.py
SOFA score from a (T, F) feature matrix.

Implements the six standard SOFA components (each 0–4, total 0–24):

  1. Respiration   — PaO2/FiO2 ratio
  2. Coagulation   — Platelets (×10³/µL)
  3. Liver         — Bilirubin total (mg/dL)
  4. Cardiovascular — MAP + vasopressor dose (norepi / epi / dopamine / dobutamine)
  5. CNS           — GCS (sum of eye/motor/verbal, or `gcs_total` if present)
  6. Renal         — Creatinine (mg/dL) OR urine_output (mL / 24 h)

Inputs are expected in physical units. To call on Z-scored Stage-3 outputs,
first un-normalise via `train_mean + value * train_std` from the norm_stats
parquet for that dataset.

The functions are pure NumPy + accept both 2-D `(T, F)` and 3-D `(B, T, F)`.
Missing values (NaN) propagate as NaN in the per-component output; the total
is computed via `np.nansum` so a missing component just contributes 0.

References
----------
- Vincent et al., Intensive Care Med 1996 — the original SOFA definition (6 organ subscores, 0–4).
- Ferreira et al., JAMA 2001 ("Serial evaluation of the SOFA score") — daily SOFA from the *worst*
  value of each organ within the 24 h period.
- Singer et al., JAMA 2016 (Sepsis-3) — ΔSOFA ≥ 2 from baseline, worst values / 24 h.

Two window-level aggregations live here — used for *different* purposes, keep them distinct:
  • ``daily_sofa(x, feats, dataset)`` — the **textbook daily SOFA**: worst (max) subscore *per organ*
    over the window, summed → one severity scalar per window. This is the standard daily
    operationalisation (Vincent 1996 components; Ferreira 2001 / Sepsis-3 worst-value-per-24 h). Used
    for the SOFA regression head and the ΔSOFA2 deterioration label (worst-target − best-input);
    robust to missingness (``nanmax`` per organ needs no imputation).
  • ``_window_sofa(x, feats, dataset)`` — the **per-hour SOFA trajectory**: the SOFA *total at each
    hour* (so the input must be LOCF-imputed to populate all six systems), then h0 / h23 / Δ / mean /
    max. Used for the forecast read-off (does the forecast capture the SOFA *trajectory* over time?).
  Either is valid; the rule is to use the **same** one on *both* sides of a head-vs-forecast
  comparison (§10.4a / §R2). (Both moved here from the retired ``eval_sofa.py``.)

Caveats for our Stage-3 pipeline
--------------------------------
- Pressor doses (norepi, epi, dopamine, dobutamine) are expected in
  µg/kg/min. Confirm the per-dataset unit before trusting SOFA-CV scores.
- Forward-fill imputation (`--imputation fwd`) propagates a single
  pressor record across the whole window. For SOFA-CV this can inflate
  the cardiovascular score; consider scoring on raw `x_target` (no
  imputation) or on `nan` imputation when comparing to ground truth.
- The respiration cutoffs assume "with respiratory support" status is
  ignored (Paper-1 simplification — overestimates respiration score
  for non-vented patients).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import warnings
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Imputation helpers
# ---------------------------------------------------------------------------

def locf(x: np.ndarray, axis: int = -2, seed: np.ndarray | None = None) -> np.ndarray:
    """Last Observation Carried Forward along `axis`.

    NaN values are replaced by the most recent non-NaN value along that axis.
    Leading NaN values (no prior observation) remain NaN unless `seed` is given.

    Parameters
    ----------
    x : ndarray  shape (..., T, ...) — time axis is `axis`
    axis : int   time axis (default -2 for shape (..., T, F))
    seed : ndarray | None
        Initial carry array with the same shape as `x` but with the `axis`
        dimension removed.  Example: for `x` of shape `(B, T, F)` with
        ``axis=1``, `seed` should be `(B, F)`.  Typically the last time step
        of the preceding window (e.g. ``x_in[:, -1, :]`` denormalised).
        Provides the initial carry so that LOCF continues across the
        input→target boundary instead of treating the target window in
        isolation.

    Returns
    -------
    ndarray  same shape as `x`.  NaN only where no prior observation exists
    (leading NaN before any real measurement and before any seed value).

    Notes
    -----
    For SOFA-CV evaluation use ``seed = x_in_phys[:, -1, :]`` (mapped to
    target feature indices) so that a vasopressor present in the input window
    IS carried forward until explicitly superseded — but the carry represents
    the *last measured* value, not a model-predicted forward-fill.
    """
    out = np.array(x, dtype=np.float32, copy=True)
    out = np.moveaxis(out, axis, 0)   # (T, ...)

    carry = (np.asarray(seed, dtype=np.float32)
             if seed is not None
             else np.full_like(out[0], np.nan))

    for t in range(out.shape[0]):
        nan_mask = np.isnan(out[t])
        out[t]   = np.where(nan_mask, carry, out[t])
        # Update carry: keep carry where still NaN (no new obs), update where real value
        carry    = np.where(np.isnan(out[t]), carry, out[t])

    return np.moveaxis(out, 0, axis)


def fill_ffb(x: np.ndarray, mask: np.ndarray, axis: int = 1) -> np.ndarray:
    """Forward- then backward-fill observed values along the time `axis`, so a window
    has no leading/trailing NaN. Used to impute BOTH the input and the target window
    *symmetrically* before computing a daily SOFA — an asymmetric fill (e.g. seeded
    LOCF on one window, sparse on the other) biases the per-organ worst values."""
    obs = np.where(mask, x, np.nan).astype(np.float32)
    ff = locf(obs, axis=axis)
    bf = np.flip(locf(np.flip(ff, axis=axis), axis=axis), axis=axis)   # backward fill
    return bf


def daily_sofa(x_phys: np.ndarray, feature_names, dataset: str | None = None,
               skip_cns: bool = False) -> np.ndarray:
    """Daily SOFA = Σ over organs of the *worst* (max) component score within the window
    (the standard 24h SOFA, on the worst sub-value per organ). Robust to hourly noise,
    unlike max/last of the per-hour SOFA *total*. `x_phys` is a fully-imputed physical
    trajectory (…, T, F); returns (…,) one score per window. UMM uses 5 components."""
    c = sofa_components(x_phys, feature_names)
    # `skip_cns` forces the 5-component score for ANY dataset, so a GCS-free SOFA can be
    # tabulated across all five databases (UMM has no usable GCS, so the GCS-bearing table
    # covers four databases and the GCS-free one covers five).
    if dataset == "umm" or skip_cns:
        c.cns = np.zeros_like(c.cns)
    parts = [c.respiration, c.coagulation, c.liver, c.cardiovascular, c.cns, c.renal]
    with warnings.catch_warnings():                     # all-NaN organ → NaN (handled by caller)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        worst = [np.nanmax(p, axis=-1) for p in parts]  # per organ: worst over time
    return np.nansum(np.stack(worst, axis=-1), axis=-1)


#: which measured channels each SOFA component is computed FROM. A component is only meaningfully
#: evaluated at an hour where its own driver was measured -- elsewhere the LOCF ground truth is a
#: carried-forward value that a persistence forecast reproduces exactly, for free.
#: What each SOFA component is actually computed FROM, as `sofa_components` computes it. The mask
#: must mirror the scoring code, not a plausible list of related channels -- four ways that can go
#: wrong, all found by review on 2026-10-14 and all fixed here:
#:   * respiration is the PaO2/FiO2 *ratio*, so it needs BOTH, not either (HiRID: 58.0 % of hours
#:     under `any` against 7.65 % under `both`);
#:   * `sofa_components` resolves MAP as `where(isnan(invasive), noninvasive, invasive)` on an
#:     already-LOCF-filled array, so invasive is essentially never NaN and noninvasive is never
#:     reached -- ORing the two flags noninvasive-only hours as observed while the score reads a
#:     carried-forward invasive value;
#:   * `urine_output` is a per-window constant whose criterion never fires, so including it in the
#:     renal driver keeps hours whose creatinine is carried forward;
#:   * the total must require only the components the dataset actually scores -- UMM zeroes CNS,
#:     so demanding it cut scorable hours 11x (1.11 % to 0.098 %).
SOFA_COMPONENT_DRIVERS = {
    "respiration":    (("pao2", "fio2"),),                 # tuple-of-tuples => ALL of the inner set
    "coagulation":    ("platelets",),
    "liver":          ("bilirubin_total",),
    "cardiovascular": ("mbp", "mbp_invasive", "mbp_noninvasive"),   # resolved below, not ORed
    "cns":            ("gcs_eye", "gcs_motor", "gcs_verbal", "gcs_total"),
    "renal":          ("creatinine",),
}
SOFA_COMPONENT_ORDER = ("respiration", "coagulation", "liver", "cardiovascular", "cns", "renal")
#: components a dataset does not score, so the all-observed total must not demand them
SOFA_SKIP_COMPONENTS = {"umm": ("cns",)}


def sofa_hour_mask(mask_target, features_target, dataset: str | None = None) -> np.ndarray:
    """Per-component observation mask, shape ``(N, T, 6)`` in `SOFA_COMPONENT_ORDER`.

    Plane *c* is True at the hours where the inputs component *c* is actually scored from were
    measured. See `SOFA_COMPONENT_DRIVERS` for why this mirrors the scoring code rather than
    listing related channels.
    """
    names = list(features_target)
    m = np.asarray(mask_target).astype(bool)
    out = np.zeros(tuple(m.shape[:2]) + (6,), dtype=bool)
    idx = lambda f: names.index(f) if f in names else None

    for c, comp in enumerate(SOFA_COMPONENT_ORDER):
        if comp == "respiration":                       # the ratio needs both terms
            a, b = idx("pao2"), idx("fio2")
            if a is not None and b is not None:
                out[..., c] = m[..., a] & m[..., b]
        elif comp == "cardiovascular":
            # Mirror `where(isnan(invasive), noninvasive, invasive)` as it behaves on the
            # LOCF-filled array the scorer actually sees: invasive is non-NaN from the first hour
            # the stay has an invasive reading onward, so the scorer reads invasive from that point
            # and noninvasive before it. Keying on invasive alone would restrict to the
            # arterial-line subpopulation -- 74.6 % of eICU windows have no invasive MAP at all,
            # and 57.6 % of MIMIC-IV -- which is a sicker cohort, not a random subset.
            k = idx("mbp")
            if k is not None:
                out[..., c] = m[..., k]
            else:
                ki, kn = idx("mbp_invasive"), idx("mbp_noninvasive")
                if ki is not None and kn is not None:
                    use_inv = np.maximum.accumulate(m[..., ki], axis=1)   # invasive seen at or before t
                    out[..., c] = np.where(use_inv, m[..., ki], m[..., kn])
                elif ki is not None:
                    out[..., c] = m[..., ki]
                elif kn is not None:
                    out[..., c] = m[..., kn]
        elif comp == "cns":
            # With no `gcs_total` channel the scorer SUMS the three subscales, so all three are
            # needed -- same defect class as the respiration ratio.
            kt = idx("gcs_total")
            if kt is not None:
                out[..., c] = m[..., kt]
            else:
                ks = [idx(f) for f in ("gcs_eye", "gcs_motor", "gcs_verbal")]
                if all(k is not None for k in ks):
                    out[..., c] = m[..., ks].all(axis=-1)
        else:
            ks = [idx(f) for f in SOFA_COMPONENT_DRIVERS[comp]]
            ks = [k for k in ks if k is not None]
            if ks:
                out[..., c] = m[..., ks].any(axis=-1)
    return out


def sofa_total_mask(hour_mask, dataset: str | None = None) -> np.ndarray:
    """Hours where every component the dataset *scores* rests on a real measurement."""
    hm = np.asarray(hour_mask, dtype=bool)
    skip = SOFA_SKIP_COMPONENTS.get((dataset or "").lower(), ())
    keep = [i for i, c in enumerate(SOFA_COMPONENT_ORDER) if c not in skip]
    return hm[..., keep].all(axis=-1)


def _nanagg(fn, arr):
    """nanmean/nanmax over the time axis, returning NaN (not a warning) for all-NaN rows —
    which is what an all-carried-forward window becomes under an hour mask."""
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    ok = ~np.all(np.isnan(arr), axis=1)
    if ok.any():
        out[ok] = fn(arr[ok], axis=1)
    return out


def _window_sofa(x_target_phys: np.ndarray, features_target, dataset: str,
                 hour_mask: np.ndarray | None = None,
                 skip_cns: bool = False) -> "pd.DataFrame":
    """Per-hour SOFA **trajectory** metrics, one row per window (moved here from eval_sofa.py).
    Expects LOCF-imputed physical-unit input (`(N, T_out, F)`) so the per-hour total covers all six
    systems. Returns h0 / h23 / Δ / mean / max of the SOFA total + per-component means. UMM uses the
    5-component SOFA (CNS excluded). Contrast `daily_sofa` (worst-per-organ scalar); see module docstring."""
    comps = sofa_components(x_target_phys, features_target)
    # UMM: 5-component SOFA (CNS/GCS excluded — deep-sedation protocol makes GCS unreliable).
    # `skip_cns` extends that to any dataset, for the GCS-free cross-database table.
    if dataset == "umm" or skip_cns:
        comps.cns = np.zeros_like(comps.cns)
    total = comps.total()                       # (N, T_out)
    if hour_mask is not None:
        # Blank each component at the hours where its own driver was not measured; the nan-aware
        # aggregations below then skip them. The TOTAL is scored only where every component is
        # genuinely observed -- the one honest definition for a sum, and deliberately strict.
        hm = np.asarray(hour_mask, dtype=bool)
        fields = {k: v for k, v in comps.__dict__.items()}
        for c, name in enumerate(SOFA_COMPONENT_ORDER):
            if name in fields and getattr(fields[name], "ndim", 0) == 2:
                fields[name] = np.where(hm[..., c], fields[name], np.nan)
        comps = comps.__class__(**fields)
        total = np.where(sofa_total_mask(hm, dataset), total, np.nan)
    sofa_h0, sofa_h23 = total[:, 0], total[:, -1]
    rows = {
        "sofa_h0":      sofa_h0,
        "sofa_h23":     sofa_h23,
        "delta_sofa":   sofa_h23 - sofa_h0,     # ΔSOFA over the 24 h window
        "sofa_mean":    _nanagg(np.nanmean, total),
        "sofa_max":     _nanagg(np.nanmax, total),
        "sofa_cv_mean": _nanagg(np.nanmean, comps.cardiovascular),
    }
    for name, arr in [("sofa_resp", comps.respiration), ("sofa_coag", comps.coagulation),
                      ("sofa_liver", comps.liver), ("sofa_cns", comps.cns),
                      ("sofa_renal", comps.renal)]:
        if name == "sofa_cns" and dataset == "umm":
            rows[name] = np.full(len(x_target_phys), np.nan, dtype=np.float32)
        else:
            rows[name] = _nanagg(np.nanmean, arr)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cutoff tables (component value → 0..4 score)
# ---------------------------------------------------------------------------
# All cutoffs are inclusive of the lower bound except where noted.
# Component returns 4 (worst) if value is ≤ first cut, 0 (best) if ≥ last.
# `_score_descending` is for "lower is worse" features (PaO2/FiO2, platelets, GCS).
# `_score_ascending`  is for "higher is worse" features (bilirubin, creatinine).


def _score_descending(x: np.ndarray, cuts: list[float]) -> np.ndarray:
    """Score for "lower is worse": returns 4 if x < cuts[0], 3 if x < cuts[1],
    …, 0 if x ≥ cuts[-1]. Length(cuts) = 4 (separates the 5 buckets)."""
    out = np.zeros_like(x, dtype=np.float32)
    out[x <  cuts[3]] = 1
    out[x <  cuts[2]] = 2
    out[x <  cuts[1]] = 3
    out[x <  cuts[0]] = 4
    out[np.isnan(x)] = np.nan
    return out


def _score_ascending(x: np.ndarray, cuts: list[float]) -> np.ndarray:
    """Score for "higher is worse": returns 0 if x < cuts[0], 1 if x < cuts[1],
    …, 4 if x ≥ cuts[3]."""
    out = np.zeros_like(x, dtype=np.float32)
    out[x >= cuts[0]] = 1
    out[x >= cuts[1]] = 2
    out[x >= cuts[2]] = 3
    out[x >= cuts[3]] = 4
    out[np.isnan(x)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Per-component scorers
# ---------------------------------------------------------------------------

def respiration(pao2: np.ndarray, fio2: np.ndarray) -> np.ndarray:
    """PaO2/FiO2 ratio. Cutoffs: 400, 300, 200, 100. We don't have
    ventilator-status here, so "with respiratory support" requirement
    for 3 and 4 is ignored (overestimates by 0–2 points for non-vented
    patients — Paper-1 makes the same simplification)."""
    fio2 = np.where(fio2 < 0.01, np.nan, fio2)  # avoid /0
    pf = pao2 / fio2
    return _score_descending(pf, [100, 200, 300, 400])


def coagulation(platelets: np.ndarray) -> np.ndarray:
    """Platelets in 10³/µL (== 'K/µL'). Cutoffs: 150, 100, 50, 20."""
    return _score_descending(platelets, [20, 50, 100, 150])


def liver(bilirubin_total: np.ndarray) -> np.ndarray:
    """Total bilirubin in mg/dL. Cutoffs: 1.2, 2.0, 6.0, 12.0."""
    return _score_ascending(bilirubin_total, [1.2, 2.0, 6.0, 12.0])


def cardiovascular(
    mbp: np.ndarray,
    dopamine:       np.ndarray | None = None,
    dobutamine:     np.ndarray | None = None,
    epinephrine:    np.ndarray | None = None,
    norepinephrine: np.ndarray | None = None,
) -> np.ndarray:
    """MAP + vasopressor dose (µg/kg/min for catecholamines).
    Score (worst applicable applies):
        0  MAP ≥ 70 and no pressors
        1  MAP < 70
        2  dopamine ≤ 5    OR dobutamine > 0
        3  dopamine > 5    OR (epinephrine ≤ 0.1 OR norepinephrine ≤ 0.1)
        4  dopamine > 15   OR (epinephrine > 0.1 OR norepinephrine > 0.1)
    """
    score = np.zeros_like(mbp, dtype=np.float32)
    score = np.where(mbp < 70, np.maximum(score, 1), score)

    def _g(x, default=0.0):
        return x if x is not None else np.full_like(mbp, default, dtype=np.float32)

    dop = _g(dopamine);  dob = _g(dobutamine)
    epi = _g(epinephrine); nor = _g(norepinephrine)

    has2 = (dop > 0) & (dop <= 5) | (dob > 0)
    score = np.where(has2, np.maximum(score, 2), score)

    has3 = (dop > 5) & (dop <= 15) | ((epi > 0) & (epi <= 0.1)) | ((nor > 0) & (nor <= 0.1))
    score = np.where(has3, np.maximum(score, 3), score)

    has4 = (dop > 15) | (epi > 0.1) | (nor > 0.1)
    score = np.where(has4, np.maximum(score, 4), score)

    score[np.isnan(mbp)] = np.nan
    return score


def cns(gcs_total: np.ndarray) -> np.ndarray:
    """GCS sum. Cutoffs: 15, 13, 10, 6.
        0 if GCS == 15
        1 if 13–14
        2 if 10–12
        3 if 6–9
        4 if < 6
    """
    out = np.zeros_like(gcs_total, dtype=np.float32)
    out[gcs_total <  15] = 1
    out[gcs_total <  13] = 2
    out[gcs_total <  10] = 3
    out[gcs_total <   6] = 4
    out[np.isnan(gcs_total)] = np.nan
    return out


def renal(
    creatinine: np.ndarray,
    urine_output_24h: np.ndarray | None = None,
) -> np.ndarray:
    """Creatinine mg/dL primarily; urine_output (mL/24h) raises to 3 or 4 if
    very low. Worst applicable applies.
        0  creat <  1.2
        1  creat <  2.0
        2  creat <  3.5
        3  creat <  5.0  OR urine < 500 mL/24h
        4  creat >= 5.0  OR urine < 200 mL/24h
    """
    out = np.zeros_like(creatinine, dtype=np.float32)
    out[creatinine >= 1.2] = 1
    out[creatinine >= 2.0] = 2
    out[creatinine >= 3.5] = 3
    out[creatinine >= 5.0] = 4
    if urine_output_24h is not None:
        out = np.where(urine_output_24h < 500, np.maximum(out, 3), out)
        out = np.where(urine_output_24h < 200, np.maximum(out, 4), out)
    out[np.isnan(creatinine)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

@dataclass
class SOFAComponents:
    respiration:    np.ndarray
    coagulation:    np.ndarray
    liver:          np.ndarray
    cardiovascular: np.ndarray
    cns:            np.ndarray
    renal:          np.ndarray

    def total(self) -> np.ndarray:
        parts = [self.respiration, self.coagulation, self.liver,
                 self.cardiovascular, self.cns, self.renal]
        return np.nansum(np.stack(parts, axis=-1), axis=-1).astype(np.float32)


def _pick(x: np.ndarray, names: Sequence[str], wanted: str) -> np.ndarray | None:
    """Return the slice for `wanted` if present, else None.
    Works for both 2-D (T, F) and 3-D (B, T, F)."""
    if wanted not in names:
        return None
    j = list(names).index(wanted)
    return x[..., j]


def sofa_components(x: np.ndarray, feature_names: Sequence[str],
                    urine_is_24h_total: bool = False) -> SOFAComponents:
    """Compute the six SOFA components from a physical-unit feature matrix.

    `urine_is_24h_total`: False (default) means the second-to-last axis is TIME and
    `urine_output` holds per-hour values that must be summed across it to get the 24 h
    quantity the renal cutoffs expect. True means `urine_output` is ALREADY a 24 h total
    (one value per row), as in a daily (n_days, F) table. The caller has to say which,
    because the two cases are indistinguishable from the array: a single window (T, F)
    and a daily table (n_days, F) are both 2-D with a 1-D urine column.

    `x` shape: (..., F) where F = len(feature_names). Last axis indexes
    features by name. Time axis (if present) is just along for the ride —
    cutoffs are applied element-wise.

    Missing components yield NaN-arrays of the right shape. `total()` sums
    via nansum (so a missing component just contributes 0).
    """
    pao2  = _pick(x, feature_names, "pao2")
    fio2  = _pick(x, feature_names, "fio2")
    plt_  = _pick(x, feature_names, "platelets")
    bili  = _pick(x, feature_names, "bilirubin_total")
    # mbp: prefer canonical, fall back to invasive → non-invasive (MIMIC-III)
    mbp = _pick(x, feature_names, "mbp")
    if mbp is None:
        mbp_inv = _pick(x, feature_names, "mbp_invasive")
        mbp_ni  = _pick(x, feature_names, "mbp_noninvasive")
        if mbp_inv is not None and mbp_ni is not None:
            mbp = np.where(np.isnan(mbp_inv), mbp_ni, mbp_inv)
        else:
            mbp = mbp_inv if mbp_inv is not None else mbp_ni
    dop   = _pick(x, feature_names, "dopamine")
    dob   = _pick(x, feature_names, "dobutamine")
    epi   = _pick(x, feature_names, "epinephrine")
    nor   = _pick(x, feature_names, "norepinephrine")
    creat = _pick(x, feature_names, "creatinine")
    uo    = _pick(x, feature_names, "urine_output")
    # Urine is a 24h quantity, not an hourly rate. The per-hour value (~always < 500 mL)
    # must NOT be tested against the 500 mL/24h threshold directly — that spuriously
    # pins the renal score high where urine is charted densely (UMM: 97.5% at SOFA 3–4)
    # and makes it swing where urine is sparse (the dominant hourly artefact). Instead
    # sum the per-hour urine across the window (missing hours contribute 0, per the
    # clinical 24h-total semantics) and broadcast that single total back over time so the
    # urine criterion is constant within the window. If NO urine was charted at all, leave
    # it undefined (NaN) so the criterion is skipped (creatinine-only), rather than
    # treating "no chart" as anuria.
    # This summation is correct only when the axis being summed is TIME. Both SOFA builders
    # (keepgate/mimic_targets.py, keepgate/01_preprocess_gtsq.py) pass an already-daily
    # (n_days, F) table, so `uo` arrived 1-D, the sum ran over DAYS, and every row received the
    # cohort-wide urine total -- always >= 500, so the urine criterion never fired and renal fell
    # back to creatinine alone (a 3-row case scored [0,0,0] where [4,0,3] is correct; on UMM it
    # changes renal on 15.7% of 64,256 days, mean renal 0.580 -> 0.998).
    #
    # The fix recorded on 2026-08-10 -- gate on `np.ndim(uo) >= 2` -- is NOT safe: stage3
    # (stage3_make_dataset.py:517) scores ONE window at a time, so its `x` is (T_out, F) and its
    # `uo` is 1-D exactly like a daily table's. That gate would have fixed the builders and
    # silently stopped summing in stage3, reinstating the per-hour artefact the block exists to
    # prevent (UMM: 97.5% of windows pinned at renal 3-4) and changing every `sofa.*` field in
    # the published eval tables. The shapes cannot be told apart, so the CALLER declares it.
    if uo is not None and not urine_is_24h_total:
        _has = np.isfinite(uo).any(axis=-1, keepdims=True)
        _tot = np.nansum(uo, axis=-1, keepdims=True)
        uo = np.broadcast_to(np.where(_has, _tot, np.nan), uo.shape).copy()

    # GCS: prefer single gcs_total, else sum the three sub-scales
    gcs_total = _pick(x, feature_names, "gcs_total")
    if gcs_total is None:
        ge = _pick(x, feature_names, "gcs_eye")
        gm = _pick(x, feature_names, "gcs_motor")
        gv = _pick(x, feature_names, "gcs_verbal")
        if ge is not None and gm is not None and gv is not None:
            gcs_total = ge + gm + gv

    template = next(
        a for a in (pao2, plt_, bili, mbp, creat, gcs_total) if a is not None
    )
    nan_like = np.full_like(template, np.nan, dtype=np.float32)

    return SOFAComponents(
        respiration    = respiration(pao2, fio2)            if (pao2 is not None and fio2 is not None) else nan_like.copy(),
        coagulation    = coagulation(plt_)                  if plt_  is not None else nan_like.copy(),
        liver          = liver(bili)                        if bili  is not None else nan_like.copy(),
        cardiovascular = cardiovascular(mbp, dop, dob, epi, nor) if mbp is not None else nan_like.copy(),
        cns            = cns(gcs_total)                     if gcs_total is not None else nan_like.copy(),
        renal          = renal(creat, uo)                   if creat is not None else nan_like.copy(),
    )


def sofa_from_matrix(x: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Total SOFA score from a (..., F) physical-unit feature matrix."""
    return sofa_components(x, feature_names).total()


def denormalise(z: np.ndarray, feature_names: Sequence[str],
                norm_stats) -> np.ndarray:
    """Convert Z-scored Stage-3 features back to physical units.

    `norm_stats` is a pandas DataFrame indexed by `variable` with columns
    `train_mean` and `train_std` (as written by stage2_preprocessing).
    Returns an array of the same shape as `z`.
    """
    # Fail loudly on a frame that is not indexed by `variable`. Passing the raw parquet (which
    # has `variable` as a COLUMN) matched nothing and silently returned the z-scores unchanged, so
    # every downstream threshold -- MAP < 70, platelets < 150, dopamine > 0 -- was applied to
    # standardised values. That produced a median admission SOFA of 17/24 and went unnoticed
    # through a full analysis cycle (2026-10-19). Silence is the defect; the check is cheap.
    if "train_mean" not in getattr(norm_stats, "columns", []):
        raise ValueError("denormalise: norm_stats needs a 'train_mean' column")
    hit = sum(1 for n in feature_names if n in norm_stats.index)
    if hit == 0:
        raise ValueError(
            "denormalise: no feature matched norm_stats.index -- the frame is almost certainly "
            "not indexed by 'variable'. Load it as pd.read_parquet(p).set_index('variable').")
    out = np.array(z, copy=True, dtype=np.float32)
    for j, name in enumerate(feature_names):
        if name in norm_stats.index:
            mu = float(norm_stats.loc[name, "train_mean"])
            sd = float(norm_stats.loc[name, "train_std"])
            out[..., j] = out[..., j] * sd + mu
    return out
