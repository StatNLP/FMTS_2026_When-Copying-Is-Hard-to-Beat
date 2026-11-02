"""
harmonize/scores/sapsii.py
SAPS-II score from a (T, F) feature matrix.

Reference
---------
- Le Gall et al., JAMA 1993. "A new Simplified Acute Physiology Score
  (SAPS II) based on a European/North American multicenter study."

SAPS-II is a *single* number per stay (worst values over the first 24 h
in the ICU). Total points 0..163, mapped to a mortality probability via

    logit = -7.7631 + 0.0737 * SAPS + 0.9971 * ln(SAPS + 1)
    p_mort = 1 / (1 + exp(-logit))

We compute the score from a (T, F) physical-unit matrix by taking the
*worst-applicable* value per variable across the time axis.

Caveats for our pipeline
------------------------
- "Chronic disease" (metastatic cancer / hematologic malignancy / AIDS)
  and "type of admission" (elective surgical / medical / unscheduled
  surgical) are not in our catalog → those points are taken as 0.
  Literature: this drops AUROC by ~0.05–0.07 vs full SAPS-II. Same
  simplification as Paper-1.
- The PaO2/FiO2 component contributes only "if mechanically ventilated".
  We don't have a reliable vent flag for all datasets, so we apply the
  PaO2/FiO2 score *whenever both PaO2 and FiO2 are recorded*. For
  spontaneously breathing patients on room air this still yields 0
  (PaO2/FiO2 ≥ 200), but vented patients with mild hypoxemia will be
  slightly over-counted. Conservative simplification.
- Urine output: SAPS-II expects mL/24h. Our `urine_output` feature is
  in mL/h; we sum over the window to get the daily total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Per-variable scoring tables. Each scorer takes a scalar (worst value over
# the window) and returns 0..NN points.
# ---------------------------------------------------------------------------

def _bins(value: float, edges: list[tuple[float, float, int]]) -> int:
    """Return points for the first matching (lo_inclusive, hi_exclusive, pts)
    bin. `np.nan` returns 0 (variable missing → no penalty)."""
    if value is None or np.isnan(value):
        return 0
    for lo, hi, pts in edges:
        if lo <= value < hi:
            return pts
    return 0


# Age (static, points): <40 → 0; 40-59 → 7; 60-69 → 12; 70-74 → 15; 75-79 → 16; ≥80 → 18
def age_pts(age_years: float) -> int:
    return _bins(age_years, [
        (-np.inf, 40, 0), (40, 60, 7), (60, 70, 12),
        (70, 75, 15), (75, 80, 16), (80, np.inf, 18),
    ])


# Heart rate (worst): <40 → 11; 40-69 → 2; 70-119 → 0; 120-159 → 4; ≥160 → 7
# "Worst" means the value with the highest points — apply to min AND max.
def hr_pts(hr_min: float, hr_max: float) -> int:
    low_pts  = _bins(hr_min, [(-np.inf, 40, 11), (40, 70, 2), (70, np.inf, 0)])
    high_pts = _bins(hr_max, [(120, 160, 4), (160, np.inf, 7), (-np.inf, 120, 0)])
    return max(low_pts, high_pts)


def sbp_pts(sbp_min: float, sbp_max: float) -> int:
    low_pts  = _bins(sbp_min, [(-np.inf, 70, 13), (70, 100, 5), (100, np.inf, 0)])
    high_pts = _bins(sbp_max, [(200, np.inf, 2), (-np.inf, 200, 0)])
    return max(low_pts, high_pts)


def temp_pts(temp_max: float) -> int:
    return _bins(temp_max, [(39, np.inf, 3), (-np.inf, 39, 0)])


def pao2_fio2_pts(pao2_min: float, fio2_when_low: float) -> int:
    """Compute worst PaO2/FiO2 ratio. Caller is responsible for taking the
    PaO2 minimum and the *paired* FiO2 at that timestamp (or the FiO2 max
    over the window as a conservative substitute)."""
    if np.isnan(pao2_min) or np.isnan(fio2_when_low) or fio2_when_low < 0.01:
        return 0
    pf = pao2_min / fio2_when_low
    return _bins(pf, [(-np.inf, 100, 11), (100, 200, 9), (200, np.inf, 6)])


def urine_24h_pts(urine_total_24h: float) -> int:
    return _bins(urine_total_24h, [
        (-np.inf, 500, 11), (500, 1000, 4), (1000, np.inf, 0),
    ])


def bun_pts(bun_max: float) -> int:
    """BUN in mg/dL. Le Gall uses urea mmol/L: <10 → 0; 10-29 → 6; ≥30 → 10.
    BUN mg/dL = urea mmol/L × 2.8 → cutoffs 28 and 84 mg/dL."""
    return _bins(bun_max, [(-np.inf, 28, 0), (28, 84, 6), (84, np.inf, 10)])


def wbc_pts(wbc_min: float, wbc_max: float) -> int:
    low_pts  = _bins(wbc_min, [(-np.inf, 1.0, 12), (1.0, np.inf, 0)])
    high_pts = _bins(wbc_max, [(20.0, np.inf, 3), (-np.inf, 20.0, 0)])
    return max(low_pts, high_pts)


def potassium_pts(k_min: float, k_max: float) -> int:
    low_pts  = _bins(k_min, [(-np.inf, 3.0, 3), (3.0, np.inf, 0)])
    high_pts = _bins(k_max, [(5.0, np.inf, 3), (-np.inf, 5.0, 0)])
    return max(low_pts, high_pts)


def sodium_pts(na_min: float, na_max: float) -> int:
    low_pts  = _bins(na_min, [(-np.inf, 125, 5), (125, np.inf, 0)])
    high_pts = _bins(na_max, [(145, np.inf, 1), (-np.inf, 145, 0)])
    return max(low_pts, high_pts)


def bicarbonate_pts(hco3_min: float) -> int:
    return _bins(hco3_min, [(-np.inf, 15, 6), (15, 20, 3), (20, np.inf, 0)])


def bilirubin_pts(bili_max: float) -> int:
    return _bins(bili_max, [(-np.inf, 4, 0), (4, 6, 4), (6, np.inf, 9)])


def gcs_pts(gcs_min: float) -> int:
    return _bins(gcs_min, [
        (-np.inf, 6, 26), (6, 9, 13), (9, 11, 7),
        (11, 14, 5), (14, np.inf, 0),
    ])


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

@dataclass
class SAPSIIComponents:
    age: int
    hr: int
    sbp: int
    temp: int
    pao2_fio2: int
    urine_24h: int
    bun: int
    wbc: int
    potassium: int
    sodium: int
    bicarbonate: int
    bilirubin: int
    gcs: int
    # We can't fill these without admission-type / chronic-disease data
    chronic_disease: int = 0
    admission_type:  int = 0

    def total(self) -> int:
        return (
            self.age + self.hr + self.sbp + self.temp + self.pao2_fio2
            + self.urine_24h + self.bun + self.wbc + self.potassium
            + self.sodium + self.bicarbonate + self.bilirubin + self.gcs
            + self.chronic_disease + self.admission_type
        )

    def mortality_prob(self) -> float:
        s = self.total()
        logit = -7.7631 + 0.0737 * s + 0.9971 * np.log(s + 1)
        return float(1.0 / (1.0 + np.exp(-logit)))


def _pick(x: np.ndarray, names: Sequence[str], wanted: str) -> np.ndarray | None:
    if wanted not in names:
        return None
    j = list(names).index(wanted)
    return x[..., j]


def _nan_min(a: np.ndarray | None) -> float:
    if a is None:
        return np.nan
    if np.all(np.isnan(a)):
        return np.nan
    return float(np.nanmin(a))


def _nan_max(a: np.ndarray | None) -> float:
    if a is None:
        return np.nan
    if np.all(np.isnan(a)):
        return np.nan
    return float(np.nanmax(a))


def sapsii_components(x: np.ndarray, feature_names: Sequence[str],
                      age_years: float | None = None,
                      admission_type_pts: float | None = None,
                      chronic_disease_pts: float | None = None) -> SAPSIIComponents:
    """SAPS-II from one stay's (T, F) physical-unit matrix.

    Worst-of-window applied per variable. `age_years`, `admission_type_pts` and
    `chronic_disease_pts` are static features passed separately.
    `admission_type_pts` should be 0 (scheduled surgical), 6 (medical),
    or 8 (unscheduled surgical) per Le Gall 1993; `chronic_disease_pts` is the
    SAPS-II chronic-disease component (0 / 9 metastatic cancer / 10 haematologic
    malignancy / 17 AIDS), from the Charlson extraction (§7.4).
    """
    hr   = _pick(x, feature_names, "hr")
    # sbp: prefer canonical, fall back to invasive → non-invasive. The model
    # tensor only carries the sbp_invasive/sbp_noninvasive variants (never a
    # merged `sbp`), so without this fallback the SBP component silently scored
    # 0 for every stay (matches sofa.py's mbp handling).
    sbp = _pick(x, feature_names, "sbp")
    if sbp is None:
        sbp_inv = _pick(x, feature_names, "sbp_invasive")
        sbp_ni  = _pick(x, feature_names, "sbp_noninvasive")
        if sbp_inv is not None and sbp_ni is not None:
            sbp = np.where(np.isnan(sbp_inv), sbp_ni, sbp_inv)
        else:
            sbp = sbp_inv if sbp_inv is not None else sbp_ni
    # temp: the tensor charts core temperature as `temp_core`, never a bare `temp`,
    # so without this fallback the temperature component silently scored 0 everywhere.
    temp = _pick(x, feature_names, "temp")
    if temp is None:
        temp = _pick(x, feature_names, "temp_core")
    pao2 = _pick(x, feature_names, "pao2")
    fio2 = _pick(x, feature_names, "fio2")
    uo   = _pick(x, feature_names, "urine_output")
    bun  = _pick(x, feature_names, "bun")
    wbc  = _pick(x, feature_names, "wbc")
    k    = _pick(x, feature_names, "potassium")
    na   = _pick(x, feature_names, "sodium")
    # hco3: the tensor charts serum bicarbonate as `hco3`, not `bicarbonate`,
    # so without this fallback the bicarbonate component silently scored 0 everywhere.
    hco3 = _pick(x, feature_names, "bicarbonate")
    if hco3 is None:
        hco3 = _pick(x, feature_names, "hco3")
    bili = _pick(x, feature_names, "bilirubin_total")

    # GCS: prefer single gcs_total else sum the three
    gcs_total = _pick(x, feature_names, "gcs_total")
    if gcs_total is None:
        ge = _pick(x, feature_names, "gcs_eye")
        gm = _pick(x, feature_names, "gcs_motor")
        gv = _pick(x, feature_names, "gcs_verbal")
        if ge is not None and gm is not None and gv is not None:
            gcs_total = ge + gm + gv

    # PaO2/FiO2: use lowest PaO2 with highest FiO2 (conservative substitute
    # for "paired sample" when we can't easily align them)
    pao2_min = _nan_min(pao2)
    fio2_max = _nan_max(fio2)
    urine_24h = float(np.nansum(uo)) if uo is not None else np.nan

    adm_pts = int(round(admission_type_pts)) if admission_type_pts is not None else 0
    chr_pts = int(round(chronic_disease_pts)) if chronic_disease_pts is not None else 0
    return SAPSIIComponents(
        age            = age_pts(float(age_years)) if age_years is not None else 0,
        chronic_disease = chr_pts,
        hr             = hr_pts(_nan_min(hr), _nan_max(hr)),
        sbp            = sbp_pts(_nan_min(sbp), _nan_max(sbp)),
        temp           = temp_pts(_nan_max(temp)),
        pao2_fio2      = pao2_fio2_pts(pao2_min, fio2_max),
        urine_24h      = urine_24h_pts(urine_24h),
        bun            = bun_pts(_nan_max(bun)),
        wbc            = wbc_pts(_nan_min(wbc), _nan_max(wbc)),
        potassium      = potassium_pts(_nan_min(k), _nan_max(k)),
        sodium         = sodium_pts(_nan_min(na), _nan_max(na)),
        bicarbonate    = bicarbonate_pts(_nan_min(hco3)),
        bilirubin      = bilirubin_pts(_nan_max(bili)),
        gcs            = gcs_pts(_nan_min(gcs_total)),
        admission_type = adm_pts,
    )


def sapsii_from_matrix(x: np.ndarray, feature_names: Sequence[str],
                       age_years: float | None = None,
                       admission_type_pts: float | None = None,
                       chronic_disease_pts: float | None = None) -> int:
    """Return total SAPS-II score (0..163). `admission_type_pts` adds 0/6/8 and
    `chronic_disease_pts` adds 0/9/10/17 points (both ordinal SAPS-II components)."""
    return sapsii_components(x, feature_names, age_years,
                             admission_type_pts, chronic_disease_pts).total()


def mortality_probability(sapsii_score: float) -> float:
    """Le Gall logistic: p_mort = sigmoid(-7.7631 + 0.0737*S + 0.9971*ln(S+1))."""
    logit = -7.7631 + 0.0737 * sapsii_score + 0.9971 * np.log(sapsii_score + 1)
    return float(1.0 / (1.0 + np.exp(-logit)))


def sapsii_batch(x: np.ndarray, feature_names: Sequence[str],
                 age_years: np.ndarray | None = None,
                 admission_type_pts: np.ndarray | None = None,
                 chronic_disease_pts: np.ndarray | None = None) -> np.ndarray:
    """Vectorise over a (B, T, F) tensor → (B,) score array. Loops over the
    batch dim (SAPS-II logic is per-stay so vector-friendly = not worth it).
    """
    if x.ndim == 2:
        x = x[None]
    B = x.shape[0]
    out = np.zeros(B, dtype=np.float32)
    for i in range(B):
        a   = float(age_years[i])          if age_years          is not None else None
        adm = float(admission_type_pts[i]) if admission_type_pts is not None else None
        chr_ = float(chronic_disease_pts[i]) if chronic_disease_pts is not None else None
        out[i] = sapsii_from_matrix(x[i], feature_names, a, adm, chr_)
    return out


SAPSII_COMPONENT_FIELDS = (
    "age", "hr", "sbp", "temp", "pao2_fio2", "urine_24h", "bun", "wbc",
    "potassium", "sodium", "bicarbonate", "bilirubin", "gcs",
    "chronic_disease", "admission_type",
)

#: The 12 window-scored physiology components, in `SAPSII_COMPONENT_FIELDS` order.
#: `age` / `chronic_disease` / `admission_type` are per-stay statics and are not window-scored.
SAPSII_PHYSIO_COMPONENTS = (
    "hr", "sbp", "temp", "pao2_fio2", "urine_24h", "bun", "wbc",
    "potassium", "sodium", "bicarbonate", "bilirubin", "gcs",
)


def sapsii_window_mask(mask_target, features_target) -> np.ndarray:
    """Per-component availability per window, shape ``(N, 12)`` in `SAPSII_PHYSIO_COMPONENTS`.

    Column *c* is True where component *c* rests on at least one real measurement inside the
    window. SAPS-II is scored worst-of-window, so availability is a per-WINDOW question — unlike
    `sofa.sofa_hour_mask`, which is per hour because SOFA is also read hour by hour.

    This exists because a missing SAPS-II variable scores **0 points, not NaN** (see `_bin_points`),
    so the total is always finite and gives no hint how many components contributed. Three
    components silently scored 0 for every stay in the past because the tensor names them
    differently (`sbp_*`, `temp_core`, `hco3`); publishing coverage is what makes that visible
    (§37.38 D20). The driver resolution below mirrors `sapsii_components` rather than listing
    related channels, so the mask says what was actually *scorable*.
    """
    names = list(features_target)
    m = np.asarray(mask_target).astype(bool)
    out = np.zeros((m.shape[0], len(SAPSII_PHYSIO_COMPONENTS)), dtype=bool)
    idx = lambda f: names.index(f) if f in names else None
    any_t = lambda k: m[:, :, k].any(axis=1)

    for c, comp in enumerate(SAPSII_PHYSIO_COMPONENTS):
        if comp == "pao2_fio2":                       # the ratio needs both terms
            a, b = idx("pao2"), idx("fio2")
            if a is not None and b is not None:
                out[:, c] = any_t(a) & any_t(b)
        elif comp == "sbp":
            # Mirror `where(isnan(sbp_inv), sbp_ni, sbp_inv)` on the filled array: invasive is
            # used from the first hour it appears onward, non-invasive before that. Keying on
            # invasive alone would restrict to the arterial-line subpopulation (cf. sofa.py).
            k = idx("sbp")
            if k is not None:
                out[:, c] = any_t(k)
            else:
                ki, kn = idx("sbp_invasive"), idx("sbp_noninvasive")
                ks = [x for x in (ki, kn) if x is not None]
                if ks:
                    out[:, c] = np.logical_or.reduce([any_t(k) for k in ks])
        elif comp == "gcs":
            # With no `gcs_total` channel the scorer SUMS the three subscales, so all three
            # are needed — same defect class as the PaO2/FiO2 ratio.
            kt = idx("gcs_total")
            if kt is not None:
                out[:, c] = any_t(kt)
            else:
                ks = [idx(f) for f in ("gcs_eye", "gcs_motor", "gcs_verbal")]
                if all(k is not None for k in ks):
                    out[:, c] = np.logical_and.reduce([any_t(k) for k in ks])
        else:
            # First driver present wins, matching `sapsii_components`'s fallback chains.
            for cand in SAPSII_COMPONENT_DRIVERS[comp]:
                k = idx(cand)
                if k is not None:
                    out[:, c] = any_t(k)
                    break
    return out


#: Feature fallback chains per component, in the order `sapsii_components` tries them.
SAPSII_COMPONENT_DRIVERS = {
    "hr":          ("hr",),
    "sbp":         ("sbp", "sbp_invasive", "sbp_noninvasive"),
    "temp":        ("temp", "temp_core"),
    "pao2_fio2":   ("pao2", "fio2"),
    "urine_24h":   ("urine_output",),
    "bun":         ("bun",),
    "wbc":         ("wbc",),
    "potassium":   ("potassium",),
    "sodium":      ("sodium",),
    "bicarbonate": ("bicarbonate", "hco3"),
    "bilirubin":   ("bilirubin_total",),
    "gcs":         ("gcs_total", "gcs_eye", "gcs_motor", "gcs_verbal"),
}


def sapsii_component_batch(x: np.ndarray, feature_names: Sequence[str],
                           age_years: np.ndarray | None = None,
                           admission_type_pts: np.ndarray | None = None,
                           chronic_disease_pts: np.ndarray | None = None) -> dict:
    """Like `sapsii_batch` but returns the **per-component** points, `{field: (B,) array}`,
    for every SAPS-II component (`SAPSII_COMPONENT_FIELDS`). Lets the eval report which
    component the forecast captures (age/admission_type/chronic_disease are per-stay statics,
    identical for pred vs gt, so they cancel in a pred−gt comparison)."""
    if x.ndim == 2:
        x = x[None]
    B = x.shape[0]
    out = {f: np.zeros(B, dtype=np.float32) for f in SAPSII_COMPONENT_FIELDS}
    for i in range(B):
        a    = float(age_years[i])          if age_years          is not None else None
        adm  = float(admission_type_pts[i]) if admission_type_pts is not None else None
        chr_ = float(chronic_disease_pts[i]) if chronic_disease_pts is not None else None
        c = sapsii_components(x[i], feature_names, a, adm, chr_)
        for f in SAPSII_COMPONENT_FIELDS:
            out[f][i] = getattr(c, f)
    return out


def load_sapsii_static_maps(dataset: str, long_root: str = "data/long") -> tuple[dict, dict]:
    """Per-stay SAPS-II **physical points** from the Stage-1 long parquet (hour 0):
    `admission_type` (0/6/8) and `chronic_disease_sapsii` (0/9/10/17). These live only in
    the long parquet (the Stage-2 splits z-score them; the cohort table lacks them), so they
    are read here directly. Returns (admission_type_map, chronic_disease_map) keyed by int
    stay_id; missing datasets/stays → absent (callers default to 0). HiRID has neither; UMM
    has no chronic-disease column.
    """
    import pandas as pd
    from pathlib import Path
    path = Path(long_root) / f"{dataset}_long.parquet"
    if not path.exists():
        return {}, {}
    df = pd.read_parquet(path, columns=["stay_id", "variable", "value"])
    out = []
    for name in ("admission_type", "chronic_disease_sapsii"):
        s = df[df["variable"] == name].dropna(subset=["value"])
        out.append({int(k): float(v) for k, v in s.groupby("stay_id")["value"].first().items()})
    return out[0], out[1]
