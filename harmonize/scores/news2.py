"""NEWS2 — National Early Warning Score 2 (Royal College of Physicians, 2017).

Why this score is here: SOFA and SAPS-II are *severity* scores; NEWS2 is explicitly a
**deterioration-detection** score, the one actually used at the bedside to trigger escalation.
That makes it the natural clinical comparator for the §15.10 dissociation — where the forecast
read-off detects deterioration near chance while a discriminative head does not.

Seven components, summed to 0–20:

    respiration rate | SpO2 (scale 1) | air-or-oxygen | systolic BP | pulse | consciousness | temperature

Two documented approximations, both forced by what ICU databases record:

  * **Consciousness (ACVPU).** Not charted as ACVPU anywhere here, so it is derived from GCS:
    GCS total 15 -> Alert (0 pts); GCS < 15 -> CVPU (3 pts). This is the standard mapping, but it
    inherits the GCS caveats — notably that UMM's GCS is unreliable under sedation (§7.1), so NEWS2
    for UMM should be read with the same caution as its SOFA-CNS.
  * **SpO2 scale 1 only.** Scale 2 (target 88–92 % for hypercapnic respiratory failure) needs a
    per-patient COPD/retainer flag that none of the five databases provides. Everything is scored
    on scale 1, which *over*-scores true scale-2 patients.

`supplemental_oxygen` is inferred as FiO2 > 0.21 (room air) or mechanical ventilation on.

Implementation mirrors `sapsii.py`: `_pick` resolves a feature by name with explicit fallbacks —
in particular **sbp -> sbp_invasive -> sbp_noninvasive**, because the tensor never carries a merged
`sbp` (omitting that fallback is precisely the bug that silently zeroed the SAPS-II SBP component).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _bins(v: float, bins) -> int:
    """First matching (lo, hi, pts) band, half-open [lo, hi). NaN -> 0 (component not scoreable)."""
    if v is None or not np.isfinite(v):
        return 0
    for lo, hi, pts in bins:
        if lo <= v < hi:
            return pts
    return 0


def rr_pts(rr: float) -> int:
    return _bins(rr, [(-np.inf, 9, 3), (9, 12, 1), (12, 21, 0), (21, 25, 2), (25, np.inf, 3)])


def spo2_pts(spo2: float) -> int:
    """Scale 1 (the default target range 94–98 %)."""
    return _bins(spo2, [(-np.inf, 92, 3), (92, 94, 2), (94, 96, 1), (96, np.inf, 0)])


def oxygen_pts(on_oxygen: bool) -> int:
    return 2 if on_oxygen else 0


def sbp_pts(sbp: float) -> int:
    return _bins(sbp, [(-np.inf, 91, 3), (91, 101, 2), (101, 111, 1), (111, 220, 0), (220, np.inf, 3)])


def hr_pts(hr: float) -> int:
    return _bins(hr, [(-np.inf, 41, 3), (41, 51, 1), (51, 91, 0), (91, 111, 1), (111, 131, 2), (131, np.inf, 3)])


def consciousness_pts(gcs_total: float) -> int:
    """ACVPU derived from GCS: 15 = Alert (0), anything less = CVPU (3)."""
    if gcs_total is None or not np.isfinite(gcs_total):
        return 0
    return 0 if gcs_total >= 15 else 3


def temp_pts(temp_c: float) -> int:
    return _bins(temp_c, [(-np.inf, 35.05, 3), (35.05, 36.05, 1), (36.05, 38.05, 0),
                          (38.05, 39.05, 1), (39.05, np.inf, 2)])


@dataclass
class NEWS2Components:
    respiration: int
    spo2: int
    oxygen: int
    sbp: int
    pulse: int
    consciousness: int
    temperature: int

    def total(self) -> int:
        return (self.respiration + self.spo2 + self.oxygen + self.sbp
                + self.pulse + self.consciousness + self.temperature)

    def risk(self) -> str:
        """RCP escalation bands: any single component = 3 is 'medium' even at a low total."""
        t = self.total()
        if t >= 7:
            return "high"
        if t >= 5 or 3 in (self.respiration, self.spo2, self.sbp, self.pulse,
                           self.consciousness, self.temperature):
            return "medium"
        return "low"


def _pick(x: np.ndarray, names: Sequence[str], *wanted: str) -> np.ndarray | None:
    """First of `wanted` present in `names` -> that column. Explicit fallback chain (see module doc)."""
    for w in wanted:
        if w in names:
            return x[..., list(names).index(w)]
    return None


def _worst(a: np.ndarray | None, fn) -> float:
    """Apply `fn` over the window and take the WORST (max points) — NEWS2 is a worst-value score."""
    if a is None:
        return 0
    vals = np.asarray(a, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    return max((fn(float(v)) for v in vals), default=0)


def news2_components(x: np.ndarray, feature_names: Sequence[str]) -> NEWS2Components:
    """Worst-of-window NEWS2 from a (T, F) physiological matrix in PHYSICAL units."""
    names = list(feature_names)
    rr = _pick(x, names, "rr")
    spo2 = _pick(x, names, "spo2", "sao2")
    # never merged in the tensor -> invasive, then non-invasive (the sapsii.py lesson)
    sbp = _pick(x, names, "sbp", "sbp_invasive", "sbp_noninvasive")
    hr = _pick(x, names, "hr")
    temp = _pick(x, names, "temp", "temp_core")
    fio2 = _pick(x, names, "fio2")
    mv = _pick(x, names, "mechanical_ventilation")

    gcs = _pick(x, names, "gcs_total")
    if gcs is None:
        ge, gm, gv = (_pick(x, names, k) for k in ("gcs_eye", "gcs_motor", "gcs_verbal"))
        gcs = ge + gm + gv if all(v is not None for v in (ge, gm, gv)) else None

    on_o2 = False
    for arr, thr in ((fio2, 0.21), (mv, 0.5)):
        if arr is not None:
            v = np.asarray(arr, dtype=float)
            v = v[np.isfinite(v)]
            on_o2 = on_o2 or bool((v > thr).any())

    # consciousness uses the WORST (lowest) GCS in the window
    gcs_min = np.nan
    if gcs is not None:
        g = np.asarray(gcs, dtype=float).ravel()
        g = g[np.isfinite(g)]
        gcs_min = float(g.min()) if len(g) else np.nan

    return NEWS2Components(
        respiration=_worst(rr, rr_pts),
        spo2=_worst(spo2, spo2_pts),
        oxygen=oxygen_pts(on_o2),
        sbp=_worst(sbp, sbp_pts),
        pulse=_worst(hr, hr_pts),
        consciousness=consciousness_pts(gcs_min),
        temperature=_worst(temp, temp_pts),
    )


def news2_from_matrix(x: np.ndarray, feature_names: Sequence[str]) -> int:
    return news2_components(x, feature_names).total()


NEWS2_COMPONENT_FIELDS = ("respiration", "spo2", "oxygen", "sbp", "pulse",
                          "consciousness", "temperature")


def news2_batch(x: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """(B, T, F) -> (B,) worst-of-window NEWS2 totals."""
    if x.ndim == 2:
        x = x[None]
    return np.array([news2_from_matrix(x[i], feature_names) for i in range(x.shape[0])],
                    dtype=np.float32)


def news2_component_batch(x: np.ndarray, feature_names: Sequence[str]) -> dict:
    """(B, T, F) -> {component: (B,) points} for all seven components."""
    if x.ndim == 2:
        x = x[None]
    B = x.shape[0]
    out = {f: np.zeros(B, dtype=np.float32) for f in NEWS2_COMPONENT_FIELDS}
    for i in range(B):
        c = news2_components(x[i], feature_names)
        for f in NEWS2_COMPONENT_FIELDS:
            out[f][i] = getattr(c, f)
    return out
