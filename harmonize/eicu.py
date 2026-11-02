"""
harmonize/eicu.py
Step-1 extraction pipeline for eICU-CRD 2.0.

Schema output: Long-Format Parquet — identical schema to mimic_iv.py / hirid.py.
  patient_id | stay_id | hospital_admission_id | hour | variable | value | unit | dataset | measurement_type

Design decisions:
  - patient_id : uniquepid encoded as deterministic sequential integer (sorted order)
  - stay_id    : patientunitstayid (int)
  - hospital_admission_id : pd.NA throughout (not available in eICU)
  - hour       : offset_minutes / 60  (all eICU offsets are in minutes)
  - vitalPeriodic / vitalAperiodic: wide format → chunked read → melt to long
  - infusionDrug: only explicit-unit drugname variants; ml/hr excluded (no concentration)
  - Vasopressors in mcg/min: divided by admissionweight (96% coverage); fallback 80 kg
  - respiratoryCare.csv: ventstartoffset/ventendoffset → mechanical_ventilation intervals
  - nurseCharting.csv (11.5 GB): chunked, GCS labels only
  - FiO2 from respiratoryCharting: stored as percent (21–100) → /100
  - FiO2 from lab.csv: typically fraction; fio2_percent_to_fraction applied defensively
  - eICU encodes 0 as "not measured" in vitalPeriodic → 0 values replaced with NaN
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from harmonize.common import (
    FeatureCatalog,
    apply_outlier_filter,
    fio2_percent_to_fraction,
)

log = logging.getLogger(__name__)

DATASET = "eicu"
CHUNKSIZE = 500_000
# Cohort-median fallback computed at runtime in _weight_lookup()

# ---------------------------------------------------------------------------
# Feature mappings
# ---------------------------------------------------------------------------

# vitalPeriodic: wide-format column → feature key
# NOTE: 'sao2' in vitalPeriodic is SpO2 (pulse oximetry), not arterial SaO2
VITAL_PERIODIC_MAP: dict[str, str] = {
    "heartrate":         "hr",
    "respiration":       "rr",
    "sao2":              "spo2",
    "temperature":       "temp_core",    # already °C
    "etco2":             "etco2",
    "systemicsystolic":  "sbp_invasive",
    "systemicdiastolic": "dbp_invasive",
    "systemicmean":      "mbp_invasive",
}

# vitalAperiodic: wide-format column → feature key
VITAL_APERIODIC_MAP: dict[str, str] = {
    "noninvasivesystolic":  "sbp_noninvasive",
    "noninvasivediastolic": "dbp_noninvasive",
    "noninvasivemean":      "mbp_noninvasive",
    "cardiacoutput":        "cardiac_output",
}

# respiratoryCharting: respchartvaluelabel → feature key
# FiO2 labels are stored as percent (21–100); fio2_percent_to_fraction applied below
RESP_CHART_MAP: dict[str, str] = {
    "FiO2":                       "fio2",
    "FIO2 (%)":                   "fio2",
    "PEEP":                       "peep",
    "PEEP/CPAP":                  "peep",
    "Tidal Volume Observed (VT)": "tidal_volume",
    "Tidal Volume (set)":         "tidal_volume",
    "Exhaled TV (machine)":       "tidal_volume",
    "Exhaled MV":                 "amv",
    "Vent Rate":                  "rr",
    "Total RR":                   "rr",
}
_FIO2_RESP_LABELS: frozenset[str] = frozenset({"FiO2", "FIO2 (%)"})

# lab.csv: labname → feature key
LAB_MAP: dict[str, str] = {
    # Blood gas
    "paO2":               "pao2",
    "paCO2":              "paco2",
    "pH":                 "ph_arterial",
    "Base Excess":        "base_excess",
    "O2 Sat (%)":         "sao2",
    "Total CO2":          "hco3",
    "bicarbonate":        "hco3",
    "HCO3":               "hco3",
    "FiO2":               "fio2",
    # Chemistry
    "glucose":            "glucose",
    "bedside glucose":    "glucose",
    "sodium":             "sodium",
    "potassium":          "potassium",
    "chloride":           "chloride",
    "creatinine":         "creatinine",
    "BUN":                "bun",
    "calcium":            "calcium_total",
    "magnesium":          "magnesium",
    "phosphate":          "phosphate",
    "albumin":            "albumin",
    "anion gap":          "anion_gap",
    "lactate":            "lactate",
    # Liver
    "total bilirubin":    "bilirubin_total",
    "direct bilirubin":   "bilirubin_direct",
    "AST (SGOT)":         "ast",
    "ALT (SGPT)":         "alt",
    "alkaline phos.":     "alp",
    # Coagulation
    "PT - INR":           "inr",
    "PT":                 "pt",
    "PTT":                "ptt",
    "fibrinogen":         "fibrinogen",
    # CBC
    "Hgb":                "hemoglobin",
    "Hct":                "hematocrit",
    "platelets x 1000":   "platelets",   # value already in 10^3/µL
    "WBC x 1000":         "wbc",         # value already in 10^3/µL
    "RBC":                "rbc",
    "MCV":                "mcv",
    "MCHC":               "mchc",
    "MCH":                "mch",
    "RDW":                "rdw",
    "MPV":                "mpv",
    "-lymphs":            "lymphocytes_pct",
    "-monos":             "monocytes_pct",
    "-basos":             "basophils_pct",
    "-eos":               "eosinophils_pct",
    "-polys":             "neutrophils_pct",
    # Other
    "troponin - I":       "troponin_i",
    "urinary creatinine": "creatinine_urine",
    "total protein":      "protein_total",
}

# nurseCharting GCS: (celltypevallabel, celltypevalname) → feature key
_GCS_MAP: dict[tuple[str, str], str] = {
    ("Glasgow coma score", "Eyes"):   "gcs_eye",
    ("Glasgow coma score", "Motor"):  "gcs_motor",
    ("Glasgow coma score", "Verbal"): "gcs_verbal",
}
_GCS_LABELS: frozenset[str] = frozenset(k[0] for k in _GCS_MAP)

# ---------------------------------------------------------------------------
# infusionDrug registries
# ---------------------------------------------------------------------------

# Vasopressors stored as mcg/min → divide by weight_kg → µg/kg/min
VASO_MCGMIN: dict[str, list[str]] = {
    "norepinephrine": ["Norepinephrine (mcg/min)"],
    "epinephrine":    ["Epinephrine (mcg/min)"],
}

# Vasopressors already in mcg/kg/min → use directly
VASO_MCGKGMIN: dict[str, list[str]] = {
    "norepinephrine": ["Norepinephrine (mcg/kg/min)"],
    "epinephrine":    ["Epinephrine (mcg/kg/min)"],
    "dopamine":       ["Dopamine (mcg/kg/min)", "dopamine (mcg/kg/min)"],
    "dobutamine":     ["Dobutamine (mcg/kg/min)"],
}

# Vasopressin in units/min → use directly (target: IU/min, absolute rate)
# All known variants merged: plain (units/min), capitalisation typos, and
# the pre-mixed bag formulations that report units/min directly.
VASO_UNITMIN: dict[str, list[str]] = {
    "vasopressin": [
        "Vasopressin (units/min)",
        "VAsopressin (units/min)",      # eICU drugname typo
        "vasopressin (units/min)",
        "Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/min)",
        "Vasopressin 40 Units Sodium Chloride 0.9% 200 ml (units/min)",
    ],
}

# Vasopressin in units/hr → /60 → IU/min
VASO_UNITHR: dict[str, list[str]] = {
    "vasopressin": [
        "Vasopressin (units/hr)",
        "Vasopressin 20 Units Sodium Chloride 0.9% 100 ml (units/hr)",
        "Vasopressin 20 Units Sodium Chloride 0.9% 250 ml (units/hr)",
        "Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/hr)",
    ],
}

# Other infusions: (exact drugname, conversion factor to catalog unit)
# ml/hr variants excluded — no concentration info available for conversion
INFUSION_OTHER: dict[str, tuple[str, float]] = {
    "fentanyl":        ("Fentanyl (mcg/hr)",           1.0),   # mcg/hr  = µg/h
    "propofol":        ("Propofol (mcg/kg/min)",        0.06),  # ×60/1000 = mg/kg/h
    "midazolam":       ("Midazolam (mg/hr)",            1.0),   # mg/hr   = mg/h
    "dexmedetomidine": ("Dexmedetomidine (mcg/kg/hr)", 1.0),   # mcg/kg/hr = µg/kg/h
    "insulin":         ("Insulin (units/hr)",           1.0),   # units/hr = IU/h
    "heparin":         ("Heparin (units/hr)",           1.0),   # units/hr = IU/h
    "furosemide":      ("Furosemide (mg/hr)",           1.0),   # mg/hr   = mg/h
    "milrinone":       ("Milrinone (mcg/kg/min)",       1.0),   # direct  = µg/kg/min
    "amiodarone":      ("Amiodarone (mg/min)",          60.0),  # ×60     = mg/h
    "pantoprazole":    ("Pantoprazole",                 1.0),   # ASSUMED mg/h
    # ^ eICU's "Pantoprazole" rows lack an explicit unit suffix; we accept the
    #   raw drugrate as mg/h (clinical typical 8 mg/h). PROTONIX (ml/hr) and
    #   PROTONIX (units/hr) variants are skipped because their unit semantics
    #   are unreliable. Pre/post outlier filter logs any pathological values.
}

# ---------------------------------------------------------------------------
# Cohort extraction
# ---------------------------------------------------------------------------

def extract_cohort(
    data_path: str | Path,
    min_stay_hours: float = 24.0,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load ICU cohort from eICU patient.csv.

    Inclusion criteria:
      - Age >= 18 ('>89' sentinel → 91.4)
      - Gender in {Male, Female}
      - ICU LOS >= min_stay_hours

    Returns DataFrame with columns:
      patient_id, stay_id, hospital_admission_id,
      age, gender, weight_kg, icu_type, discharge_status, los_hours
    """
    data_path = Path(data_path)
    log.info("Loading patient.csv...")
    pat = pd.read_parquet(data_path / "patient.parquet")

    pat.loc[pat["age"] == "> 89", "age"] = "91.4"
    pat["age"] = pd.to_numeric(pat["age"], errors="coerce")
    pat["los_hours"] = pat["unitdischargeoffset"] / 60.0

    cohort = pat.loc[
        (pat["age"] >= 18) &
        (pat["gender"].isin(["Male", "Female"])) &
        (pat["los_hours"] >= min_stay_hours)
    ].copy()

    # Encode uniquepid → deterministic integer patient_id (sorted for reproducibility)
    uid_sorted = sorted(cohort["uniquepid"].dropna().unique())
    uid_map = {uid: i for i, uid in enumerate(uid_sorted, start=1)}
    cohort["patient_id"] = cohort["uniquepid"].map(uid_map).astype("int64")
    cohort["stay_id"] = cohort["patientunitstayid"].astype("int64")
    cohort["hospital_admission_id"] = pd.NA

    cohort = cohort.rename(columns={
        "admissionweight":      "weight_kg",
        "admissionheight":      "height_cm",
        "unittype":             "icu_type",
        "unitdischargestatus":  "discharge_status",
    })

    log.info(
        "Cohort: %d stays, %d patients (min_stay_hours=%.0f)",
        len(cohort), cohort["patient_id"].nunique(), min_stay_hours,
    )

    if sample_size is not None and sample_size < cohort["patient_id"].nunique():
        rng = np.random.default_rng(random_state)
        sampled = rng.choice(cohort["patient_id"].unique(), size=sample_size, replace=False)
        cohort = cohort.loc[cohort["patient_id"].isin(sampled)].copy()
        log.info("Sampled to %d stays (%d patients)", len(cohort), cohort["patient_id"].nunique())

    return cohort.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stay_to_patient(cohort: pd.DataFrame) -> pd.Series:
    """Return Series: stay_id → patient_id."""
    return cohort.set_index("stay_id")["patient_id"]


def _weight_lookup(cohort: pd.DataFrame) -> pd.Series:
    """Return Series: stay_id → weight_kg, with cohort-median fallback for missing values."""
    wt = cohort.set_index("stay_id")["weight_kg"].copy()
    wt = pd.to_numeric(wt, errors="coerce")
    wt = wt.where(wt > 0, np.nan)
    fallback = float(wt.median())
    log.info("Vasopressor weight fallback: cohort median = %.1f kg (%d stays without weight)",
             fallback, int(wt.isna().sum()))
    return wt.fillna(fallback)


def _build_long(
    df: pd.DataFrame,
    s2p: pd.Series,
    unit_map: dict[str, str],
    mtype_map: dict[str, str],
) -> pd.DataFrame:
    """Attach metadata columns and return canonical long-format DataFrame."""
    df = df.copy()
    df["patient_id"] = df["stay_id"].map(s2p).astype("int64")
    df["hospital_admission_id"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["unit"] = df["variable"].map(unit_map).fillna("")
    df["measurement_type"] = df["variable"].map(mtype_map).fillna("")
    df["dataset"] = DATASET
    df["stay_id"] = df["stay_id"].astype("Int64")
    return df[[
        "patient_id", "stay_id", "hospital_admission_id",
        "hour", "variable", "value", "unit", "dataset", "measurement_type",
    ]].copy()


def _unit_map(catalog: FeatureCatalog) -> dict[str, str]:
    return {k: catalog.unit(k) for k in catalog.keys()}


def _mtype_map(catalog: FeatureCatalog) -> dict[str, str]:
    return {k: catalog.measurement_type(k) for k in catalog.keys()}


def _filter_outliers_by_variable(
    df: pd.DataFrame,
    catalog: FeatureCatalog,
    value_col: str = "value",
) -> pd.DataFrame:
    parts = []
    for feat, grp in df.groupby("variable"):
        vr = catalog.value_range(feat) if feat in catalog else None
        parts.append(apply_outlier_filter(grp.copy(), value_col, vr))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# vitalPeriodic extraction
# ---------------------------------------------------------------------------

def extract_vitals_periodic(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract from vitalPeriodic.csv (7.4 GB, wide format, ~5 min resolution).

    Wide columns → melted to long. eICU stores 0 as 'not measured' in this
    table, so 0 values are replaced with NaN before the outlier filter.
    """
    data_path = Path(data_path)
    stay_set = set(cohort["stay_id"])
    s2p = _stay_to_patient(cohort)
    um, mm = _unit_map(catalog), _mtype_map(catalog)

    cols = ["patientunitstayid", "observationoffset"] + list(VITAL_PERIODIC_MAP.keys())
    parts: list[pd.DataFrame] = []

    _filter_removed_vp = 0
    _filter_total_vp   = 0
    log.info("Reading vitalPeriodic in chunks...")
    for chunk in [pd.read_parquet(
        data_path / "vitalPeriodic.parquet",
        columns=cols,
    )]:
        sub = chunk[chunk["patientunitstayid"].isin(stay_set)].copy()
        if sub.empty:
            continue

        sub["hour"] = sub["observationoffset"] / 60.0
        melted = sub.melt(
            id_vars=["patientunitstayid", "hour"],
            value_vars=list(VITAL_PERIODIC_MAP.keys()),
            var_name="_col", value_name="value",
        )
        melted = melted.dropna(subset=["value"])
        melted = melted[melted["value"] != 0]   # 0 = not recorded in eICU periodic table
        melted["variable"] = melted["_col"].map(VITAL_PERIODIC_MAP)
        melted["stay_id"] = melted["patientunitstayid"]
        melted = melted.dropna(subset=["variable"])

        _n_pre = len(melted)
        _n_pre_na = int(melted['value'].isna().sum())
        melted = _filter_outliers_by_variable(melted, catalog)
        _filter_removed_vp += int(melted['value'].isna().sum()) - _n_pre_na
        _filter_total_vp   += _n_pre
        if not melted.empty:
            parts.append(melted)

    if not parts:
        log.warning("No vitalPeriodic data found.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result = _build_long(result, s2p, um, mm)
    log.info(
        "vitalPeriodic: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed_vp, _filter_total_vp,
        100.0 * _filter_removed_vp / _filter_total_vp if _filter_total_vp else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# vitalAperiodic extraction
# ---------------------------------------------------------------------------

def extract_vitals_aperiodic(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract from vitalAperiodic.csv (937 MB) — NIBP + cardiac output."""
    data_path = Path(data_path)
    stay_set = set(cohort["stay_id"])
    s2p = _stay_to_patient(cohort)
    um, mm = _unit_map(catalog), _mtype_map(catalog)

    cols = ["patientunitstayid", "observationoffset"] + list(VITAL_APERIODIC_MAP.keys())
    parts: list[pd.DataFrame] = []

    _filter_removed_va = 0
    _filter_total_va   = 0
    log.info("Reading vitalAperiodic in chunks...")
    for chunk in [pd.read_parquet(
        data_path / "vitalAperiodic.parquet",
        columns=cols,
    )]:
        sub = chunk[chunk["patientunitstayid"].isin(stay_set)].copy()
        if sub.empty:
            continue

        sub["hour"] = sub["observationoffset"] / 60.0
        melted = sub.melt(
            id_vars=["patientunitstayid", "hour"],
            value_vars=list(VITAL_APERIODIC_MAP.keys()),
            var_name="_col", value_name="value",
        )
        melted = melted.dropna(subset=["value"])
        melted = melted[melted["value"] != 0]
        melted["variable"] = melted["_col"].map(VITAL_APERIODIC_MAP)
        melted["stay_id"] = melted["patientunitstayid"]
        melted = melted.dropna(subset=["variable"])

        _n_pre = len(melted)
        _n_pre_na = int(melted['value'].isna().sum())
        melted = _filter_outliers_by_variable(melted, catalog)
        _filter_removed_va += int(melted['value'].isna().sum()) - _n_pre_na
        _filter_total_va   += _n_pre
        if not melted.empty:
            parts.append(melted)

    if not parts:
        log.warning("No vitalAperiodic data found.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result = _build_long(result, s2p, um, mm)
    log.info(
        "vitalAperiodic: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed_va, _filter_total_va,
        100.0 * _filter_removed_va / _filter_total_va if _filter_total_va else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# respiratoryCharting extraction
# ---------------------------------------------------------------------------

def extract_respiratory_charting(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract from respiratoryCharting.csv (1.2 GB) — FiO2, PEEP, tidal volume, AMV.

    FiO2 is stored as percent (21–100) in this table; converted to fraction.
    """
    data_path = Path(data_path)
    stay_set = set(cohort["stay_id"])
    s2p = _stay_to_patient(cohort)
    um, mm = _unit_map(catalog), _mtype_map(catalog)
    all_labels = set(RESP_CHART_MAP.keys())

    parts: list[pd.DataFrame] = []

    _filter_removed_rc = 0
    _filter_total_rc   = 0
    log.info("Reading respiratoryCharting in chunks...")
    for chunk in [pd.read_parquet(
        data_path / "respiratoryCharting.parquet",
        columns=["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue"],
    )]:
        mask = chunk["patientunitstayid"].isin(stay_set) & chunk["respchartvaluelabel"].isin(all_labels)
        sub = chunk[mask].copy()
        if sub.empty:
            continue

        sub["hour"] = sub["respchartoffset"] / 60.0
        sub["value"] = pd.to_numeric(sub["respchartvalue"], errors="coerce").astype("float64")
        sub["variable"] = sub["respchartvaluelabel"].map(RESP_CHART_MAP)
        sub["stay_id"] = sub["patientunitstayid"]
        sub = sub.dropna(subset=["value", "variable"])

        # FiO2: convert percent → fraction (99.75% of records are in percent form)
        fio2_mask = sub["respchartvaluelabel"].isin(_FIO2_RESP_LABELS)
        sub.loc[fio2_mask, "value"] = fio2_percent_to_fraction(sub.loc[fio2_mask, "value"])

        _n_pre = len(sub)
        _n_pre_na = int(sub['value'].isna().sum())
        sub = _filter_outliers_by_variable(sub, catalog)
        _filter_removed_rc += int(sub['value'].isna().sum()) - _n_pre_na
        _filter_total_rc   += _n_pre
        if not sub.empty:
            parts.append(sub)

    if not parts:
        log.warning("No respiratoryCharting data found.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result = _build_long(result, s2p, um, mm)
    log.info(
        "respiratoryCharting: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed_rc, _filter_total_rc,
        100.0 * _filter_removed_rc / _filter_total_rc if _filter_total_rc else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Lab extraction
# ---------------------------------------------------------------------------

def extract_labs(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract from lab.csv (2.4 GB) — all mapped labnames."""
    data_path = Path(data_path)
    stay_set = set(cohort["stay_id"])
    s2p = _stay_to_patient(cohort)
    um, mm = _unit_map(catalog), _mtype_map(catalog)
    all_labnames = set(LAB_MAP.keys())

    parts: list[pd.DataFrame] = []

    log.info("Reading lab.csv in chunks...")
    for chunk in [pd.read_parquet(
        data_path / "lab.parquet",
        columns=["patientunitstayid", "labresultoffset", "labname", "labresult"],
    )]:
        mask = chunk["patientunitstayid"].isin(stay_set) & chunk["labname"].isin(all_labnames)
        sub = chunk[mask].copy()
        if sub.empty:
            continue

        sub["hour"] = sub["labresultoffset"] / 60.0
        sub["value"] = pd.to_numeric(sub["labresult"], errors="coerce").astype("float64")
        sub["variable"] = sub["labname"].map(LAB_MAP)
        sub["stay_id"] = sub["patientunitstayid"]
        sub = sub.dropna(subset=["value", "variable"])

        # FiO2 from lab (ABG): apply defensive percent→fraction conversion
        fio2_mask = sub["labname"] == "FiO2"
        if fio2_mask.any():
            sub.loc[fio2_mask, "value"] = fio2_percent_to_fraction(sub.loc[fio2_mask, "value"])

        parts.append(sub)

    if not parts:
        log.warning("No lab data found.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    _n_pre_labs = len(result)
    _n_pre_labs_na = int(result['value'].isna().sum())
    result = _filter_outliers_by_variable(result, catalog)
    _filter_removed_labs = int(result['value'].isna().sum()) - _n_pre_labs_na
    result = _build_long(result, s2p, um, mm)
    log.info(
        "Labs: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed_labs, _n_pre_labs,
        100.0 * _filter_removed_labs / _n_pre_labs if _n_pre_labs else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Temperature extraction (nurseCharting)
# ---------------------------------------------------------------------------

# eICU nurseCharting stores temperature under these labels; Fahrenheit values are converted.
_TEMP_NC_VALNAME_C: frozenset[str] = frozenset({"Temperature (C)"})
_TEMP_NC_VALNAME_F: frozenset[str] = frozenset({"Temperature (F)"})
_TEMP_NC_VALNAMES: frozenset[str] = _TEMP_NC_VALNAME_C | _TEMP_NC_VALNAME_F


def extract_temperature_nursechart(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract core temperature from nurseCharting.csv.

    vitalPeriodic.temperature covers only invasive (Foley/PA catheter) temperature,
    giving ~12% coverage. Most temperature measurements are in nurseCharting under
    nursingchartcelltypevallabel='Temperature', with nursingchartcelltypevalname
    distinguishing 'Temperature (C)' and 'Temperature (F)'.
    """
    data_path = Path(data_path)
    stay_set = set(cohort["stay_id"])
    s2p = _stay_to_patient(cohort)
    um, mm = _unit_map(catalog), _mtype_map(catalog)
    vr = catalog.value_range("temp_core")

    parts: list[pd.DataFrame] = []
    _filter_removed_t = 0
    _filter_total_t   = 0
    log.info("Reading nurseCharting (temperature) in chunks...")
    for chunk in [pd.read_parquet(
        data_path / "nurseCharting.parquet",
        columns=[
            "patientunitstayid", "nursingchartoffset",
            "nursingchartcelltypevallabel", "nursingchartcelltypevalname",
            "nursingchartvalue",
        ],
    )]:
        mask = (
            chunk["patientunitstayid"].isin(stay_set) &
            (chunk["nursingchartcelltypevallabel"] == "Temperature") &
            chunk["nursingchartcelltypevalname"].isin(_TEMP_NC_VALNAMES)
        )
        sub = chunk[mask].copy()
        if sub.empty:
            continue

        sub["hour"]  = sub["nursingchartoffset"] / 60.0
        sub["value"] = pd.to_numeric(sub["nursingchartvalue"], errors="coerce")
        sub = sub.dropna(subset=["value"])

        # Fahrenheit → Celsius
        f_mask = sub["nursingchartcelltypevalname"].isin(_TEMP_NC_VALNAME_F)
        sub.loc[f_mask, "value"] = (sub.loc[f_mask, "value"] - 32) / 1.8

        sub["variable"] = "temp_core"
        sub["stay_id"]  = sub["patientunitstayid"]
        _n_pre_t = len(sub)
        sub = apply_outlier_filter(sub, "value", vr)
        sub = sub.dropna(subset=["value"])
        _filter_removed_t += _n_pre_t - len(sub)
        _filter_total_t   += _n_pre_t
        parts.append(sub)

    if not parts:
        log.warning("No nurseCharting temperature data found.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result = _build_long(result, s2p, um, mm)
    log.info(
        "Temperature (nurseCharting): %d rows | value_range filter removed %d / %d (%.2f%%)",
        len(result),
        _filter_removed_t, _filter_total_t,
        100.0 * _filter_removed_t / _filter_total_t if _filter_total_t else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# GCS extraction (nurseCharting)
# ---------------------------------------------------------------------------

def extract_gcs(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract GCS from nurseCharting.csv (11.5 GB) — chunked, GCS labels only."""
    data_path = Path(data_path)
    stay_set = set(cohort["stay_id"])
    s2p = _stay_to_patient(cohort)
    um, mm = _unit_map(catalog), _mtype_map(catalog)

    parts: list[pd.DataFrame] = []

    log.info("Reading nurseCharting (GCS only) in chunks...")
    for chunk in [pd.read_parquet(
        data_path / "nurseCharting.parquet",
        columns=[
            "patientunitstayid", "nursingchartoffset",
            "nursingchartcelltypevallabel", "nursingchartcelltypevalname",
            "nursingchartvalue",
        ],
    )]:
        mask = (
            chunk["patientunitstayid"].isin(stay_set) &
            chunk["nursingchartcelltypevallabel"].isin(_GCS_LABELS)
        )
        sub = chunk[mask].copy()
        if sub.empty:
            continue

        sub["hour"] = sub["nursingchartoffset"] / 60.0
        sub["value"] = pd.to_numeric(sub["nursingchartvalue"], errors="coerce").astype("float64")
        sub["_key"] = list(zip(
            sub["nursingchartcelltypevallabel"],
            sub["nursingchartcelltypevalname"],
        ))
        sub["variable"] = sub["_key"].map(_GCS_MAP)
        sub["stay_id"] = sub["patientunitstayid"]
        sub = sub.dropna(subset=["value", "variable"])
        parts.append(sub)

    if not parts:
        log.warning("No GCS data found.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    _n_pre_gcs = len(result)
    _n_pre_gcs_na = int(result['value'].isna().sum())
    result = _filter_outliers_by_variable(result, catalog)
    _filter_removed_gcs = int(result['value'].isna().sum()) - _n_pre_gcs_na
    result = _build_long(result, s2p, um, mm)
    log.info(
        "GCS: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed_gcs, _n_pre_gcs,
        100.0 * _filter_removed_gcs / _n_pre_gcs if _n_pre_gcs else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# infusionDrug extraction
# ---------------------------------------------------------------------------

def extract_infusion_drugs(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract vasopressors and other infusions from infusionDrug.csv (266 MB).

    Each row is a point-in-time infusion rate record (no start/end intervals).
    Vasopressors in mcg/min are divided by admissionweight to get µg/kg/min.
    Only explicit-unit drugname variants are used; ml/hr is excluded.
    """
    data_path = Path(data_path)
    stay_set = set(cohort["stay_id"])
    s2p = _stay_to_patient(cohort)
    weight = _weight_lookup(cohort)
    um, mm = _unit_map(catalog), _mtype_map(catalog)

    # Build complete set of drugnames we care about for efficient chunk filtering
    all_drugnames = (
        {n for names in VASO_MCGMIN.values() for n in names} |
        {n for names in VASO_MCGKGMIN.values() for n in names} |
        {n for names in VASO_UNITMIN.values() for n in names} |
        {n for names in VASO_UNITHR.values()  for n in names} |
        {drugname for drugname, _ in INFUSION_OTHER.values()}
    )

    log.info("Reading infusionDrug.csv in chunks...")
    inf_parts: list[pd.DataFrame] = []
    for chunk in [pd.read_parquet(
        data_path / "infusionDrug.parquet",
        columns=["patientunitstayid", "infusionoffset", "drugname", "drugrate"],
    )]:
        mask = chunk["patientunitstayid"].isin(stay_set) & chunk["drugname"].isin(all_drugnames)
        if mask.any():
            inf_parts.append(chunk[mask])
    if not inf_parts:
        log.warning("No infusion drug data found.")
        return pd.DataFrame()
    inf = pd.concat(inf_parts, ignore_index=True)
    inf["drugrate"] = pd.to_numeric(inf["drugrate"], errors="coerce").astype("float64")
    inf["hour"] = inf["infusionoffset"] / 60.0
    inf["stay_id"] = inf["patientunitstayid"]
    inf = inf.dropna(subset=["drugrate", "hour"])

    parts: list[pd.DataFrame] = []
    _filter_removed_inf = 0
    _filter_total_inf   = 0

    # --- Vasopressors: mcg/min → divide by weight → µg/kg/min ---
    all_mcgmin = [n for names in VASO_MCGMIN.values() for n in names]
    df_m = inf[inf["drugname"].isin(all_mcgmin)].copy()
    if not df_m.empty:
        df_m["weight_kg"] = df_m["stay_id"].map(weight)
        df_m["value"] = df_m["drugrate"] / df_m["weight_kg"]
        n2f = {n: feat for feat, names in VASO_MCGMIN.items() for n in names}
        df_m["variable"] = df_m["drugname"].map(n2f)
        df_m = df_m.dropna(subset=["value", "variable", "weight_kg"])
        _n_pre_m = len(df_m)
        _n_pre_m_na = int(df_m['value'].isna().sum())
        _filtered_m = _filter_outliers_by_variable(df_m, catalog)
        _filter_removed_inf += int(_filtered_m['value'].isna().sum()) - _n_pre_m_na
        _filter_total_inf   += _n_pre_m
        parts.append(_filtered_m)

    # --- Vasopressors: mcg/kg/min → direct ---
    all_mcgkgmin = [n for names in VASO_MCGKGMIN.values() for n in names]
    df_d = inf[inf["drugname"].isin(all_mcgkgmin)].copy()
    if not df_d.empty:
        n2f_d = {n: feat for feat, names in VASO_MCGKGMIN.items() for n in names}
        df_d["variable"] = df_d["drugname"].map(n2f_d)
        df_d["value"] = df_d["drugrate"]
        df_d = df_d.dropna(subset=["value", "variable"])
        _n_pre_d = len(df_d)
        _n_pre_d_na = int(df_d['value'].isna().sum())
        _filtered_d = _filter_outliers_by_variable(df_d, catalog)
        _filter_removed_inf += int(_filtered_d['value'].isna().sum()) - _n_pre_d_na
        _filter_total_inf   += _n_pre_d
        parts.append(_filtered_d)

    # --- Vasopressin: units/min → direct (IU/min) ---
    all_unitmin = [n for names in VASO_UNITMIN.values() for n in names]
    df_vp = inf[inf["drugname"].isin(all_unitmin)].copy()
    if not df_vp.empty:
        n2f_vp = {n: feat for feat, names in VASO_UNITMIN.items() for n in names}
        df_vp["variable"] = df_vp["drugname"].map(n2f_vp)
        df_vp["value"] = df_vp["drugrate"]
        df_vp = df_vp.dropna(subset=["value", "variable"])
        _n_pre_vp = len(df_vp)
        _n_pre_vp_na = int(df_vp['value'].isna().sum())
        _filtered_vp = _filter_outliers_by_variable(df_vp, catalog)
        _filter_removed_inf += int(_filtered_vp['value'].isna().sum()) - _n_pre_vp_na
        _filter_total_inf   += _n_pre_vp
        parts.append(_filtered_vp)

    # --- Vasopressin: units/hr → /60 → IU/min ---
    all_unithr = [n for names in VASO_UNITHR.values() for n in names]
    df_vh = inf[inf["drugname"].isin(all_unithr)].copy()
    if not df_vh.empty:
        n2f_vh = {n: feat for feat, names in VASO_UNITHR.items() for n in names}
        df_vh["variable"] = df_vh["drugname"].map(n2f_vh)
        df_vh["value"] = df_vh["drugrate"] / 60.0
        df_vh = df_vh.dropna(subset=["value", "variable"])
        _n_pre_vh = len(df_vh)
        _n_pre_vh_na = int(df_vh['value'].isna().sum())
        _filtered_vh = _filter_outliers_by_variable(df_vh, catalog)
        _filter_removed_inf += int(_filtered_vh['value'].isna().sum()) - _n_pre_vh_na
        _filter_total_inf   += _n_pre_vh
        parts.append(_filtered_vh)

    # --- Other infusions: explicit-unit only, apply conversion factor ---
    for feat_key, (drugname, factor) in INFUSION_OTHER.items():
        df_o = inf[inf["drugname"] == drugname].copy()
        if df_o.empty:
            continue
        df_o["value"] = df_o["drugrate"] * factor
        df_o["variable"] = feat_key
        df_o = df_o.dropna(subset=["value"])
        vr = catalog.value_range(feat_key) if feat_key in catalog else None
        _n_pre_o = len(df_o)
        _n_pre_o_na = int(df_o['value'].isna().sum())
        _filtered_o = apply_outlier_filter(df_o.copy(), "value", vr)
        _filter_removed_inf += int(_filtered_o['value'].isna().sum()) - _n_pre_o_na
        _filter_total_inf   += _n_pre_o
        parts.append(_filtered_o)

    if not parts:
        log.warning("No infusion drug data found.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result = result.dropna(subset=["value"])
    result = _build_long(result, s2p, um, mm)
    log.info(
        "InfusionDrug: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed_inf, _filter_total_inf,
        100.0 * _filter_removed_inf / _filter_total_inf if _filter_total_inf else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Urine output extraction
# ---------------------------------------------------------------------------

def extract_urine_output(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract urine output from intakeOutput.csv (1.9 GB) — celllabel == 'Urine'."""
    data_path = Path(data_path)
    stay_set = set(cohort["stay_id"])
    s2p = _stay_to_patient(cohort)

    parts: list[pd.DataFrame] = []

    log.info("Reading intakeOutput in chunks...")
    for chunk in [pd.read_parquet(
        data_path / "intakeOutput.parquet",
        columns=["patientunitstayid", "intakeoutputentryoffset", "celllabel", "cellvaluenumeric"],
    )]:
        mask = chunk["patientunitstayid"].isin(stay_set) & (chunk["celllabel"] == "Urine")
        sub = chunk[mask].copy()
        if sub.empty:
            continue

        sub["hour"] = sub["intakeoutputentryoffset"] / 60.0
        sub["value"] = pd.to_numeric(sub["cellvaluenumeric"], errors="coerce").astype("float64")
        sub["stay_id"] = sub["patientunitstayid"]
        sub = sub.dropna(subset=["value"])
        parts.append(sub)

    if not parts:
        log.warning("No urine output data found.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    vr = catalog.value_range("urine_output") if "urine_output" in catalog else None
    _n_pre_urine = len(result)
    _n_pre_urine_na = int(result['value'].isna().sum())
    result = apply_outlier_filter(result, "value", vr)
    _filter_removed_urine = int(result['value'].isna().sum()) - _n_pre_urine_na
    result["variable"] = "urine_output"
    result = _build_long(result, s2p, {"urine_output": "mL"}, {"urine_output": "fluid"})
    log.info(
        "Urine output: %d rows | value_range filter removed %d / %d (%.2f%%)",
        len(result),
        _filter_removed_urine, _n_pre_urine,
        100.0 * _filter_removed_urine / _n_pre_urine if _n_pre_urine else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Mechanical ventilation extraction
# ---------------------------------------------------------------------------

def _expand_mv_intervals(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    stay_id_col: str = "patientunitstayid",
) -> pd.DataFrame:
    """Expand minute-offset MV intervals to integer hourly binary flags (vectorized)."""
    df = df.copy()
    sh = df[start_col].values / 60.0
    eh = df[end_col].values / 60.0

    valid = (~np.isnan(sh)) & (~np.isnan(eh)) & (eh > sh)
    df, sh, eh = df[valid].reset_index(drop=True), sh[valid], eh[valid]
    if df.empty:
        return pd.DataFrame(columns=[stay_id_col, "hour", "value"])

    first_h = (np.floor(sh) + 1).astype(int)
    last_h = np.maximum(np.floor(eh).astype(int), first_h)
    n_hours = last_h - first_h + 1

    idx = np.repeat(np.arange(len(df)), n_hours)
    hours = np.concatenate([
        np.arange(f, l + 1, dtype=float) for f, l in zip(first_h, last_h)
    ])

    return pd.DataFrame({
        stay_id_col: df[stay_id_col].values[idx],
        "hour": hours,
        "value": 1.0,
    })


def extract_mechanical_ventilation(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract mechanical ventilation intervals from respiratoryCare.csv (78 MB).

    ventstartoffset / ventendoffset (minutes) are expanded to integer hourly
    binary flags (mechanical_ventilation = 1.0).
    """
    data_path = Path(data_path)
    stay_set = set(cohort["stay_id"])
    s2p = _stay_to_patient(cohort)

    log.info("Reading respiratoryCare.csv...")
    rc = pd.read_parquet(
        data_path / "respiratoryCare.parquet",
        columns=["patientunitstayid", "ventstartoffset", "ventendoffset"],
    )
    rc = rc[rc["patientunitstayid"].isin(stay_set)].copy()
    rc = rc.dropna(subset=["ventstartoffset", "ventendoffset"])

    if rc.empty:
        return pd.DataFrame()

    expanded = _expand_mv_intervals(rc, "ventstartoffset", "ventendoffset")
    expanded["stay_id"] = expanded["patientunitstayid"]
    expanded["variable"] = "mechanical_ventilation"
    result = _build_long(
        expanded, s2p,
        {"mechanical_ventilation": "bool"},
        {"mechanical_ventilation": "ventilation"},
    )
    log.info("Mechanical ventilation: %d rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Admission type  (SAPS-II: 0=scheduled surgical, 6=medical, 8=unscheduled)
# Derived from patient.unitadmitsource + patient.hospitaladmitsource
# ---------------------------------------------------------------------------

_SURGICAL_UNIT_ADMIT = frozenset({
    "Operating Room", "Recovery Room", "PACU",
})

def _admission_type_saps_eicu(unit_admit: str, hosp_admit: str) -> float:
    """Map eICU admit sources → SAPS-II admission type points."""
    surgical = str(unit_admit).strip() in _SURGICAL_UNIT_ADMIT
    emergency_hosp = str(hosp_admit).strip() == "Emergency Department"
    if surgical and emergency_hosp:
        return 8.0   # unscheduled surgical
    if surgical:
        return 0.0   # scheduled surgical
    return 6.0       # medical


# ---------------------------------------------------------------------------
# Blood cultures (microLab) — for Sepsis-3 suspected infection
# ---------------------------------------------------------------------------

def extract_blood_cultures(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract blood culture orders from eICU microLab.

    Filters culturesite starting with 'Blood' (Venipuncture / Central Line),
    emits value=1.0 at culturetakenoffset (minutes → hours since unit admit).
    Variable: 'blood_culture'. NOTE: eICU microbiology is sparse (~2.9k of
    132k stays have any microLab record) — Sepsis-3 culture coverage for eICU
    is therefore limited; document as a limitation.
    """
    data_path = Path(data_path)
    src = data_path / "microLab.parquet"
    if not src.exists():
        src = data_path / "microLab.csv"
    if not src.exists():
        log.warning("microLab not found — skipping eICU blood cultures")
        return pd.DataFrame()

    stay_ids = set(cohort["stay_id"].astype(int))
    pid_map = cohort.set_index("stay_id")["patient_id"].to_dict()

    cols = ["patientunitstayid", "culturetakenoffset", "culturesite"]
    df = (pd.read_parquet(src, columns=cols) if src.suffix == ".parquet"
          else pd.read_csv(src, usecols=cols))
    df = df[df["patientunitstayid"].isin(stay_ids)].copy()
    df = df[df["culturesite"].astype(str).str.startswith("Blood")].copy()
    df["hour"] = pd.to_numeric(df["culturetakenoffset"], errors="coerce") / 60.0
    df = df.dropna(subset=["hour"])
    df = df[(df["hour"] >= 0) & (df["hour"] <= 168.0)]
    df = df.drop_duplicates(subset=["patientunitstayid", "hour"])

    if df.empty:
        log.warning("extract_blood_cultures (eICU): no blood cultures in cohort")
        return pd.DataFrame()

    rows = []
    for _, r in df.iterrows():
        sid = int(r["patientunitstayid"])
        pid = pid_map.get(sid)
        if pid is None:
            continue
        rows.append({
            "patient_id": int(pid), "stay_id": sid, "hospital_admission_id": pd.NA,
            "hour": float(r["hour"]), "variable": "blood_culture", "value": 1.0,
            "unit": "bool", "dataset": DATASET, "measurement_type": "microbiology",
        })
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = pd.array([pd.NA]*len(result), dtype="Int64")
    log.info("Blood cultures (eICU): %d events in %d stays",
             len(result), result["stay_id"].nunique())
    return result


# Antibiotic drugname substring → canonical abx_* class. Ported from the MLHC-2024
# eICU Sepsis-3 script (drugname substring match) and mapped to the abx_* names used
# by MIMIC/UMM + scores.sepsis3.susinf_per_stay. First matching substring wins.
_ABX_SUBSTR_EICU: list[tuple[str, str]] = [
    ("pip", "abx_pip_tazo"), ("tazo", "abx_pip_tazo"), ("zosyn", "abx_pip_tazo"),
    ("vanco", "abx_vancomycin"),
    ("penem", "abx_meropenem"), ("merop", "abx_meropenem"),
    ("cefep", "abx_cefepime"),
    ("ceftri", "abx_ceftriaxone"),
    ("linezolid", "abx_linezolid"),
    ("sulbactam", "abx_amp_sulbactam"), ("unasyn", "abx_amp_sulbactam"),
    ("metronidazole", "abx_metronidazole"), ("flagyl", "abx_metronidazole"),
    ("ciproflox", "abx_ciprofloxacin"),
    ("levoflox", "abx_levofloxacin"), ("loxacin", "abx_levofloxacin"),
    ("clinda", "abx_clindamycin"),
    ("cillin", "abx_penicillin"),
]


def extract_antibiotics(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract IV antibiotic first-dose start events from eICU medication.csv.

    Ported from the MLHC-2024 eICU Sepsis-3 script (`eicu_04_susinf.py`: drugname
    substring match against an antibiotic list, IV route, not cancelled), mapped to
    the canonical `abx_*` variable names used by MIMIC/UMM and consumed by
    `scores.sepsis3.susinf_per_stay`. Emits value=1.0 at the hour of the first IV
    order per (stay, antibiotic class) — enabling Sepsis-3 suspected-infection on
    eICU, which was previously susinf≈0 (→ flagged not-evaluable) because antibiotics
    were never extracted. NOTE: eICU blood-culture coverage is sparse (~2.9k stays);
    antibiotics supply the larger arm of susinf. These events are NOT in the forecast
    catalog, so the forecast tensors / ladder / privacy results are unaffected.
    """
    data_path = Path(data_path)
    src = data_path / "medication.parquet"
    if not src.exists():
        src = data_path / "medication.csv"
    if not src.exists():
        log.warning("medication source not found — skipping eICU antibiotics (Sepsis-3 susinf)")
        return pd.DataFrame()

    stay_ids = set(cohort["stay_id"].astype(int))
    pid_map = cohort.set_index("stay_id")["patient_id"].to_dict()

    cols = ["patientunitstayid", "drugname", "routeadmin",
            "drugordercancelled", "drugorderoffset"]
    df = (pd.read_parquet(src, columns=cols) if src.suffix == ".parquet"
          else pd.read_csv(src, usecols=cols))
    # medication.patientunitstayid is STRING-typed in eICU (unlike microLab's int), so a raw
    # isin() against the int cohort ids silently matches nothing — cast first.
    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df = df.dropna(subset=["patientunitstayid"])
    df["patientunitstayid"] = df["patientunitstayid"].astype(int)
    df = df[df["patientunitstayid"].isin(stay_ids)].copy()
    df = df.dropna(subset=["drugname"])
    # IV route, order not cancelled (mirrors the MLHC filters)
    route = df["routeadmin"].astype(str).str.lower()
    df = df[route.str.contains("iv|intrav", na=False)
            & (df["drugordercancelled"].astype(str).str.lower() != "yes")]
    df["hour"] = pd.to_numeric(df["drugorderoffset"], errors="coerce") / 60.0
    df = df.dropna(subset=["hour"])
    df = df[(df["hour"] >= 0) & (df["hour"] <= 168.0)]

    def _abx(name: str) -> str | None:
        low = name.lower()
        for sub, canon in _ABX_SUBSTR_EICU:
            if sub in low:
                return canon
        return None

    df["variable"] = df["drugname"].astype(str).map(_abx)
    df = df.dropna(subset=["variable"])
    if df.empty:
        log.warning("extract_antibiotics (eICU): no IV antibiotics matched")
        return pd.DataFrame()
    # first start per (stay, antibiotic class)
    df = (df.sort_values("hour")
            .drop_duplicates(subset=["patientunitstayid", "variable"], keep="first"))

    rows = []
    for _, r in df.iterrows():
        pid = pid_map.get(int(r["patientunitstayid"]))
        if pid is None:
            continue
        rows.append({
            "patient_id": int(pid), "stay_id": int(r["patientunitstayid"]),
            "hospital_admission_id": pd.NA, "hour": float(r["hour"]),
            "variable": r["variable"], "value": 1.0, "unit": "bool",
            "dataset": DATASET, "measurement_type": "medication",
        })
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = pd.array([pd.NA] * len(result), dtype="Int64")
    log.info("Antibiotics (eICU): %d IV start events in %d stays (%d classes)",
             len(result), result["stay_id"].nunique(), result["variable"].nunique())
    return result


# ---------------------------------------------------------------------------
# APACHE-IV severity score (pre-computed in eICU apachePatientResult)
# ---------------------------------------------------------------------------

def extract_apache(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract the pre-computed APACHE-IVa score + predicted ICU mortality.

    eICU's apachePatientResult stores APACHE IV and IVa results per stay. We
    take version 'IVa' (recalibrated) and emit two static features at hour=0:
      apache_iv_score          — integer APACHE-IVa points (~0-160)
      apache_iv_pred_mortality — model-predicted ICU mortality probability [0,1]
    A second clinical severity anchor alongside SAPS-II (which we compute
    ourselves). eICU-only — MIMIC/HiRID/UMM have no pre-computed APACHE.
    """
    data_path = Path(data_path)
    src = data_path / "apachePatientResult.parquet"
    if not src.exists():
        src = data_path / "apachePatientResult.csv"
    if not src.exists():
        log.warning("apachePatientResult not found — skipping APACHE-IV")
        return pd.DataFrame()

    stay_ids = set(cohort["stay_id"].astype(int))
    pid_map = cohort.set_index("stay_id")["patient_id"].to_dict()

    cols = ["patientunitstayid", "apachescore", "apacheversion", "predictedicumortality"]
    df = (pd.read_parquet(src, columns=cols) if src.suffix == ".parquet"
          else pd.read_csv(src, usecols=cols))
    df = df[(df["apacheversion"] == "IVa") & df["patientunitstayid"].isin(stay_ids)].copy()

    rows = []
    for _, r in df.iterrows():
        sid = int(r["patientunitstayid"])
        pid = pid_map.get(sid)
        if pid is None:
            continue
        common = {"patient_id": int(pid), "stay_id": sid,
                  "hospital_admission_id": pd.NA, "hour": 0.0,
                  "dataset": DATASET, "measurement_type": "static"}
        score = r["apachescore"]
        if pd.notna(score) and score >= 0:
            rows.append({**common, "variable": "apache_iv_score",
                         "value": float(score), "unit": "score"})
        pm = r["predictedicumortality"]
        if pd.notna(pm) and pm >= 0:
            rows.append({**common, "variable": "apache_iv_pred_mortality",
                         "value": float(pm), "unit": "prob"})

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = pd.array([pd.NA]*len(result), dtype="Int64")
    log.info("APACHE-IVa: %d rows from %d stays",
             len(result), result["stay_id"].nunique())
    return result


# ---------------------------------------------------------------------------
# Diagnoses / comorbidities  (ICD-9, from eICU diagnosis table)
# ---------------------------------------------------------------------------

def extract_diagnoses(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract Charlson comorbidity index and SAPS-II chronic_disease from eICU.

    Reads `diagnosis.csv` (ICD-9 codes, with `diagnosisoffset`). Aggregates
    all codes per stay → computes Charlson score (Quan 2005) and SAPS-II
    chronic_disease points. Emits as static features at hour=0.

    Variables emitted:
      charlson_comorbidity_index  — integer 0–37
      chronic_disease_sapsii      — 0 / 9 / 17 (SAPS-II pts)
    """
    from harmonize.diagnoses import charlson_icd9, chronic_disease_sapsii

    data_path = Path(data_path)
    src = data_path / "diagnosis.parquet"
    if not src.exists():
        src = data_path / "diagnosis.csv"
    if not src.exists():
        log.warning("eICU diagnosis file not found — skipping comorbidities")
        return pd.DataFrame()

    stay_ids = set(cohort["stay_id"].astype(int))
    stay_meta = cohort.set_index("stay_id")[["patient_id"]].to_dict("index")

    log.info("Loading eICU diagnosis table...")
    df = (pd.read_parquet(src, columns=["patientunitstayid", "icd9code"])
          if src.suffix == ".parquet"
          else pd.read_csv(src, usecols=["patientunitstayid", "icd9code"],
                           dtype={"icd9code": str}))

    df = df[df["patientunitstayid"].isin(stay_ids)].copy()
    df["icd9code"] = df["icd9code"].fillna("").astype(str)

    # Aggregate codes per stay
    codes_per_stay = (
        df.groupby("patientunitstayid")["icd9code"]
        .apply(lambda s: [c for c in s if c.strip()])
        .to_dict()
    )
    log.info("  %d stays have ICD-9 codes", len(codes_per_stay))

    rows = []
    for sid, codes in codes_per_stay.items():
        sid = int(sid)
        meta = stay_meta.get(sid, {})
        pid = meta.get("patient_id")
        if pid is None:
            continue
        charlson = charlson_icd9(codes)
        cci = charlson["charlson_total"]
        chron = chronic_disease_sapsii(charlson)
        common = {
            "patient_id": pid, "stay_id": sid,
            "hospital_admission_id": pd.NA,
            "hour": 0.0, "dataset": DATASET,
            "measurement_type": "static",
        }
        rows.append({**common, "variable": "charlson_comorbidity_index",
                     "value": float(cci), "unit": "score"})
        rows.append({**common, "variable": "chronic_disease_sapsii",
                     "value": float(chron), "unit": "saps2_pts"})

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = pd.array([pd.NA]*len(result), dtype="Int64")
    log.info("Comorbidities (eICU): %d rows from %d stays", len(result), len(codes_per_stay))
    return result


# ---------------------------------------------------------------------------
# Static features extraction
# ---------------------------------------------------------------------------

def extract_static(cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract static features at hour=0: age, sex, weight_admission, height, mortality_icu."""
    rows: list[dict] = []
    for _, row in cohort.iterrows():
        pid = int(row["patient_id"])
        sid = int(row["stay_id"])
        common = {
            "patient_id": pid,
            "stay_id": sid,
            "hospital_admission_id": pd.NA,
            "hour": 0.0,
            "dataset": DATASET,
            "measurement_type": "demographic",
        }
        if pd.notna(row.get("age")):
            rows.append({**common, "variable": "age", "value": float(row["age"]), "unit": "years"})
        if pd.notna(row.get("gender")):
            rows.append({**common, "variable": "sex",
                         "value": 1.0 if row["gender"] == "Male" else 0.0, "unit": "M/F"})
        wt = row.get("weight_kg")
        if pd.notna(wt) and float(wt) > 0:
            rows.append({**common, "variable": "weight_admission", "value": float(wt), "unit": "kg"})
        ht = row.get("height_cm")
        if pd.notna(ht) and float(ht) > 0:
            rows.append({**common, "variable": "height", "value": float(ht), "unit": "cm"})
        # mortality_icu: unitdischargestatus == Expired (true ICU mortality)
        ds = str(row.get("discharge_status", "")).strip().lower()
        mortality = 1.0 if ds in {"expired", "dead", "deceased"} else 0.0 if ds in {"alive", "home", "transferred"} else np.nan
        if pd.notna(mortality):
            rows.append({**common, "measurement_type": "outcome",
                         "variable": "mortality_icu", "value": mortality, "unit": "bool"})
        # mortality_hospital: hospitaldischargestatus == Expired (hospital mortality)
        hd = str(row.get("hospitaldischargestatus", "")).strip().lower()
        hmort = 1.0 if hd == "expired" else 0.0 if hd == "alive" else np.nan
        if pd.notna(hmort):
            rows.append({**common, "measurement_type": "outcome",
                         "variable": "mortality_hospital", "value": hmort, "unit": "bool"})
        if pd.notna(row.get("los_hours")):
            rows.append({**common, "variable": "los_hours",
                         "value": float(row["los_hours"]), "unit": "h"})
        # admission_type: derived from admit sources (SAPS-II points)
        ua = str(row.get("unitadmitsource", "")).strip()
        ha = str(row.get("hospitaladmitsource", "")).strip()
        if ua or ha:
            pts = _admission_type_saps_eicu(ua, ha)
            rows.append({**common, "variable": "admission_type",
                         "value": pts, "unit": "saps2_pts"})

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = pd.array([pd.NA] * len(result), dtype="Int64")
    log.info("Static: %d rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_path: str | Path,
    output_path: str | Path,
    features_yaml: str | Path,
    sample_size: int | None = None,
    min_stay_hours: float = 24.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run the full eICU Step-1 extraction pipeline.

    Outputs:
      output_path/eicu_long.parquet    — long-format feature table
      output_path/eicu_cohort.parquet  — cohort metadata

    Parameters
    ----------
    data_path     : Path to eICU-CRD 2.0 root (contains patient.csv etc.)
    output_path   : Directory for output Parquet files
    features_yaml : Path to configs/features.yaml
    sample_size   : If set, subsample to N patients (for fast smoke-test)
    min_stay_hours: Minimum ICU LOS in hours for cohort inclusion
    random_state  : Seed for reproducible sampling
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    catalog = FeatureCatalog.from_yaml(features_yaml)
    log.info("Feature catalog loaded: %d features", len(list(catalog.keys())))

    cohort = extract_cohort(
        data_path, min_stay_hours=min_stay_hours,
        sample_size=sample_size, random_state=random_state,
    )

    log.info("Extracting vitalPeriodic...")
    vp = extract_vitals_periodic(data_path, cohort, catalog)

    log.info("Extracting vitalAperiodic...")
    va = extract_vitals_aperiodic(data_path, cohort, catalog)

    log.info("Extracting respiratoryCharting...")
    rc = extract_respiratory_charting(data_path, cohort, catalog)

    log.info("Extracting labs...")
    labs = extract_labs(data_path, cohort, catalog)

    log.info("Extracting temperature (nurseCharting)...")
    temp_nc = extract_temperature_nursechart(data_path, cohort, catalog)

    log.info("Extracting GCS (nurseCharting)...")
    gcs = extract_gcs(data_path, cohort, catalog)

    log.info("Extracting infusion drugs...")
    drugs = extract_infusion_drugs(data_path, cohort, catalog)

    log.info("Extracting urine output...")
    urine = extract_urine_output(data_path, cohort, catalog)

    log.info("Extracting mechanical ventilation...")
    mv = extract_mechanical_ventilation(data_path, cohort)

    log.info("Extracting diagnoses / comorbidities (ICD-9)...")
    diag = extract_diagnoses(data_path, cohort)

    log.info("Extracting APACHE-IVa severity score...")
    apache = extract_apache(data_path, cohort)

    log.info("Extracting blood cultures (microLab)...")
    bc = extract_blood_cultures(data_path, cohort)

    log.info("Extracting IV antibiotics (medication; Sepsis-3 susinf)...")
    abx = extract_antibiotics(data_path, cohort)

    log.info("Extracting static features...")
    static = extract_static(cohort)

    parts = [df for df in [vp, va, rc, labs, temp_nc, gcs, drugs, urine, mv, diag, apache, bc, abx, static] if not df.empty]
    long_df = pd.concat(parts, ignore_index=True)

    # Clip to 7-day horizon (168 h) consistent with MIMIC-IV and HiRID pipelines
    long_df = long_df.loc[(long_df["hour"] >= 0) & (long_df["hour"] <= 168.0)].copy()

    out_file = output_path / "eicu_long.parquet"
    long_df.to_parquet(out_file, index=False)
    log.info("Saved %d rows to %s", len(long_df), out_file)

    cohort_file = output_path / "eicu_cohort.parquet"
    cohort.to_parquet(cohort_file, index=False)
    log.info("Saved cohort (%d stays) to %s", len(cohort), cohort_file)

    return long_df
