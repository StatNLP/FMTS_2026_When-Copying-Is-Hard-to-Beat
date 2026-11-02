"""
harmonize/umm.py
Step-1 extraction pipeline for UMM ICU data (Mannheim University Medical Center).

Schema output: Long-Format Parquet
  patient_id | stay_id | hospital_admission_id | hour | variable | value | unit | dataset | measurement_type

Key aspects:
  - Data from UMM_Export/ (MSSQL exports, 260528 version): demographics, vitals,
    lab, medications, blood_cultures (new), diagnoses (new, not yet in Stage 1)
  - patient_id = lifeTimeNumber (persistent across encounters)
  - stay_id = encounterId, hospital_admission_id = encounterNumber
  - ICU start/end timing + deceased embedded in demographics (260520+)
  - isOutlier flag filters bad values; catalog value_range applied on top
  - Drug doses (d_*) in µg/kg/min; m_* medications in their reported rate unit
  - FiO2 (v_fio2, v_fio2ein) recorded as percentage (0-100) -> converted to fraction (0-1)
  - Antibiotics (abx_*): flag_start / flag_end events with chartTime — enables Sepsis-3
  - Blood cultures: order_time events — combined with abx timestamps for suspected infection
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
    hours_since_intime,
    make_long_df,
)

log = logging.getLogger(__name__)

DATASET       = "umm"
CHUNKSIZE     = 500_000
MINS_PER_YEAR = 365.25 * 24 * 60

# Paths relative to data_path. Updated 2026-05-28 to the new 260528 export
# (UMM_Export/) which adds blood_cultures and diagnoses tables.
# Old 260520 export archived as raw_data/260520_UMM_Export/.
_DEMO_FILE   = "UMM_Export/260528_UMM_demographics.csv"
_VITALS_FILE = "UMM_Export/260528_UMM_vitals.csv"
_LAB_FILE    = "UMM_Export/260528_UMM_lab.csv"
_MEDS_FILE   = "UMM_Export/260528_UMM_medications.csv"
# New in 260528 export — not yet fully integrated into Stage 1:
_BLOOD_CULTURES_FILE = "UMM_Export/260528_UMM_blood_cultures.csv"
_DIAGNOSES_FILE      = "UMM_Export/260528_UMM_diagnoses.csv"
# GTSQ data (Glasgow Outcome / QoL questionnaire) — not consumed in Stage 1:
#   UMM_Export/260528_UMM_gtsq_entries.csv
#   UMM_Export/260528_UMM_gtsq_dict.csv

# ---------------------------------------------------------------------------
# Variable mapping: variableName -> (canonical_name, measurement_type) | None
# ---------------------------------------------------------------------------

VAR_MAP: dict[str, tuple[str, str] | None] = {
    # ---- Vitals ----
    "v_heartrate":       ("hr",                    "vital"),
    "v_rr":              ("rr",                    "vital"),
    "v_spo2":            ("spo2",                  "vital"),
    "v_sbp_inv":         ("sbp_invasive",          "vital"),
    "v_dbp_inv":         ("dbp_invasive",          "vital"),
    "v_mbp_inv":         ("mbp_invasive",          "vital"),
    "v_sbp_noninv":      ("sbp_noninvasive",       "vital"),
    "v_dbp_noninv":      ("dbp_noninvasive",       "vital"),
    "v_mbp_noninv":      ("mbp_noninvasive",       "vital"),
    "v_temp":            ("temp_core",             "vital"),
    "v_etco2":           ("etco2",                 "vital"),
    "v_fio2":            ("fio2",                  "vital"),
    "v_fio2ein":         ("fio2",                  "vital"),
    "v_amv":             ("amv",                   "vital"),
    "v_hzv":             ("cardiac_output",        "vital"),
    "v_svol":            ("stroke_volume",         "vital"),
    "v_svri":            None,
    # GCS subscores + total
    "v_gcs_eye":         ("gcs_eye",               "vital"),
    "v_gcs_motor":       ("gcs_motor",             "vital"),
    "v_gcs_verbal":      ("gcs_verbal",            "vital"),
    "v_gcs_total":       ("gcs_total",             "vital"),
    # Ventilation parameters
    "v_mech_vent":       ("mechanical_ventilation", "ventilation"),
    "v_vent_mode":       None,                             # categorical (string) — not numeric, skip in long format
    "v_peep":            ("peep",                  "ventilation"),
    "v_tidal_volume":    ("tidal_volume",           "ventilation"),
    # ---- Fluids (stored in vitals file) ----
    "b_urine":           ("urine_output",          "fluid"),
    "b_net_balance":     None,
    # ---- Labs — blood gas ----
    "l_artph":           ("ph_arterial",           "lab"),
    "l_pao2":            ("pao2",                  "lab"),
    "l_pco2":            ("paco2",                 "lab"),
    "l_sao2":            ("sao2",                  "lab"),
    "l_sgvo2":           ("svo2",                  "lab"),
    "l_svo2":            ("svo2",                  "lab"),
    # ---- Labs — chemistry ----
    "l_baseex":          ("base_excess",           "lab"),
    "l_bicarb":          ("hco3",                  "lab"),
    "l_lactate":         ("lactate",               "lab"),
    "l_glucose":         ("glucose",               "lab"),
    "l_creatinine":      ("creatinine",            "lab"),
    "l_bun":             ("bun",                   "lab"),
    "l_natrium":         ("sodium",                "lab"),
    "l_kalium":          ("potassium",             "lab"),
    "l_calcium":         ("calcium_free",          "lab"),   # ionised Ca (BGA) — median 1.17 mmol/L
    "l_calcium_total":   ("calcium_total",         "lab"),   # total Ca (chemistry panel) — median 1.99 mmol/L
    "l_calcium_albumin_corrected": ("calcium_corrected", "lab"),  # Payne-corrected total Ca — median 2.61 mmol/L
    "l_chlorid":         ("chloride",              "lab"),
    "l_crp":             ("crp",                   "lab"),
    "l_pct":             ("pct",                   "lab"),
    "l_alat":            ("alt",                   "lab"),
    "l_asat":            ("ast",                   "lab"),
    "l_alp":             ("alp",                   "lab"),
    "l_bilirubin":       ("bilirubin_total",       "lab"),
    "l_bilirubin_direct": ("bilirubin_direct",     "lab"),   # new in 260520 export
    "l_lipase":          ("lipase",                "lab"),
    "l_albumin":         ("albumin",               "lab"),
    "l_ldh":             ("ldh",                   "lab"),
    "l_magnesium":       ("magnesium",             "lab"),
    "l_phosphate":       ("phosphate",             "lab"),
    "l_protein_total":   ("protein_total",         "lab"),   # new in 260520 export
    "l_fibrinogen":      ("fibrinogen",            "lab"),
    "l_troponin_i":      ("troponin_i",            "lab"),
    # ---- Labs — haematology ----
    "l_haemo":           ("hemoglobin",            "lab"),
    "l_thrombo":         ("platelets",             "lab"),
    "l_leuko":           ("wbc",                   "lab"),
    "l_lympho":          ("lymphocytes_abs",       "lab"),
    "l_inr":             ("inr",                   "lab"),
    "l_aptt":            ("ptt",                   "lab"),
    "l_hematocrit":      ("hematocrit",            "lab"),
    "l_mcv":             ("mcv",                   "lab"),
    "l_mch":             ("mch",                   "lab"),
    "l_mchc":            ("mchc",                  "lab"),
    "l_quick":           None,                             # Quick-Wert (%) != PT (s)
    # ---- Labs — differential ----
    "l_neutrophils_pct": ("neutrophils_pct",       "lab"),
    "l_lymphocytes_pct": ("lymphocytes_pct",       "lab"),
    "l_monocytes_pct":   ("monocytes_pct",         "lab"),
    "l_eosinophils_pct": ("eosinophils_pct",       "lab"),
    "l_basophils_pct":   ("basophils_pct",         "lab"),
    "l_basophils_abs":   None,                             # not in feature catalog
    "l_bands_pct":       None,                             # not in feature catalog
    "l_segmented_pct":   None,                             # duplicate of neutrophils_pct
    # ---- Medications — d_* in µg/kg/min (weight-normalised) ----
    "d_noradrenaline":   ("norepinephrine",        "medication"),
    "d_adrenaline":      ("epinephrine",           "medication"),
    "d_dopamine":        ("dopamine",              "medication"),
    "d_dobutamine":      ("dobutamine",            "medication"),
    # ---- Medications — m_* rate (ml/h or IU/h) ----
    "m_insulin":         ("insulin",               "medication"),
    "m_propofol":        ("propofol",              "medication"),
    "m_midazolam":       ("midazolam",             "medication"),
    "m_fentanyl":        ("fentanyl",              "medication"),
    "m_morphine":        ("morphine",              "medication"),   # new in 260520 export
    "m_dexmedetomidine": ("dexmedetomidine",       "medication"),   # new in 260520 export
    "m_heparin":         ("heparin",               "medication"),
    "m_amiodarone":      ("amiodarone",            "medication"),
    "m_furosemide":      ("furosemide",            "medication"),
}

_FIO2_VARS = {"v_fio2", "v_fio2ein"}

# admission_type_saps is exported as German text with the SAPS-II type code in
# parentheses, e.g. "Nichtgeplant chirurgisch (8)" or "Geplant chirurgisch (0)".
# Extract the integer code for the static long-format value.
import re as _re
_SAPS_CODE_RE = _re.compile(r"\((\d+)\)")

# UMM ward codes observed in the 260520 export (icu_unit_first). Mapped to
# stable integers so the static feature is numeric. Order is alphabetic for
# reproducibility; new wards seen in future exports will need an extension here.
_UMM_ICU_UNIT_CODES: dict[str, int] = {
    "10-4": 1,
    "11-4": 2,
    "32-2": 3,
    "32-3": 4,
    "36-4": 5,
    "37-4": 6,
}

# Per-variable unit conversions: UMM-native unit -> catalog target unit.
# Keys are CANONICAL feature names (the right-hand side of VAR_MAP). The
# conversion is multiplicative and applied AFTER mapping but BEFORE
# value_range filtering, so downstream stages see catalog-unit values.
#
# UMM follows German lab convention (SI units); catalog mostly follows
# US conventional units except for SI-native fields (Na/K/Cl in mmol/L,
# CRP in mg/L, etc.) which need no conversion.
UNIT_CONVERSIONS: dict[str, float] = {
    # g/L -> g/dL  (divide by 10)
    "albumin":          0.1,
    "protein_total":    0.1,
    # g/L -> mg/dL (multiply by 100)
    "fibrinogen":       100.0,
    # mmol/L -> mg/dL  (multiply by atomic/molecular weight in g/mol, then ÷10)
    "calcium_total":    4.008,    # Ca, 40.08 g/mol
    "magnesium":        2.4305,   # Mg, 24.305 g/mol
    "phosphate":        3.097,    # P,  30.97 g/mol
    # µmol/L -> mg/dL  (Bilirubin MW 584.66, factor 1/17.1 = 0.0585)
    "bilirubin_direct": 0.0585,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _intime_lookup(cohort: pd.DataFrame) -> pd.Series:
    return cohort.set_index("stay_id")["icu_start"]


def _stay_meta(cohort: pd.DataFrame) -> pd.DataFrame:
    return cohort.set_index("stay_id")[["patient_id", "hospital_admission_id"]]


def _read_mssql_demographics(path: Path) -> pd.DataFrame:
    """Read MSSQL export demographics from .parquet (preferred) or .csv (fallback).
    Filters out the dash-separator row that lives between header and data."""
    pq_path = path.with_suffix(".parquet")
    if pq_path.exists():
        df = pd.read_parquet(pq_path).astype(str)
    else:
        df = pd.read_csv(path, sep=";", skiprows=1, dtype=str, low_memory=False)
    df = df[~df[df.columns[0]].astype(str).str.startswith("-")].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Cohort extraction
# ---------------------------------------------------------------------------

def extract_cohort(
    data_path: str | Path,
    min_stay_hours: float = 24.0,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    data_path = Path(data_path)

    # --- Demographics ---
    demo = _read_mssql_demographics(data_path / _DEMO_FILE)
    demo = demo.rename(columns={
        "encounterId":    "stay_id",
        "encounterNumber": "enc_num",
        "lifeTimeNumber": "patient_id",
        "gender":         "sex_raw",
        "age":            "age_min",
    })
    demo["stay_id"]    = pd.to_numeric(demo["stay_id"],    errors="coerce")
    demo["enc_num"]    = pd.to_numeric(demo["enc_num"],    errors="coerce")
    demo["patient_id"] = pd.to_numeric(demo["patient_id"], errors="coerce")
    demo["age_min"]    = pd.to_numeric(demo["age_min"],    errors="coerce")
    demo["age"]        = demo["age_min"] / MINS_PER_YEAR
    demo["sex"]        = demo["sex_raw"].map({"M": "M", "W": "F"})
    demo["hospital_admission_id"] = demo["enc_num"]

    # Weight and height — keep only non-outlier values
    demo["weight_kg"] = pd.to_numeric(demo["weight_kg"], errors="coerce")
    demo["height_cm"] = pd.to_numeric(demo["height_cm"], errors="coerce")
    demo["weight_isOutlier"] = pd.to_numeric(demo["weight_isOutlier"], errors="coerce")
    demo["height_isOutlier"] = pd.to_numeric(demo["height_isOutlier"], errors="coerce")
    demo.loc[demo["weight_isOutlier"] == 1, "weight_kg"] = np.nan
    demo.loc[demo["height_isOutlier"] == 1, "height_cm"] = np.nan

    # --- ICU timing, mortality, admission/icu type, antibiotic exposure ---
    # (260520 export integrates these into demographics; ZZ7 no longer needed)
    demo["icu_start"]      = pd.to_datetime(demo["icu_admission"],  errors="coerce")
    demo["icu_end"]        = pd.to_datetime(demo["icu_discharge"], errors="coerce")
    demo["death_time"]     = pd.to_datetime(demo["death_time"],    errors="coerce")
    demo["deceased"]       = pd.to_numeric(demo["deceased"],       errors="coerce")
    demo["los_hours"]      = (
        (demo["icu_end"] - demo["icu_start"]).dt.total_seconds() / 3600
    )

    # --- Filter and finalise ---
    cohort = demo.dropna(subset=["stay_id", "patient_id", "icu_start", "icu_end"])
    cohort = cohort.copy()
    cohort["stay_id"]    = cohort["stay_id"].astype(int)
    cohort["patient_id"] = cohort["patient_id"].astype(int)
    cohort["hospital_admission_id"] = cohort["hospital_admission_id"].astype(int)

    cohort = cohort.loc[cohort["los_hours"] >= min_stay_hours].copy()

    log.info(
        "Cohort: %d stays, %d patients (min_stay_hours=%.0f)",
        len(cohort), cohort["patient_id"].nunique(), min_stay_hours,
    )

    if sample_size is not None and sample_size < len(cohort):
        rng = np.random.default_rng(random_state)
        sampled = rng.choice(cohort["stay_id"].values, size=sample_size, replace=False)
        cohort = cohort.loc[cohort["stay_id"].isin(sampled)].copy()
        log.info("Sampled to %d stays", len(cohort))

    return cohort.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time-series extraction
# ---------------------------------------------------------------------------

def _process_ts_file(
    path: Path,
    stay_ids: set[int],
    intime: pd.Series,
    meta: pd.DataFrame,
    active_vars: dict,
    catalog: FeatureCatalog,
    extra_cols: list[str] | None = None,
) -> list[pd.DataFrame]:
    """Read one MSSQL TS file in chunks; return processed DataFrames."""
    parts: list[pd.DataFrame] = []
    total_in = 0

    base_dtype = {
        "encounterId":   str,
        "encounterNumber": str,
        "chartTime":     str,
        "variableName":  str,
        "value":         str,
        "isOutlier":     str,
    }
    if extra_cols:
        for col in extra_cols:
            base_dtype[col] = str

    total_filtered_out = 0
    # Prefer .parquet sibling if present; otherwise read CSV in chunks
    pq_path = path.with_suffix(".parquet")
    if pq_path.exists():
        log.info("Reading %s (parquet)...", pq_path.name)
        df_full = pd.read_parquet(pq_path).astype(str)
        chunks_iter = [df_full]
    else:
        log.info("Reading %s in chunks (chunksize=%d)...", path.name, CHUNKSIZE)
        chunks_iter = pd.read_csv(
            path,
            sep=";",
            skiprows=1,
            chunksize=CHUNKSIZE,
            dtype=base_dtype,
            low_memory=False,
        )
    for chunk in chunks_iter:
        # Drop MSSQL dash separator row (only appears in first chunk)
        chunk = chunk[~chunk["encounterId"].astype(str).str.startswith("-")]
        if chunk.empty:
            continue

        chunk["encounterId"] = pd.to_numeric(chunk["encounterId"], errors="coerce")
        chunk = chunk.dropna(subset=["encounterId"])
        chunk["encounterId"] = chunk["encounterId"].astype(int)

        chunk = chunk[chunk["encounterId"].isin(stay_ids)]
        chunk = chunk[chunk["variableName"].isin(active_vars)]
        if chunk.empty:
            continue

        # isOutlier filter: keep only clean rows (0 = clean, 1 = outlier, NULL = keep)
        chunk = chunk[chunk["isOutlier"].astype(str) != "1"]

        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk["chartTime"] = pd.to_datetime(chunk["chartTime"], errors="coerce")
        chunk = chunk.dropna(subset=["value", "chartTime"])
        if chunk.empty:
            continue

        total_in += len(chunk)

        chunk["hour"] = hours_since_intime(
            chunk["chartTime"], chunk["encounterId"].map(intime)
        )
        chunk = chunk.loc[chunk["hour"] >= 0]
        if chunk.empty:
            continue

        chunk["variable"]         = chunk["variableName"].map({k: v[0] for k, v in active_vars.items()})
        chunk["measurement_type"] = chunk["variableName"].map({k: v[1] for k, v in active_vars.items()})

        # FiO2 % -> fraction
        fio2_mask = chunk["variableName"].isin(_FIO2_VARS)
        if fio2_mask.any():
            chunk.loc[fio2_mask, "value"] = fio2_percent_to_fraction(
                chunk.loc[fio2_mask, "value"]
            )

        # Per-variable SI -> catalog unit conversion (e.g. g/L -> g/dL).
        # Applied BEFORE value_range filtering so the catalog bounds match.
        for canonical, factor in UNIT_CONVERSIONS.items():
            mask = chunk["variable"] == canonical
            if mask.any():
                chunk.loc[mask, "value"] = chunk.loc[mask, "value"] * factor

        # Catalog value_range filter per variable
        _n_pre_filter = len(chunk)
        filtered: list[pd.DataFrame] = []
        for var, grp in chunk.groupby("variable"):
            vr = catalog.value_range(var) if var in catalog else None
            if vr is not None:
                grp = apply_outlier_filter(grp, "value", vr)
            filtered.append(grp.dropna(subset=["value"]))
        if not filtered:
            continue
        chunk = pd.concat(filtered, ignore_index=True)
        total_filtered_out += _n_pre_filter - len(chunk)

        chunk = chunk.rename(columns={"encounterId": "stay_id"})
        chunk = chunk.join(meta, on="stay_id")
        chunk["unit"]    = chunk["variable"].map(catalog.unit)
        chunk["dataset"] = DATASET

        parts.append(chunk[[
            "patient_id", "stay_id", "hospital_admission_id",
            "hour", "variable", "value", "unit", "dataset", "measurement_type",
        ]])

    log.info(
        "  %s: %d rows in-cohort | value_range filter removed %d / %d (%.2f%%)",
        path.name, total_in,
        total_filtered_out, total_in,
        100.0 * total_filtered_out / total_in if total_in else 0.0,
    )
    return parts


def extract_timeseries(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    data_path = Path(data_path)

    stay_ids = set(cohort["stay_id"].astype(int))
    intime   = _intime_lookup(cohort)
    meta     = _stay_meta(cohort)

    active_vars = {k: v for k, v in VAR_MAP.items() if v is not None}

    parts: list[pd.DataFrame] = []

    # Vitals and fluids
    parts.extend(_process_ts_file(
        data_path / _VITALS_FILE, stay_ids, intime, meta, active_vars, catalog,
    ))

    # Labs
    parts.extend(_process_ts_file(
        data_path / _LAB_FILE, stay_ids, intime, meta, active_vars, catalog,
    ))

    # Medications (extra doseUnit column)
    parts.extend(_process_ts_file(
        data_path / _MEDS_FILE, stay_ids, intime, meta, active_vars, catalog,
        extra_cols=["doseUnit"],
    ))

    if not parts:
        return pd.DataFrame(columns=[
            "patient_id", "stay_id", "hospital_admission_id",
            "hour", "variable", "value", "unit", "dataset", "measurement_type",
        ])

    result = pd.concat(parts, ignore_index=True)

    # Deduplicate fio2: v_fio2 and v_fio2ein -> keep mean per stay+hour
    fio2_mask = result["variable"] == "fio2"
    if fio2_mask.sum() > 0:
        fio2_df = (
            result[fio2_mask]
            .groupby(["stay_id", "hour"], as_index=False)
            .agg(patient_id=("patient_id", "first"),
                 hospital_admission_id=("hospital_admission_id", "first"),
                 value=("value", "mean"),
                 unit=("unit", "first"),
                 dataset=("dataset", "first"),
                 measurement_type=("measurement_type", "first"))
        )
        fio2_df["variable"] = "fio2"
        result = pd.concat([result[~fio2_mask], fio2_df], ignore_index=True)

    # Deduplicate svo2: l_sgvo2 (ScvO2 proxy) + l_svo2 (true) -> mean
    svo2_mask = result["variable"] == "svo2"
    if svo2_mask.sum() > 0:
        svo2_df = (
            result[svo2_mask]
            .groupby(["stay_id", "hour"], as_index=False)
            .agg(patient_id=("patient_id", "first"),
                 hospital_admission_id=("hospital_admission_id", "first"),
                 value=("value", "mean"),
                 unit=("unit", "first"),
                 dataset=("dataset", "first"),
                 measurement_type=("measurement_type", "first"))
        )
        svo2_df["variable"] = "svo2"
        result = pd.concat([result[~svo2_mask], svo2_df], ignore_index=True)

    log.info("Timeseries: %d rows, %d features", len(result), result["variable"].nunique())
    return result


# ---------------------------------------------------------------------------
# Antibiotic start/stop events (260528 export)
# ---------------------------------------------------------------------------

_ABX_VAR_MAP: dict[str, str] = {
    "abx_pip_tazo":   "abx_pip_tazo",
    "abx_meropenem":  "abx_meropenem",
    "abx_vancomycin": "abx_vancomycin",
    "abx_linezolid":  "abx_linezolid",
}


def extract_antibiotics(
    data_path: str | Path,
    cohort: pd.DataFrame,
    clip_hours: float | None = 168.0,
) -> pd.DataFrame:
    """Extract antibiotic start/stop events as time-varying binary features.

    Reads the medications file and filters to abx_* variables with
    doseUnit == "flag_start" or "flag_end". Emits value=1.0 at the hour of
    each event. Variable names: abx_pip_tazo, abx_meropenem, abx_vancomycin,
    abx_linezolid (start events); abx_*_end for stop events.

    These are not currently in the training feature set (not in features.yaml)
    but are available in the long-format parquet for Sepsis-3 analysis.
    The Sepsis-3 suspected-infection criterion requires antibiotic start
    timestamp + blood culture within ±72 h — see extract_blood_cultures().
    """
    data_path = Path(data_path)
    meds_path = data_path / _MEDS_FILE
    pq_path   = meds_path.with_suffix(".parquet")
    src       = pq_path if pq_path.exists() else meds_path

    stay_ids = set(cohort["stay_id"].astype(int))
    intime   = _intime_lookup(cohort)
    meta     = _stay_meta(cohort)

    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(
        src, sep=";", skiprows=1, dtype=str, low_memory=False
    )
    df = df[~df[df.columns[0]].astype(str).str.startswith("-")].reset_index(drop=True)
    df["encounterId"] = pd.to_numeric(df["encounterId"], errors="coerce")
    df = df[df["encounterId"].isin(stay_ids)].copy()
    df = df[df["variableName"].isin(_ABX_VAR_MAP)].copy()
    df = df[df["doseUnit"].isin(["flag_start", "flag_end"])].copy()

    if df.empty:
        log.warning("extract_antibiotics: no antibiotic flag events found")
        return pd.DataFrame()

    df["chartTime"] = pd.to_datetime(df["chartTime"], errors="coerce")
    df = df.dropna(subset=["chartTime"])
    df["stay_id"] = df["encounterId"].astype(int)
    df["hour"] = hours_since_intime(df["chartTime"], df["stay_id"].map(intime))
    df = (df[df["hour"] >= 0] if clip_hours is None
          else df[(df["hour"] >= 0) & (df["hour"] <= clip_hours)]).copy()

    # Canonical variable name: abx_pip_tazo for start, abx_pip_tazo_end for end
    df["variable"] = df.apply(
        lambda r: _ABX_VAR_MAP[r["variableName"]]
                  + ("_end" if r["doseUnit"] == "flag_end" else ""),
        axis=1,
    )
    df["value"] = 1.0

    df["patient_id"]             = df["stay_id"].map(meta["patient_id"])
    df["hospital_admission_id"]  = df["stay_id"].map(meta["hospital_admission_id"])
    df["unit"]                   = "bool"
    df["dataset"]                = DATASET
    df["measurement_type"]       = "medication"

    result = df[["patient_id", "stay_id", "hospital_admission_id",
                 "hour", "variable", "value", "unit", "dataset",
                 "measurement_type"]].dropna(subset=["patient_id"])
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = result["hospital_admission_id"].astype("Int64")
    log.info("Antibiotics: %d events (%d stays, %d distinct abx variables)",
             len(result), result["stay_id"].nunique(), result["variable"].nunique())
    return result


# ---------------------------------------------------------------------------
# Blood culture events (260528 export — key for Sepsis-3)
# ---------------------------------------------------------------------------

_BLOOD_CULTURE_KEYWORDS = ("Blutkultur", "Blutkulturflasche")


def extract_blood_cultures(
    data_path: str | Path,
    cohort: pd.DataFrame,
    clip_hours: float | None = 168.0,
) -> pd.DataFrame:
    """Extract blood culture order events as time-varying binary feature.

    Reads 260528_UMM_blood_cultures.parquet, filters to blood specimens
    (material_note contains 'Blutkultur' or 'Blutkulturflasche'), and emits
    value=1.0 at the order_time hour. Variable name: 'blood_culture'.

    Used together with extract_antibiotics() for Sepsis-3 suspected-infection:
      - First antibiotic start and first blood culture within ±72 h
      → suspected infection window.
    """
    data_path = Path(data_path)
    bc_path   = data_path / _BLOOD_CULTURES_FILE
    pq_path   = bc_path.with_suffix(".parquet")
    src       = pq_path if pq_path.exists() else bc_path

    stay_ids = set(cohort["stay_id"].astype(int))
    intime   = _intime_lookup(cohort)
    meta     = _stay_meta(cohort)

    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(
        src, sep=";", skiprows=1, dtype=str, low_memory=False
    )
    df = df[~df[df.columns[0]].astype(str).str.startswith("-")].reset_index(drop=True)
    df["encounterId"] = pd.to_numeric(df["encounterId"], errors="coerce")
    df = df[df["encounterId"].isin(stay_ids)].copy()

    # Filter to blood specimens only
    is_blood = df["material_note"].str.contains(
        "|".join(_BLOOD_CULTURE_KEYWORDS), na=False
    )
    df = df[is_blood].copy()

    if df.empty:
        log.warning("extract_blood_cultures: no blood culture events in cohort")
        return pd.DataFrame()

    # Use order_time (most reliable for Sepsis-3 timing); fall back to collection_time
    df["event_time"] = pd.to_datetime(df["order_time"],      errors="coerce")
    fallback_mask    = df["event_time"].isna()
    df.loc[fallback_mask, "event_time"] = pd.to_datetime(
        df.loc[fallback_mask, "collection_time"], errors="coerce"
    )
    df = df.dropna(subset=["event_time"])

    df["stay_id"] = df["encounterId"].astype(int)
    df["hour"]    = hours_since_intime(df["event_time"], df["stay_id"].map(intime))
    df = (df[df["hour"] >= 0] if clip_hours is None
          else df[(df["hour"] >= 0) & (df["hour"] <= clip_hours)]).copy()

    # De-duplicate: one event per stay per hour
    df = df.drop_duplicates(subset=["stay_id", "hour"])

    df["patient_id"]            = df["stay_id"].map(meta["patient_id"])
    df["hospital_admission_id"] = df["stay_id"].map(meta["hospital_admission_id"])
    df["variable"]              = "blood_culture"
    df["value"]                 = 1.0
    df["unit"]                  = "bool"
    df["dataset"]               = DATASET
    df["measurement_type"]      = "microbiology"

    result = df[["patient_id", "stay_id", "hospital_admission_id",
                 "hour", "variable", "value", "unit", "dataset",
                 "measurement_type"]].dropna(subset=["patient_id"])
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = result["hospital_admission_id"].astype("Int64")
    log.info("Blood cultures: %d events in %d stays",
             len(result), result["stay_id"].nunique())
    return result


# ---------------------------------------------------------------------------
# Diagnoses — ICD-10 → Charlson comorbidity + SAPS-II chronic_disease
# ---------------------------------------------------------------------------

def extract_diagnoses(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract Charlson comorbidity index from the UMM diagnoses table (ICD-10).

    Reads 260528_UMM_diagnoses (icd10_code per encounterId), computes the
    Charlson index (Quan 2005 ICD-10 mapping) and SAPS-II chronic_disease
    points per stay. Emits static features at hour=0:
      charlson_comorbidity_index, chronic_disease_sapsii.
    """
    from harmonize.diagnoses import charlson_icd10, chronic_disease_sapsii as _chron

    data_path = Path(data_path)
    dx_path = data_path / _DIAGNOSES_FILE
    pq_path = dx_path.with_suffix(".parquet")
    src     = pq_path if pq_path.exists() else dx_path

    stay_ids = set(cohort["stay_id"].astype(int))
    meta     = _stay_meta(cohort)

    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(
        src, sep=";", skiprows=1, dtype=str, low_memory=False
    )
    df = df[~df[df.columns[0]].astype(str).str.startswith("-")].reset_index(drop=True)
    df["encounterId"] = pd.to_numeric(df["encounterId"], errors="coerce")
    df = df[df["encounterId"].isin(stay_ids)].copy()
    df["icd10_code"] = df["icd10_code"].fillna("").astype(str)

    codes_per_stay = (
        df.groupby("encounterId")["icd10_code"]
        .apply(lambda s: [c for c in s if c.strip()])
        .to_dict()
    )

    rows = []
    for sid, codes in codes_per_stay.items():
        sid = int(sid)
        pid = meta["patient_id"].get(sid)
        haid = meta["hospital_admission_id"].get(sid)
        if pid is None or pd.isna(pid):
            continue
        charlson = charlson_icd10(codes)
        common = {
            "patient_id": int(pid), "stay_id": sid,
            "hospital_admission_id": int(haid) if pd.notna(haid) else pd.NA,
            "hour": 0.0, "dataset": DATASET, "measurement_type": "static",
        }
        rows.append({**common, "variable": "charlson_comorbidity_index",
                     "value": float(charlson["charlson_total"]), "unit": "score"})
        rows.append({**common, "variable": "chronic_disease_sapsii",
                     "value": float(_chron(charlson)), "unit": "saps2_pts"})

    if not rows:
        log.warning("extract_diagnoses (UMM): no rows produced")
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = result["hospital_admission_id"].astype("Int64")
    log.info("Diagnoses (UMM): %d rows from %d stays", len(result), len(codes_per_stay))
    return result


# ---------------------------------------------------------------------------
# Static features
# ---------------------------------------------------------------------------

def extract_static(
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in cohort.iterrows():
        stay_id = int(r["stay_id"])
        pid     = int(r["patient_id"])
        had     = int(r["hospital_admission_id"])

        sex_val      = 1.0 if str(r.get("sex", "")).upper() == "M" else 0.0
        # All-cause death flag (death_time recorded at any point — may be weeks
        # after ICU discharge; see docs §3.6 mortality discussion). `deceased`
        # is numeric (1.0/NaN) after pd.to_numeric in build_cohort.
        deceased_num = pd.to_numeric(r.get("deceased"), errors="coerce")
        deceased_val = 1.0 if pd.notna(deceased_num) and float(deceased_num) == 1.0 else 0.0
        # True ICU mortality: death_time falls within the ICU stay window.
        # The `deceased` flag alone is all-cause (incl. late post-discharge
        # deaths), which over-counts ICU mortality (27.2% vs 19.4% true).
        icu_mort_val = 0.0
        dt    = pd.to_datetime(r.get("death_time"), errors="coerce")
        start = pd.to_datetime(r.get("icu_start"),  errors="coerce")
        end   = pd.to_datetime(r.get("icu_end"),    errors="coerce")
        if pd.notna(dt) and pd.notna(start) and pd.notna(end):
            if start <= dt <= end + pd.Timedelta(hours=1):
                icu_mort_val = 1.0

        common = {
            "patient_id":            pid,
            "stay_id":               stay_id,
            "hospital_admission_id": had,
            "hour":                  0.0,
            "dataset":               DATASET,
        }

        static_feats = [
            ("age",                r["age"],      "demographic"),
            ("sex",                sex_val,       "demographic"),
            ("mortality_icu",      icu_mort_val,  "outcome"),
            ("mortality_hospital", deceased_val,  "outcome"),
        ]

        wt = r.get("weight_kg")
        if pd.notna(wt) and float(wt) > 0:
            static_feats.append(("weight_admission", float(wt), "demographic"))

        ht = r.get("height_cm")
        if pd.notna(ht) and float(ht) > 0:
            static_feats.append(("height", float(ht), "demographic"))

        los = r.get("los_hours")
        if pd.notna(los):
            static_feats.append(("los_hours", float(los), "static"))

        # admission_type: ICCA exports German text with the SAPS code in parentheses,
        # e.g. "Nichtgeplant chirurgisch (8)". Parse the integer SAPS-II type code from
        # admission_type_saps; if absent, fall back to admission_source (260701 export:
        # 99% coverage, a mix of the same SAPS strings + origin labels — parse the code
        # where present). +450 stays / +4.2 pp over admission_type_saps alone (88.7→92.9%),
        # 98.1% agreement where both carry a code. Remaining gaps are origin-only labels.
        adm_raw = r.get("admission_type_saps")
        m = _SAPS_CODE_RE.search(str(adm_raw)) if pd.notna(adm_raw) else None
        if m is None:
            src_raw = r.get("admission_source")
            m = _SAPS_CODE_RE.search(str(src_raw)) if pd.notna(src_raw) else None
        if m:
            static_feats.append(("admission_type", float(int(m.group(1))), "demographic"))

        # icu_type: encode UMM ward codes (e.g. "32-3", "36-4") as stable integers
        icu_t_raw = r.get("icu_unit_first")
        if pd.notna(icu_t_raw):
            code = _UMM_ICU_UNIT_CODES.get(str(icu_t_raw).strip())
            if code is not None:
                static_feats.append(("icu_type", float(code), "demographic"))

        # Antibiotic exposure flags (binary, stay-level — distinct from infusion features)
        for col, feat in (
            ("had_pip_tazo",   "had_pip_tazo"),
            ("had_meropenem",  "had_meropenem"),
            ("had_vancomycin", "had_vancomycin"),
            ("had_linezolid",  "had_linezolid"),
        ):
            v = pd.to_numeric(r.get(col), errors="coerce")
            if pd.notna(v):
                static_feats.append((feat, float(v), "demographic"))

        for feat, val, mtype in static_feats:
            if pd.isna(val):
                continue
            rows.append({
                **common,
                "variable":         feat,
                "value":            float(val),
                "unit":             catalog.unit(feat) if feat in catalog else "",
                "measurement_type": mtype,
            })

    result = pd.DataFrame(rows)
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
    clip_hours: float | None = 168.0,   # None = no horizon clip; non-168 -> separate output file
) -> pd.DataFrame:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    data_path   = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    catalog = FeatureCatalog.from_yaml(features_yaml)
    log.info("Feature catalog loaded: %d features", len(list(catalog.keys())))

    cohort = extract_cohort(data_path, min_stay_hours, sample_size, random_state)

    ts     = extract_timeseries(data_path, cohort, catalog)
    static = extract_static(cohort, catalog)

    abx = extract_antibiotics(data_path, cohort, clip_hours=clip_hours)
    bc  = extract_blood_cultures(data_path, cohort, clip_hours=clip_hours)
    diag = extract_diagnoses(data_path, cohort)

    long_df = pd.concat([df for df in [ts, static, abx, bc, diag] if not df.empty],
                        ignore_index=True)
    long_df = long_df.sort_values(["stay_id", "hour", "variable"]).reset_index(drop=True)

    # Clip observations to the 7-day (168h) modelling horizon, consistent with
    # eICU/HiRID/MIMIC and the Stage-3 168h window cap. UMM has very long referral
    # stays (LOS up to 6187h); without this the long parquet retains weeks of
    # data no window can ever cover. Keeps static at hour=0; cohort LOS intact.
    _n0 = len(long_df)
    if clip_hours is None:
        long_df = long_df.loc[long_df["hour"] >= 0].copy()
        log.info("no horizon clip (complete): %d rows", len(long_df))
    else:
        long_df = long_df.loc[(long_df["hour"] >= 0) & (long_df["hour"] <= clip_hours)].copy()
        log.info("%gh horizon clip: %d -> %d rows (-%d)", clip_hours, _n0, len(long_df),
                 _n0 - len(long_df))

    # Default (168h) keeps umm_long.parquet (shared with Paper-1); any other clip writes a
    # distinct file so the canonical output is never overwritten.
    if clip_hours == 168.0:
        long_name = "umm_long.parquet"
    elif clip_hours is None:
        long_name = "umm_long_complete.parquet"
    else:
        long_name = f"umm_long_clip{int(clip_hours)}h.parquet"
    out_parquet = output_path / long_name
    long_df.to_parquet(out_parquet, index=False)
    log.info("Saved %d rows to %s", len(long_df), out_parquet)

    # The cohort is clip-independent. Only (re)write the canonical umm_cohort.parquet on a
    # default run; non-default clips reuse the existing cohort and never touch shared files.
    cohort_parquet = output_path / "umm_cohort.parquet"
    if clip_hours == 168.0 or not cohort_parquet.exists():
        cohort_out = cohort[[
            "stay_id", "patient_id", "hospital_admission_id",
            "age", "sex", "icu_start", "icu_end", "los_hours", "deceased",
            "death_time", "weight_kg", "height_cm",
        ]].copy()
        cohort_out.to_parquet(cohort_parquet, index=False)
        log.info("Saved cohort (%d stays, %d patients) to %s",
                 len(cohort_out), cohort_out["patient_id"].nunique(), cohort_parquet)
    else:
        log.info("clip_hours=%s: reusing existing %s (not overwritten)", clip_hours, cohort_parquet)

    return long_df
