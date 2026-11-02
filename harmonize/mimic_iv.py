"""
harmonize/mimic_iv.py
Step-1 extraction pipeline for MIMIC-IV v3.1.

Schema output: Long-Format Parquet
  patient_id | stay_id | hospital_admission_id | hour | variable | value | unit | dataset | measurement_type

Design decisions (deviations from Paper 2 / 01_preprocess_mimic.py):
  - 24h minimum stay (not 48h)
  - NaN for outliers (not median imputation)
  - IBP and NIBP as separate features + merged variant
  - Single inputevents table (MIMIC-IV has no CV/MV split)
  - Vasopressor rates in ug/kg/min via patientweight column in inputevents
  - FiO2 NaN if not documented (not 0.21)
  - Katecholamin not documented -> NaN (not 0)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from harmonize.common import (
    FeatureCatalog,
    apply_outlier_filter,
    expand_intervals_hourly_vectorized,
    fahrenheit_to_celsius,
    fio2_percent_to_fraction,
    hours_since_intime,
    lbs_to_kg,
    make_long_df,
)

log = logging.getLogger(__name__)

DATASET = "mimic_iv"
CHUNKSIZE = 100_000

# ---------------------------------------------------------------------------
# Admission-type → SAPS-II points
# SAPS-II: scheduled surgical = 0 pts; unscheduled surgical = 8 pts; medical = 6 pts.
# MIMIC-IV lacks an explicit surgical flag, so we proxy via first_careunit.
# ---------------------------------------------------------------------------
_SURGICAL_UNITS_IV = frozenset({
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Trauma SICU (TSICU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
})

def _admission_type_saps_iv(admission_type: str, first_careunit: str) -> float:
    """Map MIMIC-IV (admission_type, first_careunit) → SAPS-II admission type points."""
    adm = str(admission_type).upper().strip()
    surgical = str(first_careunit) in _SURGICAL_UNITS_IV
    elective = adm in {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}
    if surgical and elective:
        return 0.0   # scheduled surgical
    if surgical and not elective:
        return 8.0   # unscheduled surgical
    return 6.0       # medical (default for non-surgical or unknown)

# ---------------------------------------------------------------------------
# Item-ID registries for MIMIC-IV chartevents
# Each entry: feature_key -> list of itemids, optional pre-filter/conversion fn
# ---------------------------------------------------------------------------

# Chartevents: simple numeric features (value_range from catalog, no conversion)
CHART_NUMERIC: dict[str, list[int]] = {
    "hr":              [220045],
    "rr":              [220210, 224690],
    "spo2":            [220277],
    "sbp_invasive":    [220050],
    "sbp_noninvasive": [220179],
    "dbp_invasive":    [220051],
    "dbp_noninvasive": [220180],
    "mbp_invasive":    [220052],
    "mbp_noninvasive": [220181],
    "gcs_eye":         [220739],
    "gcs_motor":       [223901],
    "gcs_verbal":      [223900],
    "peep":            [220339],
    "amv":             [224687],   # Minute Volume, L/min
    "tidal_volume":    [224685, 224684, 224686],  # observed / set / spontaneous, mL
    "etco2":           [228640],
    "cardiac_output":  [228368],
    "stroke_volume":   [228374],
    "weight":          [224639, 226512],   # kg items; lbs handled separately below
    "height":          [226730],           # cm (226707 = inches handled separately below)
}

# Weight in lbs (needs lbs->kg conversion)
CHART_WEIGHT_LBS: list[int] = [226531]

# Height in inches (needs *2.54 -> cm); 226730 (cm) is already in CHART_NUMERIC
CHART_HEIGHT_IN: list[int] = [226707]

# Temperature: Celsius items (no conversion)
CHART_TEMP_C: list[int] = [223762]
# Temperature: Fahrenheit items (needs F->C conversion)
CHART_TEMP_F: list[int] = [223761]

# FiO2: percent or fraction, needs normalization (>1.0 -> /100)
CHART_FIO2: list[int] = [223835]

# Capillary refill rate: text -> binary
CHART_CRR: list[int] = [223951]

# ---------------------------------------------------------------------------
# Labevents item-ID registries
# ---------------------------------------------------------------------------

LAB_NUMERIC: dict[str, list[int]] = {
    "base_excess":        [50802],
    "hco3":               [50882, 50804],
    "intubated":          [50812],              # Blood Gas panel "Intubated" flag
    "paco2":              [50818],
    "pao2":               [50821],
    "ph_arterial":        [50820],
    "sao2":               [50817],
    "svo2":               [50816],
    "lactate":            [50813],
    "basophils_pct":      [51146],
    "eosinophils_pct":    [51200],
    "hematocrit":         [51221],
    "hemoglobin":         [51222],
    "lymphocytes_pct":    [51244, 51245],
    "lymphocytes_abs":    [51133],
    "mch":                [51248],
    "mchc":               [51249],
    "mcv":                [51250],
    "monocytes_pct":      [51254],
    "mpv":                [52142],  # Mean Platelet Volume (added in v3.1)
    "neutrophils_pct":    [51256],
    "platelets":          [51265],
    "rbc":                [51279],
    "rdw":                [51277],
    "wbc":                [51301, 51300],
    "albumin":            [50862],
    "alp":                [50863],
    "alt":                [50861],
    "anion_gap":          [50868],
    "ast":                [50878],
    "bilirubin_direct":   [50883],
    "bilirubin_indirect": [50884],
    "bilirubin_total":    [50885],
    "bun":                [51006],
    "calcium_free":       [50808],
    "calcium_total":      [50893],
    "anion_gap":          [50868, 52500],          # 52500: 1230 extra rows, Δ16% — likely POC/different cohort
    "chloride":           [50902, 50806, 52535],   # 52535: 1235 extra rows, Δ0.8% — safe add
    "creatinine":         [50912, 52024, 52546],   # 52024 WB POC 15k rows, 52546 1.3k rows
    "creatinine_urine":   [51082],
    "crp":                [50889],
    "fibrinogen":         [51214],                 # "Fibrinogen, Functional" — 50920 (eGFR) was wrong
    "glucose":            [50809, 50931, 52569],   # 52569: 1305 extra rows, Δ10.4%
    "inr":                [51237],
    "ldh":                [50954],
    "lipase":             [50956],
    "lymphocytes_abs":    [51133, 52769],          # 52769 in cells/µL — converted to K/µL via LAB_ITEMID_CONVERSIONS
    "magnesium":          [50960],
    # pct (procalcitonin) is NOT available in MIMIC-IV — no itemid in d_labitems.
    # Previous mapping to 50812 was wrong (50812 = "Intubated").
    "phosphate":          [50970],
    "potassium":          [50971, 50822, 52610],   # 52610: 1307 extra rows, Δ4.3%
    "protein_total":      [50976],
    "pt":                 [51274],
    "ptt":                [51275],
    "bun":                [51006, 52647],          # 52647: 1228 extra rows, Δ25% — POC vs lab
    "sodium":             [50983, 50824, 52623],   # 52623: 1308 extra rows, Δ0.9%
    "troponin_i":         [51002, 52642],          # 51002 empty in v3.1; 52642 carries the 670 measurements
    "troponin_t":         [51003],                 # MIMIC-IV also has Troponin T (Roche)
    # Skipped 52xxx (all 0 rows in v3.1): 52022, 52027, 52034-52042, 52434, 52442, 52452, 52455
}

# Per-itemid value conversions applied during labevents extraction.
# Used when MIMIC-IV reports a parallel itemid in a different unit from the catalog target.
LAB_ITEMID_CONVERSIONS: dict[int, float] = {
    52769: 1 / 1000.0,   # lymphocytes_abs: cells/µL  ->  K/µL  (catalog target = ×10^9/L = K/µL)
}

# Lactate in chartevents (redundant with labevents 50813, use if lab missing)
CHART_LACTATE: list[int] = [225668]

# ---------------------------------------------------------------------------
# Inputevents item-ID registries
# ---------------------------------------------------------------------------

# Vasoactive: unit target is ug/kg/min
INPUT_VASOPRESSORS: dict[str, list[int]] = {
    "norepinephrine": [221906],
    "epinephrine":    [221289],
    "dopamine":       [221662],
    "dobutamine":     [221653],
}

# Vasopressin: unit target is IU/min
INPUT_VASOPRESSIN: dict[str, list[int]] = {
    "vasopressin": [222315],
}

# Other infusions: target units vary (see catalog)
INPUT_OTHER: dict[str, list[int]] = {
    "amiodarone":        [221347, 228339, 229654, 230034],
    "dexmedetomidine":   [225150, 229420],  # rate already in mcg/kg/hour = ug/kg/h
    "fentanyl":          [221744],
    "furosemide":        [221794],
    "heparin":           [225152],
    "insulin":           [223257, 223258],
    "midazolam":         [221668],
    "milrinone":         [221986],
    "morphine":          [225154],
    "pantoprazole":      [225910],
    "piperacillin":      [225893, 225892],
    "propofol":          [222168],
    "vancomycin":        [225798],
}

# ---------------------------------------------------------------------------
# Outputevents item-ID registries
# ---------------------------------------------------------------------------

# Urine output: only urine_output feature
URINE_ITEMIDS: list[int] = [
    226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567,
    226557, 226558, 227488, 227489,
]

# ---------------------------------------------------------------------------
# Procedureevents: Mechanical Ventilation
# ---------------------------------------------------------------------------

MV_ITEMIDS: list[int] = [
    225792,   # Invasive Ventilation
    225794,   # Non-Invasive Ventilation
]

# Ventilator settings in chartevents (for mechanical_ventilation binary flag)
MV_CHART_ITEMIDS: list[int] = [
    720,     # ventilator mode
    223849,  # ventilator mode
    226732,  # oxygen delivery device
    467,     # O2 delivery device
]

# ---------------------------------------------------------------------------
# Cohort extraction
# ---------------------------------------------------------------------------

def extract_cohort(
    data_path: str | Path,
    min_stay_hours: float = 24.0,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load ICU cohort from MIMIC-IV icustays + patients + admissions.

    Inclusion criteria:
      - ICU LOS >= min_stay_hours
      - Adult patients (anchor_age >= 18)
      - Valid intime and outtime

    Multiple stays per patient are allowed; train/test split must be done
    on patient_id level to prevent leakage.

    Returns
    -------
    DataFrame with columns:
      patient_id, stay_id, hospital_admission_id,
      intime, outtime, los_hours,
      gender, age, admission_type, first_careunit
    """
    data_path = Path(data_path)
    log.info("Loading icustays...")
    icu = pd.read_parquet(
        data_path / "icu" / "icustays.parquet",
        columns=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime", "los"],
    )
    icu = icu.dropna(subset=["intime", "outtime"])
    icu["intime"] = pd.to_datetime(icu["intime"])
    icu["outtime"] = pd.to_datetime(icu["outtime"])
    icu["los_hours"] = icu["los"] * 24.0
    icu = icu.loc[icu["los_hours"] >= min_stay_hours].copy()

    log.info("Loading patients...")
    pat = pd.read_parquet(
        data_path / "hosp" / "patients.parquet",
        columns=["subject_id", "gender", "anchor_age", "dod"],
    )
    pat = pat.rename(columns={"anchor_age": "age"})
    # MIMIC-IV uses anchor_age; age is already the age at anchor_year (admission year)
    pat = pat.loc[pat["age"] >= 18]

    log.info("Loading admissions...")
    adm = pd.read_parquet(
        data_path / "hosp" / "admissions.parquet",
        columns=["hadm_id", "admission_type", "hospital_expire_flag", "deathtime"],
    )
    adm["deathtime"] = pd.to_datetime(adm["deathtime"], errors="coerce")

    cohort = icu.merge(pat[["subject_id", "gender", "age", "dod"]], on="subject_id", how="inner")
    cohort = cohort.merge(adm, on="hadm_id", how="left")
    cohort = cohort.rename(columns={
        "subject_id": "patient_id",
        "hadm_id":    "hospital_admission_id",
    })

    log.info(
        "Cohort: %d stays, %d unique patients (min_stay_hours=%.0f)",
        len(cohort), cohort["patient_id"].nunique(), min_stay_hours,
    )

    if sample_size is not None and sample_size < cohort["patient_id"].nunique():
        rng = np.random.default_rng(random_state)
        sampled_patients = rng.choice(
            cohort["patient_id"].unique(), size=sample_size, replace=False
        )
        cohort = cohort.loc[cohort["patient_id"].isin(sampled_patients)].copy()
        log.info("Sampled to %d stays (%d patients)", len(cohort), cohort["patient_id"].nunique())

    return cohort.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _intime_lookup(cohort: pd.DataFrame) -> pd.Series:
    """Return Series: stay_id -> intime."""
    return cohort.set_index("stay_id")["intime"]


def _stay_meta(cohort: pd.DataFrame) -> pd.DataFrame:
    """Return stay_id -> {patient_id, hospital_admission_id}."""
    return cohort.set_index("stay_id")[["patient_id", "hospital_admission_id"]]


def _attach_meta(df: pd.DataFrame, stay_meta: pd.DataFrame) -> pd.DataFrame:
    """Add patient_id and hospital_admission_id columns from stay_meta lookup."""
    df = df.copy()
    df["patient_id"] = df["stay_id"].map(stay_meta["patient_id"])
    df["hospital_admission_id"] = df["stay_id"].map(stay_meta["hospital_admission_id"])
    return df


# ---------------------------------------------------------------------------
# Chartevents extraction
# ---------------------------------------------------------------------------

def extract_chartevents(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract chartevents for cohort stays.

    Handles:
      - Numeric vitals (HR, RR, SpO2, BP variants, GCS, PEEP, AMV, ...)
      - Temperature (Celsius items + Fahrenheit->Celsius conversion)
      - FiO2 (percent->fraction normalization)
      - Capillary Refill Rate (text->binary)
      - Weight (kg items + lbs->kg conversion)

    Returns Long-Format DataFrame (partial — chartevents features only).
    """
    data_path = Path(data_path)
    stay_ids = set(cohort["stay_id"])
    intime = _intime_lookup(cohort)
    stay_meta = _stay_meta(cohort)

    # Build itemid -> (feature_key, conversion_fn | None) lookup
    itemid_map: dict[int, tuple[str, Callable | None]] = {}
    for feat, iids in CHART_NUMERIC.items():
        for iid in iids:
            itemid_map[iid] = (feat, None)
    for iid in CHART_TEMP_C:
        itemid_map[iid] = ("temp_core", None)
    for iid in CHART_TEMP_F:
        itemid_map[iid] = ("temp_core", fahrenheit_to_celsius)
    for iid in CHART_FIO2:
        itemid_map[iid] = ("fio2", fio2_percent_to_fraction)
    for iid in CHART_WEIGHT_LBS:
        itemid_map[iid] = ("weight", lbs_to_kg)
    for iid in CHART_LACTATE:
        itemid_map[iid] = ("lactate", None)

    all_itemids = set(itemid_map.keys()) | set(CHART_CRR)

    parts: list[pd.DataFrame] = []
    _filter_removed = 0
    _filter_total   = 0

    log.info("Reading chartevents (parquet, itemid-filter pushdown)...")
    # Single whole-file read with itemid filter pushed down to parquet.
    # Memory: ~10-30 GB for the filtered subset (fits in 200 GB SLURM allocation).
    for i, chunk in enumerate([pd.read_parquet(
        data_path / "icu" / "chartevents.parquet",
        columns=["stay_id", "hadm_id", "subject_id", "charttime", "itemid", "value", "valuenum", "valueuom", "warning"],
        filters=[("itemid", "in", list(all_itemids))],
    )]):
        log.info("  loaded %d chartevents rows (post itemid-filter)", len(chunk))

        # Filter to cohort and remaining row-level filters
        chunk = chunk.loc[chunk["stay_id"].isin(stay_ids)]
        chunk = chunk.loc[chunk["warning"] != 1]
        chunk = chunk.dropna(subset=["charttime"])
        if chunk.empty:
            continue

        # charttime is already timestamp in parquet (no to_datetime needed)
        chunk["hour"] = hours_since_intime(chunk["charttime"], chunk["stay_id"].map(intime))

        # ---- Numeric features ----
        num_mask = chunk["itemid"].isin(itemid_map)
        ch_num = chunk.loc[num_mask].dropna(subset=["valuenum"]).copy()

        # Group by conversion function to vectorize
        # First: features without conversion
        no_conv = ch_num.loc[ch_num["itemid"].isin(
            [iid for iid, (_, fn) in itemid_map.items() if fn is None]
        )].copy()
        no_conv["variable"] = no_conv["itemid"].map({iid: feat for iid, (feat, _) in itemid_map.items()})
        no_conv["value"] = no_conv["valuenum"]

        # Temp F->C
        temp_f_mask = ch_num["itemid"].isin(CHART_TEMP_F)
        ch_tempf = ch_num.loc[temp_f_mask].copy()
        if not ch_tempf.empty:
            ch_tempf["value"] = fahrenheit_to_celsius(ch_tempf["valuenum"])
            ch_tempf["variable"] = "temp_core"
            no_conv = pd.concat([no_conv, ch_tempf], ignore_index=True)

        # FiO2 %->fraction
        fio2_mask = ch_num["itemid"].isin(CHART_FIO2)
        ch_fio2 = ch_num.loc[fio2_mask].copy()
        if not ch_fio2.empty:
            ch_fio2["value"] = fio2_percent_to_fraction(ch_fio2["valuenum"])
            ch_fio2["variable"] = "fio2"
            no_conv = pd.concat([no_conv, ch_fio2], ignore_index=True)

        # Weight lbs->kg
        wt_lb_mask = ch_num["itemid"].isin(CHART_WEIGHT_LBS)
        ch_wt_lb = ch_num.loc[wt_lb_mask].copy()
        if not ch_wt_lb.empty:
            ch_wt_lb["value"] = lbs_to_kg(ch_wt_lb["valuenum"])
            ch_wt_lb["variable"] = "weight"
            no_conv = pd.concat([no_conv, ch_wt_lb], ignore_index=True)

        # Height inches->cm (1 inch = 2.54 cm)
        ht_in_mask = ch_num["itemid"].isin(CHART_HEIGHT_IN)
        ch_ht_in = ch_num.loc[ht_in_mask].copy()
        if not ch_ht_in.empty:
            ch_ht_in["value"] = ch_ht_in["valuenum"] * 2.54
            ch_ht_in["variable"] = "height"
            no_conv = pd.concat([no_conv, ch_ht_in], ignore_index=True)

        if no_conv.empty:
            numeric_long = pd.DataFrame()
        else:
            # Apply outlier filter per variable — accumulate removed counts
            no_conv = no_conv.dropna(subset=["variable"])
            _n_pre = len(no_conv)
            filtered_parts: list[pd.DataFrame] = []
            for feat_key, grp in no_conv.groupby("variable"):
                vr = catalog.value_range(feat_key) if feat_key in catalog else None
                grp = apply_outlier_filter(grp, "value", vr)
                filtered_parts.append(grp)
            no_conv = pd.concat(filtered_parts, ignore_index=True) if filtered_parts else pd.DataFrame()
            no_conv = no_conv.dropna(subset=["value"])
            _filter_removed += _n_pre - len(no_conv)
            _filter_total   += _n_pre

            numeric_long = no_conv[["stay_id", "hour", "variable", "value"]].copy()
            numeric_long = _attach_meta(numeric_long, stay_meta)
            numeric_long["unit"] = numeric_long["variable"].map(
                {k: catalog.unit(k) for k in catalog.keys() if k in CHART_NUMERIC or k in ["temp_core", "fio2", "weight", "lactate"]}
            )
            numeric_long["measurement_type"] = numeric_long["variable"].map(
                {k: catalog.measurement_type(k) for k in catalog.keys()}
            )

        # ---- CRR (text -> binary) ----
        crr_chunk = chunk.loc[chunk["itemid"].isin(CHART_CRR)].copy()
        crr_rows: list[pd.DataFrame] = []
        if not crr_chunk.empty:
            normal_mask = crr_chunk["value"].isin(["Normal <3 Seconds", "Normal <3 secs"])
            abnormal_mask = crr_chunk["value"].isin(["Abnormal >3 Seconds", "Abnormal >3 secs"])
            crr_valid = crr_chunk.loc[normal_mask | abnormal_mask].copy()
            crr_valid["value"] = np.where(normal_mask.loc[crr_valid.index], 0.0, 1.0)
            crr_valid["variable"] = "crr"
            crr_valid["unit"] = "bool"
            crr_valid["measurement_type"] = "vital"
            crr_valid = _attach_meta(crr_valid[["stay_id", "hour", "variable", "value", "unit", "measurement_type"]], stay_meta)
            crr_rows.append(crr_valid)

        chunk_parts = [p for p in [numeric_long] + crr_rows if not p.empty]
        if chunk_parts:
            parts.append(pd.concat(chunk_parts, ignore_index=True))

    if not parts:
        log.warning("No chartevents data found for cohort.")
        return pd.DataFrame(columns=["patient_id", "stay_id", "hospital_admission_id",
                                     "hour", "variable", "value", "unit", "dataset", "measurement_type"])

    result = pd.concat(parts, ignore_index=True)
    result["dataset"] = DATASET
    log.info(
        "Chartevents: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed, _filter_total,
        100.0 * _filter_removed / _filter_total if _filter_total else 0,
    )
    return result


# ---------------------------------------------------------------------------
# Labevents extraction
# ---------------------------------------------------------------------------

def extract_labevents(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract labevents for cohort admissions.

    Labevents are linked to hadm_id (not stay_id). We assign to a stay by
    matching charttime within [intime, outtime] of each stay. If a lab falls
    in a gap between stays for the same patient, it is discarded.

    Returns Long-Format DataFrame (labevents features only).
    """
    data_path = Path(data_path)
    hadm_ids = set(cohort["hospital_admission_id"])
    stay_meta = _stay_meta(cohort)

    all_itemids = set()
    for iids in LAB_NUMERIC.values():
        all_itemids.update(iids)

    # Reverse map: itemid -> feature_key
    itemid_to_feat: dict[int, str] = {}
    for feat, iids in LAB_NUMERIC.items():
        for iid in iids:
            if iid not in itemid_to_feat:  # first mapping wins
                itemid_to_feat[iid] = feat

    log.info("Reading labevents...")
    la = pd.read_parquet(
        data_path / "hosp" / "labevents.parquet",
        columns=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
    )
    la = la.loc[la["hadm_id"].isin(hadm_ids)]
    la = la.loc[la["itemid"].isin(all_itemids)]
    la = la.dropna(subset=["charttime", "valuenum"])
    la["charttime"] = pd.to_datetime(la["charttime"])
    la["hadm_id"] = la["hadm_id"].astype("Int64")
    log.info("  labevents after filter: %d rows", len(la))

    # Assign stay_id vectorized: merge on hadm_id, then filter charttime within [intime, outtime]
    intervals = cohort[["hospital_admission_id", "stay_id", "intime", "outtime"]].copy()
    intervals["hospital_admission_id"] = intervals["hospital_admission_id"].astype("Int64")
    la = la.merge(intervals, left_on="hadm_id", right_on="hospital_admission_id", how="left")
    in_window = (la["charttime"] >= la["intime"]) & (la["charttime"] <= la["outtime"])
    la = la.loc[in_window].copy()
    la["hour"] = (la["charttime"] - la["intime"]).dt.total_seconds() / 3600.0
    la["stay_id"] = la["stay_id"].astype(int)

    la["variable"] = la["itemid"].map(itemid_to_feat)
    la = la.dropna(subset=["variable"])

    # Per-itemid unit conversion (for parallel 52000-series IDs in a different unit)
    if LAB_ITEMID_CONVERSIONS:
        conv_mask = la["itemid"].isin(LAB_ITEMID_CONVERSIONS)
        if conv_mask.any():
            factors = la.loc[conv_mask, "itemid"].map(LAB_ITEMID_CONVERSIONS)
            la.loc[conv_mask, "valuenum"] = la.loc[conv_mask, "valuenum"].astype("float64") * factors.astype("float64")

    # Apply outlier filter per variable
    _n_pre = len(la)
    filtered_parts: list[pd.DataFrame] = []
    for feat_key, grp in la.groupby("variable"):
        vr = catalog.value_range(feat_key) if feat_key in catalog else None
        grp = apply_outlier_filter(grp.copy(), "valuenum", vr)
        filtered_parts.append(grp)
    la = pd.concat(filtered_parts, ignore_index=True) if filtered_parts else la

    la["value"] = la["valuenum"]
    la = _attach_meta(la, stay_meta)
    la["unit"] = la["variable"].map({k: catalog.unit(k) for k in catalog.keys()})
    la["measurement_type"] = la["variable"].map({k: catalog.measurement_type(k) for k in catalog.keys()})
    la["dataset"] = DATASET

    result = la[["patient_id", "stay_id", "hospital_admission_id",
                 "hour", "variable", "value", "unit", "dataset", "measurement_type"]].copy()
    # apply_outlier_filter NULLS values and returns the same number of rows, and `result`
    # is only a column subset, so `_n_pre - len(result)` was identically zero. Count the
    # NaN the filter introduced (valuenum was dropna'd before it ran).
    _filter_removed = int(result["value"].isna().sum())
    log.info(
        "Labevents: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed, _n_pre, 100.0 * _filter_removed / _n_pre if _n_pre else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Inputevents extraction
# ---------------------------------------------------------------------------

def _normalize_vasopressor_rate(
    df: pd.DataFrame,
    feature_key: str,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Convert vasopressor rates to ug/kg/min.

    MIMIC-IV inputevents `rateuom` for vasopressors is typically:
      - 'mcg/kg/min'  -> already correct
      - 'mcg/min'     -> divide by patientweight
      - 'mg/min'      -> multiply by 1000, then divide by patientweight
      - 'mg/hour'     -> multiply by 1000/60, then divide by patientweight
      - 'mL/hour'     -> volume rate, not dose rate — discard (NaN)

    vasopressin is handled separately (target: IU/min).
    """
    df = df.copy()
    df["value"] = np.nan
    if "rate" not in df.columns or "rateuom" not in df.columns:
        return df

    rate = df["rate"].copy()
    uom = df["rateuom"].astype(str).str.lower()
    wt = df["patientweight"].copy()

    already_correct = uom.str.contains("mcg/kg/min", na=False)
    per_min_no_wt = uom.str.contains(r"mcg/min$", na=False, regex=True)
    mg_per_min = uom.str.contains(r"mg/min$", na=False, regex=True)
    mg_per_hr = uom.str.contains(r"mg/hour", na=False)

    df.loc[already_correct, "value"] = rate.loc[already_correct]
    df.loc[per_min_no_wt, "value"] = rate.loc[per_min_no_wt] / wt.loc[per_min_no_wt]
    df.loc[mg_per_min, "value"] = (rate.loc[mg_per_min] * 1000.0) / wt.loc[mg_per_min]
    df.loc[mg_per_hr, "value"] = (rate.loc[mg_per_hr] * 1000.0 / 60.0) / wt.loc[mg_per_hr]
    # mL/hour and unknown -> remain NaN

    vr = catalog.value_range(feature_key)
    df = apply_outlier_filter(df, "value", vr)
    return df


def _normalize_vasopressin_rate(df: pd.DataFrame, catalog: FeatureCatalog) -> pd.DataFrame:
    """Convert vasopressin rates to IU/min."""
    df = df.copy()
    df["value"] = np.nan
    rate = df["rate"].copy()
    uom = df["rateuom"].astype(str).str.lower()

    u_per_min = uom.str.contains(r"unit/min|u/min|iu/min", na=False, regex=True)
    u_per_hr = uom.str.contains(r"unit/hour|u/hour|iu/hour", na=False, regex=True)

    df.loc[u_per_min, "value"] = rate.loc[u_per_min]
    df.loc[u_per_hr, "value"] = rate.loc[u_per_hr] / 60.0

    vr = catalog.value_range("vasopressin")
    df = apply_outlier_filter(df, "value", vr)
    return df


def _normalize_other_rate(
    df: pd.DataFrame,
    feature_key: str,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Use rate column for non-vasopressor infusions, apply outlier filter."""
    df = df.copy()
    df["value"] = df["rate"].copy()
    vr = catalog.value_range(feature_key)
    df = apply_outlier_filter(df, "value", vr)
    return df


def extract_inputevents(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract continuous infusion events from MIMIC-IV inputevents.

    Strategy:
      1. Load inputevents for cohort stays.
      2. Expand [starttime, endtime] intervals to integer hourly points (vectorized).
      3. For vasopressors: compute rate in ug/kg/min using patientweight.
      4. For vasopressin: compute rate in IU/min.
      5. For other infusions: use rate column as-is.
      6. Apply outlier filter (values outside catalog range -> NaN).
      7. NaN = drug not running that hour (documented absence or below detection).

    Note: hours where no row exists mean drug was not documented (also NaN
    after wide-format aggregation in Step 2). This is consistent with NaN
    convention: no documentation -> NaN, not 0.
    """
    data_path = Path(data_path)
    stay_ids = set(cohort["stay_id"])
    intime = _intime_lookup(cohort)
    stay_meta = _stay_meta(cohort)

    all_itemids = set()
    for iids in [*INPUT_VASOPRESSORS.values(),
                 *INPUT_VASOPRESSIN.values(),
                 *INPUT_OTHER.values()]:
        all_itemids.update(iids)

    log.info("Reading inputevents...")
    ie = pd.read_parquet(
        data_path / "icu" / "inputevents.parquet",
        columns=[
            "subject_id", "hadm_id", "stay_id", "starttime", "endtime",
            "itemid", "rate", "rateuom", "patientweight", "statusdescription",
        ],
    )
    ie = ie.loc[ie["stay_id"].isin(stay_ids)]
    ie = ie.loc[ie["itemid"].isin(all_itemids)]
    # exclude cancelled orders
    ie = ie.loc[ie["statusdescription"] != "Rewritten"].copy()
    ie = ie.dropna(subset=["starttime", "endtime"])
    log.info("  inputevents after filter: %d rows", len(ie))

    if ie.empty:
        return pd.DataFrame()

    # Build reverse lookup: itemid -> (feature_key, category)
    itemid_to_info: dict[int, tuple[str, str]] = {}
    for feat, iids in INPUT_VASOPRESSORS.items():
        for iid in iids:
            itemid_to_info[iid] = (feat, "vasopressor")
    for feat, iids in INPUT_VASOPRESSIN.items():
        for iid in iids:
            itemid_to_info[iid] = (feat, "vasopressin")
    for feat, iids in INPUT_OTHER.items():
        for iid in iids:
            if iid not in itemid_to_info:
                itemid_to_info[iid] = (feat, "other")

    ie["feature_key"] = ie["itemid"].map({iid: feat for iid, (feat, _) in itemid_to_info.items()})
    ie["category"] = ie["itemid"].map({iid: cat for iid, (_, cat) in itemid_to_info.items()})

    parts: list[pd.DataFrame] = []
    _filter_removed_ie = 0
    _filter_total_ie   = 0

    for feat_key, grp in ie.groupby("feature_key"):
        category = grp["category"].iloc[0]
        # Normalize rate
        _n_pre_ie = len(grp)
        if category == "vasopressor":
            grp = _normalize_vasopressor_rate(grp, feat_key, catalog)
        elif category == "vasopressin":
            grp = _normalize_vasopressin_rate(grp, catalog)
        else:
            grp = _normalize_other_rate(grp, feat_key, catalog)

        grp = grp.dropna(subset=["value"])
        _filter_removed_ie += _n_pre_ie - len(grp)
        _filter_total_ie   += _n_pre_ie
        if grp.empty:
            continue

        # Expand intervals to hourly points
        expanded = expand_intervals_hourly_vectorized(
            grp,
            start_col="starttime",
            end_col="endtime",
            intime_lookup=intime,
            stay_id_col="stay_id",
            value_col="value",
        )
        if expanded.empty:
            continue
        expanded["variable"] = feat_key
        expanded["unit"] = catalog.unit(feat_key) if feat_key in catalog else ""
        expanded["measurement_type"] = catalog.measurement_type(feat_key) if feat_key in catalog else "medication"
        parts.append(expanded)

    if not parts:
        log.warning("No inputevents data found for cohort.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result = _attach_meta(result, stay_meta)
    result["dataset"] = DATASET

    result = result[["patient_id", "stay_id", "hospital_admission_id",
                      "hour", "variable", "value", "unit", "dataset", "measurement_type"]].copy()
    log.info(
        "Inputevents: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed_ie, _filter_total_ie,
        100.0 * _filter_removed_ie / _filter_total_ie if _filter_total_ie else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Outputevents extraction
# ---------------------------------------------------------------------------

def extract_outputevents(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract urine output from MIMIC-IV outputevents.

    Only urine_output is extracted here (itemids matching urine-related
    output items). fluid_balance is a derived feature computed in Step 3.

    Returns Long-Format rows for urine_output (mL per charted event).
    """
    data_path = Path(data_path)
    stay_ids = set(cohort["stay_id"])
    intime = _intime_lookup(cohort)
    stay_meta = _stay_meta(cohort)

    # Supplement URINE_ITEMIDS by scanning d_items labels at runtime
    d_items = pd.read_parquet(
        data_path / "icu" / "d_items.parquet",
        columns=["itemid", "label", "unitname"],
    )
    d_items["label_lc"] = d_items["label"].str.lower().fillna("")
    urine_keywords = ["urine", "foley", "void", "nephrostomy", "condom", "drainage bag"]
    cond = pd.concat(
        [d_items["label_lc"].str.contains(k) for k in urine_keywords], axis=1
    ).any(axis=1)
    dynamic_urine_ids = set(d_items.loc[cond, "itemid"].tolist())
    urine_ids = set(URINE_ITEMIDS) | dynamic_urine_ids

    oe = pd.read_parquet(
        data_path / "icu" / "outputevents.parquet",
        columns=["stay_id", "hadm_id", "subject_id", "charttime", "itemid", "value", "valueuom"],
    )
    oe = oe.loc[oe["stay_id"].isin(stay_ids)]
    oe = oe.loc[oe["itemid"].isin(urine_ids)]
    oe = oe.dropna(subset=["value", "charttime"])
    oe["valuenum"] = pd.to_numeric(oe["value"], errors="coerce")
    oe = oe.dropna(subset=["valuenum"])
    oe["charttime"] = pd.to_datetime(oe["charttime"])
    oe["hour"] = hours_since_intime(oe["charttime"], oe["stay_id"].map(intime))

    _n_pre = len(oe)
    oe = apply_outlier_filter(oe, "valuenum", catalog.value_range("urine_output") if "urine_output" in catalog else (0, 2500))
    oe["variable"] = "urine_output"
    oe["value"] = oe["valuenum"]
    oe = _attach_meta(oe, stay_meta)
    oe["unit"] = "mL"
    oe["measurement_type"] = "fluid"
    oe["dataset"] = DATASET

    result = oe[["patient_id", "stay_id", "hospital_admission_id",
                 "hour", "variable", "value", "unit", "dataset", "measurement_type"]].copy()
    # apply_outlier_filter NULLS values and returns the same number of rows, and `result`
    # is only a column subset, so `_n_pre - len(result)` was identically zero. Count the
    # NaN the filter introduced (valuenum was dropna'd before it ran).
    _filter_removed = int(result["value"].isna().sum())
    log.info(
        "Outputevents: %d rows | value_range filter removed %d / %d (%.2f%%)",
        len(result), _filter_removed, _n_pre, 100.0 * _filter_removed / _n_pre if _n_pre else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Procedureevents extraction (Mechanical Ventilation)
# ---------------------------------------------------------------------------

def extract_procedureevents(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract mechanical ventilation intervals from procedureevents.

    Expands [starttime, endtime] intervals to integer hourly binary flags
    (mechanical_ventilation = 1.0 for each ventilated hour).

    Returns Long-Format rows for mechanical_ventilation.
    """
    data_path = Path(data_path)
    stay_ids = set(cohort["stay_id"])
    intime = _intime_lookup(cohort)
    stay_meta = _stay_meta(cohort)

    pe = pd.read_parquet(
        data_path / "icu" / "procedureevents.parquet",
        columns=["stay_id", "hadm_id", "subject_id", "starttime", "endtime", "itemid"],
    )
    pe = pe.loc[pe["stay_id"].isin(stay_ids)]
    pe = pe.loc[pe["itemid"].isin(MV_ITEMIDS)]
    pe = pe.dropna(subset=["starttime", "endtime"])
    pe["value"] = 1.0

    if pe.empty:
        return pd.DataFrame()

    expanded = expand_intervals_hourly_vectorized(
        pe,
        start_col="starttime",
        end_col="endtime",
        intime_lookup=intime,
        stay_id_col="stay_id",
        value_col="value",
    )
    if expanded.empty:
        return pd.DataFrame()

    expanded["variable"] = "mechanical_ventilation"
    expanded = _attach_meta(expanded, stay_meta)
    expanded["unit"] = "bool"
    expanded["measurement_type"] = "ventilation"
    expanded["dataset"] = DATASET

    result = expanded[["patient_id", "stay_id", "hospital_admission_id",
                        "hour", "variable", "value", "unit", "dataset", "measurement_type"]].copy()
    log.info("Procedureevents (MV): %d rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Diagnoses — ICD-9/10 → Charlson comorbidity + SAPS-II chronic_disease
# ---------------------------------------------------------------------------

def extract_diagnoses(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract Charlson comorbidity index from MIMIC-IV diagnoses_icd (mixed ICD-9/10).

    Variables: charlson_comorbidity_index, chronic_disease_sapsii.
    """
    from harmonize.diagnoses import charlson_mixed, chronic_disease_sapsii as _chron

    data_path = Path(data_path)
    hosp = data_path / "hosp"
    pq  = hosp / "diagnoses_icd.parquet"
    gz  = hosp / "diagnoses_icd.csv.gz"
    src = pq if pq.exists() else gz
    log.info("Loading diagnoses_icd from %s", src.name)

    df = (pd.read_parquet(src, columns=["hadm_id", "icd_code", "icd_version"])
          if src.suffix == ".parquet"
          else pd.read_csv(src, usecols=["hadm_id", "icd_code", "icd_version"],
                           dtype={"icd_code": str, "icd_version": int}))
    df["icd_code"] = df["icd_code"].fillna("").astype(str)

    hadm_in_cohort = set(cohort["hospital_admission_id"].dropna().astype(int))
    df = df[df["hadm_id"].isin(hadm_in_cohort)].copy()

    hadm_to_stays = (
        cohort[["stay_id", "hospital_admission_id", "patient_id"]]
        .dropna(subset=["hospital_admission_id"])
        .assign(hospital_admission_id=lambda d: d["hospital_admission_id"].astype(int))
        .groupby("hospital_admission_id")
        .apply(lambda g: g[["stay_id", "patient_id"]].to_dict("records"),
               include_groups=False)
        .to_dict()
    )

    grp = df.groupby("hadm_id")
    rows = []
    for hadm, g in grp:
        codes    = g["icd_code"].tolist()
        versions = g["icd_version"].tolist()
        charlson = charlson_mixed(codes, versions)
        for stay in hadm_to_stays.get(int(hadm), []):
            common = {
                "patient_id": stay["patient_id"],
                "stay_id": int(stay["stay_id"]),
                "hospital_admission_id": int(hadm),
                "hour": 0.0, "dataset": DATASET,
                "measurement_type": "static",
            }
            rows.append({**common, "variable": "charlson_comorbidity_index",
                         "value": float(charlson["charlson_total"]), "unit": "score"})
            rows.append({**common, "variable": "chronic_disease_sapsii",
                         "value": float(_chron(charlson)), "unit": "saps2_pts"})

    if not rows:
        log.warning("extract_diagnoses (MIMIC-IV): no rows produced")
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    log.info("Diagnoses (MIMIC-IV): %d rows from %d admissions", len(result), grp.ngroups)
    return result


# ---------------------------------------------------------------------------
# Antibiotic start events — for Sepsis-3 suspected infection
# ---------------------------------------------------------------------------

# MIMIC-IV INPUTEVENTS itemids (MetaVision-only schema, overlaps with MIMIC-III MV)
_ABX_ITEMIDS_IV: dict[int, str] = {
    225893: "abx_pip_tazo",   225892: "abx_pip_tazo",
    225798: "abx_vancomycin",
    225883: "abx_meropenem",
    225851: "abx_cefepime",
    228000: "abx_linezolid",
    225879: "abx_amp_sulbactam",
    225823: "abx_metronidazole",
    225875: "abx_ciprofloxacin",
    225886: "abx_levofloxacin",
    225931: "abx_fluconazole",
}


def extract_antibiotic_events(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract antibiotic first-dose start events from MIMIC-IV inputevents.

    Emits value=1.0 at the hour of first administration start per (stay, drug).
    Variable names: abx_pip_tazo, abx_vancomycin, abx_meropenem, etc.
    """
    data_path = Path(data_path)
    icu_dir = data_path / "icu"
    pq = icu_dir / "inputevents.parquet"
    csv = icu_dir / "inputevents.csv.gz"
    src = pq if pq.exists() else csv
    log.info("Loading antibiotic events from %s (MIMIC-IV)", src.name)

    stay_ids = set(cohort["stay_id"].astype(int))
    intime   = cohort.set_index("stay_id")["intime"]
    meta     = _stay_meta(cohort)

    cols = ["stay_id", "starttime", "itemid"]
    df = (pd.read_parquet(src, columns=cols)
          if src.suffix == ".parquet"
          else pd.read_csv(src, usecols=cols, dtype={"itemid": int}))

    df = df[df["itemid"].isin(_ABX_ITEMIDS_IV)].copy()
    df = df[df["stay_id"].isin(stay_ids)].copy()
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
    df = df.dropna(subset=["starttime"])

    intime_dt = pd.to_datetime(intime)
    df["hour"] = df.apply(
        lambda r: (r["starttime"] - intime_dt.get(r["stay_id"])).total_seconds() / 3600
        if r["stay_id"] in intime_dt.index else float("nan"), axis=1
    )
    df["variable"] = df["itemid"].map(_ABX_ITEMIDS_IV)
    df = df[(df["hour"] >= -24.0) & (df["hour"] <= 168.0)]
    df = df.sort_values("hour").drop_duplicates(subset=["stay_id", "variable"])

    df["patient_id"]            = df["stay_id"].map(lambda s: meta.get(s, {}).get("patient_id"))
    df["hospital_admission_id"] = df["stay_id"].map(lambda s: meta.get(s, {}).get("hospital_admission_id"))
    df["value"]           = 1.0
    df["unit"]            = "bool"
    df["dataset"]         = DATASET
    df["measurement_type"] = "medication"

    result = df[["patient_id", "stay_id", "hospital_admission_id",
                 "hour", "variable", "value", "unit", "dataset",
                 "measurement_type"]].dropna(subset=["patient_id"])
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = result["hospital_admission_id"].astype("Int64")
    log.info("Antibiotic events (MIMIC-IV): %d events, %d stays, %d drugs",
             len(result), result["stay_id"].nunique(), result["variable"].nunique())
    return result


# ---------------------------------------------------------------------------
# Microbiology events — blood cultures (for Sepsis-3 suspected infection)
# ---------------------------------------------------------------------------

def extract_microbiologyevents(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract blood culture order events from MIMIC-IV microbiologyevents.

    Reads microbiologyevents.csv.gz (or .parquet), filters to
    spec_type_desc == 'BLOOD CULTURE', emits value=1.0 at the hour of the
    culture relative to ICU intime. Variable: 'blood_culture'.

    CHARTTIME fallback: rows with NaN charttime use chartdate + 12h.
    hadm_id can be NaN in MIMIC-IV microbiologyevents for outpatient records;
    these are dropped. ICU-stay assignment follows the same range-join logic
    as MIMIC-III: containing [intime, outtime] window preferred, closest by
    |t_event - intime| as fallback (captures ED cultures ≤24h pre-admission).
    """
    data_path = Path(data_path)
    # MIMIC-IV raw data lives under physionet.org/files/mimiciv/3.1/hosp/
    hosp_dir = data_path / "hosp"
    pq_path  = hosp_dir / "microbiologyevents.parquet"
    gz_path  = hosp_dir / "microbiologyevents.csv.gz"
    csv_path = hosp_dir / "microbiologyevents.csv"
    src = pq_path if pq_path.exists() else (gz_path if gz_path.exists() else csv_path)

    log.info("Reading microbiologyevents from %s", src.name)
    cols = ["subject_id", "hadm_id", "chartdate", "charttime", "spec_type_desc"]
    if src.suffix == ".parquet":
        df = pd.read_parquet(src, columns=cols)
    else:
        df = pd.read_csv(src, usecols=cols, dtype={"subject_id": "Int64", "hadm_id": "Int64"})

    df = df[df["spec_type_desc"] == "BLOOD CULTURE"].copy()
    df = df.dropna(subset=["hadm_id"])
    log.info("  blood cultures (MIMIC-IV): %d rows, %d admissions",
             len(df), df["hadm_id"].nunique())

    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["chartdate"] = pd.to_datetime(df["chartdate"], errors="coerce")
    fallback = df["charttime"].isna()
    df.loc[fallback, "charttime"] = df.loc[fallback, "chartdate"] + pd.Timedelta(hours=12)
    df = df.dropna(subset=["charttime"])
    df["hadm_id"] = df["hadm_id"].astype(int)

    hadm_in_cohort = set(cohort["hospital_admission_id"].dropna().astype(int))
    df = df[df["hadm_id"].isin(hadm_in_cohort)].copy()
    log.info("  after cohort filter: %d rows", len(df))

    stay_info = (
        cohort[["stay_id", "hospital_admission_id", "intime", "outtime"]]
        .dropna(subset=["intime", "outtime"]).copy()
    )
    stay_info["hospital_admission_id"] = stay_info["hospital_admission_id"].astype(int)
    stay_info["intime"]  = pd.to_datetime(stay_info["intime"])
    stay_info["outtime"] = pd.to_datetime(stay_info["outtime"])
    hadm_to_stays = stay_info.groupby("hospital_admission_id").apply(
        lambda g: g[["stay_id", "intime", "outtime"]].to_dict("records"),
        include_groups=False,
    ).to_dict()

    stay_meta = _stay_meta(cohort)
    rows: list[dict] = []
    for _, ev in df.iterrows():
        hadm  = int(ev["hadm_id"])
        t_ev  = ev["charttime"]
        stays = hadm_to_stays.get(hadm, [])
        if not stays:
            continue
        assigned = next((s for s in stays if s["intime"] <= t_ev <= s["outtime"]), None)
        if assigned is None:
            assigned = min(stays, key=lambda s: abs((t_ev - s["intime"]).total_seconds()))
        sid  = int(assigned["stay_id"])
        hour = (t_ev - assigned["intime"]).total_seconds() / 3600.0
        if hour < -24.0 or hour > 336.0:
            continue
        meta = stay_meta.get(sid, {})
        rows.append({
            "patient_id":            meta.get("patient_id"),
            "stay_id":               sid,
            "hospital_admission_id": meta.get("hospital_admission_id"),
            "hour":                  hour,
            "variable":              "blood_culture",
            "value":                 1.0,
            "unit":                  "bool",
            "dataset":               DATASET,
            "measurement_type":      "microbiology",
        })

    if not rows:
        log.warning("extract_microbiologyevents (MIMIC-IV): no events matched cohort")
        return pd.DataFrame()

    result = pd.DataFrame(rows).dropna(subset=["patient_id"])
    result = result.drop_duplicates(subset=["stay_id", "hour"])
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = result["hospital_admission_id"].astype("Int64")
    log.info("Blood cultures (MIMIC-IV): %d events in %d stays (%.1f%% of cohort)",
             len(result), result["stay_id"].nunique(),
             100 * result["stay_id"].nunique() / len(cohort))
    return result


# ---------------------------------------------------------------------------
# Static features extraction
# ---------------------------------------------------------------------------

def extract_static(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract static/demographic features: age, sex, mortality_icu, los_hours, height (incl. OMR)."""
    data_path = Path(data_path)
    rows: list[dict] = []

    for _, row in cohort.iterrows():
        pid = int(row["patient_id"])
        sid = int(row["stay_id"])
        haid = int(row["hospital_admission_id"]) if pd.notna(row["hospital_admission_id"]) else pd.NA
        demo = {
            "patient_id": pid, "stay_id": sid, "hospital_admission_id": haid,
            "hour": 0.0, "dataset": DATASET, "measurement_type": "demographic",
        }
        if pd.notna(row.get("age")):
            rows.append({**demo, "variable": "age", "value": float(row["age"]), "unit": "years"})
        if pd.notna(row.get("gender")):
            gender_val = 1.0 if str(row["gender"]).upper() == "M" else 0.0
            rows.append({**demo, "variable": "sex", "value": gender_val, "unit": "M/F"})
        # mortality_icu: true ICU death = deathtime within [intime, outtime+1h].
        # (hospital_expire_flag is hospital mortality, kept as mortality_hospital.)
        dt    = pd.to_datetime(row.get("deathtime"), errors="coerce")
        start = pd.to_datetime(row.get("intime"),    errors="coerce")
        end   = pd.to_datetime(row.get("outtime"),   errors="coerce")
        icu_mort = 0.0
        if pd.notna(dt) and pd.notna(start) and pd.notna(end) and start <= dt <= end + pd.Timedelta(hours=1):
            icu_mort = 1.0
        rows.append({**demo, "measurement_type": "outcome",
                     "variable": "mortality_icu", "value": icu_mort, "unit": "bool"})
        expire = row.get("hospital_expire_flag")
        if pd.notna(expire):
            rows.append({**demo, "measurement_type": "outcome",
                         "variable": "mortality_hospital", "value": float(expire), "unit": "bool"})
        if pd.notna(row.get("los_hours")):
            rows.append({**demo, "variable": "los_hours",
                         "value": float(row["los_hours"]), "unit": "h"})
        if pd.notna(row.get("weight_admission_kg")):
            rows.append({**demo, "variable": "weight_admission",
                         "value": float(row["weight_admission_kg"]), "unit": "kg"})
        adm_type = row.get("admission_type")
        careunit = row.get("first_careunit")
        if pd.notna(adm_type) and pd.notna(careunit):
            pts = _admission_type_saps_iv(str(adm_type), str(careunit))
            rows.append({**demo, "variable": "admission_type",
                         "value": pts, "unit": "saps2_pts"})

    # OMR height: "Height (Inches)" -> * 2.54 -> cm, stored at hour=0
    omr_path = data_path / "hosp" / "omr.parquet"
    if omr_path.exists():
        omr = pd.read_parquet(omr_path, columns=["subject_id", "result_name", "result_value"])
        omr = omr[omr["result_name"].isin(["Height (Inches)", "Height"])]
        omr["result_value"] = pd.to_numeric(omr["result_value"], errors="coerce")
        omr = omr.dropna(subset=["result_value"])
        # Convert inches to cm; "Height" entries are assumed to be in cm
        inches_mask = omr["result_name"] == "Height (Inches)"
        omr.loc[inches_mask, "result_value"] = omr.loc[inches_mask, "result_value"] * 2.54
        # Keep one height per subject (median across multiple measurements)
        omr_ht = omr.groupby("subject_id")["result_value"].median().reset_index()
        omr_ht.columns = ["patient_id", "height_cm"]
        # Filter to plausible range (100–250 cm)
        omr_ht = omr_ht[(omr_ht["height_cm"] >= 100) & (omr_ht["height_cm"] <= 250)]
        # Map patient_id to stays in cohort
        pid_to_stays = cohort.groupby("patient_id")[
            ["stay_id", "hospital_admission_id"]
        ].apply(lambda g: g.to_dict("records"), include_groups=False)
        n_omr = 0
        for _, omr_row in omr_ht.iterrows():
            pid = int(omr_row["patient_id"])
            if pid not in pid_to_stays.index:
                continue
            for stay in pid_to_stays[pid]:
                sid = stay["stay_id"]
                haid = stay["hospital_admission_id"]
                rows.append({
                    "patient_id": pid, "stay_id": sid,
                    "hospital_admission_id": haid if pd.notna(haid) else pd.NA,
                    "hour": 0.0, "dataset": DATASET, "measurement_type": "demographic",
                    "variable": "height", "value": float(omr_row["height_cm"]), "unit": "cm",
                })
                n_omr += 1
        log.info("OMR height: added %d rows from %d patients", n_omr, len(omr_ht))
    else:
        log.warning("OMR file not found at %s — skipping OMR height", omr_path)

    if not rows:
        return pd.DataFrame()

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
) -> pd.DataFrame:
    """Run the full MIMIC-IV Step-1 extraction pipeline.

    Outputs a single Parquet file at:
      output_path/mimic_iv_long.parquet

    Parameters
    ----------
    data_path : Path to MIMIC-IV root (contains hosp/ and icu/ subdirs)
    output_path : Directory for output Parquet files
    features_yaml : Path to configs/features.yaml
    sample_size : If set, subsample to N patients (for fast iteration)
    min_stay_hours : Minimum ICU LOS in hours for cohort inclusion
    random_state : Seed for reproducible sampling

    Returns
    -------
    The full Long-Format DataFrame (also written to Parquet).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    catalog = FeatureCatalog.from_yaml(features_yaml)
    log.info("Feature catalog loaded: %d features", len(list(catalog.keys())))

    cohort = extract_cohort(data_path, min_stay_hours=min_stay_hours,
                            sample_size=sample_size, random_state=random_state)

    log.info("Extracting chartevents...")
    chart = extract_chartevents(data_path, cohort, catalog)

    # Earliest weight observation per stay → static admission weight
    if not chart.empty and "weight" in chart["variable"].values:
        wt_adm = (
            chart.loc[chart["variable"] == "weight", ["stay_id", "hour", "value"]]
            .sort_values("hour")
            .groupby("stay_id")["value"].first()
            .reset_index()
            .rename(columns={"value": "weight_admission_kg"})
        )
        cohort = cohort.merge(wt_adm, on="stay_id", how="left")

    log.info("Extracting labevents...")
    lab = extract_labevents(data_path, cohort, catalog)

    log.info("Extracting inputevents...")
    inputs = extract_inputevents(data_path, cohort, catalog)

    log.info("Extracting outputevents...")
    outputs = extract_outputevents(data_path, cohort, catalog)

    log.info("Extracting procedureevents...")
    procs = extract_procedureevents(data_path, cohort)

    log.info("Extracting microbiologyevents (blood cultures)...")
    micro = extract_microbiologyevents(data_path, cohort)

    log.info("Extracting antibiotic start events...")
    abx = extract_antibiotic_events(data_path, cohort)

    log.info("Extracting diagnoses (ICD-9/10, Charlson comorbidity)...")
    diag = extract_diagnoses(data_path, cohort)

    log.info("Extracting static features...")
    static = extract_static(data_path, cohort)

    parts = [df for df in [chart, lab, inputs, outputs, procs, micro, abx, diag, static] if not df.empty]
    long_df = pd.concat(parts, ignore_index=True)

    # Clip observations to the 7-day (168h) modelling horizon, consistent with
    # eICU/HiRID and the Stage-3 168h window cap. Keeps static at hour=0.
    _n0 = len(long_df)
    long_df = long_df.loc[(long_df["hour"] >= 0) & (long_df["hour"] <= 168.0)].copy()
    log.info("168h horizon clip: %d -> %d rows (-%d)", _n0, len(long_df), _n0 - len(long_df))

    out_file = output_path / "mimic_iv_long.parquet"
    long_df.to_parquet(out_file, index=False)
    log.info("Saved %d rows to %s", len(long_df), out_file)

    # Save cohort separately (needed for Step 2 and train/test split)
    cohort_file = output_path / "mimic_iv_cohort.parquet"
    cohort.to_parquet(cohort_file, index=False)
    log.info("Saved cohort (%d stays) to %s", len(cohort), cohort_file)

    return long_df
