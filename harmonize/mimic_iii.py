"""
harmonize/mimic_iii.py
Step-1 extraction pipeline for MIMIC-III v1.4.

Schema output: Long-Format Parquet
  patient_id | stay_id | hospital_admission_id | hour | variable | value | unit | dataset | measurement_type

Key differences from MIMIC-IV:
  - Uncompressed CSV files (no .gz), uppercase column names
  - CHARTEVENTS.csv is 33 GB -> chunked reading mandatory
  - Both CareVue and Metavision item IDs for chart features
  - Only INPUTEVENTS_MV.csv (CareVue rates not standardized)
  - Age computed from DOB; MIMIC-III shifts DOB for patients >89 -> cap at 91.4
  - LABEVENTS lacks HADM_ID for many rows -> join via SUBJECT_ID + time window
  - mpv not available; CRP (50889) available; pct not available
"""

from __future__ import annotations

import logging
from pathlib import Path

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

DATASET = "mimic_iii"
CHUNKSIZE = 100_000

# ---------------------------------------------------------------------------
# Admission-type → SAPS-II points (same logic as mimic_iv, MIMIC-III careunit names)
# ---------------------------------------------------------------------------
_SURGICAL_UNITS_III = frozenset({"CSRU", "TSICU", "SICU"})

def _admission_type_saps_iii(admission_type: str, first_careunit: str) -> float:
    """Map MIMIC-III (admission_type, first_careunit) → SAPS-II admission type points."""
    adm = str(admission_type).upper().strip()
    surgical = str(first_careunit) in _SURGICAL_UNITS_III
    elective = adm == "ELECTIVE"
    if surgical and elective:
        return 0.0   # scheduled surgical
    if surgical and not elective:
        return 8.0   # unscheduled surgical
    return 6.0       # medical

# ---------------------------------------------------------------------------
# Chart item-ID registries (CareVue + Metavision)
# ---------------------------------------------------------------------------

CHART_NUMERIC: dict[str, list[int]] = {
    "hr":              [211, 220045],
    "rr":              [618, 220210, 224690],
    "spo2":            [646, 220277],
    "sbp_invasive":    [51, 220050],
    "dbp_invasive":    [8368, 220051],
    "mbp_invasive":    [52, 220052],
    "sbp_noninvasive": [455, 220179],
    "dbp_noninvasive": [8441, 220180],
    "mbp_noninvasive": [456, 220181],
    "gcs_eye":         [184, 220739],
    "gcs_motor":       [454, 223901],
    "gcs_verbal":      [723, 223900],
    "peep":            [505, 506, 220339, 224700],
    "amv":             [448, 450, 224687],
    "tidal_volume":    [681, 682, 683, 224685, 224684, 224686],
    "etco2":           [1817, 228640],
    "weight":          [762, 763, 3580, 224639, 226512],  # kg items
    "height":          [226730],                           # cm (Metavision)
}
 
CHART_WEIGHT_LBS: list[int] = [3581]          # Present Weight (lbs) -> lbs_to_kg
CHART_HEIGHT_IN: list[int] = [1394, 226707]   # Height (inches) -> *2.54 -> cm
CHART_TEMP_C: list[int] = [676, 677, 223762]  # Temperature Celsius
CHART_TEMP_F: list[int] = [678, 679, 223761]  # Temperature Fahrenheit -> convert
CHART_FIO2: list[int] = [189, 190, 727, 2981, 3420, 3422, 223835]
# Removed 227009 (FiO2_ApacheIV_old) and 227010 (FiO2_ApacheIV): these are
# APACHE-IV score *inputs*, not raw FiO2 measurements (risk of double-counting).
# Added 727 ("Vision FiO2") and 2981 ("FiO2"), both CareVue chartevents,
# present in the original Paper-2 pipeline and clinically equivalent.
CHART_CRR: list[int] = [115]                  # Capillary Refill [Right], CareVue only

# ---------------------------------------------------------------------------
# Lab item-ID registry
# ---------------------------------------------------------------------------

LAB_NUMERIC: dict[str, list[int]] = {
    "base_excess":        [50802],
    "hco3":               [50882, 50803],       # Bicarbonate / Calc. Bicarbonate WB
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
    # mpv not available in MIMIC-III
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
    "chloride":           [50902, 50806],
    "creatinine":         [50912],
    "creatinine_urine":   [51082],
    "crp":                [50889],              # available in MIMIC-III
    "fibrinogen":         [51214],
    "glucose":            [50809, 50931],
    "inr":                [51237],
    "ldh":                [50954],
    "lipase":             [50956],
    "magnesium":          [50960],
    # pct not available in MIMIC-III
    "phosphate":          [50970],
    "potassium":          [50971, 50822],
    "protein_total":      [50976],
    "pt":                 [51274],
    "ptt":                [51275],
    "sodium":             [50983, 50824],
    "troponin_i":         [51002],
}

# ---------------------------------------------------------------------------
# Inputevents (Metavision only — same item IDs as MIMIC-IV)
# ---------------------------------------------------------------------------

INPUT_VASOPRESSORS: dict[str, list[int]] = {
    "norepinephrine": [221906],
    "epinephrine":    [221289],
    "dopamine":       [221662],
    "dobutamine":     [221653],
}

INPUT_VASOPRESSIN: dict[str, list[int]] = {
    "vasopressin": [222315],
}

INPUT_OTHER: dict[str, list[int]] = {
    "amiodarone":        [221347, 228339],
    "dexmedetomidine":   [225150],
    "fentanyl":          [221744],
    "furosemide":        [221794],
    "heparin":           [225152],
    "insulin":           [223257, 223258],
    "midazolam":         [221668],
    "milrinone":         [221986],
    "morphine":          [225154],
    "piperacillin":      [225892, 225893],   # Piperacillin / Piperacillin-Tazobactam
    "propofol":          [222168],
    "vancomycin":        [225798],
}
# pantoprazole: only in INPUTEVENTS_CV (itemids 40550, 40700, 41583), affecting
# 44 of 45 253 stays (<0.1%). Loader processes INPUTEVENTS_MV only, so this
# feature stays unavailable for MIMIC-III. Not worth a CV loader for 0.1%.

# ---------------------------------------------------------------------------
# Outputevents (urine output)
# ---------------------------------------------------------------------------

URINE_ITEMIDS: list[int] = [
    # Metavision
    226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567,
    226557, 226558, 227488, 227489,
    # CareVue
    40055, 40069, 40094, 40715, 40056, 40057, 40085, 40405, 40428,
    43175, 44706, 46532, 46578, 46658, 44080, 44103, 44132, 44237,
]

# ---------------------------------------------------------------------------
# Procedureevents: Mechanical Ventilation (Metavision only)
# ---------------------------------------------------------------------------

MV_ITEMIDS: list[int] = [225792, 225794]  # Invasive + Non-Invasive Ventilation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intime_lookup(cohort: pd.DataFrame) -> pd.Series:
    return cohort.set_index("stay_id")["intime"]


def _stay_meta(cohort: pd.DataFrame) -> pd.DataFrame:
    return cohort.set_index("stay_id")[["patient_id", "hospital_admission_id"]]


def _attach_meta(df: pd.DataFrame, stay_meta: pd.DataFrame) -> pd.DataFrame:
    df = df.join(stay_meta, on="stay_id")
    return df


def _normalize_vasopressor_rate(
    df: pd.DataFrame,
    feature_key: str,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Normalize MV vasopressor rate to ug/kg/min."""
    df = df.copy()
    df["value"] = np.nan
    rate = df["RATE"].copy().astype(float)
    uom = df["RATEUOM"].astype(str).str.lower()
    wt = df["PATIENTWEIGHT"].astype(float)

    already = uom.str.contains("mcg/kg/min", na=False)
    per_min = uom.str.contains(r"mcg/min$", na=False, regex=True)
    mg_hr = uom.str.contains(r"mg/hour", na=False)

    df.loc[already, "value"] = rate.loc[already]
    median_wt = wt.median() if wt.notna().any() else 80.0
    wt_safe = wt.where(wt.notna() & (wt > 0), median_wt)
    df.loc[per_min, "value"] = rate.loc[per_min] / wt_safe.loc[per_min]
    df.loc[mg_hr, "value"] = rate.loc[mg_hr] * 1000 / 60 / wt_safe.loc[mg_hr]

    vr = catalog.value_range(feature_key)
    return apply_outlier_filter(df, "value", vr)


def _normalize_vasopressin_rate(
    df: pd.DataFrame, catalog: FeatureCatalog
) -> pd.DataFrame:
    """Convert vasopressin to IU/min."""
    df = df.copy()
    df["value"] = np.nan
    rate = df["RATE"].copy().astype(float)
    uom = df["RATEUOM"].astype(str).str.lower()
    u_min = uom.str.contains(r"unit/min|u/min|iu/min", na=False, regex=True)
    u_hr = uom.str.contains(r"unit/hour|u/hour|iu/hour", na=False, regex=True)
    df.loc[u_min, "value"] = rate.loc[u_min]
    df.loc[u_hr, "value"] = rate.loc[u_hr] / 60.0
    vr = catalog.value_range("vasopressin")
    return apply_outlier_filter(df, "value", vr)


def _normalize_other_rate(
    df: pd.DataFrame, feature_key: str, catalog: FeatureCatalog
) -> pd.DataFrame:
    df = df.copy()
    df["value"] = df["RATE"].copy().astype(float)
    vr = catalog.value_range(feature_key)
    return apply_outlier_filter(df, "value", vr)


# ---------------------------------------------------------------------------
# Cohort extraction
# ---------------------------------------------------------------------------

def extract_cohort(
    data_path: str | Path,
    min_stay_hours: float = 24.0,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load adult ICU cohort from MIMIC-III ICUSTAYS + PATIENTS + ADMISSIONS.

    Age is computed from DOB. MIMIC-III shifts DOB for patients >89 years
    (they appear ~300 years old); ages are capped at 91.4.
    """
    data_path = Path(data_path)
    log.info("Loading ICUSTAYS...")
    icu = pd.read_parquet(
        data_path / "ICUSTAYS.parquet",
        columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "FIRST_CAREUNIT", "INTIME", "OUTTIME", "LOS"],
    )
    icu["SUBJECT_ID"]  = pd.to_numeric(icu["SUBJECT_ID"],  errors="coerce")
    icu["HADM_ID"]     = pd.to_numeric(icu["HADM_ID"],     errors="coerce")
    icu["ICUSTAY_ID"]  = pd.to_numeric(icu["ICUSTAY_ID"],  errors="coerce")
    icu["LOS"]         = pd.to_numeric(icu["LOS"],         errors="coerce")
    icu = icu.dropna(subset=["INTIME", "OUTTIME"])
    icu["INTIME"] = pd.to_datetime(icu["INTIME"])
    icu["OUTTIME"] = pd.to_datetime(icu["OUTTIME"])
    icu["los_hours"] = icu["LOS"] * 24.0
    icu = icu.loc[icu["los_hours"] >= min_stay_hours].copy()

    log.info("Loading PATIENTS...")
    pat = pd.read_parquet(
        data_path / "PATIENTS.parquet",
        columns=["SUBJECT_ID", "GENDER", "DOB", "DOD", "EXPIRE_FLAG"],
    )
    pat["SUBJECT_ID"]   = pd.to_numeric(pat["SUBJECT_ID"],   errors="coerce")
    pat["EXPIRE_FLAG"]  = pd.to_numeric(pat["EXPIRE_FLAG"],  errors="coerce")
    pat["DOB"] = pd.to_datetime(pat["DOB"])

    log.info("Loading ADMISSIONS...")
    adm = pd.read_parquet(
        data_path / "ADMISSIONS.parquet",
        columns=["SUBJECT_ID", "HADM_ID", "ADMISSION_TYPE", "HOSPITAL_EXPIRE_FLAG", "DEATHTIME"],
    )
    adm["SUBJECT_ID"]          = pd.to_numeric(adm["SUBJECT_ID"],          errors="coerce")
    adm["HADM_ID"]             = pd.to_numeric(adm["HADM_ID"],             errors="coerce")
    adm["HOSPITAL_EXPIRE_FLAG"]= pd.to_numeric(adm["HOSPITAL_EXPIRE_FLAG"],errors="coerce")
    adm["DEATHTIME"]           = pd.to_datetime(adm["DEATHTIME"],          errors="coerce")

    cohort = icu.merge(pat, on="SUBJECT_ID", how="left")
    cohort = cohort.merge(adm[["HADM_ID", "ADMISSION_TYPE", "HOSPITAL_EXPIRE_FLAG", "DEATHTIME"]],
                          on="HADM_ID", how="left")

    # Age at admission (DOB shifted for >89yo -> cap at 91.4)
    cohort["age"] = (cohort["INTIME"] - cohort["DOB"]).dt.days / 365.25
    cohort["age"] = cohort["age"].clip(upper=91.4)
    cohort = cohort.loc[cohort["age"] >= 18].copy()

    cohort = cohort.rename(columns={
        "SUBJECT_ID":         "patient_id",
        "ICUSTAY_ID":         "stay_id",
        "HADM_ID":            "hospital_admission_id",
        "INTIME":             "intime",
        "OUTTIME":            "outtime",
        "GENDER":             "gender",
        "FIRST_CAREUNIT":     "first_careunit",
        "ADMISSION_TYPE":     "admission_type",
        "HOSPITAL_EXPIRE_FLAG": "hospital_mortality",
        "DEATHTIME":          "death_time",
    })

    log.info(
        "Cohort: %d stays, %d patients (min_stay_hours=%.0f)",
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
# Chartevents extraction
# ---------------------------------------------------------------------------

def extract_chartevents(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract chartevents in chunks (CHARTEVENTS.csv is ~33 GB uncompressed)."""
    data_path = Path(data_path)
    stay_ids = set(cohort["stay_id"])
    intime = _intime_lookup(cohort)
    stay_meta = _stay_meta(cohort)

    all_chart_ids: set[int] = set()
    for ids in CHART_NUMERIC.values():
        all_chart_ids.update(ids)
    all_chart_ids.update(CHART_WEIGHT_LBS)
    all_chart_ids.update(CHART_HEIGHT_IN)
    all_chart_ids.update(CHART_TEMP_C)
    all_chart_ids.update(CHART_TEMP_F)
    all_chart_ids.update(CHART_FIO2)
    all_chart_ids.update(CHART_CRR)

    itemid_to_feat: dict[int, str] = {}
    for feat, ids in CHART_NUMERIC.items():
        for iid in ids:
            itemid_to_feat.setdefault(iid, feat)
    for iid in CHART_WEIGHT_LBS:
        itemid_to_feat.setdefault(iid, "weight")
    for iid in CHART_HEIGHT_IN:
        itemid_to_feat.setdefault(iid, "height")
    for iid in CHART_TEMP_C + CHART_TEMP_F:
        itemid_to_feat.setdefault(iid, "temp_core")
    for iid in CHART_FIO2:
        itemid_to_feat.setdefault(iid, "fio2")
    for iid in CHART_CRR:
        itemid_to_feat.setdefault(iid, "crr")

    temp_f_set = set(CHART_TEMP_F)
    weight_lbs_set = set(CHART_WEIGHT_LBS)
    height_in_set = set(CHART_HEIGHT_IN)
    fio2_set = set(CHART_FIO2)
    crr_set = set(CHART_CRR)

    parts: list[pd.DataFrame] = []
    _filter_removed = 0
    _filter_total   = 0

    # The parquet was created by csv_to_parquet.py with all columns as strings.
    # Use string-valued filter for pushdown, then cast key columns to numeric.
    str_chart_ids = [str(i) for i in all_chart_ids]
    log.info("Reading CHARTEVENTS (parquet, itemid-filter pushdown)...")
    for chunk in [pd.read_parquet(
        data_path / "CHARTEVENTS.parquet",
        columns=["ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUENUM", "VALUE", "ERROR"],
        filters=[("ITEMID", "in", str_chart_ids)],
    )]:
        chunk = chunk.rename(columns={"ICUSTAY_ID": "stay_id"})
        # Cast string columns to numeric types for all downstream logic
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce")
        chunk["ITEMID"]  = pd.to_numeric(chunk["ITEMID"],  errors="coerce").astype("Int64")
        chunk["ERROR"]   = pd.to_numeric(chunk["ERROR"],   errors="coerce")
        chunk = chunk.loc[
            chunk["stay_id"].isin(stay_ids) &
            chunk["ITEMID"].isin(all_chart_ids)
        ]
        chunk = chunk.loc[chunk["ERROR"].isna() | (chunk["ERROR"] == 0)]
        chunk = chunk.dropna(subset=["CHARTTIME"])

        # Capillary refill: text -> binary
        crr_mask = chunk["ITEMID"].isin(crr_set)
        if crr_mask.any():
            crr_rows = chunk.loc[crr_mask].copy()
            crr_rows["VALUENUM"] = crr_rows["VALUE"].astype(str).str.lower().map(
                lambda v: 1.0 if "normal" in v or "< 2" in v or "<2" in v else (
                    0.0 if "delayed" in v or "> 2" in v or ">2" in v else np.nan
                ),
                na_action="ignore",
            )
            chunk = pd.concat([chunk.loc[~crr_mask], crr_rows], ignore_index=True)

        chunk = chunk.dropna(subset=["VALUENUM"])
        chunk["VALUENUM"] = pd.to_numeric(chunk["VALUENUM"], errors="coerce").astype("float64")

        # Unit conversions
        f_mask = chunk["ITEMID"].isin(temp_f_set)
        if f_mask.any():
            chunk.loc[f_mask, "VALUENUM"] = fahrenheit_to_celsius(chunk.loc[f_mask, "VALUENUM"])

        lbs_mask = chunk["ITEMID"].isin(weight_lbs_set)
        if lbs_mask.any():
            chunk.loc[lbs_mask, "VALUENUM"] = lbs_to_kg(chunk.loc[lbs_mask, "VALUENUM"])

        ht_in_mask = chunk["ITEMID"].isin(height_in_set)
        if ht_in_mask.any():
            chunk.loc[ht_in_mask, "VALUENUM"] = chunk.loc[ht_in_mask, "VALUENUM"] * 2.54

        fio2_mask = chunk["ITEMID"].isin(fio2_set)
        if fio2_mask.any():
            chunk.loc[fio2_mask, "VALUENUM"] = fio2_percent_to_fraction(
                chunk.loc[fio2_mask, "VALUENUM"]
            )

        chunk["variable"] = chunk["ITEMID"].map(itemid_to_feat)
        chunk["CHARTTIME"] = pd.to_datetime(chunk["CHARTTIME"])
        chunk["hour"] = hours_since_intime(
            chunk["CHARTTIME"], chunk["stay_id"].map(intime)
        )
        chunk = chunk.loc[chunk["hour"] >= 0].copy()

        # Outlier filter per variable — track how many values are removed
        _n_pre = len(chunk)
        _filter_total += _n_pre
        filtered = []
        for feat, grp in chunk.groupby("variable"):
            vr = catalog.value_range(feat) if feat in catalog else None
            grp = apply_outlier_filter(grp, "VALUENUM", vr)
            filtered.append(grp)
        if not filtered:
            continue
        chunk = pd.concat(filtered, ignore_index=True)
        chunk = chunk.dropna(subset=["VALUENUM"])
        _filter_removed += _n_pre - len(chunk)
        if chunk.empty:
            continue

        chunk = chunk.rename(columns={"VALUENUM": "value"})

        # Deduplicate temp_core: CareVue enters same measurement in both °C and °F
        # items simultaneously — prefer °C items, keep one row per (stay_id, CHARTTIME)
        temp_mask = chunk["variable"] == "temp_core"
        if temp_mask.any():
            temp_rows = chunk.loc[temp_mask].copy()
            temp_rows["_prio"] = temp_rows["ITEMID"].isin(temp_f_set).astype(int)
            temp_rows = (
                temp_rows.sort_values(["stay_id", "CHARTTIME", "_prio"])
                .drop_duplicates(subset=["stay_id", "CHARTTIME"])
                .drop(columns="_prio")
            )
            chunk = pd.concat([chunk.loc[~temp_mask], temp_rows], ignore_index=True)

        chunk = chunk.join(stay_meta, on="stay_id")
        chunk["unit"] = chunk["variable"].apply(
            lambda v: catalog.unit(v) if v in catalog else ""
        )
        chunk["measurement_type"] = chunk["variable"].apply(
            lambda v: catalog.measurement_type(v) if v in catalog else "vital"
        )
        chunk["dataset"] = DATASET
        parts.append(
            chunk[["patient_id", "stay_id", "hospital_admission_id",
                   "hour", "variable", "value", "unit", "dataset",
                   "measurement_type"]].copy()
        )

    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)

    # Cross-chunk safety net for temp_core dedup. The per-chunk dedup above
    # collapses simultaneous °C+°F entries within a CHARTEVENTS chunk; this
    # final pass handles the rare case where the two entries straddle a chunk
    # boundary (same stay_id + same hour-since-intime after F→C conversion).
    temp_mask = result["variable"] == "temp_core"
    if temp_mask.any():
        _temp_before = int(temp_mask.sum())
        deduped = (
            result.loc[temp_mask]
            .drop_duplicates(subset=["stay_id", "hour"], keep="first")
        )
        result = pd.concat([result.loc[~temp_mask], deduped], ignore_index=True)
        _temp_after = len(deduped)
        if _temp_before != _temp_after:
            log.info(
                "temp_core dedup: %d -> %d rows (%d cross-chunk duplicates removed)",
                _temp_before, _temp_after, _temp_before - _temp_after,
            )

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
    """Extract lab results.

    MIMIC-III LABEVENTS frequently lacks HADM_ID. Strategy: filter by
    SUBJECT_ID then match CHARTTIME within [intime, outtime] of each stay.
    """
    data_path = Path(data_path)
    subject_ids = set(cohort["patient_id"])
    stay_meta = _stay_meta(cohort)

    all_itemids: set[int] = set()
    for ids in LAB_NUMERIC.values():
        all_itemids.update(ids)

    itemid_to_feat: dict[int, str] = {}
    for feat, ids in LAB_NUMERIC.items():
        for iid in ids:
            itemid_to_feat.setdefault(iid, feat)

    str_itemids = [str(i) for i in all_itemids]
    log.info("Reading LABEVENTS (parquet, itemid-filter pushdown)...")
    la_parts: list[pd.DataFrame] = []
    for chunk in [pd.read_parquet(
        data_path / "LABEVENTS.parquet",
        columns=["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
        filters=[("ITEMID", "in", str_itemids)],
    )]:
        chunk["SUBJECT_ID"] = pd.to_numeric(chunk["SUBJECT_ID"], errors="coerce")
        chunk["ITEMID"]     = pd.to_numeric(chunk["ITEMID"],     errors="coerce").astype("Int64")
        chunk = chunk.loc[chunk["SUBJECT_ID"].isin(subject_ids) & chunk["ITEMID"].isin(all_itemids)]
        chunk = chunk.dropna(subset=["CHARTTIME", "VALUENUM"])
        if not chunk.empty:
            la_parts.append(chunk)
    if not la_parts:
        return pd.DataFrame()
    la = pd.concat(la_parts, ignore_index=True)
    la["CHARTTIME"] = pd.to_datetime(la["CHARTTIME"])
    la["VALUENUM"] = pd.to_numeric(la["VALUENUM"], errors="coerce").astype("float64")
    log.info("  labevents after subject filter: %d rows", len(la))

    # Match to stays via SUBJECT_ID + time window
    intervals = cohort[["patient_id", "stay_id", "hospital_admission_id",
                         "intime", "outtime"]].copy()
    la = la.merge(
        intervals.rename(columns={"patient_id": "SUBJECT_ID"}),
        on="SUBJECT_ID", how="inner"
    )
    in_window = (la["CHARTTIME"] >= la["intime"]) & (la["CHARTTIME"] <= la["outtime"])
    la = la.loc[in_window].copy()
    la["hour"] = (la["CHARTTIME"] - la["intime"]).dt.total_seconds() / 3600.0
    la["variable"] = la["ITEMID"].map(itemid_to_feat)

    _n_pre = len(la)
    filtered = []
    for feat, grp in la.groupby("variable"):
        vr = catalog.value_range(feat) if feat in catalog else None
        grp = apply_outlier_filter(grp, "VALUENUM", vr)
        filtered.append(grp)
    if not filtered:
        return pd.DataFrame()
    la = pd.concat(filtered, ignore_index=True).dropna(subset=["VALUENUM"])

    la = la.rename(columns={"VALUENUM": "value", "SUBJECT_ID": "patient_id"})
    la["unit"] = la["variable"].apply(lambda v: catalog.unit(v) if v in catalog else "")
    la["measurement_type"] = "lab"
    la["dataset"] = DATASET

    result = la[["patient_id", "stay_id", "hospital_admission_id",
                 "hour", "variable", "value", "unit", "dataset",
                 "measurement_type"]].copy()
    # apply_outlier_filter NULLS values and returns the same number of rows, and `result` is
    # only a column subset, so `_n_pre - len(result)` was identically zero. Count the NaN the
    # filter introduced (valuenum was dropna'd before it ran).
    _filter_removed = int(result["value"].isna().sum())
    log.info(
        "Labevents: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed, _n_pre, 100.0 * _filter_removed / _n_pre if _n_pre else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Inputevents extraction (Metavision only)
# ---------------------------------------------------------------------------

def extract_inputevents(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract continuous infusions from INPUTEVENTS_MV (same item IDs as MIMIC-IV)."""
    data_path = Path(data_path)
    stay_ids = set(cohort["stay_id"])
    intime = _intime_lookup(cohort)
    stay_meta = _stay_meta(cohort)

    all_itemids: set[int] = set()
    for iids in [*INPUT_VASOPRESSORS.values(),
                 *INPUT_VASOPRESSIN.values(),
                 *INPUT_OTHER.values()]:
        all_itemids.update(iids)

    itemid_to_info: dict[int, tuple[str, str]] = {}
    for feat, ids in INPUT_VASOPRESSORS.items():
        for iid in ids:
            itemid_to_info[iid] = (feat, "vasopressor")
    for feat, ids in INPUT_VASOPRESSIN.items():
        for iid in ids:
            itemid_to_info[iid] = (feat, "vasopressin")
    for feat, ids in INPUT_OTHER.items():
        for iid in ids:
            if iid not in itemid_to_info:
                itemid_to_info[iid] = (feat, "other")

    log.info("Reading INPUTEVENTS_MV...")
    ie = pd.read_parquet(
        data_path / "INPUTEVENTS_MV.parquet",
        columns=["ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID",
                 "RATE", "RATEUOM", "PATIENTWEIGHT", "STATUSDESCRIPTION"],
    )
    ie["ICUSTAY_ID"]     = pd.to_numeric(ie["ICUSTAY_ID"],     errors="coerce")
    ie["ITEMID"]         = pd.to_numeric(ie["ITEMID"],         errors="coerce").astype("Int64")
    ie["RATE"]           = pd.to_numeric(ie["RATE"],           errors="coerce")
    ie["PATIENTWEIGHT"]  = pd.to_numeric(ie["PATIENTWEIGHT"],  errors="coerce")
    ie = ie.rename(columns={"ICUSTAY_ID": "stay_id"})
    ie = ie.loc[
        ie["stay_id"].isin(stay_ids) &
        ie["ITEMID"].isin(all_itemids) &
        (ie["STATUSDESCRIPTION"] != "Rewritten")
    ]
    ie = ie.dropna(subset=["STARTTIME", "ENDTIME", "RATE"])
    ie["STARTTIME"] = pd.to_datetime(ie["STARTTIME"])
    ie["ENDTIME"] = pd.to_datetime(ie["ENDTIME"])
    ie["feature_key"] = ie["ITEMID"].map({iid: feat for iid, (feat, _) in itemid_to_info.items()})
    ie["category"] = ie["ITEMID"].map({iid: cat for iid, (_, cat) in itemid_to_info.items()})
    log.info("  inputevents_mv after filter: %d rows", len(ie))

    parts: list[pd.DataFrame] = []
    _filter_removed_ie = 0
    _filter_total_ie   = 0
    for feat_key, grp in ie.groupby("feature_key"):
        category = grp["category"].iloc[0]
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

        expanded = expand_intervals_hourly_vectorized(
            grp,
            start_col="STARTTIME",
            end_col="ENDTIME",
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
        log.warning("No inputevents data found.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result = _attach_meta(result, stay_meta)
    result["dataset"] = DATASET
    result = result[["patient_id", "stay_id", "hospital_admission_id",
                     "hour", "variable", "value", "unit", "dataset",
                     "measurement_type"]].copy()
    log.info(
        "Inputevents: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed_ie, _filter_total_ie,
        100.0 * _filter_removed_ie / _filter_total_ie if _filter_total_ie else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Outputevents (urine output)
# ---------------------------------------------------------------------------

def extract_outputevents(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    stay_ids = set(cohort["stay_id"])
    intime = _intime_lookup(cohort)
    stay_meta = _stay_meta(cohort)

    log.info("Reading OUTPUTEVENTS...")
    out = pd.read_parquet(
        Path(data_path) / "OUTPUTEVENTS.parquet",
        columns=["ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE", "ISERROR"],
    )
    out["ICUSTAY_ID"] = pd.to_numeric(out["ICUSTAY_ID"], errors="coerce")
    out["ITEMID"]     = pd.to_numeric(out["ITEMID"],     errors="coerce").astype("Int64")
    out["ISERROR"]    = pd.to_numeric(out["ISERROR"],    errors="coerce")
    out["VALUE"]      = pd.to_numeric(out["VALUE"],      errors="coerce")
    out = out.rename(columns={"ICUSTAY_ID": "stay_id"})
    out = out.loc[
        out["stay_id"].isin(stay_ids) &
        out["ITEMID"].isin(URINE_ITEMIDS) &
        (out["ISERROR"].isna() | (out["ISERROR"] == 0))
    ]
    out = out.dropna(subset=["CHARTTIME", "VALUE"])
    out["VALUE"] = pd.to_numeric(out["VALUE"], errors="coerce").astype("float64")
    out = out.loc[out["VALUE"] > 0]
    out["CHARTTIME"] = pd.to_datetime(out["CHARTTIME"])
    out["hour"] = hours_since_intime(out["CHARTTIME"], out["stay_id"].map(intime))
    out = out.loc[out["hour"] >= 0]

    _n_pre = len(out)
    vr = catalog.value_range("urine_output")
    out = apply_outlier_filter(out, "VALUE", vr)
    out = out.dropna(subset=["VALUE"])

    out = out.rename(columns={"VALUE": "value"})
    out = out.join(stay_meta, on="stay_id")
    out["variable"] = "urine_output"
    out["unit"] = catalog.unit("urine_output")
    out["measurement_type"] = "fluid"
    out["dataset"] = DATASET

    result = out[["patient_id", "stay_id", "hospital_admission_id",
                  "hour", "variable", "value", "unit", "dataset",
                  "measurement_type"]].copy()
    _filter_removed = _n_pre - len(result)
    log.info(
        "Outputevents (urine): %d rows | value_range filter removed %d / %d (%.2f%%)",
        len(result), _filter_removed, _n_pre, 100.0 * _filter_removed / _n_pre if _n_pre else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Procedureevents: Mechanical Ventilation
# ---------------------------------------------------------------------------

def extract_procedureevents(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    stay_ids = set(cohort["stay_id"])
    intime = _intime_lookup(cohort)
    stay_meta = _stay_meta(cohort)

    log.info("Reading PROCEDUREEVENTS_MV...")
    proc = pd.read_parquet(
        Path(data_path) / "PROCEDUREEVENTS_MV.parquet",
        columns=["ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "STATUSDESCRIPTION"],
    )
    proc["ICUSTAY_ID"] = pd.to_numeric(proc["ICUSTAY_ID"], errors="coerce")
    proc["ITEMID"]     = pd.to_numeric(proc["ITEMID"],     errors="coerce").astype("Int64")
    proc = proc.rename(columns={"ICUSTAY_ID": "stay_id"})
    proc = proc.loc[
        proc["stay_id"].isin(stay_ids) &
        proc["ITEMID"].isin(MV_ITEMIDS) &
        (proc["STATUSDESCRIPTION"] != "Paused")
    ]
    proc = proc.dropna(subset=["STARTTIME", "ENDTIME"])
    proc["STARTTIME"] = pd.to_datetime(proc["STARTTIME"])
    proc["ENDTIME"] = pd.to_datetime(proc["ENDTIME"])
    proc["value"] = 1.0

    expanded = expand_intervals_hourly_vectorized(
        proc,
        start_col="STARTTIME",
        end_col="ENDTIME",
        intime_lookup=intime,
        stay_id_col="stay_id",
        value_col="value",
    )
    if expanded.empty:
        return pd.DataFrame()

    expanded = expanded.join(stay_meta, on="stay_id")
    expanded["variable"] = "mechanical_ventilation"
    expanded["unit"] = "bool"
    expanded["measurement_type"] = "ventilation"
    expanded["dataset"] = DATASET
    result = expanded[["patient_id", "stay_id", "hospital_admission_id",
                        "hour", "variable", "value", "unit", "dataset",
                        "measurement_type"]].copy()
    log.info("Procedureevents (MV): %d rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Diagnoses — ICD-9 codes → Charlson comorbidity + SAPS-II chronic_disease
# ---------------------------------------------------------------------------

def extract_diagnoses(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract Charlson comorbidity index from DIAGNOSES_ICD (ICD-9).

    Joins via HADM_ID → computes Charlson score (Quan 2005) and SAPS-II
    chronic_disease points per stay. Emits as static features at hour=0.

    Variables: charlson_comorbidity_index, chronic_disease_sapsii.
    """
    from harmonize.diagnoses import charlson_icd9, chronic_disease_sapsii as _chron

    data_path = Path(data_path)
    pq = data_path / "DIAGNOSES_ICD.parquet"
    csv = data_path / "DIAGNOSES_ICD.csv"
    src = pq if pq.exists() else csv
    log.info("Loading DIAGNOSES_ICD from %s", src.name)
    df = (pd.read_parquet(src, columns=["HADM_ID", "ICD9_CODE"])
          if src.suffix == ".parquet"
          else pd.read_csv(src, usecols=["HADM_ID", "ICD9_CODE"],
                           dtype={"ICD9_CODE": str}))
    df["ICD9_CODE"] = df["ICD9_CODE"].fillna("").astype(str)

    hadm_in_cohort = set(cohort["hospital_admission_id"].dropna().astype(int))
    df = df[df["HADM_ID"].isin(hadm_in_cohort)].copy()
    codes_per_hadm = df.groupby("HADM_ID")["ICD9_CODE"].apply(list).to_dict()

    # Map HADM_ID → stay(s)
    hadm_to_stays = (
        cohort[["stay_id", "hospital_admission_id", "patient_id"]]
        .dropna(subset=["hospital_admission_id"])
        .assign(hospital_admission_id=lambda d: d["hospital_admission_id"].astype(int))
        .groupby("hospital_admission_id")
        .apply(lambda g: g[["stay_id", "patient_id"]].to_dict("records"),
               include_groups=False)
        .to_dict()
    )

    rows = []
    for hadm, codes in codes_per_hadm.items():
        for stay in hadm_to_stays.get(int(hadm), []):
            charlson = charlson_icd9(codes)
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
        log.warning("extract_diagnoses (MIMIC-III): no rows produced")
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    log.info("Diagnoses (MIMIC-III): %d rows from %d admissions",
             len(result), len(codes_per_hadm))
    return result


# ---------------------------------------------------------------------------
# Antibiotic start events — for Sepsis-3 suspected infection
# (MetaVision era only; CareVue antibiotic data excluded per pipeline design)
# ---------------------------------------------------------------------------

# MIMIC-III INPUTEVENTS_MV itemids for broad-spectrum antibiotics
_ABX_ITEMIDS_III: dict[int, str] = {
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
    """Extract antibiotic first-dose start events from INPUTEVENTS_MV.

    Emits value=1.0 at the hour of first administration start per (stay, drug).
    Variable names: abx_pip_tazo, abx_vancomycin, abx_meropenem, etc.

    Only MetaVision era (INPUTEVENTS_MV) — CareVue antibiotic data omitted
    because antibiotic completeness in CareVue is unreliable. When combining
    with blood cultures for Sepsis-3, restrict to stays admitted ≥ 2008.
    """
    data_path = Path(data_path)
    src = data_path / "INPUTEVENTS_MV.parquet"
    if not src.exists():
        src = data_path / "INPUTEVENTS_MV.csv"
    log.info("Loading antibiotic events from %s", src.name)

    stay_ids = set(cohort["stay_id"].astype(int))
    intime   = _intime_lookup(cohort)
    meta     = _stay_meta(cohort)

    cols = ["ICUSTAY_ID", "STARTTIME", "ITEMID"]
    df = (pd.read_parquet(src, columns=cols)
          if src.suffix == ".parquet"
          else pd.read_csv(src, usecols=cols, dtype={"ITEMID": int}))

    df = df[df["ITEMID"].isin(_ABX_ITEMIDS_III)].copy()
    df = df[df["ICUSTAY_ID"].isin(stay_ids)].copy()
    df["STARTTIME"] = pd.to_datetime(df["STARTTIME"], errors="coerce")
    df = df.dropna(subset=["STARTTIME"])
    df["stay_id"] = df["ICUSTAY_ID"].astype(int)
    df["hour"] = hours_since_intime(df["STARTTIME"], df["stay_id"].map(intime))
    df["variable"] = df["ITEMID"].map(_ABX_ITEMIDS_III)
    df = df[(df["hour"] >= -24.0) & (df["hour"] <= 168.0)]

    # Keep first event per (stay, drug)
    df = df.sort_values("hour").drop_duplicates(subset=["stay_id", "variable"])

    df["patient_id"]             = df["stay_id"].map(lambda s: meta.get(s, {}).get("patient_id"))
    df["hospital_admission_id"]  = df["stay_id"].map(lambda s: meta.get(s, {}).get("hospital_admission_id"))
    df["value"]                  = 1.0
    df["unit"]                   = "bool"
    df["dataset"]                = DATASET
    df["measurement_type"]       = "medication"

    result = df[["patient_id", "stay_id", "hospital_admission_id",
                 "hour", "variable", "value", "unit", "dataset",
                 "measurement_type"]].dropna(subset=["patient_id"])
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = result["hospital_admission_id"].astype("Int64")
    log.info("Antibiotic events (MIMIC-III): %d events, %d stays, %d distinct drugs",
             len(result), result["stay_id"].nunique(), result["variable"].nunique())
    return result


# ---------------------------------------------------------------------------
# Microbiology events — blood cultures (for Sepsis-3 suspected infection)
# ---------------------------------------------------------------------------

def extract_microbiologyevents(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract blood culture order events from MICROBIOLOGYEVENTS.

    Reads MICROBIOLOGYEVENTS.parquet (or .csv fallback), filters to
    SPEC_TYPE_DESC == 'BLOOD CULTURE', and emits value=1.0 at the hour of
    culture collection relative to ICU intime. Variable: 'blood_culture'.

    CHARTTIME is used when available; rows with NaN CHARTTIME (~4.7%) fall
    back to CHARTDATE + 12:00 (midday on the collection date).

    Each blood culture event is assigned to the ICU stay whose [intime, outtime]
    window contains the event. Events outside all ICU windows for that hospital
    admission are assigned to the stay with the smallest |event_time - intime|
    (captures pre-admission cultures drawn in the ED).

    Note on MetaVision era: blood cultures are recorded system-independently
    (separate module from charting). They exist for both CareVue and MetaVision
    patients. However, for Sepsis-3 (antibiotic + culture within ±72h), restrict
    to MetaVision-era stays (intime ≥ 2008-01-01) where antibiotic documentation
    via INPUTEVENTS_MV is complete.
    """
    data_path = Path(data_path)
    pq_path = data_path / "MICROBIOLOGYEVENTS.parquet"
    csv_path = data_path / "MICROBIOLOGYEVENTS.csv"
    src = pq_path if pq_path.exists() else csv_path

    log.info("Reading MICROBIOLOGYEVENTS from %s", src.name)
    if src.suffix == ".parquet":
        df = pd.read_parquet(src, columns=[
            "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "SPEC_TYPE_DESC",
        ])
    else:
        df = pd.read_csv(src, usecols=[
            "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "SPEC_TYPE_DESC",
        ], dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64"})

    df = df[df["SPEC_TYPE_DESC"] == "BLOOD CULTURE"].copy()
    log.info("  blood cultures total: %d rows across %d admissions",
             len(df), df["HADM_ID"].nunique())

    # Resolve event timestamp: CHARTTIME preferred, CHARTDATE+12h fallback
    df["CHARTTIME"] = pd.to_datetime(df["CHARTTIME"], errors="coerce")
    df["CHARTDATE"] = pd.to_datetime(df["CHARTDATE"], errors="coerce")
    fallback_mask = df["CHARTTIME"].isna()
    df.loc[fallback_mask, "CHARTTIME"] = (
        df.loc[fallback_mask, "CHARTDATE"] + pd.Timedelta(hours=12)
    )
    df = df.dropna(subset=["CHARTTIME", "HADM_ID"])
    df["HADM_ID"] = df["HADM_ID"].astype(int)

    # Restrict to admissions in our cohort
    hadm_in_cohort = set(cohort["hospital_admission_id"].dropna().astype(int))
    df = df[df["HADM_ID"].isin(hadm_in_cohort)].copy()
    log.info("  after cohort filter: %d rows", len(df))

    # Build per-admission lookup: HADM_ID → list of (stay_id, intime, outtime)
    stay_info = (
        cohort[["stay_id", "hospital_admission_id", "intime", "outtime"]]
        .dropna(subset=["intime", "outtime"])
        .copy()
    )
    stay_info["hospital_admission_id"] = stay_info["hospital_admission_id"].astype(int)
    stay_info["intime"]  = pd.to_datetime(stay_info["intime"])
    stay_info["outtime"] = pd.to_datetime(stay_info["outtime"])
    hadm_to_stays = stay_info.groupby("hospital_admission_id").apply(
        lambda g: g[["stay_id", "intime", "outtime"]].to_dict("records"),
        include_groups=False,
    ).to_dict()

    stay_meta = _stay_meta(cohort)  # stay_id → {patient_id, hospital_admission_id}

    rows: list[dict] = []
    for _, ev in df.iterrows():
        hadm  = int(ev["HADM_ID"])
        t_ev  = ev["CHARTTIME"]
        stays = hadm_to_stays.get(hadm, [])
        if not stays:
            continue

        # Try to assign to the ICU stay whose window contains the event
        assigned = None
        for s in stays:
            if s["intime"] <= t_ev <= s["outtime"]:
                assigned = s
                break

        # If no containing window, assign to closest stay by intime (captures ED cultures)
        if assigned is None:
            assigned = min(stays, key=lambda s: abs((t_ev - s["intime"]).total_seconds()))

        sid   = int(assigned["stay_id"])
        hour  = (t_ev - assigned["intime"]).total_seconds() / 3600.0
        if hour < -24.0 or hour > 336.0:   # discard events >14d after admission
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
        log.warning("extract_microbiologyevents: no blood culture events matched cohort")
        return pd.DataFrame()

    result = pd.DataFrame(rows).dropna(subset=["patient_id"])
    # De-duplicate: one event per (stay_id, hour) — same culture drawn at same hour
    result = result.drop_duplicates(subset=["stay_id", "hour"])
    result["stay_id"] = result["stay_id"].astype("Int64")
    result["hospital_admission_id"] = result["hospital_admission_id"].astype("Int64")
    log.info("Blood cultures: %d events in %d stays (%.1f%% of cohort)",
             len(result), result["stay_id"].nunique(),
             100 * result["stay_id"].nunique() / len(cohort))
    return result


# ---------------------------------------------------------------------------
# Static features
# ---------------------------------------------------------------------------

def extract_static(cohort: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in cohort.iterrows():
        sid = int(row["stay_id"])
        common = {
            "patient_id":           int(row["patient_id"]),
            "stay_id":              sid,
            "hospital_admission_id": int(row["hospital_admission_id"])
                                      if pd.notna(row["hospital_admission_id"]) else pd.NA,
            "hour":     0.0,
            "dataset":  DATASET,
            "measurement_type": "static",
        }
        if pd.notna(row.get("age")):
            rows.append({**common, "variable": "age",
                         "value": float(row["age"]), "unit": "years"})
        if pd.notna(row.get("gender")):
            rows.append({**common, "variable": "sex",
                         "value": 1.0 if str(row["gender"]).upper() == "M" else 0.0,
                         "unit": "M/F"})
        # mortality_icu: true ICU death = DEATHTIME within [intime, outtime+1h].
        # (HOSPITAL_EXPIRE_FLAG is hospital mortality, kept as mortality_hospital.)
        dt    = pd.to_datetime(row.get("death_time"), errors="coerce")
        start = pd.to_datetime(row.get("intime"),     errors="coerce")
        end   = pd.to_datetime(row.get("outtime"),    errors="coerce")
        icu_mort = 0.0
        if pd.notna(dt) and pd.notna(start) and pd.notna(end) and start <= dt <= end + pd.Timedelta(hours=1):
            icu_mort = 1.0
        rows.append({**common, "measurement_type": "outcome",
                     "variable": "mortality_icu", "value": icu_mort, "unit": "bool"})
        if pd.notna(row.get("hospital_mortality")):
            rows.append({**common, "measurement_type": "outcome",
                         "variable": "mortality_hospital",
                         "value": float(row["hospital_mortality"]), "unit": "bool"})
        if pd.notna(row.get("los_hours")):
            rows.append({**common, "variable": "los_hours",
                         "value": float(row["los_hours"]), "unit": "h"})
        if pd.notna(row.get("weight_admission_kg")):
            rows.append({**common, "variable": "weight_admission",
                         "value": float(row["weight_admission_kg"]), "unit": "kg"})
        if pd.notna(row.get("height_cm")):
            rows.append({**common, "variable": "height",
                         "value": float(row["height_cm"]), "unit": "cm"})
        adm_type = row.get("admission_type")
        careunit = row.get("first_careunit")
        if pd.notna(adm_type) and pd.notna(careunit):
            pts = _admission_type_saps_iii(str(adm_type), str(careunit))
            rows.append({**common, "variable": "admission_type",
                         "value": pts, "unit": "saps2_pts"})
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
    """Run the full MIMIC-III Step-1 extraction pipeline.

    Outputs:
      output_path/mimic_iii_long.parquet
      output_path/mimic_iii_cohort.parquet
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    catalog = FeatureCatalog.from_yaml(features_yaml)
    log.info("Feature catalog loaded: %d features", len(list(catalog.keys())))

    cohort = extract_cohort(data_path, min_stay_hours, sample_size, random_state)

    parts = []

    log.info("Extracting chartevents...")
    chart = extract_chartevents(data_path, cohort, catalog)
    if not chart.empty:
        parts.append(chart)

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

    # Earliest height observation per stay → static height
    if not chart.empty and "height" in chart["variable"].values:
        ht_adm = (
            chart.loc[chart["variable"] == "height", ["stay_id", "hour", "value"]]
            .sort_values("hour")
            .groupby("stay_id")["value"].first()
            .reset_index()
            .rename(columns={"value": "height_cm"})
        )
        cohort = cohort.merge(ht_adm, on="stay_id", how="left")

    log.info("Extracting labevents...")
    lab = extract_labevents(data_path, cohort, catalog)
    if not lab.empty:
        parts.append(lab)

    log.info("Extracting inputevents (MV)...")
    inp = extract_inputevents(data_path, cohort, catalog)
    if not inp.empty:
        parts.append(inp)

    log.info("Extracting outputevents (urine)...")
    out = extract_outputevents(data_path, cohort, catalog)
    if not out.empty:
        parts.append(out)

    log.info("Extracting procedureevents (MV)...")
    proc = extract_procedureevents(data_path, cohort)
    if not proc.empty:
        parts.append(proc)

    log.info("Extracting microbiologyevents (blood cultures)...")
    micro = extract_microbiologyevents(data_path, cohort)
    if not micro.empty:
        parts.append(micro)

    log.info("Extracting antibiotic start events (INPUTEVENTS_MV)...")
    abx = extract_antibiotic_events(data_path, cohort)
    if not abx.empty:
        parts.append(abx)

    log.info("Extracting diagnoses (ICD-9, Charlson comorbidity)...")
    diag = extract_diagnoses(data_path, cohort)
    if not diag.empty:
        parts.append(diag)

    log.info("Extracting static features...")
    static = extract_static(cohort)
    if not static.empty:
        parts.append(static)

    long_df = pd.concat(parts, ignore_index=True)

    # Log BP variant counts; originals are kept as-is (no canonical coalescing).
    # SOFA computation falls back to mbp_invasive → mbp_noninvasive at evaluation time.
    for inv_name, noninv_name in [
        ("mbp_invasive", "mbp_noninvasive"),
        ("sbp_invasive", "sbp_noninvasive"),
        ("dbp_invasive", "dbp_noninvasive"),
    ]:
        n_inv    = (long_df["variable"] == inv_name).sum()
        n_noninv = (long_df["variable"] == noninv_name).sum()
        log.info("BP variants %s: inv=%d  noninv=%d (kept separately, no coalescing)", inv_name.split("_")[0], n_inv, n_noninv)

    long_df = long_df.sort_values(["stay_id", "hour", "variable"]).reset_index(drop=True)

    # Clip observations to the 7-day (168h) modelling horizon, consistent with
    # eICU/HiRID (which clip in Stage-1) and the Stage-3 168h window cap. Keeps
    # static features at hour=0. Cohort LOS is left intact for reporting.
    _n0 = len(long_df)
    long_df = long_df.loc[(long_df["hour"] >= 0) & (long_df["hour"] <= 168.0)].copy()
    log.info("168h horizon clip: %d -> %d rows (-%d)", _n0, len(long_df), _n0 - len(long_df))

    long_path = output_path / "mimic_iii_long.parquet"
    cohort_path = output_path / "mimic_iii_cohort.parquet"
    long_df.to_parquet(long_path, index=False)
    log.info("Saved %d rows to %s", len(long_df), long_path)
    cohort.to_parquet(cohort_path, index=False)
    log.info("Saved cohort (%d stays) to %s", len(cohort), cohort_path)

    return long_df
