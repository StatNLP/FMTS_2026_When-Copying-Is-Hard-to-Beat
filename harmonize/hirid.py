"""
harmonize/hirid.py
Step-1 extraction pipeline for HiRID 1.1.1 (Bern Inselspital surgical ICU).

Schema output: Long-Format Parquet — identical schema to mimic_iv.py.
  patient_id | stay_id | hospital_admission_id | hour | variable | value | unit | dataset | measurement_type

Design notes:
  - HiRID has no separate icustays table: one patientid = one ICU stay.
    stay_id == patient_id; hospital_admission_id = pd.NA throughout.
  - All vitals and labs are in observation_tables (no vital/lab split).
  - Data is partitioned across 250 part files; merged_stage_index maps patient→part.
  - LOS is not available upfront; computed from max(datetime) - admissiontime
    during the observation pass and used for min_stay_hours filtering.
  - Vasopressor rates normalised to ug/kg/min using body weight from
    observation variableid 10000400. If weight is missing for a patient,
    the median cohort weight is used as fallback.
  - Dopamine is absent from HiRID (Bern protocol; not a data gap).
  - Status codes: HiRID uses positive codes for valid measurements.
    Records with status < 0 are excluded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from harmonize.common import (
    FeatureCatalog,
    apply_outlier_filter,
    hours_since_intime,
)

log = logging.getLogger(__name__)

DATASET = "hirid"

# ---------------------------------------------------------------------------
# Observation variable-ID registries
# ---------------------------------------------------------------------------

# Core vitals + labs: feature_key -> list of HiRID variableids
OBS_NUMERIC: dict[str, list[int]] = {
    # Vitals
    "hr":               [200],
    "rr":               [300, 310, 5685],
    "spo2":             [4000, 8280],
    "sbp_invasive":     [100],
    "dbp_invasive":     [120],
    "mbp_invasive":     [110],
    "sbp_noninvasive":  [600],
    "dbp_noninvasive":  [620],
    "mbp_noninvasive":  [610],
    "temp_core":        [410, 7100, 400],
    "fio2":             [2010],       # % -> /100 -> fraction
    "peep":             [2600, 2610],
    "etco2":            [2200, 8290, 30010009],
    "tidal_volume":     [2410],             # Expiratory tidal volume, mL
    # amv (Minute Ventilation, L/min): not directly measured in HiRID.
    # Derivable in Step 2 as tidal_volume * rr.
    # GCS
    "gcs_eye":          [10000300],
    "gcs_motor":        [10000200],
    "gcs_verbal":       [10000100],
    # Anthropometrics
    "weight":           [10000400],   # kg
    "height":           [10000450],   # cm
    # Blood gas
    "ph_arterial":      [20000300],
    "pao2":             [20000200],   # mmHg
    "paco2":            [20001200],   # mmHg
    "sao2":             [20000800],   # %
    "base_excess":      [20001300],   # mmol/l
    "hco3":             [20004200],   # mmol/l
    # Chemistry / electrolytes
    "lactate":          [24000524, 24000732, 24000485],  # mmol/l
    "glucose":          [20005110, 24000523, 24000585],  # mmol/l
    "sodium":           [20000400, 24000519, 24000658, 24000835, 24000866],  # mmol/l
    "potassium":        [20000500, 24000520, 24000833, 24000867],            # mmol/l
    "chloride":         [24000439, 24000521],   # mmol/l
    "magnesium":        [24000230],             # mmol/l
    "phosphate":        [20002500],             # mmol/l
    "calcium_total":    [20005100],             # mmol/l
    "calcium_free":     [24000522],             # mmol/l (ionised)
    "creatinine":       [20000600],             # umol/l -> *0.01131 mg/dl
    "bilirubin_total":  [20004300],             # umol/l -> *0.05848 mg/dl
    "bilirubin_direct": [24000560],             # umol/l -> *0.05848 mg/dl
    "hemoglobin":       [20000900, 24000836],   # g/l -> /10 g/dl
    "platelets":        [20000110],             # G/l (= 10^9/l, same scale as MIMIC)
    "albumin":          [24000605],             # g/L -> /10 g/dL
    "wbc":              [20000700],             # Leukocytes G/l
    "bun":              [20004100],             # Urea mmol/l -> *2.8 mg/dl
    "alp":              [20002700],             # U/l
    "alt":              [20002600],             # U/l
    "ast":              [24000330],             # U/l
    "lipase":           [24000555],             # U/l
    "inr":              [24000567],
    "ptt":              [20004410],             # aPTT seconds
    "fibrinogen":       [24000536],             # g/L
    "troponin_t":       [24000538, 24000806],   # Troponin T.cardiac + hs-TnT, ng/L
    # NOTE: HiRID measures Troponin T, not Troponin I (different proteins).
    # Previously emitted as `troponin_i` (cross-dataset proxy) — corrected to
    # `troponin_t` per features.yaml. troponin_i is not_available for HiRID.
    "cardiac_output":   [1000],                 # l/min
    "svo2":             [4200, 20001000, 24000737],  # mixed/central venous O2 sat %
    "neutrophils_pct":  [24000550, 24000556, 24000557],  # %
    "lymphocytes_abs":  [24000480],             # G/l
    "mcv":              [24000150],             # fl
    "mch":              [24000160],             # pg
    "mchc":             [24000170],             # g/L
    "pct":              [24000570],             # procalcitonin ug/l
    # Urine output (hourly volume — already in ml/h in HiRID)
    "urine_output":     [10020000],             # ml/h
}

# Unit conversions required to match MIMIC-IV / catalog target units
# Maps feature_key -> (factor, operation): value_out = value_in * factor
# For division, use 1/factor.
OBS_UNIT_CONVERSIONS: dict[str, float] = {
    "fio2":             1 / 100,    # % -> fraction [0, 1]
    "creatinine":       0.01131,    # umol/l -> mg/dl
    "bilirubin_total":  0.05848,    # umol/l -> mg/dl
    "bilirubin_direct": 0.05848,    # umol/l -> mg/dl
    "hemoglobin":       0.1,        # g/l -> g/dl
    "albumin":          0.1,        # g/L -> g/dL
    "mchc":             0.1,        # g/L -> g/dL
    "bun":              2.8,        # urea mmol/l -> BUN mg/dl
    "glucose":          18.016,     # mmol/l -> mg/dl
}

# Mechanical ventilation: ventilator mode variable (categorical stringvalue)
MV_VARIABLEID: list[int] = [3845]
# Any non-null, non-empty stringvalue = ventilated that hour
MV_EXCLUDE_VALUES: set[str] = {"", "nan", "none", "spontan", "spontaneous"}

# ---------------------------------------------------------------------------
# Pharma (medication) variable-ID registries
# ---------------------------------------------------------------------------

# Vasoactive: target unit ug/kg/min (rate normalised from givendose + weight)
PHARMA_VASOPRESSORS: dict[str, list[int]] = {
    "norepinephrine": [1000462, 1000656, 1000657, 1000658],
    "epinephrine":    [71, 1000649, 1000650, 1000655, 1000750],
    "dobutamine":     [426],
}
# Vasopressin: target unit IU/min
PHARMA_VASOPRESSIN: dict[str, list[int]] = {
    "vasopressin": [112, 113],
}
# Other infusions — rate output in catalog unit/h (rate_per_min × 60).
# Propofol is output as mg/h (not mg/kg/h); weight normalisation deferred to Step 2.
# pt (prothrombin time) is absent: HiRID stores prothrombin activity in %, not seconds.
PHARMA_OTHER: dict[str, list[int]] = {
    "fentanyl":   [1000251, 1000692, 1000741],
    "insulin":    [15, 1000724],                              # Actrapid + NovoRapid
    "propofol":   [208, 1000491, 1001050, 1001052, 1001053, 1000691, 1000699],  # Disoprivan
    "midazolam":  [251, 252, 1000700, 1001051, 1001054],     # Dormicum
    "morphine":   [1000444, 1000659, 1000862, 1000932, 1000684, 1000716],
    # NOTE: furosemide (variableid 1000232) and heparin (1000373) are NOT
    # extracted for HiRID: each appears in only a handful of stays (furosemide
    # 5, heparin 2), far too few for a usable forecast target — heparin's
    # near-zero coverage also produced a degenerate mean=0/std=1 normalisation
    # that blew up the MSE. See features.yaml (hirid: not_available) and docs.
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patient_to_part(data_path: Path) -> dict[int, int]:
    """Return mapping patient_id -> part-file index from merged_stage_index."""
    idx = pd.read_csv(
        data_path / "merged_stage" / "merged_stage" / "merged_stage_index.csv"
    )
    return dict(zip(idx["patientid"], idx["part"]))


def _parts_for_cohort(cohort: pd.DataFrame, patient_to_part: dict[int, int]) -> dict[int, list[int]]:
    """Return {part_number: [patient_ids]} for the cohort."""
    part_to_patients: dict[int, list[int]] = {}
    for pid in cohort["patient_id"]:
        part = patient_to_part.get(int(pid))
        if part is not None:
            part_to_patients.setdefault(part, []).append(int(pid))
    return part_to_patients


def _read_obs_part(
    part_file: Path,
    patient_ids: list[int],
    variableids: set[int],
) -> pd.DataFrame:
    """Read one observation part file, filter to patients and variable IDs."""
    df = pd.read_parquet(
        part_file,
        columns=["datetime", "patientid", "status", "value", "variableid"],
    )
    df = df.loc[df["patientid"].isin(patient_ids)]
    df = df.loc[df["variableid"].isin(variableids)]
    df = df.loc[df["status"] >= 0]          # exclude invalid status codes
    df = df.dropna(subset=["value"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def _read_pharma_part(
    part_file: Path,
    patient_ids: list[int],
    pharmaids: set[int],
) -> pd.DataFrame:
    """Read one pharma part file, filter to patients and pharma IDs."""
    df = pd.read_parquet(
        part_file,
        columns=["patientid", "pharmaid", "givenat", "givendose", "doseunit", "route", "infusionid"],
    )
    df = df.loc[df["patientid"].isin(patient_ids)]
    df = df.loc[df["pharmaid"].isin(pharmaids)]
    df = df.dropna(subset=["givenat", "givendose"])
    df["givenat"] = pd.to_datetime(df["givenat"])
    return df


def _build_weight_lookup(obs_long: pd.DataFrame) -> pd.Series:
    """Return Series patient_id -> weight_kg from observation data."""
    w = obs_long.loc[obs_long["variable"] == "weight", ["patient_id", "value"]]
    if w.empty:
        return pd.Series(dtype=float)
    # use median per patient (weight is typically recorded at admission)
    return w.groupby("patient_id")["value"].median()


# ---------------------------------------------------------------------------
# Cohort extraction
# ---------------------------------------------------------------------------

def extract_cohort(
    data_path: str | Path,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load HiRID cohort from general_table.csv.

    Note: LOS cannot be determined without reading observation data.
    min_stay_hours filtering is applied in run_pipeline() after the
    observation pass, which computes the actual last-observation hour.

    Returns DataFrame with columns:
      patient_id, admissiontime, sex, age, discharge_status, part
    """
    data_path = Path(data_path)
    log.info("Loading general_table...")
    gen = pd.read_parquet(data_path / "general_table.parquet")
    gen["admissiontime"] = pd.to_datetime(gen["admissiontime"])
    gen = gen.rename(columns={"patientid": "patient_id"})
    gen = gen.loc[gen["age"] >= 18].copy()

    log.info("Total patients in general_table: %d", len(gen))

    p2p = _patient_to_part(data_path)
    gen["part"] = gen["patient_id"].map(p2p)
    gen = gen.dropna(subset=["part"])
    gen["part"] = gen["part"].astype(int)

    if sample_size is not None and sample_size < len(gen):
        rng = np.random.default_rng(random_state)
        sampled = rng.choice(gen["patient_id"].unique(), size=sample_size, replace=False)
        gen = gen.loc[gen["patient_id"].isin(sampled)].copy()
        log.info("Sampled to %d patients", len(gen))

    return gen.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Observation extraction
# ---------------------------------------------------------------------------

def extract_observations(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    """Extract all numeric observations (vitals + labs) for the cohort.

    Reads raw_stage/observation_tables part files, one per unique part in cohort.
    Applies unit conversions and outlier filters per feature.

    Returns Long-Format DataFrame.
    """
    data_path = Path(data_path)
    obs_dir = data_path / "raw_stage" / "observation_tables" / "parquet"

    intime = cohort.set_index("patient_id")["admissiontime"]
    p2p = dict(zip(cohort["patient_id"], cohort["part"]))
    part_to_patients = _parts_for_cohort(cohort, p2p)

    # Build reverse lookup: variableid -> feature_key (first mapping wins)
    all_varids: set[int] = set()
    varid_to_feat: dict[int, str] = {}
    for feat, vids in OBS_NUMERIC.items():
        for vid in vids:
            all_varids.add(vid)
            if vid not in varid_to_feat:
                varid_to_feat[vid] = feat

    parts: list[pd.DataFrame] = []
    _filter_removed_total = 0
    _filter_total_total   = 0

    for part_num, patient_ids in sorted(part_to_patients.items()):
        part_file = obs_dir / f"part-{part_num}.parquet"
        if not part_file.exists():
            log.warning("Part file not found: %s", part_file)
            continue

        df = _read_obs_part(part_file, patient_ids, all_varids)
        if df.empty:
            continue

        df["variable"] = df["variableid"].map(varid_to_feat)
        df = df.dropna(subset=["variable"])

        # Compute hours since admission
        df["hour"] = hours_since_intime(
            df["datetime"],
            df["patientid"].map(intime),
        )

        # Apply unit conversions
        for feat, factor in OBS_UNIT_CONVERSIONS.items():
            mask = df["variable"] == feat
            df.loc[mask, "value"] = df.loc[mask, "value"] * factor

        # Apply outlier filter per feature — track removed rows
        _n_pre = len(df)
        filtered: list[pd.DataFrame] = []
        for feat_key, grp in df.groupby("variable"):
            vr = catalog.value_range(feat_key) if feat_key in catalog else None
            grp = apply_outlier_filter(grp.copy(), "value", vr)
            filtered.append(grp)
        df = pd.concat(filtered, ignore_index=True) if filtered else df
        df = df.dropna(subset=["value"])
        _filter_removed_part = _n_pre - len(df)
        _filter_removed_total += _filter_removed_part
        _filter_total_total   += _n_pre
        if _filter_removed_part > 0:
            log.debug("Part %d: value_range filter removed %d rows", part_num, _filter_removed_part)

        # Attach metadata
        df["patient_id"] = df["patientid"]
        df["stay_id"] = df["patientid"]
        df["hospital_admission_id"] = pd.NA
        df["unit"] = df["variable"].map(
            {k: catalog.unit(k) for k in catalog.keys()}
        )
        df["measurement_type"] = df["variable"].map(
            {k: catalog.measurement_type(k) for k in catalog.keys()}
        )
        df["dataset"] = DATASET

        parts.append(
            df[["patient_id", "stay_id", "hospital_admission_id",
                "hour", "variable", "value", "unit", "dataset",
                "measurement_type"]].copy()
        )

    if not parts:
        log.warning("No observation data found for cohort.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    log.info(
        "Observations: %d rows, %d features, %d patients | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(), result["patient_id"].nunique(),
        _filter_removed_total, _filter_total_total,
        100.0 * _filter_removed_total / _filter_total_total if _filter_total_total else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Mechanical ventilation extraction
# ---------------------------------------------------------------------------

def extract_mechanical_ventilation(
    data_path: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract mechanical ventilation as hourly binary flag (1.0 = ventilated).

    Derived from ventilator mode (variableid 3845): any charted mode that is
    not in MV_EXCLUDE_VALUES is treated as active mechanical ventilation.
    The flag is set to 1.0 at each charted hour; hours without a record remain
    absent (NaN after wide-format aggregation).
    """
    data_path = Path(data_path)
    obs_dir = data_path / "raw_stage" / "observation_tables" / "parquet"

    intime = cohort.set_index("patient_id")["admissiontime"]
    p2p = dict(zip(cohort["patient_id"], cohort["part"]))
    part_to_patients = _parts_for_cohort(cohort, p2p)

    parts: list[pd.DataFrame] = []

    for part_num, patient_ids in sorted(part_to_patients.items()):
        part_file = obs_dir / f"part-{part_num}.parquet"
        if not part_file.exists():
            continue

        df = pd.read_parquet(
            part_file,
            columns=["datetime", "patientid", "status", "value", "stringvalue", "variableid"],
        )
        df = df.loc[df["patientid"].isin(patient_ids)]
        df = df.loc[df["variableid"].isin(MV_VARIABLEID)]
        df = df.loc[df["status"] >= 0]
        # variableid 3845 stores ventilator mode as a numeric code (value > 0).
        # stringvalue is NaN for all records; apply exclusion only to actual string records.
        has_numeric = df["value"].notna() & (df["value"] > 0) & df["stringvalue"].isna()
        str_mask    = df["stringvalue"].notna()
        valid_str   = str_mask & (
            ~df["stringvalue"].astype(str).str.strip().str.lower().isin(MV_EXCLUDE_VALUES)
        )
        df = df.loc[has_numeric | valid_str].copy()
        if df.empty:
            continue

        df["datetime"] = pd.to_datetime(df["datetime"])
        df["hour"] = hours_since_intime(
            df["datetime"],
            df["patientid"].map(intime),
        )
        df["patient_id"] = df["patientid"]
        df["stay_id"] = df["patientid"]
        df["hospital_admission_id"] = pd.NA
        df["variable"] = "mechanical_ventilation"
        df["value"] = 1.0
        df["unit"] = "bool"
        df["measurement_type"] = "ventilation"
        df["dataset"] = DATASET

        parts.append(
            df[["patient_id", "stay_id", "hospital_admission_id",
                "hour", "variable", "value", "unit", "dataset",
                "measurement_type"]].copy()
        )

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    log.info("Mechanical ventilation: %d rows, %d patients",
             len(result), result["patient_id"].nunique())
    return result


# ---------------------------------------------------------------------------
# Pharma extraction
# ---------------------------------------------------------------------------

def _rate_from_infusion(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute dose rate (givendose / interval_minutes) per infusion record.

    For continuous infusions, givendose is the amount delivered in the interval
    between consecutive givenat timestamps. Rate = givendose / interval_min.
    For the first record of each infusionid (no predecessor), we use 2.0 min
    as the default HiRID interval.
    """
    grp = grp.sort_values("givenat").copy()
    intervals = grp["givenat"].diff().dt.total_seconds().div(60).fillna(2.0)
    intervals = intervals.clip(lower=0.5)  # guard against zero / negative gaps
    grp["rate_per_min"] = grp["givendose"] / intervals
    return grp


def extract_pharma(
    data_path: str | Path,
    cohort: pd.DataFrame,
    catalog: FeatureCatalog,
    weight_lookup: pd.Series,
) -> pd.DataFrame:
    """Extract medication infusion rates from pharma_records.

    Vasopressor rates are normalised to ug/kg/min using patient body weight.
    If weight is unavailable for a patient, the cohort median is used.
    Vasopressin is normalised to IU/min.
    Other medications use the raw rate_per_min (unit depends on doseunit).

    Parameters
    ----------
    weight_lookup : Series  patient_id -> weight_kg  (from extract_observations)
    """
    data_path = Path(data_path)
    pharma_dir = data_path / "raw_stage" / "pharma_records" / "parquet"

    intime = cohort.set_index("patient_id")["admissiontime"]
    p2p = dict(zip(cohort["patient_id"], cohort["part"]))
    part_to_patients = _parts_for_cohort(cohort, p2p)

    # Build reverse lookup: pharmaid -> (feature_key, category)
    pharmaid_to_info: dict[int, tuple[str, str]] = {}
    for feat, pids in PHARMA_VASOPRESSORS.items():
        for pid in pids:
            pharmaid_to_info[pid] = (feat, "vasopressor")
    for feat, pids in PHARMA_VASOPRESSIN.items():
        for pid in pids:
            pharmaid_to_info[pid] = (feat, "vasopressin")
    for feat, pids in PHARMA_OTHER.items():
        for pid in pids:
            if pid not in pharmaid_to_info:
                pharmaid_to_info[pid] = (feat, "other")

    all_pharmaids = set(pharmaid_to_info.keys())

    # Weight fallback: cohort median
    median_weight = weight_lookup.median() if not weight_lookup.empty else 75.0
    log.info("Vasopressor weight normalisation: cohort median weight = %.1f kg "
             "(used for %d patients without weight record)",
             median_weight,
             cohort["patient_id"].isin(weight_lookup.index).shape[0] - weight_lookup.notna().sum())

    parts: list[pd.DataFrame] = []
    _filter_removed_pharma = 0
    _filter_total_pharma   = 0

    for part_num, patient_ids in sorted(part_to_patients.items()):
        part_file = pharma_dir / f"part-{part_num}.parquet"
        if not part_file.exists():
            continue

        df = _read_pharma_part(part_file, patient_ids, all_pharmaids)
        if df.empty:
            continue

        df["feature_key"] = df["pharmaid"].map(
            {pid: feat for pid, (feat, _) in pharmaid_to_info.items()}
        )
        df["category"] = df["pharmaid"].map(
            {pid: cat for pid, (_, cat) in pharmaid_to_info.items()}
        )

        feat_parts: list[pd.DataFrame] = []

        for feat_key, fgrp in df.groupby("feature_key"):
            category = fgrp["category"].iloc[0]
            fgrp = fgrp.copy()

            # Compute rate from consecutive givenat timestamps per infusionid
            fgrp = (
                fgrp.groupby("infusionid", group_keys=False)
                .apply(_rate_from_infusion)
            )

            if category == "vasopressor":
                # rate_per_min unit depends on doseunit in the pharma records.
                # Norepinephrine / epinephrine are recorded in µg → rate is µg/min.
                # Dobutamine (pharmaid 426) is recorded in mg → convert to µg first.
                mg_mask = fgrp["doseunit"].astype(str).str.strip() == "mg"
                fgrp.loc[mg_mask, "rate_per_min"] = fgrp.loc[mg_mask, "rate_per_min"] * 1000

                def _weight(pid: int) -> float:
                    return float(weight_lookup.get(pid, median_weight))

                fgrp["weight_kg"] = fgrp["patientid"].map(_weight)
                fgrp["value"] = fgrp["rate_per_min"] / fgrp["weight_kg"]

            elif category == "vasopressin":
                # givendose in Units, rate_per_min in U/min -> IU/min
                fgrp["value"] = fgrp["rate_per_min"]

            else:
                # Convert per-minute rate to per-hour to match catalog units
                # (fentanyl: ug/h, furosemide/midazolam/morphine/propofol: mg/h,
                #  insulin/heparin: IU/h)
                fgrp["value"] = fgrp["rate_per_min"] * 60.0

            # Apply outlier filter with pre/post count
            vr = catalog.value_range(feat_key) if feat_key in catalog else None
            _n_pre_pharma = len(fgrp)
            fgrp = apply_outlier_filter(fgrp, "value", vr)
            fgrp = fgrp.dropna(subset=["value"])
            _filter_removed_pharma += _n_pre_pharma - len(fgrp)
            _filter_total_pharma   += _n_pre_pharma
            if fgrp.empty:
                continue

            fgrp["hour"] = hours_since_intime(
                fgrp["givenat"],
                fgrp["patientid"].map(intime),
            )
            fgrp["variable"] = feat_key
            fgrp["patient_id"] = fgrp["patientid"]
            fgrp["stay_id"] = fgrp["patientid"]
            fgrp["hospital_admission_id"] = pd.NA
            fgrp["unit"] = catalog.unit(feat_key) if feat_key in catalog else ""
            fgrp["measurement_type"] = (
                catalog.measurement_type(feat_key) if feat_key in catalog else "medication"
            )
            fgrp["dataset"] = DATASET
            feat_parts.append(
                fgrp[["patient_id", "stay_id", "hospital_admission_id",
                       "hour", "variable", "value", "unit", "dataset",
                       "measurement_type"]].copy()
            )

        if feat_parts:
            parts.append(pd.concat(feat_parts, ignore_index=True))

    if not parts:
        log.warning("No pharma data found for cohort.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    log.info(
        "Pharma: %d rows, %d features | value_range filter removed %d / %d (%.2f%%)",
        len(result), result["variable"].nunique(),
        _filter_removed_pharma, _filter_total_pharma,
        100.0 * _filter_removed_pharma / _filter_total_pharma if _filter_total_pharma else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Static features
# ---------------------------------------------------------------------------

def extract_static(cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract static features (age, sex) at hour=0 from general_table.

    Weight and height are recorded as observations (variableid 10000400 /
    10000450) and extracted via extract_observations(). They appear at the
    hour they were first charted, not necessarily hour=0.
    """
    rows: list[dict] = []
    for _, row in cohort.iterrows():
        pid = int(row["patient_id"])
        common = {
            "patient_id": pid,
            "stay_id": pid,
            "hospital_admission_id": pd.NA,
            "hour": 0.0,
            "dataset": DATASET,
            "measurement_type": "static",
        }
        if pd.notna(row.get("age")):
            rows.append({**common, "variable": "age",
                         "value": float(row["age"]), "unit": "years"})
        if pd.notna(row.get("sex")):
            rows.append({**common, "variable": "sex",
                         "value": 1.0 if str(row["sex"]).upper() == "M" else 0.0,
                         "unit": "M/F"})
        if pd.notna(row.get("discharge_status")):
            rows.append({**common, "measurement_type": "outcome",
                         "variable": "mortality_icu",
                         "value": 1.0 if str(row["discharge_status"]).lower() == "dead" else 0.0,
                         "unit": "bool"})
        if pd.notna(row.get("los_hours")):
            rows.append({**common, "variable": "los_hours",
                         "value": float(row["los_hours"]), "unit": "h"})
        if pd.notna(row.get("weight_admission_kg")):
            rows.append({**common, "variable": "weight_admission",
                         "value": float(row["weight_admission_kg"]), "unit": "kg"})

    result = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not result.empty:
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
    """Run the full HiRID Step-1 extraction pipeline.

    Outputs:
      output_path/hirid_long.parquet
      output_path/hirid_cohort.parquet

    Parameters
    ----------
    data_path       : HiRID root directory (contains general_table.csv,
                      raw_stage/, merged_stage/)
    output_path     : Directory for output Parquet files
    features_yaml   : Path to configs/features.yaml
    sample_size     : Subsample N patients (for fast iteration)
    min_stay_hours  : Minimum ICU LOS; patients whose last observation is
                      earlier than this threshold are excluded
    random_state    : Seed for reproducible sampling
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    catalog = FeatureCatalog.from_yaml(features_yaml)
    log.info("Feature catalog loaded: %d features", len(list(catalog.keys())))

    cohort = extract_cohort(data_path, sample_size=sample_size,
                            random_state=random_state)

    log.info("Extracting observations...")
    obs = extract_observations(data_path, cohort, catalog)

    # Apply min_stay_hours filter using max observed hour per patient
    if not obs.empty and min_stay_hours > 0:
        max_hour = obs.groupby("patient_id")["hour"].max()
        valid_patients = max_hour.index[max_hour >= min_stay_hours]
        n_before = cohort["patient_id"].nunique()
        cohort = cohort.loc[cohort["patient_id"].isin(valid_patients)].copy()
        obs = obs.loc[obs["patient_id"].isin(valid_patients)].copy()
        log.info(
            "min_stay_hours=%.0f filter: %d -> %d patients",
            min_stay_hours, n_before, cohort["patient_id"].nunique(),
        )
        # Store los_hours in cohort (max observed hour approximates ICU LOS)
        cohort = cohort.copy()
        cohort["los_hours"] = cohort["patient_id"].map(max_hour)

    log.info("Extracting mechanical ventilation...")
    mv = extract_mechanical_ventilation(data_path, cohort)

    # Earliest weight observation per patient → static admission weight
    if not obs.empty and "weight" in obs["variable"].values:
        wt_adm = (
            obs.loc[obs["variable"] == "weight", ["patient_id", "hour", "value"]]
            .sort_values("hour")
            .groupby("patient_id")["value"].first()
            .reset_index()
            .rename(columns={"value": "weight_admission_kg"})
        )
        cohort = cohort.merge(wt_adm, on="patient_id", how="left")

    log.info("Building weight lookup for vasopressor normalisation...")
    weight_lookup = _build_weight_lookup(obs)
    log.info("  Weight available for %d / %d patients",
             len(weight_lookup), cohort["patient_id"].nunique())

    log.info("Extracting pharma...")
    pharma = extract_pharma(data_path, cohort, catalog, weight_lookup)

    log.info("Extracting static features...")
    static = extract_static(cohort)

    parts = [df for df in [obs, mv, pharma, static] if not df.empty]
    long_df = pd.concat(parts, ignore_index=True)

    # Clip to 7-day horizon (168h) for cross-DB experiments
    long_df = long_df.loc[
        (long_df["hour"] >= 0) & (long_df["hour"] <= 168.0)
    ].copy()

    out_file = output_path / "hirid_long.parquet"
    long_df.to_parquet(out_file, index=False)
    log.info("Saved %d rows to %s", len(long_df), out_file)

    cohort_file = output_path / "hirid_cohort.parquet"
    cohort.to_parquet(cohort_file, index=False)
    log.info("Saved cohort (%d patients) to %s", len(cohort), cohort_file)

    return long_df
