"""
harmonize/stage2_preprocessing.py
Stage 2: Hourly aggregation, normalisation, patient-level train/val/test split.

Input:  data/long/{ds}_long.parquet + {ds}_cohort.parquet  (Stage 1 output)
Output: data/splits/
          {ds}_train.parquet
          {ds}_val.parquet
          {ds}_test.parquet
          {ds}_norm_stats.parquet   (train-set mean/std per variable)
          split_manifest.csv        (patient_id → split assignment for each dataset)

Split strategy:
  - Patient-level (no patient appears in more than one split)
  - 64 / 16 / 20  (train / val / test)
  - Random (unstratified) — mortality rates are stable at n>1000
  - Fixed random_state=42

Hourly aggregation:
  - Floor hour to integer bin
  - value_mean: mean per (stay_id, hour, variable)
  - value_first: first recorded value per (stay_id, hour, variable)

Normalisation:
  - Z-score using train-set statistics
  - Variables with std=0 or all-NaN left as-is (std set to 1)
  - Static / demographic variables (age, sex) normalised separately

Usage:
  python -m harmonize.stage2_preprocessing             # all datasets
  python -m harmonize.stage2_preprocessing --ds eicu   # single dataset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRAIN_FRAC = 0.64
VAL_FRAC   = 0.16
TEST_FRAC  = 0.20   # remainder

LONG_DIR   = Path("data/long")
OUTPUT_DIR = Path("data/splits")

DATASETS   = ["eicu", "hirid", "mimic_iv", "umm", "mimic_iii"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_long(ds: str, long_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(long_dir / f"{ds}_long.parquet")


def load_cohort(ds: str, long_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(long_dir / f"{ds}_cohort.parquet")


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Floor hour to integer; compute mean/first/min/max/std/count per bin."""
    df = df.copy()
    df["hour"] = np.floor(df["hour"]).astype(int)
    key_cols = ["patient_id", "stay_id", "hospital_admission_id",
                "hour", "variable", "unit", "measurement_type"]
    df = df.sort_values(key_cols, na_position="last")
    # dropna=False keeps rows where hospital_admission_id is NA (eICU, HiRID)
    grp = df.groupby(key_cols, as_index=False, dropna=False, sort=False)["value"]
    agg = grp.mean().rename(columns={"value": "value_mean"})
    agg["value_first"] = grp.first()["value"].values
    agg["value_min"]   = grp.min()["value"].values
    agg["value_max"]   = grp.max()["value"].values
    agg["value_std"]   = grp.std()["value"].values
    agg["value_count"] = grp.count()["value"].values
    return agg


def split_patients(
    patient_ids: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random patient-level 64/16/20 split."""
    rng = np.random.default_rng(random_state)
    ids = np.array(sorted(patient_ids))
    rng.shuffle(ids)
    n       = len(ids)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def compute_norm_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std per variable from training set (fit on value_mean)."""
    numeric = df_train[df_train["measurement_type"] != "outcome"]
    stats = (
        numeric.groupby("variable")["value_mean"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "train_mean", "std": "train_std"})
        .reset_index()
    )
    stats["train_std"] = stats["train_std"].fillna(1.0).replace(0.0, 1.0)
    return stats


def apply_normalisation(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalise all value_* columns; leave demographics unchanged."""
    stat_map = stats.set_index("variable")[["train_mean", "train_std"]]
    mask = (df["measurement_type"] != "outcome") & df["variable"].isin(stat_map.index)
    df = df.copy()
    for col in ["value_mean", "value_first", "value_min", "value_max", "value_std"]:
        df.loc[mask, col] = (
            (df.loc[mask, col] - df.loc[mask, "variable"].map(stat_map["train_mean"]))
            / df.loc[mask, "variable"].map(stat_map["train_std"])
        )
    return df


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def process_dataset(
    ds: str,
    long_dir: Path,
    output_dir: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """Full Stage 2 for one dataset. Returns split manifest."""
    log.info("=== %s ===", ds.upper())

    long   = load_long(ds, long_dir)
    cohort = load_cohort(ds, long_dir)

    log.info("Long rows: %d | Cohort stays: %d", len(long), len(cohort))

    # Hourly aggregation
    log.info("Aggregating to hourly bins...")
    long = aggregate_hourly(long)
    log.info("After aggregation: %d rows", len(long))

    # Patient-level random split
    patient_ids = cohort["patient_id"].unique()
    train_ids, val_ids, test_ids = split_patients(patient_ids, random_state)
    log.info(
        "Split: train=%d  val=%d  test=%d  (%.0f/%.0f/%.0f%%)",
        len(train_ids), len(val_ids), len(test_ids),
        100 * len(train_ids) / len(patient_ids),
        100 * len(val_ids)   / len(patient_ids),
        100 * len(test_ids)  / len(patient_ids),
    )

    # Assign split labels
    id_to_split = (
        {pid: "train" for pid in train_ids}
        | {pid: "val"   for pid in val_ids}
        | {pid: "test"  for pid in test_ids}
    )
    long["split"] = long["patient_id"].map(id_to_split)

    # Normalisation: fit on train set
    log.info("Computing normalisation statistics on train set...")
    train_df = long[long["split"] == "train"]
    norm_stats = compute_norm_stats(train_df)
    norm_stats.to_parquet(output_dir / f"{ds}_norm_stats.parquet", index=False)
    log.info("  norm_stats: %d variables", len(norm_stats))

    log.info("Applying Z-score normalisation...")
    long = apply_normalisation(long, norm_stats)

    # Save splits — naming: {ds}_seed{seed}_{split}.parquet
    for split_name in ["train", "val", "test"]:
        split_df = long[long["split"] == split_name].drop(columns="split")
        out_path = output_dir / f"{ds}_seed{random_state}_{split_name}.parquet"
        split_df.to_parquet(out_path, index=False)
        n_stays = split_df["stay_id"].nunique()
        log.info("  %s: %d rows, %d stays -> %s", split_name, len(split_df), n_stays, out_path)

    # Build manifest
    manifest_rows = []
    for pid in patient_ids:
        manifest_rows.append({"dataset": ds, "patient_id": pid,
                               "split": id_to_split.get(pid, "unknown")})
    return pd.DataFrame(manifest_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    datasets: list[str] | None = None,
    long_dir: Path = LONG_DIR,
    output_dir: Path = OUTPUT_DIR,
    random_state: int = 42,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [ds for ds in DATASETS if (long_dir / f"{ds}_long.parquet").exists()]
        log.info("Datasets found: %s", datasets)

    manifests = []
    for ds in datasets:
        long_path = long_dir / f"{ds}_long.parquet"
        if not long_path.exists():
            log.warning("Skipping %s — %s not found", ds, long_path)
            continue
        manifests.append(process_dataset(ds, long_dir, output_dir, random_state))

    if manifests:
        manifest = pd.concat(manifests, ignore_index=True)
        manifest.to_csv(output_dir / "split_manifest.csv", index=False)
        log.info("Split manifest saved (%d patients total)", len(manifest))

        # Summary
        summary = manifest.groupby(["dataset", "split"]).size().unstack(fill_value=0)
        log.info("Split summary:\n%s", summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 preprocessing")
    parser.add_argument("--ds", nargs="+", default=None,
                        help="Datasets to process (default: all available)")
    parser.add_argument("--long-dir",   default=str(LONG_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        datasets=args.ds,
        long_dir=Path(args.long_dir),
        output_dir=Path(args.output_dir),
        random_state=args.seed,
    )
