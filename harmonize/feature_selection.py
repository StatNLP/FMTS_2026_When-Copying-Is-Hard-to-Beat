"""
harmonize/feature_selection.py
Resolve a `--features <keyword>` CLI argument into a concrete list of
canonical feature names, using `configs/features.yaml` + the empirical
feature_overview tables.

Keywords:
    all                 — every feature emitted by Stage 2 long parquet (dataset-specific union)
    sofa                — catalog features with sofa_relevant: true
    core                — catalog features with tier: core
    covered<P>          — features with empirical encounter share >= P (e.g. covered10)
    custom:<path>       — newline-separated feature names in a file

Returned roles per feature are derived from catalog `type`:
    observation  ← vital, lab
    intervention ← medication, fluid, ventilation
    static       ← demographic, outcome
    skip         ← derived  (computed downstream in Stage 4 / scores)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "features.yaml"
COVERAGE_CSV_DIR = ROOT / "data" / "tables"

Role = Literal["observation", "intervention", "static", "skip"]

ROLE_BY_TYPE: dict[str, Role] = {
    "vital":       "observation",
    "lab":         "observation",
    "medication":  "intervention",
    "fluid":       "intervention",
    "ventilation": "intervention",
    "demographic": "static",
    "outcome":     "static",
    "derived":     "skip",
}


def load_catalog() -> dict:
    return yaml.safe_load(open(CATALOG))


def role_of(feature: str, catalog: dict | None = None) -> Role:
    if catalog is None:
        catalog = load_catalog()
    t = catalog.get(feature, {}).get("type", "")
    return ROLE_BY_TYPE.get(t, "skip")


def _features_all_emitted() -> set[str]:
    """All features emitted by at least one dataset (intersection of catalog and
    feature_overview_* tables, which were built from Stage-1 long output)."""
    out: set[str] = set()
    for cat_name in ("vitals", "labs", "drugs"):
        p = COVERAGE_CSV_DIR / f"feature_overview_{cat_name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0)
        out.update(df.index.tolist())
    return out


def _features_with_min_coverage(min_share: float) -> set[str]:
    """Features with `enc_share` >= min_share in at least one dataset
    (so we keep cross-dataset rare-but-useful features)."""
    out: set[str] = set()
    for cat_name in ("vitals", "labs", "drugs"):
        p = COVERAGE_CSV_DIR / f"feature_overview_{cat_name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0)
        share_cols = [c for c in df.columns if c.endswith("__enc_share")]
        # Max across datasets per feature
        max_share = df[share_cols].max(axis=1)
        out.update(max_share[max_share >= min_share].index.tolist())
    return out


def resolve_features(keyword: str, catalog: dict | None = None) -> list[str]:
    """Resolve a feature-selection keyword into a sorted list of canonical names."""
    if catalog is None:
        catalog = load_catalog()
    keyword = keyword.strip()

    if keyword == "all":
        names = _features_all_emitted()

    elif keyword == "sofa":
        names = {k for k, v in catalog.items() if v.get("sofa_relevant")}

    elif keyword == "core":
        names = {k for k, v in catalog.items() if v.get("tier") == "core"}

    elif keyword.startswith("covered"):
        try:
            pct = float(keyword.removeprefix("covered"))
        except ValueError as e:
            raise ValueError(f"invalid covered<P> keyword: {keyword}") from e
        names = _features_with_min_coverage(pct / 100.0)

    elif keyword.startswith("custom:"):
        path = Path(keyword.removeprefix("custom:")).expanduser()
        names = {ln.strip() for ln in path.read_text().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")}

    else:
        raise ValueError(
            f"unknown --features keyword: {keyword!r}. "
            f"Allowed: all | sofa | core | covered<P> | custom:<path>"
        )

    # Filter out skipped roles (e.g. SOFA-derived) and unknown features
    names = {f for f in names if role_of(f, catalog) != "skip"}
    return sorted(names)


def partition_by_role(features: list[str], catalog: dict | None = None
                      ) -> dict[Role, list[str]]:
    if catalog is None:
        catalog = load_catalog()
    out: dict[Role, list[str]] = {"observation": [], "intervention": [], "static": []}
    for f in features:
        r = role_of(f, catalog)
        if r in out:
            out[r].append(f)
    for k in out:
        out[k].sort()
    return out
