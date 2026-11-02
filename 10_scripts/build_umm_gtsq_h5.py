"""
10_scripts/build_umm_gtsq_h5.py

GTSQ → TSF experiment (UMM only): augment the UMM Stage-3 forecast windows with the
**most recent same-or-earlier-DAY GTSQ assessment** as extra *static* conditioning features.
NOT strictly prior: the GTSQ export carries a `date` with no time of day, so the selection
(`searchsorted(..., side="right") - 1`, latest date <= the window-start DAY) admits an assessment
recorded later on the same calendar day. Measured: **97.7 %** of selected assessments are same-day,
only 2.3 % from a strictly earlier day (§37.68). Requiring strictly earlier would drop coverage to
2.3 % and make the experiment infeasible as designed. Tests the
inverse of the keep-gate "+clinical" probe: does the daily clinician questionnaire
(working diagnosis, infection suspicion, organ-dysfunction items) improve the physiology
forecast?

Join (verified): h5 `stay_id` → `umm_cohort.hospital_admission_id` (= GTSQ
`EncounterNumber`) + `icu_start`; window start = `icu_start + t0_hour`; take the latest
GTSQ row with `date ≤ window-start date`. Items absent / encounter without GTSQ → 0 with
`static_mask = 0` (the model already mask-gates static inputs).

Writes `<stem>_gtsq.h5` (drop-in for train_forecast) with `static` / `static_mask` /
`features_static` extended by the 45 `gtsq_*` items. Everything else is copied verbatim.

Usage:  python 10_scripts/build_umm_gtsq_h5.py            # train + val
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GTSQ_CSV = ROOT / "keepgate" / "gtsq_day_level_encoded.csv"
COHORT   = ROOT / "data" / "long" / "umm_cohort.parquet"


def _decode(arr) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else x for x in arr]


def load_gtsq() -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Return {encounter: (sorted_date_ints, item_matrix)} and the item-name list."""
    g = pd.read_csv(GTSQ_CSV)
    # `version` is the questionnaire FORM VERSION (1.0/2.0), i.e. metadata, not a clinical item.
    # The numeric filter below ("keep any column with a numeric value") would admit it as a
    # conditioning FEATURE, which would let the model read which form era a day belongs to — a
    # site/time proxy, and a confound for an experiment about clinical judgement. Excluded
    # 2026-10-27 (§37.69): doing so also reproduces the stored *_gtsq.h5 exactly (45 items, same
    # order), which a rebuild otherwise would not, since those files predate the column.
    cand = [c for c in g.columns if c not in ("EncounterNumber", "date", "version")]
    num = g[cand].apply(pd.to_numeric, errors="coerce")
    items = [c for c in cand if num[c].notna().any()]            # drop pure-text cols (Supervisor …)
    g[items] = num[items]
    g["d"] = (pd.to_datetime(g["date"]).values.astype("datetime64[D]")
              .astype("int64"))                                  # days since epoch
    g["EncounterNumber"] = g["EncounterNumber"].astype("int64")
    g = g.sort_values(["EncounterNumber", "d"])
    by_enc: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for enc, sub in g.groupby("EncounterNumber"):
        by_enc[int(enc)] = (sub["d"].to_numpy(),
                            sub[items].to_numpy(dtype=np.float32))
    return by_enc, items


#: UMM institutional information (MF, 2026-10-27): the questionnaire lives in a SEPARATE database
#: from ICCA, carries a `date` with NO timestamp, and is filled **around 14:00 each day**. That one
#: fact is what makes a temporally sound selection possible at all, and it is recorded here because
#: it exists nowhere in the data itself.
GTSQ_FILL_HOUR = 14.0

#: "day"       — latest assessment dated <= the window-start DAY. Reproduces every existing
#:               *_gtsq.h5, so it is the default, but 60.0 % of its selections postdate the window
#:               start (§37.68 D41) and are therefore lookahead.
#: "fill_hour" — latest assessment whose assumed time (date + GTSQ_FILL_HOUR) is strictly BEFORE the
#:               window start. Temporally sound, and keeps 16 054 of the 18 726 currently covered
#:               windows (86 % of coverage) — against 434 for a strictly-earlier-day rule.
#: ANY REBUILD SHOULD PASS --gtsq-prior-mode fill_hour. Existing files are deliberately untouched.
GTSQ_PRIOR_MODE_DEFAULT = "day"


def build(h5_in: Path, h5_out: Path, by_enc, items, enc_of, start_of,
          prior_mode: str = GTSQ_PRIOR_MODE_DEFAULT) -> None:
    with h5py.File(h5_in, "r") as f:
        sid = f["stay_id"][:]
        t0  = f["t0_hour"][:]
        stat  = f["static"][:]
        smask = f["static_mask"][:]
        feats_static = _decode(f.attrs["features_static"])

        N, K = len(sid), len(items)
        gv = np.zeros((N, K), dtype=np.float32)
        gm = np.zeros((N, K), dtype=bool)

        # window start date in days-since-epoch
        start_days = np.array([start_of.get(int(s), np.datetime64("NaT"))
                               for s in sid], dtype="datetime64[ns]")
        _wstart = start_days + (t0.astype("timedelta64[h]"))
        wdate = _wstart.astype("datetime64[D]").astype("int64")  # NaT → large negative
        # window start in hours since epoch, for the fill-hour comparison
        whour = (_wstart - np.datetime64("1970-01-01")) / np.timedelta64(1, "h")

        n_cov = 0
        for i in range(N):
            enc = enc_of.get(int(sid[i]))
            if enc is None or enc not in by_enc:
                continue
            dates, mat = by_enc[enc]
            if prior_mode == "fill_hour":
                # assume each assessment happened at date + GTSQ_FILL_HOUR; take the latest one
                # STRICTLY BEFORE the window start, so nothing can postdate the window.
                eff = dates * 24.0 + GTSQ_FILL_HOUR
                j = np.searchsorted(eff, whour[i], side="right") - 1
            else:
                j = np.searchsorted(dates, wdate[i], side="right") - 1   # latest date ≤ window DAY
            if j < 0:
                continue
            row = mat[j]
            obs = ~np.isnan(row)
            gv[i, obs] = row[obs]
            gm[i, obs] = True
            n_cov += 1

        # has_gtsq flag: 1 where a prior GTSQ exists (always "observed"), so the encoder
        # can distinguish "no assessment" from a genuine 0-valued item.
        has  = gm.any(axis=1, keepdims=True).astype(np.float32)       # (N, 1)
        hmask = np.ones((N, 1), dtype=bool)

        with h5py.File(h5_out, "w") as o:
            for k in f.keys():
                if k == "static":
                    o.create_dataset("static", data=np.concatenate([stat, gv, has], axis=1))
                elif k == "static_mask":
                    o.create_dataset("static_mask",
                                     data=np.concatenate([smask, gm, hmask], axis=1))
                else:
                    o.create_dataset(k, data=f[k][:])
            for ak, av in f.attrs.items():
                o.attrs[ak] = av
            o.attrs["features_static"] = (list(feats_static)
                                          + [f"gtsq_{c}" for c in items] + ["has_gtsq"])

    print(f"  {h5_out.name}: {N} windows, {n_cov} ({100*n_cov/N:.0f}%) with a prior GTSQ; "
          f"F_static {len(feats_static)} → {len(feats_static)+K}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="UMM GTSQ -> static conditioning on the Stage-3 windows")
    ap.add_argument("--gtsq-prior-mode", choices=["day", "fill_hour"],
                    default=GTSQ_PRIOR_MODE_DEFAULT,
                    help="'day' (default) reproduces the existing *_gtsq.h5 but 60%% of its "
                         "selections postdate the window start; 'fill_hour' assumes the documented "
                         "~14:00 fill time and takes only assessments strictly before the window "
                         "start (sound; keeps 86%% of coverage). See §37.68/§37.69.")
    args = ap.parse_args()
    if args.gtsq_prior_mode == "day":
        print("WARNING: --gtsq-prior-mode day -> the 'prior' GTSQ assessment is same-DAY for 97.7% "
              "of windows and actually POSTDATES the window start for 60.0% of them, so it is "
              "lookahead (§37.68 D41). This default exists only to reproduce the existing files. "
              "Pass --gtsq-prior-mode fill_hour for a sound rebuild.", flush=True)
    by_enc, items = load_gtsq()
    coh = pd.read_parquet(COHORT)
    coh["hospital_admission_id"] = pd.to_numeric(coh["hospital_admission_id"],
                                                 errors="coerce")
    enc_of = {int(r.stay_id): int(r.hospital_admission_id)
              for r in coh.dropna(subset=["hospital_admission_id"]).itertuples()}
    start_of = {int(r.stay_id): np.datetime64(pd.Timestamp(r.icu_start))
                for r in coh.dropna(subset=["icu_start"]).itertuples()}
    print(f"GTSQ: {len(by_enc)} encounters, {len(items)} items")

    for split in ["train", "val", "test"]:
        h5_in = ROOT / "data" / "dataset" / f"umm_seed42_{split}__all_4h_fwd.h5"
        h5_out = h5_in.with_name(h5_in.stem + "_gtsq.h5")
        build(h5_in, h5_out, by_enc, items, enc_of, start_of,
              prior_mode=args.gtsq_prior_mode)


if __name__ == "__main__":
    main()
