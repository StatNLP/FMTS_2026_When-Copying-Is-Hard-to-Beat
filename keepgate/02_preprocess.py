#!/usr/bin/env python3
"""GTSQ next-day prediction preprocessing (UMM Ground-Truth Sepsis Questionnaire).

Companion to 02b_preprocess_mimic.py — identical npz layout, so 03_train.py,
05_baselines.py, 06_data_analysis.py and 07_aggregate.py run unchanged. Predicts day
D+1's 45 GTSQ items from day D ("noclin", the headline) and optionally a
backward-anchored clinical window ("+clinical").

  Input  : gtsq_day_level_encoded.csv  — one row per encounter-day, 45 numeric features
           (produced by 01_preprocess_gtsq.py from the raw UMM GTSQ export).
  Output : <save_dir>/{train,val,test}.npz + norm_stats.npz, keys identical to 02b:
             clinical (N, seq_len, V*2), gtsq_in/_mask (N[,K],G), gtsq_out/_mask (N,G),
             enc_ids, dates.

Pipeline mirrors 02b: build_pairs (D->D+1, K-day history) -> ENCOUNTER-level split (by
EncounterNumber; `--split_by patient` + `--patient_map` switches to patient-level) -> GTSQ z-norm
on TRAIN observed only -> npz.
The shipped splits are ENCOUNTER-level by design (MF, 2026-10-27). They are encounter-disjoint but
NOT patient-disjoint: measured on gtsq_pred_24h_gap6h_noclin, 24 patients span train/test (4.4 % of
the 549 test patients), 16 train/val, 7 val/test (§37.73 D43). Report it as an encounter-level
split, not a patient-level one. Clinical (optional) is
z-normed on TRAIN observed only and winsorised to +/-CLINICAL_CLIP, exactly like 02b.

Note: the encoded CSV is already numeric — the text->code mapping lives in
01_preprocess_gtsq.py (kept out of this script, which only pairs/splits/normalises).
"""

import argparse
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# GTSQ 45-feature schema  (name -> (ftype, nclass)); used for ftypes/nclasses only.
# ---------------------------------------------------------------------------
ORGANS = ["Lunge", "Niere", "Gehirn", "Herz", "Leber",
          "Gastrointestinal", "Gerinnungssystem", "Knochenmark"]
LOKALISATIONS = [
    "abdominell", "Endokarditis", "Blood-stream-infection", "Haut",
    "intrakraniell / meningeal", "katheterassoziiert", "thorakal",
    "urogenital", "Gelenke / Knochen",
]


def _gtsq_feature_types():
    """The 45 GTSQ targets and their (ftype, nclass). Knochenmark has no 'chronisch'
    in the encoded CSV; the extra key here is simply never matched (features are read
    from the CSV columns, types looked up from this map)."""
    t = {}
    for organ in ORGANS:
        t[f'i11100_{organ}_present'] = ('binary', 2)
        t[f'i11100_{organ}_chronisch'] = ('binary', 2)
        t[f'i11100_{organ}_akut oder "neu"'] = ('ordinal', 3)
        t[f'i11100_{organ}_Ursache für Dysfunktion'] = ('ordinal', 4)
    for loc in LOKALISATIONS:
        t[f'i05100_Lokalisation__{loc}'] = ('binary', 2)
    t['i09_Capillary_Leak'] = ('binary', 2)
    t['i03000_Arbeitsdiagnose'] = ('ordinal', 5)
    t['i02000_Klinisches_Gesamtbild'] = ('ordinal', 3)
    t['i04000_Infektionsverdacht'] = ('ordinal', 3)
    t['i01000_Gruppe_(0|1|2)'] = ('ordinal', 3)
    return t


GTSQ_FEATURE_TYPES = _gtsq_feature_types()
# The SOFA components are 0-4 ORDINAL scores, not continuous measurements. They are not questionnaire
# items, so the lookup above does not know them and they would default to ('continuous', 0) -- which
# trains a regressor, emits only MSE, and leaves every classification metric and baseline undefined.
# Declaring them here is what makes the UMM SOFA target comparable to the MIMIC one, which scores the
# same six components as ordinal classes.
GTSQ_FEATURE_TYPES.update({f"sofa_{k}": ("ordinal", 5) for k in
                           ("respiration", "coagulation", "liver", "cardiovascular", "cns", "renal")})
# NEVER targets. i01000_Station / i01000_Bett are ward context (carried since 2026-07-28 for the
# ward-ranking analysis); they MUST be listed here or they are silently picked up as two extra
# "features", typed continuous with nclasses=0, which yields a 47-feature split and crashes the
# classification baselines (baseline_markov: IndexError on an empty transition table).
GTSQ_META_COLS = {'EncounterNumber', 'date', 'i01000_Supervisor', 'i01000_Station',
                  'i01000_Bett', 'version'}
GTSQ_HOUR = 14          # GTSQ is system-stamped at 14:00
# A DAY-VALUED target (SOFA and friends) is an aggregate over a whole calendar/ICU day, so its window
# must end on a day boundary; the questionnaire is stamped at a time of day and its 6 h gap is
# measured from that stamp, which is deliberate. Set by the caller via --day_target.
_DAY_TARGET = False
CLINICAL_CLIP = 10.0    # winsorise clinical z-scores (guard against lab/med outliers)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_gtsq(csv_path):
    """Read the pre-encoded day-level CSV. Returns (df, feature_list)."""
    df = pd.read_csv(csv_path, low_memory=False)
    df['EncounterNumber'] = df['EncounterNumber'].astype(str)
    df['date'] = pd.to_datetime(df['date']).dt.date
    features = [c for c in df.columns if c not in GTSQ_META_COLS]
    for c in features:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)
    keep = ['EncounterNumber', 'date'] + features
    if SUP_COL in df.columns:                       # rater identity (input-only, never a target)
        df[SUP_COL] = df[SUP_COL].astype(str).str.strip().str.capitalize()
        keep = keep + [SUP_COL]
    return df[keep].copy(), features


SUP_COL = 'i01000_Supervisor'   # attending who filed that day's questionnaire
SUP_VOCAB = {}                  # name -> int, fitted on train in main(); 0 = unknown

HORIZON_H = 168   # umm_long.parquet hours-since-intime are capped at 7 ICU days


# ---------------------------------------------------------------------------
# Day-pair construction (predict day D+1 from day D; K days of history)
# ---------------------------------------------------------------------------
def build_pairs(df, features, K, min_targets, horizon=1):
    """One pair per encounter-day D that has a D+horizon (horizon=1 is next-day, the
    headline; horizon>1 is the granularity/volatility sweep — longer horizon = more
    accumulated change = weaker persistence). Mirrors 02b.build_pairs."""
    G = len(features)
    pairs = []
    for enc, grp in df.sort_values(['EncounterNumber', 'date']).groupby('EncounterNumber'):
        dates = list(grp['date'].values)
        vals = grp[features].values.astype(np.float32)        # (n_days, G)
        sup_vals = grp[SUP_COL].values if SUP_COL in grp.columns else None
        row_of = {d: i for i, d in enumerate(dates)}
        for i, d in enumerate(dates):
            if (d + timedelta(days=horizon)) not in row_of:
                continue
            out = vals[row_of[d + timedelta(days=horizon)]].copy()
            out_m = (~np.isnan(out)).astype(np.float32)
            np.nan_to_num(out, copy=False, nan=0.0)
            if (~np.isnan(vals[i])).sum() < min_targets or out_m.sum() < min_targets:
                continue
            hist_v, hist_m = [], []
            for k in range(K - 1, -1, -1):                    # oldest .. D
                dk = d - timedelta(days=k)
                if dk in row_of:
                    v = vals[row_of[dk]].copy()
                    m = (~np.isnan(v)).astype(np.float32)
                    np.nan_to_num(v, copy=False, nan=0.0)
                else:
                    v = np.zeros(G, np.float32); m = np.zeros(G, np.float32)
                hist_v.append(v); hist_m.append(m)
            gin = hist_v[0] if K == 1 else np.stack(hist_v)
            gin_m = hist_m[0] if K == 1 else np.stack(hist_m)
            sup_d = sup_n = ''
            if sup_vals is not None:
                sup_d = str(sup_vals[i])
                sup_n = str(sup_vals[row_of[d + timedelta(days=horizon)]])
            pairs.append(dict(enc=enc, date_in=d, gin=gin, gin_m=gin_m,
                              gout=out, gout_m=out_m, day_idx=i,       # i = episode-relative day (0-based)
                              sup_d=sup_d, sup_next=sup_n))
    return pairs


CULTURE_PARQUET = "raw_data/UMM_Export/260528_UMM_blood_cultures.parquet"   # legacy (blood only)
# Full microbiology export (2026-07-27): all specimen types -> organ-mapped culture features.
MICROBIOLOGY_CSV = "raw_data/UMM_Export/260727_UMM_microbiology.csv"
MICROBIOLOGY_PARQUET = "raw_data/UMM_Export/260727_UMM_microbiology.parquet"
_MICRO_COLS = ["encounterId", "encounterNumber", "material_note", "order_time", "collection_time",
               "arrival_time", "station_name", "specimen_status", "result_time", "result_type",
               "result_value", "result_status"]
# material_note substring -> organ category (first match wins; specific before generic). Maps to the
# GTSQ organ/localisation items: cns->Gehirn/intrakraniell, kidney->Niere/urogenital, lung->Lunge/
# thorakal, abdomen->GI/abdominell, line->katheter, skin->Haut, blood->systemic (sepsis). Anus/NRR/
# nasopharyngeal swabs are COLONISATION SCREENING (MRSA/VRE surveillance), not infection -> flagged
# separately so the model can down-weight them.
_ORGAN_MAP = [("liquor", "cns"), ("urin", "kidney"), ("tracheal|bronchial|bal|sputum", "lung"),
              ("pleura", "thorax"), ("aszites|punktat|drainage", "abdomen"),
              ("anus|nrr|nasen|rachen|rektal", "surveillance"), ("katheter|zvk", "line"),
              ("wund", "skin"), ("blutkultur", "blood"), ("abstrich", "swab")]


def _culture_rows(repo_root, by_hadm):
    """Microbiology events as harmonised long rows (stay_id, hour, variable, value). Each 0/1 signal
    is placed at the hour it BECAME KNOWN (leakage discipline). Richer set than drawn/positive:

    COLLECTION time (the physician's suspicion, known immediately -- what was ordered and from where):
      culture_drawn                                             a culture was taken
      culture_site_{csf,abdominal,catheter}                     specimen SITE from material_note ->
          organ hint: Liquor->brain, Aszites/Punktat->abdomen, zentral/arteriell/katheter->line.
          (This export is blood-dominated; urine/respiratory/wound are ABSENT, so kidney/lung/skin
          cannot be culture-mapped here -- would need a broader micro export.)
    ARRIVAL time (each row placed at its OWN arrival_time -- the report coming back, ~123 h later):
      culture_result_returned                                   any result returned
      culture_pathogen / culture_contaminant / culture_fungal   organism class on Keimidentifikation:
          contaminant = coag-neg staph (sta* except staaur -- the #1 'positive' is S. epidermidis, a
          skin contaminant, so the old single culture_positive conflated bacteraemia with contamination);
          fungal = Candida (can*) / yeast; pathogen = any other identified organism.
      culture_grampos / culture_gramneg                         gram stain (Mikroskopie): +k/+s/str ->
          gram-positive, -s/-k -> gram-negative, spro* -> yeast (folded into fungal). Placed at arrival
          (conservative: gram is usually known earlier, but we have no reliable gram timestamp).

    `result_time` is NOT used (collection->result delta median 0.0 h -> fake timestamp)."""
    import re
    pq = repo_root / MICROBIOLOGY_PARQUET
    if pq.exists():                                     # cached (built from the CSV once, below)
        c = pd.read_parquet(pq)
    else:
        csvp = repo_root / MICROBIOLOGY_CSV
        if not csvp.exists():
            print(f"--use_cultures: {csvp} not found, skipping"); return pd.DataFrame()
        import csv as _csv                              # the export is '"'-delimited, no header
        c = pd.read_csv(csvp, sep='"', quoting=_csv.QUOTE_NONE, skiprows=1, names=_MICRO_COLS,
                        header=None, low_memory=False, on_bad_lines="skip")
        c = c[c["encounterId"].astype(str).str.match(r"^\d+$", na=False)].copy()
        c.to_parquet(pq); print(f"--use_cultures: cached microbiology -> {pq.name}")
    c = c[c["encounterNumber"].astype(str).str.match(r"^\d+$", na=False)].copy()
    c["hadm"] = c["encounterNumber"].astype(str)
    c = c[c["hadm"].isin(by_hadm.keys())]
    for t in ("collection_time", "arrival_time"):
        c[t] = pd.to_datetime(c[t], errors="coerce")
    c = c.dropna(subset=["collection_time"])
    rv = c["result_value"].astype(str).str.lower(); rt = c["result_type"].astype(str)
    mn = c["material_note"].astype(str).str.lower()
    keim = rt.eq("Keimidentifikation"); micro = rt.str.contains("Mikroskopie|Gram", case=False, na=False)
    c["contaminant"] = keim & rv.str.startswith("sta") & ~rv.str.startswith("staaur")
    c["fungal"] = (keim & rv.str.startswith("can")) | (micro & rv.str.startswith("spro"))
    c["pathogen"] = keim & ~c["contaminant"] & ~c["fungal"]
    c["grampos"] = micro & rv.isin(["+k", "+s", "str"])
    c["gramneg"] = micro & rv.isin(["-s", "-k"])
    # organ category from material_note (first ORGAN_MAP match; specific before generic)
    _pats = [(re.compile(p), cat) for p, cat in _ORGAN_MAP]

    def _cat(x):
        for rgx, cat in _pats:
            if rgx.search(x):
                return cat
        return "other"
    c["organ"] = mn.map(_cat)

    def _locate(hadm, ts):
        """(stay_id, hours-since-icu_start) for the stay covering ts -- same rule as the pairs."""
        stays = by_hadm.get(hadm)
        if not stays or pd.isna(ts):
            return None
        cand = [s for s in stays if s[1] <= ts <= s[2] + pd.Timedelta(days=1)]
        s = (min(cand, key=lambda x: x[1]) if cand else
             (max([x for x in stays if x[1] <= ts], key=lambda x: x[1])
              if any(x[1] <= ts for x in stays) else None))
        if s is None:
            return None
        return s[0], (ts - s[1]).total_seconds() / 3600.0

    rows = []
    ARR = ["pathogen", "contaminant", "fungal", "grampos", "gramneg"]
    # collection-time signals: one event per (encounter, collection_time) -> global drawn + organ site
    for (hadm, ct), g in c.groupby(["hadm", "collection_time"]):
        d = _locate(hadm, ct)
        if not d:
            continue
        rows.append((d[0], d[1], "culture_drawn", 1.0))
        cat = g["organ"].iloc[0]                        # one specimen per collection event
        if cat != "other":
            rows.append((d[0], d[1], f"culture_site_{cat}", 1.0))   # organ-mapped suspicion
    # arrival-time signals: each result row placed at its OWN arrival_time
    res = c[c["arrival_time"].notna() & (keim | micro)]
    for r in res.itertuples(index=False):
        a = _locate(r.hadm, r.arrival_time)
        if not a:
            continue
        rows.append((a[0], a[1], "culture_result_returned", 1.0))
        for s in ARR:
            if getattr(r, s):
                rows.append((a[0], a[1], f"culture_{s}", 1.0))
    out = pd.DataFrame(rows, columns=["stay_id", "hour", "variable", "value"]).drop_duplicates()
    print(f"--use_cultures: {out['variable'].nunique()} culture variables, {len(out):,} long rows "
          f"({out['variable'].value_counts().to_dict() if len(out) else '{}'})")
    return out


def build_clinical_windows(pairs, repo_root, seq_len, gap_hours, clinical_source, horizon=1,
                           use_cultures=False):
    """(N, V, seq_len, 2) [value, mask] from the harmonised UMM clinical (data/long/
    umm_long.parquet), backward-anchored from the GTSQ target. Returns (X, variables).

    Linkage: GTSQ EncounterNumber == hospital_admission_id (umm_cohort) -> ICU stay(s).
    For each pair the stay covering day D+1 is chosen; the GTSQ target time
    (date_D+1 @ GTSQ_HOUR) is converted to hours-since-icu_start, and the window
    [target_hour-gap-seq_len, target_hour-gap) is read from umm_long's `hour` axis
    (capped at HORIZON_H). No t+1 leakage. Aligned to `pairs` order.

    Caveats: ~84% of GTSQ encounters link to a clinical stay; umm_long is capped at 7
    ICU days so day-pairs later in the stay get no clinical window; the variable set is
    the harmonised UMM panel (vitals/labs/vent/meds), not the original R20_combined.
    """
    from mimic_targets import CLINICAL_TYPES
    coh = pd.read_parquet(repo_root / "data" / "long" / "umm_cohort.parquet",
                          columns=["stay_id", "hospital_admission_id", "icu_start", "icu_end"])
    coh["hadm"] = coh["hospital_admission_id"].astype(str)
    coh["icu_start"] = pd.to_datetime(coh["icu_start"])
    coh["icu_end"] = pd.to_datetime(coh["icu_end"])
    hadm_set = {str(p["enc"]) for p in pairs}
    coh = coh[coh["hadm"].isin(hadm_set)]
    by_hadm = {}
    for r in coh.itertuples(index=False):
        by_hadm.setdefault(r.hadm, []).append((r.stay_id, r.icu_start, r.icu_end))

    # Clinical source is chosen EXPLICITLY (no auto-detect) so the split's provenance is
    # unambiguous: 'capped' = umm_long.parquet (7-day horizon, Paper-1-consistent);
    # 'complete' = umm_long_complete.parquet (full LOS, from run_umm_complete.sh).
    long_name = {"capped": "umm_long.parquet",
                 "complete": "umm_long_complete.parquet"}[clinical_source]
    long_path = repo_root / "data" / "long" / long_name
    if not long_path.exists():
        raise SystemExit(f"--clinical_source {clinical_source}: {long_path} not found "
                         f"(run jobs/run_umm_complete.sh for the 'complete' source).")
    print(f"clinical source: {clinical_source} ({long_path.name})")
    clin = pd.read_parquet(
        long_path,
        columns=["stay_id", "hour", "variable", "value", "measurement_type"],
        filters=[("measurement_type", "in", list(CLINICAL_TYPES))])
    clin = clin[clin["stay_id"].isin(set(coh["stay_id"]))]
    if use_cultures:
        cult = _culture_rows(repo_root, by_hadm)
        if len(cult):
            cult = cult[cult["stay_id"].isin(set(coh["stay_id"]))]
            clin = pd.concat([clin[["stay_id", "hour", "variable", "value"]], cult],
                             ignore_index=True)
    variables = sorted(clin["variable"].unique())
    var_to_idx = {v: i for i, v in enumerate(variables)}
    clin["var_idx"] = clin["variable"].map(var_to_idx).astype(int)
    by_stay = {sid: g[["hour", "var_idx", "value"]].to_numpy()
               for sid, g in clin.groupby("stay_id")}

    V = len(variables); N = len(pairs)
    X = np.zeros((N, V, seq_len, 2), dtype=np.float32)
    for pi, p in enumerate(pairs):
        stays = by_hadm.get(str(p["enc"]))
        if not stays:
            continue
        target = pd.Timestamp(p["date_in"] + timedelta(days=horizon)) + pd.Timedelta(hours=GTSQ_HOUR)
        cand = [s for s in stays if s[1] <= target <= s[2] + pd.Timedelta(days=1)]
        if cand:
            sid, icu_start, _ = min(cand, key=lambda s: s[1])
        else:
            before = [s for s in stays if s[1] <= target]
            if not before:
                continue
            sid, icu_start, _ = max(before, key=lambda s: s[1])
        arr = by_stay.get(sid)
        if arr is None:
            continue
        th = (target - icu_start).total_seconds() / 3600.0
        w_end = th - gap_hours; w_start = w_end - seq_len
        # `target` is stamped at GTSQ_HOUR, so w_end lands on the target DAY BOUNDARY only when
        # gap_hours == GTSQ_HOUR, and on the end of the target day (the oracle) only when
        # gap_hours == GTSQ_HOUR - 24. For a day-valued target such as SOFA that has no form time,
        # any other value silently slices into the target day: at GTSQ_HOUR=14, gap=0 leaks 14 h of
        # it and gap=6 leaks 8 h, with nothing in the output to show it. The check below refuses the
        # leak rather than trusting the caller to have done the arithmetic.
        if _DAY_TARGET and gap_hours not in (GTSQ_HOUR, GTSQ_HOUR - 24):
            raise SystemExit(
                f"ERROR: gap_hours={gap_hours} does not put the window on a day boundary "
                f"(GTSQ_HOUR={GTSQ_HOUR}). It would slice {abs(GTSQ_HOUR - gap_hours) % 24}h across "
                f"it. A day-valued target admits exactly two values: gap_hours={GTSQ_HOUR} "
                f"(causal -- window ends where the target day begins) or "
                f"gap_hours={GTSQ_HOUR - 24} (oracle -- window IS the target day).")
        hr = arr[:, 0]
        sel = (hr >= w_start) & (hr < w_end)
        if not sel.any():
            continue
        vi = arr[sel, 1].astype(int)
        hb = np.clip(np.floor(hr[sel] - w_start).astype(int), 0, seq_len - 1)
        # One value per (variable, hour) cell: if several observations of the same variable fall in
        # one hour bin, numpy fancy assignment keeps the LAST in `arr` order (arr comes from the
        # long table sorted by hour, so this is last-in-hour). Only affects +clinical runs.
        X[pi, vi, hb, 0] = arr[sel, 2]
        X[pi, vi, hb, 1] = 1.0
    return X, variables


def summarise_windows(X, variables, slots=(1, 4, 6, 12, 24)):
    """Replace the hourly clinical window (N,V,T,2) with a multi-statistic summary: per variable,
    {min,max,mean,median} over the most-recent {slots} hours -> a (N, 4V, len(slots), 2) pseudo-
    sequence (4V summary-variables x len(slots) 'slot' timesteps; [value, has-observation]). The
    daily worst-of-window (max/min) that clinical scores use is captured directly. Downstream
    normalisation / reshape / encoder consume it unchanged (V'=4V, seq_len'=len(slots))."""
    N, V, T, _ = X.shape
    stats = ("min", "max", "mean", "median")
    out = np.zeros((N, V * 4, len(slots), 2), dtype=np.float32)
    val = X[:, :, :, 0]; msk = X[:, :, :, 1] == 1
    for si, k in enumerate(slots):
        sl = slice(max(0, T - k), T)                       # the most-recent k hours
        m = msk[:, :, sl]; any_obs = m.any(axis=2)         # (N, V)
        vv = np.where(m, val[:, :, sl], np.nan)
        import warnings
        with warnings.catch_warnings():                    # all-NaN slices (unobserved) -> handled by mask
            warnings.simplefilter("ignore", category=RuntimeWarning)
            agg = (np.nanmin(vv, 2), np.nanmax(vv, 2), np.nanmean(vv, 2), np.nanmedian(vv, 2))
        for ti, a in enumerate(agg):
            out[:, ti::4, si, 0] = np.where(any_obs, np.nan_to_num(a), 0.0)
            out[:, ti::4, si, 1] = any_obs.astype(np.float32)
    names = [f"{vn}_{st}" for vn in variables for st in stats]
    return out, names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_patient_map(path):
    """EncounterNumber -> lifeTimeNumber from the UMM demographics export.

    `lifeTimeNumber` is the hospital's person-level identifier; `encounterNumber` is one admission.
    The export carries duplicate encounter rows (10,736 rows / 9,460 encounters), so the mapping is
    de-duplicated and any encounter whose rows disagree is reported rather than silently resolved.
    """
    d = pd.read_csv(path, sep=';', skiprows=[0, 2], na_values=['NULL'], low_memory=False,
                    usecols=['encounterNumber', 'lifeTimeNumber'])
    d = d.dropna(subset=['encounterNumber', 'lifeTimeNumber'])
    d['enc'] = d['encounterNumber'].astype('Int64').astype(str)
    d['pat'] = d['lifeTimeNumber'].astype('Int64').astype(str)
    g = d.groupby('enc')['pat'].nunique()
    if (g > 1).any():
        print(f"  WARNING: {(g > 1).sum()} encounters map to >1 lifeTimeNumber; taking the first")
    m = d.drop_duplicates('enc').set_index('enc')['pat'].to_dict()
    print(f"  patient map: {len(m)} encounters -> {len(set(m.values()))} patients ({os.path.basename(path)})")
    return m


def main(args):
    # The window builder consults this at module scope; set it before any window is built.
    globals()["_DAY_TARGET"] = bool(getattr(args, "day_target", False))
    K = args.gtsq_history
    seq_len = args.seq_len

    df, features = load_gtsq(args.gtsq_csv)
    G = len(features)
    ftypes = np.array([GTSQ_FEATURE_TYPES.get(f, ('continuous', 0))[0] for f in features])
    nclasses = np.array([GTSQ_FEATURE_TYPES.get(f, ('continuous', 0))[1] for f in features],
                        dtype=np.int32)
    print(f"GTSQ features: {G}  (records: {len(df)}, encounters: {df['EncounterNumber'].nunique()})")
    print(f"  types: {(ftypes=='binary').sum()} binary, {(ftypes=='ordinal').sum()} ordinal")

    pairs = build_pairs(df, features, K, args.min_gtsq_values, args.horizon)
    print(f"Day-pairs: {len(pairs)}  (history K={K}, horizon={args.horizon})")
    if not pairs:
        raise SystemExit("No day-pairs — check gtsq_csv / min_gtsq_values.")
    for i, p in enumerate(pairs):
        p["idx"] = i

    # --- split ---
    # Encounter vs PATIENT. The historical default splits by EncounterNumber, which is NOT
    # patient-disjoint: the UMM export links encounters to a person through `lifeTimeNumber`, and
    # 24 patients had encounters in both train and test under the encounter split. A claim about
    # unseen patients requires grouping every encounter of one person into the same fold, which is
    # what --split_by patient does. `pat_ids` is written to every npz either way, so downstream
    # evaluation can cluster by patient even for splits that were not built that way.
    enc2pat = load_patient_map(args.patient_map) if args.split_by == "patient" or args.patient_map else {}
    all_encs = sorted({p["enc"] for p in pairs})
    if args.split_by == "patient":
        missing = [e for e in all_encs if e not in enc2pat]
        if missing:
            # An unmapped encounter becomes its own group: it cannot leak to another fold, and
            # silently dropping it would change the cohort behind the reader's back.
            print(f"  WARNING: {len(missing)}/{len(all_encs)} encounters have no lifeTimeNumber; "
                  f"each is treated as its own patient group")
        grp_of = {e: str(enc2pat.get(e, f"enc:{e}")) for e in all_encs}
    else:
        grp_of = {e: e for e in all_encs}
    encs_of_grp = {}
    for e in all_encs:
        encs_of_grp.setdefault(grp_of[e], []).append(e)

    if args.split_mode == "temporal":
        # Chronological holdout: an encounter is assigned by the date of its FIRST day-pair, so no
        # encounter straddles a boundary and no future day informs a past one. Answers a different
        # question from the patient-level split -- whether the model survives distribution shift over
        # time (case-mix, staffing, documentation habit) rather than merely unseen patients.
        # STRICT chronological separation. Assigning a group by the date of its FIRST day-pair is
        # NOT enough once groups are patients: a patient first seen in 2016 and readmitted in 2021
        # carries 2021 data into the training fold, and the folds then overlap in time by years
        # while remaining patient-disjoint. That is what happened here before this fix -- train ran
        # to 2022-01-12 while test began 2020-12-14.
        #
        # Instead: pick two date cutoffs from the distribution of pair dates, and require EVERY pair
        # of a group to fall on one side. A group whose pairs straddle a cutoff is dropped, and the
        # number dropped is reported rather than hidden, because it is a real cost of the design.
        span = {}
        for p in pairs:
            d = str(p.get("date_in", "")); g = grp_of[p["enc"]]
            lo, hi = span.get(g, (d, d))
            span[g] = (min(lo, d), max(hi, d))
        alld = sorted(str(p.get("date_in", "")) for p in pairs)
        t1 = alld[int(len(alld) * (1 - args.test_size) * (1 - args.val_size))]
        t2 = alld[int(len(alld) * (1 - args.test_size))]
        gsplit, dropped = {}, 0
        for g, (lo, hi) in span.items():
            if hi < t1:
                gsplit[g] = "train"
            elif lo >= t1 and hi < t2:
                gsplit[g] = "val"
            elif lo >= t2:
                gsplit[g] = "test"
            else:
                dropped += 1                      # straddles a cutoff: cannot be placed
        pairs = [p for p in pairs if grp_of[p["enc"]] in gsplit]
        # `idx` was assigned over the UNFILTERED list and is used to index the clinical tensor,
        # which is built from the filtered one; without re-indexing the two disagree and the
        # clinical lookup runs off the end.
        for i, p in enumerate(pairs):
            p["idx"] = i
        print(f"  temporal split (strict): train < {t1} <= val < {t2} <= test; "
              f"{dropped} of {len(span)} groups dropped for straddling a cutoff")
    else:
        grps = sorted(encs_of_grp)
        tr_g, te_g = train_test_split(grps, test_size=args.test_size, random_state=args.random_state)
        tr_g, va_g = train_test_split(tr_g, test_size=args.val_size, random_state=args.random_state)
        gsplit = {g: "train" for g in tr_g}
        gsplit.update({g: "val" for g in va_g}); gsplit.update({g: "test" for g in te_g})
    split_of = {e: gsplit[g] for g, es in encs_of_grp.items() if g in gsplit for e in es}
    splits = {s: [p for p in pairs if split_of[p["enc"]] == s] for s in ("train", "val", "test")}
    for s in splits:
        e_s = {p["enc"] for p in splits[s]}
        pt = {grp_of[e] for e in e_s} if args.split_by == "patient" else set()
        print(f"  {s}: {len(splits[s])} pairs, {len(e_s)} encounters"
              + (f", {len(pt)} patients" if pt else ""))
    # Disjointness audit, the check O01 asks for. NOTE (2026-10-27, §37.73 D43): it can only run
    # when `enc2pat` is populated, i.e. with --split_by patient or an explicit --patient_map. Under
    # the DEFAULT (--split_by encounter, no map) enc2pat is {} and the audit was silently skipped,
    # so the encounter split's patient leak was never printed. Measured on
    # gtsq_pred_24h_gap6h_noclin via the UMM cohort's own encounter->patient map: encounters are
    # disjoint (train/test overlap 0) but PATIENTS are not -- 24 span train/test (4.4 % of the 549
    # test patients), 16 train/val, 7 val/test. The `else` branch below now says so out loud.
    if enc2pat:
        pats = {s: {str(enc2pat.get(p["enc"], f"enc:{p['enc']}")) for p in splits[s]}
                for s in splits}
        for a_, b_ in (("train", "test"), ("train", "val"), ("val", "test")):
            ov = pats[a_] & pats[b_]
            print(f"  patient overlap {a_}/{b_}: {len(ov)}"
                  + ("  <-- NOT patient-disjoint" if ov else ""))
    else:
        print("  WARNING: patient-disjointness NOT audited -- no encounter->patient map loaded "
              "(--split_by encounter without --patient_map). The split is ENCOUNTER-disjoint only; "
              "a patient with two encounters can appear in train and test. Measured for the "
              "shipped GTSQ split: 24 patients span train/test (§37.73 D43). Pass --split_by "
              "patient (or --patient_map) to split on patients and enable this audit.")

    # Supervisor vocabulary: fitted on TRAIN only; index 0 is reserved for unknown/unseen
    # raters so a val/test attending absent from train degrades gracefully instead of leaking.
    global SUP_VOCAB
    sups = sorted({p.get("sup_d", "") for p in splits["train"]} |
                  {p.get("sup_next", "") for p in splits["train"]})
    sups = [x for x in sups if x and x.lower() not in ("nan", "none", "")]
    SUP_VOCAB = {name: i + 1 for i, name in enumerate(sups)}
    print(f"  supervisors (train vocab, 0=unknown): {len(SUP_VOCAB)} -> {list(SUP_VOCAB)}")

    # --- optional clinical windows ---
    use_clin = bool(args.use_clinical)
    clin_X = None; clin_vars = []
    clin_means = np.zeros(0, np.float32); clin_stds = np.ones(0, np.float32)
    if use_clin:
        from pathlib import Path
        from mimic_targets import find_repo_root
        repo_root = Path(args.repo_root) if getattr(args, "repo_root", None) else find_repo_root()
        clin_X, clin_vars = build_clinical_windows(pairs, repo_root, seq_len, args.gap_hours,
                                                   args.clinical_source, args.horizon,
                                                   use_cultures=getattr(args, "use_cultures", False))
        if getattr(args, "clinical_repr", "hourly") == "summary":
            clin_X, clin_vars = summarise_windows(clin_X, clin_vars)
            print(f"clinical summary repr: {len(clin_vars)} summary-vars x {clin_X.shape[2]} slots")
        Vc = len(clin_vars)
        Xtr = clin_X[[p["idx"] for p in splits["train"]]]
        clin_means = np.zeros(Vc, np.float32); clin_stds = np.ones(Vc, np.float32)
        for v in range(Vc):
            obs = Xtr[:, v, :, 0][Xtr[:, v, :, 1] == 1]
            if obs.size > 1:
                clin_means[v] = obs.mean(); clin_stds[v] = max(obs.std(), 1e-8)
        print(f"Clinical: V={Vc} variables, train coverage {Xtr[:, :, :, 1].mean():.4f}")

    # --- GTSQ z-norm on TRAIN observed only ---
    def stack(split, key):
        return np.array([p[key] for p in splits[split]], dtype=np.float32)

    gin_tr = stack("train", "gin"); ginm_tr = stack("train", "gin_m")
    means = np.zeros(G, np.float32); stds = np.ones(G, np.float32)
    for g in range(G):
        obs = (gin_tr[:, :, g][ginm_tr[:, :, g] == 1] if K > 1
               else gin_tr[:, g][ginm_tr[:, g] == 1])
        if obs.size > 1:
            means[g] = obs.mean(); stds[g] = max(obs.std(), 1e-8)

    os.makedirs(args.save_dir, exist_ok=True)
    for s in ("train", "val", "test"):
        if not splits[s]:
            continue
        gin = stack(s, "gin"); ginm = stack(s, "gin_m")
        gout = stack(s, "gout"); goutm = stack(s, "gout_m")
        gin_n, gout_n = gin.copy(), gout.copy()
        for g in range(G):
            if K > 1:
                for k in range(K):
                    m = ginm[:, k, g] == 1
                    gin_n[m, k, g] = (gin_n[m, k, g] - means[g]) / stds[g]
            else:
                m = ginm[:, g] == 1
                gin_n[m, g] = (gin_n[m, g] - means[g]) / stds[g]
            mo = goutm[:, g] == 1
            gout_n[mo, g] = (gout_n[mo, g] - means[g]) / stds[g]

        N = len(splits[s])
        if use_clin:
            Xs = clin_X[[p["idx"] for p in splits[s]]].copy()      # (N, V, seq_len, 2)
            for v in range(len(clin_vars)):
                mv = Xs[:, v, :, 1] == 1
                Xs[:, v, :, 0][mv] = (Xs[:, v, :, 0][mv] - clin_means[v]) / clin_stds[v]
            np.clip(Xs[:, :, :, 0], -CLINICAL_CLIP, CLINICAL_CLIP, out=Xs[:, :, :, 0])
            # Unobserved hours keep their NaN init; zero them (mask channel still flags
            # missing) so the GRU never sees NaN — see 02b for the same guard.
            np.nan_to_num(Xs[:, :, :, 0], copy=False, nan=0.0)
            vals = Xs[:, :, :, 0].transpose(0, 2, 1)
            msks = Xs[:, :, :, 1].transpose(0, 2, 1)
            clinical = np.concatenate([vals, msks], axis=2).astype(np.float32)
        else:
            clinical = np.zeros((N, seq_len, 0), np.float32)       # score-only

        np.savez_compressed(
            os.path.join(args.save_dir, f"{s}.npz"),
            clinical=clinical,
            gtsq_in=gin_n, gtsq_in_mask=ginm,
            gtsq_out=gout_n, gtsq_out_mask=goutm,
            enc_ids=np.array([p["enc"] for p in splits[s]]),
            # patient key for cluster-robust evaluation; falls back to the encounter when the
            # export cannot link one, so the array is always a valid clustering variable
            pat_ids=np.array([str(enc2pat.get(p["enc"], f'enc:{p["enc"]}')) for p in splits[s]]),
            dates=np.array([f'{p["enc"]}_{p["date_in"]}' for p in splits[s]]),
            day_of_stay=np.array([p["day_idx"] for p in splits[s]], np.float32),  # episode-relative ICU day
            supervisor_d=np.array([SUP_VOCAB.get(p.get("sup_d", ""), 0) for p in splits[s]], np.int64),
            supervisor_next=np.array([SUP_VOCAB.get(p.get("sup_next", ""), 0) for p in splits[s]], np.int64),
        )
        print(f"  saved {s}.npz  ({N} samples, clinical {clinical.shape})")

    np.savez(os.path.join(args.save_dir, "norm_stats.npz"),
             clinical_means=clin_means, clinical_stds=clin_stds,
             clinical_variables=np.array(clin_vars),
             gtsq_means=means, gtsq_stds=stds,
             gtsq_features=np.array(features),
             gtsq_ftypes=ftypes, gtsq_nclasses=nclasses,
             seq_len=seq_len, gap_hours=args.gap_hours, gtsq_hour=GTSQ_HOUR,
             gtsq_history=K, horizon=args.horizon)
    print(f"Done -> {args.save_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="GTSQ next-day prediction preprocessing")
    p.add_argument("--gtsq_csv", type=str, default="gtsq_day_level_encoded.csv",
                   help="Pre-encoded GTSQ day-level CSV (from 01_preprocess_gtsq.py)")
    p.add_argument("--repo_root", type=str, default=None,
                   help="Dissertation root with data/long (only for --use_clinical; auto-detected)")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--gtsq_history", type=int, default=1)
    p.add_argument("--horizon", type=int, default=1,
                   help="predict day D+horizon (1=next-day headline; >1 for the horizon/granularity sweep)")
    p.add_argument("--min_gtsq_values", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--gap_hours", type=int, default=6)
    p.add_argument("--day_target", action="store_true", default=False,
                   help="the target is a DAY AGGREGATE (SOFA and friends) rather than a form stamped "
                        "at a time of day. Enables a check that the window ends on a day boundary: "
                        f"gap_hours must be {GTSQ_HOUR} (causal) or {GTSQ_HOUR - 24} (oracle). "
                        "Without it, gap_hours=0 silently leaks 14 h of the target day.")
    p.add_argument("--use_clinical", action="store_true", default=False)
    p.add_argument("--use_cultures", action="store_true", default=False,
                   help="append microbiology (blood-culture) variables to the clinical panel: "
                        "culture_drawn @collection_time (leakage-safe suspicion proxy), "
                        "culture_result_returned / culture_positive @arrival_time. Needs --use_clinical.")
    p.add_argument("--clinical_source", choices=["capped", "complete"], default="capped",
                   help="UMM clinical long: 'capped' (umm_long.parquet, 7d) or "
                        "'complete' (umm_long_complete.parquet, full LOS). Explicit, no auto-detect.")
    p.add_argument("--clinical_repr", choices=["hourly", "summary"], default="hourly",
                   help="Clinical encoding: 'hourly' (24h sequence) or 'summary' "
                        "(min/max/mean/median over 1/4/6/12/24h slots -> 4V-var x 5-slot pseudo-seq).")
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--val_size", type=float, default=0.20)
    p.add_argument("--random_state", type=int, default=42)  # dissertation-wide seed
    p.add_argument("--split_by", choices=["encounter", "patient"], default="encounter",
                   help="Grouping unit for the split. 'encounter' is the historical default and is "
                        "NOT patient-disjoint (repeat admissions land in different folds); "
                        "'patient' groups every encounter of one lifeTimeNumber into one fold and "
                        "is required for any claim about unseen patients. Needs --patient_map.")
    p.add_argument("--patient_map", type=str,
                   default="../raw_data/UMM_Export/260528_UMM_demographics.csv",
                   help="UMM demographics export providing encounterNumber -> lifeTimeNumber. Read "
                        "for --split_by patient, and always used to write pat_ids + audit overlap.")
    p.add_argument("--split_mode", choices=["patient", "temporal"], default="patient",
                   help="patient = random split by EncounterNumber (default, used everywhere in the "
                        "paper); temporal = chronological holdout, encounters ordered by their first "
                        "day-pair, for testing robustness to distribution shift over time")
    a = p.parse_args()
    if a.save_dir is None:
        # clin splits carry the source so capped/complete never share a directory
        clin = f"clin_{a.clinical_source}" if a.use_clinical else "noclin"
        repr_t = "_summary" if (a.use_clinical and a.clinical_repr == "summary") else ""
        hist = f"_h{a.gtsq_history}" if a.gtsq_history > 1 else ""
        hz = f"_hz{a.horizon}" if a.horizon > 1 else ""
        pat = "_patsplit" if a.split_by == "patient" else ""
        a.save_dir = f"01_split_sets/gtsq_pred_{a.seq_len}h_gap{a.gap_hours}h_{clin}{repr_t}{hist}{hz}{pat}"
    return a


if __name__ == "__main__":
    main(parse_args())