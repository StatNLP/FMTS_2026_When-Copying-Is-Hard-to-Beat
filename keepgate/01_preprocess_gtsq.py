#!/usr/bin/env python3
"""GTSQ raw export -> day-level encoded CSV (UMM Ground-Truth Sepsis Questionnaire).

First stage of the GTSQ pipeline; the analog of mimic_targets.load_daily_targets for the
MIMIC companion. Ports the (tested) logic of 01_dataset_analysis.ipynb to a script and
adapts it to the 260528 UMM export, which splits the questionnaire into:
  * entries (encounterId; encounterNumber; charttime; itemid; value1..6; NotInFactualStay; version)
  * dict    (id; description1..6)            — the field names per itemid
(The original notebook read a single R11_umm_GTSQ.csv with description1..6 inline.)

Output: gtsq_day_level_encoded.csv — identifiers (EncounterNumber, date) + context
(i01000_Supervisor) + the 45 numeric prediction features. Consumed by 02_preprocess.py.
"""

import argparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema (45 prediction features) + encoding maps  (from 01_dataset_analysis.ipynb)
# ---------------------------------------------------------------------------
ORGANS = ["Lunge", "Niere", "Gehirn", "Herz", "Leber",
          "Gastrointestinal", "Gerinnungssystem", "Knochenmark"]
LOKALISATIONS = [
    "abdominell", "Endokarditis", "Blood-stream-infection", "Haut",
    "intrakraniell / meningeal", "katheterassoziiert", "thorakal",
    "urogenital", "Gelenke / Knochen",
]

ORDINAL_MAPS = {
    'i03000_Arbeitsdiagnose': {'Weder SIRS noch Sepsis': 0, 'SIRS': 1, 'Sepsis': 2,
                               'Schwere Sepsis': 3, 'Septischer Schock': 4},
    'i02000_Klinisches_Gesamtbild': {'verbessert': 0, 'nicht verändert': 1, 'verschlechtert': 2},
    'i04000_Infektionsverdacht': {'Nein': 0, 'Ja': 1, 'Gesichert': 2},
}
BINARY_MAPS = {'i09_Capillary_Leak': {'Nein': 0, 'Ja': 1}}
AKUT_MAP = {'neu': 1, 'akut': 2}
URSACHE_MAP = {'nicht infektionsbedingt': 1, 'unklar': 2, 'infektionsbedingt': 3}
CHRONISCH_MAP = {'chronisch': 1}

ID_COLS = ['EncounterNumber', 'date']
# Context columns are carried into the day-level CSV but are NEVER prediction targets.
# 'version' (questionnaire v1/v2) is passed through so the v1/v2 split is reproducible from the
# CSV alone; 02_preprocess lists it in GTSQ_META_COLS so it is not mistaken for a 46th feature.
# 'i01000_Station'/'i01000_Bett' are carried so WARD-LEVEL analyses are possible: the patient-group
# item is defined RELATIVE to the other patients currently on the unit ("im Vergleich zu den aktuell
# behandelten Intensivpatienten ... zu den DREI am schwersten Erkrankten"), i.e. it is a constrained
# ranking over the ~20 occupied beds, not a per-patient label. Without the station a single-patient
# model cannot see why that item moves.
CONTEXT_COLS = ['i01000_Supervisor', 'i01000_Station', 'i01000_Bett', 'version']


def gtsq_feature_names():
    """The 45 prediction features (Knochenmark has no 'chronisch')."""
    feats = ['i01000_Gruppe_(0|1|2)', 'i02000_Klinisches_Gesamtbild',
             'i03000_Arbeitsdiagnose', 'i04000_Infektionsverdacht']
    feats += [f'i05100_Lokalisation__{loc}' for loc in LOKALISATIONS]
    feats += ['i09_Capillary_Leak']
    for organ in ORGANS:
        subs = ['present', 'akut oder "neu"', 'Ursache für Dysfunktion']
        if organ != 'Knochenmark':
            subs.append('chronisch')
        feats += [f'i11100_{organ}_{s}' for s in subs]
    return feats


# ---------------------------------------------------------------------------
# Load raw 260528 export (entries + dict) -> notebook-style long df_raw
# ---------------------------------------------------------------------------
def load_raw(entries_path, dict_path):
    """Reconstruct the description1..6 + value1..6 long format the pivot expects."""
    ent = pd.read_csv(entries_path, sep=';', skiprows=[0, 2], na_values=['NULL'],
                      on_bad_lines='skip', low_memory=False)
    dct = pd.read_csv(dict_path, sep=';', skiprows=[0, 2], na_values=['NULL'],
                      on_bad_lines='skip')
    ent = ent.rename(columns={'encounterNumber': 'EncounterNumber'})
    ent['itemid'] = pd.to_numeric(ent['itemid'], errors='coerce')
    ent = ent.dropna(subset=['itemid', 'EncounterNumber'])
    ent['itemid'] = ent['itemid'].astype(int)
    ent['EncounterNumber'] = ent['EncounterNumber'].astype(str)

    desc = [f'description{i}' for i in range(1, 7)]
    dct['id'] = pd.to_numeric(dct['id'], errors='coerce')
    dmap = dct.dropna(subset=['id']).drop_duplicates('id').set_index('id')[desc]

    df = ent.merge(dmap, left_on='itemid', right_index=True, how='left')
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')
    df = df.dropna(subset=['charttime'])
    print(f"Raw: {len(df):,} rows, {df['EncounterNumber'].nunique():,} encounters, "
          f"{df['charttime'].min().date()}–{df['charttime'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Raw long -> patient-day wide  (port of 01_dataset_analysis.ipynb cell 14)
# ---------------------------------------------------------------------------
NO_HIERARCHY_FILL = False


def pivot_to_day(df_raw):
    df_work = df_raw.drop_duplicates(
        subset=['EncounterNumber', 'charttime', 'itemid', 'description1', 'value1'],
        keep='first')

    df_11100 = df_work[df_work['itemid'] == 11100]
    df_5100 = df_work[df_work['itemid'] == 5100]
    df_other = df_work[~df_work['itemid'].isin([11100, 5100])]

    # 11100: organ-specific columns
    rec11100 = []
    for (enc, time), grp in df_11100.groupby(['EncounterNumber', 'charttime']):
        r = {'EncounterNumber': enc, 'charttime': time,
             'version': grp['version'].iloc[0], 'NotInFactualStay': grp['NotInFactualStay'].iloc[0]}
        for _, row in grp.iterrows():
            organ = row['value1']
            if pd.notna(organ):
                for k in (2, 3, 4):
                    if pd.notna(row[f'value{k}']) and pd.notna(row[f'description{k}']):
                        r[f'i11100_{organ}_{row[f"description{k}"]}'] = row[f'value{k}']
                r[f'i11100_{organ}_present'] = 1
        rec11100.append(r)
    df_11100_wide = pd.DataFrame(rec11100)

    # 5100: one-hot per localisation
    rec5100 = []
    for (enc, time), grp in df_5100.groupby(['EncounterNumber', 'charttime']):
        r = {'EncounterNumber': enc, 'charttime': time,
             'version': grp['version'].iloc[0], 'NotInFactualStay': grp['NotInFactualStay'].iloc[0]}
        for _, row in grp.iterrows():
            loc = row['value1']
            if pd.notna(loc):
                r[f'i05100_Lokalisation__{loc}'] = 1
                if pd.notna(row.get('value2')):
                    r[f'i05100_{loc}__vermutet_gesichert'] = row['value2']
                if pd.notna(row.get('value3')):
                    r[f'i05100_{loc}__fokussanierung'] = 1
        rec5100.append(r)
    df_5100_wide = pd.DataFrame(rec5100)

    # other items: melt (description, value) -> pivot
    base = ['EncounterNumber', 'charttime', 'itemid', 'version', 'NotInFactualStay']
    melted = []
    for i in range(1, 7):
        d, v = f'description{i}', f'value{i}'
        t = df_other[base + [d, v]].rename(columns={d: 'attribute', v: 'value'})
        melted.append(t.dropna(subset=['attribute', 'value']))
    long = pd.concat(melted, ignore_index=True)
    long['column_name'] = ('i' + long['itemid'].astype(str).str.zfill(5)
                           + '_' + long['attribute'].str.replace(' ', '_'))
    df_other_wide = long.pivot_table(index=['EncounterNumber', 'charttime'],
                                     columns='column_name', values='value',
                                     aggfunc='first').reset_index()
    meta = long[['EncounterNumber', 'charttime', 'version', 'NotInFactualStay']].drop_duplicates()
    df_other_wide = df_other_wide.merge(meta, on=['EncounterNumber', 'charttime'], how='left')

    merge_cols = ['EncounterNumber', 'charttime', 'version', 'NotInFactualStay']
    df_wide = df_other_wide.merge(df_11100_wide, on=merge_cols, how='outer')
    df_wide = df_wide.merge(df_5100_wide, on=merge_cols, how='outer')

    # Capillary leak: merge v1 (i09000 sub-field) + v2 (i09100 standalone) -> i09_Capillary_Leak
    v1c, v1m = 'i09000_Capillary_Leak', 'i09000_Makrozirkulatorisches_Problem'
    if v1c in df_wide.columns and v1m in df_wide.columns:
        assessed = df_wide[v1m].notna()
        v1 = df_wide[v1c].notna().astype(float).where(assessed, np.nan)
    elif v1m in df_wide.columns:
        v1 = pd.Series(0.0, index=df_wide.index).where(df_wide[v1m].notna(), np.nan)
    else:
        v1 = pd.Series(np.nan, index=df_wide.index)
    v1_txt = v1.map({1.0: 'Ja', 0.0: 'Nein'})
    if 'i09100_Capillary_leakage' in df_wide.columns:
        df_wide['i09_Capillary_Leak'] = df_wide['i09100_Capillary_leakage'].fillna(v1_txt)
    else:
        df_wide['i09_Capillary_Leak'] = v1_txt
    df_wide = df_wide.drop(columns=[c for c in df_wide.columns if c.startswith('i09000_')]
                           + ['i09100_Capillary_leakage'], errors='ignore')

    # aggregate to patient-day
    df_wide['charttime'] = pd.to_datetime(df_wide['charttime'])
    df_wide['date'] = df_wide['charttime'].dt.date
    item_cols = [c for c in df_wide.columns if c.startswith('i') and '_' in c]
    apd = df_wide.groupby(['EncounterNumber', 'date']).size()
    if apd.max() > 1:
        agg = {'version': 'first', 'NotInFactualStay': 'first', 'charttime': 'last'}
        present_like = lambda x: x.max() if len(x.dropna()) else np.nan
        last_like = lambda x: x.dropna().iloc[-1] if len(x.dropna()) else np.nan
        for c in item_cols:
            agg[c] = (present_like if (c.endswith('_present')
                      or c.startswith('i05100_Lokalisation__') or c.endswith('__fokussanierung'))
                      else last_like)
        df_daily = df_wide.groupby(['EncounterNumber', 'date']).agg(agg).reset_index()
    else:
        df_daily = df_wide
    df_daily = df_daily.sort_values(['EncounterNumber', 'date']).reset_index(drop=True)

    # Enforce the questionnaire hierarchy by zero-filling (this is a MODELLING convention, not raw
    # data: 'Organdysfunktion=Nein' => every organ absent; 'Fokus=Nein' => every localisation absent;
    # and a present organ with an unfilled sub-field => 0). NB for auditors: because of this, a
    # "parent=Nein but child flagged" check on the ENCODED CSV is tautologically 0 -- test such
    # hierarchy constraints on the raw export, not here.
    organ_pres = [c for c in df_daily.columns if c.startswith('i11100_') and c.endswith('_present')]
    if NO_HIERARCHY_FILL:
        # --no_hierarchy_fill leaves derived cells missing, so an evaluation mask built from this
        # table contains only cells the clinician actually filled. The expansion is a defensible
        # modelling convention, but it turns one answer ("no organ dysfunction") into 32 observed
        # zero cells that then repeat trivially from day to day, so a persistence result computed
        # over the filled table and one computed over explicit entries are different claims.
        print("--no_hierarchy_fill: skipping the organ/localisation zero-fill "
              "(explicit entries only)")
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        print(f"Patient-day: {df_daily.shape[0]:,} rows, "
              f"{df_daily['EncounterNumber'].nunique():,} encounters")
        return df_daily
    if 'i11000_Organdysfunktion' in df_daily.columns:
        df_daily.loc[df_daily['i11000_Organdysfunktion'] == 'Nein', organ_pres] = 0
    any_org = df_daily[organ_pres].sum(axis=1, skipna=True) > 0
    for c in organ_pres:
        df_daily.loc[any_org & df_daily[c].isna(), c] = 0
    # zero-fill localisation one-hot
    lok = [c for c in df_daily.columns if c.startswith('i05100_Lokalisation__')]
    if 'i05000_Infektiologisch_relevanter_Fokus' in df_daily.columns:
        df_daily.loc[df_daily['i05000_Infektiologisch_relevanter_Fokus'] == 'Nein', lok] = 0
    any_loc = df_daily[lok].sum(axis=1, skipna=True) > 0
    for c in lok:
        df_daily.loc[any_loc & df_daily[c].isna(), c] = 0

    df_daily['date'] = pd.to_datetime(df_daily['date'])
    print(f"Patient-day: {df_daily.shape[0]:,} rows, {df_daily['EncounterNumber'].nunique():,} encounters")
    return df_daily


# ---------------------------------------------------------------------------
# Encode -> 45 numeric features  (port of cell 24)
# ---------------------------------------------------------------------------
def encode_features(df_day):
    features = gtsq_feature_names()
    ctx = [c for c in CONTEXT_COLS if c in df_day.columns]     # tolerate an absent context col
    cols = ID_COLS + ctx + [c for c in features if c in df_day.columns]
    missing = [c for c in features if c not in df_day.columns]
    if missing:
        print(f"WARNING: {len(missing)} feature columns absent in raw: {missing}")
    df = df_day[cols].copy()

    for col, m in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(m)
    for col, m in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(m)
    for organ in ORGANS:
        pres = f'i11100_{organ}_present'
        akut = f'i11100_{organ}_akut oder "neu"'
        urs = f'i11100_{organ}_Ursache für Dysfunktion'
        chron = f'i11100_{organ}_chronisch'
        assessed = None
        if pres in df.columns:
            df[pres] = pd.to_numeric(df[pres], errors='coerce')
            assessed = df[pres].notna()
        # These three are the WITHIN-organ half of the hierarchy fill: given a recorded `_present`,
        # an unfilled sub-field is materialised as 0. --no_hierarchy_fill must skip this too, or the
        # "explicit" cohort still contains derived cells and the sub-fields track `_present` exactly.
        fill = not NO_HIERARCHY_FILL
        if akut in df.columns and assessed is not None:
            df[akut] = df[akut].map(AKUT_MAP)
            if fill:
                df.loc[assessed & df[akut].isna(), akut] = 0.0
        if urs in df.columns and assessed is not None:
            df[urs] = df[urs].map(URSACHE_MAP)
            if fill:
                df.loc[assessed & df[urs].isna(), urs] = 0.0
        if chron in df.columns and assessed is not None:
            df[chron] = df[chron].map(CHRONISCH_MAP)
            if fill:
                df.loc[assessed & df[chron].isna(), chron] = 0.0
    for loc in LOKALISATIONS:
        c = f'i05100_Lokalisation__{loc}'
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'i01000_Gruppe_(0|1|2)' in df.columns:
        df['i01000_Gruppe_(0|1|2)'] = pd.to_numeric(df['i01000_Gruppe_(0|1|2)'], errors='coerce')

    for col in df.columns:
        if col not in ID_COLS + CONTEXT_COLS:
            df[col] = df[col].astype('float32')
    return df


def build_umm_sofa(out_csv, with_cns=False, icu_day_anchor=False, max_day=0):
    """Daily SOFA on the UMM cohort -- the same six components MIMIC uses, on the same patients
    whose questionnaire this paper forecasts.

    The point is a controlled contrast: GTSQ is a physician JUDGEMENT and SOFA is COMPUTED from the
    chart, but both are recorded for the same patients in the same unit under the same charting
    system. Anything that separates them cannot be cohort, institution or data quality. Scoring
    reuses mimic_targets.SOFA_AGG / SOFA_FEATURE_NAMES and harmonize.scores.sofa.sofa_components
    unchanged, so the label is identical in definition to the MIMIC point, not a re-derivation.
    """
    import pandas as pd, numpy as np, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "keepgate"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("mt", root / "keepgate" / "mimic_targets.py")
    mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
    from harmonize.scores.sofa import sofa_components

    # The long table is keyed by stay_id and `hour` (hours since that stay's icu_start); the GTSQ
    # day-level CSV is keyed by EncounterNumber, which is the hospital_admission_id as a string.
    # The cohort table carries both, so the timestamp is icu_start + hour and the day follows.
    coh = pd.read_parquet(root / "data" / "long" / "umm_cohort.parquet",
                          columns=["stay_id", "hospital_admission_id", "icu_start"])
    coh["EncounterNumber"] = coh["hospital_admission_id"].astype(str)
    coh["icu_start"] = pd.to_datetime(coh["icu_start"])
    long = pd.read_parquet(root / "data" / "long" / "umm_long_complete.parquet",
                           columns=["stay_id", "hour", "variable", "value"])
    long = long[long["variable"].isin(mt.SOFA_AGG.keys())]
    long = long.merge(coh[["stay_id", "EncounterNumber", "icu_start"]], on="stay_id", how="inner")
    # DAY ANCHOR. The MIMIC SOFA target is indexed by ICU DAY (hours since admission // 24: its keys
    # read 30000831_d0, _d1), while this one is indexed by CALENDAR DAY (midnight to midnight, keys
    # like 10046_2014-12-15). That is an uncontrolled difference between two targets the paper
    # compares directly: a calendar day is a fixed wall-clock block, an ICU day starts at whatever
    # hour the patient arrived. --umm_sofa_icuday matches MIMIC's convention so the two can be
    # compared on the most similar setting available.
    if icu_day_anchor:
        # Emit a real TIMESTAMP at the patient's own admission hour, spaced exactly 24 h apart --
        # not a bare integer. Downstream, 02_preprocess detects consecutive days by date arithmetic
        # and maps the window onto ICU hours via (target - icu_start); an integer index breaks both
        # and yields zero day-pairs, which is what the first attempt produced. Anchoring the
        # timestamp on icu_start is what makes the block boundary fall at the admission hour rather
        # than midnight, so the time-of-day varies across patients exactly as it does in MIMIC-IV.
        long["_iday"] = (long["hour"] // 24).astype(int)
        long["date"] = long["icu_start"].dt.floor("h") + pd.to_timedelta(long["_iday"] * 24, unit="h")
    else:
        long["date"] = (long["icu_start"] + pd.to_timedelta(long["hour"], unit="h")).dt.floor("D")
    long = long.dropna(subset=["date", "value"])
    # urine_output is SUMMED over the day, so a partial first/last calendar day undercounts it and
    # inflates the renal subscore (SOFA scores LOW urine as worse). MIMIC avoids this by using whole
    # icu_day blocks. Drop each stay's first and last calendar day rather than score them wrong.
    # Only urine_output is SUMMED, so only it is distorted by a partial day. Dropping the whole day
    # removed 16% of encounter-days and 10% of encounters -- every stay of two days or fewer -- which
    # biased the cohort toward long-stayers, a worse error than the one being fixed. Blank the urine
    # measurement on edge days instead; the renal subscore then falls back to creatinine, which SOFA
    # defines it to do when urine output is unavailable.
    edges = long.groupby("EncounterNumber")["date"].agg(["min", "max"]).reset_index()
    long = long.merge(edges, on="EncounterNumber", how="left")
    edge = (long["date"] == long["min"]) | (long["date"] == long["max"])
    long = long[~(edge & (long["variable"] == "urine_output"))]
    long = long.drop(columns=["min", "max"])
    parts = []
    for how in ("min", "max", "sum"):
        vs = [v for v, h in mt.SOFA_AGG.items() if h == how]
        sub = long[long["variable"].isin(vs)]
        if len(sub):
            parts.append(sub.groupby(["EncounterNumber", "date", "variable"])["value"]
                            .agg(how).reset_index())
    daily = pd.concat(parts, ignore_index=True)
    wide = daily.pivot_table(index=["EncounterNumber", "date"], columns="variable",
                             values="value").reset_index()
    for c in ("gcs_eye", "gcs_motor", "gcs_verbal"):
        if c not in wide.columns:
            wide[c] = np.nan
    wide["gcs_total"] = wide[["gcs_eye", "gcs_motor", "gcs_verbal"]].sum(axis=1, min_count=3)
    for name in mt.SOFA_FEATURE_NAMES:
        if name not in wide.columns:
            wide[name] = np.nan
    # (n_days, F): urine_output is already a 24 h total, so it must not be summed again.
    comp = sofa_components(wide[mt.SOFA_FEATURE_NAMES].to_numpy(dtype=np.float32),
                           mt.SOFA_FEATURE_NAMES, urine_is_24h_total=True)
    out = wide[["EncounterNumber", "date"]].copy()
    # CNS is excluded by default, as in the Paper-1 analysis: GCS is charted on only 33.2% of UMM
    # days, an order of magnitude sparser than the other five components. NOTE the reason is NOT
    # that sedation removes the verbal score, as previously stated here: eye (25,054 days), motor
    # (24,876) and verbal (24,783) are present in near-lockstep, so the whole GCS is simply absent
    # on two days in three rather than one subscale being lost. This makes UMM SOFA a FIVE-component
    # target while the MIMIC SOFA point uses six -- they are not identical labels.
    # --umm_sofa_cns builds the six-component version instead, leaving CNS NaN where GCS is
    # unavailable so the masked evaluation simply scores fewer CNS cells. That makes the two SOFA
    # points component-matched, which is what the UMM-vs-MIMIC comparison needs.
    keep = ["respiration", "coagulation", "liver", "cardiovascular", "renal"]
    if with_cns:
        keep.append("cns")
    for k in keep:
        out[f"sofa_{k}"] = getattr(comp, k)
    sub = [c for c in out.columns if c.startswith("sofa_")]
    out = out.dropna(subset=sub, how="all")
    if max_day:
        # rank days within each encounter; keep the first max_day of them
        out = out.sort_values(["EncounterNumber", "date"])
        keep = out.groupby("EncounterNumber").cumcount() < max_day
        print(f"[umm_sofa] --max_day {max_day}: keeping {keep.sum():,} of {len(out):,} "
              f"encounter-days ({keep.mean():.1%})")
        out = out[keep]
    out.to_csv(out_csv, index=False)
    print(f"[umm_sofa] {len(out):,} encounter-days, {out['EncounterNumber'].nunique():,} encounters "
          f"-> {out_csv}")
    print("  per-component non-null:", {c: int(out[c].notna().sum()) for c in sub})
    return out


def fix_new_new(df):
    """AUTHORISED DATA CORRECTION (UMM clinicians, July 2026) -- must be reported in the methods.

    The form offers, per organ, 'in den letzten 24h neu aufgetreten' (1) vs 'akut seit >24h
    bestehend' (2). A dysfunction that was 'neu' yesterday and still present today therefore
    CANNOT read 'neu' again -- the form's own wording requires 'akut'. Where day D and day D+1
    both read 'neu' on the same organ we correct the day-D+1 value to 'akut'.

    Decided on the ORIGINAL series (each day judged against its raw predecessor), so a run
    neu/neu/neu corrects days 2 and 3 rather than stopping after the first fix. Only consecutive
    calendar days are touched -- across a documentation gap the rule does not apply.

    NB this REMOVES an artefact that was propping up the persistence baseline: on such a cell
    persistence predicted 'neu' and was scored correct; after the fix the target is 'akut', so
    persistence is wrong there and the new->acute rule (markov) is right. Expect the baseline
    ladder to move. Returns (df, n_corrected, n_consecutive_pairs).
    """
    cols = [c for c in df.columns if c.startswith("i11100_") and c.endswith('akut oder "neu"')]
    df = df.sort_values(["EncounterNumber", "date"]).reset_index(drop=True)
    prev_date = df.groupby("EncounterNumber")["date"].shift(1)
    consec = (df["date"] - prev_date).dt.days.eq(1)
    n_corr = 0
    for c in cols:
        prev = df.groupby("EncounterNumber")[c].shift(1)      # original column: not yet modified
        m = consec & prev.eq(1) & df[c].eq(1)
        n_corr += int(m.sum())
        df.loc[m, c] = 2.0
    return df, n_corr, int(consec.sum())


def main(args):
    global NO_HIERARCHY_FILL
    NO_HIERARCHY_FILL = getattr(args, "no_hierarchy_fill", False)
    df_raw = load_raw(args.entries, args.dict)
    df_day = pivot_to_day(df_raw)
    df_clean = encode_features(df_day)
    if args.fix_new_new:
        df_clean, n_corr, n_pairs = fix_new_new(df_clean)
        print(f"--fix_new_new: corrected {n_corr} organ cells 'neu'->'akut' "
              f"(over {n_pairs:,} consecutive day-pairs) -- REPORT THIS IN THE METHODS")
    if getattr(args, "umm_sofa", None):
        build_umm_sofa(args.umm_sofa, with_cns=args.umm_sofa_cns,
                       icu_day_anchor=args.umm_sofa_icuday,
                       max_day=args.umm_sofa_max_day); return
    df_clean.to_csv(args.out, index=False)
    nfeat = len([c for c in df_clean.columns if c not in ID_COLS + CONTEXT_COLS])
    print(f"Saved {args.out}: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols "
          f"({nfeat} features)")
    cov = df_clean.drop(columns=ID_COLS + CONTEXT_COLS).notna().mean().mean() * 100
    print(f"Mean feature coverage: {cov:.1f}%")


def parse_args():
    p = argparse.ArgumentParser(description="GTSQ raw export -> day-level encoded CSV")
    base = "../raw_data/UMM_Export"   # relative to keepgate/ (repo_root/raw_data)
    p.add_argument("--entries", default=f"{base}/260528_UMM_gtsq_entries.csv")
    p.add_argument("--dict", default=f"{base}/260528_UMM_gtsq_dict.csv")
    p.add_argument("--umm_sofa", default=None,
                   help="instead of the questionnaire, write a day-level UMM SOFA CSV to this "
                        "path (five components -- CNS excluded -- MIMIC definitions) for the within-cohort "
                        "computed-vs-judged contrast")
    p.add_argument("--umm_sofa_max_day", type=int, default=0,
                   help="keep only the first N ICU days, matching MIMIC SOFA's 168 h horizon. "
                        "MIMIC caps at 6 windows per encounter; UMM runs to 188, and 53%% of UMM "
                        "SOFA windows lie beyond ICU day 7 where MIMIC stops entirely. Late ICU "
                        "days are more stable, so persistence is stronger there and there is less "
                        "to gain -- the two targets are otherwise measured on nearly disjoint "
                        "parts of the stay.")
    p.add_argument("--umm_sofa_icuday", action="store_true", default=False,
                   help="index UMM SOFA by ICU DAY (hours since admission // 24) instead of calendar "
                        "day, matching how the MIMIC SOFA target is built. The two targets are "
                        "compared directly in the circularity analysis and this is the one "
                        "windowing difference between them that is not a consequence of the "
                        "questionnaire's 14:00 clock.")
    p.add_argument("--umm_sofa_cns", action="store_true", default=False,
                   help="include the CNS component, making UMM SOFA component-matched with the "
                        "six-component MIMIC point. GCS is charted on 33.2%% of days, so CNS is NaN "
                        "elsewhere and the masked evaluation scores fewer CNS cells.")
    p.add_argument("--out", default="gtsq_day_level_encoded.csv")
    p.add_argument("--no_hierarchy_fill", action="store_true", default=False,

                   help="do NOT zero-fill the organ/localisation hierarchy; the resulting "

                        "table marks only explicitly recorded cells as observed (audit R3)")
    p.add_argument("--fix_new_new", action="store_true", default=False,
                   help="apply the authorised 'neu'->'akut' correction on consecutive days "
                        "(see fix_new_new(); changes the target, must be reported)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())