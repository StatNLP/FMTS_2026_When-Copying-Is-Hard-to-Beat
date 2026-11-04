#!/usr/bin/env python3
"""All GTSQ descriptive/diagnostic analyses behind the Paper-3 claims, in ONE script.

Merges what were five separate files (analyse_residual, analyse_sirs, analyse_prognosis,
analyse_ward, plus the direction check) so a convention change is made once, not five times --
the same reason gtsq_eval.py exists.

Subcommands
  cohort      cohort sizes + the three masking conventions (paper Sec. 3)
  residual    is the change aleatoric? normalised ordinal error vs chance, per item (Table 2)
  direction   how much "correct direction" a know-nothing baseline gets (Sec. 4 strawman)
  sirs        do the working-diagnosis rungs agree with MEASURED SIRS criteria? (Sec. 4)
  prognosis   can the physician forecast tomorrow? v1-only prospective item (Sec. 4)
  ward        the severity ranking is a capacity-constrained ward assignment (App. G)
  peritem     per-item breakdown of the FMTS main table, all 45 items (camera-ready appendix)
  all         run everything

Conventions: target-observed masking unless stated; corrected (fix_new_new) splits; the twin/pooling
statistics live in bayes_floor.py and deliberately pool train+test there -- do not mix (doc SS.N.1b).

Usage:  python exp_fixnn/analyse_gtsq.py [subcommand]
"""
import os
import argparse, collections, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); KEEP = os.path.dirname(HERE)
sys.path.insert(0, KEEP)
import gtsq_eval as ge                                                   # noqa: E402

NOCLIN = os.path.join(HERE, "01_split_sets/gtsq_pred_24h_gap6h_noclin")
CLIN = os.path.join(HERE, "01_split_sets/gtsq_pred_24h_gap6h_clin_cult")
CSV = os.path.join(HERE, "gtsq_day_level_encoded_fixnn.csv")
# Two forms count as duplicates only if they share at least this many observed items.
DUP_MIN_ITEMS = int(os.environ.get("KG_DUP_MIN_ITEMS", "40"))
RAW = os.path.join(KEEP, "../raw_data/UMM_Export/260528_UMM_gtsq_entries.csv")
DX, IMP, INF, GRP = ("i03000_Arbeitsdiagnose", "i02000_Klinisches_Gesamtbild",
                     "i04000_Infektionsverdacht", "i01000_Gruppe_(0|1|2)")
CLINICAL = {DX: "working diagnosis (5)", IMP: "impression (3)",
            GRP: "severity ranking (3)", INF: "infection (3)"}


def meta(split=NOCLIN):
    n = np.load(os.path.join(split, "norm_stats.npz"), allow_pickle=True)
    return ([str(x) for x in n["gtsq_features"]], [str(x) for x in n["gtsq_ftypes"]],
            [int(x) for x in n["gtsq_nclasses"]], n["gtsq_means"].astype(float),
            n["gtsq_stds"].astype(float), n)


def load(name, split=NOCLIN):
    """(I, T, target_mask, pair_mask) as integer classes."""
    fe, ft, nc, gm, gs, _ = meta(split)
    d = np.load(os.path.join(split, f"{name}.npz"), allow_pickle=True)
    gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
    im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
    om = d["gtsq_out_mask"] == 1
    return (ge.denorm_to_class(gi, gm, gs, ft, nc),
            ge.denorm_to_class(d["gtsq_out"], gm, gs, ft, nc), om, (im == 1) & om)


# --------------------------------------------------------------------------- cohort
def cmd_cohort():
    print("=" * 74); print("COHORT + MASKING CONVENTIONS (paper Sec. 3)"); print("=" * 74)
    tot = 0
    for sp in ("train", "val", "test"):
        I, T, om, pm = load(sp); tot += len(I)
        print(f"  {sp:5s} pairs {len(I):6,} | target-obs cells {om.sum():8,} | "
              f"pair-obs {pm.sum():8,} | fully-obs pairs {pm.all(1).sum():5,}")
    print(f"  TOTAL {tot:,} day-pairs  (the canonical cohort; sub-cohorts must be labelled)")
    I, T, om, pm = load("test")
    print(f"  test change rate: target-obs {((I!=T)&om).sum()/om.sum()*100:.2f}%  "
          f"pair-obs {((I!=T)&pm).sum()/pm.sum()*100:.2f}%")


# --------------------------------------------------------------------------- residual
def destinations(Itr, Ttr, Mtr, nc):
    dest = {}
    for j in range(Itr.shape[1]):
        m = Mtr[:, j]; s = np.clip(Itr[m, j], 0, nc[j] - 1); t = np.clip(Ttr[m, j], 0, nc[j] - 1)
        ch = s != t
        for sv in range(nc[j]):
            sel = (s == sv) & ch
            if sel.sum():
                dest[(j, sv)] = int(np.bincount(t[sel], minlength=nc[j]).argmax())
    return dest


def cmd_residual():
    fe, ft, nc, *_ = meta()
    Itr, Ttr, otr, _ = load("train"); I, T, om, _ = load("test")
    dest = destinations(Itr, Ttr, otr, nc)
    print("=" * 74)
    print("RESIDUAL: is the CHANGE aleatoric? oracle knows WHICH cells change (Table 2)")
    print("  normalised by (k-1) so 3- and 5-class items are comparable; 'vs chance' < 1 is better")
    print("=" * 74)
    print(f"  {'item':24s}{'k':>3}{'n':>7}{'exact':>8}{'norm|d|':>9}{'chance':>8}{'ratio':>7}")
    rows = []
    for j in range(len(fe)):
        if ft[j] == "binary":
            continue
        m = om[:, j]; i = np.clip(I[m, j], 0, nc[j] - 1); t = np.clip(T[m, j], 0, nc[j] - 1)
        c = i != t
        if c.sum() < 30:
            continue
        k = nc[j]; p = np.array([dest.get((j, int(x)), int(x)) for x in i[c]]); tt = t[c]; ii = i[c]
        nd = np.abs(p - tt).mean() / (k - 1)
        cd = np.mean([np.mean([abs(x - b) for x in range(k) if x != a]) for a, b in zip(ii, tt)]) / (k - 1)
        rows.append((fe[j], k, int(c.sum()), (p == tt).mean(), nd, cd, nd / cd))
    for f, k, n_, ex, nd, cd, r in rows:
        if f in CLINICAL:
            print(f"  {CLINICAL[f]:24s}{k:>3}{n_:>7}{ex:>8.3f}{nd:>9.3f}{cd:>8.3f}{r:>7.2f}")
    oth = [r for r in rows if r[0] not in CLINICAL]
    if oth:
        w = np.array([r[2] for r in oth], float); w /= w.sum()
        agg = [float(np.sum(w * np.array([r[i] for r in oth]))) for i in (3, 4, 5, 6)]
        print(f"  {'organ ordinals (pooled)':24s}{3:>3}{sum(r[2] for r in oth):>7}"
              f"{agg[0]:>8.3f}{agg[1]:>9.3f}{agg[2]:>8.3f}{agg[3]:>7.2f}")


# --------------------------------------------------------------------------- direction
def cmd_direction():
    fe, ft, nc, *_ = meta()
    Itr, Ttr, otr, _ = load("train"); I, T, om, _ = load("test"); G = len(fe)
    P = I.copy()
    for j in range(G):
        c = np.bincount(np.clip(Ttr[otr[:, j], j], 0, nc[j] - 1), minlength=nc[j])
        order = np.argsort(c)[::-1]
        P[:, j] = [next((x for x in order if x != v), (1 - v if nc[j] == 2 else v))
                   for v in np.clip(I[:, j], 0, nc[j] - 1)]
    binm = np.array([ft[j] == "binary" for j in range(G)])
    print("=" * 74)
    print("DIRECTION: what a KNOW-NOTHING 'always-switch' baseline scores (Sec. 4)")
    print("=" * 74)
    for lab, sel in [("all items", np.ones(G, bool)), ("binary", binm), ("ordinal", ~binm)]:
        m = om & sel[None, :]; chg = (I != T) & m; moved = (P != I) & chg
        ok = (np.sign(P - I) == np.sign(T - I))[moved]
        print(f"  {lab:12s}: direction correct on {ok.mean()*100:5.1f}% of {moved.sum():6,} cells")
    print("  -> 'our overrides move in the right direction' is therefore not evidence of skill.")


# --------------------------------------------------------------------------- sirs
def cmd_sirs():
    fe, ft, nc, gm, gs, n = meta(CLIN)
    cv = [str(x) for x in n["clinical_variables"]]
    cm, cs = n["clinical_means"].astype(float), n["clinical_stds"].astype(float)
    I = M = C = None
    for sp in ("train", "val", "test"):
        d = np.load(os.path.join(CLIN, f"{sp}.npz"), allow_pickle=True)
        gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
        Ii = ge.denorm_to_class(gi, gm, gs, ft, nc)
        I = Ii if I is None else np.vstack([I, Ii])
        M = (d["gtsq_out_mask"] == 1) if M is None else np.vstack([M, d["gtsq_out_mask"] == 1])
        C = d["clinical"] if C is None else np.concatenate([C, d["clinical"]], 0)
    V = len(cv); val, msk = C[:, :, :V], C[:, :, V:]

    def crit(name, lo=None, hi=None):
        if name not in cv:
            return np.zeros(len(val), bool), np.zeros(len(val), bool)
        k = cv.index(name); v = val[:, :, k] * cs[k] + cm[k]; m = msk[:, :, k] == 1
        hit = np.zeros(len(val), bool)
        if hi is not None:
            hit |= ((v > hi) & m).any(1)
        if lo is not None:
            hit |= ((v < lo) & m).any(1)
        return hit, m.any(1)
    hr, o2 = crit("hr", hi=90.0); rr, o3 = crit("rr", hi=20.0)
    wbc, o4 = crit("wbc", lo=4.0, hi=12.0)
    j = fe.index(DX)
    print("=" * 74); print("SIRS: do the recorded rungs match the MEASURED criteria? (Sec. 4)"); print("=" * 74)
    for fev, lab in ((38.0, "temp>38 (Bone 1992)"), (38.3, "temp>38.3 (SSC 2012)")):
        t, o1 = crit("temp_core", lo=36.0, hi=fev)
        ncrit = t.astype(int) + hr.astype(int) + rr.astype(int) + wbc.astype(int)
        nev = o1.astype(int) + o2.astype(int) + o3.astype(int) + o4.astype(int)
        ok = (nev >= 2) & M[:, j]; sirs = ncrit >= 2
        s0 = ok & (I[:, j] == 0)
        print(f"  {lab:22s}: {sirs[s0].mean()*100:.1f}% of 'neither SIRS nor sepsis' days "
              f"(n={int(s0.sum()):,}) MEET SIRS   [excluded, <2 evaluable: {int((M[:,j]&~(nev>=2)).sum()):,}]")


# --------------------------------------------------------------------------- prognosis
def cmd_prognosis():
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    COLS = ['encounterId', 'encounterNumber', 'charttime', 'itemid', 'value1', 'value2', 'value3',
            'value4', 'value5', 'value6', 'NotInFactualStay', 'version']
    PROG = {'verbessern': 0, 'gleich bleiben': 1, 'verschlechtern': 2}
    e = pd.read_csv(RAW, sep=';', skiprows=[0, 2], names=COLS, header=0, low_memory=False,
                    on_bad_lines='skip')
    e = e[e['encounterNumber'].astype(str).str.match(r'^\d+$', na=False)].copy()
    e['itemid'] = pd.to_numeric(e['itemid'], errors='coerce')
    e['charttime'] = pd.to_datetime(e['charttime'], errors='coerce')
    p = e[e['itemid'] == 12000].dropna(subset=['charttime']).copy()
    p['EncounterNumber'] = p['encounterNumber'].astype(str)
    p['date'] = p['charttime'].dt.normalize(); p['prog'] = p['value1'].map(PROG)
    p = p.dropna(subset=['prog']).drop_duplicates(['EncounterNumber', 'date'])
    d = pd.read_csv(CSV, low_memory=False); d['date'] = pd.to_datetime(d['date'])
    d['EncounterNumber'] = d['EncounterNumber'].astype(str)
    feats = [c for c in d.columns if c.startswith(('i0', 'i1'))
             and not any(k in c for k in ('Supervisor', 'Station', 'Bett'))]
    d = d.sort_values(['EncounterNumber', 'date'])
    nx = d.groupby('EncounterNumber')[feats].shift(-1)
    nd = d.groupby('EncounterNumber')['date'].shift(-1)
    cons = (nd - d['date']).dt.days.eq(1)
    chg = ((d[feats].values != nx.values) & d[feats].notna().values & nx.notna().values).sum(1)
    dd = d.assign(any_chg=(chg > 0).astype(int), wnext=d.groupby('EncounterNumber')[DX].shift(-1))[cons.values]
    j = dd.merge(p[['EncounterNumber', 'date', 'prog']], on=['EncounterNumber', 'date'], how='inner')
    j2 = j.dropna(subset=[DX, 'wnext'])
    print("=" * 74); print("PROGNOSIS: can the PHYSICIAN forecast tomorrow? (v1-only item, Sec. 4)"); print("=" * 74)
    print(f"  day-pairs with a recorded prognosis: {len(j):,}   (NB not the same as the number of")
    print(f"    forms carrying the item -- a form/pair conflation caused an earlier n=12,218)")
    print(f"  AUROC prognosis -> ANY item changes    : {roc_auc_score(j['any_chg'], j['prog']):.3f}")
    print(f"  AUROC prognosis -> working dx changes  : {roc_auc_score((j2[DX]!=j2['wnext']).astype(int), j2['prog']):.3f}")
    print(f"  AUROC prognosis -> working dx WORSENS  : {roc_auc_score((j2['wnext']>j2[DX]).astype(int), j2['prog']):.3f}")
    print(f"  answers 'unchanged' on {(j['prog']==1).mean()*100:.0f}% of days -> the human also defaults to persistence")


# --------------------------------------------------------------------------- ward
def cmd_ward():
    import pandas as pd
    ST = "i01000_Station"
    d = pd.read_csv(CSV, low_memory=False); d['date'] = pd.to_datetime(d['date'])
    d = d[d[ST].notna() & d[GRP].notna()].sort_values(['EncounterNumber', 'date'])
    occ = d.groupby([ST, 'date']).agg(n_pat=(GRP, 'size'),
                                      n_top=(GRP, lambda s: int((s == 2).sum())),
                                      n_bot=(GRP, lambda s: int((s == 0).sum()))).reset_index()
    occ = occ[occ['n_pat'] >= 5]
    print("=" * 74); print("WARD: the severity ranking is a capacity-constrained assignment (App. G)"); print("=" * 74)
    print(f"  ward-days (>=5 patients) {len(occ):,} | median unit size {occ['n_pat'].median():.0f}")
    for lab, c in (("most severe (2)", "n_top"), ("least severe (0)", "n_bot")):
        v = occ[c]
        print(f"    {lab:18s} mean {v.mean():.2f}  <=3 on {(v<=3).mean()*100:.1f}% of ward-days  max {v.max()}")
    nxt = d.groupby('EncounterNumber')[GRP].shift(-1)
    nd = d.groupby('EncounterNumber')['date'].shift(-1)
    keep = (nd - d['date']).dt.days.eq(1) & nxt.notna()
    p = d[keep].copy(); p['changed'] = p[GRP] != nxt[keep]
    p = p.merge(occ[[ST, 'date', 'n_pat']], on=[ST, 'date'], how='inner')
    print(f"  P(group changes) by own group: " + "  ".join(
        f"{g}:{p[p[GRP]==g]['changed'].mean()*100:.0f}%" for g in (0, 1, 2)))
    p['q'] = pd.qcut(p['n_pat'], 4, labels=['Q1 small', 'Q2', 'Q3', 'Q4 large'], duplicates='drop')
    print("  P(group changes) by unit size: " + "  ".join(
        f"{q}:{s['changed'].mean()*100:.0f}%" for q, s in p.groupby('q', observed=True)))


# --------------------------------------------------------------------------- describe
def cmd_describe():
    """Cohort, form completeness and the >=4-item filter -- every descriptive number in Section 3."""
    import pandas as pd
    df = pd.read_csv(CSV, low_memory=False)
    fe, ft, nc, gm, gs, _ = meta(CLIN)
    df["date"] = pd.to_datetime(df["date"])
    print(f"patient-days {len(df):,}   encounters {df.EncounterNumber.nunique():,}   "
          f"{df.date.min().date()} .. {df.date.max().date()}")
    d = df.groupby("EncounterNumber").size()
    print(f"  GTSQ days per encounter: median {d.median():.0f} (IQR {d.quantile(.25):.0f}-"
          f"{d.quantile(.75):.0f}, mean {d.mean():.1f}, max {d.max()})")
    print(f"  encounters contributing >=1 day-pair: {(d >= 2).sum():,}")

    # completeness over EXACTLY the 45 modelled items (a prefix filter also catches
    # Supervisor/Station/Bett and misses i09_Capillary_Leak -- that produced an impossible '47 of 45')
    filled = df[fe].notna().sum(axis=1)
    print(f"\n  filled items per form (of {len(fe)}): median {filled.median():.0f} "
          f"(IQR {filled.quantile(.25):.0f}-{filled.quantile(.75):.0f}, mean {filled.mean():.1f})")
    for k in range(1, 4):
        print(f"    exactly {k}: {int((filled == k).sum()):6,} ({100*(filled == k).mean():5.2f}%)")
    print(f"    4-44      : {int(((filled >= 4) & (filled < 45)).sum()):6,} "
          f"({100*((filled >= 4) & (filled < 45)).mean():5.2f}%)")
    print(f"    complete  : {int((filled == 45).sum()):6,} ({100*(filled == 45).mean():5.2f}%)")
    one = df[filled == 1]
    for f in fe:
        c = int(one[f].notna().sum())
        if c:
            print(f"    the single item in 1-item forms: {f} -- {c:,}/{len(one):,} ({100*c/len(one):.1f}%)")

    drop = filled < 4
    print(f"\n  >=4-item filter drops {int(drop.sum()):,} forms ({100*drop.mean():.2f}%)")
    print("    by year: " + ", ".join(f"{y}:{n}" for y, n in
                                      df[drop].groupby(df.date.dt.year).size().items()))
    enc = df.groupby("EncounterNumber").agg(days=("date", "size"), nd=(drop.name if drop.name else "date", "size"))
    aff = df[drop].EncounterNumber.unique()
    e_aff = df[df.EncounterNumber.isin(aff)].groupby("EncounterNumber").size()
    e_un = df[~df.EncounterNumber.isin(aff)].groupby("EncounterNumber").size()
    print(f"    encounters affected: {len(aff):,} ({100*len(aff)/df.EncounterNumber.nunique():.1f}%)"
          f"  median days {e_aff.median():.0f} vs {e_un.median():.0f} unaffected")
    # Aggregate per ENCOUNTER first: the claim is about encounters, and a per-row mean would weight
    # long stays more heavily -- which is exactly the property that differs between the two groups.
    g_enc = df.groupby("EncounterNumber")[GRP].mean()
    print(f"    mean severity ranking, per encounter: affected {g_enc[g_enc.index.isin(aff)].mean():.3f} "
          f"vs {g_enc[~g_enc.index.isin(aff)].mean():.3f} unaffected "
          f"(per patient-day it is {df[df.EncounterNumber.isin(aff)][GRP].mean():.3f} vs "
          f"{df[~df.EncounterNumber.isin(aff)][GRP].mean():.3f}; the per-day figure is confounded by "
          f"exposure, since affected stays are longer)")

    # volatility on the KEPT days, affected vs unaffected -- is the filter selective on movement?
    from scipy import stats
    kept = df[~drop].sort_values(["EncounterNumber", "date"])
    rows = []
    for e, g in kept.groupby("EncounterNumber"):
        v = g[fe].values
        if len(v) < 2:
            continue
        obs = ~pd.isna(v[1:]) & ~pd.isna(v[:-1])
        rows.append((e in set(aff), ((v[1:] != v[:-1]) & obs).sum() / max(obs.sum(), 1)))
    a_ = [r for f_, r in rows if f_]; u_ = [r for f_, r in rows if not f_]
    print(f"    change rate on kept days: affected {np.mean(a_):.4f} (n={len(a_):,}) vs "
          f"unaffected {np.mean(u_):.4f} (n={len(u_):,})  Mann-Whitney p="
          f"{stats.mannwhitneyu(a_, u_, alternative='two-sided').pvalue:.3g}")

    # changed-items-per-pair histogram (Appendix table)
    print("\n  changed items per day-pair (all splits):")
    allch = []
    for sp in ("train", "val", "test"):
        I, T, om, pm = load(sp, CLIN)
        allch.append(((I != T) & pm).sum(1))
    allch = np.concatenate(allch)
    print(f"    pairs {len(allch):,}  mean {allch.mean():.2f}  median {np.median(allch):.0f}")
    for k in range(8):
        print(f"      {k}: {int((allch == k).sum()):6,} ({100*(allch == k).mean():5.1f}%)")
    print(f"      8+: {int((allch >= 8).sum()):6,} ({100*(allch >= 8).mean():5.1f}%)")


# --------------------------------------------------------------------------- purity
def cmd_purity():
    """Per-organ purity of the neu->akut transition, and where the `new` targets come from."""
    fe, ft, nc, gm, gs, _ = meta(CLIN)
    org = [j for j, f in enumerate(fe) if "akut" in f.lower()]
    lbl = {0: "none", 1: "neu", 2: "akut"}
    print("TRAIN P(next | today = neu), per organ acuity item:")
    I, T, om, pm = load("train", CLIN)
    for j in org:
        m = pm[:, j] & (I[:, j] == 1)
        if not m.sum():
            continue
        t = T[m, j]
        print(f"  {fe[j][:40]:42s} n={int(m.sum()):4d}  " +
              "  ".join(f"->{lbl[c]} {float((t == c).mean()):.2f}" for c in range(3)))
    print("\nWhere the `new` targets come from (both days observed):")
    for sp in ("train", "val", "test"):
        I, T, om, pm = load(sp, CLIN)
        tgt = pm[:, org] & (T[:, org] == 1)
        src = I[:, org]
        print(f"  {sp:6s} targets=neu: {int(tgt.sum()):4d}   " +
              "  ".join(f"from {lbl[c]}: {int((tgt & (src == c)).sum()):4d}" for c in range(3)))

    # the single most frequent fully-observed day-D state, and what it is
    X, Y = [], []
    for sp in ("train", "val", "test"):
        I, T, om, pm = load(sp, CLIN)
        full = pm.all(1)
        X.append(I[full]); Y.append(T[full])
    I = np.vstack(X); T = np.vstack(Y)
    keys = [tuple(r) for r in I]
    top, cnt = collections.Counter(keys).most_common(1)[0]
    m = np.array([k == top for k in keys])
    nxt = collections.Counter(tuple(r) for r in T[m])
    print(f"\nmost frequent day-D state: {cnt} of {len(I):,} ({100*cnt/len(I):.1f}%), "
          f"{len(nxt)} distinct successors, identical {100*nxt[top]/cnt:.0f}% of the time")
    print("  non-zero items: " + ", ".join(f"{fe[j]}={top[j]}" for j in range(len(fe)) if top[j]))

def cmd_duplicates():
    """Identical questionnaires, divergent futures -- the irreducible part of the ceiling.

    If two patient-days carry the SAME 45-item form and then diverge the next day, no function of
    the form alone can predict both. This bounds what any model reading only the questionnaire can
    achieve, independently of architecture, and is the strongest evidence that the persistence
    ceiling is a property of the task rather than a failure of the models tried.

    Reported under TWO matching rules, because the answer depends on the choice and the paper should
    not make it silently:
      strict   -- both forms fully observed on all 45 items (the earlier convention)
      observed -- both forms share the SAME observation pattern and agree on every observed item,
                  with at least KG_DUP_MIN_ITEMS observed. This is what the code does; it is NOT
                  "match on the items observed in both forms", which an earlier version of this
                  docstring claimed. Intersection matching is not transitive, so it cannot be
                  expressed as a grouping at all. Requiring an identical pattern is STRICTER, so the
                  divergence it reports upper-bounds the intersection version. `strict` is the
                  cleanest statement for the paper and is the number to quote.

    Pairs are counted COMBINATORIALLY, never enumerated. Grouping identical forms by a hash of their
    observed values makes duplicates exactly the groups of size >= 2, and within a group of size k
    whose next-day vectors fall into classes of sizes c_1..c_m, the diverging pairs are
    k(k-1)/2 - sum_i c_i(c_i-1)/2. Enumerating pairs instead is quadratic in the largest group --
    the all-observed bucket holds tens of thousands of rows -- and does not finish.
    """
    import pandas as pd, hashlib
    fe, ft, nc, gm, gs, _ = meta(CLIN)
    df = pd.read_csv(CSV, low_memory=False)
    feats = [c for c in fe if c in df.columns]
    df["_d"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["_d"]).sort_values(["EncounterNumber", "_d"]).reset_index(drop=True)
    X = df[feats].to_numpy(dtype=float)
    obs = ~np.isnan(X)
    nxt = {}
    for enc, g in df.groupby("EncounterNumber"):
        idx = g.index.to_list()
        for a, b in zip(idx, idx[1:]):
            if (df.at[b, "_d"] - df.at[a, "_d"]).days == 1:
                nxt[a] = b
    have_next = np.array(sorted(nxt))
    full = obs.all(1)
    print(f"  rows {len(df):,}   with a next-day form {len(have_next):,}   items {len(feats)}")

    def key_of(i, mask):
        return hashlib.blake2b(np.ascontiguousarray(X[i][mask]).tobytes()
                               + mask.tobytes(), digest_size=16).digest()

    for mode in ("strict", "observed"):
        pool = (np.array([i for i in have_next if full[i] and full[nxt[i]]])
                if mode == "strict" else
                np.array([i for i in have_next
                          if obs[i].sum() >= DUP_MIN_ITEMS
                          and obs[nxt[i]].sum() >= DUP_MIN_ITEMS]))
        if len(pool) < 2:
            print(f"  {mode}: too few rows"); continue
        groups = {}
        for i in pool:
            m = obs[i] if mode == "observed" else np.ones(len(feats), bool)
            groups.setdefault(key_of(i, m), []).append(i)
        n_forms = n_pairs = n_div = n_form_div = 0
        grp_rates = []; top = None
        for _, idxs in groups.items():
            k = len(idxs)
            if k < 2:
                continue
            n_forms += k
            tot = k * (k - 1) // 2
            # classify each member's NEXT-day form; same key => the futures agree
            cls = {}
            for i in idxs:
                mn = obs[nxt[i]] if mode == "observed" else np.ones(len(feats), bool)
                cls[key_of(nxt[i], mn)] = cls.get(key_of(nxt[i], mn), 0) + 1
            same = sum(c * (c - 1) // 2 for c in cls.values())
            n_pairs += tot; n_div += tot - same
            grp_rates.append((tot - same) / tot if tot else 0.0)
            biggest = max(cls.values())
            # a form has a divergent twin unless EVERY member of its group shares one future
            n_form_div += 0 if biggest == k else k
            if top is None or k > top[0]:
                top = (k, len(cls), biggest / k, idxs[0])
        print(f"  {mode:9s} pool {len(pool):6,}   forms with >=1 exact duplicate {n_forms:6,} "
              f"({n_forms/len(pool):5.1%})   pairs {n_pairs:9,}   "
              f"diverging next day {n_div:9,} ({(n_div/n_pairs if n_pairs else 0):5.1%})")
        # Two other ways to pool the same evidence. They answer different questions and differ a
        # lot, so the paper must say which one it quotes:
        #   pair-weighted  (above) -- pick a random PAIR of identical forms. Large duplicate groups
        #                  dominate quadratically, which is correct for "can a lookup get both right".
        #   group-mean     -- average the divergence rate over duplicate GROUPS, one vote each.
        #   form-level     -- share of forms that have at least one twin with a different future.
        if grp_rates:
            print(f"  {'':9s} group-mean divergence {np.mean(grp_rates):5.1%} over {len(grp_rates):,} "
                  f"groups   form-level {n_form_div/n_forms if n_forms else 0:5.1%}")
        if mode == "strict" and top is not None:
            k, ndist, same_frac, idx0 = top
            print(f"  {'':9s} most frequent day-D form: occurs {k:,} times, followed by {ndist} "
                  f"distinct next-day forms, identical {same_frac:5.1%} of the time")
    print(f"  (`observed` requires at least {DUP_MIN_ITEMS} observed items on the day-D form AND "
          f"on its successor; without the successor floor, two nearly-empty next-day forms collide "
          f"and are scored as the same future, understating divergence)")


def cmd_volatility():
    """Change rate and persistence accuracy stratified by ICU day (paper Sec. 3).

    Answers the objection that persistence is inflated by stable long-stayers. Computed from the
    day-level CSV so the ICU day index is exact, on consecutive calendar-day pairs, counting only
    items observed on BOTH days -- the paper's convention. Reported to two decimals: the contrast
    between early and late stay is the point, and one decimal hides most of it.
    """
    import pandas as pd
    fe, ft, nc, gm, gs, _ = meta(CLIN)
    df = pd.read_csv(CSV, low_memory=False)
    feats = [c for c in fe if c in df.columns]
    df["_d"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["_d"]).sort_values(["EncounterNumber", "_d"])
    rows = []; skipped = 0; bad = 0
    # ICU day must be counted from ADMISSION, not from the first questionnaire. Anchoring on the
    # first form put late-admitted patients' day 5 into the "days 1-2" band -- the long-stayer
    # confound this command exists to rule out. icu_start comes from the cohort table.
    import pathlib
    coh = pd.read_parquet(pathlib.Path(KEEP).parent / "data" / "long" / "umm_cohort.parquet",
                          columns=["hospital_admission_id", "icu_start"])
    coh["EncounterNumber"] = coh["hospital_admission_id"].astype(str)
    coh["icu_start"] = pd.to_datetime(coh["icu_start"]).dt.floor("D")
    adm = dict(zip(coh["EncounterNumber"], coh["icu_start"]))
    for enc, g in df.groupby("EncounterNumber"):
        g = g.reset_index(drop=True)
        day0 = g["_d"].iloc[0]
        a0 = adm.get(str(enc))
        if a0 is None:
            skipped += 1          # no admission time -> cannot place this stay on the ICU-day axis
            continue              # silently defaulting to day 1 would REINTRODUCE the bug being fixed
        icu_day0 = int((day0 - a0).days) + 1
        if not (0 <= icu_day0 <= 400):
            bad += 1              # implausible offset: admission/questionnaire mismatch, not a stay
            continue
        for i in range(len(g) - 1):
            if (g["_d"].iloc[i + 1] - g["_d"].iloc[i]).days != 1:
                continue
            a_ = g.loc[i, feats].to_numpy(dtype=float)
            b_ = g.loc[i + 1, feats].to_numpy(dtype=float)
            m = ~np.isnan(a_) & ~np.isnan(b_)
            if m.sum() < 4:
                continue
            rows.append((icu_day0 + (g["_d"].iloc[i] - day0).days, int(m.sum()),
                         int((a_[m] != b_[m]).sum())))
    if not rows:
        print("  no consecutive-day pairs found"); return
    arr = np.array(rows)
    bands = [("days 1-2", arr[:, 0] <= 2), ("days 3-7", (arr[:, 0] >= 3) & (arr[:, 0] <= 7)),
             ("day 8+", arr[:, 0] >= 8), ("all", np.ones(len(arr), bool))]
    print(f"  consecutive-day pairs: {len(arr):,}  (>=4 items observed on both days)")
    if skipped or bad:
        # Report the share of ENCOUNTERS and of PAIRS. They differ by two orders of magnitude here
        # (15.8% vs 0.14%) because the excluded encounters are overwhelmingly single-day, and a
        # single-day stay contributes no consecutive pair. Encounters are the wrong denominator for
        # a day-pair statistic, and quoting only the raw count invites exactly that misreading.
        print(f"  EXCLUDED: {skipped} encounters absent from the cohort table, "
              f"{bad} with an implausible admission offset "
              f"({(skipped + bad) / (len(set(df['EncounterNumber'])) or 1):.1%} of encounters; "
              f"the pairs they would have contributed are NOT in the {len(arr):,} above)")
    print(f"  {'band':10s}{'pairs':>9}{'cells':>12}{'chg rate':>10}{'chg/day':>11}{'pers acc':>10}")
    for lab, sel in bands:
        a2 = arr[sel]
        cells = a2[:, 1].sum(); chg = a2[:, 2].sum()
        print(f"  {lab:10s}{len(a2):>9,}{cells:>12,}{chg/cells*100:>9.2f}%"
              f"{a2[:, 2].mean():>11.2f}{(1 - chg/cells)*100:>9.2f}%")


def cmd_focus():
    """The two localisation fields the feature set drops: vermutet/gesichert and fokussanierung.

    Both live in the RAW export (itemid 5100, value2 and value3) and are built by 01_preprocess_gtsq
    but then dropped at its final column selection, which keeps only the 45 declared features. This
    quantifies what is being discarded, so the decision to recover them (or not) rests on numbers.
    """
    import pandas as pd
    if not os.path.exists(RAW):
        print(f"  raw export not found at {RAW}"); return
    # A few free-text cells contain an embedded ';' and produce over-long lines. They are skipped
    # and COUNTED -- silently dropping rows from a descriptive table would misstate the denominator.
    import csv as _csv
    with open(RAW, newline="", encoding="utf8", errors="replace") as fh:
        rows = list(_csv.reader(fh, delimiter=";"))
    hdr = rows[1]; n = len(hdr)
    good = [r for r in rows[3:] if len(r) == n]
    bad = len(rows) - 3 - len(good)
    df = pd.DataFrame(good, columns=hdr)
    if bad:
        print(f"  note: {bad} malformed raw lines skipped (embedded ';' in free text) of "
              f"{len(rows)-3:,} total")
    df = df[df["itemid"].astype(str).str.strip() == "5100"].copy()
    for c in ("value1", "value2", "value3"):
        df[c] = df[c].replace("NULL", np.nan)
    df["day"] = pd.to_datetime(df["charttime"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["value1", "day"])
    tot = len(df)
    print(f"  localisation rows in the raw export: {tot:,}")
    print(f"  vermutet/gesichert present: {df['value2'].notna().sum():,} "
          f"({df['value2'].notna().mean()*100:.1f}%)   "
          f"fokussanierung present: {df['value3'].notna().sum():,} "
          f"({df['value3'].notna().mean()*100:.1f}%)")
    print(f"\n  {'site':26s}{'rows':>8}{'vermutet':>10}{'gesichert':>11}{'%ges':>7}{'fokussan.':>11}")
    for site, g in sorted(df.groupby("value1"), key=lambda kv: -len(kv[1])):
        v = int((g["value2"] == "vermutet").sum()); ges = int((g["value2"] == "gesichert").sum())
        fs = int(g["value3"].notna().sum())
        pct = ges / (v + ges) * 100 if v + ges else float("nan")
        print(f"  {str(site)[:24]:26s}{len(g):>8}{v:>10}{ges:>11}{pct:>6.0f}%{fs:>11}")
    # the change the current binary encoding cannot see: a site staying present while the
    # certainty moves. Consecutive CALENDAR days only, per encounter+site.
    df = df.sort_values(["encounterNumber", "value1", "day"])
    k = df.groupby(["encounterNumber", "value1"])
    prev = k["value2"].shift(); prevday = k["day"].shift()
    consec = (df["day"] - prevday).dt.days == 1
    both = consec & prev.notna() & df["value2"].notna()
    same, moved = int((both & (prev == df["value2"])).sum()), int((both & (prev != df["value2"])).sum())
    up = int((both & (prev == "vermutet") & (df["value2"] == "gesichert")).sum())
    down = int((both & (prev == "gesichert") & (df["value2"] == "vermutet")).sum())
    print(f"\n  consecutive-day pairs where the site persists and both certainties are known: {same+moved:,}")
    if same + moved:
        print(f"    certainty unchanged {same:,} ({same/(same+moved)*100:.1f}%)   "
              f"changed {moved:,} ({moved/(same+moved)*100:.1f}%)")
        print(f"    vermutet->gesichert {up:,}   gesichert->vermutet {down:,}")
        print("    ^ these are day-to-day changes the current binary encoding cannot represent.")


def cmd_chronic():
    """Can an organ go chronic -> `neu`, and does infection suspicion move with it?

    Every organ carries TWO items: an ordinal acuity (0 none / 1 neu / 2 akut) and a SEPARATE binary
    `chronisch`. The two are independent fields, so 'new acute damage on top of chronic disease' is
    representable -- the question the meeting raised is whether it is actually recorded, and what the
    infection item does when it happens. Descriptive over all three splits (stated as such; no model
    is selected here, so pooling does not leak).
    """
    fe, ft, nc, gm, gs, _ = meta(CLIN)
    org = [j for j, f in enumerate(fe) if "akut" in f.lower()]
    chron = {}
    for j in org:                                   # pair each acuity item with its chronic flag
        stem = fe[j].split("_akut")[0]
        for k, f in enumerate(fe):
            if f.startswith(stem) and "chronisch" in f.lower():
                chron[j] = k
    inf = next((k for k, f in enumerate(fe) if f.startswith("i04000")), None)
    print("onset of `neu` (acuity 0->1) split by whether the organ was flagged chronic on day D:")
    print(f"  {'organ':26s}{'chronic=1':>22}{'chronic=0':>22}{'ratio':>8}")
    tot = {1: [0, 0], 0: [0, 0]}
    for j in org:
        k = chron.get(j)
        if k is None:
            continue
        n = {1: [0, 0], 0: [0, 0]}                  # [onsets, opportunities] per chronic flag
        for sp in ("train", "val", "test"):
            I, T, om, pm = load(sp, CLIN)
            base = pm[:, j] & pm[:, k] & (I[:, j] == 0)      # acuity 'none' on day D, both observed
            for c in (0, 1):
                sel = base & (I[:, k] == c)
                n[c][0] += int((T[sel, j] == 1).sum()); n[c][1] += int(sel.sum())
        for c in (0, 1):
            tot[c][0] += n[c][0]; tot[c][1] += n[c][1]
        r = lambda c: n[c][0] / n[c][1] * 100 if n[c][1] else float("nan")
        ratio = r(1) / r(0) if n[0][1] and r(0) > 0 else float("nan")
        print(f"  {fe[j].split('_akut')[0].replace('i11100_',''):26s}"
              f"{n[1][0]:>8d}/{n[1][1]:<7d}{r(1):>6.2f}%{n[0][0]:>8d}/{n[0][1]:<7d}{r(0):>6.2f}%{ratio:>8.1f}x")
    r = lambda c: tot[c][0] / tot[c][1] * 100 if tot[c][1] else float("nan")
    all_ratio = r(1) / r(0) if (tot[0][1] and r(0) > 0) else float("nan")
    print(f"  {'ALL ORGANS':26s}{tot[1][0]:>8d}/{tot[1][1]:<7d}{r(1):>6.2f}%"
          f"{tot[0][0]:>8d}/{tot[0][1]:<7d}{r(0):>6.2f}%{all_ratio:>8.1f}x")
    # Infection can be recorded in TWO places, and the global item is only one of them: each organ
    # carries its own `Ursache fuer Dysfunktion` field whose value 3 = infektionsbedingt. A clinician
    # can attribute the new dysfunction to infection there without touching i04000, so looking only at
    # the global item understates how often infection is named at onset.
    cause = {}
    for j in org:
        stem = fe[j].split("_akut")[0]
        for k, f in enumerate(fe):
            if f.startswith(stem) and "Ursache" in f:
                cause[j] = k
    if cause:
        print("\nthe ORGAN's own cause field at its `neu` onset (3 = infektionsbedingt):")
        for c, lab in ((1, "chronic organ"), (0, "non-chronic organ")):
            nn = inf_at = opp = 0
            for sp in ("train", "val", "test"):
                I, T, om, pm = load(sp, CLIN)
                for j in org:
                    k, u = chron.get(j), cause.get(j)
                    if k is None or u is None:
                        continue
                    sel = (pm[:, j] & pm[:, k] & pm[:, u] & (I[:, j] == 0)
                           & (I[:, k] == c) & (T[:, j] == 1))
                    inf_at += int((T[sel, u] == 3).sum()); opp += int(sel.sum())
            if opp:
                print(f"  {lab:20s} n={opp:5d}  cause recorded as infection on day D+1 "
                      f"{inf_at/opp*100:5.1f}%")
    if inf is None:
        return
    print("\ninfection suspicion around a chronic organ's `neu` onset (day D -> D+1):")
    for c, lab in ((1, "chronic organ"), (0, "non-chronic organ")):
        before = after = opp = 0
        for sp in ("train", "val", "test"):
            I, T, om, pm = load(sp, CLIN)
            for j in org:
                k = chron.get(j)
                if k is None:
                    continue
                sel = (pm[:, j] & pm[:, k] & pm[:, inf] & (I[:, j] == 0)
                       & (I[:, k] == c) & (T[:, j] == 1))
                before += int((I[sel, inf] == 1).sum()); after += int((T[sel, inf] == 1).sum())
                opp += int(sel.sum())
        if opp:
            print(f"  {lab:20s} n={opp:5d}  infection already suspected on day D {before/opp*100:5.1f}%"
                  f"   -> suspected on day D+1 {after/opp*100:5.1f}%")



# --------------------------------------------------------------------------------------------
# Per-item breakdown of the FMTS main table. Reviewer 1q3E asked for the 45 items individually
# ("which clinical parameters are easier or harder to forecast could itself be one of the main
# contributions"), and the decision letter lists their absence as a noted weakness.
#
# Nothing is recomputed from a model here, so this is a decomposition of the test evaluation that
# was already read, not a second read-off. The three deterministic rows come from the stored
# per-cell predictions; the learned rows are the mean over their five seeds of each seed's
# per-item metrics, which is the same `report_object=seedmean` the table uses. The self-check at
# the end is exact: the mean over items of each column must equal that row's published macro-F1.
FMTS_ROWS = os.path.join(HERE, "fmts_eval_rows")
PERITEM_COLS = ["Persistence", "Persistence + rule", "Gate over rule, macro-F1 selected"]


# English item labels as published with the instrument (Lindner et al. 2022, Tables 1 and 3), so
# the appendix uses the names a reader can look up; the recorded identifiers are German.
_EN_ITEM = {"i01000_Gruppe_(0|1|2)": "Daily rating of illness severity",
            "i02000_Klinisches_Gesamtbild": "Patient's development in last 24\\,h",
            "i03000_Arbeitsdiagnose": "Working diagnosis",
            "i04000_Infektionsverdacht": "Suspicion of infection",
            "i09_Capillary_Leak": "Capillary leak"}
_EN_FOCUS = {"abdominell": "abdominal", "Endokarditis": "endocarditis",
             "Blood-stream-infection": "blood stream", "Haut": "skin",
             "intrakraniell / meningeal": "intracranial/meningeal", "katheterassoziiert": "catheter",
             "thorakal": "thoracic", "urogenital": "urogenital", "Gelenke / Knochen": "bone/joint"}
_EN_ORGAN = {"Lunge": "Lung", "Niere": "Kidney", "Gehirn": "Brain", "Herz": "Heart",
             "Leber": "Liver", "Gastrointestinal": "Gastrointestinal",
             "Gerinnungssystem": "Coagulation system", "Knochenmark": "Bone marrow"}
_EN_FIELD = {"present": "present", 'akut oder "neu"': "acute or new",
             "Ursache für Dysfunktion": "cause", "chronisch": "chronic"}


def _tex(s):
    """Recorded item identifier -> its published English label, LaTeX-safe."""
    if s in _EN_ITEM:
        return _EN_ITEM[s]
    if s.startswith("i05100_Lokalisation__"):
        return "Focus of infection: " + _EN_FOCUS[s[len("i05100_Lokalisation__"):]]
    if s.startswith("i11100_"):
        organ, field = s[len("i11100_"):].split("_", 1)
        return f"{_EN_ORGAN[organ]}: {_EN_FIELD[field]}"
    raise KeyError(f"no English label for item {s!r}")


def cmd_peritem():
    import json
    man = json.load(open(os.path.join(FMTS_ROWS, "fmts_manifest.json")))
    z = np.load(os.path.join(FMTS_ROWS, "fmts_predictions.npz"), allow_pickle=True)
    items, nc, ft = list(man["items"]), man["nclasses"], man["ftypes"]
    T, I, M = z["target"], z["day_d"], z["mask"]

    # per-item f1 / acc_change, one dict per reported row
    per = {}
    for row in PERITEM_COLS:
        runs = man["rows"][row]["runs"]
        if not runs:
            rep = ge.per_feature_report(z["pred::" + row], I, T, M, ft, nc)["per_feat"]
            per[row] = {items[g]: (rep[g]["f1"], rep[g]["acc_change"]) for g in rep}
        else:
            acc = collections.defaultdict(lambda: [0.0, 0.0])
            for r in runs:
                d = json.load(open(os.path.join(KEEP, r, "gtsq_metrics_test.json")))
                for it in items:
                    acc[it][0] += d[it]["f1_macro"] / len(runs)
                    acc[it][1] += (d[it]["acc_change"] or 0.0) / len(runs)
            per[row] = {it: tuple(v) for it, v in acc.items()}

    # support and change rate are properties of the data, read once
    dom = json.load(open(os.path.join(KEEP, man["rows"][PERITEM_COLS[-1]]["runs"][0],
                                      "gtsq_metrics_test.json")))
    n_tot = {it: int(M[:, g].sum()) for g, it in enumerate(items)}
    n_chg = {it: int((I[M[:, g], g] != T[M[:, g], g]).sum()) for g, it in enumerate(items)}

    print(f"  {'item':44s}{'n':>7}{'chg%':>7}" + "".join(f"{c[:11]:>12}" for c in PERITEM_COLS)
          + f"{'dF1':>8}{'Accc':>7}")
    lines = []
    for it in items:
        f1p, f1r, f1g = (per[c][it][0] for c in PERITEM_COLS)
        accc = per[PERITEM_COLS[-1]][it][1]
        pct = 100.0 * n_chg[it] / n_tot[it] if n_tot[it] else float("nan")
        print(f"  {it[:44]:44s}{n_tot[it]:7d}{pct:7.2f}{f1p:12.4f}{f1r:12.4f}{f1g:12.4f}"
              f"{f1g - f1r:+8.4f}{accc:7.3f}")
        dlt = f1g - f1r
        dlt_s = "0.0000" if abs(dlt) < 5e-5 else f"{dlt:+.4f}"   # no signed zeros from rounding
        lines.append(f"{_tex(it)} & {n_tot[it]:,} & {pct:.2f} & "
                     f"{f1p:.4f} & {f1r:.4f} & {f1g:.4f} & {dlt_s} & "
                     + (f"{accc:.3f}" if n_chg[it] else "--") + r" \\")

    print()
    for c in PERITEM_COLS:                      # the self-check: column mean == Table 1
        m = float(np.mean([per[c][it][0] for it in items]))
        print(f"  mean over items, {c:34s} macro-F1 = {m:.4f}")

    out = os.path.join(FMTS_ROWS, "fmts_peritem.tex")
    with open(out, "w") as f:
        f.write("% generated by `python exp_fixnn/analyse_gtsq.py peritem` -- do not hand-edit\n")
        f.write(r"""% needs \usepackage{longtable} in the preamble
\begingroup\footnotesize
\begin{longtable}{@{}lrrrrrrr@{}}
\caption{Per-item test results, all 45 items. $n$ is the item's scored cells (observed on both days)
and chg\% the share of them that change; macro-F1 is the per-item class-macro F1 whose mean over
items is the aggregate in Table~\ref{tab:main}, so each F1 column averages to that row's published
value. $\Delta$ is gate $-$ rule. $\Accc$ is the gate's destination accuracy on that item's changed
cells. The gate columns are the mean over five seeds.}
\label{tab:peritem}\\
\toprule
Item & $n$ & chg\% & Persist. & Rule & Gate & $\Delta$ & $\Accc$ \\
\midrule
\endfirsthead
\toprule
Item & $n$ & chg\% & Persist. & Rule & Gate & $\Delta$ & $\Accc$ \\
\midrule
\endhead
\bottomrule
\endlastfoot
""")
        f.write("\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n")
    print(f"\n  wrote {out}")

CMDS = {"cohort": cmd_cohort, "duplicates": cmd_duplicates, "focus": cmd_focus, "volatility": cmd_volatility, "describe": cmd_describe, "purity": cmd_purity, "chronic": cmd_chronic,
        "residual": cmd_residual, "direction": cmd_direction,
        "sirs": cmd_sirs, "prognosis": cmd_prognosis, "ward": cmd_ward,
        "peritem": cmd_peritem}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", default="all", choices=list(CMDS) + ["all"])
    a = ap.parse_args()
    for k, f in (CMDS.items() if a.cmd == "all" else [(a.cmd, CMDS[a.cmd])]):
        f(); print()
