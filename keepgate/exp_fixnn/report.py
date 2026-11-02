#!/usr/bin/env python3
"""Result harvesting and figures for the Paper-3 campaign, in ONE script.

Merges aggregate_stage4.py (the 5-seed ladder over every experiment), eval_subset.py (per-subset
scoring with per-subset baselines) and fig_taxonomy.py (the volatility x circularity figure).

Subcommands
  ladder   5-seed mean +- std for every configuration group (Stages 4/5/6 + meeting asks)
  subset   >0/>1/>3 cohorts scored against baselines computed ON THAT SUBSET (App. C)
  figure   docs/figures/fig_volatility_circularity.{pdf,png}

Usage:  python exp_fixnn/report.py {ladder|subset|figure}
"""
import os
import glob, json, os, re, importlib.util, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
KEEP = os.path.dirname(HERE)
# docs/tables holds the Paper-1 forecasting eval jsons that cmd_tsfladder reads.
DOCS = os.path.join(os.path.dirname(KEEP), "docs", "tables")
OUT = os.path.join(HERE, '../../docs/figures')
SPLIT = os.path.join(HERE, 'exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_clin_cult'.replace('exp_fixnn/',''))


import glob, json, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# Minimum changed cells for an item to enter the change-F1 average. 1 = every item that changes,
# matching gtsq_eval.four_metrics. Set KG_CHGF1_MIN=10 to reproduce the older, filtered tables.
CHGF1_MIN = int(os.environ.get("KG_CHGF1_MIN", "1"))


def _one(path):
    j = json.load(open(path))
    f = [v for v in j.values() if isinstance(v, dict)
         and {"f1_macro", "accuracy", "n_total", "n_changed"} <= set(v)]
    if not f:
        return None
    nt = sum(v["n_total"] for v in f); nc = sum(v["n_changed"] for v in f)
    # change-F1 averages over every item that changes at all, matching gtsq_eval.four_metrics (the
    # shared scorer the baselines and 03_train use, locked by test_metric_parity). This harvester
    # used to require >=10 changed cells, which silently dropped 3 of the 45 items -- Endokarditis,
    # Gastrointestinal_chronisch, Gerinnungssystem_chronisch, 4-9 changes each -- and raised the mean,
    # so model numbers were being compared against unfiltered baselines. CHGF1_MIN restores it.
    return (np.mean([v["f1_macro"] for v in f]),
            sum(v["accuracy"] * v["n_total"] for v in f) / nt,
            (sum(v["acc_change"] * v["n_changed"] for v in f if v["n_changed"]) / nc) if nc else 0.0,
            np.mean([v["f1_change"] for v in f if v["n_changed"] >= CHGF1_MIN]))


# Which --loss_balance the report describes. `uniform` is the default and reported standard since
# 2026-07-30; the pre-2026-07-30 fits are all `lw`, and run-dir names carry `_lb<balance>_`, so this
# one switch keeps the two populations from being averaged together.
BALANCE = os.environ.get("KG_BALANCE", "uniform")

# Every configuration in the paper rests on the SAME five seeds. Some groups on disk carry extra seeds
# (early sweeps used 9); aggregating those would give them a quieter mean than the rest, so the report
# restricts to the canonical set rather than averaging whatever happens to exist.
CANON_SEEDS = (42, 123, 456, 789, 1337)

# Which model tree to read. The masking convention changes the loss, the checkpoint selection AND the
# metrics, so runs under the two conventions must never be averaged together: `02_models` holds the
# historical target-observed fits, `02_models_m2` the both-days-observed ones the paper reports.
TREE = os.environ.get("KG_TREE", "02_models")

# The hyperparameter sweep writes to 02_models/hp/ under BOTH conventions, distinguishing them by an
# `m2` name prefix (m2uni_...) rather than by directory -- the ladder runs use separate trees, the
# sweep does not. So `hp/...` subpaths resolve against HP_TREE, not TREE, or every hp-derived row
# silently vanishes when the ladder is read from 02_models_m2.
HP_TREE = os.environ.get("KG_HP_TREE", "02_models")


def _tree_for(subpath):
    return HP_TREE if subpath.startswith("hp/") else TREE


def _seed_of(run_dir):
    m = re.search(r"_s(\d+)(?:_|$)", os.path.basename(run_dir.rstrip("/")))
    return int(m.group(1)) if m else None


def _bal_ok(run_dir):
    b = os.path.basename(run_dir.rstrip("/"))
    return f"_lb{BALANCE}_" in b and _seed_of(run_dir) in CANON_SEEDS


def agg(subpath):
    """Average over every seed run-dir under exp_fixnn/02_models/<subpath>/ (BALANCE only)."""
    runs = [r for r in glob.glob(os.path.join(HERE, _tree_for(subpath), subpath, "*", "gtsq_metrics_test.json"))
            if _bal_ok(os.path.dirname(r))]
    vals = [_one(r) for r in runs]; vals = [v for v in vals if v]
    if not vals:
        return None
    a = np.array(vals)
    return a.mean(0), a.std(0), len(vals)


def line(name, subpath):
    r = agg(subpath)
    if r is None:
        print(f"  {name:26s}  (pending)"); return
    m, s, n = r
    # An under-seeded group is not a smaller sample, it is a DIFFERENT and usually more optimistic
    # number: a 2-seed max is drawn from the same spread as the 5-seed mean but selected on fewer
    # draws. The hyperparameter sweep runs 2 seeds by design (selection on validation), which is
    # fine -- quoting one of those groups as a reported result is not. Anything short of the five
    # canonical seeds is flagged here so it cannot reach a table unnoticed.
    warn = "" if n >= len(CANON_SEEDS) else f"  <<< ONLY {n}/{len(CANON_SEEDS)} SEEDS -- DO NOT REPORT"
    print(f"  {name:26s}  F1 {m[0]:.4f}±{s[0]:.4f}  acc {m[1]:.4f}  chg-acc {m[2]:.4f}  F1chg {m[3]:.4f}  [n={n}]{warn}")





def _mod(name, path):
    s = importlib.util.spec_from_file_location(name, os.path.join(KEEP, path))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def cmd_baselines():
    """Persistence and persistence+rule under the paper's masking convention, from the data.

    These two numbers anchor every delta in the paper, so they must never be typed in by hand or
    computed in a throwaway one-liner: a baseline on the wrong convention silently shifts every
    comparison against it (the target-observed pair understated persistence by 0.0018 macro-F1).
    `KG_MASK=target` reproduces the historical convention for comparison.
    """
    sys.path.insert(0, KEEP); import gtsq_eval as ge
    mask = os.environ.get("KG_MASK", "both")
    N = np.load(os.path.join(SPLIT, "norm_stats.npz"), allow_pickle=True)
    gm = N["gtsq_means"].astype(float); gs = N["gtsq_stds"].astype(float)
    NC = [int(x) for x in N["gtsq_nclasses"]]; FE = [str(x) for x in N["gtsq_features"]]
    d = np.load(os.path.join(SPLIT, "test.npz"), allow_pickle=True)
    gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
    im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
    I = np.round(gi * gs + gm).astype(int); T = np.round(d["gtsq_out"] * gs + gm).astype(int)
    M = (d["gtsq_out_mask"] == 1) & ((im == 1) if mask == "both" else True)
    print(f"  mask={mask}   cells scored: {int(M.sum()):,}   pairs: {len(I):,}")
    rows = [("persistence", I.copy())]
    Pm = I.copy()
    for j, name in enumerate(FE):                      # the unconditional neu->akut rule
        if "akut" in name.lower():
            Pm[:, j] = np.where(np.clip(I[:, j], 0, 2) == 1, 2, np.clip(I[:, j], 0, 2))
    rows.append(("persistence + rule", Pm))
    # chg-F1 must use the SAME per-item rule as the model harvester in _one(): average over every
    # item that changes. Both figures are printed so the restricted variant stays inspectable, but
    # the reported one is the unrestricted mean -- restricting to >=10 changes drops 3 of 45 items
    # and raises the mean, which would flatter models quoted beside an unrestricted baseline.
    from sklearn.metrics import f1_score as _f1
    for lab, P in rows:
        acc, f1, cacc, _ = ge.four_metrics(P, I, T, M, NC)
        f1c_all, f1c_10 = [], []
        for g in range(len(NC)):
            m = M[:, g].astype(bool)
            if m.sum() < 1:
                continue
            t = T[m, g]; p = np.clip(P[m, g], 0, NC[g] - 1); i = I[m, g]
            ch = i != t
            if ch.sum():
                v = _f1(t[ch], p[ch], average="macro", labels=list(range(NC[g])), zero_division=0)
                f1c_all.append(v)
                if ch.sum() >= 10:
                    f1c_10.append(v)
        print(f"  {lab:22s} acc {acc:.4f}  macro-F1 {f1:.4f}  chg-acc {cacc:.4f}  "
              f"chg-F1 {np.mean(f1c_all):.4f} (all {len(f1c_all)} changing items, matches models)  "
              f"[{np.mean(f1c_10):.4f} if restricted to >=10 changes]")


def cmd_winners():
    """The configurations quoted by name in Section 4, with their seed counts.

    These are hyperparameter-SEARCH groups, which run 2 seeds by design for validation ranking. Any
    of them quoted as a reported result must first be topped up to the five canonical seeds -- the
    change-accuracy bound read 0.154 at 2 seeds and 0.137 at 5. line() flags the shortfall.
    """
    print("  (test set, mask=both, 250,536 cells)")
    cmd_baselines()
    # Validation-selected winners per metric (hp_search.py report), each scored once on test. The
    # accuracy winner is included precisely because it reproduces the rule and gains nothing.
    for lab, sub in (("F1-sel: gate on rule",  "hp/m2uni_gru_f1_markovt50_hp09"),
                     ("F1-sel: 2nd (hp23)",    "hp/m2uni_gru_f1_markovt50_hp23"),
                     ("acc-sel: gate on rule", "hp/m2uni_gru_acc_markovt50_hp10"),
                     ("chg-sel: change-tuned", "hp/m2uniobj_gru_f1_hp10"),
                     ("contrast: gate on copy","hp/m2uni_gru_f1_hp09"),
                     ("hp09 physiology SHUF",  "shuffle_hp09")):
        line(lab, sub)


def cmd_fmcal():
    """Sanity check (1): is a foundation model weak, or is the fixed DECODING throwing its signal away?

    Zero-shot decoding is a fixed rule -- threshold binaries at 0.5, round ordinals. This learns the
    decoder instead: per item, a map from (FM's predicted class, day-D class) to an output class,
    fitted on TRAIN only and applied unchanged to TEST. It is the cheapest possible step beyond
    zero-shot and it isolates the question -- if calibration recovers a large part of the gap to
    persistence, the pretrained representation was informative and the naive decode was the problem;
    if it recovers nothing, the model genuinely has little to say about this task and a full
    backbone fine-tune is the only remaining lever.

    KG_FM_PRED_{TRAIN,TEST} point at caches written by fm_baseline.py --save_pred.
    """
    sys.path.insert(0, KEEP); import gtsq_eval as ge
    ptr = os.environ.get("KG_FM_PRED_TRAIN", "02_models/fmpred_bolt_train.npy")
    pte = os.environ.get("KG_FM_PRED_TEST", "02_models/fmpred_bolt_test.npy")
    ptr, pte = os.path.join(HERE, ptr), os.path.join(HERE, pte)
    for f in (ptr, pte):
        if not os.path.exists(f):
            print(f"  missing {f} -- run fm_baseline.py --split_name <split> --save_pred <file>"); return
    N = np.load(os.path.join(SPLIT, "norm_stats.npz"), allow_pickle=True)
    NC = [int(x) for x in N["gtsq_nclasses"]]
    gm = N["gtsq_means"].astype(float); gs = N["gtsq_stds"].astype(float)

    def arrays(split):
        d = np.load(os.path.join(SPLIT, f"{split}.npz"), allow_pickle=True)
        gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
        im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
        I = np.round(gi * gs + gm).astype(int); T = np.round(d["gtsq_out"] * gs + gm).astype(int)
        return I, T, (im == 1) & (d["gtsq_out_mask"] == 1)

    Itr, Ttr, Mtr = arrays("train"); Ite, Tte, Mte = arrays("test")
    Ftr, Fte = np.load(ptr), np.load(pte)
    if len(Ftr) != len(Itr) or len(Fte) != len(Ite):
        print(f"  cache/row mismatch: train {len(Ftr)} vs {len(Itr)}, test {len(Fte)} vs {len(Ite)}")
        return
    # Fitting on train+val is leakage-free -- val selects nothing here, the decoder is fitted not
    # searched -- and gives the rarer (FM class, day-D class) cells enough support to clear the n>=20
    # floor. Both variants are reported: if they disagree, the map is memorising rather than learning.
    pva = os.environ.get("KG_FM_PRED_VAL", "02_models/fmpred_bolt_val.npy")
    pva = os.path.join(HERE, pva)
    fits = [("train only", Itr, Ttr, Mtr, Ftr)]
    if os.path.exists(pva):
        Iva, Tva, Mva = arrays("val"); Fva = np.load(pva)
        if len(Fva) == len(Iva):
            fits.append(("train+val", np.vstack([Itr, Iva]), np.vstack([Ttr, Tva]),
                         np.vstack([Mtr, Mva]), np.vstack([Ftr, Fva])))
        else:
            print(f"  val cache row mismatch ({len(Fva)} vs {len(Iva)}) -- skipping train+val")
    else:
        print(f"  no val cache at {pva} -- reporting train-only fit")
    G = len(NC)
    results = []
    for fit_name, Ifit, Tfit, Mfit, Ffit in fits:
      cal = Ite.copy()
      for g in range(G):
        m = Mfit[:, g]
        fs = np.clip(Ffit[m, g], 0, NC[g] - 1); ds = np.clip(Ifit[m, g], 0, NC[g] - 1); t = Tfit[m, g]
        # modal target for each (FM class, day-D class) cell seen at least 20 times in train;
        # anything rarer falls back to persistence rather than to a one-example guess.
        tbl = {}
        for fv in range(NC[g]):
            for dv in range(NC[g]):
                sel = (fs == fv) & (ds == dv)
                if sel.sum() >= 20:
                    tbl[(fv, dv)] = int(np.bincount(t[sel], minlength=NC[g]).argmax())
        fe_ = np.clip(Fte[:, g], 0, NC[g] - 1); de_ = np.clip(Ite[:, g], 0, NC[g] - 1)
        for i in range(len(Ite)):
            cal[i, g] = tbl.get((fe_[i], de_[i]), Ite[i, g])
      results.append((fit_name, cal))
    print("  (test, mask=both; the decoder never sees test)")
    rows = [("persistence", Ite), ("FM zero-shot", Fte)] + \
           [(f"FM + learned decode [{n}]", c) for n, c in results]
    for lab, P in rows:
        acc, f1, ca, fc = ge.four_metrics(P, Ite, Tte, Mte.astype(bool), NC)
        print(f"  {lab:30s} acc {acc:.4f}  macro-F1 {f1:.4f}  chg-acc {ca:.4f}  chg-F1 {fc:.4f}")


def cmd_tsfladder():
    """Paper-1 TSF ladder: LOCF vs ARIMA/ETS vs the best learned model, all five datasets.

    This is the contrast the persistence paper turns on -- ML beats copying on continuous physiology
    and does not on the daily questionnaire -- so it must be regenerable, not assembled by hand.
    Reads the eval jsons already in docs/tables. eICU uses the INSULIN-CAPPED runs: the uncapped
    eICU forecaster loses to LOCF, and that shortfall was traced to impossible insulin values
    (documentation 15.8), so quoting the uncapped number would report a data error as a model result.
    """
    import glob
    def mse(f):
        try:
            j = json.load(open(f))
        except Exception:
            return None
        def dig(d):
            for k, v in (d or {}).items():
                if k in ("val_mse_per_var", "test_mse_per_var"):
                    return v
                if isinstance(v, dict):
                    r = dig(v)
                    if r is not None:
                        return r
        return dig(j)
    def find(pats):
        for pat in pats:
            for f in sorted(glob.glob(os.path.join(DOCS, pat))):
                m = mse(f)
                if m is not None:
                    return m, os.path.basename(f)
        return None, None
    # (dataset, LOCF file, best-learned file). eICU is the capfix family throughout, so its four
    # numbers come from one consistent extraction rather than a mix of capped and uncapped runs.
    SPEC = [("mimic_iii", ["persisttest_mimic_iii.json"], ["eval_informer_mimic_iii_4h_d128.json"],
             ["eval_arima_mimic_iii*.json"], ["eval_ets_mimic_iii*.json"]),
            ("mimic_iv",  ["persisttest_mimic_iv.json"],  ["eval_informer_mimic_iv_4h_d128.json"],
             ["eval_arima_mimic_iv*.json"],  ["eval_ets_mimic_iv*.json"]),
            ("eicu",      ["eval_eicu_insfix_persistence.json"], ["eval_eicu_insfix_denseAR.json"],
             ["eval_arima_eicu_4h_capfix.json"], ["eval_ets_eicu_4h_capfix.json"]),
            ("hirid",     ["persisttest_hirid.json"],     ["eval_informer_hirid_4h_d128.json"],
             ["eval_arima_hirid*.json"],     ["eval_ets_hirid*.json"]),
            ("umm",       ["persisttest_umm.json"],       ["eval_informer_umm_4h_d128.json"],
             ["eval_arima_umm*.json"],       ["eval_ets_umm*.json"])]
    print("  Paper-1 TSF ladder -- MSE per variable, LOWER is better")
    print(f"  {'dataset':11s}{'LOCF':>9}{'ARIMA':>9}{'ETS':>9}{'best ML':>9}{'ML vs LOCF':>12}  source")
    n_arima = n_ets = n_ml = n = 0
    n_arima_n = n_ets_n = 0          # denominators: how many were actually measured
    for ds, fl, fm, fa, fe in SPEC:
        locf, _ = find(fl); ml, src = find(fm); ar, _ = find(fa); et, _ = find(fe)
        if locf is None or ml is None:
            print(f"  {ds:11s} MISSING (locf={locf is not None} ml={ml is not None})"); continue
        n += 1
        # A MISSING measurement is not evidence either way. Counting `ar is None` as "not worse"
        # would silently read 4/5 for a file that was never produced.
        if ar is None or et is None:
            print(f"  {ds:11s} WARNING: missing "
                  + ("ARIMA " if ar is None else "") + ("ETS" if et is None else ""))
        n_arima_n += (ar is not None); n_ets_n += (et is not None)
        n_arima += (ar is not None and ar > locf)
        n_ets += (et is not None and et > locf)
        n_ml += (ml < locf)
        f = lambda v: f"{v:9.4f}" if isinstance(v, float) else f"{'-':>9s}"
        print(f"  {ds:11s}{f(locf)}{f(ar)}{f(et)}{f(ml)}{(locf-ml)/locf*100:>11.1f}%  {src}")
    print(f"\n  ARIMA worse than LOCF: {n_arima}/{n_arima_n}   "
          f"ETS worse than LOCF: {n_ets}/{n_ets_n}   learned better than LOCF: {n_ml}/{n}")
    print("  eICU rows are the insulin-capped runs (documentation 15.8); the uncapped eICU "
          "forecaster loses to LOCF because of a medication-extraction error, not a model failure.")


def cmd_figregimes():
    """The paper's thesis in one panel: improvement over copying, by model class, in both regimes.

    New file (fig_regimes.*); the existing figures are left untouched.

    x is model class, y is percentage improvement over the copying baseline ON THAT TASK'S OWN
    METRIC -- MSE reduction against LOCF for the continuous forecasts, macro-F1 increase against
    persistence for the questionnaire. Those are different quantities and the axis label says so;
    what makes them comparable is that both answer "how much better than copying?", which is the
    only question this figure asks.

    eICU is pinned to the insulin-capped family for BOTH its baseline and its learned point, since
    mixing capped and uncapped runs across one line would invent a difference that is a data
    artefact (documentation 15.8). No capped foundation-model run exists, so that point is omitted
    rather than silently taken from the uncapped family.
    """
    import glob
    def mse(pats):
        for pat in pats:
            for f in sorted(glob.glob(os.path.join(DOCS, pat))):
                try:
                    j = json.load(open(f))
                except Exception:
                    continue
                def dig(d):
                    for k, v in (d or {}).items():
                        if k in ("val_mse_per_var", "test_mse_per_var"):
                            return v
                        if isinstance(v, dict):
                            r = dig(v)
                            if r is not None:
                                return r
                r = dig(j)
                if r is not None:      # keep searching; a match without the key is not an answer
                    return r
        return None
    # (label, LOCF, ARIMA, ETS, Chronos-Bolt base, learned)
    TSF = [("MIMIC-III", ["persisttest_mimic_iii.json"], ["eval_arima_mimic_iii*.json"],
            ["eval_ets_mimic_iii*.json"], ["eval_chronos_chronos_bolt_base_median_mimic_iii.json"],
            ["eval_informer_mimic_iii_4h_d128.json"]),
           ("MIMIC-IV", ["persisttest_mimic_iv.json"], ["eval_arima_mimic_iv*.json"],
            ["eval_ets_mimic_iv*.json"], ["eval_chronos_chronos_bolt_base_median_mimic_iv.json"],
            ["eval_informer_mimic_iv_4h_d128.json"]),
           ("eICU", ["eval_eicu_insfix_persistence.json"], ["eval_arima_eicu_4h_capfix.json"],
            ["eval_ets_eicu_4h_capfix.json"], [], ["eval_eicu_insfix_denseAR.json"]),
           ("HiRID", ["persisttest_hirid.json"], ["eval_arima_hirid*.json"],
            ["eval_ets_hirid*.json"], ["eval_chronos_chronos_bolt_base_median_hirid.json"],
            ["eval_informer_hirid_4h_d128.json"]),
           ("UMM", ["persisttest_umm.json"], ["eval_arima_umm*.json"],
            ["eval_ets_umm*.json"], ["eval_chronos_chronos_bolt_base_median_umm.json"],
            ["eval_informer_umm_4h_d128.json"])]
    xs = ["ARIMA", "ETS", "foundation\nmodel", "learned"]
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.3), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.5); ax.set_axisbelow(True)
    ax.axhline(0, color=INK, lw=1.2)

    # UMM is drawn heavier than the other four: UMM physiology and the UMM questionnaire are the
    # SAME patients, so that blue line and the red line are the controlled comparison. The other
    # four datasets are context showing the pattern is not one hospital's quirk.
    ends = []
    for name, fl, fa, fe, fc, fm in TSF:
        locf = mse(fl)
        if locf is None:
            print(f"  MISSING baseline for {name}: {fl} -- dataset omitted from the figure")
            continue
        ys = []
        for pats in (fa, fe, fc, fm):
            v = mse(pats) if pats else None
            ys.append(np.nan if v is None else (locf - v) / locf * 100)
        if np.all(np.isnan(ys)):
            print(f"  MISSING every model for {name} -- dataset omitted"); continue
        miss = [xs[i].replace("\n", " ") for i, y in enumerate(ys) if np.isnan(y)]
        if miss:
            print(f"  {name}: no run for {', '.join(miss)} -- drawn as a GAP")
        hero = (name == "UMM")
        # NaN, not dropped: dropping the index makes matplotlib join ETS straight to `learned`
        # across the foundation-model tick, drawing a value at exactly the x we claim to omit.
        ax.plot(range(4), ys, "-o", color=BLUE, lw=2.4 if hero else 1.2, ms=6 if hero else 4,
                alpha=1.0 if hero else 0.45, zorder=4 if hero else 3)
        last = np.where(~np.isnan(ys))[0][-1]
        ends.append([ys[last], name, hero])
    # declutter the right-hand labels: four datasets finish within ~2 points of each other, so
    # place them in order and push each up to a minimum spacing rather than letting them overlap
    # Displaced labels get a LEADER back to their own endpoint. Pushing four labels apart without
    # one leaves each sitting nearer a neighbour's line than its own, which is worse than overlap:
    # it attributes the wrong number to the wrong dataset.
    ends.sort(key=lambda r: r[0])
    placed = [list(e) + [e[0]] for e in ends]          # [y_label, name, hero, y_true]
    gap = 1.9
    for i in range(1, len(placed)):
        if placed[i][0] - placed[i - 1][0] < gap:
            placed[i][0] = placed[i - 1][0] + gap
    for y, name, hero, y_true in placed:
        moved = abs(y - y_true) > 0.4
        ax.annotate(name, xy=(3.02, y_true), xytext=(3.30, y), textcoords="data",
                    fontsize=10 if hero else 9, color=BLUE, va="center",
                    weight="bold" if hero else "normal", alpha=1.0 if hero else 0.75,
                    arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8, alpha=0.55,
                                    shrinkA=0, shrinkB=2) if moved else None)
    top = max(p[0] for p in placed); lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, top + 2.0))                # displaced labels must not clip
    # questionnaire: macro-F1 relative to persistence
    # Questionnaire, macro-F1 against persistence 0.8705. Sources, since these are stated here
    # rather than read from docs/tables like the physiology rows:
    #   arima/ets  05_baselines.py --tsf on gtsq_pred_24h_gap6h_clin_cult
    #   fm         Chronos-Bolt base, zero-shot (the foundation-model appendix table)
    #   learned    validation-selected best-macro-F1 gate, five-seed mean
    P, GQ = 0.8705, dict(arima=0.8513, ets=0.8347, fm=0.8496, learned=0.8740)
    gq = [(GQ[k] - P) / P * 100 for k in ("arima", "ets", "fm", "learned")]
    ax.plot(range(4), gq, "-o", color=RED, lw=2.6, ms=7, zorder=4)
    ax.annotate("questionnaire\n(same patients)", (3, gq[-1]), textcoords="offset points",
                xytext=(9, -4), fontsize=10, color=RED, weight="bold")

    ax.set_xticks(range(4)); ax.set_xticklabels(xs, fontsize=11, color=INK)
    ax.set_ylabel("improvement over copying (%)", fontsize=12, color=INK)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.set_xlim(-0.4, 5.15)
    # anchor the word to the zero LINE, which is what it names, not to the corner of the axes
    ax.annotate("copying", (-0.33, 0), textcoords="offset points", xytext=(0, 5),
                fontsize=10, color=INK, style="italic", ha="left")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_regimes.{ext}"), dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}/fig_regimes.{{pdf,png}}")
    print("  continuous forecasts: MSE reduction vs LOCF; questionnaire: macro-F1 vs persistence")


def cmd_permimp():
    """Aggregate the permutation-importance logs over seeds (grouped and per-variable).

    Reads exp_fixnn/logs/permute_importance*.log, which PERMUTE=1 run_analysis.sh writes. Reports a
    variable or group as interpretable ONLY if its mean effect exceeds twice its own seed standard
    deviation -- on this budget (physiology contributes +0.0167 change-accuracy in total) almost
    nothing individually does, and printing an unqualified ranking would invite reading noise.
    """
    import glob, collections
    # ONE log only. Globbing them all merged runs made under DIFFERENT group definitions: the
    # original grouping had haemodynamic(11) and an `other` bucket, the corrected one has
    # haemodynamic(17) and no `other`. Averaging those under a shared name reported seeds=10 for a
    # 5-seed experiment and kept a deleted bucket in the ranking. KG_PERMUTE_LOG selects explicitly;
    # the default is the newest, and every log found is listed so a stale pick is visible.
    logs = sorted(glob.glob(os.path.join(HERE, "logs", "permute_importance*.log")),
                  key=os.path.getmtime)
    if not logs:
        print("  no permute_importance*.log -- run PERMUTE=1 sbatch exp_fixnn/run_analysis.sh"); return
    pick = os.environ.get("KG_PERMUTE_LOG")
    log = os.path.join(HERE, "logs", pick) if pick else logs[-1]
    if not os.path.exists(log):
        raise SystemExit(f"ERROR: KG_PERMUTE_LOG={pick} not found in {os.path.join(HERE, 'logs')}")
    print(f"  reading {os.path.basename(log)}"
          + (f"   (others present, NOT merged: {', '.join(os.path.basename(x) for x in logs if x != log)})"
             if len(logs) > 1 else ""))
    txt = open(log).read()
    grp = collections.defaultdict(list); var = collections.defaultdict(list)
    row = re.compile(r'^\s+(\S+?)(\[\d+\])?\s+\(all 45 items\)\s+([\d.]+)\s+([\d.]+)\s+'
                     r'([+-][\d.]+)\s+chgF1\s+([\d.]+)\s+->\s+([\d.]+)\s+([+-][\d.]+)', re.M)
    for m in row.finditer(txt):
        (grp if m.group(2) else var)[m.group(1)].append((float(m.group(5)), float(m.group(8))))
    for label, d in (("GROUPS (permuted jointly)", grp), ("SINGLE VARIABLES", var)):
        if not d:
            continue
        print(f"\n  {label} -- mean over seeds; negative = the model used it")
        print(f"  {'name':18s}{'seeds':>6}{'d macro-F1':>13}{'d change-F1':>14}{'sd':>9}  verdict")
        rows = []
        for k, v in d.items():
            a = np.array(v)
            rows.append((k, len(v), a[:, 0].mean(), a[:, 1].mean(), a[:, 1].std()))
        for k, n, dm, dc, sd in sorted(rows, key=lambda r: r[3]):
            ok = "interpretable" if (sd > 0 and dc < -2 * sd) else "within noise"
            print(f"  {k:18s}{n:>6}{dm:>+13.4f}{dc:>+14.4f}{sd:>9.4f}  {ok}")
        clear = sum(1 for r in rows if r[4] > 0 and r[3] < -2 * r[4])
        print(f"  {clear}/{len(rows)} clear twice their own seed s.d.")


def cmd_circ():
    """E9: circularity for GTSQ under every oracle definition the data supports.

    The paper defines it as F1(oracle) - F1(persistence) with the oracle given the target day's
    physiology, but our causal models stop 6h BEFORE the form, so that oracle mixes two things:
    label-recoverability-from-chart (circularity proper) and simple recency of the physiology. The
    gap runs let us separate them: +6h is causal, 0h removes the gap, -6h peeks past the form.
    """
    defs = [("persistence (reference)", None),
            ("causal   (gap +6h)", "clin_k0"),
            ("gap 0h   (gap removed)", "extra/gap0"),
            ("oracle   (gap -6h)", "extra/gapm6")]
    vals = {}
    print("  macro-F1 of the same gated model at each physiology gap (mask=both)")
    for lab, sub in defs:
        if sub is None:
            vals[lab] = 0.8705
            print(f"  {lab:26s} {0.8705:.4f}   [training-free]"); continue
        r = agg(sub)
        if r is None:
            print(f"  {lab:26s} (missing: {sub})"); continue
        vals[lab] = r[0][0]
        warn = "" if r[2] >= len(CANON_SEEDS) else f"  <<< ONLY {r[2]}/{len(CANON_SEEDS)} SEEDS"
        print(f"  {lab:26s} {r[0][0]:.4f} +- {r[1][0]:.4f}  [n={r[2]}]{warn}")
    pers = vals.get("persistence (reference)")
    cau = vals.get("causal   (gap +6h)")
    g0 = vals.get("gap 0h   (gap removed)")
    om = vals.get("oracle   (gap -6h)")
    print("\n  circularity under each candidate definition:")
    if om is not None and pers is not None:
        print(f"    (a) oracle - persistence      {om - pers:+.4f}   <- as published")
    if g0 is not None and cau is not None:
        print(f"    (b) gap-removed - causal      {g0 - cau:+.4f}   <- recency only, no peek")
    if om is not None and cau is not None:
        print(f"    (c) oracle - causal           {om - cau:+.4f}   <- peek beyond the form")
    if cau is not None and pers is not None:
        print(f"    gain (causal - persistence)   {cau - pers:+.4f}")


def cmd_subset():
    p8 = _mod("p8", "08_probes.py")
    from models.gated_predictor import build_feature_slices
    N = np.load(os.path.join(SPLIT, "norm_stats.npz"), allow_pickle=True)
    FT = [str(x) for x in N["gtsq_ftypes"]]; NC = [int(x) for x in N["gtsq_nclasses"]]
    FE = [str(x) for x in N["gtsq_features"]]; gm = N["gtsq_means"].astype(float)
    gs = N["gtsq_stds"].astype(float); G = len(FT)
    d = np.load(os.path.join(SPLIT, "test.npz"), allow_pickle=True)
    gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
    im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
    I = np.round(gi * gs + gm).astype(int); T = np.round(d["gtsq_out"] * gs + gm).astype(int)
    fo = ((im == 1) & (d["gtsq_out_mask"] == 1)).all(1)
    I, T = I[fo], T[fo]; nchg = (I != T).sum(1)
    subs = [("all", nchg >= 0), (">0", nchg >= 1), (">1", nchg >= 2), (">3", nchg >= 4)]

    def metr(P, sel):
        Ts, Ps = T[sel], P[sel]
        acc = (Ps == Ts).mean()
        f1 = np.mean([f1_score(Ts[:, j], np.clip(Ps[:, j], 0, NC[j] - 1), average="macro",
                               labels=list(range(NC[j])), zero_division=0) for j in range(G)])
        return acc, f1

    # baselines per subset
    Pm = I.copy()
    for j in range(G):
        if "akut" in FE[j].lower():
            Pm[:, j] = np.where(np.clip(I[:, j], 0, 2) == 1, 2, np.clip(I[:, j], 0, 2))

    def predict(subdir):
        runs = [r for r in sorted(glob.glob(os.path.join(HERE, _tree_for(subdir), subdir, "*")))
                if _bal_ok(r)]
        prob = [None] * G; used = 0
        for r in runs:
            if not os.path.exists(os.path.join(r, "model.pytorch")):
                continue
            try:
                model, cfg, norm, sd = p8.reconstruct(r)
                tn = np.load(os.path.join(sd, "test.npz"), allow_pickle=True); sp = {k: tn[k] for k in tn.files}
                lg, _ = p8._forward_pred_gate(model, sp, getattr(cfg, "use_clinical", False),
                                              configs=cfg, norm=norm)
            except (Exception, SystemExit):
                continue
            slices, _ = build_feature_slices(list(zip(norm["gtsq_ftypes"], norm["gtsq_nclasses"])))
            for j in range(G):
                a, b = p8._slice_bounds(slices[j]); L = lg[:, a:b]
                pr = (1 / (1 + np.exp(-L[:, 0])) if FT[j] == "binary"
                      else (lambda e: e / e.sum(1, keepdims=True))(np.exp(L - L.max(1, keepdims=True))))
                prob[j] = pr if prob[j] is None else prob[j] + pr
            used += 1
        if used == 0:
            return None
        P = np.zeros((len(fo), G), int)
        for j in range(G):
            pp = prob[j] / used
            P[:, j] = (pp > 0.5).astype(int) if FT[j] == "binary" else pp.argmax(1)
        return P[fo]

    # reconstruct once, reuse for both metric tables
    preds = [("persistence", I), ("markov(new->acute)", Pm)]
    for k in (0, 1, 2, 4):
        P = predict(f"clin_k{k}")
        preds.append((f"clin trained >={k}", P))

    hdr = "".join(f"{lab:>10}" for lab, _ in subs)
    for mi, mname in [(1, "macro-F1"), (0, "accuracy")]:
        print(f"\n=== {mname} · each predictor on every subset; baselines ON THAT SUBSET "
              f"(fully-observed n={int(fo.sum())}, 5-seed ensemble) ===")
        print(f"  {'predictor':22s}{hdr}")
        for i, (name, P) in enumerate(preds):
            if P is None:
                print(f"  {name:22s}   (missing)"); continue
            print(f"  {name:22s}" + "".join(f"{metr(P, s)[mi]:>10.4f}" for _, s in subs))
            if i == 1:
                print("  " + "-" * 62)


matplotlib.use("Agg")


# --- validated palette steps (references/palette.md) ---
BLUE, RED, GRAY = "#2a78d6", "#d03b3b", "#898781"      # diverging poles + neutral
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7"

# target, volatility %, persistence F1, best causal ML F1, oracle F1 (None = not run), note,
# (dx, dy, ha) label offset for panel A -- set per point to avoid collisions at the y=0 cluster
ROWS = [
    # All five rows recomputed 2026-08-07 on the both-days-observed mask with 5 seeds. The old
    # table was assembled target-observed; only SOFA moved materially (0.6007 -> 0.6244) because
    # it is derived from six lab components with real missingness, so a large share of its cells
    # are target-observed but not pair-observed. AKI, organ support and eICU match to 4 decimals.
    # Labelled with the component count: the UMM SOFA point excludes CNS (as in the Paper-1
    # analysis, since GCS needs a verbal score that sedated patients do not have), so the two
    # SOFA rows are NOT the same label and must not be read as one target measured twice.
    ("MIMIC SOFA (6)",    29.86, 0.6347, 0.6709, 0.8986, "computed from labs",      (0, 15, "center")),
    ("MIMIC AKI/KDIGO",   20.21, 0.6809, 0.6946, 0.7314, "computed from creatinine", (0, 15, "center")),
    ("MIMIC organ support", 7.14, 0.8696, 0.9045, 0.9488, "therapy decision",       (0, 15, "center")),
    # UMM GTSQ, refreshed 2026-08-07: persistence 0.8705 and causal 0.8701 are the common gated
    # architecture (clin_k0) used for every target here, NOT the searched winner hp09 (0.8740);
    # the oracle 0.8523 is gradient boosting on the gap -6h split, matching the other rows.
    ("UMM GTSQ",           3.8, 0.8705, 0.8701, 0.8523, "physician judgement",     (14, -34, "left")),
    ("eICU diagnosis",     0.32, 0.9877, 0.9873, 0.9877, "coded diagnosis",        (-4, 12, "left")),
]


def cmd_figure():
    """Volatility x circularity. KG_FIG_PANELS=circ emits ONLY the circularity panel.

    The paper keeps just that panel: the volatility panel makes a negative point (gain does not track
    volatility) that the text states in one clause, and a full-width two-panel float is expensive at
    ten pages. SURFACE is set to white for the single-panel version -- the off-white ground reads as
    a grey box when the figure is not full-width.
    """
    os.makedirs(OUT, exist_ok=True)
    global SURFACE
    only_circ = os.environ.get("KG_FIG_PANELS", "") == "circ"
    if only_circ:
        SURFACE = "#ffffff"
        fig, b = plt.subplots(1, 1, figsize=(6.0, 4.6), facecolor=SURFACE)
        a = b.figure.add_axes([0, 0, 1e-6, 1e-6]); a.set_visible(False)   # keep panel-A code inert
    else:
        fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.6), facecolor=SURFACE)
    for ax in ((b,) if only_circ else (a, b)):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=12)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.axhline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)

    def colour(gain):                       # diverging: ML wins (blue) / ties (gray)
        return BLUE if gain > 0.005 else GRAY

    # ---- Panel A: volatility vs ML gain ----
    for name, vol, pers, ml, orc, note, (dx, dy, ha) in ROWS:
        g = ml - pers
        a.scatter(vol, g, s=150, color=colour(g), edgecolor=SURFACE, linewidth=1.6, zorder=3)
        a.annotate(f"{name}\n({note})", (vol, g), textcoords="offset points",
                   xytext=(dx, dy), ha=ha, fontsize=11, color=INK)
    a.set_xlabel("volatility — day-to-day change rate (%)", fontsize=13, color=INK)
    a.set_ylabel("ML gain over persistence (Δ macro-F1)", fontsize=13, color=INK)
    a.set_title("A. Gain does NOT track volatility", fontsize=14, color=INK, loc="left")
    _v = np.array([r_[1] for r_ in ROWS]); _g = np.array([r_[3] - r_[2] for r_ in ROWS])
    a.text(0.97, 0.06, f"r = {np.corrcoef(_v, _g)[0, 1]:.2f}", transform=a.transAxes, ha="right",
           fontsize=13, color=MUTED, style="italic")
    a.set_ylim(-0.012, 0.075)
    a.set_ylim(-0.045, 0.085); a.set_xlim(-4, 36)

    # ---- Panel B: circularity vs ML gain ----
    pts = [(n, o - p, m - p) for n, v, p, m, o, note, _ in ROWS if o is not None]
    xs = np.array([c for _, c, _ in pts]); ys = np.array([g for _, _, g in pts])
    fit = np.polyfit(xs, ys, 1); xr = np.linspace(-0.02, 0.36, 50)
    b.plot(xr, np.polyval(fit, xr), color=GRAY, lw=1.5, ls=(0, (5, 4)), zorder=2)
    # Label placement is given EXPLICITLY as (dx, dy, ha) per point, in the order of `pts`
    # (SOFA, AKI, organ support, GTSQ, eICU). It used to derive `ha` from the sign of dx, which made
    # every position two coupled decisions instead of one and produced two defects at once: SOFA
    # (x=0.264, ha=right) grew leftwards into organ support (x=0.079, ha=left) growing rightwards,
    # and GTSQ (x=-0.018, ha=right) grew off the left spine. Anchoring is now stated, not inferred.
    # The crowded region is the pair near circularity 0 and the two high-gain points at similar y;
    # they are separated by sending SOFA and GTSQ BELOW their markers and the rest above.
    # The two tied targets sit almost on top of each other at circularity ~0 (GTSQ -0.018, eICU
    # +0.000, both at gain ~0), and the labels are wide relative to the axis -- "MIMIC AKI/KDIGO"
    # alone spans nearly half the x-range. No offset pair separates them AND keeps them inside the
    # spines, so those two get a leader line and are moved clear of the cluster instead. The fourth
    # field says whether a leader is drawn; the three well-separated points do not need one.
    offs = ([(0, -26, "center", False), (12, -18, "left", False), (0, 15, "center", False),
             (0, -40, "center", True), (60, -18, "left", True)] if only_circ
            else [(0, 15, "center", False), (8, -20, "left", False), (0, 15, "center", False),
                  (10, 12, "left", False), (10, -20, "left", False)])
    assert len(offs) == len(pts), f"label offsets ({len(offs)}) must match points ({len(pts)})"
    for (name, circ, g), (dx, dy, ha_, lead) in zip(pts, offs):
        b.scatter(circ, g, s=150, color=colour(g), edgecolor=SURFACE, linewidth=1.6, zorder=3)
        b.annotate(name, (circ, g), textcoords="offset points", xytext=(dx, dy), ha=ha_,
                   fontsize=11, color=INK, zorder=4,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                   shrinkA=2, shrinkB=7) if lead else None)
    r = float(np.corrcoef(xs, ys)[0, 1])
    b.text(0.97, 0.06, f"r = {r:.2f}", transform=b.transAxes, ha="right", fontsize=13,
           color=MUTED, style="italic")
    b.set_xlabel("circularity  =  F1(target-day oracle) − F1(persistence)", fontsize=13, color=INK)
    b.set_ylabel("ML gain over persistence (Δ macro-F1)", fontsize=13, color=INK)
    b.set_title("Gain tracks circularity" if only_circ else "B. Gain tracks circularity",
                fontsize=14, color=INK, loc="left")
    b.set_ylim(-0.026, 0.078); b.set_xlim(-0.075, 0.42)

    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, color=BLUE,
                               label="ML beats persistence"),
                        Line2D([], [], marker="o", ls="", ms=9, color=GRAY,
                               label="ML ties persistence")],
               loc="lower center", ncol=2, frameon=False, fontsize=12, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    stem = "fig_circularity" if only_circ else "fig_volatility_circularity"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=200,
                    facecolor=SURFACE)
    print(f"wrote {OUT}/{stem}.{{pdf,png}}")   # was hardcoded to the two-panel stem
    print(f"\n{'target':22s}{'vol%':>7}{'persist':>9}{'bestML':>8}{'gain':>8}{'circ':>8}")
    for n, v, p, m, o, _, _lo in ROWS:
        print(f"{n:22s}{v:>7.2f}{p:>9.4f}{m:>8.4f}{m-p:>+8.4f}"
              + (f"{o-p:>+8.4f}" if o is not None else f"{'n/a':>8}"))

def cmd_ladder():
    """5-seed mean +- std for every configuration group in the campaign."""
    # Baselines MUST share the model rows' masking convention or every delta a reader computes is
    # wrong by the convention gap. These are the both-days-observed values (250,536 test cells),
    # recomputed 2026-08-05; the previous target-observed pair (0.8687/0.9614, 0.8696/0.9632) sat
    # above mask=both model rows and understated persistence by 0.0018 macro-F1.
    print("reference: persistence (corrected, both-days-observed) F1 0.8705  acc 0.9620  chg-acc 0.0000")
    print("           persistence + neu->acute                    F1 0.8714  acc 0.9638  chg-acc 0.0475\n")
    print("=== STAGE 4 -- retrained per change-count cohort ===")
    print(" primary GTSQ+clin+cultures gate:")
    for k in (0, 1, 2, 4):
        line(f"  train >={k} changes", f"clin_k{k}")
    print(" N-matched control (random subset of the full cohort):")
    for k in (1, 2, 4):
        line(f"  ctrl @ n(>={k})", f"clin_ctrl_k{k}")
    print(" reference GTSQ-only gate:")
    for k in (0, 1, 2, 4):
        line(f"  train >={k} changes", f"noclin_k{k}")
    print("\n=== STAGE 6 -- architecture grid ===")
    for enc in ("gru", "transformer", "cnn"):
        for fuse in ("concat", "crossattn", "film"):
            line(f"{enc} x {fuse}", f"grid/{enc}_{fuse}")
    print("\n=== STAGE 5 -- detectors (override sweep -> eval_deploy.py override) ===")
    for w in (1, 3, 5):
        for fg in (0, 2):
            line(f"detector pw{w} fg{fg}", f"detector_pw{w}_fg{fg}")
    print("\n=== MEETING ASKS -- gap / history / window / encoder / cultures ===")
    line("  gap 0h", "extra/gap0"); line("  gap -6h (target-day)", "extra/gapm6")
    for k in (3, 7):
        line(f"  clin  K={k}", f"extra/clin_h{k}"); line(f"  noclin K={k}", f"extra/noclin_h{k}")
    line("  72h EHR window", "extra/seq72")
    line("  GTSQ enc depth 1", "extra/encL1"); line("  GTSQ enc depth 3", "extra/encL3")
    line("  cultures: none", "nocult_k0"); line("  cultures: organ-mapped", "cultorgan_k0")
    print("\n=== HP SEARCH -> exp_fixnn/hp_search.py report (VALIDATION-selected) ===")


# --------------------------------------------------------------------------------------------------
# cmd_table -- the big feature x model LaTeX table (colour-coded), written to docs/
# --------------------------------------------------------------------------------------------------
FEAT_GROUP = [("Klinisch", "i01000_"), ("Klinisch", "i02000_"), ("Klinisch", "i03000_"),
              ("Klinisch", "i04000_"), ("Lokalisation", "i05100_"), ("Organ", "i11100_")]

TABLE_MODELS = [                       # (column header, subpath under 02_models/, short note)
    ("GTSQ gate",       "noclin_k0",          "GTSQ only"),
    ("$+$phys",         "clin_k0",            "GTSQ + physiology + cultures"),
    ("$+$phys $K3$",    "extra/clin_h3",      "3 days of questionnaire history"),
    ("CNN$\\times$xatt", "grid/cnn_crossattn", "best architecture-grid cell"),
    ("searched",        None,                 "HP-search winner (m2uni\\_gru\\_f1\\_markovt50\\_hp09)"),
]
# The search winner is convention-dependent: hp19 won under target-observed, hp09 under the
# both-days mask the paper now reports. KG_HP_WINNER overrides for reproducing older tables.
HP_WINNER = os.environ.get("KG_HP_WINNER", "hp/m2uni_gru_f1_markovt50_hp09")


def _perfeat(subpath):
    """Seed-averaged per-feature test macro-F1 for one run group, honouring BALANCE."""
    out = {}
    for r in glob.glob(os.path.join(HERE, _tree_for(subpath), subpath, "*", "gtsq_metrics_test.json")):
        if not _bal_ok(os.path.dirname(r)):
            continue
        for k, v in json.load(open(r)).items():
            if isinstance(v, dict) and "f1" in v:
                out.setdefault(k, []).append(v["f1"])
            elif isinstance(v, dict) and "f1_macro" in v:
                out.setdefault(k, []).append(v["f1_macro"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def _baseline_perfeat():
    """persistence and persistence+rule, per feature, on the test split."""
    sys.path.insert(0, os.path.dirname(HERE))
    import gtsq_eval as ge
    S = os.path.join(HERE, "01_split_sets", "gtsq_pred_24h_gap6h_clin_cult")
    n = np.load(os.path.join(S, "norm_stats.npz"), allow_pickle=True)
    fe = [str(x) for x in n["gtsq_features"]]; ft = [str(x) for x in n["gtsq_ftypes"]]
    nc = [int(x) for x in n["gtsq_nclasses"]]
    gm, gs = n["gtsq_means"].astype(float), n["gtsq_stds"].astype(float)
    d = np.load(os.path.join(S, "test.npz"), allow_pickle=True)
    gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
    I = ge.denorm_to_class(gi, gm, gs, ft, nc)
    T = ge.denorm_to_class(d["gtsq_out"], gm, gs, ft, nc); M = d["gtsq_out_mask"] == 1
    # BOTH-DAYS-OBSERVED, to match the model columns. These baselines are differenced against
    # per-feature numbers harvested from run jsons trained and scored under mask=both; scoring the
    # baselines target-observed instead made every coloured delta in tab_feature_model straddle the
    # two conventions. 513 cells separate them, which is small in aggregate and not small per item.
    im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
    M = M & (im == 1)
    R = I.copy()
    for j, f in enumerate(fe):
        if "akut" in f.lower():
            R[:, j] = np.where(I[:, j] == 1, 2, I[:, j])
    chg = {}
    out = {}
    for nm, P in (("persistence", I), ("rule", R)):
        pf = ge.per_feature_report(P, I, T, M, ft, nc)["per_feat"]
        out[nm] = {fe[g]: v["f1"] for g, v in pf.items()}
        if nm == "persistence":
            for g, v in pf.items():
                chg[fe[g]] = (v["n_changed"], v["n_total"])
    return fe, ft, nc, out, chg


# The instrument is German; the paper is English. Item labels are translated here so the appendix
# table reads in one language, with the German original kept in the caption's note rather than in
# every row (the table is already 9 columns wide at \scriptsize).
DE_EN = {
    "Klinisches Gesamtbild": "Clinical impression", "Gruppe": "Severity ranking",
    "Arbeitsdiagnose": "Working diagnosis", "Infektionsverdacht": "Infection suspicion",
    "Lunge": "Lung", "Niere": "Kidney", "Gehirn": "Brain", "Herz": "Heart", "Leber": "Liver",
    "Gerinnungssystem": "Coagulation", "Knochenmark": "Bone marrow",
    "Gastrointestinal": "Gastrointestinal",
    "chronisch": "chronic", "Endokarditis": "Endocarditis", "Haut": "Skin",
    "Gelenke / Knochen": "Joints / bones", "Gelenke": "Joints", "Knochen": "Bones",
    "Weichteile": "Soft tissue", "Harnwege": "Urinary tract", "Atemwege": "Respiratory tract",
    "Katheter": "Catheter", "Bauchraum": "Abdomen", "Wunde": "Wound", "Blut": "Blood",
    "ZNS": "CNS", "unbekannt": "unknown", "Sonstige": "Other", "Sonstiges": "Other",
    "Blood-stream-infection": "Bloodstream infection", "Capillary Leak": "Capillary leak",
    # infection-localisation block (lower-case in the instrument)
    "abdominell": "abdominal", "thorakal": "thoracic", "urogenital": "urogenital",
    "intrakraniell / meningeal": "intracranial / meningeal", "intrakraniell": "intracranial",
    "meningeal": "meningeal", "katheterassoziiert": "catheter-associated",
}


def _tex_name(f):
    """Human-readable, English item label for the table's first column."""
    s = re.sub(r"^i\d+_", "", f)
    s = s.replace("_", " ").replace('akut oder "neu"', "acuity").replace("Ursache für Dysfunktion", "cause")
    s = s.replace("Lokalisation  ", "").replace("(0|1|2)", "")
    # longest key first so 'Gelenke / Knochen' wins over 'Gelenke', and word-bounded so 'Herz' does
    # not rewrite a longer compound that merely starts with it.
    for de in sorted(DE_EN, key=len, reverse=True):
        s = re.sub(r"(?<![A-Za-zÄÖÜäöüß])%s(?![A-Za-zÄÖÜäöüß])" % re.escape(de), DE_EN[de], s)
    return s.replace("&", "\\&").replace("%", "\\%").strip()


def cmd_table():
    """Emit docs/tab_feature_model.tex -- 45 items x model families, colour-coded by delta vs the rule."""
    fe, ft, nc, base, chg = _baseline_perfeat()
    cols = []
    for hdr, sub, note in TABLE_MODELS:
        cols.append((hdr, _perfeat(sub if sub else HP_WINNER), note))
    rows = []
    for j, f in enumerate(fe):
        r = {"feat": f, "type": ft[j], "k": nc[j],
             "chg": chg.get(f, (0, 1))[0] / max(chg.get(f, (0, 1))[1], 1),
             "pers": base["persistence"].get(f), "rule": base["rule"].get(f)}
        for hdr, d, _ in cols:
            r[hdr] = d.get(f)
        rows.append(r)

    def cell(v, ref):
        """Colour a cell by its signed distance from the rule, in F1 points."""
        if v is None:
            return "--"
        if ref is None:
            return f"{v:.3f}"
        d = v - ref
        if d > 0.02:    c = "gain!55"
        elif d > 0.005: c = "gain!28"
        elif d < -0.02: c = "loss!55"
        elif d < -0.005:c = "loss!28"
        else:           c = "flat!30"
        return f"\\cellcolor{{{c}}}{v:.3f}"

    L = []
    ncol = 5 + len(cols)                     # item, k, chg%, pers, rule, then one per model
    A = "l" + "r" * (ncol - 1)
    L.append("% GENERATED by exp_fixnn/report.py table -- do not edit by hand")
    # the grid is 10 columns wide; tighten the gutters and the body font so it fits \onecolumn
    L.append("\\begingroup\\setlength{\\tabcolsep}{3pt}\\scriptsize")
    L.append("\\begin{longtable}{%s}" % A)
    L.append("\\caption{Per-item test macro-F1 for every model family. Cells are shaded by their distance "
             "from the \\emph{persistence${+}$rule} column: green $>{+}0.005$, red $<{-}0.005$, grey within "
             "$\\pm0.005$; the darker shade marks $|\\Delta|>0.02$. Items are grouped by questionnaire "
             "domain and sorted within a group by day-to-day change rate. $k$ = number of classes, "
             "chg\\% = fraction of observed day-pairs on which the item changes.}"
             "\\label{tab:featmodel}\\\\")
    hdr = ["item", "$k$", "chg\\%", "pers.", "\\textbf{rule}"] + [c[0] for c in cols]  # 5 + models
    L.append("\\toprule " + " & ".join(hdr) + " \\\\ \\midrule")
    L.append("\\endfirsthead")
    L.append("\\toprule " + " & ".join(hdr) + " \\\\ \\midrule")
    L.append("\\endhead")
    L.append("\\midrule \\multicolumn{%d}{r}{\\emph{continued on next page}}\\\\ \\endfoot" % len(hdr))
    L.append("\\bottomrule \\endlastfoot")
    for dom, pre in (("Clinical judgement", ("i01000_", "i02000_", "i03000_", "i04000_")),
                     ("Infection localisation (binary)", ("i05100_",)),
                     ("Organ systems", ("i11100_", "i09_"))):
        sub = [r for r in rows if r["feat"].startswith(tuple(pre))]
        sub.sort(key=lambda r: -r["chg"])
        L.append("\\multicolumn{%d}{l}{\\textbf{%s}}\\\\" % (len(hdr), dom))
        for r in sub:
            cells = [_tex_name(r["feat"])[:26], str(r["k"]), f"{100*r['chg']:.1f}",
                     f"{r['pers']:.3f}" if r["pers"] is not None else "--",
                     f"\\textbf{{{r['rule']:.3f}}}" if r["rule"] is not None else "--"]
            cells += [cell(r[c[0]], r["rule"]) for c in cols]
            L.append(" & ".join(cells) + " \\\\")
        L.append("\\addlinespace")
    L.append("\\end{longtable}")
    L.append("\\endgroup")
    # The three prefix blocks must cover every item. Capillary leak (i09_) was silently outside them
    # and the table shipped 44 rows under a caption that said 45.
    covered = sum(1 for r in rows if r["feat"].startswith(
        ("i01000_", "i02000_", "i03000_", "i04000_", "i05100_", "i11100_", "i09_")))
    if covered != len(rows):
        missing = [r["feat"] for r in rows if not r["feat"].startswith(
            ("i01000_", "i02000_", "i03000_", "i04000_", "i05100_", "i11100_", "i09_"))]
        raise SystemExit(f"ERROR: {len(rows)-covered} item(s) match no block prefix: {missing}")
    dst = os.path.join(os.path.dirname(os.path.dirname(HERE)), "docs", "tab_feature_model.tex")
    open(dst, "w").write("\n".join(L) + "\n")
    print(f"wrote {dst}  ({len(rows)} items x {len(cols)} model columns)")
    for hdr_, d, note in cols:
        print(f"  column {hdr_:18s} {len(d):3d} items   [{note}]")


# --------------------------------------------------------------------------------------------------
# cmd_ablation -- Appendix "Input ablations" as generated LaTeX: all four metrics, no hand-copied cells
# --------------------------------------------------------------------------------------------------
ABLATION_BLOCKS = [
    ("Physiology gap relative to the target form", [
        ("$+6$\\,h (causal, default)", "clin_k0"),
        ("$0$\\,h",                    "extra/gap0"),
        ("$-6$\\,h (target-day oracle)", "extra/gapm6")]),
    ("Questionnaire history", [
        ("$K{=}1$ (default)", "clin_k0"),
        ("$K{=}3$",           "extra/clin_h3"),
        ("$K{=}7$",           "extra/clin_h7")]),
    ("Physiology window", [
        ("24\\,h (default)", "clin_k0"),
        ("72\\,h",           "extra/seq72")]),
    ("Microbiology", [
        ("none",                    "nocult_k0"),
        ("blood cultures (3 vars)", "clin_k0"),
        ("organ-mapped (17 vars)",  "cultorgan_k0")]),
    ("GTSQ-encoder depth", [
        ("1 (linear)",  "extra/encL1"),
        ("2 (default)", "clin_k0"),
        ("3 (deep)",    "extra/encL3")]),
    ("What the wrapper is worth", [
        ("physiology only, no questionnaire", "extra/onlyclin"),
        ("no gate, no persistence path",      "extra/nogate"),
        ("gate on copy (default)",            "clin_k0")]),
]


def cmd_ablation():
    """Emit docs/tab_ablation.tex from the run directories -- every cell has all four metrics."""
    rows = []
    for title, items in ABLATION_BLOCKS:
        rows.append(("HEADER", title, None))
        for lab, sub in items:
            rows.append(("ROW", lab, agg(sub)))
    L = ["% GENERATED by exp_fixnn/report.py ablation -- do not edit by hand",
         "\\begin{center}\\small",
         "\\begin{tabular}{lcccc}", "\\toprule",
         "Variant & macro-F1 & Acc & Acc$_c$ & F1$_c$\\\\", "\\midrule"]
    miss = []; under = []
    for kind, lab, a in rows:
        if kind == "HEADER":
            L.append(r"\multicolumn{5}{l}{\emph{%s}}\\" % lab); continue
        if a is None:
            L.append(r"\;\;%s & \multicolumn{4}{c}{\emph{run pending}}\\" % lab)
            miss.append(lab); continue
        m, sd, n = a
        # An under-seeded group is not a smaller sample, it is a different and usually more
        # optimistic number (see line()). Mark it in the table rather than letting it pass as a
        # five-seed result, and list it at the end so it cannot be missed.
        mark = "" if n >= len(CANON_SEEDS) else r"$^{\dagger}$"
        if mark:
            under.append(f"{lab} (n={n})")
        L.append(r"\;\;%s%s & %.4f & %.4f & %.4f & %.4f\\" % (lab, mark, m[0], m[1], m[2], m[3]))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    if under:
        L.append(r"{\footnotesize $^{\dagger}$fewer than %d seeds; not a reportable result.}"
                 % len(CANON_SEEDS))
    dst = os.path.join(os.path.dirname(os.path.dirname(HERE)), "docs", "tab_ablation.tex")
    open(dst, "w").write("\n".join(L) + "\n")
    print(f"wrote {dst}")
    print("  MISSING (rerun these):", miss or "none")
    print("  UNDER-SEEDED (flagged in the table):", under or "none")


def cmd_items():
    """Emit docs/tab_items.tex -- the 45-item inventory: type, classes, class distribution, change rate."""
    sys.path.insert(0, os.path.dirname(HERE))
    import gtsq_eval as ge
    S = os.path.join(HERE, "01_split_sets", "gtsq_pred_24h_gap6h_clin_cult")
    n = np.load(os.path.join(S, "norm_stats.npz"), allow_pickle=True)
    fe = [str(x) for x in n["gtsq_features"]]; ft = [str(x) for x in n["gtsq_ftypes"]]
    nc = [int(x) for x in n["gtsq_nclasses"]]
    gm, gs = n["gtsq_means"].astype(float), n["gtsq_stds"].astype(float)
    G = len(fe)
    ch = np.zeros(G); tot = np.zeros(G); dist = np.zeros((G, 5)); hist = []
    for sp in ("train", "val", "test"):
        d = np.load(os.path.join(S, f"{sp}.npz"), allow_pickle=True)
        gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
        im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
        I = ge.denorm_to_class(gi, gm, gs, ft, nc); T = ge.denorm_to_class(d["gtsq_out"], gm, gs, ft, nc)
        M = (im == 1) & (d["gtsq_out_mask"] == 1)
        ch += ((I != T) & M).sum(0); tot += M.sum(0)
        hist.append(((I != T) & M).sum(1))
        for g in range(G):                       # class distribution over OBSERVED day-D cells
            v = I[M[:, g], g]
            for c in range(min(nc[g], 5)):
                dist[g, c] += (v == c).sum()
    dist = dist / np.maximum(dist.sum(1, keepdims=True), 1)
    hist = np.concatenate(hist)

    # Class semantics per item, so a reader can tell what c0/c1/c2 mean without leaving the table.
    # The organ block is further split by ROLE (presence / acuity / cause / chronicity) because the
    # four roles have different class meanings and sorting all 31 organ items by change rate mixes
    # them into an unreadable list.
    SEM = {"i01000_": "0 = least severe three, 1 = neither, 2 = most severe three",
           "i02000_": "0 = improved, 1 = unchanged, 2 = worsened",
           "i03000_": "0 = neither, 1 = SIRS, 2 = sepsis, 3 = severe, 4 = septic shock",
           "i04000_": "0 = no infection, 1 = suspected, 2 = confirmed"}
    ROLES = [("present",  lambda f: f.endswith("_present"),                 "0 = no, 1 = yes"),
             ("acuity",   lambda f: "akut" in f.lower(),                    "0 = none, 1 = new, 2 = acute"),
             ("cause",    lambda f: "Ursache" in f,                         "1 = non-infectious, 2 = unclear, 3 = infectious"),
             ("chronic",  lambda f: "chronisch" in f.lower(),               "0 = no, 1 = yes")]
    blocks = [("Clinical judgement", ("i01000_", "i02000_", "i03000_", "i04000_")),
              ("Infection localisation (binary)", ("i05100_",)),
              ("Organ systems", ("i11100_", "i09_"))]
    L = ["% GENERATED by exp_fixnn/report.py items -- do not edit by hand",
         r"\begingroup\setlength{\tabcolsep}{3pt}\scriptsize",
         r"\begin{longtable}{@{}llrrrrrrr@{}}",
         r"\caption{The 45 questionnaire items: type, number of classes, the distribution over classes "
         r"on observed day-$D$ cells, and the day-to-day change rate. Grouped by block and ordered "
         r"within a block by change rate. Blank cells mean the class does not exist for that item. "
         r"Item names are translated from the German instrument; the organ block reads "
         r"\\emph{Lunge/Niere/Gehirn/Herz/Leber/Gerinnungssystem/Knochenmark} for "
         r"lung/kidney/brain/heart/liver/coagulation/bone marrow.}"
         r"\label{tab:items}\\",
         r"\toprule Item & type & $k$ & c0 & c1 & c2 & c3 & c4 & chg\%\\ \midrule",
         r"\endfirsthead",
         r"\toprule Item & type & $k$ & c0 & c1 & c2 & c3 & c4 & chg\%\\ \midrule",
         r"\endhead",
         r"\midrule \multicolumn{9}{r}{\emph{continued}}\\ \endfoot",
         r"\bottomrule \endlastfoot"]
    def _row(g):
        cells = " & ".join(f"{100*dist[g, c]:.1f}" if c < nc[g] else "" for c in range(5))
        lab = _tex_name(fe[g])[:28]
        for k, v in SEM.items():                       # clinical-judgement items carry their scale
            if fe[g].startswith(k):
                lab = f"{lab}\\,\\footnotesize({v})"
        return r"%s & %s & %d & %s & %.2f\\" % (lab, ft[g][:3], nc[g], cells,
                                                 100 * ch[g] / max(tot[g], 1))

    for title, pre in blocks:
        idx = [g for g in range(G) if fe[g].startswith(pre)]
        if title.startswith("Organ"):
            L.append(r"\multicolumn{9}{l}{\textbf{%s}}\\" % title)
            # Anything in this block matching no role must still be emitted -- capillary leak sits
            # here by prefix and has no organ role, and silently dropping it cost a row until caught.
            claimed = {g for g in idx for _, pr, _ in ROLES if pr(fe[g])}
            roles = ROLES + [("other", lambda f: True, "0 = no, 1 = yes")]
            for rname, pred, sem in roles:
                sel = ([g for g in idx if pred(fe[g])] if rname != "other"
                       else [g for g in idx if g not in claimed])
                if not sel:
                    continue
                L.append(r"\multicolumn{9}{l}{\quad\emph{%s} \footnotesize(%s)}\\" % (rname, sem))
                for g in sorted(sel, key=lambda g: -(ch[g] / max(tot[g], 1))):
                    L.append(r"\quad\quad " + _row(g))
            continue
        if title.startswith("Infection localisation"):
            title = "Infection localisation \\footnotesize(0 = site not named, 1 = named)"
        L.append(r"\multicolumn{9}{l}{\textbf{%s}}\\" % title)
        idx.sort(key=lambda g: -(ch[g] / max(tot[g], 1)))
        for g in idx:
            L.append(_row(g))
        L.append(r"\addlinespace")
    L += [r"\end{longtable}", r"\endgroup", "",
          r"\begin{center}\small",
          r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
          r"\toprule changed items per day-pair & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8+\\ \midrule"]
    cnt = [int((hist == k).sum()) for k in range(8)] + [int((hist >= 8).sum())]
    L.append("pairs & " + " & ".join(f"{c:,}".replace(",", "{,}") for c in cnt) + r"\\")
    L.append(r"\% & " + " & ".join(f"{100*c/len(hist):.1f}" for c in cnt) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    dst = os.path.join(os.path.dirname(os.path.dirname(HERE)), "docs", "tab_items.tex")
    open(dst, "w").write("\n".join(L) + "\n")
    print(f"wrote {dst}  ({G} items; mean {hist.mean():.2f} changed items per pair, median "
          f"{np.median(hist):.0f})")


# --------------------------------------------------------------------------------------------------
# cmd_frontier -- accuracy vs change-accuracy for every predictor, with the persistence+rule frontier
# --------------------------------------------------------------------------------------------------
# (label, accuracy, change-accuracy, family) + macro-F1 for the second panel
# Rows that correspond to a run group are read from TREE, so the figure cannot silently keep numbers
# from a different masking convention (every hardcoded value below was target-observed until
# 2026-08-05). FRONTIER_STATIC holds only the rows with no run dir: the two training-free rules, the
# training-free state lookup, the gradient-boosting baseline, and the zero-shot foundation models --
# each refreshed from its own script (report.py baselines, run_all.sh ONLY=baselines, run_fm.sh).
FRONTIER_DIRS = [   # (label, subpath under TREE, family)
    ("gate on rule",     "hp/m2uni_gru_f1_markovt50_hp09", "gate"),
    ("GTSQ gate",        "noclin_k0",                      "gate"),
    ("+ physiology",     "clin_k0",                        "gate"),
    ("change-tuned",     "hp/m2uniobj_gru_f1_hp10",        "gate"),
    ("no gate",          "extra/nogate",                   "nopers"),
    ("physiology only",  "extra/onlyclin",                 "nopers"),
]
FRONTIER_STATIC = [   # (label, accuracy, change-accuracy, macro-F1, family)  -- mask=both
    ("persistence",              0.9620, 0.0000, 0.8705, "rule"),
    ("persistence + rule",       0.9638, 0.0475, 0.8714, "rule"),
    ("state lookup",             0.9589, 0.0316, 0.8646, "rule"),      # bayes_floor, all 5,600 pairs
    ("gradient boosting",        0.9631, 0.1355, 0.8467, "nopers"),
    ("Chronos-Bolt base",        0.9605, 0.0132, 0.8496, "fm"),        # refreshed 2026-08-05
    ("Chronos-2",                0.9577, 0.0179, 0.8458, "fm"),        # refreshed 2026-08-05
    ("MIRA",                     0.9509, 0.0419, 0.8221, "fm"),        # refreshed 2026-08-05
]


def _frontier_rows():
    """Assemble the frontier from the model tree plus the static (no-run-dir) rows."""
    rows, f1 = [], {}
    for lab, sub, fam in FRONTIER_DIRS:
        r = agg(sub)
        if r is None:
            print(f"  [frontier] MISSING run group for '{lab}' ({sub}) -- row omitted"); continue
        m, s, n = r
        if n < len(CANON_SEEDS):
            print(f"  [frontier] '{lab}' has only {n}/{len(CANON_SEEDS)} seeds")
        rows.append((lab, m[1], m[2], fam)); f1[lab] = m[0]
    for lab, acc, chg, mf1, fam in FRONTIER_STATIC:
        rows.append((lab, acc, chg, fam)); f1[lab] = mf1
    return rows, f1


FRONTIER, FRONTIER_F1 = None, None   # populated by cmd_frontier (needs TREE/BALANCE resolved first)
FAM = {"rule": ("#0E7C8B", "o"), "gate": ("#2C875A", "s"),
       "nopers": ("#BC4238", "^"), "fm": ("#B37A0A", "D")}


def cmd_frontier(yaxis="acc"):
    """docs/figures/fig_frontier.pdf -- accuracy against change-accuracy for every predictor.

    The line through persistence and persistence+rule is the break-even frontier: a predictor is
    worth considering only if it sits above and to the right of it, i.e. catches more change without
    giving up accuracy relative to what the two fixed rules already achieve.
    """
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.5); ax.set_axisbelow(True)

    # Pareto front: a predictor is on it when nothing beats it on BOTH axes at once. This is the
    # defensible "which models are worth considering" line -- extrapolating the persistence->rule
    # segment would assume change-accuracy can keep being bought at the same price, which nothing shows.
    global FRONTIER, FRONTIER_F1
    if FRONTIER is None:
        FRONTIER, FRONTIER_F1 = _frontier_rows()
    P = [(lab, (a if yaxis == "acc" else FRONTIER_F1[lab]), ca) for lab, a, ca, _ in FRONTIER]
    front = [(ca, y) for lab, y, ca in P
             if not any((y2 >= y and c2 >= ca) and (y2 > y or c2 > ca) for _, y2, c2 in P)]
    front.sort()
    ax.plot([p[0] for p in front], [p[1] for p in front], color=MUTED, lw=1.4,
            ls=(0, (5, 4)), zorder=2)
    ax.annotate("Pareto front", (front[-2][0], front[-2][1]), textcoords="offset points",
                xytext=(10, -16), fontsize=10, color=MUTED, style="italic")
    pset = {p for p in front}

    for lab, acc0, ca, fam in FRONTIER:
        acc = acc0 if yaxis == "acc" else FRONTIER_F1[lab]
        c, mk = FAM[fam]
        above = (ca, acc) in pset
        ax.scatter(ca, acc, s=130 if above else 90, color=c, marker=mk,
                   edgecolor=SURFACE, linewidth=1.5, zorder=4 if above else 3,
                   alpha=1.0 if above else 0.55)
        ax.annotate(lab, (ca, acc), textcoords="offset points", xytext=(7, 5),
                    fontsize=9.5, color=INK if above else MUTED)
    ax.set_xlabel("change-accuracy  (fraction of changed cells predicted correctly)",
                  fontsize=12, color=INK)
    ax.set_ylabel("cell accuracy" if yaxis == "acc" else "macro-F1", fontsize=12, color=INK)
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0.70, 0.975) if yaxis == "acc" else ax.set_ylim(0.48, 0.90)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], ls="", marker=FAM[f][1], ms=8, color=FAM[f][0], label=n)
                       for f, n in (("rule", "fixed rule"), ("gate", "gated over persistence"),
                                    ("nopers", "no persistence path"), ("fm", "foundation model"))],
              loc="lower right", frameon=False, fontsize=10, labelcolor=INK)
    fig.tight_layout()
    tag = "" if yaxis == "acc" else "_f1"
    fig.savefig(os.path.join(OUT, f"fig_frontier{tag}.pdf"), facecolor=SURFACE)
    fig.savefig(os.path.join(OUT, f"fig_frontier{tag}.png"), dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}/fig_frontier{tag}.pdf   (y = {'accuracy' if yaxis=='acc' else 'macro-F1'})")
    for lab, y, ca in P:
        if (ca, y) in pset:
            print(f"  ON THE PARETO FRONT: {lab:22s} y {y:.4f}  chg-acc {ca:.4f}")




def cmd_cohort():
    """Cohort flow, patient linkage and split disjointness -- the audit behind Table 1 (O01).

    The manuscript reported 3,736 source encounters and 3,020 retained ones. The first number is the
    export; the second could not be reproduced from any split and appears to be an error. What the
    pipeline actually retains is counted here, at every step, from the same CSV the splits are built
    from -- together with the patient linkage the questionnaire export does not carry and the
    demographics export does (`lifeTimeNumber`), which is what makes the disjointness check possible
    at all.
    """
    import numpy as np, pandas as pd, glob, os
    HERE = os.path.dirname(os.path.abspath(__file__))
    KEEP = os.path.dirname(HERE)
    raw = os.path.join(os.path.dirname(KEEP), "raw_data/UMM_Export")

    ent = pd.read_csv(os.path.join(raw, "260528_UMM_gtsq_entries.csv"), sep=";", skiprows=[0, 2],
                      na_values=["NULL"], low_memory=False,
                      usecols=["encounterNumber", "charttime", "itemid", "version"])
    ent["ct"] = pd.to_datetime(ent.charttime, errors="coerce")
    forms = ent.groupby(["encounterNumber", "ct"]).size().reset_index(name="n_items")

    print("=" * 84)
    print("COHORT FLOW")
    print("=" * 84)
    print(f"  source: {len(ent):,} item rows, {ent.encounterNumber.nunique():,} encounters, "
          f"{len(forms):,} questionnaire days")
    print(f"  questionnaire versions: " +
          ", ".join(f"v{int(k)}={v:,}" for k, v in ent.version.value_counts().items()) +
          f", missing={int(ent.version.isna().sum())}")
    print(f"  items per form: median {forms.n_items.median():.0f} "
          f"(IQR {forms.n_items.quantile(.25):.0f}-{forms.n_items.quantile(.75):.0f}) of 24 raw item ids")
    # charttime is a REFERENCE time, not an entry time -- the single most important timing fact
    hh = ent.ct.dt.hour.value_counts(normalize=True)
    print(f"  charttime hour-of-day: {hh.idxmax():02d}:00 for {hh.max():.1%} of rows "
          f"-> a fixed 14:00-14:00 reference, NOT an entry or signing time")
    g = forms.sort_values(["encounterNumber", "ct"]).groupby("encounterNumber")["ct"].diff()
    g = (g.dt.total_seconds() / 3600).dropna()
    print(f"  interval between consecutive forms: median {g.median():.2f} h, "
          f"{(g.round(2) == 24.0).mean():.2%} exactly 24.00 h")

    csv = os.path.join(HERE, "gtsq_day_level_encoded_fixnn.csv")
    day = pd.read_csv(csv, low_memory=False)
    print(f"  day-level (corrected): {len(day):,} rows, {day.EncounterNumber.nunique():,} encounters")

    print("\n" + "=" * 84)
    print("SPLIT COMPOSITION AND DISJOINTNESS")
    print("=" * 84)
    dem = pd.read_csv(os.path.join(raw, "260528_UMM_demographics.csv"), sep=";", skiprows=[0, 2],
                      na_values=["NULL"], low_memory=False,
                      usecols=["encounterNumber", "lifeTimeNumber"]).dropna()
    dem["enc"] = dem.encounterNumber.astype("Int64").astype(str)
    dem["pat"] = dem.lifeTimeNumber.astype("Int64").astype(str)
    amb = dem.groupby("enc")["pat"].nunique()
    e2p = dem.drop_duplicates("enc").set_index("enc")["pat"].to_dict()
    print(f"  patient map: {len(e2p):,} encounters -> {len(set(e2p.values())):,} patients "
          f"({int((amb > 1).sum())} encounters ambiguous, first taken)")

    # --- what the correction does to the rule's error rate ------------------------------------
    # The rule rewrites `neu` to `akut`. On a cell that is `neu` on both days it would therefore be
    # WRONG -- but the correction removes exactly those cells, by rewriting the second day to
    # `akut`. So on corrected labels the rule's false-change rate on organ acuity is zero by
    # construction, not by discovery. This counts the pairs to make that explicit rather than
    # letting a reader read a structural zero as an empirical finding.
    print("\n" + "=" * 84)
    print("EFFECT OF THE neu->neu CORRECTION ON THE RULE")
    print("=" * 84)
    acu = [c for c in day.columns if c.startswith("i11100_") and c.endswith('akut oder "neu"')]
    for tag, path in (("corrected", csv),
                      ("raw", os.path.join(HERE, "gtsq_day_level_encoded_rawlab.csv"))):
        if not os.path.exists(path):
            print(f"  {tag:10s} not built yet ({os.path.basename(path)})"); continue
        d2 = pd.read_csv(path, low_memory=False)
        d2["date"] = pd.to_datetime(d2["date"], errors="coerce")
        d2 = d2.sort_values(["EncounterNumber", "date"])
        consec = d2.groupby("EncounterNumber")["date"].diff().dt.days.eq(1)
        nn = tot = 0
        for c in acu:
            prev = d2.groupby("EncounterNumber")[c].shift(1)
            nn += int((consec & prev.eq(1) & d2[c].eq(1)).sum())     # neu -> neu (rule is wrong)
            tot += int((consec & prev.eq(1)).sum())                  # day D was neu
        print(f"  {tag:10s} organ-acuity pairs with day D = neu: {tot:,};  "
              f"of these neu->neu: {nn:,} ({nn/max(tot,1):.2%})")

    for sd in sorted(glob.glob(os.path.join(HERE, "01_split_sets", "gtsq_*"))):
        parts, ok = {}, True
        for nm in ("train", "val", "test"):
            f = os.path.join(sd, f"{nm}.npz")
            if not os.path.exists(f):
                ok = False; break
            z = np.load(f, allow_pickle=True)
            enc = np.asarray(z["enc_ids"]).astype(str)
            pat = (np.asarray(z["pat_ids"]).astype(str) if "pat_ids" in z.files
                   else np.array([e2p.get(e, f"enc:{e}") for e in enc]))
            parts[nm] = (len(enc), set(enc), set(pat))
        if not ok:
            continue
        tot = sum(v[0] for v in parts.values())
        print(f"\n  {os.path.basename(sd)}  ({tot:,} day-pairs)")
        for nm, (n, e, q) in parts.items():
            print(f"    {nm:5s} {n:>7,} pairs  {len(e):>5,} encounters  {len(q):>5,} patients")
        for x, y in (("train", "test"), ("train", "val"), ("val", "test")):
            ov_e = parts[x][1] & parts[y][1]; ov_p = parts[x][2] & parts[y][2]
            flag = "  <-- LEAK" if ov_p else ""
            print(f"    overlap {x}/{y}: {len(ov_e)} encounters, {len(ov_p)} patients{flag}")




def cmd_markovmap():
    """Exactly which transitions the gate's baseline path installs (tau=0.5, min_n=10).

    The gated rows are named "gate over rule", and 03_train logs only a COUNT of overrides. Eight
    overrides and eight organ-acuity items is suggestive but not proof, and the paper must not
    describe the baseline as the one-line rule unless the table really is that rule. This replicates
    the construction in 03_train.py (lines ~926-945) on the training split and prints every
    (item, state) -> state entry, so the claim is checked rather than inferred from a count.
    """
    import numpy as np, os, sys
    HERE = os.path.dirname(os.path.abspath(__file__)); KEEP = os.path.dirname(HERE)
    sys.path.insert(0, KEEP); import gtsq_eval as ge
    sp = os.environ.get("MM_SPLIT",
                        os.path.join(HERE, "01_split_sets/gtsq_pred_24h_gap6h_clin_cult_patsplit"))
    tau = float(os.environ.get("MM_TAU", 0.5)); min_n = int(os.environ.get("MM_MINN", 10))
    n = np.load(os.path.join(sp, "norm_stats.npz"), allow_pickle=True)
    FE = [str(x) for x in n["gtsq_features"]]; NC = [int(x) for x in n["gtsq_nclasses"]]
    FT = [str(x) for x in n["gtsq_ftypes"]]
    gm = n["gtsq_means"].astype(float); gs = n["gtsq_stds"].astype(float)
    d = np.load(os.path.join(sp, "train.npz"), allow_pickle=True)
    gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
    I = ge.denorm_to_class(gi, gm, gs, FT, NC); T = ge.denorm_to_class(d["gtsq_out"], gm, gs, FT, NC)
    om = d["gtsq_out_mask"]
    print(f"baseline transition table  tau={tau}  min_n={min_n}  split={os.path.basename(sp)}")
    print(f"  {'item':46s}{'from':>6}{'to':>4}{'n':>7}{'purity':>8}")
    total = 0
    for g in range(len(FE)):
        nc = int(NC[g]); m = om[:, g] == 1
        if nc < 2:
            continue
        s_ = np.clip(I[m, g], 0, nc - 1); t_ = np.clip(T[m, g], 0, nc - 1)
        M = np.zeros((nc, nc)); np.add.at(M, (s_, t_), 1.0)
        for c in range(nc):
            tot = M[c].sum()
            if tot >= min_n and M[c].max() / tot >= tau and int(M[c].argmax()) != c:
                total += 1
                print(f"  {FE[g][:45]:46s}{c:>6}{int(M[c].argmax()):>4}{int(tot):>7}"
                      f"{M[c].max()/tot:>8.3f}")
    print(f"  total overrides: {total}")
    acu = [g for g, f in enumerate(FE) if "akut" in f.lower()]
    print(f"  organ-acuity items: {len(acu)}  (the one-line rule maps state 1 -> 2 on each of these)")




def cmd_maskaudit():
    """What does "observed" mean for a cell? (reviewer R3)

    The evaluation scores 99.2% of the 45x6,319 possible test cells as observed on both days, while
    the export records a median of 13 completed items per form. Those two facts are only compatible
    if one recorded entry expands into several output fields, or if absence is being turned into an
    observed value somewhere. For a paper whose headline is that copying is hard to beat, the
    difference matters: a structurally derived `none` that repeats every day is not an independent
    physician observation, and a missing field silently scored as an observed negative would inflate
    both the baseline and the apparent stability.

    This prints, per item, the observed rate in the split masks and the non-NaN rate in the
    day-level table, so the expansion can be located rather than argued about.
    """
    import numpy as np, pandas as pd, os, sys
    HERE = os.path.dirname(os.path.abspath(__file__)); KEEP = os.path.dirname(HERE)
    sys.path.insert(0, KEEP)
    sp = os.environ.get("MA_SPLIT",
                        os.path.join(HERE, "01_split_sets/gtsq_pred_24h_gap6h_clin_cult_patsplit"))
    n = np.load(os.path.join(sp, "norm_stats.npz"), allow_pickle=True)
    FE = [str(x) for x in n["gtsq_features"]]
    d = np.load(os.path.join(sp, "test.npz"), allow_pickle=True)
    om = d["gtsq_out_mask"]; im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
    both = (om == 1) & (im == 1)
    print(f"split={os.path.basename(sp)}  test pairs={len(om):,}  features={len(FE)}")
    print(f"scored cells (both days) = {int(both.sum()):,} "
          f"= {both.sum()/(len(om)*len(FE)):.3%} of all cells; "
          f"{both.sum()/len(om):.2f} items per pair\n")

    # the day-level table must match the split being audited, or the coverage column reports the
    # filled cohort while the mask columns report the explicit one
    csvname = os.environ.get("MA_CSV", "gtsq_day_level_encoded_fixnn.csv")
    df = pd.read_csv(os.path.join(HERE, csvname), low_memory=False)
    have = [c for c in FE if c in df.columns]
    rate = df[have].notna().mean()
    print(f"  {'feature':50s}{'day-level':>11}{'target obs':>12}{'both obs':>10}")
    for j, f in enumerate(FE):
        r = f"{rate[f]:>11.3%}" if f in rate.index else f"{'absent':>11}"
        print(f"  {f[:48]:50s}{r}{(om[:, j] == 1).mean():>12.3%}{both[:, j].mean():>10.3%}")
    per_row = df[have].notna().sum(1)
    print(f"\n  day-level table: {len(df):,} rows; non-NaN features per row "
          f"median {per_row.median():.0f}, mean {per_row.mean():.2f}, of {len(have)}")
    print(f"  rows with all {len(have)} features present: {(per_row == len(have)).mean():.2%}")




def cmd_drift():
    """Cohort drift across the chronological folds.

    With strict temporal separation the folds are 2016-06..2019-10, 2019-10..2020-11 and
    2020-12..2022-03, so training is entirely pre-pandemic and test entirely intra-pandemic. A
    difference between them is therefore not attributable to elapsed time alone, and the
    questionnaire-version boundary (2017-08-28) sits inside the training period. This prints what
    actually moves: volume, change prevalence, and the class mix of the items most likely to shift.
    """
    import numpy as np, pandas as pd, os, sys
    HERE = os.path.dirname(os.path.abspath(__file__)); KEEP = os.path.dirname(HERE)
    sys.path.insert(0, KEEP); import gtsq_eval as ge
    sp = os.environ.get("DRIFT_SPLIT",
                        os.path.join(HERE, "01_split_sets/gtsq_temporal_strict_clin_cult_patsplit"))
    n = np.load(os.path.join(sp, "norm_stats.npz"), allow_pickle=True)
    FE = [str(x) for x in n["gtsq_features"]]; NC = [int(x) for x in n["gtsq_nclasses"]]
    FT = [str(x) for x in n["gtsq_ftypes"]]
    gm = n["gtsq_means"].astype(float); gs = n["gtsq_stds"].astype(float)
    print(f"split={os.path.basename(sp)}")
    print(f"  {'fold':6s}{'pairs':>8}{'patients':>10}{'from':>12}{'to':>12}"
          f"{'chg prev':>10}{'mean/day':>10}")
    for nm in ("train", "val", "test"):
        d = np.load(os.path.join(sp, f"{nm}.npz"), allow_pickle=True)
        gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
        im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
        I = ge.denorm_to_class(gi, gm, gs, FT, NC)
        T = ge.denorm_to_class(d["gtsq_out"], gm, gs, FT, NC)
        M = (d["gtsq_out_mask"] == 1) & (im == 1)
        dt = pd.to_datetime(pd.Series([x.split("_", 1)[1] for x in
                                       np.asarray(d["dates"]).astype(str)]))
        pat = len(np.unique(np.asarray(d["pat_ids"]).astype(str))) if "pat_ids" in d.files else 0
        prev = (M & (T != I)).sum() / M.sum()
        print(f"  {nm:6s}{len(I):>8,}{pat:>10,}{str(dt.min().date()):>12}{str(dt.max().date()):>12}"
              f"{prev:>10.2%}{M.sum()/len(I):>10.1f}")
    # the items whose class mix moves most between train and test
    def mix(nm):
        d = np.load(os.path.join(sp, f"{nm}.npz"), allow_pickle=True)
        gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
        im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
        T = ge.denorm_to_class(d["gtsq_out"], gm, gs, FT, NC)
        M = (d["gtsq_out_mask"] == 1) & (im == 1)
        out = {}
        for j in range(len(FE)):
            v = T[M[:, j], j]
            if len(v):
                out[j] = np.bincount(np.clip(v, 0, max(1, NC[j] - 1)).astype(int),
                                     minlength=max(2, NC[j])) / len(v)
        return out
    a, b = mix("train"), mix("test")
    sh = [(j, float(np.abs(a[j] - b[j]).sum() / 2)) for j in a if j in b and len(a[j]) == len(b[j])]
    sh.sort(key=lambda x: -x[1])
    print("\n  largest train->test shift in class mix (total variation distance):")
    for j, v in sh[:8]:
        print(f"    {FE[j][:46]:48s}{v:>7.3f}   train {np.round(a[j],3)}  test {np.round(b[j],3)}")




def _explicit_mask(SPD, EXD, ge, FE, FT, NC, gm, gs):
    """Test-split arrays plus the explicit-only mask transferred onto them."""
    import numpy as np, os
    d = np.load(os.path.join(SPD, "test.npz"), allow_pickle=True)
    gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
    im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
    I = ge.denorm_to_class(gi, gm, gs, FT, NC); T = ge.denorm_to_class(d["gtsq_out"], gm, gs, FT, NC)
    M = (d["gtsq_out_mask"] == 1) & (im == 1)
    dm = np.load(os.path.join(EXD, "test.npz"), allow_pickle=True)
    mi = dm["gtsq_in_mask"]; mi = mi[:, -1, :] if mi.ndim == 3 else mi
    alt = (dm["gtsq_out_mask"] == 1) & (mi == 1)
    key = {k: i for i, k in enumerate(np.asarray(dm["dates"]).astype(str))}
    rows = [key.get(k, -1) for k in np.asarray(d["dates"]).astype(str)]
    M2 = np.zeros_like(M)
    for i, r in enumerate(rows):
        if r >= 0:
            M2[i] = alt[r]
    return I, T, M, M & M2


def cmd_absclass():
    """Class support under the two masks, and what it does to item-macro-F1.

    Excluding derived cells removes an entire class from some items, and under a fixed class set an
    absent class contributes F1 zero. That makes item-macro-F1 incomparable ACROSS masks, which is
    why the paper compares accuracy there and keeps macro-F1 for paired contrasts within a mask.
    Also reports how many CHANGED cells the explicit mask discards: it is not the filled task minus
    easy negatives.
    """
    import numpy as np, os, sys
    HERE = os.path.dirname(os.path.abspath(__file__)); KEEP = os.path.dirname(HERE)
    sys.path.insert(0, KEEP); import gtsq_eval as ge
    SPD = os.path.join(HERE, "01_split_sets/gtsq_pred_24h_gap6h_clin_cult_patsplit")
    EXD = SPD + "_explicit"
    n = np.load(os.path.join(SPD, "norm_stats.npz"), allow_pickle=True)
    FE = [str(x) for x in n["gtsq_features"]]; NC = [int(x) for x in n["gtsq_nclasses"]]
    FT = [str(x) for x in n["gtsq_ftypes"]]
    gm = n["gtsq_means"].astype(float); gs = n["gtsq_stds"].astype(float)
    I, T, M, ME = _explicit_mask(SPD, EXD, ge, FE, FT, NC, gm, gs)
    C = (T != I)
    cf, ce = int((M & C).sum()), int((ME & C).sum())
    print(f"filled   {int(M.sum()):,} cells, {cf:,} changes ({cf/M.sum():.2%})")
    print(f"explicit {int(ME.sum()):,} cells, {ce:,} changes ({ce/ME.sum():.2%})  "
          f"-> retains {ME.sum()/M.sum():.1%} of cells, {ce/cf:.1%} of changes")
    org = [j for j, f in enumerate(FE) if f.startswith("i05100_") or f.endswith("_present")]
    rem = M & ~ME & C
    cnt = sum(int((rem[:, j] & (I[:, j] == 0) & (T[:, j] > 0)).sum()) for j in org)
    print(f"  discarded changes that run derived-negative -> documented-positive "
          f"on site/presence items: {cnt:,}")
    def present(mask):
        return {j: int(len(np.unique(T[mask[:, j], j]))) for j in range(len(FE)) if mask[:, j].sum()}
    a, b = present(M), present(ME)
    lost = [j for j in a if b.get(j, 0) < a[j]]
    print(f"  items losing at least one observed class under the explicit mask: {len(lost)}/{len(FE)}")
    def macro(mask, fixed=True):
        per = []
        for j in range(len(FE)):
            m = mask[:, j]
            if not m.sum():
                continue
            t = T[m, j].astype(int); p = I[m, j].astype(int)
            ks = range(max(2, NC[j])) if fixed else sorted(set(t.tolist()))
            f1 = []
            for k in ks:
                tp = int(((p == k) & (t == k)).sum()); fp = int(((p == k) & (t != k)).sum())
                fn = int(((p != k) & (t == k)).sum())
                f1.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
            per.append(float(np.mean(f1)))
        return float(np.mean(per))
    print(f"  persistence macro-F1  filled: fixed {macro(M):.4f} / present-only {macro(M,False):.4f}"
          f"   explicit: fixed {macro(ME):.4f} / present-only {macro(ME,False):.4f}")
    print(f"  persistence accuracy  filled {(I[M]==T[M]).mean():.4f}   "
          f"explicit {(I[ME]==T[ME]).mean():.4f}")




def cmd_readout():
    """Patient-clustered intervals for the three categorical read-outs.

    The read-outs are deterministic given the data, so there is no seed variation to average; what
    there is, and what every other comparison in the paper reports, is uncertainty from the test
    patients. Without it the ordering of three numbers differing in the third decimal is an eyeball
    judgement. This bootstraps the same patients used elsewhere and reports PAIRED differences, so
    the question "do the forecasts add anything given the current state" gets an interval rather
    than an impression.
    """
    import numpy as np, os, sys
    HERE = os.path.dirname(os.path.abspath(__file__)); KEEP = os.path.dirname(HERE)
    sys.path.insert(0, KEEP); import gtsq_eval as ge
    SP = os.path.join(HERE, "01_split_sets/gtsq_pred_24h_gap6h_noclin_h7_patsplit")
    n = np.load(os.path.join(SP, "norm_stats.npz"), allow_pickle=True)
    FE = [str(x) for x in n["gtsq_features"]]; NC = [int(x) for x in n["gtsq_nclasses"]]
    FT = [str(x) for x in n["gtsq_ftypes"]]
    gm = n["gtsq_means"].astype(float); gs = n["gtsq_stds"].astype(float)
    d = np.load(os.path.join(SP, "test.npz"), allow_pickle=True)
    gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
    im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
    I = ge.denorm_to_class(gi, gm, gs, FT, NC); T = ge.denorm_to_class(d["gtsq_out"], gm, gs, FT, NC)
    M = (d["gtsq_out_mask"] == 1) & (im == 1)
    clust = (np.asarray(d["pat_ids"]).astype(str) if "pat_ids" in d.files
             else np.asarray(d["enc_ids"]).astype(str))
    F = os.path.join(HERE, "fmts_fm")
    names = [("median+round", "pred_chronos-bolt-base_test.npy"),
             ("quantile read-out", "pred_boltbase_cathead_test.npy"),
             ("state only", "pred_boltbase_state_test.npy"),
             ("state + quantiles", "pred_boltbase_stateq_test.npy")]
    P = {}
    for nm, f in names:
        fp = os.path.join(F, f)
        if os.path.exists(fp):
            P[nm] = np.load(fp)
        else:
            print(f"  missing {f} -- run fm_baseline with --save_pred for this variant")
    if len(P) < 2:
        return
    def met(pred, mask, Tx=None, Ix=None):
        # the targets must be resampled alongside the predictions and the mask; indexing the
        # unresampled T here is what made the bootstrap fail on a shape mismatch
        Tx = T if Tx is None else Tx
        Ix = I if Ix is None else Ix
        acc = float((pred[mask] == Tx[mask]).mean())
        ch = mask & (Tx != Ix)
        accc = float((pred[ch] == Tx[ch]).mean()) if ch.sum() else float("nan")
        per = []
        for j in range(len(FE)):
            m = mask[:, j]
            if not m.sum():
                continue
            t = Tx[m, j].astype(int); p = pred[m, j].astype(int); f1 = []
            for k in range(max(2, NC[j])):
                tp = int(((p == k) & (t == k)).sum()); fp = int(((p == k) & (t != k)).sum())
                fn = int(((p != k) & (t == k)).sum())
                f1.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
            per.append(float(np.mean(f1)))
        return acc, float(np.mean(per)), accc
    print(f"  {'read-out':20s}{'acc':>9}{'macro-F1':>10}{'Acc_c':>9}")
    for nm in P:
        a, f, c = met(P[nm], M)
        print(f"  {nm:20s}{a:>9.4f}{f:>10.4f}{c:>9.4f}")
    uc = np.unique(clust); idx = {c: np.where(clust == c)[0] for c in uc}
    rng = np.random.RandomState(0); B = int(os.environ.get("READOUT_BOOT", 2000))
    boot = {nm: [] for nm in P}
    for _ in range(B):
        pick = rng.randint(0, len(uc), len(uc))
        sel = np.concatenate([idx[uc[k]] for k in pick])
        Mb, Tb, Ib = M[sel], T[sel], I[sel]
        for nm in P:
            boot[nm].append(met(P[nm][sel], Mb, Tb, Ib))
    boot = {k: np.array(v) for k, v in boot.items()}
    base = "state only"
    if base not in boot:
        return
    print(f"\n  paired bootstrap, {B} resamples, cluster = patient ({len(uc)} patients)")
    print(f"  difference against '{base}':")
    print(f"  {'read-out':20s}{'accuracy':>26}{'macro-F1':>26}{'Acc_c':>26}")
    for nm in boot:
        if nm == base:
            continue
        cells = []
        for k in range(3):
            dd = boot[nm][:, k] - boot[base][:, k]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            cells.append(f"{dd.mean():>+9.4f}{star}[{lo:+.4f},{hi:+.4f}]")
        print(f"  {nm:20s}" + "".join(cells))
    print("  * = 95% interval excludes zero")


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd",choices=["ladder","subset","figure","table","ablation","items","frontier","baselines","winners","fmcal","circ","tsfladder","permimp","figregimes","cohort","markovmap","maskaudit","drift","absclass","readout"])
    p.add_argument("--balance",default=None,choices=["uniform","lw"],
                   help="which --loss_balance population to report (default uniform, the standard "
                        "since 2026-07-30; pass lw to reproduce the pre-2026-07-30 tables)")
    a=p.parse_args()
    if a.balance:
        BALANCE=a.balance; _m=sys.modules[__name__]; _m.BALANCE=a.balance
    {"ladder":cmd_ladder,"subset":cmd_subset,"figure":cmd_figure,"table":cmd_table,"ablation":cmd_ablation,"items":cmd_items,"frontier":lambda: (cmd_frontier("acc"), cmd_frontier("f1")),"baselines":cmd_baselines,"winners":cmd_winners,"fmcal":cmd_fmcal,"circ":cmd_circ,"tsfladder":cmd_tsfladder,"permimp":cmd_permimp,"figregimes":cmd_figregimes,"cohort":cmd_cohort,"markovmap":cmd_markovmap,"maskaudit":cmd_maskaudit,"drift":cmd_drift,"absclass":cmd_absclass,"readout":cmd_readout}[a.cmd]()
