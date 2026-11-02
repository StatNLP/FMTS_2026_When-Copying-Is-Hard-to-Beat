#!/usr/bin/env python3
"""Deployment-side evaluation of the change-risk alert, in ONE script.

Merges eval_override.py, eval_alarm_budget.py and decision_curve.py: all three consume the same
detector score and differ only in the question asked, so they share a file (and a masking
convention) rather than drifting apart.

Subcommands
  override   turn the detector into a predictor and ask whether it beats its base (App. D)
  alarm      precision/sensitivity under an explicit alarm BUDGET, i.e. the top-k cells (Table 3)
  decision   decision-curve net benefit vs treat-all / treat-none (Table 3, Vickers 2006)

BUDGET vs THRESHOLD (doc SS.N): the training-free detector scores per (item,state), so its 251,049
test cells take only 111 distinct values and tie in blocks. `alarm` uses BUDGET semantics (exactly k
alerts); thresholding at the (1-q) quantile fires on far more (at q=2%: 8,685 not 5,020). Never pair
a budget-derived alert rate with threshold-derived precision.

Usage:  python exp_fixnn/eval_deploy.py {override|alarm|decision} [options]
"""
import argparse, glob, importlib.util, os, sys
import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
KEEP = os.path.dirname(HERE)
SPLIT_DEFAULT = "exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_noclin"


import argparse, glob, importlib.util, os, numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
KEEP = os.path.dirname(HERE)


def _load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(KEEP, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def destinations(Itr, Ttr, Mtr, NC, FT):
    """Per (feature,state) modal CHANGE destination and its train purity. Binary is deterministic."""
    G = Itr.shape[1]; dest, pur = {}, {}
    for j in range(G):
        m = Mtr[:, j]; s = np.clip(Itr[m, j], 0, NC[j] - 1); t = np.clip(Ttr[m, j], 0, NC[j] - 1)
        ch = s != t
        for sv in range(NC[j]):
            sel = (s == sv) & ch
            if sel.sum():
                bc = np.bincount(t[sel], minlength=NC[j]); d = int(bc.argmax())
                dest[(j, sv)] = d; pur[(j, sv)] = bc[d] / bc.sum()
    return dest, pur


def _macro_f1_fast(t, p, k):
    """Macro-F1 from a bincount confusion matrix. Equivalent to sklearn's
    f1_score(average='macro', labels=range(k), zero_division=0) on integer class arrays.

    sklearn is not badly written -- it is GENERAL: it validates input, resolves label sets, supports
    sample weights, sparse input and every averaging mode, and that fixed overhead (~1.2 ms here)
    dominates when the arrays are small and called a million times. This version assumes exactly what
    we always have -- dense small-k integer classes, no weights -- so it is a single bincount.
    """
    cm = np.bincount(t * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return float(np.mean(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)))


def four_fast(P, I, T, M, NC):
    """The four metrics on a per-cell mask, without the sklearn call overhead."""
    n_ok = n_all = c_ok = c_all = 0
    f1s, f1cs = [], []
    for g in range(P.shape[1]):
        m = M[:, g]
        if not m.any():
            continue
        t = T[m, g]; p = np.clip(P[m, g], 0, NC[g] - 1); i = I[m, g]
        n_ok += int((p == t).sum()); n_all += int(m.sum())
        f1s.append(_macro_f1_fast(t, p, NC[g]))
        ch = i != t
        if ch.any():
            c_ok += int((p[ch] == t[ch]).sum()); c_all += int(ch.sum())
            f1cs.append(_macro_f1_fast(t[ch], p[ch], NC[g]))
    return (n_ok / n_all if n_all else float("nan"),
            float(np.mean(f1s)) if f1s else float("nan"),
            c_ok / c_all if c_all else float("nan"),
            float(np.mean(f1cs)) if f1cs else float("nan"))


def four(P, I, T, NC, sel=None):
    """Score a prediction. `sel` is either a per-PAIR row selector (N,) or a per-CELL mask (N,G).

    The per-cell form is the paper's convention -- a cell counts when that item is observed on both
    days (250,536 test cells) -- and is delegated to the shared scorer so the override table pools
    exactly as every other table does. The row-selector form scores all 45 cells of the pairs it is
    given, so it is only sound on fully-observed pairs (5,371); it is kept for the historical tables.
    """
    if sel is not None and getattr(sel, "ndim", 1) == 2:
        sys.path.insert(0, KEEP); import gtsq_eval as ge
        acc, mf1, cacc, _ = ge.four_metrics(P, I, T, sel.astype(bool), NC)
        return acc, mf1, cacc
    G = P.shape[1]; sel = np.ones(len(P), bool) if sel is None else sel
    Is, Ts, Ps = I[sel], T[sel], P[sel]; chg = Is != Ts
    acc = (Ps == Ts).mean()
    mf1 = np.mean([f1_score(Ts[:, j], np.clip(Ps[:, j], 0, NC[j] - 1), average="macro",
                            labels=list(range(NC[j])), zero_division=0) for j in range(G)])
    cacc = (Ps[chg] == Ts[chg]).mean() if chg.any() else float("nan")
    return acc, mf1, cacc


def cmd_override(a):
    p8 = _load("p8", "08_probes.py")
    from models.gated_predictor import build_feature_slices
    pats = a.run if isinstance(a.run, list) else [a.run]
    runs = sorted({r for pat in pats for r in glob.glob(os.path.join(KEEP, pat))})
    runs = [r for r in runs if os.path.exists(os.path.join(r, "model.pytorch"))]
    if not runs:
        raise SystemExit(f"no runs match {a.run}")
    N = np.load(os.path.join(KEEP, a.split, "norm_stats.npz"), allow_pickle=True)
    FT = [str(x) for x in N["gtsq_ftypes"]]; NC = [int(x) for x in N["gtsq_nclasses"]]
    FE = [str(x) for x in N["gtsq_features"]]
    gm = N["gtsq_means"].astype(float); gs = N["gtsq_stds"].astype(float); G = len(FT)

    def split_arrays(name):
        d = np.load(os.path.join(KEEP, a.split, f"{name}.npz"), allow_pickle=True)
        gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
        im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
        I = np.round(gi * gs + gm).astype(int); T = np.round(d["gtsq_out"] * gs + gm).astype(int)
        M = (im == 1) & (d["gtsq_out_mask"] == 1)
        return I, T, M
    # Eval split is a FLAG and defaults to val: this sweep chooses a firing threshold, which is model
    # selection, so it must not read test. The historical tables in the docs were produced on test --
    # reproduce them with `--split_name test` (the .sh runners pass it explicitly).
    Itr, Ttr, Mtr = split_arrays("train")
    Ite, Tte, Mte = split_arrays(a.split_name)
    dest, pur = destinations(Itr, Ttr, Mtr, NC, FT)

    # gamma (change prob per cell), seed-averaged over matching runs
    gam = None; used = 0
    for r in runs[:a.seeds]:
        try:
            model, cfg, norm, sd = p8.reconstruct(r)
            tn = np.load(os.path.join(sd, f"{a.split_name}.npz"), allow_pickle=True)
            sp = {k: tn[k] for k in tn.files}
            _, g = p8._forward_pred_gate(model, sp, getattr(cfg, "use_clinical", False),
                                         configs=cfg, norm=N)
        except (Exception, SystemExit) as e:
            print(f"  [skip] {os.path.basename(r)[:40]}: {repr(e)[:50]}"); continue
        gam = g if gam is None else gam + g; used += 1
    if used == 0:
        raise SystemExit("no usable detector runs")
    gam /= used

    # determined-destination prediction per cell (independent of threshold)
    D = Ite.copy()
    determined = np.zeros((len(Ite), G), bool)
    for j in range(G):
        if FT[j] == "binary":
            D[:, j] = 1 - np.clip(Ite[:, j], 0, 1); determined[:, j] = True
        else:
            for i in range(len(Ite)):
                key = (j, int(np.clip(Ite[i, j], 0, NC[j] - 1)))
                if key in dest and pur.get(key, 0) >= a.purity:
                    D[i, j] = dest[key]; determined[i, j] = True

    # base predictor the override sits ON TOP of: 'copy' (persistence) or 'markov' (copy + the
    # UNCONDITIONAL new->acute rule -- the real baseline). With base=markov the override only ADDS
    # the detector-gated binary flips (and any high-purity non-new->acute ordinal transitions), so
    # this is "apply the detector on the markov model": does gating add anything over markov?
    base = Ite.copy()
    if a.base == "markov":
        for j in range(G):
            if "akut" in FE[j].lower():
                base[:, j] = np.where(np.clip(Ite[:, j], 0, 2) == 1, 2, np.clip(Ite[:, j], 0, 2))
    base_acc, base_f1, _ = four(base, Ite, Tte, NC, Mte)
    print(f"BASE = {a.base}: acc {base_acc:.4f}  macro-F1 {base_f1:.4f}   "
          f"[{used} detector seed(s), purity>={a.purity}]")
    print(f"{'thr':>6}{'overrides':>11}{'acc':>9}{'macro-F1':>10}{'chg-acc':>9}{'det-prec':>10}{'det-rec':>9}")
    fo = Mte; chg_true = (Ite != Tte)          # per-CELL mask: the paper's 250,536-cell convention
    best = (base_acc, None, base_f1)
    for thr in a.thresholds:
        fire = (gam >= thr) & determined
        P = np.where(fire, D, base)                      # override on top of the chosen base
        acc, mf1, cacc = four(P, Ite, Tte, NC, fo)
        ov = int((fire & (D != base))[fo].sum())         # overrides that actually change the base
        chg_fire = fire & (D != base)
        tp = int((chg_fire & chg_true)[fo].sum()); fp = int((chg_fire & ~chg_true)[fo].sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / int(chg_true[fo].sum())
        flag = f"  <-- beats {a.base}" if (acc > base_acc + 1e-6 and mf1 > base_f1 - 1e-6) else ""
        print(f"{thr:6.2f}{ov:11d}{acc:9.4f}{mf1:10.4f}{cacc:9.4f}{prec:10.3f}{rec:9.3f}{flag}")
        if acc > best[0]:
            best = (acc, thr, mf1)
    print(f"\nbest overall acc {best[0]:.4f}" +
          (f" / F1 {best[2]:.4f} at thr {best[1]:.2f}  (base {a.base}: acc {base_acc:.4f} / F1 {base_f1:.4f})"
           if best[1] is not None else f" == base {a.base} -- gating adds nothing over the base"))

    if a.confirm_on_test:
        # The threshold is now FIXED by the sweep above (validation). Score that single operating
        # point on test, once -- this is the only place cmd_override may read test when selecting.
        if a.split_name != "val":
            raise SystemExit("--confirm-on-test requires the sweep to have run on val")
        # The BASE (persistence, or persistence+neu->akut) has no free parameters -- nothing about it
        # is selected -- so its test numbers are always reportable and the paper needs them. Only the
        # OVERRIDE row depends on a threshold, and that row appears only if val selected one.
        thr = best[1]
        Ix, Tx, Mx = split_arrays("test")
        gx = None; nx = 0
        for r in runs[:a.seeds]:
            try:
                model, cfg, norm, sd = p8.reconstruct(r)
                tn = np.load(os.path.join(sd, "test.npz"), allow_pickle=True)
                _, g = p8._forward_pred_gate(model, {k: tn[k] for k in tn.files},
                                             getattr(cfg, "use_clinical", False),
                                             configs=cfg, norm=N)
            except (Exception, SystemExit):
                continue
            gx = g if gx is None else gx + g; nx += 1
        gx /= max(nx, 1)
        Dx = Ix.copy(); detx = np.zeros((len(Ix), G), bool)
        for j in range(G):
            if FT[j] == "binary":
                Dx[:, j] = 1 - np.clip(Ix[:, j], 0, 1); detx[:, j] = True
            else:
                for i in range(len(Ix)):
                    key = (j, int(np.clip(Ix[i, j], 0, NC[j] - 1)))
                    if key in dest and pur.get(key, 0) >= a.purity:
                        Dx[i, j] = dest[key]; detx[i, j] = True
        bx = Ix.copy()
        if a.base == "markov":
            for j in range(G):
                if "akut" in FE[j].lower():
                    bx[:, j] = np.where(np.clip(Ix[:, j], 0, 2) == 1, 2, np.clip(Ix[:, j], 0, 2))
        fox = Mx
        ba, bf, bca = four(bx, Ix, Tx, NC, fox)
        print(f"\n  [TEST] base = {a.base} (parameter-free, nothing selected -- reportable as is)")
        print(f"    base {a.base}: acc {ba:.4f}  macro-F1 {bf:.4f}  chg-acc {bca:.4f}"
              f"   [{int(fox.sum()):,} both-observed cells]")
        if thr is None:
            print("    override:      no threshold beat the base on val -> not scored on test.")
        else:
            Px = np.where((gx >= thr) & detx, Dx, bx)
            acc, mf1, cacc = four(Px, Ix, Tx, NC, fox)
            print(f"    override @ val-selected thr {thr:.2f}: acc {acc:.4f}  macro-F1 {mf1:.4f}  "
                  f"chg-acc {cacc:.4f}"
                  f"   -> {'BEATS' if acc > ba + 1e-6 else 'does not beat'} the base on accuracy")





def cmd_alarm(a):
    sys.path.insert(0, KEEP); os.environ["BAYES_SPLIT"] = os.path.join(KEEP, a.split)
    import bayes_floor as bf
    sys.path.insert(0, KEEP); import gtsq_eval as ge
    FE = [str(x) for x in bf._N["gtsq_features"]]; NC = [int(x) for x in bf._N["gtsq_nclasses"]]
    FT = [str(x) for x in bf._N["gtsq_ftypes"]]; G = len(FE)
    Itr, Ttr, Mtr, _ = bf._load("train"); Ite, Tte, Mte, _ = bf._load("test")

    if a.run:                                            # trained neural detector (5-seed mean gamma)
        spec = importlib.util.spec_from_file_location("p8", os.path.join(KEEP, "08_probes.py"))
        p8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p8)
        gam = None; used = 0
        for r in sorted(glob.glob(os.path.join(KEEP, a.run)))[:5]:
            if not os.path.exists(os.path.join(r, "model.pytorch")):
                continue
            try:
                model, cfg, norm, sd = p8.reconstruct(r)
                tn = np.load(os.path.join(sd, "test.npz"), allow_pickle=True)
                _, g = p8._forward_pred_gate(model, {k: tn[k] for k in tn.files},
                                             getattr(cfg, "use_clinical", False),
                                             configs=cfg, norm=N)
            except (Exception, SystemExit):
                continue
            gam = g if gam is None else gam + g; used += 1
        if gam is None:
            raise SystemExit("no detector run produced a gate; check the run glob and that these "
                             "are gate models")
        gam /= used; tag = f"neural detector ({used} seeds)"
    else:                                                # training-free per-(feature,state) rate
        rate = {}
        for j in range(G):
            m = Mtr[:, j]; s = np.clip(Itr[m, j], 0, NC[j] - 1); ch = s != np.clip(Ttr[m, j], 0, NC[j] - 1)
            for sv in np.unique(s):
                rate[(j, int(sv))] = float(ch[s == sv].mean())
        gam = np.zeros(Ite.shape)
        for j in range(G):
            gam[:, j] = [rate.get((j, int(np.clip(v, 0, NC[j] - 1))), 0.0) for v in Ite[:, j]]
        tag = "training-free per-(feature,state) rate"

    y = ((Ite != Tte) & Mte); obs = Mte
    s_flat = gam[obs]; y_flat = y[obs].astype(int)
    base = y_flat.mean()
    det = ge.detector_metrics(y_flat, s_flat)
    print(f"detector: {tag}   n_cells={y_flat.size:,}  base rate (change) {base*100:.2f}%")
    print(f"  AUROC {det['auroc']:.3f}  AUPRC {det['auprc']:.3f}  best-F1 {det['f1_best']:.3f}\n")

    print("GLOBAL alarm budget -- rank all observed cells, alert on the top q%")
    print(f"  {'budget':>8}{'alerts':>9}{'precision':>11}{'recall':>9}{'lift':>7}")
    order = np.argsort(-s_flat)
    for q in (0.5, 1, 2, 5, 10, 20):
        k = max(1, int(len(s_flat) * q / 100))
        sel = order[:k]; prec = y_flat[sel].mean()
        print(f"  {q:>6.1f}%{k:>9}{prec:>11.3f}{y_flat[sel].sum()/y_flat.sum():>9.3f}{prec/base:>7.1f}x")

    # E9/meeting: what budget is needed to CATCH a given share of the changes? The budget table above
    # fixes the alert volume and reads off recall; a deployment argument usually runs the other way --
    # "I want to miss at most 1 in 100 changes, what does that cost me?" Reported as the alert rate
    # and the implied alerts per patient-day, since that is the quantity a clinician actually feels.
    print("\nRECALL-FIRST view -- budget required to catch a target share of all changes")
    print(f"  {'target':>8}{'budget':>9}{'alerts':>9}{'precision':>11}{'per day':>9}{'lift':>7}")
    cum = np.cumsum(y_flat[order])
    n_pairs = len(gam)
    for tgt in (0.50, 0.80, 0.95, 0.99, 0.999):
        need = int(np.searchsorted(cum, tgt * y_flat.sum()) + 1)
        need = min(need, len(order))
        prec = y_flat[order[:need]].mean()
        print(f"  {tgt*100:>7.1f}%{need/len(s_flat)*100:>8.1f}%{need:>9}{prec:>11.3f}"
              f"{need/n_pairs:>9.1f}{prec/base:>7.1f}x")
    print("  (per day = alerts per questionnaire; the instrument has 45 items, so >45 is impossible")
    print("   and anything near it means the alert has stopped discriminating)")

    print("\nPER-PATIENT-DAY budget -- alert on the top-k cells of each questionnaire")
    print(f"  {'k/day':>7}{'alerts':>9}{'precision':>11}{'recall':>9}{'lift':>7}")
    for k in (1, 2, 3, 5):
        sel = np.zeros_like(y, bool)
        for i in range(len(gam)):
            idx = np.where(obs[i])[0]
            if len(idx) == 0:
                continue
            top = idx[np.argsort(-gam[i, idx])[:k]]
            sel[i, top] = True
        p = y[sel].mean(); n = int(sel.sum())
        print(f"  {k:>7}{n:>9}{p:>11.3f}{y[sel].sum()/y[obs].sum():>9.3f}{p/base:>7.1f}x")
    print("\n  lift = precision / base rate. A clinically usable alert needs precision well above")
    print("  the base rate at a budget a ward round can actually absorb (1-3 cells/patient-day).")





def cmd_decision(a):
    sys.path.insert(0, KEEP); os.environ["BAYES_SPLIT"] = os.path.join(KEEP, a.split)
    import bayes_floor as bf
    NC = [int(x) for x in bf._N["gtsq_nclasses"]]; G = len(NC)
    Itr, Ttr, Mtr, _ = bf._load("train"); Ite, Tte, Mte, _ = bf._load("test")

    if a.run:
        spec = importlib.util.spec_from_file_location("p8", os.path.join(KEEP, "08_probes.py"))
        p8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p8)
        gam = None; used = 0
        for r in sorted(glob.glob(os.path.join(KEEP, a.run)))[:5]:
            if not os.path.exists(os.path.join(r, "model.pytorch")):
                continue
            try:
                model, cfg, norm, sd = p8.reconstruct(r)
                tn = np.load(os.path.join(sd, "test.npz"), allow_pickle=True)
                _, g = p8._forward_pred_gate(model, {k: tn[k] for k in tn.files},
                                             getattr(cfg, "use_clinical", False),
                                             configs=cfg, norm=N)
            except (Exception, SystemExit):
                continue
            gam = g if gam is None else gam + g; used += 1
        if gam is None:
            raise SystemExit("no detector run produced a gate; check the run glob and that these "
                             "are gate models")
        gam /= used; tag = f"trained detector ({used} seeds)"
    else:
        rate = {}
        for j in range(G):
            m = Mtr[:, j]; s = np.clip(Itr[m, j], 0, NC[j] - 1)
            ch = s != np.clip(Ttr[m, j], 0, NC[j] - 1)
            for sv in np.unique(s):
                rate[(j, int(sv))] = float(ch[s == sv].mean())
        gam = np.zeros(Ite.shape)
        for j in range(G):
            gam[:, j] = [rate.get((j, int(np.clip(v, 0, NC[j] - 1))), 0.0) for v in Ite[:, j]]
        tag = "training-free per-(feature,state) rate"

    y = ((Ite != Tte) & Mte)[Mte].astype(int)          # 1 = the cell really changes
    s = gam[Mte]
    N = len(y); prev = y.mean()
    print(f"decision-curve analysis — {tag}")
    print(f"  observed cells N={N:,}  prevalence (change rate) {prev*100:.2f}%\n")
    print(f"  {'p_t':>6}{'odds':>8}{'alerts':>9}{'NB(alert)':>11}{'NB(all)':>10}{'best':>16}")
    useful = []
    for pt in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        w = pt / (1 - pt)
        fire = s >= pt
        tp = int((fire & (y == 1)).sum()); fp = int((fire & (y == 0)).sum())
        nb = tp / N - (fp / N) * w
        nb_all = prev - (1 - prev) * w                # re-examine every cell
        best = "ALERT" if (nb > nb_all and nb > 0) else ("treat-all" if nb_all > max(nb, 0) else "treat-none")
        if best == "ALERT":
            useful.append(pt)
        print(f"  {pt:>6.2f}{w:>8.3f}{int(fire.sum()):>9}{nb:>11.4f}{nb_all:>10.4f}{best:>16}")
    print(f"\n  ALERT is the best policy for p_t in {useful if useful else 'NO threshold'}")
    if useful:
        lo, hi = min(useful), max(useful)
        print(f"  i.e. it beats both trivial policies when the clinician accepts between "
              f"{(1-lo)/lo:.0f}:1 and {(1-hi)/hi:.0f}:1 (unnecessary re-examinations : missed change).")
    print("\n  NB(alert) is the net benefit of re-examining exactly the alerted cells; NB(all) that of")
    print("  re-examining every cell; treat-none is 0. Above BOTH -> the alert adds decision value.")

def cmd_ensemble(a):
    """Ensembling and feature-restricted hybrids, scored on VALIDATION by default.

    Motivated by the per-feature attribution of the search winner: essentially ALL of its macro-F1
    gain over persistence+rule sits on the eight organ `akut/neu` items (plus one organ-cause item),
    i.e. the model has learned to apply `neu->akut` SELECTIVELY where the unconditional rule is wrong
    (that transition is only 0.61-0.89 pure, doc O.6). Everywhere else the model only loses accuracy.
    So the natural constructions are:

      ens      majority vote over the given runs, applied to every cell
      ens_org  vote on the organ acuity cells only, rule elsewhere   <- keeps the gain, drops the harm
      ens_orgp vote on the organ acuity cells only, persistence elsewhere

    Test is NOT read unless --split_name test is passed explicitly.
    """
    p8 = _load("p8", "08_probes.py")
    sys.path.insert(0, KEEP); import gtsq_eval as ge
    runs = sorted(r for pat in a.runs for r in glob.glob(os.path.join(KEEP, pat))
                  if os.path.exists(os.path.join(r, "model.pytorch")))
    if not runs:
        raise SystemExit(f"no runs match {a.runs}")
    N = np.load(os.path.join(KEEP, a.split, "norm_stats.npz"), allow_pickle=True)
    FT = [str(x) for x in N["gtsq_ftypes"]]; NC = [int(x) for x in N["gtsq_nclasses"]]
    FE = [str(x) for x in N["gtsq_features"]]
    gm = N["gtsq_means"].astype(float); gs = N["gtsq_stds"].astype(float)
    d = np.load(os.path.join(KEEP, a.split, f"{a.split_name}.npz"), allow_pickle=True)
    sp = {k: d[k] for k in d.files}
    gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
    I = ge.denorm_to_class(gi, gm, gs, FT, NC)
    T = ge.denorm_to_class(d["gtsq_out"], gm, gs, FT, NC)
    # masking convention: (i) target observed, or (ii) BOTH days observed. (ii) is the honest one for
    # a change task -- a change is undefined when day D is missing, and under (i) 513 test cells are
    # scored against a persistence prediction that copies an imputed value.
    M = d["gtsq_out_mask"] == 1
    if a.mask == "both":
        im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
        M = M & (im == 1)
    ORG = np.array(["akut" in f.lower() for f in FE])          # the 8 organ acuity items
    R = I.copy()
    for j in np.where(ORG)[0]:
        R[:, j] = np.where(I[:, j] == 1, 2, I[:, j])

    def _expand(pats):
        return sorted(r for pat in pats for r in glob.glob(os.path.join(KEEP, pat))
                      if os.path.exists(os.path.join(r, "model.pytorch")))

    def _predict_group(run_dirs, split_dir, norm=None):
        from models.gated_predictor import build_feature_slices
        sl, _tot = build_feature_slices(list(zip(FT, NC)))   # returns (slices, total_size)
        out = []
        for r in run_dirs:
            try:
                model, cfg, _, _ = p8.reconstruct(r)
                # a --drop_clinical run was trained on a narrower clinical tensor than its split
                # carries; replay it through the same slice or the forward pass raises
                # norm must be the one belonging to THIS group's split: group B is replayed on
                # --split_b, whose clinical_variables need not match --split's. Latent today (group
                # B has no drop_clinical) but it would slice by the wrong variable list.
                cols = p8.clinical_cols_for(cfg, norm if norm is not None else N)
                lg, _ = p8._forward_pred_gate(model, split_dir,
                                              getattr(cfg, "use_clinical", False),
                                              clin_cols=cols)
            except (Exception, SystemExit) as e:
                print(f"  skip {os.path.basename(r)[:40]}: {type(e).__name__}")
                continue
            p = I.copy()                                      # non-decodable dims stay at persistence
            for g, s_ in enumerate(sl):
                if FT[g] not in ("binary", "ordinal"):
                    continue
                col = lg[:, s_.start:s_.stop]
                p[:, g] = (col[:, 0] > 0).astype(int) if FT[g] == "binary" else col.argmax(1)
            out.append(p)
        return out

    P = _predict_group(runs, sp, N)
    if not P:
        raise SystemExit("no run produced predictions")
    P = np.stack(P)                                            # (R, N, G)

    def vote(stack):
        """Per-cell modal class across runs; ties resolve to the first run (deterministic)."""
        out = np.zeros_like(stack[0])
        for g in range(stack.shape[2]):
            col = stack[:, :, g]
            out[:, g] = [np.bincount(col[:, i]).argmax() for i in range(col.shape[1])]
        return out

    E = vote(P)

    # Group B, when asked for. Pairing two systems requires that row i means the same day-pair in
    # both, which is NOT guaranteed across split directories -- so it is checked, not assumed. A
    # silent misalignment here would produce a confident p-value for a comparison nobody made.
    EB = None
    if a.runs_b:
        db = np.load(os.path.join(KEEP, a.split_b or a.split, f"{a.split_name}.npz"),
                     allow_pickle=True)
        for key in ("enc_ids", "dates"):
            if key in d and key in db and not np.array_equal(d[key], db[key]):
                raise SystemExit(f"ERROR: --runs_b rows do not match --runs rows on '{key}'. "
                                 f"The two splits are not row-aligned and must not be paired.")
        if db["gtsq_out"].shape != d["gtsq_out"].shape:
            raise SystemExit("ERROR: --runs_b target shape differs from --runs; refusing to pair.")
        nb = np.load(os.path.join(KEEP, a.split_b or a.split, "norm_stats.npz"), allow_pickle=True)
        PB = _predict_group(_expand(a.runs_b), {k: db[k] for k in db.files}, nb)
        if not PB:
            raise SystemExit("no --runs_b run produced predictions")
        EB = vote(np.stack(PB))
        print(f"  group B: {len(PB)} runs from {a.split_b or a.split}")

    cands = [("persistence", I), ("persistence + rule", R), (f"ensemble (n={len(P)})", E),
             ("ensemble on organs / rule elsewhere", np.where(ORG[None, :], E, R)),
             ("ensemble on organs / persistence elsewhere", np.where(ORG[None, :], E, I))]
    if len(P) > 1:
        cands.insert(2, (f"single best-listed run", P[0]))
    print("=" * 92)
    print(f"ENSEMBLE / HYBRID -- split={a.split_name.upper()}  {len(P)} runs  {M.sum():,} scored cells")
    print("=" * 92)
    print(f"  {'predictor':44s}{'macro-F1':>10}{'acc':>9}{'chg-acc':>9}{'chg-F1':>9}")
    for nm, Pr in cands:
        acc, f1, ca, f1c = ge.four_metrics(Pr, I, T, M, NC)
        print(f"  {nm:44s}{f1:>10.4f}{acc:>9.4f}{ca:>9.4f}{f1c:>9.4f}")
    for r in runs:
        print(f"    run: {os.path.basename(r)[:86]}")

    if a.bootstrap:
        # Paired bootstrap over DAY-PAIRS (the sampling unit): resample rows with replacement, recompute
        # every predictor on the same resample, and take the distribution of the DIFFERENCE against the
        # rule. Pairing matters -- the predictors agree on ~96% of cells, so their errors are strongly
        # correlated and an unpaired interval would be far too wide to see a 0.003 effect.
        rng = np.random.RandomState(0)
        n_rows = len(I)
        names = ["macro-F1", "accuracy", "chg-acc", "chg-F1"]
        preds = [("persistence", I), ("rule", R), (f"ensemble (n={len(P)})", E)]
        boot = {nm: [] for nm, _ in preds}
        for _ in range(a.bootstrap):
            idx = rng.randint(0, n_rows, n_rows)
            Ib, Tb, Mb = I[idx], T[idx], M[idx]
            for nm, Pr in preds:
                boot[nm].append(ge.four_metrics(Pr[idx], Ib, Tb, Mb, NC))
        for nm in boot:
            boot[nm] = np.array(boot[nm])[:, [1, 0, 2, 3]]        # -> F1, acc, chg-acc, chg-F1
        print(f"\n  BOOTSTRAP over day-pairs, {a.bootstrap} resamples, 95% percentile intervals")
        print(f"  {'predictor':30s}" + "".join(f"{n:>22}" for n in names))
        for nm, _ in preds:
            b = boot[nm]
            print(f"  {nm:30s}" + "".join(
                f"{b[:, k].mean():>10.4f} [{np.percentile(b[:, k], 2.5):.4f},{np.percentile(b[:, k], 97.5):.4f}]"
                for k in range(4)))
        print(f"\n  PAIRED difference vs the rule (positive = better than persistence${{+}}$rule):")
        print(f"  {'predictor':30s}" + "".join(f"{n:>22}" for n in names))
        for nm, _ in preds:
            if nm == "rule":
                continue
            d = boot[nm] - boot["rule"]
            cells = []
            for k in range(4):
                lo, hi = np.percentile(d[:, k], [2.5, 97.5])
                star = "*" if lo > 0 or hi < 0 else " "
                cells.append(f"{d[:, k].mean():>+9.4f}{star}[{lo:+.4f},{hi:+.4f}]")
            print(f"  {nm:30s}" + "".join(cells))
        print("  * = the 95% interval excludes zero.")

    if a.arp:
        # PAIRED APPROXIMATE RANDOMISATION (Noreen 1989; Riezler & Maxwell 2005). The bootstrap above
        # estimates the SPREAD of a difference; this tests a null hypothesis directly, and is the
        # right instrument for "is this system better than the rule": under H0 the two systems are
        # interchangeable, so swapping their predictions on a randomly chosen subset of day-pairs
        # should produce a difference at least as large as the observed one reasonably often.
        #
        # The swap unit is the DAY-PAIR, not the cell: the 45 cells of one questionnaire are one
        # patient's judgement on one day and are strongly dependent, so shuffling cells individually
        # would assume an independence the data does not have and would understate p.
        #
        # p = (#{|d*| >= |d_obs|} + 1) / (R + 1); the +1 keeps p strictly positive -- with R
        # resamples the smallest reportable value is 1/(R+1), so R=10000 bottoms out at p<0.0001.
        rng = np.random.RandomState(1)
        n_rows = len(I)
        names = ["macro-F1", "accuracy", "chg-acc", "chg-F1"]
        if EB is not None:
            base_name, base_P = "group B", EB
            systems = [(f"group A (n={len(P)})", E)]
        else:
            base_name, base_P = ("rule", R) if a.base == "markov" else ("persistence", I)
            systems = [(f"ensemble (n={len(P)})", E)] + \
                      ([("single run", P[0])] if len(P) > 1 else []) + \
                      [("persistence" if a.base == "markov" else "rule", I if a.base == "markov" else R)]
        m_base = np.array(ge.four_metrics(base_P, I, T, M, NC))[[1, 0, 2, 3]]
        print(f"\n  APPROXIMATE RANDOMISATION vs {base_name}, {a.arp} shuffles, swap unit = day-pair")
        print(f"  {'system':30s}" + "".join(f"{n:>24}" for n in names))
        for nm, Psys in systems:
            m_sys = np.array(ge.four_metrics(Psys, I, T, M, NC))[[1, 0, 2, 3]]
            d_obs = np.abs(m_sys - m_base)
            ge_count = np.zeros(4)
            for _ in range(a.arp):
                swap = rng.rand(n_rows) < 0.5
                A = np.where(swap[:, None], base_P, Psys)      # swap whole day-pairs between systems
                B = np.where(swap[:, None], Psys, base_P)
                da = np.array(four_fast(A, I, T, M, NC))[[1, 0, 2, 3]]
                db = np.array(four_fast(B, I, T, M, NC))[[1, 0, 2, 3]]
                ge_count += (np.abs(da - db) >= d_obs - 1e-12)
            p = (ge_count + 1) / (a.arp + 1)
            cells = []
            for k in range(4):
                star = "***" if p[k] < 0.001 else "** " if p[k] < 0.01 else "*  " if p[k] < 0.05 else "   "
                cells.append(f"{m_sys[k] - m_base[k]:>+10.4f} p={p[k]:<7.4f}{star}")
            print(f"  {nm:30s}" + "".join(cells))
        print("  two-sided; *** p<0.001, ** p<0.01, * p<0.05")

    if a.per_class:
        # WHICH CLASS improves on the organ acuity items? They are 3-class (0=none, 1=neu, 2=akut).
        # Note the rule can never OUTPUT class 1: it copies day D and rewrites any `neu` to `akut`,
        # so every true `neu` target is a guaranteed miss for it. That is the structural gap the
        # model can close, and this table shows whether it does.
        from sklearn.metrics import precision_recall_fscore_support as prf
        org = np.where(ORG)[0]
        print(f"\n  PER-CLASS on the {len(org)} organ acuity items (0=none, 1=neu, 2=akut), "
              f"pooled over items -- {a.split_name}")
        print(f"  {'predictor':30s}{'class':>7}{'support':>9}{'prec':>8}{'rec':>8}{'F1':>8}")
        m = M[:, org]
        t = T[:, org][m]
        for nm, Pr in (("persistence + rule", R), (f"ensemble (n={len(P)})", E)):
            pr = Pr[:, org][m]
            p_, r_, f_, sup = prf(t, pr, labels=[0, 1, 2], zero_division=0)
            for c in range(3):
                print(f"  {nm if c == 0 else '':30s}{c:>7}{int(sup[c]):>9}{p_[c]:>8.3f}{r_[c]:>8.3f}{f_[c]:>8.3f}")


def cmd_itemnb(a):
    """Item-specific decision curves (reviewer request): net benefit computed PER ITEM, not pooled.

    The pooled analysis assigns one indifference probability p_t to all 45 items, which is the wrong
    model of a ward round: re-reading the working diagnosis and re-reading a rare infection site are
    not the same act and do not carry the same miss cost. This reports, for each item, the range of
    p_t over which alerting on that item beats both trivial policies -- so a deployment can switch
    items on individually rather than accept or reject the alert as a block.
    """
    sys.path.insert(0, KEEP); import gtsq_eval as ge
    N = np.load(os.path.join(KEEP, a.split, "norm_stats.npz"), allow_pickle=True)
    FT = [str(x) for x in N["gtsq_ftypes"]]; NC = [int(x) for x in N["gtsq_nclasses"]]
    FE = [str(x) for x in N["gtsq_features"]]
    gm, gs = N["gtsq_means"].astype(float), N["gtsq_stds"].astype(float)

    def arrays(name):
        d = np.load(os.path.join(KEEP, a.split, f"{name}.npz"), allow_pickle=True)
        gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
        im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
        I = ge.denorm_to_class(gi, gm, gs, FT, NC); T = ge.denorm_to_class(d["gtsq_out"], gm, gs, FT, NC)
        return I, T, (im == 1) & (d["gtsq_out_mask"] == 1)
    Itr, Ttr, Mtr = arrays("train"); Ite, Tte, Mte = arrays("test")

    rate = {}                                   # training-free lookup, fitted on TRAIN only
    for j in range(len(FE)):
        for c in range(NC[j]):
            m = Mtr[:, j] & (Itr[:, j] == c)
            if m.sum() >= 5:
                rate[(j, c)] = float((Itr[m, j] != Ttr[m, j]).mean())
    glob = float((Itr[Mtr] != Ttr[Mtr]).mean())

    pts = [0.02, 0.05, 0.10, 0.20, 0.50]
    print("=" * 104)
    print(f"ITEM-SPECIFIC NET BENEFIT -- test, alert = score >= item's own {a.budget:.0%} budget")
    print("=" * 104)
    print(f"  {'item':40s}{'base%':>8}{'PPV':>8}{'sens':>7}   " + "".join(f"{p:>9.2f}" for p in pts))
    rows = []
    for j in range(len(FE)):
        m = Mte[:, j]
        n = int(m.sum())
        if n < 100:
            continue
        y = (Ite[m, j] != Tte[m, j]).astype(int)
        sc = np.array([rate.get((j, int(v)), glob) for v in Ite[m, j]])
        k = max(int(round(a.budget * n)), 1)
        thr = np.sort(sc)[::-1][k - 1]
        fire = sc >= thr
        tp = int((fire & (y == 1)).sum()); fp = int((fire & (y == 0)).sum())
        ppv = tp / max(tp + fp, 1); sens = tp / max(int(y.sum()), 1)
        nb = [(tp / n) - (fp / n) * (p / (1 - p)) for p in pts]
        nball = [(y.mean()) - (1 - y.mean()) * (p / (1 - p)) for p in pts]
        wins = sum(1 for x, xa in zip(nb, nball) if x > max(xa, 0))
        rows.append((FE[j], 100 * y.mean(), ppv, sens, nb, nball, wins, n))
    rows.sort(key=lambda r: -r[6])
    for f, base, ppv, sens, nb, nball, wins, n in rows:
        mark = "".join(f"{x:>+9.4f}" if x > max(xa, 0) else f"{'--':>9}" for x, xa in zip(nb, nball))
        print(f"  {f[:38]:40s}{base:>8.2f}{ppv:>8.3f}{sens:>7.2f}   {mark}   [{wins}/5]")
    print("\n  Cells show net benefit only where the item's alert beats BOTH trivial policies at that")
    print("  p_t; `--` means it does not. Items are sorted by how many of the five thresholds it wins.")
    nwin = sum(1 for r in rows if r[6] == len(pts))
    print(f"\n  {nwin} of {len(rows)} items beat both policies at every p_t; "
          f"{sum(1 for r in rows if r[6] == 0)} at none.")


def cmd_detector(a):
    """Training-free detector: AUROC decomposition, and the same lookup under a chronological split.

    Two paper claims live here. (1) The 0.836 AUROC decomposes into item identity (which question)
    and within-item state (which patient) -- reported because the pooled figure is more impressive
    than the patient-specific signal it carries. (2) Refitting the lookup on an earlier period and
    scoring a later one measures how the artefact behaves under real distribution shift.
    """
    sys.path.insert(0, KEEP); import gtsq_eval as ge
    from sklearn.metrics import roc_auc_score, average_precision_score

    def arrays(split, name):
        n = np.load(os.path.join(KEEP, split, "norm_stats.npz"), allow_pickle=True)
        ft = [str(x) for x in n["gtsq_ftypes"]]; nc = [int(x) for x in n["gtsq_nclasses"]]
        gm, gs = n["gtsq_means"].astype(float), n["gtsq_stds"].astype(float)
        d = np.load(os.path.join(KEEP, split, f"{name}.npz"), allow_pickle=True)
        gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
        im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
        return (ge.denorm_to_class(gi, gm, gs, ft, nc),
                ge.denorm_to_class(d["gtsq_out"], gm, gs, ft, nc),
                (im == 1) & (d["gtsq_out_mask"] == 1), nc)

    def score(split):
        Itr, Ttr, Mtr, nc = arrays(split, "train"); Ite, Tte, Mte, _ = arrays(split, "test")
        G = Itr.shape[1]
        rate, irate = {}, np.zeros(G)
        for j in range(G):
            irate[j] = float((Itr[Mtr[:, j], j] != Ttr[Mtr[:, j], j]).mean())
            for c in range(nc[j]):
                m = Mtr[:, j] & (Itr[:, j] == c)
                if m.sum() >= 5:
                    rate[(j, c)] = float((Itr[m, j] != Ttr[m, j]).mean())
        glob = float((Itr[Mtr] != Ttr[Mtr]).mean())
        sc, y, it = [], [], []
        for j in range(G):
            for i in np.where(Mte[:, j])[0]:
                sc.append(rate.get((j, int(Ite[i, j])), glob)); y.append(int(Ite[i, j] != Tte[i, j]))
                it.append(j)
        return np.array(sc), np.array(y), np.array(it), irate

    print("=" * 96)
    print("TRAINING-FREE CHANGE DETECTOR")
    print("=" * 96)
    for label, split in (("patient split", a.split), ("chronological split", a.temporal)):
        if not os.path.isdir(os.path.join(KEEP, split)):
            print(f"  {label}: split not built ({split}) -- skipped"); continue
        sc, y, it, irate = score(split)
        order = np.argsort(-sc)
        line = (f"  {label:20s} cells {len(y):,}  base {100*y.mean():.2f}%  "
                f"AUROC {roc_auc_score(y, sc):.4f}  AUPRC {average_precision_score(y, sc):.4f}")
        for b in (0.005, 0.02, 0.10):
            sel = order[:int(round(b * len(y)))]
            line += f"   {b:.1%}: PPV {100*y[sel].mean():.1f}% sens {100*y[sel].sum()/y.sum():.1f}%"
        print(line)
        if label == "patient split":
            w = [(roc_auc_score(y[it == j], sc[it == j]), (it == j).sum())
                 for j in np.unique(it) if 0 < y[it == j].sum() < (it == j).sum()]
            w = np.array(w)
            print(f"    decomposition: item identity alone {roc_auc_score(y, irate[it]):.4f}   "
                  f"within-item (state only, support-weighted) "
                  f"{np.average(w[:, 0], weights=w[:, 1]):.4f}")
            _detector_extras(a, split, sc, y, it, irate, roc_auc_score, average_precision_score)


def _detector_extras(a, split, sc, y, it, irate, roc_auc_score, average_precision_score):
    """The three checks OPEN_ITEMS O11 asks to be RUN, not merely specified.

    (1) Rule-covered transitions removed. The organ-acuity `neu -> akut` step is already produced by
        a one-line rule, so any alert on it is not new information. Pooled discrimination is
        therefore recomputed on the changes the rule does NOT cover -- the honest question is what
        the score adds beyond the baseline a reader gets for free.
    (2) Item identity vs item+state, on exactly the same cells, with the patient as the resampling
        unit. Reported as a paired difference: the two scores are computed on the same cells, so an
        unpaired interval would be far too wide to see the gap between them.
    (3) A breakdown by item block. Pooled AUROC mixes volatile judgement fields with near-static
        infection sites; the pooled figure can be high because the score sorts those blocks apart.
    """
    import numpy as np
    n = np.load(os.path.join(KEEP, split, "norm_stats.npz"), allow_pickle=True)
    FE = [str(x) for x in n["gtsq_features"]]
    d = np.load(os.path.join(KEEP, split, "test.npz"), allow_pickle=True)
    gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
    im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
    sys.path.insert(0, KEEP); import gtsq_eval as ge
    ft = [str(x) for x in n["gtsq_ftypes"]]; nc = [int(x) for x in n["gtsq_nclasses"]]
    I = ge.denorm_to_class(gi, n["gtsq_means"].astype(float), n["gtsq_stds"].astype(float), ft, nc)
    T = ge.denorm_to_class(d["gtsq_out"], n["gtsq_means"].astype(float),
                           n["gtsq_stds"].astype(float), ft, nc)
    M = (im == 1) & (d["gtsq_out_mask"] == 1)
    clust = (np.asarray(d["pat_ids"]).astype(str) if "pat_ids" in d.files
             else np.asarray(d["enc_ids"]).astype(str))
    unit = "patient" if "pat_ids" in d.files else "encounter"

    # cells are enumerated item-major in cmd_detector.score, so rebuild the row index the same way
    rows = np.concatenate([np.where(M[:, j])[0] for j in range(M.shape[1])])
    ORG = np.array(["akut" in f.lower() for f in FE])
    covered = np.array([ORG[j] and I[r, j] == 1 and T[r, j] == 2 for r, j in zip(rows, it)])

    keep = ~covered
    print(f"\n  (1) rule-covered `neu->akut` cells removed: {covered.sum():,} of {y.sum():,} changes "
          f"({covered.sum()/max(y.sum(),1):.1%} of all changes)")
    print(f"      pooled AUROC {roc_auc_score(y, sc):.4f} -> {roc_auc_score(y[keep], sc[keep]):.4f}   "
          f"AUPRC {average_precision_score(y, sc):.4f} -> "
          f"{average_precision_score(y[keep], sc[keep]):.4f}   "
          f"(base rate {y.mean():.2%} -> {y[keep].mean():.2%})")

    if a.bootstrap:
        uc = np.unique(clust); idx_of = {c: np.where(clust == c)[0] for c in uc}
        rowset = {c: np.isin(rows, idx_of[c]) for c in uc}
        rng = np.random.RandomState(0); dif, pool, item = [], [], []
        for _ in range(a.bootstrap):
            pick = rng.randint(0, len(uc), len(uc))
            sel = np.concatenate([np.where(rowset[uc[k]])[0] for k in pick])
            yy = y[sel]
            if yy.min() == yy.max():
                continue
            ap = roc_auc_score(yy, sc[sel]); ai = roc_auc_score(yy, irate[it[sel]])
            pool.append(ap); item.append(ai); dif.append(ap - ai)
        pool, item, dif = np.array(pool), np.array(item), np.array(dif)
        print(f"\n  (2) paired bootstrap, {len(dif)} usable resamples, cluster = {unit} "
              f"({len(uc):,} clusters)")
        print(f"      item+state AUROC {pool.mean():.4f} "
              f"[{np.percentile(pool,2.5):.4f},{np.percentile(pool,97.5):.4f}]")
        print(f"      item only  AUROC {item.mean():.4f} "
              f"[{np.percentile(item,2.5):.4f},{np.percentile(item,97.5):.4f}]")
        print(f"      paired difference {dif.mean():+.4f} "
              f"[{np.percentile(dif,2.5):+.4f},{np.percentile(dif,97.5):+.4f}]"
              + ("  (excludes zero)" if np.percentile(dif,2.5) > 0 or np.percentile(dif,97.5) < 0
                 else "  (includes zero)"))

    def block(f):
        fl = f.lower()
        if fl.startswith(("i02", "i03", "i04")) or "gruppe" in fl:  return "judgement"
        if fl.startswith("i05"):                                    return "infection site"
        return "organ"
    blocks = np.array([block(FE[j]) for j in it])
    print(f"\n  (3) by item block")
    print(f"      {'block':16s}{'cells':>10}{'changes':>9}{'base':>8}{'AUROC':>8}{'AUPRC':>8}")
    for b in ("judgement", "infection site", "organ"):
        m = blocks == b
        if m.sum() == 0 or y[m].min() == y[m].max():
            continue
        print(f"      {b:16s}{m.sum():>10,}{int(y[m].sum()):>9,}{y[m].mean():>8.2%}"
              f"{roc_auc_score(y[m], sc[m]):>8.4f}{average_precision_score(y[m], sc[m]):>8.4f}")




# =============================================================================================
# FMTS submission: ONE evaluation pass behind every reported number.
# =============================================================================================
# The audit that motivated this (OPEN_ITEMS O03/O04) found the manuscript's rows could not all have
# come from one evaluation: two cell counts (250,536 vs 251,049), two change prevalences (3.80% vs
# 3.86%), and paired deltas that did not reconcile with the point estimates they were derived from.
# The cause is structural, not arithmetic -- rows were harvested from different scripts, masks and
# aggregation conventions over several months. Recomputing them one at a time would reproduce the
# same failure, so `fmts` takes a MANIFEST of rows and evaluates all of them in a single process
# against a single cohort, mask, label version and split, emitting:
#
#   fmts_predictions.npz   per-cell predictions for every row + the cohort keys (pat/enc/item)
#   fmts_metrics.csv       one row per predictor: Acc, macro-F1, Acc_c, FCR, F1_c, both aggregations
#   fmts_manifest.json     the provenance stamp: split, mask, label version, run dirs, seeds, git
#   fmts_ci.csv            patient-clustered paired bootstrap CIs against the reference predictor
#   fmts_departures.csv    DIRECT paired departure counts for the figure (not reconstructed)
#
# Every number in the paper must be traceable to these five files.

# Rows are (name, inputs, candidate globs, kind). Globs are relative to --models_root, so the same
# manifest evaluates the main, matched-cutoff, raw-label and temporal campaigns.
#   baseline  training-free, computed here
#   fixed     one configuration, five seeds
#   select    SEVERAL candidate configurations; the one with the best VALIDATION macro-F1 is chosen
#             and only that one is scored on the requested split. Selection happens inside this
#             script, on validation, every time -- not by reading a test table and picking a winner.
FMTS_ROWS = [
    ("Persistence",                       "Q",   [], "baseline"),
    ("Persistence + rule",                "Q",   [], "baseline"),
    ("Transition matrix",                 "Q",   [], "baseline"),
    ("GRU without baseline gate",         "Q+P", ["nogate/*"], "fixed"),
    ("Gate over persistence",             "Q+P", ["gate_copy/*"], "fixed"),
    ("Gate over rule, macro-F1 selected", "Q+P", ["gate_rule_a/*", "gate_rule_b/*",
                                                  "gate_rule_c/*"], "select"),
    ("Change-weighted objective",         "Q+P", ["changew/*"], "fixed"),
]


def _fcr(P, I, T, M):
    """False-change rate: P(prediction departs from day D | the label did not change).

    The counterpart to Acc_c. Equation 2 in the paper is an identity between the two, so both must
    come from the same mask or the identity silently stops holding.
    """
    stable = M & (T == I)
    return float((P[stable] != I[stable]).mean()) if stable.sum() else float("nan")


def _metrics_row(P, I, T, M, NC, ge):
    acc, f1, ca, f1c = ge.four_metrics(P, I, T, M, NC)
    return dict(acc=float(acc), macro_f1=float(f1), acc_change=float(ca),
                f1_change=float(f1c), fcr=_fcr(P, I, T, M))


def _point_metrics(entry, I, T, M, NC, ge, report):
    """The metric of the REPORTED object: the mean over seeds, or the ensemble."""
    if report == "seedmean" and "seeds" in entry:
        per = [_metrics_row(P, I, T, M, NC, ge) for P in entry["seeds"]]
        return {k: float(np.mean([q[k] for q in per])) for k in per[0]}
    return _metrics_row(entry["point"], I, T, M, NC, ge)


def _departures(P, I, T, M):
    """Direct paired counts against persistence -- what the figure must show (O16).

    Three outcomes, not two: a departure that fixes a cell persistence got wrong, a departure that
    breaks a cell persistence got right, and a departure that was wrong before and after. The
    reconstructed figure could not distinguish the third because it inverted an aggregate identity.
    """
    dep = M & (P != I)
    pers_ok = (I == T)
    return dict(n_scored=int(M.sum()), n_departures=int(dep.sum()),
                fixed=int((dep & ~pers_ok & (P == T)).sum()),
                broken=int((dep & pers_ok).sum()),
                futile=int((dep & ~pers_ok & (P != T)).sum()))


def cmd_fmts(a):
    """Single-pass evaluation of every manuscript row on one cohort/mask/label version."""
    import json, csv, subprocess
    p8 = _load("p8", "08_probes.py")
    bl = _load("bl", "05_baselines.py")
    sys.path.insert(0, KEEP); import gtsq_eval as ge

    out = os.path.join(KEEP, a.out_dir); os.makedirs(out, exist_ok=True)
    split = a.split
    N = np.load(os.path.join(KEEP, split, "norm_stats.npz"), allow_pickle=True)
    FT = [str(x) for x in N["gtsq_ftypes"]]; NC = [int(x) for x in N["gtsq_nclasses"]]
    FE = [str(x) for x in N["gtsq_features"]]
    gm = N["gtsq_means"].astype(float); gs = N["gtsq_stds"].astype(float)

    def _load_split(name, sdir=None):
        d = np.load(os.path.join(KEEP, sdir or split, f"{name}.npz"), allow_pickle=True)
        gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
        im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
        I = ge.denorm_to_class(gi, gm, gs, FT, NC)
        T = ge.denorm_to_class(d["gtsq_out"], gm, gs, FT, NC)
        M = (d["gtsq_out_mask"] == 1)
        if a.mask == "both":
            M = M & (im == 1)
        return d, I, T, M

    dte, I, T, M = _load_split(a.split_name)
    dtr, Itr, Ttr, Mtr = _load_split("train")

    # --mask_from restricts scoring to the cells observed in ANOTHER split, matched row by row on
    # the (encounter, date) key. This isolates the effect of the observation mask: the predictions,
    # the inputs and the targets are unchanged, only which cells count changes. Built for the
    # explicit-only cohort, where the questionnaire hierarchy is not zero-filled, so that the
    # headline statistics can be recomputed over cells a clinician actually filled in. A row with no
    # counterpart there had fewer than the required number of explicit items and is scored nowhere.
    if a.mask_from:
        dm = np.load(os.path.join(KEEP, a.mask_from, f"{a.split_name}.npz"), allow_pickle=True)
        mi = dm["gtsq_in_mask"]; mi = mi[:, -1, :] if mi.ndim == 3 else mi
        alt = (dm["gtsq_out_mask"] == 1) & (mi == 1)
        key = {k: i for i, k in enumerate(np.asarray(dm["dates"]).astype(str))}
        rows = [key.get(k, -1) for k in np.asarray(dte["dates"]).astype(str)]
        miss = sum(1 for r in rows if r < 0)
        M2 = np.zeros_like(M)
        for i, r in enumerate(rows):
            if r >= 0:
                M2[i] = alt[r]
        before, after = int(M.sum()), int((M & M2).sum())
        print(f"  --mask_from {os.path.basename(a.mask_from)}: "
              f"{after:,} of {before:,} scored cells retained ({after/before:.1%}); "
              f"{miss} of {len(rows)} pairs have no counterpart")
        M = M & M2

    # Clustering key. pat_ids is written by 02_preprocess.py for every split built after the
    # patient-split fix; older splits fall back to the encounter, which is a WEAKER cluster and is
    # recorded as such in the manifest rather than passed off as a patient bootstrap.
    if "pat_ids" in dte.files:
        clust = np.asarray(dte["pat_ids"]).astype(str); clust_unit = "patient"
    else:
        clust = np.asarray(dte["enc_ids"]).astype(str); clust_unit = "encounter"
        print("  WARNING: split has no pat_ids; clustering by ENCOUNTER, not patient")

    ORG = np.array(["akut" in f.lower() for f in FE])
    R = I.copy()
    for j in np.where(ORG)[0]:
        R[:, j] = np.where(I[:, j] == 1, 2, I[:, j])
    MK = bl.baseline_markov(Itr, Ttr, Mtr, I, NC)

    print("=" * 92)
    print(f"FMTS single-pass evaluation -- split={a.split_name.upper()}  dir={split}")
    ne = len(np.unique(dte["enc_ids"])); ncl = len(np.unique(clust))
    print(f"  cohort: {len(I):,} day-pairs, {ne:,} encounters"
          + (f", {ncl:,} patients" if clust_unit == "patient" else ""))
    print(f"  mask={a.mask}  scored cells={M.sum():,}  change prevalence="
          f"{(M & (T != I)).sum() / M.sum():.4%}")
    print("=" * 92)

    from models.gated_predictor import build_feature_slices
    SL, _tot = build_feature_slices(list(zip(FT, NC)))

    def _replay(run_dirs, dsplit, Ibase):
        """Per-cell class predictions for each run, on the given split arrays."""
        outs, used = [], []
        for r in run_dirs:
            try:
                model, cfg, _, _ = p8.reconstruct(r)
                cols = p8.clinical_cols_for(cfg, N)
                sp_ = {k: dsplit[k] for k in dsplit.files}
                # _forward_pred_gate refuses a model with no gamma, which is exactly the no-gate
                # row: it is a legitimate predictor, not a broken run, so fall back to the plain
                # forward pass instead of dropping the row (the smoke test dropped all five seeds).
                if getattr(cfg, "use_gate", False):
                    lg, _ = p8._forward_pred_gate(model, sp_,
                                                  getattr(cfg, "use_clinical", False),
                                                  clin_cols=cols)
                else:
                    lg = p8._forward_pred(model, sp_, getattr(cfg, "use_clinical", False))
                    if isinstance(lg, tuple):
                        lg = lg[0]
            except (Exception, SystemExit) as ex:
                print(f"    skip {os.path.basename(r)[:48]}: {type(ex).__name__}: {ex}"); continue
            P = Ibase.copy()
            for g, s_ in enumerate(SL):
                if FT[g] not in ("binary", "ordinal"):
                    continue
                col = lg[:, s_.start:s_.stop]
                P[:, g] = (col[:, 0] > 0).astype(int) if FT[g] == "binary" else col.argmax(1)
            outs.append(P); used.append(r)
        return outs, used

    def _runs_for(g):
        return sorted(r for r in glob.glob(os.path.join(KEEP, a.models_root, g))
                      if os.path.exists(os.path.join(r, "model.pytorch")))

    # validation arrays, loaded once, used only for candidate selection
    dva = Iva = Tva = Mva = None
    if any(k == "select" for *_x, k in FMTS_ROWS) and a.split_name != "val":
        dva, Iva, Tva, Mva = _load_split("val")

    preds, prov = {}, {}
    for name, inputs, globs, kind in FMTS_ROWS:
        if kind == "baseline":
            P = {"Persistence": I, "Persistence + rule": R, "Transition matrix": MK}[name]
            preds[name] = {"point": P, "inputs": inputs, "agg": "deterministic", "runs": []}
            prov[name] = dict(inputs=inputs, aggregation="deterministic", runs=[], seeds=[],
                              selected_from=None)
            continue

        chosen, sel_note = globs[0], None
        if kind == "select" and len(globs) > 1:
            # Choose the candidate configuration on VALIDATION macro-F1 (seed mean), then score only
            # that one on the requested split. When the requested split IS val, this is still the
            # same computation -- it just means the selection and the report coincide.
            scores = {}
            for g in globs:
                rs = _runs_for(g)
                if not rs:
                    continue
                if a.split_name == "val":
                    Ps, _ = _replay(rs, dte, I)
                    sc = [ge.four_metrics(P, I, T, M, NC)[1] for P in Ps]
                else:
                    Ps, _ = _replay(rs, dva, Iva)
                    sc = [ge.four_metrics(P, Iva, Tva, Mva, NC)[1] for P in Ps]
                if sc:
                    scores[g] = float(np.mean(sc))
            if not scores:
                print(f"  !! no candidate matched for row '{name}' -- SKIPPED"); continue
            chosen = max(scores, key=scores.get)
            sel_note = {g: round(v, 5) for g, v in scores.items()}
            print(f"  select '{name}': " +
                  "  ".join(f"{g.split('/')[0]}={v:.4f}" for g, v in sorted(scores.items())) +
                  f"  -> {chosen.split('/')[0]} (validation macro-F1)")

        runs = _runs_for(chosen)
        if not runs:
            print(f"  !! no runs matched {chosen} under {a.models_root} -- row '{name}' SKIPPED")
            continue
        per_seed, used = _replay(runs, dte, I)
        if not per_seed:
            print(f"  !! row '{name}' produced no predictions -- SKIPPED"); continue
        st = np.stack(per_seed)
        vote = np.zeros_like(st[0])
        for g in range(st.shape[2]):
            col = st[:, :, g]
            vote[:, g] = [np.bincount(col[:, i]).argmax() for i in range(col.shape[1])]
        preds[name] = {"point": vote, "seeds": st, "inputs": inputs, "agg": kind, "runs": used}
        prov[name] = dict(inputs=inputs, aggregation=kind,
                          runs=[os.path.relpath(r, KEEP) for r in used],
                          seeds=[seed_of_run(r) for r in used],
                          selected_from=sel_note, selected=chosen)

    # ---- metrics: BOTH aggregations reported, never mixed (O04) --------------------------------
    rows = []
    for name, d_ in preds.items():
        m = _metrics_row(d_["point"], I, T, M, NC, ge)
        m.update(predictor=name, inputs=d_["inputs"], aggregation="ensemble (majority vote)"
                 if d_["agg"] != "deterministic" else "deterministic", n_runs=len(d_["runs"]))
        rows.append(m)
        if "seeds" in d_:
            per = [_metrics_row(P, I, T, M, NC, ge) for P in d_["seeds"]]
            mm = {k: float(np.mean([q[k] for q in per])) for k in per[0]}
            sd = {k + "_sd": float(np.std([q[k] for q in per], ddof=1)) for k in per[0]}
            mm.update(sd); mm.update(predictor=name, inputs=d_["inputs"],
                                     aggregation=f"seed mean (n={len(per)})", n_runs=len(per))
            rows.append(mm)
    keys = ["predictor", "inputs", "aggregation", "n_runs", "acc", "macro_f1", "acc_change",
            "fcr", "f1_change"] + [k for k in rows[-1] if k.endswith("_sd")]
    with open(os.path.join(out, "fmts_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"  {'predictor':38s}{'agg':22s}{'Acc':>8}{'macF1':>9}{'Acc_c':>8}{'FCR':>8}{'F1_c':>8}")
    for r in rows:
        print(f"  {r['predictor'][:37]:38s}{r['aggregation'][:21]:22s}{r['acc']:>8.4f}"
              f"{r['macro_f1']:>9.4f}{r['acc_change']:>8.4f}{r['fcr']:>8.4f}{r['f1_change']:>8.4f}")

    # ---- direct departure counts for the figure (O16) ------------------------------------------
    with open(os.path.join(out, "fmts_departures.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["predictor", "n_scored", "n_departures", "fixed",
                                          "broken", "futile", "fixed_per_1000", "broken_per_1000"])
        w.writeheader()
        # COUNTS, not rates. The header used to say "per 1,000 scored cells" while printing raw
        # counts, so the printed table and the figure (which does normalise) disagreed by a factor
        # of ~282. Both are now shown, labelled.
        print(f"\n  DIRECT departures from persistence -- raw counts, and per 1,000 scored cells")
        print(f"  {'predictor':38s}{'depart':>9}{'fixed':>9}{'broken':>9}{'futile':>9}"
              f"{'fix/1k':>9}{'brk/1k':>9}")
        for name, d_ in preds.items():
            if a.report == "seedmean" and "seeds" in d_:
                # counts averaged over the five fixed seeds: the figure then shows the same
                # predictor as the table, instead of a vote the table never reports
                ds = [_departures(P, I, T, M) for P in d_["seeds"]]
                dep = {k: (float(np.mean([q[k] for q in ds])) if k != "n_scored" else ds[0][k])
                       for k in ds[0]}
            else:
                dep = _departures(d_["point"], I, T, M)
            dep["predictor"] = name
            dep["fixed_per_1000"] = 1000 * dep["fixed"] / dep["n_scored"]
            dep["broken_per_1000"] = 1000 * dep["broken"] / dep["n_scored"]
            w.writerow(dep)
            print(f"  {name[:37]:38s}{dep['n_departures']:>9,}{dep['fixed']:>9,}"
                  f"{dep['broken']:>9,}{dep['futile']:>9,}"
                  f"{dep['fixed_per_1000']:>9.2f}{dep['broken_per_1000']:>9.2f}")

    # ---- patient-clustered paired bootstrap (O06) ----------------------------------------------
    # REPORTING OBJECT. A five-seed mean and a five-model majority vote are different predictors and
    # produce different numbers (here 0.9609 vs 0.9624 accuracy for the same row). An earlier version
    # of this script reported the seed mean in the table and bootstrapped the ensemble, so every
    # interval in the paper belonged to a predictor the table did not show. `--report` now fixes ONE
    # object for the point estimate, the interval, the paired contrast and the figure.
    #   seedmean  the metric is recomputed for each fixed seed inside every bootstrap resample and
    #             then averaged, which is the statistic Table 1 displays;
    #   ensemble  the majority-vote predictions are bootstrapped directly.
    if a.bootstrap:
        uc = np.unique(clust)
        idx_of = {c: np.where(clust == c)[0] for c in uc}
        rng = np.random.RandomState(a.seed)
        ref = a.reference
        names = list(preds)
        assert ref in preds, f"--reference {ref} is not an evaluated row"
        boot = {n: [] for n in names}
        for _ in range(a.bootstrap):
            # resample CLUSTERS with replacement, keeping every day and item of a sampled cluster
            # together; the same resample scores every predictor so the deltas stay paired
            pick = rng.randint(0, len(uc), len(uc))
            idx = np.concatenate([idx_of[uc[k]] for k in pick])
            Ib, Tb, Mb = I[idx], T[idx], M[idx]
            for n in names:
                if a.report == "seedmean" and "seeds" in preds[n]:
                    per = [_metrics_row(P[idx], Ib, Tb, Mb, NC, ge) for P in preds[n]["seeds"]]
                    boot[n].append({k: float(np.mean([q[k] for q in per])) for k in per[0]})
                else:
                    boot[n].append(_metrics_row(preds[n]["point"][idx], Ib, Tb, Mb, NC, ge))
        mk = ["acc", "macro_f1", "acc_change", "fcr", "f1_change"]
        with open(os.path.join(out, "fmts_ci.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["predictor", "metric", "point", "boot_mean", "lo95", "hi95",
                        f"delta_vs_{ref}", "delta_lo95", "delta_hi95", "excludes_zero"])
            print(f"\n  PAIRED BOOTSTRAP, {a.bootstrap} resamples, cluster = {clust_unit} "
                  f"({len(uc):,} clusters), reference = {ref}")
            for n in names:
                B = {k: np.array([b[k] for b in boot[n]]) for k in mk}
                Bref = {k: np.array([b[k] for b in boot[ref]]) for k in mk}
                for k in mk:
                    pt = _point_metrics(preds[n], I, T, M, NC, ge, a.report)[k]
                    lo, hi = np.percentile(B[k], [2.5, 97.5])
                    d = B[k] - Bref[k]
                    dlo, dhi = np.percentile(d, [2.5, 97.5])
                    w.writerow([n, k, f"{pt:.6f}", f"{B[k].mean():.6f}", f"{lo:.6f}", f"{hi:.6f}",
                                f"{d.mean():+.6f}", f"{dlo:+.6f}", f"{dhi:+.6f}",
                                int(n != ref and (dlo > 0 or dhi < 0))])
                if n != ref:
                    dd = {k: np.array([b[k] for b in boot[n]]) - np.array([b[k] for b in boot[ref]])
                          for k in mk}
                    cells = "".join(f"{dd[k].mean():>+9.4f}[{np.percentile(dd[k],2.5):+.4f},"
                                    f"{np.percentile(dd[k],97.5):+.4f}]" for k in mk)
                    print(f"  {n[:30]:31s}{cells}")

    # ---- LaTeX table body, written by the evaluator itself -------------------------------------
    # Numbers reach the manuscript without being retyped. Manual transcription between a results
    # table and a .tex file is exactly how the four irreconcilable paired differences in the previous
    # version could arise, so the paper \input{}s this file instead. Rows absent from the run are
    # emitted as a visible marker, never silently dropped.
    tex = os.path.join(out, "fmts_table1.tex")
    with open(tex, "w") as f:
        f.write("% GENERATED by eval_deploy.py fmts -- do not edit by hand.\n")
        f.write(f"% split={split}  split_name={a.split_name}  mask={a.mask}  "
                f"labels={a.label_version}  cells={int(M.sum())}\n")
        want = "seed mean" if a.report == "seedmean" else "ensemble"
        seedmean = {r["predictor"]: r for r in rows if r["aggregation"].startswith(want)}
        det = {r["predictor"]: r for r in rows if r["aggregation"] == "deterministic"}
        groups = [["Persistence", "Persistence + rule", "Transition matrix"],
                  ["GRU without baseline gate", "Gate over persistence",
                   "Gate over rule, macro-F1 selected", "Change-weighted objective"]]
        for gi, grp in enumerate(groups):
            if gi:
                f.write("\\midrule\n")
            for nm in grp:
                r = det.get(nm) or seedmean.get(nm)
                inp = "Q" if nm in groups[0] else "Q+P"
                if r is None:
                    f.write(f"{nm} & {inp} & \\pending{{}} & \\pending{{}} & \\pending{{}} "
                            f"& \\pending{{}} \\\\\n")
                    continue
                f.write(f"{nm} & {inp} & {r['acc']:.4f} & {r['macro_f1']:.4f} & "
                        f"{r['acc_change']:.4f} & {r['fcr']:.4f} \\\\\n")
    print(f"  wrote {tex}")

    # ---- provenance stamp (O03/O04/O10) --------------------------------------------------------
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=KEEP, capture_output=True,
                             text=True).stdout.strip() or None
    except Exception:
        git = None
    man = dict(split_dir=split, split_name=a.split_name, mask=a.mask,
               label_version=a.label_version, cluster_unit=clust_unit,
               n_pairs=int(len(I)), n_encounters=int(len(np.unique(dte["enc_ids"]))),
               n_clusters=int(len(np.unique(clust))), n_scored_cells=int(M.sum()),
               change_prevalence=float((M & (T != I)).sum() / M.sum()),
               n_items=len(FE), items=FE, nclasses=NC, ftypes=FT,
               bootstrap=a.bootstrap, reference=a.reference, seed=a.seed, git=git,
               report_object=a.report, mask_from=a.mask_from,
               rows=prov)
    with open(os.path.join(out, "fmts_manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    np.savez_compressed(os.path.join(out, "fmts_predictions.npz"),
                        day_d=I, target=T, mask=M, cluster=clust,
                        enc_ids=np.asarray(dte["enc_ids"]).astype(str), items=np.array(FE),
                        **{f"pred::{n}": d_["point"] for n, d_ in preds.items()})
    print(f"\n  wrote {out}/fmts_{{metrics,departures,ci,manifest,predictions}}.*")


def seed_of_run(r):
    import re
    m = re.search(r"_s(\d+)", os.path.basename(r))
    return int(m.group(1)) if m else None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("override"); o.set_defaults(fn=cmd_override)
    o.add_argument("--run", required=True, nargs="+"); o.add_argument("--split", required=True)
    o.add_argument("--base", choices=["copy", "markov"], default="markov")
    o.add_argument("--purity", type=float, default=0.7)
    o.add_argument("--thresholds", type=float, nargs="+", default=[0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    o.add_argument("--seeds", type=int, default=5)
    o.add_argument("--confirm-on-test", dest="confirm_on_test", action="store_true",
                   help="after the val sweep, score the SELECTED threshold once on test")
    o.add_argument("--split_name", default="val", choices=["val", "test"],
                   help="evaluation split for the threshold sweep; VAL by default because choosing a "
                        "threshold is model selection. Historical doc tables used test.")
    e = sub.add_parser("ensemble"); e.set_defaults(fn=cmd_ensemble)
    e.add_argument("--runs", nargs="+", required=True,
                   help="glob(s) of run dirs to ensemble, relative to keepgate/")
    e.add_argument("--split", default="exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_clin_cult")
    e.add_argument("--runs_b", nargs="+", default=None,
                   help="second group of runs. When given, ARP and the bootstrap test group A "
                        "against THIS group instead of against --base -- the ablation contrasts "
                        "(cultures on/off, order-only vs result-only, physiology on/off) are "
                        "model-vs-model and cannot be expressed as model-vs-baseline.")
    e.add_argument("--split_b", default=None,
                   help="split for --runs_b (defaults to --split). Ablations that change the INPUTS "
                        "live in a different split directory while sharing the same targets; the "
                        "rows are verified identical before pairing.")
    e.add_argument("--base", choices=["copy", "markov"], default="markov",
                   help="reference the ARP test compares against: markov = persistence + the "
                        "new->acute rule (the paper's baseline), copy = plain persistence")
    e.add_argument("--arp", type=int, default=0,
                   help="paired approximate-randomisation test vs the base, N shuffles (Riezler & "
                        "Maxwell 2005). Swap unit is the day-pair. 10000 gives p down to 1e-4.")
    e.add_argument("--bootstrap", type=int, default=0,
                   help="paired bootstrap resamples over day-pairs for 95%% CIs on every metric and "
                        "on the difference against the rule (e.g. 2000)")
    e.add_argument("--mask", default="both", choices=["both", "target"],
                   help="target = target cell observed; both = both days observed (change is defined)")
    e.add_argument("--per_class", action="store_true",
                   help="print per-class precision/recall/F1 on the organ acuity items")
    e.add_argument("--split_name", default="val", choices=["val", "test"],
                   help="VAL by default; `test` must be asked for explicitly")
    dt = sub.add_parser("detector"); dt.set_defaults(fn=cmd_detector)
    dt.add_argument("--split", default="exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_clin_cult")
    dt.add_argument("--temporal", default="exp_fixnn/01_split_sets/gtsq_temporal_24h_gap6h_clin_cult")
    dt.add_argument("--bootstrap", type=int, default=0,
                    help="patient-clustered paired bootstrap of item+state vs item-only AUROC (O11)")
    fm = sub.add_parser("fmts"); fm.set_defaults(fn=cmd_fmts)
    fm.add_argument("--split", default="exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_clin_cult_patsplit",
                    help="ONE split directory behind every row (patient-disjoint by default)")
    fm.add_argument("--split_name", default="val", choices=["val", "test"],
                    help="VAL by default; test must be asked for explicitly")
    fm.add_argument("--mask", default="both", choices=["both", "target"],
                    help="both = both days observed, the only mask under which a change is defined")
    fm.add_argument("--label_version", default="corrected",
                    help="stamp recorded in the manifest: corrected (neu->neu fixed) or raw")
    fm.add_argument("--bootstrap", type=int, default=0,
                    help="paired bootstrap resamples clustered on the PATIENT (e.g. 2000)")
    fm.add_argument("--reference", default="Persistence + rule",
                    help="row every paired delta is taken against")
    fm.add_argument("--seed", type=int, default=0)
    fm.add_argument("--mask_from", default=None,
                    help="restrict scoring to cells observed in this split too, matched on "
                         "(encounter, date); use the explicit-only split to exclude cells created "
                         "by the questionnaire hierarchy zero-fill")
    fm.add_argument("--report", default="seedmean", choices=["seedmean", "ensemble"],
                    help="the ONE object behind every reported number: the five-seed mean (default, "
                         "recomputed inside each bootstrap resample) or the majority-vote ensemble")
    fm.add_argument("--models_root", default="exp_fixnn/02_models_fmts",
                    help="directory the row globs are resolved against, so one manifest serves the "
                         "main, matched-cutoff, raw-label and temporal campaigns")
    fm.add_argument("--out_dir", default="exp_fixnn/fmts_eval")
    n_ = sub.add_parser("itemnb"); n_.set_defaults(fn=cmd_itemnb)
    n_.add_argument("--split", default="exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_clin_cult")
    n_.add_argument("--budget", type=float, default=0.02)
    for nm, fn in (("alarm", cmd_alarm), ("decision", cmd_decision)):
        q = sub.add_parser(nm); q.set_defaults(fn=fn)
        q.add_argument("--split", default="exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_noclin")
        q.add_argument("--run", default=None)
    a = p.parse_args(); a.fn(a)


