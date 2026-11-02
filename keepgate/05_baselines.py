#!/usr/bin/env python3
"""Non-neural baselines for next-day GTSQ prediction (rebuttal, reviewer request).

Reviewers MPiz / ysvz / R1fp asked whether deep learning provides any value over
simple non-neural baselines on this persistence-dominated task. This script
implements, on the SAME splits and with the SAME metrics as 03_train.py:

  persistence    : predict tomorrow = today (sanity reproduction of the paper baseline)
  markov         : per-feature first-order Markov transition  P(y_{D+1} | y_D)
  logreg         : per-feature L2 multinomial logistic regression on yesterday's GTSQ
  gbt            : per-feature gradient-boosted trees on yesterday's GTSQ
  twostage       : per-feature change detector -> conditional next-class classifier
  blup           : per-feature Gaussian random-intercept BLUP (AR(1) + patient random
                   effect); because the split is by-patient, the random effect is predicted
                   ONLINE via empirical Bayes from the patient's own observed residuals so far

Input to learned baselines = yesterday's full (normalized) GTSQ vector — exactly the
information the GTSQ-only ("noclin") neural model receives. Targets/metrics are
restricted to features with an available target (out_mask == 1), identical to eval_gtsq.

Usage:
  python 05_baselines.py --data_dir 01_split_sets/gtsq_pred_24h_gap6h_first_noclin
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gtsq_eval as ge   # shared decode + metric definitions
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, cohen_kappa_score


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(data_dir, name):
    z = np.load(os.path.join(data_dir, f"{name}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def to_class(arr_norm, means, stds, ftypes, nclasses):
    """Denormalize and round to integer class labels (shared definition, see gtsq_eval)."""
    return ge.denorm_to_class(arr_norm, means, stds, ftypes, nclasses)


# ---------------------------------------------------------------------------
# Metric — replicates eval_gtsq exactly (macro over binary+ordinal features)
# ---------------------------------------------------------------------------

def evaluate(pred_cls, in_cls, out_cls, out_mask, ftypes, nclasses):
    """Return dict with f1_macro, kappa, f1_change, accuracy, acc_change and per-feature detail.

    pred_cls : (N, G) predicted class per feature
    in_cls   : (N, G) yesterday's class (for change detection)
    out_cls  : (N, G) target class
    out_mask : (N, G) 1 if target available

    Accuracy convention matches 03_train.py: BOTH 'accuracy' and 'acc_change' are CELL-POOLED
    (fraction of observed / of truly-changed cells predicted correctly), weighting every cell
    equally rather than averaging per-feature accuracies. macro-F1 stays per-feature-averaged
    (that is what 'macro' means). Per-feature accuracy is still kept in per_feat[g]['accuracy'].

    The metric definitions now live once in gtsq_eval.per_feature_report (single source of truth);
    this wrapper is kept so callers importing 05_baselines.evaluate keep working.
    """
    return ge.per_feature_report(pred_cls, in_cls, out_cls, out_mask, ftypes, nclasses)


# ---------------------------------------------------------------------------
# Baselines — each returns predicted class array (N_eval, G)
# ---------------------------------------------------------------------------

def baseline_persistence(in_cls_eval):
    return in_cls_eval.copy()


def baseline_zeroforecast(out_cls_tr, out_mask_tr, in_cls_eval, nclasses):
    """Predict the per-feature MAJORITY (most frequent) class, ignoring the input entirely.
    The 'no-memory' counterpart to persistence: where persistence copies yesterday, ZeroForecast
    always emits the modal class (estimated from the TRAIN targets). On a persistence-dominated
    task it scores high cell-accuracy on near-static features (the majority *is* yesterday) but
    catches changes only when they happen to land on the modal class, so it is the natural floor to
    separate 'copying yesterday' from 'guessing the base rate'. Reported in the paper as the
    *majority-class* baseline."""
    G = in_cls_eval.shape[1]
    pred = np.zeros_like(in_cls_eval)
    for g in range(G):
        nc = int(nclasses[g])
        y = np.clip(out_cls_tr[out_mask_tr[:, g] == 1, g], 0, nc - 1)
        pred[:, g] = int(np.bincount(y, minlength=nc).argmax()) if len(y) else 0
    return pred


def baseline_alwayschange(in_cls_tr, out_cls_tr, out_mask_tr, in_cls_eval, nclasses):
    """Adversarial 'always switch': never copy -- always emit the most likely class OTHER than
    yesterday's, per (feature, current-state) from train.

    Purpose: the mirror image of ZeroForecast. Where the majority baseline exposes that
    change-accuracy is gameable by a memoryless predictor, this exposes that change-accuracy and
    DIRECTION are gameable by a predictor that never copies -- trivially so on the 25 BINARY items,
    where a change has exactly one possible destination, so 'always flip' is right 100% of the time
    on every changed binary cell and its direction is correct by construction. Any metric this
    bluffs cannot be read on its own.
    """
    G = in_cls_eval.shape[1]
    pred = in_cls_eval.copy()
    for g in range(G):
        nc = int(nclasses[g])
        if nc < 2:      # continuous / unusable feature -> classification is undefined
            continue    # pred was initialised from the day-D values, so this leaves persistence
        m = out_mask_tr[:, g] == 1
        for s in range(nc):
            sel = m & (in_cls_tr[:, g] == s)
            alt = [c for c in range(nc) if c != s]
            if not alt:
                continue
            if sel.sum():
                cnt = np.bincount(np.clip(out_cls_tr[sel, g], 0, nc - 1), minlength=nc).astype(float)
                cnt[s] = -1.0                       # forbid the self-transition
                nxt = int(np.argmax(cnt))
            else:
                nxt = alt[0]
            pred[in_cls_eval[:, g] == s, g] = nxt
    return pred


def baseline_markov(in_cls_tr, out_cls_tr, out_mask_tr, in_cls_eval, nclasses):
    """Per-feature first-order Markov: argmax_c P(y'=c | y=state)."""
    G = in_cls_tr.shape[1]
    pred = np.zeros_like(in_cls_eval)
    for g in range(G):
        nc = int(nclasses[g])
        if nc < 2:      # continuous / unusable feature -> markov is undefined; fall back to copy
            # `pred` is already initialised to the day-D copy, so leaving the column untouched IS
            # the persistence fallback. The previous line named in_cls_eval, which is not a
            # parameter here (pasted from a function where it is) and raised NameError instead.
            continue
        m = out_mask_tr[:, g] == 1
        s = in_cls_tr[m, g]
        t = out_cls_tr[m, g]
        trans = np.zeros((nc, nc), dtype=float)
        for a, b in zip(s, t):
            if 0 <= a < nc and 0 <= b < nc:
                trans[a, b] += 1
        marginal = np.bincount(t[(t >= 0) & (t < nc)], minlength=nc).argmax() if len(t) else 0
        nxt = np.array([trans[a].argmax() if trans[a].sum() > 0 else marginal
                        for a in range(nc)])
        pred[:, g] = nxt[np.clip(in_cls_eval[:, g], 0, nc - 1)]
    return pred


def baseline_markov_cond(in_cls_tr, out_cls_tr, out_mask_tr, in_cls_eval, nclasses,
                         cond_idx, min_count=20):
    """Markov conditioned on extra context, with hierarchical backoff.

    Plain markov predicts argmax_c P(y'=c | feature g, state s). This conditions further on the
    day-D value of the features in `cond_idx` (e.g. working diagnosis, infection suspicion):
    argmax_c P(y'=c | g, s, ctx). A lookup table, so unlike the neural --use_supervisor embedding
    it costs no capacity and cannot lose to optimisation -- if the context is exploitable at all,
    this finds it.

    Sparsity is the whole difficulty: 45 features x states x |ctx| cells over ~19k train pairs.
    We therefore BACK OFF -- a (g, s, ctx) cell with < min_count observations falls back to the
    plain (g, s) table, and an unseen state falls back to copying.
    """
    G = in_cls_eval.shape[1]
    pred = in_cls_eval.copy()
    for g in range(G):
        nc = int(nclasses[g])
        if nc < 2:      # continuous / unusable feature -> markov is undefined; fall back to copy
            pred[:, g] = in_cls_eval[:, g]
            continue
        m = out_mask_tr[:, g] == 1
        if m.sum() < 1:
            continue
        # backoff level: plain (g, state)
        base = {}
        for s in range(nc):
            sel = m & (in_cls_tr[:, g] == s)
            if sel.sum():
                base[s] = int(np.bincount(np.clip(out_cls_tr[sel, g], 0, nc - 1),
                                          minlength=nc).argmax())
        # conditioned level: (g, state, ctx)
        def key(arr, i):
            return tuple(int(arr[i, c]) for c in cond_idx)
        tab = {}
        ctr = collections.defaultdict(list)
        idx = np.where(m)[0]
        for i in idx:
            ctr[(int(in_cls_tr[i, g]),) + key(in_cls_tr, i)].append(int(out_cls_tr[i, g]))
        for k, v in ctr.items():
            if len(v) >= min_count:
                tab[k] = int(np.bincount(np.clip(v, 0, nc - 1), minlength=nc).argmax())
        for i in range(len(in_cls_eval)):
            s = int(in_cls_eval[i, g])
            k = (s,) + key(in_cls_eval, i)
            if k in tab:
                pred[i, g] = tab[k]
            elif s in base:
                pred[i, g] = base[s]
    return pred


def baseline_hmm_smooth(in_cls_tr, out_cls_tr, out_mask_tr, in_cls_eval, nclasses, alpha=1.0):
    """Per-feature first-order Markov with empirical-Bayes (hierarchical) smoothing: shrink each
    state's transition row toward the marginal next-day distribution by pseudo-count alpha
    (a Dirichlet prior centred on the marginal), then predict the posterior argmax. Unlike the
    hard tau-thresholded markov, smoothing lets confident transitions surface from sparse states
    without overfitting rare counts — the reviewer's Tier-C 'smoothed transition / hierarchical
    ordinal Markov' baseline. alpha=0 reduces to the plain count-argmax markov."""
    G = in_cls_tr.shape[1]
    pred = np.zeros_like(in_cls_eval)
    for g in range(G):
        nc = int(nclasses[g])
        if nc < 2:      # continuous / unusable feature -> markov is undefined; fall back to copy
            pred[:, g] = in_cls_eval[:, g]
            continue
        m = out_mask_tr[:, g] == 1
        s = np.clip(in_cls_tr[m, g], 0, nc - 1)
        t = np.clip(out_cls_tr[m, g], 0, nc - 1)
        trans = np.zeros((nc, nc), dtype=float)
        np.add.at(trans, (s, t), 1.0)
        marg = np.bincount(t, minlength=nc).astype(float)
        marg = marg / marg.sum() if marg.sum() > 0 else np.full(nc, 1.0 / nc)
        post = trans + alpha * marg[None, :]            # P(t|s) ∝ counts[s] + alpha * marginal
        nxt = post.argmax(axis=1)
        pred[:, g] = nxt[np.clip(in_cls_eval[:, g], 0, nc - 1)]
    return pred


def _fit_per_feature(X_tr, out_cls_tr, out_mask_tr, X_eval, nclasses, make_clf):
    """Fit one classifier per target feature; predict classes on X_eval."""
    G = out_cls_tr.shape[1]
    pred = np.zeros((X_eval.shape[0], G), dtype=int)
    for g in range(G):
        nc = int(nclasses[g])
        if nc < 2:      # continuous / unusable feature -> markov is undefined; fall back to copy
            pred[:, g] = in_cls_eval[:, g]
            continue
        m = out_mask_tr[:, g] == 1
        y = out_cls_tr[m, g]
        if m.sum() < 10 or len(np.unique(y)) < 2:
            # degenerate: predict the single observed / majority class
            pred[:, g] = np.bincount(y, minlength=nc).argmax() if len(y) else 0
            continue
        clf = make_clf()
        clf.fit(X_tr[m], y)
        pred[:, g] = clf.predict(X_eval)
    return pred


def baseline_logreg(X_tr, out_cls_tr, out_mask_tr, X_eval, nclasses, C):
    return _fit_per_feature(
        X_tr, out_cls_tr, out_mask_tr, X_eval, nclasses,
        lambda: LogisticRegression(C=C, max_iter=2000, class_weight='balanced'))


def baseline_gbt(X_tr, out_cls_tr, out_mask_tr, X_eval, nclasses):
    return _fit_per_feature(
        X_tr, out_cls_tr, out_mask_tr, X_eval, nclasses,
        # early_stopping=False: the auto split is stratified and blows up when a rare
        # GTSQ ordinal class has <2 members; fixed max_iter keeps it deterministic.
        lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                max_depth=3, l2_regularization=1.0,
                                                early_stopping=False))


def baseline_twostage(X_tr, in_cls_tr, out_cls_tr, out_mask_tr,
                      X_eval, in_cls_eval, nclasses, C):
    """Stage 1: per-feature change detector (changed = y' != y).
       Stage 2: among predicted-changed samples, multinomial classifier of y'.
       Predicted-stable samples keep yesterday's class (persistence)."""
    G = out_cls_tr.shape[1]
    pred = in_cls_eval.copy()
    for g in range(G):
        nc = int(nclasses[g])
        if nc < 2:      # continuous / unusable feature -> markov is undefined; fall back to copy
            pred[:, g] = in_cls_eval[:, g]
            continue
        m = out_mask_tr[:, g] == 1
        if m.sum() < 10:
            continue
        changed = (out_cls_tr[m, g] != in_cls_tr[m, g]).astype(int)
        if len(np.unique(changed)) < 2:
            continue
        det = LogisticRegression(C=C, max_iter=2000, class_weight='balanced')
        det.fit(X_tr[m], changed)
        pred_changed = det.predict(X_eval).astype(bool)
        # Stage 2 trained on changed training rows only
        ch_tr = m & (out_cls_tr[:, g] != in_cls_tr[:, g])
        y2 = out_cls_tr[ch_tr, g]
        if ch_tr.sum() >= 10 and len(np.unique(y2)) >= 2 and pred_changed.sum() > 0:
            clf = LogisticRegression(C=C, max_iter=2000, class_weight='balanced')
            clf.fit(X_tr[ch_tr], y2)
            pred[pred_changed, g] = clf.predict(X_eval[pred_changed])
    return pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def baseline_arima_ets(in_eval, mask_eval, enc_eval, dates_eval, nclasses,
                       method="arima", min_obs=4):
    """Classical univariate TSF per (encounter, item), one-step-ahead with an expanding window.

    This is the standard forecasting toolkit applied honestly to a daily ordinal questionnaire, and
    it is here to place the bottom rung of the ladder: if ARIMA/ETS cannot beat copying, "use a
    proper time-series model" is not the missing ingredient.

    Three deliberate choices, each of which flatters the classical methods rather than us:
      * refits are ONLINE (expanding window, one step ahead), never fit on the whole episode -- a
        model fit over the full stay would see the value it is asked to predict;
      * a CONSTANT history is short-circuited to that constant. ARIMA and ETS both return the
        constant there, so the fit is skipped rather than risked on a degenerate series. This is
        not an approximation -- it is what both methods return -- and it is what makes the run
        tractable, since ~96% of cells never move;
      * continuous forecasts are rounded to the nearest valid class and clipped, and any series
        shorter than `min_obs` or any fit that fails falls back to persistence. Every fallback is
        counted and reported, because "ARIMA ties LOCF" means something very different when it is
        a fitted result than when it is a fallback.
    """
    import warnings
    G = in_eval.shape[1]
    pred = in_eval.astype(float).copy()                          # default = persistence
    enc_e = np.asarray([str(x) for x in enc_eval])
    dat_e = np.asarray([str(x) for x in dates_eval])
    stats = dict(fitted=0, constant=0, too_short=0, failed=0, rw_order=0)
    by_enc = {}
    for i in range(len(enc_e)):
        by_enc.setdefault(enc_e[i], []).append(i)
    for e, idx in by_enc.items():
        idx = sorted(idx, key=lambda i: dat_e[i])
        for g in range(G):
            nc = int(nclasses[g])
            obs = [i for i in idx if mask_eval[i, g] == 1]
            if not obs:
                continue
            hist = []
            for i in obs:
                v = float(in_eval[i, g])
                hist.append(v)
                if len(hist) < min_obs:
                    stats["too_short"] += 1; continue        # stays at persistence
                if np.ptp(hist) < 1e-9:
                    stats["constant"] += 1; continue         # ARIMA/ETS both return the constant
                y = np.asarray(hist, dtype=float)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if method == "arima":
                            from statsmodels.tsa.arima.model import ARIMA
                            # The candidate grid is restricted to a SINGLE differencing order.
                            # AIC is not comparable across d: differencing drops an observation, so
                            # the d=1 likelihood is computed on a shorter series and its AIC is
                            # systematically smaller. The old grid mixed d=0 and d=1, which biased
                            # selection toward the differenced orders and inflated the share of fits
                            # attributed to the random walk. Fixing d=1 keeps the question ("does
                            # order selection land on the random walk?") well posed and the
                            # comparison valid.
                            best, best_aic = None, np.inf
                            for order in ((0, 1, 0), (1, 1, 0), (2, 1, 0)):
                                try:
                                    f = ARIMA(y, order=order).fit()
                                except Exception:
                                    continue
                                if f.aic < best_aic:
                                    best, best_aic, best_ord = f, f.aic, order
                            if best is None:
                                stats["failed"] += 1; continue
                            if best_ord == (0, 1, 0):
                                stats["rw_order"] += 1       # a random walk IS persistence
                            yh = float(np.asarray(best.forecast(1)).ravel()[0])
                        else:
                            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                            f = SimpleExpSmoothing(y, initialization_method="estimated").fit()
                            yh = float(np.asarray(f.forecast(1)).ravel()[0])
                except Exception:
                    stats["failed"] += 1; continue
                if not np.isfinite(yh):
                    stats["failed"] += 1; continue
                pred[i, g] = float(np.clip(round(yh), 0, nc - 1))
                stats["fitted"] += 1
    # rw_order counts a SUBSET of `fitted`, so it must not enter the denominator -- including it
    # reported ARIMA as fitting 16.9% of cells when the true share is 19.5%, the same as ETS (which
    # has no rw_order and was therefore already correct). The two methods see identical series, so a
    # differing "fitted" share was the tell.
    tot = (stats["fitted"] + stats["constant"] + stats["too_short"] + stats["failed"]) or 1
    print(f"  [{method}] fitted {stats['fitted']:,} ({stats['fitted']/tot:.1%})  "
          f"constant->LOCF {stats['constant']:,}  short->LOCF {stats['too_short']:,}  "
          f"failed->LOCF {stats['failed']:,}")
    if method == "arima" and stats["fitted"]:
        print(f"  [{method}] order selection chose the random walk (0,1,0) -- i.e. LOCF -- in "
              f"{stats['rw_order']:,}/{stats['fitted']:,} fits ({stats['rw_order']/stats['fitted']:.1%})")
    return pred.astype(int)


def baseline_blup(in_tr, out_tr, mask_tr, enc_tr,
                  in_eval, out_eval, mask_eval, enc_eval, dates_eval, nclasses):
    """Per-feature Gaussian random-intercept BLUP: y' = b0 + b1·y + u_patient + ε.

    Fixed effects (b0,b1) and the variance components (τ² between-patient, σ² within) are
    fit on train. The split is by-patient, so eval patients are unseen → the random intercept
    is predicted ONLINE by empirical Bayes from the patient's own observed residuals so far in
    the eval episode: û = (n·τ²)/(n·τ²+σ²)·r̄, processing each episode in date order. Ordinal/
    binary targets are handled by rounding the Gaussian prediction (a deliberate baseline
    simplification). Degenerate features fall back to persistence.
    """
    G = in_tr.shape[1]
    pred = in_eval.astype(float).copy()                         # default = persistence
    enc_e = np.asarray([str(x) for x in enc_eval])
    dat_e = np.asarray([str(x) for x in dates_eval])
    order = sorted(range(len(enc_e)), key=lambda i: (enc_e[i], dat_e[i]))
    enc_t = np.asarray([str(x) for x in enc_tr])
    for g in range(G):
        nc = int(nclasses[g])
        mt = mask_tr[:, g] == 1
        x = in_tr[mt, g].astype(float); y = out_tr[mt, g].astype(float)
        if mt.sum() < 30 or np.std(x) < 1e-9 or len(np.unique(y)) < 2:
            continue
        try:
            A = np.c_[np.ones_like(x), x]
            b = np.linalg.lstsq(A, y, rcond=None)[0]
            r = y - A @ b
            _, inv = np.unique(enc_t[mt], return_inverse=True)
            gn = np.bincount(inv).astype(float)
            gmean = np.bincount(inv, weights=r) / gn
            gvar = np.maximum(np.bincount(inv, weights=r * r) / gn - gmean ** 2, 0.0)
            within = float((gvar * gn).sum() / max(gn.sum(), 1.0))
            tau2 = max(float(np.var(gmean)) - within / max(gn.mean(), 1.0), 0.0)
            sig2 = max(within, 1e-6)
            run = {}
            for i in order:
                if mask_eval[i, g] != 1:
                    continue
                fixed = b[0] + b[1] * float(in_eval[i, g])
                s, n = run.get(enc_e[i], (0.0, 0))
                uhat = (n * tau2) / (n * tau2 + sig2) * (s / n) if n > 0 else 0.0
                pred[i, g] = fixed + uhat
                run[enc_e[i]] = (s + (float(out_eval[i, g]) - fixed), n + 1)
            pred[:, g] = np.clip(np.round(pred[:, g]), 0, nc - 1)
        except np.linalg.LinAlgError:
            # Only a genuinely singular least-squares fit falls back to persistence; any OTHER
            # exception (a real bug) propagates instead of being silently hidden as "BLUP ties
            # persistence here".
            pred[:, g] = in_eval[:, g]
    return pred.astype(int)


def main(args):
    np.random.seed(args.seed)
    d = args.data_dir
    norm = np.load(os.path.join(d, "norm_stats.npz"), allow_pickle=True)
    means = norm['gtsq_means'].astype(np.float32)
    stds = norm['gtsq_stds'].astype(np.float32)
    ftypes = norm['gtsq_ftypes']
    nclasses = norm['gtsq_nclasses']
    gtsq_features = [str(x) for x in norm['gtsq_features']]

    tr = load_split(d, "train")
    va = load_split(d, "val")
    te = load_split(d, "test")

    def prep(split):
        gin = split['gtsq_in']
        if gin.ndim == 3:          # multi-day history -> use most recent day
            gin = gin[:, -1, :]
        in_cls = to_class(gin, means, stds, ftypes, nclasses)
        out_cls = to_class(split['gtsq_out'], means, stds, ftypes, nclasses)
        om = split['gtsq_out_mask']
        if getattr(args, "mask", "target") == "both":
            # a cell counts only when BOTH days are observed: where day D is missing there is nothing
            # to copy and no change to define, so scoring it credits a prediction nobody could make
            im = split['gtsq_in_mask']; im = im[:, -1, :] if im.ndim == 3 else im
            om = ((om == 1) & (im == 1)).astype(om.dtype)
        return gin, in_cls, out_cls, om

    Xtr, in_tr, out_tr, mask_tr = prep(tr)
    Xva, in_va, out_va, mask_va = prep(va)
    Xte, in_te, out_te, mask_te = prep(te)

    # Tier C (review): extra transition baselines only -> baseline_results_extra.json. Returns
    # early so the committed baseline_results.json is never recomputed/overwritten; 07_aggregate
    # reads the _extra file additively (original wins on any name clash).
    if getattr(args, "extra", False):
        extra = {"hmm_smooth": baseline_hmm_smooth(in_tr, out_tr, mask_tr, in_te, nclasses,
                                                    alpha=args.hmm_alpha)}
        print(f"\n{'extra baseline':<14}{'F1_macro':>10}{'Kappa':>10}{'F1_change':>12}")
        print("-" * 46)
        res = {}
        for name, p in extra.items():
            r = evaluate(p, in_te, out_te, mask_te, ftypes, nclasses)
            res[name] = {k: v for k, v in r.items() if k != 'per_feat'}
            print(f"{name:<14}{r['f1_macro']:>10.4f}{r['kappa']:>10.4f}{r['f1_change']:>12.4f}")
        out_path = os.path.join(d, "baseline_results_extra.json")
        with open(out_path, "w") as f:
            json.dump({'data_dir': d, 'hmm_alpha': args.hmm_alpha, 'results': res}, f, indent=2)
        print(f"\nSaved -> {out_path}")
        return

    # Optional +clinical ablation (reviewer-facing): append a per-variable masked-mean
    # summary of the day-D clinical window to the feature matrix for the learned baselines
    # (logreg/gbt/twostage). persistence/markov are score-only by construction and unchanged.
    if getattr(args, "use_clinical", False):
        def clin_feats(split):
            c = split.get("clinical")
            if c is None or c.ndim != 3 or c.shape[2] == 0:
                return None
            V = c.shape[2] // 2
            val, msk = c[:, :, :V], c[:, :, V:]
            denom = msk.sum(axis=1); denom[denom == 0] = 1.0
            return ((val * msk).sum(axis=1) / denom).astype(np.float32)   # (N, V) masked mean
        ctr, cva, cte = clin_feats(tr), clin_feats(va), clin_feats(te)
        if ctr is None:
            print("  --use_clinical set but split has no clinical channels; ignoring.")
        else:
            Xtr = np.hstack([Xtr, ctr]); Xva = np.hstack([Xva, cva]); Xte = np.hstack([Xte, cte])
            print(f"  +clinical: appended {ctr.shape[1]} per-variable masked-mean features")

    # Select logistic-regression C on validation (aggregate macro-F1)
    best_C, best_f1 = 1.0, -1.0
    for C in args.C_grid:
        p = baseline_logreg(Xtr, out_tr, mask_tr, Xva, nclasses, C)
        f1 = evaluate(p, in_va, out_va, mask_va, ftypes, nclasses)['f1_macro']
        print(f"  [val] logreg C={C}: F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_C = f1, C
    print(f"  selected C={best_C}")

    # The conditioned markov variants below use GTSQ item indices; only enable them when the
    # split has enough features for those indices to exist.
    # The conditioned variants below index feature positions 2 and 3, which are the GTSQ
    # working diagnosis and infection suspicion. A width test does NOT identify that: the
    # 5/6-feature SOFA splits and 39-feature eICU all pass it, and then report a variant
    # named "markov+wdx" that is actually conditioned on sofa_liver. Require the real items.
    _is_gtsq = (len(gtsq_features) > 3 and gtsq_features[2].startswith("i03000")
                and gtsq_features[3].startswith("i04000"))
    preds = {
        'persistence':  baseline_persistence(in_te),
        'zeroforecast': baseline_zeroforecast(out_tr, mask_tr, in_te, nclasses),
        'alwayschange': baseline_alwayschange(in_tr, out_tr, mask_tr, in_te, nclasses),
        'markov':       baseline_markov(in_tr, out_tr, mask_tr, in_te, nclasses),
        # Indices 2 and 3 are the GTSQ working diagnosis and infection suspicion -- context features
        # that exist only in that instrument. The MIMIC targets have 2-6 features and eICU 39
        # unrelated ones, so these conditioned variants are skipped there rather than indexing out
        # of bounds (which is what they did: aki has 2 features, support 3).
        **({'markov+wdx':  baseline_markov_cond(in_tr, out_tr, mask_tr, in_te, nclasses, [2]),
            'markov+inf':  baseline_markov_cond(in_tr, out_tr, mask_tr, in_te, nclasses, [3]),
            'markov+both': baseline_markov_cond(in_tr, out_tr, mask_tr, in_te, nclasses, [2, 3])}
           if _is_gtsq else {}),
        'logreg':      baseline_logreg(Xtr, out_tr, mask_tr, Xte, nclasses, best_C),
        'gbt':         baseline_gbt(Xtr, out_tr, mask_tr, Xte, nclasses),
        'twostage':    baseline_twostage(Xtr, in_tr, out_tr, mask_tr,
                                         Xte, in_te, nclasses, best_C),
        'blup':        baseline_blup(in_tr, out_tr, mask_tr, tr['enc_ids'],
                                     in_te, out_te, mask_te, te['enc_ids'], te['dates'], nclasses),
        # --tsf only: these refit online per (encounter, item) and are far slower than the rest.
        **({'arima': baseline_arima_ets(in_te, mask_te, te['enc_ids'], te['dates'], nclasses,
                                        method="arima", min_obs=args.tsf_min_obs),
            'ets':   baseline_arima_ets(in_te, mask_te, te['enc_ids'], te['dates'], nclasses,
                                        method="ets", min_obs=args.tsf_min_obs)}
           if getattr(args, 'tsf', False) else {}),
    }

    print(f"\n{'baseline':<14}{'Acc':>9}{'F1_macro':>10}{'Acc_chg':>9}{'F1_change':>11}{'Kappa':>9}")
    print("-" * 62)
    results = {}
    for name, p in preds.items():
        r = evaluate(p, in_te, out_te, mask_te, ftypes, nclasses)
        # --save_per_feat keeps the per-feature breakdown (e.g. per SOFA subscore) for the
        # circularity decomposition; default strips it for the compact comparison table.
        results[name] = r if getattr(args, "save_per_feat", False) else {k: v for k, v in r.items() if k != 'per_feat'}
        print(f"{name:<14}{r['accuracy']:>9.4f}{r['f1_macro']:>10.4f}"
              f"{r['acc_change']:>9.4f}{r['f1_change']:>11.4f}{r['kappa']:>9.4f}")

    # --diagnose: why do markov/GBT beat persistence on accuracy but lose on macro-F1?
    # Focus on the 3-class "akut oder neu" organ features (0=no, 1=acute, 2=new), where the
    # near-deterministic new->acute rule lives. Reports per model: per-CLASS F1 (a class that is
    # never predicted shows as 0), how often class "new" is emitted at all, and accuracy split
    # into the rule-firing cells (day-D == new) vs the rest.
    if getattr(args, "diagnose", False):
        from sklearn.metrics import f1_score as _f1
        AN = [g for g in range(in_te.shape[1])
              if int(nclasses[g]) == 3 and 'akut' in gtsq_features[g]]
        # Encoding verified empirically from the transition matrix: state 1 is RARE (n=568 test)
        # and leaves 97.7% of the time -> "new"; state 2 is common and persists 95.6% -> "acute".
        print(f"\n=== DIAGNOSE: {len(AN)} 'akut oder neu' features (0=no, 1=NEW, 2=ACUTE) ===")
        for split, I_, O_, M_ in (("train", in_tr, out_tr, mask_tr), ("test", in_te, out_te, mask_te)):
            C = collections.Counter()
            for g in AN:
                m = M_[:, g] == 1
                for a, b in zip(I_[m, g], O_[m, g]):
                    C[(int(a), int(b))] += 1
            print(f"    [{split}] full day-D -> day-D+1 transition matrix (rows = day-D state):")
            for a in (0, 1, 2):
                tot = sum(C[(a, b)] for b in (0, 1, 2))
                cells = "  ".join(f"->{b}: {C[(a, b)]:6d} ({C[(a,b)]/tot*100:5.2f}%)" if tot else f"->{b}:      -"
                                  for b in (0, 1, 2))
                print(f"      state {a}  n={tot:6d}   {cells}")

        print("\n  per-CLASS F1 on these 8 features, #times 'new' is PREDICTED, and accuracy split:")
        print(f"    {'model':<13}{'F1(no)':>8}{'F1(new)':>9}{'F1(acute)':>10}{'macroF1':>9}{'#pred acute':>13}{'acc':>8}")
        for name, p in preds.items():
            T = np.concatenate([out_te[mask_te[:, g] == 1, g] for g in AN]).astype(int)
            P = np.concatenate([np.clip(p[mask_te[:, g] == 1, g], 0, 2) for g in AN]).astype(int)
            I = np.concatenate([in_te[mask_te[:, g] == 1, g] for g in AN]).astype(int)
            per = _f1(T, P, average=None, labels=[0, 1, 2], zero_division=0)
            fire = I == 1                  # rule-firing cells: day-D state == "new"
            af = float((P[fire] == T[fire]).mean()) if fire.sum() else float('nan')
            ar = float((P[~fire] == T[~fire]).mean())
            print(f"    {name:<13}{per[0]:>8.3f}{per[1]:>9.3f}{per[2]:>9.3f}{float(np.mean(per)):>9.3f}"
                  f"{int((P == 2).sum()):>13d}{float((P == T).mean()):>8.4f}"
                  f"   [new-state cells acc {af:.3f} (n={int(fire.sum())}) | rest {ar:.4f}]")

        # Is DIRECTION accuracy a binary artefact? Split it: on a binary item a change has exactly
        # one destination, so direction is correct by construction and carries no information.
        BIN = [g for g in range(in_te.shape[1]) if str(ftypes[g]) == 'binary']
        ORD = [g for g in range(in_te.shape[1]) if int(nclasses[g]) > 2]
        print("\n  === DIRECTION accuracy, split by feature type (the binary artefact) ===")
        print(f"    {'model':<14}{'dir on BINARY':>15}{'dir on ORDINAL':>16}{'chg-acc BIN':>13}{'chg-acc ORD':>13}")
        for name, p_ in preds.items():
            row = []
            for grp in (BIN, ORD):
                dok = dall = cok = call = 0
                for g in grp:
                    m = (mask_te[:, g] == 1) & (in_te[:, g] != out_te[:, g])
                    if not m.sum(): continue
                    pg = np.clip(p_[m, g], 0, int(nclasses[g]) - 1)
                    cok += int((pg == out_te[m, g]).sum()); call += int(m.sum())
                    mv = pg != in_te[m, g]
                    if mv.sum():
                        dok += int((np.sign(pg[mv] - in_te[m, g][mv]) ==
                                    np.sign(out_te[m, g][mv] - in_te[m, g][mv])).sum())
                        dall += int(mv.sum())
                row.append((dok / dall if dall else float('nan'), cok / call if call else float('nan')))
            print(f"    {name:<14}{row[0][0]:>15.3f}{row[1][0]:>16.3f}{row[0][1]:>13.3f}{row[1][1]:>13.3f}")
        print("    (binary direction is 1.000 by construction whenever the model moves at all)")

        agr = lambda a, b: float((preds[a] == preds[b]).mean())
        print(f"\n  whole-matrix agreement: markov~gbt {agr('markov','gbt'):.4f} | "
              f"markov~persist {agr('markov','persistence'):.4f} | gbt~persist {agr('gbt','persistence'):.4f}")

    # --tsf writes its OWN file. circularity.py and report.py read baseline_results_clin.json for
    # the persistence and GBT-oracle halves of every circularity number, so a --tsf run overwriting
    # it silently re-dated the inputs of analyses that had nothing to do with ARIMA. --extra was
    # given a separate file for exactly this reason; --tsf needs the same.
    if getattr(args, "tsf", False):
        fname = "baseline_results_tsf.json"
    elif getattr(args, "use_clinical", False):
        fname = "baseline_results_clin.json"
    else:
        fname = "baseline_results.json"
    out_path = os.path.join(d, fname)
    with open(out_path, "w") as f:
        json.dump({'data_dir': d, 'logreg_C': best_C, 'results': results}, f, indent=2)
    print(f"\nSaved -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="01_split_sets/gtsq_pred_24h_gap6h_first_noclin")
    p.add_argument("--C_grid", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    p.add_argument("--use_clinical", action="store_true", default=False,
                   help="Append per-variable masked-mean clinical features to logreg/gbt/"
                        "twostage (persistence/markov stay score-only). Writes "
                        "baseline_results_clin.json.")
    p.add_argument("--tsf", action="store_true", default=False,
                   help="also run the classical univariate forecasters (ARIMA, ETS), one-step-ahead "
                        "with an expanding window per (encounter, item). Off by default: they refit "
                        "online and cost minutes rather than seconds.")
    p.add_argument("--tsf_min_obs", type=int, default=4,
                   help="shortest history ARIMA/ETS are asked to fit; below this they fall back to "
                        "persistence and the fallback is counted")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask", type=str, default="both", choices=["both", "target"],
                   help="which cells count: `both` = observed on day D and D+1 (default since "
                        "2026-08-04); `target` = the historical convention")
    p.add_argument("--diagnose", action="store_true", default=False,
                   help="per-class F1 / rule-cell breakdown on the akut-oder-neu features")
    p.add_argument("--extra", action="store_true", default=False,
                   help="Tier C: compute only the extra transition baselines (hmm_smooth) and "
                        "write baseline_results_extra.json; never touches baseline_results.json.")
    p.add_argument("--hmm_alpha", type=float, default=1.0,
                   help="(--extra) Dirichlet pseudo-count for hmm_smooth (shrink to marginal)")
    p.add_argument("--save_per_feat", action="store_true", default=False,
                   help="Keep the per-feature F1 breakdown in the json (e.g. per SOFA subscore "
                        "for the circularity decomposition).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
