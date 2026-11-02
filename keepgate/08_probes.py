#!/usr/bin/env python3
"""Training-free interpretability probes on already-trained keep-gate models (Paper 3).

Reconstructs a model from its run dir (config.json) + the split's norm_stats and
model.pytorch, then runs one of three probes — no retraining, inference only:

  wpers   : W_pers diagonal-dominance. For a LINEAR persistence path (skip_proj), read
            the learned weight and measure how much of each output feature's persistence
            mass sits on its OWN previous-day value vs other features. The submitted paper
            claims the learned path is NOT diagonal (it mixes features), which is the
            mechanistic reason the linear/skip/base variants land below true persistence.

  gate    : gate calibration. Run the gate model over the split, collect per-feature
            gamma = sigma(gate_head), and relate mean gamma to each feature's day-to-day
            change rate (and gamma on changed vs unchanged samples). Tests whether the gate
            opens (overrides persistence) preferentially on the volatile targets.

  permute : cross-modal permutation importance. Shuffle one clinical variable across
            patients in the test window and re-evaluate per-feature F1, isolating whether
            a specific clinical->GTSQ link is used (e.g. bilirubin -> Leber_present). Most
            informative on the clinical-only model, which must rely on the clinical signal.

Usage (from keepgate/):
  python 08_probes.py --run_dir 02_models/<run> --probe wpers
  python 08_probes.py --run_dir 02_models/<run> --probe gate
  python 08_probes.py --run_dir 02_models/<onlyclin run> --probe permute \
      --permute_vars bilirubin_total creatinine platelets lactate \
      --target_features i11100_Leber_present i11100_Niere_present
"""

import argparse
import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import gtsq_eval as ge
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_train_module():
    """Import 03_train.py (invalid identifier → importlib) for eval_gtsq/decode_output."""
    spec = importlib.util.spec_from_file_location("kg_train", os.path.join(HERE, "03_train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reconstruct(run_dir):
    """Rebuild (model, configs, norm, split_dir) from a run dir — no training."""
    from models import build_model
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    split_dir = cfg["data_dir"] if os.path.isabs(cfg["data_dir"]) else os.path.join(HERE, cfg["data_dir"])
    norm = np.load(os.path.join(split_dir, "norm_stats.npz"), allow_pickle=True)

    configs = SimpleNamespace(**cfg)
    configs.gtsq_means = norm["gtsq_means"].astype(np.float32)
    configs.gtsq_stds = norm["gtsq_stds"].astype(np.float32)
    configs.gtsq_feature_info = list(zip(norm["gtsq_ftypes"], norm["gtsq_nclasses"]))
    for k, v in (("factor", 1), ("embed", "timeF"), ("freq", "h"),
                 ("activation", "gelu"), ("d_layers", 2), ("copy_scale", 8.0)):
        if not hasattr(configs, k):
            setattr(configs, k, v)

    model = build_model(configs)
    state = torch.load(os.path.join(run_dir, "model.pytorch"), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, configs, norm, split_dir


# ---------------------------------------------------------------------------
# Probe 1: W_pers diagonal-dominance (no forward pass)
# ---------------------------------------------------------------------------

def probe_wpers(model, norm):
    from models.gated_predictor import build_feature_slices
    if not hasattr(model, "skip_proj"):
        raise SystemExit("This run has no linear persistence path (skip_proj). "
                         "Use a gate_linear / skip_linear run.")
    W = model.skip_proj.weight.detach().cpu().numpy()      # (total_out, G)
    feats = [str(x) for x in norm["gtsq_features"]]
    info = list(zip(norm["gtsq_ftypes"], norm["gtsq_nclasses"]))
    slices, _ = build_feature_slices(info)
    G = len(slices)
    # For each output feature g, aggregate |W| over its output rows; compare the column
    # of its OWN previous-day input (g) against all other input columns.
    self_frac = np.zeros(G)
    for g, slc in enumerate(slices):
        block = np.abs(W[slc, :])                # (rows_g, G)  rows for feature g's logits
        col_mass = block.sum(axis=0)             # (G,)  mass from each input feature
        total = col_mass.sum() + 1e-12
        self_frac[g] = col_mass[g] / total
    diag = float(self_frac.mean())
    print(f"\nW_pers diagonal-dominance (linear persistence path, shape {W.shape}):")
    print(f"  mean self-feature weight fraction = {diag:.3f}  "
          f"(1.0 = perfectly diagonal / clean per-feature persistence)")
    print(f"  median = {np.median(self_frac):.3f}   "
          f"features with <50% own mass: {(self_frac < 0.5).sum()}/{G}")
    order = np.argsort(self_frac)
    print("  least diagonal (most feature-mixing) features:")
    for g in order[:8]:
        print(f"    {self_frac[g]:.3f}  {feats[g]}")
    return {"mean_self_fraction": diag, "median_self_fraction": float(np.median(self_frac)),
            "n_below_half": int((self_frac < 0.5).sum()), "G": G,
            "per_feature": {feats[g]: float(self_frac[g]) for g in range(G)}}


# ---------------------------------------------------------------------------
# Probe 2: gate calibration (forward pass, capture gamma)
# ---------------------------------------------------------------------------

def _forward_gate(model, split, use_clinical, bs=256):
    gi_all = split["gtsq_in"]
    N = len(gi_all)
    gates = []
    has_clin = "clinical" in split and split["clinical"].shape[-1] > 0
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            gi = torch.tensor(gi_all[s:e], dtype=torch.float32)
            ci = (torch.tensor(split["clinical"][s:e], dtype=torch.float32)
                  if use_clinical and has_clin else None)
            _, g = model(clinical_input=ci, clinical_mark=None, gtsq_input=gi)
            gates.append(g.cpu().numpy() if g is not None else None)
    if gates and gates[0] is None:
        raise SystemExit("This run is not a gate model (no gamma returned).")
    return np.concatenate(gates, axis=0)              # (N, G)


def probe_gate(model, configs, norm, split):
    gates = _forward_gate(model, split, getattr(configs, "use_clinical", False))
    feats = [str(x) for x in norm["gtsq_features"]]
    ft = [str(x) for x in norm["gtsq_ftypes"]]; nc = [int(x) for x in norm["gtsq_nclasses"]]
    gm = norm["gtsq_means"].astype(float); gs = norm["gtsq_stds"].astype(float)
    gin = split["gtsq_in"]; gin = gin[:, -1, :] if gin.ndim == 3 else gin
    gout = split["gtsq_out"]; gmask = split["gtsq_out_mask"]
    in_raw = gin * gs + gm; out_raw = gout * gs + gm
    G = gates.shape[1]
    mean_gamma = np.zeros(G); change_rate = np.zeros(G)
    g_changed = np.full(G, np.nan); g_unchanged = np.full(G, np.nan)
    for g in range(G):
        m = gmask[:, g] == 1
        if m.sum() < 10:
            continue
        hi = 1 if ft[g] == "binary" else (nc[g] - 1 if ft[g] == "ordinal" else None)
        if hi is None:
            continue
        i_cls = np.clip(np.round(in_raw[m, g]), 0, hi)
        t_cls = np.clip(np.round(out_raw[m, g]), 0, hi)
        ch = i_cls != t_cls
        mean_gamma[g] = gates[m, g].mean()
        change_rate[g] = ch.mean()
        if ch.any():    g_changed[g] = gates[m, g][ch].mean()
        if (~ch).any(): g_unchanged[g] = gates[m, g][~ch].mean()
    valid = change_rate > 0
    from scipy.stats import spearmanr, pearsonr
    rho, p = spearmanr(mean_gamma[valid], change_rate[valid])
    r, _ = pearsonr(mean_gamma[valid], change_rate[valid])
    dchg = np.nanmean(g_changed - g_unchanged)
    print(f"\nGate calibration ({valid.sum()} features with observed changes):")
    print(f"  mean gamma overall = {mean_gamma[valid].mean():.3f}  "
          f"(0 = always keep persistence, 1 = always trust the model)")
    print(f"  corr(mean gamma, change rate): Spearman rho={rho:+.3f} (p={p:.1e})  Pearson r={r:+.3f}")
    print(f"  mean gamma[changed] - gamma[unchanged] = {dchg:+.4f}  "
          f"(>0 → gate opens more when the target actually changes)")
    order = np.argsort(-change_rate)
    print("  most volatile features (highest change rate):  change  gamma")
    for g in order[:8]:
        if valid[g]:
            print(f"    {change_rate[g]*100:5.1f}%   {mean_gamma[g]:.3f}   {feats[g]}")
    return {"mean_gamma": float(mean_gamma[valid].mean()),
            "spearman_gamma_change": float(rho), "pearson_gamma_change": float(r),
            "gamma_changed_minus_unchanged": float(dchg),
            "per_feature": {feats[g]: {"gamma": float(mean_gamma[g]),
                                       "change_rate": float(change_rate[g])}
                            for g in range(G) if valid[g]}}


# ---------------------------------------------------------------------------
# Probe 3: cross-modal permutation importance (forward pass, permuted clinical)
# ---------------------------------------------------------------------------

def _clinical_groups(varnames):
    """Organ-system grouping of the clinical stream, by name. Anything unmatched lands in `other`
    rather than being dropped, so the groups always partition the full set -- a silent drop here
    would attribute a variable's contribution to nothing at all.
    """
    import re
    # The first version left 34 of 93 in `other`, and `other` then came out as the LARGEST
    # contributor -- an unlabelled bucket cannot answer "which physiology is the changer?". Opening
    # it showed the regexes missed short names (hr, rr, sao2) and two whole systems: sedation/neuro
    # (GCS subscales plus propofol/midazolam/fentanyl/morphine/dexmedetomidine) and haematology.
    # Sedation matters clinically here -- depth of sedation drives what the bedside can assess, and
    # GCS is the CNS component -- so burying it in `other` hid the most interesting candidate.
    pat = [("renal",         r"creatinin|urea|urine|rrt|dialys|^bun$|furosemid"),
           ("hepatic",       r"bilirubin|^ast$|^alt$|ggt|albumin|inr_liver|^alp$|^ldh$|lipase|protein_total"),
           ("coagulation",   r"platelet|^inr$|ptt|fibrinogen|quick|heparin"),
           ("haemodynamic",  r"mbp|sbp|dbp|heart_rate|^hr$|norepi|epinephrine|dobutamine|dopamine|vaso"
                             r"|lactate|cardiac_output|stroke_volume|^svo2$|troponin|amiodarone"),
           ("respiratory",   r"pao2|fio2|spo2|^sao2$|resp_rate|^rr$|peep|tidal|vent|paco2|etco2|^amv$"),
           ("sedation_neuro", r"^gcs|propofol|midazolam|fentanyl|morphin|dexmedetomidin|sedat"),
           ("haematology",   r"hemoglobin|hematocrit|^mch$|^mchc$|^mcv$"),
           ("inflammation",  r"crp|pct|leuko|wbc|temp|il6"),
           ("antibiotics",   r"^abx_"),
           ("cultures",      r"culture"),
           ("metabolic",     r"glucose|insulin|ph_|bicarb|hco3|base_excess|sodium|potassium|chloride"
                             r"|calcium|magnesium|phosphate")]
    out, seen = {}, set()
    for name, rx in pat:
        hit = [v for v in varnames if re.search(rx, v, re.I) and v not in seen]
        if hit:
            out[name] = hit; seen.update(hit)
    rest = [v for v in varnames if v not in seen]
    if rest:
        out["other"] = rest
        # Loud, because a large `other` invalidates the ranking: an unlabelled bucket that outranks
        # the named groups means the grouping, not the model, is what is being measured.
        print(f"  [groups] WARNING: {len(rest)} of {len(varnames)} variables unmatched -> `other`: "
              f"{', '.join(sorted(rest)[:12])}{' ...' if len(rest) > 12 else ''}")
    return out


def probe_permute(model, configs, norm, split, permute_vars, target_features, seed=0, joint=False,
                  headline_only=False, groups=None):
    train = _load_train_module()
    from models.gated_predictor import build_feature_slices
    info = list(zip(norm["gtsq_ftypes"], norm["gtsq_nclasses"]))
    slices, _ = build_feature_slices(info)
    feats = [str(x) for x in norm["gtsq_features"]]
    clin_vars = [str(x) for x in norm["clinical_variables"]]
    V = len(clin_vars)
    eval_kw = dict(feature_slices=slices, gtsq_ftypes=norm["gtsq_ftypes"],
                   gtsq_nclasses=norm["gtsq_nclasses"], gtsq_means=norm["gtsq_means"].astype(np.float32),
                   gtsq_stds=norm["gtsq_stds"].astype(np.float32), gtsq_features=norm["gtsq_features"],
                   device="cpu", use_clinical=getattr(configs, "use_clinical", True))

    def per_feat_f1(spl):
        res = train.eval_gtsq(model, spl, 256, **eval_kw)
        per = res[2]                                 # per_feat dict {fname: {...}}
        return {k: v.get("f1", v.get("f1_macro")) for k, v in per.items()}

    def headline(spl):
        """(macro-F1, change-F1) -- the pair the paper reports.

        Attribution has to be read on the CHANGE metric, not macro-F1. Physiology's whole measured
        contribution is +0.0167 change-accuracy against +0.0039 macro-F1 (approximate randomisation,
        2026-08-10), so a per-variable split of the macro-F1 budget is mostly reading noise.
        """
        res = train.eval_gtsq(model, spl, 256, **eval_kw)
        return float(res[0]), float((res[4] or {}).get("f1_change", float("nan")))

    base = per_feat_f1(split)
    base_head = headline(split)
    rng = np.random.default_rng(seed)
    targets = target_features or [f for f in feats if "Leber" in f or "Niere" in f or "Gerinnung" in f]
    rows = []
    print(f"\nCross-modal permutation importance (ΔF1 on permuting one clinical var):")
    print(f"  baseline model F1 on targets recorded; negative ΔF1 = the var was used.\n")
    print(f"  {'permuted':26s} {'target feature':34s} {'baseF1':>7s} {'permF1':>7s} {'ΔF1':>8s}")

    def run_perm(varlist, label, shared):
        """Permute varlist's columns across patients (one shared index if `shared`, else an
        independent one per variable) and report ΔF1 on each target. shared=True = composite."""
        vis = [clin_vars.index(v) for v in varlist if v in clin_vars]
        spl = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in split.items()}
        c = spl["clinical"]                          # (N, seq_len, 2V): [:V]=value, [V:]=mask
        shared_perm = rng.permutation(len(c)) if shared else None
        for vi in vis:
            pv = shared_perm if shared else rng.permutation(len(c))
            c[:, :, vi] = c[pv, :, vi]               # value channel
            c[:, :, V + vi] = c[pv, :, V + vi]       # paired mask channel
        spl["clinical"] = c
        if headline_only:
            hm, hc = headline(spl)
            rows.append({"var": label, "target": "HEADLINE", "baseF1": base_head[0],
                         "permF1": hm, "dF1": hm - base_head[0],
                         "base_f1chg": base_head[1], "perm_f1chg": hc,
                         "d_f1chg": hc - base_head[1]})
            print(f"  {label:26s} {'(all 45 items)':34s} {base_head[0]:7.4f} {hm:7.4f} "
                  f"{hm - base_head[0]:+8.4f}   chgF1 {base_head[1]:.4f} -> {hc:.4f} "
                  f"{hc - base_head[1]:+7.4f}")
            return
        pf = per_feat_f1(spl)
        for tf in targets:
            if tf in base and tf in pf:
                d = pf[tf] - base[tf]
                rows.append({"var": label, "target": tf, "baseF1": base[tf], "permF1": pf[tf], "dF1": d})
                print(f"  {label:26s} {tf:34s} {base[tf]:7.3f} {pf[tf]:7.3f} {d:+8.3f}")

    present = [v for v in permute_vars if v in clin_vars]
    missing = [v for v in permute_vars if v not in clin_vars]
    for v in missing:
        print(f"  !! {v} not in clinical_variables; skipping")
    if groups:
        # Correlated variables under-attribute individually: permute creatinine alone and urine
        # output still carries renal function, so both look unused. Permuting a whole organ system
        # with ONE shared index removes the system rather than one of its proxies.
        for gname, gvars in groups.items():
            gp = [v for v in gvars if v in clin_vars]
            if gp:
                run_perm(gp, f"{gname}[{len(gp)}]", shared=True)
    elif joint:                                      # one composite permutation of the whole set
        run_perm(present, f"COMPOSITE[{len(present)}]", shared=True)
    else:                                            # independent per-variable importance
        for var in present:
            run_perm([var], var, shared=False)
    return {"baseline_f1": {t: base.get(t) for t in targets}, "permutations": rows, "joint": joint}


# ---------------------------------------------------------------------------
# Probe 4: the gate as SELECTIVE PREDICTION — reliability + risk-coverage.
#   Reviewer ask (Paper 3): is gamma a *calibrated, useful* confidence, or just a
#   no-regret floor? Per (sample, feature) cell collect gamma, whether the value
#   changed (yesterday != tomorrow), and whether the deployed gate / persistence is
#   correct. Training-free, one forward pass.
# ---------------------------------------------------------------------------

def clinical_cols_for(configs, norm):
    """Column indices a --drop_clinical run expects, or None when it used every variable.

    A dropped-channel run was trained on a NARROWER clinical tensor than its split carries, so
    replaying the split unchanged hands the model the wrong width and the forward pass raises. The
    surviving variable NAMES are on the config, so the slice is recomputed here rather than stored --
    03_train writes config.json before it records the kept indices, so the indices themselves are not
    on disk. Without this, every culture-timing run is unreloadable and its contrast untestable.
    """
    want = getattr(configs, "drop_clinical", "") or ""
    want = [w.strip() for w in want.split(",") if w.strip()]
    if not want:
        return None
    allv = [str(x) for x in norm["clinical_variables"]]
    miss = [w for w in want if w not in allv]
    if miss:
        raise SystemExit(f"ERROR: drop_clinical names {miss}, absent from this split's variables")
    keep = [i for i, v in enumerate(allv) if v not in want]
    V = len(allv)
    return keep + [V + i for i in keep]          # value half then mask half, as 03_train slices


def _forward_pred_gate(model, split, use_clinical, bs=256, clin_cols=None,
                       configs=None, norm=None):
    """Return (logits (N, total), gates (N, G)) over the split — one forward pass.

    A --drop_clinical run was trained on a NARROWER clinical tensor than its split carries, so
    replaying the split unchanged hands the model the wrong width and the forward pass raises.
    Passing `configs` and `norm` lets this derive the slice itself; `clin_cols` still overrides.
    Six of the seven call sites did not pass a slice and crashed on such runs -- deriving it here
    fixes them together rather than leaving six copies to drift apart again.
    """
    if clin_cols is None and configs is not None and norm is not None:
        clin_cols = clinical_cols_for(configs, norm)
    gi_all = split["gtsq_in"]; N = len(gi_all)
    has_clin = "clinical" in split and split["clinical"].shape[-1] > 0
    outs, gates = [], []
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            gi = torch.tensor(gi_all[s:e], dtype=torch.float32)
            ci = None
            if use_clinical and has_clin:               # keep the access INSIDE the guard
                cl = split["clinical"][s:e]
                if clin_cols is not None:
                    cl = cl[:, :, clin_cols]
                ci = torch.tensor(cl, dtype=torch.float32)
            out, g = model(clinical_input=ci, clinical_mark=None, gtsq_input=gi)
            if g is None:
                raise SystemExit("This run is not a gate model (no gamma returned).")
            outs.append(out.cpu().numpy()); gates.append(g.cpu().numpy())
    return np.concatenate(outs, 0), np.concatenate(gates, 0)


def _slice_bounds(sl):
    return (sl.start, sl.stop) if isinstance(sl, slice) else (int(sl[0]), int(sl[1]))


def probe_selective(model, configs, norm, split, n_bins=10, figdir=None):
    from models.gated_predictor import build_feature_slices
    logits, gates = _forward_pred_gate(model, split, getattr(configs, "use_clinical", False),
                                       configs=configs, norm=norm)
    ft = [str(x) for x in norm["gtsq_ftypes"]]; nc = [int(x) for x in norm["gtsq_nclasses"]]
    gm = norm["gtsq_means"].astype(float); gs = norm["gtsq_stds"].astype(float)
    slices, _ = build_feature_slices(list(zip(norm["gtsq_ftypes"], norm["gtsq_nclasses"])))
    gin = split["gtsq_in"]; gin = gin[:, -1, :] if gin.ndim == 3 else gin
    in_raw = gin * gs + gm; out_raw = split["gtsq_out"] * gs + gm
    gmask = split["gtsq_out_mask"]; G = gates.shape[1]
    gam, chg, mcorr, pcorr = [], [], [], []
    for g in range(G):
        if ft[g] not in ("binary", "ordinal"):
            continue
        hi = 1 if ft[g] == "binary" else nc[g] - 1
        m = gmask[:, g] == 1
        if m.sum() < 10:
            continue
        a, b = _slice_bounds(slices[g])
        i_cls = np.clip(np.round(in_raw[m, g]), 0, hi).astype(int)
        t_cls = np.clip(np.round(out_raw[m, g]), 0, hi).astype(int)
        lg = logits[m, a:b]
        p_cls = ((1.0 / (1.0 + np.exp(-lg[:, 0]))) > 0.5).astype(int) if ft[g] == "binary" else lg.argmax(1)
        gam.append(gates[m, g]); chg.append((i_cls != t_cls).astype(int))
        mcorr.append((p_cls == t_cls).astype(int)); pcorr.append((i_cls == t_cls).astype(int))
    gam = np.concatenate(gam); chg = np.concatenate(chg)
    mcorr = np.concatenate(mcorr); pcorr = np.concatenate(pcorr); N = len(gam)

    from sklearn.metrics import roc_auc_score
    auroc = float(roc_auc_score(chg, gam)) if chg.any() and not chg.all() else float("nan")
    qs = np.quantile(gam, np.linspace(0, 1, n_bins + 1)); qs[-1] += 1e-9
    rel = []
    print(f"\nSelective-prediction probe ({N} feature-cells; change rate {chg.mean():.3f}, "
          f"persistence acc {pcorr.mean():.3f}, deployed-gate acc {mcorr.mean():.3f}):")
    print(f"  (a) Reliability — gamma decile -> empirical P(change):")
    print(f"      {'mean_gamma':>10s} {'P(change)':>10s} {'n':>8s}")
    for k in range(n_bins):
        sel = (gam >= qs[k]) & (gam < qs[k + 1])
        if sel.sum() == 0:
            continue
        rel.append({"mean_gamma": float(gam[sel].mean()), "p_change": float(chg[sel].mean()), "n": int(sel.sum())})
        print(f"      {gam[sel].mean():10.3f} {chg[sel].mean():10.3f} {sel.sum():8d}")
    print(f"  AUROC(gamma -> changed) = {auroc:.3f}  (0.5 = gamma uninformative about change)")

    print(f"  (b) Risk-coverage — override the top-c%-by-gamma cells (deployed gate) vs keep persistence:")
    print(f"      {'coverage':>8s} {'gamma_thr':>9s} {'gate_acc':>8s} {'persist_acc':>11s} {'gain':>7s} {'net_acc':>7s}")
    order = np.argsort(-gam); rc = []
    for cov in [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]:
        k = max(1, int(round(cov * N))); idx = order[:k]
        gate_acc = float(mcorr[idx].mean()); persist_acc = float(pcorr[idx].mean())
        net = float((mcorr[idx].sum() + pcorr[order[k:]].sum()) / N)
        rc.append({"coverage": cov, "gamma_thr": float(gam[order[k - 1]]), "gate_acc": gate_acc,
                   "persist_acc": persist_acc, "gain": gate_acc - persist_acc, "net_acc": net})
        print(f"      {cov:8.2f} {gam[order[k-1]]:9.3f} {gate_acc:8.3f} {persist_acc:11.3f} "
              f"{gate_acc-persist_acc:+7.3f} {net:7.3f}")
    print(f"  (pure persistence {pcorr.mean():.3f}; cov=1.0 net_acc = pure deployed gate {mcorr.mean():.3f})")

    if figdir:
        _render_selective_fig(rel, rc, auroc, float(chg.mean()), figdir)
    return {"n_cells": N, "change_rate": float(chg.mean()), "persist_acc": float(pcorr.mean()),
            "gate_acc": float(mcorr.mean()), "auroc_gamma_change": auroc,
            "reliability": rel, "risk_coverage": rc}


def _render_selective_fig(rel, rc, auroc, base_rate, figdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot([r["mean_gamma"] for r in rel], [r["p_change"] for r in rel], "o-")
    ax[0].axhline(base_rate, ls=":", c="grey", label=f"base rate {base_rate:.2f}")
    ax[0].set_xlabel(r"gate $\gamma$ (decile mean)"); ax[0].set_ylabel("empirical P(change)")
    ax[0].set_title(rf"Reliability (AUROC $\gamma\!\to\!$change = {auroc:.2f})"); ax[0].legend(fontsize=8)
    ax[1].plot([r["coverage"] * 100 for r in rc], [r["gain"] for r in rc], "s-")
    ax[1].axhline(0, ls=":", c="grey")
    ax[1].set_xlabel(r"coverage = % cells overridden (by $\gamma$)")
    ax[1].set_ylabel("gate acc − persistence acc"); ax[1].set_title("Risk–coverage (override gain)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"gate_selective.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {figdir}/gate_selective.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Probe 5: CHANGE-STRATIFIED metrics (review-driven, Paper 3 revision). Training-free,
#   one forward pass. On the CHANGED subset (yesterday != tomorrow) it reports
#   precision / recall / F1 (model vs persistence, which scores ~0 there by construction),
#   the DIRECTION of change (escalation vs de-escalation, and whether the model's overrides
#   move the right way), and the distance-sensitive QWK on ordinal targets. For gate runs it
#   adds a change-alert precision/recall curve over the gate gamma. Reads no result file and
#   writes only where --out points, so it cannot overwrite committed numbers.
# ---------------------------------------------------------------------------

def _forward_pred(model, split, use_clinical, bs=256):
    """logits (N, total) and gates (N, G) or None — works for gate and non-gate runs."""
    gi_all = split["gtsq_in"]; N = len(gi_all)
    has_clin = "clinical" in split and split["clinical"].shape[-1] > 0
    outs, gates = [], []
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            gi = torch.tensor(gi_all[s:e], dtype=torch.float32)
            ci = (torch.tensor(split["clinical"][s:e], dtype=torch.float32)
                  if use_clinical and has_clin else None)
            out, g = model(clinical_input=ci, clinical_mark=None, gtsq_input=gi)
            outs.append(out.cpu().numpy())
            gates.append(None if g is None else g.cpu().numpy())
    g_arr = None if gates[0] is None else np.concatenate(gates, 0)
    return np.concatenate(outs, 0), g_arr


def probe_change(model, configs, norm, split, min_changed=10, verbose=True):
    from models.gated_predictor import build_feature_slices
    from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
    logits, gates = _forward_pred(model, split, getattr(configs, "use_clinical", False))
    feats = [str(x) for x in norm["gtsq_features"]]
    ft = [str(x) for x in norm["gtsq_ftypes"]]; nc = [int(x) for x in norm["gtsq_nclasses"]]
    gm = norm["gtsq_means"].astype(float); gs = norm["gtsq_stds"].astype(float)
    slices, _ = build_feature_slices(list(zip(norm["gtsq_ftypes"], norm["gtsq_nclasses"])))
    gin = split["gtsq_in"]; gin = gin[:, -1, :] if gin.ndim == 3 else gin
    in_raw = gin * gs + gm; out_raw = split["gtsq_out"] * gs + gm
    gmask = split["gtsq_out_mask"]; G = len(slices)

    def _qwk(a, b, k):
        try:
            return float(cohen_kappa_score(a, b, labels=list(range(k)), weights="quadratic"))
        except Exception:
            return float("nan")

    per = {}
    pdt, pdp = [], []           # pooled true / predicted change-direction (over all changed cells)
    gam_all, chg_all = [], []   # gate gamma vs change (for the change-alert curve)
    for g in range(G):
        if ft[g] not in ("binary", "ordinal"):
            continue
        hi = 1 if ft[g] == "binary" else nc[g] - 1
        m = gmask[:, g] == 1
        if m.sum() < min_changed:
            continue
        a, b = _slice_bounds(slices[g])
        i_cls = np.clip(np.round(in_raw[m, g]), 0, hi).astype(int)
        t_cls = np.clip(np.round(out_raw[m, g]), 0, hi).astype(int)
        lg = logits[m, a:b]
        # sigmoid(x) > 0.5  <=>  x > 0  (avoids the exp overflow on large-magnitude logits)
        p_cls = ge.decode_feature(lg, ft[g])
        ch = i_cls != t_cls
        labs = list(range(nc[g])) if ft[g] == "ordinal" else [0, 1]
        rec = {"type": ft[g], "n_total": int(m.sum()), "n_changed": int(ch.sum()),
               "change_rate": float(ch.mean())}
        if ft[g] == "ordinal":
            rec["qwk_model"] = _qwk(t_cls, p_cls, nc[g])
            rec["qwk_persist"] = _qwk(t_cls, i_cls, nc[g])
        if ch.sum() >= 1:
            tc, pm, ic = t_cls[ch], p_cls[ch], i_cls[ch]
            rec["chg_precision"] = float(precision_score(tc, pm, labels=labs, average="macro", zero_division=0))
            rec["chg_recall"] = float(recall_score(tc, pm, labels=labs, average="macro", zero_division=0))
            rec["chg_f1"] = float(f1_score(tc, pm, labels=labs, average="macro", zero_division=0))
            rec["chg_exact_acc"] = float((pm == tc).mean())
            # persistence on the changed subset predicts ic (never the new class) -> ~0 by construction:
            rec["chg_f1_persist"] = float(f1_score(tc, ic, labels=labs, average="macro", zero_division=0))
            dt = np.sign(tc - ic); dp = np.sign(pm - ic)
            ov = pm != ic
            rec["escalation_frac"] = float((dt > 0).mean())        # of the TRUE changes
            rec["override_rate"] = float(ov.mean())                # model deviates from persistence
            rec["dir_acc"] = float((dp[ov] == dt[ov]).mean()) if ov.any() else float("nan")
            rec["escal_recall"] = float((pm[dt > 0] == tc[dt > 0]).mean()) if (dt > 0).any() else float("nan")
            rec["deescal_recall"] = float((pm[dt < 0] == tc[dt < 0]).mean()) if (dt < 0).any() else float("nan")
            pdt.append(dt); pdp.append(dp)
        if gates is not None:
            gam_all.append(gates[m, g]); chg_all.append(ch.astype(int))
        per[feats[g]] = rec

    def _mean(key):                                  # macro over features; ignore NaN (x==x filters NaN)
        vals = [per[k][key] for k in per if key in per[k] and per[k][key] == per[k][key]]
        return float(np.mean(vals)) if vals else float("nan")
    agg = {"n_features": sum(1 for k in per if "chg_f1" in per[k]),
           "mean_change_rate": _mean("change_rate"),
           "chg_precision": _mean("chg_precision"), "chg_recall": _mean("chg_recall"),
           "chg_f1": _mean("chg_f1"), "chg_exact_acc": _mean("chg_exact_acc"),
           "chg_f1_persist": _mean("chg_f1_persist"),
           "qwk_model": _mean("qwk_model"), "qwk_persist": _mean("qwk_persist"),
           "override_rate": _mean("override_rate"), "dir_acc": _mean("dir_acc"),
           "escalation_frac": _mean("escalation_frac"),
           "escal_recall": _mean("escal_recall"), "deescal_recall": _mean("deescal_recall")}
    if pdt:
        DT = np.concatenate(pdt); DP = np.concatenate(pdp); ov = DP != 0
        agg["pooled_escalation_frac"] = float((DT > 0).mean())
        agg["pooled_dir_acc"] = float((DP[ov] == DT[ov]).mean()) if ov.any() else float("nan")

    alert = None
    if gates is not None and gam_all:
        from sklearn.metrics import roc_auc_score
        gam = np.concatenate(gam_all); chgv = np.concatenate(chg_all)
        auroc = float(roc_auc_score(chgv, gam)) if chgv.any() and not chgv.all() else float("nan")
        curve = []
        for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            al = gam >= thr
            if al.sum() == 0:
                continue
            tp = int((al & (chgv == 1)).sum())
            curve.append({"gamma_thr": thr, "alert_rate": float(al.mean()),
                          "precision": float(tp / al.sum()),
                          "recall": float(tp / max(int((chgv == 1).sum()), 1))})
        # Calibration of gamma as a change probability (reviewer ask): Brier + 10-bin ECE.
        brier = float(np.mean((gam - chgv) ** 2))
        edges = np.linspace(0.0, 1.0, 11); ece = 0.0
        for k in range(10):
            sel = (gam >= edges[k]) & ((gam <= edges[k + 1]) if k == 9 else (gam < edges[k + 1]))
            if sel.any():
                ece += float(sel.mean()) * abs(float(gam[sel].mean()) - float(chgv[sel].mean()))
        agg["alert_auroc"] = auroc; agg["gamma_brier"] = brier; agg["gamma_ece"] = float(ece)
        alert = {"auroc_gamma_change": auroc, "brier": brier, "ece": float(ece), "curve": curve}

    if verbose:
        print(f"\nChange-stratified metrics ({agg['n_features']} features, "
              f"mean change rate {agg['mean_change_rate']:.3f}):")
        print(f"  on CHANGED cells (macro over features):")
        print(f"    precision={agg['chg_precision']:.3f}  recall={agg['chg_recall']:.3f}  "
              f"F1={agg['chg_f1']:.3f}  exact-acc={agg['chg_exact_acc']:.3f}  "
              f"(persistence F1 here = {agg['chg_f1_persist']:.3f})")
        print(f"  direction: {agg['escalation_frac']*100:.0f}% of changes are escalations; "
              f"override rate={agg['override_rate']:.3f}, directional acc of overrides={agg['dir_acc']:.3f}")
        print(f"    escalation recall={agg['escal_recall']:.3f}  de-escalation recall={agg['deescal_recall']:.3f}")
        print(f"  ordinal QWK (distance-sensitive): model={agg['qwk_model']:.3f}  persistence={agg['qwk_persist']:.3f}")
        if alert:
            print(f"  change-alert on gate gamma: AUROC={alert['auroc_gamma_change']:.3f}  "
                  f"ECE={alert['ece']:.4f}  Brier={alert['brier']:.4f}")
            for c in alert["curve"]:
                print(f"    gamma>={c['gamma_thr']:.1f}: alert {c['alert_rate']*100:5.1f}%  "
                      f"precision {c['precision']:.3f}  recall {c['recall']:.3f}")
    return {"aggregate": agg, "per_feature": per, "change_alert": alert}


def _chg_target_of(dd):
    dd = str(dd)
    for t in ("support", "sofa", "aki"):
        if f"mimic_{t}" in dd:
            return t
    if "eicu_dx" in dd:
        return "eicu_dx"
    if "gtsq_pred" in dd:
        return "gtsq"
    return "other"


def _chg_variant_of(cfg):
    if str(cfg.get("model_type")) == "persistence":
        return "persistence"
    pp = str(cfg.get("persist_path", "linear"))
    if cfg.get("use_gate"):
        return f"gate_{pp}"
    if cfg.get("use_skip"):
        return f"skip_{pp}"
    return "base"


def _chg_is_canonical(cfg, target):
    """Mirror 07_aggregate.is_canonical for a single config: default encoder/lambda/depth/
    fusion, no input-mode/shuffle/horizon ablation, canonical loss, standard objective."""
    lb_default = "lw" if target == "gtsq" else "uniform"
    return (str(cfg.get("model_type")) == "gru"
            and cfg.get("gate_lambda") in (None, 0.5)
            and cfg.get("gtsq_enc_layers") in (None, 2)
            and cfg.get("fusion") in (None, "concat")
            and cfg.get("use_gtsq_input", True)
            and not cfg.get("shuffle_patients")
            and str(cfg.get("loss_balance")) == lb_default
            and cfg.get("ordinal_loss") in (None, "ce")
            and cfg.get("change_loss") in (None,))


def run_change_glob(pattern, out_csv, min_changed=10, all_variants=False):
    """Run probe_change over every matching run dir (read-only), write one CSV row per run to
    out_csv, and print seed-averaged means per (target, variant). Skips horizon dirs and the
    degenerate persistence run; keeps only canonical runs unless all_variants is set (Tier B/D
    ordinal/change-loss runs are non-canonical and need --all_variants to be included)."""
    import glob
    import re as _re
    import pandas as pd
    pat = pattern if os.path.isabs(pattern) else os.path.join(HERE, pattern)
    rows = []
    for rd in sorted(glob.glob(pat)):
        cfgp = os.path.join(rd, "config.json")
        if not (os.path.exists(cfgp) and os.path.exists(os.path.join(rd, "model.pytorch"))):
            continue
        cfg = json.load(open(cfgp))
        tgt = _chg_target_of(cfg.get("data_dir", rd))
        if _re.search(r"_hz\d", str(cfg.get("data_dir", ""))) or _chg_variant_of(cfg) == "persistence":
            continue
        if not all_variants and not _chg_is_canonical(cfg, tgt):
            continue
        try:
            model, configs, norm, split_dir = reconstruct(rd)
            tnpz = np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)
            split = {k: tnpz[k] for k in tnpz.files}
            res = probe_change(model, configs, norm, split, min_changed=min_changed, verbose=False)
        except SystemExit as ex:
            print(f"  skip {os.path.basename(rd)}: {ex}")
            continue
        rows.append({"run": os.path.basename(rd), "target": tgt,
                     "variant": _chg_variant_of(cfg), "seed": cfg.get("seed"),
                     **res["aggregate"]})
    if not rows:
        print(f"no runs matched {pat}")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, float_format="%.4f")
    keep = ["chg_precision", "chg_recall", "chg_f1", "chg_exact_acc", "chg_f1_persist",
            "qwk_model", "qwk_persist", "override_rate", "dir_acc", "escalation_frac",
            "escal_recall", "deescal_recall"]
    keep = [k for k in keep if k in df.columns]
    print((df.groupby(["target", "variant"])[keep].mean(numeric_only=True)).round(3).to_string())
    print(f"\nsaved per-run -> {out_csv}  ({len(df)} runs)")


# ---------------------------------------------------------------------------
# Probe 6: directed COUNTERFACTUAL input perturbation (review-driven). Overwrite one
#   clinical variable with a constant injected physical value (z-scored, mask set observed)
#   and read off how the model's prediction of a linked target moves. A dose-response that
#   steps with the clinical thresholds => the model learned the mechanistic mapping; a flat
#   curve => it ignores the input (defers to persistence). Run on a gate vs a base model to
#   quantify the gate's robustness to corrupted inputs. Training-free, one forward / value.
# ---------------------------------------------------------------------------

def probe_perturb(model, configs, norm, split, perturb_vars, target_features, values):
    from models.gated_predictor import build_feature_slices
    feats = [str(x) for x in norm["gtsq_features"]]
    ft = [str(x) for x in norm["gtsq_ftypes"]]; nc = [int(x) for x in norm["gtsq_nclasses"]]
    slices, _ = build_feature_slices(list(zip(norm["gtsq_ftypes"], norm["gtsq_nclasses"])))
    clin_vars = [str(x) for x in norm["clinical_variables"]]; V = len(clin_vars)
    cmean = norm["clinical_means"].astype(float); cstd = norm["clinical_stds"].astype(float)
    if "clinical" not in split or split["clinical"].shape[-1] == 0:
        raise SystemExit("perturb probe needs a clinical run (use_clinical / oracle split).")
    if not target_features:
        raise SystemExit("perturb probe needs --target_features (e.g. sofa_liver).")
    tgt_idx = {f: g for g, f in enumerate(feats)}

    def pred_mean(spl):
        logits, _ = _forward_pred(model, spl, True)
        out = {}
        for tf in target_features:
            g = tgt_idx[tf]; a, b = _slice_bounds(slices[g]); lg = logits[:, a:b]
            pc = ge.decode_feature(lg, ft[g])
            out[tf] = float(pc.mean())
        return out

    base = pred_mean(split)
    print("\nCounterfactual input perturbation (mean predicted class vs injected value):")
    print("  baseline: " + ", ".join(f"{tf}={base[tf]:.2f}" for tf in target_features))
    rows = []
    for pv in perturb_vars:
        if pv not in clin_vars:
            print(f"  !! {pv} not a clinical variable; skip"); continue
        vi = clin_vars.index(pv)
        if not (np.isfinite(cmean[vi]) and np.isfinite(cstd[vi])):
            print(f"  !! {pv}: NaN normalisation (clinical-feature bug not fixed for this split)"); continue
        print(f"  --- perturb {pv} ---")
        for v in values:
            spl = {k: (x.copy() if isinstance(x, np.ndarray) else x) for k, x in split.items()}
            z = float(np.clip((v - cmean[vi]) / cstd[vi], -10, 10))   # match CLINICAL_CLIP at train
            spl["clinical"][:, :, vi] = z                              # inject constant value, all hours
            spl["clinical"][:, :, V + vi] = 1.0                        # mark observed
            pm = pred_mean(spl)
            for tf in target_features:
                rows.append({"var": pv, "value": float(v), "target": tf, "mean_pred": pm[tf]})
            print(f"    {pv}={v:>8.2f}: " + ", ".join(f"{tf}={pm[tf]:.2f}" for tf in target_features))
    return {"baseline": base, "sweep": rows}


# ---------------------------------------------------------------------------
# Probe 7: ACTIVE-DAYS evaluation (group ask). Macro-F1 on ALL day-pairs vs the
#   ACTIVE subset (>=1 observed feature changes vs yesterday), for the model and for
#   persistence. Tests whether dropping the ~39% fully-unchanged "copy-all" day-pairs
#   lets a learner beat persistence on the days that actually matter.
# ---------------------------------------------------------------------------

def probe_active(model, configs, norm, split, verbose=True):
    from models.gated_predictor import build_feature_slices
    from sklearn.metrics import f1_score
    logits, _ = _forward_pred(model, split, getattr(configs, "use_clinical", False))
    ft = [str(x) for x in norm["gtsq_ftypes"]]; nc = [int(x) for x in norm["gtsq_nclasses"]]
    gm = norm["gtsq_means"].astype(float); gs = norm["gtsq_stds"].astype(float)
    slices, _ = build_feature_slices(list(zip(norm["gtsq_ftypes"], norm["gtsq_nclasses"])))
    gin = split["gtsq_in"]; gin = gin[:, -1, :] if gin.ndim == 3 else gin
    im = split["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
    in_raw = gin * gs + gm; out_raw = split["gtsq_out"] * gs + gm; om = split["gtsq_out_mask"]
    G = len(slices); N = len(gin)
    P = np.zeros((N, G), int); I = np.zeros((N, G), int); T = np.zeros((N, G), int)
    valid = np.zeros((N, G), bool)
    for g in range(G):
        if ft[g] not in ("binary", "ordinal"):
            continue
        hi = 1 if ft[g] == "binary" else nc[g] - 1
        a, b = _slice_bounds(slices[g]); lg = logits[:, a:b]
        P[:, g] = ge.decode_feature(lg, ft[g])
        I[:, g] = np.clip(np.round(in_raw[:, g]), 0, hi)
        T[:, g] = np.clip(np.round(out_raw[:, g]), 0, hi)
        valid[:, g] = om[:, g] == 1
    active_row = ((I != T) & valid & (im == 1)).any(axis=1)   # day-pair with >=1 observed change

    # labels=range(nclass) so an absent class scores F1=0 (not silently skipped) — matches
    # 05_baselines / 03_train, unifying the persistence macro-F1 to 0.871 (was 0.876 here when
    # the class range was left to sklearn's default of only-observed labels).
    _labs = [list(range(2 if ft[g] == "binary" else nc[g])) for g in range(G)]

    def macroF1(pred, rowmask):
        fs = [f1_score(T[m, g], pred[m, g], average="macro", labels=_labs[g], zero_division=0)
              for g in range(G) for m in [valid[:, g] & rowmask] if m.sum() >= 10]
        return float(np.mean(fs)) if fs else float("nan")

    def acc(pred, changed_only=False):                # cell-accuracy, optionally only on changed cells
        c = t = 0
        for g in range(G):
            m = (valid[:, g] & (I[:, g] != T[:, g])) if changed_only else valid[:, g]
            c += int((pred[m, g] == T[m, g]).sum()); t += int(m.sum())
        return float(c / max(t, 1))

    def f1change(pred):                               # macro-F1 on the changed cells (per feature)
        fs = [f1_score(T[m, g], pred[m, g], average="macro", labels=_labs[g], zero_division=0)
              for g in range(G) for m in [valid[:, g] & (I[:, g] != T[:, g])] if m.sum() >= 10]
        return float(np.mean(fs)) if fs else float("nan")

    def microf1(pred):                                # micro-F1 over abnormal (non-zero) classes, pooled
        tp = fp = fn = 0                              # across features; == accuracy for multiclass, != for binary
        for g in range(G):                            # (binary ignores the dominant TN 'normal' cells)
            m = valid[:, g]; yt = T[m, g]; yp = pred[m, g]
            hi = 2 if ft[g] == "binary" else nc[g]
            for c in range(1, hi):
                tp += int(((yp == c) & (yt == c)).sum())
                fp += int(((yp == c) & (yt != c)).sum())
                fn += int(((yp != c) & (yt == c)).sum())
        return float(2 * tp / max(2 * tp + fp + fn, 1))
    allrows = np.ones(N, bool)
    res = {"active_frac": float(active_row.mean()),
           "model_all": macroF1(P, allrows), "model_active": macroF1(P, active_row),
           "persist_all": macroF1(I, allrows), "persist_active": macroF1(I, active_row),
           "model_acc": acc(P), "persist_acc": acc(I),
           "model_chgacc": acc(P, True), "persist_chgacc": acc(I, True),
           "model_f1change": f1change(P), "persist_f1change": f1change(I),
           "model_microf1": microf1(P), "persist_microf1": microf1(I)}
    if verbose:
        print(f"\nActive-days eval (active={res['active_frac']*100:.0f}% of day-pairs):")
        print(f"  macro-F1 ALL:    model {res['model_all']:.3f}  persistence {res['persist_all']:.3f}")
        print(f"  macro-F1 ACTIVE: model {res['model_active']:.3f}  persistence {res['persist_active']:.3f}  "
              f"(delta {res['model_active']-res['persist_active']:+.3f})")
        print(f"  accuracy ALL:    model {res['model_acc']:.4f}  persistence {res['persist_acc']:.4f}")
        print(f"  micro-F1 ALL:    model {res['model_microf1']:.4f}  persistence {res['persist_microf1']:.4f}")
        print(f"  accuracy CHANGE: model {res['model_chgacc']:.4f}  persistence {res['persist_chgacc']:.4f} "
              f"(persistence=0 by construction)")
    return res


def probe_ensemble(run_a, run_b, taus=(0.7, 0.9, 0.95), verbose=True):
    """Combine GTSQ-only model A (great overall, copies) with clinical-only model B (catches
    changes, poor overall). Tests (i) does an ensemble/router beat either alone, and (ii) the
    'drop the non-changing questionnaires' question: the ORACLE router copies when a cell will not
    change and calls the clinical model when it will -- the ceiling of routing copy<->clinical.
    Test rows of the two splits must be aligned (same enc_ids/order)."""
    from models.gated_predictor import build_feature_slices
    from sklearn.metrics import f1_score
    mA, cA, norm, sdA = reconstruct(run_a)
    mB, cB, _, sdB = reconstruct(run_b)
    sA = {k: v for k, v in np.load(os.path.join(sdA, "test.npz"), allow_pickle=True).items()}
    sB = {k: v for k, v in np.load(os.path.join(sdB, "test.npz"), allow_pickle=True).items()}
    ft = [str(x) for x in norm["gtsq_ftypes"]]; nc = [int(x) for x in norm["gtsq_nclasses"]]
    gm = norm["gtsq_means"].astype(float); gs = norm["gtsq_stds"].astype(float)
    slices, _ = build_feature_slices(list(zip(norm["gtsq_ftypes"], norm["gtsq_nclasses"])))
    G = len(slices)

    def decode(model, split, use_clin):
        logits, _ = _forward_pred(model, split, use_clin)
        N = len(logits); P = np.zeros((N, G), int); conf = np.zeros((N, G))
        for g in range(G):
            if ft[g] not in ("binary", "ordinal"):
                continue
            a, b = _slice_bounds(slices[g]); lg = logits[:, a:b]
            if ft[g] == "binary":
                pr = 1.0 / (1.0 + np.exp(-lg[:, 0])); P[:, g] = (lg[:, 0] > 0).astype(int)
                conf[:, g] = np.where(P[:, g] == 1, pr, 1 - pr)
            else:
                e = np.exp(lg - lg.max(1, keepdims=True)); sm = e / e.sum(1, keepdims=True)
                P[:, g] = lg.argmax(1); conf[:, g] = sm.max(1)
        return P, conf
    PA, _ = decode(mA, sA, getattr(cA, "use_clinical", False))
    PB, confB = decode(mB, sB, getattr(cB, "use_clinical", False))
    gin = sA["gtsq_in"]; gin = gin[:, -1, :] if gin.ndim == 3 else gin
    in_raw = gin * gs + gm; out_raw = sA["gtsq_out"] * gs + gm; om = sA["gtsq_out_mask"]
    N = len(PA); I = np.zeros((N, G), int); T = np.zeros((N, G), int); valid = np.zeros((N, G), bool)
    for g in range(G):
        if ft[g] not in ("binary", "ordinal"):
            continue
        hi = 1 if ft[g] == "binary" else nc[g] - 1
        I[:, g] = np.clip(np.round(in_raw[:, g]), 0, hi); T[:, g] = np.clip(np.round(out_raw[:, g]), 0, hi)
        valid[:, g] = om[:, g] == 1

    def four(P):
        def acc(chg):
            c = t = 0
            for g in range(G):
                m = (valid[:, g] & (I[:, g] != T[:, g])) if chg else valid[:, g]
                c += int((P[m, g] == T[m, g]).sum()); t += int(m.sum())
            return c / max(t, 1)
        tp = fp = fn = 0
        for g in range(G):
            m = valid[:, g]; yt = T[m, g]; yp = P[m, g]; hi = 2 if ft[g] == "binary" else nc[g]
            for c in range(1, hi):
                tp += int(((yp == c) & (yt == c)).sum()); fp += int(((yp == c) & (yt != c)).sum())
                fn += int(((yp != c) & (yt == c)).sum())
        micro = 2 * tp / max(2 * tp + fp + fn, 1)
        _lb = [list(range(2 if ft[g] == "binary" else nc[g])) for g in range(G)]
        mf = float(np.mean([f1_score(T[m, g], P[m, g], average="macro", labels=_lb[g], zero_division=0)
                            for g in range(G) for m in [valid[:, g]] if m.sum() >= 10]))
        f1c = float(np.mean([f1_score(T[m, g], P[m, g], average="macro", labels=_lb[g], zero_division=0)
                             for g in range(G) for m in [valid[:, g] & (I[:, g] != T[:, g])] if m.sum() >= 10]))
        return acc(False), micro, acc(True), mf, f1c
    preds = {"copy (persistence)": I, "A: gtsq-only": PA, "B: clin-only": PB,
             "oracle-router (drop non-chg)": np.where(I != T, PB, I)}
    for tau in taus:
        preds[f"conf-router tau={tau}"] = np.where((PB != I) & (confB > tau), PB, I)
    res = {}
    if verbose:
        print(f"\nEnsemble  A={os.path.basename(run_a)[:34]}\n          B={os.path.basename(run_b)[:34]}")
        print(f"{'predictor':30}{'ovr-acc':>9}{'microF1':>9}{'chg-acc':>9}{'macroF1':>9}{'F1chg':>9}")
    for name, P in preds.items():
        a0, micro, a1, mf, f1c = four(P)
        res[name] = dict(acc=a0, microf1=micro, chgacc=a1, macroF1=mf, f1change=f1c)
        if verbose:
            print(f"{name:30}{a0:9.4f}{micro:9.4f}{a1:9.4f}{mf:9.4f}{f1c:9.4f}")
    return res


def run_active_glob(pattern, out_csv, all_variants=False):
    """probe_active over matching runs (read-only); per-run CSV + grouped means per variant."""
    import glob
    import re as _re
    import pandas as pd
    pat = pattern if os.path.isabs(pattern) else os.path.join(HERE, pattern)
    rows = []
    for rd in sorted(glob.glob(pat)):
        cfgp = os.path.join(rd, "config.json")
        if not (os.path.exists(cfgp) and os.path.exists(os.path.join(rd, "model.pytorch"))):
            continue
        cfg = json.load(open(cfgp))
        tgt = _chg_target_of(cfg.get("data_dir", rd))
        if _re.search(r"_hz\d", str(cfg.get("data_dir", ""))) or _chg_variant_of(cfg) == "persistence":
            continue
        if not all_variants and not _chg_is_canonical(cfg, tgt):
            continue
        try:
            model, configs, norm, split_dir = reconstruct(rd)
            tnpz = np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)
            split = {k: tnpz[k] for k in tnpz.files}
            res = probe_active(model, configs, norm, split, verbose=False)
        except SystemExit as ex:
            print(f"  skip {os.path.basename(rd)}: {ex}"); continue
        rows.append({"run": os.path.basename(rd), "target": tgt,
                     "variant": _chg_variant_of(cfg), "seed": cfg.get("seed"), **res})
    if not rows:
        print(f"no runs matched {pat}"); return
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False, float_format="%.4f")
    keep = ["model_acc", "persist_acc", "model_microf1", "persist_microf1",
            "model_chgacc", "persist_chgacc", "model_all", "persist_all",
            "model_f1change", "persist_f1change"]
    print(df.groupby(["target", "variant"])[keep].mean(numeric_only=True).round(3).to_string())
    print(f"\nsaved per-run -> {out_csv}  ({len(df)} runs)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default=None)
    p.add_argument("--run_b", default=None, help="(ensemble probe) second model B (clinical-only)")
    p.add_argument("--probe", choices=["wpers", "gate", "permute", "selective", "change", "perturb", "active", "ensemble"], required=True)
    p.add_argument("--perturb_values", nargs="+", type=float, default=[0.5, 1.5, 4.0, 8.0, 15.0],
                   help="(perturb probe) physical values to inject for --perturb_vars (default: "
                        "bilirubin sweep across the Vincent liver cutoffs 1.2/2.0/6.0/12.0)")
    p.add_argument("--permute_vars", nargs="+", default=["bilirubin_total", "creatinine", "platelets", "lactate"])
    p.add_argument("--permute_all", action="store_true", default=False,
                   help="permute EVERY clinical variable in the split, one at a time, scoring the "
                        "headline macro-F1 and change-F1")
    p.add_argument("--permute_groups", action="store_true", default=False,
                   help="permute organ-system GROUPS jointly instead of single variables. Single "
                        "variables under-attribute when proxies are correlated (creatinine vs urine "
                        "output); groups do not.")
    p.add_argument("--target_features", nargs="+", default=None)
    p.add_argument("--joint", action="store_true", default=False,
                   help="Permute all --permute_vars jointly (one shared index) as a single "
                        "composite, instead of one variable at a time (e.g. a sepsis panel).")
    p.add_argument("--figdir", default=None, help="(selective probe) render reliability + risk-coverage figure here")
    p.add_argument("--out", default=None, help="optional JSON output path")
    p.add_argument("--glob", default=None,
                   help="(change probe) run over ALL matching run dirs and write a per-run CSV")
    p.add_argument("--min_changed", type=int, default=10,
                   help="(change probe) skip features with fewer than this many observed target cells")
    p.add_argument("--all_variants", action="store_true", default=False,
                   help="(change probe --glob) include non-canonical runs (Tier B/D ordinal/change-loss)")
    a = p.parse_args()

    if a.probe == "change" and a.glob:
        run_change_glob(a.glob, a.out or os.path.join(HERE, "change_metrics.csv"),
                        a.min_changed, all_variants=a.all_variants)
        return
    if a.probe == "active" and a.glob:
        run_active_glob(a.glob, a.out or os.path.join(HERE, "active_eval.csv"),
                        all_variants=a.all_variants)
        return
    if a.probe == "ensemble":
        if not a.run_dir or not a.run_b:
            raise SystemExit("ensemble needs --run_dir (A: gtsq-only) and --run_b (B: clin-only)")
        res = probe_ensemble(a.run_dir, a.run_b)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=2, default=float); print(f"\nsaved -> {a.out}")
        return
    if not a.run_dir:
        raise SystemExit("--run_dir is required (unless using --probe change --glob ...)")

    model, configs, norm, split_dir = reconstruct(a.run_dir)
    split = {k: np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)[k]
             for k in np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True).files}
    print(f"run: {os.path.basename(a.run_dir)}")
    print(f"split: {os.path.basename(split_dir)}  (test N={len(split['gtsq_in'])})")

    if a.probe == "wpers":
        res = probe_wpers(model, norm)
    elif a.probe == "gate":
        res = probe_gate(model, configs, norm, split)
    elif a.probe == "selective":
        res = probe_selective(model, configs, norm, split, figdir=a.figdir)
    elif a.probe == "change":
        res = probe_change(model, configs, norm, split, min_changed=a.min_changed)
    elif a.probe == "perturb":
        res = probe_perturb(model, configs, norm, split, a.permute_vars, a.target_features, a.perturb_values)
    elif a.probe == "active":
        res = probe_active(model, configs, norm, split)
    else:
        # --permute_all replaces the four hand-picked variables with every clinical variable in the
        # split, which is what "which physiology is the changer?" actually asks. --permute_groups
        # switches to organ-system groups, the only level at which correlated proxies stop hiding
        # each other. Both read the headline metrics rather than three hand-picked target features.
        pv = a.permute_vars
        groups = None
        if a.permute_all or a.permute_groups:
            pv = [str(x) for x in norm["clinical_variables"]]
        if a.permute_groups:
            groups = _clinical_groups(pv)
            print(f"  grouping {len(pv)} variables into {len(groups)} organ systems: "
                  + ", ".join(f"{k}({len(v)})" for k, v in groups.items()))
        res = probe_permute(model, configs, norm, split, pv, a.target_features, joint=a.joint,
                            headline_only=a.permute_all or a.permute_groups, groups=groups)

    if a.out:
        json.dump(res, open(a.out, "w"), indent=2, default=float)
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()