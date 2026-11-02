#!/usr/bin/env python3
"""Cross-target circularity, derived from the splits and runs wherever possible.

Two subcommands, both reachable through run_analysis.sh:

  crosstarget  the five-target table (persistence, ML gain, circularity) and their correlation
  sofacomp     per-component SOFA breakdown for MIMIC and UMM -- which subsystems carry it

WHY THIS IS ITS OWN FILE. report.py harvests trained runs from the paper's model trees; this reads
FOUR different split families (GTSQ, UMM SOFA, MIMIC, eICU) plus the gradient-boosting baselines,
and it is the only analysis that mixes them. Folding it into report.py would mean that file takes a
split family as an argument in some subcommands and not others.

TWO CONVENTIONS THAT MUST NOT BE BROKEN, both of which produced wrong numbers before they were fixed:

1. Persistence is computed ON THE SPLIT the model was trained on, never borrowed from another. The
   three MIMIC SOFA splits disagree on the renal component's normalisation mean (0.9091 in noclin
   against 0.7571 in clin/oracle); that moves 2,886 target cells and 0.010 macro-F1, which is a
   quarter of the SOFA gain. A persistence value from the wrong split silently shifts every delta.

2. Circularity uses GRADIENT BOOSTING on the oracle split; gain uses the GRU on the causal split.
   The two questions want different models. A computed score is a thresholded step function of its
   inputs, which a tree expresses natively and a GRU with a copy path must approximate -- GBT beats
   the GRU by 0.044 macro-F1 on the UMM SOFA oracle. Conversely the GRU can use the 24-hour sequence
   that the tree, fed daily aggregates, cannot -- it wins on every causal arm. Using one architecture
   for both axes understates circularity exactly for the computed labels that anchor its high end.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
KEEP = os.path.dirname(HERE)
CANON_SEEDS = (42, 123, 456, 789, 1337)

# macro-F1 of gradient boosting on each ORACLE split, from
#   05_baselines.py --data_dir <split> --mask both --use_clinical
# Re-derive with: bash exp_fixnn/run_analysis.sh  (REPORT=... ) or the command above per target.
# Fallbacks only. _gbt_oracle() prefers the value written by 05_baselines into the split directory,
# so that regenerating a split cannot leave the oracle half of `circ = gbt - persistence` frozen
# while the persistence half is recomputed live. If a split has no baseline_results file the
# fallback is used AND announced, rather than silently standing in for a stale number.
# Last-known GBT oracle macro-F1 per target, for provenance only -- NEVER used as a value.
# Kept so a rebuilt baseline can be eyeballed against what the paper was written from.
GBT_ORACLE_REFERENCE = {
    "mimic_sofa": 0.8986, "mimic_aki": 0.7314, "mimic_support": 0.9488, "eicu_dx": 0.9877,
    "gtsq": 0.8523, "umm_sofa": 0.8226,
}


# the causal (gap-matched) split and the trained-run group for each target
TARGETS = [
    ("mimic_sofa",    "mimic_sofa_pred_clin",              False, "mimic_sofa_clin",    "02_models_circ"),
    ("mimic_aki",     "mimic_aki_pred_clin",               False, "mimic_aki_clin",     "02_models_circ"),
    ("mimic_support", "mimic_support_pred_clin",           False, "mimic_support_clin", "02_models_circ"),
    ("eicu_dx",       "eicu_dx_pred_clin",                 False, "eicu_dx_clin",       "02_models_circ"),
    ("gtsq",          "gtsq_pred_24h_gap6h_clin_cult",     True,  "clin_k0",            "exp_fixnn/02_models_m2"),
    # HiRID SOFA: the SAME computed score as mimic_sofa, on a different institution. It is a
    # genuine sixth point, not a control -- different cohort, different hospital -- and it is the
    # evidence that circularity is a property of the label AND the database that records it, since
    # the identical SOFA definition scores +0.264 on MIMIC-IV and -0.031 here.
    ("hirid_sofa",    "hirid_sofa_pred_clin",              False, "hirid_sofa_clin",    "exp_fixnn/02_models_circ"),
]
# reported separately: same cohort as gtsq, so it is a control rather than a fifth point
# usofa2_* are the runs on the splits rebuilt after the urine_output edge-day fix; the
# earlier usofa_* runs used a different renal subscore and must not be paired with the
# current persistence (doing so read a gain of -0.0226 instead of -0.0085).
UMM_SOFA = ("umm_sofa", "umm_sofa_pred_clin", True, "usofa2_clin", "exp_fixnn/02_models_m2")
# oracle split per target -- spelled out because the causal names do not transform into the
# oracle names by any single rule (gtsq is gap-6h_clin_cult, not _oracle_cult).
ORACLE_SPLIT = {"mimic_sofa": "mimic_sofa_pred_oracle", "mimic_aki": "mimic_aki_pred_oracle",
                "mimic_support": "mimic_support_pred_oracle", "eicu_dx": "eicu_dx_pred_oracle",
                "gtsq": "gtsq_pred_24h_gap-6h_clin_cult", "umm_sofa": "umm_sofa_pred_oracle",
                "hirid_sofa": "hirid_sofa_pred_oracle"}


def _split_dir(split, umm):
    return os.path.join(HERE if umm else KEEP, "01_split_sets", split)


def persistence_on(split, umm=False):
    """Macro-F1 of copying day D, computed on THIS split (see convention 1 in the module docstring)."""
    d = _split_dir(split, umm)
    z = np.load(os.path.join(d, "test.npz"), allow_pickle=True)
    n = np.load(os.path.join(d, "norm_stats.npz"), allow_pickle=True)
    gi = z["gtsq_in"]
    gi = gi[:, -1, :] if gi.ndim == 3 else gi
    im = z["gtsq_in_mask"]
    im = im[:, -1, :] if im.ndim == 3 else im
    mask = (im == 1) & (z["gtsq_out_mask"] == 1)
    gm, gs = n["gtsq_means"].astype(float), n["gtsq_stds"].astype(float)
    ncls = [int(x) for x in n["gtsq_nclasses"]]
    inp = np.round(gi * gs + gm).astype(int)
    tgt = np.round(z["gtsq_out"] * gs + gm).astype(int)
    per_feat = [
        f1_score(tgt[mask[:, j], j], np.clip(inp[mask[:, j], j], 0, ncls[j] - 1),
                 average="macro", labels=list(range(ncls[j])), zero_division=0)
        for j in range(len(ncls)) if mask[:, j].sum()
    ]
    return float(np.mean(per_feat)), int(mask.sum()), len(gi)


def _gbt_oracle(name, split, umm):
    """GBT macro-F1 on the oracle split, read from 05_baselines' own output where available."""
    d = _split_dir(split, umm)
    for fn in ("baseline_results_clin.json", "baseline_results.json"):
        path = os.path.join(d, fn)
        if not os.path.exists(path):
            continue
        try:
            blob = json.load(open(path))
        except Exception:
            continue
        # 05_baselines nests the models under a 'results' key; scanning only the top level made
        # every target fall through to the typed-in fallback even where a live value existed.
        res = blob.get("results", blob)
        v = res.get("gbt")
        if isinstance(v, dict) and "f1_macro" in v:
            return float(v["f1_macro"]), fn
    # Previously returned a typed-in constant. Those go stale silently -- umm_sofa's drifted 0.016
    # from the live value while still being reported as a measurement. A missing baseline is a
    # missing measurement; say so and stop rather than substituting a number nobody re-checked.
    raise SystemExit(
        f"ERROR: no baseline_results*.json for '{name}'. Re-run 05_baselines.py for this target; "
        f"circularity will not be reported from a typed-in constant.")


def _seed_of(run_dir):
    m = re.search(r"_s(\d+)(?:_|$)", os.path.basename(run_dir.rstrip("/")))
    return int(m.group(1)) if m else None


def run_macro_f1(group, root):
    """Seed-averaged macro-F1 of a trained run group, restricted to the canonical seeds."""
    vals = []
    # root is relative to KEEP, never to HERE: a target's split root and its run root are
    # independent (hirid_sofa keeps its split under keepgate/ and its runs under exp_fixnn/), and
    # deriving the run root from the `umm` split flag silently produced 0 seeds and a nan gain.
    for path in glob.glob(os.path.join(KEEP, root, group, "*", "gtsq_metrics_test.json")):
        if _seed_of(os.path.dirname(path)) not in CANON_SEEDS:
            continue
        per_feat = [v["f1_macro"] for v in json.load(open(path)).values()
                    if isinstance(v, dict) and "f1_macro" in v]
        if per_feat:
            vals.append(float(np.mean(per_feat)))
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def cmd_crosstarget():
    print("  circularity = GBT(oracle) - persistence;  gain = GRU(causal) - persistence")
    print("  persistence is computed on the SAME split as the model it is compared against\n")
    print(f"  {'target':16s}{'pers':>9}{'gain':>10}{'circularity':>13}{'cells':>10}{'seeds':>7}")
    circ, gain = [], []
    for name, split, umm, group, root in TARGETS:
        pers, cells, _ = persistence_on(split, umm)
        o, src = _gbt_oracle(name, ORACLE_SPLIT[name], umm)
        mf1, n = run_macro_f1(group, root)
        g, c = mf1 - pers, o - pers
        circ.append(c)
        gain.append(g)
        print(f"  {name:16s}{pers:>9.4f}{g:>+10.4f}{c:>+13.4f}{cells:>10,}{n:>7d}"
              f"   [{src}]")
    ok = [(c, g) for c, g in zip(circ, gain) if np.isfinite(c) and np.isfinite(g)]
    if len(ok) > 2:
        cc, gg = zip(*ok)
        print(f"\n  r(circularity, gain) = {np.corrcoef(cc, gg)[0, 1]:.2f}   over {len(ok)} targets")
    name, split, umm, group, root = UMM_SOFA
    pers, cells, _ = persistence_on(split, umm)
    mf1, n = run_macro_f1(group, root)
    print("\n  WITHIN-COHORT CONTROL -- same patients and charting as gtsq, computed label:")
    o, src = _gbt_oracle(name, ORACLE_SPLIT[name], umm)
    print(f"  {name:16s}{pers:>9.4f}{mf1 - pers:>+10.4f}{o - pers:>+13.4f}"
          f"{cells:>10,}{n:>7d}   [{src}]")
    print("  High circularity with NO causal gain; it does not lie on the trend above.")


def cmd_controls():
    """The two controls on the UMM-vs-MIMIC SOFA gap (run by CIRC_CONTROLS=1 run_extra_data.sh).

    E1 SIZE: MIMIC SOFA retrained at UMM's training-set size. E2 COMPONENTS: UMM SOFA rebuilt with
    the CNS component so the two targets are component-matched. Persistence is read from the SAME
    split as the model it is compared against, which is the convention everywhere in this file --
    reading it off the oracle split is what made the first E2 number wrong.
    """
    import glob, importlib.util
    # report.py owns the per-run metric harvester (_one); import it rather than duplicate it, and
    # fail loudly if it cannot be loaded. Guarding this with a silent `None` fallback printed
    # "(pending)" for every row and looked exactly like a job that had not run yet.
    _spec = importlib.util.spec_from_file_location("rp", os.path.join(HERE, "report.py"))
    rp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rp)

    def agg(pat):
        # DEDUPE BY SEED, keeping the newest run. Re-running a block (after fixing an unrelated bug,
        # say) leaves two directories per seed, and averaging both silently double-weights every
        # seed while reporting n=10 for a 5-seed experiment. Caught exactly that on usofa6_clin.
        hits = glob.glob(os.path.join(KEEP, pat))
        by_seed = {}
        for f in sorted(hits):
            d = os.path.basename(os.path.dirname(f))
            m = re.search(r"_s(\d+)(?:_|$)", d)
            key = m.group(1) if m else d
            by_seed[key] = f                       # sorted() puts the newest timestamp last
        v = [x for x in (rp._one(f) for f in by_seed.values()) if x]
        if hits and not v:
            raise SystemExit(f"ERROR: {len(hits)} run(s) matched {pat} but none yielded metrics")
        if len(hits) != len(by_seed):
            print(f"    (note: {len(hits)} run dirs collapsed to {len(by_seed)} unique seeds)")
        return (float(np.array(v).mean(0)[0]), len(v)) if v else (None, 0)
    def check_mask(pat, split):
        """Refuse to difference a model against a baseline computed under a different mask.

        03_train defaults to --mask target; 05_baselines defaults to --mask both. Omitting --mask on
        a training run therefore produces a mask=target model whose persistence reference is
        mask=both, and the gain is a difference of two conventions rather than a result. That is
        exactly what happened to the first E1/E2 runs, and nothing in the output showed it.
        """
        import glob as _g
        # dirname(pat) already ends in the run-dir wildcard, so strip TWO levels off the metrics
        # path to reach the group directory. Joining another "*" globbed one level too deep and
        # matched nothing, which made the guard silently unable to fire.
        grp = os.path.dirname(os.path.dirname(pat))
        cfgs = _g.glob(os.path.join(KEEP, grp, "*", "config.json"))
        if not cfgs:
            raise SystemExit(f"ERROR: no config.json under {grp}; cannot verify the mask convention")
        masks = {json.load(open(c)).get("mask") for c in cfgs}
        if masks != {"both"}:
            raise SystemExit(
                f"ERROR: runs under {os.path.dirname(pat)} were trained with mask={masks}, but the "
                f"persistence baseline in {split} is mask=both. Re-run them with --mask both; "
                f"differencing the two conventions is not a gain.")

    def pers(split):
        f = os.path.join(KEEP, split, "baseline_results_clin.json")
        if not os.path.exists(f):
            return None
        b = json.load(open(f)); b = b.get("results", b)
        p = b.get("persistence")
        return float(p["f1_macro"]) if isinstance(p, dict) else None

    print("  E1 -- is the UMM SOFA gap a TRAINING-SET-SIZE effect?")
    pm = pers("01_split_sets/mimic_sofa_pred_clin")
    for sub, lab in (("13332", "subsampled to UMM size"), ("0", "full training set")):
        check_mask(f"exp_fixnn/02_models_circ/mimic_sofa_sub{sub}/*/gtsq_metrics_test.json",
                   "01_split_sets/mimic_sofa_pred_clin")
        m, n = agg(f"exp_fixnn/02_models_circ/mimic_sofa_sub{sub}/*/gtsq_metrics_test.json")
        if m is None or pm is None:
            print(f"    MIMIC SOFA {lab:24s} (pending)"); continue
        print(f"    MIMIC SOFA {lab:24s} macro-F1 {m:.4f}  gain {m - pm:+.4f}  [n={n}]")
    print("    UMM SOFA (5 components), for comparison        gain -0.0122")

    print("\n  E2 -- is it a COMPONENT-MATCHING effect (CNS excluded from UMM)?")
    pu = pers("exp_fixnn/01_split_sets/umm_sofa6_pred_clin")
    check_mask("exp_fixnn/02_models_circ/usofa6_clin/*/gtsq_metrics_test.json",
               "exp_fixnn/01_split_sets/umm_sofa6_pred_clin")
    m, n = agg("exp_fixnn/02_models_circ/usofa6_clin/*/gtsq_metrics_test.json")
    o = os.path.join(KEEP, "exp_fixnn/01_split_sets/umm_sofa6_pred_oracle/baseline_results_clin.json")
    gbt = None
    if os.path.exists(o):
        b = json.load(open(o)); b = b.get("results", b)
        gbt = float(b["gbt"]["f1_macro"]) if isinstance(b.get("gbt"), dict) else None
    if None in (pu, m):
        print("    UMM SOFA 6-component (pending)")
    else:
        print(f"    UMM SOFA 6-component  persistence {pu:.4f}  causal {m:.4f}  gain {m - pu:+.4f}  [n={n}]")
        if gbt is not None:
            print(f"    {'':22s}GBT oracle  {gbt:.4f}  circularity {gbt - pu:+.4f}")
        print("    5-component version:  gain -0.0122   circularity +0.1624")


def cmd_sofacomp():
    def report(split, umm, causal_group, oracle_group, root):
        d = _split_dir(split, umm)
        z = np.load(os.path.join(d, "test.npz"), allow_pickle=True)
        n = np.load(os.path.join(d, "norm_stats.npz"), allow_pickle=True)
        feats = [str(x) for x in n["gtsq_features"]]
        gi = z["gtsq_in"]
        gi = gi[:, -1, :] if gi.ndim == 3 else gi
        im = z["gtsq_in_mask"]
        im = im[:, -1, :] if im.ndim == 3 else im
        mask = (im == 1) & (z["gtsq_out_mask"] == 1)
        gm, gs = n["gtsq_means"].astype(float), n["gtsq_stds"].astype(float)
        inp = np.round(gi * gs + gm).astype(int)
        tgt = np.round(z["gtsq_out"] * gs + gm).astype(int)

        def per_feature(group):
            acc = {}
            for path in glob.glob(os.path.join(HERE, root, group, "*", "gtsq_metrics_test.json")):
                if _seed_of(os.path.dirname(path)) not in CANON_SEEDS:
                    continue
                for k, v in json.load(open(path)).items():
                    if isinstance(v, dict) and "f1_macro" in v:
                        acc.setdefault(k, []).append(v["f1_macro"])
            return {k: float(np.mean(v)) for k, v in acc.items()}

        causal, oracle = per_feature(causal_group), per_feature(oracle_group)
        head = ("component", "cells", "chg%", "pers", "causal", "oracle", "circ")
        print(f"  {head[0]:16s}{head[1]:>9}{head[2]:>7}{head[3]:>8}{head[4]:>8}{head[5]:>8}{head[6]:>8}")
        for j, f in enumerate(feats):
            m = mask[:, j]
            if not m.sum():
                continue
            pers = f1_score(tgt[m, j], inp[m, j], average="macro",
                            labels=list(range(5)), zero_division=0)
            c, o = causal.get(f, float("nan")), oracle.get(f, float("nan"))
            chg = float((inp[m, j] != tgt[m, j]).mean() * 100)
            print(f"  {f.replace('sofa_', ''):16s}{int(m.sum()):>9,}{chg:>7.1f}"
                  f"{pers:>8.3f}{c:>8.3f}{o:>8.3f}{o - pers:>+8.3f}")

    print("  Per-component columns use the GRU on both arms (a per-component GBT is not run);\n"
          "  the headline circularity in `crosstarget` uses GBT and is larger.")
    print("  MIMIC SOFA (6 components)")
    report("mimic_sofa_pred_clin", False, "mimic_sofa_clin", "mimic_sofa_oracle", "../02_models_circ")
    print("\n  UMM SOFA (5 components; CNS excluded, as in the Paper-1 analysis)")
    report("umm_sofa_pred_clin", True, "usofa2_clin", "usofa2_oracle", "02_models_m2")


CMDS = {"crosstarget": cmd_crosstarget, "sofacomp": cmd_sofacomp, "controls": cmd_controls}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", default="crosstarget", choices=list(CMDS) + ["all"])
    a = ap.parse_args()
    for k, fn in (CMDS.items() if a.cmd == "all" else [(a.cmd, CMDS[a.cmd])]):
        fn()
        print()
