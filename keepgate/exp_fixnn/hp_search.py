#!/usr/bin/env python3
"""Hyperparameter search for the GTSQ next-day task -- SELECTED ON VALIDATION ONLY.

Why this exists: the paper claims no learned model beats persistence + the neu->akut rule, but every
one of the 219 fits used ONE fixed optimisation setting (lr 5e-4, d_model 128, 2 layers, d_ff 512,
batch 64). Without a search, "capacity is not the constraint" is asserted rather than shown. This
closes that hole.

METHODOLOGY (the point of the script):
  * Every selection decision uses the VALIDATION split. The test split is NEVER read here -- not for
    ranking, not for early stopping, not for reporting. `report` prints val numbers only.
  * The baselines (persistence, persistence+rule) are recomputed ON VALIDATION so the comparison is
    like-for-like; comparing a val-selected model against a test baseline would be meaningless.
  * Only after a single winner is chosen may it be run once on test, by a separate explicit command
    (`--confirm-on-test`), which is the only place test is touched.

Usage:
  python exp_fixnn/hp_search.py grid            # emit the sweep as train commands
  python exp_fixnn/hp_search.py report          # rank configs by VALIDATION macro-F1
  python exp_fixnn/hp_search.py report --confirm-on-test   # test, once, for the val winner only
"""
import argparse, glob, itertools, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); KEEP = os.path.dirname(HERE)
sys.path.insert(0, KEEP)
import gtsq_eval as ge                                                    # noqa: E402

SPLIT = "exp_fixnn/01_split_sets/gtsq_pred_24h_gap6h_clin_cult"
OUT = "exp_fixnn/02_models/hp"

# a deliberately wide sweep around the fixed setting used everywhere else
GRID = dict(
    lr=[1e-4, 5e-4, 2e-3],
    d_model=[64, 128, 256],
    e_layers=[1, 2, 3],
    d_ff=[256, 512, 1024],
    batch_size=[32, 64, 128],
    gate_lambda=[0.1, 0.5, 1.0],
    dropout=[0.0, 0.1, 0.3],
)
SEEDS = [42, 123]          # default shortlist depth; --seeds widens it (see `grid --seeds`)

# ---- stage 2: the OBJECTIVE, not the optimiser -------------------------------------------------
# Stage 1 (GRID above) showed the optimiser is not the constraint: every marginal spans <=0.0021,
# ~2-5x the seed sd, and the best epoch is 1-4 of 200 allowed, so nothing is under-trained. What was
# never varied is WHAT is optimised -- all 320 stage-1 fits used plain per-feature CE with Kendall
# weighting. These three axes target the actual failure modes (ordinal near-misses on the 3/5-class
# judgement items; the 3.8% change base rate; feature-weight starvation), full factorial 4x3x2 = 24.
# loss_balance is NOT an axis here: it is a variant-level dimension (`--balance`), because since
# 2026-07-30 `uniform` is the default and the reported standard. The lw-vs-uniform contrast comes from
# stage 1 instead, where the same 32 configurations were run under both -- 32 paired comparisons
# rather than the 12 this factorial would give.
GRID_OBJ = dict(
    ordinal_loss=[None, "emd", "ordce", "cumlink"],   # partial credit for near-miss ordinal classes
    change_loss=[None, "reweight", "focal"],          # upweight changed cells / down-weight easy ones
)
# The first objective sweep (2026-07-29, all `lw`) DID carry loss_balance as a third axis, so its 24
# config indices mean something different from the 12 above. Those runs still exist under
# `obj_gru_{f1,acc}_hpNN`; this grid is what labels them, selected automatically because they are the
# lw-namespaced objective variants. Nothing is discarded or relabelled.
GRID_OBJ_LEGACY = dict(GRID_OBJ, loss_balance=["lw", "uniform"])
# ---- stage 3: the GATE's cost of leaving persistence -------------------------------------------
# copy_scale is the logit magnitude of the copy path (the copied class gets +copy_scale). It was
# hardcoded at 8.0 -- a near-saturated vote the gate must overcome before it can change any cell, so
# it silently sets the price of departing from persistence. It is now a CLI flag; crossed with the
# gate loss weight, which is the countervailing pressure.
GRID_CS = dict(
    copy_scale=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    gate_lambda=[0.1, 0.5, 1.0],
)

# optimiser setting held fixed for stages 2-3: the stage-1 validation winner (hp27).
# loss_balance is stated explicitly rather than left to 03_train's default (which is `lw`, NOT
# `uniform`) so every emitted command is self-documenting.
BEST_OPTIM = dict(lr=2e-3, d_model=64, e_layers=1, d_ff=512, batch_size=32,
                  gate_lambda=1.0, dropout=0.3)


def sample(n=32, seed=0):
    """Random search over GRID (random search dominates grid search at equal budget)."""
    rng = np.random.RandomState(seed)
    keys = list(GRID); seen = set(); out = []
    while len(out) < n:
        c = tuple(GRID[k][rng.randint(len(GRID[k]))] for k in keys)
        if c in seen:
            continue
        seen.add(c); out.append(dict(zip(keys, c)))
    return out


def _factorial(grid):
    keys = list(grid)
    return [dict(zip(keys, c)) for c in itertools.product(*(grid[k] for k in keys))]


def sample_obj(*_a, **_k):
    """Full factorial over GRID_OBJ, in a fixed order so tag `hpNN` is stable."""
    return _factorial(GRID_OBJ)


def sample_obj_legacy(*_a, **_k):
    """The 24-config grid of the 2026-07-29 objective sweep (loss_balance was still an axis)."""
    return _factorial(GRID_OBJ_LEGACY)


def sample_cs(*_a, **_k):
    """Full factorial over GRID_CS (copy_scale x gate_lambda)."""
    return _factorial(GRID_CS)


STAGES = {                       # stage -> (prefix token, config list, grid for marginals)
    "optim":     (None,  sample,     GRID),
    "objective": ("obj", sample_obj, GRID_OBJ),
    "copyscale": ("cs",  sample_cs,  GRID_CS),
}


def variant(arch, select, persist="copy", tau=0.9, stage="optim", balance="uniform", mask="target"):
    """Directory-name prefix for an (architecture, selection-metric, persistence-path) variant.

    The original sweep -- gru, macro-F1 selection, copy path -- keeps the bare `hpNN` tags so its 64
    fits stay valid; every other variant is namespaced. Prefixes carry no digits, so `hpNN` stays
    parseable. `copy` is omitted from the prefix (it is the default path).
    """
    # `lw` was the setting for every fit up to 2026-07-30; `uniform` is now the default and the
    # reported standard, so uniform variants are namespaced `uni...` and the historical lw dirs keep
    # their names. Runs are ALSO filtered by the `_lb<balance>_` tag in the run name, so a variant can
    # never average an lw run together with a uniform one.
    bt = ("m2" if mask == "both" else "") + ("uni" if balance == "uniform" else "")
    tok = STAGES[stage][0]
    if tok:
        return f"{bt}{tok}_{arch}_{select}_"
    if (arch, select, persist) == ("gru", "f1", "copy") and balance == "lw":
        return ""
    pp = "" if persist == "copy" else f"{persist}t{int(round(tau * 100))}_"
    return (f"{bt}_" if bt else "") + f"{arch}_{select}_" + pp


def parse_variant(pre):
    """Inverse of variant(): prefix -> (stage, arch, select_metric, persist_path, balance)."""
    if pre == "":
        return "optim", "gru", "f1", "copy", "lw"
    p = pre.rstrip("_").split("_")
    bal = "lw"
    if p[0].startswith("m2"):
        p = ([p[0][2:]] if len(p[0]) > 2 else []) + p[1:]     # strip the mask token
    if p and p[0].startswith("uni"):
        bal = "uniform"
        p = ([p[0][3:]] if len(p[0]) > 3 else []) + p[1:]      # strip the `uni` token
    for st, (tok, _, _) in STAGES.items():                  # e.g. obj_<arch>_<select>_
        if tok and p[0] == tok:
            return st, p[1], p[2], "copy", bal
    return "optim", p[0], p[1], (p[2] if len(p) > 2 else "copy"), bal   # p[2] carries tau


def cmd_grid(a):
    """Emit the sweep as 03_train commands. Stage `optim` varies the optimiser, `objective` holds the
    stage-1 winner fixed and varies the loss instead."""
    py = "python"
    pre = variant(a.arch, a.select, a.persist, a.markov_tau, a.stage, a.balance, a.mask)
    fixed = a.stage != "optim"                # stages 2-3 hold the stage-1 winner's optimiser fixed
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else SEEDS
    for i, c in enumerate(STAGES[a.stage][1](a.n, a.seed)):
        cfg = dict(BEST_OPTIM, **c) if fixed else dict(c)
        cfg.setdefault("loss_balance", a.balance)
        flags = " ".join(f"--{k} {v}" for k, v in cfg.items() if v is not None)
        for s in seeds:
            tag = f"{pre}hp{i:02d}"
            print(f"{py} 03_train.py --data_dir {SPLIT} --output_dir {OUT}/{tag} "
                  f"--model {a.arch} --select_metric {a.select} "
                  f"--persist_path {a.persist} --markov_tau {a.markov_tau} --mask {a.mask} "
                  f"--use_gate " + flags + f" --seed {s}")


def baselines_on_val():
    """persistence and persistence+neu->akut, computed on VALIDATION (target-observed)."""
    n = np.load(os.path.join(KEEP, SPLIT, "norm_stats.npz"), allow_pickle=True)
    fe = [str(x) for x in n["gtsq_features"]]; ft = [str(x) for x in n["gtsq_ftypes"]]
    nc = [int(x) for x in n["gtsq_nclasses"]]
    gm, gs = n["gtsq_means"].astype(float), n["gtsq_stds"].astype(float)
    d = np.load(os.path.join(KEEP, SPLIT, "val.npz"), allow_pickle=True)
    gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
    I = ge.denorm_to_class(gi, gm, gs, ft, nc)
    T = ge.denorm_to_class(d["gtsq_out"], gm, gs, ft, nc)
    # baselines must use the SAME convention as the models they are compared against, or a mask=both
    # model is scored against a mask=target baseline and every delta is wrong
    im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
    M = (d["gtsq_out_mask"] == 1) & (im == 1)
    R = I.copy()
    for j, f in enumerate(fe):
        if "akut" in f.lower():
            R[:, j] = np.where(I[:, j] == 1, 2, I[:, j])
    out = {}
    for nm, P in (("persistence", I), ("persistence + neu->akut", R)):
        acc, f1, ca, f1c = ge.four_metrics(P, I, T, M, nc)
        out[nm] = (f1, acc, ca, f1c)
    return out


def read(run, split):
    p = os.path.join(run, f"gtsq_metrics_{split}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    f = [v for v in j.values() if isinstance(v, dict) and {"f1_macro", "accuracy", "n_total"} <= set(v)]
    if not f:
        return None
    nt = sum(v["n_total"] for v in f)
    nch = sum(v.get("n_changed", 0) for v in f)
    return (float(np.mean([v["f1_macro"] for v in f])),                       # macro-F1 (per-feature mean)
            sum(v["accuracy"] * v["n_total"] for v in f) / nt,                # cell-pooled accuracy
            (sum(v.get("acc_change", 0.0) * v.get("n_changed", 0) for v in f) / nch) if nch else 0.0,
            float(np.mean([v.get("f1_change", 0.0) for v in f])))             # macro F1-change


def _seed_of(run_dir):
    m = re.search(r"_s(\d+)(?:_|$)", os.path.basename(run_dir.rstrip("/")))
    return int(m.group(1)) if m else -1


TAG = re.compile(r"^(.*?)hp(\d+)$")        # prefix (may be empty) + config index




def _perfeat(run):
    """Per-feature val macro-F1 for one run, or None."""
    f = os.path.join(run, "gtsq_metrics_val.json")
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return {k: v["f1_macro"] for k, v in j.items()
            if isinstance(v, dict) and "f1_macro" in v}


def cmd_compare(a):
    """PAIRED uniform-vs-lw comparison -- the same configuration under both weightings.

    Pairing is by configuration index within a stage/arch/select cell, so every comparison holds the
    optimiser, the architecture and the seed set fixed and varies only --loss_balance. Reports the
    overall four metrics and then the per-feature picture, because the two weightings are expected to
    trade features against each other rather than dominate: `lw` (Kendall) lets the learned
    log-variance concentrate capacity on features it can fit, `uniform` spends it evenly.

    Validation only.
    """
    stage, arch, sel = a.stage, a.arch, a.select
    j = 0 if sel == "f1" else 1
    pre_u = variant(arch, sel, a.persist, a.markov_tau, stage, "uniform")
    pre_l = variant(arch, sel, a.persist, a.markov_tau, stage, "lw")
    cfgs = sample_obj() if stage == "objective" else STAGES[stage][1](a.n, a.seed)

    def _gather(dirs, bal):
        runs = [r for d in dirs for r in glob.glob(os.path.join(d, "*"))
                if f"_lb{bal}_" in os.path.basename(r)]
        m = [x for x in (read(r, "val") for r in runs) if x]
        pf = [x for x in (_perfeat(r) for r in runs) if x]
        return (np.mean(np.array(m), axis=0), pf, len(m)) if m else None

    def collect(pre, bal):
        """key -> (metrics, per-feature dicts, n fits). Keyed by config index, except in the
        objective stage where the lw arm lives in the 24-config legacy dirs, so both arms are keyed by
        the SETTINGS (ordinal_loss, change_loss) and pairing stays exact across the two grids."""
        out = {}
        if stage != "objective":
            for i in range(len(cfgs)):
                g = _gather([os.path.join(KEEP, OUT, f"{pre}hp{i:02d}")], bal)
                if g:
                    out[i] = g
            return out
        for i, c in enumerate(sample_obj()):                       # new 12-config uniform dirs
            key = (c["ordinal_loss"], c["change_loss"])
            d = os.path.join(KEEP, OUT, f"{variant(arch, sel, 'copy', 0.9, stage, 'uniform')}hp{i:02d}")
            g = _gather([d], bal)
            if g:
                out[key] = g
        for i, c in enumerate(sample_obj_legacy()):                # legacy 24-config dirs (both arms)
            if c["loss_balance"] != bal:
                continue
            key = (c["ordinal_loss"], c["change_loss"])
            d = os.path.join(KEEP, OUT, f"{variant(arch, sel, 'copy', 0.9, stage, 'lw')}hp{i:02d}")
            g = _gather([d], bal)
            if g and key not in out:
                out[key] = g
        return out

    U, L = collect(pre_u, "uniform"), collect(pre_l, "lw")
    both = sorted(set(U) & set(L), key=str)
    print("=" * 96)
    print(f"PAIRED  uniform vs lw   stage={stage} arch={arch} select={sel}   "
          f"{len(both)} paired configs   (VALIDATION only)")
    print("=" * 96)
    if not both:
        print("  no paired configs yet -- the uniform re-run is still in flight.")
        return
    by = {t: [x[0] for x in d.values()] for t, d in by.items()}   # one entry per seed
    base = baselines_on_val()
    for nm, v in base.items():
        print(f"  [val baseline] {nm:26s} macro-F1 {v[0]:.4f}  acc {v[1]:.4f}")

    dif = np.array([U[i][0] - L[i][0] for i in both])
    names = ["macro-F1", "accuracy", "chg-acc", "chg-F1"]
    print(f"\n  OVERALL paired delta (uniform - lw), n={len(both)} configs:")
    for k, nm in enumerate(names):
        d = dif[:, k]
        win = int((d > 0).sum())
        print(f"    {nm:9s} mean {d.mean():+.4f}  sd {d.std():.4f}  "
              f"uniform better in {win}/{len(d)} configs  [median {np.median(d):+.4f}]")
    print(f"\n  best config each way (on val {names[j]}):")
    bu = max(both, key=lambda i: U[i][0][j]); bl = max(both, key=lambda i: L[i][0][j])
    print(f"    uniform: {bu}  {U[bu][0][j]:.4f}     lw: {bl}  {L[bl][0][j]:.4f}"
          f"     -> {'uniform' if U[bu][0][j] >= L[bl][0][j] else 'lw'} wins the cell")

    # ---- per feature: where does each weighting actually win? ----
    feats = sorted({f for i in both for pf in U[i][1] + L[i][1] for f in pf})
    rows = []
    for f in feats:
        u = [x[f] for i in both for x in U[i][1] if f in x]
        l = [x[f] for i in both for x in L[i][1] if f in x]
        if len(u) < 2 or len(l) < 2:
            continue
        # noise scale: within-config spread of the same setting, pooled
        sd = float(np.mean([np.std([x[f] for x in U[i][1] if f in x]) for i in both
                            if len([x for x in U[i][1] if f in x]) > 1] or [0.0]))
        rows.append((f, float(np.mean(l)), float(np.mean(u)), float(np.mean(u) - np.mean(l)), sd))
    dead_l = [r for r in rows if r[1] < 0.01]
    dead_u = [r for r in rows if r[2] < 0.01]
    print(f"\n  PER-FEATURE over {len(rows)} features")
    print(f"    features never learned (val macro-F1 < 0.01):  lw {len(dead_l)}   uniform {len(dead_u)}")
    resc = [r for r in rows if r[1] < 0.01 <= r[2]]
    lost = [r for r in rows if r[2] < 0.01 <= r[1]]
    print(f"    rescued by uniform {len(resc)}   lost by uniform {len(lost)}")
    sig = [r for r in rows if abs(r[3]) > 2 * max(r[4], 1e-4)]
    print(f"    moves beyond 2x seed noise: {len(sig)} of {len(rows)}  "
          f"(uniform better {sum(1 for r in sig if r[3] > 0)}, lw better {sum(1 for r in sig if r[3] < 0)})")
    for ttl, sel_rows in (("uniform better", sorted([r for r in sig if r[3] > 0], key=lambda r: -r[3])),
                          ("lw better", sorted([r for r in sig if r[3] < 0], key=lambda r: r[3]))):
        if not sel_rows:
            continue
        print(f"\n    {ttl} (beyond 2x seed sd):")
        print(f"      {'feature':46s}{'lw':>8}{'uniform':>9}{'delta':>9}{'seed sd':>9}")
        for f, l, u, d, sd in sel_rows[:a.top]:
            print(f"      {f[:44]:46s}{l:8.4f}{u:9.4f}{d:+9.4f}{sd:9.4f}")



def cmd_report(a):
    runs = sorted(glob.glob(os.path.join(KEEP, OUT, "*", "*")))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "gtsq_metrics_val.json"))]
    if not runs:
        raise SystemExit(f"no completed hp runs under {OUT}")
    by = {}
    for r in runs:
        tag = r.split("/")[-2]
        m = TAG.match(tag)
        if not m:
            continue
        st, _, _, _, bal = parse_variant(m.group(1))
        # The legacy objective sweep carried loss_balance as a CONFIG axis, so both balances live
        # legitimately inside its dirs and the tag filter must not drop half of them; everywhere else
        # balance is a variant property and the filter keeps the populations apart.
        if not (st == "objective" and bal == "lw"):
            if f"_lb{bal}_" not in os.path.basename(r):
                continue
        v = read(r, "val")
        if v:
            # Overlapping resubmissions can produce two runs of the SAME (config, seed) -- the resume
            # guard checks at issue time, so two shards racing on one fit both run it. Key by seed and
            # keep the newest, or a duplicated seed would be counted twice in the mean.
            sd = _seed_of(r)
            d = by.setdefault(tag, {})
            if sd not in d or os.path.getmtime(r) > d[sd][1]:
                d[sd] = (v, os.path.getmtime(r))
    by = {t: [x[0] for x in d.values()] for t, d in by.items()}   # one entry per seed
    base = baselines_on_val()
    RULE = "persistence + neu->akut"
    print("=" * 96)
    print("HYPERPARAMETER SEARCH -- everything below is VALIDATION (test never read here)")
    print("=" * 96)
    print("  Each variant is ranked on the SAME metric its checkpoints were selected on")
    print("  (03_train.py --select_metric: `f1` = val macro-F1, `acc` = cell-pooled val accuracy),")
    print("  so ranking and checkpointing never disagree. The other columns are reported only.\n")
    for nm, (f1, acc, ca, f1c) in base.items():
        print(f"  [val baseline] {nm:26s} macro-F1 {f1:.4f}  acc {acc:.4f}  "
              f"chg-acc {ca:.4f}  chg-F1 {f1c:.4f}")

    groups = {}
    for t in by:
        m = TAG.match(t)
        if m:
            groups.setdefault(m.group(1), []).append(t)
    best_overall = None
    for pre in sorted(groups, key=lambda p: (p != "", p)):
        stage, arch, sel, pers, bal = parse_variant(pre)
        if stage == "objective" and bal == "lw":        # the 24-config legacy grid, see GRID_OBJ_LEGACY
            cfgs, sgrid = sample_obj_legacy(), GRID_OBJ_LEGACY
        else:
            cfgs, sgrid = STAGES[stage][1](a.n, a.seed), STAGES[stage][2]
        j = 0 if sel == "f1" else 1                       # rank on the selection metric
        tags = groups[pre]
        agg = {t: np.mean(np.array(by[t]), axis=0) for t in tags}
        rank = sorted(tags, key=lambda t: -agg[t][j])
        nfit = sum(len(by[t]) for t in tags)
        print(f"\n  {'-' * 92}")
        print(f"  VARIANT  {'' if stage == 'optim' else stage.upper() + ' '}arch={arch}  "
              f"select_metric={sel}  persist_path={pers}  balance={bal}   "
              f"{len(tags)} configs / {nfit} fits   ranked by val {'macro-F1' if j == 0 else 'accuracy'}")
        print(f"  {'-' * 92}")
        print(f"  {'config':22s}{'seeds':>6}{'macro-F1':>10}{'acc':>9}{'chg-acc':>9}{'chg-F1':>9}   hyperparameters")
        for t in rank[:a.top]:
            m = agg[t]
            i = int(TAG.match(t).group(2))
            hp = cfgs[i] if i < len(cfgs) else {}
            flag = "  <-- beats persistence+rule" if m[j] > base[RULE][j] else ""
            print(f"  {t:22s}{len(by[t]):>6}{m[0]:>10.4f}{m[1]:>9.4f}{m[2]:>9.4f}{m[3]:>9.4f}   "
                  + " ".join(f"{k}={v}" for k, v in hp.items()) + flag)
        if len(rank) > a.top:
            print(f"  ... {len(rank) - a.top} further configs (use --top {len(rank)} for all)")
        nbeat = sum(1 for t in tags if agg[t][j] > base[RULE][j])
        nboth = sum(1 for t in tags if len(by[t]) > 1 and all(x[j] > base[RULE][j] for x in by[t]))
        b = rank[0]
        print(f"\n  best: {b} at val {'macro-F1' if j == 0 else 'acc'} {agg[b][j]:.4f} vs rule "
              f"{base[RULE][j]:.4f} ({agg[b][j] - base[RULE][j]:+.4f});  "
              f"{nbeat}/{len(tags)} configs beat the rule ({nboth} on every seed)")

        scores = {t: float(agg[t][j]) for t in tags}
        idx = {t: int(TAG.match(t).group(2)) for t in tags}
        print("  MARGINALS -- mean over configs carrying each value:")
        for k in sgrid:
            cells, sp = [], []
            for val in sgrid[k]:
                s = [scores[t] for t in tags if idx[t] < len(cfgs) and cfgs[idx[t]][k] == val]
                cells.append(f"{val}: {np.mean(s):.4f} (n={len(s)})" if s else f"{val}: --")
                if s:
                    sp.append(float(np.mean(s)))
            d = (max(sp) - min(sp)) if len(sp) > 1 else 0.0
            print(f"    {k:12s} Δ={d:.4f}   " + "   ".join(cells))
        if best_overall is None or agg[b][j] - base[RULE][j] > best_overall[1]:
            best_overall = (b, agg[b][j] - base[RULE][j], j)

    if a.confirm_on_test:
        best_tag = a.config or best_overall[0]
        t = np.array([x for x in (read(r, "test") for r in runs
                                  if r.split("/")[-2] == best_tag) if x])
        m = t.mean(axis=0)
        print("\n  [TEST -- read ONCE, for the validation winner only]")
        print(f"    {best_tag} ({len(t)} seeds): macro-F1 {m[0]:.4f}  acc {m[1]:.4f}  "
              f"chg-acc {m[2]:.4f}  chg-F1 {m[3]:.4f}")
    else:
        print("\n  (test not read; pass --confirm-on-test only after the winner is fixed)")


def cmd_diagnose(a):
    """WHERE does the val winner's macro-F1 gain come from? Per-feature vs persistence+neu->akut.

    The aggregate number cannot distinguish "learned to forecast the questionnaire" from "learned a
    softer version of the one rule we already apply". This prints the per-feature decomposition, on
    VALIDATION, so the distinction is visible.
    """
    n = np.load(os.path.join(KEEP, SPLIT, "norm_stats.npz"), allow_pickle=True)
    fe = [str(x) for x in n["gtsq_features"]]; ft = [str(x) for x in n["gtsq_ftypes"]]
    nc = [int(x) for x in n["gtsq_nclasses"]]
    gm, gs = n["gtsq_means"].astype(float), n["gtsq_stds"].astype(float)
    d = np.load(os.path.join(KEEP, SPLIT, "val.npz"), allow_pickle=True)
    gi = d["gtsq_in"]; gi = gi[:, -1, :] if gi.ndim == 3 else gi
    I = ge.denorm_to_class(gi, gm, gs, ft, nc)
    T = ge.denorm_to_class(d["gtsq_out"], gm, gs, ft, nc)
    # baselines must use the SAME convention as the models they are compared against, or a mask=both
    # model is scored against a mask=target baseline and every delta is wrong
    im = d["gtsq_in_mask"]; im = im[:, -1, :] if im.ndim == 3 else im
    M = (d["gtsq_out_mask"] == 1) & (im == 1)
    R = I.copy()
    for j, f in enumerate(fe):
        if "akut" in f.lower():
            R[:, j] = np.where(I[:, j] == 1, 2, I[:, j])
    rep = ge.per_feature_report(R, I, T, M, ft, nc)
    rule = {fe[g]: v["f1"] for g, v in rep["per_feat"].items()}
    chg = {fe[g]: 100 * v["n_changed"] / v["n_total"] for g, v in rep["per_feat"].items()}

    tag = a.config
    if tag is None:                                            # default: the val winner
        sc = {}
        for r in sorted(glob.glob(os.path.join(KEEP, OUT, "*", "*"))):
            v = read(r, "val")
            if v:
                sc.setdefault(r.split("/")[-2], []).append(v[0])
        tag = max(sc, key=lambda t: np.mean(sc[t]))
    mdl = {}
    for p in glob.glob(os.path.join(KEEP, OUT, tag, "*", "gtsq_metrics_val.json")):
        for k, v in json.load(open(p)).items():
            if isinstance(v, dict) and "f1_macro" in v:
                mdl.setdefault(k, []).append(v["f1_macro"])
    if not mdl:
        raise SystemExit(f"no val metrics under {OUT}/{tag}")
    mdl = {k: float(np.mean(v)) for k, v in mdl.items()}
    rows = sorted([(k, rule[k], mdl[k], mdl[k] - rule[k]) for k in mdl if k in rule],
                  key=lambda x: -x[3])
    print(f"per-feature VAL macro-F1: {tag} (seed mean) vs persistence+neu->akut  [{len(rows)} features]")
    print(f"  {'feature':44s}{'rule':>8}{'model':>8}{'delta':>9}{'chg%':>7}")
    for k, u, b, dd in rows:
        if abs(dd) > 1e-6:
            print(f"  {k[:44]:44s}{u:>8.4f}{b:>8.4f}{dd:>+9.4f}{chg[k]:>7.1f}")
    D = np.array([x[3] for x in rows])
    g = sorted(D[D > 0])[::-1]
    print(f"\n  better {int((D > 1e-6).sum())}  worse {int((D < -1e-6).sum())}  "
          f"tied {int((abs(D) <= 1e-6).sum())}")
    print(f"  sum gains {D[D > 0].sum():+.4f}  sum losses {D[D < 0].sum():+.4f}  "
          f"mean delta {D.mean():+.5f} (= the aggregate macro-F1 gain)")
    if len(g) >= 3:
        print(f"  top-3 gaining features carry {sum(g[:3]) / sum(g) * 100:.0f}% of all gains")


CMDS = {"grid": cmd_grid, "report": cmd_report, "diagnose": cmd_diagnose,
        "compare": cmd_compare}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None,
                    help="diagnose/--confirm-on-test: config tag (default = val winner)")
    ap.add_argument("--arch", default="gru", choices=["gru", "transformer", "cnn"],
                    help="grid: EHR encoder for the emitted commands")
    ap.add_argument("--select", default="f1", choices=["f1", "acc"],
                    help="grid: checkpoint-selection metric for the emitted commands")
    ap.add_argument("--markov_tau", type=float, default=0.9,
                    help="grid: purity threshold for the in-model transition table. NB neu->akut is "
                         "only 0.61-0.89 pure per organ, so tau=0.9 captures almost nothing; tau=0.5 "
                         "reproduces the unconditional rule used as the paper's baseline")
    ap.add_argument("--persist", default="copy", choices=["copy", "markov"],
                    help="grid: gate persistence path -- copy (plain persistence) or markov "
                         "(copy + confident train transitions, i.e. the neu->akut rule is INSIDE "
                         "the model, so the gate only has to learn deviations from the rule)")
    ap.add_argument("--top", type=int, default=12, help="report: configs shown per variant")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds to emit (default 42,123). Selecting the max over 32 "
                         "configs from a 2-seed mean is noise-driven when the effect is ~1 seed sd; "
                         "widening to five makes the winner reproducible.")
    ap.add_argument("--mask", default="both", choices=["both", "target"],
                    help="grid: masking convention for the loss AND the metrics. `both` (default since "
                         "2026-08-04) scores a cell only when it is observed on both days.")
    ap.add_argument("--balance", default="uniform", choices=["uniform", "lw"],
                    help="grid: --loss_balance for the emitted commands (default uniform, the new "
                         "reported standard)")
    ap.add_argument("--stage", default="optim", choices=list(STAGES),
                    help="grid: `optim` sweeps the optimiser (GRID); `objective` fixes the stage-1 "
                         "winner and sweeps the loss (GRID_OBJ) -- ordinal-aware losses, change "
                         "reweighting, feature-weight balancing; `copyscale` sweeps the copy-path "
                         "logit magnitude x gate loss weight (GRID_CS)")
    ap.add_argument("--confirm-on-test", action="store_true")
    a = ap.parse_args()
    CMDS[a.cmd](a)
