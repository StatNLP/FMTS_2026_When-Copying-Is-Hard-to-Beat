#!/usr/bin/env python3
"""Distribution / change-dynamics analysis for the Keep-Gate MIMIC companion.

Computes the paper numbers on the SAME day-pair population the model sees (the npz
splits), denormalised and rounded to class space exactly like eval_gtsq. Sections:

  1. per-feature: class distribution, change rate, escalation vs weaning;
  2. full per-feature transition matrices P(D+1 | D)  (stickiness = diagonal);
  3. Δ-magnitude breakdown — fraction unchanged / ±1 / ±2 / ±3 / ±4 per feature;
  4. SOFA value frequencies (overall + per subscore) and total-SOFA histogram;
  5. per-ICU-day stratification (day 0→1 … 5→6): are early transitions different
     from late ones? plus the Sepsis-3 P(ΔSOFA_total >= 2) signal per day.

Pooled over train+val+test (descriptive stats are split-invariant). Light +
single-threaded; reads only <split>/{train,val,test}.npz + norm_stats.npz.

Usage:
  python 06_data_analysis.py --data_dir 01_split_sets/mimic_support_pred_noclin
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gtsq_eval as ge   # shared decode + metric definitions


def load_concat(data_dir):
    norm = dict(np.load(os.path.join(data_dir, "norm_stats.npz"), allow_pickle=True))
    gin, gout, min_, mout, dates = [], [], [], [], []
    for sp in ("train", "val", "test"):
        p = os.path.join(data_dir, f"{sp}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        gi = d["gtsq_in"]
        gi = gi[:, -1, :] if gi.ndim == 3 else gi
        gin.append(gi); gout.append(d["gtsq_out"])
        min_.append(d["gtsq_in_mask"]); mout.append(d["gtsq_out_mask"])
        dates.append(d["dates"])
    return (np.concatenate(gin), np.concatenate(gout),
            np.concatenate(min_), np.concatenate(mout),
            np.concatenate(dates), norm)


def to_class(arr_norm, means, stds, ftypes, nclasses):
    """Denormalize + round to class labels (shared definition, gtsq_eval)."""
    return ge.denorm_to_class(arr_norm, means, stds, ftypes, nclasses)


def _entropy_bits(counts):
    """Shannon entropy (bits) of a non-negative count vector."""
    counts = np.asarray(counts, dtype=float)
    p = counts / max(counts.sum(), 1.0)
    return float(-np.where(p > 0, p * np.log2(p), 0).sum())


# ---- Sankey rendering (patient-group day-to-day transition) ------------------------------
# The GTSQ patient group is a RELATIVE ward ranking — the 3 best / 3 worst patients on the
# ward that day, everyone else in the middle. It is NOT an absolute stability/severity score,
# so the labels deliberately say "best-3 / rest / worst-3", not "stable/critical".
_SANKEY_LBL = ["0 · beste 3 (Station)", "1 · Rest", "2 · schlechteste 3 (Station)"]
_SANKEY_COL = ["#2e7d32", "#f9a825", "#c62828"]                 # best-3 / rest / worst-3 (relative rank)


def _sankey_segments(counts, gap=0.04):
    counts = np.asarray(counts, float)
    frac = counts / max(counts.sum(), 1.0)
    avail = 1.0 - gap * (len(counts) - 1)
    segs, y = [], 1.0
    for f in frac:
        h = f * avail
        segs.append((y - h, y)); y -= h + gap
    return segs


def _draw_sankey(ax, T, title):
    """Transition Sankey for a k×k count matrix T (rows=from g_t, cols=to g_{t+1})."""
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path
    T = np.asarray(T, float); k = T.shape[0]
    left, right = _sankey_segments(T.sum(1)), _sankey_segments(T.sum(0))
    xL0, xL1, xR0, xR1 = 0.0, 0.045, 0.955, 1.0
    lcur = [s[1] for s in left]; rcur = [s[1] for s in right]
    for i in range(k):
        for j in range(k):
            w = T[i, j]
            if w <= 0:
                continue
            lh = w / T[i].sum() * (left[i][1] - left[i][0])
            rh = w / T[:, j].sum() * (right[j][1] - right[j][0])
            lt, lb, rt, rb = lcur[i], lcur[i] - lh, rcur[j], rcur[j] - rh
            xm = (xL1 + xR0) / 2
            verts = [(xL1, lb), (xm, lb), (xm, rb), (xR0, rb),
                     (xR0, rt), (xm, rt), (xm, lt), (xL1, lt), (xL1, lb)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(verts, codes), facecolor=_SANKEY_COL[i],
                                   edgecolor="none", alpha=0.6))
            lcur[i] -= lh; rcur[j] -= rh
    rs, cs = T.sum(1), T.sum(0)
    for i, (b, t) in enumerate(left):
        ax.add_patch(Rectangle((xL0, b), xL1 - xL0, t - b, color=_SANKEY_COL[i]))
        ax.text(xL0 - 0.01, (b + t) / 2, f"{_SANKEY_LBL[i]}\n{100*rs[i]/T.sum():.0f}%",
                ha="right", va="center", fontsize=9)
    for j, (b, t) in enumerate(right):
        ax.add_patch(Rectangle((xR0, b), xR1 - xR0, t - b, color=_SANKEY_COL[j]))
        ax.text(xR1 + 0.01, (b + t) / 2, f"{_SANKEY_LBL[j]}\n{100*cs[j]/T.sum():.0f}%",
                ha="left", va="center", fontsize=9)
    ax.set_xlim(-0.16, 1.16); ax.set_ylim(-0.03, 1.06); ax.axis("off")
    ax.set_title(title, fontsize=11)
    ax.text(0.0, 1.05, "heute", ha="center", fontsize=8, color="#555")
    ax.text(1.0, 1.05, "morgen", ha="center", fontsize=8, color="#555")


def render_change_histogram(report, figdir):
    """Histogram of #items changed per day-pair (§6) → PNG+PDF. The 0-bar (full stasis /
    carry-forward) is highlighted: it is the visual signature behind the keep-gate."""
    cpp = report.get("changes_per_pair")
    if not cpp or "hist" not in cpp:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [hist] matplotlib unavailable — skipping"); return
    os.makedirs(figdir, exist_ok=True)
    hist = {int(k): int(v) for k, v in cpp["hist"].items()}
    n = sum(hist.values()); CAP = 12
    ys = [100 * hist.get(k, 0) / n for k in range(CAP + 1)]
    ys.append(100 * sum(v for k, v in hist.items() if k > CAP) / n)
    labels = [str(k) for k in range(CAP + 1)] + [f"{CAP+1}+"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(range(len(ys)), ys, color=["#c62828"] + ["#1f77b4"] * (len(ys) - 1))
    ax.set_xticks(range(len(ys))); ax.set_xticklabels(labels)
    ax.set_xlabel("number of items changed per day-pair (of ~45)")
    ax.set_ylabel("% of day-pairs")
    ax.set_title(f"GTSQ — items changed per day-pair  (n={n:,})")
    ax.annotate(f"mean={cpp.get('mean', 0):.2f}   median={cpp.get('median', 0):.0f}\n"
                f"{100*cpp.get('pct_zero_change', 0):.0f}% with NO change at all (red)\n"
                f"{100*cpp.get('pct_ge5', 0):.0f}% change ≥5 items",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="#fff3cd", ec="#999"))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        stem = os.path.join(figdir, f"gtsq_change_histogram.{ext}")
        fig.savefig(stem, dpi=200, bbox_inches="tight"); print(f"  [hist] wrote {stem}")
    plt.close(fig)


def render_sankey(report, figdir, by_diagnosis=False):
    """Write the patient-group transition Sankey to <figdir>/*.png + *.pdf."""
    pg = report.get("patient_group")
    if pg is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [sankey] matplotlib unavailable — skipping figure"); return
    os.makedirs(figdir, exist_ok=True)
    T = np.array(pg["transition_counts"], float)
    two = report.get("patient_group_two_conditional", {}).get("i03000")
    if by_diagnosis and two is not None:
        ctx = two["contexts"]; k = T.shape[0]
        dl = {0: "Weder SIRS/Sepsis", 1: "SIRS", 2: "Sepsis", 3: "schwere Sepsis", 4: "septischer Schock"}
        strata = [("alle Diagnosen", T)]
        for d, lab in dl.items():
            M = np.zeros((k, k))
            for g in range(k):
                c = ctx.get(f"g{g}_i03000{d}")
                if c:
                    M[g] = np.array(c["p_next"]) * c["n"]
            if M.sum() > 0:
                strata.append((lab, M))
        ncol = 3; nrow = int(np.ceil(len(strata) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)
        for ax, (t, M) in zip(axes.flat, strata):
            _draw_sankey(ax, M, t)
        for ax in axes.flat[len(strata):]:
            ax.axis("off")
        fig.suptitle("GTSQ Patient-Gruppe (relatives Ward-Ranking: beste 3 / Rest / schlechteste 3)\n"
                     "— Tagesübergang, stratifiziert nach Arbeitsdiagnose", fontsize=12)
        stem = os.path.join(figdir, "sankey_patient_group_by_diagnosis")
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        _draw_sankey(ax, T, "GTSQ Patient-Gruppe (relatives Ward-Ranking) — Tagesübergang  "
                            f"(n={int(T.sum()):,})")
        stem = os.path.join(figdir, "sankey_patient_group")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        print(f"  [sankey] wrote {stem}.{ext}")
    plt.close(fig)


def parse_dates(dates, episode_gap_days=7):
    """Parse the per-pair date string into (day_in, weekday, stay).

    Two formats are handled:
      MIMIC companion '<stay>_d<day>'  -> day_in=<day>, weekday=-1, stay=<stay>.
      GTSQ '<hadm>_<YYYY-MM-DD>'        -> weekday from the calendar date (0=Mon..6=Sun),
            and day_in = day-WITHIN-EPISODE.

    GTSQ day_in is NOT computed against the cohort icu_start: the UMM cohort only tracks
    the *first* admission, so a re-admission's questionnaire would land hundreds of days
    after icu_start (the spurious "day 720/1400" stamps). Instead we segment each hadm's
    daily questionnaire run into episodes wherever two consecutive entries are more than
    `episode_gap_days` apart (a long gap = discharge + re-admission) and count days since
    the start of the current episode. The returned `stay` is episode-aware
    ('<hadm>#<episode>') so the per-stay logic in §7/§12 never crosses a re-admission.
    """
    n = len(dates)
    day_in = np.full(n, -1, dtype=int)
    weekday = np.full(n, -1, dtype=int)
    stay = np.empty(n, dtype=object); stay[:] = ""
    cal = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    gtsq_rows = []
    for i, s in enumerate(dates):
        s = str(s)
        head, _, tail = s.rpartition("_")
        if tail[:1] == "d" and tail[1:].isdigit():          # MIMIC '<stay>_d<day>'
            day_in[i] = int(tail[1:]); stay[i] = head or s
        else:                                                # GTSQ '<hadm>_<date>'
            try:
                ts = pd.Timestamp(tail)
                weekday[i] = int(ts.dayofweek); stay[i] = head
                cal[i] = np.datetime64(ts.date()); gtsq_rows.append(i)
            except (ValueError, TypeError):
                stay[i] = s
    if gtsq_rows:                                            # episode-relative day within each hadm
        order = sorted(gtsq_rows, key=lambda i: (str(stay[i]), cal[i]))
        prev_enc = None; prev_d = None; ep = 0; ep_start = None
        for i in order:
            e = str(stay[i]); d = cal[i]
            if e != prev_enc:
                ep, ep_start = 0, d
            elif (d - prev_d) / np.timedelta64(1, "D") > episode_gap_days:
                ep += 1; ep_start = d
            day_in[i] = int((d - ep_start) / np.timedelta64(1, "D"))
            stay[i] = f"{e}#{ep}"
            prev_enc, prev_d = e, d
    return day_in, weekday, stay, cal


def main(args):
    gin, gout, min_, mout, dates, norm = load_concat(args.data_dir)
    feats = [str(f) for f in norm["gtsq_features"]]
    ftypes = norm["gtsq_ftypes"]; nclasses = norm["gtsq_nclasses"]
    means = norm["gtsq_means"].astype(np.float32); stds = norm["gtsq_stds"].astype(np.float32)
    G = len(feats); N = len(gin)
    in_cls = to_class(gin, means, stds, ftypes, nclasses)
    out_cls = to_class(gout, means, stds, ftypes, nclasses)
    day_in, weekday, stay, cal = parse_dates(dates)
    is_sofa = all(f.startswith("sofa_") for f in feats)

    print(f"\n{'='*80}\nDATA ANALYSIS  —  {args.data_dir}")
    print(f"{N:,} day-pairs (train+val+test pooled), {G} targets, input days {day_in.min()}..{day_in.max()}")
    print(f"{'='*80}")
    report = {"data_dir": args.data_dir, "n_pairs": int(N), "features": {}}

    # ---- 1. per-feature distribution + change dynamics ----
    print("\n## 1. Per-feature distribution + change dynamics")
    hdr = f"{'feature':28s}{'type':8s}{'n_valid':>9s}{'mean(D+1)':>10s}{'chg%':>7s}{'esc%':>7s}{'wean%':>7s}"
    print(hdr); print("-" * len(hdr))
    for g in range(G):
        valid = (min_[:, g] > 0) & (mout[:, g] > 0)
        nv = int(valid.sum())
        if nv == 0:
            continue
        ic, oc = in_cls[valid, g], out_cls[valid, g]
        ft = str(ftypes[g]); nc = int(nclasses[g])
        chg = float((ic != oc).mean()); esc = float((oc > ic).mean()); wean = float((oc < ic).mean())
        print(f"{feats[g][:27]:28s}{ft:8s}{nv:>9,}{oc.mean():>10.3f}"
              f"{chg*100:>7.1f}{esc*100:>7.1f}{wean*100:>7.1f}")
        report["features"][feats[g]] = {
            "type": ft, "nclass": nc, "n_valid": nv, "out_mean": float(oc.mean()),
            "change_rate": chg, "escalation_rate": esc, "weaning_rate": wean,
        }

    # ---- 2. full transition matrices ----
    print("\n## 2. Transition matrices  P(D+1 = col | D = row)   [stickiness = diagonal]")
    for g in range(G):
        valid = (min_[:, g] > 0) & (mout[:, g] > 0)
        if valid.sum() == 0:
            continue
        nc = 2 if str(ftypes[g]) == "binary" else int(nclasses[g])
        ic, oc = in_cls[valid, g], out_cls[valid, g]
        T = np.zeros((nc, nc))
        for a, b in zip(ic, oc):
            if 0 <= a < nc and 0 <= b < nc:
                T[a, b] += 1
        rs = T.sum(axis=1, keepdims=True)
        Tn = np.divide(T, rs, out=np.zeros_like(T), where=rs > 0)
        stk = float(np.trace(T) / T.sum()) if T.sum() else float("nan")
        print(f"\n  {feats[g]}   (P(stay)={stk:.3f}, n={int(T.sum()):,})")
        head = "    from\\to " + "".join(f"{c:>8d}" for c in range(nc)) + "    row-n"
        print(head)
        for a in range(nc):
            row = "".join(f"{Tn[a, c]:>8.3f}" for c in range(nc))
            print(f"    {a:>6d}   {row}  {int(T[a].sum()):>8,}")
        report["features"][feats[g]]["stickiness"] = stk
        report["features"][feats[g]]["transition_matrix"] = Tn.round(4).tolist()
        report["features"][feats[g]]["transition_counts"] = T.astype(int).tolist()

    # ---- 3. Δ-magnitude breakdown (ordinal) ----
    ordinal_g = [g for g in range(G) if str(ftypes[g]) == "ordinal"]
    if ordinal_g:
        print("\n## 3. Δ-magnitude  P(out - in = k)   [k<0 improve, k>0 worsen]")
        deltas = list(range(-4, 5))
        print("    " + f"{'feature':24s}" + "".join(f"{('Δ'+('+' if k>0 else '')+str(k)):>8s}" for k in deltas))
        for g in ordinal_g:
            valid = (min_[:, g] > 0) & (mout[:, g] > 0)
            d = (out_cls[valid, g] - in_cls[valid, g])
            frac = {k: float((d == k).mean()) for k in deltas}
            print("    " + f"{feats[g][:23]:24s}" + "".join(f"{frac[k]*100:>8.1f}" for k in deltas))
            report["features"][feats[g]]["delta_dist_pct"] = {int(k): round(frac[k]*100, 3) for k in deltas}

    # ---- 4. value frequencies (per feature) + SOFA total histogram ----
    print("\n## 4. Class-value frequency  P(class)   [pooled over observed in+out day-values]")
    maxnc = max(int(nclasses[g]) if str(ftypes[g]) != "binary" else 2 for g in range(G))
    print("    " + f"{'feature':24s}" + "".join(f"{('cls'+str(c)):>8s}" for c in range(maxnc)))
    for g in range(G):
        vin = in_cls[min_[:, g] > 0, g]; vout = out_cls[mout[:, g] > 0, g]
        allv = np.concatenate([vin, vout])
        nc = 2 if str(ftypes[g]) == "binary" else int(nclasses[g])
        dist = {c: float((allv == c).mean()) for c in range(nc)}
        print("    " + f"{feats[g][:23]:24s}" + "".join(f"{dist.get(c,0)*100:>8.1f}" for c in range(nc)))
        report["features"][feats[g]]["value_freq_pct"] = {int(c): round(dist[c]*100, 3) for c in range(nc)}

    if is_sofa:
        tin = np.where(min_ > 0, in_cls, 0).sum(axis=1)
        tout = np.where(mout > 0, out_cls, 0).sum(axis=1)
        tot = np.concatenate([tin, tout])
        print(f"\n  SOFA total (missing=0): mean={tot.mean():.2f} p50={np.median(tot):.0f} "
              f"p90={np.percentile(tot,90):.0f} p99={np.percentile(tot,99):.0f} max={int(tot.max())}")
        hist = {int(v): int((tot == v).sum()) for v in range(0, int(tot.max()) + 1)}
        report["sofa_total_value_hist"] = hist

    # ---- 5. per-ICU-day stratification (early vs late) ----
    print("\n## 5. Per-ICU-day transition (day d -> d+1)   [GTSQ: day-within-episode; daily to d30,"
          "\n     then coarse buckets so genuine long-stayers (>100d) stay visible]")
    report["per_day"] = {}

    def _day_buckets():
        lo = max(int(day_in.min()), 0); hi = int(day_in.max())
        for dd in range(lo, min(hi, 30) + 1):
            yield f"d{dd}->d{dd+1}", (day_in == dd)
        for a, b in ((31, 60), (61, 90), (91, 10 ** 9)):
            if hi >= a:
                yield (f"d{a}-{b}" if b < 10 ** 9 else f"d{a}+"), (day_in >= a) & (day_in <= b)
        yield "ALL", np.ones(N, bool)
    if is_sofa:
        tin = np.where(min_ > 0, in_cls, 0).sum(axis=1)
        tout = np.where(mout > 0, out_cls, 0).sum(axis=1)
        dtot = tout - tin
        anychg_all = ((in_cls != out_cls) & ((min_ > 0) & (mout > 0))).any(axis=1)
        print(f"    {'pair':10s}{'n':>9s}{'meanΔSOFA':>11s}{'P(Δ>=2)%':>10s}{'P(Δ<=-2)%':>11s}{'P(any chg)%':>12s}")
        for lbl, m in _day_buckets():
            if m.sum() == 0:
                continue
            row = dict(n=int(m.sum()), mean_dsofa=float(dtot[m].mean()),
                       p_ge2=float((dtot[m] >= 2).mean()), p_le_m2=float((dtot[m] <= -2).mean()),
                       p_any_change=float(anychg_all[m].mean()))
            print(f"    {lbl:10s}{row['n']:>9,}{row['mean_dsofa']:>11.3f}"
                  f"{row['p_ge2']*100:>10.2f}{row['p_le_m2']*100:>11.2f}{row['p_any_change']*100:>12.2f}")
            report["per_day"][lbl] = row
    else:
        nsup_in = in_cls.sum(axis=1); nsup_out = out_cls.sum(axis=1); dn = nsup_out - nsup_in
        print(f"    {'pair':10s}{'n':>9s}{'any-sup%':>10s}{'P(esc)%':>9s}{'P(wean)%':>10s}{'P(unchg)%':>11s}")
        for lbl, m in _day_buckets():
            if m.sum() == 0:
                continue
            row = dict(n=int(m.sum()), any_support=float((nsup_out[m] > 0).mean()),
                       p_esc=float((dn[m] > 0).mean()), p_wean=float((dn[m] < 0).mean()),
                       p_unchanged=float((dn[m] == 0).mean()))
            print(f"    {lbl:10s}{row['n']:>9,}{row['any_support']*100:>10.2f}"
                  f"{row['p_esc']*100:>9.2f}{row['p_wean']*100:>10.2f}{row['p_unchanged']*100:>11.2f}")
            report["per_day"][lbl] = row

    # ---- 6. items changed per day-pair (activity / anchoring signature) ----
    print("\n## 6. Items changed per day-pair  [high %zero-change = stasis / carry-forward signature]")
    valid_pair = (min_ > 0) & (mout > 0)                 # (N,G) item observed both days
    changed = (in_cls != out_cls) & valid_pair           # (N,G)
    n_chg = changed.sum(axis=1); n_cmp = valid_pair.sum(axis=1)

    def _domain(f):
        if "i05100" in f or "Lokalisation" in f:
            return "Lokalisierung"
        if "i11100" in f or any(o in f for o in ("Lunge", "Niere", "Leber", "Herz", "ZNS",
                                                 "Gerinnung", "Kreislauf", "Leber")):
            return "Organ"
        return "Klinisch"
    doms = [_domain(f) for f in feats]
    cpp = {"mean": float(n_chg.mean()), "median": float(np.median(n_chg)),
           "pct_zero_change": float((n_chg == 0).mean()), "pct_ge5": float((n_chg >= 5).mean()),
           "hist": {int(k): int((n_chg == k).sum()) for k in range(0, int(n_chg.max()) + 1)},
           "per_domain": {}}
    print(f"    mean={n_chg.mean():.2f}  median={np.median(n_chg):.0f}  "
          f"%zero-change={100*(n_chg==0).mean():.1f}  %>=5={100*(n_chg>=5).mean():.1f}  "
          f"(~{int(np.median(n_cmp))} items compared/pair)")
    for dm in ("Klinisch", "Organ", "Lokalisierung"):
        cols = [g for g in range(G) if doms[g] == dm]
        if not cols:
            continue
        ch = int(changed[:, cols].sum()); cm = int(valid_pair[:, cols].sum())
        rate = ch / max(cm, 1)
        cpp["per_domain"][dm] = {"n_items": len(cols), "change_rate": float(rate)}
        print(f"      {dm:14s} {len(cols):2d} items  change_rate={100*rate:.1f}%")
    for lbl, m in (("early (d%d->)" % day_in.min(), day_in == day_in.min()), ("late (d>=5)", day_in >= 5)):
        if m.sum():
            cpp[lbl] = {"mean_changed": float(n_chg[m].mean()), "pct_zero": float((n_chg[m] == 0).mean())}
            print(f"      {lbl:14s} mean changed={n_chg[m].mean():.2f}  %zero={100*(n_chg[m]==0).mean():.1f}")
    report["changes_per_pair"] = cpp

    # ---- 7. sepsis(infection-suspicion)-on-admission stratification (GTSQ only) ----
    inf_g = next((g for g in range(G) if "Infektionsverdacht" in feats[g] or "i04000" in feats[g]), None)
    grp_g = next((g for g in range(G) if "Gruppe" in feats[g]), None)
    if inf_g is not None:
        print("\n## 7. Infection-suspicion-on-admission stratification")
        stays = stay.astype(str)         # episode-aware stay; admission = earliest day within episode
        adm = {}
        for st in np.unique(stays):
            sm = stays == st
            r0 = np.where(sm & (day_in == day_in[sm].min()))[0]
            adm[st] = int(in_cls[r0[0], inf_g] > 0) if (len(r0) and min_[r0[0], inf_g] > 0) else -1
        septic = np.array([adm.get(st, -1) for st in stays])
        report["infection_on_admission"] = {}
        for lbl, v in (("infected-on-adm", 1), ("not-infected-on-adm", 0)):
            m = septic == v
            if m.sum() == 0:
                continue
            gv = (float(changed[m][:, grp_g].sum() / max(valid_pair[m][:, grp_g].sum(), 1))
                  if grp_g is not None else None)
            report["infection_on_admission"][lbl] = {
                "n_pairs": int(m.sum()), "mean_items_changed": float(n_chg[m].mean()),
                "patient_group_change_rate": gv}
            print(f"    {lbl:22s} n_pairs={int(m.sum()):>7,}  mean changed={n_chg[m].mean():.2f}"
                  + (f"  patient-group chg={100*gv:.1f}%" if gv is not None else ""))

    # ---- 8. weekday vs weekend documentation activity (calendar-date datasets only) ----
    if (weekday >= 0).any():
        print("\n## 8. Weekday vs weekend activity  [dates[i]=input day D; the D->D+1 transition is"
              "\n     documented when the D+1 form is filled, so the 'assessment day' = D+1."
              "\n     fewer changes on a given day => documentation-effort artefact, not stability]")
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        asm = (weekday + 1) % 7                      # assessment day = D+1 (daily questionnaire)
        asm[weekday < 0] = -1
        wk = {}
        for key, wdv in (("by_input_day", weekday), ("by_assessment_day_D+1", asm)):
            print(f"    -- {key} --")
            block = {"per_weekday": {}}
            for wd in range(7):
                m = wdv == wd
                if m.sum() == 0:
                    continue
                block["per_weekday"][names[wd]] = {"n_pairs": int(m.sum()),
                    "mean_items_changed": float(n_chg[m].mean()), "pct_zero_change": float((n_chg[m] == 0).mean())}
                print(f"      {names[wd]} n={int(m.sum()):>6,}  mean changed={n_chg[m].mean():.2f}  "
                      f"%zero={100*(n_chg[m]==0).mean():.1f}")
            for lbl, m in (("weekday(Mon-Fri)", (wdv >= 0) & (wdv <= 4)), ("weekend(Sat-Sun)", wdv >= 5)):
                if m.sum():
                    block[lbl] = {"n_pairs": int(m.sum()), "mean_items_changed": float(n_chg[m].mean()),
                                  "pct_zero_change": float((n_chg[m] == 0).mean())}
                    print(f"      {lbl:18s} n={int(m.sum()):>6,}  mean changed={n_chg[m].mean():.2f}  "
                          f"%zero={100*(n_chg[m]==0).mean():.1f}")
            wk[key] = block
        report["weekday_activity"] = wk

    # ---- 9. item co-change structure (which items tend to change together) ----
    print("\n## 9. Item co-change structure  [phi-corr of change-indicators over jointly-valid pairs]")
    pairs = []
    for a in range(G):
        for b in range(a + 1, G):
            both = valid_pair[:, a] & valid_pair[:, b]
            if both.sum() < 200:
                continue
            ca = changed[both, a].astype(float); cb = changed[both, b].astype(float)
            if ca.std() < 1e-9 or cb.std() < 1e-9:
                continue
            pairs.append((float(np.corrcoef(ca, cb)[0, 1]), a, b, int(both.sum())))
    pairs.sort(reverse=True)
    cc = {"top_pairs": []}
    for phi, a, b, n in pairs[:12]:
        print(f"    phi={phi:+.2f}  {feats[a][:24]:25s} <-> {feats[b][:24]:25s} "
              f"(n={n:,}, {doms[a]}/{doms[b]})")
        cc["top_pairs"].append({"phi": phi, "a": feats[a], "b": feats[b], "n": n,
                                "domains": [doms[a], doms[b]]})
    report["co_change"] = cc

    # ---- 10. patient-group deep-dive: transition matrix + next-day predictability (Sankey data) ----
    grp_g2 = next((g for g in range(G) if "Gruppe" in feats[g] or "i01000" in feats[g]), None)
    if grp_g2 is not None:
        print(f"\n## 10. Patient-group deep-dive  ({feats[grp_g2]}) — transition matrix + next-day entropy")
        v = (min_[:, grp_g2] > 0) & (mout[:, grp_g2] > 0)
        ic, oc = in_cls[v, grp_g2], out_cls[v, grp_g2]
        k = int(max(ic.max(), oc.max())) + 1
        T = np.zeros((k, k))
        for i in range(len(ic)):
            T[ic[i], oc[i]] += 1
        rows = T.sum(1, keepdims=True)
        P = T / np.maximum(rows, 1)
        ent = -np.where(P > 0, P * np.log2(P), 0).sum(1)            # bits of next-day uncertainty per state
        print("    P(D+1|D) rows=today, cols=tomorrow:")
        for s in range(k):
            print("      " + " ".join(f"{P[s,j]:.2f}" for j in range(k))
                  + f"   (n={int(rows[s,0]):>6,}  stay={P[s,s]:.2f}  H={ent[s]:.2f}bit)")
        report["patient_group"] = {"feature": feats[grp_g2], "n_classes": k,
            "transition_counts": T.astype(int).tolist(), "transition_prob": P.round(4).tolist(),
            "diag_stay": [float(P[s, s]) for s in range(k)],
            "next_day_entropy_bits": [float(e) for e in ent]}

    # ---- 11. organ-block encoding: present / acute / chronic combinations (GTSQ only) ----
    organs = {}
    for g, name in enumerate(feats):
        if not name.startswith("i11100_"):
            continue
        parts = name.split("_", 2)                  # i11100, <organ>, <subfield>
        org = parts[1]; sub = parts[2] if len(parts) > 2 else ""
        slot = ("present" if sub == "present" else "acute" if sub.startswith("akut")
                else "chronic" if sub == "chronisch" else "cause" if sub.startswith("Ursache") else None)
        if slot:
            organs.setdefault(org, {})[slot] = g
    if organs:
        ACUTE_LBL = {0: "none", 1: "neu", 2: "akut"}        # AKUT_MAP: neu=1, akut=2; 0=present-but-neither
        print("\n## 11. Organ-block encoding — present / acute(0=none|neu|akut) / chronic (in+out pooled)")
        print("     each organ = {present(bin), akut-oder-neu(0=none,1=neu,2=akut), chronisch(bin), Ursache(0..3)}")
        hdr = (f"    {'organ':16s}{'n_obs':>8s}{'P(pres)':>8s}{'none|p':>8s}{'neu|p':>8s}"
               f"{'akut|p':>8s}{'chr|p':>8s}{'leak:ac|!p':>11s}{'p==(ac|chr)':>12s}")
        print(hdr); print("    " + "-" * (len(hdr) - 4))
        org_rep = {}; combos_all = {}
        for org, ix in organs.items():
            if not {"present", "acute", "chronic"} <= set(ix):
                continue
            pg, ag, cg = ix["present"], ix["acute"], ix["chronic"]
            P, A, C = [], [], []
            for sc, sm in ((in_cls, min_), (out_cls, mout)):
                jv = (sm[:, pg] > 0) & (sm[:, ag] > 0) & (sm[:, cg] > 0)
                P.append(sc[jv, pg]); A.append(sc[jv, ag]); C.append(sc[jv, cg])
            P = np.concatenate(P); A = np.concatenate(A); C = np.concatenate(C)
            n = len(P)
            if n == 0:
                continue
            pres = P == 1; chr_ = C == 1
            pp = float(pres.mean())
            fp = lambda cond: float(cond[pres].mean()) if pres.any() else 0.0   # P(cond | present)
            p_none, p_neu, p_akut, p_chr = fp(A == 0), fp(A == 1), fp(A == 2), fp(chr_)
            leak = float((A[~pres] > 0).mean()) if (~pres).any() else 0.0
            ident = float((pres == ((A > 0) | chr_)).mean())     # is present == (acute>0 OR chronic)?
            print(f"    {org[:15]:16s}{n:>8,}{pp:>8.3f}{p_none:>8.3f}{p_neu:>8.3f}"
                  f"{p_akut:>8.3f}{p_chr:>8.3f}{leak:>11.3f}{ident:>12.3f}")
            org_rep[org] = {"n_obs": n, "p_present": pp,
                "given_present": {"none": p_none, "neu": p_neu, "akut": p_akut, "chronic": p_chr},
                "leak_acute_given_not_present": leak, "present_equals_acute_or_chronic": ident}
            for pr, aa, cc in zip(pres.astype(int), A.astype(int), chr_.astype(int)):
                combos_all[(pr, aa, cc)] = combos_all.get((pr, aa, cc), 0) + 1
        tot = sum(combos_all.values())
        print("    combinations (present, acute=0|neu|akut, chronic) pooled over all organs:")
        for k in sorted(combos_all, key=lambda x: -combos_all[x]):
            print(f"      pres={k[0]} acute={ACUTE_LBL.get(k[1], k[1]):4s} chronic={k[2]}  "
                  f"n={combos_all[k]:>8,}  ({100*combos_all[k]/tot:5.2f}%)")
        report["organ_encoding"] = {"per_organ": org_rep,
            "combinations": {f"pres{k[0]}_ac{ACUTE_LBL.get(k[1], k[1])}_chr{k[2]}": v
                             for k, v in combos_all.items()}}

    # ---- 12. second-order Markov on patient group: P(g_{t+1} | g_t, g_{t-1}) ----
    if grp_g2 is not None and (day_in >= 0).any():
        print(f"\n## 12. Second-order Markov on {feats[grp_g2]} — does the day-before-yesterday add info?")
        pos = {}
        for i in range(N):
            if day_in[i] >= 0 and min_[i, grp_g2] > 0 and mout[i, grp_g2] > 0:
                pos[(stay[i], int(day_in[i]))] = i
        trip = []                                              # (g_{t-1}, g_t, g_{t+1})
        for (st, dd), i in pos.items():
            pv = pos.get((st, dd - 1))
            if pv is not None:
                trip.append((int(in_cls[pv, grp_g2]), int(in_cls[i, grp_g2]), int(out_cls[i, grp_g2])))
        if trip:
            trip = np.array(trip)
            k = int(trip.max()) + 1
            F = np.zeros((k, k)); S = np.zeros((k, k, k))
            for a, b, c in trip:
                F[b, c] += 1; S[a, b, c] += 1
            n_tot = len(trip)
            h1 = sum(F[b].sum() / n_tot * _entropy_bits(F[b]) for b in range(k) if F[b].sum())
            h2 = sum(S[a, b].sum() / n_tot * _entropy_bits(S[a, b])
                     for a in range(k) for b in range(k) if S[a, b].sum())
            print(f"    {n_tot:,} triplets  H(g+1|g)={h1:.3f}bit  H(g+1|g,g-1)={h2:.3f}bit  "
                  f"info-gain from lag-2 = {h1-h2:.3f}bit")
            print("    P(g_{t+1} | g_{t-1}, g_t)   [vs 1st-order P(g+1|g_t) in brackets]:")
            ctx = {}
            for b in range(k):                                 # g_t
                fo = F[b] / max(F[b].sum(), 1)
                for a in range(k):                             # g_{t-1}
                    nrow = S[a, b].sum()
                    if nrow < 30:
                        continue
                    pr = S[a, b] / nrow
                    print(f"      g-1={a} g={b} -> " + " ".join(f"{pr[c]:.2f}" for c in range(k))
                          + f"   (n={int(nrow):>5,}; 1st-order: " + " ".join(f"{fo[c]:.2f}" for c in range(k)) + ")")
                    ctx[f"gm1{a}_g{b}"] = {"n": int(nrow), "p_next": pr.round(4).tolist()}
            report["patient_group_markov2"] = {"feature": feats[grp_g2], "n_triplets": n_tot,
                "H_first_order_bits": h1, "H_second_order_bits": h2, "info_gain_lag2_bits": h1 - h2,
                "first_order": (F / np.maximum(F.sum(1, keepdims=True), 1)).round(4).tolist(),
                "second_order_context": ctx}

    # ---- 13. two-conditional Markov: P(group_{t+1} | group_t, X_t), X = another same-day feature ----
    #          (predict tomorrow's target from TWO of today's inputs: the target itself + a covariate;
    #           NOT a 2nd-order/lag-2 model — this is what the UMM meeting actually asked for.)
    if grp_g2 is not None:
        conds = [("i03000", "Arbeitsdiagnose", {0: "Weder", 1: "SIRS", 2: "Sepsis",
                                                 3: "schwSepsis", 4: "septSchock"}),
                 ("i04000", "Infektion",       {0: "Nein", 1: "Ja", 2: "Gesichert"}),
                 ("i02000", "Gesamtbild",      {0: "verbess", 1: "unveraend", 2: "verschl"})]
        m2 = {}; hdr_done = False
        for cid, cname, clab in conds:
            cg = next((g for g in range(G) if cid in feats[g]), None)
            if cg is None:
                continue
            v = (min_[:, grp_g2] > 0) & (min_[:, cg] > 0) & (mout[:, grp_g2] > 0)
            if v.sum() < 100:
                continue
            gt = in_cls[v, grp_g2]; xt = in_cls[v, cg]; gn = out_cls[v, grp_g2]
            kt = int(max(gt.max(), gn.max())) + 1; kx = int(xt.max()) + 1
            F = np.zeros((kt, kt)); S = np.zeros((kt, kx, kt))
            for a, c in zip(gt, gn):
                F[a, c] += 1
            for a, b, c in zip(gt, xt, gn):
                S[a, b, c] += 1
            n = len(gt)
            h1 = sum(F[a].sum() / n * _entropy_bits(F[a]) for a in range(kt) if F[a].sum())
            h2 = sum(S[a, b].sum() / n * _entropy_bits(S[a, b])
                     for a in range(kt) for b in range(kx) if S[a, b].sum())
            if not hdr_done:
                print("\n## 13. Two-conditional Markov: P(group_{t+1} | group_t, X_t)  "
                      "[predict tomorrow from TWO of today's features]")
                hdr_done = True
            print(f"  X = {cid}_{cname}:  H(g+1|g)={h1:.3f}bit  H(g+1|g,X)={h2:.3f}bit  "
                  f"info-gain from X = {h1-h2:.3f}bit  (n={n:,})")
            ctx = {}
            for a in range(kt):                                # group_t
                fo = F[a] / max(F[a].sum(), 1)
                for b in range(kx):                            # X_t
                    nrow = S[a, b].sum()
                    if nrow < 50:
                        continue
                    pr = S[a, b] / nrow
                    print(f"      g={a} X={clab.get(b, b):10s}-> " + " ".join(f"{pr[c]:.2f}" for c in range(kt))
                          + f"  (n={int(nrow):>5,}; base g->: " + " ".join(f"{fo[c]:.2f}" for c in range(kt)) + ")")
                    ctx[f"g{a}_{cid}{b}"] = {"n": int(nrow), "p_next": pr.round(4).tolist()}
            m2[cid] = {"feature": feats[cg], "H_group_only_bits": h1, "H_group_and_X_bits": h2,
                       "info_gain_bits": h1 - h2, "n": n, "contexts": ctx}
        if m2:
            report["patient_group_two_conditional"] = m2

    # ---- 14. infection source-site combinations (i05100) + organ-dysfunction cause (Ursache) ----
    loc_idx = [g for g in range(G) if "i05100" in feats[g] or "Lokalisation" in feats[g]]
    if loc_idx:
        from collections import Counter
        print("\n## 14. Infection source-site combinations (i05100) + organ cause (Ursache)")
        loc_names = [feats[g].split("__")[-1] for g in loc_idx]
        v = np.ones(N, bool)
        for g in loc_idx:
            v &= (min_[:, g] > 0)
        rows = np.where(v)[0]
        sites = in_cls[np.ix_(rows, loc_idx)] > 0                     # (n, n_loc) bool
        nsite = sites.sum(axis=1)
        print(f"    n_obs={len(rows):,}  sites/obs: mean={nsite.mean():.2f}  "
              f"%0={100*(nsite==0).mean():.1f} %1={100*(nsite==1).mean():.1f} "
              f"%2={100*(nsite==2).mean():.1f} %>=3={100*(nsite>=3).mean():.1f}")
        combo = Counter(tuple(loc_names[j] for j in range(len(loc_idx)) if r[j]) for r in sites)
        tot = sum(combo.values())
        print("    top site combinations:")
        for key, c in combo.most_common(12):
            lbl = "(no site)" if not key else "+".join(key)
            print(f"      {lbl[:54]:55s} n={c:>7,} ({100*c/tot:5.2f}%)")
        sec = {"n_obs": len(rows), "mean_sites": float(nsite.mean()),
               "n_sites_pct": {k: float((nsite == k).mean()) for k in (0, 1, 2)},
               "n_sites_ge3_pct": float((nsite >= 3).mean()),
               "top_combinations": {("+".join(k) or "none"): int(c) for k, c in combo.most_common(15)}}
        # link site-count to infection level (i04000) and diagnosis severity (i03000)
        sec["mean_sites_by"] = {}
        for cid in ("i04000", "i03000"):
            cg = next((g for g in range(G) if cid in feats[g]), None)
            if cg is None:
                continue
            vc = v & (min_[:, cg] > 0)
            ns_full = (in_cls[np.ix_(np.where(vc)[0], loc_idx)] > 0).sum(axis=1)
            lvf = in_cls[vc, cg]
            by = {int(k): round(float(ns_full[lvf == k].mean()), 3) for k in range(int(lvf.max()) + 1)
                  if (lvf == k).any()}
            print(f"    mean #sites by {cid}: " + "  ".join(f"{k}:{x}" for k, x in by.items()))
            sec["mean_sites_by"][cid] = by
        # organ-dysfunction cause (Ursache | present)
        URS = {0: "none", 1: "nicht-inf", 2: "unklar", 3: "infekt"}
        if organs:
            print("    organ-dysfunction cause (Ursache | organ present):")
            org_urs = {}
            for org, ix in organs.items():
                if not {"cause", "present"} <= set(ix):
                    continue
                ug, pg = ix["cause"], ix["present"]
                jv = (min_[:, ug] > 0) & (min_[:, pg] > 0) & (in_cls[:, pg] == 1)
                if jv.sum() == 0:
                    continue
                u = in_cls[jv, ug]
                dist = {URS[k]: float((u == k).mean()) for k in range(4)}
                print(f"      {org[:15]:16s} n={int(jv.sum()):>6,}  "
                      + "  ".join(f"{lab}={p*100:4.1f}%" for lab, p in dist.items()))
                org_urs[org] = {"n_present": int(jv.sum()), "cause_dist": dist}
            sec["organ_cause"] = org_urs
        report["infection_sites"] = sec

    # ---- 15. annotation coverage / informative missingness + ward-relativity of the group (GTSQ) ----
    if grp_g2 is not None and (weekday >= 0).any():
        print("\n## 15. Annotation coverage / ranking-only days + ward-relativity of the patient group")
        cov = (min_ > 0).sum(axis=1)                        # GTSQ items observed on the input day
        grp_obs = min_[:, grp_g2] > 0
        ranking_only = grp_obs & (cov <= 2)                 # group filled, ~nothing else
        print(f"    coverage/row: mean={cov.mean():.1f} median={np.median(cov):.0f}  "
              f"%full(>=40 items)={100*(cov>=40).mean():.1f}  "
              f"%ranking-only(grp & <=2 items)={100*ranking_only.mean():.1f}")
        sec = {"coverage_mean": float(cov.mean()), "pct_full_ge40": float((cov >= 40).mean()),
               "pct_ranking_only": float(ranking_only.mean())}
        wdm = (weekday >= 0) & (weekday <= 4); we = weekday >= 5
        if we.any():
            sec["ranking_only_weekday"] = float(ranking_only[wdm].mean())
            sec["ranking_only_weekend"] = float(ranking_only[we].mean())
            print(f"    ranking-only rate: weekday={100*sec['ranking_only_weekday']:.2f}%  "
                  f"weekend={100*sec['ranking_only_weekend']:.2f}%")
        vc = ~np.isnat(cal)
        df = pd.DataFrame({"date": cal[vc].astype("datetime64[D]"), "grp": in_cls[vc, grp_g2],
                           "grp_obs": grp_obs[vc], "low": ranking_only[vc]})
        date_low = df.groupby("date")["low"].mean()
        print(f"    per-date ranking-only fraction (is missingness a whole-day decision?): "
              f"%dates>=80%low={100*(date_low >= 0.8).mean():.1f}  "
              f"%dates<=20%low={100*(date_low <= 0.2).mean():.1f}  (n_dates={len(date_low):,})")
        sec["dates_ge80pct_low"] = float((date_low >= 0.8).mean())
        sec["dates_le20pct_low"] = float((date_low <= 0.2).mean())
        # ward relativity: per calendar day, ward size + #stable(group0) / #critical(group2)
        g = df[df.grp_obs]
        comp = g.groupby("date").agg(n=("grp", "size"), n0=("grp", lambda s: int((s == 0).sum())),
                                     n2=("grp", lambda s: int((s == 2).sum())))
        comp = comp[comp.n >= 5]                             # days with a meaningful ward present
        if len(comp) > 10:
            cv0 = comp.n0.std() / max(comp.n0.mean(), 1e-9); cv2 = comp.n2.std() / max(comp.n2.mean(), 1e-9)
            r0 = float(np.corrcoef(comp.n, comp.n0)[0, 1]); r2 = float(np.corrcoef(comp.n, comp.n2)[0, 1])
            print(f"    ward/day (>=5 incl. pts, n_days={len(comp):,}): size mean={comp.n.mean():.1f}  "
                  f"#stable mean={comp.n0.mean():.2f}(CV{cv0:.2f})  #critical mean={comp.n2.mean():.2f}(CV{cv2:.2f})")
            print(f"    corr(ward size, #extreme): stable r={r0:+.2f}  critical r={r2:+.2f}  "
                  f"[r~0 => fixed-count relative ranking; r~1 => fixed-fraction of ward]")
            sec.update({"ward_size_mean": float(comp.n.mean()), "n_stable_mean": float(comp.n0.mean()),
                        "n_critical_mean": float(comp.n2.mean()), "corr_size_stable": r0,
                        "corr_size_critical": r2})
        report["annotation_coverage"] = sec

    # ---- 16. two-conditional markov-gate feasibility: does a 2nd conditional add confident overrides? ----
    TAU, MINN = 0.9, 50
    ctx_feats = [g for g in range(G) if any(c in feats[g] for c in ("i01000", "i02000", "i03000", "i04000"))]
    if ctx_feats:
        print(f"\n## 16. Two-conditional markov-GATE feasibility (tau={TAU}, min n={MINN}): does conditioning"
              f"\n     on a 2nd feature add confident overrides the first-order gate misses?")
        feas = {}
        for cg in ctx_feats:
            new_cells = 0; cov_pairs = 0; helped = []
            for g in range(G):
                if g == cg:
                    continue
                v = (min_[:, g] > 0) & (min_[:, cg] > 0) & (mout[:, g] > 0)
                if v.sum() < MINN:
                    continue
                gt = in_cls[v, g]; xt = in_cls[v, cg]; gn = out_cls[v, g]
                kt = int(max(gt.max(), gn.max())) + 1; kx = int(xt.max()) + 1
                fo = {}                                       # first-order override target per class
                for c in range(kt):
                    cnt = np.bincount(gn[gt == c], minlength=kt)
                    if cnt.sum() and cnt.max() / cnt.sum() >= TAU and int(cnt.argmax()) != c:
                        fo[c] = int(cnt.argmax())
                for c in range(kt):
                    for x in range(kx):
                        m = (gt == c) & (xt == x)
                        if m.sum() < MINN:
                            continue
                        cnt = np.bincount(gn[m], minlength=kt)
                        tgt = int(cnt.argmax())
                        if cnt.max() / cnt.sum() >= TAU and tgt != c and fo.get(c, c) != tgt:
                            new_cells += 1; cov_pairs += int(m.sum())
                            if feats[g] not in helped:
                                helped.append(feats[g])
            print(f"  ctx={feats[cg][:26]:27s}: {new_cells:3d} new confident 2-cond overrides, "
                  f"{len(helped)} features, {cov_pairs:,} day-pairs affected")
            if helped:
                print("       -> " + ", ".join(h.replace('i11100_', '').split('_')[0] for h in helped[:10]))
            feas[feats[cg]] = {"new_override_cells": new_cells, "n_features_helped": len(helped),
                               "day_pairs_affected": cov_pairs, "features": helped}
        report["two_conditional_gate_feasibility"] = feas

    out_path = os.path.join(args.data_dir, "data_analysis.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {out_path}\n")

    if not args.no_figures:
        render_change_histogram(report, args.figdir)
        if "patient_group" in report:
            render_sankey(report, args.figdir, by_diagnosis=False)
            render_sankey(report, args.figdir, by_diagnosis=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--figdir", default="docs/figures/keepgate",
                   help="where to write the patient-group Sankey PNG/PDF")
    p.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    main(p.parse_args())
