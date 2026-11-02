#!/usr/bin/env python3
"""WARD-AWARE predictor for the patient-group item (the one place persistence is structurally blind).

The GTSQ patient-group question ranks a patient against the OTHER patients currently on the unit,
with a hard capacity: "die DREI am schwersten Erkrankten" / "die DREI am wenigsten schwer". Verified
(analyse_ward.py): the 3-slot cap holds on 100% of ward-days, median ward 17 beds. Encoding is
monotone: 0 = three LEAST severe, 1 = neither extreme, 2 = three MOST severe.

Two things follow that a per-patient model cannot represent:
  (a) a patient's group can change because ANOTHER patient changed  -> exogenous;
  (b) if the 3 slots are already taken, a fourth patient cannot enter that group -> hard constraint.

This script tests whether ward context actually BUYS anything, honestly:
  * strictly causal features only -- ward composition on day D (never day D+1);
  * fit on TRAIN encounters, evaluated on TEST encounters (same 42/…/ split logic as the pipeline:
    split by encounter, never by row, so a patient never appears in both);
  * compared against persistence AND against a same-features model without the ward block, so any
    gain is attributable to ward context rather than to "a model was fitted at all";
  * additionally a CONSTRAINED variant that forbids predicting group 2 when 3 slots are already
    occupied by other patients (uses only day-D occupancy, so it stays causal).

Usage:  python exp_fixnn/ward_predictor.py
"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "gtsq_day_level_encoded_fixnn.csv")
GRP, DX, IMP, INF = ("i01000_Gruppe_(0|1|2)", "i03000_Arbeitsdiagnose",
                     "i02000_Klinisches_Gesamtbild", "i04000_Infektionsverdacht")
ST = "i01000_Station"
SEED = 42


def build():
    d = pd.read_csv(CSV, low_memory=False)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d[ST].notna() & d[GRP].notna()].sort_values(["EncounterNumber", "date"])

    # ---- ward composition on day D (causal: same day as the input) ----
    occ = d.groupby([ST, "date"]).agg(
        ward_n=(GRP, "size"),
        n_top=(GRP, lambda s: int((s == 2).sum())),
        n_bot=(GRP, lambda s: int((s == 0).sum())),
        ward_mean_dx=(DX, "mean")).reset_index()
    d = d.merge(occ, on=[ST, "date"], how="left")
    # the patient's OWN contribution removed -> "how many OTHER patients hold each slot"
    d["others_top"] = d["n_top"] - (d[GRP] == 2).astype(int)
    d["others_bot"] = d["n_bot"] - (d[GRP] == 0).astype(int)
    d["top_full"] = (d["others_top"] >= 3).astype(int)     # no free "most severe" slot
    d["bot_full"] = (d["others_bot"] >= 3).astype(int)
    d["dx_rank"] = d.groupby([ST, "date"])[DX].rank(pct=True)          # severity rank in the unit
    d["dx_minus_ward"] = d[DX] - d["ward_mean_dx"]

    # ---- next-day target (consecutive days only) ----
    nxt = d.groupby("EncounterNumber")[GRP].shift(-1)
    nd = d.groupby("EncounterNumber")["date"].shift(-1)
    keep = (nd - d["date"]).dt.days.eq(1) & nxt.notna()
    p = d[keep].copy(); p["y"] = nxt[keep].astype(int)
    return p


def main():
    p = build()
    rng = np.random.RandomState(SEED)
    encs = p["EncounterNumber"].unique(); rng.shuffle(encs)
    te = set(encs[:int(0.2 * len(encs))])                  # split by ENCOUNTER, never by row
    tr = p[~p["EncounterNumber"].isin(te)]; ts = p[p["EncounterNumber"].isin(te)]
    print(f"day-pairs {len(p):,} | train {len(tr):,} test {len(ts):,} "
          f"({len(encs):,} encounters, 20% held out)")

    PAT = [GRP, DX, IMP, INF]                                            # patient-only
    WARD = ["ward_n", "others_top", "others_bot", "top_full", "bot_full",
            "dx_rank", "dx_minus_ward", "ward_mean_dx"]                  # ward context (day D)

    def run(cols, tag, constrain=False):
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_depth=4,
                                           l2_regularization=1.0, early_stopping=False,
                                           random_state=SEED)
        m.fit(tr[cols].astype(float), tr["y"])
        pr = m.predict_proba(ts[cols].astype(float)); cls = m.classes_
        pred = cls[pr.argmax(1)]
        if constrain:   # (b) cannot enter a full slot -- uses day-D occupancy only, still causal
            blk_top = ts["top_full"].to_numpy() == 1
            blk_bot = ts["bot_full"].to_numpy() == 1
            q = pr.copy()
            for k, c in enumerate(cls):
                if c == 2: q[blk_top, k] = -1
                if c == 0: q[blk_bot, k] = -1
            pred = cls[q.argmax(1)]
        y = ts["y"].to_numpy()
        return (pred == y).mean(), f1_score(y, pred, average="macro", labels=[0, 1, 2],
                                            zero_division=0)

    base = ts[GRP].to_numpy().astype(int); y = ts["y"].to_numpy()
    pacc, pf1 = (base == y).mean(), f1_score(y, base, average="macro", labels=[0, 1, 2],
                                             zero_division=0)
    print(f"\n{'predictor':38s}{'acc':>9}{'macro-F1':>10}")
    print(f"{'persistence (copy group)':38s}{pacc:>9.4f}{pf1:>10.4f}")
    for cols, tag, con in [(PAT, "patient-only GBT", False),
                           (PAT + WARD, "patient + WARD context", False),
                           (PAT + WARD, "patient + WARD + 3-slot constraint", True)]:
        a, f = run(cols, tag, con)
        print(f"{tag:38s}{a:>9.4f}{f:>10.4f}   ({a-pacc:+.4f} / {f-pf1:+.4f} vs persistence)")
    print("\n  gain of ward context = (patient+WARD) - (patient-only); anything else is just 'a model'.")
    print(f"  test day-pairs where a slot is already full: top {int(ts['top_full'].sum())}, "
          f"bot {int(ts['bot_full'].sum())} of {len(ts)}")


if __name__ == "__main__":
    main()
