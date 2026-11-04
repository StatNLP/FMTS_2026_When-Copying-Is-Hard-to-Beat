"""Departures from persistence, from DIRECT paired counts.

Replaces the algebraic reconstruction that the first draft used (OPEN_ITEMS O16). That version
inverted Equation 2 on rounded summaries, which cannot distinguish a departure that fixes a cell
from one that was wrong before and after -- both leave accuracy lower than the identity implies.
This reads the per-cell prediction files written by

    python exp_fixnn/eval_deploy.py fmts --split_name test --bootstrap 2000

and counts, on the common evaluation mask, for every predictor:

    fixed   persistence wrong, model right   (the reason to depart)
    broken  persistence right, model wrong   (the cost)
    futile  wrong before and after           (neither, but not free -- it is still a changed field
                                              put in front of a clinician)

Usage:  python make_figure.py [path/to/fmts_eval]
"""
from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
EVAL = Path(ARGS[0]) if ARGS else ROOT / "fmts_eval"

if "--schematic" in sys.argv:
    # The poster needs one picture of the MECHANISM, not of the result: the form opens
    # pre-filled with yesterday's value, so the clinician confirms most rows and changes one
    # or two. Generated rather than AI-illustrated, so the pre-fill, the carry-over arrows
    # and the palette are exact and no text can come out garbled.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    TEAL, GHOST, CORAL, LINE, INK = "#0D9DAF", "#B9DFE5", "#E8604C", "#CFE4E8", "#15303A"
    N = 9
    CHANGED = {3, 6}                      # two of nine rows differ -- the 3.81 % made visible
    fig, ax = plt.subplots(figsize=(5.4, 2.75))
    ax.set_xlim(0, 100); ax.set_ylim(2.5, 55); ax.axis("off")

    def form(x0, title, prefilled):
        ax.add_patch(FancyBboxPatch((x0, 4), 34, 38, boxstyle="round,pad=0.6,rounding_size=1.6",
                                    fc="white", ec=LINE, lw=1.6, zorder=2))
        ax.text(x0 + 17, 43.6, title, ha="center", va="bottom", fontsize=17,
                color=INK, fontweight="bold")
        for i in range(N):
            y = 38.2 - i * 3.75
            ax.add_patch(plt.Rectangle((x0 + 3, y - .75), 15.5, 1.5, fc=LINE, ec="none", zorder=3))
            changed = prefilled and i in CHANGED
            col = CORAL if changed else (GHOST if prefilled else TEAL)
            ax.add_patch(plt.Rectangle((x0 + 24.5, y - 1.15), 6.2, 2.3, fc=col, ec="none", zorder=3))
            if prefilled and not changed:      # a suggestion the clinician accepted, left as is
                ax.add_patch(plt.Rectangle((x0 + 24.5, y - 1.15), 6.2, 2.3, fc="none",
                                           ec=TEAL, lw=.8, ls=(0, (2, 1.6)), zorder=4))
        return [38.2 - i * 3.75 for i in range(N)]

    ys = form(6, "yesterday", False)
    form(60, "today", True)
    for i, y in enumerate(ys):
        ax.add_patch(FancyArrowPatch((41.5, y), (58.5, y),
                                     arrowstyle="-|>,head_width=1.5,head_length=2.6",
                                     color="#9DBCC4", lw=1.1, zorder=1))
    ax.text(50, 49.2, "every value is carried over as a suggestion", ha="center", va="bottom",
            fontsize=14, color="#516470")
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "carry_forward.pdf")
    fig.savefig(ROOT / "figures" / "carry_forward.png", dpi=260)
    print("wrote figures/carry_forward.{pdf,png}")
    raise SystemExit
DEP = EVAL / "fmts_departures.csv"
if not DEP.exists():
    raise SystemExit(f"{DEP} not found -- run `eval_deploy.py fmts` first. This script no longer "
                     f"reconstructs counts from rounded point estimates.")

rows = [r for r in csv.DictReader(DEP.open()) if r["predictor"] != "Persistence"]
order = ["Persistence + rule", "Transition matrix", "Gate over persistence",
         "Gate over rule, macro-F1 selected", "Change-weighted objective",
         "GRU without baseline gate"]
rows.sort(key=lambda r: order.index(r["predictor"]) if r["predictor"] in order else 99)
short = {"Gate over rule, macro-F1 selected": "Gate over rule",
         "GRU without baseline gate": "GRU without gate",
         "Change-weighted objective": "Change-weighted"}

n = int(float(rows[0]["n_scored"]))
lab = [short.get(r["predictor"], r["predictor"]) for r in rows]
fixed = np.array([1000 * float(r["fixed"]) / float(r["n_scored"]) for r in rows])
broken = np.array([1000 * float(r["broken"]) / float(r["n_scored"]) for r in rows])
futile = np.array([1000 * float(r["futile"]) / float(r["n_scored"]) for r in rows])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
# One shared axis cannot serve this data: the dynamic range is 70x (0.0 to 37.3), so the four
# predictors that matter most -- the rule, the transition matrix and the two gates, all under
# 7 per 1,000 -- collapse into the leftmost tenth of the plot, which is exactly where the
# interesting comparison lives (rule 1.3/0.0 against gate-over-rule 1.7/2.6). So: a 2x2 of the
# four low-departure predictors on their own shared scale, and the two ungated models below on
# theirs, with the scale change stated on the axis rather than hidden.
GOOD, BAD, NEUTRAL = "#0E9F6E", "#E8604C", "#F3C969"
SERIES = (("corrected", GOOD), ("broken", BAD), ("futile", NEUTRAL))
vals = {l: (f, b, c) for l, f, b, c in zip(lab, fixed, broken, futile)}
low  = ["Persistence + rule", "Transition matrix", "Gate over persistence", "Gate over rule"]
high = ["GRU without gate", "Change-weighted"]

fig = plt.figure(figsize=(5.6, 3.25))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=.85, wspace=.28,
                      left=.085, right=.985, top=.87, bottom=.075)

def panel(ax, names, xmax, show_names):
    y = np.arange(len(names))
    h = .26
    for k, (name_, col) in enumerate(SERIES):
        v = [vals[n][k] for n in names]
        ax.barh(y + (k - 1) * h, v, height=h, color=col, zorder=3,
                label=name_ if show_names else None)
        for yi, vi in zip(y, v):
            ax.text(vi + xmax * .015, yi + (k - 1) * h, f"{vi:.1f}", va="center",
                    fontsize=6.8, color=col, zorder=4)
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#C7D0D5")
    ax.tick_params(axis="y", length=0, colors="#2A3B44")
    ax.tick_params(axis="x", length=3, color="#C7D0D5", labelsize=7.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E8ECEF", linewidth=.6)

LO = 8.0
for idx, name in enumerate(low):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    panel(ax, [name], LO, idx == 0)
    ax.set_yticks([])
    ax.set_title(name, fontsize=8.5, color="#2A3B44", pad=3, loc="left")

# the two ungated models get their own row and their own scale; titled like the others so no
# y-axis label can clip
for idx, name in enumerate(high):
    ax = fig.add_subplot(gs[2, idx])
    panel(ax, [name], 42.0, False)
    ax.set_yticks([])
    ax.set_title(name, fontsize=8.5, color="#2A3B44", pad=3, loc="left")
# No text line under the panels: the caption states the unit (departures per 1,000 scored cells)
# and the scale break of the bottom row, so a second statement in the figure only costs height.

h_, l_ = fig.axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper left", bbox_to_anchor=(.085, .985), ncol=3, frameon=False,
           fontsize=8.5, handlelength=1.1, columnspacing=1.8)
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "departures.pdf")
fig.savefig(ROOT / "figures" / "departures.png", dpi=240)
print(f"figure from direct counts in {DEP}")
for r, f_, b_, c_ in zip(rows, fixed, broken, futile):
    print(f"  {r['predictor']:38s} fixed {f_:6.2f}  broken {b_:6.2f}  futile {c_:6.2f} /1000")
