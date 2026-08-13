"""Experiment 4, figure 1: where the integrated stack pays for itself.

H0 (flat FL, no mule) vs H1 (mule + gated scheduler + two-pass hierarchical FL),
across a benign→severe sweep of the terrain dead-zone. The crossover is the
finding: the mule costs under a clean link and rescues the session under
blockage.

Three panels, all metrics normalized to [0,1] (see
experiments/analysis/figstyle.py), with every axis shared with figure 2 so the
two can be read side by side:

  1. Final model AUC          — bars, zoomed to AUC_ZOOM_YLIM w/ axis break
  2. Participation            — updates/round ÷ N devices, full [0,1] bars
  3. Paired H1−H0 difference  — bootstrap 95% CIs on the shared difference axis

**The AUC panel is zoomed, and marked as such.** On a full [0,1] axis the AUC
differences (0.01–0.12) are invisible. Bars are kept for consistency with the
rest of the figure set, but a bar encodes magnitude by its length *from zero*,
so on a truncated axis those lengths are not proportional to the values — the
classic way a chart overstates a difference. Two mitigations, both required:
the ``//`` break marks on the axis declare the truncation, and panel 3 carries
the actual effect sizes with confidence intervals, so no conclusion rests on
comparing truncated bar heights.

**Why the zoom floor is 0.60, not 0.80.** It is the tightest window that still
contains every mean and every bootstrap CI bound in *both* figures (the lowest
is 0.657). A 0.80 floor would have cropped a bar mean of 0.7997 and four CI
bounds — hiding real results, which is the failure this figure already had once.

Per-seed points are overlaid because final_auc is bimodal: a session either
trains (~0.98) or receives no aggregated update and stays at the untrained init
(~0.25). Those below-window sessions are counted as ↓n rather than dropped. A
symmetric ±1 SD whisker is never used: it implies a unimodal spread that does
not exist and extends above AUC = 1.0.

Grayscale-safe: hatches, greys and distinct markers only.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This script lives in DeveloperDocs/, so put the repo root on sys.path — the
# figure must use the same estimator as the tables, not a reimplementation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analysis.exp4 import analyze_metric, load_exp4  # noqa: E402
from experiments.analysis.figstyle import (  # noqa: E402
    AUC_ZOOM_YLIM, DIFF_YLIM, METRIC_YLIM, assert_within_zoom, draw_diff_panel,
    draw_metric_panel,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "results", "exp4_paper", "h0h1_all.csv")
OUT = os.environ.get(
    "EXP4_FIG_OUT", os.path.join(REPO, "results", "exp4_paper", "fig_exp4_crossover.png"))

BASELINE, TREATMENT = "H0", "H1"
ARM_LABELS = ["H0: flat FL (no mule)",
              "H1: mule + scheduler + hierarchical FL"]

df = load_exp4(SRC)
N_DEVICES = int(pd.to_numeric(df["param_N"], errors="coerce").dropna().iloc[0])
# Normalize the one raw-count metric onto [0,1] so every panel shares a scale.
df["participation"] = pd.to_numeric(df["update_yield"], errors="coerce") / N_DEVICES

CONDS = [("Clean\nlink", "clean", None),
         ("0.0", "jittery", 0.0),
         ("0.2", "jittery", 0.2),
         ("0.4", "jittery", 0.4),
         ("0.6", "jittery", 0.6)]
XLABEL = "Degraded link: terrain dead-zone fraction  →  more severe"


def subset(regime, dz):
    d = df[df["param_regime"].astype(str) == regime]
    if dz is not None:
        d = d[np.isclose(pd.to_numeric(d["param_dead_zone"], errors="coerce"), dz)]
    return d


def vals(regime, dz, arm, metric):
    d = subset(regime, dz)
    return pd.to_numeric(d[d["arm"] == arm][metric], errors="coerce").dropna().tolist()


fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.4))
x = np.arange(len(CONDS))
labels = [c[0] for c in CONDS]
drawn = {}

# ---- Panel 1: AUC as bars on the shared ZOOMED axis ---------------------- #
auc_series = [[vals(rg, dz, arm, "final_auc") for _, rg, dz in CONDS]
              for arm in (BASELINE, TREATMENT)]
zoom = draw_metric_panel(
    axes[0], x, labels, auc_series, ARM_LABELS,
    ylabel="Final model AUC", title="Model quality after the session",
    ylim=AUC_ZOOM_YLIM, divider_after=1,
)
axes[0].set_xlabel(XLABEL, fontsize=9.5)
assert_within_zoom(zoom["min"], zoom["max"])

# ---- Panel 2: participation, full [0,1] (bars are honest from a zero base) - #
part_series = [[vals(rg, dz, arm, "participation") for _, rg, dz in CONDS]
               for arm in (BASELINE, TREATMENT)]
PART_LABEL = f"Participation (updates/round ÷ {N_DEVICES})"
drawn[PART_LABEL] = draw_metric_panel(
    axes[1], x, labels, part_series, ARM_LABELS,
    ylabel=PART_LABEL, title="Devices contributing per round", divider_after=1,
)
axes[1].set_xlabel(XLABEL, fontsize=9.5)

# ---- Panel 3: the paired difference, on the shared difference axis -------- #
diff_results = []
for metric in ("final_auc", "participation"):
    diff_results.append([
        analyze_metric(subset(rg, dz), rg, metric,
                       treatment=TREATMENT, baseline=BASELINE)
        for _, rg, dz in CONDS
    ])
draw_diff_panel(
    axes[2], x, labels, diff_results, ["Δ final model AUC", "Δ participation"],
    ylabel=f"Paired difference  ({TREATMENT} − {BASELINE})",
    title="Effect of the mule (95% CI)", divider_after=1,
)
axes[2].set_xlabel(XLABEL, fontsize=9.5)

# Panels 1 and 2 share arm labels, so dedupe by label to keep one legend row.
handles, lab = [], []
for a in axes:
    for h, l in zip(*a.get_legend_handles_labels()):
        if l not in lab:
            handles.append(h)
            lab.append(l)
fig.legend(handles, lab, loc="lower center", ncol=5, fontsize=8.0,
           frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.suptitle(
    "Experiment 4 (integrated, real model): the mule costs under a clean link and "
    "rescues the session under severe blockage", fontsize=11.5, y=0.985)
fig.text(0.5, 0.085,
         f"Metrics normalized to [0,1] (participation = updates/round ÷ {N_DEVICES} devices). "
         f"AUC panel is zoomed to {AUC_ZOOM_YLIM[0]:.2f}–{AUC_ZOOM_YLIM[1]:.2f} — shared with "
         f"figure 2; the // break marks the truncated axis, so bar lengths there are NOT "
         f"proportional to magnitude — read the whiskers (bootstrap 95% CI) and the difference "
         f"panel. ↓n counts sessions that received no aggregated update and sit at the "
         f"untrained init, below the window. "
         f"Right panel also shares its axis with figure 2; * = CI excludes 0 and p<0.05.",
         ha="center", fontsize=7.2, color="0.30")
fig.tight_layout(rect=[0, 0.115, 1, 0.935])
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("wrote", OUT)

# ---- Guards: nothing may be drawn outside a normalized metric's bounds ---- #
for ylabel, res in drawn.items():
    top = res["max"]
    assert top <= METRIC_YLIM[1] + 1e-9, f"{ylabel} panel draws to {top} > {METRIC_YLIM[1]}"
    print(f"  {ylabel}: highest drawn {top:.4f} <= {METRIC_YLIM[1]}")
for results in diff_results:
    for r in results:
        if r is None:
            continue
        assert DIFF_YLIM[0] <= r.ci_lo and r.ci_hi <= DIFF_YLIM[1], (
            f"difference CI [{r.ci_lo:.4f},{r.ci_hi:.4f}] falls outside the shared "
            f"axis {DIFF_YLIM} — widen DIFF_YLIM in figstyle.py")
print(f"  difference panel: all CIs inside the shared axis {DIFF_YLIM}")

# ---- Numbers backing the caption ---- #
print(f"\n{'cond':11s} {'metric':16s} {'n':>3s} {BASELINE:>7s} {TREATMENT:>7s} "
      f"{'diff':>8s} {'95% CI':>20s} {'p':>7s}  verdict")
for (lbl, rg, dz), *_ in zip(CONDS):
    for metric in ("final_auc", "participation"):
        r = analyze_metric(subset(rg, dz), rg, metric,
                           treatment=TREATMENT, baseline=BASELINE)
        if r is None:
            continue
        ci = f"[{r.ci_lo:+.4f},{r.ci_hi:+.4f}]"
        print(f"{lbl.replace(chr(10), ' '):11s} {metric:16s} {r.n_pairs:3d} "
              f"{r.mean_base:7.4f} {r.mean_treat:7.4f} {r.mean_diff:+8.4f} "
              f"{ci:>20s} {r.p_value:7.4f}  {r.verdict}")
