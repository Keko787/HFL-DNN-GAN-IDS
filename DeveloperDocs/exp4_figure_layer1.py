"""Experiment 4, figure 2: isolating the adaptive radio layer (L1).

H2 (fixed best-average band) vs H3 (adaptive band via the U(c,t) controller).
Same selector, same seeds, same channel trace; the ONLY difference is whether
the mule re-selects its band per mission.

Deliberately the SAME three-panel layout and the SAME axes as figure 1
(experiments/analysis/figstyle.py), so the two can be read side by side:

  1. Final model AUC                — [0,1]
  2. Participation                  — updates/round ÷ N devices, [0,1]
  3. Paired H3−H2 difference        — bootstrap 95% CIs, zero line,
                                      on the SAME fixed axis as figure 1

That shared difference axis is the point: it shows at a glance that the L1
effect is far smaller than the mule effect of figure 1, and that every interval
here crosses zero.

Design notes (each fixes a defect found in an earlier version of this figure):

* **Same estimator as the tables.** Every number comes from
  ``experiments.analysis.exp4`` — the module behind the methodology tables — so
  figure and text cannot disagree. An earlier version computed Cliff's delta
  *unpaired* over unequal-n groups while the protocol is paired, which flipped
  the sign of the clean cell.
* **Uncertainty is shown.** An earlier version drew effect sizes with no error
  bars under a title asserting they were positive everywhere; with CIs, they all
  straddle zero.
* **Bounded axes + bootstrap CIs**, never a symmetric ±SD whisker: final_auc is
  bimodal, so ±SD implies a spread that does not exist and renders above 1.0.
* **Excluded trials are reported, not silently dropped** — any non-``ok`` row
  (including ``no_eval``: a session that produced no model) is counted and
  printed rather than swallowed by a bare ``except``.

Data: results/exp4_paper/h2h3_dz_*.csv — the clean re-run of the H2/H3
dead-zone sweep (200/200 ok, 20/20 paired seeds per cell). The earlier
h2h3_m_*.csv sweep is superseded and removed.
"""
import glob
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
    draw_metric_panel, draw_zoomed_point_panel,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "results", "exp4_paper")
OUT = os.environ.get("EXP4_L1_OUT", os.path.join(D, "fig_exp4_layer1.png"))
GLOB = os.path.join(D, "h2h3_dz_*.csv")

BASELINE, TREATMENT = "H2", "H3"
ARM_LABELS = ["H2: fixed band", "H3: adaptive band (L1)"]

# ---- Load, and account for every row we drop ----------------------------- #
frames, raw_n, kept_n = [], 0, 0
for f in sorted(glob.glob(GLOB)):
    raw_n += len(pd.read_csv(f))
    ok = load_exp4(f)
    kept_n += len(ok)
    frames.append(ok)
if not frames:
    raise SystemExit(f"no input CSVs matched {GLOB}")
df = pd.concat(frames, ignore_index=True)
print(f"loaded {kept_n} ok rows from {raw_n} total ({raw_n - kept_n} excluded as non-ok)")

N_DEVICES = int(pd.to_numeric(df["param_N"], errors="coerce").dropna().iloc[0])
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

# ---- Panel 1: AUC on the shared ZOOMED axis (dot + CI, not bars) --------- #
auc_series = [[vals(rg, dz, arm, "final_auc") for _, rg, dz in CONDS]
              for arm in (BASELINE, TREATMENT)]
zoom = draw_zoomed_point_panel(
    axes[0], x, labels, auc_series, ARM_LABELS,
    ylabel="Final model AUC", title="Model quality by band policy",
    below_axis_note="sessions at untrained init", divider_after=1,
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

# ---- Panel 3: the paired difference, on the axis shared with figure 1 ----- #
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
    title="Effect of adapting the band (95% CI)", divider_after=1,
)
axes[2].set_xlabel(XLABEL, fontsize=9.5)
# Say the null out loud, on the same axis where figure 1 shows a real effect.
axes[2].annotate("every interval crosses 0 — no detectable effect",
                 xy=(len(CONDS) / 2 - 0.5, DIFF_YLIM[0] * 0.72), ha="center",
                 fontsize=8.2, style="italic", color="0.25")

handles, lab = [], []
for a in axes:
    for h, l in zip(*a.get_legend_handles_labels()):
        if l not in lab:
            handles.append(h)
            lab.append(l)
fig.legend(handles, lab, loc="lower center", ncol=5, fontsize=8.0,
           frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.suptitle(
    "Isolating the radio layer: same selector, same seeds, only the band policy differs",
    fontsize=11.5, y=0.985)
fig.text(0.5, 0.085,
         f"Metrics normalized to [0,1] (participation = updates/round ÷ {N_DEVICES} devices). "
         f"AUC panel is zoomed to {AUC_ZOOM_YLIM[0]:.2f}–{AUC_ZOOM_YLIM[1]:.2f} — the same "
         f"window as figure 1 — and drawn as means with percentile bootstrap 95% CIs rather "
         f"than bars, since bar length is meaningless off a zero baseline; ↓n counts sessions "
         f"that received no aggregated update and sit at the untrained init, below the window. "
         f"Both the AUC and difference panels share their axes with figure 1, so the two "
         f"effect sizes are directly comparable.",
         ha="center", fontsize=7.2, color="0.30")
fig.tight_layout(rect=[0, 0.115, 1, 0.935])
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("wrote", OUT)

# ---- Guards --------------------------------------------------------------- #
for ylabel, top in drawn.items():
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
for lbl, rg, dz in CONDS:
    for metric in ("final_auc", "participation"):
        r = analyze_metric(subset(rg, dz), rg, metric,
                           treatment=TREATMENT, baseline=BASELINE)
        if r is None:
            continue
        ci = f"[{r.ci_lo:+.4f},{r.ci_hi:+.4f}]"
        print(f"{lbl.replace(chr(10), ' '):11s} {metric:16s} {r.n_pairs:3d} "
              f"{r.mean_base:7.4f} {r.mean_treat:7.4f} {r.mean_diff:+8.4f} "
              f"{ci:>20s} {r.p_value:7.4f}  {r.verdict}")
