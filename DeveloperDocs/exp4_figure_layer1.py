"""Figure 2: isolating the adaptive radio layer (L1).

H2 (fixed best-average band) vs H3 (adaptive band via the U(c,t) controller).
Same selector, same seeds, same channel trace; the ONLY difference is whether
the mule re-selects its band per mission.

Left  - final model AUC by blockage level, both arms.
Right - the PAIRED H3-H2 difference with its bootstrap 95% CI, per level.
Grayscale-safe: hatches + greys only.

Design notes (each fixes a defect found in the first version of this figure):

* **Same estimator as the tables.** Every number here comes from
  ``experiments.analysis.exp4`` — the module that produces the methodology
  tables — so the figure cannot disagree with the text. The first version
  computed Cliff's delta *unpaired* over unequal-n groups while the paper's
  protocol is paired, which flipped the sign of the clean cell.
* **Uncertainty is shown.** The first version drew effect sizes with no error
  bars under a title asserting they were positive everywhere; with CIs added,
  most straddle zero. The right panel now plots the paired difference with its
  CI and a zero line, so a tie is visible as a tie.
* **Bounded axes.** AUC is drawn on [0,1]; a symmetric +/-1 SD whisker (the old
  approach) implied a spread that does not exist and extended above AUC = 1.0,
  which is impossible. Per-seed points are overlaid because final_auc is
  bimodal: a session either trains or receives no aggregated update and stays
  at the untrained initialisation.
* **Excluded trials are reported, not silently dropped.** Any non-``ok`` row
  (including ``no_eval`` — a session that produced no model at all) is counted
  and printed rather than swallowed by a bare ``except``.

Data: results/exp4_paper/h2h3_dz_*.csv — the clean re-run of the H2/H3
dead-zone sweep. (The earlier h2h3_m_*.csv sweep is superseded: 22/190 of its
``ok`` rows had produced no model evaluation at all, which biased the
participation means.)
"""
import glob
import os
import random
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This script lives in DeveloperDocs/, so put the repo root on sys.path to
# import the analysis module — the figure must use the same estimator as the
# tables, not a reimplementation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analysis.exp4 import analyze_metric, load_exp4  # noqa: E402

D = r"D:/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS/results/exp4_paper"
OUT = os.environ.get("EXP4_L1_OUT", os.path.join(D, "fig_exp4_layer1.png"))
GLOB = os.path.join(D, "h2h3_dz_*.csv")

BOOT_SEED = 20240611
JITTER_SEED = 7
METRIC = "final_auc"

# ---- Load, and account for every row we drop --------------------------------
frames, raw_n, kept_n = [], 0, 0
for f in sorted(glob.glob(GLOB)):
    full = pd.read_csv(f)
    ok = load_exp4(f)
    raw_n += len(full)
    kept_n += len(ok)
    frames.append(ok)
if not frames:
    raise SystemExit(f"no input CSVs matched {GLOB}")
df = pd.concat(frames, ignore_index=True)
excluded = raw_n - kept_n
print(f"loaded {kept_n} ok rows from {raw_n} total ({excluded} excluded as non-ok)")


def boot_ci(v, n=10000, seed=BOOT_SEED):
    """Percentile bootstrap 95% CI of the mean.

    Every resample mean lies in [min(v), max(v)], so for a metric bounded in
    [0,1] the interval is bounded in [0,1] BY CONSTRUCTION — no clipping.
    """
    v = list(v)
    m = st.mean(v)
    if len(v) < 2:
        return m, m, m
    rng = random.Random(seed)
    k = len(v)
    means = sorted(sum(v[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return m, means[int(0.025 * n)], means[int(0.975 * n)]


CONDS = [("Clean\nlink", "clean", None),
         ("0.0", "jittery", 0.0),
         ("0.2", "jittery", 0.2),
         ("0.4", "jittery", 0.4),
         ("0.6", "jittery", 0.6)]


def subset(regime, dz):
    d = df[df["param_regime"].astype(str) == regime]
    if dz is not None:
        d = d[np.isclose(pd.to_numeric(d["param_dead_zone"], errors="coerce"), dz)]
    return d


def arm_vals(regime, dz, arm):
    d = subset(regime, dz)
    return pd.to_numeric(d[d["arm"] == arm][METRIC], errors="coerce").dropna().tolist()


fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
x = np.arange(len(CONDS))
w = 0.36
ARMS = [("H2", "H2: fixed band", -w / 2, "white", "///", "o", "0.20"),
        ("H3", "H3: adaptive band (L1)", w / 2, "0.55", "...", "x", "black")]

# ---- Left panel: per-arm AUC, mean + bootstrap CI + per-seed points ---------
ax = axes[0]
auc_top = 0.0
for arm, label, off, color, hatch, marker, mcol in ARMS:
    series = [arm_vals(rg, dz, arm) for _, rg, dz in CONDS]
    stats = [boot_ci(v) if v else (np.nan, np.nan, np.nan) for v in series]
    m = [s[0] for s in stats]
    lo = [s[0] - s[1] for s in stats]
    hi = [s[2] - s[0] for s in stats]
    ax.bar(x + off, m, w, yerr=[lo, hi], capsize=3, label=label,
           color=color, edgecolor="black", hatch=hatch, linewidth=1.1,
           error_kw=dict(ecolor="0.15", lw=1.2), zorder=2)
    rng = random.Random(JITTER_SEED)
    for xi, v in zip(x, series):
        jx = [xi + off + (rng.random() - 0.5) * w * 0.78 for _ in v]
        kw = (dict(facecolors="none", edgecolors=mcol) if marker == "o"
              else dict(color=mcol))
        ax.scatter(jx, v, s=9, marker=marker, linewidths=0.7, alpha=0.55,
                   zorder=3, **kw)
    auc_top = max([auc_top] + [s[2] for s in stats if s[2] == s[2]]
                  + [max(v) for v in series if v])

ax.axvline(0.5, color="black", ls="--", lw=1.4, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in CONDS])
ax.set_xlabel("Degraded link: terrain dead-zone fraction  →  more severe")
ax.set_ylabel("Final model AUC")
ax.set_title("Model quality by band policy", fontsize=11)
ax.set_ylim(0.0, 1.0)          # a [0,1] metric on a [0,1] axis
init_vals = pd.to_numeric(df["init_auc"], errors="coerce").dropna()
if len(init_vals):
    ax.axhline(init_vals.mean(), color="black", ls=":", lw=1.5, zorder=1,
               label="untrained init θ₀ — sessions that received no aggregated update")
ax.axhline(0.5, color="0.45", ls=(0, (1, 3)), lw=1.0, zorder=1)
ax.annotate("chance (0.50)", xy=(len(CONDS) - 0.55, 0.515), fontsize=7.4,
            color="0.35", style="italic")
# Name the collapse mode in numbers rather than hiding it in a fat whisker.
for xi, (_, rg, dz) in zip(x, CONDS):
    for arm, _, off, _, _, _, _ in ARMS:
        v = arm_vals(rg, dz, arm)
        c = sum(1 for y in v if y < 0.5)
        if c:
            ax.annotate(f"{c}/{len(v)}", xy=(xi + off, 0.145), ha="center",
                        fontsize=7.0, color="0.10",
                        bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))
ax.grid(axis="y", ls="--", alpha=0.45)
ax.set_axisbelow(True)

# ---- Right panel: the PAIRED difference + bootstrap CI (the table's numbers) -
ax = axes[1]
diffs, los, his, ps, deltas, ns = [], [], [], [], [], []
for _, rg, dz in CONDS:
    r = analyze_metric(subset(rg, dz), rg, METRIC, treatment="H3", baseline="H2")
    if r is None:
        diffs.append(np.nan); los.append(0); his.append(0)
        ps.append(np.nan); deltas.append(np.nan); ns.append(0)
        continue
    diffs.append(r.mean_diff)
    los.append(r.mean_diff - r.ci_lo)
    his.append(r.ci_hi - r.mean_diff)
    ps.append(r.p_value)
    deltas.append(r.cliffs_delta)
    ns.append(r.n_pairs)

ax.bar(x, diffs, w * 1.5, yerr=[los, his], capsize=3,
       color="0.55", edgecolor="black", hatch="...", linewidth=1.1,
       error_kw=dict(ecolor="0.15", lw=1.2), zorder=2,
       label="paired H3 − H2 (mean difference)")
ax.axhline(0.0, color="black", lw=1.2, zorder=1)
ax.axvline(0.5, color="black", ls="--", lw=1.4, zorder=1)
for xi, (d, p, n) in enumerate(zip(diffs, ps, ns)):
    if d != d:
        continue
    star = "*" if (p == p and p < 0.05) else ""
    ax.annotate(f"n={n}{star}", xy=(xi, d + (his[xi] + 0.004) * (1 if d >= 0 else -1)),
                ha="center", va="bottom" if d >= 0 else "top", fontsize=7.2,
                color="0.15")
ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in CONDS])
ax.set_xlabel("Degraded link: terrain dead-zone fraction  →  more severe")
ax.set_ylabel("Δ final model AUC  (H3 − H2)")
ax.set_title("Paired effect of adapting the band (95% CI)", fontsize=11)
ax.grid(axis="y", ls="--", alpha=0.45)
ax.set_axisbelow(True)
span = max(0.02, max((abs(d) + h) for d, h in zip(diffs, his) if d == d) * 1.35)
ax.set_ylim(-span, span)

handles, labels = [], []
for a in axes:
    h, l = a.get_legend_handles_labels()
    handles += h
    labels += l
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.0,
           frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.suptitle(
    "Isolating the radio layer: same selector, same seeds, only the band policy differs",
    fontsize=11.5, y=0.985)
fig.text(0.5, 0.088,
         "Left: bars = mean, whiskers = percentile bootstrap 95% CI of the mean; points = "
         "individual seeds; n/N = seeds whose session received no aggregated update. "
         "Right: paired H3−H2 difference (same estimator as the methodology tables); "
         "a CI crossing 0 is a tie. * = p<0.05.",
         ha="center", fontsize=7.4, color="0.30")
fig.tight_layout(rect=[0, 0.115, 1, 0.935])
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("wrote", OUT)

# ---- Guard: an AUC panel may never draw above 1.0 ----
assert auc_top <= 1.0, f"AUC panel draws to {auc_top} > 1.0"
print(f"AUC panel highest drawn value = {auc_top:.4f} (<= 1.0), ylim = {axes[0].get_ylim()}")

# ---- Numbers backing the caption ----
print(f"{'cond':11s} {'n':>3s} {'H2':>7s} {'H3':>7s} {'H3-H2':>8s} "
      f"{'95% CI':>20s} {'p':>7s} {'delta':>6s}  verdict")
for (lbl, rg, dz), n in zip(CONDS, ns):
    r = analyze_metric(subset(rg, dz), rg, METRIC, treatment="H3", baseline="H2")
    if r is None:
        continue
    ci = f"[{r.ci_lo:+.4f},{r.ci_hi:+.4f}]"
    print(f"{lbl.replace(chr(10), ' '):11s} {r.n_pairs:3d} {r.mean_base:7.4f} "
          f"{r.mean_treat:7.4f} {r.mean_diff:+8.4f} {ci:>20s} {r.p_value:7.4f} "
          f"{r.cliffs_delta:+6.2f}  {r.verdict}")
