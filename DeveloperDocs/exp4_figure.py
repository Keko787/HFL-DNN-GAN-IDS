"""Experiment 4 rebuttal figure: where the integrated stack pays for itself.

Left  - final model AUC after the federated session.
Right - update yield (aggregated updates per round).
X axis runs benign (left) to severe (right); the crossover is the finding.
Grayscale-safe: distinct hatches, markers, and line styles.

Uncertainty: bars are the MEAN with a percentile bootstrap 95% CI of the mean --
the same estimator reported in the methodology tables (sec 5/6). Per-seed points
are overlaid because the per-seed final_auc distribution is BIMODAL: a trial
either trains (~0.98) or receives no aggregated update and stays at the untrained
init (~0.25). A symmetric +/-1 SD whisker would imply a unimodal spread and would
extend above AUC = 1.0, which is impossible.
"""
import csv, os, random, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = r"D:/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS/results/exp4_paper/h0h1_all.csv"
OUT = os.environ.get("EXP4_FIG_OUT",
                     r"D:/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS/results/exp4_paper/fig_exp4_crossover.png")

BOOT_N = 10000
BOOT_SEED = 20240611          # fixed so the figure is byte-reproducible
JITTER_SEED = 7               # fixed so the strip overlay is reproducible

rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8")) if r.get("status") == "ok"]


def vals(arm, metric, regime, dz=None):
    out = []
    for r in rows:
        if r["arm"] != arm or r["param_regime"] != regime:
            continue
        if dz is not None and r["param_dead_zone"] != dz:
            continue
        try:
            out.append(float(r[metric]))
        except (ValueError, KeyError):
            pass
    return out


def boot_ci(v, n=BOOT_N, seed=BOOT_SEED):
    """Percentile bootstrap 95% CI of the mean.

    Every resample mean lies in [min(v), max(v)], so for a metric bounded in
    [0,1] the interval is bounded in [0,1] BY CONSTRUCTION -- no clipping.
    """
    m = st.mean(v)
    if len(v) < 2:
        return m, m, m
    rng = random.Random(seed)
    k = len(v)
    means = [sum(v[rng.randrange(k)] for _ in range(k)) / k for _ in range(n)]
    means.sort()
    return m, means[int(0.025 * n)], means[int(0.975 * n)]


# Benign -> severe.
CONDS = [("Clean\nlink", "clean", None),
         ("0.0", "jittery", "0.0"),
         ("0.2", "jittery", "0.2"),
         ("0.4", "jittery", "0.4"),
         ("0.6", "jittery", "0.6")]

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
x = np.arange(len(CONDS))
w = 0.36
ARMS = [("H0", "H0: flat FL (no mule)", -w / 2, "white", "///", "o", "0.20"),
        ("H1", "H1: mule + scheduler + hierarchical FL (fixed band)", w / 2, "0.55", "...", "x", "black")]

drawn_max = {}   # panel -> highest y any artist reaches (for the assertion below)

for ax, metric, ylabel, title in [
    (axes[0], "final_auc", "Final model AUC", "Model quality after the session"),
    (axes[1], "update_yield", "Updates aggregated / round", "Update yield"),
]:
    top = 0.0
    for arm, label, off, color, hatch, marker, mcol in ARMS:
        series = [vals(arm, metric, rg, dz) for _, rg, dz in CONDS]
        stats = [boot_ci(v) for v in series]
        m = [s[0] for s in stats]
        lo = [s[0] - s[1] for s in stats]      # asymmetric: distance down to CI lo
        hi = [s[2] - s[0] for s in stats]      # asymmetric: distance up to CI hi
        ax.bar(x + off, m, w, yerr=[lo, hi], capsize=3, label=label,
               color=color, edgecolor="black", hatch=hatch, linewidth=1.1,
               error_kw=dict(ecolor="0.15", lw=1.2), zorder=2)
        # Per-seed points: the actual distribution, incl. the collapse mode.
        rng = random.Random(JITTER_SEED)
        for xi, v in zip(x, series):
            jx = [xi + off + (rng.random() - 0.5) * w * 0.78 for _ in v]
            kw = (dict(facecolors="none", edgecolors=mcol) if marker == "o"
                  else dict(color=mcol))
            ax.scatter(jx, v, s=7, marker=marker, linewidths=0.6,
                       alpha=0.45, zorder=3, **kw)
        top = max([top] + [s[2] for s in stats] + [max(v) for v in series if v])

    ax.axvline(0.5, color="black", ls="--", lw=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in CONDS])
    ax.set_xlabel("Degraded link: terrain dead-zone fraction  →  more severe")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", ls="--", alpha=0.45)
    ax.set_axisbelow(True)
    drawn_max[ylabel] = top

# --- AUC panel: a [0,1] metric gets a [0,1] axis. No impossible headroom. ---
axes[0].set_ylim(0.0, 1.0)
# Both arms start from the SAME seeded init theta (Exp4Driver.theta_seed is
# fixed), so this line is one particular untrained network, not an average over
# random inits. Its AUC (~0.27) is not a skill measure: the untrained net is a
# near-constant predictor (outputs span ~0.4995-0.5052), and AUC being rank-based
# magnifies that sub-1% tilt. Sweeping the init seed gives mean 0.41 (sd 0.20),
# i.e. chance-centred. Label it as the starting point, not as an achieved score.
init = st.mean(vals("H1", "init_auc", "jittery"))
axes[0].axhline(init, color="black", ls=":", lw=1.5, zorder=1,
                label="untrained init θ₀ — trials that received no aggregated update "
                      "(one fixed random init; ≈chance in expectation)")
axes[0].axhline(0.5, color="0.45", ls=(0, (1, 3)), lw=1.0, zorder=1)
axes[0].annotate("chance (0.50)", xy=(len(CONDS) - 0.52, 0.515), fontsize=7.4,
                 color="0.35", style="italic")

# Collapse counts: state the bimodality in numbers, do not hide it.
for xi, (_, rg, dz) in zip(x, CONDS):
    for arm, _, off, _, _, _, _ in ARMS:
        v = vals(arm, "final_auc", rg, dz)
        c = sum(1 for y in v if y < 0.5)
        if c:
            axes[0].annotate(f"{c}/{len(v)}", xy=(xi + off, 0.155), ha="center",
                             fontsize=7.0, color="0.10",
                             bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))

axes[1].set_ylim(0.0, max(4.2, drawn_max["Updates aggregated / round"] * 1.06))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.2,
           frameon=False, bbox_to_anchor=(0.5, -0.005))

fig.suptitle(
    "Experiment 4 (integrated, real model): the mule costs under a clean link and "
    "rescues the session under severe blockage",
    fontsize=11.5, y=0.985)
fig.text(0.5, 0.088,
         "Bars = mean; whiskers = percentile bootstrap 95% CI of the mean "
         "(same estimator as the paired tables). Points = individual seeds; "
         "n/N labels count seeds whose session received no aggregated update, "
         "leaving θ at its untrained init.",
         ha="center", fontsize=7.6, color="0.30")
fig.tight_layout(rect=[0, 0.115, 1, 0.935])
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("wrote", OUT)

# ---- Guard: nothing on the AUC panel may reach above 1.0 ----
worst = drawn_max["Final model AUC"]
assert worst <= 1.0, f"AUC panel draws to {worst} > 1.0"
print(f"AUC panel highest drawn value = {worst:.4f} (<= 1.0), ylim = {axes[0].get_ylim()}")

# Numbers backing the caption.
for label, rg, dz in [("clean", "clean", None), ("dz=0.6", "jittery", "0.6")]:
    a0, a1 = vals("H0", "final_auc", rg, dz), vals("H1", "final_auc", rg, dz)
    y0, y1 = vals("H0", "update_yield", rg, dz), vals("H1", "update_yield", rg, dz)
    c0 = sum(1 for v in a0 if v < 0.5); c1 = sum(1 for v in a1 if v < 0.5)
    print(f"{label:8s} AUC  H0={st.mean(a0):.3f} H1={st.mean(a1):.3f} | "
          f"yield H0={st.mean(y0):.3f} H1={st.mean(y1):.3f} | n={len(a1)} | "
          f"collapsed H0={c0} H1={c1}")
ia = vals("H1", "init_auc", "jittery"); fa = vals("H1", "final_auc", "jittery")
ac = vals("H1", "final_accuracy", "jittery")
print(f"convergence H1 jittery: init_auc={st.mean(ia):.3f} -> final_auc={st.mean(fa):.3f}, "
      f"final_acc={st.mean(ac):.3f}, n={len(fa)}")
for label, rg, dz in CONDS:
    for arm in ("H0", "H1"):
        v = vals(arm, "final_auc", rg, dz)
        m, lo, hi = boot_ci(v)
        print(f"  CI {label.replace(chr(10),' '):11s} {arm} n={len(v):3d} "
              f"mean={m:.4f} 95%CI=[{lo:.4f},{hi:.4f}] "
              f"(mean+SD would be {m + st.pstdev(v):.4f})")
