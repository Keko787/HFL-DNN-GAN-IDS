"""Experiment 4 rebuttal figure: where the integrated stack pays for itself.

Left  - final model AUC after the federated session.
Right - update yield (aggregated updates per round).
X axis runs benign (left) to severe (right); the crossover is the finding.
Grayscale-safe: distinct hatches, markers, and line styles.
"""
import csv, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = r"D:/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS/results/exp4_paper/h0h1_all.csv"
OUT = r"D:/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS/results/exp4_paper/fig_exp4_crossover.png"

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


# Benign -> severe.
CONDS = [("Clean\nlink", "clean", None),
         ("0.0", "jittery", "0.0"),
         ("0.2", "jittery", "0.2"),
         ("0.4", "jittery", "0.4"),
         ("0.6", "jittery", "0.6")]

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
x = np.arange(len(CONDS))
w = 0.36

for ax, metric, ylabel, title in [
    (axes[0], "final_auc", "Final model AUC", "Model quality after the session"),
    (axes[1], "update_yield", "Updates aggregated / round", "Update yield"),
]:
    m0 = [st.mean(vals("H0", metric, rg, dz)) for _, rg, dz in CONDS]
    s0 = [st.pstdev(vals("H0", metric, rg, dz)) for _, rg, dz in CONDS]
    m1 = [st.mean(vals("H1", metric, rg, dz)) for _, rg, dz in CONDS]
    s1 = [st.pstdev(vals("H1", metric, rg, dz)) for _, rg, dz in CONDS]

    ax.bar(x - w/2, m0, w, yerr=s0, capsize=3, label="H0: flat FL (no mule)",
           color="white", edgecolor="black", hatch="///", linewidth=1.1,
           error_kw=dict(ecolor="0.35", lw=1))
    ax.bar(x + w/2, m1, w, yerr=s1, capsize=3, label="H1: mule + scheduler + hierarchical FL (fixed band)",
           color="0.55", edgecolor="black", hatch="...", linewidth=1.1,
           error_kw=dict(ecolor="0.15", lw=1))

    ax.axvline(0.5, color="black", ls="--", lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in CONDS])
    ax.set_xlabel("Degraded link: terrain dead-zone fraction  →  more severe")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", ls="--", alpha=0.45)
    ax.set_axisbelow(True)

# Headroom above the bars so the legend never sits on top of a result.
axes[0].set_ylim(0.0, 1.52)
# Both arms start from the same seeded init theta; show where training began.
# Labelled via the legend rather than an inline annotation, which overlapped bars.
init = st.mean(vals("H1", "init_auc", "jittery"))
axes[0].axhline(init, color="black", ls=":", lw=1.5,
                label=f"untrained model (AUC {init:.2f})")
axes[0].legend(loc="upper left", fontsize=8.2, framealpha=0.95)
axes[1].set_ylim(0.0, 5.6)
axes[1].legend(loc="upper left", fontsize=8.2, framealpha=0.95)

fig.suptitle(
    "Experiment 4 (integrated, real model): the mule costs under a clean link and "
    "rescues the session under severe blockage",
    fontsize=11.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=220)
print("wrote", OUT)

# Numbers backing the caption.
for label, rg, dz in [("clean", "clean", None), ("dz=0.6", "jittery", "0.6")]:
    a0, a1 = vals("H0", "final_auc", rg, dz), vals("H1", "final_auc", rg, dz)
    y0, y1 = vals("H0", "update_yield", rg, dz), vals("H1", "update_yield", rg, dz)
    print(f"{label:8s} AUC  H0={st.mean(a0):.3f} H1={st.mean(a1):.3f} | "
          f"yield H0={st.mean(y0):.3f} H1={st.mean(y1):.3f} | n={len(a1)}")
ia = vals("H1", "init_auc", "jittery"); fa = vals("H1", "final_auc", "jittery")
ac = vals("H1", "final_accuracy", "jittery")
print(f"convergence H1 jittery: init_auc={st.mean(ia):.3f} -> final_auc={st.mean(fa):.3f}, "
      f"final_acc={st.mean(ac):.3f}, n={len(fa)}")
