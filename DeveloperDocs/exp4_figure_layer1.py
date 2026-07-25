"""Figure 2: isolating the adaptive radio layer.

H2 (fixed band) vs H3 (adaptive band). Same selector, same seeds, same channel
model; the ONLY difference is whether the mule re-selects its band per mission.

Left  - final model AUC by blockage level, both arms.
Right - Cliff's delta for H3 over H2, showing the effect growing with severity.
Grayscale-safe: hatches + greys only.
"""
import csv, glob, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = r"D:/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS/results/exp4_paper"
OUT = os.path.join(D, "fig_exp4_layer1.png")

rows = []
for f in glob.glob(os.path.join(D, "h2h3_m_*.csv")):
    rows += [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["status"] == "ok"]


def midranks(v):
    idx = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
    while i < len(idx):
        j = i
        while j+1 < len(idx) and v[idx[j+1]] == v[idx[i]]: j += 1
        a = (i+j+2)/2.0
        for k in range(i, j+1): r[idx[k]] = a
        i = j+1
    return r


def cliffs(a, b):
    n, m = len(a), len(b)
    if not n or not m: return float("nan")
    r = midranks(list(a)+list(b)); U = sum(r[:n]) - n*(n+1)/2.0
    return 2.0*U/(n*m) - 1.0


def vals(arm, regime, dz, key="final_auc"):
    out = []
    for r in rows:
        if r["arm"] != arm or r["param_regime"] != regime: continue
        if dz is not None and r["param_dead_zone"] != dz: continue
        try: out.append(float(r[key]))
        except (ValueError, KeyError): pass
    return out


CONDS = [("Clean\nlink", "clean", None),
         ("0.0", "jittery", "0.0"),
         ("0.2", "jittery", "0.2"),
         ("0.4", "jittery", "0.4"),
         ("0.6", "jittery", "0.6")]

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
x = np.arange(len(CONDS)); w = 0.36

# ---- Left: AUC by arm ----
ax = axes[0]
m2 = [st.mean(vals("H2", rg, dz)) for _, rg, dz in CONDS]
s2 = [st.pstdev(vals("H2", rg, dz)) for _, rg, dz in CONDS]
m3 = [st.mean(vals("H3", rg, dz)) for _, rg, dz in CONDS]
s3 = [st.pstdev(vals("H3", rg, dz)) for _, rg, dz in CONDS]
ax.bar(x - w/2, m2, w, yerr=s2, capsize=3, label="H2: fixed band",
       color="white", edgecolor="black", hatch="///", linewidth=1.1,
       error_kw=dict(ecolor="0.35", lw=1))
ax.bar(x + w/2, m3, w, yerr=s3, capsize=3, label="H3: adaptive band (radio layer)",
       color="0.55", edgecolor="black", hatch="...", linewidth=1.1,
       error_kw=dict(ecolor="0.15", lw=1))
ax.axvline(0.5, color="black", ls="--", lw=1.4)
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in CONDS])
ax.set_xlabel("Degraded link: terrain dead-zone fraction  →  more severe")
ax.set_ylabel("Final model AUC")
ax.set_title("Adapting the band improves model quality", fontsize=11)
ax.set_ylim(0.0, 1.35)
ax.legend(loc="upper left", fontsize=8.4, framealpha=0.95)
ax.grid(axis="y", ls="--", alpha=0.45); ax.set_axisbelow(True)

# ---- Right: effect size, showing it grows with severity ----
ax = axes[1]
d_auc = [cliffs(vals("H3", rg, dz), vals("H2", rg, dz)) for _, rg, dz in CONDS]
d_yld = [cliffs(vals("H3", rg, dz, "update_yield"), vals("H2", rg, dz, "update_yield"))
         for _, rg, dz in CONDS]
ax.bar(x - w/2, d_auc, w, label="final model AUC",
       color="0.55", edgecolor="black", hatch="...", linewidth=1.1)
ax.bar(x + w/2, d_yld, w, label="update yield",
       color="white", edgecolor="black", hatch="\\\\\\", linewidth=1.1)
for thr, name in [(0.147, "small"), (0.33, "medium")]:
    ax.axhline(thr, color="0.45", ls=":", lw=1.2)
    ax.annotate(name, xy=(-0.46, thr + 0.012), fontsize=7.6, color="0.30", style="italic")
ax.axhline(0.0, color="black", lw=1.0)
ax.axvline(0.5, color="black", ls="--", lw=1.4)
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in CONDS])
ax.set_xlabel("Degraded link: terrain dead-zone fraction  →  more severe")
ax.set_ylabel("Cliff's δ, H3 over H2")
ax.set_title("Positive at every level, largest under severe blockage", fontsize=11)
ax.set_ylim(-0.18, 0.52)
ax.legend(loc="upper left", fontsize=8.4, framealpha=0.95)
ax.grid(axis="y", ls="--", alpha=0.45); ax.set_axisbelow(True)

fig.suptitle(
    "Isolating the radio layer: same selector, same seeds, only the band policy differs",
    fontsize=11.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=220)
print("wrote", OUT)

for (lbl, rg, dz), da, dy in zip(CONDS, d_auc, d_yld):
    n3, n2 = len(vals("H3", rg, dz)), len(vals("H2", rg, dz))
    print(f"  {lbl.replace(chr(10),' '):11s} AUC H2={st.mean(vals('H2',rg,dz)):.3f} "
          f"H3={st.mean(vals('H3',rg,dz)):.3f}  d_auc={da:+.3f}  d_yield={dy:+.3f}  n={n3},{n2}")
