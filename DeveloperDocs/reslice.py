"""Re-slice the SUBMITTED exp3 dataset (legacy exp3/exp3_v8.csv) by regime and
budget tightness. Pure stdlib: Cliff's delta via midranks, O(n log n).

delta = P(a>b) - P(a<b), computed as 2*U_a/(n_a*n_b) - 1 with midrank U.
Romano thresholds: <0.147 negligible, <0.33 small, <0.474 medium, else large.
"""
import csv, statistics as st, sys, collections

SRC = r"D:/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS/results/legacy exp3/exp3_v8.csv"


def midranks(vals):
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # ranks are 1-based
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def cliffs_delta(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    r = midranks(list(a) + list(b))
    Ra = sum(r[:n])
    Ua = Ra - n * (n + 1) / 2.0
    return 2.0 * Ua / (n * m) - 1.0


def label(d):
    a = abs(d)
    return ("negligible" if a < 0.147 else
            "small" if a < 0.33 else
            "medium" if a < 0.474 else "large")


rows = list(csv.DictReader(open(SRC, newline="", encoding="utf-8")))
print(f"source: {SRC}\nrows: {len(rows)}  cells: {len({r['cell_id'] for r in rows})}")
print(f"beta levels: {sorted({r['param_beta'] for r in rows})}")
print(f"N levels: {sorted({r['param_N'] for r in rows})}  rrf: {sorted({r['param_rrf'] for r in rows})}")
print(f"jittery: {collections.Counter(r['param_jittery'] for r in rows)}")

METRICS = [
    ("mission_completion_rate", "mission completion", +1),
    ("update_yield", "update yield", +1),
    ("round_close_rate_kmin2", "quorum close (k=2)", +1),
    ("jains_fairness", "Jain fairness", +1),
    ("propulsion_energy_J", "propulsion energy (lower better)", -1),
]


def col(rs, arm, metric, **filt):
    out = []
    for r in rs:
        if r["arm"] != arm:
            continue
        if any(r[k] != v for k, v in filt.items()):
            continue
        try:
            out.append(float(r[metric]))
        except (ValueError, KeyError):
            pass
    return out


def block(title, filt):
    print(f"\n{'='*74}\n{title}\n{'='*74}")
    for metric, nice, direction in METRICS:
        line = f"  {nice:34s}"
        for a, b in (("A4", "A3"), ("A4", "A2"), ("A4", "A1"), ("A3", "A2")):
            xa, xb = col(rows, a, metric, **filt), col(rows, b, metric, **filt)
            if not xa or not xb:
                continue
            d = cliffs_delta(xa, xb)
            line += f"  {a}v{b}={d:+.3f}({label(d)[:4]})"
        print(line)


block("SUBMITTED DATA - POOLED (what the paper reports)", {})
block("CLEAN regime only", {"param_jittery": "False"})
block("JITTERY regime only", {"param_jittery": "True"})

print(f"\n{'='*74}\nA4 vs A3 mission completion, BY BUDGET TIGHTNESS (beta) x REGIME\n{'='*74}")
print(f"  {'beta':>6} {'regime':>8} {'A4 mean':>9} {'A3 mean':>9} {'delta':>8}  {'label':<11} n")
for beta in sorted({r["param_beta"] for r in rows}, key=float):
    for jit in ("False", "True"):
        f = {"param_beta": beta, "param_jittery": jit}
        xa = col(rows, "A4", "mission_completion_rate", **f)
        xb = col(rows, "A3", "mission_completion_rate", **f)
        d = cliffs_delta(xa, xb)
        reg = "clean" if jit == "False" else "jittery"
        print(f"  {beta:>6} {reg:>8} {st.mean(xa):9.3f} {st.mean(xb):9.3f} {d:+8.3f}  {label(d):<11} {len(xa)}")

print(f"\n{'='*74}\nA4 vs A1 round participation (update_yield), BY REGIME\n{'='*74}")
for jit in ("False", "True"):
    f = {"param_jittery": jit}
    xa = col(rows, "A4", "update_yield", **f)
    xb = col(rows, "A1", "update_yield", **f)
    d = cliffs_delta(xa, xb)
    reg = "clean" if jit == "False" else "jittery"
    print(f"  {reg:>8}: A4={st.mean(xa):.3f}  A1={st.mean(xb):.3f}  delta={d:+.3f} ({label(d)})  n={len(xa)}")

print(f"\n{'='*74}\nWORST-QUARTILE check: A4 vs A3 mission completion in the hardest cells\n{'='*74}")
for jit in ("False", "True"):
    f = {"param_jittery": jit}
    xa = sorted(col(rows, "A4", "mission_completion_rate", **f))
    xb = sorted(col(rows, "A3", "mission_completion_rate", **f))
    q = max(1, len(xa) // 4)
    reg = "clean" if jit == "False" else "jittery"
    print(f"  {reg:>8}: bottom-quartile mean  A4={st.mean(xa[:q]):.3f}  A3={st.mean(xb[:q]):.3f}"
          f"   full-sample  A4={st.mean(xa):.3f}  A3={st.mean(xb):.3f}")
