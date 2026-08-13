"""A4 vs the naive baseline, and dead-zone sensitivity (submitted 80% vs 7_21 60%)."""
import csv, statistics as st

SUB = r"D:/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS/results/legacy exp3/exp3_v8.csv"
NEW = r"D:/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS/results/exp3_7_21/exp3.csv"


def midranks(vals):
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def cliffs_delta(a, b):
    n, m = len(a), len(b)
    if not n or not m:
        return float("nan")
    r = midranks(list(a) + list(b))
    Ua = sum(r[:n]) - n * (n + 1) / 2.0
    return 2.0 * Ua / (n * m) - 1.0


def label(d):
    a = abs(d)
    return ("negligible" if a < 0.147 else "small" if a < 0.33
            else "medium" if a < 0.474 else "large")


def load(p):
    return list(csv.DictReader(open(p, newline="", encoding="utf-8")))


def col(rs, arm, m, **f):
    return [float(r[m]) for r in rs
            if r["arm"] == arm and all(r[k] == v for k, v in f.items()) and r.get(m, "") != ""]


sub, new = load(SUB), load(NEW)

print("A4 vs A2 (naive arrival-order baseline), mission completion, by beta x regime")
print(f"  {'beta':>6} {'regime':>8} {'A4':>7} {'A2':>7} {'delta':>8}  label")
for b in ["0.25", "0.5", "1.0", "2.0"]:
    for j in ["False", "True"]:
        f = {"param_beta": b, "param_jittery": j}
        xa, xb = col(sub, "A4", "mission_completion_rate", **f), col(sub, "A2", "mission_completion_rate", **f)
        d = cliffs_delta(xa, xb)
        reg = "clean" if j == "False" else "jittery"
        print(f"  {b:>6} {reg:>8} {st.mean(xa):7.3f} {st.mean(xb):7.3f} {d:+8.3f}  {label(d)}")

print("\nA4 vs A2 update yield, by beta x regime")
print(f"  {'beta':>6} {'regime':>8} {'A4':>7} {'A2':>7} {'delta':>8}  label")
for b in ["0.25", "0.5", "1.0", "2.0"]:
    for j in ["False", "True"]:
        f = {"param_beta": b, "param_jittery": j}
        xa, xb = col(sub, "A4", "update_yield", **f), col(sub, "A2", "update_yield", **f)
        d = cliffs_delta(xa, xb)
        reg = "clean" if j == "False" else "jittery"
        print(f"  {b:>6} {reg:>8} {st.mean(xa):7.3f} {st.mean(xb):7.3f} {d:+8.3f}  {label(d)}")

print("\n" + "=" * 74)
print("DEAD-ZONE SENSITIVITY: submitted v8 (80%) vs 7_21 run (60% default), jittery only")
print("=" * 74)
for m in ["mission_completion_rate", "update_yield"]:
    print(f"  -- {m}")
    for tag, rs in (("submitted (80%)", sub), ("7_21 run (60%)", new)):
        a1 = col(rs, "A1", m, param_jittery="True")
        a4 = col(rs, "A4", m, param_jittery="True")
        d = cliffs_delta(a4, a1)
        print(f"     {tag:17s} A1={st.mean(a1):6.3f}  A4={st.mean(a4):6.3f}  "
              f"A4vA1 delta={d:+.3f} ({label(d)})")

print("\n  sanity: do the two datasets agree on the MULE arms (which the dead zone")
print("          does not touch)?  If yes, only A1 moved.")
for arm in ["A2", "A3", "A4"]:
    s = st.mean(col(sub, arm, "mission_completion_rate", param_jittery="True"))
    n = st.mean(col(new, arm, "mission_completion_rate", param_jittery="True"))
    print(f"     {arm} jittery MCR: submitted={s:.3f}  7_21={n:.3f}  diff={n-s:+.4f}")
