"""Analyze exp4 integrated results. Pure stdlib; Cliff's delta via midranks."""
import csv, statistics as st, collections, os

D = r"D:/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS/results/exp4_paper"


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
    if not n or not m: return float('nan')
    r = midranks(list(a)+list(b)); U = sum(r[:n]) - n*(n+1)/2.0
    return 2.0*U/(n*m) - 1.0


def lab(d):
    x = abs(d)
    return "negligible" if x < .147 else "small" if x < .33 else "medium" if x < .474 else "large"


def load(f):
    rows = list(csv.DictReader(open(os.path.join(D, f), encoding='utf-8')))
    return [x for x in rows if x.get('status') == 'ok']


def num(rows, key):
    out = []
    for x in rows:
        try: out.append(float(x[key]))
        except (ValueError, KeyError, TypeError): pass
    return out


def sel(rows, **kw):
    return [x for x in rows if all(x.get(k) == v for k, v in kw.items())]


A = load('h0h1_all.csv')
print(f"h0h1_all: {len(A)} ok rows\n")

METRICS = ['final_auc', 'delta_auc', 'final_accuracy', 'update_yield',
           'mission_completion_rate', 'round_close_rate_kmin1', 'coverage']

print("=" * 78)
print("H1 (integrated mule stack) vs H0 (flat FL), by dead zone  [jittery]")
print("=" * 78)
print(f"  {'dz':>5} {'metric':<26} {'H0':>9} {'H1':>9} {'delta':>8}  label      n")
for dz in ['0.0', '0.2', '0.4', '0.6']:
    for m in ['final_auc', 'delta_auc', 'update_yield', 'round_close_rate_kmin1']:
        h0 = num(sel(A, arm='H0', param_regime='jittery', param_dead_zone=dz), m)
        h1 = num(sel(A, arm='H1', param_regime='jittery', param_dead_zone=dz), m)
        if not h0 or not h1: continue
        d = cliffs(h1, h0)
        print(f"  {dz:>5} {m:<26} {st.mean(h0):9.4f} {st.mean(h1):9.4f} {d:+8.3f}  {lab(d):<10} {len(h1)}")
    print()

print("=" * 78)
print("CLEAN regime (does the mule cost anything when the network is fine?)")
print("=" * 78)
for m in ['final_auc', 'delta_auc', 'update_yield', 'round_close_rate_kmin1']:
    h0 = num(sel(A, arm='H0', param_regime='clean'), m)
    h1 = num(sel(A, arm='H1', param_regime='clean'), m)
    if not h0 or not h1: continue
    d = cliffs(h1, h0)
    print(f"  {m:<26} H0={st.mean(h0):.4f}  H1={st.mean(h1):.4f}  delta={d:+.3f} ({lab(d)})  n={len(h1)}")

print()
print("=" * 78)
print("Model convergence: did the real model actually learn? (all jittery)")
print("=" * 78)
for arm in ['H0', 'H1']:
    rows = sel(A, arm=arm, param_regime='jittery')
    ia, fa = num(rows, 'init_auc'), num(rows, 'final_auc')
    acc = num(rows, 'final_accuracy'); re_ = num(rows, 'rounds_evaluated')
    if not fa: continue
    print(f"  {arm}: init_auc={st.mean(ia):.4f} -> final_auc={st.mean(fa):.4f} "
          f"(best={max(fa):.4f})  final_acc={st.mean(acc):.4f}  rounds_eval={st.mean(re_):.1f}  n={len(fa)}")

print()
print("=" * 78)
print("By link quality (jittery, dz=0.0)")
print("=" * 78)
for lq in ['0.3', '0.5', '0.7']:
    h0 = num(sel(A, arm='H0', param_regime='jittery', param_link_quality=lq, param_dead_zone='0.0'), 'final_auc')
    h1 = num(sel(A, arm='H1', param_regime='jittery', param_link_quality=lq, param_dead_zone='0.0'), 'final_auc')
    if not h0 or not h1: continue
    d = cliffs(h1, h0)
    print(f"  lq={lq}: H0 final_auc={st.mean(h0):.4f}  H1={st.mean(h1):.4f}  delta={d:+.3f} ({lab(d)})  n={len(h1)}")

# ---- H2 / H3 shard ----
print()
print("=" * 78)
print("h2h3_l1.csv — layer ablation (H1 det. / H2 +RL selector / H3 +adaptive L1)")
print("=" * 78)
B = load('h2h3_l1.csv')
print(f"rows: {len(B)}   arms: {collections.Counter(x['arm'] for x in B)}")
print(f"regimes: {collections.Counter(x['param_regime'] for x in B)}")
print(f"dead_zone: {collections.Counter(x['param_dead_zone'] for x in B)}")
print(f"link_quality: {collections.Counter(x['param_link_quality'] for x in B)}")
print()
arms = sorted({x['arm'] for x in B})
for m in ['final_auc', 'delta_auc', 'update_yield', 'round_close_rate_kmin1', 'mission_completion_rate']:
    line = f"  {m:<26}"
    for a in arms:
        v = num(sel(B, arm=a), m)
        if v: line += f"  {a}={st.mean(v):.4f}"
    print(line)
print()
for i in range(len(arms)-1):
    for j in range(i+1, len(arms)):
        a, b = arms[j], arms[i]
        for m in ['final_auc', 'update_yield', 'round_close_rate_kmin1']:
            va, vb = num(sel(B, arm=a), m), num(sel(B, arm=b), m)
            if not va or not vb: continue
            d = cliffs(va, vb)
            print(f"  {a} vs {b:<3} {m:<26} {st.mean(vb):.4f} -> {st.mean(va):.4f}  delta={d:+.3f} ({lab(d)})")
        print()
