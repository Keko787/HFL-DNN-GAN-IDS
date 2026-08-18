# Whole-scheduler SOTA comparison — results

**Run 2026-08-13.** 120 trials (60 per budget point), **0 failures, 0 duplicates**, both points
reconciled against the designed 3 arms × 20 seeds grid before analysis.

**Arms.** `H1` (ours: S3 deadline + S3b feasibility + S3.5) vs `D1` (MAX-AoI) vs `D2` (Oort's
statistical-utility selection) — each a **complete scheduler** owning its own admission decision,
under an identical budget and an identical travel-cost model. Jittery, N=6, `rrf`=60,
`n_missions`=4.

---

## Headline: the pre-registered primary metric found nothing

**Reach-rate at τ is a tie at both budgets.** This was chosen as primary *before* the run, tested
with exact McNemar on discordant pairs, and it is null:

| Budget | τ | ours | D1 | D2 | discordant | p |
|---|---|---|---|---|---|---|
| 120 s | 0.85 | 0.25 | 0.10 | 0.10 | 3–0 | 0.2500 |
| 120 s | 0.82 | 0.50 | 0.50 | 0.50 | 0–0 / 1–1 | 1.0000 |
| 60 s | 0.85 | 0.00 | 0.05 | 0.05 | 0–1 | 1.0000 |
| 60 s | 0.82 | 0.10 | 0.05 | 0.05 | 2–1 | 1.0000 |

**Reporting this first because it was the pre-registered metric.** The direction at τ=0.85/120 s
favours us (3–0 discordant), but at n=20 that is p=0.25 — not evidence.

**Why the metric failed here, which is worth knowing before reusing it.** Reach-rate needs τ to be
reachable often enough to resolve. At **60 s nothing reaches it** (0.00–0.10) because the tight
budget starves training; at **120 s the budget barely binds**, so the arms behave similarly. The
window where reach-rate discriminates sits between those, and neither point is in it. *τ must be
matched to the operating point, and the metric needs the constraint to bind without collapsing
training.*

## The interesting result: a crossover on the secondary metrics

Same trials, standard metric set, paired Wilcoxon + bootstrap CI + Cliff's δ.

### Loose budget (120 s) — the baselines win on throughput

| Comparison | Metric | Δ (baseline − ours) | CI | p | |
|---|---|---|---|---|---|
| D1 vs H1 | `update_yield` | **+0.225** | [+0.100, +0.350] | **0.0059** | ✅ D1 |
| D1 vs H1 | `mission_completion_rate` | +0.083 | [+0.033, +0.142] | 0.0160 | D1 |
| D2 vs H1 | `update_yield` | **+0.200** | [+0.088, +0.325] | **0.0098** | ✅ D2 |
| D2 vs H1 | `mission_completion_rate` | +0.058 | [+0.008, +0.108] | 0.0408 | D2 |
| D1 vs H1 | `final_accuracy` | −0.014 | [−0.034, −0.002] | 0.0619 | (ours, n.s.) |

### Tight budget (60 s) — we win on model quality

| Comparison | Metric | Δ (baseline − ours) | CI | p | |
|---|---|---|---|---|---|
| D1 vs H1 | `final_accuracy` | **−0.074** | [−0.141, −0.014] | 0.0329 | ours |
| D2 vs H1 | `final_accuracy` | **−0.089** | [−0.169, −0.019] | 0.0342 | ours |
| D1 vs H1 | `final_auc` | −0.096 | [−0.195, −0.004] | 0.0619 | (ours, n.s.) |
| D2 vs H1 | `final_auc` | −0.098 | [−0.200, −0.006] | 0.0712 | (ours, n.s.) |

> **⚠ Multiplicity, stated plainly.** 5 metrics × 2 budgets × 2 comparisons = **20 tests**. At
> Bonferroni over all 20 (α=0.0025) **nothing survives**. At α=0.01 per 5-metric family, only
> `update_yield` at 120 s survives (p=0.0059, 0.0098) — *a baseline win*. **Every "we win" cell above
> is suggestive, not established.**

## Why the crossover is worth believing anyway — the mechanism is visible

Admission behaviour, measured **per mission** from the traces (not inferred from outcomes):

| | 120 s | 60 s |
|---|---|---|
| missions where served sets differ across arms | 35/80 (**44 %**) | 68/80 (**85 %**) |
| mean devices served per mission — D1 | 5.64 | 4.28 |
| — D2 | 5.79 | 4.72 |
| — **H1 (ours)** | **5.36** (fewest) | **4.58** (more than D1) |

**The ordering flips.** Under a loose budget our gate is the most restrictive and serves fewest; under
a tight one it serves *more* than MAX-AoI. That is the mechanism behind the accuracy crossover:
MAX-AoI flies to the stalest device wherever it is, spending budget on travel, while our gate's
feasibility check refuses stops it cannot reach in time and spends the same budget on more of them.

**Selectivity is a cost when the constraint is loose and an advantage when it is tight.** That is a
coherent, mechanistically grounded story rather than a pattern read off noisy means — and it is
consistent across *both* independent baselines and several metrics, which is what argues against
pure chance despite the multiplicity problem.

## What can honestly be said

* ✅ **The comparison is real, not vacuous.** Unlike the retired sweep B, the arms differ
  substantially — admission binds on 85 % of missions at the tight budget.
* ✅ **Under a loose budget the baselines collect more.** `update_yield` survives per-family
  correction. Our gate's selectivity costs throughput when the constraint does not bind — the same
  shape as the H1-vs-H0 clean-regime result, and it should be reported with the same candour.
* ⚠ **Under a tight budget we produce better models** — accuracy +0.074/+0.089, p≈0.033. Suggestive;
  does not survive multiplicity. **More seeds would settle it**, and this is now the cheapest
  high-value follow-up (60 more trials at 60 s ≈ 40 min).
* ❌ **No claim on reach-rate.** The pre-registered primary is null at both points.
* ❌ **No claim that our scheduler beats SOTA generally.** The defensible claim is narrower and more
  interesting: *the advantage is conditional on the budget binding.*

## Reproduce

```bash
bash experiments/exp4/run_sota_budget_axis.sh
python -m experiments.exp4.analyze_sota
```

Raw: `results/exp4_sota/{pilot,b60}.csv` + traces. τ is recomputable from traces at any threshold.
