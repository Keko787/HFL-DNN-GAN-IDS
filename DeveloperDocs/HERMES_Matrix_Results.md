# Phase-3 matrix — results

**Run 2026-08-13.** 560 trials, **0 failures, 0 duplicates**, every sweep reconciled against its
designed cell × arm × seed product before analysis. First results with **deadline enforcement ON**
(`--mission-budget-s 120`), so Amendments A1 (in-flight abort) and A2 (starvation widening) are live
in every mule trial. S3c pinned **off**. Traces retained.

**Claim rule** (unchanged): a difference is claimed only when the bootstrap CI **excludes 0** *and*
paired Wilcoxon **p < 0.05**. Where several metrics are tested, the multiplicity position is stated
rather than left implicit.

| Sweep | Comparison | Rows | Verdict |
|---|---|---|---|
| **A1** | H1 vs H0, clean | 40 | **H0 wins participation; AUC a tie** |
| **A2** | H1 vs H0, jittery (12-cell surface) | 480 | **H1 wins everything, decisively** |
| **C** | H3 vs H2, L1 adaptivity | 120 | **CONFIRMED — AUC + accuracy both survive Bonferroni** |

---

## A2 — the headline result: H1 > H0 under jitter

| Metric | H1 | H0 | Δ | 95% CI | p | δ | |
|---|---|---|---|---|---|---|---|
| `final_auc` | 0.927 | 0.873 | **+0.054** | [+0.0373, +0.0716] | <0.0001 | +0.28 | ✅ |
| `final_accuracy` | 0.821 | 0.787 | +0.035 | [+0.0199, +0.0502] | 0.0001 | +0.18 | ✅ |
| `mission_completion_rate` | 0.669 | 0.478 | **+0.191** | [+0.1542, +0.2299] | <0.0001 | +0.45 | ✅ |
| `update_yield` | 1.666 | 1.247 | +0.419 | [+0.3073, +0.5334] | <0.0001 | +0.37 | ✅ |
| `round_close_rate_kmin2` | 0.533 | 0.356 | +0.177 | [+0.1271, +0.2282] | <0.0001 | +0.32 | ✅ |

**All five survive Bonferroni** (α = 0.05/5 = 0.01; every p ≤ 0.0001). This is a robust result, not a
single lucky metric — and it is the claim the paper is built on.

### The sensitivity surface — where the advantage comes from

`final_auc`, per `dead_zone × link_quality` cell:

| | lq=0.3 | lq=0.5 | lq=0.7 |
|---|---|---|---|
| **dz=0.0** | **+0.039** ✅ | −0.001 tie | −0.015 tie |
| **dz=0.2** | **+0.052** ✅ | +0.009 tie | −0.007 tie |
| **dz=0.4** | **+0.084** ✅ | +0.020 tie | +0.001 tie |
| **dz=0.6** | **+0.338** ✅ δ=+0.98 | **+0.096** ✅ | +0.032 (p=0.18, **not** claimable) |

**The mechanism is visible in the raw means, and it is the interesting part.** H1's `final_auc` sits
between **0.922 and 0.933 in every one of the twelve cells** — flat. H0's falls from 0.943 to
**0.587** as the infrastructure degrades. The mule does not "beat" flat FL by learning better; it is
**insensitive to backhaul degradation**, because it never uses the backhaul that is failing.

So the advantage is **entirely a function of severity**, and the surface says so cleanly: ties along
the mild edge (lq=0.7), significant wins along the severe edge (lq=0.3), and near-total separation
(δ=+0.98) at the worst corner. *That is a crossover surface, not a uniform claim* — and reporting it
as a surface is far stronger than reporting the pooled mean alone.

> **One cell needs care.** `dz=0.6, lq=0.7` shows a CI excluding 0 (+0.0039 to +0.0622) but
> **p = 0.1769**. By our own rule it is **not claimable**. It must not be counted among the wins.

## A1 — the honest cost: under clean conditions, the mule loses

| Metric | H1 | H0 | Δ | p | |
|---|---|---|---|---|---|
| `final_auc` | 0.932 | 0.950 | −0.018 | 0.1429 | tie |
| `final_accuracy` | 0.841 | 0.835 | +0.006 | 0.8408 | tie |
| `mission_completion_rate` | 0.733 | 0.967 | **−0.233** | 0.0009 | ❌ H0 |
| `update_yield` | 1.750 | 3.513 | **−1.762** | 0.0001 | ❌ H0 |
| `round_close_rate_kmin2` | 0.537 | 0.950 | **−0.412** | 0.0002 | ❌ H0 |

**This belongs in the paper, prominently.** With a healthy backhaul, flat FL collects roughly twice
the updates per round and closes far more rounds; the mule pays latency and now also pays the S3b
gate. **Model quality is nonetheless a tie** (AUC p=0.14) — H1 reaches equivalent accuracy from
fewer updates.

The honest framing: *HERMES is not a general improvement on federated learning. It is an
availability mechanism, and it costs throughput when availability is not the problem.* Stating that
makes the jittery result credible; hiding it would invite exactly the scrutiny that found the
earlier L1 problem.

## C — L1 adaptivity: CONFIRMED (was suggestive)

The first pass gave `final_auc` +0.042 at p=0.018 with n=20 — passing the claim rule but **failing**
Bonferroni for five metrics. Two independent confirmations were run rather than reporting it as-is.

### C1 — same operating point, seeds 20 → 40

| Metric | H3 | H2 | Δ | 95% CI | p | δ | |
|---|---|---|---|---|---|---|---|
| `final_auc` | 0.911 | 0.865 | **+0.046** | [+0.0207, +0.0734] | **0.0016** | +0.28 | ✅ |
| `final_accuracy` | 0.816 | 0.786 | **+0.030** | [+0.0129, +0.0478] | **0.0026** | +0.27 | ✅ |
| `mission_completion_rate` | 0.646 | 0.642 | +0.004 | [−0.0083, +0.0208] | 0.7855 | +0.01 | tie |
| `update_yield` | 1.550 | 1.544 | +0.006 | [−0.0500, +0.0688] | 0.8316 | +0.02 | tie |
| `round_close_rate_kmin2` | 0.512 | 0.519 | −0.006 | [−0.0375, +0.0187] | 0.6547 | −0.01 | tie |

**Both convergence metrics now survive Bonferroni** (α = 0.01; p = 0.0016 and 0.0026). Doubling the
seeds moved AUC from p=0.018 to p=0.0016 and pulled accuracy in with it, while the point estimate
held (+0.042 → +0.046) and the CI tightened (width 0.071 → 0.053). **That is what a real effect does
under more data; noise regresses instead.**

### C2 — independent operating point, `n_missions` 4 → 6

| Metric | H3 | H2 | Δ | 95% CI | p | |
|---|---|---|---|---|---|---|
| `final_auc` | 0.930 | 0.908 | **+0.023** | [+0.0074, +0.0397] | 0.0125 | meets the rule |
| `final_accuracy` | 0.838 | 0.815 | +0.023 | [+0.0002, +0.0447] | 0.0745 | tie |
| participation metrics | — | — | ≈0 | straddle 0 | >0.31 | tie |

Reproduces in direction and significance at a point that shares no seeds with C1. p=0.0125 is
marginal against a 5-metric Bonferroni, so C2 **corroborates** C1 rather than standing alone.

### What L1 actually does — settled from the traces, no re-run

Two hypotheses were on the table, and **both were wrong.**

I first read the AUC gap narrowing with more missions (+0.046 at 4, +0.023 at 6) as *"L1 reaches
the same model sooner"*. The traces say otherwise. `--keep-event-traces` preserved every round's
`model_eval`, so **T@τ was recomputed at arbitrary τ without re-running anything** — the first
payoff of trace retention.

**Conditional on reaching τ, the two arms are indistinguishable:**

| τ | H3 rounds | H2 rounds | diff | p |
|---|---|---|---|---|
| 0.82 | 2.83 | 2.83 | **0.000** | 1.0 |
| 0.85 | 3.33 | 3.22 | +0.111 | 1.0 |
| 0.75 | 2.27 | 2.27 | **0.000** | 1.0 |

**The entire effect is _reachability_** — how often a run gets there at all. Paired per seed, so
McNemar on the discordant pairs is the right test:

| τ | H3 reaches | H2 reaches | H3-only | H2-only | McNemar p |
|---|---|---|---|---|---|
| 0.90 | 1/40 | 2/40 | 0 | 1 | 1.0000 |
| 0.88 | 4/40 | 4/40 | 0 | 0 | 1.0000 |
| 0.85 | 16/40 | 10/40 | 7 | 1 | 0.0703 |
| **0.82** | **28/40** | **19/40** | 10 | 1 | **0.0117** ✅ |
| 0.80 | 32/40 | 24/40 | 9 | 1 | 0.0215 ✅ |
| **0.75** | **39/40** | **30/40** | 9 | **0** | **0.0039** ✅✅ |

**The discordance is essentially one-directional** — 9–10 seeds where H3 reached the target and H2
did not, against 0–1 the other way. τ=0.75 survives Bonferroni over all six thresholds (0.0039 <
0.0083).

> **The claim, stated correctly:** *L1 adaptivity does not make training faster. It makes the target
> reachable.* Choosing a better channel drops fewer backhaul uploads, so more rounds actually close,
> so more runs converge at all — and the ones that converge do so at the same rate.

This also explains the narrowing gap: with more missions H2 gets more chances to reach τ, so the
reachability advantage compresses. Consistent with the same mechanism, no longer a puzzle.

### τ was the problem, and the data proves it

The τ=0.90 row above is the diagnosis in one line: **1 vs 2 of 40 trials reach it, 38 reach
neither** — zero resolution, which is why the metric looked dead. **τ is now 0.82 by default**
(`--tau`, and `Exp4Driver.tau`), the median `final_accuracy` over the 640-trial matrix, where the
metric discriminates cleanly.

> **Asymmetry to remember: only the mule arms are traced.** H0 runs in-process with no orchestrator
> and emits no run-dir JSONL. So T@τ is recomputable for H1/H2/H3/B1/B2 but **not H0** — an
> H0-vs-H1 time-to-accuracy comparison still needs a re-run. The L1 comparison did not.

### Consequence for the SOTA comparison

This changes the metric the whole-system comparison should lead with. *Training time to target
accuracy* is the natural framing from the Oort and FedCS abstracts — but in this system,
**time-to-accuracy conditional on success is identical across arms**, so it would report a tie and
miss the real effect.

**The discriminating metric here is _probability of reaching target accuracy within the mission
budget_.** That is what differs, it is what a deployment actually cares about under DDIL, and it has
a natural paired test (McNemar). Recommend leading with reach-rate and reporting conditional
time-to-accuracy alongside it as the honest null.

---

## Provenance and caveats

* **Environment:** full multi-process, single host, real TCP, real canonical CICIOT training.
  Loopback, not a distributed deployment (see the checklist's provenance table).
* **Enforcement ON** in every mule arm — these numbers **supersede** the previously committed
  participation figures rather than amending them.
* **Not paired across sweeps.** A (no `--l1-channel`) and C (with it) use different backhaul models
  and non-aligned seeds. H2/H3 must never be compared against H0/H1.
* **Sweep B (H1 vs B1 vs B2) did not run** — vacuous at this operating point (checklist §5.1b): S3b
  fixes *who* is served before the policy ranks anything, so all three arms are identical unless the
  queue is truncated in flight.
* **Cost model was wrong by ~2.4×.** It was calibrated on `--data-source synthetic`; the matrix ran
  `canonical`. Sweep A took ~4 h against a costed 101 min. Arm *ratios* held; absolute per-trial cost
  did not transfer.
