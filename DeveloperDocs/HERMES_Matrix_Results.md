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

### A prediction that failed — and what it implies

I predicted **more** missions would give a **larger** effect, since L1 chooses a channel per mission
and the advantage should compound. **It halved: +0.046 at 4 missions, +0.023 at 6.** Both arms
improved with more rounds (H2 0.865 → 0.908; H3 0.911 → 0.930) — **H2 closes the gap.**

That falsifies "L1 reaches a better model" and points to **"L1 reaches the same model sooner"** —
choosing a better channel drops fewer backhaul uploads early, and given enough rounds the fixed-band
arm catches up. It is the same shape as the S3c pilot result, which is at least a consistent story
about this system: *these mechanisms buy convergence speed, not a better endpoint.*

> **Not established.** The two gaps' CIs overlap ([+0.021,+0.073] vs [+0.007,+0.040]), and C1/C2
> differ in sample size. The narrowing is **suggestive**, and the direct test is blocked (below).

### ⚠ The metric that would settle it is currently unusable

Time-to-accuracy (`t_at_tau_round`) is exactly the right instrument for a speed claim — and for the
SOTA comparison, where the natural framing is *improvement in training time to a target accuracy*.
**It cannot be used at τ = 0.9:** only **1 of 40** H3 trials and 2 of 40 H2 trials ever reach it.

Across all 640 matrix trials, `final_accuracy` has median **0.820** and p90 **0.888**:

| τ | trials reaching it |
|---|---|
| 0.90 (current) | **5.9 %** |
| 0.88 | 13.8 % |
| **0.85** | **30.0 %** |
| **0.82** | **50.2 %** |
| 0.80 | 61.1 % |

**Recommendation: set τ = 0.82** (the median, so ~half of trials reach it and the metric has
resolution in both directions), and report τ = 0.85 as a sensitivity check. τ = 0.9 is above p90 —
it measures almost nothing. *This is a no-re-run change if the per-round evaluation history is in the
retained traces; otherwise it needs a re-run, which is now cheap because traces are kept.*

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
