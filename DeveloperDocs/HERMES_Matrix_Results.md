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
| **C** | H3 vs H2, L1 adaptivity | 40 | **AUC only — and it does not survive correction** |

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

## C — L1 adaptivity: suggestive, not established

| Metric | H3 | H2 | Δ | 95% CI | p | δ | |
|---|---|---|---|---|---|---|---|
| `final_auc` | 0.937 | 0.895 | +0.042 | [+0.0116, +0.0823] | 0.0180 | +0.33 | meets the rule |
| `final_accuracy` | 0.835 | 0.810 | +0.025 | [+0.0010, +0.0529] | 0.0630 | +0.21 | tie |
| `mission_completion_rate` | 0.675 | 0.667 | +0.008 | [0.0000, +0.0250] | 0.3173 | +0.03 | tie |
| `update_yield` | 1.675 | 1.663 | +0.013 | [−0.0750, +0.1125] | 0.6547 | +0.04 | tie |
| `round_close_rate_kmin2` | 0.537 | 0.537 | 0.000 | [−0.0375, +0.0375] | 1.0000 | 0.00 | tie |

**⚠ It passes the claim rule but fails multiplicity correction.** Five metrics tested puts Bonferroni
at α = 0.01, and **p = 0.0180 exceeds it**. One metric of five at p<0.05 is roughly what chance
produces.

Given this project **already retracted an L1 effect once**, the standard here should be higher than
elsewhere, not lower. **Recommended: report as suggestive, with n=20 at a single cell stated, and do
not headline it.**

The pattern is at least *mechanistically coherent* — L1 improves `final_auc` while leaving every
participation metric untouched, which is what choosing a better channel should do: fewer dropped
backhaul uploads, same devices visited. Coherence is not significance, but it does argue against
this being pure noise. **A second cell, or more seeds, would settle it** — sweep C is only ~40
trials.

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
