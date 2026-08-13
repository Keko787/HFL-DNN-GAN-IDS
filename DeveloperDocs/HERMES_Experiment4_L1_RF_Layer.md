# Experiment 4 — Layer 1: the RF / adaptive-channel layer

**Scope.** How L1 is modelled, implemented and wired in Experiment 4 (arm **H3**), what it
measurably does, and — stated plainly — what it does not do. Companion to
[`HERMES_Experiment4_Methodology_and_Implementation.md`](HERMES_Experiment4_Methodology_and_Implementation.md)
(the holistic document) and
[`HERMES_Experiment4_L2_Scheduling_Layer.md`](HERMES_Experiment4_L2_Scheduling_Layer.md).
The validity record and every caveat live in
[`HERMES_Experiment4_Jittery_Methodology.md`](HERMES_Experiment4_Jittery_Methodology.md) §7.

---

## 0. The core idea

L1's job is choosing **which radio band the mule uses for its long-range backhaul upload to the
base station**. The causal chain the experiment measures is:

> a good band → high SNR → low upload loss → **the FL round closes**

Everything below exists to make that chain measurable, and to make the comparison between a
*fixed* band and an *adaptive* one fair.

| Arm | Band policy |
|---|---|
| H1, H2 | `best_average_band` — one fixed band held for the whole sortie |
| **H3** | the `U(c,t)` controller — re-selects per mission |

---

## 1. The channel environment — `experiments/exp4/channel.py`

`ChannelModel` gives each of `n_bands` (default 3) a time-varying effective SNR across the
mission sequence:

```
snr(m, c) = base + g(c) + amp · sin(2π (m/T + φ_c)) + noise
```

| Term | Value / role |
|---|---|
| `base`, `amp` | switch by regime — **clean = 12 / 1** (all bands high and flat), **jittery = 6 / 5** (bands swing into deep troughs) |
| `g(c)` | `~ U(0,3)`, a modest static per-band gain, so no band dominates outright |
| `φ_c` | **distinct per band — the load-bearing choice** |
| `noise` | Gaussian, sd 0.4 clean / 1.5 jittery |
| `T` | period, `max(2, n_missions)` |

**Why distinct phases matter.** They make the bands *cross over*: whichever band is best changes
over time. Without that, a fixed band would tie the adaptive one and L1 could earn nothing — the
experiment would be incapable of showing an effect in either direction, which is not a fair test,
it is a null test.

Everything is seeded off the trial's paired seed, so **H2 and H3 read the identical SNR trace**.

## 2. The band policy — `hermes/l1/channel_utility.py`

This is the paper's deterministic HERMES-Heuristic controller (§III-A Eq. 1), which the SEC26
audit found was **specified but never shipped as runtime code**:

```
U(c,t) = R(γ₁(t) + g(c)) − κ(c) − λ(c,t)
```

* `R(·)` — a monotone rate tier, `log2(1 + 10^(snr/10))`
* `κ(c)` — a per-band use cost
* `λ(c,t)` — a switch penalty, charged **only** when the chosen band differs from the current one,
  so the controller will not thrash on noise. Ties keep the incumbent.

**The baseline is deliberately fair.** H1/H2 hold `best_average_band`: the single band a deployer
would pick from historical averages *without* real-time tracking — Exp 2's "Expected fixed". It is
explicitly **not** a retrospective per-instant oracle, which would be an unbeatable straw man in
the other direction.

## 3. SNR → loss — `loss_from_snr`

A logistic (`mid=3, scale=2`) maps the chosen band's SNR to a per-mission upload-loss probability:

| SNR | loss |
|---|---|
| ~12 dB | ≈ 1 % |
| ~1 dB | ≈ 73 % |

## 4. How it reaches the real stack

The model runs **in-process in the driver** — deterministic, and requiring no cross-process
channel coordination — producing a `BackhaulPlan` that splits into two consumers:

| Consumer | Field | Effect |
|---|---|---|
| **Cluster** | `backhaul_loss_schedule` | `_backhaul_dropped(mission_round)` indexes it and draws a Bernoulli. On a drop the mule's aggregate is lost, **the round does not close**, and θ is carried forward (recoverable, per-mission). |
| **Mule** | `mean_chosen_snr_db` → `rf_prior_snr_db` | feeds `build_target_queue`'s S3.5 selector feature. **This is the L1→L2 edge the audit flagged as never wired** — previously a hardcoded `20.0` for every candidate, i.e. zero discriminative signal. |

Crucially, **H1/H2/H3 share the same `backhaul_rng_seed`**, so all arms face the identical
Bernoulli draw sequence — only the thresholds differ. A difference between arms therefore cannot
come from luck of the draw.

---

## 5. What it is *not* — stated plainly

This is a **modelled radio, not a real one**. There is no RF propagation, no real spectrum
sensing, and the controller observes all bands' current SNR with perfect, cost-free sensing.

Two known gaps:

1. **Switching energy** is not in a unified budget.
2. **Switching-induced upload loss** is not charged in the realised schedule — only in the decision
   utility — so the reported gain is an **upper bound assuming lossless retuning**.

And because SNR → loss is monotone on a shared trace, **the sign of the jittery benefit is close to
baked-in by construction**; the interesting quantity is the *magnitude*, which is
calibration-dependent (≈0.02–0.29 across defensible constants). All six caveats are recorded in
[`HERMES_Experiment4_Jittery_Methodology.md`](HERMES_Experiment4_Jittery_Methodology.md) §7.2.

---

## 6. What it measurably does — and does not

### 6.1 Channel-model level: a real, consistent effect

Mean per-mission backhaul-loss reduction (fixed − adaptive), 30 seeds, `n_missions=6`, 3 bands:

| regime | mean reduction | worst seed | best seed | adaptive ≤ fixed |
|---|---|---|---|---|
| clean | **+0.0003** | −0.0005 | +0.0020 | 29/30 (≈90 % at 1000 seeds) |
| jittery | **+0.1379** | +0.0540 | +0.2357 | **30/30** (1000/1000 at 1000 seeds) |

Clean is a wash — correctly, since there is nothing to track. Jittery is a consistent reduction
that never flips sign. A 20-skeptic adversarial audit found **no rigging**: the direction survives
`n_bands` 2–5, `n_missions` 2–12, the full loss-map grid, and — importantly — the *strongest* fair
fixed baseline (the loss-optimal band, not merely the mean-SNR one) still loses by +0.1344.

### 6.2 End-to-end: no detectable effect

> **This is the honest headline, and it must not be softened.**

Across the H3-vs-H2 dead-zone sweep (200/200 valid trials, 20/20 paired seeds per cell,
`n_missions=4`), **all 5 conditions × 4 metrics are ties**; two AUC cells are nominally negative.

| condition | H2 | H3 | H3−H2 | 95 % CI | verdict |
|---|---|---|---|---|---|
| clean | 0.849 | 0.863 | +0.015 | [−0.090, +0.117] | tie |
| jittery dz=0.0 | 0.910 | 0.911 | +0.001 | [−0.092, +0.092] | tie |
| jittery dz=0.2 | 0.800 | 0.867 | +0.068 | [−0.070, +0.206] | tie |
| jittery dz=0.4 | 0.949 | 0.938 | **−0.011** | [−0.088, +0.043] | tie |
| jittery dz=0.6 | 0.953 | 0.943 | **−0.010** | [−0.095, +0.047] | tie |

A single cell at a 6-mission horizon does reach significance (+0.012 AUC, p=0.044), but one
significant cell against five ties is **not a robust result**.

> **A retracted claim.** An earlier version of the layer-1 figure appeared to show a positive,
> severity-increasing effect (Cliff's δ +0.14…+0.37). That came from a contaminated sweep — 22/190
> `ok` rows had produced no model at all — combined with an **unpaired** effect size computed
> against a paired protocol. On clean, properly paired data the effect disappears. Do not reuse
> those numbers.

### 6.3 The defensible claim

> Adaptive L1 channel selection reduces *modelled* mule→BS backhaul loss under a jittery,
> band-crossing channel (≈0.13–0.14 at this calibration, never worse across 1000 seeds) and is a
> wash on a healthy backhaul. End-to-end, that advantage is small and **not robustly detectable**.
> L1 is presented as a **backhaul-robustness mechanism, not an accuracy driver.**

---

## 7. Where the code lives

| What | Path |
|---|---|
| `U(c,t)` controller | [`hermes/l1/channel_utility.py`](../hermes/l1/channel_utility.py) |
| Channel model + loss map + `BackhaulPlan` | [`experiments/exp4/channel.py`](../experiments/exp4/channel.py) |
| Per-arm plan construction (`adaptive = arm=="H3"`) | [`experiments/exp4/driver.py`](../experiments/exp4/driver.py) |
| Schedule → cluster | `ClusterConfig.backhaul_loss_schedule` → `_backhaul_dropped()` in [`hermes/processes/cluster.py`](../hermes/processes/cluster.py) |
| RF prior → mule/selector | `MuleConfig.rf_prior_snr_db` → [`hermes/processes/mule.py`](../hermes/processes/mule.py) → `build_target_queue` |
| Unit tests (31, deterministic) | [`tests/unit/test_exp4_channel.py`](../tests/unit/test_exp4_channel.py) |
| End-to-end smoke | `tests/integration/test_exp4_realmodel_smoke.py::test_exp4_h3_l1_channel_runs_end_to_end` |
| CLI | `--l1-channel`, `--l1-channel-bands` on `experiments.exp4.runner_main` |
