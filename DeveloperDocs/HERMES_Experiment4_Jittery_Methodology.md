# Experiment 4 — Jittery-Regime Methodology & Validity Record

**Status (2026-07-24):** remediation in progress. This document is the honest
methodological record for the integrated jittery experiment (H0 traditional FL
vs H1 integrated HERMES, clean vs jittery). It exists because an adversarial
review found the first jittery result was *manufactured* by asymmetric,
hand-tuned knobs rather than measured; the fixes below make the comparison
fair, and this document states exactly what is and is not claimed.

---

## 1. What the experiment compares

Both arms federate the **same real DNN-IDS** (`create_CICIOT_Model`, 21
features, balanced CICIOT-2023 via the canonical pipeline) at **paired seeds**
(same device reliabilities, same data shards, same initial θ), across two
network regimes: **clean** and **jittery**. The question is whether physically
transporting model updates on a UAV mule (H1) preserves federated learning
when the long-range backhaul degrades, versus centralized FL over that
backhaul (H0).

## 2. The physical model (symmetric, jitter hits long-range only)

The load-bearing modelling principle: **jitter degrades long-range links; it
does not degrade short-range device↔mule contact.** This is what the mule
architecture exploits and it is the only asymmetry the result is allowed to
rest on.

| Quantity | Definition | Applies to |
|---|---|---|
| Device reliability `rel_i` | `~ Uniform(0.15, 1.0)`, drawn once per seed | **both arms** (a device's availability is a device property) |
| `rf_factor` | `max(0.4, 1 − d_eff/(3·world_radius))`, `d_eff = min(dist, rf_range)`, `world_radius=100 m` | H1 short-range only |
| **H0 per-round participation** | `rel_i × link_quality`; dead-zoned clients → 0 | long-range server link |
| **H1 per-contact completion** | `rel_i × rf_factor` (regime-independent) | short-range, jitter-immune |
| **H1 backhaul loss** | jittery: whole-mission upload dropped w.p. `backhaul_loss` (2%); round does not close, θ carried to next dock (**recoverable**) | mule→BS long-range hop |
| **H0 dead-zone** | jittery: fraction of clients with **no** long-range path (permanent) | long-range server link |

* **Clean** (`link_quality=1.0`, `dead_zone=0`): H0 = `rel_i`, H1 = `rel_i × rf_factor`.
  Both imperfect. H0 is **not** idealised to perfect participation (the original bug).
* **Jittery**: H0's long-range link degrades (`link_quality<1`) and a dead-zone
  fraction is unreachable; H1's short-range collection is unaffected — only its
  single backhaul hop takes a recoverable 2% loss.

### Physical justification for the dead-zone (B2)

The dead-zone is **not** a round-count artifact. It is the fraction of edge
devices that have **no viable long-range path to the edge server** for the
duration of a mission — from terrain blockage, range-edge SNR collapse, or
line-of-sight obstruction in a DDIL deployment. This is a **spatial/deployment
property**, independent of how many FL rounds run, which is why it is
legitimate to apply at a short (2-round) horizon. Its *rate* is uncertain, so
it is reported as a **sensitivity axis** (`--dead-zone 0.0 0.2 0.4 0.6`), not a
single tuned value. Empirical anchor: Exp 1's measured jittery cell shows
centralized participation collapsing under the same 30% latency-jitter / 2%
loss link. The mule gets **no** dead-zone because it physically flies to each
device — that is the architectural thesis under test, made explicit rather than
assumed.

> Provenance note: the dead-zone / `link_quality` mechanism is the flat-FL (A1)
> model from `experiments/exp3/arm_a1.py` + the Exp-3 driver — **not** from the
> Exp-3 simulator (`sim_env.py`), which has no flat-FL arm. Earlier comments
> claiming Exp-3-sim provenance were wrong and have been corrected.

## 3. Metrics — what is honest

* **Headline (report these):** `final_auc` / `final_accuracy` on the held-out
  set, `mission_completion_rate` (fraction of devices contributing ≥1 clean
  Δθ), `update_yield` (successful updates/round), `round_close_rate@{2,N}`.
* **`round_close_rate` now honest:** a round closes only if it produced ≥1
  update **and** its backhaul upload was not dropped. Empty and
  backhaul-dropped rounds count as *not closed*, so H1's jittery penalty is
  visible (previously hard-coded closed — B5).
* **`coverage` is a by-construction descriptor, not evidence.** It counts
  `device_served` events, which include Pass-2 deliveries (which reach every
  device) and failed Pass-1 contacts — so it is ~1.0 for any mule regardless of
  collection success. Do **not** headline it; use `mission_completion_rate`.

### 3.1 `final_auc` is bimodal — the session-collapse mode (disclosed)

A trial whose session never lands an aggregated update leaves θ at its
initialisation, so its `final_auc` **is** `init_auc` (≈0.25) — not a model that
learned the wrong thing, a model that never moved. `final_auc` is therefore a
**two-component distribution**: a collapse mode at ≈0.25 and a trained mode at
≈0.96–0.99, with **nothing in between** (0 of 519 rows fall in [0.30, 0.65];
10/519 = 1.9 % sit bit-identically at `init_auc` with `delta_auc = 0`).

Three consequences, all applied in this document:

1. **Means are expectations over collapse risk**, not typical outcomes. Where a
   mean is quoted, the collapse rate is quoted with it (§6.1).
2. **± SD is never reported for AUC.** The spread is tail-driven, not unimodal;
   a symmetric ±1 SD interval both implies a spread that does not exist and
   extends past AUC = 1.0, which is impossible. Figures use the **percentile
   bootstrap CI of the mean** (bounded inside the data range by construction)
   and overlay the per-seed points so the collapse mode is visible rather than
   smeared; sessions in the collapse mode fall below the zoomed AUC window and
   are drawn as an explicit `↓n` count (§5.1).
3. **Rank-based tests are unaffected.** The paired Wilcoxon + Cliff's δ of §5
   operate on ranks, so the tail does not distort them; the bootstrap CI is
   likewise computed on the paired differences.

**On `init_auc` ≈ 0.25.** The untrained network is effectively a *constant*
predictor — its outputs span ≈0.4995–0.5052 on the held-out set, `init_loss`
≈ ln 2, and it labels nearly every row "attack". AUC is rank-based and
scale-free, so it magnifies that sub-1 % tilt into an apparently decisive 0.25.
It is **not** evidence of an anti-predictive model, and it is **not** an
inversion bug: `roc_auc_score(y, 1−p)` returns exactly 1 − 0.276, a
random-permutation baseline on the same labels gives 0.4999, and the identical
code path reports 0.99 after training (a global inversion would report 0.01).
Because `Exp4Driver.theta_seed` is fixed, every trial reuses the *same* untrained
network, so 0.25 is **one draw, not an average** — sweeping the init seed gives
mean 0.41 (sd 0.20, range 0.13–0.84), i.e. centred on chance. Describe it as
"the untrained initialisation", never as an achieved score.

## 4. Scope boundary (stated, not hidden)

The integrated experiment models the **network + computation** layers with real
fidelity (real FedAvg, real DNN-IDS, real contact/backhaul impairment). It does
**not** model **mule flight physics** — propulsion energy, mission time under a
tight flight budget β, or coverage limited by budget. Those live in the Exp-3
scheduling simulator and are complementary. Consequently this experiment
validates **participation/convergence resilience** (Observation 3), not the
budget-scheduling claim (Observation 4), which is the H2 (RL selector) work.

## 5. Statistical protocol (B1)

* **≥20 paired seeds per cell.** For each regime × metric, form the paired
  `(H1 − H0)` differences and run **paired Wilcoxon + Cliff's δ**
  (`analysis/stats.py`) and a **bootstrap 95% CI on the mean difference**
  (`python -m experiments.analysis.exp4 --csv <sweep>.csv`).
* **A verdict is claimed only when the CI excludes 0** (and Wilcoxon p<0.05):
  `H1 > H0`, `H0 > H1`, or `tie`. A tie is a legitimate, reportable outcome.
* **Sensitivity surface** (`--surface`): the jittery verdict at each
  `dead_zone × link_quality` point, so the headline is a *region* where the
  advantage holds, not a tuned point. The review's flip test (dead_zone 0.3 +
  link_quality 0.7 → H0 may exceed H1) must be reported if it occurs.

### 5.1 Figure conventions (both figures obey these)

The two paper figures — `fig_exp4_crossover.png` (H0 vs H1) and
`fig_exp4_layer1.png` (H2 vs H3) — are generated by
[`DeveloperDocs/exp4_figure.py`](exp4_figure.py) and
[`exp4_figure_layer1.py`](exp4_figure_layer1.py). Both import
[`experiments/analysis/figstyle.py`](../experiments/analysis/figstyle.py) for
scales and drawing, and take every number from `analysis/exp4.py` — the module
that produces the tables above. **A figure therefore cannot disagree with the
text, or with the other figure.** This is enforced by shared code, not by
convention: an earlier layer-1 figure was built from a different sweep than the
table it illustrated and contradicted it by ~0.19 AUC (§7.4).

**Layout.** Both figures are three panels, identical structure:

| panel | content | axis |
|---|---|---|
| 1 | Final model AUC | `AUC_ZOOM_YLIM` = **[0.60, 1.00]**, shared by both figures |
| 2 | Participation = `update_yield ÷ N` | **[0, 1]**, shared |
| 3 | Paired (treatment − baseline) + bootstrap 95 % CI | `DIFF_YLIM` = **[−0.46, +0.25]**, shared |

**Normalization.** Every plotted metric is a fraction in [0,1]. `update_yield`
is the only raw count (0–5.25 updates/round against N=6 devices), so it is
divided by the device count to become a participation fraction — otherwise a
[0,1] metric and a 0–5 count sit on adjacent panels and invite misreading.

**Uncertainty is always the percentile bootstrap CI of the mean** — the same
estimator §5 specifies, and bounded inside the data range *by construction*, so
a bounded metric can never be drawn outside its bounds. **Never a symmetric ±SD
whisker:** `final_auc` is bimodal (§3.1), so ±SD implies a spread that does not
exist *and* renders above AUC = 1.0. That defect shipped once and is what the
guards below now prevent.

**The AUC panels are truncated, and say so.** On a full [0,1] axis the AUC
differences (0.01–0.12) are visually indistinguishable — every bar looks the
same height. The panels are therefore zoomed to [0.60, 1.00] (2.5× the vertical
resolution). Three obligations come with that, all met:

* **The floor is 0.60, not higher.** It is the tightest window containing every
  mean *and* every bootstrap CI bound in **both** datasets (the lowest is
  0.6572 — figure-2 jittery dz=0.2 H2). A 0.80 floor was considered and rejected:
  it would have cropped a bar mean of 0.7997 and four CI bounds, i.e. hidden real
  results — the same class of error as drawing AUC above 1.0.
* **The truncation is declared** with the conventional `//` break marks on both
  spines, and the caption states that bar lengths in that panel are *not*
  proportional to magnitude. Bars are retained for consistency across the figure
  set; because a bar encodes magnitude by length from zero, a truncated bar chart
  would otherwise overstate differences. **No quantitative claim rests on
  comparing truncated bar heights** — panel 3 carries the effect sizes and CIs.
* **Nothing below the window is hidden.** Sessions that collapsed to the
  untrained init (§3.1) fall below 0.60 and are drawn as a per-condition `↓n`
  count, so the failure mode stays visible.

**The difference panel shares one fixed range across both figures.** This is
deliberate: it puts the mule effect (up to −0.36 participation clean, +0.18 at
dz=0.6) and the L1 effect (every interval crossing zero) on the same scale, so a
null cannot look as dramatic as a large effect. A bar is starred only where the
§5 rule permits a claim — CI excludes 0 **and** p<0.05.

**Guards.** Both scripts assert that no artist exceeds a normalized metric's
bounds, that every mean/CI lies inside the zoomed AUC window
(`assert_within_zoom`), and that every difference CI lies inside the shared
difference axis. A future condition that would fall outside fails the build
loudly rather than being silently cropped. Bootstrap and jitter use fixed seeds,
so both figures are byte-reproducible.

**Excluded rows are reported, not swallowed.** Each script prints how many rows
it dropped as non-`ok` (including `no_eval` — a session that produced no model
at all, §8); an earlier version discarded them in a bare `except`.

## 6. Result — 20 paired seeds, full dead-zone × link surface

Canonical CICIOT, N=6, `--n-missions 4`, **20 paired seeds**, the full
`dead_zone ∈ {0.0,0.2,0.4,0.6} × link_quality ∈ {0.3,0.5,0.7}` jittery surface
+ a clean reference (`dead_zone=0.0, link=0.5`). Paired Wilcoxon + Cliff's δ +
bootstrap 95% CI on the mean (H1 − H0) difference
(`python -m experiments.analysis.exp4 --csv h0h1_all.csv --surface`). ✱ = CI
excludes 0 and Wilcoxon p<0.05. Jittery rows are **pooled** over all 12 surface
cells (n≈239 pairs); the per-cell breakdown is the surface table below.

| regime | metric | H1 | H0 | H1−H0 | 95% CI | p | δ | verdict |
|---|---|---|---|---|---|---|---|---|
| clean | mission_completion_rate | 0.550 | 0.892 | −0.342 | [−0.442,−0.242] | 0.0002 | −0.76 | H0 > H1 ✱ |
| clean | update_yield | 1.30 | 3.44 | −2.14 | [−2.58,−1.76] | 0.0001 | −0.95 | H0 > H1 ✱ |
| clean | round_close_rate@2 | 0.375 | 0.963 | −0.588 | [−0.713,−0.475] | 0.0001 | −0.96 | H0 > H1 ✱ |
| clean | final_auc | 0.958 | 0.985 | −0.027 | [−0.039,−0.017] | <0.001 | −0.69 | H0 > H1 ✱ |
| jittery | mission_completion_rate | 0.660 | 0.478 | +0.183 | [+0.145,+0.222] | <0.001 | +0.41 | H1 > H0 ✱ |
| jittery | update_yield | 1.65 | 1.24 | +0.405 | [+0.281,+0.525] | <0.001 | +0.31 | H1 > H0 ✱ |
| jittery | round_close_rate@2 | 0.536 | 0.355 | +0.181 | [+0.130,+0.231] | <0.001 | +0.31 | H1 > H0 ✱ |
| jittery | final_auc | 0.958 | 0.934 | +0.023 | [+0.005,+0.043] | 0.0024 | +0.12 | H1 > H0 ✱ |
| jittery | final_accuracy | 0.868 | 0.846 | +0.022 | [+0.005,+0.038] | 0.019 | +0.10 | H1 > H0 ✱ |

**The mule's advantage is conditional, and it scales with backhaul dead-zone —
the surface (mission_completion_rate) shows the crossover boundary:**

| dead_zone | link | H1 | H0 | H1−H0 | p | δ | verdict |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.3 | 0.825 | 0.475 | +0.350 | 0.0002 | +0.89 | H1 > H0 ✱ |
| 0.0 | 0.5 | 0.783 | 0.775 | +0.008 | 1.00 | −0.01 | tie |
| **0.0** | **0.7** | 0.518 | 0.833 | **−0.316** | 0.0011 | −0.70 | **H0 > H1 ✱** |
| 0.2 | 0.3 | 0.500 | 0.417 | +0.083 | 0.14 | +0.21 | tie |
| 0.2 | 0.5 | 0.658 | 0.558 | +0.100 | 0.14 | +0.28 | tie |
| 0.2 | 0.7 | 0.750 | 0.725 | +0.025 | 0.48 | +0.12 | tie |
| 0.4 | 0.3 | 0.525 | 0.350 | +0.175 | 0.012 | +0.47 | H1 > H0 ✱ |
| 0.4 | 0.5 | 0.692 | 0.450 | +0.242 | 0.0009 | +0.61 | H1 > H0 ✱ |
| 0.4 | 0.7 | 0.725 | 0.525 | +0.200 | 0.0010 | +0.71 | H1 > H0 ✱ |
| 0.6 | 0.3 | 0.483 | 0.133 | +0.350 | 0.0003 | +0.79 | H1 > H0 ✱ |
| 0.6 | 0.5 | 0.725 | 0.233 | +0.492 | 0.0001 | +0.98 | H1 > H0 ✱ |
| 0.6 | 0.7 | 0.733 | 0.275 | +0.458 | 0.0001 | +0.97 | H1 > H0 ✱ |

**Reading — an honest, physically-sensible crossover, not a blanket win.**

* **The mule costs you under a healthy backhaul.** Clean links: H0 beats H1 on
  every metric (completion −0.34, yield −2.14, close-rate −0.59, even AUC −0.027,
  all p<0.001). Direct centralized collection reaches everyone cheaply; the mule
  is overhead. This is stated up front, not buried.
* **The mule's advantage grows monotonically with the dead-zone.** As more
  devices lose their long-range path, the mule's physical collection wins by
  more: negligible at `dead_zone=0.0` (link-dependent, and it *flips* to H0 at
  `link=0.7` — the well-connected corner where the mule is pure overhead),
  through decisive at `dead_zone=0.4` (δ +0.47…+0.71), to dominant at
  `dead_zone=0.6` (δ up to **+0.98**, H1 delivers ~0.73 completion vs H0's
  0.23–0.28). **The flip test the review demanded (dz=0.0, link=0.7 → H0 > H1)
  did occur, and is reported.**
* **The jittery AUC gap is a _session-survival_ effect, not a model-quality
  effect.** The pooled +0.023 (p=0.0024, δ=+0.12) is real, but decomposing it
  (§6.1) shows **76 %** of it comes from how often a session collapses entirely
  and only **24 %** from the quality of the model when it does train. Conditional
  on both arms actually training, the difference is +0.0070 with a bootstrap CI
  of **[−0.0009, +0.0147] — straddling 0, i.e. a tie by this document's own
  claim rule (§5)**. Report the mechanism, not just the number.

### 6.1 Decomposition — what the AUC difference actually measures

`final_auc` is a **two-component (bimodal) distribution**. A trial that never
receives an aggregated update ends with θ still at its initialisation, so its
"final" AUC *is* `init_auc` (≈0.25); every other trial lands near ≈0.96–0.99.
Across `h0h1_all.csv` there are **zero observations between 0.30 and 0.65** —
10/519 rows (1.9 %) sit at exactly `init_auc` (bit-identical, `delta_auc = 0`),
the rest above 0.65. Averaging the two modes yields a mean that no single trial
attains, so the mean must be read as an *expectation over collapse risk*.

Splitting the pooled jittery result (n = 239 paired seeds) accordingly:

| Component | H1 | H0 | Contribution to the +0.0231 gap |
|---|---|---|---|
| **Session-collapse rate** (trial ended at the untrained init) | 2/239 = **0.84 %** | 8/239 = **3.35 %** | **+0.0177 (76.3 %)** |
| **AUC given the session trained** (n = 229 pairs, neither collapsed) | 0.9646 | 0.9577 | +0.0055 (23.7 %) |
| Conditional paired test | — | — | +0.0070, CI **[−0.0009, +0.0147]**, p=0.011, δ=+0.12 → **tie** |
| Median paired difference (robust to the tail) | — | — | **+0.0023** |

No pair had *both* arms collapse. So the honest reading is: **the mule's jittery
AUC advantage is almost entirely that its sessions survive** — H0's long-range
backhaul fails outright 4× more often, and a failed session returns an untrained
model. Once a session trains at all, the two arms' models are statistically
indistinguishable at this scale (the compact IDS saturates near ~0.96–0.99).

That is a *stronger* claim than a vague accuracy edge, and it is the same
mechanism the participation metrics measure — which is why participation, not
AUC, remains the headline.

**Defensible framing:** *"When the backhaul is healthy, centralized flat FL is
strictly better — the mule is overhead. As devices lose their long-range path,
mule-based HERMES's participation advantage grows monotonically, from a tie to
decisive (Cliff's δ up to +0.98, p<0.001 at dead_zone=0.6). Its end-model AUC
advantage (+0.023 pooled) is predominantly (76 %) a matter of the session
completing at all — H0's session collapses 3.35 % of the time vs H1's 0.84 % —
rather than a better-trained model: conditional on training, the AUC difference
is not distinguishable from zero."* This is Observation 3 as a **conditional,
operating-regime-dependent** claim — the honest and far more defensible form,
now backed by 20 seeds across a 12-point surface rather than a single tuned cell.

**Caveats:** the 6-seed preview's "~5× participation, δ=+1.0" was a single
favorable cell (≈ dz=0.6); the 20-seed surface shows that point is real
(δ=+0.98 there) but *not* representative — the effect is a gradient, and near
the well-connected corner it reverses. Report the surface, not the corner.

## 7. Arm H3 — L1 adaptive channel selection (validity)

Arm **H3 = H2 + a real L1 adaptive channel**. It exercises the paper's
deterministic controller `U(c,t) = R(γ₁(t)+g(c)) − κ(c) − λ(c,t)`
(`hermes/l1/channel_utility.py`, the HERMES-Heuristic Eq. 1 the SEC26 audit
found was specified but never shipped as runtime code). An RF channel
environment (`experiments/exp4/channel.py`) gives each of `n_bands` a
time-varying effective SNR over the mission sequence and maps the chosen
channel's SNR to a per-mission **backhaul-loss probability**. The controller
picks a band each mission; the resulting loss schedule drives the cluster's
per-mission backhaul drop, and the chosen channel's mean SNR feeds the mule
selector's `rf_prior` feature (closing the L1→L2 edge the audit flagged).

**What makes it a fair, non-rigged test** — the same discipline as the
jittery remediation:

* **Bands cross over.** Each band peaks at a *different* time (distinct
  phases), so **no single fixed band is best throughout**. Adaptation can
  only help when the best band changes — the realistic time-varying case. If
  one band dominated, the fixed baseline would tie and L1 would earn nothing.
* **Fair static baseline.** H1/H2 hold `best_average_band` — the band a
  deployer picks from historical averages *without* real-time tracking (Exp-2's
  "Expected fixed"), **not** a retrospective per-instant oracle.
* **Identical conditions.** H2 and H3 read the same seeded SNR trace and share
  the backhaul-RNG seed, so only the per-mission loss thresholds differ.
* **Clean ⇒ ~no benefit, by construction.** Under clean links all bands sit
  high and stable, so fixed ≈ adaptive and L1's effect is correctly negligible.

### 7.1 What a 20-skeptic adversarial audit confirmed is sound

The H3 model was put through the same adversarial gauntlet as the jittery
H0/H1 work: five independent skeptics, each attacking a distinct validity
dimension (rigging / researcher-DOF, fair-baseline, loss-map, wiring-integrity,
robustness), every finding then independently verified by a second agent
tasked to *refute* it. **No rigging was found** (13 findings confirmed as
framing caveats, 0 blocking/major after verification, 2 refuted). What held up:

* **The direction is structural, not tuned.** Jittery reduction stays positive
  and clean stays ~0 across `n_bands` 2–5, `n_missions` 2–12, noise 0–5, base
  2–12, amp 1–10, and the full loss-map grid. The **median** jittery reduction
  over 400 random reasonable parameterizations is **+0.129 ≈ the reported
  +0.138** — the reported point is the *typical* genuinely-jittery outcome, not
  a cherry-picked corner.
* **The fixed baseline is fair — if anything generous.** Replacing
  `best_average_band` with the *strongest* fair fixed strategy (the
  loss-optimal fixed band, `argmin` mean loss — the exact objective, not mean
  SNR) still loses **+0.1344** (closes only 2.5 % of the gap, 30/30);
  median-SNR and worst-case-SNR baselines rank identically. No fixed band
  competes because a high-amplitude crossover trace forces every single band
  into deep loss troughs.
* **The wiring is correct.** `adaptive=(arm=="H3")` (H1/H2 hold the fixed band,
  only H3 adapts); the schedule reaches `cluster._backhaul_dropped` per
  mission; the shared `backhaul_rng_seed` makes H1/H2/H3 face the identical
  Bernoulli draws; `use_rl_selector` is True for **both** H2 and H3 (the
  selector is not a confound between them); `rf_prior` is a *live* selector
  feature (H2 ≈ 8 dB, H3 ≈ 12 dB).
* **It is robust at scale.** Jittery "adaptive never worse" survives
  **1000/1000** seeds and every horizon `n_missions` 2–20; the sign test gives
  **p = 1.9e-9**, three orders of magnitude below the p = 0.031 Wilcoxon floor
  that constrained the H0/H1 n = 6 work.

### 7.2 Six reviewer-grade caveats to state honestly

None of these falsify the result; each bounds how it may be framed. They are
stated here so the paper pre-empts the hostile reviewer rather than being
caught by them.

1. **This is a channel-model property; the end-to-end effect is now measured
   separately (§7.3).** The table below is `mean(backhaul_plan(...).loss_schedule)`
   computed in-process — the deterministic *input* the cluster consumes, and the
   same quantity the unit tests assert. It is **not** itself an FL measurement.
   The integrated H2-vs-H3 orchestrator sweep (§7.3) has since been run: it
   confirms the channel-model prediction end-to-end — a small, significant
   jittery gain, a clean null — so caveat 1 is now *addressed*, not outstanding.
   Still: quote the +0.14 as a channel/controller property and the §7.3 numbers
   as the FL result.
2. **The magnitude is calibration-dependent; the sign is the robust finding.**
   +0.14 is one point in a range spanning **~0.02–0.29** across defensible
   SNR/loss constants (`base`, `amp`, `mid`, `scale`). Quote it as "≈0.13–0.14
   *at this calibration*" and lead with the direction + clean/jittery contrast,
   not the point value.
3. **"Never worse" is near-guaranteed by construction at the chosen switch
   penalty.** The controller maximises `rate_tier(SNR)` while loss *decreases*
   in the same SNR on the same trace, so at `switch_cost → 0` adaptive is the
   per-instant SNR oracle and never-worse is a mathematical identity. It holds
   on a plateau `switch_cost ∈ [0, 1]` (30/30), softening to 29/30 (1.5), 26/30
   (2.0). `switch_cost = 0.5` is the dataclass default with no physical
   derivation — justify it (or report never-worse as a function of it) and
   present "never worse" as *near-tautological*, not a hard empirical
   separation.
4. **Clean is negligible, not an exact null.** At 1000 seeds it is a small,
   directionally-positive, *significant* win (~+0.0003, sign-test p = 0.001,
   897/1000 no-worse ≈ 90 %, worst −0.0008) — ~460× smaller than jittery. It is
   partly a logistic flat-top artifact: `mid = 3` maps healthy ~13 dB SNR to
   ~0 loss, hiding a real ~0.1 dB adaptation gap. State "a ~35× smaller
   adaptation gap, further compressed by the healthy-channel loss map," **not**
   "nothing to track."
5. **The metric excludes switching *outcome* loss.** `switch_cost` is charged
   only in the controller's *decision* utility; the realised `loss_schedule`
   never charges for actually switching bands (~2.8 switches / 6-mission
   trial). So +0.14 is an **upper bound assuming lossless switching**. It
   survives plausible per-switch upload penalties (0.05 → +0.12, 0.10 → +0.09,
   both never-worse) and flips only at an implausible ~0.30. The scope note now
   defers switching *energy* **and** switching-induced upload loss.
6. **"Delivers value" is now tested in the integrated sweep (§7.3).** The
   channel-model result shows the controller correctly *tracks* the best band
   under an imposed crossover; because SNR → loss is monotone on a shared trace,
   the *sign* is near-baked-in there. The H2-vs-H3 orchestrator sweep
   (SNR → loss → round-close, **non-monotone**) has now been run (§7.3): the L1
   value shows up as a small, significant jittery AUC/accuracy gain and a clean
   null — real but modest. Any such end-to-end delta is partly `rf_prior`-coupled
   (H3's higher chosen-SNR also nudges device selection); the gain is small
   enough that this confound cannot inflate it into a headline.

**Result (channel-model loss reduction, fixed − adaptive; 30 seeds,
`n_missions=6`, 3 bands — a property of the RF channel + controller, *not* an
orchestrator measurement):**

| regime | mean reduction | worst seed | best seed | adaptive ≤ fixed |
|---|---|---|---|---|
| clean | **+0.0003** | −0.0005 | +0.0020 | 29/30 (≈90 % at 1000 seeds) |
| jittery | **+0.1379** | +0.0540 | +0.2357 | **30/30** (1000/1000 at 1000 seeds) |

Codified deterministically in `tests/unit/test_exp4_channel.py` (31 fast
tests). The arm is proven to *run* over the real subprocess orchestrator in
`test_exp4_realmodel_smoke.py::test_exp4_h3_l1_channel_runs_end_to_end` — that
smoke test pins the wiring, **not** the +0.14 (see caveat 1).

**Defensible framing (audit-hardened):** *"Under a healthy backhaul, real-time
L1 channel adaptation is a wash (≈+0.0003, ~460× below the jittery effect);
under a jittery band-crossing channel it tracks the best band and cuts
*modelled* backhaul loss by ≈0.13–0.14 at this calibration, never worse across
1000 seeds. The sign is near-guaranteed by construction; the magnitude is
calibration-dependent and excludes switching overhead; and end-to-end the
effect is **small and not robustly detectable** — significant at a 6-mission
horizon (§7.3) but a tie across the whole dead-zone sweep at 4 missions
(§7.4)."* L1 is a **robustness mechanism, not an accuracy driver**.

**Scope note:** the channel model runs in-process in the driver and produces a
per-mission loss schedule the cluster applies. It does **not** yet unify
cross-layer *energy* accounting (L1 switching energy + L2 flight energy + L3
compute), and its loss metric does **not** charge switching-induced upload loss
(caveat 5) — both remain follow-ups.

### 7.3 Integrated end-to-end result (H3 vs H2, 20 paired seeds)

The pending orchestrator sweep is done — H3 (adaptive L1) vs H2 (fixed
best-average band) through the **real multi-process stack**, canonical CICIOT,
N=6, `--n-missions 6`, 20 paired seeds, jittery + clean
(`python -m experiments.analysis.exp4 --csv h2h3_l1.csv --treatment H3 --baseline H2`).

| regime | metric | H3 | H2 | H3−H2 | 95% CI | p | δ | verdict |
|---|---|---|---|---|---|---|---|---|
| clean | mission_completion_rate | 0.792 | 0.800 | −0.008 | [−0.033,+0.017] | 0.79 | −0.02 | tie |
| clean | round_close_rate@2 | 0.667 | 0.717 | −0.050 | [−0.092,−0.017] | 0.041 | −0.16 | H2 > H3 ✱ |
| clean | final_auc | 0.992 | 0.991 | +0.000 | [−0.001,+0.001] | 0.98 | −0.01 | tie |
| jittery | mission_completion_rate | 0.817 | 0.825 | −0.008 | [−0.067,+0.050] | 0.96 | −0.06 | tie |
| jittery | round_close_rate@2 | 0.600 | 0.575 | +0.025 | [−0.067,+0.117] | 0.40 | +0.07 | tie |
| jittery | final_auc | 0.987 | 0.975 | +0.012 | [+0.003,+0.021] | 0.044 | +0.41 | H3 > H2 ✱ |
| jittery | final_accuracy | 0.927 | 0.893 | +0.035 | [+0.004,+0.067] | 0.035 | +0.41 | H3 > H2 ✱ |

* **Clean is a null**, with one *tiny* `H2 > H3` on round-close (−0.05) — exactly
  the switch-cost tax caveats 3/5 predicted would surface as a small clean
  regression. Negligible and honest.
* **Under jittery, adaptive L1 buys a small, significant gain in the *converged
  model*** — final AUC +0.012 (p=0.044) and accuracy +0.035 (p=0.035), both
  δ=+0.41 — while participation (completion / yield / round-close) is a tie. The
  mechanism is exactly the model's: fewer lost backhaul rounds ⇒ slightly more
  aggregation lands ⇒ modestly better convergence; the effect is too small to
  move round-close at n=20 but does move end AUC/accuracy.
* This is the integrated evidence caveat 1 asked for — but it is **a single
  cell at one horizon**, and it does **not** replicate across the dead-zone
  sweep at a shorter horizon (§7.4). Read it together with §7.4, not alone.

### 7.4 The dead-zone L1 sweep — no detectable end-to-end effect

§7.3 tests one operating point (`dead_zone=0.0`, `n_missions=6`). A second,
wider sweep varies the terrain dead-zone at a shorter horizon
(`n_missions=4`, `dead_zone ∈ {0.0,0.2,0.4,0.6}` jittery + a clean reference,
20 paired seeds per cell, `results/exp4_paper/h2h3_dz_*.csv`). The result is
**null across the board**:

| condition | n | H2 | H3 | H3−H2 | 95% CI | p | verdict |
|---|---|---|---|---|---|---|---|
| clean | 20 | 0.8487 | 0.8634 | +0.0148 | [−0.0903,+0.1169] | 0.31 | tie |
| jittery dz=0.0 | 20 | 0.9100 | 0.9110 | +0.0010 | [−0.0919,+0.0918] | 0.60 | tie |
| jittery dz=0.2 | 20 | 0.7997 | 0.8674 | +0.0677 | [−0.0699,+0.2055] | 0.33 | tie |
| jittery dz=0.4 | 20 | 0.9490 | 0.9378 | **−0.0112** | [−0.0881,+0.0432] | 0.26 | tie |
| jittery dz=0.6 | 20 | 0.9531 | 0.9429 | **−0.0102** | [−0.0952,+0.0468] | 0.15 | tie |
| jittery pooled | 80 | 0.9029 | 0.9148 | +0.0118 | [−0.0383,+0.0576] | 0.045 | tie † |

† p<0.05 but the CI **includes 0**, so by the §5 rule (CI *and* Wilcoxon must
agree) this is a tie, not a claim.

Extending to `mission_completion_rate`, `update_yield` and
`round_close_rate@2`, **all 20 condition × metric tests are ties** — two of the
five AUC cells are even nominally negative. Data quality is not the excuse: this
sweep is 200/200 `ok` rows with **20/20 paired seeds in every cell** (see the
attrition note below).

**What this means — and what it retracts.** An earlier version of the layer-1
figure appeared to show a positive, severity-increasing L1 effect (Cliff's
δ +0.14…+0.37). That was an artifact of two defects, both now fixed: the
underlying sweep was contaminated (22/190 `ok` rows had produced no model at
all, their fabricated 0.0 participation biasing the means — §8), and the effect
size was computed **unpaired** on unequal-n groups, contradicting the paired
protocol of §5. On clean, properly paired data the effect disappears.

**Honest conclusion: no end-to-end accuracy benefit for L1 is claimed.** The
integrated effect is at best small and horizon-dependent — detectable in one
cell at `n_missions=6` (§7.3, +0.012, CI excluding 0), absent across five cells
at `n_missions=4`. Mechanistically that ordering is plausible (a longer horizon
means more per-dock backhaul draws, so a loss-rate difference has more chances
to accumulate into model quality), but one significant cell against five ties is
**not** a robust result and must not be reported as one.

**Defensible framing (both sections together):** *"Adaptive L1 channel selection
reduces modelled mule→BS backhaul loss under a jittery, band-crossing channel
(§7: ≈0.13–0.14 at this calibration, never worse across 1000 seeds), and is a
wash on a healthy backhaul. End-to-end, that advantage is small and not robustly
detectable: it reaches significance at a 6-mission horizon (+0.012 AUC) but is
statistically indistinguishable from zero across a 5-condition dead-zone sweep
at a 4-mission horizon. We therefore present L1 as a **backhaul-robustness
mechanism**, not as an accuracy driver."*

**Attrition note (why this sweep was re-run).** The first attempt lost ~30 % of
trials to `status=no_eval` — the mule's dock bootstrap failing under memory
pressure when 5 shards × ~8 TensorFlow processes ran concurrently. Failures were
arm-balanced (H2 10 / H3 9) and clustered in each shard's first trials, so they
cost statistical power rather than introducing bias — but under the pre-fix
driver they were recorded as `ok` and silently corrupted the participation
means. Re-running at concurrency 3 with staggered starts gave **0 failures in
200 trials**. Reproduce with
[`experiments/exp4/run_l1_deadzone_sweep.sh`](../experiments/exp4/run_l1_deadzone_sweep.sh)
(`MAX_PAR` caps concurrency; raise it only if the box has the RAM).

## 8. Remediation status vs the adversarial review

| Hole | Fix | Status |
|---|---|---|
| B1 no statistics | `analysis/exp4.py` (paired Wilcoxon + CI); multi-seed sweep | ✅ (20-seed run done, §6) |
| B2 hard-coded dead-zone / round-horizon | physical justification (above) + sweepable `--dead-zone`/`--link-quality` axes | ✅ (full 12-point surface, §6) |
| B3 asymmetric clean (H0 untaxed) | shared `device_reliabilities`; H0 pays the same tax | ✅ |
| B4 H1 penalty weaker than reference | per-mission backhaul loss now counts as non-close; **explicit scope note** that flight-budget is deferred | ✅ (scoped, not fully ported) |
| B5 `deadline_met` blind to H1 non-closure | derived from real cluster closure + backhaul-loss events | ✅ |
| §3 rf_factor softened | `world_radius` 150→100 (Exp-3 parity) | ✅ |
| §3 empty sortie hidden | distinct `mission_empty` event; still counted as a 0-update round | ✅ |
| SEC26 audit: `U(c,t)` controller specified but never shipped; `rf_prior` never wired at runtime | `hermes/l1/channel_utility.py` (arm H3) + `rf_prior_snr_db` threaded L1→selector | ✅ wired; channel-level effect §7, end-to-end effect **not robustly detectable** (§7.3 vs §7.4) |
| AUC drawn above 1.0 in the figures | symmetric ±SD whisker on a bounded metric → percentile bootstrap CI (bounded by construction) + per-seed strip; AUC axis bounded, then zoomed to [0.60,1.00] with `//` break marks and assertion guards (§5.1) | ✅ (data was always in range; max 0.9940) |
| Figures incommensurable with each other / with the tables | shared `figstyle.py` (one set of scales) + numbers taken from `analysis/exp4.py`; both figures now three panels on identical axes, incl. one fixed difference range (§5.1) | ✅ |
| Zero-evaluation trials recorded `status=ok` (blank AUC dropped, fabricated 0.0 participation averaged) | driver stamps `status=no_eval`; excluded from **every** metric; regression tests | ✅ (22/190 rows in the superseded sweep; clean re-run = 0/200) |
| Layer-1 figure built from a different sweep than the table it illustrated, with **unpaired** effect sizes | regenerated from the clean `h2h3_dz_*` sweep using `analysis/exp4.py` itself (same estimator as the tables), paired, with CIs | ✅ — and it **retracted** the apparent L1 effect (§7.4) |

## 9. Known limitations / open items

* The **≥20-seed sweep** (§6) and the **dead-zone × link surface** are **done**
  — 20 seeds × 12 jittery cells + clean, run in ~1 h as 6 parallel shards on a
  20-core box (`experiments/exp4/run_paper_sweep_parallel.sh`). The integrated
  **H2-vs-H3 L1** sweep (§7.3) is also done.
* H1's jittery model is **recoverable and per-mission** (architecturally
  correct for the two-pass design, which uploads once per dock), and omits
  flight-budget deadline pressure — strictly *weaker* than Exp-3's per-contact
  non-recoverable loss. This is a deliberate, stated scope choice, not a hidden
  advantage.
* Backhaul loss is drawn per dock, so at very short horizons its effect is
  small; longer `--n-missions` exercises it more.
* **Unified cross-layer energy** (L1 channel-switching + L2 flight + L3
  compute in one budget) is not yet modelled. Arm H3 wires L1 *adaptivity*
  and its backhaul-loss effect, but not its energy cost; a combined energy
  ledger is a separate follow-up.
* The H3 L1 effect is validated at the **channel-model level** (30 seeds,
  deterministic, §7) **and** end-to-end (20-seed H2-vs-H3 sweep, §7.3): a small
  significant jittery AUC/accuracy gain, clean null. Remaining L1 gaps are the
  switch-outcome-loss and energy items above, not the significance test.
