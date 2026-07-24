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

## 6. Honest current result (6 paired seeds, significance-tested)

Canonical CICIOT, N=6, `--n-missions 4`, 6 paired seeds. Paired Wilcoxon +
Cliff's δ + bootstrap 95% CI on the mean (H1 − H0) difference
(`python -m experiments.analysis.exp4 --csv <sweep>.csv`). ✱ = CI excludes 0
and Wilcoxon p<0.05.

| regime | metric | H1 | H0 | H1−H0 | 95% CI | p | δ | verdict |
|---|---|---|---|---|---|---|---|---|
| clean | final_auc | 0.903 | 0.895 | +0.008 | [−0.003,+0.018] | 0.31 | +0.33 | tie |
| clean | update_yield | 2.42 | 3.46 | −1.04 | [−1.50,−0.58] | 0.031 | −0.75 | H0 > H1 ✱ |
| jittery | final_auc | 0.903 | 0.887 | +0.016 | [−0.004,+0.038] | 0.44 | −0.06 | tie |
| jittery | mission_completion_rate | 0.75 | 0.25 | +0.50 | [+0.33,+0.67] | 0.031 | +1.00 | H1 > H0 ✱ |
| jittery | update_yield | 1.96 | 0.42 | +1.54 | [+1.29,+1.96] | 0.031 | +1.00 | H1 > H0 ✱ |
| jittery | round_close_rate@2 | 0.75 | 0.08 | +0.67 | [+0.46,+0.83] | 0.031 | +1.00 | H1 > H0 ✱ |

**Reading — the crossover is significant on _participation_, not on AUC.**

* **Participation crosses over decisively.** Centralized collects significantly
  *more* per round under clean links (update_yield 3.46 vs 2.42, p=0.031 — the
  mule's real short-range throughput cost), but under jittery the mule preserves
  **~5×** the participation (yield 1.96 vs 0.42; completion 0.75 vs 0.25;
  close-rate 0.75 vs 0.08), all at **Cliff's δ = +1.0 (perfect separation),
  p=0.031**. This is Observation 3 — a *participation* claim — validated
  end-to-end on the real integrated stack.
* **AUC is a statistical tie in both regimes** (CIs straddle 0). The compact
  DNN-IDS saturates near ~0.90 AUC on balanced CICIOT even from few updates, so
  H0's participation collapse does not tank its accuracy. **No significant AUC
  advantage is claimed from this run.**

**Defensible framing:** *"Centralized FL collects more updates per round when
the backhaul is clean, but mule-based HERMES preserves ~5× the federated
participation when the backhaul degrades (p=0.031, Cliff's δ=+1.0); the compact
IDS's end AUC is statistically indistinguishable at this scale."* This matches
the paper's Observation 3 (worded around participation, not accuracy) and does
not over-reach on AUC.

**Caveats:** 6 seeds with perfect separation hits the Wilcoxon floor p=0.031;
≥20 seeds would lower p / tighten CIs on the participation metrics and almost
certainly confirm the AUC tie. The dead-zone × link surface run is still
pending. Report participation (completion / yield / close-rate), not AUC, as the
Observation-3 evidence.

## 7. Remediation status vs the adversarial review

| Hole | Fix | Status |
|---|---|---|
| B1 no statistics | `analysis/exp4.py` (paired Wilcoxon + CI); multi-seed sweep | code ✅, ≥20-seed run pending |
| B2 hard-coded dead-zone / round-horizon | physical justification (above) + sweepable `--dead-zone`/`--link-quality` axes | ✅ (full surface run pending) |
| B3 asymmetric clean (H0 untaxed) | shared `device_reliabilities`; H0 pays the same tax | ✅ |
| B4 H1 penalty weaker than reference | per-mission backhaul loss now counts as non-close; **explicit scope note** that flight-budget is deferred | ✅ (scoped, not fully ported) |
| B5 `deadline_met` blind to H1 non-closure | derived from real cluster closure + backhaul-loss events | ✅ |
| §3 rf_factor softened | `world_radius` 150→100 (Exp-3 parity) | ✅ |
| §3 empty sortie hidden | distinct `mission_empty` event; still counted as a 0-update round | ✅ |

## 8. Known limitations / open items

* The full **≥20-seed sweep** and the **dead-zone × link surface** are compute
  jobs (each H1 trial spawns a real TF subprocess tree); run on the beefy
  node / Chameleon. The infrastructure and analysis are in place.
* H1's jittery model is **recoverable and per-mission** (architecturally
  correct for the two-pass design, which uploads once per dock), and omits
  flight-budget deadline pressure — strictly *weaker* than Exp-3's per-contact
  non-recoverable loss. This is a deliberate, stated scope choice, not a hidden
  advantage.
* Backhaul loss is drawn per dock, so at very short horizons its effect is
  small; longer `--n-missions` exercises it more.
