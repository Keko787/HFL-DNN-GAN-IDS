# Related Work — staging notes for the revision

**Purpose.** Hold the reading, the verified findings, and the framing arguments in one place so the
Related Work revision is a *writing* task, not a re-reading task. This is **not** the section text —
it is the raw material plus the decisions about how to use it.

**Distinct from [`HERMES_SOTA_Baseline_Candidates.md`](HERMES_SOTA_Baseline_Candidates.md).** That
document answers *which baselines do we run*. This one answers *what do we say about the
literature*. A paper can be worth discussing here while being unusable as an arm there — and
several are.

**Prompted by** reviewer 74A's complaint: no recent UAV-FL baselines. The response is not simply
"add citations" — it is to show we know **why** most of that literature does not transfer, which is
a stronger position than pretending it does.

**Status:** findings verified where marked; citation-readiness tracked in §7. Nothing here is
drafted prose yet.

---

## 1. The organizing distinction — use this as the section's spine

Almost every UAV-FL paper puts the drone in one of three roles. The taxonomy is worth stating
explicitly in the section, because it does the argumentative work for us:

| Role | What the UAV is | Connectivity assumption |
|---|---|---|
| **UAV as client** | a flying data source that trains on its own imagery / RF captures | it *has* a link to the aggregator |
| **UAV as flying base station / relay** | infrastructure that hovers to serve ground devices | it **restores** connectivity that was missing |
| **UAV as data mule** ← **HERMES** | transport that carries updates physically between isolated devices and the base station | it **substitutes** for connectivity that never exists end-to-end |

> **The one-sentence version, and the thesis of the section:**
> *Most UAV-FL work uses the drone to **restore** connectivity; HERMES uses it to **substitute**
> for connectivity. A scheduler written for the first case assumes state that only exists in the
> first case.*

This is why the literature does not transfer, and saying it precisely converts an apparent gap
("you didn't compare against UAV-FL work") into a contribution ("that work presumes the link we
remove"). It also tells the reader exactly which prior work *does* transfer — the retrospective
ones — which is what keeps this from sounding like an excuse.

## 2. Thread — FL client selection (the general, heavily-cited line)

The comparators reviewers expect. All three **full-text verified**; see the baseline doc §2 and §6.

**The boundary that matters is not "learned vs heuristic" — it is *when the ranking signal is
obtained*.**

| Work | Ranking signal | When obtained | Ports to a mule? |
|---|---|---|---|
| **FedCS** (ICC 2019) | fits-in-deadline, maximise client count | **Before** selection — an explicit *Resource Request* step: clients report channel state, compute capacity, data size, **every round** | **No** — not without inventing the link |
| **Power-of-Choice `pow-d`** (AISTATS 2022) | highest local loss | **Before** selection — server ships the global model to candidates, who compute and return their loss | **No** |
| **Power-of-Choice `rpow-d`** | last reported loss, as a proxy | **Retrospectively** — reuses what a client sent when it last participated | **Yes** |
| **Oort** (OSDI 2021) | statistical utility `\|B_i\|·√(mean Loss²)` × system speed, plus a staleness bonus | **Retrospectively** — "a client's utility can only be determined *after* it has participated" | **Yes** |

**Two corrections to record, because the first-pass reading had them backwards:**

1. **FedCS is the one that cannot run on a mule**, despite being the closest in *spirit* to our
   deadline gates. Its Resource Request step is pre-selection reporting.
2. **Oort runs on exactly the state a mule has.** Dismissing it would have been a factual error
   about one of the most-cited selection papers in FL — and reviewers who know it would read that
   as not having read it, which is precisely 74A's complaint.

**How to use this in the section.** Lead with the timing boundary, not with a list. It lets us say:
we are compatible with the strong general baselines, and incompatible with one specific assumption —
pre-selection reporting — which our architecture denies by construction.

## 3. Thread — UAV-specific FL scheduling

**The closest architectural prior we found:**

**UAV-Aided Multi-Community Federated Learning** — Mestoukirdi, Esrafilian, Gesbert & Li, IEEE
GLOBECOM 2022 ([arXiv:2206.02043](https://arxiv.org/abs/2206.02043)). **Full-text verified.**

* The UAV flies a trajectory of **discrete stops**; devices transmit only when it is nearby. This
  is the *only* candidate found whose physical model matches a mule rather than a hovering relay.
* Device importance: `δ_k = p_k·ψ_c·λ` if the device failed or went unscheduled last round, else
  `p_k·ψ_c`, where `ψ_c` is the coefficient of variation of validation accuracy across community c.
* Trajectory and scheduling are **jointly optimised** (alternating sub-problems) — which is why it
  is a citation and not an arm.

**Discussed but not comparable** — record the reason with each, so the section shows judgement
rather than omission:

* **Reputation-based selection for UAV-assisted vehicular FL** (CJA 2024) — reputation over data
  quality and compute, maintained on a **consortium blockchain**, with an asynchronous-parallel RL
  resource scheduler. Faithful re-implementation is a different paper.
* **Fairness-Enhanced FL scheduling for UAV emergency communication** (Sensors 2024) — UCB bandit,
  reward `α·Ē + (1−α)·FM` with freshness `FM(m,t) = t − a·C_m`. Rule *is* implementable and needs no
  polling, but its UAV is a **hovering base station** serving all devices each round.
* **Joint trajectory + resource RL** (A3C / DRL placement work) — optimises the flight path itself,
  so a faithful port would replace the system under test.
* **Aggregation-side work** (FedWT MST-weighted aggregation, ClusterAvg, over-the-air aggregation)
  — no scheduling component; an L3 question, and Exp 4 already uses two-pass hierarchical FedAvg.
* **Byzantine-robust UAV FL** — orthogonal threat model; cite only if we make robustness claims,
  which we do not.
* **UAV anomaly detection under non-IID** — closest to our *application* (IDS) and useful for the
  non-IID framing, but contains no target-scheduling rule.

## 4. Thread — Age of Information / freshness

The closest *problem shape* to a data mule: which stale node to visit next, under travel cost.

* **MAX-AoI greedy** is an established named comparator — evaluations in this literature routinely
  report against "random, round-robin, periodic update, and MAX-AoI". The greedy form selects the
  highest-AoI device and recursively finds the nearest predecessor for the path. ✅ **Implemented
  as arm `B1`** (`hermes/scheduler/policies/max_aoi.py`, 2026-08-13). Its being standard is exactly
  why it is defensible.

  **What to write about it.** B1 shares H1's transport, realism and seeds and differs *only* in the
  ranking, so B1-vs-H1 isolates the scheduling policy. Two implementation choices are worth a
  sentence each because a careful reader will ask: a contact's age is its **stalest member** (max,
  not mean — peak AoI is what the greedy rule targets, and a mean lets a neglected device hide
  behind well-served neighbours in the same cluster), and a **never-served device is infinitely
  stale**, which is both correct AoI semantics and the explore-the-unvisited behaviour. Distance is
  a tie-break only, never overriding age.
* **Topology-coupled urgency scheduling** (UAV swarm IoT collection) weights each cluster's AoI
  urgency by a connectivity score.
* AoI minimisation in UAV-aided collection is an established review area, so an AoI-greedy baseline
  needs no special justification.

## 5. Thread — starvation and fairness *(this is where our contribution is positioned)*

**The most useful narrative thread found, and it was not in the original scan.** Three independent
lines of work all encounter the same failure — *a device the scheduler keeps passing over is never
served again* — and each answers it differently:

| Work | Mechanism against starvation |
|---|---|
| **Oort** (OSDI 2021) | additive staleness bonus `Util(i) ← U(i) + 0.1·log(R)/√L(i)`, where `L(i)` is the last round i participated |
| **UAV multi-community** (GLOBECOM 2022) | multiplicative importance penalty `λ` on devices that failed or went unscheduled |
| **Fairness-enhanced UAV FL** (Sensors 2024) | freshness term `FM(m,t) = t − a·C_m` inside a UCB reward |
| **HERMES** | **per-device** window widening on a missed contact (Φ), extended by **Amendment 1 (A2)** to devices the S3b gate dropped or an abort abandoned, plus **Amendment 2 (S3c)** mission-level widening when the mule is systematically falling short |

**Why this framing is worth the space.** It positions our starvation work as *participating in an
established conversation* rather than inventing a problem. And it makes our actual novelty precise
and modest enough to defend:

* the prior mechanisms are all **per-device utility adjustments** — they make a neglected device
  more attractive;
* ours adds a **mission-level** signal, because a per-device rule cannot distinguish "this device is
  unlucky" from "the circuit is systematically infeasible" — from any one device's view those look
  identical;
* and our A2 case is one **the others do not have**: a device dropped by a *feasibility gate* never
  opens a session at all, so it generates no feedback event of any kind. That failure is created by
  having a hard gate, which the utility-ranking approaches do not have.

> **Honesty constraint on this paragraph — updated after the pilot ran (2026-08-13).**
>
> The pilot (checklist §5.0a) found a **narrow, transient** effect, and the framing above must
> match it. What is defensible to write:
>
> * S3c raised **update yield +0.194** (CI [+0.063, +0.313], p = 0.0178, δ = +0.278 *small*) at one
>   operating point with the deadline gate binding — but that **does not survive correction for
>   testing 8 metrics** (Bonferroni α = 0.00625). *Suggestive, not established.*
> * **Mission completion moved the other way** (−0.025, n.s.). Report the **trade-off**, not a win.
> * The mechanism *was* independently verified from the traces: round 1 identical across arms
>   (no history ⇒ scale exactly 1.0), divergence at rounds 2–4, convergence after.
> * **The honest claim is about the warm-up, not the steady state:** S3c reaches a workable window
>   *faster*; the per-device rule gets there on its own given enough missions. Which predicts the
>   advantage shrinks as mission count grows — sharp, falsifiable, and a better sentence than the
>   raw effect size.
>
> Until the confirmatory `n_missions` ladder runs, keep this as **design rationale with a measured
> illustration**, not as a headline result.

## 6. What we should **not** claim

Guardrails, so the revision does not overreach in either direction:

* **Do not claim** UAV-FL scheduling is unstudied. It is well studied — for a *different*
  architecture. The taxonomy in §1 is the honest framing.
* **✅ Implemented as arm `B2` (2026-08-13) — and do not call it "Oort".** Scoping it against the code (checklist §5.1a) found we
  can port its **statistical utility + staleness** — the parts needing only retrospective,
  mule-visible state — but **not its system-speed straggler penalty**, because no per-device compute
  speed exists in our model. Our loss is also the **mean** where Oort specifies the **RMS** over
  per-sample losses: monotone in the same direction, not identical. Describe the arm as *"Oort's
  statistical-utility selection"* with both deviations stated. Overclaiming exactness here is the
  same error we just corrected in the other direction.
* **Do not claim** Oort or Power-of-Choice are inapplicable to our setting. `rpow-d` and Oort port
  directly; **FedCS** and `pow-d` do not. State the boundary, not a blanket dismissal.
* **Do not claim** measured benefit for S3c, deadline enforcement, or the in-flight abort. All three
  are off in every committed result.
* **Do not cite** anything marked ⚠ in §7 without reading it first.
* **Do not imply** the starvation problem is novel. Our *mission-level* response and the
  *gate-induced* case are the contributions; the problem is shared.

## 7. Citation readiness

| Work | Venue | Verified? |
|---|---|---|
| FedCS — Nishio & Yonetani | IEEE ICC 2019 · [doi:10.1109/ICC.2019.8761315](https://doi.org/10.1109/ICC.2019.8761315) · [arXiv:1804.08333](https://arxiv.org/abs/1804.08333) | ✅ full text |
| Oort — Lai et al. | USENIX OSDI 2021 · [arXiv:2010.06081](https://arxiv.org/abs/2010.06081) | ✅ full text |
| Power-of-Choice — Cho, Wang & Joshi | AISTATS 2022, PMLR 151:10351–10375 ([proceedings](https://proceedings.mlr.press/v151/jee-cho22a.html)); preprint [arXiv:2010.01243](https://arxiv.org/abs/2010.01243) | ✅ full text — ⚠ **published retitled** *Towards Understanding Biased Client Selection in Federated Learning*. Cite AISTATS for the venue, but the `pow-d`/`rpow-d` labels come from the preprint; **confirm the naming survived** before citing `rpow-d` by name |
| UAV-Aided Multi-Community FL — Mestoukirdi et al. | IEEE GLOBECOM 2022 · [arXiv:2206.02043](https://arxiv.org/abs/2206.02043) | ✅ full text |
| Fairness-Enhanced UAV FL | Sensors 2024 · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934714/) | ✅ full text (MDPI 403s; PMC mirror works) |
| Vehicular reputation FL | Chinese J. Aeronautics 2024 · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S100093612400236X) | ⚠ abstract + summaries only — enough to exclude, **not to cite** |
| AoI in UAV-aided collection (review) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1084804523000711) | ⚠ not read in full |
| UAV swarm AoI / topology-coupled urgency | [arXiv:2608.00061](https://arxiv.org/abs/2608.00061) | ⚠ description confirmed, not read in full |
| Data-Efficient Energy-Aware Participant Selection | — | ⚠ **not cleared** |
| Privacy-Preserving FL for UAV (A3C) | — | ⚠ **not cleared** |
| Contribution-Based Resource Allocation | — | ⚠ **not cleared** |
| Broad UAV-FL sweep (client / relay / aggregation / Byzantine) | — | ⚠ triaged **by architecture class** from abstracts — enough to exclude as arms, **not** to cite |

## 8. Open items for the revision

- [ ] Confirm the `rpow-d` naming appears in the AISTATS version (§7) — our timing-boundary argument
      names it.
- [ ] Decide how much space the §1 taxonomy gets. Recommendation: a short paragraph plus the
      three-role distinction — it earns its length by making the rest of the section short.
- [ ] Read anything marked ⚠ that survives into the final citation list.
- [ ] Re-check §5's honesty constraint once the S3c pilot has run — if it shows a measured effect,
      that paragraph can be upgraded from rationale to result.
- [ ] Cross-check against the paper/code divergences recorded in the architecture review (D-1 the
      Fig. 2 vs §III-A disagreement on the RF selector, D-2 the GAN contribution never executing,
      D-3 scheduling results coming from the sim rather than the multi-process topology) — Related
      Work should not describe capabilities the results section does not demonstrate.
