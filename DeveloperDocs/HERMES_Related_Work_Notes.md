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

**Updated 6 Oct 2026 for FeRRy and IPDPS 2027: §0 below.** §§1–8 are the August notes for HERMES,
kept as written. Their arm names `B1` (MAX-AoI) and `B2` (Oort) are `D1` and `D2` now, and §5's S3c
honesty note belongs to HERMES, not to FeRRy's mechanisms. The arms themselves, with their ports and
batch 1's numbers, are in [`HERMES_SOTA_Baseline_Candidates.md`](HERMES_SOTA_Baseline_Candidates.md) §0.

---

## 0. Current state — FeRRy (6 Oct 2026)

### 0.1 What changed since August

- **The system is now FeRRy.** It makes five claims (the build plan's ledger):
  - **C1 — reach is a decision:** the band class chosen before takeoff sets range and rate.
  - **C2 — one derived objective:** FedEx's bound plus a coverage term, used as the plan score, the reward and the merge weight.
  - **C3 — one deadline, three roles:** it admits, orders, and cuts off the merge.
  - **C4 — two clocks:** a plan at the dock, and the (band, stop) choice re-decided at each stop.
  - **C5 — fairness under physical cost.**
- **The comparison widened.** It was Oort, FedCS and MAX-AoI. It now adds a mobile-transporter prior (FedEx), a value-weighted freshness index (Cui), asynchronous-FL merge rules, UAV asynchronous FL (Zhai, Ho), and a learned UAV planner (Chen).
- **§1's taxonomy still holds, with one amendment.** The mule role is no longer nearly empty: FedEx (mobile transporters), Mestoukirdi (discrete stops) and Zhai (UAV-enabled asynchronous FL) work in or near it. The thesis, that most UAV-FL work restores connectivity while FeRRy substitutes for it, still frames the field. **But FedEx is a direct competitor, not an outsider, and must be treated as one** (§0.3).

### 0.2 Each claim and the work it answers

| Claim | The work it engages | The arm or study that tests it |
|---|---|---|
| C1 Reach is a decision | FedEx and most mule work fix reach (one link model); Zhai's oracle optimises band and route offline | FB+ (band pinned), O1 (oracle gap): Study 5.4 |
| C2 One derived objective | FedEx's convergence bound (the Δ term), Zhai's coverage term; hand-set rewards in DRL work (Ho, Chen) | F−dwell, F−cov, F·hand: Study 5.7 |
| C3 One deadline, three roles | FedCS (one round deadline), Oort (preferred duration), FedAsync and Yang (staleness cutoff), FedBuff, Async-HFL | F·round, F·pref, F·add: Study 5.2. The merge rules: Study 5.1 |
| C4 Two clocks | Chen (a learned per-step next stop), Ho (a monolithic DDPG agent) | E3, M1, FX against FQ: Studies 5.5 and 5.6 |
| C5 Fairness under physical cost | MAX-AoI, Oort's staleness bonus, Cui's Whittle index on value-weighted AoU | D1, D2, D3, F−cap, F−prio: Study 5.8 |

### 0.3 New threads

**Mobile-transporter FL — FedEx** (Bian, Shen, Chen, Xu, IEEE TMC 24(6), 2025).
- **The closest prior.** Transporters carry updates between devices and the server.
- **It assigns devices to transporters** (Gibbs sampling) and **plans each tour** (CARP, a 2-OPT visit-all tour).
- **It merges by 1/N accumulation** on return, under a travel-only convergence bound.
- **It has no deadlines, band choice, admission gate or re-plan, and it never skips a device.**
- **FeRRy's difference, stated as testable properties:** reach chosen per mission (C1); a bound extended with dwell, upload and coverage (C2); deadlines that admit and cut off (C3); per-stop re-decision (C4); an age cap under a budget (C5).
- **Batch 1's evidence, to be written as found (arm D4 is FedEx's route with FeRRy's merge; D4fedex has its own merge):**
  - At N = 6 with one mule, F reaches τ sooner than D4 at both budgets: 26% at the knee (194 against 264 s) and 33% under stress (178 against 264 s).
  - At N = 12 and 24, D4 ties F.
  - FedEx's own merge (D4fedex) is what falls behind: it takes 101 s against F's 52 s with three mules.
  - So the honest claim is about the merge, and about small fleets under binding budgets, **not** "FeRRy beats FedEx" in general.

**Value-weighted freshness — Cui et al.** (IEEE TMC 23(6), 2024).
- **What it is:** data-value-based asynchronous FL for UAV swarms under unstable links. Its Whittle index on value-weighted AoU is optimal when selection is free.
- **Ported as D3,** the fairness reference. Its sequential merge (`agg:seq`) was decided out, since it needs a protocol change.
- **The claim:** F stays within a stated margin of D3's Network AoU while closing rounds under a travel budget. **Never "fairer than Cui":** in Cui's model the index is optimal.

**Asynchronous FL and staleness-aware merging.**
- **FedAsync** (Xie et al., 2019): the staleness hinge FeRRy's `agg:cutoff` starts from.
- **Yang et al.** (IEEE JSAC 2025): the cutoff.
- **Shen et al.** (IEEE IoTJ 2024): the proximal term.
- **FedBuff** (Nguyen et al., AISTATS 2022): count-triggered buffering.
- **Async-HFL** (Yu et al., IoTDI 2023): hierarchical staleness decay.
- **Where this thread lives in the paper:** C3's merge role, tested in Study 5.1. FeRRy's angle is that the deadline that admits a device also sets its merge cutoff (one deadline, three roles), rather than a separately tuned staleness function.

**UAV-enabled asynchronous FL.**
- **Zhai, Yuan, Wang, Yang** (IEEE TWC 24(3), 2025): a coverage term for devices the deadline excludes, which FeRRy's objective adopts (C2), and an offline joint design that the O1 oracle follows for the optimality gap.
- **Ho et al.** (IAAA 2025): an energy-efficient DDPG agent for UAV-assisted asynchronous FL, which M1 follows as "one agent over everything, no gates".

**Learned UAV planning — Chen et al.** (IEEE GLOBECOM Workshops 2023).
- **What it is:** model-aided federated RL for multi-UAV trajectory planning.
- **Ported as E3:** a single-agent DQN choosing the next stop. The port has stops instead of grid moves, one agent, and no model-aided learning, and **it is never called FedQMIX.**
- **The context it sits in:** Study 5.5's calibration found FeRRy's own learned score at FX's level, not above it ([findings](Experiment_5_RL_Calibration_Findings.md)). The paper's learning claim depends on the sweep's verdict.

### 0.4 Guardrails added for FeRRy (on top of §6)

- **Do not claim** FeRRy beats FedEx across the board. Claim what batch 1 shows: F leads at N = 6 with one mule; FedEx's route ties it from N = 12; FedEx's own merge falls behind.
- **Do not say** "fairer than Cui". Say "within a margin of the index that is optimal in Cui's model, while paying for travel".
- **Do not call** E3 FedQMIX, or D5 FedCS without "degraded", or D2 Oort without "statistical utility" and its deviations.
- **Do not claim** that learning helps in flight unless Study 5.5's verdict reads rising. The calibration read flat, and FX is FeRRy's filling if the sweep agrees.
- **Do not claim** Byzantine robustness (a stated non-goal), or results from studies not yet run: H0, E3, O1, D5 and the aggregation rules are in batches 2–3.
- **Do not cite** the newer papers from the build plan's summaries alone. §7's new rows are not full-text checked here.

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
| **Added for FeRRy (Oct 2026)** — from the build plan's Sources (the novelty audit, 23 Sep 2026); this document records no full-text check | | |
| FedEx — Bian, Shen, Chen, Xu | IEEE TMC 24(6), 2025 | ⚠ read before citing (the closest prior; arm D4) |
| Data Value Based Async FL for UAV Swarm — Cui, Yang, Wu, Feng, Hu | IEEE TMC 23(6), 2024 | ⚠ read before citing (arm D3) |
| Model-Aided Federated RL for Multi-UAV Trajectory Planning — Chen, Esrafilian, Bayerlein, Gesbert, Caccamo | IEEE GLOBECOM Workshops 2023 | ⚠ read before citing (arm E3; a comparison note, `HERMES_vs_Chen2023_Model-Aided_FedQMIX.md`, is not in the repository) |
| UAV-Enabled Asynchronous FL — Zhai, Yuan, Wang, Yang | IEEE TWC 24(3), 2025 | ⚠ read before citing (coverage term; arm O1) |
| Energy-Efficient DDPG-Based UAV-Assisted Async FL — Ho et al. | IAAA 2025 | ⚠ read before citing (arm M1) |
| FedAsync — Xie et al. | 2019 (arXiv) | ⚠ read before citing |
| FedBuff — Nguyen et al. | AISTATS 2022 | ⚠ read before citing |
| Async-HFL — Yu et al. | IoTDI 2023 | ⚠ read before citing |
| Yang et al. (age cutoff; contextual bandit) | IEEE JSAC 2025 | ⚠ read before citing; full title not held here |
| Shen et al. (proximal term) | IEEE IoTJ 2024 | ⚠ read before citing; full title not held here |
| Chen et al. (mobility as mixing, the theory track) | IEEE TVT 2025 | ⚠ the full reference is not held in the repository |

## 8. Open items for the revision

**Added 6 Oct 2026 (FeRRy):**

- [ ] Read the new papers in §7 in full before citing any, starting with FedEx, the closest prior, and Cui. Record what each check changes, as §2 did for Oort and FedCS.
- [ ] Recover the novelty audit (Revision 3, 23 Sep 2026) and `HERMES_vs_Chen2023_Model-Aided_FedQMIX.md`. Both are cited by the build plan and absent from the repository.
- [ ] Find the full references for Yang (JSAC 2025), Shen (IoTJ 2024) and Chen (TVT 2025).
- [ ] Write §0.3's FedEx paragraph from batch 1's numbers. Revisit it once batch 2 adds E3, H0 and the aggregation rules.
- [ ] Re-check §0.4's learning guardrail against Study 5.5's verdict.

**From August:**

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
