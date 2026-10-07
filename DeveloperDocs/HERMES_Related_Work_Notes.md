# Related Work — staging notes for the revision

**Purpose.** Hold the reading, the verified findings, and the framing arguments in one place so the
Related Work section is a *writing* task, not a re-reading task. This is **not** the section text —
it is the raw material plus the decisions about how to use it.

**Distinct from [`HERMES_SOTA_Baseline_Candidates.md`](HERMES_SOTA_Baseline_Candidates.md).** That
document answers *which baselines do we run* (its §0 has the arms, their ports and batch 1's
numbers). This one answers *what do we say about the literature*. A paper can be worth discussing
here while being unusable as an arm there — and several are.

**Prompted by** reviewer 74A's complaint (SEC 2026): no recent UAV-FL baselines. The response is not
simply "add citations" — it is to show we know **why** most of that literature does not transfer,
which is a stronger position than pretending it does.

**Status, 6 Oct 2026 — rebuilt for FeRRy and IPDPS 2027.**
- **Reading status:**
  - Read in full in August: FedCS, Oort, Power-of-Choice, Mestoukirdi et al. and the Sensors fairness paper.
  - Read in full on 6 Oct: FedEx, Cui et al., Zhai et al., Chen et al., FedAsync, FedBuff and Async-HFL.
  - Ho et al.: abstract only.
- **Where a reading contradicted how the build plan, the SOTA document or the code describes a paper,** the correction is stated in place and gathered in §6a.
- **August material written for HERMES** is kept where it still holds and marked where FeRRy superseded it.
- **The arm names changed:** `B1` and `B2` are `D1` (MAX-AoI) and `D2` (Oort) now.
- **Nothing here is drafted prose yet.**

---

## 0. FeRRy's claims and the work each one answers

FeRRy's five claims (the build plan's ledger), the work each engages, and the arm or study that
tests it:

| Claim | The work it engages | Tested by |
|---|---|---|
| **C1 Reach is a decision.** The band class chosen before takeoff sets range and rate; range sets the stops | FedEx assumes one fixed line-of-sight rate (§3.1); Zhai fixes a free-space channel (§3.3); no prior work chooses the band per mission | FB+ (band pinned), O1 (offline oracle): Study 5.4 |
| **C2 One derived objective.** FedEx's convergence bound, extended with dwell, upload and a coverage term, used as the plan score, the reward and the merge weight | FedEx's async bound, Σ_k R_k·Δ_k² (§3.1); Zhai's device-selection penalty (§3.3); hand-set rewards in learned UAV work (§5b) | F−dwell, F−cov, F·hand: Study 5.7 |
| **C3 One deadline, three roles.** It admits (S3b), orders (S3), and cuts off the merge weight (L3) | FedCS's one round deadline and Oort's preferred duration (§2); FedAsync's staleness function, FedBuff, Async-HFL, Yang's cutoff (§5a) | F·round, F·pref, F·add: Study 5.2. The merge rules: Study 5.1 |
| **C4 Two clocks, re-decided per stop.** A plan at the dock, then the (band, next stop) pair chosen at each stop | Chen's learned UAV trajectories, Ho's single DDPG agent, CEDA's monolithic multi-drone DQN (§5b) | FX against FQ, E3, M1: Studies 5.5 and 5.6 |
| **C5 Fairness under physical cost.** A coverage term, a hard age cap S, missed devices rising in rank, Network AoU reported | MAX-AoI, Oort's staleness bonus (§2, §4); Cui's Whittle index on value-weighted AoU (§4); Zhai's serve-every-S-slots constraint (§3.3, §5) | D1, D2, D3, F−cap, F−prio: Study 5.8 |

---

## 1. The organizing distinction — use this as the section's spine

Almost every UAV-FL paper puts the drone in one of a few roles. The taxonomy is worth stating
explicitly in the section, because it does the argumentative work for us. Updated 6 Oct with where
each paper read actually sits; several sit somewhere other than their titles suggest.

| Role | What the UAV is | Connectivity assumption | Who sits here (read) |
|---|---|---|---|
| **UAV as client** | a flying data source that trains on its own data | it *has* a link to the aggregator, if an unreliable one | **Cui et al.**: the UAVs are the FL clients, around one leader UAV that serves as the server; "unstable" means each client connects with probability ρ_i per iteration (§4) |
| **UAV as flying base station / aggregator** | infrastructure that hovers or flies to serve ground devices | it **restores** connectivity that was missing | **Zhai et al.**: the UAV is a flying parameter server, aggregating over the air (§3.3). The Sensors fairness paper: a hovering base station. **Ho et al.**: likely here; the abstract is unclear |
| **UAV as raw-data collector** | flies past devices and downloads their sensor data | it **restores** a link, for raw data rather than models | **Chen et al.**: UAVs harvest data units from IoT buffers; the federated part is RL across UAVs, not FL of a task model (§5b) |
| **UAV as data mule** ← **HERMES / FeRRy** | transport that carries model updates physically between isolated devices and the server | it **substitutes** for connectivity that never exists end to end | **FedEx** (mobile transporters, §3.1), **Mestoukirdi et al.** (discrete stops, §3.2) |

> **The thesis of the section, updated:** most UAV-FL work uses the drone to **restore**
> connectivity, or flies it as a client; FeRRy uses it to **substitute** for connectivity. A
> scheduler written for the first cases assumes state that exists only there. **But the mule role
> is no longer empty.** FedEx works in it directly, so FedEx is a competitor to beat, not an
> outsider to distinguish (§3.1).

This is why most of the literature does not transfer, and saying it precisely converts an apparent
gap ("you didn't compare against UAV-FL work") into a contribution ("that work presumes the link we
remove"). It also says exactly which prior work *does* transfer: the retrospective selection rules
(§2), and FedEx.

---

## 2. Thread — FL client selection: *when* the ranking signal is obtained

The comparators reviewers expect. The boundary that matters is not "learned against heuristic"; it
is **when the ranking signal is obtained**. The rows added on 6 Oct are from full-text reading.

| Work | Ranking signal | When obtained | Ports to a mule? |
|---|---|---|---|
| **FedCS** (ICC 2019) | fits in one round deadline, maximise client count | **Before** selection: an explicit *Resource Request* step, every round | **No**, not without inventing the link. Ported **degraded** as D5 (last-known state for the request) and as the deadline form F·round (Study 5.2) |
| **Power-of-Choice `pow-d`** (AISTATS 2022) | highest local loss | **Before** selection: the server ships the model to candidates, which compute and return their loss | **No** |
| **Power-of-Choice `rpow-d`** | last reported loss, as a proxy | **Retrospectively** | **Yes** (cite only; an optional arm if a reviewer asks) |
| **Oort** (OSDI 2021) | statistical utility × system speed, plus a staleness bonus | **Retrospectively**: "after it has participated" | **Yes**. Ported as D2 (statistical utility only), and its preferred duration as F·pref (Study 5.2) |
| **Cui et al.** (TMC 2024) | Whittle index on value-weighted Age of Updates, from (age, connected) | Connection state Λ **observed before** selection; data value from a **separate pre-training stage** (Shapley value, scored on a test set held at the leader); ρ_i assumed known | **Partly.** A mule cannot see who is connected before it flies there, and has no pre-training stage. Ported as D3 with ω from Oort's utility, ρ from reachability history, and an **"expected" index of our own** (§4, §6a) |
| **Zhai et al.** (TWC 2025) | joint offline choice of trajectory, selection and transmit amplitudes | **In advance**: full channel-state information assumed (eq. 7); solved before flight | **No** as a run-time rule; an offline design. The oracle O1 follows it only loosely (§3.3) |
| **FedEx** (TMC 2025) | device-to-transporter assignment and tour | **In advance**: from known positions, speeds and energy parameters; no client reports, no run-time state, no re-plan | **Yes**: it is a mule design. Never skips a device (§3.1) |
| **Async-HFL** (IoTDI 2023) | gateway-level ILP over learning utility (from PCA-compressed recent gradients), latency and data rate | **Periodically reported** by devices to gateways | **No**: it needs gradient reports from devices the mule has not reached |

**Two corrections recorded in August**, kept because the first pass had them backwards:
1. **FedCS is the one that cannot run on a mule**, despite being the closest in *spirit* to our deadline gates. Its Resource Request step is pre-selection reporting.
2. **Oort runs on exactly the state a mule has.** Dismissing it would have been a factual error about one of the most-cited selection papers in FL.

**How to use this in the section.** Lead with the timing boundary, not with a list. It lets us say:
we are compatible with the strong retrospective baselines, and incompatible with one specific
assumption, pre-selection reporting, which our architecture denies by construction.
- **The 6 Oct rows add a third category:** decisions computed **in advance with full information** (Zhai, FedEx). Those port only as offline plans.
- **FeRRy re-plans at the dock and re-decides at each stop** from what it measured, which is what C4 claims.

---

## 3. Thread — UAV-specific FL scheduling and mobile transporters

### 3.1 FedEx — the closest prior, and a competitor

**Bian, Shen, Chen, Xu, *Indirect-Communication Federated Learning via Mobile Transporters*, IEEE
TMC 24(6), 2025.** Read in full (the published version, 6 Oct).
- **An earlier version:** Bian, Shen and Xu, CISS 2023 (arXiv:2302.07323), the same protocol without the energy model.
- **arXiv:2304.10744 is not a separate paper:** it is the TMC article's preprint.

**What FedEx is.**
- **The setup:** one fixed server, N fixed clients, and **no client–server or client–client link at all.** K transporters (UAVs) each fly a closed tour over a disjoint set of clients.
- **At a visit:** the client downloads the global model the transporter picked up at the start of the tour. It uploads its cumulative update, trained since the previous visit on the previous tour's model, so **every update is one tour stale.** Local training fills the whole gap between visits.
- **The merge:** the transporter keeps a running sum of the update deltas, and the server applies **x ← x − (1/N)·Σ updates**, when the round ends (FedEx-Sync) or when each transporter returns (FedEx-Async). That is delta accumulation with no staleness weighting, not model averaging.
- **Assignment and routing (CARP):** computed **offline** from positions, speeds and energy.
  - The inner level is a 2-OPT shortest tour (a TSP).
  - The outer level is Gibbs sampling of each client's transporter, gated by a per-trip energy budget.
  - The cost is the slowest round trip (Sync) or Σ_k R_k·Δ_k² (Async).
  - **Every assigned client is visited on every tour; none is ever skipped.**
- **Convergence:**
  - The Sync error grows with the square of the slowest round trip.
  - The Async error grows with (1/N)·Σ_k R_k·Δ_k² (Theorem 2, eq. 24), which is **the bound FeRRy's C2 extends.**
  - Travel and hover time enter only through the round trips.
- **The link:** a fixed line-of-sight rate, so every transfer takes the same time.
- **Absent:** deadlines, time budgets, link variation, band or rate choice, upload failures, admission, skipping and re-planning.
- **Evaluation:** 40 clients, 4 transporters, FMNIST and SVHN. The only baselines are other assignment objectives; it compares against no other FL method.

**FeRRy's difference, as testable properties.** Each is a property FedEx lacks, and each has a
study:
- reach chosen per mission (C1, Study 5.4);
- a bound extended with dwell, upload and coverage (C2, Study 5.7);
- deadlines that admit and cut off (C3, Studies 5.1 and 5.2);
- re-decision at each stop on measured signal (C4, Studies 5.5 and 5.6);
- an age cap under a budget (C5, Study 5.8).

**How we port it (arm D4), and what that tests.** The port (`hermes/scheduler/policies/fedex_carp.py`) declares 13 deviations. The two that matter here:
- **One transporter per mission.** The device-to-mule split is made upstream, so **FedEx's Gibbs assignment is never exercised**, and D4 is CARP's inner level: a visit-all 2-OPT tour.
- **No energy gate.** The simulator has no energy model, so D4 is the conference version's unconstrained problem.

D4 runs with FeRRy's merge (`agg:cutoff`, arm "D4") and with FedEx's own (`agg:fedex`, arm "D4fedex"), so that route and merge are judged separately. **D4 tests FedEx's tour and FedEx's merge, not its assignment.**

**What batch 1 found (6 Oct; time to τ = 0.71, simulated seconds):**

| Cell | F | D4 (FedEx tour, FeRRy merge) | D4fedex (FedEx tour and merge) |
|---|---|---|---|
| N = 6, 1 mule, knee | **194** | 264 (claim for F) | 281 (n.s.) |
| N = 6, 1 mule, stress | **178** | 264 (claim for F) | 281 (n.s.) |
| N = 6, 3 mules | **52** | 55 (n.s.) | 101 (claim for F) |
| N = 12, 1 mule | 486 | 459 (n.s.) | — |
| N = 24, 1 mule | 749 | 752 (n.s.) | — |

**Write it as found:**
- **FeRRy's route beats FedEx's tour at N = 6,** by 26% at the knee and 33% under stress.
- **From N = 12 FedEx's tour keeps pace,** so the claim against FedEx is not general.
- **FedEx's own merge is what falls behind,** taking 101 s against F's 52 s with three mules. That is the attribution the two-arm design was built to make, and it points at C3 (the deadline-derived merge) rather than at the route.

### 3.2 Mestoukirdi et al. — the closest *physical* prior before FedEx (August)

**UAV-Aided Multi-Community Federated Learning**: Mestoukirdi, Esrafilian, Gesbert & Li, IEEE
GLOBECOM 2022 ([arXiv:2206.02043](https://arxiv.org/abs/2206.02043)). **Read in full.**

* The UAV flies a trajectory of **discrete stops**; devices transmit only when it is nearby.
* Device importance: `δ_k = p_k·ψ_c·λ` if the device failed or went unscheduled last round, else
  `p_k·ψ_c`, where `ψ_c` is the coefficient of variation of validation accuracy across community c.
* Trajectory and scheduling are **jointly optimised** (alternating sub-problems), which is why it
  is a citation and not an arm.

### 3.3 Zhai et al. — a flying aggregator with an age constraint

**Zhai, Yuan, Wang, Yang, *UAV-Enabled Asynchronous Federated Learning*, IEEE TWC 24(3), 2025.**
Read in full (the published version and arXiv:2403.06653v1, 6 Oct).

**The role is not a mule.** The UAV is a **flying parameter server**.
- **The aggregation:** ground devices send gradients by **over-the-air computation** on one shared resource, over a free-space line-of-sight channel, with full channel-state information assumed. In each slot the UAV takes the **plain mean** over the selected devices that are both computation-ready and in line of sight.
- **What "asynchronous" means:** the gradients in one aggregation come from different model versions. The communication itself is synchronous (Remark 3).
- **The convergence bound:** it splits the error into model asynchrony, device selection and communication error. The selection term penalises unselected devices: 24·((M − |selected|)/M)².
- **The design problem:** trajectory, selection and transmit amplitudes are chosen jointly **offline**, with a constraint that **every device be served at least once in every S slots** (31b). It is solved by alternating convex approximation to a local optimum, not exhaustively.
- **Absent:** deadlines, band or link choice, staleness weighting or cutoff, and energy.

**What we take from it, stated accurately:**
- **The serve-every-S-slots constraint is precedent for FeRRy's hard age cap S** (C5). Cite it there.
- **The selection penalty is the closest analogue of FeRRy's coverage term,** which is about devices the deadline excludes. Zhai has no deadlines, so call the coverage term "related to Zhai's selection penalty", **not "Zhai's coverage term".**
- **The oracle O1 is "after Zhai" only in being an offline, full-information joint design of route and selection.** Zhai searches no band classes, no per-stop band and no clustering, and its optimum is local.

### 3.4 Discussed but not comparable (August, extended)

Record the reason with each, so the section shows judgement rather than omission:

* **Reputation-based selection for UAV-assisted vehicular FL** (CJA 2024): reputation on a **consortium blockchain** with an asynchronous-parallel RL resource scheduler. Faithful re-implementation is a different paper.
* **Fairness-Enhanced FL scheduling for UAV emergency communication** (Sensors 2024): a UCB bandit with a freshness term. The rule *is* implementable, but its UAV is a **hovering base station** serving all devices each round.
* **Joint trajectory and resource RL** (A3C / DRL placement work): optimises the flight path itself, so a faithful port would replace the system under test.
* **Aggregation-side work** (FedWT, ClusterAvg, over-the-air aggregation): no scheduling component. Zhai's over-the-air design belongs here as much as in §3.3.
* **Byzantine-robust UAV FL**: an orthogonal threat model, and a stated non-goal of FeRRy.
* **UAV anomaly detection under non-IID data**: closest to our *application* (IDS), with no target-scheduling rule.
* **Ho et al.** (IAAA 2025, abstract only): energy-minimising control of UAV movement, CPU frequency and transmit power. Nothing read shows it choosing who is served (§5b).

---

## 4. Thread — Age of Information and freshness

The closest *problem shape* to a data mule: which stale node to visit next, under travel cost.

**MAX-AoI greedy** (D1, was `B1`) is an established named comparator. AoI evaluations routinely
report against random, round-robin, periodic update and MAX-AoI.
- **How ours differs from H1:** D1 shares H1's transport, realism and seeds and differs *only* in the ranking, so D1 against H1 isolates the scheduling policy.
- **A contact's age is its stalest member,** and a never-served device is infinitely stale. Distance is a tie-break only. Since Phase 0, age runs from the last CLEAN contact.
- **Batch 1:**
  - At N = 6 with one mule at the knee, D1 reached τ in 282 s against F's 194 (a claim for F).
  - Under the stress budget, 238 against 178, not significant.
  - D1 equals H1 at the knee, because with nothing for the budget to cut both serve the same devices.

**Cui et al.: Network Age of Updates and a Whittle index** (D3). Read in full (the published version, 6 Oct).
- **The setup:** UAV *clients* around a leader. One connected client is chosen per iteration, receives the latest model, runs n steps and returns it (sequential SGD routed through the leader).
- **The data value ω_i:** a normalised Shapley value estimated in a pre-training stage.
- **The objective:** Network AoU = the long-run average of Σ ω_i·AoU_i, with AoU_i = iterations since client i was last selected.
- **The index:** the problem is a restless bandit, decoupled by a Lagrangian. Each subproblem has an optimal threshold policy (Theorem 6), is indexable (Theorem 7), and has a closed-form index (eq. 48): I(x, 0) = 0 for a disconnected client, I(x, 1) = (ω/2)x² − (ω/2)x + (ω/ρ)x for a connected one.
- **Results:** with 8 UAVs on MNIST and FLAME, the index policy beat myopic, greedy-AoU and greedy-value selection, with 80.5% accuracy against 77, 67 and 60%.
- **Absent:** deadlines, travel, budgets, band choice and staleness cutoffs.

**How to cite Cui accurately (corrections in §6a):**
- **Not "optimal".** The optimality is per decoupled subproblem; the coupled problem is PSPACE-hard, and the paper compares only against heuristics. Say "the index policy Cui et al. derive, the best of the heuristics they test, when one connected client is served per iteration at no travel cost".
- **Not "device-to-device".** The sequential aggregation runs through the leader.
- **D3 flies our "expected" index by default.** That is ρ·I(x, 1), averaged over a connection the mule cannot observe before it flies (`hermes/scheduler/policies/whittle.py`, marked "not in the paper"). The "literal" variant is Cui's eq. 48. **Batch 1's D3 numbers are the expected variant's;** state it wherever D3 is reported.
- **Batch 1:**
  - At N = 6 with one mule at the knee, D3 took 280 s against F's 194 (a claim for F).
  - At N = 24, 911 against 749 (a claim).
  - Under stress, 209 against 178, not significant.

AoI minimisation in UAV-aided collection is an established review area, so an AoI-greedy baseline
needs no special justification.

---

## 5. Thread — starvation and fairness *(where FeRRy's C5 is positioned)*

Several independent lines of work meet the same failure: *a device the scheduler keeps passing over
is never served again*. Each answers it differently:

| Work | Mechanism against starvation | Kind |
|---|---|---|
| **Oort** (OSDI 2021) | additive staleness bonus `Util(i) ← U(i) + 0.1·log(R)/√L(i)` | per-device utility adjustment |
| **Mestoukirdi et al.** (GLOBECOM 2022) | multiplicative penalty `λ` on devices that failed or went unscheduled | per-device utility adjustment |
| **Fairness-enhanced UAV FL** (Sensors 2024) | freshness term `FM(m,t) = t − a·C_m` inside a UCB reward | per-device utility adjustment |
| **Cui et al.** (TMC 2024) | Whittle index on value-weighted AoU: age raises the index | per-device index. **Argues against hard minimum-selection constraints** as inefficient under randomness |
| **Zhai et al.** (TWC 2025) | **every device served at least once in every S slots** (31b), plus a selection penalty in the bound | **hard constraint** in an offline design |
| **CEDA** (lab thesis, 2026; §5b) | a weight-agnostic penalty per unserved patient, inside a learned policy's reward | reward shaping |
| **HERMES** (as submitted, Aug) | per-device window widening on a miss (Φ), Amendment 1 (A2) for gate-dropped devices, Amendment 2 (S3c) mission-level widening | per-device plus mission-level adjustment |
| **FeRRy** (Exp 5) | a **hard age cap S** (a capped device must be served; the S\* tool sets S = 2), a **coverage term in the plan score** for devices left out, and a **priority boost after a miss** | **hard constraint plus objective term, enforced in the plan** |

**The framing, updated for FeRRy.**
- **HERMES only adjusted ranks,** and its fairness metrics were measured, never enforced (the gap CEDA's comparison names). FeRRy **enforces**: the cap is a constraint, the coverage term is in the objective, and the plan must meet both under a travel budget.
- **That puts FeRRy between two published positions.** Zhai imposes a hard serve-every-S constraint offline with full information. Cui argues such constraints are inefficient and uses an index. FeRRy imposes the cap on-line, under travel cost and uncertain contacts.
- **Study 5.8 tests the tension directly:** F against F−cap (no cap) and F−prio, with D3 as the fairness reference. Cap violations are reported by cause, with no pass mark.
- **The defensible claim:** F stays within a stated margin of D3's Network AoU while closing rounds under the budget. **Not "fairer than Cui".**

> **The HERMES-era honesty note (Aug 2026), kept as history.** The S3c pilot found a narrow,
> transient effect: update yield +0.194 (CI [+0.063, +0.313], p = 0.0178) at one operating point,
> not surviving correction for 8 metrics, with mission completion moving the other way. The honest
> claim was about the warm-up, not the steady state. S3c belongs to HERMES and is not one of
> FeRRy's mechanisms. If HERMES's S3c is mentioned at all, keep it as design rationale with a
> measured illustration.

---

## 5a. Thread — asynchronous merging and staleness *(C3's merge role)* — new, 6 Oct

All three papers below were read in full on 6 Oct.

| Work | Rule | Staleness weight | Hard cutoff or deadline? |
|---|---|---|---|
| **FedAsync** (Xie, Koyejo, Gupta; arXiv 2019, OPT2020 workshop version) | the server mixes each arriving model at once: x ← (1−α_t)x + α_t·x_new, with α_t = α·s(staleness); locally, a proximal term (ρ/2)‖x − x_τ‖² | constant; polynomial (d+1)^−a; **hinge**: 1 up to b, then 1/(a(d−b)+1) (a = 10, b = 4 on CIFAR) | **No.** The hinge never reaches zero. Version 4's discussion remarks that very stale updates *could* be dropped, without evaluating it |
| **FedBuff** (Nguyen et al., AISTATS 2022) | buffer K client deltas, then w ← w − η_g·(sum / K); K = 10 by default, independent of concurrency; compatible with secure aggregation and DP | as a practical improvement, (1 + τ)^−0.5 | **No.** It does not drop slow clients |
| **Async-HFL** (Yu et al., IoTDI 2023) | device → gateway → cloud, each tier mixing asynchronously, plus a proximal term; gateway selection and cloud association by ILPs | **polynomial** (h − τ + 1)^−q, adopted from FedAsync. The "exponential decay factors" in its Table 2 are the mixing weights, not the staleness function | **No** |
| **FedEx** (§3.1) | delta accumulation, x ← x − (1/N)·Σ updates, on return | none (every update is one tour stale) | No |
| **Zhai et al.** (§3.3) | plain mean over the selected devices, over the air | none | No |
| **Cui et al.** (§4) | sequential: one client per iteration continues from the latest model | none needed (no stale updates) | No |

**FeRRy's rule, `agg:cutoff`, described accurately.**
- **The weight:** each update weighs n_i·v_i·s(age_i), with s **FedAsync's hinge truncated to exactly zero past a_max, a cutoff derived from the device's deadline.** One deadline admits the device, orders the flight and cuts off its update (C3).
- **The proximal term on devices:** as in FedProx and in FedAsync's own local objective. The build plan cites Shen et al. (IoTJ 2024) for it; credit FedAsync and FedProx as the origin and Shen as the source used.
- **The age cutoff:** the build plan attributes it to Yang et al. (JSAC 2025), whose full reference is still pending.

**What none of the merge papers has** is a deadline that both admits a device to the mission and
zeroes its update's weight past it. That combination is C3's claim, and Study 5.1 (batch 2) tests
it against `agg:plain`, `agg:fedbuff` and `agg:asynchfl`. Batch 1's D4/D4fedex contrast (§3.1) is
early, indirect evidence: the same tour, with FeRRy's merge against FedEx's, 52 against 101 s with
three mules.

**Two of our comparison rules need care before Study 5.1 runs (§6a):**
- **`agg:asynchfl` implements exp(−λ·a),** which Async-HFL does not use. It is polynomial.
- **`agg:fedbuff` sets K to the slice size,** not FedBuff's default of 10. It does keep FedBuff's (1 + τ)^−0.5 and a server learning rate.

---

## 5b. Thread — learned scheduling, and gated against monolithic *(C4)* — new, 6 Oct

**The design question.** Should a learned policy decide everything at once (monolithic), or rank
only among choices deterministic gates have already admitted (gated)? FeRRy takes the gated side
explicitly, and three pieces of work frame the alternatives.

**Chen et al., *Model-Aided Federated RL for Multi-UAV Trajectory Planning in IoT Networks*, IEEE GLOBECOM Workshops 2023.** Read in full (arXiv v2, 6 Oct).
- **The task:** three energy-limited UAVs harvest **raw sensor data** (data units in device buffers) from ten IoT devices, as mobile base stations. They never carry FL updates.
- **What is learned:** grid motion. Each slot, one of hover, north, west, south or east, under QMIX. A safety controller masks infeasible moves (map edge, battery against distance home).
- **Who is served is not learned:** a fixed max-SNR rule serves, each slot, the reachable device with data left. The serving order is a by-product of the learned path.
- **The observation:** per-device SNR, reachability, remaining data, distance and offsets; per-UAV states; own battery. There are no maps.
- **The reward:** a shared team reward, the data collected per slot.
- **"Model-aided":** training in a learned replica of the radio environment (a channel network, plus device positions estimated by PSO). **"Federated":** FedAvg of QMIX parameters across the per-UAV replicas. That is federated *RL*, not FL of a task model.
- **The authors call their method "model-aided FedQMIX".**

**What this means for E3** (our arm "after Chen"; corrections in §6a):
- **The mask is faithful:** E3's budget predicate plays the role of Chen's safety controller.
- **The bytes reward is a fair analogue** of Chen's data-collected team reward.
- **One deviation is bigger than declared.** E3's "next stop" choice also decides **who is served and in what order**, which Chen hands to a fixed max-SNR rule. Declare it beside "stops instead of grid moves, one agent, no model-aided learning, our masked double DQN".
- **The name ban applies to our port:** E3 is never called FedQMIX. Chen's own method may be cited by its name.

**Ho et al., *Energy-Efficient DDPG-Based UAV-Assisted Asynchronous FL with MC-NOMA in IoT
Networks*, IAAA 2025 (Springer LNNS, 2026).** Abstract only; the chapter is paywalled.
- **What it is:** a single DDPG agent (EEDDPG) minimising **device energy**, jointly controlling UAV movement, CPU frequency and transmit power under latency limits.
- **Not shown:** nothing read shows it choosing who is served, in what order, or on which band. The authors' earlier ICT Express 2024 paper, a ground-station version, serves every device every round.
- **So M1 is "in the style of" Ho:** one unplanned agent controlling mobility and radio together, not a port of Ho's method. M1 is built only if Study 5.5 keeps a learned score.

**CEDA: monolithic multi-drone RL** (a lab thesis; see
[`Comparative Analysis/CEDA_vs_HERMES.md`](Comparative%20Analysis/CEDA_vs_HERMES.md)).
- **The source:** an MS thesis defence (Bhamidipati, advisor Calyam, 2026) on multi-drone medical delivery. It is structurally the same scheduling problem: visit spatially spread endpoints under battery, a time-varying channel and deadlines of unequal weight.
- **The system:** a CTDE DQN over a 140-dimension observation per drone, choosing raw grid moves, with fairness pursued through six reward terms. Measured: 87.7% of patients delivered, weighted efficiency η 0.74–0.81 against 0.47–0.57 for EDF and nearest-weighted heuristics.
- **Its strongest finding is a cross-layer information ablation:** hold the policy fixed, delete one information channel at a time. FeRRy's Study 5.14 is the policy-side counterpart (one mechanism switched off at a time). An information ablation of FQ's 36 columns would be the CEDA-style complement.
- **The trade the comparison names:**

  | | Monolithic (CEDA; M1) | Gated (HERMES; FeRRy's FQ) |
  |---|---|---|
  | Can exploit coupling the designer didn't anticipate | yes | no |
  | Sample efficiency | poor (12,000 × 800 steps) | high |
  | One decision explained afterwards | hard | easy |
  | Fails safe under distribution shift | unknown | the gates still hold |

- ⚠ **Anonymity:** CEDA shares authors and an advisor with this work, and IPDPS 2027 review is double-anonymous. Cite it in the third person, if at all, and only as published. A defence deck may not be citable.

**FeRRy's own learned score, FQ** ([design note](FeRRy_Learned_Pair_Score.html)).
- **What it decides:** at each Pass-1 stop it scores every (band, next stop) pair **inside a mask.** A pair is allowed only if the band covers the plan, the stop is in the plan, and the rest of the flight still fits.
- **What it never does:** **FQ never leaves FeRRy's plan.** The plan decides which devices and which band class; the score only reorders the remaining stops and picks a band within them.
- **The comparison arms:**
  - E3 is the learned baseline without that structure (Chen's style).
  - M1 is one network over every pair with no plan (Ho's style, and CEDA's stance).
  - FX is the fixed rule in the same slot.
- **Study 5.5's verdict (6 Oct 2026, pre-registered): flat.**
  - No γ beat γ = 0 by ε = 0.01, and every γ > 0 is equivalent to γ = 0 within ±ε.
  - The best learned score sat **at FX's level (−0.0774 against −0.0786) and below greedy_1 (−0.0716)**, a simple one-step rule that sees the same inputs.
  - So **FX is FeRRy's in-flight rule, and the null is published** ([verdict](Experiment_5_RL_Calibration_Findings.md#the-sweeps-verdict-6-oct-2026-2310--flat-fx-stays)).
  - C4 rests on the two-clock structure (plan, then re-decide on measured signal), not on learning.
  - Report the copy-FX diagnosis with it. A one-revision learner screen remains for the revision window.

---

## 6. What we should **not** claim

Guardrails, so the revision does not overreach in either direction (August items kept; FeRRy's
added).

* **Do not claim** UAV-FL scheduling is unstudied. It is well studied for different roles (§1), and **FedEx works in ours.**
* **Do not claim** FeRRy beats FedEx across the board. Claim what batch 1 shows: F's route leads at N = 6 with one mule; FedEx's tour keeps pace from N = 12; FedEx's own merge falls behind (§3.1).
* **Do not call Cui's index "optimal"** or FeRRy "fairer than Cui" (§4). **Do not call** our D3 Cui's index without saying it flies our "expected" variant.
* **Do not call** E3 "FedQMIX", D5 "FedCS" without "degraded", D2 "Oort" without "statistical utility" and its deviations, or M1 a port of Ho.
* **Do not describe** `agg:asynchfl` as Async-HFL's staleness function while it uses an exponential (§5a, §6a).
* **Do not describe** FeRRy's coverage term as Zhai's (§3.3), or O1 as Zhai's method.
* **Do not claim** that learning helps in flight. Study 5.5's verdict read **flat**, and FX is FeRRy's in-flight rule (§5b).
* **Do not claim** Oort or Power-of-Choice are inapplicable. `rpow-d` and Oort port directly; FedCS and `pow-d` do not. State the boundary, not a blanket dismissal.
* **Do not claim** Byzantine robustness (a non-goal), or results from studies not yet run: H0, E3, O1, D5 and the merge rules are in batches 2–3.
* **Do not imply** the starvation problem is novel. FeRRy's enforcement under travel cost is the contribution; the problem is shared.
* **Do not cite** anything marked ⚠ in §7 without reading it first, or CEDA in a way that breaks anonymity.

---

## 6a. Corrections the 6 Oct readings made — where the plan, the SOTA document or the code say otherwise

Each line names a discrepancy found by reading the paper in full, where it appears, and a suggested
fix. None changes a result already run, except that D3's numbers are the "expected" variant's.

| # | Paper | Our current description | What the paper says | Where it appears | Suggested fix |
|---|---|---|---|---|---|
| 1 | Cui | D3: "optimal for value-weighted age when selection is free" | optimal per decoupled subproblem only; the joint problem is PSPACE-hard; one client per iteration | build plan (Baselines, 5.8), SOTA §0.2 | "the index policy Cui et al. derive, best of the heuristics they test" |
| 2 | Cui | D3 flies the closed-form index | D3 defaults to an "expected" index (ρ·I) not in the paper; the "literal" variant is eq. 48 | `whittle.py`, `driver.py` (`whittle_variant="expected"`) | state the variant wherever D3 is reported |
| 3 | Cui | `agg:seq`: "carry the model device to device" | sequential, but through the leader (hub and spoke) | build plan (aggregation arms) | "sequential through the server, after Cui" (the rule is decided out anyway) |
| 4 | Zhai | "Zhai's coverage term for devices the deadline excludes" | no deadlines; a symmetric device-selection penalty in the bound; plus a hard serve-every-S constraint | build plan (C2, theory track), SOTA §0.2 | "related to Zhai's selection penalty"; cite (31b) as precedent for the age cap |
| 5 | Zhai | O1 "after Zhai" | offline joint design by convex approximation to a local optimum; no band classes or exhaustive search | build plan (O1) | "an offline full-information joint design, in the spirit of Zhai" |
| 6 | Chen | E3's deviations: stops, one agent, no model-aided learning, double DQN | Chen's who-is-served is a fixed max-SNR rule, not learned | build plan (E3), SOTA §0.2 | add: "E3 learns who is served and in what order; Chen does not" |
| 7 | Ho | M1 "after Ho": one DDQN over (band, stop) | DDPG controlling movement, CPU and power for energy; no shown served-device choice (abstract only) | build plan (M1), SOTA §0.2 | "in the style of Ho" |
| 8 | FedAsync | `agg:cutoff` = "FedAsync's hinge with Yang's cutoff" | the hinge never reaches zero; FedAsync has no deadline | build plan (aggregation arms), `aggregation_rules.py` docstring | "FedAsync's hinge, truncated to zero at a deadline-derived a_max" |
| 9 | FedAsync | the proximal term "from Shen et al." | FedAsync's local objective already has one; FedProx is the usual origin | build plan | credit FedProx and FedAsync; Shen as the source used |
| 10 | Async-HFL | `agg:asynchfl`: "exponential staleness decay … after Async-HFL" | polynomial (h − τ + 1)^−q; "exponential decay factor" names the mixing weight | `aggregation_rules.py` (`exp(−λ·a)`), build plan | **before Study 5.1 runs:** switch to the polynomial form, or keep the exponential and declare it ours |
| 11 | FedBuff | `agg:fedbuff`: K = slice size | K = 10 by default, independent of concurrency | `aggregation_rules.py` (documented) | declare K = slice size as a deviation in 5.1's write-up |
| 12 | FedEx | D4 "1/N accumulation applied on return" | correct: it accumulates update *deltas*, each one tour stale; CARP names the whole two-level algorithm | SOTA §0.2 | wording only |
| 13 | FedEx | D4 = FedEx-Async with CARP | one transporter per mission (Gibbs never runs), no energy gate | `fedex_carp.py` (documented) | say "FedEx's tour and merge"; state both deviations with D4's numbers |

---

## 7. Citation readiness

| Work | Venue and links | Read? |
|---|---|---|
| FedCS — Nishio & Yonetani | IEEE ICC 2019 · [doi:10.1109/ICC.2019.8761315](https://doi.org/10.1109/ICC.2019.8761315) · [arXiv:1804.08333](https://arxiv.org/abs/1804.08333) | ✅ full text (Aug) |
| Oort — Lai et al. | USENIX OSDI 2021 · [arXiv:2010.06081](https://arxiv.org/abs/2010.06081) | ✅ full text (Aug) |
| Power-of-Choice — Cho, Wang & Joshi | AISTATS 2022, PMLR 151:10351–10375 ([proceedings](https://proceedings.mlr.press/v151/jee-cho22a.html)); preprint [arXiv:2010.01243](https://arxiv.org/abs/2010.01243) | ✅ full text (Aug). ⚠ **published retitled** *Towards Understanding Biased Client Selection in Federated Learning*; **confirm the `rpow-d` naming survived** before citing it by name |
| UAV-Aided Multi-Community FL — Mestoukirdi et al. | IEEE GLOBECOM 2022 · [arXiv:2206.02043](https://arxiv.org/abs/2206.02043) | ✅ full text (Aug) |
| Fairness-Enhanced UAV FL | Sensors 2024 · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934714/) | ✅ full text (Aug) |
| **FedEx**: *Indirect-Communication FL via Mobile Transporters* — Bian, Shen, Chen, Xu | IEEE TMC 24(6), 2025, pp. 4845–4857 · [doi:10.1109/TMC.2025.3527405](https://doi.org/10.1109/TMC.2025.3527405) · [open copy (NSF PAR)](https://par.nsf.gov/servlets/purl/10588590) · conference version CISS 2023, [arXiv:2302.07323](https://arxiv.org/abs/2302.07323) | ✅ full text, published version (6 Oct); supplementary proofs not read |
| *The Data Value Based Asynchronous FL for UAV Swarm Under Unstable Communication Scenarios* — Cui, Yang, Wu, Feng, Hu | IEEE TMC 23(6), 2024, pp. 7165–7179 · [doi:10.1109/TMC.2023.3331906](https://doi.org/10.1109/TMC.2023.3331906) | ✅ full text, published version (6 Oct); no open copy exists; supplementary proofs not read |
| *UAV-Enabled Asynchronous FL* — Zhai, Yuan, Wang, Yang | IEEE TWC 24(3), 2025, pp. 2358–2372 · [doi:10.1109/TWC.2024.3520501](https://doi.org/10.1109/TWC.2024.3520501) · [arXiv:2403.06653](https://arxiv.org/abs/2403.06653) | ✅ full text, published and arXiv versions (6 Oct); appendices in part |
| *Model-Aided Federated RL for Multi-UAV Trajectory Planning in IoT Networks* — Chen, Esrafilian, Bayerlein, Gesbert, Caccamo | IEEE GLOBECOM Workshops 2023, pp. 818–823 · [doi:10.1109/GCWkshps58843.2023.10465088](https://doi.org/10.1109/GCWkshps58843.2023.10465088) · [arXiv:2306.02029](https://arxiv.org/abs/2306.02029) | ✅ full text, arXiv v2 (6 Oct). The comparison note `HERMES_vs_Chen2023_Model-Aided_FedQMIX.md` is not in the repository |
| *Energy-Efficient DDPG-Based UAV-Assisted Asynchronous FL with MC-NOMA in IoT Networks* — M. C. Ho, Win, Do, Na, Cho | IAAA 2025, Springer LNNS (*Intelligent Aerial Access and Applications Towards 6G and Beyond*), 2026, pp. 301–313 · [doi:10.1007/978-3-032-14935-0_23](https://doi.org/10.1007/978-3-032-14935-0_23) | ⚠ **abstract only** (the chapter is paywalled) |
| FedAsync: *Asynchronous Federated Optimization* — Xie, Koyejo, Gupta | arXiv 2019, OPT2020 workshop version · [arXiv:1903.03934](https://arxiv.org/abs/1903.03934) | ✅ full text, v5 and v4 (6 Oct) |
| FedBuff: *Federated Learning with Buffered Asynchronous Aggregation* — Nguyen, Malik, Zhan, Yousefpour, Rabbat, Malek, Huba | AISTATS 2022, PMLR 151:3581–3607 · [proceedings](https://proceedings.mlr.press/v151/nguyen22b.html) · [arXiv:2106.06639](https://arxiv.org/abs/2106.06639) | ✅ full text, the camera-ready version (6 Oct) |
| Async-HFL — Yu, Cherkasova, Vardhan, Zhao, Ekaireb, Zhang, Mazumdar, Šimunić Rosing | IoTDI 2023 · [doi:10.1145/3576842.3582377](https://doi.org/10.1145/3576842.3582377) · [arXiv:2301.06646](https://arxiv.org/abs/2301.06646) | ✅ full text, arXiv v4 (6 Oct) |
| CEDA — Bhamidipati (MS thesis defence, advisor Calyam) | 2026, unpublished slides · [our comparison](Comparative%20Analysis/CEDA_vs_HERMES.md) | read from the slides only. ⚠ shares authors; mind anonymity; may not be citable |
| Yang et al. (age cutoff; contextual bandit) | IEEE JSAC 2025 | ⚠ **reference pending** (not held in the repository) |
| Shen et al. (proximal term) | IEEE IoTJ 2024 | ⚠ **reference pending** |
| Chen et al. (mobility as mixing, the theory track) | IEEE TVT 2025 | ⚠ **reference pending** |
| Vehicular reputation FL | Chinese J. Aeronautics 2024 · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S100093612400236X) | ⚠ abstract only: enough to exclude, **not to cite** |
| AoI in UAV-aided collection (review) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1084804523000711) | ⚠ not read in full |
| UAV swarm AoI / topology-coupled urgency | [arXiv:2608.00061](https://arxiv.org/abs/2608.00061) | ⚠ description confirmed, not read in full |
| Data-Efficient Energy-Aware Participant Selection | — | ⚠ **not cleared** |
| Privacy-Preserving FL for UAV (A3C) | — | ⚠ **not cleared** |
| Contribution-Based Resource Allocation | — | ⚠ **not cleared** |
| Broad UAV-FL sweep (client / relay / aggregation / Byzantine) | — | ⚠ triaged **by architecture class** from abstracts: enough to exclude as arms, **not** to cite |

---

## 8. Open items for the revision

**From the 6 Oct readings:**
- [x] Full-text reads of FedEx, Cui, Zhai, Chen, FedAsync, FedBuff and Async-HFL (§7). Ho: abstract only.
- [ ] **Decide `agg:asynchfl`'s staleness function before Study 5.1 runs** (§6a, item 10).
- [ ] Carry §6a's wording fixes into the build plan and the SOTA document. Add E3's served-device deviation, and D3's "expected" variant, to their reports.
- [ ] Find the full references for Yang (JSAC 2025), Shen (IoTJ 2024) and Chen (TVT 2025). Recover the novelty audit (Revision 3, 23 Sep 2026) and `HERMES_vs_Chen2023_Model-Aided_FedQMIX.md`; both are cited by the build plan and absent from the repository.
- [ ] Get Ho et al.'s full text, if M1 is built.
- [ ] Decide whether and how CEDA is cited under double-anonymous review.
- [x] Re-check §5b against Study 5.5's verdict: **flat**, FX stays (6 Oct).
- [ ] Re-check §3.1 against batch 2 (E3, H0, the merge rules).
- [ ] Consider a CEDA-style information ablation of FQ's features (§5b), if the learned score is kept.

**From August, still open:**
- [ ] Confirm the `rpow-d` naming appears in the AISTATS version (§7).
- [ ] Decide how much space the §1 taxonomy gets. Recommendation: a short paragraph plus the role table; it earns its length by making the rest of the section short.
- [ ] Read anything marked ⚠ that survives into the final citation list.
- [ ] Cross-check against the paper/code divergences recorded in the architecture review, so Related Work describes no capability the results section does not demonstrate.
