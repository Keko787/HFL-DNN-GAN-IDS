# CEDA vs HERMES — Breakdown and Comparative Analysis

**Source:** `Chandra MS Thesis - Slides.pptx` — 37 slides, MS thesis defence
**Author:** Subrahmanya Chandra Bhamidipati · Advisor: Dr. Prasad Calyam
**Compared against:** HERMES as read from source at `main` @ `35e2eeb`, and *HERMES: A
Layered Coordination Framework for Federated Learning with Drones as Data Mules*
(ACM/IEEE SEC 2026 submission)
**Status:** Analysis only. No project code or thesis material has been modified.
**See also:** the HERMES [problem statement](../architecture%20documents/System_Architecture_Overview.md#2-problem-statement),
[solution statement](../architecture%20documents/System_Architecture_Overview.md#3-solution-statement),
and [traceability matrix](../architecture%20documents/System_Architecture_Overview.md#4-problem--solution-traceability)

---

## 1. Why these two are worth comparing

They come from the same lab, share an advisor, and both list Bhamidipati and Kostage on
their author lines. More importantly they are **structurally the same problem wearing
different clothes**: a battery-limited aerial vehicle must visit spatially distributed
endpoints over a time-varying wireless channel, serve deadline-bound tasks of unequal
importance, and decide *where to go next* under partial information.

CEDA calls those endpoints patients and the tasks medical deliveries. HERMES calls them
edge devices and federated-learning model exchanges. The scheduling core is close enough
that the design choices transfer directly — and the two projects made **opposite** ones.

A third artefact makes the comparison sharper. `hermes_rl/` — an untracked, un-ignored
nested git repository sitting inside the HiFINS working tree (finding
[G-01](../Codebase%20Review/00_Critical_Problem_Areas.md#g-01)) — is a 2,251-LOC
single-drone data-relay RL prototype. Its problem statement is almost exactly CEDA's
minus triage, and its architecture sits precisely between CEDA's and HERMES's. It is the
missing middle term.

---

## 2. Breakdown of the deck

### 2.1 Problem and framing (slides 1–10)

Multi-drone medical supply delivery during disaster response. The argument is that three
layers are **coupled and cannot be optimised in isolation**:

| Layer | What varies | Failure if ignored |
|---|---|---|
| **Physical** | wind, obstacles, battery | drone expires before delivery |
| **Network** | intermittent connectivity, random command failure | shortest path silently drops commands in a low-signal zone |
| **Application** | triage priority, deadlines, fairness | serving the highest-priority patients first depletes drones before the rest are reached |

Three research gaps are claimed: no framework jointly optimises across all three; purely
deadline-driven scheduling (EDF) misallocates while purely weight-greedy scheduling
starves low-priority patients; and drones observe only discrete triage weights, not the
underlying survival dynamics.

Three research questions follow, with explicit targets: **completion rate > 85 %**,
**weighted triage efficiency η ≥ 0.80**, clinical priority ordering preserved with
near-zero W1/W2 mortality, and policy transfer validated on PX4 SITL.

The related-work table positions CEDA against four systems, including the lab's own
single-drone predecessor (Calyam 2026), and claims novelty in CTDE multi-agent +
triage-weighted scheduling + per-patient survival dynamics + SITL validation.

### 2.2 The CEDA framework (slides 12–20)

**Paradigm.** CTDE — Centralised Training with Decentralised Execution. Both agents
share a replay buffer with full joint-state visibility during training; at execution each
drone acts on its local observation only, with **no inter-agent communication**. The same
Q-network is queried twice per step with swapped observation ordering.

**Observation.** 140 dimensions per agent, 280 joint:

- 9 agent features — normalised position, battery, landed flag, relative position of the
  other agent, direction to landing zone
- 7 × M = 56 patient features — position, unit direction, normalised countdown
  `t_rem / T_max`, delivery flag, normalised triage weight `w_p / w_max`
- 3 × 5 × 5 = 75 environmental features — local obstacle, wind, and low-signal occupancy

Inactive patient slots are zero vectors, giving a fixed-length encoding regardless of
active patient count.

**Actions.** Six discrete: `{UP, DOWN, LEFT, RIGHT, LAND, HOVER}`. `LAND` is only
effective at the designated landing zone. Wind and low-signal zones cause stochastic
action failure with elevated battery cost, so risk-aware avoidance must be learned
implicitly through reward.

**Reward — "Priority-Preserving Fair Scheduling."** Rather than one explicit fairness
constraint, six interacting components:

1. Uniform unattended penalty `−P_death` per expired patient, **weight-agnostic**
2. Timer-scaled delivery reward `R_goal · (t_rem/T_max) · w_i`
3. Potential-based shaping on Manhattan distance to nearest undelivered patient
4. Spatial separation penalty `−C` when agents are within `r_close = 4`
5. Dynamic patient spawning every 75 steps, to defeat static greedy routing
6. Battery constraints — low-battery and depletion penalties

Fairness is measured, not enforced: `η = Σ_{i∈S} w_i / Σ_{i∈ζ} w_i`.

### 2.3 Evaluation (slides 22–33)

**Setup.** Two drones, 50 × 50 grid, 200 static obstacles, wind and low-signal zones
refreshed every 30 steps, up to M = 8 patients spawning every 75 steps, `T_max = 250`,
weights `W = {1, 2, 3}`. Trained 12,000 episodes × 800 steps on an RTX 3090.

**Convergence.** Landing rate above 90 % by episode 2,000; deliveries stabilise at 6–7
per episode; steps per episode fall 800 → 450–500 by episode 6,000; battery remaining
rises from ~20 to 52–58 symmetrically across both agents.

**Fairness.** Final 1,000 episodes:

| Class | Delivered/ep | Unserved/ep |
|---|---|---|
| W1 Stable | 2.61 | **0.00** |
| W2 Urgent | 1.10 | **0.02** |
| W3 Critical | 3.31 | 0.96 |
| **Total** | **7.02 / 8 (87.7 %)** | |

Kruskal–Wallis returns p > 0.05 for W1 vs W2 — statistically equivalent treatment of the
two lower-priority classes. The W3 residual is attributed to steep logistic decay rather
than policy neglect.

**Baselines.** Three, of increasing sophistication, across six scenarios:

| Metric | CEDA | Smart EDF | Smart NNPW | Naive NNPW |
|---|---|---|---|---|
| Triage efficiency η | **0.74–0.81** | 0.47–0.54 | 0.47–0.57 | 0.12–0.15 |
| Patients delivered/ep | **6.27–6.59** | 4.0–5.0 | 4.0–5.0 | 1.3–1.6 |
| Landing rate | **50–64 %** | — | — | ~0 % |
| W3 unserved/ep | **1.11–1.56** | 2.0–4.5 | 2.0–4.5 | 6.3–6.6 |

**Cross-layer ablation.** Six conditions, one layer removed at a time — the single most
transferable result in the deck, discussed in §5.

**3 × 3 stress interaction.** Network disruption (`p_fail ∈ {0, 0.3, 0.6}`) crossed with
application difficulty (light/baseline/heavy decay). The dominant gradient is **vertical**
— application stress drops η from 0.70–0.73 to 0.26–0.33 (> 55 %) while network stress
barely moves it. The claim: CEDA decouples network disruption from scheduling performance
under all but the most extreme combinations.

**PX4 SITL.** Two X500 quadrotors, independent autopilot instances, coordinator at 4 Hz
over an 800-step horizon. Nominal episode: 8/8 delivered, η = 1.00, both landed, minimum
battery 19.5 %, zero collisions, mean trajectory tracking error 0.03 m (max 1.76 m)
against a 2 m grid cell. Stress conditions (nominal, 35 % battery, high hazard, 3-step
action delay) hold delivery rate at 0.75–0.94 and η at 0.70–0.87 with zero collisions.
Two honest limitations are stated: repeated invalid landing attempts consume budget, and
**delayed control is the hardest condition** — the primary robustness limit is stale
actions, not environmental difficulty.

---

## 3. The central architectural difference

Both systems decompose into three layers, and both use a DQN. What differs is **how much
of the decision the network is allowed to make**.

```mermaid
D7
```

CEDA's position is that the coupling *is* the problem, so the policy must see everything
at once — "the policy internalises hazard structure during training". HERMES's position
is the opposite and is written into its design as principle 12: deterministic stages
decide *who is eligible*, and the learned selector only ranks contact positions the gates
have already admitted. A `SelectorScopeViolation` is raised if a gated-out candidate ever
reaches the selector, and the selector is skipped entirely when a bucket has fewer than
two candidates.

`hermes_rl/` is the explicit middle: its trainer docstring says *"Heuristic handles
navigation… DQN handles the complex part: which BS + channel… This reduces the RL action
space from 45 → 9 and fills the replay buffer with high-quality heuristic trajectories
from episode 1."*

Neither position is simply right. The trade is:

| | Monolithic (CEDA) | Gated (HERMES) |
|---|---|---|
| Can exploit coupling the designer didn't anticipate | **yes** | no — gates fix the structure |
| Sample efficiency | poor — 12,000 × 800 steps | high — 11 features, ~200 ops per bucket |
| Explaining one decision post-hoc | hard | easy — the gate that admitted it is a pure function |
| Ablatable by information | **yes, and CEDA did it** | in principle; never run |
| Fails safe under distribution shift | unknown | gates still hold; only ordering degrades |
| Certification / audit story | weak | strong |

For a disaster-response research prototype, CEDA's choice is defensible and produces the
headline result. For an intrusion-detection substrate meant to run unattended on a mule
NUC, HERMES's is the better default — and its own numpy DDQN, at 11 features and 16
hidden units, only makes sense *because* the gates carry the constraints.

---

## 4. Side-by-side

| | **CEDA** | **HERMES** |
|---|---|---|
| Domain | medical supply delivery, disaster response | hierarchical FL for network intrusion detection |
| Agents | 2 drones, homogeneous | 1 mule per mission; N mules per cluster, coordinated by the edge server |
| Tiers | drones ↔ patients (flat) | 4 — device, mule NUC, edge server, cloud |
| Learning | CTDE DQN, joint replay, local execution | two independent DDQNs: L1 channel, L2 intra-bucket selector |
| Policy scope | routing + assignment + navigation + energy + triage | ordering within one bucket, ≥ 2 candidates |
| Observation | 140 dims/agent (280 joint) | 11 selector features; 8-dim L1 state |
| Action space | 6 discrete moves | argmax over candidate contacts |
| Framework | PyTorch (RTX 3090) | **numpy, no ML framework in `hermes/`** |
| Coordination | emergent via reward (spatial separation penalty) | explicit — disjoint `MissionSlice` per mule from the registry |
| Task priority | 3 triage weights + logistic survival decay | 3 buckets (NEW / SCHEDULED / BEACON) + adaptive deadline window |
| Fairness | 6 reward terms + η + Kruskal–Wallis test | Jain's index, participation entropy, `completion_fairness`; deadline widening on miss — **measured, never enforced** |
| Aggregation | none — no learning is federated | two-pass partial + cross-mule FedAvg, exact by construction |
| Realism ladder | grid sim → **PX4 SITL, real autopilot** | loopback → **real processes + TCP sockets** → AERPAW pending |
| Statistics | Kruskal–Wallis; ranges across 6 scenarios | paired Wilcoxon + Cliff's δ + bootstrap CI + crossover R* |
| Reproducibility | 12,000-episode training run | paired seeds via SHA-256, checkpoint-resume CSV harness |
| Ablation | **6-condition information ablation** | 4-arm policy ablation (A1–A4) |
| Test suite | not discussed | 512 passing unit + integration tests |

The realism claims point in **different directions and are not substitutes**. CEDA is
further along sim-to-real: a real autopilot, real controller dynamics, measured tracking
error. HERMES is further along distributed-systems realism: real OS processes, real
sockets, real framed wire protocol, real FedAvg arithmetic. Neither has flown hardware;
neither should claim the other's ground.

---

## 5. The transferable result

CEDA's cross-layer **information** ablation is the finding HERMES should act on.

```mermaid
D8
```

Three things follow for HERMES.

**5.1 Its own ablation answers a different question.** Experiment 3's A1–A4 arms vary the
*policy* — centralized FL, round-robin mule, deadline-feasibility mule, HERMES selector.
That measures whether the scheduler helps. It cannot say which *input* the scheduler
actually depends on, because every arm sees the same state. CEDA's design — hold the
policy fixed, delete one information channel, re-measure — is the cheaper and more
diagnostic experiment, and HERMES is well positioned to run it: `SelectorEnv` and
`DeviceSchedulerState` are explicit, injected structures, so zeroing `mule_energy`,
`rf_prior_snr_db`, `delivery_priority`, or the deadline fields is a per-field masking
change in one place, with no retraining required for the deterministic arms.

**5.2 The empirical ranking is uncomfortable for both projects.** CEDA finds the network
layer contributes least (~10 % η) and energy state most (W3 unserved 1.4 → 4.1). HERMES's
L1 is the thinnest of its three layers, its RF prior was a hardcoded `20.0` constant until
EX-4.3 wired it, and `mule_energy` is a selector feature that no experiment has isolated.
If the same ranking holds, the effort spent on L1 buys less than an equivalent effort on
energy-aware scheduling would — worth knowing before the next revision cycle.

**5.3 The 3 × 3 stress design is directly reusable.** Experiment 3 already sweeps β ×
rrf × deadline_het × jittery. Presenting a 3 × 3 slice as a heatmap and reading the
dominant gradient is a cleaner claim than a set of marginal box plots, and it answers the
integration question — *does the advantage survive when two layers degrade together?* —
that Experiment 4 exists to answer.

---

## 6. What else HERMES could take

**Fairness *enforcement*, not measurement.** Both systems measure fairness, and HERMES's
instrumentation is if anything the more careful of the two:
[`experiments/exp3/metrics.py`](../../experiments/exp3/metrics.py) implements Jain's index
`J = (Σx)² / (N · Σx²)` over per-device service counts, Shannon participation entropy, and
a separate `completion_fairness` on per-device *completion* counts — the last specifically
because visit-based Jain is trivially 1.0 under the centralized arm's universal sampling
and therefore not contestable across arms. That distinction is sharper than anything in
the CEDA deck.

The asymmetry is in what happens next. CEDA's six reward terms *act* on fairness during
training; HERMES's metrics are computed post-hoc in the analysis layer and feed nothing
back into the scheduler. Bucket priority remains strictly ordered, so a persistently
low-priority device can still never be visited — the metric will faithfully report the
starvation it cannot prevent. The gap is a control loop, not an observation.

**A weight-agnostic floor.** CEDA's most effective fairness device is deliberately the
*simplest*: reward term 1 penalises an expired patient by `−P_death` **regardless of
triage weight**, which is what produces the near-zero W1/W2 mortality. HERMES's bucket
priority is strictly ordered with no analogous floor, so a permanently-low-priority
device can in principle never be visited. A weight-agnostic miss penalty in the deadline
fold is the equivalent mechanism.

**Explicit deterioration dynamics.** CEDA's per-patient logistic survival curve stops the
policy from exploiting a static priority. HERMES's `deadline_fulfilment_s` moves ±5/10 s
per outcome — a linear analogue. Whether device utility should decay non-linearly with
staleness is at least worth stating as a modelling decision rather than leaving implicit.

**Naming the DDQN honestly.** CEDA's deck is clear that its network is a DQN. HERMES's
selector is called a DDQN and *is* one — but the double-Q decoupling is baked in at
collection time (`selector_train.py:129,446` pick the bootstrap action with the online
net and freeze it into the `Transition`), so the stored action is stale by up to the
replay-buffer age. Textbook DDQN re-selects at update time. This is a methodology footnote
worth writing down before a reviewer asks.

---

## 7. What CEDA could take from HERMES

Offered in the same spirit; all three are cheap.

**Paired-seed experiment harness.** CEDA reports ranges across six scenarios. HERMES's
`experiments/runner/` derives each trial seed as `SHA-256(base_seed | cell_id |
trial_index)` **excluding the arm**, so every arm sees an identical environment — which
is what makes a paired Wilcoxon test valid. CEDA's baseline comparison would be
substantially stronger with paired seeds and Cliff's δ rather than reported ranges, and
the harness is domain-agnostic and already written.

**Checkpoint-and-resume trial logging.** `CSVTrialLog` keys on `(cell_id, arm,
trial_index)` and skips completed rows, so a crashed 12,000-episode run resumes exactly
where it stopped. For a multi-day RTX 3090 sweep this is a direct time saving.

**Structured event logging over scraped stdout.** HERMES emits one versioned JSONL
record per state transition per process, which is what lets an analysis script compute
per-round metrics from the actual run rather than from re-derived aggregates.

---

## 8. Where both are exposed

Recorded plainly, because both projects will face these questions from the same
reviewers.

| Shared limitation | CEDA | HERMES |
|---|---|---|
| Small N, claimed as future work | 2 drones; N > 2 listed under future work | 2 mules × 5 devices validated |
| Hand-rolled DQN, no framework baseline | PyTorch DQN, no PPO/QMIX comparison | numpy DDQN, no framework comparison |
| Headline numbers are simulation | SITL is a real autopilot, still simulated | multi-process is real IPC, still one laptop |
| A stubbed component in the loop | discrete grid abstraction over continuous flight | `StubGeneratorHost` emits zero tensors — the GAN never runs |
| Single evaluator | one thesis codebase | one review pass |

One asymmetry is worth stating: CEDA's stub is an **abstraction** (a 2 m grid cell,
validated at 0.03 m tracking error). HERMES's is a **gap** — the synthetic-augmentation
path is inert on both ends, and `experiments/exp4/model_task.py:150-152` says so
outright. CEDA's simplification was measured; HERMES's has not been.

---

## 9. Honest limits of this comparison

- **η is not comparable across the two.** CEDA's η is weighted patient delivery; HERMES's
  is a scheduling-efficiency ratio over a different denominator. Same symbol, different
  quantity. No cross-system number in this document should be read as a benchmark.
- **The deck is a defence, not a paper.** Figures were read from slide text and captions;
  no underlying data, code, or thesis chapters were available. Any number here is as
  reported on a slide.
- **Different objectives.** CEDA optimises patient survival; HERMES optimises model
  convergence per joule per byte. Architectural lessons transfer; results do not.
- **`hermes_rl/` was read as source, not as a claim.** It is untracked, undated, and has
  no accompanying write-up. It is used here as evidence of a design stance, not as a
  result.

---

## 10. Recommended follow-ups

| # | Action | Effort | Why |
|---|---|---|---|
| 1 | Run a HERMES information ablation (mask `mule_energy`, `rf_prior_snr_db`, `delivery_priority`, deadline fields one at a time) | ~2 d | Answers which layer actually carries the result; CEDA's is the strongest single finding in the deck |
| 2 | Close the fairness loop — feed the existing Jain / entropy signal back into bucket priority | ~2 d | The metrics already exist and are well designed; nothing acts on them, so starvation is observable but not preventable |
| 3 | Add a weight-agnostic miss penalty to the deadline fold | ~0.5 d | CEDA's cheapest and most effective fairness mechanism; closes the starvation hole |
| 4 | Present the Exp-3 β × jittery slice as a 3 × 3 heatmap | ~0.5 d | Answers the integration question directly; reuses data already collected |
| 5 | Record the DDQN bootstrap-staleness footnote in the design docs | ~1 h | Pre-empts a reviewer question that has a good answer |
| 6 | Offer `experiments/runner/` to the CEDA work | — | Paired seeds would materially strengthen its baseline comparison |
| 7 | Resolve `hermes_rl/`'s status — submodule, tracked, or removed | ~1 h | Finding [G-01](../Codebase%20Review/00_Critical_Problem_Areas.md#g-01): an untracked nested repo is invisible to a fresh clone |

Items 1–4 are the ones that would change what the next paper can claim.
