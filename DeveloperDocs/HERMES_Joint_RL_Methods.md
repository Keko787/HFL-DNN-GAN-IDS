# Joint RL methods in HERMES / FeRRy — by joint optimisation and by layer

*7 Oct 2026. A reading document: what is optimised jointly, what each layer contributes to it, which part is learned, and what the evidence says so far. It restates the code and the records; it decides nothing. Sources are named in each section; the status of every stage is in [Experiment_5_Readiness.md](Experiment_5_Readiness.md).*

---

## 0. The short version

HERMES (built as **FeRRy**) couples three layers that were once designed separately. A *joint optimisation* is a place where one decision spans more than one layer. There are four, on two clocks, and one prototype lineage that led to them:

| # | Joint optimisation | Clock | Variables chosen together | Layers spanned | Learned? | Status |
|---|---|---|---|---|---|---|
| **J1** | Reach and route | Plan (dock, once per mission) | band class b̄ × stop clustering × route π | L1 + L2 (+ L3 weights) | **No** — exhaustive / local search on a hand-set score V | Built; Study 5.4 (batch 2) not run |
| **J2** | Band and next stop | Flight (every Pass-1 arrival) | (band, next stop) pair | L1 + L2 (+ L3 reward) | **Tested** — learned pair score FQ vs fixed rule FX | **Flat; FX kept** (pre-registered, 6 Oct) |
| **J3** | The return path | Both | rate → dwell → clock → feasibility → re-plan | L1 → L2 | No — one shared predicate | Built (Phases 3–5) |
| **J4** | One objective | Both | plan score V, flight reward, merge weight, age cap | L2 ↔ L3 | Reward only (and only for FQ) | Built; Study 5.7 flies FX variants, not run |
| **J5** | Prototype and baselines | Flight | (waypoint, base station, channel); stop only | L1 + L2 | Yes | Lineage (`hermes_rl/`) and arm E3 |

**What the evidence says so far.** The coupling itself is real and built. The learning claim is a **null**: a learned in-flight score matched the fixed rule FX and fell short of the one-step rule `greedy_1`; looking ahead (γ > 0) added nothing. FeRRy's lead over the baselines comes from the two-clock *structure* (plan at the dock, re-decide on measured signal at each stop), not from a learned policy. That is the honest description in [the calibration findings](Experiment_5_RL_Calibration_Findings.md), and the rest of this document is organised so that claim can be read against the layer that produced it.

### 0.1 The three layers, as they matter here

| Layer | Job | Joint-method role | Code |
|---|---|---|---|
| **L1 — RF communication** | Prices a link: range, rate, dwell, SNR over time | Supplies the *cost* every joint decision trades: how far a band reaches (→ stops) and how long a stop takes (→ clock) | `hermes/l1/` — `contact_link.py`, `channel_model.py`, `mission_clock.py`; backhaul: `channel_utility.py` |
| **L2 — Mobility-aware scheduling** | Who is served, where, in what order, whether it still fits | Holds both decision slots (plan search, flight slot) and the hard gates that bound them | `hermes/scheduler/` — `plan/`, `policies/`, `selector/`, `routing/`, `stages/` |
| **L3 — Hierarchical federated learning** | Merges updates; ages and deadlines | Supplies the *value*: what a collected update is worth, and what a skipped device costs | `hermes/mission/partial_fedavg.py`, `hermes/cluster/cross_mule_fedavg.py`, `stages/s3d_age_cap.py` |

> **Naming.** FeRRy is HERMES's Path A system; `hermes/` is the package it is built in. "F" is the full system, "FX" the same system with the fixed in-flight rule, "FQ" with the learned one.

### 0.2 The two clocks

```
PLAN CLOCK — at the dock, once per mission
  demand → band class b̄ → range R(b̄) → S3a stops → route π → S3b gate → COMMIT (b̄, queue, budget)

FLIGHT CLOCK — at every Pass-1 arrival
  arrive → observe SNR on each band → choose (band, next stop) → serve (dwell = bytes/rate) → still fits? ─ no → re-plan
                                                                                                      └ yes → next stop
```

The hard gates (S1, the S3b predicate, the age cap) always run **before** anything learned ranks anything. Learning only chooses among admitted options (Freeze principle 12; `selector/scope_guard.py`). That holds for every method below.

---

## 1. J1 — Reach and route (plan clock)

**What is optimised jointly.** The band class b̄ and the Pass-1 route π, once per mission, at the dock. This is the "reach is a decision" claim (C1): a wider class serves fewer devices per stop at a faster rate; a narrower class reaches the field from one stop and dwells longer there. No rule of thumb picks between them, so the planner prices every class and commits the best.

**Why it is a joint problem and not a chain.** Class sets range, range sets S3a's clustering, clustering fixes the stops and so the route, and the route sets arrival times. Cutting the band→range edge is the *test for system versus stack* (arm **FB+**, b̄ pinned to one class).

**Is it learned? No.** Plan-time is search, and that is deliberate. A learned plan-time value is considered only if the flight-time results show a slope (build plan, Study 5.5: "a learned plan-time value, and with it a hierarchy, is considered only if the plan clock shows a slope too"). 5.5 was flat, so that door stays shut.

### 1.1 L1 — what the radio layer contributes

- **Band classes differ by bandwidth, not carrier.** 3.32 / 3.34 / 3.90 GHz differ by at most 1.4 dB of free-space loss, too little to carry a range trade (`contact_link.py`). The classes are `wide` (20 MHz), `medium` (5 MHz), `narrow` (1.4 MHz); one shared EIRP and a noise floor that scales with occupied bandwidth give R_b = R_anchor · (B_anchor / B_b)^(1/n).
- **Range and rate.** At h = 25 m and the wide anchor at 60 m planar, narrow reaches about 164–232 m (n = 3.0 / 2.2) and medium about 100–120 m; the CQI-15 peaks are 75.4 Mb/s (wide) and 4.39 Mb/s (narrow), but no class reaches them at h = 25 m; the realised ceilings are about 20.0 and 3.6 Mb/s. `dwell(N, b, s) = 8N / rate(b, s)`.
- **What the planner sees.** The *mean* SNR only. Pricing the seeded phase would be an oracle, so realised missions run longer than planned — most on narrow (+19.2 % at 1 MB, against +0.8 % on wide).
- **Before this was built, the plan-time loop had nothing to commit to.** `rf_range_m` was a fixed 60 m scalar and the contact link had no band at all (decision memo, cut 1). L1's only band decision was the *backhaul* (mule → base station), picked once per mission by the deterministic utility U(c, t) = R(γ₁ + g(c)) − κ(c) − λ(c, t) in `channel_utility.py`. `ChannelDDQN` (8→16→3) was never trained and is retired from the plan.

### 1.2 L2 — what the scheduling layer contributes

- **Clustering** (`stages/s3a_cluster.py`): S3a runs once per class at that class's radius, with a hover rule (`plan/hover.py`) that gives a capped device its own stop when its cluster cannot serve it.
- **Search** (`plan/plan_search.py`), chosen by size:

  | Search | When | What it is |
  |---|---|---|
  | `exact` | ≤ 6 devices | The whole family, depth first, with monotone pruning; exactly the optimum over the member subsets V prices |
  | `stop_subsets` | more devices, ≤ 6 stops in the class | Ordered subsets of stops, each whole if it fits else reduced greedily |
  | `local` | more stops | 2-OPT tour, member trim, first-improvement scans |

- **Gate before commit:** every stop must pass S3b (`RULE_DEADLINE_BUDGET`) from the state the previous one left, plus the energy clause and the age cap. The empty plan is a candidate and is chosen only when nothing that serves anyone is admitted.
- **Choice rule** (`plan_key`): the age-cap key first, then the served weight share (default `coverage_rank = lexicographic`), then V, then class index. A total order, so the pick never depends on enumeration order.

### 1.3 L3 — what the learning layer contributes

- **Demand and weights.** Each device carries Deadline(j) and an age a_j (the device's own mule's missions since its last merged update). `coverage_weight` weighs a demanded device by age (default) or uniformly, times (1 + miss streak) when `miss_priority` is on.
- **The score V** (`plan/plan_score.py`), declared rather than derived, in FedEx's form:

  ```
  V(b̄, π | demand) = −[ c₁(Δ/T)² + c₂·U + c₃·L ] − c₄·E/(P_hover·T)
  ```

  Δ is the *whole* mission on b̄ (Pass 1, return, upload, turnaround, and Pass 2, which also flies b̄); U the weighted coverage shortfall; L the expected link loss (outage probability at the mean SNR with the channel's spread); E the simulated energy. Constants: c₁ = 1, c₂ = κ·N_demand (κ = 1), c₃ = c₂, c₄ = 0.1, all hand-set and swept. The convex Δ² is a **surrogate for staleness, not a bound**.
- **Age cap S.** S\* = 2 at every N (the smallest S covering 90 % of layouts at both pilot budgets).

### 1.4 What the evidence says

| Question | Evidence | Status |
|---|---|---|
| Is choosing reach at plan time worth it against one fixed class? | Study 5.4: F vs FB+ pinned to wide / medium / narrow, metric Network AoU, ~960 trials | **Not run** (batch 2) |
| How far is the planner from optimal? | Oracle **O1** — exhaustive band class × clustering × route × per-stop band, N ≤ 6, "after Zhai" in spirit only (Zhai solves route and selection by convex approximation to a local optimum, with no band classes) | **Built**; in batch 2 |
| Does the full system beat the schedulers it competes with? | Batch 1, time to τ = 0.71, N = 6, 1 mule, knee: **F 194 s** against H1 282, D1 282, D2 266, D3 280, D4 264 — each a claim (Holm p ≤ 0.0083; D4's is 0.00829). N = 24: F 749 against H1 951 and D3 911, but D4 (FedEx's tour) ties at 752; D4 also ties F at N = 12 (459 vs 486), so F's lead over D4 is an N = 6 result. | **Scored** |
| Where it does not win | At the stress budget, N = 6, only D4 (264 against F's 178) is a claim; 3 mules at N = 6, every arm ties with F except D4 with its own merge | Scored |

**Caveats worth stating next to the claim.** At narrow with a declared payload and a tight budget, whole-stop admission hits a 0-or-N cliff; F and FB+ fly member subsets by default to avoid it. T_nom, the deadline unit, is priced on wide for every arm, so an arm with shorter missions fits more of them into a deadline window (documented, not corrected).

---

## 2. J2 — Band and next stop (flight clock)

**What is optimised jointly.** At each Pass-1 arrival at stop k, one decision over **(band b, next stop s)**: b is the class stop k is served on *now*, priced at the SNR observed now; s is the next stop of the remainder (home only once none is left). Taking a worse band now to reach stop k+1 at a better phase of the interference cycle is exactly the trade a one-step rule cannot make.

**Why the band decision belongs on this clock.** The channel moves *within* a sortie. From `drone_env.py`'s constants (3 bands at evenly spaced phases, ω = 0.15, period ≈ 41.9 steps, hops 25–54 steps): the band that is best at takeoff lands in a trough at two of four stops, giving a mean of **−0.14 committed against +0.75 best-available at arrival**. With equal amplitudes, *none of that gap comes from committing to the wrong band; all of it comes from committing at all* (decision memo §2).

### 2.1 The five fillings of the slot

All run inside one code path (`policies/pair_slot.py`): same pass guard, scope guard, mask and fallback, so a filling differs only in how it ranks.

| Filling | Arm | Who decides | Sees | Learned |
|---|---|---|---|---|
| Committed | **F** | Nobody — the plan's order and b̄ | — | no |
| Cross-heuristic | **FX** | Fixed rule: nearest remaining stop that keeps the rest feasible; fastest class that still reaches every device b̄ reaches, priced at the arrival SNR | Distances, measured signal | no |
| Learned pair score | **FQ** | Masked pointer double DQN, 36 → 64 → 64 → 1, tanh, numpy | `pair_v1` rows | **yes** |
| One-step references | `greedy_1`, `hyb`, `fx_pair`, `committed_pair` | Scripted scorers in the same slot | Same `PairView` FQ reads | no |
| Whole-policy learned baseline | **E3** (after Chen et al.) | DQN picks the next *stop* only, bytes reward, no plan | Chen's per-stop observation | yes |
| Monolithic | **M1** (after Ho et al.) | One network over every pair, no plan or mask | — | **not built** (only if 5.5 kept a learned score) |

`greedy_1` ranks pairs by the most targets reached at the arrival SNR, then the least dwell plus travel to the next stop, then FX's tie-breaks. It is a simple function of FQ's inputs, which is why it is the right sanity reference.

### 2.2 L1 — the signal and the cost

- **Observation:** per-class SNR now (the channel model's seeded shadowing X_j(t) plus the class interference I_b(t), a regular wave of period P_c). One reading cannot tell a rising signal from a falling one, so the score gets the previous stop's offsets, their age, and where the arrival falls in the cycle (the **phase block**, 12 of the 36 columns).
- **Actuation:** the band is actuated; dwell = bytes / rate(b, SNR) is charged to the simulated clock (`mission_clock.py`), replacing the 1 s constant `session_time`.
- **L1 as a learner:** none. The contact band is chosen by L2's slot; L1 only prices.

### 2.3 L2 — the mask, the action set and the fallback

A pair is admitted only if all three hold:

1. **The band covers the plan** — it reaches every device b̄ reaches at this stop, at the measured signal.
2. **The next stop is in the plan** — the score reorders; it never invents stops.
3. **The rest of the flight still fits** — after serving here on that band and flying to that stop, the whole remaining route priced at the observed SNR meets the deadlines, the budget and the energy limit (`FLScheduler.fits_after_service`, the same S3b predicate).

If **no pair is admitted**, the mule flies FX's pair (its band rule, with no reorder, so the plan's next stop) and logs `mask_empty`. That is common: FX's own pair overruns the budget at about 19 of 71 last-stop arrivals at N = 6. After the choice the stop moves to the front of the remainder and the normal departure check still runs, trimming or re-planning. The slot never acts in **Pass 2** (a delivery sortie that flies b̄ in queue order) or at takeoff (nothing has been observed in flight).

### 2.4 L3 — the reward and the features that carry age

**Reward per decision at stop k** (decision 4 (a)):

```
r_k = G_k − c_t · Δt_k / T          (+ at the sortie's last decision:  − c_cov · U)
```

G_k is the L3 merge weight of the updates collected at k (examples × value × age discount, zero past the cutoff) over n_ref·N, so one fresh device of standard size is worth 1/N. c_t = 0.1, c_cov = 1. There is no energy term (energy tracks time) and no lateness term (none of 1,148 probed collections was late). **Training uses expected availability**: whether a device answers is a draw independent of the pair, with noise 10–100× the time signal a decision moves, so each targeted device is credited its probability of answering. A sortie is the Q's horizon.

L3 also enters the *inputs*: `age_next`, `capped_next`, `weight_share`, `on_time_next`, `slack_next`.

### 2.5 The learner

| Setting | Value |
|---|---|
| Update | Double DQN; target y = r + γ·Q_target(s′, a\*), a\* the argmax of Q_online over the **next decision's admitted rows**; Huber (δ = 1); gradient clip 10; Adam lr 1e-3; hard target sync every 500 updates |
| Replay | 50,000 transitions, each carrying the next decision's rows **and mask** (so the target takes a real max), batch 64, warm-up 1,000 |
| Behaviour | First 500 episodes ε-greedy around FX's pair at ε = 0.3; then ε falls 0.3 → 0.05 over the first half |
| Episodes | Up to 10,000, validated every 1,000; stop after 3 validations without a new best |
| Training ground | **FerrySim** — the real FeRRy stack in one process on a virtual clock. Families `jittery56` (N = 6 and 12) and `clean` (negative control); one score per channel regime |
| Checkpoint | Format 2: `.npz` + manifest, sha256 over arrays *and* header (kind, purpose, learner revision, schema, classes). The mule verifies the sha; the runner refuses untrained, unscored, dirty-tree or tag-mismatched checkpoints |

### 2.6 What the evidence says — Study 5.5

| Step | Result |
|---|---|
| **Headroom** (200 episodes/cell) | Room over FX is thin: 0.012–0.042 overall, 0.005–0.019 for the in-flight slot. ε = max(0.01, 0.1·headroom) = **0.01** everywhere. N = 6 cells barely decide (2–13 % of sorties hold 2+ decisions). `greedy_1` beats FX in all six cells |
| **Calibration** (γ ∈ {0, 0.9} × 3 seeds, *not* pre-registered) | `jittery56`: flat; γ = 0 passed the sanity check. (The verdict JSONs read the two N = 12 cells of each family, not four.) `clean`: **sanity-failed by 0.0005** |
| **Sweep** (pre-registered; γ ∈ {0, .25, .5, .75, .9, .99} × 10 seeds; 1,000 held-out episodes/cell) | Mean returns −0.0795, −0.0776, −0.0779, **−0.0774**, −0.0778, −0.0779. **Not rising** (best gain 0.002, Holm p = 0.14); **flat** (every γ equivalent to γ = 0 within ±ε, TOST p ≤ 1.35e-4); sanity check passed (3 of 10 seeds below the floor) |
| **Against fixed rules** | Best FQ (−0.0774) ≈ FX (−0.0786); trails `greedy_1` (−0.0716) by 0.0059, p = 0.002. **`replace_fx = false`**, `keep_learned = false` |
| **Consequence** | FX stays as FeRRy's in-flight rule; `gamma_star = 0.75` recorded only; FQ arms leave batch 2 except 5.5's stack check (FQ-g75, FQ-g0); M1 not built |

**The diagnosis, held as a hypothesis not a result.** FQ at γ = 0 sees what `greedy_1` sees yet lands on FX, not on `greedy_1`. That points to a learner that *copied FX*, though no agreement-with-FX share of the trained scores is recorded, so it is untested: the 500 FX-referenced episodes are only about 6 % of the replay, and exploration then tapers from ε = 0.3 to 0.05. The validation bounce is as large as ε, and the TD loss hardly moves. The proposed one-revision screen on the clean control cells (more exploration; lr 3e-4; both; steadier selection) is written down but was **not run** — option A (the pre-registered sweep) was chosen. If reviewers press, it can run in the revision window. Until then the paper can say the learned score matched FX, not that no learner could do better.

### 2.7 E3 — the learned competitor with no FeRRy machinery

E3 (`policies/chen_dqn.py`, `policies/next_stop.py`) is a numpy port of Chen et al.'s DQN recipe, run as a legacy-mode whole-scheduler policy: it admits every S3a contact, then names the next stop at takeoff and at every Pass-1 departure among the stops S3b's budget predicate admits (Chen's *safety controller* — the same gate-then-learn pattern as FeRRy's mask). It flies the cell's one band, uses no deadline, plan, coverage term or age cap, and is rewarded in **bytes** (|C_k|/N). Five trainings at γ = 0.99 (Chen's code); the five seeds' held-out bytes returns are within 0.4 % (1.933–1.941); `rl.checkpoints.e3` is seed 0. Declared deviations: stops, not grid moves; one agent (QMIX with one agent *is* DQN); no learned digital twin; the pair learner's masked double DQN. It is labelled **"DQN over the contact graph, after Chen et al."**, never FedQMIX, and scored on update yield and round closure, never raw bytes.

---

## 3. J3 — The return path (both clocks)

**What is coupled.** What flight time learns has to reach the plan. The decision memo named three severed cuts; the build closes each.

| Cut (memo) | Was | Closed by |
|---|---|---|
| **1 — no contact-band decision** | `rf_range_m` a constant; no band on the contact link | `contact_link.py`: band classes with a range–rate model; b̄ in the plan |
| **2 — band decided on the wrong clock** | L1 re-selected per mission; the prior reaching the selector was one mission-mean scalar | Per-arrival band in the flight slot; the 36-column per-pair rows with per-class SNR |
| **3 — the return path** | Dwell was a constant `session_time = 1.0 s`; the only upward edge, `_remaining_is_feasible()`, could only **abort** | Rate-dependent dwell on a simulated mission clock; `routing/replan.py` re-plans instead |

**How the loop closes now.** After each stop the predicate re-runs on the remaining queue at the *observed* rate. If it fails, `replan_remainder` repairs it under the one predicate: protected (age-capped) stops first, then the arm's own admission, then the arm's own order (`trim`) or 2-OPT / EDF order (`reorder`). Drops are final for the mission. The pilot plan the user accepted on 2026-09-29 runs `replan` with `trim`, so each arm keeps its order, as the D arms do.

**Layer by layer.**

- **L1 → L2:** rate and SNR give dwell, dwell moves the clock, the clock decides feasibility.
- **L2 → L1:** the chosen stop and arrival time determine the phase the next observation lands in.
- **L3 → L2:** the age cap makes capped devices first in the plan key and exempts all-capped stops from their own deadline clause.
- **L2 → L3:** devices a plan leaves out are widened as misses, so ages feed the next mission's demand.

### 3.1 The RL test, as three properties on two clocks

Designed to *measure* the case for learning, not assert it (decision memo §5). Where each stands:

| Property | Plan time | Flight time |
|---|---|---|
| **(a) Decomposition** — does planning against a nominal rate and reacting later lose to anticipating? | Oracle O1's optimality gap for each decomposition, N ≤ 6. **Built, not run.** | *(spans both)* |
| **(b) Delayed consequence** — does the choice at k change what stays reachable at k+1…? | γ swept across **missions**. **Not tested** | γ swept across **stops within one sortie**. **Tested (Study 5.5): flat** — a one-step rule at each arrival suffices |
| **(c) Non-stationarity** — does the best rule change with the regime? | Clean / jittery × tight / slack. Study 5.6 as built | Transit/period swept; settled by construction since the channel moves within a sortie, so it earns no credit on its own |

Reading the memo's (b)-row decision rule against the result: flat on the flight side means **no hierarchical policy is entered**, and the plan-time column is untested only because nothing justified building a learned planner.

---

## 4. J4 — One objective across layers (L2 ↔ L3)

> **Correction (7 Oct, from documents 04 and 05).** "One objective" is a shared design idea, not literally one quantity. The merge weight *falls* with age (a discount on merge age), while V's coverage weight *rises* with age (about a² on plan age), and the flight reward mixes the two: G_k uses merge weights, U uses V's weights. In every cell flown or trained so far, G_k reduces to devices collected ÷ N, because ages are 0 and value is uniform. The Study 5.5 null is also conditional on c_t = 0.1 and c_cov = 1; the reward grid will not run, so other weightings are untested.

**What is coupled.** A single quantity — the L3 merge weight — is meant to price the plan (V), the flight reward and the merge, so L2's decisions and L3's aggregation optimise the same thing (claim C2). The coupling is *by construction*, not by training.

| Where the weight appears | Layer | Form |
|---|---|---|
| Merge | L3 | w ∝ n·v·s(age), **zero past a_max** (`agg:cutoff`); the normaliser is staleness-free, so a stale mission moves θ less |
| Cluster merge | L3 | Same weight, age in cluster rounds, server-rate mixing instead of overwrite |
| Plan | L2 | V's coverage weights (`coverage_weight`), age-weighted by default; the Δ² term as a convex staleness surrogate |
| Flight reward | L2 (trains from L3) | G_k = merge weight collected at stop k |
| Admission | L2/L3 | Age cap S and the priority key |

**What is and is not derived.** V's constants and the reward form are **hand-set** (a sweep is defined, but no run record of it was found); the theory-track derivation from FedEx-Async's bound (Σ R_k·Δ_k², for clients that train without pause) is not done, and HERMES devices train once per visit, so the bound does not carry over directly. Likewise `agg:cutoff`'s age weights are FedAsync's hinge with hand-set constants. This is an open decision (Readiness, "Decisions still open").

**What the evidence says.** Study 5.7 tests the dwell and coverage terms and the derived-versus-hand-set reward. With 5.5's null, **F·hand and the reward grid left with the learned scores**; 5.7 now flies FX, **FX-dwell** (no dwell term), **FX-cov** (no coverage term) and D4 as the travel-only reference — 40 trials per cell at N = 12, round closure against FX, no pre-registered ε. Built 7 Oct; **not run**. Study 5.1 (aggregation arms against `agg:plain`, `fedbuff`, `asynchfl`) sits in the same layer and has not run either.

---

## 5. J5 — Prototype lineage and baselines (L1 + L2)

### 5.1 `hermes_rl/` — the original joint action

`hermes_rl/drone_env.py` is a Gymnasium environment with **one joint discrete action (waypoint × base station × channel)**, 5 fixed waypoints and 3 base stations × 3 channels. Transit consumes time with no transfer; the upload rate is `R_k(d, t) = α·sin(ω·t + φ_k) + β/(1 + γ·d)` with α > β, so the time-varying term dominates distance. `train_dqn.py` trains a **hybrid** in which the DQN picks the *job* (one of the environment's jobs, with feasibility scores appended to the observation) and a heuristic then executes it, choosing the waypoint, base station and channel (`train_hybrid`, `execute_for_job`); `--mode dqn` runs the standalone DQN over the full joint action space for comparison. *(An earlier version of this document, following the module docstrings, said the DQN picked base station and channel; the code does not.)* So the prototype's **learned** part is job selection; the band decision is the environment's sinusoidal channel, which its constants use to produce the §2 table above.

**Two cautions.** Its trainer originally selected its best checkpoint and ran its final evaluation on the same fixed episode (the environment ignored seeds by default), so SEC'26 Table VI's 76.40 is a best-of-about-30 score on one instance (the committed plot shows about +76 only at episodes 50–250, then about −5, the heuristic's level, from episode 300). The copy under `experiments/sim/drone_env/` is seeded and pinned by golden rollouts; and the code has **no licence from its author** (an open item: ask or remove).

### 5.2 How the design simplified, layer by layer

| Version | L1 | L2 | Joint? |
|---|---|---|---|
| **MA-P-DQN** (original design) | Discrete channel head | Continuous trajectory head (Δposition) | Joint (Δposition, channel) |
| **Reframe** (HERMES_FL_Scheduler_Design §2.6) | Channel-only DDQN | Next-target selector; "navigation is mechanical" | Joint dropped |
| **As evaluated in Exp 4** | Backhaul band by a deterministic utility; DDQN untrained | Intra-bucket DDQN selector (random-init); runs only when a bucket holds ≥ 2 candidates | None. H2 tied H1 — the mule saturates the outcome and the old action had no consequence to learn from |
| **FeRRy** | Band classes, range–rate model, simulated clock | Plan search (J1) + flight slot (J2) | Joint, two clocks |

The old selector is a data point: (selection-only, FL-aware, *learned*) tied (selection-only, FL-aware, fixed). Learning did not help while the policy could only reorder. *(Caveat from document 02: the tie rests on the sweep-B probe in the Holistic Revision Plan, which found byte-identical outputs at N = 6 for every budget from 120 s to 15 s, and on the decision memo's statement; `HERMES_Matrix_Results.md` itself has only H1 vs H0 and H3 vs H2. It shows the arms did not differ, not that a learned selector failed to help.)*

### 5.3 Where each method sits in the 2 × 2

Two axes separate the candidates: **decision scope** (permute within a given route, or choose the next stop) and **objective denomination** (FL units, or bytes and age).

| | **FL-blind** (bytes, age) | **FL-aware** (updates, rounds, deadlines) |
|---|---|---|
| **Selection-only** | D1 MAX-AoI | D2 Oort; D5 FedCS (degraded); the old H2 selector |
| **Trajectory + selection** | **E3** — DQN over the contact graph | **FX**, **FQ**, F (committed) — same architecture, fixed / learned / no flight slot |

Three tests fall out: scope effect (read down), awareness effect (read across), and the **interaction** — whether trajectory control is worth more when the objective is FL-aware (difference of differences on paired seeds). The RL question is **nested** in the bottom-right cell: FQ vs FX holds the clock structure constant, so a fixed-rule win still reads "the architecture is right and a heuristic suffices to exploit it."

---

## 6. Summary matrix

Rows: joint optimisations. Columns: layers. Each cell says what that layer does *for* that method, and whether that part is learned.

| | **L1 — RF** | **L2 — Scheduling** | **L3 — Federated learning** |
|---|---|---|---|
| **J1 Reach and route** | Band classes set range and rate; mean SNR priced. *Fixed.* | S3a per class, exhaustive / 2-OPT search, S3b gate. *Search, not learned.* | Age-weighted coverage, deadlines, cap S in V. *Hand-set.* |
| **J2 Band and next stop** | Observed SNR per class, phase block, dwell charged to the clock. *Prices only.* | Pair mask, scope guard, FX fallback, reorder + re-plan. **Learned score (FQ) tested: flat.** | Reward G_k from merge weights; age features. *Reward only.* |
| **J3 Return path** | Rate → dwell → clock. *Fixed.* | Feasibility predicate; `replan` replaces abort. *One shared predicate.* | Age cap protects stops through re-plan. *Fixed.* |
| **J4 One objective** | — (dwell enters Δ) | V, c_t, c_cov constants. *Hand-set; sweep defined, not run.* | Merge weight, `agg:cutoff`, cap. *Hand-set hinge.* |
| **J5 Lineage / E3** | Channel head (prototype); one band (E3). *Learned in the prototype.* | Waypoint heuristic + DQN (prototype); next-stop DQN (E3). *Learned.* | None — bytes reward. |

**Where the learning is, in one line per layer.** L1: nothing (the DDQN is retired; the backhaul controller is a utility). L2: the only learned component is the in-flight pair score FQ, and it did not beat FX. L3: no learning — the merge is a hand-set weighted average whose weights the reward reuses.

---

## 7. Open items

- [ ] **Study 5.4 and O1** — the plan-time reach test and its optimality gap; in batch 2 (~50 h, ready).
- [ ] **Study 5.6** — now E3 against FX (and transit/period), since M1 is not built; batch 2.
- [ ] **The one-revision screen** (option B in the calibration findings) — only if reviewers press on "the learner copied FX". It must run on the clean control cells, not the family the sweep flies.
- [ ] **A plan-time (b) test** — γ swept across missions — is untested; the decision memo's rule says do not build a learned planner without it.
- [ ] **Bound-derived merge weights (J4)** — a derivation, a sensitivity check on the hand-set constants, or drop it; decided before batch 2's 5.1 and 5.7.
- [ ] **Citations** — Bayerlein et al. 2021 into `HERMES_Related_Work_Notes.md` §3 and §7, with a full-text verification mark. (Chen 2023 was added on 6 Oct, commit `5b9a7aec`.)
- [ ] **`drone_env` licence** — ask the author or remove the vendored copy.

---

## 8. Sources

[Layer redefinition and RL decision memo](architecture%20review/rl-decision-memo/HERMES_Layer_Redefinition_and_RL_Decision.md) · [Experiment 5 RL calibration findings](Experiment_5_RL_Calibration_Findings.md) · [Experiment 5 readiness](Experiment_5_Readiness.md) · [FeRRy build plan](FeRRy_Build_Plan.html) · [FeRRy learned pair score](FeRRy_Learned_Pair_Score.html) · [FL scheduler design](HERMES_FL_Scheduler_Design.md) · [System architecture overview](architecture%20documents/System_Architecture_Overview.md) · [SOTA baseline candidates](HERMES_SOTA_Baseline_Candidates.md) · module docstrings in `hermes/l1/contact_link.py`, `hermes/scheduler/plan/{plan_search,plan_score}.py`, `hermes/scheduler/policies/{cross_heuristic,pair_slot,chen_dqn,next_stop}.py`, `hermes/scheduler/routing/replan.py`, `hermes/scheduler/selector/{pair_q,pair_features,pair_replay}.py`, `hermes_rl/{drone_env,train_dqn}.py`.

*Numbers are copied from those records as of 7 Oct 2026; none were re-run for this document. Per-topic detail is in the [dedicated documents](Joint%20RL%20Methods/README.md).*
