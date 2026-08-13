# Experiment 4 — Layer 2: the mobility-aware scheduling layer

**Scope.** What the gated scheduler *actually does* in an Experiment-4 trial — traced against
source, not against the design documents — and, at equal length, what it does **not** do here.
Companion to [`HERMES_Experiment4_Methodology_and_Implementation.md`](HERMES_Experiment4_Methodology_and_Implementation.md)
and [`HERMES_Experiment4_L1_RF_Layer.md`](HERMES_Experiment4_L1_RF_Layer.md).

> **Read §4 before citing this layer in the paper.** Experiment 4 exercises a materially narrower
> slice of Layer 2 than the four-stage description implies: several stages the paper describes are
> inert at runtime (§4.1, §4.4).
>
> **Deadline enforcement was missing and has since been added** (S3b, §4.2) — but it is opt-in and
> was **off for every committed result**, and Exp 4 still models no flight budget or propulsion
> energy. Claims about the deadline *design* therefore remain Experiment 3's territory until an
> Exp-4 run is made with a binding budget.

---

## 1. The path Exp 4 actually takes

Exp 4 always constructs the mule with a non-`None` `rf_range_m` (default 60 m,
`experiments/exp4/driver.py:75,155`; `topology_builder.py:146`), which forces
`MuleSupervisor.run_one_mission` down the Sprint-1.5 **two-pass** branch
(`hermes/mule/mule_main.py:247-249`).

> **Consequence, and a correction to the design docs:** `build_target_queue` — the single-pass API
> named throughout the design material — is **never called in Experiment 4**. The two entry points
> that run are `FLScheduler.build_contact_queue` (Pass 1) and `build_pass_2_queue` (Pass 2).
> The per-device S3.5 placeholder `select_order` (`stages/s35_selector.py:28-45`) and
> `TargetSelectorRL.rank` are therefore dead code in this experiment; the live selector entry point
> is `TargetSelectorRL.rank_contacts`.

**The Pass-1 pipeline, in execution order:**

```
S1  eligibility        →  S3  deadline + bucket classify (per device)
                       →  S3a RF-range clustering into ContactWaypoints
                       →  bucket-priority walk (outer loop over BUCKET_PRIORITY)
                       →  S3.5 intra-bucket ordering  (deterministic | RL)
```

## 2. Stage by stage

| Stage | Role in Exp 4 | Hard gate or ordering? |
|---|---|---|
| **S1 — eligibility** | admits on mission-slice membership | **hard gate** (removes candidates) |
| **S3 — deadline + bucket** | computes `deadline_ts`, classifies into buckets | **rank tier** — a hard tier the selector cannot cross |
| **S3a — RF clustering** | groups devices within `rf_range_m` into `ContactWaypoint`s | lossless regrouping (no removal) |
| **S3b — deadline feasibility** | drops contacts that cannot be reached in time (opt-in, §4.2) | **hard gate** (removes candidates) |
| **S3.5 — intra-bucket** | orders candidates *within* one bucket | **ordering only** |
| **S3c — mission window** | scales the S3 fulfilment term from mission history (opt-in, §4.8) | neither — it changes *when*, not *who* |
| S2A — readiness | **not invoked** (see §4) | — |
| S2B — FL_READY threshold | **not invoked** (see §4) | — |

**The architectural guarantee holds, and is enforced three ways.** The paper's central claim is
that learning cannot override hard constraints. In this code path that is true, and not merely by
convention:

1. The candidate list handed to the selector is already the post-S1/S3/S3a bucket membership
   (`fl_scheduler.py:401-421`).
2. An explicit **scope guard** re-checks every contact member against the round's admitted set
   (`target_selector_rl.py:285-287` → `scope_guard.py:41-45`).
3. A **pass-kind guard** hard-fails if the selector is ever invoked during Pass 2
   (`target_selector_rl.py:52-66`).

The selector returns a *permutation of its input list* and nothing else — it cannot add, drop, or
re-bucket a candidate. The bucket walk is an outer loop, so it cannot promote a device across a
tier either.

## 3. The S3.5 selector (arms H2/H3)

A pointer-style scalar-Q network, reached through exactly one call site
(`fl_scheduler.py:421-427` → `TargetSelectorRL.rank_contacts`), and invoked only when a bucket
holds ≥2 contacts.

* **Architecture:** each candidate is scored independently by a single **11 → 16 → 1** tanh MLP;
  candidates are sorted by descending Q. The action space is the candidate list itself, not a fixed
  head — which is what lets one network rank a variable number of contacts.
* **State:** `FEATURE_DIM = 11` (`selector/features.py:49`). **Slot 10 is `rf_prior_snr_db / 30`** —
  the L1→L2 edge described in the L1 document.
* **Inference-only here.** No `.update()`, no `ReplayBuffer`, no `set_epsilon` anywhere in
  `hermes/mule`, `hermes/processes` or `experiments/exp4`. Training lives solely in
  `selector_train.py`, which Exp 4 never imports. **The selector does not learn during a trial.**

**What actually differs between H1 and H2 at runtime:** *only the within-bucket ordering of contact
waypoints.* Both arms visit every contact in the queue exactly once. H2 is not a different set of
devices, nor a different number of contacts — it is the same work in a different order.

## 4. What Experiment 4 does **not** exercise

This section exists so the paper does not claim Exp 4 validates parts of L2 that never ran.

### 4.1 Stage 2 never executes

`FLScheduler.ingest_ready_adv` — the only caller of S2A (readiness) and S2B (FL_READY threshold) —
**has no runtime callers**; it is reached only from unit tests. The design's `FL_Threshold = 0.60`
and the 5 s advert-freshness window are therefore never applied in Exp 4.

The gate that *does* fire is a separate inline check inside `HFLHostMission.run_contact`, with
`min_utility = 0.0` — which cannot reject anything, given devices pin themselves to `FL_OPEN` for
the whole run.

### 4.2 The deadline was computed but never enforced — now fixed, and off by default

**As traced, this was the most important limitation:** `deadline_ts` was computed and used as a
**sort key**, but nothing in the mule/mission/exp4 path ever compared it to a clock. A device whose
deadline had passed was still queued and still visited, so "deadline-aware scheduling" was not true
of the code.

**The gap is now closed** by **S3b — the deadline feasibility gate**
([`stages/s3b_feasibility.py`](../hermes/scheduler/stages/s3b_feasibility.py)). It runs between S3a
and the bucket walk — i.e. **before ordering**, so the learned selector cannot resurrect anything it
drops, preserving the architectural contract. A contact is dropped when either:

* it cannot be **reached before its own `deadline_ts`** (given transit at `cruise_speed_m_s`), or
* serving it would **overrun the remaining mission budget**.

The walk is greedy in EDF order, advancing a simulated pose and clock, so later estimates account
for earlier stops. Rejections are returned by reason (`dropped_overdue` / `dropped_budget`) and
logged, so a device that is not served can be explained rather than vanishing.

> **It is opt-in, and that is deliberate.** With no `mission_budget_s` configured the gate is a
> strict no-op — pinned by test — so **every previously recorded result remains reproducible**.
> Enforcement turns on by supplying a budget:
> `--mission-budget-s <seconds>` (→ `MuleConfig.mission_budget_s` → `FLScheduler`).

> Note also that Exp 4's `deadline_met` column is **redefined** as a quorum-plus-backhaul indicator
> (≥1 update **and** the backhaul upload was not dropped). It is not the L2 deadline.

#### What enforcement actually costs — measured

Two measurements, one cheap and one real.

**(a) Where the gate binds** — [`probe_s3b_binding.py`](../experiments/exp4/probe_s3b_binding.py)
runs the real scheduler over the real device layouts with no subprocesses, and separates the two
ways the gate can bite (N=6, rf_range 60 m, field radius 100 m, cruise 5 m/s, 20 layouts):

| | threshold |
|---|---|
| **Deadline floor** | **~34 % of contacts (26/76) are dropped at _any_ budget** — they cannot be *reached* before their own deadline at 5 m/s. Independent of `mission_budget_s`. |
| **Budget knee** | **~60 s.** Below it budget drops appear and grow: 46 % dropped @50 s, 58 % @40 s, 72 % @30 s, 93 % @5 s. |

**(b) What it costs end-to-end** — a 5-shard sweep (H1, jittery, N=6, `n_missions=4`, **20 paired
seeds per shard, 100/100 valid trials**), each budget compared to the gate-off control *paired by
seed* (`results/exp4_s3b/`, analysed by
[`analyze_s3b_sweep.py`](../experiments/exp4/analyze_s3b_sweep.py)):

| Budget | update_yield | mission_completion | round_close@2 | Δ completion vs control |
|---|---|---|---|---|
| **gate OFF** (control) | 2.09 | 0.767 | 0.663 | baseline |
| **120 s** — deadline only | 1.26 | **0.542** | 0.388 | **−0.225** ✱ |
| 60 s — at the knee | 1.20 | 0.483 | 0.375 | −0.283 ✱ |
| 30 s | 0.55 | 0.308 | 0.100 | −0.458 ✱ |
| 15 s | 0.43 | 0.283 | 0.013 | −0.483 ✱ |

✱ = CI excludes 0 **and** paired Wilcoxon p<0.05.

**The 120 s row is the result.** That budget is deliberately slack — well above the 77.8 s mean
queue cost — so almost nothing is dropped for *budget* reasons. The entire loss is contacts that
cannot be reached before their own deadline, and it costs **−0.225 mission completion (−29 %)**,
closely matching the probe's ~34 % contact-drop prediction.

> **So the deadline was never slack in Experiment 4.** Roughly a third of the queue was being served
> in violation of deadlines the scheduler had already computed; it only *looked* slack because
> nothing checked. Enforcement does not add a constraint — it reveals one that was always there.

Two secondary reads:

* **The budget is not the binding constraint at realistic values.** 120 s → 60 s barely moves
  anything (yield 1.26 → 1.20), exactly as the probe predicted. Below ~30 s it dominates and
  round-closure collapses to ≈0.
* **Do not read the AUC column as an ordering.** 60 s shows *higher* AUC than 120 s despite being
  tighter — the bimodal collapse mode (§3.1 of the validity record) dominates at this n.
  Participation is the trustworthy signal.

> **What this does and does not change for the paper.** Exp 4 can now speak to deadline
> *enforcement* and its cost. It still cannot speak to the deadline *design* — the adaptation rule,
> its bounds, its sensitivity — because there is still no propulsion-energy or flight-budget model
> here; that remains Experiment 3's territory. And **every committed Exp-4 result in
> `results/exp4_paper/` was produced with the gate off**, so those numbers describe the
> non-enforcing configuration.

### 4.3 No flight budget, travel time, or energy

There is no flight-budget, travel-time, propulsion-energy or communication-energy model anywhere in
the Exp-4 path — all of that lives in Exp 3's `sim_env`. `mule_energy` is a frozen `1.0`, never
mutated anywhere in `hermes/`.

### 4.3a The bucket tiers never discriminate

S3 classifies each contact into `NEW` / `SCHEDULED_THIS_ROUND` / `BEACON_ACTIVE`, and
`build_contact_queue` walks them in `BUCKET_PRIORITY` order — a **hard tier the selector cannot
cross**, and one of the three mechanisms enforcing the architectural guarantee (§2).

**In Experiment 4 that tier is never exercised.** Probing the real scheduler across six rounds:

| round | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| buckets present | `NEW`×6 | `SCHEDULED`×6 | `SCHEDULED`×6 | `SCHEDULED`×6 | `SCHEDULED`×6 | `SCHEDULED`×6 |

Every round contains **exactly one non-empty bucket**: all devices are new in round 1 and settled
thereafter, and `BEACON_ACTIVE` never populates because no beacon source is wired (§4.4). So the
priority walk is a no-op with respect to ordering — with one bucket, walking it in priority order
changes nothing.

**Two consequences, and they pull in opposite directions:**

* **Against us:** Exp 4 offers **no evidence** that the tier mechanism works. It is implemented,
  correct and tested, but unexercised here — the same status as S2A/S2B and beacons. It should be
  stated as a scope limit, not claimed.
* **For us:** it is what makes the **B1 MAX-AoI baseline fair**. The worry was that a policy in the
  `target_selector` slot only re-orders *within* a bucket, so our tiers would outrank the
  baseline's ranking and flatter us. With one bucket per round, a policy in that slot orders the
  **entire round** — B1 is a genuine full-ordering baseline rather than a re-ranking inside our own
  structure.

### 4.4 Beacons are dead; S1 rejects nothing

No beacon source is wired, so `ingest_beacon` is never called, `last_beacon_ts` stays 0, and the
`BEACON_ACTIVE` bucket is unreachable. S1 admits on slice membership alone and **rejected zero
devices in 7,200 device-missions** of probe. The `deadline_overrides` short-circuit in
`compute_deadline` likewise never fires.

### 4.5 Pass 2 explains the `coverage` metric

Pass 2 clusters and visits **every** device with `is_in_slice=True`, explicitly bypassing S1's
eligibility gate and ignoring whether Pass-1 collection succeeded (`fl_scheduler.py:463-465`), and
the device emits a `device_served` event on **both** passes. So `coverage` is a by-construction
descriptor of the two-pass design, **not an outcome measurement** — which is why the methodology
document forbids headlining it.

### 4.6 The selector's inputs are largely constant

Four of the eleven feature slots are constant within every Exp-4 ranking batch, so the selector is
discriminating on a narrower signal than the feature vector suggests.

### 4.7 Scale

N=6 devices, one mule. Multi-mule coordination and larger-N bucket behaviour are untested here.

### 4.8 The mule flew doomed queues, and the gate could starve its own drops — now fixed, both off by default

Enforcing the deadline (§4.2) fixed one defect and created two more. Both are now closed, and like
enforcement itself both are **opt-in**, so every recorded result stands.

**(a) The abort was missing.** S3b filtered the queue *before* take-off and nothing re-checked en
route. Contacts take real time and can fail, so the mule falls behind its own plan — and then keeps
flying stops it can no longer serve, burning budget and delaying delivery of the updates already
aboard. `MuleSupervisor._remaining_is_feasible()` now re-runs the S3b check from the mule's
**current** pose and clock before each stop; when the remainder is unreachable the Pass-1 loop
breaks and `close_round` + the dock deliver what was collected.

> **What "knows it will fail" can and cannot mean.** This foresees running out of **time** — a
> deterministic function of clock and geometry. It cannot foresee a *random link failure*, which is
> stochastic by construction. The edge case is implemented in its knowable form; the unknowable
> half is not a gap to be closed but a property of the model.

**(b) The gate could starve the devices it dropped.** `RoundCloseDelta` is emitted only from inside
a contact session (`host_mission.py:838`). A device dropped by S3b, or abandoned by an abort, has
no session — so it received **no feedback at all**, its fulfilment window never widened, and it was
just as un-serveable next mission. *A starvation loop created by the S3b gate itself.*
`_widen_abandoned()` now feeds those devices a `TIMEOUT` delta, exactly as a missed contact does.

**(c) S3c — the signal none of the per-device loops can see.** Fixing (b) closes the loop *per
device*. But every loop in S3 is per-device, and none of them can diagnose **"the mule is
systematically failing to complete its circuit"** — from any single device's point of view, a
schedule that is globally too tight looks identical to ordinary bad luck. So S3c adds a second,
mission-level signal: the mule reports `served/planned` after each mission, and while the rolling
rate sits below target the adapter widens **every** device's window together.

Four properties are worth stating because each rules out a way this could have gone wrong:

* **Derived, not accumulated.** The scale is a pure function of the recent record, recomputed on
  each read rather than nudged up and down — so it cannot wind up or drift, and reading it never
  perturbs it.
* **Widen-only.** At or above target the scale is exactly 1.0. Shrinking stays with the per-device
  rule, which knows *whom* to reward; S3c only knows the fleet is behind.
* **Bounded.** `max_scale` caps the stretch, so an impossible configuration degrades to "windows
  are wide" rather than "windows are unbounded".
* **The denominator includes S3b's own pre-flight drops.** Counting only the surviving queue would
  let the gate flatter itself — drop nine contacts, serve the tenth, report 100 % success, and
  never widen. That is the starvation loop hiding inside its own success metric.

S3c is **not a gate**: it changes no admission and no ordering, only the S3 term that S3b later
tests against. Enable it with `--mission-window-adaptation`; the four tunables
(`--mission-window-target|gain|history|max-scale`) are matrix parameters, not code defaults.

**Expect it to do nothing without a budget.** If the deadline is only a sort key there is nothing
for a wider window to rescue, so an adaptation-only arm should read as a tie by construction — a
useful negative control, and a check that the toggle is wired the way this section claims.

---

## 5. Two defects this trace uncovered

### 5.1 The H2/H3 dead-zone sweep did not vary the dead zone

`dead_zone` and `link_quality` are consumed **only inside the H0 branch**
(`driver.py:304-308`); the mule arms' `realism_kwargs` (`driver.py:175-188`) contains neither.

> **Therefore the five "dead-zone" cells of the H2-vs-H3 sweep are the same physical
> configuration**, differing only in the derived per-trial seeds (because `dead_zone` is part of
> `cell_id`, which seeds the trial).
>
> This does **not** invalidate the L1 null — if anything it strengthens it, since the result becomes
> ~100 paired seeds of one jittery configuration all tying, rather than five conditions. But the
> framing as a *severity sweep* is wrong wherever it appears, and is corrected in the L1 document,
> the validity record §7.4, and the revision plan.

### 5.2 The untrained selector was not reproducible — now fixed

`TargetSelectorRL(..., rng_seed=0)` seeded only the **epsilon** RNG — unused at `epsilon=0.0`. The
network itself was built by `DDQN(feature_dim=FEATURE_DIM)` **with no seed**, so its weights came
from OS entropy: two selectors constructed with `rng_seed=0` received **different weights**
(verified empirically).

**Fixed** — `rng_seed` is now threaded to the network
(`DDQN(feature_dim=FEATURE_DIM, seed=rng_seed)`). Same seed → identical weights; different seed →
different weights; no seed → still nondeterministic, by design. Regression-tested.

> **Consequence for the committed data:** the H2/H3 rows already on disk were produced under the old
> behaviour, so they are still not byte-reproducible on the selector component. This does not
> manufacture an effect — the result is a null and both arms drew fresh weights — but a re-run is
> required before H2/H3 can be described as reproducible.

---

## 6. What can honestly be claimed from Exp 4 about L2

**Can be claimed**

* The gated pipeline runs end-to-end on the real multi-process stack, and the two-pass
  collect → dock → deliver lifecycle completes under impairment.
* The architectural guarantee is real and code-enforced: the learned selector can only reorder
  within an already-admitted bucket, with a scope guard and a pass-kind guard behind it.
* A zero-update mission no longer aborts the sortie — a Pass-1 `close_round` raising
  `MissionSessionError` is caught, flagged `empty=True`, θ is restaged, and the sortie continues.

**Cannot be claimed**

* Anything about the **deadline function** — computed, never enforced here (§4.2).
* Anything about **readiness / FL_READY gating** — never invoked (§4.1).
* Anything about **energy, flight budget, or trajectory cost** — not modelled (§4.3).
* Any **RL benefit**: the selector ran untrained, unseeded, on partly-constant features, and the
  only runtime difference from H1 is within-bucket ordering (§3, §5.2).

---

## 7. Where the code lives

| What | Path |
|---|---|
| Scheduler entry points (Exp-4 path) | `FLScheduler.build_contact_queue` / `build_pass_2_queue` — [`hermes/scheduler/fl_scheduler.py`](../hermes/scheduler/fl_scheduler.py) |
| Stages | [`hermes/scheduler/stages/`](../hermes/scheduler/stages/) — `s1_eligibility`, `s2a_readiness`*, `s2b_flag`*, `s3_deadline`, `s3a_cluster`, `s3b_feasibility`, `s3c_mission_window`, `s35_selector`* (*not on the Exp-4 path) |
| Baseline policies (arms B1/B2) | [`hermes/scheduler/policies/`](../hermes/scheduler/policies/) — `max_aoi.py` (MAX-AoI greedy, B1), `oort.py` (Oort statistical utility, B2), plus Exp-3's `arrival_order` / `edf_feasibility`. All expose `rank_contacts`, so they swap through the same `target_selector` slot. **Outside the frozen surface** |
| In-flight abort + abandoned-device widening | `MuleSupervisor._remaining_is_feasible` / `._widen_abandoned` — [`hermes/mule/mule_main.py`](../hermes/mule/mule_main.py) |
| Mission accounting that feeds S3c | `mission_planned_devices` / `mission_served_devices` — [`hermes/mule/mule_main.py`](../hermes/mule/mule_main.py) |
| Selector | [`hermes/scheduler/selector/`](../hermes/scheduler/selector/) — `target_selector_rl.py`, `features.py`, `ddqn.py`, `scope_guard.py` |
| Mule supervisor / two-pass mission | [`hermes/mule/mule_main.py`](../hermes/mule/mule_main.py) |
| Subprocess wiring (`use_rl_selector`) | [`hermes/processes/mule.py`](../hermes/processes/mule.py) |
| Arm definition | [`experiments/exp4/driver.py`](../experiments/exp4/driver.py) |
