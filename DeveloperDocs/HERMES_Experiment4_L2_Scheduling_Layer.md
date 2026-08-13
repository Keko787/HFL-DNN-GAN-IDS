# Experiment 4 — Layer 2: the mobility-aware scheduling layer

**Scope.** What the gated scheduler *actually does* in an Experiment-4 trial — traced against
source, not against the design documents — and, at equal length, what it does **not** do here.
Companion to [`HERMES_Experiment4_Methodology_and_Implementation.md`](HERMES_Experiment4_Methodology_and_Implementation.md)
and [`HERMES_Experiment4_L1_RF_Layer.md`](HERMES_Experiment4_L1_RF_Layer.md).

> **Read §4 before citing this layer in the paper.** Experiment 4 exercises a materially narrower
> slice of Layer 2 than the four-stage description implies. Several stages the paper describes are
> inert at runtime, and the deadline machinery — the part reviewers asked about — does not bind in
> this experiment at all. That is Experiment 3's territory.

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
| **S3.5 — intra-bucket** | orders candidates *within* one bucket | **ordering only** |
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

### 4.2 The deadline is computed but never enforced

**This is the most important limitation.** `deadline_ts` is computed and used as a **sort key**, but
nothing in the mule/mission/exp4 path ever compares it to a clock. Deadlines sit 35–110 s in the
future and never elapse. `n_missions` is the only real termination bound.

> Experiment 4 therefore provides **no evidence about the deadline design** — not about the
> adaptation rule, not about bounds, not about feasibility. Reviewer questions about the deadline
> function must be answered from Experiment 3, which does model a flight budget.
>
> Note also that Exp 4's `deadline_met` column is **redefined** as a quorum-plus-backhaul indicator
> (≥1 update **and** the backhaul upload was not dropped). It is not the L2 deadline.

### 4.3 No flight budget, travel time, or energy

There is no flight-budget, travel-time, propulsion-energy or communication-energy model anywhere in
the Exp-4 path — all of that lives in Exp 3's `sim_env`. `mule_energy` is a frozen `1.0`, never
mutated anywhere in `hermes/`.

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

### 5.2 The untrained selector is not reproducible

`TargetSelectorRL(..., rng_seed=0)` seeds only the **epsilon** RNG — which is unused at
`epsilon=0.0`. The network itself is built by `DDQN(feature_dim=FEATURE_DIM)`
(`target_selector_rl.py:91`) **with no seed**, so its weights come from OS entropy.

Verified empirically: two selectors constructed with `rng_seed=0` receive **different weights**.

> **Consequence:** H2/H3 rows are not byte-reproducible on the selector component, and the arms are
> unpaired with respect to it. This does not confound the committed H3-vs-H2 comparison in a way
> that manufactures an effect — the result is a null, and both arms draw fresh weights — but it does
> mean a re-run will not reproduce the same orderings.
>
> **Recommended fix (not yet applied):** thread the seed through to the network, e.g.
> `DDQN(feature_dim=FEATURE_DIM, seed=rng_seed)`. Applying it changes the weights relative to the
> committed runs, so it should be paired with a re-run rather than done silently.

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
| Stages | [`hermes/scheduler/stages/`](../hermes/scheduler/stages/) — `s1_eligibility`, `s2a_readiness`*, `s2b_flag`*, `s3_deadline`, `s3a_cluster`, `s35_selector`* (*not on the Exp-4 path) |
| Selector | [`hermes/scheduler/selector/`](../hermes/scheduler/selector/) — `target_selector_rl.py`, `features.py`, `ddqn.py`, `scope_guard.py` |
| Mule supervisor / two-pass mission | [`hermes/mule/mule_main.py`](../hermes/mule/mule_main.py) |
| Subprocess wiring (`use_rl_selector`) | [`hermes/processes/mule.py`](../hermes/processes/mule.py) |
| Arm definition | [`experiments/exp4/driver.py`](../experiments/exp4/driver.py) |
