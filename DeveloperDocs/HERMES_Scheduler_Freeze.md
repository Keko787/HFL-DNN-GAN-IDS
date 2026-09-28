# Layer-2 scheduling methodology — FROZEN

**Frozen:** 2026-08-13, at commit of this document.
**Means:** the L2 decision pipeline, its gates, and their guarantees are settled. **Any change to
the files listed in §5 after this point invalidates recorded sweeps** and must go through the
[pre-re-run checklist](HERMES_PreRerun_Checklist.md).

State at freeze: working tree clean for `hermes/scheduler/` and `hermes/mule/`; **153 scheduler
tests passing**.

**Amendments** (§5a–5g): 1–4 landed before or alongside the recorded sweeps. 5 and 6 (2026-09-27
and 09-28) fix defects and change what the budgeted, `--l1-channel` and D1/D2 cells measure. 7
(2026-09-28) opens this surface for the FeRRy build, behind switches whose defaults keep this
pipeline. The code behind every recorded result is the tag `exp4-recorded`.

---

## 1. The frozen pipeline

The path Experiment 4 actually executes (`rf_range_m` is always set, so the two-pass branch is
taken and `build_target_queue` is never called):

```
S1   eligibility        HARD GATE     — admits on mission-slice membership
S3   deadline + bucket  RANK TIER     — computes Deadline(j), classifies bucket
S3a  RF clustering      REGROUP       — devices within rf_range_m → ContactWaypoints
S3b  deadline feasibility HARD GATE   — drops contacts that cannot be served in time
S3.5 intra-bucket order ORDERING ONLY — deterministic distance, or the learned selector

S3c  mission window     SCALES S3     — mission-level widening; feeds back into S3, not a stage
                                        in the per-round path (Amendment 2, default off)
```

**The architectural guarantee — frozen.** Learning may only reorder within an already-admitted
bucket. Enforced by three independent mechanisms, all under test:

1. the candidate list is already post-S1/S3/S3a/S3b;
2. a **scope guard** re-checks every contact member against the round's admitted set;
3. a **pass-kind guard** hard-fails if the selector is invoked during Pass 2.

S3b is placed **before** S3.5 deliberately — a feasibility check after ordering could be
resurrected by the selector, and would also be skipped for single-candidate buckets, which
short-circuit around the selector.

## 2. Decisions taken at freeze

| # | Decision | Rationale |
|---|---|---|
| **D1** | **The S3b mechanism is frozen; its _default_ is a matrix parameter, not a code default.** `mission_budget_s=None` (no enforcement) remains the code default. | The default only matters when we re-run. Freezing the mechanism unblocks everything else; the enforcement/no-enforcement choice belongs to the Phase-3 matrix, where it is costed once. |
| **D2** | **Feasibility constants frozen at `cruise_speed_m_s=5.0`, `session_time_s=1.0`.** | These *are* the experiment — the probe shows the ~34 % deadline floor is driven by cruise speed and field radius, not by the budget. **They still need a platform citation before publication** (same class of problem as `ε_prop`). |
| **D3** | **S2A/S2B readiness gating is REMOVED from the contribution claims, not wired.** | `ingest_ready_adv` has no runtime callers; the design's `FL_Threshold = 0.60` and 5 s advert-freshness window never execute. The gate that does run (`min_utility = 0.0`, inline in `HFLHostMission`) cannot reject anything. Wiring it would cost code **and** a re-run for a gate that currently rejects nothing. Documented as designed-but-not-exercised; future work. |
| **D4** | **Experiment 4 makes no RL claim.** The selector stays random-init there; the RL question belongs to Experiment 3. | H2-vs-H1 differs only in within-bucket ordering, with untrained weights on partly-constant features. Training weights would force a re-run for a claim Exp 3 already owns. |
| **D5** | **`dead_zone` remains H0-only. This is correct, not a bug.** | Dead-zone models *the server's* loss of reach; the mule bypasses it by flying to the device — that is the architectural thesis under test. The earlier error was **sweeping it in an H2-vs-H3 comparison where it does nothing**, which is a matrix-design fix (§3), not a code fix. |
| **D6** | **Beacon ingest / `BEACON_ACTIVE` bucket remain unexercised.** | No beacon source is wired in Exp 4, so S1 admits on slice membership alone. Stated as a scope limit rather than claimed. |

## 3. Matrix consequences (carry into Phase 3)

* **Do not sweep `dead_zone` in any mule-only comparison** (H1/H2/H3 against each other) — it varies
  nothing for those arms. Use it only where H0 is present.
* **Valid pairwise comparisons:** `H1 vs H0` and `H3 vs H2`. `H2/H3 vs H0/H1` is **not** paired —
  different backhaul model and non-aligned seeds.
* **If enforcement is turned on**, every mule-arm participation figure changes (−0.225 mission
  completion at a slack budget), so it must be decided *before* the matrix is launched, not after.

## 4. What this freeze does **not** cover

Deliberately out of scope, so the freeze is not read as more than it is:

* **The deadline _design_** — the adaptation rule, its bounds, its sensitivity. Exp 4 models no
  flight budget or propulsion energy; that remains Experiment 3's.
* **Manuscript text.** Algorithm 1 and the deadline equation still print the update with the sign
  reversed. The code is correct; the paper is not. *(Prose fix, no re-run.)*
* **The learned selector's training.** Frozen as *not claimed* here (D4), not as *finished*.

## 5. Frozen surface — changing any of these invalidates recorded sweeps

```
hermes/scheduler/fl_scheduler.py
hermes/scheduler/stages/s1_eligibility.py
hermes/scheduler/stages/s3_deadline.py
hermes/scheduler/stages/s3a_cluster.py
hermes/scheduler/stages/s3b_feasibility.py
hermes/scheduler/stages/s3c_mission_window.py
hermes/scheduler/stages/s35_selector.py
hermes/scheduler/selector/          (target_selector_rl, features, ddqn, scope_guard)
hermes/mule/mule_main.py            (MuleSupervisor: queue construction, two-pass mission)
```

Not frozen (safe to change): analysis, figures, documentation, and the *values* of matrix
parameters (`mission_budget_s`, N, `n_missions`, seeds) — those are experiment design, chosen in
Phase 3.

**`hermes/scheduler/policies/` is deliberately NOT frozen.** It holds *alternative* ranking policies
(`ArrivalOrderPolicy`, `EdfFeasibilityPolicy`, and the MAX-AoI baseline `MaxAoIPolicy`), all exposing
the same `rank_contacts` surface and swapped through the same `target_selector` slot. Adding a
comparator there **cannot change any HERMES arm's behaviour** — an arm that does not select the
policy never constructs it — so baseline work does not invalidate recorded sweeps and does not need
an amendment. What *would* need one is a baseline requiring new **state**: Oort needs a per-device
training loss that the device→mule path does not carry, and adding that field touches the frozen
surface.

## 5a. Amendment 1 — in-flight abort + deadline feedback (2026-08-13)

**Unfrozen, amended, re-frozen the same day**, before any sweep was run against the original
freeze — so nothing recorded was invalidated. Taken now precisely *because* the Phase-3 re-run had
not happened yet: batching these in costs nothing, whereas adding them after the matrix runs would
have cost a second full re-run.

Two gaps, both found by inspection:

| # | Gap | Fix |
|---|---|---|
| **A1** | **S3b was pre-flight only.** The queue was filtered before take-off and never re-checked, so once the mule fell behind its plan it kept flying stops it could no longer serve — burning budget and delaying delivery of updates already aboard. | `MuleSupervisor._remaining_is_feasible()` re-runs the S3b check from the mule's **current** pose and clock before each stop; if the next contact is unreachable in time the Pass-1 loop **breaks**, and `close_round` + the dock deliver what was collected. |
| **A2** | **Unreached devices got no feedback.** `RoundCloseDelta` is emitted only from inside a contact session, so a device dropped by S3b or abandoned by an abort never widened its window — leaving it equally un-serveable next mission. **A starvation loop created by the S3b gate itself.** | `_widen_abandoned()` feeds a `TIMEOUT` delta for every device dropped pre-flight *or* abandoned in flight, widening Φ exactly as a missed contact does. |

**Scope note.** A1 can only foresee running out of **time** — a deterministic function of clock and
geometry. It cannot foresee a *random link failure*, which is stochastic by construction. The
proposal "abort when the drone knows it will fail to reach the next node" is therefore implemented
in its knowable form.

**Both are inert without enforcement** (`mission_budget_s=None`), pinned by test — so every
previously recorded sweep remains reproducible. 8 new tests; 573 unit tests pass.

**D1 is unchanged:** the mechanism is frozen, the default remains a Phase-3 matrix parameter.

## 5b. Amendment 2 — mission-level window adaptation, S3c (2026-08-13)

**Same-day amendment, again before any sweep ran against the freeze** — nothing recorded was
invalidated.

**The gap.** S3's adaptation is *per device*: a clean contact shrinks that device's window, a
missed one widens it. Amendment 1 (A2) made sure devices the gate skipped also get that signal —
but every one of those loops is still per-device. None of them can see **"the mule is
systematically failing to complete its circuit"**, because from any single device's point of view a
systemically over-tight schedule is indistinguishable from ordinary bad luck. A2 stops a starved
device from being starved *forever*; it does not diagnose a fleet-wide mismatch between the
schedule and the geometry, cruise speed and budget the mule actually has.

**The fix — S3c.** The mule reports `served/planned` after each mission. Over a rolling window the
adapter derives a multiplier applied to **every** device's fulfilment term in `compute_deadline`.

| Property | Choice | Why |
|---|---|---|
| **Derived, not accumulated** | The scale is a pure function of the recent record, recomputed each read | An integrator would wind up and drift; a pure function means the same history always yields the same scale, and reading it never perturbs it |
| **Widen-only** | At or above `target_success` the scale is exactly 1.0 | Shrinking is the per-device rule's job — it knows *whom* to reward; S3c only knows that the fleet is behind |
| **Bounded** | `max_scale` (default 4.0) | An impossible configuration degrades to "windows are wide", never "windows are unbounded" |
| **Pooled, not averaged** | `Σserved / Σplanned`, not the mean of per-mission ratios | A 100-device mission must outweigh a 1-device one |
| **Denominator includes S3b's own drops** | `planned` = queue **+** pre-flight drops | Otherwise the gate flatters itself: drop nine contacts, serve the tenth, report 100 % success, never widen — the starvation loop hiding inside its own success metric |
| **Cluster override still wins** | `deadline_override_ts` short-circuits before scaling | The slow-phase amendment stays authoritative (§6.8) |

**Scope note.** S3c is *not* a gate and not a stage in the per-round decision path — it changes no
admission and no ordering. It only scales the S3 term that S3b later tests against, which is why it
sits outside the S1→S3.5 pipeline in §1.

**Inert by default,** pinned by test: with the toggle off the scale is exactly 1.0 and
`compute_deadline` reduces to the original formula term for term. 44 tests now cover Amendments 1
and 2 (8 abort/starvation + 36 S3c); **609 unit tests pass**.

**Matrix parameter, not a code default** — same rule as D1. The toggle
(`--mission-window-adaptation`) and its four tunables belong to the Phase-3 matrix. Expect it to
matter only where the S3b gate binds: with no budget there is nothing for a wider window to rescue.

## 5c. Amendment 3 — Oort baseline inputs on the device→mule path (2026-08-13)

**Reason.** Arm **B2** implements *Oort's statistical-utility selection* as a SOTA comparator
(Phase-2 decision, checklist §5.1a-B). Oort ranks on `|B_i|·√(mean Loss²)`, and **no per-device
training loss existed on the device→mule path**. `RoundCloseDelta.utility` is `w1·perf + w2·diversity`
— an S2B readiness term in which `perf` has already collapsed accuracy, AUC and loss into one score.
Reusing it would be a different algorithm wearing Oort's name.

**Frozen surface touched: `stages/s3_deadline.py` only, in `fold_round_close_delta`** — three
assignments beside the existing `state.last_utility` line:

```python
if delta.local_loss is not None:      state.last_loss = delta.local_loss
if delta.num_examples:                state.last_num_examples = delta.num_examples
state.last_served_round = delta.mission_round
```

**Inert for H0–H3, pinned by test.** The new fields default to `None`/`0` on both `FLReadyAdv` and
`RoundCloseDelta`, so every pre-B2 emitter folds nothing and no arm's behaviour changes. A test also
pins that a later delta *without* the fields does not wipe an earlier measurement.

Everything else is outside the frozen surface: the two new optional fields on each message type
(`types/`), retaining the raw values the device already computed and discarded
(`mission/client_mission.py`), forwarding them through the single `_record_outcome` funnel
(`mission/host_mission.py`), the policy itself (`scheduler/policies/oort.py`), and the arm wiring.

**Three fidelity deviations — stated in the module docstring, and the paper must carry them:**

| # | Deviation | Why |
|---|---|---|
| 1 | **No system-speed term** | Oort multiplies statistical utility by a straggler penalty over client compute/comm speed. We model no per-device speed, so the term is **dropped** rather than approximated by something else |
| 2 | **Mean loss, not RMS** | Oort specifies `√(Σ Loss(k)²/\|B_i\|)` over per-sample losses; our callback reports Keras' mean loss. Monotone in the same direction, not identical |
| 3 | **Rounds, not wall-clock, for staleness** | `L(i)` is the last mission round in which the device was served |

⇒ **The arm is "Oort's statistical-utility selection", not "Oort".**

**It requires `--real-model`, and refuses without it.** The stub reports `loss=uniform(0.1,0.3)` and
`num_examples=randint(4,16)` — pure noise — so ranking on it would be a random ordering wearing
Oort's name. The driver raises for `B2` without `--real-model`, and the policy itself raises
`OortUnusableError` if devices have been served but no loss signal arrived. Failing loudly beats
emitting a meaningless order that looks like a result.

25 new tests; **676 unit tests pass**. Verified end to end in-process — a real `LocalTrainResult`
loss reaches `statistical_utility` intact through advertisement, delta and fold.

## 5d. Amendment 4 — whole-scheduler baselines D1/D2 own admission (2026-08-17)

Landed in commit `2e9fdc9` and referenced from the code (`fl_scheduler.py`, the block headed
"Freeze Amendment 4") and `tests/unit/test_whole_scheduler_baselines.py`; this section was written
up on 2026-09-27.

**Reason.** The ordering-only arms B1/B2 were vacuous. S3b fixes *who* is served before any ranking
policy runs, so a baseline confined to the selector slot could only permute a list the gate had
already decided, and every arm produced byte-identical results.

**Frozen surface touched: `fl_scheduler.py`, in `build_contact_queue`.** A policy that exposes
`admit_and_order` owns both decisions: it replaces S3's deadline ordering, S3b's admission gate and
S3.5's tie-break, and returns the route directly. S1 and S3a still run for every arm, and the budget
and our `FeasibilityModel` are passed through unchanged so every arm prices travel the same way.

Outside the frozen surface: `policies/budget_walk.py` (`greedy_budget_walk` walks contacts in policy
order and skips, rather than stops at, one that does not fit), the `admit_and_order` methods of
`policies/max_aoi.py` (D1) and `policies/oort.py` (D2), and the arm wiring in
`experiments/exp4/driver.py`. **Inert for H0–H3**, which expose no `admit_and_order`.

## 5e. Amendment 5 — backhaul schedule index and baseline ages (2026-09-27)

Two defects found while checking the recorded results for the FeRRy build plan (Phase 0).

**1. The backhaul loss schedule was only ever read at mission 1.** `processes/cluster.py` passed
`getattr(up, "mission_round", None)` to `_backhaul_dropped()`, but `UpBundle` has no
`mission_round`; it lives on `up.partial_aggregate`. Every UP therefore drew against `schedule[0]`,
and the `backhaul_upload_lost` event carried no round, so `backhaul_lost_rounds` in
`experiments/exp4/events_consumer.py` was always empty. Fixed with `_up_mission_round(up)`, used for
the draw and for both events. *Not a frozen file.*

**2. Abandoning a device reset its age for D1 and D2.** `_widen_abandoned()` feeds a synthetic
TIMEOUT stamped `now`, and the fold writes every outcome into `last_contact_ts` and
`last_served_round` — the fields MAX-AoI (D1) aged a device from and Oort's staleness term (D2) read
as `L(i)`. Both now read two new fields that only a CLEAN sets.

**Frozen surface touched: `stages/s3_deadline.py` only, in `fold_round_close_delta`** — two
assignments in the CLEAN branch:

```python
state.last_clean_ts = delta.contact_ts
state.last_clean_round = delta.mission_round
```

Outside the frozen surface: the two fields on `DeviceSchedulerState` (`types/scheduler.py`, default
`0`), `contact_age` in `policies/max_aoi.py`, `staleness_bonus` in `policies/oort.py` (which still
derives its current round from `last_served_round`), and `processes/cluster.py` for defect 1.
**Inert for H0–H3**, which read neither new field.

**Recorded results affected:**

| Sweep | Effect | Action |
|---|---|---|
| Every `--l1-channel` cell, including the L1 confirmation cells C1 (H3 vs H2, n = 40, jittery, 120 s) and C2 in `HERMES_Matrix_Results.md` | Each arm's mission-1 loss probability was applied to every mission, so H3 vs H2 compared the two arms' mission-1 bands held for the whole trial, not per-mission adaptation | Re-run before citing |
| `round_close_rate_kmin*` in every run with backhaul loss (`--realism` jittery, or `--l1-channel`) | Backhaul-dropped rounds were counted as closed. Model metrics (AUC, accuracy, `t_at_tau_round`) come from `model_eval` events and are unaffected by this part | Re-score from traces where kept: `experiments/analysis/traces_scorer.py` places each `backhaul_upload_lost` event in the mission window that contains it. On the L1 cell (`C_traces`) closure at k = 1 falls from 0.831 to 0.675 for H2 and from 0.838 to 0.813 for H3; accuracy and yield re-score identically |
| D1/D2 cells (SOTA pilot, budget axis, `b60`) | Only abandoned devices are affected. D1/D2 never trigger the pre-flight widen (their admission path does not set `last_feasibility`), so only in-flight aborts feed them synthetic TIMEOUTs, and those are rare while the abort check compares wall-clock time with simulated transit. D2's staleness term is about 1e-4 of its utility in Exp 4 (`n·\|loss\|` ≈ 300–2,300 against a bonus ≤ 0.16), so D2's ranking is essentially unchanged. The bonus is that small because, unlike Oort's reference code, the utility is not normalised before the bonus is added, and the bonus is `0.1·log R/√L` rather than Oort's `√(0.1·log R/L)` — an undocumented fidelity deviation, not changed here | Re-run the 60 s cell anyway before citing it |

**Found in the same check, not fixed here** (fixed by Amendment 6). The mission budget is stamped
in `ingest_slice`, which runs only on a DOWN bundle. An empty mission skips the dock, so the next
mission plans against the old stamp and its budget shrinks. In the `b60` traces 72–81% of missions
were empty (H1 116/160, D1 126/160, D2 129/160); in one D1 trial the budget left at planning fell
from 60 s to about 39 s by mission 4.

13 new tests (`test_backhaul_schedule.py`, `test_baseline_age_after_abandonment.py`); three test
helpers now set the last-CLEAN fields. **717 unit tests pass**; the 4 `test_mode_switch` failures
are pre-existing (their subprocesses cannot import `Config`) and fail identically without this
amendment.

## 5f. Amendment 6 — every mission's budget runs from its own start (2026-09-28)

**Reason.** The S3b budget clock started only in `ingest_slice`, which runs when a DOWN bundle
arrives. A DOWN arrives mid-mission, at the inter-pass dock, so each mission's budget also carried
the previous mission's Pass 2. After an empty mission there is no dock and no DOWN at all, so the
next mission planned, and ran its in-flight check, against the old stamp. Amendment 5 records the
evidence from the `b60` traces.

**Frozen surface touched:**

- `fl_scheduler.py` — a new `start_mission()` that stamps `_mission_start_ts` from the clock and
  returns it. `ingest_slice` keeps its stamp as a fallback for callers that drive the scheduler
  without a mule.
- `mule_main.py` — `run_one_mission()` calls `self.scheduler.start_mission()` before dispatching to
  either the two-pass or the single-pass path, so every mission, empty ones included, plans and
  checks in flight against its own full budget.

**Not inert.** Every run that sets `--mission-budget-s` now behaves differently: no mission loses
budget to an earlier one. Runs without a budget are unchanged, because the stamp is read only when
a budget is set.

**Recorded sweeps affected.** Every cell run with a budget: `run_matrix.sh` and `run_matrix_a2.sh`
(the 08-13 matrix, 120 s), `run_l1_confirm.sh` (120 s), `run_s3b_budget_sweep.sh` (120, 60, 30
and 15 s; its no-budget control is unaffected), `run_s3c_pilot.sh` (120 s), and both SOTA scripts
(`run_sota_budget_axis.sh` at 120 and 60 s, `run_sota_b60_extend.sh` at 60 s). The loss grows with
the share of empty missions and is largest at tight budgets. Re-run a cell before citing it.

6 new tests (`test_mission_budget_clock.py`); each of the three behavioural ones fails when
`start_mission` is a no-op. The full suite gives 849 passed and the same 9 pre-existing failures
as under Amendment 5.

## 5g. Amendment 7 — the FeRRy build opens the frozen surface (2026-09-28)

**Reason.** FeRRy, the Path A system of the contribution ledger (C1–C5), changes what the
scheduler decides: a band class and route chosen together at the dock, a (band, next stop) pair
chosen at every arrival, a re-plan where the pipeline now aborts, an age cap, a coverage term, a
different deadline law and an age-weighted merge. Phases 1 to 5 of the build plan
(`FeRRy_Build_Plan.html`) have to edit files this freeze protects. This amendment opens them, on
three rules that keep the frozen pipeline runnable and every recorded arm meaningful. It changes
no code itself, so it invalidates no sweep.

**Rule 1 — the frozen pipeline becomes a mode, and stays the default.** Everything FeRRy adds
lands behind a switch whose default is the frozen pipeline ("legacy mode"). H0–H3, D1 and D2 run
in legacy mode unless a study sets otherwise. Each switch is recorded here when its phase lands:

| Switch | Legacy value | FeRRy value | Lands in |
|---|---|---|---|
| L3 merge rule (`ClusterConfig`/`MuleConfig.aggregation`) | `agg:plain`, the num_examples mean | `agg:cutoff`, age-weighted with the deadline as cutoff | Phase 1, commit `8f23f02` |
| FedProx term on devices (`DeviceConfig.fedprox_rho`) | 0 | swept | Phase 1, commit `8f23f02` |
| Deadline law (`MuleConfig.deadline_law`) | additive: −5 s on time, +10 s on a miss, floor 5 s, no ceiling; cluster overrides sticky | multiplicative and clamped, PARTIAL relaxing less than TIMEOUT, overrides one-shot | Phase 1, commit `8f23f02` |
| Priority key (`MuleConfig.miss_priority`) | off: S3b admits in deadline order | S3b admits by miss streak, then deadline | Phase 1, commit `8f23f02` |
| Pass-2 budget (`MuleConfig.pass_2_budget`) | off: Pass 2 delivers to the whole slice | on | Phase 1, commit `8f23f02` |
| Mission clock (`now_fn`) | wall clock | simulated seconds (`l1/mission_clock.py`) | Phase 3, planned |
| Contact band | none: one `rf_range_m` for every stop | band classes with a range and a rate | Phase 3, planned |
| Response when the remaining queue stops fitting | abort the rest (Amendment 1, A1) | re-plan with 2-OPT under S3b | Phase 3, planned |
| `plan_mode` | `legacy` | `ferry` | Phase 4, planned |
| `band_class_policy` | — | `search`; `fixed:<class>` is Path B+ | Phase 4, planned |
| Age cap `S` | off | set by build-plan decision D4 | Phase 4, planned |
| Flight-clock choice | distance order, or `TargetSelectorRL` within a bucket | masked pair score, or the cross-heuristic | Phase 5, planned |

**Rule 2 — legacy mode is the corrected pipeline, not the recorded code.** Amendments 5 and 6
changed legacy behaviour on purpose, because it was wrong, so legacy mode no longer reproduces the
recorded budgeted, `--l1-channel` and D1/D2 cells (the tables under 5e and 5f list them). The code
behind every recorded result is tagged **`exp4-recorded`** (commit `229a093`; nothing under
`hermes/`, `experiments/` or `tests/` changed after `d0e80d4`, the b60 seed-extension runner).
Re-derive a recorded number from that tag, never from `main`.

**Rule 3 — anything that changes legacy behaviour is an amendment of its own**, recorded as 5 and
6 were: reason, files, invalidated sweeps. A change confined to ferry mode needs only its row in
the table above, updated from "planned" to the commit that lands it.

**Files opened, and the phase that first needs each:**

```
hermes/scheduler/fl_scheduler.py        Phase 3 (public accessors for the budget, the mission start
                                          and the feasibility model), Phase 4 (plan mode, clustering
                                          per band class, commit, a visited set per mission)
hermes/scheduler/stages/s3_deadline.py  Phase 1 (law, priority key, PARTIAL vs TIMEOUT, overrides
                                          that expire)
hermes/scheduler/stages/s3a_cluster.py  Phase 4 (radius from the band class, once per class)
hermes/scheduler/stages/s3b_feasibility.py  Phase 1 (priority key in the walk), Phase 3 (one
                                          FeasibilityModel; single-contact predicate)
hermes/scheduler/selector/              Phase 5 (pair features, masked pointer Q, replay with the
                                          next candidates, the ferry_sim trainer, the scope guard
                                          over pairs)
hermes/mule/mule_main.py                Phase 1 (version threading, Pass-2 budget), Phase 3 (clock,
                                          band, re-plan), Phase 5 (pair choice)
```

**Landed so far (Phase 1, commit `8f23f02`; Amendments 5 and 6 are commit `dc90f84`).** In the frozen files: `fl_scheduler.py`
gained read-only properties (`mission_budget_s`, `mission_start_ts`, `feasibility_model`,
`deadline_law`, `miss_priority`) and passes the law to every deadline call and the priority key to
S3b; `s3_deadline.py` gained `DeadlineLaw`, `effective_window` and an optional `law` argument on
`compute_deadline` and both folds (None runs the recorded arithmetic verbatim), and the fold keeps
a `miss_streak`; `s3b_feasibility.py` takes an optional `priority` key that outranks the deadline
in its walk; `mule_main.py` keeps the DOWN bundle's θ version and pushes it, passes each device's
cutoff to the merge, walks Pass 2 against the budget when `pass_2_budget` is on, and uses the new
accessors.
Outside them: the version and delta fields on the wire types, the merge rules
(`mission/aggregation_rules.py`, the delta merge in `partial_fedavg.py`, the fold in
`cross_mule_fedavg.py`, the rule in `host_cluster.py`), the device's basis bookkeeping, FedProx in
`exp4/model_task.py`, and the Exp 4 flags. Legacy behaviour is unchanged: `agg:plain` runs the
original merge functions (byte-identical output, pinned by test), devices still answer in full
weights, and the additive law folds every state field exactly as before (pinned over 300 random
outcomes). The full suite gives 1018 passed; its 5 failures predate Phase 1 (the selector DoD
cell and the four `test_mode_switch` subprocess tests). Legacy traces gain additive fields only
(each update's basis version and age; each device's window and miss streak).

`s1_eligibility.py`, `s3c_mission_window.py` and `s35_selector.py` stay as they are; FeRRy does
not plan to touch them. The clock needs no scheduler change: `FLScheduler` already takes
`now_fn`. New modules (`scheduler/plan/`, `scheduler/routing/`, `stages/s3d_age_cap.py`,
`l1/contact_link.py`, `l1/channel_model.py`, `l1/mission_clock.py`) sit outside the old frozen
surface but follow the same three rules.

**Design principles restated** (numbering from `HERMES_FL_Scheduler_Design.md` §7):

- **1 · Each layer has one job.** Kept, with the coupling made explicit. *Who* and *how* now meet
  in one decision on each clock: the plan clock picks band class and route together, the flight
  clock picks band and next stop together. They meet through declared interfaces only — the S3b
  predicate (the hard coupling) and one score over (band, stop) pairs (the soft coupling). New
  code reads no other layer's private state. The one existing breach (`mule_main.py` read the
  scheduler's `_mission_budget_s`, `_mission_start_ts` and `_feasibility_model`) is closed:
  Phase 1 added read-only accessors for all three, and the mule uses them.
- **5 · Only the waypoint crosses L2→L1.** Replaced. At the dock the mule commits a band class, an
  ordered queue and a budget; at each arrival the chosen pair's band is the band the contact is
  served on, and per-band SNR at estimated arrival is a feature of each pair. The contact band is
  a separate decision from the backhaul band: the U(c, t) controller still chooses the
  mule-to-base-station band once per mission and stays channel-only.
- **12 · The learned selector is bounded to intra-bucket ordering.** Spirit kept, letter replaced.
  The learned pair score chooses among the (band, stop) pairs that S1, the S3b predicate and the
  age cap admit, at every arrival. It cannot admit a pair the gates reject, cannot drop a capped
  device (only the deterministic re-plan drops, and only under the predicate), and cannot change a
  deadline. Buckets remain provenance tags. The frozen guarantee of §1 becomes: *learning may only
  choose among admitted pairs*, enforced by the same three mechanisms — candidates are post-gate,
  the scope guard re-checks every pair's devices against the admitted set, and the pass-kind guard
  still refuses Pass 2.
- **13 · Missions are two-pass.** Kept, with one of the concerns it closed reopened on purpose.
  Once Pass 2 runs inside the budget, a device the mule does not reach trains on an older θ. The
  cluster still knows exactly which θ each update was trained on, because the basis version
  travels with the update (Phase 1), and the merge weights the update by that age (C3).
- **14 · Local training is offline; sessions are exchange-only.** Kept. In ferry mode a session's
  dwell is charged to the mission clock at its band's rate, bytes / rate, instead of a fixed 1 s.
- **15 · Contact events, not per-device visits.** Kept. In ferry mode the clustering radius is the
  committed class's range R(b̄) instead of one `rf_range_m`.

Principles 2–4 and 6–11 are unchanged. The new deadline law and merge rule change how principles 4
(two-phase deadline adaptation) and 7 (deadline-aware aggregation) are carried out, not what they
say.

**The decisions of §2, in ferry mode:**

| # | At freeze | In ferry mode |
|---|---|---|
| D1 | S3b mechanism frozen; the budget is a matrix parameter | One FeasibilityModel — transit + bytes/rate + return + upload, energy clause declared simulated. The budget stays a study parameter (60 s knee, 30 s stress). |
| D2 | 5 m/s cruise, 1 s session | Session time from bytes / rate(band). Cruise speed stays 5 m/s and still needs a platform citation. |
| D3 | S2A/S2B out of the claims | Unchanged. |
| D4 | Exp 4 makes no RL claim | Unchanged for Exp 4. Exp 5 claims learning only through tests (b) and (c) and the learned-vs-cross-heuristic comparison, with trained checkpoints committed beside their seeds. No random-init arm. |
| D5 | `dead_zone` is H0-only | Unchanged. |
| D6 | Beacons unexercised | Beacon inserts go through the S3b predicate; still unexercised until an Exp 5 topology emits beacons. |

**Records corrected.** Documents that describe Layer 1, or the selector's training, as something
the code never ran:

- The Layer-1 policy the results evaluate is the deterministic utility controller
  U(c, t) = R(γ₁(t) + g(c)) − κ(c) − λ(c, t), R ≈ log2(1 + SNR), of `hermes/l1/channel_utility.py`,
  applied to the mule-to-base-station backhaul once per mission (Exp 4 arm H3, `--l1-channel`).
  `ChannelDDQN` (8→16→3) has no trainer in this repository, and the process runtime passes no
  channel actor to the mule; where a test passes one, its choice is logged, not actuated. The
  contact link has no band. Notes that presented L1 as a working DDQN are corrected in place:
  `README.md`, `architecture documents/Hermes/HERMES_Architecture.md` and its HTML,
  `Comparative Analysis/CEDA_vs_HERMES.md` and its HTML, and the Sprint 2 entry-point item of
  `HERMES_FL_Scheduler_Implementation_Plan.md`.
- `TargetSelectorRL` was trained, where it was trained at all, in the single-agent simulators of
  `selector/sim_env.py` (BucketSim, then ContactSim) — not under CTDE on an AERPAW twin — and
  every committed Exp 4 H2/H3 row used random-init weights (D4). The two selector docstrings that
  said otherwise (`replay.py`, `selector_train.py`) are corrected; no code changed.
- The April presentation (`Presentation Documents/HERMES Improved Presentation 4-7-26.pptx`:
  MA-P-DQN with a joint (Δposition, channel) action on slides 20–22, repeated on 41, 46, 51–53
  and 67–70) and SEC'26 Fig. 2 ("Training: CTDE on AERPAW digital twin"; divergence D-1 in
  `architecture documents/System_Architecture_Overview.md`) show designs the code never ran.
  `HERMES_FL_Scheduler_Design.md` §2.6 already retired MA-P-DQN; a status note there, and one on
  its §8 mapping, now point here.
- SEC'26 Table VI matches, by numbers, scenario and metrics, the hybrid run of the `hermes_rl`
  prototype at commit `a8a453f`, now at `experiments/sim/drone_env/`. That environment ignores
  seeds by default, so the reported model was selected and evaluated on the same fixed episode
  (see the README there).

**Invalidated sweeps: none by this amendment.** Exp 5 runs every arm, H0–H3, D1 and D2 included,
on the simulated clock and the seconds-axis channel once Phase 3 lands, so each is re-baselined
there — the re-run bill the build plan accepts in its decision D2. Legacy defaults keep the Exp 4
harness runnable as it is. The pre-re-run checklist is re-opened (§1a there).

## 6. Unfreezing

Amend this document with the reason, the changed files, and which recorded sweeps are invalidated.
Then re-open the [pre-re-run checklist](HERMES_PreRerun_Checklist.md).
