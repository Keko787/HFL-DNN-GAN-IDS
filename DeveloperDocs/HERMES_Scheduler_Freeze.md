# Layer-2 scheduling methodology — FROZEN

**Frozen:** 2026-08-13, at commit of this document.
**Means:** the L2 decision pipeline, its gates, and their guarantees are settled. **Any change to
the files listed in §5 after this point invalidates recorded sweeps** and must go through the
[pre-re-run checklist](HERMES_PreRerun_Checklist.md).

State at freeze: working tree clean for `hermes/scheduler/` and `hermes/mule/`; **153 scheduler
tests passing**.

**Amendments** (§5a–5j), **Phase 4** (§5k) **and Phase 5** (§5l): 1–4 landed before or alongside the recorded sweeps. 5 and 6 (2026-09-27
and 09-28) fix defects and change what the budgeted, `--l1-channel` and D1/D2 cells measure. 7
(2026-09-28) opens this surface for the FeRRy build, behind switches whose defaults keep this
pipeline. 8 and 9 (2026-09-28) land with the Phase 0/1 audit and Phase 2, and 10 (2026-09-29, the
RF transport fix) with Phase 3. Phase 4 (2026-09-30, the plan clock) needed none: every mechanism
lands behind a switch or is additive (§5k). Phase 5 (2026-10-02, the flight clock's (band, stop)
score, FerrySim and E3) needed none either, on the same terms (§5l). The code behind every recorded
result is the tag `exp4-recorded`.

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

**2. Any non-CLEAN outcome reset a device's age for D1 and D2.** The fold writes every outcome into
`last_contact_ts` and `last_served_round`, the fields MAX-AoI (D1) aged a device from and Oort's
staleness term (D2) read as `L(i)`. So a real in-session TIMEOUT or PARTIAL (a lost uplink, a
refused advert, a failed push, no advert at all) reset the age as surely as a delivered update did,
and so did the synthetic TIMEOUT `_widen_abandoned()` feeds a device the mule dropped or abandoned.
Recorded D1 therefore aged a device from its last *attempt*, not its last delivered update. Both
now read two new fields that only a CLEAN sets. *(Corrected 2026-09-28: this paragraph first
described the synthetic TIMEOUT as the only source; see the impact table.)*

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
| D1/D2 cells (SOTA pilot, budget axis, `b60`) | Every non-CLEAN Pass-1 outcome stopped resetting age, real failed sessions included, not only abandoned devices. Under `--realism` jittery those are most outcomes: in the `b60` D1 traces 126 of 160 missions collected nothing and only 34 of 608 scheduled Pass-1 devices came back CLEAN. Under the recorded code almost every outcome reset age; under the fix a device never delivered stays maximally stale and D1 routes it first. So D1's route order and, under a budget, its admission change in most missions of every `--realism` cell, with or without a budget. D2 barely moves: its staleness term is about 1e-4 of its utility in Exp 4 (`n·\|loss\|` ≈ 300–2,300 against a bonus ≤ 0.16). The bonus is that small because, unlike Oort's reference code, the utility is not normalised before the bonus is added, and the bonus is `0.1·log R/√L` rather than Oort's `√(0.1·log R/L)` — an undocumented fidelity deviation, not changed here. *(Corrected 2026-09-28: this row first said only abandoned devices were affected, and rarely.)* | Re-run every D1/D2 cell before citing it, whatever its budget. The budgeted ones are also invalidated by Amendments 6 and 8 |

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
| Deadline law (`MuleConfig.deadline_law`) | additive: −5 s on time, +10 s on a miss, floor 5 s, no ceiling; cluster overrides sticky | multiplicative and clamped, a miss by a device that answered relaxing less than one by a device that did not, overrides one-shot | Phase 1, commit `8f23f02`; the reachability split and clamp-before-step in the Phase 1 audit, commit `c417554` (§5h) |
| Priority key (`MuleConfig.miss_priority`) | off: S3b admits in deadline order | S3b admits by miss streak, then deadline | Phase 1, commit `8f23f02` |
| Pass-2 budget (`MuleConfig.pass_2_budget`) | off: Pass 2 delivers to the whole slice | on; Pass-1 devices then train ahead on the basis they adopt | Phase 1, commit `8f23f02`; train-ahead in the Phase 1 audit, commit `c417554` (§5h) |
| Mule count (`n_mules`) | 1 | K ≥ 2 over spatial slices; D4 assigns by CARP | Phase 2, commit `c417554` (§5i) |
| Cluster quorum (`min_participation`) | 1 | 1 or K (`agg:plain` needs K; in between refused except FedBuff) | Phase 2, commit `c417554` (§5i) |
| Dock after an empty mission, bounded DOWN wait (`dock_on_empty`, `down_wait_s`) | off; one 10 s wait whose expiry ends the loop | on at K > 1, the wait at the trial budget, survived | Phase 2, commit `c417554` (§5i) |
| Whole-scheduler arms (`contact_policy`) | our pipeline; D1 `max_aoi`, D2 `oort` | D3 `whittle`, D4 `fedex`, D5 `fedcs` | Phase 2, commit `c417554` (§5i) |
| L3 merge rule `agg:fedex` | — | FedEx-Async's θ + η·Σ Δθ / N per return | Phase 2, commit `c417554` (§5i) |
| Mission clock (`MuleConfig`/`ClusterConfig.mission_clock`, `--mission-clock`) | `wall`: `time.time` stamps every mission-time read (plans, deadlines, the budget, contact outcomes); the pose carries over between missions | `sim`: one `MissionClock` per mule process (`l1/mission_clock.py`; epoch 1e6 s, refused at 1e9 s so simulated and wall stamps never mix), charged by transit, dwell, listen, return, upload, turnaround and dock wait and never paced; every mission-time stamp from it; the pose reset to the dock at each takeoff; the budget stamped at takeoff, Pass 2's at its own takeoff; the cluster echoes the latest simulated upload it ingested (`cluster_sim_ts`), and the mule syncs to it at the dock. H0 is refused on it (critic A5) | Phase 3, commit `ef1faa1` (§5j) |
| Contact band (`MuleConfig.contact_band`, `contact_band_classes`, `--contact-band`) | none: one `rf_range_m` for every stop; a broadcast solicit (on the wall clock); 1 s per contact in the cost model | `wide`, `medium` or `narrow` (decision D1; the re-baselines fly `wide`): S3a radius R_planar(b); numbered solicits to the stop's members only, gated at arrival by range and the SNR floor; dwell = 8·bytes / rate(b, SNR) charged to the clock and priced by S3b; band and SNR on the report lines. On the mission clock without a band (the channel-free control) solicits are targeted too, and a contact costs 1 s plus the listen window when a reply is missing | Phase 3, commit `ef1faa1` (§5j) |
| Response when the remaining queue stops fitting (`MuleConfig.in_flight_response`, `replan_fallback`) | `abort` the rest (Amendment 1, A1; the per-policy rule of Amendment 8) | `replan`: the whole remainder checked at every departure and repaired by `FLScheduler.replan_remainder`. Our arms keep their own order over what S3b re-admits when it fits; otherwise `replan_fallback` decides: `reorder` (2-OPT, then S3b's admission order; whenever the pre-flight check fires, H1, H2 and H3 fly the same route) or `trim` (each arm keeps its order and drops the stops that order cannot serve, so it serves fewer). D1–D3 and D5 re-admit through their own `admit_and_order`, D4 flies on and records its overrun, Pass 2 is a nearest-first budget walk. Drops are final for the mission and widened at the simulated drop time. The pilot plan, decided 2026-09-29, flies `replan` with `trim`, so each arm keeps its own order, as the D arms do (critics B5, C3; Run Guide §2.6) | Phase 3, commit `ef1faa1` (§5j) |
| Pre-flight order check (`FLScheduler(validate_flown_order=...)`) | off | on under `replan`, set by the mule: the order the arm will fly is folded before takeoff and repaired as above; needs the ferry model and a budget (no budget, no gate) | Phase 3, commit `ef1faa1` (§5j) |
| Backhaul model (`MuleConfig`/`ClusterConfig.backhaul_model`, `backhaul_policy`, `backhaul_regime`, `--backhaul-model`) | `mission`: the cluster's recorded loss (the flat `--realism` percentage, or the `--l1-channel` schedule by mission round, Amendment 5), drawn from a stream; on the mission clock the upload is still charged, timed at the fixed carrier's noise-free mean SNR | `seconds` (mission clock only): three carriers' SNR at the simulated upload start, period P_bh = `n_missions` × T_nom; the fixed carrier argmax g_c for every arm but H3, whose U(c, t) controller picks at every upload; p_loss = `loss_from_snr` (1.0 below the SNR floor, charged the floor-rate time); the loss drawn keyed by (trial seed, mule, mission round), so every arm faces the same uniform for a mission (common random numbers), though each arm's p_loss is its own, read at its own upload start and on its own carrier; the flat percentage set to 0. About 16 % of uploads lost when jittery at the fixed carrier, against today's flat 2 % (critic A4). Refused with `--l1-channel`; chosen per study | Phase 3, commit `ef1faa1` (§5j) |
| Contact reliability (`MuleConfig.contact_reliability_source`, `device_availability`) | `origin`: each device draws rel × rf_factor (distance to the origin) from its own stream | `channel` (needs a band): the SNR gate at the stop, and the availability rel_i drawn on the mule keyed by (trial seed, device, mission round); the devices are built with `contact_reliability=None`. The ground-truth map reaches no scheduler, policy, L1 state or event, only its size does; it lives in the mule's configuration alone (critic B16). Pass 2 faces the SNR gate only | Phase 3, commit `ef1faa1` (§5j) |
| Deadline time unit (`MuleConfig.deadline_time_scale`, `initial_window_s`) | 1.0 and None (Φ₀ = 60 s) | every time constant of the law (the additive −5 s / +10 s steps and 5 s floor, the multiplicative clamps, Φ₀) times the scale. Φ₀ is stated in the law's recorded unit and scaled, so None and 60 are the same Φ₀ at any scale. A numeric scale and Φ₀ in seconds are valid on either clock; the driver's `t_nom` (T_nom / 10 s) and `--initial-window-missions` (Φ₀ in missions, critic A7) need the mission clock. The pilot plan, decided 2026-09-29, sets `t_nom`, so with the default Φ₀ each device starts with about six missions' worth of window, as in the recorded runs | Phase 3, commit `ef1faa1` (§5j) |
| Payload (`MuleConfig.payload_bytes`) | None: on the wall clock nothing is priced by bytes; on the mission clock the measured bytes (θ 18,756 B at 21 inputs; a Pass-1 session 37,576 B) | bytes per direction, declared (decision D3: 1 MB and 10 MB); they price the dwell and the upload while the real θ still crosses the link | Phase 3, commit `ef1faa1` (§5j) |
| What Deadline(j) bounds (`MuleConfig.deadline_bounds`) | the arrival: clock + transit ≤ Deadline(j) | `collection` (spec Q2): arrival + dwell ≤ Deadline(j); `delivery_per_stop`: per stop, the finish plus that stop's own return plus the upload ≤ its Deadline(j) (the plan's single-contact predicate; it bounds the actual delivery, the route's landing plus the upload, only for the last stop; `delivery` at `ef1faa1`); or `delivery`: route-level, each admitted stop's home ≤ min(its own Deadline(j), `deliver_by`), so the priced landing plus the upload meets every collected deadline (drop reason `delivery`; an overrun at the stop where Pass 1 ends is recorded as `delivery_overrun_s`). Only our arms' deadline rule reads either delivery value | Phase 3, commit `ef1faa1`; the three values in commit `d175afa` (§5j, clock F1) |
| Deadline overrides (`FLScheduler(refuse_deadline_overrides=...)`) | folded as sent (the cluster sends none) | refused on the mission clock, since they are wall-clock stamps (critic B3): the cluster will not issue one, and a mule that receives one fails, with `dock_bootstrap_failed` (exit 5) at the bootstrap or `mission_failed` (exit 3) at a later dock | Phase 3, commit `ef1faa1` (§5j) |
| RF link token (`MuleConfig`/`DeviceConfig.rf_link_token`, `--rf-link-token`) | None: any registration accepted | one token per trial, derived from cell, arm, trial and seed, on by default exactly on the mission clock: a mule refuses a registration that carries another (Amendment 10) | Phase 3, commit `ef1faa1` (§5j) |
| Newest solicit only (`DeviceConfig.newest_solicit_only`) | off: solicits answered in arrival order | on in every mission-clock cell: a device answers only its newest queued solicit, since the mule accepts only adverts that name the solicit it is gathering for (critic B1) | Phase 3, commit `ef1faa1` (§5j) |
| SpectrumSig forwarding (`SpectrumSig.contact_class_snr_db`) | none: no DOWN carries a `spectrum_sig` delta | on the mission clock the cluster reads each Pass-1 line's (band, SNR), keeps the latest per class and sends it in `registry_deltas[did]["spectrum_sig"]`; the mule folds it into `DeviceSchedulerState.spectrum_snr_db`. Nothing decides on it in Phase 3 | Phase 3, commit `ef1faa1` (§5j) |
| Causal RF prior (`MuleConfig.rf_prior_schedule_db`; the seconds model's `RFPriorProducer`) | the driver's `rf_prior_snr_db`: under `--l1-channel` the chosen band's mean SNR over the whole trial, later missions included; 20 dB otherwise | on the mission clock never the trial mean (critic B4): under `seconds` the SNR last observed at an upload on the carrier used; under `mission` with `--l1-channel` the L1 trace's SNR at each upload already made, adopted after each mission that docked; 20 dB before the first upload | Phase 3, commit `ef1faa1` (§5j) |
| Simulated-order ingest (derived: mission clock, K > 1, and a quorum below K or `agg:fedbuff`; `TCPDockLinkServer(sim_markers=...)`) | off: UPs folded in arrival order, the recorded loop | the cluster holds each UP until no other mule can still send one that completed earlier (`SimOrderGate`) and folds in (`sim_upload_ts`, mule id) order; the dock queues registration, departure and clock markers with the UPs, and a mule whose dock connection ends counts as done. Needs `down_wait_s` on every mule (refused otherwise) | Phase 3, commit `ef1faa1` (§5j) |
| D4's CARP split (rides `mission_clock`) | 1 s per client at the cost model's cruise speed | the predicted Pass-1 airtime of one client at R_planar(b)/2 (1 s without a band), at the cell's cruise speed | Phase 3, commit `ef1faa1` (§5j) |
| Beacon inserts (`MuleSupervisor.offer_contact`) | none (decision D6) | mission clock only: an offered stop is inserted at its cheapest place only if the whole edited remainder passes the predicate, and never evicts a planned stop. Inert: nothing offers one in Phase 3 | Phase 3, commit `ef1faa1` (§5j) |
| Driver wall budget (`--session-ttl-s`; the trial's hard kill) | a 3 s session TTL; the kill at `--trial-budget-s` (120 s); `down_wait_s` = that budget at K > 1 | the TTL at least 2× the 95th percentile of the real model's `train_offline` time, measured with N devices training at once (the exit gate's concurrency; the pilot plan of 2026-09-29); on the mission clock the kill is the larger of the budget and a bound built from the waits the code caps (562 s at a 3 s TTL, N = 6, one mule, 4 missions; K missions' worth per mission when the cluster orders uploads), and `down_wait_s` follows it | Phase 3, commit `ef1faa1` (§5j) |
| Analysis on simulated time (consumer and scorer) | the wall clock | the clock read from `mule_ready.mission_clock` (absent: wall, every recorded trace); missions ordered by their simulated ends; `sim_s_to_τ` beside `wall_s_to_τ`; 15 simulated columns; a trace whose clocks disagree is refused (`ClockDomainError`); without a trial CSV, a mission-clock marker's `ok` is relabelled `timeout` against the soft cap the runner applied (`soft_cap_s` in `trial_status.json`, written when `runner_main` ran the trial), not against the trial's own re-costed budget | Phase 3, commit `ef1faa1` (§5j) |
| Plan mode (`MuleConfig.plan_mode`; the driver's plan arms) | `legacy`: `FLScheduler.build_contact_queue` plans each mission as recorded, and at the defaults nothing loads the plan package (an H or D arm under `subset` loads its member walk) | `ferry`: at the dock `FLScheduler.build_ferry_plan` commits each mission to one band class b̄ and a Pass-1 route as one decision (S1, S3, the age cap, S3a once per class at R_planar(c) with the hover rule, the search, a guard fold under S3b's predicate, the commit); the mule flies b̄ in both passes and closes the plan once the merge is known. Simulated clock only; needs a `contact_band` and `t_nom_s`; refuses the `reorder` fallback (under `abort` too), `pass_2_budget`, a `contact_policy`, an RL selector, and `abort` together with a cap | Phase 4, commit `69b551f` (§5k) |
| Band-class policy (`band_class_policy`) | — (one `contact_band` for every stop) | `search` (arm F): every class of the link, each mission; `contact_band` is then only the reference class, which the ferry spec, T_nom, D4's split and the H and D arms of the same CSV use. `fixed:<class>` (Path B+, the FB+ arms): the run's `contact_band` only, with the `committed` flight slot only | Phase 4, commit `69b551f` (§5k) |
| Member admission, plan arms (`member_admission`; `--member-admission`) | `whole`, the field's default (plan mode is new) | `subset`, the plan arms' default (decision 4): the search and the plan-mode trim may fly a stop reduced to the members that fit. `whole` flies whole stops in every mode, in flight too (R5), which keeps the narrow-band cliff for comparison | Phase 4, commit `69b551f` (§5k) |
| Member admission, H and D arms (`member_admission`; `--member-admission subset`) | `whole`: S3b and the D1–D3 and D5 walks admit a contact with all its members or none (the narrow-band cliff, §5j E2E1-01) | `subset`, when a run asks for it (decision 4 (b)): before takeoff only, a contact that fails whole is re-issued with the members that still fit, skip not stop, in the arm's own member order (H1–H3: own deadline, then Pass-1 dwell, then id, the miss streak first under miss priority; D1–D3: the arm's per-device score; D5: FedCS's selection key). H complements are dropped by reason and widened; D complements are reported (`pass_1_policy_drops`) and never widened. Nothing in flight reduces their contacts. Never D4, whose tour has no gate | Phase 4, commit `69b551f` (§5k) |
| Flight slot (`flight_slot`) | the queue's order (`remainder.pop(0)`): distance order, or `TargetSelectorRL` within a bucket | `committed` (F, the FB+ arms): the same pop, on b̄. `cross_heuristic` (FX, decision 5): in Pass 1, after each stop, the nearest remaining stop whose move to the front keeps the rest of the plan feasible (else the plan's next), and at each arrival the fastest class that still reaches every device b̄ reaches there, priced at the arrival SNR, so it never dwells longer than F would at that SNR. At takeoff the plan's first stop is flown (the band rule still applies on arrival there), and Pass 2 flies the committed order on b̄ | Phase 4, commit `69b551f` (§5k) |
| Flight slot `pair_q`, the learned (band, next stop) score (`flight_slot`; the FQ arms), with its checkpoint (`pair_checkpoint`, `pair_checkpoint_sha256`, `pair_checkpoint_tag`) | the slot's fixed fillings above; the three checkpoint fields None | `pair_q`, plan mode only, with `in_flight_response = replan` (resolution R3) and refused under a pinned band: at each Pass-1 arrival one decision, the class the stop is served on, at once, and the stop flown next (home only once the remainder is empty), the masked argmax of a score over the pairs that keep the plan whole (a class reaching every device b̄ reaches there at the arrival SNR, a stop of the plan, the rest of the flight still fitting: `FLScheduler.fits_after_service`), ties to the lowest row; FX's pair, recorded `mask_empty`, when no pair fits. The chosen stop is moved to the front after the stop, so the departure check folds that order. At takeoff the plan's first stop is flown (the decision still applies on arrival there), and Pass 2 flies b̄ in the queue's order. The score is a verified format-2 checkpoint, named by all three fields together (its path, the sha256 of its arrays, its tag), never a random network; each decision is recorded in `mission_completed.pass_1_pairs` | Phase 5, commit `9694775` (§5l) |
| Contact policy `chen_dqn`, arm E3 (`contact_policy`), with its checkpoint (`policy_checkpoint`, `policy_checkpoint_sha256`, `policy_checkpoint_tag`) | None or D1–D5's; the three checkpoint fields None | `chen_dqn` (decision 7 (a)), on the simulated clock in legacy mode with a `contact_band` and whole stops (refused in plan mode and on the wall clock): a numpy port of Chen et al.'s DQN as a whole scheduler that admits every S3a contact, checks nothing in flight (`none`) and names each Pass-1 stop itself, at takeoff and at every departure, among the stops S3b's single-contact budget rule admits (landing included); its None ends the pass. Flown from a verified checkpoint named by all three fields; each call recorded in `pass_1_e3`, the stops it left in `pass_1_e3_unvisited`, never widened | Phase 5, commit `9694775` (§5l) |
| E3's per-departure hook (a policy's `chooses_next_stop`, read with `getattr`, default False) | — (no policy, slot or selector declares it) | a legacy-mode whole-scheduler policy that declares it `True` names the next stop at takeoff and at every Pass-1 departure, after the departure check and the beacon hook; never in Pass 2 | Phase 5, commit `9694775` (§5l) |
| FerrySim's seam (`MuleSupervisor.install_flight_slot`) | never called on a recorded path | installs a `PairQSlot` in plan mode before the first mission (refused on a legacy mule, once a mission has started, for anything but a `PairQSlot`, and under a pinned band): FerrySim's FQ episodes fly the FX arm's configuration with it, mission for mission as the config path flies | Phase 5, commit `9694775` (§5l) |
| Arm `H1+L1` (`backhaul_policy`) | — (only H3 flies the adaptive backhaul) | H1's scheduler with H3's adaptive backhaul controller and no learned selector (decision 8 (a)): `backhaul_policy = adaptive` on the simulated clock, `backhaul_plan(adaptive=True)` under `--l1-channel`; refused where it would fly as H1 (without `--l1-channel` and without, on the simulated clock, `--backhaul-model seconds`) | Phase 5, commit `9694775` (§5l) |
| Arm lists (`driver.LEARNED_ARMS`, `PHASE_5_ARMS`, `ARMS`; `is_plan_arm`) | `ARMS` = `DEFAULT_ARMS` + `PLAN_ARMS`; every plan-mode gate reads `arm in PLAN_ARMS` | `LEARNED_ARMS` = FQ, FQ-hand, FQ-dwell, FQ-cov, FQ-g0, FQ-g25, FQ-g50, FQ-g75, FQ-g90, FQ-g99 and E3; `PHASE_5_ARMS` = `LEARNED_ARMS` + `H1+L1`; `ARMS` = `DEFAULT_ARMS` + `PLAN_ARMS` + `PHASE_5_ARMS`. `DEFAULT_ARMS` and `PLAN_ARMS` are unchanged, so the Phase 5 arms run only when named. `is_plan_arm` (the plan arms and the FQ arms) replaces every `arm in PLAN_ARMS` gate, so an FQ arm gets F's settings (critic A5) | Phase 5, commit `9694775` (§5l) |
| Runner `--require-trained` | off | refuses H2 and H3 without `--selector-weights`, whose selector would be random-init (decision 8 (a)) | Phase 5, commit `9694775` (§5l) |
| Scorer `pair_columns` (`--pair-columns`) | off: the Phase 4 row | the seven Phase 5 columns after the τ columns | Phase 5, commit `9694775` (§5l) |
| Age cap S (`age_cap_missions`; `--age-cap-missions`) | off (None) | an int ≥ 1, counted in the device's own mule's missions since its last merged update (`last_merged_round`, mission 0 for a device never merged; decision 1), plan mode only. The plan serves the oldest capped devices that fit before it weighs anything else (the cap key), a stop whose members are all capped is exempt from its deadline clause, and every capped device a mission fails is logged by cause. S is set per cell from the S\* tool: the smallest value that covers 90 % of layouts at both pilot budgets, never below 2 | Phase 4, commit `69b551f` (§5k) |
| Cap lookahead L (`age_cap_lookahead`; `--age-cap-lookahead`) | 0 | a device is capped from age S − L. Kept at 0 (R9): L = 1 does not remove the miss of critic probe A3 | Phase 4, commit `69b551f` (§5k) |
| Plan score (`plan_score_params`; `--plan-score-params`) | {}, the defaults (read in plan mode only) | V = −[c₁(Δ/T)² + c₂U + c₃L] − c₄E/(P_hover·T): Δ the whole mission on b̄ (Pass 1, the turnaround, Pass 2) against T = T_nom (decision 2 (b)); c₁ = 1 (`c_time`), c₂ = κ·N_demand with κ = 1 (`c_cov_per_device`), c₃ = c₂ (`c_link`), c₄ = 0.1 (`c_energy`); `coverage_weights` `age`, the age times (1 + miss streak) under the arm's miss priority (decision 3); `dwell_in_delta` False is F-dwell; `coverage_rank` `lexicographic` ranks the candidates by the cap key, then the served weight share, then V (R11), `weighted` by the cap key, then V (the pilot's κ sweep), and with κ = 0 (F-cov) the weighted rank applies whatever it says. Hand-set and swept, in FedEx's form with the convex-surrogate caveat | Phase 4, commit `69b551f` (§5k) |
| Plan search (`plan_search_params`; `--plan-search-params`) | {}, the defaults (read in plan mode only) | `exact` (every ordered stop sequence, each stop reduced to every member subset) when the demand has at most `exact_max_devices` (6) devices; `stop_subsets` (ordered stop subsets, each stop whole if it fits, else reduced greedily) when a class has at most `exhaustive_max_stops` (6) stops; `local` above (a 2-OPT tour, its trim, first-improvement scans), bounded by `heuristic_max_passes` (50) and `heuristic_max_evaluations` (2,000 walks per class): counts, never wall time | Phase 4, commit `69b551f` (§5k) |
| Runner's default arm list (`driver.DEFAULT_ARMS`) | every arm of `driver.ARMS`: H0–H3 and D1–D5 | `DEFAULT_ARMS` keeps those nine. The plan arms (`PLAN_ARMS`: F, FX, FB+wide, FB+medium, FB+narrow, F-cov, F-cap, F-prio; `ARMS` is both lists) run only when named with `--arms`, on the simulated clock, and the runner refuses one the driver cannot run before any trial (`Exp4Driver.check_arm`) | Phase 4, commit `69b551f` (§5k) |
| D-arm drop report (`FLScheduler.last_policy_drops`; `mission_completed.pass_1_policy_drops`) | none: D1–D5 say nothing of what they leave out before takeoff (`pass_1_preflight_drops` is []) | on the simulated clock, each contact the D1–D5 walk left out before takeoff (under `subset`, the rest of a contact it served in part), labelled with the clause that refuses it alone from takeoff under the arm's in-flight rule, else `budget`; written only when non-empty, with `"widened": false`, and never widened (decision 6). Not a switch: the one trace-event field that can appear at the defaults (§5k) | Phase 4, commit `69b551f` (§5k) |
| Hover stops (plan mode; no switch of its own) | — (S3a's stops) | under a budget and a cap, each capped device that its own S3a stop cannot serve alone within the budget leaves that stop for a one-device stop at its best hover point on the class: the point of the dock-to-device segment, within the class's reach and above the SNR floor, that minimises its alone Pass-1 mission (the user's decision of 2026-09-30, after the final check's PLAN-1 and E2E2-01). `unplannable` then means that no class the arm may fly serves the device alone within the budget even there | Phase 4, commit `69b551f` (§5k) |

The Phase 3 physics values are parameters, not switches: the SNR floor, altitude, path-loss
exponent, shadowing σ and margin quantile (D1); the interference regime and period, the noise bin,
the shadowing correlation and keying (D2); the cruise speed, turnaround, listen window and the
SIMULATED energy powers and capacity (D3). Each sits in `MuleConfig`, is written to `mule_ready`
and to the `ferry_params` provenance column, and is listed in the Configuration Reference, §17.

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
                                          and the feasibility model; as landed also the deadline
                                          time unit, the override refusal, the pre-flight order
                                          check, fold_remainder / replan_remainder and the T_nom
                                          helper, §5j), Phase 4 (as landed: plan mode, S3a once
                                          per band class, the commit and its visited set, the
                                          plan-mode re-plan; in build_contact_queue only the
                                          D arms' drop report, its reset, and the member-subset
                                          carrier, §5k), Phase 5 (as landed: one method,
                                          fits_after_service, the pair mask's predicate, §5l)
hermes/scheduler/stages/s3_deadline.py  Phase 1 (law, priority key, PARTIAL vs TIMEOUT, overrides
                                          that expire), Phase 3 (the law's time unit, the override
                                          refusal, the SpectrumSig fold, §5j)
hermes/scheduler/stages/s3a_cluster.py  Phase 4 planned the radius per band class; the stage
                                          already takes it, so the file is untouched (§5k)
hermes/scheduler/stages/s3b_feasibility.py  Phase 1 (priority key in the walk), Phase 3 (one
                                          FeasibilityModel; single-contact predicate), Phase 4
                                          (additive: dropped_plan, the member_admission values,
                                          the opt-in member-subset walk, §5k)
hermes/scheduler/selector/              Phase 5 (as landed: pair_features.py, pair_q.py and
                                          pair_replay.py, new, under this opening; in
                                          scope_guard.py one function, assert_pairs_admitted;
                                          ddqn.py, replay.py, features.py, target_selector_rl.py,
                                          selector_train.py, sim_env.py and __init__.py
                                          untouched; the trainer is experiments/ferrysim/,
                                          outside this surface, §5l)
hermes/mule/mule_main.py                Phase 1 (version threading, Pass-2 budget), Phase 3 (clock,
                                          band, re-plan), Phase 4 (plan mode, the flight slot, the
                                          exempt set in flight, the plan's close, §5k), Phase 5
                                          (as landed: the pair choice at each Pass-1 arrival, the
                                          reorder after the stop, the records closed on every
                                          exit, the pair_slot keyword, install_flight_slot and
                                          E3's per-departure hook, §5l)
hermes/scheduler/policies/              not frozen (§5); Phase 4 (the member-subset keyword in
                                          budget_walk.py, fedcs_degraded.py, max_aoi.py, oort.py
                                          and whittle.py; cross_heuristic.py, new, §5k); Phase 5
                                          (pair_slot.py, next_stop.py and chen_dqn.py, new;
                                          cross_heuristic.py and __init__.py untouched, §5l)
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
`now_fn`. (The clock's time unit did need one: the deadline constants were set against missions
of about 10 s of wall clock, and Phase 3 scales them, §5j.) New modules (`scheduler/plan/`,
`scheduler/routing/`, `stages/s3d_age_cap.py`, `l1/contact_link.py`, `l1/channel_model.py`,
`l1/mission_clock.py`) sit outside the old frozen surface but follow the same three rules.

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
| D1 | S3b mechanism frozen; the budget is a matrix parameter | One FeasibilityModel — transit + bytes/rate + return + upload, energy clause declared simulated. The budget stays a study parameter: a knee re-measured with the ferry model at the Phase 3 pilot, and a stress budget below it (the plan's 60 s and 30 s were priced without the return leg and the upload; §5j). |
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

**Invalidated sweeps: none by this amendment.** Exp 5 runs every mule arm (H1–H3 and D1–D5, later
F) on the simulated clock once Phase 3 lands, with the seconds-axis backhaul where a study chooses
it, so each is re-baselined there — the re-run bill the build plan accepts in its decision D2. H0
is not among them. *(Corrected 2026-09-29: this sentence first included H0 (critic A5) and put
every arm on the seconds-axis channel, whose backhaul is now a per-study choice (spec Q7); D3–D5
and F are added.)* H0 runs in process with no mule, and its simulated round time is outside
Phase 3, so the driver refuses H0 on the simulated clock (§5j); it stays a wall-clock reference in
a CSV of its own. Legacy defaults keep the Exp 4 harness runnable as it is. The pre-re-run
checklist is re-opened (§1a there).

## 5h. Amendment 8 — baselines are budget-checked in flight; a plan's diagnostics are its own (2026-09-28)

Found in the Phase 0/1 audit and the Phase 2 baseline scout. Landed in commit `c417554`.

**1. The in-flight re-check held D1/D2 routes to our per-device deadline.** Amendment 4 gave the
whole-scheduler baselines admission authority: they replace S3, S3b and S3.5 and return the route.
But before every Pass-1 contact `MuleSupervisor._remaining_is_feasible` still ran S3b's
`filter_feasible` on the next stop, deadline test included, whatever the arm. S3 computes a
deadline for every device in every arm, and a device idle longer than its window Φ gets one in the
past. MAX-AoI puts exactly those devices first, so the check refused D1's own first choice and the
mule aborted the rest of the route. Reproduced with `FLScheduler` + `MaxAoIPolicy`, a 60 s budget
and a device last served 200 s ago: D1 routes it first with `deadline_ts = now − 140 s`, and the
in-flight check keeps nothing.

*Fix.* A whole-scheduler policy declares what the mule re-checks in flight, as a class attribute
`in_flight_check`. `budget` (D1, D2, and the default for any policy with `admit_and_order` that
declares nothing) tests only that the next contact still fits the mission budget from the mule's
actual pose and clock, priced like `greedy_budget_walk`. `none` flies the route as planned; it is
reserved for FedEx-Async's never-skip tour (arm D4). H0–H3 have no `admit_and_order` and keep the
full S3b check.

**2. A plan's feasibility result outlived the plan.** `FLScheduler.build_contact_queue` set
`last_feasibility` only when it reached the S3b gate. A plan that returned earlier (no eligible
device, none bucketable, no contact) left the previous mission's result in place, and the mule
widened that mission's dropped devices a second time. The plan now resets `last_feasibility`, and
the new per-device deadline map `last_plan_deadlines`, on entry. This fires only for a plan with no
eligible or bucketable device, which the Exp 4 topologies do not produce (every device stays in its
mule's slice), so no recorded cell is expected to move. It is recorded here because it changes
legacy behaviour.

**Frozen surface touched:** `hermes/mule/mule_main.py` (`_remaining_is_feasible`) and
`hermes/scheduler/fl_scheduler.py` (a read-only `target_selector` property; the reset). Outside it:
`in_flight_check` on `MaxAoIPolicy` and `OortPolicy`, and the `IN_FLIGHT_BUDGET` /
`IN_FLIGHT_NONE` constants in `policies/budget_walk.py`.

**Recorded sweeps affected:** every budgeted D1/D2 cell — the SOTA pilot, the budget axis and
`b60`. All of them are already due for re-run under Amendments 5 and 6; land this before those
re-runs. Part 1 leaves H0–H3 unchanged.

**Also landed with this amendment, in ferry mode only.** These sit behind switches whose default is
the recorded pipeline (Rule 1), so they change no legacy behaviour; the Phase 1 rows of the §5g
table are updated.

- *Train-ahead* (`pass_2_budget` on). A device collected in Pass 1 and then skipped by the budgeted
  Pass 2 had nothing prepared, so its next contact trained in session on the new θ and reported
  age 0: budgeted Pass 2 spread ages only for devices missed in both passes. The Pass-1 push now
  carries `train_ahead`, and the device trains on the adopted basis on a background thread. A
  delivery that arrives meanwhile replaces the basis, and the stale result is discarded.
- *Cutoff snapshot.* The D5 cutoff a_max_j = ⌊Φ_j·s/T⌋ is computed once per mission, right after
  planning, from the window the device was admitted under. It used to be read at close, after
  every CLEAN had already tightened Φ_j.
- *Deadline law.* The factor follows reachability. `RoundCloseDelta.answered` is set when the
  device's advert arrived, and an answered miss relaxes at β_partial whatever its outcome tag (an
  Exp 4 uplink drop is a TIMEOUT but the device was reachable). The multiplicative step applies to
  the clamped window, clamp(β·clamp(Φ)). Synthetic TIMEOUTs are marked `synthetic`.
- *Age-aware merges* (`agg:cutoff`, `agg:asynchfl`). Staleness now shrinks the step instead of
  being normalised away. The mule divides by the staleness-free mass Σ n_i·v_i of the admitted
  updates, and the cluster folds θ + η·Σ M_m·s_m·Δ_m / Σ M_m over live partials.
  `value='loss'` uses the raw loss, so partials from several mules combine as one merge. A fold
  whose every partial is past `a_max` takes no step, leaves the round open and still releases every
  waiting mule with a DOWN (`cluster_merge_expired`). FedBuff skips `min_participation` (K is its
  own quorum) and defaults K to the slice size of the mule whose partial first opens the buffer
  (fixed for the run), and the cluster reports a
  `last_outcome`.
- *Trace fields.* New additive fields:
  - `pass_1_merged_devices` on `mission_completed`, and per-device deadlines in `pass_1_plan`;
  - the round report of a mission whose every update was cut off;
  - the mule's effective settings in `mule_ready`, and `fedprox_rho` in `device_ready`;
  - `mission_round` and `partials` (and `expired_partials` for a partial cut inside an applied
    fold) on the cluster's merge events;
  - a `trial_status.json` beside each kept trace.

**Also landed: analysis changes that re-score legacy traces.** These are not ferry-mode switches:
they change how `experiments/analysis/traces_scorer.py` and `experiments/exp4/metrics.py` read
every trace, recorded ones included, while leaving every run's behaviour alone. The scorer now:

- credits merged updates, not merely collected ones, and follows FedBuff deferrals to their
  flush (the metrics' quorum thresholds credit a flush with every update it releases);
- leaves `jain_merged` blank for a trial that merged nothing. 50 of the 600 kept trials change
  from 1.0 to blank: 43 of the 120 `b60` trials (H1 9, D1 16, D2 18, so the per-arm `b60` Jain
  means fall by about 0.2–0.35), 3 of the 60 pilot trials and 4 of the 380 `exp4_matrix`
  trials. All of them are in cells already due for re-run under Amendments 5 and 6. No other
  recorded column moves;
- scores misses against each device's own deadline where the trace has it;
- skips trials whose status is not ok, and carries the provenance columns.

## 5i. Amendment 9 — mule failures fail the trial; bootstrap and reconnects survive (2026-09-28)

Phase 2 makes multi-mule runs real (build plan, Phase 2; commit `c417554`). Almost all of it sits behind the new
switches in the §5g table and is inert with one mule. Four parts change one-mule behaviour, but
only on a fault path that no recorded run took.

**1. A mule failure now fails the trial.** A mule whose loop ends on `mission_failed` exits with
code 3, and one that never gets its bootstrap exits with code 4; both used to exit 0. The driver
(`Exp4Driver._run_topology`) raises `Exp4MuleFailure` when any mule exits non-zero on its own,
which includes code 1 from an uncaught exception after the last mission, and the row gets
`status=error`. Before, the exit code was ignored and a truncated trial was recorded as `ok` (or
a zeroed row, or `no_eval`). None of the 28 committed Exp 4 CSVs has `mission_failures > 0`; a
crash after the last mission would not show in a CSV either way.

**2. A slow bootstrap no longer kills the mule.** The bootstrap DOWN wait blocked for 10 s and
raised, so a mule whose first DOWN came late died with a traceback and exit code 1, and the
existing `dock_bootstrap_timeout` branch could not fire. `wait_for_initial_dock(timeout)` now
returns False on a timeout, the mule keeps waiting in 1 s ticks inside its 30 s window, and then
emits `dock_bootstrap_timeout` (exit code 4). The cluster bootstraps each mule as soon as it
registers instead of waiting for every expected mule; with one mule that is the same moment.

**3. A mule that reconnects is served again.** The cluster's bootstrapped set only grew, so a
mule that re-registered under the same id never got a second bootstrap; and the dock server
overwrote the socket without closing the old one, whose reader then closed the NEW socket when it
ended. Now the server closes the old socket on re-registration, a reader drops only its own
socket, and the cluster forgets a mule that left the dock, so a reconnecting mule gets a fresh
bootstrap DOWN (event `mule_bootstrapped`). No recorded run or single-mule test restarts a mule.

**4. A refused upload keeps its reports.** When the cluster refuses a second partial from a mule
that already has one in the open round, it now still folds that upload's round report and
Pass-2 ledger into the registry (a resend of the same mission is still ignored), and the trace
marks it `partial_refused` with `held_mission_round`. This path needs a quorum above 1, so it is
unreachable with one mule.

**Files:** `hermes/processes/mule.py`, `hermes/processes/cluster.py`,
`hermes/transport/tcp_dock_link.py`, `hermes/transport/dock_link.py`,
`hermes/mule/client_cluster.py`, `hermes/cluster/host_cluster.py`, `experiments/exp4/driver.py`.
`mule_main.py` changes only behind `dock_on_empty` and `down_wait_s`.

**Recorded sweeps affected: none.** Every recorded run used one mule that neither failed,
bootstrapped late nor reconnected. Checked by re-running four single-mule configurations
(plain, cutoff, FedBuff, FedEx; with and without backhaul loss) on the code with and without
these edits: identical event sequences and fields.

**Also landed with Phase 2, behind switches (Rule 1; rows added to the §5g table):**

- *Several mules* (`n_mules`, default 1 = the old topology, byte for byte). At K > 1 the mules
  are `exp4-mule-<k>`, the devices are split into K contiguous angular sectors (sizes within
  one), every mule starts at the dock at the origin, and backhaul loss draws from one stream per
  mule so paired seeds survive any upload order.
- *Replies only to waiting mules.* After a merge, deferral or expiry the cluster sends a DOWN
  only to the mules whose upload it holds; it used to send one to every connected mule, in
  flight or not, and each mule read the oldest of a growing backlog, so it flew stale θ. The
  mule keeps the newest DOWN. With one mule the waiting mule is always the uploader.
- *Quorum* (`min_participation`, default 1). `agg:plain` with several mules must wait for all of
  them (a quorum of 1 made it last-writer-wins); a quorum strictly between 1 and K is refused
  (it can strand the last partial) except under FedBuff. Under a quorum above 1 a lost backhaul
  upload holds its mule's place with an empty partial (`backhaul_upload_lost.awaits_quorum`), so
  the mules stay in step.
- *Docking an empty mission* (`dock_on_empty`) and *a bounded DOWN wait* (`down_wait_s`): an
  empty mission still docks with an empty partial that counts toward the quorum, and a DOWN that
  does not come in time is survived (`dock_down_timeout`: restage θ, skip Pass 2, fly on). Both
  default on at K > 1 (the wait at the trial budget) and off at K = 1.
- *Arms D3–D5* (`contact_policy` `whittle`, `fedex`, `fedcs`), the merge rule `agg:fedex`, and
  the D4 assignment: `carp_assign` once per trial at K > 1, passed as the slice assignment
  (static, since each device is wired to one mule). The D4 tour returns to the dock
  (`FedExCarpPolicy(depot=origin)`), and its in-flight check is `none` (FedEx never skips).
- *Scheduler state for D3:* `reach_attempts` / `reach_answered` (real attempts only, answered
  or not) folded in `fold_round_close_delta`, `last_merged_round` set by the mule for the
  devices its merge used, and `SelectorEnv.mission_round`. Read by nothing but D3.
- *Analysis at K > 1:* every per-mission set is keyed by (mule, mission round); Pass-2 coverage
  divides by the mule's own slice; each device ages in its own mule's missions; uploads the
  cluster never folded (refused duplicates, partials still waiting at the end) are not credited,
  and the place held for a lost upload counts only as that loss, under every rule (a FedEx or
  age-aware fold lists every empty partial apart from what it merged).
  Re-scoring the 600 kept one-mule trials gives zero differences.

## 5j. Amendment 10 — a silent device keeps its RF link (finding P-02); Phase 3 behind switches (2026-09-29)

Lands with FeRRy Phase 3 (build plan, Phase 3; commit `ef1faa1`). It is the
one change to legacy behaviour in Phase 3 (Rule 3). Everything else in the phase sits behind the
Phase 3 switches of the §5g table and is listed after it.

**The defect (finding P-02).** Both ends of the RF link set a socket timeout that bounded reads as
well as sends: 30 s on the mule's reader for each device, 60 s on the device. A device that sent
nothing for 30 s of wall time was dropped by the mule and never came back, and its service loop
then spun: `serve_once` returned None at once, forever (353,202 calls in 0.2 s in the P-02 probe).
What kept devices registered was an accident: every solicit was a broadcast that reached every
device, and every device replied. Phase 3's targeted solicits remove that keepalive, so this fix
lands first (build spec, Q3).

**What changed:**

1. *RF reads have no timeout* on either side, as the dock link's reader already had. A reader
   ends when its socket is shut down and closed. Sends stay bounded, by `SO_SNDTIMEO` (30 s on
   the mule, 60 s on the device), packed per OS by `sndtimeo_optval`
   (`hermes/transport/tcp_dock_link.py`): Winsock reads a DWORD of milliseconds, POSIX a
   `timeval`. The bound applies to each send call.
2. *The dock link's own send bound.* It packed a `timeval` on every OS, so Windows read its 60 s
   bound as 60 ms, and a bound under 1 s packed as no bound at all. The same helper fixes it. A
   POSIX host packed it correctly before and is unchanged.
3. *A device that registers again replaces its socket.* The old socket is shut down and closed.
   A reader that ends late, or a send that fails on the old socket, drops only its own socket,
   never the new entry (the dock server's pattern from Amendment 9). A failed send on the device
   now closes its socket, and a failed broadcast send drops its device under the lock.
4. *The device re-dials.* `DeviceService.run` no longer spins on a dead link, whatever took it
   down (the mule dropping the device or exiting, a failed send). It re-dials with backoff: 0.5 s,
   doubling to 10 s; a link that drops again within 30 s of a re-dial resumes from twice its last
   wait. A re-dial counts only when the mule acknowledges the registration within
   `connect_timeout_s`. The first registration asks for no acknowledgement and gets none, as
   before. A successful re-dial emits `device_reconnected` (`attempts`, `down_s`) and counts
   `rf_reconnects`. A failed one records nothing, so the link dropping at the end of every trial,
   when the mule exits, leaves the device's trace as it was.
5. *An optional link token* (`MuleConfig`/`DeviceConfig.rf_link_token`; None, unchecked, by
   default). A mule started with one refuses a registration that carries another. A device whose
   mule has exited therefore cannot re-dial into another trial's mule that later took the same
   port and evict that mule's device of the same id. The Exp 4 driver gives each mission-clock
   trial one token (a hash of cell, arm, trial index and seed); `--rf-link-token` forces it on or
   off.

The registration frame and the device's per-role JSON gain fields at their defaults, which critic
A3 allows.

**Not only fault paths (critic D2).** Before the fix, any 30 s of silence dropped a device,
whatever caused it: a quorum wait at the dock with several mules (up to `down_wait_s`, which is
the 120 s trial budget), a synchronous fit longer than 30 s, or a device that registered early and
waited through the mule's startup; with Phase 3's targeted solicits, also the stops a device is not
solicited at. Unit U0 built a K = 2 case in which one mule holds a contact open for 36 s. At
afa9526 all six devices were dropped exactly 30.0 s after their last frame: Pass 2 delivered to 0
of 3 devices on each mule, mission 2 had no submissions to aggregate, and one cluster round closed.
With the fix Pass 2 delivered to 3 of 3 each time, and two rounds closed.

**Verification.**

- The P-02 probe: at afa9526 the device is dropped and spins; with the fix it stays registered,
  makes one call, and its advert comes back.
- Before and after, on afa9526 plus only these files: K = 2 stub trials through `Exp4Driver`
  (quorum 2, `agg:plain`, `dock_on_empty`, 4 missions, N = 6, seeds 3, 7, 11 and 29) and the
  real-model trial of `test_exp4_realmodel_smoke.py`. Every CSV row matched apart from its
  wall-clock column, every role's trace had the same events once timestamps were masked, and no
  link dropped or re-dialled. The real-model trial closed 2 rounds (AUC 0.488 initial, 1.0 best)
  both ways.
- The recorded runs: across the 600 kept Exp 4 traces (all one mule) the longest device silence,
  an upper bound read from the event timestamps, is 23.3 s, and the longest startup gap 3.0 s.
  Both stay under the 30 s that dropped a device, so no recorded run is expected to move. Process
  stderr is not kept, so a drop cannot be read from the logs directly.
- 95 tests in `tests/unit/test_rf_link_targeted.py`, `test_sndtimeo.py`,
  `tests/integration/test_device_service_reconnect.py` and `test_rf_link_amendment10.py`. Run
  against afa9526, their first version failed (35 failed, 3 passed, and `test_sndtimeo.py` did
  not import).

**Files:** `hermes/transport/tcp_rf_link.py`, `rf_link.py` and `tcp_dock_link.py`,
`hermes/processes/device.py`, and two `DeviceConfig` fields in `config.py`; the token is wired in
`hermes/processes/mule.py`, `experiments/exp4/topology_builder.py` and `driver.py`. None of them
is in the frozen surface.

**Recorded sweeps affected: none expected** (see the verification). What moves is a run in which
a device is silent for more than 30 s (item 1: a quorum wait with several mules, a long fit, a slow
startup), the cluster's dock send on Windows blocks for more than 60 ms (item 2), or a device's link
drops for another reason, since the device now re-dials and re-registers (items 3 and 4). None is
expected in the recorded runs, and no run with several mules has been recorded.

**Also landed with Phase 3, behind switches** (Rule 1; the rows are in the §5g table). None of
these changes legacy behaviour: with every switch at its default the golden fixtures below pass
unchanged, and the mule and cluster processes' events and DOWNs were checked byte for byte against
afa9526's modules.

- *Golden fixtures* (`tests/golden/`, 144 tests; see its `README.md`). Oracles captured at
  afa9526 before any Phase 3 edit: the Exp 4 backhaul channel (the 120 recorded C1/C2 trials
  re-derived), the feasibility walks (2,400 seeded instances and the exact-boundary cases),
  `HFLHostMission` (the contact map's 28 scenarios with synchronous and real threads, 12
  `run_session` scenarios, critic A2's late writer, the sequential joins of both passes),
  `MuleSupervisor` (scripted loopback missions, K = 2 cases included, critic B6), and the topology
  and driver (a builder grid, stub trials, and every re-derivable kept trace in `results/`).
  Dataclasses are compared on their afa9526 fields only, so a default field added later passes and
  a removed or renamed one fails (critic A3). `pytest_baseline.txt` records the full suite at
  afa9526: 6 failures, a baseline the user signed off on 2026-09-29. The sixth, the real-model
  smoke test (`test_exp4_real_model_synthetic_converges`), fails with `rounds_closed` 0 under load
  and passes on an idle host. Open follow-up: fix it if the session-TTL pilot shows the cause is a
  device's fit outrunning the 3 s TTL under load.
- *The mission clock* (unit U1, `hermes/l1/mission_clock.py`): `MissionClock`, a zero-argument
  callable usable as `now_fn`, with `advance(dt, kind)`, the monotone `advance_to` and a
  per-mission ledger of seven kinds; negative, infinite and NaN charges are refused. `FlightModel`
  (5 m/s, one dock at the origin, 30 s turnaround, 1 s listen) and `EnergyModel` (Zeng–Xu–Zhang
  2019, SIMULATED; the energy is a function of the ledger).
- *The channel* (unit U2, `hermes/l1/channel_model.py`). The legacy `ChannelModel`,
  `loss_from_snr`, `BackhaulPlan` and `backhaul_plan` moved verbatim, and
  `experiments/exp4/channel.py` re-exports them (the channel golden pins them). New: the
  seconds-axis `ContactChannel` and `BackhaulChannel`, whose noise is a pure hashed function of
  time and whose outcome draws are keyed by (salt, key, round), and the causal `RFPriorProducer`
  in `hermes/l1/rf_prior.py` (critic B4).
- *The contact link* (unit U3, `hermes/l1/contact_link.py`): decision D1's band classes, ranges,
  rates and dwell (Configuration Reference, §17).
- *One predicate and the re-plan* (unit U4). `FeasibilityModel(ferry=FerryPhysics(...))` with
  `leg`, `admit` and `fold` under three rules; S3b, the D-arm budget walk, FedCS and the FedEx
  diagnostics are folds over it; `routing/replan.py` behind `FLScheduler.fold_remainder` and
  `replan_remainder`; the pre-flight order check; `ContactWaypoint.band`, `range_m` and
  `pred_snr_db` (all `compare=False`); the deadline time unit; the T_nom helper. With `ferry` None
  every walk reproduces the 2,400 golden instances.
- *One contact routine for both passes* (unit U5, finding D-01). `run_contact` and
  `deliver_contact` are thin wrappers over one `_serve_contact`. Its legacy path is statement for
  statement afa9526's, P-01 defects 2 and 3 included, and its sink reads `_accepted` when it
  appends, so a late gradient still lands in the next round as before (critic A2); the host
  goldens pin it. Given a `ContactPlan`, the ferry path sends numbered solicits to the targets
  only, drains stale adverts, gradients and acks and matches replies by solicit id (critics B1 and
  B2), does not wait for uplink-dropped pushes, joins once at 2 × TTL, and commits in device order
  with stamps from the clock, which closes P-01 defects 2 and 3 in ferry mode only. No wall stamp
  reaches a line, a delta or a contact record (critic B3); the receipt TTL and the busy flags stay
  on the wall clock.
- *The supervisor on the clock* (unit U6, `hermes/mule/ferry.py` and `mule_main.py`). Whole
  missions on the clock, the legacy mission bodies textually unchanged. The clock is `_now`, and
  nothing is stored as `_clock` (critic B7). δ_obs is fixed at 0 (critic C1). Pass 2's energy
  counts from its own takeoff (a recharge or swap during the turnaround). So does the L1 state's
  energy slot 7 (1 − E/E_ref): design §4.6's E_mission is read as the sortie's energy, so in Pass 2
  the L1 state and the energy clause agree on the battery (final check, clock F2; recorded only, as
  no process wires a channel actor). Oort (D2) plans with its recorded round inference, and a
  re-plan within the mission reuses its plan's round, so recorded D2 planning is unchanged.
- *Processes, driver, cluster and topology* (unit U7): the configuration fields and their guards;
  the simulated fields of `mule_ready`, `mission_started` and `mission_completed` (wall-clock
  events at the defaults keep their recorded key sets; `delivery_overrun_s`, added in commit
  `d175afa`, joins the simulated `mission_completed` only under `deadline_bounds = delivery`, so every
  other value keeps its simulated key set as well), among them `pass_1_preflight_drops`, the
  Pass-1 pre-flight drops with their reasons, so an empty plan's trace says what emptied it (final
  check, E2E1-01); the cluster's simulated time and SpectrumSig forwarding; the ferry topology; 13
  provenance columns; T_nom per cell; the input-width pin (design R8: a real-model ferry cell must
  have 21 inputs on the canonical data); the re-costed wall budget; the runner's soft cap in the
  mission-clock status marker (`soft_cap_s`); and the refusal of `time_scale` in the driver's own
  `deadline_params` (both from the final check, below).
- *Analysis on simulated time* (unit U8). The 600 kept one-mule trials re-score with zero
  differences on every existing column.
- *Several mules in simulated order* (unit U9): `SimOrderGate` in `hermes/processes/cluster.py`
  and the dock markers. Critic B9's refusal of the simulated clock with several mules below a full
  quorum is lifted. The gate cannot deadlock (argued in its docstring, tested at K = 3), with the
  one exception below. A restarted mule is tracked under its live dock session once the cluster
  knows of it (its new bootstrap or its `registered` marker); from then on its older session's
  markers change nothing, whenever the cluster reads them (final check, protocol F1). Two gaps
  remain, both because UPs carry no session:
  - an upload the crashed process sent that the cluster reads after the restart raises the live
    mule's bound and can make its next upload fold late;
  - an upload the crashed process left held stays ahead of the restarted mule's uploads. It can
    make them fold late or, when another mule's upload falls between the two, stall the fold until
    a mule's `down_wait_s` runs out: the one exception to the no-deadlock argument.

  `test_p3_sim_order.py` pins the second gap. Only a manual or fault-injected restart reaches
  either: the orchestrator spawns each mule once. The gate's counters (`mules_departed`,
  `sim_order_late_uploads`, `sim_order_unordered_uploads` and the timer `sim_order_held_s`) and the
  seconds model's `backhaul_unpriced_uploads` are in-process registry metrics, written only in the
  end-of-run `metrics_snapshot`, which the orchestrator's Windows stop never lets the cluster
  write. Kept traces carry the same facts per event (`mule_departed`, `sim_order_late`,
  `sim_order_seq`, `sim_upload_ts`, `held_wall_s`, `p_loss`; Configuration Reference §17.6), and
  `test_p3_sim_order.py` pins the equivalence against the registry, with a wall clock that moves
  on every read so the held-time samples are compared by sum, minimum and maximum, not only
  counted. No stop behaviour changes: a graceful stop would add rows to every legacy Windows trace
  (final check, E2E2-1).

The Phase 3 tests pass (2026-09-29): 1,238 new tests in 36 new files, the 144 golden tests among
them. The full suite (2,798 tests) matches the afa9526 baseline test for test
(`tests/golden/make_baseline.py compare`): no outcome changed, no known failure changed its
signature, every baseline test ran and none of the new tests fails. Its 6 failures are the baseline's own.

**Frozen surface touched,** all behind the switches: `fl_scheduler.py`, beyond the accessors
Amendment 7 planned (the deadline time unit, the override refusal, the pre-flight order check,
`fold_remainder` and `replan_remainder`, the T_nom helper); `stages/s3_deadline.py` (the law's
`time_scale`, `DeadlineOverrideRefused`, the SpectrumSig fold, windows in missions);
`stages/s3b_feasibility.py` (the one predicate); `mule_main.py` (the clock). `s1_eligibility.py`,
`s3a_cluster.py`, `s3c_mission_window.py`, `s35_selector.py` and `selector/` are untouched.

**Visible at the defaults, additive only** (so not Rule 3 changes): wire frames carry the new
fields at their defaults and grow slightly (critic D4); per-role JSON gains keys at their defaults;
every trial row gains 15 simulated columns, blank on the wall clock, and 13 provenance columns,
blank at the driver's defaults. Wall-clock rows fill some provenance columns too: `realism`,
`l1_channel` and `input_dim` whenever set (a wall-clock re-run of a recorded real-model `--realism`
cell gets `realism` 1 and `input_dim` 21), and a deadline time scale other than 1.0, an
`initial_window_s` or a session TTL other than 3 s. The CSV header therefore changes, and the
runner refuses to append to a CSV written before (critic D3): write every run from now on to a
fresh path.

**Final cross-cutting check (2026-09-29).** Four review dimensions (spec fidelity, clock
accounting, the multi-process protocol, legacy identity) and two end-to-end replays of real
simulated-clock trials (one mule, several mules). Every physics quantity matched an independent
reading of the spec exactly, bar the deviations the unit reports disclose. Six findings, all low
severity, each fixed or documented: the notes on U6, U7 and U9 above, the §5g rows, and the items
below. None changes legacy behaviour, so none is an amendment.

- *Driver (not visible at the defaults).* `trial_status.json` gains `soft_cap_s` only on
  mission-clock trials run by `runner_main`, just as `t_nom_computed` appears only on
  mission-clock trials. On a grid with several N or mission counts the runner's cap (the largest
  budget over the grid) can exceed a trial's own budget, and a trace scored without its CSV was
  relabelled `timeout` where the runner recorded `ok` (legacy F1). Wall-clock markers keep the
  afa9526 key set on both the ok and the error path, and the scorer's rule for a marker without
  `soft_cap_s` is unchanged. A `time_scale` key in the driver's own `deadline_params` is refused
  with `DeadlineLawError`, as at afa9526. Phase 3's `DeadlineLaw` carries the time unit as a field,
  so `from_config` now accepts the key, which afa9526 refused; the driver refuses it in any form
  `dict()` accepts, so the unit cannot bypass the recorded `deadline_time_scale` column. afa9526
  refused the same configurations with the same exception type, so no recorded configuration
  changes.
- *What `delivery` bounds (clock F1), decided 2026-09-29: keep both readings.* Landed in commit
  `d175afa`, behind the switch (Rule 1; the §5g row). As `ef1faa1` built it, `delivery` was
  the plan's single-contact predicate, checked per stop. That reading is now `delivery_per_stop`,
  with the same arithmetic; the renamed two-stop test in `test_p3_feasibility_predicate.py` pins
  it. `delivery` is now route-level: every update collected on the route must reach the cluster,
  the route's landing plus the upload, by its own Deadline(j). It changed meaning with no rename
  shim: the setting is sim-only and new in Phase 3, and no recorded run or committed trace used it.

  *The predicate.* `FlightState` carries `deliver_by`, the earliest Deadline(j) of the updates on
  board: ∞ at takeoff, and read or lowered only under `delivery`. Under `delivery` a stop's
  clauses are tested in this order: its own deadline, home ≤ Deadline(j) as under
  `delivery_per_stop` (`overdue`); the updates on board, home ≤ `deliver_by` (`delivery`); the
  budget; the energy. Like the own-deadline clause, the on-board clause needs a budget and our
  arms' deadline rule. An admitted stop that is not protected lowers `deliver_by` to its own
  Deadline(j). A protected stop is still held to the updates on board but adds nothing, and a
  rejected stop adds nothing either. Home never decreases along a route, so the priced landing plus
  the upload meets every collected deadline.

  *In flight* (Pass 1), the mule's departure state carries the minimum own Deadline(j) over every
  member whose update was collected CLEAN at a stop already flown: the plan's Deadline(j), or an
  inserted member's from its insertion. This equals S3b's pre-flight assumption when every planned
  member answers and is never tighter. The departure check (`abort` or `replan`), the re-plan and
  the beacon hook all start from that state. Tests pin a silent member's deadline staying off
  board, alone or sharing a stop, an inserted member keeping its own deadline, and the beacon hook
  refusing an insert that would land an update already on board late. The bound is checked at
  departures only. A contact that runs longer than priced at the stop where Pass 1 ends (a silent
  member's listen window, a noisy band) is not re-checked: the updates are on board and the only
  way left is home, so an update can still land late, as the budget can be overrun.
  `MissionRunResult.delivery_overrun_s` records by how much. It is in `mission_completed` under
  `delivery` only, so every other value keeps its trace, and
  `test_delivery_cannot_recheck_a_contact_that_overruns_where_the_flight_ends` pins the case.

  Pass 2 and the whole-scheduler baselines have no deadline clause (D1–D3 and D5 check the budget,
  D4 nothing), so `delivery` changes nothing they fly, and on their missions `delivery_overrun_s`
  only measures. A `delivery` drop is widened like a budget drop, since the device was not late
  itself, counts in S3c's planned, and is recorded with reason `delivery`
  (`FeasibilityResult.dropped_delivery`, additive, listed after the other reasons). Before takeoff
  the arm's own order can fail the on-board clause where S3b's admission order passed: `trim` then
  drops the stop as `delivery`, and `reorder` repairs the order
  (`test_delivery_drops_before_takeoff_join_last_feasibility`). No study has run `delivery`; the
  scorer's deadline misses still measure the collection, and no CSV column reads
  `delivery_overrun_s`. The default, `collection`, is unaffected: the goldens give 144 passed
  before and after. Files: `stages/s3b_feasibility.py`, `routing/replan.py`, `fl_scheduler.py`,
  `mule_main.py` and `mule/ferry.py`; `processes/config.py` and `processes/mule.py`; the Exp 4
  `driver.py` and `runner_main.py`; 36 more test cases in seven existing test files.
- *Known interaction (E2E1-01, no Phase 3 change).* On narrow, S3a forms one field-wide contact in
  about 98 % of realism layouts, and every gate admits contacts whole. Under a budget below that
  contact's predicted home time every gated arm flies empty missions. At the default deadline unit
  1.0 the deadline clause can bind first: H1–H3 then drop the contact as `overdue` at any budget,
  until missed missions widen its window past its predicted finish, and only the budget-only walks
  (D1–D3, D5) show the budget cliff. Phase 4's member-subset admission is the fix for the cliff
  (Configuration Reference §17.1). *Decided 2026-09-29:* the fix is deferred to Phase 4, and
  narrow and medium cells are not compared under budgets below the cliff until then; their budget
  knees wait for Study 5.4. *(2026-09-30: Phase 4 lands member-subset admission behind
  `member_admission`, for the plan arms by default and for H1–H3, D1–D3 and D5 when a run asks;
  `whole`, the default, keeps the cliff and its pins, §5k.)*
- *The hard kill's bound* (`ferry_wall_bound_s`) takes ceil(N/K) devices per slice. Angular or
  CARP slices can be unbalanced, so at K ≥ 3, when the cluster does not order the uploads (a full
  quorum, not FedBuff), an all-timeouts worst case can exceed it (slices of 7, 1 and 1 devices at a
  3 s TTL: 136 s per mission against the bound's 128 s). In simulated order the K× factor covers
  any split. A healthy trial never comes near it.

**Deviations from the build plan** (also in `FeRRy_Build_Plan.html`, Phase 3):

- the exit gate *re-runs* the Phase 1–2 studies on the simulated clock instead of re-scoring them:
  legacy traces carry no simulated stamps;
- the fixed 60 s and 30 s budgets become a knee re-measured with the ferry model, plus a stress
  budget below it: the 60 s and 30 s were priced without the return leg and the upload;
- decision D2's "period and gain per band" becomes one interference period for every class (60 s)
  with a seeded phase per class and no random class gain: the classes share one carrier and differ
  structurally, through R(b) (a per-class period multiplier exists on `ContactChannel`, default 1,
  and is not a configuration field);
- R(b) is the range with 90 % link availability at the edge (mean SNR 5.13 dB above the floor),
  not the floor-rate range, which for wide is about 111 m slant, 1.71× longer (critic A8-i);
- Deadline(j) bounds the collection by default, arrival + dwell ≤ Deadline(j) (spec Q2). The
  plan's single-contact min(Deadline(j), budget) on the time back at the dock, applied per stop to
  that stop's own return and upload, is `deadline_bounds = delivery_per_stop`; it bounds when an
  update is actually delivered only for a route's last stop. `deadline_bounds = delivery` bounds
  the priced delivery of every collected update (the route's landing plus the upload), and records
  any in-flight overrun at the stop where Pass 1 ends as `delivery_overrun_s` (commit `d175afa`;
  clock F1 above). By default only the budget bounds that time;
- FedCS's "skip = stop" and its "no return leg" deviation hold only for the legacy model: under
  the ferry predicate admission tests the time back at the dock (docstring updated, critic B15);
- H0 is excluded from the simulated clock (critic A5, §5g);
- T_nom is the median over 20 reference layouts drawn from their own seeds (`_u32(N, "t_nom", k)`),
  independent of the grid's seeds and trial count, so resuming a CSV never moves it; with several
  mules each layout is priced as its slowest slice;
- the seconds model's loss probability is read at the upload's start, not at `sim_upload_ts`; the
  two differ by at most the upload's duration;
- `--l1-channel` is refused together with the seconds model: that would be two backhaul loss
  models;
- with several mules, a mule whose dock connection ends counts as done: the mule-sent `done`
  marker exists and is tested but is not wired;
- the arm's own order cannot act in the pre-flight check, because S3b has just admitted the whole
  queue, so `replan_fallback` decides what the H arms fly there (critic C3);
- on the mission clock without a band, solicits are targeted, not broadcast, so device-side counts
  differ from the recorded ones (in the clock-injection suite one device is TIMEOUT as recorded
  and PARTIAL on the clock);
- fresh CSV paths (above).

**Re-run bill** (decision D2): every mule arm re-runs on the new clock: H1–H3, D1–D5 and, later,
F. The Phase 3 exit gate re-baselines H1, D1, D2, D3 and D4 once the pilots have set the session
TTL, T_nom per cell, the deadline time unit, the budget knee and the H arms' `replan_fallback`; it
waits for the go-ahead. The pilot plan was decided on 2026-09-29 (Run Guide §2.6); no pilot has
run yet:

- the deadline unit is `--deadline-time-scale t_nom` (T_nom / 10 s), so with the default Φ₀ each
  device starts with about six missions' worth of window, as in the recorded runs;
- the ferry session TTL is at least 2× the 95th percentile of the real model's `train_offline`
  time, measured at the exit gate's concurrency (N devices training at once);
- the budget knee comes from an H1 sweep of `--mission-budget-s` on `wide`, with the measured
  payload and the `t_nom` unit, for each N of the gate's grid: the knee is where the served
  fraction stops rising. Narrow and medium knees wait for Study 5.4 (the cliff above);
- the cells fly `--in-flight-response replan`, and the H arms' fallback is `--replan-fallback
  trim`: each arm keeps its own order, as the D arms do, where `reorder` would make H1–H3 fly the
  same route whenever the pre-flight check fires.

A later pilot could run the knee sweep under both `trim` and `reorder` and choose by how often
the pre-flight check fires and what each costs in coverage; that is an idea on record, not part of
the plan.

## 5k. Phase 4 behind switches, no amendment (2026-09-30)

Lands with FeRRy Phase 4 (build plan, Phase 4, "Plan clock: reach as a decision"; commit
`69b551f`, on `6e6f92d`). It is not an amendment (Rule 3): nothing changes the wall clock, or Phase 3's
simulated clock, at the defaults. Every mechanism sits behind a switch of the §5g table whose
default is the recorded pipeline, in behaviour and in trace output, or is additive: the one
trace-event field that can appear at the defaults is the D arms' drop report, and every mule's
per-role JSON gains the eight plan fields at their defaults (below). "Legacy" now has two faces, and
both are pinned: the wall clock by the 144 afa9526 goldens (§5j), and Phase 3's simulated clock with
`plan_mode = legacy` by oracles captured at `6e6f92d` before any Phase 4 edit (unit UG4, below).

**Build-plan decision D4 and the seven decisions** (the user, 2026-09-30). Each is the
recommendation except the fourth.

1. *The age cap S (D4).* A device's age is the number of missions its own mule has flown since its
   last merged update (`last_merged_round`, mission 0 for a device never merged), the scorer's own
   unit. S is the smallest value that covers 90 % of layouts at both pilot budgets (the knee and
   the stress budget), never below 2: S = 1 would cap every device at every mission (critic A1). S
   is a configuration value; the S\* tool prints it, (i), and S + 1, (ii).
2. *Plan-score time.* Δ is the whole mission on the chosen band (Pass 1, the dock turnaround and
   Pass 2), measured against T_nom (`t_nom_s`, which plan mode requires). κ = 1: serve everyone the
   budget allows, and let time break ties. The pilot sweeps κ in {0.15, 0.25, 1} and c₄ in {0, 0.1}.
3. *Coverage weight.* Age × (1 + miss streak), declared roughly quadratic in age: every device the
   plan leaves out is widened as a miss, so the streak tracks the age (critic A5). F−prio drops the
   streak factor and weighs by age alone. F−cov keeps the plan's letter and is reported as "cap-only
   service".
4. *The narrow-band cliff fix, also for the H and D arms* (option (b), not the recommendation).
   Member-subset admission lands for the F family and, as an opt-in branch whose default is `whole`,
   in S3b's `filter_feasible` (H1–H3), `greedy_budget_walk` (D1–D3) and FedCS's walk (D5). D4 has no
   gate and is unaffected.
5. *FX.* `policies/cross_heuristic.py` is built now (below); the learned pair choice stays in
   Phase 5.
6. *D-arm drops.* Reported only, as `pass_1_policy_drops`, written only when non-empty, and never
   widened.
7. *The pilots.* The recommended plan (Run Guide §2.7). Nothing runs before the user's go-ahead, and
   not before the Phase 3 pilot has set the session TTL and the knee.

**Resolutions taken during the build** (the orchestrator's, each recorded where it lands):

- *R1.* A cap without the mission round raises (critic B9). Plan mode needs the round even with the
  cap off, since the age weights read it, and the commit refuses a cap without one.
- *R2.* The cap's stop rules (B2's deadline, exempt and priority stops) have one definition:
  `plan/member_subset.py` imports `stop_deadline`, `is_exempt` and `is_priority` from
  `stages/s3d_age_cap.py`, never the reverse.
- *R3.* The link term's outage formula is defined twice, by the runtime
  (`FerryRuntime.outage_probability`, which the planner calls) and by the score
  (`plan_score.outage_by_distance`); a test ties the two within 1e-15.
- *R4.* Every plan arm's commit records the predicted whole mission, `score.mission_s`, beside V's Δ
  (`score.delta_s`, which leaves the dwell out under F−dwell).
- *R5.* `whole` means whole in every mode: the search flies whole stops in all three of its modes,
  and the plan-mode re-plan keeps or drops whole stops (`FLScheduler._trim_whole`).
- *R6.* FX's next-stop rule acts only after each Pass-1 stop, never at takeoff or in Pass 2, and its
  band rule only in Pass 1. "Never dwells longer than F" holds at the arrival SNR, which is how
  decision 5 prices it: a contact charges each target at its own session start (critic C2).
- *R7.* The local search's bound stays at 2,000 walks per class. Its cap limitation is recorded, not
  fixed: no move swaps one stop for another, so where two capped stops compete for one slot it keeps
  the one the trim took first. Forced onto 1,496 small capped class searches it ended with a worse
  cap key than the exact search in 6.1 % (a replace move would give 2.5 %, an oldest-first start
  2.5 %, both 1.9 %).
- *R8.* The empty plan's band is the first searched class (by class index).
- *R9.* The lookahead L stays 0.
- *R10.* Pass 2 is never member-reduced: it delivers to whole stops on b̄, `pass_2_budget` is
  refused in plan mode (critic B8), and the budget walk refuses the subset carrier in Pass 2.
- *R11.* Coverage first (below).
- *R12.* Under `whole` the plan's drop labels (spec item 8) judge the stop: `plan` only when the
  whole stop fits from the dock at takeoff, else the clause that refuses it, so a capped member of a
  mixed stop that is late for its co-members' deadline reads `overdue`. The cap's violations judge
  the device alone, under either admission (the hover decision, below).
- *R13.* At S = S\*, arm F on critic layout 25 crowds three times over its 3S = 6 missions
  (missions 2, 5 and 6), not once as critic A4 found under the design's score: the user's score
  (decision 2 (b)) takes narrow {d0, d1, d2} at mission 1, where the design's took medium. Pinned as
  flown. Across the 30 layouts at S\* only layouts 1, 18 and 25 crowd anyone (1, 2 and 3 times),
  and only as `crowded`.
- *R14.* `score.mission_s` is a prediction: Pass 2 is priced on the class's S3a stops at plan time,
  but the mule rebuilds Pass 2's queue after Pass 1 has moved the deadlines, so the flown mission
  can differ (at U7's build, 3 of 270 deterministic loopback missions ran 2.3 to 7.9 s longer than
  predicted; in the final check's real FX trials the gap was up to 5.8 s, either way).

**Coverage first (R11).** U7's probe showed that V alone does not keep decision 2's promise to serve
everyone the budget allows. Every plan that serves anyone pays a whole Pass 2 on its class, and the
empty plan pays only the turnaround, so the empty plan can outscore a device that fits. In the probe
(u and v 60 m either side of the dock, z 400 m out; FB+wide, 1 MB, a 37.6 s budget, T = 200 s, the
cap off) serving v scored V = −3.85 (a predicted 264 s mission, its Pass 2 flying out to z) and the
empty plan −3.02 (its 30 s turnaround); with the weights growing together, the arm flew empty six
missions running. So `plan_score_params.coverage_rank` decides how the candidates that tie on the
cap key are ranked:

- `lexicographic`, the default: the served weight share, Σ_served w / Σ_demand w, then V. Every
  demanded device weighs more than 0, so the empty plan wins only when no plan that serves anyone is
  admitted, and time breaks ties among plans that serve the same weight;
- `weighted`: V alone (U0's `Candidate.key`), trading coverage against time at the rate κ sets. The
  pilot's κ sweep flies it;
- with κ = 0 (F−cov) the weighted rank applies whatever the setting says: cap-only service
  (decision 3).

The rank changes neither V nor its terms nor `mission_s`, and `weighted` is the search as built
before it, bit for bit (a digest over 184 random problems). Under `lexicographic` a single
first-improvement scan never takes a drop or drop-member move, since each serves less weight, and
alone it ended below `weighted`'s plan under its own key on 3 of 600 random local problems at κ = 1
and 3 of 327 at drawn κ. So the local search runs up to three scans: the weighted key's from the
trim's route, the plan key's from the best plan met, and the plan key's from the trim's route when
that differs. They share the per-class bound, and the class's best is never below `weighted`'s under
the plan key (0 of the same 600 and 327 after the fix). The cost: no extra walks on the 100 m Exp 4
field, and 32 % to 56 % more on 300 m and 500 m fields within the same bound. Two consequences are
recorded. Missions can be longer, since time only breaks ties: layout 18's fourth mission is
predicted at 182 s where `weighted`'s plan took 118 s, and over 30 layouts × 6 missions at 45 s with
the cap off F's plans differ from `weighted`'s in 3 of 180 missions (a mean predicted mission of
119.0 s against 118.0 s) and FB+wide's in 14 of 180 (159.2 s against 153.5 s), where `weighted` flew
7 empty missions and `lexicographic` none. And the share is nominal: it counts a served member
fully, though on the pilots' jittery channel its outage at a class's edge is about 0.15–0.2. Layout
18's crowding at S\* falls from 3 to 2 under it (R13).

**The hover decision** (the user, 2026-09-30, after the final check's PLAN-1 and E2E2-01, below). In
plan mode a capped device (age ≥ S − L) that its own S3a stop cannot serve alone within the budget
(U1's `servable_alone`, from the dock at takeoff) leaves that stop for a one-device stop at its best
hover point on the class (`plan/hover.py`). That point p lies on the segment from the dock to the
device and minimises the device's alone mission, transit dock → p, its dwell at |device − p| at the
class's predicted rate, return p → dock and the Pass-1 upload, among the points within the class's
planar reach of the device where its predicted dwell is finite. It must be within reach by both
distance formulas in the code: the model's `** 0.5`, which decides whether the dwell is charged, and
the `math.sqrt` of S3a and of the contact gate, which decides whether the mule solicits the device.
It is found deterministically, independent of the clock, the budget and the partition: a 64-cell
grid, then at most 64 rounds that halve at most 8 cells each, while a cell's bound can still beat
the best point by more than 1e-9 s, a tie going to the point nearer the device. Because the dwell
never falls with the distance on the contact link, no point of the plane serves the device alone
sooner. The stop it leaves keeps its position and takes its remaining members' bucket and B2
deadline, and an emptied stop goes. Uncapped devices, and capped devices that their S3a stop serves
alone, keep S3a's stops; without a budget, or with the cap off, nothing moves; Pass 2 is still
priced and flown on S3a's stops. `unplannable` now means that no class the arm may fly serves the
device alone within the time budget even at its best hover point. The S\* tool uses the same stop
family, and its S + 1 claim is narrowed to what was measured, with caveats it prints (Configuration
Reference §18.7).

*Under an energy capacity* the point still minimises time. The energy clause weighs a second of
hovering (168.5 W) more than a second of flight (143.6 W), so a slightly slower point with a shorter
dwell can need less energy, and the label can then read `unplannable` while such a point would serve
the device: `unplannable` is physics for the time budget only. With no time budget binding, 34 of
the 1,080 device-class pairs of the critic's and the S\* tool's 30 layouts have such a point at 1 MB
(2 at 64 kB, 625 at 8 MB). No pilot sets a capacity; an energy-aware point is the user's call (open
items).

**What landed, by unit** (Configuration Reference §18 has each setting):

- *Goldens at `6e6f92d` (unit UG4).* `tests/golden/_build_p3_sim.py`, `data/p3_sim.json` and
  `test_golden_p3_sim.py`, captured on the untouched tree: eight stub trials of Phase 3's simulated
  clock through `Exp4Driver.run_trial`, with the real cluster, mule and device services run in
  process. They are H1 with the pilots' flags (wide, the `t_nom` unit, `replan` with `trim`,
  `agg:cutoff`, the `channel` source, 1 MB, 60 s, the seconds backhaul), D1, D3 and route-only D4 on
  its layout, the narrow cliff flown empty (T2 at 60 s) and whole (99.5 s), H1 on medium, and H1 on
  narrow at the measured payload, so contacts are flown on all three classes at both payload modes.
  Each trial's row, per-role JSON and every mule, cluster and device event is compared on its
  `6e6f92d` keys: an added key passes, a removed or renamed one fails, and values and event
  sequences must match exactly. In process the device-serve columns are harness values (`coverage`
  and `participation_entropy` 0, `jains_fairness` 1.0), so the consumer's serve fold is pinned by
  `tests/unit/test_exp4_metrics.py` instead. Every trial flies the `t_nom` deadline unit, and D5 and
  K = 2 are not among them; random-instance tests against `6e6f92d`'s modules cover D5. The goldens
  are now 228 tests: the 144 afa9526 ones and the 84 of `test_golden_p3_sim.py`. A second pass/fail
  baseline was recorded at `6e6f92d` (below).
- *Plan types (U0).* `hermes/scheduler/plan/__init__.py`, which re-exports the types only, and
  `plan/types.py`, every type that crosses a unit boundary (critic B13): the switch values,
  `AgeCapSpec`, `CapState`, `PlanScoreParams`, `PlanSearchParams`, `PlanOptions`, `PlanSetup`,
  `PlanClass`, `ScoreTerms`, `MemberFold`, `Candidate`, `SearchResult` and `ArrivalView`. Additively
  in `hermes/types/scheduler.py`: `PlanCommit`, `CapViolation`, the cap reasons, and the band-class
  policy and search modes, which the plan package re-exports. The commit is frozen and checked: it
  refuses a `fixed:<class>` policy on another class (FB+c flies only class c), an unknown search
  mode and a cap without its round, it holds no wall time (critic B12), and `close` returns a
  checked copy. Unknown score or search settings are refused, and `fixed:<class>` pairs with the
  `committed` slot only.
- *The age cap (U1).* `stages/s3d_age_cap.py`: the age m − (`last_merged_round` or 0), with no clamp
  to 1 and no fallback to `last_clean_round`; exempt, mixed and priority stops, computed on the
  route actually folded (critic B1); B2's deadline; the cap key (critic C5); `servable_alone`; the
  plan-time and close-time violations.
- *The plan score (U2).* `plan/plan_score.py`: V, the coverage weights (the age floored at 1 in the
  weight only), the mean-SNR outage, `predicted_mission_s`, and R11's `served_share`, `applied_rank`
  and `plan_key`. V of the empty plan is −c₁(t_turn/T)² − c₂. With Δ and E held and c₃ ≤ c₂, serving
  one more device never lowers V, and it raises V when c₂ > 0 unless c₃ = c₂ and the device's outage
  is 1: the plan's "V falls as coverage falls" (L842) as the tests state it.
- *Member subsets for the F family (U3).* `plan/member_subset.py`: `reduce_stop`, the F member order
  (capped first, then weight per second of predicted dwell, then id), `admit_members` (the one
  member walk, skip not stop, with an injectable order), `admit_stop`, `fold_members`, and
  `trim_members`, the in-flight trim, which flies the priority stops first and reserves their capped
  members before the rest, so a capped member is dropped only when the protected-only trim cannot
  hold it. Additively in `s3b_feasibility.py`: the `member_admission` values and
  `FeasibilityResult.dropped_plan`, appended last, so the positional construction still works and
  `REASONS` is unchanged. On the Phase 3 cliff instance (T2: `device_positions(8, 777, 100.0)`,
  narrow, 1 MB) F admits 5 devices at 60 s (home 54.490 s) and 7 at 99.0 s (home 81.903 s), where
  S3b admits none; `test_p3_final_fixes_mule.py` keeps its pins as the `whole` side.
- *Member subsets for the H and D arms (U3b).* `filter_feasible(member_subsets=...)`, with
  `MemberSubsets` and `fold_subsets`, in S3b; `greedy_budget_walk(member_subsets=...)` and
  `left_out` in `policies/budget_walk.py`; `fedcs_greedy_select(member_subsets=...)` in
  `fedcs_degraded.py` (its deviation 11); the keyword forwarded by `max_aoi.py`, `oort.py` and
  `whittle.py` (Whittle's deviation 9). The whole-scheduler contract (§5d) gains an optional
  `member_subsets=None` on `admit_and_order` and the class flag `admits_member_subsets = True`,
  which MAX-AoI, Oort, Whittle and FedCS declare and FedEx does not. Member orders: H1–H3 the
  member's own deadline, then its Pass-1 dwell at the SNR offset, then id, with the miss streak
  first under miss priority; D1–D3 the arm's own contact key on the one-member contact (the age,
  Oort's utility, the Whittle index), then id; D5 Algorithm 3's selection key on the one-member
  pick, then id. This is the fidelity argument of critic B5: the published methods select devices,
  not stops. Only the pre-flight call passes the carrier, so the in-flight check, the re-plan and
  the trim never create reduced stops for these arms, and the budget walk refuses the carrier in
  Pass 2. One interaction under `subset`: inside the pre-flight order check, S3b's re-admission
  re-sorts the kept stops by their own deadlines, and a reduced stop's deadline can be later than
  its original's. In a random probe it dropped a stop in 17 of the 1,231 checks that fired (0 under
  `whole`); such drops join `last_feasibility` and are widened. (The `trim` fallback's own drops in
  that check are Phase 3 behaviour under either admission.)
- *The search (U4).* `plan/plan_search.py`, with an independent brute force
  (`tests/unit/_p4_brute.py`: itertools over class × device subset × orders, priced by a plain
  `FeasibilityModel.fold`). Exact up to 6 devices; above that, the depth-first stop-subset family or
  the local search. The walk bound buys determinism, not a time bound: at N = 6 a whole plan took at
  most about 0.25 s (no budget, a synthetic worst case) and about 20 ms under 30–120 s budgets, but
  at N = 96 on a 500 m field at 1 MB with no budget every class reaches 2,000 walks and one plan
  took 3.1–3.4 s. The scan order and the skip of neighbours already walked are part of the contract,
  and the search's memo key includes the deadlines on board (`deliver_by`).
- *The scheduler fork (U5).* In `fl_scheduler.py`: `FLScheduler(member_admission=, plan_mode=,
  plan=)` and its refusals; `build_ferry_plan`; the spec's item 8 drop labels; the plan-mode re-plan
  (U3's member trim under `subset`, `_trim_whole` under `whole`), which dates the plan's own members
  by the plan and a beacon insert's by the mule's record; `plan_protected`; `close_plan`; and
  `last_policy_drops`, `last_plan` and `last_plan_wall_s`. Inside the frozen `build_contact_queue`
  three things change: `last_policy_drops` is reset with the other diagnostics, so an early return
  cannot re-emit a stale report (critic A9); the member-subset carrier goes to the two pre-flight
  admission calls under `subset`; and the D arms' report is set on the simulated clock, with
  `last_feasibility` left None for them. S1 and S3 are repeated in `build_ferry_plan`, not
  refactored out of the frozen method (a test pins equal deadlines and buckets), and the
  source-order pins hold. On 2,000 random instances the scheduler at its defaults equals
  `6e6f92d`'s, loaded from git.
- *The runtime and FX (U6).* In `mule/ferry.py`: `FerryRuntime.band` and `set_band`, an optional
  `band=` on every method that reads a band, per-class physics bound at construction (critic B3),
  `outage_probability`, `plan_classes` and `arrival_view`; `contact_plan` refuses a Pass-2 plan on
  any class but b̄. `policies/cross_heuristic.py` holds the slot's two fillings (§5g);
  `policies/__init__.py` does not import it, since it loads the plan package. At its defaults the
  runtime equals `6e6f92d`'s bit for bit on 27 cases (9 configurations × 3 seeds).
- *The supervisor (U7).* In `mule_main.py`: `MuleSupervisor(member_admission=, plan_mode=,
  plan_options=, t_nom_s=)` and its refusals; per mission `set_mission_round`, `build_ferry_plan`,
  `set_band(b̄)`, then the annotations; the `plan` drops widened and recorded after the other four
  reasons and left out of S3c's planned count (critic C6); the D arms' report; `close_plan` right
  after `record_merged`, with `merged = ()` on the empty path. In flight, Pass 1 only: the exempt
  set, recomputed at each departure, goes to the departure check, abort's head fold, the re-plan,
  the beacon hook's fold and FX's `fits`, and the re-plan gets the mule's deadlines; the order is
  the departure check, then the beacon hook, then the slot; a capped member never lowers
  `deliver_by`, in a mixed stop too; the beacon hook groups an offer within b̄'s range; and under
  `agg:cutoff` `not_merged` follows the merge's exclusions, not the round report. On critic B4's
  deterministic loopback (the link's σ kept, every noise term 0, availability 1): for arm F, no
  violation at S\*+1 on any of the 30 layouts at 45 and 60 s (the test flies F only; the pinned
  classes can still crowd at 45 s, below); F's plan key never above any FB+c's (critic A3); FX
  never lands later than F after the last Pass-1 stop; and a K = 2 plan-mode loopback.
- *Processes, driver and runner (U8).* The eight `MuleConfig` plan fields, `PLAN_MULE_FIELDS`,
  inside `SIM_ONLY_MULE_FIELDS` and outside `FERRY_SPEC_FIELDS`, so Phase 3's `ferry_params` are
  unchanged, with their guards in `mule_config_errors`; the plan wiring and the new `mule_ready` and
  `mission_completed` fields in `processes/mule.py`; the plan fields as explicit topology
  parameters, copied to every mule; in the driver `DEFAULT_ARMS`, `PLAN_ARMS`, each arm's band,
  member admission and `miss_priority`, `check_arm`, T_nom computed for every plan arm (per cell,
  unless `--t-nom-s` gives it), and the provenance rule (plan keys in `ferry_params` in plan mode
  only, `contact_band` reading `search` for the search arms); the runner's flags.
- *Analysis and the S\* tool (U9).* `MissionRecord` gains the plan and cap fields;
  `traces_scorer.py` gains one age walk shared by every age figure, the nine Phase 4 columns,
  `--age-cap-s` and the driver's provenance rule; `experiments/analysis/age_cap_s_star.py` is new.
  All 600 kept traces score as `6e6f92d`'s scorer scores them, with the nine columns blank (0 rows
  differ). The allowed-added set at `tests/unit/test_multi_mule_scoring.py:1084-1098` is edited on
  purpose.
- *Coverage first (R11) and the hover stops*, above.

**Visible at the defaults, additive only.** Every mule's per-role JSON, on either clock, carries
the eight plan fields at their defaults. A D-arm mission on the simulated clock that leaves a
contact out before takeoff gains `pass_1_policy_drops` (decision 6), the one trace-event field;
Phase 3's simulated D-arm traces have none, and UG4's compare reports it as an added key. Nothing
else: no plan field reaches `mule_ready` or `mission_completed` at the defaults (tests pin their
absence on H1, D4 and wall-clock missions), Phase 3 rows keep their `ferry_params`, `contact_band`
and `miss_priority` strings, and the trial CSV header is unchanged. The scorer's CSV gains nine
columns, blank on every older trace at the defaults. `FeasibilityResult.dropped_plan` is empty
outside plan mode, and `FerryRuntime.band` never moves there.

**How legacy identity was checked.** For every unit: the 228 goldens before and after, and UG4's
P3-sim compare (`8 trials x 8 parts compared (same)`); `6e6f92d`'s modules loaded from git beside
the live ones (S3b, the walks and the five D policies on 1,200 random and 180 band-priced instances,
the scheduler on 2,000, the runtime on 27 cases); fresh-interpreter tests that no path at the
defaults loads `hermes.scheduler.plan`, `s3d_age_cap` or `cross_heuristic` (an H or D walk loads
the member walk only under `subset`); and the 600 kept traces re-scored. The final check's legacy
dimension widened the in-process differential to 74 trials and found no change.

**Test baselines and the compare rule.** `tests/golden/make_baseline.py` now holds two pass/fail
baselines (`BASES`): `pytest_baseline.txt`, the full suite at afa9526 before any Phase 3 change
(1,560 tests, 6 failures; signed off by the user on 2026-09-29), and `pytest_baseline_6e6f92d.txt`,
the full suite at `6e6f92d` before any Phase 4 code change, with UG4's goldens added (2,918 tests,
the 84 of `test_golden_p3_sim.py` among them: 2,913 passed and the five deterministic afa9526
failures with their signatures; unit UG4), which gates Phase 3's own tests too. "The full suite
passes" for Phase 4 means: the same as both baselines, that is the same outcome per node id and
the same signature per known failure, new tests allowed. `compare` checks every
recorded baseline by default (`--base` picks one, `--baseline` names a file). It exits 0 when the
run is the same as each, 1 when it differs from one, and 2 when a baseline file it was to compare
with is missing, printing `MISSING` and still comparing the other; the worst status wins, so a
baseline left out of a commit cannot switch its gate off in silence. It also lists the new tests
that fail, which are never differences, so read that line. A flaky test (`FLAKY`) may pass, or fail
its known way, from run to run. Only `test_exp4_real_model_synthetic_converges` is listed: its
`rounds_closed == 0` failure under load flipped the afa9526 comparison to DIFFERS on its own. A
change of it is reported as allowed; any other failure of it, an error or not running still differs.
`--flaky NODE_ID` adds a test, and `--strict` drops the list (the old rule). `write --base 6e6f92d`
refuses unless HEAD is `6e6f92d` with `hermes/` and `experiments/` clean.

```
py -3.11 -m pytest tests -p no:cacheprovider -q -rfE --junitxml=run.xml
py -3.11 tests/golden/make_baseline.py compare run.xml                   # both baselines
py -3.11 tests/golden/make_baseline.py compare run.xml --base 6e6f92d    # one of them
```

Phase 4 adds 1,862 tests in 15 new test files (`tests/unit/test_p4_*.py`,
`tests/integration/test_p4_*.py` and `tests/golden/test_golden_p3_sim.py`), UG4's 84 goldens among
them, with three helper modules: `tests/unit/_p4_brute.py`, the search's independent brute force,
`_p4_ref.py`, which loads `6e6f92d`'s modules from git, and UG4's `tests/golden/_build_p3_sim.py`.
The `6e6f92d` baseline already records the 84 goldens, so a compare against it lists 84 fewer new
tests than the 15 files hold. The full run of the Phase 4 tree (2026-09-30: 4,696 tests, 4,690
passed and the six afa9526 failures) is the same as both baselines (exit 0): against afa9526 with
3,136 new tests and against `6e6f92d` with 1,778, none failing. The real-model smoke test, which
passed in the `6e6f92d` baseline, failed its known way, a change the flaky allow-list allows.

**Final cross-cutting check (2026-09-30).** Six review dimensions: spec fidelity; the plan pipeline
across units; legacy identity and the H and D paths; the driver, configuration and analysis; an
end-to-end replay of the plan arms with one mule (8 real trials, 40 missions, 35 in plan mode, and
285 missions in process); and one of plan mode at K = 2, the D arms and the S\* tool. Four found no
defect. The two findings, both confirmed, share one cause:

- *PLAN-1 (high).* S3a re-clusters the demand every mission on S3's deadlines. A device the plan
  leaves out is widened, so its deadline recedes and S3a anchors it last, and once it lies far from
  the rest it becomes a stop of its own at its own position, from which no plan serves it within the
  budget. The cap then logged it `unplannable`, which reads as physics, dropped it as `budget` and
  widened it again, so it starved, often for good. On critic B4's loopback, FB+medium at 45 s and
  1 MB, at S = S\*(medium) + 1 = 3, left d5, d3 and d1 of critic layouts 0, 4 and 26 `unplannable`
  from mission 5, 5 and 4 to mission 16 (ages up to 15), though each fits alone from a reachable
  point (d5 and d3 hovering at the dock, in 13.25 s). In the pilots' configuration, built by the
  driver (1 MB, S = 3), FB+medium had `unplannable` violations in 8 of 30 runs at 45 s with the
  critic's seeds and 13 of 30 with runner-style seeds, FB+wide in 13 and 14, FB+narrow in 0 and 1,
  and F, FX and F−prio in none; at 30 s F and FX in 11 of 30 each and FB+medium in 26 of 30, with 21
  empty plans; at 90 s none. Every such event named a device that some reachable point served alone
  within the budget.
- *E2E2-01 (medium).* The S\* tool priced S\* on a fresh, all-new S3a partition, from which the
  running missions drift, so its documented S\*+1 guarantee failed for the pinned classes at the
  stress budget. On its own layouts at 45 s, FB+medium (layouts 0 and 27) had 12 violations at S\*+1
  on devices the tool calls servable (9 `unplannable`, 3 `crowded`), and FB+wide (layout 29) one
  `crowded`; F and FB+narrow were clean, and every family at 60 and 90 s.

The hover decision (above) fixes both. After it, on the S\* tool's 30 layouts at 1 MB, each flown at
its own S\*+1 for 3S missions on B4's loopback (either deadline unit): at 45 s F has no violation in
276 missions (8 `unplannable` before), FB+wide 2 `crowded` in 360 (89 `unplannable` and 1
`crowded`), FB+medium 7 `crowded` in 291 with an oldest age of 6 (36 `unplannable` and 4 `crowded`,
oldest 15) and FB+narrow none in 327 (8 `unplannable`); 60 and 90 s are clean; at 30 s every family
still crowds (F 20 in 309; FB+wide 45 `crowded`, and 29 `unplannable` on its 3 devices that no point
serves alone within 30 s, with 5 empty plans; FB+medium 38 and 18 empty plans; FB+narrow 22 and 6).
In the pilots' configuration, re-run, `unplannable` fell to 0 at 45 s (FB+medium, FB+wide) and at
30 s (F, FX, FB+medium). The fix's review raised three low findings, all addressed: under an energy
capacity `unplannable` is physics for the time budget only (documented above); the hover point's
refinement constants are pinned against an exact minimum, to 1e-6 s at 1 MB, at 8 MB, and at 2 MB
with a 100 m range; and the S\* tool's (ii) lines print the caveats measured.

Other observations of the check, none a defect: the plan arms ignore `--miss-priority` and refuse
`reorder` under `abort` too (as built); with L > 0 the scorer's pair count is below the mule's log;
a trace records no position for a committed stop that was never flown (`PlanCommit.describe()`
leaves the queue out, and `pass_1_plan` holds devices and deadlines only), which limits a
trace-level check of "the predicate holds at every departure"; the S\* tool has no mule-count
option, which fits the one-mule pilots; `experiments/exp4/cost_matrix.py` knows no plan arm, so the
pilots' re-costing cannot use it as it is; and F−prio crowded twice at S\*+1 in 230 of the check's
runs, the documented myopia of a planner that looks one mission ahead.

**Frozen surface touched,** all behind the switches: `fl_scheduler.py` (plan mode; in
`build_contact_queue` only the reset, the carrier and the report above); `stages/s3b_feasibility.py`
(additive: the `member_admission` values, `dropped_plan`, the opt-in member-subset walk);
`mule_main.py` (plan mode, the flight slot, the exempt set in flight, the plan's close, the drop
report). `s3a_cluster.py` is untouched: it already takes the radius, and the plan calls it once per
class, so the §5g file list is corrected. `s1_eligibility.py`, `s3_deadline.py`,
`s3c_mission_window.py`, `s35_selector.py` and `selector/` are untouched. Outside the frozen
surface: `mule/ferry.py`, `types/scheduler.py`, the five D-arm policy files, `processes/config.py`
and `processes/mule.py`, the Exp 4 `driver.py`, `runner_main.py`, `topology_builder.py` and
`events_consumer.py`, and `analysis/traces_scorer.py`. New: `scheduler/plan/` (`__init__.py`,
`types.py`, `plan_score.py`, `member_subset.py`, `plan_search.py`, `hover.py`),
`stages/s3d_age_cap.py`, `policies/cross_heuristic.py` and `experiments/analysis/age_cap_s_star.py`.
Tests: `tests/golden/` (UG4's files, `make_baseline.py`, `README.md`), the new test files, the cliff
docstring in `test_p3_final_fixes_mule.py` and the allowed-added set in
`test_multi_mule_scoring.py`.

**Deviations from the build plan** (also in `FeRRy_Build_Plan.html`, Phase 4):

- the plan inconsistencies the Phase 4 design lists (its §0.4, items 1–24) are resolved as it
  proposes. On the plan page itself: Figure 2 cites FedEx-Async's Theorem 2 (eq. 24), not "Thm 1";
  the Energy metric row states the SIMULATED Zeng–Xu–Zhang energy that the clock's ledger gives; the
  optimality gap is V_O1 − V_arm ≥ 0, since V ≤ 0; and Chen et al. (TVT 2025), which the theory
  track cites, joins Sources;
- FX's flight slot is built in Phase 4, not Phase 5 (the exit gate flies it, L845), and Phase 4's F
  is the plan search with the `committed` slot: the learned pair Q stays in Phase 5;
- FX's band rule is the fastest class that still reaches every target of b̄, at the arrival SNR
  (critic A7), not the design's "most targets", which added up to 94 s of dwell and overran the
  budget at the last stop, where nothing re-checks; FX acts in Pass 1 only, and picks each next stop
  after serving a stop, not at takeoff;
- Δ is the whole mission on b̄ against T_nom (decision 2 (b)), not Pass 1 against the budget (L533,
  L899);
- the coverage weights are age × (1 + miss streak), not the letter's 1 − served/N, and F−cov is
  reported as cap-only service (decision 3); the age counts the device's own mule's missions, not
  cluster rounds (L531; decision 1);
- the plan ranks the served weight share before V by default (coverage first, R11);
- a capped device may be served from a hover stop, not only from S3a's stops at R(b̄) (the hover
  decision);
- member subsets also for H1–H3, D1–D3 and D5, opt-in and before takeoff only (decision 4 (b)),
  which brings those baselines closer to the published methods, which select devices, not stops
  (critic B5);
- no `reorder` in the plan-mode re-plan (L541; critic B11): it is a trim of the committed plan,
  priority stops first, and re-ordering belongs to the flight slot. The plan arms fly the `trim`
  fallback under either in-flight response, which is stricter than the Phase 4 spec's "under
  `replan`";
- `pass_2_budget` is refused in plan mode (critic B8), and so is `abort` together with a cap (critic
  A10);
- the search is exact only up to 6 devices; above that its two families are heuristics, and the
  local search can keep a younger capped stop where two compete for one slot (R7);
- each plan arm sets its own `miss_priority` (on, and off for F−prio) and ignores `--miss-priority`,
  and plan settings given outside plan mode are refused rather than ignored;
- F·round and F·pref (L705) are deferred to Study 5.2 (unit U11), and the 5.4 sweep knobs (unit U10)
  to before the 5.4 headline;
- size: about 9,000 lines of code (8,975 added and 120 removed, over 26 files) and 18,800 of tests
  (18,789 added and 51 removed), docstrings and the golden harness included, plus the `6e6f92d`
  baseline record (2,940 lines) and the `p3_sim.json` fixture (370 KiB), against the ~800 estimated
  in the plan; the Phase 4 spec had estimated about 3,000 and 3,500.

**Open items.**

- *Where the hover point sits.* It minimises time, so it is often the dock or the class's reach
  edge, where the noisy link is weakest: in the pilots' configuration, re-run with the fix,
  FB+medium's `not_merged` rose from 228 to 245 and F's `dropped_in_flight` at 30 s from 0 to 12. A
  margin inside the reach, or an outage weight, is a pilot-time choice.
- *Empty missions before the cap binds.* Uncapped devices keep S3a's stops, so a mission can still
  fly empty while nothing is capped (at 30 s on the S\* tool's layouts: 5, 18 and 6 empty plans for
  FB+wide, FB+medium and FB+narrow, none with a capped device).
- *Pass 2* still delivers at S3a's stops, so a far device is still visited at its own position.
- *Partition-drift crowding* remains at S\*+1 for the pinned classes at 45 s and for every family at
  30 s (above): a capped device's S3a stop can serve it alone, but not beside another capped device.
- *F-whole* (F under `--member-admission whole`) can fly empty at 30 s while a capped device fits
  alone at its hover point: whole admission flies the device's whole stop or none.
- *An energy-aware hover point* (above): under a capacity the point still minimises time, and the
  label reflects that point.
- *The planning-time bound.* "At most 1 s per mission" holds for the N = 6 pilots (`plan_wall_s`);
  the 2,000-walk count bounds no time at N = 96 on large fields (several seconds).
- *U10* (Study 5.4's sweep knobs) before the 5.4 headline; *U11* (F·round, F·pref) with Study 5.2.
- *The pilots* (decision 7; Run Guide §2.7) wait for the user's go-ahead, after the Phase 3 pilot
  has set the session TTL and the knee: about 1,400 stub trials, to be re-costed before the
  go-ahead.

## 5l. Phase 5 behind switches, no amendment (2026-10-02)

Lands with FeRRy Phase 5 (build plan, Phase 5, "Flight clock: the (band, stop) score"; commit
`9694775`, on `386c275`). It fills Phase 4's flight slot with a learned (band, next stop)
score (the FQ arms), builds FerrySim to train it, builds E3 to compete with it, and stops before any
campaign. It is not an amendment (Rule 3): nothing changes the wall clock, Phase 3's simulated clock
or Phase 4's plan arms at the defaults. Every mechanism sits behind a switch of the §5g table whose
default is the recorded pipeline, in behaviour and in trace output, or is additive: no trace event
gains a field at the defaults, and every mule's per-role JSON gains the six checkpoint keys at null
(below). "Legacy" now has three faces, and all three are pinned: the wall clock by the 144 afa9526
goldens (§5j), Phase 3's simulated clock with `plan_mode = legacy` by UG4's oracles at `6e6f92d`
(§5k), and Phase 4's plan arms by oracles captured at `386c275` before any Phase 5 edit (unit UG5,
below).

**The user's ten decisions** (2026-10-01), each the recommended option:

1. *What the score may choose: within the plan.* The band reaches every device the planned band b̄
   reaches at the stop; the next stop is still a stop of the plan (home only once none is left);
   the rest of the flight still fits the time and energy budget, with this stop priced at the
   observed SNR. If no pair fits, the mule flies FX's pair and logs it.
2. *Where the score practises: FerrySim*, the real system run in one process, with a stand-in for
   the devices' local training, in `experiments/ferrysim/`.
3. *The cells:* 1 MB and the jittery contact channel, at N = 6 (the control, where looking ahead
   cannot matter) and N = 12 (decision-rich), plus a clean-channel N = 12 negative control; one
   score per channel regime; stand-in budgets of 120 s and 180 s at N = 12 until a pilot measures
   its knee.
4. *The reward:* per stop, the merge weight of the updates collected there, minus c_t = 0.1 per
   nominal mission period; at the end of the flight, c_cov = 1 times the weighted share of planned
   devices left uncollected; no energy or lateness term. F·hand is today's reward, ported and
   labelled as such. Study 5.7's grid: c_t ∈ {0.03, 0.1, 0.3} × c_cov ∈ {0.25, 1, 4}.
5. *Study 5.5's rule, fixed in advance:* read on N = 12; 10 training seeds per γ ∈ {0, 0.25, 0.5,
   0.75, 0.9, 0.99}, judged on 1,000 shared held-out runs; "rising", "flat" (TOST) or
   "inconclusive", with Holm's correction. FX is replaced only if the curve is rising and beats the
   best fixed rule by ε, with ε = max(0.01, 0.1 × the validation headroom). A γ = 0 sanity check
   comes first, with one learner revision at most.
6. *Study 5.6:* keep the periodic interference, add a two-reading phase feature, and compare two
   N = 12 cells (stop spacing a quarter and a half of the period) with a clean control; irregular
   interference is built later, only if 5.5 keeps the learned score.
7. *E3, M1, O1:* E3 is a numpy port of Chen's recipe, trained in FerrySim, with its declared
   deviations; M1 only if 5.5 keeps a learned score; O1 before the 5.4 headline.
8. *H2, H3, ChannelDDQN:* H2 and H3 leave Exp 5; `H1+L1` (H1's scheduler with H3's adaptive
   backhaul) is added; an opt-in `--require-trained`; ChannelDDQN is retired from the plan, its code
   and logging kept.
9. *Checkpoints and licences:* each study's final checkpoints are committed with their manifests
   under `results/exp5/checkpoints/`, each commit only with the user's consent and after a LICENSE
   file is added (the README names MIT; no LICENSE file exists yet). Nothing is committed during the
   build.
10. *The headroom check and the exit gate:* the headroom report runs during the build, on FerrySim's
    validation stream, and the build pauses only if the best possible gain is below 0.01 per
    practice run in every cell; the exit gate is split into a code gate now (every test, all three
    baselines the same, FerrySim's parity with real processes) and a campaign gate after the
    go-ahead.

**Resolutions taken during the build** (the orchestrator's, binding):

- *R1.* A sixth deliberate Phase 4 test edit: the per-role JSON gains the six checkpoint keys
  (the spec says so), while `tests/unit/test_p4_config_driver.py:916` pinned its added keys to the
  plan fields alone, so 8 nodes failed. Lines 896–897 and 916–918 now expect `PLAN_MULE_FIELDS +
  CHECKPOINT_MULE_FIELDS`; node ids and line numbers are unchanged.
- *R2.* The checkpoint header binds `purpose` and `learner_revision`, so the sha covers them: the
  runner refuses anything but a trained checkpoint and the report checks that one revision trained
  a sweep, so neither may be relabelled. The other manifest fields stay informational.
- *R3.* `flight_slot = pair_q` needs `in_flight_response = replan`, guarded in `mule_config_errors`
  and in the driver: the mask folds the whole rest of the flight, which only the re-plan's
  departure check folds next; under `abort` that check folds the next stop alone, so the mask would
  be stricter than the check. A cap already needs `replan`, and the Phase 4 pilots fly it.
- *R4.* Accepted as built: `energy_ref_j` and `StopPrice` are kept beside the spec's three runtime
  readers; a served stop's own lateness is not masked under `collection` and `delivery_per_stop`;
  under route-level `delivery` the mask lowers `deliver_by` conservatively, by every uncapped target
  of b (the mule lowers it by the CLEAN ones).
- *R5.* Ties go to the lowest row (class-major, then the remainder's order), so with one class they
  follow the remainder's order, not `rank_contacts`' position-then-devices key.
- *R6.* Study 5.5's "FX" is the FX arm itself; `fx_pair`, the slot's version of FX's rule, is the
  warm-start reference.
- *R7.* Accepted as built: the record schema's additions (`admitted_pairs`, `scorer`, `terminal`,
  `trimmed_next`), the optional `q_values` attribute, `pair_slot` importing `pair_q` at module level
  (plan mode only) and the scope guard checking the served stop's members too.
- *R8.* U1 builds `build_pair_slot`, which needs U1's adapter; U8a builds the oracle's replay
  scorer.
- *R9.* Accepted as built: the empty round's last decision ends at the landing, read literally
  (`landing_s`); the trainer's sink takes `close_mission(((), ()))`; the two N definitions stand
  (the pair view's: the plan's demand with any beacon insert; E3's: the slice with every planned or
  inserted device); under `agg:plain` w = n_i, which is what the merge pays.
- *R10.* The held-out score's keys stay as U2 named them (`HELD_OUT_KEYS`: `episodes` and
  `return_mean`).
- *R11.* `pair_v1` keeps three columns beyond the spec's list, giving 36 columns with the phase
  block (24 without): `reach_here`, the absolute collection at this stop, because the reward's G_k
  is an absolute count and a γ > 0 target bootstraps from the row's own reward level; and
  `prev_sin` and `prev_cos`, the previous reading's phase, which survives the 4·P_c cap on the raw
  age. Recorded as feature deviations; the schema was fixed before any checkpoint was saved.
- *R12.* A relative checkpoint path is read under the repository root, and the driver writes a path
  outside the repository absolute, not "as given": the mule reads every relative path under that
  root, so a relative path from outside would name another file. The `MuleConfig` comment was
  corrected (two comment lines, no code).
- *R13.* The deliberate `:583` edit includes its import on the existing line 60 of
  `test_p4_config_driver.py`; no line moved, and it counts as part of the `:583` edit, so there are
  still six deliberate edits.
- *R14.* A pre-existing driver leak, reported and not fixed: when a mule fails at startup,
  `_run_topology` re-raises from `start_all` without `shutdown_all`, so the cluster process keeps
  running. Fixing it changes every arm's error path (Rule 3), so it is a candidate amendment for the
  user (open items).
- *R15.* Accepted as built: the runner checks every checkpoint given, even for arms it will not run,
  and `--require-trained` checks only for `--selector-weights`, as Exp 3's `--require-trained-a4`
  does.
- *R16.* E3's training defaults stay the pair learner's (lr 1e-3; ε from 0.3 to 0.05), and
  `--gamma` is required. Chen's values (lr 5e-4, ε from 1.0, Chen's γ) are the user's choice at the
  E3 training go-ahead; the manifest records whichever is used.
- *R17.* The best fixed rule is chosen among decision 5's four (the FX and F arms, `hyb` and
  `greedy_1`); `fx_pair` and `committed_pair` are reported beside them, not added.
- *R18.* Accepted as built: `greedy_1`'s flag applies the claim rule with the held-out episode as
  the unit; the sanity check is a point comparison; the γ pick reuses the validation entries,
  identically for every γ; `evaluate` reads every policy under one reward (`--reward bytes` for
  E3's own); `evaluate` keeps no traces.
- *R19.* A checkpoint of another cell family is never overwritten, even with `--overwrite`; one
  study per family; the layout is unchanged.
- *R20.* The N = 6 control stays in `evaluate`'s default cells, so Study 5.5's held-out evaluation
  takes about 3.3 h at 8 workers; `--cells` narrows it.
- *R21* (its first bullet amended on 2026-10-02 after the final check's SPEC-1). The agreement and
  re-order shares count choices. They are reported diagnostics only: no Study 5.5 step reads them,
  since critic A2 replaced the design's FX-agreement gate with the one-step sanity check, and
  FerrySim's held-out evaluation keeps no traces. Counting choices stays because the choice is the
  slot's unit of decision. `pair_fx_agree_share` includes empty-mask decisions, documented, with
  the count beside it; `pair_mask_empty` is a count; `e3_unvisited_mean` counts stops, not devices,
  averaged over all missions; the trend test is Page's L with the seeds as blocks; `mule_ready.pair`
  is not parsed, and a record whose choice cannot be read is skipped.
- *R22.* Study 5.6's cells (the final check's SPEC-2: decision 6 (a)'s code change was "the
  two-reading features and the cells", and only the features had been built). N = 12, jittery,
  1 MB, at each stand-in budget, with P_c = 4 × the lag (the quarter cell) and 2 × the lag (the
  half cell), the lag being the arrival-to-arrival time: FX's median Pass-1 lag on FerrySim's
  validation stream at the matching Study 5.5 cell at the default P_c, rounded to the nearest
  second and pinned as constants (26 s and 34 s, so P_c = 104 and 52 s at 120 s, 136 and 68 s at
  180 s). The control is the clean N = 12 cells. A second family, `jittery56`, adds the four cells
  to the jittery cells, and `jittery` is unchanged; whether the jittery score practises at the 5.6
  periods is the user's choice before the 5.5 sweep, since it changes the family's hash. Study
  5.5's verdict reads only jit-n12-120 and jit-n12-180, whichever family trained the score. The
  cells are re-pinned with the budgets after the N = 12 pilot. The comment at
  `tests/unit/test_p5_pair_features.py:808-813` is corrected: at N = 12, 30 s is the ratio-1
  aliasing point, not a 5.6 period.
- *R23.* Plan-trained ablation checkpoints (the final check's F1): critic C2 has FQ-dwell and FQ-cov
  train under their own plans, which the trainer could not do, and the runner would have flown a
  default-plan checkpoint on those arms unchecked. Built: `TrainSpec.plan_score_params`, checked
  against `PlanScoreParams` and recorded in the manifest's training spec; training and validation
  fly it as a driver override; `checkpoint_flight` flies a checkpoint on its recorded plan, so
  `evaluate --record` scores it there; `--ablation dwell|cov` (the driver's own constants, and the
  default tag) and `--plan-score-params` (the runner's format) on `train` and `sweep`. The runner
  refuses a pair checkpoint whose recorded training plan differs from the one its arm flies; no
  record means the default plan.
- *R24.* A checkpoint must match its tag (the final check's F2): the runner's campaign check also
  refuses a `gX` tag whose manifest γ is not X/100; the `hand` tag unless the reward is F·hand, and
  an F·hand checkpoint under any other pair tag; an E3 checkpoint whose reward is not decision 7's
  bytes; and a network that took no update, a "trained" network still at its initial weights, which
  the spec's "no random-init arm" forbids in substance. The row's provenance stays tag plus sha.
- *R25.* FerrySim's device-serve columns (the final check's FS-1): FerrySim does not run the
  devices' service loops, so its device traces hold only `device_ready`, and a FerrySim row
  (`EpisodeResult.row`, and the scorer's row of a kept trace) shows coverage 0.0,
  participation_entropy 0.0 and jains_fairness 1.0 as harness artifacts, as UG4 declared for its
  harness. No study reads these columns. Documented in `inprocess.py` and `episode.py`; the parity
  tests split the masked columns into wall and serve columns and pin the artifact values. No
  blanking, and no consumer or scorer change (Rule 3).
- *R26.* `report` marks a sweep as not pre-registered, in its JSON and on the printed verdict's
  second line, when its γ set, its seeds per γ or its held-out count differs from decision 5's
  grid: with fewer than 8 seeds "rising" cannot be reached after Holm while "flat" still can. It
  does not refuse, because the calibration runs the same code with 3 seeds.
- *R27.* Accepted as built: `H1+L1` is refused where it would fly as plain H1, a guard against a
  mislabelled arm; E3 flies the driver's configured in-flight response, as D1–D5 do (its Pass 1 is
  the same under either).
- *R28.* The repair round's open questions. (1) `main` stays open at any derived-reward weights (the
  manifest records them and the sha finds the manifest), while `gX`, `dwell` and `cov` take only
  decision 4 (a)'s weights; Study 5.7's grid points train under free tags that no arm flies, which
  the runner refuses. (2) ε belongs to Study 5.5, under decision 4's default reward: the headroom
  command has no `--reward`, so an ε under 5.7's grid weights would come from a default-weight
  headroom, and 5.7 has no pre-registered ε (left to 5.7's own design); recorded, no change. (3)
  Fix B's scratch lag probe is superseded by the repository helper (`evaluate.fx_lags` and
  `fx_lag_median`, `LAG_EPISODES` = 200) and a slow re-measure test. (4) A logging leak, there
  since U8b: FerrySim's three command lines (`python -m experiments.ferrysim`, `.evaluate` and
  `.headroom`) called `logging.disable(WARNING)` for the whole process and never restored it, so
  any test calling one before `tests/golden/test_golden_host_mission.py` failed 77 golden nodes,
  which the full suite's collection order hid. Each `main` now puts the caller's level back in a
  `finally` (pinned by `test_each_command_line_puts_the_callers_logging_level_back`), and the
  reproduction went from 77 failed to 85 passed.
- *R29.* What the user's choice for R22 involves, made before the 5.5 sweep: the jittery score
  trains on `jittery` or on `jittery56`. Under either, the calibration and the sweep fly the same
  family (decision 3: one score per regime), the study name carries the family (R19), and one sweep
  runs per regime. Under `jittery56` each Study 5.5 cell gets 1/8 of practice instead of 1/4; the
  half cells' periods (52 s and 68 s) lie near 60 s, so N = 12 practice near Study 5.5's periods
  stays about half; the N = 6 control's share halves and the quarter cells (104 s and 136 s) take a
  quarter; the kept weights are chosen on the mean over all eight cells; validation drops to 25
  episodes per cell at the default 200 (`--val-episodes 400` keeps 50, at about +17 % training
  time). Under `jittery`, 60 s flies lag/P_c 0.44 at 120 s and 0.57 at 180 s, so the half cells sit
  near practice and the quarter cells outside it, and Study 5.6 must declare its quarter cells out
  of practice. FerrySim keys the 5.6 cells' streams by name, so its quarter and half cells do not
  share layouts; in stack trials the runner's seeds pair them. These move together at the re-pin
  after the N = 12 pilot: `STUDY_5_6_LAGS_S` (26 s and 34 s), the four P_c constants, the
  `jittery56` hash and the test literals (with `RATIO_BOUNDS` if the spread changes). Study 5.6
  reads its full held-out sample: 12 episodes mislead at N = 12. The orchestrator recommends
  `jittery56` with `--val-episodes 400`: it keeps decision 3's one score per regime and design D-G's
  practice over 5.6's periods, and keeps the 5.6 contrast clean; most of the cost falls on the N = 6
  control.

**Recorded, no change.**

- FerrySim credits an update whose backhaul upload was lost (other choices 7: w_i is the update's
  weight), so its reward departs from "what the merge actually pays" on lost uploads only: 10 of
  320 missions under the jittery realism (the final check's FerrySim dimension).
- `LEARNER_REVISION` does not count settings changed by command-line flags (`--lr`,
  `--epsilon-start`); a sweep that mixes settings is still refused through the training-spec check.
- A beacon insert can fly ahead of the pair's chosen stop; the record's `next` then names a stop not
  flown next, and `trimmed_next` stays False. No recorded or FerrySim path offers beacons.
- The mask omits the 1 s listen window and the session-start pricing: in the final check's
  end-to-end replay 123 of 2,484 decisions were followed by a re-plan at the next departure, which
  the departure check catches as specified.
- The 35 new files are LF in the working copy, as the spec's conventions ask (a byte count; an
  earlier note that they were CRLF came from a Git Bash line count that misreads LF files). With
  `core.autocrlf=true` `git add` stores them as LF either way.

**The headroom report** (decision 10 (i)(a), run 2026-10-01). On FerrySim's validation stream
(`ferrysim-val`, episodes 0–199 of each cell), at most 512 leaves per sortie, the derived reward
(c_t 0.1, c_cov 1) and the equal device model, in process, with no training and no stack trial; it
took 2,952 s (49 min) with 12 workers, not the "few minutes" the spec expected. The headroom of an
episode is V − R(FX arm), where V is the largest of the per-sortie oracle's sum (a depth-first
search over the admitted pairs, each leaf a deterministic replay with `fx_pair` flying the other
sorties) and the whole-episode returns of the FX arm and the slot's four scripted references. F is
reported beside V, not in it: it flies b̄ where the mask refuses it, so no pair choice reaches its
flight. ε = max(0.01, 0.1 × headroom).

| Cell | FX mean | Best scripted (mean) | Headroom ± SE | ε | Sorties with ≥ 2 decisions | Truncated sorties |
|---|---|---|---|---|---|---|
| jit-n6-45 | −0.1002 | `greedy_1` (−0.0954) | 0.0167 ± 0.0062 | 0.0100 | 8.5 % | 0/800 |
| jit-n6-90 | +0.2178 | `greedy_1` (+0.2249) | 0.0212 ± 0.0058 | 0.0100 | 3.4 % | 0/800 |
| jit-n12-120 | −0.1469 | `greedy_1` (−0.1205) | 0.0911 ± 0.0110 | 0.0100 | 87.6 % | 1/800 |
| jit-n12-180 | +0.0590 | `greedy_1` (+0.0663) | 0.0229 ± 0.0039 | 0.0100 | 32.8 % | 4/800 |
| cln-n12-120 | +0.0330 | `committed_pair` (+0.0371) | 0.0411 ± 0.0069 | 0.0100 | 86.3 % | 3/800 |
| cln-n12-180 | +0.1127 | `greedy_1` (+0.1162) | 0.0126 ± 0.0031 | 0.0100 | 64.6 % | 29/800 |

No pause: every cell's headroom is at least 0.01, the largest at jit-n12-120. ε is 0.01 in every
cell, with jit-n12-120 close to rising above the floor (the bootstrap CI's upper end, 0.114, would
give 0.0114, and its last 100 episodes alone give a headroom of 0.1044); Study 5.5's report reads ε
from the mean headroom of the cells it reads, about 0.057 for jit-n12-120 and jit-n12-180, so
ε = 0.01. A truncated sortie makes its cell's headroom a lower bound (cln-n12-180's 29 most). V
exceeds the best fixed rule by 0.0647 at jit-n12-120 and 0.0156 at jit-n12-180, Study 5.5's cells,
and by only 0.0091 at cln-n12-180, below ε. *`greedy_1`'s signal:* on validation `greedy_1` beats FX
in 5 of 6 cells (it is below FX at cln-n12-120), significantly at jit-n12-120 (+0.0265 ± 0.0077,
paired); by the same paired standard error also at jit-n12-180 and cln-n12-180, and by a sign test
in no cell. Per the spec, if it also beats FX by ε on the held-out runs, the report says so and the
user decides (critic A2). F's mean is the lowest of the six policies in every cell, yet its return
exceeds V in 42 of 200 jit-n12-120 and 40 of 200 jit-n6-45 episodes (open items: an arrival
check). An independent checker re-flew three cells by its own code (3,600 flights and 2,400 oracle
searches), re-read all 1,200 episodes and reproduced every number; its caveats: with F counted in V,
jit-n12-120's headroom would be 0.1227 and its ε 0.0123, the other cells' unchanged; and much of V
is hindsight selection, as the design intends, since the best single policy's mean gain over FX
reaches 0.01 only at jit-n12-120. (U8a's first report, 100 episodes per cell and F still in V,
also found no pause.)

**What landed, by unit** (Configuration Reference §19 has each setting):

- *Goldens at `386c275` (unit UG5).* `tests/golden/_build_p4_plan.py` (which imports UG4's builder,
  unedited), `data/p4_plan.json` (695 KiB) and `test_golden_p4_plan.py` (128 tests), captured on the
  untouched tree: eight stub trials of the plan arms through `Exp4Driver.run_trial`, with the real
  services run in process: F, FX, FB+medium, F-cov, F-cap and F-prio at N = 6, 1 MB, the jittery
  contact channel, 45 s, S = 2 and 6 missions (seed 59); FX at 60 s; and FX at N = 12, 120 s,
  S = 3, 4 missions (seed 26), which pins FX's next-stop rule and the departure check after a stop.
  Beside UG4's eight parts a ninth, `flight_slot`, records every call the mule makes to its flight
  slot with its arguments, so an argument added or left out fails by name; the other parts are
  compared on their `386c275` keys (an added key passes, as in UG4). There is no K = 2 trial (critic
  A4: Phase 4's K = 2 loopback stays the pin), and no trial carries a beacon offer or makes
  `plan_protected` decisive, so the beacon hook's place and the exempt stops' protection stay
  pinned by Phase 4's tests under the `386c275` baseline. `make_baseline.py` gains its third
  `BASES` entry (below). The goldens are now 356 tests: 144 + 84 + 128.
- *Types, configuration and interfaces (U0).* In `plan/types.py`: `pair_q` among the flight slots,
  and `StopContext`, `PairView`, `FitsPair`, `PairScorer`, `check_pair_scores` and `PairChoice`,
  outside `__all__` (import them from `hermes.scheduler.plan.types`). A choice's record is
  JSON-ready, free of wall time and read-only all the way down; a zero energy reference is stored as
  None. `policies/next_stop.py`, new: E3's observation (`E3Stop`, `E3View`) and the per-departure
  protocol (`NextStopPolicy`, `checked_choice`, `pass_1_only`), with no numpy and no plan import.
  In `processes/config.py`: the six checkpoint fields, declared just before `plan_mode`, inside
  `SIM_ONLY_MULE_FIELDS` and outside `FERRY_SPEC_FIELDS` and `PLAN_MULE_FIELDS`, so no Phase 3 or
  Phase 4 `ferry_params` string changes; `chen_dqn`; and their guards. Four of the six deliberate
  test edits.
- *The learner and its replay (U2, then U2b).* `selector/pair_q.py`: a masked pointer double DQN in
  numpy, one Q per pair row from shared weights, with Adam, the Huber loss, a global-norm clip, hard
  target syncs, the behaviour schedule and checkpoint format 2; `selector/pair_replay.py`:
  transitions that carry the next decision's rows and mask, variable-length batches, sampling from
  `random.Random(seed)`. They are new modules, not edits of `ddqn.py` and `replay.py`, which refuse
  γ = 0, train by SGD on a squared loss and score one stored next row, and which the H2 golden pins.
  U2's review round gave equal rows one Q, bit for bit (so the lowest-row tie rule decides between
  them), checks `behaviour_row`'s arguments before its two draws, and refuses a held-out entry that
  holds no score. U2b bound the purpose and the learner revision into the header (R2).
- *The mask's predicate and the runtime's readers (U4).* `FLScheduler.fits_after_service`, one
  method, plan mode only (refused in legacy mode and in Pass 2); `FerryRuntime.stop_contexts`,
  `class_offsets_db`, `energy_ref_j` and `e3_observation`, and the `StopPrice` type, all pure (R4).
  The offsets are differenced per link before the median (critic C6); E3's observation reads the
  realized SNR within reach and the mean beyond it (critic B7 iv) and imports `next_stop` lazily.
  AST tests show both files differ from `386c275`'s only by these, and the runtime at its defaults
  equals `386c275`'s on 27 cases.
- *The pair slot and the scope guard (U3).* `policies/pair_slot.py`: `PairQSlot` with an injected
  scorer, the four scripted references (`fx_pair`, `committed_pair`, `hyb` and `greedy_1`, each the
  slot's version of its rule, not the fixed arm), the mask's binding (`bind_fits_pair`, which
  answers only for the view's pairs), the decision records and their close, and the trainer's seam;
  in `selector/scope_guard.py`, one function, `assert_pairs_admitted`.
- *The supervisor (U5).* In `mule_main.py`: the pair branch at each Pass-1 arrival; the chosen stop
  moved to the front after the stop; `trimmed_next` set at the departure check after a decision;
  the records closed in `_ferry_result` on all three simulated-clock exits (the empty round, no
  DOWN, the normal path); the `pair_slot` keyword and its refusals; `install_flight_slot`; and E3's
  per-departure hook. `MissionRunResult` gains `pass_1_pairs`, `pass_1_e3` and
  `pass_1_e3_unvisited`, None at the defaults, and the clock is read once more at the Pass-1
  landing (`rec.landing_s`, a pure read).
- *FerrySim (U8a).* `experiments/ferrysim/inprocess.py`, a copy of UG4's in-process orchestrator
  with the helpers it takes from `tests` (critic C7) and two hooks; `cells.py`, `episode.py`,
  `reward.py`, `evaluate.py` and `headroom.py`. With the stub device model FerrySim reproduces UG5's
  oracle of `fx_45s` and `fx_n12_120s` in all nine parts, and with no hooks UG4's
  `h1_replan_trim_wide`; a stub FX trial through the real orchestrator equals FerrySim's run on
  every mule and cluster event, wall stamps masked, and on the row bar its wall and device-serve
  columns (R25). U8a's review round nested kept traces under the cell and the policy, pinned how a
  flight is read into the reward, and took F out of the headroom's V.
- *The features (U1).* `selector/pair_features.py`: `pair_v1` (36 columns on the three-class link
  with the phase block, 24 without; R11), `covering_classes`, the learned score's adapter
  (`LearnedPairScorer`), `load_pair_scorer` and `build_pair_slot` (R8). Every refusal of what a
  checkpoint path names is a `CheckpointError`, a missing file included; the phase block is tested
  at P_c = 30, 45 and 60 s.
- *E3 (U6).* `policies/chen_dqn.py`: `ChenDQNPolicy`, its rows (`e3_v1`, 10 columns), its
  checkpoints (kind `chen_dqn`, bound to its band) and its trainer's seam. It acts on the live
  network's online Q at each call, never on the target copy.
- *Processes, driver and runner (U7).* `processes/mule.py` loads a learned filling's checkpoint
  before the process binds anything (a refusal is `CheckpointRefused`, exit 1), reads a relative
  path under `REPO_ROOT`, announces `mule_ready.pair` or `mule_ready.policy_checkpoint`, and writes
  the new `mission_completed` fields, each left out when empty; `topology_builder.py` carries the
  six fields to every mule; `driver.py` gains the arm lists, the tags, `is_plan_arm`, `H1+L1`,
  `checkpoint_settings`, the per-trial checkpoint check and the provenance keys; `runner_main.py`
  gains `--pair-checkpoint`, `--policy-checkpoint`, `--allow-dirty-checkpoint` and
  `--require-trained`; `config.py` gains R3's guard. The deliberate edit at
  `test_p4_config_driver.py:583`, with its import on line 60 (R13).
- *Analysis (U9).* `events_consumer.py`: `PairDecision`, `E3Call`, and `MissionRecord`'s
  `pair_decisions`, `e3_calls` and `e3_unvisited`, each field read only in the form the mule writes
  it; `traces_scorer.py`: `PHASE_5_COLUMNS` behind `pair_columns`, and E3's `policy_params`
  provenance; `stats.py`: `tost_paired` and `trend_test` (Page's L). The default scorer row equals
  the `386c275` scorer's byte for byte on all 600 recorded trials and on UG4's and UG5's 16 fixture
  trials.
- *The trainer and the report (U8b).* `experiments/ferrysim/train.py`, `checkpoints.py`,
  `report.py` and `__main__.py`, the command line (`headroom`, `train`, `sweep`, `evaluate`,
  `report`). U8b's review round refuses replacing another family's checkpoint (R19), puts the N = 6
  control into `evaluate`'s default cells (R20), and turns a refusal of `report.decide` into a usage
  error.
- *The final check, the fix round and its repair*, below.

**Visible at the defaults, additive only.** Every mule's per-role JSON, on either clock, carries the
six checkpoint keys at null, inserted just before `plan_mode`, so nothing else in it moves. In
memory, `MissionRunResult` and the consumer's `MissionRecord` each gain three fields at None, which
no event carries at the defaults: the wall clock's `mission_completed` is built from explicit fields,
and the simulated clock's new optional fields are left out when None or empty. Error messages that
list the allowed values now name the Phase 5 ones (`ARMS`, `FLIGHT_SLOTS` and `chen_dqn`): for
example the untouched `cross_heuristic.flight_slot_policy("pair_q")` now reads "must be one of
(..., 'pair_q'), got 'pair_q'", which no pipeline path reaches. Nothing else: no trace event gains a
field at the defaults (tests pin that no Phase 5 field reaches the `mule_ready` or
`mission_completed` of F, FX, H1 on either clock, D1 or D4), every Phase 3 and Phase 4 row keeps its
strings, the trial CSV header is unchanged, and the scorer's default row is the Phase 4 one.

**How legacy identity was checked**, on its three faces:

- *For every unit:* the goldens before and after, both builders (UG4's `8 trials x 8 parts compared
  (same)` and UG5's `8 trials x 9 parts compared (same)`, whose lists of added keys hold only the six
  checkpoint keys), and each unit's neighbouring tests compared node by node; fresh-interpreter tests
  that H1, D1, D4, F and FX load none of `pair_slot`, `pair_features`, `pair_q`, `pair_replay`,
  `chen_dqn`, `next_stop` or `experiments.ferrysim` (other choices 13); a git check that the seven
  legacy selector files equal `386c275`'s (U2; it stands in for the spec's behavioural
  `_p4_ref`-style test, which no unit owned).
- *Phase 4's plan arms:* UG5's goldens, flight-slot calls and their arguments included.
- *The final check's legacy dimension:* a hunk-by-hunk review of the 13 modified source files
  (every changed line on a legacy path additive or an identity rewrite); an A/B against a
  `git archive` of `386c275`: 18 more legacy-mode trials, 17 more plan-arm trials and UG4's and
  UG5's 16, differing only by the six null keys; 51 trials byte for byte, every run-directory file,
  with only `plan_wall_s` masked; the wall-clock builders with 0 differences; the runner on six
  recorded command lines; the scorer's default row on 50 trials; the modules loaded in fresh
  interpreters; and 34 legacy error paths, identical but for the four messages above. Its flight
  dimension flew F, FX and the legacy arms on 283 configurations, bit-identical with the archive,
  and the end-to-end replay 39 trials with 0 value differences. 2,948 existing nodes in 70 files
  were compared with the `386c275` baseline, all the same.
- *The `tests/` diff:* of the 182 test files tracked at `386c275` none is deleted and five changed:
  UG5's `make_baseline.py` and `README.md`, and the three that carry exactly the six deliberate
  edits, with their line counts and node ids unchanged:
  1. `test_p4_types.py:211`: `FLIGHT_SLOTS` gains `pair_q`;
  2. `test_p4_types.py:633`: the example bad value `"pair_q"` becomes `"pair_x"` (node `kwargs2`);
  3. `test_p4_cross_heuristic.py:110-111`: `ch._SLOTS` and its loop compare with `FLIGHT_SLOTS[:2]`
     (line 116 stays, since `flight_slot_policy` keeps only the two fixed fillings);
  4. `test_p4_config_driver.py:155`: the restated `FLIGHT_SLOTS`;
  5. `test_p4_config_driver.py:583`: `ARMS` gains `PHASE_5_ARMS`, imported on the existing line 60
     (R13);
  6. `test_p4_config_driver.py:896-897` and `:916-918`: the per-role JSON gains
     `CHECKPOINT_MULE_FIELDS` beside `PLAN_MULE_FIELDS` (R1).

  The H2 golden (`tests/golden/_mule_harness.py:600-602`) and the recorded `[60.0]` A/B failure
  stay as they were.

**Test baselines and the compare rule.** `tests/golden/make_baseline.py` now holds three pass/fail
baselines (`BASES`): `pytest_baseline.txt`, the full suite at afa9526 before any Phase 3 change
(1,560 tests, 6 failures; signed off by the user on 2026-09-29); `pytest_baseline_6e6f92d.txt`, at
`6e6f92d` before any Phase 4 code change, with UG4's goldens (2,918 tests, UG4's 84 among them);
and `pytest_baseline_386c275.txt`, the full suite at `386c275` before any Phase 5 change, with UG5's
goldens added (4,824 tests, the 128 of `test_golden_p4_plan.py` among them: 4,819 passed and the
five deterministic afa9526 failures with their signatures; unit UG5), which gates Phase 4's own
tests too. UG5 recorded it in three full runs: run A, with the baseline's own test deselected,
wrote a provisional baseline; run B, the whole suite, wrote it; and run C re-recorded it after UG5's
review round added six tests. "The full suite passes" for Phase 5 means: the same as all three
baselines, that is the same outcome per node id and the same signature per known failure, new
tests allowed. `compare` checks every recorded baseline by default (`--base` picks one,
`--baseline` names a file). It exits 0 when the run is the same as each, 1 when it differs from
one, and 2 when a baseline file it was to compare with is missing, printing `MISSING` and still
comparing the others; the worst status wins. It also lists the new tests that fail, which are never
differences, so read that line. `FLAKY` holds only `test_exp4_real_model_synthetic_converges`,
which may pass or fail its known way (it passed in the `6e6f92d` and `386c275` baselines);
`--flaky NODE_ID` adds a test and `--strict` drops the list. `write --base 386c275` refuses unless
HEAD is `386c275` with `hermes/` and `experiments/` clean.

```
py -3.11 -m pytest tests -p no:cacheprovider -q -rfE --junitxml=run.xml
py -3.11 tests/golden/make_baseline.py compare run.xml                   # all three baselines
py -3.11 tests/golden/make_baseline.py compare run.xml --base 386c275    # one of them
```

Phase 5 adds 1,310 tests in 14 new test files (`tests/unit/test_p5_*.py` and
`tests/integration/test_p5_*.py`), beside UG5's 128 goldens (`tests/golden/test_golden_p4_plan.py`,
with its helper `tests/golden/_build_p4_plan.py`). The `386c275` baseline already records UG5's
128, so a compare against it lists the 14 files' tests as new; against `6e6f92d` and afa9526 it
lists Phase 4's and UG5's tests as new too. The full run of the Phase 5 tree: 6,133 tests on 2026-10-02: 6,128 passed, and the five known failures failed their recorded way; `make_baseline.py compare` exits 0, the same as all three baselines (new tests: 4,573 against afa9526, 3,215 against 6e6f92d, 1,309 against 386c275, none failing; the flaky real-model smoke test went failed to passed, which is allowed). One later change, the headroom and evaluate commands making their `--out` folder before flying (resolution R28), passed its own and its neighbours' tests; the full re-run after it was stopped.

**The final check (2026-10-02).** Six review dimensions: spec fidelity; the flight pipeline across
units; Rule 1 on all three faces with the `tests/` diff; the driver, runner, checkpoints and
analysis; FerrySim against the real system, and the training pipeline; and an end-to-end replay of
real Phase 5 trials. Three found nothing: the flight pipeline (about 16,000 pair decisions and
3,800 E3 calls re-derived independently), legacy identity with the six test edits, and the replay
(8 real-process trials and 840 in-process FerrySim trials: 2,484 FQ decisions, 8,445 mask bits and
5,614 E3 calls, 0 failures). The FerrySim dimension found FerrySim equal to real processes mission by
mission on 8 real trials, the equal-shard stand-in changing only the merge weights, and its reward,
recomputed from the real traces, equal to FerrySim's. Five findings were confirmed:

- *SPEC-1 (low):* R21 gave a false reason for counting choices: no Study 5.5 step reads the shares.
  R21 is amended (above).
- *SPEC-2 (low):* decision 6 (a)'s Study 5.6 cells and critic A3's lag-based ratio were neither
  built nor recorded as deferred: R22.
- *F1 (medium):* FQ-dwell and FQ-cov could not be trained under their own plans (critic C2), and
  the runner would fly a default-plan checkpoint under those tags on the ablated plan, unchecked:
  R23.
- *F2 (low):* nothing tied a checkpoint's content to the tag it flies under (γ, reward): R24.
- *FS-1 (low):* FerrySim's devices never run their service loop, so its device events and three
  non-wall row columns always differ from a real trial's: R25.

**The fix round and its repair (2026-10-02).** Fix A built R23 and R24; fix B built R22; fix C built
R25 and R26, and found no code comment or doc that repeated R21's false reason.
Adversarial reviews of the three raised 11 findings, 10 confirmed and 1 refuted (RB-3, on the
lag-ratio test's strength). The repair round fixed all 10, each pinned by a test, with the defaults
and the three legacy faces unchanged:

- *A-1:* headroom and ε are read on the evaluation's plan: `headroom --plan-score-params` records
  the plan it flew, and `report` refuses a headroom report of another plan than the evaluation's;
- *A-2 and A-4:* `train` and `sweep` refuse an explicit learned-arm `--tag` that its arm would not
  fly (its γ, its reward, decision 4 (a)'s weights under `gX`, `dwell` and `cov`, an ablation's
  plan), and the runner refuses other weights under `gX`, `dwell` and `cov`;
- *A-3:* the kept weights' update count is the kept validation's, whatever came before it (tests);
- *A-5:* `--policy-checkpoint`'s help states E3's own conditions (the bytes reward, an update);
- *RB-1 and RB-2:* the lag measurement lives in the repository (`evaluate.fx_lags`,
  `fx_lag_median`), and the comment above `STUDY_5_6_LAGS_S` gives the pooled statistic, its 95 %
  interval (25.4–27.5 s and 32.2–36.9 s), the 400-episode values (26.80 s and 35.12 s, which round
  to 27 and 35) and the rule that a re-pin re-measures on the same sample and statistic;
- *RB-4:* the lag-ratio test is marked slow;
- *C-T1 and C-T2:* the pre-registration label's tests cover "more" as well as "fewer", and pin the
  verdicts off the grid, so the label can change no step of the rule.

Then the logging fix of R28 (4).

**Frozen surface touched,** all behind the switches: `fl_scheduler.py` (one method,
`fits_after_service`); `mule_main.py` (the pair branch, the reorder, the records, the `pair_slot`
keyword, the install seam and E3's hook); `selector/scope_guard.py` (one function,
`assert_pairs_admitted`). The new `selector/` files (`pair_q.py`, `pair_replay.py`,
`pair_features.py`) fall under Amendment 7's opening. Untouched: `selector/{ddqn, replay, features,
target_selector_rl, selector_train, sim_env, __init__}.py`, `policies/cross_heuristic.py` and
`policies/__init__.py`, `mission/*`, `l1/*`, `experiments/sim/drone_env/*` and the golden
harnesses (UG5's files are new). Outside the frozen surface: `mule/ferry.py`, `plan/types.py`,
`processes/config.py` and `processes/mule.py`, the Exp 4 `driver.py`, `runner_main.py`,
`topology_builder.py` and `events_consumer.py`, and `analysis/traces_scorer.py` and
`analysis/stats.py`. New: `policies/pair_slot.py`, `policies/next_stop.py`, `policies/chen_dqn.py`,
and `experiments/ferrysim/` (`__init__.py`, `__main__.py`, `cells.py`, `checkpoints.py`,
`episode.py`, `evaluate.py`, `headroom.py`, `inprocess.py`, `report.py`, `reward.py`, `train.py`).
Tests: UG5's files, `make_baseline.py` and `README.md` in `tests/golden/`, the 14 new test files,
and the six deliberate edits.

**Deviations from the build plan** (also in `FeRRy_Build_Plan.html`, Phase 5):

- FerrySim is `experiments/ferrysim/`, not `selector/ferry_sim.py` (L496, L919): `hermes/` may not
  import `experiments/`, and the scheduler may not import `hermes.l1`, while FerrySim needs both.
- New pair modules (`selector/pair_q.py`, `pair_replay.py`) replace the planned changes to
  `ddqn.py` and `replay.py` (L918), which the H2 golden and the recorded A/B failure pin.
- The pair fills the flight slot; `_pick_channel_contact`, ChannelDDQN's logging hook, is untouched
  (L923).
- Timing: the band serves stop k at once, at the SNR observed on arrival, and the next-stop half
  carries s's predicted features (L543, L917).
- "≤ 18 pairs" is not a bound (L543): the probes reached up to 15 pairs at N = 12 and 21 at
  N = 24.
- The features (L917): 36 columns, not about 15; the value proxy is dropped (critic C5); a phase
  block and pooled context are added; the slack is log-scaled; three columns go beyond the spec's
  list (R11). Six of today's eleven slots are constant, not five (seven on mission 1).
- E3 is a numpy port of Chen's recipe, trained in FerrySim, with its declared deviations, not the
  vendored joint-action DQN (L921); M1 and O1 are deferred (L922).
- Checkpoints are committed only with the user's consent, after a LICENSE is added, and the
  trained-checkpoint guard sits in the runner, not the mule, so FerrySim's E3 bootstrap checkpoint
  loads (L923).
- The γ sweep (L1151-1156): γ ∈ [0, 1] and 10 seeds per γ, read on N = 12, with the pre-registered
  rule and its references, not 5 seeds with "flat" read as "not significant".
- Study 5.6 follows decision 6 (a), at N = 12's stand-in budgets (L1167, L1171): two cells, lag/P_c
  a quarter and a half, instead of the grid {0.25, 0.5, 1, 2}, which aliases; "FX tuned separately
  per regime" is dropped, because FX has no parameters.
- Study 5.7's grid is c_t × c_cov with no c_miss (L1180), and r_k is decision 4's (L550).
- H2 and H3 leave Exp 5 and `H1+L1` keeps the adaptive backhaul's reference (decision 8);
  ChannelDDQN is retired from the plan, its code and logging kept (L473, L1047).
- Labels: Phase 4's F stays the committed slot, and the plan's F is built as FQ (L1039); the paper
  may call FQ "F".
- T2, "with one band, the pair Q reduces to today's contact ranking" (L930), is restated as a
  structural reduction: with one class there is one row per stop, ties follow the remainder's order
  (R5), and a −travel scorer reproduces FX's nearest feasible stop.
- The exit gate is split (L935): a code gate now, a campaign gate after the go-ahead.
- FerrySim's reward credits an update whose backhaul upload was lost (other choices 7), so it
  departs from "what the merge actually pays" (L912) on lost uploads only.
- The design's plan inconsistencies 1–29 are resolved as it proposes, except #2 (critic A2), #16
  (B14), #20–21 (decision 6) and #24 (decision 8).
- Size: about 14,600 lines of code (14,554 added and 73 removed, over 30 files) and 17,600 of tests
  (17,602 added and 17 removed, over 20 files), docstrings and UG5's harness included, plus the
  `386c275` baseline record (4,848 lines) and the `p4_plan.json` fixture (695 KiB), against the
  ~1,500 (~300 optional) estimated in the plan; the Phase 5 spec had estimated about 4,800 and
  4,500, and expected more than 12,000 and 20,000.

**Open items.**

- *R14's candidate amendment*, for the user: shut the topology down when a mule fails at startup
  (`_run_topology` re-raises from `start_all` without `shutdown_all`, and the cluster process keeps
  running). It is not fixed, since it changes every arm's error path (Rule 3); the driver's
  per-trial checkpoint check means a refused checkpoint normally never reaches a mule.
- *An arrival check* that refuses a stop whose landing no longer fits at the observed rate, for
  every plan arm (design Q2): FX's own pair overruns at 19 of about 71 last-stop arrivals at N = 6,
  and the headroom report found F's return above V in about a fifth of jit-n12-120's and jit-n6-45's
  episodes, so decision 1 (a)'s mask is stricter than those flights turned out to need.
- *A plan-clock slope* (design Q4); *10 MB as a 5.5 sensitivity* (Q7); *the rollout of FX at the
  mean SNR* as a further reference (critic B5); *a 5.7 cell where the merge weight varies* (critic
  B4).
- *Irregular interference* (decision 6 (b)) and *M1*, only if Study 5.5 keeps the learned score;
  *O1* before the 5.4 headline.
- *The family in the checkpoint layout* (for example `<study>/<family>/<tag>/`): the applied fix
  refuses a cross-family overwrite and asks for one study per family (R19).
- *Counting flights rather than choices* in the agreement and re-order shares (U9's open question;
  R21 keeps choices); *E3's N per call* in `pass_1_e3` (a trace reader recomputes it).
- *Refusing a checkpoint of another learner revision* in the runner, and binding `episodes_trained`
  and `dirty` into the sha (U2b's open questions).
- *A `--warmup-transitions` flag:* a short command-line run never warms the replay (1,000
  transitions), so its kept weights are the initial ones and the runner refuses it (R24); a
  development run needs a code-level warm-up.
- *The pre-registration label* checks R26's three settings only: a report read on other cells, from
  an evaluation with `--start` other than 0, or with ε given directly still reads "pre-registered".
- *The headroom command has no `--reward`*, and the report never compares the headroom report's
  reward with the evaluation's (R28 (2)).
- *The vendored drone_env* has no licence from its author and is on public `main` (decision 9: ask
  its author or remove it; Phase 5 does not use it).

**What needs the user's go-ahead.** Nothing below runs, and nothing is committed, without it:

1. *The training campaigns:* 77–119 trainings, about 11–22 h plus 1–2 h of held-out evaluation
   (R20 puts Study 5.5's held-out evaluation at about 3.3 h at 8 workers with the N = 6 control),
   run in batches, the calibration and the controls first.
2. *E3's training settings* (R16): the pair learner's defaults, or Chen's lr 5e-4 and ε from 1.0;
   `--gamma` is required.
3. *The jittery score's family* (R22, R29): `jittery` or `jittery56`; the orchestrator recommends
   `jittery56` with `--val-episodes 400`.
4. *The LICENSE* (MIT, as the README names) before any checkpoint commit, and each checkpoint
   commit (decision 9).
5. *The pilots:* the N = 12 knee and stress budgets, after the Phase 3/4 pilots; then the cells,
   the 5.6 lags and the 5.6 periods are re-pinned.
6. *The stack trials* (Study 5.5's check, 5.6, 5.7, 5.3's E3 cells), re-costed first.
7. *The campaign gate,* in decision 10 (ii)'s order: the headroom report (run during the build),
   the sweep and its evaluation, the committed checkpoints, Study 5.5's verdict, the stack trials.
8. *R14's candidate amendment.*

## 6. Unfreezing

Amend this document with the reason, the changed files, and which recorded sweeps are invalidated.
Then re-open the [pre-re-run checklist](HERMES_PreRerun_Checklist.md).
