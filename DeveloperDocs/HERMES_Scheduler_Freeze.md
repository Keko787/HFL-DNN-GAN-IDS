# Layer-2 scheduling methodology — FROZEN

**Frozen:** 2026-08-13, at commit of this document.
**Means:** the L2 decision pipeline, its gates, and their guarantees are settled. **Any change to
the files listed in §5 after this point invalidates recorded sweeps** and must go through the
[pre-re-run checklist](HERMES_PreRerun_Checklist.md).

State at freeze: working tree clean for `hermes/scheduler/` and `hermes/mule/`; **153 scheduler
tests passing**.

**Amendments** (§5a–5j): 1–4 landed before or alongside the recorded sweeps. 5 and 6 (2026-09-27
and 09-28) fix defects and change what the budgeted, `--l1-channel` and D1/D2 cells measure. 7
(2026-09-28) opens this surface for the FeRRy build, behind switches whose defaults keep this
pipeline. 8 and 9 (2026-09-28) land with the Phase 0/1 audit and Phase 2, and 10 (2026-09-29, the
RF transport fix) with Phase 3. The code behind every recorded result is the tag `exp4-recorded`.

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
| `plan_mode` | `legacy` | `ferry` | Phase 4, planned |
| `band_class_policy` | — | `search`; `fixed:<class>` is Path B+ | Phase 4, planned |
| Age cap `S` | off | set by build-plan decision D4 | Phase 4, planned |
| Flight-clock choice | distance order, or `TargetSelectorRL` within a bucket | masked pair score, or the cross-heuristic | Phase 5, planned |

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
                                          helper, §5j), Phase 4 (plan mode, clustering per band
                                          class, commit, a visited set per mission)
hermes/scheduler/stages/s3_deadline.py  Phase 1 (law, priority key, PARTIAL vs TIMEOUT, overrides
                                          that expire), Phase 3 (the law's time unit, the override
                                          refusal, the SpectrumSig fold, §5j)
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
  knees wait for Study 5.4.
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

## 6. Unfreezing

Amend this document with the reason, the changed files, and which recorded sweeps are invalidated.
Then re-open the [pre-re-run checklist](HERMES_PreRerun_Checklist.md).
