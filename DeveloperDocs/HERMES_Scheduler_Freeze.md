# Layer-2 scheduling methodology — FROZEN

**Frozen:** 2026-08-13, at commit of this document.
**Means:** the L2 decision pipeline, its gates, and their guarantees are settled. **Any change to
the files listed in §5 after this point invalidates recorded sweeps** and must go through the
[pre-re-run checklist](HERMES_PreRerun_Checklist.md).

State at freeze: working tree clean for `hermes/scheduler/` and `hermes/mule/`; **153 scheduler
tests passing**.

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

## 6. Unfreezing

Amend this document with the reason, the changed files, and which recorded sweeps are invalidated.
Then re-open the [pre-re-run checklist](HERMES_PreRerun_Checklist.md).
