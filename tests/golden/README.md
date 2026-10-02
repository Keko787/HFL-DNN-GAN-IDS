# Legacy goldens for FeRRy Phases 3, 4 and 5 (units UG, UG4 and UG5)

Most fixtures here are **afa9526 oracles**: they record what main at
`afa952682c1e8a30160a390397f7f369a898b584` computes, captured before any Phase 3
change (unit UG). One, `p3_sim.json`, is a **6e6f92d oracle**: it records Phase
3's simulated-clock pipeline as main at `6e6f92da038227147489515d876cc3f353584283`
runs it, captured before any Phase 4 change (unit UG4). One, `p4_plan.json`, is
a **386c275 oracle**: it records Phase 4's plan arms as main at
`386c27552e249da07550bc9042b4a907c7e8e684` runs them, captured before any Phase 5
change (unit UG5). Freeze Rule 1 requires every mechanism of a phase to sit
behind a switch whose default reproduces the recorded pipeline byte for byte
(additive fields only); in Phase 5 "recorded" has three faces, the wall clock
(the afa9526 fixtures), Phase 3's simulated clock with `plan_mode=legacy`
(`p3_sim.json`) and Phase 4's plan arms (`p4_plan.json`). With the switches at
their defaults, every test in this folder must pass.

## What is pinned

| Fixture (`data/`) | Test | Surface |
|---|---|---|
| `channel.json` | `test_golden_channel.py` | `experiments/exp4/channel.py`: SNR traces, loss schedules, chosen bands, chosen-band mean SNR (the mule's `rf_prior_snr_db`) over seeds x regime x missions x bands, fixed (H1/H2) and adaptive (H3); the 120 recorded C1/C2 trials re-derived from `results/exp4_matrix/C*_traces`; `hermes.l1.channel_model` checked the same way once U2 creates it |
| `feasibility.json` | `test_golden_feasibility.py` | `filter_feasible` (with and without the miss priority), `greedy_budget_walk`, MAX-AoI admission, `fedcs_greedy_select`, `FedExCarpPolicy.admit_and_order` and its `last_*` diagnostics, `MuleSupervisor._remaining_is_feasible` (S3b rule, budget rule, no check) and `_budget_pass_2`, over 2,400 seeded instances and the exact-boundary cases |
| `host_mission.json` | `test_golden_host_mission.py` | `HFLHostMission.run_contact` / `deliver_contact`: the contact map's 28 scenarios with synchronous and real threads, the late writer (critic A2), and the sequential joins of both passes (P-01 defect 2) |
| `supervisor.json` | `test_golden_supervisor.py` | `MuleSupervisor` end to end: H1, H1 with a budget (pre-flight drops; in-flight abort, also with S3c on), H2, D1 with a budget (and its abort), D4, budgeted Pass 2, multiplicative law with miss priority and S3c, and K = 2 at quorum 2 (empty missions docking; a survived DOWN timeout) |
| `topology.json` | `test_golden_topology.py` | `build_exp4_topology` over a grid (N, RF range, realism, 1-3 mules, CARP split) with the JSON each role is started with, as a real `MultiProcessOrchestrator.start_all` writes it (spawn and port read-back stubbed); `Exp4Driver.run_trial` on stub cells (topology, row with provenance columns, per-role JSON); every re-derivable kept trace in `results/` re-derived from its seed |
| `p3_sim.json` (6e6f92d) | `test_golden_p3_sim.py` | Phase 3's simulated clock end to end: eight stub trials through `Exp4Driver.run_trial`, the per-role JSON, and the real cluster, mule and device services run in this process (see [below](#phase-3s-simulated-clock-unit-ug4-p3_simjson)): each trial's row, per-role JSON, every mule event (`mule_ready`, `mission_started`, `mission_completed` and the rest, in order), every cluster event and every device event; contacts flown on wide, medium and narrow, at a declared and at the measured payload |
| `p4_plan.json` (386c275) | `test_golden_p4_plan.py` | Phase 4's plan arms end to end: eight stub trials of F, FX, FB+medium, F-cov, F-cap and F-prio run in process as UG4's are (see [below](#phase-4s-plan-arms-unit-ug5-p4_planjson)): the same parts, plus every call the mule made to its flight slot (the committed slot and FX's cross-heuristic, with their arguments, the orders FX tried and their verdicts, and what each returned); the cap binding and its violations, member subsets at the plan and in flight, hover stops, FX's band at arrival and its re-order after a stop at N = 12 |

`pytest_baseline.txt` records which tests of the full suite passed and failed at
afa9526 on this host, and for each failure its signature (failing line,
exception, first `assert` line, quoted exceptions). "The full suite passes" for
Phase 3 means: the same outcome per node id and the same signature per known
failure, new tests allowed. Six tests failed at afa9526, one more than the unit
spec expected (see the file's header); the user signed that baseline off on
2026-09-29 (the sixth, the real-model smoke test, is flaky under load; its fix
waits for the session-TTL pilot).

`pytest_baseline_6e6f92d.txt` is the same record at 6e6f92d, before any Phase 4
change (unit UG4), with this folder's Phase 4 goldens in it. Against afa9526
Phase 3's own tests are new tests, which `compare` allows even when they fail;
against 6e6f92d each must keep its outcome. "The full suite passes" for Phase 4
means: the same as the afa9526 baseline and the same as the 6e6f92d baseline.

`pytest_baseline_386c275.txt` is the same record at 386c275, before any Phase 5
change (unit UG5), with this folder's Phase 5 goldens (`test_golden_p4_plan.py`)
in it. Phase 4's own tests are new against the first two baselines, so only this
one gates them. "The full suite passes" for Phase 5 means: the same as all three
baselines (Phase 5 spec, Freeze Rule 1).

The real-model smoke test is on `make_baseline.py`'s flaky allow-list
(`FLAKY`): it may pass or fail its known way (`rounds_closed == 0`, the
signature the afa9526 run recorded), and a change between the two is reported
as "flaky tests that changed (allowed)", not as a difference. Before the list,
its passing made `compare` report DIFFERS on an otherwise identical run. Failing
in any other way, erroring or not running is still a difference; `--strict`
drops the list, `--flaky NODE_ID` adds a test to it. Compare a later run with

    py -3.11 -m pytest tests -p no:cacheprovider -q -rfE --junitxml=run.xml
    py -3.11 tests/golden/make_baseline.py compare run.xml                   # all three baselines
    py -3.11 tests/golden/make_baseline.py compare run.xml --base 386c275    # one of them

`compare` exits 1 if the run differs from any baseline it compared, and lists
the failing new tests of each; watch that line too, since new tests are never
differences. It exits 2 if a baseline it was to compare with is missing (by
default all three, so a baseline file left out of a commit cannot switch its
gate off in silence: it prints `MISSING` and still compares the others).
`test_golden_p3_sim.py` checks that `pytest_baseline_6e6f92d.txt` is there,
agrees with its header, keeps every afa9526 test and every Phase 3 test file,
and fails only the five deterministic afa9526 failures, with their signatures.
`test_golden_p4_plan.py` checks the same of `pytest_baseline_386c275.txt`
against the 6e6f92d baseline, with every Phase 4 test file (`test_p4_*.py`) and
the plan-arm goldens in it.

## Phase 3's simulated clock (unit UG4, `p3_sim.json`)

**The trials** (`_build_p3_sim.TRIALS`): the first five as the Phase 4 unit
spec names them, and three the unit's review added because none of the five
flies a contact off wide or at the measured payload. Phase 4 reworks exactly
that code (U6 gives `FerryRuntime` a mutable band and per-class physics; U7
reads the contact plan's band), and the goldens can only be captured on the
untouched tree:

| Trial | Arm and settings | What it exercises at 6e6f92d |
|---|---|---|
| `h1_replan_trim_wide` | H1; the Phase 4 pilots' flags (simulated clock, wide, the T_nom deadline unit, `replan` with the `trim` fallback, `agg:cutoff`, the `channel` reliability source), 1 MB, a 60 s budget, the seconds backhaul; N = 6, 4 missions, jittery, seed 38 | S3b's pre-flight budget drops (widened), an in-flight re-plan that trims (`arm_trimmed`) and one that drops (`arm`), a lost upload |
| `d1_max_aoi` | D1 (MAX-AoI), the same flags and layout, the recorded `mission` backhaul | the budget walk leaving stops out before takeoff (not reported at 6e6f92d), one in-flight re-plan |
| `d3_whittle` | D3 (Whittle, expected variant), as H1 | as D1, and a lost upload |
| `d4_route_only` | D4 (FedEx tour) under `agg:cutoff`, the route-only variant; as D1 | the whole tour every mission, over budget by 13.5 to 16.3 s |
| `h1_narrow_cliff_empty` | H1 on trial T2 of the Phase 3 final check: narrow, 1 MB, a 60 s budget, `device_positions(8, 777, 100.0)`, 3 missions | the cliff: S3b drops the one field-wide stop (99.1 s) before every mission, so every mission flies empty |
| `h1_narrow_cliff_flown` | T2 again at a 99.5 s budget, nothing else changed | the cliff's other side: S3b admits the field-wide stop (as `test_p3_final_fixes_mule.py` pins at the planner) and every mission flies it on narrow, all 8 members targeted in both passes, 40 to 120 s of dwell per pass; one mission overruns by 29.9 s |
| `h1_medium_replan` | H1, the pilots' flags on medium, the seconds backhaul; seed 26 (seed 38 drops nothing on medium) | S3b's pre-flight budget drops priced on medium, an in-flight re-plan (`arm`), a Pass-1 member below the SNR floor at arrival (unreachable, rate 0), backhaul carrier 2, a lost upload |
| `h1_narrow_measured` | H1, the pilots' flags on narrow at the measured payload (no `payload_bytes`, the driver's default), the seconds backhaul; seed 38 | the measured payload: sessions priced on the θ and synthetic batch the mule measures (milliseconds of airtime), the one field-wide narrow stop in both passes, a lost upload |

**How a trial runs.** `Exp4Driver.run_trial` runs unchanged (arm table, T_nom,
clock settings, topology builder, provenance, and the row it folds from the
trial's JSONL). Its orchestrator is the real one writing the per-role JSON
(spawn and port read-back stubbed, as above), and then, instead of spawning,
the real `ClusterService`, `MuleService` and `DeviceService` are built in this
process from that JSON, each writing its JSONL where its process would.
`MuleService.run` is the mule process's own service loop, so every mule event
comes from the process code. The cluster's loop is stepped by the harness:
each UP goes through `ClusterService._process_up`, the bootstrap through
`_dispatch_to_new_mules`. Each device is the `ClientMission` its
`DeviceService` builds (stub trainer, seeds, contact reliability), answering a
solicit or a push at once. The TCP links are synchronous in-process stand-ins
(the RF link token is still checked), `time.time` is a clock only the harness
moves (from 1.7e9 s), and HERMES threads run on `start()`. A trial takes about
0.1 s, and its bytes are the same under any `PYTHONHASHSEED` and whatever ran
before it in the process.

**How close to a real trial.** Every trial was also run once through the
real orchestrator at 6e6f92d (real subprocesses and TCP, 1.2 to 10 s each):
every mule and cluster event, the per-role JSON (ports aside) and the row
agreed, bar what the wall clock, the OS and the device service loops decide
there. Those are the envelope `ts`, `duration_s`, the ports, the devices'
serve events (in process the devices' traces hold only `device_ready`) and
the row columns built from them. With no `device_served` event, `coverage`
and `participation_entropy` are 0 and `jains_fairness` is 1.0 (Jain's index
of no serves) in every trial, and `mission_duration_s_mean` is the harness
clock's: harness artifacts, not Phase 3's values (a real trial folds its
devices' serves into them). So the goldens do not pin how the consumer folds
device serves; `tests/unit/test_exp4_metrics.py` does, under the two
baselines.

**How it is compared.** Every mapping in a row, a per-role JSON or an event is
a record at any depth: a key added with a default passes (Phase 4's
`pass_1_policy_drops` on a D-arm mission that drops, a new config field, a new
row column), a removed or renamed key fails. A mapping keyed by device id
(`deadline_state`, a stop's `snr_db`) is data and must keep exactly its keys.
Values, lists and each mule's sequence of event names must match exactly, so
a new event at the defaults fails. The row's JSON cells (`ferry_params`,
`policy_params`, ...) are strings and must match exactly: a Phase 4 key in
`ferry_params` at the defaults fails, as the spec requires. An added key is not
reported by the tests; `py -3.11 tests/golden/_build_p3_sim.py` lists every one
per trial and part, for the review.

## Phase 4's plan arms (unit UG5, `p4_plan.json`)

**The trials** (`_build_p4_plan.TRIALS`), as the Phase 5 spec's units table
names them (row UG5). All fly the Phase 4 pilots' flags (simulated clock, wide as
the reference class, the T_nom deadline unit with T_nom over the driver's default
20 layouts, `replan` with the `trim` fallback, `agg:cutoff`, the `channel`
reliability source, realism, the recorded `mission` backhaul, a jittery cell) at
a declared 1 MB on the jittery contact channel (the cells of the user's Phase 5
decision 3: `ferry_physics={"contact_regime": "jittery"}`). The six 45 s trials and
FX at 60 s share one layout (seed 59), as in one CSV; seeds 59 and 26 were
chosen by probes over seeds 1-60 and 1-40 because they exercise every mechanism
below:

| Trial | Arm and settings | What it exercises at 386c275 |
|---|---|---|
| `f_45s` | F; N = 6, 6 missions, a 45 s budget, S = 2; seed 59 | the search over the three classes (b̄ narrow or medium) and the committed order; the cap binding in five of six missions, violated `crowded`, `not_merged` and `dropped_in_flight`; member subsets at the plan (the complement dropped, reason `plan`, where the reduced stop flies); an in-flight member trim after a stop (`arm_trimmed`); a hover stop of a capped device; mission 6 over budget by 12.3 s |
| `fx_45s` | FX, the same cell | as F, and the band at arrival: five Pass-1 stops flown on a faster class than b̄ that reaches every one of its targets; no mission over budget |
| `fb_medium_45s` | FB+medium, the same cell | medium only, in both passes; subsets; a hover stop; two overruns |
| `f_cov_45s` | F-cov, the same cell | cap-only service: mission 1, before the cap binds, flies empty; then only capped devices are served |
| `f_cap_45s` | F-cap, the same cell | no cap (no capped device, no violation); subsets |
| `f_prio_45s` | F-prio, the same cell | weights are the ages alone (F's are age × (1 + miss streak)); on this layout it flies exactly as F does, so it differs in its plans' weights and the row's `miss_priority` |
| `fx_60s` | FX, the same cell at 60 s, the Phase 4 exit gate's FX budget | four band switches; subsets; violations |
| `fx_n12_120s` | FX; N = 12, 4 missions, 120 s (decision 3's stand-in budget), S = 3; seed 26 | FX's next-stop rule: two re-orders after a stop, its fold refusing the nearest candidate at six departures; the departure checks after the first re-order trimming a stop's members twice; the last stop of mission 1 dropped at the departure check; band switches |

There is no K = 2 trial: UG4's in-process trial runs one mule (`run_roles`;
Phase 5 critic A4), and Phase 4's K = 2 loopback in
`tests/integration/test_p4_plan_missions.py` stays that pin.

**How a trial runs.** As UG4's, with its helpers, imported from `_build_p3_sim`
and not edited. Around each trial `flight_slot_spy` wraps the two fillings of
the flight slot (`CommittedSlot`, `CrossHeuristic`: `next_stop` and
`band_at_arrival`, and each `fits` FX is handed) to record every call, its
arguments and its result. A wrapper binds the call to the method's own
signature and passes it on as it came, so it takes whatever the slot takes: a
call the slot refuses is refused by the slot, with its own error. It changes
nothing, and a test runs a trial with and without it and compares every other
part. The eight trials take about 1 s together,
and their bytes are the same under any `PYTHONHASHSEED` and whatever ran before
them in the process (checked under three seeds and three orders, UG4's trials
first in one of them). Two of them, `fx_n12_120s` and `f_45s`, were also run once
through the real orchestrator at 386c275 (real subprocesses and TCP, about 2 s
each): the row, the per-role JSON, `mule_ready`, `mission_started`,
`mission_completed` and every cluster event agreed with the fixture, bar what
the wall clock and the OS decide (the envelope `ts` and `duration_s`, the ports,
the planner's wall time, the row columns built from the devices' serves and the
harness clock, and the mule's `metrics_snapshot` timer of mission wall time).

**The planner's wall time.** `mission_completed.plan_wall_s` is
`build_ferry_plan`'s `time.perf_counter` seconds, the one wall time in a
plan-mode trace (Phase 4 critic B12). The fixture keeps the key and stores any
finite non-negative value as `"wall-seconds"` (`_build_p4_plan.WALL_TOKEN`), so a
removed or renamed key fails, and so does a value that is not such a number.

**How it is compared.** As `p3_sim.json` (above): every mapping is a record on
its 386c275 keys; values, lists and event sequences match exactly; a row's JSON
cells are exact strings, so a Phase 5 key in F's or FX's `ferry_params` at the
defaults fails. The ninth part, `flight_slot`, holds one record per call: a
`next_stop` call's `slot`, `pass`, `after_stop`, `remainder` (each
`ContactWaypoint`), departure `state` (`FlightState`), `fits` (each order tried,
as device lists, with its verdict, in the order tried) and the `index` returned;
a `band_at_arrival` call's `view` (the `ArrivalView`, or None when the slot reads
none) and the `band` returned. Phase 5's pair slot is a third filling, and the
committed and FX slots must keep their calls and arguments (Phase 5 spec, other
choices 1): this part pins that, under a rule stricter than critic A3's
(`_build_p4_plan.compare_part`, which the tests and the builder use). A changed
argument, verdict or result fails, and so does a call added or removed. So does
an argument a 386c275 call did not pass, even one the slot declares with a
default and the flight ignores: the spy passes it on, records it under
`added_arguments`, and the comparison names it per call (critic A3's rule
alone would let that key pass). An argument the call no longer passes fails as
a missing key. A field added with a default inside an argument (a
`FlightState`, `ContactWaypoint` or `ArrivalView` field) passes, as in every
other part. `py -3.11 tests/golden/_build_p4_plan.py` lists every added key per
trial and part, for the review.

**What the other tests read from the fixture:** each trial's arm (mule JSON, row,
slot); the cap's rule (capped means aged S or more) and its three violation
causes; member subsets at the plan and in flight; hover stops (a one-device
Pass-1 stop on the segment from the dock to a capped device, which the trace
does not mark); FX's class at arrival (the least dwell among the classes that
reach b̄'s targets) and its next stop (the nearest whose move to the front
`fits`, tried nearest first); the departure check before FX's pick (every Pass-1
re-plan's route is the remainder the slot then picks from); the committed slot's
plan order; F-cov's cap-only service; F-prio against F; and the device-serve row
columns, harness values as in `p3_sim.json`.

**What no trial here reaches**, and what pins it instead. Each of these is a
Phase 4 test that the 386c275 baseline records as passing
(`test_golden_p4_plan.PINNED_BY_PHASE_4`; a test checks the list against the
baseline), so the third baseline's gate holds it. Run them beside the goldens
when you change the code they name.

* **The beacon hook.** It has no source in a trial run through the driver: at
  386c275 nothing in `hermes/` or `experiments/` calls
  `MuleSupervisor.offer_contact` (only tests do). So the hook runs at every
  Pass-1 departure with an empty queue; no mission has an insert or a refused
  offer, and moving the hook after the slot's pick changes nothing here. Its
  place in the loop (departure check, then beacon hook, then the slot's pick)
  and its offers are pinned by `tests/integration/test_p4_plan_missions.py`
  (`test_the_departure_check_then_the_beacon_hook_then_the_slot`,
  `test_the_beacon_hook_groups_an_offer_within_the_committed_classs_range`).
  The goldens do pin the check before the pick.
* **The exempt stops' protection in flight** (`FLScheduler.plan_protected`,
  Phase 4 spec other choices 9; U4's `fits_after_service` reuses it). It
  returns exempt stops in seven of the eight trials, but it is never
  decisive: with `plan_protected` returning nothing, every trial is the same,
  part for part. No other seed would have made it decisive at these settings:
  a probe at 386c275 found the same for F, FX and F-cov at N = 6 over seeds
  1-60 and FX at N = 12 over seeds 1-40 (220 trials). It is pinned by
  `tests/unit/test_p4_fl_scheduler_plan.py` and
  `tests/integration/test_p4_plan_missions.py`.
* **K = 2**: `test_two_mules_each_plan_their_own_slice_from_their_own_ages`.

## How values are compared (`_canon.py`)

* Floats are stored as `"f:" + repr(x)` and compared exactly; arrays as shape,
  dtype and a SHA-256 of their bytes.
* A dataclass (or a `record`, used for rows and per-role JSON) is stored with
  the fields it had at afa9526, and the current object is compared on those
  fields only. A field Phase 3 adds with a default (`solicit_id`,
  `in_reply_to`, `uplink_drop`, `snr_db`, `band`, `bytes_sent`, new config keys
  and CSV columns) does not break a pin; a removed or renamed field does
  (critic A3). Plain dicts, lists and scalars must match exactly.
* The feasibility instances are also stored as digests. Such a digest hashes
  only the field values `ContactWaypoint` and `FeasibilityModel` had at afa9526
  (`_build_feasibility.AFA9526_FIELDS`), so U4's `band`, `range_m`,
  `pred_snr_db` and `ferry` do not move it.
* Only where real threads interleave in no fixed order (multi-device contacts,
  the late writer, the sequential joins) are lists compared as multisets and
  clock stamps masked.

## The harnesses

* **Clocks.** `time.time` is replaced by a clock that only the harness moves
  (the fake RF link, the dock, the gap between missions), never a read, so an
  extra clock read added later shifts no stamp. The supervisor's `now_fn` reads
  the same clock.
* **Threads.** HERMES worker threads run their target on `start()`, in a fixed
  order; other threads stay real. The late writer and the sequential joins run
  real threads: the late writer is ordered by events, the sequential joins by
  timing margins of one session TTL (0.3 s) each way. What they record about
  wall time are relations (was A's reply released before the routine
  returned), never raw times.
* **Devices** (`_mule_harness.py`) are real `ClientMission`s with pure trainers,
  answering every solicit at once in registration order and handling a push
  through their own Pass-1/Pass-2 handler.
* **Cluster.** A real `HFLHostCluster` runs inside the upload with
  `ClusterService`'s fold-and-dispatch policy. With two mules, a mule waiting
  for its quorum flies the other mule's mission inside its own DOWN wait.
* **Processes.** Nothing is launched: the per-role JSON comes from a real
  `MultiProcessOrchestrator` whose `_spawn` and `_wait_for_port` are stubbed
  (fixed placeholder ports) and whose `subprocess.Popen` raises. The kept-trace
  check takes the orchestrator's `write_text` calls in memory for speed; a
  test checks that this gives the same JSON as the files.

## Regenerating

    py -3.11 tests/golden/make_goldens.py            # compare only, writes nothing
    py -3.11 tests/golden/make_goldens.py --write    # only on a clean afa9526 tree
    py -3.11 tests/golden/_build_p3_sim.py           # p3_sim.json: compare, list added keys
    py -3.11 tests/golden/_build_p3_sim.py --write   # only on a clean 6e6f92d tree
    py -3.11 tests/golden/_build_p4_plan.py          # p4_plan.json: compare, list added keys
    py -3.11 tests/golden/_build_p4_plan.py --write  # only on a clean 386c275 tree
    py -3.11 tests/golden/make_baseline.py write run.xml --base 6e6f92d   # only at 6e6f92d
    py -3.11 tests/golden/make_baseline.py write run.xml --base 386c275   # only at 386c275

The tests never write a fixture. `--write` refuses unless HEAD is the base
commit and `hermes/` and `experiments/` are unchanged; `--force` overrides that
and is only right when a golden itself was wrong. Regeneration is byte-identical
(checked with different `PYTHONHASHSEED`s). `make_baseline.py write` has the
same guard, for the base it is given (`--base`, afa9526 by default).

## Legacy behaviour pinned on purpose

These are recorded as afa9526 behaves, not as it should:

* Adverts carry no reference to the solicit they answer: a later contact can
  consume an older eligible advert of a device that has since become
  unavailable, push to it, and time out (`supervisor.json`, `h1_no_budget`).
* A contact's workers are joined one after the other, each for up to 2 x TTL,
  so a slow contact holds the mule for up to 2 x TTL per device, and a reply
  that comes after the first join gave up is still in the returned map
  (`host_mission.json`, `sequential_joins:*`, P-01 defect 2).
* A Pass-1 worker that outlives its join writes into the next round: its
  gradient lands in round 2's accepted list and round 2's merge then refuses
  the mixed round (`host_mission.json`, `late_writer`, P-01 defect 3). A reply
  that comes after the return lands in the map the caller already holds
  (`sequential_joins:*`).
* Under `agg:plain` the driver writes `aggregation_params` as the rule's full
  default parameters, not `{}` (the same spec); the kept-trace check compares
  that key by the spec it builds.
* Kept D2 (Oort) traces are not re-derived: the stub driver refuses D2 without
  the real model. The S3c pilot's traces are re-derived with its
  `--h1-field-radius-m 150`, which their JSON does not record.

And as 6e6f92d behaves (`p3_sim.json`):

* The narrow cliff: S3b admits a stop whole or not at all, so under a budget
  below the one field-wide stop's time every mission flies empty
  (`h1_narrow_cliff_empty`), and at 99.5 s every mission flies all of it
  (`h1_narrow_cliff_flown`). Phase 4's member-subset admission defaults to
  `whole` for the H and D arms, which keeps both pins.
* The D arms' walks leave stops out before takeoff without reporting them:
  their `pass_1_preflight_drops` is `[]` (`d1_max_aoi`, `d3_whittle`). Phase 4
  reports them in `pass_1_policy_drops`, only when there are some; that field
  passes here as an added key.

And as 386c275 behaves (`p4_plan.json`):

* The plan prices the route at each class's mean SNR (δ_obs = 0); the dwell
  realized at the arrival SNR can run longer, and a member that does not answer
  adds a listen window, so a committed Pass 1 can land past the budget
  (`f_45s` mission 6 by 12.3 s; `fb_medium_45s` missions 4 and 5), and nothing
  re-checks after the last stop. On this layout FX's faster class at arrival
  avoids it (`fx_45s` mission 6 flies the same stop on medium).
* FX's next-stop half acts only after a Pass-1 stop, and only after the
  departure check (the beacon hook between them is pinned by Phase 4's tests;
  see above): at takeoff, in Pass 2 and for a lone stop it flies the plan's
  next stop without a fold.
* A hover stop is not marked in the trace, and the planner's wall time sits in
  `plan_wall_s` beside the plan (masked here).
* A plan that serves nobody reports the first class searched as b̄
  (`f_cov_45s` mission 1: wide).
