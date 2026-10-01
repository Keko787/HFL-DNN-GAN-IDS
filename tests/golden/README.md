# Legacy goldens for FeRRy Phases 3 and 4 (units UG and UG4)

Most fixtures here are **afa9526 oracles**: they record what main at
`afa952682c1e8a30160a390397f7f369a898b584` computes, captured before any Phase 3
change (unit UG). One, `p3_sim.json`, is a **6e6f92d oracle**: it records Phase
3's simulated-clock pipeline as main at `6e6f92da038227147489515d876cc3f353584283`
runs it, captured before any Phase 4 change (unit UG4). Freeze Rule 1 requires
every mechanism of a phase to sit behind a switch whose default reproduces the
recorded pipeline byte for byte (additive fields only); in Phase 4 "recorded"
has two faces, the wall clock (the afa9526 fixtures) and Phase 3's simulated
clock with `plan_mode=legacy` (`p3_sim.json`). With the switches at their
defaults, every test in this folder must pass.

## What is pinned

| Fixture (`data/`) | Test | Surface |
|---|---|---|
| `channel.json` | `test_golden_channel.py` | `experiments/exp4/channel.py`: SNR traces, loss schedules, chosen bands, chosen-band mean SNR (the mule's `rf_prior_snr_db`) over seeds x regime x missions x bands, fixed (H1/H2) and adaptive (H3); the 120 recorded C1/C2 trials re-derived from `results/exp4_matrix/C*_traces`; `hermes.l1.channel_model` checked the same way once U2 creates it |
| `feasibility.json` | `test_golden_feasibility.py` | `filter_feasible` (with and without the miss priority), `greedy_budget_walk`, MAX-AoI admission, `fedcs_greedy_select`, `FedExCarpPolicy.admit_and_order` and its `last_*` diagnostics, `MuleSupervisor._remaining_is_feasible` (S3b rule, budget rule, no check) and `_budget_pass_2`, over 2,400 seeded instances and the exact-boundary cases |
| `host_mission.json` | `test_golden_host_mission.py` | `HFLHostMission.run_contact` / `deliver_contact`: the contact map's 28 scenarios with synchronous and real threads, the late writer (critic A2), and the sequential joins of both passes (P-01 defect 2) |
| `supervisor.json` | `test_golden_supervisor.py` | `MuleSupervisor` end to end: H1, H1 with a budget (pre-flight drops; in-flight abort, also with S3c on), H2, D1 with a budget (and its abort), D4, budgeted Pass 2, multiplicative law with miss priority and S3c, and K = 2 at quorum 2 (empty missions docking; a survived DOWN timeout) |
| `topology.json` | `test_golden_topology.py` | `build_exp4_topology` over a grid (N, RF range, realism, 1-3 mules, CARP split) with the JSON each role is started with, as a real `MultiProcessOrchestrator.start_all` writes it (spawn and port read-back stubbed); `Exp4Driver.run_trial` on stub cells (topology, row with provenance columns, per-role JSON); every re-derivable kept trace in `results/` re-derived from its seed |
| `p3_sim.json` (6e6f92d) | `test_golden_p3_sim.py` | Phase 3's simulated clock end to end: eight stub trials through `Exp4Driver.run_trial`, the per-role JSON, and the real cluster, mule and device services run in this process (see [below](#phase-3s-simulated-clock-unit-ug4-p3_simjson)): each trial's row, per-role JSON, every mule event (`mule_ready`, `mission_started`, `mission_completed` and the rest, in order), every cluster event and every device event; contacts flown on wide, medium and narrow, at a declared and at the measured payload |

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

The real-model smoke test is on `make_baseline.py`'s flaky allow-list
(`FLAKY`): it may pass or fail its known way (`rounds_closed == 0`, the
signature the afa9526 run recorded), and a change between the two is reported
as "flaky tests that changed (allowed)", not as a difference. Before the list,
its passing made `compare` report DIFFERS on an otherwise identical run. Failing
in any other way, erroring or not running is still a difference; `--strict`
drops the list, `--flaky NODE_ID` adds a test to it. Compare a later run with

    py -3.11 -m pytest tests -p no:cacheprovider -q -rfE --junitxml=run.xml
    py -3.11 tests/golden/make_baseline.py compare run.xml                   # both baselines
    py -3.11 tests/golden/make_baseline.py compare run.xml --base 6e6f92d    # one of them

`compare` exits 1 if the run differs from any baseline it compared, and lists
the failing new tests of each; watch that line too, since new tests are never
differences. It exits 2 if a baseline it was to compare with is missing (by
default both, so a baseline file left out of a commit cannot switch its gate
off in silence: it prints `MISSING` and still compares the other).
`test_golden_p3_sim.py` checks that `pytest_baseline_6e6f92d.txt` is there,
agrees with its header, keeps every afa9526 test and every Phase 3 test file,
and fails only the five deterministic afa9526 failures, with their signatures.

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
    py -3.11 tests/golden/make_baseline.py write run.xml --base 6e6f92d   # only at 6e6f92d

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
