# Legacy goldens for FeRRy Phase 3 (unit UG)

These fixtures are **afa9526 oracles**: they record what main at
`afa952682c1e8a30160a390397f7f369a898b584` computes, captured before any Phase 3
change. Freeze Rule 1 requires every Phase 3 mechanism to sit behind a switch
whose default reproduces today's pipeline byte for byte (additive fields only).
With the switches at their defaults, every test in this folder must pass.

## What is pinned

| Fixture (`data/`) | Test | Surface |
|---|---|---|
| `channel.json` | `test_golden_channel.py` | `experiments/exp4/channel.py`: SNR traces, loss schedules, chosen bands, chosen-band mean SNR (the mule's `rf_prior_snr_db`) over seeds x regime x missions x bands, fixed (H1/H2) and adaptive (H3); the 120 recorded C1/C2 trials re-derived from `results/exp4_matrix/C*_traces`; `hermes.l1.channel_model` checked the same way once U2 creates it |
| `feasibility.json` | `test_golden_feasibility.py` | `filter_feasible` (with and without the miss priority), `greedy_budget_walk`, MAX-AoI admission, `fedcs_greedy_select`, `FedExCarpPolicy.admit_and_order` and its `last_*` diagnostics, `MuleSupervisor._remaining_is_feasible` (S3b rule, budget rule, no check) and `_budget_pass_2`, over 2,400 seeded instances and the exact-boundary cases |
| `host_mission.json` | `test_golden_host_mission.py` | `HFLHostMission.run_contact` / `deliver_contact`: the contact map's 28 scenarios with synchronous and real threads, the late writer (critic A2), and the sequential joins of both passes (P-01 defect 2) |
| `supervisor.json` | `test_golden_supervisor.py` | `MuleSupervisor` end to end: H1, H1 with a budget (pre-flight drops; in-flight abort, also with S3c on), H2, D1 with a budget (and its abort), D4, budgeted Pass 2, multiplicative law with miss priority and S3c, and K = 2 at quorum 2 (empty missions docking; a survived DOWN timeout) |
| `topology.json` | `test_golden_topology.py` | `build_exp4_topology` over a grid (N, RF range, realism, 1-3 mules, CARP split) with the JSON each role is started with, as a real `MultiProcessOrchestrator.start_all` writes it (spawn and port read-back stubbed); `Exp4Driver.run_trial` on stub cells (topology, row with provenance columns, per-role JSON); every re-derivable kept trace in `results/` re-derived from its seed |

`pytest_baseline.txt` records which tests of the full suite passed and failed at
afa9526 on this host, and for each failure its signature (failing line,
exception, first `assert` line, quoted exceptions). "The full suite passes" for
Phase 3 means: the same outcome per node id and the same signature per known
failure, new tests allowed. Six tests failed at afa9526, one more than the unit
spec expected (see the file's header); the user signed that baseline off on
2026-09-29 (the sixth, the real-model smoke test, is flaky under load; its fix
waits for the session-TTL pilot). Compare a later run with

    py -3.11 -m pytest tests -p no:cacheprovider -q -rfE --junitxml=run.xml
    py -3.11 tests/golden/make_baseline.py compare run.xml

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

The tests never write a fixture. `--write` refuses unless HEAD is the base
commit and `hermes/` and `experiments/` are unchanged; `--force` overrides that
and is only right when a golden itself was wrong. Regeneration is byte-identical
(checked with different `PYTHONHASHSEED`s). `make_baseline.py write` has the
same guard.

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
