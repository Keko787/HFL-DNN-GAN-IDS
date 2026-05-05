"""Phase 7 — assertion tests for the 15 design principles.

`HERMES_FL_Scheduler_Design.md` §7 lists 15 principles that the system
is meant to enforce. Phase-7 DoD requires every principle to have a
passing assertion test (or, for principles that are pure architectural
invariants enforced by module structure, a documented rationale for
why no runtime test exists).

This file is the index. Most principles are pinned by tests that
already live elsewhere; see the comment block under each principle.
The tests in this file fill the three gaps surfaced by the Phase-7
audit:

* Principle 4 — two-phase deadline adaptation
* Principle 6 — beacons are opportunistic, not slice-promoting
* Principle 7 — deadline-aware aggregation never stalls indefinitely

If a future principle is added, follow the same pattern: extend the
mapping table in this docstring, and add a focused test below if the
principle isn't already covered.

Coverage map (authoritative as of Phase 7 closeout)
---------------------------------------------------

* **#1 Layer separation** — *architectural invariant.* Enforced by the
  module hierarchy (`hermes.scheduler` doesn't import from
  `hermes.transport`, etc.) rather than runtime behavior. Implicit in
  every integration test that wires the layers together end-to-end
  (`tests/integration/test_e2e_topology.py`).

* **#2 HFLHost split** — `tests/unit/test_host_cluster.py::test_two_mules_get_disjoint_slices`.

* **#3 Partial FedAvg upgrades privacy** —
  `tests/unit/test_partial_fedavg.py` (cluster operates on
  `PartialAggregate`, never on raw gradients).

* **#4 Two-phase deadline adaptation** — *gap, pinned below* by
  ``test_principle_4_fast_phase_and_slow_phase_hit_different_state``.

* **#5 Only the waypoint crosses L2→L1** — `SelectorEnv` exposes RF
  priors as read-only fields (`tests/unit/test_selector_features.py`).
  L1's API has no write-back from the selector; structural.

* **#6 Beacons are opportunistic** — *gap, pinned below* by
  ``test_principle_6_beacon_does_not_promote_into_slice``.

* **#7 Deadline-aware aggregation** — *gap, pinned below* by
  ``test_principle_7_min_participation_does_not_stall_when_threshold_met``.

* **#8 Disjoint mission slicing** —
  `tests/integration/test_e2e_faults.py::test_mission_slice_collision_rejected`
  (chunk-O fault test) plus `tests/unit/test_device_registry.py`.

* **#9 θ_gen never leaves Tier 2** — type system enforces it: the
  `DownBundle` dataclass has no `theta_gen` field, only `theta_disc` +
  `synth_batch`. Pinned below by
  ``test_principle_9_down_bundle_carries_no_theta_gen``.

* **#10 Eligibility computed locally on edge (S2B)** —
  `tests/unit/test_s2b_flag.py`.

* **#11 Symmetric server/client** — *architectural invariant.* Each
  tier-boundary has exactly one server class + one client class
  (`HFLHostMission`/`ClientMission`, `HFLHostCluster`/`ClientCluster`).
  No runtime test asserts symmetry without becoming a tautology. Implicit
  in `tests/integration/test_e2e_topology.py`.

* **#12 TargetSelectorRL bounded to intra-bucket ordering** —
  `tests/unit/test_target_selector_rl.py::test_scope_guard_rejects_foreign_device`
  + `tests/integration/test_selector_scope_guard.py`.

* **#13 Two-pass missions** —
  `tests/unit/test_selector_pass_gate.py::test_select_target_in_pass_2_raises`
  + `tests/integration/test_mule_supervisor_two_pass.py::test_pass_2_does_not_invoke_selector`.

* **#14 Local training is offline; FL sessions exchange-only** —
  `tests/integration/test_two_pass_contact.py`.

* **#15 Mule's circuit is contact events, not per-device** —
  `tests/unit/test_s3a_cluster.py` + `tests/integration/test_two_pass_contact.py`.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.scheduler import FLScheduler
from hermes.transport import LoopbackDockLink
from hermes.types import (
    BeaconObservation,
    Bucket,
    ClusterAmendment,
    ContactHistory,
    DeviceID,
    DeviceRecord,
    DownBundle,
    MissionOutcome,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MissionSlice,
    MuleID,
    PartialAggregate,
    RoundCloseDelta,
    SpectrumSig,
    UpBundle,
)


# --------------------------------------------------------------------------- #
# Principle 4 — Two-phase deadline adaptation
# --------------------------------------------------------------------------- #
#
# Fast phase = in-mission, per-device session outcome (CLEAN/TIMEOUT) folds
# into ``DeviceSchedulerState.deadline_fulfilment_s``.
# Slow phase = at-dock, ``ClusterAmendment`` carries explicit overrides
# (``deadline_overrides``) and ``registry_deltas`` (positions, priorities).
#
# The principle's contract: the two phases address *different* slots of
# DeviceSchedulerState and don't overlap. Fast phase tunes the rolling
# fulfilment window; slow phase plants explicit override timestamps and
# refreshes registry-derived fields. A regression that mixed the two
# (e.g., a slow-phase amendment that wrote into the fast-phase counter)
# would break the locality story underneath the deadline math.

def test_principle_4_fast_phase_and_slow_phase_hit_different_state():
    sch = FLScheduler(now_fn=lambda: 1000.0)
    sch.ingest_slice(
        MissionSlice(
            mule_id=MuleID("m1"),
            device_ids=(DeviceID("d0"),),
            issued_round=1,
            issued_at=1000.0,
        ),
        registry_records=[
            DeviceRecord(
                device_id=DeviceID("d0"),
                last_known_position=(0.0, 0.0, 0.0),
                spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
            ),
        ],
    )

    state = sch.device_states[DeviceID("d0")]
    fulfilment_before = state.deadline_fulfilment_s
    override_before = state.deadline_override_ts
    assert override_before is None

    # ---- Fast phase: a CLEAN delta should shrink fulfilment_s ---- #
    sch.ingest_round_close_delta(
        RoundCloseDelta(
            device_id=DeviceID("d0"),
            mule_id=MuleID("m1"),
            mission_round=1,
            outcome=MissionOutcome.CLEAN,
            utility=0.9,
            contact_ts=1000.0,
        )
    )
    fulfilment_after_fast = state.deadline_fulfilment_s
    override_after_fast = state.deadline_override_ts
    assert fulfilment_after_fast < fulfilment_before, \
        "fast phase did not move deadline_fulfilment_s"
    assert override_after_fast is None, \
        "fast phase incorrectly wrote into the slow-phase override slot"

    # ---- Slow phase: a ClusterAmendment override should plant a
    # ---- deadline_override_ts WITHOUT touching fulfilment_s.
    sch.ingest_slice(
        MissionSlice(
            mule_id=MuleID("m1"),
            device_ids=(DeviceID("d0"),),
            issued_round=2,
            issued_at=2000.0,
        ),
        amendment=ClusterAmendment(
            cluster_round=1,
            deadline_overrides={DeviceID("d0"): 9999.0},
        ),
    )
    assert state.deadline_override_ts == 9999.0, \
        "slow phase did not plant deadline_override_ts"
    assert state.deadline_fulfilment_s == fulfilment_after_fast, \
        "slow phase clobbered the fast-phase fulfilment window"


# --------------------------------------------------------------------------- #
# Principle 6 — Beacons are opportunistic, never a summon signal
# --------------------------------------------------------------------------- #
#
# A beacon observation is bonus information: if the mule happens to be in
# range and the device pings, the scheduler may opportunistically include
# it in a contact. But a beacon must NOT promote a non-slice device into
# the round's slice — slice membership is set at dock time by the
# cluster, never by an in-mission beacon.

def test_principle_6_beacon_does_not_promote_into_slice():
    sch = FLScheduler(now_fn=lambda: 1000.0)
    sch.ingest_slice(
        MissionSlice(
            mule_id=MuleID("m1"),
            device_ids=(DeviceID("a"), DeviceID("b")),
            issued_round=1,
            issued_at=1000.0,
        )
    )
    # Pre-condition: c is not in the slice.
    assert DeviceID("c") not in sch.device_states or \
        sch.device_states[DeviceID("c")].is_in_slice is False

    # Fire a beacon for c.
    sch.ingest_beacon(
        BeaconObservation(device_id=DeviceID("c"), observed_at=1000.5)
    )

    # Beacon may auto-create a state row for c (so S1 can admit it),
    # but that row must NOT claim slice membership.
    c_state = sch.device_states.get(DeviceID("c"))
    assert c_state is not None, "beacon should auto-create scheduler state"
    assert c_state.is_in_slice is False, \
        "beacon promoted a non-slice device into the slice (principle 6)"
    # The slice members are unaffected.
    assert sch.device_states[DeviceID("a")].is_in_slice is True
    assert sch.device_states[DeviceID("b")].is_in_slice is True


# --------------------------------------------------------------------------- #
# Principle 7 — Deadline-aware aggregation: never stall indefinitely
# --------------------------------------------------------------------------- #
#
# The cluster's ``aggregate_pending`` must produce merged weights as
# soon as ``min_participation`` mules have reported, even if other
# mules expected this round haven't shipped UP yet. The principle
# rules out "wait forever for everyone" semantics — the cluster's
# default policy is partial-FedAvg with whoever showed up.

def test_principle_7_min_participation_does_not_stall_when_threshold_met():
    registry = DeviceRegistry()
    registry.register(
        device_id=DeviceID("d0"),
        position=(0.0, 0.0, 0.0),
        spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
    )
    registry.register(
        device_id=DeviceID("d1"),
        position=(10.0, 0.0, 0.0),
        spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
    )
    # Two mules in the registry → if the cluster waited for both, an UP
    # from only one would block aggregation. min_participation=1 means
    # one is enough.
    registry.rebalance([MuleID("m1"), MuleID("m2")], round_counter=0)

    cluster = HFLHostCluster(
        registry=registry,
        generator=StubGeneratorHost(disc_weights=[
            np.zeros((4,), dtype=np.float32),
        ]),
        dock=LoopbackDockLink(),
        synth_batch_size=2,
        min_participation=1,
    )

    # Only mule m1 ships an UP — m2 stays silent.
    mule = MuleID("m1")
    up = UpBundle(
        mule_id=mule,
        partial_aggregate=PartialAggregate(
            mule_id=mule,
            mission_round=1,
            weights=[np.ones((4,), dtype=np.float32)],
            num_examples=10,
            contributing_devices=(DeviceID("d0"),),
        ),
        round_close_report=MissionRoundCloseReport(
            mule_id=mule,
            mission_round=1,
            started_at=0.0,
            finished_at=10.0,
            lines=[
                MissionRoundCloseLine(
                    device_id=DeviceID("d0"),
                    outcome=MissionOutcome.CLEAN,
                    contact_ts=1.0,
                ),
            ],
        ),
        contact_history=ContactHistory(mule_id=mule, mission_round=1, records=[]),
    )
    cluster.ingest_up_bundle(up)

    merged = cluster.aggregate_pending()
    assert merged is not None, (
        "aggregate_pending stalled with only m1 reporting "
        "(min_participation=1 should have closed the round)"
    )


# --------------------------------------------------------------------------- #
# Principle 9 — θ_gen never leaves Tier 2 (type-level enforcement)
# --------------------------------------------------------------------------- #
#
# The DownBundle dataclass — the only payload that crosses Tier-2 → Tier-1
# (cluster → mule) and Tier-1 → device — has no `theta_gen` field. A
# regression that added one would surface here as a schema check.

def test_principle_9_down_bundle_carries_no_theta_gen():
    field_names = {f.name for f in DownBundle.__dataclass_fields__.values()}
    assert "theta_gen" not in field_names, (
        f"DownBundle gained a theta_gen field ({field_names}) — principle 9 says "
        "θ_gen never leaves Tier 2; only θ_disc + synth cross to mules + devices"
    )
    # Positive assertion: the legitimate fields are still present.
    assert "theta_disc" in field_names
    assert "synth_batch" in field_names
