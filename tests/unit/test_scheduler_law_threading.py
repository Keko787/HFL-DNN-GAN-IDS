"""FeRRy Phase 1 — the deadline law threaded through ``FLScheduler``.

``test_deadline_law.py`` pins the law on the stage functions. These tests pin
that the scheduler hands its law to every one of them: the fast-phase fold
(with the ``answered`` flag), the slow-phase amendment fold, and each queue
builder's deadline. A call site that drops ``law=`` silently runs the recorded
additive law instead, which no stage-level test can see (audit #16).

Each expectation is chosen to differ from what the regression would give (the
additive law's value, an unclamped window, the contact's shared deadline), so
a lost law cannot pass by coincidence.
"""

from __future__ import annotations

import pytest

from hermes.scheduler import FLScheduler
from hermes.scheduler.stages.s3_deadline import LAW_MULTIPLICATIVE, DeadlineLaw
from hermes.types import (
    ClusterAmendment,
    DeviceID,
    MissionOutcome,
    MissionSlice,
    MuleID,
    RoundCloseDelta,
)

D = DeviceID("d")
NOW = 1000.0
MULT = DeadlineLaw(form=LAW_MULTIPLICATIVE)
CLEAN, PARTIAL, TIMEOUT = MissionOutcome.CLEAN, MissionOutcome.PARTIAL, MissionOutcome.TIMEOUT


def _slice(*dids):
    return MissionSlice(
        mule_id=MuleID("m"), device_ids=tuple(dids or (D,)), issued_round=0, issued_at=0.0,
    )


def _scheduler(law=MULT, amendment=None, *, positions=None):
    """One-device scheduler (``d`` at (1, 0, 0)) unless ``positions`` says otherwise."""
    positions = positions or {D: (1.0, 0.0, 0.0)}
    sched = FLScheduler(now_fn=lambda: NOW, deadline_law=law)
    sched.ingest_slice(_slice(*positions), amendment)
    for did, pos in positions.items():
        sched.device_states[did].last_known_position = pos
    return sched


def _delta(outcome, *, answered=False, did=D):
    return RoundCloseDelta(
        device_id=did, mule_id=MuleID("m"), mission_round=1,
        outcome=outcome, utility=0.0, contact_ts=NOW - 1.0, answered=answered,
    )


def _phi(sched, did=D):
    return sched.device_states[did].deadline_fulfilment_s


def _plan_deadlines(sched):
    """The deadline each queue builder stamps on the one device's waypoint."""
    contacts = sched.build_contact_queue(rf_range_m=5.0)
    pass_2 = sched.build_pass_2_queue(rf_range_m=5.0)
    targets = sched.build_target_queue()
    assert len(contacts) == len(pass_2) == len(targets) == 1
    return contacts[0].deadline_ts, pass_2[0].deadline_ts, targets[0].deadline_ts


# --------------------------------------------------------------------------- #
# Fast phase — ingest_round_close_delta
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("law, after_timeout, after_clean", [
    (MULT, 90.0, 72.0),     # 60 × 1.5, then × 0.8
    (None, 70.0, 65.0),     # the recorded +10 s / −5 s
])
def test_the_fast_phase_fold_runs_the_schedulers_law(law, after_timeout, after_clean):
    sched = _scheduler(law)
    sched.ingest_round_close_delta(_delta(TIMEOUT))
    assert _phi(sched) == pytest.approx(after_timeout)
    sched.ingest_round_close_delta(_delta(CLEAN))
    assert _phi(sched) == pytest.approx(after_clean)


@pytest.mark.parametrize("outcome, answered, expected", [
    (TIMEOUT, True, 75.0),      # answered, then dropped: β_partial
    (TIMEOUT, False, 90.0),     # never answered: β_timeout
    (PARTIAL, False, 75.0),     # a PARTIAL relaxes by β_partial either way
])
def test_the_answered_flag_reaches_the_law(outcome, answered, expected):
    sched = _scheduler()
    sched.ingest_round_close_delta(_delta(outcome, answered=answered))
    assert _phi(sched) == pytest.approx(expected)


def test_the_answered_flag_leaves_the_recorded_law_alone():
    sched = _scheduler(None)
    sched.ingest_round_close_delta(_delta(TIMEOUT, answered=True))
    assert _phi(sched) == 70.0


def test_the_step_scales_the_clamped_window():
    # A stored Φ outside [Φ_min, Φ_max] is clamped before the step, so the
    # deadline's Φ and the next step agree: 0.8 × 40 and 1.5 × 100, not the
    # clamp of 0.8 × 60 (40) and of 1.5 × 60 (100).
    low_cap = _scheduler(DeadlineLaw(form=LAW_MULTIPLICATIVE, phi_max=40.0))
    assert _phi(low_cap) == 60.0
    low_cap.ingest_round_close_delta(_delta(CLEAN))
    assert _phi(low_cap) == pytest.approx(32.0)

    high_floor = _scheduler(DeadlineLaw(form=LAW_MULTIPLICATIVE, phi_min=100.0))
    assert _phi(high_floor) == 60.0
    high_floor.ingest_round_close_delta(_delta(TIMEOUT))
    assert _phi(high_floor) == pytest.approx(150.0)


# --------------------------------------------------------------------------- #
# Slow phase — ingest_slice's amendment fold
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("law, expected", [(MULT, 300.0), (None, 10_000.0)])
def test_a_cluster_window_patch_is_clamped_by_the_schedulers_law(law, expected):
    amend = ClusterAmendment(
        cluster_round=1, registry_deltas={D: {"deadline_fulfilment_s": 10_000.0}},
    )
    assert _phi(_scheduler(law, amend)) == expected


# --------------------------------------------------------------------------- #
# Overrides — every queue builder computes the deadline under the law
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("law, expected", [
    (MULT, NOW + 60.0),     # expired: back to Time + Φ (idle 0, never served)
    (None, 500.0),          # the recorded override is sticky
])
def test_a_passed_override_expires_in_every_queue_under_the_law(law, expected):
    sched = _scheduler(law, ClusterAmendment(cluster_round=1, deadline_overrides={D: 500.0}))
    assert sched.device_states[D].deadline_override_ts == 500.0
    assert _plan_deadlines(sched) == (expected, expected, expected)


def test_a_pending_override_still_wins_under_the_law():
    sched = _scheduler(MULT, ClusterAmendment(cluster_round=1, deadline_overrides={D: 1100.0}))
    assert _plan_deadlines(sched) == (1100.0, 1100.0, 1100.0)


@pytest.mark.parametrize("law, left", [(MULT, None), (None, 1100.0)])
def test_an_outcome_consumes_the_override_under_the_law(law, left):
    sched = _scheduler(law, ClusterAmendment(cluster_round=1, deadline_overrides={D: 1100.0}))
    sched.ingest_round_close_delta(_delta(TIMEOUT))
    assert sched.device_states[D].deadline_override_ts == left


# --------------------------------------------------------------------------- #
# Per-device plan deadlines
# --------------------------------------------------------------------------- #

def test_the_plan_records_each_devices_own_deadline():
    near, far = DeviceID("near"), DeviceID("far")
    sched = _scheduler(positions={near: (1.0, 0.0, 0.0), far: (2.0, 0.0, 0.0)})
    sched.device_states[far].deadline_fulfilment_s = 1000.0     # clamps to 300
    queue = sched.build_contact_queue(rf_range_m=5.0)
    # Both devices share one contact, which carries the tighter deadline; the
    # plan keeps each device's own, under the law's clamp.
    assert len(queue) == 1 and set(queue[0].devices) == {near, far}
    assert queue[0].deadline_ts == NOW + 60.0
    assert sched.last_plan_deadlines == {near: NOW + 60.0, far: NOW + 300.0}
