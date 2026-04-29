"""Phase 4 Stage 3 — deadline formula + bucket classifier tests."""

from __future__ import annotations

import pytest

from hermes.scheduler.stages import (
    classify_bucket,
    compute_deadline,
    fold_cluster_amendment,
    fold_round_close_delta,
)
from hermes.scheduler.stages.s3_deadline import (
    FAST_PHASE_MISSED_WIDEN_S,
    FAST_PHASE_ON_TIME_SHRINK_S,
    MIN_DEADLINE_FULFILMENT_S,
    compute_idle_time,
)
from hermes.types import (
    Bucket,
    ClusterAmendment,
    DeviceID,
    DeviceSchedulerState,
    MissionOutcome,
    MuleID,
    RoundCloseDelta,
)


# --------------------------------------------------------------------------- #
# compute_deadline
# --------------------------------------------------------------------------- #

def test_deadline_plain_formula_no_idle():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"),
        deadline_fulfilment_s=60.0,
        idle_time_ref_ts=0.0,
    )
    # 1000 + 60 - 0 = 1060
    assert compute_deadline(st, now=1000.0) == pytest.approx(1060.0)


def test_deadline_formula_with_idle_subtracts():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"),
        deadline_fulfilment_s=60.0,
        idle_time_ref_ts=990.0,  # 10s idle at now=1000
    )
    # 1000 + 60 - 10 = 1050
    assert compute_deadline(st, now=1000.0) == pytest.approx(1050.0)


def test_deadline_override_short_circuits():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"),
        deadline_fulfilment_s=60.0,
        idle_time_ref_ts=990.0,
        deadline_override_ts=1234.0,
    )
    assert compute_deadline(st, now=1000.0) == pytest.approx(1234.0)


def test_deadline_fulfilment_floor_respected():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"),
        deadline_fulfilment_s=1.0,  # below floor
    )
    # floor kicks in -> 1000 + MIN - 0
    assert compute_deadline(st, now=1000.0) == pytest.approx(
        1000.0 + MIN_DEADLINE_FULFILMENT_S
    )


def test_idle_time_floors_at_zero():
    st = DeviceSchedulerState(device_id=DeviceID("d"), idle_time_ref_ts=2000.0)
    # "idle_ref in the future" is nonsensical but must not go negative
    assert compute_idle_time(st, now=1000.0) == 0.0


# --------------------------------------------------------------------------- #
# classify_bucket
# --------------------------------------------------------------------------- #

def test_classify_new_device():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"), is_new=True, is_in_slice=True
    )
    assert classify_bucket(st, now=1000.0) is Bucket.NEW


def test_classify_scheduled_not_new():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"), is_new=False, is_in_slice=True
    )
    assert classify_bucket(st, now=1000.0) is Bucket.SCHEDULED_THIS_ROUND


def test_classify_beacon_active_only():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"),
        is_new=False,
        is_in_slice=False,
        last_beacon_ts=995.0,
    )
    assert classify_bucket(st, now=1000.0, beacon_window_s=30.0) is Bucket.BEACON_ACTIVE


def test_classify_unbucketable_raises():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"), is_new=False, is_in_slice=False
    )
    with pytest.raises(ValueError):
        classify_bucket(st, now=1000.0)


# --------------------------------------------------------------------------- #
# fold_round_close_delta — fast phase
# --------------------------------------------------------------------------- #

def _delta(outcome: MissionOutcome, ts: float = 1000.0) -> RoundCloseDelta:
    return RoundCloseDelta(
        device_id=DeviceID("d"),
        mule_id=MuleID("m"),
        mission_round=1,
        outcome=outcome,
        utility=0.8,
        contact_ts=ts,
    )


def test_fast_phase_clean_shrinks_window_and_clears_new():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"),
        is_new=True,
        deadline_fulfilment_s=60.0,
    )
    fold_round_close_delta(st, _delta(MissionOutcome.CLEAN, ts=1000.0))
    assert st.is_new is False
    assert st.last_outcome is MissionOutcome.CLEAN
    assert st.last_contact_ts == 1000.0
    assert st.idle_time_ref_ts == 1000.0
    assert st.deadline_fulfilment_s == pytest.approx(
        60.0 - FAST_PHASE_ON_TIME_SHRINK_S
    )


def test_fast_phase_partial_widens_window():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"),
        is_new=False,
        deadline_fulfilment_s=60.0,
        idle_time_ref_ts=800.0,
    )
    fold_round_close_delta(st, _delta(MissionOutcome.PARTIAL, ts=1000.0))
    assert st.deadline_fulfilment_s == pytest.approx(
        60.0 + FAST_PHASE_MISSED_WIDEN_S
    )
    # idle_ref must NOT move on a failed attempt
    assert st.idle_time_ref_ts == 800.0


def test_fast_phase_timeout_widens_window():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"),
        deadline_fulfilment_s=60.0,
    )
    fold_round_close_delta(st, _delta(MissionOutcome.TIMEOUT))
    assert st.deadline_fulfilment_s == pytest.approx(
        60.0 + FAST_PHASE_MISSED_WIDEN_S
    )


def test_fast_phase_shrink_respects_floor():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"),
        deadline_fulfilment_s=MIN_DEADLINE_FULFILMENT_S + 1.0,
    )
    fold_round_close_delta(st, _delta(MissionOutcome.CLEAN))
    assert st.deadline_fulfilment_s == MIN_DEADLINE_FULFILMENT_S


def test_fast_phase_rejects_cross_device_delta():
    st = DeviceSchedulerState(device_id=DeviceID("d"))
    d = RoundCloseDelta(
        device_id=DeviceID("OTHER"),
        mule_id=MuleID("m"),
        mission_round=1,
        outcome=MissionOutcome.CLEAN,
        utility=0.9,
        contact_ts=1000.0,
    )
    with pytest.raises(ValueError):
        fold_round_close_delta(st, d)


# --------------------------------------------------------------------------- #
# fold_cluster_amendment — slow phase
# --------------------------------------------------------------------------- #

def test_slow_phase_applies_deadline_override():
    d1 = DeviceID("d1")
    d2 = DeviceID("d2")
    states = {
        d1: DeviceSchedulerState(device_id=d1),
        d2: DeviceSchedulerState(device_id=d2),
    }
    amend = ClusterAmendment(
        cluster_round=1,
        deadline_overrides={d1: 2000.0},
    )
    fold_cluster_amendment(states, amend)
    assert states[d1].deadline_override_ts == 2000.0
    assert states[d2].deadline_override_ts is None


def test_slow_phase_ignores_untracked_devices():
    d1 = DeviceID("d1")
    states = {d1: DeviceSchedulerState(device_id=d1)}
    amend = ClusterAmendment(
        cluster_round=1,
        deadline_overrides={DeviceID("other"): 2000.0},
    )
    fold_cluster_amendment(states, amend)  # should not raise
    assert states[d1].deadline_override_ts is None


def test_slow_phase_applies_registry_deltas():
    d1 = DeviceID("d1")
    states = {d1: DeviceSchedulerState(device_id=d1)}
    amend = ClusterAmendment(
        cluster_round=1,
        registry_deltas={
            d1: {
                "last_known_position": (1.0, 2.0, 3.0),
                "deadline_fulfilment_s": 120.0,
            }
        },
    )
    fold_cluster_amendment(states, amend)
    assert states[d1].last_known_position == (1.0, 2.0, 3.0)
    assert states[d1].deadline_fulfilment_s == 120.0


def test_slow_phase_registry_delta_floors_fulfilment():
    d1 = DeviceID("d1")
    states = {d1: DeviceSchedulerState(device_id=d1)}
    amend = ClusterAmendment(
        cluster_round=1,
        registry_deltas={d1: {"deadline_fulfilment_s": 0.1}},
    )
    fold_cluster_amendment(states, amend)
    assert states[d1].deadline_fulfilment_s == MIN_DEADLINE_FULFILMENT_S
