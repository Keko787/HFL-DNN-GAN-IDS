"""Phase 0 — a device the mule abandons must not look freshly served to D1/D2.

For every contact it drops or abandons, the mule feeds a synthetic TIMEOUT
stamped with the time of abandonment (``MuleSupervisor._widen_abandoned``). The
fold writes that TIMEOUT into ``last_contact_ts`` and ``last_served_round`` —
the fields MAX-AoI (D1) aged a device from and Oort's staleness term (D2) read as
L(i). Both baselines now age a device from its last CLEAN outcome
(``last_clean_ts`` / ``last_clean_round``), which only a CLEAN sets.
"""

from __future__ import annotations

import math

import pytest

from hermes.scheduler.policies import MaxAoIPolicy
from hermes.scheduler.policies.max_aoi import NEVER_SERVED_AGE, contact_age
from hermes.scheduler.policies.oort import staleness_bonus
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.stages.s3_deadline import fold_round_close_delta
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionOutcome,
    MuleID,
    RoundCloseDelta,
)

NOW = 1_000.0


def _wp(x: float, *devs: str) -> ContactWaypoint:
    return ContactWaypoint(
        position=(x, 0.0, 0.0),
        devices=tuple(DeviceID(d) for d in devs),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=0.0,
    )


def _delta(did: str, outcome: MissionOutcome, *, ts: float, rnd: int):
    return RoundCloseDelta(
        device_id=DeviceID(did),
        mule_id=MuleID("m1"),
        mission_round=rnd,
        outcome=outcome,
        utility=0.0,
        contact_ts=ts,
    )


def _clean(did: str, *, ts: float, rnd: int):
    return _delta(did, MissionOutcome.CLEAN, ts=ts, rnd=rnd)


def _abandoned(did: str, *, ts: float, rnd: int):
    """The delta ``_widen_abandoned`` feeds: a TIMEOUT stamped at abandonment."""
    return _delta(did, MissionOutcome.TIMEOUT, ts=ts, rnd=rnd)


def _state(did: str, *deltas: RoundCloseDelta) -> DeviceSchedulerState:
    st = DeviceSchedulerState(device_id=DeviceID(did))
    for d in deltas:
        fold_round_close_delta(st, d)
    return st


def _env(now: float = NOW) -> SelectorEnv:
    return SelectorEnv(
        mule_pose=(0.0, 0.0, 0.0), mule_energy=1.0, rf_prior_snr_db=20.0,
        beacon_window_s=30.0, now=now,
    )


# --------------------------------------------------------------------------- #
# The fold
# --------------------------------------------------------------------------- #

def test_only_a_clean_outcome_moves_the_baseline_clock():
    st = _state("a", _clean("a", ts=100.0, rnd=1), _abandoned("a", ts=500.0, rnd=3))
    assert (st.last_clean_ts, st.last_clean_round) == (100.0, 1)
    # The bookkeeping fields still record the abandonment, exactly as before.
    assert (st.last_contact_ts, st.last_served_round) == (500.0, 3)


def test_a_failed_real_session_does_not_count_as_service_either():
    st = _state(
        "a",
        _clean("a", ts=100.0, rnd=1),
        _delta("a", MissionOutcome.PARTIAL, ts=400.0, rnd=2),
    )
    assert (st.last_clean_ts, st.last_clean_round) == (100.0, 1)


def test_a_later_clean_moves_the_clock_forward():
    st = _state(
        "a",
        _clean("a", ts=100.0, rnd=1),
        _abandoned("a", ts=500.0, rnd=3),
        _clean("a", ts=700.0, rnd=4),
    )
    assert (st.last_clean_ts, st.last_clean_round) == (700.0, 4)


# --------------------------------------------------------------------------- #
# D1 — MAX-AoI
# --------------------------------------------------------------------------- #

def test_d1_age_runs_from_the_last_clean_not_the_abandonment():
    states = {
        DeviceID("a"): _state(
            "a", _clean("a", ts=100.0, rnd=1), _abandoned("a", ts=500.0, rnd=3),
        )
    }
    assert contact_age(_wp(1.0, "a"), states, now=600.0) == pytest.approx(500.0)


def test_d1_a_never_served_device_stays_infinitely_stale_after_abandonment():
    states = {DeviceID("a"): _state("a", _abandoned("a", ts=500.0, rnd=3))}
    assert contact_age(_wp(1.0, "a"), states, now=600.0) == NEVER_SERVED_AGE


def test_d1_goes_back_first_to_the_device_it_abandoned():
    # a: last update at 100, abandoned at 900. b: last update at 300.
    # a has waited longest for an update, so MAX-AoI must rank it first.
    # Before the fix a's age read as 100 s and it sorted behind b.
    states = {
        DeviceID("a"): _state(
            "a", _clean("a", ts=100.0, rnd=1), _abandoned("a", ts=900.0, rnd=3),
        ),
        DeviceID("b"): _state("b", _clean("b", ts=300.0, rnd=2)),
    }
    ranked = MaxAoIPolicy().rank_contacts(
        [_wp(1.0, "b"), _wp(2.0, "a")], states, _env(),
    )
    assert [w.devices[0] for w in ranked] == ["a", "b"]


# --------------------------------------------------------------------------- #
# D2 — Oort's staleness term
# --------------------------------------------------------------------------- #

def test_d2_staleness_counts_from_the_last_clean_round():
    st = _state("a", _clean("a", ts=100.0, rnd=1), _abandoned("a", ts=500.0, rnd=3))
    assert staleness_bonus(st, current_round=4, weight=0.1) == pytest.approx(
        0.1 * math.log(4) / math.sqrt(1)
    )


def test_d2_a_device_that_never_participated_gets_no_staleness_credit():
    st = _state("a", _abandoned("a", ts=500.0, rnd=3))
    assert staleness_bonus(st, current_round=4) == 0.0
