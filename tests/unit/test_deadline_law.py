"""FeRRy Phase 1 — the deadline law, one-shot overrides and the miss-priority key.

Pins what the build plan asks of the law ("monotone and clamped; the priority
order holds") and that the recorded additive law is untouched: with no law, or
the default one, every state field moves exactly as before.
"""

from __future__ import annotations

import math
import random

import pytest

from hermes.scheduler import FLScheduler
from hermes.scheduler.stages.s3_deadline import (
    FAST_PHASE_MISSED_WIDEN_S,
    FAST_PHASE_ON_TIME_SHRINK_S,
    LAW_ADDITIVE,
    LAW_MULTIPLICATIVE,
    MIN_DEADLINE_FULFILMENT_S,
    DeadlineLaw,
    DeadlineLawError,
    compute_deadline,
    effective_window,
    fold_cluster_amendment,
    fold_round_close_delta,
)
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel, filter_feasible
from hermes.types import (
    Bucket,
    ClusterAmendment,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionOutcome,
    MissionSlice,
    MuleID,
    RoundCloseDelta,
)

D = DeviceID("d")
MULT = DeadlineLaw(form=LAW_MULTIPLICATIVE)
CLEAN, PARTIAL, TIMEOUT = MissionOutcome.CLEAN, MissionOutcome.PARTIAL, MissionOutcome.TIMEOUT
OUTCOMES = (CLEAN, PARTIAL, TIMEOUT)


def _delta(outcome, *, ts=1000.0, did=D, rnd=1):
    return RoundCloseDelta(
        device_id=did, mule_id=MuleID("m"), mission_round=rnd,
        outcome=outcome, utility=0.0, contact_ts=ts,
    )


def _state(phi=60.0, did=D):
    return DeviceSchedulerState(device_id=did, deadline_fulfilment_s=phi)


# --------------------------------------------------------------------------- #
# The recorded law is untouched
# --------------------------------------------------------------------------- #

def test_default_law_is_the_recorded_one():
    law = DeadlineLaw()
    assert law.form == LAW_ADDITIVE and law.is_recorded
    assert not law.expires_overrides
    assert law.break_even_on_time_rate() == pytest.approx(2 / 3)


def test_additive_law_reproduces_the_recorded_fold_exactly():
    rng = random.Random(7)
    legacy, explicit = _state(), _state()
    for step in range(300):
        outcome = rng.choice(OUTCOMES)
        d = _delta(outcome, ts=1000.0 + step, rnd=step)
        fold_round_close_delta(legacy, d)
        fold_round_close_delta(explicit, d, law=DeadlineLaw())
        assert legacy == explicit
        now = 1000.0 + step + 0.5
        assert compute_deadline(legacy, now=now) == compute_deadline(
            explicit, now=now, law=DeadlineLaw(),
        )


def test_additive_law_has_no_ceiling():
    st = _state(60.0)
    for _ in range(100):
        fold_round_close_delta(st, _delta(TIMEOUT))
    assert st.deadline_fulfilment_s == 60.0 + 100 * FAST_PHASE_MISSED_WIDEN_S


# --------------------------------------------------------------------------- #
# The multiplicative law
# --------------------------------------------------------------------------- #

def test_each_outcome_scales_the_window_by_its_factor():
    assert MULT.next_window(60.0, CLEAN) == pytest.approx(48.0)
    assert MULT.next_window(60.0, PARTIAL) == pytest.approx(75.0)
    assert MULT.next_window(60.0, TIMEOUT) == pytest.approx(90.0)


def test_the_law_is_monotone():
    grid = [MULT.phi_min + k * (MULT.phi_max - MULT.phi_min) / 200 for k in range(201)]
    for outcome in OUTCOMES:
        nxt = [MULT.next_window(phi, outcome) for phi in grid]
        assert all(a <= b for a, b in zip(nxt, nxt[1:])), outcome
    for phi in grid:
        assert MULT.next_window(phi, CLEAN) <= phi
        assert MULT.next_window(phi, PARTIAL) >= phi
        assert MULT.next_window(phi, TIMEOUT) >= MULT.next_window(phi, PARTIAL)


def test_the_law_is_clamped():
    rng = random.Random(11)
    for start in (MULT.phi_min, 30.0, 60.0, 250.0, MULT.phi_max):
        st = _state(start)
        for step in range(1000):
            fold_round_close_delta(st, _delta(rng.choice(OUTCOMES), rnd=step), law=MULT)
            assert MULT.phi_min <= st.deadline_fulfilment_s <= MULT.phi_max
    hi, lo = _state(60.0), _state(60.0)
    for _ in range(40):
        fold_round_close_delta(hi, _delta(TIMEOUT), law=MULT)
        fold_round_close_delta(lo, _delta(CLEAN), law=MULT)
    assert hi.deadline_fulfilment_s == MULT.phi_max
    assert lo.deadline_fulfilment_s == MULT.phi_min


def test_break_even_rate_separates_tightening_from_relaxing():
    p_star = MULT.break_even_on_time_rate()
    assert p_star == pytest.approx(math.log(1.5) / (math.log(1.5) - math.log(0.8)))

    def drift(p):
        return p * math.log(MULT.beta_on) + (1 - p) * math.log(MULT.beta_timeout)

    assert drift(p_star) == pytest.approx(0.0, abs=1e-12)
    assert drift(p_star + 0.1) < 0 < drift(p_star - 0.1)

    rng = random.Random(3)

    def settle(p):
        """Mean window over the second half of a 400-contact run."""
        st, tail = _state(60.0), []
        for step in range(400):
            outcome = CLEAN if rng.random() < p else TIMEOUT
            fold_round_close_delta(st, _delta(outcome, rnd=step), law=MULT)
            if step >= 200:
                tail.append(st.deadline_fulfilment_s)
        return sum(tail) / len(tail)

    assert settle(0.95) < 15.0          # reliable: sits near the floor
    assert settle(0.30) > 200.0         # unreliable: sits near the ceiling


def test_partial_relaxes_less_than_timeout():
    partial, timeout = _state(), _state()
    fold_round_close_delta(partial, _delta(PARTIAL), law=MULT)
    fold_round_close_delta(timeout, _delta(TIMEOUT), law=MULT)
    assert 60.0 < partial.deadline_fulfilment_s < timeout.deadline_fulfilment_s
    # the recorded law does not split them
    a, b = _state(), _state()
    fold_round_close_delta(a, _delta(PARTIAL))
    fold_round_close_delta(b, _delta(TIMEOUT))
    assert a.deadline_fulfilment_s == b.deadline_fulfilment_s


@pytest.mark.parametrize("bad", [
    dict(form="linear"),
    dict(form=LAW_MULTIPLICATIVE, beta_on=1.0),
    dict(form=LAW_MULTIPLICATIVE, beta_on=0.0),
    dict(form=LAW_MULTIPLICATIVE, beta_partial=0.9),
    dict(form=LAW_MULTIPLICATIVE, beta_partial=2.0, beta_timeout=1.5),
    dict(form=LAW_MULTIPLICATIVE, phi_min=0.0),
    dict(form=LAW_MULTIPLICATIVE, phi_min=100.0, phi_max=50.0),
])
def test_invalid_laws_are_refused(bad):
    with pytest.raises(DeadlineLawError):
        DeadlineLaw(**bad)


def test_from_config_round_trips_and_expiry_follows_the_form():
    law = DeadlineLaw(form=LAW_MULTIPLICATIVE, beta_on=0.7, phi_max=200.0)
    assert DeadlineLaw.from_config(law.form, law.to_params()) == law
    assert DeadlineLaw.from_config(None, None) == DeadlineLaw()
    assert DeadlineLaw.from_config(LAW_MULTIPLICATIVE, {}).expires_overrides
    assert not DeadlineLaw.from_config(
        LAW_MULTIPLICATIVE, {"expire_overrides": False}
    ).expires_overrides
    assert not DeadlineLaw.from_config(LAW_ADDITIVE, {"expire_overrides": True}).is_recorded
    with pytest.raises(DeadlineLawError, match="unknown deadline parameter"):
        DeadlineLaw.from_config(LAW_MULTIPLICATIVE, {"beta": 2.0})


def test_effective_window_is_the_floor_or_the_clamp():
    assert effective_window(_state(1.0)) == MIN_DEADLINE_FULFILMENT_S
    assert effective_window(_state(1000.0)) == 1000.0
    assert effective_window(_state(1000.0), law=MULT) == MULT.phi_max
    assert effective_window(_state(1.0), law=MULT) == MULT.phi_min


# --------------------------------------------------------------------------- #
# Cluster overrides and window patches
# --------------------------------------------------------------------------- #

def test_overrides_stay_sticky_under_the_recorded_law():
    st = _state()
    st.deadline_override_ts = 500.0
    assert compute_deadline(st, now=1000.0) == 500.0      # long past, still used
    fold_round_close_delta(st, _delta(CLEAN))
    assert st.deadline_override_ts == 500.0


def test_overrides_are_one_shot_under_the_new_law():
    st = _state(60.0)
    st.deadline_override_ts = 1100.0
    assert compute_deadline(st, now=1000.0, law=MULT) == 1100.0      # still ahead
    assert compute_deadline(st, now=1200.0, law=MULT) == 1200.0 + 60.0  # passed
    fold_round_close_delta(st, _delta(TIMEOUT), law=MULT)
    assert st.deadline_override_ts is None                          # acted on


def test_cluster_window_patch_is_clamped_by_the_law():
    for law, big in ((None, 10_000.0), (MULT, MULT.phi_max)):
        states = {D: _state()}
        amend = ClusterAmendment(
            cluster_round=1, registry_deltas={D: {"deadline_fulfilment_s": 10_000.0}},
        )
        fold_cluster_amendment(states, amend, law=law)
        assert states[D].deadline_fulfilment_s == big


# --------------------------------------------------------------------------- #
# The miss-priority key
# --------------------------------------------------------------------------- #

def test_miss_streak_counts_consecutive_misses_under_any_law():
    for law in (None, MULT):
        st = _state()
        for outcome in (TIMEOUT, PARTIAL, TIMEOUT):
            fold_round_close_delta(st, _delta(outcome), law=law)
        assert st.miss_streak == 3
        fold_round_close_delta(st, _delta(CLEAN), law=law)
        assert st.miss_streak == 0


def _contact(x, did, deadline):
    return ContactWaypoint(
        position=(x, 0.0, 0.0), devices=(DeviceID(did),),
        bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=deadline,
    )


def test_priority_outranks_the_deadline_in_the_s3b_walk():
    # 1 m/s and no session time: each contact costs its distance in seconds.
    model = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)
    early = _contact(10.0, "early", deadline=100.0)
    missed = _contact(-10.0, "missed", deadline=200.0)
    edf = filter_feasible([early, missed], now=0.0, mission_deadline_ts=15.0, model=model)
    assert edf.kept == [early] and edf.dropped_budget == [missed]
    prio = {early: 0, missed: 2}
    ranked = filter_feasible(
        [early, missed], now=0.0, mission_deadline_ts=15.0, model=model,
        priority=prio.__getitem__,
    )
    assert ranked.kept == [missed] and ranked.dropped_budget == [early]


@pytest.mark.parametrize("miss_priority, admitted", [(False, "early"), (True, "missed")])
def test_scheduler_admits_the_missed_device_first_when_the_budget_binds(miss_priority, admitted):
    early, missed = DeviceID("early"), DeviceID("missed")
    sched = FLScheduler(
        now_fn=lambda: 1000.0,
        mission_budget_s=15.0,
        feasibility_model=FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0),
        deadline_law=MULT,
        miss_priority=miss_priority,
    )
    sched.ingest_slice(MissionSlice(
        mule_id=MuleID("m"), device_ids=(early, missed), issued_round=0, issued_at=0.0,
    ))
    sched.device_states[early].last_known_position = (10.0, 0.0, 0.0)
    sched.device_states[missed].last_known_position = (-10.0, 0.0, 0.0)
    sched.start_mission()
    # The missed device's window widens, so its deadline is now the later one.
    sched.ingest_round_close_delta(_delta(TIMEOUT, did=missed, ts=999.0))
    assert sched.device_states[missed].miss_streak == 1
    queue = sched.build_contact_queue(rf_range_m=5.0)
    assert [c.devices for c in queue] == [(DeviceID(admitted),)]
