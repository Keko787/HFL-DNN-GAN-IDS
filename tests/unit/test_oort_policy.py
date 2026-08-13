"""Oort's statistical-utility selection (arm B2) — Freeze Amendment 3.

Two things under test:

1. **The policy** — ranks on `|B_i| * sqrt(mean Loss^2)` plus a staleness bonus,
   both read from what each device reported *last time it was visited*. That
   retrospective property is the whole reason Oort ports to a data mule when
   FedCS does not, so it is pinned here.
2. **The plumbing** — the raw loss and sample count must survive the trip
   device -> FLReadyAdv -> RoundCloseDelta -> DeviceSchedulerState. They are new
   optional fields, and the tests pin that adding them changed nothing for
   H0-H3: an emitter that omits them leaves the folded state untouched.
"""

from __future__ import annotations

import math

import pytest

from hermes.scheduler.policies import OortPolicy, OortUnusableError
from hermes.scheduler.policies.oort import (
    DEFAULT_STALENESS_WEIGHT,
    UNEXPLORED_UTILITY,
    staleness_bonus,
    statistical_utility,
)
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.selector.scope_guard import SelectorScopeViolation
from hermes.scheduler.stages.s3_deadline import fold_round_close_delta
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionOutcome,
    MissionPass,
    MuleID,
    RoundCloseDelta,
)

NOW = 5_000.0


def _wp(x: float, *devs: str) -> ContactWaypoint:
    return ContactWaypoint(
        position=(x, 0.0, 0.0),
        devices=tuple(DeviceID(d) for d in devs),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=0.0,
    )


def _st(did: str, *, loss=None, n=0, served_round=0, contact_ts=NOW - 10):
    return DeviceSchedulerState(
        device_id=DeviceID(did),
        last_loss=loss,
        last_num_examples=n,
        last_served_round=served_round,
        last_contact_ts=contact_ts,
    )


def _env(now=NOW) -> SelectorEnv:
    return SelectorEnv(
        mule_pose=(0.0, 0.0, 0.0), mule_energy=1.0, rf_prior_snr_db=20.0,
        beacon_window_s=30.0, now=now,
    )


def _rank(cands, states):
    return OortPolicy().rank_contacts(cands, states, _env())


# --------------------------------------------------------------------------- #
# 1. The utility formula
# --------------------------------------------------------------------------- #

def test_statistical_utility_is_samples_times_loss():
    """|B_i| * sqrt(Loss^2)."""
    assert statistical_utility(_st("a", loss=0.5, n=100)) == pytest.approx(50.0)


def test_more_data_at_equal_loss_wins():
    big, small = _st("big", loss=0.5, n=200), _st("small", loss=0.5, n=10)
    assert statistical_utility(big) > statistical_utility(small)


def test_higher_loss_at_equal_data_wins():
    """Oort's core claim: high-loss clients carry more learning signal."""
    hard, easy = _st("hard", loss=0.9, n=50), _st("easy", loss=0.05, n=50)
    assert statistical_utility(hard) > statistical_utility(easy)


def test_a_negative_loss_cannot_invert_the_ranking():
    """sqrt(loss^2) is |loss| — a sign slip upstream must not flip the order."""
    assert statistical_utility(_st("x", loss=-0.5, n=10)) == pytest.approx(5.0)


def test_an_unmeasured_device_outranks_every_measured_one():
    """Oort explores clients it has never measured."""
    assert statistical_utility(_st("new")) == UNEXPLORED_UTILITY
    assert statistical_utility(_st("new")) > statistical_utility(
        _st("known", loss=99.0, n=10_000)
    )


def test_zero_examples_counts_as_unmeasured():
    assert statistical_utility(_st("x", loss=0.5, n=0)) == UNEXPLORED_UTILITY


# --------------------------------------------------------------------------- #
# 2. The staleness bonus
# --------------------------------------------------------------------------- #

def test_staleness_bonus_matches_the_papers_form():
    st = _st("a", loss=0.1, n=10, served_round=4)
    got = staleness_bonus(st, current_round=9)
    assert got == pytest.approx(DEFAULT_STALENESS_WEIGHT * math.log(9) / 2.0)


def test_a_longer_overlooked_device_gets_a_bigger_bonus():
    """The starvation pressure our Phi-widening applies, arrived at separately."""
    recent, old = _st("r", served_round=9), _st("o", served_round=1)
    assert staleness_bonus(old, current_round=10) > staleness_bonus(
        recent, current_round=10
    )


def test_no_staleness_bonus_in_the_first_round():
    assert staleness_bonus(_st("a", served_round=0), current_round=1) == 0.0


def test_never_served_gets_no_bonus_because_infinity_already_covers_it():
    assert staleness_bonus(_st("a", served_round=0), current_round=5) == 0.0


# --------------------------------------------------------------------------- #
# 3. Ranking behaviour
# --------------------------------------------------------------------------- #

def test_the_highest_utility_contact_goes_first():
    states = {
        DeviceID("lo"): _st("lo", loss=0.1, n=10, served_round=1),
        DeviceID("hi"): _st("hi", loss=0.9, n=100, served_round=1),
    }
    got = _rank([_wp(1.0, "lo"), _wp(2.0, "hi")], states)
    assert got[0].devices[0] == "hi"


def test_unexplored_contacts_come_first():
    states = {
        DeviceID("known"): _st("known", loss=9.0, n=999, served_round=1),
        DeviceID("new"): _st("new"),
    }
    got = _rank([_wp(1.0, "known"), _wp(2.0, "new")], states)
    assert got[0].devices[0] == "new"


def test_a_contact_is_worth_the_SUM_of_its_devices():
    """Visiting a cluster collects every member, so they add up."""
    states = {
        DeviceID("a"): _st("a", loss=0.5, n=10, served_round=1),
        DeviceID("b"): _st("b", loss=0.5, n=10, served_round=1),
        DeviceID("solo"): _st("solo", loss=0.5, n=15, served_round=1),
    }
    # pair = 5+5 = 10 > solo = 7.5, even though solo's single device is richer
    got = _rank([_wp(1.0, "solo"), _wp(2.0, "a", "b")], states)
    assert set(got[0].devices) == {"a", "b"}


def test_order_is_deterministic_and_independent_of_input_order():
    states = {DeviceID(d): _st(d, loss=0.5, n=10, served_round=1)
              for d in ("a", "b", "c")}
    cands = [_wp(1.0, "a"), _wp(2.0, "b"), _wp(3.0, "c")]
    first = [w.devices[0] for w in _rank(cands, states)]
    for perm in ([cands[2], cands[0], cands[1]], [cands[1], cands[2], cands[0]]):
        assert [w.devices[0] for w in _rank(perm, states)] == first


def test_every_candidate_is_returned_exactly_once():
    states = {DeviceID(f"d{i}"): _st(f"d{i}", loss=0.1 * i, n=10, served_round=1)
              for i in range(5)}
    cands = [_wp(float(i), f"d{i}") for i in range(5)]
    got = _rank(cands, states)
    assert len(got) == 5 and {id(w) for w in got} == {id(w) for w in cands}


def test_pass_2_invocation_is_refused():
    with pytest.raises(SelectorScopeViolation):
        OortPolicy().rank_contacts(
            [_wp(1.0, "a")], {DeviceID("a"): _st("a")}, _env(),
            pass_kind=MissionPass.DELIVER,
        )


def test_a_device_outside_the_admitted_set_is_refused():
    with pytest.raises(SelectorScopeViolation):
        OortPolicy().rank_contacts(
            [_wp(1.0, "leaked")], {DeviceID("leaked"): _st("leaked")}, _env(),
            admitted=[DeviceID("other")],
        )


# --------------------------------------------------------------------------- #
# 4. The stub guard — the failure this policy must NOT fail silently
# --------------------------------------------------------------------------- #

def test_repeatedly_served_devices_with_no_loss_raise():
    """Running B2 without --real-model must fail loudly, not rank on noise."""
    st_a = _st("a", loss=None, n=0, served_round=2, contact_ts=NOW - 5)
    st_a.on_time_count = 2
    states = {DeviceID("a"): st_a}
    with pytest.raises(OortUnusableError, match="real-model"):
        _rank([_wp(1.0, "a")], states)


def test_a_first_round_with_nobody_served_yet_is_fine():
    """Round 1 legitimately has no measurements — that is not the stub bug."""
    states = {DeviceID("a"): _st("a", contact_ts=0.0)}
    assert len(_rank([_wp(1.0, "a")], states)) == 1


def test_the_warm_up_round_is_tolerated():
    """Oort is retrospective: after ONE service the signal may still be in
    flight. Raising here killed a healthy run at mission 2 — the guard must
    distinguish "not yet" from "never"."""
    st_a = _st("a", loss=None, n=0, served_round=1, contact_ts=NOW - 5)
    st_a.on_time_count = 1
    assert len(_rank([_wp(1.0, "a")], {DeviceID("a"): st_a})) == 1


def test_repeated_FAILED_contacts_do_not_trip_the_guard():
    """Contacted twice, both sessions failed -> no loss, entirely legitimate.
    Keying the guard on contact count instead of successful participation
    killed healthy trials mid-sweep."""
    st_a = _st("a", loss=None, n=0, served_round=3, contact_ts=NOW - 5)
    st_a.on_time_count = 0           # contacted repeatedly, never succeeded
    assert len(_rank([_wp(1.0, "a")], {DeviceID("a"): st_a})) == 1


def test_signal_on_any_device_suppresses_the_guard():
    states = {
        DeviceID("a"): _st("a", loss=None, n=0, served_round=3),
        DeviceID("b"): _st("b", loss=0.4, n=10, served_round=3),
    }
    assert len(_rank([_wp(1.0, "a"), _wp(2.0, "b")], states)) == 2


def test_the_gradient_carries_the_loss_so_it_arrives_in_session():
    """The advertisement is built BEFORE training, so its loss describes the
    previous round. Carrying it on the gradient removes a full round of lag."""
    from hermes.types import GradientSubmission
    import numpy as np
    g = GradientSubmission(
        device_id=DeviceID("a"), mule_id=MuleID("m1"), mission_round=1,
        delta_theta=[np.zeros((2, 2), dtype=np.float32)], num_examples=64,
        submitted_at=0.0, local_loss=0.25,
    )
    assert g.local_loss == pytest.approx(0.25) and g.num_examples == 64


# --------------------------------------------------------------------------- #
# 5. The plumbing — and that it is inert for H0-H3
# --------------------------------------------------------------------------- #

def _delta(**kw):
    base = dict(
        device_id=DeviceID("a"), mule_id=MuleID("m1"), mission_round=3,
        outcome=MissionOutcome.CLEAN, utility=0.5, contact_ts=NOW,
    )
    base.update(kw)
    return RoundCloseDelta(**base)


def test_loss_and_sample_count_reach_the_scheduler_state():
    st = DeviceSchedulerState(device_id=DeviceID("a"))
    fold_round_close_delta(st, _delta(local_loss=0.42, num_examples=128))
    assert st.last_loss == pytest.approx(0.42)
    assert st.last_num_examples == 128


def test_the_served_round_is_recorded_for_the_staleness_term():
    st = DeviceSchedulerState(device_id=DeviceID("a"))
    fold_round_close_delta(st, _delta(mission_round=7))
    assert st.last_served_round == 7


def test_an_emitter_that_omits_the_fields_changes_nothing():
    """Inert for H0-H3 — every existing emitter sends neither field."""
    st = DeviceSchedulerState(device_id=DeviceID("a"))
    fold_round_close_delta(st, _delta())
    assert st.last_loss is None and st.last_num_examples == 0


def test_a_later_omission_does_not_wipe_an_earlier_measurement():
    """A delta without the fields must not erase what we already learned."""
    st = DeviceSchedulerState(device_id=DeviceID("a"))
    fold_round_close_delta(st, _delta(local_loss=0.9, num_examples=64))
    fold_round_close_delta(st, _delta())
    assert st.last_loss == pytest.approx(0.9) and st.last_num_examples == 64


def test_the_advertisement_carries_the_fields():
    from hermes.types import FLReadyAdv, FLState
    adv = FLReadyAdv(
        device_id=DeviceID("a"), state=FLState.FL_OPEN,
        performance_score=0.5, diversity_adjusted=0.5, utility=0.5,
        local_loss=0.33, num_examples=77,
    )
    assert adv.local_loss == pytest.approx(0.33) and adv.num_examples == 77


def test_the_advertisement_defaults_keep_old_emitters_valid():
    from hermes.types import FLReadyAdv, FLState
    adv = FLReadyAdv(
        device_id=DeviceID("a"), state=FLState.FL_OPEN,
        performance_score=0.5, diversity_adjusted=0.5, utility=0.5,
    )
    assert adv.local_loss is None and adv.num_examples == 0
