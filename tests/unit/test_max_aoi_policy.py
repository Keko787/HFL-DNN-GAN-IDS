"""MAX-AoI baseline policy (arm B1).

The Age-of-Information greedy comparator: always go to whoever has been waiting
longest. A standard named baseline in the AoI / UAV-data-collection literature,
and the closest published rival to our bucket+deadline ordering that runs on
state a data mule actually has.

These tests pin the properties that make it a *fair* baseline rather than a
flattering one: it must rank on age (not on our deadline), it must treat a
never-served device as maximally stale, and it must be deterministic so a paired
comparison against H1 is reproducible.
"""

from __future__ import annotations

import pytest

from hermes.scheduler.policies import MaxAoIPolicy, contact_age
from hermes.scheduler.policies.max_aoi import NEVER_SERVED_AGE
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.selector.scope_guard import SelectorScopeViolation
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionPass,
)

NOW = 10_000.0


def _wp(x: float, *devs: str) -> ContactWaypoint:
    return ContactWaypoint(
        position=(x, 0.0, 0.0),
        devices=tuple(DeviceID(d) for d in devs),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=0.0,
    )


def _states(**last_seen: float):
    """device id -> state, with `last_contact_ts` set from kwargs."""
    return {
        DeviceID(d): DeviceSchedulerState(device_id=DeviceID(d), last_contact_ts=ts)
        for d, ts in last_seen.items()
    }


def _env(pose=(0.0, 0.0, 0.0), now=NOW) -> SelectorEnv:
    return SelectorEnv(
        mule_pose=pose, mule_energy=1.0, rf_prior_snr_db=20.0,
        beacon_window_s=30.0, now=now,
    )


def _rank(cands, states, **kw):
    return MaxAoIPolicy().rank_contacts(cands, states, _env(**kw))


# --------------------------------------------------------------------------- #
# 1. Age is the ranking signal
# --------------------------------------------------------------------------- #

def test_the_stalest_contact_goes_first():
    a, b, c = _wp(1.0, "a"), _wp(2.0, "b"), _wp(3.0, "c")
    st = _states(a=NOW - 10, b=NOW - 500, c=NOW - 100)
    assert [w.devices[0] for w in _rank([a, b, c], st)] == ["b", "c", "a"]


def test_a_never_served_device_outranks_every_finite_age():
    """Never served = maximally stale. Also gives the 'explore' behaviour."""
    fresh, never = _wp(1.0, "fresh"), _wp(2.0, "never")
    st = _states(fresh=NOW - 9_000, never=0.0)
    assert _rank([fresh, never], st)[0].devices[0] == "never"


def test_never_served_age_is_infinite():
    st = _states(x=0.0)
    assert contact_age(_wp(0.0, "x"), st, NOW) == NEVER_SERVED_AGE


def test_a_device_missing_from_state_is_treated_as_never_served():
    """Unknown device must not silently rank as fresh."""
    assert contact_age(_wp(0.0, "ghost"), {}, NOW) == NEVER_SERVED_AGE


def test_age_never_goes_negative_on_a_clock_skew():
    st = _states(x=NOW + 500)          # last contact "in the future"
    assert contact_age(_wp(0.0, "x"), st, NOW) == 0.0


# --------------------------------------------------------------------------- #
# 2. A contact's age is its STALEST member
# --------------------------------------------------------------------------- #

def test_contact_age_is_the_max_not_the_mean():
    """A neglected device must not hide behind well-served neighbours."""
    st = _states(fresh=NOW - 1, stale=NOW - 900)
    assert contact_age(_wp(0.0, "fresh", "stale"), st, NOW) == 900.0


def test_a_cluster_containing_one_starved_device_is_prioritised():
    mixed = _wp(1.0, "fresh1", "fresh2", "starved")
    moderate = _wp(2.0, "mid")
    st = _states(fresh1=NOW - 5, fresh2=NOW - 5, starved=NOW - 800, mid=NOW - 300)
    assert _rank([moderate, mixed], st)[0] is mixed


# --------------------------------------------------------------------------- #
# 3. Distance is the tie-break — the "nearest predecessor" rule
# --------------------------------------------------------------------------- #

def test_equal_age_breaks_toward_the_nearer_contact():
    near, far = _wp(5.0, "near"), _wp(500.0, "far")
    st = _states(near=NOW - 100, far=NOW - 100)
    assert _rank([far, near], st)[0].devices[0] == "near"


def test_distance_never_overrides_age():
    """A close-but-fresh contact must not jump a distant starved one."""
    near_fresh, far_stale = _wp(1.0, "nf"), _wp(9_999.0, "fs")
    st = _states(nf=NOW - 1, fs=NOW - 5_000)
    assert _rank([near_fresh, far_stale], st)[0].devices[0] == "fs"


def test_tie_break_uses_the_CURRENT_mule_pose():
    left, right = _wp(-100.0, "l"), _wp(100.0, "r")
    st = _states(l=NOW - 50, r=NOW - 50)
    assert _rank([left, right], st, pose=(-90.0, 0.0, 0.0))[0].devices[0] == "l"
    assert _rank([left, right], st, pose=(90.0, 0.0, 0.0))[0].devices[0] == "r"


# --------------------------------------------------------------------------- #
# 4. Contract: determinism, totality, guards
# --------------------------------------------------------------------------- #

def test_order_is_deterministic_and_independent_of_input_order():
    """A paired comparison against H1 must be reproducible."""
    a, b, c = _wp(1.0, "a"), _wp(2.0, "b"), _wp(3.0, "c")
    st = _states(a=NOW - 10, b=NOW - 10, c=NOW - 10)   # fully tied on age
    first = [w.devices[0] for w in _rank([a, b, c], st)]
    for perm in ([c, a, b], [b, c, a], [c, b, a]):
        assert [w.devices[0] for w in _rank(perm, st)] == first


def test_every_candidate_is_returned_exactly_once():
    """Ordering only — a baseline may not drop or duplicate work."""
    cands = [_wp(float(i), f"d{i}") for i in range(6)]
    st = _states(**{f"d{i}": NOW - i * 10 for i in range(6)})
    got = _rank(cands, st)
    assert len(got) == len(cands)
    assert {id(w) for w in got} == {id(w) for w in cands}


def test_empty_candidate_list_is_fine():
    assert _rank([], {}) == []


def test_pass_2_invocation_is_refused():
    """Pass 1 only, same contract as the other ranking policies."""
    with pytest.raises(SelectorScopeViolation):
        MaxAoIPolicy().rank_contacts(
            [_wp(1.0, "a")], _states(a=NOW), _env(),
            pass_kind=MissionPass.DELIVER,
        )


def test_a_device_outside_the_admitted_set_is_refused():
    """The scope guard must fire here exactly as it does for the RL selector."""
    with pytest.raises(SelectorScopeViolation):
        MaxAoIPolicy().rank_contacts(
            [_wp(1.0, "leaked")], _states(leaked=NOW), _env(),
            admitted=[DeviceID("only_this_one")],
        )


def test_it_exposes_the_shared_policy_surface():
    """It must be swappable through the same slot as the other policies."""
    from hermes.scheduler.policies import ArrivalOrderPolicy
    assert hasattr(MaxAoIPolicy(), "rank_contacts")
    assert MaxAoIPolicy().name != ArrivalOrderPolicy().name


# --------------------------------------------------------------------------- #
# 5. It really is a DIFFERENT policy from ours
# --------------------------------------------------------------------------- #

def test_it_ignores_the_deadline_that_our_scheduler_ranks_on():
    """If it tracked our deadline it would not be an independent baseline."""
    urgent_deadline = ContactWaypoint(
        position=(1.0, 0.0, 0.0), devices=(DeviceID("fresh"),),
        bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=1.0,   # very urgent
    )
    slack_deadline = ContactWaypoint(
        position=(2.0, 0.0, 0.0), devices=(DeviceID("stale"),),
        bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=1e9,   # no urgency
    )
    st = _states(fresh=NOW - 1, stale=NOW - 900)
    got = _rank([urgent_deadline, slack_deadline], st)
    assert got[0].devices[0] == "stale", (
        "MAX-AoI must rank on age alone; ranking on deadline_ts would make it "
        "our own policy wearing a baseline's name"
    )
