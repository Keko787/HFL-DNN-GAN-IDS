"""S3b — deadline feasibility gate.

Pins the behaviour that closes a real gap: before this stage, ``deadline_ts``
was computed by S3, propagated by S3a, and then **never compared to a clock**.
A device whose deadline had already passed was still queued and still visited,
so the scheduler's "deadline-aware" description was not true of the code.

The tests below fix the contract in both directions — the gate must bind when a
budget is configured, and must be a strict no-op when it is not, so that every
result recorded before it existed remains reproducible.
"""

from __future__ import annotations

from hermes.scheduler.stages.s3b_feasibility import (
    FeasibilityModel,
    filter_feasible,
)
from hermes.types.scheduler import Bucket, ContactWaypoint


def _wp(x: float, deadline: float, dev: str = "d") -> ContactWaypoint:
    return ContactWaypoint(
        position=(x, 0.0, 0.0),
        devices=(dev,),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=deadline,
    )


# --------------------------------------------------------------------------- #
# Off by default — previously-recorded results stay reproducible
# --------------------------------------------------------------------------- #

def test_no_budget_is_a_strict_noop():
    """With no mission budget the gate must not touch the queue at all."""
    contacts = [_wp(0.0, deadline=-1e9), _wp(1e6, deadline=-1e9)]  # both hopeless
    res = filter_feasible(contacts, now=0.0, mission_deadline_ts=None)
    assert res.kept == contacts          # same objects, same order
    assert res.n_dropped == 0


# --------------------------------------------------------------------------- #
# The gate binds
# --------------------------------------------------------------------------- #

def test_drops_a_contact_whose_deadline_cannot_be_reached():
    """A contact too far to reach before its own deadline is dropped."""
    model = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)
    near = _wp(1.0, deadline=100.0, dev="near")     # 1 s away, deadline 100 s
    far = _wp(50.0, deadline=5.0, dev="far")        # 50 s away, deadline 5 s
    res = filter_feasible(
        [near, far], now=0.0, mule_pose=(0.0, 0.0, 0.0),
        mission_deadline_ts=1000.0, model=model,
    )
    assert [c.devices for c in res.kept] == [("near",)]
    assert [c.devices for c in res.dropped_overdue] == [("far",)]
    assert not res.dropped_budget


def test_drops_a_contact_that_would_overrun_the_mission_budget():
    """Reachable in time, but no budget left to serve it."""
    model = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)
    a = _wp(3.0, deadline=1e9, dev="a")   # 3 s
    b = _wp(9.0, deadline=1e9, dev="b")   # +6 s -> 9 s total, over a 5 s budget
    res = filter_feasible(
        [a, b], now=0.0, mission_deadline_ts=5.0, model=model,
    )
    assert [c.devices for c in res.kept] == [("a",)]
    assert [c.devices for c in res.dropped_budget] == [("b",)]
    assert not res.dropped_overdue


def test_everything_feasible_is_kept():
    model = FeasibilityModel(cruise_speed_m_s=100.0, session_time_s=0.0)
    contacts = [_wp(1.0, 1e9, "a"), _wp(2.0, 1e9, "b"), _wp(3.0, 1e9, "c")]
    res = filter_feasible(contacts, now=0.0, mission_deadline_ts=1e9, model=model)
    assert len(res.kept) == 3
    assert res.n_dropped == 0


def test_ordering_is_edf_and_deterministic():
    """The greedy walk runs tightest-deadline-first, and repeatably."""
    model = FeasibilityModel(cruise_speed_m_s=1e6, session_time_s=0.0)
    contacts = [_wp(1.0, 30.0, "late"), _wp(2.0, 10.0, "early"), _wp(3.0, 20.0, "mid")]
    a = filter_feasible(contacts, now=0.0, mission_deadline_ts=1e9, model=model)
    b = filter_feasible(contacts, now=0.0, mission_deadline_ts=1e9, model=model)
    assert [c.devices[0] for c in a.kept] == ["early", "mid", "late"]
    assert [c.devices for c in a.kept] == [c.devices for c in b.kept]


def test_greedy_walk_accounts_for_earlier_stops():
    """Cost is cumulative: a stop that fits alone may not fit after another."""
    model = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)
    # Alone, the far contact costs 4 s and fits a 5 s budget. After serving
    # the near one (3 s) the mule is at x=3, so the remaining leg costs 1 s —
    # still fits. Push the budget to 3.5 s and only the first survives.
    near, far = _wp(3.0, 1e9, "near"), _wp(4.0, 1e9, "far")
    ok = filter_feasible([near, far], now=0.0, mission_deadline_ts=5.0, model=model)
    assert len(ok.kept) == 2

    tight = filter_feasible([near, far], now=0.0, mission_deadline_ts=3.5, model=model)
    assert [c.devices for c in tight.kept] == [("near",)]
    assert [c.devices for c in tight.dropped_budget] == [("far",)]


def test_expired_budget_drops_everything():
    model = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=1.0)
    contacts = [_wp(0.0, 1e9, "a")]
    res = filter_feasible(contacts, now=100.0, mission_deadline_ts=50.0, model=model)
    assert res.kept == []
    assert res.n_dropped == 1


# --------------------------------------------------------------------------- #
# Integration with FLScheduler — the gate runs BEFORE ordering
# --------------------------------------------------------------------------- #

def test_scheduler_gate_is_off_by_default():
    """No budget configured -> the scheduler must behave exactly as before."""
    from hermes.scheduler.fl_scheduler import FLScheduler

    assert FLScheduler()._mission_budget_s is None
    assert FLScheduler(mission_budget_s=30.0)._mission_budget_s == 30.0


def test_gate_precedes_the_selector_in_source():
    """The feasibility gate must be applied before the bucket/selector walk.

    Ordering matters architecturally: a hard constraint the learned selector
    could run *before* would let learning resurrect an infeasible contact.
    """
    import inspect

    from hermes.scheduler.fl_scheduler import FLScheduler

    src = inspect.getsource(FLScheduler.build_contact_queue)
    # Compare CALL sites, not prose — the docstring mentions rank_contacts.
    gate_call = src.index("filter_feasible(")
    selector_call = src.index("self._target_selector.rank_contacts(")
    assert gate_call < selector_call, (
        "the feasibility gate must run BEFORE the selector, so learning "
        "cannot resurrect a contact the gate dropped"
    )
