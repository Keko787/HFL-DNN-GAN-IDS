"""In-flight abort + the deadline-feedback loop for unreached devices.

Two gaps this pins, both found by inspection rather than by a failing test:

1. **The mule flew doomed queues.** S3b filtered the queue before take-off, but
   nothing re-checked en route — so once the mule fell behind its own plan
   (a slow contact, a failed one), it kept flying stops it could no longer serve
   in time, burning budget and delaying delivery of the updates already aboard.

2. **Unreached devices got no feedback at all.** ``RoundCloseDelta`` is emitted
   only from inside a contact session, so a device that is dropped by S3b or
   abandoned by an abort never widens its fulfilment window — leaving it just as
   un-serveable next mission. That starvation loop was introduced by the S3b
   gate itself.

Both fixes are inert unless deadline enforcement is on, which these tests also
pin, because every recorded sweep predates them.
"""

from __future__ import annotations

from hermes.scheduler.fl_scheduler import FLScheduler
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
from hermes.scheduler.stages.s3_deadline import (
    FAST_PHASE_MISSED_WIDEN_S,
)
from hermes.types import Bucket, ContactWaypoint, DeviceID, MissionSlice, MuleID


def _wp(x: float, deadline: float, *devs: str) -> ContactWaypoint:
    return ContactWaypoint(
        position=(x, 0.0, 0.0),
        devices=tuple(DeviceID(d) for d in devs),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=deadline,
    )


class _Sup:
    """The two supervisor methods under test, without a live process tree.

    MuleSupervisor needs real RF/dock links to construct, so we bind the
    unbound methods onto a stand-in carrying only the attributes they touch.
    """

    def __init__(self, scheduler, *, pose=(0.0, 0.0, 0.0), now=1000.0):
        self.scheduler = scheduler
        self.mule_pose = pose
        self.mule_id = MuleID("m1")
        self._now = lambda: now

    # bind the real implementations
    from hermes.mule.mule_main import MuleSupervisor as _MS
    _remaining_is_feasible = _MS._remaining_is_feasible
    _widen_abandoned = _MS._widen_abandoned


def _scheduler(*device_ids: str, budget=None, now=1000.0):
    sch = FLScheduler(
        now_fn=lambda: now,
        mission_budget_s=budget,
        feasibility_model=FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0),
    )
    sch.ingest_slice(MissionSlice(
        mule_id=MuleID("m1"),
        device_ids=tuple(DeviceID(d) for d in device_ids),
        issued_round=1,
        issued_at=now,
    ))
    return sch


# --------------------------------------------------------------------------- #
# 1. In-flight feasibility
# --------------------------------------------------------------------------- #

def test_no_budget_never_aborts():
    """Inert without enforcement — every recorded sweep ran this way."""
    sup = _Sup(_scheduler("a", budget=None))
    hopeless = [_wp(1e6, -1e9, "a")]
    assert sup._remaining_is_feasible(hopeless) is True


def test_empty_remainder_is_not_feasible():
    sup = _Sup(_scheduler("a", budget=100.0))
    assert sup._remaining_is_feasible([]) is False


def test_aborts_when_the_next_contact_is_unreachable_in_time():
    """The mule has fallen behind: the next stop cannot be made by its deadline."""
    sch = _scheduler("a", budget=1000.0, now=1000.0)
    sup = _Sup(sch, pose=(0.0, 0.0, 0.0), now=1000.0)
    # 500 m away at 1 m/s = 500 s, but the deadline is 10 s out.
    assert sup._remaining_is_feasible([_wp(500.0, 1010.0, "a")]) is False


def test_continues_when_the_next_contact_is_still_reachable():
    sch = _scheduler("a", budget=1000.0, now=1000.0)
    sup = _Sup(sch, pose=(0.0, 0.0, 0.0), now=1000.0)
    assert sup._remaining_is_feasible([_wp(5.0, 1e9, "a")]) is True


def test_decision_is_taken_from_the_current_pose():
    """Feasibility must be judged from where the mule IS, not from origin."""
    sch = _scheduler("a", budget=1000.0, now=1000.0)
    target = _wp(100.0, 1050.0, "a")
    far = _Sup(sch, pose=(0.0, 0.0, 0.0), now=1000.0)      # 100 s away, deadline 50 s
    near = _Sup(sch, pose=(95.0, 0.0, 0.0), now=1000.0)    # 5 s away
    assert far._remaining_is_feasible([target]) is False
    assert near._remaining_is_feasible([target]) is True


# --------------------------------------------------------------------------- #
# 2. The starvation fix
# --------------------------------------------------------------------------- #

def test_abandoned_devices_get_their_window_widened():
    """An unreached device must still receive the 'missed' signal."""
    sch = _scheduler("a", "b", budget=100.0)
    sup = _Sup(sch)
    before = sch.device_states[DeviceID("a")].deadline_fulfilment_s

    sup._widen_abandoned([_wp(1.0, 1e9, "a", "b")], mission_round=1)

    for d in ("a", "b"):
        st = sch.device_states[DeviceID(d)]
        assert st.deadline_fulfilment_s == before + FAST_PHASE_MISSED_WIDEN_S, (
            "an abandoned device's fulfilment window must widen, or S3b can "
            "drop it again forever"
        )
        assert st.missed_count == 1


def test_widening_is_cumulative_across_missions():
    """Repeated starvation keeps widening, so the device eventually fits."""
    sch = _scheduler("a", budget=100.0)
    sup = _Sup(sch)
    start = sch.device_states[DeviceID("a")].deadline_fulfilment_s
    for r in (1, 2, 3):
        sup._widen_abandoned([_wp(1.0, 1e9, "a")], mission_round=r)
    assert sch.device_states[DeviceID("a")].deadline_fulfilment_s == (
        start + 3 * FAST_PHASE_MISSED_WIDEN_S
    )


def test_widening_never_raises_on_an_unknown_device():
    """Bookkeeping must never kill the sortie."""
    sch = _scheduler("a", budget=100.0)
    sup = _Sup(sch)
    sup._widen_abandoned([_wp(1.0, 1e9, "ghost")], mission_round=1)  # no raise
