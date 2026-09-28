"""Freeze Amendment 6 — every mission's budget runs from its own start.

The S3b budget clock used to start only in ``FLScheduler.ingest_slice``, which
runs when a DOWN bundle arrives. A DOWN arrives mid-mission (the inter-pass
dock) and not at all after an empty mission, which skips the dock, so the next
mission planned against a stale stamp. In the recorded 60 s runs 72–81% of
missions were empty, and in one D1 trial the budget left at planning had fallen
to about 39 s by mission 4. The mule now starts the clock at the start of every
mission through ``FLScheduler.start_mission``.
"""

from __future__ import annotations

from hermes.scheduler.fl_scheduler import FLScheduler
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
from hermes.types import Bucket, ContactWaypoint, DeviceID, MissionSlice, MuleID


class _Clock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _scheduler(clock: _Clock, *, budget: float, positions=None) -> FLScheduler:
    """A scheduler whose slice (the bootstrap DOWN) arrived at ``clock()``."""
    positions = positions or {"a": (0.0, 0.0, 0.0)}
    sch = FLScheduler(
        now_fn=clock,
        mission_budget_s=budget,
        feasibility_model=FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0),
    )
    sch.ingest_slice(MissionSlice(
        mule_id=MuleID("m1"),
        device_ids=tuple(DeviceID(d) for d in positions),
        issued_round=1,
        issued_at=clock(),
    ))
    for d, pos in positions.items():
        sch.device_states[DeviceID(d)].last_known_position = pos
    return sch


def _wp(x: float, *devs: str, deadline: float = 1e9) -> ContactWaypoint:
    return ContactWaypoint(
        position=(x, 0.0, 0.0),
        devices=tuple(DeviceID(d) for d in devs),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=deadline,
    )


class _Mule:
    """The supervisor's mission entry point without a live process tree.

    ``MuleSupervisor`` needs real RF and dock links to construct, so the real
    methods are bound onto a stand-in. Each fake mission is empty: it takes
    ``mission_s`` seconds and never docks, so no DOWN (and no
    ``ingest_slice``) happens before the next one.
    """

    from hermes.mule.mule_main import MuleSupervisor as _MS
    run_one_mission = _MS.run_one_mission
    _remaining_is_feasible = _MS._remaining_is_feasible

    def __init__(self, scheduler, clock, *, rf_range_m=60.0, mission_s=9.0):
        self.scheduler = scheduler
        self._now = clock
        self._clock = clock
        self.rf_range_m = rf_range_m
        self.mule_pose = (0.0, 0.0, 0.0)
        self._next_theta = object()      # an empty mission re-stages θ
        self._mission_s = mission_s
        self.stamps = []

    def _fly_empty_mission(self, path: str):
        self.stamps.append((path, self.scheduler._mission_start_ts))
        self._clock.t += self._mission_s

    def _run_two_pass_mission(self):
        self._fly_empty_mission("two_pass")

    def _run_single_pass_mission(self):
        self._fly_empty_mission("single_pass")


# --------------------------------------------------------------------------- #
# The scheduler's clock
# --------------------------------------------------------------------------- #

def test_start_mission_restarts_the_clock_and_returns_the_stamp():
    clock = _Clock(1000.0)
    sch = _scheduler(clock, budget=60.0)
    clock.t = 1040.0
    assert sch.start_mission() == 1040.0
    assert sch._mission_start_ts == 1040.0


def test_ingest_slice_still_stamps_for_callers_without_a_mule():
    clock = _Clock(1000.0)
    sch = _scheduler(clock, budget=60.0)
    assert sch._mission_start_ts == 1000.0


# --------------------------------------------------------------------------- #
# The mule starts every mission's clock
# --------------------------------------------------------------------------- #

def test_every_mission_stamps_its_own_start_even_without_a_dock():
    clock = _Clock(1000.0)
    sch = _scheduler(clock, budget=60.0)      # bootstrap DOWN at t = 1000
    clock.t = 1002.0
    mule = _Mule(sch, clock)
    for _ in range(3):                        # three empty missions, no DOWN
        mule.run_one_mission()
    # Before the fix all three planned against the bootstrap stamp, 1000.
    assert [ts for _, ts in mule.stamps] == [1002.0, 1011.0, 1020.0]


def test_the_single_pass_path_is_stamped_too():
    clock = _Clock(1000.0)
    sch = _scheduler(clock, budget=60.0)
    clock.t = 1005.0
    mule = _Mule(sch, clock, rf_range_m=None)
    mule.run_one_mission()
    assert mule.stamps == [("single_pass", 1005.0)]


# --------------------------------------------------------------------------- #
# What the fresh stamp changes
# --------------------------------------------------------------------------- #

def test_planning_after_an_empty_mission_gets_the_full_budget():
    """40 s after the last DOWN, a stop 30 s away does not fit the 20 s the
    stale stamp leaves, but fits the new mission's full 60 s."""
    clock = _Clock(1000.0)
    sch = _scheduler(clock, budget=60.0, positions={"a": (30.0, 0.0, 0.0)})
    clock.t = 1040.0

    def plan():
        return sch.build_contact_queue(
            rf_range_m=10.0, mule_pose=(0.0, 0.0, 0.0), mule_energy=1.0,
        )

    assert plan() == []                       # stale: 1040 + 30 > 1000 + 60
    sch.start_mission()
    assert [wp.devices for wp in plan()] == [(DeviceID("a"),)]


def test_the_in_flight_check_runs_from_the_mission_start():
    clock = _Clock(1000.0)
    sch = _scheduler(clock, budget=60.0)
    clock.t = 1040.0
    mule = _Mule(sch, clock)
    next_stop = [_wp(30.0, "a")]
    assert mule._remaining_is_feasible(next_stop) is False   # 1070 > 1060
    sch.start_mission()
    assert mule._remaining_is_feasible(next_stop) is True    # 1070 <= 1100
