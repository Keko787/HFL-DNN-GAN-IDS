"""S3c — mission-level deadline-window adaptation.

The gap this closes: S3's adaptation is *per device*, so it can only ever learn
"this device was missed". It cannot see "the mule is systematically failing to
complete its circuit", because from any single device's point of view that
looks identical to being unlucky. S3c adds the mission-level signal and widens
every window together when the mule is falling short.

The most important tests here are the **inert-by-default** ones. Every recorded
sweep predates this stage, so the default path must produce byte-identical
deadlines — if that ever breaks, previously committed results silently stop
being reproducible.
"""

from __future__ import annotations

import pytest

from hermes.mule.mule_main import mission_planned_devices, mission_served_devices
from hermes.scheduler.fl_scheduler import FLScheduler
from hermes.scheduler.stages.s3_deadline import (
    MIN_DEADLINE_FULFILMENT_S,
    compute_deadline,
)
from hermes.scheduler.stages.s3c_mission_window import MissionWindowAdapter
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionSlice,
    MuleID,
)


def _state(fulfilment: float = 60.0, **kw) -> DeviceSchedulerState:
    return DeviceSchedulerState(
        device_id=DeviceID("d1"), deadline_fulfilment_s=fulfilment, **kw
    )


# --------------------------------------------------------------------------- #
# 1. Inert by default — the reproducibility guarantee
# --------------------------------------------------------------------------- #

def test_disabled_adapter_never_widens_however_bad_the_history():
    """Off means off — total mission failure still yields a scale of 1.0."""
    a = MissionWindowAdapter(enabled=False)
    for _ in range(10):
        a.record(0, 10)
    assert a.scale == 1.0


def test_scheduler_without_an_adapter_reports_unit_scale():
    assert FLScheduler().window_scale == 1.0


def test_recording_an_outcome_without_an_adapter_is_a_silent_no_op():
    FLScheduler().record_mission_outcome(served=0, planned=10)  # no raise


def test_default_deadline_is_byte_identical_to_the_unscaled_formula():
    """Pins the historical deadline against the new scale parameter."""
    st = _state(60.0)
    assert compute_deadline(st, now=1000.0, window_scale=1.0) == compute_deadline(
        st, now=1000.0
    )


# --------------------------------------------------------------------------- #
# 2. The scale as a function of history
# --------------------------------------------------------------------------- #

def test_no_history_yields_unit_scale():
    """Nothing has been observed yet, so there is nothing to correct."""
    assert MissionWindowAdapter(enabled=True).scale == 1.0


def test_meeting_the_target_yields_unit_scale():
    a = MissionWindowAdapter(enabled=True, target_success=0.8)
    a.record(8, 10)
    assert a.scale == 1.0


def test_exceeding_the_target_still_yields_unit_scale_and_never_shrinks():
    """S3c only widens; rewarding good behaviour is the per-device rule's job."""
    a = MissionWindowAdapter(enabled=True, target_success=0.8)
    a.record(10, 10)
    assert a.scale == 1.0


def test_falling_short_widens_in_proportion_to_the_shortfall():
    a = MissionWindowAdapter(enabled=True, target_success=0.8, gain=2.0)
    a.record(3, 10)                       # 0.5 short of target
    assert a.scale == pytest.approx(1.0 + 2.0 * 0.5)


def test_a_worse_mission_widens_further_than_a_milder_one():
    mild = MissionWindowAdapter(enabled=True)
    bad = MissionWindowAdapter(enabled=True)
    mild.record(7, 10)
    bad.record(1, 10)
    assert bad.scale > mild.scale > 1.0


def test_widening_is_capped():
    """An impossible configuration degrades to 'wide', never to 'unbounded'."""
    a = MissionWindowAdapter(enabled=True, gain=100.0, max_scale=3.0)
    a.record(0, 10)
    assert a.scale == 3.0


def test_the_window_is_rolling_so_recovery_is_visible():
    """Old failures must age out, or a bad patch would widen windows forever."""
    a = MissionWindowAdapter(enabled=True, window=2, target_success=0.8)
    a.record(0, 10)
    assert a.scale > 1.0
    a.record(10, 10)
    a.record(10, 10)                      # the failure has now aged out
    assert a.scale == 1.0


def test_the_rate_is_pooled_over_missions_not_averaged_over_ratios():
    """A 100-device mission must outweigh a 1-device one."""
    a = MissionWindowAdapter(enabled=True)
    a.record(0, 100)
    a.record(1, 1)
    assert a.success_rate == pytest.approx(1 / 101)


def test_the_scale_is_a_pure_function_of_history():
    """Same record sequence -> same scale, regardless of when it was read."""
    a, b = MissionWindowAdapter(enabled=True), MissionWindowAdapter(enabled=True)
    for served in (2, 9, 4):
        a.record(served, 10)
        _ = a.scale                        # reading must not perturb state
        b.record(served, 10)
    assert a.scale == b.scale


def test_missions_with_nothing_planned_are_ignored():
    """An empty queue is not evidence of failure."""
    a = MissionWindowAdapter(enabled=True)
    a.record(0, 0)
    assert a.n_missions == 0 and a.scale == 1.0


def test_reset_clears_the_history():
    a = MissionWindowAdapter(enabled=True)
    a.record(0, 10)
    a.reset()
    assert a.n_missions == 0 and a.scale == 1.0


@pytest.mark.parametrize("kw", [
    {"window": 0},
    {"target_success": 1.5},
    {"target_success": -0.1},
    {"min_scale": 0.5},
    {"max_scale": 0.5},
])
def test_invalid_configuration_is_rejected_at_construction(kw):
    with pytest.raises(ValueError):
        MissionWindowAdapter(**kw)


# --------------------------------------------------------------------------- #
# 3. How the scale reaches the deadline
# --------------------------------------------------------------------------- #

def test_the_scale_stretches_the_fulfilment_term():
    st = _state(60.0)
    assert compute_deadline(st, now=1000.0, window_scale=2.0) == 1000.0 + 120.0


def test_idle_time_is_subtracted_after_scaling_not_before():
    """The scale widens the window; it must not also inflate the idle penalty."""
    st = _state(60.0, idle_time_ref_ts=900.0)
    scaled = compute_deadline(st, now=1000.0, window_scale=2.0)
    idle = 1000.0 - 900.0                  # 100 s since last on-time participation
    assert scaled == pytest.approx(1000.0 + 60.0 * 2.0 - idle)


def test_the_floor_is_applied_before_scaling():
    """A sub-floor window is raised to the floor, then widened."""
    st = _state(1.0)                       # below MIN_DEADLINE_FULFILMENT_S
    got = compute_deadline(st, now=0.0, window_scale=2.0)
    assert got == pytest.approx(MIN_DEADLINE_FULFILMENT_S * 2.0)


def test_a_cluster_override_still_wins_over_the_mission_scale():
    """The slow-phase amendment stays authoritative — S3c must not override it."""
    st = _state(60.0, deadline_override_ts=5555.0)
    assert compute_deadline(st, now=1000.0, window_scale=4.0) == 5555.0


# --------------------------------------------------------------------------- #
# 4. Scheduler integration
# --------------------------------------------------------------------------- #

def _scheduler_with(adapter, *, now=1000.0):
    sch = FLScheduler(now_fn=lambda: now, mission_window_adapter=adapter)
    sch.ingest_slice(MissionSlice(
        mule_id=MuleID("m1"),
        device_ids=(DeviceID("a"),),
        issued_round=1,
        issued_at=now,
    ))
    return sch


def test_the_scheduler_surfaces_the_adapters_scale():
    a = MissionWindowAdapter(enabled=True, target_success=0.8, gain=2.0)
    sch = _scheduler_with(a)
    assert sch.window_scale == 1.0
    sch.record_mission_outcome(served=3, planned=10)
    assert sch.window_scale == pytest.approx(2.0)


def test_a_disabled_adapter_attached_to_the_scheduler_stays_inert():
    """Attaching the object is not the toggle — `enabled` is."""
    sch = _scheduler_with(MissionWindowAdapter(enabled=False))
    sch.record_mission_outcome(served=0, planned=10)
    assert sch.window_scale == 1.0


def test_a_broken_adapter_cannot_kill_the_mission():
    """Bookkeeping is never worth losing a sortie over."""
    class _Exploding:
        scale = 1.0

        def record(self, served, planned):
            raise RuntimeError("boom")

    FLScheduler(mission_window_adapter=_Exploding()).record_mission_outcome(
        served=1, planned=2
    )  # no raise


def test_widening_shows_up_in_the_deadlines_the_scheduler_builds():
    """End to end: a failed mission history pushes the contact queue's deadlines out."""
    a = MissionWindowAdapter(enabled=True, target_success=0.8, gain=2.0)
    sch = _scheduler_with(a)
    before = sch.build_contact_queue(
        rf_range_m=50.0, mule_pose=(0.0, 0.0, 0.0), mule_energy=1.0,
    )
    sch.record_mission_outcome(served=1, planned=10)
    after = sch.build_contact_queue(
        rf_range_m=50.0, mule_pose=(0.0, 0.0, 0.0), mule_energy=1.0,
    )
    assert before and after
    assert after[0].deadline_ts > before[0].deadline_ts, (
        "a systematically failing mule must give every device more room, "
        "or S3b keeps dropping the same devices forever"
    )


# --------------------------------------------------------------------------- #
# 5. The mission accounting that feeds it
# --------------------------------------------------------------------------- #

class _Feas:
    def __init__(self, overdue=(), budget=()):
        self.dropped_overdue = tuple(overdue)
        self.dropped_budget = tuple(budget)


def _c(*devs) -> ContactWaypoint:
    return ContactWaypoint(
        position=(0.0, 0.0, 0.0),
        devices=tuple(DeviceID(d) for d in devs),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=0.0,
    )


def test_planned_counts_devices_not_contacts():
    """Contacts cluster several devices; the FL unit is the device."""
    assert mission_planned_devices([_c("a", "b"), _c("c")]) == 3


def test_planned_includes_what_s3b_dropped_before_take_off():
    """The gate must not be able to hide its own drops from the success rate."""
    queue = [_c("a")]
    feas = _Feas(overdue=[_c("b", "c")], budget=[_c("d")])
    assert mission_planned_devices(queue, feas) == 4


def test_a_gate_that_drops_almost_everything_does_not_score_full_marks():
    """The exact failure this denominator exists to prevent."""
    queue, feas = [_c("j")], _Feas(budget=[_c(*"abcdefghi")])
    served = mission_served_devices(queue)
    planned = mission_planned_devices(queue, feas)
    assert served / planned == pytest.approx(0.1), (
        "serving 1 of 10 intended devices must read as 10 % success, not 100 %"
    )


def test_planned_tolerates_a_scheduler_with_no_feasibility_result():
    """Enforcement off means there is no feasibility object at all."""
    assert mission_planned_devices([_c("a", "b")], None) == 2


def test_served_excludes_the_abandoned_tail():
    queue = [_c("a"), _c("b"), _c("c")]
    assert mission_served_devices(queue, aborted=queue[1:]) == 1


def test_served_equals_planned_on_a_clean_uninterrupted_mission():
    queue = [_c("a", "b"), _c("c")]
    assert mission_served_devices(queue) == mission_planned_devices(queue) == 3


def test_an_immediate_abort_serves_nothing():
    queue = [_c("a"), _c("b")]
    assert mission_served_devices(queue, aborted=queue) == 0


def test_a_fully_aborted_mission_drives_the_scale_to_its_cap():
    """The two halves compose: accounting -> adapter -> widened windows."""
    a = MissionWindowAdapter(enabled=True, gain=2.0, max_scale=2.5)
    queue = [_c("a"), _c("b")]
    a.record(mission_served_devices(queue, aborted=queue),
             mission_planned_devices(queue))
    assert a.success_rate == 0.0
    # 1 + 2.0 x 0.8 = 2.6, clamped by max_scale.
    assert a.scale == 2.5
