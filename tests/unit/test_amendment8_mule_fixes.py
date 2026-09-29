"""Freeze Amendment 8 and the Phase 0/1 audit fixes on the mule side.

* **In flight, a baseline route is held to the budget only.** D1/D2 replace
  S3, S3b and S3.5 (Amendment 4), but the in-flight re-check still ran S3b's
  per-device deadline test on their routes. MAX-AoI puts the most overdue
  devices first, so the check refused exactly the contacts it chose. A policy
  now declares what the mule re-checks (``in_flight_check``).
* **A plan's diagnostics belong to that plan.** ``last_feasibility`` survived
  an early return, so the mule widened the previous mission's dropped devices
  a second time.
* **An all-excluded mission keeps its sessions.** Under an age cutoff every
  update can arrive on time and still be refused; the trace must show those
  sessions, and which devices were actually merged.
"""

from __future__ import annotations

from types import SimpleNamespace

from hermes.mission.aggregation_rules import AGG_CUTOFF, AggregationSpec
from hermes.processes.mule import (
    _pass_1_collected,
    _pass_1_merged_devices,
    _pass_1_outcomes_payload,
    _pass_1_plan_payload,
)
from hermes.scheduler.fl_scheduler import FLScheduler
from hermes.scheduler.policies import MaxAoIPolicy, OortPolicy
from hermes.scheduler.policies.budget_walk import (
    IN_FLIGHT_BUDGET,
    IN_FLIGHT_NONE,
)
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    MissionOutcome,
    MissionRoundCloseReport,
    MissionSlice,
    MuleID,
    RoundCloseDelta,
)
from hermes.types.round_report import MissionRoundCloseLine

NOW = 1000.0
MODEL = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)


def _wp(x: float, deadline: float, *devs: str) -> ContactWaypoint:
    return ContactWaypoint(
        position=(x, 0.0, 0.0),
        devices=tuple(DeviceID(d) for d in devs),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=deadline,
    )


class _Sup:
    """The supervisor methods under test, bound onto a stand-in."""

    def __init__(self, scheduler, *, pose=(0.0, 0.0, 0.0), now=NOW,
                 aggregation=None, mission=None):
        self.scheduler = scheduler
        self.mule_pose = pose
        self.mule_id = MuleID("m1")
        self._now = lambda: now
        self.aggregation = aggregation or AggregationSpec()
        self.mission = mission

    from hermes.mule.mule_main import MuleSupervisor as _MS
    _remaining_is_feasible = _MS._remaining_is_feasible
    _widen_abandoned = _MS._widen_abandoned
    _excluded_only_report = _MS._excluded_only_report


def _scheduler(*device_ids: str, budget=None, selector=None):
    sch = FLScheduler(
        now_fn=lambda: NOW,
        mission_budget_s=budget,
        feasibility_model=MODEL,
        target_selector=selector,
    )
    sch.ingest_slice(MissionSlice(
        mule_id=MuleID("m1"),
        device_ids=tuple(DeviceID(d) for d in device_ids),
        issued_round=1,
        issued_at=NOW,
    ))
    return sch


# --------------------------------------------------------------------------- #
# 1. In-flight check per policy
# --------------------------------------------------------------------------- #

def test_the_baselines_declare_a_budget_only_check():
    assert MaxAoIPolicy.in_flight_check == IN_FLIGHT_BUDGET
    assert OortPolicy.in_flight_check == IN_FLIGHT_BUDGET


def test_an_overdue_contact_that_fits_the_budget_is_flown_under_max_aoi():
    """The scout's case: a stale device D1 routes first, deadline 140 s past."""
    sch = _scheduler("stale", budget=60.0, selector=MaxAoIPolicy())
    sup = _Sup(sch)
    overdue = [_wp(10.0, NOW - 140.0, "stale")]
    assert sup._remaining_is_feasible(overdue) is True


def test_our_own_arms_keep_the_deadline_test():
    """H0-H3 have no admit_and_order, so S3b's full check still applies."""
    sch = _scheduler("stale", budget=60.0)
    sup = _Sup(sch)
    assert sup._remaining_is_feasible([_wp(10.0, NOW - 140.0, "stale")]) is False


def test_a_baseline_still_stops_when_the_budget_is_spent():
    sch = _scheduler("far", budget=60.0, selector=MaxAoIPolicy())
    sup = _Sup(sch)
    # 100 m at 1 m/s from the mission start: 1100 > 1060.
    assert sup._remaining_is_feasible([_wp(100.0, 1e9, "far")]) is False


def test_a_policy_that_declares_no_check_flies_its_route():
    class _NoGates(MaxAoIPolicy):
        in_flight_check = IN_FLIGHT_NONE

    sch = _scheduler("far", budget=60.0, selector=_NoGates())
    sup = _Sup(sch)
    assert sup._remaining_is_feasible([_wp(100.0, NOW - 500.0, "far")]) is True


def test_a_policy_without_a_declaration_gets_the_budget_check():
    class _Bare:
        name = "BARE"

        def admit_and_order(self, contacts, device_states, env, *,
                            mission_deadline_ts=None, feasibility_model=None):
            return list(contacts)

    sch = _scheduler("a", budget=60.0, selector=_Bare())
    sup = _Sup(sch)
    assert sup._remaining_is_feasible([_wp(10.0, NOW - 5.0, "a")]) is True
    assert sup._remaining_is_feasible([_wp(100.0, 1e9, "a")]) is False


def test_the_scheduler_exposes_its_selector():
    policy = MaxAoIPolicy()
    assert FLScheduler(target_selector=policy).target_selector is policy
    assert FLScheduler().target_selector is None


# --------------------------------------------------------------------------- #
# 2. Abandoned devices are marked synthetic and unanswered
# --------------------------------------------------------------------------- #

def test_abandoned_devices_get_a_synthetic_unanswered_timeout():
    seen = []
    sch = _scheduler("a", budget=100.0)
    sch.ingest_round_close_delta = seen.append
    _Sup(sch)._widen_abandoned([_wp(5.0, 1e9, "a")], mission_round=3)
    assert len(seen) == 1
    delta: RoundCloseDelta = seen[0]
    assert delta.outcome is MissionOutcome.TIMEOUT
    assert delta.synthetic is True and delta.answered is False


def test_a_synthetic_timeout_is_not_a_reach_attempt():
    sch = _scheduler("a", budget=100.0)
    _Sup(sch)._widen_abandoned([_wp(5.0, 1e9, "a")], mission_round=3)
    st = sch.device_states[DeviceID("a")]
    assert (st.reach_attempts, st.reach_answered) == (0, 0)
    assert st.missed_count == 1          # still a miss for the deadline law


def test_the_scheduler_hands_the_mission_round_to_a_delegating_policy():
    seen = []

    class _Spy(MaxAoIPolicy):
        def admit_and_order(self, contacts, device_states, env, **kw):
            seen.append(env.mission_round)
            return list(contacts)

    sch = _scheduler("a", selector=_Spy())
    sch.set_mission_round(7)
    sch.build_contact_queue(rf_range_m=5.0)
    assert seen == [7] and sch.mission_round == 7
    sch.record_merged([DeviceID("a"), DeviceID("ghost")], 7)
    assert sch.device_states[DeviceID("a")].last_merged_round == 7


# --------------------------------------------------------------------------- #
# 3. A plan's diagnostics are reset with the plan
# --------------------------------------------------------------------------- #

def test_last_feasibility_does_not_survive_a_plan_with_no_eligible_devices():
    sch = _scheduler("a", "b", budget=10.0)
    for did, x in (("a", 5.0), ("b", 500.0)):
        sch.device_states[DeviceID(did)].last_known_position = (x, 0.0, 0.0)
    sch.build_contact_queue(rf_range_m=1.0)
    assert sch.last_feasibility is not None and sch.last_feasibility.n_dropped == 1
    assert set(sch.last_plan_deadlines) == {DeviceID("a"), DeviceID("b")}

    # Next plan: nobody is eligible (out of the slice, no beacon), so it
    # returns before S3b runs.
    for st in sch.device_states.values():
        st.is_in_slice = False
    assert sch.build_contact_queue(rf_range_m=1.0) == []
    assert sch.last_feasibility is None
    assert sch.last_plan_deadlines == {}


# --------------------------------------------------------------------------- #
# 4. An all-excluded mission keeps its ledger
# --------------------------------------------------------------------------- #

def _report_with_lines(lines):
    rep = MissionRoundCloseReport(mule_id=MuleID("m1"), mission_round=4,
                                  started_at=NOW, finished_at=NOW + 10.0)
    for did, oc in lines:
        rep.append(MissionRoundCloseLine(device_id=DeviceID(did), outcome=oc,
                                         contact_ts=NOW + 1.0))
    return rep


def test_the_excluded_only_report_is_kept_for_age_aware_rules():
    rep = _report_with_lines([("a", MissionOutcome.CLEAN),
                              ("b", MissionOutcome.TIMEOUT)])
    mission = SimpleNamespace(last_unmerged=(rep, None))
    cutoff = _Sup(None, aggregation=AggregationSpec(rule=AGG_CUTOFF, a_max=0),
                  mission=mission)
    assert cutoff._excluded_only_report() is rep
    # agg:plain empty rounds stay exactly as recorded (Rule 1).
    plain = _Sup(None, aggregation=AggregationSpec(), mission=mission)
    assert plain._excluded_only_report() is None


def test_a_mission_that_collected_nothing_has_no_excluded_report():
    rep = _report_with_lines([("a", MissionOutcome.TIMEOUT)])
    sup = _Sup(None, aggregation=AggregationSpec(rule=AGG_CUTOFF, a_max=0),
               mission=SimpleNamespace(last_unmerged=(rep, None)))
    assert sup._excluded_only_report() is None
    sup.mission = SimpleNamespace(last_unmerged=None)
    assert sup._excluded_only_report() is None


# --------------------------------------------------------------------------- #
# 5. mission_completed payloads
# --------------------------------------------------------------------------- #

def test_merged_devices_subtract_the_excluded_updates():
    rep = _report_with_lines([("a", MissionOutcome.CLEAN),
                              ("b", MissionOutcome.CLEAN),
                              ("c", MissionOutcome.TIMEOUT)])
    agg = SimpleNamespace(excluded_devices=(DeviceID("b"),))
    result = SimpleNamespace(empty=False, report=rep, aggregate=agg)
    assert _pass_1_merged_devices(result) == ["a"]
    # agg:plain excludes nothing: the merged list is exactly the CLEAN list.
    plain = SimpleNamespace(empty=False, report=rep,
                            aggregate=SimpleNamespace(excluded_devices=()))
    assert _pass_1_merged_devices(plain) == ["a", "b"]
    assert _pass_1_merged_devices(SimpleNamespace(empty=True, report=None)) == []
    assert _pass_1_merged_devices(SimpleNamespace(empty=False, report=None)) is None


def test_an_all_excluded_mission_records_its_sessions():
    rep = _report_with_lines([("a", MissionOutcome.CLEAN)])
    result = SimpleNamespace(empty=True, report=None, unmerged_report=rep)
    rows = _pass_1_outcomes_payload(result)
    assert [(r["device"], r["outcome"]) for r in rows] == [("a", "clean")]
    # An empty mission with nothing kept still records an empty list.
    assert _pass_1_outcomes_payload(
        SimpleNamespace(empty=True, report=None, unmerged_report=None)
    ) == []


def test_the_plan_records_each_members_own_deadline():
    queue = [_wp(1.0, NOW + 60.0, "a", "b")]
    plain = _pass_1_plan_payload(queue)
    assert plain == [{"devices": ["a", "b"], "deadline_ts": NOW + 60.0}]
    rich = _pass_1_plan_payload(
        queue, {DeviceID("a"): NOW + 60.0, DeviceID("b"): NOW + 300.0},
    )
    assert rich[0]["device_deadlines"] == {"a": NOW + 60.0, "b": NOW + 300.0}
    assert rich[0]["deadline_ts"] == NOW + 60.0


def test_collected_sessions_fall_back_to_the_kept_ledger_only_when_merged_nothing():
    rep = _report_with_lines([("a", MissionOutcome.CLEAN),
                              ("b", MissionOutcome.TIMEOUT)])
    # A normal mission reads its merge report.
    assert _pass_1_collected(SimpleNamespace(report=rep)) == (1, ["a"])
    # An all-excluded mission reads its kept ledger.
    assert _pass_1_collected(
        SimpleNamespace(report=None, unmerged_report=rep, empty=True)
    ) == (1, ["a"])
    # An agg:plain empty mission keeps no ledger: both stay None, as recorded.
    assert _pass_1_collected(
        SimpleNamespace(report=None, unmerged_report=None, empty=True)
    ) == (None, None)
