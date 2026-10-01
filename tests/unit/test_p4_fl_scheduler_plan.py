"""FeRRy Phase 4 (unit U5): the scheduler fork, ``hermes/scheduler/fl_scheduler.py``.

Plan mode (``plan_mode="ferry"``, Phase 4 spec, other choices 1) plans each
mission at the dock with ``FLScheduler.build_ferry_plan``: S1 and S3, the age
cap (U1), S3a per band class, the search (U4) scored by V (U2) over member
subsets (U3), a guard fold and the commit. Member-subset admission reaches the
H and D arms through ``build_contact_queue`` (the user's decision 4 (b),
``unit_U3b.md`` section 5.2), and the D arms report what they leave out
(decision 6). Pinned here:

* **Legacy identity (Freeze Rule 1).** At its defaults the scheduler equals
  6e6f92d's on random instances of every arm, both clocks and every in-flight
  call (the recorded module loaded from git), and with the Phase 4 switches set
  explicitly to their recorded values it equals the defaults and replays the
  supervisor goldens and the Phase 3 simulated-clock trials; a legacy
  scheduler never loads the plan package; the source-order pins hold.
* **The switches.** ``member_admission`` and ``plan_mode`` refuse what they
  cannot honour (unit_U3b.md section 1.4; design section 2.2; spec items 9 and
  10), and plan mode binds each class model to the scheduler's device states.
* **The plan path.** S1 and S3 match the legacy path (equal deadlines and
  buckets); the commit is the search's best on the inputs the spec names (S3a
  at each class's radius, Pass 2 priced per class, the cap and the weights);
  the committed class's model is the one the in-flight check prices with; the
  predicted whole mission is recorded for every arm (R4); a capped plan
  without its mission round raises before anything is built (B9, R1); the
  guard folds the final route with its own exempt stops (B1) and a route the
  predicate refuses is never committed; every demanded device left out is
  labelled by the first clause that refuses it alone (spec item 8), priced
  from the plan's takeoff clock, its complements dated by B2, and the cap's
  plan-time violations are recorded, the lookahead with them; an empty demand
  commits the empty plan; FB+c searches and flies class c only; Pass 2 is
  priced from the dock; ``whole`` flies whole stops, and its labels judge
  whole stops (R5) while its violations judge the device alone, as under
  ``subset`` (the user's decision of 2026-09-30: ``unplannable`` is physics,
  for the time budget; under an energy capacity see test_p4_hover.py);
  a capped device its S3a stop cannot serve alone is offered its hover stop,
  where its label is priced (``tests/unit/test_p4_hover.py`` pins the rule);
  the plan is deterministic and the wall time stays out of it; neither the
  pre-flight order check nor the T_nom helper enters plan mode.
* **After the plan.** ``plan_protected`` and ``close_plan``; the plan-mode
  re-plan is U3's member trim under ``subset`` and a trim of whole stops under
  ``whole`` (R5), never ``replan_route``, and ``reorder`` is refused.
* **The H and D arms.** On the Phase 3 cliff ``subset`` admits the members
  that fit before takeoff, and only then: the in-flight re-plan never reduces a
  contact; the pre-flight order check's drops of reduced stops join
  ``last_feasibility``; a D arm's ``last_policy_drops`` is set and reset at
  every early return (critic A9) while ``last_feasibility`` stays None, and
  each drop is labelled as the arm's in-flight re-plan labels it (decision 6).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes.mule.mule_main as mule_main
import hermes.scheduler.fl_scheduler as FLS
from experiments.exp4.topology_builder import device_positions
from hermes.l1.mission_clock import MissionClock
from hermes.mule.ferry import FerryRuntime, FerrySpec
from hermes.mule.mule_main import mission_planned_devices
from hermes.scheduler import FLScheduler, FLSchedulerError
from hermes.scheduler.plan import (
    PLAN_MODE_LEGACY,
    PLAN_SCORE_KEYS,
    REASON_PLAN,
    SEARCH_EXACT,
    AgeCapSpec,
    Candidate,
    MemberFold,
    PlanClass,
    PlanOptions,
    PlanScoreParams,
    PlanSearchParams,
    PlanSetup,
    SearchResult,
)
from hermes.scheduler.plan import member_subset as MS
from hermes.scheduler.plan import plan_search as PS
from hermes.scheduler.plan import types as PT
from hermes.scheduler.plan.plan_score import demand_weights, predicted_mission_s
from hermes.scheduler.policies import (
    FedCSDegradedPolicy,
    FedExCarpPolicy,
    MaxAoIPolicy,
    OortPolicy,
    WhittlePolicy,
)
from hermes.scheduler.policies.budget_walk import left_out
from hermes.scheduler.policies.fedcs_degraded import VALUE_DEVICES, VALUE_UNIT
from hermes.scheduler.routing import replan as RP
from hermes.scheduler.stages import filter_eligible
from hermes.scheduler.stages import s3b_feasibility as S3B
from hermes.scheduler.stages.s3a_cluster import cluster_by_rf_range
from hermes.scheduler.stages.s3b_feasibility import (
    DEADLINE_BOUNDS,
    REASON_BUDGET,
    REASON_ENERGY,
    REASON_OVERDUE,
    RULE_DEADLINE_BUDGET,
    FeasibilityModel,
    FerryPhysics,
    FlightState,
    MemberSubsets,
)
from hermes.scheduler.stages.s3d_age_cap import (
    cap_stops,
    evaluate_cap,
    is_exempt,
    priority_first,
    with_cap_deadlines,
)
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionPass,
    MissionSlice,
    MuleID,
)
from hermes.types.registry import DeviceRecord, SpectrumSig
from hermes.types.scheduler import CAP_CROWDED, CAP_UNPLANNABLE, CapViolation

from tests.golden import _build_p3_sim as P3SIM
from tests.golden import _canon
from tests.golden import _mule_harness as M
from tests.unit import _p4_ref as REF

REPO = Path(__file__).resolve().parents[2]
DOCK = (0.0, 0.0, 0.0)
COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
P_MOVE, P_HOVER = 143.6, 168.5

#: The Phase 3 cliff instance (tests/unit/test_p3_final_fixes_mule.py): trial
#: T2's layout, narrow, 1 MB each way, the seconds backhaul.
T2_LAYOUT = device_positions(8, 777, 100.0)
CLIFF_IDS = tuple(DeviceID(f"dev-{i}") for i in range(len(T2_LAYOUT)))
#: T2's deadline unit: Deadline(j) = t0 + 1500 s, so the deadline never binds.
T2_UNIT = 25.0
T_NOM = 200.0


def _in_stop_order(stop, numbers):
    wanted = {DeviceID(f"dev-{i}") for i in numbers}
    return tuple(d for d in stop.devices if d in wanted)


def _devices(route):
    return [tuple(wp.devices) for wp in route]


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _t2_spec(band="narrow"):
    return FerrySpec.from_config(rf_range_m=60.0, seed=777, contact_band=band,
                                 backhaul_model="seconds", backhaul_period=750.0,
                                 backhaul_regime="clean", payload_bytes=1_000_000)


def _records(layout=T2_LAYOUT, ids=CLIFF_IDS):
    return [DeviceRecord(device_id=d, last_known_position=(x, y, 0.0),
                         spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)))
            for d, (x, y) in zip(ids, layout)]


def _cliff(budget, *, time_scale=T2_UNIT, selector=None, **kw):
    """H1's (or, with ``selector``, a D arm's) Pass-1 plan of trial T2 at takeoff,
    test_p3_final_fixes_mule._plan with the Phase 4 keywords ``kw``."""
    spec = _t2_spec("narrow")
    clock = MissionClock()
    sch = FLScheduler(now_fn=clock, mission_budget_s=budget,
                      feasibility_model=spec.feasibility_model(rf_range_m=60.0, theta_bytes=52),
                      deadline_time_scale=time_scale, refuse_deadline_overrides=True,
                      target_selector=selector, **kw)
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=CLIFF_IDS, issued_round=0,
                                  issued_at=clock()), registry_records=_records())
    sch.start_mission()
    sch.set_mission_round(1)
    queue = sch.build_contact_queue(rf_range_m=spec.link.range_planar_m("narrow"), mule_pose=DOCK)
    return queue, sch


def _runtime(reference="wide"):
    rt = FerryRuntime(_t2_spec(reference), None, rf_range_m=60.0)
    rt.set_payload(theta_bytes=52)
    return rt


def _t2_setup(*, reference="wide", policy="search", admission="subset", s=None, lookahead=0,
              score=None, search=None):
    """The mule's plan setup on T2's physics: the runtime's own classes (U6)."""
    rt = _runtime(reference)
    options = PlanOptions(band_class_policy=policy, member_admission=admission,
                          cap=AgeCapSpec(s_missions=s, lookahead=lookahead),
                          score=score or PlanScoreParams(), search=search or PlanSearchParams())
    setup = PlanSetup(options=options, classes=rt.plan_classes(), reference=reference,
                      t_ref_s=T_NOM, turnaround_s=rt.spec.flight.turnaround_s)
    return rt, setup


def _t2_ferry(budget, *, mission_round=1, time_scale=T2_UNIT, merged=None, streaks=None,
              extra=(), **setup_kw):
    """A plan-mode scheduler on trial T2's layout, as the mule builds it:
    the reference class's model, the simulated clock, ``replan`` with ``trim``.
    ``merged`` sets devices' ``last_merged_round`` and ``streaks`` their miss
    streak; ``extra`` adds tracked devices outside the slice (``(id, (x, y))``)."""
    rt, setup = _t2_setup(**setup_kw)
    clock = MissionClock()
    sch = FLScheduler(now_fn=clock, mission_budget_s=budget,
                      feasibility_model=rt.feasibility_model(),
                      deadline_time_scale=time_scale, refuse_deadline_overrides=True,
                      validate_flown_order=True, replan_fallback="trim", miss_priority=True,
                      member_admission=setup.options.member_admission, plan_mode="ferry",
                      plan=setup)
    records = _records() + _records(layout=[xy for _, xy in extra], ids=[d for d, _ in extra])
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=CLIFF_IDS, issued_round=0,
                                  issued_at=clock()), registry_records=records)
    for did, rnd in (merged or {}).items():
        sch.device_states[DeviceID(did)].last_merged_round = rnd
    for did, streak in (streaks or {}).items():
        sch.device_states[DeviceID(did)].miss_streak = streak
    sch.start_mission()
    sch.set_mission_round(mission_round)
    return sch, rt


class _Dwell:
    """A member's dwell: ``a + b*d`` seconds, ``d`` its planar distance to the stop."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, d, pass_kind, offset):
        return self.a + self.b * d


class _Upload:
    def __call__(self):
        return 0.5


class _NoOutage:
    def __call__(self, d):
        return 0.0


NOW = 1000.0
#: A reference time far enough back that ``idle_time_ref_ts`` puts a deadline
#: of Φ = 300 s at NOW - 600 s (compute_idle_time ignores a stamp <= 0).
PAST = 100.0


def _synthetic_setup(*, radius, a, b, s, capacity=None, admission="subset", dock=DOCK):
    """One class of synthetic physics, speed 5 m/s, a 0.5 s upload."""
    physics = FerryPhysics(dock=dock, member_dwell_s=_Dwell(a, b), upload_s=_Upload(),
                           p_move_w=P_MOVE, p_hover_w=P_HOVER, energy_capacity_j=capacity,
                           range_m=radius)
    model = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=1.0, ferry=physics)
    cls = PlanClass(name="wide", index=0, radius_m=radius, model=model, outage=_NoOutage())
    options = PlanOptions(member_admission=admission, cap=AgeCapSpec(s_missions=s))
    return PlanSetup(options=options, classes=(cls,), reference="wide", t_ref_s=300.0,
                     turnaround_s=30.0)


def _synthetic(setup, devices, *, budget, mission_round=5):
    """A plan-mode scheduler at NOW on ``devices``: ``{id: (position, Φ,
    idle reference, last_merged_round)}``, every device in the slice, not new."""
    sch = FLScheduler(now_fn=lambda: NOW, mission_budget_s=budget, replan_fallback="trim",
                      member_admission=setup.options.member_admission, plan_mode="ferry",
                      plan=setup, miss_priority=True)
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=tuple(devices), issued_round=0,
                                  issued_at=NOW))
    for did, (pos, phi, idle_ref, merged) in devices.items():
        st = sch.device_states[did]
        st.last_known_position = pos
        st.is_new = False
        st.deadline_fulfilment_s = phi
        st.idle_time_ref_ts = idle_ref
        st.last_merged_round = merged
    sch.start_mission()
    sch.set_mission_round(mission_round)
    return sch


#: Spec item 8's labels, one device each (S = 3 at mission 5: never merged is
#: capped, merged in mission 4 is not). Budget 150 s, capacity 15 kJ.
LABELLED = {
    DeviceID("n"): ((15.0, 0.0, 0.0), 300.0, 0.0, 4),     # served
    DeviceID("o"): ((15.0, 6.0, 0.0), 300.0, PAST, 4),    # n's stop; overdue alone
    DeviceID("k"): ((200.0, 0.0, 0.0), 300.0, 0.0, None),  # capped; served
    DeviceID("p"): ((-200.0, 0.0, 0.0), 300.0, 0.0, 4),   # fits alone; k takes the budget
    DeviceID("e"): ((0.0, 300.0, 0.0), 300.0, 0.0, 4),    # over the capacity alone
    DeviceID("b"): ((0.0, -400.0, 0.0), 300.0, 0.0, 4),   # over the budget alone
    DeviceID("c"): ((500.0, 0.0, 0.0), 300.0, PAST, None),  # capped, late, over the budget alone
}

#: Critic B1: a capped member already late (``a``) whose stop's uncapped
#: co-member (``u``) does not fit beside it. Budget 60 s.
B1_CASE = {
    DeviceID("a"): ((40.0, 0.0, 0.0), 300.0, PAST, None),
    DeviceID("u"): ((52.0, 0.0, 0.0), 300.0, 0.0, 4),
}

#: One S3a stop, two left-out members with labels of their own (85 s): ``x``
#: is already late, ``y`` fits alone but not beside the capped ``k``, whom the
#: plan serves.
SPLIT_CASE = {
    DeviceID("x"): ((15.0, 0.0, 0.0), 300.0, PAST, 4),
    DeviceID("y"): ((15.0, 6.0, 0.0), 300.0, 0.0, 4),
    DeviceID("k"): ((-200.0, 0.0, 0.0), 300.0, 0.0, None),
}

#: A mixed stop left out (critic B2): the capped ``c`` is already late, the
#: uncapped ``v`` is not, and each fits the 60 s budget alone at their S3a
#: stop, but the budget flies the older capped ``k`` (and ``n``), so ``c``
#: and ``v`` are left out by the plan's choice, together. The complement must
#: carry ``v``'s deadline, not ``c``'s.
B2_CASE = {
    DeviceID("n"): ((15.0, 0.0, 0.0), 300.0, 0.0, 4),
    DeviceID("k"): ((-100.0, 0.0, 0.0), 300.0, 0.0, None),
    DeviceID("c"): ((100.0, 0.0, 0.0), 300.0, PAST, 1),
    DeviceID("v"): ((105.0, 0.0, 0.0), 300.0, 0.0, 4),
}

#: The hover rule splits a mixed stop (the user's decision of 2026-09-30): the
#: capped ``c``, already late, and the uncapped ``v`` share a stop 200 m out,
#: and both are over the 60 s budget alone there; ``n`` is served.
HOVER_SPLIT = {
    DeviceID("n"): ((15.0, 0.0, 0.0), 300.0, 0.0, 4),
    DeviceID("c"): ((200.0, 0.0, 0.0), 300.0, PAST, None),
    DeviceID("v"): ((205.0, 0.0, 0.0), 300.0, 0.0, 4),
}

#: Under ``whole`` at plan time (150 s): the capped ``k`` shares a stop with
#: the uncapped ``o``, already late, so the stop is late for its uncapped
#: member (B2) and no whole plan flies ``k``; ``n`` is served.
WHOLE_MIXED = {
    DeviceID("k"): ((15.0, 0.0, 0.0), 300.0, 0.0, None),
    DeviceID("o"): ((15.0, 6.0, 0.0), 300.0, PAST, 4),
    DeviceID("n"): ((-15.0, 0.0, 0.0), 300.0, 0.0, 4),
}

#: Under ``whole`` at plan time (100 s, one stop's worth): the capped ``k``
#: (never merged, the oldest), the capped ``c1`` alone and late (an exempt
#: stop), and the capped ``c2``, late, with the uncapped ``v`` (a mixed stop).
WHOLE_CAP = {
    DeviceID("k"): ((150.0, 0.0, 0.0), 300.0, 0.0, None),
    DeviceID("c1"): ((-150.0, 0.0, 0.0), 300.0, PAST, 1),
    DeviceID("c2"): ((0.0, 150.0, 0.0), 300.0, PAST, 1),
    DeviceID("v"): ((0.0, 156.0, 0.0), 300.0, 0.0, 4),
}

#: Under ``whole`` in flight (100 s): the exempt ``k`` (capped, already
#: late), the mixed stop of the capped ``m1`` and the uncapped ``m2`` (due 20
#: s after takeoff) and the uncapped ``u``. The plan flies all three.
WHOLE_FLIGHT = {
    DeviceID("u"): ((30.0, 0.0, 0.0), 300.0, 0.0, 4),
    DeviceID("k"): ((-30.0, 0.0, 0.0), 300.0, PAST, None),
    DeviceID("m1"): ((0.0, 30.0, 0.0), 300.0, 0.0, None),
    DeviceID("m2"): ((0.0, 36.0, 0.0), 20.0, 0.0, 4),
}


# --------------------------------------------------------------------------- #
# Random legacy instances (every arm, both clocks)
# --------------------------------------------------------------------------- #

ARMS = ("H1", "H1-prio", "H1-reorder", "H1-trim", "D1", "D2", "D3", "D4", "D5-unit",
        "D5-devices")
D_ARMS = {
    "D1": MaxAoIPolicy,
    "D2": OortPolicy,
    "D3": WhittlePolicy,
    "D4": FedExCarpPolicy,
    "D5-unit": lambda: FedCSDegradedPolicy(VALUE_UNIT),
    "D5-devices": lambda: FedCSDegradedPolicy(VALUE_DEVICES),
}


class _CaseDwell:
    """``a + b*d`` seconds collecting, half delivering; below the floor beyond ``floor_m``."""

    def __init__(self, a, b, floor_m):
        self.a, self.b, self.floor_m = a, b, floor_m

    def __call__(self, d, pass_kind, offset):
        if d > self.floor_m:
            return None
        t = self.a + self.b * d
        return 0.5 * t if pass_kind is DELIVER else t


class _CaseUpload:
    def __init__(self, s):
        self.s = s

    def __call__(self):
        return self.s


def _case(seed, *, clock=None, arms=ARMS):
    """A random scheduler instance: devices in and out of the slice, new and
    not, beacon-heard, S3-refused with a stale bucket, a range of deadlines,
    ages and outcomes; the simulated clock's ferry physics (every deadline
    bound, an energy capacity now and then) or the wall clock's legacy model;
    a budget or none; the arm and its switches. The arms with the pre-flight
    order check get the dense fields of U5's pre-flight build probe (every
    device in the slice, deadlines within a few minutes, budgets near a
    mission's length), where H1's bucket-then-distance order often overruns
    what S3b's EDF order fits, so the check fires."""
    rng = random.Random(7_654_321 * seed + 3)
    now = rng.choice((1000.0, 5000.0))
    span = rng.choice((25.0, 70.0, 160.0))
    clock = clock or rng.choice(("sim", "sim", "sim", "wall"))
    arm = rng.choice(arms)
    dense = arm in ("H1-reorder", "H1-trim") and clock == "sim"
    devices = {}
    for i in range(rng.randint(3, 9) if dense else rng.choice((0, 1, 2, 3, 4, 5, 6, 7, 9))):
        in_slice = dense or rng.random() < 0.85
        new = rng.random() < (0.4 if dense else 0.35)
        beacon = now - rng.uniform(0.0, 20.0) if rng.random() < 0.2 else 0.0
        override = None
        stale = False
        if not in_slice and rng.random() < 0.5:
            override = now + rng.uniform(-50.0, 200.0)
            stale = not new and not beacon and rng.random() < 0.7
        devices[DeviceID(f"d{i}")] = dict(
            pos=(round(rng.uniform(-span, span), 2), round(rng.uniform(-span, span), 2), 0.0),
            in_slice=in_slice, new=new, beacon=beacon, override=override, stale=stale,
            phi=(rng.uniform(20.0, 300.0) if dense
                 else rng.choice((5.0, 30.0, 60.0, 120.0, 400.0))),
            idle_ref=0.0 if dense else rng.choice((0.0, 0.0, now - rng.uniform(1.0, 300.0))),
            missed=rng.randint(0, 4), streak=rng.randint(0, 3),
            merged=rng.choice((None, 0, 1, 2)), clean=rng.randint(0, 2),
            served=rng.randint(0, 3), utility=rng.uniform(0.0, 2.0),
            loss=rng.choice((None, rng.uniform(0.1, 2.0))), examples=rng.randint(0, 50),
            priority=rng.choice((0, 0, 0, 1)),
        )
    radius = rng.choice((20.0, 40.0, 60.0) if dense else (10.0, 30.0, 60.0))
    if clock == "sim":
        physics = FerryPhysics(
            dock=DOCK, member_dwell_s=_CaseDwell(rng.uniform(0.3, 4.0), rng.uniform(0.0, 0.2),
                                                 math.inf if dense
                                                 else rng.choice((math.inf, 0.8 * radius))),
            upload_s=_CaseUpload(rng.choice((0.0, 0.5, 3.0))), p_move_w=P_MOVE,
            p_hover_w=P_HOVER,
            energy_capacity_j=None if rng.random() < 0.7 else rng.uniform(5e3, 6e4),
            deadline_bounds=rng.choice(DEADLINE_BOUNDS),
            range_m=rng.choice((None, radius)),
        )
        model = FeasibilityModel(cruise_speed_m_s=5.0 if dense else rng.choice((2.0, 5.0, 12.0)),
                                 session_time_s=1.0, ferry=physics)
    else:
        model = rng.choice((None, FeasibilityModel(cruise_speed_m_s=rng.uniform(1.0, 10.0),
                                                   session_time_s=rng.uniform(0.2, 3.0))))
    budget = rng.uniform(20.0, 250.0) if dense else rng.uniform(5.0, 300.0)
    return SimpleNamespace(
        seed=seed, now=now, clock=clock, arm=arm, devices=devices, radius=radius, model=model,
        budget=None if rng.random() < 0.15 else budget,
        time_scale=1.0 if dense else rng.choice((1.0, 1.0, 25.0)), prio=arm == "H1-prio",
        validate=dense, fallback="trim" if arm == "H1-trim" else "reorder",
        mission_round=rng.choice((None, 1, 3)),
        flight=(rng.uniform(0.0, 60.0), rng.uniform(0.0, 3e3), rng.random() < 0.5),
    )


def _drive(cls, case, **switches):
    """Build ``cls`` (the live scheduler or 6e6f92d's) on ``case`` and make
    every call the mule makes of it before and during a mission; return what
    each call gave and the scheduler."""
    t = {"now": case.now}
    policy = D_ARMS[case.arm]() if case.arm in D_ARMS else None
    sch = cls(now_fn=lambda: t["now"], mission_budget_s=case.budget, feasibility_model=case.model,
              target_selector=policy, deadline_time_scale=case.time_scale,
              miss_priority=case.prio, validate_flown_order=case.validate,
              replan_fallback=case.fallback, **switches)
    in_slice = tuple(d for d, s in case.devices.items() if s["in_slice"])
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=in_slice, issued_round=1,
                                  issued_at=case.now),
                     registry_records=_records(layout=[s["pos"][:2] for s in case.devices.values()],
                                               ids=list(case.devices)))
    for did, s in case.devices.items():
        st = sch.device_states[did]
        st.is_new = s["new"]
        st.last_beacon_ts = s["beacon"]
        st.deadline_override_ts = s["override"]
        st.bucket = Bucket.SCHEDULED_THIS_ROUND if s["stale"] else None
        st.deadline_fulfilment_s = s["phi"]
        st.idle_time_ref_ts = s["idle_ref"]
        st.missed_count = s["missed"]
        st.miss_streak = s["streak"]
        st.last_merged_round = s["merged"]
        st.last_clean_round = s["clean"]
        st.last_served_round = s["served"]
        st.last_utility = s["utility"]
        st.last_loss = s["loss"]
        st.last_num_examples = s["examples"]
        st.delivery_priority = s["priority"]
    sch.start_mission()
    sch.set_mission_round(case.mission_round)
    queue = sch.build_contact_queue(rf_range_m=case.radius, mule_pose=DOCK)
    out = {
        "queue": list(queue),
        "feasibility": sch.last_feasibility,
        "order_check": sch.last_order_check,
        "deadlines": dict(sch.last_plan_deadlines),
        "states": {d: dataclasses.replace(st) for d, st in sch.device_states.items()},
        "start": sch.mission_start_ts,
    }
    dt, energy, moved = case.flight
    pose = queue[0].position if (queue and moved) else DOCK
    state = FlightState(tuple(pose), case.now + dt, energy)
    end = None if case.budget is None else case.now + case.budget
    t["now"] = case.now + dt
    pass_2 = sch.build_pass_2_queue(rf_range_m=case.radius, mule_pose=DOCK)
    out["pass_2"] = list(pass_2)
    for kind, route in ((COLLECT, queue), (DELIVER, pass_2)):
        out[f"rule_{kind.value}"] = sch.in_flight_rule(kind)
        out[f"fold_{kind.value}"] = sch.fold_remainder(route, state=state, budget_end=end,
                                                       pass_kind=kind)
        out[f"replan_{kind.value}"] = sch.replan_remainder(route, state=state, budget_end=end,
                                                           pass_kind=kind)
    return out, sch


# --------------------------------------------------------------------------- #
# The switches
# --------------------------------------------------------------------------- #

def test_the_switch_values_are_the_plan_packages():
    """The recorded values are restated in the scheduler and in S3b so that a
    legacy scheduler never loads the plan package; they must agree with it."""
    assert FLS._PLAN_MODE_LEGACY == PLAN_MODE_LEGACY == PT.PLAN_MODES[0] == "legacy"
    assert S3B.MEMBER_ADMISSIONS == PT.MEMBER_ADMISSIONS == ("whole", "subset")
    sig = inspect.signature(FLScheduler)
    assert sig.parameters["member_admission"].default == S3B.MEMBER_ADMISSION_WHOLE
    assert sig.parameters["plan_mode"].default == PLAN_MODE_LEGACY
    assert sig.parameters["plan"].default is None


def test_at_its_defaults_the_scheduler_is_the_recorded_one():
    sch = FLScheduler()
    assert (sch.plan_mode, sch.plan_setup, sch.member_admission) == ("legacy", None, "whole")
    assert (sch.last_plan, sch.last_policy_drops, sch.last_plan_wall_s) == (None, [], None)
    assert sch.plan_protected([]) == frozenset()
    with pytest.raises(FLSchedulerError, match="build_contact_queue"):
        sch.build_ferry_plan()
    with pytest.raises(FLSchedulerError, match="no plan to close"):
        sch.close_plan([], [])


def _ferry_model():
    physics = FerryPhysics(dock=DOCK, member_dwell_s=_Dwell(1.0, 0.0), upload_s=_Upload(),
                           p_move_w=P_MOVE, p_hover_w=P_HOVER)
    return FeasibilityModel(ferry=physics)


class _WholeDouble:
    """A whole-scheduler test double that never declared the carrier."""

    name = "DOUBLE"

    def admit_and_order(self, contacts, device_states, env, *, mission_deadline_ts=None,
                        feasibility_model=None):
        return list(contacts)


class _RankOnly:
    """A learned-selector double (H2/H3): ranks contacts, never admits them."""

    def rank_contacts(self, candidates, device_states, env, *, pass_kind, admitted=None):
        return list(candidates)


@pytest.mark.parametrize("kwargs, match", [
    (dict(member_admission="bogus"), "member_admission must be one of"),
    (dict(member_admission=None), "member_admission must be one of"),
    (dict(member_admission="Subset"), "member_admission must be one of"),
    (dict(member_admission="subset"), "ferry physics"),
    (dict(member_admission="subset", feasibility_model=FeasibilityModel()), "ferry physics"),
    (dict(member_admission="subset", feasibility_model=_ferry_model(),
          target_selector=FedExCarpPolicy()), "FEDEX"),
    (dict(member_admission="subset", feasibility_model=_ferry_model(),
          target_selector=_WholeDouble()), "DOUBLE"),
])
def test_member_admission_refuses_what_it_cannot_honour(kwargs, match):
    """unit_U3b.md section 1.4: an unknown value; ``subset`` without the ferry
    physics its member walk prices with (the wall clock); ``subset`` with a
    whole-scheduler policy whose walk does not take the carrier (D4, a double)."""
    with pytest.raises(FLSchedulerError, match=match):
        FLScheduler(**kwargs)


@pytest.mark.parametrize("selector", [None, _RankOnly(), MaxAoIPolicy(), OortPolicy(),
                                      WhittlePolicy(), FedCSDegradedPolicy(VALUE_UNIT)])
def test_member_admission_subset_is_taken_where_it_can_act(selector):
    sch = FLScheduler(member_admission="subset", feasibility_model=_ferry_model(),
                      target_selector=selector)
    assert sch.member_admission == "subset" and sch.plan_mode == "legacy"
    assert FLScheduler(member_admission="whole", target_selector=FedExCarpPolicy()) \
        .member_admission == "whole"


def test_plan_mode_refuses_what_it_cannot_honour():
    """Design section 2.2 and spec items 9 and 10: ``plan`` only with
    ``ferry``; ``ferry`` needs a PlanSetup, takes no selector, re-plans by
    trimming (``reorder`` refused), and takes its member admission from the
    scheduler's one switch; one mule flies from one dock."""
    _, setup = _t2_setup()
    ok = dict(replan_fallback="trim", member_admission="subset")
    FLScheduler(plan_mode="ferry", plan=setup, **ok)
    cases = [
        (dict(plan=setup), "plan_mode='ferry'"),
        (dict(plan_mode="ferry", **ok), "PlanSetup"),
        (dict(plan_mode="ferry", plan=setup.options, **ok), "PlanSetup"),
        (dict(plan_mode="bogus", plan=setup, **ok), "plan_mode must be one of"),
        (dict(plan_mode="Ferry", plan=setup, **ok), "plan_mode must be one of"),
        (dict(plan_mode="ferry", plan=setup, target_selector=MaxAoIPolicy(), **ok),
         "target_selector"),
        (dict(plan_mode="ferry", plan=setup, member_admission="subset"), "'reorder' is refused"),
        (dict(plan_mode="ferry", plan=setup, replan_fallback="trim"), "one source"),
        (dict(plan_mode="ferry", plan=setup, replan_fallback="trim", member_admission="bogus"),
         "member_admission must be one of"),
    ]
    for kwargs, match in cases:
        with pytest.raises(FLSchedulerError, match=match):
            FLScheduler(**kwargs)
    _, whole = _t2_setup(admission="whole")
    with pytest.raises(FLSchedulerError, match="one source"):
        FLScheduler(plan_mode="ferry", plan=whole, **ok)
    assert FLScheduler(plan_mode="ferry", plan=whole, replan_fallback="trim").member_admission \
        == "whole"
    far = setup.classes[1]
    moved = dataclasses.replace(far, model=dataclasses.replace(
        far.model, ferry=dataclasses.replace(far.model.ferry, dock=(1.0, 0.0, 0.0))))
    split = dataclasses.replace(setup, classes=(setup.classes[0], moved) + setup.classes[2:])
    with pytest.raises(FLSchedulerError, match="one dock"):
        FLScheduler(plan_mode="ferry", plan=split, **ok)


def test_plan_mode_binds_each_class_model_to_the_schedulers_device_states():
    """As the Phase 3 model is bound (critic B11): the committed model prices
    members where they are. The caller's setup is left as it was, and a class
    bound to a map of its own keeps it."""
    _, setup = _t2_setup()
    sch = FLScheduler(plan_mode="ferry", plan=setup, replan_fallback="trim",
                      member_admission="subset")
    assert sch.plan_mode == "ferry" and sch.plan_setup is not setup
    assert [c.name for c in sch.plan_setup.classes] == [c.name for c in setup.classes]
    for bound, given in zip(sch.plan_setup.classes, setup.classes):
        assert bound.model.ferry.device_states is sch.device_states
        assert given.model.ferry.device_states is None
        assert (bound.index, bound.radius_m, bound.outage) == (given.index, given.radius_m,
                                                                given.outage)
    own = {}
    first = setup.classes[0]
    kept = dataclasses.replace(first, model=dataclasses.replace(
        first.model, ferry=first.model.ferry.bind(own)))
    sch = FLScheduler(plan_mode="ferry", replan_fallback="trim", member_admission="subset",
                      plan=dataclasses.replace(setup, classes=(kept,) + setup.classes[1:]))
    assert sch.plan_setup.classes[0].model.ferry.device_states is own


# --------------------------------------------------------------------------- #
# Legacy identity (Freeze Rule 1)
# --------------------------------------------------------------------------- #

REF_NAME = f"hermes.scheduler._p4ref_{REF.COMMIT}_fl_scheduler"


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """6e6f92d's ``fl_scheduler.py``, loaded from git under a private name
    beside the live module, so its relative imports are the live package's:
    the stages and walks it calls are the live ones, which U3b pins equal to
    6e6f92d's without the carrier (``tests/unit/test_p4_member_subset_hd.py``).
    Skipped only when git or the commit is missing (``_p4_ref.unavailable``)."""
    why = REF.unavailable()
    if why is not None:
        pytest.skip(f"the {REF.COMMIT} reference cannot be loaded: {why}")
    blob = subprocess.run(["git", "show", f"{REF.COMMIT}:hermes/scheduler/fl_scheduler.py"],
                          cwd=REPO, capture_output=True, timeout=120)
    assert blob.returncode == 0, blob.stderr
    path = tmp_path_factory.mktemp("ref") / "fl_scheduler.py"
    path.write_bytes(blob.stdout)
    spec = importlib.util.spec_from_file_location(REF_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[REF_NAME] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(REF_NAME, None)


def _same(live, ref, case):
    assert live.keys() == ref.keys()
    for key in live:
        assert live[key] == ref[key], (case.seed, case.arm, case.clock, key)


def test_the_reference_is_the_recorded_scheduler(recorded):
    src = inspect.getsource(recorded)
    assert "build_ferry_plan" not in src and "last_policy_drops" not in src
    params = inspect.signature(recorded.FLScheduler).parameters
    assert not {"member_admission", "plan_mode", "plan"} & set(params)
    assert recorded.FLScheduler is not FLScheduler and recorded.__name__ == REF_NAME


@pytest.mark.parametrize("block", range(2))
def test_at_its_defaults_the_scheduler_equals_6e6f92ds_on_random_instances(recorded, block):
    """Every arm (H1 with and without the miss priority, the pre-flight check
    under both fallbacks, D1-D5), both clocks, with and without a budget:
    ``build_contact_queue`` and its diagnostics, the device states it leaves,
    Pass 2, the in-flight rule, the departure check and the re-plan in both
    passes all equal the recorded scheduler's. The live scheduler's only
    additions: an empty ``last_policy_drops`` unless a D arm plans on the
    simulated clock, and no plan."""
    seen = dict.fromkeys(("sim", "wall", "gate drops", "policy drops", "order check",
                          "pass 1 replan", "pass 2 replan", "refused by S3"), 0)
    for seed in range(1000 * block, 1000 * (block + 1)):
        case = _case(seed)
        live, sch = _drive(FLScheduler, case)
        ref, _ = _drive(recorded.FLScheduler, case)
        _same(live, ref, case)
        assert sch.last_plan is None and sch.plan_mode == "legacy"
        if case.arm not in D_ARMS or case.clock == "wall":
            assert sch.last_policy_drops == []
        seen[case.clock] += 1
        seen["gate drops"] += bool(getattr(live["feasibility"], "n_dropped", 0))
        seen["policy drops"] += bool(sch.last_policy_drops)
        seen["order check"] += bool(live["order_check"] and live["order_check"].changed)
        seen["pass 1 replan"] += live["replan_collect"].changed
        seen["pass 2 replan"] += live["replan_deliver"].changed
        seen["refused by S3"] += any(s["stale"] for s in case.devices.values())
    assert min(seen.values()) >= 3, seen


def test_the_t_nom_helper_equals_6e6f92ds(recorded):
    spec = _t2_spec("wide")
    model = spec.feasibility_model(rf_range_m=60.0, theta_bytes=52)
    for seed in (3, 17, 41, 777):
        layout = {DeviceID(f"d{i}"): (x, y, 0.0)
                  for i, (x, y) in enumerate(device_positions(6, seed, 100.0))}
        for band in ("wide", "narrow"):
            kw = dict(rf_range_m=spec.link.range_planar_m(band), feasibility_model=model,
                      turnaround_s=30.0)
            assert FLS.nominal_mission_period_s(layout, **kw) == \
                recorded.nominal_mission_period_s(layout, **kw)


def test_the_phase_4_switches_set_to_their_recorded_values_equal_the_defaults():
    """``member_admission="whole"``, ``plan_mode="legacy"`` and ``plan=None``,
    passed explicitly, change nothing, D arms' reports included."""
    for seed in range(600):
        case = _case(seed)
        default, a = _drive(FLScheduler, case)
        explicit, b = _drive(FLScheduler, case, member_admission="whole", plan_mode="legacy",
                             plan=None)
        _same(default, explicit, case)
        assert a.last_policy_drops == b.last_policy_drops


class _ExplicitLegacyScheduler(FLScheduler):
    """Every Phase 4 switch passed explicitly with its recorded value."""

    built = 0

    def __init__(self, **kwargs):
        type(self).built += 1
        super().__init__(member_admission="whole", plan_mode="legacy", plan=None, **kwargs)


@pytest.mark.parametrize("name", sorted(M.SCENARIOS))
def test_the_supervisor_goldens_with_the_phase_4_switches_set_explicitly(name, monkeypatch):
    monkeypatch.setattr(mule_main, "FLScheduler", _ExplicitLegacyScheduler)
    before = _ExplicitLegacyScheduler.built
    current = M.SCENARIOS[name]()
    assert _ExplicitLegacyScheduler.built > before
    _canon.assert_same(_canon.load("supervisor")["cases"][name], current, name)


@pytest.mark.parametrize("name", P3SIM.TRIAL_NAMES)
def test_the_phase_3_trials_with_the_phase_4_switches_set_explicitly(name, monkeypatch):
    """UG4's simulated-clock trials at 6e6f92d (D1, D3 and D4 among them, whose
    schedulers now report their drops), replayed uncached."""
    monkeypatch.setattr(mule_main, "FLScheduler", _ExplicitLegacyScheduler)
    before = _ExplicitLegacyScheduler.built
    current = P3SIM.capture.__wrapped__(name)
    assert _ExplicitLegacyScheduler.built > before
    golden = P3SIM.load_golden()["cases"][name]
    for part in P3SIM.PARTS:
        assert not _canon.diff(golden[part], current[part]), (name, part)


def test_the_source_order_pins_still_hold():
    """test_s3b_feasibility.py:127-144 and test_p3_replan.py:1083-1087: S3b's
    gate is called before the learned selector in build_contact_queue."""
    src = inspect.getsource(FLScheduler.build_contact_queue)
    assert src.index("filter_feasible(") < src.index("self._target_selector.rank_contacts(")
    assert "build_ferry_plan" not in src


def _python(code):
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


_RECORDED_PATH = """
import sys
from hermes.scheduler import FLScheduler
from hermes.scheduler.fl_scheduler import nominal_mission_period_s
from hermes.scheduler.policies import (FedCSDegradedPolicy, FedExCarpPolicy, MaxAoIPolicy,
                                       OortPolicy, WhittlePolicy)
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel, FerryPhysics, FlightState
from hermes.types import MissionPass, MissionSlice, MuleID

class Dwell:
    def __call__(self, d, pass_kind, offset):
        return 20.0 + d

physics = FerryPhysics(dock=(0.0, 0.0, 0.0), member_dwell_s=Dwell(), upload_s=lambda: 0.5,
                       p_move_w=143.6, p_hover_w=168.5, range_m=30.0)
ids = tuple(f"d{i}" for i in range(6))
drops = 0
for model in (FeasibilityModel(ferry=physics), FeasibilityModel(), None):
    for policy in (None, MaxAoIPolicy(), OortPolicy(), WhittlePolicy(), FedExCarpPolicy(),
                   FedCSDegradedPolicy()):
        for kw in ({}, {"member_admission": "whole", "plan_mode": "legacy", "plan": None}):
            ferry = model is not None and model.ferry is not None
            sch = FLScheduler(now_fn=lambda: 0.0, mission_budget_s=60.0, feasibility_model=model,
                              target_selector=policy, validate_flown_order=ferry,
                              replan_fallback="trim", **kw)
            sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=ids, issued_round=1,
                                          issued_at=0.0))
            for i, d in enumerate(ids):
                sch.device_states[d].last_known_position = (40.0 * (i - 2.5), 25.0 * (i % 2), 0.0)
            sch.start_mission()
            sch.set_mission_round(2)
            q = sch.build_contact_queue(rf_range_m=30.0, mule_pose=(0.0, 0.0, 0.0))
            drops += bool(sch.last_policy_drops)
            state = FlightState((0.0, 0.0, 0.0), 30.0)
            for kind in (MissionPass.COLLECT, MissionPass.DELIVER):
                route = (q if kind is MissionPass.COLLECT
                         else sch.build_pass_2_queue(rf_range_m=30.0))
                sch.fold_remainder(route, state=state, budget_end=60.0, pass_kind=kind)
                sch.replan_remainder(route, state=state, budget_end=60.0, pass_kind=kind)
            sch.plan_protected(q)
nominal_mission_period_s({"a": (10.0, 0.0, 0.0), "b": (-50.0, 5.0, 0.0)}, rf_range_m=30.0,
                         feasibility_model=FeasibilityModel(ferry=physics), turnaround_s=30.0)
assert drops > 0, drops
loaded = sorted(m for m in sys.modules
                if m.startswith("hermes.scheduler.plan") or m.endswith("s3d_age_cap"))
assert not loaded, loaded
print("ok", drops)
"""


def test_a_legacy_scheduler_never_loads_the_plan_package():
    """Freeze Rule 1: on both clocks, every arm, every call the mule makes and
    the T_nom helper, with the switches at their defaults or set explicitly,
    load neither the plan package nor the age-cap stage (D arms that report
    drops included)."""
    assert _python(_RECORDED_PATH).startswith("ok")


# --------------------------------------------------------------------------- #
# The plan path
# --------------------------------------------------------------------------- #

def _twins(**states):
    """A legacy and a plan-mode scheduler on the same clock and devices:
    new, scheduled, beacon-only, out of the slice, and one S3 refuses while
    an earlier pass's bucket is still on it."""
    rt, setup = _t2_setup()
    out = []
    for kw in ({}, dict(plan_mode="ferry", plan=setup, replan_fallback="trim",
                        member_admission="subset")):
        sch = FLScheduler(now_fn=lambda: NOW, mission_budget_s=90.0,
                          feasibility_model=rt.feasibility_model(), deadline_time_scale=2.0,
                          miss_priority=True, **kw)
        ids = tuple(DeviceID(f"dev-{i}") for i in range(6))
        sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=ids, issued_round=1,
                                      issued_at=NOW),
                         registry_records=_records(ids=list(CLIFF_IDS)))
        for i, did in enumerate(ids):
            st = sch.device_states[did]
            st.is_new = i % 3 == 0
            st.deadline_fulfilment_s = 30.0 + 20.0 * i
            st.idle_time_ref_ts = 0.0 if i % 2 else NOW - 7.0 * i
        beacon = sch.device_states[DeviceID("dev-6")]
        beacon.last_beacon_ts, beacon.is_new = NOW - 3.0, False          # beacon-only
        stale = sch.device_states[DeviceID("dev-7")]
        stale.is_new, stale.deadline_override_ts = False, NOW + 50.0     # S3 refuses it
        stale.bucket = Bucket.SCHEDULED_THIS_ROUND
        sch.start_mission()
        sch.set_mission_round(2)
        out.append(sch)
    return out


def test_s1_and_s3_match_the_legacy_path():
    """Spec item 1: the plan repeats build_contact_queue's S1 and S3 calls, and
    gives equal deadlines and buckets. Its demand is the devices S3 bucketed
    and dated, in S3's order; a device S3 refused this pass stays out even
    with an earlier pass's bucket on it, which the legacy path's S3a would
    cluster with no deadline."""
    legacy, plan = _twins()
    legacy_queue = legacy.build_contact_queue(rf_range_m=60.0, mule_pose=DOCK)
    plan.build_ferry_plan(mule_pose=DOCK)
    assert plan.last_plan_deadlines == legacy.last_plan_deadlines
    assert len(plan.last_plan_deadlines) == 7                    # six in the slice, one beacon
    buckets = {d: st.bucket for d, st in legacy.device_states.items()}
    assert {d: st.bucket for d, st in plan.device_states.items()} == buckets
    assert set(buckets.values()) == {Bucket.NEW, Bucket.SCHEDULED_THIS_ROUND,
                                     Bucket.BEACON_ACTIVE}
    assert plan.last_plan.demand == tuple(legacy.last_plan_deadlines)
    stale = DeviceID("dev-7")
    assert stale not in plan.last_plan.demand and buckets[stale] is Bucket.SCHEDULED_THIS_ROUND
    planned = list(legacy_queue) + legacy.last_feasibility.dropped
    assert any(stale in wp.devices for wp in planned)


def _expected(sch, budget):
    """The search on the inputs spec item 1 names, assembled here from the
    scheduler's own S3 results: S3a at each searched class's radius, Pass 2
    priced once per class, U1's cap and U2's weights with the arm's
    ``miss_priority``, from the dock at takeoff to the budget's end."""
    now = sch.mission_start_ts
    setup = sch.plan_setup
    deadlines = sch.last_plan_deadlines
    demand = tuple(deadlines)
    cap = evaluate_cap(demand, sch.device_states, mission_round=sch.mission_round,
                       spec=setup.options.cap)
    weights = demand_weights(demand, sch.device_states, ages=cap.ages,
                             miss_priority=sch.miss_priority,
                             mode=setup.options.score.coverage_weights)
    entries = []
    for c in setup.searched:
        stops = cluster_by_rf_range(list(demand), sch.device_states, deadlines, c.radius_m)
        queue = sch.build_pass_2_queue(rf_range_m=c.radius_m, now=now, mule_pose=DOCK)
        entries.append(PS.ClassInput(c, tuple(stops), PS.price_pass_2(c.model, queue)))
    start = FlightState(DOCK, now)
    result = PS.plan_search(setup, entries, start=start, budget_end=now + budget,
                            deadlines=deadlines, device_states=sch.device_states, cap=cap,
                            weights=weights)
    return SimpleNamespace(result=result, entries=entries, cap=cap, weights=weights, start=start,
                           demand=demand)


@pytest.mark.parametrize("budget", [30.0, 60.0, 99.0])
def test_the_plan_commits_the_searched_plan(budget):
    """The commit (spec item 1; U0's and U4's recipes): the queue is the
    search's best route; ``last_feasibility`` keeps it; ``last_plan`` records
    the band, the policy, the demand, the weights, the score with its
    constants, the search's mode, candidate count and per-class summaries, the
    budget's end, T and the ages."""
    sch, _ = _t2_ferry(budget, merged={"dev-1": 1}, streaks={"dev-2": 2})
    queue = sch.build_ferry_plan(mule_pose=DOCK)
    want = _expected(sch, budget)
    best = want.result.best
    commit = sch.last_plan
    assert queue == list(best.fold.route) and queue
    assert sch.last_feasibility.kept == queue
    assert (commit.band, commit.band_index) == (best.band, best.cls.index)
    assert commit.band_class_policy == "search"
    assert commit.queue == tuple(queue) and commit.served == best.served
    assert commit.demand == want.demand == CLIFF_IDS
    assert dict(commit.weights) == want.weights
    # age x (1 + miss streak) under the arm's miss priority (decision 3)
    assert (commit.weights["dev-2"], commit.weights["dev-1"], commit.weights["dev-0"]) == (
        3.0, 1.0, 1.0)
    assert {k: commit.score[k] for k in PLAN_SCORE_KEYS} == best.terms.as_dict()
    assert commit.constants == sch.plan_setup.options.score.constants(len(CLIFF_IDS))
    assert (commit.search_mode, commit.n_candidates) == (want.result.mode,
                                                         want.result.n_candidates)
    assert [dict(s) for s in commit.per_class] == [dict(s) for s in want.result.per_class]
    assert (commit.budget_end, commit.t_ref_s, commit.mission_round) == (
        sch.mission_start_ts + budget, T_NOM, 1)
    assert dict(commit.ages) == dict(want.cap.ages) == {d: 0 if d == "dev-1" else 1
                                                        for d in CLIFF_IDS}
    assert commit.cap_s is None and commit.capped == frozenset() and commit.violations == ()
    assert not commit.closed and sch.last_order_check is None and sch.last_policy_drops == []
    assert math.isfinite(sch.last_plan_wall_s) and sch.last_plan_wall_s >= 0.0
    # Spec item 8 on the mule's physics: each device left out, alone at its S3a
    # stop of the committed class, from the dock at takeoff.
    start = FlightState(DOCK, sch.mission_start_ts)
    radius = sch.plan_setup.class_named(commit.band).radius_m
    labels, groups = {}, set()
    for wp in cluster_by_rf_range(list(CLIFF_IDS), sch.device_states, sch.last_plan_deadlines,
                                  radius):
        for did in wp.devices:
            if did not in commit.served:
                alone = ContactWaypoint(position=wp.position, devices=(did,),
                                        bucket=sch.device_states[did].bucket,
                                        deadline_ts=sch.last_plan_deadlines[did])
                v = sch.feasibility_model.admit(start, alone, rule=RULE_DEADLINE_BUDGET,
                                                budget_end=commit.budget_end)
                labels[did] = "plan" if v.ok else v.reason
                groups.add((wp.position, labels[did]))
    feas = sch.last_feasibility
    reasons = ("overdue", "budget", "energy", "delivery", "plan")
    assert labels == {did: name for name in reasons
                      for wp in getattr(feas, "dropped_" + name) for did in wp.devices}
    assert len(labels) == len(CLIFF_IDS) - len(commit.served)
    # One waypoint per (S3a stop, reason).
    assert sorted((wp.position, name) for name in reasons
                  for wp in getattr(feas, "dropped_" + name)) == sorted(groups)


def test_the_committed_class_model_prices_the_in_flight_check_and_pass_2():
    """Spec item 1: the class model becomes the scheduler's, so fold_remainder
    (the departure check, the beacon hook's fold) and Pass 2's range price b̄;
    the next plan commits its own class."""
    sch, rt = _t2_ferry(60.0)
    reference = sch.feasibility_model
    assert reference.ferry.range_m == rt.spec.link.range_planar_m("wide")
    queue = sch.build_ferry_plan(mule_pose=DOCK)
    commit = sch.last_plan
    assert commit.band == "narrow"
    narrow = sch.plan_setup.class_named("narrow").model
    assert sch.feasibility_model is narrow
    assert sch.feasibility_model.ferry.range_m == rt.spec.link.range_planar_m("narrow")
    t0 = sch.mission_start_ts
    start, end = FlightState(DOCK, t0), t0 + 60.0
    got = sch.fold_remainder(queue, state=start, budget_end=end)
    assert got.ok and got == narrow.fold(queue, start, rule=RULE_DEADLINE_BUDGET, budget_end=end,
                                         skip=False)
    assert reference.fold(queue, start, rule=RULE_DEADLINE_BUDGET, budget_end=end,
                          skip=False).home != got.home
    # Planned 15 s into the budget, 45 s are left: the next plan commits medium.
    sch.build_ferry_plan(now=t0 + 15.0, mule_pose=DOCK)
    assert sch.last_plan.band == "medium"
    assert sch.feasibility_model is sch.plan_setup.class_named("medium").model


@pytest.mark.parametrize("dwell_in_delta", [True, False])
def test_the_predicted_whole_mission_is_recorded_for_every_arm(dwell_in_delta):
    """R4: ``score["mission_s"]`` is Pass 1 + the turnaround + Pass 2 on b̄ for
    every plan arm; it is V's Δ when the dwell is in Δ, and not under
    F-dwell (U2's hand-off)."""
    sch, rt = _t2_ferry(60.0, score=PlanScoreParams(dwell_in_delta=dwell_in_delta))
    queue = sch.build_ferry_plan(mule_pose=DOCK)
    commit = sch.last_plan
    model = sch.feasibility_model
    t0 = sch.mission_start_ts
    pass_1 = model.fold(queue, FlightState(DOCK, t0), rule=RULE_DEADLINE_BUDGET,
                        budget_end=t0 + 60.0, skip=False).home - t0
    radius = sch.plan_setup.class_named(commit.band).radius_m
    pass_2 = PS.price_pass_2(model, sch.build_pass_2_queue(rf_range_m=radius, mule_pose=DOCK))
    mission = pass_1 + rt.spec.flight.turnaround_s + pass_2.time_s
    assert commit.score["mission_s"] == mission
    assert commit.score["mission_s"] == predicted_mission_s(
        serves_any=True, pass_1_s=pass_1, turnaround_s=30.0, pass_2_s=pass_2.time_s)
    if dwell_in_delta:
        assert commit.score["delta_s"] == mission
    else:
        assert commit.score["delta_s"] < mission - 30.0
    assert commit.describe()["score"]["mission_s"] == mission


def test_a_capped_plan_without_its_mission_round_raises_before_anything_is_built(monkeypatch):
    """R1 (critic B9): the age counts the mule's missions, so a cap without the
    round raises FLSchedulerError before S1, the cap or the search run, after
    the previous plan's diagnostics are cleared. Plan mode needs the round
    without a cap too: the coverage weights read the ages."""
    sch, _ = _t2_ferry(60.0, s=2, mission_round=3)
    sch.build_ferry_plan(mule_pose=DOCK)
    assert sch.last_plan is not None and sch.last_plan.capped
    assert sch.last_plan_wall_s is not None

    def tripwire(*args, **kwargs):
        raise AssertionError("built a plan without its mission round")

    monkeypatch.setattr(PS, "plan_search", tripwire)
    monkeypatch.setattr(FLS, "filter_eligible", tripwire)
    sch.set_mission_round(None)
    # Whatever an earlier call left, the failed plan leaves none of it.
    sch.last_order_check, sch.last_policy_drops = object(), [(None, REASON_BUDGET)]
    with pytest.raises(FLSchedulerError, match="B9"):
        sch.build_ferry_plan(mule_pose=DOCK)
    assert (sch.last_plan, sch.last_feasibility, sch.last_plan_deadlines) == (None, None, {})
    assert (sch.last_order_check, sch.last_policy_drops, sch.last_plan_wall_s) == (None, [], None)
    free, _ = _t2_ferry(60.0)
    free.set_mission_round(None)
    with pytest.raises(FLSchedulerError, match="set_mission_round"):
        free.build_ferry_plan(mule_pose=DOCK)


def test_the_plan_is_made_at_the_dock():
    sch, _ = _t2_ferry(60.0)
    with pytest.raises(FLSchedulerError, match="at the dock"):
        sch.build_ferry_plan(mule_pose=(5.0, 0.0, 0.0))
    assert sch.build_ferry_plan() == sch.build_ferry_plan(mule_pose=[0, 0, 0])


def test_the_guard_folds_the_final_route_with_its_own_exempt_stops():
    """Critic B1: the plan flies the capped member ``a`` alone, cut from a
    stop whose uncapped co-member does not fit beside it, and ``a`` is already
    late. The cut stop is a new waypoint, exempt because every member of it is
    capped: the guard passes only with the exempt set of the route itself. The
    set of S3a's stops (whose stop is mixed) would leave it unprotected."""
    setup = _synthetic_setup(radius=15.0, a=2.0, b=6.0, s=3)
    sch = _synthetic(setup, B1_CASE, budget=60.0)
    (stop,) = sch.build_ferry_plan()
    assert stop.devices == ("a",) and stop.deadline_ts == NOW - 600.0
    capped = sch.last_plan.capped
    assert capped == {"a"}
    s3a = cluster_by_rf_range(list(B1_CASE), sch.device_states, sch.last_plan_deadlines, 15.0)
    assert [wp.devices for wp in s3a] == [("a", "u")] and stop.position == s3a[0].position
    model = sch.feasibility_model
    start = FlightState(DOCK, NOW)

    def fold(protected):
        return model.fold([stop], start, rule=RULE_DEADLINE_BUDGET, budget_end=NOW + 60.0,
                          skip=False, protected=protected)

    assert fold(cap_stops([stop], capped).exempt).ok
    assert not fold(cap_stops(s3a, capped).exempt).ok
    assert fold(frozenset()).rejected == ((stop, REASON_OVERDUE),)
    assert [wp.devices for wp in sch.last_feasibility.dropped_plan] == [("u",)]


def test_a_route_the_predicate_refuses_is_never_committed(monkeypatch):
    """The guard (spec item 1): a search result that fails S3b's predicate as
    flown raises FLSchedulerError, and nothing is committed: no plan, no gate
    result, and the scheduler keeps the model it had."""
    sch, _ = _t2_ferry(30.0)
    before = sch.feasibility_model

    def overrun(setup, classes, **kw):
        """The field-wide narrow stop, which takes 99 s, offered as the plan."""
        narrow = next(e for e in classes if e.cls.name == "narrow")
        (whole,) = narrow.stops
        assert len(whole.devices) == 8
        fold = narrow.cls.model.fold([whole], kw["start"], rule=RULE_DEADLINE_BUDGET,
                                     budget_end=None, skip=False)
        terms = PS.search_class(setup, narrow, **kw).best.terms
        cand = Candidate(cls=narrow.cls, fold=MemberFold(route=(whole,), dropped=(),
                                                         state=fold.state, home=fold.home,
                                                         feasible=True, energy_j=0.0),
                         terms=terms)
        return SearchResult(best=cand, mode="stop_subsets", n_candidates=1,
                            per_class=({"band": "narrow"},))

    monkeypatch.setattr(PS, "plan_search", overrun)
    with pytest.raises(FLSchedulerError, match="nothing was committed"):
        sch.build_ferry_plan(mule_pose=DOCK)
    assert (sch.last_plan, sch.last_feasibility) == (None, None)
    assert sch.feasibility_model is before


def test_each_device_left_out_is_labelled_by_the_first_clause_refusing_it_alone():
    """Spec item 8: each demanded device the plan leaves out is labelled with
    the first clause that refuses it alone on b̄, at its S3a stop, from the
    dock at takeoff: ``overdue`` (o, cut from n's stop), ``budget`` (b; and c,
    late but capped, so never ``overdue``), ``energy`` (e), else ``plan`` (p,
    which fits alone but not beside the capped k). One waypoint per (stop,
    reason); ``delivery`` never fires at takeoff; the plan's own choices stay
    out of S3c's planned count (critic C6); the capped c that no class serves
    even alone is ``unplannable``."""
    setup = _synthetic_setup(radius=10.0, a=2.0, b=0.0, s=3, capacity=15000.0)
    sch = _synthetic(setup, LABELLED, budget=150.0)
    queue = sch.build_ferry_plan()
    assert _devices(queue) == [("n",), ("k",)]
    feas = sch.last_feasibility
    got = {name: [(wp.devices, wp.deadline_ts - NOW) for wp in getattr(feas, "dropped_" + name)]
           for name in ("overdue", "budget", "energy", "delivery", "plan")}
    assert got == {
        "overdue": [(("o",), -600.0)],
        "budget": [(("c",), -600.0), (("b",), 300.0)],
        "energy": [(("e",), 300.0)],
        "delivery": [],
        "plan": [(("p",), 300.0)],
    }
    (o_stop,) = [wp for wp in cluster_by_rf_range(list(LABELLED), sch.device_states,
                                                  sch.last_plan_deadlines, 10.0)
                 if "o" in wp.devices]
    assert o_stop.devices == ("o", "n") and feas.dropped_overdue[0].position == o_stop.position
    assert mission_planned_devices(queue, feas) == 6 == len(LABELLED) - 1
    assert feas.n_dropped == 5
    assert sch.last_plan.violations == (CapViolation(DeviceID("c"), 5, CAP_UNPLANNABLE),)
    assert sch.last_plan.capped == {"c", "k"}


def test_a_stops_left_out_members_are_labelled_one_by_one():
    """Spec item 8 under ``subset``: each member a stop leaves out is judged
    alone, so one S3a stop's drops split by reason: ``x``, already late, is
    ``overdue``, and ``y``, which fits alone but not beside the capped ``k``
    the plan serves, is ``plan``; judged as a pair both would read
    ``overdue``."""
    setup = _synthetic_setup(radius=10.0, a=2.0, b=0.0, s=3)
    sch = _synthetic(setup, SPLIT_CASE, budget=85.0)
    assert _devices(sch.build_ferry_plan()) == [("k",)]
    feas = sch.last_feasibility
    assert (_devices(feas.dropped_overdue), _devices(feas.dropped_plan)) == ([("x",)], [("y",)])
    assert feas.n_dropped == 2
    (stop,) = [wp for wp in cluster_by_rf_range(list(SPLIT_CASE), sch.device_states,
                                                sch.last_plan_deadlines, 10.0)
               if "x" in wp.devices]
    assert set(stop.devices) == {"x", "y"}
    assert feas.dropped_overdue[0].position == feas.dropped_plan[0].position == stop.position


def test_the_labels_are_priced_on_the_committed_class():
    """Spec item 8 prices each label on b̄. The plan flies the fast class (the
    slow reference class cannot serve the capped ``k`` in the budget), and on
    it ``p`` fits alone but not beside ``k``: ``plan``. Alone on the reference
    class it would be over the budget."""
    classes = []
    for index, (name, a) in enumerate((("wide", 80.0), ("narrow", 2.0))):
        physics = FerryPhysics(dock=DOCK, member_dwell_s=_Dwell(a, 0.0), upload_s=_Upload(),
                               p_move_w=P_MOVE, p_hover_w=P_HOVER, range_m=10.0)
        model = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=1.0, ferry=physics)
        classes.append(PlanClass(name=name, index=index, radius_m=10.0, model=model,
                                 outage=_NoOutage()))
    setup = PlanSetup(options=PlanOptions(cap=AgeCapSpec(s_missions=3)), classes=tuple(classes),
                      reference="wide", t_ref_s=300.0, turnaround_s=30.0)
    devices = {DeviceID("k"): ((200.0, 0.0, 0.0), 300.0, 0.0, None),
               DeviceID("p"): ((-200.0, 0.0, 0.0), 300.0, 0.0, 4)}
    sch = _synthetic(setup, devices, budget=150.0)
    assert _devices(sch.build_ferry_plan()) == [("k",)] and sch.last_plan.band == "narrow"
    feas = sch.last_feasibility
    assert _devices(feas.dropped_plan) == [("p",)] and feas.dropped_budget == []
    (p_stop,) = feas.dropped_plan
    wide = sch.plan_setup.class_named("wide").model
    assert wide.admit(FlightState(DOCK, NOW), p_stop, rule=RULE_DEADLINE_BUDGET,
                      budget_end=NOW + 150.0).reason == REASON_BUDGET
    assert sch.last_plan.violations == ()


def test_a_capped_device_a_class_could_serve_alone_is_crowded():
    """Spec item 6: at 45 s and S = 2 on mission 3 every T2 device is capped;
    the plan keeps the oldest that fit and the rest are ``crowded`` (some class
    serves each of them alone) with their planning ages."""
    sch, _ = _t2_ferry(45.0, s=2, mission_round=3)
    sch.build_ferry_plan(mule_pose=DOCK)
    commit = sch.last_plan
    left = [d for d in CLIFF_IDS if d not in commit.served]
    assert left and commit.capped == frozenset(CLIFF_IDS)
    assert commit.violations == tuple(CapViolation(d, 3, CAP_CROWDED) for d in sorted(left))
    assert commit.describe()["cap"]["violations"] == [v.describe() for v in commit.violations]


def test_the_lookahead_reaches_the_commit():
    """R9 keeps the lookahead L a parameter (spec item 5): a device is capped
    from age S - L. On mission 2 every T2 device is 2 missions old, so S = 3
    caps all of them with L = 1 and none with L = 0; the commit records L,
    and the capped devices the plan leaves out are ``crowded``."""
    for lookahead, capped in ((1, frozenset(CLIFF_IDS)), (0, frozenset())):
        sch, _ = _t2_ferry(60.0, s=3, lookahead=lookahead, mission_round=2)
        sch.build_ferry_plan(mule_pose=DOCK)
        commit = sch.last_plan
        assert (commit.cap_s, commit.cap_lookahead, commit.capped) == (3, lookahead, capped)
        assert dict(commit.ages) == dict.fromkeys(CLIFF_IDS, 2)
        assert commit.describe()["cap"]["lookahead"] == lookahead
        left = sorted(capped - commit.served)
        assert commit.violations == tuple(CapViolation(d, 2, CAP_CROWDED) for d in left)
        assert bool(left) == bool(lookahead)


def test_a_mixed_complement_carries_its_uncapped_members_deadline():
    """Critic B2 on spec item 8's waypoints: the capped ``c`` (already late)
    and the uncapped ``v`` share a stop, each fits alone there (``c`` exempt
    from its own deadline), and the plan flies the older capped ``k``, so it
    drops them as one ``plan`` waypoint. It carries the earliest deadline of
    its uncapped members, ``v``'s, as every stop with a capped member does:
    ``c``'s lateness is excused, and the mule records this deadline with the
    drop (``pass_1_preflight_drops``). ``c`` fits alone at its S3a stop, so
    the hover rule leaves it there, and it is ``crowded``."""
    setup = _synthetic_setup(radius=10.0, a=2.0, b=0.0, s=3)
    sch = _synthetic(setup, B2_CASE, budget=60.0)
    assert sorted(_devices(sch.build_ferry_plan())) == [("k",), ("n",)]
    feas = sch.last_feasibility
    (dropped,) = feas.dropped_plan
    assert set(dropped.devices) == {"c", "v"} and feas.n_dropped == 1
    stops = cluster_by_rf_range(list(B2_CASE), sch.device_states, sch.last_plan_deadlines, 10.0)
    (s3a,) = [wp for wp in stops if "c" in wp.devices]
    assert dropped.position == s3a.position
    deadlines = sch.last_plan_deadlines
    assert (deadlines["c"] - NOW, deadlines["v"] - NOW) == (-600.0, 300.0)
    assert dropped.deadline_ts == deadlines["v"]
    assert sch.last_plan.capped == {"k", "c"}
    assert sch.last_plan.violations == (CapViolation(DeviceID("c"), 4, CAP_CROWDED),)


def test_the_hover_rule_splits_a_mixed_stop_and_labels_each_part_at_its_own_stop():
    """The user's decision of 2026-09-30 on spec item 8: the capped ``c`` and
    the uncapped ``v`` share a stop 200 m out, each over the 60 s budget alone
    there (``c``, exempt from its own deadline, for the budget). ``c`` leaves
    the stop for its best hover point, 10 m from it towards the dock (a
    constant dwell: the edge of its reach), still over the budget there
    (78.5 s): it is ``unplannable`` and dropped for the ``budget`` at its hover
    stop, with its own deadline. ``v`` stays at the S3a stop's position, its
    deadline its own (B2: the stop has no capped member left), and is dropped
    for the ``budget`` there: two waypoints, one per offered stop."""
    setup = _synthetic_setup(radius=10.0, a=2.0, b=0.0, s=3)
    sch = _synthetic(setup, HOVER_SPLIT, budget=60.0)
    assert _devices(sch.build_ferry_plan()) == [("n",)]
    feas = sch.last_feasibility
    got = {wp.devices: wp for wp in feas.dropped_budget}
    assert set(got) == {("c",), ("v",)} and feas.n_dropped == 2
    stops = cluster_by_rf_range(list(HOVER_SPLIT), sch.device_states, sch.last_plan_deadlines,
                                10.0)
    (s3a,) = [wp for wp in stops if "c" in wp.devices]
    assert s3a.devices == ("c", "v") and s3a.position == (202.5, 0.0, 0.0)
    assert got[("v",)].position == s3a.position
    assert got[("c",)].position == pytest.approx((190.0, 0.0, 0.0))
    deadlines = sch.last_plan_deadlines
    assert (got[("c",)].deadline_ts, got[("v",)].deadline_ts) == (deadlines["c"], deadlines["v"])
    start = FlightState(DOCK, NOW)
    v = sch.feasibility_model.admit(start, got[("c",)], rule=RULE_DEADLINE_BUDGET,
                                    budget_end=NOW + 60.0, protected=True)
    assert v.reason == REASON_BUDGET and v.home - NOW == pytest.approx(78.5)
    assert sch.last_plan.violations == (CapViolation(DeviceID("c"), 5, CAP_UNPLANNABLE),)


def test_the_labels_and_the_caps_reasons_are_priced_from_the_plans_takeoff():
    """Spec items 6 and 8 price each device from (dock, takeoff), the plan's
    own clock, against the budget's end measured from the mission's stamp.
    At S = 2 on mission 3 every T2 device is capped. Planned 55 s into a 60 s
    budget no S3a stop serves anyone alone, so every capped device is offered
    its hover stop (the user's decision of 2026-09-30). In the 5 s left only
    the four devices within reach of the dock fit alone there, hovering at
    the dock: the plan serves two (wide, dev-0 and dev-4) and the other two
    are ``crowded``, left out by the plan's choice (``plan``); the other four,
    although each fits alone from the stamp, are left out for the ``budget``
    at their hover stops and are ``unplannable``."""
    sch, _ = _t2_ferry(60.0, s=2, mission_round=3)
    t0 = sch.mission_start_ts
    route = sch.build_ferry_plan(now=t0 + 55.0, mule_pose=DOCK)
    commit, feas = sch.last_plan, sch.last_feasibility
    near = {DeviceID(f"dev-{i}") for i in (0, 1, 2, 4)}
    assert commit.budget_end == t0 + 60.0 and commit.band == "wide"
    assert commit.served == {DeviceID("dev-0"), DeviceID("dev-4")}
    assert [wp.position for wp in route] == [DOCK, DOCK]
    by = {name: sorted(d for wp in getattr(feas, "dropped_" + name) for d in wp.devices)
          for name in ("overdue", "budget", "energy", "delivery", "plan")}
    assert by == {"overdue": [], "budget": sorted(set(CLIFF_IDS) - near), "energy": [],
                  "delivery": [], "plan": sorted(near - commit.served)}
    assert commit.violations == tuple(sorted(
        (CapViolation(d, 3, CAP_CROWDED if d in near else CAP_UNPLANNABLE)
         for d in CLIFF_IDS if d not in commit.served),
        key=lambda v: (v.reason != CAP_UNPLANNABLE, v.device)))

    def alone_fits(cls, clock):
        """The devices ``cls`` serves alone at their S3a stops from the dock at ``clock``."""
        out = set()
        for wp in cluster_by_rf_range(list(CLIFF_IDS), sch.device_states,
                                      sch.last_plan_deadlines, cls.radius_m):
            for did in wp.devices:
                alone = ContactWaypoint(position=wp.position, devices=(did,), bucket=wp.bucket,
                                        deadline_ts=wp.deadline_ts)
                if cls.model.admit(FlightState(DOCK, clock), alone, rule=RULE_DEADLINE_BUDGET,
                                   budget_end=t0 + 60.0, protected=True).ok:
                    out.add(did)
        return out

    def hover_fits(cls, clock):
        """The devices ``cls`` serves alone hovering at the dock from ``clock``
        (within its reach of the dock)."""
        out = set()
        for did in CLIFF_IDS:
            alone = ContactWaypoint(position=DOCK, devices=(did,), bucket=Bucket.NEW,
                                    deadline_ts=math.inf)
            (dist,) = cls.model.ferry.member_distances_m(alone)
            if dist <= cls.radius_m and cls.model.admit(
                    FlightState(DOCK, clock), alone, rule=RULE_DEADLINE_BUDGET,
                    budget_end=t0 + 60.0, protected=True).ok:
                out.add(did)
        return out

    assert alone_fits(sch.plan_setup.class_named(commit.band), t0) == set(CLIFF_IDS)
    assert not any(alone_fits(c, t0 + 55.0) for c in sch.plan_setup.searched)
    assert set().union(*(hover_fits(c, t0 + 55.0) for c in sch.plan_setup.searched)) == near


def test_an_empty_demand_commits_the_empty_plan():
    """U0's and U4's hand-offs: with nothing demanded the search still runs,
    the empty plan on the first searched class, mode ``exact``; it pays the
    turnaround alone."""
    sch, _ = _t2_ferry(60.0)
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=(), issued_round=1,
                                  issued_at=0.0))
    assert sch.build_ferry_plan(mule_pose=DOCK) == []
    commit = sch.last_plan
    assert (commit.band, commit.search_mode, commit.demand, commit.queue) == ("wide", SEARCH_EXACT,
                                                                             (), ())
    assert commit.n_candidates == 3 and commit.score["mission_s"] == 30.0
    feas = sch.last_feasibility
    assert feas.kept == [] and feas.n_dropped == 0 and sch.last_plan_deadlines == {}


def test_a_plan_without_a_budget_gates_nothing():
    """Spec item 8: a plan cell without a budget is the control. No clause of
    the predicate fires without a budget (S3b's opt-in contract), so the plan
    serves every demanded device and has nothing to label or to guard."""
    sch, _ = _t2_ferry(None)
    queue = sch.build_ferry_plan(mule_pose=DOCK)
    commit = sch.last_plan
    assert commit.budget_end is None and commit.served == frozenset(CLIFF_IDS)
    assert sch.last_feasibility.kept == queue and sch.last_feasibility.n_dropped == 0


def test_a_pinned_band_searches_and_flies_its_class_only():
    """FB+c (``fixed:c``, decision 7): only class c is searched, and the commit
    is class c's best among F's classes (critic A3: F's key is never above it)."""
    pinned, _ = _t2_ferry(60.0, reference="medium", policy="fixed:medium")
    pinned.build_ferry_plan(mule_pose=DOCK)
    free, _ = _t2_ferry(60.0)
    free.build_ferry_plan(mule_pose=DOCK)
    commit = pinned.last_plan
    assert commit.band == "medium" and commit.band_class_policy == "fixed:medium"
    assert [s["band"] for s in commit.per_class] == ["medium"]
    own = next(s for s in free.last_plan.per_class if s["band"] == "medium")
    assert dict(commit.per_class[0]) == dict(own)
    assert commit.score["v"] == own["v"] <= free.last_plan.score["v"]
    assert pinned.feasibility_model is pinned.plan_setup.class_named("medium").model


def test_whole_flies_whole_stops_and_subset_cuts_them():
    """R5: ``member_admission="whole"`` reaches the search through the plan's
    options, which must be the scheduler's switch, and flies whole S3a stops
    in every mode; ``subset`` cuts the field-wide narrow stop at 60 s."""
    for admission in ("whole", "subset"):
        sch, _ = _t2_ferry(60.0, admission=admission)
        queue = sch.build_ferry_plan(mule_pose=DOCK)
        assert sch.member_admission == sch.plan_setup.options.member_admission == admission
        radius = sch.plan_setup.class_named(sch.last_plan.band).radius_m
        s3a = {(wp.position, frozenset(wp.devices))
               for wp in cluster_by_rf_range(list(CLIFF_IDS), sch.device_states,
                                             sch.last_plan_deadlines, radius)}
        cut = [wp for wp in queue if (wp.position, frozenset(wp.devices)) not in s3a]
        if admission == "whole":
            assert queue and not cut
        else:
            assert sch.last_plan.band == "narrow" and len(cut) == 1
            assert set(cut[0].devices) == {DeviceID(f"dev-{i}") for i in (0, 1, 2, 4, 6)}


def _whole_stops(sch, cls=None):
    """A class's S3a stops (the committed class's by default), each with B2's
    deadline under the commit's cap: the stops a ``whole`` plan flies (U4)."""
    commit = sch.last_plan
    cls = cls or sch.plan_setup.class_named(commit.band)
    stops = cluster_by_rf_range(list(commit.demand), sch.device_states, sch.last_plan_deadlines,
                                cls.radius_m)
    return with_cap_deadlines(stops, deadlines=sch.last_plan_deadlines, capped=commit.capped)


@pytest.mark.parametrize("policy, budget, flies, labels", [
    ("fixed:narrow", 60.0, [], {"budget": [(0, 1, 2, 3, 4, 5, 6, 7)]}),
    ("fixed:medium", 45.0, [(0, 1, 2, 4, 5)], {"budget": [(3, 6), (7,)]}),
    ("fixed:medium", 60.0, [(0, 1, 2, 4, 5)], {"plan": [(3, 6), (7,)]}),
    ("search", 45.0, [(0, 1, 2, 4, 5)], {"budget": [(3, 6), (7,)]}),
])
def test_under_whole_each_left_out_stop_is_labelled_as_a_whole_stop(policy, budget, flies, labels):
    """R5 on spec item 8: under ``whole`` a plan serves each S3a stop whole or
    leaves it out whole, so a stop left out is labelled by the clause that
    refuses it whole on b̄ from the dock at takeoff (B2's deadline, protected
    when every member is capped), and ``plan`` only when it fits. That is a
    shortfall S3c counts, as H1's gate reports the same contact, not a
    choice. FB+narrow at 60 s is the Phase 3 cliff: the field-wide stop is
    dropped for the budget, every device planned and none served (``subset``
    serves five, test_whole_flies_whole_stops_and_subset_cuts_them)."""
    reference = policy.split(":")[1] if ":" in policy else "wide"
    sch, _ = _t2_ferry(budget, admission="whole", reference=reference, policy=policy)
    queue = sch.build_ferry_plan(mule_pose=DOCK)
    commit, feas = sch.last_plan, sch.last_feasibility
    names = ("overdue", "budget", "energy", "delivery", "plan")
    got = {name: getattr(feas, "dropped_" + name) for name in names}
    stop = {frozenset(wp.devices): wp for wp in _whole_stops(sch)}

    def numbered(route):
        return [tuple(sorted(int(d.split("-")[1]) for d in wp.devices)) for wp in route]

    assert numbered(queue) == flies
    assert {name: numbered(route) for name, route in got.items() if route} == labels
    # Each drop is the whole stop the plan could fly, and its verdict alone.
    start = FlightState(DOCK, sch.mission_start_ts)
    for name, route in got.items():
        for wp in route:
            assert wp == stop[frozenset(wp.devices)]
            v = sch.feasibility_model.admit(start, wp, rule=RULE_DEADLINE_BUDGET,
                                            budget_end=commit.budget_end,
                                            protected=is_exempt(wp, commit.capped))
            assert (REASON_PLAN if v.ok else v.reason) == name
    planned = len(CLIFF_IDS) - sum(len(wp.devices) for wp in got["plan"])
    assert mission_planned_devices(queue, feas) == planned


def _fits_alone_anywhere(cls, did, start, budget_end, points=400):
    """True when ``cls`` serves ``did`` alone within the budget from ``start``
    at some point of a ``points + 1`` grid on the segment from the device to
    the dock, within the class's reach: this file's own pricing of the hover
    rule's premise, not ``plan/hover.py``."""
    st = cls.model.ferry.device_states[did]
    where = tuple(st.last_known_position)
    length = math.dist(where, DOCK)
    for i in range(points + 1):
        r = min(length, cls.radius_m) * i / points
        t = r / length if length else 0.0
        stop = ContactWaypoint(position=tuple(a + t * (b - a) for a, b in zip(where, DOCK)),
                               devices=(did,), bucket=Bucket.NEW, deadline_ts=math.inf)
        (dist,) = cls.model.ferry.member_distances_m(stop)
        if dist <= cls.radius_m and cls.model.admit(start, stop, rule=RULE_DEADLINE_BUDGET,
                                                    budget_end=budget_end, protected=True).ok:
            return True
    return False


def test_under_whole_the_caps_reasons_judge_the_device_alone():
    """Spec item 6 under ``whole``, as the user decided on 2026-09-30:
    ``unplannable`` is physics, no class the arm may fly serving the device
    alone even at its best hover point, so a capped device left out is
    ``crowded`` whenever one does, even when ``whole`` admission, which flies
    S3a's stops whole (R5), is what leaves it out: the arm's admission rule,
    not physics. Spec item 8's drops still judge the stop whole (R5). At S = 2
    on mission 3 every T2 device is capped. FB+narrow at 60 s cannot fly the
    field-wide stop whole, so it flies nothing (``subset`` serves five), and
    the stop is dropped for the ``budget``; each of the eight fits alone at
    that stop, so all eight are ``crowded`` (``unplannable`` under R5's
    stop-level label, before the decision). F at 45 s labels the devices it
    leaves out by whether some class serves them alone, here every one."""
    sch, _ = _t2_ferry(60.0, admission="whole", reference="narrow", policy="fixed:narrow",
                       s=2, mission_round=3)
    assert sch.build_ferry_plan(mule_pose=DOCK) == []
    assert sch.last_plan.violations == tuple(CapViolation(d, 3, CAP_CROWDED) for d in CLIFF_IDS)
    (whole,) = sch.last_feasibility.dropped_budget
    assert set(whole.devices) == set(CLIFF_IDS) and sch.last_feasibility.n_dropped == 1
    sub, _ = _t2_ferry(60.0, admission="subset", reference="narrow", policy="fixed:narrow",
                       s=2, mission_round=3)
    sub.build_ferry_plan(mule_pose=DOCK)
    assert len(sub.last_plan.served) == 5
    assert {v.reason for v in sub.last_plan.violations} == {CAP_CROWDED}
    sch, _ = _t2_ferry(45.0, admission="whole", s=2, mission_round=3)
    sch.build_ferry_plan(mule_pose=DOCK)
    commit = sch.last_plan
    start = FlightState(DOCK, sch.mission_start_ts)
    servable = {d for d in CLIFF_IDS
                if any(_fits_alone_anywhere(c, d, start, commit.budget_end)
                       for c in sch.plan_setup.searched)}
    got = {v.device: v.reason for v in commit.violations}
    assert got == {d: CAP_CROWDED if d in servable else CAP_UNPLANNABLE
                   for d in CLIFF_IDS if d not in commit.served}
    assert set(got.values()) == {CAP_CROWDED}
    # A whole stop is priced as the search prices it. The budget flies one
    # stop, and the plan keeps the oldest capped ``k``. ``c1``'s stop is exempt
    # though ``c1`` is late, and the mixed stop of the late ``c2`` carries
    # ``v``'s deadline (B2): both fit alone, so both are ``crowded``, and both
    # stops are left out by the plan's choice. The capped ``z``, 400 m out,
    # fits nowhere alone, not even at its hover stop 10 m nearer (158.5 s): it
    # is ``unplannable``, dropped for the ``budget`` at that stop.
    setup = _synthetic_setup(radius=10.0, a=2.0, b=0.0, s=3, admission="whole")
    far = dict(WHOLE_CAP, z=((400.0, 0.0, 0.0), 300.0, 0.0, 1))
    sch = _synthetic(setup, far, budget=100.0)
    assert _devices(sch.build_ferry_plan()) == [("k",)]
    assert sch.last_plan.capped == {"k", "c1", "c2", "z"}
    assert sch.last_plan.violations == (CapViolation(DeviceID("z"), 4, CAP_UNPLANNABLE),
                                        CapViolation(DeviceID("c1"), 4, CAP_CROWDED),
                                        CapViolation(DeviceID("c2"), 4, CAP_CROWDED))
    feas = sch.last_feasibility
    assert sorted(map(set, _devices(feas.dropped_plan)), key=sorted) == [{"c1"}, {"c2", "v"}]
    (z,) = feas.dropped_budget
    assert z.devices == ("z",) and z.position == pytest.approx((390.0, 0.0, 0.0))
    assert feas.n_dropped == 3


def test_under_whole_a_device_is_crowded_when_another_class_flies_its_stop():
    """Spec item 6 counts every class the arm may fly; spec item 8 labels on
    b̄. Under ``whole`` F commits wide (index 0), where the plan serves the
    oldest capped ``k``: there ``p`` shares a stop with the far ``v``, which
    is over the budget whole, so the pair is labelled ``budget``. Narrow's
    smaller radius gives ``p`` a stop of its own that fits, though narrow
    cannot serve ``k``, so ``p`` is ``crowded``, not ``unplannable``."""
    classes = []
    for index, (name, radius, a) in enumerate((("wide", 100.0, 2.0), ("narrow", 10.0, 30.0))):
        physics = FerryPhysics(dock=DOCK, member_dwell_s=_Dwell(a, 1.0), upload_s=_Upload(),
                               p_move_w=P_MOVE, p_hover_w=P_HOVER, range_m=radius)
        model = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=1.0, ferry=physics)
        classes.append(PlanClass(name=name, index=index, radius_m=radius, model=model,
                                 outage=_NoOutage()))
    setup = PlanSetup(options=PlanOptions(member_admission="whole", cap=AgeCapSpec(s_missions=3)),
                      classes=tuple(classes), reference="wide", t_ref_s=300.0, turnaround_s=30.0)
    devices = {DeviceID("k"): ((100.0, 0.0, 0.0), 300.0, 0.0, None),
               DeviceID("p"): ((-50.0, 0.0, 0.0), 300.0, 0.0, 1),
               DeviceID("v"): ((-50.0, 60.0, 0.0), 300.0, 0.0, 4)}
    sch = _synthetic(setup, devices, budget=60.0)
    assert _devices(sch.build_ferry_plan()) == [("k",)] and sch.last_plan.band == "wide"
    assert sch.last_plan.capped == {"k", "p"}
    assert sch.last_plan.violations == (CapViolation(DeviceID("p"), 4, CAP_CROWDED),)
    feas = sch.last_feasibility
    assert [set(wp.devices) for wp in feas.dropped_budget] == [{"p", "v"}]
    assert feas.n_dropped == 1
    narrow = sch.plan_setup.class_named("narrow")
    alone = [wp for wp in _whole_stops(sch, narrow) if "p" in wp.devices]
    assert [wp.devices for wp in alone] == [("p",)]
    assert narrow.model.admit(FlightState(DOCK, NOW), alone[0], rule=RULE_DEADLINE_BUDGET,
                              budget_end=NOW + 60.0, protected=True).ok


def test_under_whole_a_mixed_stop_late_for_its_uncapped_member_is_overdue_whole():
    """R5 with critic B2: under ``whole`` the capped ``k`` can only be flown
    with ``o``, already late, and a mixed stop is held to its uncapped
    members' deadline, so the stop is ``overdue`` whole, ``k`` included. ``k``
    fits alone at that stop, so it is ``crowded``: the arm's admission rule,
    not physics, leaves it out (the user's decision of 2026-09-30; R5's
    stop-level label read ``unplannable``). Under ``subset`` ``k`` is served
    alone and only ``o`` is ``overdue``: there a capped device never is (spec
    item 8)."""
    got = {}
    for admission in ("whole", "subset"):
        setup = _synthetic_setup(radius=10.0, a=2.0, b=0.0, s=3, admission=admission)
        sch = _synthetic(setup, WHOLE_MIXED, budget=150.0)
        queue = sch.build_ferry_plan()
        feas = sch.last_feasibility
        got[admission] = (
            sorted(_devices(queue)),
            [(set(wp.devices), wp.deadline_ts - NOW) for wp in feas.dropped_overdue],
            feas.n_dropped, sch.last_plan.violations,
        )
    assert got["whole"] == ([("n",)], [({"k", "o"}, -600.0)], 1,
                            (CapViolation(DeviceID("k"), 5, CAP_CROWDED),))
    assert got["subset"] == ([("k",), ("n",)], [({"o"}, -600.0)], 1, ())


def test_pass_2_is_priced_from_the_dock():
    """Decision 2 (b): V's Pass 2 is the class's Pass-2 queue ordered nearest
    first from the dock the plan is made at and folded from it (U4's
    ``price_pass_2``). With the dock off the origin, the queue ordered from
    the origin would price a different tour."""
    dock = (100.0, 0.0, 0.0)
    setup = _synthetic_setup(radius=10.0, a=2.0, b=0.0, s=None, dock=dock)
    devices = {DeviceID("a"): ((1.0, 0.0, 0.0), 300.0, 0.0, 4),
               DeviceID("b"): ((100.0, 50.0, 0.0), 300.0, 0.0, 4),
               DeviceID("c"): ((160.0, 0.0, 0.0), 300.0, 0.0, 4)}
    sch = _synthetic(setup, devices, budget=None)
    queue = sch.build_ferry_plan()
    model = sch.feasibility_model
    pass_1 = model.fold(queue, FlightState(dock, NOW), rule=RULE_DEADLINE_BUDGET,
                        budget_end=None, skip=False).home - NOW

    def mission(pose):
        pass_2 = PS.price_pass_2(model, sch.build_pass_2_queue(rf_range_m=10.0, mule_pose=pose))
        return pass_1 + 30.0 + pass_2.time_s

    assert len(queue) == 3 and sch.last_plan.score["mission_s"] == mission(dock)
    assert mission(dock) != mission(DOCK)


def test_the_plan_is_deterministic_and_keeps_wall_time_out(monkeypatch):
    """A repeated plan is the same plan, whatever the wall clock says; the
    wall time is measured into ``last_plan_wall_s`` only (critic B12)."""
    a, _ = _t2_ferry(60.0, s=2, mission_round=2)
    b, _ = _t2_ferry(60.0, s=2, mission_round=2)
    ticks = iter(range(10 ** 6))
    monkeypatch.setattr(FLS, "time", SimpleNamespace(perf_counter=lambda: 1e6 * next(ticks)))
    assert a.build_ferry_plan(mule_pose=DOCK) == b.build_ferry_plan(mule_pose=DOCK)
    assert a.last_plan.describe() == b.last_plan.describe()
    assert a.last_plan_wall_s == 1e6
    assert "wall" not in repr(a.last_plan.describe())


def test_neither_the_preflight_order_check_nor_t_nom_enters_plan_mode(monkeypatch):
    """Spec item 1: a plan-mode scheduler that validates the flown order (the
    mule's ``replan`` response) never runs the pre-flight check; the T_nom
    helper plans a legacy plan."""
    period = FLS.nominal_mission_period_s(
        {DeviceID(f"dev-{i}"): (x, y, 0.0) for i, (x, y) in enumerate(T2_LAYOUT)},
        rf_range_m=60.0, feasibility_model=_t2_spec("wide").feasibility_model(
            rf_range_m=60.0, theta_bytes=52), turnaround_s=30.0)

    def tripwire(*args, **kwargs):
        raise AssertionError("plan mode entered")

    monkeypatch.setattr(FLScheduler, "_validate_order", tripwire)
    sch, _ = _t2_ferry(60.0)
    assert sch.validates_flown_order
    sch.build_ferry_plan(mule_pose=DOCK)
    assert sch.last_order_check is None
    monkeypatch.setattr(FLScheduler, "build_ferry_plan", tripwire)
    monkeypatch.setattr(FLScheduler, "_bind_plan", tripwire)
    assert FLS.nominal_mission_period_s(
        {DeviceID(f"dev-{i}"): (x, y, 0.0) for i, (x, y) in enumerate(T2_LAYOUT)},
        rf_range_m=60.0, feasibility_model=_t2_spec("wide").feasibility_model(
            rf_range_m=60.0, theta_bytes=52), turnaround_s=30.0) == period


# --------------------------------------------------------------------------- #
# After the plan: protection, close, the plan-mode re-plan
# --------------------------------------------------------------------------- #

def _mixed_plan():
    """T2 at 99 s, S = 2 on mission 3: devices merged in mission 2 are young
    (uncapped), the rest capped, so the narrow stop the plan flies is mixed.
    dev-2 has missed twice, so it weighs as much as a capped device (3)."""
    young = {"dev-0": 2, "dev-2": 2, "dev-4": 2, "dev-6": 2}
    sch, _ = _t2_ferry(99.0, s=2, mission_round=3, merged=young, streaks={"dev-2": 2})
    queue = sch.build_ferry_plan(mule_pose=DOCK)
    return sch, queue


def test_plan_protected_is_the_exempt_stops_of_what_remains():
    """Spec item 9 (U1's hand-off): what the departure check, the re-plan and
    the beacon hook's fold protect in plan mode, recomputed on the remainder
    as it stands, found by value (the mule's annotated copies included). A
    mixed stop is not protected (critic B2); nothing is without a cap."""
    sch, queue = _mixed_plan()
    capped = sch.last_plan.capped
    assert capped == {"dev-1", "dev-3", "dev-5", "dev-7"} and len(queue) == 1
    assert sch.plan_protected(queue) == frozenset()                   # a mixed stop
    exempt = MS.reduce_stop(queue[0], [d for d in queue[0].devices if d in capped],
                            deadlines=sch.last_plan_deadlines, device_states=sch.device_states,
                            capped=capped)
    annotated = dataclasses.replace(exempt, band="narrow", range_m=232.2,
                                    pred_snr_db=(0.0,) * len(exempt.devices))
    assert sch.plan_protected([annotated, queue[0]]) == {exempt} == {annotated}
    assert sch.plan_protected([annotated]) == cap_stops([annotated], capped).exempt
    free, _ = _t2_ferry(60.0)
    free.build_ferry_plan(mule_pose=DOCK)
    assert free.plan_protected(free.last_plan.queue) == frozenset()
    legacy_queue, legacy = _cliff(1e9)
    assert legacy.plan_protected(legacy_queue) == frozenset()


def test_close_plan_records_the_visited_set_and_the_close_time_violations():
    """Spec item 6 (U1's ``close_commit``): after ``record_merged`` the mule
    closes the plan with the stops it flew and the devices its merge used. A
    capped device no flown stop held is ``dropped_in_flight``; one visited but
    not merged is ``not_merged``. ``last_plan`` becomes the closed copy; a
    second close is refused, and so is a close without a plan."""
    sch, queue = _mixed_plan()
    commit = sch.last_plan
    (stop,) = queue
    flown = MS.reduce_stop(stop, [d for d in stop.devices if d != "dev-3"],
                           deadlines=sch.last_plan_deadlines, device_states=sch.device_states,
                           capped=commit.capped)
    merged = [d for d in flown.devices if d != "dev-5"]
    closed = sch.close_plan([flown], merged)
    assert closed is sch.last_plan and closed is not commit and closed.closed
    assert closed.visited == frozenset(flown.devices)
    assert [(v.device, v.reason) for v in closed.violations] == [
        ("dev-3", "dropped_in_flight"), ("dev-5", "not_merged")]
    assert not commit.closed
    with pytest.raises(FLSchedulerError, match="already closed"):
        sch.close_plan([flown], merged)
    empty, _ = _t2_ferry(60.0, s=2, mission_round=3)
    empty.build_ferry_plan(mule_pose=DOCK)
    served = empty.last_plan.served
    shut = empty.close_plan([], ())                                    # the empty path
    assert {v.device for v in shut.violations if v.reason == "dropped_in_flight"} == served


def test_the_plan_mode_re_plan_is_the_member_trim(monkeypatch):
    """Spec item 9 under ``subset``: Pass 1 is re-planned by U3's
    ``trim_members`` on the committed model, with the commit's capped set and
    weights and the plan's deadlines, priority stops first; ``replan_route``
    is never used (its identity check forbids reduced stops), and the
    caller's ``protected`` changes nothing (the trim recomputes the exempt
    stops)."""
    sch, queue = _mixed_plan()
    commit = sch.last_plan

    def no_replan_route(*args, **kwargs):
        raise AssertionError("replan_route used in plan mode")

    monkeypatch.setattr(RP, "replan_route", no_replan_route)
    t0 = sch.mission_start_ts
    end = t0 + 99.0
    assert commit.weights["dev-2"] == 3.0 == commit.weights["dev-1"]
    # 30 s late the capped members still fit alone, and they go first: the
    # uncapped ones are dropped, dev-2 too, though it weighs as much.
    state = FlightState(DOCK, t0 + 30.0)
    got = sch.replan_remainder(queue, state=state, budget_end=end)
    want = MS.trim_members(queue, state, model=sch.feasibility_model, budget_end=end,
                           deadlines=sch.last_plan_deadlines, device_states=sch.device_states,
                           capped=commit.capped, weights=commit.weights,
                           rule=RULE_DEADLINE_BUDGET)
    assert got == want and got.order_used == RP.ORDER_ARM_TRIMMED
    (cut,) = got.route
    assert set(cut.devices) == commit.capped < set(queue[0].devices)
    assert [(set(wp.devices), why) for wp, why in got.dropped] == [
        ({"dev-0", "dev-2", "dev-4"}, REASON_BUDGET)]
    assert sch.plan_protected(got.route) == {cut}
    # 25 s late one uncapped member fits after them: the heavier dev-2, not the
    # cheaper dev-4 (the weights per second of dwell, U3's member order).
    (cut,) = sch.replan_remainder(queue, state=FlightState(DOCK, t0 + 25.0),
                                  budget_end=end).route
    assert set(cut.devices) == commit.capped | {"dev-2"}
    assert sch.replan_remainder(queue, state=state, budget_end=end,
                                protected=sch.plan_protected(queue)) == got
    assert sch.replan_remainder(queue, state=FlightState(DOCK, t0), budget_end=end).route == \
        tuple(queue)


def test_the_plan_mode_re_plan_dates_each_member(monkeypatch):
    """A reduced stop takes its members' own deadlines: the plan's for the
    plan's own members, whatever the caller passes (the committed stops carry
    the plan's), and for a stop the beacon hook inserted, the mule's record
    (``deadlines``); a member neither dates takes its stop's deadline, never
    later than its own. The trim prices at the caller's SNR offset (δ_obs)."""
    sch, queue = _mixed_plan()
    seen = []
    real = MS.trim_members

    def spy(remainder, state, **kw):
        seen.append((dict(kw["deadlines"]), kw["snr_offset_db"]))
        return real(remainder, state, **kw)

    monkeypatch.setattr(MS, "trim_members", spy)
    extra = DeviceID("dev-x")
    sch.device_states[extra] = DeviceSchedulerState(device_id=extra,
                                                    last_known_position=(5.0, 5.0, 0.0),
                                                    bucket=Bucket.BEACON_ACTIVE)
    t0 = sch.mission_start_ts
    inserted = ContactWaypoint(position=(5.0, 5.0, 0.0), devices=(extra,),
                               bucket=Bucket.BEACON_ACTIVE, deadline_ts=t0 + 70.0)
    state = FlightState(DOCK, t0 + 10.0)
    member = queue[0].devices[0]
    plan = dict(sch.last_plan_deadlines)
    assert plan[member] == t0 + 1500.0
    sch.replan_remainder(queue + [inserted], state=state, budget_end=t0 + 99.0,
                         deadlines={extra: t0 + 80.0, member: t0 + 5.0}, snr_offset_db=2.5)
    sch.replan_remainder(queue + [inserted], state=state, budget_end=t0 + 99.0)
    assert extra not in plan
    assert seen == [({**plan, extra: t0 + 80.0}, 2.5), ({**plan, extra: t0 + 70.0}, 0.0)]


def test_under_whole_the_plan_mode_re_plan_keeps_stops_whole(monkeypatch):
    """R5 in flight: under ``whole`` the Pass-1 re-plan never cuts a stop and
    never calls the member trim. It keeps the remainder when it passes as
    flown, its exempt stops protected; otherwise the priority stops fly
    first, then the rest, and each stop is kept whole if it fits from where
    the previous one left the mule, else dropped whole with the reason it
    failed (U4's whole-mode start, in flight). 30 s late the mixed stop is
    late for ``m2`` (B2) and is dropped ``overdue``, the capped ``m1`` with it,
    where the member trim keeps ``m1``; the exempt ``k`` is kept though late;
    85 s late ``u`` is over the budget; and with ``u`` first in the
    remainder, the priority stops fly ahead of it."""
    setup = _synthetic_setup(radius=10.0, a=2.0, b=0.0, s=3, admission="whole")
    sch = _synthetic(setup, WHOLE_FLIGHT, budget=100.0)
    queue = sch.build_ferry_plan()
    mixed, k, u = queue
    assert (set(mixed.devices), k.devices, u.devices) == ({"m1", "m2"}, ("k",), ("u",))
    capped, end = sch.last_plan.capped, NOW + 100.0
    assert capped == {"k", "m1"} and sch.plan_protected(queue) == {k}
    late = FlightState(DOCK, NOW + 30.0)
    cut = MS.trim_members(queue, late, model=sch.feasibility_model, budget_end=end,
                          deadlines=sch.last_plan_deadlines, device_states=sch.device_states,
                          capped=capped, weights=sch.last_plan.weights)
    assert _devices(cut.route) == [("m1",), ("k",), ("u",)]

    def no_member_trim(*args, **kwargs):
        raise AssertionError("a member trim under whole")

    monkeypatch.setattr(MS, "trim_members", no_member_trim)

    def replan(remainder, after):
        return sch.replan_remainder(remainder, state=FlightState(DOCK, NOW + after),
                                    budget_end=end)

    trimmed, result = RP.ORDER_ARM_TRIMMED, RP.ReplanResult
    assert replan(queue, 0.0) == result(tuple(queue), (), RP.ORDER_CURRENT)
    assert replan(queue, 30.0) == result((k, u), ((mixed, REASON_OVERDUE),), trimmed)
    assert replan(queue, 85.0) == result((k,), ((mixed, REASON_OVERDUE), (u, REASON_BUDGET)),
                                         trimmed)
    assert replan([u, mixed, k], 0.0) == result((mixed, k, u), (), trimmed)
    assert replan([u, mixed, k], 85.0).dropped == ((u, REASON_BUDGET), (mixed, REASON_OVERDUE))
    for res in (replan(queue, 30.0), replan([u, mixed, k], 0.0)):
        assert all(any(wp is given for given in queue) for wp in res.route)
    with pytest.raises(ValueError, match="twice"):
        replan([k, k], 30.0)


def test_under_whole_every_re_plan_flies_and_drops_the_stops_it_was_given(monkeypatch):
    """R5 on trial T2 (the mule's physics), across budgets, caps, remainder
    orders, late departures and SNR offsets (δ_obs): under ``whole`` each
    re-planned route passes the predicate as flown with its own exempt stops
    protected, it and the drops hold each stop of the remainder once, the
    very objects, and the result is the rule above, priced at the offset."""
    def no_member_trim(*args, **kwargs):
        raise AssertionError("a member trim under whole")

    monkeypatch.setattr(MS, "trim_members", no_member_trim)
    seen = dict.fromkeys(("current", "trimmed", "some kept", "offset matters"), 0)
    young = {"dev-0": 2, "dev-1": 2, "dev-2": 2}
    for budget, s, merged in ((60.0, None, None), (99.0, 2, young), (150.0, 2, young),
                              (150.0, None, None)):
        sch, _ = _t2_ferry(budget, admission="whole", s=s, mission_round=3 if s else 1,
                           merged=merged)
        queue = sch.build_ferry_plan(mule_pose=DOCK)
        capped, model, t0 = sch.last_plan.capped, sch.feasibility_model, sch.mission_start_ts
        for remainder in (queue, queue[::-1]):
            order = {id(wp): i for i, wp in enumerate(remainder)}
            for after in range(0, 100, 10):
                state = FlightState(DOCK, t0 + after)
                got = {}
                for offset in (0.0, 6.0):
                    res = sch.replan_remainder(remainder, state=state, budget_end=t0 + budget,
                                               snr_offset_db=offset)
                    rule = dict(rule=RULE_DEADLINE_BUDGET, budget_end=t0 + budget,
                                snr_offset_db=offset)
                    exempt = cap_stops(remainder, capped).exempt
                    if model.fold(remainder, state, skip=False, protected=exempt, **rule).ok:
                        want = RP.ReplanResult(tuple(remainder), (), RP.ORDER_CURRENT)
                    else:
                        walk = model.fold(priority_first(remainder, capped), state, skip=True,
                                          protected=exempt, **rule)
                        drops = sorted(walk.rejected, key=lambda pair: order[id(pair[0])])
                        want = RP.ReplanResult(walk.route, tuple(drops), RP.ORDER_ARM_TRIMMED)
                    assert res == want, (budget, s, after, offset)
                    listed = list(res.route) + [wp for wp, _ in res.dropped]
                    assert sorted(map(id, listed)) == sorted(map(id, remainder))
                    assert model.fold(res.route, state, skip=False,
                                      protected=cap_stops(res.route, capped).exempt, **rule).ok
                    got[offset] = res
                    seen["current"] += res.order_used == RP.ORDER_CURRENT
                    seen["trimmed"] += res.order_used == RP.ORDER_ARM_TRIMMED
                    seen["some kept"] += bool(res.route and res.dropped)
                seen["offset matters"] += got[0.0] != got[6.0]
    assert min(seen.values()) >= 3, seen


def test_pass_2_is_re_planned_whole_in_plan_mode(monkeypatch):
    """R10: Pass 2 is never member-reduced; plan mode re-plans it as every arm
    does (the nearest-first budget walk), with the stops it was given."""
    sch, _ = _mixed_plan()

    def no_trim(*args, **kwargs):
        raise AssertionError("Pass 2 trimmed")

    monkeypatch.setattr(MS, "trim_members", no_trim)
    radius = sch.plan_setup.class_named(sch.last_plan.band).radius_m
    pass_2 = sch.build_pass_2_queue(rf_range_m=radius, mule_pose=DOCK)
    t0 = sch.mission_start_ts
    res = sch.replan_remainder(pass_2, state=FlightState(DOCK, t0), budget_end=t0 + 20.0,
                               pass_kind=DELIVER)
    assert res.dropped
    assert all(any(wp is q for q in pass_2) for wp in res.route)
    assert [wp for wp, _ in res.dropped] == [wp for wp in pass_2 if wp not in res.route]


def test_the_plan_mode_re_plan_needs_the_committed_plan():
    sch, _ = _t2_ferry(60.0)
    with pytest.raises(FLSchedulerError, match="build_ferry_plan first"):
        sch.replan_remainder([], state=FlightState(DOCK, 0.0), budget_end=60.0)


# --------------------------------------------------------------------------- #
# The H and D arms: member subsets before takeoff, and the D arms' report
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("admission", ["whole", "subset"])
def test_the_cliff_through_build_contact_queue_for_h1(admission):
    """unit_U3b.md section 7.2: at 60 s under ``subset`` S3b keeps the members
    that fit, {0, 1, 2, 4, 6}, and drops the rest as ``budget``; under
    ``whole`` it keeps none (the Phase 3 pin). An H arm reports no policy
    drops."""
    queue, sch = _cliff(60.0, member_admission=admission)
    (contact,), _ = _cliff(1e9)
    feas = sch.last_feasibility
    if admission == "whole":
        assert queue == [] and feas.dropped_budget == [contact]
    else:
        (stop,) = queue
        assert stop.devices == _in_stop_order(contact, (0, 1, 2, 4, 6))
        assert _devices(feas.dropped_budget) == [_in_stop_order(contact, (3, 5, 7))]
    assert sch.last_policy_drops == [] and feas.dropped_plan == []


@pytest.mark.parametrize("arm, kept", [
    ("D1", (0, 1, 2, 3, 4)), ("D2", (0, 1, 2, 3, 4)), ("D3", (0, 1, 2, 3, 4)),
    ("D5-unit", (0, 1, 2, 4, 6)), ("D5-devices", (0, 1, 2, 4, 6)),
])
def test_the_cliff_through_build_contact_queue_for_the_d_arms(arm, kept):
    """unit_U3b.md section 7.2 and the user's decision 6: under ``subset`` each
    D arm keeps its own members; ``last_policy_drops`` names the rest as
    ``budget`` while ``last_feasibility`` stays None (the pin at
    test_p3_final_fixes_mule.py:138). Under ``whole`` the route is empty and
    the whole contact is reported."""
    (contact,), _ = _cliff(1e9)
    queue, sch = _cliff(60.0, selector=D_ARMS[arm](), member_admission="subset")
    (stop,) = queue
    assert stop.devices == _in_stop_order(contact, kept)
    rest = [d for d in contact.devices if d not in stop.devices]
    ((reported, why),) = sch.last_policy_drops
    assert (reported.devices, why) == (tuple(rest), REASON_BUDGET)
    assert reported.position == contact.position and sch.last_feasibility is None
    assert reported == left_out([contact], queue, member_subsets=MemberSubsets(
        sch.last_plan_deadlines, sch.device_states))[0]
    queue, sch = _cliff(60.0, selector=D_ARMS[arm]())
    assert queue == [] and sch.last_feasibility is None
    assert sch.last_policy_drops == [(contact, REASON_BUDGET)]


def test_a_d_arms_drop_is_labelled_by_the_clause_refusing_it_alone():
    """Decision 6: labelled as the in-flight re-plan labels a baseline's drop
    (fl_scheduler.py, replan_remainder): the clause that refuses the contact
    on its own from the plan's start under the arm's rule, priced as Pass 1,
    else ``budget`` (the higher-ranked contacts took the budget). The energy
    capacity lies between what the field-wide contact needs as Pass 1 and as
    Pass 2 (half the dwell), so the contact is refused for energy, before its
    budget can bind, only as Pass 1 prices it."""
    spec = _t2_spec("narrow")
    model = spec.feasibility_model(rf_range_m=60.0, theta_bytes=52)
    capacity = 12000.0
    tight = dataclasses.replace(model, ferry=dataclasses.replace(model.ferry,
                                                                 energy_capacity_j=capacity))
    clock = MissionClock()
    sch = FLScheduler(now_fn=clock, mission_budget_s=200.0, feasibility_model=tight,
                      deadline_time_scale=T2_UNIT, target_selector=MaxAoIPolicy())
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=CLIFF_IDS, issued_round=0,
                                  issued_at=clock()), registry_records=_records())
    sch.start_mission()
    assert sch.build_contact_queue(rf_range_m=spec.link.range_planar_m("narrow"),
                                   mule_pose=DOCK) == []
    ((contact, why),) = sch.last_policy_drops
    assert why == REASON_ENERGY
    bound = sch.feasibility_model
    need = {}
    for kind in (COLLECT, DELIVER):
        leg = bound.leg(DOCK, contact, pass_kind=kind)
        need[kind] = (bound.ferry.p_move_w * (leg.transit_s + leg.return_s)
                      + bound.ferry.p_hover_w * leg.dwell_s)
    assert need[DELIVER] < capacity < need[COLLECT]
    # Planned 150 s after the stamp, from the plan's own clock, the contact
    # (about 99 s) would be home after the budget's end, 200 s after the stamp:
    # the budget clause refuses it before the energy clause is reached.
    t0 = sch.mission_start_ts
    assert sch.build_contact_queue(rf_range_m=spec.link.range_planar_m("narrow"),
                                   now=t0 + 150.0, mule_pose=DOCK) == []
    assert [(wp.devices, why) for wp, why in sch.last_policy_drops] == [
        (contact.devices, REASON_BUDGET)]


@pytest.mark.parametrize("arm", ["D1", "D2", "D3", "D5-unit", "D5-devices"])
def test_a_d_arms_drop_is_labelled_under_its_own_in_flight_rule(arm):
    """Decision 6: the rule is the arm's in-flight rule, the budget only for
    D1-D3 and D5 (Freeze Amendment 8), not S3b's. At the recorded deadline
    unit the field-wide contact is due 60 s after takeoff, before its dwell
    ends, so S3b's rule would call it ``overdue``; the arm's rule reads no
    deadline, and the budget it would overrun is the label."""
    queue, sch = _cliff(60.0, time_scale=1.0, selector=D_ARMS[arm]())
    ((contact, why),) = sch.last_policy_drops
    assert queue == [] and sch.last_feasibility is None
    assert set(contact.devices) == set(CLIFF_IDS) and why == REASON_BUDGET
    t0 = sch.mission_start_ts
    assert contact.deadline_ts == t0 + 60.0
    assert sch.feasibility_model.admit(FlightState(DOCK, t0), contact, rule=RULE_DEADLINE_BUDGET,
                                       budget_end=t0 + 60.0).reason == REASON_OVERDUE


def test_a_d_arms_drops_are_labelled_as_its_in_flight_re_plan_labels_them():
    """Decision 6 on random simulated-clock instances of every D arm, under
    both admissions: the in-flight re-plan (``replan_remainder``) of each
    reported contact alone, from the takeoff state, drops it with the
    reported reason, or keeps it, when it fits alone, and then the report
    says ``budget``, the budget the policy's higher-ranked contacts took."""
    seen = dict.fromkeys(("dropped", "kept", "energy", "subset"), 0)
    arms = ("D1", "D2", "D3", "D4", "D5-unit", "D5-devices")
    for admission in ("whole", "subset"):
        for seed in range(400):
            case = _case(seed, clock="sim", arms=arms)
            if admission == "subset" and case.arm == "D4":
                continue
            _, sch = _drive(FLScheduler, case, member_admission=admission)
            start = FlightState(DOCK, case.now)
            end = None if case.budget is None else sch.mission_start_ts + case.budget
            for wp, why in sch.last_policy_drops:
                res = sch.replan_remainder([wp], state=start, budget_end=end)
                if res.dropped:
                    assert res.dropped == ((wp, why),), (admission, seed, case.arm)
                    seen["dropped"] += 1
                else:
                    assert why == REASON_BUDGET, (admission, seed, case.arm)
                    seen["kept"] += 1
                seen["energy"] += why == REASON_ENERGY
                seen["subset"] += admission == "subset"
    assert min(seen.values()) >= 5, seen


def test_a_d_arms_drops_are_this_plans_only_at_every_early_return(monkeypatch):
    """Critic A9: ``last_policy_drops`` is reset with the other diagnostics,
    so none of build_contact_queue's three early returns (no eligible device,
    none bucketed, no contact) reports the previous plan's drops again;
    ``last_feasibility`` stays None throughout."""
    queue, sch = _cliff(60.0, selector=MaxAoIPolicy())
    rf = _t2_spec("narrow").link.range_planar_m("narrow")

    def replan():
        sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=CLIFF_IDS, issued_round=1,
                                      issued_at=0.0))
        for st in sch.device_states.values():
            st.deadline_override_ts = None
        assert sch.build_contact_queue(rf_range_m=rf, mule_pose=DOCK) == []
        assert len(sch.last_policy_drops) == 1

    assert len(sch.last_policy_drops) == 1
    # 1. No eligible device.
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=(), issued_round=1,
                                  issued_at=0.0))
    assert sch.build_contact_queue(rf_range_m=rf, mule_pose=DOCK) == []
    assert sch.last_policy_drops == [] and sch.last_feasibility is None
    # 2. Eligible, but S3 buckets none (an override alone, no bucket yet).
    replan()
    for st in sch.device_states.values():
        st.is_in_slice, st.is_new, st.bucket = False, False, None
        st.deadline_override_ts = 5.0
    assert sch.build_contact_queue(rf_range_m=rf, mule_pose=DOCK) == []
    assert sch.last_plan_deadlines == {} and sch.last_policy_drops == []
    # 3. Bucketed, but S3a makes no contact.
    replan()
    monkeypatch.setattr(FLS, "cluster_by_rf_range", lambda **kw: [])
    assert sch.build_contact_queue(rf_range_m=rf, mule_pose=DOCK) == []
    assert sch.last_plan_deadlines and sch.last_policy_drops == []
    assert sch.last_feasibility is None


def test_the_wall_clock_reports_no_policy_drops():
    """Decision 6 is a simulated-clock trace field: without the ferry physics
    nothing is computed for it, even when the walk leaves contacts out."""
    t = {"now": 0.0}
    sch = FLScheduler(now_fn=lambda: t["now"], mission_budget_s=3.0,
                      feasibility_model=FeasibilityModel(cruise_speed_m_s=1.0),
                      target_selector=MaxAoIPolicy())
    ids = tuple(DeviceID(f"d{i}") for i in range(4))
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=ids, issued_round=1,
                                  issued_at=0.0))
    for i, did in enumerate(ids):
        sch.device_states[did].last_known_position = (10.0 * (i + 1), 0.0, 0.0)
    route = sch.build_contact_queue(rf_range_m=1.0, mule_pose=DOCK)
    assert len(route) < len(ids) and sch.last_policy_drops == []


class _Recorder:
    """Records the keywords each call of a function gets, then calls it."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.fn(*args, **kwargs)


def test_member_admission_reaches_s3b_and_the_walks_before_takeoff_only(monkeypatch):
    """unit_U3b.md section 5.2: under ``subset`` build_contact_queue hands the
    carrier, built from this plan's deadlines and the scheduler's device
    states, to S3b and to the policy's walk; the in-flight re-plan and the
    pre-flight order check never do. Under ``whole`` the calls are exactly the
    recorded ones: no ``member_subsets`` keyword at all."""
    (contact,), sch = _cliff(1e9)
    t0 = sch.mission_start_ts
    gate = _Recorder(S3B.filter_feasible)
    monkeypatch.setattr(S3B, "filter_feasible", gate)
    late, end = FlightState(DOCK, t0 + 30.0), t0 + 60.0
    for admission in ("subset", "whole"):
        gate.calls.clear()
        queue, sch = _cliff(60.0, member_admission=admission, validate_flown_order=True,
                            replan_fallback="trim")
        sch.replan_remainder(queue or [contact], state=late, budget_end=end)
        first, *rest = gate.calls
        assert rest and all("member_subsets" not in kw for kw in rest)
        if admission == "whole":
            assert "member_subsets" not in first
        else:
            carrier = first["member_subsets"]
            assert isinstance(carrier, MemberSubsets)
            assert carrier.deadlines == sch.last_plan_deadlines
            assert carrier.device_states is sch.device_states
    for admission in ("subset", "whole"):
        policy = MaxAoIPolicy()
        walk = _Recorder(policy.admit_and_order)
        policy.admit_and_order = walk
        queue, sch = _cliff(60.0, selector=policy, member_admission=admission)
        sch.replan_remainder(queue or [contact], state=late, budget_end=end)
        first, *rest = walk.calls
        assert rest and all("member_subsets" not in kw for kw in rest)
        assert ("member_subsets" in first) == (admission == "subset")


def test_in_flight_a_subset_scheduler_never_reduces_a_contact():
    """unit_U3b.md section 3.4: under ``subset`` the H and D arms reduce
    contacts only before takeoff. From a mid-flight state their re-plan (and
    Pass 2's) returns only stops it was given, as ``replan_route``'s identity
    check requires, and never raises."""
    arms = tuple(a for a in ARMS if a != "D4")
    reduced = replans = 0
    for seed in range(250):
        case = _case(seed, clock="sim", arms=arms)
        out, sch = _drive(FLScheduler, case, member_admission="subset")
        eligible = filter_eligible(sch.device_states, now=case.now, beacon_window_s=30.0)
        bucketed = [d for d in eligible if sch.device_states[d].bucket is not None]
        s3a = cluster_by_rf_range(bucketed, sch.device_states, out["deadlines"], case.radius)
        reduced += any(wp not in s3a for wp in out["queue"])
        for kind in (COLLECT, DELIVER):
            res = out[f"replan_{kind.value}"]
            route = out["queue"] if kind is COLLECT else out["pass_2"]
            assert all(any(wp is r for r in route) for wp in res.route), (seed, kind)
            replans += res.changed
    assert reduced >= 10 and replans >= 10, (reduced, replans)


#: the U3b unit spec, section 3.4, found by a Phase 4 build probe (seed 3645): under
#: ``subset`` S3b keeps (d3,), (d4,) cut from (d2, d4) and (d0, d5) cut from
#: (d1, d0, d5); H1 flies them by bucket and distance, which does not fit, and
#: the check's whole re-admission drops the reduced (d0, d5).
PREFLIGHT = dict(
    pos={"d0": (103.1, 129.1), "d1": (121.7, 97.3), "d2": (-28.3, -135.9),
         "d3": (111.4, -15.8), "d4": (-33.8, -106.9), "d5": (107.8, 93.4)},
    new={"d0": False, "d1": False, "d2": True, "d3": True, "d4": False, "d5": False},
    phi={"d0": 208.1, "d1": 43.3, "d2": 49.3, "d3": 30.3, "d4": 180.2, "d5": 287.0},
)


def _preflight(admission):
    physics = FerryPhysics(dock=DOCK, member_dwell_s=_Dwell(1.09, 0.009), upload_s=_Upload(),
                           p_move_w=P_MOVE, p_hover_w=P_HOVER, range_m=40.0)
    model = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=1.0, ferry=physics)
    sch = FLScheduler(now_fn=lambda: NOW, mission_budget_s=140.1, feasibility_model=model,
                      validate_flown_order=True, replan_fallback="reorder",
                      member_admission=admission)
    ids = tuple(DeviceID(d) for d in PREFLIGHT["pos"])
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=ids, issued_round=1,
                                  issued_at=NOW))
    for did in ids:
        st = sch.device_states[did]
        st.last_known_position = PREFLIGHT["pos"][did] + (0.0,)
        st.is_new = PREFLIGHT["new"][did]
        st.deadline_fulfilment_s = PREFLIGHT["phi"][did]
    sch.start_mission()
    return sch, sch.build_contact_queue(rf_range_m=40.0, mule_pose=DOCK)


def test_the_preflight_checks_drops_of_reduced_stops_join_last_feasibility():
    """unit_U3b.md section 3.4: under ``subset`` the pre-flight order check's
    S3b re-admission can drop a reduced stop, which never happens under
    ``whole`` (routing/replan.py). The drop is reported: it joins
    ``last_feasibility`` under its reason, the object itself, so the mule
    widens and counts it."""
    sch, queue = _preflight("subset")
    check, feas = sch.last_order_check, sch.last_feasibility
    assert (check.order_used, [(wp.devices, why) for wp, why in check.dropped]) == (
        RP.ORDER_ARM, [(("d0", "d5"), REASON_BUDGET)])
    ((dropped, _),) = check.dropped
    s3a = cluster_by_rf_range(list(sch.last_plan_deadlines), sch.device_states,
                              sch.last_plan_deadlines, 40.0)
    assert any(set(dropped.devices) < set(wp.devices) for wp in s3a)
    assert any(wp is dropped for wp in feas.dropped_budget)
    assert _devices(queue) == _devices(feas.kept) == [("d3",), ("d4",)]
    assert mission_planned_devices(queue, feas) == 6
    whole, _ = _preflight("whole")
    assert whole.last_order_check.dropped == ()
