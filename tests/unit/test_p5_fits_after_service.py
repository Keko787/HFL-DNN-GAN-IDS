"""FeRRy Phase 5 (unit U4): the pair mask's predicate, ``FLScheduler.fits_after_service``.

What is pinned (the Phase 5 spec, other choices 2; the user's decision 1 (a)):

* **T1.** Every pair the predicate admits passes, and every pair it refuses
  fails, an independent reference: the state after the service built by hand
  (the arrival's clock plus the dwell, its energy plus P_hover times the
  dwell, and under ``deadline_bounds="delivery"`` its ``deliver_by`` lowered
  to the own deadline of each collected member the plan did not cap, dated by
  the mule's record), then a plain ``FeasibilityModel.fold(skip=False)`` of the
  remainder in the pair's order with every stop whose members are all capped
  protected, or, for home, the landing inequality (the end of the dwell, the
  return leg and the upload within the budget end and, under ``delivery``,
  ``deliver_by``; the energy clause with a capacity). Random plan-mode
  instances: N from 2 to 12, links of 1 to 4 classes, exempt and mixed stops,
  all three ``deadline_bounds``, capacities on and off, with and without a
  budget, a beacon insert among the stops, the record passed or not; every
  class of the arrival view times every next stop, or home. Every node admits
  and refuses stop pairs and home, refuses for each clause it can, and has a
  case where the protection decides and, under ``delivery``, one where the
  lowering does.
* **A stop pair is the whole rest's fold**: the result equals
  ``fold_remainder`` of the pair's order from the state after the service,
  the plan's exempt stops protected: the fold FX's ``fits`` runs, and the one
  the departure check runs next under ``replan`` when the service goes as
  priced. Under ``abort`` that check folds the chosen stop alone, which every
  admitted pair passes; the mask is the stricter.
* **Home** is the fold of the stop served alone: one verdict from the arrival
  to the landing; the landing's clauses in ``admit``'s order (on board, then
  budget, then energy), each at its boundary; nothing gated without a budget.
* **The stop's own deadline clause** is tested by neither pair, under every
  ``deadline_bounds``. Under ``delivery`` its members' dates still bound the
  landing and the later stops through ``deliver_by``, the stop's deadline
  standing in for a member nothing else dates.
* **``deliver_by``** is lowered as the supervisor lowers it after a stop: by
  the uncapped collected members only, dated by the plan, else by the record
  passed, else by the stop served; under ``collection`` and
  ``delivery_per_stop`` the state's value is carried and never read.
* **Refusals.** Legacy mode, no commit yet, Pass 2, a bad dwell, a state away
  from the stop, collected devices outside it, a remainder holding its
  members. A call is pure.
* **Additive only (Freeze Rule 1).** ``fl_scheduler.py`` differs from 386c275's
  by this one method: every other statement and definition is the recorded
  one.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import math
import random
import subprocess
from pathlib import Path

import pytest

from hermes.l1.contact_link import CLASSES, CLASSES_WITH_10MHZ
from hermes.l1.mission_clock import SIM_EPOCH_S
from hermes.mule.ferry import FerryRuntime, FerrySpec
from hermes.scheduler import FLScheduler, FLSchedulerError
from hermes.scheduler.plan import AgeCapSpec, PlanClass, PlanOptions, PlanScoreParams, PlanSetup
from hermes.scheduler.policies.cross_heuristic import moved_to_front
from hermes.scheduler.stages.s3b_feasibility import (
    DEADLINE_BOUNDS,
    REASON_BUDGET,
    REASON_DELIVERY,
    REASON_ENERGY,
    REASON_OVERDUE,
    RULE_DEADLINE_BUDGET,
    FeasibilityModel,
    FerryPhysics,
    FlightState,
)
from hermes.scheduler.stages.s3d_age_cap import stop_deadline
from hermes.types import Bucket, ContactWaypoint, DeviceID, MissionPass
from hermes.types.scheduler import PlanCommit

REPO = Path(__file__).resolve().parents[2]
REF_COMMIT = "386c275"
COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
RF = 60.0
T0 = SIM_EPOCH_S
DOCK = (0.0, 0.0, 0.0)
P_MOVE, P_HOVER = 143.6, 168.5
_SCORE = dict(v=-1.0, delta_s=10.0, time=0.01, coverage=0.0, link=0.0, energy_j=0.0,
              energy=0.0, served_weight=1.0, demand_weight=1.0)


def _wp(x, y, *devs, deadline=T0 + 1e4):
    return ContactWaypoint(position=(float(x), float(y), 0.0),
                           devices=tuple(DeviceID(d) for d in devs),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=float(deadline))


def _commit(band, band_index, planned, capped, budget_end):
    """A plan-mode commit of ``planned`` (its demand), ``capped`` aged S = 3."""
    demand = [d for wp in planned for d in wp.devices]
    return PlanCommit(
        mission_round=5, band=band, band_index=band_index, band_class_policy="search",
        queue=tuple(planned), demand=tuple(demand), weights={d: 1.0 for d in demand},
        budget_end=budget_end, t_ref_s=200.0, score=_SCORE,
        constants=PlanScoreParams().constants(len(demand)), search_mode="exact",
        n_candidates=1, cap_s=3 if capped else None,
        ages={d: (3 if d in capped else 1) for d in demand} if capped else {},
        capped=frozenset(capped))


def _plan_scheduler(setup, model, positions, commit, plan_deadlines, *, now=T0):
    """A plan-mode scheduler after the commit: b̄'s model, the commit, the plan's dates."""
    model = dataclasses.replace(model, ferry=model.ferry.bind(positions))
    sch = FLScheduler(now_fn=lambda: now, feasibility_model=model, replan_fallback="trim",
                      member_admission="subset", plan_mode="ferry", plan=setup)
    sch.last_plan = commit
    sch.last_plan_deadlines = dict(plan_deadlines)
    return sch


# --------------------------------------------------------------------------- #
# T1: random plan-mode instances on the real runtime
# --------------------------------------------------------------------------- #

def _class_set(rng, n_classes):
    if n_classes == 1:
        return ("wide",)
    if n_classes == 2:
        return ("wide", rng.choice(["medium", "narrow"]))
    return CLASSES if n_classes == 3 else CLASSES_WITH_10MHZ


def _instance(rng, *, bounds, capacity_on, n_classes, n_devices):
    """One Pass-1 arrival in plan mode: the runtime, the scheduler after a
    hand-built commit, the stop reached, the remainder (in plan order, with a
    beacon insert at times), the arrival state and the arrival view."""
    classes = _class_set(rng, n_classes)
    cap_j = rng.uniform(15e3, 90e3) if capacity_on else None
    spec = FerrySpec.from_config(rf_range_m=RF, seed=rng.randrange(10 ** 6), contact_band="wide",
                                 band_classes=classes, payload_bytes=1_000_000,
                                 contact_regime=rng.choice(["clean", "jittery"]),
                                 deadline_bounds=bounds, energy_capacity_j=cap_j)
    rt = FerryRuntime(spec, None, rf_range_m=RF)
    rt.set_payload(theta_bytes=18_756, synth_bytes=64)
    bbar = rng.choice(classes)
    rt.set_band(bbar)
    n_stops = 1 if rng.random() < 0.3 else rng.randint(2, min(n_devices, 6))
    sizes = [1] * n_stops
    for _ in range(n_devices - n_stops):
        sizes[rng.randrange(n_stops)] += 1
    t_arr = T0 + rng.uniform(0.0, 120.0)
    p_cap = rng.choice([0.0, 0.3, 0.7, 1.0])
    positions, deadlines, stops, capped = {}, {}, [], set()
    for size in sizes:
        cx, cy = rng.uniform(-150.0, 150.0), rng.uniform(-150.0, 150.0)
        members = []
        for _ in range(size):
            r, a = rt.range_planar_m * math.sqrt(rng.random()), rng.uniform(0.0, 2.0 * math.pi)
            did = DeviceID(f"d{len(positions)}")
            positions[did] = (cx + r * math.cos(a), cy + r * math.sin(a), 0.0)
            late = rng.random() < 0.25
            deadlines[did] = t_arr + (rng.uniform(-30.0, 60.0) if late
                                      else rng.uniform(60.0, 700.0))
            if rng.random() < p_cap:
                capped.add(did)
            members.append(did)
        stops.append(ContactWaypoint(
            position=(cx, cy, 0.0), devices=tuple(members), bucket=Bucket.SCHEDULED_THIS_ROUND,
            deadline_ts=stop_deadline(members, deadlines=deadlines, capped=capped)))
    k, rest = stops[0], stops[1:]
    rng.shuffle(rest)
    # Stops the beacon hook inserted after the plan, outside it and dated by the
    # mule's record: at times one of the remainder, at times the stop reached.
    inserted = [rest.pop()] if rest and rng.random() < 0.25 else []
    planned = list(rest)
    half = len(rest) // 2
    rest = rest[:half] + inserted + rest[half:]
    if rng.random() >= 0.15:
        planned.insert(0, k)
    plan_devices = [d for wp in planned for d in wp.devices]
    capped &= set(plan_devices)
    plan_deadlines = {d: deadlines[d] for d in plan_devices}
    record = dict(deadlines)                  # the mule's: the plan's and the insert's
    budget_end = None if rng.random() < 0.08 else t_arr + rng.uniform(20.0, 600.0)
    setup = PlanSetup(options=PlanOptions(), classes=rt.plan_classes(), reference="wide",
                      t_ref_s=200.0, turnaround_s=spec.flight.turnaround_s)
    commit = _commit(bbar, rt.band_index, planned, capped, budget_end)
    sch = _plan_scheduler(setup, rt.feasibility_model(), positions, commit, plan_deadlines,
                          now=t_arr)
    e_arr = rng.uniform(0.0, 0.5 * cap_j) if cap_j else rng.uniform(0.0, 3e4)
    deliver_by = math.inf
    if bounds == "delivery" and rng.random() < 0.5:
        deliver_by = t_arr + rng.uniform(0.0, 300.0)      # updates already on board
    passed = record if rng.random() < 0.7 else None
    return dict(
        rt=rt, sch=sch, k=k, rest=rest, bounds=bounds, budget_end=budget_end, cap_j=cap_j,
        capped=frozenset(capped), passed=passed, state=FlightState(k.position, t_arr, e_arr,
                                                                   deliver_by),
        # Who dates a collected member: the record when it is passed, else the plan.
        dates=record if passed is not None else plan_deadlines,
        view=rt.arrival_view(k, positions, t_arr, pass_kind=COLLECT),
        positions=positions,
    )


def _after(inst, dwell, collected, *, lower=True):
    """The state after the service, by hand."""
    state, k = inst["state"], inst["k"]
    deliver_by = state.deliver_by
    if inst["bounds"] == "delivery" and lower:
        for did in collected:
            if did not in inst["capped"]:
                deliver_by = min(deliver_by, inst["dates"].get(did, k.deadline_ts))
    p_hover = inst["rt"].spec.flight.energy.p_hover_w
    return FlightState(k.position, state.clock + dwell, state.energy_j + p_hover * dwell,
                       deliver_by)


def _reference(inst, order, dwell, collected, *, protect=True, lower=True):
    """The independent verdict: a plain fold on b̄'s model, or the landing inequality."""
    rt = inst["rt"]
    after = _after(inst, dwell, collected, lower=lower)
    if order:
        model = rt.feasibility_model()
        model = dataclasses.replace(model, ferry=model.ferry.bind(inst["positions"]))
        exempt = frozenset(wp for wp in order if all(d in inst["capped"] for d in wp.devices))
        return model.fold(order, after, rule=RULE_DEADLINE_BUDGET, budget_end=inst["budget_end"],
                          pass_kind=COLLECT, skip=False,
                          protected=exempt if protect else frozenset()).ok
    if inst["budget_end"] is None:
        return True
    flight = rt.spec.flight
    back = flight.leg_s(inst["k"].position, flight.dock)
    landing = after.clock + back + rt.predicted_upload_s()
    if inst["bounds"] == "delivery" and landing > after.deliver_by:
        return False
    if landing > inst["budget_end"]:
        return False
    cap = inst["cap_j"]
    return cap is None or after.energy_j + flight.energy.p_move_w * back <= cap


def _pairs(inst):
    """Every class of the arrival view times every next stop (home alone at the last)."""
    rest = inst["rest"]
    nexts = list(range(len(rest))) or [None]
    for entry in inst["view"].classes:
        for index in nexts:
            yield entry, index, ([] if index is None else moved_to_front(rest, index))


def _ask(inst, entry, order):
    return inst["sch"].fits_after_service(
        order, served_at=inst["k"], state=inst["state"], dwell_s=entry.dwell_s,
        collected=entry.targets, budget_end=inst["budget_end"], deadlines=inst["passed"])


#: Per node, 60 arrivals: N runs 2..12 and the link's classes 1..4 in turn, so
#: every (N, classes) combination occurs (lcm(11, 4) = 44 < 60).
T1_INSTANCES = 60


@pytest.mark.parametrize("capacity_on", [False, True], ids=["no-capacity", "capacity"])
@pytest.mark.parametrize("bounds", DEADLINE_BOUNDS)
def test_t1_the_mask_never_admits_an_infeasible_pair_nor_refuses_a_feasible_one(
        bounds, capacity_on):
    rng = random.Random(f"t1-{bounds}-{capacity_on}")
    seen = collections.Counter()
    for i in range(T1_INSTANCES):
        n_classes, n_devices = 1 + i % 4, 2 + i % 11
        inst = _instance(rng, bounds=bounds, capacity_on=capacity_on, n_classes=n_classes,
                         n_devices=n_devices)
        assert len(inst["view"].classes) == n_classes
        assert sum(len(wp.devices) for wp in [inst["k"]] + inst["rest"]) == n_devices
        for entry, index, order in _pairs(inst):
            got = _ask(inst, entry, order)
            ref = _reference(inst, order, entry.dwell_s, entry.targets)
            assert got.ok == ref, (bounds, capacity_on, i, entry.name, index)
            kind = "home" if index is None else "stop"
            seen[(kind, got.ok)] += 1
            seen[(n_classes, got.ok)] += 1
            if not got.ok:
                seen[got.rejected[0][1]] += 1
            if order and _reference(inst, order, entry.dwell_s, entry.targets,
                                    protect=False) != ref:
                seen["protection decides"] += 1
            if _reference(inst, order, entry.dwell_s, entry.targets, lower=False) != ref:
                seen["lowering decides"] += 1
            if any(0 < len(set(wp.devices) & inst["capped"]) < len(wp.devices) for wp in order):
                seen["mixed stop"] += 1
    for kind in ("stop", "home"):
        assert seen[(kind, True)] and seen[(kind, False)], seen
    for n_classes in (1, 2, 3, 4):
        assert seen[(n_classes, True)] and seen[(n_classes, False)], seen
    assert seen[REASON_OVERDUE] and seen[REASON_BUDGET], seen
    assert bool(seen[REASON_ENERGY]) == capacity_on, seen
    assert bool(seen[REASON_DELIVERY]) == (bounds == "delivery"), seen
    assert bool(seen["lowering decides"]) == (bounds == "delivery"), seen
    assert seen["protection decides"] and seen["mixed stop"], seen


def test_a_stop_pair_is_the_whole_rest_fold_after_the_service():
    """``fold_remainder`` of the pair's order from the state after the service,
    exempt stops protected, the whole result (route, rejections, verdicts,
    state and landing) alike: the fold FX's ``fits`` runs, and the one the
    departure check runs next under ``replan`` when the service goes as
    priced. Under ``abort`` that check folds the chosen stop alone
    (``MuleSupervisor._ferry_departure``), whose verdict is the fold's first:
    every admitted pair passes it, and the mask refuses some pairs it passes,
    since the whole rest must fit."""
    rng = random.Random("departure-check")
    compared = stricter = 0
    for i in range(80):
        inst = _instance(rng, bounds=DEADLINE_BOUNDS[i % 3], capacity_on=bool(i % 2),
                         n_classes=1 + i % 4, n_devices=2 + i % 11)
        sch = inst["sch"]
        for entry, index, order in _pairs(inst):
            if index is None:
                continue
            after = _after(inst, entry.dwell_s, entry.targets)
            check = sch.fold_remainder(order, state=after, budget_end=inst["budget_end"],
                                       pass_kind=COLLECT, protected=sch.plan_protected(order))
            got = _ask(inst, entry, order)
            assert got == check
            head = sch.fold_remainder(order[:1], state=after, budget_end=inst["budget_end"],
                                      pass_kind=COLLECT, protected=sch.plan_protected(order[:1]))
            assert head.verdicts == got.verdicts[:1]
            assert head.ok or not got.ok
            stricter += head.ok and not got.ok
            compared += 1
    assert compared > 200 and stricter > 0


# --------------------------------------------------------------------------- #
# Hand-built cases on synthetic physics
# --------------------------------------------------------------------------- #

class _Dwell:
    """A member's dwell, ``a`` seconds wherever it is."""

    def __init__(self, a):
        self.a = a

    def __call__(self, d, pass_kind, offset):
        return self.a


class _Upload:
    def __call__(self):
        return 0.5


class _NoOutage:
    def __call__(self, d):
        return 0.0


K = _wp(50, 0, "a", "b", "c", "e", deadline=T0 + 1900.0)   # the stop reached
S_EXEMPT = _wp(50, 50, "x", deadline=T0 - 5.0)             # every member capped, and late
S_MIXED = _wp(100, 0, "y", "z", deadline=T0 - 5.0)         # y capped, z not: late for z
S_LATER = _wp(150, 0, "w")
POSITIONS = {d: wp.position for wp in (K, S_EXEMPT, S_MIXED, S_LATER) for d in wp.devices}
PLAN_DATES = {DeviceID("a"): T0 + 2000.0, DeviceID("b"): T0 + 1500.0,
              DeviceID("x"): T0 - 5.0, DeviceID("y"): T0 - 50.0, DeviceID("z"): T0 - 5.0,
              DeviceID("w"): T0 + 1e4}


def _synthetic(*, bounds="collection", capacity=None, capped=("b", "x", "y"), member_s=2.0,
               queue=(K, S_EXEMPT, S_MIXED, S_LATER), budget_end=T0 + 1e4, dates=None):
    """A plan-mode scheduler on one synthetic class (a dwell per member, 5 m/s,
    a 0.5 s upload) after a commit of ``queue``, its members dated by ``dates``
    (default :data:`PLAN_DATES`); there ``c`` and ``e`` at K are not in the
    plan's dates (as a beacon insert's members would not be)."""
    physics = FerryPhysics(dock=DOCK, member_dwell_s=_Dwell(member_s), upload_s=_Upload(),
                           p_move_w=P_MOVE, p_hover_w=P_HOVER, energy_capacity_j=capacity,
                           deadline_bounds=bounds, range_m=RF)
    model = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=1.0, ferry=physics)
    cls = PlanClass(name="wide", index=0, radius_m=RF, model=model, outage=_NoOutage())
    setup = PlanSetup(options=PlanOptions(cap=AgeCapSpec(s_missions=3 if capped else None)),
                      classes=(cls,), reference="wide", t_ref_s=200.0, turnaround_s=30.0)
    capped = frozenset(DeviceID(d) for d in capped)
    commit = _commit("wide", 0, queue, capped, budget_end)
    plan_dates = {d: t for d, t in (PLAN_DATES if dates is None else dates).items()
                  if any(d in wp.devices for wp in queue)}
    return _plan_scheduler(setup, model, POSITIONS, commit, plan_dates), model


def _arrive(clock=T0 + 100.0, energy=1000.0, deliver_by=math.inf):
    return FlightState(K.position, clock, energy, deliver_by)


def test_home_is_the_fold_of_the_stop_served_alone_up_to_the_landing():
    sch, model = _synthetic(queue=(K,), capped=())
    state, dwell = _arrive(), 6.25
    res = sch.fits_after_service([], served_at=K, state=state, dwell_s=dwell,
                                 collected=K.devices, budget_end=T0 + 1e4)
    after = FlightState(K.position, state.clock + dwell, state.energy_j + P_HOVER * dwell,
                        math.inf)
    landing = after.clock + 10.0 + 0.5                       # 50 m home at 5 m/s, then the upload
    assert res.ok and res.route == (K,) and res.rejected == ()
    assert res.state == after and res.home == landing
    (v,) = res.verdicts
    assert (v.ok, v.reason, v.arrival, v.finish, v.home, v.next_state) == (
        True, None, state.clock, after.clock, landing, after)


@pytest.mark.parametrize("bounds", DEADLINE_BOUNDS)
def test_the_landing_is_held_to_each_clause_at_its_boundary_in_admits_order(bounds):
    """On board (route-level ``delivery`` only), then the budget, then the energy;
    each admits at equality and refuses just past it, as ``admit`` compares."""
    state, dwell = _arrive(deliver_by=T0 + 5000.0), 6.25
    landing = state.clock + dwell + 10.0 + 0.5
    need = state.energy_j + P_HOVER * dwell + P_MOVE * 10.0

    def ask(*, budget_end, capacity, deliver_by=state.deliver_by):
        sch, _ = _synthetic(bounds=bounds, capacity=capacity, queue=(K,), capped=())
        res = sch.fits_after_service(
            [], served_at=K, state=dataclasses.replace(state, deliver_by=deliver_by),
            dwell_s=dwell, collected=(), budget_end=budget_end)
        return res.rejected[0][1] if res.rejected else None

    eps = 1e-6
    assert ask(budget_end=landing, capacity=need, deliver_by=landing) is None
    on_board = REASON_DELIVERY if bounds == "delivery" else None
    assert ask(budget_end=landing, capacity=need, deliver_by=landing - eps) == on_board
    assert ask(budget_end=landing - eps, capacity=need) == REASON_BUDGET
    assert ask(budget_end=landing, capacity=need - eps) == REASON_ENERGY
    # Several clauses fail: the first in admit's order is the reason.
    assert ask(budget_end=landing - eps, capacity=need - eps, deliver_by=landing - eps) == (
        on_board or REASON_BUDGET)
    assert ask(budget_end=landing - eps, capacity=need - eps) == REASON_BUDGET
    # No budget, no gate: S3b's opt-in contract, the capacity and deliver_by included.
    assert ask(budget_end=None, capacity=need - eps, deliver_by=landing - eps) is None


@pytest.mark.parametrize("bounds", DEADLINE_BOUNDS)
def test_the_stops_own_deadline_clause_is_never_tested(bounds):
    """The mule is at the stop whichever pair it picks: its own clause
    (``overdue``) refuses no pair, at home or before a later stop, under every
    ``deadline_bounds``. The member collected, ``a``, is dated by the plan
    after the landing, so no member's date is at stake (the next test)."""
    late = dataclasses.replace(K, deadline_ts=T0 - 1e3)
    sch, _ = _synthetic(bounds=bounds, queue=(late, S_LATER), capped=())
    for rest in ([], [S_LATER]):
        res = sch.fits_after_service(rest, served_at=late, state=_arrive(), dwell_s=6.0,
                                     collected=("a",), budget_end=T0 + 1e4)
        assert res.ok


@pytest.mark.parametrize("bounds", DEADLINE_BOUNDS)
def test_under_route_level_delivery_the_collected_dates_still_bound_the_rest(bounds):
    """What the stop's own clause would test reaches the mask through
    ``deliver_by`` under ``delivery`` alone, which refuses home and the later
    stop alike as ``delivery``: (1) every member dated by the plan at the
    stop's deadline, as S3a dates a stop, and the landing past it; (2) a late
    stop and a member neither the plan nor the record dates (``e``), for whom
    the stop's deadline stands in until the record dates it. The other two
    ``deadline_bounds`` admit every pair here."""
    route_level = bounds == "delivery"
    dated = {d: K.deadline_ts for d in K.devices}
    sch, _ = _synthetic(bounds=bounds, queue=(K, S_LATER), capped=(), dates=dated)
    state = _arrive(clock=K.deadline_ts - 10.0)           # home lands at K's deadline + 6.5 s
    for rest in ([], [S_LATER]):
        res = sch.fits_after_service(rest, served_at=K, state=state, dwell_s=6.0,
                                     collected=K.devices, budget_end=T0 + 1e4)
        assert [r for _, r in res.rejected] == ([REASON_DELIVERY] if route_level else [])
        if not rest:
            assert res.home == K.deadline_ts + 6.5
            assert res.state.deliver_by == (K.deadline_ts if route_level else math.inf)
    late = dataclasses.replace(K, deadline_ts=T0 - 1e3)
    sch, _ = _synthetic(bounds=bounds, queue=(late, S_LATER), capped=())
    assert DeviceID("e") not in sch.last_plan_deadlines
    for record, date in ((None, late.deadline_ts), ({DeviceID("e"): T0 + 5e3}, T0 + 5e3)):
        for rest in ([], [S_LATER]):
            res = sch.fits_after_service(rest, served_at=late, state=_arrive(), dwell_s=6.0,
                                         collected=("e",), budget_end=T0 + 1e4,
                                         deadlines=record)
            refused = route_level and record is None
            assert [r for _, r in res.rejected] == ([REASON_DELIVERY] if refused else [])
            if not rest:
                assert res.state.deliver_by == (date if route_level else math.inf)


def test_deliver_by_is_lowered_by_the_uncapped_collected_members_as_the_mule_lowers_it():
    """Under ``delivery``: ``a`` by the plan (the plan's date first, whatever the
    record says), ``b`` not at all (capped), ``c`` by the record passed, ``e``
    by neither, so by the stop's own deadline; with no record, ``c`` falls back
    to the stop's deadline too."""
    sch, _ = _synthetic(bounds="delivery")
    state = _arrive(deliver_by=T0 + 9e3)

    def deliver_by(collected, deadlines):
        res = sch.fits_after_service([], served_at=K, state=state, dwell_s=4.0,
                                     collected=collected, budget_end=T0 + 1e4,
                                     deadlines=deadlines)
        return res.state.deliver_by

    record = {DeviceID("a"): T0 + 10.0, DeviceID("c"): T0 + 1800.0}
    assert deliver_by(["a"], record) == T0 + 2000.0           # the plan's date wins
    assert deliver_by(["b"], record) == T0 + 9e3              # capped: never lowers
    assert deliver_by(["c"], record) == T0 + 1800.0           # the record's
    assert deliver_by(["c"], None) == K.deadline_ts           # dated by the stop served
    assert deliver_by(["e"], record) == K.deadline_ts
    assert deliver_by(["a", "b", "c"], record) == T0 + 1800.0
    assert deliver_by([], record) == T0 + 9e3                 # nothing on board from here
    # The lowered bound decides: home lands after c's date but before a's.
    tight = {DeviceID("c"): state.clock + 4.0 + 10.0 + 0.5 - 1e-6}
    for collected, ok in ((["a"], True), (["c"], False), (["b", "a"], True)):
        res = sch.fits_after_service([], served_at=K, state=state, dwell_s=4.0,
                                     collected=collected, budget_end=T0 + 1e4, deadlines=tight)
        assert res.ok is ok and (ok or res.rejected[0][1] == REASON_DELIVERY)


@pytest.mark.parametrize("bounds", ["collection", "delivery_per_stop"])
def test_deliver_by_is_carried_and_never_read_unless_delivery_is_route_level(bounds):
    sch, _ = _synthetic(bounds=bounds)
    state = _arrive(deliver_by=T0 + 50.0)                     # a value the mule never holds here
    res = sch.fits_after_service([], served_at=K, state=state, dwell_s=4.0,
                                 collected=("a", "c", "e"), budget_end=T0 + 1e4,
                                 deadlines={DeviceID("c"): T0 + 1.0})
    assert res.ok and res.state.deliver_by == T0 + 50.0 < res.home


def test_the_exempt_stops_are_protected_and_the_mixed_ones_are_not():
    """``x`` is capped, so its late stop is flown for it (Phase 4 spec, other
    choices 9); ``y`` and ``z`` share a stop late for the uncapped ``z``."""
    sch, model = _synthetic()
    state = _arrive()
    after = FlightState(K.position, state.clock + 8.0, state.energy_j + P_HOVER * 8.0, math.inf)
    bound = dataclasses.replace(model, ferry=model.ferry.bind(POSITIONS))
    for rest, ok in (([S_EXEMPT, S_LATER], True), ([S_MIXED, S_LATER], False),
                     ([S_LATER, S_EXEMPT], True), ([S_LATER, S_MIXED], False)):
        res = sch.fits_after_service(rest, served_at=K, state=state, dwell_s=8.0,
                                     collected=K.devices, budget_end=T0 + 1e4)
        assert res.ok is ok and (ok or res.rejected == ((S_MIXED, REASON_OVERDUE),))
        unprotected = bound.fold(rest, after, rule=RULE_DEADLINE_BUDGET, budget_end=T0 + 1e4,
                                 skip=False)
        assert not unprotected.ok            # without the protection both are refused
    assert sch.plan_protected([S_EXEMPT, S_MIXED, S_LATER]) == {S_EXEMPT}


def test_the_state_after_the_service_charges_the_dwell_at_hover_power():
    sch, model = _synthetic()
    state = _arrive(clock=T0 + 33.5, energy=2500.0)
    res = sch.fits_after_service([S_LATER], served_at=K, state=state, dwell_s=12.75,
                                 collected=(), budget_end=T0 + 1e4)
    (v,) = res.verdicts
    leg = model.cost(K.position, S_LATER.position)[0]
    assert v.arrival == state.clock + 12.75 + leg
    assert v.next_state.energy_j == (state.energy_j + P_HOVER * 12.75) + P_MOVE * leg \
        + P_HOVER * 2.0
    assert res.ok


# --------------------------------------------------------------------------- #
# Refusals and purity
# --------------------------------------------------------------------------- #

def test_a_legacy_scheduler_never_reaches_it():
    sch = FLScheduler(now_fn=lambda: T0)
    with pytest.raises(FLSchedulerError, match="plan_mode='ferry'"):
        sch.fits_after_service([], served_at=K, state=_arrive(), dwell_s=1.0, collected=(),
                               budget_end=None)
    assert sch.plan_mode == "legacy" and sch.last_plan is None


def test_the_mask_prices_the_committed_plan_only():
    sch, _ = _synthetic()
    sch.last_plan = None
    with pytest.raises(FLSchedulerError, match="build_ferry_plan first"):
        sch.fits_after_service([], served_at=K, state=_arrive(), dwell_s=1.0, collected=(),
                               budget_end=None)


@pytest.mark.parametrize("kwargs, error, match", [
    (dict(pass_kind=DELIVER), ValueError, "Pass-1 arrivals only"),
    (dict(pass_kind="deliver"), ValueError, "Pass-1 arrivals only"),
    (dict(pass_kind="pass_3"), ValueError, "pass_3"),
    (dict(dwell_s=True), TypeError, "dwell_s"),
    (dict(dwell_s="4"), TypeError, "dwell_s"),
    (dict(dwell_s=-0.5), ValueError, "dwell_s"),
    (dict(dwell_s=math.nan), ValueError, "dwell_s"),
    (dict(dwell_s=math.inf), ValueError, "dwell_s"),
    (dict(state=FlightState((50.0, 0.5, 0.0), T0 + 100.0)), ValueError, "pose"),
    (dict(collected=("a", "x")), ValueError, "outside the stop"),
    (dict(rest=[_wp(10, 10, "a")]), ValueError, "members of the stop served"),
])
def test_a_call_the_mask_cannot_price_is_refused(kwargs, error, match):
    sch, _ = _synthetic()
    call = dict(rest=[S_LATER], served_at=K, state=_arrive(), dwell_s=4.0, collected=("a",),
                budget_end=T0 + 1e4)
    call.update(kwargs)
    rest = call.pop("rest")
    with pytest.raises(error, match=match):
        sch.fits_after_service(rest, **call)


def test_a_call_moves_nothing():
    rng = random.Random("pure")
    for i in range(12):
        inst = _instance(rng, bounds=DEADLINE_BOUNDS[i % 3], capacity_on=bool(i % 2),
                         n_classes=1 + i % 4, n_devices=4 + i % 9)
        sch = inst["sch"]
        before = (sch.last_plan, dict(sch.last_plan_deadlines), sch.feasibility_model,
                  sch.plan_setup, dict(sch.device_states), sch.last_feasibility,
                  inst["rt"].band, inst["rt"].range_planar_m)
        rest_before = list(inst["rest"])
        for entry, _, order in _pairs(inst):
            _ask(inst, entry, order)
        assert (sch.last_plan, dict(sch.last_plan_deadlines), sch.feasibility_model,
                sch.plan_setup, dict(sch.device_states), sch.last_feasibility,
                inst["rt"].band, inst["rt"].range_planar_m) == before
        assert inst["rest"] == rest_before


# --------------------------------------------------------------------------- #
# Additive only (Freeze Rule 1)
# --------------------------------------------------------------------------- #

def _shape(source: str):
    """(module-level statements, {qualified name: definition}) as AST dumps, line
    numbers aside: every top-level statement in order, each class's body
    statements other than its methods, and each function and method."""
    tree = ast.parse(source)
    top, defs = [], {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            body = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs[f"{node.name}.{sub.name}"] = ast.dump(sub)
                else:
                    body.append(ast.dump(sub))
            top.append(("class", node.name, tuple(body),
                        tuple(ast.dump(d) for d in node.decorator_list),
                        tuple(ast.dump(b) for b in node.bases)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = ast.dump(node)
            top.append(("def", node.name))
        else:
            top.append(ast.dump(node))
    return top, defs


def test_the_scheduler_gains_this_one_method_and_nothing_else():
    """Every statement and definition of 386c275's ``fl_scheduler.py``, the
    module's and the class's, is unchanged; the live module adds
    ``FLScheduler.fits_after_service`` alone. Skipped without git history."""
    try:
        blob = subprocess.run(
            ["git", "show", f"{REF_COMMIT}:hermes/scheduler/fl_scheduler.py"], cwd=REPO,
            capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:          # pragma: no cover - no git
        pytest.skip(f"git cannot show {REF_COMMIT}'s fl_scheduler.py: {e}")
    ref_top, ref_defs = _shape(blob.decode("utf-8"))
    live_top, live_defs = _shape((REPO / "hermes/scheduler/fl_scheduler.py").read_text(
        encoding="utf-8"))
    assert live_top == ref_top
    assert set(live_defs) - set(ref_defs) == {"FLScheduler.fits_after_service"}
    assert set(ref_defs) <= set(live_defs)
    changed = sorted(name for name in ref_defs if live_defs[name] != ref_defs[name])
    assert changed == []
