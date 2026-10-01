"""FeRRy Phase 4 (unit U3b): member-subset admission for the H and D arms.

The user's decision 4 (b) of 2026-09-30 (``unit_U3b.md``): under
``member_admission="subset"`` the plan before takeoff hands a carrier
(``s3b_feasibility.MemberSubsets``) to S3b's gate (H1-H3), to the budget walk
(D1-D3) or to FedCS's walk (D5), and a contact that fails whole is re-issued
with the members that still fit, in the arm's own member order, through the
plan package's one member walk (unit U3's ``admit_members``). Pinned here:

* the Phase 3 cliff instance (``device_positions(8, 777, 100.0)``, narrow,
  1 MB): with the carrier S3b and every D arm admit their own members at 60 s
  and 99.0 s, a contact that fits whole is flown as the same object, and
  without it both Phase 3 pins hold (empty at 60 s, the whole stop at 99.5 s);
* each arm's own member order (``unit_U3b.md`` section 2): S3b's contact key
  on one-member contacts (own deadline, Pass-1 dwell at the SNR offset, id;
  miss streak first under miss priority), the D1-D3 arm's own key (age,
  utility, index), FedCS's line 3; skip, not stop;
* S3b's complements, one reduced contact per clause in that clause's list,
  never ``dropped_plan``, all counted in S3c's planned count;
  ``budget_walk.left_out`` for the D arms' report; every walk reduces through
  the one member walk;
* the carrier is inert without a budget, refused for Pass 2, and taken only
  by the capable policies (``admits_member_subsets``; not D4);
* on random instances, synthetic and on the three bands' physics: every route
  passes the predicate as flown, every reduced contact is a proper subset of
  one contact with its members' own deadline and bucket, kept and dropped
  partition the devices, the admitted sets are maximal and equal an
  independent re-implementation of each arm's walk, also when a walk is given
  a mid-flight state with the carrier; ``fold_subsets`` equals
  ``fold(skip=True)`` wherever nothing is reduced, and wherever something is
  its verdicts, state and home are those of what it flew;
* Freeze Rule 1: without the carrier S3b, both walks and the five D policies
  equal the recorded modules of 6e6f92d (``tests/unit/_p4_ref.py``), object
  for object, on random instances and on the three bands;
* layering: none of the six files imports the plan package at module level,
  the recorded path never loads it, and either import order works.
"""

from __future__ import annotations

import ast
import inspect
import math
import os
import random
import subprocess
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.exp4.topology_builder import device_positions
from hermes.l1.mission_clock import MissionClock
from hermes.mule.ferry import FerrySpec
from hermes.mule.mule_main import mission_planned_devices
from hermes.scheduler import FLScheduler
from hermes.scheduler.policies.budget_walk import greedy_budget_walk, left_out
from hermes.scheduler.policies.fedcs_degraded import (
    VALUE_DEVICES,
    VALUE_UNIT,
    FedCSDegradedPolicy,
    _selection_key,
    fedcs_greedy_select,
)
from hermes.scheduler.policies.fedex_carp import FedExCarpPolicy
from hermes.scheduler.policies.max_aoi import MaxAoIPolicy
from hermes.scheduler.policies.oort import OortPolicy
from hermes.scheduler.policies.whittle import WhittlePolicy
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.stages import s3b_feasibility as S3B
from hermes.scheduler.stages.s3a_cluster import cluster_by_rf_range
from hermes.scheduler.stages.s3b_feasibility import (
    DEADLINE_BOUNDS,
    DEADLINE_BOUNDS_COLLECTION,
    DEADLINE_BOUNDS_DELIVERY,
    REASON_BUDGET,
    REASON_DELIVERY,
    REASON_ENERGY,
    REASON_OVERDUE,
    REASONS,
    RULE_BUDGET,
    RULE_DEADLINE_BUDGET,
    RULE_NONE,
    FeasibilityModel,
    FerryPhysics,
    FlightState,
    MemberSubsets,
    filter_feasible,
    fold_subsets,
)
from hermes.types import (
    BUCKET_PRIORITY,
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionPass,
    MuleID,
)
from hermes.types.registry import DeviceRecord, MissionSlice, SpectrumSig

from tests.unit import _p4_ref as REF

REPO = Path(__file__).resolve().parents[2]
DOCK = (0.0, 0.0, 0.0)

#: The D arms as the driver builds them, one fresh instance per call (Oort
#: remembers the round of the mission it last planned).
D_ARMS = {
    "D1": MaxAoIPolicy,
    "D2": OortPolicy,
    "D3": WhittlePolicy,
    "D5-unit": lambda: FedCSDegradedPolicy(VALUE_UNIT),
    "D5-devices": lambda: FedCSDegradedPolicy(VALUE_DEVICES),
}


# --------------------------------------------------------------------------- #
# The Phase 3 cliff instance (tests/unit/test_p3_final_fixes_mule.py)
# --------------------------------------------------------------------------- #

T2_LAYOUT = device_positions(8, 777, 100.0)
CLIFF_IDS = tuple(DeviceID(f"dev-{i}") for i in range(len(T2_LAYOUT)))
#: T2's deadline unit (Deadline(j) = t0 + 1500 s) and the runner's default (t0 + 60 s).
T2_UNIT, DEFAULT_UNIT = 25.0, 1.0


def _cliff(time_scale):
    """H1's plan of trial T2 (narrow, 1 MB, the seconds backhaul), a local copy
    of test_p3_final_fixes_mule._plan (lines 66-94) under no binding budget:
    the field-wide contact (from the queue, or from ``dropped_overdue`` at the
    default unit), the bound model, the states, the plan's own deadlines, the
    mission start, and the carrier the scheduler would build from them."""
    spec = FerrySpec.from_config(rf_range_m=60.0, seed=777, contact_band="narrow",
                                 backhaul_model="seconds", backhaul_period=750.0,
                                 backhaul_regime="clean", payload_bytes=1_000_000)
    clock = MissionClock()
    sch = FLScheduler(now_fn=clock, mission_budget_s=1e9,
                      feasibility_model=spec.feasibility_model(rf_range_m=60.0, theta_bytes=52),
                      deadline_time_scale=time_scale, refuse_deadline_overrides=True)
    records = [DeviceRecord(device_id=d, last_known_position=(x, y, 0.0),
                            spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)))
               for d, (x, y) in zip(CLIFF_IDS, T2_LAYOUT)]
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=CLIFF_IDS, issued_round=0,
                                  issued_at=clock()), registry_records=records)
    t0 = sch.start_mission()
    queue = sch.build_contact_queue(rf_range_m=spec.link.range_planar_m("narrow"), mule_pose=DOCK)
    (contact,) = queue or sch.last_feasibility.dropped_overdue
    deadlines = dict(sch.last_plan_deadlines)
    return SimpleNamespace(contact=contact, t0=t0, model=sch.feasibility_model,
                           states=sch.device_states, deadlines=deadlines,
                           carrier=MemberSubsets(deadlines, sch.device_states))


@pytest.fixture(scope="module")
def t2():
    return _cliff(T2_UNIT)


@pytest.fixture(scope="module")
def unit1():
    return _cliff(DEFAULT_UNIT)


def _in_stop_order(cl, numbers):
    wanted = {DeviceID(f"dev-{i}") for i in numbers}
    return tuple(d for d in cl.contact.devices if d in wanted)


def _s3b(cl, budget, **kw):
    return filter_feasible([cl.contact], now=cl.t0, mule_pose=DOCK,
                           mission_deadline_ts=cl.t0 + budget, model=cl.model, **kw)


def _d_route(cl, arm, budget, **kw):
    env = SelectorEnv(mule_pose=DOCK, now=cl.t0, mission_round=1)
    return D_ARMS[arm]().admit_and_order([cl.contact], cl.states, env,
                                         mission_deadline_ts=cl.t0 + budget,
                                         feasibility_model=cl.model, **kw)


def _home(cl, route, budget, rule):
    """Seconds after takeoff until the mule is back with the upload done,
    ``route`` flown as it stands; it must pass the predicate."""
    walk = cl.model.fold(route, FlightState(DOCK, cl.t0), rule=rule,
                         budget_end=cl.t0 + budget, skip=False)
    assert walk.ok
    return walk.home - cl.t0


@pytest.mark.parametrize("budget, kept, home, left", [
    (60.0, (0, 1, 2, 4, 6), 54.490, (3, 5, 7)),
    (99.0, (0, 1, 2, 3, 4, 5, 6), 81.903, (7,)),
])
def test_s3b_admits_the_members_that_fit_where_the_whole_contact_does_not(t2, budget, kept,
                                                                           home, left):
    """Test 1 at T2's unit: every member is due at t0 + 1500 s, so EDF ties and
    the cheapest dwell goes first. The complement reads ``budget`` and is
    counted in S3c's planned count; without the carrier the Phase 3 pin
    holds (the contact dropped whole, the same object)."""
    feas = _s3b(t2, budget, member_subsets=t2.carrier)
    (stop,) = feas.kept
    assert stop.devices == _in_stop_order(t2, kept)
    assert stop.position == t2.contact.position and stop.bucket is Bucket.NEW
    assert stop.deadline_ts == min(t2.deadlines[d] for d in stop.devices) == t2.t0 + 1500.0
    assert _home(t2, feas.kept, budget, RULE_DEADLINE_BUDGET) == pytest.approx(home, abs=1e-3)
    assert [c.devices for c in feas.dropped_budget] == [_in_stop_order(t2, left)]
    assert feas.dropped_overdue == [] and feas.dropped_energy == []
    assert feas.dropped_delivery == [] and feas.dropped_plan == []
    assert mission_planned_devices(feas.kept, feas) == len(CLIFF_IDS)
    for whole in (_s3b(t2, budget), _s3b(t2, budget, member_subsets=None)):
        assert whole.kept == [] and whole.dropped_budget == [t2.contact]
        assert whole.dropped_budget[0] is t2.contact


def test_at_the_default_unit_s3b_keeps_the_members_that_make_their_deadline(unit1):
    """Test 1 at the runner's default unit: every member is due at t0 + 60 s
    and the deadline clause is tested first, so at any budget the members left
    out read ``overdue``, with their own deadline; without the carrier the
    contact is dropped whole as ``overdue`` (the Phase 3 pin)."""
    cl = unit1
    for budget in (60.0, 99.0, 1000.0):
        feas = _s3b(cl, budget, member_subsets=cl.carrier)
        assert [c.devices for c in feas.kept] == [_in_stop_order(cl, (0, 1, 2, 4, 6))], budget
        assert _home(cl, feas.kept, budget, RULE_DEADLINE_BUDGET) == pytest.approx(54.490, abs=1e-3)
        assert [(c.devices, c.deadline_ts - cl.t0) for c in feas.dropped_overdue] == [
            (_in_stop_order(cl, (3, 5, 7)), 60.0)], budget
        assert feas.dropped_budget == [] and feas.dropped_energy == [] and feas.dropped_plan == []
        whole = _s3b(cl, budget)
        assert whole.kept == [] and whole.dropped_overdue == [cl.contact]
        assert whole.dropped_overdue[0] is cl.contact


@pytest.mark.parametrize("arm, kept, home", [
    ("D1", (0, 1, 2, 3, 4), 57.623),
    ("D2", (0, 1, 2, 3, 4), 57.623),
    ("D3", (0, 1, 2, 3, 4), 57.623),
    ("D5-unit", (0, 1, 2, 4, 6), 54.490),
    ("D5-devices", (0, 1, 2, 4, 6), 54.490),
])
def test_each_d_arm_admits_its_own_members_on_the_cliff(t2, arm, kept, home):
    """Test 2, through ``admit_and_order``: every member is new, so D1-D3's own
    scores tie (never served, unexplored, the same age of update) and their
    order falls to the device id; FedCS's line 3 puts the least dwell first.
    At 99.0 s every arm keeps all but dev-7. ``left_out`` names the rest.
    Without the carrier the route is empty (the Phase 3 cliff)."""
    route = _d_route(t2, arm, 60.0, member_subsets=t2.carrier)
    assert [c.devices for c in route] == [_in_stop_order(t2, kept)]
    assert _home(t2, route, 60.0, RULE_BUDGET) == pytest.approx(home, abs=1e-3)
    rest = left_out([t2.contact], route, member_subsets=t2.carrier)
    assert [c.devices for c in rest] == [_in_stop_order(t2, set(range(8)) - set(kept))]
    route = _d_route(t2, arm, 99.0, member_subsets=t2.carrier)
    assert [c.devices for c in route] == [_in_stop_order(t2, range(7))]
    assert _home(t2, route, 99.0, RULE_BUDGET) == pytest.approx(81.903, abs=1e-3)
    for budget in (60.0, 99.0):
        assert _d_route(t2, arm, budget) == []
        assert _d_route(t2, arm, budget, member_subsets=None) == []


@pytest.mark.parametrize("arm", ["H1"] + list(D_ARMS))
def test_a_contact_that_fits_whole_is_flown_as_the_same_object(t2, unit1, arm):
    """Test 3: the field-wide contact fits whole at 99.5 s (T2's unit; home at
    99.115 s), and so it does for the budget-only D arms at unit 1.0 and
    1000 s. The carrier changes nothing there, not even the object."""
    cases = [(t2, 99.5)] + ([] if arm == "H1" else [(unit1, 1000.0)])
    for cl, budget in cases:
        for kw in ({}, {"member_subsets": cl.carrier}):
            if arm == "H1":
                route = _s3b(cl, budget, **kw).kept
            else:
                route = _d_route(cl, arm, budget, **kw)
            assert route == [cl.contact] and route[0] is cl.contact, (budget, kw)
            assert left_out([cl.contact], route, **kw) == []


def test_every_walk_reduces_through_the_plan_packages_one_member_walk(t2, monkeypatch):
    """unit_U3b.md section 1.5: S3b, the budget walk and FedCS re-issue a
    contact through U3's ``admit_members``, the walk the F family folds on,
    imported when called; they differ only in the order they pass."""
    from hermes.scheduler.plan import member_subset

    real = member_subset.admit_members
    orders = []

    def spy(model, state, wp, order, **kw):
        orders.append(tuple(order))
        return real(model, state, wp, order, **kw)

    monkeypatch.setattr(member_subset, "admit_members", spy)
    _s3b(t2, 60.0, member_subsets=t2.carrier)
    _d_route(t2, "D1", 60.0, member_subsets=t2.carrier)
    _d_route(t2, "D5-unit", 60.0, member_subsets=t2.carrier)
    by_dwell = tuple(DeviceID(f"dev-{i}") for i in (4, 2, 0, 1, 6, 3, 5, 7))
    assert orders == [by_dwell, tuple(sorted(CLIFF_IDS)), by_dwell]


# --------------------------------------------------------------------------- #
# Each arm's own member order, on hand-built contacts
# --------------------------------------------------------------------------- #

STOP = (10.0, 0.0, 0.0)
FAR = (-30.0, 0.0, 0.0)
NOW = 1000.0


def _members(**spec):
    """Device states from ``name=dict(dwell=..., **fields)``: each member sits
    ``dwell`` metres from STOP, so the line model prices it at ``dwell``
    seconds, and gets the other fields as given (in the given order)."""
    states = {}
    for name, given in spec.items():
        given = dict(given)
        dwell = given.pop("dwell")
        st = DeviceSchedulerState(device_id=DeviceID(name), bucket=Bucket.SCHEDULED_THIS_ROUND,
                                  last_known_position=(STOP[0], STOP[1] + dwell, 0.0))
        for key, value in given.items():
            setattr(st, key, value)
        states[DeviceID(name)] = st
    return states


def _line_model(states, *, capacity=None, bounds=DEADLINE_BOUNDS_COLLECTION):
    """1 m/s, and a member costs 1 s of dwell per metre from STOP: the contact
    costs 10 s out and 10 s back, no upload; 100 W flying, 200 W hovering
    (round numbers for the arithmetic, not the Zeng model's)."""
    physics = FerryPhysics(dock=DOCK, member_dwell_s=lambda d, p, o: d, upload_s=lambda: 0.0,
                           p_move_w=100.0, p_hover_w=200.0, energy_capacity_j=capacity,
                           deadline_bounds=bounds, device_states=states)
    return FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=1.0, ferry=physics)


def _contact(states, deadlines, position=STOP):
    devices = tuple(states)
    return ContactWaypoint(position=position, devices=devices,
                           bucket=min((states[d].bucket for d in devices), key=BUCKET_PRIORITY.index),
                           deadline_ts=min(deadlines[d] for d in devices))


def _kept(arm, states, deadlines, budget, **kw):
    """The members ``arm`` keeps of one contact at STOP over ``states``, from
    the dock at NOW with ``budget`` seconds, in the contact's order."""
    contact = _contact(states, deadlines)
    carrier = MemberSubsets(deadlines, states)
    model = _line_model(states)
    if arm == "H":
        route = filter_feasible([contact], now=NOW, mule_pose=DOCK, mission_deadline_ts=NOW + budget,
                                model=model, member_subsets=carrier, **kw).kept
    else:
        env = SelectorEnv(mule_pose=DOCK, now=NOW, mission_round=3)
        route = D_ARMS[arm]().admit_and_order([contact], states, env,
                                              mission_deadline_ts=NOW + budget,
                                              feasibility_model=model, member_subsets=carrier)
    return [str(d) for c in route for d in c.devices]


def _due(**offsets):
    return {DeviceID(n): NOW + s for n, s in offsets.items()}


def _streak(states):
    return lambda wp: max(states[d].miss_streak for d in wp.devices)


# 25.5 s: the contact's 20 s of flight leave 5.5 s of dwell (the line model).

def test_s3b_orders_members_by_own_deadline_then_dwell_then_id():
    """unit_U3b.md section 2.2: EDF inside a contact beats the cheaper dwell;
    among equal deadlines the cheaper dwell goes first; among equal both, the
    id (the contact lists b before a)."""
    edf = _members(a=dict(dwell=5.0), b=dict(dwell=1.0))
    assert _kept("H", edf, _due(a=50.0, b=80.0), 25.5) == ["a"]
    assert _kept("H", edf, _due(a=50.0, b=50.0), 25.5) == ["b"]
    ids = _members(b=dict(dwell=3.0), a=dict(dwell=3.0))
    assert _kept("H", ids, _due(a=50.0, b=50.0), 23.5) == ["a"]


def test_s3b_prices_the_member_order_at_the_snr_offset_it_is_given():
    """The dwell that breaks deadline ties is priced as the predicate prices
    it, at ``snr_offset_db``. b sits 8 m from the stop, out of reach at 0 dB
    (so free: the whole contact fits in 25 s) and reachable at 1 dB for 3.5 s
    against a's 3 s. At 1 dB either fits alone, not both (26.5 s), and a goes
    first; a key priced at 0 dB would put the free b first."""
    states = _members(a=dict(dwell=3.0), b=dict(dwell=8.0))

    def dwell(d, pass_kind, offset):
        return None if d > 5.0 + 10.0 * offset else (3.0 if d < 5.0 else 3.5)

    physics = FerryPhysics(dock=DOCK, member_dwell_s=dwell, upload_s=lambda: 0.0,
                           p_move_w=100.0, p_hover_w=200.0, device_states=states)
    model = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=1.0, ferry=physics)
    due = _due(a=50.0, b=50.0)
    contact = _contact(states, due)
    kept = {}
    for offset in (0.0, 1.0):
        feas = filter_feasible([contact], now=NOW, mule_pose=DOCK, mission_deadline_ts=NOW + 25.0,
                               model=model, snr_offset_db=offset,
                               member_subsets=MemberSubsets(due, states))
        kept[offset] = [str(d) for c in feas.kept for d in c.devices]
    assert kept == {0.0: ["a", "b"], 1.0: ["a"]}


def test_s3b_prices_the_member_order_for_pass_1():
    """S3b is the Pass-1 gate, so the dwell that breaks deadline ties is Pass
    1's, as its predicate prices it. a takes 3 s to collect and 5 s to
    deliver, b 3.5 s and 1 s: a member's dwell may depend on the pass in any
    way (the band physics only scale one rate by each pass's bytes, where the
    two orders agree). Either fits alone, not both (26.5 s), and a goes
    first; a key priced for Pass 2 would put b first."""
    states = _members(a=dict(dwell=1.0), b=dict(dwell=2.0))

    def dwell(d, pass_kind, offset):
        collect = MissionPass(pass_kind) is MissionPass.COLLECT
        if d < 1.5:
            return 3.0 if collect else 5.0
        return 3.5 if collect else 1.0

    physics = FerryPhysics(dock=DOCK, member_dwell_s=dwell, upload_s=lambda: 0.0,
                           p_move_w=100.0, p_hover_w=200.0, device_states=states)
    model = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=1.0, ferry=physics)
    due = _due(a=50.0, b=50.0)
    feas = filter_feasible([_contact(states, due)], now=NOW, mule_pose=DOCK,
                           mission_deadline_ts=NOW + 25.0, model=model,
                           member_subsets=MemberSubsets(due, states))
    assert [str(d) for c in feas.kept for d in c.devices] == ["a"]
    assert [c.devices for c in feas.dropped_budget] == [_devs("b")]


def test_under_miss_priority_s3b_serves_a_members_miss_streak_first():
    """A missed member's wider window gives it a later deadline; the miss
    priority leads inside a contact as it does across contacts."""
    states = _members(a=dict(dwell=1.0, miss_streak=0), b=dict(dwell=5.0, miss_streak=2))
    due = _due(a=50.0, b=80.0)
    assert _kept("H", states, due, 25.5) == ["a"]
    assert _kept("H", states, due, 25.5, priority=_streak(states)) == ["b"]


def test_each_d_arm_ranks_members_by_its_own_score():
    """unit_U3b.md sections 2.3-2.4: D1 the stalest first (never served
    before any), D2 the highest utility (unexplored before any), D3 the
    largest index (the oldest update), D5 the least marginal time."""
    due = _due(a=500.0, b=500.0, c=500.0)
    aoi = _members(a=dict(dwell=3.0, last_clean_ts=NOW - 10.0),
                   b=dict(dwell=3.0, last_clean_ts=NOW - 100.0))
    assert _kept("D1", aoi, due, 23.5) == ["b"]
    aoi[DeviceID("c")] = _members(c=dict(dwell=3.0, last_clean_ts=0.0))[DeviceID("c")]
    assert _kept("D1", aoi, due, 23.5) == ["c"]
    oort = _members(a=dict(dwell=3.0, last_loss=0.5, last_num_examples=10),
                    b=dict(dwell=3.0, last_loss=1.0, last_num_examples=100))
    assert _kept("D2", oort, due, 23.5) == ["b"]
    oort[DeviceID("c")] = _members(c=dict(dwell=3.0, last_loss=None))[DeviceID("c")]
    assert _kept("D2", oort, due, 23.5) == ["c"]
    whittle = _members(a=dict(dwell=3.0, last_merged_round=2), b=dict(dwell=3.0, last_merged_round=0))
    assert _kept("D3", whittle, due, 23.5) == ["b"]
    fedcs = _members(a=dict(dwell=5.0), b=dict(dwell=1.0))
    for arm in ("D5-unit", "D5-devices"):
        assert _kept(arm, fedcs, due, 25.5) == ["b"], arm


@pytest.mark.parametrize("arm, first, cheap", [
    ("H", dict(), dict()),
    ("D1", dict(last_clean_ts=NOW - 100.0), dict(last_clean_ts=NOW - 10.0)),
    ("D2", dict(last_loss=1.0, last_num_examples=100), dict(last_loss=0.5, last_num_examples=10)),
    ("D3", dict(last_merged_round=0), dict(last_merged_round=2)),
])
def test_skip_not_stop_a_later_cheaper_member_is_admitted(arm, first, cheap):
    """The arm's first member does not fit (28 s > 25.5 s), the next, cheaper
    one does: the walk skips the first and goes on. At 28.5 s the first fits
    alone and the pair (29 s) does not, which shows the order."""
    states = _members(a=dict(dwell=8.0, **first), b=dict(dwell=1.0, **cheap))
    assert _kept(arm, states, _due(a=50.0, b=80.0), 25.5) == ["b"]
    assert _kept(arm, states, _due(a=50.0, b=80.0), 28.5) == ["a"]


# --------------------------------------------------------------------------- #
# S3b's complements, left_out, and the carrier's contract
# --------------------------------------------------------------------------- #

def test_s3bs_complements_go_to_the_list_of_the_clause_that_refused_each_member():
    """Test 6, with all four clauses: the line model from the dock at NOW
    (10 s out, 10 s back), route-level delivery bounds (a stop's own deadline
    bounds its ``home``), a budget to NOW + 30 s, an update due at NOW + 40 s
    already on board, 2500 J of battery. In S3b's member order b is refused
    ``overdue`` (home +23 s alone, past its own +21.5 s), a is admitted (home
    +21 s), e ``energy`` (2000 J of flight + 200 W x 6 s > 2500 J), d
    ``budget`` (home +33 s), c ``delivery`` (home +46 s, past the +40 s on
    board). Each complement is one reduced contact in its clause's list,
    never ``dropped_plan``, and S3c's planned count sees every member."""
    states = _members(a=dict(dwell=1.0), b=dict(dwell=3.0), c=dict(dwell=25.0),
                      d=dict(dwell=12.0), e=dict(dwell=5.0))
    due = _due(a=100.0, b=21.5, c=1000.0, d=1000.0, e=1000.0)
    contact = _contact(states, due)
    model = _line_model(states, capacity=2500.0, bounds=DEADLINE_BOUNDS_DELIVERY)
    start = FlightState(DOCK, NOW, 0.0, NOW + 40.0)
    feas = filter_feasible([contact], now=NOW, mule_pose=DOCK, mission_deadline_ts=NOW + 30.0,
                           model=model, state=start, member_subsets=MemberSubsets(due, states))
    assert [c.devices for c in feas.kept] == [_devs("a")]
    by_reason = dict(zip(REASONS, (feas.dropped_overdue, feas.dropped_budget,
                                   feas.dropped_energy, feas.dropped_delivery)))
    assert {why: [c.devices for c in got] for why, got in by_reason.items()} == {
        REASON_OVERDUE: [_devs("b")], REASON_BUDGET: [_devs("d")],
        REASON_ENERGY: [_devs("e")], REASON_DELIVERY: [_devs("c")]}
    for (wp,) in by_reason.values():
        assert wp.position == STOP and wp.deadline_ts == due[wp.devices[0]]
    assert feas.dropped_plan == [] and feas.n_dropped == 4
    assert mission_planned_devices(feas.kept, feas) == 5
    # Whole, the contact fails its own deadline (+21.5): one overdue drop.
    whole = filter_feasible([contact], now=NOW, mule_pose=DOCK, mission_deadline_ts=NOW + 30.0,
                            model=model, state=start)
    assert whole.kept == [] and whole.dropped_overdue == [contact]


def _pair():
    states = _members(a=dict(dwell=1.0), b=dict(dwell=2.0), c=dict(dwell=3.0), d=dict(dwell=4.0))
    due = _due(a=50.0, b=60.0, c=70.0, d=80.0)
    x = _contact({k: states[k] for k in _devs("abc")}, due)
    y = _contact({DeviceID("d"): states[DeviceID("d")]}, due, position=FAR)
    return x, y, MemberSubsets(due, states)


def _devs(text):
    return tuple(DeviceID(ch) for ch in text)


def test_left_out_names_the_rest_of_a_reduced_contact():
    """Test 7 (U5's report of a D arm's drops, decision 6): matched by member,
    in the contacts' order; a contact the route does not touch is the object
    itself, one it serves in part gives its rest as a reduced contact."""
    x, y, carrier = _pair()
    part = carrier.reduce(x, _devs("b"))
    rest = left_out([x, y], [part], member_subsets=carrier)
    assert rest == [carrier.reduce(x, _devs("ac")), y] and rest[1] is y
    assert rest[0].devices == _devs("ac") and rest[0].deadline_ts == NOW + 50.0
    assert left_out([x, y], [x], member_subsets=carrier) == [y]
    got = left_out([x, y], [])
    assert got == [x, y] and got[0] is x and got[1] is y
    assert left_out([x, y], [y, x]) == []


def test_left_out_refuses_what_it_cannot_match():
    x, y, carrier = _pair()
    part = carrier.reduce(x, _devs("b"))
    with pytest.raises(ValueError, match="carrier"):
        left_out([x, y], [part])
    twice = ContactWaypoint(position=FAR, devices=_devs("ad"), bucket=x.bucket, deadline_ts=0.0)
    with pytest.raises(ValueError, match="two contacts"):
        left_out([x, twice], [], member_subsets=carrier)
    with pytest.raises(ValueError, match="no contact holds"):
        left_out([x], [y], member_subsets=carrier)


class _Tripwire(MemberSubsets):
    """A carrier that fails the test if a walk ever ranks members with it."""

    def reduce(self, wp, members):
        raise AssertionError("the carrier was used without a failed contact")


def test_without_a_budget_the_carrier_is_inert():
    """Test 8: no budget, no gate (``s3b_feasibility.py``, ``budget_walk.py``):
    nothing fails, so nothing is reduced and every result is the default's,
    object for object."""
    for seed in range(60):
        c = _case(seed)
        c.budget = None
        trip = _Tripwire(c.deadlines, c.states)
        model = _model(S3B, c)
        env = SelectorEnv(mule_pose=c.pose, now=c.now, mission_round=c.mission_round)
        runs = [
            lambda **kw: filter_feasible(c.stops, now=c.now, mule_pose=c.pose, mission_deadline_ts=None,
                                         model=model, priority=c.prio, **kw).kept,
            lambda **kw: greedy_budget_walk(c.stops, key=_edf, mule_pose=c.pose, now=c.now,
                                            mission_deadline_ts=None, model=model, **kw),
            lambda **kw: fedcs_greedy_select(c.stops, mule_pose=c.pose, now=c.now,
                                             mission_deadline_ts=None, model=model, **kw),
        ] + [
            (lambda make: lambda **kw: make().admit_and_order(
                c.stops, c.states, env, mission_deadline_ts=None, feasibility_model=model,
                **kw))(make)
            for make in D_ARMS.values()
        ]
        for run in runs:
            default, with_trip = run(), run(member_subsets=trip)
            assert len(default) == len(with_trip) == len(c.stops)
            assert all(a is b for a, b in zip(default, with_trip)), seed


def test_the_budget_walk_refuses_the_carrier_for_pass_2():
    """Test 9: Pass 2 delivers to whole contacts, so a carrier there is a
    caller's error, with or without a budget and in either spelling."""
    x, y, carrier = _pair()
    model = _line_model(carrier.device_states)
    for budget in (None, NOW + 30.0):
        for pass_kind in (MissionPass.DELIVER, "deliver"):
            with pytest.raises(ValueError, match="Pass 1 only"):
                greedy_budget_walk([x, y], key=_edf, mule_pose=DOCK, now=NOW,
                                   mission_deadline_ts=budget, model=model, pass_kind=pass_kind,
                                   member_subsets=carrier)
        # Without the carrier Pass 2 walks as recorded; with it, Pass 1 in
        # either spelling is accepted.
        greedy_budget_walk([x, y], key=_edf, mule_pose=DOCK, now=NOW, mission_deadline_ts=budget,
                           model=model, pass_kind=MissionPass.DELIVER)
        greedy_budget_walk([x, y], key=_edf, mule_pose=DOCK, now=NOW, mission_deadline_ts=budget,
                           model=model, pass_kind="collect", member_subsets=carrier)


def test_the_capable_policies_declare_it_and_d4_does_not():
    """Test 10: the scheduler (U5) refuses ``subset`` with a whole-scheduler
    policy that does not declare ``admits_member_subsets``: FedEx (D4) has no
    gate and takes no carrier. The keyword is keyword-only and defaults to
    None everywhere, so a caller that does not pass it is the recorded call."""
    for cls in (MaxAoIPolicy, OortPolicy, WhittlePolicy, FedCSDegradedPolicy):
        assert cls.admits_member_subsets is True
        param = inspect.signature(cls.admit_and_order).parameters["member_subsets"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None
    assert getattr(FedExCarpPolicy, "admits_member_subsets", False) is False
    assert "member_subsets" not in inspect.signature(FedExCarpPolicy.admit_and_order).parameters
    for fn in (filter_feasible, greedy_budget_walk, fedcs_greedy_select):
        param = inspect.signature(fn).parameters["member_subsets"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None


def test_the_carrier_reduces_by_the_plan_packages_rule():
    """``MemberSubsets.reduce`` is U3's ``reduce_stop``: the contact itself for
    all its members, else the members in its order, their worst bucket and
    earliest own deadline."""
    from hermes.scheduler.plan.member_subset import reduce_stop

    x, _, carrier = _pair()
    assert carrier.reduce(x, _devs("cab")) is x
    for members in ("a", "cb", "ca"):
        assert carrier.reduce(x, _devs(members)) == reduce_stop(
            x, _devs(members), deadlines=carrier.deadlines, device_states=carrier.device_states)
    assert carrier.reduce(x, _devs("cb")).devices == _devs("bc")
    with pytest.raises(ValueError):
        carrier.reduce(x, _devs("d"))
    assert "device_states" not in repr(carrier)


# --------------------------------------------------------------------------- #
# Random instances
# --------------------------------------------------------------------------- #

N_SUBSET = 600
N_WHOLE = 1200
N_BAND = 60
BANDS = ("wide", "medium", "narrow")


def _edf(c):
    return (c.deadline_ts, c.position, c.devices)


def _queue_order(c):
    return (c.position,)


class _Dwell:
    """``a + b*d`` seconds per member, 0.05 s less per dB of SNR offset, and
    unreachable (None, so free) beyond ``floor_m`` plus 10 m per dB: a better
    link reaches farther, so the offset can change which members are cheap."""

    def __init__(self, a, b, floor_m):
        self.a, self.b, self.floor_m = a, b, floor_m

    def __call__(self, d, pass_kind, offset):
        if d > self.floor_m + 10.0 * offset:
            return None
        return max(0.0, self.a + self.b * d - 0.05 * offset)


class _Upload:
    def __init__(self, s):
        self.s = s

    def __call__(self):
        return self.s


def _miss_priority(states):
    """``FLScheduler._contact_miss_priority``: a contact's longest miss streak."""
    return lambda wp: max(states[d].miss_streak for d in wp.devices)


def _state_fields(rng, st, now):
    """The fields the D arms rank on, drawn at random."""
    st.last_clean_ts = rng.choice((0.0, now - rng.uniform(1.0, 900.0)))
    st.last_clean_round = rng.randint(0, 5)
    st.last_served_round = rng.randint(0, 6)
    st.last_merged_round = rng.choice((None, rng.randint(0, 5)))
    st.miss_streak = rng.randint(0, 3)
    st.last_loss = rng.choice((None, rng.uniform(0.1, 2.0)))
    st.last_num_examples = rng.choice((0, rng.randint(10, 500)))
    st.reach_attempts = rng.randint(0, 6)
    st.reach_answered = rng.randint(0, st.reach_attempts)


def _case(seed):
    """A random plan: 1 to 12 devices in contacts of 1 to 5 at their centroid,
    each member with its own deadline, bucket and D-arm state; the ferry model
    four times in five (else legacy), sometimes with an energy capacity, a
    range and every deadline bound; a budget; a pose off the dock and a
    mid-flight state sometimes; miss priority and an SNR offset sometimes."""
    rng = random.Random(104_723 * seed + 11)
    now = rng.choice((1000.0, 5000.0))
    n = rng.randint(1, 12)
    ids = [DeviceID(f"s{seed}-d{k}") for k in range(n)]
    pos = {d: (rng.uniform(-150.0, 150.0), rng.uniform(-150.0, 150.0), 0.0) for d in ids}
    states = {}
    for d in ids:
        st = DeviceSchedulerState(device_id=d, bucket=rng.choice(list(Bucket)),
                                  last_known_position=pos[d])
        _state_fields(rng, st, now)
        states[d] = st
    # Half the plans draw deadlines from a coarse grid, so members tie as
    # every new device does at mission 1 (one t_ref, one window) and S3b's
    # dwell tie-break decides.
    if rng.random() < 0.5:
        deadlines = {d: now + rng.uniform(5.0, 400.0) for d in ids}
    else:
        deadlines = {d: now + 50.0 * rng.randint(1, 4) for d in ids}
    order = list(ids)
    rng.shuffle(order)
    stops, k = [], 0
    while k < n:
        m = rng.randint(1, min(5, n - k))
        members = tuple(order[k:k + m])
        k += m
        centre = (sum(pos[d][0] for d in members) / m, sum(pos[d][1] for d in members) / m, 0.0)
        stops.append(ContactWaypoint(
            position=centre, devices=members,
            bucket=min((states[d].bucket for d in members), key=BUCKET_PRIORITY.index),
            deadline_ts=min(deadlines[d] for d in members)))
    physics = None
    if rng.random() < 0.8:
        # 143.6 W / 168.5 W: the rotary-wing powers at 5 m/s
        # (hermes.l1.mission_clock.EnergyModel.at_speed(5.0)).
        physics = dict(
            dock=DOCK,
            member_dwell_s=_Dwell(rng.uniform(0.5, 4.0), rng.uniform(0.0, 0.2),
                                  rng.choice((math.inf, 150.0))),
            upload_s=_Upload(rng.choice((0.0, 0.5, 2.0))), p_move_w=143.6, p_hover_w=168.5,
            energy_capacity_j=rng.choice((None, None, rng.uniform(5e3, 5e4))),
            deadline_bounds=rng.choice(DEADLINE_BOUNDS), range_m=rng.choice((None, 120.0)),
            device_states=states)
    pose = rng.choice((DOCK, (rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0), 0.0)))
    flight = None
    if rng.random() < 0.4:
        flight = (pose, now, rng.uniform(0.0, 3e3),
                  rng.choice((math.inf, now + rng.uniform(20.0, 300.0))))
    return SimpleNamespace(
        seed=seed, now=now, pose=pose, states=states, deadlines=deadlines, stops=stops,
        physics=physics, speed=5.0, sess=rng.uniform(0.5, 3.0), flight=flight,
        budget=now + rng.uniform(10.0, 300.0), prio=rng.choice((None, _miss_priority(states))),
        offset=rng.choice((0.0, 0.0, 3.0)), mission_round=rng.choice((None, 3)),
        carrier=MemberSubsets(deadlines, states),
    )


_BAND_MODELS = {}


def _band_model(band):
    """The planner's model on ``band`` at the pilots' declared 1 MB (T2's
    physics on its seed), cached per band."""
    if band not in _BAND_MODELS:
        spec = FerrySpec.from_config(rf_range_m=60.0, seed=777, contact_band=band,
                                     backhaul_model="seconds", backhaul_period=750.0,
                                     backhaul_regime="clean", payload_bytes=1_000_000)
        _BAND_MODELS[band] = (spec.feasibility_model(rf_range_m=60.0, theta_bytes=52),
                              spec.link.range_planar_m(band))
    return _BAND_MODELS[band]


#: Field half-widths per band, in metres: dense enough for S3a to put several
#: devices in one contact at the band's radius (60 m wide, 119 m medium,
#: 232 m narrow at 1 MB), as the realism field (100 m) does on narrow.
_SPREADS = {"wide": (30.0, 60.0), "medium": (60.0, 100.0), "narrow": (100.0, 150.0)}


def _band_case(band, k):
    """A random layout priced with ``band``'s own physics (the real contact
    link and payload model), clustered by S3a at R_planar(band), and a budget
    between a fifth and all of the time the whole route would take."""
    rng = random.Random(7_919 * k + BANDS.index(band))
    model, radius = _band_model(band)
    now = 1000.0
    n = rng.randint(3, 12)
    ids = [DeviceID(f"{band}{k}-d{i}") for i in range(n)]
    states = {}
    for d, (x, y) in zip(ids, device_positions(n, 500 + k, rng.choice(_SPREADS[band]))):
        st = DeviceSchedulerState(device_id=d, bucket=rng.choice(list(Bucket)),
                                  last_known_position=(x, y, 0.0))
        _state_fields(rng, st, now)
        states[d] = st
    if rng.random() < 0.5:
        deadlines = {d: now + rng.uniform(20.0, 400.0) for d in ids}
    else:
        deadlines = {d: now + 100.0 * rng.randint(1, 3) for d in ids}
    stops = cluster_by_rf_range(eligible_device_ids=ids, device_states=states,
                                deadlines=deadlines, rf_range_m=radius)
    physics = {f.name: getattr(model.ferry, f.name) for f in fields(FerryPhysics)}
    physics.update(device_states=states,
                   energy_capacity_j=rng.choice((None, None, rng.uniform(2e4, 8e4))),
                   deadline_bounds=rng.choice(DEADLINE_BOUNDS))
    bound = FeasibilityModel(cruise_speed_m_s=model.cruise_speed_m_s,
                             session_time_s=model.session_time_s,
                             ferry=FerryPhysics(**physics))
    full = bound.fold(stops, FlightState(DOCK, now), rule=RULE_NONE, budget_end=None,
                      skip=False).home - now
    return SimpleNamespace(
        seed=f"{band}-{k}", now=now, pose=DOCK, states=states, deadlines=deadlines, stops=stops,
        physics=physics, speed=model.cruise_speed_m_s, sess=model.session_time_s, flight=None,
        budget=now + full * rng.uniform(0.2, 1.05), prio=rng.choice((None, _miss_priority(states))),
        offset=0.0, mission_round=rng.choice((None, 3)), carrier=MemberSubsets(deadlines, states),
    )


def _model(mod, c):
    """``c``'s model built from module ``mod``'s classes (live or 6e6f92d)."""
    if c.physics is None:
        return mod.FeasibilityModel(cruise_speed_m_s=c.speed, session_time_s=c.sess)
    return mod.FeasibilityModel(cruise_speed_m_s=c.speed, session_time_s=c.sess,
                                ferry=mod.FerryPhysics(**c.physics))


def _flight(mod, c):
    return None if c.flight is None else mod.FlightState(*c.flight)


# -- an independent re-implementation of each arm's walk (unit_U3b.md 2) ----- #

def _ref_stop(wp, keep, c):
    """A contact reduced by hand, by S3a's rules."""
    keep = set(keep)
    if keep == set(wp.devices):
        return wp
    devices = tuple(d for d in wp.devices if d in keep)
    return ContactWaypoint(
        position=wp.position, devices=devices,
        bucket=min((c.states[d].bucket for d in devices), key=BUCKET_PRIORITY.index),
        deadline_ts=min(c.deadlines[d] for d in devices))


def _ref_members(model, cur, wp, order, rule, c, offset=0.0, pass_kind=MissionPass.COLLECT):
    chosen, verdict, refused = [], None, []
    for d in order:
        v = model.admit(cur, _ref_stop(wp, chosen + [d], c), rule=rule, budget_end=c.budget,
                        pass_kind=pass_kind, snr_offset_db=offset)
        if v.ok:
            chosen.append(d)
            verdict = v
        else:
            refused.append((d, v.reason))
    return chosen, verdict, refused


def _ref_walk(model, start, ordered, member_key, rule, c, offset=0.0,
              pass_kind=MissionPass.COLLECT):
    """The whole-contact walk with member subsets: (route, drops)."""
    route, drops, cur = [], [], start
    for wp in ordered:
        v = model.admit(cur, wp, rule=rule, budget_end=c.budget, pass_kind=pass_kind,
                        snr_offset_db=offset)
        if v.ok:
            route.append(wp)
            cur = v.next_state
            continue
        order = sorted(wp.devices, key=lambda d, w=wp: member_key(w, d))
        chosen, sv, refused = _ref_members(model, cur, wp, order, rule, c, offset, pass_kind)
        if not chosen:
            drops.append((wp, v.reason))
            continue
        route.append(_ref_stop(wp, chosen, c))
        cur = sv.next_state
        for reason in REASONS:
            group = [d for d, why in refused if why == reason]
            if group:
                drops.append((_ref_stop(wp, group, c), reason))
    return route, drops


def _ref_fedcs(model, start, value, c):
    remaining, route, cur = list(c.stops), [], start
    while remaining:
        i = min(range(len(remaining)), key=lambda j: _selection_key(
            remaining[j], model.leg(cur.pose, remaining[j]).total_s, value))
        x = remaining.pop(i)
        v = model.admit(cur, x, rule=RULE_BUDGET, budget_end=c.budget)
        if not v.ok:
            def key(d, x=x, cur=cur):
                one = _ref_stop(x, [d], c)
                return (_selection_key(one, model.leg(cur.pose, one).total_s, value), d)

            chosen, v, _ = _ref_members(model, cur, x, sorted(x.devices, key=key), RULE_BUDGET, c)
            if not chosen:
                continue
            x = _ref_stop(x, chosen, c)
        route.append(x)
        cur = v.next_state
    return route


def _arm_key(arm, c, env):
    """The D1-D3 arm's own contact key, as its policy builds it."""
    if arm == "D1":
        return MaxAoIPolicy._rank_key(c.states, env.mule_pose, env.now)
    if arm == "D2":
        policy = OortPolicy()
        return policy._rank_key(c.states, policy._current_round(c.stops, c.states))
    policy = WhittlePolicy()
    inputs = policy._inputs(c.stops, c.states, env)
    return lambda wp: (-policy._contact_index(wp, inputs), wp.position, wp.devices)


def _source(c, wp):
    (src,) = [s for s in c.stops if wp.devices[0] in s.devices]
    return src


def _check_reduced(c, wp):
    """A reduced contact: a proper subset of one contact, at its position, in
    its order, with its members' own earliest deadline and worst bucket."""
    src = _source(c, wp)
    assert set(wp.devices) < set(src.devices), c.seed
    assert wp.devices == tuple(d for d in src.devices if d in set(wp.devices)), c.seed
    assert wp.position == src.position, c.seed
    assert wp.deadline_ts == min(c.deadlines[d] for d in wp.devices), c.seed
    assert wp.bucket == min((c.states[d].bucket for d in wp.devices),
                            key=BUCKET_PRIORITY.index), c.seed


def _is_input(c, wp):
    return any(wp is s for s in c.stops)


def _check_route(c, model, start, route, rest, rule, offset, tally, arm):
    """The route passes the predicate as flown; its reduced contacts are
    well formed and maximal; the route and ``rest`` partition the devices."""
    kw = dict(rule=rule, budget_end=c.budget, snr_offset_db=offset)
    assert model.fold(route, start, skip=False, **kw).ok, (c.seed, arm)
    for i, wp in enumerate(route):
        if _is_input(c, wp):
            continue
        tally[arm] += 1
        _check_reduced(c, wp)
        before = model.fold(route[:i], start, skip=False, **kw).state
        src = _source(c, wp)
        for d in src.devices:
            if d not in wp.devices:
                bigger = _ref_stop(src, wp.devices + (d,), c)
                assert not model.admit(before, bigger, **kw).ok, (c.seed, arm, d)
    for wp in rest:
        if not _is_input(c, wp):
            _check_reduced(c, wp)
    served = [d for wp in route for d in wp.devices] + [d for wp in rest for d in wp.devices]
    assert sorted(served) == sorted(d for s in c.stops for d in s.devices), (c.seed, arm)


def _same_route(c, got, want, what):
    """Equal contacts in the same order; a contact that was not reduced is
    the very input object."""
    assert got == want, (c.seed, what)
    for g, w in zip(got, want):
        if _is_input(c, w):
            assert g is w, (c.seed, what)


def _ref_h(c, model, start):
    """S3b's walk with member subsets, by hand, from ``start``: (route, drops)."""
    if c.prio is None:
        ordered = sorted(c.stops, key=_edf)
    else:
        ordered = sorted(c.stops, key=lambda s: (-c.prio(s),) + _edf(s))

    def h_key(wp, d):
        one = _ref_stop(wp, [d], c)
        lead = () if c.prio is None else (-c.prio(one),)
        return lead + (c.deadlines[d], model.leg(start.pose, one, snr_offset_db=c.offset).dwell_s, d)

    return _ref_walk(model, start, ordered, h_key, RULE_DEADLINE_BUDGET, c, c.offset)


def _ref_d(arm, c, model, start, env):
    """The D arm's walk with member subsets, by hand, from ``start``: its route."""
    if arm.startswith("D5"):
        return _ref_fedcs(model, start, VALUE_UNIT if arm == "D5-unit" else VALUE_DEVICES, c)
    key = _arm_key(arm, c, env)
    route, _ = _ref_walk(model, start, sorted(c.stops, key=key),
                         lambda wp, d: (key(_ref_stop(wp, [d], c)), d), RULE_BUDGET, c)
    return route


def _check_h(c, model, start, feas, tally):
    """S3b given the carrier, walked from ``start``: the walk by hand, clause
    by clause, and :func:`_check_route`."""
    route, drops = _ref_h(c, model, start)
    _same_route(c, feas.kept, route, "H")
    lists = (feas.dropped_overdue, feas.dropped_budget, feas.dropped_energy, feas.dropped_delivery)
    for reason, got in zip(REASONS, lists):
        _same_route(c, got, [w for w, why in drops if why == reason], ("H", reason))
    assert feas.dropped_plan == [], c.seed
    _check_route(c, model, start, feas.kept, feas.dropped, RULE_DEADLINE_BUDGET, c.offset,
                 tally, "H")


def _check_d(arm, c, model, start, env, got, tally):
    """A D arm's route given the carrier, walked from ``start``: the walk by
    hand, and :func:`_check_route` with ``left_out`` as the rest."""
    _same_route(c, got, _ref_d(arm, c, model, start, env), arm)
    rest = left_out(c.stops, got, member_subsets=c.carrier)
    _check_route(c, model, start, got, rest, RULE_BUDGET, 0.0, tally, arm)


def _check_subsets(c, tally):
    """Test 4 on one instance, for S3b and all five D policies."""
    model = _model(S3B, c)
    feas = filter_feasible(c.stops, now=c.now, mule_pose=c.pose, mission_deadline_ts=c.budget,
                           model=model, priority=c.prio, state=_flight(S3B, c),
                           snr_offset_db=c.offset, member_subsets=c.carrier)
    _check_h(c, model, _flight(S3B, c) or FlightState(c.pose, c.now), feas, tally)
    env = SelectorEnv(mule_pose=c.pose, now=c.now, mission_round=c.mission_round)
    for arm, make in D_ARMS.items():
        got = make().admit_and_order(c.stops, c.states, env, mission_deadline_ts=c.budget,
                                     feasibility_model=model, member_subsets=c.carrier)
        _check_d(arm, c, model, FlightState(c.pose, c.now), env, got, tally)


def test_every_reduced_contact_passes_the_predicate_and_equals_the_reference_walk():
    """Test 4 on synthetic instances: see :func:`_check_route` and
    :func:`_check_subsets`. Each arm must reduce contacts often enough for
    the checks to mean something."""
    tally = Counter()
    for seed in range(N_SUBSET):
        _check_subsets(_case(seed), tally)
    # Reduced contacts seen (seeded, so fixed): H 282, D1-D3 68-80, D5 66-91.
    for arm in ["H"] + list(D_ARMS):
        assert tally[arm] >= 50, tally


@pytest.mark.parametrize("band", BANDS)
def test_on_each_bands_physics_every_reduced_contact_passes_the_predicate(band):
    """Test 4 on the real contact link and payload model of each band, with
    S3a's own contacts (on narrow a few of them cover the whole field)."""
    tally = Counter()
    for k in range(N_BAND):
        _check_subsets(_band_case(band, k), tally)
    # Reduced contacts seen (seeded): 19-28 per arm on wide, 25-37 on medium,
    # 47-57 on narrow.
    for arm in ["H"] + list(D_ARMS):
        assert tally[arm] >= 15, (band, tally)


def _moved(c):
    """A mid-flight state apart from ``(c.pose, c.now)``: the mule elsewhere,
    later, with energy spent and, half the time, an update on board."""
    rng = random.Random(15_485_863 * c.seed + 5)
    pose = (c.pose[0] + rng.uniform(-80.0, 80.0), c.pose[1] + rng.uniform(-80.0, 80.0), 0.0)
    return FlightState(pose, c.now + rng.uniform(1.0, 60.0), rng.uniform(0.0, 3e3),
                       rng.choice((math.inf, c.now + rng.uniform(60.0, 400.0))))


def test_given_a_mid_flight_state_each_walk_walks_the_members_from_it():
    """The walks' own API, which no policy exercises (``admit_and_order`` takes
    no state): given the carrier and a mid-flight ``state``, S3b's gate, the
    budget walk under each D1-D3 key and FedCS in both values walk the
    contacts and their members from that state (its pose, clock, energy spent
    and ``deliver_by``), not from ``(mule_pose, now)``. Each equals the walk
    by hand from the state and passes the predicate as flown from it. The
    state is drawn apart from ``(mule_pose, now)``, and the cases where the
    walk by hand from ``(mule_pose, now)`` differs are counted, so a walk
    that ignored the state could not pass."""
    tally, apart = Counter(), Counter()
    for seed in range(N_SUBSET):
        c = _case(seed)
        model, flight, idle = _model(S3B, c), _moved(c), FlightState(c.pose, c.now)
        env = SelectorEnv(mule_pose=c.pose, now=c.now, mission_round=c.mission_round)
        given = dict(mule_pose=c.pose, now=c.now, mission_deadline_ts=c.budget, model=model,
                     state=flight, member_subsets=c.carrier)
        feas = filter_feasible(c.stops, priority=c.prio, snr_offset_db=c.offset, **given)
        _check_h(c, model, flight, feas, tally)
        apart["H"] += _ref_h(c, model, idle)[0] != feas.kept
        for arm in D_ARMS:
            if arm.startswith("D5"):
                value = VALUE_UNIT if arm == "D5-unit" else VALUE_DEVICES
                got = fedcs_greedy_select(c.stops, value=value, **given)
            else:
                got = greedy_budget_walk(c.stops, key=_arm_key(arm, c, env), **given)
            _check_d(arm, c, model, flight, env, got, tally)
            apart[arm] += _ref_d(arm, c, model, idle, env) != got
    # Seen (seeded, so fixed): reduced contacts H 372, D1-D3 56-65, D5 57-58;
    # routes apart from the walk from (mule_pose, now) H 383, D1-D3 221-228,
    # D5 265-290.
    for arm in ["H"] + list(D_ARMS):
        assert tally[arm] >= 25 and apart[arm] >= 100, (tally, apart)


def test_where_nothing_is_reduced_fold_subsets_is_fold_skip_true_object_for_object():
    """Test 12, on synthetic instances under every rule and both passes. In
    every case the fold equals the independent walk (the pass and the SNR
    offset price the member walk too); wherever no contact admitted a proper
    subset of its members, which happens both ways often, the route,
    rejections, verdicts, state and home all equal ``fold``'s."""
    seen = Counter()
    for seed in range(N_SUBSET):
        c = _case(seed)
        model = _model(S3B, c)
        start = _flight(S3B, c) or FlightState(c.pose, c.now)
        rule = (RULE_DEADLINE_BUDGET, RULE_BUDGET, RULE_NONE)[seed % 3]
        pass_kind = (MissionPass.COLLECT, MissionPass.DELIVER)[seed % 2]
        kw = dict(rule=rule, budget_end=c.budget, pass_kind=pass_kind, snr_offset_db=c.offset)
        got = fold_subsets(c.stops, start, model=model, member_order=lambda wp: sorted(wp.devices),
                           subsets=c.carrier, **kw)
        route, drops = _ref_walk(model, start, c.stops, lambda wp, d: d, rule, c, c.offset,
                                 pass_kind)
        _same_route(c, list(got.route), route, "fold")
        assert [why for _, why in got.rejected] == [why for _, why in drops], seed
        _same_route(c, [w for w, _ in got.rejected], [w for w, _ in drops], "fold")
        want = model.fold(c.stops, start, skip=True, **kw)
        reduced = not all(_is_input(c, w) for w in got.route) or not all(
            _is_input(c, w) for w, _ in got.rejected)
        seen["reduced" if reduced else "not"] += 1
        if reduced:
            continue
        assert len(got.route) == len(want.route), seed
        assert all(a is b for a, b in zip(got.route, want.route)), seed
        assert [why for _, why in got.rejected] == [why for _, why in want.rejected], seed
        assert all(a is b for (a, _), (b, _) in zip(got.rejected, want.rejected)), seed
        assert got.verdicts == want.verdicts and got.state == want.state, seed
        assert got.home == want.home, seed
    assert seen["reduced"] >= 100 and seen["not"] >= 100, seen


def test_fold_subsets_reports_the_verdicts_state_and_home_of_what_it_flew():
    """``fold_subsets``' accounts as its docstring states them, reduced
    contacts included (test 12 compares them with ``fold`` where nothing is
    reduced): one verdict per input contact, in input order, that of what was
    flown for it, the contact or its reduction, from the state the route had
    reached, or else that of its whole refusal from that state; ``state`` and
    ``home`` those of the route flown as it stands (``fold(skip=False)``), so
    a reduced contact flown last gives the home, its return leg and upload
    included. Under every rule and both passes, each contact's members in a
    random order, the route and rejections checked against the walk by hand
    as well."""
    seen = Counter()
    for seed in range(N_SUBSET):
        c = _case(seed)
        model = _model(S3B, c)
        start = _flight(S3B, c) or FlightState(c.pose, c.now)
        rule = (RULE_DEADLINE_BUDGET, RULE_BUDGET, RULE_NONE)[seed % 3]
        pass_kind = (MissionPass.COLLECT, MissionPass.DELIVER)[seed % 2]
        kw = dict(rule=rule, budget_end=c.budget, pass_kind=pass_kind, snr_offset_db=c.offset)
        rng = random.Random(seed)
        rank = {d: rng.random() for s in c.stops for d in s.devices}
        got = fold_subsets(c.stops, start, model=model, subsets=c.carrier,
                           member_order=lambda wp: sorted(wp.devices, key=rank.__getitem__), **kw)
        route, drops = _ref_walk(model, start, c.stops, lambda wp, d: rank[d], rule, c, c.offset,
                                 pass_kind)
        _same_route(c, list(got.route), route, "fold")
        assert [why for _, why in got.rejected] == [why for _, why in drops], seed
        _same_route(c, [w for w, _ in got.rejected], [w for w, _ in drops], "fold")
        flown = model.fold(got.route, start, skip=False, **kw)
        assert flown.ok, seed
        assert got.state == flown.state and got.home == flown.home, seed
        assert len(got.verdicts) == len(c.stops), seed
        cur, k = start, 0
        for wp, v in zip(c.stops, got.verdicts):
            if k < len(got.route) and set(got.route[k].devices) <= set(wp.devices):
                assert v.ok and v == flown.verdicts[k], (seed, k)
                seen["reduced and flown"] += not _is_input(c, got.route[k])
                cur, k = v.next_state, k + 1
            else:
                assert not v.ok and v == model.admit(cur, wp, **kw), seed
                seen["refused whole"] += 1
        assert k == len(got.route), seed
        seen["home from a reduced contact"] += bool(got.route) and not _is_input(c, got.route[-1])
    # Seen (seeded): 141 reduced contacts flown, 95 folds whose home is a
    # reduced contact's, 363 whole refusals.
    assert seen["reduced and flown"] >= 50 and seen["home from a reduced contact"] >= 40, seen
    assert seen["refused whole"] >= 100, seen


def test_fold_subsets_refuses_an_unknown_rule():
    x, y, carrier = _pair()
    with pytest.raises(ValueError, match="rule"):
        fold_subsets([x, y], FlightState(DOCK, NOW), model=_line_model(carrier.device_states),
                     rule="bogus", budget_end=NOW + 30.0, member_order=lambda wp: wp.devices,
                     subsets=carrier)


# --------------------------------------------------------------------------- #
# Freeze Rule 1: without the carrier, the recorded code of 6e6f92d
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    modules = REF.load(tmp_path_factory.mktemp("ref"))
    yield modules
    REF.unload(modules)


def test_the_reference_is_the_recorded_code_and_leaks_nothing(ref):
    """The loader's own contract: the reference walks price with the
    reference predicate and walk (Whittle with the reference Oort), the live
    names still map to the live modules, and the reference really is the
    code before Phase 4 (no carrier, no ``dropped_plan``)."""
    assert ref.budget_walk.FeasibilityModel is ref.s3b.FeasibilityModel
    assert ref.fedcs.FeasibilityModel is ref.s3b.FeasibilityModel
    for policy in (ref.max_aoi, ref.oort, ref.whittle):
        assert policy.greedy_budget_walk is ref.budget_walk.greedy_budget_walk
    assert ref.whittle.statistical_utility is ref.oort.statistical_utility
    assert ref.s3b.FeasibilityModel is not S3B.FeasibilityModel
    for short, dotted, _ in REF.MODULES:
        assert sys.modules[dotted] is not getattr(ref, short)
        assert getattr(ref, short).__name__ == REF.private_name(dotted)
    assert sys.modules["hermes.scheduler.stages.s3b_feasibility"] is S3B
    assert not hasattr(ref.s3b, "MemberSubsets") and hasattr(S3B, "MemberSubsets")
    assert "dropped_plan" not in {f.name for f in fields(ref.s3b.FeasibilityResult)}
    for fn in (ref.s3b.filter_feasible, ref.budget_walk.greedy_budget_walk,
               ref.fedcs.fedcs_greedy_select, ref.max_aoi.MaxAoIPolicy.admit_and_order):
        assert "member_subsets" not in inspect.signature(fn).parameters


def _index(c, route):
    """Each contact's position in the input, by identity (a whole walk only
    ever returns input objects)."""
    return [next(i for i, s in enumerate(c.stops) if s is w) for w in route]


def _check_whole(ref, c):
    """Test 11 on one instance: every entry point, without the carrier and
    with an explicit None, against 6e6f92d's, compared by index and object."""
    live_m, ref_m = _model(S3B, c), _model(ref.s3b, c)
    live_st, ref_st = _flight(S3B, c), _flight(ref.s3b, c)
    common = dict(now=c.now, mule_pose=c.pose, mission_deadline_ts=c.budget)
    env = SelectorEnv(mule_pose=c.pose, now=c.now, mission_round=c.mission_round)
    for kw in ({}, {"member_subsets": None}):
        got = filter_feasible(c.stops, model=live_m, priority=c.prio, state=live_st,
                              snr_offset_db=c.offset, **common, **kw)
        want = ref.s3b.filter_feasible(c.stops, model=ref_m, priority=c.prio, state=ref_st,
                                       snr_offset_db=c.offset, **common)
        for name in ("kept", "dropped_overdue", "dropped_budget", "dropped_energy",
                     "dropped_delivery"):
            assert _index(c, getattr(got, name)) == _index(c, getattr(want, name)), (c.seed, name)
        assert got.dropped_plan == [], c.seed
        for pass_kind in (MissionPass.COLLECT, MissionPass.DELIVER):
            for key in (_edf, _queue_order):
                got_w = greedy_budget_walk(c.stops, key=key, model=live_m, state=live_st,
                                           pass_kind=pass_kind, **common, **kw)
                want_w = ref.budget_walk.greedy_budget_walk(c.stops, key=key, model=ref_m,
                                                            state=ref_st, pass_kind=pass_kind,
                                                            **common)
                assert _index(c, got_w) == _index(c, want_w), (c.seed, pass_kind, key)
        for value in (VALUE_UNIT, VALUE_DEVICES):
            got_f = fedcs_greedy_select(c.stops, value=value, model=live_m, state=live_st,
                                        **common, **kw)
            want_f = ref.fedcs.fedcs_greedy_select(c.stops, value=value, model=ref_m,
                                                   state=ref_st, **common)
            assert _index(c, got_f) == _index(c, want_f), (c.seed, value)
        pairs = (
            (MaxAoIPolicy(), ref.max_aoi.MaxAoIPolicy()),
            (OortPolicy(), ref.oort.OortPolicy()),
            (WhittlePolicy(), ref.whittle.WhittlePolicy()),
            (FedCSDegradedPolicy(VALUE_UNIT), ref.fedcs.FedCSDegradedPolicy(VALUE_UNIT)),
            (FedCSDegradedPolicy(VALUE_DEVICES), ref.fedcs.FedCSDegradedPolicy(VALUE_DEVICES)),
        )
        for live_p, ref_p in pairs:
            got_r = live_p.admit_and_order(c.stops, c.states, env, mission_deadline_ts=c.budget,
                                           feasibility_model=live_m, **kw)
            want_r = ref_p.admit_and_order(c.stops, c.states, env, mission_deadline_ts=c.budget,
                                           feasibility_model=ref_m)
            assert _index(c, got_r) == _index(c, want_r), (c.seed, live_p.name)
            assert vars(live_p) == vars(ref_p), (c.seed, live_p.name)
            # Without a reduction left_out is the identity rule
            # (FLScheduler.replan_remainder's), object for object.
            rest = left_out(c.stops, got_r)
            assert _index(c, rest) == [i for i, s in enumerate(c.stops)
                                       if not any(s is r for r in got_r)], c.seed


def test_without_the_carrier_every_walk_equals_6e6f92d_on_random_instances(ref):
    """Test 11 (unit_U3b.md section 6): S3b's five lists, the budget walk in
    both passes, FedCS in both values and the five D policies, each from a
    mid-flight state where it takes one, with and without a budget. D5 is
    not in the goldens, so this is its pin (UG4's hand-off)."""
    for seed in range(N_WHOLE):
        c = _case(seed)
        if seed % 5 == 0:
            c.budget = None
        _check_whole(ref, c)


@pytest.mark.parametrize("band", BANDS)
def test_without_the_carrier_the_walks_equal_6e6f92d_on_each_bands_physics(ref, band):
    """Test 11 on the real physics of each band. The goldens fly the D arms on
    wide only (UG4's hand-off), so narrow and medium are pinned here."""
    for k in range(N_BAND):
        _check_whole(ref, _band_case(band, k))


# --------------------------------------------------------------------------- #
# Layering (unit_U3b.md section 1.5)
# --------------------------------------------------------------------------- #

SIX = (
    "hermes/scheduler/stages/s3b_feasibility.py",
    "hermes/scheduler/policies/budget_walk.py",
    "hermes/scheduler/policies/fedcs_degraded.py",
    "hermes/scheduler/policies/max_aoi.py",
    "hermes/scheduler/policies/oort.py",
    "hermes/scheduler/policies/whittle.py",
)


def _imports(source):
    """``(module, function)`` for every import: ``function`` is None at module
    level (in a class body, an ``if`` or a ``try`` included), else the name of
    the outermost function holding it."""
    out = []

    def names(node):
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        return ["." * node.level + (node.module or "")]

    def visit(node, where):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                out.extend((name, where) for name in names(child))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and where is None:
                visit(child, child.name)
            else:
                visit(child, where)

    visit(ast.parse(source), None)
    return out


def test_the_six_files_import_the_plan_package_only_inside_their_subset_code(ref):
    """No module-level edge from a stage or policy up to the plan package, so
    no import cycle can form; the plan package is imported only where a
    contact is reduced. MAX-AoI, Oort, Whittle and S3b import at module level
    exactly what they imported at 6e6f92d."""
    plan_users = {}
    for rel in SIX:
        for name, where in _imports((REPO / rel).read_text(encoding="utf-8")):
            if name.startswith("hermes.scheduler.plan"):
                assert where is not None, (rel, name)
                plan_users.setdefault(rel, set()).add((name, where))
    assert plan_users == {
        "hermes/scheduler/stages/s3b_feasibility.py": {
            ("hermes.scheduler.plan.member_subset", "reduce"),
            ("hermes.scheduler.plan.member_subset", "fold_subsets"),
        },
        "hermes/scheduler/policies/fedcs_degraded.py": {
            ("hermes.scheduler.plan.member_subset", "_admit_members"),
        },
    }
    for short, rel in (("s3b", SIX[0]), ("max_aoi", SIX[3]), ("oort", SIX[4]),
                       ("whittle", SIX[5])):
        live = [n for n, w in _imports((REPO / rel).read_text(encoding="utf-8")) if w is None]
        recorded = [n for n, w in _imports(Path(getattr(ref, short).__file__).read_text(
            encoding="utf-8")) if w is None]
        assert live == recorded, rel


def _python(code):
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]


def test_the_plan_package_and_the_policies_import_in_either_order():
    """Test 13: a fresh interpreter imports the member walk before the
    policies, a second one after them."""
    _python(
        "import hermes.scheduler.plan.member_subset as ms\n"
        "import hermes.scheduler.policies as policies\n"
        "from hermes.scheduler.policies.budget_walk import left_out\n"
        "from hermes.scheduler.stages.s3b_feasibility import MemberSubsets\n"
        "assert ms.admit_members and left_out and MemberSubsets\n"
        "assert policies.FedCSDegradedPolicy.admits_member_subsets\n"
    )
    _python(
        "import hermes.scheduler.policies as policies\n"
        "from hermes.scheduler.stages.s3b_feasibility import MemberSubsets, fold_subsets\n"
        "import hermes.scheduler.plan.member_subset as ms\n"
        "from hermes.scheduler import FLScheduler\n"
        "assert ms.reduce_stop and fold_subsets and FLScheduler\n"
        "assert policies.MaxAoIPolicy.admits_member_subsets\n"
    )


def test_the_recorded_path_never_loads_the_plan_package():
    """Every walk without the carrier, a contact failing whole included, loads
    no plan module; the first call with the carrier loads the member walk."""
    _python(
        "import sys\n"
        "from hermes.scheduler.policies import (FedCSDegradedPolicy, MaxAoIPolicy,\n"
        "    OortPolicy, WhittlePolicy)\n"
        "from hermes.scheduler.policies.budget_walk import greedy_budget_walk, left_out\n"
        "from hermes.scheduler.policies.fedcs_degraded import fedcs_greedy_select\n"
        "from hermes.scheduler.selector.features import SelectorEnv\n"
        "from hermes.scheduler.stages.s3b_feasibility import (FeasibilityModel,\n"
        "    MemberSubsets, filter_feasible)\n"
        "from hermes.types import Bucket, ContactWaypoint, DeviceSchedulerState\n"
        "ids = ('a', 'b')\n"
        "states = {d: DeviceSchedulerState(device_id=d, bucket=Bucket.NEW,\n"
        "          last_known_position=(100.0, 0.0, 0.0)) for d in ids}\n"
        "due = {'a': 50.0, 'b': 500.0}\n"
        "wp = ContactWaypoint(position=(100.0, 0.0, 0.0), devices=ids,\n"
        "                     bucket=Bucket.NEW, deadline_ts=50.0)\n"
        "m = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=5.0)\n"
        "env = SelectorEnv(now=0.0, mission_round=1)\n"
        "assert filter_feasible([wp], now=0.0, mission_deadline_ts=50.0, model=m).kept == []\n"
        "assert greedy_budget_walk([wp], key=lambda c: 0, mule_pose=(0.0, 0.0, 0.0), now=0.0,\n"
        "                          mission_deadline_ts=50.0, model=m) == []\n"
        "assert fedcs_greedy_select([wp], mule_pose=(0.0, 0.0, 0.0), now=0.0,\n"
        "                           mission_deadline_ts=50.0, model=m) == []\n"
        "for p in (MaxAoIPolicy(), OortPolicy(), WhittlePolicy(), FedCSDegradedPolicy()):\n"
        "    assert p.admit_and_order([wp], states, env, mission_deadline_ts=50.0,\n"
        "                             feasibility_model=m) == []\n"
        "assert left_out([wp], []) == [wp]\n"
        "plan = sorted(n for n in sys.modules if n.startswith('hermes.scheduler.plan'))\n"
        "assert not plan, plan\n"
        "got = filter_feasible([wp], now=0.0, mission_deadline_ts=1000.0, model=m,\n"
        "                      member_subsets=MemberSubsets(due, states))\n"
        "assert [c.devices for c in got.kept] == [('b',)], got\n"
        "assert 'hermes.scheduler.plan.member_subset' in sys.modules\n"
    )
