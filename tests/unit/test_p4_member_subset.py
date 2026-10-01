"""FeRRy Phase 4 (unit U3): member-subset admission, its fold and its trim.

``hermes.scheduler.plan.member_subset`` re-issues a stop that fails whole with
the members that still fit (design D-D (a); the user's decision 4). Pinned here:

* a reduced stop is an ordinary ``ContactWaypoint``: the stop's position, the
  subset in the stop's order, the worst of its buckets and the earliest of its
  own deadlines; the full set is the stop itself; under the age cap a mixed
  stop carries its uncapped members' earliest deadline (critic B2), by the
  age-cap stage's own rules, imported, not restated (one definition);
* the F family's member order: capped first, then weight per second of dwell,
  then device id (Phase 4 spec, other choices 4), each member priced for the
  pass and the SNR offset given;
* the one member walk, ``admit_members``: U3b's contract (``unit_U3b.md``
  section 1.5) with an injectable order, skip-not-stop, maximal for its order,
  priced for the pass given;
* complements, one per reason in S3b's ``REASONS`` order, widened by the
  mule's own widening;
* a stop none of whose members fits: without a capped member it is dropped
  whole as ``fold`` drops it (U3b's rule), a priority stop member by member
  with each member's own reason, so no drop labels a capped member
  ``overdue`` (spec, other choices 8);
* ``fold_members``: stops that fit whole keep their identity, ``require_all``
  prunes, an injected order is used, the route passes the no-skip fold with
  its exempt stops recomputed (critic B1), and on random instances it equals
  an independent re-implementation, in the F order and in injected orders, and
  ``fold(skip=True)`` wherever nothing is reduced;
* ``trim_members``: a route that passes is kept, else priority stops fly first
  and shed their uncapped members first, every capped member the
  protected-only trim keeps is kept, the injected order and the rule are used,
  and the trim invariants hold on random instances (critic A11 (v): priority
  stops may move to the front);
* the Phase 3 cliff instance (``device_positions(8, 777, 100.0)``, narrow,
  1 MB) admits 5 members at 60 s and 7 at 99.0 s, where S3b still admits none;
* the recorded pipeline is untouched: ``FeasibilityResult.dropped_plan`` is
  additive, last and empty; the switch values live in S3b and the plan package
  restates them; S3b and the scheduler never load the plan package or the
  age-cap stage.
"""

from __future__ import annotations

import ast
import math
import os
import random
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.exp4.topology_builder import device_positions
from hermes.l1.mission_clock import MissionClock
from hermes.mule.ferry import FerrySpec
from hermes.mule.mule_main import MuleSupervisor, mission_planned_devices
from hermes.scheduler import FLScheduler
from hermes.scheduler.plan import MEMBER_ADMISSIONS as PLAN_MEMBER_ADMISSIONS
from hermes.scheduler.plan import MemberFold, member_subset
from hermes.scheduler.plan.member_subset import (
    MemberWalk,
    StopAdmission,
    admit_members,
    admit_stop,
    complements,
    fold_members,
    member_order,
    pass_energy_j,
    reduce_stop,
    trim_members,
)
from hermes.scheduler.routing.replan import ORDER_ARM_TRIMMED, ORDER_CURRENT, ReplanResult
from hermes.scheduler.stages import s3d_age_cap
from hermes.scheduler.stages.s3d_age_cap import stop_deadline
from hermes.scheduler.stages.s3b_feasibility import (
    DEADLINE_BOUNDS,
    DEADLINE_BOUNDS_COLLECTION,
    DEADLINE_BOUNDS_DELIVERY,
    MEMBER_ADMISSION_SUBSET,
    MEMBER_ADMISSION_WHOLE,
    MEMBER_ADMISSIONS,
    REASON_BUDGET,
    REASON_OVERDUE,
    REASONS,
    RULE_BUDGET,
    RULE_DEADLINE_BUDGET,
    FeasibilityModel,
    FeasibilityResult,
    FerryPhysics,
    FlightState,
    filter_feasible,
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

REPO = Path(__file__).resolve().parents[2]
DOCK = (0.0, 0.0, 0.0)
T0 = FlightState(DOCK, 0.0)
A, B, C, D, E = (DeviceID(x) for x in ("a", "b", "c", "d", "e"))
NEW, SCHED, BEACON = Bucket.NEW, Bucket.SCHEDULED_THIS_ROUND, Bucket.BEACON_ACTIVE
STOP = (10.0, 0.0, 0.0)
FAR = (20.0, 0.0, 0.0)
WEST = (-10.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Hand-built instances: a line model where distance is dwell
# --------------------------------------------------------------------------- #

def _states(at=STOP, **spec):
    """Device states from ``name=(dwell, bucket)``: each member sits ``dwell``
    metres from ``at``, so the line model prices it at ``dwell`` seconds."""
    return {
        DeviceID(n): DeviceSchedulerState(device_id=DeviceID(n), bucket=bucket,
                                          last_known_position=(at[0], at[1] + dist, 0.0))
        for n, (dist, bucket) in spec.items()
    }


def _line_model(states, *, upload=0.0, capacity=None, bounds=DEADLINE_BOUNDS_COLLECTION,
                range_m=None):
    """1 m/s, and a member costs 1 s of dwell per metre from its stop: a stop
    at x = 10 costs 10 s out and 10 s back. 100 W flying, 200 W hovering."""
    physics = FerryPhysics(dock=DOCK, member_dwell_s=lambda d, p, o: d, upload_s=lambda: upload,
                           p_move_w=100.0, p_hover_w=200.0, energy_capacity_j=capacity,
                           deadline_bounds=bounds, range_m=range_m, device_states=states)
    return FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=1.0, ferry=physics)


def _pass_model(states):
    """The line model's flight, but a member's dwell follows the pass and the
    SNR offset: collecting, 1 s per metre from its stop less 0.5 s per dB of
    offset (a better link uploads faster); delivering, 1 s wherever it sits
    (every member downloads the same model)."""
    def dwell(d, pass_kind, offset):
        return 1.0 if pass_kind is MissionPass.DELIVER else max(0.0, d - 0.5 * offset)

    physics = FerryPhysics(dock=DOCK, member_dwell_s=dwell, upload_s=lambda: 0.0,
                           p_move_w=100.0, p_hover_w=200.0, device_states=states)
    return FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=1.0, ferry=physics)


def _labels(dropped):
    return [(wp.devices, why) for wp, why in dropped]


def _wp(devices, *, deadline, bucket=SCHED, pos=STOP):
    return ContactWaypoint(position=pos, devices=tuple(DeviceID(d) for d in devices),
                           bucket=bucket, deadline_ts=deadline)


def _devs(text):
    return tuple(DeviceID(x) for x in text)


# --------------------------------------------------------------------------- #
# Reduced stops (unit_U3b.md section 1.5; S3a's rules)
# --------------------------------------------------------------------------- #

TRIO_STATES = _states(a=(5.0, NEW), b=(1.0, SCHED), c=(2.0, BEACON))
TRIO_DEADLINES = {A: 50.0, B: 80.0, C: 120.0}
TRIO = _wp("abc", deadline=50.0, bucket=NEW)


def _reduce(members, **kw):
    return reduce_stop(TRIO, _devs(members), deadlines=TRIO_DEADLINES,
                       device_states=TRIO_STATES, **kw)


def test_a_reduced_stop_is_an_ordinary_waypoint_of_its_members():
    assert _reduce("cb") == ContactWaypoint(position=STOP, devices=(B, C), bucket=SCHED,
                                            deadline_ts=80.0)
    assert _reduce("c") == ContactWaypoint(position=STOP, devices=(C,), bucket=BEACON,
                                           deadline_ts=120.0)
    both = _reduce("ca")
    assert both.devices == (A, C) and both.bucket is NEW and both.deadline_ts == 50.0


def test_the_full_set_is_the_stop_itself():
    for members in ("abc", "cab", "bca"):
        assert _reduce(members) is TRIO
    # Whatever the maps say: a stop that is not reduced is not rebuilt.
    assert reduce_stop(TRIO, TRIO.devices, deadlines={}, device_states={}) is TRIO


def test_a_reduction_leaves_the_annotations_the_full_set_keeps():
    annotated = replace(TRIO, band="narrow", range_m=232.2, pred_snr_db=(1.0, 2.0, 3.0))
    part = reduce_stop(annotated, [A, B], deadlines=TRIO_DEADLINES, device_states=TRIO_STATES)
    assert (part.band, part.range_m, part.pred_snr_db) == (None, None, None)
    assert reduce_stop(annotated, [C, B, A], deadlines=TRIO_DEADLINES,
                       device_states=TRIO_STATES) is annotated


@pytest.mark.parametrize("members", [(), (A, A), (A, DeviceID("z"))])
def test_reduce_stop_refuses_anything_but_a_non_empty_subset(members):
    with pytest.raises(ValueError):
        reduce_stop(TRIO, members, deadlines=TRIO_DEADLINES, device_states=TRIO_STATES)


def test_a_member_without_a_deadline_or_a_bucket():
    got = reduce_stop(TRIO, [C], deadlines={A: 50.0}, device_states={})
    assert got.deadline_ts == math.inf      # S3a's convention (s3a_cluster.py:169)
    assert got.bucket is NEW                # no member bucket: the stop's own stands


def test_under_the_cap_a_mixed_stop_takes_its_uncapped_members_deadline():
    """Critic B2: an exempt stop has capped members only; a mixed stop keeps
    its clause, held to the earliest deadline of its uncapped members."""
    caps = frozenset({A})
    assert _reduce("ab", capped=caps).deadline_ts == 80.0     # b's, not a's 50
    assert _reduce("a", capped=caps).deadline_ts == 50.0      # all capped: finite
    full = _reduce("abc", capped=caps)
    assert full == replace(TRIO, deadline_ts=80.0) and full is not TRIO
    assert _reduce("abc", capped={C}) is TRIO                 # the earliest is uncapped
    assert _reduce("abc", capped={A, B, C}) is TRIO           # all capped: the minimum
    assert stop_deadline([A, B, C], deadlines=TRIO_DEADLINES, capped={A, B}) == 120.0
    assert stop_deadline([A, B, C], deadlines=TRIO_DEADLINES) == 50.0
    with pytest.raises(ValueError):
        stop_deadline([], deadlines=TRIO_DEADLINES)


# --------------------------------------------------------------------------- #
# The F family's member order (spec, other choices 4)
# --------------------------------------------------------------------------- #

ORDER_STATES = _states(a=(3.0, SCHED), b=(1.0, SCHED), c=(2.0, SCHED), d=(2.0, SCHED),
                       e=(15.0, SCHED))
FIVE = _wp("abcde", deadline=100.0)


def _order(**kw):
    # e lies beyond the 10 m range: never solicited, never charged.
    return "".join(member_order(FIVE, model=_line_model(ORDER_STATES, range_m=10.0), **kw))


def test_the_f_order_is_capped_first_then_weight_per_second_then_id():
    assert _order() == "ebcda"                       # cheapest first; c before d by id
    assert _order(weights={A: 9.0}) == "eabcd"       # 9 / 3 s
    assert _order(capped={D}) == "debca"
    assert _order(capped={A, D}) == "daebc"          # capped: d (1/2 s) before a (1/3 s)
    assert _order(weights={E: 0.0}) == "bcdae"       # free but weightless: last
    assert _order(weights={A: 0.5, B: 2.0}) == "ebcda"
    # Without ferry physics every member costs the session: weight, then id.
    assert "".join(member_order(FIVE, model=FeasibilityModel())) == "abcde"
    assert "".join(member_order(FIVE, model=FeasibilityModel(), weights={C: 2.0})) == "cabde"


@pytest.mark.parametrize("weight", [-1.0, math.nan, math.inf, True, "1"])
def test_the_f_order_refuses_a_bad_weight(weight):
    with pytest.raises(ValueError):
        _order(weights={A: weight})


def test_the_f_order_prices_each_member_for_the_pass_and_the_snr_offset_given():
    """``dwell_j`` is the model's own price of the member alone, for the pass
    and offset the caller names, so either can reorder the members."""
    weights = {A: 1.0, B: 3.0}
    stop = _wp("ab", deadline=100.0)
    near = _pass_model(_states(a=(1.0, SCHED), b=(4.0, SCHED)))
    assert member_order(stop, model=near, weights=weights) == (A, B)      # 1/1 s > 3/4 s
    assert member_order(stop, model=near, weights=weights,
                        pass_kind=MissionPass.DELIVER) == (B, A)          # 3/1 s > 1/1 s
    mid = _pass_model(_states(a=(2.0, SCHED), b=(4.0, SCHED)))
    assert member_order(stop, model=mid, weights=weights) == (B, A)       # 3/4 s > 1/2 s
    assert member_order(stop, model=mid, weights=weights,
                        snr_offset_db=3.0) == (A, B)                      # 1/0.5 s > 3/2.5 s


# --------------------------------------------------------------------------- #
# The member walk (unit_U3b.md section 1.5)
# --------------------------------------------------------------------------- #

WALK_STATES = _states(a=(5.0, SCHED), b=(1.0, SCHED), c=(2.0, SCHED))
WALK = _wp("abc", deadline=100.0)
WALK_DEADLINES = {A: 100.0, B: 100.0, C: 100.0}


def _walk(order, *, budget=26.0, rule=RULE_BUDGET, deadlines=None, **kw):
    """From the dock at t = 0 the stop costs 10 s out, 10 s back and its
    members' dwell: a budget of 26 s leaves 6 s of dwell."""
    return admit_members(_line_model(WALK_STATES), T0, WALK, _devs(order), rule=rule,
                         budget_end=budget, deadlines=deadlines or WALK_DEADLINES,
                         device_states=WALK_STATES, **kw)


def test_the_walk_admits_in_its_order_and_skips_what_does_not_fit():
    got = _walk("abc")
    assert isinstance(got, MemberWalk) and got.ok
    assert got.admitted == (A, B) and got.refused == ((C, REASON_BUDGET),)
    assert got.verdict.home == 26.0 and got.verdict.next_state.clock == 16.0
    assert got.verdict == _line_model(WALK_STATES).admit(T0, got.stop, rule=RULE_BUDGET,
                                                         budget_end=26.0)
    # Skip, not stop: c does not fit after a, and b after it still does.
    skip = _walk("acb")
    assert skip.admitted == (A, B) and skip.refused == ((C, REASON_BUDGET),)


def test_the_order_is_injectable_and_decides_the_subset():
    assert _walk("cba").admitted == (B, C)
    assert _walk("cba").refused == ((A, REASON_BUDGET),)
    assert _walk("cba").verdict.home == 23.0


def test_a_walk_that_admits_everyone_returns_the_stop_itself():
    got = _walk("bca", budget=100.0)
    assert got.stop is WALK and got.refused == ()


def test_a_walk_that_admits_nobody():
    got = _walk("abc", budget=20.5)
    assert not got.ok and got.stop is None and got.verdict is None and got.admitted == ()
    assert [d for d, _ in got.refused] == [A, B, C]


@pytest.mark.parametrize("order", ["ab", "abcc", "abd", "aab"])
def test_the_order_must_be_a_permutation_of_the_members(order):
    with pytest.raises(ValueError):
        _walk(order)


def test_capped_members_alone_are_exempt_and_a_mixed_trial_keeps_its_clause():
    """a alone finishes at 15, past its own 12. Uncapped, it is refused as
    overdue; capped, it is exempt while alone, and the uncapped c then misses
    its own deadline of 14 beside it, so c is refused (critic B2)."""
    deadlines = {A: 12.0, B: 100.0, C: 14.0}
    free = _walk("abc", budget=1e9, rule=RULE_DEADLINE_BUDGET, deadlines=deadlines)
    assert free.admitted == (B, C) and free.refused == ((A, REASON_OVERDUE),)
    capped = _walk("abc", budget=1e9, rule=RULE_DEADLINE_BUDGET, deadlines=deadlines,
                   capped={A})
    assert capped.admitted == (A, B) and capped.refused == ((C, REASON_OVERDUE),)
    assert capped.stop.deadline_ts == 100.0      # b's: the stop's clause runs
    alone = _walk("abc", budget=1e9, rule=RULE_DEADLINE_BUDGET, deadlines=deadlines,
                  capped={A, B, C})
    # Every member capped: exempt, so all are admitted, and the stop keeps its
    # members' minimum deadline, finite, as the trace records it.
    assert alone.admitted == (A, B, C) and alone.stop.deadline_ts == 12.0


def test_a_veto_refuses_what_the_predicate_admits():
    got = _walk("abc", budget=1e9, veto=lambda trial, v: "budget" if C in trial.devices else None)
    assert got.admitted == (A, B) and got.refused == ((C, "budget"),)


def test_the_walk_prices_the_pass_it_is_given():
    """U3b's contract passes ``pass_kind`` to the predicate. The 10 s Pass-1
    upload follows COLLECT only (``FeasibilityModel.leg``): against 25 s, b
    alone is home at 31 s collecting and at 21 s delivering."""
    model = _line_model(WALK_STATES, upload=10.0)
    kw = dict(rule=RULE_BUDGET, budget_end=25.0, deadlines=WALK_DEADLINES,
              device_states=WALK_STATES)
    collect = admit_members(model, T0, WALK, _devs("bca"), **kw)
    assert not collect.ok and [d for d, _ in collect.refused] == [B, C, A]
    deliver = admit_members(model, T0, WALK, _devs("bca"), pass_kind=MissionPass.DELIVER, **kw)
    assert deliver.admitted == (B, C) and deliver.refused == ((A, REASON_BUDGET),)
    assert deliver.verdict.home == 23.0


def test_complements_group_the_refused_by_reason_in_the_predicates_order():
    states = _states(a=(1.0, NEW), b=(1.0, SCHED), c=(1.0, BEACON), d=(1.0, SCHED),
                     e=(1.0, SCHED))
    stop = _wp("abcde", deadline=10.0, bucket=NEW)
    deadlines = {A: 10.0, B: 20.0, C: 30.0, D: 40.0, E: 50.0}
    got = complements(stop, [(D, "budget"), (A, "overdue"), (E, "custom"), (B, "budget"),
                             (C, "energy")], deadlines=deadlines, device_states=states)
    assert [(wp.devices, why) for wp, why in got] == [
        ((A,), "overdue"), ((B, D), "budget"), ((C,), "energy"), ((E,), "custom")]
    assert [wp.deadline_ts for wp, _ in got] == [10.0, 20.0, 30.0, 50.0]
    assert [wp.bucket for wp, _ in got] == [NEW, SCHED, BEACON, SCHED]
    assert all(wp.position == stop.position for wp, _ in got)
    assert complements(stop, [], deadlines=deadlines, device_states=states) == ()


# --------------------------------------------------------------------------- #
# One stop, and the F family's fold
# --------------------------------------------------------------------------- #

PAIR_STATES = {**_states(a=(5.0, SCHED), b=(1.0, SCHED)), **_states(at=FAR, c=(2.0, SCHED))}
PAIR = _wp("ab", deadline=100.0)
SOLO = _wp("c", deadline=100.0, pos=FAR)
PAIR_DEADLINES = {A: 100.0, B: 100.0, C: 100.0}


def _fold(route, budget, *, require_all=False, rule=RULE_DEADLINE_BUDGET, **kw):
    """Whole, PAIR is home at 26 s and SOLO after it at 48 s (10 s out, 6 s of
    dwell, 10 s on, 2 s, 20 s back)."""
    return fold_members(route, T0, model=_line_model(PAIR_STATES), rule=rule,
                        budget_end=budget, deadlines=PAIR_DEADLINES,
                        device_states=PAIR_STATES, require_all=require_all, **kw)


def test_stops_that_fit_whole_are_flown_as_given():
    got = _fold([PAIR, SOLO], 48.0)
    assert isinstance(got, MemberFold) and got.feasible
    assert got.route[0] is PAIR and got.route[1] is SOLO and got.dropped == ()
    assert got.home == 48.0


def test_a_stop_that_fails_whole_is_reduced_and_its_complement_carries_the_reason():
    got = _fold([PAIR, SOLO], 22.0)
    (stop,) = got.route
    assert stop.devices == (B,) and stop.position == PAIR.position
    assert [(wp.devices, why) for wp, why in got.dropped] == [
        ((A,), REASON_BUDGET), ((C,), REASON_BUDGET)]
    assert got.dropped[1][0] is SOLO             # no member fits: dropped whole, as given


def test_require_all_makes_a_stop_that_admits_nobody_infeasible_and_stops():
    loose = _fold([SOLO, PAIR], 20.0)
    assert loose.feasible and loose.route == () and len(loose.dropped) == 2
    strict = _fold([SOLO, PAIR], 20.0, require_all=True)
    assert not strict.feasible and strict.route == ()
    assert strict.dropped == ((SOLO, REASON_BUDGET),)    # PAIR never tried: pruned
    assert strict.state == T0


def test_the_fold_reports_home_and_the_energy_of_the_pass_with_the_return_leg():
    got = _fold([PAIR], 26.0)
    assert got.route == (PAIR,) and got.home == 26.0
    # 10 s flying and 6 s hovering to PAIR, then 10 s flying home.
    assert got.state.energy_j == 100.0 * 10 + 200.0 * 6
    assert got.energy_j == got.state.energy_j + 100.0 * 10 == pass_energy_j(
        _line_model(PAIR_STATES), got.state)
    empty = _fold([], 26.0)
    assert empty.route == () and empty.home == 0.0 and empty.energy_j == 0.0 and empty.feasible
    legacy = FlightState((5.0, 0.0, 0.0), 3.0, 7.0)
    assert pass_energy_j(FeasibilityModel(), legacy) == 7.0


def test_the_fold_refuses_an_unknown_rule_or_a_non_boolean_require_all():
    with pytest.raises(ValueError):
        _fold([], 26.0, rule="bogus")
    with pytest.raises(TypeError):
        _fold([], 26.0, require_all=1)


def test_the_guard_needs_the_exempt_set_of_the_reduced_route():
    """Critic B1: a reduced stop is a new waypoint, so a protected set
    computed on the S3a stops does not contain it and its clause would run."""
    states = _states(a=(5.0, SCHED), b=(5.0, SCHED))
    stop = _wp("ab", deadline=1.0)                     # both long overdue
    model = _line_model(states)
    got = fold_members([stop], T0, model=model, rule=RULE_DEADLINE_BUDGET, budget_end=25.0,
                       deadlines={A: 1.0, B: 1.0}, device_states=states, require_all=True,
                       capped={A, B})
    (flown,) = got.route
    assert flown.devices == (A,) and got.feasible and flown.deadline_ts == 1.0
    guard = dict(rule=RULE_DEADLINE_BUDGET, budget_end=25.0, skip=False)
    assert model.fold(got.route, T0, protected={flown}, **guard).ok
    assert not model.fold(got.route, T0, protected={stop}, **guard).ok


def test_admit_stop_computes_the_order_only_when_the_stop_fails_whole():
    def refuse(_wp):
        raise AssertionError("the order was computed for a stop that fits whole")

    got = admit_stop(_line_model(PAIR_STATES), T0, PAIR, rule=RULE_BUDGET, budget_end=26.0,
                     deadlines=PAIR_DEADLINES, device_states=PAIR_STATES, order=refuse)
    assert isinstance(got, StopAdmission) and got.ok and got.stop is PAIR
    assert got.dropped == ()
    vetoed = admit_stop(_line_model(PAIR_STATES), T0, PAIR, rule=RULE_BUDGET,
                        budget_end=26.0, deadlines=PAIR_DEADLINES, device_states=PAIR_STATES,
                        veto=lambda trial, v: "budget" if A in trial.devices else None)
    assert vetoed.stop.devices == (B,) and [(wp.devices, r) for wp, r in vetoed.dropped] == [
        ((A,), "budget")]


def test_the_snr_offset_reaches_the_f_order_as_well_as_the_predicate():
    """Weights 1 and 2.1, dwell 1 s per metre less 0.5 s per dB, a 28.5 s
    budget. With no offset b (8 m) leads a (4 m), 2.1/8 > 1/4, and alone is
    home at 28 s (both at 32 s); at 3 dB a leads, 1/2.5 > 2.1/6.5, and alone
    is home at 22.5 s (both at 29 s). The fold and the trim reduce the stop in
    the order the offset gives, so each keeps the other member."""
    states = _states(a=(4.0, SCHED), b=(8.0, SCHED))
    stop = _wp("ab", deadline=1000.0)
    kw = dict(budget_end=28.5, deadlines={A: 1000.0, B: 1000.0}, device_states=states,
              weights={A: 1.0, B: 2.1})
    model = _pass_model(states)
    for offset, kept, left in ((0.0, B, A), (3.0, A, B)):
        fold = fold_members([stop], T0, model=model, rule=RULE_BUDGET, require_all=True,
                            snr_offset_db=offset, **kw)
        assert [wp.devices for wp in fold.route] == [(kept,)], offset
        assert _labels(fold.dropped) == [((left,), REASON_BUDGET)], offset
        trim = trim_members([stop], T0, model=model, snr_offset_db=offset, **kw)
        assert [wp.devices for wp in trim.route] == [(kept,)], offset


# --------------------------------------------------------------------------- #
# A stop none of whose members fits (spec, other choices 8; unit_U3b.md 3.1)
# --------------------------------------------------------------------------- #

U, V = DeviceID("u"), DeviceID("v")
#: Capped a costs 20 s of dwell: alone it is home at 10 + 20 + 10 = 40 s, past
#: the 30 s budget. Uncapped u costs 1 s but is due at 5 s, and alone finishes
#: at 11 s. Uncapped v costs 1 s and is due late: alone it is home at 21 s.
EDGE_STATES = _states(a=(20.0, SCHED), u=(1.0, SCHED), v=(1.0, SCHED))
EDGE_DEADLINES = {A: 1000.0, U: 5.0, V: 1000.0}
EDGE_KW = dict(budget_end=30.0, deadlines=EDGE_DEADLINES, device_states=EDGE_STATES)


def test_a_priority_stop_that_keeps_no_one_drops_each_member_with_its_own_reason():
    """Whole, the mixed stop {a, u} is held to u's deadline (B2) and fails
    overdue, while a alone is exempt and fails the budget. Each member is
    dropped with its own reason, so the capped a is never reported overdue,
    and its label is the one it gets when a co-member does fit."""
    model = _line_model(EDGE_STATES)
    stop, caps = _wp("au", deadline=5.0), frozenset({A})
    whole = reduce_stop(stop, stop.devices, deadlines=EDGE_DEADLINES,
                        device_states=EDGE_STATES, capped=caps)
    assert model.admit(T0, whole, rule=RULE_DEADLINE_BUDGET,
                       budget_end=30.0).reason == REASON_OVERDUE
    want = [((U,), REASON_OVERDUE), ((A,), REASON_BUDGET)]
    got = admit_stop(model, T0, stop, rule=RULE_DEADLINE_BUDGET, capped=caps, **EDGE_KW)
    assert got.stop is None and got.verdict is None and _labels(got.dropped) == want
    for require_all in (False, True):
        fold = fold_members([stop], T0, model=model, rule=RULE_DEADLINE_BUDGET,
                            require_all=require_all, capped=caps, **EDGE_KW)
        assert fold.route == () and _labels(fold.dropped) == want, require_all
        assert fold.feasible is not require_all
    trim = trim_members([stop], T0, model=model, capped=caps, **EDGE_KW)
    assert trim.order_used == ORDER_ARM_TRIMMED and trim.route == ()
    assert _labels(trim.dropped) == want
    # With v, who fits, the stop is reduced and the labels are the same.
    some = admit_stop(model, T0, _wp("auv", deadline=5.0), rule=RULE_DEADLINE_BUDGET,
                      capped=caps, **EDGE_KW)
    assert some.stop.devices == (V,) and _labels(some.dropped) == want


def test_a_stop_without_a_capped_member_that_keeps_no_one_is_dropped_whole_as_fold_drops_it():
    """U3b's rule (unit_U3b.md section 3.1), the Phase 3 trim's and S3b's: the
    stop itself, with the reason ``fold`` gives, whether or not other devices
    are capped, so a plain stop's label does not depend on the rest of the
    mission. Here a alone would fail the budget, but the stop reads overdue."""
    model = _line_model(EDGE_STATES)
    stop = _wp("au", deadline=5.0)
    plain = model.fold([stop], T0, rule=RULE_DEADLINE_BUDGET, budget_end=30.0, skip=True)
    assert plain.rejected == ((stop, REASON_OVERDUE),)
    for caps in (frozenset(), frozenset({V})):
        got = admit_stop(model, T0, stop, rule=RULE_DEADLINE_BUDGET, capped=caps, **EDGE_KW)
        assert got.dropped == plain.rejected and got.dropped[0][0] is stop, caps
        for require_all in (False, True):
            fold = fold_members([stop], T0, model=model, rule=RULE_DEADLINE_BUDGET,
                                require_all=require_all, capped=caps, **EDGE_KW)
            assert fold.dropped == plain.rejected and fold.dropped[0][0] is stop, caps
        trim = trim_members([stop], T0, model=model, capped=caps, **EDGE_KW)
        assert trim.dropped == plain.rejected and trim.dropped[0][0] is stop, caps


# --------------------------------------------------------------------------- #
# The in-flight trim (spec, other choices 9)
# --------------------------------------------------------------------------- #

def _trim(remainder, states, budget, deadlines=None, **kw):
    deadlines = deadlines or {d: 1000.0 for d in states}
    return trim_members(remainder, T0, model=_line_model(states), budget_end=budget,
                        deadlines=deadlines, device_states=states, **kw)


def test_a_remainder_that_passes_is_kept_as_it_is():
    res = _trim([PAIR, SOLO], PAIR_STATES, 48.0)
    assert isinstance(res, ReplanResult) and res.order_used == ORDER_CURRENT
    assert res.route[0] is PAIR and res.route[1] is SOLO and res.dropped == ()
    assert _trim([PAIR, SOLO], PAIR_STATES, None).order_used == ORDER_CURRENT
    # The check protects exempt stops: an overdue stop of capped members passes.
    late_pair = _wp("ab", deadline=1.0)
    late = {A: 1.0, B: 1.0, C: 100.0}
    kept = _trim([late_pair, SOLO], PAIR_STATES, 48.0, deadlines=late, capped={A, B})
    assert kept.order_used == ORDER_CURRENT and kept.route[0] is late_pair
    trimmed = _trim([late_pair, SOLO], PAIR_STATES, 48.0, deadlines=late)
    assert trimmed.order_used == ORDER_ARM_TRIMMED and trimmed.route == (SOLO,)
    assert trimmed.dropped == ((late_pair, REASON_OVERDUE),)


def test_a_priority_stop_sheds_its_uncapped_members_before_a_later_capped_member():
    """P1 holds capped a (3 s) and uncapped b (4 s); P2 far away holds capped c
    (2 s). Whole, the route is home at 49 s; a and c alone at 45 s. A plain
    fold keeps P1 whole and loses c; the trim keeps every capped member."""
    states = {**_states(a=(3.0, SCHED), b=(4.0, SCHED)), **_states(at=FAR, c=(2.0, SCHED))}
    p1, p2 = _wp("ab", deadline=1000.0), _wp("c", deadline=1000.0, pos=FAR)
    caps = {A, C}
    plain = fold_members([p1, p2], T0, model=_line_model(states), rule=RULE_DEADLINE_BUDGET,
                         budget_end=46.0, deadlines={d: 1000.0 for d in states},
                         device_states=states, require_all=False, capped=caps)
    assert plain.route == (p1,) and plain.dropped == ((p2, REASON_BUDGET),)
    res = _trim([p1, p2], states, 46.0, capped=caps)
    assert res.order_used == ORDER_ARM_TRIMMED
    assert [wp.devices for wp in res.route] == [(A,), (C,)] and res.route[1] is p2
    assert [(wp.devices, why) for wp, why in res.dropped] == [((B,), REASON_BUDGET)]


def test_priority_stops_fly_first():
    states = {**_states(d=(1.0, SCHED)), **_states(at=WEST, a=(1.0, SCHED))}
    near, capped_west = _wp("d", deadline=1000.0), _wp("a", deadline=1000.0, pos=WEST)
    # In the given order the route is home at 42 s; the capped stop first
    # leaves no room for the other.
    res = _trim([near, capped_west], states, 30.0, capped={A})
    assert res.order_used == ORDER_ARM_TRIMMED
    assert res.route == (capped_west,) and res.dropped == ((near, REASON_BUDGET),)


def test_a_priority_stop_whose_capped_members_cannot_fit_keeps_its_uncapped_ones():
    """Capped a alone is home at 40 s against a 25 s budget, so the
    protected-only trim cannot hold it and it is dropped; the stop is not
    dropped whole with it: uncapped b still fits (home at 21 s)."""
    states = _states(a=(20.0, SCHED), b=(1.0, SCHED))
    stop = _wp("ab", deadline=1000.0)
    res = _trim([stop], states, 25.0, capped={A})
    assert [wp.devices for wp in res.route] == [(B,)]
    assert [(wp.devices, why) for wp, why in res.dropped] == [((A,), REASON_BUDGET)]


def test_the_trim_takes_an_injected_order_and_its_rule():
    """WALK against 26 s leaves 6 s of dwell: the F order (cheapest first)
    keeps b and c, an order that tries a first keeps a and b. The rule is the
    arm's: an overdue pair that fits the budget passes the budget-only rule as
    it is and is trimmed under the deadline rule."""
    f_order = _trim([WALK], WALK_STATES, 26.0)
    assert [wp.devices for wp in f_order.route] == [(B, C)]
    assert _labels(f_order.dropped) == [((A,), REASON_BUDGET)]
    injected = _trim([WALK], WALK_STATES, 26.0, order=lambda wp: _devs("abc"))
    assert [wp.devices for wp in injected.route] == [(A, B)]
    assert _labels(injected.dropped) == [((C,), REASON_BUDGET)]
    late_pair, late = _wp("ab", deadline=1.0), {A: 1.0, B: 1.0, C: 100.0}
    budget_only = _trim([late_pair, SOLO], PAIR_STATES, 48.0, deadlines=late, rule=RULE_BUDGET)
    assert budget_only.order_used == ORDER_CURRENT and budget_only.route == (late_pair, SOLO)
    deadline = _trim([late_pair, SOLO], PAIR_STATES, 48.0, deadlines=late)
    assert deadline.order_used == ORDER_ARM_TRIMMED and deadline.route == (SOLO,)


def test_the_trim_refuses_a_malformed_remainder():
    with pytest.raises(ValueError):
        _trim([PAIR, PAIR], PAIR_STATES, 10.0)
    with pytest.raises(ValueError):
        _trim([PAIR, _wp("a", deadline=100.0)], PAIR_STATES, 10.0)
    with pytest.raises(ValueError):
        _trim([PAIR], PAIR_STATES, 10.0, rule="bogus")


# --------------------------------------------------------------------------- #
# The Phase 3 cliff instance (tests/unit/test_p3_final_fixes_mule.py)
# --------------------------------------------------------------------------- #

T2_LAYOUT = device_positions(8, 777, 100.0)
CLIFF_IDS = tuple(DeviceID(f"dev-{i}") for i in range(len(T2_LAYOUT)))
#: T2's deadline unit (Deadline(j) = t0 + 1500 s) and the runner's default (t0 + 60 s).
T2_UNIT, DEFAULT_UNIT = 25.0, 1.0


def _ids(*numbers):
    return tuple(DeviceID(f"dev-{i}") for i in numbers)


def _cliff(time_scale):
    """H1's plan of trial T2 (narrow, 1 MB, the seconds backhaul), a local copy
    of test_p3_final_fixes_mule._plan under no binding budget: the field-wide
    contact with the bound model, the states and the plan's deadlines."""
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
    return SimpleNamespace(contact=contact, sch=sch, t0=t0, model=sch.feasibility_model,
                           deadlines=dict(sch.last_plan_deadlines), states=sch.device_states)


def _cliff_fold(cl, budget, *, rule=RULE_DEADLINE_BUDGET, **kw):
    return fold_members([cl.contact], FlightState(DOCK, cl.t0), model=cl.model, rule=rule,
                        budget_end=cl.t0 + budget, deadlines=cl.deadlines,
                        device_states=cl.states, require_all=True, **kw)


def _in_stop_order(cl, members):
    return tuple(d for d in cl.contact.devices if d in set(members))


@pytest.fixture(scope="module")
def cliff():
    return _cliff(T2_UNIT)


def test_the_cliff_instance_admits_a_member_subset_where_s3b_admits_none(cliff):
    cl = cliff
    assert sorted(cl.contact.devices) == sorted(CLIFF_IDS)
    for budget, kept, home, left in ((60.0, (0, 1, 2, 4, 6), 54.490, (3, 5, 7)),
                                     (99.0, (0, 1, 2, 3, 4, 5, 6), 81.903, (7,))):
        got = _cliff_fold(cl, budget)
        (stop,) = got.route
        assert got.feasible and stop.devices == _in_stop_order(cl, _ids(*kept)), budget
        assert got.home - cl.t0 == pytest.approx(home, abs=1e-3), budget
        # The reduced stop: the contact's position, the members' own deadline
        # and bucket (all NEW, all due at t0 + 1500 s on the first mission).
        assert stop.position == cl.contact.position and stop.bucket is Bucket.NEW
        assert stop.deadline_ts == min(cl.deadlines[d] for d in stop.devices) == cl.t0 + 1500.0
        assert [(wp.devices, why) for wp, why in got.dropped] == [
            (_in_stop_order(cl, _ids(*left)), REASON_BUDGET)], budget
        # S3b, the recorded whole rule, still admits nobody on the same inputs.
        feas = filter_feasible([cl.contact], now=cl.t0, mule_pose=DOCK,
                               mission_deadline_ts=cl.t0 + budget, model=cl.model)
        assert feas.kept == [] and feas.dropped_budget == [cl.contact] and feas.dropped_plan == []
    whole = _cliff_fold(cl, 99.5)
    assert whole.route == (cl.contact,) and whole.route[0] is cl.contact and not whole.dropped
    assert whole.home - cl.t0 == pytest.approx(99.1154, abs=1e-3)


def test_at_the_default_deadline_unit_the_subset_is_held_to_its_deadline():
    cl = _cliff(DEFAULT_UNIT)
    for budget in (60.0, 99.0, 1000.0):
        got = _cliff_fold(cl, budget)
        (stop,) = got.route
        assert stop.devices == _in_stop_order(cl, _ids(0, 1, 2, 4, 6)), budget
        assert [(wp.devices, why, wp.deadline_ts - cl.t0) for wp, why in got.dropped] == [
            (_in_stop_order(cl, _ids(3, 5, 7)), REASON_OVERDUE, 60.0)], budget
    # The budget-only rule (the D arms') leaves the deadline out: 7 at 99 s.
    (stop,) = _cliff_fold(cl, 99.0, rule=RULE_BUDGET).route
    assert stop.devices == _in_stop_order(cl, _ids(0, 1, 2, 3, 4, 5, 6))
    # A stop of capped members only is exempt from the clause: the same 7.
    (stop,) = _cliff_fold(cl, 99.0, capped=set(CLIFF_IDS)).route
    assert stop.devices == _in_stop_order(cl, _ids(0, 1, 2, 3, 4, 5, 6))


def test_on_the_cliff_capped_members_are_admitted_first(cliff):
    cl = cliff
    caps = set(_ids(5, 7))
    assert member_order(cl.contact, model=cl.model) == _ids(4, 2, 0, 1, 6, 3, 5, 7)
    assert member_order(cl.contact, model=cl.model, capped=caps) == _ids(5, 7, 4, 2, 0, 1, 6, 3)
    got = _cliff_fold(cl, 60.0, capped=caps)
    (stop,) = got.route
    assert stop.devices == _in_stop_order(cl, _ids(2, 4, 5, 7))
    assert [(wp.devices, why) for wp, why in got.dropped] == [
        (_in_stop_order(cl, _ids(0, 1, 3, 6)), REASON_BUDGET)]


def test_the_member_walk_takes_the_d_arms_order_on_the_cliff(cliff):
    """The injectable order: the D arms' keys reduce to the device id within
    this stop (unit_U3b.md section 2.3), and the walk then admits dev-0 to
    dev-4, home at 57.623 s, as U3b's probe found."""
    cl = cliff
    kw = dict(rule=RULE_BUDGET, budget_end=cl.t0 + 60.0, deadlines=cl.deadlines,
              device_states=cl.states)
    by_id = admit_members(cl.model, FlightState(DOCK, cl.t0), cl.contact,
                          sorted(cl.contact.devices), **kw)
    assert by_id.admitted == _in_stop_order(cl, _ids(0, 1, 2, 3, 4))
    assert by_id.verdict.home - cl.t0 == pytest.approx(57.623, abs=1e-3)
    assert [d for d, _ in by_id.refused] == list(_ids(5, 6, 7))
    f_walk = admit_members(cl.model, FlightState(DOCK, cl.t0), cl.contact,
                           member_order(cl.contact, model=cl.model), **kw)
    assert f_walk.admitted == _in_stop_order(cl, _ids(0, 1, 2, 4, 6))
    assert f_walk.verdict.home - cl.t0 == pytest.approx(54.490, abs=1e-3)


def test_the_fold_reduces_the_cliff_in_an_injected_order(cliff):
    """``order`` reaches every stop the fold reduces (U4 passes memoised
    orders so): the by-id order gives dev-0 to dev-4, home at 57.623 s, where
    the F order gives dev-0, 1, 2, 4 and 6, home at 54.490 s."""
    cl = cliff
    by_id = _cliff_fold(cl, 60.0, rule=RULE_BUDGET, order=lambda wp: sorted(wp.devices))
    (stop,) = by_id.route
    assert stop.devices == _in_stop_order(cl, _ids(0, 1, 2, 3, 4))
    assert by_id.home - cl.t0 == pytest.approx(57.623, abs=1e-3)
    assert _labels(by_id.dropped) == [(_in_stop_order(cl, _ids(5, 6, 7)), REASON_BUDGET)]
    (stop,) = _cliff_fold(cl, 60.0, rule=RULE_BUDGET).route
    assert stop.devices == _in_stop_order(cl, _ids(0, 1, 2, 4, 6))


class _Sup:
    """The mule's widening, bound onto a stand-in (the Phase 3 critic B7 pattern)."""

    _widen_abandoned = MuleSupervisor._widen_abandoned

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.mule_id = MuleID("m")
        self._now = lambda: 0.0


def test_the_trims_complement_is_widened_by_the_mules_own_widening():
    cl = _cliff(T2_UNIT)
    res = trim_members([cl.contact], FlightState(DOCK, cl.t0), model=cl.model,
                       budget_end=cl.t0 + 60.0, deadlines=cl.deadlines, device_states=cl.states)
    assert res.order_used == ORDER_ARM_TRIMMED
    assert [wp.devices for wp in res.route] == [_in_stop_order(cl, _ids(0, 1, 2, 4, 6))]
    assert [(wp.devices, why) for wp, why in res.dropped] == [
        (_in_stop_order(cl, _ids(3, 5, 7)), REASON_BUDGET)]
    before = {d: (st.missed_count, st.miss_streak) for d, st in cl.states.items()}
    _Sup(cl.sch)._widen_abandoned(res.dropped_contacts, mission_round=1)
    changed = {d for d, st in cl.states.items() if (st.missed_count, st.miss_streak) != before[d]}
    assert changed == set(_ids(3, 5, 7))
    for d in _ids(3, 5, 7):
        assert cl.states[d].missed_count == before[d][0] + 1
        assert cl.states[d].miss_streak == before[d][1] + 1


# --------------------------------------------------------------------------- #
# The recorded pipeline is unchanged (Freeze Rule 1)
# --------------------------------------------------------------------------- #

def test_dropped_plan_is_the_last_field_and_empty_unless_given():
    assert [f.name for f in fields(FeasibilityResult)] == [
        "kept", "dropped_overdue", "dropped_budget", "dropped_energy", "dropped_delivery",
        "dropped_plan"]
    x, y, z, w, v = (_wp(n, deadline=1.0) for n in "abcde")
    recorded = FeasibilityResult([x], [y], [z], [w], [v])   # fl_scheduler.py's positional build
    assert recorded.dropped_plan == [] and recorded.n_dropped == 4
    assert recorded.dropped == [y, z, w, v]
    assert FeasibilityResult([x], [], []).dropped_plan == []
    planned = FeasibilityResult([x], [y], [], dropped_plan=[z])
    assert planned.n_dropped == 2 and planned.dropped == [y, z]


def test_s3cs_planned_count_leaves_the_plans_choices_out():
    """Critic C6: the plan's own drops are choices, not deadline shortfalls,
    so S3c's planned count (the mule's four lists by name) does not see them."""
    flown, choice = _wp("ab", deadline=1.0), _wp("cde", deadline=1.0)
    assert mission_planned_devices([flown], FeasibilityResult([flown], [], [],
                                                              dropped_plan=[choice])) == 2
    assert mission_planned_devices([flown], FeasibilityResult([flown], [choice], [])) == 5


def test_the_switch_values_live_in_s3b_and_the_plan_package_restates_them():
    assert MEMBER_ADMISSIONS == (MEMBER_ADMISSION_WHOLE, MEMBER_ADMISSION_SUBSET)
    assert MEMBER_ADMISSIONS == ("whole", "subset")          # the recorded value first
    assert PLAN_MEMBER_ADMISSIONS == MEMBER_ADMISSIONS


_STDLIB = {"__future__", "collections", "dataclasses", "math", "numbers", "typing"}
_FORBIDDEN = ("numpy", "hermes.l1", "hermes.mule", "hermes.mission", "experiments",
              "hermes.scheduler.policies", "hermes.scheduler.fl_scheduler")


def _imports(path: Path):
    """(module, where) per import: ``module`` level, ``checking`` (under
    ``if TYPE_CHECKING``) or ``local`` (inside a function)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []

    def name_of(node):
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        return ["." * node.level + (node.module or "")]

    def visit(nodes, where):
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                out.extend((name, where) for name in name_of(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.extend((name, "local") for sub in ast.walk(node)
                           if isinstance(sub, (ast.Import, ast.ImportFrom))
                           for name in name_of(sub))
            elif (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                    and node.test.id == "TYPE_CHECKING"):
                visit(node.body, "checking")
                visit(node.orelse, where)
            elif isinstance(node, ast.ClassDef):
                visit(node.body, where)
            elif isinstance(node, ast.If):
                visit(node.body, where)
                visit(node.orelse, where)
            elif isinstance(node, ast.Try):
                for block in [node.body, node.orelse, node.finalbody] + [
                        h.body for h in node.handlers]:
                    visit(block, where)

    visit(tree.body, "module")
    return out


_MODULE_LEVEL = ("hermes.scheduler.stages.s3b_feasibility",
                 "hermes.scheduler.stages.s3d_age_cap", ".types")


def test_member_subset_imports_hermes_types_the_stages_and_the_plan_types_only():
    """Numpy-free, nothing from hermes.l1, the mule or experiments, and no
    module-level edge up to the policies or the scheduler (unit_U3b.md
    section 1.5): the re-plan's result type is imported where it is used. The
    age-cap stage it imports does not import it back."""
    seen = _imports(REPO / "hermes/scheduler/plan/member_subset.py")
    assert seen
    for name, where in seen:
        assert not name.startswith(_FORBIDDEN), name
        if where == "module":
            assert (name.split(".")[0] in _STDLIB or name.startswith("hermes.types.")
                    or name in _MODULE_LEVEL), name
        else:
            assert name == "hermes.scheduler.routing.replan", (name, where)
    for name, where in _imports(REPO / "hermes/scheduler/stages/s3d_age_cap.py"):
        if where == "module":
            assert "member_subset" not in name and name != "hermes.scheduler.plan", name


def test_the_caps_stop_rules_have_one_definition_the_age_cap_stages():
    """U0's hand-off to U1 (one function for B2's deadline) and the review of
    U3: the walk names the age-cap stage's rules, so the stops it builds and
    the protected sets the guard takes from that stage (``cap_stops``) follow
    one definition, and this module defines none of its own."""
    assert member_subset.stop_deadline is s3d_age_cap.stop_deadline
    assert member_subset.is_exempt is s3d_age_cap.is_exempt
    assert member_subset.is_priority is s3d_age_cap.is_priority
    tree = ast.parse((REPO / "hermes/scheduler/plan/member_subset.py").read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert not defined & {"stop_deadline", "is_exempt", "is_priority", "_exempt", "_has_capped"}
    assert "stop_deadline" not in member_subset.__all__


def test_s3b_imports_no_plan_module_at_module_level():
    for name, where in _imports(REPO / "hermes/scheduler/stages/s3b_feasibility.py"):
        if where == "module":
            assert not name.startswith("hermes.scheduler.plan"), name


def _python(code: str) -> None:
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_recorded_path_never_loads_the_plan_package():
    """S3b holds the switch values so that naming them loads no plan module
    (nor the age-cap stage the walk imports); the member walk still imports
    cleanly afterwards, first, and after the age-cap stage."""
    _python(
        "import sys\n"
        "import hermes.scheduler.stages.s3b_feasibility as s3b\n"
        "import hermes.scheduler\n"
        "import hermes.scheduler.routing.replan\n"
        "assert s3b.MEMBER_ADMISSIONS == ('whole', 'subset')\n"
        "plan = sorted(m for m in sys.modules if m.startswith('hermes.scheduler.plan'))\n"
        "assert not plan, plan\n"
        "assert 'hermes.scheduler.stages.s3d_age_cap' not in sys.modules\n"
        "import hermes.scheduler.plan.member_subset\n"
    )
    _python(
        "import hermes.scheduler.plan.member_subset as ms\n"
        "import hermes.scheduler.policies\n"
        "from hermes.scheduler import FLScheduler\n"
        "assert ms.reduce_stop and FLScheduler\n"
    )
    _python(
        "import hermes.scheduler.stages.s3d_age_cap as cap\n"
        "import hermes.scheduler.plan.member_subset as ms\n"
        "assert ms.stop_deadline is cap.stop_deadline\n"
    )


# --------------------------------------------------------------------------- #
# Random instances: an independent re-implementation and the invariants
# --------------------------------------------------------------------------- #

N_CASES = 400


class _Dwell:
    """a + b*d seconds per member, unreachable beyond ``floor_m``; each dB of
    offset saves 0.05 s."""

    def __init__(self, a, b, floor_m):
        self.a, self.b, self.floor_m = a, b, floor_m

    def __call__(self, d, pass_kind, offset):
        if d > self.floor_m:
            return None
        return max(0.0, self.a + self.b * d - 0.05 * offset)


class _Upload:
    def __init__(self, s):
        self.s = s

    def __call__(self):
        return self.s


def _case(seed):
    rng = random.Random(104_729 * seed + 17)
    speed = rng.choice((1.0, 5.0, 12.0))
    scale = rng.choice((20.0, 60.0, 150.0))
    now = rng.choice((0.0, 1e6))
    states, deadlines, stops = {}, {}, []
    for i in range(rng.choice((1, 1, 2, 2, 3, 4))):
        pos = (rng.uniform(-scale, scale), rng.uniform(-scale, scale), 0.0)
        devs = []
        for j in range(rng.choice((1, 2, 3, 3, 4, 5))):
            did = DeviceID(f"s{i}m{j}")
            r, ang = rng.uniform(0.0, 40.0), rng.uniform(0.0, 2.0 * math.pi)
            states[did] = DeviceSchedulerState(
                device_id=did, last_known_position=(pos[0] + r * math.cos(ang),
                                                    pos[1] + r * math.sin(ang), 0.0),
                bucket=rng.choice(BUCKET_PRIORITY) if rng.random() < 0.95 else None)
            if deadlines and rng.random() < 0.2:
                deadlines[did] = rng.choice(sorted(deadlines.values()))
            else:
                deadlines[did] = now + rng.uniform(-20.0, 3.0 * scale / speed + 150.0)
            devs.append(did)
        buckets = [states[d].bucket for d in devs if states[d].bucket is not None]
        stops.append(ContactWaypoint(
            position=pos, devices=tuple(devs),
            bucket=min(buckets, key=BUCKET_PRIORITY.index) if buckets else NEW,
            deadline_ts=min(deadlines[d] for d in devs)))
    if rng.random() < 0.15:
        model = FeasibilityModel(cruise_speed_m_s=speed, session_time_s=rng.choice((0.0, 1.0, 3.0)))
    else:
        physics = FerryPhysics(
            dock=DOCK,
            member_dwell_s=_Dwell(rng.choice((0.5, 2.0, 5.0)), rng.choice((0.0, 0.1, 0.3)),
                                  rng.choice((30.0, 1e9))),
            upload_s=_Upload(rng.choice((0.0, 3.0))), p_move_w=143.6, p_hover_w=168.5,
            energy_capacity_j=(None if rng.random() < 0.7
                               else rng.uniform(0.0, 200.0 * scale / speed + 5000.0)),
            deadline_bounds=rng.choice(DEADLINE_BOUNDS), range_m=rng.choice((None, None, 35.0)),
            device_states=states)
        model = FeasibilityModel(cruise_speed_m_s=speed, session_time_s=1.0, ferry=physics)
    everyone = sorted(states)
    capped = (frozenset(d for d in everyone if rng.random() < 0.35) if rng.random() < 0.5
              else frozenset())
    weights = (None if rng.random() < 0.4
               else {d: rng.choice((0.0, 0.5, 1.0, 2.0, 5.0)) for d in everyone
                     if rng.random() < 0.9})
    deliver_by = math.inf
    if (model.ferry is not None and model.ferry.deadline_bounds == DEADLINE_BOUNDS_DELIVERY
            and rng.random() < 0.5):
        deliver_by = now + rng.uniform(0.0, 3.0 * scale / speed + 200.0)
    pose = DOCK if rng.random() < 0.6 else (rng.uniform(-scale, scale),
                                            rng.uniform(-scale, scale), 0.0)
    clock = now + rng.uniform(0.0, 30.0)
    state = FlightState(pose, clock, 0.0 if rng.random() < 0.6 else rng.uniform(0.0, 3000.0),
                        deliver_by)
    budget_end = None if rng.random() < 0.08 else clock + rng.uniform(
        0.0, 4.0 * scale / speed + 200.0)
    rng.shuffle(stops)
    return SimpleNamespace(route=stops, states=states, deadlines=deadlines, model=model,
                           capped=capped, weights=weights, state=state, budget_end=budget_end,
                           offset=rng.choice((0.0, 0.0, 0.0, 3.0, -2.0)))


def _ref_exempt(wp, capped):
    return bool(capped) and all(d in capped for d in wp.devices)


def _ref_stop(wp, keep, c):
    """A reduction written out: S3a's rules, B2's deadline under the cap."""
    devs = tuple(d for d in wp.devices if d in keep)
    uncapped = [d for d in devs if d not in c.capped]
    deadline = min(c.deadlines.get(d, math.inf) for d in (uncapped or devs))
    if len(devs) == len(wp.devices):
        return replace(wp, deadline_ts=deadline)
    buckets = [c.states[d].bucket for d in devs if c.states[d].bucket is not None]
    return ContactWaypoint(position=wp.position, devices=devs,
                           bucket=min(buckets, key=BUCKET_PRIORITY.index) if buckets else wp.bucket,
                           deadline_ts=deadline)


def _ref_order(c, wp):
    """The F order written out, dwell read straight from the physics."""
    def dwell(did):
        m = c.model
        if m.ferry is None or m.ferry.member_dwell_s is None:
            return m.session_time_s
        return m.ferry.dwell_s(ContactWaypoint(position=wp.position, devices=(did,),
                                               bucket=wp.bucket, deadline_ts=wp.deadline_ts),
                               MissionPass.COLLECT, c.offset)

    def worth(did):
        w = 1.0 if c.weights is None else c.weights.get(did, 1.0)
        t = dwell(did)
        return w / t if t > 0 else (math.inf if w > 0 else 0.0)

    return sorted(wp.devices, key=lambda d: (d not in c.capped, -worth(d), d))


def _ref_admit(c, cur, trial, rule):
    return c.model.admit(cur, trial, rule=rule, budget_end=c.budget_end,
                         protected=_ref_exempt(trial, c.capped), snr_offset_db=c.offset)


def _ref_groups(wp, refused, c):
    """Complements written out: one reduction per reason, in REASONS order."""
    return [(_ref_stop(wp, {d for d, why in refused if why == reason}, c), reason)
            for reason in REASONS if any(why == reason for _, why in refused)]


def _ref_fold(c, rule, require_all, route=None, order=None, tally=None):
    """The F family's fold from scratch: every stop by the member walk alone
    (no whole try first), in the F order or ``order``. A stop that keeps no
    one is dropped whole with the reason it fails whole, unless it holds a
    capped member: then each member goes with its own reason (counted in
    ``tally``). Returns (flown, dropped, state, ok)."""
    cur, flown, dropped = c.state, [], []
    for wp in (c.route if route is None else route):
        keep, last, refused = set(), None, []
        for d in (_ref_order(c, wp) if order is None else order(wp)):
            trial = _ref_stop(wp, keep | {d}, c)
            v = _ref_admit(c, cur, trial, rule)
            if v.ok:
                keep.add(d)
                last = (trial, v)
            else:
                refused.append((d, v.reason))
        if last is None:
            if any(d in c.capped for d in wp.devices):
                dropped.extend(_ref_groups(wp, refused, c))
                if tally is not None:
                    tally["emptied"] += 1
            else:
                whole = _ref_stop(wp, set(wp.devices), c)
                dropped.append((whole, _ref_admit(c, cur, whole, rule).reason))
            if require_all:
                return flown, dropped, cur, False
            continue
        flown.append(last[0])
        dropped.extend(_ref_groups(wp, refused, c))
        cur = last[1].next_state
    return flown, dropped, cur, True


def _capped_overdue(c, dropped):
    """Capped members some drop labels overdue: none in the F order."""
    return [d for wp, why in dropped if why == REASON_OVERDUE for d in wp.devices
            if d in c.capped]


def _shuffled(c, seed):
    """A random member order per stop, as an injected ``order``."""
    rng = random.Random(seed + 7_919)
    orders = {wp.devices: tuple(rng.sample(wp.devices, len(wp.devices))) for wp in c.route}
    return lambda wp: orders[wp.devices]


def _source(c, wp):
    (src,) = [s for s in c.route if wp.devices[0] in s.devices]
    return src


def _check_route(c, route, rule, *, maximal=lambda src: True):
    """The no-skip fold passes with the exempt set recomputed on ``route``
    (critic B1); each stop is a reduction of a distinct input stop, and where
    ``maximal(source)``, no member left out would still pass the predicate
    from the state the stop is flown in. Returns the guard fold."""
    exempt = frozenset(wp for wp in route if _ref_exempt(wp, c.capped))
    guard = c.model.fold(list(route), c.state, rule=rule, budget_end=c.budget_end, skip=False,
                         protected=exempt, snr_offset_db=c.offset)
    assert guard.ok, guard.rejected
    sources = [_source(c, wp) for wp in route]
    assert len({id(s) for s in sources}) == len(sources)
    before = c.state
    for wp, src, verdict in zip(route, sources, guard.verdicts):
        assert wp.position == src.position
        assert wp.devices == tuple(d for d in src.devices if d in set(wp.devices))
        assert wp == _ref_stop(src, set(wp.devices), c)
        if set(wp.devices) == set(src.devices) and not any(d in c.capped for d in src.devices):
            assert wp is src                     # not reduced: the very object
        if maximal(src):
            for d in set(src.devices) - set(wp.devices):
                trial = _ref_stop(src, set(wp.devices) | {d}, c)
                assert not _ref_admit(c, before, trial, rule).ok
        before = verdict.next_state
    return guard


def _partition(c, route, dropped, stops=None):
    got = sorted(d for wp in list(route) + [w for w, _ in dropped] for d in wp.devices)
    return got == sorted(d for wp in (c.route if stops is None else stops) for d in wp.devices)


@pytest.mark.parametrize("rule", [RULE_DEADLINE_BUDGET, RULE_BUDGET])
def test_the_fold_equals_an_independent_member_walk_on_random_instances(rule):
    """In the F order and in a random injected order (U4's memoised orders).
    In the F order no drop labels a capped member overdue."""
    reduced = unreduced = infeasible = capped_cases = 0
    tally = {"emptied": 0}
    for seed in range(N_CASES):
        c = _case(seed)
        shuffled = _shuffled(c, seed)
        for require_all in (False, True):
            for order in (None, shuffled):
                label = (seed, rule, require_all, order is None)
                got = fold_members(c.route, c.state, model=c.model, rule=rule,
                                   budget_end=c.budget_end, deadlines=c.deadlines,
                                   device_states=c.states, require_all=require_all,
                                   capped=c.capped, weights=c.weights, order=order,
                                   snr_offset_db=c.offset)
                flown, dropped, cur, ok = _ref_fold(c, rule, require_all, order=order,
                                                    tally=tally if order is None else None)
                assert list(got.route) == flown, label
                assert list(got.dropped) == dropped, label
                assert got.state == cur and got.feasible == ok, label
                guard = _check_route(c, got.route, rule)
                if got.route:
                    assert guard.home == got.home, label
                else:
                    assert got.home == c.model.home_at(got.state), label
                assert got.energy_j == pass_energy_j(c.model, got.state), label
                assert all(why in REASONS for _, why in got.dropped), label
                if ok:
                    assert _partition(c, got.route, got.dropped), label
                if order is not None:
                    continue
                assert not _capped_overdue(c, got.dropped), label
                infeasible += not ok
                if any(not any(wp is s for s in c.route) for wp in got.route):
                    reduced += 1
                else:
                    unreduced += 1
                capped_cases += bool(c.capped)
    # The instances reach every branch, priority stops that keep no one included.
    assert reduced > 50 and unreduced > 50 and infeasible > 20 and capped_cases > 50
    assert tally["emptied"] > 10


def test_where_nothing_is_reduced_the_fold_is_fold_skip_true_object_for_object():
    compared = 0
    for seed in range(N_CASES):
        c = _case(seed)
        got = fold_members(c.route, c.state, model=c.model, rule=RULE_DEADLINE_BUDGET,
                           budget_end=c.budget_end, deadlines=c.deadlines,
                           device_states=c.states, require_all=False,
                           weights=c.weights, snr_offset_db=c.offset)
        pieces = list(got.route) + [wp for wp, _ in got.dropped]
        if not all(any(wp is s for s in c.route) for wp in pieces):
            continue
        plain = c.model.fold(c.route, c.state, rule=RULE_DEADLINE_BUDGET,
                             budget_end=c.budget_end, skip=True, snr_offset_db=c.offset)
        assert [id(wp) for wp in got.route] == [id(wp) for wp in plain.route], seed
        assert [(id(wp), why) for wp, why in got.dropped] == [
            (id(wp), why) for wp, why in plain.rejected], seed
        assert got.state == plain.state and got.home == plain.home, seed
        compared += 1
    assert compared > 100


def test_the_walk_is_the_contract_of_unit_u3b_on_random_instances():
    """With no capped member and no veto, admit_members is exactly the
    pseudo-code of unit_U3b.md section 1.5, for any order and either rule."""
    walked = 0
    for seed in range(N_CASES):
        c = _case(seed)
        rng = random.Random(seed)
        for wp in c.route:
            order = rng.sample(list(wp.devices), len(wp.devices))
            for rule in (RULE_DEADLINE_BUDGET, RULE_BUDGET):
                kw = dict(rule=rule, budget_end=c.budget_end, deadlines=c.deadlines,
                          device_states=c.states, snr_offset_db=c.offset)
                got = admit_members(c.model, c.state, wp, order, **kw)
                s, verdict, refused = [], None, []
                for d in order:
                    v = c.model.admit(c.state, reduce_stop(wp, s + [d], deadlines=c.deadlines,
                                                           device_states=c.states),
                                      rule=rule, budget_end=c.budget_end,
                                      pass_kind=MissionPass.COLLECT, snr_offset_db=c.offset)
                    if v.ok:
                        s.append(d)
                        verdict = v
                    else:
                        refused.append((d, v.reason))
                stop = reduce_stop(wp, s, deadlines=c.deadlines,
                                   device_states=c.states) if s else None
                assert (got.stop, got.verdict, list(got.refused)) == (stop, verdict, refused)
                walked += 1
    assert walked > 1000


def _ref_protected_only(c, order=None):
    """The protected-only trim: each priority stop reduced to its capped
    members, walked in the F order (or ``order``); the capped members it keeps."""
    kept, cur = set(), c.state
    for wp in c.route:
        caps = [d for d in wp.devices if d in c.capped]
        if not caps:
            continue
        keep, last = set(), None
        ranked = _ref_order(c, wp) if order is None else order(wp)
        for d in [x for x in ranked if x in c.capped]:
            v = _ref_admit(c, cur, _ref_stop(wp, keep | {d}, c), RULE_DEADLINE_BUDGET)
            if v.ok:
                keep.add(d)
                last = v
        if last is not None:
            kept |= keep
            cur = last.next_state
    return kept


def test_the_trim_invariants_hold_on_random_instances():
    """In the F order and in a random injected order; in the F order no drop
    labels a capped member overdue."""
    orders, lookahead, capped_drops = set(), 0, 0
    for seed in range(N_CASES):
        c = _case(seed)
        exempt = frozenset(wp for wp in c.route if _ref_exempt(wp, c.capped))
        current = c.model.fold(c.route, c.state, rule=RULE_DEADLINE_BUDGET,
                               budget_end=c.budget_end, skip=False, protected=exempt,
                               snr_offset_db=c.offset)
        for order in (None, _shuffled(c, seed)):
            label = (seed, order is None)
            res = trim_members(c.route, c.state, model=c.model, budget_end=c.budget_end,
                               deadlines=c.deadlines, device_states=c.states, capped=c.capped,
                               weights=c.weights, order=order, snr_offset_db=c.offset)
            orders.add(res.order_used)
            if current.ok:
                assert res.order_used == ORDER_CURRENT, label
                assert [id(wp) for wp in res.route] == [id(wp) for wp in c.route], label
                assert res.dropped == (), label
                continue
            assert res.order_used == ORDER_ARM_TRIMMED and res.changed, label
            # The route passes with its exempt stops recomputed; each stop is
            # a reduction of a distinct remainder stop, and the rest's stops
            # are maximal where they are flown (a priority stop may also leave
            # out a member that fits there but would crowd a later capped
            # member out).
            _check_route(c, res.route, RULE_DEADLINE_BUDGET,
                         maximal=lambda src: not any(d in c.capped for d in src.devices))
            # Priority stops first, each part in the remainder's relative order.
            index = {id(s): i for i, s in enumerate(c.route)}
            sources = [_source(c, wp) for wp in res.route]
            flags = [any(d in c.capped for d in s.devices) for s in sources]
            assert flags == sorted(flags, reverse=True), label
            for part in (True, False):
                idx = [index[id(s)] for s, f in zip(sources, flags) if f is part]
                assert idx == sorted(idx), label
            # Every member of the remainder once, in the route or dropped;
            # drops in the remainder's order, each with a predicate reason.
            assert _partition(c, res.route, res.dropped), label
            dropped_at = [index[id(_source(c, wp))] for wp, _ in res.dropped]
            assert dropped_at == sorted(dropped_at), label
            assert all(why in REASONS for _, why in res.dropped), label
            # A capped member is dropped only when the protected-only trim fails.
            served = {d for wp in res.route for d in wp.devices}
            protected_only = _ref_protected_only(c, order)
            assert protected_only <= served, label
            if order is not None:
                continue
            assert not _capped_overdue(c, res.dropped), label
            capped_drops += any(d in c.capped for wp, _ in res.dropped for d in wp.devices)
            # Where a plain fold in priority-first order loses a capped member
            # the protected-only trim keeps, the trim's lookahead saved it.
            first = [s for s in c.route if any(d in c.capped for d in s.devices)]
            plain, _, _, _ = _ref_fold(c, RULE_DEADLINE_BUDGET, False,
                                       route=first + [s for s in c.route if s not in first])
            if not protected_only <= {d for wp in plain for d in wp.devices}:
                lookahead += 1
    assert orders == {ORDER_CURRENT, ORDER_ARM_TRIMMED}
    assert lookahead > 0 and capped_drops > 10


def test_s3b_never_fills_dropped_plan_on_random_instances():
    for seed in range(N_CASES):
        c = _case(seed)
        feas = filter_feasible(c.route, now=c.state.clock, mule_pose=c.state.pose,
                               mission_deadline_ts=c.budget_end, model=c.model, state=c.state,
                               snr_offset_db=c.offset)
        assert feas.dropped_plan == [], seed
        assert feas.n_dropped == len(feas.dropped) == (
            len(feas.dropped_overdue) + len(feas.dropped_budget) + len(feas.dropped_energy)
            + len(feas.dropped_delivery)), seed
