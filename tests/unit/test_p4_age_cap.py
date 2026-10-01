"""FeRRy Phase 4 (unit U1): the age cap, stage 3d.

Pins the cap's rules as the later units call them (Phase 4 spec, other choices
5 and 6; the user's decision 1 of 2026-09-30):

* the age, a_j(m) = m - U_j with U_j = last_merged_round (never merged counts
  as 0), with no fallback to last_clean_round and no clamp, and its edges
  (never merged, merged in the last mission, merged in the mission being
  planned, a merge from the future, no round: critic B9). It follows the
  scheduler's own merge anchor (FLScheduler.record_merged), and it is the
  scorer's age after the mission if the mission does not merge the device
  (traces_scorer.age_profile), so the cap binds from mission S on (critic
  A11 (i));
* the capped set, age >= S - L;
* the stop sets on routes reduced to member subsets: exempt (every member
  capped, the predicate's ``protected``), mixed (the earliest uncapped
  deadline, critic B2) and priority (first in a trim, uncapped members shed
  first, each part in the order given), all recomputed on the route that is
  folded (critic B1), checked against the real predicate, and followed by
  value by U3's member walk (plan/member_subset.py), whose stops the guard
  checks with these sets;
* the cap key, which minimises the oldest unserved age, not the violation
  count (critic C5), and outranks V in the plan key;
* the violations: unplannable and crowded when the plan is built (a capped
  device alone at its class's S3a stop, priced from its distance to that
  stop, from the plan's takeoff with the Pass-1 upload, exempt from its
  deadline), dropped_in_flight and not_merged at close, through the commit's
  own checks, over a mission cycle;
* validation, and the layering: the stage is numpy-free, imports nothing from
  hermes.l1, the mule or experiments, and no legacy import path loads it.
"""

from __future__ import annotations

import ast
import itertools
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes.scheduler.fl_scheduler import FLScheduler
from hermes.scheduler.plan.types import (
    AgeCapSpec,
    Candidate,
    CapState,
    MemberFold,
    PlanClass,
    PlanScoreParams,
    ScoreTerms,
)
from hermes.scheduler.routing.replan import ORDER_ARM_TRIMMED
from hermes.scheduler.stages.s3a_cluster import cluster_by_rf_range
from hermes.scheduler.stages.s3b_feasibility import (
    REASON_BUDGET,
    REASON_ENERGY,
    REASON_OVERDUE,
    RULE_DEADLINE_BUDGET,
    FeasibilityModel,
    FerryPhysics,
    FlightState,
)
from hermes.scheduler.stages.s3d_age_cap import (
    CapStops,
    cap_key,
    cap_stops,
    capped_first,
    close_commit,
    close_violations,
    device_age,
    evaluate_cap,
    is_exempt,
    is_priority,
    plan_violations,
    priority_first,
    servable_alone,
    stop_deadline,
    visited_devices,
    with_cap_deadlines,
)
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionSlice,
    MuleID,
)
from hermes.types.scheduler import (
    CAP_CROWDED,
    CAP_DROPPED_IN_FLIGHT,
    CAP_NOT_MERGED,
    CAP_UNPLANNABLE,
    CapViolation,
    PlanCommit,
)

A, B, C, D, E, F, G, H = (DeviceID(x) for x in "abcdefgh")
DOCK = (0.0, 0.0, 0.0)
START = FlightState(DOCK, 0.0)
REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "hermes/scheduler/stages/s3d_age_cap.py"


def _state(did, merged=None, clean=0):
    st = DeviceSchedulerState(device_id=did)
    st.last_merged_round = merged
    st.last_clean_round = clean
    return st


def _states(**merged):
    return {DeviceID(d): _state(DeviceID(d), m) for d, m in merged.items()}


def _wp(devices, pos=(10.0, 0.0, 0.0), deadline=100.0, bucket=Bucket.SCHEDULED_THIS_ROUND):
    return ContactWaypoint(position=pos, devices=tuple(devices), bucket=bucket,
                           deadline_ts=deadline)


def _model(dwell_s=1.0, capacity=None, *, dwell_per_m=0.0, upload_s=0.0, states=None):
    """Ferry pricing: ``dwell_s`` per member plus ``dwell_per_m`` per metre from
    the member to its stop, 5 m/s, the return leg and an ``upload_s`` Pass-1
    upload; 100 W flying and hovering. Members are priced at the stop itself
    without ``states``, and at their last-known positions with them, as the
    class models the plan searches are bound (spec, other choices 1)."""
    physics = FerryPhysics(
        dock=DOCK, member_dwell_s=lambda d, p, o: dwell_s + dwell_per_m * d,
        upload_s=lambda: upload_s, p_move_w=100.0, p_hover_w=100.0,
        energy_capacity_j=capacity, device_states=states,
    )
    return FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=1.0, ferry=physics)


def _random_layout(rng, n_max):
    """Up to ``n_max`` devices on a 150 m field with random buckets and
    deadlines, clustered by S3a at a random radius: (ids, states, deadlines,
    S3a's stops)."""
    ids = [DeviceID(f"d{i}") for i in range(rng.randint(1, n_max))]
    states = {}
    for d in ids:
        st = DeviceSchedulerState(device_id=d)
        st.last_known_position = (rng.uniform(0.0, 150.0), rng.uniform(0.0, 150.0), 0.0)
        st.bucket = rng.choice(list(Bucket))
        states[d] = st
    deadlines = {d: rng.uniform(0.0, 100.0) for d in ids}
    return ids, states, deadlines, cluster_by_rf_range(ids, states, deadlines,
                                                       rng.choice([30.0, 60.0, 90.0]))


def _fold(route, protected, *, budget_end=1000.0):
    """The plan's guard fold: the plan's rule, no skipping."""
    return _model().fold(list(route), START, rule=RULE_DEADLINE_BUDGET,
                         budget_end=budget_end, skip=False, protected=protected)


def _scheduler(devices):
    sched = FLScheduler(now_fn=lambda: 0.0)
    sched.ingest_slice(MissionSlice(mule_id=MuleID("m1"), device_ids=tuple(devices),
                                    issued_round=0, issued_at=0.0))
    return sched


_SCORE = dict(v=-1.0, delta_s=10.0, time=0.01, coverage=0.0, link=0.0, energy_j=0.0,
              energy=0.0, served_weight=1.0, demand_weight=1.0)


def _commit(cap, queue, violations, *, mission_round=4):
    demand = tuple(cap.ages)
    return PlanCommit(
        mission_round=mission_round, band="wide", band_index=0, band_class_policy="search",
        queue=tuple(queue), demand=demand, weights={d: 1.0 for d in demand},
        budget_end=60.0, t_ref_s=200.0, score=_SCORE,
        constants=PlanScoreParams().constants(len(demand)), search_mode="exact",
        n_candidates=1, cap_s=cap.spec.s_missions, cap_lookahead=cap.spec.lookahead,
        ages=cap.ages, capped=cap.capped, violations=violations,
    )


def _reasons(violations):
    return [(v.device, v.reason) for v in violations]


# --------------------------------------------------------------------------- #
# The age
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("merged", [None, 0])
def test_a_device_never_merged_is_as_old_as_the_mission_being_planned(merged):
    """Decision 1: never merged counts as U = 0, so every device is 1 at the
    first mission (the mule numbers it 1) and m at mission m."""
    for m in (1, 2, 5):
        assert device_age(_state(A, merged), m) == m


def test_the_age_counts_missions_since_the_last_merged_update():
    """a(m) = m - U, the mission being planned included: merged in the last
    mission is 1. Merged in the mission being planned is 0, the scorer's age
    right after a merge: no clamp to 1, unlike D3's x."""
    assert device_age(_state(A, 4), 5) == 1
    assert device_age(_state(A, 2), 5) == 3
    assert device_age(_state(A, 5), 5) == 0
    assert device_age(_state(A, 0), 0) == 0


def test_the_age_never_falls_back_to_the_last_clean_round():
    """Decision 1 (a), not (c): a CLEAN whose update the merge left out is not
    service, so it never resets the age (D3 falls back to it; the cap does not)."""
    assert device_age(_state(A, None, clean=4), 5) == 5
    assert device_age(_state(A, 1, clean=4), 5) == 4


def test_a_merge_after_the_mission_being_planned_is_refused():
    with pytest.raises(ValueError, match="backwards"):
        device_age(_state(A, 6), 5)


def test_ages_need_the_mission_round():
    """Critic B9: the age counts the mule's missions, so no round, no age, with
    the cap on or off (the age coverage weights read the ages too)."""
    with pytest.raises(ValueError, match="B9"):
        device_age(_state(A), None)
    for spec in (AgeCapSpec(3), AgeCapSpec()):
        with pytest.raises(ValueError, match="B9"):
            evaluate_cap([A], _states(a=None), mission_round=None, spec=spec)


@pytest.mark.parametrize("mission_round, error", [
    (True, TypeError), (1.0, TypeError), ("3", TypeError), (-1, ValueError),
])
def test_the_age_refuses_a_bad_round(mission_round, error):
    with pytest.raises(error):
        device_age(_state(A), mission_round)


@pytest.mark.parametrize("merged, error", [
    (True, TypeError), (2.0, TypeError), ("1", TypeError), (-1, ValueError),
])
def test_the_age_refuses_a_bad_merge_anchor(merged, error):
    with pytest.raises(error):
        device_age(_state(A, merged), 5)


def test_the_age_needs_the_merge_anchor_itself():
    """A state without the field is refused, not read as never merged."""
    with pytest.raises(TypeError, match="last_merged_round"):
        device_age(SimpleNamespace(last_clean_round=3), 5)


def test_the_cap_binds_from_mission_s_on():
    """Critic A11 (i): a never-merged device is capped at mission S itself;
    the lookahead L moves that to S - L; S = 1 caps from the first mission."""
    states = _states(a=None)
    for spec, first in ((AgeCapSpec(3), 3), (AgeCapSpec(3, lookahead=1), 2), (AgeCapSpec(1), 1)):
        capped_at = [m for m in range(1, 7)
                     if A in evaluate_cap([A], states, mission_round=m, spec=spec).capped]
        assert capped_at == list(range(first, 7)), spec


def test_ages_follow_the_schedulers_own_merge_anchor():
    """The ages read what FLScheduler.record_merged writes, against the round
    the mule sets (set_mission_round), mission after mission; a mission that
    merges nothing ages every device."""
    sched = _scheduler([A, B, C])
    merges = {1: [A, B], 2: [], 3: [A], 4: [C]}
    seen = []
    for m in range(1, 6):
        sched.set_mission_round(m)
        cap = evaluate_cap([A, B, C], sched.device_states, mission_round=sched.mission_round,
                           spec=AgeCapSpec(2))
        seen.append((dict(cap.ages), sorted(cap.capped)))
        sched.record_merged(merges.get(m, []), m)
    assert seen == [
        ({A: 1, B: 1, C: 1}, []),
        ({A: 1, B: 1, C: 2}, [C]),
        ({A: 2, B: 2, C: 3}, [A, B, C]),
        ({A: 1, B: 3, C: 4}, [B, C]),
        ({A: 2, B: 4, C: 1}, [A, B]),
    ]


def _scorer_rows(merges):
    """A one-mule wall-clock trace whose Pass-1 CLEANs are what each mission
    merged, nothing lost or deferred (the shape of test_traces_scorer.py's)."""
    rows = [{"ts": 999.0, "event": "dock_bootstrapped", "role": "mule", "id": "m1"}]
    for i, devices in enumerate(merges):
        start, end = 1000.0 + 10.0 * i, 1010.0 + 10.0 * i
        rows.append({"ts": start, "event": "mission_started", "role": "mule", "id": "m1",
                     "mission_index": i})
        if not devices:
            rows.append({"ts": end, "event": "mission_empty", "role": "mule", "id": "m1",
                         "mission_round": i + 1})
        rows.append({"ts": end, "event": "mission_completed", "role": "mule", "id": "m1",
                     "mission_round": i + 1, "pass_1_contacts": 1,
                     "pass_2_contacts": 1 if devices else 0,
                     "pass_1_updates": len(devices) or None, "pass_1_scheduled": 3,
                     "pass_1_clean_devices": [str(d) for d in devices] or None})
    return rows


def test_the_age_is_the_scorers_age_after_the_mission_if_it_does_not_merge():
    """The cap counts in the scorer's unit (decision 1 (a)): planning mission
    m, a device's age is its age_profile age after mission m - 1 plus the
    mission being planned. So with L = 0 a device is capped exactly when, left
    unmerged now, the scorer counts it at age >= S after this mission."""
    from experiments.analysis.traces_scorer import age_profile  # noqa: WPS433
    from experiments.exp4.events_consumer import observation_from_rows  # noqa: WPS433

    devices = [A, B, C]
    merges = [[A, B], [], [A], [C], []]
    device_rows = [{"ts": 998.0, "event": "device_ready", "role": "device", "id": str(d)}
                   for d in devices]
    sched = _scheduler(devices)
    for k, merged in enumerate(merges, start=1):
        sched.record_merged(merged, k)
        obs = observation_from_rows(cluster_rows=[], mule_rows=_scorer_rows(merges[:k]),
                                    device_rows=device_rows, n_devices=len(devices))
        for d in devices:
            one_hot = {str(x): float(x == d) for x in devices}
            scored = age_profile(obs, [str(x) for x in devices], weights=one_hot).network_aou_final
            assert device_age(sched.device_states[d], k + 1) == scored + 1, (k, d)


def test_the_cap_state_keeps_the_demand_order_and_derives_the_capped_set():
    states = _states(a=4, b=1, c=None, d=3)
    cap = evaluate_cap([C, A, D, B], states, mission_round=5, spec=AgeCapSpec(3))
    assert list(cap.ages.items()) == [(C, 5), (A, 1), (D, 2), (B, 4)]
    assert cap.capped == {C, B}
    ahead = evaluate_cap([C, A, D, B], states, mission_round=5, spec=AgeCapSpec(3, lookahead=1))
    assert ahead.capped == {C, D, B}
    off = evaluate_cap([C, A, D, B], states, mission_round=5, spec=AgeCapSpec())
    assert dict(off.ages) == dict(cap.ages) and off.capped == frozenset()
    assert dict(evaluate_cap([], states, mission_round=5, spec=AgeCapSpec(3)).ages) == {}


@pytest.mark.parametrize("demand, error", [
    ([A, A], ValueError),              # a device demanded twice
    ("ab", TypeError),                 # one string, not device ids
    ([A, ""], TypeError),
    ([A, 7], TypeError),
    ([A, E], KeyError),                # no state for e
])
def test_the_cap_state_refuses_a_bad_demand(demand, error):
    with pytest.raises(error):
        evaluate_cap(demand, _states(a=None, b=None), mission_round=2, spec=AgeCapSpec(2))


def test_the_cap_state_needs_a_cap_spec():
    with pytest.raises(TypeError):
        evaluate_cap([A], _states(a=None), mission_round=2, spec=3)


# --------------------------------------------------------------------------- #
# The stop sets and their deadlines
# --------------------------------------------------------------------------- #

def test_a_stop_is_exempt_mixed_or_neither():
    capped = {A, C, D}
    both, one_of_two, none, alone = (
        _wp([A, C]), _wp([D, B], pos=(20.0, 0.0, 0.0)), _wp([E], pos=(0.0, 20.0, 0.0)),
        _wp([F], pos=(30.0, 0.0, 0.0)),
    )
    route = [both, one_of_two, none, alone]
    assert [is_exempt(wp, capped) for wp in route] == [True, False, False, False]
    assert [is_priority(wp, capped) for wp in route] == [True, True, False, False]
    sets = cap_stops(route, capped)
    assert sets.exempt == {both} and sets.mixed == {one_of_two}
    assert sets.priority == {both, one_of_two}
    assert cap_stops(route, capped | {F}).exempt == {both, alone}


def test_without_a_cap_nothing_is_exempt_or_first_and_the_stops_are_s3as():
    stops = [_wp([A, B], deadline=5.0), _wp([C], pos=(0.0, 20.0, 0.0), deadline=7.0)]
    assert cap_stops(stops, frozenset()) == CapStops()
    kept = with_cap_deadlines(stops, deadlines={A: 5.0, B: 9.0, C: 7.0}, capped=frozenset())
    assert all(new is old for new, old in zip(kept, stops))
    assert priority_first(stops, frozenset()) == tuple(stops)
    assert capped_first([C, A, B], frozenset()) == (C, A, B)


def test_the_stop_deadline_is_the_earliest_uncapped_one():
    deadlines = {A: 1.0, B: 50.0, C: 30.0, D: 2.0}
    assert stop_deadline([A, B, C], deadlines=deadlines) == 1.0        # nothing capped: S3a's
    assert stop_deadline([A, B, C], deadlines=deadlines, capped={A}) == 30.0   # B2
    assert stop_deadline([A, B, C, D], deadlines=deadlines, capped={A, D}) == 30.0
    assert stop_deadline([A, D], deadlines=deadlines, capped={A, D}) == 1.0    # exempt, finite
    assert stop_deadline([A, E], deadlines=deadlines, capped={A}) == math.inf  # as S3a: no deadline
    assert stop_deadline([E], deadlines=deadlines) == math.inf
    with pytest.raises(ValueError):
        stop_deadline([], deadlines=deadlines)
    with pytest.raises(TypeError):
        stop_deadline("ab", deadlines=deadlines)


def test_without_a_cap_the_stop_deadline_is_s3as_on_random_layouts():
    """One rule for whole and reduced stops, S3a's when nothing is capped."""
    rng = random.Random(7)
    for _ in range(200):
        _, _, deadlines, stops = _random_layout(rng, 8)
        for wp in stops:
            assert stop_deadline(wp.devices, deadlines=deadlines) == wp.deadline_ts
        kept = with_cap_deadlines(stops, deadlines=deadlines, capped=())
        assert all(new is old for new, old in zip(kept, stops))


def test_only_a_mixed_stop_led_by_a_capped_deadline_is_re_issued():
    """B2 applies to whole S3a stops too; every other stop is the same object."""
    deadlines = {A: 1.0, B: 50.0, C: 30.0, E: 40.0, D: 2.0, F: 3.0, G: 60.0}
    capped = {A, E, D, F}
    led = _wp([A, B], deadline=1.0, bucket=Bucket.NEW)                   # mixed: moves to 50
    trailing = _wp([C, E], pos=(0.0, 20.0, 0.0), deadline=30.0)          # mixed: stays at 30
    exempt = _wp([D, F], pos=(0.0, 40.0, 0.0), deadline=2.0)             # exempt: stays at 2
    plain = _wp([G], pos=(40.0, 0.0, 0.0), deadline=60.0)
    out = with_cap_deadlines([led, trailing, exempt, plain], deadlines=deadlines, capped=capped)
    assert out[1] is trailing and out[2] is exempt and out[3] is plain
    moved = out[0]
    assert moved != led and moved.deadline_ts == 50.0
    assert (moved.position, moved.devices, moved.bucket) == (led.position, led.devices, led.bucket)
    sets = cap_stops(out, capped)
    assert sets.exempt == {exempt} and sets.mixed == {moved, trailing}
    with pytest.raises(TypeError):
        with_cap_deadlines([("a",)], deadlines=deadlines, capped=capped)


def test_the_sets_are_recomputed_on_the_reduced_route():
    """Critic B1: a reduced stop is a new waypoint, so a set carried over from
    S3a's stops misses it; and shedding changes what a stop is."""
    deadlines = {A: 1.0, B: 50.0, C: 20.0}
    capped = {A, C}

    def reduced(members):
        return _wp(members, deadline=stop_deadline(members, deadlines=deadlines, capped=capped))

    whole = reduced([A, B, C])                     # mixed at S3a
    assert cap_stops([whole], capped).exempt == frozenset()
    shed = reduced([A, C])                         # the uncapped member shed: exempt
    assert shed not in cap_stops([whole], capped).priority
    assert cap_stops([shed], capped).exempt == {shed}
    assert shed.deadline_ts == 1.0                 # the earliest of all, finite
    kept_b = reduced([A, B])                       # still mixed, b's deadline now
    assert cap_stops([kept_b], capped).mixed == {kept_b} and kept_b.deadline_ts == 50.0
    only_b = reduced([B])                          # no capped member left
    assert cap_stops([only_b], capped) == CapStops()


def test_the_guard_needs_the_exempt_set_of_the_final_route():
    """Critic B1 with the real predicate: an overdue capped device's stop,
    reduced to that device, fails the guard fold under the set computed on the
    S3a stop and passes under the set computed on the route flown."""
    deadlines = {A: 1.0, C: 1.0}                   # both long overdue at arrival (t = 2 s)
    capped = {A, C}
    whole = _wp([A, C], deadline=stop_deadline([A, C], deadlines=deadlines, capped=capped))
    alone = _wp([A], deadline=stop_deadline([A], deadlines=deadlines, capped=capped))
    stale = cap_stops([whole], capped).exempt
    fresh = cap_stops([alone], capped).exempt
    assert alone not in stale and alone in fresh
    refused = _fold([alone], stale)
    assert not refused.ok and refused.rejected[0][1] == REASON_OVERDUE
    assert _fold([alone], fresh).ok


def test_a_mixed_stop_holds_its_uncapped_members_to_their_deadlines():
    """Critic B2 with the real predicate (finish at 4 s: 2 s out, 1 s per
    member): a capped member's own deadline is excused, an uncapped member's
    is not. Protecting the stop whole would let the uncapped member miss."""
    deadlines = {A: 1.0, B: 100.0, E: 3.0}         # a capped and overdue
    capped = {A}
    on_time, late = with_cap_deadlines(
        [_wp([A, B], deadline=1.0), _wp([A, E], pos=(0.0, 10.0, 0.0), deadline=1.0)],
        deadlines=deadlines, capped=capped,
    )
    assert (on_time.deadline_ts, late.deadline_ts) == (100.0, 3.0)
    assert cap_stops([on_time, late], capped).exempt == frozenset()
    assert _fold([on_time], cap_stops([on_time], capped).exempt).ok
    refused = _fold([late], cap_stops([late], capped).exempt)
    assert not refused.ok and refused.rejected[0][1] == REASON_OVERDUE
    # The per-stop protection B2 replaces: e rides a's exemption and is late.
    s3a_late = _wp([A, E], pos=(0.0, 10.0, 0.0), deadline=1.0)
    assert _fold([s3a_late], {s3a_late}).ok


def test_priority_stops_lead_a_trim_and_capped_members_lead_a_stop():
    capped = {A, D}
    first, mixed, last, exempt = (
        _wp([B], pos=(1.0, 0.0, 0.0)), _wp([C, A], pos=(2.0, 0.0, 0.0)),
        _wp([E], pos=(3.0, 0.0, 0.0)), _wp([D], pos=(4.0, 0.0, 0.0)),
    )
    assert priority_first([first, mixed, last, exempt], capped) == (mixed, exempt, first, last)
    assert capped_first([C, A, E, D, B], capped) == (A, D, C, E, B)
    with pytest.raises(TypeError):
        capped_first("ab", capped)


def test_each_part_keeps_the_order_it_was_given():
    """Both orders are stable partitions: within each part the route's (the
    members') order stands, not the stops' positions, ids or kinds (exempt and
    mixed stops interleave), so a trim moves a stop only across the parts."""
    capped = {A, C, E}
    plain_30, mixed_20 = _wp([F], pos=(30.0, 0.0, 0.0)), _wp([C, D], pos=(20.0, 0.0, 0.0))
    exempt_40, plain_5 = _wp([E], pos=(40.0, 0.0, 0.0)), _wp([B], pos=(5.0, 0.0, 0.0))
    mixed_10, plain_50 = _wp([A, G], pos=(10.0, 0.0, 0.0)), _wp([H], pos=(50.0, 0.0, 0.0))
    route = [plain_30, mixed_20, exempt_40, plain_5, mixed_10, plain_50]
    assert priority_first(route, capped) == (
        mixed_20, exempt_40, mixed_10, plain_30, plain_5, plain_50)
    assert capped_first([E, B, C, G, A, D], capped) == (E, C, A, B, G, D)
    rng = random.Random(3)
    for _ in range(300):
        ids = [DeviceID(f"d{i}") for i in range(rng.randint(1, 9))]
        rng.shuffle(ids)
        capped = {d for d in ids if rng.random() < 0.4}
        groups = {}
        for d in ids:
            groups.setdefault(rng.randrange(4), []).append(d)
        route = [_wp(members, pos=(rng.uniform(0.0, 100.0), rng.uniform(0.0, 100.0), 0.0))
                 for members in groups.values()]
        for given, out, first in (
            (route, priority_first(route, capped), lambda wp: is_priority(wp, capped)),
            (ids, capped_first(ids, capped), lambda d: d in capped),
        ):
            k = sum(map(first, given))
            ranks = [given.index(x) for x in out]
            assert sorted(ranks) == list(range(len(given)))              # a permutation
            assert all(map(first, out[:k])) and not any(map(first, out[k:]))
            assert ranks[:k] == sorted(ranks[:k]) and ranks[k:] == sorted(ranks[k:])


def test_the_stop_sets_refuse_what_is_not_a_waypoint():
    with pytest.raises(TypeError):
        cap_stops([_wp([A]), (A,)], {A})
    with pytest.raises(TypeError):
        CapStops(exempt=frozenset({"a"}))
    with pytest.raises(ValueError):
        CapStops(exempt=frozenset({_wp([A])}), mixed=frozenset({_wp([A])}))


def test_the_member_walk_follows_these_stop_rules(monkeypatch):
    """One rule with U3's member walk (plan/member_subset.py), which builds and
    admits the reduced stops, while U5's guard and in-flight checks take their
    protected sets from cap_stops here (critic B1): the two must agree by
    value, or the guard would refuse a correct plan. On random S3a layouts and
    capped sets: the walk's deadline for every member subset is
    stop_deadline's (B2), and it re-issues a whole stop exactly when
    with_cap_deadlines does; every predicate call its fold and its trim make
    is protected exactly when the stop is exempt and carries stop_deadline's
    deadline; the route its fold admits passes the guard fold under this
    module's exempt set; the trim flies stops in priority_first's order; and
    its member order is capped_first."""
    from hermes.scheduler.plan import member_subset  # noqa: WPS433 (unit U3)

    calls = []
    admit = FeasibilityModel.admit

    def spy(self, state, wp, **kwargs):
        calls.append((wp, kwargs.get("protected", False)))
        return admit(self, state, wp, **kwargs)

    monkeypatch.setattr(FeasibilityModel, "admit", spy)
    rng = random.Random(5)
    reissued = protected_calls = trimmed = 0
    for _ in range(150):
        ids, states, deadlines, stops = _random_layout(rng, 6)
        capped = frozenset(d for d in ids if rng.random() < 0.4)
        common = dict(deadlines=deadlines, device_states=states, capped=capped)
        for wp in stops:
            for k in range(1, len(wp.devices) + 1):
                for members in itertools.combinations(wp.devices, k):
                    deadline = stop_deadline(members, deadlines=deadlines, capped=capped)
                    assert member_subset.stop_deadline(
                        members, deadlines=deadlines, capped=capped) == deadline
                    assert member_subset.reduce_stop(wp, members, **common).deadline_ts == deadline
        planned = with_cap_deadlines(stops, deadlines=deadlines, capped=capped)
        for wp, ours in zip(stops, planned):
            theirs = member_subset.reduce_stop(wp, wp.devices, **common)
            assert theirs == ours and (theirs is wp) == (ours is wp)
            reissued += ours is not wp
        model = _model(2.0, dwell_per_m=0.05, upload_s=3.0, states=states)
        budget_end = rng.uniform(20.0, 120.0)
        walk = dict(model=model, budget_end=budget_end, **common)
        calls.clear()
        fold = member_subset.fold_members(stops, START, rule=RULE_DEADLINE_BUDGET,
                                          require_all=False, **walk)
        trim = member_subset.trim_members(planned, START, **walk)
        for wp, protected in calls:
            assert protected == is_exempt(wp, capped), wp
            assert wp.deadline_ts == stop_deadline(wp.devices, deadlines=deadlines, capped=capped)
            protected_calls += protected
        guard = model.fold(list(fold.route), START, rule=RULE_DEADLINE_BUDGET,
                           budget_end=budget_end, skip=False,
                           protected=cap_stops(fold.route, capped).exempt)
        assert guard.ok, guard.rejected
        if trim.order_used == ORDER_ARM_TRIMMED:
            trimmed += 1
            source = {d: wp for wp in planned for d in wp.devices}
            flown = [source[wp.devices[0]] for wp in trim.route]
            order = priority_first(planned, capped)
            assert flown == sorted(flown, key=order.index)
        for wp in stops:
            members = member_subset.member_order(wp, model=model, capped=capped)
            assert capped_first(members, capped) == members
    assert reissued and protected_calls and trimmed        # every branch was reached


# --------------------------------------------------------------------------- #
# The cap key
# --------------------------------------------------------------------------- #

def test_the_cap_key_minimises_the_oldest_unserved_age():
    """Critic C5: leaving three out at age 4 beats leaving one out at 5; the
    number of violations is not the key."""
    cap = CapState(AgeCapSpec(3), {A: 5, B: 3, C: 4, D: 4, E: 4})
    three_fours, five_three = cap_key({A, B}, cap), cap_key({C, D, E}, cap)
    assert (three_fours, five_three) == ((4, 4, 4), (5, 3))
    assert three_fours < five_three
    assert cap_key({A, B, C, D, E}, cap) == ()
    assert cap_key([], cap) == (5, 4, 4, 4, 3)
    assert cap_key((d for d in (A, B, C)), cap) == (4, 4)
    # Equal ages tie, whoever they are: V decides between such plans.
    assert cap_key({A, B, C, D}, cap) == cap_key({A, B, C, E}, cap) == (4,)


def test_uncapped_devices_never_enter_the_cap_key():
    cap = CapState(AgeCapSpec(3), {A: 2, B: 1, C: 3})
    assert cap_key(set(), cap) == (3,)
    assert cap_key({C}, cap) == ()
    assert cap_key(set(), CapState(AgeCapSpec(), {A: 9, B: 9})) == ()


def test_serving_more_capped_devices_always_lowers_the_cap_key():
    """L831: a plan that drops a capped device loses to any feasible one that
    keeps it and every capped device the first keeps (nested served sets)."""
    rng = random.Random(4)
    ids = [DeviceID(f"d{i}") for i in range(8)]
    for _ in range(500):
        cap = CapState(AgeCapSpec(rng.randint(1, 4)), {d: rng.randint(0, 6) for d in ids})
        smaller = {d for d in ids if rng.random() < 0.5}
        larger = smaller | {d for d in ids if rng.random() < 0.3}
        before, after = cap_key(smaller, cap), cap_key(larger, cap)
        if (larger - smaller) & cap.capped:
            assert after < before
        else:
            assert after == before


def test_the_cap_key_outranks_v_in_the_plan_key():
    """In U0's plan key a plan that keeps the oldest capped device beats one
    with a far better V that leaves it out."""
    cap = CapState(AgeCapSpec(2), {A: 3, B: 1})
    cls = PlanClass(name="wide", index=0, radius_m=60.0, model=_model(), outage=lambda d: 0.0)

    def candidate(stop, v):
        fold = MemberFold(route=(stop,), dropped=(), state=START, home=10.0, feasible=True,
                          energy_j=0.0)
        return Candidate(cls=cls, fold=fold, terms=ScoreTerms(**dict(_SCORE, v=v)),
                         cap_key=cap_key(fold.served, cap))

    keeps, drops = candidate(_wp([A]), -5.0), candidate(_wp([B], pos=(0.0, 20.0, 0.0)), -0.1)
    assert (keeps.cap_key, drops.cap_key) == ((), (3,))
    assert min([drops, keeps], key=lambda c: c.key) is keeps


# --------------------------------------------------------------------------- #
# Violations when the plan is built
# --------------------------------------------------------------------------- #
#
# Budget 30 s at 5 m/s with no upload. On "wide" (1 s per member) the stop at
# (10, 0, 0) takes 2 + 1 + 2 = 5 s alone, the one at (80, 0, 0) 16 + 1 + 16 =
# 33 s and the one at (0, 100, 0) 41 s. On "narrow" (20 s per member) the stop
# at (10, 0, 0) takes 44 s whole but 24 s for either member alone, the one at
# (5, 0, 0) 22 s and the one at (0, 100, 0) 60 s.

WIDE = (_model(dwell_s=1.0), [
    _wp([A, B], pos=(10.0, 0.0, 0.0)), _wp([C], pos=(80.0, 0.0, 0.0)),
    _wp([D], pos=(0.0, 100.0, 0.0)),
])
NARROW = (_model(dwell_s=20.0), [
    _wp([A, B], pos=(10.0, 0.0, 0.0)), _wp([C], pos=(5.0, 0.0, 0.0)),
    _wp([D], pos=(0.0, 100.0, 0.0)),
])


def test_a_device_is_servable_alone_at_its_own_s3a_stop_of_some_class():
    """a and b fit alone on either class, although on narrow their stop does
    not fit whole; c only on narrow, whose S3a stop for it is nearer; d on
    neither. A pinned band (FB+c) judges its own class only."""
    capped = [A, B, C, D]
    narrow_model, narrow_stops = NARROW
    assert not narrow_model.admit(START, narrow_stops[0], rule=RULE_DEADLINE_BUDGET,
                                  budget_end=30.0, protected=True).ok
    assert servable_alone(capped, [WIDE, NARROW], start=START, budget_end=30.0) == {A, B, C}
    assert servable_alone(capped, [WIDE], start=START, budget_end=30.0) == {A, B}
    assert servable_alone(capped, [NARROW], start=START, budget_end=30.0) == {A, B, C}
    assert servable_alone([], [WIDE], start=START, budget_end=30.0) == frozenset()


def test_a_capped_devices_own_deadline_never_makes_it_unplannable():
    """Its stop alone is exempt: an overdue deadline does not refuse it."""
    stale = (WIDE[0], [_wp([A], deadline=0.5)])
    alone = _wp([A], deadline=0.5)
    assert stale[0].admit(START, alone, rule=RULE_DEADLINE_BUDGET, budget_end=30.0).reason \
        == REASON_OVERDUE                                        # were it not capped
    assert servable_alone([A], [stale], start=START, budget_end=30.0) == {A}


def test_the_budget_and_the_energy_clause_make_a_device_unplannable():
    """Alone at (10, 0, 0) a needs 100 W x 4 s flying + 100 W x 1 s hovering
    = 500 J: a 450 J battery refuses it, though the budget would not."""
    lean = (_model(dwell_s=1.0, capacity=450.0), [_wp([A])])
    verdict = lean[0].admit(START, _wp([A]), rule=RULE_DEADLINE_BUDGET, budget_end=30.0,
                            protected=True)
    assert verdict.reason == REASON_ENERGY
    assert servable_alone([A], [lean], start=START, budget_end=30.0) == frozenset()
    far = WIDE[0].admit(START, _wp([C], pos=(80.0, 0.0, 0.0)), rule=RULE_DEADLINE_BUDGET,
                        budget_end=30.0, protected=True)
    assert far.reason == REASON_BUDGET
    assert servable_alone([C], [WIDE], start=START, budget_end=30.0) == frozenset()


def test_without_a_budget_nothing_is_unplannable():
    """No budget, no gate (the opt-in contract): every device fits alone."""
    assert servable_alone([A, B, C, D], [WIDE, NARROW], start=START, budget_end=None) \
        == {A, B, C, D}


def test_the_alone_check_prices_the_pass_1_upload():
    """Pass-1 pricing, as the plan's own folds: home = finish + return +
    upload. Alone at (40, 0, 0) with a 5 s dwell, a takes 8 + 5 + 8 = 21 s,
    and 31 s with a 10 s upload: a 25 s budget makes it unplannable only with
    the upload, and 31 s is just enough."""
    far = [_wp([A], pos=(40.0, 0.0, 0.0))]
    cap = CapState(AgeCapSpec(2), {A: 2})
    for upload, budget, servable, reason in (
        (0.0, 25.0, {A}, CAP_CROWDED),
        (10.0, 25.0, frozenset(), CAP_UNPLANNABLE),
        (10.0, 31.0, {A}, CAP_CROWDED),
    ):
        got = servable_alone([A], [(_model(5.0, upload_s=upload), far)], start=START,
                             budget_end=budget)
        assert got == servable, (upload, budget)
        assert _reasons(plan_violations(cap, (), servable=got)) == [(A, reason)]


def test_the_alone_check_flies_from_the_plans_takeoff():
    """budget_end is absolute and the mission clock need not restart at 0, so
    the check starts from the plan's own start: taking off at 500 s with the
    budget ending at 530 s, a's 31 s (as above) refuses it and b's 7 + 5 + 7 +
    10 = 29 s fits; from 0 s both would."""
    stops = [_wp([A], pos=(40.0, 0.0, 0.0)), _wp([B], pos=(35.0, 0.0, 0.0))]
    classes = [(_model(5.0, upload_s=10.0), stops)]
    takeoff = FlightState(DOCK, 500.0)
    got = servable_alone([A, B], classes, start=takeoff, budget_end=530.0)
    assert got == {B}
    assert _reasons(plan_violations(CapState(AgeCapSpec(2), {A: 2, B: 2}), (), servable=got)) \
        == [(A, CAP_UNPLANNABLE), (B, CAP_CROWDED)]
    assert servable_alone([A, B], classes, start=START, budget_end=530.0) == {A, B}


def test_the_alone_check_serves_a_device_at_its_s3a_stop_not_where_it_sits():
    """Every plan the search can fly is built from S3a's stops (spec, other
    choices 3), so a device is judged alone at its own S3a stop, priced from
    its distance to that stop by a class model bound to the device states
    (other choices 1), not at its own position. S3a anchors b (the earlier
    deadline) and takes a, 55 m away, into a stop at their centroid (32.5, 0,
    0): a sits 5 m from the dock (2 + 2 + 0 = 4 s alone there), but alone at
    its stop it takes 6.5 + (2 + 2.75) + 6.5 = 17.75 s, which a 15 s budget
    refuses, as it refuses every plan that serves a."""
    states = {}
    for d, x in ((A, 5.0), (B, 60.0)):
        states[d] = _state(d)
        states[d].last_known_position = (x, 0.0, 0.0)
        states[d].bucket = Bucket.SCHEDULED_THIS_ROUND
    stops = cluster_by_rf_range([A, B], states, {A: 50.0, B: 40.0}, 60.0)
    assert [(wp.position, wp.devices) for wp in stops] == [((32.5, 0.0, 0.0), (B, A))]
    model = _model(2.0, dwell_per_m=0.1, states=states)
    for pos, fits in (((32.5, 0.0, 0.0), False), ((5.0, 0.0, 0.0), True)):
        assert model.admit(START, _wp([A], pos=pos), rule=RULE_DEADLINE_BUDGET, budget_end=15.0,
                           protected=True).ok is fits
    got = servable_alone([A], [(model, stops)], start=START, budget_end=15.0)
    assert got == frozenset()
    assert _reasons(plan_violations(CapState(AgeCapSpec(2), {A: 3, B: 1}), (), servable=got)) \
        == [(A, CAP_UNPLANNABLE)]


def test_servable_alone_needs_each_classs_s3a_stops():
    model, stops = WIDE
    with pytest.raises(ValueError, match="no stop"):
        servable_alone([A, E], [WIDE], start=START, budget_end=30.0)
    with pytest.raises(ValueError, match="two stops"):
        servable_alone([A], [(model, stops + [_wp([A], pos=(1.0, 1.0, 0.0))])], start=START,
                       budget_end=30.0)
    with pytest.raises(TypeError):
        servable_alone([A], [(model, [(A,)])], start=START, budget_end=30.0)
    with pytest.raises(TypeError):
        servable_alone("a", [WIDE], start=START, budget_end=30.0)


def test_each_capped_device_the_plan_leaves_out_gets_one_reason():
    """Unplannable when no class serves it alone, crowded otherwise; served or
    uncapped devices get none; the planning age; the commit's order."""
    cap = CapState(AgeCapSpec(2), {A: 3, B: 1, C: 2, D: 4, E: 2})   # b is not capped
    got = plan_violations(cap, served={A, B}, servable={A, C, E})
    assert got == (CapViolation(D, 4, CAP_UNPLANNABLE), CapViolation(C, 2, CAP_CROWDED),
                   CapViolation(E, 2, CAP_CROWDED))
    assert plan_violations(cap, served=[A, C, D, E], servable=()) == ()
    assert _reasons(plan_violations(cap, served=(), servable=[A, B, C, D, E])) == [
        (A, CAP_CROWDED), (C, CAP_CROWDED), (D, CAP_CROWDED), (E, CAP_CROWDED),
    ]
    assert plan_violations(CapState(AgeCapSpec(), {A: 9}), served=(), servable=()) == ()


def test_plan_violations_refuse_a_plan_outside_its_demand():
    cap = CapState(AgeCapSpec(2), {A: 3})
    with pytest.raises(ValueError, match="outside"):
        plan_violations(cap, served={A, B}, servable=())
    with pytest.raises(TypeError):
        plan_violations(dict(cap.ages), served=(), servable=())
    with pytest.raises(TypeError):
        plan_violations(cap, served="a", servable=())


def test_the_commit_accepts_the_cap_and_its_plan_time_violations():
    """What U5 commits: evaluate_cap's ages and capped set and plan_violations'
    records satisfy the commit's own checks."""
    cap = evaluate_cap([A, B, C, D], _states(a=None, b=3, c=1, d=None), mission_round=4,
                       spec=AgeCapSpec(3))
    assert dict(cap.ages) == {A: 4, B: 1, C: 3, D: 4} and cap.capped == {A, C, D}
    commit = _commit(cap, (_wp([A, B]),), plan_violations(cap, served={A, B}, servable={C}))
    assert _reasons(commit.violations) == [(D, CAP_UNPLANNABLE), (C, CAP_CROWDED)]
    uncapped = evaluate_cap([A, B], _states(a=None, b=3), mission_round=4, spec=AgeCapSpec())
    none = plan_violations(uncapped, {A}, servable=())
    assert none == () and _commit(uncapped, (_wp([A]),), none).capped == frozenset()


# --------------------------------------------------------------------------- #
# Violations when the mission closes
# --------------------------------------------------------------------------- #

def test_close_marks_capped_served_devices_dropped_in_flight_or_not_merged():
    cap = CapState(AgeCapSpec(2), {A: 2, B: 3, C: 1, D: 5, E: 2})   # c is not capped
    queue = (_wp([A, B, C]), _wp([D], pos=(0.0, 20.0, 0.0)))
    commit = _commit(cap, queue, plan_violations(cap, served={A, B, C, D}, servable={E}))
    # In flight the first stop shed b and the second was dropped; only c merged.
    closed = close_commit(commit, [_wp([A, C])], merged=[C])
    assert closed.closed and closed.visited == {A, C}
    assert [(v.device, v.age, v.reason) for v in closed.violations] == [
        (E, 2, CAP_CROWDED), (B, 3, CAP_DROPPED_IN_FLIGHT), (D, 5, CAP_DROPPED_IN_FLIGHT),
        (A, 2, CAP_NOT_MERGED),
    ]
    assert commit.visited is None                                   # the open commit is kept
    assert close_violations(commit, visited={A, B, C, D}, merged={A, B, D}) == ()


def test_the_empty_path_closes_with_nothing_merged():
    """Pass 1 flew but collected nothing: every visited capped device is
    not_merged. A re-plan to nothing flies no stop: every one is dropped."""
    cap = CapState(AgeCapSpec(2), {A: 2, B: 3, C: 1})
    queue = (_wp([A, B, C]),)
    commit = _commit(cap, queue, ())
    assert _reasons(close_commit(commit, queue, merged=()).violations) == [
        (A, CAP_NOT_MERGED), (B, CAP_NOT_MERGED),
    ]
    assert _reasons(close_commit(commit, (), merged=()).violations) == [
        (A, CAP_DROPPED_IN_FLIGHT), (B, CAP_DROPPED_IN_FLIGHT),
    ]


def test_a_device_left_out_at_plan_time_gets_no_close_violation():
    """One violation per device per mission: b keeps its plan-time reason even
    when a stop inserted in flight reaches it and its update merges."""
    cap = CapState(AgeCapSpec(2), {A: 2, B: 4})
    commit = _commit(cap, (_wp([A]),), plan_violations(cap, served={A}, servable={B}))
    closed = close_commit(commit, [_wp([A]), _wp([B], pos=(0.0, 20.0, 0.0))], merged=[A, B])
    assert _reasons(closed.violations) == [(B, CAP_CROWDED)]
    assert closed.visited == {A, B}


def test_the_visited_set_is_the_members_of_the_stops_flown():
    assert visited_devices([_wp([A, B]), _wp([C], pos=(0.0, 5.0, 0.0))]) == {A, B, C}
    assert visited_devices([]) == frozenset()
    with pytest.raises(TypeError):
        visited_devices([A])


def test_close_refuses_a_closed_plan_and_bad_inputs():
    cap = CapState(AgeCapSpec(2), {A: 2})
    commit = _commit(cap, (_wp([A]),), ())
    closed = close_commit(commit, [_wp([A])], [A])
    with pytest.raises(ValueError, match="closed"):
        close_commit(closed, [_wp([A])], [A])
    with pytest.raises(ValueError, match="closed"):
        close_violations(closed, visited={A}, merged={A})
    with pytest.raises(TypeError):
        close_commit(commit, [A], [A])                             # flown holds stops
    with pytest.raises(TypeError):
        close_violations(commit, visited="a", merged=())
    with pytest.raises(TypeError):
        close_violations(cap, visited=(), merged=())


def test_three_missions_under_the_cap():
    """The cap over the mule's cycle, on the scheduler's own anchors: plan
    (ages, capped set, key, plan-time reasons), close, record_merged, next
    mission. S = 2, budget 30 s: c's stop (0, 150, 0) never fits alone."""
    sched = _scheduler([A, B, C])
    stops = {A: _wp([A]), B: _wp([B], pos=(0.0, 20.0, 0.0)), C: _wp([C], pos=(0.0, 150.0, 0.0))}
    classes = [(_model(), list(stops.values()))]
    served_of = {1: [A], 2: [B], 3: [A, B]}
    flown_of = {1: [A], 2: [], 3: [A, B]}        # mission 2 dropped b's stop in flight
    merged_of = {1: [A], 2: [], 3: [B]}
    record = []
    for m in (1, 2, 3):
        sched.set_mission_round(m)
        cap = evaluate_cap([A, B, C], sched.device_states, mission_round=sched.mission_round,
                           spec=AgeCapSpec(2))
        served = set(served_of[m])
        servable = servable_alone(cap.capped - served, classes, start=START, budget_end=30.0)
        commit = _commit(cap, [stops[d] for d in served_of[m]],
                         plan_violations(cap, served, servable=servable), mission_round=m)
        closed = close_commit(commit, [stops[d] for d in flown_of[m]], merged_of[m])
        sched.record_merged(merged_of[m], m)
        record.append((dict(cap.ages), cap_key(served, cap), _reasons(closed.violations)))
    assert record == [
        ({A: 1, B: 1, C: 1}, (), []),
        ({A: 1, B: 2, C: 2}, (2,), [(C, CAP_UNPLANNABLE), (B, CAP_DROPPED_IN_FLIGHT)]),
        ({A: 2, B: 3, C: 3}, (3,), [(C, CAP_UNPLANNABLE), (A, CAP_NOT_MERGED)]),
    ]


# --------------------------------------------------------------------------- #
# Layering
# --------------------------------------------------------------------------- #

def test_the_stage_imports_only_the_standard_library_hermes_types_the_plan_types_and_s3b():
    """Numpy-free, nothing from hermes.l1, the mule or experiments (spec,
    conventions): physics reaches it only through S3b's model."""
    names = set()
    for node in ast.walk(ast.parse(STAGE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    allowed = {"__future__", "dataclasses", "math", "numbers", "typing",
               "hermes.types.ids", "hermes.types.scheduler", "hermes.scheduler.plan.types",
               "hermes.scheduler.stages.s3b_feasibility"}
    assert names <= allowed, sorted(names - allowed)


def test_no_legacy_import_path_loads_the_stage():
    """Freeze Rule 1: the recorded pipelines never import the cap, nor through it
    the plan package; and the stage imports cleanly in a fresh interpreter,
    before or after the plan package (no import cycle)."""
    legacy = ("hermes.scheduler", "hermes.scheduler.fl_scheduler", "hermes.scheduler.stages",
              "hermes.scheduler.stages.s3b_feasibility", "hermes.scheduler.routing.replan",
              "hermes.scheduler.policies", "hermes.mule.mule_main", "hermes.processes.mule")
    probes = {
        "legacy": (f"for name in {legacy!r}:\n    importlib.import_module(name)\n"
                   "loaded = sorted(m for m in sys.modules if m.startswith("
                   "('hermes.scheduler.plan', 'hermes.scheduler.stages.s3d')))\n"
                   "assert not loaded, loaded\n"),
        "stage first": ("importlib.import_module('hermes.scheduler.stages.s3d_age_cap')\n"
                        "importlib.import_module('hermes.scheduler.plan')\n"),
        "plan first": ("importlib.import_module('hermes.scheduler.plan')\n"
                       "importlib.import_module('hermes.scheduler.stages.s3d_age_cap')\n"),
    }
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for name, body in probes.items():
        done = subprocess.run([sys.executable, "-c", "import importlib, sys\n" + body],
                              cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, (name, done.stderr)
