"""CMP_SC 8050 Phase 1: the plan checker on hand-built plans (exit check of Phase 1).

One small instance, worked by hand. The dock is at the origin, the drone flies
at 10 m/s, uploads for 5 s on landing and has 60 s. Every device sends
20,000 bytes: 2 s on wide (50 m reach, 10,000 B/s), 10 s on narrow (150 m,
2,000 B/s).

    device  position     importance  deadline
    a       (100, 0)     3           13 s
    b       (100, 30)    2           none
    c       (-100, 0)    1           40 s

The valid plan flies wide to (100, 0) for a (arrive 10 s, finish 12 s), then
200 m to (-100, 0) for c (arrive 32 s, finish 34 s), then 100 m home and
uploads: the trip ends at 49 s, worth 4.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from experiments.cs8050.problem import (
    RULE_DEADLINE,
    RULE_EMPTY_STOP,
    RULE_REACH,
    RULE_SERVED_TWICE,
    RULE_SETTING,
    RULE_SITE,
    RULE_TIME_LIMIT,
    RULE_UNKNOWN_DEVICE,
    Device,
    Instance,
    Plan,
    Setting,
    Stop,
    check_plan,
)

A, C, DOCK = (100.0, 0.0), (-100.0, 0.0), (0.0, 0.0)


@pytest.fixture
def inst() -> Instance:
    return Instance(
        devices=(
            Device("a", A, weight=3, deadline_s=13, size_bytes=20_000),
            Device("b", (100, 30), weight=2, size_bytes=20_000),
            Device("c", C, weight=1, deadline_s=40, size_bytes=20_000),
        ),
        settings=(
            Setting.constant("wide", 50, 10_000),
            Setting.constant("narrow", 150, 2_000),
        ),
        dock=DOCK,
        speed_m_s=10,
        time_limit_s=60,
        upload_s=5,
        sites=(A, C, DOCK),
    )


def valid_plan() -> Plan:
    return Plan("wide", (Stop(A, ("a",)), Stop(C, ("c",))))


# --------------------------------------------------------------------------- #
# The five hand-built plans of the exit check
# --------------------------------------------------------------------------- #

def test_valid_plan_is_worth_its_served_importance(inst):
    v = check_plan(inst, valid_plan())
    assert v.ok and v.rule is None
    assert v.value == 4.0
    assert [(t.arrival_s, t.dwell_s, t.finish_s, t.home_s) for t in v.timings] == [
        (10.0, 2.0, 12.0, 27.0),
        (32.0, 2.0, 34.0, 49.0),
    ]
    assert v.end_s == 49.0


def test_device_out_of_reach_is_refused(inst):
    v = check_plan(inst, Plan("wide", (Stop(A, ("a", "c")),)))
    assert not v.ok
    assert (v.rule, v.stop) == (RULE_REACH, 0)
    assert "'c'" in v.detail
    assert v.value == 0.0 and v.end_s is None


def test_deadline_binds_the_stops_finish_not_the_arrival(inst):
    # The drone reaches a at 10 s, before its 13 s deadline, but serving b
    # there too ends the stop at 14 s: late, as S3b's "collection" bound has it.
    v = check_plan(inst, Plan("wide", (Stop(A, ("a", "b")),)))
    assert (v.ok, v.rule, v.stop) == (False, RULE_DEADLINE, 0)
    assert v.timings[0].arrival_s == 10.0
    assert v.timings[0].finish_s == 14.0
    assert "'a'" in v.detail


def test_trip_over_the_time_limit_is_refused(inst):
    v = check_plan(replace(inst, time_limit_s=48), valid_plan())
    assert (v.ok, v.rule, v.stop) == (False, RULE_TIME_LIMIT, 1)
    assert v.timings[-1].home_s == 49.0


def test_time_limit_is_inclusive(inst):
    assert check_plan(replace(inst, time_limit_s=49), valid_plan()).ok


def test_device_served_twice_is_refused(inst):
    plan = Plan("wide", (Stop(A, ("a",)), Stop(C, ("c",)), Stop(C, ("c",))))
    v = check_plan(inst, plan)
    assert (v.ok, v.rule, v.stop) == (False, RULE_SERVED_TWICE, 2)
    assert "stop 1" in v.detail


# --------------------------------------------------------------------------- #
# The other rules and their order
# --------------------------------------------------------------------------- #

def test_empty_plan_is_valid_and_pays_no_upload(inst):
    v = check_plan(inst, Plan("wide"))
    assert v.ok and v.value == 0.0 and v.end_s == 0.0 and v.timings == ()


def test_unknown_setting_is_refused(inst):
    v = check_plan(inst, Plan("medium", (Stop(A, ("a",)),)))
    assert (v.ok, v.rule, v.stop) == (False, RULE_SETTING, None)


def test_empty_stop_and_unknown_device_are_refused(inst):
    assert check_plan(inst, Plan("wide", (Stop(A, ()),))).rule == RULE_EMPTY_STOP
    assert check_plan(inst, Plan("wide", (Stop(A, ("z",)),))).rule == RULE_UNKNOWN_DEVICE


def test_stop_must_be_at_a_listed_site_only_when_sites_are_listed(inst):
    plan = Plan("wide", (Stop((90, 0), ("a",)),))
    v = check_plan(inst, plan)
    assert (v.rule, v.stop) == (RULE_SITE, 0)
    anywhere = check_plan(replace(inst, sites=None), plan)
    assert anywhere.ok and anywhere.value == 3.0
    assert anywhere.end_s == 9.0 + 2.0 + 9.0 + 5.0


def test_deadline_is_reported_before_the_time_limit_at_the_same_stop(inst):
    # Narrow from the dock reaches all three; the stop ends at 30 s, after a's
    # deadline, and the trip ends at 35 s, after a 20 s limit: late is reported.
    plan = Plan("narrow", (Stop(DOCK, ("a", "b", "c")),))
    v = check_plan(replace(inst, time_limit_s=20), plan)
    assert (v.rule, v.stop) == (RULE_DEADLINE, 0)
    assert v.timings[0].home_s == 35.0


def test_structural_rules_come_before_the_walk(inst):
    # Stop 0 is late (it ends at 14 s) and stop 1 puts c 200 m from its stop:
    # reach (rule 4) is reported, though the walk would fail at stop 0 first.
    v = check_plan(inst, Plan("wide", (Stop(A, ("a", "b")), Stop(A, ("c",)))))
    assert (v.rule, v.stop) == (RULE_REACH, 1)


# --------------------------------------------------------------------------- #
# Rates by distance, travel rounding, validation
# --------------------------------------------------------------------------- #

def test_rate_steps_down_with_distance(inst):
    stepped = Setting("wide", 50, ((20, 10_000), (50, 5_000)))
    assert stepped.send_s(20_000, 0) == 2.0
    assert stepped.send_s(20_000, 20) == 2.0
    assert stepped.send_s(20_000, 30) == 4.0
    with pytest.raises(ValueError):
        stepped.rate_at(51)
    # b sits 30 m from (100, 0): 4 s, a 2 s, so the stop ends at 16 s.
    no_deadline = replace(inst, settings=(stepped,),
                          devices=tuple(replace(d, deadline_s=math.inf) for d in inst.devices))
    v = check_plan(no_deadline, Plan("wide", (Stop(A, ("a", "b")),)))
    assert v.ok and v.timings[0].dwell_s == 6.0 and v.end_s == 10.0 + 6.0 + 10.0 + 5.0


def test_travel_times_round_up_to_the_quantum(inst):
    assert replace(inst, travel_quantum_s=3.0).travel_s(DOCK, A) == 12.0
    assert replace(inst, travel_quantum_s=2.5).travel_s(DOCK, A) == 10.0
    assert replace(inst, travel_quantum_s=3.0).travel_s(A, A) == 0.0
    # Rounded up, the first leg takes 12 s, so stop 0 ends at 14 s: after a's
    # 13 s deadline. Rounding can turn a valid plan late.
    rounded = replace(inst, travel_quantum_s=3.0)
    v = check_plan(rounded, valid_plan())
    assert (v.rule, v.stop) == (RULE_DEADLINE, 0)
    # Without deadlines the legs are 12, 21 and 12 s: the trip ends at 54 s.
    free = replace(rounded, devices=tuple(replace(d, deadline_s=math.inf) for d in inst.devices))
    v = check_plan(free, valid_plan())
    assert v.ok and v.end_s == 12 + 2 + 21 + 2 + 12 + 5


@pytest.mark.parametrize("steps", [
    ((40, 10_000),),                      # does not end at the reach
    ((20, 5_000), (50, 10_000)),          # rate rises with distance
    ((30, 10_000), (20, 5_000), (50, 1)), # distances out of order
    (),                                   # no step
])
def test_bad_rate_steps_are_refused(steps):
    with pytest.raises(ValueError):
        Setting("wide", 50, steps)


def test_bad_instances_are_refused():
    dev = Device("a", A, weight=1)
    wide = Setting.constant("wide", 50, 1)
    with pytest.raises(ValueError):
        Instance(devices=(dev, dev), settings=(wide,))
    with pytest.raises(ValueError):
        Instance(devices=(dev,), settings=())
    with pytest.raises(ValueError):
        Instance(devices=(dev,), settings=(wide, wide))
    with pytest.raises(ValueError):
        Device("z", A, weight=0)
    with pytest.raises(ValueError):
        Instance(devices=(dev,), settings=(wide,), speed_m_s=0)
