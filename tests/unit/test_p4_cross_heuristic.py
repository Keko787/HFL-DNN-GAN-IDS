"""FeRRy Phase 4 (unit U6): the flight slot's fixed fillings
(``hermes/scheduler/policies/cross_heuristic.py``).

What is pinned:

* one filling per flight-slot value, named by it, in its order;
* ``committed`` (arm F) is today's flight: index 0 without a fold, and the
  committed band, without an arrival view, in both passes;
* FX's next stop is the nearest remaining stop whose move to the front keeps
  the rest of the plan feasible, tried nearest first (ties by position, then
  devices), else index 0; also with the predicate itself as ``fits``;
* FX acts in Pass 1 only (the spec, other choices 2: "FX switches band per
  stop in Pass 1 only; Pass 2 flies b̄"), and its next-stop half after each
  stop (decision 5): in Pass 2 and at takeoff it is the committed slot, with
  no fold and no view; both fillings refuse a call that does not name the
  pass and the departure;
* FX's band is never slower than the committed class at the arrival SNR and
  never reaches fewer devices: the fastest class that reaches every committed
  target, ties to more devices, then the committed class, then the index;
  with one class it is the committed one;
* critic A7 at the last Pass-1 stop, on the real runtime and channel: where a
  narrower class reaches one more member, the "most devices" rule lands past a
  budget the committed class meets and FX keeps the committed class; where a
  faster class reaches the same members, FX takes it and lands earlier;
* on the runtime's own arrival views the invariants hold and the plan on FX's
  class solicits every committed target;
* through the supervisor's seam as the hand-off to unit U7 states it (the
  slot called at every stop of both passes): Pass 2 flies b̄ in the queue's
  order at every stop, and Pass 1 starts at the plan's first stop;
* layering: nothing from numpy, ``hermes.l1``, ``hermes.mule``,
  ``hermes.mission`` or ``experiments``.

The loopback FX missions (the predicate at every departure, FX at the last
stop end to end) are the supervisor's, unit U7's
``tests/integration/test_p4_plan_missions.py``.
"""

from __future__ import annotations

import ast
import math
import random
from pathlib import Path

import pytest

from hermes.l1.mission_clock import SIM_EPOCH_S, MissionClock
from hermes.mule.ferry import FerryRuntime, FerrySpec
from hermes.scheduler.plan.types import FLIGHT_SLOTS, ArrivalClass, ArrivalView
from hermes.scheduler.policies import cross_heuristic as ch
from hermes.scheduler.policies.cross_heuristic import (
    CommittedSlot,
    CrossHeuristic,
    fastest_covering_class,
    flight_slot_policy,
    moved_to_front,
    nearest_first,
)
from hermes.scheduler.stages.s3b_feasibility import (
    RULE_DEADLINE_BUDGET,
    FeasibilityModel,
    FerryPhysics,
    FlightState,
)
from hermes.types import Bucket, ContactWaypoint, DeviceID, MissionPass

COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
T0 = SIM_EPOCH_S
RF = 60.0
DOCK = (0.0, 0.0, 0.0)
BANDS = ("wide", "medium", "narrow")
REPO = Path(__file__).resolve().parents[2]


def _wp(x, y, *devs, deadline=T0 + 1e4):
    return ContactWaypoint(position=(float(x), float(y), 0.0),
                           devices=tuple(DeviceID(d) for d in devs),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=deadline)


class _Fits:
    """A ``fits`` that answers from ``rule`` and records every order it was asked about."""

    def __init__(self, rule):
        self.rule = rule
        self.calls = []

    def __call__(self, order):
        self.calls.append(tuple(order))
        return self.rule(order)


# From the pose (0, 0): b is 10 m away, c 20 m, a 30 m, d 40 m. Plan order a, b, c, d.
A, B, C, D = _wp(30, 0, "a"), _wp(0, 10, "b"), _wp(-20, 0, "c"), _wp(0, -40, "d")
REMAINDER = [A, B, C, D]
STATE = FlightState((0.0, 0.0, 0.0), T0)


def _next(remainder, fits, state=STATE):
    """FX's next stop at a Pass-1 departure after a stop, where its rule acts."""
    return CrossHeuristic().next_stop(remainder, state, fits=fits, pass_kind=COLLECT,
                                      after_stop=True)


# --------------------------------------------------------------------------- #
# The fillings
# --------------------------------------------------------------------------- #

def test_one_filling_per_flight_slot_value_named_by_it():
    assert tuple(ch._SLOTS) == FLIGHT_SLOTS[:2]
    for name in FLIGHT_SLOTS[:2]:
        slot = flight_slot_policy(name)
        assert slot.name == name
    assert isinstance(flight_slot_policy("committed"), CommittedSlot)
    assert isinstance(flight_slot_policy("cross_heuristic"), CrossHeuristic)
    for bad in ("pair_q", "", None, 1):
        with pytest.raises(ValueError, match="flight_slot"):
            flight_slot_policy(bad)


def test_the_committed_slot_is_todays_flight():
    """Arm F: index 0 (``remainder.pop(0)``) without a fold and the committed
    band without a view, in both passes, at takeoff and after a stop."""
    slot = CommittedSlot()

    def no_fold(order):
        raise AssertionError("the committed slot never folds")

    switching = _view("narrow", ("wide", "ab", 5.0), ("medium", "ab", 9.0), ("narrow", "ab", 20.0))
    for pass_kind in (COLLECT, DELIVER, "collect", "deliver"):
        for after in (False, True):
            assert slot.next_stop(REMAINDER, STATE, fits=no_fold, pass_kind=pass_kind,
                                  after_stop=after) == 0
            assert slot.next_stop(REMAINDER, None, fits=None, pass_kind=pass_kind,
                                  after_stop=after) == 0
        assert slot.reads_arrival_view(pass_kind) is False
        assert slot.band_at_arrival(None, pass_kind=pass_kind) is None
        assert slot.band_at_arrival(switching, pass_kind=pass_kind) is None
    with pytest.raises(ValueError, match="none is left"):
        slot.next_stop([], STATE, fits=no_fold, pass_kind=COLLECT, after_stop=True)


@pytest.mark.parametrize("slot", [CommittedSlot(), CrossHeuristic()], ids=repr)
def test_both_fillings_need_the_pass_and_the_departure(slot):
    """One call site serves both fillings, and neither assumes a pass: a call
    that does not name the pass, or names none, is refused, as is an
    ``after_stop`` that is not a bool (a stop or a count passed by mistake)."""
    fits = lambda order: True  # noqa: E731
    with pytest.raises(TypeError, match="pass_kind"):
        slot.next_stop(REMAINDER, STATE, fits=fits, after_stop=True)
    with pytest.raises(TypeError, match="after_stop"):
        slot.next_stop(REMAINDER, STATE, fits=fits, pass_kind=COLLECT)
    with pytest.raises(TypeError, match="pass_kind"):
        slot.band_at_arrival(None)
    with pytest.raises(TypeError, match="pass_kind"):
        slot.reads_arrival_view()
    for bad in ("pass_1", "COLLECT", None, 1):
        with pytest.raises(ValueError, match="pass_kind must name a mission pass"):
            slot.next_stop(REMAINDER, STATE, fits=fits, pass_kind=bad, after_stop=True)
        with pytest.raises(ValueError, match="pass_kind must name a mission pass"):
            slot.reads_arrival_view(bad)
        with pytest.raises(ValueError, match="pass_kind must name a mission pass"):
            slot.band_at_arrival(None, pass_kind=bad)
    for bad in (1, 0, [A], None, "yes"):
        with pytest.raises(TypeError, match="after_stop"):
            slot.next_stop(REMAINDER, STATE, fits=fits, pass_kind=COLLECT, after_stop=bad)


# --------------------------------------------------------------------------- #
# FX: the next stop
# --------------------------------------------------------------------------- #

def test_the_candidates_are_tried_nearest_first():
    assert nearest_first(REMAINDER, (0.0, 0.0, 0.0)) == [1, 2, 0, 3]
    assert nearest_first(REMAINDER, (30.0, 0.0, 0.0)) == [0, 1, 2, 3]
    assert moved_to_front(REMAINDER, 2) == [C, A, B, D]
    assert moved_to_front(REMAINDER, 0) == REMAINDER


def test_fx_flies_to_the_nearest_stop_when_the_rest_still_fits():
    fits = _Fits(lambda order: True)
    assert _next(REMAINDER, fits) == 1
    assert fits.calls == [(B, A, C, D)]                    # the candidate, then the plan's order


def test_fx_passes_over_a_nearer_stop_that_breaks_the_rest():
    fits = _Fits(lambda order: order[0] is not B)
    assert _next(REMAINDER, fits) == 2
    assert fits.calls == [(B, A, C, D), (C, A, B, D)]


def test_fx_keeps_the_plans_next_stop_when_it_is_the_nearest_that_fits():
    fits = _Fits(lambda order: order[0] in (A, D))
    assert _next(REMAINDER, fits) == 0
    assert [order[0] for order in fits.calls] == [B, C, A]


def test_fx_falls_back_to_the_plans_next_stop_when_no_move_fits():
    fits = _Fits(lambda order: False)
    assert _next(REMAINDER, fits) == 0
    assert [order[0] for order in fits.calls] == [B, C, A, D]     # every stop tried, nearest first


def test_a_lone_stop_is_flown_without_a_fold():
    fits = _Fits(lambda order: False)
    assert _next([D], fits) == 0 and fits.calls == []


def test_fx_keeps_the_plans_first_stop_at_takeoff():
    """Decision 5: "after each stop". At takeoff nothing has been observed in
    flight and the order is the plan search's own from the dock, so FX flies
    the plan's first stop without a fold, as the plan's loop does ("take off",
    then "Arrive at stop k", build plan L536-L539). After a stop it reorders."""
    fits = _Fits(lambda order: True)
    fx = CrossHeuristic()
    assert fx.next_stop(REMAINDER, STATE, fits=fits, pass_kind=COLLECT, after_stop=False) == 0
    assert fits.calls == []
    assert fx.next_stop(REMAINDER, STATE, fits=fits, pass_kind=COLLECT, after_stop=True) == 1


def test_fx_flies_pass_2_as_the_committed_slot():
    """The spec, other choices 2: "FX switches band per stop in Pass 1 only;
    Pass 2 flies b̄". Pass 2 is the delivery walk (build plan L545), outside
    the flight clock's slot (L536-L541), so FX's next stop there is the
    queue's too. The same stops and view reorder and switch in Pass 1."""
    fx = CrossHeuristic()
    switching = _view("narrow", ("wide", "ab", 5.0), ("medium", "ab", 9.0), ("narrow", "ab", 20.0))
    assert fx.reads_arrival_view(COLLECT) is True and fx.reads_arrival_view("collect") is True
    assert fx.band_at_arrival(switching, pass_kind=COLLECT) == "wide"
    assert _next(REMAINDER, lambda order: True) == 1
    for pass_kind in (DELIVER, "deliver"):
        assert fx.reads_arrival_view(pass_kind) is False
        assert fx.band_at_arrival(switching, pass_kind=pass_kind) is None
        assert fx.band_at_arrival(None, pass_kind=pass_kind) is None     # no view is built there
        fits = _Fits(lambda order: True)
        for after in (True, False):
            assert fx.next_stop(REMAINDER, STATE, fits=fits, pass_kind=pass_kind,
                                after_stop=after) == 0
        assert fits.calls == []


def test_distance_ties_fall_to_the_position_then_the_devices():
    east, north, west, south = (_wp(10, 0, "e"), _wp(0, 10, "n"), _wp(-10, 0, "w"),
                                _wp(0, -10, "s"))
    ring = [east, north, west, south]                             # all 10 m from the pose
    fits = _Fits(lambda order: True)
    assert _next(ring, fits) == 2                                 # (-10, 0) sorts first
    assert nearest_first(ring, (0.0, 0.0, 0.0)) == [2, 3, 1, 0]
    twins = [_wp(5, 5, "y"), _wp(5, 5, "x")]                      # one position, other devices
    assert _next(twins, fits) == 1
    assert nearest_first(twins, (0.0, 0.0, 0.0)) == [1, 0]
    # The order depends on the stops alone, not on where they sit in the plan.
    reverse = ring[::-1]
    assert [reverse[i] for i in nearest_first(reverse, (0.0, 0.0, 0.0))] == \
        [ring[i] for i in nearest_first(ring, (0.0, 0.0, 0.0))] == [west, south, north, east]


def test_fx_refuses_to_reorder_unchecked_or_from_nothing():
    fx = CrossHeuristic()
    for pass_kind in (COLLECT, DELIVER):            # one call site binds fits in both passes
        for after in (True, False):
            with pytest.raises(TypeError, match="fits"):
                fx.next_stop(REMAINDER, STATE, fits=None, pass_kind=pass_kind, after_stop=after)
    with pytest.raises(ValueError, match="none is left"):
        _next([], lambda order: True)
    with pytest.raises(TypeError, match="ContactWaypoint"):
        _next([A, "b"], lambda order: True)


def test_fx_next_stop_with_the_predicate_as_fits():
    """``fits`` is the fold the supervisor binds (``FLScheduler.fold_remainder``,
    which is this fold under the arm's in-flight rule). Stop a has its own
    deadline 11 s after the departure; flown first it finishes at 9 s, after
    c (20 m away) at 10 s, but after b, the nearest (10 m), only at 12.2 s.
    FX therefore passes b over and flies c."""
    model = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=1.0, ferry=FerryPhysics(
        dock=DOCK, member_dwell_s=lambda d, pass_kind, off: 1.0, upload_s=lambda: 0.0,
        p_move_w=143.6, p_hover_w=168.5))
    a = _wp(40, 0, "a", deadline=T0 + 11.0)
    b, c = _wp(0, 10, "b"), _wp(20, 0, "c")
    remainder = [a, b, c]

    def fold(order):
        return model.fold(order, STATE, rule=RULE_DEADLINE_BUDGET, budget_end=T0 + 1000.0,
                          skip=False)

    assert fold(remainder).ok                                      # the plan's own order fits
    assert fold(moved_to_front(remainder, 1)).rejected_by("overdue") == [a]
    assert fold(moved_to_front(remainder, 2)).ok
    fits = _Fits(lambda order: fold(order).ok)
    assert _next(remainder, fits) == 2
    assert [order[0] for order in fits.calls] == [b, c]
    # With the deadline too tight for any detour, the plan's next stop.
    tight = [_wp(40, 0, "a", deadline=T0 + 9.5), b, c]
    assert _next(tight, lambda order: fold(order).ok) == 0


# --------------------------------------------------------------------------- #
# FX: the band on arrival
# --------------------------------------------------------------------------- #

def _view(committed, *entries, devices=("a", "b", "c", "z")):
    """``entries``: (name, targets, dwell) per class, in class order."""
    classes = tuple(ArrivalClass(name, i, tuple(DeviceID(t) for t in targets), dwell)
                    for i, (name, targets, dwell) in enumerate(entries))
    if not classes:
        classes = (ArrivalClass("wide", 0, (), 0.0), ArrivalClass("medium", 1, (), 0.0),
                   ArrivalClass("narrow", 2, (), 0.0))
    return ArrivalView(devices=tuple(DeviceID(d) for d in devices), committed=committed,
                       classes=classes)


def _band(view):
    """FX's band at a Pass-1 arrival, where its rule acts."""
    return CrossHeuristic().band_at_arrival(view, pass_kind=COLLECT)


def test_fx_switches_to_a_faster_class_that_reaches_the_same_members():
    view = _view("narrow", ("wide", "ab", 5.0), ("medium", "ab", 9.0), ("narrow", "ab", 20.0))
    assert _band(view) == "wide" and fastest_covering_class(view).dwell_s == 5.0


def test_fx_never_adds_dwell_for_one_more_member():
    """Critic A7: the design's rule would fly narrow here, for b, at 28 s more."""
    view = _view("wide", ("wide", "a", 2.0), ("medium", "ab", 9.0), ("narrow", "ab", 30.0))
    assert _band(view) is None


def test_a_faster_class_that_misses_a_committed_member_is_not_a_candidate():
    view = _view("narrow", ("wide", "ab", 4.0), ("medium", "abc", 12.0), ("narrow", "abc", 30.0))
    assert _band(view) == "medium"


def test_nothing_is_switched_for_nothing():
    """The committed class reaches nobody (so every class qualifies): its zero
    dwell is the least, and a class that would add members adds dwell."""
    view = _view("wide", ("wide", "", 0.0), ("medium", "a", 5.0), ("narrow", "ab", 15.0))
    assert _band(view) is None
    assert _band(_view("medium")) is None                         # nobody anywhere


def test_dwell_ties_go_to_more_members_then_the_committed_class_then_the_index():
    assert _band(_view("medium", ("wide", "ab", 5.0), ("medium", "a", 5.0))) == "wide"
    assert _band(_view("medium", ("wide", "a", 5.0), ("medium", "a", 5.0))) is None
    assert _band(_view("narrow", ("wide", "a", 5.0), ("medium", "a", 5.0),
                       ("narrow", "a", 9.0))) == "wide"


def test_with_one_class_the_band_is_the_committed_one():
    assert _band(_view("wide", ("wide", "abc", 7.0))) is None


def test_the_band_rule_reads_an_arrival_view_only():
    with pytest.raises(TypeError, match="ArrivalView"):
        _band({"committed": "wide"})
    with pytest.raises(TypeError, match="ArrivalView"):
        _band(None)                                               # Pass 1 needs the view


def test_the_band_is_never_slower_and_never_reaches_fewer_on_random_views():
    rng = random.Random(5)
    switched = 0
    for _ in range(3000):
        devices = [f"d{i}" for i in range(rng.randint(1, 6))]
        classes = []
        for i in range(rng.randint(1, 4)):
            targets = tuple(d for d in devices if rng.random() < 0.6)
            dwell = 0.0 if not targets else rng.choice([rng.uniform(0.1, 60.0), 5.0, 5.0])
            classes.append(ArrivalClass(f"c{i}", i, targets, dwell))
        committed = rng.choice(classes).name
        view = ArrivalView(devices=tuple(devices), committed=committed, classes=tuple(classes))
        name = _band(view) or committed
        pick, base = view.entry(name), view.committed_entry
        assert pick.dwell_s <= base.dwell_s                           # never slower
        assert set(base.targets) <= set(pick.targets)                 # never fewer
        covering = [c for c in classes if set(base.targets) <= set(c.targets)]
        assert pick.dwell_s == min(c.dwell_s for c in covering)       # the fastest
        assert len(pick.targets) == max(len(c.targets) for c in covering
                                        if c.dwell_s == pick.dwell_s)
        if name != committed:
            switched += 1
            assert pick.dwell_s < base.dwell_s or len(pick.targets) > len(base.targets)
    assert switched > 300


# --------------------------------------------------------------------------- #
# On the runtime's own views (the mule's channel)
# --------------------------------------------------------------------------- #

def _runtime(band, clock, **kw):
    spec = FerrySpec.from_config(rf_range_m=RF, seed=11, contact_band=band, **kw)
    rt = FerryRuntime(spec, clock, rf_range_m=RF)
    rt.set_payload(theta_bytes=18_756, synth_bytes=64)
    return rt


def test_fx_at_the_last_stop_keeps_the_class_that_lands_in_budget():
    """Critic A7 at the last Pass-1 stop, where no departure check follows. At
    this arrival e, 59 m out, is below the floor on wide and above it on the
    narrower classes: the design's "most devices" rule switches for e, and its
    landing passes a budget the committed wide class meets with a second to
    spare. FX keeps wide."""
    clock = MissionClock()
    rt = _runtime("wide", clock, payload_bytes=1_000_000)
    chan, floor = rt.spec.contact_channel, rt.spec.link.snr_floor_db
    last = _wp(0, 40, "a", "e")
    pos = {DeviceID("a"): (10.0, 40.0, 0.0), DeviceID("e"): (0.0, 99.0, 0.0)}

    def e_on(t, band):
        return chan.snr_db(t, band, 59.0, link_key="e", stop_pos=last.position)

    t = next(T0 + 0.5 * k for k in range(1, 20_000)
             if e_on(T0 + 0.5 * k, "wide") < floor <= min(e_on(T0 + 0.5 * k, "medium"),
                                                         e_on(T0 + 0.5 * k, "narrow")))
    clock.advance(t - T0, "transit")
    view = rt.arrival_view(last, pos, clock(), pass_kind=COLLECT)
    assert view.committed == "wide" and view.committed_entry.targets == ("a",)
    most = max(view.classes, key=lambda c: (len(c.targets), -c.dwell_s, -c.index))
    assert most.targets == ("a", "e") and most.dwell_s > view.committed_entry.dwell_s
    fx = _band(view) or rt.band
    assert fx == "wide"
    home = rt.spec.flight.leg_s(last.position, rt.spec.flight.dock) + rt.predicted_upload_s()
    land = {c.name: clock() + c.dwell_s + home for c in view.classes}
    budget_end = land["wide"] + 1.0
    assert land[most.name] > budget_end >= land[fx]
    plan = rt.contact_plan(last, pos, pass_kind=COLLECT, mission_round=1, band=fx)
    assert plan.targets == ("a",) and plan.unreachable == ("e",)


def test_fx_takes_a_faster_class_that_reaches_the_same_members_and_lands_sooner():
    """The common switch (42 of 50 in critic probe E at 60 s): committed narrow,
    every member within wide's reach and above its floor at this arrival."""
    clock = MissionClock()
    rt = _runtime("wide", clock, payload_bytes=1_000_000)
    rt.set_band("narrow")
    chan, floor = rt.spec.contact_channel, rt.spec.link.snr_floor_db
    last = _wp(0, 40, "a", "b")
    pos = {DeviceID("a"): (10.0, 40.0, 0.0), DeviceID("b"): (0.0, 70.0, 0.0)}      # 10 m, 30 m
    t = next(T0 + 0.5 * k for k in range(1, 20_000)
             if all(chan.snr_db(T0 + 0.5 * k, "wide", d, link_key=j, stop_pos=last.position)
                    >= floor for j, d in (("a", 10.0), ("b", 30.0))))
    clock.advance(t - T0, "transit")
    view = rt.arrival_view(last, pos, clock(), pass_kind=COLLECT)
    assert view.entry("wide").targets == view.committed_entry.targets == ("a", "b")
    fx = _band(view)
    assert fx == "wide" and view.entry(fx).dwell_s < view.committed_entry.dwell_s / 2.0
    plan = rt.contact_plan(last, pos, pass_kind=COLLECT, mission_round=1, band=fx)
    assert plan.band == "wide" and plan.band_index == 0 and plan.targets == ("a", "b")
    assert rt.band == "narrow"                                   # the committed class stays b̄


def _field(rng, n_stops):
    stops, pos, k = [], {}, 0
    for _ in range(n_stops):
        c = (rng.uniform(-150.0, 150.0), rng.uniform(-150.0, 150.0), 0.0)
        members = []
        for _ in range(rng.randint(1, 5)):
            r, a = rng.uniform(0.0, 240.0), rng.uniform(0.0, 2.0 * math.pi)
            did = DeviceID(f"d{k}")
            k += 1
            pos[did] = (c[0] + r * math.cos(a), c[1] + r * math.sin(a), 0.0)
            members.append(did)
        stops.append(ContactWaypoint(position=c, devices=tuple(members),
                                     bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=T0 + 1e4))
    return stops, pos


def test_on_the_runtimes_own_views_fx_is_never_slower_and_never_reaches_fewer():
    switched = {band: 0 for band in BANDS}
    for seed in range(8):
        rng = random.Random(seed)
        clock = MissionClock()
        rt = _runtime("wide", clock, payload_bytes=rng.choice([None, 1_000_000]),
                      contact_regime=rng.choice(["clean", "jittery"]))
        stops, pos = _field(rng, 6)
        for committed in BANDS:
            rt.set_band(committed)
            for wp in stops:
                clock.advance(rng.uniform(0.5, 90.0), "transit")
                view = rt.arrival_view(wp, pos, clock(), pass_kind=COLLECT)
                name = _band(view) or rt.band
                pick, base = view.entry(name), view.committed_entry
                assert pick.dwell_s <= base.dwell_s
                assert set(base.targets) <= set(pick.targets)
                flown = rt.contact_plan(wp, pos, pass_kind=COLLECT, mission_round=1, band=name)
                committed_plan = rt.contact_plan(wp, pos, pass_kind=COLLECT, mission_round=1)
                assert flown.band == name and committed_plan.band == committed
                assert set(committed_plan.targets) <= set(flown.targets) == set(pick.targets)
                switched[committed] += name != committed
    # Not vacuous: FX leaves each committed class somewhere on these layouts.
    assert all(n > 0 for n in switched.values()), switched


# --------------------------------------------------------------------------- #
# Through the supervisor's seam (the hand-off to unit U7)
# --------------------------------------------------------------------------- #

def _fly(slot, rt, clock, queue, pos, pass_kind, fits):
    """One pass through the seam as the hand-off to U7 states it.

    ``mule_main._ferry_fly_pass`` pops the slot's index at every departure,
    takeoff included (``after_stop``: a stop has been flown this pass), and
    ``_ferry_stop`` builds the contact plan after the transit on the slot's
    band, from a view built only when the slot reads one. The contact is
    charged at the flown class's arrival dwell. Returns (stop, plan) pairs in
    the order flown.
    """
    remainder, flown = list(queue), []
    pose = rt.spec.flight.dock
    nbytes = rt.session_bytes(pass_kind)
    while remainder:
        state = FlightState(tuple(pose), clock())
        index = slot.next_stop(remainder, state, fits=fits, pass_kind=pass_kind,
                               after_stop=bool(flown))
        wp = remainder.pop(index)
        clock.advance(rt.spec.flight.leg_s(pose, wp.position), "transit")
        pose = wp.position
        view = (rt.arrival_view(wp, pos, clock(), pass_kind=pass_kind)
                if slot.reads_arrival_view(pass_kind) else None)
        plan = rt.contact_plan(wp, pos, pass_kind=pass_kind, mission_round=1,
                               band=slot.band_at_arrival(view, pass_kind=pass_kind))
        clock.advance(sum(plan.dwell_s(nbytes, plan.snr_db[j]) for j in plan.targets), "dwell")
        flown.append((wp, plan))
    return flown


def test_through_the_seam_pass_2_flies_the_committed_class_in_the_queues_order():
    """The review's probe as a test. It fed Pass-2 stops to the hand-off's
    band expression and flew another class than b̄ at 89 of 360. Called as the
    seam calls it, at every stop of both passes, FX flies Pass 2 on b̄ at every
    stop and in the queue's order, as F does; in Pass 1 it starts at the
    plan's first stop, then reorders and switches class on these layouts (so
    the Pass-2 result is not vacuous)."""
    fx, fits = CrossHeuristic(), (lambda order: True)       # no budget: every order fits
    reordered = switched = pass_2_stops = 0
    for seed in range(8):
        rng = random.Random(seed)
        clock = MissionClock()
        rt = _runtime("wide", clock, payload_bytes=rng.choice([None, 1_000_000]),
                      contact_regime=rng.choice(["clean", "jittery"]))
        stops, pos = _field(rng, 5)
        for committed in BANDS:
            rt.set_band(committed)
            pass_1 = _fly(fx, rt, clock, stops, pos, COLLECT, fits)
            assert pass_1[0][0] is stops[0]                             # takeoff: the plan's first
            reordered += [wp for wp, _ in pass_1] != stops
            switched += sum(plan.band != committed for _, plan in pass_1)
            pass_2 = _fly(fx, rt, clock, stops, pos, DELIVER, fits)
            assert [wp for wp, _ in pass_2] == stops                    # the queue's order
            assert [plan.band for _, plan in pass_2] == [committed] * len(stops)
            pass_2_stops += len(pass_2)
            for pass_kind in (COLLECT, DELIVER):                        # F through the same seam
                flown = _fly(CommittedSlot(), rt, clock, stops, pos, pass_kind, fits)
                assert [wp for wp, _ in flown] == stops
                assert {plan.band for _, plan in flown} == {committed}
            assert rt.band == committed
    assert pass_2_stops == 8 * 3 * 5
    assert reordered > 0 and switched > 0, (reordered, switched)


# --------------------------------------------------------------------------- #
# Layering
# --------------------------------------------------------------------------- #

def test_the_policy_imports_only_the_plan_types_and_the_waypoint():
    """Numpy-free and nothing from hermes.l1, the mule, the mission package or
    experiments: physics reaches it only as the arrival view's numbers."""
    tree = ast.parse((REPO / "hermes/scheduler/policies/cross_heuristic.py").read_text(
        encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append("." * node.level + (node.module or ""))
    assert sorted(set(names)) == ["__future__", "hermes.scheduler.plan.types",
                                  "hermes.types.scheduler", "typing"]
