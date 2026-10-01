"""FeRRy Phase 4: a capped device's best hover point (``hermes/scheduler/plan/hover.py``).

The user's decision of 2026-09-30, "Fix: best hover point", on the final
check's PLAN-1 and E2E2-01: in plan mode a capped device its S3a stop cannot
serve alone within the budget leaves that stop for a stop of its own at its
best hover point, and ``unplannable`` means no class the arm may fly serves it
alone even there. Pinned here:

* **The point.** It minimises the device's alone mission (the flight out, the
  dwell at the class's predicted rate, the flight back, the upload) over the
  segment from the dock to the device, within the class's reach and above the
  floor: at the edge of the reach when the dwell is flat there, at the dock
  when the dock is within reach, at the device when the dwell grows faster
  than the flight shrinks, at the edge of a rate step, short of the floor;
  none when nothing reaches the device. On the contact link (the critic's
  reference layouts, every class) it is deterministic, within reach, priced
  as the model prices the stop, never worse than the device's S3a stop or
  than any point of a brute-force grid of the segment or of the plane, and
  equal to the segment grid's minimum to the grid's resolution; and it is
  the segment's exact minimum, found from the dwell's step edges, to 1e-6 s
  at 1 MB, 8 MB and a 100 m range, which pins the refinement's constants.
* **The offered stops** (``offer_hover_stops``): only capped devices their
  S3a stop cannot serve alone move; the stop they leave keeps its position
  and takes its remaining members' bucket and deadline (B2); every other stop
  is the very object S3a made; each device stays in one stop; without a
  budget nothing moves.
* **The labels** (the verifier's label test): a capped far singleton over the
  budget at its own position (47.4 s at 45 s, critic layout 0's d5 on medium)
  that fits at its hover point (13.25 s, the dock) is served from it when
  nothing outranks it, and is ``crowded``, never ``unplannable``, when an
  older capped device takes the budget; its drop is labelled at its hover
  stop. Under an energy capacity the label reads the time-minimising point:
  a device refused there for the energy is ``unplannable`` even where a
  slower point with a shorter dwell would serve it within both, so the label
  is physics for the time budget only (pinned as the docstrings state it).
* **Layering.** Numpy-free; nothing from ``hermes.l1``, the mule or
  ``experiments``.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from experiments.exp4.model_task import _u32
from experiments.exp4.topology_builder import device_positions, device_spread_m
from hermes.mission.contact_plan import planar_distance_m
from hermes.mule.ferry import FerryRuntime, FerrySpec
from hermes.scheduler import FLScheduler
from hermes.scheduler.plan import AgeCapSpec, PlanClass, PlanOptions, PlanSetup
from hermes.scheduler.plan import hover as HV
from hermes.scheduler.stages.s3a_cluster import cluster_by_rf_range
from hermes.scheduler.stages.s3b_feasibility import (
    RULE_DEADLINE_BUDGET,
    RULE_NONE,
    FeasibilityModel,
    FerryPhysics,
    FlightState,
)
from hermes.scheduler.stages.s3d_age_cap import servable_alone
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionPass,
    MissionSlice,
    MuleID,
)
from hermes.types.scheduler import CAP_CROWDED, CAP_UNPLANNABLE, CapViolation

REPO = Path(__file__).resolve().parents[2]
DOCK = (0.0, 0.0, 0.0)
START = FlightState(DOCK, 0.0)
SPEED = 5.0
UPLOAD = 0.5
BANDS = ("wide", "medium", "narrow")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

class _Dwell:
    """A member's dwell as a function of its distance, ``fn(d)`` seconds (None:
    below the floor)."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, d, pass_kind, offset):
        return self.fn(d)


def _states(positions, *, buckets=None):
    states = {}
    for did, pos in positions.items():
        st = DeviceSchedulerState(device_id=DeviceID(did))
        st.last_known_position = pos
        st.bucket = (buckets or {}).get(did, Bucket.SCHEDULED_THIS_ROUND)
        states[DeviceID(did)] = st
    return states


def _model(fn, *, reach, states):
    """One synthetic class: speed 5 m/s, a 0.5 s upload, the dwell ``fn``."""
    physics = FerryPhysics(dock=DOCK, member_dwell_s=_Dwell(fn), upload_s=lambda: UPLOAD,
                           p_move_w=143.6, p_hover_w=168.5, range_m=reach,
                           device_states=states)
    return FeasibilityModel(cruise_speed_m_s=SPEED, session_time_s=1.0, ferry=physics)


def _alone(model, position, did, start=START):
    """``did`` alone at ``position`` priced by the model (no gate): the verdict."""
    stop = ContactWaypoint(position=tuple(position), devices=(DeviceID(did),),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=math.inf)
    return model.admit(start, stop, rule=RULE_NONE, budget_end=None)


def _reaches(model, position, did, reach):
    stop = ContactWaypoint(position=tuple(position), devices=(DeviceID(did),),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=math.inf)
    (dist,) = model.ferry.member_distances_m(stop)
    return dist <= reach and model.ferry.member_dwell_s(dist, MissionPass.COLLECT, 0.0) is not None


def _point(where, r):
    """The point ``r`` metres from ``where`` towards the dock."""
    length = math.dist(where, DOCK)
    t = r / length if length else 0.0
    return tuple(a + t * (b - a) for a, b in zip(where, DOCK))


# --------------------------------------------------------------------------- #
# The point
# --------------------------------------------------------------------------- #

def test_a_far_device_hovers_at_the_edge_of_its_reach_when_the_dwell_is_flat():
    """100 m out, 60 m of reach, a flat dwell: every metre nearer the dock
    saves 0.4 s of flight for no dwell, so the point is the edge of the reach,
    40 m from the dock: 16 s of flight, 2 s of dwell, 0.5 s of upload."""
    model = _model(lambda d: 2.0, reach=60.0, states=_states({"a": (100.0, 0.0, 0.0)}))
    hp = HV.best_hover_point(model, DeviceID("a"), reach_m=60.0)
    assert hp.position == pytest.approx((40.0, 0.0, 0.0), abs=1e-9)
    assert hp.distance_m == pytest.approx(60.0) and hp.distance_m <= 60.0
    assert hp.home_s == pytest.approx(18.5)


def test_a_device_within_reach_of_the_dock_is_served_from_the_dock():
    model = _model(lambda d: 2.0, reach=60.0, states=_states({"a": (30.0, 40.0, 0.0)}))
    hp = HV.best_hover_point(model, DeviceID("a"), reach_m=60.0)
    assert hp.position == DOCK and hp.distance_m == pytest.approx(50.0)
    assert hp.home_s == pytest.approx(2.5)


def test_a_dwell_that_grows_faster_than_the_flight_shrinks_keeps_the_device_position():
    """1 s of dwell per metre against 0.4 s of flight saved: hovering over the
    device is best."""
    model = _model(lambda d: 1.0 + d, reach=60.0, states=_states({"a": (0.0, 100.0, 0.0)}))
    hp = HV.best_hover_point(model, DeviceID("a"), reach_m=60.0)
    assert hp.position == (0.0, 100.0, 0.0) and hp.distance_m == 0.0
    assert hp.home_s == pytest.approx(41.5)


def test_a_rate_step_puts_the_point_at_the_steps_edge():
    """A dwell of 1 s up to 30 m and 20 s beyond: the alone mission falls
    along the step and jumps at its edge, so the point is the edge (inclusive,
    as the link's CQI thresholds are), 70 m from the dock: 28 + 1 + 0.5 s."""
    model = _model(lambda d: 1.0 if d <= 30.0 else 20.0, reach=60.0,
                   states=_states({"a": (100.0, 0.0, 0.0)}))
    hp = HV.best_hover_point(model, DeviceID("a"), reach_m=60.0)
    assert hp.distance_m <= 30.0 and hp.distance_m == pytest.approx(30.0, abs=1e-6)
    assert hp.home_s == pytest.approx(29.5, abs=1e-6)


def test_the_floor_before_the_reach_bounds_the_point():
    """Below the floor beyond 40 m: the model would charge such a point no
    dwell (critic B12), so the point stops at the floor's edge."""
    model = _model(lambda d: 2.0 if d <= 40.0 else None, reach=60.0,
                   states=_states({"a": (100.0, 0.0, 0.0)}))
    hp = HV.best_hover_point(model, DeviceID("a"), reach_m=60.0)
    assert hp.distance_m <= 40.0 and hp.distance_m == pytest.approx(40.0, abs=1e-9)
    assert hp.home_s == pytest.approx(26.5, abs=1e-9)
    assert _reaches(model, hp.position, "a", 60.0)


def test_no_point_reaches_a_device_below_the_floor_everywhere():
    model = _model(lambda d: None, reach=60.0, states=_states({"a": (100.0, 0.0, 0.0)}))
    assert HV.best_hover_point(model, DeviceID("a"), reach_m=60.0) is None


def test_a_device_at_the_dock_is_served_there():
    model = _model(lambda d: 2.0 + d, reach=60.0, states=_states({"a": DOCK}))
    hp = HV.best_hover_point(model, DeviceID("a"), reach_m=60.0)
    assert (hp.position, hp.distance_m, hp.home_s) == (DOCK, 0.0, 2.5)


def test_the_point_needs_a_bound_model_with_a_member_dwell_and_a_reach():
    states = _states({"a": (100.0, 0.0, 0.0)})
    bound = _model(lambda d: 2.0, reach=60.0, states=states)
    with pytest.raises(ValueError, match="bind the model"):
        HV.best_hover_point(dataclasses.replace(bound, ferry=dataclasses.replace(
            bound.ferry, device_states=None)), DeviceID("a"), reach_m=60.0)
    with pytest.raises(ValueError, match="member dwell"):
        HV.best_hover_point(FeasibilityModel(), DeviceID("a"), reach_m=60.0)
    for reach in (0.0, -5.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="reach_m"):
            HV.best_hover_point(bound, DeviceID("a"), reach_m=reach)


#: The critic's reference layouts (U7's): the far devices of PLAN-1 (0, 4,
#: 26) and the empty missions' layout (13).
CRITIC = (0, 4, 13, 26)


def _critic_world(k, *, payload_bytes=1_000_000, rf_range_m=60.0):
    """Critic layout ``k`` on the pilots' physics (1 MB, jittery backhaul, a 60 m
    range), or with another payload or range: its device positions and one
    model per class, bound to them."""
    seed = _u32(6, "t_nom", 1000 + k)
    xy = device_positions(6, seed, device_spread_m(60.0, field_radius_m=100.0))
    positions = {f"d{i}": (x, y, 0.0) for i, (x, y) in enumerate(xy)}
    spec = FerrySpec.from_config(rf_range_m=rf_range_m, seed=seed, contact_band="wide",
                                 payload_bytes=payload_bytes, backhaul_regime="jittery")
    rt = FerryRuntime(spec, None, rf_range_m=rf_range_m)
    rt.set_payload(theta_bytes=52)
    states = _states(positions, buckets={d: Bucket.NEW for d in positions})
    classes = [(c, dataclasses.replace(c.model, ferry=c.model.ferry.bind(states)))
               for c in rt.plan_classes()]
    return positions, states, classes


@pytest.mark.parametrize("k", CRITIC)
def test_on_the_contact_link_the_point_is_the_brute_force_minimum(k):
    """Test (d): every device on every class of the critic's layout ``k``. The
    point is deterministic (a second call and a rebuilt model give the same),
    within the class's reach and above the floor, priced as the model prices
    that stop, never worse than the device's fresh S3a stop alone, nor than
    any point of a 2,001-point grid of the segment or of a planar grid within
    reach (5 m on wide and medium, 10 m on narrow), and the segment grid's
    best is never worse than it by more than the grid's step allows."""
    positions, states, classes = _critic_world(k)
    deadlines = {DeviceID(d): 1e9 for d in positions}
    for c, model in classes:
        reach = c.radius_m
        stops = cluster_by_rf_range(list(states), states, deadlines, reach)
        own = {d: wp for wp in stops for d in wp.devices}
        for d, where in positions.items():
            hp = HV.best_hover_point(model, DeviceID(d), reach_m=reach)
            assert hp == HV.best_hover_point(model, DeviceID(d), reach_m=reach)
            rebuilt = dataclasses.replace(model, ferry=dataclasses.replace(model.ferry))
            assert hp == HV.best_hover_point(rebuilt, DeviceID(d), reach_m=reach)
            assert hp.distance_m <= reach and _reaches(model, hp.position, d, reach)
            # ... and the mule solicits it there: the contact gate's metric.
            assert planar_distance_m(hp.position, where) <= reach
            assert _alone(model, hp.position, d).home == pytest.approx(hp.home_s, abs=1e-9)
            s3a = _alone(model, own[DeviceID(d)].position, d).home
            assert hp.home_s <= s3a + 1e-9, (c.name, d)
            length = math.dist(where, DOCK)
            span = min(length, reach)
            segment = min(_alone(model, p, d).home for p in
                          (_point(where, span * i / 2000) for i in range(2001))
                          if _reaches(model, p, d, reach))
            assert hp.home_s <= segment + 1e-9
            assert segment - hp.home_s <= 2.0 * (span / 2000) / SPEED + 1e-9
            step = 10.0 if c.name == "narrow" else 5.0
            n = int(reach / step)
            plane = min(
                _alone(model, p, d).home
                for p in ((where[0] + i * step, where[1] + j * step, 0.0)
                          for i in range(-n, n + 1) for j in range(-n, n + 1))
                if _reaches(model, p, d, reach))
            assert hp.home_s <= plane + 1e-9, (c.name, d)


def _dwell_edges(ferry, reach):
    """Every edge of a member's predicted dwell on [0, ``reach``]: the largest
    distance still at each lower value, to adjacent floats, the floor's
    included (the dwell then becomes ``inf``).

    The contact link's dwell never falls with the distance (the mean SNR
    falls with it, the CQI table is a step function of the SNR and the rate
    rises with the CQI), so it is constant on any interval whose ends agree,
    and halving the intervals whose ends differ finds every edge. A dwell that
    falls, or one without steps (an edge at every float), is refused.
    """
    def dwell(d):
        t = ferry.member_dwell_s(d, MissionPass.COLLECT, 0.0)
        return math.inf if t is None or not math.isfinite(float(t)) else float(t)

    sampled = [dwell(reach * i / 2000) for i in range(2001)]
    assert sampled == sorted(sampled), "the dwell falls with the distance"
    edges, todo = [], [(0.0, float(reach), dwell(0.0), dwell(float(reach)))]
    while todo:
        lo, hi, at_lo, at_hi = todo.pop()
        if at_lo == at_hi:
            continue
        assert at_lo < at_hi, (lo, hi)
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            edges.append(lo)
            assert len(edges) <= 64, "the dwell is no step function of the distance"
            continue
        at_mid = dwell(mid)
        todo += [(mid, hi, at_mid, at_hi), (lo, mid, at_lo, at_mid)]
    return sorted(edges)


def _serves(model, position, did, where, reach):
    """True when a stop at ``position`` reaches ``did``: within the reach by the
    model's metric and the contact gate's, with a finite predicted dwell."""
    stop = ContactWaypoint(position=tuple(position), devices=(DeviceID(did),),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=math.inf)
    (dist,) = model.ferry.member_distances_m(stop)
    t = model.ferry.member_dwell_s(dist, MissionPass.COLLECT, 0.0)
    return (dist <= reach and planar_distance_m(position, where) <= reach
            and t is not None and math.isfinite(float(t)))


def _exact_alone_min(model, did, where, reach, edges):
    """``did``'s least alone mission over the segment, from the dwell's edges.

    Within a step of the dwell the alone mission falls linearly towards the
    dock (the flight shrinks, the dwell holds), so its minimum over the
    distances that reach the device is at a step's edge or at the farthest
    of them, min(|device - dock|, reach). Each candidate is also nudged down
    by small relative amounts: the point interpolated at a distance r can lie
    a few ulps of its coordinates beyond r, on the dearer side of an edge.
    """
    far = min(math.dist(where, DOCK), reach)
    best = None
    for r in [0.0, far] + [e for e in edges if e <= far]:
        for rel in (0.0, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9):
            p = _point(where, r * (1.0 - rel))
            if _serves(model, p, did, where, reach):
                home = _alone(model, p, did).home
                best = home if best is None else min(best, home)
    return best


#: The exact test's physics: the pilots' (1 MB, 60 m), 8 MB, whose longer
#: dwell steps outweigh more flight, and a 100 m range at 2 MB, with more
#: steps within reach.
EXACT_PHYSICS = {
    "1MB": dict(payload_bytes=1_000_000),
    "8MB": dict(payload_bytes=8_000_000),
    "rf100_2MB": dict(payload_bytes=2_000_000, rf_range_m=100.0),
}


@pytest.mark.parametrize("physics", sorted(EXACT_PHYSICS))
def test_on_the_contact_link_the_point_is_the_exact_minimum(physics):
    """The point is the segment's exact minimum to 1e-6 s, on every device of
    the critic's layouts and every class, the minimum found from the dwell's
    step edges (:func:`_dwell_edges`), not from a grid. The 2,001-point grid
    above pins the point only to its step, tens of milliseconds, so a
    refinement that halves one cell a round (0.377 s off on narrow at 8 MB,
    critic layouts 0 and 26), stops after eight rounds (up to 5 ms off) or
    at a 1 ms tolerance (up to 1 ms off) passed it; each fails here. The
    module documents the minimum to its 1e-9 s tolerance."""
    for k in CRITIC:
        positions, _, classes = _critic_world(k, **EXACT_PHYSICS[physics])
        for c, model in classes:
            edges = _dwell_edges(model.ferry, c.radius_m)
            assert edges, c.name
            for d, where in positions.items():
                hp = HV.best_hover_point(model, DeviceID(d), reach_m=c.radius_m)
                exact = _exact_alone_min(model, d, where, c.radius_m, edges)
                assert abs(hp.home_s - exact) <= 1e-6, (k, c.name, d, hp.home_s, exact)


def test_the_reach_is_held_to_the_contact_gates_metric_too():
    """The model measures a member's distance with ``** 0.5`` and the contact
    gate (S3a's metric) with ``math.sqrt``; where the platform's ``pow`` is not
    correctly rounded (the Windows runtime here: about 1 in 2,000 values) they
    differ in the last bit, so a point at the edge of the reach that only the
    model's metric admits would not solicit the device on arrival. The hover
    module's gate distance is the gate's own, bit for bit."""
    import random

    rng = random.Random(4)
    for _ in range(20000):
        a = tuple(rng.uniform(-300.0, 300.0) for _ in range(2)) + (0.0,)
        b = tuple(rng.uniform(-300.0, 300.0) for _ in range(2)) + (0.0,)
        assert HV._gate_distance(a, b) == planar_distance_m(a, b)


def test_the_point_does_not_depend_on_the_clock_the_budget_or_the_other_devices():
    """The hover point is a property of the device, the dock and the class:
    the same with the other devices gone or moved."""
    positions, states, classes = _critic_world(0)
    c, model = classes[1]
    alone_states = {DeviceID("d5"): states[DeviceID("d5")]}
    alone_model = dataclasses.replace(model, ferry=model.ferry.bind(alone_states))
    assert (HV.best_hover_point(model, DeviceID("d5"), reach_m=c.radius_m)
            == HV.best_hover_point(alone_model, DeviceID("d5"), reach_m=c.radius_m))


# --------------------------------------------------------------------------- #
# The offered stops
# --------------------------------------------------------------------------- #

#: One class (radius 10 m, a flat 2 s dwell), budget 40 s. ``a`` (capped) is
#: alone 100 m out (42.5 s there, 38.5 s at its hover point); ``b`` (capped)
#: shares a stop with ``c`` near the dock and fits; ``d`` (uncapped) is alone
#: 100 m out and over the budget; ``e`` (capped, the earliest deadline, so the
#: anchor) shares a stop 97.5 m out with ``f`` (uncapped).
OFFER = {"a": (100.0, 0.0, 0.0), "b": (20.0, 0.0, 0.0), "c": (25.0, 0.0, 0.0),
         "d": (0.0, 100.0, 0.0), "e": (0.0, -100.0, 0.0), "f": (0.0, -95.0, 0.0)}
OFFER_CAPPED = frozenset(DeviceID(d) for d in "abe")


def _offer_world():
    buckets = {"e": Bucket.NEW, "f": Bucket.SCHEDULED_THIS_ROUND}
    states = _states(OFFER, buckets=buckets)
    deadlines = {DeviceID(d): (10.0 if d == "e" else 500.0 + i) for i, d in enumerate(OFFER)}
    model = _model(lambda d: 2.0, reach=10.0, states=states)
    stops = tuple(cluster_by_rf_range(list(states), states, deadlines, 10.0))
    return states, deadlines, model, stops


def test_only_capped_devices_their_s3a_stop_cannot_serve_alone_move():
    states, deadlines, model, stops = _offer_world()
    offered = HV.offer_hover_stops(stops, model=model, reach_m=10.0, movable=OFFER_CAPPED,
                                   start=START, budget_end=40.0, deadlines=deadlines,
                                   device_states=states, capped=OFFER_CAPPED)
    members = [d for wp in offered for d in wp.devices]
    assert sorted(members) == sorted(DeviceID(d) for d in OFFER)       # each device once
    by = {wp.devices: wp for wp in offered}
    s3a = {wp.devices: wp for wp in stops}
    # b fits alone at its stop, and d is not capped: their stops are S3a's own.
    assert by[("b", "c")] is s3a[("b", "c")] and by[("d",)] is s3a[("d",)]
    # a leaves its singleton for its hover point, 10 m nearer the dock.
    assert s3a[("a",)].position == (100.0, 0.0, 0.0)
    assert by[("a",)].position == pytest.approx((90.0, 0.0, 0.0))
    assert by[("a",)].deadline_ts == deadlines["a"]
    assert by[("a",)].bucket is Bucket.SCHEDULED_THIS_ROUND
    # e leaves the stop it anchored: f keeps its position, with its own
    # deadline and bucket (B2: no capped member left), and e follows it.
    (ef,) = [wp for wp in stops if "e" in wp.devices]
    assert ef.devices == ("e", "f") and ef.position == (0.0, -97.5, 0.0)
    assert by[("f",)].position == ef.position
    assert (by[("f",)].deadline_ts, by[("f",)].bucket) == (deadlines["f"],
                                                          Bucket.SCHEDULED_THIS_ROUND)
    assert by[("e",)].position == pytest.approx((0.0, -90.0, 0.0))
    assert (by[("e",)].deadline_ts, by[("e",)].bucket) == (deadlines["e"], Bucket.NEW)
    order = [wp.devices for wp in offered]
    assert order.index(("e",)) == order.index(("f",)) + 1
    # The capped devices fit alone at the stops offered; d, over the budget
    # and not capped, is not asked about.
    assert servable_alone(sorted(OFFER_CAPPED), [(model, offered)], start=START,
                          budget_end=40.0) == OFFER_CAPPED
    assert servable_alone(sorted(OFFER_CAPPED), [(model, stops)], start=START,
                          budget_end=40.0) == {"b"}


def test_without_a_budget_or_a_capped_device_the_stops_are_s3as():
    states, deadlines, model, stops = _offer_world()
    common = dict(model=model, reach_m=10.0, start=START, deadlines=deadlines,
                  device_states=states)
    same = HV.offer_hover_stops(stops, movable=OFFER_CAPPED, budget_end=None,
                                capped=OFFER_CAPPED, **common)
    assert same == stops and all(a is b for a, b in zip(same, stops))
    none = HV.offer_hover_stops(stops, movable=frozenset(), budget_end=40.0, **common)
    assert all(a is b for a, b in zip(none, stops)) and len(none) == len(stops)


def test_a_device_no_point_reaches_keeps_its_s3a_stop():
    states = _states({"a": (100.0, 0.0, 0.0)})
    model = _model(lambda d: None, reach=10.0, states=states)
    stops = tuple(cluster_by_rf_range(list(states), states, {DeviceID("a"): 1.0}, 10.0))
    got = HV.offer_hover_stops(stops, model=model, reach_m=10.0, movable={DeviceID("a")},
                               start=START, budget_end=1.0, deadlines={DeviceID("a"): 1.0},
                               device_states=states)
    assert got == stops


# --------------------------------------------------------------------------- #
# The labels (the verifier's label test)
# --------------------------------------------------------------------------- #

#: Critic layout 0's d5 (113 m out), and ``k``, 94.5 m out the other way, 207
#: m from it: each its own stop on medium at its own position.
D5 = device_positions(6, _u32(6, "t_nom", 1000), device_spread_m(60.0, field_radius_m=100.0))[5]
LABEL_WORLD = {"d5": (D5[0], D5[1], 0.0), "k": (-60.0, -73.0, 0.0)}
NOW = 1000.0


def _medium_plan(merged, *, mission_round):
    """FB+medium at 45 s on the pilots' physics (1 MB), S = 3: the plan-mode
    scheduler at the dock, every deadline out of the way."""
    spec = FerrySpec.from_config(rf_range_m=60.0, seed=_u32(6, "t_nom", 1000),
                                 contact_band="medium", payload_bytes=1_000_000,
                                 backhaul_regime="jittery")
    rt = FerryRuntime(spec, None, rf_range_m=60.0)
    rt.set_payload(theta_bytes=52)
    setup = PlanSetup(options=PlanOptions(band_class_policy="fixed:medium",
                                          cap=AgeCapSpec(s_missions=3)),
                      classes=rt.plan_classes(), reference="medium", t_ref_s=200.0,
                      turnaround_s=rt.spec.flight.turnaround_s)
    sch = FLScheduler(now_fn=lambda: NOW, mission_budget_s=45.0, replan_fallback="trim",
                      member_admission="subset", plan_mode="ferry", plan=setup,
                      feasibility_model=rt.feasibility_model())
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"),
                                  device_ids=tuple(map(DeviceID, LABEL_WORLD)),
                                  issued_round=0, issued_at=NOW))
    for did, pos in LABEL_WORLD.items():
        st = sch.device_states[DeviceID(did)]
        st.last_known_position = pos
        st.is_new = False
        st.deadline_fulfilment_s = 1e6
        st.last_merged_round = merged.get(did)
    sch.start_mission()
    sch.set_mission_round(mission_round)
    return sch


def _medium(sch):
    cls = sch.plan_setup.class_named("medium")
    start = FlightState(DOCK, NOW)
    stops = cluster_by_rf_range(list(map(DeviceID, LABEL_WORLD)), sch.device_states,
                                sch.last_plan_deadlines, cls.radius_m)
    return cls, start, stops


def test_a_capped_far_singleton_is_served_from_its_hover_point():
    """d5, capped (age 3), is a stop of its own at its own position, 47.4 s
    alone against 45 s: no plan of S3a's stops serves it, and before the fix
    it was ``unplannable``. Its best hover point is the dock (within medium's
    119.5 m of it), 13.25 s alone, and nothing outranks it (``k`` is not
    capped), so the plan serves it there; ``k``, which fits alone but not
    beside it, is left out by the plan's choice."""
    sch = _medium_plan({"k": 2}, mission_round=3)
    route = sch.build_ferry_plan()
    cls, start, stops = _medium(sch)
    (own,) = [wp for wp in stops if "d5" in wp.devices]
    assert own.devices == ("d5",) and own.position == LABEL_WORLD["d5"]
    assert _alone(cls.model, own.position, "d5", start).home - NOW == pytest.approx(47.40,
                                                                                  abs=0.01)
    commit = sch.last_plan
    assert commit.capped == {"d5"} and commit.violations == ()
    assert [(wp.devices, wp.position) for wp in route] == [(("d5",), DOCK)]
    assert _alone(cls.model, DOCK, "d5", start).home - NOW == pytest.approx(13.25, abs=0.01)
    assert [wp.devices for wp in sch.last_feasibility.dropped_plan] == [("k",)]


def test_a_capped_far_singleton_an_older_device_outranks_is_crowded_not_unplannable():
    """The same d5 (age 3) when ``k``, older (age 4), takes the budget: the
    two do not fit together (52.8 s), so the plan serves ``k`` at its own
    stop and d5 is ``crowded``: a hover point serves it alone, so the miss is
    the plan's, not physics. Its drop is labelled at its hover stop, the
    dock: ``plan``, it fits alone there."""
    sch = _medium_plan({"d5": 1}, mission_round=4)
    route = sch.build_ferry_plan()
    commit = sch.last_plan
    assert commit.capped == {"d5", "k"} and [wp.devices for wp in route] == [("k",)]
    assert commit.violations == (CapViolation(DeviceID("d5"), 3, CAP_CROWDED),)
    (drop,) = sch.last_feasibility.dropped_plan
    assert (drop.devices, drop.position) == (("d5",), DOCK)
    assert sch.last_feasibility.n_dropped == 1
    cls, start, _ = _medium(sch)
    v = cls.model.admit(start, drop, rule=RULE_DEADLINE_BUDGET, budget_end=commit.budget_end,
                        protected=True)
    assert v.ok


#: One class, reach 60 m, a dwell of 1 s up to 30 m and 12 s beyond, and the
#: pilots' powers (143.6 W flying, 168.5 W hovering); ``a`` is 100 m out.
#: Alone, ``a`` is 41.5 s at its own position, 28.5 s at the edge of the reach
#: (40 m out: 16 s of flight, 12 s of dwell), its best hover point, and 29.5 s
#: at the step's edge (70 m out: 28 s of flight, 1 s of dwell). The energy
#: clause's need, P_move (out and back) + P_hover x dwell, is 4,319.6 J at
#: the hover point and 4,189.3 J at the step's edge: hovering costs more.
STEP_WHERE = (100.0, 0.0, 0.0)


def _step_plan(capacity):
    """A plan-mode scheduler on ``a`` alone, capped (S = 1, mission 1), 40 s."""
    physics = FerryPhysics(dock=DOCK, member_dwell_s=_Dwell(lambda d: 1.0 if d <= 30.0 else 12.0),
                           upload_s=lambda: UPLOAD, p_move_w=143.6, p_hover_w=168.5,
                           energy_capacity_j=capacity, range_m=60.0)
    cls = PlanClass(name="wide", index=0, radius_m=60.0, outage=lambda d: 0.0,
                    model=FeasibilityModel(cruise_speed_m_s=SPEED, session_time_s=1.0,
                                           ferry=physics))
    setup = PlanSetup(options=PlanOptions(cap=AgeCapSpec(s_missions=1)), classes=(cls,),
                      reference="wide", t_ref_s=300.0, turnaround_s=30.0)
    sch = FLScheduler(now_fn=lambda: NOW, mission_budget_s=40.0, replan_fallback="trim",
                      member_admission="subset", plan_mode="ferry", plan=setup)
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=(DeviceID("a"),),
                                  issued_round=0, issued_at=NOW))
    st = sch.device_states[DeviceID("a")]
    st.last_known_position = STEP_WHERE
    st.is_new = False
    st.deadline_fulfilment_s = 1e6
    sch.start_mission()
    sch.set_mission_round(1)
    return sch


def test_under_an_energy_capacity_the_label_reads_the_time_minimising_point():
    """The hover point minimises the alone mission's time (the user's decision
    of 2026-09-30), and the energy clause's need weighs the dwell more than
    the flight, so a slower point with a shorter dwell can need less energy.
    Under a capacity between the two needs above (4,250 J) the plan offers
    ``a`` its hover point, the energy clause refuses it there, and the cap
    labels ``a`` ``unplannable``, its drop ``energy`` at that point, though
    the step's edge serves it alone within the budget and the capacity. So
    under a capacity ``unplannable`` is physics for the time budget only, as
    the docstrings say (the review of the hover fix, finding 1); a rule that
    weighed the energy is the user's call, and would change this test.
    Without the capacity the plan serves ``a`` from its hover point."""
    sch = _step_plan(4250.0)
    assert sch.build_ferry_plan() == []
    assert sch.last_plan.violations == (CapViolation(DeviceID("a"), 1, CAP_UNPLANNABLE),)
    (drop,) = sch.last_feasibility.dropped_energy
    assert drop.devices == ("a",) and drop.position == pytest.approx((40.0, 0.0, 0.0))
    model = sch.plan_setup.class_named("wide").model
    start, end = FlightState(DOCK, NOW), sch.last_plan.budget_end

    def verdict(position):
        stop = ContactWaypoint(position=position, devices=(DeviceID("a"),),
                               bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=math.inf)
        return model.admit(start, stop, rule=RULE_DEADLINE_BUDGET, budget_end=end,
                           protected=True)

    at_hover, at_edge, at_own = (verdict(p) for p in ((40.0, 0.0, 0.0), (70.0, 0.0, 0.0),
                                                       STEP_WHERE))
    assert (at_hover.ok, at_hover.reason, at_hover.home - NOW) == (False, "energy",
                                                                 pytest.approx(28.5))
    assert (at_edge.ok, at_edge.home - NOW) == (True, pytest.approx(29.5))
    assert (at_own.ok, at_own.reason, at_own.home - NOW) == (False, "budget",
                                                             pytest.approx(41.5))
    free = _step_plan(None)
    (stop,) = free.build_ferry_plan()
    assert stop.devices == ("a",) and stop.position == pytest.approx((40.0, 0.0, 0.0))
    assert free.last_plan.violations == ()


# --------------------------------------------------------------------------- #
# Layering
# --------------------------------------------------------------------------- #

_STDLIB = {"__future__", "dataclasses", "math", "typing"}
_MODULE_LEVEL = {"hermes.types.ids", "hermes.types.scheduler",
                 "hermes.scheduler.stages.s3b_feasibility",
                 "hermes.scheduler.stages.s3d_age_cap", ".member_subset"}
_FORBIDDEN = ("numpy", "hermes.l1", "hermes.mule", "hermes.mission", "experiments",
              "hermes.scheduler.policies", "hermes.scheduler.fl_scheduler")


def test_the_hover_module_imports_the_stages_and_the_member_walk_only():
    tree = ast.parse((REPO / "hermes/scheduler/plan/hover.py").read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append("." * node.level + (node.module or ""))
    assert names
    for name in names:
        assert not name.startswith(_FORBIDDEN), name
        assert name.split(".")[0] in _STDLIB or name in _MODULE_LEVEL, name
