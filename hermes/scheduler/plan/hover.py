"""FeRRy Phase 4: a capped device's best hover point (the user's decision of
2026-09-30, "Fix: best hover point", on the final check's PLAN-1 and E2E2-01).

**Why it exists.** Plan mode clusters the demand with S3a every mission, on
S3's deadlines (``FLScheduler.build_ferry_plan``). S3a anchors by
(-delivery_priority, bucket, deadline) and puts a stop that holds one device
at that device's own position (``stages/s3a_cluster.py``); a stop the plan
reduces keeps S3a's position (:func:`~.member_subset.reduce_stop`); and the
mule widens every device the plan leaves out (``mule_main.py``), so its
deadline recedes, S3a anchors it last, and once it lies far from the rest it
becomes a stop of its own at its own position. For a far device that is the
dearest place to serve it from: the mule flies all the way out to dwell at
0 m, where hovering nearer the dock costs a little more dwell and far less
flight. The final check found such a device needing 47.4 s alone at its own
position under a 45 s budget, and 13.25 s hovering at the dock. No plan
could serve it, the age cap labelled it ``unplannable``, which reads as
physics, the mule widened it again, and the loop starved it, often for good.

So in plan mode a capped device that its own S3a stop cannot serve alone
within the budget leaves that stop for a stop of its own at its best hover
point on that class (:func:`offer_hover_stops`), and the cap's
``unplannable`` means that no class the arm may fly serves it alone even
there (``stages/s3d_age_cap.py``): physics without an energy capacity (see
below). Uncapped devices, and capped devices their S3a stop serves alone,
keep S3a's stops.

**The best hover point** (:func:`best_hover_point`) of device j on a class is
the point p of the segment from the dock to j that minimises j's alone
mission: the flight dock -> p, the dwell at |j - p| at the class's predicted
rate, the flight p -> dock and the Pass-1 upload, among the points within the
class's planar reach R(c) of j where j's predicted dwell is finite (at or
above the SNR floor). A point beyond reach or below the floor would not
solicit j, and the model would charge it no dwell (``FerryPhysics.dwell_s``,
critic B12), so it is never offered. Every price is the class's own
``FeasibilityModel``'s, the planner's, so the plan, its guard fold and the
in-flight checks price the stop alike.

Why the segment: a point q at distance r from j costs at least what the point
of the segment at distance min(r, |j - dock|) from j costs. If r <= |j - dock|
that point has the same dwell and is no farther from the dock (the triangle
inequality: |q - dock| >= |j - dock| - r); otherwise the dock itself is
nearer j than q is, so its dwell is no longer, and it needs no flight. So the
segment holds the best point of the plane whenever j's dwell does not fall
with the distance, as on the contact link (the rate falls with the SNR, which
falls with the distance), and the best hover point is then never slower than
j's S3a stop, or any other. It minimises time, not the simulated energy: with
an energy capacity, a device its S3a stop serves alone keeps that stop, so
the rule never takes a stop that serves a device away from it. But the
energy clause's need, P_move for the flight out and back plus P_hover for the
dwell, weighs a second of dwell more than a second of flight (the pilots'
powers: 168.5 W hovering, 143.6 W flying), where the time weighs them alike,
so a point a little slower than the best one, with a shorter dwell, can need
less energy. Under a capacity the energy clause can refuse j at its best
hover point while such a point serves it alone within the budget and the
capacity; the rule does not look for that point, so there the cap's
``unplannable`` reads the time-minimising point and is physics for the time
budget only. On the critic's and the S* tool's 30 layouts with no budget
binding, 34 of the 1,080 device-class pairs have such a point at 1 MB (2 at
64 kB, 625 at 8 MB). No pilot sets a capacity; a rule that weighed the energy
is the user's call.

**How it is found**, deterministically: a fixed grid and a fixed number of
refinement rounds, no wall time and no randomness. The distances from j run
over [0, r_max], r_max = min(|j - dock|, R(c)), cut back by bisection to the
farthest distance at which j's dwell is still finite when the floor comes
first. A grid of :data:`GRID_CELLS` cells is priced; then, for at most
:data:`ROUNDS` rounds, every cell that could still beat the best point priced
by more than :data:`TOLERANCE_S` is halved, at most :data:`KEEP_CELLS` a
round, the lowest bounds first. A cell's bound is the flight priced at its
end nearer the dock plus the dwell priced at its end nearer j, plus the
upload: along the segment towards the dock the flight shrinks and the dwell
does not fall, so no point of the cell costs less. The best point priced is
returned, a tie going to the point nearer j. At the contact link's default
floor the dwell is a step function of the distance (the Shannon cap never
binds there, ``hermes/l1/contact_link.py``), so the alone mission falls
linearly along each step and jumps up at its edge: only the cells holding an
edge survive the bound, the halving brackets the best edge, and the result is
the minimum to within the tolerance. Where the dwell varies smoothly the cap
on cells can stop the refinement short of the tolerance; the point returned
is still the best priced. On the critic's 30 reference layouts it took 65 to
118 pricings a point.

Freeze Rule 1: only the plan path (``plan_mode = "ferry"``) imports this
module, and the recorded pipeline never loads it. Like the rest of the plan
package it is numpy-free and imports nothing from ``hermes.l1``, the mule or
``experiments``: physics reaches it as the class's ``FeasibilityModel``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel, FlightState
from hermes.scheduler.stages.s3d_age_cap import servable_alone
from hermes.types.ids import DeviceID
from hermes.types.scheduler import Bucket, ContactWaypoint, MissionPass

from .member_subset import reduce_stop

__all__ = [
    "GRID_CELLS", "ROUNDS", "KEEP_CELLS", "TOLERANCE_S", "REACH_STEPS",
    "HoverPoint", "best_hover_point", "hover_stop", "offer_hover_stops",
]

#: Cells of the first grid over the distances from the device.
GRID_CELLS = 64
#: Halving rounds of the refinement, at most.
ROUNDS = 64
#: Cells halved per round, at most: the lowest bounds first.
KEEP_CELLS = 8
#: A cell is halved only while it could beat the best point by more than this.
TOLERANCE_S = 1e-9
#: Bisection steps for the farthest distance at which the dwell is finite.
REACH_STEPS = 64

_COLLECT = MissionPass.COLLECT

Position = Tuple[float, float, float]


@dataclass(frozen=True)
class HoverPoint:
    """A device's best hover point on one class (:func:`best_hover_point`).

    ``position`` is the point, ``distance_m`` the device's distance from it as
    the class's model measures it (within the class's reach), and ``home_s``
    the device's alone mission from it, takeoff to the upload done: the
    flight out, the dwell, the flight back and the Pass-1 upload, in seconds.
    """

    position: Position
    distance_m: float
    home_s: float


class _Segment:
    """The segment from the device to the dock, priced on one class.

    A point is named by its distance r from the device, measured along the
    segment. Each price is memoised, so a point is priced once.
    """

    def __init__(self, model: FeasibilityModel, device: DeviceID, reach_m: float) -> None:
        ferry = model.ferry
        self.model = model
        self.device = device
        self.reach_m = reach_m
        self.dock: Position = tuple(float(c) for c in ferry.dock)  # type: ignore[assignment]
        probe = self._stop(self.dock)
        self.where: Position = tuple(  # type: ignore[assignment]
            float(c) for c in ferry.member_position(device, probe))
        self.length_m = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.where, self.dock)))
        self._prices: Dict[float, Optional[Tuple[float, float, float]]] = {}

    def _stop(self, position: Position) -> ContactWaypoint:
        """The device alone at ``position``: a probe priced as the stop would be."""
        return ContactWaypoint(position=position, devices=(self.device,),
                               bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=math.inf)

    def point(self, r: float) -> Position:
        """The point of the segment ``r`` metres from the device: the device
        itself at 0 and the dock itself at the segment's length."""
        if r <= 0.0 or self.length_m == 0.0:
            return self.where
        if r >= self.length_m:
            return self.dock
        t = r / self.length_m
        x, y, z = (a + t * (b - a) for a, b in zip(self.where, self.dock))
        return x, y, z

    def price(self, r: float) -> Optional[Tuple[float, float, float]]:
        """(flight out and back, dwell, upload) of the device alone at ``r``,
        or None when the point does not reach it: beyond the class's reach, or
        its predicted dwell is not finite (below the floor)."""
        if r not in self._prices:
            self._prices[r] = self._price(r)
        return self._prices[r]

    def _price(self, r: float) -> Optional[Tuple[float, float, float]]:
        ferry = self.model.ferry
        point = self.point(r)
        stop = self._stop(point)
        (distance,) = ferry.member_distances_m(stop)
        if distance > self.reach_m or _gate_distance(point, self.where) > self.reach_m:
            return None
        dwell = ferry.member_dwell_s(distance, _COLLECT, 0.0)
        if dwell is None or not math.isfinite(float(dwell)):
            return None
        leg = self.model.leg(self.dock, stop, pass_kind=_COLLECT)
        return leg.transit_s + leg.return_s, leg.dwell_s, leg.upload_s

    def home(self, r: float) -> float:
        """The alone mission from ``r`` (a point that reaches the device)."""
        flight, dwell, upload = self.price(r)  # type: ignore[misc]
        return flight + dwell + upload


def _gate_distance(a: Position, b: Position) -> float:
    """The distance S3a clusters by and the contact gate solicits by.

    ``math.sqrt``, as ``s3a_cluster._distance`` and ``contact_plan.
    planar_distance_m`` take it, where the model's ``member_distances_m``
    takes ``** 0.5``; the two can differ in the last bit. A point at the edge
    of the reach must pass both: the model's, so that the plan charges the
    device's dwell, and the gate's, so that the mule solicits it on arrival.
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def best_hover_point(
    model: FeasibilityModel,
    device: DeviceID,
    *,
    reach_m: float,
) -> Optional[HoverPoint]:
    """``device``'s best hover point on the class ``model`` prices (module docstring).

    ``model`` is the class's ``FeasibilityModel``, with ferry physics that
    price each member (a ``PlanClass`` model), bound to the device states, so
    that the device is found where the plan prices it; ``reach_m`` is the
    class's planar reach R(c), its S3a radius. Returns None when the device
    is not reached even from above it, which the contact link never gives
    (at 0 m its SNR is at its highest); the caller then keeps the device's
    S3a stop. The search assumes, as on the link, that the distances which
    reach the device run from 0 m up to a farthest one. The point does not
    depend on the clock, the budget or the other devices.
    """
    ferry = getattr(model, "ferry", None)
    if ferry is None or getattr(ferry, "member_dwell_s", None) is None:
        raise ValueError("the hover point is priced per member on the mission clock: the model "
                         "needs ferry physics with a member dwell (a PlanClass model)")
    if ferry.device_states is None:
        raise ValueError("bind the model to the device states (FerryPhysics.bind): the hover "
                         "point is found from the device's position")
    reach = float(reach_m)
    if not (math.isfinite(reach) and reach > 0.0):
        raise ValueError(f"reach_m must be finite and > 0, got {reach_m!r}")
    seg = _Segment(model, device, reach)
    if seg.price(0.0) is None:
        return None
    far = min(seg.length_m, reach)
    if seg.price(far) is None:
        # The farthest distance that still reaches the device: the floor comes
        # before the reach, or the reach itself rounds the wrong way.
        near = 0.0
        for _ in range(REACH_STEPS):
            mid = 0.5 * (near + far)
            if seg.price(mid) is None:
                far = mid
            else:
                near = mid
        far = near
    best = _refine(seg, far)
    position = seg.point(best)
    (distance,) = ferry.member_distances_m(seg._stop(position))
    return HoverPoint(position=position, distance_m=float(distance), home_s=seg.home(best))


def _refine(seg: _Segment, far: float) -> float:
    """The distance of the best point priced on [0, ``far``] (module docstring)."""
    grid = [far * i / GRID_CELLS for i in range(GRID_CELLS + 1)] if far > 0.0 else [0.0]
    best = None
    for r in grid:
        if seg.price(r) is not None and (best is None or seg.home(r) < seg.home(best)):
            best = r
    assert best is not None  # 0 m reaches the device
    cells = list(zip(grid, grid[1:]))
    for _ in range(ROUNDS):
        live: List[Tuple[float, float, float]] = []
        for a, b in cells:
            near, out = seg.price(a), seg.price(b)
            if near is None or out is None:
                continue
            bound = out[0] + near[1] + near[2]
            if bound < seg.home(best) - TOLERANCE_S:
                live.append((bound, a, b))
        if not live:
            break
        live.sort()
        cells = []
        for _, a, b in live[:KEEP_CELLS]:
            mid = 0.5 * (a + b)
            if not a < mid < b:
                continue
            if seg.price(mid) is not None:
                home = seg.home(mid)
                if home < seg.home(best) or (home == seg.home(best) and mid < best):
                    best = mid
            cells += [(a, mid), (mid, b)]
    return best


def hover_stop(
    wp: ContactWaypoint,
    device: DeviceID,
    point: HoverPoint,
    *,
    deadlines: Mapping[DeviceID, float],
    device_states: Mapping[DeviceID, Any],
    capped: Collection[DeviceID] = frozenset(),
) -> ContactWaypoint:
    """``device``'s stop of its own at ``point``, taken from its S3a stop ``wp``.

    The stop ``reduce_stop`` makes of ``wp`` with ``device`` alone (its bucket
    and its own deadline, by B2's rule under the cap: every member capped, so
    the earliest of all, its own), moved to the hover point. It is an ordinary
    ``ContactWaypoint``, exempt from its own deadline clause when the device
    is capped, as every one-device capped stop is.
    """
    alone = reduce_stop(wp, (device,), deadlines=deadlines, device_states=device_states,
                        capped=capped)
    return replace(alone, position=tuple(point.position))


def offer_hover_stops(
    stops: Sequence[ContactWaypoint],
    *,
    model: FeasibilityModel,
    reach_m: float,
    movable: Collection[DeviceID],
    start: FlightState,
    budget_end: Optional[float],
    deadlines: Mapping[DeviceID, float],
    device_states: Mapping[DeviceID, Any],
    capped: Collection[DeviceID] = frozenset(),
) -> Tuple[ContactWaypoint, ...]:
    """One class's S3a ``stops`` with the hover rule: the stops the search is offered.

    Each device of ``movable`` (the capped devices, in plan mode) that its S3a
    stop cannot serve alone from ``start`` within ``budget_end`` (U1's
    ``servable_alone``, the test the cap's labels use) leaves that stop for a
    stop of its own at its best hover point on this class
    (:func:`best_hover_point`, :func:`hover_stop`). The stop it leaves keeps
    its position, and its members' bucket and deadline follow S3a's and B2's
    rules (``reduce_stop`` under ``capped``); a stop left with no member goes.
    Every other stop is returned as it is, the same object, so each device
    stays in exactly one stop, as ``servable_alone`` and the trims require.
    Each stop a device leaves is followed by the devices' hover stops, in its
    member order. A device no point of the segment reaches keeps its S3a stop.
    Without a budget nothing is gated (S3b's opt-in contract), so nothing
    moves and the stops are S3a's.
    """
    stops = tuple(stops)
    wanted = [d for wp in stops for d in wp.devices if d in movable]
    if budget_end is None or not wanted:
        return stops
    fits = servable_alone(wanted, [(model, stops)], start=start, budget_end=budget_end)
    out: List[ContactWaypoint] = []
    for wp in stops:
        moved = []
        for d in wp.devices:
            if d in movable and d not in fits:
                point = best_hover_point(model, d, reach_m=reach_m)
                if point is not None:
                    moved.append((d, point))
        if not moved:
            out.append(wp)
            continue
        gone = {d for d, _ in moved}
        rest = [d for d in wp.devices if d not in gone]
        if rest:
            out.append(reduce_stop(wp, rest, deadlines=deadlines, device_states=device_states,
                                   capped=capped))
        out.extend(hover_stop(wp, d, point, deadlines=deadlines, device_states=device_states,
                              capped=capped) for d, point in moved)
    return tuple(out)
