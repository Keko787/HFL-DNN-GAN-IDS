"""The age cap's S*, at planning level (FeRRy Phase 4, unit U9).

The Phase 4 spec, other choices 13, and the user's decision 1 of 2026-09-30.
The age cap S forces the plan to serve a device whose update has not reached
the model for S missions of its own mule (the scorer's unit,
``traces_scorer.age_profile``). Decision 1 sets S from this tool: the smallest
value that covers 90 % of layouts at both pilot budgets (the knee and the
stress budget), never below 2 (critic A1: S = 1 caps every device at every
mission). S is a configuration value (``--age-cap-missions``); the tool prints
it, (i), and S + 1, (ii), which in critic probe A removed every miss the
one-mission-ahead planner makes at S = S*. That holds for F and FB+narrow at
45-90 s under subset admission, on the layouts S covers, not for every pinned
class nor at 30 s: see below. The report marks each (ii) it cannot vouch for
(:func:`s_plus_1_caveats`).

**What S* is.** For one layout, one budget and one arm family, S* is the fewest
Pass-1 missions that together serve every device a mission can serve at all:
the set-cover number of those devices by the device sets one mission can
serve within the budget (design D-A; research map section 2). Below S* the
cap asks for more than any schedule gives, so some capped device goes
unserved whatever the plan does; at S* a covering schedule exists but the
myopic plan can still crowd a device (critic A4, layout 25).

**What S* + 1 gives, as measured.** The mission loop on critic B4's
deterministic loopback (U7's harness, ``test_p4_plan_missions.py``), on the
tool's own 30 layouts at 1 MB, 3S missions at S = S* + 1 (each layout at its
own S*), after the hover rule of 2026-09-30 (below), with either deadline
unit: no ``unplannable`` at 45, 60 or 90 s in any family, and no violation at
all for F and FB+narrow there. A pinned class can still crowd a capped device
at the stress budget. S3a re-clusters every mission on the devices' buckets
and deadlines, so a capped device's stop can drift from the tool's fresh
partition to one that serves it alone but not beside another capped device:
FB+medium crowded 7 times in 291 missions at 45 s (6 on layout 0, one device a
mission from the fourth, each served a mission late), FB+wide twice in 360;
neither at 60 or 90 s. At 30 s every family crowds at S* + 1 (F 20 times in
309 missions). The cell's S + 1 is at least a layout's S* + 1 only where S
covers the layout: F at the pilots' budgets (90 and 45 s) has S + 1 = 3, and
it crowded 5 times in 270 missions at 45 s, all on layout 7, whose S* (3) is
above S; with 90, 60, 45 and 30 s F has S + 1 = 4, which crowded 19 times in
360 missions at 30 s, on layouts S covers, and never at 45 to 90 s. So S + 1
is no guarantee for a pinned class at the stress budget, nor for any family at
a budget outside the 45 to 90 s measured, nor under whole admission (below),
and the report says so on each (ii) line it cannot vouch for
(:func:`s_plus_1_caveats`). Before the hover rule a pinned class at the stress
budget could starve a device for good under ``unplannable`` (the final
check's PLAN-1 and E2E2-01: 36 such violations of FB+medium at 45 s, ages up
to 15).

**Planning level, and what that leaves out.** Nothing is flown and nothing is
drawn: each mission is priced by the predicate the mule plans with
(``FeasibilityModel.fold``, S3b) at the mean SNR, deterministically.

* *The budget rule.* A mission is feasible when the predicate's budget rule
  (``RULE_BUDGET``: the budget clause, and the energy clause when a capacity
  is configured) admits every stop from the dock at time 0: its home, with
  the Pass-1 upload, is within the budget (home never decreases along a
  route, so the last stop's home bounds every earlier one).
* *Deadlines are left out.* A capped device's stop is exempt from its own
  deadline clause (the predicate's ``protected``), and S counts service, not
  punctuality.
* *The stops a plan is offered.* Each class's stops are S3a's at the class's
  radius R_planar(c) on a fresh plan (every device new), as the plan path
  clusters them (``FLScheduler.build_contact_queue``), with the plan's hover
  rule (``hermes.scheduler.plan.hover``, the user's decision of 2026-09-30)
  applied to every device as if it were capped: a device its S3a stop cannot
  serve alone within the budget leaves that stop for a stop of its own at its
  best hover point, the point of the segment from the dock to it where its
  alone mission is shortest. A running mission re-clusters as the devices'
  buckets change (critic probe H), so its stops of several devices can differ
  from the tool's (the drift above). Its servable devices cannot: a device's
  best hover point depends on neither the partition nor the other devices,
  and no stop serves it alone sooner, so without an energy capacity a device
  the tool can serve is one the plan can serve once it is capped.
  ``hover=False`` (:func:`layout_s_star`) prices S3a's stops alone, the
  family before the decision, which the design probe's table used.
* *The trial's physics.* Every class is priced by the model the plan-mode
  mule builds for it (``FerryRuntime.plan_classes``), from the spec the
  driver would give the cell's mule (``Exp4Driver.ferry_settings`` through
  ``MuleConfig.ferry_spec_kwargs``, as T_nom's reference layouts are priced),
  with the payload the trial pushes: the declared ``payload_bytes``, else the
  measured θ and synthetic batch (the stub's unless ``theta_bytes`` says
  otherwise, e.g. 18,756 B for the canonical model).
* *Its own reference layouts.* Layout k is drawn as a trial's is
  (``device_positions``, the cell's spread) from the seed ``_u32(N, "s_star",
  k)``, not from the grid's seeds, so S does not move with the base seed or
  the trial count, following T_nom's precedent (Freeze L974-976).
* *Member admission.* ``subset`` (the F family's rule): a mission may serve
  any subset of the members of the stops it touches; ``whole``: only whole
  stops.
* *Devices no mission serves alone.* A device that no class of the family
  serves alone within the budget even at its best hover point (the plan's
  ``unplannable``) cannot be covered at any S; it is left out of the cover and
  reported, never counted as a missing mission. Without an energy capacity
  that is physics: no point serves the device alone sooner. With one it need
  not be: the point minimises time, and a slower point with a shorter dwell
  can need less energy (``hermes.scheduler.plan.hover``), so a device refused
  there for the energy may still be servable alone. Under ``whole`` the tool
  also leaves out a device whose whole stop no mission flies, which the plan
  labels ``crowded`` when it fits alone, so S + 1 is no guarantee there.
* *Exact, then greedy.* Up to ``exact_max_devices`` devices (6, the plan
  search's exact bound) S* is exact: every device set, every order of its
  stops, then the fewest feasible sets covering the servable devices. Above,
  it is a greedy upper bound: each mission is, for each class, the 2-OPT tour
  of the stops of the devices still uncovered, folded with the F family's
  member walk (``plan.member_subset.fold_members``), and the class serving
  the most is taken.
* *Arm families.* F searches every band class (one mission may fly any
  class); FB+<class> pins that class.

Every number is a function of the cell and the driver's settings alone.

Usage (the pilots' cell: N = 6, 1 MB, the knee and the stress budget)::

    python -m experiments.analysis.age_cap_s_star --budgets 90 45 \\
        --payload-bytes 1000000 --contact-band wide --regime jittery
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from experiments.exp4.driver import Exp4Driver
from experiments.exp4.model_task import _u32
from experiments.exp4.topology_builder import device_positions, device_spread_m
from hermes.scheduler.stages.s3b_feasibility import (
    MEMBER_ADMISSION_SUBSET,
    MEMBER_ADMISSION_WHOLE,
    MEMBER_ADMISSIONS,
    RULE_BUDGET,
    FeasibilityModel,
    FlightState,
)
from hermes.types import ContactWaypoint, DeviceID, MissionSlice, MuleID

#: The tag of the tool's own reference layouts: layout k's seed is
#: ``_u32(N, LAYOUT_TAG, k)`` (the Phase 4 spec, other choices 13).
LAYOUT_TAG = "s_star"
#: Decision 1: S covers this share of the layouts at every budget given ...
COVER_SHARE = (9, 10)
#: ... and is never below this (critic A1).
S_FLOOR = 2
#: The budgets (s) between which S + 1 was measured free of violations for F
#: under subset admission (the module docstring, "What S* + 1 gives"); the
#: report marks a (ii) whose budgets reach outside them.
S_PLUS_1_MEASURED_S = (45.0, 90.0)
#: Exact set cover up to this many devices, a greedy upper bound above
#: (``PlanSearchParams.exact_max_devices``, the plan search's own bound).
EXACT_MAX_DEVICES = 6
#: The arm family that searches every band class (the driver's arm F).
FAMILY_SEARCH = "F"
#: The pinned families are the driver's FB+<class> arms.
FAMILY_FIXED_PREFIX = "FB+"
#: The plan arms' reference spec is built for arm F: FB+<class> prices its
#: class with the same per-class physics (``FerryRuntime.plan_classes``).
_SPEC_ARM = "F"


# --------------------------------------------------------------------------- #
# Layouts and their planning worlds
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Layout:
    """One reference layout: its index, seed and each device's position."""

    index: int
    seed: int
    positions: Tuple[Tuple[str, Tuple[float, float, float]], ...]

    @property
    def devices(self) -> Tuple[str, ...]:
        return tuple(d for d, _ in self.positions)


def device_name(i: int) -> str:
    """Device ``i``'s id, as the Exp 4 topology names it (``exp4-dev-000``)."""
    return f"exp4-dev-{int(i):03d}"


def reference_layouts(
    n_devices: int, *, count: int, spread_m: float, tag: str = LAYOUT_TAG, offset: int = 0,
    far_share: Optional[float] = None, far_radius_m: Optional[float] = None,
) -> Tuple[Layout, ...]:
    """The tool's reference layouts: layout k from the seed ``_u32(N, tag, offset + k)``.

    Drawn as a trial's devices are (``device_positions``). ``tag`` and
    ``offset`` exist to reproduce other probes' layouts (the design probe drew
    ``_u32(N, "t_nom", 1000 + k)``); the tool's own are ``"s_star"`` from 0.
    ``far_share`` (unit U10) places that share beyond ``far_radius_m`` of the
    dock, as the trials of a ``--far-share`` cell do; None is the recorded draw.
    """
    if int(n_devices) < 1 or int(count) < 1:
        raise ValueError(f"need at least one device and one layout, got {n_devices}, {count}")
    out = []
    for k in range(int(count)):
        seed = _u32(int(n_devices), tag, int(offset) + k)
        if far_share is None:
            xy = device_positions(int(n_devices), seed, float(spread_m))
        else:
            xy = device_positions(int(n_devices), seed, float(spread_m),
                                  far_share=far_share, far_radius_m=far_radius_m)
        out.append(Layout(
            index=k, seed=seed,
            positions=tuple((device_name(i), (float(x), float(y), 0.0))
                            for i, (x, y) in enumerate(xy)),
        ))
    return tuple(out)


@dataclass(frozen=True)
class ClassWorld:
    """One band class of a layout as a fresh plan sees it.

    ``stops`` are S3a's at the class's radius, every device new; ``model``
    prices the class (bound to it, critic B3) with member positions looked up
    in ``device_states``.
    """

    name: str
    index: int
    radius_m: float
    model: FeasibilityModel
    stops: Tuple[ContactWaypoint, ...]
    device_states: Mapping[Any, Any]

    @property
    def dock(self) -> Tuple[float, float, float]:
        return self.model.ferry.dock  # type: ignore[union-attr]


class LayoutWorld:
    """A layout's classes, the stops a plan is offered on each, and every mission's price.

    :meth:`stops` applies the plan's hover rule at a budget, and
    :meth:`homes` prices every mission on those stops once per class,
    admission and stop family: a device set's earliest home over the orders
    the predicate admits with no budget binding (the energy clause alone, when
    a capacity is set), which every budget with the same stops shares.
    """

    def __init__(self, layout: Layout, classes: Sequence[ClassWorld]) -> None:
        self.layout = layout
        self.classes: Dict[str, ClassWorld] = {c.name: c for c in classes}
        self.class_names: Tuple[str, ...] = tuple(c.name for c in classes)
        self._stops: Dict[Tuple[str, float], Tuple[ContactWaypoint, ...]] = {}
        self._homes: Dict[Tuple[str, str, Tuple[ContactWaypoint, ...]],
                          Dict[FrozenSet[str], float]] = {}

    def stops(self, band: str, budget_s: Optional[float] = None) -> Tuple[ContactWaypoint, ...]:
        """``band``'s stops as a plan is offered them at ``budget_s``.

        S3a's fresh stops (:attr:`ClassWorld.stops`) with the plan's hover
        rule (``hermes.scheduler.plan.hover.offer_hover_stops``) applied to
        every device, as if capped: a device its S3a stop cannot serve alone
        from the dock within ``budget_s`` leaves that stop for a stop of its
        own at its best hover point on the class. The tool counts a schedule
        that serves every device within S missions, and a device the plan
        must serve is a capped one, which is what the rule moves. None gives
        S3a's stops alone: the family the tool priced before the user's
        decision of 2026-09-30, kept to reproduce the design probe's table.
        """
        cls = self.classes[band]
        if budget_s is None:
            return cls.stops
        key = (band, float(budget_s))
        if key not in self._stops:
            from hermes.scheduler.plan.hover import offer_hover_stops

            self._stops[key] = offer_hover_stops(
                cls.stops, model=cls.model, reach_m=cls.radius_m,
                movable=frozenset(d for wp in cls.stops for d in wp.devices),
                start=FlightState(cls.dock, 0.0), budget_end=float(budget_s), deadlines={},
                device_states=cls.device_states,
            )
        return self._stops[key]

    def homes(
        self, band: str, admission: str, budget_s: Optional[float] = None,
    ) -> Dict[FrozenSet[str], float]:
        """Every device set one mission of ``band`` can serve, with its earliest home.

        On the stops :meth:`stops` offers at ``budget_s`` (None: S3a's
        alone). ``subset``: every non-empty device set, each served by the
        stops it touches reduced to its members; ``whole``: every union of
        whole stops. The home is the earliest over the orders of those stops,
        with the Pass-1 upload, among the orders the budget rule admits with
        an unbounded budget (so only an energy capacity can refuse one:
        ``inf`` when it refuses them all).
        """
        stops = self.stops(band, budget_s)
        key = (band, admission, stops)
        if key not in self._homes:
            self._homes[key] = _mission_homes(self.classes[band], admission, stops)
        return self._homes[key]


def planning_world(
    driver: Exp4Driver,
    layout: Layout,
    *,
    rf_range_m: float,
    regime: str,
    theta_bytes: int,
    synth_bytes: int,
) -> LayoutWorld:
    """``layout``'s classes, priced with the spec ``driver`` gives the cell's mule.

    The spec is the one a trial of the cell would build for a plan arm
    (``ferry_settings`` of arm F, through the mule's own mapping), seeded with
    the layout's seed as T_nom's layouts are, with a placeholder backhaul
    period (planning never reads it). Each class gets its own fresh planner,
    so no class's clustering sees another's.
    """
    from hermes.mule.ferry import FerryRuntime, FerrySpec
    from hermes.scheduler.fl_scheduler import FLScheduler

    settings = driver.ferry_settings(arm=_SPEC_ARM, regime=regime)
    if settings.get("contact_band") is None:
        raise ValueError(
            "S* needs band classes: the channel-free control (no contact band) has none; "
            "give the cell's reference class (contact_band, e.g. wide)"
        )
    if settings.get("backhaul_model") == "seconds":
        settings["backhaul_period_s"] = settings.get("backhaul_period_s") or 1.0
    spec = FerrySpec.from_config(**driver._spec_kwargs(
        settings, rf_range_m=float(rf_range_m), seed=int(layout.seed), n_missions=1,
    ))
    runtime = FerryRuntime(spec, None, rf_range_m=float(rf_range_m))
    runtime.set_payload(theta_bytes=int(theta_bytes), synth_bytes=int(synth_bytes))
    classes = []
    for cls in runtime.plan_classes():
        planner = FLScheduler(now_fn=lambda: 0.0)
        planner.ingest_slice(MissionSlice(
            mule_id=MuleID("s_star"), device_ids=tuple(DeviceID(d) for d in layout.devices),
            issued_round=0, issued_at=0.0,
        ))
        for d, pos in layout.positions:
            planner.device_states[DeviceID(d)].last_known_position = pos
        model = dataclasses.replace(cls.model, ferry=cls.model.ferry.bind(planner.device_states))
        stops = tuple(planner.build_contact_queue(rf_range_m=cls.radius_m,
                                                  mule_pose=model.ferry.dock))
        classes.append(ClassWorld(name=cls.name, index=cls.index, radius_m=cls.radius_m,
                                  model=model, stops=stops,
                                  device_states=planner.device_states))
    return LayoutWorld(layout, classes)


def _fold_home(world: ClassWorld, route: Sequence[ContactWaypoint]) -> float:
    """The route's home from the dock at 0 under the budget rule with no budget
    binding; ``inf`` when an energy capacity refuses a stop."""
    got = world.model.fold(list(route), FlightState(world.dock, 0.0), rule=RULE_BUDGET,
                           budget_end=math.inf, skip=False)
    return got.home if got.ok else math.inf


def _best_home(world: ClassWorld, stops: Sequence[ContactWaypoint]) -> float:
    return min(_fold_home(world, order) for order in itertools.permutations(stops))


def _mission_homes(
    world: ClassWorld, admission: str, offered: Sequence[ContactWaypoint],
) -> Dict[FrozenSet[str], float]:
    """:meth:`LayoutWorld.homes` for one class on its ``offered`` stops (every set,
    every order)."""
    from hermes.scheduler.plan.member_subset import reduce_stop

    homes: Dict[FrozenSet[str], float] = {}
    if admission == MEMBER_ADMISSION_WHOLE:
        for r in range(1, len(offered) + 1):
            for combo in itertools.combinations(offered, r):
                served = frozenset(str(d) for wp in combo for d in wp.devices)
                homes[served] = min(homes.get(served, math.inf), _best_home(world, combo))
        return homes
    devices = sorted(str(d) for wp in offered for d in wp.devices)
    for r in range(1, len(devices) + 1):
        for subset in itertools.combinations(devices, r):
            chosen = frozenset(subset)
            stops = [
                reduce_stop(wp, [d for d in wp.devices if str(d) in chosen],
                            deadlines={}, device_states=world.device_states)
                for wp in offered if any(str(d) in chosen for d in wp.devices)
            ]
            homes[chosen] = _best_home(world, stops)
    return homes


# --------------------------------------------------------------------------- #
# S* of one layout
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LayoutStar:
    """S* of one layout at one budget for one arm family.

    ``s_star`` is the fewest missions covering the ``servable`` devices (0
    when none is): exact when ``exact``, else a greedy upper bound. ``cover``
    is one such schedule, each mission's devices. ``unservable`` are the
    devices no class of the family serves even alone (the plan's
    ``unplannable``), left out of the cover.
    """

    layout: int
    s_star: int
    exact: bool
    servable: Tuple[str, ...]
    unservable: Tuple[str, ...]
    cover: Tuple[Tuple[str, ...], ...]


def layout_s_star(
    world: LayoutWorld,
    *,
    classes: Sequence[str],
    budget_s: float,
    admission: str = MEMBER_ADMISSION_SUBSET,
    exact_max_devices: int = EXACT_MAX_DEVICES,
    hover: bool = True,
) -> LayoutStar:
    """S* of ``world`` at ``budget_s`` when each mission may fly any one of ``classes``.

    ``budget_s`` is a finite number of seconds > 0. A mission an energy
    capacity refuses is priced ``inf`` (:meth:`LayoutWorld.homes`), which an
    infinite budget would admit; a budget of 0 or less, or NaN, admits
    nothing, and S = 2 would then come from the floor alone. Each class's
    stops are those the plan is offered at ``budget_s``, the hover rule's
    (:meth:`LayoutWorld.stops`); ``hover=False`` prices S3a's fresh stops
    alone, the family before the user's decision of 2026-09-30, which the
    plan no longer flies (it reproduces the design probe's table).
    """
    if admission not in MEMBER_ADMISSIONS:
        raise ValueError(f"admission must be one of {MEMBER_ADMISSIONS}, got {admission!r}")
    unknown = [c for c in classes if c not in world.classes]
    if not classes or unknown:
        raise ValueError(f"classes must name the link's classes {world.class_names}, "
                         f"got {list(classes)}")
    budget = float(budget_s)
    if not (math.isfinite(budget) and budget > 0.0):
        raise ValueError(
            f"budget_s must be a finite number of seconds > 0, got {budget_s!r} (for the "
            f"energy clause alone, give a large finite budget)"
        )
    family = budget if hover else None
    devices = world.layout.devices
    if len(devices) <= int(exact_max_devices):
        feasible = {s for c in classes for s, home in world.homes(c, admission, family).items()
                    if home <= budget}
        servable = frozenset().union(*feasible) if feasible else frozenset()
        cover = _exact_cover(feasible, servable)
        exact = True
    else:
        servable = _servable_alone(world, classes, budget, admission, family)
        cover = _greedy_cover(world, classes, budget, admission, servable, family)
        exact = False
    return LayoutStar(
        layout=world.layout.index,
        s_star=len(cover),
        exact=exact,
        servable=tuple(d for d in devices if d in servable),
        unservable=tuple(d for d in devices if d not in servable),
        cover=tuple(tuple(d for d in devices if d in mission) for mission in cover),
    )


def _exact_cover(
    feasible: Sequence[FrozenSet[str]], target: FrozenSet[str],
) -> Tuple[FrozenSet[str], ...]:
    """The fewest ``feasible`` sets whose union is ``target``.

    The family is down-closed (a subset of a mission's devices is served by
    the same stops, reduced, in the same order, no later: dwell adds up over
    members and the triangle inequality bounds the rest), so only its maximal
    sets are needed, taken in a fixed order so the cover found is
    reproducible.
    """
    if not target:
        return ()
    maximal = sorted((s for s in feasible if not any(s < t for t in feasible)),
                     key=lambda s: (-len(s), sorted(s)))
    for k in range(1, len(target) + 1):
        for combo in itertools.combinations(maximal, k):
            if frozenset().union(*combo) == target:
                return tuple(combo)
    raise AssertionError("the servable devices are the union of the feasible sets")


def _servable_alone(
    world: LayoutWorld, classes: Sequence[str], budget: float, admission: str,
    family: Optional[float],
) -> FrozenSet[str]:
    """The devices some class serves alone: at its own offered stop reduced to
    it (``subset``), or its whole stop (``whole``), from the dock at 0."""
    from hermes.scheduler.plan.member_subset import reduce_stop

    out = set()
    for band in classes:
        cls = world.classes[band]
        for wp in world.stops(band, family):
            if admission == MEMBER_ADMISSION_WHOLE:
                if _fold_home(cls, [wp]) <= budget:
                    out.update(str(d) for d in wp.devices)
                continue
            for d in wp.devices:
                alone = reduce_stop(wp, [d], deadlines={}, device_states=cls.device_states)
                if _fold_home(cls, [alone]) <= budget:
                    out.add(str(d))
    return frozenset(out)


def _greedy_cover(
    world: LayoutWorld, classes: Sequence[str], budget: float, admission: str,
    servable: FrozenSet[str], family: Optional[float],
) -> Tuple[FrozenSet[str], ...]:
    """A greedy cover of ``servable``: each mission the class that serves the most.

    For each class, the offered stops of the devices still uncovered (reduced
    to them under ``subset``, whole under ``whole``) in their 2-OPT tour from
    the dock and back (``two_opt.order_contacts``), folded under the budget
    rule with the F family's member walk (``fold_members``) or whole stops.
    The first stop that admits anyone is flown from the dock, and a servable
    device's stop admits it there, so every mission covers at least one more
    device and the cover ends.
    """
    from hermes.scheduler.plan.member_subset import fold_members, reduce_stop
    from hermes.scheduler.routing.two_opt import order_contacts

    left = set(servable)
    cover: List[FrozenSet[str]] = []
    while left:
        best: FrozenSet[str] = frozenset()
        for band in classes:
            cls = world.classes[band]
            offered = world.stops(band, family)
            start = FlightState(cls.dock, 0.0)
            if admission == MEMBER_ADMISSION_WHOLE:
                route = [wp for wp in offered if any(str(d) in left for d in wp.devices)]
                tour = order_contacts(route, cls.dock, end=cls.dock)
                flown = cls.model.fold(tour, start, rule=RULE_BUDGET, budget_end=budget,
                                       skip=True).route
            else:
                route = [
                    reduce_stop(wp, [d for d in wp.devices if str(d) in left], deadlines={},
                                device_states=cls.device_states)
                    for wp in offered if any(str(d) in left for d in wp.devices)
                ]
                tour = order_contacts(route, cls.dock, end=cls.dock)
                flown = fold_members(tour, start, model=cls.model, rule=RULE_BUDGET,
                                     budget_end=budget, deadlines={},
                                     device_states=cls.device_states,
                                     require_all=False).route
            served = frozenset(str(d) for wp in flown for d in wp.devices)
            if len(served & left) > len(best & left):
                best = served
        if not best & left:
            raise AssertionError("a servable device's own stop admits it from the dock")
        cover.append(best)
        left -= best
    return tuple(cover)


# --------------------------------------------------------------------------- #
# Over the layouts: decision 1's S
# --------------------------------------------------------------------------- #

def cover_value(values: Sequence[int], share: Tuple[int, int] = COVER_SHARE) -> int:
    """The smallest S with ``S* <= S`` on at least ``share`` of the layouts.

    Decision 1's "covers 90 % of layouts": the ⌈0.9 n⌉-th smallest S*, in
    integer arithmetic so no rounding can move it.
    """
    if not values:
        raise ValueError("no layouts")
    num, den = share
    need = -(-num * len(values) // den)
    return sorted(int(v) for v in values)[need - 1]


def decision_1_s(per_budget: Mapping[float, Sequence[int]]) -> int:
    """Decision 1's S (i): the largest cover value over the budgets, at least 2."""
    if not per_budget:
        raise ValueError("no budgets")
    return max(S_FLOOR, max(cover_value(v) for v in per_budget.values()))


def family_classes(family: str, class_names: Sequence[str]) -> Tuple[str, ...]:
    """The classes an arm family may fly: every class for F, class c for FB+c."""
    if family == FAMILY_SEARCH:
        return tuple(class_names)
    if family.startswith(FAMILY_FIXED_PREFIX) and family[len(FAMILY_FIXED_PREFIX):] in class_names:
        return (family[len(FAMILY_FIXED_PREFIX):],)
    raise ValueError(f"unknown arm family {family!r}: F or FB+<class> of {list(class_names)}")


@dataclass(frozen=True)
class SStarReport:
    """S* over the layouts, per arm family and budget, and each family's S."""

    budgets: Tuple[float, ...]
    families: Tuple[str, ...]
    admission: str
    #: family -> budget -> one :class:`LayoutStar` per layout.
    stars: Mapping[str, Mapping[float, Tuple[LayoutStar, ...]]]

    def s(self, family: str = FAMILY_SEARCH) -> int:
        """Decision 1's S (i) for ``family``; the cell's S is F's."""
        return decision_1_s({b: [x.s_star for x in self.stars[family][b]]
                             for b in self.budgets})

    def to_json(self) -> Dict[str, Any]:
        return {
            "budgets": list(self.budgets),
            "admission": self.admission,
            "families": {
                fam: {
                    "s": self.s(fam),
                    "s_plus_1": self.s(fam) + 1,
                    "budgets": {
                        str(b): [dataclasses.asdict(x) for x in self.stars[fam][b]]
                        for b in self.budgets
                    },
                }
                for fam in self.families
            },
        }


def s_star_report(
    driver: Exp4Driver,
    *,
    n_devices: int,
    budgets: Sequence[float],
    rf_range_m: float = 60.0,
    regime: str = "jittery",
    families: Optional[Sequence[str]] = None,
    admission: str = MEMBER_ADMISSION_SUBSET,
    layouts: int = 30,
    theta_bytes: Optional[int] = None,
    synth_bytes: Optional[int] = None,
    exact_max_devices: int = EXACT_MAX_DEVICES,
    layout_tag: str = LAYOUT_TAG,
    layout_offset: int = 0,
) -> SStarReport:
    """S* of every reference layout of the cell, per family and budget.

    ``driver`` holds the settings the cell's trials run (it must be on the
    simulated clock, with a reference band). The spread is a trial's: the
    realism field when the driver has realism, else the tight cluster.
    ``families`` defaults to F and FB+<class> for every class of the link.
    ``theta_bytes`` and ``synth_bytes`` default to what a stub trial measures.
    """
    if not driver.sim:
        raise ValueError("S* is a plan-clock number: the driver must run mission_clock='sim'")
    if not budgets:
        raise ValueError("give at least one budget (the knee and the stress budget)")
    if theta_bytes is None or synth_bytes is None:
        stub_theta, stub_synth = driver._payload_bytes(None)
        theta_bytes = stub_theta if theta_bytes is None else theta_bytes
        synth_bytes = stub_synth if synth_bytes is None else synth_bytes
    spread = device_spread_m(
        float(rf_range_m),
        field_radius_m=(driver.field_radius_m(int(n_devices)) if driver.realism else None),
    )
    worlds = [
        planning_world(driver, layout, rf_range_m=rf_range_m, regime=regime,
                       theta_bytes=int(theta_bytes), synth_bytes=int(synth_bytes))
        for layout in reference_layouts(
            n_devices, count=layouts, spread_m=spread, tag=layout_tag, offset=layout_offset,
            far_share=(driver.far_share if driver.realism else None),
            far_radius_m=float(rf_range_m))
    ]
    names = worlds[0].class_names
    fams = tuple(families) if families is not None else (
        (FAMILY_SEARCH,) + tuple(FAMILY_FIXED_PREFIX + c for c in names))
    stars = {
        fam: {
            float(b): tuple(
                layout_s_star(w, classes=family_classes(fam, names), budget_s=float(b),
                              admission=admission, exact_max_devices=exact_max_devices)
                for w in worlds
            )
            for b in budgets
        }
        for fam in fams
    }
    return SStarReport(budgets=tuple(float(b) for b in budgets), families=fams,
                       admission=admission, stars=stars)


def s_plus_1_caveats(family: str, budgets: Sequence[float], admission: str) -> Tuple[str, ...]:
    """Why the report cannot vouch for S + 1, (ii), on ``family``'s line.

    What the module docstring measured ("What S* + 1 gives"): a pinned class
    can crowd a capped device at the stress budget (partition drift); every
    family crowds at 30 s, and nothing was measured outside 45 to 90 s
    (:data:`S_PLUS_1_MEASURED_S`); under whole admission the cover leaves out
    the devices whose whole stop no mission flies, which the plan reports
    every mission. Empty where (ii) holds as measured: F under subset
    admission with every budget within the measured range, on the layouts S
    covers.
    """
    why = []
    if family.startswith(FAMILY_FIXED_PREFIX):
        why.append("for a pinned class")
    low, high = S_PLUS_1_MEASURED_S
    outside = [float(b) for b in budgets if not low <= float(b) <= high]
    if outside:
        why.append("at " + ", ".join(f"{b:g}" for b in outside) + " s")
    if admission != MEMBER_ADMISSION_SUBSET:
        why.append("under whole admission")
    return tuple(why)


def _plus_1_label(caveats: Sequence[str]) -> str:
    """``(ii)``, or ``(ii, no guarantee ...)`` naming each caveat."""
    return f"(ii, no guarantee {' or '.join(caveats)})" if caveats else "(ii)"


def format_report(report: SStarReport) -> str:
    """The report as the tool prints it: per family and budget, then S, each S + 1
    marked where it is no guarantee (:func:`s_plus_1_caveats`)."""
    lines = []
    n = None
    for fam in report.families:
        for b in report.budgets:
            stars = report.stars[fam][b]
            n = len(stars)
            values = [x.s_star for x in stars]
            short = [x for x in stars if x.unservable]
            lines.append(
                f"{fam:<10} B={b:6.1f} s | S* median {statistics.median(values):4.1f}"
                f"  90% {cover_value(values)}  max {max(values)}"
                f" | {'exact' if all(x.exact for x in stars) else 'greedy upper bound'}"
                f" | layouts with unservable devices {len(short)}/{len(stars)}"
                f" ({sum(len(x.unservable) for x in short)} devices)"
            )
        caveats = s_plus_1_caveats(fam, report.budgets, report.admission)
        lines.append(f"{fam:<10} S = {report.s(fam)} (i)   S + 1 = {report.s(fam) + 1} "
                     f"{_plus_1_label(caveats)}")
    if FAMILY_SEARCH in report.families:
        s = report.s(FAMILY_SEARCH)
        caveats = s_plus_1_caveats(FAMILY_SEARCH, report.budgets, report.admission)
        lines.append(
            f"\nThe cell's S (decision 1: F's S* on 90 % of {n} layouts at every budget "
            f"given, at least {S_FLOOR}): S = {s} (i), S + 1 = {s + 1} "
            f"{_plus_1_label(caveats)}. Admission: {report.admission}."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="age_cap_s_star",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--budgets", type=float, nargs="+", required=True,
                    help="The pilot budgets in seconds (the knee and the stress budget).")
    ap.add_argument("--N", type=int, default=6, help="Devices per layout (default 6).")
    ap.add_argument("--rrf", type=float, default=60.0, help="RF range (m; default 60).")
    ap.add_argument("--regime", choices=("clean", "jittery"), default="jittery")
    ap.add_argument("--layouts", type=int, default=30, help="Reference layouts (default 30).")
    ap.add_argument("--contact-band", default="wide",
                    help="The cell's reference band class (default wide).")
    ap.add_argument("--contact-band-classes", nargs="+", default=None,
                    help="The link's band classes (default wide medium narrow).")
    ap.add_argument("--backhaul-model", choices=("mission", "seconds"), default="mission",
                    help="Prices the Pass-1 upload the budget includes.")
    ap.add_argument("--payload-bytes", type=int, default=None,
                    help="Declared payload per direction (bytes); omit for measured.")
    ap.add_argument("--theta-bytes", type=int, default=None,
                    help="The measured θ in bytes (default: the stub's; 18756 for the "
                         "canonical model).")
    ap.add_argument("--ferry-physics", default=None,
                    help="JSON of MuleConfig physics overrides (the driver's ferry_physics).")
    ap.add_argument("--no-realism", action="store_true",
                    help="Draw the tight EX-4.0 cluster instead of the realism field.")
    ap.add_argument("--field-radius-m", type=float, default=100.0,
                    help="The realism field's half-width (the driver's h1_field_radius_m, "
                         "default 100 m).")
    ap.add_argument("--field-ref-n", type=int, default=None,
                    help="Exp 5 addendum: grow the field with N at this size's density "
                         "(half-width field-radius-m * sqrt(N / ref-n); the driver's "
                         "h1_field_ref_n). Default: the fixed field.")
    ap.add_argument("--far-share", type=float, default=None,
                    help="Exp 5 addendum (unit U10): the share of each layout's devices "
                         "placed beyond --rrf of the dock (the driver's far_share). "
                         "Default: the recorded uniform draw.")
    ap.add_argument("--member-admission", choices=MEMBER_ADMISSIONS,
                    default=MEMBER_ADMISSION_SUBSET,
                    help="subset (the F family's, default) or whole stops.")
    ap.add_argument("--families", nargs="+", default=None,
                    help="Arm families (default: F and FB+<class> for every class).")
    ap.add_argument("--exact-max-devices", type=int, default=EXACT_MAX_DEVICES,
                    help="Exact set cover up to this many devices, greedy above.")
    ap.add_argument("--layout-tag", default=LAYOUT_TAG,
                    help="Seed tag of the reference layouts (default s_star).")
    ap.add_argument("--layout-offset", type=int, default=0,
                    help="First layout index (the design probe used tag t_nom, offset 1000).")
    ap.add_argument("--json", default=None, help="Also write the full report here.")
    args = ap.parse_args(argv)
    physics = {}
    if args.ferry_physics:
        try:
            physics = json.loads(args.ferry_physics)
        except json.JSONDecodeError as e:
            ap.error(f"--ferry-physics: {e}")
    # The Exp 5 addendum's field settings reach the driver only when given, so
    # the default tool builds the driver it always built.
    field_kw: Dict[str, Any] = {}
    if args.field_radius_m != 100.0:
        field_kw["h1_field_radius_m"] = float(args.field_radius_m)
    if args.field_ref_n is not None:
        field_kw["h1_field_ref_n"] = args.field_ref_n
    if args.far_share is not None:
        field_kw["far_share"] = float(args.far_share)
    try:
        driver = Exp4Driver(
            mission_clock="sim", realism=not args.no_realism, contact_band=args.contact_band,
            contact_band_classes=args.contact_band_classes, backhaul_model=args.backhaul_model,
            payload_bytes=args.payload_bytes, ferry_physics=physics, **field_kw,
        )
        report = s_star_report(
            driver, n_devices=args.N, budgets=args.budgets, rf_range_m=args.rrf,
            regime=args.regime, families=args.families, admission=args.member_admission,
            layouts=args.layouts, theta_bytes=args.theta_bytes,
            exact_max_devices=args.exact_max_devices, layout_tag=args.layout_tag,
            layout_offset=args.layout_offset,
        )
    except ValueError as e:
        ap.error(str(e))
    print(format_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report.to_json(), f, indent=1, sort_keys=True)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
