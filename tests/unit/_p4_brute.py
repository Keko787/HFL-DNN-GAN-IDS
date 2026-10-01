"""FeRRy Phase 4 (unit U4): an independent brute force of the plan search.

Critic A6: the design's reference shared the search's fold, so a test against
it checked the enumeration and nothing else, and could not see that the search
missed the optimum over the member subsets V prices. This reference shares no
code with the search: not ``plan_search``, not U3's member walk
(``member_subset``), not U2's ``score``, not U1's cap rules (``s3d_age_cap``).
Its only oracle is S3b's predicate, ``FeasibilityModel.fold`` without
skipping, which defines what "admitted" means; everything else is written out
here from the spec:

* a stop reduced to some of its members keeps its position, lists them in the
  stop's order, and carries the earliest deadline of its uncapped members, or
  of all of them when every member is capped, in which case it is exempt from
  its own deadline clause (Phase 4 spec, other choices 5; critic B1, B2);
* V = −[c₁(Δ/T)² + c₂U + c₃L] − c₄E/(P_hover·T), with Δ the whole mission on
  the class (Pass 1, the turnaround, and Pass 2 when anyone is served) and E
  both passes' energy with their return legs (other choices 7; decision 2 (b));
* the cap key is the ages of the capped devices left out, largest first, and
  the plan key is (cap key, −round(V, 9), class index, each stop's
  (position, devices)) (other choices 3 and 5), with −round(share, 9) after
  the cap key under ``coverage_rank = lexicographic`` when κ > 0, the share
  being the served devices' weight over the demand's (1 for an empty demand;
  the orchestrator's resolution R11). F-cov (κ = 0) ranks without it under
  either setting.

:func:`exact_plans` is ``itertools`` over class × device subset × every order
of the stops the subset touches: the family the exact search must be optimal
over. :func:`stop_subset_plans` is ``itertools`` over the ordered stop subsets
of a class, each stop walked whole if it fits, else reduced in the F member
order (capped first, weight per second of dwell, device id), skip not stop:
the depth-first search's own family above 6 devices. :func:`walk` flies any
such route, for the local search's neighbourhoods. :func:`priority_start` is
the local search's start under ``whole``, and :func:`local_search` replays the
local search's documented scans with this module's own walk and V (under
``lexicographic``, a scan on the key without the share first, then scans on
the plan key), so the tests can hold the search to the order it scans its
moves in, the neighbours it skips or reuses and the walks and passes it
counts.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple

from hermes.scheduler.stages.s3b_feasibility import (
    RULE_DEADLINE_BUDGET,
    RULE_NONE,
    FeasibilityModel,
    FlightState,
)
from hermes.types import BUCKET_PRIORITY, ContactWaypoint, MissionPass

COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER


@dataclass(frozen=True)
class Pass2:
    """Pass 2 as this reference prices it."""

    time_s: float
    dwell_s: float
    energy_j: float


@dataclass
class BruteClass:
    """One class: its name, index, model (bound to the device states), outage
    callable, S3a stops (whole) and Pass-2 queue."""

    name: str
    index: int
    model: FeasibilityModel
    outage: Callable[[float], float]
    stops: Sequence[ContactWaypoint]
    pass_2_queue: Sequence[ContactWaypoint] = ()
    pass_2: Pass2 = field(init=False)

    def __post_init__(self) -> None:
        self.pass_2 = pass_two(self.model, self.pass_2_queue)


@dataclass
class World:
    """Everything a plan is priced with, in plain values."""

    classes: Sequence[BruteClass]
    start: FlightState
    budget_end: Optional[float]
    deadlines: Mapping[Any, float]
    device_states: Mapping[Any, Any]
    ages: Mapping[Any, int]
    capped: FrozenSet[Any]
    weights: Mapping[Any, float]
    c_time: float = 1.0
    kappa: float = 1.0
    c_link: Optional[float] = None
    c_energy: float = 0.1
    dwell_in_delta: bool = True
    t_ref_s: float = 100.0
    turnaround_s: float = 30.0
    p_hover_w: float = 168.5
    whole: bool = False
    coverage_rank: str = "lexicographic"


@dataclass(frozen=True)
class Plan:
    """One admitted plan and its price."""

    key: Tuple[Any, ...]
    v: float
    cap_key: Tuple[int, ...]
    index: int
    band: str
    route: Tuple[ContactWaypoint, ...]
    served: FrozenSet[Any]
    share: float = 0.0

    @property
    def stops(self) -> Tuple[Tuple[Tuple[float, ...], Tuple[Any, ...]], ...]:
        return tuple((tuple(wp.position), tuple(wp.devices)) for wp in self.route)


# --------------------------------------------------------------------------- #
# Stops
# --------------------------------------------------------------------------- #

def stop(wp: ContactWaypoint, members: Sequence[Any], world: World) -> ContactWaypoint:
    """``wp`` with only ``members``, rebuilt from scratch every time."""
    keep = set(members)
    devices = tuple(d for d in wp.devices if d in keep)
    assert devices and len(devices) == len(keep), (members, wp.devices)
    uncapped = [d for d in devices if d not in world.capped]
    deadline = min(world.deadlines.get(d, math.inf) for d in (uncapped or devices))
    buckets = [getattr(world.device_states[d], "bucket", None) for d in devices]
    buckets = [b for b in buckets if b is not None]
    bucket = min(buckets, key=BUCKET_PRIORITY.index) if buckets else wp.bucket
    return ContactWaypoint(position=wp.position, devices=devices, bucket=bucket,
                           deadline_ts=deadline)


def exempt(wp: ContactWaypoint, capped: FrozenSet[Any]) -> bool:
    return bool(capped) and all(d in capped for d in wp.devices)


def _position(world: World, did: Any) -> Sequence[float]:
    return world.device_states[did].last_known_position


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #

def pass_two(model: FeasibilityModel, queue: Sequence[ContactWaypoint]) -> Pass2:
    """From the dock at 0, no gate, delivering; energy with the return leg."""
    dock = model.ferry.dock
    walk = model.fold(list(queue), FlightState(dock, 0.0), rule=RULE_NONE, budget_end=None,
                      pass_kind=DELIVER, skip=False)
    dwell = sum(v.finish - v.arrival for v in walk.verdicts)
    back = math.dist(walk.state.pose, dock) / model.cruise_speed_m_s
    return Pass2(walk.home, dwell, walk.state.energy_j + model.ferry.p_move_w * back)


def cap_key(world: World, served: FrozenSet[Any]) -> Tuple[int, ...]:
    return tuple(sorted((world.ages[d] for d in world.capped if d not in served), reverse=True))


def price(world: World, cls: BruteClass, route: Sequence[ContactWaypoint], fold: Any) -> Plan:
    """This reference's own V of a plan the plain fold admitted."""
    served = frozenset(d for wp in route for d in wp.devices)
    model, ferry = cls.model, cls.model.ferry
    pass_1 = fold.home - world.start.clock
    dwell_1 = sum(v.finish - v.arrival for v in fold.verdicts)
    back = math.dist(fold.state.pose, ferry.dock) / model.cruise_speed_m_s
    energy_1 = fold.state.energy_j + ferry.p_move_w * back
    flown = bool(served)
    p2 = cls.pass_2
    delta = pass_1 + world.turnaround_s + (p2.time_s if flown else 0.0)
    if not world.dwell_in_delta:
        delta = max(0.0, delta - dwell_1 - (p2.dwell_s if flown else 0.0))
    energy = energy_1 + (p2.energy_j if flown else 0.0)
    total = sum(world.weights.values())
    got = sum(world.weights[d] for d in served)
    lost = 0.0
    for wp in route:
        for d in wp.devices:
            lost += world.weights[d] * cls.outage(math.dist(wp.position, _position(world, d)))
    coverage = 1.0 - got / total if total > 0 else 0.0
    link = lost / total if total > 0 else 0.0
    c2 = world.kappa * len(world.weights)
    c3 = c2 if world.c_link is None else world.c_link
    energy_term = energy / (world.p_hover_w * world.t_ref_s) if world.p_hover_w > 0 else 0.0
    v = -(world.c_time * (delta / world.t_ref_s) ** 2 + c2 * coverage + c3 * link) \
        - world.c_energy * energy_term
    ck = cap_key(world, served)
    stops = tuple((tuple(wp.position), tuple(wp.devices)) for wp in route)
    share = got / total if total > 0 else 1.0
    if world.coverage_rank == "lexicographic" and world.kappa > 0:
        key = (ck, -round(share, 9), -round(v, 9), cls.index, stops)
    else:
        assert world.coverage_rank in ("lexicographic", "weighted"), world.coverage_rank
        key = (ck, -round(v, 9), cls.index, stops)
    return Plan(key=key, v=v, cap_key=ck, index=cls.index, band=cls.name, route=tuple(route),
                served=served, share=share)


def admitted(world: World, cls: BruteClass, route: Sequence[ContactWaypoint]) -> Optional[Any]:
    """The plain fold of ``route`` from takeoff, without skipping; None unless it passes."""
    fold = cls.model.fold(list(route), world.start, rule=RULE_DEADLINE_BUDGET,
                          budget_end=world.budget_end, pass_kind=COLLECT, skip=False,
                          protected=frozenset(wp for wp in route if exempt(wp, world.capped)))
    return fold if fold.ok else None


# --------------------------------------------------------------------------- #
# The exact family: class x device subset x orders of the touched stops
# --------------------------------------------------------------------------- #

def exact_plans(world: World) -> Iterator[Plan]:
    """Every admitted plan of the exact family, the empty plan of each class included."""
    demand = sorted(world.weights)
    for cls in world.classes:
        for r in range(len(demand) + 1):
            for chosen in itertools.combinations(demand, r):
                want = set(chosen)
                touched = [wp for wp in cls.stops if want & set(wp.devices)]
                if world.whole and any(not set(wp.devices) <= want for wp in touched):
                    continue
                reduced = [stop(wp, [d for d in wp.devices if d in want], world)
                           for wp in touched]
                for route in itertools.permutations(reduced):
                    fold = admitted(world, cls, route)
                    if fold is not None:
                        yield price(world, cls, route, fold)


def best(plans: Iterator[Plan]) -> Plan:
    return min(plans, key=lambda p: p.key)


# --------------------------------------------------------------------------- #
# The depth-first search's family: ordered stop subsets, greedy members
# --------------------------------------------------------------------------- #

def _worth(weight: float, dwell: float) -> float:
    if dwell > 0.0:
        return weight / dwell
    return math.inf if weight > 0.0 else 0.0


def f_order(world: World, cls: BruteClass, wp: ContactWaypoint) -> List[Any]:
    """Capped first, then weight per second of dwell (the member alone at the
    stop), highest first, then the device id."""
    def key(did: Any) -> Tuple[bool, float, Any]:
        alone = ContactWaypoint(position=wp.position, devices=(did,), bucket=wp.bucket,
                                deadline_ts=wp.deadline_ts)
        dwell = cls.model.leg(wp.position, alone, pass_kind=COLLECT).dwell_s
        return (did not in world.capped, -_worth(world.weights.get(did, 1.0), dwell), did)
    return sorted(wp.devices, key=key)


def greedy(world: World, cls: BruteClass, state: FlightState,
           wp: ContactWaypoint) -> Optional[Tuple[ContactWaypoint, Any]]:
    """One stop from ``state``: whole if it fits, else (subsets) its members in
    the F order, each kept while the stop still passes, skip not stop; None
    when nobody fits."""
    def admit(trial: ContactWaypoint) -> Any:
        return cls.model.admit(state, trial, rule=RULE_DEADLINE_BUDGET,
                               budget_end=world.budget_end, pass_kind=COLLECT,
                               protected=exempt(trial, world.capped))

    whole = stop(wp, wp.devices, world)
    v = admit(whole)
    if v.ok:
        return whole, v
    if world.whole:
        return None
    kept: List[Any] = []
    got = None
    for did in f_order(world, cls, wp):
        trial = stop(wp, kept + [did], world)
        v = admit(trial)
        if v.ok:
            kept.append(did)
            got = (trial, v)
    return got


def walk(world: World, cls: BruteClass,
         entries: Sequence[Tuple[ContactWaypoint, Sequence[Any]]]) -> Optional[Plan]:
    """Fly ``entries`` (a stop and the members it may serve) in order, each
    stop greedily; None when one admits nobody. The route is checked again by
    the plain fold before it is priced."""
    state = world.start
    route: List[ContactWaypoint] = []
    for wp, members in entries:
        got = greedy(world, cls, state, stop(wp, members, world))
        if got is None:
            return None
        route.append(got[0])
        state = got[1].next_state
    fold = admitted(world, cls, route)
    assert fold is not None, route
    return price(world, cls, route, fold)


def stop_subset_plans(world: World, cls: BruteClass) -> Iterator[Plan]:
    """Every admitted plan of ``cls``'s ordered stop subsets, the empty one included."""
    for r in range(len(cls.stops) + 1):
        for chosen in itertools.combinations(cls.stops, r):
            for route in itertools.permutations(chosen):
                plan = walk(world, cls, [(wp, wp.devices) for wp in route])
                if plan is not None:
                    yield plan


# --------------------------------------------------------------------------- #
# The local search: its start under ``whole``, and its scan replayed
# --------------------------------------------------------------------------- #

Entries = Tuple[Tuple[ContactWaypoint, Tuple[Any, ...]], ...]


def priority_start(world: World, cls: BruteClass,
                   tour: Sequence[ContactWaypoint]) -> Tuple[ContactWaypoint, ...]:
    """The local search's start under ``whole``: ``tour`` with every stop that
    holds a capped member first, each part in tour order, folded from takeoff
    with skipping, each stop whose members are all capped exempt from its own
    deadline (Phase 4 spec, other choices 5)."""
    first = [wp for wp in tour if world.capped & set(wp.devices)]
    rest = [wp for wp in tour if not world.capped & set(wp.devices)]
    fold = cls.model.fold(first + rest, world.start, rule=RULE_DEADLINE_BUDGET,
                          budget_end=world.budget_end, pass_kind=COLLECT, skip=True,
                          protected=frozenset(wp for wp in first if exempt(wp, world.capped)))
    return fold.route


@dataclass(frozen=True)
class LocalRun:
    """What :func:`local_search` did: the best plan it met under the plan key
    (the empty plan included), the passes it began, the walks it ran (the
    start's included), whether a bound ended it, the plans it scored (the
    empty plan once), the kind of each move it took, how many neighbours it
    skipped as seen, how many a scan took from an earlier one's walk, the
    start of each scan it began (as entries: (stop, members) pairs) and the
    plan it ended on, and, when its first scan moved on
    :func:`weighted_key`, the passes and walks counted when that scan ended
    (the start's walk included)."""

    plan: Plan
    passes: int
    evaluations: int
    bounded: bool
    candidates: int
    taken: Tuple[str, ...]
    skipped: int
    reused: int = 0
    starts: Tuple[Any, ...] = ()
    ends: Tuple[Plan, ...] = ()
    first: Optional[Tuple[int, int]] = None


def weighted_key(plan: Plan) -> Tuple[Any, ...]:
    """The plan key without the share: (cap key, −round(V, 9), class index, stops)."""
    return (plan.cap_key, -round(plan.v, 9), plan.index, plan.stops)


def local_search(world: World, cls: BruteClass, start: Sequence[ContactWaypoint], *,
                 max_passes: int, max_evaluations: int, single: bool = False) -> LocalRun:
    """The local search's scans as the plan search documents them, from
    ``start`` (a route of stops cut from ``cls.stops``), each neighbour flown
    by :func:`walk` and priced by :func:`price`.

    A scan's pass scans the neighbours of the current route in this order and
    moves to the first whose key is smaller: drop a stop (by its place in the
    route); insert an unrouted stop with all its members (the stops in
    (position, devices) order, each at every place from the front); reverse a
    segment of two stops or more (by its first stop, then its last); and,
    under subset admission, drop one served member of a stop that serves
    more than one (stop by stop, least worth first in the F order). The
    current route is the route as flown. A neighbour the scan walked before is
    skipped without a walk. A scan ends at a local optimum, after
    ``max_passes`` passes, or when a walk would exceed ``max_evaluations``,
    both counted over every scan the search runs.

    One scan runs, on the plan key from ``start``, unless the rank is
    lexicographic with κ > 0 (and not ``single``, which replays that one scan
    alone): then the first scan moves on :func:`weighted_key`, a second on the
    plan key from the best plan met so far (under the plan key), and a third,
    when that plan's route is not ``start``'s, on the plan key from
    ``start``, each while no bound has ended the search. A later scan does not
    walk a neighbour an earlier one walked: it takes that walk's plan, and
    counts no walk.
    """
    order = sorted(cls.stops, key=lambda wp: (tuple(wp.position), tuple(wp.devices)))
    empty = price(world, cls, (), admitted(world, cls, ()))
    worst_first = {wp: tuple(reversed(f_order(world, cls, wp))) for wp in cls.stops}

    def entries_of(route: Sequence[ContactWaypoint]) -> Entries:
        return tuple((next(s for s in cls.stops if wp.devices[0] in s.devices), tuple(wp.devices))
                     for wp in route)

    def fly(entries: Entries) -> Optional[Plan]:
        return walk(world, cls, entries) if entries else empty

    def moves(entries: Entries) -> Iterator[Tuple[str, Entries]]:
        k = len(entries)
        routed = {wp for wp, _ in entries}
        for p in range(k):
            yield "drop", entries[:p] + entries[p + 1:]
        for wp in order:
            if wp not in routed:
                for p in range(k + 1):
                    yield "insert", entries[:p] + ((wp, tuple(wp.devices)),) + entries[p:]
        for a in range(k - 1):
            for b in range(a + 1, k):
                yield "reverse", entries[:a] + entries[a:b + 1][::-1] + entries[b + 1:]
        if not world.whole:
            for p, (wp, members) in enumerate(entries):
                for did in worst_first[wp]:
                    if did in members and len(members) > 1:
                        rest = tuple(m for m in members if m != did)
                        yield "member", entries[:p] + ((wp, rest),) + entries[p + 1:]

    def plan_key(plan: Plan) -> Tuple[Any, ...]:
        return plan.key

    begin = fly(entries_of(start))
    assert begin is not None and begin.stops == tuple(
        (tuple(wp.position), tuple(wp.devices)) for wp in start), "the start must fly as given"
    walked = {entries_of(start): begin}
    tally = {"evaluations": 1, "passes": 0, "skipped": 0, "reused": 0,
             "candidates": 1 + bool(begin.route)}
    state = {"best": min(empty, begin, key=plan_key), "bounded": False}
    taken: List[str] = []
    starts: List[Entries] = []
    ends: List[Plan] = []

    def scan(here: Plan, key: Callable[[Plan], Tuple[Any, ...]]) -> None:
        current = entries_of(here.route)
        starts.append(current)
        seen = {current}
        improved = True
        while improved:
            if tally["passes"] == max_passes:
                state["bounded"] = True
                break
            tally["passes"] += 1
            improved = False
            for kind, entries in moves(current):
                if entries in seen:
                    tally["skipped"] += 1
                    continue
                if entries in walked:
                    plan = walked[entries]
                    tally["reused"] += 1
                else:
                    if tally["evaluations"] == max_evaluations:
                        state["bounded"] = True
                        break
                    tally["evaluations"] += 1
                    plan = walked[entries] = fly(entries)
                    if plan is not None:
                        tally["candidates"] += bool(plan.route)
                        state["best"] = min(state["best"], plan, key=plan_key)
                seen.add(entries)
                if plan is not None and key(plan) < key(here):
                    here, current, improved = plan, entries_of(plan.route), True
                    seen.add(current)
                    taken.append(kind)
                    break
            if state["bounded"]:
                break
        ends.append(here)

    first = None
    if single or world.coverage_rank != "lexicographic" or world.kappa <= 0:
        scan(begin, plan_key)
    else:
        scan(begin, weighted_key)
        first = (tally["passes"], tally["evaluations"])
        if not state["bounded"]:
            polish = state["best"]
            scan(polish, plan_key)
            if not state["bounded"] and entries_of(polish.route) != entries_of(begin.route):
                scan(begin, plan_key)
    return LocalRun(plan=state["best"], passes=tally["passes"],
                    evaluations=tally["evaluations"], bounded=state["bounded"],
                    candidates=tally["candidates"], taken=tuple(taken),
                    skipped=tally["skipped"], reused=tally["reused"], starts=tuple(starts),
                    ends=tuple(ends), first=first)
