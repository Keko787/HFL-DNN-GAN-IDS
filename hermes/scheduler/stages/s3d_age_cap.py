"""Stage 3d — the age cap S (FeRRy Phase 4, build-plan decision D4).

**Why this stage exists.** Nothing else bounds how long a device can go without
its update reaching the model: its deadline window has no ceiling, and past it
the device is dropped as overdue mission after mission (build plan L706, CfgRef
§15). In plan mode (``plan_mode = "ferry"``) the age cap does: once a device
has gone S of its mule's missions without a merged update it is capped, the
plan serves the oldest capped devices that fit before it weighs anything else,
and every capped device a mission fails is logged with the cause (build plan
L831, L929-932, L1196). Each rule of the cap is one function here, and the
plan path calls them: the ages and the capped set
(:func:`evaluate_cap`), the stop sets the predicate and the trims read
(:func:`cap_stops`, :func:`stop_deadline`), the cap key the plan key compares
first (:func:`cap_key`), and the violations (:func:`plan_violations`,
:func:`close_commit`). Not to be confused with the merge cutoff a_max
(``aggregation_rules.age_cap``, ``MuleSupervisor._age_caps``), which counts
cluster rounds.

**The age** (the user's decision 1, 2026-09-30). ``a_j(m) = m − U_j``, where m
is the mission being planned (``FLScheduler.mission_round``: the mule sets it
before every plan from ``HFLHostMission.open_round``, which numbers the first
mission 1) and U_j is the mission whose merge last used j's update
(``last_merged_round``, which ``FLScheduler.record_merged`` sets when the
mission closes), 0 when none has. So a_j(m) counts j's own mule's missions
since its last merged update, the mission being planned included, and it
equals the scorer's age of j after mission m if mission m does not merge j
(``traces_scorer.age_profile``: ``m − U`` after m's merges). A device is
therefore capped (with L = 0) exactly when leaving it unmerged now puts it at
age >= S in the scorer, and a never-merged device is capped from mission S on
(critic A11 (i)). This is D3's x (``whittle.py``, deviation 5) without its
fallbacks: no ``last_clean_round`` (a CLEAN the merge cutoff excluded is not
service), and no clamp to 1 (the age is 0 right after a merge, as in the
scorer). The mule cannot see backhaul loss or cluster-side deferral, so the
scorer can still count what the mule does not (design R5, critic B10).

**The cap and the stops** (Phase 4 spec, other choices 5). A device is capped
when its age reaches S − L (``AgeCapSpec.caps``; the lookahead L defaults to
0). On any route, whole or reduced to member subsets, a stop is:

* **exempt** when every member is capped. It skips its own deadline clause,
  the predicate's ``protected`` (``FeasibilityModel.admit``). It keeps its
  members' earliest deadline, finite, because the trace writes it as a float
  (``_pass_1_plan_payload``, processes/mule.py);
* **mixed** when some members are capped and some are not. It is not exempt:
  it carries the earliest deadline among its uncapped members (critic B2), so
  the capped members' lateness is excused and the uncapped members' is not.
  Protecting it whole would let the uncapped members miss their deadlines;
* **priority** when any member is capped (exempt or mixed). It goes first in a
  trim and sheds its uncapped members first.

``ContactWaypoint`` equality covers ``devices`` and ``deadline_ts``, and
``FeasibilityModel.fold`` tests membership in ``protected`` by value, so the
sets are computed on the route actually folded, never carried over from S3a's
stops (critic B1): a reduced stop is a new key. For the same reason U3's
member walk (``plan/member_subset.py``), which builds and admits the reduced
stops, must follow these rules by value: the guard fold checks its route with
the protected set :func:`cap_stops` gives.

**The cap key** (critic C5) ranks plans by the ages of the capped devices they
leave out, largest first, and the plan key compares it before V
(``Candidate.key``). It minimises the oldest unserved age, not the violation
count, as the plan's fallback "keeps the oldest capped devices that fit"
(build plan L1196).

**Violations** (Phase 4 spec, other choices 6), reported by cause with no pass
mark: device availability alone makes about 15 % of device-missions miss at
S = 3 (critic A2). When the plan is built, each capped device it leaves out is
``unplannable`` (no class the arm may fly serves it alone within the budget,
even at its best hover point: physics without an energy capacity, the user's
decision of 2026-09-30; with one, see :func:`servable_alone`) or ``crowded``
(one could, but the chosen plan does not). The plan path offers each capped
device that its S3a stop cannot serve alone a stop of its own at that point
(``plan/hover.py``), so ``servable_alone`` judges it there. When the mission
closes, each capped device the plan served is
``dropped_in_flight`` (no stop flown held it: its stop was dropped, or cut
without it) or ``not_merged`` (visited, but its update did not reach the
merge); the latter can pin a device whose availability draw fails (critic
C3).

Freeze Rule 1: only the plan path calls this module, and no legacy module
imports it, so the recorded pipelines never reach it. Like S3b it is
numpy-free and imports nothing from ``hermes.l1`` or ``experiments``: it reads
the plan's types and S3b's predicate only.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, replace
from typing import Any, Collection, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

from hermes.scheduler.plan.types import AgeCapSpec, CapState
from hermes.scheduler.stages.s3b_feasibility import (
    RULE_DEADLINE_BUDGET,
    FeasibilityModel,
    FlightState,
)
from hermes.types.ids import DeviceID
from hermes.types.scheduler import (
    CAP_CROWDED,
    CAP_DROPPED_IN_FLIGHT,
    CAP_NOT_MERGED,
    CAP_REASONS,
    CAP_UNPLANNABLE,
    CapViolation,
    ContactWaypoint,
    DeviceSchedulerState,
    PlanCommit,
)

_NO_ROUND = (
    "the age counts the mule's missions since each device's last merged update, so it "
    "needs the mission being planned (FLScheduler.mission_round, which the mule sets "
    "before every plan); it is None (critic B9)"
)


def _count(value: Any, name: str) -> int:
    """``value`` as an int >= 0; a bool is refused (``True`` is no count)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return int(value)


def _ids(values: Iterable[Any], name: str) -> Tuple[DeviceID, ...]:
    """Device ids in the given order; one string is refused (it is one id, not many)."""
    if isinstance(values, str):
        raise TypeError(f"{name} is a collection of device ids, not one string: {values!r}")
    out = tuple(values)
    for did in out:
        if not isinstance(did, str) or not did:
            raise TypeError(f"{name} holds device ids (non-empty strings), got {did!r}")
    return out  # type: ignore[return-value]


def _in_reason_order(violations: Iterable[CapViolation]) -> Tuple[CapViolation, ...]:
    """The commit's order: by reason (``CAP_REASONS``), then by device."""
    return tuple(sorted(violations, key=lambda v: (CAP_REASONS.index(v.reason), v.device)))


# --------------------------------------------------------------------------- #
# Ages and the capped set
# --------------------------------------------------------------------------- #

def device_age(state: DeviceSchedulerState, mission_round: Optional[int]) -> int:
    """The device's age when mission ``mission_round`` is planned: ``m − U``.

    U is ``state.last_merged_round``, the mission whose merge last used the
    device's update, and 0 when none has (None). The module docstring says why
    this is the scorer's age after mission m if m does not merge the device.
    U = m, a merge in the mission being planned, gives 0, the scorer's age
    right after a merge; before a plan it cannot happen, because the mule
    records the merge when the mission closes (``FLScheduler.record_merged``).

    Refused: no round (critic B9), and a merge recorded after the mission
    being planned (U > m: the rounds went backwards). There is no fallback to
    ``last_clean_round`` (the user's decision 1), so a state without the field
    is refused rather than read as never merged.
    """
    if mission_round is None:
        raise ValueError(_NO_ROUND)
    m = _count(mission_round, "mission_round")
    try:
        merged = state.last_merged_round
    except AttributeError:
        raise TypeError(
            f"the age reads last_merged_round (DeviceSchedulerState), which a "
            f"{type(state).__name__} lacks"
        ) from None
    u = 0 if merged is None else _count(merged, "last_merged_round")
    if u > m:
        raise ValueError(
            f"device {getattr(state, 'device_id', '?')!r} was last merged in mission {u}, "
            f"after mission {m} being planned: the mission rounds went backwards"
        )
    return m - u


def evaluate_cap(
    demand: Iterable[DeviceID],
    device_states: Mapping[DeviceID, DeviceSchedulerState],
    *,
    mission_round: Optional[int],
    spec: AgeCapSpec,
) -> CapState:
    """The cap when the plan is built: each demanded device's age, and the capped set.

    ``demand`` is the plan's demand, the devices left after S1 and S3, in S3's
    order, which the ages keep. The ages are computed with the cap off too,
    because the ``age`` coverage weights read them, so the round is needed
    whatever ``spec`` says (critic B9). ``CapState`` derives the capped set
    from the ages with ``spec`` (age >= S − L), so the promotion rule has one
    definition.
    """
    if not isinstance(spec, AgeCapSpec):
        raise TypeError(f"spec must be an AgeCapSpec, got {spec!r}")
    if mission_round is None:
        raise ValueError(_NO_ROUND)
    ids = _ids(demand, "demand")
    if len(set(ids)) != len(ids):
        raise ValueError(f"demand lists each device once, got {ids!r}")
    missing = [d for d in ids if d not in device_states]
    if missing:
        raise KeyError(f"no device state for the demanded device(s) {missing}")
    return CapState(spec=spec, ages={d: device_age(device_states[d], mission_round) for d in ids})


# --------------------------------------------------------------------------- #
# The stop sets and deadlines
# --------------------------------------------------------------------------- #

def is_exempt(wp: ContactWaypoint, capped: Collection[DeviceID]) -> bool:
    """True when every member of ``wp`` is capped: the stop skips its own deadline
    clause (``FeasibilityModel.admit(protected=True)``). A stop has at least one
    member, so nothing is exempt without a cap."""
    return all(d in capped for d in wp.devices)


def is_priority(wp: ContactWaypoint, capped: Collection[DeviceID]) -> bool:
    """True when any member of ``wp`` is capped: the stop goes first in a trim."""
    return any(d in capped for d in wp.devices)


@dataclass(frozen=True)
class CapStops:
    """The cap's stop sets on one route (:func:`cap_stops`), as sets of waypoints.

    ``exempt`` is what the fold takes as ``protected``; ``priority`` (exempt
    or mixed) is what a trim puts first. Waypoint values, which is how
    ``FeasibilityModel.fold`` tests membership.
    """

    exempt: FrozenSet[ContactWaypoint] = frozenset()
    mixed: FrozenSet[ContactWaypoint] = frozenset()

    def __post_init__(self) -> None:
        for name in ("exempt", "mixed"):
            stops = frozenset(getattr(self, name))
            for wp in stops:
                if not isinstance(wp, ContactWaypoint):
                    raise TypeError(f"{name} holds ContactWaypoints, got {wp!r}")
            object.__setattr__(self, name, stops)
        both = self.exempt & self.mixed
        if both:
            raise ValueError(
                f"a stop is exempt (every member capped) or mixed, not both: "
                f"{sorted(tuple(wp.devices) for wp in both)}"
            )

    @property
    def priority(self) -> FrozenSet[ContactWaypoint]:
        """Every stop holding a capped member."""
        return self.exempt | self.mixed


def cap_stops(route: Iterable[ContactWaypoint], capped: Collection[DeviceID]) -> CapStops:
    """The exempt and mixed stops of ``route`` (and so its priority stops).

    Compute them on the route that is folded, after every reduction, and
    again on every reduced route (critic B1): a stop reduced to a subset of
    its members is a different waypoint, and a mixed stop that sheds its
    uncapped members becomes exempt.
    """
    exempt, mixed = set(), set()
    for wp in route:
        if not isinstance(wp, ContactWaypoint):
            raise TypeError(f"a route holds ContactWaypoints, got {wp!r}")
        if is_exempt(wp, capped):
            exempt.add(wp)
        elif is_priority(wp, capped):
            mixed.add(wp)
    return CapStops(exempt=frozenset(exempt), mixed=frozenset(mixed))


def stop_deadline(
    members: Iterable[DeviceID],
    *,
    deadlines: Mapping[DeviceID, float],
    capped: Collection[DeviceID] = frozenset(),
) -> float:
    """The deadline a stop with these ``members`` carries under the cap.

    The earliest own deadline among its uncapped members (a mixed stop,
    critic B2). When every member is capped, the earliest among all of them:
    the stop is exempt, so its own deadline clause never reads it, and the
    trace still gets a finite number. With nothing capped this is S3a's rule
    exactly (``cluster_by_rf_range``), a member without a deadline counting as
    ``inf`` as there. The one rule for whole S3a stops
    (:func:`with_cap_deadlines`) and for stops reduced to a subset of their
    members, U3's member walk included.
    """
    members = _ids(members, "members")
    if not members:
        raise ValueError("a stop has at least one member")
    uncapped = [d for d in members if d not in capped]
    return min(deadlines.get(d, math.inf) for d in (uncapped or members))


def with_cap_deadlines(
    stops: Iterable[ContactWaypoint],
    *,
    deadlines: Mapping[DeviceID, float],
    capped: Collection[DeviceID],
) -> Tuple[ContactWaypoint, ...]:
    """``stops`` (S3a's, whole), each with the deadline :func:`stop_deadline` gives it.

    A stop whose deadline does not change is returned itself, the same
    object. With the ``deadlines`` S3a clustered with, only a mixed stop whose
    earliest deadline belongs to a capped member moves, so without a cap the
    stops are exactly S3a's. Position, members and bucket (the worst of all
    members', S3a's rule) are kept. A stop whose deadline moves is a new key:
    compute :func:`cap_stops` on the stops returned here (critic B1).
    """
    out = []
    for wp in stops:
        if not isinstance(wp, ContactWaypoint):
            raise TypeError(f"stops hold ContactWaypoints, got {wp!r}")
        deadline = stop_deadline(wp.devices, deadlines=deadlines, capped=capped)
        out.append(wp if deadline == wp.deadline_ts else replace(wp, deadline_ts=deadline))
    return tuple(out)


def priority_first(
    route: Iterable[ContactWaypoint], capped: Collection[DeviceID],
) -> Tuple[ContactWaypoint, ...]:
    """``route`` with its priority stops first; each part keeps the route's order.

    The order a trim walks (spec, other choices 5). It follows ``replan_route``,
    which admits its protected stops first "in their current relative order"
    (routing/replan.py), but every stop holding a capped member leads, not only
    the exempt ones, so an uncapped stop never spends the budget a capped
    member needs.
    """
    stops = tuple(route)
    return (tuple(wp for wp in stops if is_priority(wp, capped))
            + tuple(wp for wp in stops if not is_priority(wp, capped)))


def capped_first(
    members: Iterable[DeviceID], capped: Collection[DeviceID],
) -> Tuple[DeviceID, ...]:
    """``members`` with the capped ones first; each part keeps the given order.

    The cap's half of the member order (spec, other choices 4: capped first,
    then w/dwell descending, then device id): a stop reduced greedily in this
    order admits its capped members before any uncapped one, so a priority
    stop sheds its uncapped members first.
    """
    ids = _ids(members, "members")
    return (tuple(d for d in ids if d in capped)
            + tuple(d for d in ids if d not in capped))


# --------------------------------------------------------------------------- #
# The cap key
# --------------------------------------------------------------------------- #

def cap_key(served: Iterable[DeviceID], cap: CapState) -> Tuple[int, ...]:
    """The ages of the capped devices ``served`` leaves out, largest first.

    ``Candidate.cap_key``: the plan key compares it first, and the smaller
    tuple wins. So it minimises the oldest unserved capped age, then the next
    oldest, and only then the number left out at those ages: three left out at
    age 4 beat one at age 5, (4, 4, 4) < (5, 3) (critic C5), as the plan's
    fallback "keeps the oldest capped devices that fit" (L1196). It does not
    count violations. A plan that serves every capped device another serves,
    and more, has a strictly smaller key, so a plan that drops a capped device
    loses to any feasible one that keeps it and the rest (L831). Uncapped
    devices never enter it; with every capped device served it is ().
    ``cap`` may be a ``CapState`` or a ``PlanCommit`` (both hold the ages and
    the capped set).
    """
    if not isinstance(served, (set, frozenset)):
        served = frozenset(_ids(served, "served"))
    return tuple(sorted((cap.ages[d] for d in cap.capped if d not in served), reverse=True))


# --------------------------------------------------------------------------- #
# Violations
# --------------------------------------------------------------------------- #

def servable_alone(
    capped: Iterable[DeviceID],
    classes: Iterable[Tuple[FeasibilityModel, Sequence[ContactWaypoint]]],
    *,
    start: FlightState,
    budget_end: Optional[float],
) -> FrozenSet[DeviceID]:
    """The ``capped`` devices some class the arm may fly serves alone from ``start``.

    ``classes`` pairs each class the arm may fly (``PlanSetup.searched``:
    every class under ``search``, the pinned one under ``fixed:<c>``) with
    its model and the stops the search is offered on it, which place each
    demanded device once: S3a's stops with the hover rule
    (``plan/hover.py``), so a capped device that its S3a stop cannot serve
    alone sits in a stop of its own at its best hover point. A device is
    served alone on a class when its stop reduced to the device alone passes
    the predicate from ``start`` (the dock at the plan's takeoff clock, since
    ``budget_end`` is absolute) under the plan's rule,
    ``RULE_DEADLINE_BUDGET``, priced as Pass 1 like the plan's own folds: the
    member at its distance from the stop, and the upload in the time home.
    That stop is exempt (its only member is capped), so only the budget and
    energy clauses can refuse it, never a deadline, and at takeoff nothing is
    on board for the ``delivery`` clause to protect. It keeps the offered
    stop's position because every plan the search can fly is built from
    those stops, and a device refused there alone is refused in every plan of
    that class: other stops and members only add time and energy. The hover
    rule makes that physics for the time budget: the best hover point is no
    slower alone than any other point the class could fly. Not for an energy
    capacity: the point minimises time, and the energy clause's need weighs
    a second of dwell more than a second of flight, so a device the capacity
    refuses there can still be served alone from a slower point with a
    shorter dwell, which no stop offers it (``plan/hover.py``). The hover
    rule asks the same question of S3a's own stops, to decide who moves.
    Without a budget nothing is refused (no budget, no gate), so nothing is
    unplannable.
    """
    wanted = _ids(capped, "capped")
    out = set()
    for model, stops in classes:
        where = {}
        for wp in stops:
            if not isinstance(wp, ContactWaypoint):
                raise TypeError(f"a class's stops are ContactWaypoints, got {wp!r}")
            for d in wp.devices:
                if d in where:
                    raise ValueError(
                        f"device {d!r} is in two stops of one class; S3a places it once"
                    )
                where[d] = wp
        unplaced = [d for d in wanted if d not in where]
        if unplaced:
            raise ValueError(
                f"device(s) {unplaced} are in no stop of a class: pass each class's S3a "
                f"stops, which cover the demand"
            )
        for d in wanted:
            if d in out:
                continue
            wp = where[d]
            alone = ContactWaypoint(position=wp.position, devices=(d,), bucket=wp.bucket,
                                    deadline_ts=wp.deadline_ts)
            verdict = model.admit(start, alone, rule=RULE_DEADLINE_BUDGET,
                                  budget_end=budget_end, protected=True)
            if verdict.ok:
                out.add(d)
    return frozenset(out)


def plan_violations(
    cap: CapState, served: Iterable[DeviceID], *, servable: Iterable[DeviceID],
) -> Tuple[CapViolation, ...]:
    """One violation for each capped device the plan leaves out, with its planning age.

    ``unplannable`` when no class the arm may fly serves the device alone
    within the budget, from the dock, even at its best hover point (it is not
    in ``servable``, see :func:`servable_alone` on the offered stops),
    ``crowded`` when one could but the chosen plan does not (spec, other
    choices 6; the user's decision of 2026-09-30). ``served`` is the plan's
    served set, the members of its stops (``PlanCommit.served``). In the
    commit's order: by reason, then device.
    """
    if not isinstance(cap, CapState):
        raise TypeError(f"cap must be a CapState, got {cap!r}")
    served_set = frozenset(_ids(served, "served"))
    outside = sorted(served_set - set(cap.ages))
    if outside:
        raise ValueError(f"the plan serves devices outside its demand: {outside}")
    alone = frozenset(_ids(servable, "servable"))
    return _in_reason_order(
        CapViolation(d, cap.ages[d], CAP_CROWDED if d in alone else CAP_UNPLANNABLE)
        for d in cap.capped - served_set
    )


def visited_devices(flown: Iterable[ContactWaypoint]) -> FrozenSet[DeviceID]:
    """The plan's visited set (build plan L475): the members of the Pass-1 stops
    actually flown, after any in-flight reduction."""
    out = set()
    for wp in flown:
        if not isinstance(wp, ContactWaypoint):
            raise TypeError(f"flown holds the ContactWaypoints flown, got {wp!r}")
        out.update(wp.devices)
    return frozenset(out)


def close_violations(
    commit: PlanCommit, *, visited: Iterable[DeviceID], merged: Iterable[DeviceID],
) -> Tuple[CapViolation, ...]:
    """The close-time violations of an open ``commit``, with their planning ages.

    Of the capped devices the plan served: ``dropped_in_flight`` for each one
    not ``visited`` (the in-flight check, re-plan or trim left its stop out,
    or cut it from its stop), ``not_merged`` for each one visited whose
    update the mission's merge did not use (``merged``, what the mule passes
    to ``record_merged``). A capped device the plan left out already carries
    its plan-time violation and gets no other.
    """
    if not isinstance(commit, PlanCommit):
        raise TypeError(f"commit must be a PlanCommit, got {commit!r}")
    if commit.closed:
        raise ValueError("the plan is already closed")
    seen = frozenset(_ids(visited, "visited"))
    used = frozenset(_ids(merged, "merged"))
    held = commit.capped & commit.served
    return _in_reason_order(
        [CapViolation(d, commit.ages[d], CAP_DROPPED_IN_FLIGHT) for d in held - seen]
        + [CapViolation(d, commit.ages[d], CAP_NOT_MERGED) for d in (held & seen) - used]
    )


def close_commit(
    commit: PlanCommit, flown: Iterable[ContactWaypoint], merged: Iterable[DeviceID],
) -> PlanCommit:
    """``commit`` closed with the mission's visited set and close-time violations.

    What ``FLScheduler.close_plan(flown, merged)`` does (spec, other choices
    6): the mule calls it after ``record_merged`` with the Pass-1 stops it
    actually flew and the devices its merge used, and on the empty path (Pass
    1 collected nothing) with ``merged = ()``. The commit is frozen, so this
    returns the checked copy (``PlanCommit.close``) for the scheduler to keep
    as its ``last_plan``.
    """
    visited = visited_devices(flown)
    return commit.close(visited, close_violations(commit, visited=visited, merged=merged))
