"""FeRRy Phase 4: member-subset admission (unit U3).

**Why it exists.** S3a puts every device within R_planar(b) of a stop into one
contact, and every gate before Phase 4 admits a contact with all its members or
not at all. On narrow (and medium) with a declared payload one stop covers the
whole field, so under a budget below that stop's time the plan is empty and the
mule flies nothing: a cliff between 0 and N devices set by contact granularity,
not by the reach-against-dwell trade the band choice is about (Phase 3 final
check, finding E2E1-01, pinned in ``tests/unit/test_p3_final_fixes_mule.py``).
Here a stop that fails whole is re-issued with the members that still fit
(design D-D (a); the user's decision 4, which extends it to the H and D arms,
unit U3b).

**One reduction rule for every arm.** A reduced stop is an ordinary
``ContactWaypoint`` built by :func:`reduce_stop`: the stop's own position (S3a
placed it within range of every member, ``s3a_cluster.py:146-152``), the subset
as its devices in the stop's order, the worst of their buckets and the earliest
of their own deadlines, the rules S3a builds a stop with
(``s3a_cluster.py:154-169``). :func:`admit_members` is the member walk for one
stop: the members are tried in a given order and each is admitted when the stop
with it still passes the predicate, skipped otherwise (skip, not stop). The F
family passes :func:`member_order`; U3b passes each H and D arm's own order. So
every arm reduces stops by the same walk and differs only in the order.

**Why one pass is final.** From a fixed flight state every clause of the
predicate is monotone in a stop's member set: dwell is a sum of non-negative
member times (``FerryPhysics.dwell_s``), the deadline the stop is held to is a
minimum over its members, arrival, the return leg and the upload do not depend
on the members, and energy grows with dwell (``FeasibilityModel.admit``). A
member refused beside part of the subset stays refused beside any superset, so
one pass admits a maximal set for its order, and a stop that fits whole is
admitted whole by it. The cap below keeps this: adding a member can only end a
stop's exemption or lower its deadline.

**The age cap, F family only** (Phase 4 spec, other choices 4, 5 and 9; critic
B1, B2). The cap's stop rules are the age-cap stage's (U1,
``stages/s3d_age_cap.py``: ``stop_deadline``, ``is_exempt``, ``is_priority``),
imported here, not restated: the plan's guard fold takes its protected set from
that stage (``cap_stops``), and one definition cannot drift from the walk that
builds the stops it checks. With ``capped`` given:

* a stop all of whose members are capped is *exempt*: its own deadline clause
  is skipped (the predicate's ``protected``) and its ``deadline_ts`` stays its
  members' minimum, finite, as the trace records it;
* a *mixed* stop carries the earliest deadline among its uncapped members, and
  its clause runs: exempting the whole stop would let uncapped co-members miss
  their deadlines unchecked (critic B2). :func:`reduce_stop` gives a whole S3a
  stop that deadline too, so reduced and whole stops follow one rule;
* exemption is decided on each stop as reduced, every time: a reduced stop is a
  new waypoint (equality covers ``devices``, ``types/scheduler.py:224-230``), so
  a protected set computed on the S3a stops would not contain it (critic B1);
* capped members come first in the F order, so a *priority* stop (any member
  capped) sheds its uncapped members first, and the in-flight trim flies
  priority stops first and keeps every capped member the protected-only trim
  keeps (:func:`trim_members`);
* a priority stop none of whose members fits is dropped member by member, each
  with the reason the walk refused it, not whole (:func:`admit_stop`): whole, a
  mixed stop fails on its uncapped members' deadline, and that ``overdue`` is
  never a capped member's own (spec, other choices 8). The F order tries a
  capped member only beside capped members, so with it no drop labels a
  capped member ``overdue``.

With no capped member all of this is inert and :func:`admit_members` is exactly
the walk U3b's spec states (``unit_U3b.md`` section 1.5).

**Layering.** Numpy-free, and nothing from ``hermes.l1``, the mule or
``experiments``: physics reaches this module only through the
``FeasibilityModel``. At module level it imports only ``hermes.types``, S3b,
the age-cap stage and ``plan.types`` (the stage itself imports only
``plan.types``, S3b and ``hermes.types``); nothing under
``hermes.scheduler.plan`` imports the policies or the scheduler, and S3b and
the walks import this module only inside their subset code, so no import cycle
can form and the recorded ``whole`` path never loads it.
"""

from __future__ import annotations

import math
import numbers
from collections import Counter
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from hermes.types.ids import DeviceID
from hermes.types.scheduler import BUCKET_PRIORITY, Bucket, ContactWaypoint, MissionPass
from hermes.scheduler.stages.s3b_feasibility import (
    REASONS,
    RULE_DEADLINE_BUDGET,
    RULES,
    FeasibilityModel,
    FlightState,
    Verdict,
)
from hermes.scheduler.stages.s3d_age_cap import is_exempt, is_priority, stop_deadline

from .types import MemberFold

if TYPE_CHECKING:  # pragma: no cover - the name for the type checker only
    from hermes.scheduler.routing.replan import ReplanResult

# ``stop_deadline``, ``is_exempt`` and ``is_priority`` are U1's (the age-cap
# stage), used here by name; they are not part of this module's API.
__all__ = [
    "MemberOrder", "Veto",
    "reduce_stop", "member_order",
    "MemberWalk", "admit_members", "complements",
    "StopAdmission", "admit_stop", "fold_members", "trim_members", "pass_energy_j",
]

#: A member order for one stop: a permutation of ``wp.devices``, tried first
#: to last. :func:`member_order` is the F family's.
MemberOrder = Callable[[ContactWaypoint], Sequence[DeviceID]]

#: An extra test of a trial stop the predicate admitted, given that stop and
#: its verdict: None keeps it, a reason refuses it. The in-flight trim uses one
#: to keep a member from crowding a later capped member out.
Veto = Callable[[ContactWaypoint, Verdict], Optional[str]]

_COLLECT = MissionPass.COLLECT


def _as_set(capped: Collection[DeviceID]) -> FrozenSet[DeviceID]:
    return capped if isinstance(capped, frozenset) else frozenset(capped)


# --------------------------------------------------------------------------- #
# Reduced stops
# --------------------------------------------------------------------------- #

def _worst_bucket(
    devices: Sequence[DeviceID], device_states: Mapping[DeviceID, Any], fallback: Bucket,
) -> Bucket:
    """S3a's rule (``s3a_cluster.py:72-74, 154-161``): the members' worst bucket.

    A member without a state or a bucket does not count. When none has one (a
    stop the beacon hook inserted in flight can hold devices S3 never
    bucketed), the stop's own bucket, the worst over all its members, stands:
    S3a raises there, but a mission must not fail on a reduction.
    """
    buckets = [
        b for b in (getattr(device_states.get(d), "bucket", None) for d in devices)
        if b is not None
    ]
    return min(buckets, key=BUCKET_PRIORITY.index) if buckets else fallback


def reduce_stop(
    wp: ContactWaypoint,
    members: Iterable[DeviceID],
    *,
    deadlines: Mapping[DeviceID, float],
    device_states: Mapping[DeviceID, Any],
    capped: Collection[DeviceID] = frozenset(),
) -> ContactWaypoint:
    """``wp`` with only ``members``: an ordinary ``ContactWaypoint``.

    ``members`` must be a non-empty subset of ``wp.devices``, each once, else
    ValueError. A proper subset gives a new waypoint at ``wp.position`` with
    the subset as ``devices`` in ``wp``'s order, the worst of their buckets
    (``device_states``) and the earliest of their own deadlines, by the
    age-cap stage's ``stop_deadline`` (S3a's rule, ``s3a_cluster.py:169``, a
    member without a deadline counting as ``inf``; B2's under the cap).
    ``band``, ``range_m`` and ``pred_snr_db`` stay None, since they describe
    the stop as the mule annotated it after planning and ``pred_snr_db`` is
    per member.

    The full set gives ``wp`` itself, so a stop that needs no reduction keeps
    its identity (the walks and the re-plan compare stops by identity). Under
    the cap (``capped``, the F family only), a full stop with a capped member
    gets B2's deadline; it is ``wp`` itself when that is its deadline already,
    else a copy with that deadline and nothing else changed.
    """
    chosen = tuple(members)
    if not chosen:
        raise ValueError(f"reduce_stop: keep at least one member of {wp.devices!r}")
    wanted = set(chosen)
    if len(wanted) != len(chosen):
        raise ValueError(f"reduce_stop: members repeat in {chosen!r}")
    outside = wanted.difference(wp.devices)
    if outside:
        raise ValueError(
            f"reduce_stop: {sorted(str(d) for d in outside)} are not members of the "
            f"stop {wp.devices!r}"
        )
    caps = _as_set(capped)
    if wanted == set(wp.devices):
        if not is_priority(wp, caps):
            return wp
        deadline = stop_deadline(wp.devices, deadlines=deadlines, capped=caps)
        return wp if deadline == wp.deadline_ts else replace(wp, deadline_ts=deadline)
    devices = tuple(d for d in wp.devices if d in wanted)
    return ContactWaypoint(
        position=wp.position,
        devices=devices,
        bucket=_worst_bucket(devices, device_states, wp.bucket),
        deadline_ts=stop_deadline(devices, deadlines=deadlines, capped=caps),
    )


# --------------------------------------------------------------------------- #
# The F family's member order
# --------------------------------------------------------------------------- #

def _weight(weights: Optional[Mapping[DeviceID, float]], did: DeviceID) -> float:
    if weights is None:
        return 1.0
    w = weights.get(did, 1.0)
    if isinstance(w, bool) or not isinstance(w, numbers.Real) or not math.isfinite(w) or w < 0:
        raise ValueError(
            f"member_order: the weight of {did!r} must be a finite number >= 0, got {w!r}"
        )
    return float(w)


def _worth(weight: float, dwell_s: float) -> float:
    """Coverage weight per second of dwell; a member that costs no dwell is
    worth ``inf`` when it weighs anything, and 0 otherwise."""
    if dwell_s > 0.0:
        return weight / dwell_s
    return math.inf if weight > 0.0 else 0.0


def member_order(
    wp: ContactWaypoint,
    *,
    model: FeasibilityModel,
    capped: Collection[DeviceID] = frozenset(),
    weights: Optional[Mapping[DeviceID, float]] = None,
    pass_kind: MissionPass = _COLLECT,
    snr_offset_db: float = 0.0,
) -> Tuple[DeviceID, ...]:
    """The F family's order of ``wp``'s members (spec, other choices 4).

    Capped members first; then ``w_j / dwell_j`` descending; then the device
    id. ``dwell_j`` is member j's predicted dwell at this stop alone, as
    ``model`` prices it: 0 for a member beyond range or predicted below the SNR
    floor, which the model never charges, so such a member leads its group.
    ``w_j`` is its coverage weight (U2's demand weights): None weighs every
    member 1, the plan's letter, and so does a member missing from
    ``weights`` (a stop the beacon hook inserted in flight holds devices
    outside the plan's demand).

    Why this order: capped members must be served (decision 1), so they go
    first and a stop sheds its uncapped members first. Among the rest, the
    most weight per second of dwell first is the greedy knapsack ratio, and the
    weights keep "cheapest dwell first" from always favouring near, high-SNR
    members (design D-D; research map section 4).
    """
    caps = _as_set(capped)

    def key(did: DeviceID) -> Tuple[bool, float, DeviceID]:
        alone = ContactWaypoint(position=wp.position, devices=(did,), bucket=wp.bucket,
                                deadline_ts=wp.deadline_ts)
        dwell = model.leg(wp.position, alone, pass_kind=pass_kind,
                          snr_offset_db=snr_offset_db).dwell_s
        return (did not in caps, -_worth(_weight(weights, did), dwell), did)

    return tuple(sorted(wp.devices, key=key))


# --------------------------------------------------------------------------- #
# The member walk: one stop, one order
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MemberWalk:
    """What :func:`admit_members` admitted of one stop.

    ``stop`` is the stop reduced to the admitted members (``wp`` itself when
    every member was admitted and nothing else changed, :func:`reduce_stop`),
    or None when none was. ``verdict`` is the
    predicate's verdict for ``stop`` from the walk's state, None with it; its
    ``next_state`` is where the mule is after serving ``stop``. ``refused``
    pairs each member left out with the reason it was refused, in walk order.
    """

    stop: Optional[ContactWaypoint]
    verdict: Optional[Verdict]
    refused: Tuple[Tuple[DeviceID, str], ...]

    @property
    def ok(self) -> bool:
        """True when at least one member was admitted."""
        return self.stop is not None

    @property
    def admitted(self) -> Tuple[DeviceID, ...]:
        """The admitted members, in the stop's order."""
        return () if self.stop is None else tuple(self.stop.devices)


def admit_members(
    model: FeasibilityModel,
    state: FlightState,
    wp: ContactWaypoint,
    order: Sequence[DeviceID],
    *,
    rule: str,
    budget_end: Optional[float],
    deadlines: Mapping[DeviceID, float],
    device_states: Mapping[DeviceID, Any],
    pass_kind: MissionPass = _COLLECT,
    snr_offset_db: float = 0.0,
    capped: Collection[DeviceID] = frozenset(),
    veto: Optional[Veto] = None,
) -> MemberWalk:
    """Walk ``wp``'s members in ``order`` from ``state``: the one member walk.

    ``order`` must be a permutation of ``wp.devices``, else ValueError. Each
    member ``d`` in turn is tried as the stop reduced to the members admitted
    so far plus ``d`` (:func:`reduce_stop`), under ``model.admit`` with
    ``rule`` and ``budget_end``; admitted, it joins the stop, refused, it is
    skipped and the walk goes on (skip, not stop). With no capped member and no
    ``veto`` this is exactly U3b's walk (``unit_U3b.md`` section 1.5): the
    predicate's own call, ``protected`` False.

    The F family's arguments: ``capped`` gives a trial made of capped members
    only the predicate's exemption and a mixed trial B2's deadline (see the
    module docstring); ``veto`` may refuse a trial the predicate admitted.
    Every arm shares this walk and passes its own ``order``: the F family
    :func:`member_order`, the H and D arms theirs (unit U3b).
    """
    walk_order = tuple(order)
    if Counter(walk_order) != Counter(wp.devices):
        raise ValueError(
            f"admit_members: the order {walk_order!r} is not a permutation of the "
            f"stop's members {wp.devices!r}"
        )
    caps = _as_set(capped)
    admitted: List[DeviceID] = []
    stop: Optional[ContactWaypoint] = None
    verdict: Optional[Verdict] = None
    refused: List[Tuple[DeviceID, str]] = []
    for did in walk_order:
        trial = reduce_stop(wp, admitted + [did], deadlines=deadlines,
                            device_states=device_states, capped=caps)
        v = model.admit(state, trial, rule=rule, budget_end=budget_end, pass_kind=pass_kind,
                        protected=is_exempt(trial, caps), snr_offset_db=snr_offset_db)
        why = v.reason if not v.ok else (None if veto is None else veto(trial, v))
        if why is None:
            admitted.append(did)
            stop, verdict = trial, v
        else:
            refused.append((did, why))
    return MemberWalk(stop=stop, verdict=verdict, refused=tuple(refused))


def complements(
    wp: ContactWaypoint,
    refused: Iterable[Tuple[DeviceID, str]],
    *,
    deadlines: Mapping[DeviceID, float],
    device_states: Mapping[DeviceID, Any],
    capped: Collection[DeviceID] = frozenset(),
) -> Tuple[Tuple[ContactWaypoint, str], ...]:
    """The members a walk left out, as waypoints of their own with their reason.

    One reduced stop (:func:`reduce_stop`) per reason, holding the members
    refused for it in the stop's order; the groups follow S3b's ``REASONS``
    order (overdue, budget, energy, delivery), any other reason after them in
    the order first met. They are ordinary drops: the mule widens and records
    them as it does every dropped contact (``mule_main.py``, the pre-flight and
    in-flight widening).
    """
    groups: Dict[str, List[DeviceID]] = {}
    for did, why in refused:
        groups.setdefault(why, []).append(did)
    rank = {reason: i for i, reason in enumerate(REASONS)}
    reasons = sorted(groups, key=lambda r: rank.get(r, len(REASONS)))
    return tuple(
        (reduce_stop(wp, groups[r], deadlines=deadlines, device_states=device_states,
                     capped=capped), r)
        for r in reasons
    )


# --------------------------------------------------------------------------- #
# One stop, whole or reduced, and the F family's folds
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StopAdmission:
    """One stop admitted from one flight state (:func:`admit_stop`).

    ``stop`` is what is flown: the stop whole, or reduced to the members that
    fit, or None when none does. ``verdict`` is its verdict (None with it).
    ``dropped`` pairs what is left out with its reason: the complements of the
    members left out (:func:`complements`), or, for a stop without a capped
    member none of whose members fits, the stop whole with the reason it
    failed whole (:func:`admit_stop`).
    """

    stop: Optional[ContactWaypoint]
    verdict: Optional[Verdict]
    dropped: Tuple[Tuple[ContactWaypoint, str], ...]

    @property
    def ok(self) -> bool:
        return self.stop is not None


def admit_stop(
    model: FeasibilityModel,
    state: FlightState,
    wp: ContactWaypoint,
    *,
    rule: str,
    budget_end: Optional[float],
    deadlines: Mapping[DeviceID, float],
    device_states: Mapping[DeviceID, Any],
    capped: Collection[DeviceID] = frozenset(),
    weights: Optional[Mapping[DeviceID, float]] = None,
    order: Optional[MemberOrder] = None,
    snr_offset_db: float = 0.0,
    veto: Optional[Veto] = None,
) -> StopAdmission:
    """Admit ``wp`` from ``state`` in Pass 1: whole if it fits, else its members.

    The stop is first tried whole (with B2's deadline under the cap). If it
    fits, it is admitted as is, the very object when nothing about it changed.
    Otherwise :func:`admit_members` walks its members in ``order(wp)`` (the F
    order, :func:`member_order`, when None); the order is computed only then.
    By the down-closure in the module docstring the result is the walk's in
    every case, and the whole try only saves the walk. The members left out
    are dropped as :func:`complements`, each with the reason the walk refused
    it.

    When no member fits, a stop without a capped member is dropped whole with
    the reason it failed whole: the object and reason ``FeasibilityModel.fold``
    gives, as U3b's walks drop such a stop (``unit_U3b.md`` section 3.1). A
    priority stop is dropped as its complements instead, as when some member
    fits. Whole, a mixed stop is held to its uncapped members' deadline (B2),
    so it can fail ``overdue`` although its capped members, each exempt alone,
    are refused for another clause; the whole-stop reason would then label a
    capped device ``overdue``, which spec other choices 8 rules out, and would
    tie its label to whether a co-member happens to fit.

    This is one step of :func:`fold_members`, public so that a search can fold
    a route stop by stop from a prefix's state (U4's depth-first search).
    """
    caps = _as_set(capped)
    whole = reduce_stop(wp, wp.devices, deadlines=deadlines, device_states=device_states,
                        capped=caps)
    v = model.admit(state, whole, rule=rule, budget_end=budget_end, pass_kind=_COLLECT,
                    protected=is_exempt(whole, caps), snr_offset_db=snr_offset_db)
    why = v.reason if not v.ok else (None if veto is None else veto(whole, v))
    if why is None:
        return StopAdmission(whole, v, ())
    members = (order(wp) if order is not None
               else member_order(wp, model=model, capped=caps, weights=weights,
                                 snr_offset_db=snr_offset_db))
    walk = admit_members(model, state, wp, members, rule=rule, budget_end=budget_end,
                         deadlines=deadlines, device_states=device_states,
                         snr_offset_db=snr_offset_db, capped=caps, veto=veto)
    if walk.stop is None and not is_priority(wp, caps):
        return StopAdmission(None, None, ((whole, why),))
    return StopAdmission(
        walk.stop, walk.verdict,
        complements(wp, walk.refused, deadlines=deadlines, device_states=device_states,
                    capped=caps),
    )


def pass_energy_j(model: FeasibilityModel, state: FlightState) -> float:
    """The pass's simulated energy at ``state`` with the flight home included.

    ``FlightState.energy_j`` counts the flight and hover so far; the admit
    energy clause also charges the return leg (``P_move * return``), which the
    state leaves out. This adds it from ``state``'s pose, so a plan's E covers
    the pass as flown (``MemberFold.energy_j``). Legacy (no ferry physics):
    ``state.energy_j``, as the legacy model has no return leg.
    """
    ferry = model.ferry
    if ferry is None:
        return state.energy_j
    back, _ = model.cost(state.pose, ferry.dock)
    return state.energy_j + ferry.p_move_w * back


def _member_fold(
    model: FeasibilityModel,
    flown: List[ContactWaypoint],
    dropped: List[Tuple[ContactWaypoint, str]],
    state: FlightState,
    home: Optional[float],
    *,
    feasible: bool,
) -> MemberFold:
    return MemberFold(
        route=tuple(flown), dropped=tuple(dropped), state=state,
        home=model.home_at(state) if home is None else home,
        feasible=feasible, energy_j=pass_energy_j(model, state),
    )


def _check_rule(rule: str) -> None:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")


def fold_members(
    route: Sequence[ContactWaypoint],
    state: FlightState,
    *,
    model: FeasibilityModel,
    rule: str,
    budget_end: Optional[float],
    deadlines: Mapping[DeviceID, float],
    device_states: Mapping[DeviceID, Any],
    require_all: bool,
    capped: Collection[DeviceID] = frozenset(),
    weights: Optional[Mapping[DeviceID, float]] = None,
    order: Optional[MemberOrder] = None,
    snr_offset_db: float = 0.0,
) -> MemberFold:
    """Fold a Pass-1 ``route`` in order from ``state``, reducing stops that fail whole.

    Each stop in turn goes through :func:`admit_stop`: whole if it fits, else
    reduced to the members that fit in ``order`` (the F order when None), each
    priced from the state the previous stop left. ``require_all=True`` is a
    candidate plan (design section 2.5): every listed stop must admit someone,
    so the first that admits nobody makes the fold infeasible and ends it, and
    the search prunes every extension of that prefix. ``require_all=False``
    drops such a stop as :func:`admit_stop` does (whole with the reason it
    failed whole, or member by member for a priority stop) and goes on.

    The drops' reasons are the walk's, each from the state its stop was tried
    in. They are not the plan's labels for the devices it leaves out, which
    price every such device alone from the dock at takeoff (spec, other
    choices 8; the scheduler's commit).

    The result (``MemberFold``): the stops flown, the complements and dropped
    stops with their reasons, the state after the last stop flown, ``home``
    (that stop's verdict's: back at the dock, the upload done;
    ``model.home_at`` when nothing was flown) and the pass's energy with the
    return leg (:func:`pass_energy_j`). The route passes ``model.fold`` without
    skipping from ``state``, its exempt stops (every member capped) protected,
    by construction.

    Pass 1 only: Pass 2 delivers to every contact whole (a budgeted Pass 2 is
    refused in plan mode, critic B8), so there is no Pass-2 member fold.
    """
    _check_rule(rule)
    if not isinstance(require_all, bool):
        raise TypeError(f"require_all must be a bool, got {require_all!r}")
    caps = _as_set(capped)
    flown: List[ContactWaypoint] = []
    dropped: List[Tuple[ContactWaypoint, str]] = []
    cur, home = state, None
    for wp in route:
        got = admit_stop(model, cur, wp, rule=rule, budget_end=budget_end, deadlines=deadlines,
                         device_states=device_states, capped=caps, weights=weights,
                         order=order, snr_offset_db=snr_offset_db)
        dropped.extend(got.dropped)
        if got.stop is None:
            if require_all:
                return _member_fold(model, flown, dropped, cur, home, feasible=False)
            continue
        flown.append(got.stop)
        cur, home = got.verdict.next_state, got.verdict.home  # type: ignore[union-attr]
    return _member_fold(model, flown, dropped, cur, home, feasible=True)


def _keeps(
    later: Sequence[ContactWaypoint],
    *,
    model: FeasibilityModel,
    rule: str,
    budget_end: Optional[float],
    snr_offset_db: float,
) -> Optional[Veto]:
    """A veto that refuses a trial after which the ``later`` cores no longer fit.

    The cores are capped members only, so all of them are exempt. The reason
    is the clause that refuses the first core that fails.
    """
    if not later:
        return None
    cores = tuple(later)
    guarded = frozenset(cores)

    def veto(trial: ContactWaypoint, verdict: Verdict) -> Optional[str]:
        rest = model.fold(cores, verdict.next_state, rule=rule, budget_end=budget_end,
                          pass_kind=_COLLECT, skip=False, protected=guarded,
                          snr_offset_db=snr_offset_db)
        return None if rest.ok else rest.rejected[0][1]

    return veto


def trim_members(
    remainder: Sequence[ContactWaypoint],
    state: FlightState,
    *,
    model: FeasibilityModel,
    budget_end: Optional[float],
    deadlines: Mapping[DeviceID, float],
    device_states: Mapping[DeviceID, Any],
    capped: Collection[DeviceID] = frozenset(),
    weights: Optional[Mapping[DeviceID, float]] = None,
    order: Optional[MemberOrder] = None,
    rule: str = RULE_DEADLINE_BUDGET,
    snr_offset_db: float = 0.0,
) -> "ReplanResult":
    """The plan-mode re-plan of the rest of Pass 1: a member trim (spec, other choices 9).

    Returns a ``routing.replan.ReplanResult``. ``routing.replan.replan_route``
    is not used: its identity check (``replan.py:198-207``) refuses any stop
    it was not given, and a reduced stop is a new waypoint.

    1. If the remainder passes the predicate as flown (``model.fold`` without
       skipping, its exempt stops protected), it is kept: ``current``.
    2. Otherwise (``arm_trimmed``) priority stops (any member capped) fly
       first, in their relative order, then the rest in theirs; the trim may
       move priority stops to the front, as the Phase 3 trim moves protected
       stops (``replan.py:220-227``; critic A11 (v)):

       a. the protected-only trim: each priority stop reduced to its capped
          members and walked from ``state``; what it keeps of each is that
          stop's *core*;
       b. each priority stop is then admitted from its core up, its other
          members after it in the member order, each only while every later
          core still fits after it. So a priority stop sheds its uncapped
          members first, and a capped member is dropped only when the
          protected-only trim cannot hold it (design section 2.4; spec, other
          choices 5: dropped only when it cannot fit alone);
       c. the rest, each whole if it fits, else reduced, else dropped, as
          :func:`fold_members` walks with ``require_all=False``.

    Every member of the remainder ends in the route or in ``dropped``, once;
    the drops are listed in the remainder's order, a stop's complements in
    S3b's ``REASONS`` order. A stop without a capped member that keeps no one
    is dropped whole with the reason it failed whole, as the Phase 3 trim
    drops it; the members a priority stop leaves out are dropped each with
    the reason it was refused, so in the F order no drop labels a capped
    member ``overdue`` (:func:`admit_stop`). Each dropped Pass-1 device is
    widened by the mule (spec Q10), complements included.

    The remainder's stops must carry the deadlines the plan gave them
    (:func:`reduce_stop` with the same ``capped``), so that step 1 checks
    what the plan priced; ``deadlines`` must cover every member, those of a
    stop the beacon hook inserted included. ``rule`` is the arm's in-flight
    rule, S3b's deadline and budget for the F family. Pass 1 only, as
    :func:`fold_members`.
    """
    from hermes.scheduler.routing.replan import (  # noqa: WPS433
        ORDER_ARM_TRIMMED,
        ORDER_CURRENT,
        ReplanResult,
    )

    _check_rule(rule)
    stops = list(remainder)
    index = {id(wp): i for i, wp in enumerate(stops)}
    if len(index) != len(stops):
        raise ValueError("trim_members: the remainder lists a stop twice")
    members = [d for wp in stops for d in wp.devices]
    if len(set(members)) != len(members):
        raise ValueError("trim_members: a device is a member of two stops of the remainder")
    caps = _as_set(capped)
    common = dict(rule=rule, budget_end=budget_end, snr_offset_db=snr_offset_db)

    # 1. The current order.
    exempt = frozenset(wp for wp in stops if is_exempt(wp, caps))
    if model.fold(stops, state, pass_kind=_COLLECT, skip=False, protected=exempt, **common).ok:
        return ReplanResult(tuple(stops), (), ORDER_CURRENT)

    orders: Dict[int, Tuple[DeviceID, ...]] = {}

    def order_of(wp: ContactWaypoint) -> Tuple[DeviceID, ...]:
        if id(wp) not in orders:
            orders[id(wp)] = tuple(
                order(wp) if order is not None
                else member_order(wp, model=model, capped=caps, weights=weights,
                                  snr_offset_db=snr_offset_db)
            )
        return orders[id(wp)]

    priority = [wp for wp in stops if is_priority(wp, caps)]
    rest = [wp for wp in stops if not is_priority(wp, caps)]
    walk = dict(deadlines=deadlines, device_states=device_states, capped=caps, **common)

    # 2a. The protected-only trim: each priority stop's capped members.
    cores: List[Optional[ContactWaypoint]] = []
    cur = state
    for wp in priority:
        caps_only = reduce_stop(wp, [d for d in wp.devices if d in caps],
                                deadlines=deadlines, device_states=device_states, capped=caps)
        first = tuple(d for d in order_of(wp) if d in caps)
        got = admit_stop(model, cur, caps_only, order=lambda _w, o=first: o, **walk)
        cores.append(got.stop)
        if got.stop is not None:
            cur = got.verdict.next_state  # type: ignore[union-attr]

    route: List[ContactWaypoint] = []
    drops: List[Tuple[int, int, Tuple[ContactWaypoint, str]]] = []

    def record(wp: ContactWaypoint, got: StopAdmission) -> None:
        for seq, pair in enumerate(got.dropped):
            drops.append((index[id(wp)], seq, pair))

    # 2b. Priority stops from their cores up, while every later core fits.
    cur = state
    for i, wp in enumerate(priority):
        core = cores[i]
        kept = frozenset(() if core is None else core.devices)
        ranked = order_of(wp)
        up = tuple(d for d in ranked if d in kept) + tuple(d for d in ranked if d not in kept)
        veto = _keeps([c for c in cores[i + 1:] if c is not None], model=model, **common)
        got = admit_stop(model, cur, wp, order=lambda _w, o=up: o, veto=veto, **walk)
        record(wp, got)
        if got.stop is not None:
            route.append(got.stop)
            cur = got.verdict.next_state  # type: ignore[union-attr]

    # 2c. The rest, in its relative order.
    for wp in rest:
        got = admit_stop(model, cur, wp, order=order_of, **walk)
        record(wp, got)
        if got.stop is not None:
            route.append(got.stop)
            cur = got.verdict.next_state  # type: ignore[union-attr]

    dropped = tuple(pair for _, _, pair in sorted(drops, key=lambda t: (t[0], t[1])))
    return ReplanResult(tuple(route), dropped, ORDER_ARM_TRIMMED)
