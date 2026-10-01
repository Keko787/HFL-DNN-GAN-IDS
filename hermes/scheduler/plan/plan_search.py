"""FeRRy Phase 4: the plan search (unit U4).

**Why a search.** In plan mode the mule decides its band class b̄ and its Pass-1
route π together, once per mission, at the dock: reach becomes a decision
(build plan L822-847). A wider class serves fewer devices per stop but at a
faster rate; a narrower one reaches the whole field from one stop and dwells
longer there. No rule of thumb picks between them, so the planner prices the
candidates of every class the arm may fly and commits the best. This module
is that choice. The scheduler (U5, ``FLScheduler.build_ferry_plan``) gathers
its inputs (S1, S3, the age cap, S3a once per class at the class's radius with
the hover rule, ``plan/hover.py``, and Pass 2 once per class), runs a guard
fold on the plan it gets back and commits it.

**The candidates** (Phase 4 spec, other choices 3; critic A6). A candidate is a
class c plus an ordered sequence of distinct stops offered on c (S3a(c), where
a capped device its S3a stop cannot serve alone has a stop of its own at its
best hover point), each reduced to
a non-empty subset of its members, that passes S3b's predicate from takeoff:
each stop, in turn, admitted under ``RULE_DEADLINE_BUDGET`` from the state the
previous one left. The age cap's stop rules are U1's
(``stages/s3d_age_cap.py``), applied by value to every stop as reduced (critic
B1, B2): a stop whose members are all capped is exempt from its own deadline
clause, and a mixed stop carries the earliest deadline of its uncapped
members. The empty plan is a candidate too: it flies nothing and pays only
the dock turnaround. Under ``member_admission = whole`` every stop is flown
whole, which keeps the narrow-band cliff for comparison (design D-D).

**The choice** is the smallest plan key (U2's ``plan_key``): the cap key
first (U1's ``cap_key``, the ages of the capped devices left out, largest
first; critic C5), then, under ``coverage_rank = lexicographic`` (the
default; the orchestrator's resolution R11), the served weight share
largest first, then V (U2's ``score``), both rounded to 9 decimals, then the
class index and the stops. Under ``weighted``, and for F-cov (κ = 0) under
either, the share is left out: that key is U0's ``Candidate.key``. Either is
a total order, so the pick never depends on the order candidates are met.
Every demanded device weighs more than 0, so under ``lexicographic`` the
empty plan is chosen only when no plan that serves anyone is admitted (its
cap key is never below theirs, and its share is 0). No rule below prunes by
the key: the prefix and the down-closure prunings drop only plans the
predicate refuses, and the local search compares keys only to move and to
keep its best.

**Three searches, by size** (``PlanSearchParams``; the plan's threshold of 6,
L829; critic A6 and C7):

* ``exact``, when the demand has at most ``exact_max_devices`` (6) devices:
  the whole family, depth first. A prefix that fails prunes every extension
  (the fold prices each stop from where the previous one left the mule), and
  at one flight state a member set that fails prunes its supersets: every
  clause of the predicate is monotone in a stop's members (dwell adds up, the
  stop's deadline is a minimum over them, and adding a member can only end an
  exemption; ``plan/member_subset.py``). So it is exactly the optimum over
  the member subsets V prices. At N = 6 (the pilots) a whole plan took at
  most about 0.25 s where nothing prunes (no budget; three classes of six
  one-device stops, 5,871 candidates; 0.21 s on the mule's own classes) and
  about 20 ms under 30-120 s budgets (U4's timing probes of 2026-09-30;
  critic probe G's "0.04 s at most" is not the worst case).
* ``stop_subsets``, for a larger demand when the class has at most
  ``exhaustive_max_stops`` (6) stops: depth first over the ordered subsets of
  its stops, each stop whole if it fits from where its prefix leaves the mule,
  else reduced greedily in the F member order, skip not stop (U3's
  ``admit_stop``). This is the design's family (D-H), which never sheds a
  member from a stop that fits whole.
* ``local``, for more stops: a 2-OPT tour of the stops from the dock back to
  it (``routing.two_opt.order_contacts``), U3's member trim of it (under
  ``whole``: the tour with its priority stops first, each stop kept whole if
  it fits and the exempt ones protected, the trim's rules on whole stops),
  then first-improvement scans, each moving on one key. A scan's pass scans
  the current route's neighbours in this order and moves to the first whose
  key is smaller: drop a stop (by its place in the route); insert an
  unrouted stop with all its members (the stops in canonical order, each at
  every place from the front), walked like the depth-first search (whole if
  it fits, else reduced greedily); reverse a segment of two stops or more (by
  its first stop, then its last); and, under ``subset``, drop one served
  member of a stop that serves several (stop by stop, least worth first in
  the F order). The order is part of the contract: it decides which local
  optimum a scan reaches and how many walks the trace records. A scan moves
  between routes as flown, each stop limited to the members it serves, and a
  move is a candidate only when every stop it lists admits someone: each is
  checked by a fold that does not skip (design D-H). A neighbour the scan
  walked before is not walked again: it did not beat the current route then,
  and the current route has only improved since.

  Under ``weighted`` one scan runs, on the plan key, from the trim's route.
  Under ``lexicographic`` a scan on the plan key never takes a drop or a
  drop-member move: each serves less weight (the later stops keep their
  members at most) and leaves the cap key no smaller. So it cannot give up a
  light stop for a heavier one, and it takes the first insert that raises
  the share, however little that adds and however much of the budget it
  uses. Alone, it ended below ``weighted``'s plan, under its own key, on 3 of
  600 random local problems at κ = 1 (the review of 2026-09-30; one served 14
  of 15 devices where ``weighted`` served all 15). So up to three scans run:
  ``weighted``'s own first, on U0's ``Candidate.key`` from the trim's route,
  walk for walk; then one on the plan key from the best plan met so far
  under it; then, if that plan's route is not the trim's, one on the plan key
  from the trim's route, where the plan key's scan alone goes. A walk depends
  on its entries alone, so a later scan takes an earlier one's walk instead
  of walking again, and does not count it (the plan key's scans still scan
  the drop moves they never take, one contract for every scan). The class's
  best is the best plan met: never below ``weighted``'s under the plan key,
  and, unless a bound ends the search, never below the plan key's scan alone
  and a local optimum of every move under the plan key.

  The bounds count every scan: the search stops when its last scan reaches a
  local optimum, after ``heuristic_max_passes`` passes, or when a walk would
  exceed ``heuristic_max_evaluations`` walks (the start's included),
  whichever comes first. The first scan has the whole bound, as under
  ``weighted``, so the guarantee against ``weighted`` holds when a bound
  ends it. The later scans walked nothing new on U4's 100 m worlds (N = 24
  to 96) and added a third to a half of the walks on 300 m and 500 m fields
  (the fix's probes of 2026-09-30). Counts, never wall time, so a repeated
  trial plans the same (critic C7). The counts buy determinism, not a time
  bound: a walk costs more on a longer route, and at N = 96 on a 500 m field
  (1 MB, no budget) all three classes reach 2,000 walks and one plan took
  3.1-3.4 s (U4's timing probes of 2026-09-30). No single move swaps one
  stop for another, so where two capped stops compete for one slot it can
  keep the younger one the trim took first.

Only ``exact`` is optimal over the member subsets V prices; above 6 devices
the restriction to these two families is a plan deviation, which the Freeze
and the plan record (UD). Every mode keeps the class's best under the plan
key, and each class is searched on its own, so FB+c (``fixed:c``) commits
exactly class c's best among F's classes, and F's plan key is never above
any FB+c's (critic A3).

**Pass 2** (the user's decision 2 (b)). V's Δ is the whole mission on b̄: Pass
1, the turnaround, and Pass 2, which flies b̄ too. Pass 2 is priced once per
class per plan, as the T_nom helper prices it
(``fl_scheduler.nominal_mission_period_s``): the class's Pass-2 queue folded
from the dock at clock 0 under ``RULE_NONE``, without a budget, delivering.
The scheduler builds that queue (``FLScheduler.build_pass_2_queue`` at the
class's radius) and hands its price in as :class:`PassTwo`, through
:func:`price_pass_2`, so this module never imports the scheduler. A plan that
serves nobody does not fly Pass 2, and U2's score leaves it out then.

**Determinism.** No wall time enters a decision, the stops are searched in
their canonical order (position, then devices, as ``order_contacts`` sorts
them), and the per-class summaries record counts only, so a repeated trial
describes the same plan (critic B12).

**Layering.** Numpy-free, nothing from ``hermes.l1``, the mule, the policies,
the scheduler or ``experiments``: physics reaches the search as the classes'
``FeasibilityModel`` and ``outage`` callables. Freeze Rule 1: only the plan
path (``plan_mode = "ferry"``) imports this module, and the recorded pipeline
never loads it.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from hermes.types.ids import DeviceID
from hermes.types.scheduler import ContactWaypoint, MissionPass
from hermes.scheduler.routing.two_opt import order_contacts
from hermes.scheduler.stages.s3b_feasibility import (
    RULE_DEADLINE_BUDGET,
    RULE_NONE,
    FeasibilityModel,
    FlightState,
    Verdict,
)
from hermes.scheduler.stages.s3d_age_cap import (
    cap_key,
    cap_stops,
    is_exempt,
    priority_first,
    with_cap_deadlines,
)

from .member_subset import admit_stop, member_order, pass_energy_j, reduce_stop, trim_members
from .plan_score import applied_rank, plan_key, score, served_share
from .types import (
    COVERAGE_RANK_WEIGHTED,
    MEMBER_ADMISSION_SUBSET,
    SEARCH_EXACT,
    SEARCH_LOCAL,
    SEARCH_STOP_SUBSETS,
    Candidate,
    CapState,
    MemberFold,
    PlanClass,
    PlanSetup,
    SearchResult,
)

__all__ = [
    "PassTwo", "price_pass_2", "ClassInput", "ClassResult",
    "search_mode", "search_class", "plan_search",
]

_COLLECT = MissionPass.COLLECT
_DELIVER = MissionPass.DELIVER

#: A walk of a route in the local search: (stop index, member bitmask) per
#: stop, bit k standing for the stop's k-th member.
_Entries = Tuple[Tuple[int, int], ...]


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _nonneg(value: Any, name: str) -> float:
    """``value`` as a finite float >= 0; a bool is refused."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return out


def _budget_end(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"budget_end must be a number or None, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"budget_end must be finite or None, got {value!r}")
    return out


def _stop_key(wp: ContactWaypoint) -> Tuple[Tuple[float, ...], Tuple[DeviceID, ...]]:
    """A stop's identity by value, for the memos: where it is and whom it serves."""
    return tuple(wp.position), tuple(wp.devices)


def _state_key(state: FlightState) -> Tuple[Any, ...]:
    return tuple(state.pose), state.clock, state.energy_j, state.deliver_by


# --------------------------------------------------------------------------- #
# Pass 2 (decision 2 (b))
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PassTwo:
    """Pass 2 on one class, as V prices it (the user's decision 2 (b)).

    ``time_s`` is the pass from the dock back to the dock, ``dwell_s`` its
    predicted dwell (which V leaves out of Δ under F-dwell) and ``energy_j``
    its simulated energy with the return leg. Build it with
    :func:`price_pass_2`; the scheduler does so once per class per plan.
    """

    time_s: float
    dwell_s: float
    energy_j: float

    def __post_init__(self) -> None:
        for name in ("time_s", "dwell_s", "energy_j"):
            object.__setattr__(self, name, _nonneg(getattr(self, name), name))


def price_pass_2(model: FeasibilityModel, queue: Sequence[ContactWaypoint]) -> PassTwo:
    """Pass 2 of ``queue`` on ``model``'s class, priced as the T_nom helper prices it.

    ``queue`` is the class's Pass-2 queue (``FLScheduler.build_pass_2_queue``
    at the class's radius, nearest first from the dock) and ``model`` the
    class's model bound to the scheduler's device states. As in
    ``fl_scheduler.nominal_mission_period_s``: the queue folded from the dock
    at clock 0 under ``RULE_NONE`` without a budget, delivering, without
    skipping (Pass 2 visits every contact whole; a budgeted Pass 2 is refused
    in plan mode, critic B8). The time is that fold's ``home``, the dwell the
    stops' delivery dwell summed, and the energy the fold's with the return
    leg (U3's ``pass_energy_j``: ``FlightState`` leaves the return out). An
    empty queue costs nothing.
    """
    ferry = getattr(model, "ferry", None)
    if ferry is None:
        raise ValueError("Pass 2 is priced on the mission clock: the model needs ferry physics")
    stops = tuple(queue)
    walk = model.fold(stops, FlightState(ferry.dock, 0.0), rule=RULE_NONE, budget_end=None,
                      pass_kind=_DELIVER, skip=False)
    dwell = math.fsum(model.leg(wp.position, wp, pass_kind=_DELIVER).dwell_s for wp in stops)
    return PassTwo(time_s=walk.home, dwell_s=dwell, energy_j=pass_energy_j(model, walk.state))


# --------------------------------------------------------------------------- #
# Inputs and results
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClassInput:
    """One class the arm may fly, as the search takes it from the scheduler.

    ``cls`` carries the class's model, which must be bound to the scheduler's
    device states (``FerryPhysics.bind``) so that members are priced where they
    are; an unbound model would price every member at its stop. ``stops`` are
    the stops offered on the class, whole, each device of the demand in one
    of them: S3a's at ``cls.radius_m``, with the hover rule
    (``plan/hover.py``: a capped device its S3a stop cannot serve alone has a
    stop of its own at its best hover point); ``pass_2`` is the class's Pass
    2 (:func:`price_pass_2`).
    """

    cls: PlanClass
    stops: Tuple[ContactWaypoint, ...]
    pass_2: PassTwo

    def __post_init__(self) -> None:
        if not isinstance(self.cls, PlanClass):
            raise TypeError(f"cls must be a PlanClass, got {self.cls!r}")
        if getattr(self.cls.model.ferry, "device_states", None) is None:
            raise ValueError(
                f"class {self.cls.name!r}: bind its model to the scheduler's device states "
                f"(FerryPhysics.bind), or every member is priced at its stop"
            )
        stops = tuple(self.stops)
        members: List[DeviceID] = []
        for wp in stops:
            if not isinstance(wp, ContactWaypoint):
                raise TypeError(f"stops hold ContactWaypoints, got {wp!r}")
            members.extend(wp.devices)
        if len(set(members)) != len(members):
            raise ValueError(f"class {self.cls.name!r}: the offered stops place each device "
                             f"in one stop, as S3a does")
        if not isinstance(self.pass_2, PassTwo):
            raise TypeError(f"pass_2 must be a PassTwo, got {self.pass_2!r}")
        object.__setattr__(self, "stops", stops)


@dataclass(frozen=True)
class ClassResult:
    """One class searched (:func:`search_class`).

    ``best`` is the class's candidate with the smallest plan key (the empty
    plan included), ``mode`` the search that ran (:func:`search_mode`),
    ``n_candidates`` the candidates it scored (the empty plan once, and every
    other admitted plan it met: each one of the family for the exhaustive
    searches, each walk that passed for the local search) and ``summary`` the
    class's JSON-ready record for the commit's ``per_class``: the class's
    ``band`` and ``index``, the ``mode``, ``stops`` (how many stops were
    offered on the class, S3a's with the hover rule, not how many the best
    flies), ``candidates``, and
    this class's best's ``v``, ``served`` (a count), ``cap_key`` and
    ``served_share`` (its served weight share, U2's ``served_share``), not the
    plan's; ``coverage_rank``, the rank the search applied (U2's
    ``applied_rank``: ``weighted`` for F-cov whatever the setting), so a
    trace shows which rank chose; the local search adds ``evaluations``
    (walks run, the start's included), ``passes`` and ``bounded`` (True when
    a bound, not a local optimum, ended it), over all its scans, and under
    ``lexicographic`` ``weighted_passes`` and ``weighted_evaluations``, its
    first scan's (on the weighted key; the start's walk included).
    """

    best: Candidate
    mode: str
    n_candidates: int
    summary: Mapping[str, Any]


def search_mode(n_demand: int, n_stops: int, bounds: Any) -> str:
    """The search one class gets (Phase 4 spec, other choices 3).

    ``exact`` for a demand of at most ``bounds.exact_max_devices`` devices,
    else ``stop_subsets`` for a class of at most ``bounds.exhaustive_max_stops``
    stops, else ``local``. ``bounds`` is a ``PlanSearchParams``.
    """
    if n_demand <= bounds.exact_max_devices:
        return SEARCH_EXACT
    if n_stops <= bounds.exhaustive_max_stops:
        return SEARCH_STOP_SUBSETS
    return SEARCH_LOCAL


# --------------------------------------------------------------------------- #
# One class
# --------------------------------------------------------------------------- #

class _ClassSearch:
    """The search of one class: its inputs, memos, candidates and best so far."""

    def __init__(
        self,
        setup: PlanSetup,
        entry: ClassInput,
        *,
        start: FlightState,
        budget_end: Optional[float],
        deadlines: Mapping[DeviceID, float],
        device_states: Mapping[DeviceID, Any],
        cap: CapState,
        weights: Mapping[DeviceID, float],
    ) -> None:
        if not isinstance(setup, PlanSetup):
            raise TypeError(f"setup must be a PlanSetup, got {setup!r}")
        if not isinstance(entry, ClassInput):
            raise TypeError(f"entry must be a ClassInput, got {entry!r}")
        if not isinstance(cap, CapState):
            raise TypeError(f"cap must be a CapState, got {cap!r}")
        if not hasattr(start, "clock") or not hasattr(start, "pose"):
            raise TypeError(f"start must be a FlightState, got {start!r}")
        cls = entry.cls
        own = setup.class_named(cls.name)
        if (own.index, own.radius_m) != (cls.index, cls.radius_m):
            raise ValueError(
                f"class {cls.name!r} is class {own.index} at {own.radius_m!r} m in the setup, "
                f"not {cls.index} at {cls.radius_m!r} m"
            )
        demand = tuple(weights)
        placed = [d for wp in entry.stops for d in wp.devices]
        if set(placed) != set(demand):
            raise ValueError(
                f"class {cls.name!r}: its S3a stops must cover exactly the demand; missing "
                f"{sorted(map(str, set(demand) - set(placed)))}, extra "
                f"{sorted(map(str, set(placed) - set(demand)))}"
            )
        outside = sorted(map(str, cap.capped - set(demand)))
        if outside:
            raise ValueError(f"capped devices must be demanded: {outside}")
        undated = sorted(str(d) for d in demand if d not in deadlines)
        if undated:
            raise ValueError(f"every demanded device needs its S3 deadline: {undated}")

        options = setup.options
        self.cls = cls
        self.model: FeasibilityModel = cls.model
        self.params = options.score
        self.rank = applied_rank(options.score)
        self.bounds = options.search
        self.subset = options.member_admission == MEMBER_ADMISSION_SUBSET
        self.t_ref_s = setup.t_ref_s
        self.turnaround_s = setup.turnaround_s
        self.p_hover_w = setup.p_hover_w
        self.pass_2 = entry.pass_2
        self.start = start
        self.budget_end = _budget_end(budget_end)
        self.deadlines = deadlines
        self.device_states = device_states
        self.cap = cap
        self.capped: FrozenSet[DeviceID] = cap.capped
        self.weights = dict(weights)
        self.n_demand = len(demand)
        # B2's deadlines on the whole stops (U1's hand-off), in canonical order
        # (position, then devices, as order_contacts sorts), so every search
        # is a function of the stop set, not of the order S3a listed it in.
        self.stops: Tuple[ContactWaypoint, ...] = tuple(sorted(
            with_cap_deadlines(entry.stops, deadlines=deadlines, capped=self.capped),
            key=_stop_key,
        ))
        # Each member's outage on this class at its offered stop (its S3a
        # stop, or its hover stop): a reduced stop keeps the position, so the
        # distance, and the outage, stay the same.
        ferry = self.model.ferry
        self.outage: Dict[DeviceID, float] = {}
        for wp in self.stops:
            for did, dist in zip(wp.devices, ferry.member_distances_m(wp)):
                self.outage[did] = float(cls.outage(dist))
        self.n_candidates = 0
        self.best: Optional[Candidate] = None
        self._best_key: Tuple[Any, ...] = ()
        self._empty: Optional[Candidate] = None
        self._empty_key: Tuple[Any, ...] = ()
        self._orders: Dict[Any, Tuple[DeviceID, ...]] = {}
        self._dwells: Dict[Any, float] = {}

    # -- pricing ---------------------------------------------------------- #

    def _order(self, wp: ContactWaypoint) -> Tuple[DeviceID, ...]:
        """The F member order of ``wp`` (U3's ``member_order``), computed once per stop."""
        key = _stop_key(wp)
        if key not in self._orders:
            self._orders[key] = member_order(wp, model=self.model, capped=self.capped,
                                             weights=self.weights)
        return self._orders[key]

    def _dwell(self, wp: ContactWaypoint) -> float:
        """``wp``'s predicted Pass-1 dwell, as the predicate charges it
        (``FerryPhysics.dwell_s``: a plan class always prices per member)."""
        key = _stop_key(wp)
        if key not in self._dwells:
            self._dwells[key] = self.model.ferry.dwell_s(wp, _COLLECT)
        return self._dwells[key]

    def _admit(
        self, state: FlightState, wp: ContactWaypoint,
    ) -> Optional[Tuple[ContactWaypoint, Verdict, Tuple[Tuple[ContactWaypoint, str], ...]]]:
        """``wp`` flown from ``state``: whole if it fits, else (subset admission)
        reduced greedily in the F order; None when nobody fits. Returns the stop
        flown, its verdict and the complements the walk refused."""
        if self.subset:
            got = admit_stop(self.model, state, wp, rule=RULE_DEADLINE_BUDGET,
                             budget_end=self.budget_end, deadlines=self.deadlines,
                             device_states=self.device_states, capped=self.capped,
                             weights=self.weights, order=self._order)
            if got.stop is None:
                return None
            return got.stop, got.verdict, got.dropped  # type: ignore[return-value]
        v = self.model.admit(state, wp, rule=RULE_DEADLINE_BUDGET, budget_end=self.budget_end,
                             pass_kind=_COLLECT, protected=is_exempt(wp, self.capped))
        return (wp, v, ()) if v.ok else None

    def _consider(
        self,
        route: Tuple[ContactWaypoint, ...],
        dropped: Tuple[Tuple[ContactWaypoint, str], ...],
        state: FlightState,
        home: float,
        dwell_s: float,
    ) -> Tuple[Candidate, Tuple[Any, ...]]:
        """Score one admitted plan, count it, and keep it if its plan key (U2's
        ``plan_key`` under this arm's rank) is the smallest so far. Returns the
        candidate and its key."""
        served = frozenset(d for wp in route for d in wp.devices)
        energy_j = pass_energy_j(self.model, state)
        p2 = self.pass_2
        terms = score(
            self.params, weights=self.weights,
            served_outage={d: self.outage[d] for d in served},
            pass_1_s=home - self.start.clock, pass_1_dwell_s=dwell_s,
            pass_1_energy_j=energy_j, turnaround_s=self.turnaround_s,
            pass_2_s=p2.time_s, pass_2_dwell_s=p2.dwell_s, pass_2_energy_j=p2.energy_j,
            t_ref_s=self.t_ref_s, p_hover_w=self.p_hover_w,
        )
        fold = MemberFold(route=route, dropped=dropped, state=state, home=home, feasible=True,
                          energy_j=energy_j)
        cand = Candidate(cls=self.cls, fold=fold, terms=terms, cap_key=cap_key(served, self.cap))
        key = plan_key(self.params, cand)
        self.n_candidates += 1
        if self.best is None or key < self._best_key:
            self.best, self._best_key = cand, key
        return cand, key

    # -- the searches ----------------------------------------------------- #

    def run(self) -> ClassResult:
        mode = search_mode(self.n_demand, len(self.stops), self.bounds)
        # The empty plan: nothing flown, so home is the flight back from takeoff
        # (0 s at the dock) and V still pays the turnaround.
        self._empty, self._empty_key = self._consider((), (), self.start,
                                                      self.model.home_at(self.start), 0.0)
        extra: Dict[str, Any] = {}
        if mode == SEARCH_EXACT:
            self._exact()
        elif mode == SEARCH_STOP_SUBSETS:
            self._stop_subsets()
        else:
            extra = self._local()
        best = self.best
        assert best is not None  # the empty plan at least
        summary = {
            "band": self.cls.name,
            "index": self.cls.index,
            "mode": mode,
            "stops": len(self.stops),
            "candidates": self.n_candidates,
            "v": best.terms.v,
            "served": len(best.served),
            "cap_key": list(best.cap_key),
            "served_share": served_share(best.terms),
            "coverage_rank": self.rank,
            **extra,
        }
        return ClassResult(best=best, mode=mode, n_candidates=self.n_candidates,
                           summary=MappingProxyType(summary))

    def _exact(self) -> None:
        """Every ordered sequence of distinct stops, each reduced to every
        non-empty subset of its members (``whole``: the full set only) that the
        predicate admits from where its prefix leaves the mule."""
        stops, capped = self.stops, self.capped
        masks: List[List[int]] = []
        for wp in stops:
            full = (1 << len(wp.devices)) - 1
            if self.subset:
                # Smallest first, so a refused set is met before its supersets.
                masks.append(sorted(range(1, full + 1), key=lambda b: (bin(b).count("1"), b)))
            else:
                masks.append([full])
        trials: Dict[Tuple[int, int], Tuple[ContactWaypoint, bool]] = {}

        def trial(i: int, mask: int) -> Tuple[ContactWaypoint, bool]:
            if (i, mask) not in trials:
                wp = stops[i]
                members = [d for k, d in enumerate(wp.devices) if mask >> k & 1]
                t = reduce_stop(wp, members, deadlines=self.deadlines,
                                device_states=self.device_states, capped=capped)
                trials[(i, mask)] = (t, is_exempt(t, capped))
            return trials[(i, mask)]

        def visit(route: Tuple[ContactWaypoint, ...], state: FlightState, dwell_s: float,
                  used: int) -> None:
            for i in range(len(stops)):
                if used >> i & 1:
                    continue
                refused: List[int] = []
                for mask in masks[i]:
                    # Down-closure: from one state, a superset of a refused set is refused.
                    if any(mask & r == r for r in refused):
                        continue
                    t, exempt = trial(i, mask)
                    v = self.model.admit(state, t, rule=RULE_DEADLINE_BUDGET,
                                         budget_end=self.budget_end, pass_kind=_COLLECT,
                                         protected=exempt)
                    if not v.ok:
                        refused.append(mask)
                        continue
                    child, spent = route + (t,), dwell_s + self._dwell(t)
                    self._consider(child, (), v.next_state, v.home, spent)
                    visit(child, v.next_state, spent, used | 1 << i)

        visit((), self.start, 0.0, 0)

    def _stop_subsets(self) -> None:
        """Every ordered subset of the stops, each walked whole if it fits,
        else reduced greedily; a stop that admits nobody prunes its prefix."""
        stops = self.stops

        def visit(route: Tuple[ContactWaypoint, ...],
                  dropped: Tuple[Tuple[ContactWaypoint, str], ...],
                  state: FlightState, dwell_s: float, used: int) -> None:
            for i in range(len(stops)):
                if used >> i & 1:
                    continue
                got = self._admit(state, stops[i])
                if got is None:
                    continue
                wp, v, drops = got
                child, lost, spent = route + (wp,), dropped + drops, dwell_s + self._dwell(wp)
                self._consider(child, lost, v.next_state, v.home, spent)
                visit(child, lost, v.next_state, spent, used | 1 << i)

        visit((), (), self.start, 0.0, 0)

    # -- the local search -------------------------------------------------- #

    def _local(self) -> Dict[str, Any]:
        """The 2-OPT tour, U3's member trim of it, then the first-improvement
        scans (one under ``weighted``, up to three under ``lexicographic``),
        bounded together by passes and by walks."""
        stops = self.stops
        owner = {d: i for i, wp in enumerate(stops) for d in wp.devices}
        full = [(1 << len(wp.devices)) - 1 for wp in stops]
        tour = order_contacts(stops, self.start.pose, end=self.model.ferry.dock)
        if self.subset:
            trimmed = trim_members(tour, self.start, model=self.model, budget_end=self.budget_end,
                                   deadlines=self.deadlines, device_states=self.device_states,
                                   capped=self.capped, weights=self.weights, order=self._order,
                                   rule=RULE_DEADLINE_BUDGET).route
        else:
            # Whole stops: the tour with its priority stops first, as the trim
            # orders them, keeping each stop that fits whole.
            ahead = priority_first(tour, self.capped)
            trimmed = self.model.fold(ahead, self.start, rule=RULE_DEADLINE_BUDGET,
                                      budget_end=self.budget_end, pass_kind=_COLLECT, skip=True,
                                      protected=cap_stops(ahead, self.capped).exempt).route

        memo: Dict[Any, Any] = {}

        def entries_of(route: Sequence[ContactWaypoint]) -> _Entries:
            out = []
            for wp in route:
                i = owner[wp.devices[0]]
                out.append((i, sum(1 << stops[i].devices.index(d) for d in wp.devices)))
            return tuple(out)

        def walk(entries: _Entries):
            """The plan these entries fly (each stop limited to its mask, then
            whole if it fits, else reduced greedily), or None if a stop admits
            nobody. Admissions are memoised by (state, stop): a move keeps its
            prefix, so only the rest is priced again."""
            state, home = self.start, None
            route: List[ContactWaypoint] = []
            dropped: List[Tuple[ContactWaypoint, str]] = []
            dwell_s = 0.0
            for i, mask in entries:
                wp = stops[i]
                key = (_state_key(state), i, mask)
                if key not in memo:
                    limited = wp if mask == full[i] else reduce_stop(
                        wp, [d for k, d in enumerate(wp.devices) if mask >> k & 1],
                        deadlines=self.deadlines, device_states=self.device_states,
                        capped=self.capped)
                    memo[key] = self._admit(state, limited)
                got = memo[key]
                if got is None:
                    return None
                flown, v, drops = got
                route.append(flown)
                dropped.extend(drops)
                dwell_s += self._dwell(flown)
                state, home = v.next_state, v.home
            if home is None:
                home = self.model.home_at(state)
            return tuple(route), tuple(dropped), state, home, dwell_s

        def moves(entries: _Entries) -> Iterator[_Entries]:
            """The neighbours of ``entries`` in the documented scan order."""
            k = len(entries)
            routed = {i for i, _ in entries}
            for p in range(k):                                         # drop a stop
                yield entries[:p] + entries[p + 1:]
            for j in range(len(stops)):                                # insert a stop
                if j not in routed:
                    for p in range(k + 1):
                        yield entries[:p] + ((j, full[j]),) + entries[p:]
            for a in range(k - 1):                                     # reverse a segment
                for b in range(a + 1, k):
                    yield entries[:a] + tuple(reversed(entries[a:b + 1])) + entries[b + 1:]
            if self.subset:                                            # drop a served member
                for p, (i, mask) in enumerate(entries):
                    wp = stops[i]
                    for did in reversed(self._order(wp)):              # least worth first
                        bit = 1 << wp.devices.index(did)
                        if mask & bit and mask != bit:
                            yield entries[:p] + ((i, mask & ~bit),) + entries[p + 1:]

        def flown(entries: _Entries) -> Optional[Tuple[_Entries, Tuple[Any, Any]]]:
            """The route ``entries`` fly, as its entries, with its keys (the plan
            key, then U0's ``Candidate.key``: one key under ``weighted``), or
            None if a stop admits nobody. The empty plan is already scored."""
            walked = walk(entries)
            if walked is None:
                return None
            if not walked[0]:
                assert self._empty is not None
                return (), (self._empty_key, self._empty.key)
            cand, key = self._consider(*walked)
            return entries_of(walked[0]), (key, cand.key)

        # The trim's route passes the predicate by construction, and each of its
        # stops is admitted whole by the walk from the state the trim gave it.
        begin = entries_of(trimmed)
        first = flown(begin)
        if first is None:
            raise ValueError(
                f"class {self.cls.name!r}: the trimmed tour must pass the predicate as flown"
            )
        # Every walk run so far, by the entries walked. A walk depends on its
        # entries alone, so a later scan reuses an earlier one's walk.
        met: Dict[_Entries, Optional[Tuple[_Entries, Tuple[Any, Any]]]] = {begin: first}
        max_evaluations = self.bounds.heuristic_max_evaluations
        max_passes = self.bounds.heuristic_max_passes
        evaluations, passes, bounded = 1, 0, False

        def scan(current: _Entries, here: Tuple[Any, Any], by: int) -> None:
            """One scan from ``current`` (its keys ``here``), moving on key ``by``
            (0: the plan key, 1: the weighted key), within the class's bounds.
            The scan moves between routes as flown: each stop limited to the
            members it serves, so a member a walk refused is not kept in reserve."""
            nonlocal evaluations, passes, bounded
            seen = {current}
            improved = True
            while improved:
                if passes == max_passes:
                    bounded = True
                    break
                passes += 1
                improved = False
                for entries in moves(current):
                    if entries in seen:
                        # Walked in this scan: it did not beat the current then,
                        # and the current has only improved since.
                        continue
                    if entries in met:
                        # Walked by an earlier scan: reused, not walked or counted.
                        got = met[entries]
                    else:
                        if evaluations == max_evaluations:
                            bounded = True
                            break
                        evaluations += 1
                        got = met[entries] = flown(entries)
                    seen.add(entries)
                    if got is not None and got[1][by] < here[by]:
                        (current, here), improved = got, True
                        seen.add(current)
                        break
                if bounded:
                    break

        if self.rank == COVERAGE_RANK_WEIGHTED:
            scan(*first, 0)
            return {"evaluations": evaluations, "passes": passes, "bounded": bounded}
        # Under lexicographic: weighted's own scan, then the plan key's from the
        # best plan met, then, when that is another route, from the trim's.
        scan(*first, 1)
        weighted = {"weighted_passes": passes, "weighted_evaluations": evaluations}
        if not bounded:
            assert self.best is not None
            polish = entries_of(self.best.fold.route)
            scan(polish, (self._best_key, self.best.key), 0)
            if not bounded and polish != first[0]:
                scan(*first, 0)
        return {"evaluations": evaluations, "passes": passes, "bounded": bounded, **weighted}


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #

def search_class(
    setup: PlanSetup,
    entry: ClassInput,
    *,
    start: FlightState,
    budget_end: Optional[float],
    deadlines: Mapping[DeviceID, float],
    device_states: Mapping[DeviceID, Any],
    cap: CapState,
    weights: Mapping[DeviceID, float],
) -> ClassResult:
    """Search one class: its best candidate, the empty plan included.

    ``setup`` gives the settings (``options.score``, ``options.search``,
    ``options.member_admission``), T (``t_ref_s``, T_nom), the turnaround and
    P_hover; ``entry`` the class, its offered stops and its Pass 2. ``start`` is
    the takeoff state, ``FlightState(dock, takeoff clock)``, and
    ``budget_end`` the absolute end of the budget (None: no budget, no gate).
    ``deadlines`` are S3's per-device deadlines, which reduced stops take
    theirs from (U1's ``stop_deadline``), ``device_states`` the scheduler's map
    (their buckets), ``cap`` U1's ``evaluate_cap`` and ``weights`` U2's
    ``demand_weights``, whose keys are the demand. The search of one class
    depends on nothing about the others.
    """
    return _ClassSearch(setup, entry, start=start, budget_end=budget_end, deadlines=deadlines,
                        device_states=device_states, cap=cap, weights=weights).run()


def plan_search(
    setup: PlanSetup,
    classes: Sequence[ClassInput],
    *,
    start: FlightState,
    budget_end: Optional[float],
    deadlines: Mapping[DeviceID, float],
    device_states: Mapping[DeviceID, Any],
    cap: CapState,
    weights: Mapping[DeviceID, float],
) -> SearchResult:
    """The plan: the smallest plan key over every class the arm may fly.

    ``classes`` must be exactly ``setup.searched`` (every link class under
    ``search``, the pinned one under ``fixed:<c>``), in that order, each with
    its model bound to the device states, its offered stops and its Pass 2
    (:class:`ClassInput`): FB+c flies only class c (decision 7). Each class is
    searched by :func:`search_class`; the result holds the best candidate
    (the smallest U2 ``plan_key`` under the arm's ``coverage_rank``, the one
    each class's search ranked by), the mode of its class, the candidates
    scored in all and one summary per class. The other arguments are
    :func:`search_class`'s.
    """
    if not isinstance(setup, PlanSetup):
        raise TypeError(f"setup must be a PlanSetup, got {setup!r}")
    entries = tuple(classes)
    for entry in entries:
        if not isinstance(entry, ClassInput):
            raise TypeError(f"classes hold ClassInput entries, got {entry!r}")
    want = [c.name for c in setup.searched]
    got = [entry.cls.name for entry in entries]
    if got != want:
        raise ValueError(
            f"band_class_policy {setup.options.band_class_policy!r} searches {want}, in "
            f"link order; got {got}"
        )
    results = [
        search_class(setup, entry, start=start, budget_end=budget_end, deadlines=deadlines,
                     device_states=device_states, cap=cap, weights=weights)
        for entry in entries
    ]
    params = setup.options.score
    best = min(results, key=lambda r: plan_key(params, r.best))
    return SearchResult(best=best.best, mode=best.mode,
                        n_candidates=sum(r.n_candidates for r in results),
                        per_class=tuple(r.summary for r in results))
