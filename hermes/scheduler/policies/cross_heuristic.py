"""FeRRy Phase 4: the flight slot's two fixed fillings, arm F's and arm FX's (unit U6).

**The slot.** In flight the mule chooses, stop by stop, where to fly next and
on which band class to serve the stop it reaches (build plan L539: "Choose
(band, next stop)"). Phase 5 fills that slot with the learned pair Q (L859).
Phase 4 fills it with two fixed rules, named by ``MuleConfig.flight_slot``
(:data:`~hermes.scheduler.plan.types.FLIGHT_SLOTS`, recorded value first):

* ``committed`` (:class:`CommittedSlot`; arm F and the FB+ arms): the plan's
  next stop on the committed class b̄. That is today's ``remainder.pop(0)`` on
  the runtime's band; the slot reads neither the channel nor the predicate.
* ``cross_heuristic`` (:class:`CrossHeuristic`; arm FX, the user's decision 5):

  - **next stop** (:meth:`CrossHeuristic.next_stop`): after each stop, the
    nearest remaining stop whose move to the front keeps the rest of the plan
    feasible; when none does, the plan's next stop (index 0);
  - **band on arrival** (:meth:`CrossHeuristic.band_at_arrival`): the fastest
    class that still reaches every device the committed class reaches there,
    priced at the arrival SNR, so FX never dwells longer than F would at that
    stop.

**Where the slot acts: Pass 1, after takeoff.** The slot is Pass 1's. The
build plan's flight clock runs from takeoff through "Arrive and observe",
"Choose (band, next stop)", "Serve" and "Check and re-plan" (L536-L541).
Pass 2 is the merge clock's delivery (L545), which walks every slice contact
nearest-first with no selector (``FLScheduler.build_pass_2_queue``). The
Phase 4 spec says it of the band (other choices 2): "FX switches band per stop
in Pass 1 only; Pass 2 flies b̄". So in Pass 2, FX is the committed slot: the
queue's next stop, on b̄. In Pass 1 the band half acts at every arrival, the
first stop's included. The next-stop half acts after each stop, in decision
5's words. At takeoff the mule flies the committed plan's first stop, as the
plan's loop does ("take off", then "Arrive at stop k"): nothing has been
observed in flight yet, and that order is the plan search's own choice from
the dock. FX chooses the next stop at the departure, after the stop is served
and the departure check has run (the design's section 2.7), so it reads the
state the service left.

The supervisor calls the slot at every stop of both passes, with the pass and
whether the mule departs from a stop it served (``after_stop``). The slot
applies these rules itself, so no call site has to, and the mule's runtime
backs the band rule: ``FerryRuntime.contact_plan`` refuses to build a Pass-2
contact plan on any class but its own band, b̄.

**Why the fastest class, not the one that reaches most (critic A7).** The
design's first rule took the class reaching the most members, then the least
dwell. Critic probe E replayed the seeded channel at F's own predicted
arrivals (1 MB, N = 6, 8 time offsets): at a 60 s budget, 8 of FX's 50
switches went to a narrower class for one more member, adding a median 60.5 s
and up to 94.1 s of dwell, and 6 last-stop switches (4 at 90 s) put the
landing past a budget that b̄ would have met. After the last Pass-1 stop no
departure check runs (``mule_main.py`` flies home), so nothing catches that.
The plan's own words are the "best-rate band" (L539, L859, L978). Among the
classes that reach every committed target, b̄ itself always qualifies, so the
fastest of them never dwells longer than b̄ (at the arrival SNR) and never
reaches fewer devices; the in-flight check, which prices the remainder on b̄,
stays conservative for whatever class FX flies.

**What the slot reads.** Stops are ``ContactWaypoint`` objects; ``state`` is the
departure's flight state (``s3b_feasibility.FlightState``; only its ``pose`` is
read here); ``fits`` is the caller's test of a reordered remainder, the
supervisor's ``FLScheduler.fold_remainder(...).ok`` from that state with the
plan's protected stops, so the rule re-implements no predicate; ``pass_kind``
is the pass flown, a ``MissionPass`` or its value. The band rule reads an
:class:`~hermes.scheduler.plan.types.ArrivalView`, which the mule's runtime
builds (``FerryRuntime.arrival_view``) and which is scheduler-side data
(critic B14): this module imports neither ``hermes.mule`` nor ``hermes.l1``
nor ``experiments``, and no numpy.

Both rules are deterministic, with no wall time and no draw: distance ties
fall to the stop's position, then its devices, then its place in the plan;
dwell ties to more targets, then the committed class, then the class index.

Plan mode only: this module imports ``hermes.scheduler.plan``, so the
policies package does not import it, and nothing on a legacy path may.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from hermes.scheduler.plan.types import (
    FLIGHT_SLOT_COMMITTED,
    FLIGHT_SLOT_CROSS_HEURISTIC,
    FLIGHT_SLOTS,
    ArrivalClass,
    ArrivalView,
)
from hermes.types.scheduler import ContactWaypoint, MissionPass

__all__ = [
    "Fits",
    "CommittedSlot",
    "CrossHeuristic",
    "FlightSlot",
    "fastest_covering_class",
    "flight_slot_policy",
    "moved_to_front",
    "nearest_first",
]

#: ``fits(order) -> bool``: does the remainder, flown in ``order`` from the
#: departure state, still pass the arm's in-flight rule? The supervisor binds
#: ``FLScheduler.fold_remainder(order, state=..., budget_end=..., pass_kind=...,
#: protected=...).ok``.
Fits = Callable[[Sequence[ContactWaypoint]], bool]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    """The leg's length, with the predicate's own metric (``s3b_feasibility._euclid``)."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _remainder(remainder: Sequence[ContactWaypoint]) -> List[ContactWaypoint]:
    stops = list(remainder)
    if not stops:
        raise ValueError("the flight slot picks among the remaining stops: none is left")
    for wp in stops:
        if not isinstance(wp, ContactWaypoint):
            raise TypeError(f"the remainder holds ContactWaypoints, got {wp!r}")
    return stops


def _pass(pass_kind: Any) -> MissionPass:
    """``pass_kind`` as a ``MissionPass``; its value (``"collect"``) names it too.

    Required at every call, so no call site can leave the slot to assume
    Pass 1 at a Pass-2 stop.
    """
    try:
        return MissionPass(pass_kind)
    except ValueError:
        raise ValueError(
            f"pass_kind must name a mission pass {[p.value for p in MissionPass]}, "
            f"got {pass_kind!r}") from None


def _after_stop(after_stop: Any) -> bool:
    """``after_stop`` checked as a bool, so a stop or a count passed by mistake is refused."""
    if not isinstance(after_stop, bool):
        raise TypeError(
            "after_stop is True when the mule departs from a stop it has served and "
            f"False at takeoff, got {after_stop!r}")
    return after_stop


def nearest_first(remainder: Sequence[ContactWaypoint], pose: Sequence[float]) -> List[int]:
    """The remainder's indices, the stop nearest ``pose`` first.

    Ties in distance fall to the stop's position, then its devices (the
    design's D-E), then its place in the plan, so the order is total and
    depends on nothing but the stops.
    """
    stops = _remainder(remainder)
    pose = tuple(pose)
    return sorted(
        range(len(stops)),
        key=lambda i: (_distance(pose, stops[i].position), tuple(stops[i].position),
                       tuple(stops[i].devices), i),
    )


def moved_to_front(remainder: Sequence[ContactWaypoint], index: int) -> List[ContactWaypoint]:
    """``remainder`` with its stop ``index`` flown first and the rest in plan order."""
    stops = list(remainder)
    return [stops[index]] + stops[:index] + stops[index + 1:]


def fastest_covering_class(view: ArrivalView) -> ArrivalClass:
    """The class FX flies at this arrival (decision 5, critic A7).

    Among the classes whose targets include every target of the committed
    class, the one with the least dwell at the arrival SNR; ties go to the
    class reaching more devices, then to the committed class (no switch for
    nothing), then to the lower class index. The committed class is always
    among the candidates, so the pick never dwells longer than it and never
    reaches fewer devices.
    """
    if not isinstance(view, ArrivalView):
        raise TypeError(f"the band rule reads an ArrivalView, got {view!r}")
    need = set(view.committed_entry.targets)
    candidates = [c for c in view.classes if need.issubset(c.targets)]
    return min(candidates,
               key=lambda c: (c.dwell_s, -len(c.targets), c.name != view.committed, c.index))


class CommittedSlot:
    """Arm F's slot (and the FB+ arms'): the plan's next stop on the committed class.

    Exactly today's flight, in both passes: :meth:`next_stop` is index 0
    (``remainder.pop(0)``) and :meth:`band_at_arrival` keeps the runtime's
    band. It never calls ``fits`` and reads no arrival view
    (:meth:`reads_arrival_view`), so the supervisor builds none. Its methods
    take :class:`CrossHeuristic`'s arguments, so one call site serves both.
    """

    name: str = FLIGHT_SLOT_COMMITTED

    def next_stop(self, remainder: Sequence[ContactWaypoint], state: Any, *,
                  fits: Optional[Fits], pass_kind: MissionPass, after_stop: bool) -> int:
        """Index 0: the plan's next stop. ``state`` and ``fits`` are not read."""
        _remainder(remainder)
        _pass(pass_kind)
        _after_stop(after_stop)
        return 0

    def reads_arrival_view(self, pass_kind: MissionPass) -> bool:
        """False: the committed class needs no view of the others."""
        _pass(pass_kind)
        return False

    def band_at_arrival(self, view: Optional[ArrivalView], *,
                        pass_kind: MissionPass) -> Optional[str]:
        """None: fly the runtime's band, the committed class. ``view`` is not read."""
        _pass(pass_kind)
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class CrossHeuristic:
    """Arm FX's slot: the nearest feasible stop, then the fastest covering class.

    Pass 1 only, and its next-stop half only after a stop: in Pass 2 and at
    takeoff it flies as :class:`CommittedSlot` does (module docstring). See
    there also for why its band half is the fastest class rather than the one
    that reaches most (critic A7).
    """

    name: str = FLIGHT_SLOT_CROSS_HEURISTIC

    def next_stop(self, remainder: Sequence[ContactWaypoint], state: Any, *,
                  fits: Fits, pass_kind: MissionPass, after_stop: bool) -> int:
        """The index of the stop to fly next.

        In Pass 1, after a stop: candidates are tried nearest to
        ``state.pose`` first (:func:`nearest_first`), and the first whose move
        to the front (:func:`moved_to_front`) passes ``fits`` is returned.
        When none does, 0: the plan's next stop, whose shortfall the departure
        check and the re-plan already handle. A lone stop is 0 without a fold.
        At takeoff (``after_stop`` False) and in Pass 2, 0 without a fold: the
        committed order.
        """
        stops = _remainder(remainder)
        if not callable(fits):
            raise TypeError(
                "the cross-heuristic needs fits(order) -> bool: it never reorders unchecked")
        collect = _pass(pass_kind) is MissionPass.COLLECT
        after = _after_stop(after_stop)
        if not (collect and after) or len(stops) == 1:
            return 0
        for index in nearest_first(stops, state.pose):
            if fits(moved_to_front(stops, index)):
                return index
        return 0

    def reads_arrival_view(self, pass_kind: MissionPass) -> bool:
        """True in Pass 1, where :meth:`band_at_arrival` compares the classes."""
        return _pass(pass_kind) is MissionPass.COLLECT

    def band_at_arrival(self, view: Optional[ArrivalView], *,
                        pass_kind: MissionPass) -> Optional[str]:
        """The class to fly at this stop, or None to keep the committed class.

        In Pass 1, :func:`fastest_covering_class` of ``view``, and None when
        that is the committed class, so ``band_at_arrival(...) or
        runtime.band`` is the class the contact plan is built on. In Pass 2,
        None without reading ``view``, which the supervisor does not build
        there (:meth:`reads_arrival_view`): Pass 2 flies b̄.
        """
        if _pass(pass_kind) is not MissionPass.COLLECT:
            return None
        best = fastest_covering_class(view)
        return None if best.name == view.committed else best.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


#: What the supervisor holds in the slot.
FlightSlot = Union[CommittedSlot, CrossHeuristic]

#: One filling per ``FLIGHT_SLOTS`` value, in its order (a test pins this).
_SLOTS: Dict[str, Type[Any]] = {
    FLIGHT_SLOT_COMMITTED: CommittedSlot,
    FLIGHT_SLOT_CROSS_HEURISTIC: CrossHeuristic,
}


def flight_slot_policy(name: str) -> FlightSlot:
    """The filling ``MuleConfig.flight_slot`` names (``PlanOptions.flight_slot``).

    Unknown names are refused. ``PlanOptions`` already refuses a pinned band
    (``fixed:<class>``) with ``cross_heuristic``: FB+c flies only class c.
    """
    try:
        slot = _SLOTS[name]
    except (KeyError, TypeError):
        raise ValueError(f"flight_slot must be one of {FLIGHT_SLOTS}, got {name!r}") from None
    return slot()
