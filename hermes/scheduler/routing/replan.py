"""Re-plan the rest of a mission when it stops fitting (FeRRy Phase 3).

**Why it exists.** Before Phase 3 the mule re-checked only the *next* stop and,
when that failed, aborted the whole tail of its queue (Freeze Amendment 8, the
``abort`` response). On the simulated mission clock a remainder that no longer
fits can often be repaired: leave out what cannot be served in time and fly the
rest. This module is the scheduler-side half of the ``replan`` response
(``MuleConfig.in_flight_response``; design §3.4): the mule calls
``FLScheduler.replan_remainder`` at each departure, and before takeoff the
scheduler validates the order it is about to fly (design §3.3). The mule does
not re-implement S3b (design principle 1).

**The algorithm** (design §3.4 with critic C3, binding):

1. If the current order passes the predicate without skipping, keep it.
2. **Admission.** Protected stops go first (Phase 4's age-capped devices; the
   set is empty in Phase 3), folded in their current relative order; then the
   arm's own admission decides the rest, starting where the protected prefix
   ends: S3b for our arms, the baseline's own ``admit_and_order`` for D1-D3
   and D5, a nearest-first budget walk for Pass 2. D4 never re-plans.
3. **Order.** Keep the arm's own relative order over the admitted stops if it
   passes (critic C3: re-ordering by distance would erase H1's bucket order
   and H2's learned order, so the H arms would become identical whenever the
   budget binds). Otherwise, with the default ``reorder`` fallback, try
   ``two_opt.order_contacts(admitted, pose, end=dock)``, when the caller
   allows it, and then fly the admission order, which passes by construction;
   a final fold that skips guards that promise against an admission that does
   not price with the same predicate. With the ``trim`` fallback, keep the
   arm's order instead and leave out the stops it cannot serve in that order
   (a fold that skips; protected stops keep their admitted place in front).
4. Return :class:`ReplanResult` ``(route, dropped[(wp, reason)], order_used)``.

**Where the order preference cannot help.** Before takeoff (design §3.3) the
remainder is the whole plan and S3b has just admitted all of it from the same
state, so for our arms the admission keeps every stop and "the arm's own order
over the admitted stops" is the flown order that just failed. Under
``reorder`` the repaired route is then 2-OPT's or the admission (EDF) order,
both functions of the admitted *set* alone: arms that differ only in their
order (H1, H2, H3) fly the same repaired route whenever the check fires, and
the check never drops a stop. ``trim`` is the alternative that keeps each
arm's order at the price of serving fewer stops; which one the exit-gate arms
use (or whether they run ``abort``, critic C3's other option) is chosen at the
pilot, not here (critic B5).

**Invariants** (``tests/unit/test_p3_replan.py`` checks them on random
instances): the returned route passes the predicate's fold without skipping
from the given state; it is a subset of the remainder, each stop once; every
other stop of the remainder is in ``dropped`` with a reason, in the
remainder's order; ``trim`` never re-orders; and a protected stop is dropped
only if the protected-only route fails.

Drops are final for the mission (spec Q10): the mule widens every dropped
Pass-1 device and records every dropped Pass-2 device as a skipped delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Collection, Dict, List, Optional, Sequence, Tuple

from hermes.types.scheduler import ContactWaypoint, MissionPass

from hermes.scheduler.routing.two_opt import order_contacts
from hermes.scheduler.stages.s3b_feasibility import (
    REASON_BUDGET,
    FeasibilityModel,
    FlightState,
    FoldResult,
)

#: The current order passes: nothing changed.
ORDER_CURRENT = "current"
#: The arm's own relative order over the admitted stops.
ORDER_ARM = "arm"
#: The ``trim`` fallback: the arm's own relative order with the stops it
#: cannot serve in that order left out.
ORDER_ARM_TRIMMED = "arm_trimmed"
#: The 2-OPT fallback (a fixed-end path to the dock).
ORDER_TWO_OPT = "two_opt"
#: The admission walk's own order.
ORDER_ADMISSION = "admission"
#: The arm never re-plans (D4, ``IN_FLIGHT_NONE``): the remainder is flown as is.
ORDER_NONE = "none"
ORDERS: Tuple[str, ...] = (
    ORDER_CURRENT, ORDER_ARM, ORDER_ARM_TRIMMED, ORDER_TWO_OPT, ORDER_ADMISSION, ORDER_NONE,
)

#: What the re-plan does when the arm's own order over the admitted stops does
#: not fit. ``reorder`` (the default; design §3.4 with critic C3): the 2-OPT
#: path to the dock (when allowed), then the admission order.
FALLBACK_REORDER = "reorder"
#: ``trim``: keep the arm's order and leave out what it cannot serve in it.
FALLBACK_TRIM = "trim"
FALLBACKS: Tuple[str, ...] = (FALLBACK_REORDER, FALLBACK_TRIM)

Point = Sequence[float]

#: An arm's admission over a set of stops from a flight state: returns the
#: admitted stops in the arm's admission order (the objects it was given) and,
#: optionally, why the others were left out. A stop without a reason is
#: reported as over budget.
Admission = Callable[
    [List[ContactWaypoint], FlightState],
    Tuple[Sequence[ContactWaypoint], Sequence[Tuple[ContactWaypoint, str]]],
]


@dataclass(frozen=True)
class ReplanResult:
    """What to fly next and what was given up.

    ``route`` is the new remainder, in flying order; ``dropped`` the stops of
    the old remainder that are not in it, in their old order, each with the
    predicate's reason (``overdue``, ``budget`` or ``energy``);
    ``order_used`` one of :data:`ORDERS`.
    """

    route: Tuple[ContactWaypoint, ...]
    dropped: Tuple[Tuple[ContactWaypoint, str], ...]
    order_used: str

    @property
    def changed(self) -> bool:
        """True when the re-plan dropped or re-ordered anything."""
        return self.order_used not in (ORDER_CURRENT, ORDER_NONE)

    @property
    def dropped_contacts(self) -> List[ContactWaypoint]:
        return [wp for wp, _ in self.dropped]

    def dropped_by(self, reason: str) -> List[ContactWaypoint]:
        """The dropped stops with ``reason``, in their old order."""
        return [wp for wp, why in self.dropped if why == reason]


def replan_route(
    remainder: Sequence[ContactWaypoint],
    *,
    state: FlightState,
    model: FeasibilityModel,
    rule: str,
    budget_end: Optional[float],
    pass_kind: MissionPass,
    admission: Admission,
    protected: Collection[ContactWaypoint] = (),
    dock: Optional[Point] = None,
    two_opt_fallback: bool = True,
    snr_offset_db: float = 0.0,
    fallback: str = FALLBACK_REORDER,
) -> ReplanResult:
    """Re-plan ``remainder`` from ``state`` (see the module docstring).

    ``rule`` is the arm's rule for the predicate
    (``FLScheduler.in_flight_rule``); ``admission`` the arm's admission;
    ``protected`` the stops that must be kept if at all possible; ``dock`` the
    end of the 2-OPT fallback path (None: an open path);
    ``two_opt_fallback=False`` skips that fallback (the baselines keep their
    own order). ``snr_offset_db`` is the observed-rate adjustment δ_obs (0 by
    default, spec). ``fallback`` is what happens when the arm's own order over
    the admitted stops does not fit (:data:`FALLBACKS`).
    """
    if fallback not in FALLBACKS:
        raise ValueError(f"replan: fallback must be one of {FALLBACKS}, got {fallback!r}")
    stops = list(remainder)
    index = {id(wp): i for i, wp in enumerate(stops)}
    if len(index) != len(stops):
        raise ValueError("replan: the remainder lists a stop twice")
    guarded = tuple(wp for wp in stops if protected and wp in protected)
    guarded_ids = {id(wp) for wp in guarded}

    def fold(route: Sequence[ContactWaypoint], start: FlightState, skip: bool) -> FoldResult:
        return model.fold(
            route, start, rule=rule, budget_end=budget_end, pass_kind=pass_kind,
            skip=skip, protected=guarded, snr_offset_db=snr_offset_db,
        )

    # 1. The current order.
    if fold(stops, state, False).ok:
        return ReplanResult(tuple(stops), (), ORDER_CURRENT)

    dropped: List[Tuple[ContactWaypoint, str]] = []
    # 2a. Protected stops first, in their current relative order. One is
    # dropped only where the protected-only route itself fails.
    head = fold(list(guarded), state, True)
    dropped.extend(head.rejected)
    # 2b. The arm's own admission decides the rest, from where that prefix ends.
    rest = [wp for wp in stops if id(wp) not in guarded_ids]
    admitted_rest, reasons = admission(list(rest), head.state)
    admitted_rest = list(admitted_rest)
    rest_ids = {id(wp) for wp in rest}
    kept_ids = set()
    for wp in admitted_rest:
        if id(wp) not in rest_ids or id(wp) in kept_ids:
            raise ValueError(
                "replan: the admission returned a stop it was not given, or one twice"
            )
        kept_ids.add(id(wp))
    why: Dict[int, str] = {id(wp): r for wp, r in reasons}
    dropped.extend(
        (wp, why.get(id(wp), REASON_BUDGET)) for wp in rest if id(wp) not in kept_ids
    )
    admitted = list(head.route) + admitted_rest

    # 3. Order: the arm's own; else (trim) the arm's own with what it cannot
    # serve left out; else (reorder) 2-OPT, then the admission order.
    keep = {id(wp) for wp in admitted}
    arm = [wp for wp in stops if id(wp) in keep]
    if fold(arm, state, False).ok:
        return _result(arm, dropped, ORDER_ARM, index)
    if fallback == FALLBACK_TRIM:
        # The admitted protected stops keep their place in front, as the
        # admission flies them; the rest is walked in the arm's relative order
        # from where they end. Walking them in place instead could spend the
        # budget on stops the arm put first and then drop a protected one.
        tail = fold([wp for wp in arm if id(wp) not in guarded_ids], head.state, True)
        dropped.extend(tail.rejected)
        return _result(list(head.route) + list(tail.route), dropped, ORDER_ARM_TRIMMED, index)
    if two_opt_fallback and len(admitted) > 1:
        tour = order_contacts(admitted, state.pose, end=dock)
        if fold(tour, state, False).ok:
            return _result(tour, dropped, ORDER_TWO_OPT, index)
    guard = fold(admitted, state, True)
    dropped.extend(guard.rejected)
    return _result(list(guard.route), dropped, ORDER_ADMISSION, index)


def _result(
    route: Sequence[ContactWaypoint],
    dropped: List[Tuple[ContactWaypoint, str]],
    order_used: str,
    index: Dict[int, int],
) -> ReplanResult:
    ordered = sorted(dropped, key=lambda pair: index[id(pair[0])])
    return ReplanResult(tuple(route), tuple(ordered), order_used)
