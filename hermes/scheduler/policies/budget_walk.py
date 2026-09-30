"""Shared greedy budget walk for whole-scheduler baseline arms (D1, D2).

**What it is.** "Visit the best-ranked contacts until the mission budget runs
out." That is the obvious reading of *any* ranking policy operating under a time
budget, and it is how the baselines are given **admission authority** rather than
merely reordering a list our gate already decided.

**Why it is shared, and why it reuses S3b's cost model.** The comparison is
between *decision rules*, so every arm must price travel identically — otherwise
the experiment measures whose travel model is cheaper, not whose policy is
better. Both baselines therefore walk the route with the same
:class:`~hermes.scheduler.stages.s3b_feasibility.FeasibilityModel` that S3b uses:
same cruise speed, same per-contact session time, same Euclidean geometry.

**What differs between arms is only the key.** D1 ranks by age, D2 by Oort's
statistical utility, and our own S3b ranks by per-device deadline with an
adaptive window. Same information, same constraint, different rule.

**Greedy, not optimal.** Choosing the best subset of contacts under a travel
budget is a form of orienteering problem and is NP-hard; no cited baseline solves
it exactly either. Greedy-by-rank is what the literature's "highest AoI first,
nearest predecessor" describes, and solving it optimally for one arm while the
others act greedily would be a different unfairness.

**FeRRy Phase 3.** The walk is a fold, skipping what fails, over S3b's one
predicate (:meth:`FeasibilityModel.admit`) under the budget-only rule. In
legacy mode that is exactly the arithmetic above. With the ferry model every
arm is priced with the same physics as S3b: dwell at rate, the return leg to
the dock, the upload after Pass 1 and, if a capacity is set, the simulated
energy clause.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from hermes.types import ContactWaypoint, MissionPass

from hermes.scheduler.stages.s3b_feasibility import (
    RULE_BUDGET,
    FeasibilityModel,
    FlightState,
)

MulePose = Tuple[float, float, float]

#: What the mule re-checks before each contact in flight, declared by a
#: whole-scheduler policy as ``in_flight_check`` (Freeze Amendment 8).
#: ``budget``: the next contact must still fit the mission budget from the
#: mule's actual pose and clock. ``none``: the route is flown as planned.
#: The per-device deadline test is S3b's rule, so only our own arms get it.
IN_FLIGHT_BUDGET = "budget"
IN_FLIGHT_NONE = "none"
IN_FLIGHT_CHECKS = (IN_FLIGHT_BUDGET, IN_FLIGHT_NONE)

#: Ranking key: lower sorts first. Returning a tuple lets a policy add
#: deterministic tie-breaks after its primary key.
RankKey = Callable[[ContactWaypoint], tuple]


def greedy_budget_walk(
    contacts: Sequence[ContactWaypoint],
    *,
    key: RankKey,
    mule_pose: MulePose,
    now: float,
    mission_deadline_ts: Optional[float],
    model: Optional[FeasibilityModel] = None,
    state: Optional[FlightState] = None,
    pass_kind: MissionPass = MissionPass.COLLECT,
) -> List[ContactWaypoint]:
    """Admit contacts in ``key`` order while the budget allows; return the route.

    Walks in ranked order, advancing a simulated pose and clock exactly as S3b
    does. A contact that would overrun the budget is **skipped, not fatal** — the
    walk continues to consider later ones, because a distant high-rank contact
    should not veto a near low-rank contact that still fits. That is the standard
    greedy-knapsack reading and it is strictly more favourable to the baseline
    than stopping at the first miss.

    With ``mission_deadline_ts=None`` there is no budget, so every contact is
    admitted in ranked order — which keeps the no-enforcement path meaningful
    rather than degenerate.

    ``state`` (FeRRy Phase 3) starts the walk from a flight state, energy
    spent included, instead of ``(mule_pose, now)``; ``pass_kind`` (the enum
    or its string value) picks the ferry payload and whether the Pass-1
    upload tail applies (the budgeted Pass 2 walks with ``DELIVER``). Both
    are inert in legacy mode.
    """
    ordered = sorted(contacts, key=key)
    if mission_deadline_ts is None:
        return ordered

    m = model or FeasibilityModel()
    start = state if state is not None else FlightState(mule_pose, now)
    # Skip, not stop: a contact that does not fit is left out and the walk
    # tries the next one from the same pose and clock.
    return list(m.fold(
        ordered, start, rule=RULE_BUDGET, budget_end=mission_deadline_ts,
        pass_kind=pass_kind, skip=True,
    ).route)
