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

**FeRRy Phase 4: member subsets (opt-in).** The published methods behind
D1-D3 select devices, not stops (critic B5), so under the user's decision 4
(b) of 2026-09-30 the walk can re-issue a contact that fails whole with the
members that still fit, in the arm's own order: each member ranked by the
arm's ``key`` on its one-member contact, then by device id
(:func:`_walk_subsets`). Only the plan before takeoff passes the carrier
(``member_subsets``, when the run sets ``member_admission="subset"``); the
in-flight check and re-plan keep or drop contacts whole, and None, the
default, is the recorded walk. :func:`left_out` names what a route leaves out
of its contacts, members included, for the scheduler's report of a
baseline's drops (decision 6).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from hermes.types import ContactWaypoint, DeviceID, MissionPass

from hermes.scheduler.stages.s3b_feasibility import (
    RULE_BUDGET,
    FeasibilityModel,
    FlightState,
    MemberSubsets,
    fold_subsets,
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
    member_subsets: Optional[MemberSubsets] = None,
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

    ``member_subsets`` (FeRRy Phase 4; None, the default, is the recorded
    walk) is passed only by the plan before takeoff under
    ``member_admission="subset"``: a contact that fails whole is then
    re-issued with the members that still fit, ranked by ``key`` on their
    one-member contacts (:func:`_walk_subsets`). It is for Pass 1 only (Pass
    2 delivers to whole contacts), so any other ``pass_kind`` raises
    ValueError; without a budget it is inert, as the gate is.
    """
    if member_subsets is not None and MissionPass(pass_kind) is not MissionPass.COLLECT:
        raise ValueError(
            f"greedy_budget_walk: member subsets are for Pass 1 only (Pass 2 delivers "
            f"to whole contacts), got pass_kind={pass_kind!r}"
        )
    ordered = sorted(contacts, key=key)
    if mission_deadline_ts is None:
        return ordered

    m = model or FeasibilityModel()
    start = state if state is not None else FlightState(mule_pose, now)
    if member_subsets is not None:
        return _walk_subsets(ordered, start, m, key=key, budget_end=mission_deadline_ts,
                             subsets=member_subsets)
    # Skip, not stop: a contact that does not fit is left out and the walk
    # tries the next one from the same pose and clock.
    return list(m.fold(
        ordered, start, rule=RULE_BUDGET, budget_end=mission_deadline_ts,
        pass_kind=pass_kind, skip=True,
    ).route)


def _walk_subsets(
    ordered: Sequence[ContactWaypoint],
    start: FlightState,
    model: FeasibilityModel,
    *,
    key: RankKey,
    budget_end: float,
    subsets: MemberSubsets,
) -> List[ContactWaypoint]:
    """The Pass-1 walk with member subsets (``greedy_budget_walk`` given the carrier).

    The contacts go in ``key`` order, as the recorded walk takes them; one
    that fails whole under the budget is re-issued with the members that
    still fit (``s3b_feasibility.fold_subsets``), each member ranked by
    ``key`` on its one-member contact, then by device id.

    Why this order (critic B5; unit_U3b.md section 2.3): the published methods
    select devices, and a contact is only the unit of travel. Within one
    contact the position and the distance from the mule are common to every
    member, so ``key`` on a one-member contact reduces to the arm's own
    per-device score: D1 the member's age (never served first,
    ``max_aoi.py``), D2 its utility plus staleness bonus (unexplored first,
    ``oort.py``), D3 its Whittle index (``whittle.py``); the ports'
    ``(position, devices)`` tie-break reduces to the device id. There is no
    dwell tie-break: D1-D3 rank without travel or speed terms by design
    (``oort.py`` deviation 1, ``whittle.py`` deviation 3), and a reduction
    must not add one.
    """
    def member_order(wp: ContactWaypoint) -> List[DeviceID]:
        return sorted(wp.devices, key=lambda d: (key(subsets.reduce(wp, (d,))), d))

    return list(fold_subsets(
        ordered, start, model=model, rule=RULE_BUDGET, budget_end=budget_end,
        member_order=member_order, subsets=subsets,
    ).route)


def left_out(
    contacts: Sequence[ContactWaypoint],
    route: Sequence[ContactWaypoint],
    *,
    member_subsets: Optional[MemberSubsets] = None,
) -> List[ContactWaypoint]:
    """What ``route`` leaves out of ``contacts``, as contacts, in ``contacts`` order.

    For the scheduler's report of a baseline's pre-flight drops (the user's
    decision 6; unit U5): with member subsets a walk's route can serve part of
    a contact, and the rest must still be named. Members are matched by device
    id. A contact the route serves no member of is returned itself (the
    object, as the identity rule of ``FLScheduler.replan_remainder`` returns
    it); one it serves in part gives the rest as a reduced contact
    (``member_subsets.reduce``, ValueError without the carrier); one it serves
    whole gives nothing. S3a places each device in one contact, so where the
    route is a subset of ``contacts`` by identity (every ``whole`` walk) this
    is exactly the identity rule. Two contacts sharing a device, or a route
    serving a device no contact holds, raise ValueError: the members could
    not be matched to one contact each.
    """
    owner: Dict[DeviceID, int] = {}
    for i, wp in enumerate(contacts):
        for did in wp.devices:
            if owner.setdefault(did, i) != i:
                raise ValueError(f"left_out: device {did!r} is a member of two contacts")
    served = {did for wp in route for did in wp.devices}
    stray = served.difference(owner)
    if stray:
        raise ValueError(
            f"left_out: the route serves {sorted(str(d) for d in stray)}, which no contact holds"
        )
    out: List[ContactWaypoint] = []
    for wp in contacts:
        rest = [did for did in wp.devices if did not in served]
        if len(rest) == len(wp.devices):
            out.append(wp)
        elif rest:
            if member_subsets is None:
                raise ValueError(
                    f"left_out: the route serves part of the contact {wp.devices!r}; naming "
                    f"the rest needs the member-subset carrier"
                )
            out.append(member_subsets.reduce(wp, rest))
    return out
