"""DDIL build D2: a leave-by time for each Pass-1 stop (the unified plan's step 2).

``DeveloperDocs/FeRRy_DDIL_Unified_Plan.md``, step 2: each stop gets the plan's
predicted end plus its share of the mission's slack; after it no session,
retry or re-poll starts (``ContactPlan.leave_by_ts``, ``ContactPlan.leaves_at``),
and the members it cuts are committed as timeouts. It is what holds C2's
overrun bound once step 1's retries lengthen the stops.

The rule, computed at the arrival from the planner's own model (the
feasibility model's single-stop predicate and its walk, ``s3b_feasibility``):

* ``d`` is this stop's predicted dwell (``FerryPhysics.dwell_s``, on the class
  the stop is served on), ``home`` the predicted landing if the mule serves
  this stop and then the stops left in their order (return leg and Pass-1
  upload included), and ``slack = budget_end - home``;
* the stop's share of the slack is ``d / D``, with ``D`` this stop's predicted
  dwell plus the stops left's (each on the model's own class: their classes
  are decided later), so slack, or a deficit, is spread over the stops by the
  dwell each is predicted to need;
* ``leave_by = arrival + max(0, d + share * slack)``.

So a mission on time leaves each stop no earlier than its predicted end, a
mission behind it cuts each stop by its share of the deficit, and the rule
spends no slack on a stop priced at no dwell. Without a budget there is no
leave-by time (None), as S3b's gate is off without one.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional, Sequence

from hermes.types.scheduler import MissionPass


def stop_leave_by_ts(
    model: Any,
    wp: Any,
    remainder: Sequence[Any],
    *,
    arrival_ts: float,
    budget_end: Optional[float],
) -> Optional[float]:
    """The leave-by time of Pass-1 stop ``wp``, reached at ``arrival_ts``; None without a budget.

    ``model`` is a ferry ``FeasibilityModel`` whose physics is bound to the
    class the stop is served on and to the scheduler's device states;
    ``remainder`` the stops left once this one was taken, in their order.
    """
    if budget_end is None:
        return None
    from hermes.scheduler.stages.s3b_feasibility import RULE_NONE, FlightState

    ferry = model.ferry
    if ferry is None or ferry.member_dwell_s is None:
        return None
    here = FlightState(tuple(wp.position), float(arrival_ts), 0.0, math.inf)
    verdict = model.admit(here, wp, rule=RULE_NONE, budget_end=None,
                          pass_kind=MissionPass.COLLECT)
    d_here = float(verdict.finish) - float(arrival_ts)
    if remainder:
        walk = model.fold(list(remainder), verdict.next_state, rule=RULE_NONE, budget_end=None,
                          pass_kind=MissionPass.COLLECT, skip=False)
        home = float(walk.home)
        d_rest = sum(float(ferry.dwell_s(r, MissionPass.COLLECT)) for r in remainder)
    else:
        home = float(verdict.home)
        d_rest = 0.0
    slack = float(budget_end) - home
    total = d_here + d_rest
    share = d_here / total if total > 0.0 else 0.0
    return float(arrival_ts) + max(0.0, d_here + share * slack)


def model_on_class(model: Any, fx: Any, band: Optional[str]) -> Any:
    """``model`` with its ferry physics on ``band`` (``FerryRuntime.physics``), kept bound
    to the same device states; ``model`` itself when ``band`` is None or its own."""
    ferry = getattr(model, "ferry", None)
    if ferry is None or band is None:
        return model
    physics = fx.physics(band)
    states = getattr(ferry, "device_states", None)
    if states is not None:
        physics = physics.bind(states)
    return dataclasses.replace(model, ferry=physics)


__all__ = ["model_on_class", "stop_leave_by_ts"]
