"""Stage 3b — deadline feasibility gate.

**Why this stage exists.** S3 computes ``Deadline(j) = t_ref(j) + Φ(j)`` and
S3a propagates the tightest member deadline onto each ``ContactWaypoint`` — but
until this stage was added, nothing in the runtime path ever compared that
deadline to a clock. The deadline was a *sort key only*: a device whose deadline
had already passed, or could not possibly be reached in time, was still queued
and still visited. A source trace of Experiment 4 found exactly this, so the
scheduler's "deadline-aware" description was not true of the code.

This stage closes that gap. It is deliberately a **hard gate** (it removes
candidates) rather than an ordering, because that is the architectural contract
the rest of the scheduler rests on: deterministic rules decide what is *legal*,
and the learned selector may only reorder what survives. Placing feasibility
inside the selector would have inverted that — and would also have been skipped
entirely for single-candidate buckets, which take a short-circuit around the
selector.

**Opt-in by construction.** With no mission budget configured
(``mission_deadline_ts=None``) the gate is a no-op and the queue is unchanged.
That keeps every previously-recorded result reproducible: enforcement turns on
only when a caller supplies a budget.

**Cost model.** Deliberately simple and stated, rather than elaborate and
hidden: a contact costs ``transit + session``, where transit is straight-line
distance at ``cruise_speed_m_s``. There is no propulsion-energy, upload-rate or
return-leg term — Experiment 4 does not model those (Experiment 3 does, via
``EdfFeasibilityPolicy``). A contact is dropped when either

* it cannot be reached before **its own deadline** (`deadline_ts`), or
* serving it would exceed the **remaining mission budget**.

The walk is greedy in EDF order, advancing a simulated pose and clock, so later
estimates account for earlier stops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from hermes.types.scheduler import ContactWaypoint

MulePose = Tuple[float, float, float]


@dataclass(frozen=True)
class FeasibilityModel:
    """Transit/service cost parameters for the gate.

    Defaults are intentionally permissive; a caller that wants the gate to bind
    should set them from the deployment being modelled.
    """

    cruise_speed_m_s: float = 5.0
    session_time_s: float = 1.0

    def cost(self, frm: MulePose, to: MulePose) -> Tuple[float, float]:
        """Return ``(transit_s, total_s)`` for flying ``frm`` → ``to``."""
        transit = _euclid(frm, to) / max(self.cruise_speed_m_s, 1e-6)
        return transit, transit + self.session_time_s


def _euclid(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


@dataclass(frozen=True)
class FeasibilityResult:
    kept: List[ContactWaypoint]
    dropped_overdue: List[ContactWaypoint]
    dropped_budget: List[ContactWaypoint]

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_overdue) + len(self.dropped_budget)


def filter_feasible(
    contacts: Sequence[ContactWaypoint],
    *,
    now: float,
    mule_pose: MulePose = (0.0, 0.0, 0.0),
    mission_deadline_ts: Optional[float] = None,
    model: Optional[FeasibilityModel] = None,
) -> FeasibilityResult:
    """Drop contacts that cannot be served in time. EDF-ordered greedy walk.

    ``mission_deadline_ts`` is an **absolute** timestamp. ``None`` disables the
    gate entirely — every contact is kept, and the result is indistinguishable
    from not calling this stage at all.

    Returns the survivors plus the two rejection reasons, so a caller can log
    *why* a device was not served instead of it vanishing silently.
    """
    if mission_deadline_ts is None:
        return FeasibilityResult(list(contacts), [], [])

    mdl = model or FeasibilityModel()
    # EDF: tightest deadline first, then a stable tie-break so the gate is
    # deterministic across re-runs.
    ordered = sorted(contacts, key=lambda c: (c.deadline_ts, c.position, c.devices))

    kept: List[ContactWaypoint] = []
    overdue: List[ContactWaypoint] = []
    over_budget: List[ContactWaypoint] = []

    pose: MulePose = tuple(mule_pose)  # type: ignore[assignment]
    clock = float(now)

    for wp in ordered:
        transit, total = mdl.cost(pose, wp.position)
        arrival = clock + transit
        # (a) the contact's own deadline must still be reachable ...
        if arrival > wp.deadline_ts:
            overdue.append(wp)
            continue
        # (b) ... and serving it must fit inside the mission budget.
        if clock + total > mission_deadline_ts:
            over_budget.append(wp)
            continue
        kept.append(wp)
        clock += total
        pose = wp.position

    return FeasibilityResult(kept, overdue, over_budget)
