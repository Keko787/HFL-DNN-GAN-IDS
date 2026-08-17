"""MAX-AoI — the Age-of-Information greedy baseline.

**What it is.** "Always go to whoever has been waiting longest." A standard named
comparator in the AoI / UAV-data-collection literature, where evaluations
routinely report against *random, round-robin, periodic update and MAX-AoI*; the
greedy form selects the highest-age target and uses proximity to resolve the
path. It is the closest published rival to our bucket+deadline ordering, and it
needs **only state a data mule already has** — when each device was last served —
which is exactly why it survived the Phase-2 implementability screen when
FedCS did not.

**Why it is a fair baseline here.** A baseline must replace our *policy*, not our
*physics*:

* **kept** — S1 eligibility (slice membership is structural), S3a clustering
  (contacts are a physical fact of ``rf_range_m``), and S3b feasibility (both
  arms must face the same budget, or the comparison means nothing);
* **replaced** — S3's bucket tiers and S3.5's ordering. That is our policy, and
  it is what is under test.

In Experiment 4 the tier walk never discriminates — every round contains exactly
one non-empty bucket (``NEW`` in round 1, ``SCHEDULED_THIS_ROUND`` after, with
``BEACON_ACTIVE`` unpopulated) — so a policy in the ``target_selector`` slot
orders the entire round. That is what makes this a genuine full-ordering baseline
rather than a re-ranking inside our own tiers, which would have flattered us.

**Age, precisely.** A contact serves several clustered devices, so its urgency is
that of its **stalest member** — max, not mean. Minimising peak AoI is the
objective the greedy rule is built for, and averaging would let one long-neglected
device hide behind well-served neighbours. A device never served has
``last_contact_ts == 0.0`` and is treated as **infinitely stale**, so unexplored
devices sort first — the correct AoI reading, and it also matches the
"explore the unvisited" behaviour of the utility-based selectors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from hermes.types import (
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionPass,
)

from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.selector.scope_guard import (
    SelectorScopeViolation,
    assert_candidates_admitted,
)

from .budget_walk import greedy_budget_walk

#: Age assigned to a device that has never been served. Any finite age loses to
#: this, so unvisited devices always sort first.
NEVER_SERVED_AGE = float("inf")


def contact_age(
    wp: ContactWaypoint,
    device_states: Dict[DeviceID, DeviceSchedulerState],
    now: float,
) -> float:
    """Age of a contact = age of its **stalest** member device.

    Max rather than mean: the greedy rule targets peak AoI, and a mean would let
    one long-neglected device hide behind well-served neighbours in the same
    cluster.
    """
    ages: List[float] = []
    for did in wp.devices:
        st = device_states.get(did)
        if st is None or st.last_contact_ts <= 0.0:
            return NEVER_SERVED_AGE          # never served ⇒ maximally stale
        ages.append(max(0.0, now - st.last_contact_ts))
    return max(ages) if ages else NEVER_SERVED_AGE


class MaxAoIPolicy:
    """Order contacts by age descending; nearest-first breaks ties.

    Exposes the same ``rank_contacts`` surface as ``TargetSelectorRL`` and the
    Experiment-3 policies, so the driver swaps it through the same constructor
    slot.
    """

    name = "MAXAOI"

    def rank_contacts(
        self,
        candidates: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        pass_kind: MissionPass = MissionPass.COLLECT,
        admitted: Optional[Sequence[DeviceID]] = None,
    ) -> List[ContactWaypoint]:
        if pass_kind is not MissionPass.COLLECT:
            raise SelectorScopeViolation(
                f"MaxAoIPolicy.rank_contacts called with "
                f"pass_kind={pass_kind.value!r}; MAX-AoI is a Pass-1-only policy."
            )
        if not candidates:
            return []

        members: List[DeviceID] = []
        for wp in candidates:
            members.extend(wp.devices)
        assert_candidates_admitted(
            members, admitted if admitted is not None else members,
        )

        return sorted(candidates, key=self._rank_key(device_states, env.mule_pose,
                                                     env.now))

    # ------------------------------------------------------------------ #
    # Whole-scheduler mode (arm D1)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rank_key(device_states, mule_pose, now):
        """Age descending, nearest-first as the tie-break, id for determinism."""
        def _key(wp: ContactWaypoint):
            dist = sum(
                (a - b) ** 2 for a, b in zip(mule_pose, wp.position)
            ) ** 0.5
            # Negated age -> descending. Distance ascending is the literature's
            # "nearest predecessor" rule. Device id last so the order does not
            # depend on input order.
            return (-contact_age(wp, device_states, now),
                    dist,
                    ",".join(sorted(str(d) for d in wp.devices)))
        return _key

    def admit_and_order(
        self,
        contacts: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        mission_deadline_ts: Optional[float] = None,
        feasibility_model=None,
    ) -> List[ContactWaypoint]:
        """MAX-AoI as a **complete scheduler** — it decides *who*, not just order.

        Presence of this method is what makes the arm a whole-scheduler baseline:
        the scheduler delegates S3/S3b/S3.5 to it entirely, so this policy owns
        the admission decision our S3b gate would otherwise make. Fly to the
        stalest devices until the budget runs out.
        """
        if not contacts:
            return []
        return greedy_budget_walk(
            contacts,
            key=self._rank_key(device_states, env.mule_pose, env.now),
            mule_pose=env.mule_pose,
            now=env.now,
            mission_deadline_ts=mission_deadline_ts,
            model=feasibility_model,
        )
