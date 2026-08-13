"""Experiment-3 contact-ranking policies (A2 / A3 ablation arms).

Three baseline arms compete with the A4 :class:`TargetSelectorRL` in
the paper's scheduling-ablation experiment:

* **A1** — centralized FL, no mule (driven from
  :mod:`experiments.exp3.arm_a1`, not a contact-ranking policy).
* **A2** — :class:`ArrivalOrderPolicy`, services contacts in
  registration order. Captures the "no scheduling" baseline.
* **A3** — :class:`EdfFeasibilityPolicy`, earliest-deadline-first with
  a feasibility skip. Captures the "naive heuristic" baseline.
* **A4** — :class:`TargetSelectorRL.rank_contacts` (already shipping).

Both A2 and A3 expose the same call shape as
``TargetSelectorRL.rank_contacts``::

    policy.rank_contacts(
        candidates,
        device_states,
        env,
        *,
        pass_kind=MissionPass.COLLECT,
        admitted=None,
    ) -> List[ContactWaypoint]

so the supervisor / Experiment-3 driver can swap arms by passing one of
``{ArrivalOrderPolicy(), EdfFeasibilityPolicy(...), target_selector_rl}``
through the same constructor slot.
"""

from __future__ import annotations

from .arrival_order import ArrivalOrderPolicy
from .edf_feasibility import EdfFeasibilityPolicy
from .max_aoi import MaxAoIPolicy, contact_age
from .oort import OortPolicy, OortUnusableError, statistical_utility

__all__ = [
    "ArrivalOrderPolicy",
    "EdfFeasibilityPolicy",
    "MaxAoIPolicy",
    "contact_age",
    "OortPolicy",
    "OortUnusableError",
    "statistical_utility",
]
