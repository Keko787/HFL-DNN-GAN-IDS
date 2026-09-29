"""Contact-ranking policies: the Experiment-3 ablation arms and the SOTA baselines.

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

The Experiment-4/5 SOTA arms are *whole schedulers*: each exposes
``admit_and_order(contacts, device_states, env, *, mission_deadline_ts=None,
feasibility_model=None)``, which replaces S3, S3b and S3.5 and returns the
Pass-1 route, and declares ``in_flight_check`` (Freeze Amendment 8). The mule
builds them from ``MuleConfig.contact_policy``
(:func:`hermes.processes.mule._build_target_selector`):

* **D1** — :class:`MaxAoIPolicy` (``"max_aoi"``).
* **D2** — :class:`OortPolicy` (``"oort"``).
* **D3** — :class:`WhittlePolicy` (``"whittle"``), Cui's Whittle index.
* **D4** — :class:`FedExCarpPolicy` (``"fedex"``), FedEx-Async's visit-all
  tour; :func:`carp_assign` is its device-to-mule assignment, which the
  Exp 4 driver computes once per trial when there are several mules.
* **D5** — :class:`FedCSDegradedPolicy` (``"fedcs"``), FedCS's greedy
  selection on last-known state.
"""

from __future__ import annotations

from .arrival_order import ArrivalOrderPolicy
from .edf_feasibility import EdfFeasibilityPolicy
from .fedcs_degraded import FedCSDegradedPolicy
from .fedex_carp import FedExCarpPolicy, carp_assign
from .max_aoi import MaxAoIPolicy, contact_age
from .oort import OortPolicy, OortUnusableError, statistical_utility
from .whittle import WhittlePolicy

__all__ = [
    "ArrivalOrderPolicy",
    "EdfFeasibilityPolicy",
    "FedCSDegradedPolicy",
    "FedExCarpPolicy",
    "carp_assign",
    "MaxAoIPolicy",
    "contact_age",
    "OortPolicy",
    "OortUnusableError",
    "statistical_utility",
    "WhittlePolicy",
]
