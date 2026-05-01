"""Model-aggregate payloads.

Two distinct aggregate types live in HERMES:

* ``PartialAggregate`` — mission-scope, produced by ``HFLHostMission``'s
  partial FedAvg, uploaded by ``ClientCluster`` at dock.
* ``ClusterAmendment`` — slow-phase corrections produced by
  ``HFLHostCluster`` after the cross-mule FedAvg, dispatched DOWN.

Weight tensors are stored as ``Weights = List[np.ndarray]`` (see below)
— the same Flower convention used everywhere else in the system. Sprint
2's pickle-based wire format (``hermes.transport.wire``) serializes
numpy arrays natively, so we no longer keep a separate ``bytes``
form alongside the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .ids import DeviceID, MuleID


# A weight set is the same Flower convention: a list of numpy arrays, one
# per layer / parameter group, in a stable order matching the model.
Weights = List[np.ndarray]


@dataclass
class PartialAggregate:
    """Mission-scope partial FedAvg output uploaded UP at dock.

    ``num_examples`` is the total count of training examples behind this
    partial — the weight used in cluster-scope FedAvg.
    """

    mule_id: MuleID
    mission_round: int
    weights: Weights
    num_examples: int
    contributing_devices: Tuple[DeviceID, ...] = ()

    def is_empty(self) -> bool:
        return self.num_examples == 0 or not self.weights


@dataclass
class ClusterAmendment:
    """Slow-phase corrections dispatched DOWN after cluster reconciliation.

    The amendment carries deadline adjustments + (optionally) registry
    deltas that the mission-scope scheduler must fold in *before* the next
    in-field round.
    """

    cluster_round: int
    deadline_overrides: dict = field(default_factory=dict)  # DeviceID -> new ts
    registry_deltas: dict = field(default_factory=dict)  # DeviceID -> patch dict
    notes: str = ""


@dataclass
class ClusterPartialUpload:
    """Tier-2 → Tier-3 outbound payload (Sprint 2 cloud link).

    Carries the cluster's post-FedAvg discriminator weights upward so
    Tier-3 can run cross-cluster θ_gen refinement. ``cluster_id`` is
    a free-form string the Tier-3 endpoint uses to namespace clusters
    (one HERMES deployment can have multiple clusters).
    """

    cluster_id: str
    cluster_round: int
    theta_disc: Weights
    n_devices_contributing: int = 0


@dataclass
class GeneratorRefinement:
    """Tier-3 → Tier-2 inbound payload (Sprint 2 cloud link).

    Carries refined θ_gen weights from Tier-3's cross-cluster aggregation.
    Per design principle 9, θ_gen never leaves Tier 2 — but the *aggregated*
    cross-cluster θ_gen does flow back from Tier 3 to each Tier-2 cluster
    so they can resync. The cluster pulls these via outbound polling
    because AERPAW blocks inbound traffic from Tier 3.
    """

    refinement_round: int
    theta_gen: Weights
    notes: str = ""
