"""Model-aggregate payloads.

Two distinct aggregate types live in HERMES:

* ``PartialAggregate`` — mission-scope, produced by ``HFLHostMission``'s
  partial FedAvg, uploaded by ``ClientCluster`` at dock.
* ``ClusterAmendment`` — slow-phase corrections produced by
  ``HFLHostCluster`` after the cross-mule FedAvg, dispatched DOWN.

Weight tensors are kept as opaque ``bytes`` here (Phase 1 doesn't need to
unpack them — cross-mule FedAvg uses the parallel ``weights`` numpy list
held alongside the bundle). The bytes form is what survives the dock
transport without dragging numpy across the wire.
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
