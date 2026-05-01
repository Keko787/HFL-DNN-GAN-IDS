"""Dock-link payloads.

Mirror of the design doc's §6.9 interface contract:

* ``UpBundle``   — ``ClientCluster -> HFLHostCluster``  (UP at dock)
* ``DownBundle`` — ``HFLHostCluster -> ClientCluster``  (DOWN at dock)

Phase 1 owns the cluster (server) side of the dock link; Phase 3 builds
``ClientCluster`` to consume these bundles on the mule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .aggregate import ClusterAmendment, PartialAggregate, Weights
from .ids import MuleID
from .registry import MissionSlice
from .round_report import (
    ContactHistory,
    MissionDeliveryReport,
    MissionRoundCloseReport,
)


@dataclass
class UpBundle:
    """Mule -> Cluster dock payload.

    Sprint 1.5 added ``prev_mission_delivery_report``: the previous
    mission's Pass-2 ``MissionDeliveryReport``, carried up at the
    *next* mission's Pass-1 dock. Optional — None on cold start (no
    previous mission) or if the supervisor is still in legacy
    single-pass mode. The cluster reads it to bump
    ``DeviceRecord.delivery_priority`` on undelivered rows so they're
    pulled toward cluster anchors next slice.
    """

    mule_id: MuleID
    partial_aggregate: PartialAggregate
    round_close_report: MissionRoundCloseReport
    contact_history: ContactHistory
    bundle_sig: str = ""  # checksum/version (Phase 3 verifier)
    prev_mission_delivery_report: Optional[MissionDeliveryReport] = None

    def __post_init__(self) -> None:
        if self.partial_aggregate.mule_id != self.mule_id:
            raise ValueError("UpBundle mule_id mismatches partial_aggregate")
        if self.round_close_report.mule_id != self.mule_id:
            raise ValueError("UpBundle mule_id mismatches round_close_report")
        if (
            self.prev_mission_delivery_report is not None
            and self.prev_mission_delivery_report.mule_id != self.mule_id
        ):
            raise ValueError(
                "UpBundle mule_id mismatches prev_mission_delivery_report"
            )


@dataclass
class DownBundle:
    """Cluster -> Mule dock payload."""

    mule_id: MuleID
    mission_slice: MissionSlice
    theta_disc: Weights  # global discriminator weights
    synth_batch: List[np.ndarray]  # synth sample tensors
    cluster_amendments: ClusterAmendment
    bundle_sig: str = ""

    def __post_init__(self) -> None:
        if self.mission_slice.mule_id != self.mule_id:
            raise ValueError("DownBundle mule_id mismatches mission_slice")
