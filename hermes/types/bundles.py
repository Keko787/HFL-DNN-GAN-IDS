"""Dock-link payloads.

Mirror of the design doc's §6.9 interface contract:

* ``UpBundle``   — ``ClientCluster -> HFLHostCluster``  (UP at dock)
* ``DownBundle`` — ``HFLHostCluster -> ClientCluster``  (DOWN at dock)

Phase 1 owns the cluster (server) side of the dock link; Phase 3 builds
``ClientCluster`` to consume these bundles on the mule.

FeRRy Phase 3 (the mission clock) adds simulated time as data, never as a
clock: ``UpBundle.sim_upload_ts`` and ``UpBundle.backhaul`` say when and how
the mule's simulated upload happened, and ``DownBundle.cluster_sim_ts`` echoes
the cluster's simulated time for the mule's Lamport sync at the dock. All
three default to None, which is every legacy bundle; the signatures do not
cover them (``signatures.py`` hashes a fixed field list), so legacy
signatures are unchanged.
"""

from __future__ import annotations

import math
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


def _sim_stamp(value: Optional[float], name: str) -> Optional[float]:
    """A simulated time as a finite float (None passes through)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(f"{name} must be a number or None, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class BackhaulUpload:
    """The mule's own view of one simulated backhaul upload (FeRRy Phase 3).

    On the seconds-axis backhaul (``backhaul_model="seconds"``) the mule picks
    the carrier (the fixed ``argmax g_c``, or H3's controller), reads the
    carrier's SNR when the upload starts, and charges its clock ``upload_s``
    of simulated time for ``nbytes`` at that SNR's rate. ``p_loss`` is
    ``loss_from_snr`` of that SNR, the probability the cluster's keyed draw
    uses; ``below_floor`` marks an SNR below the contact link's floor, where
    the link carries nothing: the upload is lost (``p_loss`` 1.0) and the
    charge is capped (critic B12). The upload completes at ``t_start_s +
    upload_s``, which is ``UpBundle.sim_upload_ts``.
    """

    carrier: int
    snr_db: float
    p_loss: float
    t_start_s: float
    upload_s: float
    nbytes: int = 0
    below_floor: bool = False


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

    FeRRy Phase 3, on the simulated mission clock only (None otherwise):
    ``sim_upload_ts`` is the simulated time the upload COMPLETED (critic B8:
    what the cluster may merge by, and what it echoes back as
    ``DownBundle.cluster_sim_ts``); ``backhaul`` is how the mule priced it
    (carrier, SNR, ``p_loss``), None when no seconds-axis backhaul is wired.
    A bundle retried at a later dock keeps both: they describe the upload
    that produced it.
    """

    mule_id: MuleID
    partial_aggregate: PartialAggregate
    round_close_report: MissionRoundCloseReport
    contact_history: ContactHistory
    bundle_sig: str = ""  # checksum/version (Phase 3 verifier)
    prev_mission_delivery_report: Optional[MissionDeliveryReport] = None
    sim_upload_ts: Optional[float] = None
    backhaul: Optional[BackhaulUpload] = None

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
        self.sim_upload_ts = _sim_stamp(self.sim_upload_ts, "sim_upload_ts")
        if self.backhaul is not None and not isinstance(self.backhaul, BackhaulUpload):
            raise TypeError(
                f"backhaul must be a BackhaulUpload or None, got {type(self.backhaul).__name__}"
            )


@dataclass
class DownBundle:
    """Cluster -> Mule dock payload.

    FeRRy Phase 3: ``cluster_sim_ts`` is the cluster's simulated time (the
    latest ``sim_upload_ts`` it has ingested), for the mule's Lamport sync at
    the dock (``MissionClock.advance_to``). None, the default, means "no
    sync": every legacy DOWN, and every DOWN until the cluster echoes it.
    """

    mule_id: MuleID
    mission_slice: MissionSlice
    theta_disc: Weights  # global discriminator weights
    synth_batch: List[np.ndarray]  # synth sample tensors
    cluster_amendments: ClusterAmendment
    bundle_sig: str = ""
    cluster_sim_ts: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mission_slice.mule_id != self.mule_id:
            raise ValueError("DownBundle mule_id mismatches mission_slice")
        self.cluster_sim_ts = _sim_stamp(self.cluster_sim_ts, "cluster_sim_ts")
