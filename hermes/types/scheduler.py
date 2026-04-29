"""Scheduler-local state types.

Unlike the wire types in ``fl_messages`` and ``bundles``, these never
cross a tier boundary — they live entirely inside ``FLScheduler`` on the
mule NUC. Putting them in ``hermes.types`` anyway keeps one import root
for the rest of the build.

Design refs:
* HERMES_FL_Scheduler_Design.md §6.2 FLScheduler state
* HERMES_FL_Scheduler_Design.md §4 (bucket tags)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from .ids import DeviceID
from .round_report import MissionOutcome


class Bucket(str, Enum):
    """Coarse priority tag produced by S3's bucket classifier.

    Design §4: the scheduler has no intra-bucket rank of its own — S3.5
    (``TargetSelectorRL``, Phase 5) provides that. For Phase 4 the
    placeholder orders by ``last_known_distance`` inside each bucket.
    """

    NEW = "new"                          # registered but never served
    SCHEDULED_THIS_ROUND = "scheduled"   # in slice, has active deadline
    BEACON_ACTIVE = "beacon_active"      # recent beacon heard, opportunistic


# Buckets visit-order (design §4): new first, then scheduled, then beacon
BUCKET_PRIORITY: Tuple[Bucket, ...] = (
    Bucket.NEW,
    Bucket.SCHEDULED_THIS_ROUND,
    Bucket.BEACON_ACTIVE,
)


@dataclass
class DeviceSchedulerState:
    """Scheduler's per-device view.

    Populated from:
    * initial ``MissionSlice`` (DOWN bundle) -> ``is_in_slice``, ``is_new``
    * ``RoundCloseDelta`` (fast-phase bus) -> ``last_outcome``, ``idle_time_ref_ts``
    * ``ClusterAmendment`` (slow-phase at dock) -> ``deadline_override``
    * RF beacons -> ``last_beacon_ts``, bucket may flip to ``BEACON_ACTIVE``
    """

    device_id: DeviceID
    is_in_slice: bool = False
    is_new: bool = True

    # Contact / outcome history (this mule only)
    last_outcome: Optional[MissionOutcome] = None
    last_contact_ts: float = 0.0
    last_utility: float = 0.0

    # Deadline machinery (see Design §6.2 formula)
    deadline_fulfilment_s: float = 60.0   # default window (design §9 Q1 open)
    idle_time_ref_ts: float = 0.0         # last on-time participation ts
    deadline_override_ts: Optional[float] = None  # from ClusterAmendment

    # RF / opportunistic
    last_beacon_ts: float = 0.0

    # Output of S3's bucket classifier (set by FLScheduler each pipeline pass)
    bucket: Optional[Bucket] = None

    # Last-known position for S3.5 placeholder ordering (from DeviceRecord)
    last_known_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class BeaconObservation:
    """One RF beacon event observed by this mule.

    Used as the bus payload between the L1 RF listener and the scheduler.
    Mirrors the minimum info an RF front-end can confidently report.
    """

    device_id: DeviceID
    observed_at: float
    snr: float = 0.0


@dataclass(frozen=True)
class TargetWaypoint:
    """One entry in the scheduler's output visit queue.

    The mule supervisor walks these in order, hands each to L1 as the
    current target, and lets L1 pick the RF channel. Bucket is carried
    so L1 / observability can log the priority tier the target came from.
    """

    device_id: DeviceID
    position: Tuple[float, float, float]
    bucket: Bucket
    deadline_ts: float
