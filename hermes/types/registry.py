"""Device registry types — the cluster-scope source of truth.

A ``DeviceRecord`` is the per-device row in ``HFLHostCluster``'s registry.
A ``MissionSlice`` is a disjoint subset of those device IDs handed to one
mule for one mission.

Design refs:
* HERMES_FL_Scheduler_Design.md §6.1 (shared types)
* HERMES_FL_Scheduler_Design.md §6.7 HFLHostCluster.DeviceRegistry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .ids import DeviceID, MuleID


@dataclass(frozen=True)
class SpectrumSig:
    """RF fingerprint priors for a device.

    ``last_good_snr_per_band`` keys are RF-band indices (matching whatever the
    radio runner uses — currently 0/1/2 for 3.32/3.34/3.90 GHz per slide 26).
    """

    bands: Tuple[int, ...]
    last_good_snr_per_band: Tuple[float, ...]


@dataclass
class DeviceRecord:
    """One row of the cluster-scope ``DeviceRegistry``."""

    device_id: DeviceID
    last_known_position: Tuple[float, float, float]  # (lat, lon, alt)
    spectrum_sig: SpectrumSig
    assigned_mule: Optional[MuleID] = None
    on_time_history: int = 0  # increments each on-time submission
    missed_history: int = 0  # increments each missed deadline
    is_new: bool = True  # cleared after first successful distribution
    # Sprint 1.5: incremented by HFLHostCluster when a MissionDeliveryReport
    # row says this device was *undelivered* in Pass 2; reset to 0 on a
    # clean delivery. S3a clustering reads this as a tie-breaker so
    # high-priority (recently-undelivered) devices are pulled toward
    # cluster anchors, increasing the chance they get reached early in
    # the next mission. Design principle 13 — Pass 2 must reach every
    # member; this field is the carry-over for when it doesn't.
    delivery_priority: int = 0


@dataclass(frozen=True)
class MissionSlice:
    """Disjoint per-mule subset of the registry, refreshed every dock.

    The slice is the *single source of truth* the mule consults during a
    mission. Disjointness across mules is enforced at dispatch time
    (Design §7 principle 8).
    """

    mule_id: MuleID
    device_ids: Tuple[DeviceID, ...]
    issued_round: int  # cluster round counter at the time of dispatch
    issued_at: float  # epoch seconds — wall-clock for now (Design §9 Q1)

    def __post_init__(self) -> None:
        # cheap invariant: no duplicate IDs inside one slice
        if len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError(
                f"MissionSlice for {self.mule_id!r} has duplicate device_ids"
            )


# convenience for tests / future serialisation
@dataclass
class _RegistryStats:
    total_devices: int
    sliced_devices: int
    unassigned_devices: int

    @property
    def coverage(self) -> float:
        return (self.sliced_devices / self.total_devices) if self.total_devices else 0.0
