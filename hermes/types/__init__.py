"""Shared types crossing tier boundaries in the HERMES design.

These dataclasses are the wire format between programs. Keep them small,
serialisable, and free of behaviour — anything fancy belongs in the program
that owns the lifecycle of the value, not in the dataclass.

Imported types break down into a few groups:

* IDs and core records: ``DeviceID``, ``MuleID``, ``ServerID``,
  ``DeviceRecord``, ``MissionSlice``, ``SpectrumSig``.
* RF-link messages (Sprint 1+): ``FLReadyAdv``, ``Solicit``,
  ``DiscPush``, ``ContactWaypoint``, ``DeliveryAck``.
* Round reports + outcomes: ``MissionRoundCloseReport`` /
  ``MissionRoundCloseLine``, ``MissionDeliveryReport`` /
  ``MissionDeliveryLine``, ``ContactHistory`` / ``ContactRecord``,
  ``MissionOutcome``, ``DeliveryOutcome``.
* Aggregates + dock payloads: ``PartialAggregate``,
  ``ClusterAmendment``, ``UpBundle``, ``DownBundle``.

The full list is in ``__all__`` at the bottom of this module.
"""

from .ids import DeviceID, MuleID, ServerID
from .registry import DeviceRecord, MissionSlice, SpectrumSig
from .round_report import (
    ContactHistory,
    ContactRecord,
    DeliveryOutcome,
    MissionDeliveryLine,
    MissionDeliveryReport,
    MissionOutcome,
    MissionRoundCloseReport,
    MissionRoundCloseLine,
)
from .aggregate import (
    ClusterAmendment,
    ClusterPartialUpload,
    GeneratorRefinement,
    PartialAggregate,
    Weights,
)
from .bundles import UpBundle, DownBundle
from .fl_state import FLState
from .fl_messages import (
    DeliveryAck,
    FLOpenSolicit,
    FLReadyAdv,
    DiscPush,
    GradientSubmission,
    RoundCloseDelta,
    weights_signature,
    weights_byte_count,
)
from .signatures import (
    sign_up_bundle,
    sign_down_bundle,
    verify_up_bundle,
    verify_down_bundle,
)
from .scheduler import (
    BUCKET_PRIORITY,
    BeaconObservation,
    Bucket,
    ContactWaypoint,
    DeviceSchedulerState,
    MissionPass,
    TargetWaypoint,
)

__all__ = [
    # ids
    "DeviceID",
    "MuleID",
    "ServerID",
    # registry
    "DeviceRecord",
    "MissionSlice",
    "SpectrumSig",
    # round report
    "ContactHistory",
    "ContactRecord",
    "DeliveryOutcome",
    "MissionDeliveryLine",
    "MissionDeliveryReport",
    "MissionOutcome",
    "MissionRoundCloseReport",
    "MissionRoundCloseLine",
    # aggregate
    "PartialAggregate",
    "ClusterAmendment",
    "ClusterPartialUpload",
    "GeneratorRefinement",
    "Weights",
    # bundles
    "UpBundle",
    "DownBundle",
    # FL state + messages (Phase 2 + Sprint 1.5)
    "FLState",
    "FLOpenSolicit",
    "FLReadyAdv",
    "DiscPush",
    "GradientSubmission",
    "DeliveryAck",
    "RoundCloseDelta",
    "weights_signature",
    "weights_byte_count",
    # bundle signatures (Phase 3)
    "sign_up_bundle",
    "sign_down_bundle",
    "verify_up_bundle",
    "verify_down_bundle",
    # scheduler-local state (Phase 4)
    "BUCKET_PRIORITY",
    "BeaconObservation",
    "Bucket",
    "ContactWaypoint",
    "DeviceSchedulerState",
    "MissionPass",
    "TargetWaypoint",
]
