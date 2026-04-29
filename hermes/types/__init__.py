"""Shared types crossing tier boundaries in the HERMES design.

These dataclasses are the wire format between programs. Keep them small,
serialisable, and free of behaviour — anything fancy belongs in the program
that owns the lifecycle of the value, not in the dataclass.

Phase 1 only uses:

* ``DeviceID`` / ``MuleID`` / ``ServerID``
* ``DeviceRecord``
* ``MissionSlice``
* ``PartialAggregate``
* ``MissionRoundCloseReport`` (and its line type)
* ``ContactHistory``
* ``ClusterAmendment``
* ``UpBundle`` / ``DownBundle`` (the dock-link payloads)

Later phases extend with ``FL_READY_ADV``, ``RoundCloseDelta``,
``DeviceState``, etc. — added when the phase that needs them lands.
"""

from .ids import DeviceID, MuleID, ServerID
from .registry import DeviceRecord, MissionSlice, SpectrumSig
from .round_report import (
    ContactHistory,
    ContactRecord,
    MissionOutcome,
    MissionRoundCloseReport,
    MissionRoundCloseLine,
)
from .aggregate import PartialAggregate, ClusterAmendment, Weights
from .bundles import UpBundle, DownBundle
from .fl_state import FLState
from .fl_messages import (
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
    DeviceSchedulerState,
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
    "MissionOutcome",
    "MissionRoundCloseReport",
    "MissionRoundCloseLine",
    # aggregate
    "PartialAggregate",
    "ClusterAmendment",
    "Weights",
    # bundles
    "UpBundle",
    "DownBundle",
    # FL state + messages (Phase 2)
    "FLState",
    "FLOpenSolicit",
    "FLReadyAdv",
    "DiscPush",
    "GradientSubmission",
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
    "DeviceSchedulerState",
    "TargetWaypoint",
]
