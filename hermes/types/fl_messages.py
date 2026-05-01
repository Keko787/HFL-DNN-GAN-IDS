"""Mission-scope FL wire messages — mule <-> device over the RF link.

Four messages round-trip through one mission-round FL session:

1. ``FLOpenSolicit``    mule -> device : "are you ready?"
2. ``FLReadyAdv``       device -> mule : payload announcing readiness + utility
3. ``DiscPush``         mule -> device : push θ_disc + synth batch for local step
4. ``GradientSubmission`` device -> mule : Δθ_disc + meta, ends the session

After the session closes, the mule emits a ``RoundCloseDelta`` on its
intra-NUC bus so the scheduler can run its fast-phase deadline update.

Design refs:
* HERMES_FL_Scheduler_Design.md §5.3 and §6.3
* HERMES_FL_Scheduler_Implementation_Plan.md §3 Phase 2
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .aggregate import Weights
from .fl_state import FLState
from .ids import DeviceID, MuleID
from .round_report import MissionOutcome
from .scheduler import MissionPass


# --------------------------------------------------------------------------- #
# Handshake
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FLOpenSolicit:
    """Mule -> device — 'is anyone FL-ready on this channel?' ping.

    ``pass_kind`` distinguishes Pass-1 (COLLECT — pull prepared Δθ) from
    Pass-2 (DELIVER — push fresh θ', no Δθ requested). Devices branch on
    this field in ``serve_once`` vs ``serve_delivery``. Default stays
    COLLECT so pre-Sprint-1.5 callers still work unchanged.
    """

    mule_id: MuleID
    mission_round: int
    issued_at: float
    pass_kind: MissionPass = MissionPass.COLLECT


@dataclass(frozen=True)
class FLReadyAdv:
    """Device -> mule — readiness advertisement.

    Carries the device's own state tag, its *locally computed* utility for
    this round, and a small recent-history summary. The mule copies the
    utility into the scheduler bus after the session closes.
    """

    device_id: DeviceID
    state: FLState
    performance_score: float        # S2B sub-term
    diversity_adjusted: float       # S2B sub-term
    utility: float                  # w1*perf + w2*diversity (device-computed)
    last_round_outcome: Optional[MissionOutcome] = None
    issued_at: float = 0.0

    def is_eligible(self) -> bool:
        return self.state.can_open_session()


# --------------------------------------------------------------------------- #
# In-session payloads
# --------------------------------------------------------------------------- #

@dataclass
class DiscPush:
    """Mule -> device — push discriminator weights + synth batch.

    ``weights_sig`` is an opaque hash over the weight bytes so the device
    can reject a corrupt push without rebuilding the model.

    ``pass_kind`` mirrors :class:`FLOpenSolicit` — DELIVER means "store
    these weights, start fresh local training, send a DeliveryAck instead
    of a GradientSubmission." Default COLLECT for backward compat.
    """

    mule_id: MuleID
    mission_round: int
    theta_disc: Weights
    synth_batch: List[np.ndarray]
    weights_sig: str = ""
    pass_kind: MissionPass = MissionPass.COLLECT

    def __post_init__(self) -> None:
        if not self.weights_sig:
            self.weights_sig = weights_signature(self.theta_disc)


@dataclass
class GradientSubmission:
    """Device -> mule — gradient delta + verification metadata.

    The receipt verifier on the mule checks:
    * ``byte_count`` matches the sum of ``w.nbytes`` for every layer
    * ``checksum`` matches ``weights_signature(delta_theta)``
    * ``mission_round`` matches the currently-open round
    * timestamp is within the mule's TTL
    """

    device_id: DeviceID
    mule_id: MuleID
    mission_round: int
    delta_theta: Weights
    num_examples: int
    submitted_at: float
    byte_count: int = 0
    checksum: str = ""

    def __post_init__(self) -> None:
        if self.byte_count == 0:
            self.byte_count = sum(int(w.nbytes) for w in self.delta_theta)
        if not self.checksum:
            self.checksum = weights_signature(self.delta_theta)


# --------------------------------------------------------------------------- #
# Pass-2 delivery acknowledgment (device -> mule)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DeliveryAck:
    """Device -> mule — Pass-2 receipt acknowledgment.

    Sent in response to a Pass-2 ``DiscPush`` (``pass_kind == DELIVER``)
    instead of a ``GradientSubmission``. Confirms the device has stored
    θ' and started fresh offline training. ``weights_sig`` echoes the
    push's signature so the mule can match the ack to the push.
    """

    device_id: DeviceID
    mule_id: MuleID
    mission_round: int
    weights_sig: str
    received_at: float


# --------------------------------------------------------------------------- #
# Intra-NUC fast-phase delta (mule bus)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RoundCloseDelta:
    """Emitted by ``HFLHostMission`` to the intra-NUC scheduler bus.

    One delta per device whose session closed. FLScheduler consumes these
    for its fast-phase Deadline update (design §6.2). The *slow-phase*
    counterpart is ``ClusterAmendment``, which only arrives at dock.
    """

    device_id: DeviceID
    mule_id: MuleID
    mission_round: int
    outcome: MissionOutcome
    utility: float
    contact_ts: float


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def weights_signature(weights: Weights) -> str:
    """Stable hash over a weight set.

    Uses SHA-256 over each array's raw bytes plus its shape/dtype header.
    Empty list -> empty-string sentinel so the receiver can distinguish
    "no weights attached" from "zero-valued weights".
    """
    if not weights:
        return ""
    h = hashlib.sha256()
    for w in weights:
        h.update(str(w.shape).encode("utf-8"))
        h.update(str(w.dtype).encode("utf-8"))
        h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()


def weights_byte_count(weights: Weights) -> int:
    return sum(int(w.nbytes) for w in weights)
