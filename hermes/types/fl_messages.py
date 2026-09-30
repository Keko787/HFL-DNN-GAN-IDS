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
    # FeRRy Phase 3 (design section 4.3): the ferry path numbers each targeted
    # solicit (1, 2, ... per mule server) and devices echo the number in
    # ``FLReadyAdv.in_reply_to``, so the mule accepts only adverts that answer
    # the contact it is gathering for (critic B1). 0 means "not numbered": the
    # legacy broadcast, whose adverts carry no reference to their solicit.
    solicit_id: int = 0


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
    # --- Oort baseline inputs (arm B2) -------------------------------------- #
    # The RAW local training loss and local sample count, carried alongside the
    # derived `utility` rather than folded into it. Oort's statistical utility
    # is |B_i| * sqrt(mean Loss^2), which cannot be recovered from `utility`:
    # that is w1*perf + w2*diversity, and `perf` already collapses accuracy, AUC
    # and loss into one score. Reusing it would be a different algorithm wearing
    # Oort's name.
    #
    # Optional, defaulting to None/0 — every existing arm ignores them, so the
    # advertisement stays wire-compatible and H0-H3 behaviour is unchanged.
    local_loss: Optional[float] = None
    num_examples: int = 0
    # FeRRy Phase 3: the ``FLOpenSolicit.solicit_id`` this advert answers (0 for
    # an unnumbered solicit, and for a beacon). The ferry path discards any
    # advert whose number is not its own contact's, instead of stashing it.
    in_reply_to: int = 0

    def is_eligible(self) -> bool:
        return self.state.can_open_session()


# --------------------------------------------------------------------------- #
# In-session payloads
# --------------------------------------------------------------------------- #

#: What a device's update carries. ``weights`` is the full model after local
#: training — what every recorded arm shipped, and what ``agg:plain`` averages.
#: ``delta`` is that model minus the basis the device trained from, which the
#: age-aware merge rules need (FeRRy Phase 1).
UPDATE_FORM_WEIGHTS = "weights"
UPDATE_FORM_DELTA = "delta"
UPDATE_FORMS = (UPDATE_FORM_WEIGHTS, UPDATE_FORM_DELTA)


@dataclass
class DiscPush:
    """Mule -> device — push discriminator weights + synth batch.

    ``weights_sig`` is an opaque hash over the weight bytes so the device
    can reject a corrupt push without rebuilding the model.

    ``pass_kind`` mirrors :class:`FLOpenSolicit` — DELIVER means "store
    these weights, start fresh local training, send a DeliveryAck instead
    of a GradientSubmission." Default COLLECT for backward compat.

    ``basis_version`` is the version of ``theta_disc``: the cluster round that
    produced it. The device keeps it with its training basis and echoes it
    with the update it later trains from that basis, which is how every update
    gets an age. ``update_form`` tells a Pass-1 device which form to answer
    in. Both default to what every recorded run did (no version, full weights).
    """

    mule_id: MuleID
    mission_round: int
    theta_disc: Weights
    synth_batch: List[np.ndarray]
    weights_sig: str = ""
    pass_kind: MissionPass = MissionPass.COLLECT
    basis_version: Optional[int] = None
    update_form: str = UPDATE_FORM_WEIGHTS
    # Set on Pass-1 pushes when the mule's Pass 2 is budgeted and may not come
    # back: the device then trains ahead on the basis this push gives it, in
    # the background, instead of waiting for a delivery (principle 14).
    train_ahead: bool = False
    # FeRRy Phase 3 (contact reliability from the channel, design section 4.8):
    # the mule's keyed availability draw failed this device's uplink for this
    # contact. The device adopts the basis (and trains ahead if asked) but sends
    # no update, exactly as a failed uplink under ``contact_reliability`` does;
    # the mule records the session as timed out without waiting for it. The
    # draw is made on the mule so the device's ground-truth availability never
    # reaches the mule's decision code (critic B16), only its outcome does.
    uplink_drop: bool = False

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
    # Oort baseline input (arm B2). Carried HERE rather than only on the next
    # advertisement so it arrives in the SAME session as the update it describes
    # — otherwise the mule's ranking signal lags a full mission round behind the
    # training it summarises, which would handicap the baseline unfairly.
    local_loss: Optional[float] = None
    # The version of the basis this update was trained from (the
    # ``DiscPush.basis_version`` the device stored with it), and whether
    # ``delta_theta`` holds full weights or a delta against that basis. None
    # means the sender did not know the version.
    basis_version: Optional[int] = None
    update_form: str = UPDATE_FORM_WEIGHTS
    # FeRRy Phase 3 (critic B2): the solicit number of the contact this update
    # answers, echoed from ``FLOpenSolicit.solicit_id`` (0 when unnumbered). A
    # device's gradient queue on the mule outlives the contact that filled it,
    # so without it a gradient that came after its own contact gave up would be
    # taken as the reply to the device's next contact. ``basis_version`` cannot
    # serve: a prepared update carries the basis it was trained on, which is
    # older than the push it answers.
    in_reply_to: int = 0

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
    # FeRRy Phase 3 (critic B2): the solicit number of the delivery this ack
    # answers (0 when unnumbered). The ferry path counts an ack as DELIVERED
    # only if its round, signature and number are those of its own push.
    in_reply_to: int = 0


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
    # Oort baseline inputs (arm B2), forwarded verbatim from the device's
    # advertisement. Optional so every existing emitter stays valid.
    local_loss: Optional[float] = None
    num_examples: int = 0
    # FeRRy: whether the device answered this contact (its FLReadyAdv
    # arrived), whatever the outcome. The multiplicative deadline law relaxes
    # a reachable device's window less than an unreachable one's, and the
    # reach history D3 needs counts it. ``synthetic`` marks the TIMEOUT fed
    # for a device the mule dropped or abandoned without an attempt.
    answered: bool = False
    synthetic: bool = False


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
