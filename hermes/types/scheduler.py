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
from typing import Dict, Optional, Tuple

from .ids import DeviceID
from .round_report import MissionOutcome


#: A new device's fulfilment window Φ₀ in seconds: the default of
#: ``DeviceSchedulerState.deadline_fulfilment_s``. A placeholder, not a tuned
#: constant (design §9 Q1, critic A7). FeRRy Phase 3 lets the scheduler restate
#: it (``FLScheduler(initial_window_s=...)``, in this same recorded unit) and
#: multiplies it by ``FLScheduler(deadline_time_scale=...)`` like every other
#: constant of the deadline law.
DEFAULT_FULFILMENT_WINDOW_S: float = 60.0


class Bucket(str, Enum):
    """Coarse priority tag produced by S3's bucket classifier.

    Design §4: the scheduler has no intra-bucket rank of its own — S3.5
    (``TargetSelectorRL``, Phase 5) provides that. For Phase 4 the
    placeholder orders by ``last_known_distance`` inside each bucket.
    """

    NEW = "new"                          # registered but never served
    SCHEDULED_THIS_ROUND = "scheduled"   # in slice, has active deadline
    BEACON_ACTIVE = "beacon_active"      # recent beacon heard, opportunistic


class MissionPass(str, Enum):
    """Which half of a two-pass HERMES mission is running.

    Design §7 principle 13: a mission is Pass 1 (collect Δθ) + dock +
    Pass 2 (deliver fresh θ). Pass 1 runs scheduler-driven contact
    selection; Pass 2 walks every contact greedily for universal delivery.
    """

    COLLECT = "collect"  # Pass 1 — pull pre-prepared Δθ from devices
    DELIVER = "deliver"  # Pass 2 — push fresh θ_disc to every device


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
    #: Timestamp of this device's most recent outcome of ANY kind, including
    #: the synthetic TIMEOUT fed for a device the mule abandoned without a
    #: contact. Not an age source: use ``last_clean_ts`` for that.
    last_contact_ts: float = 0.0
    last_utility: float = 0.0
    # Running tallies that mirror DeviceRecord.on_time_history /
    # missed_history but live mule-side. The S3.5 selector reads
    # on_time_count / (on_time_count + missed_count) as its continuous
    # reliability proxy — the binary `last_outcome` was too noisy a
    # signal for the DDQN to separate flaky from reliable devices.
    on_time_count: int = 0
    missed_count: int = 0

    # Sprint 1.5: cached copy of DeviceRecord.delivery_priority pulled
    # from the most recent DOWN bundle. S3a clustering reads this as a
    # tie-breaker so a device that went UNDELIVERED in the previous
    # mission's Pass 2 gets pulled toward a cluster anchor early in the
    # next slice's circuit. Reset to 0 on a clean delivery via
    # MissionDeliveryReport ingest at the cluster.
    delivery_priority: int = 0

    # Deadline machinery (see Design §6.2 formula)
    deadline_fulfilment_s: float = DEFAULT_FULFILMENT_WINDOW_S   # default window (design §9 Q1 open)
    idle_time_ref_ts: float = 0.0         # last on-time participation ts
    deadline_override_ts: Optional[float] = None  # from ClusterAmendment

    # Oort baseline inputs (arm B2) — the raw training loss and sample count
    # from this device's most recent contact, folded in beside `last_utility`.
    # Retrospective by construction: they describe the LAST time we visited,
    # which is the only client state a data mule can hold. `None` means "never
    # served, or this arm does not carry them".
    last_loss: Optional[float] = None
    last_num_examples: int = 0
    #: Mission round of this device's most recent outcome of ANY kind,
    #: including failed and abandoned attempts. 0 = no outcome yet. The Oort
    #: arm derives its current round from this; its ``L(i)`` is
    #: ``last_clean_round``.
    last_served_round: int = 0

    # Baseline age inputs (arms D1/D2) — set ONLY on a CLEAN outcome, so a
    # failed or abandoned attempt never counts as service. MAX-AoI ages a
    # device from ``last_clean_ts``; Oort's ``L(i)`` is ``last_clean_round``.
    # 0 = never participated successfully. Inert for H0–H3, which read neither.
    last_clean_ts: float = 0.0
    last_clean_round: int = 0

    #: Consecutive non-CLEAN outcomes since the last CLEAN, synthetic TIMEOUTs
    #: for dropped or abandoned devices included. FeRRy's priority key: with
    #: ``miss_priority`` on, S3b admits contacts with a longer streak first.
    #: Maintained always; nothing reads it otherwise.
    miss_streak: int = 0

    # FeRRy Phase 2 — inputs for the Whittle baseline (arm D3), maintained
    # always and read by nothing else. ``reach_attempts`` counts real Pass-1
    # outcomes (the synthetic TIMEOUT for a device the mule dropped or
    # abandoned is not an attempt); ``reach_answered`` those whose device
    # answered (its advert arrived), whatever the outcome — so their ratio
    # estimates reachability, not reliability. ``last_merged_round`` is the
    # mission whose merge last used this device's update (None = never), the
    # device's Age-of-Update anchor; a CLEAN whose update the age cutoff
    # excluded does not move it.
    reach_attempts: int = 0
    reach_answered: int = 0
    last_merged_round: Optional[int] = None

    # RF / opportunistic
    last_beacon_ts: float = 0.0

    # Output of S3's bucket classifier (set by FLScheduler each pipeline pass)
    bucket: Optional[Bucket] = None

    # Last-known position for S3.5 placeholder ordering (from DeviceRecord)
    last_known_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    #: FeRRy Phase 3 (design §4.7, plumbing only): the latest SNR the cluster
    #: saw for this device per contact band class (``"wide"``, ``"medium"``,
    #: ``"narrow"``), folded from ``registry_deltas[did]["spectrum_sig"]`` at
    #: dock. None = never reported, which is every legacy run: legacy DOWNs
    #: never carry the key. Nothing in Phase 3 decides on it.
    spectrum_snr_db: Optional[Dict[str, float]] = None


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
    """One entry in the scheduler's output visit queue (per-device, pre-Sprint-1.5).

    Retained for backward compatibility with the Phase-4 deterministic
    pipeline + Sprint-1A `MuleSupervisor` tests. After Sprint 1.5 the
    scheduler emits ``ContactWaypoint`` instead, which covers N≥1
    devices per stop. ``TargetWaypoint`` is the degenerate N=1 case.
    """

    device_id: DeviceID
    position: Tuple[float, float, float]
    bucket: Bucket
    deadline_ts: float


@dataclass(frozen=True)
class ContactWaypoint:
    """One contact-event entry in the scheduler's output queue.

    Sprint 1.5 design §7 principle 15: the mule's circuit is decomposed
    into contact events, where each stop covers all devices within
    ``rf_range_m`` of the position. The selector picks among
    ``ContactWaypoint``s, not individual devices. The N=1 case (isolated
    device) is the degenerate-but-valid form of the same payload.

    ``devices`` is the list of in-range slice members the mule will
    serve in parallel at this stop. ``bucket`` is inherited from the
    *worst* bucket among the members (so a NEW-and-SCHEDULED mix is
    treated as NEW, drained first). ``deadline_ts`` is the *tightest*
    deadline among the members — if any member's deadline is overdue,
    the contact inherits that pressure.

    FeRRy Phase 3 (design §4.5) adds three annotations the mule fills with
    ``dataclasses.replace`` right after planning: the contact ``band`` class,
    its planar ``range_m`` R_planar(b), and each member's predicted SNR
    ``pred_snr_db`` (in ``devices`` order). They describe the stop; they do
    not identify it, so all three are left out of equality and hashing
    (``compare=False``): a waypoint annotated or not is the same dictionary
    key and set member, which the walks and ``MuleSupervisor._budget_pass_2``
    rely on.
    """

    position: Tuple[float, float, float]
    devices: Tuple[DeviceID, ...]
    bucket: Bucket
    deadline_ts: float
    band: Optional[str] = field(default=None, compare=False)
    range_m: Optional[float] = field(default=None, compare=False)
    pred_snr_db: Optional[Tuple[float, ...]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("ContactWaypoint must cover ≥1 device")
