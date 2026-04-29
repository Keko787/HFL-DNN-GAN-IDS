"""Stage 3 — Deadline formula + bucket classifier.

Design §6.8::

    Deadline(j) = Time + Deadline_Fulfilment(j) − Idle_Time(j)
        Deadline_Fulfilment = +on_time_history − missed_history
        Idle_Time low  ⇒ shorter deadline

Design §6.2 wording note: ``Deadline_Fulfilment`` here is the *per-device
fulfilment window* (seconds), not the history delta on its own. We keep
the window baseline on ``DeviceSchedulerState.deadline_fulfilment_s`` and
let the fast/slow-phase deltas nudge it.

Two-phase adaptation (design §7 principle 4):

* **Fast phase** — in-mission: :func:`fold_round_close_delta` ingests a
  ``RoundCloseDelta`` from ``HFLHostMission``. On-time shrinks the
  window; missed/partial widens it.
* **Slow phase** — at dock: :func:`fold_cluster_amendment` applies
  ``ClusterAmendment.deadline_overrides`` (explicit per-device ts) and
  merges any ``registry_deltas`` that touch deadline fields.

The bucket classifier is the *only* hard-rank layer exposed to S3.5
(§4). Inside each bucket S3.5 (placeholder or RL actor) picks an order.
"""

from __future__ import annotations

from typing import Dict, Optional

from hermes.types import (
    Bucket,
    BUCKET_PRIORITY,
    ClusterAmendment,
    DeviceID,
    DeviceSchedulerState,
    MissionOutcome,
    RoundCloseDelta,
)


# Fast-phase deltas — seconds nudged per outcome. Small so the window
# drifts, not whipsaws.
FAST_PHASE_ON_TIME_SHRINK_S: float = 5.0
FAST_PHASE_MISSED_WIDEN_S: float = 10.0

# Floor on the fulfilment window — never let the formula drive it to zero.
MIN_DEADLINE_FULFILMENT_S: float = 5.0


# --------------------------------------------------------------------------- #
# Deadline math
# --------------------------------------------------------------------------- #

def compute_idle_time(state: DeviceSchedulerState, now: float) -> float:
    """Seconds since the last on-time participation, floored at 0.

    A never-seen device with ``idle_time_ref_ts == 0`` gets idle=0, which
    keeps its first-round deadline at exactly ``Time + Deadline_Fulfilment``
    instead of being artificially short. New-device bucket handles the
    prioritisation instead.
    """
    if state.idle_time_ref_ts <= 0.0:
        return 0.0
    return max(0.0, now - state.idle_time_ref_ts)


def compute_deadline(state: DeviceSchedulerState, now: float) -> float:
    """Design §6.8 formula.

    ``deadline_override_ts`` short-circuits the formula — the cluster
    amendment is authoritative when present (slow-phase wins over
    fast-phase drift for this round).
    """
    if state.deadline_override_ts is not None:
        return state.deadline_override_ts
    fulfilment = max(MIN_DEADLINE_FULFILMENT_S, state.deadline_fulfilment_s)
    return now + fulfilment - compute_idle_time(state, now)


# --------------------------------------------------------------------------- #
# Bucket classifier
# --------------------------------------------------------------------------- #

def classify_bucket(
    state: DeviceSchedulerState,
    now: float,
    beacon_window_s: float = 30.0,
) -> Bucket:
    """Assign the design §4 bucket tag.

    Priority (see :data:`BUCKET_PRIORITY`):

    1. ``NEW``  — registered but never served (``is_new=True``)
    2. ``SCHEDULED_THIS_ROUND`` — in the current slice with a deadline
    3. ``BEACON_ACTIVE`` — recent beacon but not in slice (opportunistic)

    Devices that fit none of the above trigger a ``ValueError`` — the
    caller (S1) should not have admitted them.
    """
    if state.is_new:
        return Bucket.NEW
    if state.is_in_slice:
        return Bucket.SCHEDULED_THIS_ROUND
    if state.last_beacon_ts > 0.0 and (now - state.last_beacon_ts) <= beacon_window_s:
        return Bucket.BEACON_ACTIVE
    raise ValueError(
        f"classify_bucket: device {state.device_id!r} has no bucket "
        f"(not new, not in slice, no fresh beacon)"
    )


# --------------------------------------------------------------------------- #
# Fast-phase — consume RoundCloseDelta from HFLHostMission
# --------------------------------------------------------------------------- #

def fold_round_close_delta(
    state: DeviceSchedulerState,
    delta: RoundCloseDelta,
) -> DeviceSchedulerState:
    """Apply one in-mission delta to the scheduler's view of a device.

    Mutates-then-returns for ergonomic caller code; ``DeviceSchedulerState``
    is a plain dataclass so this is cheap.

    On-time outcome:
        - clear ``is_new`` (distribution landed, so no longer brand new)
        - shrink the fulfilment window toward the floor
        - refresh ``idle_time_ref_ts`` + ``last_contact_ts``

    Partial/timeout outcome:
        - widen the fulfilment window
        - record contact_ts but do **not** reset idle_time_ref_ts (a
          failed attempt doesn't reset the "when were you last reliable"
          clock)
    """
    if delta.device_id != state.device_id:
        raise ValueError(
            f"fold_round_close_delta: delta for {delta.device_id!r} applied "
            f"to state for {state.device_id!r}"
        )

    state.last_contact_ts = delta.contact_ts
    state.last_outcome = delta.outcome
    state.last_utility = delta.utility

    if delta.outcome is MissionOutcome.CLEAN:
        state.is_new = False
        state.idle_time_ref_ts = delta.contact_ts
        state.deadline_fulfilment_s = max(
            MIN_DEADLINE_FULFILMENT_S,
            state.deadline_fulfilment_s - FAST_PHASE_ON_TIME_SHRINK_S,
        )
    else:
        state.deadline_fulfilment_s = (
            state.deadline_fulfilment_s + FAST_PHASE_MISSED_WIDEN_S
        )

    return state


# --------------------------------------------------------------------------- #
# Slow-phase — consume ClusterAmendment at dock
# --------------------------------------------------------------------------- #

def fold_cluster_amendment(
    device_states: Dict[DeviceID, DeviceSchedulerState],
    amendment: ClusterAmendment,
) -> None:
    """Apply ``deadline_overrides`` + relevant ``registry_deltas`` to the map.

    Only devices the scheduler already tracks are touched — slice
    membership (S1) is responsible for admitting new devices, not the
    amendment fold.

    ``registry_deltas`` fields supported here:
        * ``last_known_position`` — tuple[float, float, float]
        * ``deadline_fulfilment_s`` — float override from cluster
    Anything else is ignored; the full registry row lives in
    ``HFLHostCluster``, not in scheduler state.
    """
    for did, new_ts in amendment.deadline_overrides.items():
        st = device_states.get(did)
        if st is None:
            continue
        st.deadline_override_ts = new_ts

    for did, patch in amendment.registry_deltas.items():
        st = device_states.get(did)
        if st is None or not isinstance(patch, dict):
            continue
        if "last_known_position" in patch:
            pos = patch["last_known_position"]
            if isinstance(pos, tuple) and len(pos) == 3:
                st.last_known_position = pos  # type: ignore[assignment]
        if "deadline_fulfilment_s" in patch:
            val = patch["deadline_fulfilment_s"]
            if isinstance(val, (int, float)):
                st.deadline_fulfilment_s = max(
                    MIN_DEADLINE_FULFILMENT_S, float(val)
                )
