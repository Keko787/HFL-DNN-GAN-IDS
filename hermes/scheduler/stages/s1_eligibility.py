"""Stage 1 — Eligibility filter.

Design §6.8::

    eligible(i) = has_active_deadline(i) ∨ beacon_heard_in_range(i)

This is deliberately coarse: any device the mule has *reason to visit*
passes S1. Downstream stages narrow it to *ready and worth it*.

Decisions made here:

* ``has_active_deadline`` — the device is in the current ``MissionSlice``
  (planned) OR carries a deadline override from a ``ClusterAmendment``.
  Devices not in the slice and not overridden are not planned for this
  mission even if we have a stale scheduler row.
* ``beacon_heard`` — at least one beacon within ``beacon_window_s``
  before ``now``. Beacons older than the window do not count (they
  would've expired on the device side anyway — see design §2.1
  "proximity only").
"""

from __future__ import annotations

from typing import Dict, List

from hermes.types import DeviceID, DeviceSchedulerState


def has_active_deadline(state: DeviceSchedulerState) -> bool:
    """True iff the scheduler has a reason to enforce a deadline for this device."""
    return state.is_in_slice or state.deadline_override_ts is not None


def beacon_heard(
    state: DeviceSchedulerState,
    now: float,
    beacon_window_s: float,
) -> bool:
    """True iff we heard a beacon from this device within the window."""
    if state.last_beacon_ts <= 0.0:
        return False
    if beacon_window_s <= 0.0:
        return False
    return (now - state.last_beacon_ts) <= beacon_window_s


def is_eligible(
    state: DeviceSchedulerState,
    now: float,
    beacon_window_s: float = 30.0,
) -> bool:
    """Pure-function form of the design §6.8 rule."""
    return has_active_deadline(state) or beacon_heard(state, now, beacon_window_s)


def filter_eligible(
    device_states: Dict[DeviceID, DeviceSchedulerState],
    now: float,
    beacon_window_s: float = 30.0,
) -> List[DeviceID]:
    """Return the subset of ``device_states`` that pass S1, input-order preserved."""
    return [
        did
        for did, st in device_states.items()
        if is_eligible(st, now=now, beacon_window_s=beacon_window_s)
    ]
