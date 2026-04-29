"""Stage 3.5 — Intra-bucket selector (placeholder).

Design §2.1 and §7 principle 12: the deterministic path ranks inside a
bucket by a simple, auditable rule until ``TargetSelectorRL`` ships in
Phase 5. We use last-known distance to the mule's current pose — closer
first. Ties break on ``device_id`` for determinism (test stability).

Scope guard: this function **only** orders candidates that the
deterministic stages already admitted. It cannot promote a gated-out
device.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from hermes.types import DeviceID, DeviceSchedulerState


MulePose = Tuple[float, float, float]


def _distance(a: MulePose, b: MulePose) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def select_order(
    candidates: Sequence[DeviceID],
    device_states: Dict[DeviceID, DeviceSchedulerState],
    mule_pose: MulePose = (0.0, 0.0, 0.0),
) -> List[DeviceID]:
    """Return ``candidates`` sorted by (distance-to-mule, device_id).

    Missing scheduler state is treated as "infinitely far" so the ordering
    is defined even when the caller passes a partial map (useful in
    tests). In production S3 always has state for every admitted device.
    """
    def _key(did: DeviceID) -> Tuple[float, str]:
        st = device_states.get(did)
        if st is None:
            return (math.inf, str(did))
        return (_distance(mule_pose, st.last_known_position), str(did))

    return sorted(candidates, key=_key)


def select_target(
    candidates: Sequence[DeviceID],
    device_states: Dict[DeviceID, DeviceSchedulerState],
    mule_pose: MulePose = (0.0, 0.0, 0.0),
) -> Optional[DeviceID]:
    """Single argmax — head of :func:`select_order` or ``None`` if empty."""
    ordered = select_order(candidates, device_states, mule_pose=mule_pose)
    return ordered[0] if ordered else None
