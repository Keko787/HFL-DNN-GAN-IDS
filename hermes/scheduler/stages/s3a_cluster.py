"""Stage 3a — Position clustering by RF range.

Sprint 1.5 design §7 principle 15: the mule's circuit is decomposed into
contact events, where each stop covers all devices within ``rf_range_m``
of the position. S3a runs *between* the deadline math (S3) and the
bucket classifier — it groups eligible devices into ``ContactWaypoint``s,
which are what the bucket classifier and the selector then operate on.

Algorithm (greedy, principle 12-friendly — no hidden cleverness):

    1. Pick the un-clustered device with the highest priority signal
       (`delivery_priority` first, then earliest deadline).
    2. Form a tentative cluster: that device plus every other un-clustered
       device within ``rf_range_m`` of it.
    3. Compute the centroid of the cluster's positions.
    4. If the centroid is within ``rf_range_m`` of every cluster member,
       use the centroid as the stop position. Otherwise fall back to the
       anchor's own position (geometrically guaranteed to cover everyone
       in the cluster).
    5. Mark all members clustered and repeat from step 1.

Bucket inheritance: a ContactWaypoint inherits the *worst* (highest-
priority-to-drain) bucket of its members — so a NEW + SCHEDULED mix
becomes NEW. Deadline inheritance: the *tightest* deadline among members.

The N=1 case (an isolated device or a singleton bucket member) is the
degenerate-but-valid form — the same code path produces a one-device
ContactWaypoint with the device's own position. No special branch.

Design refs:
* HERMES_FL_Scheduler_Design.md §7 principle 15 (per-contact aggregation)
* HERMES_FL_Scheduler_Design.md §6.1 (`ContactWaypoint`, `rf_range_m`)
* HERMES_FL_Scheduler_Implementation_Plan.md §3.6.2 task 2
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from hermes.types import (
    BUCKET_PRIORITY,
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
)


Position = Tuple[float, float, float]


def _distance(a: Position, b: Position) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _centroid(positions: Sequence[Position]) -> Position:
    n = len(positions)
    if n == 0:
        return (0.0, 0.0, 0.0)
    sx = sum(p[0] for p in positions) / n
    sy = sum(p[1] for p in positions) / n
    sz = sum(p[2] for p in positions) / n
    return (sx, sy, sz)


def _bucket_priority(bucket: Bucket) -> int:
    """Index in BUCKET_PRIORITY — lower = higher priority (drained earlier)."""
    return BUCKET_PRIORITY.index(bucket)


def _worst_bucket(buckets: Sequence[Bucket]) -> Bucket:
    """Highest-priority (smallest index) bucket among the inputs."""
    return min(buckets, key=_bucket_priority)


def _device_priority_key(
    device_id: DeviceID,
    state: DeviceSchedulerState,
    deadlines: Dict[DeviceID, float],
) -> Tuple[int, int, float]:
    """Sort key for picking the next anchor.

    Tuple: ``(-delivery_priority, bucket_priority, deadline_ts)``.
    More negative `delivery_priority` (i.e. higher carry-over from a
    Pass-2 miss) wins first; then earlier bucket; then earliest deadline.
    """
    # delivery_priority lives on the cluster-side DeviceRecord, not the
    # mule-side DeviceSchedulerState — but the scheduler can carry a
    # cached copy on the state row when ingesting the slice. We read it
    # via getattr to keep this function tolerant of older states that
    # don't have the field (fall back to 0).
    dp = int(getattr(state, "delivery_priority", 0))
    bp = _bucket_priority(state.bucket) if state.bucket is not None else len(BUCKET_PRIORITY)
    dl = float(deadlines.get(device_id, float("inf")))
    return (-dp, bp, dl)


def cluster_by_rf_range(
    eligible_device_ids: Sequence[DeviceID],
    device_states: Dict[DeviceID, DeviceSchedulerState],
    deadlines: Dict[DeviceID, float],
    rf_range_m: float,
) -> List[ContactWaypoint]:
    """Group eligible devices into :class:`ContactWaypoint`s.

    Args:
        eligible_device_ids: output of S1 — devices the scheduler has
            admitted for this round.
        device_states: per-device scheduler state (positions, bucket tag).
            Caller is expected to have set ``state.bucket`` already
            (S3 bucket-classify runs before this stage in the pipeline,
            so the bucket field is populated).
        deadlines: per-device deadline timestamps from S3.
        rf_range_m: mule RF range in metres — the radius for clustering.

    Returns:
        List of ``ContactWaypoint``s in *no particular order* — the bucket
        classifier and the selector handle priority. The list covers
        every input device exactly once.
    """
    if rf_range_m <= 0.0:
        raise ValueError(f"rf_range_m must be positive, got {rf_range_m}")
    if not eligible_device_ids:
        return []

    remaining: List[DeviceID] = list(eligible_device_ids)
    contacts: List[ContactWaypoint] = []

    while remaining:
        # 1. Pick the highest-priority anchor.
        remaining.sort(
            key=lambda did: _device_priority_key(did, device_states[did], deadlines)
        )
        anchor_id = remaining[0]
        anchor_state = device_states[anchor_id]
        anchor_pos = anchor_state.last_known_position

        # 2. Form a tentative cluster around the anchor.
        cluster_ids: List[DeviceID] = [anchor_id]
        for did in remaining[1:]:
            st = device_states[did]
            if _distance(anchor_pos, st.last_known_position) <= rf_range_m:
                cluster_ids.append(did)

        # 3. Compute centroid; 4. fall back if centroid moves anyone out.
        positions = [device_states[d].last_known_position for d in cluster_ids]
        centroid = _centroid(positions)
        if all(_distance(centroid, p) <= rf_range_m for p in positions):
            stop = centroid
        else:
            stop = anchor_pos  # geometric guarantee: anchor covers cluster

        # Inherit bucket + deadline from members.
        member_buckets = [
            device_states[d].bucket
            for d in cluster_ids
            if device_states[d].bucket is not None
        ]
        if member_buckets:
            bucket = _worst_bucket(member_buckets)
        else:
            # Anchor must have a bucket if it came through S3; if not, the
            # caller has bypassed bucket-classify — fail loudly.
            raise ValueError(
                f"S3a: device(s) {cluster_ids} reached clustering with no bucket"
            )

        deadline_ts = min(deadlines.get(d, float("inf")) for d in cluster_ids)

        contacts.append(
            ContactWaypoint(
                position=stop,
                devices=tuple(cluster_ids),
                bucket=bucket,
                deadline_ts=deadline_ts,
            )
        )

        # 5. Mark all members clustered.
        clustered = set(cluster_ids)
        remaining = [d for d in remaining if d not in clustered]

    return contacts


def order_pass_2_greedy(
    contacts: Sequence[ContactWaypoint],
    mule_pose: Position,
) -> List[ContactWaypoint]:
    """Pass-2 ordering: nearest-first greedy from the mule's current pose.

    Sprint 1.5 design §7 principle 13: Pass 2 walks every contact; the
    selector is bypassed; the goal is universal delivery so the cheapest
    universal cover is what we want. Greedy nearest-first from the
    post-Pass-1 mule pose minimises Pass-2 path length given an arbitrary
    starting point (a TSP solver could improve marginally; not worth it
    at slice sizes ≤ 10).

    Returns a re-ordered copy of the input list. The caller (the mule
    supervisor) advances ``mule_pose`` between contacts, so this is a
    one-shot order computation, not an interactive policy.
    """
    if not contacts:
        return []

    remaining = list(contacts)
    ordered: List[ContactWaypoint] = []
    pose = mule_pose

    while remaining:
        idx = min(
            range(len(remaining)),
            key=lambda i: _distance(pose, remaining[i].position),
        )
        chosen = remaining.pop(idx)
        ordered.append(chosen)
        pose = chosen.position

    return ordered
