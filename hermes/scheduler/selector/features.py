"""Per-candidate feature extractor.

Design §6.4::

    features(j) = {last_known_pos, SpectrumSig, distance, mule_energy,
                   rf_prior_snr, on_time_rate}

Plus the ``bucket_tag`` as one-hot so the selector can learn different
policies per bucket without having to infer it.

Feature layout (dtype = float32):

     idx  name
    ──────────────────────────────────
      0   distance (L2 mule<->device, scaled by /_DISTANCE_SCALE)
      1   pos_x  (device, scaled by /_POS_SCALE)
      2   pos_y  (device, scaled by /_POS_SCALE)
      3   pos_z  (device, scaled by /_POS_SCALE)
      4   on_time_rate (0.5 default for never-seen)
      5   beacon_fresh  (0/1)
      6   bucket_one_hot[0]   NEW
      7   bucket_one_hot[1]   SCHEDULED_THIS_ROUND
      8   bucket_one_hot[2]   BEACON_ACTIVE
      9   mule_energy (0..1 normalised)
     10   rf_prior_snr (dB, scaled by /30)

``FEATURE_DIM = 11``. The scaling is deliberately lightweight — the
selector network is linear-over-a-tanh, so feature magnitudes only need
to sit inside roughly ±3 for stable learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from hermes.types import (
    BUCKET_PRIORITY,
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
)


FEATURE_DIM: int = 11
_BUCKET_INDEX = {b: i for i, b in enumerate(BUCKET_PRIORITY)}

# Keep inputs to the tanh-activated DDQN inside roughly ±3.
# World radius in the offline sim is 100 m; diagonal ≈ 283 m → /100 keeps
# distance inside ~[0, 2.83]. Position coords are ±world_radius → ±1.0.
_DISTANCE_SCALE: float = 100.0
_POS_SCALE: float = 100.0

MulePose = Tuple[float, float, float]


@dataclass(frozen=True)
class SelectorEnv:
    """Env snapshot handed to the selector per inference.

    ``rf_prior_snr`` comes from L1's read-only env API. ``mule_energy``
    is 0..1 normalised; the supervisor converts its battery gauge.
    """

    mule_pose: MulePose = (0.0, 0.0, 0.0)
    mule_energy: float = 1.0
    rf_prior_snr_db: float = 20.0
    beacon_window_s: float = 30.0
    now: float = 0.0


def _distance(a: MulePose, b: MulePose) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _on_time_rate(state: DeviceSchedulerState) -> float:
    """Per-device reliability proxy in [0, 1].

    Prefers the running tally on ``DeviceSchedulerState`` (which mirrors
    ``DeviceRecord.on_time_history / missed_history`` for this mule's
    view) — that's a continuous signal the DDQN can actually rank.
    Falls back to the binary ``last_outcome`` tag, then to a neutral
    0.5 prior for never-seen devices.
    """
    total = state.on_time_count + state.missed_count
    if total > 0:
        return state.on_time_count / total
    if state.last_outcome is not None:
        return 1.0 if state.last_outcome.is_on_time() else 0.0
    return 0.5


def _beacon_fresh(state: DeviceSchedulerState, now: float, window_s: float) -> float:
    if state.last_beacon_ts <= 0.0 or window_s <= 0.0:
        return 0.0
    return 1.0 if (now - state.last_beacon_ts) <= window_s else 0.0


def _bucket_one_hot(bucket: Bucket) -> List[float]:
    out = [0.0] * len(BUCKET_PRIORITY)
    out[_BUCKET_INDEX[bucket]] = 1.0
    return out


def extract_features(
    state: DeviceSchedulerState,
    bucket: Bucket,
    env: SelectorEnv,
) -> np.ndarray:
    """Return the 11-float feature vector for one device."""
    dist = _distance(env.mule_pose, state.last_known_position)
    pos = state.last_known_position
    on_time = _on_time_rate(state)
    bcn_fresh = _beacon_fresh(state, env.now, env.beacon_window_s)
    bucket_oh = _bucket_one_hot(bucket)
    return np.asarray(
        [
            dist / _DISTANCE_SCALE,
            float(pos[0]) / _POS_SCALE,
            float(pos[1]) / _POS_SCALE,
            float(pos[2]) / _POS_SCALE,
            on_time,
            bcn_fresh,
            *bucket_oh,
            float(env.mule_energy),
            float(env.rf_prior_snr_db) / 30.0,
        ],
        dtype=np.float32,
    )


def extract_features_batch(
    candidates: Sequence[DeviceID],
    device_states: Dict[DeviceID, DeviceSchedulerState],
    bucket: Bucket,
    env: SelectorEnv,
) -> np.ndarray:
    """Stack a ``(K, FEATURE_DIM)`` matrix, one row per candidate.

    Unknown devices (no state row) are impossible under the scope guard,
    but we still hard-fail here rather than silently zero-fill.
    """
    rows: List[np.ndarray] = []
    for did in candidates:
        st = device_states.get(did)
        if st is None:
            raise KeyError(f"extract_features_batch: no state for {did!r}")
        rows.append(extract_features(st, bucket, env))
    if not rows:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)


# --------------------------------------------------------------------------- #
# Sprint 1.5 — per-contact feature extractor.
#
# Same FEATURE_DIM (11) so the DDQN actor architecture is reusable. The
# semantic meaning of slot 5 changes from `beacon_fresh` (per-device) to
# `member_count_normalized` (per-contact) because:
#   * The contact's bucket already encodes beacon-active-ness via the
#     "inherit worst bucket" rule from S3a, so beacon_fresh would be
#     redundant.
#   * Member count IS distinctive for contacts (a 4-device contact has
#     different value than a 1-device one) and is the cleanest replacement.
# This means a model trained on per-device features can't be re-used for
# per-contact decisions without retraining — Chunk G handles that.
# --------------------------------------------------------------------------- #

# Cap for normalising member count — most clusters at rf_range_m=60 hold
# 1–4 devices on a typical slice; clamp to 5 so the feature stays in [0, 1].
_CONTACT_MEMBER_CAP: float = 5.0


def extract_features_for_contact(
    waypoint: ContactWaypoint,
    device_states: Dict[DeviceID, DeviceSchedulerState],
    env: SelectorEnv,
) -> np.ndarray:
    """Aggregate the contact's member features into a fixed-shape vector.

    Layout (slot indices match :func:`extract_features` for slots 0–3 and
    6–10; slot 5 changes meaning per the comment above):

         idx  name
        ──────────────────────────────────
          0   distance from mule pose to contact position (scaled)
          1   contact pos_x / _POS_SCALE
          2   contact pos_y / _POS_SCALE
          3   contact pos_z / _POS_SCALE
          4   mean(on_time_rate) across waypoint.devices
          5   member_count / _CONTACT_MEMBER_CAP, clamped to [0, 1]
          6   bucket_one_hot[0]   NEW
          7   bucket_one_hot[1]   SCHEDULED_THIS_ROUND
          8   bucket_one_hot[2]   BEACON_ACTIVE
          9   mule_energy
         10   rf_prior_snr_db / 30
    """
    if not waypoint.devices:
        raise ValueError("extract_features_for_contact: empty waypoint.devices")

    pos = waypoint.position
    dist = _distance(env.mule_pose, pos)

    # Aggregate per-device terms.
    on_time_rates: List[float] = []
    for did in waypoint.devices:
        st = device_states.get(did)
        if st is None:
            raise KeyError(
                f"extract_features_for_contact: no state for {did!r} "
                f"(member of contact at {pos})"
            )
        on_time_rates.append(_on_time_rate(st))
    mean_on_time = sum(on_time_rates) / len(on_time_rates)

    member_count_norm = min(1.0, len(waypoint.devices) / _CONTACT_MEMBER_CAP)
    bucket_oh = _bucket_one_hot(waypoint.bucket)

    return np.asarray(
        [
            dist / _DISTANCE_SCALE,
            float(pos[0]) / _POS_SCALE,
            float(pos[1]) / _POS_SCALE,
            float(pos[2]) / _POS_SCALE,
            float(mean_on_time),
            float(member_count_norm),
            *bucket_oh,
            float(env.mule_energy),
            float(env.rf_prior_snr_db) / 30.0,
        ],
        dtype=np.float32,
    )


def extract_features_contact_batch(
    waypoints: Sequence[ContactWaypoint],
    device_states: Dict[DeviceID, DeviceSchedulerState],
    env: SelectorEnv,
) -> np.ndarray:
    """``(K, FEATURE_DIM)`` matrix, one row per contact waypoint.

    Empty input returns a ``(0, FEATURE_DIM)`` array so callers can
    stack-or-skip without a special branch.
    """
    rows: List[np.ndarray] = []
    for wp in waypoints:
        rows.append(extract_features_for_contact(wp, device_states, env))
    if not rows:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)
