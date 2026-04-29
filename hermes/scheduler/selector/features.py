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
    """Rough per-device reliability signal derived from scheduler state.

    We can't read the cluster-scope ``on_time_history / missed_history``
    counters from here (design §6.1 — those live in ``DeviceRecord``),
    but the last-outcome tag is a decent online proxy. Never-seen ->
    0.5 (neutral prior).
    """
    if state.last_outcome is None:
        return 0.5
    # MissionOutcome.is_on_time() returns True only for CLEAN.
    return 1.0 if state.last_outcome.is_on_time() else 0.0


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
