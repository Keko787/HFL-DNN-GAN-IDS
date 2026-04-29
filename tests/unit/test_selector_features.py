"""Phase 5 — selector feature extractor tests."""

from __future__ import annotations

import numpy as np
import pytest

from hermes.scheduler.selector import (
    FEATURE_DIM,
    SelectorEnv,
    extract_features,
    extract_features_batch,
)
from hermes.types import Bucket, DeviceID, DeviceSchedulerState, MissionOutcome


def _env(**kw) -> SelectorEnv:
    defaults = dict(
        mule_pose=(0.0, 0.0, 0.0),
        mule_energy=1.0,
        rf_prior_snr_db=30.0,
        beacon_window_s=30.0,
        now=1000.0,
    )
    defaults.update(kw)
    return SelectorEnv(**defaults)


def test_feature_vector_shape_and_dtype():
    st = DeviceSchedulerState(device_id=DeviceID("d"))
    v = extract_features(st, Bucket.NEW, _env())
    assert v.shape == (FEATURE_DIM,)
    assert v.dtype == np.float32


def test_distance_computation():
    st = DeviceSchedulerState(
        device_id=DeviceID("d"), last_known_position=(3.0, 4.0, 0.0)
    )
    v = extract_features(st, Bucket.NEW, _env(mule_pose=(0.0, 0.0, 0.0)))
    # 3-4-5 triangle, normalised by _DISTANCE_SCALE=100.
    assert float(v[0]) == pytest.approx(0.05)


def test_on_time_rate_defaults_neutral_for_never_seen():
    st = DeviceSchedulerState(device_id=DeviceID("d"))
    v = extract_features(st, Bucket.NEW, _env())
    assert float(v[4]) == 0.5


def test_on_time_rate_uses_last_outcome():
    st_ok = DeviceSchedulerState(
        device_id=DeviceID("d"), last_outcome=MissionOutcome.CLEAN
    )
    st_bad = DeviceSchedulerState(
        device_id=DeviceID("d"), last_outcome=MissionOutcome.TIMEOUT
    )
    v_ok = extract_features(st_ok, Bucket.NEW, _env())
    v_bad = extract_features(st_bad, Bucket.NEW, _env())
    assert float(v_ok[4]) == 1.0
    assert float(v_bad[4]) == 0.0


def test_beacon_fresh_flag():
    now = 1000.0
    st_fresh = DeviceSchedulerState(
        device_id=DeviceID("d"), last_beacon_ts=995.0
    )
    st_stale = DeviceSchedulerState(
        device_id=DeviceID("d"), last_beacon_ts=1.0
    )
    assert float(
        extract_features(st_fresh, Bucket.NEW, _env(now=now))[5]
    ) == 1.0
    assert float(
        extract_features(st_stale, Bucket.NEW, _env(now=now))[5]
    ) == 0.0


def test_bucket_one_hot_distinct_for_each_bucket():
    st = DeviceSchedulerState(device_id=DeviceID("d"))
    v_new = extract_features(st, Bucket.NEW, _env())
    v_sched = extract_features(st, Bucket.SCHEDULED_THIS_ROUND, _env())
    v_bcn = extract_features(st, Bucket.BEACON_ACTIVE, _env())
    # Bucket one-hot lives at indices 6,7,8.
    assert list(v_new[6:9]) == [1.0, 0.0, 0.0]
    assert list(v_sched[6:9]) == [0.0, 1.0, 0.0]
    assert list(v_bcn[6:9]) == [0.0, 0.0, 1.0]


def test_batch_stacks_rows_in_order():
    states = {
        DeviceID("a"): DeviceSchedulerState(
            device_id=DeviceID("a"), last_known_position=(1.0, 0.0, 0.0)
        ),
        DeviceID("b"): DeviceSchedulerState(
            device_id=DeviceID("b"), last_known_position=(2.0, 0.0, 0.0)
        ),
    }
    feats = extract_features_batch(
        [DeviceID("a"), DeviceID("b")], states, Bucket.NEW, _env()
    )
    assert feats.shape == (2, FEATURE_DIM)
    # Distances 1 and 2 normalised by _DISTANCE_SCALE=100.
    assert float(feats[0, 0]) == pytest.approx(0.01)
    assert float(feats[1, 0]) == pytest.approx(0.02)


def test_batch_missing_state_raises():
    with pytest.raises(KeyError):
        extract_features_batch(
            [DeviceID("ghost")], {}, Bucket.NEW, _env()
        )


def test_batch_empty_returns_zero_rows():
    feats = extract_features_batch([], {}, Bucket.NEW, _env())
    assert feats.shape == (0, FEATURE_DIM)
