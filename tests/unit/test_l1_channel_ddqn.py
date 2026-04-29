"""Phase 5 — L1 ChannelDDQN tests."""

from __future__ import annotations

import numpy as np
import pytest

from hermes.l1 import CHANNEL_FREQS_GHZ, ChannelDDQN
from hermes.l1.channel_ddqn import (
    L1_ACTION_DIM,
    L1_STATE_DIM,
    build_l1_state,
)


def _state() -> np.ndarray:
    return np.zeros(L1_STATE_DIM, dtype=np.float32)


def test_predict_shape_matches_action_dim():
    net = ChannelDDQN(seed=0)
    q = net.predict(_state())
    assert q.shape == (L1_ACTION_DIM,)
    assert q.dtype == np.float32


def test_predict_rejects_wrong_shape():
    net = ChannelDDQN(seed=0)
    with pytest.raises(ValueError):
        net.predict(np.zeros(L1_STATE_DIM - 1, dtype=np.float32))


def test_argmax_index_in_range():
    net = ChannelDDQN(seed=0)
    idx = net.argmax(_state())
    assert 0 <= idx < L1_ACTION_DIM


def test_pick_frequency_matches_table():
    net = ChannelDDQN(seed=0)
    f = net.pick_frequency_ghz(_state())
    assert f in CHANNEL_FREQS_GHZ


def test_channel_freqs_match_slide_26():
    # Slide 26 bands.
    assert CHANNEL_FREQS_GHZ == (3.32, 3.34, 3.90)


def test_weight_roundtrip_reproduces_output():
    a = ChannelDDQN(seed=0)
    b = ChannelDDQN(seed=999)
    b.set_weights(a.get_weights())
    x = np.arange(L1_STATE_DIM, dtype=np.float32) / 10.0
    assert np.allclose(a.predict(x), b.predict(x))


def test_set_weights_rejects_missing_keys():
    net = ChannelDDQN(seed=0)
    w = net.get_weights()
    del w["W2"]
    with pytest.raises(KeyError):
        net.set_weights(w)


def test_set_weights_rejects_shape_mismatch():
    net = ChannelDDQN(hidden=16, seed=0)
    bad = net.get_weights()
    bad["W1"] = np.zeros((L1_STATE_DIM, 8), dtype=np.float32)  # wrong hidden
    with pytest.raises(ValueError):
        net.set_weights(bad)


# --------------------------------------------------------------------------- #
# build_l1_state
# --------------------------------------------------------------------------- #

def test_build_l1_state_layout():
    v = build_l1_state(
        snr_by_band_db=(30.0, 15.0, 0.0),
        mule_pose=(1.0, 2.0, 3.0),
        target_pos=(1.0, 2.0, 3.0),
        mule_energy=0.75,
    )
    assert v.shape == (L1_STATE_DIM,)
    assert v.dtype == np.float32
    # Normalised SNRs.
    assert v[0] == pytest.approx(1.0)
    assert v[1] == pytest.approx(0.5)
    assert v[2] == pytest.approx(0.0)
    # Distance zero / 100 = 0.
    assert v[3] == pytest.approx(0.0)
    # Mule pose passthrough.
    assert v[4] == pytest.approx(1.0)
    assert v[5] == pytest.approx(2.0)
    assert v[6] == pytest.approx(3.0)
    # Energy passthrough.
    assert v[7] == pytest.approx(0.75)


def test_build_l1_state_distance_computation():
    v = build_l1_state(
        snr_by_band_db=(0.0, 0.0, 0.0),
        mule_pose=(0.0, 0.0, 0.0),
        target_pos=(3.0, 4.0, 0.0),
        mule_energy=0.0,
    )
    # 3-4-5 triangle; normalised by 100.
    assert float(v[3]) == pytest.approx(0.05)


def test_build_l1_state_rejects_wrong_band_count():
    with pytest.raises(ValueError):
        build_l1_state(
            snr_by_band_db=(10.0, 20.0),  # type: ignore[arg-type]
            mule_pose=(0.0, 0.0, 0.0),
            target_pos=(0.0, 0.0, 0.0),
            mule_energy=1.0,
        )
