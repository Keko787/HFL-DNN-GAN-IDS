"""Phase 5 — DDQN sanity tests.

Verifies forward pass, argmax shape, and that a SGD step on a trivial
single-transition "reward=+1 regardless" problem nudges the Q estimate
toward the target (learning happens at all).
"""

from __future__ import annotations

import numpy as np
import pytest

from hermes.scheduler.selector import DDQN, Transition


def test_forward_single_vector():
    net = DDQN(feature_dim=4, hidden=8, seed=0)
    q = net.predict(np.zeros(4, dtype=np.float32))
    assert q.shape == (1,)


def test_forward_batch():
    net = DDQN(feature_dim=4, hidden=8, seed=0)
    q = net.predict(np.zeros((5, 4), dtype=np.float32))
    assert q.shape == (5,)


def test_argmax_on_batch():
    net = DDQN(feature_dim=4, hidden=8, seed=0)
    # Craft two rows where the second should have higher output if
    # network is not pathologically broken — but we just test shape.
    feats = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    idx = net.argmax(feats)
    assert 0 <= idx < 3


def test_argmax_rejects_empty_batch():
    net = DDQN(feature_dim=4, hidden=8, seed=0)
    with pytest.raises(ValueError):
        net.argmax(np.zeros((0, 4), dtype=np.float32))


def test_update_reduces_loss_on_trivial_problem():
    """Single transition terminal with reward=5; Q should move toward 5."""
    net = DDQN(feature_dim=3, hidden=8, lr=0.1, seed=0)
    x = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    pre = float(net.predict(x)[0])
    batch = [Transition(state=x, reward=5.0, next_state=None, done=True)] * 16
    for _ in range(50):
        net.update(batch)
    post = float(net.predict(x)[0])
    # post should be strictly closer to 5 than pre.
    assert abs(post - 5.0) < abs(pre - 5.0)


def test_target_network_syncs_every_k_steps():
    net = DDQN(feature_dim=2, hidden=4, lr=0.1, target_sync_every=3, seed=0)
    x = np.array([0.5, -0.5], dtype=np.float32)
    batch = [Transition(state=x, reward=1.0, next_state=None, done=True)]
    # Before any update, target and online agree by construction.
    before = float(net._target_q(x.reshape(1, -1))[0])  # type: ignore[attr-defined]
    for _ in range(3):
        net.update(batch)
    after = float(net._target_q(x.reshape(1, -1))[0])  # type: ignore[attr-defined]
    # After 3 updates the sync fired; target now tracks the online net.
    assert after != before


def test_weight_roundtrip():
    net1 = DDQN(feature_dim=3, hidden=4, seed=0)
    w = net1.get_weights()
    net2 = DDQN(feature_dim=3, hidden=4, seed=99)
    net2.set_weights(w)
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert float(net1.predict(x)[0]) == pytest.approx(float(net2.predict(x)[0]))
