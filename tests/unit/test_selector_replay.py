"""Phase 5 — ReplayBuffer tests."""

from __future__ import annotations

import numpy as np
import pytest

from hermes.scheduler.selector import ReplayBuffer, Transition


def _t(i: int) -> Transition:
    return Transition(
        state=np.array([float(i)], dtype=np.float32),
        reward=float(i),
        next_state=None,
        done=True,
    )


def test_push_and_len():
    buf = ReplayBuffer(capacity=10, seed=0)
    assert len(buf) == 0
    buf.push(_t(1))
    assert len(buf) == 1


def test_fifo_eviction_at_capacity():
    buf = ReplayBuffer(capacity=3, seed=0)
    for i in range(5):
        buf.push(_t(i))
    assert len(buf) == 3
    # The oldest two (0, 1) are evicted; {2, 3, 4} remain.
    remaining_rewards = sorted(int(t.reward) for t in buf.sample(3))
    assert remaining_rewards == [2, 3, 4]


def test_sample_deterministic_under_seed():
    buf = ReplayBuffer(capacity=10, seed=42)
    for i in range(5):
        buf.push(_t(i))
    a = [int(t.reward) for t in buf.sample(3)]
    buf2 = ReplayBuffer(capacity=10, seed=42)
    for i in range(5):
        buf2.push(_t(i))
    b = [int(t.reward) for t in buf2.sample(3)]
    assert a == b


def test_sample_batch_too_large_raises():
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.push(_t(1))
    with pytest.raises(ValueError):
        buf.sample(5)


def test_clear_resets_state():
    buf = ReplayBuffer(capacity=10, seed=0)
    for i in range(5):
        buf.push(_t(i))
    buf.clear()
    assert len(buf) == 0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)
