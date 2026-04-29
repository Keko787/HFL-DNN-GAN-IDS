"""Replay buffer — experience storage for offline CTDE training.

Design §2.7 "replay_buffer": trained centrally, deployed small. The
buffer is intentionally tiny (tens of thousands of transitions at most)
because the intra-bucket selection problem has a low-dim state and we
only train offline on the AERPAW digital twin.

A transition is flat, independent of the selector's input shape: the
upstream caller already turned the state into the feature vector for
the chosen candidate. This keeps the buffer and the network decoupled.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Transition:
    """One (state, reward, next_state, done) tuple.

    ``state`` is the feature vector of the *chosen* candidate at step t
    (a single row from :func:`extract_features_batch`). ``next_state``
    is the feature vector chosen at step t+1 under the online policy —
    ``None`` when the episode ends (``done=True``).
    """

    state: np.ndarray
    reward: float
    next_state: Optional[np.ndarray]
    done: bool


class ReplayBuffer:
    """Fixed-capacity FIFO buffer with uniform random sampling."""

    def __init__(self, capacity: int = 10_000, seed: Optional[int] = None):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._buffer: List[Transition] = []
        self._next_idx = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(self, transition: Transition) -> None:
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._next_idx] = transition
        self._next_idx = (self._next_idx + 1) % self._capacity

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size={batch_size} > buffer size={len(self._buffer)}"
            )
        return self._rng.sample(self._buffer, batch_size)

    def clear(self) -> None:
        self._buffer.clear()
        self._next_idx = 0
