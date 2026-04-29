"""Small DDQN actor for :class:`TargetSelectorRL`.

Pointer-style Q-function: given a *single candidate's* features, output
a scalar Q. The caller scores each candidate in a bucket independently
and takes the argmax. This keeps the network tiny and the action space
input-agnostic — no fixed-size output head tied to registry size.

Architecture (numpy, no framework dep):

    q = (tanh(x W1 + b1)) W2 + b2        # x: (..., F) -> (..., 1)

Two copies of the params are kept: ``online`` (trained every step) and
``target`` (copied every ``target_sync_every`` steps). This is the
double-Q trick: action is chosen by ``online``, Q estimate at ``s'`` is
read from ``target``.

Why numpy, not TF/Keras?
* The problem is tiny: ~11 features, ~16 hidden units, a few candidates
  per pass. A framework would dominate the cost.
* The training harness runs offline on the digital twin; the inference
  path on the NUC is < 200 ops per bucket — dwarfed by the RF link.
* Keeps the hermes package free of a heavy runtime dep.

If Phase 5 later grows to a shared encoder / multi-head design, swap
this for a Keras model — the :class:`TargetSelectorRL` wrapper hides
the internals behind ``predict`` / ``update``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .replay import ReplayBuffer, Transition


@dataclass
class _Params:
    W1: np.ndarray  # (F, H)
    b1: np.ndarray  # (H,)
    W2: np.ndarray  # (H, 1)
    b2: np.ndarray  # (1,)

    def copy(self) -> "_Params":
        return _Params(
            W1=self.W1.copy(),
            b1=self.b1.copy(),
            W2=self.W2.copy(),
            b2=self.b2.copy(),
        )


def _init_params(feature_dim: int, hidden: int, rng: np.random.Generator) -> _Params:
    # Xavier-ish; tanh activation so sqrt(1/fan_in).
    W1 = rng.normal(0.0, 1.0 / np.sqrt(feature_dim), size=(feature_dim, hidden))
    W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), size=(hidden, 1))
    return _Params(
        W1=W1.astype(np.float32),
        b1=np.zeros(hidden, dtype=np.float32),
        W2=W2.astype(np.float32),
        b2=np.zeros(1, dtype=np.float32),
    )


def _forward(x: np.ndarray, p: _Params) -> Tuple[np.ndarray, np.ndarray]:
    """x: (B, F) -> (q: (B,), hidden: (B, H)) so backprop can reuse ``hidden``."""
    pre = x @ p.W1 + p.b1         # (B, H)
    h = np.tanh(pre)              # (B, H)
    q = (h @ p.W2 + p.b2).squeeze(-1)  # (B,)
    return q, h


class DDQN:
    """Double-DQN actor over scalar-Q-per-candidate feature vectors.

    Parameters are lightweight numpy arrays; ``predict`` is a pure
    forward pass; ``update`` runs one SGD step on a sampled mini-batch.

    Thread-safety: not safe under concurrent ``update`` — one trainer
    owns the actor. Inference is read-only and safe.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 16,
        lr: float = 0.01,
        gamma: float = 0.5,
        target_sync_every: int = 200,
        seed: Optional[int] = None,
    ):
        if feature_dim <= 0 or hidden <= 0:
            raise ValueError("feature_dim and hidden must be positive")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0,1), got {gamma}")
        self._rng = np.random.default_rng(seed)
        self._online = _init_params(feature_dim, hidden, self._rng)
        self._target = self._online.copy()
        self._lr = float(lr)
        self._gamma = float(gamma)
        self._target_sync_every = int(target_sync_every)
        self._steps = 0
        self._feature_dim = feature_dim
        self._hidden = hidden

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def steps(self) -> int:
        return self._steps

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Online-network Q for a batch of feature rows.

        Accepts ``(F,)`` (single candidate) or ``(K, F)`` (bucket batch).
        Returns ``(1,)`` or ``(K,)`` respectively.
        """
        if features.ndim == 1:
            q, _ = _forward(features.reshape(1, -1), self._online)
        elif features.ndim == 2:
            q, _ = _forward(features, self._online)
        else:
            raise ValueError(f"features must be 1- or 2-D, got shape {features.shape}")
        return q

    def argmax(self, features: np.ndarray) -> int:
        """Return ``argmax_k`` for a ``(K, F)`` bucket batch."""
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(
                f"argmax needs a (K, F) batch with K>=1; got {features.shape}"
            )
        q = self.predict(features)
        return int(np.argmax(q))

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def _target_q(self, x: np.ndarray) -> np.ndarray:
        q, _ = _forward(x, self._target)
        return q

    def update(self, batch: list) -> float:
        """One SGD step on ``batch: List[Transition]``. Returns mean MSE loss."""
        if not batch:
            return 0.0

        states = np.stack([t.state for t in batch], axis=0)
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        dones = np.asarray([1.0 if t.done else 0.0 for t in batch], dtype=np.float32)

        # Build the bootstrap term from target net. Terminal transitions
        # zero the bootstrap (done=1 masks it out).
        next_states_list = [t.next_state for t in batch]
        next_mask = np.asarray(
            [0.0 if (ns is None) else 1.0 for ns in next_states_list],
            dtype=np.float32,
        )
        # Stack with zero-vector stand-ins for terminal next_states so we
        # can forward in one shot; ``next_mask`` masks the contribution.
        dummy = np.zeros(self._feature_dim, dtype=np.float32)
        next_stacked = np.stack(
            [ns if ns is not None else dummy for ns in next_states_list], axis=0
        )
        q_next = self._target_q(next_stacked)  # (B,)

        targets = rewards + self._gamma * (1.0 - dones) * next_mask * q_next  # (B,)

        # Forward online net (keep hidden for backprop).
        pre = states @ self._online.W1 + self._online.b1
        h = np.tanh(pre)
        q_pred = (h @ self._online.W2 + self._online.b2).squeeze(-1)
        err = q_pred - targets  # (B,)
        loss = float(np.mean(err ** 2))

        # Backprop MSE wrt online params.
        B = states.shape[0]
        dq = (2.0 / B) * err  # (B,)
        # dL/dW2 = hᵀ dq    (H, B) · (B, 1) -> (H, 1)
        gW2 = h.T @ dq.reshape(-1, 1)
        gb2 = np.array([dq.sum()], dtype=np.float32)
        # Back through tanh: dL/dh = dq * W2ᵀ; dL/dpre = dh * (1 - h^2)
        dh = dq.reshape(-1, 1) * self._online.W2.T  # (B, H)
        dpre = dh * (1.0 - h ** 2)
        gW1 = states.T @ dpre            # (F, H)
        gb1 = dpre.sum(axis=0)           # (H,)

        # SGD step (plain).
        self._online.W1 -= self._lr * gW1.astype(np.float32)
        self._online.b1 -= self._lr * gb1.astype(np.float32)
        self._online.W2 -= self._lr * gW2.astype(np.float32)
        self._online.b2 -= self._lr * gb2.astype(np.float32)

        # Target sync.
        self._steps += 1
        if self._steps % self._target_sync_every == 0:
            self._target = self._online.copy()

        return loss

    # ------------------------------------------------------------------ #
    # State I/O — keeps training harness decoupled from file format
    # ------------------------------------------------------------------ #

    def get_weights(self) -> dict:
        return {
            "W1": self._online.W1.copy(),
            "b1": self._online.b1.copy(),
            "W2": self._online.W2.copy(),
            "b2": self._online.b2.copy(),
        }

    def set_weights(self, weights: dict) -> None:
        self._online = _Params(
            W1=np.asarray(weights["W1"], dtype=np.float32).copy(),
            b1=np.asarray(weights["b1"], dtype=np.float32).copy(),
            W2=np.asarray(weights["W2"], dtype=np.float32).copy(),
            b2=np.asarray(weights["b2"], dtype=np.float32).copy(),
        )
        self._target = self._online.copy()
