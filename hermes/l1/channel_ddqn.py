"""Channel-only DDQN — L1 RF-band picker on the NUC.

Design §2.6::

    Output: channel_index (DDQN, discrete head, argmax over RF bands)

This is deliberately a smaller/simpler DDQN than the scheduler's S3.5
actor: the candidate set is fixed (3 RF bands), so we use a Q-function
with one output per band rather than the per-candidate scorer pattern.

State vector (8 features, float32):

     idx   name
     ────────────────────────────────────────────
      0    snr_band_0  (dB / 30)
      1    snr_band_1  (dB / 30)
      2    snr_band_2  (dB / 30)
      3    distance_to_target (normalised by 100)
      4    mule_x
      5    mule_y
      6    mule_z
      7    mule_energy

Training stays on GPU/AERPAW; the NUC runs inference only (this module
exposes ``predict`` / ``argmax`` for the hot path and ``set_weights``
for deploy-time load from checkpoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Slide 26 bands.
CHANNEL_FREQS_GHZ: Tuple[float, float, float] = (3.32, 3.34, 3.90)
L1_STATE_DIM: int = 8
L1_ACTION_DIM: int = 3


def _init_l1_params(
    state_dim: int, hidden: int, action_dim: int, rng: np.random.Generator
):
    W1 = rng.normal(0.0, 1.0 / np.sqrt(state_dim), size=(state_dim, hidden))
    W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), size=(hidden, action_dim))
    return {
        "W1": W1.astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "W2": W2.astype(np.float32),
        "b2": np.zeros(action_dim, dtype=np.float32),
    }


class ChannelDDQN:
    """Small discrete-action DDQN for channel selection.

    Inference-only in production — the ``predict``/``argmax`` path has
    no randomness. ``set_weights`` loads from a checkpoint produced
    off-box (AERPAW / GPU rig); ``get_weights`` snapshots for debug.
    """

    def __init__(
        self,
        *,
        hidden: int = 16,
        seed: Optional[int] = None,
        action_dim: int = L1_ACTION_DIM,
        state_dim: int = L1_STATE_DIM,
    ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden = hidden
        self._rng = np.random.default_rng(seed)
        self._params = _init_l1_params(state_dim, hidden, action_dim, self._rng)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Q-values for each channel. State shape ``(D,)`` -> ``(A,)``."""
        if state.shape != (self._state_dim,):
            raise ValueError(
                f"state must be shape ({self._state_dim},), got {state.shape}"
            )
        p = self._params
        h = np.tanh(state @ p["W1"] + p["b1"])
        q = h @ p["W2"] + p["b2"]
        return q.astype(np.float32)

    def argmax(self, state: np.ndarray) -> int:
        """Index into :data:`CHANNEL_FREQS_GHZ` of the best band for this state."""
        return int(np.argmax(self.predict(state)))

    def pick_frequency_ghz(self, state: np.ndarray) -> float:
        return CHANNEL_FREQS_GHZ[self.argmax(state)]

    # ------------------------------------------------------------------ #
    # Deploy-time weight I/O
    # ------------------------------------------------------------------ #

    def get_weights(self) -> dict:
        return {k: v.copy() for k, v in self._params.items()}

    def set_weights(self, weights: dict) -> None:
        expected = {"W1", "b1", "W2", "b2"}
        missing = expected - set(weights)
        if missing:
            raise KeyError(f"set_weights missing keys: {missing}")
        new = {k: np.asarray(weights[k], dtype=np.float32).copy() for k in expected}
        # Shape sanity — refuse to load a non-matching checkpoint.
        if new["W1"].shape != (self._state_dim, self._hidden):
            raise ValueError(
                f"W1 shape {new['W1'].shape} != "
                f"({self._state_dim}, {self._hidden})"
            )
        if new["W2"].shape != (self._hidden, self._action_dim):
            raise ValueError(
                f"W2 shape {new['W2'].shape} != "
                f"({self._hidden}, {self._action_dim})"
            )
        self._params = new


# --------------------------------------------------------------------------- #
# State builder — how the mule supervisor assembles an input for ``predict``
# --------------------------------------------------------------------------- #

def build_l1_state(
    snr_by_band_db: Tuple[float, float, float],
    mule_pose: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    mule_energy: float,
) -> np.ndarray:
    """Pack the §2.6 features into the 8-D state vector ChannelDDQN expects."""
    if len(snr_by_band_db) != L1_ACTION_DIM:
        raise ValueError(
            f"snr_by_band_db must have {L1_ACTION_DIM} entries, "
            f"got {len(snr_by_band_db)}"
        )
    dist = float(
        np.sqrt(sum((a - b) ** 2 for a, b in zip(mule_pose, target_pos)))
    )
    return np.asarray(
        [
            snr_by_band_db[0] / 30.0,
            snr_by_band_db[1] / 30.0,
            snr_by_band_db[2] / 30.0,
            dist / 100.0,
            float(mule_pose[0]),
            float(mule_pose[1]),
            float(mule_pose[2]),
            float(mule_energy),
        ],
        dtype=np.float32,
    )
