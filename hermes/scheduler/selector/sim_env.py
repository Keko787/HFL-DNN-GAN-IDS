"""Tiny offline sim — drives :class:`TargetSelectorRL` training.

Design §2.7 reward shape::

    −time_to_complete − w·energy + completed_session_bonus

The real training domain is the AERPAW digital twin (Phase 6). This sim
is small enough to keep in-process so the Phase-5 DoD A/B test runs in
a unit test: selector vs distance-placeholder over the same seeds.

Model
-----
Each episode is a *bucket* of ``K`` candidate devices with random:

* position (2-D, z=0)
* reliability ``r ∈ [0.3, 1.0]`` — feeds per-device completion prob and
  the ``on_time_rate`` feature (after we "observe" it for a few rounds).
* last-outcome CLEAN/TIMEOUT drawn from reliability.

The mule starts at origin with ``mule_energy = 1.0`` and visits one
candidate per step. Visit outcome:

* ``time_to_complete  = dist + ε``
* ``energy_used       = dist · ENERGY_W``
* ``completed         = Bernoulli(reliability · rf_factor)``
* ``reward            = −time − w_e·energy + (BONUS if completed else 0)``

The mule pose advances to the chosen device after each step; the
remaining candidates stay where they are.

This is not physics — it's just enough noise that the right policy is
measurably better than pure distance when some candidates are
"near but flaky" and others "further but reliable".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hermes.types import (
    Bucket,
    DeviceID,
    DeviceSchedulerState,
    MissionOutcome,
)

from .features import SelectorEnv


ENERGY_W: float = 0.002
# Bonus must dominate the distance penalty so a completion-maximizing
# policy is also reward-maximizing — otherwise the RL selector optimises
# a proxy that diverges from the Phase-5 DoD metric (completion rate),
# and a near-greedy placeholder wins by accident.
COMPLETION_BONUS: float = 200.0
TIME_NOISE_STD: float = 0.5


@dataclass
class _SimDevice:
    device_id: DeviceID
    pos: Tuple[float, float, float]
    reliability: float  # ∈ [0,1], ground truth


@dataclass
class StepResult:
    device_id: DeviceID
    reward: float
    time_to_complete: float
    energy_used: float
    completed: bool


class BucketSim:
    """One bucket, K candidates, sequential visits.

    ``reset`` produces a fresh bucket; each ``step(action_idx)`` visits
    one candidate and returns a ``StepResult``. When the bucket is empty
    the episode is done.

    The env also exposes the scheduler-shaped ``DeviceSchedulerState``
    map and the ``SelectorEnv`` object so callers can feed the selector
    without reshaping anything.
    """

    def __init__(
        self,
        bucket_size: int = 6,
        world_radius: float = 100.0,
        energy_weight: float = 1.0,
        seed: Optional[int] = None,
    ):
        if bucket_size <= 1:
            raise ValueError(f"bucket_size must be > 1, got {bucket_size}")
        self._bucket_size = bucket_size
        self._world_radius = world_radius
        self._energy_weight = energy_weight
        self._rng = np.random.default_rng(seed)
        self._mule_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._mule_energy: float = 1.0
        self._devices: List[_SimDevice] = []
        self._states: Dict[DeviceID, DeviceSchedulerState] = {}
        self._step_count: int = 0

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._mule_pose = (0.0, 0.0, 0.0)
        self._mule_energy = 1.0
        self._devices = []
        self._states = {}
        self._step_count = 0

        # Pre-observe N past rounds so the `last_outcome` proxy tracks
        # the device's underlying reliability rather than a single noisy
        # Bernoulli sample. Models production, where on_time_history
        # has accumulated over many missions before the selector runs.
        _PRE_ROUNDS = 5
        for k in range(self._bucket_size):
            did = DeviceID(f"sim-{k:02d}")
            x = float(self._rng.uniform(-self._world_radius, self._world_radius))
            y = float(self._rng.uniform(-self._world_radius, self._world_radius))
            reliability = float(self._rng.uniform(0.3, 1.0))
            self._devices.append(
                _SimDevice(device_id=did, pos=(x, y, 0.0), reliability=reliability)
            )
            hits = sum(
                1 for _ in range(_PRE_ROUNDS) if self._rng.random() < reliability
            )
            last_outcome = (
                MissionOutcome.CLEAN
                if hits * 2 >= _PRE_ROUNDS
                else MissionOutcome.TIMEOUT
            )
            self._states[did] = DeviceSchedulerState(
                device_id=did,
                is_in_slice=True,
                is_new=False,
                last_known_position=(x, y, 0.0),
                last_outcome=last_outcome,
            )

    # ------------------------------------------------------------------ #
    # Snapshot accessors
    # ------------------------------------------------------------------ #

    @property
    def mule_pose(self) -> Tuple[float, float, float]:
        return self._mule_pose

    @property
    def mule_energy(self) -> float:
        return self._mule_energy

    @property
    def done(self) -> bool:
        return not self._devices

    def candidates(self) -> List[DeviceID]:
        return [d.device_id for d in self._devices]

    def device_states(self) -> Dict[DeviceID, DeviceSchedulerState]:
        return {d.device_id: self._states[d.device_id] for d in self._devices}

    def selector_env(self) -> SelectorEnv:
        return SelectorEnv(
            mule_pose=self._mule_pose,
            mule_energy=self._mule_energy,
            rf_prior_snr_db=20.0,
            now=0.0,
        )

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, action_idx: int) -> StepResult:
        if self.done:
            raise RuntimeError("step called on terminated episode")
        if action_idx < 0 or action_idx >= len(self._devices):
            raise IndexError(
                f"action_idx={action_idx} outside [0, {len(self._devices)})"
            )

        dev = self._devices[action_idx]
        dist = float(
            np.sqrt(sum((a - b) ** 2 for a, b in zip(self._mule_pose, dev.pos)))
        )
        noise = float(self._rng.normal(0.0, TIME_NOISE_STD))
        time_to_complete = max(0.0, dist + noise)
        energy_used = dist * ENERGY_W

        # Distance-tied RF noise: kept mild so reliability dominates
        # p_complete — otherwise argmin(distance) is already the oracle
        # and no selector can beat it. rf_factor ∈ [0.7, 1.0].
        rf_factor = max(0.7, 1.0 - dist / (5 * self._world_radius))
        p_complete = max(0.0, min(1.0, dev.reliability * rf_factor))
        completed = bool(self._rng.random() < p_complete)

        reward = (
            -time_to_complete
            - self._energy_weight * energy_used
            + (COMPLETION_BONUS if completed else 0.0)
        )

        # State advancement.
        self._mule_pose = dev.pos
        self._mule_energy = max(0.0, self._mule_energy - energy_used)
        self._devices.pop(action_idx)
        self._step_count += 1

        return StepResult(
            device_id=dev.device_id,
            reward=reward,
            time_to_complete=time_to_complete,
            energy_used=energy_used,
            completed=completed,
        )


def distance_policy_choice(
    candidates: Sequence[DeviceID],
    device_states: Dict[DeviceID, DeviceSchedulerState],
    mule_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> int:
    """Index-of-nearest argmin — the placeholder the RL selector must beat."""
    def _key(i: int) -> float:
        st = device_states.get(candidates[i])
        if st is None:
            return float("inf")
        pos = st.last_known_position
        return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(mule_pose, pos))))

    return min(range(len(candidates)), key=_key)
