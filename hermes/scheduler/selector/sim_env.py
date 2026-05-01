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
* reliability ``r ∈ [0.15, 1.0]`` — feeds per-device completion prob and
  the ``on_time_rate`` feature (after we "observe" it for a few rounds).
* last-outcome CLEAN/TIMEOUT drawn from reliability.

The mule starts at origin with ``mule_energy = 1.0`` and visits one
candidate per step. Visit outcome:

* ``time_to_complete  = SESSION_TIME + TIME_PER_DIST · dist + ε``
* ``energy_used       = dist · ENERGY_W``
* ``rf_factor         = max(0.4, 1 − dist / (3 · world_radius))``
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
# A visit's wall-clock is dominated by the FL session itself (handshake +
# local train + gradient receive), not the flight. Modeling time as
# `SESSION_TIME + TIME_PER_DIST·dist` keeps the distance penalty small
# next to the completion bonus, so the optimal policy is "pick reliable"
# rather than "pick nearest" — which is the regime HERMES actually
# operates in. Otherwise distance penalty dominates reward and any
# selector collapses onto the distance-greedy oracle.
SESSION_TIME: float = 30.0
TIME_PER_DIST: float = 0.1
TIME_NOISE_STD: float = 1.0


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


@dataclass
class EpisodeMetrics:
    """Per-episode aggregate stats for the multi-metric A/B scoreboard.

    All fields are post-hoc roll-ups of the per-step ``StepResult`` stream,
    so the smart-vs-dumb test can score on energy / path / retry without
    re-running the sim.
    """

    visits: int = 0
    completions: int = 0
    energy_total: float = 0.0
    path_length: float = 0.0
    time_total: float = 0.0

    @property
    def completion_rate(self) -> float:
        return self.completions / self.visits if self.visits else 0.0

    @property
    def retry_rate(self) -> float:
        # A non-completed visit in HERMES would surface as partial/timeout,
        # which is wasted energy + a session that has to be re-attempted.
        return 1.0 - self.completion_rate


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
        # Mission time budget in the same units as ``time_to_complete``.
        # The default lets ~4 of 6 devices get visited on average, which
        # is what creates the skip pressure that makes selector ordering
        # matter. With unbounded visits the problem reduces to TSP and
        # distance-greedy is near-optimal — no selector can beat it.
        mission_budget: float = 150.0,
        seed: Optional[int] = None,
    ):
        if bucket_size <= 1:
            raise ValueError(f"bucket_size must be > 1, got {bucket_size}")
        if mission_budget <= 0.0:
            raise ValueError(f"mission_budget must be > 0, got {mission_budget}")
        self._bucket_size = bucket_size
        self._world_radius = world_radius
        self._energy_weight = energy_weight
        self._mission_budget = mission_budget
        self._rng = np.random.default_rng(seed)
        self._mule_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._mule_energy: float = 1.0
        self._budget_remaining: float = mission_budget
        self._devices: List[_SimDevice] = []
        self._states: Dict[DeviceID, DeviceSchedulerState] = {}
        self._step_count: int = 0
        self._episode_metrics: EpisodeMetrics = EpisodeMetrics()

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._mule_pose = (0.0, 0.0, 0.0)
        self._mule_energy = 1.0
        self._budget_remaining = self._mission_budget
        self._devices = []
        self._states = {}
        self._step_count = 0
        self._episode_metrics = EpisodeMetrics()

        # Pre-observe N past rounds so the `last_outcome` proxy tracks
        # the device's underlying reliability rather than a single noisy
        # Bernoulli sample. Models production, where on_time_history
        # has accumulated over many missions before the selector runs.
        # 11 rounds is odd (no 50/50 ties) and large enough that the
        # majority vote concentrates around the true reliability — without
        # this, a 0.9-reliability device flips to TIMEOUT often enough to
        # poison the selector's only signal that distinguishes devices.
        _PRE_ROUNDS = 11
        for k in range(self._bucket_size):
            did = DeviceID(f"sim-{k:02d}")
            x = float(self._rng.uniform(-self._world_radius, self._world_radius))
            y = float(self._rng.uniform(-self._world_radius, self._world_radius))
            # Wider reliability range — some devices are clearly bad bets
            # and some clearly good. The tighter [0.3, 1.0] from the first
            # cut left no clearly-skip-worthy devices, so picking nearest
            # was always within a few % of optimal.
            reliability = float(self._rng.uniform(0.15, 1.0))
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
                # Continuous reliability proxy — what the selector sees.
                on_time_count=hits,
                missed_count=_PRE_ROUNDS - hits,
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
        # Bucket exhausted, OR no time left to even start another visit
        # (cheapest possible visit is SESSION_TIME). The budget cutoff is
        # what gives the selector something to optimize — choosing well
        # determines which devices get served before time runs out.
        if not self._devices:
            return True
        return self._budget_remaining < SESSION_TIME

    @property
    def budget_remaining(self) -> float:
        return self._budget_remaining

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

    @property
    def episode_metrics(self) -> EpisodeMetrics:
        """Snapshot of the running aggregate stats — read after each step."""
        return self._episode_metrics

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
        # Wall-clock = fixed FL session cost + a small flight-time term.
        # In real HERMES the radio session dominates, so a pure
        # `time = dist` model overstates the cost of going slightly
        # farther for a more reliable device.
        time_to_complete = max(0.0, SESSION_TIME + TIME_PER_DIST * dist + noise)
        energy_used = dist * ENERGY_W

        # Distance-driven RF degradation. Floor=0.4 over the working range
        # makes it materially worth checking last_outcome before flying
        # somewhere far — picking a far flaky device tanks completion
        # probability. Earlier cut floored at 0.7, which capped the
        # downside of a wrong pick and made distance-greedy near-optimal.
        rf_factor = max(0.4, 1.0 - dist / (3.0 * self._world_radius))
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
        self._budget_remaining = max(0.0, self._budget_remaining - time_to_complete)
        self._devices.pop(action_idx)
        self._step_count += 1

        # Aggregate roll-up for the multi-metric scoreboard.
        self._episode_metrics.visits += 1
        self._episode_metrics.completions += int(completed)
        self._episode_metrics.energy_total += energy_used
        self._episode_metrics.path_length += dist
        self._episode_metrics.time_total += time_to_complete

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


# --------------------------------------------------------------------------- #
# Sprint 1.5 — ContactSim
#
# Per-contact-event simulation environment. Generates N devices spread in
# a world, S3a-clusters them by ``rf_range_m`` into contact waypoints, and
# steps over CONTACTS instead of individual devices.
#
# Each ``step`` visits one contact event (parallel sessions with all
# in-range devices). The reward sums the completion bonuses across the
# contact's members, minus flight time and energy.
#
# The selector's per-contact methods (``select_contact``/``rank_contacts``)
# operate directly on the candidates this sim produces, so the offline
# A/B test in Chunk G uses the same code path as the live MuleSupervisor
# at Sprint 1.5.
# --------------------------------------------------------------------------- #

from hermes.scheduler.stages.s3a_cluster import cluster_by_rf_range  # noqa: E402
from hermes.types import Bucket, ContactWaypoint  # noqa: E402


@dataclass
class ContactStepResult:
    """One contact event's outcome — superset of StepResult.

    ``per_device_completed`` is a bool list aligned with the contact's
    ``devices`` tuple so callers can reconstruct who completed and who
    didn't. ``completed_count`` is the sum, useful as the per-contact
    reward driver.
    """

    contact: ContactWaypoint
    reward: float
    time_to_complete: float
    energy_used: float
    per_device_completed: List[bool]

    @property
    def completed_count(self) -> int:
        return sum(self.per_device_completed)

    @property
    def member_count(self) -> int:
        return len(self.per_device_completed)


@dataclass
class ContactEpisodeMetrics:
    """Per-episode aggregate stats for the contact-event scoreboard."""

    contacts_visited: int = 0
    devices_visited: int = 0
    devices_completed: int = 0
    energy_total: float = 0.0
    path_length: float = 0.0
    time_total: float = 0.0

    @property
    def completion_rate(self) -> float:
        return self.devices_completed / self.devices_visited if self.devices_visited else 0.0

    @property
    def retry_rate(self) -> float:
        return 1.0 - self.completion_rate

    @property
    def rho_contact(self) -> float:
        """Mean devices-per-contact — paper §V.D Sprint-1.5 metric.

        Captures how effectively the scheduler exploits RF range.
        ρ_contact = 1.0 corresponds to per-device sessions; HERMES at
        moderate rf_range_m should hit 2–3 on a 5–10 device slice.
        """
        return self.devices_visited / self.contacts_visited if self.contacts_visited else 0.0


class ContactSim:
    """Contact-event simulator for Sprint-1.5 selector A/B.

    Lifecycle:

    1. ``reset()`` — generate ``n_devices`` random devices, then S3a-cluster
       them by ``rf_range_m`` into ``ContactWaypoint``s.
    2. ``candidates()`` — returns the unvisited contacts (the selector's
       choice set).
    3. ``step(action_idx)`` — visits the chosen contact: every in-range
       device is served in parallel, completion is Bernoulli per device,
       and a single per-contact ``ContactStepResult`` is returned.
    4. Episode ends when no time budget remains or the contact list is
       exhausted.

    Compared to :class:`BucketSim`, the key change is that the unit of
    work is a *position* (covering N≥1 devices), not a single device.
    This drives the per-contact partial-FedAvg reward shape and the
    ``ρ_contact`` metric.
    """

    def __init__(
        self,
        n_devices: int = 8,
        world_radius: float = 100.0,
        energy_weight: float = 1.0,
        mission_budget: float = 200.0,
        rf_range_m: float = 60.0,
        seed: Optional[int] = None,
    ):
        if n_devices < 1:
            raise ValueError(f"n_devices must be >= 1, got {n_devices}")
        if rf_range_m <= 0.0:
            raise ValueError(f"rf_range_m must be > 0, got {rf_range_m}")
        if mission_budget <= 0.0:
            raise ValueError(f"mission_budget must be > 0, got {mission_budget}")
        self._n_devices = n_devices
        self._world_radius = world_radius
        self._energy_weight = energy_weight
        self._mission_budget = mission_budget
        self._rf_range_m = rf_range_m
        self._rng = np.random.default_rng(seed)

        self._mule_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._mule_energy: float = 1.0
        self._budget_remaining: float = mission_budget

        self._devices: List[_SimDevice] = []
        self._states: Dict[DeviceID, DeviceSchedulerState] = {}
        self._contacts: List[ContactWaypoint] = []
        self._episode_metrics: ContactEpisodeMetrics = ContactEpisodeMetrics()

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._mule_pose = (0.0, 0.0, 0.0)
        self._mule_energy = 1.0
        self._budget_remaining = self._mission_budget
        self._devices = []
        self._states = {}
        self._contacts = []
        self._episode_metrics = ContactEpisodeMetrics()

        # 1. Spawn devices with reliability + pre-observed last_outcome.
        _PRE_ROUNDS = 11
        deadlines: Dict[DeviceID, float] = {}
        for k in range(self._n_devices):
            did = DeviceID(f"sim-{k:02d}")
            x = float(self._rng.uniform(-self._world_radius, self._world_radius))
            y = float(self._rng.uniform(-self._world_radius, self._world_radius))
            reliability = float(self._rng.uniform(0.15, 1.0))
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
            st = DeviceSchedulerState(
                device_id=did,
                is_in_slice=True,
                is_new=False,
                last_known_position=(x, y, 0.0),
                last_outcome=last_outcome,
                on_time_count=hits,
                missed_count=_PRE_ROUNDS - hits,
            )
            st.bucket = Bucket.SCHEDULED_THIS_ROUND
            self._states[did] = st
            deadlines[did] = 999_999.0  # unused; ContactSim ignores deadlines

        # 2. Cluster the devices into contacts by rf_range_m.
        self._contacts = cluster_by_rf_range(
            eligible_device_ids=list(self._states.keys()),
            device_states=self._states,
            deadlines=deadlines,
            rf_range_m=self._rf_range_m,
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
    def budget_remaining(self) -> float:
        return self._budget_remaining

    @property
    def rf_range_m(self) -> float:
        return self._rf_range_m

    @property
    def done(self) -> bool:
        if not self._contacts:
            return True
        return self._budget_remaining < SESSION_TIME

    def candidates(self) -> List[ContactWaypoint]:
        return list(self._contacts)

    def device_states(self) -> Dict[DeviceID, DeviceSchedulerState]:
        # Selector's `select_contact` / `rank_contacts` need device states
        # for every member of every candidate contact.
        return dict(self._states)

    def selector_env(self) -> SelectorEnv:
        return SelectorEnv(
            mule_pose=self._mule_pose,
            mule_energy=self._mule_energy,
            rf_prior_snr_db=20.0,
            now=0.0,
        )

    @property
    def episode_metrics(self) -> ContactEpisodeMetrics:
        return self._episode_metrics

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, action_idx: int) -> ContactStepResult:
        if self.done:
            raise RuntimeError("step called on terminated episode")
        if action_idx < 0 or action_idx >= len(self._contacts):
            raise IndexError(
                f"action_idx={action_idx} outside [0, {len(self._contacts)})"
            )

        contact = self._contacts[action_idx]
        # Distance from current mule pose to the contact's stop position.
        dist = float(
            np.sqrt(sum((a - b) ** 2 for a, b in zip(self._mule_pose, contact.position)))
        )
        noise = float(self._rng.normal(0.0, TIME_NOISE_STD))
        # One flight + one parallel-session block. Session time is per
        # contact, not per device — that's the whole point of contact
        # events: serve N devices in one shared dwell window.
        time_to_complete = max(0.0, SESSION_TIME + TIME_PER_DIST * dist + noise)
        energy_used = dist * ENERGY_W

        # Per-device completion: each member's RF factor is computed from
        # ITS distance to the contact's stop position (not the mule's
        # original pose), since they share the same dwell window.
        per_device_completed: List[bool] = []
        for did in contact.devices:
            dev = next((d for d in self._devices if d.device_id == did), None)
            if dev is None:
                per_device_completed.append(False)
                continue
            d_dist = float(
                np.sqrt(sum((a - b) ** 2 for a, b in zip(contact.position, dev.pos)))
            )
            rf_factor = max(0.4, 1.0 - d_dist / (3.0 * self._world_radius))
            p_complete = max(0.0, min(1.0, dev.reliability * rf_factor))
            per_device_completed.append(bool(self._rng.random() < p_complete))

        # Reward = sum of completion bonuses across in-range devices,
        # minus the (single) flight + session cost. Big completion-bonus
        # delta favours contacts with multiple reliable members.
        n_completed = sum(per_device_completed)
        reward = (
            COMPLETION_BONUS * n_completed
            - time_to_complete
            - self._energy_weight * energy_used
        )

        # Advance state.
        self._mule_pose = contact.position
        self._mule_energy = max(0.0, self._mule_energy - energy_used)
        self._budget_remaining = max(0.0, self._budget_remaining - time_to_complete)
        # Drop the contact (and its members) from the unvisited set.
        self._contacts.pop(action_idx)
        for did in contact.devices:
            self._states.pop(did, None)
            # Also remove from the device list so future contacts don't
            # accidentally pick them up. (The contacts list is already
            # disjoint from S3a, so this is belt-and-suspenders.)
            self._devices = [d for d in self._devices if d.device_id != did]

        # Roll up metrics.
        self._episode_metrics.contacts_visited += 1
        self._episode_metrics.devices_visited += len(contact.devices)
        self._episode_metrics.devices_completed += n_completed
        self._episode_metrics.energy_total += energy_used
        self._episode_metrics.path_length += dist
        self._episode_metrics.time_total += time_to_complete

        return ContactStepResult(
            contact=contact,
            reward=reward,
            time_to_complete=time_to_complete,
            energy_used=energy_used,
            per_device_completed=per_device_completed,
        )


def distance_policy_choice_contact(
    candidates: Sequence[ContactWaypoint],
    mule_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> int:
    """Per-contact dumb baseline — picks the nearest contact.

    Sprint 1.5 analog of :func:`distance_policy_choice` but operates on
    contact waypoints. Used as the dumb arm in the Chunk G A/B sweep.
    """
    if not candidates:
        raise ValueError("distance_policy_choice_contact: empty candidates")

    def _key(i: int) -> float:
        return float(
            np.sqrt(sum((a - b) ** 2 for a, b in zip(mule_pose, candidates[i].position)))
        )

    return min(range(len(candidates)), key=_key)
