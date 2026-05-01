"""Offline training harness for :class:`TargetSelectorRL`.

Design §2.7 + Implementation Plan §3 Phase 5 task 3::

    Offline CTDE training on AERPAW digital twin.
    Reward: −time_to_complete − w·energy + completed_session_bonus.

We keep the harness framework-agnostic (pure numpy) so it runs in a
pytest fixture, inside a jupyter cell, or in the AERPAW control loop.
The outer loop is the standard DDQN recipe:

    for episode in episodes:
        env.reset()
        while not env.done:
            feats      = extract_features_batch(candidates, ...)
            action_idx = selector picks (ε-greedy over feats)
            step       = env.step(action_idx)           # observe reward
            next_feats = lookahead under online policy  # for bootstrap
            buffer.push(Transition(feats[action_idx], reward, next_feats, done))
            if len(buffer) >= warmup:
                batch = buffer.sample(batch_size)
                ddqn.update(batch)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from hermes.types import Bucket, MissionPass

from .features import extract_features_batch, extract_features_contact_batch
from .replay import ReplayBuffer, Transition
from .sim_env import (
    BucketSim,
    ContactSim,
    distance_policy_choice,
    distance_policy_choice_contact,
)
from .target_selector_rl import TargetSelectorRL


@dataclass
class TrainConfig:
    episodes: int = 400
    bucket_size: int = 6
    batch_size: int = 32
    warmup: int = 64
    buffer_capacity: int = 4_000
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 300
    seed: int = 0
    # Raw rewards from :class:`BucketSim` under the current physics
    # range roughly [-60, +170] per step (session-time + flight-time
    # penalty vs the +200 completion bonus). With Q targets on that
    # scale and inputs O(1), SGD overshoots at lr=0.01. Scaling rewards
    # by ~1/150 lands targets near ±1, matching the input scale and
    # keeping updates stable. Argmax (the only thing inference uses) is
    # scale-invariant, so this doesn't change policy semantics.
    reward_scale: float = 1.0 / 150.0


@dataclass
class TrainMetrics:
    mean_reward_by_episode: List[float] = field(default_factory=list)
    loss_by_step: List[float] = field(default_factory=list)


def _linear_epsilon(ep: int, cfg: TrainConfig) -> float:
    if ep >= cfg.epsilon_decay_episodes:
        return cfg.epsilon_end
    t = ep / max(1, cfg.epsilon_decay_episodes)
    return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * t


def train_selector(
    selector: Optional[TargetSelectorRL] = None,
    cfg: Optional[TrainConfig] = None,
) -> Tuple[TargetSelectorRL, TrainMetrics]:
    """Train the selector on :class:`BucketSim`; return it plus metrics."""
    cfg = cfg or TrainConfig()
    selector = selector if selector is not None else TargetSelectorRL(rng_seed=cfg.seed)
    env = BucketSim(bucket_size=cfg.bucket_size, seed=cfg.seed)
    buf = ReplayBuffer(capacity=cfg.buffer_capacity, seed=cfg.seed)
    metrics = TrainMetrics()
    # One persistent RNG for ε-greedy exploration — respawning per step
    # with near-identical seeds (the previous approach) correlates draws
    # and collapses exploration.
    train_rng = np.random.default_rng(cfg.seed + 7919)

    for ep in range(cfg.episodes):
        selector.set_epsilon(_linear_epsilon(ep, cfg))
        env.reset()
        ep_rewards: List[float] = []

        while not env.done:
            candidates = env.candidates()
            states = env.device_states()
            feats = extract_features_batch(
                candidates, states, bucket=Bucket.SCHEDULED_THIS_ROUND,
                env=env.selector_env(),
            )

            # ε-greedy argmax (persistent RNG).
            if selector.epsilon > 0.0 and train_rng.random() < selector.epsilon:
                action_idx = int(train_rng.integers(0, feats.shape[0]))
            else:
                action_idx = selector.ddqn.argmax(feats)

            chosen_feats = feats[action_idx].copy()
            step = env.step(action_idx)
            ep_rewards.append(step.reward)

            # Next-state bootstrap: greedy-policy features of the remaining bucket.
            if env.done:
                next_feats: Optional[np.ndarray] = None
                done = True
            else:
                next_cands = env.candidates()
                next_states = env.device_states()
                next_batch = extract_features_batch(
                    next_cands, next_states, bucket=Bucket.SCHEDULED_THIS_ROUND,
                    env=env.selector_env(),
                )
                next_idx = selector.ddqn.argmax(next_batch)
                next_feats = next_batch[next_idx].copy()
                done = False

            buf.push(Transition(
                state=chosen_feats,
                reward=float(step.reward) * cfg.reward_scale,
                next_state=next_feats,
                done=done,
            ))

            if len(buf) >= cfg.warmup:
                batch = buf.sample(cfg.batch_size)
                loss = selector.ddqn.update(batch)
                metrics.loss_by_step.append(float(loss))

        metrics.mean_reward_by_episode.append(
            float(np.mean(ep_rewards)) if ep_rewards else 0.0
        )

    # Production inference mode.
    selector.set_epsilon(0.0)
    return selector, metrics


# --------------------------------------------------------------------------- #
# A/B evaluation — Phase 5 DoD check
# --------------------------------------------------------------------------- #

def _safe_lift(baseline: float, treatment: float) -> float:
    """Relative lift (positive = treatment is larger)."""
    if baseline == 0.0:
        return float("inf") if treatment > 0.0 else 0.0
    return (treatment - baseline) / baseline


def _safe_savings(baseline: float, treatment: float) -> float:
    """Relative savings (positive = treatment uses less / has lower)."""
    if baseline == 0.0:
        return 0.0
    return (baseline - treatment) / baseline


@dataclass
class PolicyScore:
    """One policy's mean per-episode metrics across the A/B grid."""

    mean_reward: float = 0.0
    completion_rate: float = 0.0
    energy_per_episode: float = 0.0
    path_length_per_episode: float = 0.0
    retry_rate: float = 0.0
    decisions_per_episode: float = 0.0
    mean_compute_us: float = 0.0  # mean wall-clock per policy invocation


@dataclass
class ABResult:
    """Multi-metric A/B scoreboard for the smart-vs-dumb comparison.

    Aligned with paper §V.C Experiment 3 dependent variables: completion,
    energy (idle + tx + propulsion proxy via path length), retry rate,
    and policy compute cost per decision.
    """

    placeholder: PolicyScore = field(default_factory=PolicyScore)
    selector: PolicyScore = field(default_factory=PolicyScore)

    # ---- backward-compat shims for the old single-metric ABResult ----- #
    # Older tests / callers built ABResult positionally with the four
    # legacy fields. The factory below preserves that surface.

    @classmethod
    def from_legacy(
        cls,
        placeholder_mean_reward: float,
        placeholder_completion_rate: float,
        selector_mean_reward: float,
        selector_completion_rate: float,
    ) -> "ABResult":
        return cls(
            placeholder=PolicyScore(
                mean_reward=placeholder_mean_reward,
                completion_rate=placeholder_completion_rate,
            ),
            selector=PolicyScore(
                mean_reward=selector_mean_reward,
                completion_rate=selector_completion_rate,
            ),
        )

    @property
    def placeholder_mean_reward(self) -> float:
        return self.placeholder.mean_reward

    @property
    def placeholder_completion_rate(self) -> float:
        return self.placeholder.completion_rate

    @property
    def selector_mean_reward(self) -> float:
        return self.selector.mean_reward

    @property
    def selector_completion_rate(self) -> float:
        return self.selector.completion_rate

    # ------------------------- comparison axes ------------------------- #

    @property
    def completion_rate_lift(self) -> float:
        return _safe_lift(self.placeholder.completion_rate, self.selector.completion_rate)

    @property
    def energy_savings(self) -> float:
        return _safe_savings(
            self.placeholder.energy_per_episode, self.selector.energy_per_episode
        )

    @property
    def path_savings(self) -> float:
        return _safe_savings(
            self.placeholder.path_length_per_episode,
            self.selector.path_length_per_episode,
        )

    @property
    def retry_rate_savings(self) -> float:
        return _safe_savings(self.placeholder.retry_rate, self.selector.retry_rate)

    @property
    def compute_overhead_x(self) -> float:
        """Smart compute time / dumb compute time. >1 means smart costs more."""
        if self.placeholder.mean_compute_us <= 0.0:
            return float("inf") if self.selector.mean_compute_us > 0.0 else 1.0
        return self.selector.mean_compute_us / self.placeholder.mean_compute_us

    # --------------------- composite pass criterion -------------------- #

    def passes_dod(
        self,
        *,
        completion_tolerance: float = 0.02,
        win_margin: float = 0.05,
        compute_ceiling_x: float = 50.0,
    ) -> bool:
        """Smart passes the multi-metric DoD if and only if:

        * completion rate is within ``completion_tolerance`` of the dumb
          baseline (e.g. doesn't tank), **and**
        * smart wins by ≥ ``win_margin`` on at least one of
          {energy, retry rate}, **and**
        * smart's per-decision compute cost is ≤ ``compute_ceiling_x``
          times the dumb baseline (caps "win by burning a GPU").
        """
        if self.completion_rate_lift < -completion_tolerance:
            return False
        if self.compute_overhead_x > compute_ceiling_x:
            return False
        return (
            self.energy_savings >= win_margin
            or self.retry_rate_savings >= win_margin
        )


def _time_call(fn, *args, **kwargs) -> Tuple[object, float]:
    """Run ``fn`` and return (result, elapsed_microseconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed_us = (time.perf_counter() - t0) * 1e6
    return out, elapsed_us


def run_ab_evaluation(
    selector: TargetSelectorRL,
    *,
    episodes: int = 200,
    bucket_size: int = 6,
    seed: int = 1,
) -> ABResult:
    """Run both policies on identically-seeded episodes, compare scoreboard.

    Returns an :class:`ABResult` with completion, energy, path length,
    retry rate, decisions per episode, and mean per-decision wall-clock
    for both arms.
    """

    def _rollout(use_selector: bool, seed_: int) -> PolicyScore:
        env = BucketSim(bucket_size=bucket_size, seed=seed_)
        rewards: List[float] = []
        completions: List[float] = []
        energies: List[float] = []
        paths: List[float] = []
        retries: List[float] = []
        decisions_per_ep: List[int] = []
        compute_us_per_call: List[float] = []

        for _ in range(episodes):
            env.reset()
            ep_r = 0.0
            ep_decisions = 0
            while not env.done:
                candidates = env.candidates()
                states = env.device_states()
                if use_selector:
                    feats = extract_features_batch(
                        candidates, states,
                        bucket=Bucket.SCHEDULED_THIS_ROUND,
                        env=env.selector_env(),
                    )
                    action_idx, elapsed_us = _time_call(
                        selector.ddqn.argmax, feats
                    )
                else:
                    action_idx, elapsed_us = _time_call(
                        distance_policy_choice,
                        candidates, states, mule_pose=env.mule_pose,
                    )
                compute_us_per_call.append(elapsed_us)
                step = env.step(int(action_idx))
                ep_r += step.reward
                ep_decisions += 1

            m = env.episode_metrics
            rewards.append(ep_r / max(1, m.visits))
            completions.append(m.completion_rate)
            energies.append(m.energy_total)
            paths.append(m.path_length)
            retries.append(m.retry_rate)
            decisions_per_ep.append(ep_decisions)

        return PolicyScore(
            mean_reward=float(np.mean(rewards)),
            completion_rate=float(np.mean(completions)),
            energy_per_episode=float(np.mean(energies)),
            path_length_per_episode=float(np.mean(paths)),
            retry_rate=float(np.mean(retries)),
            decisions_per_episode=float(np.mean(decisions_per_ep)),
            mean_compute_us=float(np.mean(compute_us_per_call))
            if compute_us_per_call else 0.0,
        )

    ph = _rollout(use_selector=False, seed_=seed)
    sel = _rollout(use_selector=True, seed_=seed)
    return ABResult(placeholder=ph, selector=sel)


# --------------------------------------------------------------------------- #
# Sprint 1.5 — contact-event training + A/B
# --------------------------------------------------------------------------- #

@dataclass
class ContactTrainConfig(TrainConfig):
    """Training config tuned for :class:`ContactSim`.

    Inherits the per-device defaults but adds two contact-specific knobs:
    ``n_devices`` (the slice size) and ``rf_range_m`` (the cluster radius).
    """

    n_devices: int = 8
    rf_range_m: float = 60.0
    mission_budget: float = 200.0


def train_selector_contact(
    selector: Optional[TargetSelectorRL] = None,
    cfg: Optional[ContactTrainConfig] = None,
) -> Tuple[TargetSelectorRL, TrainMetrics]:
    """Train the per-contact selector against :class:`ContactSim`.

    Sprint 1.5 path: identical SGD recipe to :func:`train_selector` but
    swaps the per-device sim for ``ContactSim`` and the per-device
    feature extractor for the contact-aware one. The trained selector
    is then ready for ``select_contact`` / ``rank_contacts`` invocations.
    """
    cfg = cfg or ContactTrainConfig()
    selector = selector if selector is not None else TargetSelectorRL(rng_seed=cfg.seed)
    env = ContactSim(
        n_devices=cfg.n_devices,
        rf_range_m=cfg.rf_range_m,
        mission_budget=cfg.mission_budget,
        seed=cfg.seed,
    )
    buf = ReplayBuffer(capacity=cfg.buffer_capacity, seed=cfg.seed)
    metrics = TrainMetrics()
    train_rng = np.random.default_rng(cfg.seed + 7919)

    for ep in range(cfg.episodes):
        selector.set_epsilon(_linear_epsilon(ep, cfg))
        env.reset()
        ep_rewards: List[float] = []

        while not env.done:
            candidates = env.candidates()
            states = env.device_states()
            feats = extract_features_contact_batch(
                candidates, states, env=env.selector_env(),
            )

            if selector.epsilon > 0.0 and train_rng.random() < selector.epsilon:
                action_idx = int(train_rng.integers(0, feats.shape[0]))
            else:
                action_idx = selector.ddqn.argmax(feats)

            chosen_feats = feats[action_idx].copy()
            step = env.step(action_idx)
            ep_rewards.append(step.reward)

            if env.done:
                next_feats: Optional[np.ndarray] = None
                done = True
            else:
                next_cands = env.candidates()
                next_states = env.device_states()
                next_batch = extract_features_contact_batch(
                    next_cands, next_states, env=env.selector_env(),
                )
                next_idx = selector.ddqn.argmax(next_batch)
                next_feats = next_batch[next_idx].copy()
                done = False

            buf.push(Transition(
                state=chosen_feats,
                reward=float(step.reward) * cfg.reward_scale,
                next_state=next_feats,
                done=done,
            ))

            if len(buf) >= cfg.warmup:
                batch = buf.sample(cfg.batch_size)
                loss = selector.ddqn.update(batch)
                metrics.loss_by_step.append(float(loss))

        metrics.mean_reward_by_episode.append(
            float(np.mean(ep_rewards)) if ep_rewards else 0.0
        )

    selector.set_epsilon(0.0)
    return selector, metrics


@dataclass
class ContactPolicyScore(PolicyScore):
    """Per-arm contact-event metrics — adds ρ_contact and per-mission devices."""

    rho_contact: float = 0.0
    devices_per_episode: float = 0.0
    contacts_per_episode: float = 0.0


@dataclass
class ContactABResult:
    """A/B scoreboard for the contact-event sim.

    Re-uses the same multi-axis pass criterion as :class:`ABResult` so
    the Phase-5 DoD bar applies cleanly. ``rho_contact`` is logged
    per arm but not part of the pass criterion (it's a sim diagnostic
    that should be ~equal across arms at a given ``rf_range_m``).
    """

    placeholder: ContactPolicyScore = field(default_factory=ContactPolicyScore)
    selector: ContactPolicyScore = field(default_factory=ContactPolicyScore)
    rf_range_m: float = 60.0

    @property
    def completion_rate_lift(self) -> float:
        return _safe_lift(self.placeholder.completion_rate, self.selector.completion_rate)

    @property
    def energy_savings(self) -> float:
        return _safe_savings(
            self.placeholder.energy_per_episode, self.selector.energy_per_episode
        )

    @property
    def retry_rate_savings(self) -> float:
        return _safe_savings(self.placeholder.retry_rate, self.selector.retry_rate)

    @property
    def compute_overhead_x(self) -> float:
        if self.placeholder.mean_compute_us <= 0.0:
            return float("inf") if self.selector.mean_compute_us > 0.0 else 1.0
        return self.selector.mean_compute_us / self.placeholder.mean_compute_us

    def passes_dod(
        self,
        *,
        completion_tolerance: float = 0.02,
        win_margin: float = 0.05,
        compute_ceiling_x: float = 50.0,
    ) -> bool:
        """Same criterion as :meth:`ABResult.passes_dod`."""
        if self.completion_rate_lift < -completion_tolerance:
            return False
        if self.compute_overhead_x > compute_ceiling_x:
            return False
        return (
            self.energy_savings >= win_margin
            or self.retry_rate_savings >= win_margin
        )


def run_ab_evaluation_contact(
    selector: TargetSelectorRL,
    *,
    episodes: int = 200,
    n_devices: int = 8,
    rf_range_m: float = 60.0,
    mission_budget: float = 200.0,
    seed: int = 1,
) -> ContactABResult:
    """Run smart-vs-dumb on :class:`ContactSim` at a given ``rf_range_m``.

    Both arms see identically-seeded episodes. Returns a
    :class:`ContactABResult` with completion / energy / retry /
    ρ_contact / compute for both policies.
    """

    def _rollout(use_selector: bool, seed_: int) -> ContactPolicyScore:
        env = ContactSim(
            n_devices=n_devices,
            rf_range_m=rf_range_m,
            mission_budget=mission_budget,
            seed=seed_,
        )
        rewards: List[float] = []
        completions: List[float] = []
        energies: List[float] = []
        paths: List[float] = []
        retries: List[float] = []
        contacts_per_ep: List[int] = []
        devices_per_ep: List[int] = []
        rho_per_ep: List[float] = []
        compute_us_per_call: List[float] = []

        for _ in range(episodes):
            env.reset()
            ep_r = 0.0
            ep_decisions = 0
            while not env.done:
                candidates = env.candidates()
                if not candidates:
                    break
                states = env.device_states()
                if use_selector:
                    feats = extract_features_contact_batch(
                        candidates, states, env=env.selector_env(),
                    )
                    action_idx, elapsed_us = _time_call(
                        selector.ddqn.argmax, feats,
                    )
                else:
                    action_idx, elapsed_us = _time_call(
                        distance_policy_choice_contact,
                        candidates, mule_pose=env.mule_pose,
                    )
                compute_us_per_call.append(elapsed_us)
                step = env.step(int(action_idx))
                ep_r += step.reward
                ep_decisions += 1

            m = env.episode_metrics
            rewards.append(ep_r / max(1, m.contacts_visited))
            completions.append(m.completion_rate)
            energies.append(m.energy_total)
            paths.append(m.path_length)
            retries.append(m.retry_rate)
            contacts_per_ep.append(m.contacts_visited)
            devices_per_ep.append(m.devices_visited)
            rho_per_ep.append(m.rho_contact)

        return ContactPolicyScore(
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            completion_rate=float(np.mean(completions)) if completions else 0.0,
            energy_per_episode=float(np.mean(energies)) if energies else 0.0,
            path_length_per_episode=float(np.mean(paths)) if paths else 0.0,
            retry_rate=float(np.mean(retries)) if retries else 0.0,
            decisions_per_episode=float(np.mean(contacts_per_ep)) if contacts_per_ep else 0.0,
            mean_compute_us=float(np.mean(compute_us_per_call)) if compute_us_per_call else 0.0,
            rho_contact=float(np.mean(rho_per_ep)) if rho_per_ep else 0.0,
            devices_per_episode=float(np.mean(devices_per_ep)) if devices_per_ep else 0.0,
            contacts_per_episode=float(np.mean(contacts_per_ep)) if contacts_per_ep else 0.0,
        )

    ph = _rollout(use_selector=False, seed_=seed)
    sel = _rollout(use_selector=True, seed_=seed)
    return ContactABResult(placeholder=ph, selector=sel, rf_range_m=rf_range_m)
