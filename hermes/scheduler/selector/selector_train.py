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

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from hermes.types import Bucket

from .features import extract_features_batch
from .replay import ReplayBuffer, Transition
from .sim_env import BucketSim, distance_policy_choice
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
    # Raw rewards from :class:`BucketSim` are dominated by the
    # completion bonus (+50) and penalty-by-distance (−100+) — so Q
    # targets sit on the ±100 scale while inputs are O(1). Scaling
    # rewards into roughly ±1 before pushing to the replay buffer keeps
    # SGD stable at ``lr=0.05``. Argmax (the only thing inference
    # uses) is scale-invariant, so this doesn't change semantics.
    reward_scale: float = 1.0 / 50.0


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

@dataclass
class ABResult:
    placeholder_mean_reward: float
    placeholder_completion_rate: float
    selector_mean_reward: float
    selector_completion_rate: float

    @property
    def completion_rate_lift(self) -> float:
        base = self.placeholder_completion_rate
        if base <= 0.0:
            return float("inf") if self.selector_completion_rate > 0.0 else 0.0
        return (self.selector_completion_rate - base) / base


def run_ab_evaluation(
    selector: TargetSelectorRL,
    *,
    episodes: int = 200,
    bucket_size: int = 6,
    seed: int = 1,
) -> ABResult:
    """Run both policies on identically-seeded episodes, compare completions."""
    def _rollout(use_selector: bool, seed_: int) -> Tuple[float, float]:
        env = BucketSim(bucket_size=bucket_size, seed=seed_)
        rewards: List[float] = []
        completions: List[int] = []
        for _ in range(episodes):
            env.reset()
            ep_r = 0.0
            ep_c = 0
            ep_total = 0
            while not env.done:
                candidates = env.candidates()
                states = env.device_states()
                if use_selector:
                    feats = extract_features_batch(
                        candidates, states,
                        bucket=Bucket.SCHEDULED_THIS_ROUND,
                        env=env.selector_env(),
                    )
                    action_idx = selector.ddqn.argmax(feats)
                else:
                    action_idx = distance_policy_choice(
                        candidates, states, mule_pose=env.mule_pose
                    )
                step = env.step(action_idx)
                ep_r += step.reward
                ep_c += int(step.completed)
                ep_total += 1
            rewards.append(ep_r / max(1, ep_total))
            completions.append(ep_c / max(1, ep_total))
        return float(np.mean(rewards)), float(np.mean(completions))

    ph_r, ph_c = _rollout(use_selector=False, seed_=seed)
    sel_r, sel_c = _rollout(use_selector=True, seed_=seed)
    return ABResult(
        placeholder_mean_reward=ph_r,
        placeholder_completion_rate=ph_c,
        selector_mean_reward=sel_r,
        selector_completion_rate=sel_c,
    )
