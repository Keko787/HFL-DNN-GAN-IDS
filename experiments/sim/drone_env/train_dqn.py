"""
Hybrid Heuristic + DQN trainer for DroneDataRelayEnv
=====================================================
Heuristic handles navigation (which waypoint to visit next).
DQN handles the complex part: which BS + channel to use for upload,
exploiting the time-varying sinusoidal channel structure.

This reduces the RL action space from 45 -> 9 and fills the replay
buffer with high-quality heuristic trajectories from episode 1.

Vendored from github.com/FyneappleJuice/hermes_rl at commit a8a453f; see
README.md in this directory. Changes from that commit: package imports, the
scenario built by ``drone_env.overloaded_config`` (``make_env_config`` returns
it), and the plot written under ``results/drone_env/`` by default. Needs torch.

Usage (from the repository root):
    python -m experiments.sim.drone_env.train_dqn                   # hybrid mode (default)
    python -m experiments.sim.drone_env.train_dqn --mode dqn        # standalone DQN (for comparison)
    python -m experiments.sim.drone_env.train_dqn --episodes 1500
"""

import argparse
import collections
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .drone_env import DroneDataRelayEnv, EnvConfig, overloaded_config


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    mode            = "hybrid",      # "hybrid" or "dqn"
    episodes        = 1500,
    lr              = 5e-5,          # lower: stops good policy being overwritten
    gamma           = 0.99,
    batch_size      = 64,
    buffer_size     = 50_000,        # larger buffer = more diverse samples
    target_update   = 1000,          # less frequent = more stable targets
    eps_start       = 1.0,
    eps_end         = 0.05,
    eps_decay_ep    = 1000,          # slower decay = more exploration time
    hidden          = 128,
    seed            = 42,
    eval_every      = 50,
    eval_episodes   = 20,            # more episodes = lower variance eval signal
    plot_path       = "results/drone_env/dqn_results.png",
)


# ---------------------------------------------------------------------------
# Heuristic agent (ported from drone_demo.py — no pygame dependency)
# ---------------------------------------------------------------------------

class HeuristicAgent:
    """Priority-based greedy agent: collect most urgent job, then upload."""

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg

    def _dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _arrival_time(self, from_wp, to_wp):
        return self.env.t + self.env._transit_time(from_wp, to_wp)

    def _rate(self, bs_idx, ch_idx, wp_idx, t):
        wp_pos = self.cfg.waypoints[wp_idx]
        bs     = self.cfg.base_stations[bs_idx]
        d      = self._dist(wp_pos, bs.position)
        return 0.0 if d > bs.radius else self.env._upload_rate(bs_idx, ch_idx, d, t)

    def _best_channel(self, bs_idx, wp_idx, t):
        best_ch, best_r = 0, -1.0
        for ci in range(self.env.n_channels):
            r = self._rate(bs_idx, ci, wp_idx, t)
            if r > best_r:
                best_r = r
                best_ch = ci
        return best_ch, best_r

    def _best_upload_action(self):
        env = self.env
        cfg = self.cfg
        best_act, best_score = None, -1.0
        for wi in range(env.n_waypoints):
            wp_pos = cfg.waypoints[wi]
            arr_t  = self._arrival_time(env.current_wp, wi)
            travel = env._transit_time(env.current_wp, wi)
            for bi in range(env.n_bs):
                d = self._dist(wp_pos, cfg.base_stations[bi].position)
                if d > cfg.base_stations[bi].radius:
                    continue
                ch, r = self._best_channel(bi, wi, arr_t)
                if r <= 0:
                    continue
                score = r / (1.0 + 0.05 * travel)
                if score > best_score:
                    best_score = score
                    best_act   = wi * env.n_bs * env.n_channels + bi * env.n_channels + ch
        return best_act if best_act is not None else (
            1 * env.n_bs * env.n_channels + 1 * env.n_channels
        )

    def _collect_action(self, ji):
        env = self.env
        cfg = self.cfg
        sen = cfg.sensors[cfg.jobs[ji].sensor_idx]

        def wp_score(w):
            d = self._dist(cfg.waypoints[w], sen.position)
            return (0 if d <= sen.radius else 1, d)

        best_wp = min(range(env.n_waypoints), key=wp_score)
        arr_t   = self._arrival_time(env.current_wp, best_wp)
        best_bs, best_ch, best_r = 0, 0, -1.0
        for bi in range(env.n_bs):
            ch, r = self._best_channel(bi, best_wp, arr_t)
            if r > best_r:
                best_r = r
                best_bs = bi
                best_ch = ch
        return best_wp * env.n_bs * env.n_channels + best_bs * env.n_channels + best_ch

    def act(self):
        env = self.env
        cfg = self.cfg
        if env.is_moving:
            return (env.target_wp * env.n_bs * env.n_channels
                    + env.active_bs * env.n_channels + env.active_ch)

        pending = [
            ji for ji in range(env.n_jobs)
            if env.job_collected[ji] and not env.job_done[ji] and not env.job_failed[ji]
        ]
        uncollected = [
            ji for ji in range(env.n_jobs)
            if not env.job_collected[ji] and not env.job_done[ji] and not env.job_failed[ji]
        ]

        if pending:
            wp_pos = cfg.waypoints[env.current_wp]
            cur_best_r, cur_best_bs, cur_best_ch = 0.0, 0, 0
            for bi in range(env.n_bs):
                d = self._dist(wp_pos, cfg.base_stations[bi].position)
                if d <= cfg.base_stations[bi].radius:
                    ch, r = self._best_channel(bi, env.current_wp, env.t)
                    if r > cur_best_r:
                        cur_best_r = r
                        cur_best_bs = bi
                        cur_best_ch = ch
            if cur_best_r > 0:
                return (env.current_wp * env.n_bs * env.n_channels
                        + cur_best_bs * env.n_channels + cur_best_ch)
            return self._best_upload_action()

        if uncollected:
            ji = min(uncollected, key=lambda j: env.job_deadlines[j])
            return self._collect_action(ji)

        return env.current_wp * env.n_bs * env.n_channels


# ---------------------------------------------------------------------------
# Reward shaping wrapper
# ---------------------------------------------------------------------------

_FIELD_DIAG = math.hypot(100.0, 100.0)


class ShapedEnv:
    """Dense auxiliary rewards for training only (not used in eval)."""

    def __init__(self, env, collect_scale=1.0, proximity_scale=0.5):
        self.env  = env
        self.cs   = collect_scale
        self.ps   = proximity_scale
        self._prev_phi = 0.0

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def _phi(self):
        env = self.env
        cfg = env.cfg
        wp_idx = env.target_wp if env.is_moving else env.current_wp
        wp_pos = cfg.waypoints[wp_idx]

        pending = [j for j in range(env.n_jobs)
                   if env.job_collected[j] and not env.job_done[j] and not env.job_failed[j]]
        if pending:
            min_d = min(float(np.linalg.norm(wp_pos - bs.position))
                        for bs in cfg.base_stations)
            return 1.0 - min_d / _FIELD_DIAG

        uncollected = [j for j in range(env.n_jobs)
                       if not env.job_collected[j]
                       and not env.job_done[j] and not env.job_failed[j]]
        if uncollected:
            urgent = min(uncollected, key=lambda j: env.job_deadlines[j])
            s_pos  = cfg.sensors[cfg.jobs[urgent].sensor_idx].position
            d      = float(np.linalg.norm(wp_pos - s_pos))
            return 0.5 * (1.0 - d / _FIELD_DIAG)

        return 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        env = self.env
        if not env.is_moving and env.active_sensor >= 0:
            reward += self.cs * 0.05
        new_phi        = self._phi()
        reward        += self.ps * (0.99 * new_phi - self._prev_phi)
        self._prev_phi = new_phi
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Augmented observation wrapper
# ---------------------------------------------------------------------------

class AugmentedEnv:
    """
    Appends a per-job feasibility score to every observation.

    feasibility[j] = (deadline_remaining - steps_needed) / max_steps
      > 0  → job is still achievable
      < 0  → job will almost certainly be missed
      -1   → job is already done or failed

    Giving this directly to the Q-network means it never has to learn
    the relationship between deadline_remaining and data_remaining from
    sparse rewards — it can just read "negative feasibility = skip this job."
    """

    _DOWNLOAD_RATE = 0.05   # MB/step (hardcoded in drone_env)
    _AVG_TRANSIT   = 60     # conservative estimate of transit steps
    _AVG_UPLOAD    = 10     # rough upload steps for 2-4 MB at ~0.4 MB/step

    def __init__(self, env):
        self.env = env
        orig_dim = env.observation_space.shape[0]

        class _Space:
            def __init__(self, shape):
                self.shape = shape
        self.observation_space = _Space((orig_dim + env.n_jobs,))

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def _feasibility(self):
        env = self.env
        scores = []
        for j in range(env.n_jobs):
            if env.job_done[j] or env.job_failed[j]:
                scores.append(-1.0)
                continue
            dl_left = float(env.job_deadlines[j] - env.t)
            if env.job_collected[j]:
                needed = self._AVG_TRANSIT + self._AVG_UPLOAD
            else:
                needed = (float(env.job_data_remaining[j]) / self._DOWNLOAD_RATE
                          + self._AVG_TRANSIT * 2 + self._AVG_UPLOAD)
            scores.append(float(np.clip((dl_left - needed) / env.cfg.max_steps, -1.0, 1.0)))
        return np.array(scores, dtype=np.float32)

    def _aug(self, obs):
        return np.concatenate([obs, self._feasibility()])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._aug(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return self._aug(obs), r, term, trunc, info


def feasibility_job(env, n_jobs):
    """
    Exploration policy: pick the most feasible active job
    (most deadline margin), breaking ties by urgency.
    This teaches the Q-network to skip dying jobs — the exact
    behaviour the heuristic cannot do.
    """
    _DOWNLOAD_RATE = 0.05
    _AVG_TRANSIT   = 60
    _AVG_UPLOAD    = 10

    def margin(j):
        if env.job_done[j] or env.job_failed[j]:
            return -9999.0
        dl_left = float(env.job_deadlines[j] - env.t)
        if env.job_collected[j]:
            needed = _AVG_TRANSIT + _AVG_UPLOAD
        else:
            needed = (float(env.job_data_remaining[j]) / _DOWNLOAD_RATE
                      + _AVG_TRANSIT * 2 + _AVG_UPLOAD)
        return dl_left - needed

    active   = [j for j in range(n_jobs) if not env.job_done[j] and not env.job_failed[j]]
    if not active:
        return random.randrange(n_jobs)

    margins  = [(j, margin(j)) for j in active]
    feasible = [(j, m) for j, m in margins if m > 0]

    if feasible:
        # Most urgent among the jobs that can still be done
        return min(feasible, key=lambda x: env.job_deadlines[x[0]])[0]
    # All infeasible — pick the least bad
    return max(margins, key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

Transition = collections.namedtuple(
    "Transition", ("obs", "action", "reward", "next_obs", "done")
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, n):
        return Transition(*zip(*random.sample(self.buf, n)))

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, obs_dim, n_actions, cfg):
        self.n_actions     = n_actions
        self.gamma         = cfg.gamma
        self.batch         = cfg.batch_size
        self.target_update = cfg.target_update
        self.steps         = 0

        self.online = QNet(obs_dim, n_actions, cfg.hidden)
        self.target = QNet(obs_dim, n_actions, cfg.hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.buffer_size)

    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(obs_t).argmax(1).item())

    def push(self, obs, action, reward, next_obs, done):
        self.buf.push(obs, action, reward, next_obs, done)

    def update(self):
        if len(self.buf) < self.batch:
            return

        batch    = self.buf.sample(self.batch)
        obs      = torch.tensor(np.array(batch.obs),      dtype=torch.float32)
        actions  = torch.tensor(batch.action,              dtype=torch.long).unsqueeze(1)
        rewards  = torch.tensor(batch.reward,              dtype=torch.float32).unsqueeze(1)
        next_obs = torch.tensor(np.array(batch.next_obs), dtype=torch.float32)
        dones    = torch.tensor(batch.done,                dtype=torch.float32).unsqueeze(1)

        q_vals = self.online(obs).gather(1, actions)

        with torch.no_grad():
            next_a   = self.online(next_obs).argmax(1, keepdim=True)
            next_q   = self.target(next_obs).gather(1, next_a)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_vals, target_q)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(env, action_fn, n_episodes, seed_offset=0):
    returns, done_counts, failed_counts = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_ret = 0.0
        while True:
            action = action_fn(obs)
            obs, r, terminated, truncated, info = env.step(action)
            ep_ret += r
            if terminated or truncated:
                break
        returns.append(ep_ret)
        done_counts.append(sum(info["job_done"]))
        failed_counts.append(sum(info["job_failed"]))
    return np.mean(returns), np.mean(done_counts), np.mean(failed_counts)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(train_rets, eval_x, hybrid_eval, heur_ret, heur_done, heur_fail,
                 n_jobs, path, mode_label):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"{mode_label} vs Heuristic — DroneDataRelayEnv",
                 fontsize=13, fontweight="bold")

    window   = min(50, len(train_rets))
    smoothed = np.convolve(train_rets, np.ones(window) / window, mode="valid")

    ax = axes[0]
    ax.plot(train_rets, alpha=0.2, color="steelblue", linewidth=0.7, label="raw")
    ax.plot(range(window - 1, len(train_rets)), smoothed,
            color="steelblue", linewidth=2, label=f"{window}-ep avg")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Shaped return")
    ax.set_title("Training return (shaped env)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    rl_r, rl_d, rl_f = zip(*hybrid_eval)
    ax = axes[1]
    ax.plot(eval_x, rl_r, marker="o", color="steelblue", linewidth=2, label=mode_label)
    ax.axhline(heur_ret, color="darkorange", linewidth=2, linestyle="--", label="Heuristic")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Mean return (raw env)")
    ax.set_title("Evaluation return"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(eval_x, rl_d, marker="o", color="seagreen",  linewidth=2, label=f"{mode_label} done")
    ax.plot(eval_x, rl_f, marker="s", color="tomato",    linewidth=2, label=f"{mode_label} failed")
    ax.axhline(heur_done, color="seagreen", linewidth=2, linestyle="--", label="Heuristic done")
    ax.axhline(heur_fail, color="tomato",   linewidth=2, linestyle=":",  label="Heuristic failed")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Mean jobs / episode")
    ax.set_title("Jobs done vs failed"); ax.legend(fontsize=8)
    ax.set_ylim(-0.1, n_jobs + 0.3); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130)
    print(f"\nPlot saved -> {path}")


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

def make_env_config():
    """The overloaded 5-job scenario; see :func:`drone_env.overloaded_config`,
    where it now lives so that it can be built without torch."""
    return overloaded_config()


# ---------------------------------------------------------------------------
# Heuristic job executor — runs heuristic logic but locked to one job
# ---------------------------------------------------------------------------

def execute_for_job(env, heuristic, job_idx):
    """
    Run heuristic execution focused on a specific job.
    Falls back to default heuristic if the chosen job is done/failed,
    or if the drone is mid-transit (can't interrupt).
    """
    if env.is_moving:
        return heuristic.act()
    if env.job_done[job_idx] or env.job_failed[job_idx]:
        return heuristic.act()
    if env.job_collected[job_idx]:
        return heuristic._best_upload_action()
    return heuristic._collect_action(job_idx)


# ---------------------------------------------------------------------------
# Training — Hybrid mode
# RL picks WHICH JOB to focus on (5 actions).
# Heuristic handles all execution: waypoint, BS, channel.
#
# Why this works: the heuristic fails because it blindly chases the earliest
# deadline even when a job is no longer achievable.  RL can learn to skip
# hopeless jobs and finish more of the feasible ones.
# Action space: 5  (one per job)  vs 63 for full DQN  ->  stable learning.
# ---------------------------------------------------------------------------

def train_hybrid(cfg):
    import copy

    raw_env   = DroneDataRelayEnv(config=make_env_config())
    aug_env   = AugmentedEnv(raw_env)   # feasibility scores appended to obs
    obs_dim   = aug_env.observation_space.shape[0]   # 21 + 5 = 26
    n_jobs    = raw_env.n_jobs

    agent     = DQNAgent(obs_dim, n_jobs, cfg)
    heuristic = HeuristicAgent(raw_env)

    print("Mode: HYBRID  (RL job selector + feasibility obs + heuristic execution)")
    print(f"RL action space: {n_jobs} jobs  |  Obs dim: {obs_dim}  (21 raw + {n_jobs} feasibility)\n")

    print("Running heuristic baseline...")
    heur_ret, heur_done, heur_fail = run_episodes(
        aug_env, lambda _: heuristic.act(), cfg.eval_episodes, seed_offset=5000
    )
    print(f"  Heuristic -> return={heur_ret:.2f}  "
          f"done={heur_done:.1f}/{n_jobs}  failed={heur_fail:.1f}/{n_jobs}\n")

    def hybrid_act(obs):
        job_idx = agent.select_action(obs, 0.0)
        return execute_for_job(raw_env, heuristic, job_idx)

    train_returns, eval_results, eval_x = [], [], []
    best_eval_ret = -np.inf
    best_weights  = None

    header = (f"{'Episode':>8}  {'Ret(50avg)':>10}  {'Eps':>6}  "
              f"{'EvalRet':>9}  {'Done':>5}  {'Failed':>6}  {'vs Heur':>8}  {'Best':>7}")
    print(header)
    print("-" * len(header))

    t0 = time.time()

    for ep in range(1, cfg.episodes + 1):
        eps = max(cfg.eps_end,
                  cfg.eps_start - (cfg.eps_start - cfg.eps_end) * ep / cfg.eps_decay_ep)

        obs, _ = aug_env.reset(seed=ep)
        ep_ret = 0.0
        while True:
            # Feasibility-guided exploration: teaches "skip dying jobs"
            if random.random() < eps:
                job_idx = feasibility_job(raw_env, n_jobs)
            else:
                job_idx = agent.select_action(obs, 0.0)

            action             = execute_for_job(raw_env, heuristic, job_idx)
            next_obs, r, terminated, truncated, _ = aug_env.step(action)
            agent.push(obs, job_idx, r, next_obs, float(terminated or truncated))
            agent.update()
            ep_ret += r
            obs    = next_obs
            if terminated or truncated:
                break
        train_returns.append(ep_ret)

        if ep % cfg.eval_every == 0 or ep == cfg.episodes:
            eval_ret, eval_done, eval_fail = run_episodes(
                aug_env, lambda obs: hybrid_act(obs),
                cfg.eval_episodes, seed_offset=2000
            )
            eval_results.append((eval_ret, eval_done, eval_fail))
            eval_x.append(ep)

            is_best = eval_ret > best_eval_ret
            if is_best:
                best_eval_ret = eval_ret
                best_weights  = copy.deepcopy(agent.online.state_dict())

            recent  = np.mean(train_returns[-min(50, len(train_returns)):])
            elapsed = time.time() - t0
            marker  = " <--" if is_best else ""
            print(f"{ep:>8}  {recent:>10.2f}  {eps:>6.3f}  "
                  f"{eval_ret:>9.2f}  {eval_done:>5.1f}  {eval_fail:>6.1f}  "
                  f"{eval_ret - heur_ret:>+8.2f}  {best_eval_ret:>7.2f}  [{elapsed:.0f}s]{marker}")

    print(f"\nTotal training time: {time.time()-t0:.1f}s")

    if best_weights is not None:
        agent.online.load_state_dict(best_weights)
        print(f"Restored best checkpoint (eval return = {best_eval_ret:.2f})")

    final_ret,  final_done,  final_fail  = run_episodes(
        aug_env, lambda obs: hybrid_act(obs), 50, seed_offset=9000
    )
    final_hret, final_hdone, final_hfail = run_episodes(
        aug_env, lambda _: heuristic.act(), 50, seed_offset=9000
    )
    print("\n" + "=" * 60)
    print("Final greedy evaluation (50 episodes, same seeds)")
    print("=" * 60)
    print(f"  {'':26s}  {'Return':>9}  {'Done':>6}  {'Failed':>7}")
    print(f"  {'Hybrid (RL job + Heur exec)':26s}  {final_ret:>9.2f}  {final_done:>6.1f}  {final_fail:>7.1f}")
    print(f"  {'Heuristic alone':26s}  {final_hret:>9.2f}  {final_hdone:>6.1f}  {final_hfail:>7.1f}")
    print(f"  {'Delta':26s}  {final_ret-final_hret:>+9.2f}  "
          f"{final_done-final_hdone:>+6.1f}  {final_fail-final_hfail:>+7.1f}")
    print("=" * 60)

    plot_results(train_returns, eval_x, eval_results,
                 heur_ret, heur_done, heur_fail, n_jobs, cfg.plot_path, "Hybrid")
    return agent


# ---------------------------------------------------------------------------
# Training — Standalone DQN mode (45 actions, for comparison)
# ---------------------------------------------------------------------------

def train_dqn(cfg):
    raw_env   = DroneDataRelayEnv(config=make_env_config())
    train_env = ShapedEnv(raw_env)

    obs_dim = raw_env.observation_space.shape[0]
    n_act   = raw_env.action_space.n   # 45
    n_jobs  = raw_env.n_jobs

    agent     = DQNAgent(obs_dim, n_act, cfg)
    heuristic = HeuristicAgent(raw_env)

    print("Mode: STANDALONE DQN  (45 actions)\n")

    print("Running heuristic baseline...")
    heur_ret, heur_done, heur_fail = run_episodes(
        raw_env, lambda _: heuristic.act(), cfg.eval_episodes, seed_offset=5000
    )
    print(f"  Heuristic -> return={heur_ret:.2f}  "
          f"done={heur_done:.1f}/{n_jobs}  failed={heur_fail:.1f}/{n_jobs}\n")

    train_returns, eval_results, eval_x = [], [], []

    header = (f"{'Episode':>8}  {'ShapedRet(20avg)':>16}  {'Eps':>6}  "
              f"{'EvalRet':>9}  {'Done':>5}  {'Failed':>6}  {'vs Heur':>8}")
    print(header)
    print("-" * len(header))

    t0 = time.time()

    for ep in range(1, cfg.episodes + 1):
        eps = max(cfg.eps_end,
                  cfg.eps_start - (cfg.eps_start - cfg.eps_end) * ep / cfg.eps_decay_ep)

        obs, _ = train_env.reset(seed=ep)
        ep_ret = 0.0
        while True:
            action   = agent.select_action(obs, eps)
            next_obs, r, terminated, truncated, _ = train_env.step(action)
            agent.push(obs, action, r, next_obs, float(terminated or truncated))
            agent.update()
            ep_ret += r
            obs     = next_obs
            if terminated or truncated:
                break
        train_returns.append(ep_ret)

        if ep % cfg.eval_every == 0 or ep == cfg.episodes:
            eval_ret, eval_done, eval_fail = run_episodes(
                raw_env, lambda obs: agent.select_action(obs, 0.0),
                cfg.eval_episodes, seed_offset=2000
            )
            eval_results.append((eval_ret, eval_done, eval_fail))
            eval_x.append(ep)

            recent  = np.mean(train_returns[-cfg.eval_every:])
            elapsed = time.time() - t0
            print(f"{ep:>8}  {recent:>16.2f}  {eps:>6.3f}  "
                  f"{eval_ret:>9.2f}  {eval_done:>5.1f}  {eval_fail:>6.1f}  "
                  f"{eval_ret - heur_ret:>+8.2f}   [{elapsed:.0f}s]")

    print(f"\nTotal training time: {time.time()-t0:.1f}s")

    final_ret, final_done, final_fail = run_episodes(
        raw_env, lambda obs: agent.select_action(obs, 0.0), 50, seed_offset=9000
    )
    print("\n" + "=" * 60)
    print("Final greedy evaluation on raw env (50 episodes)")
    print("=" * 60)
    print(f"  {'':26s}  {'Return':>9}  {'Done':>6}  {'Failed':>7}")
    print(f"  {'Standalone DQN':26s}  {final_ret:>9.2f}  {final_done:>6.1f}  {final_fail:>7.1f}")
    print(f"  {'Heuristic alone':26s}  {heur_ret:>9.2f}  {heur_done:>6.1f}  {heur_fail:>7.1f}")
    print(f"  {'Delta':26s}  {final_ret-heur_ret:>+9.2f}  "
          f"{final_done-heur_done:>+6.1f}  {final_fail-heur_fail:>+7.1f}")
    print("=" * 60)

    plot_results(train_returns, eval_x, eval_results,
                 heur_ret, heur_done, heur_fail, n_jobs, cfg.plot_path, "Standalone DQN")
    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    class Cfg:
        pass
    cfg = Cfg()
    for k, v in vars(args).items():
        setattr(cfg, k, v)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.mode == "hybrid":
        train_hybrid(cfg)
    else:
        train_dqn(cfg)