"""
Drone Data Relay RL Environment
================================
Vendored from github.com/FyneappleJuice/hermes_rl at commit a8a453f
(2026-04-28); see README.md in this directory. With default settings it
behaves exactly as that commit does. Two additions: opt-in seeded episode
variation (:class:`ScenarioRandomization`, off by default) and
:func:`overloaded_config`, the training scenario moved here from
``train_dqn.make_env_config`` so it builds without torch.

A Gymnasium-compatible environment where a drone collects data from ground
sensors and uploads it to base stations via time-varying sinusoidal channels.

Key rules:
 - Drone moves between 5 fixed waypoints. Travel takes time and NO data
   transfer (upload or collection) happens while in transit.
 - Sensors are in the bottom-left. Each has a collection radius.
   When the drone is AT a waypoint inside that radius it instantly collects.
 - 3 base stations at the top, each with 3 channels.
   Upload rate: R_k(d,t) = alpha*sin(omega*t + phi_k) + beta/(1 + gamma*d)
   with alpha > beta so sinusoid dominates.
 - Each job has a total data size and a deadline (in timesteps).
 - The agent picks: next waypoint + which BS + which channel.
   It executes the move (silent transit), then uploads/collects at the waypoint.
"""

import copy

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Minimal spaces shim (drop-in if gymnasium is unavailable)
# ---------------------------------------------------------------------------

class _Discrete:
    def __init__(self, n, seed=None):
        self.n = n
        self._rng = np.random.default_rng(seed)
    def sample(self):
        return int(self._rng.integers(0, self.n))

class _Box:
    def __init__(self, low, high, dtype=np.float32):
        self.low = low; self.high = high; self.dtype = dtype
        self.shape = low.shape

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GymBase = gym.Env
except ImportError:
    try:
        import gym
        from gym import spaces
        _GymBase = gym.Env
    except ImportError:
        class spaces:
            Discrete = _Discrete
            Box = _Box
        class _GymBase:
            def reset(self, *, seed=None, options=None):
                if seed is not None:
                    np.random.seed(seed)
            def step(self, action): pass
        _GymBase = _GymBase


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SensorConfig:
    position: np.ndarray          # (x, y)
    radius: float                 # collection radius
    job_idx: int                  # which job this sensor carries


@dataclass
class BaseStationConfig:
    position: np.ndarray          # (x, y)
    radius: float                 # upload radius
    n_channels: int = 3
    phase_offsets: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 2.094, 4.189])  # 0, 2pi/3, 4pi/3
    )


@dataclass
class JobConfig:
    total_data: float             # total MB to upload
    deadline: int                 # timesteps from episode start
    sensor_idx: int               # which sensor holds this data


@dataclass
class ScenarioRandomization:
    """Opt-in, seeded variation of each episode's world (added when vendored).

    Drawn at every ``reset(seed=...)`` from a generator seeded with that seed,
    so one seed gives one world: paired seeds across arms see the same
    positions, phases, deadlines and noise. A ``reset`` without a seed
    continues the generator, as Gymnasium's does. With every field at its
    default nothing varies and the environment is the a8a453f original.

    * ``position_jitter`` — waypoints and sensors move by U(−j, j) per axis,
      clipped to the plane. Base stations stay put. A large jitter can take a
      sensor out of every waypoint's reach.
    * ``random_phases`` — every (base station, channel) phase drawn from
      U(0, 2π) instead of 0, 2π/3, 4π/3.
    * ``deadline_jitter`` — each job's deadline scaled by U(1 − j, 1 + j),
      rounded, at least 1.
    * ``rate_noise_std`` — Gaussian noise on the upload rate, drawn once per
      episode as a table over (timestep, base station, channel). The noise is
      a function of time, not of how often rates are queried, so an agent that
      peeks at ``channel_rates_at`` sees the same world as one that does not.
      The table covers timesteps up to ``2·max_steps``; later ones are
      noiseless.
    """

    position_jitter: float = 0.0
    random_phases: bool = False
    deadline_jitter: float = 0.0
    rate_noise_std: float = 0.0

    def __post_init__(self):
        if self.position_jitter < 0 or self.rate_noise_std < 0:
            raise ValueError("position_jitter and rate_noise_std must be non-negative")
        if not 0 <= self.deadline_jitter < 1:
            raise ValueError("deadline_jitter must be in [0, 1)")

    @property
    def active(self) -> bool:
        return bool(
            self.position_jitter or self.random_phases
            or self.deadline_jitter or self.rate_noise_std
        )


@dataclass
class EnvConfig:
    # 2D plane dimensions
    plane_w: float = 100.0
    plane_h: float = 100.0

    # Fixed waypoints: (x, y)
    waypoints: np.ndarray = field(default_factory=lambda: np.array([
        [20.0, 70.0],   # W0 – near BS1, within some sensor radii
        [50.0, 60.0],   # W1 – central, good upload coverage
        [80.0, 70.0],   # W2 – near BS3
        [35.0, 35.0],   # W3 – middle ground, sensor proximity
        [15.0, 20.0],   # W4 – deep in sensor territory (bottom-left)
    ]))

    # Sensors in the bottom-left quadrant
    sensors: list = field(default_factory=lambda: [
        SensorConfig(np.array([10.0, 10.0]), radius=18.0, job_idx=0),
        SensorConfig(np.array([25.0, 8.0]),  radius=15.0, job_idx=1),
        SensorConfig(np.array([8.0,  25.0]), radius=16.0, job_idx=2),
    ])

    # Base stations along the top
    base_stations: list = field(default_factory=lambda: [
        BaseStationConfig(np.array([15.0, 95.0]), radius=40.0),
        BaseStationConfig(np.array([50.0, 95.0]), radius=40.0),
        BaseStationConfig(np.array([85.0, 95.0]), radius=40.0),
    ])

    # Jobs (one per sensor)
    jobs: list = field(default_factory=lambda: [
        JobConfig(total_data=50.0,  deadline=120, sensor_idx=0),
        JobConfig(total_data=80.0,  deadline=150, sensor_idx=1),
        JobConfig(total_data=30.0,  deadline=80,  sensor_idx=2),
    ])

    # Channel / upload rate parameters
    alpha: float = 0.5      # sinusoid amplitude  (slowed 10x)      # sinusoid amplitude  (dominant)
    beta:  float = 0.15     # distance decay gain (slowed 10x)      # distance decay gain (secondary)
    gamma: float = 0.05     # distance scale in denominator
    omega: float = 0.15     # angular frequency of sinusoid

    # Transit model
    drone_speed: float = 1.0    # units per timestep (slowed 10x)   # units per timestep

    # Rewards
    upload_reward_scale: float = 0.1   # per MB uploaded
    completion_bonus:    float = 50.0  # per job finished before deadline
    deadline_penalty:    float = 30.0  # per job whose deadline expires
    movement_cost:       float = 0.05  # per timestep in transit

    # Episode
    max_steps: int = 300
    dt: float = 1.0   # seconds per timestep (for sinusoid)

    # Seeded per-episode variation; off by default (added when vendored).
    randomization: ScenarioRandomization = field(default_factory=ScenarioRandomization)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DroneDataRelayEnv(_GymBase):
    """
    Observation vector (flat):
        [waypoint_id (int, 0-4),
         is_moving (0/1),
         transit_steps_remaining,
         global_time,
         active_bs (0-2),
         active_channel (0-2),
         for each job: data_remaining, deadline_remaining, collected (0/1)]

    Action (Discrete):
        index = waypoint * (n_bs * n_ch) + bs * n_ch + ch
        Total: 5 * 3 * 3 = 45
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode=None):
        super().__init__()
        # ``cfg`` is the world the current episode runs in. Without
        # randomization it is the config passed in; with it, ``reset`` redraws
        # a copy of ``base_cfg`` for every episode.
        self.base_cfg = config or EnvConfig()
        self.cfg = self.base_cfg
        self.render_mode = render_mode
        self._world_rng = np.random.default_rng()
        self._rate_noise = None   # (timestep, bs, ch) table when rate noise is on

        n_wp = len(self.cfg.waypoints)
        n_bs = len(self.cfg.base_stations)
        n_ch = self.cfg.base_stations[0].n_channels
        n_jobs = len(self.cfg.jobs)

        self.n_waypoints = n_wp
        self.n_bs = n_bs
        self.n_channels = n_ch
        self.n_jobs = n_jobs

        # Action space: (waypoint, base_station, channel)
        self.action_space = spaces.Discrete(n_wp * n_bs * n_ch)

        # Observation space
        obs_dim = (
            1   # waypoint id
            + 1 # is_moving
            + 1 # transit_steps_remaining
            + 1 # global_time (normalised)
            + 1 # active_bs
            + 1 # active_channel
            + n_jobs * 3  # per job: data_remaining, deadline_remaining, collected
        )
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self._reset_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_action(self, action: int):
        ch = action % self.n_channels
        bs = (action // self.n_channels) % self.n_bs
        wp = action // (self.n_bs * self.n_channels)
        return wp, bs, ch

    def _dist(self, pos_a, pos_b):
        return float(np.linalg.norm(np.array(pos_a) - np.array(pos_b)))

    def _upload_rate(self, bs_idx: int, ch_idx: int, distance: float, t: float) -> float:
        cfg = self.cfg
        bs_cfg = cfg.base_stations[bs_idx]
        phase = bs_cfg.phase_offsets[ch_idx]
        sinusoid = cfg.alpha * np.sin(cfg.omega * t + phase)
        dist_term = cfg.beta / (1.0 + cfg.gamma * distance)
        rate = sinusoid + dist_term
        if self._rate_noise is not None:
            idx = int(round(t / cfg.dt))
            if 0 <= idx < self._rate_noise.shape[0]:
                rate += self._rate_noise[idx, bs_idx, ch_idx]
        return max(0.0, rate)   # can't be negative

    def _transit_time(self, from_wp: int, to_wp: int) -> int:
        if from_wp == to_wp:
            return 0
        d = self._dist(self.cfg.waypoints[from_wp], self.cfg.waypoints[to_wp])
        steps = max(1, int(np.ceil(d / self.cfg.drone_speed)))
        return steps

    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        max_data = max(j.total_data for j in cfg.jobs)
        max_dl   = max(j.deadline   for j in cfg.jobs)

        obs = [
            self.current_wp / (self.n_waypoints - 1),           # normalised wp id
            float(self.is_moving),
            self.transit_remaining / cfg.max_steps,
            self.t / cfg.max_steps,
            self.active_bs / (self.n_bs - 1),
            self.active_ch / (self.n_channels - 1),
        ]
        for i in range(self.n_jobs):
            obs.append(self.job_data_remaining[i] / max_data)
            obs.append(max(0, self.job_deadlines[i] - self.t) / max_dl)
            obs.append(float(self.job_collected[i]))

        return np.clip(np.array(obs, dtype=np.float32), -1, 1)

    def _reset_state(self):
        self.t = 0
        self.step_count = 0
        self.current_wp = 0
        self.is_moving = False
        self.transit_remaining = 0
        self.target_wp = 0
        self.active_bs = 0
        self.active_ch = 0
        self.active_sensor = -1
        self.collect_progress = 0.0

        self.job_data_remaining = np.array(
            [j.total_data for j in self.cfg.jobs], dtype=float
        )
        self.job_deadlines = np.array(
            [j.deadline for j in self.cfg.jobs], dtype=float
        )
        self.job_collected = np.zeros(self.n_jobs, dtype=bool)
        self.job_done = np.zeros(self.n_jobs, dtype=bool)
        self.job_failed = np.zeros(self.n_jobs, dtype=bool)

        # History for rendering / analysis
        self.history = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        randomization = self.base_cfg.randomization
        if randomization.active:
            if seed is not None:
                self._world_rng = np.random.default_rng(seed)
            self.cfg = self._draw_world(self._world_rng)
            self._rate_noise = (
                self._world_rng.normal(
                    0.0, randomization.rate_noise_std,
                    size=(2 * self.cfg.max_steps + 1, self.n_bs, self.n_channels),
                )
                if randomization.rate_noise_std > 0 else None
            )
        self._reset_state()
        info = {"jobs": len(self.cfg.jobs)}
        return self._obs(), info

    def _draw_world(self, rng) -> "EnvConfig":
        """One episode's world: a copy of ``base_cfg`` with the variation drawn.

        Draws in a fixed order (waypoints, sensors, phases, deadlines) so a
        seed always yields the same world.
        """
        base = self.base_cfg
        r = base.randomization
        cfg = copy.deepcopy(base)
        lo = np.zeros(2)
        hi = np.array([base.plane_w, base.plane_h])
        if r.position_jitter > 0:
            j = r.position_jitter
            cfg.waypoints = np.clip(
                base.waypoints + rng.uniform(-j, j, size=base.waypoints.shape), lo, hi,
            )
            for sensor in cfg.sensors:
                sensor.position = np.clip(sensor.position + rng.uniform(-j, j, size=2), lo, hi)
        if r.random_phases:
            for bs in cfg.base_stations:
                bs.phase_offsets = rng.uniform(0.0, 2 * np.pi, size=bs.n_channels)
        if r.deadline_jitter > 0:
            j = r.deadline_jitter
            for job in cfg.jobs:
                job.deadline = max(1, int(round(job.deadline * rng.uniform(1 - j, 1 + j))))
        return cfg

    def step(self, action: int):
        cfg = self.cfg
        reward = 0.0
        terminated = False
        truncated = False

        target_wp, target_bs, target_ch = self._decode_action(action)

        # --- Phase 1: transit (silent) ---
        if not self.is_moving:
            # Initiate move (could be same waypoint = 0 steps)
            self.target_wp = target_wp
            self.active_bs = target_bs
            self.active_ch = target_ch
            transit_steps = self._transit_time(self.current_wp, target_wp)
            self.transit_remaining = transit_steps
            if transit_steps > 0:
                self.is_moving = True

        # Consume one transit step if moving
        if self.is_moving:
            self.transit_remaining -= 1
            reward -= cfg.movement_cost
            self.t += cfg.dt
            self.step_count += 1
            if self.transit_remaining <= 0:
                self.is_moving = False
                self.current_wp = self.target_wp
        else:
            # Already at waypoint — execute action this step
            wp_pos = cfg.waypoints[self.current_wp]

            # --- Sensor collection (one sensor at a time, progressive) ---
            collecting_now = False
            for s_idx, sensor in enumerate(cfg.sensors):
                j_idx = sensor.job_idx
                if not self.job_collected[j_idx]:
                    d = self._dist(wp_pos, sensor.position)
                    if d <= sensor.radius:
                        # start or continue downloading this sensor
                        if self.active_sensor != s_idx:
                            self.active_sensor   = s_idx
                            self.collect_progress = 0.0
                        download_rate = 0.05   # MB per step
                        self.collect_progress += download_rate
                        collecting_now = True
                        if self.collect_progress >= cfg.jobs[j_idx].total_data:
                            self.job_collected[j_idx] = True
                            self.active_sensor   = -1
                            self.collect_progress = 0.0
                        break   # only one sensor at a time
            if not collecting_now:
                self.active_sensor   = -1
                self.collect_progress = 0.0

            # --- Upload ---
            bs_cfg = cfg.base_stations[self.active_bs]
            d_to_bs = self._dist(wp_pos, bs_cfg.position)
            if d_to_bs <= bs_cfg.radius:
                rate = self._upload_rate(self.active_bs, self.active_ch, d_to_bs, self.t)
                for j_idx in range(self.n_jobs):
                    if self.job_collected[j_idx] and not self.job_done[j_idx]:
                        upload = min(rate * cfg.dt, self.job_data_remaining[j_idx])
                        self.job_data_remaining[j_idx] -= upload
                        reward += upload * cfg.upload_reward_scale
                        if self.job_data_remaining[j_idx] <= 0:
                            if self.t <= self.job_deadlines[j_idx]:
                                reward += cfg.completion_bonus
                            self.job_done[j_idx] = True

            self.t += cfg.dt
            self.step_count += 1

        # --- Deadline checks ---
        for j_idx in range(self.n_jobs):
            if not self.job_done[j_idx] and not self.job_failed[j_idx]:
                if self.t > self.job_deadlines[j_idx]:
                    reward -= cfg.deadline_penalty
                    self.job_failed[j_idx] = True

        # --- Termination ---
        if all(self.job_done | self.job_failed):
            terminated = True
        if self.step_count >= cfg.max_steps:
            truncated = True

        # Store history snapshot
        self.history.append({
            "t": self.t,
            "step": self.step_count,
            "wp": self.current_wp,
            "is_moving": self.is_moving,
            "bs": self.active_bs,
            "ch": self.active_ch,
            "job_data_remaining": self.job_data_remaining.copy(),
            "job_collected": self.job_collected.copy(),
            "job_done": self.job_done.copy(),
            "reward": reward,
        })

        info = {
            "t": self.t,
            "job_collected": self.job_collected.tolist(),
            "job_done": self.job_done.tolist(),
            "job_failed": self.job_failed.tolist(),
            "upload_rate": 0.0,  # filled above when uploading
        }

        return self._obs(), reward, terminated, truncated, info

    def channel_rates_at(self, t: float):
        """Return upload rate for every (bs, ch) combo at time t for current waypoint."""
        wp_pos = self.cfg.waypoints[self.current_wp]
        rates = np.zeros((self.n_bs, self.n_channels))
        for b, bs_cfg in enumerate(self.cfg.base_stations):
            d = self._dist(wp_pos, bs_cfg.position)
            in_range = d <= bs_cfg.radius
            for c in range(self.n_channels):
                rates[b, c] = self._upload_rate(b, c, d, t) if in_range else 0.0
        return rates

    def render(self):
        """Basic text render."""
        print(f"t={self.t:.1f} | wp={self.current_wp} | moving={self.is_moving} "
              f"| bs={self.active_bs} ch={self.active_ch}")
        for i, job in enumerate(self.cfg.jobs):
            status = "DONE" if self.job_done[i] else ("FAIL" if self.job_failed[i] else "open")
            col = "yes" if self.job_collected[i] else "no "
            print(f"  job{i}: {self.job_data_remaining[i]:.1f}/{job.total_data:.1f} MB "
                  f"| deadline={self.job_deadlines[i]:.0f} | collected={col} | {status}")

    def action_meanings(self):
        meanings = []
        for wp in range(self.n_waypoints):
            for bs in range(self.n_bs):
                for ch in range(self.n_channels):
                    meanings.append(f"W{wp}→BS{bs+1} ch{ch+1}")
        return meanings


# ---------------------------------------------------------------------------
# The training scenario (moved from train_dqn.make_env_config, unchanged)
# ---------------------------------------------------------------------------

def overloaded_config() -> EnvConfig:
    """
    Overloaded scenario: 5 jobs, only ~3 completable within the episode.
    Forces real prioritisation — the heuristic will start dropping jobs,
    giving RL genuine room to improve via better channel/BS timing.

    This is the scenario ``train_dqn.py`` trains and evaluates on, and the
    one behind the a8a453f results plot. It lives here so that it can be
    built without torch; ``train_dqn.make_env_config`` returns it.

    Waypoints:
      W0 [10,15]  covers S0 [10,10] r=15  and S2 [8,25] r=12
      W1 [25, 9]  covers S1 [25, 8] r=12
      W2 [45,10]  covers S3 [45,10] r=12
      W3 [65,10]  covers S4 [65,10] r=12
      W4 [15,70]  upload near BS1 (dist=25 < 40)
      W5 [50,70]  upload near BS2 (dist=25 < 40)
      W6 [85,70]  upload near BS3 (dist=25 < 40)

    Collection time @ 0.05 MB/step:  2 MB=40s, 3 MB=60s, 4 MB=80s
    Min cycle (transit+collect+transit+upload) ~180-230 steps per job
    5 jobs * ~200 steps >> max_steps=700 → forced prioritisation.
    """
    cfg = EnvConfig()

    cfg.waypoints = np.array([
        [10.0, 15.0],   # W0 - sensor cluster A
        [25.0,  9.0],   # W1 - sensor B
        [45.0, 10.0],   # W2 - sensor C
        [65.0, 10.0],   # W3 - sensor D
        [15.0, 70.0],   # W4 - upload near BS1
        [50.0, 70.0],   # W5 - upload near BS2
        [85.0, 70.0],   # W6 - upload near BS3
    ])

    cfg.sensors = [
        SensorConfig(np.array([10.0, 10.0]), radius=15.0, job_idx=0),
        SensorConfig(np.array([25.0,  8.0]), radius=12.0, job_idx=1),
        SensorConfig(np.array([ 8.0, 25.0]), radius=12.0, job_idx=2),
        SensorConfig(np.array([45.0, 10.0]), radius=12.0, job_idx=3),
        SensorConfig(np.array([65.0, 10.0]), radius=12.0, job_idx=4),
    ]

    cfg.base_stations = [
        BaseStationConfig(np.array([15.0, 95.0]), radius=40.0),
        BaseStationConfig(np.array([50.0, 95.0]), radius=40.0),
        BaseStationConfig(np.array([85.0, 95.0]), radius=40.0),
    ]

    # Deadlines are intentionally tight — not all 5 can be done.
    # A good agent should finish 3-4; a great one might squeeze out 4.
    cfg.jobs = [
        JobConfig(total_data=2.0, deadline=260, sensor_idx=0),  # urgent, small
        JobConfig(total_data=3.0, deadline=380, sensor_idx=1),  # medium
        JobConfig(total_data=2.0, deadline=300, sensor_idx=2),  # urgent, small
        JobConfig(total_data=4.0, deadline=480, sensor_idx=3),  # large, relaxed
        JobConfig(total_data=2.5, deadline=340, sensor_idx=4),  # medium-urgent
    ]

    # Faster channel variation — makes timing matter more for RL
    cfg.omega = 0.25
    cfg.max_steps = 700
    return cfg


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = DroneDataRelayEnv()
    obs, info = env.reset(seed=42)
    print(f"Observation dim: {obs.shape}")
    print(f"Action space: {env.action_space.n}")

    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            env.render()
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"\nTotal reward (random policy, 50 steps): {total_reward:.2f}")
    print(f"Jobs done: {info['job_done']}")
    print(f"Jobs failed: {info['job_failed']}")