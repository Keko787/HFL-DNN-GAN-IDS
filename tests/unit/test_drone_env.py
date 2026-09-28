"""
Tests for DroneDataRelayEnv (experiments/sim/drone_env).

The unittest classes are the original hermes_rl suite at commit a8a453f,
with only the import changed. The pytest functions after them were added
when the environment was vendored: golden rollouts recorded from the
original files, which pin the default behaviour to that commit, and tests
for the opt-in seeded variation.

Run with:  python -m pytest tests/unit/test_drone_env.py -v
"""

import math
import unittest
import numpy as np
import pytest

from experiments.sim.drone_env import (
    BaseStationConfig,
    DroneDataRelayEnv,
    EnvConfig,
    JobConfig,
    ScenarioRandomization,
    SensorConfig,
    overloaded_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(seed=0, **cfg_overrides):
    cfg = EnvConfig(**cfg_overrides) if cfg_overrides else EnvConfig()
    env = DroneDataRelayEnv(config=cfg)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class TestActionSpace(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_action_count(self):
        n_wp = len(self.env.cfg.waypoints)
        n_bs = len(self.env.cfg.base_stations)
        n_ch = self.env.cfg.base_stations[0].n_channels
        self.assertEqual(self.env.action_space.n, n_wp * n_bs * n_ch)

    def test_decode_round_trip(self):
        """Every action index decodes to valid (wp, bs, ch) and re-encodes correctly."""
        env = self.env
        for action in range(env.action_space.n):
            wp, bs, ch = env._decode_action(action)
            self.assertIn(wp, range(env.n_waypoints))
            self.assertIn(bs, range(env.n_bs))
            self.assertIn(ch, range(env.n_channels))
            re_encoded = wp * env.n_bs * env.n_channels + bs * env.n_channels + ch
            self.assertEqual(re_encoded, action)

    def test_decode_known_values(self):
        env = self.env
        # action 0  -> wp=0, bs=0, ch=0
        self.assertEqual(env._decode_action(0), (0, 0, 0))
        # action 1  -> wp=0, bs=0, ch=1
        self.assertEqual(env._decode_action(1), (0, 0, 1))
        # last action -> wp=n_wp-1, bs=n_bs-1, ch=n_ch-1
        last = env.action_space.n - 1
        self.assertEqual(env._decode_action(last),
                         (env.n_waypoints - 1, env.n_bs - 1, env.n_channels - 1))


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

class TestObservationSpace(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_obs_shape_matches_space(self):
        obs, _ = self.env.reset(seed=1)
        self.assertEqual(obs.shape, self.env.observation_space.shape)

    def test_obs_dtype(self):
        obs, _ = self.env.reset(seed=1)
        self.assertEqual(obs.dtype, np.float32)

    def test_obs_bounds_after_reset(self):
        obs, _ = self.env.reset(seed=1)
        self.assertTrue(np.all(obs >= -1.0), "Obs below -1 after reset")
        self.assertTrue(np.all(obs <=  1.0), "Obs above  1 after reset")

    def test_obs_bounds_throughout_episode(self):
        env = self.env
        env.reset(seed=42)
        for _ in range(200):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            self.assertTrue(np.all(obs >= -1.0), "Obs below -1 during episode")
            self.assertTrue(np.all(obs <=  1.0), "Obs above  1 during episode")
            if terminated or truncated:
                break

    def test_obs_dim_formula(self):
        env = self.env
        expected = 6 + env.n_jobs * 3
        self.assertEqual(env.observation_space.shape[0], expected)


# ---------------------------------------------------------------------------
# Reset reproducibility
# ---------------------------------------------------------------------------

class TestReset(unittest.TestCase):

    def test_same_seed_same_obs(self):
        env = DroneDataRelayEnv()
        obs1, _ = env.reset(seed=7)
        obs2, _ = env.reset(seed=7)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_clears_state(self):
        env = make_env()
        for _ in range(50):
            env.step(env.action_space.sample())
        env.reset(seed=0)
        self.assertEqual(env.t, 0)
        self.assertEqual(env.step_count, 0)
        self.assertFalse(env.is_moving)
        self.assertTrue(np.all(~env.job_done))
        self.assertTrue(np.all(~env.job_failed))
        self.assertTrue(np.all(~env.job_collected))

    def test_reset_restores_data(self):
        env = make_env()
        for _ in range(100):
            env.step(env.action_space.sample())
        env.reset(seed=0)
        for i, job in enumerate(env.cfg.jobs):
            self.assertAlmostEqual(env.job_data_remaining[i], job.total_data)


# ---------------------------------------------------------------------------
# Upload rate
# ---------------------------------------------------------------------------

class TestUploadRate(unittest.TestCase):

    def setUp(self):
        self.env = make_env()
        self.cfg = self.env.cfg

    def test_rate_non_negative(self):
        env = self.env
        for bs in range(env.n_bs):
            for ch in range(env.n_channels):
                for d in [0.0, 10.0, 50.0, 100.0]:
                    rate = env._upload_rate(bs, ch, d, t=0.0)
                    self.assertGreaterEqual(rate, 0.0)

    def test_rate_formula(self):
        env = self.env
        cfg = self.cfg
        bs_idx, ch_idx = 0, 0
        d, t = 10.0, 5.0
        phase = cfg.base_stations[bs_idx].phase_offsets[ch_idx]
        expected = max(0.0,
            cfg.alpha * math.sin(cfg.omega * t + phase)
            + cfg.beta / (1.0 + cfg.gamma * d)
        )
        self.assertAlmostEqual(env._upload_rate(bs_idx, ch_idx, d, t), expected, places=6)

    def test_rate_decreases_with_distance(self):
        """Holding time fixed, rate (distance component) should decrease as distance grows."""
        env = self.env
        cfg = self.cfg
        # Use t where sin is positive to ensure the rate is positive at all distances
        t = math.pi / (2 * cfg.omega)  # sin(omega*t) = 1 (peak)
        rates = [env._upload_rate(0, 0, d, t) for d in [0.0, 10.0, 30.0, 60.0]]
        for i in range(len(rates) - 1):
            self.assertGreaterEqual(rates[i], rates[i + 1])

    def test_channel_rates_at_shape(self):
        env = self.env
        env.reset(seed=0)
        rates = env.channel_rates_at(t=0.0)
        self.assertEqual(rates.shape, (env.n_bs, env.n_channels))
        self.assertTrue(np.all(rates >= 0.0))


# ---------------------------------------------------------------------------
# Transit time
# ---------------------------------------------------------------------------

class TestTransitTime(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_same_waypoint_zero_steps(self):
        env = self.env
        for wp in range(env.n_waypoints):
            self.assertEqual(env._transit_time(wp, wp), 0)

    def test_transit_positive_between_distinct(self):
        env = self.env
        for a in range(env.n_waypoints):
            for b in range(env.n_waypoints):
                if a != b:
                    self.assertGreater(env._transit_time(a, b), 0)

    def test_transit_is_ceil_of_distance(self):
        env = self.env
        cfg = env.cfg
        for a in range(env.n_waypoints):
            for b in range(env.n_waypoints):
                if a != b:
                    d = np.linalg.norm(cfg.waypoints[a] - cfg.waypoints[b])
                    expected = max(1, int(math.ceil(d / cfg.drone_speed)))
                    self.assertEqual(env._transit_time(a, b), expected)


# ---------------------------------------------------------------------------
# Step mechanics
# ---------------------------------------------------------------------------

class TestStepMechanics(unittest.TestCase):

    def test_step_count_increments(self):
        env = make_env()
        for i in range(1, 11):
            env.step(0)
            self.assertEqual(env.step_count, i)

    def test_time_increments(self):
        env = make_env()
        for _ in range(10):
            prev_t = env.t
            env.step(0)
            self.assertGreater(env.t, prev_t)

    def test_movement_cost_applied_while_moving(self):
        """When the drone is in transit, reward should include a movement cost deduction."""
        env = DroneDataRelayEnv()
        env.reset(seed=0)
        # Action that moves to a different waypoint (wp=4, bs=0, ch=0)
        action = 4 * env.n_bs * env.n_channels
        _, reward, _, _, _ = env.step(action)
        # If the drone started moving, movement_cost must be deducted
        if env.is_moving or env.transit_remaining > 0 or env.current_wp == 4:
            # reward may include cost; just assert it's not spuriously positive
            self.assertLessEqual(reward, 0.0 + env.cfg.completion_bonus * env.n_jobs)

    def test_truncation_at_max_steps(self):
        cfg = EnvConfig(max_steps=10)
        env = DroneDataRelayEnv(config=cfg)
        env.reset(seed=0)
        truncated = False
        for _ in range(15):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        self.assertTrue(truncated or terminated)

    def test_step_returns_five_tuple(self):
        env = make_env()
        result = env.step(0)
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_info_keys_present(self):
        env = make_env()
        _, _, _, _, info = env.step(0)
        for key in ("t", "job_collected", "job_done", "job_failed", "upload_rate"):
            self.assertIn(key, info)

    def test_history_grows_per_step(self):
        env = make_env()
        for i in range(1, 6):
            env.step(env.action_space.sample())
            self.assertEqual(len(env.history), i)


# ---------------------------------------------------------------------------
# Reward signals
# ---------------------------------------------------------------------------

class TestRewards(unittest.TestCase):

    def test_deadline_penalty_fires(self):
        """Force a deadline to expire and check the penalty is applied."""
        cfg = EnvConfig(max_steps=500, deadline_penalty=30.0)
        # Set an extremely tight deadline so it expires immediately
        cfg.jobs[0] = JobConfig(total_data=1000.0, deadline=1, sensor_idx=0)
        env = DroneDataRelayEnv(config=cfg)
        env.reset(seed=0)
        total_reward = 0.0
        for _ in range(5):
            _, r, terminated, truncated, _ = env.step(0)
            total_reward += r
            if terminated or truncated:
                break
        self.assertLess(total_reward, 0.0)

    def test_completion_bonus_fires(self):
        """Manually drive job to completion and verify bonus appears in reward."""
        cfg = EnvConfig(
            max_steps=5000,
            completion_bonus=50.0,
            upload_reward_scale=0.1,
        )
        # Tiny job, long deadline
        cfg.jobs = [JobConfig(total_data=0.01, deadline=4000, sensor_idx=0)]
        # Place sensor right at waypoint 4 (15, 20) with large radius
        cfg.sensors = [SensorConfig(np.array([15.0, 20.0]), radius=20.0, job_idx=0)]
        env = DroneDataRelayEnv(config=cfg)
        env.reset(seed=0)

        # Manually mark the job as collected so we skip the collection phase
        env.job_collected[0] = True

        total_reward = 0.0
        found_bonus = False
        for _ in range(4000):
            _, r, terminated, truncated, info = env.step(0)
            total_reward += r
            if info["job_done"][0]:
                found_bonus = True
                break
            if terminated or truncated:
                break

        self.assertTrue(found_bonus, "Job was never completed")
        self.assertGreater(total_reward, cfg.completion_bonus / 2,
                           "Total reward should include the completion bonus")


# ---------------------------------------------------------------------------
# Full episode rollout
# ---------------------------------------------------------------------------

class TestEpisodeRollout(unittest.TestCase):

    def test_random_policy_completes(self):
        """A full random-policy episode must terminate within max_steps."""
        env = make_env()
        env.reset(seed=99)
        steps = 0
        for _ in range(env.cfg.max_steps + 10):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            if terminated or truncated:
                break
        self.assertLessEqual(steps, env.cfg.max_steps + 1)

    def test_multiple_resets_independent(self):
        """Two independent resets with different seeds should yield different observations."""
        env = DroneDataRelayEnv()
        obs_a, _ = env.reset(seed=1)
        for _ in range(20):
            env.step(env.action_space.sample())
        obs_b, _ = env.reset(seed=999)
        # obs_b should be same as a fresh seed=999 reset
        env2 = DroneDataRelayEnv()
        obs_ref, _ = env2.reset(seed=999)
        np.testing.assert_array_equal(obs_b, obs_ref)

    def test_episode_history_length(self):
        env = make_env()
        env.reset(seed=0)
        n_steps = 30
        for _ in range(n_steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        self.assertEqual(len(env.history), env.step_count)

    def test_jobs_eventually_fail_or_done(self):
        """By episode end, every job must be either done or failed."""
        env = make_env()
        env.reset(seed=5)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        for i in range(env.n_jobs):
            self.assertTrue(
                env.job_done[i] or env.job_failed[i],
                f"Job {i} is neither done nor failed at episode end",
            )


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------

class TestCustomConfig(unittest.TestCase):

    def test_custom_config_applied(self):
        cfg = EnvConfig(max_steps=50, drone_speed=5.0, completion_bonus=99.0)
        env = DroneDataRelayEnv(config=cfg)
        self.assertEqual(env.cfg.max_steps, 50)
        self.assertEqual(env.cfg.drone_speed, 5.0)
        self.assertEqual(env.cfg.completion_bonus, 99.0)

    def test_action_space_scales_with_waypoints(self):
        cfg = EnvConfig()
        cfg.waypoints = np.array([[0.0, 0.0], [100.0, 0.0]])  # 2 waypoints
        env = DroneDataRelayEnv(config=cfg)
        env.reset(seed=0)
        self.assertEqual(env.action_space.n, 2 * env.n_bs * env.n_channels)


# ---------------------------------------------------------------------------
# Added when vendored: golden rollouts recorded from the a8a453f originals
# ---------------------------------------------------------------------------
#
# Each rollout was run against the original hermes_rl files (the scenario from
# the original train_dqn.make_env_config) and its result recorded here. The
# vendored copy must reproduce them exactly: with default settings it is the
# original environment.

def _random_rollout(cfg, n_steps, action_seed):
    env = DroneDataRelayEnv(config=cfg)
    env.reset(seed=0)
    actions = np.random.default_rng(action_seed).integers(0, env.action_space.n, size=n_steps)
    return _finish(env, (int(a) for a in actions))


def _scripted_rollout(cfg):
    """Collect at W0 for 90 steps, then fly to W4 and upload on BS1's best channel."""
    env = DroneDataRelayEnv(config=cfg)
    env.reset(seed=0)
    per_wp = env.n_bs * env.n_channels

    def policy():
        for _ in range(90):
            yield 0
        while True:
            yield 4 * per_wp + int(np.argmax(env.channel_rates_at(env.t)[0]))

    return _finish(env, policy(), limit=450)


def _finish(env, actions, limit=10_000):
    total, steps = 0.0, 0
    for a in actions:
        _, r, terminated, truncated, _ = env.step(a)
        total += r
        steps += 1
        if terminated or truncated or steps >= limit:
            break
    return total, env


def test_golden_default_scenario_random_actions():
    total, env = _random_rollout(EnvConfig(), 300, action_seed=0)
    assert total == pytest.approx(-97.5499999999998, rel=1e-12)
    assert (env.t, env.step_count) == (151.0, 151)
    assert env.job_failed.tolist() == [True, True, True]


def test_golden_training_scenario_random_actions():
    total, env = _random_rollout(overloaded_config(), 700, action_seed=1)
    assert total == pytest.approx(-173.95000000000084, rel=1e-12)
    assert (env.t, env.step_count) == (481.0, 481)
    assert env.job_failed.tolist() == [True] * 5


def test_golden_training_scenario_collect_and_upload():
    """Exercises collection, upload, completion bonus and deadline penalty."""
    total, env = _scripted_rollout(overloaded_config())
    assert total == pytest.approx(37.60000000000001, rel=1e-12)
    assert (env.t, env.step_count) == (450.0, 450)
    assert env.job_collected.tolist() == [True, False, True, False, False]
    assert env.job_done.tolist() == [True, False, True, False, False]
    assert env.job_failed.tolist() == [False, True, False, False, True]


def test_the_training_scenario_is_the_documented_one():
    cfg = overloaded_config()
    assert (len(cfg.waypoints), len(cfg.sensors), len(cfg.jobs)) == (7, 5, 5)
    assert (cfg.omega, cfg.max_steps) == (0.25, 700)
    assert [j.deadline for j in cfg.jobs] == [260, 380, 300, 480, 340]


# ---------------------------------------------------------------------------
# Added when vendored: opt-in seeded variation
# ---------------------------------------------------------------------------

VARIED = dict(position_jitter=5.0, random_phases=True, deadline_jitter=0.2, rate_noise_std=0.05)


def _varied_env(**overrides):
    cfg = overloaded_config()
    cfg.randomization = ScenarioRandomization(**{**VARIED, **overrides})
    return DroneDataRelayEnv(config=cfg)


def _world(env):
    return (
        env.cfg.waypoints.copy(),
        [s.position.copy() for s in env.cfg.sensors],
        [bs.phase_offsets.copy() for bs in env.cfg.base_stations],
        [j.deadline for j in env.cfg.jobs],
    )


def _same_world(a, b):
    return (
        np.array_equal(a[0], b[0])
        and all(np.array_equal(x, y) for x, y in zip(a[1], b[1]))
        and all(np.array_equal(x, y) for x, y in zip(a[2], b[2]))
        and a[3] == b[3]
    )


def test_variation_is_off_by_default_and_seeds_change_nothing():
    env = DroneDataRelayEnv(config=overloaded_config())
    assert not env.cfg.randomization.active
    env.reset(seed=1)
    first = _world(env)
    env.reset(seed=2)
    assert _same_world(first, _world(env))
    assert env.cfg is env.base_cfg


def test_one_seed_gives_one_world_and_one_episode():
    a, b = _varied_env(), _varied_env()
    a.reset(seed=11)
    b.reset(seed=11)
    assert _same_world(_world(a), _world(b))
    assert _scripted_total(a) == _scripted_total(b)


def test_different_seeds_give_different_worlds():
    env = _varied_env()
    env.reset(seed=11)
    first = _world(env)
    env.reset(seed=12)
    assert not _same_world(first, _world(env))


def test_a_reset_without_a_seed_continues_the_stream():
    env = _varied_env()
    env.reset(seed=11)
    first = _world(env)
    env.reset()
    assert not _same_world(first, _world(env))


def test_the_base_config_is_never_mutated():
    env = _varied_env()
    before = env.base_cfg.waypoints.copy()
    for seed in range(5):
        env.reset(seed=seed)
    assert np.array_equal(env.base_cfg.waypoints, before)
    assert env.base_cfg.base_stations[0].phase_offsets.tolist() == [0.0, 2.094, 4.189]


def test_jittered_positions_stay_on_the_plane():
    env = _varied_env(position_jitter=50.0)
    for seed in range(20):
        env.reset(seed=seed)
        pts = np.vstack([env.cfg.waypoints] + [s.position for s in env.cfg.sensors])
        assert pts.min() >= 0.0 and pts[:, 0].max() <= 100.0 and pts[:, 1].max() <= 100.0


def test_deadlines_stay_within_the_jitter_band():
    env = _varied_env(deadline_jitter=0.2)
    base = [j.deadline for j in overloaded_config().jobs]
    for seed in range(20):
        env.reset(seed=seed)
        for d0, d in zip(base, (j.deadline for j in env.cfg.jobs)):
            assert round(d0 * 0.8) <= d <= round(d0 * 1.2)


def test_rate_noise_depends_on_time_not_on_queries():
    """An agent that peeks at channel rates must see the same world as one
    that does not, or paired seeds stop being paired."""
    quiet, peeking = _varied_env(), _varied_env()
    quiet.reset(seed=5)
    peeking.reset(seed=5)
    for t in range(0, 200, 7):
        peeking.channel_rates_at(float(t))
    assert _scripted_total(quiet) == _scripted_total(peeking)


def test_rate_noise_moves_the_rate_only_when_on():
    noisy, clean = _varied_env(), _varied_env(rate_noise_std=0.0)
    noisy.reset(seed=5)
    clean.reset(seed=5)
    assert _same_world(_world(noisy), _world(clean))   # same draws before the noise
    # W4 is an upload spot; the reset position (W0) reaches no base station.
    noisy.current_wp = clean.current_wp = 4
    differs = [
        not np.allclose(noisy.channel_rates_at(float(t)), clean.channel_rates_at(float(t)))
        for t in range(60)
    ]
    assert any(differs)


@pytest.mark.parametrize("bad", [
    dict(position_jitter=-1.0), dict(rate_noise_std=-0.1), dict(deadline_jitter=1.0),
])
def test_invalid_variation_is_refused(bad):
    with pytest.raises(ValueError):
        ScenarioRandomization(**bad)


def _scripted_total(env):
    """The collect-then-upload script, from the env's current episode."""
    per_wp = env.n_bs * env.n_channels
    total = 0.0
    for step in range(450):
        action = 0 if step < 90 else 4 * per_wp + int(np.argmax(env.channel_rates_at(env.t)[0]))
        _, r, terminated, truncated, _ = env.step(action)
        total += r
        if terminated or truncated:
            break
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
