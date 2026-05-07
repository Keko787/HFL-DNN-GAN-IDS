"""EX-3.3 — Exp3Sim sim env tests.

Pins the contracts the experiments plan §4.2 calls out:

* 3 base stations along the field edge (default).
* Time-varying per-channel upload rates depend on contact-to-BS distance.
* Fixed-speed mule: ``transit_s = dist / cruise_speed_m_s``.
* Configurable N / β / deadline_het / rrf affect the produced episode.
"""

from __future__ import annotations

import pytest

from experiments.exp3.sim_env import Exp3Sim, Exp3SimConfig


def _basic_cfg(**overrides):
    base = dict(
        n_devices=8, beta=1.0, deadline_heterogeneity=False,
        rf_range_m=60.0, world_radius=100.0,
        cruise_speed_m_s=10.0,
        mission_budget_s=600.0,
        seed=0,
    )
    base.update(overrides)
    return Exp3SimConfig(**base)


def test_default_has_three_base_stations():
    sim = Exp3Sim(_basic_cfg())
    assert len(sim.base_station_positions) == 3


def test_reset_partitions_devices_into_contacts():
    sim = Exp3Sim(_basic_cfg(n_devices=10, rf_range_m=300.0))
    sim.reset()
    contacts = sim.candidates()
    assert sum(len(c.devices) for c in contacts) == 10


def test_step_uses_fixed_speed_mule():
    sim = Exp3Sim(_basic_cfg(cruise_speed_m_s=5.0, n_devices=4, rf_range_m=300.0))
    sim.reset()
    contacts = sim.candidates()
    assert len(contacts) == 1
    chosen = contacts[0]
    result = sim.step(chosen)
    # transit_s should equal dist / cruise_speed_m_s within float precision.
    expected = result.transit_distance_m / 5.0
    assert result.transit_s == pytest.approx(expected, rel=1e-9)


def test_upload_rate_depends_on_distance_to_nearest_bs():
    sim = Exp3Sim(_basic_cfg(seed=42))
    sim.reset()
    # Sample the rate at a position very close to a BS vs far away.
    bs = sim.base_station_positions[0]
    near_rate = sim._upload_rate_at((bs[0], bs[1], 0.0))
    far_rate = sim._upload_rate_at((-10000.0, -10000.0, 0.0))
    # The far position is clamped via the floor (0.4·nominal); the near
    # position gets the full nominal rate (× a small Gaussian jitter).
    # Average over a few draws to make the assertion robust to jitter.
    near_samples = [sim._upload_rate_at((bs[0], bs[1], 0.0)) for _ in range(10)]
    far_samples = [sim._upload_rate_at((-10000.0, -10000.0, 0.0)) for _ in range(10)]
    assert sum(near_samples) / 10 > sum(far_samples) / 10


def test_deadline_heterogeneity_creates_two_groups():
    sim = Exp3Sim(_basic_cfg(deadline_heterogeneity=True, n_devices=10))
    sim.reset()
    windows = sorted({
        sim.device_states()[did].deadline_fulfilment_s
        for did in sim.device_states()
    })
    # Two distinct deadline windows when heterogeneity is on.
    assert len(windows) == 2


def test_deadline_uniform_when_heterogeneity_off():
    sim = Exp3Sim(_basic_cfg(deadline_heterogeneity=False, n_devices=10))
    sim.reset()
    windows = sorted({
        sim.device_states()[did].deadline_fulfilment_s
        for did in sim.device_states()
    })
    assert len(windows) == 1


def test_step_records_per_device_completion_counters():
    sim = Exp3Sim(_basic_cfg(n_devices=4, rf_range_m=300.0, seed=0))
    sim.reset()
    chosen = sim.candidates()[0]
    sim.step(chosen)
    visits = sim.episode_metrics.per_device_visits
    assert sum(visits.values()) == 4  # all 4 served in parallel


def test_done_when_budget_exhausted():
    sim = Exp3Sim(_basic_cfg(mission_budget_s=10.0, n_devices=10, beta=1.0))
    sim.reset()
    # Budget = 10s but session_time alone is 30s → done immediately after
    # construction (or no contact fits).
    assert sim.done


def test_record_pass2_deliveries_validates_input():
    sim = Exp3Sim(_basic_cfg())
    sim.reset()
    sim.record_pass2_deliveries(3)
    assert sim.episode_metrics.pass2_devices_reached == 3
    with pytest.raises(ValueError):
        sim.record_pass2_deliveries(-1)


def test_step_advances_now_and_consumes_budget():
    sim = Exp3Sim(_basic_cfg(n_devices=4, rf_range_m=300.0, mission_budget_s=300.0))
    sim.reset()
    before_budget = sim.budget_remaining
    before_now = sim.now
    sim.step(sim.candidates()[0])
    assert sim.budget_remaining < before_budget
    assert sim.now > before_now


def test_selector_env_carries_mission_deadline_and_rate_hook():
    sim = Exp3Sim(_basic_cfg(beta=2.0, mission_budget_s=100.0))
    sim.reset()
    env = sim.selector_env()
    assert env.mission_deadline_s == pytest.approx(200.0)
    # The env's upload-rate hook returns a positive number at any pos.
    assert env.upload_rate_bps_at is not None
    assert env.upload_rate_bps_at((0.0, 0.0, 0.0)) > 0.0


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        Exp3Sim(Exp3SimConfig(n_devices=0))
    with pytest.raises(ValueError):
        Exp3Sim(Exp3SimConfig(rf_range_m=-1.0))
    with pytest.raises(ValueError):
        Exp3Sim(Exp3SimConfig(cruise_speed_m_s=0.0))
    with pytest.raises(ValueError):
        Exp3Sim(Exp3SimConfig(beta=0.0))
    with pytest.raises(ValueError):
        Exp3Sim(Exp3SimConfig(base_station_positions=()))


# --------------------------------------------------------------------------- #
# Two-pass mission fidelity — fixes (1)–(3) + (5) from the design audit
# --------------------------------------------------------------------------- #


def test_mule_starts_at_bs_centroid_not_origin():
    """The PRE-MISSION dock-load happens at the cluster, not at (0,0,0).

    With default BS positions ``[(-80,100,0),(0,100,0),(80,100,0)]``,
    the centroid is ``(0,100,0)`` — that's where the mule starts.
    """
    sim = Exp3Sim(_basic_cfg())
    sim.reset()
    pose = sim.mule_pose
    assert pose == pytest.approx((0.0, 100.0, 0.0))


def test_initial_contacts_snapshot_persists_across_pass1():
    """``pass2_candidates`` must return every contact produced at reset,
    regardless of which were serviced in Pass 1.
    """
    sim = Exp3Sim(_basic_cfg(n_devices=12, rf_range_m=40.0, seed=0))
    sim.reset()
    initial = sim.pass2_candidates()
    n_initial = len(initial)
    # Burn through a couple of Pass-1 steps.
    if sim.candidates():
        sim.step(sim.candidates()[0])
    if sim.candidates():
        sim.step(sim.candidates()[0])
    # Pass-1 list should have shrunk; Pass-2 snapshot is unchanged.
    assert len(sim.candidates()) <= n_initial - 1
    assert len(sim.pass2_candidates()) == n_initial


def test_pass1_step_does_not_pop_device_state():
    """Pass-2 needs to look up device positions; ``step`` must retain them.
    """
    sim = Exp3Sim(_basic_cfg(n_devices=4, rf_range_m=300.0, seed=0))
    sim.reset()
    contact = sim.candidates()[0]
    devices_before = set(contact.devices)
    sim.step(contact)
    # Every device that was in the contact is still in the sim's state map.
    states = sim.device_states()
    for did in devices_before:
        assert did in states


def test_step_applies_deadline_gating():
    """A device whose absolute deadline has elapsed must be marked
    ``not completed`` even if its reliability roll would otherwise pass.
    """
    sim = Exp3Sim(_basic_cfg(n_devices=4, rf_range_m=300.0, seed=0,
                             cruise_speed_m_s=100.0))
    sim.reset()
    # Force the simulator's clock past every device's deadline.
    deadlines = sim.deadlines()
    max_deadline = max(deadlines.values()) if deadlines else 0.0
    sim._now = max_deadline + 1.0
    chosen = sim.candidates()[0]
    result = sim.step(chosen)
    # All four devices were behind the deadline → no completions.
    assert result.completed_count == 0
    # And the diagnostic counter was bumped.
    assert sim.episode_metrics.pass1_devices_deadline_missed == 4


def test_dock_at_bs_charges_transit_plus_dock_time():
    sim = Exp3Sim(_basic_cfg(n_devices=4, rf_range_m=300.0,
                             mission_budget_s=600.0, seed=0,
                             dock_time_s=30.0, cruise_speed_m_s=10.0))
    sim.reset()
    sim.step(sim.candidates()[0])
    budget_before = sim.budget_remaining
    now_before = sim.now
    result = sim.dock_at_bs()
    assert result.success
    # Charged transit_s + dock_time_s exactly.
    expected_cost = result.transit_s + result.dock_time_s
    assert sim.budget_remaining == pytest.approx(
        max(0.0, budget_before - expected_cost), rel=1e-9,
    )
    assert sim.now == pytest.approx(now_before + expected_cost, rel=1e-9)
    # Mule is at the BS now.
    assert sim.mule_pose in sim.base_station_positions
    # Episode metrics record the dock event.
    em = sim.episode_metrics
    assert em.dock_attempted is True
    assert em.dock_success is True
    assert em.dock_transit_distance_m == pytest.approx(result.transit_distance_m)
    assert em.dock_time_s == pytest.approx(30.0)


def test_dock_at_bs_fails_if_budget_too_small():
    """If residual budget can't cover transit + dock_time, the dock event
    must report ``success=False`` and leave state unchanged.
    """
    sim = Exp3Sim(_basic_cfg(n_devices=4, rf_range_m=300.0,
                             mission_budget_s=600.0, seed=0,
                             dock_time_s=30.0, cruise_speed_m_s=10.0))
    sim.reset()
    sim.step(sim.candidates()[0])
    # Drain the budget below the dock cost.
    sim._budget_remaining = 1.0
    pose_before = sim.mule_pose
    now_before = sim.now
    result = sim.dock_at_bs()
    assert result.success is False
    # Pose / clock untouched.
    assert sim.mule_pose == pose_before
    assert sim.now == now_before
    # Episode metric flags the attempt but records no success.
    assert sim.episode_metrics.dock_attempted is True
    assert sim.episode_metrics.dock_success is False


def test_step_deliver_walks_remaining_contacts_and_charges_session():
    sim = Exp3Sim(_basic_cfg(n_devices=4, rf_range_m=300.0,
                             mission_budget_s=600.0, seed=0,
                             delivery_session_s=5.0, cruise_speed_m_s=10.0))
    sim.reset()
    contact = sim.pass2_candidates()[0]
    budget_before = sim.budget_remaining
    result = sim.step_deliver(contact)
    # Transit + delivery_session_s charged, no upload.
    expected_cost = result.transit_s + result.delivery_s
    assert sim.budget_remaining == pytest.approx(
        budget_before - expected_cost, rel=1e-9,
    )
    em = sim.episode_metrics
    assert em.pass2_contacts_visited == 1
    assert em.pass2_devices_reached == result.delivered_count
    assert em.pass2_delivery_time_s == pytest.approx(5.0)
    assert sim.mule_pose == contact.position


def test_step_deliver_skips_when_budget_exhausted():
    """No state advance when residual budget < cost; no metric updates."""
    sim = Exp3Sim(_basic_cfg(n_devices=4, rf_range_m=300.0,
                             mission_budget_s=600.0, seed=0))
    sim.reset()
    contact = sim.pass2_candidates()[0]
    sim._budget_remaining = 0.5
    pose_before = sim.mule_pose
    result = sim.step_deliver(contact)
    assert result.delivered_count == 0
    assert sim.mule_pose == pose_before
    assert sim.episode_metrics.pass2_contacts_visited == 0


def test_path_length_includes_dock_and_pass2_segments():
    """``path_length_m`` must reflect the full flight path:
    Pass-1 transits + dock leg + Pass-2 walk.
    """
    sim = Exp3Sim(_basic_cfg(n_devices=4, rf_range_m=300.0,
                             mission_budget_s=600.0, seed=0))
    sim.reset()
    sim.step(sim.candidates()[0])
    pass1_dist = sim.episode_metrics.transit_distance_m
    dock = sim.dock_at_bs()
    delivery = sim.step_deliver(sim.pass2_candidates()[0])
    em = sim.episode_metrics
    assert em.path_length_m == pytest.approx(
        pass1_dist + dock.transit_distance_m + delivery.transit_distance_m,
        rel=1e-9,
    )
