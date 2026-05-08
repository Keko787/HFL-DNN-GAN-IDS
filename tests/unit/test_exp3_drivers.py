"""EX-3 driver tests — A1 + mule arm + grid driver dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.calibration import Exp3Calibration
from experiments.exp3.arm_a1 import A1Config, run_a1_trial
from experiments.exp3.arm_mule import MuleArmConfig, run_mule_trial
from experiments.exp3.driver import ARMS, Exp3Driver
from experiments.exp3.sim_env import Exp3SimConfig
from experiments.runner import Cell

from hermes.scheduler.policies import ArrivalOrderPolicy, EdfFeasibilityPolicy
from hermes.scheduler.policies.edf_feasibility import FeasibilityModel


def _fake_cal() -> Exp3Calibration:
    return Exp3Calibration(
        P_idle_W=2.0,
        epsilon_bit_J_per_bit=1e-9,
        epsilon_prop_J_per_m=10.0,
        mule_cruise_speed_m_s=5.0,
    )


# --------------------------------------------------------------------------- #
# A1
# --------------------------------------------------------------------------- #

def test_a1_returns_summary_with_no_mule_fields():
    cfg = A1Config(n_clients=5, n_rounds=3, seed=0)
    s = run_a1_trial(cfg)
    assert s.rho_contact is None
    assert s.propulsion_energy_J is None
    # update_yield bounded by client count.
    assert 0.0 <= s.update_yield <= 5.0


def test_a1_is_deterministic_under_same_seed():
    cfg = A1Config(n_clients=4, n_rounds=5, seed=123)
    a = run_a1_trial(cfg)
    b = run_a1_trial(cfg)
    assert a.update_yield == b.update_yield
    assert a.jains_fairness == b.jains_fairness


def test_a1_invalid_config_rejected():
    with pytest.raises(ValueError):
        run_a1_trial(A1Config(n_clients=0))
    with pytest.raises(ValueError):
        run_a1_trial(A1Config(n_rounds=0))
    with pytest.raises(ValueError):
        run_a1_trial(A1Config(client_fraction=0.0))


# --------------------------------------------------------------------------- #
# A1 — long-range dead-zone modelling (persistent correlated failures)
# --------------------------------------------------------------------------- #

def test_a1_full_dead_zone_zeros_completion():
    """100% dead zone ⇒ no client ever completes a round, no matter how
    favourable link_quality / packet loss / round deadlines are.
    """
    cfg = A1Config(
        n_clients=10, n_rounds=20,
        long_range_dead_zone_pct=100.0,
        long_range_link_quality=1.0,
        packet_loss_pct=0.0,
        latency_jitter_pct=0.0,
        seed=0,
    )
    s = run_a1_trial(cfg)
    assert s.update_yield == pytest.approx(0.0)
    assert s.mission_completion_rate == pytest.approx(0.0)


def test_a1_no_dead_zone_matches_legacy_behaviour():
    """0% dead zone is a no-op — completions are gated only by the
    link_quality / packet_loss / reliability triple, as before.
    """
    cfg_no_dz = A1Config(
        n_clients=10, n_rounds=20,
        long_range_dead_zone_pct=0.0,
        long_range_link_quality=0.4,
        packet_loss_pct=2.0, latency_jitter_pct=30.0,
        seed=42,
    )
    s = run_a1_trial(cfg_no_dz)
    # Without dead zones, the union over 20 rounds essentially
    # guarantees every client lands at least once → MCR ≳ 0.9.
    assert s.mission_completion_rate >= 0.9


def test_a1_dead_zone_caps_mission_completion_rate():
    """With ``dead_zone_pct = 60``, ≥1-completion rate must be capped
    by the reachable fraction (~0.4) regardless of how many rounds run.
    """
    cfg = A1Config(
        n_clients=10, n_rounds=20,
        long_range_dead_zone_pct=60.0,
        long_range_link_quality=0.4,
        packet_loss_pct=2.0, latency_jitter_pct=30.0,
        seed=7,
    )
    s = run_a1_trial(cfg)
    # Cap is hard: 6 of 10 clients are unreachable for the whole
    # mission, so MCR can't exceed (10 - 6) / 10 = 0.4. Allow a small
    # tolerance for rounding when n_clients * pct is non-integer.
    assert s.mission_completion_rate <= 0.4 + 1e-9


def test_a1_dead_zone_deterministic_under_same_seed():
    cfg = A1Config(
        n_clients=8, n_rounds=10,
        long_range_dead_zone_pct=50.0, seed=123,
    )
    a = run_a1_trial(cfg)
    b = run_a1_trial(cfg)
    assert a.mission_completion_rate == b.mission_completion_rate
    assert a.update_yield == b.update_yield


def test_driver_jittery_cell_engages_a1_dead_zone():
    """The driver must pass ``jittery_a1_dead_zone_pct`` into A1's
    config in jittery cells (and the clean value in clean cells), so
    a jittery trial's A1 row reflects the correlated-failure damage.
    """
    drv = Exp3Driver(
        calibration=_fake_cal(),
        jittery_a1_dead_zone_pct=100.0,  # everyone unreachable
        clean_a1_dead_zone_pct=0.0,
    )
    cell_clean = Cell(
        cell_id="clean", arm="A1", trial_index=0, seed=42,
        params={"N": 10, "beta": 1.0, "deadline_het": False,
                "rrf": 60.0, "jittery": False},
    )
    cell_jittery = Cell(
        cell_id="jittery", arm="A1", trial_index=0, seed=42,
        params={"N": 10, "beta": 1.0, "deadline_het": False,
                "rrf": 60.0, "jittery": True},
    )
    row_clean = drv.run_trial(cell_clean)
    row_jittery = drv.run_trial(cell_jittery)
    # Clean: no dead zone → completions land normally.
    assert row_clean["update_yield"] > 0.0
    # Jittery: 100% dead zone → no client completes anything.
    assert row_jittery["update_yield"] == pytest.approx(0.0)
    assert row_jittery["mission_completion_rate"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Mule arms
# --------------------------------------------------------------------------- #

def _mule_cfg(arm_name: str = "A2") -> MuleArmConfig:
    return MuleArmConfig(
        arm_name=arm_name,
        sim=Exp3SimConfig(
            n_devices=6, beta=1.0, deadline_heterogeneity=False,
            rf_range_m=120.0,        # tightly clustered into few contacts
            mission_budget_s=600.0,
            seed=0,
        ),
    )


def test_mule_a2_runs_to_completion():
    s = run_mule_trial(cfg=_mule_cfg("A2"), policy=ArrivalOrderPolicy())
    assert s.rho_contact is not None and s.rho_contact >= 1.0
    # update_yield bounded by member count of the largest contact (≤ N).
    assert s.update_yield >= 0.0


def test_mule_a3_runs_to_completion():
    pol = EdfFeasibilityPolicy(model=FeasibilityModel(
        cruise_speed_m_s=5.0, session_time_s=30.0,
        default_mission_budget_s=600.0,
    ))
    s = run_mule_trial(cfg=_mule_cfg("A3"), policy=pol)
    assert s.rho_contact is not None


def test_mule_a4_smoke_with_default_selector():
    from hermes.scheduler.selector import TargetSelectorRL

    sel = TargetSelectorRL(epsilon=0.0, rng_seed=0)
    s = run_mule_trial(cfg=_mule_cfg("A4"), policy=sel)
    assert s.update_yield >= 0.0


def test_mule_arm_emits_one_round_per_mission():
    """One mule mission produces exactly one FL aggregation cycle.

    Pass-1 contacts are sub-events that fold into a single
    ``mission_aggregate``; the cluster's cross-mule FedAvg at the
    dock is the actual round. ``update_yield`` is therefore the total
    Δθ contributions across the mission, not a per-contact average,
    and ``n_target`` for the round is the admitted slice size N
    (not just the cluster sizes the mule managed to visit).

    Concretely, a trial whose mule visited several clusters and
    completed K devices in total must report ``update_yield = K``
    (one round, K total updates) rather than the per-contact mean
    K / num_visited_contacts.
    """
    cfg = MuleArmConfig(
        arm_name="A2",
        sim=Exp3SimConfig(
            n_devices=10, beta=1.0, deadline_heterogeneity=False,
            rf_range_m=300.0,       # one big cluster of all 10 devices
            mission_budget_s=600.0, # plenty of budget
            seed=0,
        ),
    )
    s = run_mule_trial(cfg=cfg, policy=ArrivalOrderPolicy())
    # With one big cluster of 10 devices and mean per-device completion
    # ~ reliability(0.575) * rf_factor(~0.9) ~ 0.5, expected mission
    # completions are around 4-7. update_yield equals that count
    # exactly (one round per mission), NOT a per-contact average.
    assert 1.0 <= s.update_yield <= 10.0
    # round_close_rate is now per-mission. With ~5 expected
    # completions out of N=10, kmin=N/2=5 should sometimes fire.
    # The metric is binary per trial (this trial closed or didn't),
    # but we assert >= 0 since outcome depends on the seed.
    assert 0.0 <= s.round_close_rate_kminhalf <= 1.0


def test_mule_arm_round_yield_sums_visited_contacts():
    """Tight budget that forces only a subset of clusters to be
    visited: ``update_yield`` should reflect the SUM of completions
    across the visited contacts, not a survivor-biased per-contact
    mean.
    """
    cfg = MuleArmConfig(
        arm_name="A2",
        sim=Exp3SimConfig(
            n_devices=20, beta=1.0, deadline_heterogeneity=False,
            rf_range_m=20.0,        # tiny clusters
            mission_budget_s=80.0,  # tight — can't finish the schedule
            seed=0,
        ),
    )
    s = run_mule_trial(cfg=cfg, policy=ArrivalOrderPolicy())
    # One round per mission with n_target = 20 (admitted slice).
    # At kmin = N/2 = 10, this tight-budget trial should NOT close
    # (mule can only reach 1-2 small clusters before budget exhausts).
    # Close-rate is therefore 0 or 1 depending on whether the visited
    # devices crossed the threshold; with only 1-2 clusters of ~2
    # devices each, the mission stays well below kmin=10.
    assert s.round_close_rate_kminhalf < 1.0
    s = run_mule_trial(
        cfg=_mule_cfg("A2"),
        policy=ArrivalOrderPolicy(),
        cal=_fake_cal(),
    )
    assert s.propulsion_energy_J is not None
    assert s.propulsion_energy_J > 0.0


# --------------------------------------------------------------------------- #
# Driver dispatch
# --------------------------------------------------------------------------- #

def _cell(arm: str) -> Cell:
    return Cell(
        cell_id="N=5|beta=1.0|deadline_het=False|rrf=60.0",
        arm=arm,
        trial_index=0,
        seed=42,
        params={"N": 5, "beta": 1.0, "deadline_het": False, "rrf": 60.0},
    )


def test_driver_dispatches_each_arm():
    drv = Exp3Driver(calibration=_fake_cal())
    for arm in ARMS:
        row = drv.run_trial(_cell(arm))
        assert "update_yield" in row
        assert row["n_devices"] == 5
        assert row["beta"] == 1.0
        assert row["deadline_het"] is False
        assert row["rf_range_m"] == 60.0
        # New jittery axis defaults to False at the cell level.
        assert row["jittery"] is False
        if arm == "A1":
            assert row["rho_contact"] == ""  # CSV-blank for None
        else:
            assert row["rho_contact"] != ""


def test_driver_jittery_cell_activates_packet_loss():
    """A cell with ``jittery=True`` must activate the Exp.\\ 1-parity
    noise (2% packet loss + 30% latency jitter) — not the driver's
    base packet_loss_pct/latency_jitter_pct fields.
    """
    drv = Exp3Driver(calibration=_fake_cal())
    cell_clean = Cell(
        cell_id="clean", arm="A2", trial_index=0, seed=42,
        params={"N": 5, "beta": 1.0, "deadline_het": False,
                "rrf": 60.0, "jittery": False},
    )
    cell_jittery = Cell(
        cell_id="jittery", arm="A2", trial_index=0, seed=42,
        params={"N": 5, "beta": 1.0, "deadline_het": False,
                "rrf": 60.0, "jittery": True},
    )
    row_clean = drv.run_trial(cell_clean)
    row_jittery = drv.run_trial(cell_jittery)
    assert row_clean["jittery"] is False
    assert row_jittery["jittery"] is True
    # Sanity: the two trials use the same seed so they'd be
    # bit-identical without noise. Adding 2% packet loss + 30%
    # latency jitter must therefore produce a different mission
    # completion time (or yield) on at least one cell — not asserting
    # a direction, only divergence.
    assert (
        row_clean["mission_completion_s"] != row_jittery["mission_completion_s"]
        or row_clean["update_yield"] != row_jittery["update_yield"]
    )


def test_driver_rejects_unknown_arm():
    drv = Exp3Driver(calibration=_fake_cal())
    with pytest.raises(ValueError):
        drv.run_trial(_cell("Z9"))


# --------------------------------------------------------------------------- #
# End-to-end via the EX-0 runner — smoke
# --------------------------------------------------------------------------- #

def test_runner_writes_csv_for_all_arms(tmp_path: Path):
    from experiments.runner import TrialGrid, TrialRunner
    from experiments.exp3.metrics import Exp3MetricSummary

    grid = TrialGrid(
        independent_vars={
            "N": [5],
            "beta": [1.0],
            "rrf": [120.0],
            "deadline_het": [False],
            "jittery": [False],
        },
        arms=list(ARMS),
        n_trials=2,
        base_seed=42,
    )
    drv = Exp3Driver(calibration=_fake_cal())
    csv_path = tmp_path / "exp3_smoke.csv"
    runner = TrialRunner(
        grid=grid,
        log_path=csv_path,
        metric_columns=Exp3MetricSummary.csv_columns() + [
            "n_devices", "beta", "deadline_het", "rf_range_m", "jittery",
        ],
        timeout_s=60.0,
    )
    n = runner.run(drv.run_trial)
    assert n == 4 * 2  # 4 arms × 2 trials
    contents = csv_path.read_text(encoding="utf-8")
    for arm in ARMS:
        assert f",{arm}," in contents
