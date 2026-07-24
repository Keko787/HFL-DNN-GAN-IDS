"""EX-4.1 slow smoke — real DNN-IDS converging through the integrated stack.

Drives ``Exp4Driver(real_model=True)`` for one tiny trial on the synthetic
(real-shaped, no-dataset) task and asserts the cluster emitted a
per-round ``model_eval`` convergence trace and that the aggregated model
improves off the seeded random baseline. This is the end-to-end proof that
the real ``create_CICIOT_Model`` trains on the devices, is FedAvg-aggregated
by the cluster, and is scored on the held-out set — the EX-4.1 headline.

Marked ``slow``: spawns a real subprocess tree, each running TensorFlow.
Uses the synthetic data source so it needs no dataset and stays
deterministic; the canonical (real-CICIOT) path is the same wiring with a
different loader.
"""

from __future__ import annotations

import pytest

from experiments.exp4.driver import Exp4Driver
from experiments.exp4.metrics import Exp4MetricSummary
from experiments.runner.grid import Cell


@pytest.mark.slow
def test_exp4_real_model_synthetic_converges():
    driver = Exp4Driver(
        real_model=True,
        data_source="synthetic",
        default_n_devices=2,
        default_rf_range_m=60.0,
        default_n_missions=2,
        local_epochs=8,
        synth_rows_per_device=400,
        synth_test_rows=400,
        tau=0.9,
        trial_budget_s=240.0,
        startup_timeout_s=60.0,
    )
    cell = Cell(
        cell_id="rm-smoke", arm="H1", trial_index=0, seed=7,
        params={"N": 2, "rrf": 60.0, "n_missions": 2},
    )

    row = dict(driver.run_trial(cell))

    # Row is complete + CSV-shaped.
    assert set(row.keys()) == set(Exp4MetricSummary.csv_columns())

    # The integrated stack ran: real two-pass + cross-mule FedAvg.
    assert row["missions_completed"] >= 1, row
    assert row["rounds_closed"] >= 1, row

    # The real model was in the loop: a convergence trace was emitted
    # (baseline round 0 + at least one aggregated round).
    assert row["rounds_evaluated"] >= 2, (
        f"expected a model_eval trace; got rounds_evaluated={row['rounds_evaluated']}"
    )
    assert row["init_auc"] != "", "no baseline model_eval — real model not wired"
    assert row["final_auc"] != ""
    assert row["update_yield"] >= 1.0

    # On the separable synthetic task the aggregated model beats the random
    # init baseline and learns the task well above chance.
    assert float(row["best_auc"]) >= float(row["init_auc"])
    assert float(row["best_auc"]) > 0.65, (
        f"model did not learn the separable task: best_auc={row['best_auc']}"
    )


@pytest.mark.slow
def test_exp4_h0_flat_fl_synthetic_converges():
    """H0 (traditional flat FL) — the paired null — also converges.

    Runs in-process (no mule / orchestrator) and emits the same convergence
    columns as H1, with the mule-only metrics blanked (N/A).
    """
    driver = Exp4Driver(
        real_model=True,
        data_source="synthetic",
        default_n_devices=3,
        local_epochs=8,
        synth_rows_per_device=400,
        synth_test_rows=400,
        tau=0.9,
        h0_client_fraction=1.0,
    )
    cell = Cell(
        cell_id="h0-smoke", arm="H0", trial_index=0, seed=11,
        params={"N": 3, "rrf": 60.0, "n_missions": 3},
    )

    row = dict(driver.run_trial(cell))

    assert set(row.keys()) == set(Exp4MetricSummary.csv_columns())
    # Baseline (round 0) + 3 aggregated rounds.
    assert row["rounds_evaluated"] >= 4
    assert row["rounds_closed"] == 3
    assert row["missions_completed"] == 0        # no mule
    # Mule-only metrics are N/A for flat FL.
    assert row["pass2_coverage"] == ""
    assert row["rho_contact"] == ""
    # All 3 clients participate every round at fraction 1.0.
    # H0 now pays the same per-device reliability tax as H1 (fair clean
    # comparison), so per-round yield is below the client count and coverage
    # need not be a trivial 1.0.
    assert 0.0 < row["update_yield"] <= 3.0
    assert 0.0 < row["coverage"] <= 1.0
    # Convergence off the random baseline.
    assert float(row["best_auc"]) >= float(row["init_auc"])
    assert float(row["best_auc"]) > 0.65


@pytest.mark.slow
def test_exp4_h0_jittery_collapses_participation():
    """EX-4.2 — under jittery, H0's backhaul dead-zone collapses participation.

    Clean: all clients reach the server every round. Jittery: a dead-zone
    fraction is persistently unreachable + reachable clients fail
    intermittently, so coverage / update-yield / round-close all drop.
    """
    driver = Exp4Driver(
        real_model=True,
        data_source="synthetic",
        default_n_devices=5,
        local_epochs=4,
        synth_rows_per_device=200,
        synth_test_rows=300,
        tau=0.9,
        jittery_dead_zone_frac=0.6,   # 3 of 5 unreachable
        jittery_link_quality=0.4,
    )
    common = dict(arm="H0", trial_index=0, seed=21)
    clean = dict(driver.run_trial(Cell(
        cell_id="rg", params={"N": 5, "rrf": 60.0, "n_missions": 4, "regime": "clean"},
        **common,
    )))
    jittery = dict(driver.run_trial(Cell(
        cell_id="rg", params={"N": 5, "rrf": 60.0, "n_missions": 4, "regime": "jittery"},
        **common,
    )))

    # Clean: every client every round.
    # Clean H0 now pays the reliability tax too (fair), so it need not be
    # perfect — but it is far healthier than jittery.
    assert 0.0 < clean["update_yield"] <= 5.0
    assert clean["coverage"] > 0.4

    # Jittery collapses participation relative to clean.
    assert jittery["coverage"] < clean["coverage"]
    assert jittery["update_yield"] < clean["update_yield"]
    # At most the reachable 2/5 can ever be covered (dead-zone 0.6).
    assert float(jittery["coverage"]) <= 0.4 + 1e-9
    # Round-close rate does not improve under jitter.
    assert float(jittery["round_close_rate_kmin1"]) <= float(clean["round_close_rate_kmin1"])


@pytest.mark.slow
def test_exp4_h1_realism_is_imperfect_and_does_not_collapse():
    """EX-4.2 — with realism on, H1 is not a perfect line.

    Per-device short-range contact reliability (U(0.15,1.0) x rf_factor)
    means some contacts fail even under clean links, so H1's per-round
    update-yield is strictly below N and clean coverage is not trivially
    1.0. Under jittery the recoverable backhaul loss adds a small penalty,
    but H1 does NOT collapse (the mule still physically reaches devices) —
    the authentic contrast with H0's dead-zone collapse.
    """
    driver = Exp4Driver(
        real_model=True,
        data_source="synthetic",
        realism=True,
        default_n_devices=6,
        local_epochs=2,
        synth_rows_per_device=160,
        synth_test_rows=240,
        h1_field_radius_m=90.0,
        jittery_backhaul_loss_pct=2.0,
        trial_budget_s=300.0,
        startup_timeout_s=60.0,
    )
    common = dict(arm="H1", trial_index=0, seed=5)
    clean = dict(driver.run_trial(Cell(
        cell_id="rl", params={"N": 6, "rrf": 60.0, "n_missions": 4, "regime": "clean"},
        **common,
    )))
    jittery = dict(driver.run_trial(Cell(
        cell_id="rl", params={"N": 6, "rrf": 60.0, "n_missions": 4, "regime": "jittery"},
        **common,
    )))

    # The stack still ran end to end.
    assert clean["missions_completed"] >= 1
    assert clean["rounds_closed"] >= 1
    # Realism fired: contacts fail, so per-round yield is strictly below N
    # (without realism it is exactly N). This is the anti-rig check.
    assert 0.0 < float(clean["update_yield"]) < float(clean["n_devices"])
    # H1 does not collapse under jittery — it still reaches most devices,
    # far above H0's jittery dead-zone floor.
    assert float(jittery["coverage"]) > 0.4
    assert float(jittery["update_yield"]) > 0.0


@pytest.mark.slow
def test_exp4_h2_rl_selector_runs_end_to_end():
    """EX-4.2 arm H2 — the RL target selector runs through the REAL
    orchestrator (S3.5 tie-break in FLScheduler), producing a valid trial.

    (Whether H2 beats H1 needs trained weights + a paired sweep; here we
    only pin that the selector is wired and the integrated trial completes.)
    """
    driver = Exp4Driver(
        real_model=True,
        data_source="synthetic",
        realism=True,               # spread -> multiple contacts, so the
        default_n_devices=6,        # selector actually has candidates to rank
        local_epochs=2,
        synth_rows_per_device=160,
        synth_test_rows=240,
        h1_field_radius_m=90.0,
        trial_budget_s=300.0,
        startup_timeout_s=60.0,
    )
    cell = Cell(
        cell_id="h2", arm="H2", trial_index=0, seed=9,
        params={"N": 6, "rrf": 60.0, "n_missions": 3, "regime": "clean"},
    )
    row = dict(driver.run_trial(cell))
    assert set(row.keys()) == set(Exp4MetricSummary.csv_columns())
    assert row["missions_completed"] >= 1
    assert row["rounds_closed"] >= 1
    assert row["rounds_evaluated"] >= 2


@pytest.mark.slow
def test_exp4_h3_l1_channel_runs_end_to_end():
    """EX-4.3 arm H3 — the adaptive L1 channel is wired through the REAL stack.

    Proves the plumbing: the driver builds a per-mission backhaul-loss
    schedule from the channel model, threads it into the cluster (which
    applies it as the per-mission Bernoulli drop), and feeds the chosen
    channel's mean SNR to the mule's target selector as its RF prior. The
    jittery trial completes end-to-end with a convergence trace.

    (That adaptive H3 loses less backhaul than fixed H2 is proved
    deterministically across seeds in tests/unit/test_exp4_channel.py; the
    stochastic subprocess trial only pins the wiring here.)
    """
    driver = Exp4Driver(
        real_model=True,
        data_source="synthetic",
        realism=True,
        l1_channel=True,            # arm H3 backhaul comes from the channel model
        default_n_devices=6,
        local_epochs=2,
        synth_rows_per_device=160,
        synth_test_rows=240,
        h1_field_radius_m=90.0,
        trial_budget_s=300.0,
        startup_timeout_s=60.0,
    )
    cell = Cell(
        cell_id="h3", arm="H3", trial_index=0, seed=17,
        params={"N": 6, "rrf": 60.0, "n_missions": 4, "regime": "jittery"},
    )
    row = dict(driver.run_trial(cell))
    assert set(row.keys()) == set(Exp4MetricSummary.csv_columns())
    # The integrated stack ran with the channel-driven backhaul + RL selector.
    assert row["missions_completed"] >= 1, row
    assert row["rounds_closed"] >= 1, row
    assert row["rounds_evaluated"] >= 2, row
    # H3 still reaches devices over short-range contact (jitter hits only the
    # long-range backhaul), so it does not collapse.
    assert float(row["update_yield"]) > 0.0


@pytest.mark.slow
def test_exp4_h0_requires_real_model():
    """H0 without real_model is a clear error, not a silent stub run."""
    driver = Exp4Driver(real_model=False)
    cell = Cell(
        cell_id="h0-err", arm="H0", trial_index=0, seed=1,
        params={"N": 2, "rrf": 60.0, "n_missions": 1},
    )
    with pytest.raises(ValueError, match="H0.*real"):
        driver.run_trial(cell)
