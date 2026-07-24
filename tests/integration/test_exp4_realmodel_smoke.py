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
    assert row["update_yield"] == pytest.approx(3.0)
    assert row["coverage"] == pytest.approx(1.0)
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
    assert clean["coverage"] == pytest.approx(1.0)
    assert clean["update_yield"] == pytest.approx(5.0)

    # Jittery collapses participation.
    assert jittery["coverage"] < clean["coverage"]
    assert jittery["update_yield"] < clean["update_yield"]
    # At most the reachable 2/5 can ever be covered.
    assert float(jittery["coverage"]) <= 0.4 + 1e-9
    # Round-close rate does not improve under jitter.
    assert float(jittery["round_close_rate_kmin1"]) <= float(clean["round_close_rate_kmin1"])


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
