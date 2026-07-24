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
