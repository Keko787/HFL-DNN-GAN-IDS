"""EX-4.0 slow smoke — one real integrated trial end-to-end.

Drives the *real* multi-process orchestrator through ``Exp4Driver`` for a
single tiny H1 trial (1 cluster + 2 devices + 1 mule, one mission) and
asserts the driver produces a well-formed metric row computed from the
real JSONL event stream. This is the integration counterpart to the fast
``tests/unit/test_exp4_metrics.py`` unit test.

Marked ``slow`` because it spawns a subprocess tree over real TCP.
"""

from __future__ import annotations

import pytest

from experiments.exp4.driver import Exp4Driver
from experiments.exp4.metrics import Exp4MetricSummary
from experiments.runner.grid import Cell


@pytest.mark.slow
def test_exp4_one_trial_runs_real_orchestrator():
    driver = Exp4Driver(
        default_n_devices=2,
        default_rf_range_m=60.0,
        default_n_missions=1,
        trial_budget_s=90.0,
        startup_timeout_s=30.0,
    )
    cell = Cell(
        cell_id="smoke",
        arm="H1",
        trial_index=0,
        seed=12345,
        params={"N": 2, "rrf": 60.0, "n_missions": 1},
    )

    row = dict(driver.run_trial(cell))

    # The row is complete and CSV-shaped.
    assert set(row.keys()) == set(Exp4MetricSummary.csv_columns())

    # The real two-pass cycle ran: at least one mission completed and the
    # cluster closed at least one round (i.e. real cross-mule FedAvg fired).
    assert row["missions_completed"] >= 1, (
        f"no mission completed in the real run: {row!r}"
    )
    assert row["rounds_closed"] >= 1, (
        f"cluster never closed a round (FedAvg never ran): {row!r}"
    )
    assert row["mission_failures"] == 0
    # With 2 devices in a tight cluster, Pass 1 should collect ≥1 update.
    assert row["update_yield"] >= 1.0
    assert 0.0 <= row["coverage"] <= 1.0
    assert row["n_devices"] == 2
