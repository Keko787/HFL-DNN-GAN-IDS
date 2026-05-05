"""Chunk EX-0 — trial harness unit tests.

Pins the contracts the experiments plan §2 calls out:

* Cartesian-product grid generation is deterministic.
* Paired seeds: same (cell_id, trial_index) → same seed across arms.
* CSV append is idempotent on (cell_id, arm, trial_index).
* Resume = read existing CSV, skip already-done.
* Failing one trial doesn't break subsequent trials.
* Schema-mismatch on resume is rejected (no silent column drift).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import pytest

from experiments.runner import Cell, CSVTrialLog, TrialGrid, TrialRunner


# --------------------------------------------------------------------------- #
# TrialGrid
# --------------------------------------------------------------------------- #

def test_grid_total_matches_iteration_count():
    grid = TrialGrid(
        independent_vars={"a": [1, 2, 3], "b": ["x", "y"]},
        arms=["FL", "Cent"],
        n_trials=5,
        base_seed=42,
    )
    cells = list(grid)
    assert len(cells) == grid.total() == 3 * 2 * 5 * 2  # = 60


def test_grid_iteration_is_deterministic():
    grid = TrialGrid(
        independent_vars={"a": [1, 2], "b": ["x", "y"]},
        arms=["FL", "Cent"],
        n_trials=3,
        base_seed=42,
    )
    a = [(c.cell_id, c.arm, c.trial_index, c.seed) for c in grid]
    b = [(c.cell_id, c.arm, c.trial_index, c.seed) for c in grid]
    assert a == b


def test_grid_paired_seeds_match_across_arms():
    """Same (cell_id, trial_index) must produce the same seed on every arm.

    This is the foundation of paired statistical tests: if FL trial 7
    and Centralized trial 7 saw different seeds, the per-trial difference
    isn't a paired observation.
    """
    grid = TrialGrid(
        independent_vars={"Dpd": ["10MB", "100MB"], "alpha": [0.5, 1.0]},
        arms=["FL", "Centralized"],
        n_trials=4,
        base_seed=123,
    )
    by_pair: dict[tuple[str, int], dict[str, int]] = {}
    for c in grid:
        by_pair.setdefault((c.cell_id, c.trial_index), {})[c.arm] = c.seed

    for (cid, ti), arm_seeds in by_pair.items():
        seeds = list(arm_seeds.values())
        assert seeds[0] == seeds[1], (
            f"paired-seed broken at cell={cid} trial={ti}: arms got {arm_seeds}"
        )


def test_grid_seed_changes_with_trial_index():
    """Different trial indices within a cell should produce different seeds."""
    grid = TrialGrid(
        independent_vars={"a": [1]}, arms=["x"], n_trials=10, base_seed=0,
    )
    seeds = {c.seed for c in grid}
    assert len(seeds) == 10, "trial indices should produce distinct seeds"


def test_grid_seed_changes_with_base_seed():
    grid_a = TrialGrid(independent_vars={"a": [1]}, arms=["x"], n_trials=3, base_seed=0)
    grid_b = TrialGrid(independent_vars={"a": [1]}, arms=["x"], n_trials=3, base_seed=999)
    seeds_a = [c.seed for c in grid_a]
    seeds_b = [c.seed for c in grid_b]
    assert seeds_a != seeds_b


def test_grid_filter_params_narrows_emission():
    grid = TrialGrid(
        independent_vars={"a": [1, 2], "b": ["x", "y"]},
        arms=["A"],
        n_trials=2,
        filter_params={"a": 1},
    )
    cells = list(grid)
    assert all(c.params["a"] == 1 for c in cells)
    assert {c.params["b"] for c in cells} == {"x", "y"}
    assert len(cells) == 2 * 2  # 1 × 2 b-values × 2 trials × 1 arm


def test_grid_rejects_zero_n_trials():
    with pytest.raises(ValueError):
        TrialGrid(independent_vars={"a": [1]}, arms=["x"], n_trials=0)


def test_grid_rejects_empty_arms():
    with pytest.raises(ValueError):
        TrialGrid(independent_vars={"a": [1]}, arms=[], n_trials=1)


def test_grid_rejects_empty_value_list():
    with pytest.raises(ValueError, match="empty value list"):
        TrialGrid(independent_vars={"a": []}, arms=["x"], n_trials=1)


def test_grid_singleton_when_no_independent_vars():
    grid = TrialGrid(independent_vars={}, arms=["x"], n_trials=2)
    cells = list(grid)
    assert len(cells) == 2
    assert {c.cell_id for c in cells} == {"_singleton_"}


def test_grid_cell_id_is_dict_order_independent():
    """Same params with different insertion order must hash to the same cell_id."""
    g1 = TrialGrid(independent_vars={"a": [1], "b": ["x"]}, arms=["A"], n_trials=1)
    g2 = TrialGrid(independent_vars={"b": ["x"], "a": [1]}, arms=["A"], n_trials=1)
    c1 = next(iter(g1))
    c2 = next(iter(g2))
    assert c1.cell_id == c2.cell_id


# --------------------------------------------------------------------------- #
# CSVTrialLog
# --------------------------------------------------------------------------- #

def _make_cell(*, cell_id="a=1", arm="FL", trial_index=0, seed=99) -> Cell:
    return Cell(
        cell_id=cell_id, arm=arm, trial_index=trial_index, seed=seed,
        params={"a": 1},
    )


def _basic_fields():
    return [
        "cell_id", "arm", "trial_index", "seed", "param_a",
        "metric_x", "status", "duration_s", "error",
    ]


def test_csv_append_writes_one_row_per_call(tmp_path):
    log = CSVTrialLog(tmp_path / "out.csv", fieldnames=_basic_fields())
    log.append(_make_cell(trial_index=0), {"metric_x": 1.0, "status": "ok",
                                            "duration_s": 0.1, "error": ""})
    log.append(_make_cell(trial_index=1), {"metric_x": 2.0, "status": "ok",
                                            "duration_s": 0.2, "error": ""})
    rows = list(csv.DictReader((tmp_path / "out.csv").open()))
    assert len(rows) == 2
    assert rows[0]["metric_x"] == "1.0"
    assert rows[1]["trial_index"] == "1"


def test_csv_append_is_idempotent_on_repeat(tmp_path):
    log = CSVTrialLog(tmp_path / "out.csv", fieldnames=_basic_fields())
    cell = _make_cell()
    log.append(cell, {"metric_x": 1.0, "status": "ok", "duration_s": 0.1, "error": ""})
    log.append(cell, {"metric_x": 2.0, "status": "ok", "duration_s": 0.1, "error": ""})
    rows = list(csv.DictReader((tmp_path / "out.csv").open()))
    assert len(rows) == 1, "second append for same cell should be a no-op"
    assert rows[0]["metric_x"] == "1.0", "first row's value must be preserved"


def test_csv_resume_loads_done_set(tmp_path):
    """A new log instance over an existing file must skip already-done cells."""
    path = tmp_path / "out.csv"
    log1 = CSVTrialLog(path, fieldnames=_basic_fields())
    log1.append(_make_cell(trial_index=0), {"metric_x": 1.0, "status": "ok",
                                              "duration_s": 0.1, "error": ""})
    log1.append(_make_cell(trial_index=1), {"metric_x": 2.0, "status": "ok",
                                              "duration_s": 0.1, "error": ""})

    log2 = CSVTrialLog(path, fieldnames=_basic_fields())
    assert log2.already_done(_make_cell(trial_index=0))
    assert log2.already_done(_make_cell(trial_index=1))
    assert not log2.already_done(_make_cell(trial_index=2))


def test_csv_resume_rejects_missing_key_columns(tmp_path):
    """A pre-existing file without the (cell_id, arm, trial_index) keys
    is malformed for our purposes — refuse to silently keep going."""
    path = tmp_path / "out.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["foo", "bar"])
        w.writerow(["1", "2"])
    with pytest.raises(ValueError, match="missing required key column"):
        CSVTrialLog(path, fieldnames=_basic_fields())


def test_csv_unknown_column_in_row_raises(tmp_path):
    """A driver returning a column not in the schema is a typo; surface it."""
    log = CSVTrialLog(tmp_path / "out.csv", fieldnames=_basic_fields())
    with pytest.raises(ValueError, match="not in the CSV schema"):
        log.append(_make_cell(), {"metric_x": 1.0, "typoed_field": 0.0,
                                   "status": "ok", "duration_s": 0.1, "error": ""})


def test_csv_cell_metadata_overrides_row(tmp_path):
    """A driver can't accidentally overwrite cell_id / seed / etc."""
    log = CSVTrialLog(tmp_path / "out.csv", fieldnames=_basic_fields())
    cell = _make_cell(cell_id="legit", seed=12345)
    log.append(cell, {"cell_id": "TRYING_TO_OVERWRITE", "seed": 0,
                       "metric_x": 1.0, "status": "ok",
                       "duration_s": 0.1, "error": ""})
    rows = list(csv.DictReader((tmp_path / "out.csv").open()))
    assert rows[0]["cell_id"] == "legit"
    assert rows[0]["seed"] == "12345"


def test_csv_schema_change_allowed_when_flag_set(tmp_path):
    """allow_schema_change=True lets a new column join the file."""
    path = tmp_path / "out.csv"
    log1 = CSVTrialLog(path, fieldnames=_basic_fields())
    log1.append(_make_cell(trial_index=0), {"metric_x": 1.0, "status": "ok",
                                              "duration_s": 0.1, "error": ""})

    extended = _basic_fields() + ["new_metric"]
    log2 = CSVTrialLog(path, fieldnames=extended, allow_schema_change=True)
    log2.append(_make_cell(trial_index=1),
                {"metric_x": 2.0, "new_metric": 9.0, "status": "ok",
                 "duration_s": 0.1, "error": ""})

    rows = list(csv.DictReader(path.open()))
    assert "new_metric" in rows[0].keys()
    assert rows[1]["new_metric"] == "9.0"
    # Pre-extension row should have empty new_metric.
    assert rows[0]["new_metric"] == ""


# --------------------------------------------------------------------------- #
# TrialRunner
# --------------------------------------------------------------------------- #

def _runner_grid() -> TrialGrid:
    return TrialGrid(
        independent_vars={"x": [1, 2]},
        arms=["A", "B"],
        n_trials=3,
        base_seed=7,
    )


def test_runner_writes_one_row_per_cell(tmp_path):
    grid = _runner_grid()
    runner = TrialRunner(grid, tmp_path / "out.csv",
                         metric_columns=["metric"])

    def driver(cell: Cell) -> Mapping[str, Any]:
        return {"metric": cell.seed % 100}

    written = runner.run(driver)
    assert written == 2 * 3 * 2  # x∈{1,2} × n_trials=3 × arms=2 = 12
    rows = list(csv.DictReader((tmp_path / "out.csv").open()))
    assert len(rows) == 12
    assert all(r["status"] == "ok" for r in rows)


def test_runner_skips_already_done_cells_on_resume(tmp_path):
    grid = _runner_grid()
    runner = TrialRunner(grid, tmp_path / "out.csv", metric_columns=["metric"])

    def driver(cell: Cell) -> Mapping[str, Any]:
        return {"metric": 1.0}

    runner.run(driver)
    # Re-run; nothing new should be written.
    runner2 = TrialRunner(grid, tmp_path / "out.csv", metric_columns=["metric"])
    written = runner2.run(driver)
    assert written == 0


def test_runner_failing_trial_does_not_abort_the_loop(tmp_path):
    grid = TrialGrid(independent_vars={"x": [1, 2, 3]}, arms=["A"], n_trials=1)
    runner = TrialRunner(grid, tmp_path / "out.csv", metric_columns=["metric"])

    fail_on = {2}

    def driver(cell: Cell) -> Mapping[str, Any]:
        if cell.params["x"] in fail_on:
            raise RuntimeError("simulated failure")
        return {"metric": float(cell.params["x"])}

    runner.run(driver)
    rows = list(csv.DictReader((tmp_path / "out.csv").open()))
    statuses = {r["param_x"]: r["status"] for r in rows}
    assert statuses == {"1": "ok", "2": "error", "3": "ok"}


def test_runner_records_error_traceback_summary(tmp_path):
    grid = TrialGrid(independent_vars={"x": [1]}, arms=["A"], n_trials=1)
    runner = TrialRunner(grid, tmp_path / "out.csv", metric_columns=["metric"])

    def driver(_cell: Cell) -> Mapping[str, Any]:
        raise ValueError("kaboom on purpose")

    runner.run(driver)
    rows = list(csv.DictReader((tmp_path / "out.csv").open()))
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "kaboom" in rows[0]["error"]


def test_runner_soft_timeout_marks_status(tmp_path, monkeypatch):
    grid = TrialGrid(independent_vars={"x": [1]}, arms=["A"], n_trials=1)
    # Fake clock: each call returns the next value in a list so the
    # measured duration is deterministic.
    times = iter([100.0, 105.0])
    runner = TrialRunner(
        grid, tmp_path / "out.csv",
        metric_columns=["metric"],
        timeout_s=2.0,
        clock=lambda: next(times),
    )

    def driver(_cell: Cell) -> Mapping[str, Any]:
        return {"metric": 1.0}

    runner.run(driver)
    rows = list(csv.DictReader((tmp_path / "out.csv").open()))
    assert rows[0]["status"] == "timeout"
    assert "5.0s" in rows[0]["error"]
    assert "2.0s" in rows[0]["error"]


def test_runner_iter_outcomes_reports_per_trial(tmp_path):
    grid = TrialGrid(independent_vars={"x": [1, 2]}, arms=["A"], n_trials=1)
    runner = TrialRunner(grid, tmp_path / "out.csv", metric_columns=["metric"])

    def driver(cell: Cell) -> Mapping[str, Any]:
        return {"metric": float(cell.params["x"])}

    outcomes = list(runner.iter_outcomes(driver))
    assert [o.status for o in outcomes] == ["ok", "ok"]
    assert all(o.duration_s >= 0.0 for o in outcomes)
    assert all(o.row is not None and o.row["metric"] == float(o.cell.params["x"])
               for o in outcomes)
