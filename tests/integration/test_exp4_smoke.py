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

from experiments.exp4.driver import PROVENANCE_COLUMNS, Exp4Driver
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
    assert set(row.keys()) == set(Exp4MetricSummary.csv_columns()) | set(PROVENANCE_COLUMNS)

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
    assert row["aggregation"] == "agg:plain" and row["aggregation_params"] == ""
    assert row["fedprox_rho"] == 0.0 and row["pass_2_budget"] == 0
    assert row["deadline_law"] == "additive" and row["deadline_params"] == ""
    assert row["miss_priority"] == 0


@pytest.mark.slow
def test_exp4_age_aware_merge_runs_through_the_real_orchestrator(tmp_path):
    """FeRRy Phase 1: agg:cutoff, a deadline-derived cutoff, a budgeted Pass 2,
    the multiplicative deadline law, the miss-priority key and FedProx, across
    processes and TCP. Every setting is read back from the processes' own
    events, not from the driver's row, so a flag dropped between the driver
    and the process fails here (audit #15)."""
    import itertools
    import json

    from experiments.exp4.driver import trace_dir_name

    driver = Exp4Driver(
        default_n_devices=3,
        default_rf_range_m=60.0,
        default_n_missions=2,
        trial_budget_s=120.0,
        startup_timeout_s=30.0,
        mission_budget_s=60.0,
        aggregation="agg:cutoff",
        # T = 30 s with no fixed a_max: the cap ⌊Φ/T⌋ can only come from
        # period_s reaching the mule.
        aggregation_params={"period_s": 30.0},
        pass_2_budget=True,
        deadline_law="multiplicative",
        miss_priority=True,
        fedprox_rho=0.01,
        trace_root=tmp_path,
    )
    cell = Cell(
        cell_id="phase1-smoke", arm="H1", trial_index=0, seed=4242,
        params={"N": 3, "rrf": 60.0, "n_missions": 2},
    )
    row = dict(driver.run_trial(cell))
    assert row["mission_failures"] == 0 and row["rounds_closed"] >= 1

    trace = tmp_path / trace_dir_name(cell)

    def _events(pattern, name):
        out = []
        for path in trace.glob(pattern):
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("event") == name:
                    out.append(rec)
        return out

    # What the mule actually runs, reported by the mule process itself.
    (ready,) = _events("mule-*.jsonl", "mule_ready")
    assert ready["aggregation"] == "agg:cutoff"
    assert ready["aggregation_params"]["period_s"] == 30.0
    assert ready["aggregation_params"]["a_max"] is None
    assert ready["deadline_law"] == "multiplicative"
    assert ready["deadline_params"]["beta_on"] == 0.8
    assert ready["miss_priority"] is True and ready["pass_2_budget"] is True
    assert ready["mission_budget_s"] == 60.0
    devices_ready = _events("device-*.jsonl", "device_ready")
    assert len(devices_ready) == 3
    assert all(d["fedprox_rho"] == 0.01 for d in devices_ready)

    merges, windows = [], []
    for rec in _events("mule-*.jsonl", "mission_completed"):
        if rec.get("pass_1_merge"):
            merges.append(rec["pass_1_merge"])
        windows.extend((rec.get("deadline_state") or {}).values())
        # Audit #3/#13 trace fields.
        assert isinstance(rec.get("pass_1_merged_devices"), list)
        for contact in rec.get("pass_1_plan") or ():
            assert set(contact["device_deadlines"]) == set(contact["devices"])
    assert merges, "no mission_completed event carried a merge record"
    assert all(m["rule"] == "agg:cutoff" for m in merges)
    assert merges[0]["base_version"] == 0
    assert all(a is not None and a >= 0 for m in merges for a in m["ages"])

    # The law's trajectory. Two missions give each device at most two folds,
    # so every window is 60·β₁·β₂ for β in {β_on, β_partial, β_timeout}. The
    # additive law's ±5/10 s steps reach none of these but 60.
    assert windows, "no mission_completed event carried the deadline state"
    betas = (1.0, 0.8, 1.25, 1.5)
    reachable = {round(60.0 * a * b, 6) for a, b in itertools.product(betas, betas)}
    assert all(round(w["phi_s"], 6) in reachable for w in windows), windows
    assert any(round(w["phi_s"], 6) != 60.0 for w in windows)
    assert all(w["miss_streak"] >= 0 for w in windows)
