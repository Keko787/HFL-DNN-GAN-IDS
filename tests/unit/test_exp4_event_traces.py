"""Event-trace retention (`--keep-event-traces`).

Why this exists: the per-contact record lives in the orchestrator's run dir,
`consume_run_dir` folds it into aggregates, and teardown deletes it. So a
finished sweep cannot be re-scored against a new scheduling baseline — there is
nothing left to replay, and answering "how would policy X have done?" costs a
full re-run. Retention is the cheap insurance against that.

The naming is where the real failure modes are. Cell ids are built from the grid
axes and contain `|` and `=` (`N=6|dead_zone=0.0|regime=jittery`), so a directory
named after one is unopenable on Windows; and two trials sharing a directory
would interleave their events and quietly corrupt both.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from experiments.exp4 import driver as driver_module
from experiments.exp4.driver import (
    TRIAL_STATUS_FILE,
    Exp4Driver,
    Exp4TrialTimeout,
    trace_dir_name,
)
from experiments.runner import Cell
from hermes.processes.config import device_config_to_json, mule_config_to_json


class _Cell:
    def __init__(self, cell_id, arm="H1", trial_index=0, seed=42):
        self.cell_id = cell_id
        self.arm = arm
        self.trial_index = trial_index
        self.seed = seed


REAL_CELL_ID = "N=6|dead_zone=0.0|link_quality=0.3|n_missions=4|regime=jittery|rrf=60.0"


# --------------------------------------------------------------------------- #
# 1. The name must be usable as a path
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ch", list('<>:"/\\|?*'))
def test_windows_illegal_characters_are_removed(ch):
    name = trace_dir_name(_Cell(f"a{ch}b"))
    assert ch not in name


def test_a_real_cell_id_survives_sanitisation():
    """The pipe characters in a genuine cell id are the whole problem."""
    name = trace_dir_name(_Cell(REAL_CELL_ID))
    assert "|" not in name
    assert "N=6" in name and "regime=jittery" in name


def test_the_directory_can_actually_be_created(tmp_path):
    """The real test of a filename is whether the filesystem accepts it."""
    d = tmp_path / trace_dir_name(_Cell(REAL_CELL_ID))
    d.mkdir(parents=True)
    assert d.is_dir()


def test_no_trailing_dot_or_space():
    """Windows silently rejects both."""
    name = trace_dir_name(_Cell("trailing. "))
    assert not name.endswith((".", " "))


# --------------------------------------------------------------------------- #
# 2. The name must identify the row it came from, uniquely
# --------------------------------------------------------------------------- #

def test_name_encodes_arm_trial_and_seed():
    """A trace nobody can match back to a CSV row is worthless."""
    name = trace_dir_name(_Cell("c", arm="H3", trial_index=7, seed=999))
    assert "H3" in name and "t7" in name and "s999" in name


@pytest.mark.parametrize("a,b", [
    (_Cell("c", arm="H1"), _Cell("c", arm="H2")),
    (_Cell("c", trial_index=0), _Cell("c", trial_index=1)),
    (_Cell("c", seed=1), _Cell("c", seed=2)),
    (_Cell("c1"), _Cell("c2")),
])
def test_trials_that_differ_get_different_directories(a, b):
    """Sharing a directory would interleave two trials' events."""
    assert trace_dir_name(a) != trace_dir_name(b)


def test_long_names_are_truncated_but_stay_unique():
    """Truncation must not turn 'too long' into 'silently collides'."""
    base = "axis=value|" * 40
    a = trace_dir_name(_Cell(base + "one"))
    b = trace_dir_name(_Cell(base + "two"))
    assert len(a) <= 120 and len(b) <= 120
    assert a != b


def test_the_same_cell_always_yields_the_same_name():
    """Resuming a sweep must land traces in the same place."""
    assert trace_dir_name(_Cell(REAL_CELL_ID)) == trace_dir_name(_Cell(REAL_CELL_ID))


# --------------------------------------------------------------------------- #
# 3. Capture behaviour
# --------------------------------------------------------------------------- #

def _run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "mule-m1.jsonl").write_text('{"ts": 1, "event": "device_served"}\n')
    (d / "device-d0.jsonl").write_text('{"ts": 2, "event": "device_ready"}\n')
    (d / "device-d0.json").write_text('{"position": [1, 2, 0]}')
    (d / "cluster.port").write_text("5000")
    return d


def test_disabled_by_default_nothing_is_written(tmp_path):
    """Off is the default; every committed sweep ran that way."""
    driver = Exp4Driver()
    assert driver.trace_root is None
    driver._capture_traces(_run_dir(tmp_path), _Cell("c"))
    assert not (tmp_path / "traces").exists()


def test_events_and_configs_are_both_kept(tmp_path):
    """Configs carry device positions; the events do not.

    Without positions, no spatial policy (MAX-AoI's nearest-predecessor
    pathing, any travel-cost rule) can be scored from the trace.
    """
    root = tmp_path / "traces"
    Exp4Driver(trace_root=root)._capture_traces(_run_dir(tmp_path), _Cell("c"))

    kept = {p.name for p in (root / trace_dir_name(_Cell("c"))).iterdir()}
    assert "mule-m1.jsonl" in kept, "per-contact events are the point"
    assert "device-d0.json" in kept, "positions live only in the configs"


def test_trace_content_is_preserved_verbatim(tmp_path):
    root = tmp_path / "traces"
    Exp4Driver(trace_root=root)._capture_traces(_run_dir(tmp_path), _Cell("c"))
    got = (root / trace_dir_name(_Cell("c")) / "mule-m1.jsonl").read_text()
    assert got == '{"ts": 1, "event": "device_served"}\n'


def test_two_trials_do_not_share_a_directory(tmp_path):
    root = tmp_path / "traces"
    driver = Exp4Driver(trace_root=root)
    src = _run_dir(tmp_path)
    driver._capture_traces(src, _Cell(REAL_CELL_ID, trial_index=0))
    driver._capture_traces(src, _Cell(REAL_CELL_ID, trial_index=1))
    assert len(list(root.iterdir())) == 2


def test_a_capture_failure_never_kills_the_trial(tmp_path):
    """Bookkeeping is not worth losing a 70-second real-model trial over."""
    driver = Exp4Driver(trace_root=tmp_path / "traces")
    driver._capture_traces(tmp_path / "does-not-exist", _Cell("c"))  # no raise


def test_capture_is_idempotent_for_a_rerun_of_the_same_cell(tmp_path):
    """Re-running a cell overwrites its trace rather than failing on mkdir."""
    root = tmp_path / "traces"
    driver = Exp4Driver(trace_root=root)
    src = _run_dir(tmp_path)
    driver._capture_traces(src, _Cell("c"))
    driver._capture_traces(src, _Cell("c"))  # no raise
    assert (root / trace_dir_name(_Cell("c")) / "mule-m1.jsonl").exists()


# --------------------------------------------------------------------------- #
# 4. The trial-status marker (FeRRy audit #11)
# --------------------------------------------------------------------------- #
#
# A kept trace is re-scored from its directory alone, so without a marker a
# timed-out or no_eval trial is indistinguishable from a good one. The trial
# runs end to end below on a stand-in orchestrator that writes the real
# topology's configs and a canned event stream instead of spawning anything.

TRIAL_CELL = Cell(
    cell_id="N=2|n_missions=2|regime=clean|rrf=60.0", arm="H1", trial_index=3,
    seed=77, params={"N": 2, "rrf": 60.0, "n_missions": 2, "regime": "clean"},
)


def _events(topo, *, evaluated=True):
    mule = topo.mules[0].mule_id
    mule_rows = [{"ts": 1.0, "event": "mule_ready"}, {"ts": 2.0, "event": "dock_bootstrapped"}]
    cluster_rows = [{"ts": 1.0, "event": "cluster_ready"}]
    for i in range(2):
        t = 10.0 * (i + 1)
        mule_rows += [
            {"ts": t, "event": "mission_started", "id": mule, "mission_index": i},
            {"ts": t + 5, "event": "mission_completed", "id": mule, "mission_round": i + 1,
             "pass_1_contacts": 1, "pass_2_contacts": 1, "pass_1_updates": 2,
             "pass_1_scheduled": 2, "pass_1_clean_devices": [d.device_id for d in topo.devices],
             "delivered": 2, "undelivered": 0, "duration_s": 5.0},
        ]
        cluster_rows.append({"ts": t + 2, "event": "cluster_round_closed", "cluster_round": i + 1})
        if evaluated:
            cluster_rows.append({"ts": t + 3, "event": "model_eval", "cluster_round": i + 1,
                                 "accuracy": 0.8, "auc": 0.8, "loss": 0.5, "n_test": 10})
    return {f"mule-{mule}.jsonl": mule_rows, "cluster-c.jsonl": cluster_rows}


def _fake_orchestrator(tmp_path, *, evaluated=True, startup_s=0.0):
    class FakeOrchestrator:
        def __init__(self, topo, capture_output=True):
            self.tmpdir = tmp_path / f"run{len(list(tmp_path.glob('run*')))}"
            self.tmpdir.mkdir()
            self.mule_handles = {}
            for m in topo.mules:
                (self.tmpdir / f"mule-{m.mule_id}.json").write_text(mule_config_to_json(m))
            for d in topo.devices:
                (self.tmpdir / f"device-{d.device_id}.json").write_text(device_config_to_json(d))
            for name, rows in _events(topo, evaluated=evaluated).items():
                (self.tmpdir / name).write_text("".join(json.dumps(r) + "\n" for r in rows))

        def start_all(self, timeout):
            time.sleep(startup_s)

        def shutdown_all(self, timeout, cleanup_tmpdir):
            pass

        def cleanup(self):
            pass

    return FakeOrchestrator


def _marker(root, cell=TRIAL_CELL):
    return json.loads((root / trace_dir_name(cell) / TRIAL_STATUS_FILE).read_text())


def test_an_ok_trial_is_marked_ok_and_its_row_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(driver_module, "MultiProcessOrchestrator", _fake_orchestrator(tmp_path))
    untraced = Exp4Driver(default_n_missions=2).run_trial(TRIAL_CELL)
    root = tmp_path / "traces"
    traced = Exp4Driver(default_n_missions=2, trace_root=root).run_trial(TRIAL_CELL)
    assert traced == untraced
    assert "status" not in traced                   # the runner stamps ok itself
    marker = _marker(root)
    assert 0.0 <= marker.pop("run_s") < 120.0
    assert marker == {"status": "ok", "error": "", "n_missions_target": 2,
                      "trial_budget_s": 120.0}


def test_a_no_eval_trial_is_marked_as_its_row_is(tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver_module, "MultiProcessOrchestrator", _fake_orchestrator(tmp_path, evaluated=False),
    )
    root = tmp_path / "traces"
    row = Exp4Driver(real_model=True, trace_root=root)._run_topology(
        driver_module.build_exp4_topology(n_devices=2, rf_range_m=60.0, n_missions=2, seed=77),
        cell=TRIAL_CELL, n_devices=2, rf_range_m=60.0, n_missions=2,
    )
    assert row["status"] == "no_eval"
    # Not started through run_trial, so there is no run time to record.
    assert _marker(root) == {"status": "no_eval", "error": row["error"], "n_missions_target": 2,
                             "run_s": None, "trial_budget_s": 120.0}


def test_a_timed_out_trial_is_marked_as_the_runner_records_it(tmp_path, monkeypatch):
    """The runner writes any raise as status=error with its last line."""
    monkeypatch.setattr(driver_module, "MultiProcessOrchestrator", _fake_orchestrator(tmp_path))
    root = tmp_path / "traces"
    driver = Exp4Driver(default_n_missions=2, trace_root=root)
    monkeypatch.setattr(driver, "_await_mules", lambda orch, budget_s: False)
    with pytest.raises(Exp4TrialTimeout):
        driver.run_trial(TRIAL_CELL)
    marker = _marker(root)
    assert marker["status"] == "error"
    assert marker["error"].startswith("experiments.exp4.driver.Exp4TrialTimeout: exp4 trial exceeded")
    assert marker["run_s"] >= 0.0


@pytest.mark.parametrize("late", [False, True], ids=["in-time", "past-soft-cap"])
def test_a_trial_the_runner_relabels_timeout_is_scored_timeout(tmp_path, monkeypatch, late):
    """The runner relabels a trial that returned past its soft cap only after
    the marker is written; the marker keeps the run time and budget, so the
    scorer reaches the CSV row's verdict with or without the CSV."""
    from experiments.analysis.traces_scorer import trial_status
    from experiments.exp4.driver import PROVENANCE_COLUMNS
    from experiments.exp4.metrics import Exp4MetricSummary
    from experiments.runner import TrialGrid, TrialRunner

    # The stand-in never waits on a mule, so the trial's length is its startup:
    # 0.2 s, well clear of Windows' 15.6 ms clock tick on both sides of 0.05 s.
    monkeypatch.setattr(
        driver_module, "MultiProcessOrchestrator", _fake_orchestrator(tmp_path, startup_s=0.2),
    )
    budget = 0.05 if late else 120.0
    grid = TrialGrid(
        independent_vars={"N": [2], "rrf": [60.0], "n_missions": [2], "regime": ["clean"]},
        arms=["H1"], n_trials=1,
    )
    trial_csv = tmp_path / "trials.csv"
    runner = TrialRunner(
        grid, trial_csv,
        metric_columns=list(Exp4MetricSummary.csv_columns()) + list(PROVENANCE_COLUMNS),
        timeout_s=budget,                # runner_main's default: the trial budget
    )
    root = tmp_path / "trials_traces"
    (outcome,) = runner.iter_outcomes(Exp4Driver(trial_budget_s=budget, trace_root=root).run_trial)
    expected = "timeout" if late else "ok"
    assert outcome.status == expected

    cell = next(iter(grid))
    trace = root / trace_dir_name(cell)
    marker = _marker(root, cell)
    assert marker["status"] == "ok"      # what the driver handed the runner
    assert (marker["run_s"] > marker["trial_budget_s"]) == late
    alone = trial_status(trace)
    assert (alone.status, alone.source) == (expected, "soft_cap" if late else "marker")
    joined = trial_status(trace, trial_csv)
    assert (joined.status, joined.source) == (expected, "csv" if late else "marker")


def test_no_marker_without_a_kept_trace(tmp_path):
    Exp4Driver()._write_trial_status(_Cell("c"), status="ok", error="", n_missions=2)
    driver = Exp4Driver(trace_root=tmp_path / "traces")
    driver._write_trial_status(_Cell("c"), status="ok", error="", n_missions=2)
    assert not (tmp_path / "traces").exists()


def test_a_marker_failure_never_kills_the_trial(tmp_path):
    root = tmp_path / "traces"
    (root / trace_dir_name(_Cell("c")) / TRIAL_STATUS_FILE).mkdir(parents=True)  # unwritable
    Exp4Driver(trace_root=root)._write_trial_status(
        _Cell("c"), status="ok", error="", n_missions=2,
    )  # no raise


def test_the_kept_trace_scores_with_the_rows_provenance(tmp_path, monkeypatch):
    """Scorer and driver rows join: same provenance columns, same values."""
    from experiments.analysis.traces_scorer import score_trial
    from experiments.exp4.driver import PROVENANCE_COLUMNS

    monkeypatch.setattr(driver_module, "MultiProcessOrchestrator", _fake_orchestrator(tmp_path))
    root = tmp_path / "traces"
    driver = Exp4Driver(
        default_n_missions=2, trace_root=root,
        aggregation="agg:fedbuff", aggregation_params={"buffer_k": 2},
        deadline_law="multiplicative", deadline_params={"beta_on": 0.7},
        fedprox_rho=0.01, mission_budget_s=90.0, pass_2_budget=True, miss_priority=True,
        mission_window_adaptation=True,
    )
    row = driver.run_trial(TRIAL_CELL)
    score = score_trial(root / trace_dir_name(TRIAL_CELL))
    assert score.status == "ok"
    assert {c: score.to_row()[c] for c in PROVENANCE_COLUMNS} == {
        c: row[c] for c in PROVENANCE_COLUMNS
    }


# --------------------------------------------------------------------------- #
# 5. Several mules (FeRRy Phase 2)
# --------------------------------------------------------------------------- #
#
# The orchestrator writes each mule's slice into its config
# (``mule-<id>.json``'s ``expected_devices``), and the kept trace keeps the
# configs, so both the driver's row and a later re-score can take each
# mission's Pass-2 coverage against its own mule's slice.

def _two_mule_topology():
    from dataclasses import replace

    topo = driver_module.build_exp4_topology(n_devices=4, rf_range_m=60.0, n_missions=2, seed=77)
    devices = [d.device_id for d in topo.devices]
    base = topo.mules[0]
    topo.mules = [
        replace(base, mule_id="exp4-mule-0", expected_devices=devices[:2]),
        replace(base, mule_id="exp4-mule-1", expected_devices=devices[2:]),
    ]
    return topo


def _two_mule_orchestrator(tmp_path):
    """Mule 0 delivers Pass 2 to its whole slice every mission, mule 1 to half."""
    class FakeOrchestrator:
        def __init__(self, topo, capture_output=True):
            self.tmpdir = tmp_path / "run-two-mules"
            self.tmpdir.mkdir()
            self.mule_handles = {}
            cluster_rows, closed = [{"ts": 1.0, "event": "cluster_ready"}], 0
            for k, m in enumerate(topo.mules):
                (self.tmpdir / f"mule-{m.mule_id}.json").write_text(mule_config_to_json(m))
                rows = []
                for i in range(2):
                    t = 10.0 * (i + 1) + k
                    rows += [
                        {"ts": t, "event": "mission_started", "id": m.mule_id},
                        {"ts": t + 5, "event": "mission_completed", "id": m.mule_id,
                         "mission_round": i + 1, "pass_1_contacts": 1, "pass_2_contacts": 1,
                         "pass_1_updates": 2, "pass_1_scheduled": 2,
                         "pass_1_clean_devices": list(m.expected_devices),
                         "delivered": 2 - k, "undelivered": k, "duration_s": 5.0},
                    ]
                    closed += 1
                    cluster_rows += [
                        {"ts": t + 2, "event": "up_bundle_ingested", "mule_id": m.mule_id,
                         "mission_round": i + 1},
                        {"ts": t + 2, "event": "cluster_round_closed", "cluster_round": closed},
                    ]
                (self.tmpdir / f"mule-{m.mule_id}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
            for d in topo.devices:
                (self.tmpdir / f"device-{d.device_id}.json").write_text(device_config_to_json(d))
            (self.tmpdir / "cluster-c.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in cluster_rows)
            )

        def start_all(self, timeout):
            pass

        def shutdown_all(self, timeout, cleanup_tmpdir):
            pass

        def cleanup(self):
            pass

    return FakeOrchestrator


def test_a_two_mule_trial_measures_pass_2_against_each_mules_slice(tmp_path, monkeypatch):
    from experiments.analysis.traces_scorer import score_trial

    monkeypatch.setattr(
        driver_module, "MultiProcessOrchestrator", _two_mule_orchestrator(tmp_path),
    )
    root = tmp_path / "traces"
    row = Exp4Driver(trace_root=root)._run_topology(
        _two_mule_topology(), cell=TRIAL_CELL, n_devices=4, rf_range_m=60.0, n_missions=2,
    )
    # 2/2, 2/2, 1/2, 1/2; against all four devices it would be 0.375.
    assert row["pass2_coverage"] == pytest.approx(0.75)
    score = score_trial(root / trace_dir_name(TRIAL_CELL))
    assert score.n_mules == 2 and score.to_row()["n_mules"] == 2
    assert score.summary.pass2_coverage == pytest.approx(0.75)
    assert score.summary.round_close_rate_kmin1 == pytest.approx(1.0)
