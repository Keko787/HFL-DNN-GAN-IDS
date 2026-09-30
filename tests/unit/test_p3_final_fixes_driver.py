"""FeRRy Phase 3, final check — the driver group's two fixes.

* **The runner's soft cap reaches the status marker** (final check, legacy
  F1). On the mission clock each cell's wall budget is re-costed for the
  session TTL (critic B14), but the runner applies one soft cap to every trial:
  the largest budget over the grid, or ``--timeout-s``. The marker recorded
  only the trial's own budget, so a trace scored without its trial CSV was
  relabelled ``timeout`` by the scorer when it returned between its own
  budget and the runner's cap, a trial the CSV records ``ok``. ``runner_main``
  now tells the driver the cap (``Exp4Driver.soft_cap_s``), a simulated-clock
  marker records it as ``soft_cap_s``, and the scorer compares the run time
  with it when present. Wall markers keep their recorded key set, and a
  simulated-clock marker without the cap keeps the recorded rule.
* **The law's time unit is refused in ``deadline_params``** (final check,
  legacy notes item 8). ``DeadlineLaw`` gained ``time_scale``, so
  ``deadline_params={"time_scale": ...}`` would run the scheduler at that
  unit with the ``deadline_time_scale`` column blank, and beside
  ``deadline_time_scale`` it failed only at mule start. afa9526 refused the
  key at construction, in any shape ``dict()`` takes, with
  ``DeadlineLawError``; the driver refuses it again, in the same shapes and
  with the same type, pointing to ``deadline_time_scale``
  (``--deadline-time-scale``).
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.analysis.traces_scorer import parse_trial_dir, trial_status
from experiments.exp4 import driver as driver_module
from experiments.exp4 import runner_main
from experiments.exp4.driver import TRIAL_STATUS_FILE, Exp4Driver, trace_dir_name
from experiments.runner import Cell
from experiments.runner import runner as runner_module
from hermes.processes.config import device_config_to_json, mule_config_to_json
from hermes.scheduler.stages.s3_deadline import DeadlineLawError

from tests.golden import _build_topology as T

#: The keys of every marker the driver wrote at afa9526; a wall-clock marker
#: keeps exactly these.
AFA9526_MARKER_KEYS = {"status", "error", "n_missions_target", "run_s", "trial_budget_s"}


def _cell(arm="H1", seed=7, **params) -> Cell:
    p = {"N": 6, "rrf": 60.0, "n_missions": 4, "regime": "jittery"}
    p.update(params)
    return Cell(cell_id="|".join(f"{k}={v}" for k, v in sorted(p.items())), arm=arm,
                trial_index=0, seed=seed, params=p)


def _sim(**kw) -> Exp4Driver:
    kw.setdefault("mission_clock", "sim")
    kw.setdefault("realism", True)
    return Exp4Driver(**kw)


def _marker(root, cell) -> dict:
    return json.loads((root / trace_dir_name(cell) / TRIAL_STATUS_FILE).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. The driver records the runner's cap on simulated-clock markers only
# --------------------------------------------------------------------------- #

def test_the_soft_cap_field_is_appended_and_unset_by_default():
    """Appended after every other field, so each keeps its position; None
    (nothing told) writes no marker key."""
    assert fields(Exp4Driver)[-1].name == "soft_cap_s"
    assert Exp4Driver().soft_cap_s is None and _sim().soft_cap_s is None


@pytest.mark.parametrize("sim", [False, True], ids=["wall", "sim"])
def test_the_soft_cap_is_recorded_on_simulated_markers_only(tmp_path, sim):
    """N = 6 and 4 missions: the trial's own budget is 562 s on the clock and
    120 s on the wall; the cap a two-N grid (N = 6, 12) gives the runner is
    994 s. The wall marker keeps its afa9526 key set although the driver was
    told the cap."""
    drv = (_sim if sim else Exp4Driver)(realism=True, trace_root=tmp_path)
    drv.soft_cap_s = 994.0                    # as runner_main sets it
    cell = _cell("H1")
    T.run_stub_trial(drv, cell)
    marker = _marker(tmp_path, cell)
    assert marker["status"] == "ok"
    if sim:
        assert set(marker) == AFA9526_MARKER_KEYS | {"t_nom_computed", "soft_cap_s"}
        assert (marker["trial_budget_s"], marker["soft_cap_s"]) == (562.0, 994.0)
    else:
        assert set(marker) == AFA9526_MARKER_KEYS
        assert marker["trial_budget_s"] == 120.0


def test_a_simulated_marker_without_a_cap_has_no_soft_cap_key(tmp_path):
    """A driver used without runner_main is told no cap: its marker keeps the
    key set it had before the fix, and the scorer the trial's own budget."""
    drv = _sim(trace_root=tmp_path)
    cell = _cell("H1")
    T.run_stub_trial(drv, cell)
    marker = _marker(tmp_path, cell)
    assert set(marker) == AFA9526_MARKER_KEYS | {"t_nom_computed"}
    assert marker["trial_budget_s"] == 562.0


@pytest.mark.parametrize("sim", [False, True], ids=["wall", "sim"])
def test_a_killed_trial_is_marked_by_the_same_rule(monkeypatch, tmp_path, sim):
    """The error marker is written by the same rule as the ok one. runner_main
    tells the driver the cap on the wall clock too, so the wall case pins that
    a hard-killed wall trial keeps its afa9526 key set."""
    monkeypatch.setattr(Exp4Driver, "_await_mules", lambda self, orch, budget_s: False)
    drv = (_sim if sim else Exp4Driver)(realism=True, trace_root=tmp_path)
    drv.soft_cap_s = 994.0                    # as runner_main sets it
    cell = _cell("H1")
    budget = 562 if sim else 120
    with pytest.raises(driver_module.Exp4TrialTimeout, match=f"exceeded {budget}s budget"):
        T.run_stub_trial(drv, cell)
    marker = _marker(tmp_path, cell)
    assert (marker["status"], marker["trial_budget_s"]) == ("error", float(budget))
    if sim:
        assert set(marker) == AFA9526_MARKER_KEYS | {"t_nom_computed", "soft_cap_s"}
        assert marker["soft_cap_s"] == 994.0
    else:
        assert set(marker) == AFA9526_MARKER_KEYS


# --------------------------------------------------------------------------- #
# 2. runner_main tells the driver the cap it hands TrialRunner
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("argv, cap", [
    # The grid's largest re-costed budget: N = 12, 4 missions at the 3 s TTL.
    (["--mission-clock", "sim", "--N", "6", "12"], 994.0),
    (["--mission-clock", "sim", "--N", "6", "12", "--timeout-s", "500"], 500.0),
    # The wall clock: --trial-budget-s, as recorded (the marker leaves it out).
    (["--N", "6", "12"], 120.0),
    (["--N", "6", "12", "--timeout-s", "150"], 150.0),
], ids=["sim-grid-max", "sim-timeout-s", "wall-default", "wall-timeout-s"])
def test_the_runner_tells_the_driver_the_soft_cap_it_applies(monkeypatch, tmp_path, argv, cap):
    drivers, runner_kwargs = [], {}

    class _Driver(Exp4Driver):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            drivers.append(self)

    class _Runner:
        def __init__(self, *args, **kwargs):
            runner_kwargs.update(kwargs)

        def run(self, run_trial):
            return 0

    monkeypatch.setattr(runner_main, "Exp4Driver", _Driver)
    monkeypatch.setattr(runner_main, "TrialRunner", _Runner)
    assert runner_main.main(["--csv", str(tmp_path / "t.csv"), "--arms", "H1",
                             "--n-missions", "4", *argv]) == 0
    (drv,) = drivers
    assert runner_kwargs["timeout_s"] == cap
    assert drv.soft_cap_s == cap and type(drv.soft_cap_s) is float


# --------------------------------------------------------------------------- #
# 3. The scorer applies the cap the marker records
# --------------------------------------------------------------------------- #

TRIAL = "N=6-n_missions=3-regime=clean-rrf=60.0__H1__t0__s42"


def _marked(root, **timing) -> Path:
    d = root / TRIAL
    d.mkdir(parents=True, exist_ok=True)
    (d / TRIAL_STATUS_FILE).write_text(json.dumps(
        {"status": "ok", "error": "", "n_missions_target": 3, **timing}), encoding="utf-8")
    return d


@pytest.mark.parametrize("run_s, status, source", [
    (458.0, "ok", "marker"),           # past the trial's own budget, inside the cap
    (768.0, "ok", "marker"),           # at the cap: the runner relabels only past it
    (778.0, "timeout", "soft_cap"),    # past the cap
])
def test_a_marker_with_the_runners_cap_is_scored_against_it(tmp_path, run_s, status, source):
    got = trial_status(_marked(tmp_path, run_s=run_s, trial_budget_s=444.0, soft_cap_s=768.0))
    assert (got.status, got.source, got.marker_status) == (status, source, "ok")


def test_a_marker_without_the_cap_keeps_the_recorded_rule(tmp_path):
    """Wall markers, and simulated ones the runner did not stamp: the trial
    budget is the cap. A marker with no run time keeps its ok."""
    got = trial_status(_marked(tmp_path, run_s=458.0, trial_budget_s=444.0))
    assert (got.status, got.source) == ("timeout", "soft_cap")
    got = trial_status(_marked(tmp_path, run_s=None, trial_budget_s=444.0, soft_cap_s=768.0))
    assert (got.status, got.source) == ("ok", "marker")


def test_the_trial_csv_still_has_the_final_word(tmp_path):
    d = _marked(tmp_path, run_s=458.0, trial_budget_s=444.0, soft_cap_s=768.0)
    trial_csv = tmp_path / "trials.csv"
    trial_csv.write_text("cell_id,arm,trial_index,seed,status,error\n"
                         "N=6|n_missions=3|regime=clean|rrf=60.0,H1,0,42,timeout,\n",
                         encoding="utf-8")
    got = trial_status(d, trial_csv)
    assert (got.status, got.source, got.marker_status) == ("timeout", "csv", "ok")


# --------------------------------------------------------------------------- #
# 4. End to end: runner_main -> TrialRunner -> run_trial -> marker -> scorer
# --------------------------------------------------------------------------- #
#
# The real CLI, TrialRunner, run_trial, hard kill (_await_mules), trace
# capture, marker and scorer. Only process spawning and time are faked: one
# fake clock drives the driver's time.monotonic and the runner's time.time,
# so a trial's length is exact on both sides and costs no real time.

STARTUP_S, SHUTDOWN_S = 25.0, 3.0


class _Clock:
    now = 1_000_000.0


def _one_clock(monkeypatch) -> _Clock:
    clock = _Clock()
    fake = SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now,
                           sleep=lambda s: None)
    monkeypatch.setattr(driver_module, "time", fake)
    monkeypatch.setattr(runner_module, "time", fake)
    return clock


def _timed_orchestrator(tmp_path, clock, mule_s):
    """Stands in for ``MultiProcessOrchestrator`` on the fake clock.

    Nothing is spawned; the run directory holds the topology's per-role
    configs, as a kept trace does. Starting takes ``STARTUP_S``, the trial's
    mule runs ``mule_s[N]`` seconds and exits 0 unless the driver's hard kill
    comes first, and shutting down takes ``SHUTDOWN_S``.
    """

    class _Mule:
        def __init__(self, run_s):
            self.run_s, self.rc = float(run_s), None
            self.proc = self

        def wait(self, timeout=None):
            if timeout is not None and self.run_s > timeout:
                clock.now += float(timeout)
                raise subprocess.TimeoutExpired("mule", timeout)
            clock.now += self.run_s
            self.rc = 0
            return 0

        def returncode(self):
            return self.rc

    class FakeOrchestrator:
        def __init__(self, topo, capture_output=True):
            topo.validate()
            self.tmpdir = Path(tempfile.mkdtemp(prefix="run", dir=tmp_path))
            for m in topo.mules:
                (self.tmpdir / f"mule-{m.mule_id}.json").write_text(
                    mule_config_to_json(m), encoding="utf-8")
            for d in topo.devices:
                (self.tmpdir / f"device-{d.device_id}.json").write_text(
                    device_config_to_json(d), encoding="utf-8")
            self.mule_handles = {m.mule_id: _Mule(mule_s[len(topo.devices)]) for m in topo.mules}

        def start_all(self, timeout):
            clock.now += STARTUP_S

        def shutdown_all(self, timeout, cleanup_tmpdir):
            clock.now += SHUTDOWN_S

        def cleanup(self):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    return FakeOrchestrator


# 3 missions at the 3 s TTL: the N = 6 cell's own budget is 444 s on the clock,
# the N = 12 cell's 768 s; on the wall both are --trial-budget-s (120 s). Each
# trial's run time is 25 s + the mule's seconds + 3 s.
GRIDS = [
    # The finding: the N = 6 trial returns at 458 s, past its own budget but
    # inside the runner's cap (the grid's largest, 768 s), so the CSV says ok;
    # the N = 12 trial returns at 778 s, past the cap: timeout either way.
    pytest.param(["--mission-clock", "sim", "--N", "6", "12"], {6: 430.0, 12: 750.0},
                 {6: 444.0, 12: 768.0}, 768.0, {6: "ok", 12: "timeout"}, id="sim-two-N"),
    # The same N = 6 trial alone: the cap is its own budget, and 458 s is late.
    pytest.param(["--mission-clock", "sim", "--N", "6"], {6: 430.0},
                 {6: 444.0}, 444.0, {6: "timeout"}, id="sim-one-N"),
    # The wall clock, two N: one cap, 120 s, for both (the recorded rule).
    pytest.param(["--N", "6", "12"], {6: 100.0, 12: 60.0},
                 {6: 120.0, 12: 120.0}, 120.0, {6: "timeout", 12: "ok"}, id="wall-two-N"),
]


@pytest.mark.parametrize("argv, mule_s, own, cap, expected", GRIDS)
def test_the_scorer_reaches_the_runners_verdict_without_the_trial_csv(
        monkeypatch, tmp_path, argv, mule_s, own, cap, expected):
    """The invariant of ``trial_status``: a trace scored alone gets the status
    the runner wrote in its CSV row, on either clock and on any grid."""
    clock = _one_clock(monkeypatch)
    monkeypatch.setattr(driver_module, "MultiProcessOrchestrator",
                        _timed_orchestrator(tmp_path, clock, mule_s))
    trial_csv, root = tmp_path / "trials.csv", tmp_path / "t"
    assert runner_main.main(["--csv", str(trial_csv), "--arms", "H1", "--n-missions", "3",
                             "--keep-event-traces", "--trace-dir", str(root), *argv]) == 0
    with open(trial_csv, newline="", encoding="utf-8") as f:
        rows = {(r["arm"], int(r["trial_index"]), int(r["seed"])): r for r in csv.DictReader(f)}
    verdicts, markers = {}, {}
    for trace in sorted(p for p in root.iterdir() if p.is_dir()):
        key = parse_trial_dir(trace.name)
        row = rows[(key.arm, key.trial_index, key.seed)]
        n = int(row["param_N"])
        assert float(row["duration_s"]) == STARTUP_S + mule_s[n] + SHUTDOWN_S
        alone, joined = trial_status(trace), trial_status(trace, trial_csv)
        verdicts[n] = (row["status"], alone.status, alone.source, joined.status)
        markers[n] = json.loads((trace / TRIAL_STATUS_FILE).read_text(encoding="utf-8"))
    # The runner's verdict (the CSV row), then the scorer's without the CSV
    # (and the rule that gave it) and with it.
    assert verdicts == {
        n: (s, s, "soft_cap" if s == "timeout" else "marker", s) for n, s in expected.items()
    }
    # The marker: the driver's ok, the run time, the trial's own budget, and on
    # the simulated clock only, the runner's cap.
    for n, marker in markers.items():
        assert (marker["status"], marker["run_s"], marker["trial_budget_s"]) == (
            "ok", STARTUP_S + mule_s[n] + SHUTDOWN_S, own[n])
        if "--mission-clock" in argv:
            assert marker["soft_cap_s"] == cap
        else:
            assert set(marker) == AFA9526_MARKER_KEYS


# --------------------------------------------------------------------------- #
# 5. The deadline law's time unit is refused in deadline_params
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kwargs", [
    dict(deadline_params={"time_scale": 2.0}),
    dict(deadline_law="multiplicative", deadline_params={"beta_on": 0.7, "time_scale": 2.0}),
    # afa9526 refused the key at any value, the recorded one included.
    dict(deadline_params={"time_scale": 1.0}),
    # Beside the time unit's own field it used to fail only at mule start.
    dict(deadline_params={"time_scale": 2.0}, deadline_time_scale=3.0),
    dict(mission_clock="sim", deadline_params={"time_scale": 2.0}),
    # from_config takes any shape dict() takes, as at afa9526, which refused
    # these too.
    dict(deadline_params=[("time_scale", 2.0)]),
    dict(deadline_params=(("beta_on", 0.7), ("time_scale", 2.0))),
], ids=["additive", "multiplicative", "recorded-value", "with-deadline-time-scale", "sim",
        "pairs-list", "pairs-tuple"])
def test_the_laws_time_unit_is_refused_in_deadline_params(kwargs):
    """Refused with afa9526's exception type, a ValueError, which runner_main
    reports as a usage error."""
    with pytest.raises(DeadlineLawError, match=r"deadline_time_scale \(--deadline-time-scale\)"):
        Exp4Driver(**kwargs)
    assert issubclass(DeadlineLawError, ValueError)


def test_a_one_shot_deadline_params_reaches_the_law_whole():
    """The refusal reads the parameters once, as from_config does: an iterable
    of pairs that can be read only once still sets the law, as at afa9526."""
    drv = Exp4Driver(deadline_law="multiplicative",
                     deadline_params=iter([("beta_on", 0.7), ("phi_max", 200.0)]))
    assert (drv._deadline_law.beta_on, drv._deadline_law.phi_max) == (0.7, 200.0)


def test_the_time_unit_through_its_own_field_runs_and_is_recorded():
    """The supported route: the scheduler runs the unit, the column records it,
    and the law's own parameters stay free of it."""
    row, topo = T.run_stub_trial(Exp4Driver(deadline_time_scale=2.0), _cell("H1", regime="clean"))
    assert row["deadline_time_scale"] == 2.0 and row["deadline_params"] == ""
    (mule,) = topo.mules
    assert mule.deadline_time_scale == 2.0 and "time_scale" not in (mule.deadline_params or {})
    drv = Exp4Driver(deadline_law="multiplicative", deadline_params={"beta_on": 0.7})
    assert json.loads(T.run_stub_trial(drv, _cell("H1", regime="clean"))[0]["deadline_params"])[
        "beta_on"] == 0.7
