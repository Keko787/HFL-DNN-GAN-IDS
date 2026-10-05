"""Scheduler Freeze Amendment 11 (resolution R14): a trial that raises before its
shutdown shuts its processes down.

Pinned:

* A raise in ``start_all`` (a mule or a device that fails at startup) still
  shuts the topology down, once, keeping the tmpdir for the trace, and then
  cleans up; the trial's own error is what the caller sees.
* A shutdown that itself fails on that path never masks the trial's error.
* A trial that reaches its shutdown is shut down exactly once, as recorded.
"""

from __future__ import annotations

import pytest

from experiments.exp4 import driver as driver_module
from experiments.exp4.driver import Exp4Driver
from experiments.runner.grid import Cell


CELL = Cell(cell_id="N=2|rrf=60.0", arm="H1", trial_index=0, seed=7, params={})


def _fake_orchestrator(calls, *, start_raises=None, shutdown_raises=None):
    class FakeOrchestrator:
        def __init__(self, topo, capture_output=True, **_):
            self.tmpdir = None
            self.mule_handles = {}

        def start_all(self, timeout):
            calls.append("start_all")
            if start_raises is not None:
                raise start_raises

        def shutdown_all(self, timeout, cleanup_tmpdir):
            calls.append(("shutdown_all", cleanup_tmpdir))
            if shutdown_raises is not None:
                raise shutdown_raises

        def cleanup(self):
            calls.append("cleanup")

    return FakeOrchestrator


def _topology():
    return driver_module.build_exp4_topology(n_devices=2, rf_range_m=60.0, n_missions=1, seed=7)


def _run(driver):
    return driver._run_topology(_topology(), cell=CELL, n_devices=2, rf_range_m=60.0,
                                n_missions=1)


def test_a_startup_failure_shuts_the_started_processes_down(monkeypatch):
    calls = []
    boom = RuntimeError("mule exp4-mule failed to start")
    monkeypatch.setattr(driver_module, "MultiProcessOrchestrator",
                        _fake_orchestrator(calls, start_raises=boom))
    with pytest.raises(RuntimeError, match="failed to start"):
        _run(Exp4Driver(default_n_missions=1))
    assert calls == ["start_all", ("shutdown_all", False), "cleanup"]


def test_a_failing_shutdown_never_masks_the_trials_error(monkeypatch):
    calls = []
    monkeypatch.setattr(driver_module, "MultiProcessOrchestrator", _fake_orchestrator(
        calls, start_raises=RuntimeError("device exp4-dev-001 failed to start"),
        shutdown_raises=OSError("terminate failed")))
    with pytest.raises(RuntimeError, match="device exp4-dev-001"):
        _run(Exp4Driver(default_n_missions=1))
    assert calls == ["start_all", ("shutdown_all", False), "cleanup"]


def test_a_trial_that_reaches_its_shutdown_is_shut_down_once(monkeypatch):
    calls = []
    monkeypatch.setattr(driver_module, "MultiProcessOrchestrator", _fake_orchestrator(calls))
    # No mule ever exits in the fake, so the trial times out after its shutdown,
    # on the recorded path: one shutdown, then the timeout is raised.
    monkeypatch.setattr(Exp4Driver, "_await_mules", lambda self, orch, budget_s: False)
    monkeypatch.setattr(Exp4Driver, "_capture_traces", lambda self, tmpdir, cell: None)
    with pytest.raises(driver_module.Exp4TrialTimeout):
        _run(Exp4Driver(default_n_missions=1))
    assert calls == ["start_all", ("shutdown_all", False), "cleanup"]
