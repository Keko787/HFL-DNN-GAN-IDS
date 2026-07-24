"""Experiment-4 trial driver (chunk EX-4.0).

Plugs into the shared :class:`~experiments.runner.TrialRunner`'s
``run_trial(cell)`` slot. For each cell it:

1. Builds a finite topology (1 cluster + 1 mule + N devices, mule capped
   at ``n_missions``) from the cell's sweep coordinates + paired seed.
2. Brings it up on the **real** :class:`MultiProcessOrchestrator` (real
   subprocesses, real TCP, real two-pass Pass-1 → dock → Pass-2 with real
   cross-mule FedAvg).
3. Waits for the mule to exit *naturally* (n_missions reached) within a
   hard per-trial budget — killing the tree on overrun so a hung socket
   never blocks the whole sweep (the harness's own timeout only *labels*;
   it does not kill).
4. Reads the per-process JSONL and rolls it up into the federation-side
   metrics.

Only arm **H1** exists in EX-4.0 (mule + gated scheduler + two-pass HFL,
deterministic distance ranking, no RL selector, no L1). H0/H2/H3 arrive
in later chunks; an unknown arm is rejected loudly.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from experiments.runner import Cell

from hermes.processes import MultiProcessOrchestrator

from .events_consumer import consume_run_dir
from .metrics import Exp4MetricSummary, summarise_observation
from .topology_builder import build_exp4_topology

log = logging.getLogger("experiments.exp4.driver")


ARMS = ("H1",)


class Exp4TrialTimeout(RuntimeError):
    """Raised when a trial blows its hard wall-clock budget.

    The orchestrator is killed before this propagates; the harness
    records the row with ``status=error`` and the sweep continues.
    """


@dataclass
class Exp4Driver:
    """Owns per-trial dispatch over the real multi-process orchestrator.

    Defaults are deliberately small — EX-4.0 is about proving the
    integrated measurement plumbing, not running the paper grid. The
    sweep knobs (N, rf_range_m, n_missions) come off the cell; anything
    the cell omits falls back to these.
    """

    default_n_devices: int = 2
    default_rf_range_m: float = 60.0
    default_n_missions: int = 2
    # Hard per-trial wall-clock budget (seconds). A trial that runs long
    # is killed and recorded as an error rather than hanging the sweep.
    trial_budget_s: float = 120.0
    startup_timeout_s: float = 30.0
    shutdown_timeout_s: float = 10.0

    def run_trial(self, cell: Cell) -> Mapping[str, Any]:
        params = cell.params
        arm = cell.arm
        if arm not in ARMS:
            raise ValueError(
                f"unknown arm {arm!r}; EX-4.0 ships {ARMS} "
                f"(H0/H2/H3 arrive in later chunks)"
            )

        n_devices = int(params.get("N", params.get("n_devices", self.default_n_devices)))
        rf_range_m = float(params.get("rrf", params.get("rf_range_m", self.default_rf_range_m)))
        n_missions = int(params.get("n_missions", self.default_n_missions))

        topo = build_exp4_topology(
            n_devices=n_devices,
            rf_range_m=rf_range_m,
            n_missions=n_missions,
            seed=cell.seed,
        )

        orch = MultiProcessOrchestrator(topo, capture_output=True)
        try:
            orch.start_all(timeout=self.startup_timeout_s)
            timed_out = not self._await_mules(orch, self.trial_budget_s)
            # Reap dead procs (mules exited naturally); terminate the
            # long-running cluster + any devices still up. Keep the tmpdir
            # so we can read the JSONL the mule/cluster already flushed.
            orch.shutdown_all(
                timeout=self.shutdown_timeout_s, cleanup_tmpdir=False,
            )
            if timed_out:
                raise Exp4TrialTimeout(
                    f"exp4 trial exceeded {self.trial_budget_s:.0f}s budget "
                    f"(cell={cell.cell_id}, trial={cell.trial_index}); "
                    f"orchestrator killed"
                )
            obs = consume_run_dir(orch.tmpdir, n_devices=n_devices)
            if obs.missions_completed == 0:
                log.warning(
                    "exp4 trial produced no completed missions "
                    "(cell=%s trial=%d); mule_ready=%s dock_bootstrapped=%s "
                    "cluster_ready=%s — recording a zeroed row",
                    cell.cell_id, cell.trial_index,
                    obs.mule_ready, obs.dock_bootstrapped, obs.cluster_ready,
                )
            summary: Exp4MetricSummary = summarise_observation(
                obs,
                n_devices=n_devices,
                rf_range_m=rf_range_m,
                n_missions_target=n_missions,
            )
            return summary.to_row()
        finally:
            orch.cleanup()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _await_mules(
        self, orch: MultiProcessOrchestrator, budget_s: float,
    ) -> bool:
        """Block until every mule subprocess exits, or the budget runs out.

        Returns True if all mules exited within budget, False on timeout.
        The mules are capped at ``n_missions`` so they terminate on their
        own; blocking on their handles means we only read the logs once
        the missions really finished.
        """
        deadline = time.monotonic() + budget_s
        for handle in orch.mule_handles.values():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                handle.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                return False
        return True
