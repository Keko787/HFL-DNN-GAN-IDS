"""Experiment-4 trial driver (chunks EX-4.0 + EX-4.1).

Plugs into the shared :class:`~experiments.runner.TrialRunner`'s
``run_trial(cell)`` slot. For each cell it:

1. Builds a finite topology (1 cluster + 1 mule + N devices, mule capped
   at ``n_missions``) from the cell's sweep coordinates + paired seed.
2. Brings it up on the **real** :class:`MultiProcessOrchestrator` (real
   subprocesses, real TCP, real two-pass Pass-1 → dock → Pass-2 with real
   cross-mule FedAvg).
3. Waits for the mule to exit *naturally* within a hard per-trial budget
   (killing the tree on overrun so a hung socket never blocks the sweep).
4. Reads the per-process JSONL and rolls it up into the metrics.

**EX-4.0 (``real_model=False``, default):** a noise-stub model — the
federation-side scheduling metrics from the real L2+L3 stack.

**EX-4.1 (``real_model=True``):** the *real* canonical DNN-IDS. The driver
prepares the ``CiciotTask`` once per trial ("driver-prepares-once"),
serializes each device's shard + the shared held-out test set + the real
seed weights, and points the subprocesses at them. The cluster seeds the
global model from those weights and scores each aggregated θ on the
held-out set, so the trial emits accuracy/AUC-over-rounds + T@τ.

Only arm **H1** exists so far (mule + gated scheduler + two-pass HFL,
deterministic ranking, no RL selector, no L1). H0/H2/H3 arrive in later
chunks; an unknown arm is rejected loudly.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from experiments.runner import Cell

from hermes.processes import MultiProcessOrchestrator

from .events_consumer import consume_run_dir
from .metrics import Exp4MetricSummary, summarise_observation
from .prep import prepare_trial
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

    ``real_model=False`` is the EX-4.0 instrumentation path (noise stub,
    fast). ``real_model=True`` is EX-4.1 — the real canonical DNN-IDS with
    per-round convergence; each trial spawns real TensorFlow fits in every
    device subprocess, so budget accordingly.
    """

    default_n_devices: int = 2
    default_rf_range_m: float = 60.0
    default_n_missions: int = 2
    # Hard per-trial wall-clock budget (seconds). A trial that runs long
    # is killed and recorded as an error rather than hanging the sweep.
    trial_budget_s: float = 120.0
    startup_timeout_s: float = 30.0
    shutdown_timeout_s: float = 10.0

    # ---- EX-4.1 real-model knobs (ignored when real_model=False) ---- #
    real_model: bool = False
    data_source: str = "canonical"  # "canonical" | "synthetic"
    local_epochs: int = 1
    local_batch_size: int = 64
    tau: float = 0.9
    theta_seed: int = 12345
    # canonical loader knobs
    train_files: int = 3
    test_files: int = 1
    train_dataset_size: int = 20000
    test_dataset_size: int = 8000
    attack_eval_ratio: float = 0.5
    # synthetic loader knobs
    synth_rows_per_device: int = 512
    synth_test_rows: int = 512

    def run_trial(self, cell: Cell) -> Mapping[str, Any]:
        params = cell.params
        arm = cell.arm
        if arm not in ARMS:
            raise ValueError(
                f"unknown arm {arm!r}; EX-4.0/4.1 ship {ARMS} "
                f"(H0/H2/H3 arrive in later chunks)"
            )

        n_devices = int(params.get("N", params.get("n_devices", self.default_n_devices)))
        rf_range_m = float(params.get("rrf", params.get("rf_range_m", self.default_rf_range_m)))
        n_missions = int(params.get("n_missions", self.default_n_missions))

        prep_dir: Optional[Path] = None
        try:
            if self.real_model:
                prep_dir = Path(tempfile.mkdtemp(prefix="exp4_prep_"))
                task = self._build_task(n_devices, cell.seed)
                prep = prepare_trial(prep_dir, task=task, theta_seed=self.theta_seed)
                log.info(
                    "exp4 real-model trial cell=%s trial=%d: source=%s "
                    "input_dim=%d n_train=%d synthetic=%s",
                    cell.cell_id, cell.trial_index, self.data_source,
                    prep.input_dim, prep.n_train, prep.is_synthetic,
                )
                topo = build_exp4_topology(
                    n_devices=n_devices,
                    rf_range_m=rf_range_m,
                    n_missions=n_missions,
                    seed=cell.seed,
                    train_shard_paths=prep.shard_paths,
                    input_dim=prep.input_dim,
                    local_epochs=self.local_epochs,
                    local_batch_size=self.local_batch_size,
                    init_theta_path=prep.init_theta_path,
                    eval_test_path=prep.test_path,
                )
            else:
                topo = build_exp4_topology(
                    n_devices=n_devices,
                    rf_range_m=rf_range_m,
                    n_missions=n_missions,
                    seed=cell.seed,
                )

            return self._run_topology(
                topo, cell=cell, n_devices=n_devices,
                rf_range_m=rf_range_m, n_missions=n_missions,
            )
        finally:
            if prep_dir is not None:
                shutil.rmtree(prep_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_task(self, n_devices: int, seed: int):
        from .model_task import load_ciciot_task_canonical, synthetic_task

        if self.data_source == "canonical":
            return load_ciciot_task_canonical(
                n_devices=n_devices,
                seed=seed,
                train_files=self.train_files,
                test_files=self.test_files,
                train_dataset_size=self.train_dataset_size,
                test_dataset_size=self.test_dataset_size,
                attack_eval_ratio=self.attack_eval_ratio,
            )
        if self.data_source == "synthetic":
            return synthetic_task(
                n_devices=n_devices,
                rows_per_device=self.synth_rows_per_device,
                test_rows=self.synth_test_rows,
                seed=seed,
            )
        raise ValueError(
            f"unknown data_source {self.data_source!r}; "
            f"expected 'canonical' or 'synthetic'"
        )

    def _run_topology(self, topo, *, cell, n_devices, rf_range_m, n_missions):
        orch = MultiProcessOrchestrator(topo, capture_output=True)
        try:
            orch.start_all(timeout=self.startup_timeout_s)
            timed_out = not self._await_mules(orch, self.trial_budget_s)
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
                tau=self.tau,
            )
            return summary.to_row()
        finally:
            orch.cleanup()

    def _await_mules(
        self, orch: MultiProcessOrchestrator, budget_s: float,
    ) -> bool:
        """Block until every mule subprocess exits, or the budget runs out.

        Returns True if all mules exited within budget, False on timeout.
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
