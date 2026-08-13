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


ARMS = ("H0", "H1", "H2", "H3")

#: Scheduler-configuration columns the driver stamps on every row, on top of
#: the metric schema. They exist so a results CSV is self-describing: rows
#: recorded with the S3b deadline gate or S3c window adaptation active are NOT
#: comparable with historical rows, and without these nothing in the file says
#: which is which. Owned by the driver, not the metric summary — they describe
#: how the trial was configured, not what it measured.
PROVENANCE_COLUMNS = ("mission_budget_s", "mission_window_adaptation")


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
    # EX-4.2 arm H2 — trained DDQN target selector (.npz from exp3.train_a4).
    # None -> H2 uses a random-init selector (plumbing smoke, not paper-grade).
    selector_weights_path: Optional[str] = None
    # H0 (traditional flat FL) — fraction of clients sampled per round.
    # 1.0 = every client every round (the reliable-infrastructure baseline,
    # matching Exp 1's fully-participating clients).
    h0_client_fraction: float = 1.0
    # ---- EX-4.2 jittery-regime knobs ---- #
    # H0's long-range backhaul degrades under jittery: a dead-zone fraction of
    # clients are persistently unreachable from the central server, and the
    # reachable ones contribute each round with prob reliability_i x
    # link_quality. H1's mule reaches devices over short-range contact
    # (jitter-immune), so it gets NO dead-zone.
    #
    # PROVENANCE + CAVEAT: this dead-zone / link_quality mechanism is the
    # flat-FL (A1) model from experiments/exp3/arm_a1.py + the exp3 driver —
    # NOT from the Exp 3 simulator (sim_env), which has no flat-FL arm. It was
    # tuned there for A1's 20-round horizon; Exp 4 runs few rounds, so the
    # dead-zone rate must be justified physically (fraction of devices with no
    # long-range path — terrain / range-edge) and reported as a SENSITIVITY
    # axis, not a single tuned point. The 0.6 default is one point on the sweep.
    clean_dead_zone_frac: float = 0.0
    jittery_dead_zone_frac: float = 0.6
    clean_link_quality: float = 1.0
    jittery_link_quality: float = 0.4
    # H1 (mule) realism, opt-in. Applies the per-device short-range contact
    # reliability (reliability_i x rf_factor, the SAME reliability draw H0
    # uses) in every regime, plus — under jittery — a long-range backhaul
    # upload loss that marks the round as not-closed (the recoverable, one-hop
    # cost of routing through the mule). SCOPE: this models the NETWORK +
    # computation layers only; it does NOT model mule flight-budget / deadline
    # pressure (fewer contacts under a tight budget) — that is the scheduling
    # experiment's (Exp 3) domain and is deferred here. Devices are spread so
    # S3a forms multiple contacts. Off -> the ideal EX-4.1 links.
    realism: bool = False
    h1_field_radius_m: float = 100.0
    h1_world_radius_m: float = 100.0
    clean_backhaul_loss_pct: float = 0.0
    jittery_backhaul_loss_pct: float = 2.0
    # EX-4.3 arm H3 — L1 adaptive channel selection. When on, the mule arms'
    # backhaul loss comes from the RF channel model (experiments.exp4.channel):
    # H1/H2 hold the best-average fixed band, H3 runs the U(c,t) controller.
    # Use with --realism (contact reliability) for the H1->H2->H3 ladder.
    l1_channel: bool = False
    l1_channel_bands: int = 3
    # S3b — per-mission time budget (seconds). None (default) reproduces every
    # previously-recorded result: the S3 deadline stays a sort key. Setting it
    # turns on the feasibility gate so the deadline actually binds.
    mission_budget_s: Optional[float] = None
    # S3c — mission-level window adaptation. False (default) reproduces every
    # previously-recorded result exactly: the window scale stays 1.0. Toggled
    # on, systemic mission shortfall widens all windows together.
    mission_window_adaptation: bool = False
    mission_window_history: int = 5
    mission_window_target: float = 0.8
    mission_window_gain: float = 2.0
    mission_window_max_scale: float = 4.0

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
        regime = str(params.get("regime", "clean"))

        if arm == "H0":
            if not self.real_model:
                raise ValueError(
                    "arm H0 (traditional flat FL) is a real-model convergence "
                    "baseline; run with real_model=True (--real-model)"
                )
            return self._run_h0(
                cell, n_devices=n_devices, rf_range_m=rf_range_m,
                n_rounds=n_missions, regime=regime,
            )

        # arm H1 — integrated stack over the real multi-process orchestrator.
        # EX-4.2 realism (opt-in): Exp 3's mule-arm impairment, asymmetric to
        # H0 (short-range contact reliability always; recoverable backhaul
        # loss under jittery; no dead-zone — the mule physically reaches
        # devices).
        realism_kwargs: dict = {}
        if self.realism:
            from .model_task import device_reliabilities
            realism_kwargs = dict(
                device_reliability=True,
                reliabilities=device_reliabilities(cell.seed, n_devices),
                world_radius_m=self.h1_world_radius_m,
                field_radius_m=self.h1_field_radius_m,
                backhaul_loss_pct=(
                    self.jittery_backhaul_loss_pct if regime == "jittery"
                    else self.clean_backhaul_loss_pct
                ),
                backhaul_rng_seed=(cell.seed ^ 0x0BACC0DE),
            )

        # EX-4.2/4.3 arms: H1 deterministic ranking; H2 = H1 + RL selector;
        # H3 = H2 + adaptive L1 channel.
        selector_kwargs = dict(
            use_rl_selector=(arm in ("H2", "H3")),
            selector_weights_path=self.selector_weights_path,
        )
        if self.mission_budget_s is not None:
            selector_kwargs["mission_budget_s"] = float(self.mission_budget_s)
        if self.mission_window_adaptation:
            selector_kwargs.update(
                mission_window_adaptation=True,
                mission_window_history=int(self.mission_window_history),
                mission_window_target=float(self.mission_window_target),
                mission_window_gain=float(self.mission_window_gain),
                mission_window_max_scale=float(self.mission_window_max_scale),
            )

        # EX-4.3 arm H3 — L1 adaptive channel. H1/H2 hold the best-average
        # fixed band; H3 runs the U(c,t) controller. The per-mission loss
        # schedule (cluster) + chosen-channel mean SNR (selector RF prior)
        # replace the flat backhaul loss for all mule arms in this mode.
        if self.l1_channel:
            from .channel import ChannelModel, backhaul_plan
            model = ChannelModel(
                n_bands=self.l1_channel_bands, n_missions=n_missions,
                seed=cell.seed, jittery=(regime == "jittery"),
            )
            plan = backhaul_plan(model, adaptive=(arm == "H3"))
            realism_kwargs["backhaul_loss_schedule"] = plan.loss_schedule
            realism_kwargs["rf_prior_snr_db"] = plan.mean_chosen_snr_db
            # The schedule drives the cluster's per-mission Bernoulli draw; seed
            # it off the paired seed so H1/H2/H3 share the identical draw
            # sequence (paired comparison holds even without --realism).
            realism_kwargs.setdefault("backhaul_rng_seed", cell.seed ^ 0x0BACC0DE)
            log.info(
                "exp4 %s L1 channel cell=%s trial=%d regime=%s: adaptive=%s "
                "mean_snr=%.1fdB mean_loss=%.3f bands=%s",
                arm, cell.cell_id, cell.trial_index, regime, plan.adaptive,
                plan.mean_chosen_snr_db,
                sum(plan.loss_schedule) / max(1, len(plan.loss_schedule)),
                plan.chosen_bands,
            )

        prep_dir: Optional[Path] = None
        try:
            if self.real_model:
                prep_dir = Path(tempfile.mkdtemp(prefix="exp4_prep_"))
                task = self._build_task(n_devices, cell.seed)
                prep = prepare_trial(prep_dir, task=task, theta_seed=self.theta_seed)
                log.info(
                    "exp4 real-model H1 trial cell=%s trial=%d regime=%s "
                    "realism=%s: source=%s input_dim=%d n_train=%d synthetic=%s",
                    cell.cell_id, cell.trial_index, regime, self.realism,
                    self.data_source, prep.input_dim, prep.n_train, prep.is_synthetic,
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
                    **realism_kwargs,
                    **selector_kwargs,
                )
            else:
                topo = build_exp4_topology(
                    n_devices=n_devices,
                    rf_range_m=rf_range_m,
                    n_missions=n_missions,
                    seed=cell.seed,
                    **realism_kwargs,
                    **selector_kwargs,
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

    def _run_h0(self, cell: Cell, *, n_devices, rf_range_m, n_rounds, regime="clean"):
        """Traditional flat FL (H0) — the paired real-model null.

        Runs synchronous FedAvg **in process** (no mule, no orchestrator):
        every round each reachable+sampled client trains the real DNN-IDS
        from the current global θ, the server ``partial_fedavg``-aggregates
        their weights, and the aggregated θ is scored on the shared held-out
        set. Uses the same task (same paired seed -> same shards), same
        seeded init θ, and the same convergence definitions as H1.

        EX-4.2 jittery regime: H0 relies on the long-range backhaul to every
        client, so under jitter a ``dead_zone_frac`` of clients are
        persistently unreachable and the reachable ones succeed each round
        only with prob ``link_quality`` — modelling the degraded-line-of-sight
        backhaul that collapses centralized participation (Exp 3's A1 model).
        A round with no successful updates does not close (deadline unmet).
        """
        import numpy as np

        from hermes.mission.partial_fedavg import partial_fedavg
        from hermes.types import DeviceID, GradientSubmission, MuleID

        from .metrics import summarise_flat_fl
        from .model_task import (
            _u32, device_reliabilities, initial_theta, make_local_train_fn,
        )
        from experiments.exp3.metrics import Exp3RoundLog

        jittery = regime == "jittery"
        # Dead-zone / link-quality are sweepable per cell (the sensitivity
        # surface, B2) — a cell param overrides the driver default. Only
        # meaningful under jittery.
        if jittery:
            dead_zone_frac = float(cell.params.get("dead_zone", self.jittery_dead_zone_frac))
            link_quality = float(cell.params.get("link_quality", self.jittery_link_quality))
        else:
            dead_zone_frac = self.clean_dead_zone_frac
            link_quality = self.clean_link_quality
        # Shared per-device reliability — the SAME draw H1 uses, so the clean
        # comparison is fair (H0 is not idealised to perfect participation).
        # H0 is all long-range: a reachable client contributes each round with
        # prob reliability_i x link_quality (link_quality = 1.0 clean, <1
        # jittery). Dead-zoned clients never contribute (permanent).
        rels = device_reliabilities(cell.seed, n_devices)

        task = self._build_task(n_devices, cell.seed)
        input_dim = task.input_dim

        # Persistent long-range dead zone — clients the central server never
        # reaches this mission (deterministic from the paired seed).
        n_dead = int(round(n_devices * dead_zone_frac))
        dz_rng = np.random.default_rng(_u32(cell.seed, "h0_deadzone"))
        dead = set(
            int(i) for i in dz_rng.choice(n_devices, size=n_dead, replace=False)
        ) if n_dead > 0 else set()
        reachable = [i for i in range(n_devices) if i not in dead]

        log.info(
            "exp4 H0 flat-FL cell=%s trial=%d regime=%s: source=%s input_dim=%d "
            "n_train=%d rounds=%d reachable=%d/%d link_quality=%.2f",
            cell.cell_id, cell.trial_index, regime, self.data_source, input_dim,
            task.n_train, n_rounds, len(reachable), n_devices, link_quality,
        )

        # Same seeded init θ as H1 so both arms start from the same model.
        theta = initial_theta(input_dim, seed=self.theta_seed)
        # Build a trainer only for reachable clients (dead ones never fit).
        client_fns = {
            i: make_local_train_fn(
                task.device_shards[i][0], task.device_shards[i][1],
                input_dim=input_dim, epochs=self.local_epochs,
                batch_size=self.local_batch_size, seed=self.theta_seed,
            )
            for i in reachable
        }

        evals = [self._eval_point(0, theta, task, input_dim)]
        round_logs: list = []
        participation = {i: 0 for i in range(n_devices)}
        samp_rng = np.random.default_rng(_u32(cell.seed, "h0_sampling"))
        link_rng = np.random.default_rng(_u32(cell.seed, "h0_link"))
        n_sample = max(1, int(round(len(reachable) * self.h0_client_fraction))) if reachable else 0

        for r in range(1, n_rounds + 1):
            if not reachable:
                sampled = []
            elif n_sample >= len(reachable):
                sampled = list(reachable)
            else:
                sampled = sorted(
                    int(i) for i in samp_rng.choice(reachable, size=n_sample, replace=False)
                )
            subs = []
            for i in sampled:
                # Long-range participation: device availability x link quality.
                # Applies in clean too (link_quality=1.0 -> prob = reliability),
                # so H0 pays the same heterogeneity tax as H1 — no idealised
                # clean win.
                p_i = rels[i] * link_quality
                if float(link_rng.random()) >= p_i:
                    continue
                res = client_fns[i](theta, [])
                participation[i] += 1
                subs.append(
                    GradientSubmission(
                        device_id=DeviceID(f"exp4-dev-{i:03d}"),
                        mule_id=MuleID("h0-server"),
                        mission_round=r,
                        delta_theta=res.delta_theta,
                        num_examples=res.num_examples,
                        submitted_at=0.0,
                    )
                )
            if subs:
                theta = partial_fedavg(MuleID("h0-server"), r, subs).weights
                closed = True
            else:
                # No client reached the server this round — no aggregation,
                # θ carries over, the round does not close.
                closed = False
            evals.append(self._eval_point(r, theta, task, input_dim))
            round_logs.append(
                Exp3RoundLog(
                    round_index=r, n_updates=len(subs),
                    n_target=n_devices, deadline_met=closed,
                )
            )

        summary = summarise_flat_fl(
            model_evals=evals,
            round_logs=round_logs,
            per_client_participation=participation,
            n_devices=n_devices,
            rf_range_m=rf_range_m,
            n_missions_target=n_rounds,
            tau=self.tau,
        )
        return summary.to_row()

    def _eval_point(self, cluster_round, theta, task, input_dim):
        from .events_consumer import ModelEvalPoint
        from .model_task import evaluate_theta

        m = evaluate_theta(theta, task.X_test, task.y_test, input_dim=input_dim)
        return ModelEvalPoint(
            cluster_round=int(cluster_round),
            accuracy=float(m["accuracy"]),
            auc=float(m["auc"]),
            loss=float(m["loss"]),
            n_test=int(len(task.y_test)),
        )

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
            row = summary.to_row()
            # Provenance for the two opt-in scheduler mechanisms. Recorded on
            # every row so a CSV is self-describing: without these, an
            # enforcement or window-adaptation run is indistinguishable from a
            # historical one after the fact, and the two must never be pooled.
            row["mission_budget_s"] = (
                "" if self.mission_budget_s is None else float(self.mission_budget_s)
            )
            row["mission_window_adaptation"] = int(bool(self.mission_window_adaptation))
            # A trial that produced NO model evaluation at all never trained a
            # model — its convergence columns are blank while its federation
            # columns are hard zeros. Recorded as `ok`, that asymmetry biases
            # the analysis: the blank AUC is dropped by `.dropna()` while the
            # 0.0 participation is averaged as if it were a real observation
            # (it flipped the sign of several H3−H2 differences in the first
            # H2/H3 dead-zone sweep). Mark it so `status == "ok"` filters
            # exclude it from EVERY metric, not just the ones that are blank.
            if self.real_model and int(summary.rounds_evaluated) == 0:
                row["status"] = "no_eval"
                row["error"] = (
                    f"trial produced no model_evaluation events "
                    f"(mule_ready={obs.mule_ready} "
                    f"dock_bootstrapped={obs.dock_bootstrapped} "
                    f"cluster_ready={obs.cluster_ready}); not a valid trial"
                )
                log.warning(
                    "exp4 trial cell=%s trial=%d arm=%s: no model evaluations "
                    "— recording status=no_eval (excluded from analysis)",
                    cell.cell_id, cell.trial_index, cell.arm,
                )
            return row
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
