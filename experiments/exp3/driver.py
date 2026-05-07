"""Experiment-3 trial-grid driver — selects and runs the right arm.

Plugs into :class:`~experiments.runner.TrialRunner`'s
``run_trial(cell)`` slot. The cell's ``arm`` field selects A1 / A2 /
A3 / A4; the cell's ``params`` carries the sweep coordinates (N, β,
deadline_het, rrf).

Returns a metric dict the runner writes as one CSV row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from experiments.calibration import Exp3Calibration, load_calibration
from experiments.runner import Cell

from hermes.scheduler.policies import (
    ArrivalOrderPolicy,
    EdfFeasibilityPolicy,
)
from hermes.scheduler.policies.edf_feasibility import FeasibilityModel
from hermes.scheduler.selector import TargetSelectorRL

from .arm_a1 import A1Config, run_a1_trial
from .arm_mule import MuleArmConfig, run_mule_trial
from .metrics import Exp3MetricSummary
from .sim_env import Exp3SimConfig

log = logging.getLogger("experiments.exp3.driver")


ARMS = ("A1", "A2", "A3", "A4")


@dataclass
class Exp3Driver:
    """Owns the per-trial dispatch + the (optional) shared A4 selector.

    The driver carries **separate** upload-model defaults for the
    clean and jittery regimes so the two cells of the ``jittery``
    sweep axis test materially different network conditions:

    * **Clean regime** — minimal network pressure. Default 1 MB at
      10 Mbps gives upload_s ≈ 0.8 s per contact, dwarfed by the 30 s
      collect window, so A3's feasibility filter rarely fires and
      the strategies are compared in a near-ideal scheduling regime.
    * **Jittery regime** — heavy network pressure. Default 10 MB at
      1 Mbps gives upload_s ≈ 80 s per contact, plus 2% packet loss
      and 30% latency jitter (Exp.\ 1 ``--jittery`` parity). A3's
      filter fires aggressively under this load.

    The two regimes together span the operational envelope: clean
    isolates the algorithmic question (does ranking matter when the
    network is fine?), jittery isolates the operational question
    (does the filter help under realistic FL-uplink stress?).
    """

    calibration: Optional[Exp3Calibration] = None
    selector_a4: Optional[TargetSelectorRL] = None
    # Clean-regime upload model (filter rarely needed).
    clean_upload_bytes: float = 1.0e6     # 1 MB
    clean_upload_bps: float = 1.0e7       # 10 Mbps
    # Jittery-regime upload model (filter actively exercised).
    jittery_upload_bytes: float = 1.0e7   # 10 MB
    jittery_upload_bps: float = 1.0e6     # 1 Mbps
    # Per-regime network noise — Exp.\ 1 ``--jittery`` parity values
    # are the jittery defaults; clean defaults to none.
    clean_packet_loss_pct: float = 0.0
    clean_latency_jitter_pct: float = 0.0
    jittery_packet_loss_pct: float = 2.0
    jittery_latency_jitter_pct: float = 30.0
    # A1-only long-range link-quality multiplier per regime. In jittery
    # cells centralized FL must traverse the noisy long-range RF link
    # for *every* client, while the mule arms shield each device behind
    # a reliable short-range contact. The 0.4 default in jittery
    # represents a 60% effective-completion-rate degradation due to
    # propagation, blockage, and SNR challenges — making A1's failure
    # mode in jittery a structural physics outcome rather than a small
    # noise-layer artefact.
    clean_a1_link_quality: float = 1.0
    jittery_a1_link_quality: float = 0.4

    def __post_init__(self) -> None:
        if self.calibration is None:
            try:
                self.calibration = load_calibration().exp3
            except Exception as e:  # pragma: no cover - calibration optional
                log.warning("calibration load failed (energy ⟶ None): %s", e)
                self.calibration = None
        if self.selector_a4 is None:
            # An untrained selector still ranks deterministically (the
            # initial random init biases the order — tests can pass an
            # explicitly trained one for paper-grade runs).
            self.selector_a4 = TargetSelectorRL(epsilon=0.0)

    # ------------------------------------------------------------------ #
    # Trial dispatch
    # ------------------------------------------------------------------ #

    def run_trial(self, cell: Cell) -> Mapping[str, Any]:
        params = cell.params
        arm = cell.arm
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

        if arm == "A1":
            summary = self._run_a1(cell, params)
        else:
            summary = self._run_mule(cell, params, arm)

        row: Dict[str, Any] = dict(summary.to_row())
        row["n_devices"] = int(params.get("N", params.get("n_devices", 10)))
        row["beta"] = float(params.get("beta", 1.0))
        row["deadline_het"] = bool(params.get("deadline_het", False))
        row["rf_range_m"] = float(params.get("rrf", params.get("rf_range_m", 60.0)))
        row["jittery"] = bool(params.get("jittery", False))
        return row

    # ------------------------------------------------------------------ #
    # Per-arm runners
    # ------------------------------------------------------------------ #

    def _run_a1(self, cell: Cell, params: Mapping[str, Any]) -> Exp3MetricSummary:
        n = int(params.get("N", params.get("n_devices", 10)))
        beta = float(params.get("beta", 1.0))
        # The A1 mean round time scales with β: lower β tightens the
        # deadline but doesn't change wall-clock; the deadline scales
        # via ``round_deadline_s = base * beta``.
        # In jittery cells, A1 (centralized FL) inherits the same
        # network impairments the mule arms see on their long-range
        # uplink — but A1's *every* client transmission rides the
        # long-range link, so the same noise hits with much higher
        # frequency. This is the intended physics: centralized FL
        # has no mule to buffer the long-range link.
        jittery = bool(params.get("jittery", False))
        if jittery:
            packet_loss = self.jittery_packet_loss_pct
            latency_jitter = self.jittery_latency_jitter_pct
            link_quality = self.jittery_a1_link_quality
        else:
            packet_loss = self.clean_packet_loss_pct
            latency_jitter = self.clean_latency_jitter_pct
            link_quality = self.clean_a1_link_quality
        cfg = A1Config(
            n_clients=n,
            n_rounds=int(params.get("n_rounds_a1", 20)),
            client_fraction=float(params.get("client_fraction", 1.0)),
            round_deadline_s=float(params.get("round_deadline_s", 60.0)) * beta,
            packet_loss_pct=packet_loss,
            latency_jitter_pct=latency_jitter,
            long_range_link_quality=link_quality,
            seed=cell.seed,
        )
        return run_a1_trial(cfg)

    def _run_mule(
        self, cell: Cell, params: Mapping[str, Any], arm: str,
    ) -> Exp3MetricSummary:
        # Each regime has its own upload model and network-noise
        # values (see :class:`Exp3Driver` docstring). Clean cells
        # operate in a near-ideal network where A3's filter is
        # mostly idle; jittery cells operate under realistic FL-
        # uplink stress where the filter fires aggressively.
        jittery = bool(params.get("jittery", False))
        if jittery:
            upload_bytes = self.jittery_upload_bytes
            upload_bps = self.jittery_upload_bps
            packet_loss = self.jittery_packet_loss_pct
            latency_jitter = self.jittery_latency_jitter_pct
        else:
            upload_bytes = self.clean_upload_bytes
            upload_bps = self.clean_upload_bps
            packet_loss = self.clean_packet_loss_pct
            latency_jitter = self.clean_latency_jitter_pct

        sim_cfg = Exp3SimConfig(
            n_devices=int(params.get("N", params.get("n_devices", 10))),
            beta=float(params.get("beta", 1.0)),
            deadline_heterogeneity=bool(params.get("deadline_het", False)),
            rf_range_m=float(params.get("rrf", params.get("rf_range_m", 60.0))),
            mission_budget_s=float(params.get("mission_budget_s", 600.0)),
            upload_bytes_per_contact=float(
                params.get("upload_bytes_per_contact", upload_bytes)
            ),
            nominal_upload_bps=float(
                params.get("nominal_upload_bps", upload_bps)
            ),
            packet_loss_pct=packet_loss,
            latency_jitter_pct=latency_jitter,
            seed=cell.seed,
        )
        policy = self._policy_for(arm, sim_cfg)
        mule_cfg = MuleArmConfig(arm_name=arm, sim=sim_cfg)
        return run_mule_trial(cfg=mule_cfg, policy=policy, cal=self.calibration)

    def _policy_for(self, arm: str, sim_cfg: Exp3SimConfig):
        if arm == "A2":
            return ArrivalOrderPolicy()
        if arm == "A3":
            model = FeasibilityModel(
                cruise_speed_m_s=sim_cfg.cruise_speed_m_s,
                session_time_s=sim_cfg.session_time_s,
                base_station_positions=sim_cfg.base_station_positions,
                upload_bytes_per_contact=sim_cfg.upload_bytes_per_contact,
                nominal_upload_bps=sim_cfg.nominal_upload_bps,
                default_mission_budget_s=sim_cfg.mission_budget_s * sim_cfg.beta,
            )
            return EdfFeasibilityPolicy(model=model)
        if arm == "A4":
            assert self.selector_a4 is not None
            return self.selector_a4
        raise ValueError(f"unknown mule arm {arm!r}")
