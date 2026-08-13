"""Sprint 2 — multi-process topology configuration.

One :class:`TopologyConfig` describes the AVN-shaped layout the
orchestrator brings up: 1 cluster, N mules, M devices, with all the
host:port pairs that wire them together. The orchestrator (chunk L)
serializes per-role configs to JSON files; each entry-point script
(``hermes.processes.{cluster,mule,device}``) reads its config from a
``--config`` arg and runs its service loop.

Maps onto AERPAW's AVN model 1:1:

* Cluster config → one fixed AVN running ``HFLHostCluster``.
* Mule config → one mobile AVN per mule, running ``MuleSupervisor``.
* Device config → one fixed or mobile AVN per device, running
  ``ClientMission``.

When AERPAW returns, the only thing that changes is the host strings
(localhost → AVN routable IPs); the rest of the wiring stays.

Schema is plain dataclasses with JSON helpers — no extra deps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


Position = Tuple[float, float, float]


@dataclass
class _SeedDevice:
    """Lightweight device registration row for cluster pre-seeding."""

    device_id: str
    position: Position = (0.0, 0.0, 0.0)
    assigned_mule: Optional[str] = None


@dataclass
class ClusterConfig:
    """Settings for the single edge-server (cluster) process."""

    cluster_id: str
    # TCP host/port the dock listens on. Mules connect here.
    dock_host: str = "127.0.0.1"
    dock_port: int = 0  # 0 = ephemeral; orchestrator reads it back
    # Mules expected to register before the cluster dispatches the
    # first DOWN bundle. The orchestrator populates this from the
    # topology — the cluster service waits until all show up.
    expected_mules: List[str] = field(default_factory=list)
    # Devices to pre-seed in the registry before any mule docks. Each
    # entry carries (device_id, position, assigned_mule). The cluster
    # registers them and rebalances onto the listed mules so the very
    # first DOWN bundle dispatches a populated MissionSlice.
    seed_devices: List[dict] = field(default_factory=list)
    # Cluster-controlled tunables.
    synth_batch_size: int = 4
    # L-L2: minimum number of mules that must contribute an UP bundle
    # before the cluster aggregates and closes a round. Set to the
    # number of mules in the topology for full-FedAvg semantics; set to
    # 1 for partial-FedAvg (cluster aggregates as soon as any mule
    # reports, accepting staleness from absent mules). Defaults to 1
    # because in Sprint 2 demos we want forward progress with a single
    # mule; production deployments typically pin this to len(mules).
    min_participation: int = 1
    # Optional Tier-3 endpoint (cloud link). When set, the cluster
    # service polls / posts on its own cadence. None = no cloud link.
    tier3_url: Optional[str] = None
    # EX-4.1 — real DNN-IDS. When ``init_theta_path`` is set, the global
    # model is seeded from those weights (real create_CICIOT_Model) instead
    # of the 13-param stub, so the whole pipeline carries the real shapes.
    # When ``eval_test_path`` + ``input_dim`` are set, the cluster evaluates
    # the aggregated θ on the held-out test set after each round and emits a
    # ``model_eval`` event (accuracy/auc/loss). All None -> the stub path.
    init_theta_path: Optional[str] = None
    eval_test_path: Optional[str] = None
    input_dim: Optional[int] = None
    # EX-4.2 — long-range mule->base-station backhaul loss (%). Under the
    # jittery regime the mule's aggregate upload drops with this probability
    # per dock; the round does not close but θ' still flows for Pass 2, so
    # the update is carried, not lost (recoverable — unlike H0's permanent
    # dead-zone). 0.0 -> reliable backhaul.
    backhaul_loss_pct: float = 0.0
    # Seed for the backhaul-loss RNG so the loss pattern is deterministic and
    # varies per trial. Set by the driver from the paired trial seed.
    backhaul_rng_seed: Optional[int] = None
    # EX-4.3 arm H3 — per-mission backhaul-loss probabilities from the L1
    # channel model (index = mission_round-1). When set, it overrides the
    # flat ``backhaul_loss_pct``: adaptive channel selection (H3) yields a
    # lower-loss schedule than the fixed channel (H1/H2), so L1's effect is
    # a real, seed-consistent reduction in dropped rounds.
    backhaul_loss_schedule: Optional[List[float]] = None


@dataclass
class MuleConfig:
    """Settings for one mule process."""

    mule_id: str
    # Mule's RF link is a TCP server — devices connect inbound to here.
    rf_host: str = "127.0.0.1"
    rf_port: int = 0
    # Cluster's dock to connect outbound to.
    dock_host: str = "127.0.0.1"
    dock_port: int = 0
    # Devices expected to register on the RF link before the mule
    # starts running missions (otherwise contacts would broadcast to
    # an empty room). Populated by the orchestrator from the topology.
    expected_devices: List[str] = field(default_factory=list)
    # Two-pass / clustering tunables. ``rf_range_m=None`` keeps the
    # legacy single-pass path (Sprint 1A); set it to enable Sprint 1.5.
    rf_range_m: Optional[float] = 60.0
    session_ttl_s: float = 5.0
    # Number of mission cycles to run before the service exits. None =
    # run until shutdown signal.
    n_missions: Optional[int] = None
    # EX-4.2 arm H2 — RL target selector (S3.5 tie-break). When
    # ``use_rl_selector`` is set, the mule builds a ``TargetSelectorRL`` and
    # passes it to its supervisor (deterministic distance ranking otherwise,
    # = arm H1). ``selector_weights_path`` loads a trained DDQN (.npz from
    # experiments.exp3.train_a4); omit it for a random-init selector (smoke
    # only — not paper-grade).
    use_rl_selector: bool = False
    selector_weights_path: Optional[str] = None
    # EX-4.3 arm H3 — the L1->L2 edge. When set, the mule's scheduler feeds
    # this (the mean effective SNR of the L1-chosen channel) to the target
    # selector's rf_prior feature, instead of the hardcoded 20.0 default. This
    # is the real RF prior the SEC26 audit found was never wired at runtime.
    rf_prior_snr_db: Optional[float] = None
    # S3b — per-mission time budget (seconds). When set, the scheduler's
    # deadline feasibility gate is ACTIVE: contacts that cannot be reached
    # before their own deadline, or that would overrun this budget, are
    # dropped before ordering. ``None`` keeps the historical behaviour in
    # which Deadline(j) is only a sort key.
    mission_budget_s: Optional[float] = None


@dataclass
class DeviceConfig:
    """Settings for one edge-device process."""

    device_id: str
    # Mule whose RF this device connects to.
    mule_rf_host: str = "127.0.0.1"
    mule_rf_port: int = 0
    position: Position = (0.0, 0.0, 0.0)
    # Number of solicits to serve before exiting. None = run forever.
    n_serves: Optional[int] = None
    # EX-4.1 — real DNN-IDS training. When ``train_shard_path`` points at a
    # serialized ``(X, y)`` CICIOT shard, the device builds a real
    # ``local_train`` over it (experiments.exp4.model_task) instead of the
    # noise stub. ``input_dim`` must match the cluster's seeded model.
    # Left None -> the Sprint-2 stub trainer (backward compatible).
    train_shard_path: Optional[str] = None
    input_dim: Optional[int] = None
    local_epochs: int = 1
    local_batch_size: int = 64
    # EX-4.2 — per-device short-range contact reliability (device<->mule).
    # p that a Pass-1 collect delivers this device's Δθ, modelling Exp 3's
    # ``reliability x rf_factor`` completion. None -> always completes
    # (the EX-4.0/4.1 behaviour). Set by the driver from a seeded
    # Uniform(0.15, 1.0) reliability x distance falloff.
    contact_reliability: Optional[float] = None


class TopologyValidationError(ValueError):
    """Raised by :meth:`TopologyConfig.validate` on a malformed deployment."""


@dataclass
class TopologyConfig:
    """One AVN-shaped deployment description."""

    cluster: ClusterConfig
    mules: List[MuleConfig] = field(default_factory=list)
    devices: List[DeviceConfig] = field(default_factory=list)
    # L-M4: per-device → mule assignment. Populated by ``validate()`` from
    # MuleConfig.expected_devices, or round-robin if not specified. The
    # orchestrator reads this map (NOT MuleConfig.expected_devices
    # directly) to avoid the L-H1 bug where assignment depends on
    # config fields that haven't been populated yet.
    device_to_mule: Dict[str, str] = field(default_factory=dict)

    # ------------------------- Validation ---------------------------- #

    def validate(self) -> None:
        """Catch malformed topologies before subprocesses launch.

        Sprint 2 L-M4: empty mules with non-empty devices, dangling
        ``assigned_mule`` references, duplicate IDs, conflicting ports.
        Also populates :attr:`device_to_mule` so later steps don't have
        to re-derive assignment.
        """
        # Duplicate ID check.
        mule_ids = [m.mule_id for m in self.mules]
        if len(set(mule_ids)) != len(mule_ids):
            raise TopologyValidationError(
                f"duplicate mule_id in topology: {mule_ids}"
            )
        device_ids = [d.device_id for d in self.devices]
        if len(set(device_ids)) != len(device_ids):
            raise TopologyValidationError(
                f"duplicate device_id in topology: {device_ids}"
            )

        # Mule with devices but no mule.
        if self.devices and not self.mules:
            raise TopologyValidationError(
                f"{len(self.devices)} devices configured but no mules to serve them"
            )

        # Conflicting non-zero ports across mules.
        nonzero_rf = [m.rf_port for m in self.mules if m.rf_port != 0]
        if len(set(nonzero_rf)) != len(nonzero_rf):
            raise TopologyValidationError(
                f"conflicting non-zero rf_port across mules: {nonzero_rf}"
            )

        # Build / validate the device → mule assignment.
        # Strategy:
        #   1. If MuleConfig.expected_devices is populated, honour it.
        #   2. Otherwise, round-robin distribute devices across mules
        #      in declaration order (deterministic).
        # An assigned_mule that doesn't reference a real mule is rejected.
        explicit: Dict[str, str] = {}
        for m in self.mules:
            for did in m.expected_devices:
                if did in explicit:
                    raise TopologyValidationError(
                        f"device {did!r} is in expected_devices of multiple mules"
                    )
                if did not in device_ids:
                    raise TopologyValidationError(
                        f"mule {m.mule_id!r} expected_devices references "
                        f"unknown device {did!r}"
                    )
                explicit[did] = m.mule_id

        # Round-robin everything not explicitly claimed.
        assignment: Dict[str, str] = dict(explicit)
        unclaimed = [d.device_id for d in self.devices if d.device_id not in explicit]
        for i, did in enumerate(unclaimed):
            if not self.mules:
                # Empty-devices case already raised above; defensive.
                break
            assignment[did] = self.mules[i % len(self.mules)].mule_id

        self.device_to_mule = assignment

    def mule_for(self, device_id: str) -> str:
        """Return the mule assigned to ``device_id`` post-:meth:`validate`."""
        if not self.device_to_mule:
            raise TopologyValidationError(
                "topology not validated yet — call validate() first"
            )
        try:
            return self.device_to_mule[device_id]
        except KeyError:
            raise TopologyValidationError(
                f"no mule assignment for device {device_id!r}"
            )

    def devices_of(self, mule_id: str) -> List[str]:
        """Return device ids assigned to ``mule_id`` post-:meth:`validate`."""
        if not self.device_to_mule:
            raise TopologyValidationError(
                "topology not validated yet — call validate() first"
            )
        return [d for d, m in self.device_to_mule.items() if m == mule_id]

    # ------------------------- JSON helpers -------------------------- #

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "TopologyConfig":
        raw = json.loads(payload)
        return cls(
            cluster=ClusterConfig(**raw["cluster"]),
            mules=[MuleConfig(**m) for m in raw["mules"]],
            devices=[DeviceConfig(**d) for d in raw["devices"]],
            device_to_mule=dict(raw.get("device_to_mule", {})),
        )

    @classmethod
    def from_file(cls, path: Path) -> "TopologyConfig":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# Per-role config helpers — entry points read JSON of just one of these
# rather than the whole topology, so a single mule process doesn't see
# device positions it has no need for.

def cluster_config_to_json(cfg: ClusterConfig) -> str:
    return json.dumps(asdict(cfg), indent=2)


def cluster_config_from_json(payload: str) -> ClusterConfig:
    return ClusterConfig(**json.loads(payload))


def mule_config_to_json(cfg: MuleConfig) -> str:
    return json.dumps(asdict(cfg), indent=2)


def mule_config_from_json(payload: str) -> MuleConfig:
    return MuleConfig(**json.loads(payload))


def device_config_to_json(cfg: DeviceConfig) -> str:
    raw = asdict(cfg)
    # asdict converts the position tuple to a list — preserve the
    # tuple-shape on the inverse via a custom decoder below.
    return json.dumps(raw, indent=2)


def device_config_from_json(payload: str) -> DeviceConfig:
    raw = json.loads(payload)
    pos = raw.get("position", (0.0, 0.0, 0.0))
    if isinstance(pos, list):
        pos = tuple(pos)
    raw["position"] = pos
    return DeviceConfig(**raw)
