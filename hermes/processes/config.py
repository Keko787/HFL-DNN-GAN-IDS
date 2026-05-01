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
