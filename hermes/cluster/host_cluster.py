"""``HFLHostCluster`` — Tier-2 cluster-scope FL coordinator.

Per Design §2.5 / §6.7 this program owns:

* ``DeviceRegistry``        — the authoritative cluster registry,
* ``MissionSlice`` dispatch — disjoint per-mule slicing every dock,
* cross-mule FedAvg         — over partial aggregates from N mules,
* ``ClusterAmendment``s     — slow-phase corrections folded back DOWN,
* θ_gen + synth-batch hosting (delegated to a pluggable generator).

θ_gen never leaves Tier 2 (Design §7 principle 9). The generator object
is injected, not constructed here, so the cluster stays decoupled from
the GAN training stack and remains testable with a stub.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np

from hermes.transport import DockLink, DockLinkError
from hermes.types import (
    ClusterAmendment,
    DeviceID,
    DownBundle,
    MissionSlice,
    MuleID,
    PartialAggregate,
    UpBundle,
    sign_down_bundle,
)
from hermes.types.aggregate import Weights

from .cross_mule_fedavg import FedAvgError, cross_mule_fedavg
from .device_registry import DeviceRegistry

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pluggable generator surface
# --------------------------------------------------------------------------- #

class GeneratorHost(Protocol):
    """Minimum surface the cluster needs from the θ_gen + synth pipeline.

    This is intentionally tiny — the heavy AC-GAN code in
    ``App/TrainingApp/HFLHost`` plugs in unchanged behind this protocol.
    """

    def make_synth_batch(self, n: int) -> List[np.ndarray]: ...

    def get_global_disc_weights(self) -> Weights: ...

    def update_disc_from_cluster_avg(self, weights: Weights) -> None:
        """Apply the post-FedAvg discriminator weights into the held global."""


@dataclass
class StubGeneratorHost:
    """Placeholder used in tests + the Phase-1 demo.

    Carries the discriminator weights as a list of numpy arrays and emits
    fixed-shape zero tensors as 'synth samples'. Real generator plugs in
    via the ``GeneratorHost`` protocol.
    """

    disc_weights: Weights
    synth_shape: Tuple[int, ...] = (8,)

    def make_synth_batch(self, n: int) -> List[np.ndarray]:
        return [np.zeros(self.synth_shape, dtype=np.float32) for _ in range(n)]

    def get_global_disc_weights(self) -> Weights:
        # return copies so downstream mutation can't leak back in
        return [w.copy() for w in self.disc_weights]

    def update_disc_from_cluster_avg(self, weights: Weights) -> None:
        self.disc_weights = [w.copy() for w in weights]


# --------------------------------------------------------------------------- #
# HFLHostCluster
# --------------------------------------------------------------------------- #

@dataclass
class _PendingRound:
    """Per-cluster-round state collected as mules dock."""

    cluster_round: int
    started_at: float
    partials: List[PartialAggregate]
    seen_mules: List[MuleID]


class HFLHostCluster:
    """Cluster-scope coordinator. One instance per edge server."""

    def __init__(
        self,
        *,
        registry: DeviceRegistry,
        generator: GeneratorHost,
        dock: DockLink,
        synth_batch_size: int = 32,
        min_participation: int = 1,
    ) -> None:
        self.registry = registry
        self.generator = generator
        self.dock = dock
        self.synth_batch_size = synth_batch_size
        self.min_participation = min_participation

        self._cluster_round: int = 0
        self._pending: Optional[_PendingRound] = None
        self._lock = threading.RLock()

        # last amendment we shipped — kept so a re-docking mule can see what
        # it last acknowledged.
        self._last_amendment: ClusterAmendment = ClusterAmendment(cluster_round=0)

    # -------------------------------------------------------------- registry

    def known_mules(self) -> List[MuleID]:
        """Return the list of mules currently assigned in the registry."""
        return sorted(self.registry.snapshot().by_mule.keys())

    # ----------------------------------------------------------- dock ingest

    def ingest_up_bundle(self, bundle: UpBundle) -> None:
        """Accept a mule's mission output. Folds the round-close report
        into per-device counters and parks the partial for cross-mule FedAvg.
        """
        with self._lock:
            self._ensure_pending_round()
            assert self._pending is not None  # for type-checkers

            if bundle.mule_id in self._pending.seen_mules:
                log.warning(
                    "duplicate UpBundle from mule=%s in cluster_round=%d "
                    "(ignoring later submission)",
                    bundle.mule_id,
                    self._pending.cluster_round,
                )
                return

            self._pending.partials.append(bundle.partial_aggregate)
            self._pending.seen_mules.append(bundle.mule_id)

            # apply per-device counter updates from the round-close report
            for line in bundle.round_close_report.lines:
                self.registry.update_after_round(
                    device_id=line.device_id,
                    on_time=line.outcome.is_on_time(),
                )

            log.info(
                "ingested UpBundle mule=%s round=%d devices=%d on_time=%d missed=%d",
                bundle.mule_id,
                self._pending.cluster_round,
                len(bundle.round_close_report.lines),
                *bundle.round_close_report.counts(),
            )

    # ----------------------------------------------- cross-mule aggregation

    def aggregate_pending(self) -> Optional[Weights]:
        """Run cross-mule FedAvg if enough partials have arrived.

        Returns the merged weights or ``None`` if the participation
        threshold is not met. The merged weights are also pushed back into
        the held generator's global discriminator state.
        """
        with self._lock:
            if self._pending is None:
                return None
            if len(self._pending.partials) < self.min_participation:
                log.debug(
                    "aggregate_pending: %d partials < min=%d",
                    len(self._pending.partials),
                    self.min_participation,
                )
                return None
            try:
                merged = cross_mule_fedavg(self._pending.partials)
            except FedAvgError:
                log.exception("cross_mule_fedavg failed; dropping cluster round")
                self._reset_pending()
                raise
            self.generator.update_disc_from_cluster_avg(merged)
            return merged

    # ----------------------------------------------- bundle DOWN dispatch

    def make_mission_slice(self, mule_id: MuleID) -> MissionSlice:
        """Read the current slice for one mule (no rebalance)."""
        device_ids = self.registry.slice_for(mule_id)
        return MissionSlice(
            mule_id=mule_id,
            device_ids=device_ids,
            issued_round=self._cluster_round,
            issued_at=time.time(),
        )

    def rebalance_for(
        self, mule_ids: Iterable[MuleID]
    ) -> Dict[MuleID, MissionSlice]:
        """Rebalance the registry across the given mules.

        Returns the freshly-issued ``MissionSlice`` per mule. Caller hands
        each slice into the corresponding ``DownBundle``.
        """
        return self.registry.rebalance(
            mule_ids,
            round_counter=self._cluster_round,
        )

    def dispatch_down_bundle(
        self,
        mule_id: MuleID,
        *,
        amendment: Optional[ClusterAmendment] = None,
    ) -> DownBundle:
        """Build the DOWN bundle for a departing mule.

        Caller may pass a freshly-built ``ClusterAmendment``; otherwise the
        last one shipped is reused (keeps the contract stable across calls).
        """
        with self._lock:
            mission_slice = self.make_mission_slice(mule_id)
            bundle = DownBundle(
                mule_id=mule_id,
                mission_slice=mission_slice,
                theta_disc=self.generator.get_global_disc_weights(),
                synth_batch=self.generator.make_synth_batch(self.synth_batch_size),
                cluster_amendments=amendment or self._last_amendment,
            )
            sign_down_bundle(bundle)
            return bundle

    # ----------------------------------------------- end-of-round close

    def close_cluster_round(
        self,
        *,
        deadline_overrides: Optional[Dict[DeviceID, float]] = None,
        notes: str = "",
    ) -> ClusterAmendment:
        """Finalise the current cluster round and produce an amendment.

        Increments ``cluster_round``, clears pending state, and stores the
        amendment so future ``dispatch_down_bundle`` calls can reuse it.
        """
        with self._lock:
            self._cluster_round += 1
            amendment = ClusterAmendment(
                cluster_round=self._cluster_round,
                deadline_overrides=dict(deadline_overrides or {}),
                notes=notes,
            )
            self._last_amendment = amendment
            self._reset_pending()
            log.info("closed cluster_round=%d", self._cluster_round)
            return amendment

    # ----------------------------------------------- dock-link server loop

    def serve_one_dock(
        self,
        *,
        timeout: Optional[float] = None,
        amendment_for_down: Optional[ClusterAmendment] = None,
    ) -> Tuple[UpBundle, DownBundle]:
        """Block on one mule's UP, then send that mule its DOWN bundle.

        Convenience wrapper around the dock link for tests + the Phase 1
        demo. A real long-running server (Phase 6) will spin a thread per
        mule around this primitive.
        """
        try:
            up = self.dock.recv_up(timeout=timeout)
        except DockLinkError:
            log.exception("dock recv_up failed")
            raise
        self.ingest_up_bundle(up)
        down = self.dispatch_down_bundle(
            up.mule_id, amendment=amendment_for_down
        )
        self.dock.send_down(down)
        return up, down

    # ----------------------------------------------- properties / introspection

    @property
    def cluster_round(self) -> int:
        with self._lock:
            return self._cluster_round

    def pending_partials(self) -> int:
        with self._lock:
            return 0 if self._pending is None else len(self._pending.partials)

    # ----------------------------------------------- internal

    def _ensure_pending_round(self) -> None:
        if self._pending is None:
            self._pending = _PendingRound(
                cluster_round=self._cluster_round + 1,
                started_at=time.time(),
                partials=[],
                seen_mules=[],
            )

    def _reset_pending(self) -> None:
        self._pending = None
