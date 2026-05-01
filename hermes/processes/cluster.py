"""Sprint 2 — cluster-process entry point + service loop.

Run with::

    python -m hermes.processes.cluster --config /path/to/cluster.json

The cluster process:

1. Reads the :class:`ClusterConfig` from the JSON file on argv.
2. Stands up an empty :class:`DeviceRegistry` (positions arrive via
   the registered mules' UP bundles' contact_history; alternatively,
   tests may pre-populate via direct registry calls — but in the
   multi-process flow, the orchestrator pre-seeds the registry by
   issuing registry.register calls for every device the cluster owns).
3. Builds a :class:`HFLHostCluster` with a :class:`TCPDockLinkServer`.
4. Optionally connects an :class:`HTTPCloudLink` to Tier-3 if
   ``cluster.tier3_url`` is set.
5. Runs the service loop: wait for expected mules → dispatch initial
   DOWN to each → loop {recv UP, ingest, aggregate when quorum,
   dispatch DOWN to each docked mule}.
6. Exits cleanly on SIGTERM / SIGINT.

Logs go to stderr in plain text. Chunk M wraps these in structured JSON.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.observability import (
    JsonEventEmitter,
    MetricsRegistry,
    NullEventEmitter,
)
from hermes.transport import (
    HTTPCloudLink,
    TCPDockLinkServer,
)
from hermes.types import DeviceID, MuleID, SpectrumSig

from .config import ClusterConfig, cluster_config_from_json

log = logging.getLogger("hermes.processes.cluster")


def _spectrum_sig_from_raw(raw: Optional[dict]) -> SpectrumSig:
    """L-L3: build a SpectrumSig from a JSON-shaped dict (or fallback).

    Accepts ``{"bands": [...], "last_good_snr_per_band": [...]}`` with
    list-or-tuple values. ``None`` returns the placeholder used pre-RF-survey.
    """
    if raw is None:
        return SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,))
    bands = tuple(int(b) for b in raw.get("bands", (0,)))
    snrs = tuple(float(s) for s in raw.get("last_good_snr_per_band", (20.0,)))
    if len(bands) != len(snrs):
        raise ValueError(
            f"spectrum_sig bands/last_good_snr_per_band length mismatch: "
            f"{bands!r} vs {snrs!r}"
        )
    return SpectrumSig(bands=bands, last_good_snr_per_band=snrs)


class ClusterService:
    """Lifecycle holder for a cluster-process service loop."""

    def __init__(
        self,
        cfg: ClusterConfig,
        *,
        events: Optional[JsonEventEmitter] = None,
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        self.cfg = cfg
        self._stop_event = threading.Event()

        # Chunk M observability — null defaults so tests can construct the
        # service without setting up a JSONL file. The CLI entry point
        # below builds a real emitter when ``--run-dir`` is supplied.
        self.events = events or NullEventEmitter(role="cluster", node_id=cfg.cluster_id)
        self.metrics = metrics or MetricsRegistry()

        self.dock = TCPDockLinkServer(host=cfg.dock_host, port=cfg.dock_port)
        self.dock.start()
        # Read back the actual port — the orchestrator may have asked
        # for ephemeral.
        self.actual_dock_port = self.dock.port

        self.registry = DeviceRegistry()
        # Pre-seed the registry from the config's seed_devices list so
        # the very first DOWN bundle dispatches a populated MissionSlice.
        # Without this the slice is empty and the mule's contact queue
        # is empty, every mission fails immediately with
        # "no submissions to aggregate".
        self._seed_registry_from_config()

        self.generator = StubGeneratorHost(
            disc_weights=[
                np.zeros((4,), dtype=np.float32),
                np.ones((3, 3), dtype=np.float32) * 0.01,
            ]
        )
        self.cluster = HFLHostCluster(
            registry=self.registry,
            generator=self.generator,
            dock=self.dock,
            synth_batch_size=cfg.synth_batch_size,
            min_participation=cfg.min_participation,
        )

        # Optional Tier-3 outbound link.
        self.cloud: Optional[HTTPCloudLink] = None
        if self.cfg.tier3_url:
            self.cloud = HTTPCloudLink(base_url=self.cfg.tier3_url)

        self.events.emit(
            "cluster_ready",
            dock_host=self.cfg.dock_host,
            dock_port=self.actual_dock_port,
            expected_mules=list(self.cfg.expected_mules),
            seed_devices=len(self.cfg.seed_devices),
            synth_batch_size=self.cfg.synth_batch_size,
            min_participation=self.cfg.min_participation,
            tier3_wired=self.cloud is not None,
        )

    def _seed_registry_from_config(self) -> None:
        """Register every seed device + rebalance across listed mules.

        L-L3: each seed_devices entry may include a ``spectrum_sig``
        field with ``{bands, last_good_snr_per_band}`` keys; without it
        we fall back to the placeholder single-band 20 dB prior. Real
        deployments populate the priors from the offline RF survey
        before launch.
        """
        if not self.cfg.seed_devices:
            return

        # Group devices by their assigned mule so we can rebalance
        # disjointly. Devices without an assigned_mule fall to the
        # first mule in expected_mules (single-mule deployments).
        for raw in self.cfg.seed_devices:
            did = raw["device_id"]
            pos = tuple(raw.get("position", (0.0, 0.0, 0.0)))
            self.registry.register(
                device_id=DeviceID(did),
                position=pos,
                spectrum_sig=_spectrum_sig_from_raw(raw.get("spectrum_sig")),
            )

        # Rebalance: build a map mule_id → [device_ids] then call
        # registry.rebalance with the list of mules. The DeviceRegistry's
        # rebalance distributes devices round-robin across mules; for
        # deterministic per-device assignment we explicitly assign.
        if self.cfg.expected_mules:
            mules = [MuleID(m) for m in self.cfg.expected_mules]
            self.registry.rebalance(mules, round_counter=0)
            # Then override assignments per the config's per-device map.
            for raw in self.cfg.seed_devices:
                did = raw["device_id"]
                assigned = raw.get("assigned_mule")
                if assigned:
                    rec = self.registry.get(DeviceID(did))
                    if rec is not None:
                        rec.assigned_mule = MuleID(assigned)
            log.info(
                "cluster %s pre-seeded %d devices across mules %s",
                self.cfg.cluster_id, len(self.cfg.seed_devices),
                self.cfg.expected_mules,
            )

    def seed_registry_from_devices(
        self, devices: List["DeviceSeed"], mule_id: MuleID,
    ) -> None:
        """Pre-populate the registry before mules dock.

        Called by the orchestrator (chunk L) so the very first DOWN
        bundle dispatched to a mule contains a populated MissionSlice.
        """
        for d in devices:
            self.registry.register(
                device_id=d.device_id,
                position=d.position,
                spectrum_sig=SpectrumSig(
                    bands=(0,), last_good_snr_per_band=(20.0,),
                ),
            )
        self.registry.rebalance([mule_id], round_counter=0)

    def request_stop(self) -> None:
        self._stop_event.set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    # L-L6: cap how often we poll Tier-3. Every loop iteration would
    # mean ~1 poll/s with a 0.5 s timeout each — burns a thread for no
    # benefit. Tier-3 refinements arrive on cluster-round cadence (tens
    # of seconds), so 5 s is plenty.
    _TIER3_POLL_INTERVAL_S: float = 5.0

    def run(self) -> None:
        """Service loop — runs until ``request_stop`` is called.

        Loop:
            1. Wait for every expected mule to register (with a long
               but bounded timeout).
            2. Dispatch the initial DOWN bundle to each mule
               (bootstrap; gives the mule its slice + θ).
            3. Loop forever:
                 a. Try recv_up (1s timeout).
                 b. L-M1: check stop_event before doing the ingest work
                    (we may have been signalled while blocked on recv).
                 c. On UP arrival: ingest, then if min_participation
                    is met, run cross-mule FedAvg + close round +
                    dispatch fresh DOWN to every currently-docked mule.
                 d. L-H2: detect newly-docked mules each iteration and
                    dispatch DOWN to them so a reconnecting mule doesn't
                    sit slice-less waiting for the next aggregation.
                 e. L-L6: periodic Tier-3 poll on a throttled cadence.
        """
        expected_mules = [MuleID(m) for m in self.cfg.expected_mules]
        log.info(
            "cluster %s ready on dock 127.0.0.1:%d, expecting %d mule(s)",
            self.cfg.cluster_id, self.actual_dock_port, len(expected_mules),
        )

        if expected_mules:
            if not self.dock.wait_for_mules(expected_mules, timeout=60.0):
                log.error(
                    "cluster %s: not all mules registered within 60s "
                    "(saw %s, wanted %s); proceeding with whoever's here",
                    self.cfg.cluster_id,
                    sorted(self.registry.snapshot().by_mule.keys()),
                    expected_mules,
                )

        # L-H2: track mules we've already bootstrapped so we can detect
        # mid-flight reconnects (mule died, restarted, redocked) and
        # send them a fresh DOWN bundle without waiting for the next
        # aggregation cycle.
        bootstrapped: set = set()
        self._dispatch_to_new_mules(bootstrapped)

        last_tier3_poll = 0.0

        # Service loop.
        while not self._stop_event.is_set():
            try:
                up = self.dock.recv_up(timeout=1.0)
            except Exception:
                up = None

            # L-M1: bail out before we start work if shutdown was
            # requested while we were blocked in recv_up. Otherwise we
            # might burn a full ingest+aggregate+dispatch cycle after
            # the operator pressed Ctrl-C.
            if self._stop_event.is_set():
                break

            # L-H2: pick up any mule that docked (or re-docked) since
            # last iteration, regardless of whether an UP arrived.
            self._dispatch_to_new_mules(bootstrapped)

            if up is not None:
                try:
                    self.cluster.ingest_up_bundle(up)
                    self.events.emit(
                        "up_bundle_ingested",
                        mule_id=str(up.mule_id),
                        mission_round=getattr(up, "mission_round", None),
                    )
                    self.metrics.increment("up_bundles_ingested")
                    merged = self.cluster.aggregate_pending()
                    if merged is not None:
                        self.cluster.close_cluster_round()
                        self.events.emit(
                            "cluster_round_closed",
                            cluster_round=self.cluster._cluster_round,
                        )
                        self.metrics.increment("cluster_rounds_closed")
                        # Dispatch a fresh DOWN to every currently-docked
                        # mule (registered_mules, not expected_mules —
                        # if a mule is offline we'd just hit a write
                        # error trying to send down a dead socket).
                        for mid in self.dock.registered_mules():
                            try:
                                self.dock.send_down(
                                    self.cluster.dispatch_down_bundle(mid)
                                )
                                self.metrics.increment("down_bundles_dispatched")
                            except Exception:
                                log.exception(
                                    "post-aggregation DOWN failed for %s",
                                    mid,
                                )
                                self.metrics.increment("dispatch_down_failures")
                except Exception:
                    log.exception("ingest_up_bundle / aggregate failed")
                    self.metrics.increment("ingest_failures")

            now = time.time()
            if now - last_tier3_poll >= self._TIER3_POLL_INTERVAL_S:
                self._poll_tier3_if_wired()
                last_tier3_poll = now

        log.info("cluster %s service loop exiting", self.cfg.cluster_id)

    def _dispatch_to_new_mules(self, bootstrapped: set) -> None:
        """L-H2: dispatch a DOWN bundle to any mule we haven't yet.

        ``bootstrapped`` is mutated in place so the caller's tracking
        set stays accurate across iterations.
        """
        for mid in self.dock.registered_mules():
            if mid in bootstrapped:
                continue
            try:
                self.dock.send_down(self.cluster.dispatch_down_bundle(mid))
                bootstrapped.add(mid)
                log.info("cluster %s: DOWN dispatched to mule %s",
                         self.cfg.cluster_id, mid)
                self.events.emit("mule_bootstrapped", mule_id=str(mid))
                self.metrics.increment("mules_bootstrapped")
            except Exception:
                log.exception("DOWN dispatch to %s failed", mid)
                self.metrics.increment("dispatch_down_failures")

    def _poll_tier3_if_wired(self) -> None:
        if self.cloud is None:
            return
        # Best-effort, non-fatal. Phase 7: when Tier-3 returns a refinement
        # (HTTP 200 with a pickled GeneratorRefinement), fold it into the
        # cluster's GeneratorHost so subsequent ``make_synth_batch`` calls
        # draw from the cross-cluster aggregated θ_gen. A 204 (no pending
        # refinement) returns ``None`` and we just loop. Errors are
        # transient — Tier-3 is outbound polling, never on the hot path.
        try:
            refinement = self.cloud.poll_refinement(
                self.cfg.cluster_id, timeout_s=0.5,
            )
        except Exception:
            log.debug("tier3 poll failed (transient)")
            self.metrics.increment("tier3_poll_failures")
            return
        if refinement is None:
            return
        try:
            self.generator.apply_tier3_gen_refinement(
                refinement.theta_gen,
                refinement_round=refinement.refinement_round,
            )
            self.events.emit(
                "tier3_refinement_applied",
                refinement_round=refinement.refinement_round,
                notes=refinement.notes,
            )
            self.metrics.increment("tier3_refinements_applied")
        except Exception:
            log.exception(
                "tier3 refinement fold failed (round=%s)",
                refinement.refinement_round,
            )
            self.metrics.increment("tier3_refinement_fold_failures")

    def shutdown(self) -> None:
        self.request_stop()
        try:
            self.dock.close()
        except Exception:
            pass
        if self.cloud is not None:
            try:
                self.cloud.close()
            except Exception:
                pass
        try:
            self.events.emit("metrics_snapshot", metrics=self.metrics.snapshot())
            self.events.emit("service_stopped")
            self.events.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

class DeviceSeed:
    """Lightweight value type for pre-seeding the registry from the orchestrator."""

    def __init__(self, device_id, position):
        self.device_id = device_id
        self.position = position


def _install_signal_handlers(svc: ClusterService) -> None:
    def _handle(_signum, _frame):
        log.info("cluster received shutdown signal")
        svc.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):  # pragma: no cover — non-main thread
            pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes.processes.cluster")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--port-out",
        type=Path,
        help=(
            "If set, write the actual bound dock port to this file "
            "after start. Used by the orchestrator when the config "
            "asks for an ephemeral port (port=0)."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Chunk M observability: directory where the per-process "
            "JSONL event log is written. Filename is "
            "``cluster-<cluster_id>.jsonl``. If omitted, events are "
            "dropped (NullEventEmitter); useful for ad-hoc CLI runs."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cfg = cluster_config_from_json(args.config.read_text(encoding="utf-8"))

    events: Optional[JsonEventEmitter] = None
    if args.run_dir is not None:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        events = JsonEventEmitter(
            args.run_dir / f"cluster-{cfg.cluster_id}.jsonl",
            role="cluster",
            node_id=cfg.cluster_id,
        )

    svc = ClusterService(cfg, events=events)
    _install_signal_handlers(svc)

    if args.port_out is not None:
        args.port_out.write_text(str(svc.actual_dock_port), encoding="utf-8")
        log.info("cluster wrote actual dock port %d to %s",
                 svc.actual_dock_port, args.port_out)

    try:
        svc.run()
    finally:
        svc.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
