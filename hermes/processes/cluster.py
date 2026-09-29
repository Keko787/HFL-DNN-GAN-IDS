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
5. Runs the service loop: dispatch an initial DOWN to each expected
   mule as it registers → loop {recv UP, ingest, aggregate when quorum,
   dispatch DOWN to each mule waiting at the dock}.
6. Exits cleanly on SIGTERM / SIGINT.

Logs go to stderr in plain text. Chunk M wraps these in structured JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import (
    OUTCOME_DEFERRED,
    OUTCOME_EXPIRED,
    StubGeneratorHost,
)
from hermes.observability import (
    JsonEventEmitter,
    MetricsRegistry,
    NullEventEmitter,
)
from hermes.transport import (
    HTTPCloudLink,
    TCPDockLinkServer,
)
from hermes.mission.aggregation_rules import AGG_FEDBUFF
from hermes.types import (
    ContactHistory,
    DeviceID,
    MissionRoundCloseReport,
    MuleID,
    PartialAggregate,
    SpectrumSig,
    UpBundle,
)

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


def _up_mission_round(up) -> Optional[int]:
    """The mission round an UP bundle closes, or None if it carries none.

    ``UpBundle`` has no ``mission_round`` of its own; the round lives on its
    partial aggregate. Reading ``up.mission_round`` returned None, so the
    per-mission backhaul schedule was indexed at mission 1 for every mission
    and ``backhaul_upload_lost`` events carried no round.
    """
    pa = getattr(up, "partial_aggregate", None)
    mission_round = getattr(pa, "mission_round", None)
    return None if mission_round is None else int(mission_round)


def _lost_upload_stand_in(up, mission_round: Optional[int], spec) -> UpBundle:
    """The empty partial the cluster holds for ``up`` when its upload was lost.

    Tagged with the cluster's own rule so it passes the form check whatever the
    lost partial held, and with the lost partial's mission and base version so
    the fold's event fields name the right mission and age. Nothing else of the
    bundle is used: its model, round report and Pass-2 ledger were lost.
    """
    lost = up.partial_aggregate
    rnd = int(mission_round) if mission_round is not None else 0
    return UpBundle(
        mule_id=up.mule_id,
        partial_aggregate=PartialAggregate(
            mule_id=up.mule_id,
            mission_round=rnd,
            weights=[],
            num_examples=0,
            rule=spec.rule,
            update_form=spec.update_form,
            base_version=getattr(lost, "base_version", None),
        ),
        round_close_report=MissionRoundCloseReport(
            mule_id=up.mule_id, mission_round=rnd, started_at=0.0, finished_at=0.0,
        ),
        contact_history=ContactHistory(mule_id=up.mule_id, mission_round=rnd),
    )


def _mule_stream_key(mule_id) -> int:
    """A stable 32-bit key for ``mule_id``, to seed its own backhaul stream.

    ``hash()`` is salted per process, so it cannot seed anything that must
    reproduce across runs.
    """
    digest = hashlib.sha256(str(mule_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


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

        # EX-4.1 — seed the global model. When ``init_theta_path`` is set,
        # broadcast the *real* DNN-IDS weights so the whole pipeline carries
        # its shapes (partial_fedavg enforces shape consistency); otherwise
        # the Sprint-2 13-param stub the integration tests rely on.
        if getattr(cfg, "init_theta_path", None):
            from experiments.exp4.model_task import load_weights
            disc_weights = load_weights(cfg.init_theta_path)
            log.info(
                "cluster %s: seeded real DNN-IDS global model from %s "
                "(%d layers)",
                cfg.cluster_id, cfg.init_theta_path, len(disc_weights),
            )
        else:
            disc_weights = [
                np.zeros((4,), dtype=np.float32),
                np.ones((3, 3), dtype=np.float32) * 0.01,
            ]
        self.generator = StubGeneratorHost(disc_weights=disc_weights)

        # EX-4.1 — held-out eval set for per-round convergence (``model_eval``).
        # Loaded as plain numpy here (cheap); the TF evaluate happens in the
        # service loop so it never delays port binding at startup.
        self._eval_X = None
        self._eval_y = None
        self._eval_input_dim = getattr(cfg, "input_dim", None)
        # EX-4.2 — long-range backhaul (mule->BS) upload loss.
        self._backhaul_loss_pct = float(getattr(cfg, "backhaul_loss_pct", 0.0) or 0.0)
        self._backhaul_rng = np.random.default_rng(getattr(cfg, "backhaul_rng_seed", None))
        # FeRRy Phase 2 — with several mules, one shared stream would hand its
        # draws out in upload-arrival order, which depends on process timing,
        # so a paired seed would not reproduce the same losses. Each mule then
        # draws from its own stream, seeded from the trial's seed and its id.
        # One mule keeps the single stream above, draw for draw.
        self._backhaul_per_mule = len(cfg.expected_mules) > 1
        self._backhaul_mule_rngs: Dict[str, np.random.Generator] = {}
        # Mules blocked at the dock for a DOWN: each ingested UP adds its mule,
        # each DOWN that answers one removes it (FeRRy Phase 2). Ordered, no
        # repeats. Only these mules get a DOWN after a merge; a mule still in
        # flight would otherwise find a stale one queued at its next dock.
        self._awaiting: List[MuleID] = []
        # EX-4.3 — per-mission loss schedule (probabilities, 0..1) from the L1
        # channel model; overrides the flat pct when set.
        self._backhaul_loss_schedule = getattr(cfg, "backhaul_loss_schedule", None)
        if getattr(cfg, "eval_test_path", None) and self._eval_input_dim:
            from experiments.exp4.model_task import load_xy
            self._eval_X, self._eval_y = load_xy(cfg.eval_test_path)
            log.info(
                "cluster %s: loaded held-out eval set %s (rows=%d, dim=%s)",
                cfg.cluster_id, cfg.eval_test_path,
                len(self._eval_y), self._eval_input_dim,
            )

        # FeRRy Phase 1 — the L3 merge rule; agg:plain unless configured.
        from hermes.mission.aggregation_rules import AggregationSpec

        self.aggregation = AggregationSpec.from_config(
            getattr(cfg, "aggregation", None),
            getattr(cfg, "aggregation_params", None),
        )
        self.cluster = HFLHostCluster(
            registry=self.registry,
            generator=self.generator,
            dock=self.dock,
            synth_batch_size=cfg.synth_batch_size,
            min_participation=cfg.min_participation,
            aggregation=self.aggregation,
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

    #: How long startup waits for every expected mule to register, and how
    #: often it bootstraps the ones that already have (FeRRy Phase 2).
    _BOOTSTRAP_WAIT_S: float = 60.0
    _BOOTSTRAP_TICK_S: float = 1.0

    def run(self) -> None:
        """Service loop — runs until ``request_stop`` is called.

        Loop:
            1. Wait for every expected mule to register (with a long
               but bounded timeout), dispatching each one's initial DOWN
               bundle (bootstrap; gives the mule its slice + θ) as soon as
               it has registered.
            2. Loop forever:
                 a. Try recv_up (1s timeout).
                 b. L-M1: check stop_event before doing the ingest work
                    (we may have been signalled while blocked on recv).
                 c. On UP arrival: ingest, then if min_participation
                    is met, run cross-mule FedAvg + close round +
                    dispatch a fresh DOWN to every mule waiting at the
                    dock — the mules whose UP was ingested and not yet
                    answered, never a mule still in flight. When an
                    age-aware fold leaves θ unchanged without a quorum
                    wait, the round stays open and only the mules left
                    waiting get a DOWN: the uploader when FedBuff defers,
                    and every waiting mule when all the fold's partials
                    expired (or were empty). A lost backhaul upload is
                    answered at once, except under a quorum above 1, where
                    an empty partial holds the mule's place in the round.
                 d. L-H2: detect newly-docked mules each iteration and
                    dispatch DOWN to them so a reconnecting mule doesn't
                    sit slice-less waiting for the next aggregation.
                 e. L-L6: periodic Tier-3 poll on a throttled cadence.

        With one mule every DOWN still goes where it always went: that mule
        is the only uploader, the only mule waiting, and the only one docked.
        """
        expected_mules = [MuleID(m) for m in self.cfg.expected_mules]
        log.info(
            "cluster %s ready on dock 127.0.0.1:%d, expecting %d mule(s)",
            self.cfg.cluster_id, self.actual_dock_port, len(expected_mules),
        )

        # EX-4.1 — baseline convergence point (round 0, the seeded init θ),
        # before any aggregation. Also warms TensorFlow while mules register.
        self._emit_model_evaluation(0)

        # L-H2: track mules we've already bootstrapped so we can detect
        # mid-flight reconnects (mule died, restarted, redocked) and
        # send them a fresh DOWN bundle without waiting for the next
        # aggregation cycle.
        bootstrapped: set = set()
        if expected_mules:
            # FeRRy Phase 2: bootstrap each mule as soon as it registers. The
            # bootstraps used to wait for the last expected mule, up to 60 s,
            # while a mule gives up on its bootstrap after 30 s, so one slow
            # mule start could take every other mule down with it.
            deadline = time.monotonic() + self._BOOTSTRAP_WAIT_S
            while True:
                self._dispatch_to_new_mules(bootstrapped)
                if set(expected_mules) <= bootstrapped or self._stop_event.is_set():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.error(
                        "cluster %s: not all mules registered within %.0fs "
                        "(saw %s, wanted %s); proceeding with whoever's here",
                        self.cfg.cluster_id,
                        self._BOOTSTRAP_WAIT_S,
                        sorted(self.registry.snapshot().by_mule.keys()),
                        expected_mules,
                    )
                    break
                self.dock.wait_for_mules(
                    expected_mules, timeout=min(self._BOOTSTRAP_TICK_S, remaining),
                )
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

            up_round = _up_mission_round(up) if up is not None else None
            if up is not None and self._backhaul_dropped(up_round, up.mule_id):
                if self._lost_upload_holds_quorum():
                    self._hold_lost_upload(up, up_round)
                else:
                    # EX-4.2: model long-range mule->BS backhaul upload loss.
                    # Drop this mule's aggregate (the round does not close) but
                    # still send DOWN with the current θ so the mule can finish
                    # its two-pass mission — the update is carried, not lost
                    # (reconciled at a later dock, unlike H0's permanent loss).
                    self.events.emit(
                        "backhaul_upload_lost",
                        mule_id=str(up.mule_id),
                        mission_round=up_round,
                    )
                    self.metrics.increment("backhaul_uploads_lost")
                    self._stop_waiting(up.mule_id)
                    try:
                        self.dock.send_down(self.cluster.dispatch_down_bundle(up.mule_id))
                    except Exception:
                        log.exception("post-loss DOWN failed for %s", up.mule_id)
                up = None  # consumed as lost; skip the ingest path below

            if up is not None:
                try:
                    accepted = self.cluster.ingest_up_bundle(up)
                    # Waiting from here until a DOWN answers it, whether or not
                    # its partial was kept: either way the mule is blocked at
                    # its inter-pass dock.
                    self._start_waiting(up.mule_id)
                    if accepted:
                        self.events.emit(
                            "up_bundle_ingested",
                            mule_id=str(up.mule_id),
                            mission_round=up_round,
                        )
                        self.metrics.increment("up_bundles_ingested")
                        self._fold_pending(up.mule_id, up_round)
                    else:
                        self._note_refused_partial(up, up_round)
                except Exception:
                    log.exception("ingest_up_bundle / aggregate failed")
                    self.metrics.increment("ingest_failures")

            now = time.time()
            if now - last_tier3_poll >= self._TIER3_POLL_INTERVAL_S:
                self._poll_tier3_if_wired()
                last_tier3_poll = now

        log.info("cluster %s service loop exiting", self.cfg.cluster_id)

    def _fold_pending(
        self, mule_id: MuleID, up_round: Optional[int], *, stand_in: bool = False,
    ) -> None:
        """Run the fold after ``mule_id``'s partial joined the open round.

        ``up_round`` is the mission of that partial. ``stand_in`` marks the
        empty partial held for a lost upload (:meth:`_hold_lost_upload`); a
        round it closes names ``mule_id`` on ``cluster_round_closed``, since no
        ``up_bundle_ingested`` precedes it to say whose upload closed it.
        Raises what the fold or a DOWN send raises; the caller counts it.
        """
        merged = self.cluster.aggregate_pending()
        outcome = self.cluster.last_outcome
        if merged is None and outcome in (OUTCOME_DEFERRED, OUTCOME_EXPIRED):
            # θ is unchanged and the round stays open — agg:fedbuff is still
            # filling its buffer, or every pending partial was past the cutoff
            # (or empty) — but the mule is waiting at its inter-pass dock for a
            # DOWN (after an expiry, so is every mule whose partial was in the
            # fold). A quorum wait sends none: the merge that meets quorum
            # answers every waiting mule.
            self.events.emit(
                "cluster_merge_deferred" if outcome == OUTCOME_DEFERRED
                else "cluster_merge_expired",
                mule_id=str(mule_id),
                **self._merge_event_fields(up_round),
            )
            self._stop_waiting(mule_id)
            try:
                self.dock.send_down(self.cluster.dispatch_down_bundle(mule_id))
                self.metrics.increment("down_bundles_dispatched")
            finally:
                # A lost uploader must not strand the others. A FedBuff
                # deferral leaves nobody else waiting: each buffered mule got
                # its DOWN when deferred.
                if outcome == OUTCOME_EXPIRED:
                    self._release_waiting("post-expiry")
        if merged is not None:
            if self.cluster.last_merge is not None:
                # Age-aware rules only; agg:plain traces unchanged.
                self.events.emit(
                    "cluster_merge",
                    **self._merge_event_fields(up_round),
                )
            self.cluster.close_cluster_round()
            closer = {"mule_id": str(mule_id)} if stand_in else {}
            self.events.emit(
                "cluster_round_closed",
                cluster_round=self.cluster._cluster_round,
                **closer,
            )
            self.metrics.increment("cluster_rounds_closed")
            # EX-4.1 — convergence point for the just-aggregated θ'.
            self._emit_model_evaluation(self.cluster._cluster_round)
            # Answer every mule waiting at the dock with the new θ. A mule
            # still in flight gets nothing: it reads one DOWN per dock, so a
            # DOWN sent now would sit in its queue and be read as the answer
            # to its next upload, a θ behind by every merge since (FeRRy
            # Phase 2).
            self._release_waiting("post-aggregation")

    # ---------------------------------------- lost and refused uploads

    def _lost_upload_holds_quorum(self) -> bool:
        """True when a lost upload must still count toward the quorum.

        A quorum above 1 (FedBuff aside: its K is its own quorum) merges one
        partial from each of several mules at a time. Answering a lost
        uploader at once, with nothing in the round, lets it fly its next
        mission while the others wait, so the mules drift out of step; and
        once one mule has lost more uploads than another, the other's last
        partial waits for a partner that has already finished its run, until
        its ``down_wait_s`` (the whole trial budget under the Exp 4 driver)
        runs out (FeRRy Phase 2). With a quorum of 1, every recorded run
        included, a lost upload is answered at once, as it always was.
        """
        return (
            self.aggregation.rule != AGG_FEDBUFF
            and int(self.cluster.min_participation) > 1
        )

    def _hold_lost_upload(self, up, up_round: Optional[int]) -> None:
        """Count a lost upload toward the quorum as an empty partial.

        The mule's update never arrived, but the cluster knows the mule docked
        (it has always answered a lost upload): it holds an empty partial in
        the mule's place, which counts toward the quorum and adds nothing to θ,
        as a ``dock_on_empty`` mission's does, and the mule waits at the dock
        like any uploader until the fold answers it. Its round report and
        Pass-2 ledger were lost with it, as before. If the round already holds
        a partial from this mule (it stopped waiting and flew on), that one
        keeps its place and the mule simply waits again.
        """
        self.events.emit(
            "backhaul_upload_lost",
            mule_id=str(up.mule_id),
            mission_round=up_round,
            awaits_quorum=True,
        )
        self.metrics.increment("backhaul_uploads_lost")
        try:
            accepted = self.cluster.ingest_up_bundle(
                _lost_upload_stand_in(up, up_round, self.aggregation)
            )
            self._start_waiting(up.mule_id)
            if accepted:
                self._fold_pending(up.mule_id, up_round, stand_in=True)
        except Exception:
            log.exception("lost-upload stand-in / aggregate failed for %s", up.mule_id)
            self.metrics.increment("ingest_failures")

    def _note_refused_partial(self, up, up_round: Optional[int]) -> None:
        """Trace an upload whose partial the open round refused.

        The round already holds a partial from this mule, the one its quorum
        counts: the mule stopped waiting for the DOWN (``down_wait_s``) and
        flew another mission. The bundle's round report and Pass-2 ledger were
        folded all the same, so it is still logged as ingested, marked
        ``partial_refused`` with the mission whose partial the round kept, so
        a trace does not credit its updates as merged. Never happens with one
        mule: each of its uploads is answered before the next.
        """
        self.events.emit(
            "up_bundle_ingested",
            mule_id=str(up.mule_id),
            mission_round=up_round,
            partial_refused=True,
            held_mission_round=self.cluster.held_mission_round(up.mule_id),
        )
        self.metrics.increment("up_bundles_ingested")
        self.metrics.increment("up_partials_refused")

    def _merge_event_fields(self, mission_round: Optional[int]) -> dict:
        """The cluster's ``last_merge``, plus the UP's mission, as event fields.

        ``mission_round`` is the round of the UP bundle whose ingest ran the
        fold, so a merge line can be joined to the mission that triggered it.
        It goes last so the existing fields keep their order, and any
        ``mission_round`` or ``mule_id`` key in ``last_merge`` is dropped so
        the emit never receives a keyword twice.
        """
        fields = dict(self.cluster.last_merge or {})
        fields.pop("mission_round", None)
        fields.pop("mule_id", None)
        fields["mission_round"] = mission_round
        return fields

    # ---------------------------------------------------- waiting mules

    def _start_waiting(self, mule_id: MuleID) -> None:
        """``mule_id`` uploaded and now waits at the dock for a DOWN."""
        if mule_id not in self._awaiting:
            self._awaiting.append(mule_id)

    def _stop_waiting(self, mule_id: MuleID) -> None:
        """``mule_id`` is being answered, or has left the dock."""
        if mule_id in self._awaiting:
            self._awaiting.remove(mule_id)

    def _release_waiting(self, context: str) -> None:
        """Send the current θ to every mule waiting at the dock, then forget them.

        After a merge these are the mules whose partials it used (a quorum
        answers all of them at once), and after an expiry the mules whose
        partials it dropped: they are still blocked at their inter-pass dock
        and cannot upload again until answered, so without a DOWN here they
        would time out. Sends are best-effort: a mule whose send fails is
        logged, counted and dropped from the list, and a mule no longer
        docked is skipped, since there is no socket to answer on.
        """
        waiting, self._awaiting = list(self._awaiting), []
        docked = set(self.dock.registered_mules())
        for mid in waiting:
            if mid not in docked:
                continue
            try:
                self.dock.send_down(self.cluster.dispatch_down_bundle(mid))
                self.metrics.increment("down_bundles_dispatched")
            except Exception:
                log.exception("%s DOWN failed for %s", context, mid)
                self.metrics.increment("dispatch_down_failures")

    def _dispatch_to_new_mules(self, bootstrapped: set) -> None:
        """L-H2: dispatch a DOWN bundle to any mule we haven't yet.

        ``bootstrapped`` is mutated in place so the caller's tracking
        set stays accurate across iterations. A mule that has left the dock
        is dropped from it, so the same id registering again (the mule
        restarted) is bootstrapped again instead of waiting for a DOWN that
        never comes. It is also no longer waiting for an answer: the process
        that uploaded is gone, and its bootstrap is the DOWN it gets.
        """
        docked = self.dock.registered_mules()
        bootstrapped &= set(docked)
        self._awaiting = [m for m in self._awaiting if m in bootstrapped]
        for mid in docked:
            if mid in bootstrapped:
                continue
            try:
                self._stop_waiting(mid)
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

    def _backhaul_dropped(self, mission_round=None, mule_id=None) -> bool:
        """EX-4.2/4.3 — Bernoulli draw for a lost mule->BS backhaul upload.

        Uses the per-mission L1 loss schedule (probabilities, index =
        mission_round-1) when configured; otherwise the flat pct. With
        several expected mules each mule draws from its own stream
        (:meth:`_backhaul_rng_for`); with one, from the single stream.
        """
        rng = self._backhaul_rng_for(mule_id)
        sched = self._backhaul_loss_schedule
        if sched:
            idx = (int(mission_round) - 1) if mission_round else 0
            idx = min(max(idx, 0), len(sched) - 1)
            p = float(sched[idx])
            return p > 0.0 and float(rng.random()) < p
        if self._backhaul_loss_pct <= 0.0:
            return False
        return float(rng.random()) < (self._backhaul_loss_pct / 100.0)

    def _backhaul_rng_for(self, mule_id) -> np.random.Generator:
        """The stream ``mule_id``'s backhaul draws come from.

        One mule: the single stream seeded with ``backhaul_rng_seed``, so its
        draws are the recorded ones. Several: a stream per mule, seeded with
        (``backhaul_rng_seed``, a stable key of the mule id), so each mule's
        losses depend only on its own missions, not on the order in which the
        mules' uploads happen to arrive. An unseeded run stays unseeded.
        """
        if not self._backhaul_per_mule or mule_id is None:
            return self._backhaul_rng
        key = str(mule_id)
        rng = self._backhaul_mule_rngs.get(key)
        if rng is None:
            seed = getattr(self.cfg, "backhaul_rng_seed", None)
            rng = np.random.default_rng(
                None if seed is None else [int(seed), _mule_stream_key(key)]
            )
            self._backhaul_mule_rngs[key] = rng
        return rng

    def _emit_model_evaluation(self, cluster_round: int) -> None:
        """EX-4.1 — score the current global θ on the held-out test set.

        No-op when the real-model eval set was not configured (the stub
        integration path). Best-effort: a scoring failure logs and drops the
        sample rather than killing the cluster loop.
        """
        if self._eval_X is None:
            return
        try:
            from experiments.exp4.model_task import evaluate_theta

            theta = self.cluster.generator.get_global_disc_weights()
            m = evaluate_theta(
                theta, self._eval_X, self._eval_y,
                input_dim=self._eval_input_dim,
            )
            self.events.emit(
                "model_eval",
                cluster_round=int(cluster_round),
                accuracy=float(m["accuracy"]),
                auc=float(m["auc"]),
                loss=float(m["loss"]),
                n_test=int(len(self._eval_y)),
            )
            self.metrics.observe("model_auc", float(m["auc"]))
            log.info(
                "cluster %s: model_eval round=%d acc=%.4f auc=%.4f loss=%.4f",
                self.cfg.cluster_id, cluster_round,
                m["accuracy"], m["auc"], m["loss"],
            )
        except Exception:
            log.exception(
                "cluster %s: model scoring failed", self.cfg.cluster_id,
            )
            self.metrics.increment("model_eval_failures")

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
