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

FeRRy Phase 3 (``sim_clock=True``, design sections 2.2, 2.5 and 4.7). The
cluster keeps no clock of its own: it receives simulated time only as data.
It tracks :attr:`HFLHostCluster.sim_ts`, the latest ``UpBundle.sim_upload_ts``
it has ingested (an upload's simulated COMPLETION time, critic B8), and every
DOWN carries it as ``cluster_sim_ts`` so the mule can make its Lamport sync
at the dock. It refuses wall-clock deadline overrides (critic B3), and
forwards each device's latest contact SNR per band class, read off the Pass-1
report lines, as ``registry_deltas[did]["spectrum_sig"]``. Without it (the
default) nothing of this runs and every DOWN is the recorded one.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

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
from hermes.mission.aggregation_rules import (
    AGG_FEDBUFF,
    AGG_FEDEX,
    AggregationConfigError,
    AggregationSpec,
    FedBuffBuffer,
    check_partial_form,
    partial_age,
    partial_staleness,
)

from .cross_mule_fedavg import FedAvgError, apply_weighted_deltas, cross_mule_fedavg
from .device_registry import DeviceRegistry

log = logging.getLogger(__name__)

# What one ``aggregate_pending`` call did (``HFLHostCluster.last_outcome``).
# The service reads it to decide whether the uploading mule gets a DOWN now.
#: θ changed; the caller closes the round.
OUTCOME_MERGED = "merged"
#: Fewer partials than ``min_participation``; the round waits for more mules.
OUTCOME_QUORUM = "quorum"
#: ``agg:fedbuff`` buffered the partials and has fewer than K updates.
OUTCOME_DEFERRED = "deferred"
#: Every pending partial was past the cutoff (or empty): no step, round open.
OUTCOME_EXPIRED = "expired"


def _member_list(members: Iterable[Tuple[MuleID, int]]) -> List[list]:
    """(mule_id, mission_round) pairs as JSON-ready ``[str, int]`` lists."""
    return [[str(m), int(r)] for m, r in members]


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

    def apply_tier3_gen_refinement(
        self, weights: Weights, refinement_round: int = 0
    ) -> None:
        """Phase 7: apply a Tier-3-aggregated θ_gen back into the local generator.

        Called by ``ClusterService`` when ``HTTPCloudLink.poll_refinement``
        returns a fresh :class:`GeneratorRefinement`. The synth-batch
        generation in subsequent missions then draws from the updated
        cross-cluster θ_gen instead of the stale local one. Implementations
        should ignore older ``refinement_round`` numbers (out-of-order
        delivery from Tier-3) and otherwise replace their generator
        weights wholesale.
        """


@dataclass
class StubGeneratorHost:
    """Placeholder used in tests + the Phase-1 demo.

    Carries the discriminator weights as a list of numpy arrays and emits
    fixed-shape zero tensors as 'synth samples'. Real generator plugs in
    via the ``GeneratorHost`` protocol.
    """

    disc_weights: Weights
    synth_shape: Tuple[int, ...] = (8,)
    # Phase 7: held θ_gen so the tier-3 refinement fold has somewhere to land.
    # The stub doesn't actually use it for synth generation (we still emit
    # zeros); real GeneratorHost implementations replace this with the
    # AC-GAN generator weights.
    gen_weights: Weights = field(default_factory=list)
    last_refinement_round: int = -1

    def make_synth_batch(self, n: int) -> List[np.ndarray]:
        return [np.zeros(self.synth_shape, dtype=np.float32) for _ in range(n)]

    def get_global_disc_weights(self) -> Weights:
        # return copies so downstream mutation can't leak back in
        return [w.copy() for w in self.disc_weights]

    def update_disc_from_cluster_avg(self, weights: Weights) -> None:
        self.disc_weights = [w.copy() for w in weights]

    def apply_tier3_gen_refinement(
        self, weights: Weights, refinement_round: int = 0
    ) -> None:
        if refinement_round < self.last_refinement_round:
            # Out-of-order Tier-3 delivery — keep the newer state we
            # already have. Tier-3 is best-effort polled; old packets
            # can lag a fresh one.
            return
        self.gen_weights = [w.copy() for w in weights]
        self.last_refinement_round = refinement_round


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
    # Sprint 1.5 observability: how many devices were UNDELIVERED in the
    # previous mission's Pass 2 across all docked mules in this cluster
    # round. Bumped in ``ingest_up_bundle`` whenever a non-empty
    # ``prev_mission_delivery_report`` arrives.
    n_undelivered_carryover: int = 0


class HFLHostCluster:
    """Cluster-scope coordinator. One instance per edge server.

    ``min_participation`` controls how many mules must contribute an
    UpBundle before ``aggregate_pending`` returns merged weights. The
    default of 1 implements **partial-FedAvg** — the cluster aggregates
    as soon as any mule reports, accepting one-round staleness from
    absent mules. Set to ``len(mules)`` for full-FedAvg semantics
    (everyone in lockstep, no aggregation until the slowest mule
    reports). Sprint 2's chunk-L orchestrator surfaces this through
    ``ClusterConfig.min_participation``. Under ``agg:fedbuff`` it does not
    apply: FedBuff's buffer size K is its quorum.

    ``sim_clock`` (FeRRy Phase 3; off by default, as in every recorded run)
    puts the simulated-time bookkeeping on: see the module docstring.
    ``contact_band_classes`` names the contact link's band classes in index
    order, to read a report line's ``band``; None is the D1 set (wide,
    medium, narrow, and the optional 10 MHz class after them).
    """

    def __init__(
        self,
        *,
        registry: DeviceRegistry,
        generator: GeneratorHost,
        dock: DockLink,
        synth_batch_size: int = 32,
        min_participation: int = 1,
        aggregation: Optional[AggregationSpec] = None,
        sim_clock: bool = False,
        contact_band_classes: Optional[Sequence[str]] = None,
    ) -> None:
        self.registry = registry
        self.generator = generator
        self.dock = dock
        self.synth_batch_size = synth_batch_size
        # FeRRy Phase 3 — simulated time as data (module docstring). Off, no
        # attribute below is read, so the recorded paths are untouched.
        self._sim_clock = bool(sim_clock)
        self._sim_ts: Optional[float] = None
        if contact_band_classes is None and self._sim_clock:
            from hermes.l1.contact_link import CLASSES_WITH_10MHZ

            contact_band_classes = CLASSES_WITH_10MHZ
        self._band_names: Tuple[str, ...] = tuple(contact_band_classes or ())
        self.min_participation = min_participation
        # FeRRy Phase 1 — the L3 merge rule. The default, agg:plain, is the
        # merge every recorded run used and keeps its original code path.
        self.aggregation: AggregationSpec = aggregation or AggregationSpec()
        self._fedbuff: Optional[FedBuffBuffer] = None
        #: What the latest age-aware fold did, for the service's event log.
        #: None under agg:plain, and reset by every ``aggregate_pending`` call.
        self.last_merge: Optional[dict] = None
        #: One of the ``OUTCOME_*`` values for the latest ``aggregate_pending``
        #: call, or None when it had nothing pending (or raised).
        self.last_outcome: Optional[str] = None

        self._cluster_round: int = 0
        self._pending: Optional[_PendingRound] = None
        self._lock = threading.RLock()

        # last amendment we shipped — kept so a re-docking mule can see what
        # it last acknowledged.
        self._last_amendment: ClusterAmendment = ClusterAmendment(cluster_round=0)

    # ------------------------------------------------ simulated time (Phase 3)

    @property
    def sim_clock(self) -> bool:
        """True when the cluster keeps the simulated-time bookkeeping."""
        return self._sim_clock

    @property
    def sim_ts(self) -> Optional[float]:
        """The latest simulated upload time ingested (None before any, or off).

        An upload's ``sim_upload_ts`` is when it COMPLETED on the mule's
        mission clock. Every bundle that reaches the open round counts: an
        accepted partial, a refused one whose reports were still folded, and
        the empty stand-in held for a lost upload (it keeps the lost UP's
        time, critic B8). A duplicate resend, ignored whole, does not.
        """
        with self._lock:
            return self._sim_ts

    def _note_sim_upload(self, bundle: UpBundle) -> None:
        """Advance :attr:`sim_ts` to ``bundle``'s upload time (caller holds the lock).

        A stamp at or past the mission clock's ceiling is a wall-clock time,
        never simulated time: it is logged and ignored, so the cluster never
        echoes one (the mule would refuse the DOWN carrying it).
        """
        if not self._sim_clock:
            return
        ts = getattr(bundle, "sim_upload_ts", None)
        if ts is None:
            return
        from hermes.l1.mission_clock import SIM_CEILING_S

        if not ts < SIM_CEILING_S:
            log.error(
                "UpBundle from mule=%s carries sim_upload_ts=%r, a wall-clock stamp; "
                "not taken as simulated time", bundle.mule_id, ts,
            )
            return
        if self._sim_ts is None or ts > self._sim_ts:
            self._sim_ts = float(ts)

    def _note_contact_snr(self, bundle: UpBundle) -> None:
        """Fold the report lines' ``(band, snr_db)`` into each device's SpectrumSig.

        Design section 4.7, plumbing only: the latest reading per band class
        wins. A line without a band or an SNR (a contact without a band, a
        channel-free control) carries nothing; an unknown band index is
        skipped. Caller holds the lock.
        """
        if not self._sim_clock:
            return
        for line in bundle.round_close_report.lines:
            band = getattr(line, "band", None)
            snr = getattr(line, "snr_db", None)
            if band is None or snr is None:
                continue
            try:
                name = self._band_names[int(band)]
            except (IndexError, TypeError, ValueError):
                log.debug("line of %s carries unknown band %r; skipped", line.device_id, band)
                continue
            if not isinstance(snr, (int, float)) or not math.isfinite(float(snr)):
                continue
            rec = self.registry.get(line.device_id)
            if rec is None:
                continue
            rec.spectrum_sig = rec.spectrum_sig.with_contact_class_snr({name: float(snr)})

    def _refuse_overrides(self, overrides, where: str) -> None:
        """On the simulated clock, deadline overrides are refused (critic B3).

        They are absolute wall-clock stamps, and a sim-mode mule refuses the
        whole DOWN that carries them (a fatal ``MuleSupervisorError``).
        """
        if self._sim_clock and overrides:
            raise ValueError(
                f"{where}: deadline overrides for "
                f"{sorted(map(str, overrides))} refused on the simulated mission clock: "
                "they are wall-clock stamps"
            )

    # -------------------------------------------------------------- registry

    def known_mules(self) -> List[MuleID]:
        """Return mules **assigned** in the registry (logical assignment).

        This is the planning view — mules the cluster *expects* to dock
        because they own a slice. It is NOT the connectivity view: a
        mule that's assigned but currently offline still appears here.
        For the live-connection set used by Sprint 2's orchestrator
        re-dispatch logic, use ``TCPDockLinkServer.registered_mules()``
        instead.
        """
        return sorted(self.registry.snapshot().by_mule.keys())

    # ----------------------------------------------------------- dock ingest

    def ingest_up_bundle(self, bundle: UpBundle) -> bool:
        """Accept a mule's mission output. Folds the round-close report
        into per-device counters and parks the partial for cross-mule FedAvg.

        Sprint 1.5: also ingests the optional
        ``prev_mission_delivery_report`` (Pass-2 ledger from the
        *previous* mission). For each line, bumps
        ``DeviceRecord.delivery_priority`` on UNDELIVERED rows and
        resets it on DELIVERED — design §7 principle 13. The
        ``n_undelivered_carryover`` metric is bumped so observability
        can detect Pass-2 coverage degrading.

        Returns True when the partial was parked, False when it was refused
        because the open round already holds one from this mule. The first
        partial is the one kept: it is the one the quorum already counts. A
        refused bundle from a *later* mission (FeRRy Phase 2: a mule that
        stopped waiting for its quorum's DOWN flew another mission) still has
        its round report and Pass-2 ledger folded, since those sessions and
        deliveries happened; a resend of the same mission is ignored whole,
        as every refusal was before.
        """
        with self._lock:
            self._ensure_pending_round()
            assert self._pending is not None  # for type-checkers

            if bundle.mule_id in self._pending.seen_mules:
                held = self._pending.partials[
                    self._pending.seen_mules.index(bundle.mule_id)
                ]
                if held.mission_round == bundle.partial_aggregate.mission_round:
                    log.warning(
                        "duplicate UpBundle from mule=%s in cluster_round=%d "
                        "(ignoring later submission)",
                        bundle.mule_id,
                        self._pending.cluster_round,
                    )
                    return False
                self._note_sim_upload(bundle)
                n_undelivered = self._fold_bundle_reports(bundle)
                log.warning(
                    "UpBundle from mule=%s for mission %d refused in "
                    "cluster_round=%d: the round already holds its mission-%d "
                    "partial (round report and %d undelivered carryover folded)",
                    bundle.mule_id,
                    bundle.partial_aggregate.mission_round,
                    self._pending.cluster_round,
                    held.mission_round,
                    n_undelivered,
                )
                return False

            self._pending.partials.append(bundle.partial_aggregate)
            self._pending.seen_mules.append(bundle.mule_id)
            self._note_sim_upload(bundle)
            n_undelivered = self._fold_bundle_reports(bundle)

            log.info(
                "ingested UpBundle mule=%s round=%d devices=%d "
                "on_time=%d missed=%d undelivered_carryover=%d",
                bundle.mule_id,
                self._pending.cluster_round,
                len(bundle.round_close_report.lines),
                *bundle.round_close_report.counts(),
                n_undelivered,
            )
            return True

    def held_mission_round(self, mule_id: MuleID) -> Optional[int]:
        """Mission round of ``mule_id``'s partial in the open round, or None."""
        with self._lock:
            if self._pending is None or mule_id not in self._pending.seen_mules:
                return None
            held = self._pending.partials[self._pending.seen_mules.index(mule_id)]
            return int(held.mission_round)

    def _fold_bundle_reports(self, bundle: UpBundle) -> int:
        """Fold a bundle's round report and Pass-2 ledger into the registry.

        Caller holds the lock and has an open round. Returns the number of
        UNDELIVERED rows in the ledger, also added to the round's carryover.
        """
        assert self._pending is not None
        # apply per-device counter updates from the round-close report
        for line in bundle.round_close_report.lines:
            self.registry.update_after_round(
                device_id=line.device_id,
                on_time=line.outcome.is_on_time(),
            )
        # FeRRy Phase 3: the contact SNR per band class (sim clock only).
        self._note_contact_snr(bundle)

        # Sprint 1.5 — fold the previous mission's delivery report.
        n_undelivered = 0
        if bundle.prev_mission_delivery_report is not None:
            for line in bundle.prev_mission_delivery_report.lines:
                delivered = line.outcome.is_delivered()
                self.registry.update_after_delivery(
                    device_id=line.device_id,
                    delivered=delivered,
                )
                if not delivered:
                    n_undelivered += 1
            self._pending.n_undelivered_carryover += n_undelivered
        return n_undelivered

    # ----------------------------------------------- cross-mule aggregation

    def aggregate_pending(self) -> Optional[Weights]:
        """Run cross-mule FedAvg if enough partials have arrived.

        Returns the merged weights, or ``None`` when θ did not change; the
        merged weights are also pushed back into the held generator's global
        discriminator state. ``last_outcome`` says which case happened
        (``OUTCOME_*``) and ``last_merge`` describes an age-aware fold; both
        are reset on every call, so neither ever reports an earlier call.
        """
        with self._lock:
            self.last_merge = None
            self.last_outcome = None
            if self._pending is None:
                return None
            if self.aggregation.rule == AGG_FEDBUFF:
                # K is FedBuff's own quorum. Gating on min_participation too
                # would hold a partial outside the buffer and refuse the same
                # mule's next UP as a duplicate, so every partial goes in now.
                return self._aggregate_age_aware()
            if len(self._pending.partials) < self.min_participation:
                log.debug(
                    "aggregate_pending: %d partials < min=%d",
                    len(self._pending.partials),
                    self.min_participation,
                )
                self.last_outcome = OUTCOME_QUORUM
                return None
            if not self.aggregation.is_plain:
                return self._aggregate_age_aware()
            partials = self._pending.partials
            if all(p.is_empty() for p in partials):
                # Only a mule with dock_on_empty (FeRRy Phase 2) uploads an
                # empty partial: it counts toward the quorum and carries no
                # model. With nothing to average, the round stays open and the
                # waiting mules are released with the current θ, as when every
                # age-aware partial expires. A recorded run never gets here.
                version = self._cluster_round
                return self._expire_pending(
                    [partial_age(p, version) for p in partials],
                    [0.0] * len(partials),
                    _member_list((p.mule_id, p.mission_round) for p in partials),
                )
            try:
                merged = cross_mule_fedavg(self._pending.partials)
            except FedAvgError:
                log.exception("cross_mule_fedavg failed; dropping cluster round")
                self._reset_pending()
                raise
            self.generator.update_disc_from_cluster_avg(merged)
            self.last_outcome = OUTCOME_MERGED
            return merged

    @property
    def defers_merges(self) -> bool:
        """True when the rule buffers partials across UPs (``agg:fedbuff``).

        Informational only. Whether the uploading mule needs a DOWN before the
        round closes comes from ``last_outcome`` (``deferred`` or ``expired``):
        a cutoff rule can also leave θ unchanged, and a FedBuff call can merge.
        """
        return self.aggregation.rule == AGG_FEDBUFF

    def _aggregate_age_aware(self) -> Optional[Weights]:
        """Fold pending delta partials into θ under the configured rule.

        Caller holds the lock and has checked min_participation (FedBuff is
        not gated by it). Returns the new θ, or None with ``last_outcome``
        ``deferred`` while FedBuff fills its buffer, or ``expired`` when no
        pending partial is live (all past ``a_max``, or empty). An expired fold
        takes no step and leaves the round open: raising would drop the round
        as if the merge were malformed, while it is simply stale.
        """
        assert self._pending is not None
        spec = self.aggregation
        version = self._cluster_round
        partials = list(self._pending.partials)
        members = _member_list((p.mule_id, p.mission_round) for p in partials)
        try:
            for p in partials:
                check_partial_form(spec, p)
            theta = self.generator.get_global_disc_weights()
            ages = [partial_age(p, version) for p in partials]
            if spec.rule == AGG_FEDEX:
                # FedEx-Async: θ + η·Σ_m u_m / N, every returning partial at
                # full weight whatever its age, N the total client count.
                live = [p for p in partials if not p.is_empty()]
                if not live:
                    return self._expire_pending(ages, [0.0] * len(partials), members)
                n_clients = int(spec.fedex_n or max(1, len(self.registry.all())))
                merged = apply_weighted_deltas(
                    theta, partials, [0.0 if p.is_empty() else 1.0 for p in partials],
                    server_lr=spec.server_lr, normalizer=float(n_clients),
                )
                # As in the weighted rules below, ``partials`` names only what
                # reached θ; an empty partial (a mule that docked with nothing,
                # or the place held for a lost upload) is listed apart.
                self.last_merge = {
                    "rule": spec.rule, "applied": True,
                    "partial_ages": ages, "n_clients": n_clients,
                    "n_updates": sum(p.n_updates for p in live),
                    "partials": [m for p, m in zip(partials, members) if not p.is_empty()],
                    "expired_partials": [
                        m for p, m in zip(partials, members) if p.is_empty()
                    ],
                }
            elif spec.rule == AGG_FEDBUFF:
                if self._fedbuff is None:
                    opener = partials[0].mule_id if partials else None
                    self._fedbuff = FedBuffBuffer(spec=spec, k=self._fedbuff_k(opener))
                for p in partials:
                    self._fedbuff.add(p, cluster_version=version)
                # The partials now live in the buffer. Clear them from the open
                # round so the same mule's next UP is not refused as a duplicate.
                self._pending.partials = []
                self._pending.seen_mules = []
                if not self._fedbuff.ready:
                    self.last_outcome = OUTCOME_DEFERRED
                    self.last_merge = {
                        "rule": spec.rule, "applied": False,
                        "buffered": self._fedbuff.count, "k": self._fedbuff.k,
                        "partial_ages": ages,
                        "partials": _member_list(self._fedbuff.members),
                    }
                    return None
                buffered = self._fedbuff.count
                flushed = _member_list(self._fedbuff.members)
                merged = self._fedbuff.apply(theta)
                self.last_merge = {
                    "rule": spec.rule, "applied": True,
                    "buffered": buffered, "k": self._fedbuff.k,
                    "partial_ages": ages,
                    "partials": flushed,
                }
            else:
                stale = [partial_staleness(spec, p, version) for p in partials]
                weights = [p.weight_mass * s for p, s in zip(partials, stale)]
                live = [
                    p for p, s in zip(partials, stale)
                    if s > 0.0 and not p.is_empty()
                ]
                if not live:
                    return self._expire_pending(ages, weights, members)
                # Divide by the live partials' staleness-free mass, not by
                # Σ weights, so a stale partial's s_m < 1 shrinks the step. An
                # expired partial is left out of both sums.
                merged = apply_weighted_deltas(
                    theta, partials, weights, server_lr=spec.server_lr,
                    normalizer=sum(p.weight_mass for p in live),
                )
                # ``partials`` names only what reached θ; a partial cut to zero
                # weight in the same fold is listed apart, so a trace scorer
                # does not credit its updates as merged.
                live_ids = {id(p) for p in live}
                self.last_merge = {
                    "rule": spec.rule, "applied": True,
                    "partial_ages": ages, "partial_weights": weights,
                    "n_updates": sum(p.n_updates for p in live),
                    "partials": [m for p, m in zip(partials, members) if id(p) in live_ids],
                    "expired_partials": [
                        m for p, m in zip(partials, members) if id(p) not in live_ids
                    ],
                }
        except (FedAvgError, AggregationConfigError):
            log.exception("%s merge failed; dropping cluster round", spec.rule)
            self._reset_pending()
            raise
        self.generator.update_disc_from_cluster_avg(merged)
        self.last_outcome = OUTCOME_MERGED
        return merged

    def _expire_pending(
        self, ages: List[int], weights: List[float], members: List[list],
    ) -> None:
        """Drop pending partials that carry no weight, without a step.

        The round stays open (``cluster_round`` unchanged) so a fresh partial
        can still close it; the partials and ``seen_mules`` are cleared so the
        expired mules' next UPs are accepted rather than refused as duplicates.
        """
        assert self._pending is not None
        self._pending.partials = []
        self._pending.seen_mules = []
        self.last_outcome = OUTCOME_EXPIRED
        self.last_merge = {
            "rule": self.aggregation.rule, "applied": False,
            "outcome": OUTCOME_EXPIRED,
            "partial_ages": ages, "partial_weights": weights,
            "partials": members,
        }
        log.info(
            "%s: all %d pending partial(s) expired (ages %s); no step, "
            "cluster_round=%d stays open",
            self.aggregation.rule, len(members), ages, self._cluster_round,
        )
        return None

    def _fedbuff_k(self, mule_id: Optional[MuleID] = None) -> int:
        """FedBuff's K: ``buffer_k`` when set, else the opening mule's slice size.

        The slice size is what ``aggregation_rules`` and ``buffer_k`` document:
        about one mission's worth of updates from the mule that opens the
        buffer. With several mules the registered-device count is several
        missions' worth, so every flush would wait on several missions. Falls
        back to that count when the slice is empty or the mule unknown. K is
        fixed when the buffer is created and kept across flushes.
        """
        if self.aggregation.buffer_k is not None:
            return int(self.aggregation.buffer_k)
        if mule_id is not None:
            n_slice = len(self.registry.slice_for(mule_id))
            if n_slice > 0:
                return n_slice
        return max(1, len(self.registry.all()))

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

        Sprint 1.5 H7 — fold per-device ``last_known_position`` and
        ``delivery_priority`` from the cluster registry into the
        amendment's ``registry_deltas`` so the mule's scheduler sees
        real positions (S3a clustering depends on them) and the
        cluster-side ``delivery_priority`` carries forward (S3a
        tie-breaker depends on this). Without this, the mule's
        ``DeviceSchedulerState.last_known_position`` stays at the
        dataclass default ``(0, 0, 0)`` and every device clusters into
        one contact at origin.

        FeRRy Phase 3, with ``sim_clock`` only: the bundle carries
        ``cluster_sim_ts`` (:attr:`sim_ts`; None before any upload), each
        slice member whose contact SNR is known gets
        ``registry_deltas[did]["spectrum_sig"]``, and an amendment carrying
        deadline overrides is refused (``ValueError``). The bundle signature
        covers none of these, so a legacy DOWN signs as it always did.
        """
        with self._lock:
            mission_slice = self.make_mission_slice(mule_id)
            base_amendment = amendment or self._last_amendment
            self._refuse_overrides(base_amendment.deadline_overrides, "dispatch_down_bundle")
            # Build a fresh amendment that carries the positions + priorities
            # for *this mule's* slice members, layered on top of any
            # amendment fields (deadline overrides, notes) the caller passed.
            registry_deltas = dict(base_amendment.registry_deltas)
            for did in mission_slice.device_ids:
                rec = self.registry.get(did)
                if rec is None:
                    continue
                patch = dict(registry_deltas.get(did, {}))
                patch["last_known_position"] = rec.last_known_position
                patch["delivery_priority"] = rec.delivery_priority
                if self._sim_clock and rec.spectrum_sig.contact_class_snr_db:
                    patch["spectrum_sig"] = rec.spectrum_sig
                registry_deltas[did] = patch
            enriched_amendment = ClusterAmendment(
                cluster_round=base_amendment.cluster_round,
                deadline_overrides=dict(base_amendment.deadline_overrides),
                registry_deltas=registry_deltas,
                notes=base_amendment.notes,
            )
            sim_fields = {"cluster_sim_ts": self._sim_ts} if self._sim_clock else {}
            bundle = DownBundle(
                mule_id=mule_id,
                mission_slice=mission_slice,
                theta_disc=self.generator.get_global_disc_weights(),
                synth_batch=self.generator.make_synth_batch(self.synth_batch_size),
                cluster_amendments=enriched_amendment,
                **sim_fields,
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
        On the simulated clock (``sim_clock``) deadline overrides are
        refused (``ValueError``, before anything changes): they are
        wall-clock stamps (critic B3).
        """
        with self._lock:
            self._refuse_overrides(deadline_overrides, "close_cluster_round")
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

    def pending_undelivered_carryover(self) -> int:
        """Sprint 1.5 observability — Pass-2 misses carried over this round.

        Returns the count of UNDELIVERED rows ingested via
        ``prev_mission_delivery_report`` for the *current* (not-yet-
        closed) cluster round. Resets when ``close_cluster_round`` is
        called. Used by tests + the supervisor's metrics emitter.
        """
        with self._lock:
            return 0 if self._pending is None else self._pending.n_undelivered_carryover

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
