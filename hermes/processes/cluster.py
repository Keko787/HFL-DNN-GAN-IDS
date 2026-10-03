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

FeRRy Phase 3 (``ClusterConfig.mission_clock == "sim"``). The cluster keeps
no clock: it reads simulated time off the UP bundles (``sim_upload_ts``, the
upload's completion) and echoes the latest one it has ingested on every DOWN
(``cluster_sim_ts``, the mules' Lamport sync), and its events gain the
simulated fields of design section 2.5: ``up_bundle_ingested`` and
``backhaul_upload_lost`` carry ``sim_upload_ts``, ``carrier``, ``snr_db`` and
``p_loss``; ``cluster_round_closed`` and ``model_eval`` carry ``sim_ts``.
Under ``backhaul_model == "seconds"`` an upload is lost with the probability
the mule priced it at (``UpBundle.backhaul.p_loss``, the seconds-axis SNR at
the upload; 1.0 below the SNR floor), drawn KEYED by (trial seed, mule,
mission round) rather than from a stream, so the same mission of two arms is
decided by the same uniform. The draw applies to every UP, an empty partial's
included. On the wall clock every event and draw is the recorded one.

Several mules on the simulated clock below a full quorum (FeRRy Phase 3, unit
U9; :func:`needs_sim_order`). A mule spends about 10 wall seconds per
mission against 120-280 simulated ones, so the order in which UPs arrive over
the dock is not the order in which the uploads happened, and an asynchronous
merge (a quorum below the mule count, ``agg:fedbuff``) folded in arrival
order would merge them out of simulated order and drag each answered mule's
clock to another mule's later upload (critic B9, design risk R2). The cluster
then folds in simulated-time order instead: :class:`SimOrderGate` holds each
UP until no other mule can still send one that completed earlier, and every
released UP goes through exactly the steps an arriving one always did
(:meth:`ClusterService._process_up`). The dock queues what it knows of each
mule's time with the UPs (``TCPDockLinkServer(sim_markers=True)``:
registrations, departures, and any ``clock``/``done`` marker a mule sends).
``cluster_ready`` names the rule (``sim_order``), each released UP's event
carries its place in that order and the wall time it was held, and
``mule_departed`` marks a mule the cluster stops waiting for. A quorum of
every mule is served as before: each merge waits for one partial from every
mule, and the Lamport sync at the dock already puts every mule at the merge's
time.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import logging
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Deque, Dict, List, Optional, Tuple

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
from hermes.transport.dock_link import (
    MARKER_CLOCK,
    MARKER_REGISTERED,
    DockClockMarker,
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

from .config import (
    BACKHAUL_MISSION,
    BACKHAUL_SECONDS,
    CLOCK_SIM,
    CLOCK_WALL,
    ClusterConfig,
    cluster_config_errors,
    cluster_config_from_json,
    mission_schedule_index,
)

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

    FeRRy Phase 3 (critic B8): it keeps the lost UP's ``sim_upload_ts`` and
    ``backhaul``. The upload did happen at that simulated time, and the mule
    waits at the dock for the quorum like any uploader, so the time the
    cluster echoes must not fall behind it. Both are None on the wall clock.
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
        sim_upload_ts=getattr(up, "sim_upload_ts", None),
        backhaul=getattr(up, "backhaul", None),
    )


def stub_disc_weights() -> List[np.ndarray]:
    """The 13-parameter stub global model the cluster seeds without a real θ.

    Exposed so the Exp 4 driver can measure the stub payload (52 B) it prices
    T_nom and the D4 split with (FeRRy Phase 3) from the same arrays.
    """
    return [
        np.zeros((4,), dtype=np.float32),
        np.ones((3, 3), dtype=np.float32) * 0.01,
    ]


def _mule_stream_key(mule_id) -> int:
    """A stable 32-bit key for ``mule_id``, to seed its own backhaul stream.

    ``hash()`` is salted per process, so it cannot seed anything that must
    reproduce across runs.
    """
    digest = hashlib.sha256(str(mule_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# --------------------------------------------------------------------------- #
# FeRRy Phase 3, unit U9 — folding several mules' uploads in simulated order
# --------------------------------------------------------------------------- #

#: ``cluster_ready.sim_order``: the cluster folds uploads in simulated-time
#: order under the conservative rule of :class:`SimOrderGate`.
SIM_ORDER_CONSERVATIVE = "conservative"


def needs_sim_order(cfg: ClusterConfig) -> bool:
    """True when the cluster must fold uploads in simulated-time order (unit U9).

    On the simulated clock with several mules whenever a merge can take
    fewer partials than there are mules: a quorum below the mule count, or
    ``agg:fedbuff``, whose buffer, not the quorum, times each merge. Arrival
    order over the dock is then not simulated order (module docstring). A
    quorum of every mule needs no gate and keeps its recorded service: each
    merge waits for one partial from every mule, and the Lamport sync at the
    dock puts every mule at the merge's time. One mule, or the wall clock,
    never needs one.
    """
    if getattr(cfg, "mission_clock", CLOCK_WALL) != CLOCK_SIM:
        return False
    k = len(cfg.expected_mules)
    if k <= 1:
        return False
    from hermes.mission.aggregation_rules import AggregationSpec

    spec = AggregationSpec.from_config(
        getattr(cfg, "aggregation", None), getattr(cfg, "aggregation_params", None),
    )
    return spec.rule == AGG_FEDBUFF or int(cfg.min_participation) < k


@dataclass
class _HeldUp:
    """One UP waiting in :class:`SimOrderGate`."""

    up: UpBundle
    mule: str
    #: The upload's simulated completion (``sim_upload_ts``); None when it
    #: carries no simulated time (or a wall stamp), which cannot be ordered.
    ts: Optional[float]
    #: Arrival number: the last tie-breaker, and a mule's FIFO order.
    arrival: int
    #: ``time.monotonic()`` when it arrived, for the wall time it was held.
    held_at: float


@dataclass(frozen=True)
class ReleasedUp:
    """An upload :class:`SimOrderGate` hands to the fold."""

    up: UpBundle
    #: Its place in the cluster's simulated order, from 1.
    seq: int
    #: Wall seconds it was held (a measurement, like ``duration_s``).
    held_s: float
    #: It was released after an upload that completed strictly later: the
    #: order could not be kept. A mule that broke the waiting rule can send
    #: one (see :class:`SimOrderGate`), and so can a mule that joins (its
    #: bootstrap, or a restart) behind an upload lost at a quorum of 1 or
    #: under FedBuff, whose time the cluster does not echo, or a restarted
    #: mule behind an UP of its crashed session, read after the restart or
    #: still held at it (:class:`SimOrderGate`, sessions). An upload that
    #: ties the latest one released is not late: equal times are
    #: simultaneous.
    late: bool = False
    #: It carries no usable simulated time, so it was not ordered at all.
    unordered: bool = False


class SimOrderGate:
    """Hold each mule's UPs until no other mule can still send an earlier one (FeRRy Phase 3, U9).

    **The rule.** An upload's simulated time is its ``sim_upload_ts`` (its
    completion, critic B8); uploads are folded in the order of ``(sim time,
    mule id)``, ties broken by mule id so the order is deterministic. A mule
    synced to an upload already folded (the DOWN that answered or
    bootstrapped it carried that time) can tie it and sort before it; its
    upload happened after that fold, and follows it (see exempt mules). For
    every mule it tracks, the gate keeps a lower bound on the simulated time
    of every UP that mule may still send: the latest time the mule reported,
    by an UP (a mule's clock never goes back, so its next upload completes no
    earlier) or by a ``clock`` marker, and minus infinity before its first
    report. A held UP ``(t, m)`` is released when, for every other tracked
    mule ``m'`` that is not exempt, ``(bound(m'), m') > (t, m)``: no mule can
    still send an upload that sorts before it. This is the conservative
    synchronisation of parallel discrete-event simulation (Chandy and Misra
    1979), with UPs as the time-stamped messages and the bounds as the
    channel clocks. A mule stops being tracked when it departs (its dock
    connection ended, or it said ``done``), so nobody waits for a mule that has
    exited; each mule's own UPs are released in their arrival order.

    **Sessions.** The dock numbers each connection (its session,
    ``TCPDockLinkServer.session_of``) in the order the connections register,
    stamps every marker of the connection with it, and queues its
    ``registered`` marker before any of its UPs. The gate tracks a mule
    under the latest session it knows of, from that marker or from the
    mule's bootstrap DOWN, which can come first (:meth:`register`). A newer
    session (the mule registered again) starts the mule over at minus
    infinity, since a restarted mule's clock starts at the epoch. A marker
    of a session older than the tracked one is stale and changes nothing, a
    departure included: a mule restarted before the cluster read its
    crashed session's markers (during the startup wait, say) is tracked
    under its live session from its bootstrap on, and stays tracked. Two
    gaps remain, both because an UP carries no session. An UP the crashed
    session sent, read only after the live session is tracked, raises the
    live mule's bound to its time: the gate may then fold other mules'
    later uploads before the live mule's next one, which is folded ``late``
    and answered with a later time than its own. And an UP the crashed
    session left held stays at the head of the mule's queue (a mule's own
    UPs leave in arrival order), where the mule's own bound never holds it
    back. It may be folded before the restarted mule's first uploads, which
    are then ``late``. Otherwise those uploads, which complete earlier,
    queue behind it and set the mule's bound below it. Another mule's
    upload that falls between the two times then waits for the restarted
    mule, while the crashed UP waits for that other mule: nothing is
    released until a waiting mule's ``down_wait_s`` (the whole trial's wall
    budget under the Exp 4 driver) runs out. That stall is the one
    exception to why the gate cannot deadlock (below). The Exp 4
    orchestrator spawns each mule once and never restarts one, so only a
    manual or fault-injected restart reaches either gap.

    **Exempt mules.** A mule whose UP the cluster has folded and not yet
    answered (the service's ``_awaiting``) is blocked at its dock until a DOWN
    comes, and syncs its clock to that DOWN's ``cluster_sim_ts``, the latest
    upload folded before it (MissionClock ``advance_to``). So no upload it
    sends later can complete before anything folded while it waits, and it
    blocks nothing. It can tie the latest one, when its next mission takes no
    simulated time, and then sort before it by mule id; equal times are
    simultaneous, so such an upload is ordered as usual and never flagged
    (``late`` compares simulated time alone). Without the exemption a quorum
    above 1 below every mule would deadlock: the waiting mules' bounds would
    hold back the very uploads that complete their quorum. A mule that stops
    waiting (its ``down_wait_s``, the whole trial's wall budget under the
    Exp 4 driver, ran out) breaks this; its next upload may then complete
    before one already folded, is ``late``, and is folded at once and
    flagged, since holding it cannot restore the order.

    **Why it cannot deadlock** (any K, any quorum; a mule restarted with an
    UP of its crashed session still held is the one exception, see
    sessions). While an UP is held, each tracked mule is in one of three
    states: *held* (its own UP is held; it waits at its dock), *exempt* (see
    above) or *flying* (it has its DOWN and is flying, or has not uploaded
    yet). Take the held UP with the smallest ``(t, m)``. A held mule ``m'``
    has ``bound(m') >= t'`` for its own held ``(t', m') > (t, m)`` (a mule's
    UPs arrive in the order of their times, except across a restart), and
    exempt mules are skipped, so the smallest UP can wait only for flying
    mules. A flying mule does not depend on the cluster: its Pass 2 and next
    Pass 1 need no DOWN, so within one mission of wall time it either
    uploads (it becomes held, and its bound rises) or finishes its run and
    departs. Every report therefore moves a flying mule
    to held or gone, a mule becomes flying again only after one of its UPs
    was released and answered, and when no mule is flying the smallest held
    UP is released at once. So every held UP is released within a finite
    number of reports, whether one mule is fast in simulated time, another
    slow, or one finishes early. The price is wall time: at worst the mules
    run one at a time, which the driver's hard kill allows for
    (``Exp4Driver.ferry_wall_bound_s``). A mule that hangs without leaving
    the dock stalls the others until that kill, as a hung process would stall
    any conservative simulation.

    Pure bookkeeping: no clock of its own, no I/O, and not thread-safe (the
    service's single loop drives it).
    """

    def __init__(self) -> None:
        from hermes.l1.mission_clock import SIM_CEILING_S

        self._ceiling = float(SIM_CEILING_S)
        self._held: Dict[str, Deque[_HeldUp]] = {}
        #: Tracked mules: the lower bound of their future uploads' times.
        self._bound: Dict[str, float] = {}
        #: The dock session each tracked mule registered with (None: unknown).
        self._session: Dict[str, Optional[int]] = {}
        self._arrivals = 0
        self._released = 0
        #: ``(sim time, mule)`` of the latest ordered upload released.
        self._frontier: Optional[Tuple[float, str]] = None

    # ------------------------------------------------ what the dock reports

    def register(self, mule: str, session: Optional[int] = None) -> None:
        """Track ``mule``: a new connection of it, or its bootstrap DOWN.

        A mule's clock starts at the epoch and syncs to its bootstrap DOWN, so
        nothing is known yet of its time: its bound is minus infinity. The
        service calls this for the ``registered`` marker the dock queues, and
        again when it bootstraps the mule, which can come first (the marker
        may sit behind other mules' UPs in the queue while the mule, already
        bootstrapped, can upload any time after the epoch). Already tracked
        under the same session, or with no session given, nothing changes:
        the marker precedes all of the session's UPs, so no report is lost. A
        newer session (the mule registered again) starts over; UPs of the
        older one still held stay at the head of the mule's queue (class
        docstring, sessions). An older one
        changes nothing: the dock numbers sessions in the order they
        register, so it is a leftover of a connection the mule has replaced,
        read only after the service bootstrapped the restarted mule under its
        live session (class docstring, sessions); :meth:`depart` then ignores
        that session's departure too.
        """
        mule = str(mule)
        tracked = self._session.get(mule)
        if mule in self._bound and (
            session is None or session == tracked
            or (tracked is not None and session < tracked)
        ):
            return
        self._bound[mule] = float("-inf")
        self._session[mule] = session

    def depart(self, mule: str, session: Optional[int] = None) -> bool:
        """Stop tracking ``mule``: it will send no more UPs. True if it was tracked.

        A departure of an older session (the mule has registered again since)
        is ignored. UPs of the mule still held stay held and are folded in
        order: they happened.
        """
        mule = str(mule)
        if mule not in self._bound:
            return False
        tracked = self._session.get(mule)
        if session is not None and tracked is not None and session != tracked:
            return False
        del self._bound[mule]
        self._session.pop(mule, None)
        return True

    def clock(self, mule: str, sim_ts: Optional[float], session: Optional[int] = None) -> None:
        """``mule`` reports that its later UPs complete at or after ``sim_ts``.

        A report of an older session (the mule has registered again since) is
        ignored, as its departure is: a crashed session's clock says nothing
        of the restarted mule's (class docstring, sessions).
        """
        mule = str(mule)
        tracked = self._session.get(mule)
        if session is not None and tracked is not None and session != tracked:
            return
        ts = self._simulated(sim_ts)
        if mule in self._bound and ts is not None and ts > self._bound[mule]:
            self._bound[mule] = ts

    def hold(self, up: UpBundle, *, now: float) -> None:
        """Hold an arriving UP; it also reports its mule's time.

        An UP from a mule the gate does not track (it never registered with
        markers, or said ``done`` and uploaded anyway) tracks it again, bound
        at the UP's time, until it departs.
        """
        mule = str(up.mule_id)
        ts = self._simulated(getattr(up, "sim_upload_ts", None))
        self._arrivals += 1
        self._held.setdefault(mule, collections.deque()).append(
            _HeldUp(up=up, mule=mule, ts=ts, arrival=self._arrivals, held_at=float(now))
        )
        if ts is None:
            return
        if mule not in self._bound:
            self._bound[mule] = ts
            self._session[mule] = None
        elif ts > self._bound[mule]:
            self._bound[mule] = ts

    # ------------------------------------------------ release

    def blockers(self, mule: str, sim_ts: float, exempt: Collection[str] = ()) -> List[str]:
        """The mules that could still send an upload sorting before ``(sim_ts, mule)``."""
        mule = str(mule)
        return sorted(
            m for m, b in self._bound.items()
            if m != mule and m not in exempt and not (b > sim_ts or (b == sim_ts and m > mule))
        )

    def pop_ready(self, *, exempt: Collection[str] = (), now: float) -> Optional[ReleasedUp]:
        """The next upload in simulated order, if no mule can still precede it; else None.

        ``exempt``: the mules waiting at their dock for the answer to an UP
        already folded (class docstring). An UP without simulated time cannot
        be ordered and goes at once; so does one that completed strictly
        before an upload already released (``late``). One that ties the
        latest upload released, even sorting before it by mule id, is
        ordered as usual (class docstring, exempt mules).
        """
        exempt = {str(m) for m in exempt}
        heads = [(m, q[0]) for m, q in sorted(self._held.items()) if q]
        if not heads:
            return None
        for mule, head in heads:
            if head.ts is None:
                return self._release(mule, now, unordered=True)
        mule, head = min(heads, key=lambda mh: (mh[1].ts, mh[0], mh[1].arrival))
        late = self._frontier is not None and head.ts < self._frontier[0]
        if not late and self.blockers(mule, head.ts, exempt):
            return None
        return self._release(mule, now, late=late)

    def _release(self, mule: str, now: float, *, late: bool = False,
                 unordered: bool = False) -> ReleasedUp:
        entry = self._held[mule].popleft()
        self._released += 1
        if entry.ts is not None:
            key = (entry.ts, mule)
            if self._frontier is None or key > self._frontier:
                self._frontier = key
        return ReleasedUp(
            up=entry.up, seq=self._released, held_s=max(0.0, float(now) - entry.held_at),
            late=late, unordered=unordered,
        )

    # ------------------------------------------------ introspection

    def tracked(self) -> List[str]:
        """The mules the gate waits for (not yet departed), sorted."""
        return sorted(self._bound)

    def bound(self, mule: str) -> Optional[float]:
        """``mule``'s lower bound (-inf before any report), or None when untracked."""
        return self._bound.get(str(mule))

    def held_count(self, mule: Optional[str] = None) -> int:
        """How many UPs are held, of ``mule`` or of every mule."""
        if mule is not None:
            return len(self._held.get(str(mule), ()))
        return sum(len(q) for q in self._held.values())

    @property
    def frontier(self) -> Optional[float]:
        """Simulated time of the latest ordered upload released (None before any)."""
        return None if self._frontier is None else self._frontier[0]

    def _simulated(self, ts) -> Optional[float]:
        """``ts`` as a simulated time, or None (absent, not finite, or a wall stamp)."""
        if ts is None or isinstance(ts, bool):
            return None
        try:
            value = float(ts)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or not value < self._ceiling:
            return None
        return value


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
        # FeRRy Phase 3 — refuse clock settings that cannot run (critic B16)
        # before binding anything; the recorded defaults pass.
        errors = cluster_config_errors(cfg)
        if errors:
            raise ValueError("cluster config: " + "; ".join(errors))
        self._sim = getattr(cfg, "mission_clock", CLOCK_WALL) == CLOCK_SIM
        self._backhaul_model = getattr(cfg, "backhaul_model", BACKHAUL_MISSION)
        #: Salt of the seconds model's keyed loss draw (None otherwise).
        self._loss_salt: Optional[int] = None
        if self._backhaul_model == BACKHAUL_SECONDS:
            from hermes.l1.channel_model import SALT_BACKHAUL_LOSS, ferry_salt

            self._loss_salt = ferry_salt(int(cfg.trial_seed), SALT_BACKHAUL_LOSS)
        # FeRRy Phase 3, unit U9: several mules on the simulated clock below a
        # full quorum fold their uploads in simulated order (module
        # docstring); the dock then queues the mules' time markers with the
        # UPs. Otherwise None, and the recorded dock and loop run.
        self._sim_order: Optional[SimOrderGate] = (
            SimOrderGate() if self._sim and needs_sim_order(cfg) else None
        )
        #: The upload being folded out of the gate (its event fields).
        self._releasing: Optional[ReleasedUp] = None

        # Chunk M observability — null defaults so tests can construct the
        # service without setting up a JSONL file. The CLI entry point
        # below builds a real emitter when ``--run-dir`` is supplied.
        self.events = events or NullEventEmitter(role="cluster", node_id=cfg.cluster_id)
        self.metrics = metrics or MetricsRegistry()

        dock_kwargs = {"sim_markers": True} if self._sim_order is not None else {}
        self.dock = TCPDockLinkServer(host=cfg.dock_host, port=cfg.dock_port, **dock_kwargs)
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
            disc_weights = stub_disc_weights()
        self.generator = StubGeneratorHost(disc_weights=disc_weights)

        # EX-4.1 — held-out eval set for per-round convergence (``model_eval``).
        # Loaded as plain numpy here (cheap); the TF evaluate happens in the
        # service loop so it never delays port binding at startup.
        self._eval_X = None
        self._eval_y = None
        #: Exp 5 addendum (Study 5.13): each test row's attack family, when the
        #: test set carries them; ``model_eval`` then adds ``detection``.
        self._eval_family = None
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
            from experiments.exp4.model_task import load_family, load_xy
            self._eval_X, self._eval_y = load_xy(cfg.eval_test_path)
            self._eval_family = load_family(cfg.eval_test_path)
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
        # FeRRy Phase 3: on the simulated clock the cluster tracks the latest
        # simulated upload, echoes it on DOWNs and forwards contact SNRs; the
        # recorded cluster is built with exactly its old arguments.
        sim_kwargs: Dict[str, Any] = {}
        if self._sim:
            sim_kwargs = dict(
                sim_clock=True,
                contact_band_classes=getattr(cfg, "contact_band_classes", None),
            )
        self.cluster = HFLHostCluster(
            registry=self.registry,
            generator=self.generator,
            dock=self.dock,
            synth_batch_size=cfg.synth_batch_size,
            min_participation=cfg.min_participation,
            aggregation=self.aggregation,
            **sim_kwargs,
        )

        # Optional Tier-3 outbound link.
        self.cloud: Optional[HTTPCloudLink] = None
        if self.cfg.tier3_url:
            self.cloud = HTTPCloudLink(base_url=self.cfg.tier3_url)

        # FeRRy Phase 3, additive and on the simulated clock only: which
        # clock and backhaul model the cluster runs, and (unit U9) the rule
        # it folds several mules' uploads by when it orders them.
        ready_sim = (
            dict(mission_clock=CLOCK_SIM, backhaul_model=self._backhaul_model)
            if self._sim else {}
        )
        if self._sim_order is not None:
            ready_sim["sim_order"] = SIM_ORDER_CONSERVATIVE
        self.events.emit(
            "cluster_ready",
            dock_host=self.cfg.dock_host,
            dock_port=self.actual_dock_port,
            expected_mules=list(self.cfg.expected_mules),
            seed_devices=len(self.cfg.seed_devices),
            synth_batch_size=self.cfg.synth_batch_size,
            min_participation=self.cfg.min_participation,
            tier3_wired=self.cloud is not None,
            **ready_sim,
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

        FeRRy Phase 3, unit U9: with several mules on the simulated clock
        below a full quorum, step 2 is :meth:`_run_in_sim_order`: the same
        steps, with each UP held until the simulated order allows it.
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

        if getattr(self, "_sim_order", None) is not None:
            self._run_in_sim_order(bootstrapped)
            log.info("cluster %s service loop exiting", self.cfg.cluster_id)
            return

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
            if up is not None:
                self._process_up(up, up_round)

            now = time.time()
            if now - last_tier3_poll >= self._TIER3_POLL_INTERVAL_S:
                self._poll_tier3_if_wired()
                last_tier3_poll = now

        log.info("cluster %s service loop exiting", self.cfg.cluster_id)

    def _process_up(self, up, up_round: Optional[int]) -> None:
        """Everything one UP bundle sets off: the loss draw, the ingest, the
        fold, the event and the DOWN(s). Step 2c of :meth:`run`, unchanged;
        ``up_round`` is the mission its partial closes. Under the simulated
        order (unit U9) it runs for each UP as the gate releases it."""
        if self._upload_lost(up, up_round):
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
                    **self._sim_up_fields(up, up_round),
                )
                self.metrics.increment("backhaul_uploads_lost")
                self._stop_waiting(up.mule_id)
                try:
                    self.dock.send_down(self.cluster.dispatch_down_bundle(up.mule_id))
                except Exception:
                    log.exception("post-loss DOWN failed for %s", up.mule_id)
            return  # consumed as lost; skip the ingest path below

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
                    **self._sim_up_fields(up, up_round),
                )
                self.metrics.increment("up_bundles_ingested")
                self._fold_pending(up.mule_id, up_round)
            else:
                self._note_refused_partial(up, up_round)
        except Exception:
            log.exception("ingest_up_bundle / aggregate failed")
            self.metrics.increment("ingest_failures")

    # ------------------------------ FeRRy Phase 3, unit U9: simulated order

    def _run_in_sim_order(self, bootstrapped: set) -> None:
        """Step 2 of :meth:`run` with the uploads folded in simulated order.

        Each pass reads the next dock event (an UP, or a mule's registration,
        time report or departure; ``TCPDockLinkServer.recv_dock_event``),
        bootstraps newly docked mules as always, feeds the event to the gate,
        and then folds every UP the gate releases, in simulated order, each
        through :meth:`_process_up` exactly as an arriving UP is on the
        recorded loop. Tier-3 is polled on the same cadence.

        The gate tracks every mule from its bootstrap DOWN on (those of the
        startup wait first): from then on the mule can upload, and its
        ``registered`` marker may still sit behind other mules' UPs. It
        tracks the mule under its live dock session, so the markers an
        earlier, crashed session left unread in the queue (the mule was
        restarted during the startup wait) change nothing
        (:class:`SimOrderGate`, sessions).
        """
        self._track_bootstrapped(bootstrapped)
        last_tier3_poll = 0.0
        while not self._stop_event.is_set():
            try:
                item = self.dock.recv_dock_event(timeout=1.0)
            except Exception:
                item = None

            # L-M1, as on the recorded loop.
            if self._stop_event.is_set():
                break

            # L-H2, as on the recorded loop.
            before = set(bootstrapped)
            self._dispatch_to_new_mules(bootstrapped)
            self._track_bootstrapped(bootstrapped - before)

            if item is not None:
                self._note_dock_event(item)
            self._release_in_sim_order()

            now = time.time()
            if now - last_tier3_poll >= self._TIER3_POLL_INTERVAL_S:
                self._poll_tier3_if_wired()
                last_tier3_poll = now

    def _track_bootstrapped(self, mules) -> None:
        """Have the gate track ``mules``, just bootstrapped, under their dock session."""
        session_of = getattr(self.dock, "session_of", None)
        for mid in sorted(mules, key=str):
            self._sim_order.register(
                str(mid), None if session_of is None else session_of(mid),
            )

    def _note_dock_event(self, item) -> None:
        """Feed one dock event to the gate: hold an UP, or apply a marker.

        A departure (the mule's connection ended) or a ``done`` from the mule
        stops the gate waiting for it, and is traced as ``mule_departed``
        with what it knew: the reason, the mule's final clock when it said
        one, and how many of its UPs are still held (they are folded in order
        all the same). A marker of an older dock session changes nothing and
        is not traced (:class:`SimOrderGate`, sessions).

        The ``mules_departed`` counter counts the same departures. Like every
        registry metric it reaches a trace only in the end-of-run
        ``metrics_snapshot`` (:meth:`shutdown`), which a cluster stopped with
        ``TerminateProcess`` (the Exp 4 orchestrator, on Windows) never
        writes; its per-event equivalent is the count of ``mule_departed``.
        """
        gate = self._sim_order
        if isinstance(item, DockClockMarker):
            mule = str(item.mule_id)
            if item.kind == MARKER_REGISTERED:
                gate.register(mule, item.session)
            elif item.kind == MARKER_CLOCK:
                gate.clock(mule, item.sim_ts, item.session)
            elif gate.depart(mule, item.session):
                self.events.emit(
                    "mule_departed",
                    mule_id=mule,
                    reason=item.kind,
                    sim_ts=item.sim_ts,
                    held=gate.held_count(mule),
                )
                self.metrics.increment("mules_departed")
            return
        if not isinstance(item, UpBundle):
            log.warning("cluster %s: ignored a dock event of type %s",
                        self.cfg.cluster_id, type(item).__name__)
            return
        gate.hold(item, now=time.monotonic())

    def _release_in_sim_order(self) -> None:
        """Fold every UP the gate releases now, in simulated order.

        The mules waiting at their dock for an answer (``_awaiting``) are
        exempt from the gate's wait (:class:`SimOrderGate`), and each fold can
        change who waits, so the gate is asked again after every one. An UP
        the gate could not order (``late``, or without simulated time) is
        logged and counted, and folded like any other.

        The counters ``sim_order_late_uploads`` and
        ``sim_order_unordered_uploads`` and the timer ``sim_order_held_s``
        reach a trace only in the end-of-run ``metrics_snapshot``
        (:meth:`shutdown`), which a cluster stopped with ``TerminateProcess``
        (the Exp 4 orchestrator, on Windows) never writes. Each released UP's
        fold event (``up_bundle_ingested`` or ``backhaul_upload_lost``,
        :meth:`_sim_up_fields`) carries the same facts: the late count is the
        fold events with ``sim_order_late`` true, the unordered count those
        with a ``sim_order_seq`` whose ``sim_upload_ts`` is null or at least
        ``SIM_CEILING_S`` (1e9), and the timer's samples are their
        ``held_wall_s``. The one exception is an UP whose ingest raised: it is
        counted and timed but has no fold event (``ingest_failures``).
        """
        gate = self._sim_order
        while not self._stop_event.is_set():
            released = gate.pop_ready(
                exempt={str(m) for m in self._awaiting}, now=time.monotonic(),
            )
            if released is None:
                return
            if released.late or released.unordered:
                log.warning(
                    "cluster %s: UP from %s (sim_upload_ts=%r) %s; folded at once",
                    self.cfg.cluster_id, released.up.mule_id,
                    getattr(released.up, "sim_upload_ts", None),
                    "arrived behind a later upload already folded" if released.late
                    else "carries no simulated time",
                )
                self.metrics.increment(
                    "sim_order_late_uploads" if released.late else "sim_order_unordered_uploads"
                )
            self.metrics.observe("sim_order_held_s", float(released.held_s))
            self._releasing = released
            try:
                self._process_up(released.up, _up_mission_round(released.up))
            finally:
                self._releasing = None

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
                **self._sim_ts_fields(),
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
            **self._sim_up_fields(up, up_round),
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
            **self._sim_up_fields(up, up_round),
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

    def _upload_lost(self, up, mission_round) -> bool:
        """Whether ``up``'s backhaul upload is lost, under the configured model.

        The recorded ``mission`` model is :meth:`_backhaul_dropped`, called
        exactly as it always was; ``seconds`` (FeRRy Phase 3) is the keyed
        draw of :meth:`_seconds_backhaul_dropped`, which touches no stream.
        """
        if self._backhaul_model == BACKHAUL_SECONDS:
            return self._seconds_backhaul_dropped(up, mission_round, up.mule_id)
        return self._backhaul_dropped(mission_round, up.mule_id)

    def _backhaul_dropped(self, mission_round=None, mule_id=None) -> bool:
        """EX-4.2/4.3 — Bernoulli draw for a lost mule->BS backhaul upload.

        Uses the per-mission L1 loss schedule (probabilities, index =
        mission_round-1, clamped: ``mission_schedule_index``) when configured;
        otherwise the flat pct. With several expected mules each mule draws
        from its own stream (:meth:`_backhaul_rng_for`); with one, from the
        single stream.
        """
        rng = self._backhaul_rng_for(mule_id)
        sched = self._backhaul_loss_schedule
        if sched:
            p = float(sched[mission_schedule_index(mission_round, len(sched))])
            return p > 0.0 and float(rng.random()) < p
        if self._backhaul_loss_pct <= 0.0:
            return False
        return float(rng.random()) < (self._backhaul_loss_pct / 100.0)

    # ------------------------------------ FeRRy Phase 3: the seconds model

    @staticmethod
    def _priced_loss(up) -> Optional[float]:
        """The loss probability the mule priced ``up``'s upload at, or None.

        ``UpBundle.backhaul.p_loss``: ``loss_from_snr`` of the seconds-axis SNR
        of the carrier the mule held (H3: its controller's pick) when the
        upload started, and 1.0 when that SNR was below the floor (critic
        B12). None when the UP carries no pricing.
        """
        bh = getattr(up, "backhaul", None)
        return None if bh is None else float(bh.p_loss)

    def _seconds_backhaul_dropped(self, up, mission_round, mule_id) -> bool:
        """The keyed loss draw of the seconds model (design section 1 D2).

        Lost when ``keyed_uniform(ferry_salt(seed, "backhaul_loss"), mule,
        mission_round) < p``, with ``p`` the UP's own :meth:`_priced_loss`. The
        uniform is a pure function of (trial seed, mule id, mission round):
        two arms' same mission face the same uniform (common random numbers,
        paired by mission, not by upload count), and no earlier upload or
        arrival order moves it. Applied to every UP, an empty partial's too.
        An UP the mule did not price (it is not on the seconds model) is never
        lost: it is counted as ``backhaul_unpriced_uploads`` and logged. That
        counter reaches a trace only in the end-of-run ``metrics_snapshot``
        (:meth:`shutdown`), which a cluster stopped with ``TerminateProcess``
        (the Exp 4 orchestrator, on Windows) never writes. Its per-event
        equivalent is the count of fold events with ``p_loss`` null (each an
        ``up_bundle_ingested``, as such an UP is never lost) in a trace whose
        ``cluster_ready.backhaul_model`` is ``"seconds"``, bar an UP whose
        ingest raised (``ingest_failures``).
        """
        from hermes.l1.channel_model import keyed_uniform

        p = self._priced_loss(up)
        if p is None:
            log.error(
                "cluster %s: UP from %s carries no backhaul pricing under the "
                "seconds model; not drawing a loss for it",
                self.cfg.cluster_id, getattr(up, "mule_id", mule_id),
            )
            self.metrics.increment("backhaul_unpriced_uploads")
            return False
        key = str(mule_id if mule_id is not None else up.mule_id)
        u = keyed_uniform(self._loss_salt, key, int(mission_round or 0))
        return u < p

    def _loss_probability(self, up, mission_round) -> Optional[float]:
        """The probability the loss draw of ``up`` used (no draw is made).

        The seconds model: the UP's :meth:`_priced_loss`. The recorded model:
        the schedule's entry for the mission, read through the same
        ``mission_schedule_index`` as the draw, or the flat percentage over 100.
        """
        if self._backhaul_model == BACKHAUL_SECONDS:
            return self._priced_loss(up)
        sched = self._backhaul_loss_schedule
        if sched:
            return float(sched[mission_schedule_index(mission_round, len(sched))])
        return self._backhaul_loss_pct / 100.0

    def _sim_up_fields(self, up, mission_round) -> Dict[str, Any]:
        """The simulated fields of an UP's event (design section 2.5); {} on the wall clock.

        ``sim_upload_ts`` is the upload's simulated completion; ``carrier`` and
        ``snr_db`` the mule's pricing (None under the recorded model, where
        the mule prices none); ``p_loss`` the probability the loss draw used.

        Under the simulated order (unit U9) an UP the gate released adds
        ``sim_order_seq`` (its place in the cluster's simulated order, from
        1), ``held_wall_s`` (the wall seconds it was held; a measurement, like
        ``duration_s``) and ``sim_order_late`` (it arrived behind an upload
        already folded that completed strictly later, so the order could not
        be kept; a tie is not late).
        """
        if not self._sim:
            return {}
        bh = getattr(up, "backhaul", None)
        fields = {
            "sim_upload_ts": getattr(up, "sim_upload_ts", None),
            "carrier": None if bh is None else int(bh.carrier),
            "snr_db": None if bh is None else float(bh.snr_db),
            "p_loss": self._loss_probability(up, mission_round),
        }
        released = getattr(self, "_releasing", None)
        if released is not None and released.up is up:
            fields.update(
                sim_order_seq=int(released.seq),
                held_wall_s=float(released.held_s),
                sim_order_late=bool(released.late),
            )
        return fields

    def _sim_ts_fields(self) -> Dict[str, Any]:
        """``{"sim_ts": ...}`` on the simulated clock: the latest simulated upload
        ingested, None before any (the epoch); {} on the wall clock."""
        if not self._sim:
            return {}
        return {"sim_ts": self.cluster.sim_ts}

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

        Exp 5 addendum (Study 5.13): when the test set carries each row's
        attack family, the event adds ``detection``, the confusion counts,
        TPR, FPR, precision, F1 and each family's recall
        (``model_task.detection_metrics``); without them it is the recorded
        event.
        """
        if self._eval_X is None:
            return
        try:
            from experiments.exp4.model_task import evaluate_theta

            theta = self.cluster.generator.get_global_disc_weights()
            extra = {} if self._eval_family is None else {"family": self._eval_family}
            m = evaluate_theta(
                theta, self._eval_X, self._eval_y,
                input_dim=self._eval_input_dim, **extra,
            )
            detection = {} if "detection" not in m else {"detection": m["detection"]}
            self.events.emit(
                "model_eval",
                cluster_round=int(cluster_round),
                accuracy=float(m["accuracy"]),
                auc=float(m["auc"]),
                loss=float(m["loss"]),
                n_test=int(len(self._eval_y)),
                **self._sim_ts_fields(),
                **detection,
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
