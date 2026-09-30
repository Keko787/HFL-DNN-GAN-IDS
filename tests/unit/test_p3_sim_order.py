"""FeRRy Phase 3, unit U9 — several mules on the simulated clock: uploads in simulated order.

* The gate (``SimOrderGate``): an UP is held until every other mule's reported
  simulated time has passed it (ties broken by mule id); the release order is
  the simulated order whatever the arrival order; a clock marker, a ``done``
  marker and a departure release; the markers of an older session (its
  registration, clock and departure) change nothing; mules waiting for the
  answer to a folded UP are exempt; an UP
  without simulated time, or one that completed before an upload already
  released, is folded at once and flagged; one that only ties the latest
  upload released (a waiting mule synced to it, whose next mission took no
  simulated time) is ordered as usual and not late.
* No deadlock: a model of K mules and the cluster's fold (fast, slow and
  early-finishing mules; quorum 1, FedBuff, and a quorum between 1 and K)
  over many random wall schedules releases every UP, in simulated order, and
  at quorum 1 answers each mule with its own upload time. Without the
  exemption a quorum between 1 and K would deadlock. With missions that take
  no simulated time, mules that keep the waiting rule tie the upload they
  were synced to, and none is ever flagged late.
* The service: the gate is on exactly for several mules on the simulated
  clock below a full quorum or under FedBuff, never for one mule (the dock
  then queues markers and ``cluster_ready`` names the rule); uploads are
  folded in simulated order, and each DOWN at quorum 1 carries the
  uploader's own time (no drag, critic B9); the markers release; a mule
  bootstrapped before its registration marker is read is waited for, and a
  restarted mule under its new dock session from its new bootstrap (its old
  session's markers, read later, change nothing, also when the restart came
  during the startup wait); an UP its crashed session left held makes the
  restarted mule's first uploads late, or holds every upload until a DOWN
  wait runs out (a documented gap, pinned); lost uploads keep their order; an
  upload tying the one its mule was synced to is not flagged; UPs without
  simulated time are folded at once and counted; the gate's counters and
  the seconds model's, which reach a trace only in the end-of-run
  ``metrics_snapshot``, equal their per-event equivalents (with a wall
  clock that moves, so the held-time samples are not all zero); a quorum
  of every mule, one mule and the wall clock keep the recorded loop.
* The dock (real TCP): the marker server queues registration, UPs, markers
  and departure in order; a replaced connection is no departure; the
  recorded server makes no marker and ignores a mule's; ``recv_up`` skips
  markers; a mule cannot send server markers or report for another mule.
* The driver and the topology: several mules below a full quorum, or under
  FedBuff, run on the simulated clock (critic B9's refusal lifted), and the
  hard kill allows the mules to run one at a time; one mule, FedBuff or not,
  is not ordered and needs no DOWN wait of its own.
"""

from __future__ import annotations

import logging
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest

from experiments.exp4.driver import Exp4Driver
from experiments.runner import Cell
from hermes.l1.mission_clock import SIM_CEILING_S
from hermes.mission.aggregation_rules import AggregationSpec
from hermes.processes.cluster import (
    SIM_ORDER_CONSERVATIVE,
    ClusterService,
    SimOrderGate,
    needs_sim_order,
    stub_disc_weights,
)
from hermes.processes.config import ClusterConfig, DeviceConfig, MuleConfig, TopologyConfig
from hermes.transport import LoopbackDockLink, TCPDockLinkClient, TCPDockLinkServer
from hermes.transport.dock_link import (
    MARKER_CLOCK,
    MARKER_DEPARTED,
    MARKER_DONE,
    MARKER_REGISTERED,
    DockClockMarker,
    DockLinkError,
    DockLinkTimeout,
)
from hermes.transport.wire import send_message
from hermes.types import (
    ContactHistory,
    DeviceID,
    MissionRoundCloseReport,
    MuleID,
    PartialAggregate,
    UpBundle,
)
from hermes.types.bundles import BackhaulUpload

from tests.golden import _build_topology as T

T0 = 1.0e6
TURNAROUND_S = 30.0
CUTOFF = AggregationSpec(rule="agg:cutoff")


def _up(mule: str, mission_round: int, ts: Optional[float], *, spec=CUTOFF,
        base_version: int = 0, p_loss: Optional[float] = None) -> UpBundle:
    """An UP; ``p_loss`` prices its upload as the seconds model does (else unpriced)."""
    m = MuleID(mule)
    theta = stub_disc_weights()
    partial = PartialAggregate(
        mule_id=m, mission_round=mission_round,
        weights=[w + 0.1 for w in theta] if spec.is_plain
        else [np.full(w.shape, 0.1, dtype=np.float32) for w in theta],
        num_examples=5, contributing_devices=(DeviceID(f"{mule}-d0"),),
        rule=spec.rule, update_form=spec.update_form, base_version=base_version,
        weight_mass=5.0, n_updates=1,
    )
    return UpBundle(
        mule_id=m, partial_aggregate=partial,
        round_close_report=MissionRoundCloseReport(
            mule_id=m, mission_round=mission_round, started_at=T0, finished_at=T0,
        ),
        contact_history=ContactHistory(mule_id=m, mission_round=mission_round),
        sim_upload_ts=ts,
        backhaul=None if p_loss is None else BackhaulUpload(
            carrier=2, snr_db=9.5, p_loss=float(p_loss),
            t_start_s=(ts or T0) - 0.25, upload_s=0.25, nbytes=52),
    )


def _released(gate, exempt=(), now=0.0):
    """Every UP the gate releases now, as (mule, sim time)."""
    out = []
    while True:
        r = gate.pop_ready(exempt=exempt, now=now)
        if r is None:
            return out
        out.append((str(r.up.mule_id), r.up.sim_upload_ts))


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

def test_an_up_is_held_until_every_other_mules_reported_time_passes_it():
    g = SimOrderGate()
    for m in ("m1", "m2", "m3"):
        g.register(m, session=1)
    g.hold(_up("m1", 1, T0 + 100.0), now=0.0)
    assert _released(g) == []                          # m2, m3: nothing reported yet
    assert g.blockers("m1", T0 + 100.0) == ["m2", "m3"]
    g.clock("m2", T0 + 150.0)
    assert _released(g) == []                          # m3 still could precede it
    g.hold(_up("m3", 1, T0 + 99.0), now=0.0)           # m3's upload is earlier ...
    assert _released(g) == [("m3", T0 + 99.0)]         # ... and goes first
    assert _released(g) == []                          # m1 waits for m3 to pass 100
    g.clock("m3", T0 + 100.5)
    assert _released(g) == [("m1", T0 + 100.0)]


def test_ties_are_broken_by_mule_id():
    g = SimOrderGate()
    g.register("a", 1)
    g.register("b", 2)
    g.hold(_up("b", 1, T0 + 50.0), now=0.0)
    g.clock("a", T0 + 50.0)                   # a may still send (50, a) < (50, b)
    assert _released(g) == []
    g.hold(_up("a", 1, T0 + 50.0), now=0.0)
    assert _released(g) == [("a", T0 + 50.0)]  # a's own report does not block a; b waits
    g.clock("a", T0 + 50.0)
    assert _released(g) == []
    g.clock("a", T0 + 50.001)
    assert _released(g) == [("b", T0 + 50.0)]
    # The other way round: a's upload at a tie with b's bound goes at once.
    g = SimOrderGate()
    g.register("a", 1)
    g.register("b", 2)
    g.clock("b", T0 + 50.0)
    g.hold(_up("a", 1, T0 + 50.0), now=0.0)
    assert _released(g) == [("a", T0 + 50.0)]


def test_the_release_order_is_the_simulated_order_whatever_the_arrival_order():
    rng = random.Random(9)
    for _ in range(200):
        mules = ["m1", "m2", "m3"]
        per_mule = {m: sorted(rng.sample(range(1, 400), 5)) for m in mules}
        arrivals = [(m, i) for m in mules for i in range(5)]
        # Any interleaving that keeps each mule's own uploads in their order.
        rng.shuffle(arrivals)
        seen = {m: 0 for m in mules}
        order = []
        for m, _ in arrivals:
            order.append((m, per_mule[m][seen[m]]))
            seen[m] += 1
        g = SimOrderGate()
        for m in mules:
            g.register(m, 1)
        out = []
        for m, t in order:
            g.hold(_up(m, 1, T0 + t), now=0.0)
            out += _released(g)
        for m in mules:
            g.depart(m, 1)
            out += _released(g)
        assert out == sorted(((m, T0 + t) for m, t in order), key=lambda x: (x[1], x[0]))
        assert g.held_count() == 0


def test_a_mules_own_ups_leave_in_arrival_order():
    g = SimOrderGate()
    g.register("m1", 1)
    g.register("m2", 1)
    g.hold(_up("m1", 1, T0 + 10.0), now=0.0)
    g.hold(_up("m1", 2, T0 + 20.0), now=0.0)
    g.depart("m2", 1)
    rounds = []
    while (r := g.pop_ready(now=0.0)) is not None:
        rounds.append(r.up.partial_aggregate.mission_round)
    assert rounds == [1, 2]


@pytest.mark.parametrize("how", ["done", "departed"])
def test_the_done_marker_or_a_departure_releases(how):
    g = SimOrderGate()
    g.register("fast", 1)
    g.register("gone", 2)
    g.hold(_up("fast", 1, T0 + 500.0), now=0.0)
    assert _released(g) == []
    assert g.depart("gone", 2 if how == "departed" else None)
    assert g.tracked() == ["fast"]
    assert _released(g) == [("fast", T0 + 500.0)]
    assert not g.depart("gone", 2)                    # a second departure changes nothing


def test_the_departure_of_an_older_session_is_ignored_and_a_new_session_starts_over():
    g = SimOrderGate()
    g.register("m1", 1)
    g.register("m2", 1)
    g.hold(_up("m2", 3, T0 + 900.0), now=0.0)
    g.clock("m1", T0 + 950.0)
    g.register("m1", 2)                   # m1 restarted: its clock syncs anew
    assert g.bound("m1") == float("-inf")
    assert _released(g) == []
    assert not g.depart("m1", 1)          # the old connection's end
    assert _released(g) == []
    assert g.depart("m1", 2)
    assert _released(g) == [("m2", T0 + 900.0)]


def test_the_bootstrap_tracking_keeps_a_known_session_and_its_reports():
    g = SimOrderGate()
    g.register("m1", 4)
    g.hold(_up("m1", 1, T0 + 10.0), now=0.0)
    g.register("m1", None)                # the bootstrap diff, after the marker
    g.register("m1", 4)                   # the marker, after the bootstrap
    assert g.bound("m1") == T0 + 10.0


def test_the_markers_of_an_older_session_change_nothing():
    """A restarted mule is tracked under its live session from its bootstrap;
    the markers its crashed session left in the dock's queue, read only now,
    are stale: they neither restart nor end the live mule's tracking, and
    the crashed process's clock is not the restarted one's (review P3
    final, protocol F1)."""
    g = SimOrderGate()
    g.register("m1", 3)                        # the restarted m1's bootstrap
    g.register("m2", 2)
    g.hold(_up("m2", 1, T0 + 500.0), now=0.0)
    g.clock("m1", T0 + 50.0)                   # the live m1 reports
    g.register("m1", 1)                        # the crashed session's registration ...
    assert g.bound("m1") == T0 + 50.0          # ... keeps the live mule's report
    assert not g.depart("m1", 1)               # ... its departure
    g.clock("m1", T0 + 900.0, session=1)       # ... and its clock
    assert g.tracked() == ["m1", "m2"] and g.bound("m1") == T0 + 50.0
    assert _released(g) == []                  # the live m1 may still upload first
    g.clock("m1", T0 + 600.0, session=3)       # its own session's report counts
    assert _released(g) == [("m2", T0 + 500.0)]


def test_mules_waiting_for_their_answer_are_exempt():
    g = SimOrderGate()
    for m in ("a", "b", "c"):
        g.register(m, 1)
    g.hold(_up("a", 1, T0 + 100.0), now=0.0)
    g.hold(_up("b", 1, T0 + 150.0), now=0.0)
    g.hold(_up("c", 1, T0 + 200.0), now=0.0)
    assert _released(g) == [("a", T0 + 100.0)]
    # A quorum of 2: a's partial waits for b's. a is blocked at its dock and
    # its next upload cannot precede b's, so it must not hold b back.
    assert _released(g) == []
    assert _released(g, exempt={"a"}) == [("b", T0 + 150.0)]


def test_an_up_without_simulated_time_or_with_a_wall_stamp_goes_at_once():
    g = SimOrderGate()
    g.register("m1", 1)
    g.register("m2", 1)
    g.hold(_up("m1", 1, None), now=0.0)
    g.hold(_up("m2", 1, 1.7e9), now=0.0)      # a wall stamp is no simulated time
    rel = [g.pop_ready(now=0.0), g.pop_ready(now=0.0)]
    assert [str(r.up.mule_id) for r in rel] == ["m1", "m2"]
    assert all(r.unordered and not r.late for r in rel)
    assert g.bound("m2") == float("-inf") and g.frontier is None


def test_an_up_behind_the_frontier_is_late_and_goes_at_once():
    g = SimOrderGate()
    g.register("a", 1)
    g.register("b", 1)
    g.hold(_up("b", 1, T0 + 300.0), now=0.0)
    assert _released(g, exempt={"a"}) == [("b", T0 + 300.0)]
    assert g.frontier == T0 + 300.0
    # a stopped waiting (its down_wait_s ran out) and uploads an earlier one.
    g.hold(_up("a", 2, T0 + 250.0), now=0.0)
    r = g.pop_ready(now=0.0)
    assert r.late and not r.unordered and str(r.up.mule_id) == "a"
    assert g.frontier == T0 + 300.0


def test_a_late_up_does_not_wait_for_mules_that_have_not_reported():
    """Holding a late upload cannot restore the order, so it does not wait
    even for a mule that could still send an earlier one."""
    g = SimOrderGate()
    for m in ("a", "b", "c"):
        g.register(m, 1)
    g.hold(_up("b", 1, T0 + 300.0), now=0.0)
    assert _released(g, exempt={"a", "c"}) == [("b", T0 + 300.0)]
    g.hold(_up("a", 2, T0 + 250.0), now=0.0)
    assert g.blockers("a", T0 + 250.0) == ["c"]            # c has reported nothing
    r = g.pop_ready(now=0.0)
    assert r is not None and r.late and str(r.up.mule_id) == "a"


def test_an_up_tying_the_latest_released_one_is_ordered_as_usual_and_not_late():
    """A mule answered while it waited (exempt) syncs to the latest upload
    folded. When its next mission takes no simulated time its next upload
    ties that one, and sorts before it by mule id. It broke no rule: equal
    times are simultaneous, so it waits like any UP for the mules that could
    still precede it, and is not flagged late (review U9, finding 3)."""
    g = SimOrderGate()
    for m in ("a", "b", "c"):
        g.register(m, 1)
    g.hold(_up("a", 1, T0 + 100.0), now=0.0)
    g.hold(_up("c", 1, T0 + 120.0), now=0.0)
    g.hold(_up("b", 1, T0 + 200.0), now=0.0)
    # a and c wait at their docks (exempt) until b's upload is folded, and all
    # three are answered with its 200.
    assert _released(g) == [("a", T0 + 100.0)]
    assert _released(g, exempt={"a"}) == [("c", T0 + 120.0)]
    assert _released(g, exempt={"a", "c"}) == [("b", T0 + 200.0)]
    g.hold(_up("a", 2, T0 + 200.0), now=0.0)          # (200, a) sorts before (200, b)
    assert g.blockers("a", T0 + 200.0) == ["c"]        # c has said nothing past its 120
    assert g.pop_ready(now=0.0) is None
    g.clock("c", T0 + 200.0)                           # (200, c) sorts after (200, a)
    r = g.pop_ready(now=0.0)
    assert (str(r.up.mule_id), r.seq, r.late, r.unordered) == ("a", 4, False, False)
    assert g.frontier == T0 + 200.0
    # Strictly earlier than an upload already released is still late.
    g.hold(_up("c", 2, T0 + 199.0), now=0.0)
    r = g.pop_ready(now=0.0)
    assert (str(r.up.mule_id), r.late) == ("c", True)


def test_held_s_is_the_wall_time_held_and_seq_the_simulated_rank():
    g = SimOrderGate()
    g.register("m1", 1)
    g.register("m2", 1)
    g.hold(_up("m1", 1, T0 + 5.0), now=10.0)
    g.hold(_up("m2", 1, T0 + 1.0), now=12.5)
    g.depart("m1", 1)
    g.depart("m2", 1)
    a, b = g.pop_ready(now=20.0), g.pop_ready(now=21.0)
    assert (str(a.up.mule_id), a.seq, a.held_s) == ("m2", 1, 7.5)
    assert (str(b.up.mule_id), b.seq, b.held_s) == ("m1", 2, 11.0)


def test_an_up_from_an_untracked_mule_tracks_it_until_it_departs():
    g = SimOrderGate()
    g.register("m1", 1)
    g.hold(_up("m2", 1, T0 + 40.0), now=0.0)        # m2 said nothing before
    assert g.tracked() == ["m1", "m2"] and g.bound("m2") == T0 + 40.0
    g.hold(_up("m1", 1, T0 + 60.0), now=0.0)
    assert _released(g) == [("m2", T0 + 40.0)]
    assert g.depart("m2", 7)                          # any session: it had none
    assert _released(g) == [("m1", T0 + 60.0)]


# --------------------------------------------------------------------------- #
# No deadlock: a model of the mules and the cluster's fold
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _ModelUp:
    mule_id: str
    sim_upload_ts: float


class _ModelMule:
    def __init__(self, name: str, n_missions: int, pass_1, pass_2):
        self.name, self.left = name, n_missions
        self.pass_1, self.pass_2 = pass_1, pass_2
        self.clock = T0
        self.state = "flying"            # flying | docked | gone
        self.up_ts: Optional[float] = None
        self.answers: List[tuple] = []   # (own upload, DOWN's cluster time)


def _run_model(rng, mules, *, quorum: int, fedbuff_k: Optional[int] = None,
               exempt_waiting: bool = True, max_steps: int = 20000):
    """K mules and the cluster, stepped in a random wall order.

    A mule's step is its Pass 1 (it uploads and waits at the dock) or, once it
    has flown every mission, its departure. A cluster step reads one dock
    event, then folds every UP the gate releases: at a quorum, the partial
    waits until ``quorum`` are pending and the merge answers every waiting
    mule; under FedBuff every UP is answered at once. An answered mule syncs
    to ``max(own upload + turnaround, cluster time)`` and flies its Pass 2.
    Returns the gate, the released ``(sim time, mule)`` in fold order, the
    mules still waiting, and each release's ``late`` flag.
    """
    gate = SimOrderGate()
    by_name = {m.name: m for m in mules}
    for m in mules:
        gate.register(m.name, 1)
    dock: List[tuple] = []
    waiting, pending = set(), []
    cluster_ts = None
    released, late = [], []
    for _ in range(max_steps):
        flying = [m for m in mules if m.state == "flying"]
        moves = (["fly"] if flying else []) + (["dock"] if dock else [])
        if not moves:
            break
        if rng.choice(moves) == "fly":
            m = rng.choice(flying)
            if m.left == 0:
                m.state = "gone"
                dock.append(("gone", m.name))
                continue
            m.clock += m.pass_1(rng)
            m.up_ts, m.left, m.state = m.clock, m.left - 1, "docked"
            dock.append(("up", m.name, m.clock))
            continue
        item = dock.pop(0)
        if item[0] == "gone":
            gate.depart(item[1], 1)
        else:
            gate.hold(_ModelUp(item[1], item[2]), now=0.0)
        while True:
            r = gate.pop_ready(exempt=waiting if exempt_waiting else (), now=0.0)
            if r is None:
                break
            name, ts = r.up.mule_id, r.up.sim_upload_ts
            released.append((ts, name))
            late.append(r.late)
            cluster_ts = ts if cluster_ts is None else max(cluster_ts, ts)
            if fedbuff_k is not None:
                pending.append(name)
                if len(pending) >= fedbuff_k:
                    pending.clear()
                answered = [name]
            else:
                pending.append(name)
                waiting.add(name)
                answered = []
                if len(pending) >= quorum:
                    answered, pending = sorted(waiting), []
                    waiting.clear()
            for a in answered:
                mm = by_name[a]
                mm.answers.append((mm.up_ts, cluster_ts))
                mm.clock = max(mm.up_ts + TURNAROUND_S, cluster_ts) + mm.pass_2(rng)
                mm.state = "flying"
    else:
        pytest.fail("the model did not settle")
    return gate, released, waiting, late


def _three_mules(n_early: int = 1, *, empty_missions: bool = False):
    """One fast mule (short missions), one slow, one that finishes early.

    ``empty_missions``: each pass collects nothing half the time and then takes
    no simulated time at all (no stop, nothing to upload).
    """
    def span(lo, hi):
        if empty_missions:
            return lambda r: 0.0 if r.random() < 0.5 else r.uniform(lo, hi)
        return lambda r: r.uniform(lo, hi)

    return [
        _ModelMule("fast", 6, span(40, 60), span(20, 40)),
        _ModelMule("slow", 6, span(250, 300), span(100, 150)),
        _ModelMule("early", n_early, span(100, 200), span(50, 80)),
    ]


@pytest.mark.parametrize("mode", ["quorum-1", "fedbuff"])
def test_no_deadlock_at_k3_with_a_fast_a_slow_and_an_early_finishing_mule(mode):
    for seed in range(300):
        rng = random.Random(seed)
        mules = _three_mules()
        gate, released, waiting, late = _run_model(
            rng, mules, quorum=1, fedbuff_k=(2 if mode == "fedbuff" else None))
        assert all(m.state == "gone" for m in mules), seed        # every mule finished
        assert gate.held_count() == 0 and not waiting
        assert len(released) == 6 + 6 + 1
        assert released == sorted(released), seed                  # the simulated order
        assert not any(late), seed
        for m in mules:                                            # no drag (critic B9)
            assert all(cluster == own for own, cluster in m.answers), (seed, m.name)


def test_a_quorum_between_1_and_k_never_holds_an_up_forever():
    """At a quorum of 2 among 3 the gate releases every UP; a mule can still
    be left waiting at the end for a quorum that no longer forms (the other
    mules finished), which is the quorum's own limit (the Exp 4 driver refuses
    such a quorum), never the gate's."""
    for seed in range(300):
        rng = random.Random(seed)
        mules = _three_mules(n_early=2)
        gate, released, waiting, late = _run_model(rng, mules, quorum=2)
        assert gate.held_count() == 0, seed
        assert released == sorted(released), seed
        assert not any(late), seed
        assert all(m.state == "gone" or m.name in waiting for m in mules), seed


def test_mules_keeping_the_waiting_rule_are_never_late_even_when_they_tie():
    """A mule answered after waiting for its quorum syncs to the upload that
    closed it; when its next missions take no simulated time its next upload
    ties that one, and sorts before it when its id is lower. Every UP is still
    folded in time order, and none is flagged late (review U9, finding 3)."""
    inverted_ties = 0
    for seed in range(300):
        rng = random.Random(seed)
        mules = _three_mules(n_early=2, empty_missions=True)
        gate, released, _waiting, late = _run_model(rng, mules, quorum=2)
        assert gate.held_count() == 0, seed
        times = [t for t, _m in released]
        assert times == sorted(times), seed
        assert not any(late), seed
        inverted_ties += sum(
            1 for (t0, m0), (t1, m1) in zip(released, released[1:]) if t0 == t1 and m1 < m0
        )
    assert inverted_ties > 0          # the case this pins happened


def test_without_the_exemption_a_quorum_between_1_and_k_deadlocks():
    stuck = 0
    for seed in range(50):
        rng = random.Random(seed)
        gate, _released_ups, _waiting, _late = _run_model(
            rng, _three_mules(n_early=2), quorum=2, exempt_waiting=False)
        stuck += gate.held_count() > 0
    assert stuck > 0


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #

class _Recorder:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))

    def close(self):
        return

    def named(self, name):
        return [f for e, f in self.events if e == name]


@dataclass(frozen=True)
class _Docked:
    """A script step: from now on exactly these mules are docked, under these
    dock sessions (``{mule: session}``); a mule that reconnects gets a new one."""

    sessions: dict


class _EventDock:
    """A stand-in dock for the simulated order: scripted dock events, recorded DOWNs.

    A list in the script sets which mules are registered from then on (that
    call times out, as ``tests/unit/test_p3_cluster_sim.py``'s dock does); a
    :class:`_Docked` step does the same and also sets their dock sessions
    (``session_of``: ``sessions``, else 1).
    """

    def __init__(self, svc, script, registered, sessions=None):
        self._svc, self._script = svc, list(script)
        self.registered = list(registered)
        self.sessions = dict(sessions or {})
        self.downs = []
        #: Reads the service should not have made (the loop swallows errors,
        #: so a wrong read stops the service and is recorded instead).
        self.misused: List[str] = []

    def _misuse(self, what):
        self.misused.append(what)
        self._svc.request_stop()
        raise TimeoutError(what)

    def registered_mules(self):
        return [MuleID(m) for m in self.registered]

    def wait_for_mules(self, mules, timeout=None):
        return {str(m) for m in mules} <= set(self.registered)

    def session_of(self, mule_id):
        return self.sessions.get(str(mule_id), 1)

    def recv_dock_event(self, timeout=None):
        if not self._script:
            self._svc.request_stop()
            raise TimeoutError("script exhausted")
        step = self._script.pop(0)
        if isinstance(step, _Docked):
            self.registered = list(step.sessions)
            self.sessions.update(step.sessions)
            raise TimeoutError("no event this tick")
        if isinstance(step, list):
            self.registered = list(step)
            raise TimeoutError("no event this tick")
        return step

    def recv_up(self, timeout=None):
        self._misuse("recv_up: the simulated order reads dock events, not bare UPs")

    def send_down(self, bundle):
        self.downs.append(bundle)

    def close(self):
        return


def _service(*, mules=("m1", "m2"), quorum=1, spec=CUTOFF, clock="sim", **kw):
    rec = _Recorder()
    extra = dict(mission_clock=clock) if clock == "sim" else {}
    extra.update(kw)
    cfg = ClusterConfig(
        cluster_id="c-u9", dock_port=0, expected_mules=list(mules),
        seed_devices=[{"device_id": f"d{i}", "position": [float(i), 0.0, 0.0],
                       "assigned_mule": mules[i % len(mules)]} for i in range(2 * len(mules))],
        synth_batch_size=1, min_participation=quorum,
        aggregation=spec.rule, aggregation_params=spec.to_params(), **extra,
    )
    svc = ClusterService(cfg, events=rec)
    markers = svc.dock.sim_markers
    svc.dock.close()
    return svc, rec, markers


def _run(svc, script, registered=("m1", "m2"), sessions=None):
    dock = _EventDock(svc, script, registered, sessions)
    svc.dock = dock
    try:
        svc.run()
    finally:
        svc.shutdown()
    assert not dock.misused, dock.misused
    return dock


def _mk(mule, kind, ts=None, session=1):
    return DockClockMarker(mule_id=MuleID(mule), kind=kind, sim_ts=ts, session=session)


def _folded(rec):
    """(mule, round, sim time, seq) of every UP the service folded, in order."""
    return [(f["mule_id"], f["mission_round"], f["sim_upload_ts"], f["sim_order_seq"])
            for e, f in rec.events if e in ("up_bundle_ingested", "backhaul_upload_lost")]


@pytest.mark.parametrize("mules,quorum,spec,expected", [
    (("m1", "m2"), 1, CUTOFF, True),
    (("m1", "m2", "m3"), 2, CUTOFF, True),
    (("m1", "m2"), 2, AggregationSpec(rule="agg:fedbuff", buffer_k=2), True),
    (("m1", "m2"), 2, CUTOFF, False),                 # a quorum of every mule
    (("m1", "m2"), 2, AggregationSpec(), False),
    (("m1",), 1, CUTOFF, False),                      # one mule
    (("m1",), 1, AggregationSpec(rule="agg:fedbuff", buffer_k=2), False),   # FedBuff or not
])
def test_the_gate_is_on_exactly_below_a_full_quorum_on_the_simulated_clock(
        mules, quorum, spec, expected):
    svc, rec, markers = _service(mules=mules, quorum=quorum, spec=spec)
    svc.shutdown()
    assert needs_sim_order(svc.cfg) is expected
    assert (svc._sim_order is not None) is expected and markers is expected
    (ready,) = rec.named("cluster_ready")
    assert ready.get("sim_order") == (SIM_ORDER_CONSERVATIVE if expected else None)
    assert ("sim_order" in ready) is expected


def test_the_wall_clock_never_orders():
    svc, rec, markers = _service(clock="wall")
    svc.shutdown()
    assert not needs_sim_order(svc.cfg) and svc._sim_order is None and not markers
    (ready,) = rec.named("cluster_ready")
    assert "sim_order" not in ready and "mission_clock" not in ready


def test_uploads_are_folded_in_simulated_order_and_answered_with_their_own_time():
    """m2's upload arrives first but completed later: m1's is folded first, and
    each DOWN carries its uploader's own time (no drag to the other mule's
    later upload, critic B9)."""
    svc, rec, _ = _service()
    dock = _run(svc, [
        _mk("m1", MARKER_REGISTERED), _mk("m2", MARKER_REGISTERED),
        _up("m2", 1, T0 + 300.0),
        _up("m1", 1, T0 + 100.0),
        _up("m1", 2, T0 + 420.0, base_version=1),
        _mk("m2", MARKER_DEPARTED), _mk("m1", MARKER_DEPARTED),
    ])
    assert _folded(rec) == [("m1", 1, T0 + 100.0, 1), ("m2", 1, T0 + 300.0, 2),
                            ("m1", 2, T0 + 420.0, 3)]
    answers = [(str(d.mule_id), d.cluster_sim_ts) for d in dock.downs[2:]]
    assert answers == [("m1", T0 + 100.0), ("m2", T0 + 300.0), ("m1", T0 + 420.0)]
    closed = rec.named("cluster_round_closed")
    assert [c["sim_ts"] for c in closed] == [T0 + 100.0, T0 + 300.0, T0 + 420.0]
    ingested = rec.named("up_bundle_ingested")
    assert all(isinstance(f["held_wall_s"], float) and f["sim_order_late"] is False
               for f in ingested)
    assert [f["mule_id"] for f in rec.named("mule_departed")] == ["m2", "m1"]


def test_a_clock_marker_releases_and_the_done_marker_releases():
    svc, rec, _ = _service()
    _run(svc, [
        _up("m1", 1, T0 + 200.0),
        _mk("m2", MARKER_CLOCK, T0 + 110.0),     # not past m1's upload yet
        _up("m2", 1, T0 + 120.0),                # m2's own upload goes first
        _mk("m2", MARKER_CLOCK, T0 + 250.0),     # now past it
        _up("m2", 2, T0 + 600.0),
        _mk("m1", MARKER_DONE, T0 + 700.0),      # m1 will upload no more
    ])
    assert [(m, r) for m, r, _t, _s in _folded(rec)] == [("m2", 1), ("m1", 1), ("m2", 2)]
    (gone,) = rec.named("mule_departed")
    assert gone == {"mule_id": "m1", "reason": MARKER_DONE, "sim_ts": T0 + 700.0, "held": 0}


def test_a_mule_bootstrapped_before_its_registration_is_read_is_waited_for():
    """m2 registered after m1 uploaded, so its marker sits behind m1's UP, but
    it was bootstrapped at the epoch and may upload anything from there."""
    svc, rec, _ = _service()
    _run(svc, [
        _mk("m1", MARKER_REGISTERED),
        _up("m1", 1, T0 + 500.0),
        _mk("m2", MARKER_REGISTERED),
        _up("m2", 1, T0 + 90.0),
        _mk("m1", MARKER_DEPARTED), _mk("m2", MARKER_DEPARTED),
    ])
    assert [(m, t) for m, _r, t, _s in _folded(rec)] == [("m2", T0 + 90.0), ("m1", T0 + 500.0)]


def test_a_stray_dock_event_is_ignored():
    svc, rec, _ = _service()
    _run(svc, [
        {"not": "an upload"},
        _up("m1", 1, T0 + 10.0), _up("m2", 1, T0 + 20.0),
        _mk("m1", MARKER_DEPARTED), _mk("m2", MARKER_DEPARTED),
    ])
    assert [m for m, _r, _t, _s in _folded(rec)] == ["m1", "m2"]


def test_a_mule_registering_again_is_tracked_under_its_new_session():
    """m2's process registered again without a new bootstrap (the cluster
    never saw it gone) and left again: its new session's departure is its
    departure, so m1's upload is not held for a mule that is gone."""
    svc, rec, _ = _service()
    _run(svc, [
        _up("m1", 1, T0 + 100.0),                 # m2 has reported nothing yet
        _mk("m2", MARKER_REGISTERED, session=2),
        _mk("m2", MARKER_DEPARTED, session=2),
        _mk("m1", MARKER_DEPARTED),
    ])
    assert [(m, t) for m, _r, t, _s in _folded(rec)] == [("m1", T0 + 100.0)]
    assert [f["mule_id"] for f in rec.named("mule_departed")] == ["m2", "m1"]


def test_a_restarted_mule_is_tracked_under_its_new_session_from_its_new_bootstrap():
    """m1 restarts: the cluster sees its connection gone, then back as dock
    session 3, and bootstraps it again before it reads the old session's
    departure (that marker sat behind m2's UP). The gate tracks m1 under
    session 3 from that bootstrap, so the stale departure is ignored and
    m2's UP at 500 stays held: the restarted m1 synced to its bootstrap DOWN
    and may still upload before 500, as it does (review U9, finding 1)."""
    svc, rec, _ = _service()
    dock = _run(svc, [
        _mk("m1", MARKER_REGISTERED, session=1), _mk("m2", MARKER_REGISTERED, session=2),
        _up("m2", 1, T0 + 500.0),
        _Docked({"m2": 2}),                           # m1's connection ended
        _Docked({"m1": 3, "m2": 2}),                  # m1 is back as session 3
        _mk("m1", MARKER_DEPARTED, session=1),        # the old session's end, read only now
        _mk("m1", MARKER_REGISTERED, session=3),
        _up("m1", 1, T0 + 200.0),                     # the restarted m1's first upload
        _mk("m1", MARKER_DEPARTED, session=3), _mk("m2", MARKER_DEPARTED, session=2),
    ], sessions={"m1": 1, "m2": 2})
    assert [f["mule_id"] for f in rec.named("mule_bootstrapped")] == ["m1", "m2", "m1"]
    assert [str(d.mule_id) for d in dock.downs[:3]] == ["m1", "m2", "m1"]
    ingested = rec.named("up_bundle_ingested")
    assert [(f["mule_id"], f["sim_upload_ts"], f["sim_order_seq"], f["sim_order_late"])
            for f in ingested] == [("m1", T0 + 200.0, 1, False), ("m2", T0 + 500.0, 2, False)]
    # Only the live sessions' departures stop the wait, and are traced.
    assert [(f["mule_id"], f["reason"]) for f in rec.named("mule_departed")] == [
        ("m1", MARKER_DEPARTED), ("m2", MARKER_DEPARTED)]
    assert svc.metrics.snapshot().get("counter.sim_order_late_uploads", 0) == 0


def _restart_in_the_startup_wait(stale_clock: Optional[float] = None):
    """m1 docked (dock session 1) and crashed while the cluster waited for its
    mules, m2 docked (session 2) and uploaded at 500, and m1 was restarted
    (session 3). The startup wait bootstrapped both before the service read
    the queue, so the gate tracks m1 under its live session 3 first. The
    script is the queue in the order a real ``TCPDockLinkServer`` filled it
    (review P3 final, protocol F1); ``stale_clock``: the crashed process also
    reported that clock before it crashed."""
    crashed = [_mk("m1", MARKER_REGISTERED, session=1)]
    if stale_clock is not None:
        crashed.append(_mk("m1", MARKER_CLOCK, stale_clock, session=1))
    crashed.append(_mk("m1", MARKER_DEPARTED, session=1))    # its end, read only now
    svc, rec, _ = _service()
    dock = _run(svc, crashed + [
        _mk("m2", MARKER_REGISTERED, session=2),
        _up("m2", 1, T0 + 500.0),                     # held: the live m1 has said nothing
        _mk("m1", MARKER_REGISTERED, session=3),      # the restarted m1, bootstrapped already
        _up("m1", 1, T0 + 200.0),                     # its clock started over at the epoch
        _mk("m1", MARKER_CLOCK, T0 + 600.0, session=3),
        _mk("m1", MARKER_DEPARTED, session=3), _mk("m2", MARKER_DEPARTED, session=2),
    ], sessions={"m1": 3, "m2": 2})
    return svc, rec, dock


@pytest.mark.parametrize("stale_clock", [None, T0 + 900.0])
def test_a_mule_restarted_in_the_startup_wait_stays_tracked_under_its_live_session(stale_clock):
    """The crashed session's markers change nothing: m2's upload stays held
    until the live m1 reports, m1's upload at 200 is folded first, each mule
    is answered with its own time (no drag, critic B9), and only the live
    sessions' departures are traced."""
    svc, rec, dock = _restart_in_the_startup_wait(stale_clock)
    assert [f["mule_id"] for f in rec.named("mule_bootstrapped")] == ["m1", "m2"]
    folds = [(f["mule_id"], f["sim_upload_ts"], f["sim_order_seq"], f["sim_order_late"])
             for e, f in rec.events if e in ("up_bundle_ingested", "backhaul_upload_lost")]
    assert folds == [("m1", T0 + 200.0, 1, False), ("m2", T0 + 500.0, 2, False)]
    assert [(str(d.mule_id), d.cluster_sim_ts) for d in dock.downs] == [
        ("m1", None), ("m2", None),                   # the bootstraps
        ("m1", T0 + 200.0), ("m2", T0 + 500.0)]
    assert [f["mule_id"] for f in rec.named("mule_departed")] == ["m1", "m2"]
    assert svc.metrics.snapshot().get("counter.sim_order_late_uploads", 0) == 0


@pytest.mark.parametrize("m2_ts,expected", [
    # m2's upload falls between m1's two: m2's waits for the restarted m1, the
    # crashed UP for m2, and nothing is folded until the restarted m1 leaves.
    (T0 + 250.0, [("departed", "m1"), ("departed", "m1"),
                  ("m2", T0 + 250.0, 1, False), ("departed", "m2"),
                  ("m1", T0 + 300.0, 2, False), ("m1", T0 + 200.0, 3, True)]),
    # It does not: the crashed UP is folded without waiting for the restarted
    # m1, whose first upload is then late.
    (T0 + 350.0, [("departed", "m1"), ("m1", T0 + 300.0, 1, False),
                  ("m1", T0 + 200.0, 2, True), ("departed", "m1"),
                  ("m2", T0 + 350.0, 3, False), ("departed", "m2")]),
], ids=["stall", "late"])
def test_an_up_the_crashed_session_left_held_is_a_documented_gap(m2_ts, expected):
    """m1 crashes with its UP at 300 held (m2 has reported nothing yet) and is
    restarted as dock session 3; the restarted m1 uploads at 200, then m2.
    The crashed UP stays at the head of m1's queue, and m1's own bound never
    holds it back (:class:`SimOrderGate`, sessions: the second gap). Pinned
    so the documentation stays true: a fix fails this test and updates the
    docs (review of the P3 final fixes, cluster finding 2)."""
    svc, rec, _ = _service()
    dock = _run(svc, [
        _mk("m1", MARKER_REGISTERED, session=1), _mk("m2", MARKER_REGISTERED, session=2),
        _up("m1", 1, T0 + 300.0),                     # held: m2 has said nothing yet
        _Docked({"m2": 2}),                           # m1 crashed
        _mk("m1", MARKER_DEPARTED, session=1),
        _Docked({"m1": 3, "m2": 2}),                  # m1 restarted, bootstrapped again
        _mk("m1", MARKER_REGISTERED, session=3),
        _up("m1", 1, T0 + 200.0),                     # its clock started over
        _up("m2", 1, m2_ts),
        _mk("m1", MARKER_DEPARTED, session=3),        # its DOWN wait ran out, say
        _mk("m2", MARKER_DEPARTED, session=2),
    ], sessions={"m1": 1, "m2": 2})
    trace = [("departed", f["mule_id"]) if e == "mule_departed" else
             (f["mule_id"], f["sim_upload_ts"], f["sim_order_seq"], f["sim_order_late"])
             for e, f in rec.events if e in ("mule_departed", "up_bundle_ingested")]
    assert trace == expected
    assert [f["mule_id"] for f in rec.named("mule_bootstrapped")] == ["m1", "m2", "m1"]
    assert svc.metrics.snapshot()["counter.sim_order_late_uploads"] == 1
    # After the three bootstraps, both of m1's answers carry the crashed UP's
    # time: the restarted m1's upload at 200 is answered with a later one.
    assert sorted(d.cluster_sim_ts for d in dock.downs[3:] if str(d.mule_id) == "m1") == [
        T0 + 300.0, T0 + 300.0]


def test_a_late_registering_mule_is_waited_for_from_its_bootstrap():
    svc, rec, _ = _service()
    svc._BOOTSTRAP_WAIT_S = 0.0          # m2 is not there at the start
    dock = _run(svc, [
        _up("m1", 1, T0 + 100.0),               # only m1 at the start: goes at once
        ["m1", "m2"],                           # m2 docks late: bootstrapped now
        _up("m1", 2, T0 + 400.0, base_version=1),
        _up("m2", 1, T0 + 350.0),
        _mk("m1", MARKER_DEPARTED), _mk("m2", MARKER_DEPARTED),
    ], registered=("m1",))
    assert [(m, t) for m, _r, t, _s in _folded(rec)] == [
        ("m1", T0 + 100.0), ("m2", T0 + 350.0), ("m1", T0 + 400.0)]
    boot_m2 = [d for d in dock.downs if str(d.mule_id) == "m2"][0]
    assert boot_m2.cluster_sim_ts == T0 + 100.0      # it joins at the cluster's time


def test_a_mule_bootstrapped_at_an_uploads_time_can_tie_it_and_is_not_late():
    """m0 docks late and is bootstrapped at m1's folded 100. Its first
    mission takes no simulated time, so its upload ties m1's and sorts before
    it by mule id; it happened after that fold, so it follows it, unflagged."""
    svc, rec, _ = _service(mules=("m0", "m1"))
    svc._BOOTSTRAP_WAIT_S = 0.0          # m0 is not there at the start
    dock = _run(svc, [
        _up("m1", 1, T0 + 100.0),
        ["m0", "m1"],                           # m0 docks late: bootstrapped now
        _up("m0", 1, T0 + 100.0),
        _mk("m0", MARKER_DEPARTED), _mk("m1", MARKER_DEPARTED),
    ], registered=("m1",))
    boot_m0 = [d for d in dock.downs if str(d.mule_id) == "m0"][0]
    assert boot_m0.cluster_sim_ts == T0 + 100.0
    assert [(f["mule_id"], f["sim_order_seq"], f["sim_order_late"])
            for f in rec.named("up_bundle_ingested")] == [("m1", 1, False), ("m0", 2, False)]
    assert svc.metrics.snapshot().get("counter.sim_order_late_uploads", 0) == 0


def test_lost_uploads_are_folded_in_order_too():
    svc, rec, _ = _service()
    svc._upload_lost = lambda up, rnd: str(up.mule_id) == "m2"
    dock = _run(svc, [
        _up("m2", 1, T0 + 80.0), _up("m1", 1, T0 + 60.0),
        _mk("m1", MARKER_DEPARTED), _mk("m2", MARKER_DEPARTED),
    ])
    assert _folded(rec) == [("m1", 1, T0 + 60.0, 1), ("m2", 1, T0 + 80.0, 2)]
    (lost,) = rec.named("backhaul_upload_lost")
    assert lost["sim_order_seq"] == 2 and "awaits_quorum" not in lost
    assert [(str(d.mule_id), d.cluster_sim_ts) for d in dock.downs[2:]] == [
        ("m1", T0 + 60.0), ("m2", T0 + 60.0)]


def test_fedbuff_with_several_mules_buffers_in_simulated_order():
    # K counts weight mass: two of these partials (5 examples each) fill it.
    spec = AggregationSpec(rule="agg:fedbuff", buffer_k=10)
    svc, rec, _ = _service(quorum=2, spec=spec)
    _run(svc, [
        _up("m2", 1, T0 + 30.0, spec=spec), _up("m1", 1, T0 + 20.0, spec=spec),
        _up("m1", 2, T0 + 90.0, spec=spec), _up("m2", 2, T0 + 40.0, spec=spec),
        _mk("m1", MARKER_DEPARTED), _mk("m2", MARKER_DEPARTED),
    ])
    assert [(m, t) for m, _r, t, _s in _folded(rec)] == [
        ("m1", T0 + 20.0), ("m2", T0 + 30.0), ("m2", T0 + 40.0), ("m1", T0 + 90.0)]
    merges = [f for f in rec.named("cluster_merge") if f["applied"]]
    assert [m["partials"] for m in merges] == [[["m1", 1], ["m2", 1]], [["m2", 2], ["m1", 2]]]


def test_a_quorum_of_two_among_three_is_not_held_back_by_the_waiting_mule():
    svc, rec, _ = _service(mules=("a", "b", "c"), quorum=2)
    dock = _run(svc, [
        _up("a", 1, T0 + 100.0), _up("b", 1, T0 + 150.0), _up("c", 1, T0 + 200.0),
        _mk("a", MARKER_DEPARTED), _mk("b", MARKER_DEPARTED), _mk("c", MARKER_DEPARTED),
    ], registered=("a", "b", "c"))
    assert [m for m, _r, _t, _s in _folded(rec)] == ["a", "b", "c"]
    (closed,) = rec.named("cluster_round_closed")
    assert closed["sim_ts"] == T0 + 150.0
    answered = sorted((str(d.mule_id), d.cluster_sim_ts) for d in dock.downs[3:])
    assert answered == [("a", T0 + 150.0), ("b", T0 + 150.0)]     # c still waits


def test_a_mule_that_stops_waiting_can_upload_late_and_is_flagged():
    """a and b wait for a quorum of 3 that does not form; a's DOWN wait runs
    out (``down_wait_s``) and it flies on: its next upload completed before
    b's, already folded while a was exempt. It is folded at once (refused by
    the open round, which holds a's first partial) and flagged."""
    svc, rec, _ = _service(mules=("a", "b", "c", "d"), quorum=3)
    _run(svc, [
        _mk("c", MARKER_DEPARTED), _mk("d", MARKER_DEPARTED),
        _up("a", 1, T0 + 100.0),
        _up("b", 1, T0 + 300.0),
        _up("a", 2, T0 + 200.0),                  # behind b's 300: late
        _mk("a", MARKER_DEPARTED), _mk("b", MARKER_DEPARTED),
    ], registered=("a", "b", "c", "d"))
    ingested = rec.named("up_bundle_ingested")
    assert [(f["mule_id"], f["sim_order_late"]) for f in ingested] == [
        ("a", False), ("b", False), ("a", True)]
    assert ingested[2]["partial_refused"] is True
    assert svc.metrics.snapshot()["counter.sim_order_late_uploads"] == 1


def test_an_upload_tying_the_one_its_mule_was_synced_to_is_not_flagged(caplog):
    """A quorum of 2 among 3: a waits at its dock; b's upload at 200 closes
    the quorum and both are answered at 200. a's next mission takes no
    simulated time, so its upload ties b's and sorts before it by mule id.
    a broke no rule: it is folded next and not flagged late (review U9,
    finding 3)."""
    svc, rec, _ = _service(mules=("a", "b", "c"), quorum=2)
    with caplog.at_level(logging.WARNING, logger="hermes.processes.cluster"):
        dock = _run(svc, [
            _mk("c", MARKER_DEPARTED),
            _up("a", 1, T0 + 100.0),
            _up("b", 1, T0 + 200.0),
            _up("a", 2, T0 + 200.0, base_version=1),      # synced to 200; 0 s mission
            _mk("a", MARKER_DEPARTED), _mk("b", MARKER_DEPARTED),
        ], registered=("a", "b", "c"))
    assert sorted((str(d.mule_id), d.cluster_sim_ts) for d in dock.downs[3:]) == [
        ("a", T0 + 200.0), ("b", T0 + 200.0)]
    ingested = rec.named("up_bundle_ingested")
    assert [(f["mule_id"], f["mission_round"], f["sim_upload_ts"], f["sim_order_seq"],
             f["sim_order_late"]) for f in ingested] == [
        ("a", 1, T0 + 100.0, 1, False), ("b", 1, T0 + 200.0, 2, False),
        ("a", 2, T0 + 200.0, 3, False)]
    assert svc.metrics.snapshot().get("counter.sim_order_late_uploads", 0) == 0
    assert not [r for r in caplog.records if "already folded" in r.getMessage()]


def test_uploads_without_simulated_time_are_folded_at_once_logged_and_counted(caplog):
    """An UP without ``sim_upload_ts``, or with a wall stamp in it, cannot be
    ordered: it is folded at once, though neither mule has reported a time,
    logged, and counted as ``sim_order_unordered_uploads`` (review U9,
    finding 4)."""
    svc, rec, _ = _service()
    with caplog.at_level(logging.WARNING, logger="hermes.processes.cluster"):
        _run(svc, [
            _mk("m1", MARKER_REGISTERED), _mk("m2", MARKER_REGISTERED),
            _up("m2", 1, None),
            _up("m1", 1, 1.7e9),                          # a wall stamp is no simulated time
            _mk("m1", MARKER_DEPARTED), _mk("m2", MARKER_DEPARTED),
        ])
    assert _folded(rec) == [("m2", 1, None, 1), ("m1", 1, 1.7e9, 2)]
    assert all(f["sim_order_late"] is False for f in rec.named("up_bundle_ingested"))
    metrics = svc.metrics.snapshot()
    assert metrics["counter.sim_order_unordered_uploads"] == 2
    assert metrics.get("counter.sim_order_late_uploads", 0) == 0
    warned = [r for r in caplog.records if "carries no simulated time" in r.getMessage()]
    assert [r.levelno for r in warned] == [logging.WARNING, logging.WARNING]


_FOLD_EVENTS = ("up_bundle_ingested", "backhaul_upload_lost")


class _MovingWallClock:
    """The cluster module's ``time``, with ``monotonic`` moving 0.125 s per call.

    A scripted run takes less than one tick of the real clock (15.6 ms on
    Windows), so every UP would be held 0.0 s and a timer compared by its
    sum, min and max would pin only its count (review of the P3 final fixes,
    cluster finding 1). Each read moves this clock on, so every UP is held
    a positive time; a step that is a power of two keeps each sample and
    each sum exact. Everything else is the real ``time``.
    """

    STEP_S = 0.125

    def __init__(self, real):
        self._real = real
        self._reads = 0

    def monotonic(self) -> float:
        self._reads += 1
        return self.STEP_S * self._reads

    def __getattr__(self, name):
        return getattr(self._real, name)


def _counters_from_events(rec):
    """The gate's counters and the seconds model's, rebuilt from per-event fields.

    What a kept trace holds in their place: the counters themselves reach a
    trace only in the end-of-run ``metrics_snapshot``, which a cluster the
    Exp 4 orchestrator stops (TerminateProcess, on Windows) never writes.
    """
    (ready,) = rec.named("cluster_ready")
    folds = [f for e, f in rec.events if e in _FOLD_EVENTS]
    ordered = [f for f in folds if "sim_order_seq" in f]
    held = [f["held_wall_s"] for f in ordered]
    seconds = ready.get("backhaul_model") == "seconds"
    return {
        "counter.mules_departed": len(rec.named("mule_departed")),
        "counter.sim_order_late_uploads": sum(f["sim_order_late"] is True for f in ordered),
        "counter.sim_order_unordered_uploads": sum(
            f["sim_upload_ts"] is None or f["sim_upload_ts"] >= SIM_CEILING_S for f in ordered),
        "counter.backhaul_unpriced_uploads":
            sum(f["p_loss"] is None for f in folds) if seconds else 0,
        # The samples themselves, as far as the timer keeps them.
        "timer.sim_order_held_s": (len(held), sum(held), min(held, default=None),
                                   max(held, default=None)),
    }


def _counters_from_registry(snapshot):
    """The same counters as the registry holds them (an unused one is absent)."""
    out = {name: snapshot.get(name, 0) for name in (
        "counter.mules_departed", "counter.sim_order_late_uploads",
        "counter.sim_order_unordered_uploads", "counter.backhaul_unpriced_uploads")}
    timer = snapshot.get("timer.sim_order_held_s",
                         {"count": 0, "sum_s": 0.0, "min_s": None, "max_s": None})
    out["timer.sim_order_held_s"] = (timer["count"], timer["sum_s"], timer["min_s"],
                                     timer["max_s"])
    return out


def _counter_scenario(name):
    """(service settings, dock script, docked mules, and the expected
    departed, late, unordered and unpriced counts and held UPs)."""
    if name == "late":
        # a's DOWN wait ran out: its next upload completed before b's, folded.
        return dict(mules=("a", "b", "c", "d"), quorum=3), [
            _mk("c", MARKER_DEPARTED), _mk("d", MARKER_DEPARTED),
            _up("a", 1, T0 + 100.0), _up("b", 1, T0 + 300.0), _up("a", 2, T0 + 200.0),
            _mk("a", MARKER_DEPARTED), _mk("b", MARKER_DEPARTED),
        ], ("a", "b", "c", "d"), (4, 1, 0, 0, 3)
    if name == "unordered":
        return {}, [
            _mk("m1", MARKER_REGISTERED), _mk("m2", MARKER_REGISTERED),
            _up("m2", 1, None), _up("m1", 1, 1.7e9),          # no simulated time
            _up("m2", 2, T0 + 300.0, base_version=1), _up("m1", 2, T0 + 100.0, base_version=1),
            _mk("m1", MARKER_DEPARTED), _mk("m2", MARKER_DEPARTED),
        ], ("m1", "m2"), (2, 0, 2, 0, 4)
    if name == "departures":
        # m1 says done, so its connection's end is no departure; m2 registers
        # again under a new session, which then ends.
        return {}, [
            _up("m1", 1, T0 + 200.0),
            _mk("m2", MARKER_REGISTERED, session=2),
            _mk("m2", MARKER_CLOCK, T0 + 250.0, session=2),
            _mk("m1", MARKER_DONE, T0 + 700.0), _mk("m1", MARKER_DEPARTED),
            _mk("m2", MARKER_DEPARTED, session=2),
        ], ("m1", "m2"), (2, 0, 0, 0, 1)
    assert name == "unpriced"
    # The seconds model: two UPs the mules priced (one lost for sure, one
    # never) and two they did not, which are never lost.
    return dict(backhaul_model="seconds", trial_seed=7), [
        _mk("m1", MARKER_REGISTERED), _mk("m2", MARKER_REGISTERED),
        _up("m1", 1, T0 + 100.0), _up("m2", 1, T0 + 150.0, p_loss=1.0),
        _up("m1", 2, T0 + 200.0, base_version=1, p_loss=0.0), _up("m2", 2, T0 + 250.0),
        _mk("m1", MARKER_DEPARTED), _mk("m2", MARKER_DEPARTED),
    ], ("m1", "m2"), (2, 0, 0, 2, 4)


@pytest.mark.parametrize("scenario", ["late", "unordered", "departures", "unpriced"])
def test_the_counters_equal_their_per_event_equivalents(scenario, monkeypatch):
    """``mules_departed``, ``sim_order_late_uploads``,
    ``sim_order_unordered_uploads``, the ``sim_order_held_s`` timer and the
    seconds model's ``backhaul_unpriced_uploads`` are registry metrics: they
    reach a trace only in the ``metrics_snapshot`` that ``shutdown()`` writes
    at the end of a run, never when the Exp 4 orchestrator stops the cluster
    (TerminateProcess, on Windows). The per-event fields of a kept trace
    rebuild each exactly (review P3 final, e2e2 E2E2-1). The cluster's wall
    clock moves on every read (:class:`_MovingWallClock`), so the timer's
    samples are non-zero and its sum, min and max are compared too."""
    cluster_module = sys.modules[ClusterService.__module__]
    monkeypatch.setattr(cluster_module, "time", _MovingWallClock(cluster_module.time))
    kw, script, docked, expected = _counter_scenario(scenario)
    svc, rec, _ = _service(**kw)
    dock = _EventDock(svc, script, docked)
    svc.dock = dock
    try:
        svc.run()
        assert not dock.misused, dock.misused
        assert not rec.named("metrics_snapshot")      # nothing of the registry yet
        derived = _counters_from_events(rec)
    finally:
        svc.shutdown()
    registry = svc.metrics.snapshot()
    (snapshot,) = rec.named("metrics_snapshot")
    assert snapshot["metrics"] == registry
    assert derived == _counters_from_registry(registry)
    assert (derived["counter.mules_departed"], derived["counter.sim_order_late_uploads"],
            derived["counter.sim_order_unordered_uploads"],
            derived["counter.backhaul_unpriced_uploads"],
            derived["timer.sim_order_held_s"][0]) == expected
    assert derived["timer.sim_order_held_s"][2] > 0.0        # every sample is informative


@pytest.mark.parametrize("kw", [
    dict(mules=("m1", "m2"), quorum=2),
    dict(mules=("m1",), quorum=1),
    dict(clock="wall"),
])
def test_a_full_quorum_one_mule_and_the_wall_clock_keep_the_recorded_loop(kw):
    """No gate: the recorded loop reads bare UPs (``recv_up``) and the events
    carry no ordering fields."""
    svc, rec, markers = _service(**kw)
    assert svc._sim_order is None and not markers
    reads = []

    class _Dock(_EventDock):
        def recv_dock_event(self, timeout=None):
            self._misuse("recv_dock_event: the recorded loop reads bare UPs")

        def recv_up(self, timeout=None):
            reads.append(1)
            if not self._script:
                self._svc.request_stop()
                raise TimeoutError("script exhausted")
            return self._script.pop(0)

    mules = kw.get("mules", ("m1", "m2"))
    dock = _Dock(svc, [_up(m, 1, T0 + 10.0 * (i + 1)) for i, m in enumerate(mules)], mules)
    svc.dock = dock
    try:
        svc.run()
    finally:
        svc.shutdown()
    assert not dock.misused and len(reads) == len(mules) + 1
    ingested = rec.named("up_bundle_ingested")
    assert [f["mule_id"] for f in ingested] == list(mules)
    for f in ingested:
        assert not {"sim_order_seq", "held_wall_s", "sim_order_late"} & set(f)
    assert rec.named("mule_departed") == []


# --------------------------------------------------------------------------- #
# The dock (real TCP)
# --------------------------------------------------------------------------- #

def _events(server, n, timeout=5.0):
    return [server.recv_dock_event(timeout=timeout) for _ in range(n)]


def _nothing_more(server, wait=0.3):
    with pytest.raises(DockLinkTimeout):
        server.recv_dock_event(timeout=wait)


@pytest.fixture
def marker_server():
    server = TCPDockLinkServer(port=0, sim_markers=True)
    server.start()
    yield server
    server.close()


def _client(server, mule="m1"):
    client = TCPDockLinkClient(MuleID(mule), "127.0.0.1", server.port)
    assert server.wait_for_mules([MuleID(mule)], timeout=5.0)
    return client


def test_the_marker_server_queues_registration_ups_markers_and_departure_in_order(marker_server):
    client = _client(marker_server)
    session = marker_server.session_of(MuleID("m1"))
    client.client_send_up(_up("m1", 1, T0 + 10.0))
    assert client.client_send_clock(MuleID("m1"), T0 + 40.0)
    client.client_send_up(_up("m1", 2, T0 + 90.0))
    assert client.client_send_clock(MuleID("m1"), None, done=True)
    client.close()
    got = _events(marker_server, 6)
    kinds = [(e.kind, e.sim_ts, e.session) if isinstance(e, DockClockMarker)
             else ("up", e.sim_upload_ts, None) for e in got]
    assert kinds == [
        (MARKER_REGISTERED, None, session),
        ("up", T0 + 10.0, None),
        (MARKER_CLOCK, T0 + 40.0, session),
        ("up", T0 + 90.0, None),
        (MARKER_DONE, None, session),
        (MARKER_DEPARTED, None, session),
    ]
    assert all(str(e.mule_id) == "m1" for e in got)
    assert marker_server.session_of(MuleID("m1")) is None
    _nothing_more(marker_server)


def test_a_replaced_connection_is_no_departure(marker_server):
    first = _client(marker_server)
    old_reader = marker_server._reader_threads[MuleID("m1")]
    s1 = marker_server.session_of(MuleID("m1"))
    second = TCPDockLinkClient(MuleID("m1"), "127.0.0.1", marker_server.port)
    deadline = time.monotonic() + 5.0
    while marker_server.session_of(MuleID("m1")) == s1 and time.monotonic() < deadline:
        time.sleep(0.01)
    s2 = marker_server.session_of(MuleID("m1"))
    old_reader.join(timeout=5.0)
    assert not old_reader.is_alive() and s2 not in (None, s1)
    got = _events(marker_server, 2)
    assert [(e.kind, e.session) for e in got] == [(MARKER_REGISTERED, s1), (MARKER_REGISTERED, s2)]
    _nothing_more(marker_server)                   # the old connection's end: nothing
    second.close()
    (gone,) = _events(marker_server, 1)
    assert (gone.kind, gone.session) == (MARKER_DEPARTED, s2)
    first.close()


def test_recv_up_skips_the_markers(marker_server):
    client = _client(marker_server)
    client.client_send_clock(MuleID("m1"), T0 + 1.0)
    client.client_send_up(_up("m1", 1, T0 + 2.0))
    up = marker_server.recv_up(timeout=5.0)
    assert isinstance(up, UpBundle) and up.sim_upload_ts == T0 + 2.0
    client.close()
    with pytest.raises(DockLinkTimeout):
        marker_server.recv_up(timeout=0.3)         # only the departure is left


def test_a_mule_cannot_send_server_markers_or_report_for_another_mule(marker_server):
    client = _client(marker_server)
    send_message(client._sock, DockClockMarker(MuleID("m1"), MARKER_DEPARTED, session=99))
    send_message(client._sock, DockClockMarker(MuleID("m1"), MARKER_REGISTERED, session=99))
    send_message(client._sock, DockClockMarker(MuleID("m2"), MARKER_CLOCK, sim_ts=T0))
    client.client_send_clock(MuleID("m1"), T0 + 5.0)
    got = _events(marker_server, 2)
    assert [(e.kind, str(e.mule_id), e.sim_ts) for e in got] == [
        (MARKER_REGISTERED, "m1", None), (MARKER_CLOCK, "m1", T0 + 5.0)]
    assert got[1].session == got[0].session         # the server's session, not the mule's
    client.close()


def test_the_recorded_server_makes_no_marker_and_ignores_a_mules():
    server = TCPDockLinkServer(port=0)
    server.start()
    try:
        assert not server.sim_markers
        client = _client(server)
        client.client_send_clock(MuleID("m1"), T0 + 1.0)
        client.client_send_up(_up("m1", 1, T0 + 2.0))
        assert isinstance(server.recv_dock_event(timeout=5.0), UpBundle)
        client.close()
        deadline = time.monotonic() + 5.0
        while server.registered_mules() and time.monotonic() < deadline:
            time.sleep(0.01)
        _nothing_more(server)
        assert server.session_of(MuleID("m1")) is None
    finally:
        server.close()


def test_client_send_clock_checks_its_mule_and_a_closed_link(marker_server):
    client = _client(marker_server)
    with pytest.raises(DockLinkError, match="m2"):
        client.client_send_clock(MuleID("m2"), T0)
    with pytest.raises(ValueError, match="sim_ts"):
        client.client_send_clock(MuleID("m1"), None)   # a clock marker needs its time
    client.close()
    with pytest.raises(DockLinkError):
        client.client_send_clock(MuleID("m1"), T0)


@pytest.mark.parametrize("kw,err", [
    (dict(kind="tick"), ValueError),
    (dict(kind=MARKER_CLOCK), ValueError),
    (dict(kind=MARKER_CLOCK, sim_ts=float("nan")), ValueError),
    (dict(kind=MARKER_CLOCK, sim_ts="soon"), TypeError),
    (dict(kind=MARKER_DONE, sim_ts=True), TypeError),
])
def test_a_marker_is_validated(kw, err):
    with pytest.raises(err):
        DockClockMarker(mule_id=MuleID("m1"), **kw)
    assert DockClockMarker(MuleID("m1"), MARKER_CLOCK, np.float32(3.5)).sim_ts == 3.5


def test_a_transport_without_markers_sends_none_and_reads_bundles():
    link = LoopbackDockLink()
    assert link.client_send_clock(MuleID("m1"), T0) is False
    up = _up("m1", 1, T0)
    link.client_send_up(up)
    assert link.recv_dock_event(timeout=1.0) is up


def test_the_service_builds_its_dock_with_markers_and_reads_them_over_tcp():
    """End to end over TCP: two raw dock clients, the service's own server."""
    svc, rec, _ = _service()
    svc.dock.close()
    svc.dock = TCPDockLinkServer(port=0, sim_markers=True)
    svc.dock.start()
    svc._BOOTSTRAP_WAIT_S = 5.0
    ran = threading.Thread(target=svc.run, daemon=True)
    m1 = TCPDockLinkClient(MuleID("m1"), "127.0.0.1", svc.dock.port)
    m2 = TCPDockLinkClient(MuleID("m2"), "127.0.0.1", svc.dock.port)
    ran.start()
    try:
        for c in (m1, m2):
            c.client_recv_down(c.mule_id, timeout=10.0)            # bootstrap
        m2.client_send_up(_up("m2", 1, T0 + 300.0))
        m1.client_send_up(_up("m1", 1, T0 + 100.0))
        d1 = m1.client_recv_down(MuleID("m1"), timeout=10.0)
        assert d1.cluster_sim_ts == T0 + 100.0
        with pytest.raises(DockLinkError):
            m2.client_recv_down(MuleID("m2"), timeout=0.5)         # held: m1 is at 100
        m1.close()                                                  # m1 exits
        d2 = m2.client_recv_down(MuleID("m2"), timeout=10.0)
        assert d2.cluster_sim_ts == T0 + 300.0
    finally:
        m2.close()
        svc.request_stop()
        ran.join(timeout=10.0)
        svc.shutdown()
    assert [(m, t) for m, _r, t, _s in _folded(rec)] == [("m1", T0 + 100.0), ("m2", T0 + 300.0)]
    first = rec.named("mule_departed")[0]
    assert (first["mule_id"], first["reason"]) == ("m1", MARKER_DEPARTED)


# --------------------------------------------------------------------------- #
# The driver and the topology
# --------------------------------------------------------------------------- #

def _cell(arm="H1", seed=7, **params) -> Cell:
    p = {"N": 9, "rrf": 60.0, "n_missions": 3, "regime": "jittery"}
    p.update(params)
    return Cell(cell_id="|".join(f"{k}={v}" for k, v in sorted(p.items())), arm=arm,
                trial_index=0, seed=seed, params=p)


@pytest.mark.parametrize("kw", [
    dict(n_mules=3, aggregation="agg:cutoff"),
    dict(n_mules=2, aggregation="agg:fedex"),
    dict(n_mules=2, min_participation=2, aggregation="agg:fedbuff"),
])
def test_several_mules_below_a_full_quorum_run_on_the_simulated_clock(kw):
    driver = Exp4Driver(mission_clock="sim", realism=True, **kw)
    row, topo = T.run_stub_trial(driver, _cell())
    assert row["mission_clock"] == "sim" and len(topo.mules) == kw["n_mules"]
    # The driver re-costs its hard kill for exactly the cells the cluster orders.
    assert needs_sim_order(topo.cluster) and driver._sim_ordered


@pytest.mark.parametrize("kw", [
    dict(mission_clock="sim", n_mules=3, min_participation=3),
    dict(mission_clock="sim", n_mules=1),
    dict(mission_clock="sim", n_mules=1, aggregation="agg:fedbuff"),
    dict(n_mules=3, aggregation="agg:cutoff"),
])
def test_a_full_quorum_one_mule_or_the_wall_clock_is_not_ordered(kw):
    driver = Exp4Driver(realism=True, **kw)
    _row, topo = T.run_stub_trial(driver, _cell())
    assert not needs_sim_order(topo.cluster) and not driver._sim_ordered


def _sim_topology(quorum, rule, down_wait_s, k=3):
    mules = [MuleConfig(mule_id=f"m{i}", rf_range_m=60.0, mission_clock="sim", trial_seed=1,
                        expected_devices=[f"d{i}"], down_wait_s=down_wait_s) for i in range(k)]
    return TopologyConfig(
        cluster=ClusterConfig(cluster_id="c", mission_clock="sim", min_participation=quorum,
                              aggregation=rule),
        mules=mules, devices=[DeviceConfig(device_id=f"d{i}") for i in range(k)],
    )


def test_the_topology_validates_several_mules_below_a_full_quorum_on_the_clock():
    for quorum, rule in ((1, "agg:cutoff"), (2, "agg:fedbuff"), (3, "agg:fedbuff")):
        _sim_topology(quorum, rule, 60.0).validate()
    # A full quorum is not ordered and needs no DOWN wait of its own; nor does
    # one mule, FedBuff or not.
    _sim_topology(3, "agg:plain", None).validate()
    _sim_topology(1, "agg:fedbuff", None, k=1).validate()
    _sim_topology(1, "agg:cutoff", None, k=1).validate()


@pytest.mark.parametrize("quorum,rule", [(1, "agg:cutoff"), (3, "agg:fedbuff")])
def test_an_ordered_topology_needs_a_down_wait_on_every_mule(quorum, rule):
    """The gate may hold an upload for a while; the recorded 10 s DOWN wait's
    expiry would end the mule's run (``mission_failed``)."""
    from hermes.processes.config import TopologyValidationError

    with pytest.raises(TopologyValidationError, match="down_wait_s"):
        _sim_topology(quorum, rule, None).validate()


def test_the_hard_kill_allows_the_mules_to_run_one_at_a_time():
    one_at_a_time = Exp4Driver(mission_clock="sim", n_mules=3, aggregation="agg:cutoff")
    full_quorum = Exp4Driver(mission_clock="sim", n_mules=3, min_participation=3)
    per_mission = 2 * 3 * 3.0 * 3.0 + 10.0               # slice of 3, 3 s TTL
    assert one_at_a_time.ferry_wall_bound_s(n_devices=9, n_missions=4) == 90.0 + 4 * 3 * per_mission
    assert full_quorum.ferry_wall_bound_s(n_devices=9, n_missions=4) == 90.0 + 4 * 2 * per_mission
    fedbuff = Exp4Driver(mission_clock="sim", n_mules=3, min_participation=3,
                         aggregation="agg:fedbuff")
    assert fedbuff.ferry_wall_bound_s(n_devices=9, n_missions=4) == 90.0 + 4 * 3 * per_mission
    # Two mules: the same bound either way (and the recorded wall clock untouched).
    two = Exp4Driver(mission_clock="sim", n_mules=2, aggregation="agg:fedex")
    two_full = Exp4Driver(mission_clock="sim", n_mules=2, min_participation=2)
    assert two.ferry_wall_bound_s(n_devices=8, n_missions=4) == \
        two_full.ferry_wall_bound_s(n_devices=8, n_missions=4)
    assert Exp4Driver(n_mules=3, aggregation="agg:cutoff").trial_wall_budget_s(
        n_devices=9, n_missions=4) == 120.0
