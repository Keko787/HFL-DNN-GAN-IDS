"""FeRRy Phase 2 — the dock when several mules share one cluster.

* R1: the cluster answers only the mules waiting at the dock, and a mule keeps
  the newest DOWN (and, when it may outlive a wait, drops stale ones before it
  uploads), so no mule carries a θ from a backlog.
* R2: a DOWN wait that runs out can be survived (``DownTimeout``); without the
  switch it fails exactly as it always did.
* R4: a bootstrap wait returns rather than raising, and the cluster bootstraps
  each mule as soon as it registers.
* R5: a mule that re-docks under its old id is bootstrapped again, and its old
  socket's reader cannot close the new one.
* Per-mule backhaul streams: with several mules each mule's loss draws do not
  depend on upload order; with one, the single recorded stream is unchanged.
* An all-empty ``agg:plain`` fold (every partial from ``dock_on_empty``) keeps
  the round open instead of raising.
* A lost upload under a quorum above 1 holds its mule's place with an empty
  partial, so the mules stay in step to the last mission; a partial the open
  round refuses is traced as refused, and its reports still reach the registry.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import (
    OUTCOME_EXPIRED,
    OUTCOME_MERGED,
    StubGeneratorHost,
)
from hermes.mission.aggregation_rules import AGG_CUTOFF, AGG_FEDBUFF, AggregationSpec
from hermes.mule import ClientCluster, ClientClusterError, MuleSupervisor
from hermes.mule.client_cluster import DownTimeout
from hermes.processes.cluster import ClusterService
from hermes.processes.config import ClusterConfig
from hermes.transport import (
    DockLinkError,
    LoopbackDockLink,
    LoopbackRFLink,
    TCPDockLinkClient,
    TCPDockLinkServer,
)
from hermes.transport.dock_link import DockLinkTimeout
from hermes.types import (
    ClusterAmendment,
    ContactHistory,
    DeviceID,
    DownBundle,
    MissionOutcome,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MissionSlice,
    MuleID,
    PartialAggregate,
    SpectrumSig,
    UpBundle,
    sign_down_bundle,
)

MA = MuleID("mA")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _down(mule: MuleID, issued_round: int) -> DownBundle:
    bundle = DownBundle(
        mule_id=mule,
        mission_slice=MissionSlice(
            mule_id=mule, device_ids=(DeviceID("d1"),),
            issued_round=issued_round, issued_at=time.time(),
        ),
        theta_disc=[np.full((2,), float(issued_round), dtype=np.float32)],
        synth_batch=[np.ones((3,), dtype=np.float32)],
        cluster_amendments=ClusterAmendment(cluster_round=issued_round),
    )
    sign_down_bundle(bundle)
    return bundle


def _stage(cc: ClientCluster, mission_round: int = 1) -> None:
    report = MissionRoundCloseReport(
        mule_id=cc.mule_id, mission_round=mission_round, started_at=0.0, finished_at=1.0,
    )
    report.append(MissionRoundCloseLine(
        device_id=DeviceID("d1"), outcome=MissionOutcome.CLEAN, contact_ts=0.5,
    ))
    cc.collect(
        partial_aggregate=PartialAggregate(
            mule_id=cc.mule_id, mission_round=mission_round,
            weights=[np.array([0.1, 0.2], dtype=np.float32)], num_examples=4,
            contributing_devices=(DeviceID("d1"),),
        ),
        report=report,
        contacts=ContactHistory(mule_id=cc.mule_id, mission_round=mission_round),
    )


class _AnsweringDock(LoopbackDockLink):
    """Loopback whose 'cluster' answers every UP with a DOWN after ``delay_s``."""

    def __init__(self, answer_round: int, delay_s: float = 0.2) -> None:
        super().__init__()
        self._answer_round = answer_round
        self._delay_s = delay_s

    def client_send_up(self, bundle):
        super().client_send_up(bundle)
        threading.Timer(
            self._delay_s, lambda: self.send_down(_down(bundle.mule_id, self._answer_round)),
        ).start()


# --------------------------------------------------------------------------- #
# R1 — the mule keeps the newest DOWN, and drops stale ones before uploading
# --------------------------------------------------------------------------- #

def test_the_newest_of_several_queued_downs_wins_and_the_rest_are_dropped():
    dock = LoopbackDockLink()
    cc = ClientCluster(mule_id=MA, dock=dock)
    _stage(cc)
    for issued in (3, 5, 4):
        dock.send_down(_down(MA, issued))
    got = cc.run_dock_cycle()
    assert got.mission_slice.issued_round == 5
    assert cc.stale_downs_dropped == 2
    assert dock.client_drain_down(MA) == []          # nothing left to mislead the next dock


def test_a_lone_down_is_read_unchanged():
    dock = LoopbackDockLink()
    cc = ClientCluster(mule_id=MA, dock=dock)
    _stage(cc)
    sent = _down(MA, 2)
    dock.send_down(sent)
    assert cc.run_dock_cycle() is sent
    assert cc.stale_downs_dropped == 0


def test_a_recoverable_mule_drops_a_stale_down_before_uploading():
    """A DOWN queued before the upload answers an older one; with a survivable
    wait it is dropped and the real answer is waited for."""
    dock = _AnsweringDock(answer_round=7)
    dock.send_down(_down(MA, 2))                     # late answer to a wait given up on
    cc = ClientCluster(mule_id=MA, dock=dock, recoverable_down_wait=True)
    _stage(cc)
    got = cc.run_dock_cycle(down_timeout_s=2.0)
    assert got.mission_slice.issued_round == 7
    assert cc.stale_downs_dropped == 1


def test_without_the_switch_nothing_is_drained_before_the_upload():
    """The recorded contract: a DOWN already queued is read as the answer."""
    dock = _AnsweringDock(answer_round=7)
    dock.send_down(_down(MA, 2))
    cc = ClientCluster(mule_id=MA, dock=dock)
    _stage(cc)
    assert cc.run_dock_cycle().mission_slice.issued_round == 2


# --------------------------------------------------------------------------- #
# R2 — a survivable DOWN wait
# --------------------------------------------------------------------------- #

def test_a_recoverable_wait_raises_down_timeout():
    cc = ClientCluster(mule_id=MA, dock=LoopbackDockLink(), recoverable_down_wait=True)
    _stage(cc)
    with pytest.raises(DownTimeout, match="timed out after 0.05s"):
        cc.run_dock_cycle(down_timeout_s=0.05)
    # The UP went out; the mule can keep waiting without re-uploading.
    with pytest.raises(DownTimeout):
        cc.await_down(timeout=0.05)


def test_await_down_distributes_a_down_that_arrives_later():
    dock = LoopbackDockLink()
    seen = []
    cc = ClientCluster(mule_id=MA, dock=dock, recoverable_down_wait=True)
    cc.distributor.on_model_version = seen.append
    threading.Timer(0.1, lambda: dock.send_down(_down(MA, 4))).start()
    assert cc.await_down(timeout=2.0).mission_slice.issued_round == 4
    assert seen == [4]


def test_the_recorded_wait_still_fails_with_the_recorded_error():
    """Without the switch a timeout is the plain ClientClusterError it was,
    same message, so a single mule's failure path and trace are unchanged."""
    cc = ClientCluster(mule_id=MA, dock=LoopbackDockLink(), down_timeout_s=0.05)
    _stage(cc)
    with pytest.raises(ClientClusterError) as info:
        cc.run_dock_cycle()
    assert not isinstance(info.value, DownTimeout)
    assert str(info.value) == "DOWN recv failed: client_recv_down for 'mA' timed out after 0.05s"


def test_a_closed_link_is_never_a_survivable_timeout():
    dock = LoopbackDockLink()
    cc = ClientCluster(mule_id=MA, dock=dock, recoverable_down_wait=True)
    dock.close()
    with pytest.raises(ClientClusterError) as info:
        cc.await_down(timeout=0.05)
    assert not isinstance(info.value, DownTimeout)


# --------------------------------------------------------------------------- #
# R4 — bootstrap waits return instead of raising
# --------------------------------------------------------------------------- #

def test_a_bounded_bootstrap_wait_returns_none_when_nothing_came():
    dock = LoopbackDockLink()
    cc = ClientCluster(mule_id=MA, dock=dock)
    assert cc.bootstrap_down_only(timeout=0.05) is None
    dock.send_down(_down(MA, 0))
    assert cc.bootstrap_down_only(timeout=0.05).mission_slice.issued_round == 0


def test_the_unbounded_bootstrap_wait_still_raises():
    cc = ClientCluster(mule_id=MA, dock=LoopbackDockLink(), down_timeout_s=0.05)
    with pytest.raises(ClientClusterError, match="timed out"):
        cc.bootstrap_down_only()


def test_wait_for_initial_dock_times_out_with_false_not_an_exception():
    """The mule service polls this in 1 s ticks inside a 30 s window; a tick
    with no bootstrap yet used to raise after 10 s and kill the process."""
    dock = LoopbackDockLink()
    sup = MuleSupervisor(mule_id=MA, rf=LoopbackRFLink(), dock=dock, rf_range_m=60.0)
    started = time.monotonic()
    assert sup.wait_for_initial_dock(timeout=0.1) is False
    assert time.monotonic() - started < 2.0
    dock.send_down(_down(MA, 0))
    assert sup.wait_for_initial_dock(timeout=0.1) is True
    assert sup._next_theta_version == 0


# --------------------------------------------------------------------------- #
# TCP transport: timeouts, draining, and a re-registered mule (R5)
# --------------------------------------------------------------------------- #

def _wait_until(pred, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


@pytest.fixture
def tcp_server():
    server = TCPDockLinkServer(host="127.0.0.1", port=0)
    server.start()
    clients: List[TCPDockLinkClient] = []

    def connect(mule: MuleID) -> TCPDockLinkClient:
        c = TCPDockLinkClient(mule_id=mule, host="127.0.0.1", port=server.port)
        clients.append(c)
        return c

    yield server, connect
    for c in clients:
        c.close()
    server.close()


def test_tcp_client_drains_queued_downs_and_times_out_distinctly(tcp_server):
    server, connect = tcp_server
    client = connect(MA)
    assert server.wait_for_mules([MA], timeout=3.0)
    server.send_down(_down(MA, 1))
    server.send_down(_down(MA, 2))
    drained: List[DownBundle] = []
    assert _wait_until(lambda: drained.extend(client.client_drain_down(MA)) or len(drained) == 2)
    assert [d.mission_slice.issued_round for d in drained] == [1, 2]
    with pytest.raises(DockLinkTimeout):
        client.client_recv_down(MA, timeout=0.05)
    with pytest.raises(DockLinkTimeout):
        server.recv_up(timeout=0.05)
    assert issubclass(DockLinkTimeout, DockLinkError)   # every old handler still catches it


def test_a_mule_that_re_registers_keeps_its_new_connection(tcp_server):
    server, connect = tcp_server
    old = connect(MA)
    assert server.wait_for_mules([MA], timeout=3.0)
    new = connect(MA)
    # The server closes the old socket, so the old client's reader ends ...
    assert _wait_until(lambda: not old.is_available())
    # ... and that reader's exit must not take the new connection with it.
    time.sleep(0.2)
    assert server.registered_mules() == [MA]
    server.send_down(_down(MA, 9))
    assert new.client_recv_down(MA, timeout=2.0).mission_slice.issued_round == 9


# --------------------------------------------------------------------------- #
# The cluster service: who is answered (R1), bootstrap (R4, R5)
# --------------------------------------------------------------------------- #

def _up(mule: str, mission_round: int, *, spec: AggregationSpec, base_version: int = 0,
        empty: bool = False) -> UpBundle:
    m = MuleID(mule)
    theta = [np.zeros((4,), dtype=np.float32), np.ones((3, 3), dtype=np.float32) * 0.01]
    partial = PartialAggregate(
        mule_id=m, mission_round=mission_round,
        weights=[] if empty else [w + 0.1 for w in theta] if spec.is_plain
        else [np.full(w.shape, 0.1, dtype=np.float32) for w in theta],
        num_examples=0 if empty else 5,
        contributing_devices=() if empty else (DeviceID(f"{mule}-d0"),),
        rule=spec.rule, update_form=spec.update_form, base_version=base_version,
        weight_mass=0.0 if empty else 5.0, n_updates=0 if empty else 1,
    )
    return UpBundle(
        mule_id=m, partial_aggregate=partial,
        round_close_report=MissionRoundCloseReport(
            mule_id=m, mission_round=mission_round, started_at=0.0, finished_at=0.0,
        ),
        contact_history=ContactHistory(mule_id=m, mission_round=mission_round),
    )


class _FakeDock:
    """Stands in for ``TCPDockLinkServer`` so the service loop runs in-thread.

    ``script`` is consumed one step per ``recv_up``: an UpBundle is handed out,
    a list sets which mules are registered from then on (and the call times
    out), and when it runs out the service is stopped. Each DOWN is recorded
    as ``(n_up, mule)``: the number of UPs handed out when it was sent.
    """

    def __init__(self, svc, script, registered):
        self._svc = svc
        self._script = list(script)
        self.registered = list(registered)
        self.n_up = 0
        self.downs = []

    def registered_mules(self):
        return list(self.registered)

    def wait_for_mules(self, mules, timeout=None):
        return set(mules) <= set(self.registered)

    def recv_up(self, timeout=None):
        if not self._script:
            self._svc.request_stop()
            raise TimeoutError("script exhausted")
        step = self._script.pop(0)
        if isinstance(step, list):
            self.registered = list(step)
            raise TimeoutError("no UP this tick")
        self.n_up += 1
        return step

    def send_down(self, bundle):
        self.downs.append((self.n_up, str(bundle.mule_id)))

    def downs_after(self, n_up):
        return sorted(m for n, m in self.downs if n == n_up)

    def close(self):
        return


def _service(spec: AggregationSpec, *, min_participation: int = 1, mules=("m1", "m2"),
             **cfg_kwargs) -> ClusterService:
    cfg = ClusterConfig(
        cluster_id="cluster-multi-mule",
        dock_host="127.0.0.1",
        dock_port=0,
        expected_mules=list(mules),
        seed_devices=[{"device_id": f"d{i}", "position": [float(i), 0.0, 0.0],
                       "assigned_mule": mules[i % len(mules)]} for i in range(4)],
        synth_batch_size=1,
        min_participation=min_participation,
        aggregation=spec.rule,
        aggregation_params=spec.to_params(),
        **cfg_kwargs,
    )
    svc = ClusterService(cfg)
    svc.dock.close()   # the real TCP server is not needed
    return svc


def _run(svc: ClusterService, script, registered=("m1", "m2")) -> _FakeDock:
    dock = _FakeDock(svc, script, registered)
    svc.dock = dock
    try:
        svc.run()
    finally:
        svc.shutdown()
    return dock


def test_a_merge_answers_only_the_uploader_not_a_mule_in_flight():
    """Before Phase 2 every docked mule got a DOWN after every merge, so a mule
    in flight found a stale one queued at its next dock."""
    spec = AggregationSpec(rule=AGG_CUTOFF)
    svc = _service(spec)
    dock = _run(svc, [_up("m1", 1, spec=spec), _up("m2", 1, spec=spec, base_version=0),
                      _up("m1", 2, spec=spec, base_version=1)])
    assert dock.downs_after(0) == ["m1", "m2"]        # bootstrap
    assert dock.downs_after(1) == ["m1"]
    assert dock.downs_after(2) == ["m2"]
    assert dock.downs_after(3) == ["m1"]
    assert svc.cluster.cluster_round == 3
    assert svc._awaiting == []


def test_a_quorum_merge_answers_every_waiting_mule_once():
    """A second UP from a mule already waiting (it gave up and flew again) is
    refused as a duplicate and must not earn it a second DOWN."""
    spec = AggregationSpec(rule=AGG_CUTOFF)
    svc = _service(spec, min_participation=2)
    dock = _run(svc, [_up("m1", 1, spec=spec), _up("m1", 2, spec=spec),
                      _up("m2", 1, spec=spec)])
    assert dock.downs_after(1) == [] and dock.downs_after(2) == []
    assert dock.downs_after(3) == ["m1", "m2"]


def test_a_lost_backhaul_upload_answers_the_uploader_and_leaves_nobody_waiting():
    spec = AggregationSpec(rule=AGG_CUTOFF)
    svc = _service(spec, backhaul_loss_pct=100.0, backhaul_rng_seed=3)
    events = _record_events(svc)
    dock = _run(svc, [_up("m1", 1, spec=spec)])
    assert dock.downs_after(1) == ["m1"]
    assert svc._awaiting == [] and svc.cluster.cluster_round == 0
    # A quorum of 1 (every recorded run): the loss event is the recorded one.
    assert _named(events, "backhaul_upload_lost") == [{"mule_id": "m1", "mission_round": 1}]


# --------------------------------------------------------------------------- #
# Lost uploads under a quorum above 1, and refused partials
# --------------------------------------------------------------------------- #

def _record_events(svc: ClusterService) -> list:
    events = []
    svc.events.emit = lambda event, **fields: events.append((event, fields))
    return events


def _named(events, name) -> list:
    return [f for e, f in events if e == name]


def _losing(svc: ClusterService, lost) -> None:
    """Make exactly the uploads ``lost`` ((mule, mission_round) pairs) lost."""
    svc._backhaul_dropped = lambda mission_round, mule_id: (str(mule_id), mission_round) in lost


def test_a_lost_upload_holds_its_mules_place_in_a_full_quorum():
    """Quorum 2 under agg:plain: m1's upload is lost. It used to be answered at
    once with nothing in the round, so m1 flew on and the mules fell out of
    step; now an empty partial holds m1's place and m1 waits for the merge."""
    spec = AggregationSpec()
    svc = _service(spec, min_participation=2)
    _losing(svc, {("m1", 1)})
    events = _record_events(svc)
    m2_up = _up("m2", 1, spec=spec)
    dock = _run(svc, [_up("m1", 1, spec=spec), m2_up])
    assert dock.downs_after(1) == []                  # m1 waits, it is not answered
    assert dock.downs_after(2) == ["m1", "m2"]        # the merge answers both
    assert svc.cluster.cluster_round == 1 and svc._awaiting == []
    # The empty partial added nothing: θ is m2's partial alone.
    for got, exp in zip(svc.generator.get_global_disc_weights(),
                        m2_up.partial_aggregate.weights):
        np.testing.assert_allclose(got, exp)
    assert _named(events, "backhaul_upload_lost") == [
        {"mule_id": "m1", "mission_round": 1, "awaits_quorum": True},
    ]
    # m2's ingest closed the round, so the recorded close event is unchanged.
    assert _named(events, "cluster_round_closed") == [{"cluster_round": 1}]
    assert svc.metrics.counter_value("backhaul_uploads_lost") == 1


def test_a_round_closed_by_a_lost_upload_names_its_mule():
    spec = AggregationSpec()
    svc = _service(spec, min_participation=2)
    _losing(svc, {("m1", 1)})
    events = _record_events(svc)
    dock = _run(svc, [_up("m2", 1, spec=spec), _up("m1", 1, spec=spec)])
    assert dock.downs_after(2) == ["m1", "m2"]
    # No up_bundle_ingested precedes the close to say whose upload closed it.
    assert _named(events, "cluster_round_closed") == [{"cluster_round": 1, "mule_id": "m1"}]


def test_two_lost_uploads_expire_the_fold_and_release_both_mules():
    spec = AggregationSpec(rule=AGG_CUTOFF)
    svc = _service(spec, min_participation=2)
    _losing(svc, {("m1", 1), ("m2", 1)})
    events = _record_events(svc)
    dock = _run(svc, [_up("m1", 1, spec=spec), _up("m2", 1, spec=spec)])
    assert dock.downs_after(1) == []
    assert dock.downs_after(2) == ["m1", "m2"]
    assert svc.cluster.cluster_round == 0 and svc._awaiting == []
    (expired,) = _named(events, "cluster_merge_expired")
    assert expired["mule_id"] == "m2" and expired["partials"] == [["m1", 1], ["m2", 1]]
    assert svc.metrics.counter_value("ingest_failures") == 0


def test_lost_uploads_keep_a_full_quorum_in_step_to_the_last_mission():
    """m1 loses missions 1 and 3, m2 none. When a loss was answered at once,
    m1 had two partials in the cluster to m2's three, so m2's last one waited
    for a partner that had finished its run. Now every mule's every mission
    holds a place, and the last merge answers both mules."""
    spec = AggregationSpec()
    svc = _service(spec, min_participation=2)
    _losing(svc, {("m1", 1), ("m1", 3)})
    dock = _run(svc, [
        _up("m1", 1, spec=spec), _up("m2", 1, spec=spec),
        _up("m2", 2, spec=spec), _up("m1", 2, spec=spec),
        _up("m1", 3, spec=spec), _up("m2", 3, spec=spec),
    ])
    assert [dock.downs_after(n) for n in range(1, 7)] == [
        [], ["m1", "m2"], [], ["m1", "m2"], [], ["m1", "m2"],
    ]
    assert svc.cluster.cluster_round == 3 and svc._awaiting == []


def test_fedbuff_answers_a_lost_upload_at_once_whatever_the_quorum():
    """FedBuff's K is its own quorum; min_participation does not gate it."""
    spec = AggregationSpec(rule=AGG_FEDBUFF, buffer_k=2)
    svc = _service(spec, min_participation=2)
    _losing(svc, {("m1", 1)})
    events = _record_events(svc)
    dock = _run(svc, [_up("m1", 1, spec=spec)])
    assert dock.downs_after(1) == ["m1"]
    assert _named(events, "backhaul_upload_lost") == [{"mule_id": "m1", "mission_round": 1}]
    assert svc.cluster.pending_partials() == 0


def _up_with_report(mule: str, mission_round: int, *, spec: AggregationSpec,
                    device: str) -> UpBundle:
    up = _up(mule, mission_round, spec=spec)
    up.round_close_report.lines.append(MissionRoundCloseLine(
        device_id=DeviceID(device), outcome=MissionOutcome.CLEAN, contact_ts=0.5,
    ))
    return up


def test_a_refused_partial_is_traced_as_refused_and_its_reports_still_count():
    """m1 gave up waiting (down_wait_s) and uploaded mission 2 while mission 1
    still waited for m2. The round keeps mission 1's partial; mission 2's
    session still reaches the registry, and its ingest line says the partial
    was refused, so a trace does not credit it as merged."""
    spec = AggregationSpec(rule=AGG_CUTOFF)
    svc = _service(spec, min_participation=2)
    events = _record_events(svc)
    dock = _run(svc, [
        _up_with_report("m1", 1, spec=spec, device="d0"),
        _up_with_report("m1", 2, spec=spec, device="d2"),
        _up("m2", 1, spec=spec),
    ])
    ingested = _named(events, "up_bundle_ingested")
    assert ingested == [
        {"mule_id": "m1", "mission_round": 1},
        {"mule_id": "m1", "mission_round": 2, "partial_refused": True,
         "held_mission_round": 1},
        {"mule_id": "m2", "mission_round": 1},
    ]
    (merge,) = _named(events, "cluster_merge")
    assert merge["partials"] == [["m1", 1], ["m2", 1]]
    assert svc.registry.get(DeviceID("d2")).on_time_history == 1
    assert svc.metrics.counter_value("up_partials_refused") == 1
    assert dock.downs_after(3) == ["m1", "m2"]      # answered once, at the merge


def test_an_all_empty_plain_fold_releases_both_mules_and_closes_nothing():
    """dock_on_empty: two empty partials meet the quorum but carry no model."""
    spec = AggregationSpec()
    svc = _service(spec, min_participation=2)
    dock = _run(svc, [_up("m1", 1, spec=spec, empty=True), _up("m2", 1, spec=spec, empty=True)])
    assert dock.downs_after(2) == ["m1", "m2"]
    assert svc.cluster.cluster_round == 0
    assert svc.metrics.counter_value("ingest_failures") == 0


class _SlowRegistrationDock(_FakeDock):
    """``wait_for_mules`` blocks like the real server's, until all are there."""

    def wait_for_mules(self, mules, timeout=None):
        deadline = time.monotonic() + (timeout or 0.0)
        while time.monotonic() < deadline:
            if set(mules) <= set(self.registered):
                return True
            time.sleep(0.01)
        return set(mules) <= set(self.registered)


def test_each_mule_is_bootstrapped_as_soon_as_it_registers():
    """m2 registers 0.4 s late; m1's bootstrap must not wait for it. It used
    to: the cluster waited (up to 60 s) for every expected mule first."""
    spec = AggregationSpec(rule=AGG_CUTOFF)
    svc = _service(spec)
    svc._BOOTSTRAP_TICK_S = 0.05
    dock = _SlowRegistrationDock(svc, [], registered=["m1"])
    svc.dock = dock
    seen_when_sent = []
    original_send = dock.send_down

    def send_down(bundle):
        seen_when_sent.append((str(bundle.mule_id), tuple(dock.registered)))
        original_send(bundle)

    dock.send_down = send_down
    threading.Timer(0.4, lambda: setattr(dock, "registered", ["m1", "m2"])).start()
    try:
        svc.run()
    finally:
        svc.shutdown()
    assert seen_when_sent == [("m1", ("m1",)), ("m2", ("m1", "m2"))]


def test_a_mule_that_re_docks_is_bootstrapped_again():
    spec = AggregationSpec(rule=AGG_CUTOFF)
    svc = _service(spec, mules=("m1",))
    events = []
    svc.events.emit = lambda event, **fields: events.append((event, fields))
    dock = _run(svc, [[], ["m1"]], registered=["m1"])   # drops off, then back
    assert [m for _, m in dock.downs] == ["m1", "m1"]
    assert [f["mule_id"] for e, f in events if e == "mule_bootstrapped"] == ["m1", "m1"]


# --------------------------------------------------------------------------- #
# Backhaul-loss streams
# --------------------------------------------------------------------------- #

def _loss_service(expected_mules) -> ClusterService:
    cfg = ClusterConfig(
        cluster_id="cluster-backhaul-streams", dock_host="127.0.0.1", dock_port=0,
        expected_mules=list(expected_mules), backhaul_loss_pct=50.0, backhaul_rng_seed=11,
    )
    svc = ClusterService(cfg)
    svc.dock.close()
    return svc


def test_one_mule_keeps_the_single_recorded_stream():
    svc = _loss_service(["exp4-mule"])
    try:
        got = [svc._backhaul_dropped(m, MuleID("exp4-mule")) for m in range(1, 21)]
        legacy = np.random.default_rng(11)
        assert got == [float(legacy.random()) < 0.5 for _ in range(20)]
    finally:
        svc.shutdown()


def test_several_mules_draw_from_their_own_streams_whatever_the_upload_order():
    a, b = MuleID("exp4-mule-0"), MuleID("exp4-mule-1")
    first, second = _loss_service([a, b]), _loss_service([a, b])
    try:
        in_turn = [first._backhaul_dropped(m, mule) for m in range(1, 11) for mule in (a, b)]
        a_then_b = ([second._backhaul_dropped(m, a) for m in range(1, 11)]
                    + [second._backhaul_dropped(m, b) for m in range(1, 11)])
        assert in_turn[0::2] == a_then_b[:10]         # mule a's losses, either order
        assert in_turn[1::2] == a_then_b[10:]
        assert in_turn[0::2] != in_turn[1::2]         # and the two streams differ
    finally:
        first.shutdown()
        second.shutdown()


# --------------------------------------------------------------------------- #
# HFLHostCluster: empty partials under agg:plain
# --------------------------------------------------------------------------- #

def _plain_cluster(min_participation: int) -> HFLHostCluster:
    reg = DeviceRegistry()
    for i in range(2):
        reg.register(
            device_id=DeviceID(f"d{i}"), position=(float(i), 0.0, 0.0),
            spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
        )
    reg.rebalance([MuleID("m1"), MuleID("m2")], round_counter=0)
    return HFLHostCluster(
        registry=reg,
        generator=StubGeneratorHost(disc_weights=[
            np.zeros((4,), dtype=np.float32), np.ones((3, 3), dtype=np.float32) * 0.01,
        ]),
        dock=LoopbackDockLink(),
        min_participation=min_participation,
    )


def test_an_empty_partial_counts_toward_the_quorum_and_adds_nothing():
    spec = AggregationSpec()
    c = _plain_cluster(min_participation=2)
    full = _up("m1", 1, spec=spec)
    c.ingest_up_bundle(full)
    c.ingest_up_bundle(_up("m2", 1, spec=spec, empty=True))
    merged = c.aggregate_pending()
    assert c.last_outcome == OUTCOME_MERGED
    for got, exp in zip(merged, full.partial_aggregate.weights):
        np.testing.assert_allclose(got, exp)


def test_an_all_empty_plain_fold_expires_instead_of_raising():
    spec = AggregationSpec()
    c = _plain_cluster(min_participation=2)
    theta = c.generator.get_global_disc_weights()
    c.ingest_up_bundle(_up("m1", 1, spec=spec, empty=True))
    c.ingest_up_bundle(_up("m2", 1, spec=spec, empty=True))
    assert c.aggregate_pending() is None
    assert c.last_outcome == OUTCOME_EXPIRED
    assert c.last_merge["partials"] == [["m1", 1], ["m2", 1]]
    assert c.cluster_round == 0
    for got, exp in zip(c.generator.get_global_disc_weights(), theta):
        np.testing.assert_array_equal(got, exp)
    # The round stayed open for the mules' next uploads, which are not duplicates.
    c.ingest_up_bundle(_up("m1", 2, spec=spec))
    assert c.pending_partials() == 1
