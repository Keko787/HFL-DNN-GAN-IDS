"""FeRRy Phase 2 — a mule that shares its cluster with other mules.

In-process mule, cluster and devices on loopback links:

* ``down_wait_s`` (R2): an inter-pass dock with no answer within the wait is
  survived — Pass 2 is skipped, the mission's own θ and version are restaged —
  and the late answer is dropped before the next upload (R1), so the next
  mission reads its own answer. Without it the wait fails as it always did.
* ``dock_on_empty`` (R3): a mission that collected nothing still docks, with
  an empty partial the cluster counts and merges nothing from, and picks up
  the current θ. Without it an empty mission does not dock.
* The mule process exits non-zero when its loop ends on a failure, and emits
  ``dock_down_timeout`` for a survived wait.
"""

from __future__ import annotations

import threading
from typing import List

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import OUTCOME_EXPIRED, StubGeneratorHost
from hermes.mission import ClientMission, LocalTrainResult
from hermes.mule import ClientClusterError, MissionRunResult, MuleSupervisor, MuleSupervisorError
from hermes.mule.client_cluster import DownTimeout
from hermes.processes import mule as mule_process
from hermes.processes.config import MuleConfig
from hermes.transport import (
    DockLinkError,
    LoopbackDockLink,
    LoopbackRFLink,
    TCPDockLinkServer,
)
from hermes.types import DeviceID, FLState, MuleID, SpectrumSig

MULE = MuleID("mule-p2")
DEVICES = [DeviceID("dev-00"), DeviceID("dev-01")]
POSITIONS = [(0.0, 0.0, 0.0), (10.0, 5.0, 0.0)]


def _train(seed: int):
    rng = np.random.default_rng(seed)

    def _fit(theta, synth):
        after = [w + rng.normal(0.0, 0.01, size=w.shape).astype(w.dtype) for w in theta]
        return LocalTrainResult(
            delta_theta=after, num_examples=8, accuracy=0.8, auc=0.8, loss=0.2,
            theta_after=after,
        )
    return _fit


def _setup(*, serving: bool = True, **sup_kwargs):
    """Mule + cluster + two devices; the cluster has bootstrapped the mule."""
    rf, dock = LoopbackRFLink(), LoopbackDockLink()
    devices: List[ClientMission] = []
    for i, did in enumerate(DEVICES):
        rf.register_device(did)
        cm = ClientMission(
            device_id=did, rf=rf, local_train=_train(40 + i),
            solicit_timeout_s=1.5, disc_push_timeout_s=1.5,
        )
        cm.set_state(FLState.FL_OPEN)
        devices.append(cm)
    registry = DeviceRegistry()
    for did, pos in zip(DEVICES, POSITIONS):
        registry.register(
            device_id=did, position=pos,
            spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
        )
    registry.rebalance([MULE], round_counter=0)
    cluster = HFLHostCluster(
        registry=registry,
        generator=StubGeneratorHost(disc_weights=[
            np.zeros((4,), dtype=np.float32), np.ones((3, 3), dtype=np.float32) * 0.01,
        ]),
        dock=dock, synth_batch_size=1,
    )
    sup = MuleSupervisor(
        mule_id=MULE, rf=rf, dock=dock, session_ttl_s=0.5, rf_range_m=60.0,
        **sup_kwargs,
    )
    cluster.dock.send_down(cluster.dispatch_down_bundle(MULE))
    assert sup.wait_for_initial_dock(timeout=2.0)
    return sup, cluster, devices if serving else []


def _serve_devices(devices, n: int) -> List[threading.Thread]:
    workers = []
    for cm in devices:
        def _loop(client=cm):
            for _ in range(n):
                client.serve_once()
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        workers.append(t)
    return workers


def _answer_one_up(cluster: HFLHostCluster, timeout: float = 5.0) -> None:
    """The cluster side of one dock: ingest, fold, and answer the uploader."""
    up = cluster.dock.recv_up(timeout=timeout)
    cluster.ingest_up_bundle(up)
    if cluster.aggregate_pending() is not None:
        cluster.close_cluster_round()
    cluster.dock.send_down(cluster.dispatch_down_bundle(up.mule_id))


def _join(threads, timeout: float = 5.0) -> None:
    for t in threads:
        t.join(timeout=timeout)


# --------------------------------------------------------------------------- #
# down_wait_s — the inter-pass dock survives a missing DOWN
# --------------------------------------------------------------------------- #

def test_an_unanswered_dock_skips_pass_2_and_keeps_the_missions_theta():
    sup, cluster, devices = _setup(down_wait_s=0.3)
    theta_1 = [w.copy() for w in sup._next_theta]
    workers = _serve_devices(devices, 1)                # Pass 1 only: no Pass 2 comes
    result = sup.run_one_mission()
    _join(workers)

    assert result.down_timeout is True
    assert result.aggregate is not None and result.aggregate.base_version == 0
    assert result.pass_2_queue == [] and result.delivery_report is None
    # The next mission flies the same θ at the same version.
    assert sup._next_theta_version == 0
    for got, exp in zip(sup._next_theta, theta_1):
        np.testing.assert_array_equal(got, exp)


def test_the_late_answer_is_dropped_and_the_next_mission_reads_its_own():
    sup, cluster, devices = _setup(down_wait_s=0.3)
    workers = _serve_devices(devices, 1)
    assert sup.run_one_mission().down_timeout is True
    _join(workers)

    # The cluster now answers mission 1's upload: too late, the mule has left.
    _answer_one_up(cluster)
    assert cluster.cluster_round == 1

    dock_t = threading.Thread(target=_answer_one_up, args=(cluster,), daemon=True)
    dock_t.start()
    workers = _serve_devices(devices, 2)
    result = sup.run_one_mission()
    dock_t.join(timeout=5.0)
    _join(workers)

    assert result.down_timeout is False
    assert result.aggregate.base_version == 0           # it flew the restaged θ
    assert sup.client_cluster.stale_downs_dropped == 1   # round 1's late answer
    assert cluster.cluster_round == 2
    assert sup._next_theta_version == 2                  # its own answer, not round 1's
    assert result.delivery_report is not None


def test_without_down_wait_s_a_missing_down_fails_as_before():
    sup, cluster, devices = _setup()
    assert sup.down_wait_s is None
    assert sup.client_cluster.recoverable_down_wait is False
    sup.client_cluster.down_timeout_s = 0.2              # instead of the recorded 10 s
    workers = _serve_devices(devices, 1)
    with pytest.raises(ClientClusterError) as info:
        sup.run_one_mission()
    _join(workers)
    assert not isinstance(info.value, DownTimeout)


def test_down_wait_s_must_be_positive():
    with pytest.raises(MuleSupervisorError):
        MuleSupervisor(
            mule_id=MULE, rf=LoopbackRFLink(), dock=LoopbackDockLink(), down_wait_s=0.0,
        )


# --------------------------------------------------------------------------- #
# dock_on_empty — an empty mission still docks
# --------------------------------------------------------------------------- #

def test_an_empty_mission_docks_with_an_empty_partial_and_picks_up_the_current_theta():
    sup, cluster, _ = _setup(serving=False, dock_on_empty=True, down_wait_s=2.0)
    # Other mules' merges have moved the cluster on while this one flew.
    cluster.close_cluster_round()
    cluster.close_cluster_round()
    seen = []

    def _cluster_side():
        up = cluster.dock.recv_up(timeout=5.0)
        seen.append(up)
        cluster.ingest_up_bundle(up)
        assert cluster.aggregate_pending() is None
        cluster.dock.send_down(cluster.dispatch_down_bundle(up.mule_id))

    t = threading.Thread(target=_cluster_side, daemon=True)
    t.start()
    result = sup.run_one_mission()                      # no device answers
    t.join(timeout=5.0)

    assert result.empty is True and result.docked_empty is True
    assert result.down_timeout is False
    (up,) = seen
    pa = up.partial_aggregate
    assert pa.is_empty() and pa.num_examples == 0 and pa.weights == []
    assert pa.base_version == 0 and pa.rule == sup.aggregation.rule
    assert pa.update_form == sup.aggregation.update_form
    assert [line.device_id for line in up.round_close_report.lines] == DEVICES
    assert cluster.last_outcome == OUTCOME_EXPIRED        # nothing to average
    assert cluster.cluster_round == 2                    # no round closed
    assert sup._next_theta_version == 2                  # the θ the others moved on to


def test_an_empty_mission_whose_dock_goes_unanswered_restages_its_theta():
    sup, cluster, _ = _setup(serving=False, dock_on_empty=True, down_wait_s=0.3)
    result = sup.run_one_mission()
    # It docked: the empty partial was uploaded and the cluster holds it, so
    # the trace must not say otherwise (``mission_empty.docked``). Only the
    # DOWN is missing, which ``down_timeout`` records.
    assert result.empty is True and result.docked_empty is True
    assert result.down_timeout is True
    assert sup._next_theta_version == 0
    assert cluster.dock.recv_up(timeout=0.5).partial_aggregate.is_empty()


def test_without_dock_on_empty_an_empty_mission_does_not_dock():
    sup, cluster, _ = _setup(serving=False)
    result = sup.run_one_mission()
    assert result.empty is True and result.docked_empty is False
    assert result.down_timeout is False
    with pytest.raises(DockLinkError):
        cluster.dock.recv_up(timeout=0.2)


# --------------------------------------------------------------------------- #
# The mule process: events and exit status
# --------------------------------------------------------------------------- #

class _Events:
    def __init__(self):
        self.lines = []

    def emit(self, event, **fields):
        self.lines.append((event, fields))

    def close(self):
        return

    def named(self, event):
        return [f for e, f in self.lines if e == event]


@pytest.fixture
def mule_service():
    server = TCPDockLinkServer(host="127.0.0.1", port=0)
    server.start()
    made = []

    def _make(**cfg_kwargs):
        events = _Events()
        cfg = MuleConfig(mule_id="m-exit", dock_port=server.port, **cfg_kwargs)
        svc = mule_process.MuleService(cfg, events=events)
        svc.supervisor.wait_for_initial_dock = lambda timeout=None: True
        made.append(svc)
        return svc, events

    yield _make
    for svc in made:
        svc.shutdown()
    server.close()


def test_a_finished_run_exits_zero_and_reports_a_survived_wait(mule_service):
    svc, events = mule_service(n_missions=2, down_wait_s=5.0)
    results = iter([
        MissionRunResult(mission_round=1, down_timeout=True),
        MissionRunResult(mission_round=2),
    ])
    svc.supervisor.run_one_mission = lambda: next(results)
    svc.run()
    assert svc.exit_code == 0
    assert events.named("dock_down_timeout") == [{"mission_round": 1, "down_wait_s": 5.0}]
    assert len(events.named("mission_completed")) == 2
    (ready,) = events.named("mule_ready")
    assert ready["down_wait_s"] == 5.0 and ready["dock_on_empty"] is False


def test_a_mission_failure_ends_the_run_with_a_non_zero_status(mule_service):
    svc, events = mule_service(n_missions=3)

    def _boom():
        raise ClientClusterError("DOWN recv failed: client_recv_down timed out")

    svc.supervisor.run_one_mission = _boom
    svc.run()
    assert svc.exit_code == mule_process.EXIT_MISSION_FAILED
    (failed,) = events.named("mission_failed")
    assert failed["kind"] == "unexpected"
    assert events.named("dock_down_timeout") == []


def test_a_missing_bootstrap_ends_the_run_with_a_non_zero_status(mule_service):
    svc, events = mule_service(n_missions=1)
    svc._wait_with_stop = lambda fn, total_timeout: False
    svc.run()
    assert svc.exit_code == mule_process.EXIT_BOOTSTRAP_TIMEOUT
    assert len(events.named("dock_bootstrap_timeout")) == 1


def test_main_returns_the_services_exit_status(monkeypatch, tmp_path):
    cfg_path = tmp_path / "mule.json"
    cfg_path.write_text('{"mule_id": "m"}', encoding="utf-8")

    class _Svc:
        def __init__(self, cfg, events=None):
            self.exit_code = mule_process.EXIT_MISSION_FAILED
            self.actual_rf_port = 0

        def run(self):
            return

        def shutdown(self):
            return

        def request_stop(self):
            return

    monkeypatch.setattr(mule_process, "MuleService", _Svc)
    assert mule_process.main(["--config", str(cfg_path)]) == mule_process.EXIT_MISSION_FAILED
