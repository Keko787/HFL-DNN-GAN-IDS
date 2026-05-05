"""Phase 6 / Sprint 1 — :class:`MuleSupervisor` end-to-end integration.

Drives the supervisor against an in-process cluster + devices to confirm:

* Initial dock bootstraps the model + slice into the supervisor.
* :meth:`MuleSupervisor.run_one_mission` produces a non-empty queue,
  closes a round, and dock-cycles UP+DOWN atomically.
* Round 2 consumes the cluster amendment from round 1's dock.
* Optional :class:`TargetSelectorRL` and :class:`ChannelDDQN` plug in
  cleanly via constructor kwargs.

This is not a behavioral A/B — that lives in the selector A/B test.
This is the wiring test: are the four mule-side programs talking to
each other in the right order, with the right payloads.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.l1.channel_ddqn import ChannelDDQN
from hermes.mission import ClientMission, LocalTrainResult
from hermes.mule import MuleSupervisor, MuleSupervisorError
from hermes.scheduler.selector import TargetSelectorRL
from hermes.transport import LoopbackDockLink, LoopbackRFLink
from hermes.types import (
    ClusterAmendment,
    DeviceID,
    FLState,
    MuleID,
    SpectrumSig,
)


MULE = MuleID("mule-test")
DEVICE_IDS = [DeviceID(f"dev-{i:02d}") for i in range(4)]


def _train_factory(seed: int):
    rng = np.random.default_rng(seed)

    def _train(theta, synth):
        noise = [rng.normal(0.0, 0.01, size=w.shape).astype(w.dtype) for w in theta]
        delta = [w + n for w, n in zip(theta, noise)]
        return LocalTrainResult(
            delta_theta=delta,
            num_examples=int(rng.integers(4, 16)),
            accuracy=float(rng.uniform(0.75, 0.95)),
            auc=float(rng.uniform(0.75, 0.95)),
            loss=float(rng.uniform(0.1, 0.3)),
            theta_after=delta,
        )
    return _train


def _make_devices(rf: LoopbackRFLink) -> List[ClientMission]:
    devices: List[ClientMission] = []
    for i, did in enumerate(DEVICE_IDS):
        rf.register_device(did)
        cm = ClientMission(
            device_id=did,
            rf=rf,
            local_train=_train_factory(seed=200 + i),
            solicit_timeout_s=2.0,
            disc_push_timeout_s=2.0,
        )
        cm.set_state(FLState.FL_OPEN)
        devices.append(cm)
    return devices


def _make_cluster(dock: LoopbackDockLink) -> HFLHostCluster:
    registry = DeviceRegistry()
    for did in DEVICE_IDS:
        registry.register(
            device_id=did,
            position=(float(int(did[-2:]) * 10), 0.0, 0.0),
            spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
        )
    registry.rebalance([MULE], round_counter=0)
    generator = StubGeneratorHost(
        disc_weights=[
            np.zeros((4,), dtype=np.float32),
            np.ones((3, 3), dtype=np.float32) * 0.01,
        ]
    )
    return HFLHostCluster(
        registry=registry,
        generator=generator,
        dock=dock,
        synth_batch_size=4,
    )


def _drive_devices(devices: List[ClientMission], n_sessions: int) -> List[threading.Thread]:
    """One worker thread per device serving up to one solicit each.

    The mule visits N times per mission round; we spin N threads, one
    per device-side ``serve_once`` call. Threads die on RF timeout.
    """
    workers: List[threading.Thread] = []
    for cm in devices:
        t = threading.Thread(target=cm.serve_once, daemon=True)
        t.start()
        workers.append(t)
    return workers


def _dispatch_initial_down(cluster: HFLHostCluster) -> None:
    """Cluster pre-dispatches a DOWN bundle before the supervisor docks."""
    down = cluster.dispatch_down_bundle(MULE)
    cluster.dock.send_down(down)


def _serve_dock_up_then_down(
    cluster: HFLHostCluster,
    *,
    amendment: Optional[ClusterAmendment] = None,
) -> None:
    """Cluster-side: receive UP, run cross-mule FedAvg, dispatch DOWN."""
    up = cluster.dock.recv_up(timeout=2.0)
    cluster.ingest_up_bundle(up)
    cluster.aggregate_pending()
    cluster.close_cluster_round(
        deadline_overrides=(amendment.deadline_overrides if amendment else None),
        notes=(amendment.notes if amendment else ""),
    )
    down = cluster.dispatch_down_bundle(MULE)
    cluster.dock.send_down(down)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_supervisor_bootstrap_then_one_mission():
    """Initial dock -> mission -> dock UP/DOWN round-trip succeeds."""
    rf = LoopbackRFLink()
    dock = LoopbackDockLink()

    cluster = _make_cluster(dock)
    devices = _make_devices(rf)

    sup = MuleSupervisor(mule_id=MULE, rf=rf, dock=dock, session_ttl_s=2.0)

    _dispatch_initial_down(cluster)
    assert sup.wait_for_initial_dock(timeout=2.0)

    # Cluster-side serves the post-mission UP+DOWN in a worker.
    cluster_t = threading.Thread(
        target=_serve_dock_up_then_down,
        args=(cluster,),
        kwargs={"amendment": None},
        daemon=True,
    )
    cluster_t.start()
    devs_t = _drive_devices(devices, n_sessions=len(DEVICE_IDS))

    result = sup.run_one_mission()

    cluster_t.join(timeout=5.0)
    for t in devs_t:
        t.join(timeout=2.0)

    assert result.mission_round == 1
    assert len(result.queue) == len(DEVICE_IDS), (
        f"expected scheduler to enqueue all {len(DEVICE_IDS)} slice members, "
        f"got {len(result.queue)}"
    )
    assert result.aggregate is not None
    assert result.report is not None
    assert result.contacts is not None
    on_time, missed = result.report.counts()
    assert on_time + missed >= 1, "no sessions completed"


def test_supervisor_two_missions_consume_amendment():
    """Round-2 dock-down carries the cluster amendment from round 1."""
    rf = LoopbackRFLink()
    dock = LoopbackDockLink()

    cluster = _make_cluster(dock)
    devices = _make_devices(rf)

    sup = MuleSupervisor(mule_id=MULE, rf=rf, dock=dock, session_ttl_s=2.0)

    _dispatch_initial_down(cluster)
    assert sup.wait_for_initial_dock(timeout=2.0)

    # ---- Round 1 ------------------------------------------------------
    amend = ClusterAmendment(
        cluster_round=1,
        deadline_overrides={DEVICE_IDS[0]: time.time() + 99.0},
        notes="r1 amendment",
    )
    cluster_t = threading.Thread(
        target=_serve_dock_up_then_down,
        args=(cluster,),
        kwargs={"amendment": amend},
        daemon=True,
    )
    cluster_t.start()
    devs_t = _drive_devices(devices, n_sessions=len(DEVICE_IDS))

    r1 = sup.run_one_mission()

    cluster_t.join(timeout=5.0)
    for t in devs_t:
        t.join(timeout=2.0)

    # The slow-phase amendment from round 1 should have been folded into
    # the scheduler's view of dev-00 by the post-round dock cycle.
    sched_state = sup.scheduler.get_state(DEVICE_IDS[0])
    assert sched_state is not None
    assert sched_state.deadline_override_ts is not None, (
        "round 1 amendment did not reach the scheduler via DOWN distribution"
    )

    # ---- Round 2 ------------------------------------------------------
    # Re-set devices to FL_OPEN so they answer again.
    for cm in devices:
        cm.set_state(FLState.FL_OPEN)

    cluster_t2 = threading.Thread(
        target=_serve_dock_up_then_down,
        args=(cluster,),
        kwargs={"amendment": None},
        daemon=True,
    )
    cluster_t2.start()
    devs_t2 = _drive_devices(devices, n_sessions=len(DEVICE_IDS))

    r2 = sup.run_one_mission()

    cluster_t2.join(timeout=5.0)
    for t in devs_t2:
        t.join(timeout=2.0)

    assert r2.mission_round == r1.mission_round + 1


def test_supervisor_run_one_mission_without_dock_raises():
    """Calling run_one_mission before a DOWN was distributed is a programming error."""
    sup = MuleSupervisor(
        mule_id=MULE, rf=LoopbackRFLink(), dock=LoopbackDockLink(),
        session_ttl_s=0.1,
    )
    with pytest.raises(MuleSupervisorError):
        sup.run_one_mission()


def test_supervisor_with_smart_selector_and_l1():
    """Pluggable selector + L1 actor inject without breaking the loop."""
    rf = LoopbackRFLink()
    dock = LoopbackDockLink()

    cluster = _make_cluster(dock)
    devices = _make_devices(rf)

    selector = TargetSelectorRL(rng_seed=0)
    l1 = ChannelDDQN(seed=0)

    sup = MuleSupervisor(
        mule_id=MULE, rf=rf, dock=dock,
        target_selector=selector,
        channel_actor=l1,
        session_ttl_s=2.0,
    )

    _dispatch_initial_down(cluster)
    assert sup.wait_for_initial_dock(timeout=2.0)

    cluster_t = threading.Thread(
        target=_serve_dock_up_then_down,
        args=(cluster,),
        daemon=True,
    )
    cluster_t.start()
    devs_t = _drive_devices(devices, n_sessions=len(DEVICE_IDS))

    result = sup.run_one_mission()

    cluster_t.join(timeout=5.0)
    for t in devs_t:
        t.join(timeout=2.0)

    # L1 should have produced a channel choice per visited device.
    assert len(result.channel_choices) == len(result.queue)
    # Every channel choice is a valid band index (0, 1, or 2).
    assert all(0 <= ch < 3 for ch in result.channel_choices)
