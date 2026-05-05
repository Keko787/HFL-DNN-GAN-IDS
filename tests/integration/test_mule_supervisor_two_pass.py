"""Sprint 1.5 — :class:`MuleSupervisor` two-pass + per-contact integration.

Drives the supervisor with ``rf_range_m`` set, against an in-process
cluster + devices, to confirm:

* Pass 1 builds a contact queue and runs ``run_contact`` per contact
  (parallel exchange-only).
* Inter-pass dock fires (UP → cluster runs cross-mule FedAvg → DOWN).
* Pass 2 walks every slice contact greedily and pushes θ' via
  ``deliver_contact``; every device gets a DeliveryAck.
* The result carries non-empty ``pass_1_queue``, ``pass_2_queue``, and
  a populated ``MissionDeliveryReport``.
* Selector pass-gate (principle 13): no selector calls happen during
  Pass 2.

The legacy single-pass path is unchanged — see
``test_mule_supervisor.py``.
"""

from __future__ import annotations

import threading
import time
from typing import List

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.mission import ClientMission, LocalTrainResult
from hermes.mule import MuleSupervisor, MuleSupervisorError
from hermes.scheduler.selector import (
    SelectorScopeViolation,
    TargetSelectorRL,
)
from hermes.transport import LoopbackDockLink, LoopbackRFLink
from hermes.types import (
    ClusterAmendment,
    ContactWaypoint,
    DeliveryOutcome,
    DeviceID,
    FLState,
    MissionPass,
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
            accuracy=float(rng.uniform(0.7, 0.9)),
            auc=float(rng.uniform(0.7, 0.9)),
            loss=float(rng.uniform(0.1, 0.3)),
            theta_after=delta,
        )
    return _train


_DEFAULT_TIGHT_POSITIONS = [
    (0.0, 0.0, 0.0),
    (10.0, 5.0, 0.0),
    (20.0, 0.0, 0.0),
    (15.0, 15.0, 0.0),
]
# Two clearly-separated clusters so S3a yields ≥2 contact waypoints at
# rf_range_m=60. Used by the test that asserts the selector ran in Pass 1
# — design §2.7 short-circuits the selector on singleton buckets.
_TWO_CLUSTER_POSITIONS = [
    (0.0, 0.0, 0.0),
    (10.0, 5.0, 0.0),
    (200.0, 0.0, 0.0),
    (210.0, 15.0, 0.0),
]


def _make_devices(rf: LoopbackRFLink, positions=None) -> List[ClientMission]:
    """Build 4 devices at the given positions (defaults to a tight cluster).

    Pass ``_TWO_CLUSTER_POSITIONS`` when a test needs S3a to produce two
    contacts so the ≥2-candidate selector path actually fires.
    """
    if positions is None:
        positions = _DEFAULT_TIGHT_POSITIONS
    devices = []
    for i, (did, pos) in enumerate(zip(DEVICE_IDS, positions)):
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
    return devices, positions


def _make_cluster(dock: LoopbackDockLink, positions) -> HFLHostCluster:
    registry = DeviceRegistry()
    for did, pos in zip(DEVICE_IDS, positions):
        registry.register(
            device_id=did,
            position=pos,
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


def _drive_devices_for_pass_1_and_2(
    devices: List[ClientMission], n_per_device: int
) -> List[threading.Thread]:
    """Each device runs serve_once `n_per_device` times — once per Pass."""
    workers = []
    for cm in devices:
        def _loop(client=cm):
            for _ in range(n_per_device):
                client.serve_once()
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        workers.append(t)
    return workers


def _dispatch_initial_down(cluster: HFLHostCluster) -> None:
    down = cluster.dispatch_down_bundle(MULE)
    cluster.dock.send_down(down)


def _serve_dock_up_then_down(
    cluster: HFLHostCluster,
    *,
    amendment=None,
) -> None:
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
# Two-pass smoke
# --------------------------------------------------------------------------- #

def test_two_pass_mission_runs_pass_1_dock_pass_2():
    """End-to-end: Pass 1 → inter-pass dock → Pass 2 → delivery report."""
    rf = LoopbackRFLink()
    dock = LoopbackDockLink()

    devices, positions = _make_devices(rf)
    cluster = _make_cluster(dock, positions)

    sup = MuleSupervisor(
        mule_id=MULE, rf=rf, dock=dock, session_ttl_s=2.0,
        rf_range_m=60.0,  # tight cluster → 1 contact event
    )

    _dispatch_initial_down(cluster)
    assert sup.wait_for_initial_dock(timeout=2.0)

    # Cluster-side serves the inter-pass UP → DOWN cycle.
    cluster_t = threading.Thread(
        target=_serve_dock_up_then_down,
        args=(cluster,),
        daemon=True,
    )
    cluster_t.start()
    # Each device serves twice: once for Pass 1 (collect), once for Pass 2 (deliver).
    devs_t = _drive_devices_for_pass_1_and_2(devices, n_per_device=2)

    result = sup.run_one_mission()

    cluster_t.join(timeout=5.0)
    for t in devs_t:
        t.join(timeout=3.0)

    # Pass-1 queue should be one contact (all 4 devices in tight cluster).
    assert len(result.pass_1_queue) == 1
    assert len(result.pass_1_queue[0].devices) == 4

    # Pass-2 queue covers every slice member.
    pass_2_devices = set()
    for c in result.pass_2_queue:
        pass_2_devices.update(c.devices)
    assert pass_2_devices == set(DEVICE_IDS)

    # Delivery report should be present and have one row per device.
    assert result.delivery_report is not None
    delivered, undelivered = result.delivery_report.counts()
    assert delivered + undelivered == len(DEVICE_IDS)


# --------------------------------------------------------------------------- #
# Selector is bypassed in Pass 2
# --------------------------------------------------------------------------- #

class _CallCountingSelector(TargetSelectorRL):
    """Wraps TargetSelectorRL and counts each per-pass invocation."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.collect_calls = 0
        self.deliver_violations = 0

    def rank_contacts(self, candidates, device_states, env, **kwargs):
        # Forward whatever the scheduler passes (Sprint 1.5 M2 added
        # ``admitted=...``) — the mock only counts pass_kind separately.
        if kwargs.get("pass_kind", MissionPass.COLLECT) is MissionPass.COLLECT:
            self.collect_calls += 1
        else:
            self.deliver_violations += 1
        return super().rank_contacts(
            candidates, device_states, env, **kwargs,
        )


def test_pass_2_does_not_invoke_selector():
    """Principle 13 — selector is Pass-1-only.

    Uses ``_TWO_CLUSTER_POSITIONS`` so S3a yields two contacts and the
    Pass-1 bucket has ≥2 candidates — design §2.7 short-circuits the
    selector on singletons, which would mask the Pass-1 invocation
    we need to observe here.
    """
    rf = LoopbackRFLink()
    dock = LoopbackDockLink()
    devices, positions = _make_devices(rf, positions=_TWO_CLUSTER_POSITIONS)
    cluster = _make_cluster(dock, positions)

    selector = _CallCountingSelector(rng_seed=0)
    sup = MuleSupervisor(
        mule_id=MULE, rf=rf, dock=dock, session_ttl_s=2.0,
        rf_range_m=60.0, target_selector=selector,
    )

    _dispatch_initial_down(cluster)
    assert sup.wait_for_initial_dock(timeout=2.0)

    cluster_t = threading.Thread(
        target=_serve_dock_up_then_down, args=(cluster,), daemon=True,
    )
    cluster_t.start()
    devs_t = _drive_devices_for_pass_1_and_2(devices, n_per_device=2)

    sup.run_one_mission()

    cluster_t.join(timeout=5.0)
    for t in devs_t:
        t.join(timeout=3.0)

    # Selector ranked at least once during Pass 1 (two contact events).
    assert selector.collect_calls >= 1
    # No DELIVER call landed (and if any did, the parent class would have
    # raised SelectorScopeViolation, killing the test before this assert).
    assert selector.deliver_violations == 0


def test_direct_select_target_in_pass_2_raises():
    """Belt-and-suspenders: the API itself rejects pass=DELIVER."""
    selector = TargetSelectorRL(rng_seed=0)
    from hermes.scheduler.selector import SelectorEnv
    env = SelectorEnv(
        mule_pose=(0.0, 0.0, 0.0),
        mule_energy=1.0,
        rf_prior_snr_db=20.0,
        now=0.0,
    )
    with pytest.raises(SelectorScopeViolation):
        selector.rank_contacts(
            candidates=[],  # empty even — guard fires before the empty check
            device_states={},
            env=env,
            pass_kind=MissionPass.DELIVER,
        )


# --------------------------------------------------------------------------- #
# Single-pass legacy still works (rf_range_m=None)
# --------------------------------------------------------------------------- #

def test_legacy_single_pass_unchanged():
    """rf_range_m=None → existing Sprint-1A path runs as before."""
    rf = LoopbackRFLink()
    dock = LoopbackDockLink()
    devices, positions = _make_devices(rf)
    cluster = _make_cluster(dock, positions)

    sup = MuleSupervisor(
        mule_id=MULE, rf=rf, dock=dock, session_ttl_s=2.0,
        # rf_range_m omitted → legacy mode
    )

    _dispatch_initial_down(cluster)
    assert sup.wait_for_initial_dock(timeout=2.0)

    cluster_t = threading.Thread(
        target=_serve_dock_up_then_down, args=(cluster,), daemon=True,
    )
    cluster_t.start()
    devs_t = _drive_devices_for_pass_1_and_2(devices, n_per_device=1)

    result = sup.run_one_mission()

    cluster_t.join(timeout=5.0)
    for t in devs_t:
        t.join(timeout=3.0)

    # Legacy path populates `queue` (per-device), not pass_1/pass_2 queues.
    assert len(result.queue) == len(DEVICE_IDS)
    assert result.pass_1_queue == []
    assert result.pass_2_queue == []
    assert result.delivery_report is None
