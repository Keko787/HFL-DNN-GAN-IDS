"""FeRRy Phase 1 — versions, ages and the budgeted Pass 2 through a real mission.

In-process cluster + mule + devices on loopback links, two-pass missions:

* The DOWN bundle's version reaches the partial (``base_version``) and every
  push, and a device's next update comes back aged against it.
* Under ``agg:cutoff`` the cluster's new θ is the old θ plus the mule's
  merged update.
* A budgeted Pass 2 skips what does not fit; those devices get a SKIPPED line
  and keep their older basis, which is how ages come to spread.
"""

from __future__ import annotations

import threading
from typing import List

import numpy as np

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.mission import ClientMission, LocalTrainResult
from hermes.mission.aggregation_rules import AGG_CUTOFF, AggregationSpec
from hermes.mule import MuleSupervisor
from hermes.transport import LoopbackDockLink, LoopbackRFLink
from hermes.types import (
    ContactWaypoint,
    Bucket,
    DeliveryOutcome,
    DeviceID,
    FLState,
    MuleID,
    SpectrumSig,
)

MULE = MuleID("mule-p1")
DEVICE_IDS = [DeviceID(f"dev-{i:02d}") for i in range(4)]


def _train_factory(seed: int):
    rng = np.random.default_rng(seed)

    def _train(theta, synth):
        after = [w + rng.normal(0.0, 0.01, size=w.shape).astype(w.dtype) for w in theta]
        return LocalTrainResult(
            delta_theta=after, num_examples=int(rng.integers(4, 16)),
            accuracy=0.8, auc=0.8, loss=0.2, theta_after=after,
        )
    return _train


def _setup(positions, spec: AggregationSpec, **sup_kwargs):
    rf, dock = LoopbackRFLink(), LoopbackDockLink()
    devices: List[ClientMission] = []
    for i, did in enumerate(DEVICE_IDS):
        rf.register_device(did)
        cm = ClientMission(
            device_id=did, rf=rf, local_train=_train_factory(300 + i),
            solicit_timeout_s=1.5, disc_push_timeout_s=1.5,
        )
        cm.set_state(FLState.FL_OPEN)
        devices.append(cm)
    registry = DeviceRegistry()
    for did, pos in zip(DEVICE_IDS, positions):
        registry.register(
            device_id=did, position=pos,
            spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
        )
    registry.rebalance([MULE], round_counter=0)
    cluster = HFLHostCluster(
        registry=registry,
        generator=StubGeneratorHost(disc_weights=[
            np.zeros((4,), dtype=np.float32),
            np.ones((3, 3), dtype=np.float32) * 0.01,
        ]),
        dock=dock, synth_batch_size=2, aggregation=spec,
    )
    sup = MuleSupervisor(
        mule_id=MULE, rf=rf, dock=dock, session_ttl_s=1.5,
        rf_range_m=60.0, aggregation=spec, **sup_kwargs,
    )
    cluster.dock.send_down(cluster.dispatch_down_bundle(MULE))
    assert sup.wait_for_initial_dock(timeout=2.0)
    return sup, cluster, devices


def _serve_dock(cluster: HFLHostCluster) -> None:
    up = cluster.dock.recv_up(timeout=5.0)
    cluster.ingest_up_bundle(up)
    cluster.aggregate_pending()
    cluster.close_cluster_round()
    cluster.dock.send_down(cluster.dispatch_down_bundle(MULE))


def _run_mission(sup, cluster, devices):
    dock_t = threading.Thread(target=_serve_dock, args=(cluster,), daemon=True)
    dock_t.start()
    workers = []
    for cm in devices:
        def _loop(client=cm):
            for _ in range(2):            # one Pass-1 and one Pass-2 contact
                client.serve_once()
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        workers.append(t)
    result = sup.run_one_mission()
    dock_t.join(timeout=5.0)
    for t in workers:
        t.join(timeout=5.0)
    return result


_TIGHT = [(0.0, 0.0, 0.0), (10.0, 5.0, 0.0), (20.0, 0.0, 0.0), (15.0, 15.0, 0.0)]


def test_versions_flow_from_the_down_bundle_into_pushes_and_partials():
    spec = AggregationSpec(rule=AGG_CUTOFF)
    sup, cluster, devices = _setup(_TIGHT, spec)

    r1 = _run_mission(sup, cluster, devices)
    assert r1.aggregate is not None and r1.aggregate.base_version == 0
    assert set(r1.aggregate.device_ages) == {0}
    # Pass 2 delivered θ at version 1, and every device trained ahead on it.
    assert all(cm._prepared_basis_version == 1 for cm in devices)
    assert sup._next_theta_version == 1

    theta_1 = [w.copy() for w in cluster.generator.get_global_disc_weights()]
    r2 = _run_mission(sup, cluster, devices)
    agg = r2.aggregate
    assert agg.base_version == 1 and agg.rule == AGG_CUTOFF
    assert set(agg.device_basis_versions) == {1} and set(agg.device_ages) == {0}
    lines = [l for l in r2.report.lines if l.outcome.is_on_time()]
    assert lines and all(l.age == 0 and l.basis_version == 1 for l in lines)
    # θ_2 = θ_1 + Δ_m (one mule, η = 1)
    for new, old, d in zip(cluster.generator.get_global_disc_weights(), theta_1, agg.weights):
        np.testing.assert_allclose(new, old + d, rtol=1e-6, atol=1e-7)


def test_budgeted_pass_2_skips_what_does_not_fit_and_leaves_its_basis_old():
    # Two groups 200 m apart: at 5 m/s the far group costs ~40 s to reach.
    positions = [(0.0, 0.0, 0.0), (10.0, 5.0, 0.0), (200.0, 0.0, 0.0), (210.0, 15.0, 0.0)]
    spec = AggregationSpec(rule=AGG_CUTOFF)
    sup, cluster, devices = _setup(
        positions, spec, mission_budget_s=25.0, pass_2_budget=True,
    )
    r = _run_mission(sup, cluster, devices)
    skipped = {
        l.device_id for l in r.delivery_report.lines
        if l.outcome is DeliveryOutcome.SKIPPED
    }
    delivered = set(r.delivery_report.delivered())
    assert skipped == {DEVICE_IDS[2], DEVICE_IDS[3]}
    assert delivered == {DEVICE_IDS[0], DEVICE_IDS[1]}
    assert r.delivery_report.counts() == (2, 2)
    by_id = {cm.device_id: cm for cm in devices}
    assert all(by_id[d]._theta_basis_version == 1 for d in delivered)
    assert all(by_id[d]._theta_basis_version in (None, 0) for d in skipped)


def test_pass_2_budget_walk_keeps_order_and_skips_rather_than_stops():
    sup = MuleSupervisor(
        mule_id=MULE, rf=LoopbackRFLink(), dock=LoopbackDockLink(),
        rf_range_m=60.0, mission_budget_s=10.0, pass_2_budget=True,
    )
    wp = lambda x, d: ContactWaypoint(  # noqa: E731
        position=(x, 0.0, 0.0), devices=(DeviceID(d),),
        bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=0.0,
    )
    # From the origin at 5 m/s with 1 s sessions: a at 20 m costs 5 s (clock 5);
    # b at 60 m costs 9 s more (14 > 10, skipped); c at 30 m costs 3 s (8).
    queue = [wp(20.0, "a"), wp(60.0, "b"), wp(30.0, "c")]
    fly, skip = sup._budget_pass_2(queue)
    assert [w.devices[0] for w in fly] == ["a", "c"]
    assert [w.devices[0] for w in skip] == ["b"]

    no_budget = MuleSupervisor(
        mule_id=MULE, rf=LoopbackRFLink(), dock=LoopbackDockLink(),
        rf_range_m=60.0, pass_2_budget=True,
    )
    assert no_budget._budget_pass_2(queue) == (queue, [])
