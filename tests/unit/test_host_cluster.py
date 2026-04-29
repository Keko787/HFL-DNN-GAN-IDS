"""Integration-style tests — HFLHostCluster end-to-end at one server.

This is the Phase 1 DoD demo target as a test:
"Two fake mules dock one after the other, upload disjoint partials, and
receive disjoint slices."
"""

from __future__ import annotations

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.transport import LoopbackDockLink
from hermes.types import (
    ContactHistory,
    DeviceID,
    MissionOutcome,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MuleID,
    PartialAggregate,
    SpectrumSig,
    UpBundle,
)


def _seed_registry(reg: DeviceRegistry, n: int) -> list[DeviceID]:
    sig = SpectrumSig(bands=(0, 1, 2), last_good_snr_per_band=(10.0, 12.0, 8.0))
    out = []
    for i in range(n):
        d = DeviceID(f"d{i:03d}")
        reg.register(d, (0.0, 0.0, 0.0), sig)
        out.append(d)
    return out


def _make_cluster(reg: DeviceRegistry, dock=None, *, min_part: int = 1) -> HFLHostCluster:
    gen = StubGeneratorHost(disc_weights=[np.zeros(4, dtype=np.float32)])
    return HFLHostCluster(
        registry=reg,
        generator=gen,
        dock=dock or LoopbackDockLink(),
        synth_batch_size=2,
        min_participation=min_part,
    )


def _fake_up_for(mule: MuleID, devs: list[DeviceID]) -> UpBundle:
    pa = PartialAggregate(
        mule_id=mule,
        mission_round=1,
        weights=[np.ones(4, dtype=np.float32)],
        num_examples=len(devs) * 5,
        contributing_devices=tuple(devs),
    )
    rep = MissionRoundCloseReport(
        mule_id=mule,
        mission_round=1,
        started_at=0.0,
        finished_at=10.0,
        lines=[
            MissionRoundCloseLine(
                device_id=d, outcome=MissionOutcome.CLEAN, contact_ts=1.0
            )
            for d in devs
        ],
    )
    ch = ContactHistory(mule_id=mule, mission_round=1, records=[])
    return UpBundle(
        mule_id=mule,
        partial_aggregate=pa,
        round_close_report=rep,
        contact_history=ch,
    )


def test_two_mules_get_disjoint_slices():
    reg = DeviceRegistry()
    _seed_registry(reg, 8)
    cluster = _make_cluster(reg)

    slices = cluster.rebalance_for([MuleID("m1"), MuleID("m2")])
    s1 = set(slices[MuleID("m1")].device_ids)
    s2 = set(slices[MuleID("m2")].device_ids)

    assert s1.isdisjoint(s2)
    assert s1 | s2 == set(d.device_id for d in reg.all())


def test_dispatch_down_bundle_carries_slice_and_synth():
    reg = DeviceRegistry()
    _seed_registry(reg, 4)
    cluster = _make_cluster(reg)
    cluster.rebalance_for([MuleID("m1")])

    bundle = cluster.dispatch_down_bundle(MuleID("m1"))
    assert bundle.mule_id == MuleID("m1")
    assert len(bundle.mission_slice.device_ids) == 4
    assert len(bundle.synth_batch) == 2  # synth_batch_size in _make_cluster


def test_ingest_up_bundle_updates_registry_counters():
    reg = DeviceRegistry()
    devs = _seed_registry(reg, 3)
    cluster = _make_cluster(reg)
    cluster.rebalance_for([MuleID("m1")])

    up = _fake_up_for(MuleID("m1"), devs)
    cluster.ingest_up_bundle(up)

    for d in devs:
        rec = reg.get(d)
        assert rec.on_time_history == 1
        assert rec.is_new is False


def test_aggregate_pending_runs_fedavg_and_pushes_to_generator():
    reg = DeviceRegistry()
    devs = _seed_registry(reg, 4)
    cluster = _make_cluster(reg)
    cluster.rebalance_for([MuleID("m1"), MuleID("m2")])

    cluster.ingest_up_bundle(_fake_up_for(MuleID("m1"), devs[:2]))
    cluster.ingest_up_bundle(_fake_up_for(MuleID("m2"), devs[2:]))

    merged = cluster.aggregate_pending()
    assert merged is not None
    # both partials are all-ones, so the average is ones too
    np.testing.assert_allclose(merged[0], np.ones(4, dtype=np.float32))
    # generator must hold the new global
    assert cluster.generator.disc_weights[0].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_min_participation_threshold_blocks_aggregation():
    reg = DeviceRegistry()
    devs = _seed_registry(reg, 2)
    cluster = _make_cluster(reg, min_part=2)
    cluster.rebalance_for([MuleID("m1"), MuleID("m2")])

    cluster.ingest_up_bundle(_fake_up_for(MuleID("m1"), devs))
    assert cluster.aggregate_pending() is None  # only one partial; threshold=2


def test_close_cluster_round_emits_amendment_and_resets():
    reg = DeviceRegistry()
    devs = _seed_registry(reg, 2)
    cluster = _make_cluster(reg)
    cluster.rebalance_for([MuleID("m1")])

    cluster.ingest_up_bundle(_fake_up_for(MuleID("m1"), devs))
    assert cluster.pending_partials() == 1

    amend = cluster.close_cluster_round(notes="end of round 1")
    assert amend.cluster_round == 1
    assert cluster.pending_partials() == 0

    # next dispatch carries that amendment
    bundle = cluster.dispatch_down_bundle(MuleID("m1"))
    assert bundle.cluster_amendments.notes == "end of round 1"


def test_serve_one_dock_round_trip_via_loopback():
    reg = DeviceRegistry()
    devs = _seed_registry(reg, 2)
    dock = LoopbackDockLink()
    cluster = _make_cluster(reg, dock=dock)
    cluster.rebalance_for([MuleID("m1")])

    # mule pushes UP, then waits for DOWN
    up = _fake_up_for(MuleID("m1"), devs)
    dock.client_send_up(up)

    cluster.serve_one_dock(timeout=1.0)

    down = dock.client_recv_down(MuleID("m1"), timeout=1.0)
    assert down.mule_id == MuleID("m1")
    assert set(down.mission_slice.device_ids) == set(devs)


def test_duplicate_up_in_same_round_is_ignored():
    reg = DeviceRegistry()
    devs = _seed_registry(reg, 2)
    cluster = _make_cluster(reg)
    cluster.rebalance_for([MuleID("m1")])

    up = _fake_up_for(MuleID("m1"), devs)
    cluster.ingest_up_bundle(up)
    cluster.ingest_up_bundle(up)  # duplicate
    assert cluster.pending_partials() == 1
