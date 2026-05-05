"""Sprint 1.5 — MissionDeliveryReport carryover into the cluster registry.

Pins down the Chunk F path: an UpBundle that carries a Pass-2 delivery
report from the previous mission causes:

* ``DeviceRecord.delivery_priority`` to bump on each UNDELIVERED row.
* ``DeviceRecord.delivery_priority`` to reset to 0 on each DELIVERED row.
* ``HFLHostCluster.pending_undelivered_carryover()`` to expose the
  per-round metric.
* On the next mule's slice, the bumped device gets pulled toward a
  cluster anchor by S3a (priority tie-breaker).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.scheduler import FLScheduler
from hermes.transport import LoopbackDockLink
from hermes.types import (
    Bucket,
    ContactHistory,
    DeliveryOutcome,
    DeviceID,
    MissionDeliveryLine,
    MissionDeliveryReport,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MissionOutcome,
    MissionSlice,
    MuleID,
    PartialAggregate,
    SpectrumSig,
    UpBundle,
)


MULE = MuleID("mule-test")


def _make_cluster() -> HFLHostCluster:
    registry = DeviceRegistry()
    for i, pos in enumerate([(0.0, 0.0, 0.0), (50.0, 0.0, 0.0), (1000.0, 0.0, 0.0)]):
        registry.register(
            device_id=DeviceID(f"d{i}"),
            position=pos,
            spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
        )
    registry.rebalance([MULE], round_counter=0)
    return HFLHostCluster(
        registry=registry,
        generator=StubGeneratorHost(disc_weights=[
            np.zeros((4,), dtype=np.float32),
        ]),
        dock=LoopbackDockLink(),
        synth_batch_size=1,
        min_participation=1,
    )


def _aggregate(mule_id: MuleID = MULE) -> PartialAggregate:
    return PartialAggregate(
        mule_id=mule_id,
        mission_round=1,
        weights=[np.zeros((4,), dtype=np.float32)],
        num_examples=1,
    )


def _round_report() -> MissionRoundCloseReport:
    rep = MissionRoundCloseReport(
        mule_id=MULE, mission_round=1, started_at=0.0, finished_at=1.0,
    )
    return rep


def _delivery_report(undelivered_ids: List[str], delivered_ids: List[str]) -> MissionDeliveryReport:
    rep = MissionDeliveryReport(
        mule_id=MULE, mission_round=1, started_at=0.0, finished_at=1.0,
    )
    for did in undelivered_ids:
        rep.append(
            MissionDeliveryLine(
                device_id=DeviceID(did),
                outcome=DeliveryOutcome.UNDELIVERED,
                contact_ts=1.0,
            )
        )
    for did in delivered_ids:
        rep.append(
            MissionDeliveryLine(
                device_id=DeviceID(did),
                outcome=DeliveryOutcome.DELIVERED,
                contact_ts=1.0,
            )
        )
    return rep


# --------------------------------------------------------------------------- #
# Direct registry path
# --------------------------------------------------------------------------- #

def test_update_after_delivery_bumps_on_undelivered():
    cluster = _make_cluster()
    cluster.registry.update_after_delivery(DeviceID("d0"), delivered=False)
    cluster.registry.update_after_delivery(DeviceID("d0"), delivered=False)
    rec = cluster.registry.get(DeviceID("d0"))
    assert rec is not None
    assert rec.delivery_priority == 2


def test_update_after_delivery_resets_on_delivered():
    cluster = _make_cluster()
    # Bump twice, then a clean delivery resets to 0.
    cluster.registry.update_after_delivery(DeviceID("d0"), delivered=False)
    cluster.registry.update_after_delivery(DeviceID("d0"), delivered=False)
    cluster.registry.update_after_delivery(DeviceID("d0"), delivered=True)
    rec = cluster.registry.get(DeviceID("d0"))
    assert rec is not None
    assert rec.delivery_priority == 0


def test_update_after_delivery_unknown_device_silent():
    cluster = _make_cluster()
    # No-op; no exception.
    cluster.registry.update_after_delivery(DeviceID("ghost"), delivered=False)


# --------------------------------------------------------------------------- #
# UpBundle ingest path
# --------------------------------------------------------------------------- #

def test_ingest_up_bundle_with_delivery_report_bumps_priority():
    cluster = _make_cluster()

    bundle = UpBundle(
        mule_id=MULE,
        partial_aggregate=_aggregate(),
        round_close_report=_round_report(),
        contact_history=ContactHistory(mule_id=MULE, mission_round=1),
        prev_mission_delivery_report=_delivery_report(
            undelivered_ids=["d0", "d2"],
            delivered_ids=["d1"],
        ),
    )

    cluster.ingest_up_bundle(bundle)

    assert cluster.registry.get(DeviceID("d0")).delivery_priority == 1
    assert cluster.registry.get(DeviceID("d1")).delivery_priority == 0
    assert cluster.registry.get(DeviceID("d2")).delivery_priority == 1
    # Per-round metric reflects the count of undelivered rows in this UP.
    assert cluster.pending_undelivered_carryover() == 2


def test_ingest_up_bundle_without_delivery_report_unchanged():
    cluster = _make_cluster()

    bundle = UpBundle(
        mule_id=MULE,
        partial_aggregate=_aggregate(),
        round_close_report=_round_report(),
        contact_history=ContactHistory(mule_id=MULE, mission_round=1),
        # prev_mission_delivery_report omitted → None
    )

    cluster.ingest_up_bundle(bundle)

    for did in ["d0", "d1", "d2"]:
        assert cluster.registry.get(DeviceID(did)).delivery_priority == 0
    assert cluster.pending_undelivered_carryover() == 0


def test_carryover_resets_per_cluster_round():
    cluster = _make_cluster()

    cluster.ingest_up_bundle(
        UpBundle(
            mule_id=MULE,
            partial_aggregate=_aggregate(),
            round_close_report=_round_report(),
            contact_history=ContactHistory(mule_id=MULE, mission_round=1),
            prev_mission_delivery_report=_delivery_report(
                undelivered_ids=["d0"], delivered_ids=[],
            ),
        )
    )
    assert cluster.pending_undelivered_carryover() == 1

    # Close the round → carryover resets for the next round.
    cluster.aggregate_pending()
    cluster.close_cluster_round()
    assert cluster.pending_undelivered_carryover() == 0


# --------------------------------------------------------------------------- #
# Tie-breaker effect on S3a clustering — happens MULE-SIDE
# --------------------------------------------------------------------------- #

def test_undelivered_carryover_routes_priority_in_clustering():
    """Mule-side: a high-delivery_priority device wins the anchor pick.

    Sets up two isolated devices (separated by > rf_range), one with
    delivery_priority=0 and one with delivery_priority=5. The greedy S3a
    algorithm picks the high-priority one as the anchor of the FIRST
    contact, which means it gets reached earliest in Pass 1. The second
    contact then covers the low-priority device.
    """
    from hermes.scheduler.stages.s3a_cluster import cluster_by_rf_range
    from hermes.types import DeviceSchedulerState

    states = {
        DeviceID("low"): DeviceSchedulerState(
            device_id=DeviceID("low"),
            last_known_position=(0.0, 0.0, 0.0),
            is_in_slice=True,
            is_new=False,
            delivery_priority=0,
        ),
        DeviceID("high"): DeviceSchedulerState(
            device_id=DeviceID("high"),
            last_known_position=(500.0, 0.0, 0.0),
            is_in_slice=True,
            is_new=False,
            delivery_priority=5,
        ),
    }
    for s in states.values():
        s.bucket = Bucket.SCHEDULED_THIS_ROUND
    deadlines = {DeviceID("low"): 100.0, DeviceID("high"): 100.0}

    contacts = cluster_by_rf_range(
        eligible_device_ids=list(states.keys()),
        device_states=states,
        deadlines=deadlines,
        rf_range_m=60.0,
    )

    assert len(contacts) == 2
    # The high-priority device is the anchor of the first formed contact.
    assert contacts[0].devices == (DeviceID("high"),)
    assert contacts[1].devices == (DeviceID("low"),)
