"""Sprint 1.5 — FLScheduler contact-aware queue integration.

Pins down the Pass-1 ``build_contact_queue`` and Pass-2
``build_pass_2_queue`` paths:

* Pass-1 queue runs S1 → S3 → S3a clustering → bucket-prioritised
  contact ordering. Selector is consulted (when wired) only for
  COLLECT-mode ranking inside each bucket.
* Pass-2 queue covers every slice member regardless of S1 eligibility,
  ordered nearest-first from the post-Pass-1 mule pose.
"""

from __future__ import annotations

from typing import List

import pytest

from hermes.scheduler import FLScheduler
from hermes.scheduler.selector import TargetSelectorRL
from hermes.types import (
    Bucket,
    BUCKET_PRIORITY,
    ContactWaypoint,
    DeviceID,
    DeviceRecord,
    MissionSlice,
    MuleID,
    SpectrumSig,
)


MULE = MuleID("mule-test")
NOW = 1000.0


def _record(did: str, pos=(0.0, 0.0, 0.0), *, is_new: bool = True) -> DeviceRecord:
    return DeviceRecord(
        device_id=DeviceID(did),
        spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
        is_new=is_new,
        last_known_position=pos,
    )


def _slice(*device_ids: str) -> MissionSlice:
    return MissionSlice(
        mule_id=MULE,
        device_ids=tuple(DeviceID(d) for d in device_ids),
        issued_round=1,
        issued_at=NOW,
    )


# --------------------------------------------------------------------------- #
# build_contact_queue
# --------------------------------------------------------------------------- #

def test_build_contact_queue_empty_for_no_eligible():
    fs = FLScheduler(now_fn=lambda: NOW)
    q = fs.build_contact_queue(rf_range_m=60.0, mule_pose=(0.0, 0.0, 0.0))
    assert q == []


def test_build_contact_queue_invalid_rf_range_raises():
    fs = FLScheduler(now_fn=lambda: NOW)
    with pytest.raises(Exception, match="rf_range_m > 0"):
        fs.build_contact_queue(rf_range_m=0.0, mule_pose=(0.0, 0.0, 0.0))


def test_build_contact_queue_clusters_in_range_devices():
    """Three devices in a tight cluster → one ContactWaypoint covers all."""
    fs = FLScheduler(now_fn=lambda: NOW)
    records = [
        _record("d0", pos=(0.0, 0.0, 0.0)),
        _record("d1", pos=(20.0, 0.0, 0.0)),
        _record("d2", pos=(40.0, 0.0, 0.0)),
    ]
    fs.ingest_slice(_slice("d0", "d1", "d2"), registry_records=records)

    contacts = fs.build_contact_queue(rf_range_m=60.0, mule_pose=(0.0, 0.0, 0.0))
    assert len(contacts) == 1
    assert set(contacts[0].devices) == {DeviceID("d0"), DeviceID("d1"), DeviceID("d2")}
    assert contacts[0].bucket is Bucket.NEW  # all is_new=True → inherits NEW


def test_build_contact_queue_separates_far_clusters():
    """Devices > rf_range apart split into separate contacts."""
    fs = FLScheduler(now_fn=lambda: NOW)
    records = [
        _record("a0", pos=(0.0, 0.0, 0.0)),
        _record("a1", pos=(20.0, 0.0, 0.0)),
        _record("b0", pos=(500.0, 0.0, 0.0)),
        _record("b1", pos=(515.0, 0.0, 0.0)),
    ]
    fs.ingest_slice(_slice("a0", "a1", "b0", "b1"), registry_records=records)

    contacts = fs.build_contact_queue(rf_range_m=60.0, mule_pose=(0.0, 0.0, 0.0))
    assert len(contacts) == 2

    # Each cluster covers exactly its members.
    members = sorted([sorted(c.devices) for c in contacts])
    assert members == [
        sorted([DeviceID("a0"), DeviceID("a1")]),
        sorted([DeviceID("b0"), DeviceID("b1")]),
    ]


def test_build_contact_queue_walks_bucket_priority():
    """Mixed-bucket slice — NEW contacts must come before SCHEDULED contacts."""
    fs = FLScheduler(now_fn=lambda: NOW)
    # Two clusters: cluster A is all-NEW (is_new=True), cluster B is all-SCHEDULED.
    records = [
        _record("new0", pos=(0.0, 0.0, 0.0), is_new=True),
        _record("new1", pos=(20.0, 0.0, 0.0), is_new=True),
        _record("sch0", pos=(500.0, 0.0, 0.0), is_new=False),
        _record("sch1", pos=(515.0, 0.0, 0.0), is_new=False),
    ]
    fs.ingest_slice(_slice("new0", "new1", "sch0", "sch1"), registry_records=records)

    contacts = fs.build_contact_queue(rf_range_m=60.0, mule_pose=(250.0, 0.0, 0.0))

    assert len(contacts) == 2
    # NEW bucket comes first regardless of distance.
    assert contacts[0].bucket is Bucket.NEW
    assert contacts[1].bucket is Bucket.SCHEDULED_THIS_ROUND


def test_build_contact_queue_with_selector_calls_rank_contacts():
    """When a TargetSelectorRL is wired, it ranks within each bucket."""
    selector = TargetSelectorRL(rng_seed=0)
    fs = FLScheduler(now_fn=lambda: NOW, target_selector=selector)
    records = [
        _record("d0", pos=(0.0, 0.0, 0.0)),
        _record("d1", pos=(200.0, 0.0, 0.0)),
        _record("d2", pos=(400.0, 0.0, 0.0)),
    ]
    fs.ingest_slice(_slice("d0", "d1", "d2"), registry_records=records)

    # rf_range_m=60 → each device is its own contact.
    contacts = fs.build_contact_queue(rf_range_m=60.0, mule_pose=(0.0, 0.0, 0.0))
    assert len(contacts) == 3
    # Selector argmax over an untrained DDQN with seed=0 is deterministic;
    # we don't assert order — just that all three are present.
    seen = set()
    for c in contacts:
        seen.update(c.devices)
    assert seen == {DeviceID("d0"), DeviceID("d1"), DeviceID("d2")}


# --------------------------------------------------------------------------- #
# build_pass_2_queue
# --------------------------------------------------------------------------- #

def test_build_pass_2_queue_covers_every_slice_member():
    """Pass 2 visits ALL slice members regardless of bucket / eligibility."""
    fs = FLScheduler(now_fn=lambda: NOW)
    records = [
        _record("d0", pos=(0.0, 0.0, 0.0)),
        _record("d1", pos=(20.0, 0.0, 0.0)),
        _record("d2", pos=(500.0, 0.0, 0.0)),
    ]
    fs.ingest_slice(_slice("d0", "d1", "d2"), registry_records=records)

    contacts = fs.build_pass_2_queue(rf_range_m=60.0, mule_pose=(0.0, 0.0, 0.0))

    seen = set()
    for c in contacts:
        seen.update(c.devices)
    assert seen == {DeviceID("d0"), DeviceID("d1"), DeviceID("d2")}


def test_build_pass_2_queue_orders_nearest_first():
    """Three isolated devices → Pass-2 visits them in nearest-first order."""
    fs = FLScheduler(now_fn=lambda: NOW)
    records = [
        _record("far", pos=(200.0, 0.0, 0.0)),
        _record("near", pos=(20.0, 0.0, 0.0)),
        _record("mid", pos=(100.0, 0.0, 0.0)),
    ]
    fs.ingest_slice(_slice("far", "near", "mid"), registry_records=records)

    # Each device > 60 m from the others → 3 separate contacts.
    contacts = fs.build_pass_2_queue(rf_range_m=60.0, mule_pose=(0.0, 0.0, 0.0))
    assert len(contacts) == 3

    visit_order = [c.devices[0] for c in contacts]
    assert visit_order == [DeviceID("near"), DeviceID("mid"), DeviceID("far")]


def test_build_pass_2_queue_starts_from_given_pose():
    fs = FLScheduler(now_fn=lambda: NOW)
    records = [
        _record("a", pos=(0.0, 0.0, 0.0)),
        _record("b", pos=(100.0, 0.0, 0.0)),
        _record("c", pos=(200.0, 0.0, 0.0)),
    ]
    fs.ingest_slice(_slice("a", "b", "c"), registry_records=records)

    # Start at the far end → reversed visit order.
    contacts = fs.build_pass_2_queue(rf_range_m=30.0, mule_pose=(300.0, 0.0, 0.0))
    visit_order = [c.devices[0] for c in contacts]
    assert visit_order == [DeviceID("c"), DeviceID("b"), DeviceID("a")]
