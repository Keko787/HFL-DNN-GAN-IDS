"""Phase 4 integration — 5-device re-rank after injected round-close delta.

Phase 4 DoD (Implementation Plan §3):

    Integration test: scheduler correctly re-ranks a 5-device list after
    an injected round-close delta.

Setup: 5 devices at five distinct positions, all in the slice, all new.
Round 1 queue is pure distance order. After we inject one CLEAN delta on
the *nearest* device and one TIMEOUT delta on a mid-distance device, the
re-ranked queue must reflect:

    * CLEAN device drops out of NEW bucket (is_new -> False) and sits
      in SCHEDULED_THIS_ROUND, which is ranked *after* the NEW bucket.
    * TIMEOUT device stays in NEW (clean never landed) and its
      deadline_fulfilment_s widens — deadline timestamp for that device
      moves later.
"""

from __future__ import annotations

from hermes.scheduler import FLScheduler
from hermes.scheduler.stages.s3_deadline import (
    FAST_PHASE_MISSED_WIDEN_S,
    FAST_PHASE_ON_TIME_SHRINK_S,
)
from hermes.types import (
    Bucket,
    DeviceID,
    DeviceRecord,
    MissionOutcome,
    MissionSlice,
    MuleID,
    RoundCloseDelta,
    SpectrumSig,
)


MULE = MuleID("m1")
NOW = 1000.0


def _records():
    """5 devices at increasing x distance from origin."""
    sig = SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,))
    return [
        DeviceRecord(
            device_id=DeviceID(f"d{i}"),
            last_known_position=(float(i), 0.0, 0.0),
            spectrum_sig=sig,
        )
        for i in range(5)
    ]


def _slice():
    return MissionSlice(
        mule_id=MULE,
        device_ids=tuple(DeviceID(f"d{i}") for i in range(5)),
        issued_round=1,
        issued_at=NOW,
    )


def _delta(did: str, outcome: MissionOutcome, ts: float = NOW) -> RoundCloseDelta:
    return RoundCloseDelta(
        device_id=DeviceID(did),
        mule_id=MULE,
        mission_round=1,
        outcome=outcome,
        utility=0.85,
        contact_ts=ts,
    )


def test_round1_queue_is_pure_distance_order():
    sch = FLScheduler(now_fn=lambda: NOW)
    sch.ingest_slice(_slice(), registry_records=_records())
    queue = sch.build_target_queue(now=NOW, mule_pose=(0.0, 0.0, 0.0))
    # All in NEW bucket, sorted by distance d0 < d1 < ... < d4
    assert [wp.device_id for wp in queue] == [DeviceID(f"d{i}") for i in range(5)]
    assert all(wp.bucket is Bucket.NEW for wp in queue)


def test_clean_then_timeout_reranks_across_buckets():
    sch = FLScheduler(now_fn=lambda: NOW)
    # Use a fresh registry with is_new=True (the default).
    sch.ingest_slice(_slice(), registry_records=_records())

    # Inject fast-phase deltas:
    #   d0 CLEAN  -> drops out of NEW, into SCHEDULED_THIS_ROUND
    #   d2 TIMEOUT -> stays NEW, but widened fulfilment window
    sch.ingest_round_close_delta(_delta("d0", MissionOutcome.CLEAN))
    sch.ingest_round_close_delta(_delta("d2", MissionOutcome.TIMEOUT))

    queue = sch.build_target_queue(now=NOW, mule_pose=(0.0, 0.0, 0.0))
    ids = [wp.device_id for wp in queue]
    buckets = [wp.bucket for wp in queue]

    # Remaining NEW devices (d1, d2, d3, d4) should come first, in distance
    # order. d0 is now SCHEDULED (is_new=False) and goes to the back.
    assert ids == [
        DeviceID("d1"),
        DeviceID("d2"),
        DeviceID("d3"),
        DeviceID("d4"),
        DeviceID("d0"),
    ]
    assert buckets == [
        Bucket.NEW,
        Bucket.NEW,
        Bucket.NEW,
        Bucket.NEW,
        Bucket.SCHEDULED_THIS_ROUND,
    ]

    # d0's deadline_fulfilment_s shrank; d2's widened.
    d0_fulfilment = sch.device_states[DeviceID("d0")].deadline_fulfilment_s
    d2_fulfilment = sch.device_states[DeviceID("d2")].deadline_fulfilment_s
    d1_fulfilment = sch.device_states[DeviceID("d1")].deadline_fulfilment_s  # untouched

    assert d0_fulfilment == d1_fulfilment - FAST_PHASE_ON_TIME_SHRINK_S
    assert d2_fulfilment == d1_fulfilment + FAST_PHASE_MISSED_WIDEN_S
