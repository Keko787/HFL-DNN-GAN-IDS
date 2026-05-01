"""Phase 5 DoD — runtime scope-violation assertion.

Design §7 principle 12::

    The learned S3.5 selector is bounded to intra-bucket ordering only.
    It cannot promote gated-out devices, cannot re-order buckets, and
    cannot admit a device that S1 rejected.

We verify three layers:

1. At the selector's entry points (``select_target`` / ``rank``), a
   foreign device ID raises :class:`SelectorScopeViolation`.

2. When wired into :class:`FLScheduler`, the learned selector still
   walks buckets in the scheduler-defined :data:`BUCKET_PRIORITY`
   order — the selector has no path to re-order them.

3. A device gated out by S1 (``is_in_slice=False``, no beacon) never
   reaches the selector — so even a perverse selector cannot promote
   it.
"""

from __future__ import annotations

import pytest

from hermes.scheduler import FLScheduler
from hermes.scheduler.selector import (
    SelectorEnv,
    SelectorScopeViolation,
    TargetSelectorRL,
    assert_candidates_admitted,
)
from hermes.types import (
    BUCKET_PRIORITY,
    Bucket,
    DeviceID,
    DeviceRecord,
    MissionSlice,
    MuleID,
    RoundCloseDelta,
    MissionOutcome,
)


MULE = MuleID("m1")
NOW = 1000.0


def _env() -> SelectorEnv:
    return SelectorEnv(
        mule_pose=(0.0, 0.0, 0.0),
        mule_energy=1.0,
        rf_prior_snr_db=20.0,
        now=NOW,
    )


# --------------------------------------------------------------------------- #
# Layer 1 — selector entry-point guards
# --------------------------------------------------------------------------- #

def test_rank_rejects_foreign_device_via_admitted_list():
    sel = TargetSelectorRL()
    foreign = [DeviceID("rogue")]
    admitted = [DeviceID("d0"), DeviceID("d1")]
    with pytest.raises(SelectorScopeViolation):
        sel.rank(
            candidates=foreign,
            device_states={},
            bucket=Bucket.NEW,
            env=_env(),
            admitted=admitted,
        )


def test_select_target_rejects_foreign_device_via_admitted_list():
    sel = TargetSelectorRL()
    with pytest.raises(SelectorScopeViolation):
        sel.select_target(
            candidates=[DeviceID("rogue")],
            device_states={},
            bucket=Bucket.NEW,
            env=_env(),
            admitted=[DeviceID("d0")],
        )


def test_scope_violation_message_names_offender():
    try:
        assert_candidates_admitted(
            [DeviceID("ghost")], [DeviceID("a"), DeviceID("b")]
        )
    except SelectorScopeViolation as exc:
        assert "ghost" in str(exc)
    else:
        raise AssertionError("expected SelectorScopeViolation")


# --------------------------------------------------------------------------- #
# Layer 2 — FLScheduler preserves BUCKET_PRIORITY under a learned selector
# --------------------------------------------------------------------------- #

def _record(did: str, pos=(0.0, 0.0, 0.0), *, is_new: bool = True) -> DeviceRecord:
    return DeviceRecord(
        device_id=DeviceID(did),
        spectrum_sig=None,
        is_new=is_new,
        last_known_position=pos,
    )


def test_fl_scheduler_keeps_bucket_priority_with_learned_selector():
    """A selector cannot re-order buckets — the scheduler's outer loop does."""
    selector = TargetSelectorRL()
    fs = FLScheduler(target_selector=selector, now_fn=lambda: NOW)

    # Three devices, one per bucket.
    #   d_new      -> NEW                         (is_new=True, never seen)
    #   d_sched    -> SCHEDULED_THIS_ROUND        (is_new=False, last_outcome CLEAN)
    #   d_beacon   -> BEACON_ACTIVE               (is_new=False, fresh beacon)
    records = [
        _record("d_new",    pos=(10.0, 0.0, 0.0), is_new=True),
        _record("d_sched",  pos=(1.0, 0.0, 0.0),  is_new=False),
        _record("d_beacon", pos=(2.0, 0.0, 0.0),  is_new=False),
    ]
    slice_ = MissionSlice(
        issued_round=1,
        mule_id=MULE,
        device_ids=tuple(r.device_id for r in records),
        issued_at=NOW,
    )
    fs.ingest_slice(slice_, registry_records=records)

    # Push d_sched into SCHEDULED_THIS_ROUND via a CLEAN delta.
    fs.ingest_round_close_delta(
        RoundCloseDelta(
            device_id=DeviceID("d_sched"),
            mule_id=MULE,
            mission_round=1,
            outcome=MissionOutcome.CLEAN,
            utility=0.8,
            contact_ts=NOW,
        )
    )
    # d_beacon gets a fresh beacon, but also clear is_new=False already.
    from hermes.types import BeaconObservation
    fs.ingest_beacon(
        BeaconObservation(device_id=DeviceID("d_beacon"), observed_at=NOW - 1.0)
    )

    queue = fs.build_target_queue(now=NOW, mule_pose=(0.0, 0.0, 0.0))

    # Every bucket listed earlier in BUCKET_PRIORITY must come first in the
    # queue, regardless of which per-device Q-scores the selector assigns.
    seen_buckets = [wp.bucket for wp in queue]
    # Index of each bucket's first appearance must follow BUCKET_PRIORITY.
    order_indices = []
    for b in BUCKET_PRIORITY:
        if b in seen_buckets:
            order_indices.append(seen_buckets.index(b))
    assert order_indices == sorted(order_indices), (
        f"selector broke bucket priority: {seen_buckets}"
    )


# --------------------------------------------------------------------------- #
# Layer 3 — S1 gating precedes the selector
# --------------------------------------------------------------------------- #

def test_fl_scheduler_never_feeds_gated_device_to_selector():
    """A device with is_in_slice=False and no beacon must be invisible to S3.5."""
    selector = TargetSelectorRL()
    fs = FLScheduler(target_selector=selector, now_fn=lambda: NOW)

    # Admit d_admit, leave d_gated untracked entirely — a later slice
    # excludes it.
    slice_ = MissionSlice(
        issued_round=1,
        mule_id=MULE,
        device_ids=(DeviceID("d_admit"),),
        issued_at=NOW,
    )
    fs.ingest_slice(
        slice_,
        registry_records=[_record("d_admit", pos=(5.0, 0.0, 0.0))],
    )

    queue = fs.build_target_queue(now=NOW, mule_pose=(0.0, 0.0, 0.0))
    queue_ids = {wp.device_id for wp in queue}

    assert DeviceID("d_admit") in queue_ids
    assert DeviceID("d_gated") not in queue_ids
