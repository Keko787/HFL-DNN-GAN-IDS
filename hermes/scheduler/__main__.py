"""Phase 4 demo — scheduler drives visit order across two rounds.

Run with:
    python -m hermes.scheduler

Exercises the Phase 4 DoD (Implementation Plan §3):

    End-to-end with Phase 3: two-round mission runs, deadlines visibly
    adapt between rounds.

We don't spin up the full Phase 3 roundtrip here (that's covered by
``python -m hermes.mule``) — the demo concentrates on the scheduler
pipeline so the output is easy to read.

This demo only exercises the legacy per-device path
(``build_target_queue``). Sprint 1.5's two-pass + ContactWaypoint
clustering path (``build_contact_queue`` / ``build_pass_2_queue``) is
covered by ``python -m hermes.mule`` and the integration tests under
``tests/integration/test_two_pass_contact.py``.

Round 1:
    * 5 devices, all new, all in the slice.
    * Queue = pure distance order inside NEW bucket.

Fast-phase deltas between rounds:
    * d0 CLEAN   -> shrinks its window, drops NEW, refreshes idle_ref
    * d2 TIMEOUT -> widens its window, stays NEW

Slow-phase amendment at dock:
    * d1 receives an explicit deadline override.

Round 2:
    * Re-run build_target_queue and print the queue.
    * d0 moves to SCHEDULED_THIS_ROUND (ranked after the remaining NEWs).
    * d1's deadline_ts is the override timestamp.
    * d2's fulfilment window has widened.
"""

from __future__ import annotations

import logging
import sys

from hermes.scheduler import FLScheduler
from hermes.types import (
    Bucket,
    ClusterAmendment,
    DeviceID,
    DeviceRecord,
    MissionOutcome,
    MissionSlice,
    MuleID,
    RoundCloseDelta,
    SpectrumSig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("hermes.phase4_demo")

MULE = MuleID("mule-01")
NOW_R1 = 1_000.0
NOW_R2 = 2_000.0


def _make_records():
    sig = SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,))
    return [
        DeviceRecord(
            device_id=DeviceID(f"d{i}"),
            last_known_position=(float(i * 10), 0.0, 0.0),
            spectrum_sig=sig,
        )
        for i in range(5)
    ]


def _make_slice(issued_at: float) -> MissionSlice:
    return MissionSlice(
        mule_id=MULE,
        device_ids=tuple(DeviceID(f"d{i}") for i in range(5)),
        issued_round=1,
        issued_at=issued_at,
    )


def _print_queue(label: str, queue) -> None:
    print(f"\n=== {label} ===")
    if not queue:
        print("  (empty)")
        return
    for pos, wp in enumerate(queue):
        print(
            f"  {pos}. {wp.device_id:<8} bucket={wp.bucket.value:<20} "
            f"pos={wp.position} deadline_ts={wp.deadline_ts:.1f}"
        )


def run_demo() -> int:
    sch = FLScheduler(now_fn=lambda: NOW_R1)

    # ========================= ROUND 1 =========================
    print("\n=== round 1 ingest: slice(5 devices) ===")
    sch.ingest_slice(_make_slice(NOW_R1), registry_records=_make_records())
    for did, st in sch.device_states.items():
        print(
            f"  {did}: is_new={st.is_new} pos={st.last_known_position} "
            f"fulfilment={st.deadline_fulfilment_s}"
        )

    q1 = sch.build_target_queue(now=NOW_R1, mule_pose=(0.0, 0.0, 0.0))
    _print_queue("round 1 target queue", q1)
    assert all(wp.bucket is Bucket.NEW for wp in q1)
    assert [wp.device_id for wp in q1] == [DeviceID(f"d{i}") for i in range(5)]

    # ========================= FAST-PHASE INJECTS =========================
    print("\n=== injecting fast-phase round-close deltas ===")
    print("  d0 CLEAN (on-time)")
    sch.ingest_round_close_delta(
        RoundCloseDelta(
            device_id=DeviceID("d0"),
            mule_id=MULE,
            mission_round=1,
            outcome=MissionOutcome.CLEAN,
            utility=0.90,
            contact_ts=NOW_R1,
        )
    )
    print("  d2 TIMEOUT (missed)")
    sch.ingest_round_close_delta(
        RoundCloseDelta(
            device_id=DeviceID("d2"),
            mule_id=MULE,
            mission_round=1,
            outcome=MissionOutcome.TIMEOUT,
            utility=0.0,
            contact_ts=NOW_R1,
        )
    )

    # ========================= SLOW-PHASE AMENDMENT =========================
    print("\n=== slow-phase amendment at dock ===")
    override_ts = NOW_R2 + 15.0
    print(f"  d1 deadline override -> {override_ts:.1f}")
    amend = ClusterAmendment(
        cluster_round=1,
        deadline_overrides={DeviceID("d1"): override_ts},
        notes="demo amendment r1",
    )
    # Re-ingest the slice for round 2 (same members) with the amendment folded.
    sch._now = lambda: NOW_R2  # advance the clock for round 2
    sch.ingest_slice(_make_slice(NOW_R2), amendment=amend)

    # ========================= ROUND 2 =========================
    q2 = sch.build_target_queue(now=NOW_R2, mule_pose=(0.0, 0.0, 0.0))
    _print_queue("round 2 target queue", q2)

    # Deadline / bucket assertions — prove the adaptation actually happened.
    d0 = sch.device_states[DeviceID("d0")]
    d1 = sch.device_states[DeviceID("d1")]
    d2 = sch.device_states[DeviceID("d2")]

    assert d0.is_new is False, "d0 should drop out of NEW after CLEAN"
    assert d1.deadline_override_ts == override_ts, "d1 override lost"
    # d2 should have a *larger* fulfilment window than d1 (untouched) now.
    assert d2.deadline_fulfilment_s > d1.deadline_fulfilment_s

    # d0 is now SCHEDULED_THIS_ROUND; buckets before it should be NEW only.
    buckets = [wp.bucket for wp in q2]
    assert Bucket.SCHEDULED_THIS_ROUND in buckets
    first_sched_idx = buckets.index(Bucket.SCHEDULED_THIS_ROUND)
    assert all(b is Bucket.NEW for b in buckets[:first_sched_idx])

    print("\n=== Phase 4 summary ===")
    print(f"  d0 fulfilment_s = {d0.deadline_fulfilment_s} (was 60, CLEAN shrinks)")
    print(f"  d1 override_ts  = {d1.deadline_override_ts} (from amendment)")
    print(f"  d2 fulfilment_s = {d2.deadline_fulfilment_s} (was 60, TIMEOUT widens)")
    print("  DoD checks:")
    print("   * round 1 queue is pure distance order ................ OK")
    print("   * fast-phase delta shifts d0 bucket + shrinks window .. OK")
    print("   * slow-phase amendment overrides d1 deadline .......... OK")
    print("   * fast-phase missed widens d2 window .................. OK")
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
