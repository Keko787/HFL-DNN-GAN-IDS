"""Unit tests — LoopbackDockLink ordering + isolation."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from hermes.transport import DockLinkError, LoopbackDockLink
from hermes.types import (
    ClusterAmendment,
    ContactHistory,
    DeviceID,
    DownBundle,
    MissionOutcome,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MissionSlice,
    MuleID,
    PartialAggregate,
    SpectrumSig,
    UpBundle,
)


def _make_up(mule: str = "m1", round_id: int = 1) -> UpBundle:
    mid = MuleID(mule)
    pa = PartialAggregate(
        mule_id=mid,
        mission_round=round_id,
        weights=[np.array([1.0, 2.0], dtype=np.float32)],
        num_examples=5,
    )
    rep = MissionRoundCloseReport(
        mule_id=mid, mission_round=round_id, started_at=0.0, finished_at=1.0,
        lines=[
            MissionRoundCloseLine(
                device_id=DeviceID("d1"),
                outcome=MissionOutcome.CLEAN,
                contact_ts=0.5,
            )
        ],
    )
    ch = ContactHistory(mule_id=mid, mission_round=round_id, records=[])
    return UpBundle(mule_id=mid, partial_aggregate=pa, round_close_report=rep, contact_history=ch)


def _make_down(mule: str = "m1") -> DownBundle:
    mid = MuleID(mule)
    return DownBundle(
        mule_id=mid,
        mission_slice=MissionSlice(mid, (), 0, 0.0),
        theta_disc=[np.zeros(2, dtype=np.float32)],
        synth_batch=[],
        cluster_amendments=ClusterAmendment(cluster_round=0),
    )


def test_up_round_trip():
    link = LoopbackDockLink()
    link.client_send_up(_make_up("m1"))
    received = link.recv_up(timeout=0.5)
    assert received.mule_id == MuleID("m1")


def test_recv_up_timeout_raises():
    link = LoopbackDockLink()
    with pytest.raises(DockLinkError):
        link.recv_up(timeout=0.05)


def test_per_mule_down_isolation():
    link = LoopbackDockLink()
    link.send_down(_make_down("m1"))
    link.send_down(_make_down("m2"))
    # m2 reads its own queue first; should NOT get m1's bundle
    got = link.client_recv_down(MuleID("m2"), timeout=0.5)
    assert got.mule_id == MuleID("m2")
    got = link.client_recv_down(MuleID("m1"), timeout=0.5)
    assert got.mule_id == MuleID("m1")


def test_close_blocks_subsequent_ops():
    link = LoopbackDockLink()
    link.close()
    with pytest.raises(DockLinkError):
        link.recv_up(timeout=0.05)
    with pytest.raises(DockLinkError):
        link.send_down(_make_down())


def test_concurrent_uploads_serialise():
    link = LoopbackDockLink()
    n = 10

    def producer():
        for i in range(n):
            link.client_send_up(_make_up(f"m{i}", round_id=i))

    t = threading.Thread(target=producer)
    t.start()
    seen = []
    for _ in range(n):
        seen.append(link.recv_up(timeout=1.0).mule_id)
    t.join()
    assert sorted(seen) == sorted([MuleID(f"m{i}") for i in range(n)])


def test_up_bundle_mismatch_rejected():
    """UpBundle constructor enforces mule-id consistency."""
    mid = MuleID("m1")
    other = MuleID("m2")
    pa = PartialAggregate(other, 1, [np.array([1.0])], num_examples=1)
    rep = MissionRoundCloseReport(mid, 1, 0.0, 1.0)
    ch = ContactHistory(mid, 1, [])
    with pytest.raises(ValueError):
        UpBundle(mule_id=mid, partial_aggregate=pa, round_close_report=rep, contact_history=ch)
