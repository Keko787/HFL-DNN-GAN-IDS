"""Unit tests — DeviceRegistry slicing invariants.

Covers Implementation Plan §6.1 + §7 risk row "cross-mule slicing bug
dispatches overlapping slices". Property: across any rebalance,
``⋃ slices == registry`` and ``∀ i≠j: slice_i ∩ slice_j = ∅``.
"""

from __future__ import annotations

import pytest

from hermes.cluster import DeviceRegistry
from hermes.types import DeviceID, MuleID, SpectrumSig


def _spec() -> SpectrumSig:
    return SpectrumSig(bands=(0, 1, 2), last_good_snr_per_band=(10.0, 12.0, 8.0))


def _seed(reg: DeviceRegistry, n: int) -> list[DeviceID]:
    ids: list[DeviceID] = []
    for i in range(n):
        did = DeviceID(f"d{i:03d}")
        reg.register(did, position=(0.0, 0.0, 0.0), spectrum_sig=_spec())
        ids.append(did)
    return ids


def test_register_idempotent():
    reg = DeviceRegistry()
    a = reg.register(DeviceID("d1"), (0, 0, 0), _spec())
    b = reg.register(DeviceID("d1"), (1, 1, 1), _spec())
    assert a is b
    assert reg.get(DeviceID("d1")).last_known_position == (0, 0, 0)


def test_rebalance_disjoint_and_complete():
    reg = DeviceRegistry()
    devices = _seed(reg, 13)
    mules = [MuleID("m1"), MuleID("m2"), MuleID("m3")]
    slices = reg.rebalance(mules)

    # disjoint
    seen: set[DeviceID] = set()
    for s in slices.values():
        bucket = set(s.device_ids)
        assert bucket.isdisjoint(seen), f"overlap detected for {s.mule_id}"
        seen |= bucket

    # complete
    assert seen == set(devices)


def test_rebalance_round_robin_balanced():
    reg = DeviceRegistry()
    _seed(reg, 9)
    slices = reg.rebalance([MuleID("a"), MuleID("b"), MuleID("c")])
    sizes = sorted(len(s.device_ids) for s in slices.values())
    # 9 devices / 3 mules = 3 each, perfectly balanced
    assert sizes == [3, 3, 3]


def test_rebalance_uneven_split_no_loss():
    reg = DeviceRegistry()
    _seed(reg, 10)
    slices = reg.rebalance([MuleID("a"), MuleID("b"), MuleID("c")])
    sizes = sorted(len(s.device_ids) for s in slices.values())
    # 10 devices / 3 mules = 4,3,3
    assert sizes == [3, 3, 4]
    assert sum(sizes) == 10


def test_rebalance_increments_round_counter():
    reg = DeviceRegistry()
    _seed(reg, 4)
    slices1 = reg.rebalance([MuleID("a"), MuleID("b")])
    slices2 = reg.rebalance([MuleID("a"), MuleID("b")])
    r1 = next(iter(slices1.values())).issued_round
    r2 = next(iter(slices2.values())).issued_round
    assert r2 == r1 + 1


def test_rebalance_rejects_duplicate_mules():
    reg = DeviceRegistry()
    _seed(reg, 2)
    with pytest.raises(ValueError):
        reg.rebalance([MuleID("a"), MuleID("a")])


def test_rebalance_rejects_empty_mules():
    reg = DeviceRegistry()
    with pytest.raises(ValueError):
        reg.rebalance([])


def test_update_after_round_bumps_counters():
    reg = DeviceRegistry()
    _seed(reg, 1)
    did = DeviceID("d000")
    reg.update_after_round(did, on_time=True)
    reg.update_after_round(did, on_time=True)
    reg.update_after_round(did, on_time=False)
    rec = reg.get(did)
    assert rec.on_time_history == 2
    assert rec.missed_history == 1
    assert rec.is_new is False  # cleared after first round


def test_update_after_round_unknown_device_silent():
    reg = DeviceRegistry()
    # unknown device — must not raise (cluster ingest will log it instead)
    reg.update_after_round(DeviceID("ghost"), on_time=True)


def test_snapshot_reports_assignment():
    reg = DeviceRegistry()
    _seed(reg, 5)
    reg.rebalance([MuleID("a"), MuleID("b")])
    snap = reg.snapshot()
    assert snap.total == 5
    assert snap.assigned == 5
    assert sorted(snap.by_mule.keys()) == [MuleID("a"), MuleID("b")]


def test_mission_slice_rejects_duplicate_ids():
    """``MissionSlice.__post_init__`` invariant — paranoia layer."""
    from hermes.types import MissionSlice

    with pytest.raises(ValueError):
        MissionSlice(
            mule_id=MuleID("m1"),
            device_ids=(DeviceID("d1"), DeviceID("d1")),
            issued_round=0,
            issued_at=0.0,
        )
