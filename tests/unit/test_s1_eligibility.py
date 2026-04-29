"""Phase 4 Stage 1 — eligibility filter tests."""

from __future__ import annotations

from hermes.scheduler.stages import filter_eligible, is_eligible
from hermes.scheduler.stages.s1_eligibility import beacon_heard, has_active_deadline
from hermes.types import DeviceID, DeviceSchedulerState


def _st(**kw) -> DeviceSchedulerState:
    return DeviceSchedulerState(device_id=DeviceID(kw.pop("did", "d1")), **kw)


def test_has_active_deadline_for_slice_member():
    assert has_active_deadline(_st(is_in_slice=True)) is True


def test_has_active_deadline_for_override_only():
    assert has_active_deadline(_st(deadline_override_ts=123.0)) is True


def test_no_deadline_without_slice_or_override():
    assert has_active_deadline(_st()) is False


def test_beacon_heard_within_window():
    st = _st(last_beacon_ts=1000.0)
    assert beacon_heard(st, now=1010.0, beacon_window_s=30.0) is True


def test_beacon_not_heard_outside_window():
    st = _st(last_beacon_ts=1000.0)
    assert beacon_heard(st, now=1100.0, beacon_window_s=30.0) is False


def test_beacon_never_observed():
    assert beacon_heard(_st(), now=1000.0, beacon_window_s=30.0) is False


def test_beacon_window_zero_rejects():
    st = _st(last_beacon_ts=1000.0)
    assert beacon_heard(st, now=1000.0, beacon_window_s=0.0) is False


def test_is_eligible_combines_rules():
    # Neither: rejected.
    assert is_eligible(_st(), now=1000.0) is False
    # Slice only: admitted.
    assert is_eligible(_st(is_in_slice=True), now=1000.0) is True
    # Beacon only: admitted.
    assert is_eligible(_st(last_beacon_ts=995.0), now=1000.0) is True
    # Both: admitted.
    assert is_eligible(
        _st(is_in_slice=True, last_beacon_ts=995.0), now=1000.0
    ) is True


def test_filter_eligible_preserves_input_order():
    states = {
        DeviceID("a"): DeviceSchedulerState(
            device_id=DeviceID("a"), is_in_slice=True
        ),
        DeviceID("b"): DeviceSchedulerState(device_id=DeviceID("b")),
        DeviceID("c"): DeviceSchedulerState(
            device_id=DeviceID("c"), last_beacon_ts=999.0
        ),
    }
    got = filter_eligible(states, now=1000.0, beacon_window_s=30.0)
    assert got == [DeviceID("a"), DeviceID("c")]
