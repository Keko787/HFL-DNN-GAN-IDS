"""Phase 4 Stage 3.5 — intra-bucket selector (placeholder) tests."""

from __future__ import annotations

from hermes.scheduler.stages import select_order, select_target
from hermes.types import DeviceID, DeviceSchedulerState


def _states(**positions):
    return {
        DeviceID(k): DeviceSchedulerState(
            device_id=DeviceID(k), last_known_position=v
        )
        for k, v in positions.items()
    }


def test_order_sorted_by_distance_to_mule():
    states = _states(
        far=(100.0, 0.0, 0.0),
        near=(1.0, 0.0, 0.0),
        mid=(10.0, 0.0, 0.0),
    )
    got = select_order(
        [DeviceID("far"), DeviceID("near"), DeviceID("mid")],
        states,
        mule_pose=(0.0, 0.0, 0.0),
    )
    assert got == [DeviceID("near"), DeviceID("mid"), DeviceID("far")]


def test_order_tie_breaks_on_device_id():
    states = _states(
        b=(5.0, 0.0, 0.0),
        a=(5.0, 0.0, 0.0),
    )
    got = select_order(
        [DeviceID("b"), DeviceID("a")], states, mule_pose=(0.0, 0.0, 0.0)
    )
    assert got == [DeviceID("a"), DeviceID("b")]


def test_order_handles_missing_state_as_infinite():
    states = _states(near=(1.0, 0.0, 0.0))
    got = select_order(
        [DeviceID("ghost"), DeviceID("near")],
        states,
        mule_pose=(0.0, 0.0, 0.0),
    )
    assert got == [DeviceID("near"), DeviceID("ghost")]


def test_target_returns_head():
    states = _states(far=(100.0, 0.0, 0.0), near=(1.0, 0.0, 0.0))
    assert select_target(
        [DeviceID("far"), DeviceID("near")], states
    ) == DeviceID("near")


def test_target_on_empty_returns_none():
    assert select_target([], {}) is None
