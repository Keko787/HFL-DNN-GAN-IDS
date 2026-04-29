"""Phase 2 partial-FedAvg tests (mission-scope on-mule aggregator)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from hermes.mission import PartialFedAvgError, partial_fedavg
from hermes.types import DeviceID, GradientSubmission, MuleID


def _sub(device: str, round_: int, weights, n_examples: int) -> GradientSubmission:
    return GradientSubmission(
        device_id=DeviceID(device),
        mule_id=MuleID("m1"),
        mission_round=round_,
        delta_theta=weights,
        num_examples=n_examples,
        submitted_at=time.time(),
    )


def test_raises_on_empty_input():
    with pytest.raises(PartialFedAvgError):
        partial_fedavg(MuleID("m1"), 1, [])


def test_raises_on_mixed_round():
    w = [np.ones((2,), dtype=np.float32)]
    subs = [_sub("d1", 1, w, 10), _sub("d2", 2, w, 10)]
    with pytest.raises(PartialFedAvgError):
        partial_fedavg(MuleID("m1"), 1, subs)


def test_raises_when_all_num_examples_zero():
    w = [np.ones((2,), dtype=np.float32)]
    subs = [_sub("d1", 1, w, 0), _sub("d2", 1, w, 0)]
    with pytest.raises(PartialFedAvgError):
        partial_fedavg(MuleID("m1"), 1, subs)


def test_raises_on_layer_count_mismatch():
    subs = [
        _sub("d1", 1, [np.zeros((2,))], 1),
        _sub("d2", 1, [np.zeros((2,)), np.zeros((2,))], 1),
    ]
    with pytest.raises(PartialFedAvgError):
        partial_fedavg(MuleID("m1"), 1, subs)


def test_raises_on_shape_mismatch():
    subs = [
        _sub("d1", 1, [np.zeros((2,))], 1),
        _sub("d2", 1, [np.zeros((3,))], 1),
    ]
    with pytest.raises(PartialFedAvgError):
        partial_fedavg(MuleID("m1"), 1, subs)


def test_weighted_average_matches_reference():
    # Two devices: d1 with 10 examples, d2 with 30. weighted avg should
    # pull toward d2 by 3x.
    w1 = [np.array([0.0, 0.0], dtype=np.float32)]
    w2 = [np.array([4.0, 8.0], dtype=np.float32)]
    subs = [_sub("d1", 1, w1, 10), _sub("d2", 1, w2, 30)]
    agg = partial_fedavg(MuleID("m1"), 1, subs)

    # expected = (10*[0,0] + 30*[4,8]) / 40 = [3,6]
    assert agg.num_examples == 40
    assert np.allclose(agg.weights[0], np.array([3.0, 6.0], dtype=np.float32))
    assert agg.contributing_devices == (DeviceID("d1"), DeviceID("d2"))


def test_zero_example_submissions_are_dropped_but_others_succeed():
    w = [np.array([2.0, 4.0], dtype=np.float32)]
    subs = [
        _sub("d0", 1, w, 0),  # dropped
        _sub("d1", 1, w, 5),
    ]
    agg = partial_fedavg(MuleID("m1"), 1, subs)
    assert agg.contributing_devices == (DeviceID("d1"),)
    assert agg.num_examples == 5
    assert np.allclose(agg.weights[0], w[0])


def test_dtype_preserved_from_first_submission():
    subs = [
        _sub("d1", 1, [np.ones((4,), dtype=np.float32)], 2),
        _sub("d2", 1, [np.ones((4,), dtype=np.float32)], 2),
    ]
    agg = partial_fedavg(MuleID("m1"), 1, subs)
    assert agg.weights[0].dtype == np.float32


def test_mule_and_round_stamped_onto_aggregate():
    subs = [_sub("d1", 7, [np.zeros((2,))], 3)]
    agg = partial_fedavg(MuleID("mA"), 7, subs)
    assert agg.mule_id == MuleID("mA")
    assert agg.mission_round == 7
