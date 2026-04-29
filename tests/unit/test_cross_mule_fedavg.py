"""Unit tests — cross-mule FedAvg vs hand-computed reference.

Implementation-Plan §3 Phase 1 DoD: "Cross-mule FedAvg unit test:
weighted merge matches hand-computed reference."
"""

from __future__ import annotations

import numpy as np
import pytest

from hermes.cluster import FedAvgError, cross_mule_fedavg
from hermes.types import MuleID, PartialAggregate


def _partial(mid: str, w0: list[float], n: int) -> PartialAggregate:
    return PartialAggregate(
        mule_id=MuleID(mid),
        mission_round=1,
        weights=[np.array(w0, dtype=np.float32)],
        num_examples=n,
    )


def test_simple_two_mule_average_matches_hand_calc():
    p1 = _partial("m1", [1.0, 2.0, 3.0], n=10)
    p2 = _partial("m2", [3.0, 6.0, 9.0], n=30)
    # weighted: (1*10 + 3*30)/40, (2*10 + 6*30)/40, (3*10 + 9*30)/40
    #         = 100/40, 200/40, 300/40
    #         = 2.5, 5.0, 7.5
    out = cross_mule_fedavg([p1, p2])
    np.testing.assert_allclose(out[0], np.array([2.5, 5.0, 7.5], dtype=np.float32))


def test_empty_partial_is_skipped():
    p1 = _partial("m1", [1.0, 1.0], n=10)
    p_empty = PartialAggregate(
        mule_id=MuleID("m2"),
        mission_round=1,
        weights=[],
        num_examples=0,
    )
    out = cross_mule_fedavg([p1, p_empty])
    np.testing.assert_allclose(out[0], np.array([1.0, 1.0], dtype=np.float32))


def test_all_empty_raises():
    p_empty = PartialAggregate(
        mule_id=MuleID("m1"),
        mission_round=1,
        weights=[],
        num_examples=0,
    )
    with pytest.raises(FedAvgError):
        cross_mule_fedavg([p_empty])


def test_layer_count_mismatch_raises():
    p1 = _partial("m1", [1.0], n=1)
    p2 = PartialAggregate(
        mule_id=MuleID("m2"),
        mission_round=1,
        weights=[np.array([1.0]), np.array([2.0])],  # 2 layers vs 1
        num_examples=1,
    )
    with pytest.raises(FedAvgError, match="layer count"):
        cross_mule_fedavg([p1, p2])


def test_layer_shape_mismatch_raises():
    p1 = _partial("m1", [1.0, 2.0], n=1)
    p2 = _partial("m2", [1.0, 2.0, 3.0], n=1)
    with pytest.raises(FedAvgError, match="shape mismatch"):
        cross_mule_fedavg([p1, p2])


def test_three_mules_multilayer():
    """Multi-layer model — exercises the per-layer loop."""
    weights1 = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([0.5, 1.5], dtype=np.float32),
    ]
    weights2 = [
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        np.array([2.5, 3.5], dtype=np.float32),
    ]
    weights3 = [
        np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
        np.array([4.5, 5.5], dtype=np.float32),
    ]
    p1 = PartialAggregate(MuleID("m1"), 1, weights1, num_examples=10)
    p2 = PartialAggregate(MuleID("m2"), 1, weights2, num_examples=20)
    p3 = PartialAggregate(MuleID("m3"), 1, weights3, num_examples=70)

    out = cross_mule_fedavg([p1, p2, p3])

    # hand calc layer 0[0,0]: (1*10 + 5*20 + 9*70)/100 = (10+100+630)/100 = 7.4
    assert out[0][0, 0] == pytest.approx(7.4)
    # layer 1[0]: (0.5*10 + 2.5*20 + 4.5*70)/100 = (5+50+315)/100 = 3.7
    assert out[1][0] == pytest.approx(3.7)
