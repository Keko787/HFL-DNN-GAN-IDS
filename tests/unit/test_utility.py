"""Phase 2 utility-formula tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hermes.mission import (
    cosine_similarity,
    diversity_adjusted,
    performance_score,
    utility,
)


def test_performance_score_bounds():
    assert performance_score(0.0, 0.0, 1e9) == pytest.approx(0.0)
    # perfect metrics, zero loss -> 1.0
    assert performance_score(1.0, 1.0, 0.0) == pytest.approx(1.0)


def test_performance_score_weights_default_sum_to_one():
    # acc=1, auc=1, loss=0 -> 0.5+0.3+0.2 == 1.0
    assert performance_score(1.0, 1.0, 0.0) == pytest.approx(1.0)


def test_performance_score_loss_is_clipped_and_inverted():
    high_loss = performance_score(1.0, 1.0, 10.0)  # at cap -> loss_score 0
    no_loss = performance_score(1.0, 1.0, 0.0)
    assert high_loss < no_loss
    assert high_loss == pytest.approx(0.5 + 0.3)  # just acc + auc terms


def test_cosine_similarity_identical_weights_is_one():
    a = [np.array([1.0, 2.0, 3.0])]
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_is_zero():
    a = [np.array([1.0, 0.0])]
    b = [np.array([0.0, 1.0])]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_is_minus_one():
    a = [np.array([1.0, 0.0])]
    b = [np.array([-1.0, 0.0])]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_norm_is_zero():
    a = [np.zeros(4)]
    b = [np.array([1.0, 2.0, 3.0, 4.0])]
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_empty_is_zero():
    assert cosine_similarity([], []) == 0.0


def test_cosine_similarity_shape_mismatch_raises():
    a = [np.ones((3,))]
    b = [np.ones((4,))]
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_cosine_similarity_layer_count_mismatch_raises():
    a = [np.ones((2,))]
    b = [np.ones((2,)), np.ones((2,))]
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_diversity_adjusted_scales_by_perf_discount():
    a = [np.array([1.0, 2.0])]
    b = [np.array([1.0, 2.0])]
    full = diversity_adjusted(a, b, perf_discount=1.0)
    half = diversity_adjusted(a, b, perf_discount=0.5)
    assert full == pytest.approx(1.0)
    assert half == pytest.approx(0.5)


def test_utility_default_weights():
    # Design-doc defaults are w1=0.7, w2=0.3
    u = utility(performance=1.0, diversity=0.0)
    assert u == pytest.approx(0.7)
    u = utility(performance=0.0, diversity=1.0)
    assert u == pytest.approx(0.3)


def test_utility_custom_weights():
    u = utility(performance=0.5, diversity=0.5, w1=0.2, w2=0.8)
    assert u == pytest.approx(0.2 * 0.5 + 0.8 * 0.5)
