"""FeRRy Phase 1 — FedProx's proximal term in the Exp 4 local trainer.

The term is (ρ/2)·‖θ − θ_received‖²; its gradient is ρ·(θ − θ_received). With
ρ > 0 local training runs a custom loop that adds it; with ρ = 0 it keeps the
plain ``model.fit`` every recorded run used.

Marked ``slow``: these import TensorFlow and fit the real Keras model.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from experiments.exp4.model_task import (  # noqa: E402
    INPUT_DIM,
    initial_theta,
    make_local_train_fn,
    proximal_term,
    synthetic_task,
)


@pytest.mark.slow
def test_proximal_gradient_is_rho_times_the_displacement():
    rho = 0.7
    w = [tf.Variable([1.0, -2.0, 3.0]), tf.Variable([[0.5, 0.25]])]
    anchors = [tf.constant([0.0, 1.0, 1.0]), tf.constant([[0.5, -0.75]])]
    with tf.GradientTape() as tape:
        value = proximal_term(w, anchors, rho)
    grads = tape.gradient(value, w)
    expected_value = 0.5 * rho * ((1 + 9 + 4) + (0 + 1))
    assert float(value) == pytest.approx(expected_value)
    for g, v, a in zip(grads, w, anchors):
        np.testing.assert_allclose(g.numpy(), rho * (v.numpy() - a.numpy()), rtol=1e-6)


@pytest.mark.slow
def test_large_rho_keeps_local_training_near_the_received_model():
    task = synthetic_task(n_devices=1, rows_per_device=256, test_rows=64, seed=5)
    X, y = task.device_shards[0]
    theta = initial_theta(task.input_dim, seed=5)

    def drift(rho):
        fn = make_local_train_fn(
            X, y, input_dim=task.input_dim, epochs=3, batch_size=32,
            learning_rate=1e-2, seed=5, fedprox_rho=rho,
        )
        result = fn(theta, [])
        return sum(
            float(np.sum((a.astype(np.float64) - b) ** 2))
            for a, b in zip(result.delta_theta, theta)
        ), result

    free, _ = drift(0.0)
    tiny, _ = drift(1e-4)
    held, result = drift(100.0)
    assert result.num_examples == 256
    # The custom loop really trains: a negligible ρ moves about as far as fit.
    assert tiny > 0.5 * free, (tiny, free)
    # A large ρ holds the trainable weights near the received model (the
    # remaining drift is BatchNorm's moving statistics, which ρ does not touch).
    assert held < 0.25 * free, (held, free)


@pytest.mark.slow
def test_negative_rho_is_refused():
    X = np.zeros((4, INPUT_DIM), dtype=np.float32)
    y = np.zeros((4,), dtype=np.float32)
    with pytest.raises(ValueError, match="fedprox_rho"):
        make_local_train_fn(X, y, fedprox_rho=-1.0)
