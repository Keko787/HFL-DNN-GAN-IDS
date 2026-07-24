"""EX-4.1 foundation tests — the real DNN-IDS learning task.

Exercises the full train -> aggregate -> evaluate path in-process on a
deterministic synthetic (real-shaped) task: the canonical
``create_CICIOT_Model`` is trained on each device shard, aggregated with
the *real* ``partial_fedavg``, and the aggregated θ is shown to improve on
a held-out test set. This is the decision-independent core that EX-4.1's
orchestrator wiring builds on.

Marked ``slow`` — importing TensorFlow and fitting a Keras model is heavy
relative to the pure-Python unit suite (no subprocesses, though).
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp4.model_task import (
    INPUT_DIM,
    build_ids_model,
    evaluate_theta,
    initial_theta,
    make_local_train_fn,
    synthetic_task,
)


@pytest.mark.slow
def test_initial_theta_is_deterministic_and_model_shaped():
    t1 = initial_theta(INPUT_DIM, seed=3)
    t2 = initial_theta(INPUT_DIM, seed=3)
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        assert a.shape == b.shape
        assert np.allclose(a, b), "initial_theta must be reproducible"
    # Shapes line up with a freshly-built model (so set_weights accepts it).
    model_shapes = [w.shape for w in build_ids_model(INPUT_DIM).get_weights()]
    assert [w.shape for w in t1] == model_shapes


@pytest.mark.slow
def test_fedavg_round_improves_heldout_and_shapes_satisfy_partial_fedavg():
    from hermes.mission.partial_fedavg import partial_fedavg
    from hermes.types import DeviceID, GradientSubmission, MuleID

    task = synthetic_task(
        n_devices=3, rows_per_device=256, test_rows=256, seed=7,
    )
    assert task.input_dim == INPUT_DIM
    assert task.n_devices == 3

    theta0 = initial_theta(task.input_dim, seed=7)
    before = evaluate_theta(theta0, task.X_test, task.y_test, input_dim=task.input_dim)

    # One federated round: every device trains from the *same* θ0.
    submissions = []
    for i, (X, y) in enumerate(task.device_shards):
        fn = make_local_train_fn(
            X, y, input_dim=task.input_dim, epochs=12, batch_size=64, seed=7,
        )
        res = fn(theta0, [])
        assert res.num_examples == len(y)
        submissions.append(
            GradientSubmission(
                device_id=DeviceID(f"exp4-dev-{i:03d}"),
                mule_id=MuleID("exp4-mule"),
                mission_round=0,
                delta_theta=res.delta_theta,
                num_examples=res.num_examples,
                submitted_at=0.0,
            )
        )

    # All device weight lists must share layer count + shapes, else the
    # real aggregator rejects them. Calling it *is* the assertion.
    agg = partial_fedavg(MuleID("exp4-mule"), 0, submissions)
    assert agg.num_examples == sum(len(y) for _, y in task.device_shards)

    after = evaluate_theta(
        agg.weights, task.X_test, task.y_test, input_dim=task.input_dim,
    )

    # On a linearly-separable task, one aggregated round moves the model
    # off the random-init baseline: loss drops and accuracy climbs well
    # above chance.
    assert after["loss"] < before["loss"], (before, after)
    assert after["accuracy"] > before["accuracy"], (before, after)
    assert after["accuracy"] > 0.75, after
    assert 0.0 <= after["auc"] <= 1.0


@pytest.mark.slow
def test_evaluate_theta_handles_single_class_test_set():
    task = synthetic_task(n_devices=1, rows_per_device=32, test_rows=16, seed=1)
    theta = initial_theta(task.input_dim, seed=1)
    # All-benign test set — AUC is undefined and must degrade gracefully.
    y_single = np.zeros_like(task.y_test)
    m = evaluate_theta(theta, task.X_test, y_single, input_dim=task.input_dim)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["auc"] == pytest.approx(0.5)
