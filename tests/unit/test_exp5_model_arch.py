"""Exp 5 addendum, Study 5.12: the IDS architecture as a setting (``--model-arch``).

``build_ids_model`` hard-coded ``create_CICIOT_Model``, and the repository's
other builders were not wired. Pinned:

* **The registry** (``model_task.MODEL_ARCHS``): each architecture builds a
  binary classifier over the 21 canonical inputs with one sigmoid output,
  deterministically from the seed, trains a round and evaluates; their θ
  differ in size (the canonical model's is the plan's measured 18,756 B, and
  ``high_performance``'s is the largest). An unknown name is refused. The
  default (None) is the canonical model, built exactly as recorded.
* **The configs**: ``DeviceConfig`` and ``ClusterConfig`` carry
  ``model_arch``, written to the per-role JSON only when set; the topology
  builder passes it to both; the device's trainer, the cluster's evaluation
  and the seed θ are built on it, each called as recorded when it is None.
* **The driver** refuses an architecture without the real model or unknown;
  the runner passes ``--model-arch`` only when given.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.exp4 import runner_main
from experiments.exp4.driver import Exp4Driver
from experiments.exp4.model_task import (
    MODEL_ARCHS,
    build_ids_model,
    check_model_arch,
    evaluate_theta,
    initial_theta,
    load_weights,
    make_local_train_fn,
    synthetic_task,
)
from experiments.exp4.prep import prepare_trial
from experiments.exp4.topology_builder import build_exp4_topology
from hermes.processes.config import (
    CONFIG_FIELDS_OMITTED_AT_NONE,
    ClusterConfig,
    DeviceConfig,
    cluster_config_from_json,
    cluster_config_to_json,
    device_config_from_json,
    device_config_to_json,
)

from tests.unit import test_p3_cluster_sim as CS

DIM = 21
#: Each architecture's θ in bytes at 21 inputs (float32).
THETA_BYTES = {"ciciot": 18_756, "optimized": 92_676, "balanced": 41_360,
               "high_performance": 347_396}


@pytest.fixture(scope="module")
def task():
    return synthetic_task(n_devices=1, rows_per_device=64, test_rows=64, seed=1, input_dim=DIM)


def test_the_registry_and_its_refusals():
    assert MODEL_ARCHS == ("ciciot", "optimized", "balanced", "high_performance")
    assert check_model_arch(None) == "ciciot"
    with pytest.raises(ValueError, match="model_arch"):
        check_model_arch("resnet")


@pytest.mark.parametrize("arch", MODEL_ARCHS)
def test_each_architecture_is_a_binary_ids_that_trains_and_evaluates(arch, task):
    model = build_ids_model(DIM, arch=arch)
    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation.__name__ == "sigmoid"
    theta = initial_theta(DIM, seed=3, arch=arch)
    assert all(np.array_equal(a, b) for a, b in zip(theta, initial_theta(DIM, seed=3, arch=arch)))
    assert sum(w.nbytes for w in theta) == THETA_BYTES[arch]
    result = make_local_train_fn(*task.device_shards[0], input_dim=DIM, arch=arch)(theta, [])
    assert result.num_examples == 64
    assert [w.shape for w in result.delta_theta] == [w.shape for w in theta]
    m = evaluate_theta(result.delta_theta, task.X_test, task.y_test, input_dim=DIM, arch=arch)
    assert 0.0 <= m["accuracy"] <= 1.0 and 0.0 <= m["auc"] <= 1.0


def test_the_default_is_the_canonical_model_as_recorded():
    a, b = initial_theta(DIM, seed=5), initial_theta(DIM, seed=5, arch="ciciot")
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert THETA_BYTES["high_performance"] == max(THETA_BYTES.values())


def test_the_configs_write_the_architecture_only_when_set():
    assert CONFIG_FIELDS_OMITTED_AT_NONE == ("model_arch",)
    for cls, to_json, from_json in (
            (DeviceConfig, device_config_to_json, device_config_from_json),
            (ClusterConfig, cluster_config_to_json, cluster_config_from_json)):
        ident = {"device_id": "d"} if cls is DeviceConfig else {"cluster_id": "c"}
        plain = cls(**ident)
        assert plain.model_arch is None and "model_arch" not in json.loads(to_json(plain))
        assert from_json(to_json(plain)) == plain
        chosen = cls(**ident, model_arch="balanced")
        assert json.loads(to_json(chosen))["model_arch"] == "balanced"
        assert from_json(to_json(chosen)) == chosen


def test_the_builder_hands_the_architecture_to_every_device_and_the_cluster():
    kw = dict(n_devices=3, seed=7, rf_range_m=60.0, n_missions=1,
              train_shard_paths=["a", "b", "c"], input_dim=DIM, init_theta_path="t",
              eval_test_path="e")
    plain = build_exp4_topology(**kw)
    assert {d.model_arch for d in plain.devices} == {None} and plain.cluster.model_arch is None
    chosen = build_exp4_topology(**kw, model_arch="optimized")
    assert {d.model_arch for d in chosen.devices} == {"optimized"}
    assert chosen.cluster.model_arch == "optimized"


def test_the_seed_theta_is_the_architectures(tmp_path, task):
    prep = prepare_trial(tmp_path / "p", task=task, theta_seed=2, arch="balanced")
    theta = load_weights(prep.init_theta_path)
    assert sum(w.nbytes for w in theta) == THETA_BYTES["balanced"]


def test_the_device_trains_the_configured_architecture(monkeypatch, tmp_path, task):
    import experiments.exp4.model_task as model_task
    from experiments.exp4.model_task import save_xy
    from hermes.processes import device as device_process

    calls = []
    monkeypatch.setattr(model_task, "make_local_train_fn",
                        lambda X, y, **kw: calls.append(kw) or "fn")
    save_xy(tmp_path / "s.npz", *task.device_shards[0])
    for arch in (None, "high_performance"):
        cfg = DeviceConfig(device_id="d", train_shard_path=str(tmp_path / "s.npz"),
                           input_dim=DIM, model_arch=arch)
        assert device_process._build_local_train(cfg, seed=1) == "fn"
    assert "arch" not in calls[0] and calls[1]["arch"] == "high_performance"


def test_the_cluster_evaluates_the_configured_architecture(monkeypatch):
    import experiments.exp4.model_task as model_task

    seen = []

    def fake(theta, X, y, input_dim, **kw):
        seen.append(kw)
        return {"accuracy": 0.5, "auc": 0.5, "loss": 1.0}

    monkeypatch.setattr(model_task, "evaluate_theta", fake)
    for arch in (None, "optimized"):
        svc = CS._service(clock="sim", model_arch=arch)
        svc._eval_X, svc._eval_y, svc._eval_input_dim = np.zeros((2, 3)), np.zeros(2), 3
        CS._run(svc, [CS._up("m1", 1, sim_ts=CS.T0 + 7.0, p_loss=0.0)])
    assert all(kw == {} for kw in seen[:2]) and all(kw == {"arch": "optimized"}
                                                    for kw in seen[2:])


def test_the_driver_refuses_an_architecture_it_cannot_build():
    with pytest.raises(ValueError, match="real_model"):
        Exp4Driver(model_arch="balanced")
    with pytest.raises(ValueError, match="model_arch must be one of"):
        Exp4Driver(real_model=True, model_arch="resnet")
    assert Exp4Driver(real_model=True, model_arch="balanced")._arch_kw() == {"arch": "balanced"}
    assert Exp4Driver(real_model=True)._arch_kw() == {}


def _driver_kwargs(monkeypatch, argv):
    seen = {}

    class _Stop(Exception):
        pass

    def fake(**kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(runner_main, "Exp4Driver", fake)
    with pytest.raises(_Stop):
        runner_main.main(argv)
    return seen


def test_the_runner_passes_the_architecture_only_when_given(tmp_path, monkeypatch):
    base = ["--csv", str(tmp_path / "t.csv"), "--arms", "H1", "--real-model"]
    assert "model_arch" not in _driver_kwargs(monkeypatch, base)
    got = _driver_kwargs(monkeypatch, base + ["--model-arch", "high_performance"])
    assert got["model_arch"] == "high_performance"
