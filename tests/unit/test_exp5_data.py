"""Exp 5 addendum, Study 5.13: data heterogeneity and the detector's own metrics.

Pinned:

* **The partitioner** (``experiments/exp4/partition.py``): ``iid`` is the
  recorded ``partition_indices`` split exactly; ``dirichlet`` skews each
  class over the devices (strongly at alpha 0.1, hardly at 100), ``quantity``
  the shard sizes; every draw is a function of the seed, the partition and
  alpha; no shard is ever empty, and a partition that cannot fill every
  device is refused; a partition or alpha that cannot be drawn is refused.
* **The task**: by default the synthetic task is drawn exactly as recorded;
  ``families`` labels each row (Benign exactly for class 0) without moving a
  row, and a skewed partition re-cuts the IID rows (same rows, same test set).
  ``save_xy`` stores a family only when given; ``prepare_trial`` writes the
  families and refuses an empty shard. The canonical loader's
  ``keep_original_label`` is off by default.
* **The detector's metrics** (``detection_metrics``), hand-worked, and in the
  evaluation only with the families; the cluster's ``model_eval`` adds
  ``detection`` only when its test set carries them; the consumer reads it
  strictly.
* **The driver** refuses a non-IID partition or the families without the
  real model, passes them to the task only when set, and records them with
  each device's shard size in the kept trace's status marker; the runner
  passes them only when given.
* **The scorer's detection columns**, hand-worked, only with
  ``detection_columns``, last.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from experiments.analysis.traces_scorer import (
    COST_COLUMNS,
    DETECTION_COLUMNS,
    DetectionReport,
    detection_report,
    main,
    score_trial,
)
from experiments.exp1.data_partition import partition_indices
from experiments.exp4 import partition as P
from experiments.exp4 import runner_main
from experiments.exp4.driver import Exp4Driver
from experiments.exp4.events_consumer import Detection, observation_from_rows
from experiments.exp4.model_task import (
    CiciotTask,
    _eval_with_model,
    detection_metrics,
    load_family,
    load_xy,
    save_xy,
    synthetic_task,
)
from experiments.exp4.prep import prepare_trial
from hermes.l1.mission_clock import SIM_EPOCH_S

from tests.unit import test_p3_cluster_sim as CS

LABELS = np.repeat(np.arange(4), 250)          # four classes, 250 rows each


# --------------------------------------------------------------------------- #
# 1. The partitioner
# --------------------------------------------------------------------------- #

def test_iid_is_the_recorded_split_exactly():
    shards = P.partition_rows(LABELS, n_devices=6, seed=42)
    assert [s.tolist() for s in shards] == [list(s) for s in partition_indices(1000, 6, seed=42)]


def _concentration(shards):
    """The mean over devices of the largest class's share of the shard."""
    return float(np.mean([np.bincount(LABELS[s], minlength=4).max() / len(s) for s in shards]))


def test_dirichlet_skews_each_class_and_alpha_sets_how_much():
    strong = P.partition_rows(LABELS, n_devices=6, seed=1, partition="dirichlet", alpha=0.1)
    mild = P.partition_rows(LABELS, n_devices=6, seed=1, partition="dirichlet", alpha=100.0)
    for shards in (strong, mild):
        rows = np.concatenate(shards)
        assert sorted(rows.tolist()) == list(range(1000))          # every row, once
        assert min(len(s) for s in shards) >= 1
    assert _concentration(strong) > 0.6 > 0.35 > _concentration(mild)
    again = P.partition_rows(LABELS, n_devices=6, seed=1, partition="dirichlet", alpha=0.1)
    assert [s.tolist() for s in again] == [s.tolist() for s in strong]
    other = P.partition_rows(LABELS, n_devices=6, seed=2, partition="dirichlet", alpha=0.1)
    assert [s.tolist() for s in other] != [s.tolist() for s in strong]


def test_quantity_skews_the_shard_sizes_not_the_class_mix():
    shards = P.partition_rows(LABELS, n_devices=6, seed=3, partition="quantity", alpha=0.5)
    sizes = [len(s) for s in shards]
    assert sum(sizes) == 1000 and min(sizes) >= 1 and max(sizes) > 2 * min(sizes)
    big = max(shards, key=len)
    assert _concentration([big]) < 0.4                              # a mix, as IID


def test_no_shard_is_empty_and_an_unfillable_partition_is_refused():
    # 6 rows over 6 devices at a tiny alpha: some device is always left empty
    with pytest.raises(ValueError, match="fewer than 1 row"):
        P.partition_rows(np.arange(6) % 2, n_devices=6, seed=0, partition="dirichlet",
                         alpha=0.01)
    with pytest.raises(ValueError, match="cannot give"):
        P.partition_rows(np.zeros(5), n_devices=6, seed=0)
    shards = P.partition_rows(LABELS, n_devices=6, seed=0, partition="dirichlet",
                              alpha=0.05, min_rows=20)
    assert min(len(s) for s in shards) >= 20


@pytest.mark.parametrize("partition, alpha, match", [
    ("iid", 1.0, "takes no alpha"), ("dirichlet", None, "finite alpha"),
    ("dirichlet", 0.0, "finite alpha"), ("quantity", float("inf"), "finite alpha"),
    ("dirichlet", True, "finite alpha"), ("label", 1.0, "must be one of"),
])
def test_a_partition_that_cannot_be_drawn_is_refused(partition, alpha, match):
    with pytest.raises(ValueError, match=match):
        P.partition_rows(LABELS, n_devices=6, seed=0, partition=partition, alpha=alpha)


# --------------------------------------------------------------------------- #
# 2. The task and its files
# --------------------------------------------------------------------------- #

def _task(**kw):
    return synthetic_task(n_devices=6, rows_per_device=60, test_rows=80, seed=11, **kw)


def _same_arrays(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b)) and len(a) == len(b)


def test_families_label_each_row_without_moving_one():
    plain, labelled = _task(), _task(families=True)
    for (X0, y0), (X1, y1), fam in zip(plain.device_shards, labelled.device_shards,
                                       labelled.device_families):
        assert np.array_equal(X0, X1) and np.array_equal(y0, y1)
        assert np.array_equal(fam == 0, y1 < 0.5)                  # Benign exactly for class 0
        assert set(fam[y1 >= 0.5].tolist()) <= set(range(1, 8))
    assert np.array_equal(plain.X_test, labelled.X_test)
    assert np.array_equal(labelled.family_test == 0, labelled.y_test < 0.5)
    assert plain.device_families is None and plain.family_test is None


def test_a_skewed_partition_recuts_the_iid_rows():
    iid = _task(families=True)
    skewed = _task(families=True, partition="dirichlet", alpha=0.1)
    assert np.array_equal(iid.X_test, skewed.X_test) and np.array_equal(iid.y_test, skewed.y_test)
    rows = lambda t: sorted(map(tuple, np.concatenate([X for X, _ in t.device_shards]).tolist()))
    assert rows(iid) == rows(skewed)
    assert [len(y) for _, y in skewed.device_shards] != [60] * 6
    for (X, y), fam in zip(skewed.device_shards, skewed.device_families):
        assert len(fam) == len(y) and np.array_equal(fam == 0, y < 0.5)
    binary = _task(partition="quantity", alpha=0.5)
    assert binary.device_families is None and len(binary.device_shards) == 6


def test_save_xy_stores_a_family_only_when_given(tmp_path):
    X, y = np.ones((3, 2)), np.array([0.0, 1.0, 1.0])
    save_xy(tmp_path / "plain.npz", X, y)
    with np.load(tmp_path / "plain.npz") as d:
        assert sorted(d.files) == ["X", "y"]
    assert load_family(tmp_path / "plain.npz") is None
    save_xy(tmp_path / "fam.npz", X, y, family=[0, 3, 5])
    assert load_family(tmp_path / "fam.npz").tolist() == [0, 3, 5]
    assert [a.tolist() for a in load_xy(tmp_path / "fam.npz")] == [X.tolist(), y.tolist()]
    with pytest.raises(ValueError, match="families"):
        save_xy(tmp_path / "bad.npz", X, y, family=[0, 1])


def test_prepare_trial_writes_the_families_and_refuses_an_empty_shard(tmp_path):
    task = _task(families=True)
    prep = prepare_trial(tmp_path / "p", task=task, theta_seed=1)
    assert load_family(prep.test_path).tolist() == task.family_test.tolist()
    assert load_family(prep.shard_paths[2]).tolist() == task.device_families[2].tolist()
    empty = CiciotTask(input_dim=46, device_shards=[(np.ones((2, 46)), np.ones(2)),
                                                     (np.zeros((0, 46)), np.zeros(0))],
                       X_test=np.ones((2, 46)), y_test=np.ones(2), is_synthetic=True)
    with pytest.raises(ValueError, match=r"shard\(s\) \[1\] hold no rows"):
        prepare_trial(tmp_path / "e", task=empty, theta_seed=1)


def test_the_canonical_loader_keeps_the_fine_label_only_when_asked():
    import inspect

    from Config.DatasetConfig.CICIOT2023_Sampling import ciciot2023DatasetLoadV2 as L

    sig = inspect.signature(L.load_and_balance_data_stratified)
    assert sig.parameters["keep_original_label"].default is False
    assert P.FAMILIES == ("Benign",) + tuple(dict.fromkeys(
        v for v in L.DICT_7CLASSES.values() if v != "Benign"))


# --------------------------------------------------------------------------- #
# 3. The detector's metrics
# --------------------------------------------------------------------------- #

def test_detection_metrics_by_hand():
    #        Benign x3      DDoS x2   Recon x3      Web x2
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    fam = np.array([0, 0, 0, 1, 1, 4, 4, 4, 6, 6])
    preds = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 1], dtype=np.float32)
    got = detection_metrics(y, preds, fam)
    assert (got["tp"], got["fp"], got["tn"], got["fn"]) == (5, 1, 2, 2)
    assert got["tpr"] == pytest.approx(5 / 7) and got["fpr"] == pytest.approx(1 / 3)
    assert got["precision"] == pytest.approx(5 / 6)
    assert got["f1"] == pytest.approx(2 * (5 / 7) * (5 / 6) / (5 / 7 + 5 / 6))
    assert got["recall_by_family"] == pytest.approx(
        {"Benign": 2 / 3, "DDoS": 1.0, "Recon": 1 / 3, "Web": 1.0})
    assert got["n_by_family"] == {"Benign": 3, "DDoS": 2, "Recon": 3, "Web": 2}
    none = detection_metrics(np.zeros(2), np.zeros(2), np.zeros(2))
    assert (none["tpr"], none["precision"], none["f1"], none["fpr"]) == (None, None, None, 0.0)
    with pytest.raises(ValueError, match="length"):
        detection_metrics(y, preds, fam[:-1])


class _Model:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float32)

    def predict(self, X, verbose=0):
        return self.probs.reshape(-1, 1)


def test_the_evaluation_adds_detection_only_with_the_families():
    X, y = np.zeros((4, 2)), np.array([0.0, 1.0, 1.0, 0.0])
    model = _Model([0.2, 0.9, 0.4, 0.6])
    plain = _eval_with_model(model, X, y)
    assert set(plain) == {"accuracy", "auc", "loss"} and plain["accuracy"] == 0.5
    with_fam = _eval_with_model(model, X, y, family=np.array([0, 2, 5, 0]))
    assert {k: with_fam[k] for k in plain} == plain
    assert with_fam["detection"]["recall_by_family"] == {"Benign": 0.5, "DoS": 1.0,
                                                          "Spoofing": 0.0}


def test_the_clusters_model_eval_adds_detection_only_with_the_families(monkeypatch):
    import experiments.exp4.model_task as model_task

    def fake(theta, X, y, input_dim, family=None):
        out = {"accuracy": 0.5, "auc": 0.5, "loss": 1.0}
        if family is not None:
            out["detection"] = {"tpr": 0.75, "recall_by_family": {"DDoS": 0.75}}
        return out

    monkeypatch.setattr(model_task, "evaluate_theta", fake)
    for family, expected in ((None, None), (np.array([0, 1]), {
            "tpr": 0.75, "recall_by_family": {"DDoS": 0.75}})):
        svc = CS._service(clock="sim")
        svc._eval_X, svc._eval_y, svc._eval_input_dim = np.zeros((2, 3)), np.zeros(2), 3
        svc._eval_family = family
        _dock, events = CS._run(svc, [CS._up("m1", 1, sim_ts=CS.T0 + 7.0, p_loss=0.0)])
        evals = CS._named(events, "model_eval")
        assert len(evals) == 2
        assert all(e.get("detection") == expected for e in evals)
        assert all(("detection" in e) == (family is not None) for e in evals)


def _eval_row(**detection):
    row = {"ts": 1.7e9, "event": "model_eval", "role": "cluster", "id": "c1",
           "cluster_round": 1, "accuracy": 0.9, "auc": 0.95, "loss": 0.2, "n_test": 10}
    if detection:
        row["detection"] = detection
    return row


def test_the_consumer_reads_detection_strictly():
    obs = observation_from_rows(cluster_rows=[_eval_row(), _eval_row(
        tp=5, fp=1, tn=2, fn=True, tpr=0.7, fpr=1.5, precision="0.8", f1=None,
        recall_by_family={"DDoS": 1.0, "Web": -1.0, "Recon": "x"}, n_by_family={"DDoS": 2})],
        mule_rows=[], device_rows=[], n_devices=1)
    plain, read = obs.model_evals
    assert plain.detection is None
    assert read.detection == Detection(tp=5, fp=1, tn=2, fn=None, tpr=0.7, fpr=None,
                                       precision=None, f1=None,
                                       recall_by_family={"DDoS": 1.0}, n_by_family={"DDoS": 2})


# --------------------------------------------------------------------------- #
# 4. The driver and the runner
# --------------------------------------------------------------------------- #

def test_the_driver_refuses_skew_or_families_without_the_real_model():
    for kw in (dict(partition="dirichlet", dirichlet_alpha=1.0), dict(family_labels=True)):
        with pytest.raises(ValueError, match="real_model"):
            Exp4Driver(**kw)
        Exp4Driver(real_model=True, data_source="synthetic", **kw)
    with pytest.raises(ValueError, match="finite alpha"):
        Exp4Driver(real_model=True, partition="quantity")
    with pytest.raises(ValueError, match="takes no alpha"):
        Exp4Driver(real_model=True, dirichlet_alpha=1.0)


def test_the_driver_builds_the_recorded_task_by_default():
    plain = Exp4Driver(real_model=True, data_source="synthetic")._build_task(6, 11)
    reference = synthetic_task(n_devices=6, rows_per_device=512, test_rows=512, seed=11)
    assert plain.device_families is None
    assert _same_arrays([X for X, _ in plain.device_shards],
                        [X for X, _ in reference.device_shards])
    skewed = Exp4Driver(real_model=True, data_source="synthetic", partition="dirichlet",
                        dirichlet_alpha=0.1, family_labels=True)._build_task(6, 11)
    assert skewed.device_families is not None
    assert [len(y) for _, y in skewed.device_shards] != [512] * 6


def test_the_marker_records_the_data_and_each_devices_shard(tmp_path, monkeypatch):
    """A family-labelled real-model trial with nothing spawned (the goldens'
    fake orchestrator): its kept trace's status marker holds ``data``, the
    settings and every device's shard rows; a default trial's marker has none."""
    from experiments.exp4 import driver as D
    from experiments.exp4.driver import TRIAL_STATUS_FILE, trace_dir_name
    from tests.golden import _build_topology as T

    seen = []
    real = D.prepare_trial

    def spy(prep_dir, *, task, theta_seed):
        seen.append([len(y) for _, y in task.device_shards])
        return real(prep_dir, task=task, theta_seed=theta_seed)

    monkeypatch.setattr(D, "prepare_trial", spy)
    common = dict(real_model=True, data_source="synthetic", synth_rows_per_device=40,
                  synth_test_rows=40)
    cell = T._cell("H1")
    markers = {}
    for name, kw in (("skewed", dict(partition="quantity", dirichlet_alpha=0.5,
                                     family_labels=True)), ("plain", {})):
        drv = Exp4Driver(**common, **kw, trace_root=tmp_path / name)
        T.run_stub_trial(drv, cell)
        assert drv._trial_data is None
        markers[name] = json.loads(
            (tmp_path / name / trace_dir_name(cell) / TRIAL_STATUS_FILE).read_text("utf-8"))
    assert "data" not in markers["plain"]
    data = markers["skewed"]["data"]
    assert (data["partition"], data["dirichlet_alpha"], data["family_labels"]) == (
        "quantity", 0.5, True)
    assert set(data["shard_rows"]) == {f"exp4-dev-{i:03d}" for i in range(6)}
    skewed_rows, plain_rows = seen
    assert sorted(data["shard_rows"].values()) == sorted(skewed_rows)
    assert len(set(skewed_rows)) > 1 and plain_rows == [40] * 6   # quantity skew


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


def test_the_runner_passes_the_data_settings_only_when_given(tmp_path, monkeypatch):
    base = ["--csv", str(tmp_path / "t.csv"), "--arms", "H1", "--real-model"]
    plain = _driver_kwargs(monkeypatch, base)
    assert not {"partition", "dirichlet_alpha", "family_labels"} & set(plain)
    skew = _driver_kwargs(monkeypatch, base + ["--partition", "dirichlet",
                                               "--dirichlet-alpha", "0.1", "--family-labels"])
    assert (skew["partition"], skew["dirichlet_alpha"], skew["family_labels"]) == (
        "dirichlet", 0.1, True)


# --------------------------------------------------------------------------- #
# 5. The scorer's detection columns
# --------------------------------------------------------------------------- #

E = SIM_EPOCH_S
DEVICES = ("a", "b")


def _mission(rnd, merged):
    return [{"ts": 1.7e9 + 10.0 * rnd, "event": "mission_started", "role": "mule", "id": "m1"},
            {"ts": 1.7e9 + 10.0 * rnd + 5.0, "event": "mission_completed", "role": "mule",
             "id": "m1", "mission_round": rnd, "pass_1_contacts": len(merged),
             "pass_2_contacts": 0, "pass_1_clean_devices": list(merged),
             "pass_1_merged_devices": list(merged), "pass_1_merged_updates": len(merged)}]


#: Two missions: a merged in both, b never: ages after each mission a 0, 0 and
#: b 1, 2, so the plain Network AoU is mean(0.5, 1.0) = 0.75, and weighted by
#: shard rows a 30, b 10 it is mean(0.25, 0.5) = 0.375.
MISSIONS = _mission(1, ["a"]) + _mission(2, ["a"])
DETECTION = dict(tp=8, fp=1, tn=9, fn=2, tpr=0.8, fpr=0.1, precision=8 / 9, f1=0.8421,
                 recall_by_family={"Benign": 0.9, "DDoS": 1.0, "Recon": 0.5},
                 n_by_family={"Benign": 10, "DDoS": 6, "Recon": 4})
MARKER = {"status": "ok", "data": {"partition": "quantity", "dirichlet_alpha": 0.5,
                                   "family_labels": True, "shard_rows": {"a": 30, "b": 10}}}


def _obs(evals):
    return observation_from_rows(cluster_rows=evals, mule_rows=[
        {"ts": 1.7e9, "event": "mule_ready", "role": "mule", "id": "m1"}] + MISSIONS,
        device_rows=[], n_devices=2)


def test_the_detection_columns_by_hand():
    obs = _obs([_eval_row(**dict(DETECTION, tpr=0.1)), dict(_eval_row(**DETECTION),
                                                             cluster_round=2)])
    got = detection_report(obs, DEVICES, MARKER)
    assert (got.data_partition, got.data_alpha) == ("quantity", 0.5)
    assert got.network_aou_shard_weighted_mean == pytest.approx(0.375)
    assert (got.tpr_final, got.fpr_final, got.f1_final) == (0.8, 0.1, 0.8421)
    assert got.precision_final == pytest.approx(8 / 9)
    assert got.recall_family_min == 0.5                          # Recon; Benign left out
    assert got.recall_by_family_final == DETECTION["recall_by_family"]


def test_a_recorded_trace_scores_blank():
    assert detection_report(_obs([_eval_row()]), DEVICES, {"status": "ok"}) == DetectionReport()
    partial = dict(MARKER, data=dict(MARKER["data"], shard_rows={"a": 30}))
    got = detection_report(_obs([]), DEVICES, partial)
    assert got.network_aou_shard_weighted_mean is None and got.data_partition == "quantity"


def _write_trace(root, marker):
    d = root / "N=2-regime=jittery-rrf=60.0__H1__t0__s42"
    d.mkdir(parents=True)
    files = {"cluster-c1.jsonl": [{"ts": 1.7e9 - 6, "event": "cluster_ready", "role": "cluster",
                                   "id": "c1"}, dict(_eval_row(**DETECTION), cluster_round=1)],
             "mule-m1.jsonl": [{"ts": 1.7e9, "event": "mule_ready", "role": "mule",
                                "id": "m1"}] + MISSIONS}
    for name, rows in files.items():
        (d / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (d / "mule-m1.json").write_text(json.dumps({"mule_id": "m1", "rf_range_m": 60.0,
                                                "n_missions": 2}), encoding="utf-8")
    (d / "cluster.json").write_text(json.dumps({"seed_devices": [
        {"device_id": dev, "position": [10.0, 0.0, 0.0]} for dev in DEVICES]}),
        encoding="utf-8")
    (d / "trial_status.json").write_text(json.dumps(marker), encoding="utf-8")
    return d


def test_the_columns_appear_only_when_asked_and_last(tmp_path):
    d = _write_trace(tmp_path, MARKER)
    plain = score_trial(d).to_row()
    assert not set(DETECTION_COLUMNS) & set(plain)
    row = score_trial(d, detection_columns=True).to_row()
    assert list(row) == list(plain) + list(DETECTION_COLUMNS)
    both = score_trial(d, cost_columns=True, detection_columns=True).to_row()
    assert list(both) == list(plain) + list(COST_COLUMNS) + list(DETECTION_COLUMNS)
    assert json.loads(row["recall_by_family_final"]) == DETECTION["recall_by_family"]
    out = tmp_path / "scored.csv"
    assert main(["--traces", str(tmp_path), "--csv", str(out), "--detection-columns"]) == 0
    with open(out, newline="", encoding="utf-8") as f:
        (scored,) = list(csv.DictReader(f))
    assert scored["data_partition"] == "quantity" and float(scored["tpr_final"]) == 0.8
