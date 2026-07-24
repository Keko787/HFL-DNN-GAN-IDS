"""Real DNN-IDS learning task for Experiment 4 (chunk EX-4.1 foundation).

This is the decision-independent core of "put the real DNN-IDS in the
loop": everything needed to train and evaluate the canonical CICIOT-2023
classifier as the federated model, decoupled from *where* it runs (the
in-process unit test vs the multi-process orchestrator wiring, which is
EX-4.1's heavier second step).

Pieces
------
* :func:`build_ids_model` — the canonical ``create_CICIOT_Model`` (5-dense
  stack ``64->32->16->8->4->1``, sigmoid), compiled for binary IDS.
* :func:`initial_theta` — deterministic seed weights so the cluster can
  broadcast a *real* global model instead of the 13-param stub, and every
  device trains from the same θ (``partial_fedavg`` requires identical
  layer shapes across all submissions).
* :func:`make_local_train_fn` — a ``local_train(theta, synth)`` callable
  matching :class:`hermes.mission.LocalTrainResult`; it sets θ, fits a few
  local epochs on the device shard, and returns the post-training weights
  (FedAvg averages *weights*, so ``delta_theta`` carries the full model).
* :func:`evaluate_theta` — accuracy / AUC / loss of an aggregated θ on the
  held-out test set. This is the per-round convergence signal EX-4.1 emits.
* :func:`load_ciciot_task` — deterministic per-device shards + a shared
  held-out test set from the real CICIOT CSVs, via the EX-1.2
  :func:`~experiments.exp1.data_partition.partition_indices`. Falls back to
  :func:`synthetic_task` (real-shaped Gaussian blobs) when the dataset is
  not present, so tests and dev runs work on any machine.

TensorFlow is imported lazily inside the functions that need it, so the
data-loading path (numpy/pandas only) stays importable in environments
without a GPU/TF warm-up cost.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from experiments.exp1.data_partition import partition_indices

log = logging.getLogger("experiments.exp4.model_task")

# CICIOT-2023 (this repo's parquet-exported CSVs): 46 numeric feature
# columns + a string ``label`` column. The DNN-IDS is a *binary*
# classifier — benign vs. attack — so the multi-class label is collapsed.
CICIOT_LABEL_COL = "label"
CICIOT_BENIGN_LABEL = "BenignTraffic"
CICIOT_FEATURE_COUNT = 46
INPUT_DIM = CICIOT_FEATURE_COUNT


Weights = List[np.ndarray]
# local_train(theta, synth_batch) -> LocalTrainResult  (matches ClientMission)
LocalTrainFn = Callable[[Weights, Sequence[np.ndarray]], "object"]


# --------------------------------------------------------------------------- #
# Task container
# --------------------------------------------------------------------------- #

@dataclass
class CiciotTask:
    """A partitioned binary-IDS task: per-device train shards + held-out test."""

    input_dim: int
    device_shards: List[Tuple[np.ndarray, np.ndarray]]  # (X, y) per device, float32
    X_test: np.ndarray
    y_test: np.ndarray
    is_synthetic: bool
    feature_names: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_devices(self) -> int:
        return len(self.device_shards)

    @property
    def n_train(self) -> int:
        return int(sum(len(y) for _, y in self.device_shards))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_ids_model(
    input_dim: int = INPUT_DIM,
    *,
    l2_alpha: float = 1e-3,
    learning_rate: float = 1e-3,
    regularization: bool = True,
):
    """Return the compiled canonical CICIOT DNN-IDS (binary classifier)."""
    import tensorflow as tf  # lazy — heavy import

    from Config.modelStructures.NIDS.NIDS_Struct import create_CICIOT_Model

    model = create_CICIOT_Model(
        input_dim=input_dim,
        regularizationEnabled=regularization,
        DP_enabled=False,
        l2_alpha=l2_alpha,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def initial_theta(input_dim: int = INPUT_DIM, *, seed: int = 0) -> Weights:
    """Deterministic seed weights for the global model.

    Every device and the cluster's broadcast θ share these shapes; that
    is a hard requirement of ``partial_fedavg`` (it rejects mismatched
    layer counts / shapes). Seeded so the same ``(input_dim, seed)``
    reproduces the same initial model across runs / processes.
    """
    import tensorflow as tf  # lazy

    tf.keras.utils.set_random_seed(int(seed))
    model = build_ids_model(input_dim)
    return [w.copy() for w in model.get_weights()]


# --------------------------------------------------------------------------- #
# Local training + evaluation
# --------------------------------------------------------------------------- #

def make_local_train_fn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    input_dim: Optional[int] = None,
    epochs: int = 1,
    batch_size: int = 64,
    l2_alpha: float = 1e-3,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> LocalTrainFn:
    """Build a ``local_train(theta, synth)`` callable over one device shard.

    The model is built **once** (closure) and re-fitted each round from the
    pushed θ — building a fresh Keras model per round would dominate
    runtime. ``synth_batch`` (the GAN synthetic augmentation) is accepted
    for signature compatibility with :class:`hermes.mission.ClientMission`
    but ignored in EX-4.1 (the GAN is separate, revision-scope work).
    """
    import tensorflow as tf  # lazy

    from hermes.mission import LocalTrainResult

    dim = int(input_dim if input_dim is not None else X.shape[1])
    tf.keras.utils.set_random_seed(int(seed))
    model = build_ids_model(dim, l2_alpha=l2_alpha, learning_rate=learning_rate)

    Xf = np.asarray(X, dtype=np.float32)
    yf = np.asarray(y, dtype=np.float32).reshape(-1)
    n = int(len(yf))

    def _local_train(theta: Weights, synth_batch) -> "LocalTrainResult":
        model.set_weights(theta)
        if n > 0:
            model.fit(
                Xf, yf,
                epochs=epochs,
                batch_size=min(batch_size, max(1, n)),
                verbose=0,
                shuffle=True,
            )
        theta_after = [w.copy() for w in model.get_weights()]
        metrics = _eval_with_model(model, Xf, yf) if n > 0 else {
            "accuracy": 0.0, "auc": 0.0, "loss": 0.0,
        }
        return LocalTrainResult(
            delta_theta=theta_after,          # FedAvg averages full weights
            num_examples=n,
            accuracy=float(metrics["accuracy"]),
            auc=float(metrics["auc"]),
            loss=float(metrics["loss"]),
            theta_after=theta_after,
        )

    return _local_train


def evaluate_theta(
    theta: Weights,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    input_dim: Optional[int] = None,
    l2_alpha: float = 1e-3,
) -> dict:
    """Accuracy / AUC / binary-cross-entropy of aggregated θ on held-out data.

    This is the per-round convergence signal for EX-4.1 — evaluated after
    the cluster's cross-mule FedAvg produces θ' each round.
    """
    dim = int(input_dim if input_dim is not None else X_test.shape[1])
    model = build_ids_model(dim, l2_alpha=l2_alpha)
    model.set_weights(theta)
    return _eval_with_model(
        model,
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32).reshape(-1),
    )


def _eval_with_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Predict-based metrics — deterministic, independent of stateful
    Keras metric objects (BatchNorm/Dropout are inference-mode here)."""
    if len(y) == 0:
        return {"accuracy": 0.0, "auc": 0.0, "loss": 0.0}
    probs = model.predict(X, verbose=0).reshape(-1)
    eps = 1e-7
    p = np.clip(probs, eps, 1.0 - eps)
    loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    preds = (probs >= 0.5).astype(np.float32)
    accuracy = float(np.mean(preds == y))
    auc = _safe_auc(y, probs)
    return {"accuracy": accuracy, "auc": auc, "loss": loss}


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC, guarding the single-class case (undefined → 0.5)."""
    classes = np.unique(y_true)
    if classes.size < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:  # pragma: no cover — sklearn always present here
        return 0.5


# --------------------------------------------------------------------------- #
# Data loading — real CICIOT with a synthetic fallback
# --------------------------------------------------------------------------- #

def default_ciciot_dir() -> Optional[Path]:
    """Best-effort locate the CICIOT-2023 CSV directory.

    In this repo the dataset lives one level above the project root
    (``../datasets/CICIOT2023``, gitignored). Returns ``None`` if not
    found — callers fall back to :func:`synthetic_task`.
    """
    here = Path(__file__).resolve()
    # experiments/exp4/model_task.py -> repo root is parents[2]
    repo_root = here.parents[2]
    candidates = [
        repo_root.parent / "datasets" / "CICIOT2023",
        repo_root / "datasets" / "CICIOT2023",
    ]
    env = os.environ.get("HERMES_CICIOT_DIR")
    if env:
        candidates.insert(0, Path(env))
    for c in candidates:
        if c.is_dir() and any(c.glob("*.csv")):
            return c
    return None


def load_ciciot_task(
    *,
    n_devices: int,
    rows_per_device: int,
    test_rows: int,
    seed: int,
    data_dir: Optional[Path] = None,
    max_source_files: int = 2,
) -> CiciotTask:
    """Deterministic per-device CICIOT shards + a shared held-out test set.

    Reads only as many rows as needed (``n_devices * rows_per_device +
    test_rows``) from the first ``max_source_files`` CSV parts, binarizes
    the label (benign=0, attack=1), standard-scales the 46 features on the
    pooled training rows, and partitions the train rows across devices with
    the EX-1.2 deterministic index partition. Falls back to
    :func:`synthetic_task` when the dataset is unavailable.
    """
    data_dir = data_dir or default_ciciot_dir()
    if data_dir is None:
        log.warning(
            "CICIOT-2023 not found (set HERMES_CICIOT_DIR); "
            "falling back to a synthetic real-shaped task"
        )
        return synthetic_task(
            n_devices=n_devices, rows_per_device=rows_per_device,
            test_rows=test_rows, seed=seed,
        )

    import pandas as pd

    need = n_devices * rows_per_device + test_rows
    frames: List["pd.DataFrame"] = []
    got = 0
    for csv in sorted(Path(data_dir).glob("*.csv"))[:max_source_files]:
        remaining = need - got
        if remaining <= 0:
            break
        df = pd.read_csv(csv, nrows=remaining)
        frames.append(df)
        got += len(df)
    df = pd.concat(frames, ignore_index=True)

    # Deterministic shuffle of the pooled rows before the train/test split.
    rng = np.random.default_rng(_u32(seed, "ciciot", need))
    perm = rng.permutation(len(df))
    df = df.iloc[perm].reset_index(drop=True)

    feat_cols = [c for c in df.columns if c != CICIOT_LABEL_COL]
    X_all = df[feat_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float32)
    y_all = (
        df[CICIOT_LABEL_COL].astype(str) != CICIOT_BENIGN_LABEL
    ).to_numpy(dtype=np.float32)

    n_total = len(y_all)
    n_test = min(test_rows, max(0, n_total - n_devices))
    X_test, y_test = X_all[:n_test], y_all[:n_test]
    X_train, y_train = X_all[n_test:], y_all[n_test:]

    # Standard-scale on the training rows; apply to both.
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    shards = _partition_rows(X_train, y_train, n_devices=n_devices, seed=seed)
    return CiciotTask(
        input_dim=X_all.shape[1],
        device_shards=shards,
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        is_synthetic=False,
        feature_names=tuple(feat_cols),
    )


def load_ciciot_task_canonical(
    *,
    n_devices: int,
    seed: int,
    data_dir: Optional[Path] = None,
    train_files: int = 3,
    test_files: int = 1,
    train_dataset_size: int = 20000,
    test_dataset_size: int = 8000,
    attack_eval_ratio: float = 0.5,
) -> CiciotTask:
    """Paper-faithful CICIOT task via the **production** pipeline.

    Reuses the exact canonical logic the standalone DNN-IDS was trained on:
    ``load_and_balance_data_stratified`` (50/50 Attack/Benign stratified
    undersampling, ``DICT_2CLASSES`` label collapse) + the real
    ``preprocess_dataset`` (drops the 25 ``IRRELEVANT_FEATURES`` → 21
    features, Benign=0/Attack=1 encoding, MinMax[0,1] scaling, stratified
    train/val split). Only the file-discovery loop is re-implemented so the
    dataset directory is configurable — the canonical ``loadCICIOT``
    hardcodes a relative path and takes no directory argument.

    The balanced train rows (train+val recombined) are partitioned across
    ``n_devices`` with the EX-1.2 deterministic index partition; the
    canonical held-out test split is the shared eval set. Falls back to
    :func:`synthetic_task` when the dataset is unavailable.
    """
    data_dir = data_dir or default_ciciot_dir()
    if data_dir is None:
        log.warning(
            "CICIOT-2023 not found (set HERMES_CICIOT_DIR); canonical "
            "loader falling back to a synthetic real-shaped task"
        )
        return synthetic_task(
            n_devices=n_devices, rows_per_device=2000, test_rows=2000, seed=seed,
        )

    import contextlib
    import io
    import random as _random

    import numpy as _np
    import pandas as pd

    # The fidelity-defining canonical helpers — reused verbatim.
    from Config.DatasetConfig.CICIOT2023_Sampling.ciciot2023DatasetLoadV2 import (
        DICT_2CLASSES,
        IRRELEVANT_FEATURES,
        load_and_balance_data_stratified,
        reduce_attack_samples,
    )
    from Config.DatasetConfig.Dataset_Preprocessing.datasetPreprocess import (
        preprocess_dataset,
    )

    csvs = sorted(str(p) for p in Path(data_dir).glob("*.csv"))
    if len(csvs) < train_files + test_files:
        raise ValueError(
            f"need >= {train_files + test_files} CICIOT csv files, "
            f"found {len(csvs)} in {data_dir}"
        )
    # Deterministic disjoint train/test file selection (mirrors loadCICIOT).
    rs = _random.Random(seed)
    tr_files = rs.sample(csvs, train_files)
    te_files = rs.sample([c for c in csvs if c not in tr_files], test_files)

    benign_train_limit = train_dataset_size // 2
    benign_test_limit = test_dataset_size // 2

    def _load_balanced(files, benign_limit):
        pool = pd.DataFrame()
        benign = 0
        for f in files:
            if benign >= benign_limit:
                break
            df, bc = load_and_balance_data_stratified(
                f, DICT_2CLASSES, benign, benign_limit, verbose=False,
            )
            pool = pd.concat([pool, df])
            benign += bc
        return pool

    train_df = _load_balanced(tr_files, benign_train_limit)
    test_df = _load_balanced(te_files, benign_test_limit)
    test_df = reduce_attack_samples(test_df, attack_eval_ratio)

    # The canonical preprocess is chatty — silence its prints.
    with contextlib.redirect_stdout(io.StringIO()):
        X_tr, X_val, y_tr, y_val, X_te, y_te = preprocess_dataset(
            "CICIOT", train_df, test_df,
            irrelevant_features_ciciot=list(IRRELEVANT_FEATURES),
        )

    X_train = pd.concat([X_tr, X_val]).to_numpy(dtype=_np.float32)
    y_train = pd.concat([y_tr, y_val]).to_numpy(dtype=_np.float32).reshape(-1)
    X_test = X_te.to_numpy(dtype=_np.float32)
    y_test = y_te.to_numpy(dtype=_np.float32).reshape(-1)

    shards = _partition_rows(X_train, y_train, n_devices=n_devices, seed=seed)
    return CiciotTask(
        input_dim=int(X_train.shape[1]),
        device_shards=shards,
        X_test=X_test,
        y_test=y_test,
        is_synthetic=False,
        feature_names=tuple(X_tr.columns),
    )


def synthetic_task(
    *,
    n_devices: int,
    rows_per_device: int,
    test_rows: int,
    seed: int,
    input_dim: int = INPUT_DIM,
    class_sep: float = 1.5,
) -> CiciotTask:
    """Real-shaped, linearly-separable two-class Gaussian task.

    Deterministic and CPU-cheap so unit tests / dev runs exercise the full
    training + FedAvg + eval path without the 13 GB dataset. The two
    classes are Gaussian blobs whose means differ by ``±class_sep`` in
    *every* feature (a random sign per feature), so the signal is spread
    across all ``input_dim`` dimensions — strongly separable even for the
    heavily-regularized DNN-IDS (Dropout 0.4 on five layers). Balanced.
    """
    rng = np.random.default_rng(_u32(seed, "synthetic", input_dim))
    # Per-feature class-mean offset: a random ±class_sep in each dimension.
    center = rng.choice((-1.0, 1.0), size=input_dim).astype(np.float32) * float(class_sep)

    def _make(n: int) -> Tuple[np.ndarray, np.ndarray]:
        y = (rng.random(n) < 0.5).astype(np.float32)
        X = rng.normal(size=(n, input_dim)).astype(np.float32)
        # Class 1 sits near +center, class 0 near -center.
        X += np.outer((y * 2.0 - 1.0), center).astype(np.float32)
        return X, y

    shards: List[Tuple[np.ndarray, np.ndarray]] = [
        _make(rows_per_device) for _ in range(n_devices)
    ]
    X_test, y_test = _make(test_rows)
    return CiciotTask(
        input_dim=input_dim,
        device_shards=shards,
        X_test=X_test,
        y_test=y_test,
        is_synthetic=True,
        feature_names=tuple(f"f{i}" for i in range(input_dim)),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _partition_rows(
    X: np.ndarray, y: np.ndarray, *, n_devices: int, seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Deterministic disjoint per-device shards via the EX-1.2 partition."""
    shards_idx = partition_indices(len(y), n_devices, seed=seed)
    return [(X[idx], y[idx]) for idx in shards_idx]


def _u32(base_seed: int, *parts) -> int:
    import hashlib
    payload = "|".join([str(base_seed), *(str(p) for p in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


# --------------------------------------------------------------------------- #
# (De)serialization for the driver -> subprocess handoff (numpy .npz only)
# --------------------------------------------------------------------------- #
# These read/write plain numeric ndarray archives. np.load defaults to
# refusing object arrays, so the handoff is data-only (no code execution).

def save_weights(path, weights: Weights) -> None:
    """Serialize a list of ndarrays (a model's ``get_weights()``) to ``.npz``."""
    np.savez(
        str(path),
        _n=np.array(len(weights), dtype=np.int64),
        **{f"w{i}": np.asarray(w) for i, w in enumerate(weights)},
    )


def load_weights(path) -> Weights:
    """Inverse of :func:`save_weights` — preserves layer order."""
    with np.load(str(path)) as d:
        n = int(d["_n"])
        return [np.array(d[f"w{i}"]) for i in range(n)]


def save_xy(path, X: np.ndarray, y: np.ndarray) -> None:
    """Serialize an ``(X, y)`` shard / test set to ``.npz`` (float32)."""
    np.savez(
        str(path),
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32).reshape(-1),
    )


def load_xy(path) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`save_xy`."""
    with np.load(str(path)) as d:
        return np.array(d["X"], dtype=np.float32), np.array(d["y"], dtype=np.float32)
