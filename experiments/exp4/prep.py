"""Per-trial data/model preparation for the real-model path (EX-4.1).

"Driver-prepares-once": the driver builds the :class:`CiciotTask` a single
time per trial, serializes each device's shard, the shared held-out test
set, and the real DNN-IDS seed weights to a prep directory, and hands the
subprocesses their paths via config. This keeps the heavy canonical load +
the paired-seed determinism in one place, instead of every device
subprocess re-loading the 13 GB corpus independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .model_task import CiciotTask, initial_theta, save_weights, save_xy


@dataclass(frozen=True)
class TrialPrep:
    """Paths the topology hands to the subprocesses for one real-model trial."""

    input_dim: int
    shard_paths: List[str]   # index i -> device i's (X, y) shard
    test_path: str           # shared held-out eval set (cluster reads it)
    init_theta_path: str     # real DNN-IDS seed weights (cluster broadcasts)
    is_synthetic: bool
    n_train: int


def prepare_trial(prep_dir, *, task: CiciotTask, theta_seed: int) -> TrialPrep:
    """Serialize one trial's shards + test set + seed weights to ``prep_dir``.

    ``theta_seed`` seeds the deterministic initial model so every arm in a
    paired cell starts from the same global θ.

    Exp 5 addendum (Study 5.13): a task with families writes each row's
    family beside it (``save_xy``'s ``family``: the shards, which the devices
    never read, and the test set, from which the cluster's evaluation adds
    the detection metrics); a task without them writes the recorded files.
    A shard with no rows is refused: its device would train nothing and
    report zero metrics silently.
    """
    prep_dir = Path(prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)

    empty = [i for i, (_X, y) in enumerate(task.device_shards) if len(y) == 0]
    if empty:
        raise ValueError(
            f"device shard(s) {empty} hold no rows: such a device trains nothing and reports "
            f"zero metrics; give the trial more rows or fewer devices")
    shard_paths: List[str] = []
    for i, (X, y) in enumerate(task.device_shards):
        p = prep_dir / f"shard-{i:03d}.npz"
        family = None if task.device_families is None else task.device_families[i]
        save_xy(p, X, y, family=family)
        shard_paths.append(str(p))

    test_path = prep_dir / "test.npz"
    save_xy(test_path, task.X_test, task.y_test, family=task.family_test)

    theta_path = prep_dir / "theta_init.npz"
    save_weights(theta_path, initial_theta(task.input_dim, seed=theta_seed))

    return TrialPrep(
        input_dim=task.input_dim,
        shard_paths=shard_paths,
        test_path=str(test_path),
        init_theta_path=str(theta_path),
        is_synthetic=task.is_synthetic,
        n_train=task.n_train,
    )
