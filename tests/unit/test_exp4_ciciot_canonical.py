"""EX-4.1 — the paper-faithful (canonical) CICIOT loader.

Verifies ``load_ciciot_task_canonical`` reuses the production balancing +
preprocessing correctly: 21 features (46 minus the 25 IRRELEVANT_FEATURES),
Benign=0/Attack=1, MinMax-scaled, balanced, partitioned across devices.

Skipped when the CICIOT-2023 dataset is not present (so CI without the
13 GB corpus stays green); marked ``slow`` because it reads + balances real
CSV parts.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp4.model_task import (
    default_ciciot_dir,
    load_ciciot_task_canonical,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        default_ciciot_dir() is None,
        reason="CICIOT-2023 dataset not present (set HERMES_CICIOT_DIR)",
    ),
]


def test_canonical_loader_shapes_balance_and_scaling():
    task = load_ciciot_task_canonical(
        n_devices=4,
        seed=42,
        train_files=2,
        test_files=1,
        train_dataset_size=8000,
        test_dataset_size=3000,
    )
    # Canonical feature-selection drops 25 of 46 features -> 21.
    assert task.input_dim == 21
    assert not task.is_synthetic
    assert task.n_devices == 4
    assert task.X_test.shape[1] == 21
    assert len(task.feature_names) == 21

    # Disjoint, complete device shards over the balanced train pool.
    sizes = [len(y) for _, y in task.device_shards]
    assert min(sizes) > 0
    assert sum(sizes) == task.n_train

    # Labels are binary; the balanced source keeps both classes present.
    all_y = np.concatenate([y for _, y in task.device_shards])
    assert set(np.unique(all_y)).issubset({0.0, 1.0})
    assert 0.30 < float(all_y.mean()) < 0.70, "train pool should be ~balanced"

    # MinMax[0,1] scaling: train shard features sit within [0, 1].
    X0 = task.device_shards[0][0]
    assert X0.min() >= -1e-4
    assert X0.max() <= 1.0 + 1e-4
