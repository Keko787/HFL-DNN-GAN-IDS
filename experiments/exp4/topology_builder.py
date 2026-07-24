"""Build a finite, natural-exit topology for one Experiment-4 trial.

EX-4.0 arm **H1**: 1 cluster + 1 mule + N devices, the mule capped at
``n_missions`` so the whole process tree exits on its own once the
missions are done (the driver then reads the JSONL logs). Devices are
placed in a tight cluster near the origin so the Four-Stage Gated
Scheduler's S3a range-clustering reliably forms at least one contact
event within ``rf_range_m`` — the point of EX-4.0 is to measure the real
two-pass path, not to stress contact formation (that is the sweep's job
once the plumbing is proven).

Positions are seeded off the trial's paired seed so the same
``(cell, trial_index)`` lays devices out identically across arms — the
paired-seed property the analysis relies on.
"""

from __future__ import annotations

import random
from typing import List, Optional

from hermes.processes import (
    ClusterConfig,
    DeviceConfig,
    MuleConfig,
    TopologyConfig,
)


def build_exp4_topology(
    *,
    n_devices: int,
    rf_range_m: float,
    n_missions: int,
    seed: int,
    spread_m: Optional[float] = None,
    session_ttl_s: float = 3.0,
    synth_batch_size: int = 2,
    min_participation: int = 1,
    cluster_id: str = "exp4-cluster",
    mule_id: str = "exp4-mule",
    # EX-4.1 real-model wiring (all optional; omitted -> EX-4.0 stub path).
    train_shard_paths: Optional[List[str]] = None,
    input_dim: Optional[int] = None,
    local_epochs: int = 1,
    local_batch_size: int = 64,
    init_theta_path: Optional[str] = None,
    eval_test_path: Optional[str] = None,
) -> TopologyConfig:
    """Return a validated :class:`TopologyConfig` for one H1 trial.

    ``spread_m`` bounds the square the devices are scattered in; it
    defaults to a fraction of ``rf_range_m`` (capped) so the cluster
    stays inside one contact radius.

    When ``train_shard_paths`` is given (EX-4.1 real-model path), device
    ``i`` is pointed at ``train_shard_paths[i]`` and the cluster is seeded
    from ``init_theta_path`` + scored on ``eval_test_path``.
    """
    if n_devices < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_devices}")
    if n_missions < 1:
        raise ValueError(f"n_missions must be >= 1, got {n_missions}")
    if train_shard_paths is not None and len(train_shard_paths) != n_devices:
        raise ValueError(
            f"train_shard_paths has {len(train_shard_paths)} entries, "
            f"expected n_devices={n_devices}"
        )

    rng = random.Random(seed)
    if spread_m is None:
        spread_m = min(rf_range_m * 0.4, 25.0)

    devices: List[DeviceConfig] = []
    for i in range(n_devices):
        x = rng.uniform(-spread_m, spread_m)
        y = rng.uniform(-spread_m, spread_m)
        devices.append(
            DeviceConfig(
                device_id=f"exp4-dev-{i:03d}",
                position=(float(x), float(y), 0.0),
                train_shard_path=(
                    train_shard_paths[i] if train_shard_paths else None
                ),
                input_dim=input_dim,
                local_epochs=local_epochs,
                local_batch_size=local_batch_size,
            )
        )

    cluster = ClusterConfig(
        cluster_id=cluster_id,
        dock_host="127.0.0.1",
        dock_port=0,
        synth_batch_size=synth_batch_size,
        min_participation=min_participation,
        init_theta_path=init_theta_path,
        eval_test_path=eval_test_path,
        input_dim=input_dim,
    )
    mule = MuleConfig(
        mule_id=mule_id,
        rf_host="127.0.0.1",
        rf_port=0,
        rf_range_m=float(rf_range_m),
        session_ttl_s=session_ttl_s,
        n_missions=int(n_missions),
    )
    topo = TopologyConfig(cluster=cluster, mules=[mule], devices=devices)
    topo.validate()
    return topo
