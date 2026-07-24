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
    # EX-4.2 realism wiring (all optional; omitted -> ideal links).
    device_reliability: bool = False,
    world_radius_m: float = 150.0,
    field_radius_m: Optional[float] = None,
    backhaul_loss_pct: float = 0.0,
    backhaul_rng_seed: Optional[int] = None,
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
        # field_radius_m (EX-4.2) spreads devices across the field so S3a
        # forms multiple contacts; otherwise the tight EX-4.0/4.1 cluster.
        spread_m = field_radius_m if field_radius_m is not None else min(rf_range_m * 0.4, 25.0)
    # Separate RNG so contact reliability doesn't correlate with placement.
    rel_rng = random.Random(seed ^ 0x5555AAAA)

    devices: List[DeviceConfig] = []
    for i in range(n_devices):
        x = rng.uniform(-spread_m, spread_m)
        y = rng.uniform(-spread_m, spread_m)
        contact_reliability: Optional[float] = None
        if device_reliability:
            # Exp 3's model: per-device reliability ~ U(0.15, 1.0) times a
            # distance falloff rf_factor = max(0.4, 1 - d/(3*world_radius)),
            # where d is the device's distance from the field centre (a
            # proxy for its distance to the mule's contact stop). Applies in
            # clean too, so H1 coverage is realistically < 1.0.
            reliability = rel_rng.uniform(0.15, 1.0)
            d = (float(x) ** 2 + float(y) ** 2) ** 0.5
            # The mule flies to *within* rf_range of the contact, so the
            # effective device<->mule distance is bounded by rf_range — Exp 3
            # uses device-to-contact-stop distance, not device-to-origin.
            # Without this cap, spread devices are over-penalised.
            d_eff = min(d, rf_range_m)
            rf_factor = max(0.4, 1.0 - d_eff / (3.0 * world_radius_m))
            contact_reliability = max(0.0, min(1.0, reliability * rf_factor))
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
                contact_reliability=contact_reliability,
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
        backhaul_loss_pct=backhaul_loss_pct,
        backhaul_rng_seed=backhaul_rng_seed,
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
