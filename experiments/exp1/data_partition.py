"""Deterministic N-way CICIOT-2023 partition.

Experiment 1 needs the same shard assignment every run so paired
trial seeds are meaningful: trial index 7 across the FL and Centralized
arms must see the *same* per-client data shards.

Two layers of determinism:

1. **Index partition**: given a dataset size and a partition seed, this
   module splits indices ``[0, N)`` into ``n_clients`` disjoint
   subsets. The split is reproducible across Python versions because
   it uses a SHA-256-seeded ``numpy.random.Generator`` rather than the
   stdlib ``random`` (which has historically changed its algorithm).

2. **Bytes-on-wire shape**: the experiment ships ``|D|pd`` bytes per
   client, where ``|D|pd`` is configurable. For correctness against
   the paper's wire-level metrics this module needs only to produce
   *index* sets — the actual bytes are filler at the network layer.
   When the operator wants to ship a real CICIOT shard (paper run on
   AERPAW), :func:`materialize_shard` serializes the partition to
   disk; the experiment client reads + ships those bytes.

Most dev runs use the index-only path. The materialization path is a
follow-up for paper-grade reproducibility on real hardware.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PartitionSpec:
    """One client's slice of the dataset."""

    client_id: str
    partition_index: int  # 0..n_clients-1
    indices: np.ndarray  # int64 indices into the source dataset

    @property
    def size(self) -> int:
        return int(self.indices.size)


def _seed_to_uint32(base_seed: int, *parts: object) -> int:
    """SHA-256-derived 32-bit seed.

    Cross-platform / cross-Python-version stable. Same inputs → same seed.
    Numpy's ``Generator`` accepts any non-negative int, but bounding to
    uint32 keeps the seed string compact in logs.
    """
    payload = "|".join([str(base_seed), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def partition_indices(
    dataset_size: int,
    n_clients: int,
    *,
    seed: int,
) -> List[np.ndarray]:
    """Split ``[0, dataset_size)`` into ``n_clients`` disjoint shards.

    Two contracts:

    * **Disjoint and complete:** every index appears in exactly one
      shard; no gaps.
    * **Deterministic:** same ``(dataset_size, n_clients, seed)`` →
      same shard contents (including row order within each shard).

    Shards are sized as evenly as possible — the first
    ``dataset_size % n_clients`` shards get one extra row.
    """
    if dataset_size < 0:
        raise ValueError(f"dataset_size must be ≥ 0, got {dataset_size}")
    if n_clients < 1:
        raise ValueError(f"n_clients must be ≥ 1, got {n_clients}")

    rng = np.random.default_rng(_seed_to_uint32(seed, "partition", dataset_size))
    permutation = rng.permutation(dataset_size)

    base, extra = divmod(dataset_size, n_clients)
    shards: List[np.ndarray] = []
    cursor = 0
    for i in range(n_clients):
        size = base + (1 if i < extra else 0)
        shards.append(permutation[cursor : cursor + size].copy())
        cursor += size
    assert cursor == dataset_size
    return shards


def make_partition_specs(
    dataset_size: int,
    client_ids: Sequence[str],
    *,
    seed: int,
) -> List[PartitionSpec]:
    """Higher-level wrapper: return one :class:`PartitionSpec` per client.

    ``client_ids`` order matches partition index order — ``client_ids[0]``
    gets partition 0, etc. This is the convention every other
    Experiment-1 module follows (``data_partition`` is an integer slot
    in the topology config).
    """
    if not client_ids:
        raise ValueError("client_ids must contain at least one entry")
    shards = partition_indices(dataset_size, len(client_ids), seed=seed)
    return [
        PartitionSpec(
            client_id=cid,
            partition_index=i,
            indices=shards[i],
        )
        for i, cid in enumerate(client_ids)
    ]


# --------------------------------------------------------------------------- #
# Materialization (paper-run path; index-only is fine for wire metrics)
# --------------------------------------------------------------------------- #

def materialize_shard(
    spec: PartitionSpec,
    *,
    source: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
    target_bytes: Optional[int] = None,
) -> bytes:
    """Produce the shard bytes the client will actually upload.

    Three modes, mutually exclusive:

    * ``source`` provided → take ``source[spec.indices]`` and serialize
      with ``numpy.save``. Bytes are real CICIOT data.
    * ``target_bytes`` provided → return ``target_bytes`` zero-filled
      bytes. Used in dev mode when no CICIOT data is loaded.
    * neither → raise.

    When ``out_path`` is set, the bytes are also written to disk so the
    client can ``mmap`` them at trial time without re-serializing.
    """
    if source is not None and target_bytes is not None:
        raise ValueError(
            "specify exactly one of source or target_bytes, not both"
        )
    if source is None and target_bytes is None:
        raise ValueError("specify either source or target_bytes")

    if source is not None:
        if spec.indices.size == 0:
            payload = b""
        else:
            slice_ = source[spec.indices]
            import io
            buf = io.BytesIO()
            np.save(buf, slice_, allow_pickle=False)
            payload = buf.getvalue()
    else:
        # target_bytes path
        assert target_bytes is not None
        if target_bytes < 0:
            raise ValueError(f"target_bytes must be ≥ 0, got {target_bytes}")
        payload = bytes(target_bytes)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(payload)
    return payload
