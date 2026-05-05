"""EX-1.2 — partition utility tests.

Pins the two contracts the rest of Exp 1 depends on:

* **Disjoint + complete:** every index in [0, N) appears in exactly
  one shard; shard sizes are within 1 of each other.
* **Deterministic:** same (dataset_size, n_clients, seed) reproduces
  the same shards across runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp1.data_partition import (
    PartitionSpec,
    make_partition_specs,
    materialize_shard,
    partition_indices,
)


# --------------------------------------------------------------------------- #
# partition_indices
# --------------------------------------------------------------------------- #

def test_partition_disjoint_and_complete():
    shards = partition_indices(100, 4, seed=42)
    all_indices = np.concatenate(shards)
    assert all_indices.size == 100
    assert sorted(all_indices.tolist()) == list(range(100))
    # Shard sizes within 1 of each other.
    sizes = sorted(s.size for s in shards)
    assert sizes[-1] - sizes[0] <= 1


def test_partition_deterministic_same_inputs():
    a = partition_indices(100, 4, seed=42)
    b = partition_indices(100, 4, seed=42)
    for sa, sb in zip(a, b):
        np.testing.assert_array_equal(sa, sb)


def test_partition_changes_with_seed():
    a = partition_indices(100, 4, seed=42)
    b = partition_indices(100, 4, seed=43)
    # At least one shard's contents differ.
    assert any(not np.array_equal(sa, sb) for sa, sb in zip(a, b))


def test_partition_uneven_split():
    """100 indices across 3 clients: shard sizes 34, 33, 33."""
    shards = partition_indices(100, 3, seed=0)
    sizes = sorted(s.size for s in shards)
    assert sizes == [33, 33, 34]


def test_partition_zero_size_dataset():
    """All shards empty when dataset_size == 0."""
    shards = partition_indices(0, 4, seed=42)
    assert len(shards) == 4
    assert all(s.size == 0 for s in shards)


def test_partition_rejects_negative_dataset_size():
    with pytest.raises(ValueError, match="dataset_size"):
        partition_indices(-1, 4, seed=42)


def test_partition_rejects_zero_clients():
    with pytest.raises(ValueError, match="n_clients"):
        partition_indices(100, 0, seed=42)


def test_partition_single_client_gets_everything():
    shards = partition_indices(50, 1, seed=42)
    assert len(shards) == 1
    assert shards[0].size == 50
    assert sorted(shards[0].tolist()) == list(range(50))


# --------------------------------------------------------------------------- #
# make_partition_specs
# --------------------------------------------------------------------------- #

def test_make_partition_specs_aligns_client_ids():
    specs = make_partition_specs(100, ["d1", "d2", "d3", "d4"], seed=42)
    assert [s.client_id for s in specs] == ["d1", "d2", "d3", "d4"]
    assert [s.partition_index for s in specs] == [0, 1, 2, 3]


def test_make_partition_specs_total_size_matches():
    specs = make_partition_specs(123, ["a", "b", "c"], seed=42)
    assert sum(s.size for s in specs) == 123


def test_make_partition_specs_rejects_empty_client_list():
    with pytest.raises(ValueError, match="at least one"):
        make_partition_specs(100, [], seed=42)


# --------------------------------------------------------------------------- #
# materialize_shard
# --------------------------------------------------------------------------- #

def test_materialize_with_target_bytes():
    spec = PartitionSpec(client_id="d1", partition_index=0, indices=np.arange(10))
    payload = materialize_shard(spec, target_bytes=1024)
    assert isinstance(payload, bytes)
    assert len(payload) == 1024


def test_materialize_with_source_array(tmp_path):
    source = np.arange(100, dtype=np.float32).reshape(100, 1)
    spec = PartitionSpec(
        client_id="d1", partition_index=0,
        indices=np.array([5, 7, 9, 11], dtype=np.int64),
    )
    payload = materialize_shard(spec, source=source)
    assert isinstance(payload, bytes)
    # Round-trip via numpy.load.
    import io
    arr = np.load(io.BytesIO(payload))
    np.testing.assert_array_equal(arr.flatten(), [5.0, 7.0, 9.0, 11.0])


def test_materialize_writes_to_disk(tmp_path):
    spec = PartitionSpec(client_id="d1", partition_index=0, indices=np.arange(0))
    out = tmp_path / "shards" / "d1.bin"
    payload = materialize_shard(spec, target_bytes=512, out_path=out)
    assert out.exists()
    assert out.read_bytes() == payload
    assert out.stat().st_size == 512


def test_materialize_rejects_both_source_and_target():
    spec = PartitionSpec(client_id="d1", partition_index=0, indices=np.arange(5))
    with pytest.raises(ValueError, match="exactly one"):
        materialize_shard(spec, source=np.zeros(10), target_bytes=1024)


def test_materialize_rejects_neither():
    spec = PartitionSpec(client_id="d1", partition_index=0, indices=np.arange(5))
    with pytest.raises(ValueError, match="either source or target_bytes"):
        materialize_shard(spec)


def test_materialize_rejects_negative_target_bytes():
    spec = PartitionSpec(client_id="d1", partition_index=0, indices=np.arange(5))
    with pytest.raises(ValueError, match="target_bytes"):
        materialize_shard(spec, target_bytes=-1)
