"""Device shards beyond IID: label and quantity skew (Exp 5 addendum, Study 5.13).

Every recorded trial split its training rows IID: an even partition of a
seeded permutation (``experiments.exp1.data_partition.partition_indices``), so
every device held the same class mix and nearly the same number of rows, and
a device's value to the merge was uniform. Study 5.13 asks whether FeRRy's age,
coverage and merge choices matter more when devices hold different data. The
partitions:

* ``iid`` (the default, Dirichlet alpha = infinity): the recorded partition,
  exactly (:func:`partition_rows` calls ``partition_indices``).
* ``dirichlet`` (label skew; Hsu, Qi and Brown 2019, the NIID-Bench
  convention): for each class c, the shares of its rows go to the devices by
  p_c ~ Dir(alpha * 1_N), each class's rows in a seeded order. Small alpha
  concentrates each class on few devices; alpha = 1 is moderate, 0.1 strong.
  The classes are the labels given: on CICIoT2023 the seven attack families
  and Benign (``FAMILIES``), so a device can hold mostly DDoS and another
  mostly Recon; on a binary label, attack against benign.
* ``quantity`` (quantity skew): every class mixed as in IID, but the shard
  sizes drawn from Dir(alpha * 1_N) over the devices.

**The empty-shard guard.** A device with no rows trains nothing and reports
zero metrics silently (``model_task.make_local_train_fn``), and a merge would
drop its updates as weightless. So every shard here holds at least
``min_rows`` rows: a draw that leaves a device short is drawn again (the
NIID-Bench rule), from the same seeded stream, up to :data:`MAX_DRAWS` times,
and a partition that still cannot fill every device is refused
(``ValueError``), never run with an empty shard.

Every draw is a function of the trial seed, the partition and alpha, so the
arms of a cell, which share the seed, hold the same shards (paired by
construction), and an alpha sweep shares the test set (the partition only
moves the training rows). numpy only.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Optional, Sequence

import numpy as np

from experiments.exp1.data_partition import partition_indices

__all__ = [
    "FAMILIES",
    "MAX_DRAWS",
    "PARTITIONS",
    "PARTITION_DIRICHLET",
    "PARTITION_IID",
    "PARTITION_QUANTITY",
    "check_partition",
    "partition_rows",
    "shard_summary",
]

PARTITION_IID = "iid"
PARTITION_DIRICHLET = "dirichlet"
PARTITION_QUANTITY = "quantity"
PARTITIONS = (PARTITION_IID, PARTITION_DIRICHLET, PARTITION_QUANTITY)

#: CICIoT2023's seven attack families and Benign, in label order (the legacy
#: loader's ``DICT_7CLASSES``; Benign = 0, as the binary label's benign is 0).
FAMILIES = ("Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce")

#: Draws of a skewed partition before a shard short of ``min_rows`` refuses it.
MAX_DRAWS = 1000


def check_partition(partition: str, alpha: Optional[float]) -> None:
    """Refuse a partition or alpha that cannot be drawn."""
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    if partition == PARTITION_IID:
        if alpha is not None:
            raise ValueError("the iid partition takes no alpha (it is Dirichlet alpha = "
                             "infinity, the recorded split)")
        return
    if alpha is None or isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (
            math.isfinite(float(alpha)) and float(alpha) > 0.0):
        raise ValueError(f"the {partition} partition needs a finite alpha > 0, got {alpha!r}")


def _rng(seed: int, partition: str, alpha: float, n_devices: int) -> np.random.Generator:
    payload = f"{seed}|partition|{partition}|{alpha!r}|{n_devices}"
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],
                                                "big"))


def _split(rows: np.ndarray, shares: np.ndarray) -> List[np.ndarray]:
    """``rows`` cut into consecutive pieces by ``shares`` (summing to 1)."""
    cuts = (np.cumsum(shares)[:-1] * len(rows)).astype(int)
    return np.split(rows, cuts)


def partition_rows(
    labels: Sequence,
    *,
    n_devices: int,
    seed: int,
    partition: str = PARTITION_IID,
    alpha: Optional[float] = None,
    min_rows: int = 1,
) -> List[np.ndarray]:
    """The row indices of each device's shard, ``n_devices`` lists.

    ``labels`` is one class label per row (any hashable values; the
    families, or the binary label). ``iid`` is ``partition_indices``'s even
    split, exactly; ``dirichlet`` and ``quantity`` are drawn as the module
    docstring says, every shard with at least ``min_rows`` rows. Within a
    shard the rows keep the order they were drawn in.
    """
    check_partition(partition, alpha)
    labels = np.asarray(labels)
    n = int(len(labels))
    if isinstance(n_devices, bool) or not isinstance(n_devices, int) or n_devices < 1:
        raise ValueError(f"n_devices must be an int >= 1, got {n_devices!r}")
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 0:
        raise ValueError(f"min_rows must be an int >= 0, got {min_rows!r}")
    if n < n_devices * min_rows:
        raise ValueError(f"{n} rows cannot give {n_devices} devices {min_rows} row(s) each")
    if partition == PARTITION_IID:
        shards = [np.asarray(idx, dtype=np.int64)
                  for idx in partition_indices(n, n_devices, seed=seed)]
        _guard(shards, min_rows, partition, alpha)
        return shards
    rng = _rng(seed, partition, float(alpha), n_devices)
    classes = sorted(set(labels.tolist()), key=repr)
    for _ in range(MAX_DRAWS):
        if partition == PARTITION_DIRICHLET:
            parts: List[List[np.ndarray]] = [[] for _ in range(n_devices)]
            for c in classes:
                rows = rng.permutation(np.flatnonzero(labels == c))
                shares = rng.dirichlet(np.full(n_devices, float(alpha)))
                for d, piece in enumerate(_split(rows, shares)):
                    parts[d].append(piece)
            shards = [np.concatenate(p).astype(np.int64) for p in parts]
        else:
            rows = rng.permutation(n)
            shares = rng.dirichlet(np.full(n_devices, float(alpha)))
            shards = [p.astype(np.int64) for p in _split(rows, shares)]
        if min(len(s) for s in shards) >= min_rows:
            return shards
    _guard(shards, min_rows, partition, alpha)
    return shards


def _guard(shards: Sequence[np.ndarray], min_rows: int, partition: str,
           alpha: Optional[float]) -> None:
    short = [d for d, s in enumerate(shards) if len(s) < min_rows]
    if short:
        raise ValueError(
            f"the {partition} partition (alpha {alpha}) leaves device(s) {short} with fewer "
            f"than {min_rows} row(s) after {MAX_DRAWS} draws: a device with an empty shard "
            f"trains nothing and reports zero metrics, so the trial is refused (more rows, "
            f"fewer devices or a larger alpha)")


def shard_summary(labels: Sequence, shards: Sequence[np.ndarray]) -> dict:
    """Per device, its rows and its class counts (for logs and tests)."""
    labels = np.asarray(labels)
    out = []
    for s in shards:
        values, counts = np.unique(labels[np.asarray(s, dtype=np.int64)], return_counts=True)
        out.append({"rows": int(len(s)),
                    "classes": {str(v): int(k) for v, k in zip(values.tolist(), counts)}})
    return {"devices": out}
