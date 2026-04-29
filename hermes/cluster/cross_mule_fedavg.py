"""Cross-mule FedAvg.

Cluster-scope merge of mission-scope ``PartialAggregate``s. ``N`` is the
number of *mules* (small) — not the number of devices (large). Each
partial is already a weighted intra-mission FedAvg done by
``HFLHostMission``, so the cluster-level math is simply a second weighted
average using ``num_examples`` as the weight.

Pulled out of ``HFLHostCluster`` so it can be unit-tested against
hand-computed references without spinning up the whole cluster server.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from hermes.types import PartialAggregate
from hermes.types.aggregate import Weights


class FedAvgError(ValueError):
    """Raised when partials cannot be merged (shape mismatch, all empty, etc.)."""


def cross_mule_fedavg(partials: Sequence[PartialAggregate]) -> Weights:
    """Weighted FedAvg over mission-scope partials.

    Args:
        partials: at least one non-empty ``PartialAggregate``.

    Returns:
        A new ``Weights`` list (numpy arrays, one per layer/parameter).

    Raises:
        FedAvgError: if no non-empty partials are provided or layer shapes
            disagree across partials.
    """
    non_empty = [p for p in partials if not p.is_empty()]
    if not non_empty:
        raise FedAvgError("cross_mule_fedavg requires at least one non-empty partial")

    # Shape check — every partial must have the same #layers and per-layer
    # shape. Mismatch is a programmer error, not a runtime corner case.
    n_layers = len(non_empty[0].weights)
    layer_shapes = [w.shape for w in non_empty[0].weights]
    for p in non_empty[1:]:
        if len(p.weights) != n_layers:
            raise FedAvgError(
                f"layer count mismatch: partial from {p.mule_id!r} has "
                f"{len(p.weights)} layers; expected {n_layers}"
            )
        for i, w in enumerate(p.weights):
            if w.shape != layer_shapes[i]:
                raise FedAvgError(
                    f"layer {i} shape mismatch: {w.shape} vs {layer_shapes[i]} "
                    f"(mule={p.mule_id!r})"
                )

    total_examples = sum(p.num_examples for p in non_empty)
    if total_examples == 0:
        raise FedAvgError("cross_mule_fedavg total num_examples is zero")

    merged: List[np.ndarray] = []
    for layer_idx in range(n_layers):
        # accumulate as float64 to avoid drift on big sums, then cast back
        acc = np.zeros(layer_shapes[layer_idx], dtype=np.float64)
        for p in non_empty:
            acc += p.weights[layer_idx].astype(np.float64) * p.num_examples
        acc /= total_examples
        # match the dtype of the first partial's layer for downstream callers
        merged.append(acc.astype(non_empty[0].weights[layer_idx].dtype))

    return merged
