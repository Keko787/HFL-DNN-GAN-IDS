"""Cross-mule FedAvg.

Cluster-scope merge of mission-scope ``PartialAggregate``s. ``N`` is the
number of *mules* (small) — not the number of devices (large). Each
partial is already a weighted intra-mission FedAvg done by
``HFLHostMission``, so the cluster-level math is simply a second weighted
average using ``num_examples`` as the weight.

Pulled out of ``HFLHostCluster`` so it can be unit-tested against
hand-computed references without spinning up the whole cluster server.

:func:`apply_weighted_deltas` is the age-aware counterpart (FeRRy Phase 1):
partials carry updates rather than models, and the cluster adds their
staleness-weighted sum over the live partials' mass to θ at a server rate η.
The rules that feed it live in ``hermes.mission.aggregation_rules``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

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


def apply_weighted_deltas(
    theta: Weights,
    partials: Sequence[PartialAggregate],
    partial_weights: Sequence[float],
    *,
    server_lr: float = 1.0,
    normalizer: Optional[float] = None,
) -> Weights:
    """θ + η · Σ_m W_m·Δ_m / N over delta-form partials.

    FeRRy Phase 1: the age-aware rules mix the merged update into the global θ
    at a server rate η instead of overwriting it. ``partial_weights`` are the
    W_m the rule computed, M_m·s_m (the partial's staleness-free mass times its
    staleness); a partial with W_m = 0 contributes nothing to the sum.

    ``normalizer`` is N. The cluster passes Σ M_m over the live partials, so a
    stale partial's s_m < 1 shrinks the step instead of cancelling out: one
    partial gives θ + η·s_m·Δ_m. None divides by Σ W_m, the weights' own sum,
    which gives a weighted mean of the Δ_m and normalises staleness away.

    Raises:
        FedAvgError: every W_m is zero, ``normalizer`` is not positive, or
            shapes disagree with θ.
    """
    if len(partial_weights) != len(partials):
        raise FedAvgError(
            f"{len(partial_weights)} weights for {len(partials)} partials"
        )
    live = [
        (p, float(w)) for p, w in zip(partials, partial_weights)
        if w > 0.0 and not p.is_empty()
    ]
    total = sum(w for _, w in live)
    if not live or not total > 0.0:
        raise FedAvgError("apply_weighted_deltas: no partial carries weight")
    if normalizer is not None:
        if not float(normalizer) > 0.0:
            raise FedAvgError(
                f"apply_weighted_deltas: normalizer must be > 0, got {normalizer}"
            )
        total = float(normalizer)

    layer_shapes = [np.shape(t) for t in theta]
    for p, _ in live:
        if len(p.weights) != len(layer_shapes):
            raise FedAvgError(
                f"layer count mismatch: partial from {p.mule_id!r} has "
                f"{len(p.weights)} layers; θ has {len(layer_shapes)}"
            )
        for i, w in enumerate(p.weights):
            if w.shape != layer_shapes[i]:
                raise FedAvgError(
                    f"layer {i} shape mismatch: {w.shape} vs {layer_shapes[i]} "
                    f"(mule={p.mule_id!r})"
                )

    out: List[np.ndarray] = []
    for layer_idx, base in enumerate(theta):
        acc = np.zeros(layer_shapes[layer_idx], dtype=np.float64)
        for p, w in live:
            acc += p.weights[layer_idx].astype(np.float64) * (w / total)
        updated = np.asarray(base, dtype=np.float64) + server_lr * acc
        out.append(updated.astype(np.asarray(base).dtype))
    return out
