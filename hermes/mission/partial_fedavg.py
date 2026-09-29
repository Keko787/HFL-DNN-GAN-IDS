"""Mission-scope partial FedAvg on the mule.

Aggregates per-device ``GradientSubmission``s collected during one
in-field mission round into a single ``PartialAggregate`` that the mule
uploads at dock. Weights are ``num_examples``.

Kept as a pure function (mirrors ``cluster.cross_mule_fedavg``) so it's
directly unit-testable against hand-computed references.

Design ref: HERMES_FL_Scheduler_Design.md §6.3, Implementation Plan
Phase 2 task "Partial FedAvg accumulator".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from hermes.types import (
    DeviceID,
    GradientSubmission,
    MuleID,
    PartialAggregate,
    Weights,
)
from hermes.types.fl_messages import UPDATE_FORM_DELTA


class PartialFedAvgError(RuntimeError):
    """Raised on invalid inputs to ``partial_fedavg``."""


def partial_fedavg(
    mule_id: MuleID,
    mission_round: int,
    submissions: Sequence[GradientSubmission],
) -> PartialAggregate:
    """Weighted average over verified submissions for one mission round.

    Rejects:
    * empty input (nothing to aggregate)
    * mixed-round submissions (bug upstream)
    * mismatched layer counts / shapes

    Zero-example submissions are dropped silently — a device that
    answered the solicit but had no data does not pull the aggregate.
    """
    if not submissions:
        raise PartialFedAvgError("no submissions to aggregate")

    # Round consistency
    rounds = {s.mission_round for s in submissions}
    if rounds != {mission_round}:
        raise PartialFedAvgError(
            f"partial_fedavg: mixed mission_rounds {rounds} "
            f"(expected {mission_round})"
        )

    # Drop zero-example submissions
    effective = [s for s in submissions if s.num_examples > 0]
    if not effective:
        raise PartialFedAvgError(
            "every submission had num_examples=0; nothing to aggregate"
        )

    # Shape / layer-count validation
    reference_layers = [w.shape for w in effective[0].delta_theta]
    for s in effective[1:]:
        if len(s.delta_theta) != len(reference_layers):
            raise PartialFedAvgError(
                f"layer-count mismatch in submission from {s.device_id!r}: "
                f"got {len(s.delta_theta)} layers, expected "
                f"{len(reference_layers)}"
            )
        for idx, (w, ref) in enumerate(zip(s.delta_theta, reference_layers)):
            if w.shape != ref:
                raise PartialFedAvgError(
                    f"layer {idx} shape mismatch in submission from "
                    f"{s.device_id!r}: got {w.shape}, expected {ref}"
                )

    # Weighted sum in float64, then divide
    total_examples = sum(s.num_examples for s in effective)
    accumulators: Weights = [
        np.zeros(shape, dtype=np.float64) for shape in reference_layers
    ]
    for s in effective:
        w = s.num_examples / total_examples
        for idx, layer in enumerate(s.delta_theta):
            accumulators[idx] += layer.astype(np.float64) * w

    # Cast back to the dtype of the first submission (stable by convention)
    output_dtypes = [w.dtype for w in effective[0].delta_theta]
    merged: Weights = [
        acc.astype(dtype) for acc, dtype in zip(accumulators, output_dtypes)
    ]

    contributing: tuple[DeviceID, ...] = tuple(s.device_id for s in effective)
    return PartialAggregate(
        mule_id=mule_id,
        mission_round=mission_round,
        weights=merged,
        num_examples=total_examples,
        contributing_devices=contributing,
    )


def partial_fedavg_delta(
    mule_id: MuleID,
    mission_round: int,
    submissions: Sequence[GradientSubmission],
    *,
    weights: Sequence[float],
    normalizer: float,
) -> PartialAggregate:
    """Delta-form merge: Σ_i w_i·Δθ_i / ``normalizer``.

    FeRRy Phase 1. Each submission carries Δθ_i, its model minus the basis it
    trained from (``update_form == "delta"``); ``weights`` are the rule's raw
    weights w_i, already zero-free (see ``aggregation_rules.merge_on_mule``).
    ``normalizer`` is the staleness-free mass M_m = Σ n_i·v_i of the same
    updates for the weighted rules and the update count for FedBuff — not
    Σ w_i, so a common staleness factor shrinks the result instead of
    cancelling. It is stored as ``weight_mass``. The result is an update to
    add to the global θ, not a model.

    Rejects the same malformed inputs as :func:`partial_fedavg`, plus a
    submission in full-weights form and a non-positive normalizer.
    """
    if not submissions:
        raise PartialFedAvgError("no submissions to aggregate")
    if len(weights) != len(submissions):
        raise PartialFedAvgError(
            f"{len(weights)} weights for {len(submissions)} submissions"
        )
    if not normalizer > 0.0:
        raise PartialFedAvgError(
            f"merge weights sum to {normalizer}; nothing to aggregate"
        )
    rounds = {s.mission_round for s in submissions}
    if rounds != {mission_round}:
        raise PartialFedAvgError(
            f"partial_fedavg_delta: mixed mission_rounds {rounds} "
            f"(expected {mission_round})"
        )
    for s in submissions:
        if s.update_form != UPDATE_FORM_DELTA:
            raise PartialFedAvgError(
                f"submission from {s.device_id!r} is in {s.update_form!r} form; "
                f"the delta merge needs {UPDATE_FORM_DELTA!r}"
            )

    reference_layers = [w.shape for w in submissions[0].delta_theta]
    for s in submissions[1:]:
        if len(s.delta_theta) != len(reference_layers):
            raise PartialFedAvgError(
                f"layer-count mismatch in submission from {s.device_id!r}: "
                f"got {len(s.delta_theta)} layers, expected "
                f"{len(reference_layers)}"
            )
        for idx, (w, ref) in enumerate(zip(s.delta_theta, reference_layers)):
            if w.shape != ref:
                raise PartialFedAvgError(
                    f"layer {idx} shape mismatch in submission from "
                    f"{s.device_id!r}: got {w.shape}, expected {ref}"
                )

    accumulators: Weights = [
        np.zeros(shape, dtype=np.float64) for shape in reference_layers
    ]
    for s, w in zip(submissions, weights):
        scale = float(w) / float(normalizer)
        for idx, layer in enumerate(s.delta_theta):
            accumulators[idx] += layer.astype(np.float64) * scale

    output_dtypes = [w.dtype for w in submissions[0].delta_theta]
    merged: Weights = [
        acc.astype(dtype) for acc, dtype in zip(accumulators, output_dtypes)
    ]
    return PartialAggregate(
        mule_id=mule_id,
        mission_round=mission_round,
        weights=merged,
        num_examples=sum(s.num_examples for s in submissions),
        contributing_devices=tuple(s.device_id for s in submissions),
        update_form=UPDATE_FORM_DELTA,
        weight_mass=float(normalizer),
        n_updates=len(submissions),
    )
