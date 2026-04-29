"""S2B utility formulas used by ``ClientMission``.

From HERMES_FL_Scheduler_Design.md §6.1:

    Performance_score = f(Acc, AUC, Loss)
    diversity_adjusted = cosine(theta_local, theta_global) * perf_discount
    utility(i) = w1 * Performance_score + w2 * diversity_adjusted

The functions here are deliberately pure so the scheduler stage tests can
pin them to reference tables (design principle 1 — deterministic before
learned).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from hermes.types import Weights


# --------------------------------------------------------------------------- #
# Low-level
# --------------------------------------------------------------------------- #

def cosine_similarity(a: Weights, b: Weights) -> float:
    """Cosine similarity between two flattened weight sets.

    Returns 0.0 if either side is empty or zero-norm, so the scheduler
    can't be poisoned by a NaN/Inf from an untrained device.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(
            f"cosine_similarity: weight-count mismatch ({len(a)} vs {len(b)})"
        )
    flat_a = np.concatenate([x.ravel().astype(np.float64) for x in a])
    flat_b = np.concatenate([y.ravel().astype(np.float64) for y in b])
    if flat_a.shape != flat_b.shape:
        raise ValueError(
            f"cosine_similarity: flattened-shape mismatch "
            f"({flat_a.shape} vs {flat_b.shape})"
        )
    na = float(np.linalg.norm(flat_a))
    nb = float(np.linalg.norm(flat_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(flat_a, flat_b) / (na * nb))


# --------------------------------------------------------------------------- #
# Sub-scores
# --------------------------------------------------------------------------- #

def performance_score(
    accuracy: float,
    auc: float,
    loss: float,
    *,
    w_acc: float = 0.5,
    w_auc: float = 0.3,
    w_loss: float = 0.2,
    loss_cap: float = 10.0,
) -> float:
    """Scalar in ``[0, 1]`` summarising local model quality.

    * ``accuracy`` and ``auc`` are expected in ``[0, 1]``; clipped defensively.
    * ``loss`` is turned into a "lower-is-better" score via
      ``1 - min(loss, loss_cap) / loss_cap`` so a diverging loss can't
      drag the final utility negative.
    """
    acc = float(np.clip(accuracy, 0.0, 1.0))
    auc_c = float(np.clip(auc, 0.0, 1.0))
    loss_c = float(np.clip(loss, 0.0, loss_cap))
    loss_score = 1.0 - loss_c / loss_cap
    return w_acc * acc + w_auc * auc_c + w_loss * loss_score


def diversity_adjusted(
    theta_local: Weights,
    theta_global: Weights,
    perf_discount: float,
) -> float:
    """Cosine-similarity-weighted novelty term.

    Lower cosine (more novel local weights) multiplied by a perf discount
    means an under-performing but divergent device still contributes, but
    its weight falls off with lower perf. Matches the design formula.
    """
    cos = cosine_similarity(theta_local, theta_global)
    return cos * float(np.clip(perf_discount, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #

def utility(
    performance: float,
    diversity: float,
    *,
    w1: float = 0.7,
    w2: float = 0.3,
) -> float:
    """``utility(i) = w1 * Performance_score + w2 * diversity_adjusted``.

    Weights default to design-doc values. Callers tweak per-experiment.
    """
    return w1 * float(performance) + w2 * float(diversity)
