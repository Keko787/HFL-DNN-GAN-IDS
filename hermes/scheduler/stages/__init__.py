"""Scheduler pipeline stages — pure functions, no I/O.

Each stage is a thin, hand-auditable implementation of one row from
design doc §2.1. ``FLScheduler`` composes them; tests import them
directly.

Stage index:

* ``s1_eligibility``  — ``eligible(i) = active_deadline ∨ beacon_heard``
* ``s2a_readiness``   — on-contact FL_READY gate
* ``s2b_flag``        — utility > FL_Threshold (mule-side verification)
* ``s3_deadline``     — Deadline(j) formula + bucket classifier
* ``s35_selector``    — intra-bucket ordering (distance-sorted placeholder)
"""

from __future__ import annotations

from .s1_eligibility import is_eligible, filter_eligible
from .s2a_readiness import is_on_contact_ready
from .s2b_flag import passes_fl_threshold
from .s3_deadline import (
    compute_deadline,
    classify_bucket,
    fold_round_close_delta,
    fold_cluster_amendment,
)
from .s3a_cluster import cluster_by_rf_range, order_pass_2_greedy
from .s35_selector import select_order, select_target

__all__ = [
    "is_eligible",
    "filter_eligible",
    "is_on_contact_ready",
    "passes_fl_threshold",
    "compute_deadline",
    "classify_bucket",
    "fold_round_close_delta",
    "fold_cluster_amendment",
    "cluster_by_rf_range",
    "order_pass_2_greedy",
    "select_order",
    "select_target",
]
