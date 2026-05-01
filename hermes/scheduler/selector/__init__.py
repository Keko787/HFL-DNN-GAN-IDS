"""TargetSelectorRL — Phase 5 intra-bucket selector.

Design refs:
* HERMES_FL_Scheduler_Design.md §2.7 TargetSelectorRL (sub-model of L2 S3)
* HERMES_FL_Scheduler_Design.md §6.4 (features)
* HERMES_FL_Scheduler_Design.md §7 principle 12 (scope guard — selector
  only ranks *within* a bucket, never promotes gated-out devices)
* HERMES_FL_Scheduler_Implementation_Plan.md §3 Phase 5

Public surface:
* :class:`TargetSelectorRL` — the Phase-5 DDQN-backed selector.
* :class:`SelectorScopeViolation` — raised when a caller asks the
  selector to consider a gated-out device.
* :func:`extract_features` — pure feature extractor (Phase 5 DoD
  testable shape).
"""

from __future__ import annotations

from .features import (
    FEATURE_DIM,
    SelectorEnv,
    extract_features,
    extract_features_batch,
    extract_features_contact_batch,
    extract_features_for_contact,
)
from .scope_guard import SelectorScopeViolation, assert_candidates_admitted
from .replay import ReplayBuffer, Transition
from .ddqn import DDQN
from .target_selector_rl import TargetSelectorRL

__all__ = [
    "FEATURE_DIM",
    "SelectorEnv",
    "extract_features",
    "extract_features_batch",
    "extract_features_for_contact",
    "extract_features_contact_batch",
    "SelectorScopeViolation",
    "assert_candidates_admitted",
    "ReplayBuffer",
    "Transition",
    "DDQN",
    "TargetSelectorRL",
]
