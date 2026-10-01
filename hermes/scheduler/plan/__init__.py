"""FeRRy Phase 4: the plan clock, reach as a decision (build plan L822-847).

With ``plan_mode = "ferry"`` the mule plans each mission at the dock as one
decision over band classes and routes: S1 and S3, the age cap, S3a once per
band class, a search scored by V under S3b's predicate, a guard fold, and the
commit (:class:`~hermes.types.scheduler.PlanCommit`: the band, the ordered
queue and the budget). This package holds that path, one module per unit:

* ``types`` (U0): the types the units share, all re-exported here;
* ``plan_score`` (U2): V and the coverage weights;
* ``member_subset`` (U3): member-subset admission, its fold and trim;
* ``plan_search`` (U4): the search over classes and routes;
* ``hover``: a capped device's best hover point, where it gets a stop of its
  own when its S3a stop cannot serve it alone (the user's decision of
  2026-09-30, on the final check's PLAN-1).

Only the types are re-exported, so importing the package loads no planner
code. Every mechanism runs only with ``plan_mode = "ferry"``; ``legacy`` is
the recorded pipeline and never reaches it (Freeze Rule 1).
"""

from __future__ import annotations

from .types import (
    BAND_POLICY_FIXED_PREFIX,
    BAND_POLICY_SEARCH,
    CAP_CLOSE_REASONS,
    CAP_CROWDED,
    CAP_DROPPED_IN_FLIGHT,
    CAP_NOT_MERGED,
    CAP_PLAN_REASONS,
    CAP_REASONS,
    CAP_UNPLANNABLE,
    COVERAGE_WEIGHTS,
    COVERAGE_WEIGHTS_AGE,
    COVERAGE_WEIGHTS_UNIFORM,
    FLIGHT_SLOT_COMMITTED,
    FLIGHT_SLOT_CROSS_HEURISTIC,
    FLIGHT_SLOTS,
    MEMBER_ADMISSION_SUBSET,
    MEMBER_ADMISSION_WHOLE,
    MEMBER_ADMISSIONS,
    PLAN_MODE_FERRY,
    PLAN_MODE_LEGACY,
    PLAN_MODES,
    PLAN_SCORE_KEYS,
    REASON_PLAN,
    SEARCH_EXACT,
    SEARCH_LOCAL,
    SEARCH_MODES,
    SEARCH_STOP_SUBSETS,
    AgeCapSpec,
    ArrivalClass,
    ArrivalView,
    Candidate,
    CapState,
    CapViolation,
    MemberFold,
    PlanClass,
    PlanCommit,
    PlanOptions,
    PlanScoreParams,
    PlanSearchParams,
    PlanSetup,
    ScoreTerms,
    SearchResult,
    fixed_band_policy,
    parse_band_policy,
)
from .types import __all__ as _TYPES_ALL

__all__ = list(_TYPES_ALL)
