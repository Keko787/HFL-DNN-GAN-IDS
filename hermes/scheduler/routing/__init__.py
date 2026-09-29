"""Shared routing primitives (FeRRy Phase 2 onward).

One implementation of "order these stops into a short route", used by the
baseline arms (FedEx/CARP's closed 2-OPT tours, arm D4), and later by the
Phase-3 mid-mission re-plan and the Phase-4 route search. Keeping a single
router means every arm that needs a route gets the same quality of route, so a
comparison between arms measures their *policies*, not their route solvers.

See :mod:`hermes.scheduler.routing.two_opt` for the method and its limits.
"""

from __future__ import annotations

from .two_opt import (
    DEFAULT_MAX_PASSES,
    IMPROVEMENT_EPS,
    best_order,
    cheapest_insertion,
    nearest_neighbour,
    order_contacts,
    path_cost,
    two_opt,
)

__all__ = [
    "DEFAULT_MAX_PASSES",
    "IMPROVEMENT_EPS",
    "best_order",
    "cheapest_insertion",
    "nearest_neighbour",
    "order_contacts",
    "path_cost",
    "two_opt",
]
