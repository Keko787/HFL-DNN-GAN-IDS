"""Selector scope guard — enforces design §7 principle 12.

    `TargetSelectorRL` is bounded to intra-bucket ordering. The selector
    runs *after* the deterministic gates (S1/S2A/S2B) and *after* the
    deadline math (S3). It cannot promote a gated-out device, cannot
    reorder buckets, and cannot override a deadline — it only breaks
    ties within a bucket.

This module provides a single checked exception + a lightweight asserter
that the selector wrapper invokes before consulting the RL actor.
Runtime cost is one set-membership check per bucket — negligible.
"""

from __future__ import annotations

from typing import Iterable, Set

from hermes.types import DeviceID


class SelectorScopeViolation(RuntimeError):
    """Raised when a caller asks the selector to pick from an illegal set.

    Principle #12: the selector can only order devices that S1 / S2A /
    S2B / S3 have already admitted. Seeing a gated-out device in the
    candidate list is a wiring bug (not a data issue) — we fail loudly.
    """


def assert_candidates_admitted(
    candidates: Iterable[DeviceID],
    admitted: Iterable[DeviceID],
) -> None:
    """Raise :class:`SelectorScopeViolation` if any candidate is not admitted.

    ``admitted`` is the set produced by the deterministic pipeline up to
    and including S3 bucket-classification. ``candidates`` is the subset
    the caller wants ranked.
    """
    admitted_set: Set[DeviceID] = set(admitted)
    illegal = [c for c in candidates if c not in admitted_set]
    if illegal:
        raise SelectorScopeViolation(
            f"selector cannot consider non-admitted devices: {illegal}"
        )
