"""Selector scope guard — enforces design §7 principle 12.

    `TargetSelectorRL` is bounded to intra-bucket ordering. The selector
    runs *after* the deterministic gates (S1/S2A/S2B) and *after* the
    deadline math (S3). It cannot promote a gated-out device, cannot
    reorder buckets, and cannot override a deadline — it only breaks
    ties within a bucket.

This module provides a single checked exception + a lightweight asserter
that the selector wrapper invokes before consulting the RL actor.
Runtime cost is one set-membership check per bucket — negligible.

FeRRy Phase 5 adds a second asserter, :func:`assert_pairs_admitted`, for the
flight slot's learned (band, next stop) score (``flight_slot = "pair_q"``,
``hermes/scheduler/policies/pair_slot.py``). The same principle in the plan's
terms (the user's decision 1 (a)): the score chooses only among pairs that
stay within the plan, so it can neither drop nor add a device the
deterministic pipeline did not decide on. Every D arm loads this module
through the selector package, so the addition imports nothing new, and
:func:`assert_candidates_admitted` is unchanged.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Set, Tuple

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


def assert_pairs_admitted(
    pairs: Iterable[Tuple[str, Optional[int]]],
    *,
    remainder: Sequence[Any],
    classes: Iterable[str],
    admitted: Iterable[DeviceID],
    serving: Iterable[DeviceID] = (),
) -> None:
    """Raise :class:`SelectorScopeViolation` unless every pair stays within the plan.

    FeRRy Phase 5's pair slot ranks (band, next stop) pairs at a Pass-1
    arrival: the class the stop is served on, and the index in ``remainder``
    of the stop flown next, or None for home. Each pair must keep the plan
    whole (Freeze principle 12; the Phase 5 spec, other choices 1 and 2):

    * its band is one of ``classes``, the classes that reach every device the
      committed class reaches at this stop (FX's candidates, the pair view's
      ``covering``); any other class would leave a committed device unserved;
    * its next stop is a stop of ``remainder`` (the rest of the plan, in its
      order), and home is offered only once that is empty, since flying home
      with stops left would drop them;
    * every member of the stop being served (``serving``) and of every stop
      of ``remainder`` is in ``admitted``: the devices the deterministic
      pipeline admitted this mission, the plan's own and the beacon hook's
      inserts.

    As for :func:`assert_candidates_admitted`, a violation is a wiring bug,
    not a data issue, so it fails loudly. ``remainder`` holds the plan's
    waypoints; only their ``devices`` are read.
    """
    admitted_set: Set[DeviceID] = set(admitted)
    stops = list(remainder)
    foreign = [d for d in serving if d not in admitted_set]
    foreign += [d for wp in stops for d in wp.devices if d not in admitted_set]
    if foreign:
        raise SelectorScopeViolation(
            f"the pair slot cannot serve or fly to non-admitted devices: {foreign}"
        )
    allowed = set(classes)
    for band, index in pairs:
        if band not in allowed:
            raise SelectorScopeViolation(
                f"the pair slot cannot serve the stop on {band!r}: the classes that reach "
                f"every committed device there are {sorted(allowed)}"
            )
        if index is None:
            if stops:
                raise SelectorScopeViolation(
                    f"the pair slot cannot fly home with {len(stops)} stop(s) of the plan left"
                )
        elif isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(stops):
            raise SelectorScopeViolation(
                f"the pair slot cannot fly to stop {index!r}: the remainder holds "
                f"{len(stops)} stop(s)"
            )
