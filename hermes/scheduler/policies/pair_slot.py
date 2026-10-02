"""FeRRy Phase 5 (unit U3): the pair slot, the flight slot's third filling (``pair_q``).

**What it decides.** At each Pass-1 arrival at stop k the slot makes one
decision, a pair (b, s) (the Phase 5 spec, other choices 1; the user's
decision 1 (a)): b is the class k is served on, at once, priced at the SNR
observed now; s is the stop of the remainder flown next, or home, which only
an empty remainder offers. The supervisor (unit U5) builds the
:class:`~hermes.scheduler.plan.types.PairView`, binds the mask's predicate
(:func:`bind_fits_pair`, over ``FLScheduler.fits_after_service``), calls
:meth:`PairQSlot.pair_at_arrival`, builds the contact plan on the chosen band
and, after the stop, moves s to the front (``cross_heuristic.moved_to_front``).
The next departure then runs the departure check on that order (trimming it
if it fails), the beacon hook, and :meth:`PairQSlot.next_stop`, which returns
0. At takeoff the plan's first stop is flown, and Pass 2 flies b̄ in the
queue's order (R6 and R10, Freeze L1086 and L1096), so the slot decides
nothing there: nothing has been observed in flight at takeoff, and Pass 2 is
the merge clock's delivery.

**One code path for every scorer.** What ranks the pairs is injected
(:class:`~hermes.scheduler.plan.types.PairScorer`): the learned score (the
pair features' adapter over the pair network, units U1 and U2) or one of the
four scripted references below. The slot applies the same rules to all of
them, so a reference differs from the learned score only in its ranking:

* the pass guard: Pass 1 only;
* the scope guard (``selector.scope_guard.assert_pairs_admitted``): every
  pair serves the stop on a class that reaches every committed device there
  and flies to a stop of the plan (home only once none is left), over
  admitted devices;
* the mask: ``fits_pair`` is asked about every pair, and only a pair it
  admits is chosen (Freeze principle 12: learning chooses only among admitted
  pairs);
* the pick: the masked argmax of the scorer's numbers, ties to the lowest row
  (the pairs are class-major in link order, then in the remainder's order;
  home stands alone), the rule the learner's own target takes
  (``pair_q.masked_argmax``);
* the fallback: when no pair fits, FX's pair, its fastest covering class with
  no reorder, recorded ``mask_empty``. That is common: FX's own pair overruns
  the budget at the arrival SNR at about 19 of 71 last-stop arrivals at N = 6
  (the Phase 5 design, finding 3). The departure check then folds the plan's
  order as it does for F and FX, re-planning it under ``in_flight_response =
  replan`` and giving up the pass under ``abort`` when the next stop no longer
  fits, and the slot flies what is left in its order (``next_stop`` returns
  0; the spec, other choices 1);
* the record (``mission_completed.pass_1_pairs``), JSON-ready and without a
  wall time (:data:`DECISION_KEYS`), which the supervisor closes at the
  mission's end (:func:`closed_record`, :data:`CLOSE_KEYS`).

**The scripted references** (critic A2 and B5; the spec, other choices 8).
Each ranks every pair totally, so the slot's tie rule never decides for it,
and each is its rule inside the slot's rules above (the mask, the fallback
and the reorder after the stop), so it faces what the learned score faces.
That makes none of them the fixed arm of its name: where ``fx_pair``,
``committed_pair`` and ``hyb`` part from FX, F and a fixed HYB is stated with
each, as priced, past the departure after the stop (the real-runtime tests
model the supervisor's loop there under both in-flight responses):

* ``fx_pair`` (:class:`FXPairScorer`): FX's band rule first, then its
  next-stop rule, so where FX's pair is admitted it is chosen. It flies the
  FX arm's flight wherever the arm's departure check keeps the remainder as
  it stands, and the record keeps FX's pair (``fx_band``, ``fx_next``) for
  the agreement share. Under ``in_flight_response = abort`` that check folds
  the next stop alone, which passes whenever a stop on FX's band is admitted
  (a route that fits holds that stop too), so the two part only where the
  mask refuses every stop on FX's band while it admits another band's pair.
  Under ``replan`` the check folds the plan's order before FX's rule picks
  and re-plans it when it fails, which, as priced, is exactly when (FX's
  band, 0) is refused: the arm then re-plans first (and may drop stops) and
  FX picks from what is left, while the slot reorders to FX's nearest
  admitted stop and drops nothing or, on an empty mask, flies the head of
  the same re-plan;
* ``committed_pair`` (:class:`CommittedPairScorer`): F's ranking, b̄ and then
  the plan's order. Where F's pair (b̄, 0) is admitted it flies the F arm's
  flight (the committed slot). Elsewhere it parts from F, which always flies
  b̄ to the plan's next stop and leaves the rest to its departure check: the
  slot reorders to b̄'s earliest admitted stop (or another band's), and on
  an empty mask it flies FX's pair, which is F's only when FX's band is b̄;
* ``hyb`` (:class:`HybScorer`): critic B5's HYB, FX's band with the plan's
  order. Where (FX's band, 0) is admitted, or no pair is, it flies a fixed
  HYB's flight (FX's band, then the committed slot's order); where (FX's
  band, 0) is refused but another pair fits, it reorders, or changes band,
  where a fixed HYB flies stop 0. No fixed HYB filling exists: the flight
  slot's fixed fillings are F's and FX's;
* ``greedy_1`` (:class:`Greedy1Scorer`): the one-step greedy rule under the
  user's decision 4 (critic A2): the most targets at the arrival SNR, then the
  least dwell plus travel, then FX's tie-breaks. One device is worth 1/N of
  the reward, more than any time a pair can save at c_t = 0.1 per T_nom, so
  the reward ranks by targets first, which FX does not. It is the "most
  devices" band rule Phase 4 rejected for landing past the budget after the
  last stop (``cross_heuristic.py``, critic A7); the mask now prices that
  landing at the observed rate, but the realized dwell still runs longer than
  priced, so it is a reference to beat, not a filling.

**Training (FerrySim only).** :meth:`PairQSlot.attach_trainer` makes every
decision ε-greedy over the admitted pairs (``pair_q.behaviour_row``; critic
C4): around FX's pair in the training schedule's reference phase, else around
the scorer's argmax. Each decision is kept as a :class:`PairStep`, and
:meth:`PairQSlot.close_mission`, which the supervisor calls with the
mission's closed records, hands the steps and the records to the trainer's
sink, which forms the transitions.

**Layering.** Plan mode only: this module imports the plan types and the FX
filling, so neither package's ``__init__`` imports it, and only the
``pair_q`` path loads it. It imports nothing from ``hermes.l1``,
``hermes.mule``, ``hermes.mission`` or ``experiments``: the runtime reaches it
as a :class:`~hermes.scheduler.plan.types.PairView` and the predicate as a
callable. It is deterministic: no wall time, and no draw but the trainer's
seeded stream.
"""

from __future__ import annotations

import collections.abc
import math
import numbers
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

from hermes.scheduler.plan.types import (
    FLIGHT_SLOT_PAIR_Q,
    PAIR_FALLBACK_MASK_EMPTY,
    FitsPair,
    Pair,
    PairChoice,
    PairScorer,
    PairView,
    check_pair_scores,
)
from hermes.scheduler.policies.cross_heuristic import (
    fastest_covering_class,
    moved_to_front,
    nearest_first,
)
from hermes.scheduler.selector.pair_q import behaviour_row, masked_argmax
from hermes.scheduler.selector.scope_guard import assert_pairs_admitted
from hermes.types.ids import DeviceID
from hermes.types.scheduler import ContactWaypoint, MissionPass

__all__ = [
    "CLOSE_KEYS",
    "COMMITTED_PAIR",
    "DECISION_KEYS",
    "FX_PAIR",
    "GREEDY_1",
    "HOME",
    "HYB",
    "SCRIPTED_SCORERS",
    "CommittedPairScorer",
    "FXPairScorer",
    "Greedy1Scorer",
    "HybScorer",
    "PairQSlot",
    "PairSink",
    "PairStep",
    "bind_fits_pair",
    "closed_record",
    "scripted_scorer",
]

#: The scripted references' names (the spec, other choices 8): each scorer's
#: ``name``, and so the ``scorer`` its decisions record.
FX_PAIR = "fx_pair"
COMMITTED_PAIR = "committed_pair"
HYB = "hyb"
GREEDY_1 = "greedy_1"
SCRIPTED_SCORERS: Tuple[str, ...] = (FX_PAIR, COMMITTED_PAIR, HYB, GREEDY_1)

#: A record's ``next`` when the pair flies home.
HOME = "home"

#: A decision record's keys, in the order it is written at the arrival
#: (``mission_completed.pass_1_pairs``; the design's section 2.8, spec other
#: choices 9):
#:
#: * ``t_s``: the arrival on the mission clock (the view's ``clock_s``);
#:   ``devices``: the stop's members; ``committed``: b̄;
#: * ``band`` and ``next_index``: the pair flown (``next_index`` 0 keeps the
#:   plan's order, None is home); ``next``: the next stop's members, or
#:   :data:`HOME`;
#: * ``pairs`` and ``feasible``: how many pairs were offered and how many the
#:   mask admitted; ``admitted_pairs``: those, as ``[band, index]`` in row
#:   order, so a reader can check the choice against its mask;
#: * ``fallback``: ``mask_empty`` when no pair fitted, else None;
#: * ``fx_band``, ``fx_next`` and ``agrees_fx``: FX's pair by its own rules at
#:   this arrival, on the remainder as it stands and priced as the mask prices
#:   it (FX's band; the nearest stop whose pair on that band the mask admits,
#:   else 0; None for home), and whether the pair flown is it, for the
#:   scorer's agreement share. That is FX's rule, not the FX arm's flight:
#:   under ``in_flight_response = replan`` the arm's departure check folds the
#:   plan's order first and re-plans it when it fails, which, as priced, is
#:   exactly when ``[fx_band, 0]`` is missing from ``admitted_pairs`` with
#:   stops left, and a reader comparing with the FX arm separates those
#:   decisions by that test (the module docstring, ``fx_pair``);
#: * ``scorer``: the scorer's name; ``q`` and ``q_fx``: the scorer's numbers
#:   for the pair flown and for FX's, to 6 places, when they are Q values
#:   (the scorer's ``q_values``), else None.
DECISION_KEYS: Tuple[str, ...] = (
    "t_s", "devices", "committed", "band", "next_index", "next", "pairs", "feasible",
    "admitted_pairs", "fallback", "fx_band", "fx_next", "agrees_fx", "scorer", "q", "q_fx",
)

#: The keys the supervisor adds when it closes a record at the mission's end
#: (:func:`closed_record`; the spec, other choices 1 and 7):
#:
#: * ``collected``: the stop's members whose updates were collected CLEAN
#:   there; ``w``: their raw L3 weights, in that order (``update_weights``,
#:   the merge's own), from which the reward's G_k is taken; ``late``: those
#:   collected after their own deadline;
#: * ``t_next_s``: the next Pass-1 arrival, or the end of the upload after the
#:   sortie's last decision, so Δt_k = ``t_next_s`` − ``t_s``;
#: * ``terminal``: no later Pass-1 decision follows in the sortie (a ``home``
#:   decision that a beacon insert follows is not terminal, critic B12);
#: * ``trimmed_next``: the departure check after this stop trimmed the order
#:   the pair set.
CLOSE_KEYS: Tuple[str, ...] = ("collected", "w", "late", "t_next_s", "terminal", "trimmed_next")


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def _pass(pass_kind: Any) -> MissionPass:
    """``pass_kind`` as a ``MissionPass``; its value (``"collect"``) names it too.

    Required at every call, as the fixed fillings require it, so no call site
    can leave the slot to assume Pass 1 at a Pass-2 stop.
    """
    try:
        return MissionPass(pass_kind)
    except ValueError:
        raise ValueError(
            f"pass_kind must name a mission pass {[p.value for p in MissionPass]}, "
            f"got {pass_kind!r}") from None


def _check_view(view: Any) -> PairView:
    if not isinstance(view, PairView):
        raise TypeError(f"the pair slot reads a PairView, got {view!r}")
    return view


def _verdict(value: Any, band: str, index: Optional[int]) -> bool:
    """``fits_pair``'s answer, which must be a bool (the predicate's ``.ok``).

    Anything else is refused rather than read for its truth: a fold result
    passed whole would be truthy and admit every pair.
    """
    if not isinstance(value, bool):
        raise TypeError(
            f"fits_pair({band!r}, {index!r}) must answer a bool (the predicate's .ok), "
            f"got {value!r}")
    return value


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return value


def _device_list(values: Any, name: str) -> List[str]:
    """``values`` as a list of distinct device ids (non-empty strings)."""
    if isinstance(values, (str, bytes)) or not isinstance(values, collections.abc.Iterable):
        raise TypeError(f"{name} lists device ids, got {values!r}")
    out: List[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} holds device ids (non-empty strings), got {value!r}")
        out.append(str(value))
    if len(set(out)) != len(out):
        raise ValueError(f"{name} lists each device once, got {out}")
    return out


def _plain(value: Any) -> Any:
    """A record's value with plain containers: maps as dicts, sequences as lists."""
    if isinstance(value, collections.abc.Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# --------------------------------------------------------------------------- #
# FX's pair, and the scripted references
# --------------------------------------------------------------------------- #

def _band_ranks(view: PairView) -> Dict[str, int]:
    """FX's ranking of the covering classes, best first.

    ``cross_heuristic.fastest_covering_class``'s key extended to a full order:
    the least dwell at the arrival SNR, then the class reaching more devices,
    then the committed class (no switch for nothing), then the lower index.
    Its first class is FX's band.
    """
    committed = view.arrival.committed
    ranked = sorted(view.covering,
                    key=lambda c: (c.dwell_s, -len(c.targets), c.name != committed, c.index))
    return {entry.name: rank for rank, entry in enumerate(ranked)}


def _nearest_ranks(view: PairView) -> Dict[Optional[int], int]:
    """FX's ranking of the next stops: nearest the stop first (``nearest_first``,
    ties to the position, the devices, then the plan's order); home alone."""
    if view.homebound:
        return {None: 0}
    return {index: rank for rank, index in enumerate(nearest_first(view.remainder, view.pose))}


def _plan_rank(index: Optional[int]) -> int:
    """The plan's order: the remainder's index (home alone)."""
    return 0 if index is None else index


def _ranked(view: PairView, key: Callable[[str, Optional[int]], Tuple[Any, ...]]
            ) -> Tuple[float, ...]:
    """One number per pair of ``view.pairs`` from a total order: 0 for the best, then -1, ..."""
    pairs = view.pairs
    order = sorted(range(len(pairs)), key=lambda row: key(*pairs[row]))
    scores = [0.0] * len(pairs)
    for rank, row in enumerate(order):
        scores[row] = float(-rank)
    return tuple(scores)


def _fx_next(view: PairView, rows: Mapping[Pair, int], mask: Sequence[bool],
             fx_band: str) -> Optional[int]:
    """FX's next stop by its rule at this arrival, on the remainder as it
    stands: home when it is the only next half; else the nearest stop whose
    pair on FX's band the mask admits, else 0.

    FX's own rule (``CrossHeuristic.next_stop``) runs at the departure after
    this stop, after the departure check: the nearest stop whose move to the
    front passes ``fits``, the departure check's fold from the state the
    service left. The mask's stop pair on FX's band is that fold from the
    service priced as served on FX's band (``FLScheduler.fits_after_service``),
    so this is the FX arm's pick whenever the service goes as priced and the
    departure check keeps the remainder as it stands: under ``abort`` always,
    unless the check gives up the pass (it folds the next stop alone), and
    under ``replan`` exactly when (FX's band, 0) fits (it folds the plan's
    order). Otherwise the arm re-plans the order and FX picks from what is
    left, which the slot cannot price: it holds the mask's predicate, not the
    re-plan. A lone stop is 0 either way.
    """
    if view.homebound:
        return None
    for index in nearest_first(view.remainder, view.pose):
        if mask[rows[(fx_band, index)]]:
            return index
    return 0


class FXPairScorer:
    """FX's pair as a ranking: FX's band rule first, then its next-stop rule.

    The slot's version of FX; where it parts from the FX arm is in the
    module docstring.
    """

    name: str = FX_PAIR
    q_values: bool = False

    def score(self, view: PairView, *, mask: Tuple[bool, ...]) -> Tuple[float, ...]:
        """FX's band rank, then the nearest stop. ``mask`` is not read."""
        bands, stops = _band_ranks(_check_view(view)), _nearest_ranks(view)
        return _ranked(view, lambda band, index: (bands[band], stops[index]))


class CommittedPairScorer:
    """F's pair as a ranking: the committed class b̄ and the plan's next stop.

    The slot's version of F, not the F arm (the committed slot), which flies
    (b̄, 0) whatever the mask says. Where F's pair is refused this takes the
    nearest thing to it that fits: b̄ on the plan's earliest stop that fits,
    and only when no stop fits on b̄, the fastest other covering class (FX's
    band order) in the plan's order. On an empty mask the slot flies FX's
    pair (module docstring).
    """

    name: str = COMMITTED_PAIR
    q_values: bool = False

    def score(self, view: PairView, *, mask: Tuple[bool, ...]) -> Tuple[float, ...]:
        """b̄ first, then FX's band order, then the plan's order. ``mask`` is not read."""
        committed = _check_view(view).arrival.committed
        bands = _band_ranks(view)
        return _ranked(view, lambda band, index: (band != committed, bands[band],
                                                  _plan_rank(index)))


class HybScorer:
    """HYB (critic B5): FX's band with the plan's order, FX's band half alone.

    The slot's version of HYB: where (FX's band, 0) is refused this takes
    FX's band on the plan's earliest stop that fits, then the next band in
    FX's order, where a fixed HYB would fly stop 0; no fixed HYB filling
    exists (module docstring).
    """

    name: str = HYB
    q_values: bool = False

    def score(self, view: PairView, *, mask: Tuple[bool, ...]) -> Tuple[float, ...]:
        """FX's band rank, then the plan's order. ``mask`` is not read."""
        bands = _band_ranks(_check_view(view))
        return _ranked(view, lambda band, index: (bands[band], _plan_rank(index)))


class Greedy1Scorer:
    """``greedy_1`` (critic A2): the one-step greedy rule under decision 4's reward.

    The most targets at the arrival SNR; then the least dwell at k on the
    class plus the leg to the next stop (to the dock for home), the decision's
    Δt_k to the next arrival, the listen window aside since it is charged only
    when a reply is missing; then FX's tie-breaks, its band order and then its
    nearest-first order.
    """

    name: str = GREEDY_1
    q_values: bool = False

    def score(self, view: PairView, *, mask: Tuple[bool, ...]) -> Tuple[float, ...]:
        """Most targets, then least dwell plus travel, then FX's order. ``mask`` is not read."""
        bands, stops = _band_ranks(_check_view(view)), _nearest_ranks(view)
        entries = {entry.name: entry for entry in view.covering}
        travel = {ctx.index: ctx.travel_s for ctx in view.stops}

        def key(band: str, index: Optional[int]) -> Tuple[Any, ...]:
            entry = entries[band]
            return (-len(entry.targets), entry.dwell_s + travel[index], bands[band], stops[index])

        return _ranked(view, key)


_SCORERS: Dict[str, Callable[[], PairScorer]] = {
    FX_PAIR: FXPairScorer,
    COMMITTED_PAIR: CommittedPairScorer,
    HYB: HybScorer,
    GREEDY_1: Greedy1Scorer,
}


def scripted_scorer(name: str) -> PairScorer:
    """The scripted reference named ``name`` (:data:`SCRIPTED_SCORERS`)."""
    try:
        make = _SCORERS[name]
    except (KeyError, TypeError):
        raise ValueError(f"a scripted scorer is one of {SCRIPTED_SCORERS}, got {name!r}") from None
    return make()


# --------------------------------------------------------------------------- #
# The mask's binding
# --------------------------------------------------------------------------- #

def bind_fits_pair(
    scheduler: Any,
    view: PairView,
    *,
    served_at: ContactWaypoint,
    state: Any,
    budget_end: Optional[float],
    deadlines: Optional[Mapping[DeviceID, float]] = None,
) -> FitsPair:
    """The mask's predicate at this arrival, as ``fits_pair(band, index) -> bool``.

    The spec's mask (other choices 2) in one place: the service of
    ``served_at`` priced as ``view`` prices ``band`` there (its dwell at the
    arrival SNR, its targets collected), then the remainder flown with its
    stop ``index`` first and the rest in plan order, or the landing for home
    (None): ``scheduler.fits_after_service(...).ok``, which folds on b̄ under
    the arm's in-flight rule with the plan's exempt stops protected.
    ``state`` is the arrival's flight state at ``served_at`` (the transit
    charged, nothing else yet), ``budget_end`` Pass 1's, and ``deadlines`` the
    mule's record of its members' own deadlines, which dates a beacon
    insert's members. Pure: every call is a fold, and nothing moves.

    It answers for the pairs of ``view.pairs`` only, as the scope guard holds
    the slot to them, and refuses anything else with ValueError before it
    folds: home with stops left, a stop with none left, an index that is not
    an int of the remainder (a bool, a negative index, which would fold a
    route holding stops twice, or one past the end), or a class that misses
    a committed target. It is public (the supervisor and FerrySim ask it),
    so an answer always means the pair the view offers.
    """
    view = _check_view(view)
    if not isinstance(served_at, ContactWaypoint):
        raise TypeError(f"served_at is the stop the mule is at, got {served_at!r}")
    if tuple(served_at.devices) != tuple(view.arrival.devices):
        raise ValueError(
            f"the view is of the stop served: its members {list(view.arrival.devices)} are "
            f"not {list(served_at.devices)}")
    remainder = list(view.remainder)
    offered = frozenset(view.pairs)

    def fits_pair(band: str, index: Optional[int]) -> bool:
        if (index is None) != (not remainder):
            raise ValueError(
                f"home is the next half exactly when no stop is left: index {index!r} with "
                f"{len(remainder)} stop(s) left")
        if (not isinstance(band, str) or isinstance(index, bool)
                or not (index is None or isinstance(index, int))
                or (band, index) not in offered):
            raise ValueError(
                f"fits_pair answers for the view's pairs only: ({band!r}, {index!r}) is not "
                f"one of {[list(pair) for pair in view.pairs]}")
        entry = view.arrival.entry(band)
        rest = [] if index is None else moved_to_front(remainder, index)
        return scheduler.fits_after_service(
            rest, served_at=served_at, state=state, dwell_s=entry.dwell_s,
            collected=entry.targets, budget_end=budget_end, pass_kind=MissionPass.COLLECT,
            deadlines=deadlines,
        ).ok

    return fits_pair


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

def _decision_fields(record: Any, name: str) -> Dict[str, Any]:
    """A decision record's fields (:data:`DECISION_KEYS`) as plain values; refuses any other."""
    if not isinstance(record, collections.abc.Mapping):
        raise TypeError(f"{name} must be a pair decision's record (a mapping), got {record!r}")
    keys = set(record)
    if keys != set(DECISION_KEYS):
        closed = sorted(keys & set(CLOSE_KEYS))
        if closed:
            raise ValueError(f"{name} is closed already: it holds {closed}")
        raise ValueError(
            f"{name} is a pair decision's record: missing {sorted(set(DECISION_KEYS) - keys)}, "
            f"unknown {sorted(map(str, keys - set(DECISION_KEYS)))}")
    return {key: _plain(record[key]) for key in DECISION_KEYS}


def closed_record(
    record: Mapping[str, Any],
    *,
    collected: Sequence[DeviceID],
    weights: Mapping[DeviceID, float],
    late: Collection[DeviceID],
    t_next_s: float,
    terminal: bool,
    trimmed_next: bool,
) -> Dict[str, Any]:
    """A decision record closed at the mission's end: a fresh, JSON-ready dict.

    ``record`` is the decision's (``PairChoice.record``, or a ``describe()``
    copy); the result adds :data:`CLOSE_KEYS`: ``collected``, the members
    whose updates were collected CLEAN at the stop (they must be its members),
    ``w`` their raw L3 weights from ``weights`` (one finite weight >= 0 for
    each, and none for anyone else), ``late`` those of them collected after
    their own deadline, in ``collected``'s order, ``t_next_s`` (no earlier
    than the arrival), ``terminal`` and ``trimmed_next``. The supervisor
    closes every decision of the mission on each of its exits (the spec, other
    choices 1), and the trainer's sink checks that each record it is handed was
    closed here. A record is closed once.
    """
    out = _decision_fields(record, "record")
    got = _device_list(collected, "collected")
    outside = sorted(set(got) - set(out["devices"]))
    if outside:
        raise ValueError(f"collected names devices outside the stop: {outside}")
    if not isinstance(weights, collections.abc.Mapping):
        raise TypeError(f"weights map each collected device to its L3 weight, got {weights!r}")
    keys = list(weights)
    if any(not isinstance(key, str) for key in keys) or set(keys) != set(got):
        raise ValueError(
            f"weights must hold exactly the collected devices: collected {got}, weights "
            f"{sorted(map(str, keys))}")
    w = []
    for did in got:
        weight = _real(weights[did], f"weights[{did!r}]")
        if weight < 0.0:
            raise ValueError(f"weights[{did!r}] must be >= 0, got {weights[did]!r}")
        w.append(weight)
    late_set = set(_device_list(late, "late"))
    if not late_set <= set(got):
        raise ValueError(f"late names devices not collected: {sorted(late_set - set(got))}")
    t_next = _real(t_next_s, "t_next_s")
    if t_next < out["t_s"]:
        raise ValueError(
            f"t_next_s ({t_next}) is before the arrival ({out['t_s']}): the next arrival, or "
            f"the landing, comes after the arrival")
    out.update(
        collected=got, w=w, late=[did for did in got if did in late_set], t_next_s=t_next,
        terminal=_flag(terminal, "terminal"), trimmed_next=_flag(trimmed_next, "trimmed_next"),
    )
    return out


def _checked_closed(record: Any, decision: Mapping[str, Any], name: str) -> Dict[str, Any]:
    """``record`` as :func:`closed_record` writes it, closing ``decision``; refused otherwise."""
    if not isinstance(record, collections.abc.Mapping):
        raise TypeError(f"{name} must be a closed pair record (a mapping), got {record!r}")
    keys, want = set(record), set(DECISION_KEYS) | set(CLOSE_KEYS)
    if keys != want:
        raise ValueError(
            f"{name} is a closed pair record: missing {sorted(want - keys)}, unknown "
            f"{sorted(map(str, keys - want))}")
    part = {key: _plain(record[key]) for key in DECISION_KEYS}
    if part != dict(decision):
        raise ValueError(f"{name} is not the record of the decision made there: {part} != "
                         f"{dict(decision)}")
    stored = _plain(record)
    collected, w = stored["collected"], stored["w"]
    if not isinstance(collected, list) or not isinstance(w, list) or len(collected) != len(w):
        raise ValueError(f"{name}: w holds one weight per collected device")
    rebuilt = closed_record(
        part, collected=collected, weights=dict(zip(collected, w)), late=stored["late"],
        t_next_s=stored["t_next_s"], terminal=stored["terminal"],
        trimmed_next=stored["trimmed_next"],
    )
    if rebuilt != stored:
        raise ValueError(f"{name} was not closed by closed_record: {stored} != {rebuilt}")
    return rebuilt


# --------------------------------------------------------------------------- #
# The slot
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PairStep:
    """One decision as the trainer reads it (FerrySim only).

    ``view`` is what the slot saw, from which the pair features build the
    rows (unit U1); ``mask`` is ``fits_pair``'s verdict on every pair, in
    ``view.pairs`` order; ``scores`` are the scorer's numbers; ``row`` is the
    pair flown; ``choice`` is the slot's decision, its record included.
    """

    view: PairView
    mask: Tuple[bool, ...]
    scores: Tuple[float, ...]
    row: int
    choice: PairChoice

    @property
    def effective_mask(self) -> Tuple[bool, ...]:
        """The pairs the decision was taken among: the mask, or the pair flown
        alone when none fitted.

        The spec (other choices 1): on an empty mask the mule flies FX's pair,
        whose effective mask is that one pair, so the learner stores the
        decision and bootstraps through it like any other.
        """
        if any(self.mask):
            return self.mask
        return tuple(row == self.row for row in range(len(self.mask)))


#: The trainer's sink: called at each mission's close with that mission's
#: decisions and their closed records, in decision order, one record per step.
PairSink = Callable[[Tuple[PairStep, ...], Tuple[Dict[str, Any], ...]], None]


@dataclass(frozen=True)
class _Trainer:
    epsilon: float
    around_reference: bool
    rng: Any
    sink: PairSink


class PairQSlot:
    """The pair slot (``flight_slot = "pair_q"``): one (band, next stop) decision
    at each Pass-1 arrival, ranked by an injected scorer.

    It takes the fixed fillings' calls (``next_stop``, ``reads_arrival_view``,
    ``band_at_arrival``), so the supervisor's existing call sites serve it
    too, and adds the arrival's decision (:meth:`decides_at_arrival`,
    :meth:`pair_at_arrival`). See the module docstring for the rules it
    applies to every scorer.
    """

    name: str = FLIGHT_SLOT_PAIR_Q

    def __init__(self, scorer: PairScorer) -> None:
        if not isinstance(scorer, PairScorer):
            raise TypeError(
                f"the pair slot ranks with a PairScorer (a name and score(view, *, mask)), "
                f"got {scorer!r}")
        if not isinstance(scorer.name, str) or not scorer.name:
            raise TypeError(f"a scorer's name labels its records: a non-empty string, got "
                            f"{scorer.name!r}")
        q_values = getattr(scorer, "q_values", False)
        if not isinstance(q_values, bool):
            raise TypeError(f"a scorer's q_values says whether its numbers are Q values: a "
                            f"bool, got {q_values!r}")
        self._scorer = scorer
        self._q_values = q_values
        self._decided = False
        self._trainer: Optional[_Trainer] = None
        self._steps: List[PairStep] = []

    @property
    def scorer(self) -> PairScorer:
        return self._scorer

    @property
    def training(self) -> bool:
        """True once a trainer is attached (FerrySim only)."""
        return self._trainer is not None

    def next_stop(self, remainder: Sequence[ContactWaypoint], state: Any, *,
                  fits: Any, pass_kind: MissionPass, after_stop: bool) -> int:
        """Index 0, in every call; ``state`` and ``fits`` are not read.

        At takeoff (``after_stop`` False) the plan's first stop: nothing has
        been observed in flight, and that order is the plan search's own
        choice from the dock (R6). After a Pass-1 stop, the stop the pair chose
        at the arrival, which the supervisor has moved to the front and the
        departure check has folded, and trimmed if it failed (the spec, other
        choices 1). In Pass 2, the queue's order on b̄ (R10).
        """
        stops = list(remainder)
        if not stops:
            raise ValueError("the flight slot picks among the remaining stops: none is left")
        for wp in stops:
            if not isinstance(wp, ContactWaypoint):
                raise TypeError(f"the remainder holds ContactWaypoints, got {wp!r}")
        _pass(pass_kind)
        if not isinstance(after_stop, bool):
            raise TypeError(
                "after_stop is True when the mule departs from a stop it has served and "
                f"False at takeoff, got {after_stop!r}")
        return 0

    def reads_arrival_view(self, pass_kind: MissionPass) -> bool:
        """True in Pass 1, where the pair view holds the arrival view."""
        return _pass(pass_kind) is MissionPass.COLLECT

    def decides_at_arrival(self, pass_kind: MissionPass) -> bool:
        """True in Pass 1: the band and the next stop are one decision at the
        arrival (:meth:`pair_at_arrival`). False in Pass 2, which flies b̄."""
        return _pass(pass_kind) is MissionPass.COLLECT

    def band_at_arrival(self, view: Any, *, pass_kind: MissionPass) -> Optional[str]:
        """None in Pass 2: fly the runtime's band, b̄, without reading ``view``.

        Refused in Pass 1, where the band comes with the next stop as one
        decision (:meth:`pair_at_arrival`): a call there is a wiring error,
        and answering it would fly a band no record holds.
        """
        if _pass(pass_kind) is MissionPass.COLLECT:
            raise RuntimeError(
                "the pair slot names Pass 1's band with the next stop, as one decision at the "
                "arrival (pair_at_arrival), not through band_at_arrival")
        return None

    def pair_at_arrival(self, view: PairView, *, fits_pair: FitsPair, pass_kind: MissionPass,
                        admitted: Collection[DeviceID]) -> PairChoice:
        """The pair to fly from this Pass-1 arrival, with its record.

        ``fits_pair`` is the mask's predicate (:func:`bind_fits_pair`), asked
        once about every pair of ``view.pairs``, in order, after the scope
        guard has checked them against ``admitted``, the devices the
        pipeline admitted this mission (the plan's and the beacon hook's
        inserts). The scorer then ranks the pairs, given the mask, and the
        slot takes the admitted pair it ranks highest, ties to the lowest row;
        with a trainer attached, ε-greedy over the admitted pairs instead.
        When no pair fits, FX's pair with no reorder, recorded ``mask_empty``.
        """
        if _pass(pass_kind) is not MissionPass.COLLECT:
            raise ValueError(
                "the pair slot decides at Pass-1 arrivals only: Pass 2 flies b̄ in the "
                "queue's order (the Phase 5 spec, other choices 1)")
        view = _check_view(view)
        if not callable(fits_pair):
            raise TypeError(
                "the pair slot needs fits_pair(band, index) -> bool: it never flies an "
                "unchecked pair")
        pairs = view.pairs
        assert_pairs_admitted(
            pairs, remainder=view.remainder, classes=[entry.name for entry in view.covering],
            admitted=admitted, serving=view.arrival.devices,
        )
        mask = tuple(_verdict(fits_pair(band, index), band, index) for band, index in pairs)
        rows = {pair: row for row, pair in enumerate(pairs)}
        fx_band = fastest_covering_class(view.arrival).name
        fx_next = _fx_next(view, rows, mask, fx_band)
        scores = check_pair_scores(view, self._scorer.score(view, mask=mask))
        feasible = sum(mask)
        trainer = self._trainer
        fallback: Optional[str] = None
        if not feasible:
            row = rows[(fx_band, None if view.homebound else 0)]
            fallback = PAIR_FALLBACK_MASK_EMPTY
        else:
            if trainer is None:
                row = masked_argmax(scores, mask)
            else:
                reference = rows[(fx_band, fx_next)] if trainer.around_reference else None
                row = behaviour_row(scores, mask, epsilon=trainer.epsilon, rng=trainer.rng,
                                    reference=reference)
            if not mask[row]:
                raise RuntimeError(
                    f"principle 12: the pair slot picked {pairs[row]}, which the mask refused")
        band, next_index = pairs[row]
        q = scores[row] if self._q_values else None
        q_fx = scores[rows[(fx_band, fx_next)]] if self._q_values else None
        record = {
            "t_s": view.clock_s,
            "devices": [str(d) for d in view.arrival.devices],
            "committed": view.arrival.committed,
            "band": band,
            "next_index": next_index,
            "next": HOME if next_index is None else [
                str(d) for d in view.remainder[next_index].devices],
            "pairs": len(pairs),
            "feasible": feasible,
            "admitted_pairs": [[b, i] for (b, i), ok in zip(pairs, mask) if ok],
            "fallback": fallback,
            "fx_band": fx_band,
            "fx_next": fx_next,
            "agrees_fx": (band, next_index) == (fx_band, fx_next),
            "scorer": self._scorer.name,
            "q": None if q is None else round(q, 6),
            "q_fx": None if q_fx is None else round(q_fx, 6),
        }
        choice = PairChoice(band=band, next_index=next_index, total=len(pairs),
                            feasible=feasible, fallback=fallback, q=q, fx_band=fx_band,
                            fx_next=fx_next, record=record)
        self._decided = True
        if trainer is not None:
            self._steps.append(PairStep(view=view, mask=mask, scores=scores, row=row,
                                        choice=choice))
        return choice

    def close_mission(self, records: Sequence[Mapping[str, Any]]) -> None:
        """The mission's closed records, one per decision in order: a no-op unless training.

        With a trainer attached, each record must close the matching decision
        (:func:`closed_record` of that decision's record), and the mission's
        steps and records then go to the trainer's sink, once.
        """
        trainer = self._trainer
        if trainer is None:
            return
        got = list(records)
        if len(got) != len(self._steps):
            raise ValueError(
                f"the mission made {len(self._steps)} pair decision(s), and {len(got)} "
                f"record(s) close them")
        closed = tuple(_checked_closed(record, step.choice.describe(), f"records[{i}]")
                       for i, (step, record) in enumerate(zip(self._steps, got)))
        steps, self._steps = tuple(self._steps), []
        trainer.sink(steps, closed)

    def attach_trainer(self, *, epsilon: float, rng: Any, sink: PairSink,
                       around_reference: bool = False) -> None:
        """Fly every later decision ε-greedy, and hand each mission's steps to ``sink``.

        FerrySim only (the spec, other choices 3 and 8; critic C4). With
        probability ``epsilon`` a decision is uniform over the admitted pairs;
        otherwise it is FX's pair when ``around_reference`` (the training
        schedule's reference phase) and FX's pair is admitted, else the
        scorer's masked argmax (``pair_q.behaviour_row``, which draws two
        numbers from ``rng`` at every decision with an admitted pair, so runs
        that differ in ε draw alike). An empty mask still flies FX's pair and
        draws nothing. ``pair_q.Behaviour`` holds the two settings: pass
        ``**dataclasses.asdict(behaviour)``. Attached before the slot's first
        decision, so every step of an episode is trained alike: FerrySim
        installs a fresh slot per episode.
        """
        if self._decided:
            raise RuntimeError(
                "attach the trainer before the slot's first decision: a training episode "
                "installs a fresh slot")
        if isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real):
            raise TypeError(f"epsilon must be a number in [0, 1], got {epsilon!r}")
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon!r}")
        if not callable(getattr(rng, "random", None)):
            raise TypeError(f"rng is the episode's seeded stream (random.Random), got {rng!r}")
        if not callable(sink):
            raise TypeError(f"sink takes each mission's steps and closed records, got {sink!r}")
        self._trainer = _Trainer(epsilon=float(epsilon),
                                 around_reference=_flag(around_reference, "around_reference"),
                                 rng=rng, sink=sink)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(scorer={self._scorer.name!r})"
