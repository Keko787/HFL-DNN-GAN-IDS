"""FeRRy Phase 4: the plan clock's shared types (unit U0).

**Why a types module comes first.** Phase 4's plan path is built by several
units at once: the age cap (U1, ``stages/s3d_age_cap.py``), the plan score (U2,
``plan/plan_score.py``), member-subset admission (U3, ``plan/member_subset.py``)
and the ferry runtime with the FX flight slot (U6, ``mule/ferry.py``,
``policies/cross_heuristic.py``); then the search (U4, ``plan/plan_search.py``),
the scheduler fork (U5), the supervisor (U7), the processes and driver (U8) and
the analysis (U9). As designed, their types referred to each other in a cycle
(critic B13): the plan options needed U1's cap spec and U4's search parameters
and classes, the score took U3's fold, and U6 built U4's classes. So every type
that crosses a unit boundary is defined here, once, before any of them. The
functions stay with their units: this module holds data, its validation and
the few derived values that define what the data means (the cap threshold, the
resolved score constants, the plan key), and nothing that plans.

What lives here:

* the switches' values (:data:`PLAN_MODES`, the band-class policy,
  :data:`MEMBER_ADMISSIONS`, :data:`FLIGHT_SLOTS`; each lists its recorded
  value first) and their settings, :class:`AgeCapSpec`,
  :class:`PlanScoreParams` and :class:`PlanSearchParams`, gathered in
  :class:`PlanOptions`; the mule's inputs to the planner in :class:`PlanSetup`,
  with one :class:`PlanClass` per band class;
* what the units hand each other: :class:`CapState` (U1), :class:`ScoreTerms`
  (U2), :class:`MemberFold` (U3), :class:`Candidate` and :class:`SearchResult`
  (U4), and :class:`ArrivalView` (U6's runtime to the flight slot, critic B14);
* the commit, :class:`~hermes.types.scheduler.PlanCommit`, with the cap
  violation record and its reasons, and the band-class policy and search
  modes it records. They live in ``hermes.types`` beside the waypoint the
  commit holds, and are re-exported here.

**FeRRy Phase 5 (its unit U0).** The flight slot's third filling,
``pair_q``, the learned (band, next stop) score, is built by several units
too: the mask's predicate and the runtime's hooks (U4), the slot (U3), the
supervisor (U5), the features (U1) and the learner's trainer (U8), so what
they hand each other is defined here first (critic B11): what the slot sees
at an arrival (:class:`PairView`, with one :class:`StopContext` per candidate
next stop), the mask's predicate (:data:`FitsPair`), what ranks the pairs
(:class:`PairScorer`) and what the slot decided (:class:`PairChoice`). They
are scheduler-side data and a protocol, numpy-free like the rest. They stay
out of ``__all__``, as the coverage ranks below do: the package re-exports
``__all__`` name for name (``plan/__init__.py``, pinned in
``tests/unit/test_p4_types.py``), so they are imported from this module.

Freeze Rule 1: the plan path runs only with ``plan_mode = "ferry"``; with
``legacy`` nothing here is reached. The module is numpy-free and imports
nothing from ``hermes.l1``, ``hermes.mule`` or ``experiments``: physics
reaches the plan as floats and callables, as it reaches S3b
(``stages/s3b_feasibility.py``). It imports nothing from the scheduler
either: S3b's types are named for the type checker only, so a stage (U1's
age cap) or the scheduler itself can import this module without a cycle.
"""

from __future__ import annotations

import collections.abc
import math
import numbers
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from hermes.types.ids import DeviceID
from hermes.types.scheduler import (
    BAND_POLICY_FIXED_PREFIX,
    BAND_POLICY_SEARCH,
    CAP_CLOSE_REASONS,
    CAP_CROWDED,
    CAP_DROPPED_IN_FLIGHT,
    CAP_NOT_MERGED,
    CAP_PLAN_REASONS,
    CAP_REASONS,
    CAP_UNPLANNABLE,
    PLAN_SCORE_KEYS,
    SEARCH_EXACT,
    SEARCH_LOCAL,
    SEARCH_MODES,
    SEARCH_STOP_SUBSETS,
    CapViolation,
    ContactWaypoint,
    PlanCommit,
    fixed_band_policy,
    parse_band_policy,
    plan_class_summaries,
)

if TYPE_CHECKING:  # pragma: no cover - names for the type checker only
    from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel, FlightState

__all__ = [
    "PLAN_MODE_LEGACY", "PLAN_MODE_FERRY", "PLAN_MODES",
    "BAND_POLICY_SEARCH", "BAND_POLICY_FIXED_PREFIX", "parse_band_policy", "fixed_band_policy",
    "MEMBER_ADMISSION_WHOLE", "MEMBER_ADMISSION_SUBSET", "MEMBER_ADMISSIONS",
    "FLIGHT_SLOT_COMMITTED", "FLIGHT_SLOT_CROSS_HEURISTIC", "FLIGHT_SLOTS",
    "COVERAGE_WEIGHTS_AGE", "COVERAGE_WEIGHTS_UNIFORM", "COVERAGE_WEIGHTS",
    "SEARCH_EXACT", "SEARCH_STOP_SUBSETS", "SEARCH_LOCAL", "SEARCH_MODES",
    "REASON_PLAN",
    "CAP_UNPLANNABLE", "CAP_CROWDED", "CAP_DROPPED_IN_FLIGHT", "CAP_NOT_MERGED",
    "CAP_PLAN_REASONS", "CAP_CLOSE_REASONS", "CAP_REASONS", "CapViolation",
    "PLAN_SCORE_KEYS", "PlanCommit",
    "AgeCapSpec", "CapState", "PlanScoreParams", "PlanSearchParams", "ScoreTerms",
    "MemberFold", "PlanClass", "Candidate", "SearchResult", "ArrivalClass", "ArrivalView",
    "PlanOptions", "PlanSetup",
]

# --------------------------------------------------------------------------- #
# Switch values
# --------------------------------------------------------------------------- #

#: ``plan_mode`` (``FLScheduler``, ``MuleConfig``): ``legacy`` plans with
#: ``build_contact_queue`` exactly as recorded; ``ferry`` with the plan path
#: (U5's ``build_ferry_plan``).
PLAN_MODE_LEGACY = "legacy"
PLAN_MODE_FERRY = "ferry"
PLAN_MODES: Tuple[str, ...] = (PLAN_MODE_LEGACY, PLAN_MODE_FERRY)

# ``band_class_policy`` (``search`` or ``fixed:<class>``) and the search modes
# (``SEARCH_MODES``) are recorded by the commit, which checks them, so they are
# defined beside it in ``hermes.types.scheduler`` and re-exported here with
# ``parse_band_policy`` and ``fixed_band_policy``.

#: ``member_admission`` (the user's decision 4 (b)): ``whole`` admits a stop
#: with all its members or not at all, the recorded rule of S3b and the D-arm
#: walks and so their default; ``subset`` may admit part of a stop's members,
#: which removes the narrow-band cliff and is the F family's default
#: (:class:`PlanOptions`).
MEMBER_ADMISSION_WHOLE = "whole"
MEMBER_ADMISSION_SUBSET = "subset"
MEMBER_ADMISSIONS: Tuple[str, ...] = (MEMBER_ADMISSION_WHOLE, MEMBER_ADMISSION_SUBSET)

#: ``flight_slot``: what picks the next stop, and the band, at each arrival.
#: ``committed`` (arm F) takes the plan's next stop on the committed band,
#: exactly today's ``remainder.pop(0)``; ``cross_heuristic`` (arm FX, decision
#: 5) flies to the nearest remaining stop that keeps the rest of the plan
#: feasible and, on arrival, switches to the fastest band that still reaches
#: every device the committed band reaches. ``pair_q`` (FeRRy Phase 5, the FQ
#: arms) chooses both at each Pass-1 arrival as one decision, the learned
#: (band, next stop) score over the pairs the mask admits (:class:`PairView`),
#: and flies only from a verified checkpoint, which the mule's config names.
#: A pinned band (``fixed:<class>``) flies ``committed`` only (:class:`PlanOptions`).
FLIGHT_SLOT_COMMITTED = "committed"
FLIGHT_SLOT_CROSS_HEURISTIC = "cross_heuristic"
FLIGHT_SLOT_PAIR_Q = "pair_q"
FLIGHT_SLOTS: Tuple[str, ...] = (
    FLIGHT_SLOT_COMMITTED, FLIGHT_SLOT_CROSS_HEURISTIC, FLIGHT_SLOT_PAIR_Q,
)

#: ``PlanScoreParams.coverage_weights`` (decision 3): ``age`` weighs a demanded
#: device by its age a_j, ``uniform`` by 1 (the plan's letter, 1 − served/N);
#: either is multiplied by (1 + miss streak) when the arm's ``miss_priority`` is
#: on. Every device left out is widened as a miss, so the streak tracks the age
#: and F's weight is roughly quadratic in age, while F-prio weighs by age alone
#: (critic A5).
COVERAGE_WEIGHTS_AGE = "age"
COVERAGE_WEIGHTS_UNIFORM = "uniform"
COVERAGE_WEIGHTS: Tuple[str, ...] = (COVERAGE_WEIGHTS_AGE, COVERAGE_WEIGHTS_UNIFORM)

#: ``PlanScoreParams.coverage_rank`` (the orchestrator's resolution R11 of
#: 2026-09-30, on the user's decision 2): how the search ranks the candidates
#: that tie on the cap key. ``lexicographic``, the default, ranks them by the
#: served weight share (Σ_served w / Σ_demand w) first and by V only among
#: equal shares: decision 2's "κ = 1: serve everyone the budget allows; time
#: only breaks ties" as a rule. V alone does not keep that promise, because
#: every plan that serves someone pays a whole Pass 2 and the empty plan does
#: not, so the empty plan can outscore a device that fits (U7's probe, u and v
#: 60 m either side of the dock on FB+wide at 1 MB). ``weighted`` ranks by V
#: alone, trading coverage against time at the rate κ sets: the pilot's κ
#: sweep flies it. With the coverage term off (κ = 0, arm F-cov) V alone
#: ranks, whatever this says: cap-only service (decision 3). The package
#: re-exports ``__all__`` name for name (``plan/__init__.py``), so these three
#: stay out of it and are imported from here or from ``plan_score``.
COVERAGE_RANK_LEXICOGRAPHIC = "lexicographic"
COVERAGE_RANK_WEIGHTED = "weighted"
COVERAGE_RANKS: Tuple[str, ...] = (COVERAGE_RANK_LEXICOGRAPHIC, COVERAGE_RANK_WEIGHTED)

#: The drop reason of a demanded device the plan leaves out by choice, when no
#: clause of the predicate refuses it alone from the dock (spec, other choices
#: 8). Plan-level only: S3b's ``REASONS`` stays as pinned
#: (``test_p3_feasibility_predicate.py``, ``test_p3_replan.py``), and such drops
#: stay out of S3c's planned count (critic C6).
REASON_PLAN = "plan"


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _count(value: Any, name: str) -> int:
    """``value`` as an int >= 0; a bool is refused (``True`` is no count)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return int(value)


def _positive_count(value: Any, name: str) -> int:
    out = _count(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return out


def _finite(value: Any, name: str) -> float:
    """``value`` as a finite float; a bool is refused."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _nonneg(value: Any, name: str) -> float:
    out = _finite(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return out


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _device_ids(values: Any, name: str) -> Tuple[DeviceID, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} is a collection of device ids, not one string: {values!r}")
    out = tuple(values)
    for did in out:
        _name(did, f"{name} (device id)")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} lists each device once, got {out!r}")
    return out  # type: ignore[return-value]


def _choice(value: Any, allowed: Tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


def _settings(cls: type, params: Optional[Mapping[str, Any]], name: str) -> Dict[str, Any]:
    """``params`` as keyword arguments of ``cls``. An unknown key is refused, so
    a misspelt setting cannot silently leave its default in place."""
    if params is None:
        return {}
    if not isinstance(params, collections.abc.Mapping):
        raise TypeError(f"{name} must be a mapping, got {params!r}")
    known = [f.name for f in fields(cls)]
    unknown = sorted(str(k) for k in params if k not in known)
    if unknown:
        raise ValueError(f"{name}: unknown settings {unknown}; known: {known}")
    return dict(params)


# --------------------------------------------------------------------------- #
# The age cap (build-plan decision D4)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AgeCapSpec:
    """The hard age cap S (build-plan decision D4; the user's decision 1).

    A device's age is counted in its own mule's missions since its last merged
    update (``DeviceSchedulerState.last_merged_round``; never merged counts as
    0), the scorer's own unit (``traces_scorer.py``). Once the age reaches
    ``s_missions`` less ``lookahead`` the device is capped and the plan must
    serve it (:meth:`caps`). ``s_missions`` None, the default, switches the cap
    off (arm F-cap); a lookahead is then inert.

    ``s_missions`` is a configuration value, an int >= 1. The S* tool (U9)
    recommends the smallest S that covers 90 % of layouts at both pilot budgets
    and never below 2, because S = 1 caps every device at every mission (critic
    A1); 1 stays legal, since the D4 check flies F at S − 1. The lookahead L
    defaults to 0, the plan's "reaches S" (L831); L = 1 does not remove the
    miss of critic probe A3.
    """

    s_missions: Optional[int] = None
    lookahead: int = 0

    def __post_init__(self) -> None:
        if self.s_missions is not None:
            object.__setattr__(self, "s_missions", _positive_count(self.s_missions, "s_missions"))
        object.__setattr__(self, "lookahead", _count(self.lookahead, "lookahead"))

    @property
    def enabled(self) -> bool:
        return self.s_missions is not None

    @property
    def threshold(self) -> Optional[int]:
        """The age from which a device is capped, S − L (None while off)."""
        return None if self.s_missions is None else self.s_missions - self.lookahead

    def caps(self, age: int) -> bool:
        """True when a device of this age is capped; never while the cap is off."""
        age = _count(age, "age")
        threshold = self.threshold
        return threshold is not None and age >= threshold


@dataclass(frozen=True)
class CapState:
    """The cap at planning time (U1's ``evaluate_cap``): ages and capped set.

    ``ages`` holds each demanded device's age (:class:`AgeCapSpec`), read only.
    ``capped`` is derived from them by the spec, so the promotion rule has one
    definition. The ages are kept with the cap off too: the ``age`` coverage
    weights read them.
    """

    spec: AgeCapSpec
    ages: Mapping[DeviceID, int]
    capped: FrozenSet[DeviceID] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, AgeCapSpec):
            raise TypeError(f"spec must be an AgeCapSpec, got {self.spec!r}")
        if not isinstance(self.ages, collections.abc.Mapping):
            raise TypeError(f"ages must be a mapping of device id to age, got {self.ages!r}")
        ages: Dict[DeviceID, int] = {}
        for did, age in self.ages.items():
            ages[_name(did, "ages (device id)")] = _count(age, f"ages[{did!r}]")  # type: ignore[index]
        object.__setattr__(self, "ages", MappingProxyType(ages))
        object.__setattr__(self, "capped", frozenset(d for d, a in ages.items() if self.spec.caps(a)))


# --------------------------------------------------------------------------- #
# The score (spec, other choices 7)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PlanScoreParams:
    """The plan score's settings, ``MuleConfig.plan_score_params``.

    V(b̄, π | demand) = −[c₁(Δ/T)² + c₂U + c₃L] − c₄E/(P_hover·T), where Δ is
    the whole mission on the chosen band b̄ (Pass 1, the dock turnaround and
    Pass 2) and T is the cell's nominal mission period T_nom (the user's
    decision 2 (b)); U is the weighted coverage shortfall, L the expected link
    loss and E the simulated energy of both passes. The constants are hand-set
    and swept, in FedEx's form (Thm 2, eq. 24) with the convex-surrogate
    caveat: the repo holds no derivation from the theory track (build plan
    L898-899).

    * ``c_time`` is c₁ = 1 (spec, other choices 7).
    * ``c_cov_per_device`` is κ, with c₂ = κ·N_demand: one average device is
      worth κ full T² of time. κ = 1 (decision 2): serve every device the
      budget allows, and let time break ties. The pilot sweeps κ in {0.15,
      0.25, 1}; at 0.15, 2 of 30 plans already fly empty at 30 s and 1 MB
      (the Phase 4 spec's probe, decision 2).
    * ``c_link`` is c₃; None, the default, means c₃ = c₂, so that c₂U + c₃L =
      c₂(1 − the expected weighted served share).
    * ``c_energy`` is c₄ = 0.1: E is nearly collinear with Δ (143.6 W flying
      against 168.5 W hovering), so energy only breaks ties (design D-B); the
      pilot sweeps it in {0, 0.1}.
    * ``coverage_weights``: :data:`COVERAGE_WEIGHTS` (decision 3).
    * ``dwell_in_delta``: False takes the dwell out of Δ in the score only (arm
      F-dwell, Study 5.7); the predicate still prices it.
    * ``coverage_rank``: :data:`COVERAGE_RANKS`, how candidates that tie on the
      cap key are ranked (R11): ``lexicographic`` (the default) by the served
      weight share, then V; ``weighted`` by V alone (the pilot's κ sweep). It
      changes the rank only, never V (``plan_score.plan_key``).

    F-cov is ``c_cov_per_device=0, c_link=0``. It then serves capped devices
    only, since otherwise the empty plan scores best ("cap-only service",
    decision 3); with κ = 0 V alone ranks, whatever ``coverage_rank`` says.
    """

    c_time: float = 1.0
    c_cov_per_device: float = 1.0
    c_link: Optional[float] = None
    c_energy: float = 0.1
    coverage_weights: str = COVERAGE_WEIGHTS_AGE
    dwell_in_delta: bool = True
    coverage_rank: str = COVERAGE_RANK_LEXICOGRAPHIC

    def __post_init__(self) -> None:
        for name in ("c_time", "c_cov_per_device", "c_energy"):
            object.__setattr__(self, name, _nonneg(getattr(self, name), name))
        if self.c_link is not None:
            object.__setattr__(self, "c_link", _nonneg(self.c_link, "c_link"))
        _choice(self.coverage_weights, COVERAGE_WEIGHTS, "coverage_weights")
        if not isinstance(self.dwell_in_delta, bool):
            raise TypeError(f"dwell_in_delta must be a bool, got {self.dwell_in_delta!r}")
        _choice(self.coverage_rank, COVERAGE_RANKS, "coverage_rank")

    def constants(self, n_demand: int) -> Tuple[float, float, float, float]:
        """(c1, c2, c3, c4) for a demand of ``n_demand`` devices."""
        c_cov = self.c_cov_per_device * _count(n_demand, "n_demand")
        c_link = c_cov if self.c_link is None else self.c_link
        return (self.c_time, c_cov, c_link, self.c_energy)

    @classmethod
    def from_mapping(cls, params: Optional[Mapping[str, Any]]) -> "PlanScoreParams":
        """From the config dict (None or {}: the defaults); unknown keys are refused."""
        return cls(**_settings(cls, params, "plan_score_params"))

    def as_dict(self) -> Dict[str, Any]:
        """Every setting, defaults included (``mule_ready``'s resolved dict)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class ScoreTerms:
    """One candidate's score and its parts (U2's ``score``).

    ``v`` is V with the resolved constants; ``delta_s`` is Δ in simulated
    seconds; ``time`` ((Δ/T)²), ``coverage`` (U), ``link`` (L) and ``energy``
    (E/(P_hover·T)) are V's four terms unweighted, so that V = −[c₁·time +
    c₂·coverage + c₃·link] − c₄·energy; ``energy_j`` is E in simulated joules;
    ``served_weight`` and ``demand_weight`` are the sums of the coverage
    weights over the served and the demanded devices. Every value is a finite
    float, and Δ, the time and energy terms and the weights are >= 0. The
    field order is ``PLAN_SCORE_KEYS``, the keys a commit's ``score`` records.
    """

    v: float
    delta_s: float
    time: float
    coverage: float
    link: float
    energy_j: float
    energy: float
    served_weight: float
    demand_weight: float

    def __post_init__(self) -> None:
        for f in fields(self):
            value = _finite(getattr(self, f.name), f.name)
            if f.name in ("delta_s", "time", "energy_j", "energy", "served_weight",
                          "demand_weight") and value < 0.0:
                raise ValueError(f"{f.name} must be >= 0, got {value!r}")
            object.__setattr__(self, f.name, value)

    def as_dict(self) -> Dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --------------------------------------------------------------------------- #
# The search (spec, other choices 3)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PlanSearchParams:
    """The plan search's bounds, ``MuleConfig.plan_search_params``.

    * ``exact_max_devices`` = 6: a demand this small is searched exactly (see
      :data:`SEARCH_EXACT`; critic A6). At N = 6 (the pilots) a whole plan
      took at most about 0.25 s where nothing prunes, and about 20 ms under
      30-120 s budgets (U4's timing probes; ``plan_search``'s module notes).
    * ``exhaustive_max_stops`` = 6, the plan's threshold (L829; design D-H):
      above the exact demand, a class with at most this many stops is searched
      depth-first over ordered stop subsets.
    * ``heuristic_max_passes`` = 50 (design D-H) and
      ``heuristic_max_evaluations`` = 2,000: above that, the local search
      stops after this many passes or this many walks per class, whichever
      comes first, counted over all of its scans. A count, never wall time, so
      a repeated trial plans the same (critic C7). The counts buy determinism,
      not a time bound: a walk costs more on a longer route, and where all
      three classes reach 2,000 walks (N = 96 on a 500 m field, 1 MB, no
      budget) one plan took 3.1-3.4 s (U4's timing probes).
    * ``hover_stops`` = True (the Exp 5 addendum, Study 5.14's "hover stops
      off"): the stop family the search runs over is S3a's stops with the
      hover rule applied (``plan/hover.py``: a capped device its S3a stop
      cannot serve alone within the budget gets a one-device stop at its best
      hover point), as recorded; False leaves S3a's stops alone, so the cap's
      ``unplannable`` then reads them. Left out of :meth:`as_dict` at True,
      so a recorded plan's ``mule_ready`` keeps its keys.
    """

    exact_max_devices: int = 6
    exhaustive_max_stops: int = 6
    heuristic_max_passes: int = 50
    heuristic_max_evaluations: int = 2000
    hover_stops: bool = True

    def __post_init__(self) -> None:
        for name in ("exact_max_devices", "exhaustive_max_stops"):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        for name in ("heuristic_max_passes", "heuristic_max_evaluations"):
            object.__setattr__(self, name, _positive_count(getattr(self, name), name))
        _flag(self.hover_stops, "hover_stops")

    @classmethod
    def from_mapping(cls, params: Optional[Mapping[str, Any]]) -> "PlanSearchParams":
        """From the config dict (None or {}: the defaults); unknown keys are refused."""
        return cls(**_settings(cls, params, "plan_search_params"))

    def as_dict(self) -> Dict[str, Any]:
        """Every bound, and ``hover_stops`` only when it is off (the Exp 5
        addendum's switch, at its recorded default left out)."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        if self.hover_stops:
            del out["hover_stops"]
        return out


@dataclass(frozen=True)
class MemberFold:
    """One walk of a route with member-subset admission (U3's fold).

    ``route`` is what would be flown: S3a stops, some reduced to a subset of
    their members (an ordinary ``ContactWaypoint``: the same position,
    ``devices`` the subset). ``dropped`` pairs each complement, the members a
    stop left out as a waypoint of their own, with its reason. ``state`` is the
    flight state after the last stop flown and ``home`` the clock back at the
    dock with the upload done (as ``FoldResult.home``). ``energy_j`` is the
    simulated energy of the pass as flown with the return leg included, which
    ``FlightState.energy_j`` leaves out (``s3b_feasibility.py``, the admit
    energy clause). ``feasible`` is False when the walk required every listed
    stop, as a candidate does, and one of them admitted nobody. A device
    appears at most once, in the route or among the drops.
    """

    route: Tuple[ContactWaypoint, ...]
    dropped: Tuple[Tuple[ContactWaypoint, str], ...]
    state: "FlightState"
    home: float
    feasible: bool
    energy_j: float

    def __post_init__(self) -> None:
        route = tuple(self.route)
        for wp in route:
            if not isinstance(wp, ContactWaypoint):
                raise TypeError(f"route holds ContactWaypoints, got {wp!r}")
        dropped = []
        for pair in self.dropped:
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or not isinstance(pair[0], ContactWaypoint)):
                raise TypeError(f"dropped pairs a ContactWaypoint with its reason, got {pair!r}")
            dropped.append((pair[0], _name(pair[1], "drop reason")))
        members = [d for wp in route for d in wp.devices] + [d for wp, _ in dropped for d in wp.devices]
        if len(set(members)) != len(members):
            twice = sorted({str(d) for d in members if members.count(d) > 1})
            raise ValueError(f"a device appears once in a fold, in the route or dropped: {twice}")
        if not hasattr(self.state, "clock"):
            raise TypeError(f"state must be a FlightState, got {self.state!r}")
        if not isinstance(self.feasible, bool):
            raise TypeError(f"feasible must be a bool, got {self.feasible!r}")
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "dropped", tuple(dropped))
        object.__setattr__(self, "home", _finite(self.home, "home"))
        object.__setattr__(self, "energy_j", _nonneg(self.energy_j, "energy_j"))

    @property
    def served(self) -> FrozenSet[DeviceID]:
        """The devices the route serves: the members of its stops."""
        return frozenset(d for wp in self.route for d in wp.devices)


@dataclass(frozen=True)
class PlanClass:
    """One band class the plan may fly, with its own physics (the mule's, U6).

    ``radius_m`` is R_planar(c): S3a's radius and the contact gate's range for
    the class. ``model`` is the ``FeasibilityModel`` that prices the class. Its
    ferry physics carries a member dwell and is bound to this class when it is
    built: a model that read the runtime's current band would price every
    class at the last band set (critic B3). So a range it carries must be the
    class's radius. ``outage(d)`` is the probability that a member at planar
    distance d is below the SNR floor at the class's mean SNR (the score's link
    term). ``index`` is the class's place in the link's class order; it breaks
    the plan key's ties.
    """

    name: str
    index: int
    radius_m: float
    model: "FeasibilityModel"
    outage: Callable[[float], float]

    def __post_init__(self) -> None:
        _name(self.name, "name")
        object.__setattr__(self, "index", _count(self.index, "index"))
        radius = _finite(self.radius_m, "radius_m")
        if radius <= 0.0:
            raise ValueError(f"radius_m must be > 0, got {self.radius_m!r}")
        object.__setattr__(self, "radius_m", radius)
        if not callable(self.outage):
            raise TypeError("outage must be a callable of the planar distance")
        ferry = getattr(self.model, "ferry", None)
        if ferry is None or getattr(ferry, "member_dwell_s", None) is None:
            raise ValueError(
                f"class {self.name!r}: the plan prices dwell per member on the mission "
                f"clock, so its model needs ferry physics with a member dwell"
            )
        range_m = getattr(ferry, "range_m", None)
        if range_m is not None and not math.isclose(float(range_m), radius, rel_tol=1e-9):
            raise ValueError(
                f"class {self.name!r}: the model prices a {range_m!r} m range but the "
                f"class radius is {radius!r} m (per-class physics, critic B3)"
            )


@dataclass(frozen=True)
class Candidate:
    """One admitted plan (b̄, π) and its score (U4's search).

    ``fold`` is the member fold of the route on the class. A candidate is
    admitted by construction, so an infeasible fold is refused. ``cap_key`` is
    U1's: the ages of the capped devices the plan leaves out, largest first.
    """

    cls: PlanClass
    fold: MemberFold
    terms: ScoreTerms
    cap_key: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name, kind in (("cls", PlanClass), ("fold", MemberFold), ("terms", ScoreTerms)):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"{name} must be a {kind.__name__}, got {getattr(self, name)!r}")
        if not self.fold.feasible:
            raise ValueError("a candidate is an admitted plan: its fold must be feasible")
        key = tuple(_count(a, "cap_key") for a in self.cap_key)
        if any(a < b for a, b in zip(key, key[1:])):
            raise ValueError(f"cap_key lists the unserved capped ages largest first, got {self.cap_key!r}")
        object.__setattr__(self, "cap_key", key)

    @property
    def band(self) -> str:
        return self.cls.name

    @property
    def served(self) -> FrozenSet[DeviceID]:
        return self.fold.served

    @property
    def key(self) -> Tuple[Any, ...]:
        """The plan key under the "weighted" rank, a total order: the smallest wins.

        The search ranks by ``plan_score.plan_key``, which returns this key under
        ``coverage_rank = "weighted"`` (and for F-cov), and by default
        ("lexicographic", R11) inserts the served weight share, descending,
        right after the cap key: (cap key, −round(share, 9), −round(V, 9), ...).

        (cap key, −round(V, 9), class index, each stop's (position, devices)).
        The cap key comes first, so a plan that leaves the oldest capped device
        out loses to any that keeps it; among plans that leave the same oldest
        age out, fewer such devices wins. It minimises the oldest unserved age,
        not the violation count: (4, 4, 4) beats (5, 3) (critic C5), as the
        plan's fallback "keeps the oldest capped devices that fit" (L1196).
        Then the higher V; V is rounded to 9 decimals so that the same plan
        priced along different float paths ties, and a tie falls to the class
        and the stops, deterministically.
        """
        return (
            self.cap_key,
            -round(self.terms.v, 9),
            self.cls.index,
            tuple((tuple(wp.position), tuple(wp.devices)) for wp in self.fold.route),
        )


@dataclass(frozen=True)
class SearchResult:
    """What the search returns (U4) and the scheduler commits (U5).

    ``best`` has the smallest plan key (``plan_score.plan_key``: the
    coverage-first key by default, :attr:`Candidate.key` under "weighted") over
    every class the policy lets the arm fly; ``mode`` is the search that produced the best class's
    candidates (:data:`SEARCH_MODES`); ``n_candidates`` counts the candidates
    evaluated in all, deterministically; ``per_class`` holds one JSON-ready
    summary per class searched, each naming its class under ``band``, with
    the best class among them when any is given. The commit records the
    summaries, so they are checked here exactly as it checks them
    (:func:`~hermes.types.scheduler.plan_class_summaries`): a summary the
    commit would refuse fails the search, not the mission.
    """

    best: Candidate
    mode: str
    n_candidates: int
    per_class: Tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.best, Candidate):
            raise TypeError(f"best must be a Candidate, got {self.best!r}")
        _choice(self.mode, SEARCH_MODES, "mode")
        object.__setattr__(self, "n_candidates", _positive_count(self.n_candidates, "n_candidates"))
        object.__setattr__(self, "per_class", plan_class_summaries(self.per_class, self.best.band))


# --------------------------------------------------------------------------- #
# The flight slot (decision 5)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ArrivalClass:
    """One band class as the mule sees it on arrival at a stop (U6).

    ``targets`` are the stop's members the class would solicit at the realized
    SNR (within R_planar(c) and at or above the floor), in member order;
    ``dwell_s`` is the predicted dwell of serving them all on the class, priced
    at that SNR (critic A7).
    """

    name: str
    index: int
    targets: Tuple[DeviceID, ...]
    dwell_s: float

    def __post_init__(self) -> None:
        _name(self.name, "name")
        object.__setattr__(self, "index", _count(self.index, "index"))
        object.__setattr__(self, "targets", _device_ids(self.targets, "targets"))
        object.__setattr__(self, "dwell_s", _nonneg(self.dwell_s, "dwell_s"))


@dataclass(frozen=True)
class ArrivalView:
    """What the flight slot sees on arrival at a stop (FX, decision 5).

    Scheduler-side data (critic B14): the flight-slot policies in
    ``hermes.scheduler.policies`` read it and the mule's runtime builds it, so
    neither imports the other. ``devices`` are the stop's members,
    ``committed`` is the mission's committed class, and ``classes`` has one
    entry per class of the link. FX switches to the fastest class whose targets
    include every target of the committed class, so it never dwells longer
    than F would there.
    """

    devices: Tuple[DeviceID, ...]
    committed: str
    classes: Tuple[ArrivalClass, ...]

    def __post_init__(self) -> None:
        devices = _device_ids(self.devices, "devices")
        if not devices:
            raise ValueError("an arrival is at a stop with at least one member")
        classes = tuple(self.classes)
        for entry in classes:
            if not isinstance(entry, ArrivalClass):
                raise TypeError(f"classes hold ArrivalClass entries, got {entry!r}")
            outside = sorted(set(entry.targets) - set(devices))
            if outside:
                raise ValueError(f"class {entry.name!r} targets devices outside the stop: {outside}")
        names = [entry.name for entry in classes]
        if len(set(names)) != len(names) or len({e.index for e in classes}) != len(classes):
            raise ValueError(f"classes are distinct, by name and by index: {names}")
        if self.committed not in names:
            raise ValueError(f"the committed class {self.committed!r} is not among {names}")
        object.__setattr__(self, "devices", devices)
        object.__setattr__(self, "classes", classes)

    def entry(self, name: str) -> ArrivalClass:
        """The class named ``name``; KeyError if the link has none."""
        for entry in self.classes:
            if entry.name == name:
                return entry
        raise KeyError(name)

    @property
    def committed_entry(self) -> ArrivalClass:
        return self.entry(self.committed)


# --------------------------------------------------------------------------- #
# The pair slot (FeRRy Phase 5)
# --------------------------------------------------------------------------- #

#: ``PairChoice.fallback``. ``mask_empty``: no pair fits at the arrival, so the
#: mule flies FX's pair, its fastest covering class with no reorder, and the
#: record says so (the Phase 5 spec, other choices 1). The case is common: at
#: N = 6 and 1 MB, FX's own pair already overruns the budget, priced at the
#: arrival SNR, at 19 of about 71 last-stop arrivals (the Phase 5 design,
#: finding 3), so the fallback decides the flight there.
PAIR_FALLBACK_MASK_EMPTY = "mask_empty"
PAIR_FALLBACKS: Tuple[str, ...] = (PAIR_FALLBACK_MASK_EMPTY,)

#: One pair: the class the arrival's stop is served on, and the index in the
#: remainder of the stop flown next, or None for home.
Pair = Tuple[str, Optional[int]]

#: ``fits_pair(band, index) -> bool``: does the pair keep the rest of the
#: flight feasible? The supervisor binds the scheduler's mask predicate
#: (``FLScheduler.fits_after_service``; the Phase 5 spec, other choices 2):
#: the state after serving the stop on ``band`` at the arrival's dwell, then
#: the remainder flown with its stop ``index`` first and the rest in plan
#: order, folded on b̄ under the arm's in-flight rule with the exempt stops
#: protected; for None (home), the landing within the budget end and
#: ``deliver_by``, and the energy clause. The slot asks it about every pair
#: and re-implements none of it.
FitsPair = Callable[[str, Optional[int]], bool]


def _index(value: Any, name: str) -> Optional[int]:
    """``value`` as a remainder index (an int >= 0), or None for home."""
    return None if value is None else _count(value, name)


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return value


def _positive(value: Any, name: str) -> float:
    out = _finite(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return out


def _energy_ref(value: Any, name: str) -> Optional[float]:
    """An energy reference as a float > 0, or None for none; 0 is stored as None.

    The reference is ``FerryRuntime.l1_state``'s (``mule/ferry.py``): the
    capacity if one is set, else P_hover times the budget. That product is 0
    when P_hover is, which ``EnergyModel`` accepts (``l1/mission_clock.py``,
    P_hover >= 0) while a capacity must be > 0, and ``l1_state`` reads a 0
    reference as none (``if not e_ref``). So a 0 is stored as None here: a
    reader tests ``is None`` alone and never divides by 0.
    """
    if value is None:
        return None
    out = _nonneg(value, name)
    return out if out > 0.0 else None


def _numbers(values: Any, name: str) -> Tuple[float, ...]:
    """``values`` as a tuple of finite floats (a string is no sequence of numbers)."""
    if isinstance(values, (str, bytes)) or not isinstance(values, collections.abc.Iterable):
        raise TypeError(f"{name} must be a sequence of numbers, got {values!r}")
    return tuple(_finite(v, name) for v in values)


def _per_class(values: Any, n_classes: int, name: str) -> Tuple[float, ...]:
    """One finite float per class of the link, in link order."""
    out = _numbers(values, name)
    if len(out) != n_classes:
        raise ValueError(
            f"{name} holds one number per class of the link ({n_classes}), got {len(out)}"
        )
    return out


def _json_record(value: Any, name: str) -> Any:
    """A fresh JSON-ready copy of ``value``, tuples as lists.

    Refuses what JSON cannot carry exactly (non-finite numbers, keys that are
    not strings, other objects) and any key that names a wall time: a
    decision record holds none, so a repeated trial records the same
    decisions (critic B12, as for the plan commit).
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return _finite(value, name)
    if isinstance(value, collections.abc.Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name}: keys must be strings, got {key!r}")
            if "wall" in key:
                raise ValueError(
                    f"{name}: {key!r} names a wall time; a decision record holds none, so a "
                    f"repeated trial records the same decisions (critic B12)"
                )
            out[key] = _json_record(item, f"{name}[{key!r}]")
        return out
    if isinstance(value, (list, tuple)):
        return [_json_record(item, name) for item in value]
    raise TypeError(f"{name} must be JSON-ready (numbers, strings, lists, dicts), got {value!r}")


def _frozen(value: Any) -> Any:
    """A JSON-ready value (:func:`_json_record`'s) made read-only all the way
    down: lists as tuples, maps as read-only maps.

    A record is checked once, when its choice is made, so a nested list or map
    left mutable would let a wall time or a non-JSON value in after the check,
    and :meth:`PairChoice.describe` would then fail at the mission's close.
    """
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    return value


@dataclass(frozen=True)
class StopContext:
    """One candidate for the pair's second half: a stop of the remainder, or home.

    At a Pass-1 arrival at stop k the mule builds one per stop of the
    remainder, in its order (``index`` is the stop's place there), or the
    single :meth:`home` when nothing remains: home is offered only then,
    since the slot never drops a stop (the Phase 5 spec, other choices 1;
    Freeze principle 12). It holds what the pair features read about the
    candidate (other choices 4), priced from k before k is served:

    * ``travel_s``: the leg from k to the stop (to the dock for home), on the
      mission clock's flight model;
    * ``pred_dwell_s``: serving every member on the committed class b̄ at the
      mean SNR, as the departure check prices the rest of the route (δ_obs =
      0, Freeze L829);
    * ``pred_snr_db``: per class of the link, in link order, the median over
      the members of their mean SNR at the stop. The band at the stop is
      chosen at its own arrival, so the features carry every class's
      prospect there (other choices 4, feature 7);
    * ``capped``: some member is capped (``PlanCommit.capped``); ``exempt``:
      the cap binds every member, so the stop is exempt from its own deadline
      clause (``FLScheduler.plan_protected``), which implies ``capped``;
    * ``age``: the mean of the members' plan ages (``PlanCommit.ages``, in
      their own mule's missions; a member without one, a beacon insert,
      counts 0);
    * ``on_time``: the mean of the members' on-time rates, in [0, 1]
      (``selector.features._on_time_rate``);
    * ``weight``: the sum of the members' coverage weights
      (``PlanCommit.weights``; 0 for a member without one), from which the
      features take the committed weight still to collect.

    The stop's own deadline is its ``deadline_ts``; the slack is the
    features' to price. Home has no stop and nothing but its leg, so its row
    depends on the leg alone.
    """

    stop: Optional[ContactWaypoint]
    index: Optional[int]
    travel_s: float
    pred_dwell_s: float
    pred_snr_db: Tuple[float, ...]
    capped: bool
    exempt: bool
    age: float
    on_time: float
    weight: float

    def __post_init__(self) -> None:
        travel = _nonneg(self.travel_s, "travel_s")
        dwell = _nonneg(self.pred_dwell_s, "pred_dwell_s")
        snr = _numbers(self.pred_snr_db, "pred_snr_db")
        capped = _flag(self.capped, "capped")
        exempt = _flag(self.exempt, "exempt")
        age = _nonneg(self.age, "age")
        on_time = _nonneg(self.on_time, "on_time")
        if on_time > 1.0:
            raise ValueError(f"on_time is a rate in [0, 1], got {self.on_time!r}")
        weight = _nonneg(self.weight, "weight")
        if self.stop is None:
            if self.index is not None:
                raise ValueError(
                    f"home is no stop of the remainder: its index is None, got {self.index!r}"
                )
            if dwell or snr or capped or exempt or age or on_time or weight:
                raise ValueError(
                    "home carries its leg only: no dwell, SNR, cap flags, age, rate or weight"
                )
        else:
            if not isinstance(self.stop, ContactWaypoint):
                raise TypeError(
                    f"stop must be a ContactWaypoint (None for home), got {self.stop!r}")
            object.__setattr__(self, "index", _count(self.index, "index"))
            if not snr:
                raise ValueError("a stop's pred_snr_db holds one mean SNR per class of the link")
            if exempt and not capped:
                raise ValueError(
                    "an exempt stop is one the cap binds for every member: exempt implies capped"
                )
        for name, value in (("travel_s", travel), ("pred_dwell_s", dwell), ("pred_snr_db", snr),
                            ("age", age), ("on_time", on_time), ("weight", weight)):
            object.__setattr__(self, name, value)

    @classmethod
    def home(cls, travel_s: float) -> "StopContext":
        """The return to the dock, offered when the remainder is empty: its leg."""
        return cls(stop=None, index=None, travel_s=travel_s, pred_dwell_s=0.0, pred_snr_db=(),
                   capped=False, exempt=False, age=0.0, on_time=0.0, weight=0.0)

    @property
    def is_home(self) -> bool:
        return self.stop is None


@dataclass(frozen=True)
class PairView:
    """What the pair slot sees at a Pass-1 arrival (``flight_slot = "pair_q"``).

    Scheduler-side data, like :class:`ArrivalView` (critic B14): the mule
    builds it from its runtime (``FerryRuntime.arrival_view``, ``observe``,
    ``class_offsets_db`` and ``stop_contexts``), its commit (``last_plan``)
    and its device states, and the slot and the pair features read it, so
    neither side imports the other. It is taken at the arrival at stop k,
    after the transit is charged and before the contact, at one simulated
    instant, ``clock_s``:

    * ``arrival``: what each class reaches at k and the dwell of serving it,
      at the arrival SNR. Its classes are in link order (their indices
      increase), and every per-class tuple here follows them;
    * ``pose``: where the mule is, k's position;
    * ``observed_snr_db``: per class, the median realized SNR over k's
      members now (``StopObservation.class_snr_db``);
    * ``offsets_db``: per class, the median over k's members of realized
      minus mean SNR, each link differenced before the median (critic C6),
      from which the features read the interference phase;
    * ``previous_offsets_db`` and ``previous_age_s``: the offsets the
      previous Pass-1 arrival of this trial observed and how long ago that
      was, carried across stops and sorties; both None at the trial's first
      (critic A3: one snapshot cannot tell a rising phase from a falling
      one);
    * ``period_s``: P_c, the contact channel's interference period (its
      configuration, not its phases);
    * ``budget_end`` (absolute; None without a budget) and ``budget_s``, the
      mission budget's length (None without one);
    * ``t_ref_s``: T_nom, the T of the plan score;
    * ``energy_j``: the simulated energy spent this sortie, and
      ``energy_ref_j`` the reference ``FerryRuntime.l1_state`` uses: the
      capacity if one is set, else P_hover times the budget, else None. A 0
      reference (P_hover = 0) is none there, and is stored as None here;
    * ``stops``: the candidates for the next half (:class:`StopContext`),
      the remainder in its order, or home alone;
    * ``demand``: N, the devices the mission's plan demanded, with
      ``demand_weight`` the sum of their coverage weights, and ``cap_s`` the
      cap S (None: off).

    The pairs (:attr:`pairs`) are the covering classes (:attr:`covering`)
    times the candidates, class-major in link order and then in the
    remainder's order (home is alone): the slot breaks ties to the lowest
    row, and a scorer's numbers come in this order. A device appears once
    among k and the remainder: a plan serves each device at one stop, a trim
    only removes members, and the beacon hook refuses a device already
    planned this mission.
    """

    arrival: ArrivalView
    pose: Tuple[float, float, float]
    observed_snr_db: Tuple[float, ...]
    offsets_db: Tuple[float, ...]
    previous_offsets_db: Optional[Tuple[float, ...]]
    previous_age_s: Optional[float]
    period_s: float
    clock_s: float
    budget_end: Optional[float]
    budget_s: Optional[float]
    t_ref_s: float
    energy_j: float
    energy_ref_j: Optional[float]
    stops: Tuple[StopContext, ...]
    demand: int
    demand_weight: float
    cap_s: Optional[int]

    def __post_init__(self) -> None:
        if not isinstance(self.arrival, ArrivalView):
            raise TypeError(f"arrival must be an ArrivalView, got {self.arrival!r}")
        indices = [entry.index for entry in self.arrival.classes]
        if indices != sorted(indices):
            raise ValueError(f"the arrival's classes are in link order, by index: {indices}")
        n = len(indices)
        pose = _numbers(self.pose, "pose")
        if len(pose) != 3:
            raise ValueError(f"pose is (x, y, z), got {self.pose!r}")
        if (self.previous_offsets_db is None) != (self.previous_age_s is None):
            raise ValueError(
                "previous_offsets_db and previous_age_s come together: both None before the "
                "trial's first Pass-1 observation, both set after it"
            )
        stops = tuple(self.stops)
        if not stops:
            raise ValueError("an arrival offers a next stop or home: stops is never empty")
        for ctx in stops:
            if not isinstance(ctx, StopContext):
                raise TypeError(f"stops hold StopContext entries, got {ctx!r}")
        if any(ctx.is_home for ctx in stops):
            if len(stops) != 1:
                raise ValueError("home is offered only when the remainder is empty, and alone")
        else:
            if [ctx.index for ctx in stops] != list(range(len(stops))):
                raise ValueError(
                    f"stops are the remainder in its order, indexed 0..{len(stops) - 1}: "
                    f"{[ctx.index for ctx in stops]}"
                )
            for ctx in stops:
                _per_class(ctx.pred_snr_db, n, f"stops[{ctx.index}].pred_snr_db")
        devices = list(self.arrival.devices) + [d for ctx in stops if ctx.stop is not None
                                                for d in ctx.stop.devices]
        if len(set(devices)) != len(devices):
            twice = sorted({str(d) for d in devices if devices.count(d) > 1})
            raise ValueError(f"a device appears once among the stop and the remainder: {twice}")
        values = {
            "pose": pose,
            "observed_snr_db": _per_class(self.observed_snr_db, n, "observed_snr_db"),
            "offsets_db": _per_class(self.offsets_db, n, "offsets_db"),
            "previous_offsets_db": None if self.previous_offsets_db is None else _per_class(
                self.previous_offsets_db, n, "previous_offsets_db"),
            "previous_age_s": None if self.previous_age_s is None else _nonneg(
                self.previous_age_s, "previous_age_s"),
            "period_s": _positive(self.period_s, "period_s"),
            "clock_s": _finite(self.clock_s, "clock_s"),
            "budget_end": None if self.budget_end is None else _finite(
                self.budget_end, "budget_end"),
            "budget_s": None if self.budget_s is None else _positive(self.budget_s, "budget_s"),
            "t_ref_s": _positive(self.t_ref_s, "t_ref_s"),
            "energy_j": _nonneg(self.energy_j, "energy_j"),
            "energy_ref_j": _energy_ref(self.energy_ref_j, "energy_ref_j"),
            "stops": stops,
            "demand": _positive_count(self.demand, "demand"),
            "demand_weight": _nonneg(self.demand_weight, "demand_weight"),
            "cap_s": None if self.cap_s is None else _positive_count(self.cap_s, "cap_s"),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def covering(self) -> Tuple[ArrivalClass, ...]:
        """The classes whose targets include every target of the committed class.

        FX's candidate set (``cross_heuristic.fastest_covering_class``): a
        pair's band reaches every device the committed class reaches at this
        stop, a capped one included (the Phase 5 spec, other choices 2). The
        committed class is always among them. In link order.
        """
        need = set(self.arrival.committed_entry.targets)
        return tuple(entry for entry in self.arrival.classes if need.issubset(entry.targets))

    @property
    def remainder(self) -> Tuple[ContactWaypoint, ...]:
        """The stops still to fly, in order (empty when home is offered)."""
        return tuple(ctx.stop for ctx in self.stops if ctx.stop is not None)

    @property
    def homebound(self) -> bool:
        """True when the remainder is empty, so the next half is home."""
        return self.stops[0].is_home

    @property
    def pairs(self) -> Tuple[Pair, ...]:
        """Every pair offered, class-major: the slot's rows, in tie order."""
        return tuple((entry.name, ctx.index) for entry in self.covering for ctx in self.stops)

    def row(self, band: str, next_index: Optional[int]) -> int:
        """The row of the pair (``band``, ``next_index``) in :attr:`pairs`."""
        try:
            return self.pairs.index((band, next_index))
        except ValueError:
            raise ValueError(
                f"the view offers no pair ({band!r}, {next_index!r}); its pairs: {self.pairs}"
            ) from None


@runtime_checkable
class PairScorer(Protocol):
    """What ranks the pairs at an arrival: the learned score or a scripted rule.

    The pair slot's injectable scorer (the Phase 5 spec, unit U3). The
    learned score (the pair features' adapter over the pair network, units
    U1 and U2) and the scripted references (FX's pair, the committed pair,
    HYB and ``greedy_1``, unit U3; critic A2 and B5) fill it alike, so the
    slot and its guards are one code path for all of them.

    ``score(view, mask=...)`` returns one finite number per pair of
    ``view.pairs``, in that order; higher is better (:func:`check_pair_scores`
    checks the count and the numbers). ``mask`` is ``fits_pair``'s verdict on
    every pair, in the same order, for a scorer that records or replays its
    branches (the headroom oracle's search). The scorer ranks and the slot
    picks, the masked argmax with ties to the lowest row, so whatever fills
    it chooses among admitted pairs only (Freeze principle 12), and the one
    fallback when none fits is the slot's. ``name`` labels the scorer in the
    slot's records.
    """

    name: str

    def score(self, view: PairView, *, mask: Tuple[bool, ...]) -> Sequence[float]:
        """One finite number per pair of ``view.pairs``; higher is better."""
        ...


def check_pair_scores(view: PairView, scores: Any) -> Tuple[float, ...]:
    """A scorer's numbers checked against ``view``: one finite float per pair, in order."""
    if not isinstance(view, PairView):
        raise TypeError(f"view must be a PairView, got {view!r}")
    out = _numbers(scores, "score")
    if len(out) != len(view.pairs):
        raise ValueError(
            f"a scorer gives one number per pair: {len(view.pairs)} pairs, {len(out)} scores"
        )
    return out


@dataclass(frozen=True)
class PairChoice:
    """What the pair slot decided at an arrival, and the decision's record.

    ``band`` is the class stop k is served on, at once. ``next_index`` is the
    stop of the remainder flown next, which the supervisor moves to the
    front after the stop (``cross_heuristic.moved_to_front``; 0 keeps the
    plan's order), or None for home, which only an empty remainder offers;
    the departure check then folds that order and may trim it (the Phase 5
    spec, other choices 1). ``total`` counts the pairs the view offered and
    ``feasible`` those ``fits_pair`` admitted.

    When none fits, ``fallback`` is ``mask_empty`` and the pair is FX's: its
    band, the fastest covering class, with no reorder (``next_index`` 0, or
    None for home), so the fallback never moves a stop. Otherwise
    ``fallback`` is None and the pair is one the mask admitted, which the
    slot guarantees. ``q`` is the chosen pair's score where the scorer's
    numbers mean one (None otherwise); ``fx_band`` and ``fx_next`` are FX's
    pair at this arrival, kept for the agreement share. ``record`` is the
    decision's trace entry (``mission_completed.pass_1_pairs``), JSON-ready
    and without a wall time (critic B12), and read-only all the way down (its
    lists come back as tuples and its maps as read-only maps), so nothing gets
    past those checks once they have run; :meth:`describe` gives a fresh plain
    copy, lists and dicts, to write, or to complete at the mission's close.
    """

    band: str
    next_index: Optional[int]
    total: int
    feasible: int
    fallback: Optional[str]
    q: Optional[float]
    fx_band: str
    fx_next: Optional[int]
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        _name(self.band, "band")
        _name(self.fx_band, "fx_band")
        object.__setattr__(self, "next_index", _index(self.next_index, "next_index"))
        object.__setattr__(self, "fx_next", _index(self.fx_next, "fx_next"))
        total = _positive_count(self.total, "total")
        feasible = _count(self.feasible, "feasible")
        if feasible > total:
            raise ValueError(f"feasible ({feasible}) counts pairs among the total ({total})")
        if self.fallback is not None:
            _choice(self.fallback, PAIR_FALLBACKS, "fallback")
        if (self.fallback == PAIR_FALLBACK_MASK_EMPTY) != (feasible == 0):
            raise ValueError(
                f"fallback {self.fallback!r} with {feasible} feasible pair(s): the slot falls "
                f"back exactly when no pair fits"
            )
        if self.fallback is not None and (
                self.band != self.fx_band or self.next_index not in (0, None)):
            raise ValueError(
                f"the {self.fallback!r} fallback flies FX's band with no reorder, got "
                f"({self.band!r}, {self.next_index!r}) with FX's band {self.fx_band!r}"
            )
        if self.q is not None:
            object.__setattr__(self, "q", _finite(self.q, "q"))
        if not isinstance(self.record, collections.abc.Mapping):
            raise TypeError(f"record must be a mapping, got {self.record!r}")
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "feasible", feasible)
        object.__setattr__(self, "record", _frozen(_json_record(self.record, "record")))

    @property
    def pair(self) -> Pair:
        return (self.band, self.next_index)

    @property
    def agrees_fx(self) -> bool:
        """True when the pair is FX's pair at this arrival."""
        return self.pair == (self.fx_band, self.fx_next)

    def describe(self) -> Dict[str, Any]:
        """A fresh, JSON-ready copy of :attr:`record`, with plain lists and dicts."""
        return _json_record(self.record, "record")


# --------------------------------------------------------------------------- #
# Options and the mule's setup
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PlanOptions:
    """A plan-mode arm's settings: the ``MuleConfig`` plan fields, validated.

    The defaults are arm F's: search every band class, member subsets, the
    committed flight slot, no cap, and the default score and search settings.

    A pinned band (``fixed:<class>``) flies the ``committed`` slot only. FB+c
    flies only class c (Phase 4 spec, decision 7), while ``cross_heuristic``
    switches band on arrival (decision 5), and FX without that switch is not
    FX (decision 5 (c)). No arm pairs the two (spec, other choices 11), so the
    pair is refused rather than given a meaning of its own. FeRRy Phase 5's
    ``pair_q`` chooses the band on arrival too, so a pinned band refuses it by
    the same rule (the Phase 5 spec's switches table): with one class the
    pair would be the stop alone.
    """

    band_class_policy: str = BAND_POLICY_SEARCH
    member_admission: str = MEMBER_ADMISSION_SUBSET
    flight_slot: str = FLIGHT_SLOT_COMMITTED
    cap: AgeCapSpec = field(default_factory=AgeCapSpec)
    score: PlanScoreParams = field(default_factory=PlanScoreParams)
    search: PlanSearchParams = field(default_factory=PlanSearchParams)

    def __post_init__(self) -> None:
        pinned = parse_band_policy(self.band_class_policy)
        _choice(self.member_admission, MEMBER_ADMISSIONS, "member_admission")
        _choice(self.flight_slot, FLIGHT_SLOTS, "flight_slot")
        if pinned is not None and self.flight_slot != FLIGHT_SLOT_COMMITTED:
            raise ValueError(
                f"band_class_policy {self.band_class_policy!r} pins one class, but the "
                f"{self.flight_slot!r} flight slot switches band on arrival: a pinned band "
                f"flies the {FLIGHT_SLOT_COMMITTED!r} slot (FB+c flies only class c)"
            )
        for name, kind in (("cap", AgeCapSpec), ("score", PlanScoreParams),
                           ("search", PlanSearchParams)):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"{name} must be a {kind.__name__}, got {getattr(self, name)!r}")

    @property
    def fixed_band(self) -> Optional[str]:
        """The class a ``fixed:<class>`` policy pins; None under ``search``."""
        return parse_band_policy(self.band_class_policy)

    @classmethod
    def from_config(
        cls,
        *,
        band_class_policy: str = BAND_POLICY_SEARCH,
        member_admission: str = MEMBER_ADMISSION_SUBSET,
        flight_slot: str = FLIGHT_SLOT_COMMITTED,
        age_cap_missions: Optional[int] = None,
        age_cap_lookahead: int = 0,
        plan_score_params: Optional[Mapping[str, Any]] = None,
        plan_search_params: Optional[Mapping[str, Any]] = None,
    ) -> "PlanOptions":
        """The options from ``MuleConfig``'s plan fields, which keep these names."""
        return cls(
            band_class_policy=band_class_policy,
            member_admission=member_admission,
            flight_slot=flight_slot,
            cap=AgeCapSpec(s_missions=age_cap_missions, lookahead=age_cap_lookahead),
            score=PlanScoreParams.from_mapping(plan_score_params),
            search=PlanSearchParams.from_mapping(plan_search_params),
        )

    def describe(self) -> Dict[str, Any]:
        """JSON-ready, keyed by the config fields (``mule_ready`` in plan mode);
        ``from_config(**describe())`` gives these options back."""
        return {
            "band_class_policy": self.band_class_policy,
            "flight_slot": self.flight_slot,
            "member_admission": self.member_admission,
            "age_cap_missions": self.cap.s_missions,
            "age_cap_lookahead": self.cap.lookahead,
            "plan_score_params": self.score.as_dict(),
            "plan_search_params": self.search.as_dict(),
        }


@dataclass(frozen=True)
class PlanSetup:
    """What the mule gives the planner once (``FLScheduler(plan=...)``, U5 and U7).

    ``classes`` has one :class:`PlanClass` per class of the link, in link
    order; the policy picks the ones searched (:attr:`searched`).
    ``reference`` is the run's ``contact_band``: in ``search`` mode only the
    reference class, which the ferry spec, T_nom, D4's split and the other arms
    of the CSV still use; under ``fixed:<class>`` it must be that class (spec,
    other choices 2). ``t_ref_s`` is T in the score, the cell's T_nom
    (``t_nom_s``, which plan mode requires; decision 2 (b)), and
    ``turnaround_s`` the dock turnaround that Δ counts between the passes.
    P_hover, which scales the energy term, is the classes' own
    (:attr:`p_hover_w`): one energy model prices every class.
    """

    options: PlanOptions
    classes: Tuple[PlanClass, ...]
    reference: str
    t_ref_s: float
    turnaround_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.options, PlanOptions):
            raise TypeError(f"options must be PlanOptions, got {self.options!r}")
        classes = tuple(self.classes)
        if not classes:
            raise ValueError("the plan needs at least one band class")
        for c in classes:
            if not isinstance(c, PlanClass):
                raise TypeError(f"classes hold PlanClass entries, got {c!r}")
        names = [c.name for c in classes]
        if len(set(names)) != len(names) or len({c.index for c in classes}) != len(classes):
            raise ValueError(f"classes are distinct, by name and by index: {names}")
        object.__setattr__(self, "classes", classes)
        if self.reference not in names:
            raise ValueError(f"the reference class {self.reference!r} is not among {names}")
        fixed = self.options.fixed_band
        if fixed is not None and fixed != self.reference:
            raise ValueError(
                f"band_class_policy {self.options.band_class_policy!r} must pin the run's "
                f"contact_band {self.reference!r}"
            )
        t_ref = _finite(self.t_ref_s, "t_ref_s")
        if t_ref <= 0.0:
            raise ValueError(f"t_ref_s must be > 0, got {self.t_ref_s!r}")
        object.__setattr__(self, "t_ref_s", t_ref)
        object.__setattr__(self, "turnaround_s", _nonneg(self.turnaround_s, "turnaround_s"))
        powers = {float(getattr(c.model.ferry, "p_hover_w", math.nan)) for c in classes}
        if len(powers) != 1 or not all(math.isfinite(p) and p >= 0.0 for p in powers):
            raise ValueError(f"every class must price hovering at one power, got {sorted(powers)}")
        if self.options.score.c_energy > 0.0 and self.p_hover_w <= 0.0:
            raise ValueError("the energy term E/(P_hover·T) needs P_hover > 0 when c_energy > 0")

    @property
    def p_hover_w(self) -> float:
        return float(self.classes[0].model.ferry.p_hover_w)

    @property
    def searched(self) -> Tuple[PlanClass, ...]:
        """The classes the arm may fly: all under ``search``, the pinned one under ``fixed:``."""
        fixed = self.options.fixed_band
        return self.classes if fixed is None else (self.class_named(fixed),)

    def class_named(self, name: str) -> PlanClass:
        """The class named ``name``; KeyError if the link has none."""
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(name)
