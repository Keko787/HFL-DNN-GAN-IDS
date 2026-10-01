"""Scheduler-local state types.

Unlike the wire types in ``fl_messages`` and ``bundles``, these never
cross a tier boundary — they live entirely inside ``FLScheduler`` on the
mule NUC. Putting them in ``hermes.types`` anyway keeps one import root
for the rest of the build.

Design refs:
* HERMES_FL_Scheduler_Design.md §6.2 FLScheduler state
* HERMES_FL_Scheduler_Design.md §4 (bucket tags)
"""

from __future__ import annotations

import collections.abc
import math
import numbers
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

from .ids import DeviceID
from .round_report import MissionOutcome


#: A new device's fulfilment window Φ₀ in seconds: the default of
#: ``DeviceSchedulerState.deadline_fulfilment_s``. A placeholder, not a tuned
#: constant (design §9 Q1, critic A7). FeRRy Phase 3 lets the scheduler restate
#: it (``FLScheduler(initial_window_s=...)``, in this same recorded unit) and
#: multiplies it by ``FLScheduler(deadline_time_scale=...)`` like every other
#: constant of the deadline law.
DEFAULT_FULFILMENT_WINDOW_S: float = 60.0


class Bucket(str, Enum):
    """Coarse priority tag produced by S3's bucket classifier.

    Design §4: the scheduler has no intra-bucket rank of its own — S3.5
    (``TargetSelectorRL``, Phase 5) provides that. For Phase 4 the
    placeholder orders by ``last_known_distance`` inside each bucket.
    """

    NEW = "new"                          # registered but never served
    SCHEDULED_THIS_ROUND = "scheduled"   # in slice, has active deadline
    BEACON_ACTIVE = "beacon_active"      # recent beacon heard, opportunistic


class MissionPass(str, Enum):
    """Which half of a two-pass HERMES mission is running.

    Design §7 principle 13: a mission is Pass 1 (collect Δθ) + dock +
    Pass 2 (deliver fresh θ). Pass 1 runs scheduler-driven contact
    selection; Pass 2 walks every contact greedily for universal delivery.
    """

    COLLECT = "collect"  # Pass 1 — pull pre-prepared Δθ from devices
    DELIVER = "deliver"  # Pass 2 — push fresh θ_disc to every device


# Buckets visit-order (design §4): new first, then scheduled, then beacon
BUCKET_PRIORITY: Tuple[Bucket, ...] = (
    Bucket.NEW,
    Bucket.SCHEDULED_THIS_ROUND,
    Bucket.BEACON_ACTIVE,
)


@dataclass
class DeviceSchedulerState:
    """Scheduler's per-device view.

    Populated from:
    * initial ``MissionSlice`` (DOWN bundle) -> ``is_in_slice``, ``is_new``
    * ``RoundCloseDelta`` (fast-phase bus) -> ``last_outcome``, ``idle_time_ref_ts``
    * ``ClusterAmendment`` (slow-phase at dock) -> ``deadline_override``
    * RF beacons -> ``last_beacon_ts``, bucket may flip to ``BEACON_ACTIVE``
    """

    device_id: DeviceID
    is_in_slice: bool = False
    is_new: bool = True

    # Contact / outcome history (this mule only)
    last_outcome: Optional[MissionOutcome] = None
    #: Timestamp of this device's most recent outcome of ANY kind, including
    #: the synthetic TIMEOUT fed for a device the mule abandoned without a
    #: contact. Not an age source: use ``last_clean_ts`` for that.
    last_contact_ts: float = 0.0
    last_utility: float = 0.0
    # Running tallies that mirror DeviceRecord.on_time_history /
    # missed_history but live mule-side. The S3.5 selector reads
    # on_time_count / (on_time_count + missed_count) as its continuous
    # reliability proxy — the binary `last_outcome` was too noisy a
    # signal for the DDQN to separate flaky from reliable devices.
    on_time_count: int = 0
    missed_count: int = 0

    # Sprint 1.5: cached copy of DeviceRecord.delivery_priority pulled
    # from the most recent DOWN bundle. S3a clustering reads this as a
    # tie-breaker so a device that went UNDELIVERED in the previous
    # mission's Pass 2 gets pulled toward a cluster anchor early in the
    # next slice's circuit. Reset to 0 on a clean delivery via
    # MissionDeliveryReport ingest at the cluster.
    delivery_priority: int = 0

    # Deadline machinery (see Design §6.2 formula)
    deadline_fulfilment_s: float = DEFAULT_FULFILMENT_WINDOW_S   # default window (design §9 Q1 open)
    idle_time_ref_ts: float = 0.0         # last on-time participation ts
    deadline_override_ts: Optional[float] = None  # from ClusterAmendment

    # Oort baseline inputs (arm B2) — the raw training loss and sample count
    # from this device's most recent contact, folded in beside `last_utility`.
    # Retrospective by construction: they describe the LAST time we visited,
    # which is the only client state a data mule can hold. `None` means "never
    # served, or this arm does not carry them".
    last_loss: Optional[float] = None
    last_num_examples: int = 0
    #: Mission round of this device's most recent outcome of ANY kind,
    #: including failed and abandoned attempts. 0 = no outcome yet. The Oort
    #: arm derives its current round from this; its ``L(i)`` is
    #: ``last_clean_round``.
    last_served_round: int = 0

    # Baseline age inputs (arms D1/D2) — set ONLY on a CLEAN outcome, so a
    # failed or abandoned attempt never counts as service. MAX-AoI ages a
    # device from ``last_clean_ts``; Oort's ``L(i)`` is ``last_clean_round``.
    # 0 = never participated successfully. Inert for H0–H3, which read neither.
    last_clean_ts: float = 0.0
    last_clean_round: int = 0

    #: Consecutive non-CLEAN outcomes since the last CLEAN, synthetic TIMEOUTs
    #: for dropped or abandoned devices included. FeRRy's priority key: with
    #: ``miss_priority`` on, S3b admits contacts with a longer streak first.
    #: Maintained always; nothing reads it otherwise.
    miss_streak: int = 0

    # FeRRy Phase 2 — inputs for the Whittle baseline (arm D3), maintained
    # always and read by nothing else. ``reach_attempts`` counts real Pass-1
    # outcomes (the synthetic TIMEOUT for a device the mule dropped or
    # abandoned is not an attempt); ``reach_answered`` those whose device
    # answered (its advert arrived), whatever the outcome — so their ratio
    # estimates reachability, not reliability. ``last_merged_round`` is the
    # mission whose merge last used this device's update (None = never), the
    # device's Age-of-Update anchor; a CLEAN whose update the age cutoff
    # excluded does not move it.
    reach_attempts: int = 0
    reach_answered: int = 0
    last_merged_round: Optional[int] = None

    # RF / opportunistic
    last_beacon_ts: float = 0.0

    # Output of S3's bucket classifier (set by FLScheduler each pipeline pass)
    bucket: Optional[Bucket] = None

    # Last-known position for S3.5 placeholder ordering (from DeviceRecord)
    last_known_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    #: FeRRy Phase 3 (design §4.7, plumbing only): the latest SNR the cluster
    #: saw for this device per contact band class (``"wide"``, ``"medium"``,
    #: ``"narrow"``), folded from ``registry_deltas[did]["spectrum_sig"]`` at
    #: dock. None = never reported, which is every legacy run: legacy DOWNs
    #: never carry the key. Nothing in Phase 3 decides on it.
    spectrum_snr_db: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class BeaconObservation:
    """One RF beacon event observed by this mule.

    Used as the bus payload between the L1 RF listener and the scheduler.
    Mirrors the minimum info an RF front-end can confidently report.
    """

    device_id: DeviceID
    observed_at: float
    snr: float = 0.0


@dataclass(frozen=True)
class TargetWaypoint:
    """One entry in the scheduler's output visit queue (per-device, pre-Sprint-1.5).

    Retained for backward compatibility with the Phase-4 deterministic
    pipeline + Sprint-1A `MuleSupervisor` tests. After Sprint 1.5 the
    scheduler emits ``ContactWaypoint`` instead, which covers N≥1
    devices per stop. ``TargetWaypoint`` is the degenerate N=1 case.
    """

    device_id: DeviceID
    position: Tuple[float, float, float]
    bucket: Bucket
    deadline_ts: float


@dataclass(frozen=True)
class ContactWaypoint:
    """One contact-event entry in the scheduler's output queue.

    Sprint 1.5 design §7 principle 15: the mule's circuit is decomposed
    into contact events, where each stop covers all devices within
    ``rf_range_m`` of the position. The selector picks among
    ``ContactWaypoint``s, not individual devices. The N=1 case (isolated
    device) is the degenerate-but-valid form of the same payload.

    ``devices`` is the list of in-range slice members the mule will
    serve in parallel at this stop. ``bucket`` is inherited from the
    *worst* bucket among the members (so a NEW-and-SCHEDULED mix is
    treated as NEW, drained first). ``deadline_ts`` is the *tightest*
    deadline among the members — if any member's deadline is overdue,
    the contact inherits that pressure.

    FeRRy Phase 3 (design §4.5) adds three annotations the mule fills with
    ``dataclasses.replace`` right after planning: the contact ``band`` class,
    its planar ``range_m`` R_planar(b), and each member's predicted SNR
    ``pred_snr_db`` (in ``devices`` order). They describe the stop; they do
    not identify it, so all three are left out of equality and hashing
    (``compare=False``): a waypoint annotated or not is the same dictionary
    key and set member, which the walks and ``MuleSupervisor._budget_pass_2``
    rely on.
    """

    position: Tuple[float, float, float]
    devices: Tuple[DeviceID, ...]
    bucket: Bucket
    deadline_ts: float
    band: Optional[str] = field(default=None, compare=False)
    range_m: Optional[float] = field(default=None, compare=False)
    pred_snr_db: Optional[Tuple[float, ...]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("ContactWaypoint must cover ≥1 device")


# --------------------------------------------------------------------------- #
# FeRRy Phase 4: the plan commit (build plan Fig. 1, "Commit the plan")
# --------------------------------------------------------------------------- #

#: Why a capped device went a mission without a merged update (FeRRy Phase 4,
#: build-plan decision D4 as the user decided it on 2026-09-30). A device is
#: capped when its age, counted in its own mule's missions since its last
#: merged update (``last_merged_round``; never merged counts as 0), reaches the
#: cap S less the lookahead (``hermes.scheduler.plan.AgeCapSpec``); the plan
#: must then serve it. The first two reasons are given when the plan is built,
#: to every capped device it leaves out: ``unplannable`` when no band class the
#: arm may fly serves the device alone from the dock at takeoff within the time
#: budget, at the stop the plan offers it (its S3a stop, or its best hover point
#: when that stop cannot serve it alone; ``hermes.scheduler.plan.hover``; under
#: an energy capacity this reflects the time-minimising hover point), and
#: ``crowded`` when one could but the chosen plan does not. The last two are given when the mission
#: closes, to the capped devices the plan did serve: ``dropped_in_flight`` when
#: the in-flight check, re-plan or trim left the device out, ``not_merged`` when
#: it was visited but its update did not reach the mule's merge. Violations are
#: reported by cause, with no pass mark: device availability alone makes about
#: 15 % of device-missions miss at S = 3 (Phase 4 spec, decision 7; critic A2).
CAP_UNPLANNABLE = "unplannable"
CAP_CROWDED = "crowded"
CAP_DROPPED_IN_FLIGHT = "dropped_in_flight"
CAP_NOT_MERGED = "not_merged"
CAP_PLAN_REASONS: Tuple[str, ...] = (CAP_UNPLANNABLE, CAP_CROWDED)
CAP_CLOSE_REASONS: Tuple[str, ...] = (CAP_DROPPED_IN_FLIGHT, CAP_NOT_MERGED)
CAP_REASONS: Tuple[str, ...] = CAP_PLAN_REASONS + CAP_CLOSE_REASONS

#: The terms every committed plan's ``score`` records, in the field order of
#: ``hermes.scheduler.plan.ScoreTerms``: V, the time Δ that V prices (the whole
#: mission on the chosen band, less the dwell under F-dwell; the predicted
#: mission itself is recorded as ``mission_s`` for every arm), V's four terms
#: unweighted ((Δ/T)², the coverage shortfall U, the expected link
#: loss L, the energy E/(P_hover·T)), E in joules, and the served and demanded
#: coverage weights.
PLAN_SCORE_KEYS: Tuple[str, ...] = (
    "v", "delta_s", "time", "coverage", "link", "energy_j", "energy",
    "served_weight", "demand_weight",
)

#: ``band_class_policy``: ``search`` searches every band class of the link each
#: mission (arm F); ``fixed:<class>`` pins one class (Path B+, the FB+ arms),
#: which must be the run's ``contact_band`` (Phase 4 spec, other choices 2).
#: The commit records the policy and must fly the class it pins, so the policy
#: is defined here, beside the commit; ``hermes.scheduler.plan`` re-exports it.
BAND_POLICY_SEARCH = "search"
BAND_POLICY_FIXED_PREFIX = "fixed:"

#: The search that produced a class's candidates (Phase 4 spec, other choices
#: 3), which the commit records. ``exact`` enumerates every ordered stop
#: sequence of the class with each stop reduced to every non-empty subset of
#: its members, when the demand has at most ``exact_max_devices`` devices;
#: ``stop_subsets`` searches ordered stop subsets depth-first, each stop whole
#: if it fits and else reduced greedily, when the class has at most
#: ``exhaustive_max_stops`` stops; ``local`` builds a 2-OPT tour, trims it and
#: improves it by local search, within an evaluation count. Only ``exact`` is
#: optimal over the member subsets V prices; above it the restriction is a
#: recorded plan deviation.
SEARCH_EXACT = "exact"
SEARCH_STOP_SUBSETS = "stop_subsets"
SEARCH_LOCAL = "local"
SEARCH_MODES: Tuple[str, ...] = (SEARCH_EXACT, SEARCH_STOP_SUBSETS, SEARCH_LOCAL)


def parse_band_policy(policy: Any) -> Optional[str]:
    """The class ``fixed:<class>`` pins, or None for ``search``.

    Raises ValueError for anything else. Which classes exist is the link's
    (``hermes.l1``) to say, so the name is checked against the classes by
    ``hermes.scheduler.plan.PlanSetup``, not here.
    """
    if isinstance(policy, str):
        if policy == BAND_POLICY_SEARCH:
            return None
        if policy.startswith(BAND_POLICY_FIXED_PREFIX):
            band = policy[len(BAND_POLICY_FIXED_PREFIX):]
            if band and band == band.strip() and ":" not in band:
                return band
    raise ValueError(
        f"band_class_policy must be {BAND_POLICY_SEARCH!r} or "
        f"'{BAND_POLICY_FIXED_PREFIX}<class>', got {policy!r}"
    )


def fixed_band_policy(band: str) -> str:
    """The policy that pins ``band``, ``fixed:<band>`` (the FB+ arms)."""
    policy = f"{BAND_POLICY_FIXED_PREFIX}{band}"
    parse_band_policy(policy)
    return policy


def _count(value: Any, name: str) -> int:
    """``value`` as an int >= 0; a bool is refused (``True`` is no count)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return int(value)


def _finite(value: Any, name: str) -> float:
    """``value`` as a finite float; a bool is refused."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _device_ids(values: Iterable[Any], name: str) -> Tuple[DeviceID, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} is a collection of device ids, not one string: {values!r}")
    out = tuple(values)
    for did in out:
        if not isinstance(did, str) or not did:
            raise TypeError(f"{name} holds device ids (non-empty strings), got {did!r}")
    return out  # type: ignore[return-value]


def _json_ready(value: Any, name: str) -> Any:
    """A fresh copy of ``value`` with tuples as lists; refuses what JSON cannot
    carry exactly (non-finite floats, non-string keys, other objects)."""
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
            out[key] = _json_ready(item, f"{name}[{key!r}]")
        return out
    if isinstance(value, (list, tuple)):
        return [_json_ready(item, name) for item in value]
    raise TypeError(f"{name} must be JSON-ready (numbers, strings, lists, dicts), got {value!r}")


def _refuse_wall_keys(value: Any, name: str) -> None:
    """Critic B12: the planner's wall time travels in ``plan_wall_s``, outside
    the commit, so that a repeated trial describes the same plan; a key that
    names a wall time is refused wherever the planner may add one."""
    if isinstance(value, collections.abc.Mapping):
        for key, item in value.items():
            if "wall" in str(key):
                raise ValueError(
                    f"{name}: {key!r} names a wall time; the commit holds none (plan_wall_s "
                    f"is a trace field of its own, critic B12)"
                )
            _refuse_wall_keys(item, name)
    elif isinstance(value, list):
        for item in value:
            _refuse_wall_keys(item, name)


def plan_class_summaries(
    entries: Iterable[Any], chosen: Optional[str] = None,
) -> Tuple[Mapping[str, Any], ...]:
    """The search's per-class summaries as both of their holders check them.

    The search's result (``hermes.scheduler.plan.SearchResult``, U4) and the
    commit (:class:`PlanCommit`, U5) hold the same summaries, and the commit
    writes them to the trace, so one check serves both: a summary the commit
    would refuse fails the search's own result, not the commit mid-mission.
    Each summary names its class under ``band``, one summary per class
    searched, JSON-ready and without a wall time (critic B12); when any is
    given, the ``chosen`` class has one. Returns read-only JSON-ready copies.
    """
    out = []
    for entry in entries:
        band = entry.get("band") if isinstance(entry, collections.abc.Mapping) else None
        if not isinstance(band, str) or not band:
            raise TypeError(
                f"each per_class entry is a mapping naming its class under 'band', got {entry!r}"
            )
        summary = _json_ready(entry, "per_class")
        _refuse_wall_keys(summary, "per_class")
        out.append(MappingProxyType(summary))
    bands = [summary["band"] for summary in out]
    if len(set(bands)) != len(bands):
        raise ValueError(f"per_class holds one summary per class searched, got {bands}")
    if chosen is not None and out and chosen not in bands:
        raise ValueError(f"per_class has no summary of the chosen class {chosen!r}: {bands}")
    return tuple(out)


@dataclass(frozen=True, order=True)
class CapViolation:
    """One capped device a mission failed, and why (:data:`CAP_REASONS`).

    ``age`` is the device's age when the plan was built (see
    :data:`CAP_UNPLANNABLE`). Sorting orders by device, then age, then reason.
    """

    device: DeviceID
    age: int
    reason: str

    def __post_init__(self) -> None:
        _device_ids((self.device,), "device")
        object.__setattr__(self, "age", _count(self.age, "age"))
        if self.reason not in CAP_REASONS:
            raise ValueError(f"reason must be one of {CAP_REASONS}, got {self.reason!r}")

    def describe(self) -> Dict[str, Any]:
        """JSON-ready: ``device``, ``age``, ``reason``."""
        return {"device": str(self.device), "age": self.age, "reason": self.reason}


@dataclass(frozen=True)
class PlanCommit:
    """What a plan-mode mission committed to at the dock (FeRRy Phase 4).

    The build plan's commit, "(b̄, ordered queue, budget) for this mule's
    slice" (Fig. 1, L359-360; L535): ``band`` b̄ and its ``band_index``, the
    ``queue`` to fly (S3a stops, some reduced to a subset of their members,
    and one-device stops at the best hover point of a capped device its S3a
    stop could not serve alone),
    ``budget_end`` (absolute, on the mission clock; None without a budget),
    and what the choice rested on: the ``demand`` (the devices left after S1
    and S3, in S3's order) and each one's coverage ``weights``, the ``score``
    (:data:`PLAN_SCORE_KEYS`, plus any other finite terms the planner adds)
    with the resolved ``constants`` (c₁, c₂, c₃, c₄) and ``t_ref_s`` (T, the
    cell's T_nom: the user's decision 2 (b)), the ``search_mode`` of the
    committed class (:data:`SEARCH_MODES`), ``n_candidates`` and one
    JSON-ready summary per class searched (``per_class``, see
    :func:`plan_class_summaries`), and the age cap: ``cap_s`` S (None: off),
    ``cap_lookahead``, every demanded device's ``ages`` (or none, when nothing
    needed them), the ``capped`` set and the ``violations``.

    The commit flies the class its ``band_class_policy`` pins, if it pins one:
    FB+c flies only class c (Phase 4 spec, decision 7). A cap needs the
    ``mission_round``, because ages count the mule's missions (critic B9).

    The plan serves exactly the members of its stops (:attr:`served`), a
    subset of the demand. Every capped device the plan leaves out carries one
    plan-time violation (``unplannable`` or ``crowded``). When the mission
    ends, :meth:`close` records ``visited``, the devices of the stops actually
    flown (the plan's "visited set", L475), and the close-time violations:
    ``dropped_in_flight`` for exactly the capped served devices not visited,
    ``not_merged`` only for visited ones. A device violates at most once per
    mission.

    Frozen and checked: :meth:`close` returns a checked copy rather than
    editing this one, so the scheduler replaces its ``last_plan`` with it.
    It holds no wall time: :meth:`describe` must come out identical when a
    trial is repeated, so the planner's wall time travels in a trace field of
    its own (``plan_wall_s``, critic B12).
    """

    mission_round: Optional[int]
    band: str
    band_index: int
    band_class_policy: str
    queue: Tuple[ContactWaypoint, ...]
    demand: Tuple[DeviceID, ...]
    weights: Mapping[DeviceID, float]
    budget_end: Optional[float]
    t_ref_s: float
    score: Mapping[str, float]
    constants: Tuple[float, float, float, float]
    search_mode: str
    n_candidates: int
    per_class: Tuple[Mapping[str, Any], ...] = ()
    cap_s: Optional[int] = None
    cap_lookahead: int = 0
    ages: Mapping[DeviceID, int] = field(default_factory=dict)
    capped: FrozenSet[DeviceID] = frozenset()
    violations: Tuple[CapViolation, ...] = ()
    visited: Optional[FrozenSet[DeviceID]] = None

    def __post_init__(self) -> None:
        if self.mission_round is not None:
            object.__setattr__(self, "mission_round", _count(self.mission_round, "mission_round"))
        if not isinstance(self.band, str) or not self.band:
            raise TypeError(f"band must be a non-empty string, got {self.band!r}")
        pinned = parse_band_policy(self.band_class_policy)
        if pinned is not None and pinned != self.band:
            raise ValueError(
                f"band_class_policy {self.band_class_policy!r} pins {pinned!r}, but the "
                f"commit flies {self.band!r}: FB+c flies only class c (Phase 4 spec, decision 7)"
            )
        if self.search_mode not in SEARCH_MODES:
            raise ValueError(f"search_mode must be one of {SEARCH_MODES}, got {self.search_mode!r}")
        object.__setattr__(self, "band_index", _count(self.band_index, "band_index"))
        object.__setattr__(self, "n_candidates", _count(self.n_candidates, "n_candidates"))

        queue = tuple(self.queue)
        for wp in queue:
            if not isinstance(wp, ContactWaypoint):
                raise TypeError(f"queue holds ContactWaypoints, got {wp!r}")
        members = [d for wp in queue for d in wp.devices]
        if len(set(members)) != len(members):
            twice = sorted({str(d) for d in members if members.count(d) > 1})
            raise ValueError(f"a device is served by one stop of a plan at most: {twice}")
        object.__setattr__(self, "queue", queue)

        demand = _device_ids(self.demand, "demand")
        if len(set(demand)) != len(demand):
            raise ValueError(f"demand lists each device once, got {demand!r}")
        object.__setattr__(self, "demand", demand)
        outside = sorted(set(members) - set(demand))
        if outside:
            raise ValueError(f"the plan serves devices outside its demand: {outside}")

        weights = dict(self.weights)
        if set(weights) != set(demand):
            raise ValueError(
                f"weights must hold exactly the demanded devices; missing "
                f"{sorted(set(demand) - set(weights))}, extra {sorted(map(str, set(weights) - set(demand)))}"
            )
        normalized: Dict[DeviceID, float] = {}
        for did in demand:
            w = _finite(weights[did], f"weights[{did!r}]")
            if w < 0.0:
                raise ValueError(f"weights[{did!r}] must be >= 0, got {w!r}")
            normalized[did] = w
        object.__setattr__(self, "weights", MappingProxyType(normalized))

        if self.budget_end is not None:
            object.__setattr__(self, "budget_end", _finite(self.budget_end, "budget_end"))
        t_ref = _finite(self.t_ref_s, "t_ref_s")
        if t_ref <= 0.0:
            raise ValueError(f"t_ref_s must be > 0, got {self.t_ref_s!r}")
        object.__setattr__(self, "t_ref_s", t_ref)

        score = dict(self.score)
        missing = [k for k in PLAN_SCORE_KEYS if k not in score]
        if missing:
            raise ValueError(f"score lacks {missing}; it records {PLAN_SCORE_KEYS}")
        terms = {k: _finite(score[k], f"score[{k!r}]") for k in PLAN_SCORE_KEYS}
        for key in sorted(set(score) - set(PLAN_SCORE_KEYS), key=str):
            if not isinstance(key, str) or not key:
                raise TypeError(f"score keys are non-empty strings, got {key!r}")
            if key in ("c", "t_ref_s"):
                raise ValueError(f"score[{key!r}] is reserved: describe() writes it")
            terms[key] = _finite(score[key], f"score[{key!r}]")
        _refuse_wall_keys(terms, "score")
        object.__setattr__(self, "score", MappingProxyType(terms))

        constants = tuple(self.constants)
        if len(constants) != 4:
            raise ValueError(f"constants are (c1, c2, c3, c4), got {self.constants!r}")
        resolved = tuple(_finite(c, "constants") for c in constants)
        if any(c < 0.0 for c in resolved):
            raise ValueError(f"constants must be >= 0, got {self.constants!r}")
        object.__setattr__(self, "constants", resolved)

        object.__setattr__(self, "per_class", plan_class_summaries(self.per_class, self.band))

        self._check_cap(demand, frozenset(members))

    def _check_cap(self, demand: Tuple[DeviceID, ...], served: FrozenSet[DeviceID]) -> None:
        if self.cap_s is not None:
            cap_s = _count(self.cap_s, "cap_s")
            if cap_s < 1:
                raise ValueError(f"cap_s must be >= 1 or None (off), got {self.cap_s!r}")
            if self.mission_round is None:
                raise ValueError(
                    "a cap needs the mission_round: ages count the mule's missions since "
                    "each device's last merged update (critic B9)"
                )
            object.__setattr__(self, "cap_s", cap_s)
        object.__setattr__(self, "cap_lookahead", _count(self.cap_lookahead, "cap_lookahead"))

        ages = dict(self.ages)
        if ages and set(ages) != set(demand):
            raise ValueError("ages must hold every demanded device, or none")
        if self.cap_s is not None and not ages and demand:
            raise ValueError("with a cap every demanded device has an age")
        object.__setattr__(self, "ages", MappingProxyType(
            {d: _count(ages[d], f"ages[{d!r}]") for d in demand if d in ages}
        ))

        capped = frozenset(_device_ids(self.capped, "capped"))
        if self.cap_s is None:
            if capped:
                raise ValueError("no device is capped without a cap (cap_s None)")
        else:
            threshold = self.cap_s - self.cap_lookahead
            expected = frozenset(d for d, a in self.ages.items() if a >= threshold)
            if capped != expected:
                raise ValueError(
                    f"capped must be the devices aged >= cap_s - cap_lookahead = {threshold}: "
                    f"expected {sorted(expected)}, got {sorted(capped)}"
                )
        object.__setattr__(self, "capped", capped)

        visited = None
        if self.visited is not None:
            visited = frozenset(_device_ids(self.visited, "visited"))
        object.__setattr__(self, "visited", visited)

        violations = tuple(self.violations)
        seen = set()
        for v in violations:
            if not isinstance(v, CapViolation):
                raise TypeError(f"violations hold CapViolations, got {v!r}")
            if v.device in seen:
                raise ValueError(f"device {v.device!r} violates the cap at most once per mission")
            seen.add(v.device)
            if v.device not in capped:
                raise ValueError(f"only a capped device can violate the cap, not {v.device!r}")
            if v.age != self.ages[v.device]:
                raise ValueError(
                    f"violation of {v.device!r} records age {v.age}, but its planning age "
                    f"is {self.ages[v.device]}"
                )
        at_plan = {v.device for v in violations if v.reason in CAP_PLAN_REASONS}
        if at_plan != capped - served:
            raise ValueError(
                f"every capped device the plan leaves out, and only those, carries a "
                f"plan-time violation: expected {sorted(capped - served)}, got {sorted(at_plan)}"
            )
        dropped = {v.device for v in violations if v.reason == CAP_DROPPED_IN_FLIGHT}
        not_merged = {v.device for v in violations if v.reason == CAP_NOT_MERGED}
        if visited is None:
            if dropped or not_merged:
                raise ValueError("close-time violations come with close(): visited is not set")
        else:
            if dropped != (capped & served) - visited:
                raise ValueError(
                    f"dropped_in_flight is every capped served device not visited, and "
                    f"only those: expected {sorted((capped & served) - visited)}, got {sorted(dropped)}"
                )
            # So not_merged is for visited devices only, with no check of its
            # own: its device is capped and violates once, so it is served
            # (else it owes a plan-time reason) and visited (else it owes
            # dropped_in_flight).
        object.__setattr__(self, "violations", tuple(sorted(
            violations, key=lambda v: (CAP_REASONS.index(v.reason), v.device),
        )))

    @property
    def served(self) -> FrozenSet[DeviceID]:
        """The devices the plan serves: the members of its stops."""
        return frozenset(d for wp in self.queue for d in wp.devices)

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has recorded the mission's visited set."""
        return self.visited is not None

    def close(
        self, visited: Iterable[DeviceID], violations: Iterable[CapViolation] = (),
    ) -> "PlanCommit":
        """A copy closed with the mission's ``visited`` set and close-time ``violations``.

        Only :data:`CAP_CLOSE_REASONS` may be added, and only once: a second
        close is refused. The copy re-checks every invariant.
        """
        if self.visited is not None:
            raise ValueError("the plan is already closed")
        visited = frozenset(_device_ids(visited, "visited"))
        added = tuple(violations)
        for v in added:
            if not isinstance(v, CapViolation) or v.reason not in CAP_CLOSE_REASONS:
                raise ValueError(
                    f"close() adds {CAP_CLOSE_REASONS} violations only, got {v!r}"
                )
        return replace(self, visited=visited, violations=self.violations + added)

    def describe(self) -> Dict[str, Any]:
        """The commit as ``mission_completed.plan`` records it: JSON-ready.

        Deterministic: sets become sorted lists, mappings keep the demand's
        order, and nothing is a wall time. ``score`` adds ``c`` (c₁ to c₄) and
        ``t_ref_s`` to the terms; ``visited`` is None until :meth:`close`.
        """
        score: Dict[str, Any] = dict(self.score)
        score["c"] = list(self.constants)
        score["t_ref_s"] = self.t_ref_s
        return {
            "mission_round": self.mission_round,
            "band": self.band,
            "band_index": self.band_index,
            "band_class_policy": self.band_class_policy,
            "search": self.search_mode,
            "candidates": self.n_candidates,
            "per_class": [_json_ready(entry, "per_class") for entry in self.per_class],
            "budget_end": self.budget_end,
            "score": score,
            "demand": [str(d) for d in self.demand],
            "weights": {str(d): w for d, w in self.weights.items()},
            "served": sorted(str(d) for d in self.served),
            "cap": {
                "s": self.cap_s,
                "lookahead": self.cap_lookahead,
                "ages": {str(d): a for d, a in self.ages.items()},
                "capped": sorted(str(d) for d in self.capped),
                "violations": [v.describe() for v in self.violations],
            },
            "visited": None if self.visited is None else sorted(str(d) for d in self.visited),
        }
