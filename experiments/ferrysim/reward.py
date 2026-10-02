"""FerrySim's reward: what each Pass-1 decision earns (FeRRy Phase 5, unit U8a).

**The derived reward** (the user's decision 4 (a); the Phase 5 spec, other
choices 7). A decision is made at each Pass-1 arrival at a stop k (the pair:
the class k is served on and the stop flown next). Its reward is::

    r_k = G_k - c_t * dt_k / T - c_e * dE_k / (P_hover * T)
          (- c_cov * U at the sortie's last decision)
    G_k = sum of w_i over the updates collected CLEAN at k, / (n_ref * N)
    U   = sum of omega_j over the committed devices left uncollected
          / sum of omega_j over the demand     (omega_j: the plan's coverage weights)

* w_i is the raw L3 weight the mule's merge gave the update
  (``hermes.mission.aggregation_rules.update_weights``: n_i v_i s(a_i), 0 past
  the cutoff), :func:`raw_merge_weights`; n_ref is the cell's declared example
  count and N the mission's demand, so one device of the reference size, fresh,
  is worth 1/N. At FerrySim's training cells (one mule, ``agg:cutoff``,
  ``value=uniform``, the equal device model) every collected update weighs
  n_ref, so G_k is the stop's collected count over N (critic B4).
* dt_k runs from the arrival at k to the next Pass-1 arrival, or, after the
  sortie's last decision, to the end of the Pass-1 upload (the landing on the
  empty round, which uploads nothing the merge used): the closed record's
  ``t_next_s`` (``policies.pair_slot.CLOSE_KEYS``). T is T_nom, the plan's
  T. dE_k is the energy spent over the same span by the mule's own model
  (``hermes.l1.mission_clock.EnergyModel``: hover power while it dwells and
  listens at k, flight power until the next arrival or the landing).
* The committed devices are the plan's (``PlanCommit.served``), the demand
  its ``demand`` with its coverage weights (``PlanCommit.weights``).
* Defaults c_t = 0.1, c_e = 0, c_cov = 1 (decision 4: no energy term, energy
  tracks time; no lateness term, none of 1,148 probed collections was late,
  critic B3). Study 5.7 sweeps c_t over (0.03, 0.1, 0.3) and c_cov over (0.25,
  1, 4) (:data:`GRID_C_T`, :data:`GRID_C_COV`).

**Expected availability** (critic C1; ``RewardSpec.expected_availability``).
The keyed availability draw (``FerryRuntime.uplink_drops``, keyed by trial,
device and round) does not depend on the pair, and its noise is 10 to 100
times the time signal a decision moves. Training therefore replaces each
targeted member's draw by its probability rel_j, holding the realized flight:
a member collected at k, or one whose uplink the draw dropped there, is
credited rel_j times its weight (its realized w_j when collected, n_ref, its
equal shard at age 0, when dropped), and U counts it uncollected with
probability 1 - rel_j. The policy never reads rel_j; held-out evaluation and
every reported number use the realized draw. Given the flight up to the
arrival at k, the expected credit is the realized credit's mean, so the two
returns agree in expectation (a test checks both the formula, against the
draw itself, and FerrySim's returns). The dropped member's n_ref is exact for
an update of age 0, which at K = 1 is every update but one whose device a
Pass 2 missed (the Phase 5 design, finding 4), and the training cells' equal
device model gives every update n_ref examples.

**The other two kinds.** ``hand`` is F·hand, "today's reward" (build plan
L1043), ContactSim's own ported and declared as such: r_k = (200 |C_k| - dt_k
- 0.002 m_k) / 150 in ContactSim's units (``selector/sim_env.py``:
COMPLETION_BONUS 200, the time, ENERGY_W 0.002 per metre;
``selector_train.py``: reward_scale 1/150), with m_k the metres flown over
dt_k and no terminal term. ``bytes`` is E3's: |C_k| / N (decision 7; the
payload is fixed, so bytes are updates).

A sortie's decisions are its Pass-1 stops flown, so every arm's return is
read off its flight the same way, whether a slot decided there or not (FX's
and F's arrivals are FQ's decision points). A sortie that flies no Pass-1
stop makes no decision and adds nothing; its shortfall is reported, not
charged (:attr:`SortieRecord.undecided_shortfall`).
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

REWARD_DERIVED = "derived"
REWARD_HAND = "hand"
REWARD_BYTES = "bytes"
REWARD_KINDS: Tuple[str, ...] = (REWARD_DERIVED, REWARD_HAND, REWARD_BYTES)

#: The equal device model's example count n_i, the same for every update, and
#: so the training cells' n_ref (``inprocess.equal_shard_trainer``). A round
#: number inside the stub's range [4, 15] (``hermes/processes/device.py``).
EQUAL_SHARD_EXAMPLES = 10
#: The stack's stub's mean example count, E[n_i] for n_i uniform on the
#: integers 4..15 (``hermes/processes/device.py``: ``rng.integers(4, 16)``):
#: n_ref for an episode under the stub device model.
STUB_MEAN_EXAMPLES = 9.5

#: Study 5.7's grid (decision 4): c_t x c_cov.
GRID_C_T: Tuple[float, ...] = (0.03, 0.1, 0.3)
GRID_C_COV: Tuple[float, ...] = (0.25, 1.0, 4.0)

#: F·hand's constants, ContactSim's own (``selector/sim_env.py``:
#: COMPLETION_BONUS, ENERGY_W with energy_weight 1; ``selector_train.py``:
#: reward_scale = 1/150).
HAND_PER_UPDATE = 200.0
HAND_PER_METRE = 0.002
HAND_SCALE = 150.0

_EPS_SPAN = 1e-9


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _nonneg(value: Any, name: str) -> float:
    out = _real(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return out


def _ids(values: Any, name: str) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} lists device ids, got {values!r}")
    out = tuple(str(v) for v in values)
    if len(set(out)) != len(out):
        raise ValueError(f"{name} lists each device once, got {list(out)}")
    return out


def _pose(values: Any, name: str) -> Tuple[float, ...]:
    out = tuple(_real(v, name) for v in values)
    if len(out) not in (2, 3):
        raise ValueError(f"{name} is a pose (x, y[, z]), got {values!r}")
    return out


# --------------------------------------------------------------------------- #
# The specification
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class RewardSpec:
    """Which reward, with which weights (the spec, other choices 7).

    ``kind`` is :data:`REWARD_KINDS`; the weights c_t, c_e and c_cov belong to
    the derived reward, so ``hand`` and ``bytes``, whose definitions fix their
    own, take them at 0. ``n_ref`` is the example count one update of the
    reference size carries (:data:`EQUAL_SHARD_EXAMPLES` under the training
    cells' device model). ``expected_availability`` credits each targeted
    member at its availability probability (training, critic C1); off, the
    realized draw (evaluation).
    """

    kind: str = REWARD_DERIVED
    c_t: float = 0.1
    c_e: float = 0.0
    c_cov: float = 1.0
    n_ref: float = float(EQUAL_SHARD_EXAMPLES)
    expected_availability: bool = False

    def __post_init__(self) -> None:
        if self.kind not in REWARD_KINDS:
            raise ValueError(f"kind must be one of {REWARD_KINDS}, got {self.kind!r}")
        for name in ("c_t", "c_e", "c_cov"):
            object.__setattr__(self, name, _nonneg(getattr(self, name), name))
        n_ref = _real(self.n_ref, "n_ref")
        if n_ref <= 0.0:
            raise ValueError(f"n_ref must be > 0, got {self.n_ref!r}")
        object.__setattr__(self, "n_ref", n_ref)
        if not isinstance(self.expected_availability, bool):
            raise TypeError(f"expected_availability must be a bool, got "
                            f"{self.expected_availability!r}")
        if self.kind != REWARD_DERIVED and (self.c_t or self.c_e or self.c_cov):
            raise ValueError(
                f"the {self.kind!r} reward fixes its own weights: c_t, c_e and c_cov "
                f"belong to the derived reward, so give them as 0")

    def to_json(self) -> Dict[str, Any]:
        """JSON-ready, the manifest's reward spec (the spec, other choices 6)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "RewardSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown reward spec key(s) {unknown}")
        return cls(**dict(data))


#: The defaults (decision 4 (a)): the derived reward at c_t 0.1, c_cov 1.
DERIVED = RewardSpec()
#: F·hand, ContactSim's reward ported (decision 4: "today's reward").
HAND = RewardSpec(kind=REWARD_HAND, c_t=0.0, c_cov=0.0)
#: E3's bytes reward (decision 7).
BYTES = RewardSpec(kind=REWARD_BYTES, c_t=0.0, c_cov=0.0)


def grid_specs(*, expected_availability: bool = False) -> Tuple[RewardSpec, ...]:
    """Study 5.7's nine derived rewards, c_t-major (decision 4)."""
    return tuple(RewardSpec(c_t=ct, c_cov=cc, expected_availability=expected_availability)
                 for ct in GRID_C_T for cc in GRID_C_COV)


# --------------------------------------------------------------------------- #
# What a sortie flew, as the reward reads it
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class StopRecord:
    """One Pass-1 stop flown: a decision point and what followed it.

    ``t_s`` is the arrival; ``targets`` the members the contact solicited on
    the band flown (``band``), ``collected`` those whose sessions were CLEAN,
    with ``w`` their raw L3 weights in that order (0 for an update the merge
    left out), ``uplink_dropped`` the targets the availability draw dropped,
    and ``late`` the collected members past their own deadline (diagnostic).
    ``end_s`` is the contact's end, ``flight_end_s`` the end of the flight
    after it (the next arrival, or the landing) and ``t_next_s`` the end of the
    decision (the next arrival, or the end of the Pass-1 upload, or the
    landing on the empty round), with ``terminal`` True exactly at the
    sortie's last stop. ``position`` and ``next_position`` are the stop's pose
    and the next stop's, or the dock's.
    """

    t_s: float
    devices: Tuple[str, ...]
    targets: Tuple[str, ...]
    collected: Tuple[str, ...]
    w: Tuple[float, ...]
    uplink_dropped: Tuple[str, ...]
    t_next_s: float
    terminal: bool
    end_s: float
    flight_end_s: float
    dwell_s: float
    listen_s: float
    position: Tuple[float, ...]
    next_position: Tuple[float, ...]
    band: str = ""
    late: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        devices = _ids(self.devices, "devices")
        targets = _ids(self.targets, "targets")
        collected = _ids(self.collected, "collected")
        dropped = _ids(self.uplink_dropped, "uplink_dropped")
        late = _ids(self.late, "late")
        if not set(targets) <= set(devices):
            raise ValueError(f"targets outside the stop: {sorted(set(targets) - set(devices))}")
        if not set(collected) <= set(targets):
            raise ValueError(f"collected members that were not targets: "
                             f"{sorted(set(collected) - set(targets))}")
        if not set(dropped) <= set(targets) or set(dropped) & set(collected):
            raise ValueError("uplink_dropped are targets the draw dropped, none collected")
        if not set(late) <= set(collected):
            raise ValueError("late names collected members only")
        w = tuple(_nonneg(x, "w") for x in self.w)
        if len(w) != len(collected):
            raise ValueError(f"w holds one weight per collected member: {len(collected)} "
                             f"collected, {len(w)} weights")
        t = _real(self.t_s, "t_s")
        end = _real(self.end_s, "end_s")
        flight_end = _real(self.flight_end_s, "flight_end_s")
        t_next = _real(self.t_next_s, "t_next_s")
        if not t <= end + _EPS_SPAN or not end <= flight_end + _EPS_SPAN \
                or not flight_end <= t_next + _EPS_SPAN:
            raise ValueError(
                f"a stop's times run arrival <= contact end <= flight end <= decision end: "
                f"{t}, {end}, {flight_end}, {t_next}")
        if not isinstance(self.terminal, bool):
            raise TypeError(f"terminal must be a bool, got {self.terminal!r}")
        values = dict(devices=devices, targets=targets, collected=collected, w=w,
                      uplink_dropped=dropped, late=late, t_s=t, end_s=end,
                      flight_end_s=flight_end, t_next_s=t_next,
                      dwell_s=_nonneg(self.dwell_s, "dwell_s"),
                      listen_s=_nonneg(self.listen_s, "listen_s"),
                      position=_pose(self.position, "position"),
                      next_position=_pose(self.next_position, "next_position"),
                      band=str(self.band))
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def dt_s(self) -> float:
        """The decision's span: arrival to the next arrival, or to the sortie's end."""
        return self.t_next_s - self.t_s

    @property
    def flown_m(self) -> float:
        """Metres flown after the stop: to the next stop, or to the dock."""
        a, b = self.position, self.next_position
        n = max(len(a), len(b))
        a = tuple(a) + (0.0,) * (n - len(a))
        b = tuple(b) + (0.0,) * (n - len(b))
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def to_json(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        for key in ("devices", "targets", "collected", "w", "uplink_dropped", "late",
                    "position", "next_position"):
            out[key] = list(out[key])
        return out


@dataclasses.dataclass(frozen=True)
class SortieRecord:
    """One mission's Pass 1, as the reward reads it.

    ``stops`` are its Pass-1 stops flown, in order (its decisions); N is
    ``n_demand``, the plan's demand with any beacon insert (the pair view's
    N); ``demand`` and ``coverage_weights`` the plan's demand and coverage
    weights, ``committed`` the devices the plan served (empty, as both are,
    outside plan mode). ``t_ref_s`` is T, T_nom (None for a legacy mule that
    has none, which only the derived reward reads). ``availability`` is the
    ground-truth rel_j the mule's draw uses ({} without the channel
    reliability source), read only by the expected-availability reward. The
    powers are the mule's energy model's.
    """

    mission_round: int
    stops: Tuple[StopRecord, ...]
    n_demand: int
    demand: Tuple[str, ...]
    committed: Tuple[str, ...]
    coverage_weights: Mapping[str, float]
    t_ref_s: Optional[float]
    availability: Mapping[str, float]
    p_hover_w: float
    p_move_w: float
    empty: bool = False

    def __post_init__(self) -> None:
        stops = tuple(self.stops)
        for stop in stops:
            if not isinstance(stop, StopRecord):
                raise TypeError(f"stops hold StopRecords, got {stop!r}")
        if stops:
            if [s.terminal for s in stops] != [False] * (len(stops) - 1) + [True]:
                raise ValueError("terminal is True exactly at the sortie's last stop")
            for a, b in zip(stops, stops[1:]):
                if a.t_next_s != b.t_s:
                    raise ValueError("a decision runs to the next stop's arrival")
        n = self.n_demand
        if isinstance(n, bool) or not isinstance(n, int) or n < 0 or (stops and n < 1):
            raise ValueError(f"n_demand is the mission's N (>= 1 once a stop is flown), "
                             f"got {n!r}")
        demand = _ids(self.demand, "demand")
        committed = _ids(self.committed, "committed")
        if not set(committed) <= set(demand):
            raise ValueError("the committed devices are part of the demand")
        weights = {str(k): _nonneg(v, f"coverage_weights[{k!r}]")
                   for k, v in dict(self.coverage_weights).items()}
        if set(weights) != set(demand):
            raise ValueError("coverage_weights hold exactly the demanded devices")
        availability = {}
        for k, v in dict(self.availability).items():
            rel = _nonneg(v, f"availability[{k!r}]")
            if rel > 1.0:
                raise ValueError(f"availability[{k!r}] is a probability, got {v!r}")
            availability[str(k)] = rel
        t_ref = None if self.t_ref_s is None else _real(self.t_ref_s, "t_ref_s")
        if t_ref is not None and t_ref <= 0.0:
            raise ValueError(f"t_ref_s must be > 0, got {self.t_ref_s!r}")
        values = dict(stops=stops, demand=demand, committed=committed,
                      coverage_weights=MappingProxyType(weights),
                      availability=MappingProxyType(availability), t_ref_s=t_ref,
                      p_hover_w=_nonneg(self.p_hover_w, "p_hover_w"),
                      p_move_w=_nonneg(self.p_move_w, "p_move_w"))
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if not isinstance(self.empty, bool):
            raise TypeError(f"empty must be a bool, got {self.empty!r}")

    @property
    def collected(self) -> Tuple[str, ...]:
        """Every update collected CLEAN in the sortie, in stop order."""
        return tuple(d for s in self.stops for d in s.collected)

    @property
    def undecided_shortfall(self) -> float:
        """U of a sortie that flew no Pass-1 stop (0 otherwise): reported, never
        charged, since no decision was made to charge it to."""
        if self.stops:
            return 0.0
        return _shortfall(self, {})

    def to_json(self) -> Dict[str, Any]:
        return {
            "mission_round": int(self.mission_round),
            "stops": [s.to_json() for s in self.stops],
            "n_demand": int(self.n_demand),
            "demand": list(self.demand),
            "committed": list(self.committed),
            "coverage_weights": dict(self.coverage_weights),
            "t_ref_s": self.t_ref_s,
            "p_hover_w": self.p_hover_w,
            "p_move_w": self.p_move_w,
            "empty": self.empty,
        }


# --------------------------------------------------------------------------- #
# The reward
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class RewardTerms:
    """One reward and its terms, each a positive amount: total = gain - the rest."""

    gain: float = 0.0
    time: float = 0.0
    energy: float = 0.0
    distance: float = 0.0
    coverage: float = 0.0

    @property
    def total(self) -> float:
        return self.gain - self.time - self.energy - self.distance - self.coverage

    def __add__(self, other: "RewardTerms") -> "RewardTerms":
        if not isinstance(other, RewardTerms):
            return NotImplemented
        return RewardTerms(*(a + b for a, b in zip(dataclasses.astuple(self),
                                                     dataclasses.astuple(other))))

    def to_json(self) -> Dict[str, float]:
        out = dataclasses.asdict(self)
        out["total"] = self.total
        return out


def _probability(sortie: SortieRecord, device: str) -> float:
    """rel_j, the availability the mule's keyed draw tests against; 1.0 for a
    member without an entry, which the draw treats as always available
    (``FerryRuntime.uplink_drops``)."""
    return sortie.availability.get(device, 1.0)


def _check_expectation(spec: RewardSpec, sortie: SortieRecord) -> None:
    if spec.expected_availability and not sortie.availability and any(
            s.targets for s in sortie.stops):
        raise ValueError(
            "the expected-availability reward needs the ground-truth availability "
            "(contact_reliability_source='channel'): without it no draw is made")


def _credits(spec: RewardSpec, sortie: SortieRecord, stop: StopRecord) -> Tuple[float, float]:
    """(count, weight) credited to ``stop``: realized, or at the availability."""
    if not spec.expected_availability:
        return float(len(stop.collected)), float(sum(stop.w))
    count = weight = 0.0
    for did, w in zip(stop.collected, stop.w):
        p = _probability(sortie, did)
        count += p
        weight += p * w
    for did in stop.uplink_dropped:
        p = _probability(sortie, did)
        count += p
        weight += p * spec.n_ref
    return count, weight


def _shortfall(sortie: SortieRecord, collected_prob: Mapping[str, float]) -> float:
    """U: the committed weight left uncollected over the demand's weight."""
    total = sum(sortie.coverage_weights[d] for d in sortie.demand)
    if total <= 0.0:
        return 0.0
    missed = sum(sortie.coverage_weights[d] * (1.0 - collected_prob.get(d, 0.0))
                 for d in sortie.committed)
    return missed / total


def coverage_shortfall(spec: RewardSpec, sortie: SortieRecord) -> float:
    """U of ``sortie``: realized, or with each member that had a session
    collected at its availability (expected availability)."""
    prob: Dict[str, float] = {}
    for stop in sortie.stops:
        for did in stop.collected:
            prob[did] = _probability(sortie, did) if spec.expected_availability else 1.0
        if spec.expected_availability:
            for did in stop.uplink_dropped:
                prob[did] = _probability(sortie, did)
    return _shortfall(sortie, prob)


def stop_energy_j(sortie: SortieRecord, stop: StopRecord) -> float:
    """dE_k: hover power over the contact's dwell and listen, flight power
    until the next arrival or the landing (the mule's ``EnergyModel``)."""
    return (sortie.p_hover_w * (stop.dwell_s + stop.listen_s)
            + sortie.p_move_w * max(0.0, stop.flight_end_s - stop.end_s))


def sortie_rewards(spec: RewardSpec, sortie: SortieRecord) -> Tuple[RewardTerms, ...]:
    """One :class:`RewardTerms` per decision of ``sortie``, in order."""
    if not isinstance(spec, RewardSpec):
        raise TypeError(f"spec must be a RewardSpec, got {spec!r}")
    _check_expectation(spec, sortie)
    if spec.kind == REWARD_DERIVED and sortie.stops and sortie.t_ref_s is None:
        raise ValueError("the derived reward prices time in T_nom: this sortie has no T")
    out = []
    shortfall = None
    for stop in sortie.stops:
        count, weight = _credits(spec, sortie, stop)
        if spec.kind == REWARD_BYTES:
            out.append(RewardTerms(gain=count / sortie.n_demand))
            continue
        if spec.kind == REWARD_HAND:
            out.append(RewardTerms(
                gain=HAND_PER_UPDATE * count / HAND_SCALE,
                time=stop.dt_s / HAND_SCALE,
                distance=HAND_PER_METRE * stop.flown_m / HAND_SCALE,
            ))
            continue
        energy = 0.0
        if spec.c_e:
            if sortie.p_hover_w <= 0.0:
                raise ValueError("an energy weight needs P_hover > 0 to normalise by")
            energy = spec.c_e * stop_energy_j(sortie, stop) / (sortie.p_hover_w * sortie.t_ref_s)
        coverage = 0.0
        if stop.terminal and spec.c_cov:
            if shortfall is None:
                shortfall = coverage_shortfall(spec, sortie)
            coverage = spec.c_cov * shortfall
        out.append(RewardTerms(
            gain=weight / (spec.n_ref * sortie.n_demand),
            time=spec.c_t * stop.dt_s / sortie.t_ref_s,
            energy=energy,
            coverage=coverage,
        ))
    return tuple(out)


def sortie_return(spec: RewardSpec, sortie: SortieRecord) -> RewardTerms:
    """The sortie's undiscounted return, term by term."""
    total = RewardTerms()
    for terms in sortie_rewards(spec, sortie):
        total = total + terms
    return total


def episode_return(spec: RewardSpec, sorties: Iterable[SortieRecord]) -> RewardTerms:
    """An episode's undiscounted return (the spec, other choices 12), term by term."""
    total = RewardTerms()
    for sortie in sorties:
        total = total + sortie_return(spec, sortie)
    return total


# --------------------------------------------------------------------------- #
# The merge's weights
# --------------------------------------------------------------------------- #

def raw_merge_weights(spec, aggregate, report) -> Dict[str, float]:
    """Each merged update's raw L3 weight w_i, by device id: what the merge gave it.

    ``agg:plain`` averages by example count (``partial_fedavg``), so w_i is the
    n_i its CLEAN report line carries; the age-aware rules store w_i / M_m and
    M_m (``merge_on_mule``: ``device_weights`` and ``weight_mass``), so w_i is
    their product, ``update_weights``' raw weight. An update the merge left out
    is absent (it weighs 0), and so is every update when there is no
    aggregate (the empty round). The same reading as the mule's own closing
    of the pair records (``mule_main._raw_merge_weights``), which a test
    checks, and as ``update_weights`` on hand-built submissions.
    """
    if aggregate is None:
        return {}
    if spec.is_plain:
        examples = {str(line.device_id): line.num_examples
                    for line in getattr(report, "lines", ()) if line.outcome.is_on_time()}
        return {str(d): float(examples.get(str(d), 0)) for d in aggregate.contributing_devices}
    return {str(d): float(w) * float(aggregate.weight_mass)
            for d, w in zip(aggregate.contributing_devices, aggregate.device_weights)}


__all__ = [
    "BYTES",
    "DERIVED",
    "EQUAL_SHARD_EXAMPLES",
    "GRID_C_COV",
    "GRID_C_T",
    "HAND",
    "REWARD_BYTES",
    "REWARD_DERIVED",
    "REWARD_HAND",
    "REWARD_KINDS",
    "RewardSpec",
    "RewardTerms",
    "STUB_MEAN_EXAMPLES",
    "SortieRecord",
    "StopRecord",
    "coverage_shortfall",
    "episode_return",
    "grid_specs",
    "raw_merge_weights",
    "sortie_return",
    "sortie_rewards",
    "stop_energy_j",
]
