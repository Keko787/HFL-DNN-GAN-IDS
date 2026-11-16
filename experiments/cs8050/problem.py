"""CMP_SC 8050 project, Phase 1: the simplified trip-planning problem and its plan checker.

The project asks how hard FeRRy's trip-planning problem is and how close the
planner (``hermes/scheduler/plan/plan_search.py``) gets to the best plan. This
module fixes the simplified problem that every proof and solver in the project
uses (``DeveloperDocs/cs8050/problem_statement.md``) and checks a plan against
it. It imports nothing from ``hermes``; the Phase 3 harness translates between
the two.

**Instance** (:class:`Instance`): the devices, each with a position, an
importance w_j > 0, a deadline D_j and a data size s_j; the radio settings,
each with a reach r_b and a rate that may step down with distance; the dock,
the drone's speed v, the time limit T, and the upload time u charged when the
drone lands with data. The finite list of sites where stops may be (decision
1.5), and optionally a quantum that travel times are rounded up to (decision
1.6: the proofs round, the experiments keep exact times).

**Plan** (:class:`Plan`): one setting and an ordered list of stops, each a
site and the devices served there. The empty plan is a plan.

**Rules** :func:`check_plan` enforces, in the order it reports them:

1. the setting exists;
2. stop by stop, every stop serves at least one device, every device is known,
   and no device is served twice in the plan;
3. every stop is at a listed site (only when the instance lists sites);
4. every device served at a stop is within the setting's reach of it;
5. flying the route from the dock at time 0, stop by stop: the stop's dwell is
   the sum of its devices' send times, its finish (arrival plus dwell) must be
   no later than the deadline of every device it serves, and then the drone
   must still be able to fly home and upload within T.

Rule 5 follows S3b's predicate (``FeasibilityModel.admit`` in
``hermes/scheduler/stages/s3b_feasibility.py``, with the default
``deadline_bounds = "collection"`` and the budget clause). The deadline binds
a stop's finish, not its arrival, and every stop is held to the time limit as
if the drone flew home right after it. Travel times here obey the triangle
inequality (Euclidean, or Euclidean rounded up), so the last stop's return is
the latest and the time limit is the whole trip's.

**Value** of a valid plan: the total importance of the devices it serves.

Comparisons allow an absolute slack of :data:`TOL` (seconds and metres alike)
so that float rounding cannot flip a verdict that exact arithmetic would give.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Point", "TOL",
    "RULE_SETTING", "RULE_EMPTY_STOP", "RULE_UNKNOWN_DEVICE", "RULE_SERVED_TWICE",
    "RULE_SITE", "RULE_REACH", "RULE_DEADLINE", "RULE_TIME_LIMIT", "RULES",
    "Device", "Setting", "Instance", "Stop", "Plan", "StopTiming", "Verdict",
    "check_plan",
]

Point = Tuple[float, float]

#: Absolute slack on every comparison, in seconds and in metres alike: a value
#: x meets a bound B when x <= B + TOL. The proofs use exact arithmetic; the
#: slack only keeps float rounding from flipping a verdict.
TOL = 1e-9

#: The rules :func:`check_plan` reports, in the order it checks them.
RULE_SETTING = "unknown_setting"
RULE_EMPTY_STOP = "empty_stop"
RULE_UNKNOWN_DEVICE = "unknown_device"
RULE_SERVED_TWICE = "served_twice"
RULE_SITE = "not_a_site"
RULE_REACH = "out_of_reach"
RULE_DEADLINE = "late"
RULE_TIME_LIMIT = "over_time_limit"
RULES: Tuple[str, ...] = (
    RULE_SETTING, RULE_EMPTY_STOP, RULE_UNKNOWN_DEVICE, RULE_SERVED_TWICE,
    RULE_SITE, RULE_REACH, RULE_DEADLINE, RULE_TIME_LIMIT,
)


# --------------------------------------------------------------------------- #
# Checked values
# --------------------------------------------------------------------------- #

def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if math.isnan(out):
        raise ValueError(f"{name} must not be NaN")
    return out


def _finite(value: Any, name: str) -> float:
    out = _real(value, name)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _positive(value: Any, name: str) -> float:
    out = _finite(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return out


def _nonneg(value: Any, name: str, *, allow_inf: bool = False) -> float:
    out = _real(value, name) if allow_inf else _finite(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return out


def _point(value: Any, name: str) -> Point:
    try:
        coords = tuple(value)
    except TypeError:
        raise TypeError(f"{name} must be an (x, y) pair, got {value!r}") from None
    if len(coords) != 2:
        raise ValueError(f"{name} must be an (x, y) pair, got {value!r}")
    return (_finite(coords[0], f"{name}[0]"), _finite(coords[1], f"{name}[1]"))


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string, got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# The instance
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Device:
    """One device: where it is, how important it is, its deadline and its data.

    ``deadline_s`` is D_j in seconds after takeoff (``math.inf``: none).
    ``size_bytes`` is s_j, what the device sends when served.
    """

    id: str
    position: Point
    weight: float
    deadline_s: float = math.inf
    size_bytes: float = 0.0

    def __post_init__(self) -> None:
        _name(self.id, "device id")
        object.__setattr__(self, "position", _point(self.position, f"{self.id}.position"))
        object.__setattr__(self, "weight", _positive(self.weight, f"{self.id}.weight"))
        object.__setattr__(self, "deadline_s",
                           _nonneg(self.deadline_s, f"{self.id}.deadline_s", allow_inf=True))
        object.__setattr__(self, "size_bytes", _nonneg(self.size_bytes, f"{self.id}.size_bytes"))


@dataclass(frozen=True)
class Setting:
    """One radio setting: its reach and its rate by distance.

    ``rate_steps`` is ``((d_1, rho_1), ..., (d_k, rho_k))``: a device at most
    d_1 metres from the stop sends at rho_1 bytes/s, one further out but at most
    d_2 at rho_2, and so on. The d are strictly increasing and end at
    ``reach_m``; the rho are positive and never increase, as the CQI steps of
    ``hermes/l1/contact_link.py`` fall with distance. One step is a constant
    rate, the proposal's model (:meth:`constant`).
    """

    name: str
    reach_m: float
    rate_steps: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        _name(self.name, "setting name")
        reach = _positive(self.reach_m, f"{self.name}.reach_m")
        object.__setattr__(self, "reach_m", reach)
        steps = tuple(tuple(s) for s in self.rate_steps)
        if not steps:
            raise ValueError(f"{self.name}: rate_steps needs at least one step")
        out = []
        prev_d, prev_rate = 0.0, math.inf
        for i, step in enumerate(steps):
            if len(step) != 2:
                raise ValueError(f"{self.name}: rate step {i} must be (up_to_m, rate), got {step!r}")
            d = _positive(step[0], f"{self.name}.rate_steps[{i}] distance")
            rate = _positive(step[1], f"{self.name}.rate_steps[{i}] rate")
            if d <= prev_d:
                raise ValueError(f"{self.name}: rate step distances must increase, got {steps!r}")
            if rate > prev_rate:
                raise ValueError(f"{self.name}: rates must not increase with distance, got {steps!r}")
            out.append((d, rate))
            prev_d, prev_rate = d, rate
        if out[-1][0] != reach:
            raise ValueError(
                f"{self.name}: the last rate step must end at the reach ({reach!r} m), "
                f"got {out[-1][0]!r}"
            )
        object.__setattr__(self, "rate_steps", tuple(out))

    @classmethod
    def constant(cls, name: str, reach_m: float, rate_bytes_s: float) -> "Setting":
        """A setting with one rate out to its reach."""
        return cls(name, reach_m, ((reach_m, rate_bytes_s),))

    def reaches(self, distance_m: float) -> bool:
        """True when a device ``distance_m`` metres from the stop is in reach (inclusive)."""
        return _nonneg(distance_m, "distance_m") <= self.reach_m + TOL

    def rate_at(self, distance_m: float) -> float:
        """The rate in bytes/s for a device ``distance_m`` metres from the stop."""
        d = _nonneg(distance_m, "distance_m")
        for up_to, rate in self.rate_steps:
            if d <= up_to + TOL:
                return rate
        raise ValueError(f"{self.name}: {d!r} m is beyond the reach of {self.reach_m!r} m")

    def send_s(self, size_bytes: float, distance_m: float) -> float:
        """Seconds to send ``size_bytes`` from ``distance_m`` metres away."""
        return _nonneg(size_bytes, "size_bytes") / self.rate_at(distance_m)


@dataclass(frozen=True)
class Instance:
    """One planning problem.

    * ``devices``, ``settings``: ids and names are unique; at least one setting.
    * ``dock``: where the trip starts and ends.
    * ``speed_m_s``: v, the drone's speed; travel is straight-line.
    * ``time_limit_s``: T; ``math.inf`` for none.
    * ``upload_s``: u, charged once when the drone lands with data. The empty
      plan never takes off, so it pays no upload, as in the planner.
    * ``sites``: the allowed stop positions. Decision 1.5 makes them a finite
      list in the input; the experiments use two lists (the planner's stops,
      and the full list). None lifts the rule, to check a plan without a list.
    * ``travel_quantum_s``: decision 1.6. The proofs round each leg's time up
      to a multiple of a quantum; the experiments use None, the exact
      Euclidean time, as the planner does. Rounding up keeps the triangle
      inequality: ceil(a + b) <= ceil(a) + ceil(b).
    """

    devices: Tuple[Device, ...]
    settings: Tuple[Setting, ...]
    dock: Point = (0.0, 0.0)
    speed_m_s: float = 5.0
    time_limit_s: float = math.inf
    upload_s: float = 0.0
    sites: Optional[Tuple[Point, ...]] = None
    travel_quantum_s: Optional[float] = None
    _by_id: Mapping[str, Device] = field(init=False, repr=False, compare=False)
    _by_name: Mapping[str, Setting] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        devices = tuple(self.devices)
        for d in devices:
            if not isinstance(d, Device):
                raise TypeError(f"devices hold Device values, got {d!r}")
        by_id = {d.id: d for d in devices}
        if len(by_id) != len(devices):
            raise ValueError("device ids must be unique")
        settings = tuple(self.settings)
        if not settings:
            raise ValueError("an instance needs at least one setting")
        for s in settings:
            if not isinstance(s, Setting):
                raise TypeError(f"settings hold Setting values, got {s!r}")
        by_name = {s.name: s for s in settings}
        if len(by_name) != len(settings):
            raise ValueError("setting names must be unique")
        object.__setattr__(self, "devices", devices)
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "dock", _point(self.dock, "dock"))
        object.__setattr__(self, "speed_m_s", _positive(self.speed_m_s, "speed_m_s"))
        object.__setattr__(self, "time_limit_s",
                           _nonneg(self.time_limit_s, "time_limit_s", allow_inf=True))
        object.__setattr__(self, "upload_s", _nonneg(self.upload_s, "upload_s"))
        if self.sites is not None:
            object.__setattr__(self, "sites", tuple(
                _point(p, f"sites[{i}]") for i, p in enumerate(self.sites)))
        if self.travel_quantum_s is not None:
            object.__setattr__(self, "travel_quantum_s",
                               _positive(self.travel_quantum_s, "travel_quantum_s"))
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))

    def has_device(self, did: str) -> bool:
        return did in self._by_id

    def device(self, did: str) -> Device:
        """The device with id ``did`` (KeyError when there is none)."""
        return self._by_id[did]

    def setting(self, name: str) -> Setting:
        """The setting named ``name`` (KeyError when there is none)."""
        return self._by_name[name]

    def travel_s(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Flying time from ``a`` to ``b``: distance over speed, rounded up when quantized.

        A leg within :data:`TOL` quanta of a multiple rounds to that multiple,
        so that float error never adds a whole quantum.
        """
        t = math.dist(a, b) / self.speed_m_s
        q = self.travel_quantum_s
        if q is None:
            return t
        return max(0, math.ceil(t / q - TOL)) * q


# --------------------------------------------------------------------------- #
# The plan and the verdict
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Stop:
    """One stop: a site and the devices served there, in any order."""

    site: Point
    devices: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "site", _point(self.site, "stop site"))
        if isinstance(self.devices, str):
            raise TypeError(f"a stop's devices are a collection of ids, not one string: "
                            f"{self.devices!r}")
        object.__setattr__(self, "devices", tuple(self.devices))


@dataclass(frozen=True)
class Plan:
    """One setting and the stops flown on it, in order. No stops is the empty plan."""

    setting: str
    stops: Tuple[Stop, ...] = ()

    def __post_init__(self) -> None:
        stops = tuple(self.stops)
        for s in stops:
            if not isinstance(s, Stop):
                raise TypeError(f"stops hold Stop values, got {s!r}")
        object.__setattr__(self, "stops", stops)

    @property
    def served(self) -> Tuple[str, ...]:
        """Every device the plan lists, in route order."""
        return tuple(d for s in self.stops for d in s.devices)


@dataclass(frozen=True)
class StopTiming:
    """When the drone reaches a stop, how long it stays, and when it could be home.

    ``home_s`` is the time back at the dock with the upload done if the drone
    flew home right after this stop: S3b's ``Verdict.home``.
    """

    arrival_s: float
    dwell_s: float
    finish_s: float
    home_s: float


@dataclass(frozen=True)
class Verdict:
    """What :func:`check_plan` found.

    A valid plan has ``ok`` True, ``value`` its served importance, ``end_s``
    the time the trip ends (0 for the empty plan) and a timing per stop. A
    refused plan has ``ok`` False, ``value`` 0, ``end_s`` None, ``rule`` the
    first rule broken (one of :data:`RULES`), ``stop`` the index of the stop
    that broke it (None for an unknown setting), a readable ``detail``, and the
    timings of the stops walked up to and including that one (none when a rule
    before rule 5 broke).
    """

    ok: bool
    value: float
    end_s: Optional[float]
    rule: Optional[str] = None
    stop: Optional[int] = None
    detail: str = ""
    timings: Tuple[StopTiming, ...] = ()


def _refuse(rule: str, stop: Optional[int], detail: str,
            timings: Tuple[StopTiming, ...] = ()) -> Verdict:
    return Verdict(ok=False, value=0.0, end_s=None, rule=rule, stop=stop, detail=detail,
                   timings=timings)


def check_plan(instance: Instance, plan: Plan) -> Verdict:
    """Check ``plan`` against ``instance``: its value, or the first rule it breaks.

    Rules 1 to 4 (see the module notes) are checked over the whole plan, rule
    by rule, each stop by stop. Rule 5 walks the route: at each stop the
    deadline is checked before the time limit, the order of S3b's clauses.
    """
    if not isinstance(instance, Instance):
        raise TypeError(f"instance must be an Instance, got {instance!r}")
    if not isinstance(plan, Plan):
        raise TypeError(f"plan must be a Plan, got {plan!r}")

    # Rule 1: the setting.
    try:
        setting = instance.setting(plan.setting)
    except KeyError:
        names = [s.name for s in instance.settings]
        return _refuse(RULE_SETTING, None, f"no setting named {plan.setting!r}; have {names}")

    # Rule 2: non-empty stops, known devices, each served once.
    first_stop = {}
    for k, stop in enumerate(plan.stops):
        if not stop.devices:
            return _refuse(RULE_EMPTY_STOP, k, f"stop {k} serves no device")
        for did in stop.devices:
            if not instance.has_device(did):
                return _refuse(RULE_UNKNOWN_DEVICE, k, f"stop {k} serves {did!r}, not a device "
                                                       f"of the instance")
            if did in first_stop:
                return _refuse(RULE_SERVED_TWICE, k, f"{did!r} is served at stop "
                                                     f"{first_stop[did]} and again at stop {k}")
            first_stop[did] = k

    # Rule 3: listed sites.
    if instance.sites is not None:
        allowed = set(instance.sites)
        for k, stop in enumerate(plan.stops):
            if stop.site not in allowed:
                return _refuse(RULE_SITE, k, f"stop {k} is at {stop.site}, not a listed site")

    # Rule 4: reach.
    for k, stop in enumerate(plan.stops):
        for did in stop.devices:
            d = math.dist(stop.site, instance.device(did).position)
            if not setting.reaches(d):
                return _refuse(RULE_REACH, k, f"{did!r} is {d:.6g} m from stop {k}; "
                                              f"{setting.name} reaches {setting.reach_m:.6g} m")

    # Rule 5: deadlines and the time limit, flying the route.
    timings = []
    clock, here = 0.0, instance.dock
    for k, stop in enumerate(plan.stops):
        arrival = clock + instance.travel_s(here, stop.site)
        members = [instance.device(did) for did in stop.devices]
        dwell = math.fsum(
            setting.send_s(dev.size_bytes, math.dist(stop.site, dev.position)) for dev in members
        )
        finish = arrival + dwell
        home = finish + instance.travel_s(stop.site, instance.dock) + instance.upload_s
        timings.append(StopTiming(arrival_s=arrival, dwell_s=dwell, finish_s=finish, home_s=home))
        late = [dev for dev in members if finish > dev.deadline_s + TOL]
        if late:
            first = min(late, key=lambda dev: (dev.deadline_s, dev.id))
            return _refuse(
                RULE_DEADLINE, k,
                f"stop {k} finishes at {finish:.6g} s, after the deadline of {first.id!r} "
                f"({first.deadline_s:.6g} s); the drone arrived at {arrival:.6g} s",
                tuple(timings),
            )
        if home > instance.time_limit_s + TOL:
            return _refuse(
                RULE_TIME_LIMIT, k,
                f"from stop {k} the drone is home with the upload done at {home:.6g} s, "
                f"after the time limit of {instance.time_limit_s:.6g} s",
                tuple(timings),
            )
        clock, here = finish, stop.site

    value = math.fsum(instance.device(did).weight for did in plan.served)
    end = timings[-1].home_s if timings else 0.0
    return Verdict(ok=True, value=value, end_s=end, timings=tuple(timings))
