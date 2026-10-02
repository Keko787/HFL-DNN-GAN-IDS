"""FeRRy Phase 5: the per-departure next-stop protocol and E3's observation (unit U0).

**Why a module of its own.** Arm E3, after Chen et al. (GLOBECOM Workshops
2023; build plan L1026, L1304; the user's decision 7 (a)), is a
whole-scheduler policy in legacy mode (``contact_policy = "chen_dqn"``,
``policies/chen_dqn.py``, unit U6). Unlike D1-D5, whose route is fixed
before takeoff (``admit_and_order``, Freeze Amendment 8), it chooses each next
stop in flight: at takeoff and at every Pass-1 departure, among the stops
still to fly that S3b's single-contact predicate admits from where the mule
is. Three units meet at that call: the mule's runtime builds what E3 observes
(U4, ``FerryRuntime.e3_observation``), the policy reads it (U6), and the
supervisor makes the call and records it (U5). So the observation and the
call's contract are defined here, once, before any of them (critic B11).

**The protocol.** A policy that declares ``chooses_next_stop = True``
(:class:`NextStopPolicy`) is called as ``next_stop(remainder, state, *, view,
admissible, pass_kind, after_stop)``:

* ``remainder``: the stops not yet flown, ``ContactWaypoint`` objects in the
  order the policy's ``admit_and_order`` gave them, less those flown;
* ``state``: the departure's flight state (``s3b_feasibility.FlightState``);
* ``view``: :class:`E3View`, one :class:`E3Stop` per remainder stop, in its
  order;
* ``admissible(i)``: S3b's own predicate for the remainder's stop ``i``,
  ``FeasibilityModel.admit(state, remainder[i], rule=RULE_BUDGET,
  budget_end=..., pass_kind=COLLECT).ok``: transit, dwell, the return leg and
  the upload within the budget end, and the energy clause. That is Chen's
  safety controller (the Phase 5 spec, other choices 11), with no deadline
  and no plan, since E3 flies none of FeRRy's machinery (build plan L1026);
* ``pass_kind``: Pass 1, always (:func:`pass_1_only`); ``after_stop``: False
  at takeoff, True at a departure from a stop served.

It returns the index of the stop to fly next, an admissible one, or None
exactly when no stop is admissible: the pass ends, the mule flies home, and
the stops left are reported (``mission_completed.pass_1_e3_unvisited``), never
widened (decision 6's rule for a baseline's drops). :func:`checked_choice`
holds an answer to that. The supervisor reads the flag with
``getattr(policy, "chooses_next_stop", False)``, so every existing policy,
which declares none, flies its recorded path (Freeze Rule 1). It never calls
the policy in Pass 2, which delivers to every slice stop in the queue's order
(Freeze principle 12; critic B7 i), and the policy refuses such a call itself
(:func:`pass_1_only`).

**Layering.** Numpy-free, and nothing from the plan package, ``hermes.l1``,
``hermes.mule``, ``hermes.mission`` or ``experiments`` (the Phase 5 spec,
other choices 11; critic B7 iii). The runtime imports this module lazily, as
``FerryRuntime.arrival_view`` imports the plan types, so a mission of any other
arm never loads it. ``policies/__init__`` does not import it, and E3, which
flies none of the plan's machinery, never loads the plan package through it.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple, runtime_checkable

from hermes.types.scheduler import ContactWaypoint, MissionPass

__all__ = [
    "Admissible",
    "E3Stop",
    "E3View",
    "NextStopPolicy",
    "checked_choice",
    "pass_1_only",
]

#: ``admissible(i) -> bool``: may the mule serve the remainder's stop ``i``
#: next, from the departure state? The supervisor binds S3b's single-contact
#: predicate under ``RULE_BUDGET``, the landing included (module docstring).
Admissible = Callable[[int], bool]


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


def _nonneg(value: Any, name: str) -> float:
    out = _finite(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return out


def _positive(value: Any, name: str) -> float:
    out = _finite(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return out


def _share(value: Any, name: str) -> float:
    out = _nonneg(value, name)
    if out > 1.0:
        raise ValueError(f"{name} is a share in [0, 1], got {value!r}")
    return out


def _energy_ref(value: Any, name: str) -> Optional[float]:
    """An energy reference as a float > 0, or None for none; 0 is stored as None.

    ``FerryRuntime.l1_state``'s reference (``mule/ferry.py``) is the capacity
    if one is set, else P_hover times the budget, and that product is 0 when
    P_hover is, which ``EnergyModel`` accepts (``l1/mission_clock.py``, P_hover
    >= 0). ``l1_state`` reads a 0 reference as none (``if not e_ref``), so a 0
    is stored as None: a reader tests ``is None`` alone and never divides by 0.
    The plan types hold the same rule (``plan/types.py``), which this module
    may not import.
    """
    if value is None:
        return None
    out = _nonneg(value, name)
    return out if out > 0.0 else None


@dataclass(frozen=True)
class E3Stop:
    """Chen's observation of one candidate stop, aggregated over its members.

    After Chen et al.'s per-device observation (the Phase 5 design, D-J (a)),
    taken from the mule's pose at the departure on the band E3 flies:

    * ``members``: how many devices the stop serves (N normalises it);
    * ``remaining``: the share of its members whose update this mission has
      not collected yet, Chen's remaining data. Every candidate is unvisited
      and each device holds one update of one size, so it is 1 at every call:
      kept for fidelity and declared constant (critic B7 ii);
    * ``snr_db``: the median over the members of the SNR now from the pose:
      realized for a member within reach, and the mean ("radio map") SNR
      beyond it, since the realized SNR out of reach is the simulator's alone
      (critic B7 iv);
    * ``reachable``: the share of the members reachable now from the pose,
      within the band's R_planar and at or above the SNR floor. It is near 0
      from most poses, kept and declared like ``remaining`` (critic B7 ii);
    * ``dx_m``, ``dy_m`` and ``distance_m``: where the stop lies from the
      pose, and the leg's length on the flight model's metric;
    * ``return_energy_j``: the simulated energy of flying from the stop back
      to the dock.
    """

    members: int
    remaining: float
    snr_db: float
    reachable: float
    dx_m: float
    dy_m: float
    distance_m: float
    return_energy_j: float

    def __post_init__(self) -> None:
        members = _count(self.members, "members")
        if members < 1:
            raise ValueError("a stop serves at least one device: members must be >= 1")
        values = {
            "members": members,
            "remaining": _share(self.remaining, "remaining"),
            "snr_db": _finite(self.snr_db, "snr_db"),
            "reachable": _share(self.reachable, "reachable"),
            "dx_m": _finite(self.dx_m, "dx_m"),
            "dy_m": _finite(self.dy_m, "dy_m"),
            "distance_m": _nonneg(self.distance_m, "distance_m"),
            "return_energy_j": _nonneg(self.return_energy_j, "return_energy_j"),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class E3View:
    """What E3 sees at a departure: every candidate stop, and the sortie's state.

    ``stops`` holds one :class:`E3Stop` per stop of the remainder, in its
    order: index i is the remainder's stop i, as for ``admissible`` and the
    policy's answer. ``band`` is the contact band E3 flies, the cell's own:
    E3 is band-fixed (build plan L1026). ``demand`` is N, the devices of the
    mule's slice this mission. ``clock_s``, ``budget_end`` (absolute; None
    without a budget), ``budget_s`` (the budget's length; None without one),
    ``energy_j`` (the simulated energy spent this sortie) and
    ``energy_ref_j`` (the capacity if one is set, else P_hover times the
    budget, else None, as ``FerryRuntime.l1_state`` takes it; a 0 reference,
    P_hover = 0, is none there and is stored as None here) give Chen's global
    observation: the time and the energy left.
    """

    stops: Tuple[E3Stop, ...]
    band: str
    demand: int
    clock_s: float
    budget_end: Optional[float]
    budget_s: Optional[float]
    energy_j: float
    energy_ref_j: Optional[float]

    def __post_init__(self) -> None:
        stops = tuple(self.stops)
        if not stops:
            raise ValueError("E3 chooses among the stops still to fly: the view needs at least one")
        for stop in stops:
            if not isinstance(stop, E3Stop):
                raise TypeError(f"stops hold E3Stop entries, got {stop!r}")
        if not isinstance(self.band, str) or not self.band:
            raise TypeError(f"band must be a non-empty string, got {self.band!r}")
        demand = _count(self.demand, "demand")
        if demand < 1:
            raise ValueError("demand is N, the devices E3 may visit: it must be >= 1")
        values = {
            "stops": stops,
            "demand": demand,
            "clock_s": _finite(self.clock_s, "clock_s"),
            "budget_end": None if self.budget_end is None else _finite(
                self.budget_end, "budget_end"),
            "budget_s": None if self.budget_s is None else _positive(self.budget_s, "budget_s"),
            "energy_j": _nonneg(self.energy_j, "energy_j"),
            "energy_ref_j": _energy_ref(self.energy_ref_j, "energy_ref_j"),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def time_left_s(self) -> Optional[float]:
        """The budget end less the clock (negative once overrun); None without a budget."""
        return None if self.budget_end is None else self.budget_end - self.clock_s


@runtime_checkable
class NextStopPolicy(Protocol):
    """A whole-scheduler policy that chooses each next stop in flight (module docstring).

    ``chooses_next_stop`` is True; the supervisor calls :meth:`next_stop` at
    takeoff and at every Pass-1 departure while stops remain.
    """

    chooses_next_stop: bool

    def next_stop(self, remainder: Sequence[ContactWaypoint], state: Any, *, view: E3View,
                  admissible: Admissible, pass_kind: Any, after_stop: bool) -> Optional[int]:
        """The remainder index to fly next, an admissible one; None when none is."""
        ...


def checked_choice(choice: Any, remainder: Sequence[ContactWaypoint], *,
                   admissible: Optional[Admissible] = None) -> Optional[int]:
    """A policy's answer held to the protocol; returns it as an int or None.

    The answer is None or the index of a stop of ``remainder``. Given
    ``admissible``, an index must pass it, as Chen's safety controller
    requires, and None is right only when no stop does: E3 ends its pass
    because nothing fits, never by choice (the Phase 5 spec, other choices 11).
    """
    n = len(remainder)
    if choice is None:
        if admissible is not None:
            fits = [i for i in range(n) if admissible(i)]
            if fits:
                raise ValueError(
                    f"the pass ends (None) only when no stop is admissible, but stops {fits} are"
                )
        return None
    if isinstance(choice, bool) or not isinstance(choice, numbers.Integral):
        raise TypeError(f"a next stop is a remainder index or None, got {choice!r}")
    index = int(choice)
    if not 0 <= index < n:
        raise ValueError(f"next stop {index} is outside the remainder of {n} stop(s)")
    if admissible is not None and not admissible(index):
        raise ValueError(f"next stop {index} is not admissible: E3 flies admissible stops only")
    return index


def pass_1_only(pass_kind: Any) -> MissionPass:
    """``pass_kind`` as a ``MissionPass``, refused unless it is Pass 1 (collect).

    A next-stop policy chooses Pass-1 stops only: Pass 2 delivers to every
    slice stop in the queue's order with no selector (Freeze principle 12;
    critic B7 i).
    """
    try:
        kind = MissionPass(pass_kind)
    except ValueError:
        raise ValueError(
            f"pass_kind must name a mission pass {[p.value for p in MissionPass]}, "
            f"got {pass_kind!r}") from None
    if kind is not MissionPass.COLLECT:
        raise ValueError(
            "a next-stop policy chooses Pass-1 stops only: Pass 2 delivers to every slice "
            "stop in the queue's order (Freeze principle 12; critic B7 i)")
    return kind
