"""Simulated mission clock, flight constants and energy model (FeRRy Phase 3).

**Why this module exists.** The mule used one wall clock for three separate
jobs: mission time (planning ``now``, Deadline(j), the S3b budget stamp, the
in-flight check, contact stamps), transport and coordination timers (session
TTLs, socket, dock and bootstrap waits) and measurement (event ``ts``,
``duration_s``). It never flew either: its pose jumped from stop to stop, so the
in-flight budget check counted host compute and TTL waits instead of flight and
airtime. FeRRy Phase 3 puts mission time on a :class:`MissionClock` that is
charged with the physics of each step, and leaves transport and measurement on
the wall clock.

**What charges the clock.** Every charge names a ``kind``, and the clock keeps a
per-mission ledger of the seconds charged to each (:data:`LEDGER_KINDS`):

* ``transit`` - the supervisor, before each stop: ``|pose - stop| / v``.
* ``dwell`` - the host's commit, once per contact: the sum over the answered
  targets of ``8 * bytes / rate_bps(band, SNR)`` (``ContactLink.dwell_s``:
  rates are in bit/s), each target's SNR taken at its own session start; with
  no band, the cost model's ``session_time_s`` per contact.
* ``listen`` - the host's commit: :attr:`FlightModel.listen_s` once per contact
  that misses any expected reply, uplink-dropped devices included (the mule
  cannot tell a dropped uplink from silence).
* ``return`` - the supervisor, after the last stop of each pass, also after an
  abort, a re-plan to nothing or a pass with no stops: ``|pose - dock| / v``.
* ``upload`` - the supervisor at the inter-pass dock: ``8 * UP bytes /
  rate_bps(wide, backhaul SNR)``.
* ``turnaround`` - the supervisor, once per mission after Pass 1's return,
  whether or not anything was uploaded, so every mission advances the clock.
* ``dock_wait`` - the supervisor on the DOWN: :meth:`MissionClock.advance_to`
  the cluster's simulated time (the Lamport sync when several mules share it).

**Invariants.**

* *Monotone.* :meth:`MissionClock.advance` refuses a negative charge and
  :meth:`MissionClock.advance_to` is a max, so the clock never goes back.
* *Finite.* An infinite or NaN charge raises instead of poisoning every later
  stamp. A rate of 0 below the SNR floor has to become a finite decision before
  it reaches the clock: a contact member below the floor is unreachable and
  costs no dwell; a backhaul below the floor is a lost upload whose charge the
  caller caps.
* *Never paced.* A charge is an addition. Nothing sleeps for simulated time, so
  a 5 km leg costs no wall time and cannot open the 30 s silence behind finding
  P-02 (``DeveloperDocs/Codebase Review/00_Critical_Problem_Areas.md``).
* *Absolute.* The clock is never reset: Deadline(j), ``last_clean_ts`` and the
  cluster's deadline overrides are absolute stamps compared across missions.
  Only the ledger is per mission (:meth:`MissionClock.reset_ledger`, at
  takeoff).

**The epoch.** The clock starts at :data:`SIM_EPOCH_S` = 1e6 s. It has to be
above 0.0, which scheduler state uses for "never" (``idle_time_ref_ts``,
``last_clean_ts``, ``last_beacon_ts``; at afa9526 ``s3_deadline.py:231``,
``max_aoi.py:79``, ``s1_eligibility.py:40``): a CLEAN stamped at 0.0 would
read as a device never served. It has to stay below :data:`SIM_CEILING_S` =
1e9 s, which wall time passed in 2001, so a stamp's value says which clock made
it. The clock refuses to reach the ceiling and ``advance_to`` refuses a wall
stamp, so the two domains cannot mix silently. They must not: with a wall
reference and a simulated ``now`` every idle term clamps to 0, and with a
simulated reference and a wall ``now`` every served device is overdue.

**Energy** is not accumulated here. The model is time-based, so a mission's
energy is a pure function of its ledger (:meth:`EnergyModel.energy_j`, reached
through :attr:`FlightModel.energy`). Resetting the ledger resets the energy,
and the two can never drift apart. The flight power is the Zeng-Xu-Zhang
model's at the cruise speed (:func:`zeng_power_w`), and a :class:`FlightModel`
refuses an energy model priced for another speed.

The module is numpy-free and imports nothing from the scheduler or from
``experiments/`` (finding A-01). ``hermes.l1`` does not re-export it; the
package's exports stay as they were.
"""

from __future__ import annotations

import math
import numbers
import threading
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

Pose = Tuple[float, float, float]

#: Simulated time at which every mission clock starts, in seconds.
SIM_EPOCH_S: float = 1.0e6

#: The clock never reaches this. ``time.time()`` has been above it since
#: 2001-09-09, so any stamp below it is simulated and any stamp above it is wall
#: time.
SIM_CEILING_S: float = 1.0e9

#: The one dock every mule takes off from and lands at: the origin, which is
#: also where ``MuleSupervisor``'s pose starts and where D4's FedEx tour returns.
#: ``hermes.processes.mule.DOCK_POSE`` is the same pose.
DOCK_POSE: Pose = (0.0, 0.0, 0.0)

#: Everything the clock can be charged for, in the order a ledger lists them.
LEDGER_KINDS: Tuple[str, ...] = (
    "transit",
    "dwell",
    "listen",
    "return",
    "upload",
    "turnaround",
    "dock_wait",
)

#: The kinds the energy model charges at flight power: the mule is flying.
MOVE_KINDS: Tuple[str, ...] = ("transit", "return")
#: The kinds it charges at hover power: the mule holds position at a stop.
HOVER_KINDS: Tuple[str, ...] = ("dwell", "listen")
#: The kinds it does not charge: the mule is on the ground at the dock.
GROUND_KINDS: Tuple[str, ...] = ("upload", "turnaround", "dock_wait")

_KINDS = frozenset(LEDGER_KINDS)


def _real(value, name: str) -> float:
    """Return ``value`` as a finite float, or raise.

    Durations and stamps are computed numbers. A bool, a string or None that
    reaches the clock is a bug upstream, and an infinite or NaN one would carry
    into every stamp after it, so each is refused loudly rather than coerced.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    try:
        out = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def _check_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError(
            f"unknown ledger kind {kind!r}; expected one of {LEDGER_KINDS}"
        )


class MissionClock:
    """Simulated seconds for one mule, charged step by step.

    Calling the clock returns the current simulated time, so one instance can
    be handed to ``FLScheduler(now_fn=...)`` and ``HFLHostMission(now_fn=...)``
    and every mission-time read sees the same value. ``MuleSupervisor`` keeps
    it under its existing ``_now`` attribute: some tests bind supervisor
    methods onto stand-ins that carry their own ``_clock``, so the mission
    clock must not be stored under that name.

    Only the supervisor thread charges the clock. The host's commit runs on the
    calling thread, which is the supervisor's, and worker threads never touch
    it. The lock makes each charge atomic, so a read from another thread (a
    trace emitter) never sees one half-applied; it does not make charging from
    several threads meaningful, because their charges would land in wall-clock
    order.

    ``__slots__`` is deliberate: stand-in clocks in older tests are moved with
    ``clock.t = ...``, and on this class such an assignment raises instead of
    silently doing nothing.
    """

    __slots__ = ("_lock", "_now_s", "_ledger", "_ledger_start_s")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._now_s: float = SIM_EPOCH_S
        self._ledger: Dict[str, float] = dict.fromkeys(LEDGER_KINDS, 0.0)
        self._ledger_start_s: float = SIM_EPOCH_S

    def __call__(self) -> float:
        """The current simulated time, in seconds (a drop-in ``now_fn``)."""
        with self._lock:
            return self._now_s

    def advance(self, dt_s: float, kind: str) -> float:
        """Charge ``dt_s`` seconds of ``kind`` and return the new time.

        A zero charge is legal (a stop at the current pose, a contact with
        nothing to send). A negative, infinite or NaN ``dt_s``, an unknown
        ``kind``, or a charge that would carry the clock to
        :data:`SIM_CEILING_S` raises ``ValueError``; a non-number raises
        ``TypeError``. A refused charge changes nothing.
        """
        dt = _real(dt_s, "dt_s")
        _check_kind(kind)
        if dt < 0.0:
            raise ValueError(
                f"dt_s must be >= 0, got {dt!r} for {kind!r}: "
                "the mission clock never goes back"
            )
        with self._lock:
            t = self._now_s + dt
            if t >= SIM_CEILING_S:
                raise ValueError(
                    f"charging {dt!r} s of {kind!r} would carry the clock to "
                    f"{t!r}, at or past {SIM_CEILING_S!r}, where stamps read as "
                    "wall time"
                )
            self._now_s = t
            self._ledger[kind] += dt
            return t

    def advance_to(self, t_s: float, kind: str) -> float:
        """Move the clock forward to ``t_s`` if that is later; return the time.

        A monotone max, for the Lamport sync at the dock: the mule adopts the
        cluster's simulated time when the cluster is ahead and keeps its own
        otherwise, and the gap is charged to ``kind``. A ``t_s`` at or before
        the current time changes nothing, including a 0.0 that stands for
        "nothing ingested yet". A wall stamp (``t_s >= SIM_CEILING_S``), an
        infinite or NaN ``t_s`` or an unknown ``kind`` raises ``ValueError``
        and changes nothing.
        """
        t = _real(t_s, "t_s")
        _check_kind(kind)
        if t >= SIM_CEILING_S:
            raise ValueError(
                f"t_s={t!r} is at or past {SIM_CEILING_S!r}: a wall-clock stamp "
                "cannot become simulated time"
            )
        with self._lock:
            if t <= self._now_s:
                return self._now_s
            self._ledger[kind] += t - self._now_s
            self._now_s = t
            return t

    def ledger(self) -> Dict[str, float]:
        """Seconds charged per kind since the last :meth:`reset_ledger`.

        A fresh dict holding every kind of :data:`LEDGER_KINDS` in that order,
        zeros included, so a trace's breakdown always has the same keys. Its
        values sum to ``clock() - ledger_start_s`` up to float rounding.
        """
        with self._lock:
            return dict(self._ledger)

    def reset_ledger(self) -> None:
        """Start a new mission's ledger. The clock itself does not move."""
        with self._lock:
            self._ledger = dict.fromkeys(LEDGER_KINDS, 0.0)
            self._ledger_start_s = self._now_s

    @property
    def ledger_start_s(self) -> float:
        """The time of the last :meth:`reset_ledger` (the epoch before any)."""
        with self._lock:
            return self._ledger_start_s

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"MissionClock(now_s={self._now_s!r}, "
                f"ledger_start_s={self._ledger_start_s!r})"
            )


# --------------------------------------------------------------------------- #
# Zeng, Xu and Zhang (2019): rotary-wing propulsion power
# --------------------------------------------------------------------------- #

# Table I of the paper, the 2 kg-class quadrotor behind the default powers
# (sources in the EnergyModel docstring). The model needs the rotor radius R
# and the blade speed Omega only through A and U_tip = Omega R.
_ZENG_W_N = 20.0        # aircraft weight W, N
_ZENG_RHO = 1.225       # air density rho, kg/m^3
_ZENG_A_M2 = 0.503      # rotor disc area A, m^2, as the table lists it
_ZENG_U_TIP = 120.0     # blade tip speed U_tip, m/s
_ZENG_S = 0.05          # rotor solidity s
_ZENG_D0 = 0.6          # fuselage drag ratio d0
_ZENG_K = 0.1           # induced-power correction factor k
_ZENG_DELTA = 0.012     # profile drag coefficient delta

# v0 = 4.03 m/s, P0 = 79.86 W and Pi = 88.63 W.
_ZENG_V0 = math.sqrt(_ZENG_W_N / (2.0 * _ZENG_RHO * _ZENG_A_M2))
_ZENG_P0 = _ZENG_DELTA / 8.0 * _ZENG_RHO * _ZENG_S * _ZENG_A_M2 * _ZENG_U_TIP ** 3
_ZENG_PI = (1.0 + _ZENG_K) * _ZENG_W_N ** 1.5 / math.sqrt(2.0 * _ZENG_RHO * _ZENG_A_M2)


def zeng_power_w(speed_m_s: float) -> float:
    """Propulsion power in W at forward speed ``speed_m_s``, SIMULATED.

    Eq. (6) of Zeng, Xu and Zhang (2019) with their Table I set (see
    :class:`EnergyModel`); ``zeng_power_w(0.0)`` is the hover power P0 + Pi.

    The induced term is evaluated as ``sqrt(1 / (sqrt(1 + x^2) + x))`` with
    ``x = V^2 / (2 v0^2)``. That is the paper's
    ``(sqrt(1 + V^4 / (4 v0^4)) - V^2 / (2 v0^2))^(1/2)`` rearranged: the
    paper's difference cancels catastrophically as the speed grows, and by
    about 1e5 m/s rounding can leave it negative, so its square root fails.
    The two agree to float rounding at flight speeds.

    A negative, infinite or NaN speed, or one so large that the power is not
    finite, raises ``ValueError``; a non-number raises ``TypeError``.
    """
    v = _real(speed_m_s, "speed_m_s")
    if v < 0.0:
        raise ValueError(f"speed_m_s must be >= 0, got {v!r}")
    x = v * v / (2.0 * _ZENG_V0 * _ZENG_V0)
    power = (
        _ZENG_P0 * (1.0 + 3.0 * v * v / (_ZENG_U_TIP * _ZENG_U_TIP))
        + _ZENG_PI * math.sqrt(1.0 / (math.sqrt(1.0 + x * x) + x))
        + 0.5 * _ZENG_D0 * _ZENG_RHO * _ZENG_S * _ZENG_A_M2 * v * v * v
    )
    if not math.isfinite(power):
        raise ValueError(f"the propulsion power at {v!r} m/s is not finite")
    return power


@dataclass(frozen=True)
class EnergyModel:
    """Rotary-wing propulsion energy, counted by time. SIMULATED, not measured.

    Model: Y. Zeng, J. Xu and R. Zhang, "Energy Minimization for Wireless
    Communication With Rotary-Wing UAV", IEEE Trans. Wireless Commun. 18(4),
    2019 (arXiv:1804.02238), eq. (6), the propulsion power at forward speed V
    (:func:`zeng_power_w`)::

        P(V) = P0 * (1 + 3 V^2 / U_tip^2)                           blade profile
             + Pi * (sqrt(1 + V^4 / (4 v0^4)) - V^2 / (2 v0^2))^(1/2)  induced
             + 0.5 * d0 * rho * s * A * V^3                           parasite

        P0 = (delta / 8) * rho * s * A * Omega^3 * R^3
        Pi = (1 + k) * W^(3/2) / sqrt(2 rho A)
        U_tip = Omega R,   v0 = sqrt(W / (2 rho A))

    Parameter set: Table I of the journal version, a 2 kg-class quadrotor:
    W = 20 N, rho = 1.225 kg/m^3, A = 0.503 m^2, U_tip = 120 m/s, s = 0.05,
    d0 = 0.6, k = 0.1, delta = 0.012, so v0 = 4.03 m/s. The set was read
    through secondary sources: W from Yan, Chen and Yang (IEEE WCL, 2021); A,
    v0, d0, U_tip, s and rho from Liu et al. (arXiv:2403.15410, Table 1);
    delta and k from the arXiv v1, which shares them. R and Omega enter only
    as A and U_tip = Omega R (Omega^3 R^3 = U_tip^3); R = 0.4 m and
    Omega = 300 rad/s fit both. The arXiv v1's heavier set (W = 100 N) gives
    257 J/m at 5 m/s and is not used.

    With the table's A = 0.503 m^2 that gives P0 = 79.86 W and Pi = 88.63 W.
    The exact disc area pi R^2 = 0.5027 m^2 would give 79.80 W and 88.66 W.
    Either way the defaults are the same to 0.1 W:

    * ``p_move_w`` = P(5 m/s) = 143.6 W while the mule flies (transit and
      return), which is 28.7 J/m at 5 m/s;
    * ``p_hover_w`` = P(0) = P0 + Pi = 168.5 W while it holds position at a
      stop (dwell and listen);
    * ``speed_m_s`` = 5 m/s, the forward speed ``p_move_w`` is the power at.
      :meth:`at_speed` gives the model at another speed.

    A mission's energy is ``E = p_move_w * t_move + p_hover_w * (t_dwell +
    t_listen)``, taken from the clock's ledger by :meth:`energy_j`.

    Declared assumptions, all modelled rather than measured:

    * ``p_move_w`` is the power at one speed, ``speed_m_s``. A
      :class:`FlightModel` whose cruise speed differs refuses the model, so a
      speed sweep cannot silently keep the 5 m/s power: at 10 m/s that would
      overstate flight energy by 14 %.
    * 5 m/s (the Freeze's cruise speed) is below this model's minimum-power
      speed, about 10.2 m/s (126 W, 12.3 J/m). At 5 m/s the mule draws 85 % of
      hover power, which is why energy is counted by time and hovering is
      charged.
    * Climb and descent to the flight altitude are not charged; legs are
      planar.
    * Ground time (upload, turnaround, dock wait) is not charged.
    * The radio is not charged: sending 25 KB costs about 1e-4 J, at least
      four orders of magnitude below the hover energy over the same airtime.
    * The capacity clause is off by default (``capacity_j=None``). When set,
      the admission predicate enforces it; this class only carries the value.
    * Other platforms are a sensitivity case, not the default: a DJI M100
      class is about 83-100 J/m and an AERPAW LAM6 class about 233-391 J/m at
      5 m/s (both derived from published figures, not measured). Exp 3's
      placeholder of 10 J/m (``experiments/calibration.toml``) stays in Exp 3.
    * ``status`` labels every energy figure the model produces: "simulated".
    """

    p_move_w: float = 143.6
    p_hover_w: float = 168.5
    capacity_j: Optional[float] = None
    status: str = "simulated"
    speed_m_s: float = 5.0

    def __post_init__(self) -> None:
        for name in ("p_move_w", "p_hover_w"):
            value = _real(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
            object.__setattr__(self, name, value)
        speed = _real(self.speed_m_s, "speed_m_s")
        if speed <= 0.0:
            raise ValueError(f"speed_m_s must be > 0, got {speed!r}")
        object.__setattr__(self, "speed_m_s", speed)
        if self.capacity_j is not None:
            capacity = _real(self.capacity_j, "capacity_j")
            if capacity <= 0.0:
                raise ValueError(
                    f"capacity_j must be > 0 or None (no clause), got {capacity!r}"
                )
            object.__setattr__(self, "capacity_j", capacity)
        if not isinstance(self.status, str) or not self.status:
            raise ValueError(
                f"status must be a non-empty label such as 'simulated', "
                f"got {self.status!r}"
            )

    @classmethod
    def at_speed(
        cls, speed_m_s: float, *, capacity_j: Optional[float] = None
    ) -> "EnergyModel":
        """The Zeng model for flight at ``speed_m_s``.

        ``p_move_w`` = P(speed) and ``p_hover_w`` = P(0), both rounded to
        0.1 W like the published figures, so ``at_speed(5.0)`` is exactly the
        default ``EnergyModel()``. ``capacity_j`` passes through; a speed that
        is not > 0 raises ``ValueError``.
        """
        speed = _real(speed_m_s, "speed_m_s")
        return cls(
            p_move_w=round(zeng_power_w(speed), 1),
            p_hover_w=round(zeng_power_w(0.0), 1),
            capacity_j=capacity_j,
            speed_m_s=speed,
        )

    def energy_j(self, ledger: Mapping[str, float]) -> float:
        """Joules spent over the time in ``ledger`` (a ``MissionClock.ledger()``).

        Flight time (transit, return) at ``p_move_w``, hover time (dwell,
        listen) at ``p_hover_w``, ground time (upload, turnaround, dock_wait)
        free. A missing kind counts as 0. An unknown kind or a negative,
        infinite or NaN entry raises ``ValueError``.
        """
        seconds: Dict[str, float] = {}
        for kind, value in ledger.items():
            _check_kind(kind)
            s = _real(value, f"ledger[{kind!r}]")
            if s < 0.0:
                raise ValueError(f"ledger[{kind!r}] must be >= 0, got {s!r}")
            seconds[kind] = s
        move_s = sum(seconds.get(k, 0.0) for k in MOVE_KINDS)
        hover_s = sum(seconds.get(k, 0.0) for k in HOVER_KINDS)
        return self.p_move_w * move_s + self.p_hover_w * hover_s


@dataclass(frozen=True)
class FlightModel:
    """The mule's flight constants in ferry mode. Assumptions, all sweepable.

    * ``cruise_speed_m_s`` = 5 m/s: the Freeze's cruise speed (its decision
      D2) and the S3b cost model's default. It still awaits a platform
      citation.
    * ``dock`` = :data:`DOCK_POSE`: one dock shared by every mule, where each
      mission takes off and lands. Queueing at it is not modelled.
    * ``turnaround_s`` = 30 s, once per mission after Pass 1's return, upload
      or not: Exp 3's ``dock_time_s`` (``experiments/exp3/sim_env.py:157``).
    * ``listen_s`` = 1 s, once per contact that misses any expected reply:
      the frozen 1 s session.
    * ``energy``: the :class:`EnergyModel` for flight at ``cruise_speed_m_s``,
      never None once constructed. None (the default) derives it,
      ``EnergyModel.at_speed(cruise_speed_m_s)``, which at 5 m/s is
      ``EnergyModel()``. One passed in must be for the cruise speed
      (``energy.speed_m_s == cruise_speed_m_s``) or construction raises
      ``ValueError``: ``EnergyModel(capacity_j=...)``, the way the capacity
      clause is switched on, carries the 5 m/s power. For the same reason
      ``dataclasses.replace(flight, cruise_speed_m_s=v)`` raises unless it
      also passes ``energy=None`` (or an energy model for ``v``).

    :meth:`leg_s` prices a leg with exactly the arithmetic of the S3b cost
    model (``FeasibilityModel.cost``), so at the same speed the clock charges,
    bit for bit, the leg the planner predicted.
    """

    cruise_speed_m_s: float = 5.0
    dock: Pose = DOCK_POSE
    turnaround_s: float = 30.0
    listen_s: float = 1.0
    energy: Optional[EnergyModel] = None

    def __post_init__(self) -> None:
        speed = _real(self.cruise_speed_m_s, "cruise_speed_m_s")
        if speed <= 0.0:
            raise ValueError(f"cruise_speed_m_s must be > 0, got {speed!r}")
        object.__setattr__(self, "cruise_speed_m_s", speed)
        for name in ("turnaround_s", "listen_s"):
            value = _real(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
            object.__setattr__(self, name, value)
        dock = tuple(self.dock)
        if len(dock) != 3:
            raise ValueError(f"dock must be an (x, y, z) pose, got {self.dock!r}")
        object.__setattr__(
            self, "dock", tuple(_real(c, "dock coordinate") for c in dock)
        )
        energy = self.energy
        if energy is None:
            energy = EnergyModel.at_speed(speed)
            object.__setattr__(self, "energy", energy)
        elif not isinstance(energy, EnergyModel):
            raise TypeError(
                f"energy must be an EnergyModel or None, got {type(energy).__name__}"
            )
        if energy.speed_m_s != speed:
            raise ValueError(
                f"energy.p_move_w={energy.p_move_w!r} W is the power at "
                f"{energy.speed_m_s!r} m/s, not at cruise_speed_m_s={speed!r}: "
                "pass energy=None to derive it, "
                f"energy=EnergyModel.at_speed({speed!r}), or an EnergyModel "
                f"declared with speed_m_s={speed!r}"
            )

    def leg_s(self, frm: Sequence[float], to: Sequence[float]) -> float:
        """Seconds to fly the straight line ``frm`` -> ``to`` at cruise speed.

        The same expression, and so the same float operations, as the transit
        term of ``FeasibilityModel.cost`` in ``stages/s3b_feasibility.py``.
        """
        dist = sum((x - y) ** 2 for x, y in zip(frm, to)) ** 0.5
        return dist / max(self.cruise_speed_m_s, 1e-6)

    @property
    def move_j_per_m(self) -> float:
        """Flight energy per metre at cruise speed (28.7 J/m by default)."""
        return self.energy.p_move_w / self.cruise_speed_m_s


__all__ = [
    "DOCK_POSE",
    "GROUND_KINDS",
    "HOVER_KINDS",
    "LEDGER_KINDS",
    "MOVE_KINDS",
    "SIM_CEILING_S",
    "SIM_EPOCH_S",
    "EnergyModel",
    "FlightModel",
    "MissionClock",
    "zeng_power_w",
]
