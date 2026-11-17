"""The ferry contact plan: what the mission server needs for one stop.

FeRRy Phase 3 (design section 4.2). On the mission clock the supervisor builds
one :class:`ContactPlan` when the mule arrives at a stop and hands it to
``HFLHostMission.run_contact(plan=...)`` / ``deliver_contact(plan=...)``. The
plan says who is solicited, what the contact's airtime costs and which clock
pays for it; the host's commit turns the sessions into ledger lines stamped in
simulated seconds and charges the clock once (design section 4.3).

**Why a plan and not the physics.** The mission package must not know the radio
model: the band classes, the SNR model and the rate table live in ``hermes.l1``
(``contact_link``, ``channel_model``), and the mule wires them. So this module
holds primitives and callables only and imports nothing from ``hermes.l1``,
the scheduler or ``experiments/``. The supervisor passes the link's numbers as
callables bound to the contact's band.

**The gate** (design section 4.4, :meth:`ContactPlan.at_arrival`). A member is
a *target*, and is solicited, when it is within the band's planar range
R_planar(b) of the stop AND its SNR at arrival is at or above the floor.
Everyone else is *unreachable*: never solicited, recorded as TIMEOUT
``answered=False`` (Pass 2: UNDELIVERED) at the arrival time, with no airtime
charged, because a rate of 0 below the floor must never become an infinite
dwell (critic B12). The range test uses S3a's own metric, so at the wide class,
whose planar range is exactly ``rf_range_m``, it never drops a member S3a
admitted; the SNR test bites through the realized channel.

**The clock without a band** (critic A1). ``band=None`` is the channel-free
mode: every member is a target, there is no SNR and no rate, and each contact
costs the cost model's ``session_time_s`` once (the planner's 1 s per contact),
plus the listen window if a reply is missing. All stamps still come from the
clock, so no wall time reaches the scheduler.

**Ground truth stays out** (critic B16). ``drop_uplink`` is the outcome of the
supervisor's keyed availability draw for this contact, not the availability
itself: the host only marks those pushes, and the scheduler sees the TIMEOUT
that follows, as it would for any failed uplink.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from hermes.types import DeviceID

#: The mission clock's ledger kinds the host's commit charges
#: (``hermes.l1.mission_clock.LEDGER_KINDS``; copied, not imported, because
#: this package must not import ``hermes.l1``).
KIND_DWELL = "dwell"
KIND_LISTEN = "listen"

#: ``(device_id, planar distance in m, simulated time in s) -> SNR in dB``,
#: e.g. ``lambda j, d, t: channel.snr_db(t, band, d, link_key=j)``.
SnrFn = Callable[[DeviceID, float, float], float]
#: ``(bytes, SNR in dB) -> seconds``, or None when the SNR is below the floor,
#: e.g. ``lambda n, s: link.dwell_s(n, band, s)``.
DwellFn = Callable[[int, float], Optional[float]]
#: ``SNR in dB -> bit/s`` on the contact's band, 0 below the floor, e.g.
#: ``lambda s: link.rate_bps(band, s)`` (DDIL build D1's session model).
RateFn = Callable[[float], float]

#: DDIL build D1: the longest one session walk may run before it counts as
#: failed, whatever ``session_t_out_s`` says (the step 0 replay's cap).
SESSION_WALK_CAP_S = 600.0


@dataclass(frozen=True)
class SessionWalk:
    """One session under D1's session model (:meth:`ContactPlan.session_walk`).

    ``dur_s`` is the airtime it used, ``done`` whether every bit arrived, and
    ``bits`` how many had arrived by its end (with the bits it started from).
    """

    dur_s: float
    done: bool
    bits: float
    capped: bool = False


def planar_distance_m(a: Sequence[float], b: Sequence[float]) -> float:
    """Distance between two poses with S3a's exact float operations.

    ``hermes.scheduler.stages.s3a_cluster._distance``: the stop and every
    member's position share z = 0 (legs and S3a stay planar; the altitude
    enters only the SNR), so this is the planar distance, and at the wide class
    the gate's ``d <= R_planar(wide) == rf_range_m`` agrees bit for bit with the
    membership test S3a built the stop with.
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def _nonneg(value, name: str) -> float:
    out = _finite(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {out!r}")
    return out


@dataclass(frozen=True, eq=False)
class ContactPlan:
    """One stop's contact, as the supervisor sees it at arrival.

    Build it with :meth:`at_arrival`, which applies the gate; the constructor
    only validates. Fields:

    * ``arrival_ts``: the mission clock at arrival, simulated seconds. The host
      refuses a plan whose clock has moved since (its commit charges the clock
      from here, and ``clock() == max(contact_ts)`` must hold afterwards).
    * ``targets`` / ``unreachable``: the members solicited and the rest, each in
      member order. Together they are exactly the contact's devices.
    * ``clock`` / ``advance``: the mission clock (a zero-argument callable) and
      its ``advance(dt_s, kind)``. Only the host's commit calls ``advance``, on
      the supervisor's thread, once for ``dwell`` and at most once for
      ``listen``. It must be the clock the host was built with
      (``HFLHostMission(now_fn=...)``): after charging, the commit raises
      ``RuntimeError`` unless both read the contact's end.
    * ``band`` / ``band_index``: the band class name and its index (the lines'
      ``band``); both None in the channel-free mode (critic A1).
    * ``stop``: the stop's pose, for the record.
    * ``snr_db``: every member's SNR at arrival (empty without a band).
    * ``snr_floor_db``: the floor the gate used; a session whose SNR at its own
      start has fallen below it is priced at the floor rate (see
      :meth:`session_dwell_s`).
    * ``dwell_fn``: the band's airtime ``(bytes, snr_db) -> s | None``.
    * ``snr_at_fn``: ``(device_id, t_s) -> dB``, the member's SNR at simulated
      time ``t_s`` from this stop (critic C2: each target is priced at its own
      session start while the dwell accumulates); :meth:`at_arrival` binds it
      from the channel's ``snr_fn`` and each member's distance. None: the
      arrival SNR holds for the whole contact.
    * ``drop_uplink``: members whose uplink the supervisor's availability draw
      failed this contact (Pass 1 only). Only targets are affected.
    * ``listen_s``: charged once when any expected reply is missing.
    * ``session_time_s``: the per-contact charge without a band.
    * ``payload_bytes``: the declared payload per direction (D3), or None to
      price the bytes the ledger measured.
    * ``not_ready``: Exp 5 addendum (Study 5.12), Pass 1 only: members whose
      local fit has not finished at arrival on the supervisor's fit clock
      (``hermes/mule/fit_clock.py``). A target among them is told so in the
      solicit, answers with its advert, and is pushed nothing: the commit
      stamps it at the current time with no airtime and no listen, as a
      refused one. Disjoint from ``drop_uplink`` (a device with no update has
      no uplink to drop). Empty in every recorded run.

    DDIL build D1 (the unified plan's step 1), every field off in every
    recorded run:

    * ``session_t_out_s``: the session model. None prices each session at the
      SNR at its own start, the floor rate below the floor (the recorded
      model). A number makes the rate follow the channel in steps of
      ``session_step_s`` (:meth:`session_walk`, through ``rate_fn``): nothing
      moves below the floor, and a session fails once the SNR has stayed
      below it for ``session_t_out_s`` in a row. The host's commit retries a
      failed session once, at the end of the stop's queue.
    * ``session_resumable``: a retry sends only the bits the failed try did
      not deliver (False: it starts again).
    * ``repoll_candidates``: Pass 1, the unreachable members within the band's
      range (gated by the floor at arrival). The host solicits those above the
      floor once more at the stop's end. Empty: no re-poll.
    """

    arrival_ts: float
    targets: Tuple[DeviceID, ...]
    clock: Callable[[], float]
    advance: Callable[[float, str], float]
    unreachable: Tuple[DeviceID, ...] = ()
    band: Optional[str] = None
    band_index: Optional[int] = None
    stop: Optional[Tuple[float, ...]] = None
    snr_db: Mapping[DeviceID, float] = field(default_factory=dict)
    snr_floor_db: Optional[float] = None
    dwell_fn: Optional[DwellFn] = None
    snr_at_fn: Optional[Callable[[DeviceID, float], float]] = None
    drop_uplink: FrozenSet[DeviceID] = frozenset()
    listen_s: float = 1.0
    session_time_s: float = 1.0
    payload_bytes: Optional[int] = None
    not_ready: FrozenSet[DeviceID] = frozenset()
    rate_fn: Optional[RateFn] = None
    session_t_out_s: Optional[float] = None
    session_resumable: bool = False
    session_step_s: float = 1.0
    repoll_candidates: Tuple[DeviceID, ...] = ()
    #: DDIL build D2: the stop's leave-by time on the mission clock (None: no
    #: limit, every recorded run). No session, retry or re-poll starts at or
    #: after it (:meth:`leaves_at`); the members it cuts are committed as timeouts.
    leave_by_ts: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arrival_ts", _finite(self.arrival_ts, "arrival_ts"))
        targets = tuple(self.targets)
        unreachable = tuple(self.unreachable)
        members = targets + unreachable
        if not members:
            raise ValueError("a ContactPlan needs at least one member")
        for did in members:
            if not isinstance(did, str):
                raise TypeError(f"device ids must be strings, got {did!r}")
        if len(set(members)) != len(members):
            raise ValueError(
                f"targets and unreachable must be disjoint and without repeats: "
                f"targets={targets!r}, unreachable={unreachable!r}"
            )
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "unreachable", unreachable)
        if not callable(self.clock) or not callable(self.advance):
            raise TypeError("clock and advance must be callables (the mission clock)")

        drop = frozenset(self.drop_uplink)
        stray = sorted(drop - set(members))
        if stray:
            raise ValueError(f"drop_uplink names devices outside the contact: {stray}")
        object.__setattr__(self, "drop_uplink", drop)
        not_ready = frozenset(self.not_ready)
        stray = sorted(not_ready - set(members))
        if stray:
            raise ValueError(f"not_ready names devices outside the contact: {stray}")
        both = sorted(not_ready & drop)
        if both:
            raise ValueError(
                f"not_ready and drop_uplink overlap on {both}: a device with no update "
                "ready has no uplink to drop"
            )
        object.__setattr__(self, "not_ready", not_ready)

        object.__setattr__(self, "listen_s", _nonneg(self.listen_s, "listen_s"))
        object.__setattr__(
            self, "session_time_s", _nonneg(self.session_time_s, "session_time_s"),
        )
        if self.payload_bytes is not None:
            if isinstance(self.payload_bytes, bool) or not isinstance(
                self.payload_bytes, numbers.Integral
            ):
                raise TypeError(
                    f"payload_bytes must be an int or None, got {self.payload_bytes!r}"
                )
            if self.payload_bytes < 0:
                raise ValueError(f"payload_bytes must be >= 0, got {self.payload_bytes}")
            object.__setattr__(self, "payload_bytes", int(self.payload_bytes))
        if self.stop is not None:
            object.__setattr__(
                self, "stop", tuple(_finite(c, "stop coordinate") for c in self.stop),
            )
        repoll = tuple(self.repoll_candidates)
        stray = [did for did in repoll if did not in unreachable]
        if stray or len(set(repoll)) != len(repoll):
            raise ValueError(
                f"repoll_candidates must be distinct unreachable members, got {repoll!r}")
        object.__setattr__(self, "repoll_candidates", repoll)
        object.__setattr__(self, "session_resumable", bool(self.session_resumable))
        if self.session_resumable and self.session_t_out_s is None:
            raise ValueError("session_resumable needs the session model (session_t_out_s)")
        step = _finite(self.session_step_s, "session_step_s")
        if step <= 0.0:
            raise ValueError(f"session_step_s must be > 0, got {step!r}")
        object.__setattr__(self, "session_step_s", step)
        if self.leave_by_ts is not None:
            object.__setattr__(self, "leave_by_ts", _finite(self.leave_by_ts, "leave_by_ts"))
            if self.band is None:
                raise ValueError("a ContactPlan without a band takes no leave_by_ts: the "
                                 "leave-by time cuts a banded stop's sessions (DDIL build D2)")

        if self.band is None:
            # The channel-free mode (critic A1): no link numbers at all, so a
            # half-wired band cannot silently price contacts at 1 s.
            extra = [
                name for name, value in (
                    ("band_index", self.band_index),
                    ("snr_floor_db", self.snr_floor_db),
                    ("dwell_fn", self.dwell_fn),
                    ("snr_at_fn", self.snr_at_fn),
                    ("rate_fn", self.rate_fn),
                    ("session_t_out_s", self.session_t_out_s),
                ) if value is not None
            ]
            if self.snr_db:
                extra.append("snr_db")
            if repoll:
                extra.append("repoll_candidates")
            if extra:
                raise ValueError(
                    f"a ContactPlan without a band takes no link fields; got {extra}"
                )
            if unreachable:
                raise ValueError(
                    "a ContactPlan without a band has no gate: every member is a target"
                )
            object.__setattr__(self, "snr_db", MappingProxyType({}))
            return

        if not isinstance(self.band, str) or not self.band:
            raise TypeError(f"band must be a class name such as 'wide', got {self.band!r}")
        if (
            isinstance(self.band_index, bool)
            or not isinstance(self.band_index, numbers.Integral)
            or self.band_index < 0
        ):
            raise ValueError(
                f"band_index must be the band class index (an int >= 0), "
                f"got {self.band_index!r}"
            )
        object.__setattr__(self, "band_index", int(self.band_index))
        object.__setattr__(
            self, "snr_floor_db", _finite(self.snr_floor_db, "snr_floor_db"),
        )
        if not callable(self.dwell_fn):
            raise TypeError("a ContactPlan with a band needs dwell_fn(bytes, snr_db)")
        if self.snr_at_fn is not None and not callable(self.snr_at_fn):
            raise TypeError("snr_at_fn must be callable or None")
        snr = {}
        for did in members:
            if did not in self.snr_db:
                raise ValueError(f"snr_db has no arrival SNR for member {did!r}")
            snr[did] = _finite(self.snr_db[did], f"snr_db[{did!r}]")
        object.__setattr__(self, "snr_db", MappingProxyType(snr))
        if self.session_t_out_s is not None:
            t_out = _finite(self.session_t_out_s, "session_t_out_s")
            if t_out <= 0.0:
                raise ValueError(f"session_t_out_s must be > 0, got {t_out!r}")
            object.__setattr__(self, "session_t_out_s", t_out)
            if not callable(self.rate_fn) or not callable(self.snr_at_fn):
                raise TypeError(
                    "the session model (session_t_out_s) needs rate_fn(snr_db) and snr_at_fn")
        if self.rate_fn is not None and not callable(self.rate_fn):
            raise TypeError("rate_fn must be callable or None")

    @property
    def session_model(self) -> bool:
        """True when D1's session model prices this contact (``session_t_out_s`` set)."""
        return self.session_t_out_s is not None

    def leaves_at(self, t_s: float) -> bool:
        """DDIL build D2: True when a session starting at ``t_s`` would start at or
        after the stop's leave-by time; always False without one."""
        return self.leave_by_ts is not None and float(t_s) >= self.leave_by_ts

    def session_walk(self, device_id: DeviceID, t0_s: float, nbytes: int,
                     start_bits: float = 0.0) -> SessionWalk:
        """One session of ``nbytes`` from ``t0_s`` under D1's session model.

        The rate follows the member's SNR in steps of ``session_step_s``, read at
        each step's start (``rate_fn(snr_at(device, t))``); nothing moves while
        the rate is 0 (below the floor). The session fails once the SNR has been
        below the floor for ``session_t_out_s`` in a row, and costs the time it
        used up to then; it also fails, capped, at :data:`SESSION_WALK_CAP_S`.
        ``start_bits`` were delivered by an earlier try (a resumable retry).
        This is the step 0 replay's ``walk_follow``
        (``scripts/exp5/ddil_replay.py``), so its pricing (s) carries over.
        """
        if not self.session_model:
            raise ValueError("session_walk needs the session model (session_t_out_s)")
        t_out = float(self.session_t_out_s)  # type: ignore[arg-type]
        step = self.session_step_s
        rate_fn = self.rate_fn
        assert rate_fn is not None
        need = 8.0 * float(nbytes) - float(start_bits)
        got = 0.0
        t = float(t0_s)
        below_since: Optional[float] = None
        while True:
            if t - t0_s >= SESSION_WALK_CAP_S:
                return SessionWalk(t - t0_s, False, start_bits + got, capped=True)
            snr = self.snr_at(device_id, t)
            rate = float(rate_fn(float(snr)))  # type: ignore[arg-type]
            if rate <= 0.0:
                if below_since is None:
                    below_since = t
                if t + step - below_since >= t_out:
                    return SessionWalk(below_since + t_out - t0_s, False, start_bits + got)
                t += step
                continue
            below_since = None
            if got + rate * step >= need:
                return SessionWalk(t - t0_s + (need - got) / rate, True, start_bits + need)
            got += rate * step
            t += step

    # ------------------------------------------------------------------ #
    # Construction at arrival (the gate)
    # ------------------------------------------------------------------ #

    @classmethod
    def at_arrival(
        cls,
        members: Sequence[DeviceID],
        *,
        clock: Callable[[], float],
        advance: Callable[[float, str], float],
        band: Optional[str] = None,
        band_index: Optional[int] = None,
        stop: Optional[Sequence[float]] = None,
        positions: Optional[Mapping[DeviceID, Sequence[float]]] = None,
        range_planar_m: Optional[float] = None,
        snr_fn: Optional[SnrFn] = None,
        snr_floor_db: Optional[float] = None,
        dwell_fn: Optional[DwellFn] = None,
        drop_uplink: Iterable[DeviceID] = (),
        listen_s: float = 1.0,
        session_time_s: float = 1.0,
        payload_bytes: Optional[int] = None,
        rate_fn: Optional[RateFn] = None,
        session_t_out_s: Optional[float] = None,
        session_resumable: bool = False,
        session_step_s: float = 1.0,
        repoll: bool = False,
    ) -> "ContactPlan":
        """The plan for the stop the mule has just reached; ``arrival_ts = clock()``.

        With a ``band``: every member's planar distance to ``stop`` (from
        ``positions``) and its SNR at arrival, ``snr_fn(j, d_j, arrival_ts)``,
        decide the gate: a target is within ``range_planar_m`` (inclusive, as
        S3a; None skips the range test) AND at or above ``snr_floor_db``. The
        plan keeps ``snr_fn`` bound to each member's distance, so the commit can
        read a target's SNR at its own session start.

        Without a band (``band=None``, critic A1): every member is a target, and
        the link arguments must be left out.

        DDIL build D1: ``rate_fn``, ``session_t_out_s``, ``session_resumable``
        and ``session_step_s`` pass to the plan as given (the session model);
        ``repoll`` makes every unreachable member within range a re-poll
        candidate. All off by default, the recorded plan.
        """
        members = list(members)
        if not members:
            raise ValueError("a contact needs at least one member")
        if len(set(members)) != len(members):
            raise ValueError(f"members repeat: {members!r}")
        arrival = _finite(clock(), "clock()")
        common = dict(
            clock=clock, advance=advance, drop_uplink=frozenset(drop_uplink),
            listen_s=listen_s, session_time_s=session_time_s,
            payload_bytes=payload_bytes,
        )
        if band is None:
            given = [
                name for name, value in (
                    ("band_index", band_index), ("positions", positions),
                    ("range_planar_m", range_planar_m), ("snr_fn", snr_fn),
                    ("snr_floor_db", snr_floor_db), ("dwell_fn", dwell_fn),
                    ("rate_fn", rate_fn), ("session_t_out_s", session_t_out_s),
                ) if value is not None
            ]
            if repoll:
                given.append("repoll")
            if given:
                raise ValueError(f"without a band the plan takes no link arguments; got {given}")
            return cls(
                arrival_ts=arrival, targets=tuple(members), unreachable=(),
                stop=None if stop is None else tuple(stop), **common,
            )

        if stop is None or positions is None:
            raise ValueError("a banded contact needs the stop and the members' positions")
        if not callable(snr_fn):
            raise TypeError("a banded contact needs snr_fn(device_id, d_planar_m, t_s)")
        floor = _finite(snr_floor_db, "snr_floor_db")
        rng = None if range_planar_m is None else _nonneg(range_planar_m, "range_planar_m")
        stop_t = tuple(stop)

        distance: Dict[DeviceID, float] = {}
        snr: Dict[DeviceID, float] = {}
        targets, unreachable, candidates = [], [], []
        for did in members:
            if did not in positions:
                raise ValueError(f"no position for member {did!r}")
            d = planar_distance_m(stop_t, positions[did])
            s = snr_fn(did, d, arrival)
            if isinstance(s, bool) or not isinstance(s, numbers.Real) or math.isnan(s):
                raise ValueError(f"snr_fn gave {s!r} for member {did!r}")
            distance[did] = d
            snr[did] = float(s)
            in_range = rng is None or d <= rng
            if in_range and snr[did] >= floor:
                targets.append(did)
            else:
                unreachable.append(did)
                if repoll and in_range:
                    candidates.append(did)

        def snr_at(did: DeviceID, t_s: float, _fn=snr_fn, _d=distance) -> float:
            return _fn(did, _d[did], t_s)

        d1: Dict[str, object] = {}
        if rate_fn is not None:
            d1["rate_fn"] = rate_fn
        if session_t_out_s is not None:
            d1.update(session_t_out_s=session_t_out_s, session_resumable=session_resumable,
                      session_step_s=session_step_s)
        if candidates:
            d1["repoll_candidates"] = tuple(candidates)
        return cls(
            arrival_ts=arrival, targets=tuple(targets), unreachable=tuple(unreachable),
            band=band, band_index=band_index, stop=stop_t, snr_db=snr,
            snr_floor_db=floor, dwell_fn=dwell_fn, snr_at_fn=snr_at, **common, **d1,
        )

    # ------------------------------------------------------------------ #
    # Pricing (used by the host's commit)
    # ------------------------------------------------------------------ #

    @property
    def members(self) -> Tuple[DeviceID, ...]:
        """Targets then unreachable members (not the contact's order)."""
        return self.targets + self.unreachable

    def snr_at(self, device_id: DeviceID, t_s: float) -> Optional[float]:
        """The member's contact SNR at simulated time ``t_s``; None without a band.

        With no ``snr_at_fn`` the arrival reading holds for the whole contact.
        """
        if self.band is None:
            return None
        if self.snr_at_fn is None:
            return self.snr_db[device_id]
        s = self.snr_at_fn(device_id, t_s)
        if isinstance(s, bool) or not isinstance(s, numbers.Real) or math.isnan(s):
            raise ValueError(f"snr_at_fn gave {s!r} for {device_id!r} at t={t_s!r}")
        return float(s)

    def session_bytes(self, measured: int, directions: int) -> int:
        """The bytes a session's airtime is priced for.

        ``measured`` is what the ledger counted on the wire (a Pass-1 session's
        push plus update, a Pass-2 push, a push whose reply never came). With a
        declared ``payload_bytes`` (per direction, design D3) the session is
        priced for ``directions`` times that instead: the real θ still crosses
        the link, only the charge changes.
        """
        if self.payload_bytes is None:
            return int(measured)
        return int(self.payload_bytes) * int(directions)

    def dwell_s(self, nbytes: int, snr_db: float) -> Optional[float]:
        """The band's airtime for ``nbytes`` at ``snr_db``: ``dwell_fn`` as given.

        None below the floor (the member is unreachable there, critic B12).
        The commit prices sessions with :meth:`session_dwell_s`, which never
        returns None.
        """
        if self.band is None or self.dwell_fn is None:
            raise ValueError("dwell_s needs a band")
        return self.dwell_fn(int(nbytes), float(snr_db))

    def session_dwell_s(self, nbytes: int, snr_db: float) -> float:
        """Seconds of airtime for ``nbytes`` at ``snr_db`` on the band; finite.

        A target passed the gate at arrival, so its session happens. If its SNR
        at its own session start has since fallen below the floor, the band's
        rate there is 0 and ``dwell_fn`` gives None; the session is then priced
        at the floor rate (CQI 1), the slowest the link runs while it is up,
        rather than charged as infinite (critic B12). A dwell that is not a
        finite number >= 0 is refused before anything is charged.
        """
        if self.band is None or self.dwell_fn is None:
            raise ValueError("session_dwell_s needs a band")
        d = self.dwell_fn(int(nbytes), float(snr_db))
        if d is None:
            d = self.dwell_fn(int(nbytes), float(self.snr_floor_db))
            if d is None:
                raise ValueError(
                    f"dwell_fn gave None at the floor {self.snr_floor_db!r} dB: "
                    "its floor disagrees with the plan's"
                )
        return _nonneg(d, "dwell_fn result")


@dataclass(frozen=True)
class ContactCommit:
    """What the host's commit did for one ferry contact (the supervisor's record).

    ``HFLHostMission.last_contact`` holds the latest one. ``dwell_s`` and
    ``listen_s`` are the seconds charged to the clock under those kinds, so
    ``end_ts == arrival_ts + dwell_s + listen_s`` and, right after the commit,
    ``clock() == end_ts == max(contact_ts.values())``. ``session_dwell_s``
    breaks the dwell down per device (zero-dwell devices left out; without a
    band the whole ``session_time_s`` is one contact-wide charge and the map is
    empty). ``missing`` are the targets whose expected reply never came: silent
    (no advert answered this contact's solicit), uplink dropped, or pushed with
    no matching reply by the TTL or by the join deadline. They are what the
    listen window was charged for, and they are stamped at its end. A worker
    still inside its push at the join deadline is not missing: the mule gave
    up on that push, so it is committed as a failed push (TIMEOUT, nothing
    sent, no airtime, stamped when the sessions before it ended, no listen).
    ``uplink_dropped`` are the targets whose push went out marked
    ``uplink_drop`` (a subset of ``missing``). ``stale_discarded`` counts
    adverts, gradients and acks that belonged to another contact (or repeated
    an advert already taken), drained before the solicit or discarded while
    waiting (critic B1/B2).
    """

    arrival_ts: float
    end_ts: float
    dwell_s: float
    listen_s: float
    band: Optional[str]
    band_index: Optional[int]
    targets: Tuple[DeviceID, ...]
    unreachable: Tuple[DeviceID, ...]
    solicited: Tuple[DeviceID, ...]
    answered: Tuple[DeviceID, ...]
    missing: Tuple[DeviceID, ...]
    uplink_dropped: Tuple[DeviceID, ...]
    contact_ts: Mapping[DeviceID, float]
    snr_db: Mapping[DeviceID, Optional[float]]
    session_dwell_s: Mapping[DeviceID, float]
    stale_discarded: Mapping[str, int]
    solicit_id: int
    # Exp 5 addendum (Study 5.12): the targets that answered and were found
    # with no update ready (``ContactPlan.not_ready``), and the targets a push
    # went out to (a model reached them, so each starts a fit on the fit
    # clock), each in member order. Both empty without the fit clock's marks.
    not_ready: Tuple[DeviceID, ...] = ()
    pushed: Tuple[DeviceID, ...] = ()
    # Study 5.12, the device's transmit energy: each collected update's own
    # uplink airtime, its bytes (the declared payload, or the update's
    # measured size) at the SNR its session started at; banded contacts only.
    uplink_dwell_s: Mapping[DeviceID, float] = field(default_factory=dict)
    # DDIL build D1, each in member order and empty when the switch is off:
    # the sessions the session model retried, those that failed on their
    # last try (an answered target whose update or delivery was then lost),
    # and the re-poll's targets (unreachable at arrival, above the floor at
    # the stop's end, solicited then).
    session_retried: Tuple[DeviceID, ...] = ()
    session_failed: Tuple[DeviceID, ...] = ()
    repolled: Tuple[DeviceID, ...] = ()
    # DDIL build D2, in member order and empty without a leave-by time: the
    # members whose session (or retry, or re-poll session) the leave-by time
    # kept from starting. A first try cut is a timeout with nothing sent; a
    # retry cut leaves its failed first try in ``session_failed``.
    left: Tuple[DeviceID, ...] = ()


__all__ = [
    "KIND_DWELL",
    "KIND_LISTEN",
    "SESSION_WALK_CAP_S",
    "ContactCommit",
    "ContactPlan",
    "DwellFn",
    "RateFn",
    "SessionWalk",
    "SnrFn",
    "planar_distance_m",
]
