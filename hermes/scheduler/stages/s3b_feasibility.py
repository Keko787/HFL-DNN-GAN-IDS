"""Stage 3b — deadline feasibility gate.

**Why this stage exists.** S3 computes ``Deadline(j) = t_ref(j) + Φ(j)`` and
S3a propagates the tightest member deadline onto each ``ContactWaypoint`` — but
until this stage was added, nothing in the runtime path ever compared that
deadline to a clock. The deadline was a *sort key only*: a device whose deadline
had already passed, or could not possibly be reached in time, was still queued
and still visited. A source trace of Experiment 4 found exactly this, so the
scheduler's "deadline-aware" description was not true of the code.

This stage closes that gap. It is deliberately a **hard gate** (it removes
candidates) rather than an ordering, because that is the architectural contract
the rest of the scheduler rests on: deterministic rules decide what is *legal*,
and the learned selector may only reorder what survives. Placing feasibility
inside the selector would have inverted that — and would also have been skipped
entirely for single-candidate buckets, which take a short-circuit around the
selector.

**Opt-in by construction.** With no mission budget configured
(``mission_deadline_ts=None``) the gate is a no-op and the queue is unchanged.
That keeps every previously-recorded result reproducible: enforcement turns on
only when a caller supplies a budget.

**Cost model.** Deliberately simple and stated, rather than elaborate and
hidden: a contact costs ``transit + session``, where transit is straight-line
distance at ``cruise_speed_m_s``. There is no propulsion-energy, upload-rate or
return-leg term — Experiment 4 does not model those (Experiment 3 does, via
``EdfFeasibilityPolicy``). A contact is dropped when either

* it cannot be reached before **its own deadline** (`deadline_ts`), or
* serving it would exceed the **remaining mission budget**.

The walk is greedy in EDF order, advancing a simulated pose and clock, so later
estimates account for earlier stops.

**FeRRy Phase 3: one model, one predicate** (design §3.1-§3.2, Freeze
Amendment 7 opens this file). Before Phase 3 the same single-contact check was
copied into four walks (this gate, the D-arm budget walk, FedCS and the mule's
in-flight check). They are now folds over one predicate,
:meth:`FeasibilityModel.admit`, under one of three rules:

* :data:`RULE_DEADLINE_BUDGET` — our arms (S3b): the contact's own deadline
  and the mission budget;
* :data:`RULE_BUDGET` — the whole-scheduler baselines and Pass 2: the budget
  only (Freeze Amendment 8: the per-device deadline is S3b's rule);
* :data:`RULE_NONE` — FedEx (arm D4), which has no gate: always admitted, but
  the times are still reported so its overrun can be measured.

With ``FeasibilityModel.ferry`` None (the default, "legacy mode") the
predicate is today's arithmetic, compared in today's order, and every walk
returns exactly what it returned at afa9526 (pinned by
``tests/golden/test_golden_feasibility.py``). With a :class:`FerryPhysics`
the mission is priced on the simulated mission clock: a contact's dwell is
``Σ bytes / rate`` over the members predicted reachable, the mule must still
fly home and upload inside the budget, and an energy clause, labelled
simulated, can bind; it needs both a capacity and a budget (no budget, no
gate). What ``Deadline(j)`` bounds there is a switch (:data:`DEADLINE_BOUNDS`):
the collection (the default), each stop's own hypothetical return, or the
route's actual delivery. ``cost()`` is unchanged in both modes, because the
D4 CARP split and the probes call it directly.

``pass_kind`` may be given as a :class:`MissionPass` or as its string value
(``"collect"``, ``"deliver"``); the predicate normalises it before pricing,
so the two spellings always price the same (``MissionPass`` is a ``str``
enum, and an identity test would have silently dropped the Pass-1 upload for
the string form).

The scheduler must not import ``hermes.l1``: every piece of physics reaches
this module as a float or a callable that the mule builds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Collection, List, Mapping, Optional, Sequence, Tuple

from hermes.types.scheduler import ContactWaypoint, MissionPass

MulePose = Tuple[float, float, float]

# --------------------------------------------------------------------------- #
# Rules, drop reasons and deadline semantics
# --------------------------------------------------------------------------- #

#: The contact's own deadline and the mission budget (S3b; our arms).
RULE_DEADLINE_BUDGET = "deadline+budget"
#: The mission budget only (D1-D3 and D5, Freeze Amendment 8; Pass 2).
RULE_BUDGET = "budget"
#: No gate (D4): always admitted, times still reported.
RULE_NONE = "none"
RULES: Tuple[str, ...] = (RULE_DEADLINE_BUDGET, RULE_BUDGET, RULE_NONE)

#: Why the predicate rejected a contact. ``delivery`` (route-level
#: ``deadline_bounds="delivery"`` only): serving the contact would deliver an
#: update already on board after its deadline; the contact itself was not
#: late. Appended last, so the order of the other three is unchanged.
REASON_OVERDUE = "overdue"
REASON_BUDGET = "budget"
REASON_ENERGY = "energy"
REASON_DELIVERY = "delivery"
REASONS: Tuple[str, ...] = (REASON_OVERDUE, REASON_BUDGET, REASON_ENERGY, REASON_DELIVERY)

#: What ``Deadline(j)`` bounds in ferry mode (spec Q2; the user kept both
#: delivery readings on 2026-09-29, final check clock F1).
#:
#: * ``collection``, the default: the mule must have finished the contact,
#:   ``arrival + dwell <= Deadline(j)`` — what the scorer tests and what
#:   "participate by" means.
#: * ``delivery_per_stop``: the plan's single-contact formula, ``finish`` +
#:   the return from THIS stop + the upload ``<= Deadline(j)``
#:   (:attr:`Verdict.home`): the update could have been delivered in time had
#:   the mule flown home right after the contact. Checked per stop, it does
#:   not bound when the update actually reaches the cluster, the route's
#:   landing plus the upload (:attr:`FoldResult.home` of the route), which is
#:   later for every stop but the last. This is what ``delivery`` meant at
#:   ef1faa1, renamed, with the same arithmetic.
#: * ``delivery``: route-level. Every update collected on the route must
#:   reach the cluster by its own ``Deadline(j)``. The stop's own clause is
#:   the per-stop one, and a second clause holds its ``home`` to
#:   :attr:`FlightState.deliver_by`, the earliest deadline of the updates
#:   already on board (refused as :data:`REASON_DELIVERY`). ``home`` never
#:   decreases along a route (the triangle inequality; dwell and upload are
#:   >= 0), and the last stop's ``home`` is the route's landing plus the
#:   upload, so holding every admitted stop to both clauses is exactly
#:   holding the route's delivery, as priced, to every collected deadline.
#:   The bound is on the priced route, like the budget clause: the mule
#:   re-checks it at every departure (hermes/mule/mule_main.py), but a
#:   contact that runs longer than priced (a silent member's listen window,
#:   a noisy band) at the stop where Pass 1 ends is never re-checked, since
#:   the updates are on board and the only way left is home, so the landing
#:   can slip past ``deliver_by``. The mission records by how much
#:   (``MissionRunResult.delivery_overrun_s``), as it records the budget's
#:   overrun.
#:
#: ``delivery`` changed meaning after ef1faa1 with no rename shim: the setting
#: is sim-only and new in Phase 3, and no recorded run or committed trace used
#: it.
DEADLINE_BOUNDS_COLLECTION = "collection"
DEADLINE_BOUNDS_DELIVERY_PER_STOP = "delivery_per_stop"
DEADLINE_BOUNDS_DELIVERY = "delivery"
DEADLINE_BOUNDS: Tuple[str, ...] = (
    DEADLINE_BOUNDS_COLLECTION, DEADLINE_BOUNDS_DELIVERY_PER_STOP, DEADLINE_BOUNDS_DELIVERY,
)

#: ``member_admission`` (FeRRy Phase 4, the user's decision 4 (b) of
#: 2026-09-30). ``whole`` admits a contact with all its members or not at all:
#: the recorded rule of this gate and of the D-arm walks, and the default of
#: every arm outside the F family. It leaves the narrow-band cliff (one
#: field-wide contact, so 0 or N devices under a budget), pinned in
#: tests/unit/test_p3_final_fixes_mule.py.
#: ``subset``, on the simulated clock only, re-issues a contact that fails
#: whole with the members that still fit
#: (``hermes.scheduler.plan.member_subset``): the F family's default, and
#: H1-H3's, D1-D3's and D5's when a run asks for it. For the H and D arms only
#: the admission before takeoff reduces a contact; their in-flight check and
#: re-plan keep or drop contacts whole. The values live here, not in
#: ``hermes.scheduler.plan``, so that this gate, the walks and the scheduler
#: can name them without putting the plan package on the recorded import
#: path; ``hermes.scheduler.plan`` restates the tuple, pinned equal by
#: tests/unit/test_p4_member_subset.py.
MEMBER_ADMISSION_WHOLE = "whole"
MEMBER_ADMISSION_SUBSET = "subset"
MEMBER_ADMISSIONS: Tuple[str, ...] = (MEMBER_ADMISSION_WHOLE, MEMBER_ADMISSION_SUBSET)


def _check_rule(rule: str) -> None:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")


# --------------------------------------------------------------------------- #
# Flight state, legs and verdicts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FlightState:
    """Where the mule is, what its clock says, and the energy spent so far.

    ``clock`` is on whatever clock the caller plans with: the wall clock in
    legacy runs, the mission clock in ferry mode. ``energy_j`` is the
    simulated energy spent since takeoff (0 at the dock); only the ferry
    energy clause reads it, and the walks carry it forward so a re-plan does
    not restart from an empty tank (critic B10).

    ``deliver_by`` is the earliest ``Deadline(j)`` of the updates on board:
    the time by which the route must be back at the dock with the upload
    done. Only the route-level ``deadline_bounds="delivery"`` clause reads
    it, and only that mode lowers it (:meth:`FeasibilityModel.admit`); every
    other mode, and the legacy model, carries it through untouched. The
    default ``inf`` (nothing on board, as at takeoff) keeps every caller that
    does not set it exactly where it was. A walk that starts mid-flight must
    be given it, as it is given the energy spent, or it would forget the
    updates it already carries.
    """

    pose: MulePose
    clock: float
    energy_j: float = 0.0
    deliver_by: float = math.inf


@dataclass(frozen=True)
class Leg:
    """The price of flying to one contact and serving it.

    ``total_s`` is the leg's marginal time, ``transit_s + dwell_s``; in
    legacy mode it is exactly ``cost()``'s total (``dwell_s`` is the
    session time). ``return_s`` (stop to dock) and ``upload_s`` (the Pass-1
    upload at the dock) are the tail the ferry budget clause adds; both are
    0.0 in legacy mode, which has no return leg.
    """

    transit_s: float
    dwell_s: float
    total_s: float
    return_s: float = 0.0
    upload_s: float = 0.0


@dataclass(frozen=True)
class Verdict:
    """The predicate's answer for one contact from one flight state.

    ``reason`` is None when admitted, else one of :data:`REASONS`. The times
    are absolute: ``arrival`` at the stop, ``finish`` when its dwell ends
    (the collection time), ``home`` when the mule would be back at the dock
    with the upload done if it flew home right after (legacy: ``finish``, as
    the legacy model has no return leg). ``next_state`` is the state after
    serving the contact, whether or not it was admitted, so a fold that does
    not skip can fly a rejected stop anyway.
    """

    ok: bool
    reason: Optional[str]
    arrival: float
    finish: float
    home: float
    next_state: FlightState


@dataclass(frozen=True)
class FoldResult:
    """One walk of a route under the predicate (:meth:`FeasibilityModel.fold`).

    ``route`` is what gets flown: the admitted contacts when the fold skips,
    every contact when it does not. ``rejected`` lists the contacts the
    predicate refused, in route order, with their reasons; with ``skip`` they
    were left out, without it they were flown anyway. ``verdicts`` has one
    entry per input contact. ``state`` is the flight state after the last
    flown contact and ``home`` the clock back at the dock from there: the
    last flown contact's :attr:`Verdict.home` (after Pass 1 the upload
    included), or :meth:`FeasibilityModel.home_at` when nothing was flown,
    which adds no upload (the fold cannot know whether stops served before
    ``state`` left anything to upload; the caller adds it).
    """

    route: Tuple[ContactWaypoint, ...]
    rejected: Tuple[Tuple[ContactWaypoint, str], ...]
    verdicts: Tuple[Verdict, ...]
    state: FlightState
    home: float

    @property
    def ok(self) -> bool:
        """True when no contact was rejected."""
        return not self.rejected

    def rejected_by(self, reason: str) -> List[ContactWaypoint]:
        """The rejected contacts with ``reason``, in route order."""
        return [wp for wp, why in self.rejected if why == reason]


# --------------------------------------------------------------------------- #
# Ferry physics (Phase 3)
# --------------------------------------------------------------------------- #

#: One member's predicted dwell: ``(planar distance to the stop in metres,
#: pass, SNR offset in dB) -> seconds``. The mule builds it from the contact
#: link and the payload model. ``None`` (or ``inf``) means the member is
#: predicted below the SNR floor: unreachable, so it is never charged
#: (critic B12).
MemberDwellFn = Callable[[float, MissionPass, float], Optional[float]]


@dataclass(frozen=True)
class FerryPhysics:
    """The physics the ferry-mode predicate prices with, built by the mule.

    * ``dock``: where every mission starts and ends; the return leg is priced
      to it.
    * ``member_dwell_s``: one member's predicted dwell (:data:`MemberDwellFn`).
      :meth:`dwell_s` sums it over a contact's members, looking each member's
      position up in ``device_states`` itself (critic B11: the waypoints'
      ``pred_snr_db`` is attached only after planning, so S3b and the
      pre-flight order check cannot read it). None: the mission clock runs
      with no contact band, and a contact costs the model's
      ``session_time_s`` once, as the host's commit charges it (critic A1).
    * ``upload_s``: the predicted Pass-1 upload at the dock, ``UP bytes /
      rate(wide, mean SNR of the held carrier)``: causal, no noise. It must
      return a finite time: a rate of 0 is never charged as ``inf`` (critic
      B12), so a carrier predicted below the floor is the mule's to cap, and
      :meth:`upload_time_s` refuses anything else.
    * ``p_move_w`` / ``p_hover_w``: simulated propulsion power while flying
      and while hovering (the Zeng-Xu-Zhang 2019 rotary-wing model in
      ``hermes.l1.mission_clock.EnergyModel``; SIMULATED, not measured).
    * ``energy_capacity_j``: the simulated battery; None (default) switches
      the energy clause off. A capacity alone does not switch it on: the
      clause is part of the gate, and with no mission budget there is no gate
      at all (the opt-in contract, :meth:`FeasibilityModel.admit`), so it
      binds only when a budget is set as well. Without a budget the capacity
      only describes the simulated battery (e.g. as ``E_ref`` for the L1
      state, design §4.6); a caller that means it as a limit must also set
      the budget.
    * ``deadline_bounds``: :data:`DEADLINE_BOUNDS_COLLECTION` (default),
      :data:`DEADLINE_BOUNDS_DELIVERY_PER_STOP` (each stop by its own
      return and upload) or :data:`DEADLINE_BOUNDS_DELIVERY` (the route's
      actual delivery by every collected deadline); see
      :data:`DEADLINE_BOUNDS`.
    * ``range_m``: the stop's planar range R_planar(b). A member farther away
      is not solicited and costs no dwell; None skips the test (S3a already
      clusters within the range, so it only matters for stale positions).
    * ``device_states``: where member positions come from — the scheduler's
      device-state map (or any mapping from device id to an object with
      ``last_known_position``, or to a position). :class:`FLScheduler` binds
      its own map when this is None. With no map at all, every member is
      priced at the stop itself. Not part of equality or hashing.
    """

    dock: MulePose
    member_dwell_s: Optional[MemberDwellFn]
    upload_s: Callable[[], float]
    p_move_w: float
    p_hover_w: float
    energy_capacity_j: Optional[float] = None
    deadline_bounds: str = DEADLINE_BOUNDS_COLLECTION
    range_m: Optional[float] = None
    device_states: Optional[Mapping[Any, Any]] = field(
        default=None, compare=False, repr=False,
    )

    def __post_init__(self) -> None:
        dock = tuple(float(c) for c in self.dock)
        if len(dock) != 3:
            raise ValueError(f"dock must be an (x, y, z) pose, got {self.dock!r}")
        object.__setattr__(self, "dock", dock)
        if self.member_dwell_s is not None and not callable(self.member_dwell_s):
            raise TypeError("member_dwell_s must be a callable or None")
        if not callable(self.upload_s):
            raise TypeError("upload_s must be a callable")
        for name in ("p_move_w", "p_hover_w"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
            object.__setattr__(self, name, value)
        if self.energy_capacity_j is not None:
            cap = float(self.energy_capacity_j)
            if math.isnan(cap):
                raise ValueError("energy_capacity_j must be a number or None")
            object.__setattr__(self, "energy_capacity_j", cap)
        if self.deadline_bounds not in DEADLINE_BOUNDS:
            raise ValueError(
                f"deadline_bounds must be one of {DEADLINE_BOUNDS}, "
                f"got {self.deadline_bounds!r}"
            )
        if self.range_m is not None:
            rng = float(self.range_m)
            if not rng > 0.0:
                raise ValueError(f"range_m must be > 0 or None, got {self.range_m!r}")
            object.__setattr__(self, "range_m", rng)

    # -- members ---------------------------------------------------------- #

    def member_position(self, did: Any, wp: ContactWaypoint) -> Sequence[float]:
        """A member's last-known position (the stop itself with no map bound)."""
        states = self.device_states
        if states is None:
            return wp.position
        try:
            st = states[did]
        except KeyError:
            raise KeyError(
                f"FerryPhysics: no device state for member {did!r} of the contact at "
                f"{tuple(wp.position)}; members are priced at their last-known "
                f"positions (critic B11)"
            ) from None
        return getattr(st, "last_known_position", st)

    def member_distances_m(self, wp: ContactWaypoint) -> Tuple[float, ...]:
        """Each member's planar distance to the stop, in ``wp.devices`` order.

        The same Euclidean metric S3a clusters with, so every member S3a
        placed at this stop is within its range.
        """
        return tuple(
            _euclid(wp.position, self.member_position(did, wp)) for did in wp.devices
        )

    def dwell_s(
        self,
        wp: ContactWaypoint,
        pass_kind: MissionPass = MissionPass.COLLECT,
        snr_offset_db: float = 0.0,
    ) -> float:
        """Predicted dwell at ``wp``: Σ over the members predicted reachable.

        A member beyond ``range_m`` is not solicited; a member predicted
        below the SNR floor (``member_dwell_s`` gives None or ``inf``) is
        unreachable. Neither is charged (critic B12). One shared channel, so
        the members' times add up (spec Q9). ``snr_offset_db`` shifts every
        member's predicted SNR (the observed-rate adjustment, δ_obs; 0 by
        default, spec). ``member_dwell_s`` always receives the pass as a
        :class:`MissionPass`, whichever spelling the caller used.
        """
        pass_kind = MissionPass(pass_kind)
        if self.member_dwell_s is None:
            raise TypeError(
                "FerryPhysics has no member dwell (no contact band): the model "
                "prices one session per contact"
            )
        total = 0.0
        for d in self.member_distances_m(wp):
            if self.range_m is not None and d > self.range_m:
                continue
            t = self.member_dwell_s(d, pass_kind, snr_offset_db)
            if t is None:
                continue
            t = float(t)
            if math.isnan(t) or t < 0.0:
                raise ValueError(f"member_dwell_s returned {t!r}; need a time >= 0")
            if math.isinf(t):
                continue
            total += t
        return total

    def upload_time_s(self) -> float:
        """The predicted Pass-1 upload, checked: finite and >= 0 (critic B12)."""
        upload = float(self.upload_s())
        if not math.isfinite(upload) or upload < 0.0:
            raise ValueError(
                f"upload_s returned {upload!r}; need a finite time >= 0 (a rate "
                f"of 0 is never charged as inf, critic B12: cap it)"
            )
        return upload

    # -- copies ----------------------------------------------------------- #

    def bind(self, device_states: Mapping[Any, Any]) -> "FerryPhysics":
        """A copy that looks member positions up in ``device_states``."""
        return replace(self, device_states=device_states)

    def with_energy_spent(self, energy_j: float) -> "FerryPhysics":
        """A copy whose capacity is what is left after ``energy_j``.

        For code that forwards only the model — a baseline's own
        ``admit_and_order`` re-admitting mid-flight walks from energy 0, so it
        is handed the remaining capacity instead (critic B10). Unchanged
        without a capacity.
        """
        if self.energy_capacity_j is None or not energy_j:
            return self
        return replace(self, energy_capacity_j=self.energy_capacity_j - float(energy_j))


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FeasibilityModel:
    """Transit/service cost parameters for the gate.

    Defaults are intentionally permissive; a caller that wants the gate to bind
    should set them from the deployment being modelled.

    ``ferry`` (FeRRy Phase 3) is None in legacy mode, which prices a contact
    as ``transit + session_time_s`` with no return leg, exactly as before.
    With a :class:`FerryPhysics` the predicate prices dwell at rate, the
    return leg, the Pass-1 upload and (with a capacity) simulated energy.
    """

    cruise_speed_m_s: float = 5.0
    session_time_s: float = 1.0
    ferry: Optional[FerryPhysics] = None

    def cost(self, frm: MulePose, to: MulePose) -> Tuple[float, float]:
        """Return ``(transit_s, total_s)`` for flying ``frm`` → ``to``."""
        transit = _euclid(frm, to) / max(self.cruise_speed_m_s, 1e-6)
        return transit, transit + self.session_time_s

    # -- Phase 3: legs, the predicate, folds ------------------------------ #

    @property
    def is_ferry(self) -> bool:
        return self.ferry is not None

    def leg(
        self,
        frm: MulePose,
        wp: ContactWaypoint,
        *,
        pass_kind: MissionPass = MissionPass.COLLECT,
        snr_offset_db: float = 0.0,
    ) -> Leg:
        """Price flying ``frm`` → ``wp`` and serving it.

        Legacy: ``cost()``'s transit and total, the session as the dwell, no
        tail. Ferry: the same transit, :meth:`FerryPhysics.dwell_s` as the
        dwell (``session_time_s`` once when the physics has no member dwell:
        the clock without a band), the return leg to the dock, and the upload
        after Pass 1 only. ``pass_kind`` may be the enum or its string value.
        """
        pass_kind = MissionPass(pass_kind)
        transit, total = self.cost(frm, wp.position)
        ferry = self.ferry
        if ferry is None:
            return Leg(transit_s=transit, dwell_s=self.session_time_s, total_s=total)
        if ferry.member_dwell_s is None:
            dwell = self.session_time_s
        else:
            dwell = ferry.dwell_s(wp, pass_kind, snr_offset_db)
        back, _ = self.cost(wp.position, ferry.dock)
        upload = ferry.upload_time_s() if pass_kind is MissionPass.COLLECT else 0.0
        return Leg(transit_s=transit, dwell_s=dwell, total_s=transit + dwell,
                   return_s=back, upload_s=upload)

    def admit(
        self,
        state: FlightState,
        wp: ContactWaypoint,
        *,
        rule: str,
        budget_end: Optional[float],
        pass_kind: MissionPass = MissionPass.COLLECT,
        protected: bool = False,
        snr_offset_db: float = 0.0,
    ) -> Verdict:
        """The single-contact predicate: may the mule serve ``wp`` next?

        ``budget_end`` is the absolute end of the budget; None switches every
        clause off, the energy clause included (the opt-in contract: no
        budget, no gate — in ferry mode too, so a ferry cell without a budget
        stays a control). ``protected`` exempts the contact from its own
        deadline clause (Phase 4's capped devices).

        Legacy (``ferry`` None) reproduces today's comparisons in today's
        order: overdue iff ``clock + transit > deadline_ts`` (deadline rule
        only), then over budget iff ``clock + total > budget_end``; the next
        state is ``(wp.position, clock + total)``, the energy and
        ``deliver_by`` carried through.

        Ferry: ``arrival = clock + transit``, ``finish = arrival + dwell``,
        ``home = finish + return + upload`` (the upload after Pass 1 only).
        The clauses, in the order they are tested (the first that fails is
        the reason):

        1. own deadline (deadline rule, not protected): ``finish <=
           deadline_ts`` under ``collection``, ``home <= deadline_ts`` under
           ``delivery_per_stop`` and ``delivery`` (this stop's own return);
           ``overdue``;
        2. on board (deadline rule, ``delivery`` only, protected or not):
           ``home <= state.deliver_by``, so no update already carried would
           land late as priced; ``delivery``;
        3. budget (deadline and budget rules): ``home <= budget_end``;
        4. energy (only with a capacity, and like every clause only with a
           budget): ``energy + P_move * (transit + return) + P_hover * dwell
           <= capacity``.

        The two deadline clauses come first, as ``overdue`` always has: they
        are the rule's own half, and a contact that fails a deadline is
        reported so even when the budget would refuse it too. Of the two,
        the stop's own is first: a contact that cannot make its own deadline
        is refused for that whatever is on board. A protected contact is
        exempt from its own deadline only: it is still held to the updates
        on board, since serving it would make THEM late, and its own
        deadline, which it may miss, does not lower ``deliver_by`` (it would
        otherwise refuse every later stop for an update that is late
        anyway).

        The next state is ``(wp.position, finish, energy + P_move * transit +
        P_hover * dwell, deliver_by)``, where ``deliver_by`` becomes
        ``min(state.deliver_by, deadline_ts)`` for an admitted contact under
        ``delivery`` whose own clause applied, and is carried through
        unchanged otherwise (every other mode and rule, and a rejected
        contact). :data:`RULE_NONE` always admits and still reports
        ``home``.
        """
        _check_rule(rule)
        pass_kind = MissionPass(pass_kind)
        gated = budget_end is not None and rule != RULE_NONE
        check_deadline = gated and rule == RULE_DEADLINE_BUDGET and not protected
        ferry = self.ferry
        if ferry is None:
            transit, total = self.cost(state.pose, wp.position)
            arrival = state.clock + transit
            finish = state.clock + total
            reason: Optional[str] = None
            if check_deadline and arrival > wp.deadline_ts:
                reason = REASON_OVERDUE
            elif gated and finish > budget_end:
                reason = REASON_BUDGET
            return Verdict(
                ok=reason is None, reason=reason, arrival=arrival, finish=finish,
                home=finish,
                next_state=FlightState(wp.position, finish, state.energy_j, state.deliver_by),
            )

        leg = self.leg(state.pose, wp, pass_kind=pass_kind, snr_offset_db=snr_offset_db)
        arrival = state.clock + leg.transit_s
        finish = arrival + leg.dwell_s
        home = finish + leg.return_s + leg.upload_s
        route_level = ferry.deadline_bounds == DEADLINE_BOUNDS_DELIVERY
        reason = None
        if check_deadline:
            bound = finish if ferry.deadline_bounds == DEADLINE_BOUNDS_COLLECTION else home
            if bound > wp.deadline_ts:
                reason = REASON_OVERDUE
        if (reason is None and route_level and gated and rule == RULE_DEADLINE_BUDGET
                and home > state.deliver_by):
            reason = REASON_DELIVERY
        if reason is None and gated and home > budget_end:
            reason = REASON_BUDGET
        if reason is None and gated and ferry.energy_capacity_j is not None:
            need = (state.energy_j + ferry.p_move_w * (leg.transit_s + leg.return_s)
                    + ferry.p_hover_w * leg.dwell_s)
            if need > ferry.energy_capacity_j:
                reason = REASON_ENERGY
        spent = state.energy_j + ferry.p_move_w * leg.transit_s + ferry.p_hover_w * leg.dwell_s
        deliver_by = state.deliver_by
        if route_level and check_deadline and reason is None:
            deliver_by = min(deliver_by, wp.deadline_ts)
        return Verdict(
            ok=reason is None, reason=reason, arrival=arrival, finish=finish, home=home,
            next_state=FlightState(wp.position, finish, spent, deliver_by),
        )

    def fold(
        self,
        route: Sequence[ContactWaypoint],
        state: FlightState,
        *,
        rule: str,
        budget_end: Optional[float],
        pass_kind: MissionPass = MissionPass.COLLECT,
        skip: bool,
        protected: Collection[ContactWaypoint] = frozenset(),
        snr_offset_db: float = 0.0,
    ) -> FoldResult:
        """Walk ``route`` in order under :meth:`admit`.

        ``skip=True`` is the admission walk every gate runs: a rejected
        contact is left out and the state does not move. ``skip=False`` asks
        whether the route passes *as flown*: every contact is flown, a
        rejected one is reported, and :attr:`FoldResult.ok` is the answer.
        A contact in ``protected`` skips its own deadline clause. Each
        contact is priced from the state the previous one left, so under
        ``deadline_bounds="delivery"`` the deadlines of the admitted
        contacts ride along in :attr:`FlightState.deliver_by`.
        """
        _check_rule(rule)
        pass_kind = MissionPass(pass_kind)
        flown: List[ContactWaypoint] = []
        rejected: List[Tuple[ContactWaypoint, str]] = []
        verdicts: List[Verdict] = []
        cur = state
        home: Optional[float] = None
        for wp in route:
            v = self.admit(
                cur, wp, rule=rule, budget_end=budget_end, pass_kind=pass_kind,
                protected=bool(protected) and wp in protected,
                snr_offset_db=snr_offset_db,
            )
            verdicts.append(v)
            if not v.ok:
                rejected.append((wp, v.reason))  # type: ignore[arg-type]
                if skip:
                    continue
            flown.append(wp)
            cur = v.next_state
            home = v.home
        if home is None:
            home = self.home_at(cur)
        return FoldResult(
            route=tuple(flown), rejected=tuple(rejected), verdicts=tuple(verdicts),
            state=cur, home=home,
        )

    def home_at(self, state: FlightState) -> float:
        """The clock back at the dock if the mule flies home from ``state`` now.

        Legacy: ``state.clock`` (no return leg). Ferry: plus the return
        transit; no upload, as nothing more is collected.
        """
        if self.ferry is None:
            return state.clock
        back, _ = self.cost(state.pose, self.ferry.dock)
        return state.clock + back

    def with_energy_spent(self, energy_j: float) -> "FeasibilityModel":
        """This model with the ferry capacity reduced by ``energy_j``.

        See :meth:`FerryPhysics.with_energy_spent`. Returns ``self`` in
        legacy mode and whenever nothing changes.
        """
        if self.ferry is None:
            return self
        physics = self.ferry.with_energy_spent(energy_j)
        return self if physics is self.ferry else replace(self, ferry=physics)


def _euclid(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


@dataclass(frozen=True)
class FeasibilityResult:
    kept: List[ContactWaypoint]
    dropped_overdue: List[ContactWaypoint]
    dropped_budget: List[ContactWaypoint]
    #: FeRRy Phase 3: contacts the simulated energy clause refused. Always
    #: empty in legacy mode and without a capacity. Counted in ``n_dropped``;
    #: a reader that widens the dropped devices must include it (critic B10).
    dropped_energy: List[ContactWaypoint] = field(default_factory=list)
    #: Contacts the route-level on-board clause refused
    #: (:data:`REASON_DELIVERY`): serving them would have delivered an update
    #: already on board late. Always empty unless ``deadline_bounds`` is
    #: ``delivery``. Counted in ``n_dropped``; a reader that widens the
    #: dropped devices must include it, as it includes ``dropped_budget``
    #: (the device was not late itself either).
    dropped_delivery: List[ContactWaypoint] = field(default_factory=list)
    #: FeRRy Phase 4, plan mode only (``plan_mode="ferry"``): the demanded
    #: devices the plan leaves out by choice, those no clause of the predicate
    #: refuses alone from the dock (drop reason ``plan``,
    #: ``hermes.scheduler.plan.REASON_PLAN``; Phase 4 spec, other choices 8),
    #: as one waypoint per stop. Always empty in legacy mode; appended last so
    #: the positional construction of the fields above is unchanged. Counted in
    #: ``n_dropped``; a reader that widens the dropped devices must include it,
    #: but S3c's planned count (``mission_planned_devices`` in the mule) leaves
    #: it out: these are the plan's choices, not deadline shortfalls (critic C6).
    dropped_plan: List[ContactWaypoint] = field(default_factory=list)

    @property
    def n_dropped(self) -> int:
        return (len(self.dropped_overdue) + len(self.dropped_budget) + len(self.dropped_energy)
                + len(self.dropped_delivery) + len(self.dropped_plan))

    @property
    def dropped(self) -> List[ContactWaypoint]:
        """Every dropped contact: overdue, then over budget, then energy, then
        on-board delivery, then the plan's own choices (the last two are empty
        in every mode but ``delivery`` and plan mode, so the others keep their
        order)."""
        return (list(self.dropped_overdue) + list(self.dropped_budget)
                + list(self.dropped_energy) + list(self.dropped_delivery)
                + list(self.dropped_plan))


def filter_feasible(
    contacts: Sequence[ContactWaypoint],
    *,
    now: float,
    mule_pose: MulePose = (0.0, 0.0, 0.0),
    mission_deadline_ts: Optional[float] = None,
    model: Optional[FeasibilityModel] = None,
    priority: Optional[Callable[[ContactWaypoint], float]] = None,
    state: Optional[FlightState] = None,
    snr_offset_db: float = 0.0,
    member_subsets: Optional[MemberSubsets] = None,
) -> FeasibilityResult:
    """Drop contacts that cannot be served in time. EDF-ordered greedy walk.

    ``mission_deadline_ts`` is an **absolute** timestamp. ``None`` disables the
    gate entirely — every contact is kept, and the result is indistinguishable
    from not calling this stage at all.

    ``priority`` (FeRRy Phase 1, off by default) is a key that outranks the
    deadline in the walk: higher-priority contacts are admitted first and the
    rest share what budget remains. The scheduler passes each contact's
    longest miss streak, so a device that was missed is not also pushed to the
    back of the queue by the wider window its miss earned it.

    ``state`` (FeRRy Phase 3) starts the walk from a flight state instead of
    ``(mule_pose, now)`` with no energy spent — the mid-mission re-admission,
    which must carry the energy already spent (critic B10) and, under
    ``deadline_bounds="delivery"``, the earliest deadline of the updates
    already on board (``deliver_by``). ``snr_offset_db`` is the observed-rate
    adjustment for ferry pricing (0 by default).

    ``member_subsets`` (FeRRy Phase 4, ``member_admission="subset"``, the
    user's decision 4 (b)) is None by default, the recorded walk. Before
    takeoff the scheduler passes the plan's per-member deadlines and device
    states (:class:`MemberSubsets`), and a contact that fails whole is
    re-issued with the members that still fit (:func:`_filter_subsets`); the
    members left out are dropped under the clause that refused them. The
    re-admission in flight never passes it, so the in-flight re-plan keeps or
    drops contacts whole (``routing.replan``'s identity check). Without a
    budget it is inert, as the gate is.

    The walk is a fold, skipping what fails, over
    :meth:`FeasibilityModel.admit` with :data:`RULE_DEADLINE_BUDGET`.

    Returns the survivors plus the rejection reasons, so a caller can log
    *why* a device was not served instead of it vanishing silently.
    """
    if mission_deadline_ts is None:
        return FeasibilityResult(list(contacts), [], [])

    mdl = model or FeasibilityModel()
    # EDF: tightest deadline first, then a stable tie-break so the gate is
    # deterministic across re-runs.
    if priority is None:
        ordered = sorted(contacts, key=lambda c: (c.deadline_ts, c.position, c.devices))
    else:
        ordered = sorted(
            contacts,
            key=lambda c: (-priority(c), c.deadline_ts, c.position, c.devices),
        )

    start = state if state is not None else FlightState(tuple(mule_pose), float(now))  # type: ignore[arg-type]
    if member_subsets is not None:
        return _filter_subsets(ordered, start, mdl, mission_deadline_ts, priority,
                               snr_offset_db, member_subsets)
    walk = mdl.fold(
        ordered, start, rule=RULE_DEADLINE_BUDGET, budget_end=mission_deadline_ts,
        skip=True, snr_offset_db=snr_offset_db,
    )
    return FeasibilityResult(
        list(walk.route),
        walk.rejected_by(REASON_OVERDUE),
        walk.rejected_by(REASON_BUDGET),
        walk.rejected_by(REASON_ENERGY),
        walk.rejected_by(REASON_DELIVERY),
    )


# --------------------------------------------------------------------------- #
# FeRRy Phase 4: member subsets for the whole-contact walks (opt-in)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class MemberSubsets:
    """What a whole-contact walk needs to re-issue part of a contact.

    An S3a contact carries only its members' earliest deadline and worst
    bucket. A contact reduced to some of its members needs each member's own
    ``Deadline(j)`` (S3's, the plan's ``last_plan_deadlines``) and bucket (the
    device states), which only the scheduler holds. So under
    ``member_admission="subset"`` (the user's decision 4 (b) of 2026-09-30)
    the scheduler builds one per plan before takeoff and hands it to this gate
    (:func:`filter_feasible`) or to the arm's own walk (``greedy_budget_walk``,
    ``fedcs_greedy_select``). Nothing in flight builds one: the in-flight
    check, the re-plan and the trim keep or drop contacts whole, as
    ``routing.replan``'s identity check requires.

    :meth:`reduce` is the plan package's ``reduce_stop`` (unit U3), the one
    reduction rule the F family uses too: the contact's position, the members
    in its order, the worst of their buckets and the earliest of their own
    deadlines. It is imported when called, so neither this module nor any
    walk that imports it loads a plan module on the recorded path
    (unit_U3b.md section 1.5).
    """

    deadlines: Mapping[Any, float]
    device_states: Mapping[Any, Any] = field(repr=False)

    def reduce(self, wp: ContactWaypoint, members: Collection[Any]) -> ContactWaypoint:
        """``wp`` with only ``members``: ``wp`` itself when they are all of them."""
        from hermes.scheduler.plan.member_subset import reduce_stop  # noqa: WPS433

        return reduce_stop(wp, members, deadlines=self.deadlines,
                           device_states=self.device_states)


def fold_subsets(
    route: Sequence[ContactWaypoint],
    state: FlightState,
    *,
    model: FeasibilityModel,
    rule: str,
    budget_end: Optional[float],
    member_order: Callable[[ContactWaypoint], Sequence[Any]],
    subsets: MemberSubsets,
    pass_kind: MissionPass = MissionPass.COLLECT,
    snr_offset_db: float = 0.0,
) -> FoldResult:
    """:meth:`FeasibilityModel.fold` with ``skip=True``, re-issuing what fails whole.

    Each contact is tried whole from the state the previous one left, as
    ``fold`` tries it; admitted, it is flown as it is (the same object).
    Refused, its members are walked in ``member_order(wp)`` by the plan
    package's one member walk (``admit_members``, unit U3): a member is
    admitted when the contact reduced to the members admitted so far plus it
    still passes the predicate, and skipped otherwise (skip, not stop). If
    some member fits, the reduced contact is flown and the state moves on its
    verdict; the members left out are rejected as one reduced contact per
    reason, in :data:`REASONS` order (``complements``). If none fits, the
    contact is rejected whole with the reason it failed whole, the object and
    reason ``fold`` gives, and the state does not move.

    Why one pass is final: from a fixed flight state every clause of
    :meth:`FeasibilityModel.admit` is monotone in a contact's member set. The
    dwell is a sum of member times, each >= 0 (:meth:`FerryPhysics.dwell_s`);
    the deadline the contact is held to is its members' minimum; arrival, the
    return leg and the upload do not depend on the members; the energy grows
    with the dwell. So a member refused beside part of the admitted set stays
    refused beside all of it, the pass admits a maximal set for its order, and
    a contact that fits whole would be admitted whole by it: the whole try
    only saves the walk.

    ``verdicts`` has one entry per input contact: the verdict of what was
    flown for it, or of its refusal. ``home`` is as in ``fold``. Where no
    contact is reduced, the result equals ``model.fold(route, state,
    skip=True)`` object for object. There is no ``protected`` set: no
    whole-contact gate has one before takeoff.
    """
    from hermes.scheduler.plan.member_subset import admit_members, complements  # noqa: WPS433

    _check_rule(rule)
    pass_kind = MissionPass(pass_kind)
    flown: List[ContactWaypoint] = []
    rejected: List[Tuple[ContactWaypoint, str]] = []
    verdicts: List[Verdict] = []
    cur = state
    home: Optional[float] = None
    for wp in route:
        v = model.admit(cur, wp, rule=rule, budget_end=budget_end, pass_kind=pass_kind,
                        snr_offset_db=snr_offset_db)
        if not v.ok:
            walk = admit_members(
                model, cur, wp, member_order(wp), rule=rule, budget_end=budget_end,
                deadlines=subsets.deadlines, device_states=subsets.device_states,
                pass_kind=pass_kind, snr_offset_db=snr_offset_db,
            )
            if walk.stop is None:
                verdicts.append(v)
                rejected.append((wp, v.reason))  # type: ignore[arg-type]
                continue
            rejected.extend(complements(wp, walk.refused, deadlines=subsets.deadlines,
                                        device_states=subsets.device_states))
            wp, v = walk.stop, walk.verdict  # type: ignore[assignment]
        verdicts.append(v)
        flown.append(wp)
        cur = v.next_state
        home = v.home
    if home is None:
        home = model.home_at(cur)
    return FoldResult(
        route=tuple(flown), rejected=tuple(rejected), verdicts=tuple(verdicts),
        state=cur, home=home,
    )


def _filter_subsets(
    ordered: Sequence[ContactWaypoint],
    start: FlightState,
    mdl: FeasibilityModel,
    budget_end: float,
    priority: Optional[Callable[[ContactWaypoint], float]],
    snr_offset_db: float,
    subsets: MemberSubsets,
) -> FeasibilityResult:
    """S3b's walk with member subsets (:func:`filter_feasible` given ``member_subsets``).

    The contacts go in S3b's own order; one that fails whole is re-issued with
    the members that still fit (:func:`fold_subsets`). Each member is ranked by
    S3b's contact key applied to its one-member contact (unit_U3b.md section
    2.2):

    * its own ``Deadline(j)`` first: EDF inside a contact as across contacts.
      Service rotates: a served member's next deadline moves later, while a
      member left out keeps its early one and leads the next mission; the
      cheapest dwell first would instead favour the same near, high-SNR
      members every mission (research map section 4);
    * then its predicted dwell. The contact key's position term is common to
      every member of a contact, so the dwell breaks deadline ties before the
      id: among equal deadlines the cheapest first admits the most members (at
      mission 1 every new device shares t_ref and the default window);
    * then the device id;
    * with ``priority`` (``miss_priority``), the member's own priority, its
      miss streak, leads: a missed member's wider window gives it a later
      deadline, and without this it would go last inside its contact, the
      inversion miss priority exists to prevent.

    The members left out join the drop list of the clause that refused them,
    one reduced contact per clause, never ``dropped_plan``; the mule widens
    and records them like any other drop. Like S3b's EDF across contacts, EDF
    inside a contact is not count-optimal when deadlines differ.
    """
    def member_order(wp: ContactWaypoint) -> List[Any]:
        def key(did: Any) -> tuple:
            one = subsets.reduce(wp, (did,))
            lead = () if priority is None else (-priority(one),)
            return lead + (one.deadline_ts,
                           mdl.leg(start.pose, one, snr_offset_db=snr_offset_db).dwell_s, did)
        return sorted(wp.devices, key=key)

    walk = fold_subsets(ordered, start, model=mdl, rule=RULE_DEADLINE_BUDGET,
                        budget_end=budget_end, member_order=member_order, subsets=subsets,
                        snr_offset_db=snr_offset_db)
    return FeasibilityResult(
        list(walk.route),
        walk.rejected_by(REASON_OVERDUE),
        walk.rejected_by(REASON_BUDGET),
        walk.rejected_by(REASON_ENERGY),
        walk.rejected_by(REASON_DELIVERY),
    )
