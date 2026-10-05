"""Stage 3 — Deadline formula + bucket classifier.

Design §6.8::

    Deadline(j) = Time + Deadline_Fulfilment(j) − Idle_Time(j)
        Deadline_Fulfilment = +on_time_history − missed_history
        Idle_Time low  ⇒ shorter deadline

Design §6.2 wording note: ``Deadline_Fulfilment`` here is the *per-device
fulfilment window* (seconds), not the history delta on its own. We keep
the window baseline on ``DeviceSchedulerState.deadline_fulfilment_s`` and
let the fast/slow-phase deltas nudge it.

Two-phase adaptation (design §7 principle 4):

* **Fast phase** — in-mission: :func:`fold_round_close_delta` ingests a
  ``RoundCloseDelta`` from ``HFLHostMission``. On-time shrinks the
  window; missed/partial widens it.
* **Slow phase** — at dock: :func:`fold_cluster_amendment` applies
  ``ClusterAmendment.deadline_overrides`` (explicit per-device ts) and
  merges any ``registry_deltas`` that touch deadline fields.

The bucket classifier is the *only* hard-rank layer exposed to S3.5
(§4). Inside each bucket S3.5 (placeholder or RL actor) picks an order.

FeRRy Phase 1 adds :class:`DeadlineLaw`: the recorded additive law stays the
default and runs its original arithmetic; the multiplicative, clamped law, the
PARTIAL/TIMEOUT split and one-shot cluster overrides are opt-in. Every function
takes the law as an optional argument, and ``None`` is the recorded behaviour.

FeRRy Phase 3 (spec Q1) adds the law's ``time_scale``. The recorded constants
(the −5 s / +10 s steps and the 5 s floor, the multiplicative law's 5 s / 300 s
clamps, the 60 s initial window) were set against missions of about 10 s of
wall clock; on the simulated mission clock a two-pass mission takes minutes
(design §0 finding 2), so they all move together by one factor. At the
recorded value 1.0 every function here computes exactly what it did before.
Each constant is stated in the recorded unit and multiplied by the factor
where it is applied; ``FLScheduler(initial_window_s=...)`` states Φ₀ the same
way, so the recorded 60 s scales whether it is given or left at its default.
The helpers at the bottom express a window in missions (critic A7).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from hermes.types import (
    Bucket,
    BUCKET_PRIORITY,
    ClusterAmendment,
    DeviceID,
    DeviceSchedulerState,
    MissionOutcome,
    RoundCloseDelta,
)


# Fast-phase deltas — seconds nudged per outcome. Small so the window
# drifts, not whipsaws.
FAST_PHASE_ON_TIME_SHRINK_S: float = 5.0
FAST_PHASE_MISSED_WIDEN_S: float = 10.0

# Floor on the fulfilment window — never let the formula drive it to zero.
MIN_DEADLINE_FULFILMENT_S: float = 5.0

# Probation length for the NEW bucket. A never-served device holds the
# top rank tier so it cannot be starved before its first contact, but a
# node that is physically unreachable never succeeds, so ``is_new`` never
# clears and it would hold that tier forever — outranking scheduled work
# on every round while the widening fulfilment window that is supposed to
# shed it only ever moves it within its own bucket. After this many failed
# attempts the device is demoted to the ordinary tier, where its large
# Φ deprioritises it as intended.
NEW_BUCKET_ATTEMPT_LIMIT: int = 3


# --------------------------------------------------------------------------- #
# The deadline law (FeRRy Phase 1)
# --------------------------------------------------------------------------- #

LAW_ADDITIVE = "additive"
LAW_MULTIPLICATIVE = "multiplicative"
#: The Exp 5 addendum's unit U11 (Study 5.2): the selection literature's
#: deadlines in place of FeRRy's per-device one. ``round`` (after FedCS): every
#: device is due by the round's end, ``round_s`` after the plan, and its merge
#: window is the round, so updates older than about one round are cut.
#: ``pref`` (after Oort): no per-device cutoff at all (due by the round's end,
#: which the budget already enforces, and no merge cutoff); the arm weighs
#: slow devices down instead (``plan_score.demand_weights``' speed factor).
#: Neither moves a device's window with its outcomes.
LAW_ROUND = "round"
LAW_PREF = "pref"
ROUND_LAWS = (LAW_ROUND, LAW_PREF)
DEADLINE_LAWS = (LAW_ADDITIVE, LAW_MULTIPLICATIVE) + ROUND_LAWS


class DeadlineLawError(ValueError):
    """Raised for an unknown law or an invalid law parameter."""


class DeadlineOverrideRefused(ValueError):
    """A cluster deadline override reached a scheduler that must refuse it.

    Overrides are absolute timestamps stamped on the cluster's wall clock; on
    the simulated mission clock they would compare a wall stamp with a
    simulated one (critic B3), so the mule refuses them in that mode.
    """


def _scaled(value: float, scale: float) -> float:
    """``value`` in the law's time unit; the value itself at the recorded 1.0."""
    return value if scale == 1.0 else value * scale


@dataclass(frozen=True)
class DeadlineLaw:
    """How the fast phase moves a device's fulfilment window Φ.

    ``additive`` is the law every recorded run used: −5 s after an on-time
    delivery, +10 s after any miss, floored at 5 s and unbounded above. It has
    no parameters, so it is always exactly that law.

    ``multiplicative`` multiplies Φ by a factor per outcome and clamps it::

        Φ ← min(Φ_max, max(Φ_min, β · Φ)),   β = β_on | β_partial | β_timeout

    with β_on < 1 after a CLEAN and 1 ≤ β_partial ≤ β_timeout after a miss.
    A miss by a device that answered (a PARTIAL, or a TIMEOUT after its
    advert arrived, such as a dropped uplink) relaxes Φ by β_partial, less
    than β_timeout for a device that never answered, because it was reachable.
    The step scales the clamped window, so Φ·β holds even when the stored
    value starts outside [Φ_min, Φ_max].

    *Why this form.* The window acts as a per-device service threshold — a
    device is due once its deadline passes — which is the threshold form
    Cui et al.'s age-of-update policy takes (the novelty audit's "deadline as
    age threshold"). Under the multiplicative law log Φ moves by log β per
    outcome, so for a device on time with probability p whose misses are
    timeouts the expected drift per contact is p·log β_on + (1 − p)·log
    β_timeout. It is zero at p* = log β_timeout / (log β_timeout − log β_on),
    about 0.65 with the defaults: more reliable devices tighten toward Φ_min,
    less reliable ones relax toward Φ_max, and the clamps keep Φ bounded
    either way. The additive law breaks even at p = 2/3 but has no ceiling,
    so for any device on time less often than that Φ grows without bound and
    its deadline recedes indefinitely.

    ``expire_overrides`` makes a cluster deadline override one-shot: it stops
    applying once its time has passed or once the device's next outcome
    arrives. None follows the form (on for multiplicative), since the recorded
    law never clears an override (SEC26_Code_Audit.md: "the bypass is sticky").

    ``time_scale`` (FeRRy Phase 3, spec Q1) multiplies every time constant of
    the law: the additive steps and floor, and the multiplicative law's
    clamps ``phi_min`` / ``phi_max`` (stated in recorded seconds and applied
    as ``phi_min * time_scale``, ``phi_max * time_scale``). The factors β are
    ratios and do not move. 1.0, the default, is the recorded law exactly and
    is left out of :meth:`to_params`, so recorded configurations, traces and
    CSV cells are unchanged; :func:`time_scale_for_period` gives the value
    for a mission period.
    """

    form: str = LAW_ADDITIVE
    beta_on: float = 0.8
    beta_partial: float = 1.25
    beta_timeout: float = 1.5
    phi_min: float = MIN_DEADLINE_FULFILMENT_S
    phi_max: float = 300.0
    expire_overrides: Optional[bool] = None
    time_scale: float = 1.0
    #: Unit U11: the round's length (s), every device's deadline after the plan
    #: under ``round`` and ``pref`` (the arm's mission budget); None otherwise.
    round_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.form not in DEADLINE_LAWS:
            raise DeadlineLawError(
                f"unknown deadline law {self.form!r}; choose one of {DEADLINE_LAWS}"
            )
        if self.form in ROUND_LAWS:
            r = self.round_s
            if (isinstance(r, bool) or not isinstance(r, (int, float))
                    or not math.isfinite(r) or r <= 0.0):
                raise DeadlineLawError(
                    f"the {self.form!r} law needs round_s, the round's length > 0 "
                    f"(the arm's mission budget), got {r!r}"
                )
            object.__setattr__(self, "round_s", float(r))
        elif self.round_s is not None:
            raise DeadlineLawError(
                f"round_s belongs to the {ROUND_LAWS} laws, not {self.form!r}")
        if not 0.0 < self.beta_on < 1.0:
            raise DeadlineLawError(f"beta_on must be in (0, 1), got {self.beta_on}")
        if not 1.0 <= self.beta_partial <= self.beta_timeout:
            raise DeadlineLawError(
                "need 1 <= beta_partial <= beta_timeout, got "
                f"{self.beta_partial} and {self.beta_timeout}"
            )
        if not 0.0 < self.phi_min <= self.phi_max:
            raise DeadlineLawError(
                f"need 0 < phi_min <= phi_max, got {self.phi_min} and {self.phi_max}"
            )
        if (isinstance(self.time_scale, bool)
                or not isinstance(self.time_scale, (int, float))
                or not math.isfinite(self.time_scale) or self.time_scale <= 0.0):
            raise DeadlineLawError(
                f"time_scale must be a finite number > 0, got {self.time_scale!r}"
            )
        # JSON may hand back an int; the law always holds a float.
        object.__setattr__(self, "time_scale", float(self.time_scale))

    @property
    def is_additive(self) -> bool:
        return self.form == LAW_ADDITIVE

    @property
    def is_round(self) -> bool:
        """Unit U11's ``round`` or ``pref``: one deadline for every device."""
        return self.form in ROUND_LAWS

    @property
    def is_recorded(self) -> bool:
        """Exactly the law every recorded run used: additive, sticky overrides,
        recorded time unit."""
        return self.is_additive and not self.expires_overrides and self.time_scale == 1.0

    @property
    def expires_overrides(self) -> bool:
        if self.expire_overrides is None:
            return self.form == LAW_MULTIPLICATIVE or self.form in ROUND_LAWS
        return bool(self.expire_overrides)

    # The law's time constants in its own unit (the recorded ones at 1.0).

    @property
    def floor_s(self) -> float:
        """The additive law's floor on Φ (5 s recorded)."""
        return _scaled(MIN_DEADLINE_FULFILMENT_S, self.time_scale)

    @property
    def on_time_shrink_s(self) -> float:
        """The additive law's step after a CLEAN (−5 s recorded)."""
        return _scaled(FAST_PHASE_ON_TIME_SHRINK_S, self.time_scale)

    @property
    def missed_widen_s(self) -> float:
        """The additive law's step after a miss (+10 s recorded)."""
        return _scaled(FAST_PHASE_MISSED_WIDEN_S, self.time_scale)

    @property
    def phi_bounds(self) -> Tuple[float, float]:
        """The multiplicative law's clamps as applied: ``(Φ_min, Φ_max) * time_scale``."""
        return (_scaled(self.phi_min, self.time_scale),
                _scaled(self.phi_max, self.time_scale))

    def with_time_scale(self, time_scale: float) -> "DeadlineLaw":
        """This law in another time unit (validated like the constructor)."""
        return replace(self, time_scale=time_scale)

    def clamp(self, phi: float) -> float:
        """Bound a window: the 5 s floor alone under the additive law (unit U11's
        laws keep the stored window as it is: they never read it)."""
        if self.is_round:
            return float(phi)
        if self.is_additive:
            return max(self.floor_s, float(phi))
        lo, hi = self.phi_bounds
        return min(hi, max(lo, float(phi)))

    def next_window(
        self, phi: float, outcome: MissionOutcome, *, answered: bool = False,
    ) -> float:
        """Φ after one outcome.

        Under the multiplicative law a miss relaxes by β_partial when the
        device was reachable — a PARTIAL, or a TIMEOUT after its advert
        arrived (``answered``, e.g. an uplink dropped mid-session) — and by
        β_timeout when it never answered. The step scales the clamped window,
        the Φ the deadline actually used, not the raw stored value. Unit U11's
        laws do not move it.
        """
        if self.is_round:
            return float(phi)
        if self.is_additive:
            if outcome is MissionOutcome.CLEAN:
                return max(self.floor_s, phi - self.on_time_shrink_s)
            return phi + self.missed_widen_s
        if outcome is MissionOutcome.CLEAN:
            beta = self.beta_on
        elif outcome is MissionOutcome.PARTIAL or answered:
            beta = self.beta_partial
        else:
            beta = self.beta_timeout
        return self.clamp(beta * self.clamp(phi))

    def break_even_on_time_rate(self) -> float:
        """On-time rate p* at which Φ neither tightens nor relaxes on average.

        For the multiplicative law with all misses timeouts; for the additive
        law, the rate at which −5 s and +10 s cancel (2/3).
        """
        if self.is_additive:
            return FAST_PHASE_MISSED_WIDEN_S / (
                FAST_PHASE_MISSED_WIDEN_S + FAST_PHASE_ON_TIME_SHRINK_S
            )
        up, down = math.log(self.beta_timeout), math.log(self.beta_on)
        return up / (up - down)

    def to_params(self) -> dict:
        """Everything but the form, for ``MuleConfig.deadline_params``.

        ``time_scale`` appears only when it is not the recorded 1.0, so every
        recorded configuration, trace and CSV cell keeps its exact form.
        """
        raw = asdict(self)
        raw.pop("form")
        if raw.get("time_scale") == 1.0:
            raw.pop("time_scale")
        if raw.get("round_s") is None:
            raw.pop("round_s")              # unit U11's, only under its laws
        return raw

    @classmethod
    def from_config(
        cls, form: Optional[str], params: Optional[Mapping] = None
    ) -> "DeadlineLaw":
        kwargs = dict(params or {})
        unknown = set(kwargs) - set(cls.__dataclass_fields__) - {"form"}
        if unknown:
            raise DeadlineLawError(f"unknown deadline parameter(s): {sorted(unknown)}")
        kwargs.pop("form", None)
        return cls(form=form or LAW_ADDITIVE, **kwargs)


# The additive law's constants for an optional law: the module constants
# themselves when there is no law (the recorded behaviour), else the law's,
# which are the same values at the recorded time scale.

def _floor(law: Optional[DeadlineLaw]) -> float:
    return MIN_DEADLINE_FULFILMENT_S if law is None else law.floor_s


def _shrink(law: Optional[DeadlineLaw]) -> float:
    return FAST_PHASE_ON_TIME_SHRINK_S if law is None else law.on_time_shrink_s


def _widen(law: Optional[DeadlineLaw]) -> float:
    return FAST_PHASE_MISSED_WIDEN_S if law is None else law.missed_widen_s


# --------------------------------------------------------------------------- #
# Deadline math
# --------------------------------------------------------------------------- #

def compute_idle_time(state: DeviceSchedulerState, now: float) -> float:
    """Seconds since the last on-time participation, floored at 0.

    A never-seen device with ``idle_time_ref_ts == 0`` gets idle=0, which
    keeps its first-round deadline at exactly ``Time + Deadline_Fulfilment``
    instead of being artificially short. New-device bucket handles the
    prioritisation instead.
    """
    if state.idle_time_ref_ts <= 0.0:
        return 0.0
    return max(0.0, now - state.idle_time_ref_ts)


def compute_deadline(
    state: DeviceSchedulerState,
    now: float,
    window_scale: float = 1.0,
    *,
    law: Optional[DeadlineLaw] = None,
) -> float:
    """Design §6.8 formula.

    ``deadline_override_ts`` short-circuits the formula — the cluster
    amendment is authoritative when present (slow-phase wins over
    fast-phase drift for this round). Under a ``law`` that expires overrides,
    one whose time has passed no longer applies.

    ``window_scale`` is the mission-level multiplier from S3c
    (:class:`~hermes.scheduler.stages.s3c_mission_window.MissionWindowAdapter`).
    It stretches the *fulfilment* term only, leaving the per-device state it was
    derived from untouched — so the two adaptation loops stay separable, and at
    the default of 1.0 this is exactly the original formula.
    """
    if state.deadline_override_ts is not None:
        expired = (
            law is not None
            and law.expires_overrides
            and now > state.deadline_override_ts
        )
        if not expired:
            return state.deadline_override_ts
    if law is not None and law.is_round:
        # Unit U11: one deadline for every device, the round's end.
        return now + law.round_s
    fulfilment = effective_window(state, law=law)
    return now + fulfilment * window_scale - compute_idle_time(state, now)


def effective_window(
    state: DeviceSchedulerState, *, law: Optional[DeadlineLaw] = None,
) -> float:
    """The fulfilment window Φ the deadline uses, before any S3c scaling.

    The recorded 5 s floor under the additive law (or none), the law's clamps
    under the multiplicative one, both in the law's time unit. The merge
    cutoff of decision D5 reads the same value, so the deadline and the cutoff
    never disagree about Φ.
    """
    if law is None or law.is_additive:
        return max(_floor(law), state.deadline_fulfilment_s)
    if law.form == LAW_ROUND:
        return law.round_s          # unit U11 (FedCS): the merge window is the round
    if law.form == LAW_PREF:
        return math.inf             # unit U11 (Oort): no cutoff
    return law.clamp(state.deadline_fulfilment_s)


# --------------------------------------------------------------------------- #
# Bucket classifier
# --------------------------------------------------------------------------- #

def classify_bucket(
    state: DeviceSchedulerState,
    now: float,
    beacon_window_s: float = 30.0,
    new_attempt_limit: int = NEW_BUCKET_ATTEMPT_LIMIT,
) -> Bucket:
    """Assign the design §4 bucket tag.

    Priority (see :data:`BUCKET_PRIORITY`):

    1. ``NEW``  — registered but never served (``is_new=True``), for up to
       ``new_attempt_limit`` failed attempts; see
       :data:`NEW_BUCKET_ATTEMPT_LIMIT`
    2. ``SCHEDULED_THIS_ROUND`` — in the current slice with a deadline
    3. ``BEACON_ACTIVE`` — recent beacon but not in slice (opportunistic)

    Devices that fit none of the above trigger a ``ValueError`` — the
    caller (S1) should not have admitted them.
    """
    beacon_fresh = (
        state.last_beacon_ts > 0.0
        and (now - state.last_beacon_ts) <= beacon_window_s
    )
    if state.is_new:
        # Hold the top tier while the device is still on probation. Past
        # the limit, demote — but only when a lower tier will actually
        # accept it, so demotion never converts a bucketed device into a
        # silent drop at the caller's ``except ValueError``.
        still_on_probation = state.missed_count < new_attempt_limit
        if still_on_probation or not (state.is_in_slice or beacon_fresh):
            return Bucket.NEW
    if state.is_in_slice:
        return Bucket.SCHEDULED_THIS_ROUND
    if beacon_fresh:
        return Bucket.BEACON_ACTIVE
    raise ValueError(
        f"classify_bucket: device {state.device_id!r} has no bucket "
        f"(not new, not in slice, no fresh beacon)"
    )


# --------------------------------------------------------------------------- #
# Fast-phase — consume RoundCloseDelta from HFLHostMission
# --------------------------------------------------------------------------- #

def fold_round_close_delta(
    state: DeviceSchedulerState,
    delta: RoundCloseDelta,
    *,
    law: Optional[DeadlineLaw] = None,
) -> DeviceSchedulerState:
    """Apply one in-mission delta to the scheduler's view of a device.

    Mutates-then-returns for ergonomic caller code; ``DeviceSchedulerState``
    is a plain dataclass so this is cheap.

    On-time outcome:
        - clear ``is_new`` (distribution landed, so no longer brand new)
        - shrink the fulfilment window toward the floor
        - refresh ``idle_time_ref_ts`` + ``last_contact_ts``
        - refresh ``last_clean_ts`` + ``last_clean_round`` (baseline ages)
        - reset ``miss_streak``

    Partial/timeout outcome:
        - widen the fulfilment window
        - extend ``miss_streak``
        - record contact_ts but do **not** reset idle_time_ref_ts (a
          failed attempt doesn't reset the "when were you last reliable"
          clock), nor the ``last_clean_*`` fields the baselines age from

    ``law`` None or additive moves Φ by the recorded −5 s / +10 s; a
    multiplicative law scales and clamps it, and one that expires overrides
    clears the device's override on any outcome (it has been acted on).
    """
    if delta.device_id != state.device_id:
        raise ValueError(
            f"fold_round_close_delta: delta for {delta.device_id!r} applied "
            f"to state for {state.device_id!r}"
        )

    state.last_contact_ts = delta.contact_ts
    state.last_outcome = delta.outcome
    state.last_utility = delta.utility
    # Freeze Amendment 3 — carry the Oort baseline's raw inputs alongside the
    # derived utility. Inert for H0–H3: emitters that do not set these send
    # None/0, so the fold is a no-op and no HERMES arm's behaviour changes.
    if delta.local_loss is not None:
        state.last_loss = delta.local_loss
    if delta.num_examples:
        state.last_num_examples = delta.num_examples
    state.last_served_round = delta.mission_round
    # FeRRy Phase 2 — reachability history for the Whittle baseline (D3). A
    # synthetic TIMEOUT (a device dropped or abandoned without an attempt) is
    # not an observation of reachability. Read by nothing else.
    if not getattr(delta, "synthetic", False):
        state.reach_attempts += 1
        if delta.outcome is MissionOutcome.CLEAN or getattr(delta, "answered", False):
            state.reach_answered += 1

    if delta.outcome is MissionOutcome.CLEAN:
        state.is_new = False
        state.idle_time_ref_ts = delta.contact_ts
        # Freeze Amendment 5 — the D1/D2 baselines age a device from its last
        # successful participation. Only a CLEAN sets these, so the synthetic
        # TIMEOUT fed for an abandoned device cannot reset its age. Inert for
        # H0–H3, which read neither field.
        state.last_clean_ts = delta.contact_ts
        state.last_clean_round = delta.mission_round
        state.on_time_count += 1
        state.miss_streak = 0
        if law is None or law.is_additive:
            state.deadline_fulfilment_s = max(
                _floor(law),
                state.deadline_fulfilment_s - _shrink(law),
            )
        else:
            state.deadline_fulfilment_s = law.next_window(
                state.deadline_fulfilment_s, delta.outcome,
            )
    else:
        state.missed_count += 1
        state.miss_streak += 1
        if law is None or law.is_additive:
            state.deadline_fulfilment_s = (
                state.deadline_fulfilment_s + _widen(law)
            )
        else:
            state.deadline_fulfilment_s = law.next_window(
                state.deadline_fulfilment_s, delta.outcome,
                answered=bool(getattr(delta, "answered", False)),
            )

    if law is not None and law.expires_overrides:
        state.deadline_override_ts = None

    return state


# --------------------------------------------------------------------------- #
# Slow-phase — consume ClusterAmendment at dock
# --------------------------------------------------------------------------- #

def fold_cluster_amendment(
    device_states: Dict[DeviceID, DeviceSchedulerState],
    amendment: ClusterAmendment,
    *,
    law: Optional[DeadlineLaw] = None,
    refuse_overrides: bool = False,
) -> None:
    """Apply ``deadline_overrides`` + relevant ``registry_deltas`` to the map.

    Only devices the scheduler already tracks are touched — slice
    membership (S1) is responsible for admitting new devices, not the
    amendment fold.

    ``registry_deltas`` fields supported here:
        * ``last_known_position`` — tuple[float, float, float]
        * ``deadline_fulfilment_s`` — float override from cluster, clamped
          to the ``law``'s bounds (the 5 s floor alone when it is additive)
        * ``spectrum_sig`` (FeRRy Phase 3, design §4.7) — the device's latest
          SNR per contact band class, as a ``SpectrumSig`` carrying
          ``contact_class_snr_db`` or as that mapping itself; each class it
          reports updates ``spectrum_snr_db``, the others keep their last
          value. Legacy DOWNs never carry it.
    Anything else is ignored; the full registry row lives in
    ``HFLHostCluster``, not in scheduler state.

    ``refuse_overrides`` (FeRRy Phase 3, set on the simulated mission clock)
    raises :class:`DeadlineOverrideRefused` before anything is folded when the
    amendment carries deadline overrides: they are absolute wall-clock stamps
    (critic B3). None are issued today.
    """
    if refuse_overrides and amendment.deadline_overrides:
        raise DeadlineOverrideRefused(
            f"cluster deadline overrides for {sorted(map(str, amendment.deadline_overrides))} "
            "refused: they are wall-clock stamps and this scheduler runs on the "
            "simulated mission clock"
        )
    for did, new_ts in amendment.deadline_overrides.items():
        st = device_states.get(did)
        if st is None:
            continue
        st.deadline_override_ts = new_ts

    for did, patch in amendment.registry_deltas.items():
        st = device_states.get(did)
        if st is None or not isinstance(patch, dict):
            continue
        if "last_known_position" in patch:
            pos = patch["last_known_position"]
            if isinstance(pos, tuple) and len(pos) == 3:
                st.last_known_position = pos  # type: ignore[assignment]
        if "deadline_fulfilment_s" in patch:
            val = patch["deadline_fulfilment_s"]
            if isinstance(val, (int, float)):
                if law is None or law.is_additive:
                    st.deadline_fulfilment_s = max(
                        _floor(law), float(val)
                    )
                else:
                    st.deadline_fulfilment_s = law.clamp(float(val))
        if "spectrum_sig" in patch:
            snr = _contact_class_snr(patch["spectrum_sig"])
            if snr:  # an empty report observes nothing: keep what is known
                # The latest reading per class: a class this report does not
                # mention keeps its last known value.
                merged = dict(st.spectrum_snr_db or {})
                merged.update(snr)
                st.spectrum_snr_db = merged
        # Sprint 1.5 H7 — delivery_priority flows cluster→mule via
        # registry_deltas so S3a clustering's tie-breaker reads the
        # current cluster-bumped value. Accept ``int`` and ``float``
        # (post-pickle round-trips can promote ints to floats); reject
        # ``bool`` explicitly because in Python ``bool`` is a subclass
        # of ``int`` and ``True/False`` would silently become 1/0.
        if "delivery_priority" in patch:
            val = patch["delivery_priority"]
            if isinstance(val, bool):
                pass  # not a meaningful priority; ignore.
            elif isinstance(val, (int, float)):
                st.delivery_priority = int(val)


def _contact_class_snr(sig: Any) -> Optional[Dict[str, float]]:
    """``{class: SNR dB}`` from a SpectrumSig-like value, or None if it has none.

    Accepts an object carrying ``contact_class_snr_db`` or that mapping
    itself; non-numeric entries, booleans and non-finite values are dropped
    (a NaN would poison every later comparison).
    """
    raw = getattr(sig, "contact_class_snr_db", sig)
    if not isinstance(raw, Mapping):
        return None
    return {
        str(k): float(v) for k, v in raw.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    }


# --------------------------------------------------------------------------- #
# Windows in missions (FeRRy Phase 3, spec Q1, critic A7)
# --------------------------------------------------------------------------- #

#: The mission length, in seconds, that the recorded time constants were set
#: against: Exp 4's unbudgeted cells run about 10 s of wall clock per mission
#: (design §0 finding 2; budgeted cells run 3-7 s, critic A7).
LEGACY_MISSION_PERIOD_S: float = 10.0


def time_scale_for_period(
    t_nom_s: float, legacy_period_s: float = LEGACY_MISSION_PERIOD_S,
) -> float:
    """The ``time_scale`` that stretches the recorded constants to ``t_nom_s``.

    Design Q1: ``deadline_time_scale = T_nom / 10 s``, where ``T_nom`` is the
    cell's nominal two-pass mission period on the simulated clock. The same
    number of missions then fits in a window as did on the wall clock.
    """
    t = float(t_nom_s)
    ref = float(legacy_period_s)
    if not (math.isfinite(t) and t > 0.0 and math.isfinite(ref) and ref > 0.0):
        raise ValueError(
            f"need finite periods > 0, got t_nom_s={t_nom_s!r}, "
            f"legacy_period_s={legacy_period_s!r}"
        )
    return t / ref


def initial_window_for_missions(
    missions: float, t_nom_s: float, *, time_scale: float,
) -> float:
    """``FLScheduler(initial_window_s=...)`` for a Φ₀ of ``missions`` periods.

    Critic A7: Φ₀ = 60 s was a placeholder, so Phase 3 sweeps it in missions,
    the unit that means the same on either clock. The window itself is
    ``missions * t_nom_s`` seconds on the scheduler's clock; the scheduler
    states Φ₀ in the deadline law's recorded unit and multiplies it by its
    ``deadline_time_scale``, like every other constant of the law (spec Q1),
    so this returns ``missions * t_nom_s / time_scale``. ``time_scale`` must
    be the scheduler's own ``deadline_time_scale`` and has no default on
    purpose: a window in clock seconds handed over unscaled would be
    stretched a second time. At the Q1 value ``time_scale = T_nom / 10 s``
    the result is ``10 * missions``, so the recorded 60 s is six missions.
    """
    m, t, s = float(missions), float(t_nom_s), float(time_scale)
    if not (math.isfinite(m) and m > 0.0 and math.isfinite(t) and t > 0.0
            and math.isfinite(s) and s > 0.0):
        raise ValueError(
            f"need missions, t_nom_s and time_scale > 0, got {missions!r}, "
            f"{t_nom_s!r} and {time_scale!r}"
        )
    return m * t / s


def window_in_missions(window_s: float, t_nom_s: float) -> float:
    """A window in seconds on the scheduler's clock as a number of nominal
    mission periods (e.g. ``FLScheduler.effective_initial_window_s``)."""
    t = float(t_nom_s)
    if not (math.isfinite(t) and t > 0.0):
        raise ValueError(f"t_nom_s must be finite and > 0, got {t_nom_s!r}")
    return float(window_s) / t
