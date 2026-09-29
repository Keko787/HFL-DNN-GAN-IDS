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
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

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
DEADLINE_LAWS = (LAW_ADDITIVE, LAW_MULTIPLICATIVE)


class DeadlineLawError(ValueError):
    """Raised for an unknown law or an invalid law parameter."""


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
    """

    form: str = LAW_ADDITIVE
    beta_on: float = 0.8
    beta_partial: float = 1.25
    beta_timeout: float = 1.5
    phi_min: float = MIN_DEADLINE_FULFILMENT_S
    phi_max: float = 300.0
    expire_overrides: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.form not in DEADLINE_LAWS:
            raise DeadlineLawError(
                f"unknown deadline law {self.form!r}; choose one of {DEADLINE_LAWS}"
            )
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

    @property
    def is_additive(self) -> bool:
        return self.form == LAW_ADDITIVE

    @property
    def is_recorded(self) -> bool:
        """Exactly the law every recorded run used: additive, sticky overrides."""
        return self.is_additive and not self.expires_overrides

    @property
    def expires_overrides(self) -> bool:
        if self.expire_overrides is None:
            return self.form == LAW_MULTIPLICATIVE
        return bool(self.expire_overrides)

    def clamp(self, phi: float) -> float:
        """Bound a window: the 5 s floor alone under the additive law."""
        if self.is_additive:
            return max(MIN_DEADLINE_FULFILMENT_S, float(phi))
        return min(self.phi_max, max(self.phi_min, float(phi)))

    def next_window(
        self, phi: float, outcome: MissionOutcome, *, answered: bool = False,
    ) -> float:
        """Φ after one outcome.

        Under the multiplicative law a miss relaxes by β_partial when the
        device was reachable — a PARTIAL, or a TIMEOUT after its advert
        arrived (``answered``, e.g. an uplink dropped mid-session) — and by
        β_timeout when it never answered. The step scales the clamped window,
        the Φ the deadline actually used, not the raw stored value.
        """
        if self.is_additive:
            if outcome is MissionOutcome.CLEAN:
                return max(MIN_DEADLINE_FULFILMENT_S, phi - FAST_PHASE_ON_TIME_SHRINK_S)
            return phi + FAST_PHASE_MISSED_WIDEN_S
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
        """Everything but the form, for ``MuleConfig.deadline_params``."""
        raw = asdict(self)
        raw.pop("form")
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
    fulfilment = effective_window(state, law=law)
    return now + fulfilment * window_scale - compute_idle_time(state, now)


def effective_window(
    state: DeviceSchedulerState, *, law: Optional[DeadlineLaw] = None,
) -> float:
    """The fulfilment window Φ the deadline uses, before any S3c scaling.

    The recorded 5 s floor under the additive law (or none), the law's clamps
    under the multiplicative one. The merge cutoff of decision D5 reads the
    same value, so the deadline and the cutoff never disagree about Φ.
    """
    if law is None or law.is_additive:
        return max(MIN_DEADLINE_FULFILMENT_S, state.deadline_fulfilment_s)
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
                MIN_DEADLINE_FULFILMENT_S,
                state.deadline_fulfilment_s - FAST_PHASE_ON_TIME_SHRINK_S,
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
                state.deadline_fulfilment_s + FAST_PHASE_MISSED_WIDEN_S
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
) -> None:
    """Apply ``deadline_overrides`` + relevant ``registry_deltas`` to the map.

    Only devices the scheduler already tracks are touched — slice
    membership (S1) is responsible for admitting new devices, not the
    amendment fold.

    ``registry_deltas`` fields supported here:
        * ``last_known_position`` — tuple[float, float, float]
        * ``deadline_fulfilment_s`` — float override from cluster, clamped
          to the ``law``'s bounds (the 5 s floor alone when it is additive)
    Anything else is ignored; the full registry row lives in
    ``HFLHostCluster``, not in scheduler state.
    """
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
                        MIN_DEADLINE_FULFILMENT_S, float(val)
                    )
                else:
                    st.deadline_fulfilment_s = law.clamp(float(val))
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
