"""Whittle index over Age of Update — SOTA baseline (arm D3).

Ports the training-stage scheduler of **Cui et al.**: Z. Cui, T. Yang, X. Wu,
H. Feng and B. Hu, *The Data Value Based Asynchronous Federated Learning for
UAV Swarm Under Unstable Communication Scenarios*, IEEE Transactions on Mobile
Computing, vol. 23, no. 6, pp. 7165-7179, June 2024 (DOI
10.1109/TMC.2023.3331906). Sec. V-B, Theorem 8, eq. (48), p. 7173; Algorithm 2,
p. 7174.

**What the paper does.** A leader UAV serves ONE follower per iteration. Each
follower is a restless arm with state ``(AoU, Λ)``: ``AoU`` is its Age of
Update, reset to 1 when it is selected and +1 otherwise (eq. 33, initialised to
1 by Algorithm 2 line 2), and ``Λ ∈ {0, 1}`` says whether its link to the
leader is up this iteration, i.i.d. Bernoulli(ρ_i). The objective is the
long-run weighted Network AoU ``Σ_i ω_i AoU_i`` (eq. 34), ω_i the normalised
Shapley data value (eq. 35). A Lagrangian relaxation decouples the arms and
gives the closed-form index (48)::

    I(x, 0) = 0
    I(x, 1) = ω · [x(x − 1)/2 + x/ρ]          (x = AoU ≥ 1)

and the leader serves the connected UAV with the largest index. What is proved
is threshold-optimality of each relaxed single-arm problem (Thm 6),
indexability (Thm 7) and the closed form (Thm 8); there is no optimality
theorem for the coupled N-arm policy. Call it *Cui's Whittle-index policy*
(indexable, closed form, best of the four policies in their simulations), not
"the optimal policy".

**Deviations of this port, each forced by the mule and each stated so the
paper can state it too:**

1. **Λ is not observed at planning time — the central gap.** Cui's leader sees
   who is connected *before* it chooses; a mule commits flight time before it
   learns whether a device answers. Two variants, chosen by ``variant``:

   * ``'expected'`` (default, NOT in the paper — derived here): the index in
     expectation over the unobserved state, Λ ~ Bernoulli(ρ)::

         W(x) = E_Λ[I(x, Λ)] = ρ·I(x, 1) + (1 − ρ)·0 = ω · [ρ·x(x − 1)/2 + x]

     The same formula is also the Whittle index of the *commit-before-observe*
     version of Cui's single-arm MDP (activate before Λ is revealed, charge the
     activation cost C' on every attempt): under a threshold policy the
     post-action AoU chain (43) is unchanged, attempts happen at rate π_1/ρ
     (only a fraction ρ of them reset the age), so C' enters the average cost
     (38) as C'/ρ, and solving the indifference condition (49)
     φ(x) = φ(x + 1) gives C' = ρ·I(x, 1). Derived for this port and checked
     analytically and by Monte Carlo (x = 3, ω = 0.18, ρ = 0.5 gives W = 0.81
     and φ(3) = φ(4) = 0.9); the unit tests pin the indifference condition.
     Threshold-optimality and indexability of that variant are NOT proven, so
     W is a derived heuristic, not a theorem of the paper. Effect: at equal
     age, reliable devices rank higher, because ρ̂ damps the quadratic term.
     The pressure toward a dead device is weaker than under ``'literal'``,
     but it is NOT bounded: W ≥ ω·x for every ρ̂, so age alone keeps it
     rising. Worked case, a device attempted every mission that never
     answers (x = k + 1 after k attempts, ρ̂ = 1/(k + 2)): W is already 2.33ω
     after one failed attempt (x = 2), above every device merged last
     mission (W = ω at x = 1). W/x climbs slowly (1.36 at x = 6, 1.45 at
     x = 19), because ρ̂·x(x − 1)/2 ≈ x/2 while ρ̂ keeps falling. Once ρ̂ hits
     ``rho_min`` (x = 19 at 0.05) the term is 0.025·x(x − 1), which is
     quadratic and overtakes the linear term past x ≈ 1 + 2/``rho_min`` = 41
     (W/x = 2.0 there, 3.5 at x = 101). The literal index of the same device
     is already 57ω at x = 6, against W = 8.1ω.
   * ``'literal'``: I(x, 1) for every device, i.e. assume every device is
     connected. Kept for sensitivity only, because it is a poor port: in Cui
     the 1/ρ term rewards seizing a *currently open* link to a flaky UAV; with
     Λ assumed, it just prefers unreliable devices. x/ρ grows as ρ falls, and
     a persistently unreachable device (ρ → ``rho_min``, x rising every
     mission because nothing merges) climbs to the top of every plan and
     becomes a budget sink.

2. **Many devices per mission, bundled into contacts.** Cui selects exactly one
   UAV per iteration; a multi-UAV extension is claimed but not given. The
   standard multi-activation Whittle heuristic activates the arms with the
   largest indices. Our unit of travel is a contact (S3a) and serving it serves
   every member, so a contact's index is the **sum** of its members' indices:
   under the Lagrangian relaxation the per-arm charges are additive, so a bundle
   is worth what its members are worth together. Max would ignore members
   collected for free; a mean would penalise co-located devices.

3. **Admission under a travel budget.** Cui's model charges nothing for
   travel. Contacts are ranked by index descending and admitted with
   :func:`~hermes.scheduler.policies.budget_walk.greedy_budget_walk` under the
   mission budget, pricing travel with the scheduler's own
   :class:`~hermes.scheduler.stages.s3b_feasibility.FeasibilityModel` — the
   same walk and physics arms D1/D2 use. The route is the rank order, not a
   tour. The index is deliberately *not* divided by travel cost: that would be
   a travel-aware extension outside the paper, and D3 is meant to be faithful.

4. **No Λ = 0 exclusion.** Cui never selects a disconnected UAV (index 0).
   With Λ unobserved no device has index 0, so every candidate is admissible
   while budget remains, and with ``mission_deadline_ts=None`` everything is
   admitted, as for D1/D2.

5. **Age from merge rounds, with Cui's off-by-one.** ``x = R − U``, where R is
   the mission being planned (``SelectorEnv.mission_round``) and U the last
   mission whose merge used the device's update (``last_merged_round``, 0 =
   never). That equals the scorer's age + 1: ``traces_scorer.age_profile``
   computes ``m − U`` after mission m's merges, which is 0 right after a merge,
   whereas Cui's (33) resets to 1, and feeding x = 0 would give I(0, 1) = 0,
   tying with a disconnected device. A never-merged device has x = R, so every
   device starts at 1 in mission 1 (Algorithm 2 line 2). A visit whose update
   does not merge leaves x growing, which is Cui's a = 1, Λ = 0 transition.
   Fallbacks while those fields are absent: U ← ``last_clean_round``; R ← 1 +
   the largest ``last_served_round`` over all device states. That is Oort's
   inference, taken over every known device rather than only the candidates,
   because the mission counter is global.

6. **ρ is estimated, from attempts only.** Cui assumes each ρ_i known and
   stationary, and Λ observed for every UAV every iteration. A mule observes Λ
   only for devices it attempts, so ρ̂ = (answered + 1)/(attempts + 2) over
   non-synthetic Pass-1 outcomes (``reach_answered`` / ``reach_attempts``):
   Laplace-smoothed, so a never-attempted device gets 0.5. Λ = 1 means the
   device answered (its advert arrived), i.e. the link was up, whether or not
   its update then merged; a failed merge shows up in x instead. While the
   reach fields are absent the same formula runs over ``on_time_count`` and
   ``on_time_count + missed_count``, which conflates the link with timely
   delivery and counts synthetic misses for devices never attempted, so it
   underestimates ρ. ρ̂ is clamped to ``[rho_min, 1]`` because (48) divides by
   ρ; the default 0.05 is below the lowest ρ Cui simulates (0.1, Fig. 11).
   Reachability in the testbed is not i.i.d. (dead zones, persistent outages),
   so the Bernoulli assumption behind (43) does not hold and the index is a
   plug-in.

7. **ω is not a Shapley value.** Cui's ω comes from Algorithm 1: about M·ρ_i
   sequential visits per device (M = 8000 in Fig. 6), i.e. thousands of tours
   for a mule. ``weights='uniform'`` (default) gives every device 1.0; the
   index is linear in ω, so this ranks exactly as ω = 1/N would.
   ``weights='oort'`` uses Oort's statistical utility
   (:func:`~hermes.scheduler.policies.oort.statistical_utility`) from each
   device's last visit, normalised to mean 1 over the devices that have
   reported. A never-measured device (utility ``inf``, or a non-finite loss)
   gets the mean of the measured ones, 1.0 (and 1.0 if none are measured), so
   ``inf`` never reaches the index. Measured weights are floored at
   :data:`OMEGA_FLOOR` so a device that reported zero loss keeps an age-driven
   rank rather than collapsing to index 0. Semantically this is "how much is
   left to learn" (loss), not an accuracy marginal contribution, and it is
   recomputed every mission where Cui fixes ω after pre-training, which is
   again outside Thms 6-8. Oort's staleness bonus is NOT added: x already
   carries staleness, so it would be counted twice.

8. **Tie-break.** The paper states none. Equal indices are ordered by
   ``(position, devices)``, so the route does not depend on input order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from hermes.types import ContactWaypoint, DeviceID, DeviceSchedulerState

from hermes.scheduler.selector.features import SelectorEnv

from .budget_walk import IN_FLIGHT_BUDGET, greedy_budget_walk
from .oort import statistical_utility

#: Index in expectation over the unobserved connection state (deviation 1).
VARIANT_EXPECTED = "expected"
#: Cui's I(x, 1) with every device assumed connected (deviation 1).
VARIANT_LITERAL = "literal"
VARIANTS = (VARIANT_EXPECTED, VARIANT_LITERAL)

#: ω = 1 for every device (deviation 7).
WEIGHTS_UNIFORM = "uniform"
#: ω from Oort's statistical utility, normalised to mean 1 (deviation 7).
WEIGHTS_OORT = "oort"
WEIGHT_MODES = (WEIGHTS_UNIFORM, WEIGHTS_OORT)

#: Lower clamp on ρ̂. (48) divides by ρ; 0.05 keeps x/ρ within 20·x and sits
#: below the lowest connection probability Cui simulates (0.1).
DEFAULT_RHO_MIN = 0.05

#: Floor on a measured Oort weight (mean-1 scale), so a zero-loss report does
#: not zero the index and tie the device with every other zero-loss device.
OMEGA_FLOOR = 1e-3

#: (x, ω, ρ) — one device's inputs to the index.
DeviceInputs = Tuple[int, float, float]


# --------------------------------------------------------------------------- #
# The index (Cui eq. 48) and its expectation
# --------------------------------------------------------------------------- #

def _check_args(x: float, omega: float, rho: float) -> None:
    # `not (a >= b)` rather than `a < b` so NaN is rejected too: a NaN index
    # would make the sort order depend on input order.
    if not x >= 1:
        raise ValueError(f"AoU x must be >= 1 (Cui eq. 33 resets to 1), got {x!r}")
    if not omega >= 0:
        raise ValueError(f"weight omega must be >= 0, got {omega!r}")
    if not (0 < rho <= 1):
        raise ValueError(f"connection probability rho must be in (0, 1], got {rho!r}")


def whittle_index(x: float, connected: bool, omega: float, rho: float) -> float:
    """Cui eq. (48): ``I(x, Λ)``.

    ``0`` when the device is disconnected (Λ = 0), else
    ``ω · [x(x − 1)/2 + x/ρ]``. Written so exact rationals
    (:class:`fractions.Fraction`) pass through unrounded, which lets the tests
    compare against the paper's hand-worked values exactly.
    """
    _check_args(x, omega, rho)
    if not connected:
        return 0.0
    return x * (x - 1) * omega / 2 + x * omega / rho


def expected_whittle_index(x: float, omega: float, rho: float) -> float:
    """``E_Λ[I(x, Λ)] = ρ·I(x, 1) = ω · [ρ·x(x − 1)/2 + x]`` (deviation 1).

    Not in the paper. It is eq. (48) averaged over Λ ~ Bernoulli(ρ), and also
    the break-even activation cost of Cui's single-arm MDP when the activation
    is committed before Λ is revealed.
    """
    _check_args(x, omega, rho)
    return rho * x * (x - 1) * omega / 2 + x * omega


# --------------------------------------------------------------------------- #
# Per-device inputs, read from scheduler state
# --------------------------------------------------------------------------- #

def _planning_round(
    device_states: Dict[DeviceID, DeviceSchedulerState], env: SelectorEnv,
) -> int:
    """The 1-based mission being planned (deviation 5).

    Prefers ``env.mission_round``. Without it (absent, ``None``, or not a valid
    1-based round), 1 + the latest round in which any known device had an
    outcome: synthetic TIMEOUTs move ``last_served_round`` too, so this keeps
    pace with the mission counter in any mission that planned at least one
    device.
    """
    r = getattr(env, "mission_round", None)
    if r is not None and int(r) >= 1:
        return int(r)
    return 1 + max(
        (int(st.last_served_round) for st in device_states.values()), default=0,
    )


def _last_merged_round(state: DeviceSchedulerState) -> int:
    """U: the last mission whose merge used this device's update (0 = never).

    ``None`` falls back as well as absence: a field that exists but is never
    written by some run would otherwise read as "never merged" for every
    device and inflate every age alike.
    """
    r = getattr(state, "last_merged_round", None)
    if r is None:
        r = state.last_clean_round
    return max(0, int(r))


def connection_probability(
    state: DeviceSchedulerState, *, rho_min: float = DEFAULT_RHO_MIN,
) -> float:
    """ρ̂ = (answered + 1)/(attempts + 2), clamped to ``[rho_min, 1]`` (deviation 6)."""
    attempts = getattr(state, "reach_attempts", None)
    answered = getattr(state, "reach_answered", None)
    if attempts is None or answered is None:
        # Legacy tallies. Biased low (synthetic misses count as attempts), so
        # used only until the reach fields exist.
        answered = state.on_time_count
        attempts = state.on_time_count + state.missed_count
    attempts = max(0, int(attempts))
    answered = min(max(0, int(answered)), attempts)
    rho = (answered + 1) / (attempts + 2)
    return min(1.0, max(float(rho_min), rho))


def _measured_utility(state: DeviceSchedulerState) -> Optional[float]:
    """Oort utility if the device has reported a usable one, else ``None``.

    ``inf`` (never reported) and NaN (a diverged loss) both count as "not
    measured": neither may reach the index.
    """
    u = statistical_utility(state)
    return u if math.isfinite(u) else None


@dataclass(frozen=True)
class _PlanContext:
    """What every device's inputs share within one plan, computed once."""

    mission_round: int
    weights: str
    #: Mean of the measured Oort utilities; ``None`` means "use ω = 1".
    utility_scale: Optional[float]
    rho_min: float

    @classmethod
    def build(
        cls,
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        weights: str,
        rho_min: float,
    ) -> "_PlanContext":
        scale: Optional[float] = None
        if weights == WEIGHTS_OORT:
            measured = [
                u for st in device_states.values()
                if (u := _measured_utility(st)) is not None
            ]
            if measured:
                # fsum: exactly rounded, so the mean does not depend on the
                # order the dict happens to iterate in.
                mean = math.fsum(measured) / len(measured)
                # All-zero utilities carry no ranking information: uniform.
                scale = mean if mean > 0.0 else None
        return cls(
            mission_round=_planning_round(device_states, env),
            weights=weights,
            utility_scale=scale,
            rho_min=float(rho_min),
        )

    def omega(self, state: DeviceSchedulerState) -> float:
        if self.weights == WEIGHTS_UNIFORM or self.utility_scale is None:
            return 1.0
        u = _measured_utility(state)
        if u is None:
            return 1.0          # the mean of the measured weights, on this scale
        return max(OMEGA_FLOOR, u / self.utility_scale)

    def inputs(self, state: DeviceSchedulerState) -> DeviceInputs:
        x = max(1, self.mission_round - _last_merged_round(state))
        return (
            x,
            self.omega(state),
            connection_probability(state, rho_min=self.rho_min),
        )


def _validate(weights: str, rho_min: float) -> None:
    if weights not in WEIGHT_MODES:
        raise ValueError(f"weights must be one of {WEIGHT_MODES}, got {weights!r}")
    if not (0.0 < rho_min <= 1.0):
        raise ValueError(f"rho_min must be in (0, 1], got {rho_min!r}")


def device_inputs(
    state: DeviceSchedulerState,
    device_states: Dict[DeviceID, DeviceSchedulerState],
    env: SelectorEnv,
    *,
    weights: str = WEIGHTS_UNIFORM,
    rho_min: float = DEFAULT_RHO_MIN,
) -> DeviceInputs:
    """``(x, ω, ρ)`` for one device, exactly as :class:`WhittlePolicy` reads them.

    ``device_states`` supplies what is shared across devices: the planning-round
    fallback and the Oort normalisation.
    """
    _validate(weights, rho_min)
    ctx = _PlanContext.build(device_states, env, weights=weights, rho_min=rho_min)
    return ctx.inputs(state)


# --------------------------------------------------------------------------- #
# The policy (arm D3)
# --------------------------------------------------------------------------- #

class WhittlePolicy:
    """Rank contacts by summed Whittle index; admit greedily under the budget.

    A whole-scheduler baseline: :meth:`admit_and_order` replaces S3, S3b and
    S3.5 and returns the Pass-1 route itself, exactly like arms D1/D2.
    """

    name = "WHITTLE"
    # Freeze Amendment 8 — the mule re-checks the mission budget only in
    # flight. The per-device deadline is S3b's rule, which this arm replaces.
    in_flight_check = IN_FLIGHT_BUDGET

    def __init__(
        self,
        variant: str = VARIANT_EXPECTED,
        *,
        weights: str = WEIGHTS_UNIFORM,
        rho_min: float = DEFAULT_RHO_MIN,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        _validate(weights, rho_min)
        self.variant = variant
        self.weights = weights
        self.rho_min = float(rho_min)
        #: Diagnostics only: the (x, ω, ρ) each candidate device was ranked on
        #: in the most recent plan, so a trace can record the ω actually used.
        #: Never read back by the policy.
        self.last_device_inputs: Dict[DeviceID, DeviceInputs] = {}

    def device_index(self, x: float, omega: float, rho: float) -> float:
        """One device's index under this policy's variant."""
        if self.variant == VARIANT_LITERAL:
            return whittle_index(x, True, omega, rho)
        return expected_whittle_index(x, omega, rho)

    def _inputs(
        self,
        contacts: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
    ) -> Dict[DeviceID, DeviceInputs]:
        ctx = _PlanContext.build(
            device_states, env, weights=self.weights, rho_min=self.rho_min,
        )
        inputs: Dict[DeviceID, DeviceInputs] = {}
        for wp in contacts:
            for did in wp.devices:
                if did in inputs:
                    continue
                st = device_states.get(did)
                if st is None:
                    # No history at all: a blank local state reads as never
                    # merged, never attempted, never measured. Not stored back.
                    st = DeviceSchedulerState(device_id=did)
                inputs[did] = ctx.inputs(st)
        return inputs

    def _contact_index(
        self, wp: ContactWaypoint, inputs: Dict[DeviceID, DeviceInputs],
    ) -> float:
        # Sum over members (deviation 2): serving the contact serves them all.
        return math.fsum(self.device_index(*inputs[did]) for did in wp.devices)

    def contact_indices(
        self,
        contacts: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
    ) -> List[float]:
        """Each contact's index, aligned with ``contacts``."""
        inputs = self._inputs(contacts, device_states, env)
        return [self._contact_index(wp, inputs) for wp in contacts]

    def admit_and_order(
        self,
        contacts: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        mission_deadline_ts: Optional[float] = None,
        feasibility_model=None,
    ) -> List[ContactWaypoint]:
        """Cui's index as a **complete scheduler** — it decides *who*, not just order.

        Ranks contacts by summed index, highest first, ``(position, devices)``
        breaking ties, then admits them with the shared greedy budget walk
        under ``mission_deadline_ts`` and ``feasibility_model``. Returns the
        admitted contacts in rank order. Reads ``device_states``; never writes
        them.
        """
        if not contacts:
            self.last_device_inputs = {}
            return []
        inputs = self._inputs(contacts, device_states, env)
        self.last_device_inputs = dict(inputs)

        def _key(wp: ContactWaypoint) -> tuple:
            # Negated -> descending: greedy_budget_walk sorts ascending.
            return (-self._contact_index(wp, inputs), wp.position, wp.devices)

        return greedy_budget_walk(
            contacts,
            key=_key,
            mule_pose=env.mule_pose,
            now=env.now,
            mission_deadline_ts=mission_deadline_ts,
            model=feasibility_model,
        )
