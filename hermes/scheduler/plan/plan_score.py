"""FeRRy Phase 4: the plan score V and its coverage weights (unit U2).

**Why a score.** In plan mode the mule chooses its band class b̄ and its
route π together, once per mission at the dock (build plan L822-847). The
search (U4, ``plan/plan_search.py``) ranks every admitted candidate by the
plan key, the age cap's key first, then (by default) the served weight
share, then V (:func:`plan_key`). This module prices V, the plan's extended
bound (L830), in the form fixed on 2026-09-30 (Phase 4 spec, other choices 7,
with the user's decisions 2 and 3), and states the rank::

    V(b̄, π | demand) = −[c₁(Δ/T)² + c₂U + c₃L] − c₄E/(P_hover·T)
    U     = 1 − Σ_served w / Σ_demand w
    L     = Σ_served w·p_out / Σ_demand w
    p_out = Φ((floor − SNR_b̄(d)) / σ_eff),   σ_eff = √(σ_sh² + σ_I² + A²/2)

* **Δ is the whole mission on b̄** (decision 2 (b)): Pass 1 (transit, dwell at
  the class's predicted rate, the return leg and the upload), the dock
  turnaround, and Pass 2, which flies b̄ too. FedEx's Δ_k is a round trip as
  well (``policies/fedex_carp.py``, eq. 9), and the band sets Pass 2's length:
  at 1 MB and N = 6 its median is 94 s on wide, 62 s on medium and 45 s on
  narrow (critic C2). A plan that serves nobody flies neither pass, since a
  mission that collects nothing skips Pass 2 (``MuleSupervisor``'s empty
  round), but it still pays the turnaround. T is the cell's nominal mission
  period T_nom (``t_nom_s``), which plan mode requires.
* **U is the coverage shortfall**, weighted by :func:`coverage_weight`. The
  demand is the device set left after S1 and S3; the served devices are the
  members of the plan's stops, some of them reduced to a subset.
* **L is the expected link loss**, the design's link option (ii): a served
  member at planar distance d misses its contact with the probability that
  its SNR falls below the floor, around the mean SNR the planner prices
  (δ_obs = 0, design D-M) with the channel's spread (:func:`sigma_eff_db`). It
  does not depend on the arrival time: the planner prices the mean only, and
  pricing the channel's seeded phase would be an oracle (option (iii)).
* **E is the simulated energy of both passes**, return legs included; over
  P_hover·T it reads as a share of a mission spent hovering.

With c₃ = c₂, the default, the coverage and link terms add up to c₂(1 − the
expected weighted served share). Whenever c₃ ≤ c₂, serving one more device at
the same Δ and E never lowers V: it gains (w_j / Σ_demand w)(c₂ − c₃·p_out,j).
With c₂ > 0 the gain is positive unless c₃ = c₂ and the device is certain to
be in outage, where it is 0. With c₂ = 0, and so c₃ = 0 (F-cov), it is 0 for
every device whatever its outage: V does not see coverage at all. That is the
plan's "V falls as coverage falls" (L842) as the design states it, with Δ and
E held and c₂ > 0 (design §4), since serving fewer devices usually shortens Δ
too (design §0.4 item 16).

**Declared, not derived.** The repo holds no derivation from the theory track
(L898-899), so V is hand-set in FedEx's form with the convex-surrogate caveat
(L1198). FedEx-Async's Theorem 2 (eq. 24) bounds the error by a term in
Σ_k R_k·Δ_k², for clients that train without pause between visits; a HERMES
device trains once per visit, so Δ² here is a convex surrogate for staleness,
not a bound. With one mule the sum is Δ² itself, so the exponent matters only
in the trade against coverage and energy. The constants
(``PlanScoreParams``, resolved per demand) are hand-set and swept:

* c₁ = 1;
* c₂ = κ·N_demand with κ = 1: one average device is worth one full T² of
  time, so the plan serves every device the budget allows and time breaks
  ties (decision 2);
* c₃ = c₂;
* c₄ = 0.1: E is nearly collinear with Δ (143.6 W flying at 5 m/s, 168.5 W
  hovering: the Zeng-Xu-Zhang 2019 model, build plan L925), so energy breaks
  ties (design D-B).

The pilot sweeps κ over :data:`PILOT_KAPPAS` and c₄ over
:data:`PILOT_C_ENERGIES`, reporting the plans that fly empty.

**Coverage weights** (decision 3). ``age`` weighs a demanded device by its age
a_j, its own mule's missions since its last merged update (U1,
``stages/s3d_age_cap.py``); ``uniform`` weighs it by 1, the plan's letter
(1 − served/N, L830). Either is multiplied by (1 + the device's miss streak)
when the arm's ``miss_priority`` is on. Every device a plan leaves out is
widened as a miss and a clean contact clears the streak, so under a plan the
streak tracks the age (m_j = a_j − 1) and F's weight is about a_j²: F's
objective is declared quadratic in age (critic A5). F-prio, with
``miss_priority`` off, weighs by age alone, so its 1 − U is the share of the
Network AoU the mission removes (design D-A). F-cov (c₂ = c₃ = 0) serves
capped devices only, since the empty plan otherwise scores best: it is
reported as "cap-only service".

**The rank** (the orchestrator's resolution R11 of 2026-09-30, on decision
2). κ = 1 is meant to "serve everyone the budget allows; time only breaks
ties", but V alone does not keep that promise: every plan that serves anyone
pays a whole Pass 2 on its class and the empty plan pays none, so V can
prefer flying empty to serving a device that fits. U7's probe: u and v 60 m
either side of the dock, z 400 m out, FB+wide at 1 MB, a 37.6 s budget and
T = 200 s, the cap off; serving v scores −3.85 (a predicted 264 s mission,
its Pass 2 flying out to z), the empty plan −3.02 (its 30 s turnaround), and
with the weights growing together it flew empty mission after mission. So
``coverage_rank`` (``PlanScoreParams``) chooses how the search ranks the
candidates that tie on the cap key:

* ``lexicographic``, the default: the served weight share Σ_served w /
  Σ_demand w (:func:`served_share`) first, then V. Every demanded device
  weighs more than 0, so the empty plan wins only when no plan that serves
  anyone is admitted, and time (with energy and the link) breaks ties among
  plans that serve the same weight;
* ``weighted``: V alone, trading coverage against time at the rate κ sets.
  The pilot's κ sweep flies it; at κ = 1 it can fly empty as above.

With the coverage term off (κ = 0, arm F-cov) V alone ranks whatever the
setting says, since a share-first rank would serve everyone the budget
allows and F-cov is cap-only service (decision 3; :func:`applied_rank`). The
rank never changes V, its terms or the predicted mission: :func:`plan_key`
reads them.

``dwell_in_delta=False`` (arm F-dwell, Study 5.7) takes the dwell of both
passes out of Δ in the score only: the predicate still prices it, and E still
counts the hovering it costs. ``ScoreTerms.delta_s`` is the Δ that V prices,
so under F-dwell it is not the mission the mule is predicted to fly. The
prediction the trace sets against the realized ledger (design D-M) is
:func:`predicted_mission_s`, the same for every arm, which the commit records
under the extra score key :data:`MISSION_SCORE_KEY`.

**Primitive inputs** (critic B13). :func:`score` takes floats keyed by device,
never U3's fold or U6's physics, so the units build in parallel without an
import cycle. The planner derives them: Pass 1 from the member fold, each
served member's outage from ``PlanClass.outage`` at its
``FerryPhysics.member_distances_m`` distance, and Pass 2 once per class, as
T_nom prices it (``fl_scheduler.nominal_mission_period_s``).

**The outage formula has two definitions.** The ``PlanClass.outage`` the
planner calls is the mule runtime's own, ``FerryRuntime.outage_probability``
bound to the class (U6, ``hermes/mule/ferry.py``, through
``statistics.NormalDist``). This module's :func:`outage_by_distance` is the
score's statement of the same formula, and nothing in the planner calls it.
The U2 tests tie the two together: they agree within 1e-15 for every class of
both class sets, in both contact regimes and on a noise-free channel.

Numpy-free: this module imports the standard library and the plan types
only, and nothing from ``hermes.l1``, ``hermes.mule``, ``experiments``, the
policies or the scheduler. Freeze Rule 1: only the plan path
(``plan_mode = "ferry"``) calls it, and the legacy pipeline never imports it.
"""

from __future__ import annotations

import collections.abc
import math
import numbers
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from hermes.types.ids import DeviceID

from .types import (
    COVERAGE_RANK_LEXICOGRAPHIC,
    COVERAGE_RANK_WEIGHTED,
    COVERAGE_RANKS,
    COVERAGE_WEIGHTS,
    COVERAGE_WEIGHTS_AGE,
    Candidate,
    PlanScoreParams,
    ScoreTerms,
)

__all__ = [
    "PILOT_KAPPAS", "PILOT_C_ENERGIES", "MISSION_SCORE_KEY",
    "COVERAGE_RANK_LEXICOGRAPHIC", "COVERAGE_RANK_WEIGHTED", "COVERAGE_RANKS",
    "sigma_eff_db", "mean_snr_outage", "outage_by_distance",
    "coverage_weight", "demand_weights", "predicted_mission_s", "score",
    "served_share", "applied_rank", "plan_key",
]

#: The pilot's sweep of κ, the coverage constant per demanded device (c₂ =
#: κ·N_demand), and of c₄, the energy constant (Phase 4 spec, decision 2, as
#: the user took it on 2026-09-30). The defaults, κ = 1 and c₄ = 0.1, are in
#: both. Lower κ was probed and left out: with Δ over the whole mission, 2 of
#: 30 plans already fly empty at κ = 0.15 (30 s, 1 MB) and 1 to 9 of 30 at
#: κ = 0.1, which is why the critic's range of 0.1-0.25 (C1) was rejected.
PILOT_KAPPAS: Tuple[float, ...] = (0.15, 0.25, 1.0)
PILOT_C_ENERGIES: Tuple[float, ...] = (0.0, 0.1)

#: The key under which a commit's ``score`` records :func:`predicted_mission_s`
#: beside ``PLAN_SCORE_KEYS`` (``PlanCommit`` takes any other finite term the
#: planner adds), so that ``mission_completed.plan.score`` carries the
#: prediction design D-M compares with the realized ledger, for every arm.
MISSION_SCORE_KEY = "mission_s"

# A dwell summed apart from the pass it belongs to may exceed that pass's
# time by rounding alone (both are sums of the same legs); anything larger
# is a caller's error.
_DWELL_SLACK = 1e-9


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _real(value: Any, name: str) -> float:
    """``value`` as a finite float; a bool is refused."""
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


def _count(value: Any, name: str) -> int:
    """``value`` as an int >= 0; a bool is refused (``True`` is no count)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return int(value)


def _device(value: Any, name: str) -> DeviceID:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} holds device ids (non-empty strings), got {value!r}")
    return value  # type: ignore[return-value]


def _mode(value: Any) -> str:
    if not isinstance(value, str) or value not in COVERAGE_WEIGHTS:
        raise ValueError(f"mode must be one of {COVERAGE_WEIGHTS}, got {value!r}")
    return value


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return value


def _mapping(value: Any, name: str) -> Mapping[Any, Any]:
    if not isinstance(value, collections.abc.Mapping):
        raise TypeError(f"{name} must be a mapping keyed by device id, got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# The link term: mean-SNR outage (design link option (ii))
# --------------------------------------------------------------------------- #

def sigma_eff_db(
    shadow_sigma_db: float, interference_sigma_db: float, interference_amp_db: float,
) -> float:
    """σ_eff = √(σ_sh² + σ_I² + A²/2), in dB: how far a member's SNR strays from its mean.

    The contact channel's realized SNR is the planner's mean SNR plus the
    shadowing X_j(t) of the member's radio link (σ_sh), the band's
    interference wave A·sin(·) and the interference noise (σ_I), added
    (``ContactChannel.snr_db``,
    ``hermes/l1/channel_model.py``). The three are independent and a sine of
    uniform phase has variance A²/2, so σ_eff² is the variance of their sum.
    :func:`mean_snr_outage` treats the sum as one Gaussian of that spread: a
    moment match, exact for the shadowing and the noise but not for the wave.
    All three are the contact channel's (as ``FerryRuntime.outage_probability``
    reads them): its σ_sh is the link's in every channel ``from_link`` builds,
    and 0 on critic B4's noise-free channel, whose outage is then the step.
    """
    shadow = _nonneg(shadow_sigma_db, "shadow_sigma_db")
    noise = _nonneg(interference_sigma_db, "interference_sigma_db")
    amp = _nonneg(interference_amp_db, "interference_amp_db")
    return math.sqrt(shadow * shadow + noise * noise + amp * amp / 2.0)


def mean_snr_outage(mean_snr_db: float, *, snr_floor_db: float, sigma_db: float) -> float:
    """p_out = Φ((floor − mean) / σ): the chance a member's SNR is below the floor.

    ``mean_snr_db`` is the class's mean SNR at the member's planar distance,
    the SNR the planner prices dwell at (δ_obs = 0, design D-M), and
    ``sigma_db`` is :func:`sigma_eff_db`. At a class's edge R(b) the mean SNR
    sits the shadowing margin Φ⁻¹(0.9)·σ_sh above the floor
    (``ContactLink.range_m``), so with the shadowing alone the outage there is
    0.1; the interference raises it. Φ is evaluated as erfc(−x/√2)/2, which
    stays accurate in both tails. With σ = 0, a channel without spread, the
    outage is the limit: 1 below the floor, 0 at or above it, because the
    floor is inclusive (``ContactLink.above_floor``).
    """
    mean = _real(mean_snr_db, "mean_snr_db")
    floor = _real(snr_floor_db, "snr_floor_db")
    sigma = _nonneg(sigma_db, "sigma_db")
    if sigma == 0.0:
        return 1.0 if mean < floor else 0.0
    return 0.5 * math.erfc((mean - floor) / (sigma * math.sqrt(2.0)))


def outage_by_distance(
    mean_snr_db: Callable[[float], float],
    *,
    snr_floor_db: float,
    sigma_db: float,
    range_m: Optional[float] = None,
) -> Callable[[float], float]:
    """One class's outage as a function of a member's planar distance.

    The score's statement of what ``PlanClass.outage`` computes. The planner
    calls the mule runtime's own copy, ``FerryRuntime.outage_probability``
    bound to the class (U6), and not this function; the U2 tests hold the two
    within 1e-15 of each other (:func:`mean_snr_outage` evaluates Φ through
    erfc, the runtime through ``NormalDist``'s erf, so the last bits differ).
    Bound as the runtime reads its physics, ``mean_snr_db`` is
    ``lambda d: link.mean_snr_db(band, d)``, ``snr_floor_db`` is
    ``link.snr_floor_db``, ``sigma_db`` is :func:`sigma_eff_db` of the
    contact channel's ``shadow_sigma_db``, ``interference_sigma_db`` and
    ``interference_amp_db``, and ``range_m`` is ``link.range_planar_m(band)``.
    The runtime could bind this function so that the formula has one
    definition; the scheduler would still reach the physics as a callable,
    never by importing ``hermes.l1``. ``range_m`` is the contact gate's range,
    R_planar(b) (inclusive, as ``ContactLink.in_range``): a member farther
    away is never solicited, so its outage is 1. S3a places every member
    within it, so only a stale position reaches that case; None leaves it out.
    """
    if not callable(mean_snr_db):
        raise TypeError("mean_snr_db must be a callable of the planar distance")
    floor = _real(snr_floor_db, "snr_floor_db")
    sigma = _nonneg(sigma_db, "sigma_db")
    reach = None if range_m is None else _real(range_m, "range_m")
    if reach is not None and reach <= 0.0:
        raise ValueError(f"range_m must be > 0 or None, got {range_m!r}")

    def outage(d_planar: float) -> float:
        d = _nonneg(d_planar, "d_planar")
        if reach is not None and d > reach:
            return 1.0
        return mean_snr_outage(mean_snr_db(d), snr_floor_db=floor, sigma_db=sigma)

    return outage


# --------------------------------------------------------------------------- #
# Coverage weights (decision 3)
# --------------------------------------------------------------------------- #

def coverage_weight(
    age: Optional[int], miss_streak: int, *, miss_priority: bool, mode: str,
) -> float:
    """One demanded device's coverage weight w_j (decision 3), always >= 1.

    * ``age``: max(a_j, 1), times (1 + m_j) with ``miss_priority`` on (arm F,
      about a_j², critic A5), alone with it off (F-prio);
    * ``uniform``: 1, times (1 + m_j) with ``miss_priority`` on; ``age`` is
      then not read and may be None.

    a_j is U1's age of the device at planning time
    (``stages/s3d_age_cap.device_age``): m − (``last_merged_round`` or 0) for
    the mission m being planned, its own mule's missions since its last
    merged update (Phase 4 spec, decision 1), unclamped, as the scorer counts
    it. The mule numbers missions from 1 (``HFLHostMission.open_round``) and
    records a merge only after the mission's Pass 1
    (``FLScheduler.record_merged``), so a_j >= 1 whenever the mule plans. Age
    0 means a merge in the mission being planned or a round-0 caller off the
    mule's path; the weight, not the age, floors it at 1 (as the design's
    ``aou_age`` did), so such a device weighs as the freshest one and never
    0: every demanded device counts in U. The cap reads the age unfloored.
    m_j is ``DeviceSchedulerState.miss_streak``.
    """
    mode = _mode(mode)
    miss_priority = _flag(miss_priority, "miss_priority")
    streak = _count(miss_streak, "miss_streak")
    if mode == COVERAGE_WEIGHTS_AGE:
        if age is None:
            raise ValueError("age weights need the device's age (U1's CapState.ages)")
        base = float(max(_count(age, "age"), 1))
    else:
        base = 1.0
    return base * (1 + streak) if miss_priority else base


#: The smallest speed factor: Oort's penalty is 0 for a device that never
#: finishes, and a coverage weight must stay positive (:func:`score`).
MIN_SPEED_FACTOR = 1e-12


def oort_speed_factors(
    demand: Iterable[DeviceID],
    *,
    fit_s: Optional[Callable[[DeviceID], Optional[float]]],
    dwell_s: float,
    t_ref_s: float,
    alpha: float,
) -> Dict[DeviceID, float]:
    """Unit U11 (arm F-pref): Oort's system-speed factor for each demanded device.

    ``(T / t_j) ** alpha`` when ``t_j > T``, else 1 (Lai et al., OSDI 2021; the
    D2 port's ``speed_penalty``), with ``T = t_ref_s`` (the cell's T_nom) and
    ``t_j`` the device's round time as D2 reads it: its fit time (``fit_s``;
    0 without training times, or with no clock) plus its predicted dwell,
    here at its own position on the reference class (``dwell_s``: the plan
    weighs devices before it groups them into stops). A device that never
    finishes gets :data:`MIN_SPEED_FACTOR`, so its weight stays positive.
    """
    out: Dict[DeviceID, float] = {}
    t_ref = float(t_ref_s)
    for did in demand:
        fit = 0.0 if fit_s is None else float(fit_s(did) or 0.0)
        t = fit + float(dwell_s)
        if t <= t_ref:
            out[did] = 1.0
        elif math.isinf(t):
            out[did] = MIN_SPEED_FACTOR
        else:
            out[did] = max(MIN_SPEED_FACTOR, (t_ref / t) ** float(alpha))
    return out


def demand_weights(
    demand: Iterable[DeviceID],
    device_states: Mapping[DeviceID, Any],
    *,
    ages: Optional[Mapping[DeviceID, int]],
    miss_priority: bool,
    mode: str,
    speed: Optional[Any] = None,
) -> Dict[DeviceID, float]:
    """Every demanded device's :func:`coverage_weight`, in demand order.

    ``device_states`` is the scheduler's device-state map, whose entries'
    ``miss_streak`` is read. ``ages`` are U1's (``CapState.ages``): the
    ``age`` mode needs one for every demanded device, whether the cap is on or
    off (so an empty demand needs none), and ``uniform`` reads none.
    ``miss_priority`` is the arm's (``FLScheduler.miss_priority``) and
    ``mode`` is ``PlanScoreParams.coverage_weights``. The result is both the
    commit's ``weights`` and :func:`score`'s.

    ``speed`` (the Exp 5 addendum's unit U11, arm F-pref after Oort): None, the
    default, leaves every weight as above; else a mapping of device to factor
    in (0, 1], or a callable of the demand that returns one, and each weight is
    multiplied by its device's factor (1 for a device it leaves out).
    """
    if isinstance(demand, str):
        raise TypeError(f"demand is a collection of device ids, not one string: {demand!r}")
    devices = tuple(_device(d, "demand") for d in demand)
    if len(set(devices)) != len(devices):
        raise ValueError(f"demand lists each device once, got {devices!r}")
    states = _mapping(device_states, "device_states")
    mode = _mode(mode)
    miss_priority = _flag(miss_priority, "miss_priority")
    if mode == COVERAGE_WEIGHTS_AGE and devices:
        if ages is None:
            raise ValueError("age weights need the demanded devices' ages (U1's CapState.ages)")
        ages = _mapping(ages, "ages")
        missing = [d for d in devices if d not in ages]
        if missing:
            raise ValueError(
                f"age weights need an age for every demanded device; missing {missing}"
            )
    out: Dict[DeviceID, float] = {}
    for did in devices:
        try:
            state = states[did]
        except KeyError:
            raise ValueError(f"no device state for the demanded device {did!r}") from None
        if not hasattr(state, "miss_streak"):
            raise TypeError(f"the device state of {did!r} carries no miss_streak: {state!r}")
        age = ages[did] if mode == COVERAGE_WEIGHTS_AGE else None  # type: ignore[index]
        out[did] = coverage_weight(
            age, state.miss_streak, miss_priority=miss_priority, mode=mode,
        )
    if speed is not None:
        factors = speed(devices) if callable(speed) else speed
        for did in devices:
            f = float(factors.get(did, 1.0))
            if not (math.isfinite(f) and 0.0 < f <= 1.0):
                raise ValueError(f"a speed factor is in (0, 1], got {f!r} for {did!r}")
            out[did] *= f
    return out


# --------------------------------------------------------------------------- #
# The score
# --------------------------------------------------------------------------- #

def _mission_s(flown: bool, pass_1_s: float, turnaround_s: float, pass_2_s: float) -> float:
    """Pass 1 + turnaround + Pass 2, Pass 2 only when flown: one rule for Δ and the prediction."""
    return pass_1_s + turnaround_s + (pass_2_s if flown else 0.0)


def predicted_mission_s(
    *, serves_any: bool, pass_1_s: float, turnaround_s: float, pass_2_s: float,
) -> float:
    """The mission a plan is predicted to fly on b̄, in simulated seconds, for every arm.

    Pass 1, the dock turnaround and Pass 2, the last only when the plan serves
    anyone (``serves_any``), since a mission that collects nothing skips Pass
    2: V's Δ with the dwell in (decision 2 (b)), from the primitives
    :func:`score` takes, by the same rule, so the two agree bit for bit
    whenever ``dwell_in_delta`` is True. Under F-dwell, ``ScoreTerms.delta_s``
    leaves both passes' dwell out, which on narrow is most of a mission (one
    1 MB narrow plan in the U2 review: 141 s predicted, 36 s priced). This is
    the prediction design D-M sets against the realized ledger: the scheduler
    (U5) records it in the commit's ``score`` under :data:`MISSION_SCORE_KEY`.
    """
    flown = _flag(serves_any, "serves_any")
    return _mission_s(
        flown, _nonneg(pass_1_s, "pass_1_s"), _nonneg(turnaround_s, "turnaround_s"),
        _nonneg(pass_2_s, "pass_2_s"),
    )


def score(
    params: PlanScoreParams,
    *,
    weights: Mapping[DeviceID, float],
    served_outage: Mapping[DeviceID, float],
    pass_1_s: float,
    pass_1_dwell_s: float,
    pass_1_energy_j: float,
    turnaround_s: float,
    pass_2_s: float,
    pass_2_dwell_s: float,
    pass_2_energy_j: float,
    t_ref_s: float,
    p_hover_w: float,
) -> ScoreTerms:
    """V of one candidate plan (b̄, π), with its terms.

    Every input is a primitive (critic B13):

    * ``weights``: the demand, each device with its coverage weight, all
      > 0 (:func:`demand_weights`); N_demand = ``len(weights)`` resolves c₂
      and c₃ (``PlanScoreParams.constants``).
    * ``served_outage``: each device the plan serves (the members of its
      stops) with its outage p_out on b̄: ``PlanClass.outage`` at the member's
      planar distance (``FerryPhysics.member_distances_m``). A subset of the
      demand; empty for the empty plan.
    * Pass 1 on b̄ from takeoff: ``pass_1_s``, the member fold's home less the
      takeoff time (return leg and upload included; 0 for the empty plan at
      the dock, where the fold adds neither); ``pass_1_dwell_s``, the stops'
      predicted dwell summed; ``pass_1_energy_j``, the fold's energy with the
      return leg (``MemberFold.energy_j``).
    * ``turnaround_s``: the dock turnaround (``PlanSetup.turnaround_s``).
    * Pass 2 on b̄, once per class: the whole slice at R(b̄), nearest first,
      without a budget, DELIVER bytes, from the dock back to the dock, as
      ``fl_scheduler.nominal_mission_period_s`` prices it: its time
      ``pass_2_s``, dwell ``pass_2_dwell_s`` and energy ``pass_2_energy_j``,
      the return leg included. A plan that serves nobody does not fly it, so
      it is not counted then.
    * ``t_ref_s``: T, the cell's T_nom (``PlanSetup.t_ref_s``), > 0.
    * ``p_hover_w``: P_hover (``PlanSetup.p_hover_w``). It may be 0 only with
      c₄ = 0, and the energy term is then left at 0, as it cannot be
      normalised.

    Then, in mission order, Δ = pass_1_s + turnaround_s + pass_2_s, less both
    passes' dwell under ``dwell_in_delta=False`` (never below 0), and E =
    pass_1_energy_j + pass_2_energy_j, each Pass-2 part only when something
    is served; U = 1 − Σ_served w / Σ_demand w and L = Σ_served w·p_out /
    Σ_demand w, both 0 for an empty demand, which forgoes nothing; and V =
    −(c₁(Δ/T)² + c₂U + c₃L) − c₄E/(P_hover·T). The weight sums are exactly
    rounded (``math.fsum``), so V does not depend on the order of either
    mapping and the same plan priced twice ties bit for bit. ``delta_s`` is
    the Δ V prices, so under ``dwell_in_delta=False`` it is not the predicted
    mission; :func:`predicted_mission_s` is, for every arm.
    """
    if not isinstance(params, PlanScoreParams):
        raise TypeError(f"params must be PlanScoreParams, got {params!r}")
    demand: Dict[DeviceID, float] = {}
    for did, w in _mapping(weights, "weights").items():
        w = _real(w, f"weights[{did!r}]")
        if w <= 0.0:
            raise ValueError(
                f"weights[{did!r}] must be > 0, got {w!r}: every demanded device "
                f"counts in the coverage term"
            )
        demand[_device(did, "weights")] = w
    served: Dict[DeviceID, float] = {}
    for did, p in _mapping(served_outage, "served_outage").items():
        did = _device(did, "served_outage")
        if did not in demand:
            raise ValueError(f"the plan serves {did!r}, which is not in the demand")
        p = _real(p, f"served_outage[{did!r}]")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"served_outage[{did!r}] is a probability, got {p!r}")
        served[did] = p

    p1 = _nonneg(pass_1_s, "pass_1_s")
    d1 = _nonneg(pass_1_dwell_s, "pass_1_dwell_s")
    e1 = _nonneg(pass_1_energy_j, "pass_1_energy_j")
    turn = _nonneg(turnaround_s, "turnaround_s")
    p2 = _nonneg(pass_2_s, "pass_2_s")
    d2 = _nonneg(pass_2_dwell_s, "pass_2_dwell_s")
    e2 = _nonneg(pass_2_energy_j, "pass_2_energy_j")
    for dwell, total, name in ((d1, p1, "pass_1"), (d2, p2, "pass_2")):
        if dwell > total + _DWELL_SLACK * max(1.0, total):
            raise ValueError(
                f"{name}_dwell_s ({dwell!r}) exceeds {name}_s ({total!r}): the dwell is "
                f"part of the pass"
            )
    t_ref = _real(t_ref_s, "t_ref_s")
    if t_ref <= 0.0:
        raise ValueError(f"t_ref_s must be > 0, got {t_ref_s!r}")
    p_hover = _nonneg(p_hover_w, "p_hover_w")
    c1, c2, c3, c4 = params.constants(len(demand))
    if p_hover == 0.0 and c4 > 0.0:
        raise ValueError("the energy term E/(P_hover·T) needs P_hover > 0 when c_energy > 0")

    flown = bool(served)
    delta = _mission_s(flown, p1, turn, p2)
    energy_j = e1 + (e2 if flown else 0.0)
    if not params.dwell_in_delta:
        delta = max(0.0, delta - (d1 + (d2 if flown else 0.0)))
    time = (delta / t_ref) ** 2
    energy = energy_j / (p_hover * t_ref) if p_hover > 0.0 else 0.0

    demand_weight = math.fsum(demand.values())
    served_weight = math.fsum(demand[d] for d in served)
    if demand_weight > 0.0:
        coverage = 1.0 - served_weight / demand_weight
        link = math.fsum(demand[d] * p for d, p in served.items()) / demand_weight
    else:
        coverage = link = 0.0
    # 0.0 - x rather than -x, so that a plan with nothing to price scores 0.0, not -0.0.
    v = 0.0 - (c1 * time + c2 * coverage + c3 * link) - c4 * energy
    return ScoreTerms(
        v=v, delta_s=delta, time=time, coverage=coverage, link=link, energy_j=energy_j,
        energy=energy, served_weight=served_weight, demand_weight=demand_weight,
    )


# --------------------------------------------------------------------------- #
# The rank (R11)
# --------------------------------------------------------------------------- #

def served_share(terms: ScoreTerms) -> float:
    """The plan's served weight share, Σ_served w / Σ_demand w, which is 1 − U.

    What the ``lexicographic`` rank compares after the cap key
    (:func:`plan_key`). It is read from the terms' weight sums, which
    :func:`score` rounds exactly, so a served set has one share however its
    plan was priced. Every demanded device weighs more than 0, so the share
    is 0 for the empty plan, above 0 for any plan that serves anyone and 1 for
    a plan that serves the whole demand. An empty demand forgoes nothing (U =
    0), so its share is 1.
    """
    if not isinstance(terms, ScoreTerms):
        raise TypeError(f"terms must be ScoreTerms, got {terms!r}")
    if terms.demand_weight > 0.0:
        return terms.served_weight / terms.demand_weight
    return 1.0


def applied_rank(params: PlanScoreParams) -> str:
    """The rank the search applies under ``params``: :data:`COVERAGE_RANKS`.

    ``params.coverage_rank``, except that with the coverage term off (κ = 0,
    arm F-cov) it is ``weighted`` whatever the setting says. A share-first
    rank would make F-cov serve everyone the budget allows, while F-cov is
    the plan's letter with coverage out of the objective: V then prices time
    and energy only, so the empty plan wins unless the cap's key forces
    someone in ("cap-only service", decision 3). The search records this
    value in each class's summary, so a trace shows which rank chose.
    """
    if not isinstance(params, PlanScoreParams):
        raise TypeError(f"params must be PlanScoreParams, got {params!r}")
    if params.c_cov_per_device == 0.0:
        return COVERAGE_RANK_WEIGHTED
    return params.coverage_rank


def plan_key(params: PlanScoreParams, candidate: Candidate) -> Tuple[Any, ...]:
    """The key the search ranks ``candidate`` by under ``params``: the smallest wins.

    * ``lexicographic``: (cap key, −round(share, 9), −round(V, 9), class
      index, each stop's (position, devices)), the share being
      :func:`served_share`;
    * ``weighted``, and F-cov under either setting (:func:`applied_rank`):
      (cap key, −round(V, 9), class index, each stop's (position, devices)),
      U0's ``Candidate.key`` itself.

    The cap key comes first under both, so a plan that leaves an older
    capped device out loses to any that keeps it (critic C5). The share and
    V are rounded to 9 decimals so that the same plan priced along different
    float paths ties, and a tie falls to the class and then the stops: a
    total order, the same in every search mode and for every class, so the
    pick never depends on the order candidates are met and F's key is never
    above any FB+c's (critic A3).
    """
    if not isinstance(candidate, Candidate):
        raise TypeError(f"candidate must be a Candidate, got {candidate!r}")
    key = candidate.key
    if applied_rank(params) == COVERAGE_RANK_WEIGHTED:
        return key
    return (candidate.cap_key, -round(served_share(candidate.terms), 9)) + key[1:]
