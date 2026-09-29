"""Cui et al.'s Whittle index over Age of Update (arm D3).

What is pinned here:

1. **The index** — eq. (48) of Cui et al. (IEEE TMC 2024) on the paper's
   hand-worked values, exactly, and the indifference condition (49) that
   defines it. The derived ``'expected'`` index is pinned the same way,
   against the commit-before-observe version of the single-arm MDP.
2. **The 4-UAV round** — Cui's selection of B with I = 23/15, reproduced
   first from the formula and then through the policy's own input path.
3. **The port decision** — with Λ unobserved, the ``'literal'`` variant puts a
   flaky device first and the ``'expected'`` variant puts it last.
4. **Inputs and fallbacks** — x, ρ and ω read from state, including the
   fallbacks used until the new state fields land. Those tests hide the new
   fields explicitly, so they keep testing the fallback after the fields
   exist with defaults.
5. **The scheduler contract** — budget-respecting admission with the given
   travel model, everything admitted without a budget, determinism, and no
   mutation of ``device_states``.
"""

from __future__ import annotations

import copy
import math
import random
from fractions import Fraction as F

import pytest

from hermes.scheduler.policies.budget_walk import IN_FLIGHT_BUDGET
from hermes.scheduler.policies.whittle import (
    DEFAULT_RHO_MIN,
    OMEGA_FLOOR,
    WhittlePolicy,
    connection_probability,
    device_inputs,
    expected_whittle_index,
    whittle_index,
)
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
from hermes.types import Bucket, ContactWaypoint, DeviceID, DeviceSchedulerState

NOW = 1_000.0
#: 1 m/s and no session time, so travel seconds == metres.
MODEL = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)
#: State fields the orchestrator adds after this module; the fallback tests
#: hide them.
NEW_STATE_FIELDS = ("reach_attempts", "reach_answered", "last_merged_round")


class _Hidden:
    """Read-only view of ``obj`` with some attributes made absent.

    Stands in for a state or env from before the new fields existed.
    """

    def __init__(self, obj, *hidden: str) -> None:
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_hidden", frozenset(hidden))

    def __getattr__(self, name):
        if name in self._hidden:
            raise AttributeError(name)
        return getattr(self._obj, name)


def _wp(x: float, *devs: str) -> ContactWaypoint:
    return ContactWaypoint(
        position=(x, 0.0, 0.0),
        devices=tuple(DeviceID(d) for d in devs),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=0.0,
    )


def _env(*, mission_round=None, pose=(0.0, 0.0, 0.0), now=NOW) -> SelectorEnv:
    env = SelectorEnv(mule_pose=pose, mule_energy=1.0, rf_prior_snr_db=20.0,
                      beacon_window_s=30.0, now=now)
    if mission_round is not None:
        # Works whether or not SelectorEnv has the field yet (it is frozen).
        object.__setattr__(env, "mission_round", mission_round)
    return env


def _st(did: str, *, merged=None, attempts=None, answered=None,
        clean_round=0, served_round=None, on_time=0, missed=0,
        loss=None, n=0) -> DeviceSchedulerState:
    st = DeviceSchedulerState(
        device_id=DeviceID(did),
        on_time_count=on_time,
        missed_count=missed,
        last_loss=loss,
        last_num_examples=n,
        last_clean_round=clean_round,
        last_served_round=clean_round if served_round is None else served_round,
    )
    # The new fields are set only when a test gives them, and set as plain
    # attributes so this works before and after they become dataclass fields.
    for name, value in (("last_merged_round", merged),
                        ("reach_attempts", attempts),
                        ("reach_answered", answered)):
        if value is not None:
            setattr(st, name, value)
    return st


def _states(*states) -> dict:
    return {st.device_id: st for st in states}


def _ids(route) -> list:
    return [",".join(w.devices) for w in route]


def _route_seconds(route, model=MODEL, pose=(0.0, 0.0, 0.0)) -> float:
    total = 0.0
    for wp in route:
        total += model.cost(pose, wp.position)[1]
        pose = wp.position
    return total


# --------------------------------------------------------------------------- #
# 1. Eq. (48) on the paper's hand-worked values, and condition (49)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("x, omega, rho, expected", [
    (3, F(9, 50), F(1, 2), F(81, 50)),        # 1.62
    (1, F(1, 10), F(9, 10), F(1, 9)),         # = omega / rho at x = 1
    (4, F(1, 4), F(1, 5), F(13, 2)),          # 6.5
    (10, F(1, 8), F(1, 4), F(85, 8)),         # 10.625
])
def test_eq48_matches_the_hand_worked_values_exactly(x, omega, rho, expected):
    assert whittle_index(x, True, omega, rho) == expected
    # And in floats, as the policy calls it.
    assert whittle_index(x, True, float(omega), float(rho)) == pytest.approx(
        float(expected), rel=1e-12)


def test_eq48_is_zero_for_a_disconnected_device():
    assert whittle_index(5, False, 0.3, 0.7) == 0.0


def _phi(X, omega, rho, c_prime, *, per_attempt=False):
    """Cui eq. (38): average cost of the threshold-X policy.

    ``per_attempt`` charges C' on every attempt instead of every success, which
    is the commit-before-observe variant: attempts run at pi_1/rho, so C'
    enters the numerator as C'/rho.
    """
    charge = c_prime / rho if per_attempt else c_prime
    num = (omega * X * X / 2 + omega * (1 / rho - F(1, 2)) * X
           + omega / rho ** 2 - omega / rho + charge)
    return num / (X + (1 - rho) / rho)


GRID = [(x, omega, rho)
        for x in range(1, 13)
        for omega in (F(1, 10), F(9, 50), F(1, 2), F(1))
        for rho in (F(1, 20), F(1, 5), F(1, 2), F(9, 10), F(1))]


def test_eq48_is_the_break_even_cost_of_eq49():
    """The index of state (x, 1) is the C' making thresholds x and x+1 equal."""
    for x, omega, rho in GRID:
        c = whittle_index(x, True, omega, rho)
        assert _phi(x, omega, rho, c) == _phi(x + 1, omega, rho, c), (x, omega, rho)


def test_expected_index_is_the_break_even_cost_when_committing_before_observing():
    for x, omega, rho in GRID:
        c = expected_whittle_index(x, omega, rho)
        assert (_phi(x, omega, rho, c, per_attempt=True)
                == _phi(x + 1, omega, rho, c, per_attempt=True)), (x, omega, rho)


# --------------------------------------------------------------------------- #
# 2. Cui's 4-UAV round (extraction fixture 6)
# --------------------------------------------------------------------------- #

#: UAV: (omega, rho, AoU x, connected Λ).
CUI_ROUND = {
    "A": (F(3, 10), F(9, 10), 2, True),
    "B": (F(1, 5), F(3, 10), 2, True),
    "C": (F(1, 2), F(4, 5), 1, True),
    "D": (F(1, 4), F(1, 5), 4, False),
}


def test_the_four_uav_round_selects_b_with_index_23_over_15():
    idx = {u: whittle_index(x, lam, w, r) for u, (w, r, x, lam) in CUI_ROUND.items()}
    assert idx == {"A": F(29, 30), "B": F(23, 15), "C": F(5, 8), "D": 0}
    assert max(idx, key=idx.get) == "B"
    # The myopic key omega*x would have picked A: the index discriminates.
    myopic = {u: w * x for u, (w, r, x, lam) in CUI_ROUND.items() if lam}
    assert max(myopic, key=myopic.get) == "A"


def _cui_round_states():
    """The round as scheduler state, planning mission 5.

    x = 5 - last_merged_round gives (2, 2, 1, 4). Laplace rho over 8 attempts
    gives (0.9, 0.3, 0.8, 0.2). Oort utilities 100 * loss are proportional to
    Cui's omega, so the mean-1 weights are omega / 0.3125.
    """
    spec = {"A": (3, 8, F(3, 10)), "B": (3, 2, F(1, 5)),
            "C": (4, 7, F(1, 2)), "D": (1, 1, F(1, 4))}
    return _states(*(
        _st(u, merged=merged, attempts=8, answered=answered,
            loss=float(omega), n=100)
        for u, (merged, answered, omega) in spec.items()
    ))


def test_the_policy_reads_the_round_as_cui_states_it():
    states = _cui_round_states()
    env = _env(mission_round=5)
    for u, (omega, rho, x, _lam) in CUI_ROUND.items():
        got_x, got_w, got_rho = device_inputs(states[u], states, env, weights="oort")
        assert got_x == x
        assert got_rho == pytest.approx(float(rho))
        assert got_w == pytest.approx(float(omega) / 0.3125)


def test_restricted_to_the_connected_uavs_the_literal_policy_picks_b():
    states = _cui_round_states()
    connected = [_wp(1.0, "A"), _wp(2.0, "B"), _wp(3.0, "C")]
    pol = WhittlePolicy("literal", weights="oort")
    route = pol.admit_and_order(connected, states, _env(mission_round=5))
    assert _ids(route) == ["B", "A", "C"]
    b_index = pol.contact_indices(connected, states, _env(mission_round=5))[1]
    assert b_index == pytest.approx((23 / 15) / 0.3125)


def test_with_lambda_unobserved_the_variants_order_the_round_differently():
    """Without Λ, the literal index ranks the flaky B above A and C; the
    expected index ranks it last. D, the oldest (x = 4), leads both, and under
    the literal index its low rho (0.2) adds to that lead."""
    states = _cui_round_states()
    contacts = [_wp(1.0, "A"), _wp(2.0, "B"), _wp(3.0, "C"), _wp(4.0, "D")]
    env = _env(mission_round=5)
    literal = WhittlePolicy("literal", weights="oort")
    expected = WhittlePolicy("expected", weights="oort")
    assert _ids(literal.admit_and_order(contacts, states, env)) == ["D", "B", "A", "C"]
    assert _ids(expected.admit_and_order(contacts, states, env)) == ["D", "A", "C", "B"]
    # Paper-scale expected indices: D 1.3, A 0.87, C 0.5, B 0.46.
    got = expected.contact_indices(contacts, states, env)
    assert [g * 0.3125 for g in got] == pytest.approx([0.87, 0.46, 0.5, 1.3])


def test_after_b_merges_the_next_plan_sees_cuis_next_aou():
    """Eq. (33): the selected UAV resets to 1, every other one ages by 1."""
    states = _cui_round_states()
    states[DeviceID("B")].last_merged_round = 5
    env = _env(mission_round=6)
    xs = {u: device_inputs(states[u], states, env)[0] for u in "ABCD"}
    assert xs == {"A": 3, "B": 1, "C": 2, "D": 5}


# --------------------------------------------------------------------------- #
# 3. The expected index and the port decision
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("x, omega, rho", [
    (3, F(9, 50), F(1, 2)), (1, F(1, 10), F(9, 10)), (4, F(1, 4), F(1, 5)),
    (10, F(1, 8), F(1, 4)), (2, F(1, 5), F(3, 10)), (7, F(1), F(1)),
])
def test_expected_index_is_rho_times_the_literal_index(x, omega, rho):
    literal = whittle_index(x, True, omega, rho)
    assert expected_whittle_index(x, omega, rho) == rho * literal


@pytest.mark.parametrize("x, omega, rho, w", [
    (3, 0.18, 0.5, 0.81), (4, 0.25, 0.2, 1.3), (2, 0.2, 0.3, 0.46),
])
def test_expected_index_matches_the_hand_worked_mule_values(x, omega, rho, w):
    assert expected_whittle_index(x, omega, rho) == pytest.approx(w, rel=1e-12)


def test_at_equal_age_literal_prefers_low_rho_and_expected_prefers_high_rho():
    lo, hi = 0.2, 0.8
    assert whittle_index(3, True, 1.0, lo) > whittle_index(3, True, 1.0, hi)
    assert expected_whittle_index(3, 1.0, lo) < expected_whittle_index(3, 1.0, hi)


def test_a_dead_devices_expected_index_is_unbounded_but_below_literal():
    """The docstring's worked case: attempted every mission, never answers.

    W >= omega * x always, so the pressure is unbounded. It is roughly linear
    while rho-hat falls, and quadratic once rho-hat sits at rho_min. The
    literal index is far above it throughout.
    """
    def dead(x):
        st = _st("dead", merged=0, attempts=x - 1, answered=0)
        return device_inputs(st, _states(st), _env(mission_round=x))

    ratios = {}
    for x in range(1, 102):
        got_x, omega, rho = dead(x)
        assert got_x == x
        w = expected_whittle_index(x, omega, rho)
        assert w >= omega * x
        assert w <= whittle_index(x, True, omega, rho)
        ratios[x] = w / x
    assert expected_whittle_index(*dead(2)) == pytest.approx(7 / 3)
    assert [ratios[x] for x in (6, 19, 41, 101)] == pytest.approx(
        [19 / 14, 1.45, 2.0, 3.5])
    assert dead(19)[2] == DEFAULT_RHO_MIN
    assert whittle_index(6, True, *dead(6)[1:]) == pytest.approx(57.0)


def _flaky_and_steady():
    # Same age (x = 4 - 1 = 3), same weight; rho 2/12 versus 10/12.
    return _states(_st("flaky", merged=1, attempts=10, answered=1),
                   _st("steady", merged=1, attempts=10, answered=9))


def test_the_literal_policy_ranks_the_flaky_device_first_and_expected_last():
    states = _flaky_and_steady()
    env = _env(mission_round=4)
    # The flaky device sits at the larger position, so the tie-break alone
    # would put it second: only the index can put it first.
    contacts = [_wp(10.0, "flaky"), _wp(-10.0, "steady")]
    literal, expected = WhittlePolicy("literal"), WhittlePolicy("expected")
    assert _ids(literal.admit_and_order(contacts, states, env)) == ["flaky", "steady"]
    assert _ids(expected.admit_and_order(contacts, states, env)) == ["steady", "flaky"]
    assert literal.contact_indices(contacts, states, env) == pytest.approx([21.0, 6.6])
    assert expected.contact_indices(contacts, states, env) == pytest.approx([3.5, 5.5])


def test_under_a_one_contact_budget_each_variant_admits_its_favourite():
    states = _flaky_and_steady()
    env = _env(mission_round=4)
    contacts = [_wp(10.0, "flaky"), _wp(-10.0, "steady")]
    kw = dict(mission_deadline_ts=NOW + 15.0, feasibility_model=MODEL)
    assert _ids(WhittlePolicy("literal").admit_and_order(
        contacts, states, env, **kw)) == ["flaky"]
    assert _ids(WhittlePolicy("expected").admit_and_order(
        contacts, states, env, **kw)) == ["steady"]


def test_expected_is_the_default_variant():
    assert WhittlePolicy().variant == "expected"
    assert WhittlePolicy().weights == "uniform"


# --------------------------------------------------------------------------- #
# 4. Inputs: x, rho, omega, and their fallbacks
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("on_time, missed, want_rho", [
    # Counts chosen so the specified attempts = on_time + missed is told apart
    # from plausible wrong ones: attempts = missed would give 3/7 and 4/5,
    # attempts = on_time would give 3/4 and 4/5.
    (2, 5, F(1, 3)),        # (2 + 1) / (7 + 2)
    (3, 3, F(1, 2)),        # (3 + 1) / (6 + 2)
])
def test_without_the_new_fields_round_and_rho_fall_back(on_time, missed, want_rho):
    """R = 1 + max last_served_round over ALL states; U = last_clean_round;
    rho over on_time_count / (on_time_count + missed_count)."""
    a = _Hidden(_st("a", clean_round=2, served_round=3,
                    on_time=on_time, missed=missed),
                *NEW_STATE_FIELDS)
    # Not a candidate, but its outcome in round 6 fixes the round being planned.
    b = _Hidden(_st("b", served_round=6), *NEW_STATE_FIELDS)
    states = {DeviceID("a"): a, DeviceID("b"): b}
    env = _Hidden(_env(), "mission_round")
    x, omega, rho = device_inputs(a, states, env)
    assert x == 7 - 2
    assert rho == pytest.approx(float(want_rho), rel=1e-12)
    assert omega == 1.0


def test_the_fallback_round_also_applies_when_mission_round_is_none_or_zero():
    a = _st("a", merged=0, served_round=4)
    states = _states(a)
    assert device_inputs(a, states, _env(mission_round=None))[0] == 5
    env0 = _env()
    object.__setattr__(env0, "mission_round", 0)
    assert device_inputs(a, states, env0)[0] == 5


def test_the_new_fields_take_precedence_over_the_fallbacks():
    st = _st("a", merged=2, attempts=4, answered=1,
             clean_round=5, served_round=5, on_time=9, missed=0)
    x, _omega, rho = device_inputs(st, _states(st), _env(mission_round=8))
    assert x == 8 - 2
    assert rho == pytest.approx((1 + 1) / (4 + 2))


def test_a_never_attempted_device_gets_rho_one_half_even_with_on_time_counts():
    st = _st("a", attempts=0, answered=0, on_time=9, missed=0)
    assert connection_probability(st) == 0.5


def test_a_none_last_merged_round_falls_back_to_last_clean_round():
    st = _st("a", clean_round=3)
    st.last_merged_round = None
    assert device_inputs(st, _states(st), _env(mission_round=5))[0] == 2


def test_age_is_at_least_one():
    """Cui's AoU starts at 1 (Algorithm 2 line 2) and never drops below it."""
    fresh = _st("a", merged=0)
    assert device_inputs(fresh, _states(fresh), _env(mission_round=1))[0] == 1
    ahead = _st("b", merged=9)          # inconsistent input: merged "after" R
    assert device_inputs(ahead, _states(ahead), _env(mission_round=5))[0] == 1


def test_rho_is_clamped_to_rho_min_and_answered_to_attempts():
    dead = _st("a", attempts=100, answered=0)
    assert connection_probability(dead) == DEFAULT_RHO_MIN
    assert connection_probability(dead, rho_min=0.2) == 0.2
    over = _st("b", attempts=4, answered=9)
    assert connection_probability(over) == pytest.approx(5 / 6)


def test_oort_weights_are_mean_one_and_the_never_measured_get_the_mean():
    a = _st("a", loss=0.5, n=100)       # U = 50
    b = _st("b", loss=1.5, n=100)       # U = 150
    c = _st("c")                        # never reported: U = inf
    states = _states(a, b, c)
    env = _env(mission_round=2)
    w = {s.device_id: device_inputs(s, states, env, weights="oort")[1]
         for s in (a, b, c)}
    assert w == {"a": pytest.approx(0.5), "b": pytest.approx(1.5), "c": 1.0}


def test_oort_weights_with_nothing_measured_are_uniform():
    a, b = _st("a"), _st("b", loss=None, n=40)
    states = _states(a, b)
    for s in (a, b):
        assert device_inputs(s, states, _env(), weights="oort")[1] == 1.0


def test_a_nan_loss_counts_as_unmeasured():
    a = _st("a", loss=float("nan"), n=100)
    b = _st("b", loss=1.0, n=100)
    c = _st("c", loss=3.0, n=100)
    states = _states(a, b, c)
    w = {s.device_id: device_inputs(s, states, _env(), weights="oort")[1]
         for s in (a, b, c)}
    assert w == {"a": 1.0, "b": pytest.approx(0.5), "c": pytest.approx(1.5)}


def test_a_zero_loss_weight_is_floored_not_zero():
    z = _st("z", loss=0.0, n=100)
    b = _st("b", loss=1.0, n=100)
    states = _states(z, b)
    assert device_inputs(z, states, _env(), weights="oort")[1] == OMEGA_FLOOR


def test_infinite_utility_never_reaches_the_index():
    states = _states(_st("fresh"), _st("seen", loss=0.4, n=50, merged=1))
    contacts = [_wp(5.0, "fresh"), _wp(6.0, "seen")]
    for variant in ("expected", "literal"):
        pol = WhittlePolicy(variant, weights="oort")
        got = pol.contact_indices(contacts, states, _env(mission_round=3))
        assert all(math.isfinite(g) for g in got)


def test_a_member_with_no_state_is_treated_as_fresh_and_not_stored():
    states = _states(_st("known", merged=2))
    contacts = [_wp(5.0, "ghost")]
    pol = WhittlePolicy()
    pol.admit_and_order(contacts, states, _env(mission_round=4))
    assert pol.last_device_inputs[DeviceID("ghost")] == (4, 1.0, 0.5)
    assert DeviceID("ghost") not in states


def test_a_contact_index_is_the_sum_of_its_members():
    states = _states(_st("a", merged=1, attempts=3, answered=2),
                     _st("b", merged=3, attempts=5, answered=0))
    env = _env(mission_round=5)
    for variant in ("expected", "literal"):
        pol = WhittlePolicy(variant)
        pair, a, b = pol.contact_indices(
            [_wp(1.0, "a", "b"), _wp(2.0, "a"), _wp(3.0, "b")], states, env)
        assert pair == pytest.approx(a + b)


# --------------------------------------------------------------------------- #
# 5. The scheduler contract
# --------------------------------------------------------------------------- #

def _ladder():
    """Four contacts whose rank rises with distance: d(40) > c > b > a(10)."""
    states = _states(_st("a", merged=4), _st("b", merged=3),
                     _st("c", merged=2), _st("d", merged=1))
    contacts = [_wp(10.0, "a"), _wp(20.0, "b"), _wp(30.0, "c"), _wp(40.0, "d")]
    return contacts, states, _env(mission_round=5)


def test_without_a_budget_every_contact_is_admitted_in_index_order():
    contacts, states, env = _ladder()
    for variant in ("expected", "literal"):
        route = WhittlePolicy(variant).admit_and_order(contacts, states, env)
        assert _ids(route) == ["d", "c", "b", "a"]


@pytest.mark.parametrize("budget_s, expected", [
    (60.0, ["d", "c", "b"]),     # 40 + 10 + 10 = 60 fits; a would need 70
    (45.0, ["d"]),               # c would need 50
    (15.0, ["a"]),               # the far high-rank contacts do not veto a
])
def test_admission_respects_the_budget(budget_s, expected):
    contacts, states, env = _ladder()
    route = WhittlePolicy().admit_and_order(
        contacts, states, env,
        mission_deadline_ts=NOW + budget_s, feasibility_model=MODEL,
    )
    assert _ids(route) == expected
    assert _route_seconds(route) <= budget_s


def test_travel_is_priced_with_the_given_model():
    contacts, states, env = _ladder()
    fast = FeasibilityModel(cruise_speed_m_s=10.0, session_time_s=0.0)
    route = WhittlePolicy().admit_and_order(
        contacts, states, env,
        mission_deadline_ts=NOW + 15.0, feasibility_model=fast,
    )
    assert _ids(route) == ["d", "c", "b", "a"]      # 4 + 1 + 1 + 1 = 7 s


def test_the_route_is_deterministic_and_independent_of_input_order():
    # Fresh devices all tie, so position then devices decide.
    states = _states(*(_st(d) for d in "abcde"))
    contacts = [_wp(30.0, "c"), _wp(10.0, "b"), _wp(10.0, "a"),
                _wp(50.0, "e"), _wp(20.0, "d")]
    pol = WhittlePolicy()
    env = _env(mission_round=3)
    first = pol.admit_and_order(contacts, states, env)
    assert _ids(first) == ["a", "b", "d", "c", "e"]
    rng = random.Random(0)
    for _ in range(10):
        shuffled = list(contacts)
        rng.shuffle(shuffled)
        assert pol.admit_and_order(shuffled, states, env) == first


def test_admit_and_order_does_not_mutate_device_states():
    states = _cui_round_states()
    states[DeviceID("E")] = _st("E", clean_round=2, on_time=1, missed=2,
                                loss=0.3, n=10)
    before = copy.deepcopy({k: vars(v) for k, v in states.items()})
    contacts = [_wp(1.0, "A"), _wp(2.0, "B", "C"), _wp(4.0, "D"), _wp(5.0, "E")]
    for variant in ("expected", "literal"):
        for weights in ("uniform", "oort"):
            WhittlePolicy(variant, weights=weights).admit_and_order(
                contacts, states, _env(mission_round=5),
                mission_deadline_ts=NOW + 6.0, feasibility_model=MODEL,
            )
    assert {k: vars(v) for k, v in states.items()} == before


def test_only_given_contacts_are_returned_each_once():
    contacts, states, env = _ladder()
    route = WhittlePolicy().admit_and_order(
        contacts, states, env,
        mission_deadline_ts=NOW + 60.0, feasibility_model=MODEL,
    )
    assert all(any(r is c for c in contacts) for r in route)
    assert len({id(r) for r in route}) == len(route)


def test_no_contacts_gives_an_empty_route():
    assert WhittlePolicy().admit_and_order([], {}, _env()) == []


def test_the_policy_declares_its_name_and_in_flight_rule():
    assert WhittlePolicy.name == "WHITTLE"
    assert WhittlePolicy.in_flight_check == IN_FLIGHT_BUDGET


@pytest.mark.parametrize("kwargs", [
    {"variant": "optimal"},
    {"weights": "shapley"},
    {"rho_min": 0.0},
    {"rho_min": 1.5},
])
def test_bad_constructor_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        WhittlePolicy(**kwargs)


@pytest.mark.parametrize("x, omega, rho", [
    (0, 1.0, 0.5),            # scorer age fed without the +1
    (2, 1.0, 0.0),            # eq. (48) divides by rho
    (2, 1.0, 1.5),
    (2, -1.0, 0.5),
    (float("nan"), 1.0, 0.5),
])
def test_the_index_rejects_inputs_outside_cuis_model(x, omega, rho):
    with pytest.raises(ValueError):
        whittle_index(x, True, omega, rho)
    with pytest.raises(ValueError):
        expected_whittle_index(x, omega, rho)
