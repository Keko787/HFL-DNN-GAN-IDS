"""FedEx-Async with CARP (arm D4; Bian, Shen, Chen, Xu, IEEE TMC 24(6), 2025).

Two halves. The CARP functions are pinned to the worked example that the
paper extraction computed and the checker re-derived by brute force (depot at
the origin, A(3,0), B(3,4), C(-3,0), D(-3,-4), K = 2, T_trans = 1): costs,
eq. (30) probabilities, the argmin of every objective by enumeration, and the
eq. (29) energy gate. The chain's internals are pinned directly (its cached
round trips stay equal to a fresh evaluation, the temperature follows the
geometric schedule, feasible states outrank cheaper infeasible ones). The
Gibbs search is then checked against exhaustive search with exact (Held-Karp)
tours, both on the family its annealing defaults were tuned on and on a fresh
family, where it is allowed its measured, occasional miss.

The policy half pins what makes D4 FedEx rather than one of our own arms: it
visits every contact on a closed tour (not an open path), never drops one for
a deadline or a budget, and only *records* whether the tour fits; with a dock
given it flies home to the dock rather than to wherever the mule stood.
"""

from __future__ import annotations

import copy
import itertools
import math
import random

import pytest

import hermes.scheduler.policies.fedex_carp as fedex_mod
from hermes.scheduler import FLScheduler
from hermes.scheduler.policies.budget_walk import IN_FLIGHT_NONE
from hermes.scheduler.policies.fedex_carp import (
    DEFAULT_INITIAL_ACCEPTANCE,
    DEFAULT_SWEEPS,
    DEFAULT_TOUR_RESTARTS,
    FedExCarpPolicy,
    carp_assign,
    carp_cost,
    carp_search,
    gibbs_conditional,
    tour_time_s,
)
from hermes.scheduler.routing import order_contacts, path_cost
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionOutcome,
    MissionSlice,
    MuleID,
)

# --------------------------------------------------------------------------- #
# Worked-example fixture
# --------------------------------------------------------------------------- #

DEPOT = (0.0, 0.0)
FIX = {
    DeviceID("A"): (3.0, 0.0),
    DeviceID("B"): (3.0, 4.0),
    DeviceID("C"): (-3.0, 0.0),
    DeviceID("D"): (-3.0, -4.0),
}
IDS = sorted(FIX)
ACD_TOUR = 10.0 + 2.0 * math.sqrt(13.0)      # 0-A-D-C-0 = 17.2111
ABCD_TOUR = 6.0 + ACD_TOUR                   # optimum over all four = 23.2111


def _a(code: str):
    """Paper notation '1122' -> {A: 0, B: 0, C: 1, D: 1} (labels 1..K -> 0..K-1)."""
    return {d: int(ch) - 1 for d, ch in zip(IDS, code)}


def _cost(code: str, speeds=(1.0, 1.0)):
    return carp_cost(_a(code), FIX, n_transporters=2, depot=DEPOT, speeds=speeds,
                     t_trans=1.0)


def _enumerate(speeds, objective):
    """{paper code: cost} for all 16 assignments."""
    return {
        "".join(str(k + 1) for k in combo): _cost(
            "".join(str(k + 1) for k in combo), speeds).objective(objective)
        for combo in itertools.product((0, 1), repeat=4)
    }


def _argmin(table):
    best = min(table.values())
    return {code for code, v in table.items() if abs(v - best) < 1e-9}, best


def _second(table):
    best = min(table.values())
    rest = {c: v for c, v in table.items() if v > best + 1e-9}
    return _argmin(rest)


def _groups(assignment):
    """Partition as a set of frozensets: equality up to relabelling."""
    out = {}
    for did, k in assignment.items():
        out.setdefault(k, set()).add(did)
    return {frozenset(s) for s in out.values()}


# --------------------------------------------------------------------------- #
# 1. Costs: eqs. (9), (25), (27)
# --------------------------------------------------------------------------- #

def test_case1_optimum_costs_sync_14_async_784():
    c = _cost("1122")
    assert c.tour_lengths == pytest.approx((12.0, 12.0))
    assert c.deltas == pytest.approx((14.0, 14.0))
    assert c.counts == (2, 2)
    assert c.sync_cost == 14.0
    assert c.async_cost == 784.0
    assert c.total_cost == 28.0


def test_case1_moving_a_to_transporter_2():
    """R_1 = {B}: tour 10, Delta 11. R_2 = {A,C,D}: best tour 0-A-D-C-0."""
    c = _cost("2122")
    assert c.tour_lengths == pytest.approx((10.0, ACD_TOUR), abs=1e-12)
    assert c.deltas == pytest.approx((11.0, 3.0 + ACD_TOUR), abs=1e-12)
    assert c.sync_cost == pytest.approx(20.2111, abs=1e-4)
    assert c.async_cost == pytest.approx(121.0 + 3.0 * (3.0 + ACD_TOUR) ** 2, rel=1e-12)
    assert c.async_cost == pytest.approx(1346.466, abs=1e-3)
    assert set(c.tours[1]) == {"A", "C", "D"}
    assert c.tours[1] in (("A", "D", "C"), ("C", "D", "A"))


def test_case2_heterogeneous_speeds():
    c = _cost("1122", speeds=(1.0, 2.0))
    assert c.deltas == pytest.approx((14.0, 8.0))
    assert c.sync_cost == 14.0
    assert c.async_cost == 520.0
    moved = _cost("2122", speeds=(1.0, 2.0))
    assert moved.deltas == pytest.approx((11.0, 3.0 + ACD_TOUR / 2.0), abs=1e-12)
    assert moved.sync_cost == pytest.approx(11.6056, abs=1e-4)
    assert moved.async_cost == pytest.approx(525.0665, abs=1e-4)


def test_empty_transporter_has_zero_round_trip():
    c = _cost("1111")
    assert c.deltas == pytest.approx((4.0 + ABCD_TOUR, 0.0), abs=1e-12)
    assert c.counts == (4, 0)
    assert c.tours[1] == ()
    assert c.total_cost == pytest.approx(27.2111, abs=1e-4)


def test_every_tour_is_a_permutation_of_its_clients():
    c = _cost("1212")
    assert sorted(c.tours[0]) == ["A", "C"]
    assert sorted(c.tours[1]) == ["B", "D"]


# --------------------------------------------------------------------------- #
# 2. Argmin by enumeration (extraction + checker values)
# --------------------------------------------------------------------------- #

def test_case1_enumeration():
    for obj, best, runner_up in (("sync", 14.0, 17.2111), ("async", 784.0, 1184.8882)):
        table = _enumerate((1.0, 1.0), obj)
        codes, v = _argmin(table)
        assert codes == {"1122", "2211"} and v == pytest.approx(best)
        codes2, v2 = _second(table)
        assert codes2 == {"1221", "2112"}           # the checker's tie
        assert v2 == pytest.approx(runner_up, abs=1e-4)
    # Shortest-Total prefers one transporter doing everything: a tie.
    codes, v = _argmin(_enumerate((1.0, 1.0), "total"))
    assert codes == {"1111", "2222"} and v == pytest.approx(27.2111, abs=1e-4)


def test_case2_enumeration():
    speeds = (1.0, 2.0)
    codes, v = _argmin(_enumerate(speeds, "total"))
    assert codes == {"2222"} and v == pytest.approx(15.6056, abs=1e-4)
    codes, v = _argmin(_enumerate(speeds, "sync"))
    assert codes == {"2122", "2221"} and v == pytest.approx(11.6056, abs=1e-4)
    table = _enumerate(speeds, "async")
    codes, v = _argmin(table)
    assert codes == {"1122", "2211"} and v == pytest.approx(520.0)
    codes2, v2 = _second(table)
    assert codes2 == {"2122", "2221"} and v2 == pytest.approx(525.0665, abs=1e-4)


# --------------------------------------------------------------------------- #
# 3. Eq. (30) and the eq. (29) gate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("speeds,objective,q,p1,p2", [
    ((1.0, 1.0), "sync", 2.0, 0.957121, 0.042879),
    ((1.0, 1.0), "async", 100.0, 0.996405, 0.003595),
    ((1.0, 2.0), "sync", 2.0, 0.231969, 0.768031),
    ((1.0, 2.0), "async", 100.0, 0.512663, 0.487337),
])
def test_gibbs_probabilities_for_client_a(speeds, objective, q, p1, p2):
    got = gibbs_conditional(DeviceID("A"), _a("1122"), FIX, n_transporters=2,
                            depot=DEPOT, speeds=speeds, t_trans=1.0, q=q,
                            objective=objective)
    assert got[0] == pytest.approx(p1, abs=1e-6)
    assert got[1] == pytest.approx(p2, abs=1e-6)


def test_zero_temperature_is_greedy():
    got = gibbs_conditional(DeviceID("A"), _a("1122"), FIX, n_transporters=2,
                            depot=DEPOT, q=0.0)
    assert got == {0: 1.0, 1: 0.0}


def _energy_gate(speeds=(1.0, 1.0), p_hover=2.0, p_slf=3.0, cap=50.0):
    """The extraction's eq. (29) test: E = P_hover*R_k*T_trans + P_SLF*L_k/V_k."""
    def budget(k, members, tour_length):
        return p_hover * len(members) * 1.0 + p_slf * tour_length / speeds[k] <= cap
    return budget


def test_energy_gate_removes_the_infeasible_transporter():
    budget = _energy_gate()
    assert 2.0 * 2 + 3.0 * 12.0 == 40.0                     # E_1({A,B}) <= 50
    assert 2.0 * 3 + 3.0 * ACD_TOUR == pytest.approx(57.633, abs=1e-3)  # > 50
    got = gibbs_conditional(DeviceID("A"), _a("1122"), FIX, n_transporters=2,
                            depot=DEPOT, q=100.0, budget=budget)
    assert got == {0: 1.0}


def test_an_empty_feasible_set_keeps_the_client_where_it_is():
    never = lambda k, members, length: False            # noqa: E731
    got = gibbs_conditional(DeviceID("A"), _a("2122"), FIX, n_transporters=2,
                            depot=DEPOT, q=1.0, budget=never)
    assert got == {1: 1.0}


def test_search_repairs_an_infeasible_start_under_the_gate():
    budget = _energy_gate()
    res = carp_search(FIX, n_transporters=2, depot=DEPOT, budget=budget,
                      initial=_a("1111"), seed=0)
    assert res.feasible
    assert _groups(res.assignment) == {frozenset("AB"), frozenset("CD")}


def test_a_feasible_state_outranks_a_cheaper_infeasible_start():
    """Shortest-Total under the eq. (29) gate, everything on transporter 1.
    The start is the cheapest assignment there is (total 27.2111) but needs
    E = 2*4 + 3*23.2111 = 77.6 > 50; every feasible partition costs more, the
    best being {A,B},{C,D} at 28. The async-objective repair test above
    cannot see the ranking rule, because there the infeasible start is also
    the dearer one (1360 > 784)."""
    budget = _energy_gate()
    start = _cost("1111")
    assert start.total_cost == pytest.approx(27.2111, abs=1e-4)
    assert not budget(0, frozenset(FIX), start.tour_lengths[0])
    res = carp_search(FIX, n_transporters=2, depot=DEPOT, objective="total",
                      budget=budget, initial=_a("1111"), seed=0)
    assert res.feasible
    assert _groups(res.assignment) == {frozenset("AB"), frozenset("CD")}
    assert res.cost.total_cost == pytest.approx(28.0)
    assert res.cost.total_cost > res.initial_cost.total_cost


# --------------------------------------------------------------------------- #
# 4. The Gibbs search
# --------------------------------------------------------------------------- #

def test_carp_assign_finds_the_fixture_optimum_up_to_relabelling():
    got = carp_assign(FIX, n_transporters=2, depot=DEPOT, t_trans=1.0, seed=0)
    assert _groups(got) == {frozenset("AB"), frozenset("CD")}
    c = carp_cost(got, FIX, n_transporters=2, depot=DEPOT)
    assert (c.sync_cost, c.async_cost) == (14.0, 784.0)


@pytest.mark.parametrize("speeds,objective,expected", [
    ((1.0, 1.0), "async", 784.0),
    ((1.0, 1.0), "sync", 14.0),
    ((1.0, 1.0), "total", 4.0 + ABCD_TOUR),
    ((1.0, 2.0), "async", 520.0),
    ((1.0, 2.0), "sync", 3.0 + ACD_TOUR / 2.0),
    ((1.0, 2.0), "total", 4.0 + ABCD_TOUR / 2.0),
])
def test_search_reaches_every_fixture_argmin(speeds, objective, expected):
    res = carp_search(FIX, n_transporters=2, depot=DEPOT, speeds=speeds,
                      objective=objective, seed=1)
    assert res.cost.objective(objective) == pytest.approx(expected, abs=1e-9)


def test_gibbs_lowers_the_async_cost_from_a_bad_start():
    bad = _a("1212")                 # {A,C} and {B,D}: both tours cross the depot
    assert _cost("1212").async_cost == pytest.approx(1360.0)
    res = carp_search(FIX, n_transporters=2, depot=DEPOT, initial=bad, seed=3,
                      polish=False)
    assert res.initial_cost.async_cost == pytest.approx(1360.0)
    assert res.cost.async_cost == 784.0
    assert res.updates == 4 * DEFAULT_SWEEPS


def test_gibbs_improves_a_random_instance_from_everything_on_one_transporter():
    rng = random.Random(4)
    pos = {DeviceID(f"d{i}"): (rng.uniform(-100, 100), rng.uniform(-100, 100), 0.0)
           for i in range(10)}
    start = {d: 0 for d in pos}
    res = carp_search(pos, n_transporters=3, t_trans=5.0, initial=start, seed=4,
                      polish=False)
    assert res.cost.async_cost < 0.5 * res.initial_cost.async_cost


def _nine_client_instance(seed=4):
    rng = random.Random(seed)
    return {DeviceID(f"d{i}"): (rng.uniform(-100, 100), rng.uniform(-100, 100), 0.0)
            for i in range(9)}


def test_temperature_is_geometric_and_constant_within_a_sweep(monkeypatch):
    """q_s = q0 * gamma^s, one value per sweep (deviation 5)."""
    seen = []
    real = fedex_mod._conditional

    def spy(*args, **kwargs):
        seen.append(kwargs["q"])
        return real(*args, **kwargs)

    monkeypatch.setattr(fedex_mod, "_conditional", spy)
    q0, gamma, sweeps = 50.0, 0.9, 6
    carp_search(FIX, n_transporters=2, depot=DEPOT, q0=q0, gamma=gamma,
                sweeps=sweeps, polish=False, seed=0)
    n = len(FIX)
    assert len(seen) == sweeps * n
    for s_idx in range(sweeps):
        block = seen[s_idx * n:(s_idx + 1) * n]
        assert len(set(block)) == 1
        assert block[0] == pytest.approx(q0 * gamma ** s_idx, rel=1e-15)


def test_derived_initial_temperature_matches_the_uphill_moves(monkeypatch):
    """q0=None: the mean uphill single-client move from the start, divided by
    ln(1/acceptance) -- recomputed here from carp_cost alone."""
    start = _a("1212")                               # the round-robin start
    base = _cost("1212").async_cost
    ups = []
    for d in IDS:
        moved = dict(start)
        moved[d] = 1 - start[d]
        c = carp_cost(moved, FIX, n_transporters=2, depot=DEPOT).async_cost
        if c > base:
            ups.append(c - base)
    assert ups
    expected_q0 = (math.fsum(ups) / len(ups)) / math.log(1.0 / DEFAULT_INITIAL_ACCEPTANCE)

    seen = []
    real = fedex_mod._conditional

    def spy(*args, **kwargs):
        seen.append(kwargs["q"])
        return real(*args, **kwargs)

    monkeypatch.setattr(fedex_mod, "_conditional", spy)
    carp_search(FIX, n_transporters=2, depot=DEPOT, sweeps=2, gamma=0.5,
                polish=False, seed=0)
    n = len(FIX)
    # The first n calls price the start at q = 0 to derive q0; then 2 sweeps.
    assert seen[:n] == [0.0] * n
    assert seen[n:2 * n] == [pytest.approx(expected_q0, rel=1e-12)] * n
    assert seen[2 * n:] == [pytest.approx(expected_q0 * 0.5, rel=1e-12)] * n


def test_chain_state_matches_a_fresh_evaluation_at_every_step(monkeypatch):
    """The chain caches Delta_k and updates only the donor and the receiver on
    a move; at every eq. (30) draw and after every move those caches must
    equal a fresh evaluation, and the best state it reports (no polish) must
    be the cheapest state it visited, priced from scratch."""
    pos = _nine_client_instance()
    k, t_trans = 3, 5.0
    visited = []
    real_cond, real_move = fedex_mod._conditional, fedex_mod._move

    def check(ev, sets, deltas):
        for t, members in enumerate(sets):
            assert deltas[t] == ev.delta(t, members)
        visited.append({d: t for t, members in enumerate(sets) for d in members})

    def spy_cond(ev, client, current, sets, deltas, **kwargs):
        check(ev, sets, deltas)
        assert client in sets[current]
        return real_cond(ev, client, current, sets, deltas, **kwargs)

    def spy_move(ev, sets, deltas, client, k_from, k_to):
        out = real_move(ev, sets, deltas, client, k_from, k_to)
        check(ev, out, deltas)
        return out

    monkeypatch.setattr(fedex_mod, "_conditional", spy_cond)
    monkeypatch.setattr(fedex_mod, "_move", spy_move)
    res = carp_search(pos, n_transporters=k, t_trans=t_trans, sweeps=40, seed=5,
                      polish=False, initial={d: 0 for d in pos})
    assert res.updates == 40 * len(pos)
    monkeypatch.undo()
    fresh = [carp_cost(a, pos, n_transporters=k, t_trans=t_trans, seed=5).async_cost
             for a in visited]
    assert res.cost.async_cost == pytest.approx(min(fresh), rel=1e-12)
    assert res.cost == carp_cost(res.assignment, pos, n_transporters=k,
                                 t_trans=t_trans, seed=5)


def test_no_sweeps_and_no_polish_returns_the_start():
    res = carp_search(FIX, n_transporters=2, depot=DEPOT, initial=_a("1212"),
                      sweeps=0, polish=False)
    assert res.assignment == _a("1212") and res.updates == 0


def test_carp_is_deterministic_under_a_seed():
    rng = random.Random(8)
    pos = {DeviceID(f"d{i}"): (rng.uniform(0, 50), rng.uniform(0, 50), 0.0)
           for i in range(9)}
    a = carp_search(pos, n_transporters=3, seed=11, sweeps=30)
    b = carp_search(pos, n_transporters=3, seed=11, sweeps=30)
    assert a.assignment == b.assignment and a.cost == b.cost


def test_carp_does_not_touch_the_global_rng():
    random.seed(99)
    expected = random.random()
    random.seed(99)
    carp_assign(FIX, n_transporters=2, depot=DEPOT, sweeps=5)
    assert random.random() == expected


def test_one_transporter_and_no_clients_are_trivial():
    one = carp_search(FIX, n_transporters=1, depot=DEPOT)
    assert one.assignment == {d: 0 for d in FIX} and one.updates == 0
    assert carp_assign({}, n_transporters=3) == {}


@pytest.mark.parametrize("kwargs", [
    {"n_transporters": 0},
    {"n_transporters": 2, "speeds": (1.0,)},
    {"n_transporters": 2, "speeds": (1.0, 0.0)},
    {"n_transporters": 2, "objective": "fastest"},
    {"n_transporters": 2, "gamma": 0.0},
    {"n_transporters": 2, "t_trans": -1.0},
    {"n_transporters": 2, "initial": {DeviceID("A"): 5, DeviceID("B"): 0,
                                      DeviceID("C"): 0, DeviceID("D"): 0}},
    {"n_transporters": 2, "initial": {DeviceID("A"): 0}},
])
def test_bad_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        carp_search(FIX, depot=DEPOT, **kwargs)


# --------------------------------------------------------------------------- #
# 5. Brute force with exact tours
# --------------------------------------------------------------------------- #

def _held_karp(points, depot):
    """Exact closed-tour length depot -> points -> depot (independent oracle)."""
    n = len(points)
    if n == 0:
        return 0.0
    d0 = [math.dist(depot, p) for p in points]
    dist = [[math.dist(a, b) for b in points] for a in points]
    dp = {(1 << j, j): d0[j] for j in range(n)}
    for mask in range(1, 1 << n):
        for j in range(n):
            v = dp.get((mask, j))
            if v is None:
                continue
            for k in range(n):
                if mask >> k & 1:
                    continue
                key = (mask | 1 << k, k)
                if v + dist[j][k] < dp.get(key, math.inf):
                    dp[key] = v + dist[j][k]
    full = (1 << n) - 1
    return min(dp[(full, j)] + d0[j] for j in range(n))


def _brute_force(pos, k, speeds, t_trans, objective):
    ids = sorted(pos)
    lengths = {}
    best = math.inf
    for combo in itertools.product(range(k), repeat=len(ids)):
        sets = [frozenset(i for i, kk in zip(ids, combo) if kk == t) for t in range(k)]
        deltas = []
        for t, s in enumerate(sets):
            if s not in lengths:
                lengths[s] = _held_karp([pos[i] for i in sorted(s)], (0.0, 0.0, 0.0))
            deltas.append(len(s) * t_trans + lengths[s] / speeds[t] if s else 0.0)
        counts = [len(s) for s in sets]
        if objective == "async":
            c = math.fsum(r * d * d for r, d in zip(counts, deltas))
        else:
            c = max(deltas)
        best = min(best, c)
    return best


def _instance(seed):
    """The family the annealing defaults were TUNED on (in-sample)."""
    rng = random.Random(seed)
    n, k = 4 + seed % 5, 2 + seed % 2          # 4..8 clients, K in {2, 3}
    pos = {DeviceID(f"d{i}"): (rng.uniform(-100, 100), rng.uniform(-100, 100), 0.0)
           for i in range(n)}
    speeds = tuple(rng.choice([1.0, 2.0, 5.0]) for _ in range(k))
    return pos, k, speeds, rng.uniform(1.0, 30.0)


@pytest.mark.parametrize("objective", ["async", "sync"])
@pytest.mark.parametrize("seed", list(range(10)))
def test_carp_matches_brute_force_on_random_instances(seed, objective):
    pos, k, speeds, t_trans = _instance(seed)
    opt = _brute_force(pos, k, speeds, t_trans, objective)
    got = carp_assign(pos, n_transporters=k, speeds=speeds, t_trans=t_trans,
                      objective=objective, seed=seed)
    val = carp_cost(got, pos, n_transporters=k, speeds=speeds,
                    t_trans=t_trans).objective(objective)
    assert val == pytest.approx(opt, rel=1e-9)


def _fresh_instance(seed):
    """A family never used for tuning: same ranges, different generator."""
    rng = random.Random(10_000 + seed)
    n, k = rng.randint(4, 8), rng.choice([2, 3])
    pos = {DeviceID(f"d{i}"): (rng.uniform(-100, 100), rng.uniform(-100, 100), 0.0)
           for i in range(n)}
    speeds = tuple(rng.choice([1.0, 2.0, 5.0]) for _ in range(k))
    return pos, k, speeds, rng.uniform(1.0, 30.0)


#: Out-of-sample allowance. Over 400 fresh instances the defaults missed the
#: optimum on 1.5% (async) / 2.8% (sync) of them, by at most 9.0%; seeds
#: 0-39 contain one miss per objective (async seed 30, +9.0%; sync seed 35,
#: +0.3%). A regression past 2 misses in 40 or past 10% excess fails.
FRESH_SEEDS = range(40)
FRESH_MAX_MISSES = 2
FRESH_MAX_EXCESS = 0.10


@pytest.mark.parametrize("objective", ["async", "sync"])
def test_carp_is_near_optimal_out_of_sample(objective):
    misses = []
    for seed in FRESH_SEEDS:
        pos, k, speeds, t_trans = _fresh_instance(seed)
        opt = _brute_force(pos, k, speeds, t_trans, objective)
        got = carp_assign(pos, n_transporters=k, speeds=speeds, t_trans=t_trans,
                          objective=objective, seed=seed)
        val = carp_cost(got, pos, n_transporters=k, speeds=speeds,
                        t_trans=t_trans).objective(objective)
        assert val >= opt * (1.0 - 1e-9)
        if val > opt * (1.0 + 1e-9):
            misses.append((seed, val / opt - 1.0))
    assert len(misses) <= FRESH_MAX_MISSES, misses
    assert all(excess <= FRESH_MAX_EXCESS for _, excess in misses), misses


# --------------------------------------------------------------------------- #
# 6. FedExCarpPolicy: the single-mule arm
# --------------------------------------------------------------------------- #

NOW = 5_000.0


def _wp(x, y, name, *, deadline=0.0, bucket=Bucket.SCHEDULED_THIS_ROUND):
    return ContactWaypoint(
        position=(float(x), float(y), 0.0),
        devices=(DeviceID(name),),
        bucket=bucket,
        deadline_ts=deadline,
    )


def _env(pose=(0.0, 0.0, 0.0), now=NOW):
    return SelectorEnv(mule_pose=pose, mule_energy=1.0, rf_prior_snr_db=20.0,
                       beacon_window_s=30.0, now=now)


def _contacts():
    rng = random.Random(21)
    return [_wp(rng.uniform(-200, 200), rng.uniform(-200, 200), f"c{i}",
                deadline=NOW - 100.0 * i) for i in range(7)]


def _states(contacts):
    return {
        wp.devices[0]: DeviceSchedulerState(
            device_id=wp.devices[0], is_in_slice=True, last_clean_ts=NOW - 10 * i,
            last_clean_round=i, last_served_round=i, on_time_count=i,
            last_outcome=MissionOutcome.CLEAN if i % 2 else None,
        )
        for i, wp in enumerate(contacts)
    }


def test_policy_identity():
    pol = FedExCarpPolicy()
    assert pol.name == "FEDEX"
    assert pol.in_flight_check == IN_FLIGHT_NONE
    assert pol.restarts == DEFAULT_TOUR_RESTARTS
    assert pol.depot is None


def test_policy_returns_every_contact_exactly_once():
    cs = _contacts()
    route = FedExCarpPolicy().admit_and_order(cs, _states(cs), _env())
    assert len(route) == len(cs)
    assert set(route) == set(cs)


def test_policy_route_is_the_optimal_closed_tour():
    cs = _contacts()
    route = FedExCarpPolicy().admit_and_order(cs, {}, _env())
    start = (0.0, 0.0, 0.0)
    got = path_cost([wp.position for wp in route], start, closed=True)
    opt = min(path_cost([cs[i].position for i in p], start, closed=True)
              for p in itertools.permutations(range(len(cs))))
    assert got == pytest.approx(opt, abs=1e-9)


def _abc():
    """A(10,0), B(10,10), C(-3,0) from the origin: the closed tour (A,B,C
    either way round, 39.40) and the open path (C,A,B, 26.00) differ; flown
    as a closed tour, C,A,B would cost 40.14. The route-optimality fixture
    above cannot tell the two shapes apart: there the optimal open path
    happens to be the reverse of the optimal closed tour."""
    return [_wp(10, 0, "A"), _wp(10, 10, "B"), _wp(-3, 0, "C")]


def test_policy_route_is_a_closed_tour_not_an_open_path():
    cs = _abc()
    route = [wp.devices[0] for wp in FedExCarpPolicy().admit_and_order(cs, {}, _env())]
    assert route in (["A", "B", "C"], ["C", "B", "A"])
    open_path = order_contacts(cs, (0.0, 0.0, 0.0), closed=False, restarts=8)
    assert [wp.devices[0] for wp in open_path] == ["C", "A", "B"]


@pytest.mark.parametrize("seed", [6, 57])
def test_policy_restarts_escape_a_two_opt_trap(seed):
    """Instances on which cheapest insertion + one 2-OPT run is stuck in a
    local optimum (test_two_opt.TRAPPED): the policy's restarts are what
    reach the optimum, so they must actually be used."""
    rng = random.Random(seed)
    n = 3 + seed % 5
    pts = [tuple(rng.uniform(0.0, 100.0) for _ in range(3)) for _ in range(n)]
    start = tuple(rng.uniform(0.0, 100.0) for _ in range(3))
    cs = [ContactWaypoint(position=p, devices=(DeviceID(f"t{i}"),),
                          bucket=Bucket.NEW, deadline_ts=0.0)
          for i, p in enumerate(pts)]
    opt = min(path_cost([pts[i] for i in perm], start, closed=True)
              for perm in itertools.permutations(range(n)))

    def tour(restarts):
        route = FedExCarpPolicy(restarts=restarts).admit_and_order(
            cs, {}, _env(pose=start))
        return path_cost([wp.position for wp in route], start, closed=True)

    assert tour(0) > opt + 1e-6
    assert tour(20) == pytest.approx(opt, abs=1e-9)


def test_policy_walks_a_convex_ring_in_hull_order():
    """The mule starts on vertex 0 of a regular 9-gon and the contacts are the
    other vertices: all in convex position, so the optimal closed tour is the
    hull order, in one direction or the other."""
    n = 9
    verts = [(100 * math.cos(2 * math.pi * k / n), 100 * math.sin(2 * math.pi * k / n))
             for k in range(n)]
    ring = [_wp(x, y, f"r{k}") for k, (x, y) in enumerate(verts)][1:]
    shuffled = list(ring)
    random.Random(2).shuffle(shuffled)
    route = FedExCarpPolicy().admit_and_order(
        shuffled, {}, _env(pose=(verts[0][0], verts[0][1], 0.0)))
    idx = [ring.index(wp) + 1 for wp in route]
    assert idx in (list(range(1, n)), list(range(n - 1, 0, -1)))


def test_policy_ignores_deadlines_and_buckets():
    cs = _contacts()
    renamed = [
        ContactWaypoint(position=wp.position, devices=wp.devices,
                        bucket=Bucket.NEW, deadline_ts=NOW + 1e6)
        for wp in cs
    ]
    a = FedExCarpPolicy().admit_and_order(cs, {}, _env())
    b = FedExCarpPolicy().admit_and_order(renamed, {}, _env())
    assert [wp.devices for wp in a] == [wp.devices for wp in b]


def test_policy_never_skips_even_with_a_tiny_deadline():
    cs = _contacts()
    pol = FedExCarpPolicy()
    route = pol.admit_and_order(cs, {}, _env(), mission_deadline_ts=NOW + 1e-3,
                                feasibility_model=FeasibilityModel())
    assert set(route) == set(cs)
    assert pol.last_tour_fits is False
    assert pol.last_tour_overrun_s == pytest.approx(NOW + pol.last_tour_cost_s
                                                    - (NOW + 1e-3))


def test_policy_fits_flag_true_with_a_generous_deadline():
    cs = _contacts()
    pol = FedExCarpPolicy()
    pol.admit_and_order(cs, {}, _env(), mission_deadline_ts=NOW + 1e6)
    assert pol.last_tour_fits is True
    assert pol.last_tour_overrun_s == 0.0


def test_policy_fits_flag_is_none_without_a_deadline():
    cs = _contacts()
    pol = FedExCarpPolicy()
    pol.admit_and_order(cs, {}, _env(), mission_deadline_ts=NOW + 1e-3)
    pol.admit_and_order(cs, {}, _env())            # diagnostics reset per plan
    assert pol.last_tour_fits is None
    assert pol.last_tour_overrun_s is None
    assert pol.last_fits_without_return is None
    assert pol.last_overrun_without_return_s is None
    assert pol.last_tour_cost_s is not None and pol.last_tour_cost_s > 0.0
    assert pol.last_return_leg_s is not None and pol.last_return_leg_s > 0.0


def test_policy_fits_exactly_at_the_deadline():
    """now + tour == deadline is on time (<=, as in S3b's walk). The numbers
    are exact in binary: 40 + 30 + 50 m at 4 m/s plus two 7 s sessions = 44 s."""
    model = FeasibilityModel(cruise_speed_m_s=4.0, session_time_s=7.0)
    cs = [_wp(40.0, 0.0, "a"), _wp(40.0, 30.0, "b")]
    pol = FedExCarpPolicy()
    pol.admit_and_order(cs, {}, _env(), mission_deadline_ts=NOW + 44.0,
                        feasibility_model=model)
    assert pol.last_tour_cost_s == 44.0
    assert pol.last_tour_fits is True
    assert pol.last_tour_overrun_s == 0.0
    pol.admit_and_order(cs, {}, _env(), mission_deadline_ts=NOW + 43.5,
                        feasibility_model=model)
    assert pol.last_tour_fits is False
    assert pol.last_tour_overrun_s == 0.5
    # Without the return leg the same rule: the canonical route is a, b, then
    # 50 m home (12.5 s), so the outbound part is exactly 31.5 s.
    route = pol.admit_and_order(cs, {}, _env(), mission_deadline_ts=NOW + 31.5,
                                feasibility_model=model)
    assert [wp.devices[0] for wp in route] == ["a", "b"]
    assert pol.last_return_leg_s == 12.5
    assert pol.last_fits_without_return is True
    assert pol.last_overrun_without_return_s == 0.0
    assert pol.last_tour_fits is False


def test_tour_cost_is_priced_with_the_feasibility_model_including_the_return():
    model = FeasibilityModel(cruise_speed_m_s=4.0, session_time_s=7.0)
    a, b = _wp(40.0, 0.0, "a"), _wp(40.0, 30.0, "b")
    pol = FedExCarpPolicy()
    route = pol.admit_and_order([b, a], {}, _env(), feasibility_model=model)
    # Closed tour 0 -> a -> b -> 0 (or reversed): 40 + 30 + 50 = 120 m at
    # 4 m/s = 30 s, plus one 7 s session per contact, none at the depot.
    assert pol.last_tour_cost_s == pytest.approx(30.0 + 2 * 7.0)
    assert tour_time_s(route, (0.0, 0.0, 0.0), model) == pol.last_tour_cost_s


def test_policy_reports_the_route_with_and_without_the_return_leg():
    """Until Phase 3 the mule never flies home, so the diagnostics also give
    the fit of the outbound part alone (what S3b's walk would price)."""
    model = FeasibilityModel(cruise_speed_m_s=4.0, session_time_s=7.0)
    cs = [_wp(40.0, 0.0, "a"), _wp(40.0, 30.0, "b")]
    pol = FedExCarpPolicy()
    route = pol.admit_and_order(cs, {}, _env(), mission_deadline_ts=NOW + 40.0,
                                feasibility_model=model)
    back = math.dist(route[-1].position, (0.0, 0.0, 0.0)) / 4.0
    assert pol.last_return_leg_s == pytest.approx(back)
    assert pol.last_tour_cost_s == pytest.approx(44.0)
    assert pol.last_tour_fits is False
    assert pol.last_tour_overrun_s == pytest.approx(4.0)
    assert back >= 10.0                          # either direction: 12.5 or 10 s
    assert pol.last_fits_without_return is True
    assert pol.last_overrun_without_return_s == 0.0


def test_policy_depot_at_the_mule_pose_is_the_closed_tour():
    cs = _contacts()
    a, b = FedExCarpPolicy(), FedExCarpPolicy(depot=(0.0, 0.0, 0.0))
    ra = a.admit_and_order(cs, {}, _env(), mission_deadline_ts=NOW + 100.0)
    rb = b.admit_and_order(cs, {}, _env(), mission_deadline_ts=NOW + 100.0)
    assert ra == rb
    assert (a.last_tour_cost_s, a.last_return_leg_s, a.last_tour_fits) == (
        b.last_tour_cost_s, b.last_return_leg_s, b.last_tour_fits)


def test_policy_with_a_depot_flies_home_to_it_not_to_the_start():
    """Mission 2 in today's simulator: the mule starts where it last stopped.
    With the dock given, the route is the shortest path from the pose through
    every contact to the dock, and the depot -- not the router's tie-break --
    picks the direction. On _abc() with the dock at (-3,-5): A,B,C then 5 m
    home (41.40); the closed tour around the origin runs C,B,A."""
    cs = _abc()
    model = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)
    dock = (-3.0, -5.0, 0.0)
    pol = FedExCarpPolicy(depot=dock)
    route = pol.admit_and_order(cs, {}, _env(), mission_deadline_ts=NOW + 38.0,
                                feasibility_model=model)
    assert [wp.devices[0] for wp in route] == ["A", "B", "C"]
    closed = FedExCarpPolicy().admit_and_order(cs, {}, _env())
    assert [wp.devices[0] for wp in closed] == ["C", "B", "A"]
    outbound = 10.0 + 10.0 + math.dist((10.0, 10.0), (-3.0, 0.0))
    assert pol.last_return_leg_s == pytest.approx(5.0)
    assert pol.last_tour_cost_s == pytest.approx(outbound + 5.0)
    assert pol.last_tour_cost_s == tour_time_s(route, (0.0, 0.0, 0.0), model, end=dock)
    assert pol.last_tour_fits is False                  # 41.40 > 38
    assert pol.last_fits_without_return is True         # 36.40 <= 38


def test_policy_with_a_depot_is_the_optimal_path_home():
    cs = _contacts()
    pose, dock = (150.0, -120.0, 0.0), (0.0, 0.0, 0.0)
    route = FedExCarpPolicy(depot=dock).admit_and_order(cs, {}, _env(pose=pose))
    assert set(route) == set(cs)
    got = path_cost([wp.position for wp in route], pose, end=dock)
    opt = min(path_cost([cs[i].position for i in p], pose, end=dock)
              for p in itertools.permutations(range(len(cs))))
    assert got == pytest.approx(opt, abs=1e-9)


def test_policy_rejects_a_depot_that_is_not_a_pose():
    with pytest.raises(ValueError):
        FedExCarpPolicy(depot=(1.0, 2.0))


def test_policy_defaults_to_the_s3b_model_when_none_is_given():
    cs = _contacts()
    pol = FedExCarpPolicy()
    route = pol.admit_and_order(cs, {}, _env())
    assert pol.last_tour_cost_s == pytest.approx(
        tour_time_s(route, (0.0, 0.0, 0.0), FeasibilityModel()))


def test_policy_does_not_mutate_device_states():
    cs = _contacts()
    states = _states(cs)
    before = copy.deepcopy(states)
    FedExCarpPolicy().admit_and_order(cs, states, _env(),
                                      mission_deadline_ts=NOW + 1.0)
    assert states == before


def test_policy_is_deterministic_and_input_order_free():
    cs = _contacts()
    ref = FedExCarpPolicy(seed=3).admit_and_order(cs, {}, _env())
    for s in range(4):
        shuffled = list(cs)
        random.Random(s).shuffle(shuffled)
        assert FedExCarpPolicy(seed=3).admit_and_order(shuffled, {}, _env()) == ref


def test_policy_empty_input():
    pol = FedExCarpPolicy()
    assert pol.admit_and_order([], {}, _env(), mission_deadline_ts=NOW + 1.0) == []
    assert pol.last_tour_cost_s == 0.0
    assert pol.last_tour_fits is True


def test_policy_rejects_negative_restarts():
    with pytest.raises(ValueError):
        FedExCarpPolicy(restarts=-1)


def test_scheduler_delegates_and_keeps_every_contact_over_budget():
    """Through FLScheduler.build_contact_queue: a 1 s mission budget would make
    S3b (and D1/D2's budget walk) drop almost everything; FedEx keeps all."""
    pol = FedExCarpPolicy()
    sch = FLScheduler(now_fn=lambda: 1000.0, target_selector=pol,
                      mission_budget_s=1.0)
    names = ("a", "b", "c", "d")
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m1"),
                                  device_ids=tuple(DeviceID(n) for n in names),
                                  issued_round=1, issued_at=1000.0))
    for i, n in enumerate(names):
        sch.device_states[DeviceID(n)].last_known_position = (100.0 * (i + 1), 0.0, 0.0)
    route = sch.build_contact_queue(rf_range_m=10.0, now=1000.0,
                                    mule_pose=(0.0, 0.0, 0.0))
    assert {d for wp in route for d in wp.devices} == {DeviceID(n) for n in names}
    assert pol.last_tour_fits is False
