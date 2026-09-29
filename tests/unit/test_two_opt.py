"""Shared routing primitives: cheapest insertion + 2-OPT (FeRRy Phase 2).

One router serves every arm that needs a route (FedEx/CARP now, the Phase-3
re-plan and the Phase-4 search later), so it is pinned against known optima
and against brute force: convex polygons (the optimal closed tour is the hull
order), a 2x3 grid, collinear points, and exhaustive search over every visiting
order for random instances with n <= 7. 2-OPT alone is allowed to stop at a
local optimum; with 20 seeded restarts it must reach the brute-force optimum on
the fixed instances below. The same brute force pins the fixed-end path (start
somewhere, finish at a given depot), the shape FedEx's policy flies when the
mule is not at its depot.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from hermes.scheduler.routing import (
    IMPROVEMENT_EPS,
    best_order,
    cheapest_insertion,
    nearest_neighbour,
    order_contacts,
    path_cost,
    two_opt,
)
from hermes.types import Bucket, ContactWaypoint, DeviceID

TOL = 1e-9


def _cost(order, points, start, closed, end=None):
    return path_cost([points[i] for i in order], start, closed=closed, end=end)


def _brute(points, start, closed, end=None):
    return min(
        _cost(p, points, start, closed, end)
        for p in itertools.permutations(range(len(points)))
    )


def _is_two_opt_local_optimum(order, points, start, closed, end=None):
    """No segment reversal shortens the route (the full 2-OPT neighbourhood
    with the start fixed), checked independently of the implementation."""
    base = _cost(order, points, start, closed, end)
    n = len(order)
    for i in range(n - 1):
        for j in range(i + 1, n):
            cand = list(order[:i]) + list(order[i:j + 1])[::-1] + list(order[j + 1:])
            if _cost(cand, points, start, closed, end) < base - IMPROVEMENT_EPS:
                return False
    return True


def _random_points(seed, n, dim=3):
    rng = random.Random(seed)
    pts = [tuple(rng.uniform(0.0, 100.0) for _ in range(dim)) for _ in range(n)]
    start = tuple(rng.uniform(0.0, 100.0) for _ in range(dim))
    return pts, start


def _wp(x, y, *devs):
    return ContactWaypoint(
        position=(float(x), float(y), 0.0),
        devices=tuple(DeviceID(d) for d in devs),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=0.0,
    )


# --------------------------------------------------------------------------- #
# 1. path_cost
# --------------------------------------------------------------------------- #

def test_path_cost_open_and_closed():
    pts = [(3.0, 0.0), (3.0, 4.0)]
    assert path_cost(pts, (0.0, 0.0)) == pytest.approx(7.0)
    assert path_cost(pts, (0.0, 0.0), closed=True) == pytest.approx(12.0)


def test_path_cost_is_three_dimensional():
    assert path_cost([(1.0, 2.0, 2.0)], (0.0, 0.0, 0.0)) == pytest.approx(3.0)


def test_mismatched_dimensions_raise_rather_than_truncate():
    with pytest.raises(ValueError):
        path_cost([(1.0, 2.0)], (0.0, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# 2. Known optima
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [5, 8, 12])
def test_regular_polygon_closed_tour_is_the_hull_order(n):
    """Start on vertex 0; the rest shuffled. Convex position => the optimal
    closed tour walks the hull, in one direction or the other."""
    verts = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n))
             for k in range(n)]
    start, rest = verts[0], verts[1:]
    perm = list(range(len(rest)))
    random.Random(n).shuffle(perm)
    pts = [rest[p] for p in perm]
    got = [perm[i] + 1 for i in best_order(pts, start, closed=True)]
    hull = list(range(1, n))
    assert got in (hull, hull[::-1])
    perimeter = n * 2 * math.sin(math.pi / n)
    assert _cost([perm.index(v - 1) for v in got], pts, start, True) == pytest.approx(perimeter)


def test_irregular_convex_polygon_closed_tour_is_the_hull_order():
    hull = [(0.0, 0.0), (4.0, -1.0), (9.0, 0.5), (11.0, 4.0), (8.0, 9.0),
            (3.0, 8.5), (-1.0, 5.0)]
    start, rest = hull[0], hull[1:]
    pts = [rest[i] for i in (3, 0, 5, 2, 4, 1)]
    got = [rest.index(pts[i]) for i in best_order(pts, start, closed=True)]
    assert got in ([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0])


def test_two_by_three_grid():
    """Unit grid, start at a corner: closed optimum 6 (the boundary), open
    optimum 5 (five unit edges, a Hamiltonian path)."""
    start = (0.0, 0.0)
    pts = [(2.0, 1.0), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (1.0, 1.0)]
    assert _cost(best_order(pts, start, closed=True), pts, start, True) == pytest.approx(6.0)
    assert _cost(best_order(pts, start, closed=False), pts, start, False) == pytest.approx(5.0)


def test_collinear_open_path_from_one_end_visits_in_line_order():
    start = (0.0, 0.0, 0.0)
    xs = [4.0, 1.0, 6.0, 3.0, 5.0, 2.0]
    pts = [(x, 0.0, 0.0) for x in xs]
    order = best_order(pts, start, closed=False)
    assert [xs[i] for i in order] == sorted(xs)
    assert _cost(order, pts, start, False) == pytest.approx(6.0)


def test_two_opt_untangles_a_crossing():
    """Square from a corner: the crossed order 1-3-2 becomes the perimeter."""
    start = (0.0, 0.0)
    pts = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    crossed = [0, 1, 2]                 # (1,0) -> (0,1) -> (1,1) crosses
    fixed = two_opt(crossed, pts, start, closed=True)
    assert _cost(fixed, pts, start, True) == pytest.approx(4.0)


def test_open_two_opt_can_swap_the_free_tail():
    """With the end free, reversing a tail segment is a move: from the origin,
    far-then-near becomes near-then-far."""
    start = (0.0, 0.0)
    pts = [(10.0, 0.0), (1.0, 0.0)]
    assert two_opt([0, 1], pts, start, closed=False) == [1, 0]


# --------------------------------------------------------------------------- #
# 3. Brute force, n <= 7
# --------------------------------------------------------------------------- #

BRUTE_SEEDS = list(range(12))


@pytest.mark.parametrize("closed", [True, False])
@pytest.mark.parametrize("seed", BRUTE_SEEDS)
def test_restarts_reach_the_brute_force_optimum(seed, closed):
    n = 3 + seed % 5                   # n in 3..7
    pts, start = _random_points(seed, n)
    opt = _brute(pts, start, closed)

    # Plain construction + 2-OPT: a local optimum, never better than optimal.
    local = two_opt(cheapest_insertion(pts, start, closed=closed), pts, start,
                    closed=closed)
    assert sorted(local) == list(range(n))
    assert _cost(local, pts, start, closed) >= opt - TOL
    assert _is_two_opt_local_optimum(local, pts, start, closed)

    # With restarts: the optimum.
    got = best_order(pts, start, closed=closed, restarts=20, seed=seed)
    assert sorted(got) == list(range(n))
    assert _cost(got, pts, start, closed) == pytest.approx(opt, abs=TOL)


#: Instances (seed, closed) on which cheapest insertion + one 2-OPT run stops
#: at a strictly suboptimal local optimum -- found by scanning seeds 0..299,
#: where this happened on 49 of 600 instances. They show why the paper
#: restarts 2-OPT from several initial routes.
TRAPPED = [(6, True), (57, True), (57, False), (58, False), (59, False)]


@pytest.mark.parametrize("seed,closed", TRAPPED)
def test_restarts_escape_a_two_opt_local_optimum(seed, closed):
    n = 3 + seed % 5
    pts, start = _random_points(seed, n)
    opt = _brute(pts, start, closed)
    local = two_opt(cheapest_insertion(pts, start, closed=closed), pts, start,
                    closed=closed)
    assert _is_two_opt_local_optimum(local, pts, start, closed)
    assert _cost(local, pts, start, closed) > opt + 1e-6
    got = best_order(pts, start, closed=closed, restarts=20, seed=seed)
    assert _cost(got, pts, start, closed) == pytest.approx(opt, abs=TOL)


@pytest.mark.parametrize("closed", [True, False])
def test_two_opt_from_a_random_order_is_a_local_optimum(closed):
    pts, start = _random_points(99, 7)
    order = list(range(7))
    random.Random(5).shuffle(order)
    out = two_opt(order, pts, start, closed=closed)
    assert sorted(out) == list(range(7))
    assert _is_two_opt_local_optimum(out, pts, start, closed)
    assert _cost(out, pts, start, closed) <= _cost(order, pts, start, closed) + TOL


@pytest.mark.parametrize("closed", [True, False])
def test_every_two_opt_pass_shortens_the_route(closed):
    """Each move is taken only if it shortens the route, so no pass can end
    longer than it began. This pins the within-pass bookkeeping: after a
    reversal the segment's first node changes, and pricing later moves off
    the stale node takes lengthening moves on some of these instances."""
    for seed in range(160):
        rng = random.Random(seed)
        n = rng.randint(4, 10)
        pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
        start = (rng.uniform(0, 100), rng.uniform(0, 100))
        order = list(range(n))
        rng.shuffle(order)
        prev = _cost(order, pts, start, closed)
        for passes in range(1, 6):
            out = two_opt(order, pts, start, closed=closed, max_passes=passes)
            cost = _cost(out, pts, start, closed)
            assert cost <= prev + TOL, (seed, passes)
            prev = cost


# --------------------------------------------------------------------------- #
# 3b. Fixed-end paths (start here, finish at the depot)
# --------------------------------------------------------------------------- #

def test_path_cost_with_a_fixed_end():
    pts = [(3.0, 0.0), (3.0, 4.0)]
    assert path_cost(pts, (0.0, 0.0), end=(0.0, 4.0)) == pytest.approx(10.0)
    # An empty route still has to reach the end.
    assert path_cost([], (0.0, 0.0), end=(3.0, 4.0)) == pytest.approx(5.0)
    # A fixed end at the start is the closed tour.
    assert path_cost(pts, (0.0, 0.0), end=(0.0, 0.0)) == path_cost(
        pts, (0.0, 0.0), closed=True)


@pytest.mark.parametrize("call", [
    lambda: path_cost([(1.0, 0.0)], (0.0, 0.0), closed=True, end=(1.0, 1.0)),
    lambda: cheapest_insertion([(1.0, 0.0)], (0.0, 0.0), closed=True, end=(1.0, 1.0)),
    lambda: two_opt([0], [(1.0, 0.0)], (0.0, 0.0), closed=True, end=(1.0, 1.0)),
    lambda: best_order([(1.0, 0.0)], (0.0, 0.0), closed=True, end=(1.0, 1.0)),
    lambda: best_order([], (0.0, 0.0), closed=True, end=(1.0, 1.0)),
])
def test_closed_and_end_together_are_rejected(call):
    with pytest.raises(ValueError):
        call()


def test_fixed_end_collinear_path_runs_toward_the_end():
    """Start at 0, end at 10: the stops in between are visited in line order."""
    start, end = (0.0, 0.0), (10.0, 0.0)
    xs = [7.0, 2.0, 5.0, 9.0]
    pts = [(x, 0.0) for x in xs]
    order = best_order(pts, start, end=end)
    assert [xs[i] for i in order] == sorted(xs)
    assert _cost(order, pts, start, False, end) == pytest.approx(10.0)


def test_fixed_end_two_opt_can_swap_two_stops():
    start, end = (0.0, 0.0), (10.0, 0.0)
    pts = [(9.0, 0.0), (1.0, 0.0)]
    assert two_opt([0, 1], pts, start, end=end) == [1, 0]


def test_fixed_end_differs_from_closed_and_open():
    """From the origin, A(10,0), B(10,10), C(-3,0). The closed tour is A,B,C
    either way round (39.40) and the open path C,A,B (26.00). With the end
    fixed at (10,-5), next to A: C,B,A then 5 m home is 3 + 16.40 + 10 + 5
    = 34.40, against A,B,C then home 10 + 10 + 16.40 + 13.93 = 50.33 and
    C,A,B then home 3 + 13 + 10 + 15 = 41.00."""
    start, depot = (0.0, 0.0), (10.0, -5.0)
    pts = [(10.0, 0.0), (10.0, 10.0), (-3.0, 0.0)]           # A, B, C
    fixed = best_order(pts, start, end=depot, restarts=5)
    assert fixed == [2, 1, 0]
    assert _cost(fixed, pts, start, False, depot) == pytest.approx(
        _brute(pts, start, False, depot))
    assert best_order(pts, start, closed=True, restarts=5) == [0, 1, 2]
    assert best_order(pts, start, closed=False, restarts=5) == [2, 0, 1]


@pytest.mark.parametrize("seed", BRUTE_SEEDS)
def test_fixed_end_restarts_reach_the_brute_force_optimum(seed):
    n = 3 + seed % 5
    pts, start = _random_points(seed, n)
    end = tuple(random.Random(1000 + seed).uniform(0.0, 100.0) for _ in range(3))
    opt = _brute(pts, start, False, end)
    local = two_opt(cheapest_insertion(pts, start, end=end), pts, start, end=end)
    assert sorted(local) == list(range(n))
    assert _cost(local, pts, start, False, end) >= opt - TOL
    assert _is_two_opt_local_optimum(local, pts, start, False, end)
    got = best_order(pts, start, end=end, restarts=20, seed=seed)
    assert sorted(got) == list(range(n))
    assert _cost(got, pts, start, False, end) == pytest.approx(opt, abs=TOL)


def test_fixed_end_at_the_start_is_the_canonical_closed_tour():
    pts, start = _random_points(11, 6)
    assert best_order(pts, start, end=start, restarts=5) == best_order(
        pts, start, closed=True, restarts=5)
    cs = [_wp(30, 0, "a"), _wp(0, 30, "b"), _wp(-30, 0, "c"), _wp(20, 20, "e")]
    home = (0.0, 0.0, 0.0)
    assert order_contacts(cs, home, end=home, restarts=3) == order_contacts(
        cs, home, closed=True, restarts=3)


def test_order_contacts_with_a_fixed_end():
    cs = [_wp(7, 0, "c"), _wp(2, 0, "a"), _wp(5, 0, "b")]
    out = order_contacts(cs, (0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0), restarts=3)
    assert [c.devices[0] for c in out] == ["a", "b", "c"]
    out = order_contacts(cs, (10.0, 0.0, 0.0), end=(0.0, 0.0, 0.0), restarts=3)
    assert [c.devices[0] for c in out] == ["c", "b", "a"]


# --------------------------------------------------------------------------- #
# 4. Constructions
# --------------------------------------------------------------------------- #

def test_nearest_neighbour_greedy_order():
    start = (0.0, 0.0)
    pts = [(5.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    assert nearest_neighbour(pts, start) == [1, 2, 0]


def test_nearest_neighbour_ties_go_to_the_lower_index():
    start = (0.0, 0.0)
    pts = [(0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)]
    assert nearest_neighbour(pts, start)[0] == 0


def test_cheapest_insertion_returns_a_permutation():
    pts, start = _random_points(3, 9)
    for closed in (True, False):
        assert sorted(cheapest_insertion(pts, start, closed=closed)) == list(range(9))


def test_two_opt_rejects_a_non_permutation():
    with pytest.raises(ValueError):
        two_opt([0, 0, 1], [(0.0,), (1.0,), (2.0,)], (0.0,))


def test_negative_restarts_rejected():
    with pytest.raises(ValueError):
        best_order([(1.0, 0.0)], (0.0, 0.0), restarts=-1)


# --------------------------------------------------------------------------- #
# 5. Determinism and degenerate inputs
# --------------------------------------------------------------------------- #

def test_best_order_is_deterministic_under_a_seed():
    pts, start = _random_points(7, 12)
    a = best_order(pts, start, closed=True, restarts=10, seed=3)
    b = best_order(pts, start, closed=True, restarts=10, seed=3)
    assert a == b


def test_best_order_does_not_touch_the_global_rng():
    random.seed(1234)
    expected = random.random()
    random.seed(1234)
    pts, start = _random_points(8, 8)
    best_order(pts, start, closed=True, restarts=5, seed=0)
    assert random.random() == expected


def test_closed_tour_orientation_is_canonical():
    pts, start = _random_points(11, 6)
    order = best_order(pts, start, closed=True, restarts=5)
    assert order[0] < order[-1]


def test_order_contacts_ignores_input_order():
    cs = [_wp(30, 0, "a"), _wp(0, 30, "b"), _wp(-30, 0, "c"), _wp(0, -30, "d"),
          _wp(20, 20, "e"), _wp(-20, 20, "f"), _wp(10, -25, "g")]
    ref = order_contacts(cs, (0.0, 0.0, 0.0), closed=True, restarts=5, seed=1)
    for s in range(5):
        shuffled = list(cs)
        random.Random(s).shuffle(shuffled)
        got = order_contacts(shuffled, (0.0, 0.0, 0.0), closed=True,
                             restarts=5, seed=1)
        assert got == ref
    assert sorted(ref, key=lambda c: c.devices) == sorted(cs, key=lambda c: c.devices)


def test_order_contacts_tie_breaks_co_located_contacts_by_devices():
    a, b = _wp(5, 5, "b"), _wp(5, 5, "a")
    out = order_contacts([a, b], (0.0, 0.0, 0.0), closed=False)
    assert [c.devices[0] for c in out] == ["a", "b"]


def test_empty_inputs():
    start = (0.0, 0.0, 0.0)
    assert path_cost([], start) == 0.0
    assert path_cost([], start, closed=True) == 0.0
    assert nearest_neighbour([], start) == []
    assert cheapest_insertion([], start, closed=True) == []
    assert two_opt([], [], start, closed=True) == []
    assert best_order([], start, closed=True, restarts=3) == []
    assert order_contacts([], start, closed=True, restarts=3) == []
    assert best_order([], start, end=(1.0, 0.0, 0.0), restarts=3) == []
    assert order_contacts([], start, end=(1.0, 0.0, 0.0)) == []


def test_single_point():
    start = (0.0, 0.0, 0.0)
    pts = [(3.0, 4.0, 0.0)]
    assert path_cost(pts, start) == pytest.approx(5.0)
    assert path_cost(pts, start, closed=True) == pytest.approx(10.0)
    assert nearest_neighbour(pts, start) == [0]
    assert cheapest_insertion(pts, start, closed=True) == [0]
    assert two_opt([0], pts, start, closed=True) == [0]
    assert best_order(pts, start, closed=True, restarts=3) == [0]
    assert path_cost(pts, start, end=(3.0, 0.0, 0.0)) == pytest.approx(9.0)
    assert best_order(pts, start, end=(3.0, 0.0, 0.0), restarts=3) == [0]
    wp = _wp(3, 4, "x")
    assert order_contacts([wp], start, closed=True) == [wp]
