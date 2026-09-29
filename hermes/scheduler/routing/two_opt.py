"""Tour construction and 2-OPT improvement over Euclidean points.

**What it is.** The routing primitive the SOTA baselines and the later FeRRy
phases share: given the mule's start pose and a set of stops, find a short
visiting order. Three shapes are supported, because the callers need them:

* **closed tour** (``closed=True``) -- start, every stop once, back to start.
  This is FedEx/CARP's inner problem (Bian et al., IEEE TMC 24(6), 2025,
  Sec. VI-B-1, eqs. 5-6): a transporter leaves the server, visits its clients
  and returns.
* **open path** (``closed=False``, no ``end``) -- start fixed, end free. This
  is the shape of a Pass-1 route that is re-planned mid-mission (Phase 3) and
  of the candidate routes a search scores (Phase 4).
* **fixed-end path** (``end=<point>``) -- start fixed, every stop once, then
  finish at ``end``. This is a closed tour whose vehicle does not start at
  the depot: a mule that is somewhere in the field and must fly home, the
  case FedEx's policy meets from HERMES's second mission on and the Phase-3
  re-plan meets once the mission clock returns the mule to its dock. A
  fixed end equal to the start *is* a closed tour, and :func:`best_order`
  treats it as one.

**Method.** Cheapest insertion builds a start order, then first-improvement
2-OPT removes crossings (every segment reversal is tried; an improving one is
applied at once and the scan continues). Optionally, ``restarts`` extra 2-OPT
runs from seeded random orders are tried and the shortest result is kept,
which is how the FedEx paper escapes 2-OPT's local optima ("run 2-OPT several
times from different initial routes"). The paper samples edge pairs at random
and stops "when no improvement can be made"; a deterministic full-neighbourhood
scan is the standard way to make that stopping test exact, and it is what the
tests need to be reproducible.

**Why it is dependency-light.** Only ``math`` and ``random``: this module is
imported by planning code that runs on the mule, and routing must not pull in
the ML stack. Positions are plain ``(x, y, z)`` tuples (any dimension works,
as long as all points share it); ``math.dist`` raises on a dimension mismatch
rather than silently truncating.

**Determinism.** Every function is a pure function of its inputs (and of
``seed`` where randomness is used). Ties are broken by index, and
:func:`order_contacts` first sorts contacts by ``(position, devices)`` so the
route does not depend on the order the caller listed them in. A closed tour
and its reverse cost the same, so :func:`best_order` returns the orientation
whose first stop has the lower index.

**Not optimal, and not claimed to be.** TSP is NP-hard; 2-OPT with restarts is
a heuristic that finds the optimum on small instances in practice (the unit
tests check it against brute force for n <= 7) but carries no guarantee.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, List, Optional, Sequence

if TYPE_CHECKING:  # typing only: keeps this module free of hermes imports
    from hermes.types import ContactWaypoint

Point = Sequence[float]

#: A 2-OPT move or a restart must shorten the route by more than this many
#: distance units (metres in HERMES) to count. Without a strict margin, moves
#: whose gain is float round-off (collinear points, mirrored tours) could be
#: taken back and forth; a nanometre is far below any real saving.
IMPROVEMENT_EPS = 1e-9

#: Safety cap on full 2-OPT passes. Each pass that changes the tour shortens
#: it by more than IMPROVEMENT_EPS, so the search always terminates; the cap
#: only bounds the worst case on pathological inputs.
DEFAULT_MAX_PASSES = 1000

# Distance matrix convention: rows/cols 0..n-1 are the points, row n is the
# start, and row n + 1 is the fixed end when there is one.
_Matrix = List[List[float]]


def _matrix(points: Sequence[Point], start: Point, end: Optional[Point] = None) -> _Matrix:
    nodes = list(points) + [start]
    if end is not None:
        nodes.append(end)
    return [[math.dist(a, b) for b in nodes] for a in nodes]


def _terminal(n: int, closed: bool, end: Optional[Point]) -> Optional[int]:
    """Matrix index the route must finish at, or None for a free end.

    Closed tours finish at the start (index n); fixed-end paths at the end
    node (index n + 1). One index lets construction, 2-OPT and costing treat
    the three route shapes with the same code.
    """
    if closed and end is not None:
        raise ValueError(
            "closed=True already ends the route at the start; pass closed or end, not both"
        )
    if closed:
        return n
    if end is not None:
        return n + 1
    return None


def _order_cost(order: Sequence[int], dist: _Matrix, s: int, term: Optional[int]) -> float:
    total = 0.0
    prev = s
    for j in order:
        total += dist[prev][j]
        prev = j
    if term is not None:
        total += dist[prev][term]
    return total


def _check_order(order: Sequence[int], n: int) -> None:
    if sorted(order) != list(range(n)):
        raise ValueError(
            f"order must be a permutation of range({n}), got {list(order)!r}"
        )


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #

def path_cost(
    points: Sequence[Point],
    start: Point,
    *,
    closed: bool = False,
    end: Optional[Point] = None,
) -> float:
    """Length of ``start -> points[0] -> ... -> points[-1]``, then ``-> start``
    if ``closed`` or ``-> end`` if ``end`` is given.

    ``points`` are already in visiting order. An empty route costs 0 (closed
    or open) or ``|start - end|`` (fixed end: the vehicle still has to get
    there). Passing both ``closed=True`` and ``end`` raises ``ValueError``.
    """
    _terminal(len(points), closed, end)
    total = 0.0
    prev = start
    for p in points:
        total += math.dist(prev, p)
        prev = p
    if closed and points:
        total += math.dist(prev, start)
    elif end is not None:
        total += math.dist(prev, end)
    return total


# --------------------------------------------------------------------------- #
# Constructions
# --------------------------------------------------------------------------- #

def _nearest_neighbour(dist: _Matrix) -> List[int]:
    n = len(dist) - 1
    unvisited = list(range(n))
    cur = n
    order: List[int] = []
    while unvisited:
        # (distance, index): the lower index wins an exact tie.
        nxt = min(unvisited, key=lambda j: (dist[cur][j], j))
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return order


def nearest_neighbour(points: Sequence[Point], start: Point) -> List[int]:
    """Greedy nearest-unvisited order from ``start``; ties go to the lower index."""
    return _nearest_neighbour(_matrix(points, start))


def _cheapest_insertion(dist: _Matrix, n: int, term: Optional[int]) -> List[int]:
    s = n
    route: List[int] = []
    remaining = list(range(n))
    while remaining:
        best: Optional[tuple] = None
        for j in remaining:
            for pos in range(len(route) + 1):
                prev = s if pos == 0 else route[pos - 1]
                if pos < len(route):
                    nxt: Optional[int] = route[pos]
                else:
                    nxt = term          # None: open path, appending at the free end
                if nxt is None:
                    delta = dist[prev][j]
                else:
                    delta = dist[prev][j] + dist[j][nxt] - dist[prev][nxt]
                # Tuple order is the tie-break: cheaper, then lower point
                # index, then the LATER position, so points that tie (e.g.
                # co-located stops) keep their canonical relative order.
                cand = (delta, j, -pos)
                if best is None or cand < best:
                    best = cand
        assert best is not None
        _, j, neg_pos = best
        route.insert(-neg_pos, j)
        remaining.remove(j)
    return route


def cheapest_insertion(
    points: Sequence[Point],
    start: Point,
    *,
    closed: bool = False,
    end: Optional[Point] = None,
) -> List[int]:
    """Insert, one at a time, the point whose cheapest insertion adds least length.

    For a closed tour every edge (including the return edge) is a candidate
    slot, and likewise for a fixed-end path (the last edge runs to ``end``);
    for an open path the free end is one extra slot costing only the edge
    into the new point. O(n^3), which is immaterial at mission sizes.
    """
    n = len(points)
    term = _terminal(n, closed, end)
    return _cheapest_insertion(_matrix(points, start, end), n, term)


# --------------------------------------------------------------------------- #
# 2-OPT
# --------------------------------------------------------------------------- #

def _two_opt(
    order: Sequence[int], dist: _Matrix, term: Optional[int], max_passes: int,
) -> List[int]:
    n = len(order)
    s = n
    tour = list(order)
    # Closed tours with < 3 stops have one tour up to reversal; open and
    # fixed-end paths with 2 stops can still gain by swapping them.
    if n < 2 or (term == s and n < 3):
        return tour
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = s if i == 0 else tour[i - 1]
            b = tour[i]
            d_ab = dist[a][b]
            for j in range(i + 1, n):
                c = tour[j]
                if j < n - 1:
                    d: Optional[int] = tour[j + 1]
                else:
                    d = term            # None: open path, the tail end is free
                # Reversing tour[i..j] replaces edges (a,b),(c,d) by (a,c),(b,d).
                if d is None:
                    delta = dist[a][c] - d_ab
                else:
                    delta = dist[a][c] + dist[b][d] - d_ab - dist[c][d]
                if delta < -IMPROVEMENT_EPS:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
                    b = tour[i]         # the segment now starts at old c
                    d_ab = dist[a][b]
        if not improved:
            break
    return tour


def two_opt(
    order: Sequence[int],
    points: Sequence[Point],
    start: Point,
    *,
    closed: bool = False,
    end: Optional[Point] = None,
    max_passes: int = DEFAULT_MAX_PASSES,
) -> List[int]:
    """First-improvement 2-OPT on an index ``order`` of ``points`` from ``start``.

    ``start`` is fixed as the first node. For ``closed=True`` the route returns
    to ``start`` and the return edge takes part in moves; with ``end`` the
    route finishes at ``end`` and the edge into it takes part likewise; for
    an open path the end is free, so reversing a tail segment (which removes
    one edge and adds one) is also a move. A move is taken only if it
    shortens the route by more than :data:`IMPROVEMENT_EPS`. Returns a 2-OPT
    local optimum (or the state after ``max_passes`` passes).
    """
    n = len(points)
    term = _terminal(n, closed, end)
    _check_order(order, n)
    if max_passes < 0:
        raise ValueError(f"max_passes must be >= 0, got {max_passes}")
    return _two_opt(order, _matrix(points, start, end), term, max_passes)


def best_order(
    points: Sequence[Point],
    start: Point,
    *,
    closed: bool = False,
    end: Optional[Point] = None,
    restarts: int = 0,
    seed: int = 0,
    max_passes: int = DEFAULT_MAX_PASSES,
) -> List[int]:
    """Cheapest insertion + 2-OPT, then ``restarts`` seeded random 2-OPT runs.

    The deterministic construction is always the first candidate; a random
    restart replaces it only if it is shorter by more than
    :data:`IMPROVEMENT_EPS`, so ties go to the earlier candidate. Restart
    orders come from ``random.Random(seed)``, never the global RNG.

    A fixed ``end`` equal to ``start`` is solved as the closed tour it is, so
    a caller that always passes its depot gets the canonical closed-tour
    orientation whenever the vehicle happens to be at the depot.
    """
    if restarts < 0:
        raise ValueError(f"restarts must be >= 0, got {restarts}")
    n = len(points)
    _terminal(n, closed, end)                   # rejects closed + end
    if end is not None and tuple(end) == tuple(start):
        closed, end = True, None
    if n == 0:
        return []
    term = _terminal(n, closed, end)
    dist = _matrix(points, start, end)
    best = _two_opt(_cheapest_insertion(dist, n, term), dist, term, max_passes)
    best_cost = _order_cost(best, dist, n, term)
    rng = random.Random(seed)
    for _ in range(restarts):
        perm = list(range(n))
        rng.shuffle(perm)
        cand = _two_opt(perm, dist, term, max_passes)
        cost = _order_cost(cand, dist, n, term)
        if cost < best_cost - IMPROVEMENT_EPS:
            best, best_cost = cand, cost
    # A closed tour and its reverse are the same tour; fix one orientation so
    # the output never depends on which direction the search happened to find.
    # (Open and fixed-end paths have a direction, so they are left alone.)
    if closed and n >= 2 and best[0] > best[-1]:
        best.reverse()
    return best


def order_contacts(
    contacts: Sequence["ContactWaypoint"],
    start_pose: Point,
    *,
    closed: bool = False,
    end: Optional[Point] = None,
    restarts: int = 0,
    seed: int = 0,
) -> List["ContactWaypoint"]:
    """Order contacts by :func:`best_order` over their positions.

    Contacts are first sorted by ``(position, devices)``, the same stable key
    S3b uses, so the route is a function of the contact *set* and not of the
    order the caller listed it in. Returns the contacts themselves (not
    copies), each exactly once.
    """
    canon = sorted(contacts, key=lambda c: (tuple(c.position), tuple(c.devices)))
    idx = best_order(
        [c.position for c in canon], start_pose,
        closed=closed, end=end, restarts=restarts, seed=seed,
    )
    return [canon[i] for i in idx]
