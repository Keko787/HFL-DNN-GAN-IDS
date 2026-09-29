"""FedEx-Async with CARP routing -- SOTA baseline (arm D4).

Ports **FedEx** ("Federated Learning via Model Express Delivery") from
J. Bian, C. Shen, M. Chen, J. Xu, "Indirect-Communication Federated Learning
via Mobile Transporters," *IEEE Transactions on Mobile Computing* 24(6),
pp. 4845-4857, June 2025, doi:10.1109/TMC.2025.3527405. That is the journal
extension of J. Bian, C. Shen, J. Xu, "Federated learning via indirect
server-client communications," Proc. CISS 2023; the journal adds
heterogeneous transporter speeds and an energy budget. Equation numbers below
are the journal's.

**What FedEx is.** One server (the depot) and N stationary clients with no
direct link between them. K mobile transporters partition the clients
(transporter k serves the set R_k, |R_k| = R_k) and each flies a closed tour
depot -> its clients -> depot, over and over. At every visit the client
uploads its *cumulative local update* (CLU) m_i -- everything it trained since
the previous visit -- and downloads the model the transporter carries. In
FedEx-Async (Sec. IV-E) a transporter never waits for the others: on each
return the server applies that transporter's CLUs at once and the transporter
leaves again with the new model.

**What CARP minimises.** FedEx-Async's convergence bound, Theorem 2 /
eq. (24), has the delay term (44 eta^2 G^2 L^2 / N) * sum_k R_k * Delta_k^2,
so the paper's "Client Assignment and Route Planning" (CARP, Sec. VI; the
string "CARD" does not occur in the paper) solves eq. (27)::

    min_a  sum_k R_k(a) * Delta_k(a)^2        (the "SWS" objective)

with Delta_k the round-trip time of eq. (9)::

    Delta_k = R_k * T_trans + T^k_SLF,   T^k_SLF = (closed tour length) / V_k

i.e. one transmission time per client plus straight-and-level flight at the
transporter's speed V_k. (Theorem 1 / eq. (18) is FedEx-*Sync*; its delay term
is ((T - Delta)/T) * 10 eta^2 G^2 L^2 Delta^2 with Delta = max_k Delta_k, i.e.
proportional to (max_k Delta_k)^2, which CARP's Sync objective eq. (25)
minimises through min_a max_k Delta_k -- squaring does not move the argmin.)
CARP is bi-level. The **inner** level (Sec. VI-B-1) is a TSP per transporter,
solved by 2-OPT with restarts; speed only rescales length, so the shortest
tour is the fastest for every V_k. The **outer** level (Sec. VI-B-2) is Gibbs
sampling over the assignment vector a: visit clients in a fixed order and
resample one client's transporter from eq. (30)::

    P(a_i = k) = exp(-C(k, a_-i) / q) / sum_{k' in S_i} exp(-C(k', a_-i) / q)

where C(k, a_-i) is the cost with client i moved to k and everything else
held, S_i the transporters that stay within their energy budget with i added
(eq. 29), and the temperature q decreases to 0.

**The merge -- documented here, implemented elsewhere.** FedEx's server rule
(eq. 14) is ``x <- x - (1/N) * u_k`` applied the moment transporter k returns,
where u_k = sum of m_i over the clients k collected on that tour (eq. 12, one
on-board accumulator) and m_i = eta * sum of the client's gradients since its
last visit = x_start - x_end. N is the TOTAL number of clients in the
deployment -- fixed; not the number collected, not the slice size. There is no
server learning rate (it is 1) and no staleness weighting, although each CLU
is up to 2 * Delta_k old when applied (eq. 21). If several transporters return
in the same slot their u_k are summed, still with 1/N (eq. 15). HERMES devices
report d_theta_i = theta_i - theta_base = -m_i, so the faithful rule is
``theta <- theta + (1/N) * sum_k sum_{i in collected_k} d_theta_i``: an
unweighted sum (eq. (1) weights clients 1/N, not by sample count), applied to
the CURRENT theta. On a transporter's first tour no client has a CLU yet
(inferred, not printed), so an empty collection must be a zero update, not an
error. That rule belongs to the merge-rule registry (``agg:fedex``) and is NOT
implemented in this module.

**Deviations of this port, and why.** Each is forced by our model or fills a
gap the paper leaves open; each must be stated wherever D4 is reported.

1. **One transporter per mission, so the policy is a visit-all 2-OPT tour.**
   A HERMES mission plans one mule, and the device-to-mule partition is made
   upstream by the cluster's mission slices, not by this policy. With K = 1
   the assignment is trivial, so :class:`FedExCarpPolicy` is exactly CARP's
   inner level: a 2-OPT tour over every contact, closed at the depot (see
   deviation 13 for what "the depot" is here). The outer Gibbs level
   (:func:`carp_assign`) is provided for the multi-mule phases and is not
   called by the policy.
2. **No energy gate by default.** The journal gates each Gibbs move on
   eq. (29); the simulator has no energy model, so D4 as run is
   "FedEx-Async/CARP without the energy gate", i.e. the conference version's
   unconstrained problem. :func:`carp_assign` accepts an optional ``budget``
   callable implementing eq. (29); default ``None`` makes every assignment
   feasible. Note that a plain time budget is eq. (29)'s special case only
   when the flight power equals hover plus transmit power: with the paper's
   ~30 W flight and ~20.1 W hover-plus-transmit, the gate is a *weighted*
   time constraint, 30 * T_SLF + 20.1 * R_k * T_trans <= E_budget.
3. **Contact-level stops, priced with S3b's model.** S1 (slice membership)
   and S3a (RF clustering) still run upstream -- they are physics, not
   policy. The policy's stops are therefore S3a contacts, each serving all of
   its clustered devices in one session, and the tour time is priced with the
   shared :class:`~hermes.scheduler.stages.s3b_feasibility.FeasibilityModel`
   (transit at cruise speed plus ``session_time_s`` per contact), so every arm
   faces the same physics. FedEx instead pays T_trans per client,
   sequentially (eq. 9); :func:`carp_cost` keeps the paper's per-client form.
4. **Seconds, not slots.** The paper states Delta_k in slots (one slot = one
   local step, or "one local epoch" = 1 min in Sec. VII) and never says how
   RTT is rounded; here Delta_k is in whatever units ``t_trans`` and
   distance / speed share (seconds in HERMES), unrounded. The argmin of
   eqs. (25)/(27) is invariant to rescaling time; the Gibbs probabilities are
   not, which is why the default temperature is derived from the cost
   differences themselves (deviation 5).
5. **Temperature schedule, and a greedy polish.** The paper requires only
   that q_l decreases to 0 and states no q_0, schedule, or iteration count.
   We use a geometric schedule held constant within a sweep (one sweep = one
   eq. (30) update per client): q_s = q0 * gamma^s, gamma =
   :data:`DEFAULT_GAMMA`, :data:`DEFAULT_SWEEPS` sweeps. q0 comes from the
   standard annealing initialisation: an average uphill single-client move
   from the initial assignment gets relative weight
   :data:`DEFAULT_INITIAL_ACCEPTANCE`, so q0 is on the scale of the cost
   differences whatever the time unit. After the schedule, the best state
   seen is descended greedily (single-client moves into S_i, strict
   improvement only) -- the schedule's q -> 0 limit, run to convergence;
   ``polish=False`` disables it. The paper claims convergence to the argmin
   "with probability 1"; classical annealing guarantees need logarithmic
   cooling, so a geometric schedule inherits no guarantee. It is a heuristic,
   and the tests check it against brute force.
6. **Sweep order and initial assignment.** "According to the pre-defined
   order" is unspecified; we sweep clients in sorted-id order. The initial
   assignment is unstated; we use round-robin over sorted ids unless the
   caller passes ``initial``.
7. **Best-so-far is returned**, not the last sample (unstated in the paper;
   it cannot make the result worse). Feasible assignments rank before
   infeasible ones when a budget is given.
8. **Eq. (30) as intended, not as printed.** The printed exponent has /q
   inside C(.) and the denominator sums over all K although only S_i is
   evaluated; we use exp(-C/q) normalised over S_i, shifted by min C for
   numerical stability (the probabilities are unchanged). q = 0 samples
   uniformly among the cheapest moves.
9. **Empty feasible set and empty transporters.** If S_i is empty the client
   keeps its current transporter; only the receiving transporter is checked,
   as in eq. (29). A transporter may end up with no clients, with
   Delta(empty) = 0. The paper states neither.
10. **2-OPT details.** The paper starts from a random route and samples edge
    pairs at random; we start from cheapest insertion and run deterministic
    first-improvement passes over the full neighbourhood, plus seeded random
    restarts (count unstated in the paper). Tour lengths are cached per client
    set, as the paper suggests.
11. **Deadlines are ignored and the tour is never truncated.** FedEx has no
    gates and no partial-tour mechanism, so the policy returns every contact
    and declares ``in_flight_check = IN_FLIGHT_NONE``: the mule flies the whole
    tour even when it overruns the mission budget. ``last_tour_fits`` and
    ``last_tour_overrun_s`` record whether it did. If the simulator hard-stops
    a mission at the budget elsewhere, that truncation is a deviation from
    FedEx to report alongside D4.
12. **Return leg, no depot time.** The tour time includes the return transit
    to the depot (FedEx's tours are closed) but no session there: eq. (9)
    does not count server-side time. S3b's own walk has no return-leg term,
    so ``last_tour_fits`` is a stricter test than S3b's admission; the
    ``*_without_return`` diagnostics give S3b's view of the same route.
13. **The depot, and a mule that is not at it.** FedEx's transporter starts
    every tour at the server. In today's simulator the mule's pose is the
    last stop of its previous mission: ``mule_main`` sets
    ``self.mule_pose = wp.position`` after every Pass-1 and Pass-2 contact
    and nothing resets it until FeRRy Phase 3 lands the mission clock; nor
    is the return leg ever flown. So from mission 2 on, ``env.mule_pose`` is
    a device location, not a depot. Two modes follow:

    * ``depot=None`` (the default): ``env.mule_pose`` is taken *as* the
      depot and the route is the closed tour around it. Exact in mission 1
      (the mule starts at the dock); from mission 2 on, the return leg in
      ``last_tour_cost_s`` / ``last_tour_fits`` prices a flight to a device
      location that is never flown, and because a closed tour costs the same
      in both directions, the direction -- hence the open prefix the mule
      actually flies and where its Pass 2 starts -- is fixed only by the
      router's canonical orientation, not by any physical reason.
    * ``depot=<pose>`` (the dock): the route is the shortest *fixed-end
      path* from ``env.mule_pose`` through every contact to ``depot``
      (:func:`~hermes.scheduler.routing.two_opt.best_order` with ``end``),
      and the return leg is priced to the dock. Identical to the closed tour
      whenever the mule is at the dock; otherwise it is FedEx's tour for a
      transporter that must first come home, and the depot, not the
      router's tie-break, picks the direction. This is the faithful mode
      and the one D4 should run with once the scheduler supplies the dock.

    Either way, ``last_return_leg_s``, ``last_fits_without_return`` and
    ``last_overrun_without_return_s`` report the route as the simulator
    flies it until Phase 3 (no return leg), next to the FedEx view.
14. **Local work.** A FedEx client trains continuously between visits
    (Delta_k local steps); a HERMES device trains a fixed amount per contact.
    That is a fidelity deviation of the FL process, outside this module, and
    it is not emulated here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from hermes.types import ContactWaypoint, DeviceID, DeviceSchedulerState

from hermes.scheduler.routing.two_opt import best_order, order_contacts, path_cost
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel

from .budget_walk import IN_FLIGHT_NONE

Point = Sequence[float]
MulePose = Tuple[float, float, float]

#: Eq. (29) as a callable: ``budget(k, clients, tour_length)`` is True when
#: transporter ``k`` can serve the client set ``clients`` (a closed tour of
#: ``tour_length`` distance units from the depot) within its energy budget.
BudgetFn = Callable[[int, FrozenSet[DeviceID], float], bool]

#: CARP objectives. ``async`` is eq. (27) (the paper's SWS, arm D4's),
#: ``sync`` is eq. (25) (Min-Max), ``total`` is the paper's Shortest-Total
#: comparison baseline, sum_k Delta_k.
OBJECTIVES = ("async", "sync", "total")

#: 2-OPT random restarts per tour. The paper says only "several"; 5-10 is the
#: usual range. The policy plans one tour per mission and can afford more; the
#: Gibbs loop prices thousands of client sets and uses fewer.
DEFAULT_TOUR_RESTARTS = 8
DEFAULT_CARP_TOUR_RESTARTS = 5

#: Annealing defaults (deviation 5). q0 is set so that an average uphill move
#: from the initial assignment has weight 0.8 against staying put; then
#: q_s = q0 * 0.975^s over 200 sweeps, ending near 0.006 * q0 -- effectively
#: greedy for the last sweeps -- followed by a greedy polish.
#:
#: How good that is, honestly. The setting was *tuned* against brute force on
#: 200 instances (``random.Random(seed)``, seeds 0-199, 4-8 clients, K = 2-3,
#: both the async and the sync objective): in-sample it found the async
#: optimum on all 200 and missed the sync optimum on 2, where 100 sweeps at
#: 0.95 missed 17 of 400 and a fixed q0 = 0.1 * C(a_0) froze the chain in its
#: initial local optimum. Those in-sample figures overstate it. On a fresh
#: family never used for tuning (``random.Random(10_000 + seed)``, seeds
#: 0-399, same ranges, exact brute force) it missed the async optimum on
#: 6/400 (1.5%) and the sync optimum on 11/400 (2.8%), 0.3% to 9.0% above
#: the optimum; on the 8 misses examined the tours were exact and the loss
#: was the *assignment*. The unit tests' brute-force seeds 0-9 come from the
#: tuning family;
#: the out-of-sample test pins the fresh one. So CARP here is a good
#: heuristic, not an exact solver; if it ever drives multi-mule assignment,
#: keeping the best of several independent chains is the cheap fix. A
#: 40-client, 4-transporter instance (the paper's size) takes about 5 s.
DEFAULT_INITIAL_ACCEPTANCE = 0.8
DEFAULT_GAMMA = 0.975
DEFAULT_SWEEPS = 200

# Relative margin for "strictly better" when tracking the best-so-far, so an
# exact tie (e.g. two transporters of equal speed swapping labels) never flips
# the incumbent on float round-off.
_REL_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Cost model: eqs. (9), (25), (27)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CarpCost:
    """Per-transporter round trips of one assignment, and the three objectives.

    Index k of every tuple is transporter k. ``tours`` gives each
    transporter's visiting order (closed, from and back to the depot).
    """

    counts: Tuple[int, ...]              # R_k
    tour_lengths: Tuple[float, ...]      # closed-tour length, distance units
    deltas: Tuple[float, ...]            # Delta_k, eq. (9)
    tours: Tuple[Tuple[DeviceID, ...], ...]
    async_cost: float                    # sum_k R_k Delta_k^2, eq. (27)
    sync_cost: float                     # max_k Delta_k, eq. (25)
    total_cost: float                    # sum_k Delta_k, Shortest-Total

    def objective(self, name: str) -> float:
        _check_objective(name)
        return {"async": self.async_cost, "sync": self.sync_cost,
                "total": self.total_cost}[name]


def _check_objective(name: str) -> None:
    if name not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {name!r}")


def _objective_value(name: str, counts: Sequence[int], deltas: Sequence[float]) -> float:
    # fsum is exactly rounded, so the value does not depend on the order of
    # transporters -- relabelling equal-speed transporters gives equal costs.
    if name == "async":
        return math.fsum(r * d * d for r, d in zip(counts, deltas))
    if name == "sync":
        return max(deltas) if deltas else 0.0
    return math.fsum(deltas)


class _CarpEvaluator:
    """Delta_k and tour lengths, with the inner 2-OPT cached per client set."""

    def __init__(
        self,
        positions: Mapping[DeviceID, Point],
        *,
        n_transporters: int,
        depot: Point,
        speeds: Optional[Sequence[float]],
        t_trans: float,
        tour_restarts: int,
        seed: int,
    ) -> None:
        if n_transporters < 1:
            raise ValueError(f"n_transporters must be >= 1, got {n_transporters}")
        if speeds is None:
            speeds = (1.0,) * n_transporters
        if len(speeds) != n_transporters:
            raise ValueError(
                f"speeds has {len(speeds)} entries for {n_transporters} transporters"
            )
        if any(v <= 0.0 for v in speeds):
            raise ValueError(f"every speed must be > 0, got {tuple(speeds)}")
        if t_trans < 0.0:
            raise ValueError(f"t_trans must be >= 0, got {t_trans}")
        self.positions = positions
        self.k = n_transporters
        self.depot = tuple(depot)
        self.speeds = tuple(float(v) for v in speeds)
        self.t_trans = float(t_trans)
        self.tour_restarts = tour_restarts
        self.seed = seed
        self._tours: Dict[FrozenSet[DeviceID], Tuple[float, Tuple[DeviceID, ...]]] = {}

    def tour(self, members: FrozenSet[DeviceID]) -> Tuple[float, Tuple[DeviceID, ...]]:
        """(closed-tour length, visiting order) for ``members``.

        A pure function of the set: members are put in a canonical order and
        the restart seed is fixed, so the cache never makes the result depend
        on the order in which sets were first priced.
        """
        hit = self._tours.get(members)
        if hit is not None:
            return hit
        if not members:
            out: Tuple[float, Tuple[DeviceID, ...]] = (0.0, ())
        else:
            ids = sorted(members, key=lambda d: (tuple(self.positions[d]), str(d)))
            pts = [self.positions[d] for d in ids]
            order = best_order(pts, self.depot, closed=True,
                               restarts=self.tour_restarts, seed=self.seed)
            length = path_cost([pts[j] for j in order], self.depot, closed=True)
            out = (length, tuple(ids[j] for j in order))
        self._tours[members] = out
        return out

    def delta(self, k: int, members: FrozenSet[DeviceID]) -> float:
        """Eq. (9): R_k * T_trans + tour length / V_k; 0 for an empty set."""
        if not members:
            return 0.0
        return len(members) * self.t_trans + self.tour(members)[0] / self.speeds[k]

    def cost(self, sets: Sequence[FrozenSet[DeviceID]]) -> CarpCost:
        counts = tuple(len(s) for s in sets)
        tours = [self.tour(s) for s in sets]
        deltas = tuple(self.delta(k, s) for k, s in enumerate(sets))
        return CarpCost(
            counts=counts,
            tour_lengths=tuple(t[0] for t in tours),
            deltas=deltas,
            tours=tuple(t[1] for t in tours),
            async_cost=_objective_value("async", counts, deltas),
            sync_cost=_objective_value("sync", counts, deltas),
            total_cost=_objective_value("total", counts, deltas),
        )

    def feasible(self, k: int, members: FrozenSet[DeviceID],
                 budget: Optional[BudgetFn]) -> bool:
        """Eq. (29) for one transporter; an empty set needs no energy."""
        if budget is None or not members:
            return True
        return bool(budget(k, members, self.tour(members)[0]))


def _sets_from(
    assignment: Mapping[DeviceID, int],
    positions: Mapping[DeviceID, Point],
    n_transporters: int,
) -> List[FrozenSet[DeviceID]]:
    if set(assignment) != set(positions):
        raise ValueError("assignment must cover exactly the clients in positions")
    sets: List[set] = [set() for _ in range(n_transporters)]
    for did, k in assignment.items():
        if not 0 <= k < n_transporters:
            raise ValueError(
                f"client {did!r} assigned to transporter {k}, "
                f"outside 0..{n_transporters - 1}"
            )
        sets[k].add(did)
    return [frozenset(s) for s in sets]


def carp_cost(
    assignment: Mapping[DeviceID, int],
    positions: Mapping[DeviceID, Point],
    *,
    n_transporters: int,
    depot: Point = (0.0, 0.0, 0.0),
    speeds: Optional[Sequence[float]] = None,
    t_trans: float = 1.0,
    tour_restarts: int = DEFAULT_CARP_TOUR_RESTARTS,
    seed: int = 0,
) -> CarpCost:
    """Evaluate an assignment (client -> transporter index 0..K-1).

    Each transporter's Delta_k is eq. (9) over its 2-OPT closed tour from
    ``depot``; ``speeds`` defaults to 1 for every transporter. The result
    carries the async cost sum_k R_k Delta_k^2 (eq. 27), the sync cost
    max_k Delta_k (eq. 25) and the Shortest-Total sum_k Delta_k.
    """
    ev = _CarpEvaluator(positions, n_transporters=n_transporters, depot=depot,
                        speeds=speeds, t_trans=t_trans,
                        tour_restarts=tour_restarts, seed=seed)
    return ev.cost(_sets_from(assignment, positions, n_transporters))


# --------------------------------------------------------------------------- #
# Outer level: Gibbs sampling, eq. (30)
# --------------------------------------------------------------------------- #

def _conditional(
    ev: _CarpEvaluator,
    client: DeviceID,
    current: int,
    sets: Sequence[FrozenSet[DeviceID]],
    deltas: Sequence[float],
    *,
    q: float,
    objective: str,
    budget: Optional[BudgetFn],
) -> List[Tuple[int, float, float]]:
    """Eq. (30) for one client: [(k, C(k, a_-i), P(a_i = k))] over S_i.

    Moving the client changes only two tours (donor and receiver), so only
    those two Delta values are recomputed.
    """
    counts = [len(s) for s in sets]
    donor = sets[current] - {client}
    d_donor = ev.delta(current, donor)
    cands: List[Tuple[int, float]] = []
    for k in range(ev.k):
        recv = sets[k] | {client}
        if not ev.feasible(k, recv, budget):
            continue                    # eq. (29): k is not in S_i
        if k == current:
            c = _objective_value(objective, counts, deltas)
        else:
            new_deltas = list(deltas)
            new_counts = list(counts)
            new_deltas[current], new_counts[current] = d_donor, len(donor)
            new_deltas[k], new_counts[k] = ev.delta(k, recv), len(recv)
            c = _objective_value(objective, new_counts, new_deltas)
        cands.append((k, c))
    if not cands:
        # Empty S_i is unspecified in the paper (deviation 9): stay put.
        return [(current, _objective_value(objective, counts, deltas), 1.0)]
    c_min = min(c for _, c in cands)
    if q > 0.0:
        # exp(-(C - min C)/q): same probabilities as exp(-C/q), no underflow.
        weights = [math.exp(-(c - c_min) / q) for _, c in cands]
    else:
        weights = [1.0 if c == c_min else 0.0 for _, c in cands]
    z = math.fsum(weights)
    return [(k, c, w / z) for (k, c), w in zip(cands, weights)]


def gibbs_conditional(
    client: DeviceID,
    assignment: Mapping[DeviceID, int],
    positions: Mapping[DeviceID, Point],
    *,
    n_transporters: int,
    q: float,
    objective: str = "async",
    depot: Point = (0.0, 0.0, 0.0),
    speeds: Optional[Sequence[float]] = None,
    t_trans: float = 1.0,
    budget: Optional[BudgetFn] = None,
    tour_restarts: int = DEFAULT_CARP_TOUR_RESTARTS,
    seed: int = 0,
) -> Dict[int, float]:
    """The sampling distribution of eq. (30) for ``client``, as {k: P(a_i = k)}.

    Transporters outside the feasible set S_i (eq. 29) are absent. If S_i is
    empty the client keeps its transporter with probability 1.
    """
    _check_objective(objective)
    if q < 0.0:
        raise ValueError(f"q must be >= 0, got {q}")
    ev = _CarpEvaluator(positions, n_transporters=n_transporters, depot=depot,
                        speeds=speeds, t_trans=t_trans,
                        tour_restarts=tour_restarts, seed=seed)
    sets = _sets_from(assignment, positions, n_transporters)
    deltas = [ev.delta(k, s) for k, s in enumerate(sets)]
    rows = _conditional(ev, client, assignment[client], sets, deltas,
                        q=q, objective=objective, budget=budget)
    return {k: p for k, _, p in rows}


@dataclass(frozen=True)
class CarpResult:
    """Outcome of :func:`carp_search`."""

    assignment: Dict[DeviceID, int]      # best assignment seen
    cost: CarpCost                       # its evaluation
    feasible: bool                       # every transporter passes eq. (29)
    initial_cost: CarpCost               # the starting assignment's evaluation
    updates: int                         # single-client Gibbs updates run


def _move(
    ev: _CarpEvaluator,
    sets: Sequence[FrozenSet[DeviceID]],
    deltas: List[float],
    client: DeviceID,
    k_from: int,
    k_to: int,
) -> List[FrozenSet[DeviceID]]:
    """Move ``client`` between transporters; updates ``deltas`` in place."""
    out = list(sets)
    out[k_from] = sets[k_from] - {client}
    out[k_to] = sets[k_to] | {client}
    deltas[k_from] = ev.delta(k_from, out[k_from])
    deltas[k_to] = ev.delta(k_to, out[k_to])
    return out


def _initial_temperature(
    ev: _CarpEvaluator,
    clients: Sequence[DeviceID],
    assign: Mapping[DeviceID, int],
    sets: Sequence[FrozenSet[DeviceID]],
    deltas: Sequence[float],
    *,
    objective: str,
    budget: Optional[BudgetFn],
    acceptance: float,
) -> float:
    """q0 at which an average uphill single-client move from the start has
    Boltzmann weight ``acceptance`` relative to staying put.

    The standard simulated-annealing initialisation (Kirkpatrick, Gelatt,
    Vecchi, Science 1983; Johnson et al., Oper. Res. 1989): the temperature
    is set from the cost *differences* the chain will face, so it suits any
    objective and any unit of time. A fixed fraction of C(a_0) was tried
    first and froze the chain -- SWS differences are often a large share of
    C, so the sampler never left the initial local optimum.
    """
    base = _objective_value(objective, [len(s) for s in sets], deltas)
    ups: List[float] = []
    downs: List[float] = []
    for client in clients:
        k0 = assign[client]
        for k, c, _p in _conditional(ev, client, k0, sets, deltas, q=0.0,
                                     objective=objective, budget=budget):
            if k == k0:
                continue
            if c > base:
                ups.append(c - base)
            elif c < base:
                downs.append(base - c)
    diffs = ups or downs
    if not diffs:
        return 0.0                      # every move is neutral: nothing to anneal
    return (math.fsum(diffs) / len(diffs)) / math.log(1.0 / acceptance)


def carp_search(
    positions: Mapping[DeviceID, Point],
    *,
    n_transporters: int,
    depot: Point = (0.0, 0.0, 0.0),
    speeds: Optional[Sequence[float]] = None,
    t_trans: float = 1.0,
    objective: str = "async",
    sweeps: int = DEFAULT_SWEEPS,
    q0: Optional[float] = None,
    gamma: float = DEFAULT_GAMMA,
    seed: int = 0,
    budget: Optional[BudgetFn] = None,
    initial: Optional[Mapping[DeviceID, int]] = None,
    tour_restarts: int = DEFAULT_CARP_TOUR_RESTARTS,
    polish: bool = True,
) -> CarpResult:
    """CARP's outer level: annealed Gibbs sampling over client assignments.

    One *sweep* resamples every client once, in sorted-id order, from
    eq. (30) at temperature q_s = q0 * gamma^s (deviation 5). ``q0=None``
    derives q0 from the start's uphill move costs
    (:func:`_initial_temperature`, acceptance
    :data:`DEFAULT_INITIAL_ACCEPTANCE`). With ``polish`` the best assignment
    seen is then descended greedily -- the schedule's q -> 0 limit -- until
    no single-client move into S_i lowers the cost.

    All randomness comes from ``random.Random(seed)``; the inner tours use
    the same ``seed`` for their restarts, so the result is a pure function of
    the arguments. Returns the best assignment seen (feasible before
    infeasible, then lower cost; the earlier one on a tie).
    """
    _check_objective(objective)
    if sweeps < 0:
        raise ValueError(f"sweeps must be >= 0, got {sweeps}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if q0 is not None and q0 < 0.0:
        raise ValueError(f"q0 must be >= 0, got {q0}")

    ev = _CarpEvaluator(positions, n_transporters=n_transporters, depot=depot,
                        speeds=speeds, t_trans=t_trans,
                        tour_restarts=tour_restarts, seed=seed)
    clients = sorted(positions, key=str)
    if initial is None:
        assign = {c: idx % n_transporters for idx, c in enumerate(clients)}
    else:
        assign = dict(initial)
    sets = _sets_from(assign, positions, n_transporters)
    deltas = [ev.delta(k, s) for k, s in enumerate(sets)]
    initial_cost = ev.cost(sets)

    def _rank(cur_sets: Sequence[FrozenSet[DeviceID]], cost: float) -> Tuple[bool, float]:
        ok = all(ev.feasible(k, s, budget) for k, s in enumerate(cur_sets))
        return (not ok, cost)

    def _better(a: Tuple[bool, float], b: Tuple[bool, float]) -> bool:
        if a[0] != b[0]:
            return a[0] < b[0]
        return a[1] < b[1] - _REL_EPS * max(1.0, abs(b[1]))

    cur = _objective_value(objective, [len(s) for s in sets], deltas)
    best_rank = _rank(sets, cur)
    best_assign = dict(assign)
    rng = random.Random(seed)
    updates = 0

    # With one transporter (or no clients) there is only one assignment.
    if n_transporters > 1 and clients:
        temp0 = float(q0) if q0 is not None else _initial_temperature(
            ev, clients, assign, sets, deltas, objective=objective,
            budget=budget, acceptance=DEFAULT_INITIAL_ACCEPTANCE,
        )
        for s_idx in range(sweeps):
            q = temp0 * gamma ** s_idx
            for client in clients:
                k0 = assign[client]
                rows = _conditional(ev, client, k0, sets, deltas,
                                    q=q, objective=objective, budget=budget)
                u = rng.random()
                acc = 0.0
                # Falls back to the last row if round-off leaves sum(p) < u.
                k_new, c_new = rows[-1][0], rows[-1][1]
                for k, c, p in rows:
                    acc += p
                    if u < acc:
                        k_new, c_new = k, c
                        break
                updates += 1
                if k_new != k0:
                    sets = _move(ev, sets, deltas, client, k0, k_new)
                    assign[client] = k_new
                    cur = c_new
                rank = _rank(sets, cur)
                if _better(rank, best_rank):
                    best_rank, best_assign = rank, dict(assign)

        if polish:
            # Zero-temperature descent from the best state. A client moves
            # only to the cheapest transporter in S_i (lowest k on a tie) and
            # only if that strictly lowers the cost, so the loop terminates
            # and is deterministic.
            assign = dict(best_assign)
            sets = _sets_from(assign, positions, n_transporters)
            deltas = [ev.delta(k, s) for k, s in enumerate(sets)]
            cur = _objective_value(objective, [len(s) for s in sets], deltas)
            improved = True
            while improved:
                improved = False
                for client in clients:
                    k0 = assign[client]
                    rows = _conditional(ev, client, k0, sets, deltas,
                                        q=0.0, objective=objective, budget=budget)
                    c_best, k_best = min((c, k) for k, c, _p in rows)
                    if c_best < cur - _REL_EPS * max(1.0, abs(cur)):
                        sets = _move(ev, sets, deltas, client, k0, k_best)
                        assign[client] = k_best
                        cur = c_best
                        improved = True
            rank = _rank(sets, cur)
            if _better(rank, best_rank):
                best_rank, best_assign = rank, dict(assign)

    best_sets = _sets_from(best_assign, positions, n_transporters)
    return CarpResult(
        assignment=best_assign,
        cost=ev.cost(best_sets),
        feasible=not best_rank[0],
        initial_cost=initial_cost,
        updates=updates,
    )


def carp_assign(
    positions: Mapping[DeviceID, Point],
    *,
    n_transporters: int,
    depot: Point = (0.0, 0.0, 0.0),
    speeds: Optional[Sequence[float]] = None,
    t_trans: float = 1.0,
    objective: str = "async",
    sweeps: int = DEFAULT_SWEEPS,
    q0: Optional[float] = None,
    gamma: float = DEFAULT_GAMMA,
    seed: int = 0,
    budget: Optional[BudgetFn] = None,
    initial: Optional[Mapping[DeviceID, int]] = None,
    tour_restarts: int = DEFAULT_CARP_TOUR_RESTARTS,
) -> Dict[DeviceID, int]:
    """CARP's client assignment: {client: transporter index 0..K-1}.

    Thin wrapper over :func:`carp_search`; see it for the parameters. The
    default objective is FedEx-Async's eq. (27). Use :func:`carp_cost` on the
    result for the per-transporter tours and round trips.
    """
    return carp_search(
        positions, n_transporters=n_transporters, depot=depot, speeds=speeds,
        t_trans=t_trans, objective=objective, sweeps=sweeps, q0=q0,
        gamma=gamma, seed=seed, budget=budget, initial=initial,
        tour_restarts=tour_restarts,
    ).assignment


# --------------------------------------------------------------------------- #
# The single-mule arm
# --------------------------------------------------------------------------- #

def _tour_parts_s(
    route: Sequence[ContactWaypoint],
    start: MulePose,
    end: MulePose,
    model: FeasibilityModel,
) -> Tuple[float, float]:
    """(outbound, return leg) seconds: every stop's transit + session from
    ``start``, then the transit from the last stop to ``end``."""
    outbound = 0.0
    pose: Sequence[float] = start
    for wp in route:
        _transit, leg = model.cost(tuple(pose), wp.position)
        outbound += leg
        pose = wp.position
    back = 0.0
    if route:
        back, _ = model.cost(tuple(pose), tuple(end))
    return outbound, back


def tour_time_s(
    route: Sequence[ContactWaypoint],
    start: MulePose,
    model: FeasibilityModel,
    *,
    end: Optional[MulePose] = None,
) -> float:
    """Time to fly ``route`` from ``start`` and home, priced with S3b's model.

    Home is ``end`` (the depot) when given, else ``start``. Each stop costs
    ``model.cost``'s total (transit + session); the return leg costs its
    transit only, since there is no session at the depot and eq. (9) counts
    no server-side time. An empty route costs 0: with nothing to visit the
    mule does not take off.
    """
    outbound, back = _tour_parts_s(route, start, start if end is None else end, model)
    return outbound + back


class FedExCarpPolicy:
    """Visit every contact on a 2-OPT tour that ends at the depot.

    CARP's inner level for the single transporter a HERMES mission plans
    (module deviation 1). A whole-scheduler baseline: it exposes
    ``admit_and_order``, so the scheduler hands it S3/S3b/S3.5 entirely, and
    like FedEx it has no gates -- every contact is admitted.

    ``depot`` is the dock the tour returns to. ``None`` (default) takes the
    mule's pose at planning time as the depot, i.e. a closed tour around
    ``env.mule_pose``; a pose makes the route the shortest path from
    ``env.mule_pose`` through every contact to that dock (deviation 13).
    """

    name = "FEDEX"
    # Freeze Amendment 8 -- what the mule re-checks before each contact in
    # flight: nothing. FedEx has no partial-tour mechanism, so the planned
    # tour is flown to the end even past the mission budget (deviation 11).
    in_flight_check = IN_FLIGHT_NONE

    def __init__(
        self,
        *,
        restarts: int = DEFAULT_TOUR_RESTARTS,
        seed: int = 0,
        depot: Optional[MulePose] = None,
    ) -> None:
        if restarts < 0:
            raise ValueError(f"restarts must be >= 0, got {restarts}")
        self.restarts = int(restarts)
        self.seed = int(seed)
        if depot is not None and len(depot) != 3:
            raise ValueError(f"depot must be an (x, y, z) pose, got {depot!r}")
        self.depot: Optional[MulePose] = (
            None if depot is None
            else (float(depot[0]), float(depot[1]), float(depot[2]))
        )
        #: Diagnostics of the most recent plan, reset on every call (the same
        #: pattern as ``FLScheduler.last_feasibility``). Tour time in seconds,
        #: including the return leg to the depot; whether ``now + tour`` fits
        #: the mission deadline (None when no deadline is enforced); and by
        #: how much it overruns (0.0 when it fits, None without a deadline).
        self.last_tour_cost_s: Optional[float] = None
        self.last_tour_fits: Optional[bool] = None
        self.last_tour_overrun_s: Optional[float] = None
        #: The same route as the simulator flies it until FeRRy Phase 3
        #: (deviation 13): the return leg's transit in seconds, and the fit
        #: and overrun of ``now + tour - return leg``. S3b's walk prices this.
        self.last_return_leg_s: Optional[float] = None
        self.last_fits_without_return: Optional[bool] = None
        self.last_overrun_without_return_s: Optional[float] = None

    def admit_and_order(
        self,
        contacts: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        mission_deadline_ts: Optional[float] = None,
        feasibility_model: Optional[FeasibilityModel] = None,
    ) -> List[ContactWaypoint]:
        """FedEx as a **complete scheduler**: every contact, shortest tour home.

        ``device_states`` is not read -- FedEx's route depends only on where
        its clients are -- and never mutated. Deadlines and buckets are
        ignored; ``mission_deadline_ts`` only feeds the ``last_tour_fits``
        diagnostics and never removes a contact.
        """
        self.last_tour_cost_s = None
        self.last_tour_fits = None
        self.last_tour_overrun_s = None
        self.last_return_leg_s = None
        self.last_fits_without_return = None
        self.last_overrun_without_return_s = None

        start = env.mule_pose
        if self.depot is None:
            home = start
            route = order_contacts(contacts, start, closed=True,
                                   restarts=self.restarts, seed=self.seed)
        else:
            # A fixed end equal to the start is solved as the closed tour,
            # so a mule sitting at its dock gets exactly the depot=None route.
            home = self.depot
            route = order_contacts(contacts, start, end=home,
                                   restarts=self.restarts, seed=self.seed)
        model = feasibility_model or FeasibilityModel()
        outbound, back = _tour_parts_s(route, start, home, model)
        self.last_tour_cost_s = outbound + back
        self.last_return_leg_s = back
        if mission_deadline_ts is not None:
            finish = env.now + self.last_tour_cost_s
            self.last_tour_fits = finish <= mission_deadline_ts
            self.last_tour_overrun_s = max(0.0, finish - mission_deadline_ts)
            flown = env.now + outbound
            self.last_fits_without_return = flown <= mission_deadline_ts
            self.last_overrun_without_return_s = max(0.0, flown - mission_deadline_ts)
        return route
