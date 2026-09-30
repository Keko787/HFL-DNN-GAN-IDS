"""FedCS client selection, degraded to a data mule — SOTA baseline (arm D5).

Ports Algorithm 3, "Client Selection in Protocol 2", of **FedCS**: T. Nishio and
R. Yonetani, *Client Selection for Federated Learning with Heterogeneous
Resources in Mobile Edge*, Proc. IEEE ICC 2019, Shanghai, pp. 1-7,
doi:10.1109/ICC.2019.8761315 (arXiv:1804.08333v2; Algorithm 3 and Eqs. (1)-(4)
in Sec. III-C, p. 4; Protocol 2 on p. 3; T_cs = T_agg = 0 in Sec. IV-A, p. 5).

Not to be confused with *Decision* D5 (the age unit and the merge cutoff
``a_max_j = floor(Phi_j * s / T)`` in ``mule_main._age_caps``). This is the
optional Phase-2 *arm* D5 of the FeRRy build plan: "the largest set that fits
one round deadline, greedily, with last-known state in place of FedCS's
Resource Request", labelled degraded. Its purpose in the comparison is
admission by **one round deadline** against our **per-device deadlines**.

**What the paper does.** Each round the operator polls ceil(K*C) random clients
(the Resource Request; clients report resource information "such as" channel
state, compute capacity and data size), then picks the largest ordered set S
(the upload order) that finishes inside a single round deadline::

    max_S |S|  s.t.  T_round >= T_cs + T^d_S + Theta_|S| + T_agg         (Eq. 4)
    Theta_i = Theta_{i-1} + t^UL_{k_i} + max{0, t^UD_{k_i} - Theta_{i-1}}
                                        (Eqs. 1-3, in incremental form)

Algorithm 3 solves it greedily: take
``x = argmax_k 1 / (T^d_{S+k} - T^d_S + t^UL_k + max{0, t^UD_k - Theta})``
over the remaining candidates (the reciprocal of the marginal round time),
remove x **unconditionally** (line 4), and append it to S only if
``T_cs + T^d_{S+x} + Theta' + T_agg < T_round`` (line 7). The argmax is
recomputed every iteration because Theta and S change.

**The port.** From ``env.mule_pose`` and ``env.now``, repeatedly price every
remaining contact with the shared S3b cost model,
``(transit_c, total_c) = FeasibilityModel.cost(pose, c.position)``, take the one
with the best ``value(c) / total_c``, remove it, and admit it if
``clock + total_c <= mission_deadline_ts``, advancing pose and clock only on
admission. With ``mission_deadline_ts=None`` the same loop admits everything,
so the arm still has an order (the D1/D2 convention).

Term mapping, paper -> mule:

* round -> one mission's Pass 1 (``MissionPass.COLLECT``); the ordered S -> the
  returned route (visit order stands in for upload order);
* K' -> the ``contacts`` argument: S3a ``ContactWaypoint``s over S1-eligible
  devices;
* Resource Request -> the scheduler's last-known state (deviation 1);
* t^UL_k -> ``FeasibilityModel.session_time_s``, a constant per contact;
* t^UD_k -> 0, so ``max{0, t^UD_k - Theta}`` -> 0 (deviation 3);
* T^d_{S+k} - T^d_S -> ``transit_c``, the straight-line flight time from the
  current end of the route (deviation 4); so T^d_S -> the summed transits of
  the route so far;
* Theta -> ``|route| * session_time_s`` (Eqs. 1-3 with t^UD = 0), so the
  line-3 denominator is ``total_c = transit_c + session_time_s``, and
  T^d_S + Theta -> ``clock - env.now``;
* line 6's t (with T_cs = T_agg = 0) -> ``(clock + total_c) - env.now``;
  line 7 compares it with T_round, i.e. admits on
  ``clock + total_c <= mission_deadline_ts`` (deviation 5);
* T_cs, T_agg -> 0, as in the paper's own experiments (S3b has no return leg);
* T_round -> the *remaining* mission budget, ``mission_deadline_ts - env.now``,
  where ``mission_deadline_ts = mission_start_ts + mission_budget_s``; the
  scheduler may plan after the mission started, so it is not
  ``mission_budget_s`` itself (deviation 6).

**Under the ferry predicate (FeRRy Phase 3).** The mapping above is the legacy
model's (``FeasibilityModel.ferry`` None). With the ferry model the arm is
priced with the same physics as every other arm (design §3.2): the selection
key's ``total_c`` becomes the leg's marginal time ``transit_c + dwell_c``,
where t^UL_k is the predicted dwell ``Σ bytes / rate`` over the contact's
reachable members rather than a constant, and admission is S3b's single
predicate under the budget rule, which also requires the mule to fly home
from the contact and upload before the budget ends. That return leg plus the
upload plays the part of T_agg (bringing the updates to the server), so
"T_cs = T_agg = 0 (S3b has no return leg)" holds for the legacy model only,
and the energy clause applies when a capacity is set.

Deviations from the paper, each with its reason:

1. **No Resource Request; last-known state instead.** A mule cannot poll a
   device before flying to it, which is why the build plan labels this arm
   degraded. Of the reported items, channel state has no per-device field
   (``last_beacon_ts`` is a timestamp and ``SelectorEnv.rf_prior_snr_db`` is
   mule-wide), and compute capacity has none either, so both drop out, as
   Oort's system-speed term does in ``oort.py``. Data size exists
   (``last_num_examples``), but without a compute rate it cannot become
   t^UD, and inventing a rate would be worse than omitting it. The only
   last-known per-device state that enters the cost is position
   (``last_known_position``, carried by S3a as ``ContactWaypoint.position``).
   ``device_states`` is therefore accepted for the interface and not read.
2. **C = 1.** Every S3a contact is a candidate; there is no random ceil(K*C)
   subsample (the paper uses C = 0.1, following McMahan et al.). A random
   subsample here would add a handicap that no other arm carries; that
   reasoning is ours, not the paper's.
3. **t^UD := 0.** Training runs offline between visits and Pass 1 pulls a
   pre-prepared update, so there is no update time for an upload to hide
   behind and the overlap term vanishes.
4. **The distribution increment becomes transit — an analogy, not a
   structural match.** Sec. III-C only requires that T^d_S depend on S. In
   the paper's experiments (Sec. IV-A, p. 5), T^d_S = D_m / min_{k in S}
   {theta_k}, a bottleneck over the whole set: order-independent, and its
   increment is 0 unless k has the worst channel so far. Transit adds up hop
   by hop and depends on pose and order. The greedy key is still the marginal
   elapsed time, so Algorithm 3 itself is unchanged; only its reading
   differs. The model's distribution is our Pass 2, walked separately.
5. **Admission is ``<=``, not ``<``.** Algorithm 3 line 7 is strict, while
   Eq. (4) itself uses ``>=`` (the paper is inconsistent at the boundary). We
   admit on ``clock + total <= mission_deadline_ts`` to match
   ``greedy_budget_walk`` and S3b, so every arm prices the budget identically.
6. **T_round is the remaining budget** (see the mapping above).
7. **Selection value.** ``value='unit'`` (the default) is the letter of line 3:
   value 1 per candidate, so the argmax of ``1/total`` is the argmin of the
   marginal time and is implemented as exactly that. S3a may put several
   devices in one contact, while the objective of Eq. (4) counts *clients*;
   ``value='devices'`` scores ``len(c.devices) / total_c``, the knapsack
   reading of that objective. This option is a port choice, not the paper's.
   The two coincide when every contact holds one device.
8. **Tie-break.** The paper states none. Ties go to the smaller
   ``(position, devices)``, the same tie-break S3b uses, so the route does not
   depend on input order.
9. **No per-device deadline and no bucket priority.** FedCS has one round
   deadline, not per-client deadlines; ``wp.deadline_ts`` and ``wp.bucket``
   are never read. That is the contrast the arm exists to isolate.
10. **Late arrivals.** The paper does not say what FedCS does with a selected
    client that overruns T_round when its estimates were wrong. Here the mule
    re-checks the mission budget (and only the budget) before each contact in
    flight, from its actual pose and clock (``in_flight_check =
    IN_FLIGHT_BUDGET``).

**Skip versus stop.** Line 4 removes the pick whether or not it is admitted,
and the loop continues; this port does the same. For the ``unit`` key under
the legacy model it makes no difference: the pick has the smallest ``total``
of all remaining contacts, and a rejection leaves pose and clock unchanged, so
every remaining contact costs at least as much and fails too. Skip then
returns the same route as stopping at the first miss, consistent with the
paper's stated O(|K'||S|). For the ``devices`` key a rejected dense far
contact can be followed by a sparse near one that fits, so skipping matters
there. Under the ferry predicate the equivalence no longer holds for either
key: admission tests the time back home, not the marginal time the key ranks
on, so a contact with a larger ``total`` that lies nearer the dock can fit
after a nearer-to-the-mule one fails. The loop is O(|K'|^2) cost evaluations,
trivial at our contact counts.

**What it reduces to, stated honestly.** Under S3b's cost model the arm sees
no per-device resource signal except position and device count. With
``unit`` it is a budgeted nearest-neighbour chain; with ``devices``, a
devices-per-second greedy. It differs from our H1 in four ways: the argmax is
re-evaluated from the current end of the route at every step, there is no
per-device overdue check, there is no bucket priority, and it maximises how
many are served rather than who is due.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from hermes.types import (
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionPass,
)

from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.selector.scope_guard import (
    SelectorScopeViolation,
    assert_candidates_admitted,
)
from hermes.scheduler.stages.s3b_feasibility import (
    RULE_BUDGET,
    FeasibilityModel,
    FlightState,
)

from .budget_walk import IN_FLIGHT_BUDGET

MulePose = Tuple[float, float, float]

#: Value 1 per contact: the letter of Algorithm 3 line 3 (argmin marginal time).
VALUE_UNIT = "unit"
#: Value = devices served at the contact: the knapsack reading of Eq. (4)'s
#: client count for S3a's multi-device contacts. A port choice.
VALUE_DEVICES = "devices"
VALUE_KINDS: Tuple[str, ...] = (VALUE_UNIT, VALUE_DEVICES)


def _selection_key(wp: ContactWaypoint, total: float, value: str) -> tuple:
    """Sort key for Algorithm 3 line 3 at the current pose; lower is better.

    ``unit`` ranks on ``total`` directly rather than on ``1/total``: the two
    order identically in exact arithmetic, but two close totals can round to
    the same reciprocal, and the skip/stop equivalence rests on picking the
    true minimum.
    """
    tie = (tuple(wp.position), tuple(wp.devices))
    if value == VALUE_UNIT:
        return (total,) + tie
    # A zero-cost contact (co-located, zero session time) is free, so it
    # outranks any contact that costs something.
    ratio = len(wp.devices) / total if total > 0.0 else float("inf")
    return (-ratio,) + tie


def fedcs_greedy_select(
    contacts: Sequence[ContactWaypoint],
    *,
    value: str = VALUE_UNIT,
    mule_pose: MulePose,
    now: float,
    mission_deadline_ts: Optional[float],
    model: Optional[FeasibilityModel] = None,
    state: Optional[FlightState] = None,
) -> List[ContactWaypoint]:
    """Algorithm 3 on the mule's cost model; returns the ordered route.

    Recomputes the argmax from the current pose after every admission, removes
    each pick whether or not it fits (skip, not stop), and advances pose and
    clock only on admission. ``mission_deadline_ts=None`` admits every contact
    in the same greedy order.

    The pick ranks on the leg's marginal time (``FeasibilityModel.leg``) and
    the admission test is S3b's single predicate under the budget rule
    (``FeasibilityModel.admit``); in legacy mode those are exactly
    ``cost()``'s total and ``clock + total <= mission_deadline_ts``.
    ``state`` (FeRRy Phase 3) starts from a flight state, energy spent
    included, instead of ``(mule_pose, now)``.
    """
    if value not in VALUE_KINDS:
        raise ValueError(f"value must be one of {VALUE_KINDS}, got {value!r}")
    m = model or FeasibilityModel()
    remaining: List[ContactWaypoint] = list(contacts)
    route: List[ContactWaypoint] = []
    cur = state if state is not None else FlightState(
        tuple(mule_pose), float(now))  # type: ignore[arg-type]

    while remaining:
        # Line 3: price every candidate from where the route currently ends.
        best_i = 0
        best_key: Optional[tuple] = None
        for i, wp in enumerate(remaining):
            total = m.leg(cur.pose, wp).total_s
            key = _selection_key(wp, total, value)
            if best_key is None or key < best_key:
                best_i, best_key = i, key
        # Line 4: removed unconditionally, before the admission test.
        x = remaining.pop(best_i)
        # Lines 6-7, with <= (deviation 5): the shared predicate, budget rule
        # (no gate without a deadline). T_cs = T_agg = 0 in legacy mode.
        verdict = m.admit(cur, x, rule=RULE_BUDGET, budget_end=mission_deadline_ts)
        if not verdict.ok:
            continue
        # Lines 8-9.
        cur = verdict.next_state
        route.append(x)
    return route


class FedCSDegradedPolicy:
    """FedCS Algorithm 3 as a whole-scheduler baseline (arm D5, degraded).

    ``value`` picks the Algorithm 3 score: ``'unit'`` (the paper's letter,
    default) or ``'devices'`` (a port choice for multi-device contacts).
    """

    name = "FEDCS"
    # Freeze Amendment 8 — what the mule re-checks before each contact in
    # flight: the mission budget only. FedCS has one round deadline and no
    # per-device deadline, so holding its route to S3b's overdue test in
    # flight would re-impose exactly the rule this arm is compared against.
    in_flight_check = IN_FLIGHT_BUDGET

    def __init__(self, value: str = VALUE_UNIT) -> None:
        if value not in VALUE_KINDS:
            raise ValueError(
                f"FedCSDegradedPolicy value must be one of {VALUE_KINDS}, "
                f"got {value!r}"
            )
        self.value = value

    def rank_contacts(
        self,
        candidates: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        pass_kind: MissionPass = MissionPass.COLLECT,
        admitted: Optional[Sequence[DeviceID]] = None,
    ) -> List[ContactWaypoint]:
        """Ordering-only surface: Algorithm 3's greedy order, no admission cut.

        Kept so the policy swaps through the same slot as the other baselines.
        The scheduler calls :meth:`admit_and_order` instead whenever it exists.
        """
        if pass_kind is not MissionPass.COLLECT:
            raise SelectorScopeViolation(
                f"FedCSDegradedPolicy.rank_contacts called with "
                f"pass_kind={pass_kind.value!r}; FedCS is a Pass-1-only policy."
            )
        if not candidates:
            return []

        members: List[DeviceID] = []
        for wp in candidates:
            members.extend(wp.devices)
        assert_candidates_admitted(
            members, admitted if admitted is not None else members,
        )
        return fedcs_greedy_select(
            candidates,
            value=self.value,
            mule_pose=env.mule_pose,
            now=env.now,
            mission_deadline_ts=None,
            model=None,
        )

    # ------------------------------------------------------------------ #
    # Whole-scheduler mode (arm D5)
    # ------------------------------------------------------------------ #

    def admit_and_order(
        self,
        contacts: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        mission_deadline_ts: Optional[float] = None,
        feasibility_model=None,
    ) -> List[ContactWaypoint]:
        """FedCS as a **complete scheduler**: the largest greedy set that fits.

        The scheduler delegates S3's ordering, S3b and S3.5 to this method, so
        the policy owns admission. Travel is priced only with the scheduler's
        ``feasibility_model`` so every arm faces the same physics.
        ``device_states`` is not read (deviation 1) and nothing is mutated.
        """
        if not contacts:
            return []
        return fedcs_greedy_select(
            contacts,
            value=self.value,
            mule_pose=env.mule_pose,
            now=env.now,
            mission_deadline_ts=mission_deadline_ts,
            model=feasibility_model,
        )
