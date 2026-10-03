"""Oort's statistical-utility selection — SOTA baseline (arm B2).

Ports the part of **Oort** (Lai et al., *Efficient Federated Learning via Guided
Participant Selection*, USENIX OSDI 2021) that a data mule can actually run.

**Why it ports at all — the finding that reversed our first-pass reading.** Oort
is *retrospective by design*: "a client's utility can only be determined after it
has participated in training." The server never polls candidates before choosing;
it caches what each client reported the last time it was selected, and adds a
staleness bonus for clients it has not seen lately. That is exactly the state a
mule holds — what it learned last time it flew there — which is why Oort ports
directly and FedCS (whose Resource Request step polls every candidate each round)
does not.

**Name it honestly.** This is *Oort's statistical-utility selection*, not Oort.
Three deviations, each forced by our model rather than chosen, and each stated
here so the paper can state them too:

1. **No system-speed term** — unless the devices' fits take simulated time.
   Oort multiplies statistical utility by a straggler penalty over client
   compute/communication speed. Without a modelled compute speed the term is
   **dropped** — not approximated by something else, which would be worse than
   omitting it. Exp 5 addendum, Study 5.12: with the training time on the
   simulated clock (``hermes/mule/fit_clock.py``) the speed exists, and the
   term is restored in whole-scheduler mode (arm D2; the user's decision of
   2026-10-03): each explored member's utility, staleness bonus included, is
   multiplied by ``(T / t_i) ** alpha`` when ``t_i > T`` (Oort's Eq. 2 and
   Algorithm 1), with ``t_i`` the device's fit time plus its predicted dwell
   at the contact (the shared feasibility model's per-member dwell; one
   session without a band; a member predicted unreachable has ``t_i =
   inf``), ``T`` the cell's T_nom (the preferred round duration: one
   nominal mission) and ``alpha`` Oort's default 2
   (:meth:`OortPolicy.bind_fit_clock`).
2. **Mean loss, not RMS.** Oort specifies ``sqrt(Σ_k Loss(k)² / |B_i|)`` over
   per-sample losses. Our training callback reports Keras' **mean** loss.
   Monotone in the same direction, not identical.
3. **Rounds, not wall-clock, for staleness.** ``L(i)`` is the last mission round
   in which device *i* participated successfully (a CLEAN outcome). Failed
   sessions and the synthetic TIMEOUT fed for an abandoned device do not count,
   matching Oort, which updates a client's round only when its feedback lands.

**It requires real training.** On the stub path the reported loss and sample
count are random draws, so this policy would rank on noise — a random-order
baseline wearing Oort's name. Run arm B2 with ``--real-model`` only; the guard
below raises rather than silently producing a meaningless ordering.
"""

from __future__ import annotations

import math
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

from .budget_walk import IN_FLIGHT_BUDGET, greedy_budget_walk

#: Oort's exploration weight on the staleness bonus (paper's Algorithm 1).
DEFAULT_STALENESS_WEIGHT = 0.1

#: Oort's straggler penalty exponent alpha (its default), for the restored
#: system-speed term (deviation 1, Study 5.12).
DEFAULT_SPEED_ALPHA = 2.0

#: Utility given to a device never yet served. Oort explores unselected clients
#: explicitly; here "never measured" outranks any measured utility, which gives
#: the same behaviour without a separate exploration branch.
UNEXPLORED_UTILITY = float("inf")


class OortUnusableError(RuntimeError):
    """Raised when no candidate carries the training signal Oort needs.

    Almost always means arm B2 was run without ``--real-model``: the stub's
    loss and sample count are random, so ranking on them is noise. Failing
    loudly beats emitting a meaningless order that looks like a result.
    """


def statistical_utility(state: DeviceSchedulerState) -> float:
    """Oort's ``|B_i| · sqrt(mean Loss²)``, from the device's last contact.

    Returns :data:`UNEXPLORED_UTILITY` when this device has never reported —
    unexplored clients sort first, matching Oort's exploration of clients it has
    not yet measured.
    """
    if state.last_loss is None or state.last_num_examples <= 0:
        return UNEXPLORED_UTILITY
    # sqrt(loss^2) == |loss|; written this way to stay legible against the
    # paper's formula. See deviation 2 in the module docstring: our loss is the
    # mean over samples, where Oort specifies the RMS.
    return float(state.last_num_examples) * math.sqrt(state.last_loss ** 2)


def staleness_bonus(
    state: DeviceSchedulerState,
    *,
    current_round: int,
    weight: float = DEFAULT_STALENESS_WEIGHT,
) -> float:
    """Oort's ``weight · log(R) / sqrt(L(i))`` temporal-uncertainty term.

    Grows the utility of devices the scheduler has been overlooking — the same
    starvation pressure our Φ-widening applies, arrived at independently. Zero
    in round 1, where there is no history to be stale relative to.
    """
    if current_round <= 1:
        return 0.0
    if state.last_clean_round <= 0:
        return 0.0                      # never served: UNEXPLORED_UTILITY covers it
    return weight * math.log(current_round) / math.sqrt(state.last_clean_round)


def _member_dwell_s(model, wp: ContactWaypoint, did: DeviceID) -> float:
    """One member's predicted dwell at ``wp`` on the shared feasibility model.

    The ferry physics' per-member dwell at the member's planar distance (Pass
    1, no SNR offset); ``inf`` beyond the stop's range or below the floor, as
    :meth:`FerryPhysics.dwell_s` leaves such a member out. Without a band (or
    without the ferry physics) one session, ``session_time_s``; 0 without a
    model.
    """
    if model is None:
        return 0.0
    ferry = getattr(model, "ferry", None)
    if ferry is None or ferry.member_dwell_s is None:
        return float(model.session_time_s)
    d = ferry.member_distances_m(wp)[list(wp.devices).index(did)]
    if ferry.range_m is not None and d > ferry.range_m:
        return math.inf
    t = ferry.member_dwell_s(d, MissionPass.COLLECT, 0.0)
    return math.inf if t is None else float(t)


class OortPolicy:
    """Rank contacts by cached statistical utility plus a staleness bonus.

    Exposes the shared ``rank_contacts`` surface, so it swaps through the same
    ``target_selector`` slot as the RL selector and the other baselines.
    """

    name = "OORT"
    # Freeze Amendment 8 — the mule re-checks the budget only in flight.
    in_flight_check = IN_FLIGHT_BUDGET
    # FeRRy Phase 4 (decision 4 (b)): admit_and_order takes the pre-flight
    # member-subset carrier; its walk ranks members by this arm's own key.
    admits_member_subsets = True

    # Exp 5 addendum (Study 5.12, deviation 1): the mule's fit clock and T, set
    # on the instance only by :meth:`bind_fit_clock` (so an unbound policy's
    # state is the recorded one); unbound, the recorded key, no speed term.
    _fits = None
    _t_round_s: Optional[float] = None
    speed_alpha = DEFAULT_SPEED_ALPHA

    def __init__(self, *, staleness_weight: float = DEFAULT_STALENESS_WEIGHT):
        self.staleness_weight = float(staleness_weight)
        # FeRRy Phase 3 — (mission round, current round) of the last plan made
        # in whole-scheduler mode: a re-plan of that mission ranks with the
        # round the plan used (see ``_whole_scheduler_round``).
        self._planned_round: Optional[Tuple[int, int]] = None

    def bind_fit_clock(self, fits, *, t_nom_s: Optional[float]) -> None:
        """Restore Oort's system-speed term from the mule's fit clock (Study 5.12).

        ``fits`` is a ``hermes.mule.fit_clock.FitClock`` (each device's fit
        time); ``t_nom_s`` is T, the preferred round duration: the cell's
        T_nom, required (finite and > 0).
        """
        if t_nom_s is None or isinstance(t_nom_s, bool) or not (
                math.isfinite(float(t_nom_s)) and float(t_nom_s) > 0.0):
            raise ValueError(
                f"Oort's system-speed term needs T, the preferred round duration (the "
                f"cell's T_nom): got t_nom_s={t_nom_s!r}"
            )
        self._fits = fits
        self._t_round_s = float(t_nom_s)

    def speed_penalty(self, round_s: float) -> float:
        """Oort's ``(T / t_i) ** alpha`` for ``t_i > T``, else 1 (0 at ``t_i = inf``)."""
        t_round = self._t_round_s
        if t_round is None or round_s <= t_round:
            return 1.0
        if math.isinf(round_s):
            return 0.0
        return (t_round / round_s) ** self.speed_alpha

    def round_time_s(self, wp: ContactWaypoint, did: DeviceID, model) -> float:
        """Device ``did``'s round time at ``wp``: its fit time plus its predicted
        dwell there (the model's per-member dwell; one session without a band;
        ``inf`` when predicted unreachable)."""
        fit = 0.0 if self._fits is None else (self._fits.train_time_s(did) or 0.0)
        return float(fit) + _member_dwell_s(model, wp, did)

    def rank_contacts(
        self,
        candidates: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        pass_kind: MissionPass = MissionPass.COLLECT,
        admitted: Optional[Sequence[DeviceID]] = None,
    ) -> List[ContactWaypoint]:
        if pass_kind is not MissionPass.COLLECT:
            raise SelectorScopeViolation(
                f"OortPolicy.rank_contacts called with "
                f"pass_kind={pass_kind.value!r}; B2 is a Pass-1-only policy."
            )
        if not candidates:
            return []

        members: List[DeviceID] = []
        for wp in candidates:
            members.extend(wp.devices)
        assert_candidates_admitted(
            members, admitted if admitted is not None else members,
        )

        # Current round is DERIVED from state rather than counted per call:
        # counting couples the policy to how often the scheduler happens to
        # invoke it (once per non-empty bucket), which is not the same thing as
        # a mission round and would silently drift. `last_served_round` moves on
        # every outcome, so it keeps pace with the mission counter even in
        # rounds where nothing succeeded; L(i) itself is `last_clean_round`.
        current_round = 1 + max(
            (st.last_served_round
             for d in members
             if (st := device_states.get(d)) is not None),
            default=0,
        )

        # Guard the stub path — but tolerate the legitimate warm-up.
        #
        # Oort is retrospective, so a device's utility simply does not exist
        # until it has participated. Raising the first time a served device has
        # no loss would kill a healthy run during its warm-up. What is NOT
        # legitimate is a device that has been served REPEATEDLY and still
        # reports nothing: that is the stub path, where the loss is a random
        # draw and ranking on it would be a random ordering wearing Oort's name.
        # `on_time_count`, not `last_served_round`: the latter counts CONTACTS,
        # including failed ones, and a device contacted twice whose sessions both
        # failed has no loss entirely legitimately. Only a device that has
        # *successfully participated* twice and still reports nothing indicates
        # the stub.
        repeatedly_served = any(
            (st := device_states.get(d)) is not None and st.on_time_count >= 2
            for d in members
        )
        has_signal = any(
            (st := device_states.get(d)) is not None
            and st.last_loss is not None
            and st.last_num_examples > 0
            for d in members
        )
        if repeatedly_served and not has_signal:
            raise OortUnusableError(
                "arm B2 (Oort) has no per-device loss to rank on although "
                "devices have now been served twice or more — this is what "
                "running B2 without --real-model looks like. The stub's loss is "
                "a random draw, so ranking on it would be a random ordering "
                "wearing Oort's name."
            )

        def _contact_utility(wp: ContactWaypoint) -> float:
            """A contact's utility is the SUM over its clustered devices.

            Oort selects *clients*; our unit of travel is a contact serving
            several. Summing keeps the client-level semantics — visiting a
            contact collects every member, so its worth is what they are worth
            together. (Max would ignore the extra clients collected for free.)
            """
            total = 0.0
            for did in wp.devices:
                st = device_states.get(did)
                if st is None:
                    return UNEXPLORED_UTILITY
                u = statistical_utility(st)
                if u == UNEXPLORED_UTILITY:
                    return UNEXPLORED_UTILITY
                total += u + staleness_bonus(
                    st, current_round=current_round,
                    weight=self.staleness_weight,
                )
            return total

        return sorted(candidates, key=self._rank_key(device_states, current_round))

    # ------------------------------------------------------------------ #
    # Whole-scheduler mode (arm D2)
    # ------------------------------------------------------------------ #

    def _rank_key(self, device_states, current_round: int, model=None):
        """Descending contact utility; device id breaks ties deterministically.

        With the fit clock bound (deviation 1, Study 5.12), each explored
        member's utility is multiplied by its speed penalty, priced on
        ``model``; without it, the recorded key.
        """
        speed = self._fits is not None

        def _key(wp: ContactWaypoint) -> Tuple[float, str]:
            total = 0.0
            for did in wp.devices:
                st = device_states.get(did)
                if st is None:
                    return (-UNEXPLORED_UTILITY,
                            ",".join(sorted(str(d) for d in wp.devices)))
                u = statistical_utility(st)
                if u == UNEXPLORED_UTILITY:
                    return (-UNEXPLORED_UTILITY,
                            ",".join(sorted(str(d) for d in wp.devices)))
                util = u + staleness_bonus(
                    st, current_round=current_round,
                    weight=self.staleness_weight,
                )
                if speed:
                    util *= self.speed_penalty(self.round_time_s(wp, did, model))
                total += util
            return (-total, ",".join(sorted(str(d) for d in wp.devices)))
        return _key

    def _current_round(self, contacts, device_states) -> int:
        members = [d for wp in contacts for d in wp.devices]
        return 1 + max(
            (st.last_served_round
             for d in members
             if (st := device_states.get(d)) is not None),
            default=0,
        )

    def _whole_scheduler_round(self, contacts, device_states, env) -> int:
        """R for the staleness bonus in whole-scheduler mode (arm D2).

        The plan of a mission infers R as it always has: 1 + the latest round
        in which one of the candidates' devices had an outcome. That is also
        what every recorded D2 run did, although the scheduler has handed the
        mule's round in ``SelectorEnv.mission_round`` since Phase 2, so the
        plan is unchanged.

        FeRRy Phase 3's in-flight re-plan (``FLScheduler.replan_remainder``)
        calls this method again within the mission, over the remainder only.
        Inferred from those members alone, R could come out lower than the
        plan's (none of them may have had an outcome last mission), and the
        re-plan would rank by a different staleness term than the plan it
        repairs. So a call for the mission already planned (the same
        ``env.mission_round``) reuses the plan's R. Without a mission round
        in ``env`` every call infers, as before.
        """
        inferred = self._current_round(contacts, device_states)
        mission = getattr(env, "mission_round", None)
        if mission is None:
            return inferred
        planned = self._planned_round
        if planned is not None and planned[0] == int(mission):
            return planned[1]
        self._planned_round = (int(mission), inferred)
        return inferred

    def admit_and_order(
        self,
        contacts: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        mission_deadline_ts: Optional[float] = None,
        feasibility_model=None,
        member_subsets=None,
    ) -> List[ContactWaypoint]:
        """Oort as a **complete scheduler** — it decides *who*, not just order.

        Presence of this method makes the arm a whole-scheduler baseline: the
        scheduler delegates S3/S3b/S3.5 entirely, so this policy owns the
        admission decision our S3b gate would otherwise make. Visit the
        highest-utility contacts until the budget runs out. ``member_subsets``
        (FeRRy Phase 4) is passed only by the plan before takeoff under
        ``member_admission="subset"``: a contact that fails whole is then
        re-issued with its highest-utility members that still fit (None, the
        default, is the recorded walk).
        """
        if not contacts:
            return []
        return greedy_budget_walk(
            contacts,
            key=self._rank_key(device_states,
                               self._whole_scheduler_round(contacts, device_states, env),
                               model=feasibility_model),
            mule_pose=env.mule_pose,
            now=env.now,
            mission_deadline_ts=mission_deadline_ts,
            model=feasibility_model,
            member_subsets=member_subsets,
        )
