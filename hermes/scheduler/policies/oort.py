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

1. **No system-speed term.** Oort multiplies statistical utility by a straggler
   penalty over client compute/communication speed. Our devices have no modelled
   compute speed, so the term is **dropped** — not approximated by something
   else, which would be worse than omitting it.
2. **Mean loss, not RMS.** Oort specifies ``sqrt(Σ_k Loss(k)² / |B_i|)`` over
   per-sample losses. Our training callback reports Keras' **mean** loss.
   Monotone in the same direction, not identical.
3. **Rounds, not wall-clock, for staleness.** ``L(i)`` is the last mission round
   in which device *i* was served.

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

#: Oort's exploration weight on the staleness bonus (paper's Algorithm 1).
DEFAULT_STALENESS_WEIGHT = 0.1

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
    if state.last_served_round <= 0:
        return 0.0                      # never served: UNEXPLORED_UTILITY covers it
    return weight * math.log(current_round) / math.sqrt(state.last_served_round)


class OortPolicy:
    """Rank contacts by cached statistical utility plus a staleness bonus.

    Exposes the shared ``rank_contacts`` surface, so it swaps through the same
    ``target_selector`` slot as the RL selector and the other baselines.
    """

    name = "OORT"

    def __init__(self, *, staleness_weight: float = DEFAULT_STALENESS_WEIGHT):
        self.staleness_weight = float(staleness_weight)

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
        # a mission round and would silently drift.
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

        def _key(wp: ContactWaypoint) -> Tuple[float, str]:
            # Descending utility; device id breaks ties so the order is
            # deterministic and independent of input order.
            return (-_contact_utility(wp),
                    ",".join(sorted(str(d) for d in wp.devices)))

        return sorted(candidates, key=_key)
