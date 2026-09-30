"""FeRRy Phase 3 (unit U4): legacy identity of the one predicate (Freeze Rule 1).

``tests/golden/test_golden_feasibility.py`` pins the walks as the scheduler
and the mule call them. This file goes one step further: it rebuilds each
walk from the new primitives alone — ``FeasibilityModel.fold`` and
``FeasibilityModel.admit`` with an explicit ``FlightState`` and an explicit
``ferry=None``, and the walks' new ``state=`` arguments — and requires the
afa9526 digest of every one of the 2,400 recorded instances. It also replays
the ``MuleSupervisor`` goldens with every Phase 3 switch of this unit set
explicitly to its legacy value (``deadline_time_scale=1.0`` folded into an
explicit additive law, ``initial_window_s=60``) and the two opt-in flags
switched on, which must be inert without a ferry model and without overrides.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import hermes.mule.mule_main as mule_main
from hermes.scheduler import FLScheduler
from hermes.scheduler.policies.budget_walk import greedy_budget_walk
from hermes.scheduler.policies.fedcs_degraded import VALUE_UNIT, fedcs_greedy_select
from hermes.scheduler.stages.s3_deadline import DeadlineLaw
from hermes.scheduler.stages.s3b_feasibility import (
    REASON_BUDGET,
    REASON_OVERDUE,
    RULE_BUDGET,
    RULE_DEADLINE_BUDGET,
    FeasibilityModel,
    FlightState,
    filter_feasible,
)

from tests.golden import _build_feasibility as B
from tests.golden import _mule_harness as M
from tests.golden._canon import assert_same, digest, load

FEAS = load("feasibility")
RANDOM_KEYS = [k for k in FEAS["cases"] if k.startswith("random:")]
SUP = load("supervisor")


def _explicit_legacy(model):
    """The instance's model, rebuilt with ``ferry=None`` stated explicitly."""
    m = model if model is not None else FeasibilityModel()
    return FeasibilityModel(cruise_speed_m_s=m.cruise_speed_m_s,
                            session_time_s=m.session_time_s, ferry=None)


def _s3b_by_fold(inst, m, prio) -> List[Any]:
    contacts, now, pose = inst["contacts"], inst["now"], inst["pose"]
    mdl = None if inst["budget"] is None else inst["start"] + inst["budget"]
    if mdl is None:                     # the opt-in no-op: same list, same order
        kept, overdue, over = list(contacts), [], []
    else:
        if prio is None:
            key = lambda c: (c.deadline_ts, c.position, c.devices)            # noqa: E731
        else:
            key = lambda c: (-prio[c.devices], c.deadline_ts, c.position, c.devices)  # noqa: E731
        walk = m.fold(sorted(contacts, key=key), FlightState(tuple(pose), float(now)),
                      rule=RULE_DEADLINE_BUDGET, budget_end=mdl, skip=True)
        kept = list(walk.route)
        overdue, over = walk.rejected_by(REASON_OVERDUE), walk.rejected_by(REASON_BUDGET)
    return [B._idx(contacts, kept), B._idx(contacts, overdue), B._idx(contacts, over),
            len(overdue) + len(over)]


def _s3b_by_state(inst, m, prio) -> List[Any]:
    contacts = inst["contacts"]
    mdl = None if inst["budget"] is None else inst["start"] + inst["budget"]
    res = filter_feasible(
        contacts, now=-1.0, mule_pose=(9e9, 9e9, 9e9), mission_deadline_ts=mdl, model=m,
        priority=None if prio is None else (lambda c: prio[c.devices]),
        state=FlightState(tuple(inst["pose"]), float(inst["now"])),
    )
    return [B._idx(contacts, res.kept), B._idx(contacts, res.dropped_overdue),
            B._idx(contacts, res.dropped_budget), res.n_dropped]


def _greedy_by_fold(inst, m, key) -> List[int]:
    contacts = inst["contacts"]
    mdl = None if inst["budget"] is None else inst["start"] + inst["budget"]
    ordered = sorted(contacts, key=key)
    if mdl is None:
        return B._idx(contacts, ordered)
    walk = m.fold(ordered, FlightState(inst["pose"], inst["now"]), rule=RULE_BUDGET,
                  budget_end=mdl, skip=True)
    return B._idx(contacts, walk.route)


def _inflight_by_admit(inst, m) -> Dict[str, List[bool]]:
    """The mule's in-flight check is the predicate on the next stop."""
    contacts, budget = inst["contacts"], inst["budget"]
    state = FlightState(inst["pose"], inst["now"])
    out: Dict[str, List[bool]] = {}
    for name, rule in (("s3b", RULE_DEADLINE_BUDGET), ("budget_rule", RULE_BUDGET),
                       ("none_rule", None)):
        row = []
        for k in range(len(contacts) + 1):
            rest = contacts[k:]
            if not rest:
                row.append(False)
            elif budget is None or rule is None:
                row.append(True)
            else:
                end = inst["start"] + float(budget)
                row.append(m.admit(state, rest[0], rule=rule, budget_end=end).ok)
        out[name] = row
    return out


def _pass_2_by_fold(inst, m) -> List[List[int]]:
    contacts, budget = inst["contacts"], inst["budget"]
    if budget is None or not contacts:
        return [B._idx(contacts, contacts), []]
    walk = m.fold(list(contacts), FlightState(inst["pose"], 0.0), rule=RULE_BUDGET,
                  budget_end=float(budget), skip=True)
    flown = {id(w) for w in walk.route}
    return [B._idx(contacts, walk.route),
            B._idx(contacts, [w for w in contacts if id(w) not in flown])]


def test_the_one_predicate_reproduces_every_recorded_walk():
    """All 2,400 instances: S3b (EDF and miss priority), the D-arm walk (EDF
    key and queue order), FedCS, the in-flight check for all three rules and
    the Pass-2 walk, each rebuilt from fold/admit, and through the walks'
    ``state=`` arguments, match the afa9526 digest."""
    assert len(RANDOM_KEYS) >= 2000
    bad = []
    for key in RANDOM_KEYS:
        inst = B.make_instance(int(key.split(":")[1]))
        golden = FEAS["cases"][key]
        assert B.instance_digest(inst) == golden["in"], key
        m = _explicit_legacy(inst["model"])
        base = B.run_instance(inst)
        contacts, prio = inst["contacts"], inst["priority"]
        mdl = None if inst["budget"] is None else inst["start"] + inst["budget"]
        order = {id(c): i for i, c in enumerate(contacts)}
        edf_key = lambda c: (c.deadline_ts, c.position, c.devices)          # noqa: E731
        order_key = lambda c: (order[id(c)],)                                # noqa: E731
        start = FlightState(tuple(inst["pose"]), float(inst["now"]))

        by_fold = dict(base)
        by_fold["s3b"] = _s3b_by_fold(inst, m, None)
        by_fold["s3b_prio"] = _s3b_by_fold(inst, m, prio)
        by_fold["greedy_edf"] = _greedy_by_fold(inst, m, edf_key)
        by_fold["greedy_order"] = _greedy_by_fold(inst, m, order_key)
        by_fold["inflight"] = _inflight_by_admit(inst, m)
        by_fold["pass_2"] = _pass_2_by_fold(inst, m)

        by_state = dict(base)
        by_state["s3b"] = _s3b_by_state(inst, m, None)
        by_state["s3b_prio"] = _s3b_by_state(inst, m, prio)
        for name, k in (("greedy_edf", edf_key), ("greedy_order", order_key)):
            by_state[name] = B._idx(contacts, greedy_budget_walk(
                contacts, key=k, mule_pose=(9e9, 9e9, 9e9), now=-1.0,
                mission_deadline_ts=mdl, model=m,
                state=FlightState(inst["pose"], inst["now"])))
        by_state["fedcs_unit"] = B._idx(contacts, fedcs_greedy_select(
            contacts, value=VALUE_UNIT, mule_pose=(9e9, 9e9, 9e9), now=-1.0,
            mission_deadline_ts=mdl, model=m, state=start))

        for label, result in (("run_instance", base), ("fold/admit", by_fold),
                              ("state=", by_state)):
            if digest(result) != golden["out"]:
                bad.append(f"{key} via {label}")
    assert not bad, f"{len(bad)} walk(s) moved:\n  " + "\n  ".join(bad[:12])


class _ExplicitLegacyScheduler(FLScheduler):
    """Every switch this unit adds, set explicitly to its legacy value; the
    flags that need a ferry model or an override to act are switched on."""

    built = 0

    def __init__(self, **kwargs):
        type(self).built += 1
        law = kwargs.pop("deadline_law", None)
        super().__init__(
            deadline_law=law if law is not None else DeadlineLaw(time_scale=1.0),
            deadline_time_scale=1.0,
            initial_window_s=60.0,
            refuse_deadline_overrides=True,
            validate_flown_order=True,
            replan_fallback="trim",
            **kwargs,
        )


@pytest.mark.parametrize("name", sorted(M.SCENARIOS))
def test_supervisor_goldens_with_the_switches_set_explicitly(name, monkeypatch):
    monkeypatch.setattr(mule_main, "FLScheduler", _ExplicitLegacyScheduler)
    before = _ExplicitLegacyScheduler.built
    current = M.SCENARIOS[name]()
    assert _ExplicitLegacyScheduler.built > before      # the replay used the switches
    assert_same(SUP["cases"][name], current, name)
