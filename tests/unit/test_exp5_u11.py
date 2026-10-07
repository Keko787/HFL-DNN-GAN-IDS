"""Exp 5 addendum, unit U11 (Study 5.2's deadline forms): arms F-round and F-pref.

Pinned:

* **The laws** (``s3_deadline``): ``round`` and ``pref`` need ``round_s`` and
  give every device the same deadline, the plan's time plus the round; neither
  moves a window with outcomes. The merge window is the round under ``round``
  (a cutoff of about one round, after FedCS) and infinite under ``pref`` (no
  cutoff, after Oort), and ``age_cap`` reads an infinite window as no cutoff.
  The recorded laws' parameters are unchanged (no ``round_s`` key).
* **Oort's speed factor** (``plan_score``): ``(T / t) ** alpha`` above T, 1
  below, a positive floor for a device that never finishes; ``demand_weights``
  multiplies by it only when given one.
* **The arms** (driver): F-round and F-pref are plan arms with the round's
  deadline set to the cell's budget (refused without one), F-pref with
  ``plan_speed_alpha`` = 2; every other arm keeps the driver's law. On a real
  FerrySim mission: each row records its arm's law, only F-pref's
  ``ferry_params`` shows the exponent, only its scheduler holds a speed hook,
  and the merge cutoffs are one round (F-round) or none (F-pref).
"""

from __future__ import annotations

import json
import logging
import math
from types import SimpleNamespace

import pytest

from experiments.exp4.driver import (
    ADDENDUM_PLAN_ARMS, Exp4Driver, is_plan_arm,
)
from hermes.mission.aggregation_rules import AGG_CUTOFF, AggregationSpec, age_cap
from hermes.processes.config import MuleConfig, mule_config_errors
from hermes.scheduler.plan.plan_score import (
    MIN_SPEED_FACTOR, demand_weights, oort_speed_factors,
)
from hermes.scheduler.stages.s3_deadline import (
    LAW_MULTIPLICATIVE, LAW_PREF, LAW_ROUND, DeadlineLaw, DeadlineLawError,
    compute_deadline, effective_window,
)
from hermes.types.round_report import MissionOutcome


# --------------------------------------------------------------------------- #
# The laws
# --------------------------------------------------------------------------- #

def test_the_round_laws_need_the_round_and_only_they_take_it():
    for form in (LAW_ROUND, LAW_PREF):
        with pytest.raises(DeadlineLawError, match="round_s"):
            DeadlineLaw(form=form)
        with pytest.raises(DeadlineLawError, match="round_s"):
            DeadlineLaw(form=form, round_s=0.0)
    with pytest.raises(DeadlineLawError, match="round_s"):
        DeadlineLaw(form=LAW_MULTIPLICATIVE, round_s=90.0)
    assert "round_s" not in DeadlineLaw().to_params()
    assert "round_s" not in DeadlineLaw(form=LAW_MULTIPLICATIVE).to_params()
    assert DeadlineLaw(form=LAW_ROUND, round_s=90).to_params()["round_s"] == 90.0


@pytest.mark.parametrize("form", [LAW_ROUND, LAW_PREF])
def test_every_device_gets_the_rounds_deadline_and_no_window_moves(form):
    law = DeadlineLaw(form=form, round_s=90.0)
    for phi, idle in ((10.0, 0.0), (300.0, 50.0)):
        st = SimpleNamespace(deadline_override_ts=None, deadline_fulfilment_s=phi,
                             idle_time_ref_ts=1000.0 - idle)
        assert compute_deadline(st, 1000.0, 2.0, law=law) == 1090.0
    for outcome in MissionOutcome:
        assert law.next_window(42.0, outcome) == 42.0
    st = SimpleNamespace(deadline_fulfilment_s=7.0)
    assert effective_window(st, law=law) == (90.0 if form == LAW_ROUND else math.inf)


def test_age_cap_cuts_at_one_round_or_not_at_all():
    spec = AggregationSpec(rule=AGG_CUTOFF, period_s=100.0)
    assert age_cap(spec, 250.0) == 2                     # the recorded rule, unchanged
    assert age_cap(spec, 90.0) == 0 and age_cap(spec, 100.0) == 1
    assert age_cap(spec, math.inf) is None               # pref: no cutoff
    assert age_cap(AggregationSpec(rule=AGG_CUTOFF, period_s=100.0, a_max=3), math.inf) == 3


# --------------------------------------------------------------------------- #
# Oort's speed factor
# --------------------------------------------------------------------------- #

def test_oorts_factor():
    fits = {"a": 0.0, "b": 150.0, "c": math.inf}.get
    f = oort_speed_factors(["a", "b", "c"], fit_s=fits, dwell_s=50.0, t_ref_s=100.0, alpha=2.0)
    assert f["a"] == 1.0                                  # t = 50 <= T
    assert f["b"] == pytest.approx((100.0 / 200.0) ** 2)  # t = 200 > T
    assert f["c"] == MIN_SPEED_FACTOR                      # never finishes
    assert oort_speed_factors(["a"], fit_s=None, dwell_s=50.0, t_ref_s=100.0,
                              alpha=2.0) == {"a": 1.0}


def test_demand_weights_take_the_factor_only_when_given():
    states = {d: SimpleNamespace(miss_streak=1) for d in ("a", "b")}
    kw = dict(ages={"a": 2, "b": 3}, miss_priority=True, mode="age")
    plain = demand_weights(["a", "b"], states, **kw)
    assert plain == demand_weights(["a", "b"], states, **kw, speed=None)
    assert demand_weights(["a", "b"], states, **kw, speed={"b": 0.25}) == {
        "a": plain["a"], "b": plain["b"] * 0.25}
    assert demand_weights(["a", "b"], states, **kw, speed=lambda ds: {d: 0.5 for d in ds}) == {
        d: w * 0.5 for d, w in plain.items()}
    for bad in (0.0, 1.5, math.nan):
        with pytest.raises(ValueError, match="speed factor"):
            demand_weights(["a"], states, ages={"a": 1}, miss_priority=False, mode="age",
                           speed={"a": bad})


# --------------------------------------------------------------------------- #
# The arms
# --------------------------------------------------------------------------- #

def test_the_arms_and_their_laws():
    assert ADDENDUM_PLAN_ARMS[1:3] == ("F-round", "F-pref")
    assert all(is_plan_arm(a) for a in ("F-round", "F-pref"))
    d = Exp4Driver(mission_clock="sim", contact_band="wide", mission_budget_s=90.0,
                   deadline_law="multiplicative")
    assert d.arm_deadline_law("F").form == LAW_MULTIPLICATIVE
    assert d.arm_deadline_law("H1").form == LAW_MULTIPLICATIVE
    assert (d.arm_deadline_law("F-round").form, d.arm_deadline_law("F-round").round_s) == (
        LAW_ROUND, 90.0)
    assert d.arm_deadline_law("F-pref").form == LAW_PREF
    assert d.plan_settings("F-pref")["plan_speed_alpha"] == 2.0
    assert "plan_speed_alpha" not in d.plan_settings("F-round")
    assert "plan_speed_alpha" not in d.plan_settings("F")
    with pytest.raises(ValueError, match="mission_budget_s"):
        Exp4Driver(mission_clock="sim", contact_band="wide").check_arm("F-round")


def test_the_config_refuses_a_misplaced_or_bad_exponent():
    errs = mule_config_errors(MuleConfig(mule_id="m", plan_speed_alpha=2.0))
    assert any("plan_speed_alpha" in e and "simulated mission clock" in e for e in errs)
    sim = dict(mule_id="m", mission_clock="sim", trial_seed=7, contact_band="wide")
    errs = mule_config_errors(MuleConfig(**sim, plan_speed_alpha=2.0))
    assert any("plan_mode='ferry'" in e for e in errs)
    errs = mule_config_errors(MuleConfig(**sim, plan_speed_alpha=-1.0))
    assert any("finite number > 0" in e for e in errs)


#: The N = 6 control _mission flies (its budget is the re-pin's knee: read it from
#: the cell, never as a literal, so a re-pin that renames the cell keeps this test).
CONTROL = "jit-n6-150"


def _mission(arm):
    """One FerrySim episode of ``arm`` on the N = 6 control: its row, and what the
    mule's scheduler held at the first plan."""
    from experiments.ferrysim import cells as C
    from experiments.ferrysim.episode import Policy, run_episode

    seen = {}

    def tap(service):
        sup = service.supervisor
        sch = sup.scheduler
        original = sch.build_ferry_plan

        def wrapped(*args, **kw):
            route = original(*args, **kw)
            if not seen:
                seen["speed"] = getattr(sch, "plan_speed", None)
                seen["caps"] = sup._age_caps()
                seen["period"] = sup.aggregation.period_s
                seen["weights"] = dict(sch.last_plan.weights)
            return route

        sch.build_ferry_plan = wrapped

    cell = C.cell_named(CONTROL)
    logging.disable(logging.WARNING)
    try:
        # agg:cutoff cuts by age only with a merge period: T_nom, as the campaign runs it.
        got = run_episode(cell, C.stream_seeds(C.VAL_STREAM, cell.name, 1)[0],
                          Policy.of_arm(arm), hooks=(tap,),
                          driver_overrides={"agg_period_t_nom": True})
    finally:
        logging.disable(logging.NOTSET)
    return got.row, seen


def test_on_a_real_mission_each_arm_flies_its_own_deadline():
    from experiments.ferrysim import cells as C

    budget = C.cell_named(CONTROL).budget_s      # F-round's one round: the mission budget
    f_row, f_seen = _mission("F")
    r_row, r_seen = _mission("F-round")
    p_row, p_seen = _mission("F-pref")
    assert r_row["deadline_law"] == "round" and p_row["deadline_law"] == "pref"
    assert json.loads(r_row["deadline_params"])["round_s"] == budget
    assert "plan_speed_alpha" not in json.loads(f_row["ferry_params"])
    assert "plan_speed_alpha" not in json.loads(r_row["ferry_params"])
    assert json.loads(p_row["ferry_params"])["plan_speed_alpha"] == 2.0
    assert f_seen["speed"] is None and r_seen["speed"] is None and p_seen["speed"] is not None
    # No training times here: t_j is the dwell alone, under T, so Oort's factor is 1.
    assert p_seen["weights"] == pytest.approx(r_seen["weights"])
    caps = r_seen["caps"]
    assert caps and set(caps.values()) == {int(budget // r_seen["period"])}
    assert set(p_seen["caps"].values()) == {None}
