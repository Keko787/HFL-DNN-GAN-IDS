"""Exp 5 addendum, Studies 5.14 and 5.15: the F+L1 arm.

``F+L1`` is F's plan with H3's adaptive backhaul controller, as ``H1+L1`` is
H1's scheduler with it (FeRRy Phase 5, decision 8 (a)). Pinned:

* **Arms and labels.** ``ADDENDUM_ARMS`` is ``("F+L1",)``, a plan arm
  (``is_plan_arm``) outside the pinned ``PLAN_ARMS``; its label survives a
  trace directory and the scorer parses it back.
* **It is F** with ``backhaul_policy='adaptive'``: on the seconds-axis
  backhaul its mule config is F's but for that field, and its row differs
  from F's only there; it has F's settings everywhere F has them (member
  subsets, the miss priority, the trim fallback, T_nom, the plan settings).
* **On the L1 channel** its cluster draws H3's adaptive loss schedule and its
  mule gets H3's RF prior schedule, where F holds the fixed band's.
* **It is refused** where it would fly as F: on the wall clock (a plan arm),
  and on the simulated clock's mission backhaul without the L1 channel.
"""

from __future__ import annotations

import json

import pytest

from experiments.analysis.traces_scorer import parse_trial_dir
from experiments.exp4.driver import (
    ADDENDUM_ARMS,
    ADDENDUM_PLAN_ARMS,
    ARMS,
    PLAN_ARMS,
    Exp4Driver,
    is_plan_arm,
    trace_dir_name,
)

from tests.golden import _build_p4_plan as UG5
from tests.golden import _build_topology as T
from tests.unit.test_p5_config_driver import _cell, _plain, _refused_before_anything_is_built

#: F's plan-pilot flags on the seconds-axis backhaul, where the controller flies.
SECONDS = dict(UG5.PLAN_PILOT, backhaul_model="seconds", t_nom_layouts=4)


def test_f_l1_is_the_addendums_first_arm_and_a_plan_arm():
    # Unit U11's F-round and F-pref follow it (tests/unit/test_exp5_u11.py), then
    # Study 5.7's FX-dwell and FX-cov (tests/unit/test_exp5_fx_ablations.py).
    assert ADDENDUM_ARMS == ADDENDUM_PLAN_ARMS == (
        "F+L1", "F-round", "F-pref", "FX-dwell", "FX-cov")
    assert ARMS[-5] == "F+L1" and "F+L1" not in PLAN_ARMS
    assert is_plan_arm("F+L1")
    name = trace_dir_name(_cell("F+L1", seed=2191267877, trial=3))
    assert "__F+L1__" in name
    key = parse_trial_dir(name)
    assert (key.arm, key.trial_index, key.seed) == ("F+L1", 3, 2191267877)


def test_f_l1_has_fs_settings_everywhere_f_has_them():
    drv = Exp4Driver(**SECONDS)
    assert drv.plan_settings("F+L1") == drv.plan_settings("F")
    assert drv.effective_member_admission("F+L1") == drv.effective_member_admission("F")
    assert drv.effective_miss_priority("F+L1") is drv.effective_miss_priority("F") is True
    f, fl = drv.ferry_settings(arm="F", regime="jittery"), drv.ferry_settings(
        arm="F+L1", regime="jittery")
    assert (f.pop("backhaul_policy"), fl.pop("backhaul_policy")) == ("fixed", "adaptive")
    assert fl == f


def test_f_l1_is_f_with_h3s_adaptive_backhaul_on_the_seconds_model():
    drv = Exp4Driver(**SECONDS)
    rows, mules = {}, {}
    for arm in ("F", "F+L1"):
        rows[arm], topo = T.run_stub_trial(drv, _cell(arm))
        (mules[arm],) = topo.mules
    f, fl = mules["F"], mules["F+L1"]
    assert (f.backhaul_policy, fl.backhaul_policy) == ("fixed", "adaptive")
    assert (f.use_rl_selector, fl.use_rl_selector) == (False, False)
    assert (fl.plan_mode, fl.flight_slot, fl.replan_fallback) == ("ferry", "committed", "trim")
    assert _plain(fl, {"backhaul_policy", "rf_link_token"}) == _plain(
        f, {"backhaul_policy", "rf_link_token"})
    assert json.loads(rows["F+L1"]["ferry_params"]) == dict(
        json.loads(rows["F"]["ferry_params"]), backhaul_policy="adaptive")
    assert {k: v for k, v in rows["F+L1"].items() if k != "ferry_params"} == {
        k: v for k, v in rows["F"].items() if k != "ferry_params"}


def test_f_l1_flies_h3s_adaptive_loss_schedule_on_the_l1_channel():
    """The goldens' H3 L1 seed, where the adaptive and fixed schedules differ."""
    drv = Exp4Driver(**dict(UG5.PLAN_PILOT, l1_channel=True, t_nom_layouts=4))
    topos = {}
    for arm in ("F", "F+L1", "H3"):
        _row, topos[arm] = T.run_stub_trial(drv, _cell(arm, seed=2191267877))
    schedule = {arm: topos[arm].cluster.backhaul_loss_schedule for arm in topos}
    assert schedule["F+L1"] == schedule["H3"] != schedule["F"]
    priors = {arm: topos[arm].mules[0].rf_prior_schedule_db for arm in topos}
    assert priors["F+L1"] == priors["H3"] != priors["F"]
    assert topos["F+L1"].mules[0].use_rl_selector is False


@pytest.mark.parametrize("kw, match", [
    (dict(), "flies the plan clock"),
    (dict(UG5.PLAN_PILOT), "here it would fly as F"),
], ids=["wall", "sim-mission-backhaul"])
def test_f_l1_is_refused_where_it_would_fly_as_f(kw, match):
    _refused_before_anything_is_built(Exp4Driver(**kw), "F+L1", match)


def test_the_runner_refuses_f_l1_before_any_trial(tmp_path, capsys):
    from experiments.exp4 import runner_main

    with pytest.raises(SystemExit) as e:
        runner_main.main(["--csv", str(tmp_path / "t.csv"), "--arms", "F+L1",
                          "--mission-clock", "sim", "--contact-band", "wide"])
    assert e.value.code == 2 and "here it would fly as F" in capsys.readouterr().err
