"""Study 5.7's plan-term ablations of FX: the FX-dwell and FX-cov arms.

Study 5.5's verdict (6 Oct 2026) kept FX as F's in-flight rule, so Study 5.7's
dwell and coverage ablations fly FX's cross-heuristic slot, where FQ-dwell and
FQ-cov would have flown the pair slot (critic C2's fallback). Pinned:

* **Arms and labels.** Both are addendum plan arms (``is_plan_arm``) outside
  the pinned ``PLAN_ARMS``; their labels survive a trace directory and the
  trace scorer parses them back.
* **Each is FX with one change:** FX-dwell takes the dwell out of Δ
  (``FQ_DWELL_SCORE``, FQ-dwell's change), FX-cov turns the coverage and link
  terms off (``F_COV_SCORE``, F-cov's). Their mule config and row differ from
  FX's there only.
* **They are refused** on the wall clock, as every plan arm is.
* **The launcher** flies them in batch 2's Study 5.7 beside FX and D4, with
  the plan arms' age cap and the campaign's deadline law, scored against FX;
  stage rl-s57 trains nothing once the verdict did not keep the learned score.
"""

from __future__ import annotations

import copy
import json

import pytest

from experiments.analysis.traces_scorer import parse_trial_dir
from experiments.exp4.driver import (
    ADDENDUM_PLAN_ARMS,
    ARMS,
    F_COV_SCORE,
    FQ_DWELL_SCORE,
    PLAN_ARMS,
    Exp4Driver,
    is_plan_arm,
    trace_dir_name,
)

from tests.golden import _build_p4_plan as UG5
from tests.golden import _build_topology as T
from tests.unit.test_exp5_scoring import L, _filled_settings
from tests.unit.test_p5_config_driver import _cell, _plain, _refused_before_anything_is_built

FX_ABLATIONS = ("FX-dwell", "FX-cov")
#: Each arm's change to FX's plan score, and the F-family arm that makes it.
CHANGE = {"FX-dwell": FQ_DWELL_SCORE, "FX-cov": F_COV_SCORE}
MIRROR = {"FX-dwell": "FQ-dwell", "FX-cov": "F-cov"}


def test_the_fx_ablations_are_addendum_plan_arms():
    assert ADDENDUM_PLAN_ARMS[-2:] == FX_ABLATIONS and ARMS[-2:] == FX_ABLATIONS
    for arm in FX_ABLATIONS:
        assert is_plan_arm(arm) and arm not in PLAN_ARMS
        name = trace_dir_name(_cell(arm, seed=2191267877, trial=3))
        assert f"__{arm}__" in name
        key = parse_trial_dir(name)
        assert (key.arm, key.trial_index, key.seed) == (arm, 3, 2191267877)


@pytest.mark.parametrize("arm", FX_ABLATIONS)
def test_an_fx_ablations_plan_settings_are_fxs_with_one_score_change(arm):
    drv = Exp4Driver(**UG5.PLAN_PILOT)
    fx, ab = drv.plan_settings("FX"), drv.plan_settings(arm)
    assert ab["flight_slot"] == fx["flight_slot"] == "cross_heuristic"
    assert ab["plan_score_params"] == {**fx["plan_score_params"], **CHANGE[arm]}
    assert ab["plan_score_params"] == drv.plan_settings(MIRROR[arm])["plan_score_params"]
    assert ab["plan_score_params"] != fx["plan_score_params"]
    assert ({k: v for k, v in ab.items() if k != "plan_score_params"}
            == {k: v for k, v in fx.items() if k != "plan_score_params"})
    assert drv.effective_member_admission(arm) == drv.effective_member_admission("FX")
    assert drv.effective_miss_priority(arm) is drv.effective_miss_priority("FX") is True


@pytest.mark.parametrize("arm", FX_ABLATIONS)
def test_an_fx_ablations_mule_and_row_are_fxs_but_the_score(arm):
    """The mule config equals FX's but for the plan score (and the per-trial RF
    token, which names the arm); the row differs only in ferry_params' score."""
    drv = Exp4Driver(**UG5.PLAN_PILOT)
    x_row, x_topo = T.run_stub_trial(drv, _cell("FX"))
    a_row, a_topo = T.run_stub_trial(drv, _cell(arm))
    (x,), (a,) = x_topo.mules, a_topo.mules
    own = {"rf_link_token", "plan_score_params"}
    assert _plain(a, own) == _plain(x, own)
    assert (a.plan_mode, a.flight_slot) == ("ferry", "cross_heuristic")
    assert a.plan_score_params == {**x.plan_score_params, **CHANGE[arm]}
    assert {k: v for k, v in a_row.items() if k != "ferry_params"} == {
        k: v for k, v in x_row.items() if k != "ferry_params"}
    xp, ap = json.loads(x_row["ferry_params"]), json.loads(a_row["ferry_params"])
    assert ap == {**xp, "plan_score_params": a.plan_score_params}


@pytest.mark.parametrize("arm", FX_ABLATIONS)
def test_an_fx_ablation_is_refused_on_the_wall_clock(arm):
    _refused_before_anything_is_built(Exp4Driver(), arm, "flies the plan clock")


# --------------------------------------------------------------------------- #
# The launcher: Study 5.7 under the 5.5 verdict (FX kept)
# --------------------------------------------------------------------------- #

def _fx_kept_settings():
    """The filled settings with 5.5's verdict as it read: the learned score not kept."""
    s = _filled_settings()
    d = copy.deepcopy(s.data)
    d["rl"]["keep_learned"] = False
    return L.Settings(d)


def test_params_fly_fx_and_its_ablations_in_study_5_7():
    s, _ = L.load_settings(L.PARAMS)
    assert s.get("s57.arms") == ["FX", "FX-dwell", "FX-cov", "D4"]
    assert s.get("score.s57.reference") == "FX"
    assert s.get("rl.keep_learned") is False


def test_batch2_flies_study_5_7s_fx_ablations_as_plan_arms(tmp_path):
    s = _fx_kept_settings()
    jobs = [j for j in L.build("batch2", s, None, str(tmp_path))
            if j.kind == "runner" and j.study == "s57"]
    arms = {j.args[j.args.index("--arms") + 1] for j in jobs}
    assert arms == {"FX", "FX-dwell", "FX-cov", "D4"}
    assert len(jobs) == 4 * len(s.get("s57.budgets"))
    law = s.get("campaign.ferry_arm_deadline_law", None)
    for j in jobs:
        arm = j.args[j.args.index("--arms") + 1]
        assert not j.blocked, (j.name, j.blocked)
        assert ("--age-cap-missions" in j.args) == (arm != "D4"), j.name
        if law:
            assert ("--deadline-law" in j.args) == (arm != "D4"), j.name
        assert "--pair-checkpoint" not in j.args


def test_rl_s57_trains_nothing_when_the_learned_score_is_not_kept(tmp_path, capsys):
    jobs = L.build("rl-s57", _fx_kept_settings(), None, str(tmp_path))
    assert jobs == []
    assert "FX-dwell and FX-cov" in capsys.readouterr().out
