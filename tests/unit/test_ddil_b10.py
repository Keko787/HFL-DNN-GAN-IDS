"""DDIL build B10 (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md, The FQ track).

* ``FerryCell``'s DDIL fields: build D1's channel and step 1's switches, left out
  of a cell's JSON at None, so every recorded family hashes as it did;
* the two DDIL families: ``jittery56-s1`` (FQ-s1's) and ``ddil`` (FQ-d's);
* a training run's cells share one set of step 1's switches, which its manifest
  records, and the runner flies a checkpoint trained on them only on them;
* the launcher's variants: FQ-s1 beside FQ-g75 with its own checkpoint, and
  step 2's F variants with their own flags, each under its own job name.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
from types import SimpleNamespace

import pytest

from experiments.ferrysim import cells as C
from experiments.ferrysim import checkpoints as K
from experiments.ferrysim import train as T
from tests.unit.test_exp5_ferrysim_scale import SCALE_SHA256
from tests.unit.test_exp5_scoring import L, _filled_settings
from tests.unit.test_p5_ferrysim_runner import CLEAN_SHA256, JITTERY56_SHA256, JITTERY_SHA256

S1 = C.STEP1_SWITCHES


# --------------------------------------------------------------------------- #
# The cells and the families
# --------------------------------------------------------------------------- #

def test_the_recorded_families_hash_as_they_did():
    assert C.family_sha256("jittery") == JITTERY_SHA256
    assert C.family_sha256("clean") == CLEAN_SHA256
    assert C.family_sha256("jittery56") == JITTERY56_SHA256
    assert C.family_sha256("scale") == SCALE_SHA256
    for cell in C.CELLS + C.STUDY_5_6_CELLS + C.SCALE_CELLS:
        assert not set(C.DDIL_FIELDS) & set(cell.to_json())
        assert not set(C.DDIL_FIELDS) & set(cell.driver_settings()["ferry_physics"])
        assert cell.ddil_switches() == {}


def test_jittery56_s1_is_jittery56_on_step_1s_physics_and_nothing_else():
    base, s1 = C.FAMILIES["jittery56"], C.FAMILIES["jittery56-s1"]
    assert [c.name for c in s1] == [f"{c.name}-s1" for c in base]
    for b, c in zip(base, s1):
        assert c.ddil_switches() == S1
        assert dataclasses.replace(c, name=b.name, family=b.family,
                                   **{f: None for f in C.DDIL_SWITCH_FIELDS}) == b
        physics = c.driver_settings()["ferry_physics"]
        assert {k: physics[k] for k in S1} == {k: (float(v) if isinstance(v, float) else v)
                                              for k, v in S1.items()}
        assert set(C.DDIL_CHANNEL_FIELDS).isdisjoint(physics)
    assert C.family_sha256("jittery56-s1") != JITTERY56_SHA256


def test_the_ddil_family_is_five_channels_at_each_sizes_knee():
    cells = C.FAMILIES["ddil"]
    assert [c.name for c in cells] == [
        f"ddl-n{n}-{k:g}-{s}" for n, k in ((6, C.BUDGETS_S[6][1]), (12, C.BUDGETS_S[12][1]))
        for s in ("def", "a11", "a17", "b10", "b40")]
    for c in cells:
        assert c.ddil_switches() == S1 and c.budget_s == C.BUDGETS_S[c.n_devices][1]
        assert c.contact_regime == "jittery" and c.family == "ddil"
        physics = c.driver_settings()["ferry_physics"]
        assert "n_pl" not in physics                 # n = 2.7 stays out (step 9 tests it)
    by = {c.name.rsplit("-", 1)[1]: c for c in cells if c.n_devices == 6}
    assert by["def"].interference_amp_db is None and by["def"].burst_duty is None
    assert by["a17"].interference_amp_db == 17.0
    assert (by["b40"].burst_duty, by["b40"].burst_mean_s, by["b40"].burst_depth_db) == (
        0.4, 6.7, 17.0)
    assert by["b10"].burst_duty == 0.1 and by["b10"].interference_amp_db is None


@pytest.mark.parametrize("kw, match", [
    (dict(burst_duty=0.1), "together"),
    (dict(burst_duty=1.0, burst_mean_s=6.7, burst_depth_db=17.0), "burst_duty"),
    (dict(contact_repoll="yes"), "bool"),
    (dict(session_resumable=True), "session_t_out_s"),
    (dict(session_t_out_s=0.0), "session_t_out_s"),
    (dict(interference_shared=""), "non-empty"),
])
def test_a_cell_refuses_a_malformed_ddil_field(kw, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(C.CELLS[0], **kw)


def _repolled(root):
    """Each mission's Pass-1 ``repolled`` lists, in order, from the kept mule trace."""
    (trace,) = [d for d in root.rglob("*__t*__s*") if d.is_dir()]
    out = []
    for line in (trace / "mule-exp4-mule.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("event") == "mission_completed":
            out.append([list(f.get("repolled") or ()) for f in e.get("pass_1_flown") or ()])
    return out


def test_ferrysim_reads_a_re_polled_member_as_a_target(tmp_path):
    """Step 1's re-poll solicits members below the floor at arrival; the mule
    records them apart from the arrival's targets, and FerrySim's reading of a
    stop takes both, so a re-polled member that is collected or dropped is a
    target's outcome (it raised "collected members that were not targets")."""
    from experiments.ferrysim.episode import Policy, run_episode

    cell = C.cell_named("ddl-n6-150-a17")
    seen = 0
    logging.disable(logging.WARNING)
    try:
        for i, seed in enumerate(C.stream_seeds(C.VAL_STREAM, cell.name, 3)):
            root = tmp_path / str(i)
            result = run_episode(cell, seed, Policy.of_arm("F"), trial_index=i,
                                 driver_overrides={"trace_root": str(root)})
            for sortie, flown in zip(result.sorties, _repolled(root)):
                assert len(sortie.stops) == len(flown)
                for stop, repolled in zip(sortie.stops, flown):
                    assert set(repolled) <= set(stop.targets)
                    seen += len(repolled)
    finally:
        logging.disable(logging.NOTSET)
    assert seen, "no member was re-polled: the test no longer reaches the case"


# --------------------------------------------------------------------------- #
# Training records the switches; the runner holds a checkpoint to them
# --------------------------------------------------------------------------- #

def test_a_run_flies_one_set_of_step_1s_switches_and_records_it():
    plain = T.TrainSpec(family="jittery56")
    assert plain.ddil_switches == {} and "ddil_switches" not in plain.to_json()
    s1 = T.TrainSpec(family="jittery56-s1")
    assert s1.ddil_switches == S1 and s1.to_json()["ddil_switches"] == S1
    assert s1.family_sha256 == C.family_sha256("jittery56-s1")
    with pytest.raises(ValueError, match="step 1's switches"):
        T.TrainSpec(family="mixed", cells=(C.FAMILIES["jittery"][0],
                                           C.FAMILIES["jittery56-s1"][0]))


def test_trained_ddil_reads_the_spec_and_ddil_differences_compares_switches():
    assert K.trained_ddil({}) is None
    assert K.trained_ddil({"training": {"spec": {}}}) == {}
    assert K.trained_ddil({"training": {"spec": {"ddil_switches": S1}}}) == S1
    with pytest.raises(ValueError):
        K.trained_ddil({"training": {"spec": {"ddil_switches": [1]}}})
    same = dict(S1, session_t_out_s=10, session_resumable=False)
    assert K.ddil_differences(S1, same) == {}
    assert K.ddil_differences(S1, {}) == {
        "session_t_out_s": (10.0, None), "contact_repoll": (True, None),
        "interference_shared": ("narrow", None), "channel_estimate": ("causal", None)}
    assert K.ddil_differences(S1, dict(S1, interference_shared="wide")) == {
        "interference_shared": ("narrow", "wide")}


def test_the_runner_flies_a_step_1_checkpoint_only_on_its_own_switches(monkeypatch):
    from experiments.exp4 import runner_main as RM
    from hermes.scheduler.selector import pair_q

    manifests = {"rec.npz": {"training": {"spec": {}}},
                 "s1.npz": {"training": {"spec": {"ddil_switches": S1}}}}
    monkeypatch.setattr(pair_q, "verify_checkpoint", lambda path: manifests[path])

    def driver(path, physics):
        return SimpleNamespace(pair_checkpoints={"g75": path}, ferry_physics=physics)

    RM._check_checkpoint_ddil(SimpleNamespace(pair_checkpoints={}, ferry_physics={}))
    RM._check_checkpoint_ddil(driver("rec.npz", dict(S1)))        # FQ-g75 on step 1's model
    RM._check_checkpoint_ddil(driver("rec.npz", {}))
    RM._check_checkpoint_ddil(driver("s1.npz", dict(S1, contact_regime="jittery")))
    with pytest.raises(ValueError, match="step-1 switches"):
        RM._check_checkpoint_ddil(driver("s1.npz", {}))
    with pytest.raises(ValueError, match="channel_estimate"):
        RM._check_checkpoint_ddil(driver("s1.npz", {k: v for k, v in S1.items()
                                                    if k != "channel_estimate"}))


# --------------------------------------------------------------------------- #
# The launcher's variants
# --------------------------------------------------------------------------- #

def _batch3(tmp_path, s515):
    data = copy.deepcopy(_filled_settings().data)
    data["s515"].update(arms=[], amps_db=[], shadow_sigmas_db=[], mission_backhaul_arms=[],
                        backhaul="mission", **s515)
    return [j for j in L.build("batch3", L.Settings(data), ["s515"], str(tmp_path))
            if j.study == "s515" and "_dx" in j.name]


def test_a_learned_variant_flies_its_own_checkpoint_under_its_own_name(tmp_path):
    cells = [{"tag": "_dx", "args": ["--session-t-out-s", "10"], "arms": ["FQ-g75", "FQ-s1"]},
             {"tag": "_dx", "args": [], "arms": ["FQ-s1", "FX"], "n": 12}]
    jobs = _batch3(tmp_path, {"d1_cells": cells, "learned_arms": {
        "FQ-s1": {"arm": "FQ-g75", "checkpoint": "results/x/g75/g0.75_s4.npz"}}})
    by = {j.name.split("/")[-1]: j for j in jobs}
    assert set(by) == {"n6k1_knee_dx__FQ-g75", "n6k1_knee_dx__FQ-s1", "n12k1_knee_dx__FQ-s1",
                       "n12k1_knee_dx__FX"}
    mine = by["n6k1_knee_dx__FQ-s1"].args
    assert mine[mine.index("--pair-checkpoint") + 1] == "g75=results/x/g75/g0.75_s4.npz"
    theirs = by["n6k1_knee_dx__FQ-g75"].args
    assert theirs[theirs.index("--pair-checkpoint") + 1] == "g75=ck/best.npz"   # rl.checkpoints.best
    assert "--session-t-out-s" in mine and not by["n6k1_knee_dx__FQ-s1"].blocked
    unset = _batch3(tmp_path, {"d1_cells": cells[:1],
                               "learned_arms": {"FQ-s1": {"arm": "FQ-g75"}}})
    blocked = [j for j in unset if j.name.endswith("__FQ-s1")]
    assert blocked and blocked[0].blocked == ["s515.learned_arms.FQ-s1.checkpoint"]


def test_the_sweep_trains_fq_s1_and_reads_it_on_step_1s_cells(tmp_path):
    data = copy.deepcopy(_filled_settings().data)
    s1_cells = [f"{c.name}-s1" for c in C.FAMILIES["jittery"]]
    data["rl"].update(family="jittery56-s1", sweep_gammas=[0.75])
    data["rl"]["val_episodes"]["jittery56-s1"] = 400
    jobs = {j.name: j for j in L.build("rl-sweep", L.Settings(data), None, str(tmp_path))}
    trains = [j for j in jobs.values() if j.kind == "rl-train"]
    assert len(trains) == 10 and all("--family" in j.args for j in trains)
    a = trains[0].args
    assert a[a.index("--family") + 1] == "jittery56-s1" and a[a.index("--val-episodes") + 1] == "400"
    ev = jobs["5.5-jittery56-s1/evaluate"].args
    assert "--cells" not in ev                          # evaluate's default: the jittery cells
    data["rl"]["eval_cells"] = s1_cells
    jobs = {j.name: j for j in L.build("rl-sweep", L.Settings(data), None, str(tmp_path))}
    ev = jobs["5.5-jittery56-s1/evaluate"].args
    assert ev[ev.index("--cells") + 1:ev.index("--cells") + 5] == s1_cells
    assert all(name in C.CELLS_BY_NAME for name in s1_cells)


def test_an_arm_variant_flies_its_flags_under_its_own_name(tmp_path):
    cells = [{"tag": "_dx", "args": ["--interference-amp-db", "17"], "arms": ["F", "F-rp"]}]
    jobs = _batch3(tmp_path, {"d1_cells": cells, "arm_variants": {
        "F-rp": {"arm": "F", "args": ["--dwell-pricing", "expected", "--leave-by", "true"]}}})
    by = {j.name.split("/")[-1]: j for j in jobs}
    assert set(by) == {"n6k1_knee_dx__F", "n6k1_knee_dx__F-rp"}
    rp, f = by["n6k1_knee_dx__F-rp"].args, by["n6k1_knee_dx__F"].args
    assert rp[rp.index("--arms") + 1] == f[f.index("--arms") + 1] == "F"
    assert "--dwell-pricing" in rp and "--leave-by" in rp and "--dwell-pricing" not in f
    assert "--interference-amp-db" in rp and "--pair-checkpoint" not in rp
