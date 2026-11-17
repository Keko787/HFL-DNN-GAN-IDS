"""The DDIL scoring scripts (scripts/exp5/ddil_score_probes.py, ddil_step1_beside_recorded.py):
the parallel scorer's plan, and step 1's table beside the recorded numbers, which must
regenerate from the committed scored files."""

from __future__ import annotations

import pytest

from tests.unit.test_exp5_scoring import EXP5, _load

SP = _load("exp5_ddil_score_probes", EXP5 / "ddil_score_probes.py")
BR = _load("exp5_ddil_step1_beside_recorded", EXP5 / "ddil_step1_beside_recorded.py")
LP = _load("exp5_loss_probes_scripts", EXP5 / "loss_probes.py")


def test_the_table_reads_the_probes_tau():
    assert BR.TAU == LP.TAU


def test_an_arm_is_paired_with_itself_on_the_seeds_both_have():
    before = {("s0", "0"): {"x": "10"}, ("s1", "1"): {"x": "20"}, ("s2", "2"): {"x": ""}}
    after = {("s0", "0"): {"x": "7"}, ("s1", "1"): {"x": "14"}, ("s2", "2"): {"x": "1"},
             ("s9", "9"): {"x": "0"}}
    mean_b, mean_a, diff, lo, hi, n = BR.paired(before, after, "x", seed=1)
    assert (mean_b, mean_a, diff, n) == (15.0, 10.5, -4.5, 2)     # s2 blank, s9 unpaired
    assert -6.0 <= lo <= diff <= hi <= -3.0
    assert BR.paired(before, {}, "x", seed=1) is None


def test_step_1s_table_regenerates_from_the_committed_scored_files():
    if not (BR.RECORDED / "n6k1_knee_ampharsh__F_scored_ddil.csv").exists():
        pytest.skip("the DDIL scored files are not in this checkout")
    assert BR.main(["--check"]) == 0


def test_the_parallel_scorer_plans_each_flight_and_scores_a_shared_file_once():
    got = SP.plan(["DDILFQ1", "DDIL2"])
    assert [(f.probe, f.flight.label, f.entries, f.scored_files) for f in got] == [
        ("DDILFQ1", "", 44, 44), ("DDIL2", "honest", 83, 83), ("DDIL2", "recorded", 27, 27)]
    outs = [j.out for f in got for j in f.todo]
    assert len(outs) == len(set(outs))
    assert all(o.endswith("_scored_ddil.csv") for o in outs)
    assert all("--ddil-columns" in j.args for f in got for j in f.todo)


def test_the_parallel_scorer_refuses_an_unknown_probe(capsys):
    with pytest.raises(SystemExit):
        SP.main(["--plan", "NOPE"])
    assert "not a probe: NOPE" in capsys.readouterr().err
