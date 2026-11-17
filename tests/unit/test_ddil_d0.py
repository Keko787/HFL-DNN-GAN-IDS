"""DDIL build D0's scorer entries (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md, row 2 of the
task table): hover and updates per mission as paired endpoints, and time to tau with a
missed trial given the trial's end.

* the trace scorer's ``--ddil-columns``: the trial's simulated end, and each tau's time
  with a miss at that end, last in the row and only with the flag;
* scoring.py's ``_ddil`` scored files, which the DDIL probes ask for through a score
  setting that moves no runner job;
* the probe scorer's D0 endpoints and its other pairs, each in a Holm family of its own,
  with every other probe's comparisons as they were.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io

import pytest

from experiments.analysis import traces_scorer as TS
from tests.unit.test_exp5_scoring import EXP5, SC, L, _load, _plan
from tests.unit.test_p3_sim_scoring import (
    E, SIM_TRIAL, WALL_TRIAL, _cluster_rows, _obs, _write_trial,
)

LP = _load("exp5_loss_probes_d0", EXP5 / "loss_probes.py")
COLS = ("sim_s_trial_end", "sim_s_to_tau0.82_or_end", "sim_s_to_tau0.9_or_end")


# --------------------------------------------------------------------------- #
# The trace scorer's --ddil-columns
# --------------------------------------------------------------------------- #

def test_the_trial_end_is_the_later_of_the_last_landing_and_the_last_evaluation():
    """The fixture's three missions land at 500 s and its last evaluation is at 230 s
    (both after the first takeoff, at 0 s); an evaluation after the landing moves it."""
    assert TS.trial_sim_end_s(_obs("sim")) == pytest.approx(500.0)
    assert TS.trial_sim_end_s(_obs("wall")) is None
    late = _cluster_rows() + [{"ts": 1_700_000_019.0, "event": "model_eval", "role": "cluster",
                               "id": "c1", "sim_ts": E + 520.0, "cluster_round": 3,
                               "accuracy": 0.86, "auc": 0.9, "loss": 0.4, "n_test": 100}]
    assert TS.trial_sim_end_s(_obs("sim", cluster_rows=late)) == pytest.approx(520.0)


def test_a_reached_tau_keeps_its_time_and_a_miss_gets_the_trial_end():
    obs = _obs("sim")
    report = TS.ddil_report(obs, [TS.tau_reach(obs, 0.82), TS.tau_reach(obs, 0.9)])
    assert report.sim_s_trial_end == pytest.approx(500.0)
    assert report.to_tau_or_end == ((0.82, 230.0), (0.9, pytest.approx(500.0)))
    assert TS.ddil_columns((0.82, 0.9)) == COLS
    wall = _obs("wall")
    assert TS.ddil_report(wall, [TS.tau_reach(wall, 0.82)]).to_row() == {
        "sim_s_trial_end": "", "sim_s_to_tau0.82_or_end": ""}


def test_the_ddil_columns_come_last_and_only_with_the_flag(tmp_path):
    d = _write_trial(tmp_path, SIM_TRIAL)
    plain = TS.score_trial(d, taus=(0.82, 0.9)).to_row()
    assert not set(COLS) & set(plain)
    flagged = TS.score_trial(d, taus=(0.82, 0.9), ddil_columns=True).to_row()
    assert tuple(flagged)[-3:] == COLS
    assert {k: v for k, v in flagged.items() if k not in COLS} == plain   # nothing else moves
    assert (flagged["sim_s_to_tau0.82"], flagged["sim_s_to_tau0.82_or_end"]) == (230.0, 230.0)
    assert (flagged["sim_s_to_tau0.9"], flagged["sim_s_to_tau0.9_or_end"]) == (
        "", pytest.approx(500.0))
    wall = TS.score_trial(_write_trial(tmp_path, WALL_TRIAL, clock="wall"), taus=(0.82, 0.9),
                          ddil_columns=True).to_row()
    assert list(wall) == list(flagged) and all(wall[c] == "" for c in COLS)


def test_the_cli_takes_the_ddil_flag(tmp_path):
    _write_trial(tmp_path / "root", SIM_TRIAL)
    out = tmp_path / "scored.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        TS.main(["--traces", str(tmp_path / "root"), "--tau", "0.82", "0.9", "--ddil-columns",
                 "--csv", str(out)])
    row, = csv.DictReader(out.open(encoding="utf-8"))
    assert float(row["sim_s_to_tau0.9_or_end"]) == pytest.approx(500.0)
    assert float(row["sim_s_trial_end"]) == pytest.approx(500.0)


# --------------------------------------------------------------------------- #
# scoring.py and the DDIL probes' flights
# --------------------------------------------------------------------------- #

def test_a_study_may_ask_for_the_ddil_columns_into_a_scored_file_of_its_own():
    plan = _plan(scorer_flags=["--ddil-columns"])
    args, out, missing = SC.scorer_call("r/s515/c__F.csv", plan, plan.specs["sx"], None)
    assert "--ddil-columns" in args and out == "r/s515/c__F_scored_ddil.csv" and not missing
    plain = _plan()
    assert SC.scorer_call("r/s515/c__F.csv", plain, plain.specs["sx"], None)[1] == \
        "r/s515/c__F_scored.csv"


def _flight_jobs(flight, sets=None):
    s, _ = L.load_settings(L.PARAMS)
    ns = argparse.Namespace(trials=flight.trials, set=list(flight.sets if sets is None else sets),
                            dataset=None, **{o: None for o, _, _ in L.LEVERS})
    with contextlib.redirect_stdout(io.StringIO()):
        L.apply_levers(s, ns)
        jobs = L.build(flight.stage, s, None, flight.out_root)
    return s, jobs


@pytest.mark.parametrize("name", ["DDIL1", "DDILFQ1", "DDIL2"])
def test_the_ddil_probes_score_into_ddil_files_and_fly_the_jobs_they_flew(name):
    probe = LP.PROBES[name]
    assert probe.ddil_endpoints
    for flight in probe.flights:
        assert LP.DDIL_SCORE_SET in flight.sets
        s, jobs = _flight_jobs(flight)
        _, without = _flight_jobs(flight, [x for x in flight.sets if x != LP.DDIL_SCORE_SET])
        assert [(j.name, j.args) for j in jobs] == [(j.name, j.args) for j in without]
        with contextlib.redirect_stdout(io.StringIO()):
            _, entries, _, _ = L.score_plan(flight.stage, s, jobs)
        assert entries["s515"] and all(str(e.scored).endswith("_scored_ddil.csv")
                                       for e in entries["s515"])


def test_p1_still_scores_into_its_recorded_files():
    probe = LP.PROBES["P1"]
    assert not probe.ddil_endpoints and not probe.pairs
    for flight in probe.flights:
        s, jobs = _flight_jobs(flight)
        with contextlib.redirect_stdout(io.StringIO()):
            _, entries, _, _ = L.score_plan(flight.stage, s, jobs)
        assert entries["s515"] and all(str(e.scored).endswith("_scored.csv")
                                       for e in entries["s515"])


# --------------------------------------------------------------------------- #
# The probe scorer's endpoints and pairs
# --------------------------------------------------------------------------- #

def _rows(n, *, time, hover, updates, end=600.0, miss=()):
    return [{"cell_id": "c", "trial_index": str(i), "seed": f"s{i}", "status": "ok",
             "reached_tau0.71": "0" if i in miss else "1",
             "sim_s_to_tau0.71": "" if i in miss else str(time + i),
             "sim_s_to_tau0.71_or_end": str(end if i in miss else time + i),
             "sim_dwell_s_mean": str(hover + 0.5 * (i % 3)), "update_yield": str(updates)}
            for i in range(n)]


def _entry(variant, rows, cell="n6k1_knee_d1x"):
    e = SC.Entry("s515", cell, variant, variant, None, None, len(rows))
    e.scored_rows = rows
    return e


def _cells():
    return {"n6k1_knee_d1x": {
        "F": _entry("F", _rows(12, time=300.0, hover=130.0, updates=3.3)),
        "FX": _entry("FX", _rows(12, time=290.0, hover=110.0, updates=3.4)),
        "FQ-s1": _entry("FQ-s1", _rows(12, time=295.0, hover=112.0, updates=3.4, miss=(3,))),
        "FQ-g75": _entry("FQ-g75", _rows(12, time=296.0, hover=113.0, updates=3.4)),
        "H1": _entry("H1", _rows(12, time=400.0, hover=9.0, updates=2.6, miss=(1, 5))),
    }}


def _probe(**kw):
    return LP.Probe("PX", "a test probe", [], **kw)


def test_a_probe_without_d0_compares_time_and_reach_as_before():
    comps = LP.probe_comparisons(_probe(), _cells(), _plan(), SC)
    assert sorted({c.study for c in comps}) == ["PX/baselines/reach", "PX/baselines/time",
                                                "PX/ferry/reach", "PX/ferry/time"]
    assert len(comps) == 4 * 2 and all(c.reference == "F" for c in comps)


def test_d0_adds_hover_updates_and_time_with_misses_each_in_its_own_family():
    comps = LP.probe_comparisons(_probe(ddil_endpoints=True), _cells(), _plan(), SC)
    kinds = {LP.kind_of(c.metric) for c in comps}
    assert kinds == {"time", "reach", "hover", "updates", "time_or_end"}
    by = {(c.variant, LP.kind_of(c.metric)): c for c in comps}
    hover = by[("H1", "hover")]
    assert (hover.study, hover.better, hover.n_pairs) == ("PX/baselines/hover", "lower", 12)
    assert hover.mean_diff == pytest.approx(121.0)                 # F hovers 121 s more
    assert by[("FX", "updates")].better == "higher"
    # Time to tau drops H1's two misses; with each miss at the trial's end, none is dropped.
    assert by[("H1", "time")].n_pairs == 10
    assert by[("H1", "time_or_end")].n_pairs == 12
    assert by[("H1", "time_or_end")].metric == "sim_s_to_tau0.71_or_end"
    assert all(c.p_holm is not None for c in comps if c.p_value is not None)


def test_a_probes_pairs_are_compared_in_families_of_their_own():
    probe = _probe(ddil_endpoints=True, pairs=[("FQ-s1", "FQ-g75"), ("FQ-s1", "FX"),
                                               ("FQ-d", "FX")])           # FQ-d not flown
    comps = LP.probe_comparisons(probe, _cells(), _plan(), SC)
    pairs = [c for c in comps if c.study.startswith("PX/pairs/")]
    assert {(c.variant, c.reference) for c in pairs} == {("FQ-s1", "FQ-g75"), ("FQ-s1", "FX")}
    assert len(pairs) == 2 * 5
    hover = next(c for c in pairs if c.reference == "FQ-g75" and LP.kind_of(c.metric) == "hover")
    assert hover.mean_diff == pytest.approx(1.0)                   # FQ-g75 - FQ-s1
    # The pairs leave the comparisons against F as they are.
    alone = LP.probe_comparisons(_probe(ddil_endpoints=True), _cells(), _plan(), SC)
    assert [c.row() for c in comps if not c.study.startswith("PX/pairs/")] == \
        [c.row() for c in alone]


def test_the_report_tables_name_every_endpoint_and_pair():
    probe = _probe(ddil_endpoints=True, pairs=[("FQ-s1", "FQ-g75")])
    cells = _cells()
    comps = LP.probe_comparisons(probe, cells, _plan(), SC)
    by = {(c.cell, c.variant, c.reference, LP.kind_of(c.metric)): c for c in comps}
    table = LP._d0_table("n6k1_knee_d1x", cells["n6k1_knee_d1x"], {}, None, by, SC)
    assert table[3].count("|") == table[4].count("|") == 14
    f_row = next(x for x in table if x.startswith("| F |"))
    assert "130.5" in f_row and "3.30" in f_row                    # F's own means, no diffs
    assert all(x.count("|") == 14 for x in table[5:])
    pairs = LP._pairs_table(comps)
    assert any("| FQ-s1 | FQ-g75 | hover per mission |" in x for x in pairs)
    assert LP._pairs_table([c for c in comps if "pairs" not in c.study])[-1] == \
        "- none flown (so far)"


def test_step_2s_variants_are_ferrys_own_arms():
    assert LP.is_ferry("F-rp") and LP.is_ferry("F-rpnl") and LP.is_ferry("FQ-s1")
    assert not LP.is_ferry("F") and not LP.is_ferry("H1") and not LP.is_ferry("D4fedex")
