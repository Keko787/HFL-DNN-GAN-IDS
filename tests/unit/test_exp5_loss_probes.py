"""Exp 5's loss-probe scorer (scripts/exp5/loss_probes.py).

The exact McNemar test, the reach comparison (paired by trial and seed, claiming for the
arm that reaches tau more often), time to tau's survivor bias and the reach check on it,
a probe's stand-in cells and families, and that every registered probe's flights build."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.test_exp5_scoring import EXP5, SC, L, _load, _plan
from tests.unit.test_exp5_scoring import _entry as _unloaded

LP = _load("exp5_loss_probes", EXP5 / "loss_probes.py")
TAU = 0.82                                   # the scoring tests' fixtures write tau 0.82


def _entry(tmp: Path, variant: str, values, **kw):
    """The scoring tests' entry (None = never reached tau), its rows loaded."""
    e = _unloaded(tmp, variant, values, **kw)
    SC.load(e)
    return e


def test_mcnemar_exact_is_the_two_sided_binomial_on_the_discordant_pairs():
    assert LP.mcnemar_exact(0, 0) == 1.0
    assert LP.mcnemar_exact(0, 6) == pytest.approx(2 * 0.5 ** 6)
    assert LP.mcnemar_exact(6, 0) == LP.mcnemar_exact(0, 6)
    assert LP.mcnemar_exact(1, 5) == pytest.approx(2 * (1 + 6) / 64)
    assert LP.mcnemar_exact(3, 3) == 1.0                     # capped at 1


def test_reach_pairs_by_seed_and_claims_for_the_arm_that_reaches_tau_more_often(tmp_path):
    plan = _plan()
    f = _entry(tmp_path, "F", [100.0] * 12 + [100.0])
    d = _entry(tmp_path, "D1", [None] * 12 + [100.0],
               seeds=[f"s{i}" for i in range(12)] + ["other"])
    c = LP.compare_reach(f, d, TAU, "fam", plan, SC)
    assert c.n_pairs == 12 and c.pairing_errors == 1        # the last trial's seeds differ
    assert c.ref_mean == 1.0 and c.variant_mean == 0.0 and c.mean_diff == 1.0
    assert c.p_value == pytest.approx(2 * 0.5 ** 12)
    assert "12 reached by F only, 0 by D1 only" in c.note
    SC.holm([c], plan)
    assert c.claim and c.favours == "F"


def test_equal_reach_is_no_claim(tmp_path):
    plan = _plan()
    f = _entry(tmp_path, "F", [100.0, None] * 6)
    d = _entry(tmp_path, "D1", [150.0, None] * 6)
    c = LP.compare_reach(f, d, TAU, "fam", plan, SC)
    SC.holm([c], plan)
    assert c.p_value == 1.0 and (c.ci_low, c.ci_high) == (0.0, 0.0) and not c.claim


def test_time_to_tau_can_favour_the_arm_that_reaches_it_less_often(tmp_path):
    """F is fast when it gets there but gets there in half the trials; the baseline always
    gets there, more slowly. Time to tau sees only F's six survivors and claims for F;
    reach claims for the baseline, and reach_contradicts flags the time claim."""
    plan = _plan()
    f = _entry(tmp_path, "F", [100.0 + i for i in range(6)] + [None] * 6)
    d = _entry(tmp_path, "D1", [200.0 + i for i in range(12)])
    t = LP.compare_time(f, d, TAU, "time", plan, SC)
    r = LP.compare_reach(f, d, TAU, "reach", plan, SC)
    SC.holm([t, r], plan)
    assert t.n_pairs == 6 and t.claim and t.favours == "F"
    assert r.n_pairs == 12 and r.claim and r.favours == "D1"
    assert LP.reach_contradicts(t.favours, r)
    assert not LP.reach_contradicts("D1", r)


def _empty(tmp: Path, variant: str):
    e = SC.Entry("sx", "n6k1_knee", variant, variant, tmp / "none.csv", tmp / "none_scored.csv", 20)
    SC.load(e)
    return e


def test_a_probe_borrows_the_baselines_of_its_stand_in_cell(tmp_path):
    """F flies the short-reach cell; the baselines (wide-band) are read from the base cell."""
    plan = _plan()
    base, short = tmp_path / "base", tmp_path / "short"
    base.mkdir()
    short.mkdir()
    probe = LP.Probe("PX", "test", [], ref_cells={"n6k1_knee_1mb_ratio2": "n6k1_knee_1mb"})
    cells = {"n6k1_knee_1mb": {"F": _entry(base, "F", [100.0] * 8),
                               "D1": _entry(base, "D1", [150.0] * 8)},
             "n6k1_knee_1mb_ratio2": {"F": _entry(short, "F", [120.0] * 8),
                                      "D1": _empty(short, "D1")}}
    comps = LP.probe_comparisons(probe, cells, plan, SC, tau=TAU)
    short_cmp = [c for c in comps if c.cell == "n6k1_knee_1mb_ratio2"]
    assert {(c.variant, c.metric) for c in short_cmp} == {("D1", "sim_s_to_tau0.82"),
                                                          ("D1", "reached_tau0.82")}
    t = next(c for c in short_cmp if c.metric.startswith("sim"))
    assert t.n_pairs == 8 and t.mean_diff == pytest.approx(-30.0)   # F 120 s against D1 150 s


def test_a_probe_keeps_ferry_arms_and_each_metric_in_families_of_their_own(tmp_path):
    plan = _plan()
    probe = LP.Probe("PX", "test", [])
    cells = {"seconds backhaul: n6k1_knee": {"F": _entry(tmp_path, "F", [100.0] * 8),
                                             "FX": _entry(tmp_path, "FX", [90.0] * 8),
                                             "D1": _entry(tmp_path, "D1", [150.0] * 8)}}
    comps = LP.probe_comparisons(probe, cells, plan, SC, tau=TAU)
    assert {c.study for c in comps} == {"PX/baselines/time", "PX/baselines/reach",
                                        "PX/ferry/time", "PX/ferry/reach"}
    assert all(c.cell == "seconds backhaul: n6k1_knee" for c in comps)


def test_a_cell_without_its_reference_is_not_compared(tmp_path):
    plan = _plan()
    cells = {"n6k1_knee": {"F": _empty(tmp_path, "F"),
                           "D1": _entry(tmp_path, "D1", [150.0] * 8)}}
    assert LP.probe_comparisons(LP.Probe("PX", "test", []), cells, plan, SC, tau=TAU) == []


def test_p10_compares_a_baseline_with_f_pinned_f_and_itself_on_wide(tmp_path):
    plan = _plan()
    pool = {name: _entry(tmp_path, name, [t] * 8)
            for name, t in (("F", 100.0), ("F@narrow", 105.0), ("H1@wide", 150.0),
                            ("H1@narrow", 110.0))}
    comps = LP.p10_comparisons({"knee": pool}, plan, SC, tau=TAU)
    times = {c.reference: c for c in comps if c.metric.startswith("sim")}
    assert set(times) == {"F", "F@narrow", "H1@wide"}
    assert times["F"].mean_diff == pytest.approx(-10.0)
    assert times["F@narrow"].mean_diff == pytest.approx(-5.0)
    assert times["H1@wide"].mean_diff == pytest.approx(40.0)
    assert len({c.study for c in comps}) == 6                # three references, two metrics


def test_p10_flights_build_their_cells_at_40_trials():
    want = [(LP.P10_WIDE, {"H1", "D4"}), (LP.P10_FLOWN["narrow"], {"H1", "D4"}),
            (LP.P10_FLOWN["medium"], {"H1", "D4"}),
            (LP.P10_PINNED[0], {"FBpnarrow", "FBpmedium"}),
            (LP.P10_PINNED[1], {"FBpnarrow", "FBpmedium"})]
    for flight, variants in want:
        cells, _ = LP.flight_cells(L, SC, flight)
        assert cells and all(set(vs) == variants for vs in cells.values()), flight.out_root
        assert all(e.trials == 40 for vs in cells.values() for e in vs.values())
    assert set(LP.flight_cells(L, SC, LP.P10_FLOWN["narrow"])[0]) == {"n6k1_knee", "n6k1_stress"}
    assert set(LP.flight_cells(L, SC, LP.P10_PINNED[1])[0]) == {"n6k1_stress_1mb"}


def test_p11_reads_the_ten_mission_compute_cells_at_40_trials():
    cells, _ = LP.flight_cells(L, SC, LP.PROBES["P11"].flights[0])
    assert set(cells) == {"n6k1_knee_1mb_spread", "n6k1_knee_1mb_stragglers"}
    assert all(set(vs) == {"F", "FX", "H1", "D3", "D4", "D5"} for vs in cells.values())
    assert all(e.trials == 40 for vs in cells.values() for e in vs.values())
    assert all("exp5_rd10" in str(e.csv_path) for vs in cells.values() for e in vs.values())


@pytest.mark.parametrize("name", sorted(LP.PROBES))
def test_every_registered_probe_builds_its_cells_with_f_beside_its_arms(name):
    for flight in LP.PROBES[name].flights:
        cells, plan = LP.flight_cells(L, SC, flight)
        assert cells, f"{name} / {flight.label or flight.study}: no cells"
        with_f = [c for c, vs in cells.items() if "F" in vs]
        assert with_f, f"{name} / {flight.label or flight.study}: F in no cell"
        assert all(len(vs) > 1 for vs in cells.values())


def _trial(tmp: Path, name: str, spans):
    """A mule trace of missions (start, end, merged), with noise the reader must skip."""
    folder = tmp / "F_traces" / name
    folder.mkdir(parents=True)
    rows = [{"event": "mule_ready"}]
    rows += [{"event": "mission_completed", "sim_start_s": a, "sim_end_s": b,
              "pass_1_merged_updates": m} for a, b, m in spans]
    (folder / "mule-exp4-mule.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                                 encoding="utf-8")


def test_a_flight_runs_from_first_takeoff_to_last_landing(tmp_path):
    _trial(tmp_path, "a", [(5.0, 100.0, 1), (100.0, 205.0, 2)])
    _trial(tmp_path, "b", [(0.0, 50.0, 0)])
    e = SC.Entry("sx", "c", "F", "F", tmp_path / "F.csv", tmp_path / "F_scored.csv", 20)
    assert sorted(LP.trial_flights(e)) == [(50.0, 0, 1), (200.0, 3, 2)]


def test_as_of_keeps_early_arrivals_and_counts_late_ones_and_misses_at_the_horizon(tmp_path):
    e = _entry(tmp_path, "F", [100.0, 300.0, None])
    seen = LP.as_of(e, 250.0, SC, tau=TAU)
    assert [SC._num(r[f"sim_s_to_tau{TAU:g}"]) for r in seen.scored_rows] == [100.0, 250.0, 250.0]
    assert [SC._num(r[f"reached_tau{TAU:g}"]) for r in seen.scored_rows] == [1.0, 0.0, 0.0]
    assert [SC._num(r[f"sim_s_to_tau{TAU:g}"]) for r in e.scored_rows][1] == 300.0   # untouched


def test_the_horizon_finds_the_gap_time_to_tau_drops(tmp_path):
    """F is at tau early in some trials and never in the rest; the baseline is there a
    little later in all of them. Over the trials both reach, F looks faster; at the horizon
    the misses count and the claim goes to the baseline."""
    plan = _plan()
    f = _entry(tmp_path, "F", [100.0] * 6 + [None] * 14)
    d = _entry(tmp_path, "D1", [150.0 + i for i in range(20)])
    t = LP.compare_time(f, d, TAU, "time", plan, SC)
    h = LP.compare_time(LP.as_of(f, 400.0, SC, tau=TAU), LP.as_of(d, 400.0, SC, tau=TAU), TAU,
                        "horizon", plan, SC)
    SC.holm([t, h], plan)
    assert t.claim and t.favours == "F"
    assert h.claim and h.favours == "D1"


def test_the_delay_cost_is_the_difference_of_means_inside_its_ci():
    d, lo, hi = LP.delay_cost([110.0, 120.0, 130.0, 140.0], [100.0, 105.0, 95.0], n=2000)
    assert d == pytest.approx(25.0)
    assert lo < d < hi
    assert lo > 0.0


def test_the_no_delay_cell_is_in_p11s_baseline_flight():
    cells, _ = LP.flight_cells(L, SC, LP.P11_NO_DELAY)
    assert set(cells[LP.P11_NO_DELAY_CELL]) == {"F", "FX", "H1", "D3", "D4", "D5"}
