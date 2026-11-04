"""Exp 5's loss-probe scorer (scripts/exp5/loss_probes.py).

The exact McNemar test, the reach comparison (paired by trial and seed, claiming for the
arm that reaches tau more often), time to tau's survivor bias and the reach check on it,
a probe's stand-in cells and families, and that every registered probe's flights build."""

from __future__ import annotations

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


@pytest.mark.parametrize("name", sorted(LP.PROBES))
def test_every_registered_probe_builds_its_cells_with_f_beside_its_arms(name):
    for flight in LP.PROBES[name].flights:
        cells, plan = LP.flight_cells(L, SC, flight)
        assert cells, f"{name} / {flight.label or flight.study}: no cells"
        with_f = [c for c, vs in cells.items() if "F" in vs]
        assert with_f, f"{name} / {flight.label or flight.study}: F in no cell"
        assert all(len(vs) > 1 for vs in cells.values())
