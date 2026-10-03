"""Exp 5 addendum, Study 5.11 (c): FerrySim beyond the stack, on a field that grows with N.

Pinned:

* **The field rule.** ``grown_field_radius_m`` keeps N = 6's density in the
  realism field's 100 m: 200, 282.8 and 400 m at N = 24, 48 and 96, and the
  reference size's own field at its own N. The driver's ``h1_field_ref_n``
  applies it to a trial's devices and to T_nom's reference layouts (needs
  realism; None keeps the recorded fixed field), the runner passes it only
  when ``--h1-field-ref-n`` is given, and the S* tool takes it.
* **The scale family.** Six cells, N = 24, 48 and 96 at the stand-in budgets
  (the binding edge and 1.5 times it), S = 2, on the grown field; out of
  ``CELLS`` and every other family, whose JSON and hashes are unchanged
  (a cell's field is left out of its JSON at the driver's default); the S*
  tool's binding edges reproduce at planning level.
* **The pilot** flies a cell at budgets that override its own, on the
  validation stream, and reports the served share and the wall times; its
  table folds them per (cell, budget, policy). An episode's wall times are
  never part of its equality.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.analysis import age_cap_s_star as SS
from experiments.exp4 import runner_main
from experiments.exp4.driver import Exp4Driver
from experiments.exp4.topology_builder import grown_field_radius_m
from experiments.ferrysim import __main__ as CLI
from experiments.ferrysim import cells as C
from experiments.ferrysim import pilot as P
from experiments.ferrysim.episode import Policy, run_episode

REPO = Path(__file__).resolve().parents[2]
#: The families' hashes before the scale family (test_p5_ferrysim_runner.py's).
JITTERY_SHA256 = "32b5cb6bc119f1e2b13423dd178cef81c4f7032f24bd199b732f53bd296d3e91"
CLEAN_SHA256 = "76955c9b637ab875c761bf0ce21181ce1983d9c5df02b6183e74abbfc2915902"
JITTERY56_SHA256 = "0079de11cdbe8b3fc71f7f6bdd1cf1d859e600031dd625b7996b35b7767fac66"
SCALE_SHA256 = "d410f0d105186a376c1830721dd3bee7d46b49c159e9ff30a0ab46d8c8f770ad"
#: (N, field, binding edge, 1.5 x edge), the cells' module docstring.
SCALE = ((24, 200.0, 350.0, 525.0), (48, 282.8, 680.0, 1020.0), (96, 400.0, 1330.0, 1995.0))


# --------------------------------------------------------------------------- #
# The field rule
# --------------------------------------------------------------------------- #

def test_the_grown_field_keeps_n6s_density():
    assert grown_field_radius_m(100.0, 6, 6) == 100.0
    assert grown_field_radius_m(100.0, 12, 6) == 141.4
    for n, field, _, _ in SCALE:
        assert grown_field_radius_m(100.0, n, 6) == field == C.SCALE_FIELD_M[n]
        # the density: N devices over the square of side 2 x field
        assert n / (2 * field) ** 2 == pytest.approx(6 / 200.0 ** 2, rel=1e-3)
    assert grown_field_radius_m(150.0, 40, 10) == 300.0


@pytest.mark.parametrize("n, ref", [(0, 6), (6, 0), (True, 6), (6, 2.0)])
def test_the_grown_field_refuses_a_size_it_cannot_read(n, ref):
    with pytest.raises(ValueError):
        grown_field_radius_m(100.0, n, ref)


def test_the_driver_grows_the_field_only_when_asked():
    fixed = Exp4Driver(realism=True)
    assert fixed.h1_field_ref_n is None
    assert [fixed.field_radius_m(n) for n in (6, 24, 96)] == [100.0] * 3
    grown = Exp4Driver(realism=True, h1_field_ref_n=6)
    assert [grown.field_radius_m(n) for n in (6, 24, 48, 96)] == [100.0, 200.0, 282.8, 400.0]
    with pytest.raises(ValueError, match="realism"):
        Exp4Driver(h1_field_ref_n=6)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="h1_field_ref_n"):
            Exp4Driver(realism=True, h1_field_ref_n=bad)


def _t_nom(driver, n):
    settings = driver.ferry_settings(arm="F", regime="jittery")
    theta, synth = driver._payload_bytes(None)
    return driver.nominal_period_s(n_devices=n, rf_range_m=60.0, regime="jittery",
                                   settings=settings, theta_bytes=theta, synth_bytes=synth)


def test_t_nom_prices_the_grown_field_as_the_cell_names_it():
    """The stack's rule and the cell's literal field give the same T_nom, and
    one unlike the fixed field's (the reference layouts are drawn on it)."""
    base = dict(C.cell_named("scl-n24-350").driver_settings())
    base.pop("h1_field_radius_m")
    grown = Exp4Driver(**base, h1_field_ref_n=6)
    cell = Exp4Driver(**C.cell_named("scl-n24-350").driver_settings())
    fixed = Exp4Driver(**base)
    assert _t_nom(grown, 24) == _t_nom(cell, 24) > 2 * _t_nom(fixed, 24)
    assert _t_nom(grown, 6) == _t_nom(fixed, 6)


def _driver_kwargs(monkeypatch, argv):
    seen = {}

    class _Stop(Exception):
        pass

    def fake(**kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(runner_main, "Exp4Driver", fake)
    with pytest.raises(_Stop):
        runner_main.main(argv)
    return seen


def test_the_runner_passes_the_field_rule_only_when_given(tmp_path, monkeypatch):
    csv = str(tmp_path / "t.csv")
    plain = _driver_kwargs(monkeypatch, ["--csv", csv, "--arms", "H1", "--realism"])
    assert "h1_field_ref_n" not in plain
    grown = _driver_kwargs(monkeypatch, ["--csv", csv, "--arms", "H1", "--realism",
                                         "--h1-field-ref-n", "6"])
    assert grown.pop("h1_field_ref_n") == 6 and grown == plain


# --------------------------------------------------------------------------- #
# The scale family
# --------------------------------------------------------------------------- #

def test_the_scale_cells_are_the_plans():
    assert C.FAMILY_SCALE == "scale" and C.SCALE_REF_N == 6
    assert [(c.name, c.n_devices, c.budget_s, c.field_radius_m) for c in C.SCALE_CELLS] == [
        (f"scl-n{n}-{b:g}", n, b, field)
        for n, field, edge, wide in SCALE for b in (edge, wide)]
    for c in C.SCALE_CELLS:
        assert (c.family, c.role, c.budget_role, c.cap_s, c.contact_regime) == (
            "scale", C.ROLE_DECISION_RICH, C.BUDGET_STAND_IN, 2, "jittery")
        assert (c.payload_bytes, c.n_missions, c.rf_range_m, c.network_regime) == (
            1_000_000, 4, 60.0, "jittery")
        assert C.cell_named(c.name) is c
        settings = c.driver_settings()
        assert settings["h1_field_radius_m"] == c.field_radius_m
        assert settings["mission_budget_s"] == c.budget_s and settings["realism"] is True
    assert C.FAMILIES["scale"] == C.SCALE_CELLS


def test_the_scale_family_moves_no_other_cell_or_family():
    assert not set(C.SCALE_CELLS) & set(C.CELLS + C.STUDY_5_6_CELLS)
    assert (C.family_sha256("jittery"), C.family_sha256("clean"),
            C.family_sha256("jittery56")) == (JITTERY_SHA256, CLEAN_SHA256, JITTERY56_SHA256)
    assert C.family_sha256("scale") == SCALE_SHA256
    for c in C.CELLS + C.STUDY_5_6_CELLS:
        assert c.field_radius_m is None
        assert "field_radius_m" not in c.to_json()
        assert "h1_field_radius_m" not in c.driver_settings()
    assert C.SCALE_CELLS[0].to_json()["field_radius_m"] == 200.0
    with pytest.raises(ValueError, match="field_radius_m"):
        C.FerryCell("x", "scale", C.ROLE_CONTROL, 6, 60.0, C.BUDGET_STAND_IN, 2, "jittery",
                    field_radius_m=0.0)
    # the training stream draws a scale cell only from its own family
    drawn = {C.train_episode(0, "scale", e)[0].name for e in range(60)}
    assert drawn == {c.name for c in C.SCALE_CELLS}


def _q90(n, budget):
    d = Exp4Driver(mission_clock="sim", realism=True, contact_band="wide",
                   payload_bytes=1_000_000, ferry_physics={"contact_regime": "jittery"},
                   h1_field_ref_n=C.SCALE_REF_N)
    report = SS.s_star_report(d, n_devices=n, budgets=[float(budget)], regime="jittery",
                              families=["F"])
    return report, SS.cover_value([x.s_star for x in report.stars["F"][float(budget)]])


@pytest.mark.parametrize("n, field, edge, wide", [
    SCALE[0], SCALE[1], pytest.param(*SCALE[2], marks=pytest.mark.slow)])
def test_each_scale_budget_is_the_s_star_tools_binding_edge(n, field, edge, wide):
    """F's S* on 90 % of the 30 layouts is 2 at the edge and 1 ten seconds
    above it (and at 1.5 x), so S is decision 1's floor, 2."""
    logging.disable(logging.WARNING)
    try:
        assert _q90(n, edge)[1] == 2 and _q90(n, edge + 10.0)[1] == 1
        report, q = _q90(n, wide)
        assert q == 1 and report.s("F") == 2
    finally:
        logging.disable(logging.NOTSET)


def test_the_s_star_cli_takes_the_field_rule(capsys):
    argv = ["--N", "24", "--budgets", "350", "--contact-band", "wide", "--payload-bytes",
            "1000000", "--ferry-physics", '{"contact_regime": "jittery"}', "--families", "F"]
    assert SS.main(argv + ["--field-ref-n", "6"]) == 0
    grown = capsys.readouterr().out
    assert "90% 2" in grown
    assert SS.main(argv) == 0
    assert "90% 2" not in capsys.readouterr().out      # the 100 m field: one mission serves all
    assert SS.main(argv + ["--field-radius-m", "200"]) == 0
    assert capsys.readouterr().out == grown              # the same field, given whole
    with pytest.raises(SystemExit):
        SS.main(argv + ["--field-ref-n", "6", "--no-realism"])


# --------------------------------------------------------------------------- #
# The pilot
# --------------------------------------------------------------------------- #

CELL = "scl-n24-350"


@pytest.fixture(scope="module")
def flown(tmp_path_factory):
    root = tmp_path_factory.mktemp("pilot")
    logging.disable(logging.WARNING)
    try:
        summaries = P.pilot([CELL], [Policy.of_arm("FX"), Policy.scripted("greedy_1")],
                            budgets=[300.0, 350.0], episodes=1, trace_root=str(root))
    finally:
        logging.disable(logging.NOTSET)
    return root, summaries


def test_the_pilot_flies_each_budget_on_the_same_validation_episode(flown):
    root, summaries = flown
    assert [(s["budget_s"], s["policy"]) for s in summaries] == [
        (300.0, "FX"), (300.0, "greedy_1"), (350.0, "FX"), (350.0, "greedy_1")]
    seeds = {s["seed"] for s in summaries}
    assert seeds == {C.stream_seeds(C.VAL_STREAM, CELL, 1)[0]}
    for s in summaries:
        assert s["stream"] == C.VAL_STREAM and s["n_devices"] == 24
        assert len(s["served_share"]) == len(s["plan_wall_s"]) == s["missions"] == 4
        assert all(0.0 <= v <= 1.0 for v in s["served_share"])
        assert all(v is None or 0.0 <= v <= 1.0 for v in s["served_of_demand"])
        assert s["wall_s"] > 0.0 and all(v > 0.0 for v in s["plan_wall_s"])
        assert len(s["decide_s"]) == len(s["mask_s"])
        assert (len(s["decide_s"]) == sum(s["decisions"])) == (s["policy"] == "greedy_1")
    json.dumps(summaries)
    for budget in ("300", "350"):
        for policy in ("FX", "greedy_1"):
            assert len(list((root / f"budget={budget}" / CELL / policy).iterdir())) == 1


def test_the_pilots_table_folds_each_cell_budget_and_policy(flown):
    _, summaries = flown
    table = P.pilot_table(summaries)
    assert [(r["budget_s"], r["policy"], r["episodes"]) for r in table] == [
        (300.0, "FX", 1), (300.0, "greedy_1", 1), (350.0, "FX", 1), (350.0, "greedy_1", 1)]
    for r, s in zip(table, summaries):
        assert r["served_share_mean"] == pytest.approx(sum(s["served_share"]) / 4)
        assert r["episode_wall_s_mean"] == r["episode_wall_s_max"] == s["wall_s"]
        assert (r["decide_s_mean"] is None) == (s["policy"] == "FX")


def test_the_pilots_table_by_hand():
    def summary(policy, wall, plans, decide, served):
        return {"cell": "c", "budget_s": 60.0, "policy": policy, "n_devices": 4,
                "served_share": served, "served_of_demand": [None] * len(served),
                "return": 0.5, "decisions": [2, 1], "wall_s": wall, "plan_wall_s": plans,
                "decide_s": decide, "mask_s": [d / 2 for d in decide]}
    table = P.pilot_table([summary("FX", 1.0, [0.1, 0.3], [], [0.5, 0.25]),
                           summary("FX", 3.0, [0.2, None], [], [1.0, 0.75])])
    (row,) = table
    assert row["episodes"] == 2 and row["served_share_mean"] == pytest.approx(0.625)
    assert row["episode_wall_s_mean"] == 2.0 and row["episode_wall_s_p95"] == pytest.approx(2.9)
    assert row["plan_wall_s_mean"] == pytest.approx(0.2)
    assert row["plan_wall_s_p95"] == pytest.approx(0.29)
    assert (row["decide_s_mean"], row["decide_s_p95"], row["mask_s_mean"]) == (None,) * 3
    assert row["served_of_demand_mean"] is None and row["decisions_per_sortie_mean"] == 1.5


def test_the_pilot_refuses_what_it_cannot_fly():
    fx = [Policy.of_arm("FX")]
    for kw, match in [(dict(budgets=[0.0]), "budgets"), (dict(budgets=[]), "budgets"),
                      (dict(episodes=0), "episodes"),
                      (dict(driver_overrides={"mission_budget_s": 9.0}), "mission_budget_s"),
                      (dict(driver_overrides={"trace_root": "x"}), "trace_root")]:
        with pytest.raises(ValueError, match=match):
            P.pilot([CELL], fx, **kw)
    with pytest.raises(ValueError, match="policy"):
        P.pilot([CELL], [])
    with pytest.raises(ValueError, match="no FerrySim cell"):
        P.pilot(["scl-n12-120"], fx)


def test_an_episodes_wall_times_are_not_part_of_its_equality():
    cell = C.cell_named(CELL)
    seed = C.stream_seeds(C.VAL_STREAM, CELL, 1)[0]
    logging.disable(logging.WARNING)
    try:
        a = run_episode(cell, seed, Policy.scripted("greedy_1"))
        b = run_episode(cell, seed, Policy.scripted("greedy_1"))
    finally:
        logging.disable(logging.NOTSET)
    assert a.walls != b.walls and a == b
    assert len(a.walls) == len(a.sorties)
    for w, records in zip(a.walls, a.pair_records):
        assert w["plan_wall_s"] > 0.0 and w["e3"] == []
        assert len(w["pairs"]) == len(records or ())
    assert "walls" not in a.summary()


def test_the_command_line_writes_the_pilot(tmp_path, capsys):
    out = tmp_path / "sub" / "pilot.json"
    assert CLI.main(["pilot", "--cells", CELL, "--episodes", "1", "--budgets", "350",
                     "--plan-search-params", '{"exact_max_devices": 0, '
                                             '"exhaustive_max_stops": 0}',
                     "--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["settings"]["overrides"] == {
        "plan_search_params": {"exact_max_devices": 0, "exhaustive_max_stops": 0}}
    assert [r["policy"] for r in saved["table"]] == ["FX"]
    assert "scl-n24-350" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        CLI.main(["pilot", "--help"])
    assert e.value.code == 0
    with pytest.raises(SystemExit):
        CLI.main(["pilot", "--cells", CELL, "--plan-search-params", "[1]"])
