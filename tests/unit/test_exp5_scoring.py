"""Exp 5's score stage (scripts/exp5/scoring.py and launch.py's `score`).

The comparisons on synthetic scored files (pairing, complete cases, Holm, the
claim's direction), the scorer's arguments, and that params.toml's [score]
plan names variants the launcher's jobs actually have."""

from __future__ import annotations

import copy
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXP5 = REPO / "scripts" / "exp5"


def _load(name: str, path: Path):
    if str(EXP5) not in sys.path:
        sys.path.insert(0, str(EXP5))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SC = _load("scoring", EXP5 / "scoring.py")
L = _load("exp5_launch", EXP5 / "launch.py")


def _plan(**over):
    data = {"score": {"tau": [0.82], "alpha": 0.05, "family": "study", "n_bootstraps": 500,
                      "also": ["reached_tau"],
                      "sx": {"metric": "sim_s_to_tau", "reference": "F", **over}}}
    plan, problems = SC.plan_from(data)
    assert problems == []
    return plan


def _write(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _entry(tmp: Path, variant: str, values, *, seeds=None, status=None, trials=None):
    """A job's trial CSV and scored CSV, one row per value (None = never reached tau)."""
    seeds = seeds or [f"s{i}" for i in range(len(values))]
    status = status or ["ok"] * len(values)
    trial = [{"cell_id": "c", "arm": variant, "trial_index": i, "seed": seeds[i],
              "status": status[i]} for i in range(len(values))]
    scored = [{"cell_id": "c", "arm": variant, "trial_index": i, "seed": seeds[i],
               "status": status[i],
               "reached_tau0.82": int(v is not None),
               "sim_s_to_tau0.82": "" if v is None else v}
              for i, v in enumerate(values) if status[i] == "ok"]
    c = tmp / f"n6k1_knee__{variant}.csv"
    s = tmp / f"n6k1_knee__{variant}_scored.csv"
    _write(c, trial)
    _write(s, scored)
    return SC.Entry("sx", "n6k1_knee", variant, variant, c, s, trials or len(values))


def test_params_plan_waits_for_the_pilots_tau():
    s, _ = L.load_settings(L.PARAMS)
    unset = copy.deepcopy(s.data)
    unset["pilot_outputs"].pop("tau", None)                  # before the knee pilot
    _, problems = SC.plan_from(unset)
    assert len(problems) == 1 and "pilot_outputs.tau" in problems[0]
    plan, problems = SC.plan_from(_filled_settings().data)
    assert problems == [] and plan.taus == [0.7, 0.82]


def test_params_plan_parses_and_names_known_flags():
    plan, problems = SC.plan_from(_filled_settings().data)
    assert problems == []
    assert plan.tau == 0.7 and plan.family == "study"
    assert SC.column("sim_s_to_tau", 0.7) == "sim_s_to_tau0.7"
    assert SC.column("network_aou_mean", 0.82) == "network_aou_mean"
    assert plan.specs["s514"].reference_of("secFL1") == "secF"
    assert plan.specs["s514"].reference_of("secF") is None
    assert plan.specs["s514"].reference_of("whole") == "capS"


def test_scorer_call_suffixes_and_missing_s_star():
    plan, _ = SC.plan_from(_filled_settings().data)
    args, out, missing = SC.scorer_call("r/b2/s58/x.csv", plan, plan.specs["s58"], 2)
    assert out == "r/b2/s58/x_scored_cap2.csv" and missing == []
    assert args[args.index("--age-cap-s") + 1] == "2"
    assert args[args.index("--traces") + 1] == "r/b2/s58/x_traces"
    assert args[args.index("--status-csv") + 1] == "r/b2/s58/x.csv"
    _, _, missing = SC.scorer_call("r/x.csv", plan, plan.specs["s58"], None)
    assert missing == ["pilot_outputs.s_star"]
    _, out, _ = SC.scorer_call("r/x.csv", plan, plan.specs["s513"], None)
    assert out == "r/x_scored_det.csv"
    _, out, _ = SC.scorer_call("r/x.csv", plan, None, None)
    assert out == "r/x_scored.csv"


def test_a_faster_variant_is_claimed_for_the_variant(tmp_path):
    plan = _plan()
    ref = _entry(tmp_path, "F", [100.0 + i for i in range(12)])
    fast = _entry(tmp_path, "FX", [80.0 + i for i in range(12)])
    same = _entry(tmp_path, "H1", [100.0 + i for i in range(12)])
    arms, cmps = SC.study_results("sx", [ref, fast, same], plan.specs["sx"], plan)
    SC.holm(cmps, plan)
    by = {c.variant: c for c in cmps}
    assert set(by) == {"FX", "H1"}                       # the reference is not compared
    assert by["FX"].n_pairs == 12 and by["FX"].mean_diff == pytest.approx(20.0)
    assert by["FX"].claim and by["FX"].favours == "FX"   # lower is better: FX wins
    assert not by["H1"].claim and by["H1"].favours == "no claim"
    assert by["FX"].p_holm >= by["FX"].p_value
    row = next(a for a in arms if a["variant"] == "FX")
    assert row["metric_mean"] == pytest.approx(85.5) and row["reach_rate_tau0.82"] == 1.0


def test_higher_is_better_flips_the_side(tmp_path):
    plan = _plan(better="higher")
    ref = _entry(tmp_path, "F", [100.0 + i for i in range(12)])
    low = _entry(tmp_path, "FX", [80.0 + i for i in range(12)])
    _, cmps = SC.study_results("sx", [ref, low], plan.specs["sx"], plan)
    SC.holm(cmps, plan)
    assert cmps[0].claim and cmps[0].favours == "F"


def test_complete_pairs_only_and_reach_rate(tmp_path):
    plan = _plan()
    ref = _entry(tmp_path, "F", [100.0, None, 102.0, 103.0, None, 105.0])
    var = _entry(tmp_path, "FX", [90.0, 91.0, None, 93.0, None, 95.0])
    arms, cmps = SC.study_results("sx", [ref, var], plan.specs["sx"], plan)
    assert cmps[0].n_pairs == 3                          # trials 0, 3 and 5
    f = next(a for a in arms if a["variant"] == "F")
    assert f["metric_n"] == 4 and f["reach_rate_tau0.82"] == pytest.approx(4 / 6, rel=1e-4)


def test_failed_trials_and_seed_mismatches_leave_the_comparison(tmp_path):
    plan = _plan()
    n = 8
    ref = _entry(tmp_path, "F", [100.0 + i for i in range(n)],
                 status=["ok"] * (n - 1) + ["timeout"])
    seeds = [f"s{i}" for i in range(n)]
    seeds[2] = "other"
    var = _entry(tmp_path, "FX", [90.0 + i for i in range(n)], seeds=seeds)
    arms, cmps = SC.study_results("sx", [ref, var], plan.specs["sx"], plan)
    c = cmps[0]
    assert c.pairing_errors == 1 and c.n_pairs == n - 2
    assert "pairing error" in c.note
    f = next(a for a in arms if a["variant"] == "F")
    assert (f["rows"], f["rows_not_ok"], f["scored"]) == (n, 1, n - 1)


def test_a_study_reads_only_the_trials_it_asked_for(tmp_path):
    plan = _plan()
    ref = _entry(tmp_path, "F", [100.0 + i for i in range(10)], trials=4)
    var = _entry(tmp_path, "FX", [90.0 + i for i in range(10)], trials=4)
    arms, cmps = SC.study_results("sx", [ref, var], plan.specs["sx"], plan)
    assert cmps[0].n_pairs == 4
    assert all(a["rows"] == 4 and a["scored"] == 4 for a in arms)


def test_too_few_pairs_is_noted_not_tested(tmp_path):
    plan = _plan()
    ref = _entry(tmp_path, "F", [100.0, None, None])
    var = _entry(tmp_path, "FX", [90.0, 91.0, None])
    _, cmps = SC.study_results("sx", [ref, var], plan.specs["sx"], plan)
    SC.holm(cmps, plan)
    assert cmps[0].p_value is None and cmps[0].p_holm is None and not cmps[0].claim
    assert "fewer than 2" in cmps[0].note


def test_holm_family_per_study_or_per_cell(tmp_path):
    def run(family):
        plan = _plan()
        plan.family = family
        cmps = []
        for cell in ("a", "b"):
            d = tmp_path / family / cell
            d.mkdir(parents=True)
            ref = _entry(d, "F", [100.0 + i for i in range(10)])
            var = _entry(d, "FX", [99.0 + i + (0.5 if i % 2 else -0.5) for i in range(10)])
            for e in (ref, var):
                e.cell = cell
            cmps += SC.study_results("sx", [ref, var], plan.specs["sx"], plan)[1]
        SC.holm(cmps, plan)
        return cmps
    study, cell = run("study"), run("cell")
    assert all(c.p_holm == pytest.approx(min(1.0, 2 * c.p_value)) for c in study)
    assert all(c.p_holm == pytest.approx(c.p_value) for c in cell)


def test_markdown_and_index_are_written(tmp_path):
    plan = _plan()
    ref = _entry(tmp_path, "F", [100.0 + i for i in range(6)])
    var = _entry(tmp_path, "FX", [80.0 + i for i in range(6)])
    arms, cmps = SC.study_results("sx", [ref, var], plan.specs["sx"], plan)
    SC.holm(cmps, plan)
    out = tmp_path / "scores"
    SC.write_study(out, "sx", plan.specs["sx"], plan, arms, cmps)
    path = SC.write_index(out, "batch1", [{"study": "sx", "metric": "m", "variants": 2,
                                           "comparisons": 1, "claims": 1, "scored": 12,
                                           "trials": 12}], [("t/x", "t/x.json", False)], plan)
    text = (out / "sx.md").read_text(encoding="utf-8")
    assert "## n6k1_knee" in text and "| FX |" in text
    assert (out / "sx_arms.csv").exists() and (out / "sx_comparisons.csv").exists()
    assert "not written yet" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The plan against the jobs the launcher builds
# --------------------------------------------------------------------------- #

def _filled_settings():
    """params.toml with every pilot output and decision set to a placeholder."""
    s, _ = L.load_settings(L.PARAMS)
    d = copy.deepcopy(s.data)
    per_n = lambda v: {str(n): v for n in (6, 12, 18, 24)}   # noqa: E731
    d["pilot_outputs"].update(knee_s=per_n(90.0), stress_s=per_n(45.0), s_star=per_n(2),
                              knee_meas_s={"6": 60.0}, stress_meas_s={"6": 30.0}, tau=0.7)
    d["rl"].update(keep_learned=True, gamma_star=0.9, repinned=True)
    d["rl"]["e3"]["gamma"] = 0.9
    d["rl"]["checkpoints"] = {k: f"ck/{k}.npz" for k in
                              ("main", "best", "g0", "hand", "dwell", "cov", "e3")}
    d["rl"]["checkpoints"]["best_tag"] = "g90"
    d["pilot_outputs"]["train_levels"] = {"none": [0.0, 0.0, 0.0, 1.0],
                                          "spread": [30.0, 0.5, 0.0, 1.0]}
    d["s515"].update(harsher_amp_db=11.0, lossier_n_pl=2.7)
    d["s511c"]["knee_s"] = {c: 1.0 for c in d["p511c"]["cells"]}
    return L.Settings(d)


@pytest.mark.parametrize("stage", L.SCORED)
def test_every_study_has_a_plan_whose_variants_exist(stage, tmp_path):
    s = _filled_settings()
    jobs = L.build(stage, s, None, str(tmp_path))
    runs, entries, tools, plan = L.score_plan(stage, s, jobs)
    assert entries, stage
    assert runs == []                                     # nothing has run in tmp_path
    for study, members in entries.items():
        spec = plan.specs.get(study)
        assert spec is not None and spec.metric, f"{study} has no [score.{study}]"
        variants = {e.variant for e in members}
        assert spec.reference in variants, (study, spec.reference, sorted(variants))
        for v, r in spec.versus.items():
            assert v in variants and (r is None or r in variants), (study, v, r)
        cells = {}
        for e in members:
            cells.setdefault(e.cell, set()).add(e.variant)
        assert sum(1 for vs in cells.values() if spec.reference in vs) >= 1
        keys = [(e.cell, e.variant) for e in members]
        assert len(keys) == len(set(keys)), f"{study}: a (cell, variant) twice"


def test_s51_compares_rules_within_route_and_pass_2(tmp_path):
    s = _filled_settings()
    jobs = [j for j in L.build("batch2", s, ["s51"], str(tmp_path))]
    _, entries, _, _ = L.score_plan("batch2", s, jobs)
    cells = {}
    for e in entries["s51"]:
        cells.setdefault(e.cell, set()).add(e.variant)
    assert cells["n6k1_knee_H1_budge"] == {"plain", "cutoff", "cutoff_fedprox", "fedbuff",
                                           "asynchfl"}
    assert "n6k1_knee_F_unbud" in cells and "n6k1_knee_F_budge" not in cells
