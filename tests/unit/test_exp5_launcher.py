"""Exp 5 launcher stages: the quick reproduction and the TTL sensitivity.

quick must fly batch 1's own jobs (same arguments, hence the same seeds) into
CSVs of its own; sens must read batch 1 for its factor-1 cell and scale the
session TTL for the others; `report quick` must pair trials by seed."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.unit.test_exp5_scoring import L, _filled_settings


def _ttl(job):
    return float(job.args[job.args.index("--session-ttl-s") + 1])


def test_quick_flies_batch_1s_jobs_into_its_own_csvs(tmp_path):
    s = _filled_settings()
    quick = L.build("quick", s, None, str(tmp_path))
    builder = L.Builder(L.Settings(s.data), str(tmp_path))
    builder.batch1()
    recorded = {L._key(j) for j in builder.jobs if j.kind == "runner"}
    assert quick
    for j in quick:
        assert j.kind == "runner" and j.alias_of is None and not j.blocked
        assert L._key(j) in recorded, j.name                  # same arguments: same seeds
        assert "/quick/s53/" in j.out.replace("\\", "/")
        assert j.trials == s.get("quick.n_trials")
    assert {j.args[j.args.index("--arms") + 1] for j in quick} == set(s.get("quick.arms"))
    assert len(quick) == (len(s.get("quick.K")) * len(s.get("quick.budgets"))
                          * (len(s.get("quick.arms")) + int(s.get("quick.d4_faithful"))))


def test_sens_reads_batch_1_at_factor_1_and_scales_the_ttl(tmp_path):
    s = _filled_settings()
    jobs = L.build("sens", s, None, str(tmp_path))
    base = float(s.get("pilot_outputs.session_ttl_s.6"))
    by_factor = {}
    for j in jobs:
        factor = j.name.split("_ttl", 1)[1].split("__", 1)[0]
        by_factor.setdefault(factor, []).append(j)
    assert set(by_factor) == {"0.75", "1", "1.5"}
    for j in by_factor["1"]:
        assert (j.alias_of or "").startswith("batch1:") and _ttl(j) == base
    for factor in ("0.75", "1.5"):
        for j in by_factor[factor]:
            assert j.alias_of is None and _ttl(j) == base * float(factor)
            assert "/sens/ttl/" in j.out.replace("\\", "/")


def test_campaign_runs_sens_after_batch1_and_never_quick():
    assert L.CAMPAIGN.index("sens") == L.CAMPAIGN.index("batch1") + 1
    assert "quick" not in L.CAMPAIGN and "quick" not in L.REUSES_BATCH1


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _rows(n, shift=0.0, seed_of=lambda i: f"s{i}"):
    return [{"cell_id": "c", "arm": "F", "trial_index": i, "seed": seed_of(i), "status": "ok",
             **{c: 0.5 + 0.01 * i + (shift if c == "final_accuracy" else 0.0)
                for c in L.REPRO_COLUMNS}} for i in range(n)]


def test_report_quick_pairs_trials_by_seed(tmp_path, capsys):
    s = _filled_settings()
    root = str(tmp_path)
    quick = [j for j in L.build("quick", s, None, root)
             if j.args[j.args.index("--arms") + 1] in ("F", "H1")]
    builder = L.Builder(L.Settings(s.data), root)
    builder.batch1()
    recorded = {L._key(j): j for j in L.dedupe(builder.jobs)
                if j.kind == "runner" and not j.blocked and j.alias_of is None}
    f, h1 = (next(j for j in quick if j.args[j.args.index("--arms") + 1] == a)
             for a in ("F", "H1"))
    # F: batch 1 recorded 20 trials; the reproduction flew 10, all identical.
    _write(L.REPO / recorded[L._key(f)].out, _rows(20))
    _write(L.REPO / f.out, _rows(10))
    # H1: same seeds, final accuracy off by 0.02 on every trial.
    _write(L.REPO / recorded[L._key(h1)].out, _rows(20))
    _write(L.REPO / h1.out, _rows(10, shift=0.02))
    assert L.report_quick(s, quick, recorded_root=root) == 0
    report = json.loads((Path(f.out).parent.parent / "reproduction.json").read_text("utf-8"))
    rf, rh = report["jobs"][f.name], report["jobs"][h1.name]
    assert (rf["pairs"], rf["identical"]) == (10, 10)
    assert (rh["pairs"], rh["identical"]) == (10, 0)
    assert abs(rh["columns"]["final_accuracy"]["max_abs_diff"] - 0.02) < 1e-12
    assert rh["columns"]["update_yield"]["max_abs_diff"] == 0.0
    assert "10 trials paired" in capsys.readouterr().out


def _ns(**kw):
    import argparse
    base = dict(trials=None, seed=None, missions=None, contact_regime=None, tau=None,
                mem_gb=None, devices=None, set=None, dataset=None, study=None, arms=None,
                smoke=False, out_root="results/exp5")
    base.update(kw)
    return argparse.Namespace(**base)


def test_levers_change_the_settings_and_are_recorded():
    s, _ = L.load_settings(L.PARAMS)
    L.apply_levers(s, _ns(trials=3, seed=7, missions=8, tau=[0.75, 0.8],
                          set=["s58.missions=[4,8,12]", "campaign.regime=clean",
                               "pilot_outputs.knee_s={ \"6\" = 90.0 }"]))
    assert s.get("s53.n_trials") == 3 and s.get("knee.n_trials") == 3
    assert s.get("campaign.base_seed") == 7 and s.get("campaign.n_missions") == 8
    assert s.get("score.tau") == [0.75, 0.8]
    assert s.get("s58.missions") == [4, 8, 12]
    assert s.get("campaign.regime") == "clean"                 # a bare word is a string
    assert s.get("pilot_outputs.knee_s.6") == 90.0
    assert len(s.overrides) == 1 + 3 + 3
    assert any("--trials 3" in o for o in s.overrides)


def test_set_refuses_an_unknown_table_and_a_missing_value():
    s, _ = L.load_settings(L.PARAMS)
    with pytest.raises(SystemExit):
        L.apply_levers(s, _ns(set=["quik.n_trials=5"]))
    with pytest.raises(SystemExit):
        L.apply_levers(s, _ns(set=["s53.n_trials"]))


def test_arms_keep_only_those_stack_trials(tmp_path):
    s = _filled_settings()
    jobs = L.jobs_for("batch1", s, _ns(arms=["F", "H1"], out_root=str(tmp_path)))
    assert jobs and all(j.kind == "runner" for j in jobs)
    assert {j.args[j.args.index("--arms") + 1] for j in jobs} == {"F", "H1"}


def test_groups_cover_the_campaign():
    assert L.GROUPS["all"] == L.CAMPAIGN
    grouped = [st for k, v in L.GROUPS.items() if k != "all" for st in v]
    assert sorted(grouped) == sorted(L.CAMPAIGN)


def test_write_pilot_outputs_edits_only_its_keys(tmp_path):
    p = tmp_path / "params.toml"
    original = L.PARAMS.read_text(encoding="utf-8-sig")
    p.write_text(original, encoding="utf-8")
    changes = L.write_pilot_outputs(p, {
        "knee_s": '{ "6" = 90.0, "12" = 120.0 }',              # commented out: replaced
        "session_ttl_s": '{ "6" = 30.0 }',                     # set: replaced
        "extra_s": '{ "6" = 1.0 }',                            # new: appended to the table
    })
    assert [old.lstrip().startswith("#") for old, _ in changes] == [True, False, False]
    import tomllib
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    assert data["pilot_outputs"]["knee_s"] == {"6": 90.0, "12": 120.0}
    assert data["pilot_outputs"]["session_ttl_s"] == {"6": 30.0}
    assert data["pilot_outputs"]["extra_s"] == {"6": 1.0}
    assert "stress_s" not in data["pilot_outputs"]            # still commented out
    before = [l for l in original.splitlines() if "knee_s " not in l and "session_ttl_s" not in l]
    after = [l for l in p.read_text(encoding="utf-8").splitlines()
             if "knee_s " not in l and "session_ttl_s" not in l and "extra_s" not in l]
    assert before == after                                    # every other line kept


def test_report_apply_refuses_an_unfinished_pilot(tmp_path, capsys):
    s = _filled_settings()
    p = tmp_path / "params.toml"
    p.write_text(L.PARAMS.read_text(encoding="utf-8-sig"), encoding="utf-8")
    jobs = L.build("knee", s, None, str(tmp_path))
    assert L.cmd_report("knee", s, jobs, apply_to=p) == 1
    assert "nothing written" in capsys.readouterr().out
    assert p.read_text(encoding="utf-8") == L.PARAMS.read_text(encoding="utf-8-sig")


def test_report_quick_never_pairs_different_seeds(tmp_path):
    s = _filled_settings()
    root = str(tmp_path)
    quick = [j for j in L.build("quick", s, None, root)
             if j.args[j.args.index("--arms") + 1] == "F"]
    builder = L.Builder(L.Settings(s.data), root)
    builder.batch1()
    recorded = {L._key(j): j for j in L.dedupe(builder.jobs)
                if j.kind == "runner" and not j.blocked and j.alias_of is None}
    _write(L.REPO / recorded[L._key(quick[0])].out, _rows(10))
    _write(L.REPO / quick[0].out, _rows(10, seed_of=lambda i: f"other{i}"))
    L.report_quick(s, quick, recorded_root=root)
    report = json.loads((Path(quick[0].out).parent.parent / "reproduction.json").read_text("utf-8"))
    assert report["jobs"][quick[0].name]["pairs"] == 0
