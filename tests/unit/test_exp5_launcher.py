"""Exp 5 launcher stages: the quick reproduction and the TTL sensitivity.

quick must fly batch 1's own jobs (same arguments, hence the same seeds) into
CSVs of its own; sens must read batch 1 for its factor-1 cell and scale the
session TTL for the others; `report quick` must pair trials by seed."""

from __future__ import annotations

import csv
import json
from pathlib import Path

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
