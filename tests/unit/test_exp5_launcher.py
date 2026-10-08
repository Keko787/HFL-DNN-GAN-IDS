"""Exp 5 launcher stages: the quick reproduction and the TTL sensitivity.

quick must fly batch 1's own jobs (same arguments, hence the same seeds) into
CSVs of its own; sens must read batch 1 for its factor-1 cell and scale the
session TTL for the others; `report quick` must pair trials by seed."""

from __future__ import annotations

import copy
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


#: A params file's [pilot_outputs] before its pilots, and the table after it.
SAMPLE_PARAMS = """[pilot_outputs]
# From stage ttl.
session_ttl_s = { "6" = 36.0, "12" = 34.0 }
# knee_s        = { "6" = 0.0, "12" = 0.0 }
# stress_s      = { "6" = 0.0, "12" = 0.0 }

# --------------------------------------------------------------------------
# Stage ttl.
# --------------------------------------------------------------------------
[ttl]
N = [6, 12]
"""


def test_write_pilot_outputs_edits_only_its_keys(tmp_path):
    p = tmp_path / "params.toml"
    original = SAMPLE_PARAMS
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
    assert data["ttl"] == {"N": [6, 12]}                      # the next table untouched


def test_report_apply_refuses_an_unfinished_pilot(tmp_path, capsys):
    s = _filled_settings()
    p = tmp_path / "params.toml"
    p.write_text(L.PARAMS.read_text(encoding="utf-8-sig"), encoding="utf-8")
    jobs = L.build("knee", s, None, str(tmp_path))
    assert L.cmd_report("knee", s, jobs, apply_to=p) == 1
    assert "nothing written" in capsys.readouterr().out
    assert p.read_text(encoding="utf-8") == L.PARAMS.read_text(encoding="utf-8-sig")


def test_pilot_tau_is_the_largest_every_n_reaches_in_the_share():
    tau, per_n = L.pilot_tau({6: [0.80] * 8 + [0.60, 0.65],      # 8 of 10 reach 0.80
                              12: [0.7] * 8 + [0.50, 0.55],      # exactly 0.70: kept
                              24: [0.7349] * 9 + [0.1]},         # floors to 0.73
                             share=0.8, step=0.01)
    assert per_n == {6: 0.8, 12: 0.7, 24: 0.73}
    assert tau == 0.7
    assert L.pilot_tau({6: []}, 0.8, 0.01) == (None, {})


def _cluster_trace(d: Path, trial: int, accuracies):
    t = d / f"N=6-x__H1__t{trial}__s{100 + trial}"
    t.mkdir(parents=True)
    lines = [json.dumps({"event": "model_eval", "cluster_round": r, "accuracy": acc})
             for r, acc in enumerate(accuracies)]
    lines.insert(1, json.dumps({"event": "round_closed", "cluster_round": 1}))
    (t / "cluster-exp4-cluster.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_trial_best_accuracy_skips_the_initial_model(tmp_path):
    d = tmp_path / "x_traces"
    _cluster_trace(d, 0, [0.95, 0.60, 0.72, 0.70])   # round 0 is the initial model
    _cluster_trace(d, 3, [0.36, 0.81])
    assert L.trial_best_accuracy(d) == {0: 0.72, 3: 0.81}
    assert L.trial_best_accuracy(tmp_path / "absent") == {}


def test_pack_and_unpack_round_trip(tmp_path, capsys):
    a = _ns(out_root=str(tmp_path))
    traces = tmp_path / "knee" / "knee" / "n6_1mb_b0090_traces"
    _cluster_trace(traces, 0, [0.36, 0.7])
    _cluster_trace(traces, 1, [0.36, 0.75])
    (tmp_path / "knee" / "knee" / "n6_1mb_b0090.csv").write_text("x\n", encoding="utf-8")
    assert L.cmd_pack("knee", a) == 0
    arch = tmp_path / "archives" / "knee_traces.tar.gz"
    sums = (tmp_path / "archives" / "SHA256SUMS").read_text(encoding="utf-8")
    assert arch.exists() and "knee_traces.tar.gz" in sums
    listing = json.loads((tmp_path / "archives" / "knee_traces.json").read_text("utf-8"))
    assert listing["folders"][0]["trials"] == 2
    import shutil
    shutil.rmtree(traces)
    assert L.cmd_unpack("knee", a) == 0
    assert L.trial_best_accuracy(traces) == {0: 0.7, 1: 0.75}


def test_unpack_refuses_a_damaged_archive(tmp_path):
    a = _ns(out_root=str(tmp_path))
    _cluster_trace(tmp_path / "knee" / "knee" / "c_traces", 0, [0.36, 0.7])
    L.cmd_pack("knee", a)
    arch = tmp_path / "archives" / "knee_traces.tar.gz"
    arch.write_bytes(arch.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="SHA-256"):
        L.cmd_unpack("knee", a)


def test_sstar_measures_on_the_campaigns_contact_channel(tmp_path):
    s = _filled_settings()
    for j in L.build("sstar", s, None, str(tmp_path)):
        physics = json.loads(j.args[j.args.index("--ferry-physics") + 1])
        assert physics == {"contact_regime": s.get("campaign.contact_regime")} == {
            "contact_regime": "jittery"}


def test_decided_values_are_set():
    s, _ = L.load_settings(L.PARAMS)
    assert s.get("rl.e3.gamma") == 0.99 and s.get("rl.e3.settings") == "chen"
    assert s.get("s51.fedprox_rho") == 0.01 and s.get("s56.n_trials") == 40


def test_p512_flies_h1_at_multiples_of_the_mission_cycle(tmp_path):
    s = _filled_settings()                                  # knee_s 90 s at every N
    jobs = [j for j in L.build("pilot3", s, None, str(tmp_path)) if j.study == "p512"]
    cycle = 90.0 + s.get("p512.turnaround_s")
    medians = [float(j.args[j.args.index("--train-time-s") + 1]) for j in jobs]
    assert medians == [round(f * cycle) for f in s.get("p512.cycle_factors")]
    assert all(j.args[j.args.index("--arms") + 1] == "H1" and not j.blocked for j in jobs)
    unset = L.Settings(copy.deepcopy(s.data))               # no knee yet: blocked, by name
    unset.data["pilot_outputs"].pop("knee_s")
    blocked = [j for j in L.build("pilot3", unset, None, str(tmp_path)) if j.study == "p512"]
    assert blocked and all("pilot_outputs.knee_s.6" in j.blocked for j in blocked)


def test_p512_rule_picks_the_level_nearest_the_bands_middle():
    levels = [(30.0, 0.02), (60.0, 0.12), (120.0, 0.25), (240.0, 0.6)]
    assert L.pick_spread(levels, [0.1, 0.3]) == 120.0           # 0.25 is nearer 0.2
    assert L.pick_spread([(30.0, 0.02), (60.0, None)], [0.1, 0.3]) is None


def test_batch3_waits_for_the_pilots_levels(tmp_path):
    unset = _filled_settings()
    unset.data["pilot_outputs"].pop("train_levels")         # before pilot3
    jobs = [j for j in L.build("batch3", unset, None, str(tmp_path)) if j.study == "s512"]
    assert jobs and all("pilot_outputs.train_levels" in j.blocked for j in jobs)


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


SIZED_ON = {"physical_cpus": 8, "logical_cpus": 16, "ram_gb": 94.7}
AUTO_KEYS = ("machine.max_jobs", "machine.max_device_processes", "machine.mem_budget_gb",
             "rl.max_jobs")


def _auto_settings():
    return L.Settings({"machine": {k.split(".")[1]: "auto" for k in AUTO_KEYS[:3]},
                       "rl": {"max_jobs": "auto"}})


def test_auto_limits_give_the_sized_on_hosts_values_there():
    s = _auto_settings()
    L.resolve_machine(s, SIZED_ON)
    assert [s.get(k) for k in AUTO_KEYS] == [6, 36, 75.0, 14]
    assert len(s.auto) == 4 and all("(" in note for note in s.auto)


def test_auto_limits_scale_with_the_host():
    s = _auto_settings()                                    # 8 P + 4 E cores, 64 GB
    L.resolve_machine(s, {"physical_cpus": 12, "logical_cpus": 20, "ram_gb": 63.8})
    assert [s.get(k) for k in AUTO_KEYS] == [7, 45, 51.0, 18]
    s = _auto_settings()                                    # no SMT: the cores bind
    L.resolve_machine(s, {"physical_cpus": 4, "logical_cpus": 4, "ram_gb": 16.0})
    assert [s.get(k) for k in AUTO_KEYS] == [1, 9, 12.0, 2]
    s = _auto_settings()                                    # never below one N = 6 trial
    L.resolve_machine(s, {"physical_cpus": 1, "logical_cpus": 1, "ram_gb": 4.0})
    assert [s.get(k) for k in AUTO_KEYS] == [1, 6, 3.0, 1]


def test_auto_leaves_numbers_and_follows_a_set_device_count():
    s = _auto_settings()
    s.data["machine"]["max_device_processes"] = 24
    s.data["rl"]["max_jobs"] = 3
    L.resolve_machine(s, SIZED_ON)
    assert [s.get(k) for k in AUTO_KEYS] == [4, 24, 75.0, 3]
    assert len(s.auto) == 2


def test_auto_memory_needs_the_ram():
    with pytest.raises(SystemExit, match="psutil"):
        L.resolve_machine(_auto_settings(), {"physical_cpus": 8, "logical_cpus": 16})


def test_params_limits_are_auto_and_load_as_numbers():
    text = L.PARAMS.read_text(encoding="utf-8-sig")
    raw = L.tomllib.loads(text)
    assert all(raw[k.split(".")[0]][k.split(".")[1]] == "auto" for k in AUTO_KEYS)
    s, _ = L.load_settings(L.PARAMS)
    assert all(isinstance(s.get(k), (int, float)) and s.get(k) > 0 for k in AUTO_KEYS)
    assert L.Builder(s).concurrency(6) >= 1


def test_pilot_host_note_names_another_host(tmp_path, monkeypatch):
    here = {"processor": "here-cpu", "physical_cpus": 12, "logical_cpus": 20, "ram_gb": 63.8,
            "platform": "Windows-11"}
    monkeypatch.setattr(L, "host_info", lambda: dict(here))
    monkeypatch.setattr(L, "REPO", tmp_path)
    assert L.pilot_host_note("out") is None                 # no TTL manifest yet
    launcher = tmp_path / "out" / "ttl" / "_launcher"
    launcher.mkdir(parents=True)
    (launcher / "manifest_20261005_123722.json").write_text(
        json.dumps({"environment": {"host": dict(here, ram_gb=63.7)}}), encoding="utf-8")
    assert L.pilot_host_note("out") is None                 # the same host
    (launcher / "manifest_20261006_090000.json").write_text(
        json.dumps({"environment": {"host": dict(SIZED_ON, processor="there-cpu")}}),
        encoding="utf-8")
    note = L.pilot_host_note("out")                         # the latest manifest's host
    assert note and "there-cpu" in note and "here-cpu" in note and "--out-root" in note


def test_smoke_trainings_get_past_the_learners_warmup(tmp_path):
    s = _filled_settings()
    jobs = [j for j in L.build("rl-e3", L.Settings(s.data), None, str(tmp_path))
            if j.kind == "rl-train"]
    L.smoke(jobs)
    assert jobs and all(j.args[j.args.index("--warmup-transitions") + 1]
                        == str(L.SMOKE_WARMUP) for j in jobs)
    assert all(int(j.args[j.args.index("--episodes") + 1]) == 30 for j in jobs)


def test_rl_report_shows_the_verdict_the_report_file_nests(tmp_path, capsys):
    """ferrysim report writes {epsilon_source, evaluation, verdict}; the launcher
    prints the verdict's outcome, its picks and the greedy_1 flag."""
    s = L.Settings(_filled_settings().data)
    jobs = L.build("rl-sweep", s, None, str(tmp_path))
    report = next(j for j in jobs if j.kind == "tool" and j.name.endswith("/report"))
    out = L.REPO / report.out
    out.parent.mkdir(parents=True, exist_ok=True)
    verdict = {"outcome": "rising", "replace_fx": False, "greedy_1_flag": True,
               "stack_check": {"best_gamma": 0.25}, "epsilon": 0.01, "curves": {}}
    out.write_text(json.dumps({"epsilon_source": "x", "evaluation": "y",
                               "verdict": verdict}), encoding="utf-8")
    L.cmd_report("rl-sweep", s, jobs)
    shown = capsys.readouterr().out
    assert '"outcome": "rising"' in shown and '"greedy_1_flag": true' in shown
    assert '"best_gamma": 0.25' in shown and "curves" not in shown


def test_smoke_flies_the_o1_oracle_on_two_episodes(tmp_path):
    s = _filled_settings()
    jobs = [j for j in L.build("batch2", L.Settings(s.data), None, str(tmp_path))
            if "experiments.analysis.o1_oracle" in j.args]
    assert jobs and all(j.args[j.args.index("--episodes") + 1] == "30" for j in jobs)
    L.smoke(jobs)
    assert all(j.args[j.args.index("--episodes") + 1] == "2" for j in jobs)


def test_p512_picks_its_level_by_the_share_after_the_first_mission(tmp_path, monkeypatch):
    """Every fit starts at the first takeoff, so the first mission's contacts find
    few updates ready at any level; the rule reads the later missions' share
    (decided 6 Oct 2026), here inside the band only at the shortest level."""
    from types import SimpleNamespace

    from experiments.analysis import traces_scorer as TS

    s = _filled_settings()
    jobs = [j for j in L.build("pilot3", s, None, str(tmp_path)) if j.study == "p512"]
    after = [0.22, 0.41, 0.73, 0.88]                    # the smoke run's, by level
    every = [0.375, 0.565, 0.81, 0.913]                 # all above the band
    by_trace = {}
    for j, a, e in zip(jobs, after, every):
        out = L.REPO / j.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("status\n", encoding="utf-8")
        traces = L.REPO / (j.out[:-4] + "_traces")
        traces.mkdir()
        by_trace[str(traces)] = SimpleNamespace(compute=SimpleNamespace(
            not_ready_share_after_first=a, not_ready_share=e))
    monkeypatch.setattr(TS, "load_status_csv", lambda path: None)
    monkeypatch.setattr(TS, "score_traces", lambda traces, **kw: [by_trace[str(traces)]])
    problems = []
    levels = L.p512_levels(s, jobs, problems)
    first = float(jobs[0].args[jobs[0].args.index("--train-time-s") + 1])
    assert problems == [] and levels is not None and f"spread = [{first:.1f}," in levels


# --------------------------------------------------------------------------- #
# Windows power throttling: every job opted out, the machine kept awake
# --------------------------------------------------------------------------- #

def _guard(trees, applied, refuse=()):
    awake = []
    it = iter(trees)

    def apply(pid):
        applied.append(pid)
        return pid not in refuse

    g = L.ThrottleGuard(True, apply=apply, tree=lambda: next(it),
                        awake=lambda on: awake.append(on) or True)
    return g, awake


def test_the_guard_opts_each_process_out_once_while_it_runs():
    applied = []
    me = L.os.getpid()
    g, awake = _guard([[me, 10], [me, 10, 11, 12], [me, 12, 10]], applied, refuse={11})
    g.start()                          # itself (watch), then the first tree
    assert applied == [me, 10] and awake == [True]
    g.watch(10)                        # a job already seen is not opted out again
    g.sweep()                          # 11 and 12 appear under a job
    assert applied == [me, 10, 11, 12]
    g.sweep()                          # 11 has gone: forgotten, never retried while gone
    assert applied == [me, 10, 11, 12]
    g.stop()
    g.stop()
    assert awake == [True, False]      # released once
    assert g.record() == {"unthrottle": True, "keep_awake": True, "sees_job_children": True,
                          "processes_unthrottled": 3, "refused": 1}


def test_a_reused_pid_is_opted_out_again():
    applied = []
    me = L.os.getpid()
    g, _ = _guard([[me, 20], [me], [me, 20]], applied)
    g.start()
    g.sweep()
    g.sweep()
    assert applied == [me, 20, 20]


def test_the_guard_off_does_nothing():
    calls = []
    g = L.ThrottleGuard(False, apply=calls.append, tree=lambda: [1, 2],
                        awake=calls.append).start()
    g.watch(5)
    g.sweep()
    g.stop()
    assert calls == [] and g.record()["unthrottle"] is False


def test_a_failing_tree_never_stops_the_stage():
    def tree():
        raise RuntimeError("psutil hiccup")

    applied = []
    g = L.ThrottleGuard(True, apply=lambda pid: applied.append(pid) or True, tree=tree,
                        awake=lambda on: True).start()
    g.sweep()
    assert applied == [L.os.getpid()]


def test_params_turn_the_guard_on():
    s, _ = L.load_settings(L.PARAMS)
    assert s.get("machine.unthrottle") is True


@pytest.mark.skipif(L.os.name != "nt", reason="Windows power throttling")
def test_unthrottle_takes_on_a_real_process():
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert L.unthrottle(child.pid) is True
        assert L.throttling_state(child.pid) == (L.POWER_THROTTLING_EXECUTION_SPEED, 0)
    finally:
        child.kill()
        child.wait()
    assert L.unthrottle(0) is False                  # not openable: refused, not raised
    assert L.throttling_state(0) is None


@pytest.mark.skipif(L.os.name != "nt", reason="Windows power throttling")
def test_a_stage_runs_its_jobs_unthrottled_and_records_it(tmp_path):
    """End to end: a tool job (and the process it starts) reads its own power
    throttling state; the manifest records the guard."""
    import json as _json

    s, _ = L.load_settings(L.PARAMS)
    out = tmp_path / "state.json"
    probe = (
        "import json, subprocess, sys, time\n"
        "sys.path.insert(0, r'%s')\n"
        "import importlib.util as u\n"
        "spec = u.spec_from_file_location('l', r'%s'); l = u.module_from_spec(spec)\n"
        "sys.modules['l'] = l; spec.loader.exec_module(l)\n"
        "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'])\n"
        "time.sleep(5)\n"
        "json.dump({'job': l.throttling_state(l.os.getpid()),\n"
        "           'child': l.throttling_state(kid.pid)}, open(r'%s', 'w'))\n"
        "kid.kill()\n"
    ) % (L.REPO, L.HERE / "launch.py", out)
    job = L.Job(stage="sens", study="t", name="t/probe", args=["-c", probe], out=str(out),
                kind="tool")
    rc = L._run_jobs("sens", s, "", [job], [job], {"commit": "x"}, {}, 1, str(tmp_path))
    assert rc == 0
    state = _json.loads(out.read_text())
    assert state == {"job": [1, 0], "child": [1, 0]}
    (manifest,) = (tmp_path / L.STAGE_DIR["sens"] / "_launcher").glob("manifest_*.json")
    power = _json.loads(manifest.read_text(encoding="utf-8"))["power"]
    assert power["unthrottle"] is True and power["keep_awake"] is True
    assert power["processes_unthrottled"] >= 3 and power["refused"] == 0


def test_a_shared_cell_reads_the_csv_batch_1_wrote_under_a_trials_lever(tmp_path):
    """Batch 1 flew 5.3's F cell inside 5.14's (40 trials beat 20). Under
    --trials 20 the two tie, and the dedupe would name 5.3's CSV, never
    written; a later cell that shares it must read 5.14's, the one with rows."""
    s = _filled_settings()
    b1 = L.Builder(L.Settings(copy.deepcopy(s.data)), str(tmp_path))
    b1.batch1()
    flown = L.dedupe(b1.jobs)
    cap = next(j for j in flown if j.name == "s514/n6k1_knee__capS")
    f53 = next(j for j in flown if j.name == "s53/n6k1_knee__F")
    assert L._key(cap) == L._key(f53) and f53.alias_of == cap.name
    out = L.REPO / cap.out                 # the root is tmp_path, so this is absolute
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("trial_index,status\n" + "".join(f"{i},ok\n" for i in range(40)),
                   encoding="utf-8")
    d = copy.deepcopy(s.data)
    L.set_everywhere(d, "n_trials", 20)
    jobs = L.build("batch2", L.Settings(d), ["s51"], str(tmp_path))
    (j,) = [j for j in jobs if j.name == "s51/n6k1_knee_F_unbud__cutoff"]
    assert j.alias_of == f"batch1:{cap.name}" and j.out == cap.out
