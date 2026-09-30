"""Phase 0 — scoring retained Exp 4 traces.

Three layers under test:

1. **The mule's new trace fields** (``pass_1_plan``, ``pass_1_outcomes`` on
   ``mission_completed``), which make the deadline-miss rate scorable.
2. **The consumer**, which now keeps timestamps and, for traces recorded
   before Freeze Amendment 5, places each ``backhaul_upload_lost`` in the
   mission whose time window contains it — so round closure stops counting
   dropped rounds as closed.
3. **The scorer**, end to end on a synthetic trial whose ages, closure and
   time to τ are worked out by hand below.

The same scorer, run on the recorded L1 cell (``results/exp4_matrix/C_traces``),
reproduces the reach counts in ``HERMES_Matrix_Results.md`` (τ = 0.82: 28/40
vs 19/40; τ = 0.75: 39/40 vs 30/40) and every unaffected CSV column.
"""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from experiments.analysis.traces_scorer import (
    age_profile,
    deadline_misses,
    main,
    merged_devices,
    parse_trial_dir,
    score_trial,
    score_traces,
    tau_reach,
    trial_provenance,
    trial_status,
    write_scores_csv,
)
from experiments.exp4.driver import PROVENANCE_COLUMNS
from experiments.exp4.events_consumer import observation_from_rows
from experiments.exp4.metrics import summarise_observation
from hermes.mission.aggregation_rules import AggregationSpec
from hermes.processes.mule import _pass_1_outcomes_payload, _pass_1_plan_payload
from hermes.scheduler.stages.s3_deadline import DeadlineLaw
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    MissionOutcome,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MuleID,
)

DEVICES = ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# A four-mission trial, in the shape of a trace recorded before Amendment 5
# --------------------------------------------------------------------------- #
#
#   mission  window       Pass-1 CLEAN   upload             merged
#   1        1000–1010    a, b           ingested @1005     a, b   (eval r1 @1006, acc 0.70)
#   2        1010–1020    —              (empty, no dock)   —
#   3        1020–1030    a              LOST @1025         —
#   4        1030–1040    c              ingested @1035     c      (eval r2 @1036, acc 0.85)
#
# Ages after each mission (a, b, c): (0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 0)
# Network AoU, uniform weights: 1/3, 4/3, 7/3, 2  -> mean 1.5, final 2.0
# Round closure (>= 1 update and upload not lost): missions 1 and 4 -> 0.5

def _missions():
    return [
        dict(rnd=1, start=1000.0, end=1010.0, clean=["a", "b"]),
        dict(rnd=2, start=1010.0, end=1020.0, clean=[]),
        dict(rnd=3, start=1020.0, end=1030.0, clean=["a"]),
        dict(rnd=4, start=1030.0, end=1040.0, clean=["c"]),
    ]


def _mule_rows(missions, mule="m1"):
    rows = [{"ts": 999.0, "event": "dock_bootstrapped", "role": "mule", "id": mule}]
    for i, m in enumerate(missions):
        rows.append({"ts": m["start"], "event": "mission_started", "role": "mule",
                     "id": mule, "mission_index": i})
        if not m["clean"]:
            rows.append({"ts": m["end"], "event": "mission_empty", "role": "mule",
                         "id": mule, "mission_round": m["rnd"]})
        row = {
            "ts": m["end"], "event": "mission_completed", "role": "mule", "id": mule,
            "mission_round": m["rnd"], "pass_1_contacts": 1,
            "pass_2_contacts": 1 if m["clean"] else 0,
            "pass_1_updates": len(m["clean"]) if m["clean"] else None,
            "pass_1_scheduled": 3,
            "pass_1_clean_devices": m["clean"] or None,
        }
        for field in ("pass_1_plan", "pass_1_outcomes"):
            if field in m:
                row[field] = m[field]
        rows.append(row)
    return rows


def _cluster_rows():
    def ev(ts, event, **kw):
        return {"ts": ts, "event": event, "role": "cluster", "id": "c1", **kw}

    return [
        ev(998.0, "cluster_ready"),
        ev(999.0, "model_eval", cluster_round=0, accuracy=0.36, auc=0.3, loss=0.7, n_test=100),
        ev(1005.0, "up_bundle_ingested", mule_id="m1", mission_round=None),
        ev(1005.0, "cluster_round_closed", cluster_round=1),
        ev(1006.0, "model_eval", cluster_round=1, accuracy=0.70, auc=0.7, loss=0.6, n_test=100),
        ev(1025.0, "backhaul_upload_lost", mule_id="m1", mission_round=None),
        ev(1035.0, "up_bundle_ingested", mule_id="m1", mission_round=None),
        ev(1035.0, "cluster_round_closed", cluster_round=2),
        ev(1036.0, "model_eval", cluster_round=2, accuracy=0.85, auc=0.9, loss=0.4, n_test=100),
    ]


def _device_rows():
    return [{"ts": 998.0, "event": "device_ready", "role": "device", "id": d} for d in DEVICES]


def _obs(missions=None, cluster_rows=None):
    return observation_from_rows(
        cluster_rows=_cluster_rows() if cluster_rows is None else cluster_rows,
        mule_rows=_mule_rows(_missions() if missions is None else missions),
        device_rows=_device_rows(),
        n_devices=len(DEVICES),
    )


# --------------------------------------------------------------------------- #
# 1. The mule's trace fields
# --------------------------------------------------------------------------- #

def test_the_plan_payload_records_each_contacts_devices_and_deadline():
    queue = [
        ContactWaypoint(position=(1.0, 0.0, 0.0), devices=(DeviceID("a"), DeviceID("b")),
                        bucket=Bucket.NEW, deadline_ts=1005.0),
        ContactWaypoint(position=(9.0, 0.0, 0.0), devices=(DeviceID("c"),),
                        bucket=Bucket.NEW, deadline_ts=1008.0),
    ]
    assert _pass_1_plan_payload(queue) == [
        {"devices": ["a", "b"], "deadline_ts": 1005.0},
        {"devices": ["c"], "deadline_ts": 1008.0},
    ]


def test_the_outcomes_payload_records_every_session():
    report = MissionRoundCloseReport(
        mule_id=MuleID("m1"), mission_round=1, started_at=0.0, finished_at=1.0,
        lines=[
            MissionRoundCloseLine(device_id=DeviceID("a"), outcome=MissionOutcome.CLEAN,
                                  contact_ts=1004.0, basis_version=2, age=1),
            MissionRoundCloseLine(device_id=DeviceID("b"), outcome=MissionOutcome.TIMEOUT,
                                  contact_ts=1006.0),
        ],
    )
    assert _pass_1_outcomes_payload(SimpleNamespace(report=report, empty=False)) == [
        {"device": "a", "outcome": "clean", "contact_ts": 1004.0,
         "basis_version": 2, "age": 1},
        {"device": "b", "outcome": "timeout", "contact_ts": 1006.0,
         "basis_version": None, "age": None},
    ]


def test_an_empty_mission_records_no_sessions_rather_than_no_field():
    assert _pass_1_outcomes_payload(SimpleNamespace(report=None, empty=True)) == []
    assert _pass_1_outcomes_payload(SimpleNamespace(report=None, empty=False)) is None


# --------------------------------------------------------------------------- #
# 2. The consumer
# --------------------------------------------------------------------------- #

def test_old_traces_get_their_lost_rounds_back_from_timestamps():
    obs = _obs()
    assert obs.backhaul_lost_rounds == {3}
    closure = summarise_observation(
        obs, n_devices=3, rf_range_m=60.0, n_missions_target=4,
    ).round_close_rate_kmin1
    assert closure == pytest.approx(0.5)      # the recorded CSVs said 0.75


def test_a_recorded_round_is_used_directly():
    cluster = [r for r in _cluster_rows() if r["event"] != "backhaul_upload_lost"]
    cluster.append({"ts": 1015.0, "event": "backhaul_upload_lost", "role": "cluster",
                    "id": "c1", "mule_id": "m1", "mission_round": 2})
    assert _obs(cluster_rows=cluster).backhaul_lost_rounds == {2}


def test_rows_without_timestamps_are_left_as_before():
    """Hand-built rows carry no ``ts``: nothing can be placed, nothing changes."""
    cluster = [{k: v for k, v in r.items() if k != "ts"} for r in _cluster_rows()]
    missions = [{k: v for k, v in r.items() if k != "ts"} for r in _mule_rows(_missions())]
    obs = observation_from_rows(
        cluster_rows=cluster, mule_rows=missions, device_rows=_device_rows(), n_devices=3,
    )
    assert obs.backhaul_lost_rounds == set()
    assert obs.backhaul_losses == 1


def test_missions_keep_their_time_window_and_mule():
    m1 = _obs().missions[0]
    assert (m1.mule_id, m1.started_ts, m1.completed_ts) == ("m1", 1000.0, 1010.0)
    assert m1.contains(1005.0) and not m1.contains(1011.0)
    assert m1.pass_1_deadlines is None and m1.pass_1_outcomes is None


# --------------------------------------------------------------------------- #
# 3. The scorer's parts
# --------------------------------------------------------------------------- #

def test_ages_and_network_aou_match_the_hand_worked_trial():
    ages = age_profile(_obs(), DEVICES)
    assert ages.n_missions == 4
    assert ages.network_aou_mean == pytest.approx(1.5)
    assert ages.network_aou_final == pytest.approx(2.0)
    assert ages.age_max == 3
    assert ages.age_p95 == pytest.approx(3.0)
    assert ages.merged_updates == {"a": 1, "b": 1, "c": 1}   # a's mission-3 update was lost
    assert ages.jain_merged == pytest.approx(1.0)


def test_weights_tilt_network_aou_toward_the_weighted_devices():
    # All weight on c: its ages are 1, 2, 3, 0.
    ages = age_profile(_obs(), DEVICES, weights={"a": 0.0, "b": 0.0, "c": 5.0})
    assert ages.network_aou_mean == pytest.approx(1.5)
    assert ages.network_aou_final == pytest.approx(0.0)
    with pytest.raises(ValueError, match="missing"):
        age_profile(_obs(), DEVICES, weights={"a": 1.0})


def test_time_to_tau_in_missions_rounds_and_seconds():
    obs = _obs()
    late = tau_reach(obs, 0.82)
    assert (late.reached, late.mission, late.cluster_round, late.wall_s) == (True, 4, 2, 36.0)
    early = tau_reach(obs, 0.65)
    assert (early.mission, early.cluster_round, early.wall_s) == (1, 1, 6.0)
    assert not tau_reach(obs, 0.9).reached


def test_the_seed_model_never_counts_as_reaching_tau():
    assert tau_reach(_obs(), 0.3).cluster_round == 1     # round 0 scored 0.36


def test_an_evaluation_just_after_its_mission_belongs_to_that_mission():
    cluster = _cluster_rows()
    cluster[-1] = dict(cluster[-1], ts=1040.5)            # after mission 4 closed
    assert tau_reach(_obs(cluster_rows=cluster), 0.82).mission == 4


def test_deadline_misses_count_late_failed_aborted_and_empty():
    missions = [
        dict(rnd=1, start=1000.0, end=1010.0, clean=["a", "b"],
             pass_1_plan=[{"devices": ["a"], "deadline_ts": 1005.0},
                          {"devices": ["b", "c", "d"], "deadline_ts": 1008.0}],
             pass_1_outcomes=[{"device": "a", "outcome": "clean", "contact_ts": 1004.0},
                              {"device": "b", "outcome": "clean", "contact_ts": 1009.0},
                              {"device": "c", "outcome": "timeout", "contact_ts": 1007.0}]),
        # d was aborted: planned, no session. The empty mission misses everyone.
        dict(rnd=2, start=1010.0, end=1020.0, clean=[],
             pass_1_plan=[{"devices": ["a"], "deadline_ts": 1015.0}],
             pass_1_outcomes=[]),
    ]
    misses = deadline_misses(_obs(missions=missions))
    assert (misses.missions_with_plan, misses.admitted, misses.missed) == (2, 5, 4)
    assert misses.rate == pytest.approx(0.8)


def test_traces_without_the_plan_have_no_deadline_rate():
    assert deadline_misses(_obs()).rate is None


# --------------------------------------------------------------------------- #
# 4. The scorer end to end, on a trial directory
# --------------------------------------------------------------------------- #

TRIAL = "N=3-regime=jittery-rrf=60.0__H1__t7__s42"


def _write_trial(root, name=TRIAL, *, mules=("m1",)):
    d = root / name
    d.mkdir(parents=True)
    rows = _cluster_rows()
    (d / "cluster-c1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for mule in mules:
        (d / f"mule-{mule}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in _mule_rows(_missions(), mule=mule)) + "\n"
        )
        (d / f"mule-{mule}.json").write_text(json.dumps({"rf_range_m": 60.0, "n_missions": 4}))
    for dev in DEVICES:
        (d / f"device-{dev}.jsonl").write_text(
            json.dumps({"ts": 998.0, "event": "device_ready", "role": "device", "id": dev}) + "\n"
        )
    (d / "cluster.json").write_text(json.dumps(
        {"seed_devices": [{"device_id": dev, "position": [0, 0, 0]} for dev in DEVICES]}
    ))
    return d


def test_trial_directory_names_parse():
    key = parse_trial_dir(TRIAL)
    assert (key.cell_id, key.arm, key.trial_index, key.seed) == (
        "N=3-regime=jittery-rrf=60.0", "H1", 7, 42,
    )
    with pytest.raises(ValueError, match="not a trace directory"):
        parse_trial_dir("notes")


def test_score_trial_end_to_end(tmp_path):
    score = score_trial(_write_trial(tmp_path), taus=(0.82, 0.9))
    assert score.key.arm == "H1" and score.n_devices == 3
    assert score.summary.round_close_rate_kmin1 == pytest.approx(0.5)
    assert (score.missions_empty, score.backhaul_lost_missions) == (1, 1)
    assert score.ages.network_aou_mean == pytest.approx(1.5)
    assert [r.reached for r in score.tau] == [True, False]
    assert score.deadlines.rate is None

    row = score.to_row()
    assert row["reached_tau0.82"] == 1 and row["missions_to_tau0.82"] == 4
    assert row["reached_tau0.9"] == 0 and row["missions_to_tau0.9"] == ""
    assert row["deadline_miss_rate"] == ""


def test_score_traces_skips_other_directories_and_filters_arms(tmp_path):
    _write_trial(tmp_path)
    _write_trial(tmp_path, "N=3-regime=jittery-rrf=60.0__D1__t7__s42")
    (tmp_path / "notes").mkdir()
    assert [s.key.arm for s in score_traces(tmp_path)] == ["D1", "H1"]
    assert [s.key.arm for s in score_traces(tmp_path, arms=["H1"])] == ["H1"]


def test_multi_mule_traces_are_scored_per_mule(tmp_path):
    """Phase 2 lifted the single-mule refusal. Both mules fly rounds 1–4 in
    the same windows; only m1's round-3 upload was lost, so m2's round 3
    still closes (tests/unit/test_multi_mule_scoring.py has the details)."""
    score = score_trial(_write_trial(tmp_path, mules=("m1", "m2")))
    assert score.n_mules == 2 and score.to_row()["n_mules"] == 2
    assert score.backhaul_lost_missions == 1
    # m1 closes 1 and 4 (2 is empty, 3 lost); m2 closes 1, 3 and 4.
    assert score.summary.round_close_rate_kmin1 == pytest.approx(5 / 8)
    single = score_trial(_write_trial(tmp_path / "one"))
    assert single.n_mules == 1 and single.to_row()["n_mules"] == 1


def test_csv_and_cli(tmp_path):
    root = tmp_path / "traces"
    _write_trial(root)
    out = tmp_path / "scored.csv"
    write_scores_csv(score_traces(root), out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["network_aou_mean"] == "1.5"

    cli_out = tmp_path / "cli.csv"
    assert main(["--traces", str(root), "--tau", "0.82", "--csv", str(cli_out)]) == 0
    assert cli_out.exists()


# --------------------------------------------------------------------------- #
# 5. FeRRy audit fixes: merged vs collected (#3), the merge ledger (#4),
#    degenerate trials (#10), status (#11), provenance (#12), per-device
#    deadlines (#13)
# --------------------------------------------------------------------------- #

def _obs_with(missions, extra=None, cluster_rows=()):
    """``_obs`` with extra ``mission_completed`` fields, keyed by mission round."""
    rows = _mule_rows(missions)
    for r in rows:
        if r["event"] == "mission_completed":
            r.update((extra or {}).get(r["mission_round"], {}))
    return observation_from_rows(
        cluster_rows=list(cluster_rows), mule_rows=rows,
        device_rows=_device_rows(), n_devices=len(DEVICES),
    )


def _cev(ts, event, **kw):
    return {"ts": ts, "event": event, "role": "cluster", "id": "c1", **kw}


def _summary(obs, n_missions=4):
    return summarise_observation(obs, n_devices=3, rf_range_m=60.0, n_missions_target=n_missions)


# ---- #3: an update the mule's merge excluded never reached the model ---- #

def test_an_update_the_merge_excluded_is_never_credited():
    missions = [dict(rnd=1, start=1000.0, end=1010.0, clean=["a", "b"])]
    obs = _obs_with(missions, {1: {"pass_1_merged_devices": ["a"], "pass_1_merged_updates": 1}})
    mission = obs.missions[0]
    assert mission.pass_1_clean_devices == ("a", "b")
    assert (mission.pass_1_merged_devices, mission.pass_1_merged_updates) == (("a",), 1)
    assert merged_devices(obs, mission) == ("a",)
    assert age_profile(obs, DEVICES).merged_updates == {"a": 1, "b": 0, "c": 0}

    s = _summary(obs, n_missions=1)
    assert s.update_yield == pytest.approx(1.0)              # merged, not the 2 collected
    assert s.round_close_rate_kmin2 == pytest.approx(0.0)
    assert s.mission_completion_rate == pytest.approx(2 / 3)  # b still finished its session


def test_traces_without_the_merged_fields_credit_every_clean_update():
    obs = _obs()
    assert obs.missions[0].pass_1_merged_devices is None
    assert merged_devices(obs, obs.missions[0]) == ("a", "b")


def test_an_all_excluded_mission_credits_nobody_but_keeps_its_on_time_sessions():
    missions = [
        dict(rnd=1, start=1000.0, end=1010.0, clean=["a"]),
        # Reported as mission_empty: b's update was collected on time, then
        # excluded past its cutoff.
        dict(rnd=2, start=1010.0, end=1020.0, clean=[],
             pass_1_plan=[{"devices": ["b"], "deadline_ts": 1015.0}],
             pass_1_outcomes=[{"device": "b", "outcome": "clean", "contact_ts": 1014.0}]),
    ]
    obs = _obs_with(missions, {
        1: {"pass_1_merged_devices": ["a"], "pass_1_merged_updates": 1},
        2: {"pass_1_merged_devices": [], "pass_1_merged_updates": 0},
    })
    assert obs.missions_empty == 1 and obs.missions[1].pass_1_updates is None
    assert merged_devices(obs, obs.missions[1]) == ()
    assert age_profile(obs, DEVICES).merged_updates == {"a": 1, "b": 0, "c": 0}
    assert _summary(obs, n_missions=2).round_close_rate_kmin1 == pytest.approx(0.5)
    misses = deadline_misses(obs)
    assert (misses.admitted, misses.missed) == (1, 0)          # b was on time


# ---- #4: FedBuff deferral and expiry ---- #
#
#   mission  Pass-1 CLEAN  cluster
#   1        a             deferred (buffer 1/2)
#   2        b             applied: flushes 1 and 2
#   3        c             deferred, still buffered when the trial ends
#
# Ages (a, b, c): (1, 1, 1), (0, 0, 2), (1, 1, 3) -> NAoU 1, 2/3, 5/3; mean 10/9

FEDBUFF_MISSIONS = [
    dict(rnd=1, start=1000.0, end=1010.0, clean=["a"]),
    dict(rnd=2, start=1010.0, end=1020.0, clean=["b"]),
    dict(rnd=3, start=1020.0, end=1030.0, clean=["c"]),
]


def _fedbuff_cluster(recorded=True):
    rows = [
        _cev(1005.0, "up_bundle_ingested", mule_id="m1", mission_round=1),
        _cev(1005.0, "cluster_merge_deferred", mule_id="m1", mission_round=1,
             rule="agg:fedbuff", applied=False, buffered=1, k=2, partials=[["m1", 1]]),
        _cev(1015.0, "up_bundle_ingested", mule_id="m1", mission_round=2),
        _cev(1015.0, "cluster_merge", mission_round=2, rule="agg:fedbuff", applied=True,
             buffered=2, k=2, partials=[["m1", 1], ["m1", 2]]),
        _cev(1015.0, "cluster_round_closed", cluster_round=1),
        _cev(1025.0, "up_bundle_ingested", mule_id="m1", mission_round=3),
        _cev(1025.0, "cluster_merge_deferred", mule_id="m1", mission_round=3,
             rule="agg:fedbuff", applied=False, buffered=1, k=2, partials=[["m1", 3]]),
    ]
    if not recorded:
        # Phase 1 traces: no mission round and no partials on the merge events.
        for r in rows:
            if r["event"].startswith("cluster_merge"):
                del r["mission_round"], r["partials"]
    return rows


@pytest.mark.parametrize("recorded", [True, False], ids=["recorded", "placed-by-time"])
def test_a_fedbuff_deferral_is_credited_at_the_mission_that_flushed_it(recorded):
    obs = _obs_with(FEDBUFF_MISSIONS, cluster_rows=_fedbuff_cluster(recorded))
    assert obs.deferred_rounds == {1, 3}
    assert obs.flush_of == {1: 2}                 # 3 was never flushed
    assert obs.expired_rounds == set()
    m1, m2, m3 = obs.missions
    assert merged_devices(obs, m1) == ()
    assert merged_devices(obs, m2) == ("b", "a")
    assert merged_devices(obs, m3) == ()

    ages = age_profile(obs, DEVICES)
    assert ages.merged_updates == {"a": 1, "b": 1, "c": 0}
    assert ages.network_aou_mean == pytest.approx(10 / 9)
    assert ages.network_aou_final == pytest.approx(5 / 3)
    assert ages.age_max == 3
    # Only the flushing mission's round closes.
    assert _summary(obs, n_missions=3).round_close_rate_kmin1 == pytest.approx(1 / 3)


def test_without_partials_a_merge_flushes_every_deferral_since_the_last():
    missions = FEDBUFF_MISSIONS
    cluster = [
        _cev(1005.0, "cluster_merge_deferred", mule_id="m1", applied=False),
        _cev(1015.0, "cluster_merge_deferred", mule_id="m1", applied=False),
        _cev(1025.0, "cluster_merge", rule="agg:fedbuff", applied=True),
    ]
    obs = _obs_with(missions, cluster_rows=cluster)
    assert obs.deferred_rounds == {1, 2} and obs.flush_of == {1: 3, 2: 3}
    assert sorted(merged_devices(obs, obs.missions[2])) == ["a", "b", "c"]
    assert age_profile(obs, DEVICES).network_aou_final == pytest.approx(0.0)


@pytest.mark.parametrize("recorded", [True, False], ids=["recorded", "placed-by-time"])
def test_an_expired_merge_credits_nobody_and_does_not_close(recorded):
    missions = FEDBUFF_MISSIONS[:2]
    expired = _cev(1015.0, "cluster_merge_expired", mule_id="m1", mission_round=2)
    if not recorded:
        del expired["mission_round"]
    cluster = [
        _cev(1005.0, "cluster_merge", mission_round=1, rule="agg:cutoff", applied=True),
        _cev(1005.0, "cluster_round_closed", cluster_round=1),
        expired,
    ]
    obs = _obs_with(missions, cluster_rows=cluster)
    assert obs.expired_rounds == {2} and obs.deferred_rounds == set()
    assert merged_devices(obs, obs.missions[1]) == ()
    assert age_profile(obs, DEVICES).merged_updates == {"a": 1, "b": 0, "c": 0}
    assert _summary(obs, n_missions=2).round_close_rate_kmin1 == pytest.approx(0.5)


def test_plain_traces_have_an_empty_merge_ledger():
    obs = _obs()
    assert (obs.deferred_rounds, obs.expired_rounds, obs.flush_of) == (set(), set(), {})


# ---- #10: degenerate trials report blanks, not flattering numbers ---- #

def test_a_trial_that_merged_nothing_has_no_jain_index():
    missions = [dict(rnd=1, start=1000.0, end=1010.0, clean=[]),
                dict(rnd=2, start=1010.0, end=1020.0, clean=[])]
    ages = age_profile(_obs(missions=missions, cluster_rows=[]), DEVICES)
    assert ages.merged_total == 0 and ages.jain_merged is None
    assert ages.network_aou_final == pytest.approx(2.0)     # the ages still exist


def test_a_trial_without_missions_has_no_ages(tmp_path):
    ages = age_profile(_obs(missions=[], cluster_rows=[]), DEVICES)
    assert ages.n_missions == 0
    assert (ages.network_aou_mean, ages.network_aou_final, ages.age_max,
            ages.age_p95, ages.jain_merged) == (None, None, None, None, None)

    root = tmp_path / "traces"
    d = _write_trial(root)
    (d / "mule-m1.jsonl").write_text(json.dumps(_mule_rows([])[0]) + "\n")
    row = score_trial(d).to_row()
    assert (row["network_aou_mean"], row["network_aou_final"], row["age_max"],
            row["age_p95"], row["jain_merged"], row["merged_total"]) == ("", "", "", "", "", 0)
    # The summary's means skip the blanks rather than failing on them.
    _write_trial(root, "N=3-regime=jittery-rrf=60.0__H1__t8__s43")
    assert main(["--traces", str(root)]) == 0


# ---- #13: each member is held to its own Deadline(j) ---- #

def _deadline_mission(rnd, start, *, per_device):
    contact = {"devices": ["a", "b"], "deadline_ts": start + 5.0}   # a's, the tightest
    if per_device:
        contact["device_deadlines"] = {"a": start + 5.0, "b": start + 8.0}
    return dict(
        rnd=rnd, start=start, end=start + 10.0, clean=["a", "b"],
        pass_1_plan=[contact],
        # b finishes after the contact's deadline but before its own.
        pass_1_outcomes=[{"device": "a", "outcome": "clean", "contact_ts": start + 4.0},
                         {"device": "b", "outcome": "clean", "contact_ts": start + 7.0}],
    )


def test_a_later_member_on_time_under_its_own_deadline_is_not_a_miss():
    obs = _obs(missions=[_deadline_mission(1, 1000.0, per_device=True)])
    mission = obs.missions[0]
    assert mission.pass_1_deadlines == (("a", 1005.0), ("b", 1008.0))
    assert mission.deadline_basis == "device"
    misses = deadline_misses(obs)
    assert (misses.missed, misses.basis) == (0, "device")


def test_legacy_plans_fall_back_to_the_contact_deadline():
    obs = _obs(missions=[_deadline_mission(1, 1000.0, per_device=False)])
    assert obs.missions[0].pass_1_deadlines == (("a", 1005.0), ("b", 1005.0))
    misses = deadline_misses(obs)
    assert (misses.missed, misses.basis) == (1, "contact")   # b, late only by a's deadline


def test_the_basis_is_mixed_across_missions_and_reaches_the_row(tmp_path):
    obs = _obs(missions=[_deadline_mission(1, 1000.0, per_device=True),
                         _deadline_mission(2, 1010.0, per_device=False)])
    assert deadline_misses(obs).basis == "mixed"
    assert deadline_misses(_obs()).basis is None                 # no plan at all
    row = score_trial(_write_trial(tmp_path)).to_row()
    assert row["deadline_basis"] == ""


# ---- #11: trial status ---- #

TRIAL_8 = "N=3-regime=jittery-rrf=60.0__H1__t8__s43"


def _mark(trial_dir, status, error="", **timing):
    (trial_dir / "trial_status.json").write_text(
        json.dumps({"status": status, "error": error, "n_missions_target": 4, **timing})
    )


def _status_csv(path, rows):
    """A trial CSV as the runner writes it; the cell id keeps its ``|``."""
    lines = ["cell_id,arm,trial_index,seed,status,error"]
    lines += [f"N=3|regime=jittery|rrf=60.0,{arm},{t},{s},{status}," for arm, t, s, status in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_failed_trial_is_left_out_unless_asked_for(tmp_path):
    _write_trial(tmp_path)
    _mark(_write_trial(tmp_path, TRIAL_8), "no_eval", "no model_evaluation events")
    assert [s.key.trial_index for s in score_traces(tmp_path)] == [7]
    both = score_traces(tmp_path, include_failed=True)
    assert [(s.key.trial_index, s.status) for s in both] == [(7, "ok"), (8, "no_eval")]
    assert both[1].to_row()["status"] == "no_eval"
    assert both[0].to_row()["status"] == "ok"


def test_legacy_traces_take_their_status_from_the_trial_csv(tmp_path):
    root = tmp_path / "traces"
    _write_trial(root)
    _write_trial(root, TRIAL_8)
    trial_csv = _status_csv(tmp_path / "trials.csv", [("H1", 7, 42, "ok"), ("H1", 8, 43, "error")])
    assert [s.key.trial_index for s in score_traces(root, status_csv=trial_csv)] == [7]
    assert [s.key.trial_index for s in score_traces(root)] == [7, 8]      # neither: ok
    status = trial_status(root / TRIAL_8, trial_csv)
    assert (status.status, status.source) == ("error", "csv")
    assert trial_status(root / TRIAL_8).source == "default"


def test_a_failure_in_the_trial_csv_overrides_an_ok_marker(tmp_path, capsys):
    """The runner writes the row after the marker, relabelling a late trial
    ``timeout``: the row's failure is the final word."""
    root = tmp_path / "traces"
    _mark(_write_trial(root), "ok")
    trial_csv = _status_csv(tmp_path / "trials.csv", [("H1", 7, 42, "timeout")])
    status = trial_status(root / TRIAL, trial_csv)
    assert (status.status, status.source) == ("timeout", "csv")
    assert (status.marker_status, status.csv_status) == ("ok", "timeout")
    assert score_traces(root, status_csv=trial_csv) == []
    assert main(["--traces", str(root), "--status-csv", str(trial_csv)]) == 1
    out = capsys.readouterr().out
    assert "trial CSV says 'timeout'; using the trial CSV" in out
    assert "excluded (status not ok): H1 1" in out


def test_an_ok_in_the_trial_csv_never_clears_a_marked_failure(tmp_path):
    """The runner never turns a failed trial ok, so such a row is another run's."""
    root = tmp_path / "traces"
    _mark(_write_trial(root), "no_eval")
    trial_csv = _status_csv(tmp_path / "trials.csv", [("H1", 7, 42, "ok")])
    status = trial_status(root / TRIAL, trial_csv)
    assert (status.status, status.source) == ("no_eval", "marker")
    _mark(root / TRIAL, "ok")
    assert trial_status(root / TRIAL, trial_csv).source == "marker"      # they agree


def test_without_the_csv_a_late_ok_marker_is_a_soft_timeout(tmp_path):
    """The runner's default soft cap is the trial budget the marker records."""
    root = tmp_path / "traces"
    d = _write_trial(root)
    _mark(d, "ok", run_s=121.5, trial_budget_s=120.0)
    status = trial_status(d)
    assert (status.status, status.source, status.marker_status) == ("timeout", "soft_cap", "ok")
    assert score_traces(root) == []
    assert [s.status for s in score_traces(root, include_failed=True)] == ["timeout"]
    # A trial CSV row, when given, is the runner's own verdict — e.g. under an
    # explicit --timeout-s above the budget.
    ok_csv = _status_csv(tmp_path / "trials.csv", [("H1", 7, 42, "ok")])
    assert trial_status(d, ok_csv).status == "ok"
    # In time, or a marker that records no timing: the marker's ok stands.
    _mark(d, "ok", run_s=119.0, trial_budget_s=120.0)
    assert trial_status(d).status == "ok"
    _mark(d, "ok", run_s=None, trial_budget_s=120.0)
    assert trial_status(d).status == "ok"


def test_each_status_csv_is_joined_only_to_its_own_trace_root(tmp_path, capsys):
    """Paired sweeps share every (arm, trial_index, seed) key; merging their
    trial CSVs would apply one sweep's statuses to the other's traces."""
    off, on = tmp_path / "off_traces", tmp_path / "on_traces"
    _write_trial(off)
    _write_trial(on)
    off_csv = _status_csv(tmp_path / "off.csv", [("H1", 7, 42, "ok")])
    on_csv = _status_csv(tmp_path / "on.csv", [("H1", 7, 42, "no_eval")])
    out_csv = tmp_path / "scored.csv"
    assert main(["--traces", str(off), str(on), "--status-csv", str(off_csv), str(on_csv),
                 "--csv", str(out_csv)]) == 0
    assert "excluded (status not ok): H1 1" in capsys.readouterr().out
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    assert [r["trace_root"] for r in rows] == [str(off)]
    # Reversed, the other root's trial is the one kept.
    assert main(["--traces", str(off), str(on), "--status-csv", str(on_csv), str(off_csv),
                 "--csv", str(out_csv)]) == 0
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    assert [r["trace_root"] for r in rows] == [str(on)]
    # '-' stands for a root without a trial CSV.
    assert main(["--traces", str(off), str(on), "--status-csv", "-", str(on_csv),
                 "--csv", str(out_csv)]) == 0
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    assert [r["trace_root"] for r in rows] == [str(off)]
    # One CSV for two roots is refused rather than guessed at.
    with pytest.raises(SystemExit):
        main(["--traces", str(off), str(on), "--status-csv", str(off_csv)])


def test_the_cli_reports_exclusions_per_arm(tmp_path, capsys):
    root = tmp_path / "traces"
    _write_trial(root)
    _mark(_write_trial(root, TRIAL_8), "no_eval")
    _write_trial(root, "N=3-regime=jittery-rrf=60.0__D1__t7__s42")
    assert main(["--traces", str(root)]) == 0
    assert "excluded (status not ok): D1 0, H1 1" in capsys.readouterr().out

    out = tmp_path / "all.csv"
    assert main(["--traces", str(root), "--include-failed", "--csv", str(out)]) == 0
    assert "status not ok" not in capsys.readouterr().out
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert sorted(r["status"] for r in rows) == ["no_eval", "ok", "ok"]


# ---- #12: provenance ---- #

RECORDED_PROVENANCE = {
    "mission_budget_s": "", "mission_window_adaptation": 0,
    "aggregation": "agg:plain", "aggregation_params": "", "fedprox_rho": 0.0,
    "pass_2_budget": 0, "deadline_law": "additive", "deadline_params": "",
    "miss_priority": 0,
    # Phase 2: one mule, a quorum of 1 (blank), the recorded dock and no
    # policy options.
    "n_mules": 1, "min_participation": "", "dock_params": "", "policy_params": "",
    # Phase 3: the wall clock and every clock setting at its recorded value;
    # this hand-built trial has no L1 schedule, device reliability or model
    # width, so those three are blank too.
    "mission_clock": "", "contact_band": "", "in_flight_response": "",
    "backhaul_model": "", "contact_reliability_source": "", "deadline_time_scale": "",
    "initial_window_s": "", "t_nom_s": "", "session_ttl_s": "", "ferry_params": "",
    "l1_channel": "", "realism": "", "input_dim": "",
}


def _configure(trial_dir, *, fedprox_rho=0.0, **mule):
    """Rewrite a trial's configs the way the topology builder fills them."""
    cfg = {"rf_range_m": 60.0, "n_missions": 4, **mule}
    for path in trial_dir.glob("mule-*.json"):
        path.write_text(json.dumps(cfg))
    for dev in DEVICES:
        (trial_dir / f"device-{dev}.json").write_text(
            json.dumps({"device_id": dev, "fedprox_rho": fedprox_rho})
        )


def test_a_legacy_trace_gets_the_recorded_provenance(tmp_path):
    score = score_trial(_write_trial(tmp_path))
    assert score.provenance == RECORDED_PROVENANCE
    assert set(RECORDED_PROVENANCE) == set(PROVENANCE_COLUMNS)


def test_provenance_follows_the_identity_columns(tmp_path):
    row = score_trial(_write_trial(tmp_path)).to_row()
    assert list(row)[:6 + len(PROVENANCE_COLUMNS)] == [
        "cell_id", "arm", "trial_index", "seed", "trace_root", *PROVENANCE_COLUMNS, "status",
    ]
    assert row["trace_root"] == str(tmp_path)


def test_provenance_is_read_from_the_configs_in_the_drivers_format(tmp_path):
    spec = AggregationSpec.from_config("agg:fedbuff", {"buffer_k": 2})
    law = DeadlineLaw.from_config("multiplicative", {"beta_on": 0.7})
    d = _write_trial(tmp_path)
    _configure(
        d, fedprox_rho=0.01,
        aggregation=spec.rule, aggregation_params=spec.to_params(),
        deadline_law=law.form, deadline_params=law.to_params(),
        mission_budget_s=90, mission_window_adaptation=True,
        pass_2_budget=True, miss_priority=True,
    )
    assert trial_provenance(d) == {
        "mission_budget_s": 90.0, "mission_window_adaptation": 1,
        "aggregation": "agg:fedbuff",
        "aggregation_params": json.dumps(spec.to_params(), sort_keys=True),
        "fedprox_rho": 0.01, "pass_2_budget": 1,
        "deadline_law": "multiplicative",
        "deadline_params": json.dumps(law.to_params(), sort_keys=True),
        "miss_priority": 1,
        "n_mules": 1, "min_participation": "", "dock_params": "", "policy_params": "",
        **{col: "" for col in PROVENANCE_COLUMNS[-13:]},
    }


@pytest.mark.parametrize("policy, options, params", [
    ("whittle", {"whittle_variant": "literal", "whittle_weights": "oort"},
     {"variant": "literal", "weights": "oort"}),
    ("fedcs", {"fedcs_value": "devices"}, {"value": "devices"}),
    ("max_aoi", {"whittle_variant": "literal"}, None),
])
def test_the_fleet_columns_are_read_from_the_configs_in_the_drivers_format(
    tmp_path, policy, options, params,
):
    """Phase 2: the mule count, the cluster's quorum, the dock settings and
    the D3/D5 options, each formatted as the driver's row formats it."""
    d = _write_trial(tmp_path, mules=("m1", "m2"))
    _configure(d, contact_policy=policy, dock_on_empty=True, down_wait_s=120, **options)
    (d / "cluster.json").write_text(json.dumps(
        dict(json.loads((d / "cluster.json").read_text()), min_participation=2)
    ))
    provenance = trial_provenance(d)
    assert (provenance["n_mules"], provenance["min_participation"]) == (2, 2)
    assert provenance["dock_params"] == json.dumps(
        {"dock_on_empty": True, "down_wait_s": 120.0}, sort_keys=True,
    )
    assert provenance["policy_params"] == (
        "" if params is None else json.dumps(params, sort_keys=True)
    )
    # Two mules with a quorum of 1 are not the recorded topology either.
    (d / "cluster.json").write_text(json.dumps({"min_participation": 1}))
    assert (trial_provenance(d)["n_mules"], trial_provenance(d)["min_participation"]) == (2, 1)


def test_the_cli_groups_by_provenance_and_flags_a_trial_scored_twice(tmp_path, capsys):
    plain, buffered = tmp_path / "plain", tmp_path / "fedbuff"
    _write_trial(plain)
    _configure(_write_trial(buffered), aggregation="agg:fedbuff", aggregation_params={"buffer_k": 2})
    assert main(["--traces", str(plain), str(buffered)]) == 0
    out = capsys.readouterr().out
    assert "aggregation=agg:plain" in out and "aggregation=agg:fedbuff" in out
    assert "different provenances" in out
