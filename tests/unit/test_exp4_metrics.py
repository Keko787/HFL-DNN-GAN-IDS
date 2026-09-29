"""EX-4.0 unit tests — JSONL event stream -> federation metric roll-up.

Pins the pure ``observation_from_rows`` + ``summarise_observation`` path
with hand-authored event envelopes, so the metric aggregation is
verified deterministically without spawning the real orchestrator (the
slow end-to-end wiring is covered by
``tests/integration/test_exp4_smoke.py``).
"""

from __future__ import annotations

import math

import pytest

from experiments.exp4 import (
    consume_run_dir,
    observation_from_rows,
    summarise_observation,
)


# --------------------------------------------------------------------------- #
# Synthetic three-mission scenario with known metric values
# --------------------------------------------------------------------------- #

def _mission_completed(
    *, round_, contacts, updates, scheduled, clean, delivered, undelivered, dur,
):
    return {
        "event": "mission_completed",
        "role": "mule",
        "id": "exp4-mule",
        "mission_round": round_,
        "pass_1_contacts": contacts,
        "pass_2_contacts": contacts,
        "pass_1_updates": updates,
        "pass_1_scheduled": scheduled,
        "pass_1_clean_devices": clean,
        "delivered": delivered,
        "undelivered": undelivered,
        "duration_s": dur,
    }


def _scenario_rows():
    devices = ["d0", "d1", "d2", "d3"]
    device_rows = [
        {"event": "device_ready", "role": "device", "id": d} for d in devices
    ]
    # Serve counts: d0=6, d1=6, d2=4, d3=2.
    serve_counts = {"d0": 6, "d1": 6, "d2": 4, "d3": 2}
    for d, n in serve_counts.items():
        for _ in range(n):
            device_rows.append(
                {"event": "device_served", "role": "device", "id": d,
                 "outcome": "clean"}
            )

    mule_rows = [
        {"event": "mule_ready", "role": "mule", "id": "exp4-mule"},
        {"event": "dock_bootstrapped", "role": "mule", "id": "exp4-mule"},
        _mission_completed(round_=0, contacts=1, updates=4, scheduled=4,
                           clean=["d0", "d1", "d2", "d3"], delivered=4,
                           undelivered=0, dur=1.0),
        _mission_completed(round_=1, contacts=1, updates=3, scheduled=4,
                           clean=["d0", "d1", "d2"], delivered=3,
                           undelivered=1, dur=1.2),
        _mission_completed(round_=2, contacts=1, updates=2, scheduled=4,
                           clean=["d0", "d1"], delivered=2,
                           undelivered=2, dur=0.8),
    ]

    cluster_rows = [
        {"event": "cluster_ready", "role": "cluster", "id": "exp4-cluster"},
    ]
    for i in range(3):
        cluster_rows.append(
            {"event": "up_bundle_ingested", "role": "cluster",
             "id": "exp4-cluster", "mule_id": "exp4-mule", "mission_round": i}
        )
        cluster_rows.append(
            {"event": "cluster_round_closed", "role": "cluster",
             "id": "exp4-cluster", "cluster_round": i}
        )
    return cluster_rows, mule_rows, device_rows


def test_observation_from_rows_parses_all_streams():
    cluster_rows, mule_rows, device_rows = _scenario_rows()
    obs = observation_from_rows(
        cluster_rows=cluster_rows, mule_rows=mule_rows,
        device_rows=device_rows, n_devices=4,
    )
    assert obs.cluster_rounds_closed == 3
    assert obs.up_bundles_ingested == 3
    assert obs.missions_completed == 3
    assert obs.mission_failures == 0
    assert obs.per_device_serves == {"d0": 6, "d1": 6, "d2": 4, "d3": 2}
    assert obs.cluster_ready and obs.mule_ready and obs.dock_bootstrapped
    # Mission fields round-trip.
    m0 = obs.missions[0]
    assert m0.pass_1_updates == 4
    assert m0.pass_1_scheduled == 4
    assert m0.pass_1_clean_devices == ("d0", "d1", "d2", "d3")
    assert m0.delivered == 4 and m0.undelivered == 0


def test_summary_metric_values():
    cluster_rows, mule_rows, device_rows = _scenario_rows()
    obs = observation_from_rows(
        cluster_rows=cluster_rows, mule_rows=mule_rows,
        device_rows=device_rows, n_devices=4,
    )
    s = summarise_observation(
        obs, n_devices=4, rf_range_m=60.0, n_missions_target=3,
    )

    # Yield + quorum close rates.
    assert s.update_yield == pytest.approx(3.0)  # (4+3+2)/3
    assert s.round_close_rate_kmin1 == pytest.approx(1.0)
    assert s.round_close_rate_kmin2 == pytest.approx(1.0)
    assert s.round_close_rate_kminhalf == pytest.approx(1.0)   # k=2
    assert s.round_close_rate_kminN == pytest.approx(1.0 / 3)  # only m0 has 4

    # Coverage + fairness over visits [6,6,4,2].
    assert s.coverage == pytest.approx(1.0)
    assert s.jains_fairness == pytest.approx(324.0 / 368.0)      # 18^2/(4*92)
    assert s.participation_entropy == pytest.approx(_entropy([6, 6, 4, 2]))

    # Completion counts [3,3,2,1] over 4 devices.
    assert s.mission_completion_rate == pytest.approx(1.0)
    assert s.completion_fairness == pytest.approx(81.0 / 92.0)   # 9^2/(4*23)

    # Two-pass / contact structure.
    assert s.pass2_coverage == pytest.approx((1.0 + 0.75 + 0.5) / 3)
    assert s.rho_contact == pytest.approx(4.0)                   # 12 devices / 3 contacts

    # Run-shape counters.
    assert s.rounds_closed == 3
    assert s.missions_completed == 3
    assert s.mission_failures == 0
    assert s.pass1_contacts_mean == pytest.approx(1.0)
    assert s.pass2_contacts_mean == pytest.approx(1.0)
    assert s.mission_duration_s_mean == pytest.approx(1.0)      # (1.0+1.2+0.8)/3
    assert s.n_devices == 4
    assert s.rf_range_m == pytest.approx(60.0)
    assert s.n_missions_target == 3

    # Row is CSV-shaped and complete.
    row = s.to_row()
    from experiments.exp4.metrics import Exp4MetricSummary
    assert set(row.keys()) == set(Exp4MetricSummary.csv_columns())


def test_zero_serve_device_is_padded_into_fairness_denominator():
    """A device that announces but never serves lowers coverage/fairness."""
    device_rows = [
        {"event": "device_ready", "role": "device", "id": d}
        for d in ["d0", "d1", "d2", "d3"]
    ]
    # Only d0..d2 ever serve; d3 stays silent.
    for d in ["d0", "d1", "d2"]:
        device_rows.append({"event": "device_served", "role": "device", "id": d})
    obs = observation_from_rows(
        cluster_rows=[], mule_rows=[], device_rows=device_rows, n_devices=4,
    )
    assert obs.per_device_serves == {"d0": 1, "d1": 1, "d2": 1, "d3": 0}
    s = summarise_observation(obs, n_devices=4, rf_range_m=60.0, n_missions_target=1)
    assert s.coverage == pytest.approx(3.0 / 4.0)
    # Jain's over [1,1,1,0]: 3^2/(4*3) = 0.75 — the silent device drags it down.
    assert s.jains_fairness == pytest.approx(0.75)


def test_empty_observation_is_degenerate_not_crashing():
    obs = observation_from_rows(
        cluster_rows=[], mule_rows=[], device_rows=[], n_devices=4,
    )
    s = summarise_observation(obs, n_devices=4, rf_range_m=60.0, n_missions_target=2)
    assert s.missions_completed == 0
    assert s.rounds_closed == 0
    assert s.update_yield == pytest.approx(0.0)
    assert s.round_close_rate_kmin1 == pytest.approx(0.0)
    assert s.pass2_coverage == pytest.approx(0.0)
    assert s.rho_contact == pytest.approx(0.0)
    # Degenerate fairness over an empty distribution is defined as 1.0
    # (matches the Exp-3 convention), not NaN.
    assert s.jains_fairness == pytest.approx(1.0)


def test_convergence_metrics_from_model_eval_events():
    """EX-4.1 — model_eval trace -> init/final/best AUC, ΔAUC, T@τ."""
    cluster_rows = [
        {"event": "cluster_ready", "role": "cluster", "id": "c"},
        {"event": "model_eval", "role": "cluster", "id": "c",
         "cluster_round": 0, "accuracy": 0.50, "auc": 0.50, "loss": 0.70, "n_test": 100},
        {"event": "model_eval", "role": "cluster", "id": "c",
         "cluster_round": 1, "accuracy": 0.80, "auc": 0.85, "loss": 0.40, "n_test": 100},
        {"event": "model_eval", "role": "cluster", "id": "c",
         "cluster_round": 2, "accuracy": 0.92, "auc": 0.95, "loss": 0.25, "n_test": 100},
    ]
    obs = observation_from_rows(
        cluster_rows=cluster_rows, mule_rows=[], device_rows=[], n_devices=4,
    )
    assert len(obs.model_evals) == 3
    s = summarise_observation(
        obs, n_devices=4, rf_range_m=60.0, n_missions_target=2, tau=0.9,
    )
    assert s.init_auc == pytest.approx(0.50)
    assert s.final_auc == pytest.approx(0.95)
    assert s.best_auc == pytest.approx(0.95)
    assert s.delta_auc == pytest.approx(0.45)
    assert s.rounds_evaluated == 3
    assert s.t_at_tau_round == 2      # first round>0 hitting acc>=0.9
    assert s.tau == pytest.approx(0.9)


def test_convergence_absent_on_stub_path():
    """No model_eval events -> convergence columns are blank, not crashing."""
    obs = observation_from_rows(
        cluster_rows=[], mule_rows=[], device_rows=[], n_devices=4,
    )
    s = summarise_observation(obs, n_devices=4, rf_range_m=60.0, n_missions_target=2)
    assert s.final_auc is None
    assert s.rounds_evaluated == 0
    assert s.t_at_tau_round is None
    row = s.to_row()
    assert row["final_auc"] == ""    # blank CSV cell
    assert row["rounds_evaluated"] == 0


def test_consume_run_dir_reads_role_globs(tmp_path):
    """End-to-end of the file layer: role-prefixed JSONL -> observation."""
    import json

    cluster_rows, mule_rows, device_rows = _scenario_rows()
    _write_jsonl(tmp_path / "cluster-exp4-cluster.jsonl", cluster_rows)
    _write_jsonl(tmp_path / "mule-exp4-mule.jsonl", mule_rows)
    # Two device files (multi-device topology) — globbed + concatenated.
    d_first = [r for r in device_rows if r["id"] in ("d0", "d1")]
    d_second = [r for r in device_rows if r["id"] in ("d2", "d3")]
    _write_jsonl(tmp_path / "device-d0.jsonl", d_first)
    _write_jsonl(tmp_path / "device-d2.jsonl", d_second)

    obs = consume_run_dir(tmp_path, n_devices=4)
    assert obs.cluster_rounds_closed == 3
    assert obs.missions_completed == 3
    assert obs.per_device_serves == {"d0": 6, "d1": 6, "d2": 4, "d3": 2}


# --------------------------------------------------------------------------- #
# FeRRy audit #3 / #4 — merged updates and the cluster's merge ledger
# --------------------------------------------------------------------------- #

def test_the_merged_count_drives_yield_and_quorum_but_not_completion():
    """An update excluded past its age cutoff was collected, not merged."""
    _, mule_rows, device_rows = _scenario_rows()
    mule_rows[2] = dict(mule_rows[2], pass_1_merged_devices=["d0", "d1"],
                        pass_1_merged_updates=2)          # d2, d3 excluded
    mule_rows[3] = dict(mule_rows[3], pass_1_merged_devices=["d0", "d1", "d2"])
    obs = observation_from_rows(
        cluster_rows=[], mule_rows=mule_rows, device_rows=device_rows, n_devices=4,
    )
    s = summarise_observation(obs, n_devices=4, rf_range_m=60.0, n_missions_target=3)
    assert s.update_yield == pytest.approx((2 + 3 + 2) / 3)   # m2 has no merged fields: 2
    assert s.round_close_rate_kminN == pytest.approx(0.0)      # no round merged all 4
    assert s.round_close_rate_kmin2 == pytest.approx(1.0)
    # Completion still counts finished sessions: [3, 3, 2, 1] as before.
    assert s.mission_completion_rate == pytest.approx(1.0)
    assert s.completion_fairness == pytest.approx(81.0 / 92.0)


def test_deferred_and_expired_rounds_do_not_close():
    """Mission 0 merges on arrival, 1 is buffered and flushed by 2, 3 expires."""
    _, mule_rows, device_rows = _scenario_rows()
    mule_rows.append(_mission_completed(
        round_=3, contacts=1, updates=1, scheduled=4, clean=["d3"],
        delivered=1, undelivered=3, dur=1.0,
    ))
    cluster_rows = [
        {"event": "cluster_merge", "role": "cluster", "id": "c", "mission_round": 0,
         "rule": "agg:fedbuff", "applied": True, "partials": [["exp4-mule", 0]]},
        {"event": "cluster_merge_deferred", "role": "cluster", "id": "c",
         "mule_id": "exp4-mule", "mission_round": 1, "applied": False},
        {"event": "cluster_merge", "role": "cluster", "id": "c", "mission_round": 2,
         "rule": "agg:fedbuff", "applied": True,
         "partials": [["exp4-mule", 1], ["exp4-mule", 2]]},
        {"event": "cluster_merge_expired", "role": "cluster", "id": "c",
         "mule_id": "exp4-mule", "mission_round": 3},
    ]
    obs = observation_from_rows(
        cluster_rows=cluster_rows, mule_rows=mule_rows, device_rows=device_rows, n_devices=4,
    )
    assert obs.deferred_rounds == {1}
    assert obs.flush_of == {1: 2}
    assert obs.expired_rounds == {3}
    s = summarise_observation(obs, n_devices=4, rf_range_m=60.0, n_missions_target=4)
    assert s.round_close_rate_kmin1 == pytest.approx(0.5)      # rounds 0 and 2
    assert s.update_yield == pytest.approx((4 + 3 + 2 + 1) / 4)  # yield is per mission
    # The flush closes round 2 with its own 2 updates plus round 1's 3, so it
    # meets the full-slice quorum (4) that its own updates alone would miss.
    assert s.round_close_rate_kminN == pytest.approx(0.5)      # rounds 0 and 2
    assert s.round_close_rate_kmin2 == pytest.approx(0.5)


def test_a_phase_1_trace_recovers_its_exclusions_from_the_merge_record():
    """Traces from Phase 1 until the merged fields existed record the
    exclusions only in ``pass_1_merge.excluded``; they are not credited."""
    _, mule_rows, device_rows = _scenario_rows()
    mule_rows[2] = dict(mule_rows[2], pass_1_merge={
        "rule": "agg:cutoff", "devices": ["d0", "d1"], "excluded": ["d2", "d3"],
    })
    obs = observation_from_rows(
        cluster_rows=[], mule_rows=mule_rows, device_rows=device_rows, n_devices=4,
    )
    assert obs.missions[0].pass_1_merged_devices == ("d0", "d1")
    assert obs.missions[0].pass_1_merged_updates == 2
    assert obs.missions[1].pass_1_merged_devices is None   # no merge record
    s = summarise_observation(obs, n_devices=4, rf_range_m=60.0, n_missions_target=3)
    assert s.update_yield == pytest.approx((2 + 3 + 2) / 3)
    assert s.round_close_rate_kminN == pytest.approx(0.0)


def test_an_all_excluded_mission_still_counts_its_completed_sessions():
    """Every update past its cutoff: nothing merged, but the sessions finished."""
    _, mule_rows, device_rows = _scenario_rows()
    mule_rows[4] = dict(mule_rows[4], pass_1_merged_devices=[],
                        pass_1_merged_updates=0)             # d0, d1 excluded
    obs = observation_from_rows(
        cluster_rows=[], mule_rows=mule_rows, device_rows=device_rows, n_devices=4,
    )
    s = summarise_observation(obs, n_devices=4, rf_range_m=60.0, n_missions_target=3)
    assert s.update_yield == pytest.approx((4 + 3 + 0) / 3)
    assert s.round_close_rate_kmin1 == pytest.approx(2 / 3)
    # Completions [3, 3, 2, 1] as in the scenario: the excluded sessions count.
    assert s.mission_completion_rate == pytest.approx(1.0)
    assert s.completion_fairness == pytest.approx(81.0 / 92.0)


def test_a_partial_cut_inside_an_applied_fold_is_expired():
    """A mixed fold lists its zero-weight partials apart; they never reached θ."""
    _, mule_rows, device_rows = _scenario_rows()
    cluster_rows = [
        {"event": "cluster_merge", "role": "cluster", "id": "c", "mission_round": 2,
         "rule": "agg:cutoff", "applied": True, "partials": [["exp4-mule", 2]],
         "expired_partials": [["exp4-mule", 1]]},
    ]
    obs = observation_from_rows(
        cluster_rows=cluster_rows, mule_rows=mule_rows, device_rows=device_rows, n_devices=4,
    )
    assert obs.expired_rounds == {1}
    s = summarise_observation(obs, n_devices=4, rf_range_m=60.0, n_missions_target=3)
    assert s.round_close_rate_kmin1 == pytest.approx(2 / 3)    # rounds 0 and 2


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _entropy(counts):
    total = sum(counts)
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def _write_jsonl(path, rows):
    import json
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
