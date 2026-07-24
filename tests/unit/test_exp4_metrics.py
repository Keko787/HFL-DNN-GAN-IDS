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
