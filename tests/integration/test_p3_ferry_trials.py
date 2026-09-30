"""FeRRy Phase 3, unit U7 — ferry stub trials through the real orchestrator.

Real subprocesses over real TCP, through ``Exp4Driver`` with its traces kept:

* **Determinism.** Two ferry stub trials with the same seed give identical
  simulated payloads: every mission's sim stamps, ledger, flown stops,
  backhaul upload and outcomes, and the cluster's simulated event fields.
  The channel and the draws are pure functions of the trial seed and the
  simulated time, and the contact commit writes in device order, so nothing
  the wall clock or thread timing does reaches them.
* **The Lamport sync at K = 2, quorum 2.** Every merge takes one partial from
  each mule; the DOWN answering it carries the later upload's simulated time
  (lost uploads included, through their stand-ins), and each mule takes off
  for Pass 2 at ``max(own upload + turnaround, cluster time)``, the gap
  charged as ``dock_wait``.
* **The RF link token** reaches every role's JSON, one value per trial, and
  the devices register with it (the trials run).
* **The causal RF prior (critic B4)** under ``--l1-channel`` and the recorded
  ``mission`` backhaul model: each mission plans with the chosen carrier's
  SNR at the last upload already made, never the trial's mean.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.exp4.driver import Exp4Driver, trace_dir_name
from experiments.runner import Cell

SIM_CEILING = 1.0e9
TURNAROUND_S = 30.0
MISSION_SIM_KEYS = (
    "sim_start_s", "sim_end_s", "sim_ledger", "sim_pass_2_start_s", "pass_1_flown",
    "pass_2_flown", "replans", "aborts", "inserts", "offers_refused", "budget_overrun_s",
    "pass_2_budget_overrun_s", "energy_j", "band", "backhaul", "pass_1_outcomes", "pass_1_plan",
    "delivered", "undelivered", "pass_1_clean_devices", "deadline_state", "pass_1_preflight_drops",
)
CLUSTER_SIM_EVENTS = ("up_bundle_ingested", "backhaul_upload_lost", "cluster_round_closed")


def _cell(k: int, seed: int) -> Cell:
    params = {"N": 4 * k, "rrf": 60.0, "n_missions": 3, "regime": "jittery"}
    return Cell(cell_id="|".join(f"{a}={b}" for a, b in sorted(params.items())), arm="H1",
                trial_index=0, seed=seed, params=params)


def _trial(root: Path, *, k: int = 1, seed: int = 31337, **kw) -> Path:
    kw.update(mission_clock="sim", realism=True, contact_band="wide", backhaul_model="seconds",
              trial_budget_s=240.0, trace_root=root)
    if k > 1:
        kw.update(n_mules=k, min_participation=k)
    cell = _cell(k, seed)
    row = dict(Exp4Driver(**kw).run_trial(cell))
    assert row["mission_failures"] == 0 and row["missions_completed"] == 3 * k, row
    assert row["mission_clock"] == "sim"
    return root / trace_dir_name(cell)


def _events(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sim_payload(trace: Path) -> dict:
    out = {}
    for f in sorted(trace.glob("mule-*.jsonl")):
        out[f.name] = [
            {k: e.get(k) for k in MISSION_SIM_KEYS} if e["event"] == "mission_completed"
            else {"sim_start_s": e.get("sim_start_s")}
            for e in _events(f) if e["event"] in ("mission_completed", "mission_started")
        ]
    for f in sorted(trace.glob("cluster-*.jsonl")):
        out[f.name] = [
            {k: v for k, v in e.items() if k != "ts"}
            for e in _events(f) if e["event"] in CLUSTER_SIM_EVENTS
        ]
    return out


@pytest.mark.slow
def test_two_ferry_trials_with_one_seed_give_identical_simulated_payloads(tmp_path):
    a = _sim_payload(_trial(tmp_path / "a"))
    b = _sim_payload(_trial(tmp_path / "b"))
    assert a == b
    missions = [m for m in a["mule-exp4-mule.jsonl"] if "sim_ledger" in m]
    assert len(missions) == 3
    for m in missions:
        assert abs(sum(m["sim_ledger"].values()) - (m["sim_end_s"] - m["sim_start_s"])) < 1e-6
        stamps = [o["contact_ts"] for o in m["pass_1_outcomes"]]
        stamps += [p["deadline_ts"] for p in m["pass_1_plan"]]
        assert stamps and all(1.0e6 <= s < SIM_CEILING for s in stamps)    # no wall stamp
        assert m["band"] == "wide" and m["backhaul"]["t_upload_s"] > m["sim_start_s"]
    # Each mission takes off where the last one landed.
    assert all(missions[i + 1]["sim_start_s"] == missions[i]["sim_end_s"] for i in range(2))


@pytest.mark.slow
def test_two_mules_at_quorum_two_sync_to_the_later_upload(tmp_path):
    trace = _trial(tmp_path / "k2", k=2, aggregation="agg:plain")
    uploads = {}          # mission round -> {mule: upload completion}
    completed = {}
    for f in sorted(trace.glob("mule-*.jsonl")):
        mule = f.stem[len("mule-"):]
        for e in _events(f):
            if e["event"] == "mule_ready":
                assert e["mission_clock"] == "sim" and e["backhaul_model"] == "seconds"
            if e["event"] == "mission_completed":
                uploads.setdefault(e["mission_round"], {})[mule] = e["backhaul"]["t_upload_s"]
                completed[(mule, e["mission_round"])] = e
    assert set(uploads) == {1, 2, 3} and all(len(v) == 2 for v in uploads.values())

    cluster = _events(next(trace.glob("cluster-*.jsonl")))
    closed = [e for e in cluster if e["event"] == "cluster_round_closed"]
    for e in cluster:
        if e["event"] in ("up_bundle_ingested", "backhaul_upload_lost"):
            assert e["sim_upload_ts"] == uploads[e["mission_round"]][e["mule_id"]]
    # A closed round's time is the later of its two uploads.
    for c in closed:
        assert c["sim_ts"] in {max(v.values()) for v in uploads.values()}

    synced = 0
    for (mule, rnd), e in completed.items():
        latest = max(uploads[rnd].values())
        own = uploads[rnd][mule]
        if e["sim_pass_2_start_s"] is None:
            continue                          # an empty mission: no Pass 2
        assert e["sim_pass_2_start_s"] == pytest.approx(max(own + TURNAROUND_S, latest), abs=1e-9)
        assert e["sim_ledger"]["dock_wait"] == pytest.approx(
            max(0.0, latest - (own + TURNAROUND_S)), abs=1e-9)
        synced += e["sim_ledger"]["dock_wait"] > 0.0
    assert synced >= 1, "no mule ever waited for the other: the sync was not exercised"


@pytest.mark.slow
def test_the_trials_link_token_is_on_every_role_and_reproducible(tmp_path):
    trace = _trial(tmp_path / "tok", seed=2024)
    token = Exp4Driver.trial_link_token(_cell(1, 2024))
    configs = [json.loads(p.read_text(encoding="utf-8"))
               for p in list(trace.glob("mule-*.json")) + list(trace.glob("device-*.json"))]
    assert len(configs) == 5
    assert {c["rf_link_token"] for c in configs} == {token}
    assert all(c["newest_solicit_only"] for c in configs if "device_id" in c)


@pytest.mark.slow
def test_an_l1_channel_trial_plans_each_mission_with_the_last_uploads_snr(tmp_path):
    """Critic B4 through real processes, under ``--l1-channel`` and the
    recorded ``mission`` backhaul model: each mission's Pass-1 plan is handed
    the chosen carrier's SNR at the last upload already made (20 dB before the
    first), never the trial's mean over every mission; a mission that did not
    dock leaves it as it was."""
    from experiments.exp4.channel import ChannelModel, backhaul_plan
    from experiments.exp4.driver import chosen_snr_schedule

    params = {"N": 4, "rrf": 60.0, "n_missions": 3, "regime": "jittery"}
    cell = Cell(cell_id="|".join(f"{a}={b}" for a, b in sorted(params.items())), arm="H3",
                trial_index=0, seed=4099, params=params)
    root = tmp_path / "l1"
    row = dict(Exp4Driver(mission_clock="sim", realism=True, l1_channel=True,
                          trial_budget_s=240.0, trace_root=root).run_trial(cell))
    assert row["mission_failures"] == 0 and row["missions_completed"] == 3, row
    assert (row["mission_clock"], row["l1_channel"], row["backhaul_model"]) == ("sim", 1, "")

    events = _events(next((root / trace_dir_name(cell)).glob("mule-*.jsonl")))
    (ready,) = [e for e in events if e["event"] == "mule_ready"]
    assert ready["rf_prior_source"] == "mission_schedule" and ready["backhaul_model"] == "mission"
    model = ChannelModel(n_bands=3, n_missions=3, seed=cell.seed, jittery=True)
    plan = backhaul_plan(model, adaptive=True)
    schedule = chosen_snr_schedule(model, plan)
    undocked = {e["mission_round"] for e in events if e["event"] == "mission_empty"}
    expected, prior = [], 20.0
    for e in events:
        if e["event"] == "mission_completed":
            expected.append(prior)
            if e["mission_round"] not in undocked:
                prior = schedule[e["mission_round"] - 1]
    started = [e["rf_prior_snr_db"] for e in events if e["event"] == "mission_started"]
    assert started == expected and started[0] == 20.0
    assert len(set(started)) > 1, "no mission docked: the prior was never fed"
