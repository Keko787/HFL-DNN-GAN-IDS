"""FeRRy Phase 5 (unit U7): the learned arms through the real orchestrator.

Real subprocesses over real TCP, through ``Exp4Driver`` with the traces kept
(the Phase 5 spec's units table, row U7):

* **A stub trial of FQ and of E3**, each flying a checkpoint made in the test
  (under ``tmp_path``), on the Phase 4 pilots' flags. The trial ends ok; the
  kept per-role mule JSON names the checkpoint by path, sha and tag; the mule
  process verified it and says so in ``mule_ready`` (``pair``,
  ``policy_checkpoint``); its missions carry their decision records; the row
  names the checkpoint by tag and sha only, and the trace scorer derives the
  same ``ferry_params`` from the kept trace (the shared helper).
* **A mule process refuses a checkpoint** whose arrays are not the ones its
  config names: it exits before it binds a port, with the reason on stderr.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.analysis.traces_scorer import trial_provenance
from experiments.exp4.driver import Exp4Driver, learned_policy_params, trace_dir_name
from experiments.runner import Cell
from hermes.processes.config import MuleConfig, mule_config_errors, mule_config_to_json
from hermes.scheduler.policies.chen_dqn import new_e3_network, save_e3_checkpoint
from hermes.scheduler.selector.pair_features import PairFeatureSchema
from hermes.scheduler.selector.pair_q import KIND_PAIR_Q, PairQConfig, PairQNet

REPO = Path(__file__).resolve().parents[2]

#: The pilots' flags (the Phase 4 spec, decision 7) at a declared 1 MB and the
#: plan's 60 s budget, the cap S = 2 and T_nom over 5 reference layouts, as in
#: tests/integration/test_p4_plan_trials.py.
PILOT = dict(
    mission_clock="sim", realism=True, contact_band="wide", deadline_time_scale="t_nom",
    in_flight_response="replan", replan_fallback="trim", aggregation="agg:cutoff",
    contact_reliability_source="channel", payload_bytes=1_000_000, mission_budget_s=60.0,
    age_cap_missions=2, t_nom_layouts=5, trial_budget_s=240.0,
)

#: A trained checkpoint's provenance (critic B9 would let a campaign fly it).
TRAINED = dict(reward={"kind": "derived", "c_t": 0.1, "c_cov": 1.0},
               training={"episodes": 2000}, seeds={"init": 1, "train": 7},
               cell_family="jittery", cell_family_sha256="ab" * 32, trainer_commit="a" * 40,
               dirty=False, episodes_trained=2000, validation=[],
               held_out={"episodes": 1000, "return_mean": 0.33})


def _pair_checkpoint(path: Path, seed: int = 0) -> str:
    schema = PairFeatureSchema(("wide", "medium", "narrow"))
    net = PairQNet(schema.dim, PairQConfig(gamma=0.9), seed=seed)
    return net.save(path, kind=KIND_PAIR_Q, purpose="trained", schema=schema.to_json(),
                    classes=list(schema.classes), provenance=TRAINED)


def _cell(arm: str, seed: int, **params) -> Cell:
    p = {"N": 4, "rrf": 60.0, "n_missions": 2, "regime": "jittery"}
    p.update(params)
    return Cell(cell_id="|".join(f"{k}={v}" for k, v in sorted(p.items())), arm=arm,
                trial_index=0, seed=seed, params=p)


def _events(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@pytest.mark.slow
@pytest.mark.parametrize("arm", ["FQ", "E3"])
def test_a_learned_arm_runs_as_a_real_trial(tmp_path, arm):
    ckpt = tmp_path / "ckpt" / f"{arm}.npz"
    if arm == "FQ":
        sha = _pair_checkpoint(ckpt)
        settings = dict(PILOT, pair_checkpoints={"main": str(ckpt)})
    else:
        sha = save_e3_checkpoint(new_e3_network(seed=3), ckpt, band="wide", purpose="trained",
                                 provenance=dict(TRAINED, reward={"kind": "bytes"}))
        settings = dict(PILOT, policy_checkpoints={"e3": str(ckpt)})
    root = tmp_path / "traces"
    cell = _cell(arm, 4242)
    row = dict(Exp4Driver(**settings, trace_root=root).run_trial(cell))
    assert (row["missions_completed"], row["mission_failures"]) == (2, 0), row
    trace = root / trace_dir_name(cell)
    status = json.loads((trace / "trial_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "ok"
    (mule,) = [json.loads(p.read_text(encoding="utf-8")) for p in trace.glob("mule-*.json")]
    (log,) = list(trace.glob("mule-*.jsonl"))
    by_name = {}
    for e in _events(log):
        by_name.setdefault(e["event"], []).append(e)
    (ready,) = by_name["mule_ready"]
    done = by_name["mission_completed"]
    prefix, field = ("pair", "pair") if arm == "FQ" else ("policy", "policy_checkpoint")
    assert mule[f"{prefix}_checkpoint"] == str(ckpt.resolve())
    assert (mule[f"{prefix}_checkpoint_sha256"], mule[f"{prefix}_checkpoint_tag"]) == (
        sha, "main" if arm == "FQ" else "e3")
    assert (ready[field]["sha256"], ready[field]["tag"]) == (sha, mule[f"{prefix}_checkpoint_tag"])
    assert ready[field]["kind"] == ("pair_q" if arm == "FQ" else "chen_dqn")
    params = json.loads(row["ferry_params"])
    if arm == "FQ":
        assert mule["flight_slot"] == "pair_q" and "policy_checkpoint" not in ready
        flown = [e for e in done if e["pass_1_flown"]]
        assert flown and all(len(e["pass_1_pairs"]) == len(e["pass_1_flown"]) for e in flown)
        assert (params["pair_tag"], params["pair_sha256"]) == ("main", sha)
        assert row["policy_params"] == ""
    else:
        assert mule["contact_policy"] == "chen_dqn" and "pair" not in ready
        assert all("pass_1_e3" in e and "pass_1_pairs" not in e for e in done)
        assert row["policy_params"] == json.dumps(learned_policy_params(mule), sort_keys=True)
        assert not [k for k in params if k.startswith(("pair_", "policy_"))]
    # The scorer derives the same ferry_params from the kept trace.
    assert trial_provenance(trace)["ferry_params"] == row["ferry_params"]


@pytest.mark.slow
def test_a_mule_process_refuses_a_checkpoint_it_may_not_fly(tmp_path):
    """Other choices 6: the mule process verifies the sha before it binds a
    port and exits non-zero on a mismatch, so its trial cannot run."""
    ckpt = tmp_path / "main.npz"
    _pair_checkpoint(ckpt, seed=1)
    other = _pair_checkpoint(tmp_path / "other.npz", seed=2)
    cfg = MuleConfig(
        mule_id="m", rf_range_m=60.0, n_missions=1, mission_clock="sim", trial_seed=7,
        contact_band="wide", t_nom_s=200.0, in_flight_response="replan",
        replan_fallback="trim", plan_mode="ferry", member_admission="subset",
        flight_slot="pair_q", pair_checkpoint=str(ckpt), pair_checkpoint_sha256=other,
        pair_checkpoint_tag="main",
    )
    assert mule_config_errors(cfg) == []
    config, port = tmp_path / "mule.json", tmp_path / "mule.port"
    config.write_text(mule_config_to_json(cfg), encoding="utf-8")
    out = subprocess.run(
        [sys.executable, "-m", "hermes.processes.mule", "--config", str(config),
         "--port-out", str(port)],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode != 0 and not port.exists()
    assert "CheckpointRefused" in out.stderr and "not the expected" in out.stderr
