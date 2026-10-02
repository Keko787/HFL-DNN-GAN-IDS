"""FeRRy Phase 5 (unit U8b): FerrySim's trainer, its checkpoints, Study 5.5's report, the CLI.

Pinned here (the Phase 5 spec's units table, row U8b; the user's decisions 4,
5 and 9; orchestrator resolutions R2, R6, R9, R10 and R11):

* **T3 for training**: a 50-episode smoke run at N = 3, done twice, writes the
  same arrays (the same sha) and the same manifest, byte for byte; another
  seed writes others; every γ of one seed starts from the same weights and
  replay seed (common random numbers across the grid).
* **The run's episodes**: the family's training stream (``cells.train_episode``'s
  draw), the validation stream's first episodes spread over the cells, disjoint
  streams, and the behaviour schedule (critic C4) and each episode's own stream
  reaching every episode, for the pair score and for E3.
* **Transitions** carry the next decision's rows and mask (the effective mask
  after an empty one), ``done`` exactly at a sortie's last decision, for the
  pair score and for E3; the rewards pushed are the reward's terms and sum to
  the episode's return.
* **Validation keeps the best**: the saved weights are the best validation's, a
  tie is no new best, and the run stops after three validations without one.
* **The manifest**: every field, the training spec field by field, R2's purpose
  and learner revision in the header, no wall time; the runner's refusals
  before and after the held-out score lands; E3's checkpoint as its loader
  reads it.
* **One checkpoint, two paths**: FerrySim's install path flies as the config
  path (arm FQ with the checkpoint) on every mission, cluster and device event;
  and one stub FQ trial through the real orchestrator equals FerrySim's run of
  it on the row, the configs and every mule and cluster event (critic B8), bar
  what the wall clock and the OS decide and the row's three device-serve
  columns, whose harness artifacts FerrySim's row holds: its devices log only
  ``device_ready`` (resolution R25).
* **The plan** (the orchestrator's resolution R23; critic C2): a run trains and
  validates under its plan score settings (FQ-dwell's, FQ-cov's, a pilot's),
  every episode handing them over as a driver override, and records them;
  a run on the cells' own plan hands over and records none (its calls,
  arrays and manifest are the ones they always were); a checkpoint flies,
  and is scored, on its recorded plan, FQ-dwell's and FQ-cov's install path
  flying as their config path; one evaluation reads its policies on one
  plan; the command line's ``--ablation`` and ``--plan-score-params``; and ε,
  read only from a headroom report flown on the evaluation's plan (the Phase 5
  repair round's A-1).
* **A checkpoint's tag** (resolution R24) on the trainer's own manifests: the
  γ tag of its γ, its tag's reward (the trainer's rewards, so E3 has only its
  bytes), and the update count of the weights it keeps; and on the command
  line, an explicit learned arm's tag takes only a run that arm flies (its γ,
  its reward at decision 4 (a)'s weights under a γ tag, dwell and cov, and an
  ablation's plan; the repair round's A-2 and A-4).
* **Feature support** (critic B15, C5): the device model moves no ``pair_v1``
  column, and the stack's stub trials stay inside FerrySim's training ranges.
* **The checkpoint layout**: the study, the arm's tag, γ in hundredths and the
  seed; a checkpoint of another cell family is never replaced, since the
  layout has no family (the calibration's two families, one study each).
* **The report** on synthetic tables: rising (with and without beating the best
  fixed rule by ε, the margin alone deciding), flat (TOST at ε), inconclusive,
  the sanity check's failure and its boundary, the best γ picked on
  validation, Holm, decision 5's four fixed rules with FX the arm itself (R6),
  ``greedy_1``'s flag against the FX arm (critic A2), the trend and the stack
  check; the table from an evaluation file (the best γ picked on the cells
  read, not on the N = 6 control) and its refusals (one learner revision,
  trained checkpoints, one sweep, the rule's references).
* **The pre-registered grid** (resolution R26): decision 5 (a)'s six γ, 10
  seeds each and 1,000 held-out episodes per cell are labelled pre-registered;
  a sweep off it, with fewer or more (3, 7, 8, 9 or 12 seeds; a missing,
  off-grid or extra γ; a short, long or unrecorded held-out count; the
  calibration), is labelled not, with each reason, in the verdict's JSON and on
  its second line, never refused, and decided as it would be on the grid: the
  verdict the rule gives today, which the label leaves alone (the repair
  round's C-T1 and C-T2); the report command prints and saves the label.
* **The command line**: a dirty tree is refused unless allowed (train and
  sweep), a checkpoint is never overwritten unasked, the learner flags reach
  the manifest, evaluate flies the N = 6 control by default, reads every policy
  under ``--reward`` and records it, and report reads an evaluation file and
  refuses one it cannot read as a usage error; every command line puts the
  caller's logging level back when it returns, refuses or raises.
* **Study 5.6's family and cells** (resolution R22): the command line takes
  ``jittery56`` and Study 5.6's cells wherever it names a family or a cell, and
  no default moves; a jittery56 score trains under a study of its own (R19);
  Study 5.5's verdict reads jit-n12-120 and jit-n12-180, whichever family
  trained the score.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.exp4.driver import (
    CHECKPOINT_TAGS,
    F_COV_SCORE,
    FQ_DWELL_SCORE,
    PAIR_CHECKPOINT_TAGS,
    Exp4Driver,
    trace_dir_name,
)
from experiments.ferrysim import __main__ as CLI
from experiments.ferrysim import cells as C
from experiments.ferrysim import checkpoints as K
from experiments.ferrysim import episode as E
from experiments.ferrysim import inprocess as IP
from experiments.ferrysim import report as RP
from experiments.ferrysim import reward as R
from experiments.ferrysim import train as T
from hermes.l1.contact_link import CLASSES
from hermes.scheduler.policies import chen_dqn
from hermes.scheduler.selector.pair_features import (
    SPARSE_COLUMNS,
    LearnedPairScorer,
    PairFeatureSchema,
    load_pair_scorer,
    pair_rows,
)
from hermes.scheduler.selector.pair_q import (
    LEARNER_REVISION,
    MANIFEST_KEYS,
    BehaviourSchedule,
    LearnerSettings,
    PairQConfig,
    PairQNet,
    campaign_refusals,
    verify_checkpoint,
)

from tests.golden import _build_p4_plan as UG5
from tests.golden import _canon

REPO = Path(__file__).resolve().parents[2]

#: The smoke cell at N = 3 (the spec's row U8b), an episode costing about 0.1 s.
#: Its 20 s budget gives sorties of none to three decisions, so a smoke run's
#: transitions bootstrap, and some missions make no decision at all.
N3 = C.FerryCell("smoke-n3", C.FAMILY_JITTERY, C.ROLE_CONTROL, 3, 20.0, C.BUDGET_STRESS_PRIOR,
                 2, "jittery")
N12 = C.cell_named("jit-n12-120")
#: A learner small enough to update inside a smoke run.
SMOKE_LEARNER = LearnerSettings(batch=16, replay_capacity=1000, warmup_transitions=32,
                                behaviour=BehaviourSchedule(reference_episodes=10))
TREE = K.TreeState(commit="0123456789abcdef0123456789abcdef01234567", dirty=False)


def _smoke_spec(seed: int = 3) -> T.TrainSpec:
    return T.pair_spec(0.9, seed, family="smoke", cells=(N3,), episodes=50, eval_every=25,
                       val_episodes=4, learner=SMOKE_LEARNER)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """One smoke run at N = 3 (T3's first), its checkpoint shared by the tests that
    fly a trained checkpoint; a test that changes the manifest copies it first."""
    return T.train(_smoke_spec(), tmp_path_factory.mktemp("smoke") / "g0.9_s3.npz", tree=TREE)


def _val_seed(cell=N12, index=0):
    return C.stream_seeds(C.VAL_STREAM, cell.name, 1, start=index)[0]


def _bootstrap_provenance():
    return {"reward": {}, "training": {}, "seeds": {}, "cell_family": None,
            "cell_family_sha256": None, "trainer_commit": None, "dirty": True,
            "episodes_trained": 0, "validation": [], "held_out": None}


def _keys(value):
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def _copy_checkpoint(path, where):
    path = Path(path)
    for suffix in (".npz", ".json"):
        shutil.copy(path.with_suffix(suffix), where / path.with_suffix(suffix).name)
    return where / path.name


def _planned_copy(path, where, plan):
    """A copy of the checkpoint ``path`` whose training spec records the plan ``plan``
    (resolution R23): the same arrays and sha, the record being outside the sha,
    read as a checkpoint trained under that plan."""
    where.mkdir(parents=True, exist_ok=True)
    copy = _copy_checkpoint(path, where)
    manifest = json.loads(copy.with_suffix(".json").read_text(encoding="utf-8"))
    manifest["training"]["spec"]["plan_score_params"] = dict(plan)
    copy.with_suffix(".json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                                         encoding="utf-8")
    assert verify_checkpoint(copy)["sha256"] == verify_checkpoint(path)["sha256"]
    return copy


def _close(a, b):
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def _empty(cell, seed, index, reward):
    """An episode that flew no mission."""
    return E.EpisodeResult(cell=cell.name, seed=seed, trial_index=index, policy="learned",
                           arm="FX", reward=reward, sorties=(), rewards=(), pair_records=(),
                           steps=(), row={})


def _fly_nothing(monkeypatch):
    """Runs that fly nothing: every training and validation episode is empty, so a
    run takes no time, makes no update and saves its initial weights."""
    monkeypatch.setattr(T, "fly_pair_episode",
                        lambda net, schema, cell, seed, *, index, reward, trainer=None:
                        _empty(cell, seed, index, reward))
    monkeypatch.setattr(T, "fly_e3_episode",
                        lambda net, cell, seed, *, index, reward, bootstrap, epsilon=0.0,
                        rng_seed=0: (_empty(cell, seed, index, reward), []))


# --------------------------------------------------------------------------- #
# T3, the run's episodes and its behaviour
# --------------------------------------------------------------------------- #

def test_t3_a_50_episode_smoke_run_at_n3_twice_writes_the_same_sha(smoke, tmp_path):
    again = T.train(_smoke_spec(), tmp_path / "again.npz", tree=TREE)
    assert again.sha256 == smoke.sha256
    assert (tmp_path / "again.json").read_bytes() == Path(smoke.path).with_suffix(
        ".json").read_bytes()
    assert smoke.episodes_run == 50 and len(smoke.curve) == 2
    outcome = smoke.manifest["training"]["outcome"]
    # one update per decision once the replay holds the warm-up's transitions
    assert outcome["transitions"] >= 150
    assert outcome["updates"] == outcome["transitions"] - SMOKE_LEARNER.warmup_transitions + 1
    other = T.train(_smoke_spec(seed=4), tmp_path / "other.npz", tree=TREE)
    assert other.sha256 != smoke.sha256


@pytest.mark.parametrize("kind", ["pair_q", "chen_dqn"])
def test_every_gamma_of_one_seed_starts_from_the_same_weights_and_replay(kind, monkeypatch,
                                                                          tmp_path):
    """Common random numbers across Study 5.5's grid (``train.run_seeds``): a run's
    initial weights and its replay's sampling seed are its seed's, whatever its
    γ, and another seed's differ. A run that flies nothing makes no update, so
    its checkpoint holds the weights it started from."""
    _fly_nothing(monkeypatch)
    replays = []
    real = T.PairReplay

    def replay(capacity, *, seed):
        replays.append(seed)
        return real(capacity, seed=seed)

    monkeypatch.setattr(T, "PairReplay", replay)
    build = T.e3_spec if kind == "chen_dqn" else T.pair_spec

    def initial(gamma, seed):
        res = T.train(build(gamma, seed, episodes=1, eval_every=1),
                      tmp_path / f"g{gamma}_s{seed}.npz", tree=TREE)
        with np.load(res.path, allow_pickle=False) as data:
            return res.sha256, {name: data[name] for name in data.files
                                if name not in ("header", "format_version")}

    sha0, at0 = initial(0.0, 5)
    sha9, at9 = initial(0.9, 5)
    _, elsewhere = initial(0.9, 6)
    assert len(at0) >= 4 and sorted(at0) == sorted(at9)
    assert all(np.array_equal(at0[name], at9[name]) for name in at0)
    assert sha0 != sha9      # the header binds γ, the weights are the seed's
    assert not all(np.array_equal(at9[name], elsewhere[name]) for name in at9)
    assert replays == [T.run_seeds(5)["replay"]] * 2 + [T.run_seeds(6)["replay"]]


def test_a_runs_episodes_are_its_familys_training_stream():
    spec = T.pair_spec(0.0, 5)
    assert spec.cells == C.FAMILIES[C.FAMILY_JITTERY]
    assert spec.family_sha256 == C.family_sha256(C.FAMILY_JITTERY)
    assert T.pair_spec(0.9, 5).family_sha256 == spec.family_sha256
    assert T.TrainSpec(family=C.FAMILY_CLEAN).family_sha256 == C.family_sha256(C.FAMILY_CLEAN)
    train = [T.training_episode(5, spec.family, spec.cells, i) for i in range(40)]
    assert train == [C.train_episode(5, C.FAMILY_JITTERY, i) for i in range(40)]
    val = T.validation_episodes(spec.cells, 200)
    assert [sum(1 for c, _, _ in val if c is cell) for cell in spec.cells] == [50] * 4
    for cell in spec.cells:
        mine = [(i, s) for c, i, s in val if c is cell]
        assert [s for _, s in mine] == list(C.stream_seeds(C.VAL_STREAM, cell.name, 50))
        assert [i for i, _ in mine] == list(range(50))
    assert [sum(1 for c, _, _ in T.validation_episodes(spec.cells, 6) if c is cell)
            for cell in spec.cells] == [2, 2, 1, 1]
    C.check_disjoint([(C.train_stream(5), [s for _, s in train]),
                      (C.VAL_STREAM, [s for _, _, s in val])])
    seeds = T.run_seeds(5)
    assert seeds == T.run_seeds(5) and seeds["run"] == 5
    assert seeds["train_stream"] == "ferrysim-train-5" and seeds["val_stream"] == C.VAL_STREAM
    assert seeds["init"] != T.run_seeds(6)["init"] and seeds["init"] != seeds["replay"]
    assert len({T.behaviour_seed(5, e) for e in range(100)}) == 100
    with pytest.raises(ValueError, match="val_episodes"):
        T.pair_spec(0.0, 0, val_episodes=3)


@pytest.mark.parametrize("kind", ["pair_q", "chen_dqn"])
def test_every_episode_flies_its_behaviour_from_its_own_stream(kind, monkeypatch, tmp_path):
    """Critic C4: episode e flies the schedule's ε and reference phase at e of the
    planned episodes, from the episode's own stream, on the training reward;
    E3 flies its arm from the run's bootstrap checkpoint, with no reference,
    and the bootstrap is scratch."""
    calls = []

    def fly_pair(net, schema, cell, seed, *, index, reward, trainer=None):
        calls.append((cell, seed, index, reward, trainer.epsilon, trainer.around_reference,
                      trainer.rng_seed))
        return _empty(cell, seed, index, reward)

    def fly_e3(net, cell, seed, *, index, reward, bootstrap, epsilon=0.0, rng_seed=0):
        manifest = verify_checkpoint(bootstrap)
        assert (manifest["kind"], manifest["purpose"], manifest["classes"]) == (
            "chen_dqn", "bootstrap", ["wide"])
        calls.append((cell, seed, index, reward, epsilon, False, rng_seed))
        return _empty(cell, seed, index, reward), []

    monkeypatch.setattr(T, "fly_pair_episode", fly_pair)
    monkeypatch.setattr(T, "fly_e3_episode", fly_e3)
    schedule = BehaviourSchedule(reference_episodes=0 if kind == "chen_dqn" else 4,
                                 epsilon_start=0.4, epsilon_end=0.1, decay_fraction=0.75)
    build = T.e3_spec if kind == "chen_dqn" else T.pair_spec
    spec = build(0.5, 7, episodes=12, eval_every=12, val_episodes=4,
                 learner=LearnerSettings(behaviour=schedule))
    T.train(spec, tmp_path / "x.npz", tree=TREE, validator=lambda net, s, e: (0.0, {}, 0))
    assert len(calls) == 12
    for e, (cell, seed, index, reward, epsilon, around, rng_seed) in enumerate(calls):
        behaviour = schedule.at(e, 12)
        assert (cell, seed) == T.training_episode(7, C.FAMILY_JITTERY, spec.cells, e)
        assert index == e and reward == spec.reward and reward.expected_availability
        assert (epsilon, around) == (behaviour.epsilon, behaviour.around_reference)
        assert rng_seed == T.behaviour_seed(7, e)
    assert len({round(c[4], 9) for c in calls}) > 2
    if kind == "pair_q":
        assert [c[5] for c in calls] == [True] * 4 + [False] * 8
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.npz"]


def test_a_spec_refuses_what_a_run_may_not_train(tmp_path, monkeypatch):
    def flown(*args, **kwargs):
        raise AssertionError("an episode flew before the path was checked")

    monkeypatch.setattr(T, "fly_pair_episode", flown)
    # refused before a single episode, not by the save after a whole run
    with pytest.raises(ValueError, match=".npz"):
        T.train(_smoke_spec(), tmp_path / "g0.9_s3.pt", tree=TREE)
    assert not any(tmp_path.iterdir())
    with pytest.raises(ValueError, match="expected availability"):
        T.pair_spec(0.9, 0, reward=R.DERIVED)
    with pytest.raises(ValueError, match="trains on"):
        T.pair_spec(0.9, 0, reward=T.BYTES_TRAINING_REWARD)
    with pytest.raises(ValueError, match="trains on"):
        T.e3_spec(0.9, 0, reward=T.TRAINING_REWARD)
    with pytest.raises(ValueError, match="reference"):
        T.e3_spec(0.9, 0, learner=LearnerSettings())
    with pytest.raises(ValueError, match="kind"):
        T.TrainSpec(kind="ddqn")
    with pytest.raises(ValueError, match="cells"):
        T.TrainSpec(family="nowhere")
    assert T.pair_spec(0.25, 1, reward=T.HAND_TRAINING_REWARD).gamma == 0.25


def test_a_spec_records_its_plan_only_when_it_has_one():
    """Resolution R23: a run's plan score settings are ``PlanScoreParams``' fields, of
    their types and ranges, as the driver reads them, kept as the spec's own copy;
    the training spec records them only when set, so a default spec's record is
    the one it always was, and a spec stays hashable. E3 flies no plan, and a
    run's cells set none, so the record is the plan the run flew."""
    default = T.pair_spec(0.9, 0)
    assert default.plan_score_params == {} and "plan_score_params" not in default.to_json()
    assert T.ABLATIONS == {"dwell": FQ_DWELL_SCORE, "cov": F_COV_SCORE}
    assert set(T.ABLATIONS) == {CHECKPOINT_TAGS["FQ-dwell"], CHECKPOINT_TAGS["FQ-cov"]}
    plan = {"c_cov_per_device": 0.25, "dwell_in_delta": False}
    spec = T.pair_spec(0.9, 0, plan_score_params=plan)
    assert spec.to_json() == dict(default.to_json(), plan_score_params=plan)
    plan["c_energy"] = 0.0
    assert spec.plan_score_params == {"c_cov_per_device": 0.25, "dwell_in_delta": False}
    again = T.pair_spec(0.9, 0, plan_score_params={"dwell_in_delta": False,
                                                   "c_cov_per_device": 0.25})
    assert again == spec and hash(again) == hash(spec) and spec != default
    assert hash(default) == hash(T.pair_spec(0.9, 0))
    assert dataclasses.replace(spec, seed=1).plan_score_params == spec.plan_score_params
    for bad, error, match in (({"kappa": 1.0}, ValueError, r"unknown settings \['kappa'\]"),
                              ({"dwell_in_delta": 0}, TypeError, "dwell_in_delta must be a bool"),
                              ({"c_link": -1.0}, ValueError, "c_link must be >= 0"),
                              ({"coverage_rank": "best"}, ValueError, "coverage_rank"),
                              ([("c_link", 0.0)], TypeError, "a mapping")):
        with pytest.raises(error, match=match):
            T.pair_spec(0.9, 0, plan_score_params=bad)
    with pytest.raises(ValueError, match="E3 flies legacy mode, with no plan"):
        T.e3_spec(0.9, 0, plan_score_params=FQ_DWELL_SCORE)
    assert T.e3_spec(0.9, 0).plan_score_params == {}

    class Planned(C.FerryCell):
        def driver_settings(self):
            return dict(super().driver_settings(), plan_score_params={"c_energy": 0.0})

    with pytest.raises(ValueError, match=r"cells \['smoke-n3'\] set plan_score_params"):
        T.pair_spec(0.9, 0, family="smoke", cells=(Planned(**N3.to_json()),), val_episodes=1)


def test_a_runs_plan_reaches_every_training_and_validation_episode(monkeypatch, tmp_path):
    """Resolution R23 (the final check's F1; critic C2): a 2-episode FQ-dwell run
    trains and validates under FQ-dwell's plan, every episode handing it to
    ``run_episode`` as a driver override, and its manifest's training spec records
    it, so ``checkpoint_flight`` flies the checkpoint on it; a run on the cells'
    own plan hands over none, its calls being the ones they always were."""
    calls = []
    fly_real, run_real = T.fly_pair_episode, T.run_episode

    def fly(*args, **kwargs):
        calls.append(("fly", kwargs.get("trainer") is not None,
                      kwargs.get("plan_score_params", "none")))
        return fly_real(*args, **kwargs)

    def run(*args, **kwargs):
        calls.append(("run", kwargs.get("driver_overrides", "none")))
        return run_real(*args, **kwargs)

    monkeypatch.setattr(T, "fly_pair_episode", fly)
    monkeypatch.setattr(T, "run_episode", run)
    dwell = T.ABLATIONS["dwell"]
    spec = T.pair_spec(0.9, 0, family="smoke", cells=(N3,), episodes=2, eval_every=1,
                       val_episodes=1, learner=SMOKE_LEARNER, plan_score_params=dwell)
    res = T.train(spec, tmp_path / "dwell.npz", tree=TREE)
    # each training episode (a trainer attached) is followed by a validation one
    assert [c[1:] for c in calls if c[0] == "fly"] == [(True, dwell), (False, dwell)] * 2
    assert [c[1] for c in calls if c[0] == "run"] == [{"plan_score_params": dwell}] * 4
    assert res.manifest["training"]["spec"]["plan_score_params"] == dwell
    assert K.trained_plan(res.manifest) == dwell
    assert K.checkpoint_flight(res.path)[1] == {"plan_score_params": dwell}
    calls.clear()
    res = T.train(dataclasses.replace(spec, plan_score_params={}), tmp_path / "plain.npz",
                  tree=TREE)
    assert [c[1:] for c in calls if c[0] == "fly"] == [(True, "none"), (False, "none")] * 2
    assert [c[1] for c in calls if c[0] == "run"] == ["none"] * 4
    assert "plan_score_params" not in res.manifest["training"]["spec"]
    assert K.trained_plan(res.manifest) == {} and K.checkpoint_flight(res.path)[1] == {}


# --------------------------------------------------------------------------- #
# Transitions and rewards
# --------------------------------------------------------------------------- #

def _pair_episode(index=0):
    spec = T.pair_spec(0.9, 0)
    schema = spec.schema()
    net = PairQNet(schema.dim, spec.network, seed=7)
    episode = T.fly_pair_episode(net, schema, N12, _val_seed(index=index), index=index,
                                 reward=T.TRAINING_REWARD,
                                 trainer=E.Trainer(epsilon=0.3, rng_seed=index))
    return schema, episode


def test_pair_transitions_carry_the_next_decisions_rows_and_mask():
    schema, ep = _pair_episode()
    transitions = T.pair_transitions(ep, schema)
    assert len(transitions) == ep.decisions >= 8
    at = 0
    after_empty = 0
    for steps, sortie, rewards in zip(ep.steps, ep.sorties, ep.rewards):
        assert len(steps) == len(sortie.stops) == len(rewards)
        for k, step in enumerate(steps):
            t = transitions[at]
            at += 1
            rows = pair_rows(step.view, schema)[0]
            assert np.array_equal(t.x, rows[step.row])
            assert t.reward == rewards[k].total
            if k == len(steps) - 1:
                assert t.done and t.next_rows is None and t.next_mask is None
                assert sortie.stops[k].terminal
                continue
            nxt = steps[k + 1]
            assert not t.done and not sortie.stops[k].terminal
            assert np.array_equal(t.next_rows, pair_rows(nxt.view, schema)[0])
            assert tuple(t.next_mask) == nxt.effective_mask
            if not any(nxt.mask):
                # an empty mask flew FX's pair alone, so that pair is the next mask
                assert tuple(t.next_mask) == tuple(r == nxt.row for r in range(len(nxt.mask)))
                after_empty += 1
    assert after_empty >= 1
    assert any(len(steps) >= 3 for steps in ep.steps)
    with pytest.raises(ValueError, match="line up"):
        T.pair_transitions(dataclasses.replace(ep, steps=(ep.steps[0][:-1],) + ep.steps[1:]),
                           schema)
    with pytest.raises(ValueError, match="trainer"):
        T.pair_transitions(dataclasses.replace(ep, steps=()), schema)


def test_a_mission_without_a_decision_reaches_the_trainer_empty_and_adds_nothing():
    """Resolution R9: the slot closes every mission, one with no Pass-1 decision
    as ``close_mission(((), ()))``; the trainer reads it as a mission with no
    transition, and the next mission's decisions are its own sortie's."""
    spec = T.pair_spec(0.9, 0)
    schema = spec.schema()
    net = PairQNet(schema.dim, spec.network, seed=7)
    sunk = []
    ep = E.run_episode(N3, _val_seed(N3), E.Policy(
        "learned", scorer=functools.partial(LearnedPairScorer, net, schema)),
        reward=T.TRAINING_REWARD, trainer=E.Trainer(epsilon=0.3, rng_seed=0),
        sink=lambda steps, records: sunk.append((steps, records)))
    assert sunk[0] == ((), ()) and ep.steps[0] == () and ep.sorties[0].stops == ()
    transitions = T.pair_transitions(ep, schema)
    assert len(transitions) == ep.decisions == sum(len(s) for s in ep.steps) >= 3
    assert [t.done for t in transitions] == [
        k == len(s.stops) - 1 for s in ep.sorties for k in range(len(s.stops))]
    assert not all(t.done for t in transitions)


def test_the_rewards_pushed_are_the_terms_and_sum_to_the_return():
    schema, ep = _pair_episode(index=5)
    transitions = T.pair_transitions(ep, schema)
    terms = [t for rewards in ep.rewards for t in rewards]
    assert [t.reward for t in transitions] == [x.total for x in terms]
    for x in terms:
        assert x.total == x.gain - x.time - x.energy - x.distance - x.coverage
        assert x.energy == 0.0 and x.distance == 0.0
    assert sum(x.coverage > 0 for x in terms) >= 1
    assert _close(sum(t.reward for t in transitions), ep.ret)
    assert _close(sum(ep.sortie_returns), ep.ret)
    total = ep.terms
    assert _close(total.total, ep.ret)
    assert _close(total.gain - total.time - total.coverage, ep.ret)
    # training reads the expected availability, validation the realized draw
    realized = ep.rescored(dataclasses.replace(T.TRAINING_REWARD, expected_availability=False))
    assert realized.ret != ep.ret


def _e3_bootstrap(where):
    path = where / "e3_bootstrap.npz"
    chen_dqn.save_e3_checkpoint(chen_dqn.new_e3_network(seed=0), path, band="wide",
                                purpose="bootstrap", provenance=_bootstrap_provenance())
    return path


def test_e3_transitions_carry_the_next_decisions_rows_and_mask(tmp_path):
    net = chen_dqn.new_e3_network(seed=1)
    ep, steps = T.fly_e3_episode(net, N12, _val_seed(), index=0,
                                 reward=T.BYTES_TRAINING_REWARD,
                                 bootstrap=_e3_bootstrap(tmp_path), epsilon=0.3, rng_seed=5)
    transitions = T.e3_transitions(ep, steps)
    assert len(transitions) == len(steps) == ep.decisions >= 8
    at = 0
    for sortie, rewards in zip(ep.sorties, ep.rewards):
        mine = steps[at:at + len(sortie.stops)]
        for k, step in enumerate(mine):
            t = transitions[at + k]
            assert np.array_equal(t.x, step.x) and t.reward == rewards[k].total
            assert tuple(step.devices) == sortie.stops[k].devices
            if k == len(mine) - 1:
                assert t.done and t.next_rows is None
            else:
                assert np.array_equal(t.next_rows, mine[k + 1].matrix)
                assert tuple(t.next_mask) == mine[k + 1].mask
        at += len(sortie.stops)
    assert any(len(s.stops) >= 3 for s in ep.sorties)
    assert _close(sum(t.reward for t in transitions), ep.ret)
    assert all(x.gain == x.total for rewards in ep.rewards for x in rewards)
    with pytest.raises(ValueError):
        T.e3_transitions(ep, steps[:-1])


# --------------------------------------------------------------------------- #
# Validation and the checkpoint
# --------------------------------------------------------------------------- #

def test_validation_keeps_the_best_and_stops_after_three_without_one(tmp_path):
    scores = iter([0.1, 0.3, 0.2, 0.35, 0.35, 0.34, 0.33, 0.9])
    seen = {}

    def scripted(net, spec, episode):
        seen[episode] = net.weights()
        return next(scores), {"smoke-n3": 0.25}, 1

    spec = T.pair_spec(0.9, 1, family="smoke", cells=(N3,), episodes=20, eval_every=2,
                       val_episodes=1, learner=LearnerSettings(
                           batch=8, replay_capacity=200, warmup_transitions=8,
                           behaviour=BehaviourSchedule(reference_episodes=2)))
    res = T.train(spec, tmp_path / "v.npz", tree=TREE, validator=scripted)
    assert sorted(seen) == [2, 4, 6, 8, 10, 12, 14]
    assert (res.best_episode, res.episodes_run, res.stopped_early) == (8, 14, True)
    net, manifest = PairQNet.load(res.path, expect_sha256=res.sha256, expect_kind="pair_q",
                                  expect_schema=spec.schema().to_json(),
                                  expect_classes=list(CLASSES))
    saved = net.weights()
    assert all(np.array_equal(saved[k], seen[8][k]) for k in saved)
    assert not all(np.array_equal(seen[8][k], seen[14][k]) for k in saved)
    assert manifest["episodes_trained"] == 8
    assert [p["score"] for p in manifest["validation"]] == [0.1, 0.3, 0.2, 0.35, 0.35, 0.34,
                                                            0.33]
    assert manifest["training"]["outcome"]["best_episode"] == 8
    with pytest.raises(FileExistsError):
        T.train(spec, tmp_path / "v.npz", tree=TREE, validator=scripted)


def test_validation_flies_the_validation_stream_greedily_at_the_realized_draw(smoke):
    spec = _smoke_spec()
    net, _ = PairQNet.load(smoke.path, expect_sha256=smoke.sha256, expect_kind="pair_q",
                           expect_schema=spec.schema().to_json(), expect_classes=list(CLASSES))
    score, cells, flown = T.validate(net, spec)
    returns = [E.run_episode(N3, seed, K.checkpoint_flight(smoke.path)[0], trial_index=i,
                             reward=spec.validation_reward).ret
               for _, i, seed in T.validation_episodes(spec.cells, 4)]
    assert flown == 4 and cells == {"smoke-n3": statistics.fmean(returns)}
    kept = next(p for p in smoke.curve if p.episode == smoke.best_episode)
    assert score == statistics.fmean(returns) == kept.score
    # every cell weighs the same: three episodes over two cells (two and one)
    other = dataclasses.replace(N3, name="smoke-n3-60", budget_s=60.0)
    two = dataclasses.replace(spec, cells=(N3, other), val_episodes=3)
    score, cells, flown = T.validate(net, two)
    assert flown == 3 and set(cells) == {"smoke-n3", "smoke-n3-60"}
    assert cells["smoke-n3"] == statistics.fmean(returns[:2])
    assert score == statistics.fmean(cells.values())
    assert score != statistics.fmean(returns[:2] + [cells["smoke-n3-60"]])


def test_the_manifest_records_the_run(smoke, tmp_path):
    path = _copy_checkpoint(smoke.path, tmp_path)
    m = verify_checkpoint(path)
    spec = _smoke_spec()
    schema = PairFeatureSchema(tuple(CLASSES))
    assert set(m) == set(MANIFEST_KEYS)
    assert (m["kind"], m["purpose"], m["learner_revision"]) == ("pair_q", "trained",
                                                                LEARNER_REVISION)
    assert m["schema"] == schema.to_json() and m["schema"]["dim"] == 36
    assert m["classes"] == ["wide", "medium", "narrow"]
    assert m["gamma"] == m["network"]["gamma"] == 0.9
    assert m["reward"] == spec.reward.to_json() and m["reward"]["expected_availability"]
    # the training spec field by field, not read back through TrainSpec.to_json:
    # the report's one-learner check compares exactly this (U2's hand-off)
    assert m["training"]["spec"] == {
        "kind": "pair_q", "seed": 3, "family": "smoke", "cells": ["smoke-n3"],
        "episodes": 50, "eval_every": 25, "val_episodes": 4, "patience": 3, "phase": True,
        "network": dict(PairQConfig().to_json(), gamma=0.9),
        "learner": {"batch": 16, "replay_capacity": 1000, "warmup_transitions": 32,
                    "behaviour": {"reference_episodes": 10, "epsilon_start": 0.3,
                                  "epsilon_end": 0.05, "decay_fraction": 0.5}},
        "reward": R.RewardSpec(expected_availability=True).to_json(),
        "validation_reward": R.DERIVED.to_json(),
        "device_model": IP.DEVICE_MODEL_EQUAL,
    }
    outcome = m["training"]["outcome"]
    assert outcome == {"episodes_run": 50, "best_episode": smoke.best_episode,
                       "stopped_early": False, "updates": smoke.curve[-1].updates,
                       "transitions": smoke.curve[-1].transitions}
    assert outcome["updates"] == outcome["transitions"] - SMOKE_LEARNER.warmup_transitions + 1
    assert m["seeds"] == T.run_seeds(3)
    assert (m["cell_family"], m["cell_family_sha256"]) == ("smoke", spec.family_sha256)
    assert (m["trainer_commit"], m["dirty"]) == (TREE.commit, False)
    assert m["episodes_trained"] == smoke.best_episode in (25, 50)
    assert m["validation"] == [p.to_json() for p in smoke.curve]
    assert [p["episode"] for p in m["validation"]] == [25, 50]
    assert all(set(p["cells"]) == {"smoke-n3"} for p in m["validation"])
    assert m["held_out"] is None
    assert not [k for k in _keys(m) if "wall" in k]
    assert campaign_refusals(m) == ["it has no held-out score (the evaluator fills it)"]
    summaries = K.evaluate_checkpoints([path], [N3], episodes=2)[str(path)]
    scored = K.record_held_out_score(path, summaries, R.DERIVED)
    assert scored["sha256"] == smoke.sha256 and campaign_refusals(scored) == []
    assert scored["held_out"]["episodes"] == 2
    assert scored["held_out"]["return_mean"] == statistics.fmean(s["return"] for s in summaries)
    assert scored["held_out"]["reward"] == R.DERIVED.to_json()
    scorer = load_pair_scorer(path, expect_sha256=smoke.sha256, classes=CLASSES)
    assert scorer.manifest["held_out"] == scored["held_out"]


def test_a_held_out_score_weighs_its_cells_alike_and_lands_on_its_own_checkpoint(smoke,
                                                                                tmp_path):
    def summary(cell, i, ret):
        return {"cell": cell, "stream": C.HELDOUT_STREAM, "policy": "p", "index": i,
                "return": ret, "decisions": [1],
                "terms": {k: 0.0 for k in ("gain", "time", "energy", "distance", "coverage")}}

    group = [summary("a", 0, 1.0), summary("a", 1, 3.0), summary("b", 0, 0.0)]
    score = K.held_out_score(group, R.DERIVED)
    assert (score["episodes"], score["return_mean"]) == (3, 1.0)
    assert set(score["cells"]) == {"a", "b"}
    with pytest.raises(ValueError, match="held-out"):
        K.held_out_score([dict(s, stream=C.VAL_STREAM) for s in group], R.DERIVED)
    with pytest.raises(ValueError, match="one policy"):
        K.held_out_score(group + [dict(summary("a", 2, 0.0), policy="q")], R.DERIVED)
    path = _copy_checkpoint(smoke.path, tmp_path)
    with pytest.raises(ValueError, match="summaries are of"):
        K.record_held_out_score(path, [dict(s, checkpoint="elsewhere.npz") for s in group],
                                R.DERIVED)
    # two paths of one checkpoint, and a training stream, are refused before any flight
    with pytest.raises(ValueError, match="one checkpoint"):
        K.evaluate_checkpoints([smoke.path, path], [N3], episodes=1)
    with pytest.raises(ValueError, match="training stream"):
        K.evaluate_checkpoints([path], [N3], stream=C.train_stream(0), episodes=1)


def test_e3_trains_from_a_bootstrap_it_does_not_keep_and_saves_a_chen_dqn_checkpoint(tmp_path):
    spec = T.e3_spec(0.9, 2, family="smoke", cells=(N3,), episodes=8, eval_every=4,
                     val_episodes=1, learner=LearnerSettings(
                         batch=8, replay_capacity=200, warmup_transitions=8,
                         behaviour=T.E3_BEHAVIOUR))
    res = T.train(spec, tmp_path / "e3.npz", tree=TREE)
    m = res.manifest
    assert (m["kind"], m["purpose"], m["classes"]) == ("chen_dqn", "trained", ["wide"])
    assert m["schema"] == chen_dqn.e3_schema()
    assert m["reward"] == T.BYTES_TRAINING_REWARD.to_json()
    # E3's own behaviour: the pair learner's ε schedule with no reference phase
    assert m["training"]["spec"]["learner"] == {
        "batch": 8, "replay_capacity": 200, "warmup_transitions": 8,
        "behaviour": {"reference_episodes": 0, "epsilon_start": 0.3, "epsilon_end": 0.05,
                      "decay_fraction": 0.5}}
    assert m["training"]["outcome"]["updates"] > 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["e3.json", "e3.npz"]
    chen_dqn.load_e3_network(res.path, expect_sha256=res.sha256, band="wide")
    policy = chen_dqn.ChenDQNPolicy.from_checkpoint(res.path, expect_sha256=res.sha256,
                                                    band="wide")
    assert policy.manifest["sha256"] == res.sha256
    flight, overrides = K.checkpoint_flight(res.path)
    assert flight.arm == "E3" and overrides == {"policy_checkpoints": {"e3": res.path}}
    n6 = C.cell_named("jit-n6-90")
    ep = E.run_episode(n6, _val_seed(n6), flight, driver_overrides=overrides, reward=R.BYTES)
    assert ep.decisions >= 1
    assert json.loads(ep.row["policy_params"]) == {"policy_sha256": res.sha256,
                                                   "policy_tag": "e3"}


def test_checkpoints_are_scored_alike_in_worker_processes(smoke):
    """A checkpoint's tasks travel to spawned workers (the loader as a
    ``functools.partial``), and the workers fly what this process flies."""
    here = K.evaluate_checkpoints([smoke.path], [N3], episodes=2)
    there = K.evaluate_checkpoints([smoke.path], [N3], episodes=2, workers=2)
    assert here == there and len(here[smoke.path]) == 2


# --------------------------------------------------------------------------- #
# One checkpoint, two paths; the real orchestrator
# --------------------------------------------------------------------------- #

#: Where FQ-dwell's and FQ-cov's plans change the smoke checkpoint's flight: the
#: first validation episode of jit-n12-180 (the final check's F1 probe used that
#: cell; at jit-n12-120 the dwell change leaves the first one as it is).
ABLATION_CELL = C.cell_named("jit-n12-180")


@pytest.mark.parametrize("arm", ["FQ", "FQ-dwell", "FQ-cov"])
def test_the_install_path_flies_as_the_config_path(arm, smoke, tmp_path):
    """FerrySim trains and scores through the install path (FX's configuration,
    ``install_flight_slot`` and the plan the checkpoint trained under); stack
    trials fly the config path (the arm with the checkpoint of its tag). One
    checkpoint gives the same missions both ways (the spec, other choices 8):
    FQ's on Study 5.5's N = 12 cell, and FQ-dwell's and FQ-cov's, trained under
    their arms' plans (resolution R23), on a seed where that plan changes the
    flight."""
    tag = CHECKPOINT_TAGS[arm]
    if arm == "FQ":
        cell, path, plan = N12, Path(smoke.path), {}
    else:
        cell, plan = ABLATION_CELL, T.ABLATIONS[tag]
        path = _planned_copy(smoke.path, tmp_path, plan)
    seed = _val_seed(cell)
    policy, overrides = K.checkpoint_flight(path)
    assert overrides == ({"plan_score_params": plan} if plan else {})
    assert policy.arm == "FX" and policy.pair_slot
    install = E.run_episode(cell, seed, policy, keep_case=True, driver_overrides=overrides)
    config = E.run_episode(cell, seed, E.Policy.of_arm(arm), keep_case=True,
                           driver_overrides={"pair_checkpoints": {tag: str(path)}})
    assert [s.to_json() for s in install.sorties] == [s.to_json() for s in config.sorties]
    assert install.pair_records == config.pair_records
    # FQ-cov's plan can be empty (a mission with no Pass-1 stop holds no record)
    records = [r for rs in install.pair_records if rs for r in rs]
    assert len(records) >= (8 if arm == "FQ" else 4) and all(
        r["scorer"] == "pair_v1" and r["q"] is not None for r in records)
    if plan:
        # the plan matters here: the cells' own plan flies this checkpoint otherwise
        own = E.run_episode(cell, seed, policy)
        assert [s.to_json() for s in own.sorties] != [s.to_json() for s in install.sorties]
    for part in ("mission_started", "mission_completed", "cluster_events", "device_events",
                 "mule_events"):
        assert install.case[part] == config.case[part], part
    (a,), (b,) = install.case["mule_ready"].values(), config.case["mule_ready"].values()
    assert (a[0]["flight_slot"], b[0]["flight_slot"]) == ("cross_heuristic", "pair_q")
    assert "pair" not in a[0] and (b[0]["pair"]["sha256"], b[0]["pair"]["tag"]) == (
        smoke.sha256, tag)

    def rest(event):
        return {k: v for k, v in event.items() if k not in ("flight_slot", "pair")}

    assert [rest(e) for e in a] == [rest(e) for e in b]
    fx_params = json.loads(install.row["ferry_params"])
    fq_params = json.loads(config.row["ferry_params"])
    assert fq_params == dict(fx_params, flight_slot="pair_q", pair_tag=tag,
                             pair_sha256=smoke.sha256)
    assert fq_params["plan_score_params"] == plan


#: What a real process's wall clock and OS decide (UG5's probe, U8a's FX test).
_WALL_KEYS = {"ts", "duration_s", "rf_port", "dock_port", "mule_rf_port", "port",
              "timer.mission_duration_s", "plan_wall_s"}
#: The row's wall column (the harness clock's in process, U8a's FX test).
_ROW_WALL = {"mission_duration_s_mean"}
#: The row's device-serve columns, built from ``device_served``, which only the
#: device process's service loop logs; FerrySim runs none, so its row holds
#: UG4's harness artifacts (resolution R25; U8a's FX test).
_ROW_SERVES = {"coverage", "jains_fairness", "participation_entropy"}
FERRYSIM_SERVES = {"coverage": 0.0, "jains_fairness": 1.0, "participation_entropy": 0.0}


def _masked(value, keys):
    if isinstance(value, dict):
        return {k: ("<masked>" if k in keys else _masked(v, keys)) for k, v in value.items()}
    if isinstance(value, list):
        return [_masked(v, keys) for v in value]
    return value


def _device_events(case):
    """The names of every event a canonical case's devices logged."""
    return {e["event"] for events in case["device_events"].values() for e in events}


def _number(value):
    """A canonical case's float (``"f:<repr>"``) as a float."""
    return float(value[2:]) if isinstance(value, str) and value.startswith("f:") else value


@pytest.mark.slow
def test_a_stub_fq_trial_through_the_real_orchestrator_is_ferrysims(smoke, tmp_path):
    """Critic B8: one stub FQ trial (UG5's N = 12 layout, the checkpoint trained
    above) through the real orchestrator (real processes and TCP), against
    FerrySim's run of it: the row, the configs and every mule and cluster event,
    bar what the wall clock and the OS decide (the stamps, the ports, the row's
    mean mission duration) and the row's three device-serve columns (B8's
    scope; device events are compared by name only and exit codes not, as in
    U8a's FX test). Those columns come from the devices' serves: the real
    devices log ``device_served`` and FerrySim's only ``device_ready``
    (resolution R25), so FerrySim's row holds the harness artifacts."""
    settings, fx_cell = UG5.TRIALS["fx_n12_120s"]
    settings = dict(settings, pair_checkpoints={"main": smoke.path})
    cell = dataclasses.replace(fx_cell, arm="FQ")
    row = dict(Exp4Driver(**settings, trace_root=tmp_path, trial_budget_s=300.0).run_trial(cell))
    trace = tmp_path / trace_dir_name(cell)
    files = {p.name: p.read_text(encoding="utf-8") for p in sorted(trace.iterdir())
             if p.suffix in (".json", ".jsonl") and p.name != "trial_status.json"}
    devices = [SimpleNamespace(device_id=json.loads(text)["device_id"])
               for f, text in files.items() if f.startswith("device-") and f.endswith(".json")]
    real = IP.case_of(settings, cell, row, SimpleNamespace(
        topology=SimpleNamespace(devices=devices), files=files, exit_codes={}))
    sim = E.run_episode_on(Exp4Driver(**settings), cell, E.Policy.of_arm("FQ"),
                           device_model=IP.DEVICE_MODEL_STUB, keep_case=True,
                           case_settings=settings).case
    for part in ("row", "configs", "mule_ready", "mission_started", "mission_completed",
                 "cluster_events"):
        keys = _WALL_KEYS | (_ROW_WALL | _ROW_SERVES if part == "row" else set())
        assert _masked(sim[part], keys) == _masked(real[part], keys), part
    for part in ("names", "other"):
        assert _canon.diff(_masked(sim["mule_events"][part], _WALL_KEYS),
                           _masked(real["mule_events"][part], _WALL_KEYS)) == [], part
    completed = real["mission_completed"]["exp4-mule"]
    assert len(completed) == 4 and sum(len(e.get("pass_1_pairs", [])) for e in completed) >= 8
    assert real["mule_ready"]["exp4-mule"][0]["pair"]["sha256"] == smoke.sha256
    # the serve columns: the real devices served, FerrySim's only announced
    # themselves, so FerrySim's row holds the artifacts and the real row does not
    assert set(real["device_events"]) == set(sim["device_events"])
    assert _device_events(real) >= {"device_ready", "device_served"}
    assert _device_events(sim) == {"device_ready"}
    assert {k: _number(sim["row"][k]) for k in _ROW_SERVES} == FERRYSIM_SERVES
    assert _number(real["row"]["coverage"]) > 0.0
    assert _number(real["row"]["participation_entropy"]) > 0.0


# --------------------------------------------------------------------------- #
# Feature support (critic B15, C5)
# --------------------------------------------------------------------------- #

def _rows(episode, schema):
    return np.concatenate([pair_rows(step.view, schema)[0]
                           for steps in episode.steps for step in steps], axis=0)


def test_ferrysims_feature_ranges_cover_a_stub_stack_trials(smoke):
    """The stack's stub trials (UG5's: N = 12 at 120 s with S = 3, and N = 6 at
    45 s), flown by the trained score, against FerrySim's training sample (the
    jittery family's training stream, 16 episodes at N = 12 and 8 at N = 6,
    flown as the reference phase flies). The device model moves no column (the
    same trial gives the same rows under the stub and under equal shards; critic
    C5's value-of-s column would not), every value keeps its declared bounds,
    and, but for the columns U1 declares sparse (``pair_features.SPARSE_COLUMNS``:
    events the cells make rare, which a sample of a few episodes can miss),
    every flag value occurs in the sample and no value lies beyond the sample's
    range by more than a quarter of that range's width (a continuous column's
    extremes move with the sample's size; here the largest excursion is under a
    tenth)."""
    schema = PairFeatureSchema(tuple(CLASSES))
    family = C.FAMILIES[C.FAMILY_JITTERY]
    draws = [T.training_episode(0, C.FAMILY_JITTERY, family, i) for i in range(120)]
    picked = ([d for d in draws if d[0].n_devices == 12][:16]
              + [d for d in draws if d[0].n_devices == 6][:8])
    sample = np.concatenate([
        _rows(E.run_episode(cell, seed, E.Policy.scripted("fx_pair"), trial_index=i,
                            trainer=E.Trainer(epsilon=0.3, rng_seed=i, around_reference=True)),
              schema) for i, (cell, seed) in enumerate(picked)], axis=0)
    lo, hi = sample.min(axis=0), sample.max(axis=0)
    policy = K.checkpoint_flight(smoke.path)[0]
    for name in ("fx_n12_120s", "fx_45s"):
        settings, cell = UG5.TRIALS[name]
        rows = {model: _rows(E.run_episode_on(Exp4Driver(**settings), cell, policy,
                                              device_model=model, trainer=E.Trainer()), schema)
                for model in (IP.DEVICE_MODEL_STUB, IP.DEVICE_MODEL_EQUAL)}
        stub = rows[IP.DEVICE_MODEL_STUB]
        assert stub.shape[0] >= 15 and np.array_equal(stub, rows[IP.DEVICE_MODEL_EQUAL]), name
        for j, column in enumerate(schema.column_specs):
            values = stub[:, j]
            if column.low is not None:
                assert values.min() >= column.low, (name, column.name)
            if column.high is not None:
                assert values.max() <= column.high, (name, column.name)
            if column.name in SPARSE_COLUMNS:
                continue
            if column.flag:
                assert set(values) <= set(sample[:, j]), (name, column.name)
            margin = 0.25 * (hi[j] - lo[j])
            assert lo[j] - margin <= values.min() and values.max() <= hi[j] + margin, (
                name, column.name, values.min(), values.max(), lo[j], hi[j])


# --------------------------------------------------------------------------- #
# The checkpoint layout and the tree
# --------------------------------------------------------------------------- #

def test_the_checkpoint_layout_names_the_study_the_tag_gamma_and_seed(tmp_path):
    assert K.checkpoint_path(tmp_path, "5.5", "g90", 0.9, 3) == tmp_path / "5.5" / "g90" / (
        "g0.9_s3.npz")
    assert K.checkpoint_path("r", "5.5", "g0", 0.0, 0) == Path("r/5.5/g0/g0_s0.npz")
    assert [K.gamma_tag(g) for g in RP.GAMMAS] == ["g0", "g25", "g50", "g75", "g90", "g99"]
    assert {K.gamma_tag(g) for g in RP.GAMMAS} == {
        tag for arm, tag in CHECKPOINT_TAGS.items() if arm.startswith("FQ-g")}
    # γ in hundredths, rounded: 0.29 * 100 is 28.999... in binary
    assert [K.gamma_tag(g) for g in (0.29, 0.57, 0.58, 1.0)] == ["g29", "g57", "g58", "g100"]
    assert K.default_root() == REPO / "results" / "exp5" / "checkpoints"
    for bad in ("", ".", "..", "a/b", "a\\b", " x", "x.", "a:b"):
        with pytest.raises(ValueError, match="path component"):
            K.checkpoint_path(tmp_path, bad, "g0", 0.0, 0)
    with pytest.raises(ValueError, match="hundredths"):
        K.gamma_tag(0.955)
    with pytest.raises(ValueError):
        K.checkpoint_name(0.5, -1)
    with pytest.raises(ValueError, match="neither"):
        K.checkpoint_files([tmp_path / "missing.npz"])


def test_a_checkpoint_of_another_family_is_never_replaced(monkeypatch, tmp_path, capsys):
    """The layout has no family (other choices 6), so the calibration's two families
    (the jittery family and the clean control, other choices 12) meet at one
    path under one study: a run never replaces another family's checkpoint,
    even asked, and a sweep refuses before its first run; asked, a run replaces
    its own family's; one study per family keeps both."""
    _fly_nothing(monkeypatch)
    monkeypatch.setattr(K, "tree_state", lambda: TREE)

    def sweep(study, family, *extra):
        return ["sweep", "--study", study, "--family", family, "--root", str(tmp_path),
                "--gammas", "0", "0.9", "--seeds", "0", "--episodes", "1", "--eval-every",
                "1", "--val-episodes", "4", *extra]

    def files():
        return {p: p.read_bytes() for p in sorted((tmp_path / "calibration").rglob("*.*"))}

    assert CLI.main(sweep("calibration", C.FAMILY_JITTERY)) == 0
    path = K.checkpoint_path(tmp_path, "calibration", "g90", 0.9, 0)
    kept = files()
    assert len(kept) == 4
    capsys.readouterr()
    for extra in ((), ("--overwrite",)):
        with pytest.raises(SystemExit) as refused:
            CLI.main(sweep("calibration", C.FAMILY_CLEAN, *extra))
        assert refused.value.code == 2
        assert "cell family 'jittery'" in capsys.readouterr().err
    assert files() == kept
    with pytest.raises(FileExistsError, match="own --study"):
        T.train(T.pair_spec(0.9, 0, family=C.FAMILY_CLEAN), path, tree=TREE, overwrite=True)
    # asked, a run replaces its own family's checkpoint
    assert CLI.main(sweep("calibration", C.FAMILY_JITTERY, "--overwrite", "--lr", "5e-4")) == 0
    assert verify_checkpoint(path)["network"]["lr"] == 5e-4
    # one study per family keeps both
    assert CLI.main(sweep("calibration-clean", C.FAMILY_CLEAN)) == 0
    assert {(p.parts[-3], verify_checkpoint(p)["cell_family"])
            for p in K.checkpoint_files([tmp_path])} == {
        ("calibration", C.FAMILY_JITTERY), ("calibration-clean", C.FAMILY_CLEAN)}
    # the rule itself; a manifest that names no family protects none
    assert K.replace_refusal(tmp_path / "new" / "g0_s0.npz", "clean", overwrite=False) is None
    assert "exists" in K.replace_refusal(path, C.FAMILY_JITTERY, overwrite=False)
    assert K.replace_refusal(path, C.FAMILY_JITTERY, overwrite=True) is None
    assert "own --study" in K.replace_refusal(path, C.FAMILY_CLEAN, overwrite=False)
    for case in ("broken", "orphan"):
        (tmp_path / case).mkdir()
    broken = _copy_checkpoint(path, tmp_path / "broken")
    broken.with_suffix(".json").write_text("not a manifest", encoding="utf-8")
    assert K.replace_refusal(broken, C.FAMILY_CLEAN, overwrite=True) is None
    assert "exists" in K.replace_refusal(broken, C.FAMILY_CLEAN, overwrite=False)
    orphan = _copy_checkpoint(path, tmp_path / "orphan")
    orphan.unlink()      # a manifest alone still holds the path
    assert "exists" in K.replace_refusal(orphan, C.FAMILY_JITTERY, overwrite=False)
    assert "own --study" in K.replace_refusal(orphan, C.FAMILY_CLEAN, overwrite=True)


def test_the_tree_state_is_the_repositorys_head_and_its_source_trees(tmp_path):
    state = K.tree_state()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", "hermes", "experiments"],
                            cwd=REPO, capture_output=True, text=True).stdout
    assert state == K.TreeState(commit=head, dirty=bool(status.strip()))
    assert K.tree_state(tmp_path) == K.TreeState(commit=None, dirty=True)


def test_dirty_is_any_change_under_the_source_trees_untracked_files_included(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, check=True)

    git("init", "-q")
    git("config", "user.email", "u8b@example.invalid")
    git("config", "user.name", "u8b")
    for name, text in (("hermes/a.py", "a = 1\n"), ("experiments/b.py", "b = 1\n"),
                       ("docs/c.txt", "c\n"), (".gitignore", "*.log\n")):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True,
                          text=True, check=True).stdout.strip()
    assert K.tree_state(tmp_path) == K.TreeState(commit=head, dirty=False)
    (tmp_path / "docs" / "d.txt").write_text("outside the source trees\n", encoding="utf-8")
    (tmp_path / "experiments" / "run.log").write_text("ignored\n", encoding="utf-8")
    assert not K.tree_state(tmp_path).dirty
    (tmp_path / "experiments" / "new.py").write_text("untracked\n", encoding="utf-8")
    assert K.tree_state(tmp_path) == K.TreeState(commit=head, dirty=True)
    (tmp_path / "experiments" / "new.py").unlink()
    (tmp_path / "hermes" / "a.py").write_text("a = 2\n", encoding="utf-8")
    assert K.tree_state(tmp_path).dirty


# --------------------------------------------------------------------------- #
# The report on synthetic tables
# --------------------------------------------------------------------------- #

def _centred(r, n, sd):
    x = r.normal(0.0, sd, n) if sd else np.zeros(n)
    return x - x.mean()


def _table(means, *, spread=0.002, seed_effect=0.01, validation=None, refs=None, seeds=10,
           eps=0.01, rng=0, held_out_episodes=None):
    """Seed scores around ``means`` (each γ's mean exact), a common seed effect
    across γ, validation scores around ``validation`` (``means`` by default) and
    the references' per-episode returns (:func:`_refs`); ``held_out_episodes``
    per cell, when given, is what they flew on each of the two cells."""
    r = np.random.default_rng(rng)
    base = _centred(r, seeds, seed_effect)
    held = {g: dict(enumerate(m + base + _centred(r, seeds, spread))) for g, m in means.items()}
    validation = means if validation is None else validation
    val = {g: dict(enumerate(validation[g] + _centred(r, seeds, spread))) for g in means}
    return RP.SweepTable(epsilon=eps, held_out=held, validation=val,
                         references=refs if refs is not None else _refs(),
                         cells=("jit-n12-120", "jit-n12-180"),
                         held_out_episodes=held_out_episodes)


def _refs(fx=0.0, greedy=0.0, f=-0.03, hyb=0.0, fx_pair=0.0, committed=-0.02, n=400,
          noise=0.002, block=None, rng=1):
    """Per-episode held-out returns: a shared episode effect (common random
    numbers), each rule's exact mean, a little noise of its own; every ``block``
    episodes (a cell's, in an evaluation file) keep the exact means."""
    r = np.random.default_rng(rng)
    block = n if block is None else block

    def centred(sd):
        return np.concatenate([_centred(r, block, sd) for _ in range(n // block)])

    episode = centred(0.1)
    return {label: list(mean + episode + centred(noise))
            for label, mean in (("FX", fx), ("F", f), ("hyb", hyb), ("greedy_1", greedy),
                                ("fx_pair", fx_pair), ("committed_pair", committed))}


FLATISH = {0.0: 0.0, 0.25: 0.001, 0.5: 0.0, 0.75: -0.001, 0.9: 0.0, 0.99: 0.001}
#: Seed noise tight enough that every interval and test holds, so only the
#: margin ε decides.
TIGHT = dict(spread=0.0005, seed_effect=0.0005)


def test_rising_beating_the_best_fixed_rule_replaces_fx():
    v = RP.decide(_table({**FLATISH, 0.9: 0.05}, refs=_refs(greedy=0.004)))
    assert v.sanity["passed"] and v.sanity["best_rule"] == "greedy_1"
    assert v.best_gamma == 0.9 and v.outcome == RP.OUTCOME_RISING
    c = v.rising["contrasts"]["0.9"]
    assert c["gain"] == pytest.approx(0.05) and c["ci_low"] > 0
    assert c["p_value"] == pytest.approx(2 / 1024) and c["p_holm"] < 0.05
    assert v.fixed_rule["rule"] == "greedy_1" and v.fixed_rule["gain"] == pytest.approx(0.046)
    assert v.fixed_rule["beats"] and v.replace_fx
    assert v.trend is not None and tuple(v.trend["levels"]) == RP.GAMMAS
    assert not v.flat["flat"] and not v.flat["contrasts"]["0.9"]["equivalent"]
    json.dumps(v.to_json())


def test_rising_that_does_not_beat_the_best_fixed_rule_by_epsilon_keeps_fx():
    """``hyb`` is a fixed rule but not a one-step rule. 0.005 below the best γ, with
    the interval above 0 and p below 0.05, the margin alone keeps FX although
    the curve rises and the sanity check passes; 0.015 below, FX is replaced."""
    v = RP.decide(_table({**FLATISH, 0.9: 0.05}, refs=_refs(hyb=0.045), **TIGHT))
    assert v.sanity["passed"] and v.outcome == RP.OUTCOME_RISING
    f = v.fixed_rule
    assert f["rule"] == "hyb" and f["gain"] == pytest.approx(0.005)
    assert f["ci_low"] > 0 and f["p_value"] < 0.05 and not f["beats"]
    assert not v.replace_fx
    v = RP.decide(_table({**FLATISH, 0.9: 0.05}, refs=_refs(hyb=0.035), **TIGHT))
    assert v.fixed_rule["rule"] == "hyb" and v.fixed_rule["gain"] == pytest.approx(0.015)
    assert v.fixed_rule["beats"] and v.replace_fx


def test_the_fixed_rules_are_decision_5s_four_with_fx_the_arm_itself():
    """Decision 5 (a)'s fixed rules: the FX and F arms, FX's band with the planned
    order and "most devices, then least time", the best of them the one to
    beat. FX is the FX arm, not the slot's ``fx_pair`` (resolution R6), which is
    reported beside the rule with ``committed_pair``."""
    assert RP.FIXED_RULES == ("FX", "F", "hyb", "greedy_1")
    assert RP.ONE_STEP_RULES == ("FX", "greedy_1")
    assert RP.SLOT_REFERENCES == ("fx_pair", "committed_pair")
    assert RP.GAMMAS == (0.0, 0.25, 0.5, 0.75, 0.9, 0.99)
    # the FX arm 0.007 below the best γ and γ = 0 within ε of it: the curve
    # rises, but FX stays, however low the slot's fx_pair
    lifted = {g: 0.08 + m for g, m in {**FLATISH, 0.9: 0.012}.items()}
    v = RP.decide(_table(lifted, refs=_refs(fx=0.085, fx_pair=0.0), **TIGHT))
    assert v.sanity["passed"] and v.outcome == RP.OUTCOME_RISING
    assert v.fixed_rule["rule"] == "FX" and v.fixed_rule["rule_mean"] == pytest.approx(0.085)
    assert v.fixed_rule["gain"] == pytest.approx(0.007) and not v.fixed_rule["beats"]
    assert not v.replace_fx and v.references["fx_pair"] == pytest.approx(0.0)
    # F the best of the four, 0.005 below the best γ: FX stays
    v = RP.decide(_table({**FLATISH, 0.9: 0.05}, refs=_refs(f=0.045), **TIGHT))
    assert v.outcome == RP.OUTCOME_RISING and v.fixed_rule["rule"] == "F"
    assert v.fixed_rule["gain"] == pytest.approx(0.005) and not v.fixed_rule["beats"]
    assert not v.replace_fx


def test_flat_is_every_gamma_within_epsilon_by_tost():
    v = RP.decide(_table(FLATISH))
    assert v.outcome == RP.OUTCOME_FLAT and not v.replace_fx
    assert all(c["equivalent"] and -0.01 < c["ci_low"] <= c["ci_high"] < 0.01
               for c in v.flat["contrasts"].values())
    assert not v.rising["rising"]
    # one γ out of the margin, though not significantly better, is no longer flat
    v = RP.decide(_table({**FLATISH, 0.5: 0.012}, spread=0.02))
    assert v.outcome == RP.OUTCOME_INCONCLUSIVE and not v.flat["contrasts"]["0.5"]["equivalent"]
    # the margin is ε: a γ 1.5 ε above γ = 0, tightly, is within 2 ε but not ε (and
    # the best γ on validation, 0.25, does not rise)
    v = RP.decide(_table({**FLATISH, 0.5: 0.015}, validation={**FLATISH, 0.25: 0.003},
                         **TIGHT))
    c = v.flat["contrasts"]["0.5"]
    assert 0.01 < c["ci_low"] <= c["ci_high"] < 0.02 and not c["equivalent"]
    assert v.best_gamma == 0.25 and v.outcome == RP.OUTCOME_INCONCLUSIVE


def test_inconclusive_is_neither_and_has_no_second_look():
    v = RP.decide(_table({**FLATISH, 0.9: 0.02}, spread=0.05))
    assert v.outcome == RP.OUTCOME_INCONCLUSIVE and not v.replace_fx
    c = v.rising["contrasts"]["0.9"]
    assert c["gain"] == pytest.approx(0.02) and c["ci_low"] < 0
    assert not v.flat["flat"]


def test_beating_the_fixed_rules_without_rising_keeps_fx():
    """FX is replaced only if the curve rises AND the best γ beats the best fixed
    rule: here every γ beats every fixed rule by 0.05, but the curve is flat."""
    v = RP.decide(_table({g: m + 0.05 for g, m in FLATISH.items()}))
    assert v.outcome == RP.OUTCOME_FLAT and v.fixed_rule["beats"]
    assert v.fixed_rule["gain"] >= 0.05 - 0.001 and not v.replace_fx


def test_the_claim_rule_needs_the_margin_the_interval_and_the_p_value():
    """"Beats by ε" (rising, the best fixed rule, greedy_1's flag): the gain at
    least ε, its interval's low end above 0, p below alpha."""
    assert RP._claim(0.02, 0.001, 0.01, 0.01, 0.05)
    assert RP._claim(0.01, 0.001, 0.01, 0.01, 0.05)
    assert not RP._claim(0.0099, 0.001, 0.01, 0.01, 0.05)
    assert not RP._claim(0.02, 0.0, 0.01, 0.01, 0.05)
    assert not RP._claim(0.02, 0.001, 0.05, 0.01, 0.05)


def test_the_sanity_check_comes_first_and_a_failure_reads_no_curve():
    """γ = 0 below the best one-step rule less ε: the curve is not read, even
    where γ = 0.9 would rise."""
    v = RP.decide(_table({**FLATISH, 0.0: -0.05, 0.9: 0.05},
                         refs=_refs(fx=-0.01, greedy=0.0)))
    assert v.outcome == RP.OUTCOME_SANITY_FAILED and not v.replace_fx
    assert v.rising is None and v.flat is None and v.fixed_rule is None
    s = v.sanity
    assert (s["passed"], s["best_rule"]) == (False, "greedy_1")
    assert s["floor"] == pytest.approx(-0.01) and s["shortfall"] == pytest.approx(0.05)
    assert s["seeds_below_floor"] == 10
    assert "FAILED" in RP.format_verdict(v)


def test_the_sanity_floor_is_the_best_one_step_rule_less_epsilon_inclusive():
    exact = {**FLATISH, 0.0: -0.01}
    passed = RP.decide(_table(exact, spread=0.0, seed_effect=0.0, refs=_refs(greedy=0.0)))
    assert passed.sanity["gamma0_mean"] == passed.sanity["floor"] == -0.01
    assert passed.sanity["passed"]
    below = RP.decide(_table({**exact, 0.0: -0.0100001}, spread=0.0, seed_effect=0.0,
                             refs=_refs(greedy=0.0)))
    assert not below.sanity["passed"]
    # F and hyb are no one-step rules: a strong hyb never fails the sanity check
    strong = RP.decide(_table(FLATISH, refs=_refs(hyb=0.2, f=0.3)))
    assert strong.sanity["passed"] and strong.sanity["rules"].keys() == {"FX", "greedy_1"}


def test_the_best_gamma_is_picked_on_validation_not_on_the_held_out_runs():
    held = {**FLATISH, 0.99: 0.05}
    on_val = RP.decide(_table(held, validation={**FLATISH, 0.5: 0.05}))
    assert on_val.best_gamma == 0.5 and on_val.outcome == RP.OUTCOME_INCONCLUSIVE
    assert on_val.rising["contrasts"]["0.99"]["beats_gamma0"]
    assert RP.decide(_table(held)).outcome == RP.OUTCOME_RISING
    # ties go to the lower γ
    tie = RP.decide(_table(FLATISH, spread=0.0, validation={g: 0.0 for g in FLATISH}))
    assert tie.best_gamma == 0.25


def test_holm_over_the_contrasts_decides_rising():
    """Six seeds: every γ > 0 beats γ = 0 in every seed, so each exact Wilcoxon p is
    1/32 < 0.05, but Holm over the five contrasts lifts each to 5/32."""
    v = RP.decide(_table({g: (0.0 if g == 0.0 else 0.05) for g in FLATISH}, seeds=6))
    for c in v.rising["contrasts"].values():
        assert c["p_value"] == pytest.approx(1 / 32)
        assert c["p_holm"] == pytest.approx(5 / 32) and not c["beats_gamma0"]
    assert v.outcome == RP.OUTCOME_INCONCLUSIVE


def test_greedy_1_beating_fx_by_epsilon_is_flagged_for_the_user():
    flagged = RP.decide(_table(FLATISH, refs=_refs(greedy=0.02, noise=0.01)))
    assert flagged.greedy_1_flag and flagged.greedy_1["gain"] == pytest.approx(0.02)
    assert flagged.greedy_1["ci_low"] > 0 and flagged.greedy_1["episodes"] == 400
    assert "for the user to decide" in RP.format_verdict(flagged)
    assert not flagged.replace_fx
    small = RP.decide(_table(FLATISH, refs=_refs(greedy=0.005, noise=0.01)))
    assert small.greedy_1["p_value"] < 0.05 and not small.greedy_1_flag
    noisy = RP.decide(_table(FLATISH, refs=_refs(greedy=0.02, noise=0.6)))
    assert noisy.greedy_1["ci_low"] < 0 and not noisy.greedy_1_flag


def test_greedy_1s_flag_reads_the_fx_arm_not_the_slots_fx_pair():
    """Resolution R6 for critic A2's flag: greedy_1 is read against the FX arm."""
    lifted = {g: 0.02 + m for g, m in FLATISH.items()}
    near = RP.decide(_table(lifted, refs=_refs(greedy=0.025, fx=0.02, fx_pair=0.0, noise=0.01)))
    assert near.sanity["passed"] and near.greedy_1["gain"] == pytest.approx(0.005)
    assert not near.greedy_1_flag
    far = RP.decide(_table(lifted, refs=_refs(greedy=0.025, fx=0.0, fx_pair=0.02, noise=0.01)))
    assert far.greedy_1["gain"] == pytest.approx(0.025) and far.greedy_1_flag


def test_the_trend_and_the_stack_checks_picks_are_reported_alongside():
    table = _table({**FLATISH, 0.9: 0.05}, validation={**FLATISH, 0.9: 0.05})
    v = RP.decide(table)
    assert v.trend["alternative"] == "increasing" and -1.0 <= v.trend["rho"] <= 1.0
    ranked = sorted(table.seeds, key=lambda s: (table.validation[0.9][s], s))
    assert v.stack_check == {"best_gamma": 0.9, "best_gamma_seed": ranked[4],
                             "gamma0_seed": RP.median_seed(table, 0.0)}
    calibration = RP.decide(_table({0.0: 0.0, 0.9: 0.0}, seeds=3))
    assert calibration.trend is None and calibration.gammas == (0.0, 0.9)
    assert set(v.references) >= set(RP.FIXED_RULES) | set(RP.SLOT_REFERENCES)


def test_a_table_refuses_what_is_not_a_sweep():
    with pytest.raises(ValueError, match="γ = 0"):
        _table({0.5: 0.0, 0.9: 0.0})
    with pytest.raises(ValueError, match="at least one γ > 0"):
        _table({0.0: 0.0})
    with pytest.raises(ValueError, match="two training seeds"):
        _table(FLATISH, seeds=1)
    table = _table(FLATISH)
    held = dict(table.held_out)
    held[0.5] = {s: v for s, v in held[0.5].items() if s != 3}
    with pytest.raises(ValueError, match="paired unit"):
        RP.SweepTable(epsilon=0.01, held_out=held, validation=table.validation,
                      references=table.references)
    with pytest.raises(ValueError, match="epsilon"):
        RP.SweepTable(epsilon=0.0, held_out=table.held_out, validation=table.validation,
                      references=table.references)
    with pytest.raises(ValueError, match="paired episode by episode"):
        RP.SweepTable(epsilon=0.01, held_out=table.held_out, validation=table.validation,
                      references=dict(table.references, F=[0.0]))
    with pytest.raises(ValueError, match="no held-out returns"):
        RP.decide(RP.SweepTable(epsilon=0.01, held_out=table.held_out,
                                validation=table.validation,
                                references={k: v for k, v in table.references.items()
                                            if k != "hyb"}))
    # refused before a number is read: a table whose sanity check fails never
    # reaches the fixed rules
    failing = _table({**FLATISH, 0.0: -0.05},
                     refs={k: v for k, v in _refs().items() if k not in ("F", "hyb")})
    with pytest.raises(ValueError, match=r"no held-out returns of \['F', 'hyb'\]"):
        RP.decide(failing)


def test_epsilon_reads_the_validation_headroom_of_the_cells_read():
    report = {"stream": C.VAL_STREAM, "cells": {"jit-n12-120": {"headroom": 0.0911},
                                                "jit-n12-180": {"headroom": 0.0229},
                                                "jit-n6-45": {"headroom": 0.5}}}
    assert RP.epsilon_from_headroom_report(report, ["jit-n12-120", "jit-n12-180"]) == 0.01
    assert RP.epsilon_from_headroom_report(report, ["jit-n6-45"]) == pytest.approx(0.05)
    # the score is the mean over its cells, so its headroom is theirs: (0.5 + 0.0911) / 2
    assert RP.epsilon_from_headroom_report(report, ["jit-n6-45", "jit-n12-120"]) == (
        pytest.approx(0.1 * (0.5 + 0.0911) / 2))
    for stream in (C.HELDOUT_STREAM, None):
        with pytest.raises(ValueError, match="validation"):
            RP.epsilon_from_headroom_report(dict(report, stream=stream), ["jit-n6-45"])
    with pytest.raises(ValueError, match="no cells"):
        RP.epsilon_from_headroom_report(report, ["cln-n12-120"])


# --------------------------------------------------------------------------- #
# The pre-registered grid (the orchestrator's resolution R26)
# --------------------------------------------------------------------------- #

RISING = {**FLATISH, 0.9: 0.05}
GRID = "{0, 0.25, 0.5, 0.75, 0.9, 0.99}"


def _grid_table(means=RISING, *, seeds=10, episodes=1000):
    """A synthetic table on decision 5 (a)'s grid by default: ``means`` by γ, at
    ``seeds`` per γ, the references flying ``episodes`` held-out episodes on each
    of the two cells read (None: the same 1,000, their count not recorded)."""
    n = 1000 if episodes is None else episodes
    return _table(means, seeds=seeds, refs=_refs(greedy=0.004, n=2 * n, block=n),
                  held_out_episodes=episodes)


def _label_aside(verdict):
    """A verdict's JSON without the grid's label."""
    return {k: v for k, v in verdict.to_json().items()
            if k not in ("preregistered", "preregistration")}


def test_decision_5s_grid_is_labelled_pre_registered():
    """Resolution R26: six γ, 10 seeds each, on 1,000 shared held-out episodes per
    cell (the evaluate command's default) is decision 5 (a)'s grid; the verdict
    says so in its JSON and on its second line. The same table with its
    held-out count not recorded is not shown to be on it, and is decided alike."""
    assert (RP.GAMMAS, RP.SEEDS_PER_GAMMA, RP.HELD_OUT_EPISODES) == (
        (0.0, 0.25, 0.5, 0.75, 0.9, 0.99), 10, 1000)
    assert CLI.parser().parse_args(["evaluate", "--out", "e.json"]).episodes == 1000
    table = _grid_table()
    grid = {"gammas": list(RP.GAMMAS), "seeds_per_gamma": 10, "held_out_episodes": 1000}
    assert RP.preregistration(table) == {"reasons": [], "grid": grid, "sweep": grid}
    v = RP.decide(table)
    assert v.preregistered and v.preregistration == RP.preregistration(table)
    assert (v.outcome, v.best_gamma, v.replace_fx) == (RP.OUTCOME_RISING, 0.9, True)
    saved = json.loads(json.dumps(v.to_json()))
    assert saved["preregistered"] is True and saved["preregistration"]["reasons"] == []
    assert RP.format_verdict(v).splitlines()[1] == (
        f"Grid: pre-registered (decision 5 (a)): γ ∈ {GRID}, 10 seeds per γ, 1000 held-out "
        f"episodes per cell")
    unrecorded = dataclasses.replace(table, held_out_episodes=None)
    w = RP.decide(unrecorded)
    assert not w.preregistered and w.preregistration["reasons"] == [
        "held-out episodes per cell not recorded (the grid has 1000)"]
    assert _label_aside(w) == _label_aside(v)


#: The verdict (outcome, best γ, replace FX) the rule gives a rising sweep off the grid.
RISES = (RP.OUTCOME_RISING, 0.9, True)


@pytest.mark.parametrize("change, reasons, decided", [
    (dict(seeds=3), ["3 seeds per γ, not 10"], (RP.OUTCOME_INCONCLUSIVE, 0.9, False)),
    # Holm over five contrasts: rising is out of reach below 8 seeds, not at 8 or 9
    (dict(seeds=7), ["7 seeds per γ, not 10"], (RP.OUTCOME_INCONCLUSIVE, 0.9, False)),
    (dict(seeds=8), ["8 seeds per γ, not 10"], RISES),
    (dict(seeds=9), ["9 seeds per γ, not 10"], RISES),
    (dict(seeds=12), ["12 seeds per γ, not 10"], RISES),
    (dict(means={g: m for g, m in RISING.items() if g != 0.75}),
     [f"γ ∈ {{0, 0.25, 0.5, 0.9, 0.99}}, not {GRID}"], RISES),
    (dict(means={(0.3 if g == 0.25 else g): m for g, m in RISING.items()}),
     [f"γ ∈ {{0, 0.3, 0.5, 0.75, 0.9, 0.99}}, not {GRID}"], RISES),
    (dict(means={**RISING, 0.95: 0.0}),
     [f"γ ∈ {{0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99}}, not {GRID}"], RISES),
    (dict(episodes=999), ["999 held-out episodes per cell, not 1000"], RISES),
    (dict(episodes=1001), ["1001 held-out episodes per cell, not 1000"], RISES),
    (dict(means={0.0: 0.0, 0.9: 0.0}, seeds=3, episodes=4),
     [f"γ ∈ {{0, 0.9}}, not {GRID}", "3 seeds per γ, not 10",
      "4 held-out episodes per cell, not 1000"], (RP.OUTCOME_FLAT, 0.9, False)),
], ids=["3-seeds", "7-seeds", "8-seeds", "9-seeds", "12-seeds", "missing-gamma",
        "off-grid-gamma", "extra-gamma", "short-held-out", "long-held-out", "calibration"])
def test_a_sweep_off_decision_5s_grid_is_labelled_and_decided_as_on_it(monkeypatch, change,
                                                                       reasons, decided):
    """Resolution R26: a sweep whose γ set, seeds per γ or held-out count is not
    decision 5 (a)'s, with fewer or more than the grid (the repair round's C-T1),
    is labelled not pre-registered with each reason, in the verdict's JSON and on
    its second line; it is never refused (the calibration, 2 γ x 3 seeds, is read
    by this code), and no step reads the label or the grid's settings: the
    verdict is the one the rule gives these returns (``decided``: rising, best γ
    0.9, FX replaced, wherever the seeds allow it; C-T2), and, label aside, the
    one it gives under the grid's label."""
    table = _grid_table(**change)
    v = RP.decide(table)
    assert not v.preregistered and v.preregistration["reasons"] == reasons
    assert (v.outcome, v.best_gamma, v.replace_fx) == decided
    assert v.preregistration["sweep"] == {"gammas": list(table.gammas),
                                          "seeds_per_gamma": len(table.seeds),
                                          "held_out_episodes": table.held_out_episodes}
    saved = json.loads(json.dumps(v.to_json()))
    assert saved["preregistered"] is False and saved["preregistration"]["reasons"] == reasons
    lines = RP.format_verdict(v).splitlines()
    assert lines[1] == "Grid: NOT pre-registered (decision 5 (a)): " + "; ".join(reasons)
    grid = RP.preregistration(_grid_table())
    monkeypatch.setattr(RP, "preregistration", lambda _table: grid)
    as_grid = RP.decide(table)
    assert as_grid.preregistered and _label_aside(as_grid) == _label_aside(v)
    assert RP.format_verdict(as_grid).splitlines()[2:] == lines[2:]


def test_a_table_holds_the_held_out_count_its_references_flew():
    """The count the label reads is the references' own: per cell read, ≥ 1."""
    assert _grid_table(episodes=7).held_out_episodes == 7
    with pytest.raises(ValueError, match="not 1000 on each of the 2 cell"):
        _table(RISING, refs=_refs(n=400), held_out_episodes=1000)
    for bad in (0, True, 1000.0):
        with pytest.raises(ValueError, match="count per cell"):
            _table(RISING, refs=_refs(n=2000, block=1000), held_out_episodes=bad)


# --------------------------------------------------------------------------- #
# The table from an evaluation file
# --------------------------------------------------------------------------- #

CELLS_READ = ("jit-n12-120", "jit-n12-180")
FLOWN = CELLS_READ + ("jit-n6-90",)


def _manifest(gamma, seed, *, revision=0, purpose="trained", kind="pair_q", val=0.0,
              n6=None):
    """A sweep checkpoint's manifest, as far as the report reads it: its kept
    validation (episode 2000) is not its last (3000), and the N = 6 control
    validates at ``n6`` (``val`` by default)."""
    n6 = val if n6 is None else n6
    curve = [{"episode": e, "score": val,
              "cells": {**{c: val + e / 1e6 for c in CELLS_READ}, "jit-n6-90": n6 + e / 1e6}}
             for e in (1000, 2000, 3000)]
    return {"kind": kind, "purpose": purpose, "learner_revision": revision, "gamma": gamma,
            "cell_family": "jittery", "cell_family_sha256": "0" * 64,
            "reward": T.TRAINING_REWARD.to_json(),
            "training": {"spec": T.pair_spec(gamma, seed).to_json()},
            "seeds": {"run": seed}, "episodes_trained": 2000, "validation": curve,
            "sha256": f"{seed:02d}{int(gamma * 100):03d}" + "0" * 59}


def _evaluation(table, *, episodes=4, n6=None, **manifest_kw):
    """An evaluation file whose seed means are ``table``'s on the cells read (one
    more on the N = 6 control), and whose references are its per-episode returns,
    ``episodes`` per cell; ``n6`` maps γ to its N = 6 validation score."""
    refs = {label: {"returns": {c: list(v[i * episodes:(i + 1) * episodes])
                                for i, c in enumerate(FLOWN)},
                    "decisions": {c: [[1, 2]] * episodes for c in FLOWN}}
            for label, v in table.references.items()}
    checkpoints = []
    for g, by in table.held_out.items():
        for s, score in by.items():
            m = _manifest(g, s, val=table.validation[g][s],
                          n6=None if n6 is None else n6[g], **manifest_kw)
            checkpoints.append({"path": f"g{g}_s{s}.npz", "manifest": m,
                                "returns": {c: [score + (0.0 if c in CELLS_READ else 1.0)]
                                            * episodes for c in FLOWN},
                                "decisions": {c: [[1]] * episodes for c in FLOWN}})
    return {"format": RP.EVALUATION_FORMAT, "stream": C.HELDOUT_STREAM, "episodes": episodes,
            "start": 0, "cells": list(FLOWN), "references": refs, "checkpoints": checkpoints}


def test_the_table_from_an_evaluation_file_reads_seed_means_on_the_cells_read():
    """The held-out seed means, the kept validation that picks the best γ and the
    curves are all read on the cells read; the N = 6 control, where γ = 0.5
    validates best here, is reported beside them and decides nothing."""
    direct = _table({**FLATISH, 0.9: 0.05}, refs=_refs(greedy=0.004, n=12, block=4))
    n6 = {g: (1.0 if g == 0.5 else -1.0) for g in direct.gammas}
    table = RP.sweep_table(_evaluation(direct, n6=n6), cells=CELLS_READ, epsilon=0.01)
    assert table.cells == CELLS_READ and table.seeds == direct.seeds
    for g in direct.gammas:
        assert table.column(g) == pytest.approx(direct.column(g))
        # the kept validation entry (episode 2000) on the cells read
        assert [table.validation[g][s] for s in table.seeds] == pytest.approx(
            [direct.validation[g][s] + 0.002 for s in direct.seeds])
    assert table.references["FX"] == pytest.approx(direct.references["FX"][:8])
    assert table.decisions["jit-n6-90"]["FX"] == 0.5
    assert table.per_cell["jit-n6-90"]["0.9"] == pytest.approx(direct.mean(0.9) + 1.0)
    assert table.per_cell["jit-n12-120"]["0.9"] == pytest.approx(direct.mean(0.9))
    assert table.per_cell["jit-n6-90"]["FX"] == pytest.approx(
        statistics.fmean(direct.references["FX"][8:12]))
    for g in (0.5, 0.9):
        val_mean = statistics.fmean(direct.validation[g].values())
        assert [tuple(p) for p in table.curves[g]] == [
            (e, pytest.approx(val_mean + e / 1e6), 10) for e in (1000, 2000, 3000)]
    v = RP.decide(table)
    assert v.outcome == RP.OUTCOME_RISING and v.best_gamma == 0.9


@pytest.mark.parametrize("change, message", [
    (dict(revision=2), "revision"),
    (dict(purpose="bootstrap"), "trained"),
    (dict(kind="chen_dqn"), "pair score"),
])
def test_the_table_refuses_a_checkpoint_the_sweep_may_not_read(change, message):
    with pytest.raises(ValueError, match=message):
        RP.sweep_table(_evaluation(_table(FLATISH), **change), cells=CELLS_READ, epsilon=0.01)


def test_the_table_refuses_two_learners_in_one_sweep():
    def evaluation(change):
        data = _evaluation(_table(FLATISH))
        change(data)
        return data

    cases = [
        (lambda d: d["checkpoints"][3]["manifest"].update(learner_revision=1),
         "learner revisions"),
        (lambda d: d["checkpoints"][3]["manifest"]["training"]["spec"]["network"].update(
            lr=5e-4), "training specs"),
        (lambda d: d["checkpoints"][3]["manifest"].update(cell_family="clean"),
         "cell families"),
        (lambda d: d["checkpoints"][3]["manifest"].update(
            reward=T.HAND_TRAINING_REWARD.to_json()), "rewards"),
        (lambda d: d["checkpoints"].append(d["checkpoints"][0]), "two checkpoints"),
        (lambda d: d["references"]["FX"]["returns"]["jit-n12-120"].pop(), "held-out episode"),
        (lambda d: d.update(stream=C.VAL_STREAM), "held-out runs"),
    ]
    for change, message in cases:
        with pytest.raises(ValueError, match=message):
            RP.sweep_table(evaluation(change), cells=CELLS_READ, epsilon=0.01)
    with pytest.raises(ValueError, match="flew"):
        RP.sweep_table(_evaluation(_table(FLATISH)), cells=("cln-n12-120",), epsilon=0.01)


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #

def _cli_train(tmp_path, *extra):
    return ["train", "--study", "smoke", "--gamma", "0.9", "--seed", "0", "--root",
            str(tmp_path), "--episodes", "2", "--eval-every", "2", "--val-episodes", "4",
            *extra]


def test_the_cli_refuses_a_dirty_tree_unless_allowed_and_never_overwrites(monkeypatch,
                                                                         tmp_path, capsys):
    monkeypatch.setattr(K, "tree_state", lambda: K.TreeState(commit=TREE.commit, dirty=True))
    with pytest.raises(SystemExit) as refused:
        CLI.main(_cli_train(tmp_path))
    assert refused.value.code == 2 and "dirty" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())
    assert CLI.main(_cli_train(tmp_path, "--allow-dirty")) == 0
    path = tmp_path / "smoke" / "g90" / "g0.9_s0.npz"
    m = verify_checkpoint(path)
    assert (m["dirty"], m["trainer_commit"], m["episodes_trained"]) == (True, TREE.commit, 2)
    assert m["training"]["spec"]["cells"] == [c.name for c in C.FAMILIES[C.FAMILY_JITTERY]]
    capsys.readouterr()
    with pytest.raises(SystemExit):
        CLI.main(_cli_train(tmp_path, "--allow-dirty"))
    assert "exists" in capsys.readouterr().err
    monkeypatch.setattr(K, "tree_state", lambda: TREE)
    with pytest.raises(SystemExit):
        CLI.main(_cli_train(tmp_path, "--c-t", "0.3"))      # 5.7's grid names its tag
    assert "own --tag" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        CLI.main(_cli_train(tmp_path, "--kind", "chen_dqn", "--reference-episodes", "5"))
    assert "reference" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        CLI.main(["sweep", "--study", "s", "--gammas", "0", "0", "--seeds", "1"])
    assert "once" in capsys.readouterr().err


def test_the_cli_sweep_refuses_a_dirty_tree_before_its_first_run(monkeypatch, tmp_path,
                                                                  capsys):
    """Critic B9 on the command that runs Study 5.5's 60 trainings."""
    _fly_nothing(monkeypatch)
    sweep = ["sweep", "--study", "smoke", "--root", str(tmp_path), "--gammas", "0", "0.9",
             "--seeds", "0", "--episodes", "1", "--eval-every", "1", "--val-episodes", "4"]
    monkeypatch.setattr(K, "tree_state", lambda: K.TreeState(commit=TREE.commit, dirty=True))
    with pytest.raises(SystemExit) as refused:
        CLI.main(sweep)
    assert refused.value.code == 2 and "dirty" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())
    assert CLI.main(sweep + ["--allow-dirty"]) == 0
    for tag, gamma in (("g0", 0.0), ("g90", 0.9)):
        m = verify_checkpoint(K.checkpoint_path(tmp_path, "smoke", tag, gamma, 0))
        assert (m["gamma"], m["dirty"], m["trainer_commit"]) == (gamma, True, TREE.commit)


def test_the_cli_trains_with_the_learner_settings_it_is_given(monkeypatch, tmp_path):
    """``--lr``, the behaviour's ε and reference phase, and ``--no-phase`` reach the
    run as its manifest records them; E3 keeps its own behaviour (no reference
    phase) under the same flags (Chen's settings are the user's open choice)."""
    _fly_nothing(monkeypatch)
    monkeypatch.setattr(K, "tree_state", lambda: TREE)
    assert CLI.main(_cli_train(tmp_path, "--lr", "5e-4", "--epsilon-start", "0.2",
                               "--epsilon-end", "0.02", "--reference-episodes", "1",
                               "--no-phase")) == 0
    m = verify_checkpoint(tmp_path / "smoke" / "g90" / "g0.9_s0.npz")
    spec = m["training"]["spec"]
    assert m["network"]["lr"] == spec["network"]["lr"] == 5e-4
    assert spec["learner"]["behaviour"] == {"reference_episodes": 1, "epsilon_start": 0.2,
                                            "epsilon_end": 0.02, "decay_fraction": 0.5}
    assert spec["phase"] is False and m["schema"]["dim"] == 24
    assert m["schema"] == PairFeatureSchema(tuple(CLASSES), phase=False).to_json()
    assert CLI.main(_cli_train(tmp_path, "--kind", "chen_dqn", "--lr", "5e-4",
                               "--epsilon-start", "1.0")) == 0
    m = verify_checkpoint(tmp_path / "smoke" / "e3" / "g0.9_s0.npz")
    assert (m["kind"], m["network"]["lr"]) == ("chen_dqn", 5e-4)
    assert m["training"]["spec"]["learner"]["behaviour"] == {
        "reference_episodes": 0, "epsilon_start": 1.0, "epsilon_end": 0.05,
        "decay_fraction": 0.5}
    assert m["reward"] == R.RewardSpec(kind=R.REWARD_BYTES, c_t=0.0, c_cov=0.0,
                                       expected_availability=True).to_json()
    # the training reward: Study 5.7's weights under their own tag, and F·hand
    assert CLI.main(_cli_train(tmp_path, "--c-t", "0.3", "--c-cov", "4", "--tag",
                               "ct0.3-cc4")) == 0
    m = verify_checkpoint(tmp_path / "smoke" / "ct0.3-cc4" / "g0.9_s0.npz")
    assert m["reward"] == R.RewardSpec(c_t=0.3, c_cov=4.0, expected_availability=True).to_json()
    assert CLI.main(_cli_train(tmp_path, "--reward", "hand")) == 0
    m = verify_checkpoint(tmp_path / "smoke" / "hand" / "g0.9_s0.npz")
    assert m["reward"] == R.RewardSpec(kind=R.REWARD_HAND, c_t=0.0, c_cov=0.0,
                                       expected_availability=True).to_json()


def test_the_cli_names_each_checkpoint_by_its_arms_tag():
    """The layout's tag is the arm's (``experiments.exp4.driver.CHECKPOINT_TAGS``):
    ``g<100 γ>`` for Study 5.5's sweep, ``hand`` for F·hand, ``e3`` for E3; Study
    5.7's other weights need a tag of their own; and a learned arm's tag, given,
    takes only a run the runner would fly under it."""
    plain = SimpleNamespace(tag=None)
    assert CLI._tag(plain, T.pair_spec(0.9, 0)) == "g90" == CHECKPOINT_TAGS["FQ-g90"]
    assert CLI._tag(plain, T.pair_spec(0.0, 0)) == "g0"
    assert CLI._tag(plain, T.pair_spec(0.9, 0, reward=T.HAND_TRAINING_REWARD)) == (
        CHECKPOINT_TAGS["FQ-hand"])
    assert CLI._tag(plain, T.e3_spec(0.9, 0)) == CHECKPOINT_TAGS["E3"]
    other = T.pair_spec(0.9, 0, reward=dataclasses.replace(T.TRAINING_REWARD, c_cov=4.0))
    with pytest.raises(ValueError, match="own --tag"):
        CLI._tag(plain, other)
    assert CLI._tag(SimpleNamespace(tag="ct0.1-cc4"), other) == "ct0.1-cc4"
    # resolution R23: an ablation's score goes under its arm's tag, and no other
    for ablation, arm in (("dwell", "FQ-dwell"), ("cov", "FQ-cov")):
        spec = T.pair_spec(0.9, 0, plan_score_params=T.ABLATIONS[ablation])
        assert CLI._tag(SimpleNamespace(tag=None, ablation=ablation), spec) == (
            CHECKPOINT_TAGS[arm])
        assert CLI._tag(SimpleNamespace(tag=ablation, ablation=ablation), spec) == ablation
        with pytest.raises(ValueError, match=f"arm {arm}'s score.*contradicts"):
            CLI._tag(SimpleNamespace(tag="g90", ablation=ablation), spec)
        for reward in (T.HAND_TRAINING_REWARD, other.reward):
            with pytest.raises(ValueError, match="derived reward at its own weights"):
                CLI._tag(SimpleNamespace(tag=None, ablation=ablation),
                         dataclasses.replace(spec, reward=reward))
    # resolution R24: every tag the command line gives by default flies its reward
    for spec in (T.pair_spec(0.25, 0), T.pair_spec(0.9, 0, reward=T.HAND_TRAINING_REWARD),
                 T.e3_spec(0.9, 0)):
        assert K.tag_reward(CLI._tag(plain, spec)) == spec.reward.kind
    # the repair round's A-2 and A-4: a learned arm's tag, given, takes only a run its
    # arm flies (checkpoints.tag_mismatches, the runner's own rules, and an ablation's
    # plan); main, at any weights, and a tag no arm flies stay open
    dwell = T.pair_spec(0.9, 0, plan_score_params=FQ_DWELL_SCORE)
    for tag, spec, why in (
            ("g90", other, "it trained on the derived reward at c_t 0.1, c_cov 4, and tag "
                           "'g90' flies it at decision 4 (a)'s weights, c_t 0.1 and c_cov 1"),
            ("g50", T.pair_spec(0.9, 0), "its γ is 0.9, which flies as tag 'g90', not 'g50'"),
            ("g90", T.pair_spec(0.9, 0, reward=T.HAND_TRAINING_REWARD),
             "it trained on the 'hand' reward, and tag 'g90' flies the 'derived' reward's "
             "score"),
            ("main", T.e3_spec(0.9, 0), "it trained on the 'bytes' reward, and tag 'main' "
                                        "flies the 'derived' reward's score"),
            ("dwell", T.pair_spec(0.9, 0), "the arm flies the plan score settings "
                                           "{'dwell_in_delta': False}, and this run trains "
                                           "under dwell_in_delta True: give --ablation dwell "
                                           "(or those settings in --plan-score-params)"),
            ("cov", T.pair_spec(0.9, 0, plan_score_params={"c_cov_per_device": 0.0}),
             "the arm flies the plan score settings {'c_cov_per_device': 0.0, 'c_link': "
             "0.0}, and this run trains under c_link None: give --ablation cov"),
            ("dwell", dataclasses.replace(dwell, reward=other.reward), "c_t 0.1, c_cov 4")):
        arm = {t: a for a, t in CHECKPOINT_TAGS.items()}[tag]
        with pytest.raises(ValueError) as refused:
            CLI._tag(SimpleNamespace(tag=tag), spec)
        assert str(refused.value).startswith(
            f"--tag {tag} names arm {arm}'s checkpoint, which the runner flies only when it "
            f"is that arm's (resolutions R23 and R24): "), tag
        assert why in str(refused.value), tag
    for tag, spec in (("main", other), ("g90", T.pair_spec(0.9, 0)), ("dwell", dwell),
                      ("dwell", T.pair_spec(0.9, 0, plan_score_params=dict(FQ_DWELL_SCORE,
                                                                           c_energy=0.0))),
                      ("cov", T.pair_spec(0.9, 0, plan_score_params=F_COV_SCORE)),
                      ("hand", T.pair_spec(0.9, 0, reward=T.HAND_TRAINING_REWARD)),
                      ("e3", T.e3_spec(0.9, 0)),
                      ("ct0.1-cc4", T.pair_spec(0.9, 0, reward=T.HAND_TRAINING_REWARD))):
        assert CLI._tag(SimpleNamespace(tag=tag), spec) == tag


def test_the_cli_trains_each_ablation_under_its_arms_plan_and_tag(monkeypatch, tmp_path,
                                                                   capsys):
    """Resolution R23: ``--ablation dwell|cov`` trains under the driver's own
    ``FQ_DWELL_SCORE`` or ``F_COV_SCORE`` and saves under the arm's tag;
    ``--plan-score-params``, in the runner's format, gives a pilot's settings,
    which an ablation's may not repeat; every training and validation episode
    flies the plan, the manifest records it, and a sweep does the same for each
    run. Refused, as usage errors: a tag the ablation contradicts, another reward
    or other weights, E3, and settings ``PlanScoreParams`` does not have; and (the
    repair round's A-2 and A-4) a learned arm's tag given as ``--tag`` that the
    run does not match, in train and sweep alike: an ablation's tag without the
    ablation's plan, a γ tag at another γ, and other weights than decision 4
    (a)'s under a γ tag or an ablation's; given with the ablation's plan, the
    tag trains as ``--ablation`` does."""
    flown = []

    def fly(net, schema, cell, seed, *, index, reward, trainer=None, plan_score_params=None):
        flown.append(plan_score_params)
        return _empty(cell, seed, index, reward)

    monkeypatch.setattr(T, "fly_pair_episode", fly)
    monkeypatch.setattr(K, "tree_state", lambda: TREE)
    for extra, tag, plan in (
            (("--ablation", "dwell"), "dwell", FQ_DWELL_SCORE),
            (("--ablation", "cov", "--tag", "cov", "--plan-score-params", '{"c_energy": 0}'),
             "cov", dict(F_COV_SCORE, c_energy=0)),
            (("--plan-score-params", '{"c_cov_per_device": 0.25}'), "g90",
             {"c_cov_per_device": 0.25})):
        flown.clear()
        assert CLI.main(_cli_train(tmp_path, *extra)) == 0
        m = verify_checkpoint(tmp_path / "smoke" / tag / "g0.9_s0.npz")
        assert m["training"]["spec"]["plan_score_params"] == plan and K.trained_plan(m) == plan
        assert flown == [plan] * 6, extra          # 2 training and 4 validation episodes
    flown.clear()
    assert CLI.main(["sweep", "--study", "sweep", "--root", str(tmp_path), "--gammas", "0",
                     "0.9", "--seeds", "0", "--episodes", "1", "--eval-every", "1",
                     "--val-episodes", "4", "--ablation", "cov"]) == 0
    for gamma in (0.0, 0.9):
        m = verify_checkpoint(K.checkpoint_path(tmp_path, "sweep", "cov", gamma, 0))
        assert m["gamma"] == gamma and K.trained_plan(m) == F_COV_SCORE
    assert flown == [F_COV_SCORE] * 10
    # an ablation's tag given without --ablation trains under its arm's plan, or not
    # at all (the repair round's A-2)
    flown.clear()
    assert CLI.main(_cli_train(tmp_path / "explicit", "--tag", "dwell", "--plan-score-params",
                               '{"dwell_in_delta": false}')) == 0
    m = verify_checkpoint(tmp_path / "explicit" / "smoke" / "dwell" / "g0.9_s0.npz")
    assert K.trained_plan(m) == FQ_DWELL_SCORE and flown == [FQ_DWELL_SCORE] * 6
    for sweep in (["--tag", "dwell"], ["--tag", "g90"]):
        capsys.readouterr()
        with pytest.raises(SystemExit) as refused:
            CLI.main(["sweep", "--study", "sweep", "--root", str(tmp_path / "refused"),
                      "--gammas", "0", "0.9", "--seeds", "0", "--episodes", "1",
                      "--eval-every", "1", "--val-episodes", "4", *sweep])
        assert refused.value.code == 2 and "names arm FQ-" in capsys.readouterr().err
    for extra, why in (
            (("--tag", "dwell"), "the arm flies the plan score settings {'dwell_in_delta': "
                                 "False}, and this run trains under dwell_in_delta True"),
            (("--tag", "cov"), "this run trains under c_cov_per_device 1.0, c_link None"),
            (("--tag", "cov", "--plan-score-params", '{"c_cov_per_device": 0.0}'),
             "this run trains under c_link None"),
            (("--c-t", "0.3", "--tag", "g90"), "at c_t 0.3, c_cov 1, and tag 'g90' flies it "
                                               "at decision 4 (a)'s weights"),
            (("--c-t", "0.3", "--c-cov", "4", "--tag", "dwell", "--plan-score-params",
              '{"dwell_in_delta": false}'), "at c_t 0.3, c_cov 4, and tag 'dwell'"),
            (("--tag", "g50"), "its γ is 0.9, which flies as tag 'g90', not 'g50'"),
            (("--ablation", "dwell", "--tag", "g90"), "contradicts it"),
            (("--ablation", "cov", "--plan-score-params", '{"c_link": 0.5}'),
             "--ablation cov sets ['c_link'] itself"),
            (("--ablation", "dwell", "--reward", "hand"), "derived reward at its own weights"),
            (("--ablation", "dwell", "--c-t", "0.3"), "derived reward at its own weights"),
            (("--ablation", "dwell", "--kind", "chen_dqn"), "E3 flies legacy mode, with no plan"),
            (("--plan-score-params", '{"kappa": 1}'),
             "--plan-score-params: plan_score_params: unknown settings ['kappa']"),
            (("--plan-score-params", '{"dwell_in_delta": "no"}'),
             "--plan-score-params: dwell_in_delta must be a bool"),
            (("--plan-score-params", "[1]"), "--plan-score-params must be a JSON object"),
            (("--ablation", "both"), "invalid choice")):
        capsys.readouterr()
        with pytest.raises(SystemExit) as refused:
            CLI.main(_cli_train(tmp_path / "refused", *extra))
        assert refused.value.code == 2 and why in capsys.readouterr().err, extra
    assert not (tmp_path / "refused").exists()


def test_a_checkpoint_flies_only_as_its_own_tag(smoke, monkeypatch, tmp_path):
    """Resolution R24 on the trainer's own manifests: the γ tag of its γ, its tag's
    reward (the tags' rewards are the trainer's, ``KIND_REWARDS``, so E3 has no
    reward but its bytes) at the weights its arm fixes (decision 4 (a)'s under a
    γ tag, dwell and cov; the repair round's A-4), and the update count of the
    weights it keeps, which a run whose replay never warmed up records as 0."""
    m = smoke.manifest
    kept = next(p for p in smoke.curve if p.episode == smoke.best_episode)
    assert K.kept_updates(m) == kept.updates > 0
    for tag in ("g90", "main", "dwell", "cov"):
        assert K.tag_refusals(tag, m) == [], tag
    assert K.tag_refusals("g0", m) == ["its γ is 0.9, which flies as tag 'g90', not 'g0'"]
    assert K.tag_refusals("hand", m) == [
        "it trained on the 'derived' reward, and tag 'hand' flies the 'hand' reward's score"]
    assert {K.tag_reward(t) for t in PAIR_CHECKPOINT_TAGS} == set(T.KIND_REWARDS["pair_q"])
    assert (K.tag_reward(K.E3_TAG),) == T.KIND_REWARDS["chen_dqn"] == (R.REWARD_BYTES,)
    assert K.HAND_TAG == CHECKPOINT_TAGS["FQ-hand"] and K.E3_TAG == CHECKPOINT_TAGS["E3"]
    assert K.ABLATION_TAGS == tuple(T.ABLATIONS) == (CHECKPOINT_TAGS["FQ-dwell"],
                                                       CHECKPOINT_TAGS["FQ-cov"])
    tags = PAIR_CHECKPOINT_TAGS + (K.E3_TAG,)
    assert {tag: K.tag_weights(tag) for tag in tags} == {
        tag: None if tag in ("main", K.HAND_TAG, K.E3_TAG) else (R.DERIVED.c_t, R.DERIVED.c_cov)
        for tag in tags}
    grid = dict(m, reward=dict(m["reward"], c_t=0.3, c_cov=4.0))       # Study 5.7's grid
    for tag in ("g90", "dwell", "cov"):
        assert K.tag_refusals(tag, grid) == [
            f"it trained on the derived reward at c_t 0.3, c_cov 4, and tag {tag!r} flies it "
            f"at decision 4 (a)'s weights, c_t 0.1 and c_cov 1"], tag
    assert K.tag_refusals("main", grid) == []
    _fly_nothing(monkeypatch)
    res = T.train(T.pair_spec(0.5, 1, episodes=2, eval_every=1, val_episodes=4),
                  tmp_path / "g0.5_s1.npz", tree=TREE)
    assert res.manifest["training"]["outcome"]["updates"] == 0
    assert K.kept_updates(res.manifest) == 0
    assert K.tag_refusals("g50", res.manifest) == [
        f"its network took no update: the weights it keeps (validated after episode "
        f"{res.best_episode}) are its initial ones"]


def test_the_cli_evaluates_records_the_held_out_score_and_reports(smoke, tmp_path, capsys):
    path = _copy_checkpoint(smoke.path, tmp_path)
    out = tmp_path / "out" / "evaluation.json"
    assert CLI.main(["evaluate", "--checkpoints", str(tmp_path), "--cells", "jit-n6-45",
                     "--episodes", "2", "--references", "FX", "greedy_1", "--record",
                     "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["format"] == RP.EVALUATION_FORMAT and data["cells"] == ["jit-n6-45"]
    assert sorted(data["references"]) == ["FX", "greedy_1"]
    assert data["seeds"]["jit-n6-45"] == list(C.stream_seeds(C.HELDOUT_STREAM, "jit-n6-45", 2))
    (entry,) = data["checkpoints"]
    assert entry["path"] == str(path) and len(entry["returns"]["jit-n6-45"]) == 2
    held = verify_checkpoint(path)["held_out"]
    assert entry["manifest"]["held_out"] == held and held["episodes"] == 2
    assert held["return_mean"] == pytest.approx(statistics.fmean(entry["returns"]["jit-n6-45"]))
    assert campaign_refusals(entry["manifest"]) == []
    # the report refuses to read a checkpoint on cells its validation never flew
    capsys.readouterr()
    with pytest.raises(SystemExit):
        CLI.main(["report", "--evaluation", str(out), "--epsilon", "0.01", "--cells",
                  "jit-n6-45"])
    assert "its validation did not fly the cells ['jit-n6-45']" in capsys.readouterr().err


@pytest.mark.parametrize("flags, reward", [
    (["--c-t", "0.3", "--c-cov", "4"], R.RewardSpec(c_t=0.3, c_cov=4.0)),
    (["--reward", "hand"], R.HAND),
    (["--reward", "bytes"], R.BYTES),
], ids=["derived-weights", "hand", "bytes"])
def test_the_cli_evaluates_and_records_under_the_reward_it_is_given(smoke, tmp_path, flags,
                                                                    reward):
    """Every policy is read under ``--reward`` (Study 5.7's weights, F·hand, E3's
    bytes) at the realized draw, episode by episode as FerrySim flies it, and
    the recorded held-out score names that reward."""
    path = _copy_checkpoint(smoke.path, tmp_path)
    out = tmp_path / "evaluation.json"
    cell = C.cell_named("jit-n6-45")
    assert CLI.main(["evaluate", "--checkpoints", str(path), "--cells", cell.name,
                     "--episodes", "2", "--references", "FX", "greedy_1", "--record",
                     "--out", str(out), *flags]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["reward"] == reward.to_json()
    refs = {p.label: p for p in E.reference_policies()}
    flown = {"FX": refs["FX"], "greedy_1": refs["greedy_1"],
             "checkpoint": K.checkpoint_flight(path)[0]}
    seeds = C.stream_seeds(C.HELDOUT_STREAM, cell.name, 2)
    episodes = {label: [E.run_episode(cell, seed, policy, trial_index=i, reward=reward)
                        for i, seed in enumerate(seeds)] for label, policy in flown.items()}
    returns = {label: [ep.ret for ep in eps] for label, eps in episodes.items()}
    (entry,) = data["checkpoints"]
    assert entry["returns"][cell.name] == returns["checkpoint"]
    for label in ("FX", "greedy_1"):
        assert data["references"][label]["returns"][cell.name] == returns[label]
    # the reward moved the numbers: the same flights read at the default differ
    assert [ep.rescored(R.DERIVED).ret for ep in episodes["checkpoint"]] != returns["checkpoint"]
    held = verify_checkpoint(path)["held_out"]
    assert held["reward"] == reward.to_json() and entry["manifest"]["held_out"] == held
    assert held["return_mean"] == pytest.approx(statistics.fmean(returns["checkpoint"]))


def test_evaluate_reads_each_checkpoint_on_its_plan_and_one_file_on_one_plan(smoke, tmp_path,
                                                                             capsys):
    """Resolution R23: a pair checkpoint flies the plan it trained under
    (``checkpoint_flight``), so ``evaluate --record`` scores FQ-cov's score on
    FQ-cov's plan, which changes every flight; the references fly the cells' own
    plan or ``--plan-score-params`` (the runner's format, recorded in the file),
    and a checkpoint flown beside them must have trained on theirs, so one
    evaluation file reads its policies on one plan. Without a reference the flag
    would set nothing and is refused."""
    cov = _planned_copy(smoke.path, tmp_path / "cov", F_COV_SCORE)
    cell = C.cell_named("jit-n6-45")
    out = tmp_path / "evaluation.json"
    argv = ["evaluate", "--checkpoints", str(cov), "--cells", cell.name, "--episodes", "2",
            "--out", str(out)]
    capsys.readouterr()
    with pytest.raises(SystemExit) as refused:
        CLI.main(argv + ["--references", "FX"])
    assert refused.value.code == 2 and not out.exists()
    assert ("trained under other plan score settings than the references fly: "
            "c_cov_per_device (trained 0.0, the references 1.0), c_link (trained 0.0, the "
            "references None)") in capsys.readouterr().err
    policy, overrides = K.checkpoint_flight(cov)
    assert overrides == {"plan_score_params": F_COV_SCORE}
    seeds = C.stream_seeds(C.HELDOUT_STREAM, cell.name, 2)
    on_plan = [E.run_episode(cell, s, policy, trial_index=i, driver_overrides=overrides).ret
               for i, s in enumerate(seeds)]
    assert on_plan != [E.run_episode(cell, s, policy, trial_index=i).ret
                       for i, s in enumerate(seeds)]
    assert CLI.main(argv + ["--no-references", "--record"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["checkpoints"][0]["returns"][cell.name] == on_plan
    assert "plan_score_params" not in data
    assert verify_checkpoint(cov)["held_out"]["return_mean"] == pytest.approx(
        statistics.fmean(on_plan))
    assert CLI.main(argv + ["--references", "FX", "--plan-score-params",
                            json.dumps(F_COV_SCORE)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["plan_score_params"] == F_COV_SCORE
    assert data["checkpoints"][0]["returns"][cell.name] == on_plan
    assert data["references"]["FX"]["returns"][cell.name] == [
        E.run_episode(cell, s, E.Policy.of_arm("FX"), trial_index=i,
                      driver_overrides=overrides).ret for i, s in enumerate(seeds)]
    capsys.readouterr()
    with pytest.raises(SystemExit):
        CLI.main(argv + ["--no-references", "--plan-score-params", '{"c_energy": 0}'])
    assert "none flies here" in capsys.readouterr().err


def test_evaluate_flies_the_n6_control_by_default_and_report_reads_n12():
    """Decision 5 (a): the rule is read on N = 12 and N = 6 is reported as the control
    where looking ahead cannot matter, so evaluate flies the whole jittery family
    by default (each cell's means and sorties with two or more decisions), and
    report reads Study 5.5's cells."""
    evaluate = CLI.parser().parse_args(["evaluate", "--out", "evaluation.json"])
    assert evaluate.cells == [c.name for c in C.FAMILIES[C.FAMILY_JITTERY]]
    assert {C.cell_named(c).n_devices for c in evaluate.cells} == {6, 12}
    report = CLI.parser().parse_args(["report", "--evaluation", "evaluation.json"])
    assert report.cells == [c.name for c in C.STUDY_5_5_CELLS]
    assert {C.cell_named(c).n_devices for c in report.cells} == {12}


# --------------------------------------------------------------------------- #
# Study 5.6's family and cells (the orchestrator's resolution R22)
# --------------------------------------------------------------------------- #

def test_the_cli_takes_jittery56_and_study_5_6s_cells_and_moves_no_default(monkeypatch,
                                                                           capsys):
    """Train and sweep take the family jittery56, and evaluate, headroom and report
    take Study 5.6's cells, by name; no default moves: a run trains on the
    jittery family, evaluate flies it (R20's cost), report reads Study 5.5's two
    cells and headroom decision 3's six."""
    from experiments.ferrysim import headroom as HR

    names = [c.name for c in C.STUDY_5_6_CELLS]
    assert names == ["jit-n12-120-q", "jit-n12-120-h", "jit-n12-180-q", "jit-n12-180-h"]
    for argv in (["train", "--study", "s", "--gamma", "0", "--seed", "0"],
                 ["sweep", "--study", "s", "--gammas", "0", "--seeds", "0"]):
        assert CLI.parser().parse_args(argv + ["--family", "jittery56"]).family == "jittery56"
        assert CLI.parser().parse_args(argv).family == "jittery"
        with pytest.raises(SystemExit):
            CLI.parser().parse_args(argv + ["--family", "jittery57"])
    assert T.TrainSpec().family == "jittery"
    evaluate = CLI.parser().parse_args(["evaluate", "--out", "e.json", "--cells", *names])
    assert [C.cell_named(c) for c in evaluate.cells] == list(C.STUDY_5_6_CELLS)
    assert CLI.parser().parse_args(["evaluate", "--out", "e.json"]).cells == [
        "jit-n6-45", "jit-n6-90", "jit-n12-120", "jit-n12-180"]
    assert CLI.parser().parse_args(["report", "--evaluation", "e.json"]).cells == [
        "jit-n12-120", "jit-n12-180"]
    assert CLI.parser().parse_args(["report", "--evaluation", "e.json",
                                    "--cells", *names]).cells == names
    # headroom: the episodes its command would fly, none flown
    built = []
    monkeypatch.setattr(HR, "parallel_map",
                        lambda fn, tasks, workers=1: built.append(list(tasks)) or [])
    monkeypatch.setattr(HR, "cell_headroom", lambda episodes: {"headroom": 1.0})
    monkeypatch.setattr(HR, "format_report", lambda report: "")
    assert CLI.main(["headroom", "--cells", *names, "--episodes", "2"]) == 0
    assert [(t.cell, t.index, t.seed) for t in built[0]] == [
        (c, i, seed) for c in C.STUDY_5_6_CELLS
        for i, seed in enumerate(C.stream_seeds(C.VAL_STREAM, c.name, 2))]
    assert CLI.main(["headroom", "--episodes", "1"]) == 0
    assert [t.cell for t in built[1]] == list(C.CELLS)
    capsys.readouterr()
    with pytest.raises(SystemExit) as refused:
        CLI.main(["headroom", "--cells", "jit-n12-120-x"])
    assert refused.value.code == 2 and "no FerrySim cell" in capsys.readouterr().err


def test_evaluate_flies_study_5_6s_cells_when_named(tmp_path):
    """Named, each of Study 5.6's cells flies its own seeds as FerrySim flies it."""
    out = tmp_path / "evaluation.json"
    names = ["jit-n12-120-q", "jit-n12-180-h"]
    assert CLI.main(["evaluate", "--cells", *names, "--stream", "val", "--episodes", "1",
                     "--references", "FX", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["cells"] == names
    for name in names:
        seeds = list(C.stream_seeds(C.VAL_STREAM, name, 1))
        assert data["seeds"][name] == seeds
        assert data["references"]["FX"]["returns"][name] == [E.run_episode(
            C.cell_named(name), seeds[0], E.Policy.of_arm("FX"), trial_index=0).ret]


def test_a_jittery56_score_trains_under_a_study_of_its_own(monkeypatch, tmp_path, capsys):
    """R19 with the new family: a jittery56 run practises over its eight cells and
    its manifest records the family and its hash; one study per family still
    holds, so a jittery56 run never replaces a jittery checkpoint, even asked,
    nor the reverse; and its validation needs an episode in each of the eight."""
    _fly_nothing(monkeypatch)
    monkeypatch.setattr(K, "tree_state", lambda: TREE)

    def sweep(study, family, *extra, val="8"):
        return ["sweep", "--study", study, "--family", family, "--root", str(tmp_path),
                "--gammas", "0", "--seeds", "0", "--episodes", "1", "--eval-every", "1",
                "--val-episodes", val, *extra]

    assert CLI.main(sweep("5.5", C.FAMILY_JITTERY)) == 0
    path = K.checkpoint_path(tmp_path, "5.5", "g0", 0.0, 0)
    kept = path.read_bytes()
    capsys.readouterr()
    for extra in ((), ("--overwrite",)):
        with pytest.raises(SystemExit) as refused:
            CLI.main(sweep("5.5", C.FAMILY_JITTERY_56, *extra))
        assert refused.value.code == 2
        assert "cell family 'jittery'" in capsys.readouterr().err
    assert path.read_bytes() == kept
    assert CLI.main(sweep("5.5-jittery56", C.FAMILY_JITTERY_56)) == 0
    own = K.checkpoint_path(tmp_path, "5.5-jittery56", "g0", 0.0, 0)
    m = verify_checkpoint(own)
    assert (m["cell_family"], m["cell_family_sha256"]) == (
        "jittery56", C.family_sha256(C.FAMILY_JITTERY_56))
    assert m["training"]["spec"]["family"] == "jittery56"
    assert m["training"]["spec"]["cells"] == [c.name for c in C.FAMILIES[C.FAMILY_JITTERY_56]]
    assert "own --study" in K.replace_refusal(own, C.FAMILY_JITTERY, overwrite=True)
    capsys.readouterr()
    with pytest.raises(SystemExit) as refused:
        CLI.main(sweep("5.5-jittery56-b", C.FAMILY_JITTERY_56, val="4"))
    assert refused.value.code == 2 and "val_episodes" in capsys.readouterr().err


def _as_jittery56(evaluation):
    """``evaluation`` as a sweep trained over jittery56 whose evaluation also flew
    Study 5.6's cells, where reading them would change the verdict: FX returns
    2.0 there and every other reference 0.5, γ = 0.5's checkpoints return 1.0 and
    the rest 0.0, and γ = 0.5 validates best."""
    data = json.loads(json.dumps(evaluation))
    episodes = data["episodes"]
    names = [c.name for c in C.STUDY_5_6_CELLS]
    data["cells"] += names
    for label, entry in data["references"].items():
        level, step = (2.0, 0.02) if label == "FX" else (0.5, 0.01)
        for name in names:
            entry["returns"][name] = [level + step * i for i in range(episodes)]
            entry["decisions"][name] = [[2, 2]] * episodes
    for entry in data["checkpoints"]:
        m = entry["manifest"]
        best = m["gamma"] == 0.5
        m.update(cell_family=C.FAMILY_JITTERY_56,
                 cell_family_sha256=C.family_sha256(C.FAMILY_JITTERY_56))
        m["training"]["spec"] = T.pair_spec(m["gamma"], m["seeds"]["run"],
                                            family=C.FAMILY_JITTERY_56).to_json()
        for point in m["validation"]:
            point["cells"].update({name: 1.0 if best else -1.0 for name in names})
        for name in names:
            entry["returns"][name] = [1.0 if best else 0.0] * episodes
            entry["decisions"][name] = [[2]] * episodes
    return data


def test_study_5_5s_verdict_reads_its_two_cells_whichever_family_trained_the_score(tmp_path,
                                                                                  capsys):
    """A sweep trained over jittery56, its evaluation flying Study 5.6's cells too:
    the rule reads jit-n12-120 and jit-n12-180 alone, the verdict the same sweep
    trained over jittery gives; Study 5.6's cells are reported beside them, and
    read in their place they would give another verdict. One family per sweep."""
    direct = _table({**FLATISH, 0.9: 0.05}, refs=_refs(greedy=0.004, n=12, block=4))
    plain = _evaluation(direct)
    sweep = _as_jittery56(plain)
    read = [c.name for c in C.STUDY_5_5_CELLS]
    assert read == ["jit-n12-120", "jit-n12-180"]
    table = RP.sweep_table(sweep, cells=read, epsilon=0.01)
    same = RP.sweep_table(plain, cells=read, epsilon=0.01)
    assert (table.held_out, table.validation, table.references) == (
        same.held_out, same.validation, same.references)
    v, w = RP.decide(table), RP.decide(same)
    assert (v.outcome, v.best_gamma, v.replace_fx, v.cells) == (
        w.outcome, w.best_gamma, w.replace_fx, w.cells) == (
        RP.OUTCOME_RISING, 0.9, True, tuple(read))
    assert set(v.per_cell) == set(sweep["cells"]) and "jit-n12-180-h" in v.decisions
    other = RP.decide(RP.sweep_table(sweep, cells=[c.name for c in C.STUDY_5_6_CELLS],
                                     epsilon=0.01))
    assert (other.outcome, other.best_gamma) == (RP.OUTCOME_SANITY_FAILED, 0.5)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps(sweep), encoding="utf-8")
    verdict = tmp_path / "verdict.json"
    assert CLI.main(["report", "--evaluation", str(evaluation), "--epsilon", "0.01",
                     "--out", str(verdict)]) == 0
    assert "Study 5.5 on jit-n12-120, jit-n12-180:" in capsys.readouterr().out
    saved = json.loads(verdict.read_text(encoding="utf-8"))["verdict"]
    assert (saved["cells"], saved["outcome"], saved["best_gamma"], saved["replace_fx"]) == (
        read, "rising", 0.9, True)
    sweep["checkpoints"][0]["manifest"]["cell_family"] = C.FAMILY_JITTERY
    with pytest.raises(ValueError, match="cell families"):
        RP.sweep_table(sweep, cells=read, epsilon=0.01)


def test_the_cli_reports_a_sweep(tmp_path, capsys):
    sweep = _evaluation(_table({**FLATISH, 0.9: 0.05}, refs=_refs(greedy=0.004, n=12, block=4)))
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps(sweep), encoding="utf-8")
    headroom = tmp_path / "headroom.json"
    headroom.write_text(json.dumps({"stream": C.VAL_STREAM, "cells": {
        c: {"headroom": 0.05} for c in CELLS_READ}}), encoding="utf-8")
    verdict = tmp_path / "verdict.json"
    assert CLI.main(["report", "--evaluation", str(evaluation), "--headroom", str(headroom),
                     "--out", str(verdict)]) == 0
    printed = capsys.readouterr().out
    assert "Outcome: rising" in printed and "Replace FX: YES" in printed
    saved = json.loads(verdict.read_text(encoding="utf-8"))["verdict"]
    assert (saved["outcome"], saved["replace_fx"], saved["epsilon"]) == ("rising", True, 0.01)
    with pytest.raises(SystemExit):
        CLI.main(["report", "--evaluation", str(evaluation)])
    # an evaluation without some of the rule's references (evaluate --references FX
    # greedy_1) is refused as a usage error, like every other refusal here
    narrowed = tmp_path / "narrowed.json"
    narrowed.write_text(json.dumps(dict(sweep, references={
        k: v for k, v in sweep["references"].items() if k in ("FX", "greedy_1")})),
        encoding="utf-8")
    capsys.readouterr()
    with pytest.raises(SystemExit) as refused:
        CLI.main(["report", "--evaluation", str(narrowed), "--epsilon", "0.01"])
    assert refused.value.code == 2
    assert "no held-out returns of ['F', 'hyb']" in capsys.readouterr().err


@pytest.fixture
def logging_put_back():
    """The command line's ``main`` turns HERMES' logs below errors off while it
    runs and puts the level back itself; this puts it back after the test too."""
    level = logging.root.manager.disable
    yield
    logging.disable(level)


def test_epsilon_is_read_on_the_plan_the_evaluation_flew(tmp_path, capsys, logging_put_back):
    """Resolution R23 (the repair round's A-1): ε applies the validation headroom to
    returns flown on the evaluation's plan, so it is read only from a headroom
    report that flew the same plan score settings, compared with their defaults
    filled in: none recorded is the cells' own plan, and so are its defaults
    written out. Another plan is refused, naming each setting, and the report
    command refuses it as a usage error."""
    pilot = {"c_cov_per_device": 0.25}
    own = {"stream": C.VAL_STREAM, "cells": {c: {"headroom": 0.3} for c in CELLS_READ}}
    on_pilot = dict(own, plan_score_params=pilot)
    defaults = {"c_cov_per_device": 1, "dwell_in_delta": True, "c_link": None}
    for headroom, plan in ((own, None), (own, {}), (own, defaults), (on_pilot, pilot),
                           (dict(own, plan_score_params=defaults), None)):
        assert RP.epsilon_from_headroom_report(headroom, CELLS_READ, plan_score_params=plan) == (
            pytest.approx(0.03))
    for headroom, plan, why in (
            (own, pilot, "c_cov_per_device (headroom 1.0, evaluation 0.25)"),
            (on_pilot, None, "c_cov_per_device (headroom 0.25, evaluation 1.0)"),
            (on_pilot, F_COV_SCORE, "c_cov_per_device (headroom 0.25, evaluation 0.0), c_link "
                                    "(headroom None, evaluation 0.0)")):
        with pytest.raises(ValueError, match="other plan score settings") as refused:
            RP.epsilon_from_headroom_report(headroom, CELLS_READ, plan_score_params=plan)
        assert f"(resolution R23): {why}; read ε from a headroom report flown on the " \
               f"evaluation's plan" in str(refused.value)
    sweep = _evaluation(_table(RISING, refs=_refs(greedy=0.004, n=12, block=4)))
    planned = json.loads(json.dumps(sweep))
    planned["plan_score_params"] = pilot          # as evaluate --plan-score-params records it
    for entry in planned["checkpoints"]:
        entry["manifest"]["training"]["spec"]["plan_score_params"] = dict(pilot)
    files = {}
    for name, data in (("own", own), ("on_pilot", on_pilot), ("sweep", sweep),
                       ("planned", planned), ("defaults", dict(sweep, plan_score_params=defaults))):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(data), encoding="utf-8")
    verdict = tmp_path / "verdict.json"
    for evaluation, headroom in (("sweep", "own"), ("planned", "on_pilot"), ("defaults", "own")):
        assert CLI.main(["report", "--evaluation", str(files[evaluation]), "--headroom",
                         str(files[headroom]), "--out", str(verdict)]) == 0
        assert json.loads(verdict.read_text(encoding="utf-8"))["verdict"]["epsilon"] == (
            pytest.approx(0.03))
    for evaluation, headroom, why in (
            ("planned", "own", "c_cov_per_device (headroom 1.0, evaluation 0.25)"),
            ("sweep", "on_pilot", "c_cov_per_device (headroom 0.25, evaluation 1.0)")):
        capsys.readouterr()
        with pytest.raises(SystemExit) as refused:
            CLI.main(["report", "--evaluation", str(files[evaluation]), "--headroom",
                      str(files[headroom])])
        assert refused.value.code == 2 and why in capsys.readouterr().err


def test_the_cli_prints_and_saves_the_grids_label(tmp_path, capsys):
    """Resolution R26 through the report command: an evaluation of decision 5
    (a)'s grid (1,000 held-out episodes per cell) prints and saves
    pre-registered; one of 4 episodes per cell, and the calibration's 2 γ x 3
    seeds, print and save not pre-registered with their reasons, and are read."""
    cases = (
        ("grid", _evaluation(_table(RISING, refs=_refs(greedy=0.004, n=3000, block=1000)),
                             episodes=1000), []),
        ("short", _evaluation(_table(RISING, refs=_refs(greedy=0.004, n=12, block=4))),
         ["4 held-out episodes per cell, not 1000"]),
        ("calibration", _evaluation(_table({0.0: 0.0, 0.9: 0.0}, seeds=3,
                                           refs=_refs(n=12, block=4))),
         [f"γ ∈ {{0, 0.9}}, not {GRID}", "3 seeds per γ, not 10",
          "4 held-out episodes per cell, not 1000"]),
    )
    for name, data, reasons in cases:
        evaluation, verdict = tmp_path / f"{name}.json", tmp_path / f"{name}-verdict.json"
        evaluation.write_text(json.dumps(data), encoding="utf-8")
        capsys.readouterr()
        assert CLI.main(["report", "--evaluation", str(evaluation), "--epsilon", "0.01",
                         "--out", str(verdict)]) == 0, name
        printed = capsys.readouterr().out.splitlines()
        saved = json.loads(verdict.read_text(encoding="utf-8"))["verdict"]
        assert saved["preregistered"] is (not reasons), name
        assert saved["preregistration"]["reasons"] == reasons, name
        assert saved["preregistration"]["sweep"]["held_out_episodes"] == data["episodes"]
        assert printed[1] == (
            f"Grid: pre-registered (decision 5 (a)): γ ∈ {GRID}, 10 seeds per γ, 1000 "
            f"held-out episodes per cell" if not reasons
            else "Grid: NOT pre-registered (decision 5 (a)): " + "; ".join(reasons)), name
        assert saved["outcome"] in RP.OUTCOMES


def test_the_cli_runs_as_a_module_and_hands_headroom_its_own_options(capsys):
    out = subprocess.run([sys.executable, "-m", "experiments.ferrysim", "--help"], cwd=REPO,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0
    for command in CLI.COMMANDS:
        assert command in out.stdout
    with pytest.raises(SystemExit) as shown:
        CLI.main(["headroom", "--help"])
    assert shown.value.code == 0 and "--max-leaves" in capsys.readouterr().out


def test_each_command_line_puts_the_callers_logging_level_back(monkeypatch, capsys):
    """FerrySim's command lines (``__main__``, evaluate's and headroom's) turn
    HERMES' logs below errors off while their episodes fly, and put the caller's
    level back when they return, refuse or raise: a test that calls one leaves
    the logs of the tests after it alone (the golden host-mission scenarios
    compare theirs)."""
    from experiments.ferrysim import evaluate as EV
    from experiments.ferrysim import headroom as HR

    seen = []

    def runs(*args, **kwargs):
        seen.append(logging.root.manager.disable)
        return 0

    def refuses(*args, **kwargs):
        seen.append(logging.root.manager.disable)
        raise ValueError("refused")

    before = logging.root.manager.disable
    try:
        for level in (logging.NOTSET, logging.INFO):
            logging.disable(level)
            monkeypatch.setattr(CLI, "_cmd_report", runs)
            assert CLI.main(["report", "--evaluation", "e.json", "--epsilon", "0.01"]) == 0
            assert logging.root.manager.disable == level
            monkeypatch.setattr(CLI, "_cmd_report", refuses)
            with pytest.raises(ValueError, match="refused"):
                CLI.main(["report", "--evaluation", "e.json", "--epsilon", "0.01"])
            assert logging.root.manager.disable == level
            for module, name in ((EV, "evaluate"), (HR, "headroom_report")):
                monkeypatch.setattr(module, name, refuses)
                with pytest.raises(SystemExit) as refused:
                    module.main(["--cells", "jit-n12-120", "--episodes", "1"])
                assert refused.value.code == 2 and "refused" in capsys.readouterr().err
                assert logging.root.manager.disable == level
        assert seen == [logging.WARNING] * 8
    finally:
        logging.disable(before)
