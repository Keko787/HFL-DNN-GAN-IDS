"""FeRRy Phase 5 (unit U7): the learned arms through the config guards, the mule
process, the builder, the driver and the runner.

Pinned (the Phase 5 spec's units table, row U7; decisions 8 (a) and 9 (a);
other choices 5 and 6; critic A5, A6, A10, B9 and B10; the orchestrator's
resolution R3):

* **The guards.** The pair slot flies ``in_flight_response='replan'`` only
  (R3), refused by ``mule_config_errors`` as one reason (A10's when a cap is
  set too) and by the driver before anything is built. A learned arm without
  its checkpoint is refused (no random-init arm), and so is a checkpoint its
  mule would refuse (kind, classes, band, file, or one removed or rewritten
  since the first check), before any trial and always as a ValueError; the
  driver checks no training state, so a bootstrap checkpoint flies. A
  learned arm flies on the simulated clock only, and E3 flies the configured
  in-flight response, as the D arms do.
* **Arms and labels.** ``PHASE_5_ARMS`` is the learned arms and ``H1+L1``;
  ``ARMS`` gains them and ``DEFAULT_ARMS`` and ``PLAN_ARMS`` keep their pinned
  values; every label survives a trace directory and the scorer parses it back.
* **FQ is F (A5).** Each FQ arm's mule config is F's but for the flight slot,
  the checkpoint fields and its ablation's own score field.
* **H1+L1** is H1's scheduler with H3's adaptive backhaul and no learned
  selector, on the seconds-axis model and on the L1 channel's loss schedule,
  and is refused where it would fly as H1.
* **Provenance** names a checkpoint by tag and sha, ``pair_tag`` and
  ``pair_sha256`` in ``ferry_params`` for the FQ arms and ``policy_tag`` and
  ``policy_sha256`` in ``policy_params`` for E3, never by path; every other
  row keeps its Phase 3 or Phase 4 strings (UG4's and UG5's oracles re-run).
  A checkpoint inside the repository travels repo-relative however it was
  given, and any other as its resolved absolute path (not other choices 6's
  "as given": the mule reads a relative path under the repository root).
* **The mule** builds the pair slot or E3's policy from its verified
  checkpoint before it binds anything, refuses any other, announces it in
  ``mule_ready`` (``pair``, ``policy_checkpoint``) and carries the missions'
  decisions (``pass_1_pairs``, ``pass_1_e3``, ``pass_1_e3_unvisited``), each
  only on those mules and only when it has one.
* **Stub trials of FQ and E3** through the driver and the real services in
  this process, with checkpoints made in the test (under ``tmp_path``).
* **Rule 1 at the defaults.** No recorded trial gains a Phase 5 field, and a
  recorded path loads no Phase 5 module.
* **The runner** refuses a checkpoint a campaign may not fly (bootstrap, no
  episode, no held-out score, dirty without ``--allow-dirty-checkpoint``; B9)
  under ``--pair-checkpoint`` and ``--policy-checkpoint`` alike, since the
  driver would fly it, and, under ``--require-trained``, H2 and H3 without
  ``--selector-weights``; each as a usage error whose message says why (the
  usage it prints first names every flag). It also refuses a checkpoint that
  is not its tag's (the orchestrator's resolution R24: a γ tag's γ, the tag's
  reward, at decision 4 (a)'s weights under a γ tag, dwell and cov, and a
  network still at its initial weights, read off the kept validation whatever
  validated before it; the Phase 5 repair round's A-3 and A-4) and a pair
  checkpoint trained on another plan than its arm flies under the run's flags
  (resolution R23), and flies each under its own tag and plan. Its help gives
  each checkpoint flag its own conditions (A-5).

The real-process trials are in ``tests/integration/test_p5_learned_trials.py``.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.analysis.traces_scorer import parse_trial_dir
from experiments.exp4.driver import (
    ARMS,
    CHECKPOINT_TAGS,
    DEFAULT_ARMS,
    F_COV_SCORE,
    FQ_DWELL_SCORE,
    LEARNED_ARMS,
    PAIR_ARMS,
    PAIR_CHECKPOINT_TAGS,
    PHASE_5_ARMS,
    PLAN_ARMS,
    POLICY_CHECKPOINT_TAGS,
    PROVENANCE_COLUMNS,
    Exp4Driver,
    is_plan_arm,
    learned_policy_params,
    plan_ferry_params,
    trace_dir_name,
)
from experiments.runner import Cell
from hermes.mule.mule_main import MissionRunResult
from hermes.processes import config as C
from hermes.processes import mule as mule_process
from hermes.processes.config import (
    PAIR_CHECKPOINT_FIELDS,
    PLAN_MULE_FIELDS,
    POLICY_CHECKPOINT_FIELDS,
    MuleConfig,
    mule_config_errors,
)
from hermes.scheduler.policies.chen_dqn import (
    ChenDQNPolicy,
    new_e3_network,
    save_e3_checkpoint,
)
from hermes.scheduler.policies.pair_slot import CLOSE_KEYS, DECISION_KEYS, PairQSlot
from hermes.scheduler.selector.pair_features import PairFeatureSchema
from hermes.scheduler.selector.pair_q import (
    KIND_CHEN_DQN,
    KIND_PAIR_Q,
    PairQConfig,
    PairQNet,
    manifest_path,
    manifest_provenance,
    verify_checkpoint,
)
from hermes.scheduler.selector.pair_replay import PairReplay, PairTransition
from hermes.transport import TCPDockLinkServer

from tests.golden import _build_p3_sim as P3
from tests.golden import _build_p4_plan as UG5
from tests.golden import _build_topology as T

REPO = Path(__file__).resolve().parents[2]

#: The D1 link's classes in link order, the pilots' (``contact_band_classes`` None).
LINK = ("wide", "medium", "narrow")
#: UG5's flags (the Phase 4 pilots': the simulated clock, wide as the reference
#: class, the T_nom deadline unit, re-plan with the trim fallback, agg:cutoff,
#: the channel reliability source, 1 MB, the jittery contact channel, 45 s,
#: S = 2), with T_nom over 5 reference layouts.
PILOT = dict(UG5.PLAN_PILOT, t_nom_layouts=5)
#: The fields Phase 5 adds to ``mule_ready`` and ``mission_completed``.
READY_FIELDS = ("pair", "policy_checkpoint")
MISSION_FIELDS = ("pass_1_pairs", "pass_1_e3", "pass_1_e3_unvisited")
#: The modules no recorded arm may load (the Phase 5 spec, other choices 13).
PHASE_5_MODULES = ("pair_slot", "pair_features", "pair_q", "pair_replay", "chen_dqn",
                   "next_stop")


# --------------------------------------------------------------------------- #
# Checkpoints, made under tmp_path
# --------------------------------------------------------------------------- #

#: The updates a trained checkpoint's kept weights took (resolution R24 refuses a
#: network still at its initial weights).
UPDATES = 3


def _trained(**changes):
    """A checkpoint's provenance as the trainer writes it (unit U8b) once the
    evaluator has scored it (U8a): trained on episodes from a clean tree, its
    kept weights ``UPDATES`` updates on (the validation at its kept episode
    records them, as ``train.ValidationPoint`` does), with a held-out score,
    so no campaign refusal applies (critic B9, resolution R24)."""
    out = dict(reward={"kind": "derived", "c_t": 0.1, "c_cov": 1.0},
               training={"episodes": 2000}, seeds={"init": 1, "train": 7},
               cell_family="jittery", cell_family_sha256="ab" * 32, trainer_commit="a" * 40,
               dirty=False, episodes_trained=2000,
               validation=[{"episode": 2000, "return_mean": 0.31, "updates": UPDATES}],
               held_out={"episodes": 1000, "return_mean": 0.33})
    out.update(changes)
    return out


def _bootstrap(**changes):
    """A bootstrap checkpoint's provenance: no episode, no score, a dirty tree."""
    out = _trained(training={"episodes": 0}, trainer_commit=None, dirty=True,
                   episodes_trained=0, validation=[], held_out=None)
    out.update(changes)
    return out


def _updated(net, provenance):
    """``net`` after the updates its provenance's kept validation records, on a fixed
    batch of done transitions, so a checkpoint's weights are what its manifest says:
    a trained network's are no longer its initial ones."""
    kept = [p for p in provenance["validation"]
            if p.get("episode") == provenance["episodes_trained"]]
    replay = PairReplay(4, seed=0)
    for i in range(4):
        replay.push(PairTransition(x=np.linspace(-1.0, 1.0, net.feature_dim) * (i + 1),
                                   reward=0.25 * i, done=True))
    for _ in range(kept[0].get("updates", 0) if kept else 0):
        net.update(replay.sample(4))
    return net


def _pair_checkpoint(path, *, seed=0, classes=LINK, purpose="trained", provenance=None,
                     gamma=0.9):
    """A pair_q checkpoint over ``pair_v1`` on ``classes``; returns its sha256."""
    provenance = _trained() if provenance is None else provenance
    schema = PairFeatureSchema(tuple(classes))
    net = _updated(PairQNet(schema.dim, PairQConfig(gamma=gamma), seed=seed), provenance)
    return net.save(Path(path), kind=KIND_PAIR_Q, purpose=purpose, schema=schema.to_json(),
                    classes=list(schema.classes), provenance=provenance)


def _e3_checkpoint(path, *, seed=0, band="wide", purpose="trained", provenance=None):
    """An E3 (chen_dqn) checkpoint on ``band``; returns its sha256."""
    provenance = _trained(reward={"kind": "bytes"}) if provenance is None else provenance
    return save_e3_checkpoint(_updated(new_e3_network(seed=seed), provenance), Path(path),
                              band=band, purpose=purpose, provenance=provenance)


#: The plan each ablation's checkpoint trains under (resolution R23).
ABLATION_PLANS = {"dwell": FQ_DWELL_SCORE, "cov": F_COV_SCORE}


def _planned(plan, **changes):
    """A trained provenance whose training spec records the plan ``plan``."""
    return _trained(training={"episodes": 2000, "spec": {"plan_score_params": dict(plan)}},
                    **changes)


def _tag_gamma(tag):
    """The γ a γ sweep tag flies (``g25``: 0.25); 0.9 under every other tag."""
    return int(tag[1:]) / 100 if tag[:1] == "g" and tag[1:].isdigit() else 0.9


def _tag_provenance(tag):
    """A trained provenance of ``tag``'s arm (resolutions R23 and R24): F·hand for
    ``hand``, FQ-dwell's or FQ-cov's plan for ``dwell`` and ``cov``."""
    if tag == "hand":
        return _trained(reward={"kind": "hand"})
    if tag in ABLATION_PLANS:
        return _planned(ABLATION_PLANS[tag])
    return _trained()


@dataclasses.dataclass(frozen=True)
class Checkpoints:
    """One checkpoint per learned arm's tag: its path (str) and sha256."""

    root: Path
    paths: dict
    shas: dict

    def pair(self):
        return {tag: self.paths[tag] for tag in PAIR_CHECKPOINT_TAGS}

    def policy(self):
        return {"e3": self.paths["e3"]}


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory) -> Checkpoints:
    """One trained checkpoint per tag, each its tag's (γ, reward and plan), so the
    runner flies any of them under its tag (resolutions R23 and R24)."""
    root = tmp_path_factory.mktemp("u7_ckpt")
    paths, shas = {}, {}
    for i, tag in enumerate(PAIR_CHECKPOINT_TAGS):
        gamma = _tag_gamma(tag)
        path = root / tag / f"g{gamma:g}_s{i}.npz"
        shas[tag] = _pair_checkpoint(path, seed=i, gamma=gamma, provenance=_tag_provenance(tag))
        paths[tag] = str(path)
    path = root / "e3" / "e3_s0.npz"
    shas["e3"] = _e3_checkpoint(path, seed=3)
    paths["e3"] = str(path)
    return Checkpoints(root, paths, shas)


def _settings(ckpts, **kw):
    out = dict(PILOT, pair_checkpoints=ckpts.pair(), policy_checkpoints=ckpts.policy())
    out.update(kw)
    return out


def _cell(arm, seed=7, trial=0, **params) -> Cell:
    p = {"N": 6, "rrf": 60.0, "n_missions": 4, "regime": "jittery"}
    p.update(params)
    return Cell(cell_id="|".join(f"{k}={v}" for k, v in sorted(p.items())), arm=arm,
                trial_index=trial, seed=seed, params=p)


def _refused_before_anything_is_built(drv, arm, match):
    with pytest.raises(ValueError, match=match):
        drv.check_arm(arm)
    T.FakeOrchestrator.last = None
    with pytest.raises(ValueError, match=match):
        T.run_stub_trial(drv, _cell(arm))
    assert T.FakeOrchestrator.last is None


# --------------------------------------------------------------------------- #
# Configs for the guards
# --------------------------------------------------------------------------- #

SHA = "0123456789abcdef" * 4
PAIR = dict(pair_checkpoint="results/exp5/checkpoints/s55/main/g0.9_s0.npz",
            pair_checkpoint_sha256=SHA, pair_checkpoint_tag="main")
POLICY = dict(policy_checkpoint="results/exp5/checkpoints/s53/e3/e3_s0.npz",
              policy_checkpoint_sha256=SHA, policy_checkpoint_tag="e3")


def _sim_mule(**kw) -> MuleConfig:
    kw.setdefault("mule_id", "m")
    kw.setdefault("mission_clock", "sim")
    kw.setdefault("trial_seed", 7)
    kw.setdefault("n_missions", 4)
    kw.setdefault("rf_range_m", 60.0)
    return MuleConfig(**kw)


def _plan_mule(**kw) -> MuleConfig:
    """Arm F's mule config: plan mode on the pilots' settings."""
    base = dict(plan_mode="ferry", contact_band="wide", t_nom_s=200.0,
                in_flight_response="replan", replan_fallback="trim",
                member_admission="subset", age_cap_missions=2)
    base.update(kw)
    return _sim_mule(**base)


def _fq(**kw) -> MuleConfig:
    return _plan_mule(**dict(dict(flight_slot="pair_q", **PAIR), **kw))


R3_REFUSAL = (
    "flight_slot='pair_q' needs in_flight_response='replan', got 'abort': the pair slot "
    "admits a (band, next stop) pair only when the whole rest of the flight still fits after "
    "the stop, which is the fold the re-plan's departure check runs next (resolution R3)"
)


# --------------------------------------------------------------------------- #
# The guards: R3
# --------------------------------------------------------------------------- #

def test_the_pair_slot_flies_replan_only():
    """Resolution R3, in ``mule_config_errors``: the pair slot's mask folds the
    whole rest of the flight, which only the re-plan's departure check folds
    next. One reason per mistake: with a cap, A10's reason (which asks for
    'replan' too) is the one; outside plan mode U0's "plan mode only"."""
    assert mule_config_errors(_fq(in_flight_response="abort", age_cap_missions=None)) == [
        R3_REFUSAL]
    errors = mule_config_errors(_fq(in_flight_response="abort"))
    assert len(errors) == 1 and "critic A10" in errors[0]
    assert mule_config_errors(_sim_mule(contact_band="wide", flight_slot="pair_q", **PAIR)) == [
        "flight_slot: plan mode only; set plan_mode='ferry' or leave the default"]
    assert mule_config_errors(_fq()) == [] and mule_config_errors(_fq(age_cap_missions=None)) == []
    # The fixed fillings still fly abort without a cap, as recorded.
    for slot in ("committed", "cross_heuristic"):
        assert mule_config_errors(_plan_mule(flight_slot=slot, in_flight_response="abort",
                                             age_cap_missions=None)) == []


def test_the_driver_refuses_an_fq_arm_without_replan_before_anything_is_built(ckpts):
    drv = Exp4Driver(**_settings(ckpts, in_flight_response="abort", age_cap_missions=None))
    for arm in PAIR_ARMS:
        _refused_before_anything_is_built(drv, arm, r"--in-flight-response replan; resolution R3")
    for arm in ("F", "FX", "E3"):              # the fixed slots and E3 fly abort
        drv.check_arm(arm)


@pytest.mark.parametrize("response", ["abort", "replan"])
def test_e3_flies_the_configured_in_flight_response(response, ckpts):
    """R3 binds the pair slot only. E3 is a legacy-mode whole scheduler and
    flies the driver's in-flight response, as D1 does: its Pass 1 is the same
    under either (unit U6), and only Pass 2's handling differs, as for every
    arm. The row records the response as for D1 ("" for abort)."""
    drv = Exp4Driver(**_settings(ckpts, in_flight_response=response, age_cap_missions=None))
    rows = {}
    for arm in ("E3", "D1"):
        rows[arm], topo = T.run_stub_trial(drv, _cell(arm))
        assert topo.mules[0].in_flight_response == response
    assert rows["E3"]["in_flight_response"] == rows["D1"]["in_flight_response"] == (
        "" if response == "abort" else "replan")


# --------------------------------------------------------------------------- #
# Arms, labels and tags
# --------------------------------------------------------------------------- #

def test_the_phase_5_arm_lists_and_tags():
    """Other choices 5 and decision 8 (a). The default list and the plan arms
    keep their pinned values; each learned arm has a tag the config accepts."""
    assert DEFAULT_ARMS == ("H0", "H1", "H2", "H3", "D1", "D2", "D3", "D4", "D5")
    assert PLAN_ARMS == ("F", "FX", "FB+wide", "FB+medium", "FB+narrow", "F-cov", "F-cap",
                         "F-prio")
    assert PAIR_ARMS == ("FQ", "FQ-hand", "FQ-dwell", "FQ-cov", "FQ-g0", "FQ-g25", "FQ-g50",
                         "FQ-g75", "FQ-g90", "FQ-g99")
    assert LEARNED_ARMS == PAIR_ARMS + ("E3",) and PHASE_5_ARMS == LEARNED_ARMS + ("H1+L1",)
    assert ARMS == DEFAULT_ARMS + PLAN_ARMS + PHASE_5_ARMS and len(set(ARMS)) == len(ARMS)
    assert CHECKPOINT_TAGS == {
        "FQ": "main", "FQ-hand": "hand", "FQ-dwell": "dwell", "FQ-cov": "cov", "FQ-g0": "g0",
        "FQ-g25": "g25", "FQ-g50": "g50", "FQ-g75": "g75", "FQ-g90": "g90", "FQ-g99": "g99",
        "E3": "e3"}
    assert PAIR_CHECKPOINT_TAGS + POLICY_CHECKPOINT_TAGS == tuple(CHECKPOINT_TAGS.values())
    for tag in CHECKPOINT_TAGS.values():
        assert mule_config_errors(_fq(pair_checkpoint_tag=tag)) == []
    assert [arm for arm in ARMS if is_plan_arm(arm)] == list(PLAN_ARMS + PAIR_ARMS)


@pytest.mark.parametrize("arm", PHASE_5_ARMS)
def test_a_phase_5_label_survives_its_trace_directory(arm):
    """ASCII, at most 28 characters, no "__" and none of <>:"/\\|?*: a kept
    trace's directory holds the label whole and the scorer parses it back."""
    assert arm.isascii() and len(arm) <= 28 and arm == arm.strip(" .")
    assert "__" not in arm and not set('<>:"/\\|?*') & set(arm)
    cell = _cell(arm, seed=2191267877, trial=3)
    name = trace_dir_name(cell)
    assert f"__{arm}__" in name
    key = parse_trial_dir(name)
    assert (key.arm, key.trial_index, key.seed) == (arm, 3, 2191267877)


def test_each_phase_5_arms_admission_priority_and_t_nom(ckpts):
    """An FQ arm runs as F (critic A5): subsets unless set otherwise, the miss
    priority on, T_nom always; H1+L1 as H1; E3 visits whole stops whatever the
    setting, and runs the configured priority."""
    for setting in (None, "subset", "whole"):
        drv = Exp4Driver(**_settings(ckpts, member_admission=setting, miss_priority=False))
        for arm in PAIR_ARMS:
            assert drv.effective_member_admission(arm) == (setting or "subset")
            assert drv.effective_miss_priority(arm) is True and drv._needs_t_nom(arm)
            assert drv.plan_settings(arm)["member_admission"] == (setting or "subset")
        assert drv.effective_member_admission("H1+L1") == drv.effective_member_admission("H1")
        assert drv.plan_settings("H1+L1") == drv.plan_settings("H1")
        assert drv.effective_member_admission("E3") == "whole" and drv.plan_settings("E3") == {}
        assert drv.effective_miss_priority("E3") is False
        assert drv.effective_miss_priority("H1+L1") is False


def test_an_fq_arm_gets_fs_t_nom_and_trim_fallback_where_no_other_arm_needs_them(ckpts):
    """Critic A5. Where no H arm needs T_nom (the recorded time unit, a given
    backhaul period) and the configured fallback is the recorded 'reorder', an
    FQ arm still gets T_nom, T in its plan score (decision 2 (b)), and the
    'trim' fallback, which plan mode requires, as F does; E3 and H1+L1 keep the
    configured ones."""
    drv = Exp4Driver(mission_clock="sim", realism=True, contact_band="wide", t_nom_layouts=4,
                     in_flight_response="replan", backhaul_model="seconds",
                     backhaul_period_s=600.0, pair_checkpoints=ckpts.pair(),
                     policy_checkpoints=ckpts.policy())
    for arm in ("H1", "E3", "H1+L1"):
        assert not drv._needs_t_nom(arm)
        assert drv.ferry_settings(arm=arm, regime="jittery")["replan_fallback"] == "reorder"
    for arm in ("F",) + PAIR_ARMS:
        assert drv._needs_t_nom(arm)
        assert drv.ferry_settings(arm=arm, regime="jittery")["replan_fallback"] == "trim"
    _f_row, f_topo = T.run_stub_trial(drv, _cell("F"))
    q_row, q_topo = T.run_stub_trial(drv, _cell("FQ"))
    (f,), (q,) = f_topo.mules, q_topo.mules
    assert q.t_nom_s == f.t_nom_s and q.t_nom_s is not None and q.replan_fallback == "trim"
    assert json.loads(q_row["ferry_params"])["t_nom_computed"] is True
    _e_row, e_topo = T.run_stub_trial(drv, _cell("E3"))
    assert e_topo.mules[0].t_nom_s is None and e_topo.mules[0].replan_fallback == "reorder"


# --------------------------------------------------------------------------- #
# No random-init arm, and the checkpoint checks
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", LEARNED_ARMS)
def test_a_learned_arm_without_its_checkpoint_is_refused(arm, ckpts):
    """Other choices 5: no random-init arm. Another tag's checkpoint does not
    stand in for the arm's own."""
    tag = CHECKPOINT_TAGS[arm]
    pair = {t: p for t, p in ckpts.pair().items() if t != tag}
    policy = {t: p for t, p in ckpts.policy().items() if t != tag}
    drv = Exp4Driver(**_settings(ckpts, pair_checkpoints=pair, policy_checkpoints=policy))
    _refused_before_anything_is_built(drv, arm, f"tagged '{tag}'.*no random-init arm")


def test_a_learned_arm_on_the_wall_clock_is_refused_with_its_own_reason():
    """A learned filling flies on the simulated mission clock only (the
    checkpoint mappings are refused there when the driver is built, so these
    arms have none): each learned arm says so, and a plan arm keeps Phase 4's
    reason."""
    drv = Exp4Driver()
    for arm in LEARNED_ARMS:
        _refused_before_anything_is_built(drv, arm, re.escape(
            f"arm {arm} flies a learned filling (FeRRy Phase 5) on the simulated mission "
            f"clock: run it with mission_clock='sim' (--mission-clock sim)"))
    _refused_before_anything_is_built(drv, "F", re.escape(
        "arm F flies the plan clock (FeRRy Phase 4) on the simulated mission clock"))


def _edit_arrays(path):
    with np.load(path) as archive:
        data = {name: archive[name] for name in archive.files}
    data["layer0_b"] = data["layer0_b"] + 1.0
    np.savez(path, **data)


def _mentions(text, path) -> bool:
    """Whether ``text`` names ``path`` in any of its spellings, JSON's included."""
    path = Path(path)
    spellings = {str(path), path.as_posix(), json.dumps(str(path))[1:-1]}
    return any(s in text for s in spellings)


@pytest.mark.parametrize("case,match", [
    ("pair_kind", "arm FQ: checkpoint .* is a 'chen_dqn' checkpoint, and arm FQ flies a "
                  "'pair_q' one"),
    ("e3_kind", "arm E3: checkpoint .* is a 'pair_q' checkpoint, and arm E3 flies a "
                "'chen_dqn' one"),
    ("classes", "arm FQ: its mule would refuse its checkpoint: .*trained on the classes"),
    ("band", "arm E3: its mule would refuse its checkpoint: .*trained on the classes"),
    ("missing", "arm FQ: checkpoint .* is refused: pair checkpoint not found"),
    ("not_npz", "arm FQ: checkpoint .* is refused: a format-2 checkpoint is an .npz file"),
    ("no_manifest", "arm FQ: checkpoint .* is refused: .*has no manifest beside it"),
    ("arrays_edited", "arm FQ: checkpoint .* is refused: .*its arrays' sha256 is"),
])
def test_the_driver_refuses_a_checkpoint_its_mule_would_refuse(case, match, ckpts, tmp_path):
    """Before any trial: a checkpoint of another kind (an E3 network for an FQ
    arm, a pair score for E3), one trained on other classes or on another band,
    and a file that is not there, not an ``.npz``, has no manifest, or whose
    arrays are not its manifest's."""
    path = tmp_path / "x.npz"
    arm, pair, policy = "FQ", ckpts.pair(), ckpts.policy()
    if case == "pair_kind":
        _e3_checkpoint(path)
        pair = dict(pair, main=str(path))
    elif case == "e3_kind":
        arm = "E3"
        _pair_checkpoint(path)
        policy = {"e3": str(path)}
    elif case == "classes":
        _pair_checkpoint(path, classes=("wide", "narrow"))
        pair = dict(pair, main=str(path))
    elif case == "band":
        arm = "E3"
        _e3_checkpoint(path, band="narrow")
        policy = {"e3": str(path)}
    elif case == "missing":
        pair = dict(pair, main=str(tmp_path / "gone.npz"))
    elif case == "not_npz":
        pair = dict(pair, main=str(tmp_path / "x.bin"))
    elif case == "no_manifest":
        _pair_checkpoint(path)
        manifest_path(path).unlink()
        pair = dict(pair, main=str(path))
    else:
        _pair_checkpoint(path)
        _edit_arrays(path)
        pair = dict(pair, main=str(path))
    drv = Exp4Driver(**_settings(ckpts, pair_checkpoints=pair, policy_checkpoints=policy))
    _refused_before_anything_is_built(drv, arm, match)


@pytest.mark.parametrize("arm", ["FQ", "E3"])
def test_a_checkpoint_removed_after_the_first_check_is_refused_as_a_value_error(arm, tmp_path):
    """``check_arm`` refuses as a ValueError only, which the runner reports as a
    usage error. Once the first check has verified the file, a later trial's
    load is the mule's own, whose loaders differ: the pair score's raises a
    CheckpointError for a missing file, E3's a FileNotFoundError. Both are
    refused as a ValueError before the trial builds anything."""
    path = tmp_path / f"{arm}.npz"
    if arm == "E3":
        _e3_checkpoint(path)
        drv = Exp4Driver(**dict(PILOT, policy_checkpoints={"e3": str(path)}))
    else:
        _pair_checkpoint(path)
        drv = Exp4Driver(**dict(PILOT, pair_checkpoints={"main": str(path)}))
    drv.check_arm(arm)
    path.unlink()
    _refused_before_anything_is_built(
        drv, arm, f"arm {arm}: its mule would refuse its checkpoint: .*not found")


@pytest.mark.parametrize("kw,match", [
    (dict(PILOT, pair_checkpoints={"g1": "x.npz"}), "'g1' is no learned arm's tag"),
    (dict(PILOT, pair_checkpoints={"e3": "x.npz"}), "'e3' is no learned arm's tag"),
    (dict(PILOT, policy_checkpoints={"main": "x.npz"}), "'main' is no learned arm's tag"),
    (dict(PILOT, policy_checkpoints={"E3": "x.npz"}), "'E3' is no learned arm's tag"),
    (dict(PILOT, pair_checkpoints={"main": ""}), r"pair_checkpoints\['main'\] must be"),
    (dict(PILOT, pair_checkpoints={"main": 3}), r"pair_checkpoints\['main'\] must be"),
    (dict(PILOT, pair_checkpoints=["main", "x.npz"]), "must be a mapping"),
    (dict(pair_checkpoints={"main": "x.npz"}),
     "pair_checkpoints: only on the simulated mission clock"),
    (dict(policy_checkpoints={"e3": "x.npz"}),
     "policy_checkpoints: only on the simulated mission clock"),
])
def test_the_driver_refuses_checkpoint_settings_no_arm_could_fly(kw, match):
    """A tag no learned arm flies, a path that is none, and checkpoints on the
    wall clock, where no learned arm runs: refused when the driver is built,
    before any file is read."""
    with pytest.raises(ValueError, match=match):
        Exp4Driver(**kw)


def test_the_driver_checks_no_training_state(tmp_path):
    """Other choices 6 and critic B9: the runner refuses an untrained
    checkpoint, not the driver, so FerrySim's bootstrap checkpoints fly."""
    pair, e3 = tmp_path / "boot.npz", tmp_path / "e3.npz"
    sha = _pair_checkpoint(pair, purpose="bootstrap", provenance=_bootstrap())
    e3_sha = _e3_checkpoint(e3, purpose="bootstrap", provenance=_bootstrap())
    drv = Exp4Driver(**dict(PILOT, pair_checkpoints={"main": str(pair)},
                            policy_checkpoints={"e3": str(e3)}))
    for arm, prefix, want in (("FQ", "pair", sha), ("E3", "policy", e3_sha)):
        drv.check_arm(arm)
        _row, topo = T.run_stub_trial(drv, _cell(arm))
        assert getattr(topo.mules[0], f"{prefix}_checkpoint_sha256") == want
        assert verify_checkpoint(e3 if arm == "E3" else pair)["purpose"] == "bootstrap"


def test_a_checkpoint_inside_the_repo_travels_repo_relative(monkeypatch, tmp_path):
    """Other choices 6 (critic A10 (iii)): the driver writes a checkpoint path
    relative to the repository root when the file lies inside it, with "/"
    separators, however it was given: a string or a ``Path``, absolute or
    relative to the working directory, from the root, below it or outside
    it. Any other file is written as its resolved absolute path, not "as
    given" as other choices 6 says, since the mule reads every relative path
    under that root, whatever its working directory; the driver's check reads
    each back as the mule will."""
    repo = tmp_path / "repo"
    inside = repo / "results" / "exp5" / "checkpoints" / "s55" / "main" / "g0.9_s1.npz"
    outside = tmp_path / "elsewhere" / "e3.npz"
    sha = _pair_checkpoint(inside)
    e3_sha = _e3_checkpoint(outside)
    repo = repo.resolve()
    monkeypatch.setattr(mule_process, "REPO_ROOT", repo)
    written = "results/exp5/checkpoints/s55/main/g0.9_s1.npz"
    # (the working directory, the pair path as given, E3's as given)
    for cwd, pair, e3 in (
            (tmp_path, str(inside), str(outside)),
            (tmp_path, inside, outside),
            (repo, written, "../elsewhere/e3.npz"),
            (repo / "results", Path("exp5/checkpoints/s55/main/g0.9_s1.npz"),
             "../../elsewhere/e3.npz"),
            (outside.parent, f"../repo/{written}", "e3.npz")):
        monkeypatch.chdir(cwd)
        drv = Exp4Driver(**dict(PILOT, pair_checkpoints={"main": pair},
                                policy_checkpoints={"e3": e3}))
        assert drv.checkpoint_settings("FQ") == {
            "pair_checkpoint": written, "pair_checkpoint_sha256": sha,
            "pair_checkpoint_tag": "main"}, (cwd, pair)
        assert drv.checkpoint_settings("E3") == {
            "policy_checkpoint": str(outside.resolve()), "policy_checkpoint_sha256": e3_sha,
            "policy_checkpoint_tag": "e3"}, (cwd, e3)
        drv.check_arm("FQ")
        drv.check_arm("E3")
    assert drv.checkpoint_settings("FX") == drv.checkpoint_settings("H1+L1") == {}
    monkeypatch.chdir(tmp_path)
    assert mule_process.checkpoint_path("results/exp5/a.npz") == repo / "results/exp5/a.npz"
    assert mule_process.checkpoint_path(str(outside)) == outside
    for nothing in (None, "", "  "):
        with pytest.raises(ValueError, match="never a random one"):
            mule_process.checkpoint_path(nothing)


# --------------------------------------------------------------------------- #
# FQ is F (critic A5)
# --------------------------------------------------------------------------- #

def _plain(config, drop=()):
    return {k: v for k, v in dataclasses.asdict(config).items() if k not in drop}


@pytest.mark.parametrize("arm", PAIR_ARMS)
def test_an_fq_arms_config_is_fs_but_the_slot_its_checkpoint_and_its_ablation(arm, ckpts):
    """Other choices 5: each FQ arm's mule config equals F's except for
    ``flight_slot``, the checkpoint fields and its ablation's own field (FQ-cov
    F-cov's score, FQ-dwell the dwell out of Δ). The per-trial RF token names
    the arm, so it differs for every arm (Amendment 10); the cluster and the
    devices are the paired trial's."""
    drv = Exp4Driver(**_settings(ckpts, plan_score_params={"c_energy": 0.0}))
    f_row, f_topo = T.run_stub_trial(drv, _cell("F"))
    q_row, q_topo = T.run_stub_trial(drv, _cell(arm))
    (f,), (q,) = f_topo.mules, q_topo.mules
    own = {"flight_slot", "rf_link_token", *PAIR_CHECKPOINT_FIELDS}
    if arm in ("FQ-cov", "FQ-dwell"):
        own.add("plan_score_params")
    assert _plain(q, own) == _plain(f, own)
    assert (f.flight_slot, q.flight_slot) == ("committed", "pair_q")
    tag = CHECKPOINT_TAGS[arm]
    assert (q.pair_checkpoint, q.pair_checkpoint_sha256, q.pair_checkpoint_tag) == (
        str(Path(ckpts.paths[tag]).resolve()), ckpts.shas[tag], tag)
    assert all(getattr(f, name) is None for name in PAIR_CHECKPOINT_FIELDS
               + POLICY_CHECKPOINT_FIELDS)
    change = {"FQ-cov": F_COV_SCORE, "FQ-dwell": FQ_DWELL_SCORE}.get(arm, {})
    assert q.plan_score_params == {**f.plan_score_params, **change}
    assert (q.miss_priority, q.member_admission, q.replan_fallback) == (True, "subset", "trim")
    assert dataclasses.asdict(q_topo.cluster) == dataclasses.asdict(f_topo.cluster)
    assert [_plain(d, {"rf_link_token"}) for d in q_topo.devices] == [
        _plain(d, {"rf_link_token"}) for d in f_topo.devices]
    # The rows differ only in ferry_params' own keys.
    assert {k: v for k, v in q_row.items() if k != "ferry_params"} == {
        k: v for k, v in f_row.items() if k != "ferry_params"}
    fp, qp = json.loads(f_row["ferry_params"]), json.loads(q_row["ferry_params"])
    added = {"pair_tag": tag, "pair_sha256": ckpts.shas[tag], "flight_slot": "pair_q",
             "plan_score_params": q.plan_score_params}
    assert qp == {**fp, **added}


# --------------------------------------------------------------------------- #
# H1+L1 (decision 8 (a); critic A6)
# --------------------------------------------------------------------------- #

def test_h1_l1_is_h1_with_h3s_adaptive_backhaul_on_the_seconds_model():
    """On the simulated clock's seconds-axis backhaul, H1+L1's mule is H1's
    with ``backhaul_policy='adaptive'``, which is H3's without the RL
    selector; its row differs from H1's only in that setting."""
    drv = Exp4Driver(**dict(P3.PILOT, backhaul_model="seconds", t_nom_layouts=4))
    rows, mules = {}, {}
    for arm in ("H1", "H3", "H1+L1"):
        rows[arm], topo = T.run_stub_trial(drv, _cell(arm))
        (mules[arm],) = topo.mules
    h1, h3, hl = mules["H1"], mules["H3"], mules["H1+L1"]
    assert (h1.backhaul_policy, h3.backhaul_policy, hl.backhaul_policy) == (
        "fixed", "adaptive", "adaptive")
    assert (h1.use_rl_selector, h3.use_rl_selector, hl.use_rl_selector) == (False, True, False)
    assert _plain(hl, {"backhaul_policy", "rf_link_token"}) == _plain(
        h1, {"backhaul_policy", "rf_link_token"})
    assert _plain(hl, {"use_rl_selector", "rf_link_token"}) == _plain(
        h3, {"use_rl_selector", "rf_link_token"})
    assert json.loads(rows["H1+L1"]["ferry_params"]) == dict(
        json.loads(rows["H1"]["ferry_params"]), backhaul_policy="adaptive")
    assert rows["H1+L1"]["ferry_params"] == rows["H3"]["ferry_params"]
    assert rows["H1+L1"]["policy_params"] == rows["H1"]["policy_params"] == ""


@pytest.mark.parametrize("clock", ["wall", "sim"])
def test_h1_l1_flies_h3s_adaptive_loss_schedule_on_the_l1_channel(clock):
    """With ``--l1-channel`` (EX-4.3) H1+L1's cluster draws H3's adaptive loss
    schedule and its mule gets H3's RF prior (the trial's mean on the wall
    clock, the per-mission schedule on the simulated one, critic B4), where
    H1 holds the best fixed band; H1+L1 still has no RL selector. The seed is
    the goldens' H3 L1 trial's, where the two schedules differ."""
    kw = dict(realism=True, l1_channel=True)
    if clock == "sim":
        kw.update(mission_clock="sim", contact_band="wide", in_flight_response="replan",
                  t_nom_layouts=4)
    drv = Exp4Driver(**kw)
    topos = {}
    for arm in ("H1", "H3", "H1+L1"):
        _row, topos[arm] = T.run_stub_trial(drv, _cell(arm, seed=2191267877))
    schedule = {arm: topos[arm].cluster.backhaul_loss_schedule for arm in topos}
    assert schedule["H1+L1"] == schedule["H3"] != schedule["H1"]
    prior = "rf_prior_schedule_db" if clock == "sim" else "rf_prior_snr_db"
    priors = {arm: getattr(topos[arm].mules[0], prior) for arm in topos}
    assert priors["H1+L1"] == priors["H3"] != priors["H1"]
    assert topos["H1+L1"].mules[0].use_rl_selector is False


@pytest.mark.parametrize("kw", [
    dict(),
    dict(mission_clock="sim", contact_band="wide"),
], ids=["wall", "sim-mission-backhaul"])
def test_h1_l1_is_refused_where_it_would_fly_as_h1(kw):
    _refused_before_anything_is_built(Exp4Driver(**kw), "H1+L1", "here it would fly as H1")


# --------------------------------------------------------------------------- #
# Provenance (other choices 6; critic B10)
# --------------------------------------------------------------------------- #

OTHER_ARMS = ("F", "FX", "FB+narrow", "F-cov", "F-cap", "F-prio", "H1", "H2", "H3", "D1",
              "D3", "D4", "D5")


@pytest.mark.parametrize("arm", PHASE_5_ARMS + OTHER_ARMS)
def test_provenance_names_a_checkpoint_only_for_the_phase_5_arms(arm, ckpts):
    """An FQ arm's ``ferry_params`` gains ``pair_tag`` and ``pair_sha256`` and
    E3's ``policy_params`` is ``policy_tag`` and ``policy_sha256``, from the
    mule config as the scorer reads the per-role JSON; no row names a
    checkpoint's path, and every other arm's strings are as before."""
    drv = Exp4Driver(**_settings(ckpts, backhaul_model="seconds"))
    row, topo = T.run_stub_trial(drv, _cell(arm))
    (mule,) = topo.mules
    params = json.loads(row["ferry_params"])
    learned = {k: v for k, v in params.items() if k.startswith(("pair_", "policy_"))}
    assert not [v for v in row.values() if isinstance(v, str) and _mentions(v, ckpts.root)]
    tag = CHECKPOINT_TAGS.get(arm)
    if arm in PAIR_ARMS:
        assert learned == {"pair_tag": tag, "pair_sha256": ckpts.shas[tag]}
        assert (mule.pair_checkpoint_tag, mule.pair_checkpoint_sha256) == (tag, ckpts.shas[tag])
        assert row["policy_params"] == "" and row["contact_band"] == "search"
        assert {f: params[f] for f in PLAN_MULE_FIELDS} == {f: getattr(mule, f)
                                                            for f in PLAN_MULE_FIELDS}
        return
    assert learned == {}
    if arm == "E3":
        assert row["policy_params"] == json.dumps(
            {"policy_sha256": ckpts.shas["e3"], "policy_tag": "e3"}, sort_keys=True)
        assert mule.contact_policy == "chen_dqn" and mule.member_admission == "whole"
        assert (mule.policy_checkpoint_tag, mule.policy_checkpoint_sha256) == (
            "e3", ckpts.shas["e3"])
        return
    expected = {"D3": json.dumps({"variant": "expected", "weights": "uniform"}, sort_keys=True),
                "D5": json.dumps({"value": "unit"}, sort_keys=True)}.get(arm, "")
    assert row["policy_params"] == expected
    assert all(getattr(mule, f) is None for f in PAIR_CHECKPOINT_FIELDS + POLICY_CHECKPOINT_FIELDS)


@pytest.mark.parametrize("face,name", [("p3", n) for n in P3.TRIAL_NAMES]
                         + [("p4", n) for n in UG5.TRIAL_NAMES])
def test_the_recorded_rows_keep_their_provenance_strings(face, name):
    """Phase 3's simulated-clock oracles (6e6f92d) and Phase 4's plan arms
    (386c275), re-run: every provenance column reads as recorded."""
    module = P3 if face == "p3" else UG5
    golden, case = module.load_golden()["cases"][name]["row"], module.capture(name)["row"]
    assert {c: case[c] for c in PROVENANCE_COLUMNS} == {c: golden[c] for c in PROVENANCE_COLUMNS}


def test_the_shared_provenance_helpers_read_a_config_or_a_kept_trace():
    """``plan_ferry_params`` and ``learned_policy_params`` take a mule config as
    a mapping (the scorer's per-role JSON, possibly from before Phase 5): a
    missing key is the recorded default, and only the learned fillings add."""
    fq, f = dataclasses.asdict(_fq()), dataclasses.asdict(_plan_mule())
    assert plan_ferry_params(fq) == dict({k: fq[k] for k in PLAN_MULE_FIELDS},
                                         pair_tag="main", pair_sha256=SHA)
    assert plan_ferry_params(f) == {k: f[k] for k in PLAN_MULE_FIELDS}
    old = {k: v for k, v in f.items() if k not in C.CHECKPOINT_MULE_FIELDS}
    assert plan_ferry_params(old) == plan_ferry_params(f)
    e3 = dataclasses.asdict(_sim_mule(contact_band="wide", contact_policy="chen_dqn", **POLICY))
    assert learned_policy_params(e3) == {"policy_tag": "e3", "policy_sha256": SHA}
    for other in (f, fq, old, {}, dataclasses.asdict(_sim_mule(contact_policy="whittle"))):
        assert learned_policy_params(other) == {}


# --------------------------------------------------------------------------- #
# The mule process
# --------------------------------------------------------------------------- #

class _Events:
    def __init__(self):
        self.lines = []

    def emit(self, event, **fields):
        json.dumps(fields)                   # as the JSONL emitter serialises
        self.lines.append((event, fields))

    def close(self):
        return

    def named(self, event):
        return [f for e, f in self.lines if e == event]


@pytest.fixture
def service(monkeypatch):
    """A ``MuleService`` factory (a real dock server, the bootstrap skipped) that
    records the keywords each supervisor was built with."""
    server = TCPDockLinkServer(host="127.0.0.1", port=0)
    server.start()
    made, built = [], []
    real = mule_process.MuleSupervisor

    class _Recorded(real):
        def __init__(self, **kwargs):
            built.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(mule_process, "MuleSupervisor", _Recorded)

    def _make(cfg: MuleConfig):
        events = _Events()
        cfg = dataclasses.replace(cfg, dock_port=server.port)
        svc = mule_process.MuleService(cfg, events=events)
        svc.supervisor.wait_for_initial_dock = lambda timeout=None: True
        made.append(svc)
        return svc, events, built[-1]

    yield _make
    for svc in made:
        svc.shutdown()
    server.close()


def _fq_on(ckpts, tag="main", **kw):
    return _fq(**dict(dict(pair_checkpoint=ckpts.paths[tag], pair_checkpoint_sha256=ckpts.shas[tag],
                           pair_checkpoint_tag=tag), **kw))


def _e3_on(ckpts, **kw):
    return _sim_mule(**dict(dict(contact_band="wide", contact_policy="chen_dqn",
                                 policy_checkpoint=ckpts.paths["e3"],
                                 policy_checkpoint_sha256=ckpts.shas["e3"],
                                 policy_checkpoint_tag="e3"), **kw))


def test_a_pair_q_mule_flies_the_slot_its_checkpoint_builds(service, ckpts):
    """Its supervisor gets the pair slot around the verified checkpoint's
    score; ``mule_ready`` adds exactly ``pair``, the manifest's provenance and
    the config's tag, to F's key set."""
    svc, events, kwargs = service(_fq_on(ckpts, "g25"))
    slot = kwargs["pair_slot"]
    assert isinstance(slot, PairQSlot) and svc.supervisor._flight_slot is slot
    manifest = slot.scorer.manifest
    assert manifest == verify_checkpoint(ckpts.paths["g25"])
    assert manifest["sha256"] == ckpts.shas["g25"] and list(manifest["classes"]) == list(LINK)
    (ready,) = events.named("mule_ready")
    assert ready["pair"] == json.loads(json.dumps(dict(manifest_provenance(manifest), tag="g25")))
    assert ready["flight_slot"] == "pair_q" and not _mentions(json.dumps(ready), ckpts.root)
    _f, f_events, f_kwargs = service(_plan_mule())
    assert "pair_slot" not in f_kwargs
    (f_ready,) = f_events.named("mule_ready")
    assert set(ready) == set(f_ready) | {"pair"}


def test_a_pair_q_mule_reads_its_checkpoint_over_its_own_link(service, tmp_path):
    """The score is read over the mule's link classes in link order (here a
    two-class link), so a checkpoint trained on that link flies there."""
    path = tmp_path / "two.npz"
    sha = _pair_checkpoint(path, classes=("wide", "narrow"))
    cfg = _fq(pair_checkpoint=str(path), pair_checkpoint_sha256=sha,
              contact_band_classes=["wide", "narrow"])
    _svc, events, kwargs = service(cfg)
    assert kwargs["pair_slot"].scorer.schema.classes == ("wide", "narrow")
    (ready,) = events.named("mule_ready")
    assert ready["pair"]["classes"] == ["wide", "narrow"]


def test_an_e3_mule_flies_its_checkpoint(service, ckpts, monkeypatch):
    """Its scheduler's policy is E3 from the verified checkpoint, on the
    contact band, read once, before the process binds anything, and flown as
    read; ``mule_ready`` adds exactly ``policy_checkpoint`` to H1's."""
    order, load, rf_server = [], ChenDQNPolicy.from_checkpoint, mule_process.TCPRFLinkServer
    monkeypatch.setattr(ChenDQNPolicy, "from_checkpoint",
                        lambda *a, **k: order.append(load(*a, **k)) or order[-1])
    monkeypatch.setattr(mule_process, "TCPRFLinkServer",
                        lambda *a, **k: order.append("bind") or rf_server(*a, **k))
    svc, events, kwargs = service(_e3_on(ckpts))
    policy = svc.supervisor.scheduler.target_selector
    assert len(order) == 2 and order[0] is policy and order[1] == "bind"
    assert isinstance(policy, ChenDQNPolicy) and kwargs["target_selector"] is policy
    assert policy.band == "wide" and policy.manifest["sha256"] == ckpts.shas["e3"]
    assert "pair_slot" not in kwargs and getattr(svc.supervisor, "_flight_slot", None) is None
    (ready,) = events.named("mule_ready")
    assert ready["policy_checkpoint"] == json.loads(json.dumps(
        dict(manifest_provenance(policy.manifest), tag="e3")))
    _h, h_events, _k = service(_sim_mule(contact_band="wide"))
    (h_ready,) = h_events.named("mule_ready")
    assert set(ready) == set(h_ready) | {"policy_checkpoint"}


@pytest.mark.parametrize("case", ["sha", "missing", "not_npz", "pair_kind", "classes",
                                  "e3_band", "e3_sha", "e3_missing"])
def test_the_mule_refuses_a_checkpoint_it_may_not_fly(case, ckpts, tmp_path, monkeypatch):
    """Other choices 6: the mule verifies the sha, the kind, the schema and the
    classes, and refuses to run on any failure, before it binds anything."""
    bound = []
    monkeypatch.setattr(mule_process, "TCPRFLinkServer",
                        lambda *a, **k: bound.append(a) or pytest.fail("bound an RF port"))
    path = tmp_path / "c.npz"
    if case == "sha":
        cfg = _fq_on(ckpts, pair_checkpoint_sha256=ckpts.shas["g0"])
    elif case == "missing":
        cfg = _fq_on(ckpts, pair_checkpoint=str(tmp_path / "gone.npz"))
    elif case == "not_npz":
        cfg = _fq_on(ckpts, pair_checkpoint=str(tmp_path / "c.bin"))
    elif case == "pair_kind":
        cfg = _fq_on(ckpts, pair_checkpoint=ckpts.paths["e3"],
                     pair_checkpoint_sha256=ckpts.shas["e3"])
    elif case == "classes":
        sha = _pair_checkpoint(path, classes=("wide", "narrow"))
        cfg = _fq_on(ckpts, pair_checkpoint=str(path), pair_checkpoint_sha256=sha)
    elif case == "e3_band":
        cfg = _e3_on(ckpts, contact_band="narrow")
    elif case == "e3_sha":
        cfg = _e3_on(ckpts, policy_checkpoint_sha256=ckpts.shas["main"])
    else:
        cfg = _e3_on(ckpts, policy_checkpoint=str(tmp_path / "gone.npz"))
    assert mule_config_errors(cfg) == []
    field = "pair_checkpoint" if cfg.flight_slot == "pair_q" else "policy_checkpoint"
    with pytest.raises(mule_process.CheckpointRefused, match=f"mule m: {field}="):
        mule_process.MuleService(cfg, events=_Events())
    assert bound == []


@pytest.mark.parametrize("cfg", [
    MuleConfig(mule_id="m-wall", rf_range_m=60.0),
    _sim_mule(mule_id="m-h1", contact_band="wide"),
    _sim_mule(mule_id="m-d4", contact_band="wide", contact_policy="fedex"),
    _plan_mule(mule_id="m-f"),
    _plan_mule(mule_id="m-fx", flight_slot="cross_heuristic"),
], ids=["wall", "sim-H1", "sim-D4", "F", "FX"])
def test_a_recorded_mule_announces_no_checkpoint(service, cfg):
    """Freeze Rule 1: no pair slot reaches its supervisor and ``mule_ready``
    gains neither ``pair`` nor ``policy_checkpoint``."""
    svc, events, kwargs = service(cfg)
    assert "pair_slot" not in kwargs
    assert not isinstance(getattr(svc.supervisor, "_flight_slot", None), PairQSlot)
    (ready,) = events.named("mule_ready")
    assert not set(READY_FIELDS) & set(ready)


def _result(clock, rnd, **kw) -> MissionRunResult:
    start = clock()
    return MissionRunResult(
        mission_round=rnd, empty=True, sim_start_s=start, sim_end_s=start + 30.0,
        sim_ledger={"turnaround": 30.0}, pass_1_flown=[], pass_2_flown=[], replans=[],
        aborts=[], inserts=[], offers_refused=[], energy_j=100.0, band="narrow",
        pass_1_preflight_drops=[], **kw)


def test_mission_completed_carries_the_learned_records_only_when_a_mission_has_them(service):
    """Other choices 9: ``pass_1_pairs``, ``pass_1_e3`` and
    ``pass_1_e3_unvisited`` after the Phase 4 fields and before
    ``energy_status``, each left out when None or empty, so a mission without
    them keeps its key set; their numbers become JSON numbers."""
    svc, events = service(_sim_mule(contact_band="wide", n_missions=5))[:2]
    clock = svc.supervisor.mission_clock
    pairs = [{"t_s": np.float64(1.0e6 + 12.5), "devices": ("d0",), "q": [np.float64(0.25)],
              "w": [np.float32(10.0)]}]
    calls = [{"t_s": 1.0e6, "after_stop": False, "next_index": np.int64(0), "next": ["d0"]}]
    left = [{"position": (1.0, 2.0, 0.0), "devices": ["d1"], "deadline_ts": 1.0e6 + 99.0,
             "widened": False}]
    results = iter([
        _result(clock, 1),
        _result(clock, 2, pass_1_pairs=pairs),
        _result(clock, 3, pass_1_e3=calls, pass_1_e3_unvisited=left),
        _result(clock, 4, pass_1_e3=calls),
        _result(clock, 5, pass_1_pairs=[], pass_1_e3=[], pass_1_e3_unvisited=[]),
    ])
    svc.supervisor.run_one_mission = lambda: next(results)
    svc.run()
    assert svc.exit_code == 0
    plain, paired, e3, e3_all_flown, empty = events.named("mission_completed")
    assert set(empty) == set(plain) and not set(MISSION_FIELDS) & set(plain)
    assert set(paired) == set(plain) | {"pass_1_pairs"}
    assert list(paired)[-2:] == ["pass_1_pairs", "energy_status"]
    assert paired["pass_1_pairs"] == [{"t_s": 1.0e6 + 12.5, "devices": ["d0"], "q": [0.25],
                                       "w": [10.0]}]
    assert type(paired["pass_1_pairs"][0]["q"][0]) is float
    assert set(e3) == set(plain) | {"pass_1_e3", "pass_1_e3_unvisited"}
    assert list(e3)[-3:] == ["pass_1_e3", "pass_1_e3_unvisited", "energy_status"]
    assert e3["pass_1_e3_unvisited"] == [dict(left[0], position=[1.0, 2.0, 0.0])]
    assert type(e3["pass_1_e3"][0]["next_index"]) is int
    assert set(e3_all_flown) == set(plain) | {"pass_1_e3"}


# --------------------------------------------------------------------------- #
# Stub trials of FQ and E3, through the driver and the real services in process
# --------------------------------------------------------------------------- #

def _in_process(settings, cell):
    """(row, events by name, orchestrator) of one trial run through the real
    services in this process (UG4's harness, ``tests/golden/_build_p3_sim.py``)."""
    driver = Exp4Driver(**settings)
    with P3.in_process_orchestrator():
        P3.InProcessOrchestrator.last = None
        row = dict(driver.run_trial(cell))
        orch = P3.InProcessOrchestrator.last
    assert orch.exit_codes == {"mule-exp4-mule": 0}
    by_name = {}
    for f, text in sorted(orch.files.items()):
        if f.startswith("mule-") and f.endswith(".jsonl"):
            for line in text.splitlines():
                if line.strip():
                    e = json.loads(line)
                    by_name.setdefault(e["event"], []).append(e)
    return row, by_name, orch


def test_fq_flies_a_stub_trial_from_its_checkpoint(ckpts):
    """FQ at N = 12, 120 s, S = 3 (Study 5.5's decision-rich cell, UG5's FX
    trial): every mission completes; the per-role JSON names the checkpoint;
    ``mule_ready`` and each mission with a Pass-1 stop add exactly ``pair`` and
    ``pass_1_pairs`` to FX's key sets on the same cell, with one closed
    decision record per Pass-1 stop flown, scored by the learned score."""
    settings = _settings(ckpts, mission_budget_s=120.0, age_cap_missions=3)
    row, events, orch = _in_process(settings, UG5._cell("FQ", UG5.N12_SEED, N=12,
                                                         n_missions=4))
    _fx_row, fx_events, _o = _in_process(settings, UG5._cell("FX", UG5.N12_SEED, N=12,
                                                             n_missions=4))
    assert (row["missions_completed"], row["mission_failures"]) == (4, 0)
    mule = json.loads(orch.files["mule-exp4-mule.json"])
    assert (mule["flight_slot"], mule["pair_checkpoint_tag"], mule["pair_checkpoint_sha256"]) == (
        "pair_q", "main", ckpts.shas["main"])
    assert mule["pair_checkpoint"] == str(Path(ckpts.paths["main"]).resolve())
    (ready,), (fx_ready,) = events["mule_ready"], fx_events["mule_ready"]
    assert set(ready) == set(fx_ready) | {"pair"}
    assert (ready["pair"]["tag"], ready["pair"]["sha256"]) == ("main", ckpts.shas["main"])
    fx_keys = set(fx_events["mission_completed"][0])
    flown = 0
    for e in events["mission_completed"]:
        stops = e["pass_1_flown"]
        flown += len(stops)
        assert not {"pass_1_e3", "pass_1_e3_unvisited"} & set(e)
        if not stops:
            assert set(e) == fx_keys
            continue
        assert set(e) == fx_keys | {"pass_1_pairs"}
        records = e["pass_1_pairs"]
        assert [r["devices"] for r in records] == [s["devices"] for s in stops]
        for r in records:
            assert set(r) == set(DECISION_KEYS + CLOSE_KEYS) and r["scorer"] == "pair_v1"
            assert r["q"] is not None
        assert [r["terminal"] for r in records] == [False] * (len(records) - 1) + [True]
    assert flown > 4                                   # several stops a sortie at N = 12
    params = json.loads(row["ferry_params"])
    assert (params["pair_tag"], params["pair_sha256"]) == ("main", ckpts.shas["main"])


def test_e3_flies_a_stub_trial_from_its_checkpoint(ckpts):
    """E3 on the Phase 3 pilots' flags (wide, 60 s, legacy mode): every mission
    completes; ``mule_ready`` adds exactly ``policy_checkpoint`` to H1's, and a
    mission adds only E3's own records to H1's key set; E3 admits every
    contact, so no policy drop is reported; the row's ``policy_params`` names
    the checkpoint."""
    settings = dict(P3.PILOT, t_nom_layouts=5, policy_checkpoints=ckpts.policy())
    row, events, orch = _in_process(settings, P3._cell("E3", P3.SEED))
    _h_row, h_events, _o = _in_process(settings, P3._cell("H1", P3.SEED))
    assert (row["missions_completed"], row["mission_failures"]) == (4, 0)
    mule = json.loads(orch.files["mule-exp4-mule.json"])
    assert (mule["contact_policy"], mule["policy_checkpoint_tag"]) == ("chen_dqn", "e3")
    (ready,), (h_ready,) = events["mule_ready"], h_events["mule_ready"]
    assert set(ready) == set(h_ready) | {"policy_checkpoint"}
    assert ready["policy_checkpoint"]["sha256"] == ckpts.shas["e3"]
    h_keys = set(h_events["mission_completed"][0]) - {"pass_1_policy_drops"}
    for e in events["mission_completed"]:
        assert "pass_1_policy_drops" not in e and "pass_1_pairs" not in e
        assert set(e) - {"pass_1_e3", "pass_1_e3_unvisited"} == h_keys and "pass_1_e3" in e
        named = [c for c in e["pass_1_e3"] if c["next"] != "home"]
        assert [c["next"] for c in named] == [s["devices"] for s in e["pass_1_flown"]]
    assert row["policy_params"] == json.dumps(
        {"policy_sha256": ckpts.shas["e3"], "policy_tag": "e3"}, sort_keys=True)


@pytest.mark.parametrize("given", ["absolute", "relative"])
def test_a_checkpoint_inside_the_repo_flies_from_its_repo_relative_path(
        given, monkeypatch, tmp_path):
    """The mule reads the driver's repo-relative path under the repository root,
    whatever its working directory (other choices 6): here outside the
    repository with the path given absolute, or its ``results`` folder with
    the path given relative to it, where reading from the working directory
    would find nothing."""
    repo = tmp_path / "repo"
    path = repo / "results" / "exp5" / "checkpoints" / "s55" / "main" / "g0.9_s0.npz"
    _pair_checkpoint(path)
    monkeypatch.setattr(mule_process, "REPO_ROOT", repo)
    if given == "absolute":
        monkeypatch.chdir(tmp_path)
        value = str(path)
    else:
        monkeypatch.chdir(repo / "results")
        value = "exp5/checkpoints/s55/main/g0.9_s0.npz"
    row, events, orch = _in_process(dict(PILOT, pair_checkpoints={"main": value}),
                                    UG5._cell("FQ", UG5.SEED, n_missions=2))
    mule = json.loads(orch.files["mule-exp4-mule.json"])
    assert mule["pair_checkpoint"] == "results/exp5/checkpoints/s55/main/g0.9_s0.npz"
    assert row["missions_completed"] == 2 and events["mule_ready"][0]["pair"]["tag"] == "main"


def test_a_checkpoint_rewritten_after_the_first_check_is_refused(tmp_path, monkeypatch):
    """Every trial's config names the sha the driver verified first, so its
    provenance cannot move under a campaign: a file rewritten since is refused
    before the next trial builds anything, and, were the driver's check
    skipped, by the trial's mule, which then does not fly."""
    path = tmp_path / "main.npz"
    first = _pair_checkpoint(path, seed=1)
    driver = Exp4Driver(**dict(PILOT, pair_checkpoints={"main": str(path)}))
    driver.check_arm("FQ")
    _pair_checkpoint(path, seed=2)                     # another network, another sha
    assert driver.checkpoint_settings("FQ")["pair_checkpoint_sha256"] == first
    _refused_before_anything_is_built(driver, "FQ", "not the expected")
    monkeypatch.setattr(Exp4Driver, "_check_learned", lambda self, arm, cfg, spec: None)
    with P3.in_process_orchestrator():
        P3.InProcessOrchestrator.last = None
        with pytest.raises(mule_process.CheckpointRefused, match="not the expected"):
            driver.run_trial(UG5._cell("FQ", UG5.SEED, n_missions=2))
        assert P3.InProcessOrchestrator.last is not None        # the trial had started


# --------------------------------------------------------------------------- #
# Rule 1 at the defaults
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("face,name", [("p3", n) for n in P3.TRIAL_NAMES]
                         + [("p4", n) for n in UG5.TRIAL_NAMES])
def test_no_recorded_trial_gains_a_phase_5_field(face, name):
    """UG4's and UG5's oracles let an added key pass, so the absence is pinned
    here: no ``mule_ready`` gains ``pair`` or ``policy_checkpoint``, no
    ``mission_completed`` a decision record, no row a checkpoint key, and the
    per-role mule JSON holds the six checkpoint fields at None."""
    case = (P3 if face == "p3" else UG5).capture(name)
    for events in case["mule_ready"].values():
        assert events and all(not set(READY_FIELDS) & set(e) for e in events)
    for events in case["mission_completed"].values():
        assert events and all(not set(MISSION_FIELDS) & set(e) for e in events)
    row = json.dumps(case["row"])
    assert not [k for k in ("pair_tag", "pair_sha256", "policy_tag", "policy_sha256")
                if k in row]
    mule = case["configs"]["mule-exp4-mule.json"]
    assert {f: mule[f] for f in C.CHECKPOINT_MULE_FIELDS} == dict.fromkeys(
        C.CHECKPOINT_MULE_FIELDS)


def test_a_recorded_path_loads_no_phase_5_module():
    """Other choices 13: in a fresh interpreter the runner, stub trials of H1,
    D1 and D4 (both clocks) and of F and FX through the driver, and their
    mules' processes load none of the Phase 5 modules."""
    code = r"""
import sys
from experiments.exp4 import runner_main  # noqa: F401
from experiments.exp4.driver import Exp4Driver
from experiments.runner import Cell
from hermes.processes import mule as mule_process
from hermes.processes.config import MuleConfig
from hermes.transport import TCPDockLinkServer
from tests.golden import _build_topology as T

p = {"N": 4, "rrf": 60.0, "n_missions": 2, "regime": "clean"}
cid = "|".join(f"{k}={v}" for k, v in sorted(p.items()))
pilot = dict(mission_clock="sim", realism=True, contact_band="wide", deadline_time_scale="t_nom",
             in_flight_response="replan", replan_fallback="trim", aggregation="agg:cutoff",
             contact_reliability_source="channel", payload_bytes=1_000_000,
             mission_budget_s=45.0, age_cap_missions=2, t_nom_layouts=2)
for kw, arms in ((dict(), ("H1", "D1", "D4")), (pilot, ("H1", "D1", "D4", "F", "FX"))):
    for arm in arms:
        T.run_stub_trial(Exp4Driver(**kw), Cell(cid, arm, 0, 7, p))
plan = dict(mission_clock="sim", trial_seed=1, n_missions=1, contact_band="wide",
            plan_mode="ferry", t_nom_s=200.0, in_flight_response="replan",
            replan_fallback="trim", member_admission="subset")
server = TCPDockLinkServer(host="127.0.0.1", port=0)
server.start()
for kw in (dict(), dict(mission_clock="sim", trial_seed=1, n_missions=1, contact_band="wide"),
           dict(mission_clock="sim", trial_seed=1, n_missions=1, contact_band="wide",
                contact_policy="fedex"), plan, dict(plan, flight_slot="cross_heuristic")):
    svc = mule_process.MuleService(MuleConfig(mule_id="m", rf_range_m=60.0,
                                              dock_port=server.port, **kw))
    svc.shutdown()
server.close()
names = %r
print(sorted(m for m in sys.modules if m.startswith("experiments.ferrysim")
             or m.rsplit(".", 1)[-1] in names))
""" % (PHASE_5_MODULES,)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                         text=True, timeout=240)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #

def _runner(monkeypatch, argv):
    from experiments.exp4 import runner_main

    captured, grids = {}, []

    class _Driver(Exp4Driver):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    class _Runner:
        def __init__(self, *args, **kwargs):
            return

        def run(self, run_trial):
            return 0

    real = runner_main._build_grid
    monkeypatch.setattr(runner_main, "Exp4Driver", _Driver)
    monkeypatch.setattr(runner_main, "TrialRunner", _Runner)
    monkeypatch.setattr(runner_main, "_build_grid", lambda **kw: grids.append(kw) or real(**kw))
    assert runner_main.main(argv) == 0
    return captured, grids[0]


def _refused(monkeypatch, capsys, argv) -> str:
    """The message of the usage error (exit code 2) the runner gives ``argv``.

    argparse prints the usage before it, which names every flag, so a test
    reads the error line alone, not the whole of stderr."""
    capsys.readouterr()
    with pytest.raises(SystemExit) as exit_:
        _runner(monkeypatch, argv)
    assert exit_.value.code == 2
    prefix = "experiments.exp4.runner_main: error: "
    (line,) = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith(prefix)]
    return line[len(prefix):]


SIM_FLAGS = ["--mission-clock", "sim", "--contact-band", "wide", "--in-flight-response",
             "replan", "--replan-fallback", "trim"]

#: The runner's checkpoint flags: (the tag a value names, the tag the driver
#: gets, the arm that flies it, the kind the flag takes, the other kind).
RUNNER_FLAGS = {
    "--pair-checkpoint": ("main", "main", "FQ", KIND_PAIR_Q, KIND_CHEN_DQN),
    "--policy-checkpoint": ("E3", "e3", "E3", KIND_CHEN_DQN, KIND_PAIR_Q),
}


def _flag_checkpoint(flag, path, *, purpose="trained", **changes):
    """A checkpoint of the kind the runner's ``flag`` takes, its provenance a
    trained one's (``_trained``) or a bootstrap's (``_bootstrap``) with
    ``changes``; E3's reward is its own, |C_k| / N."""
    provenance = (_bootstrap if purpose == "bootstrap" else _trained)(**changes)
    if flag == "--policy-checkpoint":
        return _e3_checkpoint(path, purpose=purpose,
                              provenance=dict(provenance, reward={"kind": "bytes"}))
    return _pair_checkpoint(path, purpose=purpose, provenance=provenance)


def test_the_runner_defaults_are_the_recorded_run(monkeypatch, tmp_path):
    """No checkpoint, no refusal: H2 and H3 run random-init as recorded."""
    kwargs, grid = _runner(monkeypatch, ["--csv", str(tmp_path / "t.csv")])
    assert (kwargs["pair_checkpoints"], kwargs["policy_checkpoints"]) == ({}, {})
    assert grid["arms"] == [a for a in DEFAULT_ARMS if a != "H0"]
    assert kwargs["selector_weights_path"] is None


def test_the_runner_passes_trained_checkpoints_to_the_driver(monkeypatch, tmp_path, ckpts):
    kwargs, grid = _runner(monkeypatch, [
        "--csv", str(tmp_path / "t.csv"), *SIM_FLAGS, "--arms", "FQ", "FQ-g0", "E3", "FX",
        "--pair-checkpoint", f"main={ckpts.paths['main']}",
        "--pair-checkpoint", f"g0={ckpts.paths['g0']}",
        "--policy-checkpoint", f"E3={ckpts.paths['e3']}",
    ])
    assert kwargs["pair_checkpoints"] == {"main": ckpts.paths["main"], "g0": ckpts.paths["g0"]}
    assert kwargs["policy_checkpoints"] == {"e3": ckpts.paths["e3"]}
    assert grid["arms"] == ["FQ", "FQ-g0", "E3", "FX"]
    kwargs, _grid = _runner(monkeypatch, [
        "--csv", str(tmp_path / "u.csv"), *SIM_FLAGS, "--arms", "E3",
        "--policy-checkpoint", f"e3={ckpts.paths['e3']}"])
    assert kwargs["policy_checkpoints"] == {"e3": ckpts.paths["e3"]}


@pytest.mark.parametrize("case,why", [
    ("bootstrap", "its purpose is 'bootstrap', not 'trained'"),
    ("no_episode", "it trained on 0 episodes"),
    ("unscored", "it has no held-out score"),
    ("dirty", "it was trained from a dirty tree"),
    ("kind", "a '{other}' checkpoint, and this flag takes '{kind}' ones"),
    ("missing", "pair checkpoint not found"),
    ("relabelled", "the manifest's purpose is not the arrays' header's"),
    ("tag", "'g1' is no learned arm's tag"),
    ("twice", "tag '{tag}' is given twice"),
    ("malformed", "takes TAG=PATH"),
])
@pytest.mark.parametrize("flag", RUNNER_FLAGS)
def test_the_runner_refuses_a_checkpoint_a_campaign_may_not_fly(
        flag, case, why, monkeypatch, tmp_path, ckpts, capsys):
    """Critic B9 and other choices 6, under either flag, for the arm that flies
    it: judged on the verified manifest, so a bootstrap relabelled as trained
    is refused too (resolution R2). The driver flies the first four (it
    checks no training state, so FerrySim's bootstrap checkpoints load), so
    the runner is all that keeps them out of a campaign. Each is a usage
    error naming the flag; ``E3`` and ``e3`` name one tag."""
    name, tag, arm, kind, other = RUNNER_FLAGS[flag]
    own, foreign = ckpts.paths[tag], ckpts.paths["main" if tag == "e3" else "e3"]
    path = tmp_path / "c.npz"
    value, extra = f"{name}={path}", []
    if case == "bootstrap":
        _flag_checkpoint(flag, path, purpose="bootstrap")
    elif case == "no_episode":
        _flag_checkpoint(flag, path, episodes_trained=0)
    elif case == "unscored":
        _flag_checkpoint(flag, path, held_out=None)
    elif case == "dirty":
        _flag_checkpoint(flag, path, dirty=True)
    elif case == "kind":
        value = f"{name}={foreign}"
    elif case == "missing":
        pass
    elif case == "relabelled":
        _flag_checkpoint(flag, path, purpose="bootstrap", episodes_trained=10,
                         held_out={"episodes": 5, "return_mean": 0.1}, dirty=False)
        manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
        manifest["purpose"] = "trained"
        manifest_path(path).write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                                       encoding="utf-8")
    elif case == "tag":
        value = f"g1={own}"
    elif case == "twice":
        value, extra = f"{name}={own}", [flag, f"{tag}={own}"]
    else:
        value = own
    message = _refused(monkeypatch, capsys, ["--csv", str(tmp_path / "t.csv"), *SIM_FLAGS,
                                             "--arms", arm, flag, value, *extra])
    assert message.startswith(flag)
    assert why.format(kind=kind, other=other, tag=tag) in message


@pytest.mark.parametrize("flag", RUNNER_FLAGS)
def test_the_runner_flies_a_dirty_checkpoint_only_when_allowed(
        flag, monkeypatch, tmp_path, capsys):
    """``--allow-dirty-checkpoint`` lifts the dirty-tree refusal and no other: a
    bootstrap, which is dirty too, is still refused for its other reasons."""
    name, tag, arm, _kind, _other = RUNNER_FLAGS[flag]
    dirty, boot = tmp_path / "dirty.npz", tmp_path / "boot.npz"
    _flag_checkpoint(flag, dirty, dirty=True)
    _flag_checkpoint(flag, boot, purpose="bootstrap")
    argv = ["--csv", str(tmp_path / "t.csv"), *SIM_FLAGS, "--arms", arm]
    message = _refused(monkeypatch, capsys, argv + [flag, f"{name}={dirty}"])
    assert message.endswith("a campaign does not fly this checkpoint (critic B9): it was "
                            "trained from a dirty tree (allow it with --allow-dirty-checkpoint)")
    kwargs, _grid = _runner(monkeypatch, argv + [flag, f"{name}={dirty}",
                                                 "--allow-dirty-checkpoint"])
    given = {tag: str(dirty)}
    assert (kwargs["pair_checkpoints"], kwargs["policy_checkpoints"]) == (
        ({}, given) if tag == "e3" else (given, {}))
    message = _refused(monkeypatch, capsys, argv + [flag, f"{name}={boot}",
                                                    "--allow-dirty-checkpoint"])
    assert message.endswith("its purpose is 'bootstrap', not 'trained'; it trained on 0 "
                            "episodes (at least 1); it has no held-out score (the evaluator "
                            "fills it)")


#: Checkpoints their tag may not fly (resolution R24): the flag, the tag the value
#: names, the checkpoint's γ (None: E3's network), its provenance's changes from
#: a trained one's, and the reason the runner gives.
TAG_REFUSALS = {
    "g25-gamma-0.5": ("--pair-checkpoint", "g25", 0.5, {},
                      "its γ is 0.5, which flies as tag 'g50', not 'g25'"),
    "g0-gamma-0.9": ("--pair-checkpoint", "g0", 0.9, {},
                     "its γ is 0.9, which flies as tag 'g90', not 'g0'"),
    "g25-gamma-off-grid": ("--pair-checkpoint", "g25", 0.255, {},
                           "its γ is 0.255, which no tag names (γ in whole hundredths), "
                           "not 'g25'"),
    "hand-derived": ("--pair-checkpoint", "hand", 0.9, {},
                     "it trained on the 'derived' reward, and tag 'hand' flies the 'hand' "
                     "reward's score"),
    "g50-hand": ("--pair-checkpoint", "g50", 0.5, {"reward": {"kind": "hand"}},
                 "it trained on the 'hand' reward, and tag 'g50' flies the 'derived' "
                 "reward's score"),
    "dwell-hand": ("--pair-checkpoint", "dwell", 0.9,
                   {"reward": {"kind": "hand"}, "training": {
                       "episodes": 2000, "spec": {"plan_score_params": FQ_DWELL_SCORE}}},
                   "it trained on the 'hand' reward, and tag 'dwell' flies the 'derived' "
                   "reward's score"),
    "main-no-update": ("--pair-checkpoint", "main", 0.9,
                       {"validation": [{"episode": 1000, "updates": 0},
                                       {"episode": 2000, "updates": 0}]},
                       "its network took no update: the weights it keeps (validated after "
                       "episode 2000) are its initial ones"),
    "main-kept-before-any-update": ("--pair-checkpoint", "main", 0.9,
                                    {"episodes_trained": 1000,
                                     "validation": [{"episode": 1000, "updates": 0},
                                                    {"episode": 2000, "updates": 40}]},
                                    "its network took no update: the weights it keeps "
                                    "(validated after episode 1000) are its initial ones"),
    "main-no-update-record": ("--pair-checkpoint", "main", 0.9,
                              {"validation": [{"episode": 1000, "return_mean": 0.31}]},
                              "its manifest records no update count for the weights it keeps "
                              "(no validation at its kept episode 2000)"),
    "e3-no-update": ("--policy-checkpoint", "E3", None,
                     {"reward": {"kind": "bytes"}, "validation": [{"episode": 2000,
                                                                   "updates": 0}]},
                     "its network took no update"),
    "e3-derived": ("--policy-checkpoint", "E3", None, {"reward": {"kind": "derived"}},
                   "it trained on the 'derived' reward, and tag 'e3' flies the 'bytes' "
                   "reward's score"),
    # Study 5.7's weight grid under a tag whose arm flies decision 4 (a)'s weights
    # (the repair round's A-4), each weight alone, each on its arm's own plan
    "g90-grid": ("--pair-checkpoint", "g90", 0.9,
                 {"reward": {"kind": "derived", "c_t": 0.3, "c_cov": 4.0}},
                 "it trained on the derived reward at c_t 0.3, c_cov 4, and tag 'g90' flies it "
                 "at decision 4 (a)'s weights, c_t 0.1 and c_cov 1"),
    "dwell-grid": ("--pair-checkpoint", "dwell", 0.9,
                   {"reward": {"kind": "derived", "c_t": 0.03, "c_cov": 1.0}, "training": {
                       "episodes": 2000, "spec": {"plan_score_params": FQ_DWELL_SCORE}}},
                   "it trained on the derived reward at c_t 0.03, c_cov 1, and tag 'dwell' "
                   "flies it at decision 4 (a)'s weights, c_t 0.1 and c_cov 1"),
    "cov-grid": ("--pair-checkpoint", "cov", 0.9,
                 {"reward": {"kind": "derived", "c_t": 0.1, "c_cov": 0.25}, "training": {
                     "episodes": 2000, "spec": {"plan_score_params": F_COV_SCORE}}},
                 "it trained on the derived reward at c_t 0.1, c_cov 0.25, and tag 'cov' "
                 "flies it at decision 4 (a)'s weights, c_t 0.1 and c_cov 1"),
}


@pytest.mark.parametrize("case", TAG_REFUSALS)
def test_the_runner_refuses_a_checkpoint_its_tag_may_not_fly(case, monkeypatch, tmp_path,
                                                              capsys):
    """Resolution R24 (the final check's F2): a γ tag flies its own γ (bound by the
    sha through the header's network), ``hand`` F·hand, ``e3`` E3's bytes (the
    trainer's only reward for E3) and every other tag the derived reward, at
    decision 4 (a)'s weights under a γ tag, ``dwell`` and ``cov`` (the repair
    round's A-4), and no tag a network still at its initial weights, the kept
    validation's update count being 0 or unrecorded (a "trained" network that
    never updated is a random-init arm in substance). The driver would fly each,
    so the runner refuses it as a usage error naming the flag, the tag and the
    reason, whatever arm runs."""
    flag, name, gamma, changes, why = TAG_REFUSALS[case]
    path = tmp_path / "c.npz"
    if gamma is None:
        _e3_checkpoint(path, provenance=_trained(**changes))
    else:
        _pair_checkpoint(path, gamma=gamma, provenance=_trained(**changes))
    tag = "e3" if name == "E3" else name             # E3 and e3 name one tag
    message = _refused(monkeypatch, capsys, ["--csv", str(tmp_path / "t.csv"), *SIM_FLAGS,
                                             "--arms", "FX", flag, f"{name}={path}"])
    assert message.startswith(f"{flag} {tag}={path}: a campaign does not fly this checkpoint "
                              f"as tag {tag!r} (resolution R24): ")
    assert why in message


@pytest.mark.parametrize("tag,gamma,changes", [
    ("g0", 0.0, {}),
    ("g25", 0.25, {}),
    ("g50", 0.5, {}),
    ("g99", 0.99, {}),
    ("hand", 0.9, {"reward": {"kind": "hand"}}),
    ("main", 0.5, {}),                              # the main score flies its own γ
    ("main", 0.9, {"reward": {"kind": "derived", "c_t": 0.3, "c_cov": 4.0}}),   # 5.7's grid
    ("main", 0.9, {"episodes_trained": 1000,         # kept after its first updates
                   "validation": [{"episode": 1000, "updates": 1},
                                  {"episode": 2000, "updates": 900}]}),
    # kept after a first validation that came before the replay's warm-up (A-3)
    ("main", 0.9, {"validation": [{"episode": 1000, "updates": 0},
                                  {"episode": 2000, "updates": 40}]}),
    ("g90", 0.9, {"reward": {"kind": "derived", "c_t": 0.1, "c_cov": 1}}),  # 1 is 1.0
])
def test_the_runner_flies_a_checkpoint_under_its_own_tag(tag, gamma, changes, monkeypatch,
                                                         tmp_path):
    """Resolution R24's positive side: each checkpoint flies under the tag it
    matches, through the driver's own checks of the arm."""
    path = tmp_path / "c.npz"
    _pair_checkpoint(path, gamma=gamma, provenance=_trained(**changes))
    arm = {t: a for a, t in CHECKPOINT_TAGS.items()}[tag]
    kwargs, _grid = _runner(monkeypatch, ["--csv", str(tmp_path / "t.csv"), *SIM_FLAGS,
                                          "--arms", arm, "--pair-checkpoint", f"{tag}={path}"])
    assert kwargs["pair_checkpoints"] == {tag: str(path)}


def test_the_kept_weights_updates_are_the_kept_validations_whatever_came_before():
    """The repair round's A-3: the update count R24 reads is the kept validation's
    (the entry at ``episodes_trained``), not the run's first validation's nor the
    least over the run, so a run validated once before its replay warmed up still
    flies the later weights it kept, which took updates; weights kept at that
    first validation are refused."""
    from experiments.ferrysim.checkpoints import kept_updates, tag_refusals

    late = dict(_trained(validation=[{"episode": 1000, "updates": 0},
                                     {"episode": 2000, "updates": 40},
                                     {"episode": 3000, "updates": 75}]), gamma=0.9)
    assert late["episodes_trained"] == 2000
    assert kept_updates(late) == 40 and tag_refusals("main", late) == []
    early = dict(late, episodes_trained=1000)
    assert kept_updates(early) == 0
    assert tag_refusals("main", early) == [
        "its network took no update: the weights it keeps (validated after episode 1000) "
        "are its initial ones"]


#: Pair checkpoints trained on another plan than their arm flies (resolution
#: R23): the tag, the plan its training spec records, the run's own plan flags,
#: the arm run, and what differs.
PLAN_REFUSALS = {
    "dwell-on-the-default-plan": (
        "dwell", None, [], "FQ-dwell", "dwell_in_delta (trained True, flown False)"),
    "main-on-dwells-plan": (
        "main", FQ_DWELL_SCORE, [], "FQ", "dwell_in_delta (trained False, flown True)"),
    "cov-on-dwells-plan": (
        "cov", FQ_DWELL_SCORE, [], "FQ-cov",
        "c_cov_per_device (trained 1.0, flown 0.0), c_link (trained None, flown 0.0), "
        "dwell_in_delta (trained False, flown True)"),
    "main-against-a-pilots-kappa": (
        "main", None, ["--plan-score-params", '{"c_cov_per_device": 0.25}'], "FQ",
        "c_cov_per_device (trained 1.0, flown 0.25)"),
    "dwell-against-a-pilots-c4": (
        "dwell", FQ_DWELL_SCORE, ["--plan-score-params", '{"c_energy": 0}'], "FQ-dwell",
        "c_energy (trained 0.1, flown 0.0)"),
    "dwell-given-while-fx-runs": (
        "dwell", None, [], "FX", "dwell_in_delta (trained True, flown False)"),
}


@pytest.mark.parametrize("case", PLAN_REFUSALS)
def test_the_runner_refuses_a_pair_checkpoint_trained_on_another_plan(case, monkeypatch,
                                                                       tmp_path, capsys):
    """Resolution R23 (the final check's F1; critic C2): a pair checkpoint flies
    only on the plan score settings it trained under, which its manifest's
    training spec records (none: the cells' default plan); its arm flies the
    driver's plan under this run's flags (``--plan-score-params``, and FQ-dwell's
    or FQ-cov's own change on top). Refused as a usage error naming each setting
    that differs, for every checkpoint given, whichever arms run."""
    tag, plan, flags, arm, why = PLAN_REFUSALS[case]
    path = tmp_path / "c.npz"
    _pair_checkpoint(path, provenance=_trained() if plan is None else _planned(plan))
    message = _refused(monkeypatch, capsys, ["--csv", str(tmp_path / "t.csv"), *SIM_FLAGS,
                                             "--arms", arm, *flags,
                                             "--pair-checkpoint", f"{tag}={path}"])
    flies = {t: a for a, t in CHECKPOINT_TAGS.items()}[tag]
    assert message == (f"--pair-checkpoint {tag}={path}: trained under other plan score "
                       f"settings than arm {flies} flies here (resolution R23): {why}; a "
                       f"checkpoint flies on the plan it trained under")


@pytest.mark.parametrize("tag,plan,flags", [
    ("main", {}, []),
    ("dwell", FQ_DWELL_SCORE, []),
    ("cov", F_COV_SCORE, []),
    ("g90", {}, []),
    # the defaults written out are the default plan, and 0 is 0.0
    ("main", {"c_cov_per_device": 1, "dwell_in_delta": True, "c_link": None}, []),
    ("main", {"c_energy": 0}, ["--plan-score-params", '{"c_energy": 0.0}']),
    ("cov", dict(F_COV_SCORE, c_energy=0.0), ["--plan-score-params", '{"c_energy": 0}']),
    ("dwell", dict(FQ_DWELL_SCORE, c_cov_per_device=0.25),
     ["--plan-score-params", '{"c_cov_per_device": 0.25}']),
])
def test_the_runner_flies_a_pair_checkpoint_on_the_plan_it_trained_under(
        tag, plan, flags, monkeypatch, tmp_path):
    """Resolution R23's positive side: the settings are compared with
    ``PlanScoreParams``' defaults filled in on both sides."""
    path = tmp_path / "c.npz"
    _pair_checkpoint(path, provenance=_planned(plan) if plan else _trained())
    arm = {t: a for a, t in CHECKPOINT_TAGS.items()}[tag]
    kwargs, _grid = _runner(monkeypatch, ["--csv", str(tmp_path / "t.csv"), *SIM_FLAGS,
                                          "--arms", arm, *flags,
                                          "--pair-checkpoint", f"{tag}={path}"])
    assert kwargs["pair_checkpoints"] == {tag: str(path)}


def test_the_runners_help_gives_each_checkpoint_flag_its_own_conditions():
    """The repair round's A-5: E3's flag names E3's conditions, its bytes reward and a
    network that took an update, not the pair flag's (the derived reward, decision 4
    (a)'s weights and a plan score), which E3 does not fly (``tag_reward('e3')``)."""
    import argparse

    from experiments.exp4 import runner_main
    from experiments.ferrysim.checkpoints import E3_TAG, tag_reward

    parser = argparse.ArgumentParser()
    runner_main._add_phase_5_flags(parser)
    helps = {flag: action.help for action in parser._actions for flag in action.option_strings}
    policy, pair = helps["--policy-checkpoint"], helps["--pair-checkpoint"]
    assert tag_reward(E3_TAG) == "bytes" and "E3's bytes reward" in policy
    for condition in ("trained, on at least one episode", "held-out runs",
                      "--allow-dirty-checkpoint", "took an update"):
        assert condition in policy and condition in pair, condition
    for other in ("same conditions", "derived", "plan score", "weights"):
        assert other not in policy, other
    assert "derived reward (at decision 4 (a)'s weights under gX, dwell and cov)" in pair
    assert "plan score settings its arm flies" in pair


@pytest.mark.parametrize("arms,weights,ok", [
    (None, False, False),                        # the default list holds H2 and H3
    (["H1", "H2"], False, False),
    (["H3"], False, False),
    (["H1", "D1", "D4"], False, True),
    (["H2", "H3"], True, True),
])
def test_require_trained_refuses_h2_and_h3_without_weights(monkeypatch, tmp_path, capsys,
                                                           arms, weights, ok):
    """Decision 8 (a): opt-in; the weights are loaded by the mule, as recorded."""
    argv = ["--csv", str(tmp_path / "t.csv"), "--require-trained"]
    if arms is not None:
        argv += ["--arms", *arms]
    if weights:
        argv += ["--selector-weights", str(tmp_path / "ddqn.npz")]
    if ok:
        kwargs, _grid = _runner(monkeypatch, argv)
        assert kwargs["selector_weights_path"] == (str(tmp_path / "ddqn.npz") if weights
                                                   else None)
        return
    untrained = [arm for arm in (arms or DEFAULT_ARMS) if arm in ("H2", "H3")]
    assert _refused(monkeypatch, capsys, argv).startswith(
        f"--require-trained: arms {', '.join(untrained)} fly the RL selector, which without "
        f"--selector-weights is random-init")
    # Without the flag the recorded runner runs them (random-init, warned by the mule).
    _runner(monkeypatch, [a for a in argv if a != "--require-trained"])


@pytest.mark.parametrize("extra,why", [
    (["--arms", "FQ", *SIM_FLAGS], "no random-init arm"),
    (["--arms", "FQ", "--mission-clock", "sim", "--contact-band", "wide",
      "--replan-fallback", "trim", "--pair-checkpoint", "main={main}"], "resolution R3"),
    (["--arms", "E3", "--policy-checkpoint", "E3={e3}"], "only on the simulated mission clock"),
    (["--arms", "E3", *SIM_FLAGS], "no random-init arm"),
    (["--arms", "H1+L1", *SIM_FLAGS], "here it would fly as H1"),
    (["--arms", "FQ", *SIM_FLAGS, "--pair-checkpoint", "main={e3}"], "this flag takes"),
])
def test_the_runner_refuses_a_learned_arm_it_cannot_fly(monkeypatch, tmp_path, ckpts, capsys,
                                                        extra, why):
    argv = ["--csv", str(tmp_path / "t.csv")] + [
        a.format(main=ckpts.paths["main"], e3=ckpts.paths["e3"]) for a in extra]
    assert why in _refused(monkeypatch, capsys, argv)
