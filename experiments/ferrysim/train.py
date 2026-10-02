"""FerrySim's trainer: the pair score (and E3) learn on the real system, in process.

FeRRy Phase 5, unit U8b (the Phase 5 spec, other choices 3, 7, 8 and 12; the
user's decisions 4 and 5 (a); critic C1 and C4; orchestrator resolutions R2,
R10 and R11). One call of :func:`train` is one training run: one learner, at
one γ, from one seed, over one cell family's training stream, ending in one
format-2 checkpoint with its manifest.

**The learner** (other choices 3). The pair learner of unit U2
(``hermes.scheduler.selector.pair_q``): a masked pointer double DQN, one Q per
pair row from shared weights (2 x 64, tanh), Adam at 1e-3, the Huber loss,
a global-norm clip of 10, a hard target sync every 500 updates, over the
replay of ``selector.pair_replay`` (50,000 transitions, batches of 64, a
warm-up of 1,000). Its rows are ``pair_v1`` (``selector.pair_features``,
unit U1; resolution R11: 36 columns on the three-class link, the phase block
on). E3 (arm ``chen_dqn``, unit U6) trains the same learner over its own
per-stop rows (``policies.chen_dqn``), on its bytes reward.

**An episode** (:mod:`experiments.ferrysim.episode`). Training episode e of
run ``seed`` is a trial of a cell of the family, drawn with the trial seed
from the run's own stream ``ferrysim-train-<seed>`` (:func:`training_episode`,
the draw of ``cells.train_episode``), so a run's episodes are a pure function
of its seed, every γ of one seed meets the same episodes, and no training
episode is a validation or held-out one. The pair score flies FX's
configuration with a fresh pair slot around the live network
(``pair_features.LearnedPairScorer``, the network held, not copied),
installed before the first mission (``MuleSupervisor.install_flight_slot``);
E3 flies its arm through the config path from a bootstrap checkpoint the run
writes once at its start, and the run's live network is attached before the
policy's first call (``ChenDQNPolicy.attach_trainer``). The devices train on
equal shards (``inprocess.equal_shard_trainer``; the spec, other choices 8).

**The plan** (the orchestrator's resolution R23; critic C2). The pair score
trains and validates under its run's plan score settings
(:attr:`TrainSpec.plan_score_params`), a driver override on the cells'
settings, which set none: FQ-dwell's and FQ-cov's scores under their arms'
plans (:data:`ABLATIONS`, the driver's own ``FQ_DWELL_SCORE`` and
``F_COV_SCORE``), or a pilot's settings; the manifest's training spec records
them, and the runner flies the checkpoint only on that plan. The default,
none, flies the cells' own plan, passes no override and records nothing, so
a default run, its checkpoint and its manifest are the ones they always
were; a spec without the record reads as that plan
(``checkpoints.trained_plan``).

**Behaviour** (critic C4; ``pair_q.BehaviourSchedule``). Episode e flies the
schedule's behaviour at e of the planned episodes: for the pair score, its
first 500 episodes ε-greedy around FX's pair at ε = 0.3, then ε-greedy on Q
with ε falling from 0.3 to 0.05 over the first half of the run; ε never
starts at 1.0. E3 has no reference phase (Chen's recipe explores around its
own Q). Every decision draws from the episode's own seeded stream
(:func:`behaviour_seed`).

**The reward** (decision 4 (a); other choices 7; critic C1). The derived
reward at c_t = 0.1 and c_cov = 1 (``reward.RewardSpec``), F·hand for the
FQ-hand score, E3's bytes for E3, each taken at the expectation over the
keyed availability draw while training (the draw does not depend on the
pair, and its noise is 10 to 100 times the time signal a decision moves).
Validation, the held-out evaluation and every reported number read the
realized draw.

**Transitions** (:func:`pair_transitions`, :func:`e3_transitions`). Each
Pass-1 decision of a mission is one transition: the row of the pair taken,
the decision's reward, and the next decision's every candidate row with the
mask it was taken among (the effective mask: on an empty mask, the pair
flown alone; other choices 1), or ``done`` at the sortie's last decision,
since the flight Q's horizon is the sortie (memo L264). A ``home`` decision
that a beacon insert follows bootstraps from the insert's decision (critic
B12). A mission with no Pass-1 decision (the slot closes it as
``close_mission(((), ()))``, resolution R9) adds no transition. An episode is
flown with the network as it stood at its start; then its transitions are
pushed in decision order, each followed by one update once the replay is warm
(``pair_q.PairQLearner.observe``): one update per decision after the warm-up
(other choices 3). So the network moves between episodes and never inside
one: a decision's reward closes only with its mission, and FerrySim reads an
episode's missions once it has flown (``episode.EpisodeResult``), so every
decision of an episode scores with the weights the episode began with.

**Validation and the checkpoint** (other choices 12). Every ``eval_every``
episodes (1,000), and after the last, the network flies the validation
episodes greedily: ``val_episodes`` (200) spread over the family's cells,
the first of each cell's validation stream ``ferrysim-val``, the same at every
validation and in every run (critic B14: the validation stream is the
headroom report's, never the held-out one). The validation score is the mean
over the cells of each cell's mean realized return; the run keeps the
weights of its best validation (a strictly higher score is a new best) and
stops after ``patience`` (3) validations without one. The kept weights are
saved as a ``trained`` checkpoint (``PairQNet.save``; resolution R2: the
purpose and ``pair_q.LEARNER_REVISION`` are in the header, so the sha binds
them), with a manifest recording the reward, the training spec and its
outcome, the seeds, the cell family and its hash, the trainer's commit and
dirty flag, the episodes the kept weights trained on, the validation curve,
and no held-out score yet (the evaluator writes it,
:func:`checkpoints.record_held_out_score`).

**Determinism** (the spec's conventions; T3). Every draw comes from a seeded
stream: the initial weights and the replay's sampling from seeds derived
from the run's (:func:`run_seeds`), each episode's trial from the run's
stream, each episode's behaviour from its own stream. With
``OPENBLAS_NUM_THREADS=1`` two runs of one spec write the same arrays (the
same sha) and the same manifest, bit for bit. No wall time enters a decision
or a record.

Not thread-safe: an in-process episode patches process-wide names, so a
process trains one run at a time; a sweep runs its runs in worker processes
(:func:`run_training_task`).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from experiments.exp4.driver import CHECKPOINT_TAGS, F_COV_SCORE, FQ_DWELL_SCORE
from experiments.ferrysim import cells as C
from experiments.ferrysim import inprocess
from experiments.ferrysim.checkpoints import (
    ARM_E3,
    E3_TAG,
    TreeState,
    plan_score_settings,
    replace_refusal,
)
from experiments.ferrysim.episode import ARM_FX, EpisodeResult, Policy, Trainer, run_episode
from experiments.ferrysim.reward import (
    DERIVED,
    REWARD_BYTES,
    REWARD_DERIVED,
    REWARD_HAND,
    RewardSpec,
)
from hermes.scheduler.selector.pair_q import (
    KIND_CHEN_DQN,
    KIND_PAIR_Q,
    PURPOSE_BOOTSTRAP,
    PURPOSE_TRAINED,
    BehaviourSchedule,
    LearnerSettings,
    PairQConfig,
    PairQLearner,
    PairQNet,
    verify_checkpoint,
)
from hermes.scheduler.selector.pair_replay import PairReplay, PairTransition

PathLike = Union[str, Path]

#: The kinds a run trains: the pair score (the FQ arms) and E3.
TRAIN_KINDS: Tuple[str, ...] = (KIND_PAIR_Q, KIND_CHEN_DQN)

#: The rewards each kind trains on (decision 4 (a), other choices 7): the pair
#: score the derived reward (or F·hand, "today's reward", for FQ-hand), E3
#: its bytes (decision 7 (a)).
KIND_REWARDS: Dict[str, Tuple[str, ...]] = {
    KIND_PAIR_Q: (REWARD_DERIVED, REWARD_HAND),
    KIND_CHEN_DQN: (REWARD_BYTES,),
}

#: The training rewards: each at the expectation over the availability draw
#: (critic C1).
TRAINING_REWARD = dataclasses.replace(DERIVED, expected_availability=True)
HAND_TRAINING_REWARD = RewardSpec(kind=REWARD_HAND, c_t=0.0, c_cov=0.0,
                                  expected_availability=True)
BYTES_TRAINING_REWARD = RewardSpec(kind=REWARD_BYTES, c_t=0.0, c_cov=0.0,
                                   expected_availability=True)

#: Study 5.5's run (the spec, other choices 12): 10,000 episodes, 200
#: validation episodes every 1,000, stop after three without a new best.
EPISODES = 10_000
EVAL_EVERY = 1_000
VAL_EPISODES = 200
PATIENCE = 3

#: The device model every training run flies (other choices 8): equal shards.
TRAINING_DEVICE_MODEL = inprocess.DEVICE_MODEL_EQUAL

#: E3's behaviour: the pair learner's ε schedule with no reference phase.
E3_BEHAVIOUR = BehaviourSchedule(reference_episodes=0)

#: Study 5.7's plan-term ablations, by their arms' tags (critic C2; resolution
#: R23): FQ-dwell's score trains with the dwell out of Δ and FQ-cov's with the
#: coverage term off, the plans those arms fly (the driver's own settings).
ABLATIONS: Dict[str, Mapping[str, Any]] = {
    CHECKPOINT_TAGS["FQ-dwell"]: FQ_DWELL_SCORE,
    CHECKPOINT_TAGS["FQ-cov"]: F_COV_SCORE,
}


def _link_classes() -> Tuple[str, ...]:
    """The link's classes in link order: every FerrySim cell flies the default
    three-class link (``hermes.l1.contact_link.CLASSES``)."""
    from hermes.l1.contact_link import CLASSES

    return tuple(CLASSES)


def _int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} is an int >= {minimum}, got {value!r}")
    return value


def _seed_of(text: str) -> int:
    """A 31-bit seed, the first four bytes of the SHA-256 of ``text``."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") % (2 ** 31)


# --------------------------------------------------------------------------- #
# What a run is
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class TrainSpec:
    """One training run (the spec, other choices 3 and 12).

    ``kind`` is ``pair_q`` (the FQ arms' score) or ``chen_dqn`` (E3); ``seed``
    the run's training seed, the unit of Study 5.5's replication; ``family``
    the cell family it practises over (one score per contact regime, decision
    3) and ``cells`` its cells, the family's own by default (a test's or a
    probe's cells under its own family label otherwise). ``network`` holds γ
    (``network.gamma``) and the network's settings, ``learner`` the batch,
    replay, warm-up and behaviour schedule, ``reward`` the training reward
    (taken at the expected availability, critic C1). ``episodes``,
    ``eval_every``, ``val_episodes`` and ``patience`` shape the run;
    ``phase`` keeps the pair rows' phase block (``pair_v1``, design D-D (b)).
    ``plan_score_params`` are the plan score settings the pair score trains
    and validates under, by ``PlanScoreParams`` field (resolution R23: an
    ablation's, :data:`ABLATIONS`, or a pilot's); empty, the default, is the
    cells' own plan.
    """

    kind: str = KIND_PAIR_Q
    seed: int = 0
    family: str = C.FAMILY_JITTERY
    cells: Tuple[C.FerryCell, ...] = ()
    network: PairQConfig = dataclasses.field(default_factory=PairQConfig)
    learner: LearnerSettings = dataclasses.field(default_factory=LearnerSettings)
    reward: RewardSpec = TRAINING_REWARD
    episodes: int = EPISODES
    eval_every: int = EVAL_EVERY
    val_episodes: int = VAL_EPISODES
    patience: int = PATIENCE
    phase: bool = True
    plan_score_params: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in TRAIN_KINDS:
            raise ValueError(f"kind is one of {TRAIN_KINDS}, got {self.kind!r}")
        _int(self.seed, "seed", 0)
        if not isinstance(self.family, str) or not self.family:
            raise ValueError(f"family names the cells' family, got {self.family!r}")
        cells = tuple(self.cells) or C.FAMILIES.get(self.family, ())
        if not cells:
            raise ValueError(f"no cells: {self.family!r} is not one of the families "
                             f"{sorted(C.FAMILIES)}, so name the cells")
        for cell in cells:
            if not isinstance(cell, C.FerryCell):
                raise TypeError(f"cells hold FerryCells, got {cell!r}")
        if len({cell.name for cell in cells}) != len(cells):
            raise ValueError("a run's cells have distinct names")
        bands = {cell.driver_settings()["contact_band"] for cell in cells}
        if len(bands) != 1:
            raise ValueError(f"a run's cells fly one contact band, got {sorted(bands)}")
        object.__setattr__(self, "cells", cells)
        if not isinstance(self.network, PairQConfig):
            raise TypeError(f"network is a PairQConfig, got {self.network!r}")
        if not isinstance(self.learner, LearnerSettings):
            raise TypeError(f"learner is LearnerSettings, got {self.learner!r}")
        if not isinstance(self.reward, RewardSpec):
            raise TypeError(f"reward is a RewardSpec, got {self.reward!r}")
        if self.reward.kind not in KIND_REWARDS[self.kind]:
            raise ValueError(f"a {self.kind!r} run trains on {KIND_REWARDS[self.kind]}, got "
                             f"{self.reward.kind!r}")
        if not self.reward.expected_availability:
            raise ValueError("a training reward is taken at the expected availability (critic "
                             "C1); the realized draw is for validation and evaluation")
        if self.kind == KIND_CHEN_DQN and self.learner.behaviour.reference_episodes:
            raise ValueError("E3 has no reference to fly around (Chen's recipe explores around "
                             "its own Q): give a behaviour schedule with reference_episodes=0")
        _int(self.episodes, "episodes", 1)
        _int(self.eval_every, "eval_every", 1)
        _int(self.val_episodes, "val_episodes", len(cells))
        _int(self.patience, "patience", 1)
        if not isinstance(self.phase, bool):
            raise TypeError(f"phase is a bool, got {self.phase!r}")
        plan = plan_score_settings(self.plan_score_params)
        if plan and self.kind != KIND_PAIR_Q:
            raise ValueError("plan_score_params set the plan score, which only the pair score "
                             "flies under: E3 flies legacy mode, with no plan (decision 7 (a))")
        planned = [cell.name for cell in cells if "plan_score_params" in cell.driver_settings()]
        if planned:
            raise ValueError(f"cells {planned} set plan_score_params: a run's cells set none, "
                             f"since the run's own are the plan its manifest records "
                             f"(resolution R23)")
        object.__setattr__(self, "plan_score_params", plan)

    def __hash__(self) -> int:
        """The generated hash, the plan's settings taken as their sorted items, so a
        spec stays hashable."""
        return hash(tuple(tuple(sorted(self.plan_score_params.items()))
                          if f.name == "plan_score_params" else getattr(self, f.name)
                          for f in dataclasses.fields(self)))

    @property
    def gamma(self) -> float:
        return self.network.gamma

    @property
    def band(self) -> str:
        """The cells' contact band: E3's one class."""
        return str(self.cells[0].driver_settings()["contact_band"])

    @property
    def validation_reward(self) -> RewardSpec:
        """The training reward at the realized draw: what validation reads."""
        return dataclasses.replace(self.reward, expected_availability=False)

    @property
    def family_sha256(self) -> str:
        """SHA-256 of the cells as JSON, ``cells.family_sha256``'s formula, so a
        family's own cells hash as the family does."""
        blob = json.dumps([c.to_json() for c in self.cells], sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha256(blob.encode("ascii")).hexdigest()

    def schema(self):
        """The pair rows' schema (``pair_v1`` over the link's classes); pair_q only."""
        from hermes.scheduler.selector.pair_features import PairFeatureSchema

        return PairFeatureSchema(_link_classes(), phase=self.phase)

    def to_json(self) -> Dict[str, Any]:
        """The training spec a manifest records (JSON-ready, no wall time).

        ``plan_score_params`` only when set (resolution R23): a run on the
        cells' own plan records none, so its spec is the one it always was.
        """
        out = {
            "kind": self.kind,
            "seed": self.seed,
            "family": self.family,
            "cells": [cell.name for cell in self.cells],
            "episodes": self.episodes,
            "eval_every": self.eval_every,
            "val_episodes": self.val_episodes,
            "patience": self.patience,
            "phase": self.phase,
            "network": self.network.to_json(),
            "learner": self.learner.to_json(),
            "reward": self.reward.to_json(),
            "validation_reward": self.validation_reward.to_json(),
            "device_model": TRAINING_DEVICE_MODEL,
        }
        if self.plan_score_params:
            out["plan_score_params"] = dict(self.plan_score_params)
        return out


def pair_spec(gamma: float, seed: int, **kw: Any) -> TrainSpec:
    """A pair-score run at ``gamma`` from ``seed``: the spec's settings, ``kw`` on top
    (``family``, ``cells``, ``reward``, ``episodes``, ...; ``network`` and
    ``learner`` replace the defaults whole, with γ set in the network)."""
    network = dataclasses.replace(kw.pop("network", PairQConfig()), gamma=gamma)
    return TrainSpec(kind=KIND_PAIR_Q, seed=seed, network=network, **kw)


def e3_spec(gamma: float, seed: int, **kw: Any) -> TrainSpec:
    """An E3 run at ``gamma`` from ``seed``: the pair learner's settings without a
    reference phase (:data:`E3_BEHAVIOUR`) and the bytes reward, ``kw`` on top."""
    network = dataclasses.replace(kw.pop("network", PairQConfig()), gamma=gamma)
    kw.setdefault("learner", LearnerSettings(behaviour=E3_BEHAVIOUR))
    kw.setdefault("reward", BYTES_TRAINING_REWARD)
    return TrainSpec(kind=KIND_CHEN_DQN, seed=seed, network=network, **kw)


# --------------------------------------------------------------------------- #
# Seeds and episodes
# --------------------------------------------------------------------------- #

def run_seeds(seed: int) -> Dict[str, Any]:
    """The seeds a run derives from its training seed (the manifest's ``seeds``).

    The network's initial weights and the replay's sampling each have their
    own seed, a hash of the run's stream, so two runs of one seed start alike
    whatever their γ (common random numbers across Study 5.5's grid).
    """
    stream = C.train_stream(seed)
    return {
        "run": seed,
        "init": _seed_of(f"{stream}|init"),
        "replay": _seed_of(f"{stream}|replay"),
        "train_stream": stream,
        "val_stream": C.VAL_STREAM,
        "val_first_index": 0,
        "behaviour": f"sha256('{stream}|behaviour|<episode>')[:4] mod 2**31",
    }


def behaviour_seed(seed: int, episode: int) -> int:
    """The seed of episode ``episode``'s behaviour stream (``random.Random``), from
    which each of its decisions draws (``pair_q.behaviour_row``)."""
    return _seed_of(f"{C.train_stream(seed)}|behaviour|{_int(episode, 'episode', 0)}")


def training_episode(seed: int, family: str, cells: Sequence[C.FerryCell],
                     index: int) -> Tuple[C.FerryCell, int]:
    """Training episode ``index`` of run ``seed``: its cell and its trial seed.

    ``cells.train_episode``'s draw over ``cells`` (a keyed draw of the run's
    stream, the family and the index picks the cell; the run's stream gives the
    trial seed), so a family's own cells give exactly ``train_episode``'s
    episodes.
    """
    stream = C.train_stream(seed)
    _int(index, "index", 0)
    digest = hashlib.sha256(f"{stream}|cell|{family}|{index}".encode("utf-8")).digest()
    cell = cells[int.from_bytes(digest[:4], "big") % len(cells)]
    return cell, C.trial_seed(stream, cell.name, index)


def validation_episodes(cells: Sequence[C.FerryCell],
                        total: int) -> Tuple[Tuple[C.FerryCell, int, int], ...]:
    """The validation episodes: (cell, index, trial seed), ``total`` spread over the
    cells as evenly as can be (the first cells take one more), each cell's first
    episodes of the validation stream."""
    _int(total, "total", len(cells))
    out: List[Tuple[C.FerryCell, int, int]] = []
    share, extra = divmod(total, len(cells))
    for i, cell in enumerate(cells):
        count = share + (1 if i < extra else 0)
        for index, seed in enumerate(C.stream_seeds(C.VAL_STREAM, cell.name, count)):
            out.append((cell, index, seed))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Flying episodes
# --------------------------------------------------------------------------- #

def _bootstrap_overrides(bootstrap: Optional[PathLike]) -> Dict[str, Any]:
    if bootstrap is None:
        raise ValueError("an E3 episode flies its arm from a bootstrap checkpoint: give one")
    return {"policy_checkpoints": {E3_TAG: str(bootstrap)}}


def fly_pair_episode(net: PairQNet, schema: Any, cell: C.FerryCell, seed: int, *,
                     index: int, reward: RewardSpec, trainer: Optional[Trainer] = None,
                     plan_score_params: Optional[Mapping[str, Any]] = None) -> EpisodeResult:
    """One episode of the pair score ``net``: FX's configuration with a fresh slot
    around the live network, a trainer attached when given (it keeps each
    mission's decisions, ``EpisodeResult.steps``). ``plan_score_params``, when
    given, are the plan score settings it flies, a driver override on the
    cell's settings (resolution R23); without them the cell's own plan flies
    and the episode is the one it always was."""
    from hermes.scheduler.selector.pair_features import LearnedPairScorer

    policy = Policy(label="learned", arm=ARM_FX,
                    scorer=functools.partial(LearnedPairScorer, net, schema))
    plan = ({"driver_overrides": {"plan_score_params": dict(plan_score_params)}}
            if plan_score_params else {})
    return run_episode(cell, seed, policy, trial_index=index, reward=reward,
                       device_model=TRAINING_DEVICE_MODEL, trainer=trainer, **plan)


def _plan(spec: TrainSpec) -> Dict[str, Any]:
    """:func:`fly_pair_episode`'s plan keyword for ``spec``'s episodes: none on the
    cells' own plan, so a default run's calls are the ones they always were."""
    return ({"plan_score_params": dict(spec.plan_score_params)} if spec.plan_score_params
            else {})


def fly_e3_episode(net: PairQNet, cell: C.FerryCell, seed: int, *, index: int,
                   reward: RewardSpec, bootstrap: PathLike, epsilon: float = 0.0,
                   rng_seed: int = 0) -> Tuple[EpisodeResult, List[Any]]:
    """One E3 episode: arm E3 from the bootstrap checkpoint (the config path), the
    live ``net`` attached before the policy's first call, ε-greedy from the
    episode's stream; the episode and its decisions (``chen_dqn.E3Step``)."""
    steps: List[Any] = []

    def attach(service) -> None:
        policy = service.supervisor.scheduler.target_selector
        policy.attach_trainer(net=net, epsilon=float(epsilon), rng=random.Random(rng_seed),
                              sink=steps.append)

    result = run_episode(cell, seed, Policy(label="learned", arm=ARM_E3), trial_index=index,
                         reward=reward, device_model=TRAINING_DEVICE_MODEL, hooks=[attach],
                         driver_overrides=_bootstrap_overrides(bootstrap))
    return result, steps


# --------------------------------------------------------------------------- #
# Transitions
# --------------------------------------------------------------------------- #

def pair_transitions(episode: EpisodeResult, schema: Any) -> List[PairTransition]:
    """The pair learner's transitions of a training episode, in decision order.

    Mission m's decisions are ``episode.steps[m]`` (the slot's ``PairStep``s),
    aligned with the sortie's stops and their rewards (``episode.rewards[m]``,
    under the episode's reward). Transition k is the row of the pair taken
    (``pair_rows`` of the step's view, at ``step.row``), r_k, and, unless k is
    the sortie's last decision (``done``), the next decision's rows and its
    effective mask. Raises ValueError when the steps, the stops and the
    rewards do not line up (a decision the reward does not read, or the
    reverse), or when the view's N is not the reward's.
    """
    from hermes.scheduler.selector.pair_features import pair_rows

    if not episode.steps and episode.sorties:
        raise ValueError("the episode kept no decisions: fly it with a trainer")
    out: List[PairTransition] = []
    for m, (steps, sortie, rewards) in enumerate(zip(episode.steps, episode.sorties,
                                                     episode.rewards)):
        if not len(steps) == len(sortie.stops) == len(rewards):
            raise ValueError(f"mission {m}: {len(steps)} decision(s), {len(sortie.stops)} "
                             f"Pass-1 stop(s) and {len(rewards)} reward(s) do not line up")
        rows = [pair_rows(step.view, schema)[0] for step in steps]
        for k, (step, stop, terms) in enumerate(zip(steps, sortie.stops, rewards)):
            devices = tuple(str(d) for d in step.view.arrival.devices)
            if devices != stop.devices or step.view.clock_s != stop.t_s:
                raise ValueError(f"mission {m}, decision {k}: the step is not the stop's "
                                 f"({devices} at {step.view.clock_s}, {stop.devices} at "
                                 f"{stop.t_s})")
            if int(step.view.demand) != sortie.n_demand:
                raise ValueError(f"mission {m}, decision {k}: the view's N is "
                                 f"{step.view.demand}, the reward's {sortie.n_demand}")
            done = k == len(steps) - 1
            if stop.terminal is not done:
                raise ValueError(f"mission {m}, decision {k}: terminal is {stop.terminal}")
            nxt = None if done else steps[k + 1]
            out.append(PairTransition(
                x=rows[k][step.row], reward=terms.total, done=done,
                next_rows=None if done else rows[k + 1],
                next_mask=None if done else np.array(nxt.effective_mask, dtype=bool)))
    return out


def e3_transitions(episode: EpisodeResult, steps: Sequence[Any]) -> List[PairTransition]:
    """E3's transitions of a training episode, in decision order.

    E3's sink sees every decision of the episode in order, one per Pass-1 stop
    flown (``chen_dqn.E3Step``), so mission m's decisions are the next
    ``len(sortie.stops)`` steps; each is checked against its stop (the same
    devices) and its view's N against the reward's. Transition k is the row of
    the stop taken, r_k (E3's bytes), and the next step's rows and mask, or
    ``done`` at the sortie's last decision.
    """
    out: List[PairTransition] = []
    at = 0
    for m, (sortie, rewards) in enumerate(zip(episode.sorties, episode.rewards)):
        mine = list(steps[at:at + len(sortie.stops)])
        at += len(sortie.stops)
        if len(mine) != len(sortie.stops) or len(rewards) != len(sortie.stops):
            raise ValueError(f"mission {m}: {len(mine)} E3 decision(s) for "
                             f"{len(sortie.stops)} Pass-1 stop(s)")
        for k, (step, stop, terms) in enumerate(zip(mine, sortie.stops, rewards)):
            if tuple(step.devices) != stop.devices:
                raise ValueError(f"mission {m}, decision {k}: E3 named {step.devices}, the "
                                 f"mule flew {stop.devices}")
            if int(step.view.demand) != sortie.n_demand:
                raise ValueError(f"mission {m}, decision {k}: E3's view has N = "
                                 f"{step.view.demand}, the reward {sortie.n_demand}")
            done = k == len(mine) - 1
            nxt = None if done else mine[k + 1]
            out.append(PairTransition(
                x=step.x, reward=terms.total, done=done,
                next_rows=None if done else nxt.matrix,
                next_mask=None if done else np.array(nxt.effective_mask, dtype=bool)))
    if at != len(steps):
        raise ValueError(f"E3 made {len(steps)} decision(s) and the mule flew {at} Pass-1 "
                         f"stop(s)")
    return out


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class ValidationPoint:
    """One validation of a run: after ``episode`` training episodes, the mean over
    the cells (``cells``: each cell's mean realized return) and the validation
    episodes flown, with the training window's numbers beside it."""

    episode: int
    score: float
    cells: Mapping[str, float]
    episodes: int
    train_return_mean: Optional[float] = None
    loss_mean: Optional[float] = None
    updates: int = 0
    transitions: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "episode": int(self.episode), "score": float(self.score),
            "cells": {k: float(v) for k, v in sorted(self.cells.items())},
            "episodes": int(self.episodes),
            "train_return_mean": self.train_return_mean, "loss_mean": self.loss_mean,
            "updates": int(self.updates), "transitions": int(self.transitions),
        }


def validate(net: PairQNet, spec: TrainSpec, *, bootstrap: Optional[PathLike] = None
             ) -> Tuple[float, Dict[str, float], int]:
    """The validation score of ``net`` now: (score, each cell's mean, episodes).

    Every validation episode (:func:`validation_episodes`) is flown greedily
    (ε = 0, no reference), on the run's plan, and read at the realized draw
    (:attr:`TrainSpec.validation_reward`); the score is the mean over the cells
    of each cell's mean return, so every cell weighs the same. E3 needs its
    run's ``bootstrap`` checkpoint (the config path).
    """
    reward = spec.validation_reward
    returns: Dict[str, List[float]] = {}
    schema = spec.schema() if spec.kind == KIND_PAIR_Q else None
    for cell, index, seed in validation_episodes(spec.cells, spec.val_episodes):
        if spec.kind == KIND_PAIR_Q:
            result = fly_pair_episode(net, schema, cell, seed, index=index, reward=reward,
                                      **_plan(spec))
        else:
            result = fly_e3_episode(net, cell, seed, index=index, reward=reward,
                                    bootstrap=bootstrap)[0]
        returns.setdefault(cell.name, []).append(result.ret)
    means = {name: statistics.fmean(values) for name, values in returns.items()}
    return statistics.fmean(means.values()), means, sum(len(v) for v in returns.values())


#: A validator: (the live network, the spec, the episodes trained so far) ->
#: (score, each cell's mean, episodes flown). :func:`validate` by default.
Validator = Callable[[PairQNet, TrainSpec, int], Tuple[float, Mapping[str, float], int]]


# --------------------------------------------------------------------------- #
# A run
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class TrainResult:
    """A finished run: its checkpoint, the sha of its arrays and its manifest."""

    path: str
    sha256: str
    manifest: Dict[str, Any]
    curve: Tuple[ValidationPoint, ...]
    best_episode: int
    episodes_run: int
    stopped_early: bool

    def summary(self) -> Dict[str, Any]:
        best = next(p for p in self.curve if p.episode == self.best_episode)
        return {"path": self.path, "sha256": self.sha256, "kind": self.manifest["kind"],
                "gamma": self.manifest["gamma"], "seed": self.manifest["seeds"]["run"],
                "best_episode": self.best_episode, "best_score": best.score,
                "episodes_run": self.episodes_run, "stopped_early": self.stopped_early}


def _new_network(spec: TrainSpec, seeds: Mapping[str, Any]) -> PairQNet:
    if spec.kind == KIND_PAIR_Q:
        return PairQNet(spec.schema().dim, spec.network, seed=seeds["init"])
    from hermes.scheduler.policies.chen_dqn import new_e3_network

    return new_e3_network(seed=seeds["init"], config=spec.network)


def _provenance(spec: TrainSpec, tree: TreeState, *, curve: Sequence[ValidationPoint],
                episodes_trained: int, outcome: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The manifest's provenance (``pair_q.PROVENANCE_KEYS``), held-out score empty."""
    training: Dict[str, Any] = {"spec": spec.to_json()}
    if outcome is not None:
        training["outcome"] = dict(outcome)
    return {
        "reward": spec.reward.to_json(),
        "training": training,
        "seeds": run_seeds(spec.seed),
        "cell_family": spec.family,
        "cell_family_sha256": spec.family_sha256,
        "trainer_commit": tree.commit,
        "dirty": tree.dirty,
        "episodes_trained": int(episodes_trained),
        "validation": [point.to_json() for point in curve],
        "held_out": None,
    }


def _save(net: PairQNet, spec: TrainSpec, path: PathLike, purpose: str,
          provenance: Mapping[str, Any]) -> str:
    if spec.kind == KIND_PAIR_Q:
        schema = spec.schema()
        return net.save(path, kind=KIND_PAIR_Q, purpose=purpose, schema=schema.to_json(),
                        classes=list(schema.classes), provenance=provenance)
    from hermes.scheduler.policies.chen_dqn import save_e3_checkpoint

    return save_e3_checkpoint(net, path, band=spec.band, purpose=purpose,
                              provenance=provenance)


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def train(spec: TrainSpec, path: PathLike, *, tree: TreeState,
          validator: Optional[Validator] = None, overwrite: bool = False,
          log: Optional[Callable[[str], None]] = None) -> TrainResult:
    """Train one run (module docstring) and save its best weights at ``path``.

    ``path`` is the checkpoint's ``.npz`` (its manifest goes beside it); an
    existing checkpoint there is refused unless ``overwrite``, and one of
    another cell family always (``checkpoints.replace_refusal``), before any
    episode flies. ``tree`` is the trainer's tree as the manifest records it
    (``checkpoints.tree_state``; the command line refuses a dirty one unless
    allowed). ``validator`` replaces :func:`validate` (tests script the
    scores); ``log`` receives one line per validation. Returns the saved
    checkpoint's sha, its manifest and the validation curve.
    """
    if not isinstance(spec, TrainSpec):
        raise TypeError(f"spec is a TrainSpec, got {spec!r}")
    if not isinstance(tree, TreeState):
        raise TypeError(f"tree is a TreeState (checkpoints.tree_state), got {tree!r}")
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"a checkpoint is an .npz file, got {str(path)!r}")
    refusal = replace_refusal(path, spec.family, overwrite=overwrite)
    if refusal is not None:
        raise FileExistsError(refusal)
    seeds = run_seeds(spec.seed)
    net = _new_network(spec, seeds)
    learner = PairQLearner(net, PairReplay(spec.learner.replay_capacity, seed=seeds["replay"]),
                           spec.learner)
    with tempfile.TemporaryDirectory(prefix="ferrysim-run-") as scratch:
        bootstrap = None
        if spec.kind == KIND_CHEN_DQN:
            # The config path E3 flies needs a checkpoint; the live network
            # replaces its weights before the policy's first call.
            bootstrap = Path(scratch) / "e3_bootstrap.npz"
            _save(net, spec, bootstrap, PURPOSE_BOOTSTRAP,
                  _provenance(spec, tree, curve=(), episodes_trained=0, outcome=None))
        check = validator if validator is not None else functools.partial(
            _validate_at, bootstrap=bootstrap)
        result = _run(spec, net, learner, check, bootstrap=bootstrap, log=log)
    curve, best_episode, best_weights, episodes_run, stopped = result
    net.set_weights(best_weights)
    outcome = {"episodes_run": episodes_run, "best_episode": best_episode,
               "stopped_early": stopped, "updates": net.updates,
               "transitions": learner.replay.pushed}
    manifest_provenance = _provenance(spec, tree, curve=curve, episodes_trained=best_episode,
                                      outcome=outcome)
    sha = _save(net, spec, path, PURPOSE_TRAINED, manifest_provenance)
    return TrainResult(path=str(path), sha256=sha, manifest=verify_checkpoint(path),
                       curve=tuple(curve), best_episode=best_episode, episodes_run=episodes_run,
                       stopped_early=stopped)


def _validate_at(net: PairQNet, spec: TrainSpec, episode: int, *,
                 bootstrap: Optional[PathLike]) -> Tuple[float, Dict[str, float], int]:
    return validate(net, spec, bootstrap=bootstrap)


def _run(spec: TrainSpec, net: PairQNet, learner: PairQLearner, check: Validator, *,
         bootstrap: Optional[PathLike], log: Optional[Callable[[str], None]]):
    """The run's loop: episodes, transitions and updates, validations, the best."""
    schema = spec.schema() if spec.kind == KIND_PAIR_Q else None
    curve: List[ValidationPoint] = []
    best: Optional[Tuple[float, int, Dict[str, np.ndarray]]] = None
    misses = 0
    window_returns: List[float] = []
    window_losses: List[float] = []
    episodes_run = 0
    for e in range(spec.episodes):
        cell, seed = training_episode(spec.seed, spec.family, spec.cells, e)
        behaviour = spec.learner.behaviour.at(e, spec.episodes)
        rng_seed = behaviour_seed(spec.seed, e)
        if spec.kind == KIND_PAIR_Q:
            trainer = Trainer(epsilon=behaviour.epsilon, rng_seed=rng_seed,
                              around_reference=behaviour.around_reference)
            result = fly_pair_episode(net, schema, cell, seed, index=e, reward=spec.reward,
                                      trainer=trainer, **_plan(spec))
            transitions = pair_transitions(result, schema)
        else:
            result, steps = fly_e3_episode(net, cell, seed, index=e, reward=spec.reward,
                                           bootstrap=bootstrap, epsilon=behaviour.epsilon,
                                           rng_seed=rng_seed)
            transitions = e3_transitions(result, steps)
        for transition in transitions:
            loss = learner.observe(transition)
            if loss is not None:
                window_losses.append(loss)
        window_returns.append(result.ret)
        episodes_run = e + 1
        if episodes_run % spec.eval_every and episodes_run != spec.episodes:
            continue
        score, cells, flown = check(net, spec, episodes_run)
        score = float(score)
        if not math.isfinite(score):
            raise FloatingPointError(f"validation after {episodes_run} episodes scored {score}")
        point = ValidationPoint(
            episode=episodes_run, score=score, cells=dict(cells), episodes=int(flown),
            train_return_mean=_mean_or_none(window_returns),
            loss_mean=_mean_or_none(window_losses), updates=net.updates,
            transitions=learner.replay.pushed)
        curve.append(point)
        window_returns, window_losses = [], []
        if best is None or score > best[0]:
            best = (score, episodes_run, net.weights())
            misses = 0
        else:
            misses += 1
        if log is not None:
            log(f"{spec.kind} gamma {spec.gamma:g} seed {spec.seed}: episode {episodes_run}, "
                f"validation {score:+.4f} (best {best[0]:+.4f} at {best[1]})")
        if misses >= spec.patience:
            break
    assert best is not None  # the last episode always validates
    stopped = episodes_run < spec.episodes
    return curve, best[1], best[2], episodes_run, stopped


# --------------------------------------------------------------------------- #
# Runs in worker processes (a sweep)
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class TrainTask:
    """One run of a sweep: its spec, its checkpoint's path and the tree it records."""

    spec: TrainSpec
    path: str
    tree: TreeState
    overwrite: bool = False


def run_training_task(task: TrainTask) -> Dict[str, Any]:
    """Train one :class:`TrainTask` (a worker's unit of work); its summary."""
    return train(task.spec, task.path, tree=task.tree, overwrite=task.overwrite).summary()


__all__ = [
    "ABLATIONS",
    "BYTES_TRAINING_REWARD",
    "E3_BEHAVIOUR",
    "EPISODES",
    "EVAL_EVERY",
    "HAND_TRAINING_REWARD",
    "KIND_REWARDS",
    "PATIENCE",
    "TRAINING_DEVICE_MODEL",
    "TRAINING_REWARD",
    "TRAIN_KINDS",
    "TrainResult",
    "TrainSpec",
    "TrainTask",
    "VAL_EPISODES",
    "ValidationPoint",
    "Validator",
    "behaviour_seed",
    "e3_spec",
    "e3_transitions",
    "fly_e3_episode",
    "fly_pair_episode",
    "pair_spec",
    "pair_transitions",
    "run_seeds",
    "run_training_task",
    "train",
    "training_episode",
    "validate",
    "validation_episodes",
]
