"""FerrySim's checkpoints: where they live, the tree a run records, how one flies and is scored.

FeRRy Phase 5, unit U8b (the Phase 5 spec, other choices 6; the user's
decision 9 (a); critic B9; orchestrator resolution R2). A trained score is a
format-2 checkpoint of the pair learner (``hermes.scheduler.selector.pair_q``):
an ``.npz`` of the online weights whose sha binds the header (the kind, the
purpose, the learner's revision, the network, the feature schema and the
classes), with a JSON manifest beside it. The trainer
(:mod:`experiments.ferrysim.train`) writes them; this module holds what
FerrySim adds around that format.

* **Where** (other choices 6). A study's checkpoints live at
  ``results/exp5/checkpoints/<study>/<tag>/g<γ>_s<seed>.npz``
  (:func:`checkpoint_path`), the manifest beside each. The tag is a learned
  arm's (``experiments.exp4.driver.CHECKPOINT_TAGS``: ``main``, ``hand``,
  ``dwell``, ``cov``, ``g0`` ... ``g99``, ``e3``), so a study's directory
  names the arm that flies it. The root is the caller's: the command line
  defaults to the repository's ``results/exp5/checkpoints``
  (:data:`CHECKPOINT_ROOT`), and tests and scratch runs name their own
  directory, so nothing is written into the repository unless a user's run
  asks for it. Nothing here commits (decision 9 (a): each
  study's final checkpoints are committed with the user's consent only).
  The layout has no cell family, so a study holds one family's runs: the
  calibration's two families (other choices 12) are two studies, and a run
  never replaces another family's checkpoint (:func:`replace_refusal`).
* **The tree** (critic B9). A manifest records the trainer's commit and
  whether its tree was dirty (:func:`tree_state`): "dirty" is any change git
  reports under the source trees FerrySim runs, ``hermes/`` and
  ``experiments/`` (:data:`TREE_PATHS`), untracked files included, since a
  checkpoint is reproducible from its commit and seeds only when the code that
  trained it is the commit's. The command line refuses a dirty tree unless
  ``--allow-dirty`` is given, and the manifest says so; the runner refuses a
  dirty checkpoint unless ``--allow-dirty-checkpoint`` is given.
* **Flying one** (:func:`checkpoint_flight`). A ``pair_q`` checkpoint flies
  FX's configuration with the pair slot installed around its verified score
  (``pair_features.load_pair_scorer``), the path training flies, on the plan
  it trained under (:func:`trained_plan`, the orchestrator's resolution R23:
  FQ-dwell's and FQ-cov's scores train under their arms' plans); a test pins
  that the config path (arm FQ, FQ-dwell or FQ-cov with the checkpoint, which
  stack trials fly) gives the same missions. A ``chen_dqn`` checkpoint flies
  arm E3 through the config path (``Exp4Driver.policy_checkpoints``), the
  only path E3 has.
* **What may fly one** (resolution R24; :func:`tag_refusals`). The runner
  flies a checkpoint under a learned arm's tag only when it is that arm's: a
  γ sweep tag's γ, the tag's reward (at decision 4 (a)'s weights under a γ
  sweep tag, ``dwell`` and ``cov``), and a network that took an update; and,
  for an FQ arm, only on the plan it trained under (:func:`plan_differences`
  against the plan the arm flies under the run's flags). The trainer's
  command line judges a run's spec by the same rules (:func:`tag_mismatches`)
  before it trains under a learned arm's tag.
* **Scoring one** (:func:`evaluate_checkpoints`, :func:`record_held_out_score`).
  Each checkpoint flies the held-out episodes of each cell (common random
  numbers with every reference, ``evaluate.evaluate``), and its manifest gets
  the held-out score, the mean undiscounted return over those episodes
  (``pair_q.HELD_OUT_KEYS``), with the reward it was scored under, so the
  runner's "no held-out score" refusal is lifted only by a real evaluation.

Every checkpoint read here is verified first (``pair_q.verify_checkpoint``),
so a manifest relabelled since its save is refused rather than flown or
scored (resolution R2).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from experiments.exp4.driver import CHECKPOINT_TAGS
from experiments.ferrysim import cells as C
from experiments.ferrysim import inprocess
from experiments.ferrysim.episode import ARM_FX, Policy, run_episode
from experiments.ferrysim.evaluate import episode_summary, parallel_map, score_summary
from experiments.ferrysim.reward import (
    DERIVED,
    REWARD_BYTES,
    REWARD_DERIVED,
    REWARD_HAND,
    RewardSpec,
)

#: The repository root: the directory that holds ``experiments`` and ``hermes``.
REPO_ROOT = Path(__file__).resolve().parents[2]
#: The checkpoint layout's root, under the repository root (other choices 6).
CHECKPOINT_ROOT = Path("results") / "exp5" / "checkpoints"
#: The source trees a run's "dirty" flag covers: what FerrySim imports.
TREE_PATHS: Tuple[str, ...] = ("hermes", "experiments")

#: E3's arm and its checkpoint tag (``experiments.exp4.driver``: arm ``E3``,
#: tag ``e3``), the config path a ``chen_dqn`` checkpoint flies.
ARM_E3 = "E3"
E3_TAG = "e3"
#: FQ-hand's tag, the score trained on F·hand (decision 4).
HAND_TAG = CHECKPOINT_TAGS["FQ-hand"]
#: Study 5.7's plan-term ablations' tags, FQ-dwell's and FQ-cov's (critic C2).
ABLATION_TAGS: Tuple[str, ...] = (CHECKPOINT_TAGS["FQ-dwell"], CHECKPOINT_TAGS["FQ-cov"])
#: A γ sweep tag, as :func:`gamma_tag` writes one (``g0`` ... ``g99``).
_GAMMA_TAG = re.compile(r"g[0-9]+")

#: What a directory name of the layout may not hold: Windows' reserved
#: characters (the driver's rule for its trace directories), so a study or a
#: tag is one plain path component.
_PATH_UNSAFE = '<>:"/\\|?*'
#: A commit's object name as ``git rev-parse HEAD`` prints it (SHA-1, or SHA-256
#: in a repository of that object format).
_COMMIT_HEX = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Where checkpoints live
# --------------------------------------------------------------------------- #

def path_component(value: Any, what: str) -> str:
    """``value`` as one directory name of the layout; ValueError otherwise.

    Refused rather than rewritten, so two studies or tags never share a
    directory: no reserved character, no leading or trailing dot or space,
    not ``.`` or ``..``.
    """
    if (not isinstance(value, str) or not value or value in (".", "..")
            or value != value.strip(" .")
            or any(c in _PATH_UNSAFE or ord(c) < 32 for c in value)):
        raise ValueError(
            f"{what} {value!r} names a directory of the checkpoint layout: it must be a plain "
            f"path component (none of {_PATH_UNSAFE!r}, no leading or trailing dot or space)")
    return value


def gamma_tag(gamma: float) -> str:
    """The γ sweep's checkpoint tag of ``gamma``: ``g0``, ``g25``, ..., ``g99``.

    The driver's tags (``CHECKPOINT_TAGS``: ``FQ-g0`` ... ``FQ-g99``) are γ in
    hundredths, so a γ that is not a whole number of hundredths has no tag
    (ValueError) and needs one named by the caller.
    """
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma is a number in [0, 1], got {gamma!r}")
    hundredths = round(float(gamma) * 100.0)
    if abs(float(gamma) * 100.0 - hundredths) > 1e-9:
        raise ValueError(f"gamma {gamma!r} is not a whole number of hundredths: name its tag")
    return f"g{hundredths}"


def checkpoint_name(gamma: float, seed: int) -> str:
    """``g<γ>_s<seed>.npz``, the layout's file name (other choices 6)."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"a training seed is an int >= 0, got {seed!r}")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma is a number in [0, 1], got {gamma!r}")
    return f"g{float(gamma):g}_s{seed}.npz"


def checkpoint_path(root: PathLike, study: str, tag: str, gamma: float, seed: int) -> Path:
    """``<root>/<study>/<tag>/g<γ>_s<seed>.npz`` (other choices 6)."""
    return (Path(root) / path_component(study, "the study") / path_component(tag, "the tag")
            / checkpoint_name(gamma, seed))


def default_root() -> Path:
    """The command line's root: the repository's ``results/exp5/checkpoints``."""
    return REPO_ROOT / CHECKPOINT_ROOT


def replace_refusal(path: PathLike, family: str, *, overwrite: bool) -> Optional[str]:
    """Why a run over the cell family ``family`` may not save its checkpoint at
    ``path``; None when it may.

    A checkpoint there (its ``.npz`` or its manifest) is replaced only when
    asked (``overwrite``, the command line's ``--overwrite``), and never by a
    run of another family, even when asked: the layout has no family
    (:func:`checkpoint_path`), so the calibration's jittery and clean runs
    (other choices 12) meet at one path under one study, and replacing one
    with the other would leave a single family's score where the study
    needs both. One study per family keeps both. The family is read from the
    manifest's ``cell_family`` (outside the sha, but written by the save
    beside the arrays it describes); a manifest that is missing, unreadable
    or names no family names none to protect.
    """
    from hermes.scheduler.selector.pair_q import manifest_path

    npz = Path(path)
    beside = manifest_path(npz)
    if not (npz.exists() or beside.exists()):
        return None
    try:
        held = json.loads(beside.read_text(encoding="utf-8")).get("cell_family")
    except (OSError, ValueError, AttributeError):
        held = None
    if held is not None and held != family:
        return (f"{npz} holds a checkpoint of the cell family {held!r}, not this run's "
                f"{family!r}, and is never replaced by it: the layout has no family, so "
                f"give each family its own --study")
    if not overwrite:
        return (f"{npz} exists: a checkpoint is replaced only when asked (overwrite=True, "
                f"--overwrite)")
    return None


# --------------------------------------------------------------------------- #
# The trainer's tree
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class TreeState:
    """The trainer's tree as a manifest records it: the commit (None when git
    cannot say) and whether the source trees differ from it."""

    commit: Optional[str]
    dirty: bool

    def __post_init__(self) -> None:
        if self.commit is not None and not (isinstance(self.commit, str) and self.commit):
            raise ValueError(f"commit is a non-empty string or None, got {self.commit!r}")
        if not isinstance(self.dirty, bool):
            raise TypeError(f"dirty is a bool, got {self.dirty!r}")


def _git(args: Sequence[str], repo: Path) -> str:
    out = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True,
                         encoding="utf-8", timeout=60, check=True)
    return out.stdout


def tree_state(repo: PathLike = REPO_ROOT) -> TreeState:
    """The commit of ``repo``'s HEAD and whether :data:`TREE_PATHS` are dirty.

    Dirty is anything ``git status`` reports under those trees: a modified,
    staged, deleted or untracked file (ignored files aside). When git is
    missing or ``repo`` is not a work tree, the commit is None and the tree
    counts as dirty, since nothing then shows that the code is a commit's.
    """
    repo = Path(repo)
    try:
        head = _git(["rev-parse", "HEAD"], repo).strip()
        status = _git(["status", "--porcelain", "--untracked-files=normal", "--",
                       *TREE_PATHS], repo)
    except (OSError, subprocess.SubprocessError):
        return TreeState(commit=None, dirty=True)
    if not _COMMIT_HEX.fullmatch(head):
        return TreeState(commit=None, dirty=True)
    return TreeState(commit=head, dirty=bool(status.strip()))


# --------------------------------------------------------------------------- #
# What a checkpoint trained under, and what may fly it
# --------------------------------------------------------------------------- #

def plan_score_settings(params: Any, what: str = "plan_score_params") -> Dict[str, Any]:
    """``params`` as plan score settings: a plain dict of ``PlanScoreParams`` fields.

    Read as the driver and the mule read ``plan_score_params``
    (``PlanScoreParams.from_mapping``: an unknown field, or a value of the
    wrong type or range, is refused), and JSON-ready, since a manifest records
    it (the orchestrator's resolution R23). ``{}`` is the default plan, read
    without loading the plan package.
    """
    if not isinstance(params, Mapping):
        raise TypeError(f"{what} is a mapping of PlanScoreParams settings, got {params!r}")
    out = dict(params)
    if out:
        from hermes.scheduler.plan.types import PlanScoreParams

        PlanScoreParams.from_mapping(out)
        try:
            json.dumps(out, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{what} are recorded as JSON: {e}") from None
    return out


def trained_plan(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """The plan score settings a checkpoint trained under (resolution R23).

    Its training spec's ``plan_score_params`` (``train.TrainSpec``), read as
    :func:`plan_score_settings` reads them; ``{}``, the cells' default plan,
    when the spec records none, since a run on that plan records none and no
    run did before the record existed. The record is provenance, outside the
    sha, so a check on it catches a mistake, not a hand edit.
    """
    training = manifest.get("training")
    spec = training.get("spec") if isinstance(training, Mapping) else None
    if not isinstance(spec, Mapping):
        return {}
    return plan_score_settings(spec.get("plan_score_params", {}),
                               "the training spec's plan_score_params")


def plan_differences(trained: Mapping[str, Any], flown: Mapping[str, Any]
                     ) -> Dict[str, Tuple[Any, Any]]:
    """Where two plan score settings differ: {setting: (``trained``'s, ``flown``'s)}.

    Each side is read with ``PlanScoreParams``' defaults filled in, so ``{}``
    (the default plan) equals the defaults written out and 0 equals 0.0; the
    settings come in ``PlanScoreParams``' field order, and none differ when
    the two plans are one (resolution R23).
    """
    from hermes.scheduler.plan.types import PlanScoreParams

    mine = PlanScoreParams.from_mapping(dict(trained)).as_dict()
    theirs = PlanScoreParams.from_mapping(dict(flown)).as_dict()
    return {name: (mine[name], theirs[name]) for name in mine if mine[name] != theirs[name]}


def kept_updates(manifest: Mapping[str, Any]) -> Optional[int]:
    """The updates the network a checkpoint keeps had taken; None when not recorded.

    The trainer keeps the weights of its best validation and records each
    validation's update count (``train.ValidationPoint.updates``), so the kept
    weights' count is that of the validation at ``episodes_trained``, the kept
    episode (``report`` reads the kept validation score the same way). 0 means
    the kept weights are the network's initial ones: the replay had not
    warmed up (``LearnerSettings.warmup_transitions``) when they were kept.
    """
    kept = [point for point in manifest.get("validation") or ()
            if isinstance(point, Mapping)
            and point.get("episode") == manifest.get("episodes_trained")]
    if len(kept) != 1:
        return None
    updates = kept[0].get("updates")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        return None
    return updates


def tag_reward(tag: str) -> str:
    """The reward kind the checkpoint of a learned arm's ``tag`` trains on: F·hand
    for ``hand`` (FQ-hand, decision 4), E3's bytes for ``e3`` (decision 7 (a);
    the trainer's only reward for E3, ``train.KIND_REWARDS``) and the derived
    reward for every other tag (decision 4 (a))."""
    if tag == HAND_TAG:
        return REWARD_HAND
    if tag == E3_TAG:
        return REWARD_BYTES
    return REWARD_DERIVED


def tag_weights(tag: str) -> Optional[Tuple[float, float]]:
    """The derived reward's weights (c_t, c_cov) the checkpoint of a learned arm's ``tag``
    trains at, where the arm fixes them (resolution R24; the Phase 5 repair round's
    A-4): decision 4 (a)'s own (``reward.DERIVED``) under a γ sweep tag (Study 5.5's
    sweep) and under ``dwell`` and ``cov`` (Study 5.7's plan-term ablations, critic
    C2, which change FQ's plan and not its reward). None under ``main``, FQ's tag,
    which flies its checkpoint at the weights the manifest records (Study 5.7's
    weight grid's included; the row's sha leads to them), and under ``hand`` and
    ``e3``, whose rewards have no such weights (:func:`tag_reward`)."""
    if _GAMMA_TAG.fullmatch(tag) or tag in ABLATION_TAGS:
        return float(DERIVED.c_t), float(DERIVED.c_cov)
    return None


def _number(value: Any) -> str:
    """A setting as a reason names it: a number as ``f"{x:g}"``, anything else as repr."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}"
    return repr(value)


def tag_mismatches(tag: str, gamma: Any, reward: Mapping[str, Any]) -> List[str]:
    """What a checkpoint trained at ``gamma`` on ``reward`` (``RewardSpec.to_json``'s
    keys) holds that the learned arm of ``tag`` does not fly; empty when nothing,
    and for a tag no learned arm flies (not in ``CHECKPOINT_TAGS``: a run's own
    directory, as Study 5.7's weight grid names one).

    The orchestrator's resolution R24, and the Phase 5 repair round's A-4:

    * a γ sweep tag ``g<X>`` flies γ = X/100 (:func:`gamma_tag`);
    * the tag's reward kind (:func:`tag_reward`);
    * the weights the tag fixes (:func:`tag_weights`): under a γ sweep tag,
      ``dwell`` and ``cov``, the derived reward at decision 4 (a)'s c_t and
      c_cov.

    The runner reads them off a checkpoint's manifest (:func:`tag_refusals`), and
    the trainer's command line off a run's spec before it trains under an
    explicit ``--tag``, so the trainer saves under a learned arm's tag only what
    the runner flies there.
    """
    if tag not in CHECKPOINT_TAGS.values():
        return []
    reasons: List[str] = []
    if _GAMMA_TAG.fullmatch(tag):
        try:
            own = gamma_tag(gamma)
        except ValueError:
            own = None
        if own != tag:
            named = (f"which flies as tag {own!r}" if own is not None
                     else "which no tag names (γ in whole hundredths)")
            reasons.append(f"its γ is {_number(gamma)}, {named}, not {tag!r}")
    want = tag_reward(tag)
    kind = reward.get("kind")
    if kind != want:
        reasons.append(f"it trained on the {kind!r} reward, and tag {tag!r} flies the {want!r} "
                       f"reward's score")
    weights = tag_weights(tag)
    if kind == REWARD_DERIVED and weights is not None:
        c_t, c_cov = reward.get("c_t"), reward.get("c_cov")
        if (c_t, c_cov) != weights:
            reasons.append(f"it trained on the derived reward at c_t {_number(c_t)}, c_cov "
                           f"{_number(c_cov)}, and tag {tag!r} flies it at decision 4 (a)'s "
                           f"weights, c_t {weights[0]:g} and c_cov {weights[1]:g}")
    return reasons


def tag_refusals(tag: str, manifest: Mapping[str, Any]) -> List[str]:
    """Why a campaign may not fly the checkpoint of ``manifest`` as the learned arm of
    ``tag``; empty when it may (the orchestrator's resolution R24).

    The runner judges these on the manifest ``pair_q.verify_checkpoint``
    returns, once ``pair_q.campaign_refusals`` has passed it (critic B9):

    * what it trained on that the tag's arm does not fly (:func:`tag_mismatches`):
      a γ sweep tag's γ (the header's network's, which the sha binds), the tag's
      reward kind, and decision 4 (a)'s weights under a γ sweep tag, ``dwell``
      and ``cov``; the reward is outside the sha, so these catch a mistake
      rather than a hand edit;
    * a network that took an update: kept weights that took none
      (:func:`kept_updates`) are the network's initial ones, a random-init
      arm in substance (the spec, other choices 5), and a manifest without
      the record shows no update either.

    The plan an FQ arm flies depends on the run's flags, so the runner
    compares it itself (:func:`trained_plan`, :func:`plan_differences`).
    """
    reasons = tag_mismatches(tag, manifest.get("gamma"), manifest.get("reward") or {})
    updates = kept_updates(manifest)
    if updates is None:
        reasons.append(f"its manifest records no update count for the weights it keeps (no "
                       f"validation at its kept episode {manifest.get('episodes_trained')!r})")
    elif updates == 0:
        reasons.append(f"its network took no update: the weights it keeps (validated after "
                       f"episode {manifest['episodes_trained']}) are its initial ones")
    return reasons


# --------------------------------------------------------------------------- #
# Flying a checkpoint
# --------------------------------------------------------------------------- #

def _verified(path: PathLike) -> Dict[str, Any]:
    from hermes.scheduler.selector.pair_q import verify_checkpoint

    return verify_checkpoint(path)


def checkpoint_label(manifest: Mapping[str, Any]) -> str:
    """A checkpoint's label in results: ``ckpt-`` and the first 12 hex digits of
    its sha, one per checkpoint whatever its path."""
    return f"ckpt-{manifest['sha256'][:12]}"


def checkpoint_flight(path: PathLike, *, label: Optional[str] = None,
                      manifest: Optional[Mapping[str, Any]] = None
                      ) -> Tuple[Policy, Dict[str, Any]]:
    """What flies a checkpoint in FerrySim: its :class:`Policy` and the driver settings
    the episode adds (``run_episode``'s ``driver_overrides``).

    A ``pair_q`` checkpoint: FX's configuration with a fresh pair slot around
    the verified learned score (``pair_features.load_pair_scorer`` over the
    link's classes, against the manifest's sha), on the plan it trained under
    (:func:`trained_plan`; resolution R23): ``plan_score_params`` when its
    training spec records them, no extra settings on the default plan. A
    ``chen_dqn`` checkpoint: arm E3 with ``policy_checkpoints = {"e3": path}``,
    which the driver verifies and the mule loads. The checkpoint is verified
    here (``manifest`` is the verified one, when the caller has it), and the
    policy can be sent to a worker process (a ``functools.partial`` of the
    loader). ``label`` defaults to :func:`checkpoint_label`.
    """
    from hermes.scheduler.selector.pair_features import load_pair_scorer
    from hermes.scheduler.selector.pair_q import KIND_CHEN_DQN, KIND_PAIR_Q

    if manifest is None:
        manifest = _verified(path)
    label = checkpoint_label(manifest) if label is None else label
    if manifest["kind"] == KIND_PAIR_Q:
        scorer = functools.partial(load_pair_scorer, str(path),
                                   expect_sha256=str(manifest["sha256"]),
                                   classes=tuple(manifest["classes"]))
        plan = trained_plan(manifest)
        return (Policy(label=label, arm=ARM_FX, scorer=scorer),
                {"plan_score_params": plan} if plan else {})
    if manifest["kind"] == KIND_CHEN_DQN:
        return Policy(label=label, arm=ARM_E3), {"policy_checkpoints": {E3_TAG: str(path)}}
    raise ValueError(f"{path}: a {manifest['kind']!r} checkpoint flies in no FerrySim arm")


# --------------------------------------------------------------------------- #
# Scoring checkpoints on the held-out stream
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class CheckpointTask:
    """One episode of one checkpoint (a worker's unit of work)."""

    cell: C.FerryCell
    stream: str
    index: int
    seed: int
    path: str
    policy: Policy
    overrides: Mapping[str, Any]
    reward: RewardSpec = DERIVED
    device_model: str = inprocess.DEVICE_MODEL_EQUAL


def run_checkpoint_task(task: CheckpointTask) -> Dict[str, Any]:
    """Fly one :class:`CheckpointTask`; its episode summary (``evaluate.episode_summary``)."""
    result = run_episode(task.cell, task.seed, task.policy, trial_index=task.index,
                         reward=task.reward, device_model=task.device_model,
                         driver_overrides=dict(task.overrides))
    out = episode_summary(result)
    out["stream"] = task.stream
    out["checkpoint"] = task.path
    return out


def evaluate_checkpoints(
    paths: Sequence[PathLike],
    cells: Iterable[Any],
    *,
    stream: str = C.HELDOUT_STREAM,
    episodes: int = 1000,
    start: int = 0,
    reward: RewardSpec = DERIVED,
    device_model: str = inprocess.DEVICE_MODEL_EQUAL,
    workers: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    """Every checkpoint on episodes ``start`` .. ``start + episodes - 1`` of each cell.

    The episodes are each cell's ``stream`` (the held-out stream by default),
    the same for every checkpoint and every reference (``evaluate.evaluate``
    on the same arguments), so differences are paired (common random
    numbers). Returns each path's episode summaries, cell-major then episode,
    the summaries' ``policy`` being the checkpoint's label. Refuses a training
    stream (critic B14) and two paths of one checkpoint.
    """
    resolved = [C.cell_named(cell) for cell in cells]
    if C.stream_kind(stream) == C.STREAM_TRAIN:
        raise ValueError("checkpoints are scored on the validation or held-out stream, never "
                         "a training stream (critic B14)")
    flights = []
    labels: Dict[str, str] = {}
    for path in paths:
        manifest = _verified(path)
        policy, overrides = checkpoint_flight(path, manifest=manifest)
        if policy.label in labels:
            raise ValueError(f"{path} and {labels[policy.label]} are one checkpoint "
                             f"({manifest['sha256']})")
        labels[policy.label] = str(path)
        flights.append((str(path), policy, overrides))
    tasks = []
    for cell in resolved:
        seeds = C.stream_seeds(stream, cell.name, int(episodes), start=int(start))
        for path, policy, overrides in flights:
            for e, seed in enumerate(seeds):
                tasks.append(CheckpointTask(cell=cell, stream=stream, index=int(start) + e,
                                            seed=seed, path=path, policy=policy,
                                            overrides=overrides, reward=reward,
                                            device_model=device_model))
    results = parallel_map(run_checkpoint_task, tasks, workers=workers)
    out: Dict[str, List[Dict[str, Any]]] = {path: [] for path, _, _ in flights}
    for summary in results:
        out[summary["checkpoint"]].append(summary)
    return out


def held_out_score(summaries: Sequence[Mapping[str, Any]],
                   reward: RewardSpec) -> Dict[str, Any]:
    """A checkpoint's held-out score over its episodes of one or more cells.

    ``episodes`` and ``return_mean`` (``pair_q.HELD_OUT_KEYS``): every episode
    scored, and the mean over the cells of each cell's mean undiscounted
    return (equal weight per cell, as the trainer's validation score), with
    each cell's ``evaluate.score_summary``, the stream, the first episode's
    index and the reward it was scored under. Refuses summaries of more than
    one policy or stream, or of no held-out stream.
    """
    if not summaries:
        raise ValueError("a held-out score needs at least one episode")
    for key in ("policy", "stream"):
        values = {s.get(key) for s in summaries}
        if len(values) != 1:
            raise ValueError(f"a held-out score is of one {key}: got {sorted(map(str, values))}")
    stream = summaries[0]["stream"]
    if C.stream_kind(stream) != C.STREAM_HELDOUT:
        raise ValueError(f"a held-out score reads {C.HELDOUT_STREAM!r}, got {stream!r}")
    by_cell: Dict[str, List[Mapping[str, Any]]] = {}
    for s in summaries:
        by_cell.setdefault(s["cell"], []).append(s)
    cells = {name: score_summary(group) for name, group in sorted(by_cell.items())}
    return {
        "episodes": sum(c["episodes"] for c in cells.values()),
        "return_mean": statistics.fmean(c["return_mean"] for c in cells.values()),
        "stream": stream,
        "first_index": min(c["first_index"] for c in cells.values()),
        "reward": reward.to_json(),
        "cells": cells,
    }


def record_held_out_score(path: PathLike, summaries: Sequence[Mapping[str, Any]],
                          reward: RewardSpec) -> Dict[str, Any]:
    """Write a checkpoint's held-out score (:func:`held_out_score`) into its manifest.

    ``pair_q.record_held_out``: the checkpoint is verified first and only the
    manifest's ``held_out`` changes, which is outside the sha. Returns the new
    manifest. The summaries must be this checkpoint's (their ``checkpoint``,
    when present, names this path).
    """
    from hermes.scheduler.selector.pair_q import record_held_out

    named = {s.get("checkpoint") for s in summaries} - {None}
    if named and named != {str(path)}:
        raise ValueError(f"the summaries are of {sorted(named)}, not {str(path)!r}")
    return record_held_out(path, held_out_score(summaries, reward))


def checkpoint_files(paths: Iterable[PathLike]) -> List[Path]:
    """The checkpoints ``paths`` name: a file as it is, a directory's ``.npz``
    files at any depth, in sorted order; ValueError for a path that is neither."""
    out: List[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.npz")))
        elif path.suffix == ".npz" and path.is_file():
            out.append(path)
        else:
            raise ValueError(f"{str(item)!r} is neither a checkpoint (.npz) nor a directory")
    return out


__all__ = [
    "ABLATION_TAGS",
    "ARM_E3",
    "CHECKPOINT_ROOT",
    "CheckpointTask",
    "E3_TAG",
    "HAND_TAG",
    "REPO_ROOT",
    "TREE_PATHS",
    "TreeState",
    "checkpoint_files",
    "checkpoint_flight",
    "checkpoint_label",
    "checkpoint_name",
    "checkpoint_path",
    "default_root",
    "evaluate_checkpoints",
    "gamma_tag",
    "held_out_score",
    "kept_updates",
    "path_component",
    "plan_differences",
    "plan_score_settings",
    "record_held_out_score",
    "replace_refusal",
    "run_checkpoint_task",
    "tag_mismatches",
    "tag_refusals",
    "tag_reward",
    "tag_weights",
    "trained_plan",
    "tree_state",
]
