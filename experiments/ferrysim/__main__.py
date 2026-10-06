"""FerrySim's command line: ``python -m experiments.ferrysim <command>`` (FeRRy Phase 5, unit U8b).

The commands that fly FerrySim episodes run for minutes to hours, and none
runs unless a user runs it (the Phase 5 spec, "What needs your go-ahead":
training campaigns, the sweep and its evaluation wait for the user's
go-ahead). The commands, in a campaign's order (the spec, other choices 12):

``headroom``
    The headroom report on the validation stream, and ε
    (``experiments.ferrysim.headroom``; the user's decision 10 (i)(a)), on
    the cells' own plan or ``--plan-score-params``: a sweep trained under a
    pilot's plan reads its ε from a headroom report flown on that plan.
``train``
    One training run (``experiments.ferrysim.train``): ``--kind pair_q`` (the
    FQ arms' score) or ``chen_dqn`` (E3), at ``--gamma`` from ``--seed``, over
    ``--family``'s cells, saved at ``<root>/<study>/<tag>/g<γ>_s<seed>.npz``
    with its manifest (``experiments.ferrysim.checkpoints``). The tag defaults
    to the arm's: ``g<100 γ>`` for the pair score on the derived reward
    (Study 5.5's ``FQ-g0`` ... ``FQ-g99``), ``hand`` on F·hand, ``e3`` for E3.
    The pair score trains under the plan its arm flies (the orchestrator's
    resolution R23): ``--ablation dwell`` or ``cov`` gives FQ-dwell's or
    FQ-cov's (the driver's own settings), and its tag, and
    ``--plan-score-params`` a pilot's settings, as the runner's flag takes
    them; the manifest records the plan, and the runner flies the checkpoint
    only on it. A learned arm's tag given as ``--tag`` takes only a run that
    arm flies (resolutions R23 and R24): a γ sweep tag its γ, every tag its
    arm's reward, at decision 4 (a)'s weights under a γ sweep tag, ``dwell``
    and ``cov``, and ``dwell`` or ``cov`` its ablation's plan settings; Study
    5.7's weight grid saves under a tag of its own.
``sweep``
    Every (γ, seed) of a grid, one run per worker process. The calibration
    comes first (other choices 12), one study per family, since the layout has
    no family: ``--study 5.5-calibration-jittery --family jittery`` and
    ``--study 5.5-calibration-clean --family clean``, each with ``--gammas 0
    0.9 --seeds 0 1 2``. Then Study 5.5: ``--study 5.5 --gammas 0 0.25 0.5
    0.75 0.9 0.99 --seeds 0 1 2 3 4 5 6 7 8 9`` on the jittery family. A
    jittery score that also practises at Study 5.6's periods trains on
    ``--family jittery56`` (the jittery cells and Study 5.6's four), under a
    study of its own; whether it does is the user's choice before the 5.5
    sweep (the orchestrator's resolution R22).
``evaluate``
    Checkpoints and the references (the FX and F arms, the slot's scripted
    rules) on the held-out episodes of ``--cells`` (common random numbers),
    each read under one reward at the realized draw (``--reward``, the derived
    reward by default), written to ``--out`` as the report reads it. The cells
    default to the jittery family's: the N = 12 cells the rule is read on and
    the N = 6 control the report shows beside them (decision 5 (a)); a clean
    study names the clean cells, and Study 5.6's cells fly only when named
    (resolution R22). ``--record`` writes each checkpoint's
    held-out score, with that reward, into its manifest, which the runner
    needs before a campaign flies it (critic B9). A pair checkpoint flies the
    plan it trained under, and the references the cells' own or
    ``--plan-score-params``; beside the references, a checkpoint must have
    trained on theirs, so one file reads its policies on one plan.
``report``
    Study 5.5's rule on an evaluation file (``experiments.ferrysim.report``),
    read on ``--cells`` (by default Study 5.5's two N = 12 cells, jit-n12-90
    and jit-n12-180, whichever family trained the score: resolution R22), with
    ε given (``--epsilon``) or read from the headroom report (``--headroom``),
    which must have flown the plan the evaluation's policies flew (resolution
    R23).
    The verdict says whether the sweep is decision 5 (a)'s pre-registered grid
    (its six γ, 10 seeds each, 1,000 held-out episodes per cell) and, if not,
    why: a sweep off it, the calibration's for one, is read by the same rule
    and labelled, not refused (resolution R26).

``pilot``
    The Exp 5 addendum's budget pilot and decision cost
    (``experiments.ferrysim.pilot``; Study 5.11 (a) and (c)), its own options
    (``pilot --help``): every (cell, budget, policy) on the first episodes of
    each cell's validation stream, with the served share per mission and the
    episode's, the planner's and each flight decision's wall time; the
    scale family's knee is read off its table.

``train`` and ``sweep`` refuse a dirty tree unless ``--allow-dirty`` is given,
and the manifest records the tree's commit and dirty flag (other choices 6).
They never overwrite a checkpoint unless ``--overwrite`` is given, and never
one of another cell family even then (``checkpoints.replace_refusal``); a
sweep checks every path before its first run. The default root is the
repository's ``results/exp5/checkpoints``; committing what it holds needs the
user's consent (decision 9 (a)).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from experiments.ferrysim import cells as C
from experiments.ferrysim.reward import REWARD_BYTES, REWARD_DERIVED, REWARD_HAND, RewardSpec

PROG = "python -m experiments.ferrysim"
COMMANDS = ("headroom", "train", "sweep", "evaluate", "report", "pilot")


def _write_json(path: str, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")


# --------------------------------------------------------------------------- #
# train and sweep
# --------------------------------------------------------------------------- #

def _add_run_args(ap: argparse.ArgumentParser) -> None:
    from experiments.ferrysim import train as T

    ap.add_argument("--kind", choices=T.TRAIN_KINDS, default="pair_q",
                    help="pair_q: the FQ arms' score; chen_dqn: E3 (default pair_q).")
    ap.add_argument("--family", choices=sorted(C.FAMILIES), default=C.FAMILY_JITTERY,
                    help="The cell family the score practises over (decision 3; jittery56: "
                         "the jittery cells and Study 5.6's, resolution R22).")
    ap.add_argument("--study", required=True,
                    help="The study's directory under the root, one per cell family (e.g. "
                         "5.5, 5.5-calibration-jittery, 5.5-calibration-clean).")
    ap.add_argument("--tag", default=None,
                    help="The checkpoint tag (default: the arm's). A learned arm's tag takes "
                         "only a run that arm flies: its γ, reward and plan.")
    ap.add_argument("--root", default=None,
                    help="The checkpoint root (default: the repository's "
                         "results/exp5/checkpoints).")
    ap.add_argument("--episodes", type=int, default=T.EPISODES,
                    help=f"Training episodes (default {T.EPISODES}).")
    ap.add_argument("--eval-every", type=int, default=T.EVAL_EVERY,
                    help=f"Episodes between validations (default {T.EVAL_EVERY}).")
    ap.add_argument("--val-episodes", type=int, default=T.VAL_EPISODES,
                    help=f"Validation episodes, over the family's cells (default "
                         f"{T.VAL_EPISODES}).")
    ap.add_argument("--patience", type=int, default=T.PATIENCE,
                    help=f"Stop after this many validations without a new best (default "
                         f"{T.PATIENCE}).")
    ap.add_argument("--reward", choices=(REWARD_DERIVED, REWARD_HAND, REWARD_BYTES),
                    default=None, help="The training reward (default: derived for pair_q, "
                                       "bytes for chen_dqn).")
    ap.add_argument("--c-t", type=float, default=None,
                    help="The derived reward's time weight (default 0.1).")
    ap.add_argument("--c-cov", type=float, default=None,
                    help="The derived reward's coverage weight (default 1).")
    ap.add_argument("--lr", type=float, default=None,
                    help="Adam's learning rate (default the pair learner's, 1e-3).")
    ap.add_argument("--epsilon-start", type=float, default=None,
                    help="The behaviour's first ε (default 0.3; critic C4).")
    ap.add_argument("--epsilon-end", type=float, default=None,
                    help="The behaviour's last ε (default 0.05).")
    ap.add_argument("--reference-episodes", type=int, default=None,
                    help="Episodes flown around FX's pair (pair_q only; default 500).")
    ap.add_argument("--no-phase", action="store_true",
                    help="Leave the pair rows' phase block out (pair_q only).")
    ap.add_argument("--ablation", choices=sorted(T.ABLATIONS), default=None,
                    help="Train FQ-dwell's or FQ-cov's score under its arm's plan (Study "
                         "5.7, critic C2), on the derived reward, under that arm's tag "
                         "(pair_q only).")
    ap.add_argument("--plan-score-params", default=None, metavar="JSON",
                    help="The plan score's settings the run trains under, as the runner's "
                         "--plan-score-params takes them (a JSON object of PlanScoreParams "
                         "fields), e.g. a pilot's kappa; --ablation's go on top and may not "
                         "be repeated here. The runner flies the checkpoint only on this "
                         "plan (pair_q only).")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="Train from a dirty tree; the manifest records it.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace an existing checkpoint at the path.")


def _reward(args: argparse.Namespace) -> RewardSpec:
    from experiments.ferrysim import train as T

    kind = args.reward or (REWARD_BYTES if args.kind == "chen_dqn" else REWARD_DERIVED)
    if kind != REWARD_DERIVED and (args.c_t is not None or args.c_cov is not None):
        raise ValueError(f"--c-t and --c-cov weigh the derived reward, not {kind!r}")
    if kind == REWARD_DERIVED:
        base = T.TRAINING_REWARD
        return dataclasses.replace(
            base, c_t=base.c_t if args.c_t is None else args.c_t,
            c_cov=base.c_cov if args.c_cov is None else args.c_cov)
    return T.HAND_TRAINING_REWARD if kind == REWARD_HAND else T.BYTES_TRAINING_REWARD


def _run_plan(args: argparse.Namespace, ap: argparse.ArgumentParser) -> Dict[str, Any]:
    """The plan score settings a run trains under (resolution R23): ``--plan-score-params``,
    read by the runner's own parser, with ``--ablation``'s settings on top, which
    it may not set too; {} for the cells' own plan."""
    from experiments.exp4.runner_main import _json_object
    from experiments.ferrysim import train as T
    from experiments.ferrysim.checkpoints import plan_score_settings

    plan = _json_object(args.plan_score_params, "--plan-score-params", ap)
    try:
        plan_score_settings(plan, "--plan-score-params")
    except (TypeError, ValueError) as e:
        raise ValueError(f"--plan-score-params: {e}") from None
    if args.ablation is not None:
        own = T.ABLATIONS[args.ablation]
        shared = sorted(set(plan) & set(own))
        if shared:
            raise ValueError(f"--ablation {args.ablation} sets {shared} itself: "
                             f"--plan-score-params may not set them too")
        plan.update(own)
    return plan


def _spec(args: argparse.Namespace, gamma: float, seed: int,
          plan: Optional[Dict[str, Any]] = None):
    from experiments.ferrysim import train as T
    from hermes.scheduler.selector.pair_q import BehaviourSchedule, LearnerSettings, PairQConfig

    network = PairQConfig() if args.lr is None else PairQConfig(lr=args.lr)
    default = T.E3_BEHAVIOUR if args.kind == "chen_dqn" else BehaviourSchedule()
    behaviour = dataclasses.replace(default, **{
        name: value for name, value in (("epsilon_start", args.epsilon_start),
                                        ("epsilon_end", args.epsilon_end),
                                        ("reference_episodes", args.reference_episodes))
        if value is not None})
    common = dict(family=args.family, network=network, reward=_reward(args),
                  learner=LearnerSettings(behaviour=behaviour), episodes=args.episodes,
                  eval_every=args.eval_every, val_episodes=args.val_episodes,
                  patience=args.patience, plan_score_params=dict(plan or {}))
    if args.kind == "chen_dqn":
        if args.no_phase:
            raise ValueError("--no-phase is the pair rows' (pair_q only)")
        if plan:
            raise ValueError("--ablation and --plan-score-params set the plan the pair score "
                             "trains under (pair_q only): E3 flies legacy mode, with no plan")
        return T.e3_spec(gamma, seed, **common)
    return T.pair_spec(gamma, seed, phase=not args.no_phase, **common)


def _tag_mismatches(tag: str, spec) -> List[str]:
    """What a run of ``spec`` holds that the learned arm of ``tag`` does not fly: what
    ``checkpoints.tag_mismatches`` finds (a γ sweep tag's γ, the tag's reward, and
    decision 4 (a)'s weights under a γ sweep tag, dwell and cov; resolution R24), and
    under an ablation's tag a plan without the ablation's own settings, which its
    arm flies whatever a pilot's plan adds (resolution R23). Empty for a tag no
    learned arm flies (the Phase 5 repair round's A-2 and A-4)."""
    from experiments.ferrysim import train as T
    from experiments.ferrysim.checkpoints import plan_differences, tag_mismatches

    reasons = tag_mismatches(tag, spec.gamma, spec.reward.to_json())
    own = T.ABLATIONS.get(tag)
    if own is not None:
        differ = {name: pair for name, pair in plan_differences(own, spec.plan_score_params)
                  .items() if name in own}
        if differ:
            reasons.append(f"the arm flies the plan score settings {dict(own)}, and this run "
                           f"trains under " + ", ".join(
                               f"{name} {theirs!r}" for name, (_own, theirs) in differ.items())
                           + f": give --ablation {tag} (or those settings in "
                             f"--plan-score-params)")
    return reasons


def _tag(args: argparse.Namespace, spec) -> str:
    from experiments.exp4.driver import CHECKPOINT_TAGS
    from experiments.ferrysim import train as T
    from experiments.ferrysim.checkpoints import E3_TAG, gamma_tag, tag_reward

    arms = {tag: name for name, tag in CHECKPOINT_TAGS.items()}
    ablation = getattr(args, "ablation", None)
    if ablation is not None:
        # The ablation's arm flies its own tag's checkpoint (resolutions R23, R24).
        arm = arms[ablation]
        if args.tag is not None and args.tag != ablation:
            raise ValueError(f"--ablation {ablation} trains arm {arm}'s score, whose tag is "
                             f"{ablation!r}: --tag {args.tag!r} contradicts it")
        if spec.reward.kind != tag_reward(ablation) or (spec.reward.c_t, spec.reward.c_cov) != (
                T.TRAINING_REWARD.c_t, T.TRAINING_REWARD.c_cov):
            raise ValueError(f"--ablation {ablation} trains arm {arm}'s score on the derived "
                             f"reward at its own weights (decision 4 (a)), the one its arm "
                             f"flies")
        return ablation
    if args.tag is not None:
        # A learned arm's tag, given, holds only what that arm flies, so no run trains
        # for a checkpoint the runner refuses (the repair round's A-2 and A-4).
        reasons = _tag_mismatches(args.tag, spec)
        if reasons:
            raise ValueError(f"--tag {args.tag} names arm {arms[args.tag]}'s checkpoint, which "
                             f"the runner flies only when it is that arm's (resolutions R23 "
                             f"and R24): " + "; ".join(reasons))
        return args.tag
    if spec.kind == "chen_dqn":
        return E3_TAG
    if spec.reward.kind == REWARD_HAND:
        return "hand"
    if (spec.reward.c_t, spec.reward.c_cov) != (T.TRAINING_REWARD.c_t, T.TRAINING_REWARD.c_cov):
        raise ValueError("a derived reward with other weights (Study 5.7's grid) needs its "
                         "own --tag")
    return gamma_tag(spec.gamma)


def _tree(args: argparse.Namespace, ap: argparse.ArgumentParser):
    from experiments.ferrysim.checkpoints import tree_state

    tree = tree_state()
    if tree.dirty and not args.allow_dirty:
        ap.error("the tree is dirty (hermes/ or experiments/ differ from "
                 f"{tree.commit or 'any commit'}): commit first, or pass --allow-dirty and "
                 "the manifest records it")
    return tree


def _path(args: argparse.Namespace, spec):
    """The run's checkpoint path; ValueError when the run may not save there."""
    from experiments.ferrysim.checkpoints import checkpoint_path, default_root, replace_refusal

    path = checkpoint_path(args.root or default_root(), args.study, _tag(args, spec),
                           spec.gamma, spec.seed)
    refusal = replace_refusal(path, spec.family, overwrite=args.overwrite)
    if refusal is not None:
        raise ValueError(refusal)
    return path


def _cmd_train(args: argparse.Namespace, ap: argparse.ArgumentParser) -> int:
    from experiments.ferrysim import train as T

    try:
        spec = _spec(args, args.gamma, args.seed, _run_plan(args, ap))
        path = _path(args, spec)
    except (TypeError, ValueError) as e:
        ap.error(str(e))
    tree = _tree(args, ap)
    result = T.train(spec, path, tree=tree, overwrite=args.overwrite, log=print)
    print(json.dumps(result.summary(), sort_keys=True))
    return 0


def _cmd_sweep(args: argparse.Namespace, ap: argparse.ArgumentParser) -> int:
    from experiments.ferrysim import train as T
    from experiments.ferrysim.evaluate import parallel_map

    tasks: List[Any] = []
    try:
        if len(set(args.gammas)) != len(args.gammas) or len(set(args.seeds)) != len(args.seeds):
            raise ValueError("a sweep names each γ and each seed once")
        plan = _run_plan(args, ap)
        tree = _tree(args, ap)
        for gamma in args.gammas:
            for seed in args.seeds:
                spec = _spec(args, gamma, seed, plan)
                tasks.append(T.TrainTask(spec=spec, path=str(_path(args, spec)), tree=tree,
                                         overwrite=args.overwrite))
    except (TypeError, ValueError) as e:
        ap.error(str(e))
    for summary in parallel_map(T.run_training_task, tasks, workers=args.workers):
        print(json.dumps(summary, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #

def _by_cell(summaries: Sequence[Dict[str, Any]], cells: Sequence[str]) -> Dict[str, Any]:
    """Returns and decisions per sortie of one policy, per cell in episode order."""
    out: Dict[str, Any] = {"returns": {c: [] for c in cells}, "decisions": {c: [] for c in cells}}
    for s in sorted(summaries, key=lambda s: (cells.index(s["cell"]), s["index"])):
        out["returns"][s["cell"]].append(s["return"])
        out["decisions"][s["cell"]].append(list(s["decisions"]))
    return out


def evaluation_file(*, cells: Sequence[str], stream: str, episodes: int, start: int,
                    reward: RewardSpec, references: Dict[str, Sequence[Dict[str, Any]]],
                    checkpoints: Dict[str, Sequence[Dict[str, Any]]],
                    manifests: Dict[str, Dict[str, Any]],
                    plan_score_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The evaluation file (``report.EVALUATION_FORMAT``) from the episodes' summaries.

    ``plan_score_params``, the plan the references flew when it is not the
    cells' own (resolution R23), is recorded only when given, so a file of
    the cells' own plan is the one it always was.
    """
    from experiments.ferrysim.report import EVALUATION_FORMAT

    cells = list(cells)
    out = {
        "format": EVALUATION_FORMAT,
        "stream": stream,
        "episodes": int(episodes),
        "start": int(start),
        "cells": cells,
        "reward": reward.to_json(),
        "seeds": {c: list(C.stream_seeds(stream, c, int(episodes), start=int(start)))
                  for c in cells},
        "references": {label: _by_cell(group, cells) for label, group in references.items()},
        "checkpoints": [dict(_by_cell(group, cells), path=path, manifest=manifests[path])
                        for path, group in checkpoints.items()],
    }
    if plan_score_params:
        out["plan_score_params"] = dict(plan_score_params)
    return out


def _evaluation_reward(args: argparse.Namespace) -> RewardSpec:
    """The reward an evaluation reads, at the realized draw (other choices 7)."""
    from experiments.ferrysim.reward import BYTES, DERIVED, HAND

    kind = args.reward
    if kind != REWARD_DERIVED and (args.c_t is not None or args.c_cov is not None):
        raise ValueError(f"--c-t and --c-cov weigh the derived reward, not {kind!r}")
    if kind == REWARD_DERIVED:
        return dataclasses.replace(
            DERIVED, c_t=DERIVED.c_t if args.c_t is None else args.c_t,
            c_cov=DERIVED.c_cov if args.c_cov is None else args.c_cov)
    return HAND if kind == REWARD_HAND else BYTES


def _evaluation_plan(args: argparse.Namespace, ap: argparse.ArgumentParser,
                     paths: Sequence[Path], references: Sequence[str]) -> Dict[str, Any]:
    """The plan score settings the references fly (resolution R23).

    ``--plan-score-params``, read by the runner's own parser; {} for the cells'
    own plan. Each pair checkpoint flies the plan it trained under
    (``checkpoints.checkpoint_flight``), so one flown beside references must
    have trained on theirs, or the file would read its policies on two plans;
    and with no reference the flag would set nothing. Both are refused.
    """
    from experiments.exp4.runner_main import _json_object
    from experiments.ferrysim import checkpoints as K
    from hermes.scheduler.selector.pair_q import KIND_PAIR_Q, verify_checkpoint

    plan = _json_object(args.plan_score_params, "--plan-score-params", ap)
    try:
        K.plan_score_settings(plan, "--plan-score-params")
    except (TypeError, ValueError) as e:
        raise ValueError(f"--plan-score-params: {e}") from None
    if not references:
        if plan:
            raise ValueError("--plan-score-params is the plan the references fly, and none "
                             "flies here: each checkpoint flies the plan it trained under")
        return plan
    for path in paths:
        manifest = verify_checkpoint(path)
        if manifest["kind"] != KIND_PAIR_Q:
            continue
        differ = K.plan_differences(K.trained_plan(manifest), plan)
        if differ:
            raise ValueError(
                f"{path} trained under other plan score settings than the references fly: "
                + ", ".join(f"{name} (trained {mine!r}, the references {theirs!r})"
                            for name, (mine, theirs) in differ.items())
                + "; give its plan as --plan-score-params, or evaluate it with "
                  "--no-references, so that one file reads its policies on one plan")
    return plan


def _cmd_evaluate(args: argparse.Namespace, ap: argparse.ArgumentParser) -> int:
    from experiments.ferrysim import checkpoints as K
    from experiments.ferrysim.episode import reference_policies
    from experiments.ferrysim.evaluate import evaluate
    from hermes.scheduler.selector.pair_q import verify_checkpoint

    stream = {"heldout": C.HELDOUT_STREAM, "val": C.VAL_STREAM}[args.stream]
    try:
        cells = [C.cell_named(c).name for c in args.cells]
        paths = K.checkpoint_files(args.checkpoints)
        refs = {p.label: p for p in reference_policies()}
        wanted = [] if args.no_references else (args.references or list(refs))
        unknown = [r for r in wanted if r not in refs]
        if unknown:
            raise ValueError(f"unknown references {unknown}; they are {sorted(refs)}")
        if not paths and not wanted:
            raise ValueError("nothing to evaluate: no checkpoint and no reference")
        if args.record and stream != C.HELDOUT_STREAM:
            raise ValueError("--record writes a held-out score: evaluate the held-out stream")
        reward = _evaluation_reward(args)
        plan = _evaluation_plan(args, ap, paths, wanted)
    except (TypeError, ValueError) as e:
        ap.error(str(e))
    # The references' plan: none added on the cells' own plan (resolution R23).
    on_plan = {"driver_overrides": {"plan_score_params": plan}} if plan else {}
    summaries = evaluate(cells, [refs[r] for r in wanted], stream=stream,
                         episodes=args.episodes, start=args.start, reward=reward,
                         workers=args.workers, **on_plan) if wanted else []
    by_ref: Dict[str, List[Dict[str, Any]]] = {r: [] for r in wanted}
    for s in summaries:
        by_ref[s["policy"]].append(s)
    by_ckpt = K.evaluate_checkpoints(paths, cells, stream=stream, episodes=args.episodes,
                                     start=args.start, reward=reward, workers=args.workers)
    manifests = {}
    for path, group in by_ckpt.items():
        if args.record:
            manifests[path] = K.record_held_out_score(path, group, reward)
        else:
            manifests[path] = verify_checkpoint(path)
    data = evaluation_file(cells=cells, stream=stream, episodes=args.episodes,
                           start=args.start, reward=reward, references=by_ref,
                           checkpoints=by_ckpt, manifests=manifests, plan_score_params=plan)
    for label, entry in list(data["references"].items()) + [
            (Path(e["path"]).name, e) for e in data["checkpoints"]]:
        means = ", ".join(f"{c} {sum(v) / len(v):+.4f}" for c, v in entry["returns"].items() if v)
        print(f"{label:24s} {means}")
    _write_json(args.out, data)
    print(f"wrote {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def _cmd_report(args: argparse.Namespace, ap: argparse.ArgumentParser) -> int:
    from experiments.ferrysim import report as R

    try:
        with open(args.evaluation, encoding="utf-8") as fh:
            evaluation = json.load(fh)
        if (args.epsilon is None) == (args.headroom is None):
            raise ValueError("give ε as --epsilon or read it from --headroom (one of the two)")
        if args.headroom is not None:
            # ε on the plan the evaluation's policies flew (none recorded: the cells')
            with open(args.headroom, encoding="utf-8") as fh:
                epsilon = R.epsilon_from_headroom_report(
                    json.load(fh), args.cells,
                    plan_score_params=evaluation.get("plan_score_params", {}))
        else:
            epsilon = args.epsilon
        table = R.sweep_table(evaluation, cells=args.cells, epsilon=epsilon)
        # decide refuses a table without the references the rule reads
        verdict = R.decide(table)
    except (OSError, KeyError, TypeError, ValueError) as e:
        ap.error(str(e))
    print(R.format_verdict(verdict))
    if args.out:
        _write_json(args.out, {"evaluation": str(args.evaluation),
                               "epsilon_source": ("headroom: " + str(args.headroom)
                                                  if args.headroom else "given"),
                               "verdict": verdict.to_json()})
        print(f"wrote {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# The parser
# --------------------------------------------------------------------------- #

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True, metavar="{" + ",".join(COMMANDS) + "}")
    sub.add_parser("headroom", help="The headroom report (its own options: headroom --help).",
                   add_help=False)
    sub.add_parser("pilot", help="The Exp 5 addendum's budget pilot and decision cost (its own "
                                 "options: pilot --help).", add_help=False)
    train = sub.add_parser("train", help="One training run.")
    _add_run_args(train)
    train.add_argument("--gamma", type=float, required=True, help="The run's γ, in [0, 1].")
    train.add_argument("--seed", type=int, required=True, help="The run's training seed.")
    sweep = sub.add_parser("sweep", help="Every (γ, seed) of a grid, in worker processes.")
    _add_run_args(sweep)
    sweep.add_argument("--gammas", type=float, nargs="+", required=True,
                       help="The grid's γ (Study 5.5: 0 0.25 0.5 0.75 0.9 0.99).")
    sweep.add_argument("--seeds", type=int, nargs="+", required=True,
                       help="The training seeds (Study 5.5: 0 to 9).")
    sweep.add_argument("--workers", type=int, default=1,
                       help="Worker processes, one run each at a time (default 1).")
    evaluate = sub.add_parser("evaluate", help="Checkpoints and references on held-out runs.")
    evaluate.add_argument("--checkpoints", nargs="*", default=[],
                          help="Checkpoint files, or directories searched for .npz files.")
    evaluate.add_argument("--cells", nargs="+",
                          default=[c.name for c in C.FAMILIES[C.FAMILY_JITTERY]],
                          help="The cells flown (default: the jittery family's, Study 5.5's "
                               "N = 12 cells and the N = 6 control, decision 5; Study 5.6's "
                               "cells only when named).")
    evaluate.add_argument("--stream", choices=("heldout", "val"), default="heldout")
    evaluate.add_argument("--episodes", type=int, default=1000,
                          help="Episodes per cell (default 1000, other choices 12).")
    evaluate.add_argument("--start", type=int, default=0,
                          help="The first episode's index in the stream (default 0).")
    evaluate.add_argument("--references", nargs="+", default=None,
                          help="Reference labels (default: FX, F and the four scripted ones).")
    evaluate.add_argument("--no-references", action="store_true")
    evaluate.add_argument("--reward", choices=(REWARD_DERIVED, REWARD_HAND, REWARD_BYTES),
                          default=REWARD_DERIVED,
                          help="The reward every policy is read under (default derived).")
    evaluate.add_argument("--c-t", type=float, default=None,
                          help="The derived reward's time weight (default 0.1).")
    evaluate.add_argument("--c-cov", type=float, default=None,
                          help="The derived reward's coverage weight (default 1).")
    evaluate.add_argument("--workers", type=int, default=1,
                          help="Worker processes, one episode each at a time (default 1).")
    evaluate.add_argument("--plan-score-params", default=None, metavar="JSON",
                          help="The plan score's settings the references fly, as the runner's "
                               "--plan-score-params takes them (default: the cells' own). A "
                               "pair checkpoint flies the plan it trained under, which beside "
                               "the references must be theirs.")
    evaluate.add_argument("--record", action="store_true",
                          help="Write each checkpoint's held-out score into its manifest.")
    evaluate.add_argument("--out", required=True, help="The evaluation file (JSON).")
    report = sub.add_parser("report", help="Study 5.5's rule on an evaluation file.")
    report.add_argument("--evaluation", required=True, help="The evaluate command's file.")
    report.add_argument("--epsilon", type=float, default=None, help="ε, fixed before the sweep.")
    report.add_argument("--headroom", default=None,
                        help="A headroom report (JSON) to read ε from, flown on the "
                             "evaluation's plan (headroom --plan-score-params).")
    report.add_argument("--cells", nargs="+", default=[c.name for c in C.STUDY_5_5_CELLS],
                        help="The cells the rule is read on (default: Study 5.5's, "
                             "jit-n12-90 and jit-n12-180, decision 5, whichever family "
                             "trained the score).")
    report.add_argument("--out", default=None, help="Write the verdict here (JSON).")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "headroom":
        from experiments.ferrysim.headroom import main as headroom_main

        return headroom_main(argv[1:])
    if argv and argv[0] == "pilot":
        from experiments.ferrysim.pilot import main as pilot_main

        return pilot_main(argv[1:])
    ap = parser()
    args = ap.parse_args(argv)
    command = {"train": _cmd_train, "sweep": _cmd_sweep, "evaluate": _cmd_evaluate,
               "report": _cmd_report}[args.command]
    # HERMES' logs below errors are off while the command runs, and put back
    # after it, so an in-process caller (a test) keeps its own level
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        return command(args, ap)
    finally:
        logging.disable(previous)


if __name__ == "__main__":
    sys.exit(main())
