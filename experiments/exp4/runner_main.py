"""Experiment-4 CLI entry point (chunk EX-4.0).

Drives the integrated two-pass orchestrator over a small grid via the
shared :class:`~experiments.runner.TrialRunner` and writes a resumable
per-trial CSV. This is the first *genuinely integrated* L2+L3
measurement — real subprocesses, real TCP, real cross-mule FedAvg —
distinct from Experiment 3's abstracted sim.

Usage::

    python -m experiments.exp4.runner_main \\
        --csv results/exp4_smoke.csv \\
        --N 2 --rrf 60 --n-missions 1 --n-trials 1

Defaults are a tiny smoke grid (each trial spawns a real process tree,
so keep the grid small until the paper run). Only arm **H1** exists in
EX-4.0; H0/H2/H3 arrive in later chunks.

FeRRy Phase 4: the plan arms (``driver.PLAN_ARMS``) run only when named with
``--arms``, on ``--mission-clock sim``; the default arm list is still the nine
Phase 3 arms (``driver.DEFAULT_ARMS``). The plan flags set the plan arms' age
cap, score and search settings (the score's and search's as JSON objects, so a
pilot can sweep kappa and the coverage rank), and ``--member-admission`` the
H and D arms' member subsets. A pilot takes its own ``--base-seed`` (the
Phase 4 spec, decision 7), since the grid derives every trial's seed from it.

FeRRy Phase 5: the learned arms and ``H1+L1`` (``driver.PHASE_5_ARMS``) run
only when named. An FQ arm flies the pair checkpoint of its tag
(``--pair-checkpoint TAG=PATH``, repeatable) and E3 its policy checkpoint
(``--policy-checkpoint E3=PATH``). The runner is a campaign's entry, so it
refuses a checkpoint a campaign may not fly (critic B9): one that is not
``trained``, trained on no episode, has no held-out score, or was trained from
a dirty tree without ``--allow-dirty-checkpoint``, each judged on the verified
manifest. It also refuses one that is not its tag's (the orchestrator's
resolution R24): a ``g<X>`` tag's γ other than X/100, a reward other than its
arm's (F·hand for ``hand``, E3's bytes for ``e3``, the derived reward
otherwise, at decision 4 (a)'s weights under ``g<X>``, ``dwell`` and ``cov``),
or a network that took no update; and a pair checkpoint trained
under other plan score settings than its arm flies under this run's flags
(resolution R23: FQ-dwell's and FQ-cov's plans, ``--plan-score-params``).
The driver then checks each file as its mule will. The opt-in
``--require-trained`` refuses H2 and H3 without ``--selector-weights``, whose
selector would be random-init (decision 8 (a)); off, the default, a run is
the recorded one. None of these flags is a grid axis, so each setting of them
gets a CSV of its own.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from experiments.runner import TrialGrid, TrialRunner

from hermes.mission.aggregation_rules import IMPLEMENTED_RULES
from hermes.scheduler.policies.fedcs_degraded import VALUE_KINDS as FEDCS_VALUES
from hermes.scheduler.policies.whittle import VARIANTS as WHITTLE_VARIANTS
from hermes.scheduler.policies.whittle import WEIGHT_MODES as WHITTLE_WEIGHTS
from hermes.scheduler.stages.s3b_feasibility import DEADLINE_BOUNDS

from .driver import (
    ADDENDUM_ARMS,
    CHECKPOINT_TAGS,
    DEFAULT_ARMS,
    PAIR_CHECKPOINT_TAGS,
    PHASE_5_ARMS,
    PLAN_ARMS,
    POLICY_CHECKPOINT_TAGS,
    PROVENANCE_COLUMNS,
    Exp4Driver,
)
from .metrics import Exp4MetricSummary

log = logging.getLogger("experiments.exp4.runner_main")


def _build_grid(
    *,
    arms: Sequence[str],
    Ns: Sequence[int],
    rrfs: Sequence[float],
    n_missions_values: Sequence[int],
    regimes: Sequence[str],
    dead_zones: Sequence[float],
    link_qualities: Sequence[float],
    n_trials: int,
    base_seed: int = 42,
) -> TrialGrid:
    return TrialGrid(
        independent_vars={
            "N": list(Ns),
            "rrf": list(rrfs),
            "n_missions": list(n_missions_values),
            "regime": list(regimes),
            "dead_zone": list(dead_zones),
            "link_quality": list(link_qualities),
        },
        arms=list(arms),
        n_trials=n_trials,
        base_seed=base_seed,
    )


#: FeRRy Phase 3 physics flags -> ``Exp4Driver.ferry_physics`` keys
#: (``MuleConfig`` fields). Only the flags given are passed; the rest keep the
#: design's defaults (Phase 3 design section 1, decisions D1-D3).
_PHYSICS_FLAGS = (
    ("snr_floor_db", float, "D1: the SNR floor, CQI 1 (dB; default -6.7)."),
    ("altitude_m", float, "D1: the mule's altitude (m; default 25)."),
    ("n_pl", float, "D1: path-loss exponent (default 2.2; 3.0 is the sensitivity case)."),
    ("shadow_sigma_db", float, "D1/D2: shadowing sigma (dB; default 4)."),
    ("margin_quantile", float, "D1: edge-availability quantile of R(b) (default 0.9)."),
    ("contact_regime", str, "D2: contact interference regime, clean (default) or "
                            "jittery (test (c))."),
    ("interference_period_s", float, "D2: interference period P_c (s; default 60)."),
    ("interference_amp_db", float, "Exp 5 addendum (Study 5.15): the contact channel's "
                                   "interference amplitude A (dB; default the regime's, "
                                   "1 clean and 5 jittery)."),
    ("interference_sigma_db", float, "Exp 5 addendum (Study 5.15): the interference noise "
                                     "sigma_I (dB; default the regime's, 0.4 clean and 1.5 "
                                     "jittery)."),
    ("narrow_range_ratio", float, "Exp 5 addendum (unit U10, Study 5.4): the narrow class's "
                                  "planar reach as a multiple of --rrf (default the D1 "
                                  "derivation's, 3.87); its mean SNR curve moves with it."),
    ("noise_bin_s", float, "D2: noise bin (s; default 1)."),
    ("shadow_corr_s", float, "D2: shadowing correlation time (s; default 7.4)."),
    ("shadow_keying", str, "D2: shadowing keyed by 'time' (default) or 'position'."),
    ("cruise_speed_m_s", float, "D3: cruise speed (m/s; default 5)."),
    ("turnaround_s", float, "D3: dock turnaround once per mission (s; default 30)."),
    ("listen_s", float, "D3: listen window per contact with a missing reply (s; default 1)."),
    ("energy_capacity_j", float, "D3: SIMULATED battery capacity; switches the energy "
                                 "clause on (binds only with a budget; default off)."),
    ("p_move_w", float, "D3: SIMULATED flight power (W; default Zeng 2019 at the speed)."),
    ("p_hover_w", float, "D3: SIMULATED hover power (W; default Zeng 2019, 168.5)."),
)


def _add_phase_3_flags(parser: argparse.ArgumentParser) -> None:
    """FeRRy Phase 3 — the mission clock and the contact link (mule arms).

    Every flag defaults to the recorded run. On the wall clock the driver
    accepts only a numeric ``--deadline-time-scale``, ``--initial-window-s``,
    ``--session-ttl-s`` and the RF token (``--t-nom-layouts`` is accepted but
    unused); every other flag, the ``t_nom`` time unit and Φ₀ in missions
    included, needs ``--mission-clock sim``.
    """
    g = parser.add_argument_group("FeRRy Phase 3: mission clock and contact link")
    g.add_argument(
        "--mission-clock", choices=("wall", "sim"), default="wall",
        help="'sim' flies every mule arm on the simulated mission clock with the "
             "contact link (hermes/mule/ferry.py); 'wall' (default) is every "
             "recorded run. H0 has no simulated round time and is refused (or "
             "dropped from the default arm list). Write to a fresh CSV: the "
             "Phase 3 provenance columns change the header.",
    )
    g.add_argument("--contact-band", default=None,
                   help="Band class every stop flies (wide, medium, narrow; the "
                        "Phase 3 re-baselines fly wide). Omit for the channel-free "
                        "control.")
    g.add_argument("--contact-band-classes", nargs="+", default=None,
                   help="The link's band classes (default wide medium narrow; add "
                        "medium_wide for the 10 MHz option).")
    g.add_argument("--in-flight-response", choices=("abort", "replan"), default="abort",
                   help="When the rest of a pass stops fitting: abort (Amendment 8, "
                        "default) or replan.")
    g.add_argument("--replan-fallback", choices=("reorder", "trim"), default="reorder",
                   help="The re-plan's fallback for our arms (unit U4).")
    g.add_argument("--backhaul-model", choices=("mission", "seconds"), default="mission",
                   help="mission (default): the recorded loss schedule / flat pct; "
                        "seconds: the seconds-axis channel, H3 adaptive, keyed loss "
                        "draw. Losses per upload are ~16%% jittery at a fixed carrier "
                        "(critic A4).")
    g.add_argument("--contact-reliability-source", choices=("origin", "channel"),
                   default="origin",
                   help="origin (default): the devices' own rel x rf_factor draw; "
                        "channel: the SNR gate plus the availability drawn on the mule "
                        "(needs --contact-band).")
    g.add_argument("--payload-bytes", type=int, default=None,
                   help="Declared payload per direction (bytes); omit for measured.")
    g.add_argument("--deadline-bounds", choices=DEADLINE_BOUNDS,
                   default="collection",
                   help="What Deadline(j) bounds (spec Q2): the collection, arrival + "
                        "dwell (default); 'delivery_per_stop': per stop, that stop's "
                        "own return to the dock plus the upload (it does not bound "
                        "when the earlier stops' updates actually reach the cluster; "
                        "'delivery' before the route-level variant existed); or "
                        "'delivery': route-level, the route's landing plus the upload "
                        "meets the deadline of every update collected on it (a stop "
                        "that would make an update on board late is refused as "
                        "'delivery').")
    g.add_argument("--backhaul-period-s", type=float, default=None,
                   help="Seconds backhaul period P_bh (s); default n_missions x T_nom.")
    g.add_argument("--t-nom-s", type=float, default=None,
                   help="T_nom (s) for every cell; default: computed per cell when a "
                        "setting needs it.")
    g.add_argument("--t-nom-layouts", type=int, default=20,
                   help="Reference layouts T_nom is the median over (default 20).")
    g.add_argument("--deadline-time-scale", default="1.0",
                   help="The deadline law's time unit: a number (1.0 = recorded, "
                        "either clock) or 't_nom' for T_nom / 10 s.")
    g.add_argument("--initial-window-s", type=float, default=None,
                   help="Φ₀ in the law's recorded unit (default 60 s).")
    g.add_argument("--initial-window-missions", type=float, default=None,
                   help="Φ₀ as a number of nominal mission periods (critic A7).")
    g.add_argument("--agg-period-t-nom", action="store_true",
                   help="agg:cutoff: set the D5 period_s to the cell's T_nom.")
    g.add_argument("--session-ttl-s", type=float, default=None,
                   help="The mule's wall-clock session TTL (default 3 s); ferry cells "
                        "set it from the measured real-model fit time (>= 2x).")
    g.add_argument("--rf-link-token", action=argparse.BooleanOptionalAction, default=None,
                   help="Give each trial one RF link token (Amendment 10). Default: on "
                        "exactly with --mission-clock sim.")
    g.add_argument("--expected-input-dim", type=int, default=None,
                   help="The input width a ferry cell's real model must have (design "
                        "R8); default 21 on the canonical data.")
    for name, kind, text in _PHYSICS_FLAGS:
        g.add_argument(f"--{name.replace('_', '-')}", dest=f"phys_{name}", type=kind,
                       default=None, help=text)


def _phase_3_driver_kwargs(args, parser: argparse.ArgumentParser) -> dict:
    """The ``Exp4Driver`` keywords of the Phase 3 flags."""
    scale = args.deadline_time_scale
    if scale != "t_nom":
        try:
            scale = float(scale)
        except ValueError:
            parser.error(f"--deadline-time-scale must be a number or 't_nom', got {scale!r}")
    physics = {
        name: getattr(args, f"phys_{name}")
        for name, _kind, _text in _PHYSICS_FLAGS
        if getattr(args, f"phys_{name}") is not None
    }
    return dict(
        mission_clock=args.mission_clock,
        contact_band=args.contact_band,
        contact_band_classes=(
            None if args.contact_band_classes is None else list(args.contact_band_classes)
        ),
        in_flight_response=args.in_flight_response,
        replan_fallback=args.replan_fallback,
        backhaul_model=args.backhaul_model,
        contact_reliability_source=args.contact_reliability_source,
        payload_bytes=args.payload_bytes,
        deadline_bounds=args.deadline_bounds,
        ferry_physics=physics,
        backhaul_period_s=args.backhaul_period_s,
        t_nom_s=args.t_nom_s,
        t_nom_layouts=int(args.t_nom_layouts),
        deadline_time_scale=scale,
        initial_window_s=args.initial_window_s,
        initial_window_missions=args.initial_window_missions,
        agg_period_t_nom=bool(args.agg_period_t_nom),
        session_ttl_s=args.session_ttl_s,
        rf_link_token=args.rf_link_token,
        expected_input_dim=args.expected_input_dim,
    )


def _add_phase_4_flags(parser: argparse.ArgumentParser) -> None:
    """FeRRy Phase 4 — the plan arms' settings and member subsets (simulated clock).

    Every flag defaults to the recorded run, and none is a grid axis, so the
    trial seeds do not move. On the wall clock the driver refuses them all.
    """
    g = parser.add_argument_group("FeRRy Phase 4: the plan clock")
    g.add_argument(
        "--member-admission", choices=("whole", "subset"), default=None,
        help="Member subsets (decision 4 (b)): 'subset' lets a stop that fails whole "
             "admit the members that still fit, 'whole' is the recorded rule. Default: "
             "each arm's own, 'subset' for the plan arms and 'whole' for H1-H3, D1-D3 "
             "and D5; D4 always runs whole.",
    )
    g.add_argument(
        "--age-cap-missions", type=int, default=None,
        help="The plan arms' age cap S (decision 1): a device not merged for S of its "
             "mule's missions must be served. Default off; F-cap always runs without it.",
    )
    g.add_argument(
        "--age-cap-lookahead", type=int, default=0,
        help="The cap's lookahead L: a device is capped from age S - L (default 0).",
    )
    g.add_argument(
        "--plan-score-params", default=None, metavar="JSON",
        help="The plan score's settings as a JSON object of PlanScoreParams fields, "
             "e.g. '{\"c_cov_per_device\": 0.25, \"c_energy\": 0, \"coverage_rank\": "
             "\"weighted\"}' for the pilot's sweep of kappa, c4 and the coverage rank; "
             "unknown keys are refused. F-cov sets the coverage term off on top.",
    )
    g.add_argument(
        "--plan-search-params", default=None, metavar="JSON",
        help="The plan search's bounds as a JSON object of PlanSearchParams fields "
             "(exact_max_devices, exhaustive_max_stops, heuristic_max_passes, "
             "heuristic_max_evaluations).",
    )


def _json_object(text: Optional[str], flag: str, parser: argparse.ArgumentParser) -> dict:
    """A JSON-object flag's value as a dict ({} when not given)."""
    if text is None:
        return {}
    try:
        value = json.loads(text)
    except ValueError as e:
        parser.error(f"{flag} must be a JSON object: {e}")
    if not isinstance(value, dict):
        parser.error(f"{flag} must be a JSON object, got {text!r}")
    return value


def _phase_4_driver_kwargs(args, parser: argparse.ArgumentParser) -> dict:
    """The ``Exp4Driver`` keywords of the Phase 4 flags."""
    return dict(
        member_admission=args.member_admission,
        age_cap_missions=args.age_cap_missions,
        age_cap_lookahead=int(args.age_cap_lookahead),
        plan_score_params=_json_object(args.plan_score_params, "--plan-score-params", parser),
        plan_search_params=_json_object(args.plan_search_params, "--plan-search-params", parser),
    )


#: The arms whose RL selector is random-init without ``--selector-weights``
#: (``hermes.processes.mule._build_target_selector``), which ``--require-trained``
#: refuses (FeRRy Phase 5, decision 8 (a)).
SELECTOR_ARMS = ("H2", "H3")


def _add_phase_5_flags(parser: argparse.ArgumentParser) -> None:
    """FeRRy Phase 5 — the learned arms' checkpoints and the trained-weights guard.

    Every flag defaults to the recorded run, and none is a grid axis, so the
    trial seeds do not move and each setting of them gets a CSV of its own.
    """
    g = parser.add_argument_group("FeRRy Phase 5: the learned arms")
    g.add_argument(
        "--pair-checkpoint", action="append", default=None, metavar="TAG=PATH",
        help="The pair_q checkpoint (.npz, its manifest beside it) the FQ arm of TAG flies; "
             f"repeatable. Tags: {', '.join(PAIR_CHECKPOINT_TAGS)} (FQ flies main, FQ-hand "
             "hand, FQ-g0 g0, ...). It must be trained, on at least one episode, and scored "
             "on the held-out runs; one trained from a dirty tree needs "
             "--allow-dirty-checkpoint. It must be its tag's: gX's γ X/100, hand's reward "
             "F·hand and every other tag's the derived reward (at decision 4 (a)'s weights "
             "under gX, dwell and cov), a network that took an update, and the plan score "
             "settings its arm flies under this run's flags.",
    )
    g.add_argument(
        "--policy-checkpoint", action="append", default=None, metavar="E3=PATH",
        help="The chen_dqn checkpoint arm E3 flies. It must be trained, on at least one "
             "episode, and scored on the held-out runs; one trained from a dirty tree needs "
             "--allow-dirty-checkpoint. It must be E3's: trained on E3's bytes reward, by a "
             "network that took an update.",
    )
    g.add_argument(
        "--allow-dirty-checkpoint", action="store_true",
        help="Fly a checkpoint trained from a dirty tree (its manifest records it): for "
             "development and tests, not for a campaign.",
    )
    g.add_argument(
        "--require-trained", action="store_true",
        help="Refuse H2 and H3 without --selector-weights, whose selector would then be "
             "random-init (decision 8 (a)). Off by default, as recorded.",
    )


def _checkpoint_flags(
    values: Optional[Sequence[str]],
    flag: str,
    tags: Sequence[str],
    aliases: Mapping[str, str],
    parser: argparse.ArgumentParser,
) -> Dict[str, str]:
    """``TAG=PATH`` values of a checkpoint flag as {tag: path} ({} when not given).

    ``aliases`` maps another name of a tag to it (``E3`` for ``e3``). A tag no
    learned arm flies, one given twice, or an empty path is a usage error.
    """
    out: Dict[str, str] = {}
    for value in values or ():
        name, sep, path = value.partition("=")
        tag = aliases.get(name, name)
        if not sep or not path.strip():
            parser.error(f"{flag} takes TAG=PATH, got {value!r}")
        if tag not in tags:
            parser.error(f"{flag} {value!r}: {name!r} is no learned arm's tag; the tags are "
                         f"{', '.join(list(tags) + list(aliases))}")
        if tag in out:
            parser.error(f"{flag}: tag {tag!r} is given twice")
        out[tag] = path
    return out


def _campaign_checkpoint(
    flag: str, tag: str, path: str, *, kind: str, allow_dirty: bool,
    parser: argparse.ArgumentParser,
) -> None:
    """Refuse a checkpoint a campaign may not fly (critic B9), as a usage error.

    Judged on the verified manifest (``pair_q.verify_checkpoint``: the arrays
    against the manifest, the purpose and the learner's revision included,
    which the sha binds), never on the manifest alone, which a hand edit could
    relabel: not readable or not the arrays' (``CheckpointError``), not the
    flag's kind, or refused by ``pair_q.campaign_refusals`` (not trained, no
    episode, no held-out score, or dirty without ``--allow-dirty-checkpoint``).
    Then as its tag (the orchestrator's resolution R24): FerrySim's reading of
    its own manifests (``experiments.ferrysim.checkpoints.tag_refusals``: a γ
    tag's γ, the tag's reward, at decision 4 (a)'s weights under a γ tag, dwell
    and cov, and a network that took an update), loaded only when a checkpoint
    is given.
    """
    from hermes.scheduler.selector.pair_q import campaign_refusals, verify_checkpoint

    where = f"{flag} {tag}={path}"
    try:
        manifest = verify_checkpoint(path)
        reasons = campaign_refusals(manifest, allow_dirty=allow_dirty)
    except (ValueError, OSError) as e:
        parser.error(f"{where}: {e}")
    if manifest["kind"] != kind:
        parser.error(f"{where}: a {manifest['kind']!r} checkpoint, and this flag takes "
                     f"{kind!r} ones")
    if reasons:
        parser.error(f"{where}: a campaign does not fly this checkpoint (critic B9): "
                     + "; ".join(reasons))
    from experiments.ferrysim.checkpoints import tag_refusals

    reasons = tag_refusals(tag, manifest)
    if reasons:
        parser.error(f"{where}: a campaign does not fly this checkpoint as tag {tag!r} "
                     f"(resolution R24): " + "; ".join(reasons))


def _check_checkpoint_plans(driver: Exp4Driver) -> None:
    """Refuse a pair checkpoint trained under other plan score settings than its arm
    flies here (the orchestrator's resolution R23), as a ValueError.

    Each pair checkpoint given (whichever arms run, as the campaign checks)
    flies as the FQ arm of its tag (:data:`CHECKPOINT_TAGS`), on the plan score
    settings the driver gives that arm under this run's flags
    (``Exp4Driver.plan_settings``: ``--plan-score-params``, and FQ-dwell's or
    FQ-cov's own change on top); its manifest records the settings it trained
    under (``experiments.ferrysim.checkpoints.trained_plan``, none being the
    cells' default plan). The two are compared with ``PlanScoreParams``'
    defaults filled in, so a default left out equals one written out.
    """
    if not driver.pair_checkpoints:
        return
    from experiments.ferrysim.checkpoints import plan_differences, trained_plan
    from hermes.scheduler.selector.pair_q import verify_checkpoint

    arms = {tag: arm for arm, tag in CHECKPOINT_TAGS.items()}
    for tag, path in driver.pair_checkpoints.items():
        arm = arms[tag]
        where = f"--pair-checkpoint {tag}={path}"
        try:
            differ = plan_differences(trained_plan(verify_checkpoint(path)),
                                      driver.plan_settings(arm)["plan_score_params"])
        except (TypeError, ValueError, OSError) as e:
            raise ValueError(f"{where}: {e}") from e
        if differ:
            raise ValueError(
                f"{where}: trained under other plan score settings than arm {arm} flies here "
                f"(resolution R23): "
                + ", ".join(f"{name} (trained {trained!r}, flown {flown!r})"
                            for name, (trained, flown) in differ.items())
                + "; a checkpoint flies on the plan it trained under")


def _phase_5_driver_kwargs(
    args, parser: argparse.ArgumentParser, arms: List[str],
) -> dict:
    """The ``Exp4Driver`` keywords of the Phase 5 flags, once the runner's refusals pass.

    Under ``--require-trained``, an H2 or H3 among ``arms`` without
    ``--selector-weights`` is refused. Each checkpoint given is refused if a
    campaign may not fly it (:func:`_campaign_checkpoint`), whichever arms
    run, since it names the run's provenance; the checkpoints the named arms
    need are the driver's to require (``Exp4Driver.check_arm``).
    """
    if args.require_trained and args.selector_weights is None:
        untrained = [arm for arm in arms if arm in SELECTOR_ARMS]
        if untrained:
            parser.error(
                f"--require-trained: arms {', '.join(untrained)} fly the RL selector, which "
                f"without --selector-weights is random-init (decision 8 (a)); give trained "
                f"weights or drop the arms"
            )
    pair = _checkpoint_flags(args.pair_checkpoint, "--pair-checkpoint", PAIR_CHECKPOINT_TAGS,
                             {}, parser)
    policy = _checkpoint_flags(args.policy_checkpoint, "--policy-checkpoint",
                               POLICY_CHECKPOINT_TAGS, {"E3": "e3"}, parser)
    if pair or policy:
        # The pair learner's module, loaded only when a checkpoint is given.
        from hermes.scheduler.selector.pair_q import KIND_CHEN_DQN, KIND_PAIR_Q

        for flag, given, kind in (("--pair-checkpoint", pair, KIND_PAIR_Q),
                                  ("--policy-checkpoint", policy, KIND_CHEN_DQN)):
            for tag, path in given.items():
                _campaign_checkpoint(flag, tag, path, kind=kind,
                                     allow_dirty=bool(args.allow_dirty_checkpoint),
                                     parser=parser)
    return dict(pair_checkpoints=pair, policy_checkpoints=policy)


def _train_time_params(args) -> Optional[Dict[str, float]]:
    """Study 5.12's settings from the flags; None without ``--train-time-s``.

    The spread and straggler flags each need the median; only those given are
    passed (the rest take ``experiments/exp4/compute.py``'s defaults).
    """
    extra = {name: getattr(args, attr) for name, attr in (
        ("sigma", "train_time_sigma"), ("straggler_share", "straggler_share"),
        ("straggler_factor", "straggler_factor")) if getattr(args, attr) is not None}
    if args.train_time_s is None:
        if extra:
            raise SystemExit(
                f"--train-time-sigma, --straggler-share and --straggler-factor shape the "
                f"fit times of --train-time-s; given without it: {sorted(extra)}")
        return None
    return {"median_s": float(args.train_time_s), **{k: float(v) for k, v in extra.items()}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.exp4.runner_main")
    parser.add_argument("--csv", required=True, type=Path,
                        help="Per-trial CSV path (created if missing; resumable).")
    parser.add_argument("--n-trials", type=int, default=1,
                        help="Trials per cell (paired across arms).")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Salt of every trial's seed (sha256 of base seed, cell and "
                             "trial index). A pilot takes one of its own, so its seeds are "
                             "not the headline's (FeRRy Phase 4 spec, decision 7).")
    parser.add_argument("--arms", nargs="+", default=None,
                        help=f"Which arms to run (default: the Phase 3 arms "
                             f"{list(DEFAULT_ARMS)}). The FeRRy Phase 4 plan arms "
                             f"{list(PLAN_ARMS)} run only when named, with "
                             f"--mission-clock sim, and so do the FeRRy Phase 5 arms "
                             f"{list(PHASE_5_ARMS)} (a learned arm with its checkpoint) and "
                             f"the Exp 5 addendum's {list(ADDENDUM_ARMS)}.")
    parser.add_argument("--N", nargs="+", type=int, default=[2],
                        help="Device-population sweep.")
    parser.add_argument("--rrf", nargs="+", type=float, default=[60.0],
                        help="rf_range_m sweep.")
    parser.add_argument("--n-missions", nargs="+", type=int, default=[2],
                        help="Missions (FL rounds) per trial.")
    parser.add_argument(
        "--regime", nargs="+", choices=["clean", "jittery"], default=["clean"],
        help="Network-regime axis. 'jittery' degrades H0's long-range "
             "backhaul (dead-zone unreachable clients + intermittent link "
             "failures); H1's short-range mule contact stays reliable, so "
             "H0 participation collapses while H1 holds (the paper's "
             "Observation 3). Sweep both for the clean-vs-jittery contrast.",
    )
    parser.add_argument(
        "--trial-budget-s", type=float, default=120.0,
        help="Hard per-trial wall-clock budget; the process tree is "
             "killed on overrun and the row recorded as an error.",
    )
    parser.add_argument(
        "--startup-timeout-s", type=float, default=30.0,
        help="Timeout for the topology to come up (all ports bound).",
    )
    parser.add_argument(
        "--timeout-s", type=float, default=None,
        help="Soft harness timeout (warning-only label). Defaults to "
             "trial-budget-s so a killed trial is also labelled; on the "
             "mission clock, to the largest re-costed trial budget over the grid.",
    )
    # ---- EX-4.1 real-model flags ---- #
    parser.add_argument(
        "--real-model", action="store_true",
        help="Run the real canonical DNN-IDS in the loop (EX-4.1): real "
             "training on each device + per-round held-out convergence. "
             "Omit for the EX-4.0 stub (federation metrics only).",
    )
    parser.add_argument(
        "--data-source", choices=["canonical", "synthetic"], default="canonical",
        help="Real-model data: 'canonical' = the production CICIOT pipeline "
             "(balanced, 21 features, paper-faithful); 'synthetic' = a "
             "real-shaped separable task (fast, no dataset needed).",
    )
    parser.add_argument(
        "--model-arch", choices=["ciciot", "optimized", "balanced", "high_performance"],
        default=None,
        help="Exp 5 addendum (Study 5.12): the IDS architecture the real model trains "
             "(experiments.exp4.model_task.MODEL_ARCHS; default the canonical CICIoT "
             "model). Its weights, and so the measured payload, have their own size. "
             "Needs --real-model.",
    )
    parser.add_argument(
        "--train-time-s", type=float, default=None, metavar="MEDIAN",
        help="Exp 5 addendum (Study 5.12): each device's local fit takes simulated "
             "time, MEDIAN seconds at the median (experiments/exp4/compute.py); a "
             "Pass-1 contact before a device's fit ends finds no update ready. "
             "Simulated clock only (--mission-clock sim). Default: off (updates are "
             "always ready, every recorded run).",
    )
    parser.add_argument(
        "--train-time-sigma", type=float, default=None,
        help="Study 5.12: the log-normal spread of the fit times around the median "
             "(default 0: every device the median). Needs --train-time-s.",
    )
    parser.add_argument(
        "--straggler-share", type=float, default=None,
        help="Study 5.12: the share of devices that are stragglers, exactly "
             "round(share * N) of them (default 0). Needs --train-time-s.",
    )
    parser.add_argument(
        "--straggler-factor", type=float, default=None,
        help="Study 5.12: the stragglers' fit time as a multiple (default 1; the "
             "plan's '20%% stragglers at 5x' is --straggler-share 0.2 "
             "--straggler-factor 5). Needs --train-time-s.",
    )
    parser.add_argument(
        "--partition", choices=["iid", "dirichlet", "quantity"], default="iid",
        help="Exp 5 addendum (Study 5.13): how the training rows are split over the "
             "devices: 'iid' (default, the recorded split), 'dirichlet' (label skew, "
             "Dir(alpha) per class over the devices; over the attack families with "
             "--family-labels) or 'quantity' (shard sizes Dir(alpha)). Every shard "
             "non-empty. Needs --real-model.",
    )
    parser.add_argument(
        "--dirichlet-alpha", type=float, default=None,
        help="The partition's alpha (> 0; 1 moderate, 0.1 strong skew). Needed by "
             "'dirichlet' and 'quantity', refused with 'iid' (alpha = infinity).",
    )
    parser.add_argument(
        "--family-labels", action="store_true",
        help="Exp 5 addendum (Study 5.13): keep each row's CICIoT2023 attack family "
             "beside the binary label, so the cluster's model_eval adds the detection "
             "metrics (TPR, FPR, precision, F1, recall per family) and 'dirichlet' "
             "skews over the families. Needs --real-model.",
    )
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument(
        "--tau", type=float, default=0.82,
        help="Target accuracy for the T@tau (time-to-accuracy) metric. "
             "0.82 is the MEDIAN final_accuracy over the 640-trial Phase-3 "
             "matrix, so ~half of trials reach it and the metric has "
             "resolution in both directions. The previous default of 0.9 sat "
             "ABOVE the p90 (0.888) and was reached by only 5.9%% of trials, "
             "which made T@tau unusable -- 1 of 40 in the L1 sweep. Use 0.85 "
             "(30%% reach it) as a sensitivity check.")
    parser.add_argument("--train-files", type=int, default=3,
                        help="canonical: CICIOT csv parts to draw train from.")
    parser.add_argument("--test-files", type=int, default=1,
                        help="canonical: CICIOT csv parts to draw test from.")
    parser.add_argument("--train-dataset-size", type=int, default=20000,
                        help="canonical: total balanced train rows (50/50).")
    parser.add_argument("--test-dataset-size", type=int, default=8000,
                        help="canonical: total balanced test rows before attack reduction.")
    parser.add_argument("--attack-eval-ratio", type=float, default=0.5,
                        help="canonical: attack fraction kept in the test set.")
    parser.add_argument("--synth-rows-per-device", type=int, default=512)
    parser.add_argument("--synth-test-rows", type=int, default=512)
    parser.add_argument(
        "--dead-zone", nargs="+", type=float, default=[0.6],
        help="H0 jittery dead-zone fraction — a SWEEP axis (B2 sensitivity "
             "surface). Fraction of clients with no long-range path (physical: "
             "terrain / range-edge). Sweep e.g. 0.0 0.2 0.4 0.6 to find where "
             "the mule's jittery advantage holds vs flips.",
    )
    parser.add_argument(
        "--link-quality", nargs="+", type=float, default=[0.4],
        help="H0 jittery per-round success prob for a reachable client — a "
             "SWEEP axis. Sweep e.g. 0.3 0.5 0.7.",
    )
    parser.add_argument(
        "--realism", action="store_true",
        help="Enable H1 (mule) realism: Exp 3's per-device short-range "
             "contact reliability (U(0.15,1.0) x rf_factor) in every regime, "
             "plus a recoverable long-range backhaul loss under jittery, with "
             "devices spread so S3a forms multiple contacts. Without this, H1 "
             "runs over ideal links (not review-grade for the jittery claim).",
    )
    parser.add_argument(
        "--jittery-backhaul-loss-pct", type=float, default=2.0,
        help="H1 jittery: mule->BS backhaul upload loss (%%). Recoverable.",
    )
    parser.add_argument(
        "--h1-field-radius-m", type=float, default=100.0,
        help="H1 realism: device scatter radius (larger -> more contacts).",
    )
    parser.add_argument(
        "--h1-field-ref-n", type=int, default=None,
        help="Exp 5 addendum (Studies 5.9, 5.11): grow the realism field with N "
             "at this size's density, half-width h1-field-radius-m * sqrt(N / "
             "ref-n), so --N 6 12 24 with --h1-field-ref-n 6 keeps N = 6's density. "
             "Needs --realism. The row does not record it: write each setting to "
             "its own CSV. Default: the fixed field.",
    )
    parser.add_argument(
        "--far-share", type=float, default=None,
        help="Exp 5 addendum (unit U10, Study 5.4): place exactly round(share x N) "
             "devices beyond --rrf of the dock (wide's reach; the trace scorer's far "
             "devices) and the rest within it, in every trial and in T_nom's reference "
             "layouts. Needs --realism. The row does not record it: write each setting "
             "to its own CSV. Default: the recorded uniform draw.",
    )
    parser.add_argument(
        "--selector-weights", type=Path, default=None,
        help="Arm H2: trained DDQN .npz (from experiments.exp3.train_a4). "
             "Omit for a random-init selector (H2 plumbing smoke only).",
    )
    parser.add_argument(
        "--l1-channel", action="store_true",
        help="Arm H3: L1 adaptive channel selection. The mule arms' "
             "backhaul-loss schedule comes from the multi-band RF channel "
             "model (experiments.exp4.channel): H1/H2 hold the best-average "
             "fixed band; H3 runs the U(c,t) controller that re-selects the "
             "band per mission. Under 'jittery' this gives H3 lower backhaul "
             "loss (the paper's L1-adaptivity claim); under 'clean' the "
             "effect is ~null by construction. Use with --realism. With "
             "--mission-clock sim (--backhaul-model mission only) the "
             "selector's RF prior is the chosen band's SNR at the last "
             "upload made, not the trial's mean SNR (critic B4).",
    )
    parser.add_argument(
        "--mission-budget-s", type=float, default=None,
        help="Per-mission time budget (s). When set, the S3 deadline is "
             "ENFORCED: the S3b gate drops contacts that cannot be reached "
             "before their own deadline or would overrun the budget. Omit "
             "(default) to keep the historical behaviour where the deadline "
             "is only a sort key -- that is what the committed results used.",
    )
    parser.add_argument(
        "--keep-event-traces", action="store_true",
        help="Keep each trial's raw per-contact event stream (and the configs "
             "carrying device positions) next to the CSV, instead of deleting "
             "the run dir at teardown. Costs a little disk and changes NO trial "
             "behaviour. Without it a finished sweep cannot be re-scored "
             "against any new scheduling baseline -- the per-contact record is "
             "gone -- so answering 'how would policy X have done?' means "
             "re-running everything.",
    )
    parser.add_argument(
        "--trace-dir", type=Path, default=None,
        help="Where --keep-event-traces writes. Defaults to a '<csv-stem>_traces' "
             "directory beside the CSV.",
    )
    parser.add_argument(
        "--footprint-probe", action="store_true",
        help="Exp 5 addendum, Study 5.11: sample the resident memory of every "
             "process a trial starts (the cluster, the mules, the devices) and "
             "write footprint.json beside its kept trace: the processes, the "
             "concurrent peak and each role's peak, which traces_scorer "
             "--cost-columns reads. Reads only, so no trial changes. Needs "
             "--keep-event-traces and psutil. Off by default.",
    )
    parser.add_argument(
        "--footprint-interval-s", type=float, default=0.5,
        help="The footprint probe's sampling interval in seconds (default 0.5).",
    )
    parser.add_argument(
        "--mission-window-adaptation", action="store_true",
        help="S3c: adapt the deadline window at MISSION level from the mule's "
             "recent success history. The per-device rule only sees 'this "
             "device was missed'; this sees 'the mule is systematically not "
             "completing its circuit' and widens every window together. Off by "
             "default -- that is what the committed results used. Toggle it to "
             "measure its effect against an otherwise identical run.",
    )
    parser.add_argument(
        "--mission-window-history", type=int, default=5,
        help="S3c: how many recent missions inform the scale (default 5).",
    )
    parser.add_argument(
        "--mission-window-target", type=float, default=0.8,
        help="S3c: served/planned at or above which NO widening is applied "
             "(default 0.8).",
    )
    parser.add_argument(
        "--mission-window-gain", type=float, default=2.0,
        help="S3c: widening per unit of shortfall below the target "
             "(default 2.0).",
    )
    parser.add_argument(
        "--mission-window-max-scale", type=float, default=4.0,
        help="S3c: hard cap on the window multiplier (default 4.0), so an "
             "impossible configuration degrades to wide rather than unbounded.",
    )
    parser.add_argument(
        "--l1-channel-bands", type=int, default=3,
        help="Arm H3: number of RF bands the controller chooses among.",
    )
    # FeRRy Phase 1 — L3 merge rules, FedProx, budgeted Pass 2 (mule arms).
    parser.add_argument(
        "--aggregation", default="agg:plain",
        choices=IMPLEMENTED_RULES,
        help="L3 merge rule for the cluster and the mule "
             "(hermes/mission/aggregation_rules.py). agg:plain (default) is the "
             "num_examples mean every recorded run used; the others merge "
             "deltas weighted by age in cluster rounds, except agg:fedex, "
             "FedEx-Async's unweighted θ + (1/N)·ΣΔθ on each return (arm D4's "
             "faithful merge).",
    )
    parser.add_argument(
        "--agg-server-lr", type=float, default=None,
        help="Server rate η: the cluster adds η times the merged update to θ "
             "(default 1.0).",
    )
    parser.add_argument(
        "--agg-a-max", type=int, default=None,
        help="agg:cutoff: fixed age cutoff in cluster rounds; weight is exactly "
             "0 past it.",
    )
    parser.add_argument(
        "--agg-period-s", type=float, default=None,
        help="agg:cutoff: mission period T (s). Each device's cutoff becomes "
             "floor(deadline window / T) rounds (decision D5); combined with "
             "--agg-a-max, the smaller wins.",
    )
    parser.add_argument(
        "--agg-hinge-a", type=float, default=None,
        help="agg:cutoff: FedAsync hinge slope a (default 1.0).",
    )
    parser.add_argument(
        "--agg-hinge-b", type=float, default=None,
        help="agg:cutoff: FedAsync hinge knee b in rounds (default 0).",
    )
    parser.add_argument(
        "--agg-decay", type=float, default=None,
        help="agg:asynchfl: staleness decay λ in exp(-λ·age) (default 0.5).",
    )
    parser.add_argument(
        "--agg-value", choices=("uniform", "loss"), default=None,
        help="Value proxy v_i in w_i = n_i·v_i·s(age_i) (default uniform).",
    )
    parser.add_argument(
        "--agg-buffer-k", type=int, default=None,
        help="agg:fedbuff: updates buffered per server step (default: the "
             "slice size of the mule whose partial first opens the buffer).",
    )
    parser.add_argument(
        "--agg-fedex-n", type=int, default=None,
        help="agg:fedex: N, the total client count in θ + (1/N)·ΣΔθ "
             "(default: the devices registered at the cluster).",
    )
    parser.add_argument(
        "--fedprox-rho", type=float, default=0.0,
        help="FedProx weight ρ on every device: local loss + (ρ/2)·||θ − "
             "θ_received||². 0 (default) keeps the plain Keras fit.",
    )
    parser.add_argument(
        "--pass-2-budget", action="store_true",
        help="Walk Pass 2 against --mission-budget-s instead of delivering to "
             "the whole slice; devices it skips keep their older basis, so "
             "update ages spread. Requires --mission-budget-s.",
    )
    # FeRRy Phase 1 — the deadline law and the priority key (mule arms).
    parser.add_argument(
        "--deadline-law", default="additive",
        choices=("additive", "multiplicative"),
        help="How each outcome moves a device's window Φ. additive (default) "
             "is the recorded -5 s / +10 s law, unbounded above; "
             "multiplicative is Φ <- clamp(β·Φ) with β_on < 1 after an "
             "on-time delivery and β_partial <= β_timeout after a miss, and "
             "makes cluster overrides one-shot.",
    )
    parser.add_argument(
        "--deadline-beta-on", type=float, default=None,
        help="multiplicative: factor after an on-time delivery (default 0.8).",
    )
    parser.add_argument(
        "--deadline-beta-partial", type=float, default=None,
        help="multiplicative: factor after a PARTIAL (default 1.25).",
    )
    parser.add_argument(
        "--deadline-beta-timeout", type=float, default=None,
        help="multiplicative: factor after a TIMEOUT (default 1.5).",
    )
    parser.add_argument(
        "--deadline-phi-min", type=float, default=None,
        help="multiplicative: lower clamp on Φ in seconds (default 5).",
    )
    parser.add_argument(
        "--deadline-phi-max", type=float, default=None,
        help="multiplicative: upper clamp on Φ in seconds (default 300).",
    )
    parser.add_argument(
        "--miss-priority", action="store_true",
        help="S3b admits contacts by their members' consecutive misses before "
             "their deadline, so a missed device is not pushed back by the "
             "wider window its miss earned. Off by default.",
    )
    # FeRRy Phase 2 — several mules, and the options of arms D3/D5.
    parser.add_argument(
        "--n-mules", type=int, default=1,
        help="Mules sharing the cluster (mule arms). The --N devices are split "
             "between them into disjoint spatial slices (D4: by CARP). 1 "
             "(default) is the recorded single-mule topology. A driver setting, "
             "not a grid axis, so K=1 and K=3 runs of the same cell share their "
             "seeds; write them to separate CSVs.",
    )
    parser.add_argument(
        "--min-participation", type=int, default=1,
        help="The cluster's quorum: partials a merge waits for, 1 or "
             "--n-mules (agg:fedbuff ignores it). agg:plain with several mules "
             "requires --min-participation equal to --n-mules.",
    )
    parser.add_argument(
        "--dock-on-empty", action=argparse.BooleanOptionalAction, default=None,
        help="Whether a mission that collected nothing still docks, with an "
             "empty partial that counts toward the quorum. Default: on with "
             "several mules, off with one (the recorded dock).",
    )
    parser.add_argument(
        "--down-wait-s", type=float, default=None,
        help="How long a mule waits at its inter-pass dock for the DOWN before "
             "skipping Pass 2 and flying on its own θ. Default: the trial budget "
             "with several mules; with one, the recorded 10 s wait whose expiry "
             "ends the mule's run.",
    )
    parser.add_argument(
        "--whittle-variant", choices=WHITTLE_VARIANTS, default="expected",
        help="Arm D3: the Whittle index in expectation over the unobserved "
             "connection state (default) or Cui's literal I(x, 1).",
    )
    parser.add_argument(
        "--whittle-weights", choices=WHITTLE_WEIGHTS, default="uniform",
        help="Arm D3: ω = 1 for every device (default) or Oort's statistical "
             "utility (requires --real-model).",
    )
    parser.add_argument(
        "--fedcs-value", choices=FEDCS_VALUES, default="unit",
        help="Arm D5: FedCS's greedy score, one per contact (default, the "
             "paper's letter) or the contact's device count.",
    )
    _add_phase_3_flags(parser)
    _add_phase_4_flags(parser)
    _add_phase_5_flags(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    # H0 (traditional flat FL) is a real-model convergence baseline; drop it
    # from a stub run rather than erroring every H0 trial.
    explicit_arms = args.arms is not None
    arms = list(args.arms) if explicit_arms else list(DEFAULT_ARMS)
    if args.mission_clock == "sim" and "H0" in arms:
        # FeRRy Phase 3 (critic A5): H0 has no simulated round time.
        if explicit_arms:
            parser.error(
                "H0 has no simulated round time (critic A5): run it with "
                "--mission-clock wall, in a CSV of its own"
            )
        log.warning("H0 has no simulated round time; dropping it from this "
                    "--mission-clock sim run")
        arms = [a for a in arms if a != "H0"]
    if not args.real_model and "H0" in arms:
        log.warning("H0 requires --real-model; dropping it from this stub run")
        arms = [a for a in arms if a != "H0"]
    if not arms:
        parser.error("no runnable arms left (H0 needs --real-model)")

    grid = _build_grid(
        arms=arms,
        Ns=args.N,
        rrfs=args.rrf,
        n_missions_values=args.n_missions,
        regimes=args.regime,
        dead_zones=args.dead_zone,
        link_qualities=args.link_quality,
        n_trials=args.n_trials,
        base_seed=args.base_seed,
    )

    if args.pass_2_budget and args.mission_budget_s is None:
        parser.error("--pass-2-budget needs --mission-budget-s")
    driver_kwargs = dict(
        trial_budget_s=float(args.trial_budget_s),
        startup_timeout_s=float(args.startup_timeout_s),
        real_model=bool(args.real_model),
        data_source=args.data_source,
        local_epochs=int(args.local_epochs),
        local_batch_size=int(args.local_batch_size),
        tau=float(args.tau),
        train_files=int(args.train_files),
        test_files=int(args.test_files),
        train_dataset_size=int(args.train_dataset_size),
        test_dataset_size=int(args.test_dataset_size),
        attack_eval_ratio=float(args.attack_eval_ratio),
        synth_rows_per_device=int(args.synth_rows_per_device),
        synth_test_rows=int(args.synth_test_rows),
        jittery_dead_zone_frac=float(args.dead_zone[0]),
        jittery_link_quality=float(args.link_quality[0]),
        realism=bool(args.realism),
        jittery_backhaul_loss_pct=float(args.jittery_backhaul_loss_pct),
        h1_field_radius_m=float(args.h1_field_radius_m),
        selector_weights_path=(str(args.selector_weights) if args.selector_weights else None),
        l1_channel=bool(args.l1_channel),
        l1_channel_bands=int(args.l1_channel_bands),
        mission_budget_s=args.mission_budget_s,
        mission_window_adaptation=bool(args.mission_window_adaptation),
        mission_window_history=int(args.mission_window_history),
        mission_window_target=float(args.mission_window_target),
        mission_window_gain=float(args.mission_window_gain),
        mission_window_max_scale=float(args.mission_window_max_scale),
        trace_root=(
            (args.trace_dir or args.csv.with_name(f"{args.csv.stem}_traces"))
            if args.keep_event_traces else None
        ),
        aggregation=args.aggregation,
        aggregation_params={
            key: value
            for key, value in (
                ("server_lr", args.agg_server_lr),
                ("a_max", args.agg_a_max),
                ("period_s", args.agg_period_s),
                ("hinge_a", args.agg_hinge_a),
                ("hinge_b", args.agg_hinge_b),
                ("decay", args.agg_decay),
                ("value", args.agg_value),
                ("buffer_k", args.agg_buffer_k),
                ("fedex_n", args.agg_fedex_n),
            )
            if value is not None
        },
        fedprox_rho=float(args.fedprox_rho),
        pass_2_budget=bool(args.pass_2_budget),
        deadline_law=args.deadline_law,
        deadline_params={
            key: value
            for key, value in (
                ("beta_on", args.deadline_beta_on),
                ("beta_partial", args.deadline_beta_partial),
                ("beta_timeout", args.deadline_beta_timeout),
                ("phi_min", args.deadline_phi_min),
                ("phi_max", args.deadline_phi_max),
            )
            if value is not None
        },
        miss_priority=bool(args.miss_priority),
        n_mules=int(args.n_mules),
        min_participation=int(args.min_participation),
        dock_on_empty=args.dock_on_empty,
        down_wait_s=args.down_wait_s,
        whittle_variant=args.whittle_variant,
        whittle_weights=args.whittle_weights,
        fedcs_value=args.fedcs_value,
    )
    if args.partition != "iid" or args.dirichlet_alpha is not None or args.family_labels:
        # Exp 5 addendum (Study 5.13); passed only when set.
        driver_kwargs.update(partition=args.partition, dirichlet_alpha=args.dirichlet_alpha,
                             family_labels=bool(args.family_labels))
    if args.model_arch is not None:
        # Exp 5 addendum (Study 5.12); passed only when given.
        driver_kwargs.update(model_arch=args.model_arch)
    train_time = _train_time_params(args)
    if train_time is not None:
        # Exp 5 addendum (Study 5.12); passed only when given.
        driver_kwargs.update(train_time_params=train_time)
    if args.h1_field_ref_n is not None:
        # Exp 5 addendum; passed only when given, as the footprint probe is.
        driver_kwargs.update(h1_field_ref_n=int(args.h1_field_ref_n))
    if args.far_share is not None:
        # Exp 5 addendum, unit U10; passed only when given.
        driver_kwargs.update(far_share=float(args.far_share))
    if args.footprint_probe:
        # Exp 5 addendum, Study 5.11; passed only when asked, so the default
        # driver is built exactly as before.
        driver_kwargs.update(footprint_probe=True,
                             footprint_interval_s=float(args.footprint_interval_s))
    driver_kwargs.update(_phase_3_driver_kwargs(args, parser))
    driver_kwargs.update(_phase_4_driver_kwargs(args, parser))
    driver_kwargs.update(_phase_5_driver_kwargs(args, parser, arms))
    try:
        driver = Exp4Driver(**driver_kwargs)
        # FeRRy Phase 5 (resolution R23): a pair checkpoint flies only on the
        # plan it trained under.
        _check_checkpoint_plans(driver)
        # FeRRy Phase 4: a plan arm the driver cannot run (on the wall clock,
        # without a band, with a budgeted Pass 2, ...) is refused here, before
        # any trial, rather than as an error row per trial. FeRRy Phase 5: so
        # is a learned arm without its checkpoint, an FQ arm without 'replan'
        # and H1+L1 without the adaptive backhaul it is named for.
        for arm in arms:
            if arm in PLAN_ARMS or arm in PHASE_5_ARMS or arm in ADDENDUM_ARMS:
                driver.check_arm(arm)
    except ValueError as e:
        # A combination the driver refuses (e.g. agg:plain with several mules
        # and a smaller quorum): say so as a usage error, before any trial.
        parser.error(str(e))
    if args.real_model:
        log.info(
            "EX-4.1 real-model run: source=%s epochs=%d batch=%d tau=%.2f",
            args.data_source, args.local_epochs, args.local_batch_size, args.tau,
        )
    # The soft cap labels a trial that returned past it. On the mission clock
    # each cell's hard kill is re-costed for the session TTL (critic B14), so
    # the default cap is the largest of them over the grid; on the wall clock
    # it is --trial-budget-s, as recorded.
    soft_cap = args.timeout_s
    if soft_cap is None:
        soft_cap = max(
            driver.trial_wall_budget_s(n_devices=int(n), n_missions=int(m))
            for n in args.N for m in args.n_missions
        )
    # A mission-clock trial's own budget can be below that cap, so its status
    # marker records the cap as well: a trace scored without this CSV then
    # applies the cap the runner applied. Wall markers leave it out.
    driver.soft_cap_s = float(soft_cap)
    runner = TrialRunner(
        grid=grid,
        log_path=args.csv,
        metric_columns=list(Exp4MetricSummary.csv_columns()) + list(PROVENANCE_COLUMNS),
        timeout_s=soft_cap,
    )
    log.info(
        "exp4 grid: arms=%s N=%s rrf=%s n_missions=%s regime=%s trials=%d (%d cells)",
        arms, args.N, args.rrf, args.n_missions, args.regime, args.n_trials,
        grid.total(),
    )
    n = runner.run(driver.run_trial)
    print(f"wrote {n} new trial rows to {args.csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
