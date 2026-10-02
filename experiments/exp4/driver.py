"""Experiment-4 trial driver (chunks EX-4.0 + EX-4.1).

Plugs into the shared :class:`~experiments.runner.TrialRunner`'s
``run_trial(cell)`` slot. For each cell it:

1. Builds a finite topology (1 cluster + 1 mule + N devices, mule capped
   at ``n_missions``) from the cell's sweep coordinates + paired seed.
2. Brings it up on the **real** :class:`MultiProcessOrchestrator` (real
   subprocesses, real TCP, real two-pass Pass-1 → dock → Pass-2 with real
   cross-mule FedAvg).
3. Waits for the mule to exit *naturally* within a hard per-trial budget
   (killing the tree on overrun so a hung socket never blocks the sweep).
4. Reads the per-process JSONL and rolls it up into the metrics.

**EX-4.0 (``real_model=False``, default):** a noise-stub model — the
federation-side scheduling metrics from the real L2+L3 stack.

**EX-4.1 (``real_model=True``):** the *real* canonical DNN-IDS. The driver
prepares the ``CiciotTask`` once per trial ("driver-prepares-once"),
serializes each device's shard + the shared held-out test set + the real
seed weights, and points the subprocesses at them. The cluster seeds the
global model from those weights and scores each aggregated θ on the
held-out set, so the trial emits accuracy/AUC-over-rounds + T@τ.

Only arm **H1** exists so far (mule + gated scheduler + two-pass HFL,
deterministic ranking, no RL selector, no L1). H0/H2/H3 arrive in later
chunks; an unknown arm is rejected loudly.

**FeRRy Phase 3 (``mission_clock="sim"``): ferry cells.** The mule arms fly on
the simulated mission clock with the contact link and seconds-axis channel
(hermes/mule/ferry.py). The driver builds the cell's ferry settings
(:meth:`Exp4Driver.ferry_settings`), computes the cell's nominal mission
period T_nom when a setting needs it (:meth:`Exp4Driver.nominal_period_s`:
``deadline_time_scale = T_nom / 10 s``, Φ₀ in missions, D5's ``period_s`` and
the backhaul period ``P_bh = n_missions * T_nom``), gives every trial one RF
link token (Amendment 10), pins the model's input width (design R8), prices
the D4 CARP split with the predicted per-client dwell, and re-costs its wall
budget for the mission clock's session TTL (critic B14). Under ``l1_channel``
a ferry cell keeps the recorded per-mission loss schedule, and its mule
adopts the chosen carrier's SNR mission by mission as the selector's RF prior
instead of the trial's mean (critic B4, :func:`chosen_snr_schedule`). H0 is
refused on the simulated clock (critic A5: its simulated round time is
outside Phase 3). With the defaults every trial is the recorded wall-clock one.

**FeRRy Phase 4: the plan arms** (:data:`PLAN_ARMS`; the Phase 4 spec, other
choices 11 and 12). F, FX, FB+wide, FB+medium, FB+narrow, F-cov, F-cap and
F-prio fly the plan clock (``MuleConfig.plan_mode="ferry"``) on the simulated
clock only. Each arm's plan fields are explicit topology parameters
(:meth:`Exp4Driver.plan_settings`); FB+<class>'s band travels in its ferry
settings (:meth:`Exp4Driver.ferry_settings`, critic A8); a plan arm always
has T_nom (T in its score) and the ``trim`` fallback (its Pass-1 re-plan
trims the committed plan: members under ``subset``, whole stops only under
``whole``), and its row records its own ``miss_priority``. The provenance
shows the plan fields in ``ferry_params`` in plan mode only, and
``contact_band`` reads ``search`` for an arm that searches the classes
(:func:`plan_ferry_params`, :func:`contact_band_column`, which the trace
scorer shares). An H or D arm may run member subsets (``member_admission``,
the user's decision 4 (b)); D4 always runs whole. The runner's default arm
list stays the nine Phase 3 arms (:data:`DEFAULT_ARMS`), and the trial CSV
header is unchanged.

**FeRRy Phase 5: the learned arms and H1+L1** (:data:`PHASE_5_ARMS`; the Phase
5 spec, other choices 5 and 6, decision 8 (a)). The FQ arms (:data:`PAIR_ARMS`)
are F with the learned (band, next stop) score in the flight slot
(``flight_slot="pair_q"``), each on F's settings everywhere F has them
(:func:`is_plan_arm`, critic A5), and E3 is Chen et al.'s DQN as a legacy-mode
whole scheduler (``contact_policy="chen_dqn"``). A learned arm flies the
verified checkpoint of its tag (:data:`CHECKPOINT_TAGS`; ``pair_checkpoints``
and ``policy_checkpoints``) and is refused without one (no random-init arm);
an FQ arm needs ``in_flight_response="replan"`` (resolution R3). The driver
verifies each file as the mule will but checks no training state, which is
the runner's to refuse (critic B9), so FerrySim's bootstrap checkpoints fly.
``H1+L1`` is H1 with H3's adaptive backhaul and no learned selector. The
provenance names each checkpoint by tag and sha only, ``pair_tag`` and
``pair_sha256`` in ``ferry_params`` and ``policy_tag`` and ``policy_sha256``
in ``policy_params``, for those arms only (:func:`plan_ferry_params`,
:func:`learned_policy_params`, which the trace scorer shares); every other
row reads as before, and the default arm list is still :data:`DEFAULT_ARMS`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import statistics
import subprocess
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from experiments.runner import Cell

from hermes.processes import MultiProcessOrchestrator

from .events_consumer import consume_run_dir
from .metrics import Exp4MetricSummary, summarise_observation
from .prep import prepare_trial
from .topology_builder import (
    SESSION_TTL_S,
    SYNTH_BATCH_SIZE,
    angular_slices,
    build_exp4_topology,
    device_positions,
    device_spread_m,
    grown_field_radius_m,
)

log = logging.getLogger("experiments.exp4.driver")


#: D1/D2 are the SOTA baseline arms — WHOLE SCHEDULERS, not tie-breakers. Each
#: shares H1's transport, realism and seeds but replaces S3 + S3b + S3.5 with its
#: own rule, so it owns the ADMISSION decision our gate would otherwise make.
#:
#: They supersede the earlier ordering-only B1/B2, which were vacuous: S3b fixes
#: who is served before any ranking policy runs, so those arms could only permute
#: a list our gate had already decided and produced byte-identical results.
#:
#: FeRRy Phase 2 adds D3 (Cui's Whittle index), D4 (FedEx-Async's visit-all
#: tour; with several mules its devices are split between them by CARP) and D5
#: (FedCS's greedy selection, degraded to last-known state). D4's two runs in
#: the build plan differ only in ``aggregation`` (agg:fedex faithful, agg:cutoff
#: route-only), so they share the label.
#:
#: These nine are the runner's default arm list (FeRRy Phase 4 keeps it, so a
#: run that names no arms runs what it always ran).
DEFAULT_ARMS = ("H0", "H1", "H2", "H3", "D1", "D2", "D3", "D4", "D5")

#: FeRRy Phase 4 — the plan arms (build plan L977-984; the Phase 4 spec, other
#: choices 11). Each flies the plan clock, which commits every mission at the
#: dock to a band class and a route as one decision, on the simulated clock
#: only. ``F`` searches every band class of the link; ``FX`` is F with the
#: cross-heuristic flight slot (decision 5); ``FB+<class>`` pins that class
#: (Path B+). The ablations drop one part of F: ``F-cov`` the coverage term
#: (decision 3, "cap-only service"), ``F-cap`` the age cap and ``F-prio`` the
#: miss-streak factor of the coverage weight. The labels are ASCII, with no
#: "__" and none of the characters Windows forbids in a path, so a kept
#: trace's directory (:func:`trace_dir_name`) holds them whole and the trace
#: scorer parses them back (``parse_trial_dir``).
PLAN_ARMS = ("F", "FX", "FB+wide", "FB+medium", "FB+narrow", "F-cov", "F-cap", "F-prio")

#: FeRRy Phase 5 — the FQ arms (the Phase 5 spec, other choices 5; design D-L
#: (a)): F with the learned (band, next stop) score in its flight slot
#: (``flight_slot="pair_q"``), the plan's "F" of L1039, while Phase 4's F stays
#: the committed slot (the paper may call FQ "F"). ``FQ`` flies the main score,
#: ``FQ-hand`` the one trained on today's reward (F·hand, decision 4),
#: ``FQ-dwell`` and ``FQ-cov`` the ones trained under F's plan-term ablations
#: (Study 5.7, only if Study 5.5 keeps the score, critic C2), and ``FQ-g0`` ...
#: ``FQ-g99`` the γ sweep's (Study 5.5). Labels as the plan arms'.
PAIR_ARMS = ("FQ", "FQ-hand", "FQ-dwell", "FQ-cov", "FQ-g0", "FQ-g25", "FQ-g50", "FQ-g75",
             "FQ-g90", "FQ-g99")

#: The learned arms: the FQ arms and ``E3``, the numpy port of Chen et al.'s DQN
#: (decision 7 (a); ``contact_policy="chen_dqn"``). Each flies one verified
#: checkpoint, named by its tag (:data:`CHECKPOINT_TAGS`): there is no
#: random-init arm (:meth:`Exp4Driver.check_arm`).
LEARNED_ARMS = PAIR_ARMS + ("E3",)

#: The arms Phase 5 adds: the learned ones and ``H1+L1``, H1's scheduler with
#: H3's adaptive backhaul controller and no learned selector, so that the
#: adaptive backhaul keeps a reference once H2 and H3 leave Exp 5 (decision 8
#: (a); critic A6). They run only when named.
PHASE_5_ARMS = LEARNED_ARMS + ("H1+L1",)

#: Every arm the driver runs.
ARMS = DEFAULT_ARMS + PLAN_ARMS + PHASE_5_ARMS

#: Each learned arm's checkpoint tag (the Phase 5 spec, other choices 5): the
#: runner's ``--pair-checkpoint TAG=PATH`` and ``--policy-checkpoint E3=PATH``
#: name the file each tag flies (``Exp4Driver.pair_checkpoints`` and
#: ``policy_checkpoints``), and the tag travels with the checkpoint's sha in the
#: mule config and the row's provenance. A tag also names a directory of the
#: checkpoint layout (``results/exp5/checkpoints/<study>/<tag>/``).
CHECKPOINT_TAGS = {
    "FQ": "main", "FQ-hand": "hand", "FQ-dwell": "dwell", "FQ-cov": "cov",
    "FQ-g0": "g0", "FQ-g25": "g25", "FQ-g50": "g50", "FQ-g75": "g75", "FQ-g90": "g90",
    "FQ-g99": "g99", "E3": "e3",
}
#: The tags of the pair checkpoints, in :data:`PAIR_ARMS` order, and of E3's.
PAIR_CHECKPOINT_TAGS = tuple(CHECKPOINT_TAGS[arm] for arm in PAIR_ARMS)
POLICY_CHECKPOINT_TAGS = (CHECKPOINT_TAGS["E3"],)

#: ``MuleConfig.contact_policy`` of each whole-scheduler arm.
_ARM_POLICY = {
    "D1": "max_aoi", "D2": "oort", "D3": "whittle", "D4": "fedex", "D5": "fedcs",
    "E3": "chen_dqn",
}

#: Each plan arm's change to F's plan fields (:meth:`Exp4Driver.plan_settings`).
#: F-prio's change is its ``miss_priority`` (:meth:`Exp4Driver.effective_miss_priority`),
#: which is not a plan field. FeRRy Phase 5: every FQ arm flies the pair slot;
#: FQ-dwell's and FQ-cov's score changes are in :data:`_ARM_SCORE`.
_PLAN_ARM = {
    "F": {},
    "FX": {"flight_slot": "cross_heuristic"},
    "FB+wide": {"band_class_policy": "fixed:wide"},
    "FB+medium": {"band_class_policy": "fixed:medium"},
    "FB+narrow": {"band_class_policy": "fixed:narrow"},
    "F-cov": {},
    "F-cap": {"age_cap_missions": None, "age_cap_lookahead": 0},
    "F-prio": {},
    **{arm: {"flight_slot": "pair_q"} for arm in PAIR_ARMS},
}

#: F-cov's plan score settings, over the driver's own: the coverage term off,
#: c2 = c3 = 0 (``PlanScoreParams``: c3 = c2 unless set; unit U0's F-cov).
#: With coverage worth nothing an empty plan scores best, so the arm serves
#: capped devices only: "cap-only service" (decision 3).
F_COV_SCORE = {"c_cov_per_device": 0.0, "c_link": 0.0}

#: FQ-dwell's plan score settings, over the driver's own: the dwell taken out
#: of Δ in the score (``PlanScoreParams.dwell_in_delta``; Study 5.7's dwell
#: ablation, design D-L (a)).
FQ_DWELL_SCORE = {"dwell_in_delta": False}

#: Each arm's change to the plan score's settings (:meth:`Exp4Driver.plan_settings`):
#: F-cov's and FQ-cov's coverage term off, FQ-dwell's dwell out of Δ.
_ARM_SCORE = {"F-cov": F_COV_SCORE, "FQ-cov": F_COV_SCORE, "FQ-dwell": FQ_DWELL_SCORE}

#: The arms that may admit part of a stop (``member_admission="subset"``,
#: decision 4 (b)): the plan arms, H1-H3 (S3b's gate) and D1-D3 and D5 (their
#: walks). D4's visit-all tour has no gate and always runs whole; H0 has no
#: mule. FeRRy Phase 5: the FQ arms as F, H1+L1 as H1; E3 visits a stop for all
#: its members, as D4 does, so it runs whole.
_SUBSET_ARMS = frozenset(("H1", "H2", "H3", "D1", "D2", "D3", "D5") + PLAN_ARMS + PAIR_ARMS
                         + ("H1+L1",))

#: The arms that fly H3's adaptive backhaul controller: on the simulated clock
#: ``backhaul_policy="adaptive"``, and with ``l1_channel`` the adaptive
#: per-mission loss schedule (``backhaul_plan(adaptive=True)``). FeRRy Phase 5
#: adds H1+L1 (decision 8 (a); critic A6).
_ADAPTIVE_BACKHAUL_ARMS = ("H3", "H1+L1")


def is_plan_arm(arm: str) -> bool:
    """Whether ``arm`` flies the plan clock: the Phase 4 plan arms and the FQ arms.

    The one predicate every plan-mode gate of the driver and the runner reads
    (the Phase 5 spec, other choices 5; critic A5): an FQ arm is F with the
    pair slot, so it gets F's settings wherever F has them (the trim fallback,
    member subsets, the miss priority, T_nom and the pre-trial check), while
    :data:`PLAN_ARMS` keeps its pinned value.
    """
    return arm in PLAN_ARMS or arm in PAIR_ARMS


def plan_ferry_params(mule: Mapping[str, Any]) -> Dict[str, Any]:
    """The FeRRy Phase 4 keys of a trial's ``ferry_params`` column.

    ``mule`` is a mule's config as a mapping: a ``MuleConfig``'s fields, or the
    per-role JSON a kept trace holds; a missing key takes ``MuleConfig``'s
    default. In plan mode, every plan field (``PLAN_MULE_FIELDS``) as the
    config holds it; outside it, ``member_admission`` only when it is not the
    recorded ``whole`` (an H or D arm run with subsets, whose rows must not
    read like the recorded ones, unit_U3b.md section 5.4); {} at the defaults,
    so a Phase 3 row's ``ferry_params`` is unchanged (Freeze Rule 1). The
    trace scorer derives the same keys from a kept trace with this function,
    so the two columns agree.

    FeRRy Phase 5 (other choices 6; critic B10): a mule flying the pair slot
    (``flight_slot="pair_q"``) adds its checkpoint's provenance, ``pair_tag``
    and ``pair_sha256``, from the config's tag and sha; never its path. No
    other slot adds anything, so every Phase 4 row keeps its string.
    """
    from hermes.processes.config import (
        FLIGHT_SLOT_PAIR_Q,
        MEMBER_ADMISSION_WHOLE,
        PLAN_MODE_FERRY,
        PLAN_MULE_FIELDS,
        MuleConfig,
    )

    defaults = MuleConfig(mule_id="defaults")

    def value(name: str) -> Any:
        return mule[name] if name in mule else getattr(defaults, name)

    if value("plan_mode") == PLAN_MODE_FERRY:
        out = {name: value(name) for name in PLAN_MULE_FIELDS}
        if value("flight_slot") == FLIGHT_SLOT_PAIR_Q:
            out.update(pair_tag=value("pair_checkpoint_tag"),
                       pair_sha256=value("pair_checkpoint_sha256"))
        return out
    admission = value("member_admission")
    return {} if admission == MEMBER_ADMISSION_WHOLE else {"member_admission": admission}


def learned_policy_params(mule: Mapping[str, Any]) -> Dict[str, Any]:
    """The FeRRy Phase 5 keys of a trial's ``policy_params`` column: arm E3's checkpoint.

    ``mule`` is a mule's config as a mapping, as for :func:`plan_ferry_params`.
    For ``contact_policy="chen_dqn"`` (arm E3) its checkpoint's provenance,
    ``policy_tag`` and ``policy_sha256``, from the config's tag and sha, never
    its path (the Phase 5 spec, other choices 6); {} for every other policy,
    so no Phase 3 or Phase 4 row's ``policy_params`` changes. The trace scorer
    derives the same keys from a kept trace's per-role JSON with this
    function, so the two columns agree.
    """
    from hermes.processes.config import CONTACT_POLICY_CHEN_DQN

    if mule.get("contact_policy") != CONTACT_POLICY_CHEN_DQN:
        return {}
    return {"policy_tag": mule.get("policy_checkpoint_tag"),
            "policy_sha256": mule.get("policy_checkpoint_sha256")}


def contact_band_column(mule: Mapping[str, Any]) -> str:
    """The ``contact_band`` provenance column for a mule's config (as a mapping).

    ``search`` for a plan arm that searches the band classes, whose config's
    ``contact_band`` is only the reference class (critic B14); otherwise the
    band the mule flies, the pinned class of FB+<class> included, and "" for
    the channel-free control and the wall clock. Shared with the trace scorer,
    like :func:`plan_ferry_params`.
    """
    from hermes.processes.config import BAND_POLICY_SEARCH, PLAN_MODE_FERRY

    if (mule.get("plan_mode") == PLAN_MODE_FERRY
            and mule.get("band_class_policy", BAND_POLICY_SEARCH) == BAND_POLICY_SEARCH):
        return BAND_POLICY_SEARCH
    return mule.get("contact_band") or ""

#: Scheduler-configuration columns the driver stamps on every row, on top of
#: the metric schema. They exist so a results CSV is self-describing: rows
#: recorded with the S3b deadline gate or S3c window adaptation active are NOT
#: comparable with historical rows, and without these nothing in the file says
#: which is which. Owned by the driver, not the metric summary — they describe
#: how the trial was configured, not what it measured.
PROVENANCE_COLUMNS = (
    "mission_budget_s", "mission_window_adaptation",
    # FeRRy Phase 1: the L3 merge rule and its parameters (JSON; blank under
    # agg:plain), the devices' FedProx weight, and the budgeted Pass 2.
    "aggregation", "aggregation_params", "fedprox_rho", "pass_2_budget",
    # ...and the deadline law with its parameters (JSON; blank when additive)
    # and the miss-streak priority key.
    "deadline_law", "deadline_params", "miss_priority",
    # FeRRy Phase 2: the mule count, the cluster's quorum (blank for one mule
    # with a quorum of 1), the dock settings (JSON of dock_on_empty and
    # down_wait_s; blank when both are off) and the D3/D5 policy options
    # (JSON; blank for every other arm). New columns only: the ones above keep
    # their values, so a single-mule row differs by n_mules=1 and three blanks.
    "n_mules", "min_participation", "dock_params", "policy_params",
    # FeRRy Phase 3: the clock and contact-link switches, the deadline time
    # unit, T_nom, the session TTL, the ferry parameters (JSON; blank on the
    # wall clock), and three settings no CSV recorded before: the L1 channel,
    # realism and the model's input width. New columns only, so a recorded
    # row reads the same in the columns above. The exact formats, and how
    # each is derived from a kept trace that predates them, are in
    # ``Exp4Driver._clock_provenance`` (critic B13, D3: every CSV header
    # changes, so a Phase 3 run needs a fresh CSV path).
    "mission_clock", "contact_band", "in_flight_response", "backhaul_model",
    "contact_reliability_source", "deadline_time_scale", "initial_window_s",
    "t_nom_s", "session_ttl_s", "ferry_params", "l1_channel", "realism", "input_dim",
)

#: The width of the canonical CICIoT model's input: the 46 raw features minus
#: the 25 the canonical loader drops (``model_task.load_ciciot_task_canonical``;
#: Phase 3 design section 0, finding 1: θ is then 18,756 B). A ferry cell on
#: the canonical data that gets another width (the loader's silent synthetic
#: fallback builds a 46-input model) is refused (design R8).
CANONICAL_INPUT_DIM = 21

#: FeRRy Phase 3 (critic B14): the mule's wall-clock startup waits, device
#: registration (60 s) and bootstrap DOWN (30 s), and the recorded DOWN wait at
#: a single mule's dock (10 s, ``ClientCluster.down_timeout_s``). Used by
#: :meth:`Exp4Driver.ferry_wall_bound_s`.
_STARTUP_WALL_S = 90.0
_DOCK_WAIT_S = 10.0

#: ``MuleConfig`` physics fields a ferry cell may override (``ferry_physics``).
FERRY_PHYSICS_FIELDS = (
    "snr_floor_db", "altitude_m", "n_pl", "shadow_sigma_db", "margin_quantile",
    "contact_regime", "interference_period_s", "noise_bin_s", "shadow_corr_s",
    "shadow_keying", "cruise_speed_m_s", "turnaround_s", "listen_s",
    "energy_capacity_j", "p_move_w", "p_hover_w",
    # Exp 5 addendum (Study 5.15): the regime's interference amplitude and noise.
    "interference_amp_db", "interference_sigma_db",
)

#: Marker written next to every kept trace: the ``status`` and ``error`` the
#: driver hands the runner, the mission target, and the trial's run time and
#: budget (on the mission clock, also the runner's soft cap when the driver is
#: told it) — the runner relabels a trial that returned past its soft cap
#: ``timeout`` only after the marker is written, so the marker keeps what that
#: decision is made from. A trace outlives its CSV row's context (it is
#: re-scored from the directory alone), and without this the trace scorer
#: cannot tell a timed-out or ``no_eval`` trial from a good one.
TRIAL_STATUS_FILE = "trial_status.json"

#: Characters Windows forbids in a path component. Cell ids are built from the
#: grid axes and contain ``|`` and ``=`` (e.g.
#: ``N=6|dead_zone=0.0|regime=jittery``), so a trace directory named after one
#: is unopenable on Windows unless it is sanitised.
_ILLEGAL_PATH_CHARS = '<>:"/\\|?*'
#: Keep a comfortable margin under Windows' 260-char MAX_PATH once the results
#: root and the per-file names are appended.
_MAX_TRACE_NAME = 120


def trace_dir_name(cell) -> str:
    """Filesystem-safe, collision-free directory name for one trial's traces.

    Encodes cell / arm / trial / seed so a trace can be matched back to its CSV
    row, which is the entire point of keeping it. Over-long names are truncated
    and given a hash suffix rather than silently colliding — two trials sharing
    a directory would interleave their events and quietly corrupt both.
    """
    raw = f"{cell.cell_id}__{cell.arm}__t{cell.trial_index}__s{cell.seed}"
    safe = "".join("-" if c in _ILLEGAL_PATH_CHARS else c for c in raw)
    safe = safe.strip(" .")  # Windows rejects trailing dots/spaces
    if len(safe) > _MAX_TRACE_NAME:
        import hashlib
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[: _MAX_TRACE_NAME - 9]}-{digest}"
    return safe


def chosen_snr_schedule(model, plan) -> list:
    """The SNR (dB) of the carrier ``plan`` chose at each mission, in mission order.

    ``model`` is the trial's legacy ``ChannelModel`` and ``plan`` its
    ``backhaul_plan``: entry m is ``model.snr(m, plan.chosen_bands[m])``, the
    value ``plan.loss_schedule[m]`` is ``loss_from_snr`` of. Under
    ``--l1-channel`` on the mission clock (recorded ``mission`` backhaul
    model) this is the mule's causal RF prior, adopted entry by entry as the
    missions upload (``MuleConfig.rf_prior_schedule_db``, critic B4), in place
    of ``plan.mean_chosen_snr_db``, the mean over every mission, later ones
    included.
    """
    return [float(model.snr(m, band)) for m, band in enumerate(plan.chosen_bands)]


class Exp4TrialTimeout(RuntimeError):
    """Raised when a trial blows its hard wall-clock budget.

    The orchestrator is killed before this propagates; the harness
    records the row with ``status=error`` and the sweep continues.
    """


class Exp4MuleFailure(RuntimeError):
    """Raised when a mule process exits non-zero on its own.

    A mule whose mission loop ends on a failure it cannot recover from exits
    non-zero (``hermes.processes.mule.EXIT_*``), and so does one that crashes.
    Its trial ran fewer missions than asked for, so the harness records it as
    ``status=error`` rather than as a short but valid trial.
    """


def d4_slice_assignment(
    devices,
    n_mules: int,
    seed: int,
    *,
    t_trans_s: Optional[float] = None,
    cruise_speed_m_s: Optional[float] = None,
) -> Dict[int, int]:
    """Arm D4's device-to-mule split: CARP over the trial's seeded positions.

    FedEx-Async's Gibbs-sampled assignment (``carp_assign``, objective
    Σ_k R_k·Δ_k²), with every tour closed at the dock every mule starts from
    (``DOCK_POSE``, the origin), priced with the shared ``FeasibilityModel``
    defaults (cruise speed as every mule's speed, the session time as the
    per-client time) and seeded from the trial seed, so paired trials split
    identically. Computed once per trial: devices are wired to one mule's RF
    link at launch, so the split cannot change while the trial runs. Returns
    device index -> mule index.

    FeRRy Phase 3: a ferry cell passes ``t_trans_s``, the predicted Pass-1
    airtime of one client at half the contact range R_planar(b)/2
    (:meth:`Exp4Driver.carp_t_trans_s`), and its flight model's
    ``cruise_speed_m_s``. None keeps the recorded values.
    """
    from hermes.processes.mule import DOCK_POSE
    from hermes.scheduler.policies import carp_assign
    from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
    from hermes.types import DeviceID

    from .model_task import _u32

    model = FeasibilityModel()
    speed = model.cruise_speed_m_s if cruise_speed_m_s is None else float(cruise_speed_m_s)
    t_trans = model.session_time_s if t_trans_s is None else float(t_trans_s)
    positions = {
        DeviceID(d.device_id): tuple(float(c) for c in d.position) for d in devices
    }
    by_id = carp_assign(
        positions,
        n_transporters=int(n_mules),
        depot=DOCK_POSE,
        speeds=[speed] * int(n_mules),
        t_trans=t_trans,
        seed=_u32(seed, "d4_carp"),
    )
    return {i: int(by_id[DeviceID(d.device_id)]) for i, d in enumerate(devices)}


@dataclass
class _TrialClock:
    """One trial's clock settings as the driver resolved them (FeRRy Phase 3).

    ``settings`` are the ``MuleConfig`` ferry fields handed to the builder
    (empty on the wall clock); the rest are the per-cell values the driver
    derived: T_nom and whether it computed it, the deadline time unit, Φ₀,
    the merge spec (D5's period may be T_nom), the wall budget and DOWN wait,
    the model's input width and the RF link token.
    """

    sim: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)
    t_nom_s: Optional[float] = None
    t_nom_computed: bool = False
    deadline_time_scale: float = 1.0
    initial_window_s: Optional[float] = None
    aggregation_spec: Any = None
    budget_s: Optional[float] = None
    down_wait_s: Optional[float] = None
    input_dim: Optional[int] = None
    rf_link_token: Optional[str] = None


@dataclass
class Exp4Driver:
    """Owns per-trial dispatch over the real multi-process orchestrator.

    ``real_model=False`` is the EX-4.0 instrumentation path (noise stub,
    fast). ``real_model=True`` is EX-4.1 — the real canonical DNN-IDS with
    per-round convergence; each trial spawns real TensorFlow fits in every
    device subprocess, so budget accordingly.
    """

    default_n_devices: int = 2
    default_rf_range_m: float = 60.0
    default_n_missions: int = 2
    # Hard per-trial wall-clock budget (seconds). A trial that runs long
    # is killed and recorded as an error rather than hanging the sweep.
    trial_budget_s: float = 120.0
    startup_timeout_s: float = 30.0
    shutdown_timeout_s: float = 10.0

    # ---- EX-4.1 real-model knobs (ignored when real_model=False) ---- #
    real_model: bool = False
    data_source: str = "canonical"  # "canonical" | "synthetic"
    local_epochs: int = 1
    local_batch_size: int = 64
    # Target accuracy for T@tau. 0.82 = median final_accuracy over the
    # Phase-3 matrix; 0.9 sat above p90 and made the metric unusable.
    tau: float = 0.82
    theta_seed: int = 12345
    # canonical loader knobs
    train_files: int = 3
    test_files: int = 1
    train_dataset_size: int = 20000
    test_dataset_size: int = 8000
    attack_eval_ratio: float = 0.5
    # synthetic loader knobs
    synth_rows_per_device: int = 512
    synth_test_rows: int = 512
    # EX-4.2 arm H2 — trained DDQN target selector (.npz from exp3.train_a4).
    # None -> H2 uses a random-init selector (plumbing smoke, not paper-grade).
    selector_weights_path: Optional[str] = None
    # H0 (traditional flat FL) — fraction of clients sampled per round.
    # 1.0 = every client every round (the reliable-infrastructure baseline,
    # matching Exp 1's fully-participating clients).
    h0_client_fraction: float = 1.0
    # ---- EX-4.2 jittery-regime knobs ---- #
    # H0's long-range backhaul degrades under jittery: a dead-zone fraction of
    # clients are persistently unreachable from the central server, and the
    # reachable ones contribute each round with prob reliability_i x
    # link_quality. H1's mule reaches devices over short-range contact
    # (jitter-immune), so it gets NO dead-zone.
    #
    # PROVENANCE + CAVEAT: this dead-zone / link_quality mechanism is the
    # flat-FL (A1) model from experiments/exp3/arm_a1.py + the exp3 driver —
    # NOT from the Exp 3 simulator (sim_env), which has no flat-FL arm. It was
    # tuned there for A1's 20-round horizon; Exp 4 runs few rounds, so the
    # dead-zone rate must be justified physically (fraction of devices with no
    # long-range path — terrain / range-edge) and reported as a SENSITIVITY
    # axis, not a single tuned point. The 0.6 default is one point on the sweep.
    clean_dead_zone_frac: float = 0.0
    jittery_dead_zone_frac: float = 0.6
    clean_link_quality: float = 1.0
    jittery_link_quality: float = 0.4
    # H1 (mule) realism, opt-in. Applies the per-device short-range contact
    # reliability (reliability_i x rf_factor, the SAME reliability draw H0
    # uses) in every regime, plus — under jittery — a long-range backhaul
    # upload loss that marks the round as not-closed (the recoverable, one-hop
    # cost of routing through the mule). SCOPE: this models the NETWORK +
    # computation layers only; it does NOT model mule flight-budget / deadline
    # pressure (fewer contacts under a tight budget) — that is the scheduling
    # experiment's (Exp 3) domain and is deferred here. Devices are spread so
    # S3a forms multiple contacts. Off -> the ideal EX-4.1 links.
    realism: bool = False
    h1_field_radius_m: float = 100.0
    h1_world_radius_m: float = 100.0
    clean_backhaul_loss_pct: float = 0.0
    jittery_backhaul_loss_pct: float = 2.0
    # EX-4.3 arm H3 — L1 adaptive channel selection. When on, the mule arms'
    # backhaul loss comes from the RF channel model (experiments.exp4.channel):
    # H1/H2 hold the best-average fixed band, H3 runs the U(c,t) controller.
    # Use with --realism (contact reliability) for the H1->H2->H3 ladder.
    l1_channel: bool = False
    l1_channel_bands: int = 3
    # S3b — per-mission time budget (seconds). None (default) reproduces every
    # previously-recorded result: the S3 deadline stays a sort key. Setting it
    # turns on the feasibility gate so the deadline actually binds.
    mission_budget_s: Optional[float] = None
    # S3c — mission-level window adaptation. False (default) reproduces every
    # previously-recorded result exactly: the window scale stays 1.0. Toggled
    # on, systemic mission shortfall widens all windows together.
    mission_window_adaptation: bool = False
    mission_window_history: int = 5
    mission_window_target: float = 0.8
    mission_window_gain: float = 2.0
    mission_window_max_scale: float = 4.0
    # Per-contact event traces. The run-dir JSONL is normally folded into
    # aggregate metrics and then DELETED at teardown, which is why no scheduling
    # policy can be scored retroactively against a finished sweep — there is
    # nothing left to replay. Setting a trace root copies each trial's raw
    # events (and the configs carrying device positions) alongside the CSV, so a
    # future baseline is a re-parse instead of another full re-run, and labels
    # each with its status (TRIAL_STATUS_FILE). Changes no trial behaviour; it
    # only stops the deletion.
    trace_root: Optional[Path] = None
    # FeRRy Phase 1 — mule arms only (H0 is flat FL and keeps its own mean).
    # ``aggregation`` names the L3 merge rule the cluster AND the mule run
    # (hermes/mission/aggregation_rules.py) and ``aggregation_params`` its
    # parameters; ``fedprox_rho`` is the devices' proximal weight;
    # ``pass_2_budget`` walks Pass 2 against ``mission_budget_s`` so the
    # devices it cannot reach keep an older basis. The defaults reproduce
    # every recorded run.
    aggregation: str = "agg:plain"
    aggregation_params: Dict[str, Any] = field(default_factory=dict)
    fedprox_rho: float = 0.0
    pass_2_budget: bool = False
    # FeRRy Phase 1 — the deadline law on the mule's scheduler ("additive" is
    # the recorded law) and the miss-streak priority key in S3b.
    deadline_law: str = "additive"
    deadline_params: Dict[str, Any] = field(default_factory=dict)
    miss_priority: bool = False
    # FeRRy Phase 2 — mule arms only. ``n_mules`` mules share the cluster over
    # disjoint spatial slices of the same N devices (N is the total, not per
    # slice); ``min_participation`` is the cluster's quorum, 1 or ``n_mules``
    # (FedBuff ignores it). 1 and 1 are the recorded single-mule topology. ``dock_on_empty`` and ``down_wait_s`` are
    # the mules' dock settings; None picks them from ``n_mules``: off for one
    # mule (the recorded dock), and for several an empty mission still docks
    # (so no quorum waits on it) and a DOWN is waited for as long as the trial
    # budget (so a quorum wait never ends a mule's run on its own).
    n_mules: int = 1
    min_participation: int = 1
    dock_on_empty: Optional[bool] = None
    down_wait_s: Optional[float] = None
    # FeRRy Phase 2 — D3/D5 options (the policies' defaults). D3's
    # weights='oort' ranks on Oort's utility, which needs --real-model.
    whittle_variant: str = "expected"
    whittle_weights: str = "uniform"
    fedcs_value: str = "unit"
    # ---- FeRRy Phase 3: the mission clock and the contact link ---- #
    # ``mission_clock="sim"`` runs every mule arm on the simulated mission
    # clock (hermes/mule/ferry.py); "wall" is every recorded run, and every
    # field below except the deadline time unit, the session TTL and the RF
    # token then keeps its default. The switches (design section 5.1):
    # ``contact_band`` (None = the channel-free control), ``in_flight_response``
    # ("abort" | "replan") with ``replan_fallback`` ("reorder" | "trim"),
    # ``backhaul_model`` ("mission" | "seconds"; H3 runs the adaptive carrier
    # policy, every other arm the fixed one), ``contact_reliability_source``
    # ("origin" | "channel"), ``payload_bytes`` (None = measured) and
    # ``deadline_bounds`` ("collection" | "delivery_per_stop" | "delivery";
    # validated by building the FerrySpec, recorded in ``ferry_params``).
    # ``ferry_physics`` overrides D1-D3 parameters by
    # ``MuleConfig`` field name (``FERRY_PHYSICS_FIELDS``). The exit-gate
    # configuration is chosen at the pilot, not here (critic B5).
    mission_clock: str = "wall"
    contact_band: Optional[str] = None
    contact_band_classes: Optional[Sequence[str]] = None
    in_flight_response: str = "abort"
    replan_fallback: str = "reorder"
    backhaul_model: str = "mission"
    contact_reliability_source: str = "origin"
    payload_bytes: Optional[int] = None
    deadline_bounds: str = "collection"
    ferry_physics: Dict[str, Any] = field(default_factory=dict)
    # The seconds-axis backhaul period P_bh; None = n_missions * T_nom.
    backhaul_period_s: Optional[float] = None
    # T_nom (spec Q1): given, or None to compute it per cell when a setting
    # needs it (:meth:`nominal_period_s`) over ``t_nom_layouts`` reference
    # layouts.
    t_nom_s: Optional[float] = None
    t_nom_layouts: int = 20
    # The deadline law's time unit, on either clock: a number (1.0 is the
    # recorded law) or "t_nom" for T_nom / 10 s. Φ₀ as ``initial_window_s``
    # (the law's recorded unit) or as ``initial_window_missions`` nominal
    # mission periods (critic A7); ``agg_period_t_nom`` sets agg:cutoff's
    # D5 ``period_s`` to T_nom. The values are chosen at the pilot.
    deadline_time_scale: Any = 1.0
    initial_window_s: Optional[float] = None
    initial_window_missions: Optional[float] = None
    agg_period_t_nom: bool = False
    # The mule's wall-clock session TTL; None = the builder's recorded 3 s.
    # Ferry cells set it from the measured real-model fit time (>= 2x, spec
    # Q12); the pilot measures it.
    session_ttl_s: Optional[float] = None
    # Amendment 10: one RF link token per trial, derived from the trial's
    # identity (:func:`trial_link_token`). None = on exactly on the simulated
    # clock; True / False force it.
    rf_link_token: Optional[bool] = None
    # Design R8: the input width a ferry cell's real model must have; None =
    # the data source's own (21 canonical, the synthetic task's otherwise).
    expected_input_dim: Optional[int] = None
    # ---- FeRRy Phase 4: the plan clock (simulated clock only) ---- #
    # ``member_admission`` ("whole" | "subset"; decision 4 (b)): None gives each
    # arm its own default, "subset" for the plan arms and "whole" for H1-H3,
    # D1-D3 and D5; D4 always runs whole (``effective_member_admission``). The
    # rest configure the plan arms (``PLAN_ARMS``) and are chosen at the
    # pilots: the age cap S in missions (None: off; decision 1, the S* tool
    # recommends it) with its lookahead L, and the plan score's and search's
    # settings by ``PlanScoreParams`` / ``PlanSearchParams`` field name (the
    # pilot sweeps kappa as ``c_cov_per_device``, and c4 as ``c_energy``). Each
    # arm's own change goes on top (``plan_settings``). Declared before
    # ``soft_cap_s``, which the Phase 3 final check pins as the last field
    # (tests/unit/test_p3_final_fixes_driver.py); every caller passes these by
    # keyword.
    member_admission: Optional[str] = None
    age_cap_missions: Optional[int] = None
    age_cap_lookahead: int = 0
    plan_score_params: Dict[str, Any] = field(default_factory=dict)
    plan_search_params: Dict[str, Any] = field(default_factory=dict)
    # ---- FeRRy Phase 5: the learned arms' checkpoints (simulated clock only) ---- #
    # ``pair_checkpoints`` maps a pair tag (``PAIR_CHECKPOINT_TAGS``: main, hand,
    # dwell, cov, g0 ... g99) to the pair_q checkpoint its FQ arm flies, and
    # ``policy_checkpoints`` maps E3's tag (e3) to its chen_dqn checkpoint; a
    # learned arm whose tag has none is refused (no random-init arm). The runner
    # fills them (--pair-checkpoint TAG=PATH, --policy-checkpoint E3=PATH) after
    # refusing any a campaign may not fly; the driver verifies each file as the
    # mule will, but no training state (the Phase 5 spec, other choices 6).
    # Declared before ``soft_cap_s``, the last field (see above).
    pair_checkpoints: Dict[str, Any] = field(default_factory=dict)
    policy_checkpoints: Dict[str, Any] = field(default_factory=dict)
    # Exp 5 addendum, Study 5.11: the footprint probe. ``footprint_probe``
    # samples the RSS of every process a real-process trial starts (the
    # cluster, the mules, the devices) every ``footprint_interval_s`` and
    # writes ``footprint.json`` beside the kept trace (FOOTPRINT_FILE), which
    # the scorer reads into its cost columns. It reads only, so no trial
    # changes; it needs a trace root to write to and psutil to read with. Off
    # by default. Declared before ``soft_cap_s``, the last field.
    footprint_probe: bool = False
    footprint_interval_s: float = 0.5
    # Exp 5 addendum, Studies 5.9 and 5.11: a field that grows with N. With
    # ``h1_field_ref_n`` set, a realism trial of N devices scatters them over
    # the half-width ``h1_field_radius_m * sqrt(N / h1_field_ref_n)``
    # (``topology_builder.grown_field_radius_m``), so the device density is the
    # reference size's at every N, T_nom's reference layouts included; None
    # keeps the recorded fixed field. The kept traces hold the positions; the
    # row does not record it, so write each setting to its own CSV.
    h1_field_ref_n: Optional[int] = None
    # The runner's soft cap on a trial's run time, as the caller applies it
    # (runner_main sets it to the cap it hands TrialRunner: --timeout-s, else
    # the largest wall budget over the grid). On the mission clock a trial's
    # own budget is re-costed per cell and can be below that cap, so its status
    # marker records the cap as ``soft_cap_s`` for a trace scored without its
    # trial CSV. None = not told: no such key. Wall markers never carry it.
    soft_cap_s: Optional[float] = None

    def __post_init__(self) -> None:
        from hermes.mission.aggregation_rules import AggregationSpec
        from hermes.scheduler.policies.fedcs_degraded import VALUE_KINDS
        from hermes.scheduler.policies.whittle import VARIANTS, WEIGHT_MODES
        from hermes.scheduler.stages.s3_deadline import DeadlineLaw, DeadlineLawError

        # Fail on a bad rule or law before any process is spawned.
        self._aggregation_spec = AggregationSpec.from_config(
            self.aggregation, self.aggregation_params,
        )
        # Normalised as from_config normalises it, and only once, so a pairs
        # list is checked like a mapping and a one-shot iterable still reaches
        # the law whole.
        deadline_params = dict(self.deadline_params or {})
        if "time_scale" in deadline_params:
            # FeRRy Phase 3 gave the law a time_scale, which from_config now
            # accepts (afa9526 refused the key, with DeadlineLawError).
            # Through deadline_params the scheduler would run that unit while
            # the deadline_time_scale column stays blank, and beside
            # deadline_time_scale the mule would refuse the pair only at start.
            raise DeadlineLawError(
                "deadline_params cannot set the deadline law's time_scale: give "
                "the time unit as deadline_time_scale (--deadline-time-scale), "
                "which the row records"
            )
        self._deadline_law = DeadlineLaw.from_config(
            self.deadline_law, deadline_params,
        )
        if self.fedprox_rho < 0.0:
            raise ValueError(f"fedprox_rho must be >= 0, got {self.fedprox_rho}")
        if self.pass_2_budget and self.mission_budget_s is None:
            raise ValueError(
                "pass_2_budget walks Pass 2 against mission_budget_s; set a "
                "budget (--mission-budget-s) or drop --pass-2-budget"
            )
        if self.whittle_variant not in VARIANTS:
            raise ValueError(
                f"whittle_variant must be one of {VARIANTS}, got {self.whittle_variant!r}"
            )
        if self.whittle_weights not in WEIGHT_MODES:
            raise ValueError(
                f"whittle_weights must be one of {WEIGHT_MODES}, "
                f"got {self.whittle_weights!r}"
            )
        if self.fedcs_value not in VALUE_KINDS:
            raise ValueError(
                f"fedcs_value must be one of {VALUE_KINDS}, got {self.fedcs_value!r}"
            )
        self._check_footprint_probe()
        if self.h1_field_ref_n is not None:
            ref_n = self.h1_field_ref_n
            if isinstance(ref_n, bool) or not isinstance(ref_n, int) or ref_n < 1:
                raise ValueError(f"h1_field_ref_n must be an int >= 1 or None, got {ref_n!r}")
            if not self.realism:
                raise ValueError(
                    "h1_field_ref_n scales the realism field: without realism the devices "
                    "sit in the tight cluster and no field is drawn (--realism)"
                )
        self._check_multi_mule()
        #: T_nom per cell, computed once (:meth:`nominal_period_s`).
        self._t_nom_cache: Dict[str, float] = {}
        self._check_clock()
        self._check_plan()
        self._check_checkpoints()

    @property
    def sim(self) -> bool:
        """True when the mule arms run on the simulated mission clock."""
        return self.mission_clock == "sim"

    def _check_clock(self) -> None:
        """Refuse clock settings that cannot run or would mis-measure (FeRRy Phase 3).

        On the wall clock every ferry-only setting must keep its default. On
        the simulated clock: a known backhaul model, not both backhaul loss
        models (``l1_channel``'s recorded schedule with the seconds model;
        under the ``mission`` model ``l1_channel`` keeps its loss schedule and
        the mule's RF prior is fed from past uploads, critic B4:
        :func:`chosen_snr_schedule`), the channel reliability source only
        with a band (critic B16), one way of stating Φ₀, and D5's T_nom
        period only on agg:cutoff. Every value the ferry spec takes is checked
        by building one. Several mules below a quorum of every mule, or under
        FedBuff, run as on the wall clock (:meth:`_check_multi_mule`): the
        cluster folds their uploads in simulated-time order (critic B9, unit
        U9; ``hermes.processes.cluster.SimOrderGate``).
        """
        from hermes.processes.config import MISSION_CLOCKS

        if self.mission_clock not in MISSION_CLOCKS:
            raise ValueError(
                f"mission_clock must be one of {MISSION_CLOCKS}, got {self.mission_clock!r}"
            )
        scale = self.deadline_time_scale
        if isinstance(scale, str):
            if scale != "t_nom":
                raise ValueError(
                    f"deadline_time_scale must be a number or 't_nom', got {scale!r}"
                )
        elif isinstance(scale, bool) or not (
            isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0.0
        ):
            raise ValueError(f"deadline_time_scale must be finite and > 0, got {scale!r}")
        if self.initial_window_s is not None and self.initial_window_missions is not None:
            raise ValueError("give Φ₀ as initial_window_s or initial_window_missions, not both")
        if self.session_ttl_s is not None and not float(self.session_ttl_s) > 0.0:
            raise ValueError(f"session_ttl_s must be > 0, got {self.session_ttl_s}")
        unknown = sorted(set(self.ferry_physics) - set(FERRY_PHYSICS_FIELDS))
        if unknown:
            raise ValueError(
                f"ferry_physics keys {unknown} are not ferry physics fields "
                f"({list(FERRY_PHYSICS_FIELDS)})"
            )
        if not self.sim:
            ferry_only = {
                "contact_band": self.contact_band is not None,
                "contact_band_classes": self.contact_band_classes is not None,
                "in_flight_response": self.in_flight_response != "abort",
                "replan_fallback": self.replan_fallback != "reorder",
                "backhaul_model": self.backhaul_model != "mission",
                "contact_reliability_source": self.contact_reliability_source != "origin",
                "payload_bytes": self.payload_bytes is not None,
                "deadline_bounds": self.deadline_bounds != "collection",
                "ferry_physics": bool(self.ferry_physics),
                "backhaul_period_s": self.backhaul_period_s is not None,
                "t_nom_s": self.t_nom_s is not None,
                "deadline_time_scale='t_nom'": isinstance(scale, str),
                "initial_window_missions": self.initial_window_missions is not None,
                "agg_period_t_nom": bool(self.agg_period_t_nom),
                "expected_input_dim": self.expected_input_dim is not None,
                # FeRRy Phase 4: member subsets and the plan arms' settings.
                "member_admission": self.member_admission not in (None, "whole"),
                "age_cap_missions": self.age_cap_missions is not None,
                "age_cap_lookahead": self.age_cap_lookahead != 0,
                "plan_score_params": bool(self.plan_score_params),
                "plan_search_params": bool(self.plan_search_params),
                # FeRRy Phase 5: the learned arms' checkpoints.
                "pair_checkpoints": bool(self.pair_checkpoints),
                "policy_checkpoints": bool(self.policy_checkpoints),
            }
            changed = [k for k, v in ferry_only.items() if v]
            if changed:
                raise ValueError(
                    f"{', '.join(changed)}: only on the simulated mission clock; set "
                    f"mission_clock='sim' (--mission-clock sim) or leave the default"
                )
            return
        if self.backhaul_model not in ("mission", "seconds"):
            raise ValueError(
                f"backhaul_model must be 'mission' or 'seconds', got {self.backhaul_model!r}"
            )
        if self.l1_channel and self.backhaul_model == "seconds":
            raise ValueError(
                "l1_channel's per-mission loss schedule and backhaul_model='seconds' are "
                "two backhaul loss models: the seconds model already runs H3's adaptive "
                "carrier; drop --l1-channel"
            )
        if self.contact_reliability_source == "channel" and self.contact_band is None:
            raise ValueError(
                "contact_reliability_source='channel' needs a contact_band: the "
                "reliability is then the SNR gate at the stop (critic B16)"
            )
        from hermes.mission.aggregation_rules import AGG_CUTOFF

        if self.agg_period_t_nom:
            if self._aggregation_spec.rule != AGG_CUTOFF:
                raise ValueError("agg_period_t_nom sets agg:cutoff's period_s; use --aggregation agg:cutoff")
            if "period_s" in (self.aggregation_params or {}):
                raise ValueError("agg_period_t_nom and aggregation_params['period_s'] are exclusive")
        if self.t_nom_s is not None and not (math.isfinite(float(self.t_nom_s))
                                             and float(self.t_nom_s) > 0.0):
            raise ValueError(f"t_nom_s must be finite and > 0, got {self.t_nom_s}")
        if int(self.t_nom_layouts) < 1:
            raise ValueError(f"t_nom_layouts must be >= 1, got {self.t_nom_layouts}")
        # Every value the ferry spec takes, checked by building one now (a
        # placeholder seed and backhaul period: neither is validated).
        from hermes.mule.ferry import FerrySpec

        probe = self.ferry_settings(arm="H1", regime="clean")
        if probe.get("backhaul_model") == "seconds":
            probe["backhaul_period_s"] = probe.get("backhaul_period_s") or 1.0
        FerrySpec.from_config(**self._spec_kwargs(
            probe, rf_range_m=float(self.default_rf_range_m), seed=0,
            n_missions=int(self.default_n_missions),
        ))

    def ferry_settings(self, *, arm: str, regime: str) -> Dict[str, Any]:
        """The ``MuleConfig`` ferry fields of a trial of ``arm`` in ``regime``.

        The switches as configured; H3 runs the adaptive backhaul carrier
        policy (its L1 controller at every upload), every other arm the fixed
        one; the backhaul regime is the cell's; the ferry physics overrides
        on top. T_nom is added per cell by :meth:`run_trial` when known.

        FeRRy Phase 4: a plan arm flies :meth:`arm_contact_band` (FB+<class>
        its class, which travels here because the builder takes the band only
        in the ferry settings, critic A8; the search arms the configured band,
        their reference class) and re-plans with the ``trim`` fallback, whatever
        the configured one: plan mode's Pass-1 re-plan is a trim of the
        committed plan (members under ``subset``, whole stops only under
        ``whole``), priority stops first, and the scheduler refuses ``reorder``
        there, as re-ordering belongs to the flight slot (critic B11). Every
        other arm's settings are exactly the configured ones.

        FeRRy Phase 5: an FQ arm's are F's (:func:`is_plan_arm`), and H1+L1
        flies H3's adaptive backhaul policy.
        """
        out: Dict[str, Any] = {
            "contact_band": self.arm_contact_band(arm),
            "contact_band_classes": (
                None if self.contact_band_classes is None else list(self.contact_band_classes)
            ),
            "in_flight_response": self.in_flight_response,
            "replan_fallback": "trim" if is_plan_arm(arm) else self.replan_fallback,
            "backhaul_model": self.backhaul_model,
            "backhaul_policy": "adaptive" if arm in _ADAPTIVE_BACKHAUL_ARMS else "fixed",
            "backhaul_regime": "jittery" if regime == "jittery" else "clean",
            "backhaul_period_s": self.backhaul_period_s,
            "contact_reliability_source": self.contact_reliability_source,
            "payload_bytes": self.payload_bytes,
            "deadline_bounds": self.deadline_bounds,
        }
        out.update(self.ferry_physics)
        return out

    # ------------------------------------------------------------------ #
    # FeRRy Phase 4 — the plan arms and member subsets
    # ------------------------------------------------------------------ #

    def _check_plan(self) -> None:
        """Refuse plan settings no arm could run (FeRRy Phase 4).

        On the wall clock :meth:`_check_clock` has already refused every one
        that is set. Here: a known ``member_admission``, a cap S that is None
        or an int >= 1, a lookahead that is an int >= 0 (the types' own rules,
        ``AgeCapSpec``), and settings that are mappings. The score and search
        settings, when given, are built as a plan mule builds them
        (``PlanOptions.from_config``, so an unknown key is refused before any
        trial), which loads the plan package on that path only.
        """
        from hermes.processes.config import MEMBER_ADMISSIONS

        if self.member_admission is not None and self.member_admission not in MEMBER_ADMISSIONS:
            raise ValueError(
                f"member_admission must be one of {MEMBER_ADMISSIONS} or None (each arm's "
                f"own default), got {self.member_admission!r}"
            )
        cap = self.age_cap_missions
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError(f"age_cap_missions must be None or an int >= 1, got {cap!r}")
        lookahead = self.age_cap_lookahead
        if isinstance(lookahead, bool) or not isinstance(lookahead, int) or lookahead < 0:
            raise ValueError(f"age_cap_lookahead must be an int >= 0, got {lookahead!r}")
        for name in ("plan_score_params", "plan_search_params"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be a mapping, got {getattr(self, name)!r}")
        if self.plan_score_params or self.plan_search_params:
            from hermes.scheduler.plan.types import PlanOptions

            try:
                PlanOptions.from_config(
                    plan_score_params=dict(self.plan_score_params),
                    plan_search_params=dict(self.plan_search_params),
                )
            except (TypeError, ValueError) as e:
                raise ValueError(f"plan settings: {e}") from e

    def arm_contact_band(self, arm: str) -> Optional[str]:
        """The band class a trial of ``arm`` gives its mule's ``contact_band``.

        FB+<class> flies its class (the arm pins it); every other arm the
        configured ``contact_band``, which for the searching plan arms is only
        the reference class (the ferry spec, T_nom, D4's split and the H and D
        arms of the same CSV use it; the Phase 4 spec, other choices 2).
        """
        pinned = _PLAN_ARM.get(arm, {}).get("band_class_policy")
        if pinned is not None:
            return pinned[len("fixed:"):]
        return self.contact_band

    def effective_member_admission(self, arm: str) -> str:
        """``member_admission`` for a trial of ``arm`` (decision 4 (b); unit_U3b.md 5.4).

        The setting, else the arm's own default: ``subset`` for a plan arm
        (the F family's rule), ``whole`` for H1-H3, D1-D3 and D5 (the recorded
        gates). D4 runs ``whole`` whatever the setting: its tour has no gate.
        FeRRy Phase 5: an FQ arm as F, H1+L1 as H1, and E3 ``whole`` always,
        since it visits a stop for all its members (the config guard's rule).
        """
        if arm not in _SUBSET_ARMS:
            return "whole"
        if self.member_admission is not None:
            return self.member_admission
        return "subset" if is_plan_arm(arm) else "whole"

    def effective_miss_priority(self, arm: str) -> bool:
        """``miss_priority`` for a trial of ``arm``, as its mule runs and its row records it.

        A plan arm's coverage weight is the device's age times (1 + its miss
        streak) when its ``miss_priority`` is on (decision 3): on for every
        plan arm but F-prio, which weighs by age alone, so on for every FQ arm
        too, as for F. Every other arm runs the configured value, as recorded.
        """
        if is_plan_arm(arm):
            return arm != "F-prio"
        return bool(self.miss_priority)

    def plan_settings(self, arm: str) -> Dict[str, Any]:
        """The ``MuleConfig`` plan fields of a trial of ``arm``, as topology parameters.

        A plan arm gets every plan field: F's (``plan_mode="ferry"``, the
        ``search`` band-class policy, the ``committed`` flight slot), the
        driver's member admission (else ``subset``), cap S, lookahead and score
        and search settings, and on top the arm's own change (FX's slot,
        FB+<class>'s pinned class, F-cov's coverage term off, F-cap's cap off).
        An H or D arm gets ``member_admission`` only when it is not the
        recorded ``whole``, and every other arm nothing, so a recorded arm's
        topology is built with exactly the arguments it always was.

        FeRRy Phase 5 (critic A5): an FQ arm gets F's fields with the pair slot
        (``flight_slot="pair_q"``), FQ-cov F-cov's score and FQ-dwell the dwell
        out of Δ (:data:`_ARM_SCORE`); its checkpoint is not a plan field
        (:meth:`checkpoint_settings`).
        """
        admission = self.effective_member_admission(arm)
        if not is_plan_arm(arm):
            return {} if admission == "whole" else {"member_admission": admission}
        out: Dict[str, Any] = {
            "plan_mode": "ferry",
            "band_class_policy": "search",
            "member_admission": admission,
            "flight_slot": "committed",
            "age_cap_missions": self.age_cap_missions,
            "age_cap_lookahead": int(self.age_cap_lookahead),
            "plan_score_params": dict(self.plan_score_params),
            "plan_search_params": dict(self.plan_search_params),
        }
        if arm in _ARM_SCORE:
            out["plan_score_params"].update(_ARM_SCORE[arm])
        out.update(_PLAN_ARM[arm])
        return out

    # ------------------------------------------------------------------ #
    # FeRRy Phase 5 — the learned arms' checkpoints, and H1+L1
    # ------------------------------------------------------------------ #

    def _check_checkpoints(self) -> None:
        """Refuse checkpoint settings no learned arm could fly (FeRRy Phase 5).

        Each mapping runs from a learned arm's tag to a path: a pair tag
        (:data:`PAIR_CHECKPOINT_TAGS`) in ``pair_checkpoints``, E3's in
        ``policy_checkpoints``. A tag no arm flies is refused rather than
        ignored, and so is a path that is not a non-empty string or path. The
        files are read only when an arm that flies one is checked
        (:meth:`check_arm`), so building a driver reads none. On the wall clock
        :meth:`_check_clock` has refused both mappings already.
        """
        for name, tags in (("pair_checkpoints", PAIR_CHECKPOINT_TAGS),
                           ("policy_checkpoints", POLICY_CHECKPOINT_TAGS)):
            given = getattr(self, name)
            if not isinstance(given, Mapping):
                raise ValueError(f"{name} must be a mapping of checkpoint tag to path, got "
                                 f"{given!r}")
            unknown = sorted(repr(tag) for tag in given if tag not in tags)
            if unknown:
                raise ValueError(f"{name}: {', '.join(unknown)} is no learned arm's tag; the "
                                 f"tags are {list(tags)}")
            for tag, path in given.items():
                if not isinstance(path, (str, Path)) or not str(path).strip():
                    raise ValueError(f"{name}[{tag!r}] must be the checkpoint's path, got "
                                     f"{path!r}")
        #: (kind, tag) -> (the config's path, the verified sha256), read once
        #: (:meth:`_verified_checkpoint`), so every trial names the same sha.
        self._checkpoints: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def checkpoint_settings(self, arm: str) -> Dict[str, Any]:
        """The ``MuleConfig`` checkpoint fields of a trial of ``arm``, as topology parameters.

        FeRRy Phase 5 (the Phase 5 spec, other choices 5 and 6). An FQ arm
        flies the pair checkpoint of its tag and E3 its policy checkpoint, each
        as (path, sha256, tag): the path relative to the repository root when
        the file lies inside it, else absolute; the sha the verified manifest's
        (:meth:`_verified_checkpoint`). Every other arm gets nothing, so its
        topology is built with exactly the arguments it always was. Raises
        ValueError for a learned arm whose tag has no checkpoint (no
        random-init arm) or whose file is refused.
        """
        if arm not in LEARNED_ARMS:
            return {}
        path, sha256 = self._verified_checkpoint(arm)
        prefix = "policy" if arm == "E3" else "pair"
        return {f"{prefix}_checkpoint": path, f"{prefix}_checkpoint_sha256": sha256,
                f"{prefix}_checkpoint_tag": CHECKPOINT_TAGS[arm]}

    def _verified_checkpoint(self, arm: str) -> Tuple[str, str]:
        """(the config's path, the sha256) of ``arm``'s checkpoint, verified once.

        The path as given is read against the working directory, as a command
        line reads it, and verified whole (``pair_q.verify_checkpoint``: the
        format, the header and the arrays against the manifest), so the sha a
        trial's config names, and its row records, is that of the arrays
        checked here; the mule refuses any others. It must be the arm's kind
        (``pair_q`` for an FQ arm, ``chen_dqn`` for E3). The config gets the
        path relative to the repository root when the file lies inside it,
        with ``/`` separators, so a kept per-role JSON names no host directory
        (the mule reads it under the same root,
        ``hermes.processes.mule.REPO_ROOT``), and the resolved absolute path
        otherwise. That is not the Phase 5 spec's "as given" (other choices
        6): the mule reads every relative path under that root, so a relative
        path from outside the repository, as given, would name another file.
        No training state is checked: the runner refuses what a campaign may
        not fly (critic B9), so FerrySim's bootstrap checkpoints fly here.
        """
        tag = CHECKPOINT_TAGS[arm]
        e3 = arm == "E3"
        given = (self.policy_checkpoints if e3 else self.pair_checkpoints).get(tag)
        if given is None:
            flag = "--policy-checkpoint E3=PATH" if e3 else f"--pair-checkpoint {tag}=PATH"
            raise ValueError(
                f"arm {arm} flies the checkpoint tagged {tag!r}, and none was given ({flag}): "
                f"a learned arm flies a verified checkpoint, never a random one (no random-init "
                f"arm; the Phase 5 spec, other choices 5)"
            )
        from hermes.processes import mule as mule_process
        from hermes.scheduler.selector.pair_q import (
            KIND_CHEN_DQN,
            KIND_PAIR_Q,
            verify_checkpoint,
        )

        kind = KIND_CHEN_DQN if e3 else KIND_PAIR_Q
        cached = self._checkpoints.get((kind, tag))
        if cached is not None:
            return cached
        resolved = Path(given).resolve()
        try:
            manifest = verify_checkpoint(resolved)
        except (ValueError, OSError) as e:
            raise ValueError(f"arm {arm}: checkpoint {str(given)!r} is refused: {e}") from e
        if manifest["kind"] != kind:
            raise ValueError(
                f"arm {arm}: checkpoint {str(given)!r} is a {manifest['kind']!r} checkpoint, and "
                f"arm {arm} flies a {kind!r} one"
            )
        try:
            written = resolved.relative_to(mule_process.REPO_ROOT).as_posix()
        except ValueError:
            written = str(resolved)
        cached = (written, str(manifest["sha256"]))
        self._checkpoints[(kind, tag)] = cached
        return cached

    def _check_learned(self, arm: str, cfg, spec) -> None:
        """Load ``arm``'s checkpoint exactly as its mule will (FeRRy Phase 5).

        ``cfg`` is the arm's mule config and ``spec`` its ``FerrySpec``. The
        pair score is loaded under its own schema over the link's classes
        (``selector.pair_features.load_pair_scorer``) and E3's network on the
        contact band (``policies.chen_dqn.load_e3_network``), each against the
        config's sha, from the path the mule reads
        (``hermes.processes.mule.checkpoint_path``). So a checkpoint trained on
        other classes or rows is refused before any trial, and a file that is
        no longer the one whose sha the provenance names (rewritten since the
        first check) before each trial, rather than by a mule process the trial
        has already started. A few kilobytes per trial.
        """
        from hermes.processes.mule import checkpoint_path

        try:
            if arm == "E3":
                from hermes.scheduler.policies.chen_dqn import load_e3_network

                load_e3_network(checkpoint_path(cfg.policy_checkpoint),
                                expect_sha256=cfg.policy_checkpoint_sha256,
                                band=cfg.contact_band)
            else:
                from hermes.scheduler.selector.pair_features import load_pair_scorer

                load_pair_scorer(checkpoint_path(cfg.pair_checkpoint),
                                 expect_sha256=cfg.pair_checkpoint_sha256,
                                 classes=spec.link.names)
        except (ValueError, OSError) as e:
            raise ValueError(f"arm {arm}: its mule would refuse its checkpoint: {e}") from e

    def _check_adaptive_backhaul(self, arm: str) -> None:
        """Refuse ``H1+L1`` where it would fly as H1 (FeRRy Phase 5; decision 8 (a)).

        H1+L1 is H1's scheduler with H3's adaptive backhaul controller (critic
        A6), which flies only with the L1 channel (``l1_channel``: the adaptive
        per-mission loss schedule, on either clock) or, on the simulated clock,
        the seconds-axis backhaul (``backhaul_model="seconds"``: the controller
        at every upload). Anywhere else its trial would be H1's under another
        label, so it is refused rather than run.
        """
        if self.l1_channel or (self.sim and self.backhaul_model == "seconds"):
            return
        raise ValueError(
            f"arm {arm} is H1 with H3's adaptive backhaul (decision 8 (a)), which flies only "
            f"with the L1 channel (--l1-channel) or, on the simulated clock, the seconds-axis "
            f"backhaul (--backhaul-model seconds); here it would fly as H1"
        )

    def check_arm(self, arm: str) -> None:
        """Refuse an arm this driver cannot run, before any trial of it starts.

        Only the plan arms have needs to check (FeRRy Phase 4); every other
        known arm passes, and an unknown one raises as :meth:`run_trial` does.
        A plan arm needs the simulated clock (refused on the wall clock). Then
        its mule config is built as its trial would build it, with a
        placeholder T_nom and backhaul period (the trial computes both), and
        refused with the mule's own guards (``mule_config_errors``: a band
        class, which FB+<class> brings and the search arms take from the
        configured ``contact_band`` as their reference class; the budgeted
        Pass 2, critic B8; abort with a cap, critic A10; unknown score or search
        settings; the rest of the Phase 4 spec's other choices 10) and the
        ferry spec's (a class the link does not have).

        FeRRy Phase 5 (:data:`PHASE_5_ARMS`). An FQ arm is checked as F is,
        and also needs ``in_flight_response="replan"`` (the orchestrator's
        resolution R3: its mask folds the whole rest of the flight, which only
        the re-plan's departure check folds next) and the checkpoint of its
        tag. E3 needs the simulated clock and its checkpoint, and is checked
        against its mule's guards (a contact band, whole stops). There is no
        random-init arm, and each checkpoint is loaded as its mule will load it
        (:meth:`_check_learned`), but no training state is checked: that is
        the runner's (critic B9). H1+L1 needs the adaptive backhaul it is named
        for (:meth:`_check_adaptive_backhaul`).
        """
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; the driver runs {ARMS}")
        if arm == "H1+L1":
            self._check_adaptive_backhaul(arm)
            return
        if not is_plan_arm(arm) and arm not in LEARNED_ARMS:
            return
        if not self.sim:
            if arm in PLAN_ARMS:
                raise ValueError(
                    f"arm {arm} flies the plan clock (FeRRy Phase 4) on the simulated mission "
                    f"clock: run it with mission_clock='sim' (--mission-clock sim)"
                )
            raise ValueError(
                f"arm {arm} flies a learned filling (FeRRy Phase 5) on the simulated mission "
                f"clock: run it with mission_clock='sim' (--mission-clock sim)"
            )
        if arm in PAIR_ARMS and self.in_flight_response != "replan":
            raise ValueError(
                f"arm {arm} flies the pair score, whose mask admits a pair only when the whole "
                f"rest of the flight still fits, which is what the re-plan's departure check "
                f"folds next: run it with in_flight_response='replan' (--in-flight-response "
                f"replan; resolution R3), got {self.in_flight_response!r}"
            )
        settings = self.ferry_settings(arm=arm, regime="clean")
        from hermes.mule.ferry import FerrySpec
        from hermes.processes.config import MuleConfig, mule_config_errors

        if settings.get("backhaul_model") == "seconds":
            settings["backhaul_period_s"] = settings.get("backhaul_period_s") or 1.0
        settings["t_nom_s"] = float(self.t_nom_s) if self.t_nom_s is not None else 1.0
        policy = {"contact_policy": _ARM_POLICY[arm]} if arm in _ARM_POLICY else {}
        cfg = MuleConfig(
            mule_id=f"check-{arm}", rf_range_m=float(self.default_rf_range_m),
            n_missions=int(self.default_n_missions), mission_clock="sim", trial_seed=0,
            mission_budget_s=self.mission_budget_s, pass_2_budget=bool(self.pass_2_budget),
            miss_priority=self.effective_miss_priority(arm),
            **settings, **self.plan_settings(arm), **policy, **self.checkpoint_settings(arm),
        )
        errors = mule_config_errors(cfg)
        if errors:
            raise ValueError(f"arm {arm}: " + "; ".join(errors))
        try:
            spec = FerrySpec.from_config(**cfg.ferry_spec_kwargs())
        except (TypeError, ValueError) as e:
            raise ValueError(f"arm {arm}: {e}") from e
        if arm in LEARNED_ARMS:
            self._check_learned(arm, cfg, spec)

    @staticmethod
    def _spec_kwargs(
        settings: Mapping[str, Any], *, rf_range_m: float, seed: int, n_missions: int,
    ) -> Dict[str, Any]:
        """``FerrySpec.from_config`` keywords for ``settings``, through the
        mule's own mapping (``MuleConfig.ferry_spec_kwargs``), so the driver
        prices with exactly the spec the mule process will build."""
        from hermes.processes.config import MuleConfig

        cfg = MuleConfig(
            mule_id="t_nom", rf_range_m=float(rf_range_m), n_missions=int(n_missions),
            mission_clock="sim", trial_seed=int(seed), **dict(settings),
        )
        return cfg.ferry_spec_kwargs()

    def field_radius_m(self, n_devices: int) -> float:
        """The realism field's half-width for a trial of ``n_devices`` devices.

        ``h1_field_radius_m``, the recorded fixed field; with ``h1_field_ref_n``
        set, grown at the reference size's density (Exp 5 addendum,
        ``topology_builder.grown_field_radius_m``). Used only when realism is on.
        """
        if self.h1_field_ref_n is None:
            return float(self.h1_field_radius_m)
        return grown_field_radius_m(float(self.h1_field_radius_m), int(n_devices),
                                    int(self.h1_field_ref_n))

    def _check_footprint_probe(self) -> None:
        """Study 5.11's footprint probe needs somewhere to write and psutil to read with."""
        if not self.footprint_probe:
            return
        from experiments.exp4.footprint import probe_available

        if self.trace_root is None:
            raise ValueError(
                "footprint_probe writes footprint.json beside each kept trace: give a "
                "trace_root (--keep-event-traces)"
            )
        if not self.footprint_interval_s > 0:
            raise ValueError(
                f"footprint_interval_s must be > 0, got {self.footprint_interval_s!r}"
            )
        if not probe_available():
            raise ValueError("footprint_probe needs psutil (pip install psutil)")

    def _check_multi_mule(self) -> None:
        """Refuse a mule count and quorum that cannot run or would mis-measure."""
        if int(self.n_mules) < 1:
            raise ValueError(f"n_mules must be >= 1, got {self.n_mules}")
        if not 1 <= int(self.min_participation) <= int(self.n_mules):
            raise ValueError(
                f"min_participation must be in 1..n_mules={self.n_mules}, got "
                f"{self.min_participation}: a quorum larger than the mules "
                f"that can meet it never closes a round"
            )
        if self.down_wait_s is not None and not float(self.down_wait_s) > 0.0:
            raise ValueError(f"down_wait_s must be > 0, got {self.down_wait_s}")
        if (
            int(self.n_mules) > 1
            and self._aggregation_spec.is_plain
            and int(self.min_participation) != int(self.n_mules)
        ):
            # agg:plain overwrites θ with the mean of the merge's partials; a
            # quorum below every mule makes that one mule's models each time.
            raise ValueError(
                f"agg:plain with n_mules={self.n_mules} needs "
                f"min_participation={self.n_mules} (got {self.min_participation}): "
                f"with fewer, each merge overwrites θ with one mule's models, "
                f"last writer wins. Use an age-aware rule for asynchronous merges."
            )
        from hermes.mission.aggregation_rules import AGG_FEDBUFF

        if (
            1 < int(self.min_participation) < int(self.n_mules)
            and self._aggregation_spec.rule != AGG_FEDBUFF
        ):
            # Each merge takes the first quorum of mules to upload, so partials
            # pair up in no fixed order, and near the end of the run the last
            # mule's final partial can wait for mules that have finished — for
            # the whole trial budget, by default, and the trial times out.
            raise ValueError(
                f"min_participation={self.min_participation} with "
                f"n_mules={self.n_mules}: use 1 (asynchronous merges) or "
                f"{self.n_mules} (every mule in each merge); a quorum between "
                f"them can leave the last partial of the run waiting for mules "
                f"that have already finished"
            )
        if (
            int(self.min_participation) > 1
            and self._aggregation_spec.rule != AGG_FEDBUFF
            and not self.effective_dock_on_empty
        ):
            raise ValueError(
                f"min_participation={self.min_participation} needs dock_on_empty: "
                f"a mule whose mission collects nothing would never dock, and "
                f"the quorum could never close"
            )

    @property
    def effective_dock_on_empty(self) -> bool:
        """``dock_on_empty``, or on exactly when there are several mules."""
        if self.dock_on_empty is not None:
            return bool(self.dock_on_empty)
        return int(self.n_mules) > 1

    @property
    def effective_down_wait_s(self) -> Optional[float]:
        """``down_wait_s``, or the trial budget when there are several mules."""
        if self.down_wait_s is not None:
            return float(self.down_wait_s)
        return float(self.trial_budget_s) if int(self.n_mules) > 1 else None

    def _policy_params(self, arm: str) -> Dict[str, Any]:
        """The options of the arm's policy (D3, D5); empty for every other arm."""
        if arm == "D3":
            return {"variant": self.whittle_variant, "weights": self.whittle_weights}
        if arm == "D5":
            return {"value": self.fedcs_value}
        return {}

    # ------------------------------------------------------------------ #
    # FeRRy Phase 3 — per-cell clock settings
    # ------------------------------------------------------------------ #

    @property
    def effective_session_ttl_s(self) -> float:
        """The mule's wall-clock session TTL: the configured one or the recorded 3 s."""
        return float(SESSION_TTL_S if self.session_ttl_s is None else self.session_ttl_s)

    def ferry_wall_bound_s(self, *, n_devices: int, n_missions: int) -> float:
        """The longest wall time a healthy trial on the mission clock can take (critic B14).

        Built from the waits the code bounds, not from a typical run: the
        mule's startup waits (devices 60 s, bootstrap DOWN 30 s); per mission,
        two passes of at most one contact per device of the largest slice,
        each contact at most one TTL gathering adverts and a 2 x TTL join
        (the ferry contact routine's caps), plus the recorded 10 s DOWN wait;
        with several mules, a quorum wait of up to one more such mission. When
        the cluster folds the uploads in simulated order (several mules below
        a full quorum, or FedBuff: unit U9) a mule's upload can wait for every
        other mule to pass it in simulated time, and at worst the K mules run
        one at a time: K such missions per mission. The mission clock's flight
        costs no wall time at all. At the recorded 3 s TTL, N = 6, one mule
        and 4 missions that is 562 s; at a 30 s TTL, 4,450 s. A trial that
        runs past it is hung, not slow.
        """
        ttl = self.effective_session_ttl_s
        k = max(1, int(self.n_mules))
        slice_n = math.ceil(int(n_devices) / k)
        mission = 2 * slice_n * 3.0 * ttl + _DOCK_WAIT_S
        if k > 1:
            mission *= max(2.0, float(k)) if self._sim_ordered else 2.0
        return _STARTUP_WALL_S + int(n_missions) * mission

    @property
    def _sim_ordered(self) -> bool:
        """True when the cluster folds the uploads in simulated-time order (unit U9).

        Several mules on the simulated clock, below a quorum of every mule or
        under FedBuff: ``hermes.processes.cluster.needs_sim_order`` of the
        cluster this driver configures.
        """
        from hermes.mission.aggregation_rules import AGG_FEDBUFF

        return self.sim and int(self.n_mules) > 1 and (
            int(self.min_participation) < int(self.n_mules)
            or self._aggregation_spec.rule == AGG_FEDBUFF
        )

    def trial_wall_budget_s(self, *, n_devices: int, n_missions: int) -> float:
        """The trial's hard wall-clock kill: ``trial_budget_s``, raised on the
        mission clock to :meth:`ferry_wall_bound_s` (critic B14), so a larger
        session TTL cannot kill a healthy trial. Wall-clock trials keep
        ``trial_budget_s`` exactly."""
        if not self.sim:
            return float(self.trial_budget_s)
        return max(float(self.trial_budget_s),
                   self.ferry_wall_bound_s(n_devices=n_devices, n_missions=n_missions))

    def _down_wait_for(self, budget_s: float) -> Optional[float]:
        """``down_wait_s``, or the trial's wall budget when there are several mules."""
        if self.down_wait_s is not None:
            return float(self.down_wait_s)
        return float(budget_s) if int(self.n_mules) > 1 else None

    @staticmethod
    def trial_link_token(cell) -> str:
        """The trial's RF link token (Amendment 10): one value per trial,
        shared by its mules and devices, derived from the trial's identity
        (cell, arm, trial index, seed) so the per-role JSON reproduces and no
        two trials, arms included, share one."""
        raw = f"{cell.cell_id}|{cell.arm}|{cell.trial_index}|{cell.seed}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _use_link_token(self) -> bool:
        return self.sim if self.rf_link_token is None else bool(self.rf_link_token)

    def declared_input_dim(self) -> int:
        """The input width a ferry cell's real model must have (design R8)."""
        if self.expected_input_dim is not None:
            return int(self.expected_input_dim)
        if self.data_source == "canonical":
            return CANONICAL_INPUT_DIM
        from .model_task import INPUT_DIM

        return int(INPUT_DIM)

    def _payload_bytes(self, init_theta_path: Optional[str]) -> Tuple[int, int]:
        """(θ bytes, synthetic batch bytes) the mule will push: the real seed
        weights when the trial has them, else the cluster's stub model, with
        the cluster's synthetic batch (``SYNTH_BATCH_SIZE`` stub samples)."""
        from hermes.cluster.host_cluster import StubGeneratorHost
        from hermes.processes.cluster import stub_disc_weights
        from hermes.types import weights_byte_count

        if init_theta_path is not None:
            from .model_task import load_weights

            theta = load_weights(init_theta_path)
        else:
            theta = stub_disc_weights()
        synth = StubGeneratorHost(disc_weights=[]).make_synth_batch(SYNTH_BATCH_SIZE)
        return weights_byte_count(theta), int(sum(int(a.nbytes) for a in synth))

    def _needs_t_nom(self, arm: Optional[str] = None) -> bool:
        """Whether a trial of ``arm`` needs T_nom: a setting derived from it, or a
        plan arm, whose score measures the mission against it (FeRRy Phase 4,
        decision 2 (b); an FQ arm too, as F); computed per cell when not given."""
        return self.sim and (
            is_plan_arm(arm)
            or (self.backhaul_model == "seconds" and self.backhaul_period_s is None)
            or self.deadline_time_scale == "t_nom"
            or self.initial_window_missions is not None
            or bool(self.agg_period_t_nom)
        )

    def nominal_period_s(
        self,
        *,
        n_devices: int,
        rf_range_m: float,
        regime: str,
        settings: Mapping[str, Any],
        theta_bytes: int,
        synth_bytes: int,
    ) -> float:
        """T_nom (spec Q1): the cell's median nominal two-pass mission period.

        One value per cell, the same for every arm and every trial of it:
        over ``t_nom_layouts`` reference layouts drawn as a trial's are (the
        builder's positions, the cell's device count and spread) from the
        seeds ``_u32(n_devices, "t_nom", k)``, each priced by unit U4's
        ``nominal_mission_period_s`` with a spec built as the mule's is but on
        the wide band (``R_planar(wide) == rf_range_m``), the reference
        seed's channel, a placeholder backhaul period (the planner never reads
        it) and this trial's payload. With several mules each layout is split
        into the default angular slices and priced as its slowest slice (a
        quorum of every mule waits for the slowest). The reference layouts do
        not depend on the grid's base seed or trial count, so extending or
        resuming a CSV never moves T_nom. Cached per cell.
        """
        from hermes.mule.ferry import FerrySpec
        from hermes.scheduler.fl_scheduler import nominal_mission_period_s
        from hermes.types import DeviceID

        from .model_task import _u32

        key = json.dumps({
            "n": int(n_devices), "rrf": float(rf_range_m), "regime": regime,
            "settings": dict(settings), "theta": int(theta_bytes), "synth": int(synth_bytes),
            "k": int(self.n_mules), "layouts": int(self.t_nom_layouts),
            "realism": bool(self.realism), "field": self.field_radius_m(n_devices),
        }, sort_keys=True, default=str)
        cached = self._t_nom_cache.get(key)
        if cached is not None:
            return cached
        spread = device_spread_m(
            rf_range_m, field_radius_m=(self.field_radius_m(n_devices) if self.realism else None),
        )
        base = dict(settings)
        base.update(contact_band="wide", backhaul_regime=regime)
        base.pop("t_nom_s", None)
        if base.get("backhaul_model") == "seconds":
            base["backhaul_period_s"] = 1.0          # placeholder: never read
        periods = []
        for k in range(int(self.t_nom_layouts)):
            ref_seed = _u32(int(n_devices), "t_nom", k)
            xy = device_positions(int(n_devices), ref_seed, spread)
            spec = FerrySpec.from_config(**self._spec_kwargs(
                base, rf_range_m=rf_range_m, seed=ref_seed, n_missions=1,
            ))
            model = spec.feasibility_model(
                rf_range_m=float(rf_range_m), theta_bytes=int(theta_bytes),
                synth_bytes=int(synth_bytes),
            )
            if int(self.n_mules) > 1:
                assign = angular_slices(xy, int(self.n_mules))
                slices = [[i for i in range(len(xy)) if assign[i] == m]
                          for m in range(int(self.n_mules))]
            else:
                slices = [list(range(len(xy)))]
            periods.append(max(
                nominal_mission_period_s(
                    {DeviceID(f"exp4-dev-{i:03d}"): (xy[i][0], xy[i][1], 0.0) for i in members},
                    rf_range_m=float(rf_range_m), feasibility_model=model,
                    turnaround_s=spec.flight.turnaround_s,
                )
                for members in slices
            ))
        t_nom = float(statistics.median(periods))
        self._t_nom_cache[key] = t_nom
        return t_nom

    def carp_t_trans_s(
        self, settings: Mapping[str, Any], *, rf_range_m: float, seed: int,
        theta_bytes: int, synth_bytes: int,
    ) -> float:
        """Arm D4's per-client time for the CARP split in a ferry cell.

        The predicted Pass-1 airtime of one client at half the contact range,
        R_planar(b)/2, at the band's mean SNR there (design section 3.2), with
        this trial's payload; without a band (the channel-free control) the
        cost model's per-contact session time, the recorded value.
        """
        from hermes.mule.ferry import FerryRuntime, FerrySpec
        from hermes.types import MissionPass

        spec_settings = dict(settings)
        if spec_settings.get("backhaul_model") == "seconds":
            spec_settings["backhaul_period_s"] = spec_settings.get("backhaul_period_s") or 1.0
        spec = FerrySpec.from_config(**self._spec_kwargs(
            spec_settings, rf_range_m=rf_range_m, seed=seed, n_missions=1,
        ))
        rt = FerryRuntime(spec, None, rf_range_m=float(rf_range_m))
        if not rt.banded:
            return float(rt.session_time_s)
        rt.set_payload(theta_bytes=int(theta_bytes), synth_bytes=int(synth_bytes))
        dwell = rt.member_dwell_s(rt.range_planar_m / 2.0, MissionPass.COLLECT, 0.0)
        return float(rt.session_time_s if dwell is None else dwell)

    def _resolve_clock(
        self,
        cell,
        *,
        arm: str,
        regime: str,
        n_devices: int,
        rf_range_m: float,
        n_missions: int,
        init_theta_path: Optional[str],
        input_dim: Optional[int],
    ) -> _TrialClock:
        """One trial's clock settings (FeRRy Phase 3); the recorded ones on the wall clock."""
        scale = self.deadline_time_scale
        budget = self.trial_wall_budget_s(n_devices=n_devices, n_missions=n_missions)
        token = self.trial_link_token(cell) if self._use_link_token() else None
        if not self.sim:
            return _TrialClock(
                deadline_time_scale=float(scale), initial_window_s=self.initial_window_s,
                aggregation_spec=self._aggregation_spec, budget_s=budget,
                down_wait_s=self._down_wait_for(budget), input_dim=input_dim,
                rf_link_token=token,
            )
        if self.real_model and input_dim is not None and int(input_dim) != self.declared_input_dim():
            raise ValueError(
                f"ferry cell refused: the model's input_dim is {input_dim}, not the declared "
                f"{self.declared_input_dim()} (design R8: a missing CICIoT dataset silently "
                f"builds the synthetic 46-input model, whose payload is 25,156 B, not 18,756 B)"
            )
        settings = self.ferry_settings(arm=arm, regime=regime)
        t_nom = None if self.t_nom_s is None else float(self.t_nom_s)
        computed = False
        if t_nom is None and self._needs_t_nom(arm):
            theta_b, synth_b = self._payload_bytes(init_theta_path)
            t_nom = self.nominal_period_s(
                n_devices=n_devices, rf_range_m=rf_range_m, regime=settings["backhaul_regime"],
                settings=settings, theta_bytes=theta_b, synth_bytes=synth_b,
            )
            computed = True
        if t_nom is not None:
            settings["t_nom_s"] = t_nom
        from hermes.mission.aggregation_rules import AggregationSpec
        from hermes.scheduler.stages.s3_deadline import (
            initial_window_for_missions,
            time_scale_for_period,
        )

        scale_value = time_scale_for_period(t_nom) if scale == "t_nom" else float(scale)
        phi0 = self.initial_window_s
        if self.initial_window_missions is not None:
            phi0 = initial_window_for_missions(
                self.initial_window_missions, t_nom, time_scale=scale_value,
            )
        agg = self._aggregation_spec
        if self.agg_period_t_nom:
            agg = AggregationSpec.from_config(
                agg.rule, {**agg.to_params(), "period_s": float(t_nom)},
            )
        return _TrialClock(
            sim=True, settings=settings, t_nom_s=t_nom, t_nom_computed=computed,
            deadline_time_scale=scale_value, initial_window_s=phi0,
            aggregation_spec=agg, budget_s=budget, down_wait_s=self._down_wait_for(budget),
            input_dim=input_dim, rf_link_token=token,
        )

    def run_trial(self, cell: Cell) -> Mapping[str, Any]:
        # Only the trial-status marker reads this: the runner's soft cap is
        # applied to the whole call, data preparation included.
        started = time.monotonic()
        params = cell.params
        arm = cell.arm
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; the driver runs {ARMS}")
        if arm in PLAN_ARMS or arm in PHASE_5_ARMS:
            # FeRRy Phase 4 and 5: refused before anything is prepared or spawned.
            self.check_arm(arm)

        n_devices = int(params.get("N", params.get("n_devices", self.default_n_devices)))
        rf_range_m = float(params.get("rrf", params.get("rf_range_m", self.default_rf_range_m)))
        n_missions = int(params.get("n_missions", self.default_n_missions))
        regime = str(params.get("regime", "clean"))

        if arm == "H0" and self.sim:
            # Critic A5: H0 (in-process flat FL) has no simulated round time
            # and no seconds-axis link; that is outside Phase 3.
            raise ValueError(
                "arm H0 has no simulated round time (critic A5, outside Phase 3): run "
                "it on the wall clock, in a CSV of its own"
            )
        if arm == "H0":
            if not self.real_model:
                raise ValueError(
                    "arm H0 (traditional flat FL) is a real-model convergence "
                    "baseline; run with real_model=True (--real-model)"
                )
            return self._run_h0(
                cell, n_devices=n_devices, rf_range_m=rf_range_m,
                n_rounds=n_missions, regime=regime,
            )

        # arm H1 — integrated stack over the real multi-process orchestrator.
        # EX-4.2 realism (opt-in): Exp 3's mule-arm impairment, asymmetric to
        # H0 (short-range contact reliability always; recoverable backhaul
        # loss under jittery; no dead-zone — the mule physically reaches
        # devices).
        realism_kwargs: dict = {}
        if self.realism:
            from .model_task import device_reliabilities
            realism_kwargs = dict(
                device_reliability=True,
                reliabilities=device_reliabilities(cell.seed, n_devices),
                world_radius_m=self.h1_world_radius_m,
                field_radius_m=self.field_radius_m(n_devices),
                backhaul_loss_pct=(
                    self.jittery_backhaul_loss_pct if regime == "jittery"
                    else self.clean_backhaul_loss_pct
                ),
                backhaul_rng_seed=(cell.seed ^ 0x0BACC0DE),
            )
            if self.sim and self.backhaul_model == "seconds":
                # FeRRy Phase 3 (spec Q7): the seconds-axis channel replaces
                # the flat realism percentage; the cluster draws against each
                # upload's own loss probability.
                realism_kwargs["backhaul_loss_pct"] = 0.0

        # EX-4.2/4.3 arms: H1 deterministic ranking; H2 = H1 + RL selector;
        # H3 = H2 + adaptive L1 channel.
        selector_kwargs = dict(
            use_rl_selector=(arm in ("H2", "H3")),
            selector_weights_path=self.selector_weights_path,
        )
        # D1-D5 — whole schedulers (``_ARM_POLICY`` names each one's policy).
        # Each replaces our policy outright and does not compose with the RL
        # selector, so use_rl_selector stays off. D1 is MAX-AoI; D3-D5 are the
        # FeRRy Phase 2 arms, in the same slot.
        if arm in _ARM_POLICY:
            selector_kwargs["contact_policy"] = _ARM_POLICY[arm]
        # D2 — Oort's statistical-utility selection. Needs REAL training: the
        # stub's loss is a random draw, so ranking on it would be a random
        # ordering wearing Oort's name. The policy itself raises, but fail here
        # with a clearer message.
        if arm == "D2" and not self.real_model:
            raise ValueError(
                "arm D2 (Oort) requires --real-model: the stub reports a "
                "random loss, so its ranking signal would be pure noise"
            )
        if arm == "D3":
            if self.whittle_weights == "oort" and not self.real_model:
                raise ValueError(
                    "arm D3 with whittle_weights='oort' requires --real-model: "
                    "its ω is Oort's utility, and the stub's loss is a random "
                    "draw; use whittle_weights='uniform' on the stub"
                )
            selector_kwargs.update(
                whittle_variant=self.whittle_variant,
                whittle_weights=self.whittle_weights,
            )
        if arm == "D5":
            selector_kwargs["fedcs_value"] = self.fedcs_value
        if self.mission_budget_s is not None:
            selector_kwargs["mission_budget_s"] = float(self.mission_budget_s)
        if self.mission_window_adaptation:
            selector_kwargs.update(
                mission_window_adaptation=True,
                mission_window_history=int(self.mission_window_history),
                mission_window_target=float(self.mission_window_target),
                mission_window_gain=float(self.mission_window_gain),
                mission_window_max_scale=float(self.mission_window_max_scale),
            )
        # FeRRy Phase 1 — one rule for cluster and mule, FedProx on devices,
        # and the budgeted Pass 2.
        selector_kwargs.update(
            aggregation=self._aggregation_spec.rule,
            aggregation_params=self._aggregation_spec.to_params(),
            fedprox_rho=float(self.fedprox_rho),
            pass_2_budget=bool(self.pass_2_budget),
            deadline_law=self._deadline_law.form,
            deadline_params=(
                {} if self._deadline_law.is_recorded
                else self._deadline_law.to_params()
            ),
            # FeRRy Phase 4: a plan arm runs its own (the configured value for
            # every other arm, as recorded).
            miss_priority=self.effective_miss_priority(arm),
        )
        # FeRRy Phase 4 — a plan arm's plan fields, or an H or D arm's member
        # subsets, as the builder's own parameters; empty for a recorded arm,
        # whose topology is then built with exactly the recorded arguments.
        plan_kwargs = self.plan_settings(arm)
        # FeRRy Phase 5 — a learned arm's checkpoint (path, sha256, tag), as
        # verified by check_arm above; empty for every other arm.
        plan_kwargs.update(self.checkpoint_settings(arm))
        # FeRRy Phase 2 — the mule count, the quorum and the dock settings. At
        # one mule with the defaults these are the builder's own defaults, so
        # the topology is the recorded one.
        if int(self.n_mules) > 1 or int(self.min_participation) != 1:
            selector_kwargs.update(
                n_mules=int(self.n_mules),
                min_participation=int(self.min_participation),
            )
        if self.effective_dock_on_empty:
            selector_kwargs["dock_on_empty"] = True
        # ``down_wait_s`` (and the merge spec above) are set per trial below,
        # once the trial's wall budget is known (FeRRy Phase 3).

        # EX-4.3 arm H3 — L1 adaptive channel. H1/H2 hold the best-average
        # fixed band; H3 runs the U(c,t) controller. The per-mission loss
        # schedule (cluster) + chosen-channel mean SNR (selector RF prior)
        # replace the flat backhaul loss for all mule arms in this mode.
        # FeRRy Phase 5: H1+L1 runs H3's controller too (decision 8 (a)).
        if self.l1_channel:
            from .channel import ChannelModel, backhaul_plan
            model = ChannelModel(
                n_bands=self.l1_channel_bands, n_missions=n_missions,
                seed=cell.seed, jittery=(regime == "jittery"),
            )
            plan = backhaul_plan(model, adaptive=(arm in _ADAPTIVE_BACKHAUL_ARMS))
            realism_kwargs["backhaul_loss_schedule"] = plan.loss_schedule
            if not self.sim:
                realism_kwargs["rf_prior_snr_db"] = plan.mean_chosen_snr_db
            else:
                # FeRRy Phase 3 (critic B4): the mean over the realized trace
                # uses the future, so a ferry cell's mule is not handed it. It
                # gets the chosen carrier's SNR at each mission instead and
                # adopts one entry per upload already made, so each Pass-1
                # plan sees only past uploads (the seconds model, which
                # _check_clock refuses with --l1-channel, has its own producer).
                realism_kwargs["rf_prior_schedule_db"] = chosen_snr_schedule(model, plan)
            # The schedule drives the cluster's per-mission Bernoulli draw; seed
            # it off the paired seed so H1/H2/H3 share the identical draw
            # sequence (paired comparison holds even without --realism).
            realism_kwargs.setdefault("backhaul_rng_seed", cell.seed ^ 0x0BACC0DE)
            log.info(
                "exp4 %s L1 channel cell=%s trial=%d regime=%s: adaptive=%s "
                "mean_snr=%.1fdB mean_loss=%.3f bands=%s",
                arm, cell.cell_id, cell.trial_index, regime, plan.adaptive,
                plan.mean_chosen_snr_db,
                sum(plan.loss_schedule) / max(1, len(plan.loss_schedule)),
                plan.chosen_bands,
            )

        prep_dir: Optional[Path] = None
        try:
            if self.real_model:
                prep_dir = Path(tempfile.mkdtemp(prefix="exp4_prep_"))
                task = self._build_task(n_devices, cell.seed)
                prep = prepare_trial(prep_dir, task=task, theta_seed=self.theta_seed)
                log.info(
                    "exp4 real-model H1 trial cell=%s trial=%d regime=%s "
                    "realism=%s: source=%s input_dim=%d n_train=%d synthetic=%s",
                    cell.cell_id, cell.trial_index, regime, self.realism,
                    self.data_source, prep.input_dim, prep.n_train, prep.is_synthetic,
                )
                model_kwargs = dict(
                    train_shard_paths=prep.shard_paths,
                    input_dim=prep.input_dim,
                    local_epochs=self.local_epochs,
                    local_batch_size=self.local_batch_size,
                    init_theta_path=prep.init_theta_path,
                    eval_test_path=prep.test_path,
                )
            else:
                model_kwargs = {}

            # FeRRy Phase 3 — this trial's clock settings: on the wall clock
            # the recorded merge spec, DOWN wait and budget; on the simulated
            # one the ferry settings, T_nom and what derives from it.
            clock = self._resolve_clock(
                cell, arm=arm, regime=regime, n_devices=n_devices,
                rf_range_m=rf_range_m, n_missions=n_missions,
                init_theta_path=model_kwargs.get("init_theta_path"),
                input_dim=model_kwargs.get("input_dim"),
            )
            selector_kwargs.update(
                aggregation=clock.aggregation_spec.rule,
                aggregation_params=clock.aggregation_spec.to_params(),
            )
            selector_kwargs.pop("down_wait_s", None)
            if clock.down_wait_s is not None:
                selector_kwargs["down_wait_s"] = clock.down_wait_s
            clock_kwargs: Dict[str, Any] = {}
            if clock.sim:
                clock_kwargs.update(mission_clock="sim", ferry_settings=clock.settings)
            if clock.deadline_time_scale != 1.0:
                clock_kwargs["deadline_time_scale"] = clock.deadline_time_scale
            if clock.initial_window_s is not None:
                clock_kwargs["initial_window_s"] = clock.initial_window_s
            if clock.rf_link_token is not None:
                clock_kwargs["rf_link_token"] = clock.rf_link_token
            if self.session_ttl_s is not None:
                clock_kwargs["session_ttl_s"] = float(self.session_ttl_s)

            def _build(**extra):
                return build_exp4_topology(
                    n_devices=n_devices,
                    rf_range_m=rf_range_m,
                    n_missions=n_missions,
                    seed=cell.seed,
                    **model_kwargs,
                    **realism_kwargs,
                    **selector_kwargs,
                    **clock_kwargs,
                    **plan_kwargs,
                    **extra,
                )

            topo = _build()
            if arm == "D4" and int(self.n_mules) > 1:
                # FedEx's CARP split, once per trial, over the positions this
                # seed lays out (the same ones the default split was built on).
                # FeRRy Phase 3: a ferry cell prices each client at its
                # predicted Pass-1 airtime and flies at its own cruise speed.
                carp_kwargs: Dict[str, Any] = {}
                if clock.sim:
                    theta_b, synth_b = self._payload_bytes(model_kwargs.get("init_theta_path"))
                    carp_kwargs = dict(
                        t_trans_s=self.carp_t_trans_s(
                            clock.settings, rf_range_m=rf_range_m, seed=cell.seed,
                            theta_bytes=theta_b, synth_bytes=synth_b,
                        ),
                        cruise_speed_m_s=float(topo.mules[0].cruise_speed_m_s),
                    )
                topo = _build(slice_assignment=d4_slice_assignment(
                    topo.devices, int(self.n_mules), cell.seed, **carp_kwargs,
                ))

            return self._run_topology(
                topo, cell=cell, n_devices=n_devices,
                rf_range_m=rf_range_m, n_missions=n_missions, started=started,
                clock=clock,
            )
        finally:
            if prep_dir is not None:
                shutil.rmtree(prep_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_h0(self, cell: Cell, *, n_devices, rf_range_m, n_rounds, regime="clean"):
        """Traditional flat FL (H0) — the paired real-model null.

        Runs synchronous FedAvg **in process** (no mule, no orchestrator):
        every round each reachable+sampled client trains the real DNN-IDS
        from the current global θ, the server ``partial_fedavg``-aggregates
        their weights, and the aggregated θ is scored on the shared held-out
        set. Uses the same task (same paired seed -> same shards), same
        seeded init θ, and the same convergence definitions as H1.

        EX-4.2 jittery regime: H0 relies on the long-range backhaul to every
        client, so under jitter a ``dead_zone_frac`` of clients are
        persistently unreachable and the reachable ones succeed each round
        only with prob ``link_quality`` — modelling the degraded-line-of-sight
        backhaul that collapses centralized participation (Exp 3's A1 model).
        A round with no successful updates does not close (deadline unmet).
        """
        import numpy as np

        from hermes.mission.partial_fedavg import partial_fedavg
        from hermes.types import DeviceID, GradientSubmission, MuleID

        from .metrics import summarise_flat_fl
        from .model_task import (
            _u32, device_reliabilities, initial_theta, make_local_train_fn,
        )
        from experiments.exp3.metrics import Exp3RoundLog

        jittery = regime == "jittery"
        # Dead-zone / link-quality are sweepable per cell (the sensitivity
        # surface, B2) — a cell param overrides the driver default. Only
        # meaningful under jittery.
        if jittery:
            dead_zone_frac = float(cell.params.get("dead_zone", self.jittery_dead_zone_frac))
            link_quality = float(cell.params.get("link_quality", self.jittery_link_quality))
        else:
            dead_zone_frac = self.clean_dead_zone_frac
            link_quality = self.clean_link_quality
        # Shared per-device reliability — the SAME draw H1 uses, so the clean
        # comparison is fair (H0 is not idealised to perfect participation).
        # H0 is all long-range: a reachable client contributes each round with
        # prob reliability_i x link_quality (link_quality = 1.0 clean, <1
        # jittery). Dead-zoned clients never contribute (permanent).
        rels = device_reliabilities(cell.seed, n_devices)

        task = self._build_task(n_devices, cell.seed)
        input_dim = task.input_dim

        # Persistent long-range dead zone — clients the central server never
        # reaches this mission (deterministic from the paired seed).
        n_dead = int(round(n_devices * dead_zone_frac))
        dz_rng = np.random.default_rng(_u32(cell.seed, "h0_deadzone"))
        dead = set(
            int(i) for i in dz_rng.choice(n_devices, size=n_dead, replace=False)
        ) if n_dead > 0 else set()
        reachable = [i for i in range(n_devices) if i not in dead]

        log.info(
            "exp4 H0 flat-FL cell=%s trial=%d regime=%s: source=%s input_dim=%d "
            "n_train=%d rounds=%d reachable=%d/%d link_quality=%.2f",
            cell.cell_id, cell.trial_index, regime, self.data_source, input_dim,
            task.n_train, n_rounds, len(reachable), n_devices, link_quality,
        )

        # Same seeded init θ as H1 so both arms start from the same model.
        theta = initial_theta(input_dim, seed=self.theta_seed)
        # Build a trainer only for reachable clients (dead ones never fit).
        client_fns = {
            i: make_local_train_fn(
                task.device_shards[i][0], task.device_shards[i][1],
                input_dim=input_dim, epochs=self.local_epochs,
                batch_size=self.local_batch_size, seed=self.theta_seed,
            )
            for i in reachable
        }

        evals = [self._eval_point(0, theta, task, input_dim)]
        round_logs: list = []
        participation = {i: 0 for i in range(n_devices)}
        samp_rng = np.random.default_rng(_u32(cell.seed, "h0_sampling"))
        link_rng = np.random.default_rng(_u32(cell.seed, "h0_link"))
        n_sample = max(1, int(round(len(reachable) * self.h0_client_fraction))) if reachable else 0

        for r in range(1, n_rounds + 1):
            if not reachable:
                sampled = []
            elif n_sample >= len(reachable):
                sampled = list(reachable)
            else:
                sampled = sorted(
                    int(i) for i in samp_rng.choice(reachable, size=n_sample, replace=False)
                )
            subs = []
            for i in sampled:
                # Long-range participation: device availability x link quality.
                # Applies in clean too (link_quality=1.0 -> prob = reliability),
                # so H0 pays the same heterogeneity tax as H1 — no idealised
                # clean win.
                p_i = rels[i] * link_quality
                if float(link_rng.random()) >= p_i:
                    continue
                res = client_fns[i](theta, [])
                participation[i] += 1
                subs.append(
                    GradientSubmission(
                        device_id=DeviceID(f"exp4-dev-{i:03d}"),
                        mule_id=MuleID("h0-server"),
                        mission_round=r,
                        delta_theta=res.delta_theta,
                        num_examples=res.num_examples,
                        submitted_at=0.0,
                    )
                )
            if subs:
                theta = partial_fedavg(MuleID("h0-server"), r, subs).weights
                closed = True
            else:
                # No client reached the server this round — no aggregation,
                # θ carries over, the round does not close.
                closed = False
            evals.append(self._eval_point(r, theta, task, input_dim))
            round_logs.append(
                Exp3RoundLog(
                    round_index=r, n_updates=len(subs),
                    n_target=n_devices, deadline_met=closed,
                )
            )

        summary = summarise_flat_fl(
            model_evals=evals,
            round_logs=round_logs,
            per_client_participation=participation,
            n_devices=n_devices,
            rf_range_m=rf_range_m,
            n_missions_target=n_rounds,
            tau=self.tau,
        )
        return summary.to_row()

    def _eval_point(self, cluster_round, theta, task, input_dim):
        from .events_consumer import ModelEvalPoint
        from .model_task import evaluate_theta

        m = evaluate_theta(theta, task.X_test, task.y_test, input_dim=input_dim)
        return ModelEvalPoint(
            cluster_round=int(cluster_round),
            accuracy=float(m["accuracy"]),
            auc=float(m["auc"]),
            loss=float(m["loss"]),
            n_test=int(len(task.y_test)),
        )

    def _build_task(self, n_devices: int, seed: int):
        from .model_task import load_ciciot_task_canonical, synthetic_task

        if self.data_source == "canonical":
            return load_ciciot_task_canonical(
                n_devices=n_devices,
                seed=seed,
                train_files=self.train_files,
                test_files=self.test_files,
                train_dataset_size=self.train_dataset_size,
                test_dataset_size=self.test_dataset_size,
                attack_eval_ratio=self.attack_eval_ratio,
            )
        if self.data_source == "synthetic":
            return synthetic_task(
                n_devices=n_devices,
                rows_per_device=self.synth_rows_per_device,
                test_rows=self.synth_test_rows,
                seed=seed,
            )
        raise ValueError(
            f"unknown data_source {self.data_source!r}; "
            f"expected 'canonical' or 'synthetic'"
        )

    def _capture_traces(self, run_dir, cell) -> None:
        """Copy this trial's raw events + configs out before teardown deletes them.

        Copies both kinds of file deliberately:

        * ``*.jsonl`` — the per-contact event stream (``device_served``,
          ``device_serve_failed``, ``mission_completed`` …) with timestamps.
        * ``*.json``  — the process configs, which carry **device positions**.
          The events do not; without positions no spatial policy (MAX-AoI's
          nearest-predecessor pathing, any travel-cost rule) can be scored.

        Never raises. Losing a trace is a nuisance; losing the trial that
        produced it because bookkeeping failed is not acceptable.
        """
        if self.trace_root is None:
            return
        try:
            dest = Path(self.trace_root) / trace_dir_name(cell)
            dest.mkdir(parents=True, exist_ok=True)
            n = 0
            for pattern in ("*.jsonl", "*.json"):
                for src in Path(run_dir).glob(pattern):
                    shutil.copy2(src, dest / src.name)
                    n += 1
            log.info(
                "exp4 trial cell=%s trial=%d arm=%s: kept %d trace file(s) in %s",
                cell.cell_id, cell.trial_index, cell.arm, n, dest,
            )
        except Exception:
            log.warning(
                "exp4 trial cell=%s trial=%d arm=%s: could not keep event "
                "traces (continuing; the trial itself is unaffected)",
                cell.cell_id, cell.trial_index, cell.arm, exc_info=True,
            )

    def _write_trial_status(
        self, cell, *, status, error, n_missions, started=None, budget_s=None,
        t_nom_computed: Optional[bool] = None, soft_cap_s: Optional[float] = None,
    ) -> None:
        """Label this trial's kept trace with the status its CSV row records.

        Called once the status is decided: ``ok`` / ``no_eval`` from the row,
        or ``error`` with the exception's last line when the trial raises —
        which is what the runner writes for a raise, an
        :class:`Exp4TrialTimeout` included. The runner's own soft timeout is
        decided after this returns, against the caller's cap. So the marker
        records what it is decided from instead: ``run_s``, the seconds since
        ``run_trial`` began (``started``; None when the trial was not started
        through it), and ``trial_budget_s``, the trial's own wall budget
        (``budget_s``, re-costed on the mission clock in FeRRy Phase 3; None
        records the driver's ``trial_budget_s``). On the wall clock that
        budget is the runner's default cap; a ``--timeout-s`` that overrides
        it is known only to the trial CSV. On the mission clock the runner's
        cap is one number for the whole grid (the largest budget, or
        ``--timeout-s``) and can exceed the trial's own budget, so
        ``soft_cap_s`` records it when the caller has told the driver (the
        driver's ``soft_cap_s``, which ``runner_main`` sets). ``run_s`` leaves
        out the teardown still to come (removing the run and prep
        directories), so it can fall a fraction of a second short of the
        runner's duration. ``t_nom_computed`` records whether the driver
        computed T_nom itself, which no trace file says otherwise: the scorer
        reads it here before inferring it. ``t_nom_computed`` and
        ``soft_cap_s`` are passed for simulated-clock trials only, and None
        leaves each key out, so wall markers keep their key set.

        Never raises, and writes nothing when there is no kept trace to label.
        """
        if self.trace_root is None:
            return
        try:
            dest = Path(self.trace_root) / trace_dir_name(cell)
            if not dest.is_dir():
                return
            run_s = None if started is None else time.monotonic() - started
            with open(dest / TRIAL_STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "status": str(status),
                        "error": str(error or ""),
                        "n_missions_target": int(n_missions),
                        "run_s": run_s,
                        "trial_budget_s": float(
                            self.trial_budget_s if budget_s is None else budget_s
                        ),
                        **({} if t_nom_computed is None
                           else {"t_nom_computed": bool(t_nom_computed)}),
                        **({} if soft_cap_s is None
                           else {"soft_cap_s": float(soft_cap_s)}),
                    },
                    f,
                )
        except Exception:
            log.warning(
                "exp4 trial cell=%s trial=%d arm=%s: could not write %s "
                "(continuing; the trial itself is unaffected)",
                cell.cell_id, cell.trial_index, cell.arm, TRIAL_STATUS_FILE,
                exc_info=True,
            )

    def _write_footprint(self, cell, footprint) -> None:
        """Write Study 5.11's footprint (``footprint.json``) beside the kept trace.

        Called once the trace is captured, a timed-out trial's included. Never
        raises: the footprint is a measurement of the trial, not part of it.
        """
        if self.trace_root is None:
            return
        try:
            dest = Path(self.trace_root) / trace_dir_name(cell)
            if dest.is_dir():
                footprint.write(dest)
        except Exception:  # noqa: BLE001
            log.warning(
                "exp4 trial cell=%s trial=%d arm=%s: could not write the footprint "
                "(continuing; the trial itself is unaffected)",
                cell.cell_id, cell.trial_index, cell.arm, exc_info=True,
            )

    def _run_topology(
        self, topo, *, cell, n_devices, rf_range_m, n_missions, started=None,
        clock: Optional[_TrialClock] = None,
    ):
        # FeRRy Phase 3: ``clock`` is the trial's resolved clock settings
        # (``_resolve_clock``); None, as for a caller that builds its own
        # topology, is the recorded wall-clock trial.
        if clock is None:
            clock = _TrialClock(
                aggregation_spec=self._aggregation_spec,
                budget_s=float(self.trial_budget_s),
                down_wait_s=self.effective_down_wait_s,
            )
        budget_s = float(self.trial_budget_s if clock.budget_s is None else clock.budget_s)
        probe = footprint = None
        if self.footprint_probe:
            # Exp 5 addendum, Study 5.11: follow every process from the instant
            # the orchestrator launches it until just before shutdown. Reads only.
            from experiments.exp4.footprint import FootprintProbe

            probe = FootprintProbe(interval_s=self.footprint_interval_s)
            orch = MultiProcessOrchestrator(topo, capture_output=True, on_spawn=probe.on_spawn)
        else:
            orch = MultiProcessOrchestrator(topo, capture_output=True)
        captured = False
        try:
            if probe is not None:
                probe.start()
            orch.start_all(timeout=self.startup_timeout_s)
            timed_out = not self._await_mules(orch, budget_s)
            # Read before shutdown, which would give any mule still running a
            # non-zero status of its own making.
            failed = {} if timed_out else self._failed_mules(orch)
            if probe is not None:
                footprint, probe = probe.stop(), None
            orch.shutdown_all(
                timeout=self.shutdown_timeout_s, cleanup_tmpdir=False,
            )
            # Capture BEFORE the timeout check below, so a timed-out trial keeps
            # its trace too — those are the runs whose events are most worth
            # having, and they are exactly the ones that would otherwise raise
            # straight past this and be deleted in the `finally`.
            self._capture_traces(orch.tmpdir, cell)
            captured = True
            if footprint is not None:
                self._write_footprint(cell, footprint)
            if timed_out:
                raise Exp4TrialTimeout(
                    f"exp4 trial exceeded {budget_s:.0f}s budget "
                    f"(cell={cell.cell_id}, trial={cell.trial_index}); "
                    f"orchestrator killed"
                )
            if failed:
                # FeRRy Phase 2: a mule that ended on an unrecovered failure
                # ran fewer missions than asked; it used to exit 0 and its
                # truncated trial was recorded as ok.
                raise Exp4MuleFailure(
                    f"mule process(es) exited non-zero {failed} "
                    f"(cell={cell.cell_id}, trial={cell.trial_index}); "
                    f"see mission_failed / dock_bootstrap_timeout in the trace"
                )
            obs = consume_run_dir(orch.tmpdir, n_devices=n_devices)
            if obs.missions_completed == 0:
                log.warning(
                    "exp4 trial produced no completed missions "
                    "(cell=%s trial=%d); mule_ready=%s dock_bootstrapped=%s "
                    "cluster_ready=%s — recording a zeroed row",
                    cell.cell_id, cell.trial_index,
                    obs.mule_ready, obs.dock_bootstrapped, obs.cluster_ready,
                )
            summary: Exp4MetricSummary = summarise_observation(
                obs,
                n_devices=n_devices,
                rf_range_m=rf_range_m,
                n_missions_target=n_missions,
                tau=self.tau,
            )
            row = summary.to_row()
            # Provenance for the two opt-in scheduler mechanisms. Recorded on
            # every row so a CSV is self-describing: without these, an
            # enforcement or window-adaptation run is indistinguishable from a
            # historical one after the fact, and the two must never be pooled.
            row["mission_budget_s"] = (
                "" if self.mission_budget_s is None else float(self.mission_budget_s)
            )
            row["mission_window_adaptation"] = int(bool(self.mission_window_adaptation))
            spec = clock.aggregation_spec or self._aggregation_spec
            row["aggregation"] = spec.rule
            row["aggregation_params"] = (
                "" if spec.is_plain else json.dumps(spec.to_params(), sort_keys=True)
            )
            row["fedprox_rho"] = float(self.fedprox_rho)
            row["pass_2_budget"] = int(bool(self.pass_2_budget))
            law = self._deadline_law
            row["deadline_law"] = law.form
            row["deadline_params"] = (
                "" if law.is_recorded
                else json.dumps(law.to_params(), sort_keys=True)
            )
            # FeRRy Phase 4: the arm's own value (a plan arm's is its own; any
            # other arm's the configured one, as recorded).
            row["miss_priority"] = int(self.effective_miss_priority(getattr(cell, "arm", "")))
            row.update(self._multi_mule_provenance(
                getattr(cell, "arm", ""), down_wait_s=clock.down_wait_s,
                mule=topo.mules[0] if topo.mules else None,
            ))
            row.update(self._clock_provenance(topo, clock))
            # A trial that produced NO model evaluation at all never trained a
            # model — its convergence columns are blank while its federation
            # columns are hard zeros. Recorded as `ok`, that asymmetry biases
            # the analysis: the blank AUC is dropped by `.dropna()` while the
            # 0.0 participation is averaged as if it were a real observation
            # (it flipped the sign of several H3−H2 differences in the first
            # H2/H3 dead-zone sweep). Mark it so `status == "ok"` filters
            # exclude it from EVERY metric, not just the ones that are blank.
            if self.real_model and int(summary.rounds_evaluated) == 0:
                row["status"] = "no_eval"
                row["error"] = (
                    f"trial produced no model_evaluation events "
                    f"(mule_ready={obs.mule_ready} "
                    f"dock_bootstrapped={obs.dock_bootstrapped} "
                    f"cluster_ready={obs.cluster_ready}); not a valid trial"
                )
                log.warning(
                    "exp4 trial cell=%s trial=%d arm=%s: no model evaluations "
                    "— recording status=no_eval (excluded from analysis)",
                    cell.cell_id, cell.trial_index, cell.arm,
                )
            self._write_trial_status(
                cell, status=row.get("status", "ok"), error=row.get("error", ""),
                n_missions=n_missions, started=started, budget_s=budget_s,
                t_nom_computed=(clock.t_nom_computed
                                if clock is not None and clock.sim else None),
                soft_cap_s=(self.soft_cap_s
                            if clock is not None and clock.sim else None),
            )
            return row
        except Exception as exc:
            # The runner records any raise, the timeout above included, as
            # status=error with the exception's last line; say the same beside
            # the trace so it is not scored as a good trial.
            if captured:
                self._write_trial_status(
                    cell, status="error",
                    error=traceback.format_exception_only(type(exc), exc)[-1].strip(),
                    n_missions=n_missions, started=started, budget_s=budget_s,
                    t_nom_computed=(clock.t_nom_computed
                                    if clock is not None and clock.sim else None),
                    soft_cap_s=(self.soft_cap_s
                                if clock is not None and clock.sim else None),
                )
            raise
        finally:
            if probe is not None:
                # A raise between start and stop: end the thread, keep nothing.
                try:
                    probe.stop()
                except Exception:  # noqa: BLE001 - never mask the trial's own error
                    log.warning("exp4 trial cell=%s: footprint probe did not stop cleanly",
                                cell.cell_id, exc_info=True)
            orch.cleanup()

    def _multi_mule_provenance(
        self, arm: str, *, down_wait_s: Any = "default", mule: Any = None,
    ) -> Dict[str, Any]:
        """The FeRRy Phase 2 provenance columns for a row of ``arm``.

        ``n_mules`` is always the count, as the trace scorer reports it. The
        others are blank at their recorded value — a quorum of 1 with one
        mule, the recorded dock, no policy options — so a single-mule row
        reads as it always did. ``down_wait_s`` is the trial's own DOWN wait
        (FeRRy Phase 3: the re-costed wall budget with several mules on the
        mission clock); by default the driver's. FeRRy Phase 5: ``mule``, the
        trial's mule config, adds arm E3's checkpoint to ``policy_params``
        (:func:`learned_policy_params`, read off the config as the scorer
        reads it off the per-role JSON); nothing for any other mule.
        """
        recorded_topology = int(self.n_mules) == 1 and int(self.min_participation) == 1
        dock_on_empty = self.effective_dock_on_empty
        if down_wait_s == "default":
            down_wait_s = self.effective_down_wait_s
        policy = dict(self._policy_params(arm))
        if mule is not None:
            policy.update(learned_policy_params(asdict(mule)))
        return {
            "n_mules": int(self.n_mules),
            "min_participation": "" if recorded_topology else int(self.min_participation),
            "dock_params": (
                "" if not dock_on_empty and down_wait_s is None
                else json.dumps(
                    {"dock_on_empty": bool(dock_on_empty), "down_wait_s": down_wait_s},
                    sort_keys=True,
                )
            ),
            "policy_params": json.dumps(policy, sort_keys=True) if policy else "",
        }

    def _clock_provenance(self, topo, clock: _TrialClock) -> Dict[str, Any]:
        """The FeRRy Phase 3 provenance columns (``PROVENANCE_COLUMNS``' last 13).

        Like the Phase 2 columns, each is BLANK at the driver's defaults, so a
        row at the default settings reads as it always did plus blanks, and
        a trace kept before these columns existed derives each one (critic
        B13; such traces are all wall-clock). The clock does not blank them:
        ``l1_channel``, ``realism`` and ``input_dim`` describe the recorded
        cells too (a re-run of a recorded real-model cell writes realism 1
        and input_dim 21), and a numeric time unit, Φ₀ or TTL fills its column
        on either clock. Formats, and the derivation from an old trace:

        * ``mission_clock``: "sim", "" on the wall clock. Old: "".
        * ``contact_band``: the class name, "" without a band. Old: "".
        * ``in_flight_response``: "replan", "" for "abort". Old: "".
        * ``backhaul_model``: "seconds", "" for "mission". Old: "".
        * ``contact_reliability_source``: "channel", "" for "origin". Old: "".
        * ``deadline_time_scale``: the float the scheduler ran (T_nom / 10 s
          resolved), "" at 1.0. Old: "".
        * ``initial_window_s``: Φ₀ in the law's recorded unit as the mule got
          it (float), "" when unset. Old: "".
        * ``t_nom_s``: the cell's T_nom in simulated seconds (float), "" when
          none was given or needed. Old: "".
        * ``session_ttl_s``: the mule's wall TTL (float), "" at the recorded
          3.0 s. Old: "" when the mule JSON's ``session_ttl_s`` is 3.0 (every
          kept Exp 4 trace), else that value.
        * ``ferry_params``: JSON (sorted keys) of every other ferry setting the
          mules ran, as ``MuleConfig`` fields, resolved (``backhaul_period_s``
          is P_bh when the seconds model computes it from T_nom), plus
          ``t_nom_computed``; "" on the wall clock. Old: "".
        * ``l1_channel``: 1, "" when off. Old: 1 iff the cluster JSON has a
          non-null ``backhaul_loss_schedule``.
        * ``realism``: 1, "" when off. Old: 1 iff a device JSON has a non-null
          ``contact_reliability``. (A ferry trace under the channel source
          keeps the draw in the mule JSON's ``device_availability`` instead.)
        * ``input_dim``: the model's input width (int), "" on the stub. Old:
          the cluster JSON's ``input_dim`` ("" when null).

        FeRRy Phase 4 (other choices 12; unit_U3b.md section 5.4), both read
        off the mule config so the trace scorer derives the same strings from
        its JSON (:func:`contact_band_column`, :func:`plan_ferry_params`):
        ``contact_band`` reads ``search`` for a plan arm that searches the band
        classes (FB+<class> keeps its class), and ``ferry_params`` gains the
        plan fields in plan mode, or an H or D arm's ``member_admission`` when
        it is ``subset``. Neither changes a row at the defaults. No column is
        added, so the trial CSV header is the Phase 3 one.
        """
        switches = dict(clock.settings)
        mule = topo.mules[0] if topo.mules else None

        def _unless(value, recorded):
            return "" if value is None or value == recorded else value

        ttl = float(mule.session_ttl_s if mule is not None else self.effective_session_ttl_s)
        row: Dict[str, Any] = {
            "mission_clock": "sim" if clock.sim else "",
            "contact_band": switches.get("contact_band") or "",
            "in_flight_response": _unless(switches.get("in_flight_response"), "abort"),
            "backhaul_model": _unless(switches.get("backhaul_model"), "mission"),
            "contact_reliability_source": _unless(
                switches.get("contact_reliability_source"), "origin",
            ),
            "deadline_time_scale": _unless(float(clock.deadline_time_scale), 1.0),
            "initial_window_s": (
                "" if clock.initial_window_s is None else float(clock.initial_window_s)
            ),
            "t_nom_s": "" if clock.t_nom_s is None else float(clock.t_nom_s),
            "session_ttl_s": _unless(ttl, float(SESSION_TTL_S)),
            "ferry_params": "",
            "l1_channel": 1 if self.l1_channel else "",
            "realism": 1 if self.realism else "",
            "input_dim": "" if clock.input_dim is None else int(clock.input_dim),
        }
        if clock.sim and mule is not None:
            from hermes.processes.config import FERRY_PARAMS_OMITTED_AT_NONE, FERRY_SPEC_FIELDS

            shown = {
                name: getattr(mule, name) for name in FERRY_SPEC_FIELDS
                if name not in ("contact_band", "in_flight_response", "backhaul_model",
                                "contact_reliability_source", "device_availability",
                                "t_nom_s")
                and not (name in FERRY_PARAMS_OMITTED_AT_NONE and getattr(mule, name) is None)
            }
            if (mule.backhaul_model == "seconds" and mule.backhaul_period_s is None
                    and clock.t_nom_s is not None):
                from hermes.l1.channel_model import backhaul_period_s

                # P_bh as the mule's spec computes it: n_missions * T_nom.
                shown["backhaul_period_s"] = backhaul_period_s(
                    int(mule.n_missions), float(clock.t_nom_s),
                )
            shown["t_nom_computed"] = bool(clock.t_nom_computed)
            fields = asdict(mule)
            shown.update(plan_ferry_params(fields))
            row["ferry_params"] = json.dumps(shown, sort_keys=True, default=str)
            if getattr(mule, "plan_mode", "legacy") == "ferry":
                row["contact_band"] = contact_band_column(fields)
        return row

    @staticmethod
    def _failed_mules(orch: MultiProcessOrchestrator) -> Dict[str, int]:
        """Mules that have exited with a non-zero status: mule id -> status."""
        out: Dict[str, int] = {}
        for mule_id, handle in orch.mule_handles.items():
            rc = handle.returncode()
            if rc not in (None, 0):
                out[str(mule_id)] = int(rc)
        return out

    def _await_mules(
        self, orch: MultiProcessOrchestrator, budget_s: float,
    ) -> bool:
        """Block until every mule subprocess exits, or the budget runs out.

        Returns True if all mules exited within budget, False on timeout.
        """
        deadline = time.monotonic() + budget_s
        for handle in orch.mule_handles.values():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                handle.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                return False
        return True
