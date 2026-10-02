"""Sprint 2 — mule-process entry point + service loop.

Run with::

    python -m hermes.processes.mule --config /path/to/mule.json

The mule process:

1. Reads :class:`MuleConfig` from JSON.
2. Stands up :class:`TCPRFLinkServer` on the mule's RF port.
3. Stands up :class:`TCPDockLinkClient` connecting to the cluster's
   dock port.
4. Builds :class:`MuleSupervisor` wiring scheduler + mission server +
   client_cluster.
5. Waits for expected devices to register on the RF link.
6. Calls :meth:`MuleSupervisor.wait_for_initial_dock` (consumes the
   cluster's bootstrap DOWN).
7. Loops :meth:`MuleSupervisor.run_one_mission` for ``n_missions``
   iterations (or until shutdown).

Logs go to stderr in plain text (chunk M wraps them in JSON later).

FeRRy Phase 3 (``MuleConfig.mission_clock == "sim"``, design section 2.2).
The process builds one ``MissionClock`` and the ``FerrySpec`` its config
describes (:func:`ferry_spec_from_config`) and hands both to the supervisor,
which then flies every mission on simulated time. Its events gain the
simulated fields of design section 2.5, and only on that clock:
``mule_ready`` the clock and every ferry setting (``FerrySpec.describe``),
``mission_started`` ``sim_start_s`` and the RF prior the mission plans with,
and ``mission_completed`` the mission's simulated record
(``MissionRunResult``'s ``sim_*`` fields, the stops flown, re-plans, the
simulated energy, the backhaul upload and the Pass-1 pre-flight drops).
Under the recorded ``mission`` backhaul model with the L1 channel, the process
feeds the planner's RF prior causally from the missions already uploaded
(``MuleConfig.rf_prior_schedule_db``, critic B4). A bootstrap DOWN the mule
must refuse there (its slice carries wall-clock deadline overrides) ends the
process with ``EXIT_BOOTSTRAP_FAILED``.
On the wall clock every build step and event is the recorded one; a deadline
time unit other than the recorded one (valid on either clock) adds its three
fields to ``mule_ready``.

FeRRy Phase 4 (the Phase 4 spec, other choices 10 and 12), on the simulated
clock only. ``MuleConfig.plan_mode == "ferry"`` hands the supervisor the plan
options its config describes (``hermes.scheduler.plan.PlanOptions``, imported
in plan mode only), ``t_nom_s`` and ``member_admission``; an H or D arm's
``member_admission`` reaches the supervisor only when it is not the recorded
``whole``. ``mule_ready`` then states the plan settings the scheduler runs
(read back from it, audit #15), and ``mission_completed`` carries the
mission's closed plan (``plan``, ``PlanCommit.describe()``), the plan's wall
time beside it (``plan_wall_s``, left out of every determinism comparison,
critic B12) and a baseline's pre-flight drops (``pass_1_policy_drops``, the
user's decision 6), each only when the mission has one, so every other trace
keeps its key set (critic D2).

FeRRy Phase 5 (the Phase 5 spec, other choices 5, 6 and 9), on the simulated
clock only. A learned filling flies a verified checkpoint and nothing else
(there is no random-init arm): ``flight_slot="pair_q"`` hands the supervisor
the pair slot built around the score of ``pair_checkpoint``
(``selector.pair_features.build_pair_slot``, over the link's classes), and
``contact_policy="chen_dqn"`` builds arm E3 from ``policy_checkpoint``
(``policies.chen_dqn.ChenDQNPolicy.from_checkpoint``). Both are read before
the process binds anything, and a checkpoint whose arrays are not the sha its
config names, or whose kind, schema or classes are not what the arm reads,
stops the mule there (:class:`CheckpointRefused`). A relative checkpoint path
is read under the repository root (:data:`REPO_ROOT`), where the Exp 4 driver
writes one that lies inside it. ``mule_ready`` then announces the checkpoint
flown, ``pair`` or ``policy_checkpoint`` (its verified manifest's provenance
and the config's tag, no path), and ``mission_completed`` carries the
mission's decisions, ``pass_1_pairs`` or E3's ``pass_1_e3`` and
``pass_1_e3_unvisited``, each only on those mules and only when it has one,
so every other trace keeps its key set (Freeze Rule 1).
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

from hermes.mule import MuleSupervisor, MuleSupervisorError
from hermes.observability import (
    JsonEventEmitter,
    MetricsRegistry,
    NullEventEmitter,
)
from hermes.transport import TCPDockLinkClient, TCPRFLinkServer
from hermes.types import DeviceID, MuleID
from hermes.types.scheduler import DEFAULT_FULFILMENT_WINDOW_S

from .config import (
    CLOCK_SIM,
    CONTACT_POLICY_CHEN_DQN,
    FLIGHT_SLOT_PAIR_Q,
    MEMBER_ADMISSION_WHOLE,
    PLAN_MODE_FERRY,
    PLAN_MODE_LEGACY,
    PLAN_OPTION_FIELDS,
    MuleConfig,
    mission_schedule_index,
    mule_config_errors,
    mule_config_from_json,
)

log = logging.getLogger("hermes.processes.mule")

#: Exit status of a mule whose mission loop ended on an unrecovered failure
#: (``mission_failed``), so a driver cannot mistake a truncated run for a
#: finished one. 1 stays Python's own status for an uncaught exception.
EXIT_MISSION_FAILED = 3
#: Exit status of a mule that never received its bootstrap DOWN.
EXIT_BOOTSTRAP_TIMEOUT = 4
#: Exit status of a mule on the simulated clock that refused its bootstrap
#: DOWN (FeRRy Phase 3: a slice carrying wall-clock deadline overrides, or a
#: ``cluster_sim_ts`` that is no simulated time). Fatal: the mule would fly
#: with no slice.
EXIT_BOOTSTRAP_FAILED = 5

#: Where every mule starts and docks: the origin, the pose ``MuleSupervisor``
#: starts from. D4's FedEx tour returns here. FeRRy Phase 3: re-exported from
#: the mission clock's module, the one definition (design section 2.1).
from hermes.l1.mission_clock import DOCK_POSE  # noqa: E402

#: FeRRy Phase 5 — the repository root, the directory that holds ``hermes``. A
#: relative checkpoint path is read under it (:func:`checkpoint_path`), not under
#: the working directory, which an orchestrated mule inherits from whatever runs
#: the trial: the Exp 4 driver writes the path of a checkpoint that lies inside
#: the repository relative to this root, so a kept per-role JSON names no host
#: directory, and any other one absolute, not "as given" as the Phase 5 spec's
#: other choices 6 says, since a relative one read here would name another file.
REPO_ROOT = Path(__file__).resolve().parents[2]


class CheckpointRefused(ValueError):
    """A learned filling's checkpoint the mule refuses to fly (FeRRy Phase 5).

    The mule flies a checkpoint only once its arrays match the sha its config
    names and its kind, schema and classes are what the arm reads (other
    choices 6), so anything else stops the process before it binds a port, as
    a config the guards refuse does.
    """


def checkpoint_path(path: Any) -> Path:
    """Where the checkpoint a config names lies: an absolute path as it is, a
    relative one under :data:`REPO_ROOT`, whatever the working directory.

    Raises ValueError for no path at all, since a learned filling flies a
    verified checkpoint, never a random one (no random-init arm).
    """
    if path is None or (isinstance(path, str) and not path.strip()):
        raise ValueError("no checkpoint path: a learned filling flies a verified checkpoint, "
                         "never a random one")
    out = Path(path)
    return out if out.is_absolute() else REPO_ROOT / out


@contextlib.contextmanager
def _refused_checkpoint(cfg: MuleConfig, name: str) -> Iterator[None]:
    """A loader's refusal of the checkpoint ``cfg.<name>``, raised as the mule's.

    The loaders raise ``pair_q.CheckpointError`` (a ValueError) for a file they
    refuse, FileNotFoundError for a path with nothing behind it and a plain
    ValueError for one that is no ``.npz``: to the mule each is a checkpoint it
    does not fly, refused with its config field named.
    """
    try:
        yield
    except (ValueError, OSError) as exc:
        raise CheckpointRefused(
            f"mule {cfg.mule_id}: {name}={getattr(cfg, name, None)!r} is refused, and the "
            f"mule flies a verified checkpoint only: {exc}"
        ) from exc


def _checkpoint_provenance(manifest: Any, tag: Any) -> dict:
    """What ``mule_ready`` announces of a checkpoint flown (FeRRy Phase 5).

    The verified manifest's provenance (``pair_q.manifest_provenance``: the
    sha, kind and purpose, the schema's version and the classes, γ, the
    reward, the seeds, the episodes, FerrySim's cell family with its hash and
    the learner's revision; the sha binds the first six) and the config's
    tag. No path: provenance is by tag and sha (other choices 6).
    """
    from hermes.scheduler.selector.pair_q import manifest_provenance

    return _jsonable(dict(manifest_provenance(manifest), tag=str(tag)))


def _build_pair_slot(cfg: MuleConfig, spec):
    """The pair slot of a ``flight_slot="pair_q"`` mule in plan mode; None for any other.

    FeRRy Phase 5 (resolution R8; other choices 5 and 6). The slot ranks the
    pairs with the learned score of the config's checkpoint, verified whole
    against ``pair_checkpoint_sha256`` and read under its own schema over the
    link's classes in link order (``selector.pair_features.build_pair_slot``),
    so it never flies a random network, nor rows other than those it was
    trained on. ``spec`` is the mule's ``FerrySpec``, whose link names the
    classes. The pair modules are imported on this path only.
    """
    if (getattr(cfg, "plan_mode", PLAN_MODE_LEGACY) != PLAN_MODE_FERRY
            or getattr(cfg, "flight_slot", None) != FLIGHT_SLOT_PAIR_Q):
        return None
    from hermes.scheduler.selector.pair_features import build_pair_slot

    with _refused_checkpoint(cfg, "pair_checkpoint"):
        return build_pair_slot(
            checkpoint_path(cfg.pair_checkpoint),
            expect_sha256=cfg.pair_checkpoint_sha256,
            classes=spec.link.names,
        )


def _learned_fillings(cfg: MuleConfig, spec) -> Tuple[Any, Any]:
    """(pair slot, E3 policy) a mule flies from its checkpoints, each None if it has none.

    FeRRy Phase 5: the mule reads both before it binds anything, so a refused
    checkpoint stops it at once (:class:`CheckpointRefused`). ``spec`` is the
    mule's ``FerrySpec``, None on the wall clock, where the guards refuse both
    fillings.
    """
    if spec is None:
        return None, None
    policy = None
    if getattr(cfg, "contact_policy", None) == CONTACT_POLICY_CHEN_DQN:
        policy = _build_target_selector(cfg)
    return _build_pair_slot(cfg, spec), policy


def _build_target_selector(cfg: MuleConfig):
    """EX-4.2 — build the S3.5 RL target selector when configured (arm H2).

    Returns ``None`` (deterministic distance ranking = arm H1) unless
    ``cfg.use_rl_selector`` is set. With ``selector_weights_path`` it loads a
    trained DDQN (.npz); otherwise it builds a random-init selector — a
    plumbing smoke, NOT paper-grade (train weights via
    ``experiments.exp3.train_a4`` for a real H2-vs-H1 comparison).
    """
    # SOTA baseline arm — MAX-AoI greedy. Checked first: it is an alternative
    # POLICY, so it replaces the ranking entirely rather than composing with the
    # RL selector. Both occupying one slot is a configuration error, not a blend.
    policy = getattr(cfg, "contact_policy", None)
    if policy:
        if policy == "max_aoi":
            from hermes.scheduler.policies import MaxAoIPolicy
            log.info("mule %s: MAX-AoI baseline policy (SOTA comparator)",
                     cfg.mule_id)
            return MaxAoIPolicy()
        if policy == "oort":
            from hermes.scheduler.policies import OortPolicy
            log.info("mule %s: Oort statistical-utility baseline policy "
                     "(SOTA comparator; requires --real-model)", cfg.mule_id)
            return OortPolicy()
        # FeRRy Phase 2 — arms D3-D5, whole schedulers like D1/D2.
        if policy == "whittle":
            from hermes.scheduler.policies import WhittlePolicy
            variant = getattr(cfg, "whittle_variant", "expected")
            weights = getattr(cfg, "whittle_weights", "uniform")
            log.info("mule %s: Whittle-index baseline policy (Cui et al.; "
                     "variant=%s, weights=%s)", cfg.mule_id, variant, weights)
            return WhittlePolicy(variant=variant, weights=weights)
        if policy == "fedex":
            from hermes.scheduler.policies import FedExCarpPolicy
            # The dock is the mule's start pose, so the tour is the shortest
            # path from wherever the mule is through every contact and home
            # (the policy's faithful depot mode, its deviation 13).
            log.info("mule %s: FedEx-Async/CARP tour baseline policy "
                     "(visit every contact, depot %s)", cfg.mule_id, DOCK_POSE)
            return FedExCarpPolicy(depot=DOCK_POSE)
        if policy == "fedcs":
            from hermes.scheduler.policies import FedCSDegradedPolicy
            value = getattr(cfg, "fedcs_value", "unit")
            log.info("mule %s: FedCS (degraded) greedy baseline policy "
                     "(value=%s)", cfg.mule_id, value)
            return FedCSDegradedPolicy(value=value)
        # FeRRy Phase 5 — arm E3, after Chen et al. (decision 7 (a)): a whole
        # scheduler that names each next stop in flight, flown from its verified
        # checkpoint only (on the cell's contact band, the checkpoint's class).
        # Its module is imported here and nowhere else on a mule's path.
        if policy == CONTACT_POLICY_CHEN_DQN:
            from hermes.scheduler.policies.chen_dqn import ChenDQNPolicy

            log.info("mule %s: E3 baseline policy (Chen et al.'s DQN, numpy port) from "
                     "checkpoint tag %s", cfg.mule_id, getattr(cfg, "policy_checkpoint_tag", None))
            with _refused_checkpoint(cfg, "policy_checkpoint"):
                return ChenDQNPolicy.from_checkpoint(
                    checkpoint_path(getattr(cfg, "policy_checkpoint", None)),
                    expect_sha256=cfg.policy_checkpoint_sha256,
                    band=cfg.contact_band,
                )
        raise ValueError(
            f"unknown contact_policy {policy!r}; expected 'max_aoi', 'oort', "
            f"'whittle', 'fedex', 'fedcs', 'chen_dqn' or None"
        )

    if not getattr(cfg, "use_rl_selector", False):
        return None
    from hermes.scheduler.selector import TargetSelectorRL

    path = getattr(cfg, "selector_weights_path", None)
    if path:
        from hermes.scheduler.selector.ddqn import DDQN
        log.info("mule %s: RL target selector from trained weights %s",
                 cfg.mule_id, path)
        return TargetSelectorRL(ddqn=DDQN.load(path), epsilon=0.0)
    log.warning(
        "mule %s: RL target selector with RANDOM-INIT weights (H2 plumbing "
        "smoke only — not paper-grade)", cfg.mule_id,
    )
    return TargetSelectorRL(epsilon=0.0, rng_seed=0)


def _pass_1_plan_payload(pass_1_queue, device_deadlines=None) -> List[dict]:
    """The committed Pass-1 plan for ``mission_completed``: each contact's
    devices and the deadline it was admitted under (its tightest member's).

    ``device_deadlines`` adds each member's own Deadline(j), so a miss can be
    scored against the device's deadline rather than its contact's tightest
    (audit #13). Omitted when the plan did not record them.
    """
    out: List[dict] = []
    for wp in pass_1_queue:
        entry = {
            "devices": [str(d) for d in wp.devices],
            "deadline_ts": float(wp.deadline_ts),
        }
        if device_deadlines:
            own = {
                str(d): float(device_deadlines[d])
                for d in wp.devices if d in device_deadlines
            }
            if own:
                entry["device_deadlines"] = own
        out.append(entry)
    return out


def _pass_1_collected(result) -> Tuple[Optional[int], Optional[List[str]]]:
    """``(pass_1_updates, pass_1_clean_devices)``: the CLEAN sessions collected.

    A mission whose every update was past its age cutoff has no merge report
    but kept its ledger (``unmerged_report``): its sessions still completed,
    so they count as collections here, while the merged fields say nothing
    reached the model. ``unmerged_report`` is None under agg:plain, whose
    empty missions stay exactly as recorded (both None).
    """
    report = getattr(result, "report", None) or getattr(result, "unmerged_report", None)
    if report is None:
        return None, None
    clean = [str(line.device_id) for line in report.lines if line.outcome.is_on_time()]
    return report.counts()[0], clean


def _pass_1_merged_devices(result) -> Optional[List[str]]:
    """Pass-1 devices whose update the mule's merge actually used.

    The CLEAN sessions minus the updates the age cutoff excluded; built by
    subtraction so agg:plain, which excludes nothing, reports exactly its CLEAN
    list (audit #3). An empty mission merged nothing. None when the mission
    has no round report and is not empty (a trace field the path never had).
    """
    if getattr(result, "empty", False):
        return []
    report = getattr(result, "report", None)
    if report is None:
        return None
    agg = getattr(result, "aggregate", None)
    excluded = {str(d) for d in getattr(agg, "excluded_devices", ())}
    return [
        str(line.device_id)
        for line in report.lines
        if line.outcome.is_on_time() and str(line.device_id) not in excluded
    ]


def _pass_1_outcomes_payload(result) -> Optional[List[dict]]:
    """Every Pass-1 session the mule recorded: device, outcome, contact time.

    An empty mission has no round report (nothing reached the mule), so it
    records an empty list rather than None: the trace scorer reads None as
    "a trace from before these fields existed". The exception is a mission
    whose sessions completed but whose every update was past its age cutoff:
    its ledger is kept as ``unmerged_report`` and recorded here, so those
    on-time sessions are not scored as deadline misses.
    """
    report = getattr(result, "report", None)
    if report is None:
        report = getattr(result, "unmerged_report", None)
    if report is None:
        return [] if getattr(result, "empty", False) else None
    return [
        {
            "device": str(line.device_id),
            "outcome": line.outcome.value,
            "contact_ts": float(line.contact_ts),
            # FeRRy Phase 1: the collected update's basis and age in cluster
            # rounds (None when the session collected nothing).
            "basis_version": getattr(line, "basis_version", None),
            "age": getattr(line, "age", None),
        }
        for line in report.lines
    ]


def _pass_1_merge_payload(result) -> Optional[dict]:
    """How the mule merged this mission's updates (FeRRy Phase 1).

    The rule, the version of the θ the mule carried, and per merged device its
    age and, under the age-aware rules, its weight share w_i / M_m, where M_m
    is the staleness-free mass; the shares sum to the mass-weighted mean
    staleness (1 only when no update is discounted). agg:plain records no
    shares, so ``weights`` is empty there. ``excluded`` lists updates past
    their cutoff.
    None for an empty mission.
    """
    agg = getattr(result, "aggregate", None)
    if agg is None:
        return None
    return {
        "rule": getattr(agg, "rule", "agg:plain"),
        "base_version": getattr(agg, "base_version", None),
        "devices": [str(d) for d in agg.contributing_devices],
        "ages": list(getattr(agg, "device_ages", ())),
        "weights": [float(w) for w in getattr(agg, "device_weights", ())],
        "excluded": [str(d) for d in getattr(agg, "excluded_devices", ())],
    }


def _deadline_state_payload(device_states) -> dict:
    """Each device's window Φ and miss streak after the mission (FeRRy
    Phase 1), so the deadline law's trajectory can be read from the trace."""
    return {
        str(did): {
            "phi_s": round(float(st.deadline_fulfilment_s), 6),
            "miss_streak": int(getattr(st, "miss_streak", 0)),
        }
        for did, st in device_states.items()
    }


def _pass_2_skipped(result) -> Optional[int]:
    """Devices a budgeted Pass 2 did not fly to; None without a Pass 2."""
    report = getattr(result, "delivery_report", None)
    if report is None:
        return None
    return sum(1 for line in report.lines if line.outcome.value == "skipped")


# --------------------------------------------------------------------------- #
# FeRRy Phase 3 — the mission clock
# --------------------------------------------------------------------------- #

def ferry_spec_from_config(cfg: MuleConfig):
    """The ``FerrySpec`` a sim-clock mule flies with (hermes/mule/ferry.py).

    Built from :meth:`MuleConfig.ferry_spec_kwargs`, the one mapping the Exp 4
    driver also uses, so the process prices exactly what the driver planned
    T_nom and the D4 split with. Raises ``ValueError``/``TypeError`` on a
    value the spec refuses.
    """
    from hermes.mule.ferry import FerrySpec

    return FerrySpec.from_config(**cfg.ferry_spec_kwargs())


#: ``mule_ready.rf_prior_source`` on the mission clock: where the planner's
#: RF prior (the S3.5 selector's ``rf_prior_snr_db``) comes from (critic B4).
#: ``seconds_backhaul``: the SNR last observed on the held carrier at an
#: upload (the seconds model's producer, hermes/l1/rf_prior.py);
#: ``mission_schedule``: the L1 trace's SNR at each past upload under the
#: recorded ``mission`` model (``MuleConfig.rf_prior_schedule_db``);
#: ``constant``: neither, the configured ``rf_prior_snr_db`` (20 dB default)
#: throughout. The non-causal trial mean is never one of them.
RF_PRIOR_SECONDS_BACKHAUL = "seconds_backhaul"
RF_PRIOR_MISSION_SCHEDULE = "mission_schedule"
RF_PRIOR_CONSTANT = "constant"

#: ``MissionRunResult`` fields ``mission_completed`` carries on the mission
#: clock (design section 2.5), in this order; all None on the wall clock.
#: ``pass_1_preflight_drops`` is the diagnostic of finding E2E1-01.
SIM_MISSION_FIELDS = (
    "sim_start_s", "sim_end_s", "sim_ledger", "sim_pass_2_start_s",
    "pass_1_flown", "pass_2_flown", "replans", "aborts", "inserts",
    "offers_refused", "budget_overrun_s", "pass_2_budget_overrun_s",
    "energy_j", "band", "backhaul", "pass_1_preflight_drops",
)

#: Carried after them only when the result sets it (not None): the route-level
#: ``deadline_bounds="delivery"`` mission's ``delivery_overrun_s``. Left out
#: otherwise, so a mission under any other value keeps the key set above and
#: its trace byte for byte (Freeze Rule 1). FeRRy Phase 4 (other choices 12)
#: adds three the same way: a plan-mode mission's ``plan`` (its closed
#: ``PlanCommit.describe()``, which holds no wall time) and ``plan_wall_s``
#: (the plan's wall time, kept outside ``plan`` so determinism comparisons
#: drop it, critic B12), and a baseline's ``pass_1_policy_drops`` (decision 6),
#: left out when empty as well (critic D2). FeRRy Phase 5 (other choices 9)
#: adds three after them, each left out when None or empty: a ``pair_q``
#: mission's closed decision records (``pass_1_pairs``), and arm E3's record of
#: each next-stop call (``pass_1_e3``) and of the stops its pass left
#: (``pass_1_e3_unvisited``), so only those missions gain a key. The Exp 5
#: addendum (Study 5.11 (a)) adds two after them the same way: the wall time of
#: each pair decision (``pass_1_pairs_wall``) and of each of E3's calls
#: (``pass_1_e3_wall``), kept outside the records so that determinism
#: comparisons drop them as they drop ``plan_wall_s``.
SIM_MISSION_OPTIONAL_FIELDS = (
    "delivery_overrun_s", "plan", "plan_wall_s", "pass_1_policy_drops",
    "pass_1_pairs", "pass_1_e3", "pass_1_e3_unvisited",
    "pass_1_pairs_wall", "pass_1_e3_wall",
)

#: Optional fields left out when empty too, not only when None.
_OMITTED_WHEN_EMPTY = frozenset({
    "pass_1_policy_drops", "pass_1_pairs", "pass_1_e3", "pass_1_e3_unvisited",
    "pass_1_pairs_wall", "pass_1_e3_wall",
})


def _jsonable(value):
    """``value`` with every container a JSON list/dict and every number a Python one.

    The emitter serialises outside its own error handling, so one numpy
    scalar in a nested record would raise into the mission loop.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "item"):          # numpy scalars
        return _jsonable(value.item())
    return value


def _sim_mission_fields(result) -> dict:
    """``mission_completed``'s simulated fields from a sim-clock ``MissionRunResult``.

    Times are simulated seconds; ``energy_j`` is SIMULATED (the Zeng 2019
    model, ``energy_status``). ``pass_1_plan[].deadline_ts`` and
    ``pass_1_outcomes[].contact_ts`` keep their names and are simulated too,
    as ``mule_ready.mission_clock`` says.
    """
    out = {name: _jsonable(getattr(result, name, None)) for name in SIM_MISSION_FIELDS}
    for name in SIM_MISSION_OPTIONAL_FIELDS:
        value = getattr(result, name, None)
        if value is None or (name in _OMITTED_WHEN_EMPTY and not value):
            continue
        out[name] = _jsonable(value)
    out["energy_status"] = "simulated"
    return out


class MuleService:
    """Lifecycle holder for a mule-process service loop."""

    def __init__(
        self,
        cfg: MuleConfig,
        *,
        events: Optional[JsonEventEmitter] = None,
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        self.cfg = cfg
        self._stop_event = threading.Event()
        #: The process exit status ``main`` returns: 0 unless the service loop
        #: ended on an unrecovered failure (``EXIT_*``).
        self.exit_code = 0

        self.events = events or NullEventEmitter(role="mule", node_id=cfg.mule_id)
        self.metrics = metrics or MetricsRegistry()

        # FeRRy Phase 3 — refuse clock settings that cannot run (critic B16)
        # before binding anything; the recorded defaults pass. On the
        # simulated clock the FerrySpec is built (and so fully validated) here.
        errors = mule_config_errors(cfg)
        if errors:
            raise ValueError(f"mule {cfg.mule_id} config: " + "; ".join(errors))
        self._sim = getattr(cfg, "mission_clock", "wall") == CLOCK_SIM
        ferry_spec = ferry_spec_from_config(cfg) if self._sim else None
        #: Critic B4: the recorded ``mission`` backhaul model's causal RF prior,
        #: one SNR per mission round (:meth:`_feed_rf_prior`); None without one.
        schedule = getattr(cfg, "rf_prior_schedule_db", None)
        self._rf_prior_schedule: Optional[List[float]] = (
            [float(v) for v in schedule] if self._sim and schedule is not None else None
        )
        # FeRRy Phase 5 — a learned filling's verified checkpoint, read here,
        # before anything is bound, so a refused one (CheckpointRefused) stops
        # the mule at once. Both None on every other mule.
        pair_slot, e3_policy = _learned_fillings(cfg, ferry_spec)

        # 1. Mule's RF server — devices connect here. Amendment 10: with a
        # token (one per trial) it refuses registrations carrying another.
        rf_kwargs = {}
        token = getattr(cfg, "rf_link_token", None)
        if token is not None:
            rf_kwargs["link_token"] = str(token)
        self.rf = TCPRFLinkServer(host=cfg.rf_host, port=cfg.rf_port, **rf_kwargs)
        self.rf.start()
        self.actual_rf_port = self.rf.port

        # 2. Mule's dock client — connects outbound to the cluster.
        self.dock = TCPDockLinkClient(
            mule_id=MuleID(cfg.mule_id),
            host=cfg.dock_host,
            port=cfg.dock_port,
        )

        # 3. Supervisor (Sprint 1.5 two-pass when rf_range_m is set).
        # EX-4.2 arm H2: an RL target selector is injected when configured;
        # otherwise the supervisor uses deterministic distance ranking (H1).
        # EX-4.3 arm H3: a real L1 RF prior (mean SNR of the chosen channel)
        # feeds the selector instead of the hardcoded 20.0 default.
        sup_kwargs = {}
        rf_prior = getattr(cfg, "rf_prior_snr_db", None)
        if rf_prior is not None:
            sup_kwargs["rf_prior_snr_db"] = float(rf_prior)
        budget = getattr(cfg, "mission_budget_s", None)
        if budget is not None:
            sup_kwargs["mission_budget_s"] = float(budget)
        # S3c — build the adapter here rather than in the config, because it
        # carries live per-mission history and this config crosses a process
        # boundary. Only attached when the toggle is on, so the default path is
        # byte-identical to every recorded sweep.
        if getattr(cfg, "mission_window_adaptation", False):
            from hermes.scheduler.stages.s3c_mission_window import (
                MissionWindowAdapter,
            )
            sup_kwargs["mission_window_adapter"] = MissionWindowAdapter(
                enabled=True,
                window=int(getattr(cfg, "mission_window_history", 5)),
                target_success=float(getattr(cfg, "mission_window_target", 0.8)),
                gain=float(getattr(cfg, "mission_window_gain", 2.0)),
                max_scale=float(getattr(cfg, "mission_window_max_scale", 4.0)),
            )
        # FeRRy Phase 1 — the merge rule (must match the cluster's) and the
        # budgeted Pass 2. Defaults reproduce every recorded run.
        from hermes.mission.aggregation_rules import AggregationSpec

        sup_kwargs["aggregation"] = AggregationSpec.from_config(
            getattr(cfg, "aggregation", None),
            getattr(cfg, "aggregation_params", None),
        )
        sup_kwargs["pass_2_budget"] = bool(getattr(cfg, "pass_2_budget", False))
        from hermes.scheduler.stages.s3_deadline import DeadlineLaw

        sup_kwargs["deadline_law"] = DeadlineLaw.from_config(
            getattr(cfg, "deadline_law", None),
            getattr(cfg, "deadline_params", None),
        )
        sup_kwargs["miss_priority"] = bool(getattr(cfg, "miss_priority", False))
        # FeRRy Phase 2 — sharing the cluster with other mules. Both default
        # off (None / False), the recorded single-mule dock.
        sup_kwargs["down_wait_s"] = getattr(cfg, "down_wait_s", None)
        sup_kwargs["dock_on_empty"] = bool(getattr(cfg, "dock_on_empty", False))
        sup_kwargs["should_stop"] = self._stop_event.is_set
        # FeRRy Phase 3 (spec Q1) — the deadline law's time unit, on either
        # clock; passed only when it is not the recorded one.
        scale = getattr(cfg, "deadline_time_scale", 1.0)
        if isinstance(scale, bool) or scale != 1.0:
            sup_kwargs["deadline_time_scale"] = scale
        phi0 = getattr(cfg, "initial_window_s", None)
        if phi0 is not None:
            sup_kwargs["initial_window_s"] = float(phi0)
        # FeRRy Phase 3 — one mission clock per mule process and the ferry
        # spec (design section 2.2). Never with a now_fn: every mission-time
        # read then comes from the clock.
        if self._sim:
            from hermes.l1.mission_clock import MissionClock

            sup_kwargs["mission_clock"] = MissionClock()
            sup_kwargs["ferry"] = ferry_spec
        # FeRRy Phase 4 (simulated clock only; the guards above refuse the
        # plan fields on the wall clock). Plan mode gets its options, built
        # from the config's plan fields exactly as the guards checked them, T
        # for its score (t_nom_s) and member_admission, which must equal the
        # options' (the scheduler's one source, unit_U3b.md section 1.1). An H
        # or D arm's member_admission is passed only when it is not the
        # recorded "whole", so a recorded mule builds its supervisor with
        # exactly the arguments it always did, and loads no plan module.
        admission = getattr(cfg, "member_admission", MEMBER_ADMISSION_WHOLE)
        if getattr(cfg, "plan_mode", PLAN_MODE_LEGACY) == PLAN_MODE_FERRY:
            from hermes.scheduler.plan.types import PlanOptions

            sup_kwargs.update(
                plan_mode=PLAN_MODE_FERRY,
                plan_options=PlanOptions.from_config(
                    **{name: getattr(cfg, name) for name in PLAN_OPTION_FIELDS}
                ),
                t_nom_s=cfg.t_nom_s,
                member_admission=admission,
            )
            # FeRRy Phase 5: the pair slot fills a pair_q mule's flight slot.
            # Passed to that mule only, so F, FX and every other plan mule
            # build their supervisor with exactly the arguments they did.
            if pair_slot is not None:
                sup_kwargs["pair_slot"] = pair_slot
        elif admission != MEMBER_ADMISSION_WHOLE:
            sup_kwargs["member_admission"] = admission
        self.supervisor = MuleSupervisor(
            mule_id=MuleID(cfg.mule_id),
            rf=self.rf,
            dock=self.dock,
            session_ttl_s=cfg.session_ttl_s,
            rf_range_m=cfg.rf_range_m,
            target_selector=(
                e3_policy if e3_policy is not None else _build_target_selector(cfg)
            ),
            **sup_kwargs,
        )

        # The settings the supervisor actually runs, read back from it rather
        # than from the config, so a trace shows what a study arm really ran
        # (audit #15). Additive fields.
        sched = self.supervisor.scheduler
        law = sched.deadline_law
        self.events.emit(
            "mule_ready",
            rf_host=self.cfg.rf_host,
            rf_port=self.actual_rf_port,
            dock_host=self.cfg.dock_host,
            dock_port=self.cfg.dock_port,
            expected_devices=list(self.cfg.expected_devices),
            rf_range_m=self.cfg.rf_range_m,
            session_ttl_s=self.cfg.session_ttl_s,
            n_missions=self.cfg.n_missions,
            aggregation=self.supervisor.aggregation.rule,
            aggregation_params=self.supervisor.aggregation.to_params(),
            deadline_law=(law.form if law is not None else "additive"),
            deadline_params=(law.to_params() if law is not None else None),
            miss_priority=bool(sched.miss_priority),
            pass_2_budget=bool(self.supervisor.pass_2_budget),
            mission_budget_s=sched.mission_budget_s,
            down_wait_s=self.supervisor.down_wait_s,
            dock_on_empty=bool(self.supervisor.dock_on_empty),
            **self._sim_ready_fields(),
            **self._plan_ready_fields(),
            **self._learned_ready_fields(),
            **self._wall_time_unit_fields(),
        )

    def _time_unit_fields(self) -> dict:
        """The deadline law's time unit as the scheduler runs it (spec Q1).

        Read back from the scheduler (audit #15): the scale, Φ₀ as configured
        in the law's recorded unit, and Φ₀ in the scheduler's clock seconds,
        the window a newly tracked device starts with.
        """
        sched = self.supervisor.scheduler
        return dict(
            deadline_time_scale=float(sched.deadline_time_scale),
            initial_window_s=float(sched.initial_window_s),
            effective_initial_window_s=float(sched.effective_initial_window_s),
        )

    def _wall_time_unit_fields(self) -> dict:
        """``mule_ready``'s time-unit fields on the wall clock (spec Q1).

        The time unit is valid on either clock, but ``deadline_params`` shows
        only the scale, never Φ₀. So a wall-clock mule whose unit is not the
        recorded one (scale 1.0, Φ₀ 60 s) reports all three fields of
        :meth:`_time_unit_fields`, as the simulated clock always does; at the
        recorded unit it adds nothing, so every recorded ``mule_ready`` keeps
        its key set. {} on the simulated clock, whose own fields carry them.
        """
        if self._sim:
            return {}
        unit = self._time_unit_fields()
        if (unit["deadline_time_scale"] == 1.0
                and unit["effective_initial_window_s"] == DEFAULT_FULFILMENT_WINDOW_S):
            return {}
        return unit

    def _sim_ready_fields(self) -> dict:
        """``mule_ready``'s simulated-clock fields (design section 2.5); {} on the wall clock.

        Read back from the supervisor and its scheduler, like the fields
        above (audit #15): the clock and its epoch, every ferry setting
        (``FerrySpec.describe``: band classes with their slant and planar
        ranges, channel parameters, backhaul model and policy, response,
        reliability source, payload, SIMULATED energy parameters with the
        speed, turnaround, listen window, deadline bounds), the deadline time
        unit (:meth:`_time_unit_fields`), T_nom, the trial seed, the model's
        input width and where the planner's RF prior comes from
        (``rf_prior_source``, critic B4). The ground-truth availability map is
        never emitted (critic B16), only its size (``device_availability_n``).
        """
        if not self._sim:
            return {}
        from hermes.l1.mission_clock import SIM_EPOCH_S

        sup = self.supervisor
        spec = sup.ferry
        fields = {"mission_clock": CLOCK_SIM, "clock_epoch_s": SIM_EPOCH_S}
        fields.update(_jsonable(spec.describe()))
        if spec.backhaul is not None:
            rf_prior_source = RF_PRIOR_SECONDS_BACKHAUL
        elif self._rf_prior_schedule is not None:
            rf_prior_source = RF_PRIOR_MISSION_SCHEDULE
        else:
            rf_prior_source = RF_PRIOR_CONSTANT
        fields.update(
            payload_bytes=spec.payload.payload_bytes,
            **self._time_unit_fields(),
            t_nom_s=self.cfg.t_nom_s,
            trial_seed=self.cfg.trial_seed,
            input_dim=self.cfg.input_dim,
            rf_link_token_set=self.cfg.rf_link_token is not None,
            rf_prior_source=rf_prior_source,
        )
        return fields

    def _plan_ready_fields(self) -> dict:
        """``mule_ready``'s FeRRy Phase 4 fields: the plan settings the scheduler runs.

        Read back from the scheduler, like the fields above (audit #15). In
        plan mode: ``plan_mode`` and the options by their config names
        (``PlanOptions.describe``: the band-class policy, the flight slot, the
        cap S and lookahead, and the score and search settings resolved, their
        defaults included), with the scheduler's own ``member_admission``.
        Otherwise only a ``member_admission`` other than the recorded
        ``whole`` (an H or D arm run with subsets). {} at the defaults, so a
        recorded ``mule_ready`` keeps its key set (critic D2).
        """
        sched = self.supervisor.scheduler
        admission = getattr(sched, "member_admission", MEMBER_ADMISSION_WHOLE)
        if getattr(sched, "plan_mode", PLAN_MODE_LEGACY) != PLAN_MODE_FERRY:
            return {} if admission == MEMBER_ADMISSION_WHOLE else {"member_admission": admission}
        fields = {"plan_mode": sched.plan_mode}
        fields.update(_jsonable(sched.plan_setup.options.describe()))
        fields["member_admission"] = admission
        return fields

    def _learned_ready_fields(self) -> dict:
        """``mule_ready``'s FeRRy Phase 5 fields: the checkpoint a learned filling flies.

        ``pair`` on a mule whose flight slot is the pair slot, and
        ``policy_checkpoint`` on arm E3's (the Phase 5 spec, other choices 6),
        each the verified manifest's provenance with the config's tag
        (:func:`_checkpoint_provenance`), read back from what the supervisor
        flies (audit #15): the slot's learned score, the scheduler's policy.
        {} on every other mule, so a recorded ``mule_ready`` keeps its key set
        (Freeze Rule 1).
        """
        cfg = self.cfg
        fields = {}
        if getattr(cfg, "flight_slot", None) == FLIGHT_SLOT_PAIR_Q:
            slot = getattr(self.supervisor, "_flight_slot", None)
            scorer = getattr(slot, "scorer", None)
            if scorer is not None:
                fields["pair"] = _checkpoint_provenance(scorer.manifest, cfg.pair_checkpoint_tag)
        if getattr(cfg, "contact_policy", None) == CONTACT_POLICY_CHEN_DQN:
            policy = getattr(self.supervisor.scheduler, "target_selector", None)
            if policy is not None:
                fields["policy_checkpoint"] = _checkpoint_provenance(
                    policy.manifest, cfg.policy_checkpoint_tag)
        return fields

    def _feed_rf_prior(self, result) -> None:
        """The causal RF prior under the recorded ``mission`` backhaul model (critic B4).

        With ``--l1-channel`` a ferry cell is not handed the driver's mean
        SNR over the whole trial (it uses the future); it gets the SNR its
        chosen carrier has at each mission's upload instead
        (``MuleConfig.rf_prior_schedule_db``). Once mission round r has
        uploaded (its partial, or the empty one a ``dock_on_empty`` mission
        docks with), the mule has observed entry ``mission_schedule_index(r)``
        and the planner's ``rf_prior_snr_db`` becomes it: the last SNR seen on
        the carrier used, as the seconds model's producer sets it at each
        upload. A mission that did not dock observed nothing, and the prior
        keeps its value (the configured 20 dB until the first upload).

        The update runs between missions. The only reader of the prior in a
        mule process is the next mission's Pass-1 plan
        (``build_contact_queue``; the selector's feature 10; no L1 channel
        actor is wired in processes), so every plan gets the prior an update
        at the dock would give it. No-op without a schedule.
        """
        schedule = self._rf_prior_schedule
        if not schedule:
            return
        if getattr(result, "empty", False) and not getattr(result, "docked_empty", False):
            return
        idx = mission_schedule_index(getattr(result, "mission_round", None), len(schedule))
        self.supervisor.rf_prior_snr_db = float(schedule[idx])

    def request_stop(self) -> None:
        self._stop_event.set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    # L-M2: chunk size for the bootstrap waits. Short enough that a
    # SIGTERM during startup is honoured within ~1 second instead of
    # hanging for up to 60 s on a never-arriving device or DOWN.
    _BOOTSTRAP_TICK_S: float = 1.0

    def _wait_with_stop(self, fn, total_timeout: float) -> bool:
        """Run ``fn(timeout)`` in tick-sized chunks, bailing on stop_event.

        ``fn`` must return a truthy value when it's satisfied (e.g.,
        ``rf.wait_for_devices`` or
        ``supervisor.wait_for_initial_dock``). Returns the latest call's
        result, or False if the stop_event fires first.
        """
        deadline = time.time() + total_timeout
        while not self._stop_event.is_set():
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            slice_t = min(self._BOOTSTRAP_TICK_S, remaining)
            if fn(slice_t):
                return True
        return False

    def run(self) -> None:
        """Service loop — runs until ``request_stop`` or n_missions hit."""
        log.info(
            "mule %s ready: RF on 127.0.0.1:%d, dock client to %s:%d, "
            "expecting %d device(s)",
            self.cfg.mule_id, self.actual_rf_port,
            self.cfg.dock_host, self.cfg.dock_port,
            len(self.cfg.expected_devices),
        )

        # Wait for every expected device to register on the RF link.
        # L-M2: chunked so a shutdown during this 60 s window is honoured
        # within one tick.
        if self.cfg.expected_devices:
            wanted = [DeviceID(d) for d in self.cfg.expected_devices]
            ok = self._wait_with_stop(
                lambda t: self.rf.wait_for_devices(wanted, timeout=t),
                total_timeout=60.0,
            )
            if not ok and not self._stop_event.is_set():
                log.warning(
                    "mule %s: not all devices registered within 60s "
                    "— proceeding with whoever showed up",
                    self.cfg.mule_id,
                )
            if self._stop_event.is_set():
                log.info("mule %s: stop signalled during device wait", self.cfg.mule_id)
                return

        # Bootstrap dock — wait for the cluster's initial DOWN bundle.
        try:
            ok = self._wait_with_stop(
                self.supervisor.wait_for_initial_dock,
                total_timeout=30.0,
            )
        except MuleSupervisorError as e:
            if not self._sim:
                raise
            # FeRRy Phase 3: on the simulated clock the mule refuses a
            # bootstrap DOWN whose slice carries wall-clock deadline overrides
            # (or whose cluster_sim_ts is no simulated time). Fatal: flying on
            # would mean flying with no slice.
            log.error("mule %s: bootstrap DOWN refused: %s", self.cfg.mule_id, e)
            self.events.emit("dock_bootstrap_failed", reason=str(e))
            self.metrics.increment("dock_bootstrap_failures")
            self.exit_code = EXIT_BOOTSTRAP_FAILED
            return
        if self._stop_event.is_set():
            log.info("mule %s: stop signalled during dock bootstrap", self.cfg.mule_id)
            return
        if not ok:
            log.error(
                "mule %s: cluster did not deliver bootstrap DOWN within 30s",
                self.cfg.mule_id,
            )
            self.events.emit("dock_bootstrap_timeout")
            self.metrics.increment("dock_bootstrap_timeouts")
            self.exit_code = EXIT_BOOTSTRAP_TIMEOUT
            return

        self.events.emit("dock_bootstrapped")

        # Mission loop.
        n_missions = self.cfg.n_missions
        completed = 0
        while not self._stop_event.is_set():
            if n_missions is not None and completed >= n_missions:
                log.info(
                    "mule %s: completed %d missions, exiting",
                    self.cfg.mule_id, completed,
                )
                break
            mission_started_at = time.time()
            # FeRRy Phase 3: the simulated takeoff time, read just before the
            # mission, and the RF prior its Pass-1 plan is handed (critic B4;
            # additive, on the simulated clock only).
            started_sim = (
                {"sim_start_s": float(self.supervisor.mission_clock()),
                 "rf_prior_snr_db": float(self.supervisor.rf_prior_snr_db)}
                if self._sim else {}
            )
            self.events.emit("mission_started", mission_index=completed, **started_sim)
            try:
                result = self.supervisor.run_one_mission()
                completed += 1
                if self._sim:
                    self._feed_rf_prior(result)
                duration_s = time.time() - mission_started_at
                self.metrics.observe("mission_duration_s", duration_s)
                self.metrics.increment("missions_completed")
                queue_size = len(result.pass_1_queue) or len(result.queue)
                log.info(
                    "mule %s: mission %d complete (queue_size=%d)",
                    self.cfg.mule_id, result.mission_round, queue_size,
                )
                # EX-4.2 — flag a recoverable empty round so it isn't mistaken
                # for a productive mission. It is still emitted as a
                # (zero-update, non-closing) mission_completed below so it
                # counts as a round in the metrics.
                if getattr(result, "empty", False):
                    empty_fields = {}
                    if self.supervisor.dock_on_empty:
                        # FeRRy Phase 2, additive: whether this empty mission
                        # still docked. Absent unless dock_on_empty is set.
                        empty_fields["docked"] = bool(result.docked_empty)
                    self.events.emit(
                        "mission_empty", mission_round=result.mission_round,
                        **empty_fields,
                    )
                    self.metrics.increment("missions_empty")
                if getattr(result, "down_timeout", False):
                    # FeRRy Phase 2: the upload stands, but no DOWN came within
                    # down_wait_s, so Pass 2 was skipped and the next mission
                    # flies this one's θ. Never emitted without down_wait_s.
                    self.events.emit(
                        "dock_down_timeout",
                        mission_round=result.mission_round,
                        down_wait_s=self.supervisor.down_wait_s,
                    )
                    self.metrics.increment("dock_down_timeouts")
                # EX-4.0 instrumentation — surface the Pass-1 aggregation
                # ledger so an integrated-experiment consumer can compute
                # update-yield / round-close-rate from the real event
                # stream. ``report`` is the MissionRoundCloseReport from
                # ``close_round`` (populated by both the single-pass and
                # two-pass paths); ``pass_1_queue`` carries the scheduled
                # contact membership. All three fields are additive and
                # optional per the observability schema policy, so adding
                # them does not break existing consumers.
                pass_1_updates, pass_1_clean_devices = _pass_1_collected(result)
                pass_1_scheduled = (
                    sum(len(c.devices) for c in result.pass_1_queue) or None
                )
                _merged = _pass_1_merged_devices(result)
                self.events.emit(
                    "mission_completed",
                    mission_round=result.mission_round,
                    queue_size=queue_size,
                    pass_1_contacts=len(result.pass_1_queue),
                    pass_2_contacts=len(result.pass_2_queue),
                    duration_s=duration_s,
                    pass_1_updates=pass_1_updates,
                    pass_1_scheduled=pass_1_scheduled,
                    pass_1_clean_devices=pass_1_clean_devices,
                    delivered=(
                        result.delivery_report.counts()[0]
                        if result.delivery_report is not None
                        else None
                    ),
                    undelivered=(
                        result.delivery_report.counts()[1]
                        if result.delivery_report is not None
                        else None
                    ),
                    # Trace-scorer fields (additive, optional): the plan with
                    # its deadlines and every session's outcome, so the
                    # deadline-miss rate can be scored from the trace alone.
                    pass_1_plan=_pass_1_plan_payload(
                        result.pass_1_queue,
                        getattr(result, "pass_1_device_deadlines", None),
                    ),
                    pass_1_outcomes=_pass_1_outcomes_payload(result),
                    # Audit #3: the devices whose update was merged, as
                    # distinct from pass_1_clean_devices (sessions completed).
                    pass_1_merged_devices=_merged,
                    pass_1_merged_updates=(
                        None if _merged is None else len(_merged)
                    ),
                    pass_1_merge=_pass_1_merge_payload(result),
                    pass_2_skipped=_pass_2_skipped(result),
                    deadline_state=_deadline_state_payload(
                        self.supervisor.scheduler.device_states
                    ),
                    # FeRRy Phase 3 (design section 2.5): the mission's
                    # simulated record, on the simulated clock only; a
                    # wall-clock mission_completed is exactly the recorded one.
                    **(_sim_mission_fields(result) if self._sim else {}),
                )
            except MuleSupervisorError as e:
                log.error("mule %s: supervisor error: %s", self.cfg.mule_id, e)
                self.events.emit("mission_failed", reason=str(e), kind="supervisor")
                self.metrics.increment("mission_failures")
                self.exit_code = EXIT_MISSION_FAILED
                break
            except Exception as e:
                log.exception("mule %s: unexpected mission failure", self.cfg.mule_id)
                self.events.emit("mission_failed", reason=repr(e), kind="unexpected")
                self.metrics.increment("mission_failures")
                self.exit_code = EXIT_MISSION_FAILED
                break

        log.info("mule %s service loop exiting", self.cfg.mule_id)

    def shutdown(self) -> None:
        self.request_stop()
        try:
            self.rf.close()
        except Exception:
            pass
        try:
            self.dock.close()
        except Exception:
            pass
        try:
            self.events.emit("metrics_snapshot", metrics=self.metrics.snapshot())
            self.events.emit("service_stopped")
            self.events.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _install_signal_handlers(svc: MuleService) -> None:
    def _handle(_signum, _frame):
        log.info("mule received shutdown signal")
        svc.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):  # pragma: no cover
            pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes.processes.mule")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--port-out", type=Path,
        help="If set, write the actual bound RF port here after start.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Chunk M observability: directory for the per-process JSONL log.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cfg = mule_config_from_json(args.config.read_text(encoding="utf-8"))

    events: Optional[JsonEventEmitter] = None
    if args.run_dir is not None:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        events = JsonEventEmitter(
            args.run_dir / f"mule-{cfg.mule_id}.jsonl",
            role="mule",
            node_id=cfg.mule_id,
        )

    svc = MuleService(cfg, events=events)
    _install_signal_handlers(svc)

    if args.port_out is not None:
        args.port_out.write_text(str(svc.actual_rf_port), encoding="utf-8")

    try:
        svc.run()
    finally:
        svc.shutdown()
    # Non-zero only when the loop ended on a failure it could not recover
    # from, so the run's driver can tell a truncated trial from a finished one.
    return svc.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
