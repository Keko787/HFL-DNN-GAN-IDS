"""One FerrySim episode: one trial of a cell, flown by one policy (FeRRy Phase 5, U8a).

An episode is the Exp 4 driver's own trial (``Exp4Driver.run_trial``) of a
FerrySim cell (:mod:`experiments.ferrysim.cells`) at one trial seed, run in
this process (:mod:`experiments.ferrysim.inprocess`; the user's decision 2 (a)),
and read as the reward reads it (:mod:`experiments.ferrysim.reward`): one
:class:`~experiments.ferrysim.reward.SortieRecord` per mission, with one
decision per Pass-1 stop flown, its reward, and the episode's undiscounted
return.

**What flies it** (:class:`Policy`). Either an arm's own configuration and
slot (FX's cross-heuristic, F's committed slot: Study 5.5's references are the
arms themselves, orchestrator resolution R6), or a pair slot: the arm's FX
configuration with the episode's ``PairQSlot`` installed before the first
mission through ``MuleSupervisor.install_flight_slot``, FerrySim's seam (the
spec, other choices 8), ranked by a scripted reference
(``policies.pair_slot.scripted_scorer``), the headroom oracle's replay
scorer, or a learned score. A fresh slot is built per episode, and a trainer
(ε, the episode's seeded stream, the reference phase) is attached before its
first decision when asked; each mission's decisions (``PairStep``) and their
closed records are then kept for the trainer (unit U8b).

**How it is read.** FerrySim observes each mission in process, changing
nothing (a test compares an episode's trace with UG5's oracle): it wraps the
mule supervisor's ``run_one_mission``, whose result carries the stops flown,
the session outcomes, the merge and the closed plan, and ``_ferry_result``,
where every simulated-clock mission closes, for the two instants the result
does not hold: the end of the Pass-1 upload and the landing. Each Pass-1 stop
flown is a decision, its span runs to the next arrival or, after the last, to
the end of the upload (the landing on the empty round), exactly as the mule
closes a pair decision's record (``MuleSupervisor._ferry_close_pairs``). So
FX's, F's and a pair slot's returns are read one way. Under a pair slot the
mule's own closed records (``pass_1_pairs``) are compared with this reading,
stop for stop, and a disagreement raises.

The episode is a pure function of its cell, seed, policy, device model and
(for a trainer) ε and stream: the same inputs give the same records, rewards,
returns and trace, bar the planner's wall time (T3). Not thread-safe: the
in-process trial patches process-wide names (run one episode per process at a
time; parallel runs use worker processes).

**Kept traces.** :func:`run_episode` with a ``trace_root`` keeps the trial's
traces (for the trace scorer) under ``<trace_root>/<cell name>/<policy
label>/``, in the driver's own directory there (``trace_dir_name``: the
runner's cell id, the arm, the trial and the seed). The driver's name alone
cannot tell the episodes of one seed apart: every pair slot flies FX's
configuration, so its trace is filed under arm FX, FerrySim cells of one size
share a runner cell id, and the driver overwrites a directory that is already
there. The trace and the driver's row name the arm whose configuration flew
(FX for every pair slot); the directory names the policy. A kept trace's
devices log only ``device_ready``, since FerrySim runs no device service loop
(:mod:`experiments.ferrysim.inprocess`), so the trace scorer's row of it has
``coverage`` 0.0, ``participation_entropy`` 0.0 and ``jains_fairness`` 1.0,
as the driver's row has (:class:`EpisodeResult`): harness artifacts, not the
trial's values (the orchestrator's resolution R25), which no study reads.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from experiments.exp4.driver import PLAN_ARMS, Exp4Driver
from experiments.ferrysim import inprocess
from experiments.ferrysim.cells import FerryCell
from experiments.ferrysim.reward import (
    DERIVED,
    RewardSpec,
    RewardTerms,
    SortieRecord,
    StopRecord,
    episode_return,
    raw_merge_weights,
    sortie_rewards,
)
from experiments.runner import Cell

ARM_FX = "FX"
ARM_F = "F"

#: The closed records' fields FerrySim's own reading of a flight must equal,
#: stop for stop (``policies.pair_slot`` ``DECISION_KEYS`` and ``CLOSE_KEYS``).
_RECORD_FIELDS = ("t_s", "devices", "collected", "w", "late", "t_next_s", "terminal")

#: What a kept trace's directory names may not hold: the driver's own rule for
#: its trace directories (``experiments.exp4.driver.trace_dir_name``), which
#: are Windows' reserved characters.
_PATH_UNSAFE = '<>:"/\\|?*'


# --------------------------------------------------------------------------- #
# What flies an episode
# --------------------------------------------------------------------------- #

def _scripted(name: str):
    from hermes.scheduler.policies.pair_slot import scripted_scorer

    return scripted_scorer(name)


@dataclasses.dataclass(frozen=True)
class Policy:
    """What flies an episode.

    ``arm`` is the driver arm whose configuration the trial runs. With no
    ``scorer`` the arm's own flight slot flies (FX, F, or any plan arm). With
    one, ``scorer`` is a zero-argument factory of a ``PairScorer`` and the
    episode flies a fresh ``PairQSlot`` around it on the arm's configuration
    (FX's: the slot needs plan mode, ``replan`` and no pinned band). A factory
    the worker pool can send to another process (a module-level function, or
    ``functools.partial`` of one) lets the policy run in a worker. ``label``
    names it in results.
    """

    label: str
    arm: str = ARM_FX
    scorer: Optional[Callable[[], Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError(f"a policy's label is a non-empty string, got {self.label!r}")
        if self.scorer is not None and not callable(self.scorer):
            raise TypeError(f"scorer is a factory of a PairScorer, got {self.scorer!r}")
        if self.scorer is not None and self.arm not in PLAN_ARMS:
            raise ValueError(f"a pair slot flies on a plan arm's configuration, got {self.arm!r}")

    @property
    def pair_slot(self) -> bool:
        return self.scorer is not None

    @classmethod
    def of_arm(cls, arm: str) -> "Policy":
        """The arm itself, its own slot (Study 5.5's FX and F: resolution R6)."""
        return cls(label=str(arm), arm=str(arm))

    @classmethod
    def scripted(cls, name: str) -> "Policy":
        """A pair slot ranked by U3's scripted reference ``name``
        (``fx_pair``, ``committed_pair``, ``hyb``, ``greedy_1``)."""
        from hermes.scheduler.policies.pair_slot import SCRIPTED_SCORERS

        if name not in SCRIPTED_SCORERS:
            raise ValueError(f"a scripted reference is one of {SCRIPTED_SCORERS}, got {name!r}")
        return cls(label=name, arm=ARM_FX, scorer=functools.partial(_scripted, name))


def reference_policies() -> Tuple[Policy, ...]:
    """Study 5.5's references (decision 5; critic A2, B5; resolution R6): the
    FX and F arms themselves, then U3's four scripted references in the slot."""
    from hermes.scheduler.policies.pair_slot import SCRIPTED_SCORERS

    return ((Policy.of_arm(ARM_FX), Policy.of_arm(ARM_F))
            + tuple(Policy.scripted(name) for name in SCRIPTED_SCORERS))


@dataclasses.dataclass(frozen=True)
class Trainer:
    """A trainer attached to an episode's pair slot (FerrySim only; unit U8b).

    ``epsilon`` and ``around_reference`` are the episode's behaviour
    (``pair_q.BehaviourSchedule.at``); ``rng_seed`` seeds the episode's stream
    (``random.Random``), from which every decision with an admitted pair
    draws two numbers (``pair_q.behaviour_row``). ε = 0 with no reference
    flies exactly as no trainer does, and only collects the decisions (U3's
    hand-off).
    """

    epsilon: float = 0.0
    rng_seed: int = 0
    around_reference: bool = False


# --------------------------------------------------------------------------- #
# Drivers, one per cell setting, so T_nom is computed once per process
# --------------------------------------------------------------------------- #

_DRIVERS: Dict[str, Exp4Driver] = {}


def driver_for(settings: Mapping[str, Any]) -> Exp4Driver:
    """The process's ``Exp4Driver`` for these settings, built once.

    T_nom is cached per driver and cell (``Exp4Driver.nominal_period_s``), so a
    worker that keeps its drivers prices T_nom once per cell (the design 2.5).
    """
    key = json.dumps(dict(settings), sort_keys=True, default=str)
    driver = _DRIVERS.get(key)
    if driver is None:
        driver = Exp4Driver(**dict(settings))
        _DRIVERS[key] = driver
    return driver


# --------------------------------------------------------------------------- #
# Observing the mule
# --------------------------------------------------------------------------- #

class _MissionTap:
    """FerrySim's view of the mule's missions, in process; it changes nothing.

    Installed by an ``on_mule`` hook (:class:`inprocess.RoleHooks`) on the
    supervisor instance: ``run_one_mission`` (the result) and ``_ferry_result``
    (the end of the Pass-1 upload and the landing, which the result does not
    hold), each wrapped to call the original with its own arguments and
    return its own result. Before each mission it tells the slot's scorer
    which mission starts (a scorer with ``begin_mission``: the headroom
    oracle's replay scorer), and after mission ``stop_after`` it asks the
    service to stop, so an oracle leaf flies no mission it does not read.
    """

    def __init__(self, *, scorer: Any = None, stop_after: Optional[int] = None) -> None:
        self.scorer = scorer
        self.stop_after = stop_after
        #: What the tap raised inside the mule's loop, which turns it into a
        #: mule failure; the episode raises it instead (:func:`run_episode_on`).
        self.error: Optional[BaseException] = None
        self.sorties: List[SortieRecord] = []
        self.pair_records: List[Optional[Tuple[Dict[str, Any], ...]]] = []
        self.steps: List[Tuple[Any, ...]] = []
        self._closing: Optional[Dict[str, Any]] = None
        self._sunk: List[Tuple[Tuple[Any, ...], Tuple[Dict[str, Any], ...]]] = []

    def sink(self, steps, records) -> None:
        """The trainer's sink: one call per mission close (U3, U5)."""
        self._sunk.append((tuple(steps), tuple(records)))

    def install(self, service) -> None:
        sup = service.supervisor
        for name in ("run_one_mission", "_ferry_result"):
            if not callable(getattr(sup, name, None)):
                raise AssertionError(
                    f"MuleSupervisor.{name} is gone: update FerrySim's mission tap "
                    f"(experiments/ferrysim/episode.py)")
        if getattr(sup, "mission_clock", None) is None:
            raise ValueError("FerrySim flies the simulated mission clock (mission_clock='sim')")
        run_one = sup.run_one_mission
        close = sup._ferry_result

        def ferry_result(**kwargs):
            result = close(**kwargs)
            rec = kwargs["rec"]
            self._closing = {
                "t_pass_1_end": float(kwargs["t_pass_1_end"]),
                "landing_s": getattr(rec, "landing_s", None),
                "empty": bool(kwargs.get("empty", False)),
                "deadlines": {str(d): float(t)
                              for d, t in getattr(rec, "deadlines", {}).items()},
            }
            return result

        def run_one_mission():
            index = len(self.sorties)
            begin = getattr(self.scorer, "begin_mission", None)
            if callable(begin):
                begin(index)
            self._closing = None
            before = len(self._sunk)
            result = run_one()
            try:
                self._read(service, result, index, before)
            except Exception as exc:
                # The mule's loop turns this into mission_failed and a non-zero
                # exit; keep it, so the episode raises it rather than the driver's.
                self.error = exc
                raise
            return result

        sup.run_one_mission = run_one_mission
        sup._ferry_result = ferry_result

    def _read(self, service, result, index: int, before: int) -> None:
        if self._closing is None:
            raise AssertionError(
                "a simulated-clock mission ended without MuleSupervisor._ferry_result: "
                "FerrySim's mission tap no longer sees the close")
        sortie = _sortie_of(service, result, self._closing)
        pairs = getattr(result, "pass_1_pairs", None)
        if pairs is not None:
            _check_pair_records(sortie, pairs)
        sunk = self._sunk[before:]
        if len(sunk) > 1:
            raise AssertionError("the slot's sink was called more than once in a mission")
        self.sorties.append(sortie)
        self.pair_records.append(None if pairs is None else tuple(pairs))
        self.steps.append(sunk[0][0] if sunk else ())
        if self.stop_after is not None and index >= self.stop_after:
            service.request_stop()


def _sortie_of(service, result, closing: Mapping[str, Any]) -> SortieRecord:
    """One mission's Pass 1 as the reward reads it, from the mule's own record.

    Each Pass-1 stop flown (``pass_1_flown``) is a decision; a member is
    collected when its session's outcome was CLEAN (the round's report, or the
    one kept when the age cutoff refused every update), with the raw weight
    the merge gave it (:func:`reward.raw_merge_weights`). The last decision
    ends with the Pass-1 upload, or at the landing on the empty round, as the
    mule closes a pair record.
    """
    from hermes.types import MissionOutcome

    sup = service.supervisor
    cfg = service.cfg
    spec = sup.ferry
    flown = list(getattr(result, "pass_1_flown", None) or [])
    report = getattr(result, "report", None)
    if report is None:
        report = getattr(result, "unmerged_report", None)
    lines = list(getattr(report, "lines", ()) or ())
    outcome = {str(line.device_id): line.outcome for line in lines}
    stamp = {str(line.device_id): float(line.contact_ts) for line in lines}
    weights = raw_merge_weights(sup.aggregation, getattr(result, "aggregate", None),
                                getattr(result, "report", None))
    landing = closing["landing_s"]
    end = landing if (closing["empty"] and landing is not None) else closing["t_pass_1_end"]
    flight_end_last = landing if landing is not None else end
    deadlines = closing["deadlines"]
    dock = tuple(float(c) for c in spec.flight.dock)
    stops: List[StopRecord] = []
    for k, f in enumerate(flown):
        last = k == len(flown) - 1
        devices = [str(d) for d in f["devices"]]
        collected = [d for d in devices if outcome.get(d) is MissionOutcome.CLEAN]
        late = [d for d in collected if stamp[d] > deadlines.get(d, float(f["deadline_ts"]))]
        nxt = None if last else flown[k + 1]
        stops.append(StopRecord(
            t_s=float(f["arrival_s"]),
            devices=tuple(devices),
            targets=tuple(str(d) for d in f["targets"]),
            collected=tuple(collected),
            w=tuple(weights.get(d, 0.0) for d in collected),
            uplink_dropped=tuple(str(d) for d in f.get("uplink_dropped") or ()),
            t_next_s=float(end if last else nxt["arrival_s"]),
            terminal=last,
            end_s=float(f["end_s"]),
            flight_end_s=float(flight_end_last if last else nxt["arrival_s"]),
            dwell_s=float(f["dwell_s"]),
            listen_s=float(f["listen_s"]),
            position=tuple(float(c) for c in f["position"]),
            next_position=dock if last else tuple(float(c) for c in nxt["position"]),
            band=str(f.get("band") or ""),
            late=tuple(late),
        ))
    plan = getattr(result, "plan", None)
    inserted = {str(d) for ins in (getattr(result, "inserts", None) or ())
                for d in ins.get("devices", ())}
    if plan is not None:
        demand = tuple(str(d) for d in plan["demand"])
        committed = tuple(str(d) for d in plan["served"])
        coverage = {str(d): float(w) for d, w in plan["weights"].items()}
        n_demand = len(set(demand) | inserted)
    else:
        demand, committed, coverage = (), (), {}
        n_demand = len(set(str(d) for d in cfg.expected_devices) | inserted)
    channel = getattr(cfg, "contact_reliability_source", "origin") == "channel"
    availability = dict(getattr(cfg, "device_availability", None) or {}) if channel else {}
    energy = spec.flight.energy
    return SortieRecord(
        mission_round=int(result.mission_round),
        stops=tuple(stops),
        n_demand=n_demand,
        demand=demand,
        committed=committed,
        coverage_weights=coverage,
        t_ref_s=None if cfg.t_nom_s is None else float(cfg.t_nom_s),
        availability=availability,
        p_hover_w=float(energy.p_hover_w),
        p_move_w=float(energy.p_move_w),
        empty=bool(getattr(result, "empty", False)),
    )


def _check_pair_records(sortie: SortieRecord, records: Sequence[Mapping[str, Any]]) -> None:
    """The mule's closed pair records against FerrySim's reading of the same flight.

    One record per Pass-1 stop flown, and each field of :data:`_RECORD_FIELDS`
    equal; raises AssertionError on any difference (two readings of one flight
    that disagree are a bug in one of them).
    """
    if len(records) != len(sortie.stops):
        raise AssertionError(f"mission {sortie.mission_round}: {len(records)} pair record(s) for "
                             f"{len(sortie.stops)} Pass-1 stop(s) flown")
    for k, (rec, stop) in enumerate(zip(records, sortie.stops)):
        mine = {"t_s": stop.t_s, "devices": list(stop.devices), "collected": list(stop.collected),
                "w": list(stop.w), "late": list(stop.late), "t_next_s": stop.t_next_s,
                "terminal": stop.terminal}
        theirs = {key: rec[key] for key in _RECORD_FIELDS}
        theirs["devices"] = list(theirs["devices"])
        theirs["collected"] = list(theirs["collected"])
        theirs["w"] = [float(x) for x in theirs["w"]]
        theirs["late"] = list(theirs["late"])
        if mine != theirs:
            diff = sorted(key for key in _RECORD_FIELDS if mine[key] != theirs[key])
            raise AssertionError(
                f"mission {sortie.mission_round}, decision {k}: the mule's record and "
                f"FerrySim's reading differ in {diff}: {theirs} != {mine}")


# --------------------------------------------------------------------------- #
# An episode
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class EpisodeResult:
    """One episode: what it flew, what each decision earned, and its return.

    ``sorties`` holds one record per mission flown (fewer than the cell's
    missions when the episode stopped early), ``rewards`` each sortie's
    rewards per decision under ``reward``, ``pair_records`` the mule's closed
    pair records per mission (None but under a pair slot), and ``steps`` each
    mission's decisions as the trainer reads them (``PairStep``; only with a
    trainer; they hold read-only views and cannot leave the process, so
    :meth:`summary` leaves them out). ``row`` is the driver's row. Its
    device-serve columns are harness artifacts: FerrySim runs no device
    service loop, so no device logs ``device_served``, and every episode's
    row has ``coverage`` 0.0, ``participation_entropy`` 0.0 and
    ``jains_fairness`` 1.0, whatever flies it; its ``mission_duration_s_mean``
    is the harness clock's (:mod:`experiments.ferrysim.inprocess`; the
    orchestrator's resolution R25). No study reads them, and :meth:`summary`
    leaves the row out. ``case`` is the trial's canonical trace (only when
    asked), the planner's wall time masked, that row among its parts.
    """

    cell: str
    seed: int
    trial_index: int
    policy: str
    arm: str
    reward: RewardSpec
    sorties: Tuple[SortieRecord, ...]
    rewards: Tuple[Tuple[RewardTerms, ...], ...]
    pair_records: Tuple[Optional[Tuple[Dict[str, Any], ...]], ...]
    steps: Tuple[Tuple[Any, ...], ...]
    row: Dict[str, Any]
    case: Optional[Dict[str, Any]] = None

    @property
    def terms(self) -> RewardTerms:
        """The undiscounted return, term by term."""
        return episode_return(self.reward, self.sorties)

    @property
    def ret(self) -> float:
        """The undiscounted return (the spec, other choices 12)."""
        return self.terms.total

    @property
    def sortie_returns(self) -> Tuple[float, ...]:
        return tuple(sum(t.total for t in rewards) for rewards in self.rewards)

    @property
    def decisions(self) -> int:
        return sum(len(s.stops) for s in self.sorties)

    def rescored(self, reward: RewardSpec) -> "EpisodeResult":
        """The same flight read under another reward (nothing is flown again)."""
        return dataclasses.replace(
            self, reward=reward,
            rewards=tuple(sortie_rewards(reward, s) for s in self.sorties))

    def summary(self) -> Dict[str, Any]:
        """JSON-ready: the cell, seed, policy, return and terms, and per sortie
        its decisions, returns and records (no steps, no trace)."""
        return {
            "cell": self.cell,
            "seed": int(self.seed),
            "trial_index": int(self.trial_index),
            "policy": self.policy,
            "arm": self.arm,
            "reward": self.reward.to_json(),
            "return": self.ret,
            "terms": self.terms.to_json(),
            "decisions": self.decisions,
            "missions": len(self.sorties),
            "sorties": [
                {"record": s.to_json(), "rewards": [t.to_json() for t in r],
                 "return": sum(t.total for t in r),
                 "undecided_shortfall": s.undecided_shortfall}
                for s, r in zip(self.sorties, self.rewards)
            ],
            "pair_records": [None if p is None else list(p) for p in self.pair_records],
        }


def run_episode_on(
    driver: Exp4Driver,
    cell: Cell,
    policy: Policy,
    *,
    reward: RewardSpec = DERIVED,
    device_model: str = inprocess.DEVICE_MODEL_EQUAL,
    trainer: Optional[Trainer] = None,
    sink: Optional[Callable[..., None]] = None,
    hooks: Sequence[Callable[[Any], None]] = (),
    stop_after: Optional[int] = None,
    keep_case: bool = False,
    case_settings: Optional[Mapping[str, Any]] = None,
    cell_name: str = "",
) -> EpisodeResult:
    """One episode of ``driver``'s trial of ``cell`` flown by ``policy``.

    The low-level entry (:func:`run_episode` takes a FerrySim cell and seed).
    ``cell.arm`` must be ``policy.arm``. ``device_model`` is the in-process
    trial's (``equal`` for training cells, ``stub`` for parity with the
    stack's own trials). ``trainer`` attaches a trainer to the pair slot
    before its first decision and keeps each mission's decisions; ``sink``,
    if given, also receives each mission's (steps, closed records) as the
    slot hands them over. ``hooks`` are further ``on_mule`` callables, run
    after FerrySim's own (E3's trainer, unit U8b). ``stop_after`` stops the
    mule once that mission (0-based) has closed. ``keep_case`` keeps the
    trial's canonical trace, with ``case_settings`` as its recorded inputs.
    """
    if cell.arm != policy.arm:
        raise ValueError(f"the cell's arm {cell.arm!r} is not the policy's {policy.arm!r}")
    if trainer is not None and not policy.pair_slot:
        raise ValueError("a trainer attaches to a pair slot: this policy flies the arm's own")
    if sink is not None and trainer is None:
        raise ValueError("a sink receives a trainer's steps: give a trainer (epsilon 0 just "
                         "collects them)")
    scorer = policy.scorer() if policy.pair_slot else None
    tap = _MissionTap(scorer=scorer, stop_after=stop_after)

    def fly(service) -> None:
        if scorer is None:
            return
        from hermes.scheduler.policies.pair_slot import PairQSlot

        slot = PairQSlot(scorer)
        if trainer is not None:
            def both(steps, records):
                tap.sink(steps, records)
                if sink is not None:
                    sink(steps, records)

            slot.attach_trainer(epsilon=float(trainer.epsilon),
                                rng=random.Random(int(trainer.rng_seed)), sink=both,
                                around_reference=bool(trainer.around_reference))
        service.supervisor.install_flight_slot(slot)

    role_hooks = inprocess.RoleHooks(on_mule=(fly, tap.install) + tuple(hooks),
                                     device_model=device_model)
    try:
        run = inprocess.run_trial(driver, cell, hooks=role_hooks)
    except inprocess.TrialFailure as failure:
        if tap.error is not None:
            raise tap.error from failure
        raise
    sorties = tuple(tap.sorties)
    case = inprocess.trial_case(cell, run, case_settings) if keep_case else None
    return EpisodeResult(
        cell=cell_name or cell.cell_id,
        seed=int(cell.seed),
        trial_index=int(cell.trial_index),
        policy=policy.label,
        arm=policy.arm,
        reward=reward,
        sorties=sorties,
        rewards=tuple(sortie_rewards(reward, s) for s in sorties),
        pair_records=tuple(tap.pair_records),
        steps=tuple(tap.steps) if trainer is not None else (),
        row=dict(run.row),
        case=case,
    )


def _path_part(value: str, what: str) -> str:
    """``value`` as one directory name of a kept trace's path; ValueError otherwise.

    Refused rather than rewritten, as the driver's ``trace_dir_name`` rewrites
    its names, because two labels rewritten alike would share a directory.
    """
    if (not value or value in (".", "..") or value != value.strip(" .")
            or any(c in _PATH_UNSAFE or ord(c) < 32 for c in value)):
        raise ValueError(
            f"{what} {value!r} names a directory of the kept trace: it must be a plain "
            f"path component (none of {_PATH_UNSAFE!r}, no leading or trailing dot or space)")
    return value


def run_episode(
    cell: FerryCell,
    seed: int,
    policy: Policy,
    *,
    trial_index: int = 0,
    reward: RewardSpec = DERIVED,
    device_model: str = inprocess.DEVICE_MODEL_EQUAL,
    trainer: Optional[Trainer] = None,
    sink: Optional[Callable[..., None]] = None,
    hooks: Sequence[Callable[[Any], None]] = (),
    stop_after: Optional[int] = None,
    keep_case: bool = False,
    driver_overrides: Optional[Mapping[str, Any]] = None,
) -> EpisodeResult:
    """One episode of FerrySim cell ``cell`` at trial seed ``seed``.

    The trial seed comes from one of the cells' seed streams
    (:func:`cells.stream_seeds`, :func:`cells.train_episode`);
    ``trial_index`` is the episode's index there. ``driver_overrides`` are
    extra ``Exp4Driver`` settings on top of the cell's (a test's T_nom, a
    trace root to keep the trial's traces for the trace scorer). A
    ``trace_root`` keeps them under ``<trace_root>/<cell name>/<policy
    label>/`` (the module docstring), so the cell's name and the policy's
    label must be plain path components. See :func:`run_episode_on` for the
    rest; it keeps traces where its driver's own ``trace_root`` says.
    """
    if not isinstance(cell, FerryCell):
        raise TypeError(f"cell must be a FerryCell, got {cell!r}")
    settings = cell.driver_settings()
    overrides = dict(driver_overrides or {})
    if overrides.get("trace_root") is not None:
        # A string, as a kept case records its settings (``inprocess.canon``).
        overrides["trace_root"] = str(Path(overrides["trace_root"])
                                      / _path_part(cell.name, "the cell's name")
                                      / _path_part(policy.label, "the policy's label"))
    settings.update(overrides)
    driver = driver_for(settings)
    return run_episode_on(
        driver, cell.cell(policy.arm, seed, trial_index), policy, reward=reward,
        device_model=device_model, trainer=trainer, sink=sink, hooks=hooks,
        stop_after=stop_after, keep_case=keep_case, case_settings=settings,
        cell_name=cell.name,
    )


__all__ = [
    "ARM_F",
    "ARM_FX",
    "EpisodeResult",
    "Policy",
    "Trainer",
    "driver_for",
    "reference_policies",
    "run_episode",
    "run_episode_on",
]
