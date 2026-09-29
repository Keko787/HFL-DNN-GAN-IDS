"""JSONL event-stream consumer for Experiment 4 (chunk EX-4.0).

The multi-process orchestrator writes one JSONL file per role under the
run dir (``cluster-<id>.jsonl``, ``mule-<id>.jsonl``,
``device-<id>.jsonl``). Each line is one event envelope::

    {"ts": ..., "schema_version": 1, "role": "mule", "id": "...",
     "event": "mission_completed", ...payload}

This module folds those three streams into one :class:`Exp4Observation`
— the structured, per-trial view the metric layer rolls up. It reads
only *already-flushed per-event lines* (the emitter is line-buffered),
never the end-of-run ``metrics_snapshot``, so it is robust to a hard
``TerminateProcess`` shutdown on Windows that skips the cluster's
``finally`` block.

The parsing is split into a pure :func:`observation_from_rows` (row
dicts → observation, unit-testable without spawning anything) and a thin
:func:`consume_run_dir` that reads the files first.

Several mules (FeRRy Phase 2): each mule numbers its own missions from 1,
so a mission round alone no longer names a mission. The observation keys
its per-mission ledger by :data:`MissionKey` ``(mule_id, mission_round)``;
the round-keyed fields every single-mule caller reads are that ledger's
projection onto rounds, which is exact while one mule flies. Several mules
also let an upload reach the cluster and still never reach θ: a mule that
gave up waiting for its quorum's DOWN uploads again while its first partial
still sits in the open round, and the cluster refuses the second as a
duplicate, though it logs its ingest like any other. The ledger replays the
cluster's open round to find those (see :func:`_unmerged_uploads`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple


#: A mission's identity in a trial: ``(mule_id, mission_round)``. The mule half
#: is normalised by :meth:`Exp4Observation.mule_key`, so with one mule every
#: key carries the same id (or None) whatever a row called it.
MissionKey = Tuple[Optional[str], int]


# --------------------------------------------------------------------------- #
# Structured observation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MissionRecord:
    """One mule mission (= one FL round in the integrated stack).

    Sourced from a mule ``mission_completed`` event. ``pass_1_updates`` /
    ``pass_1_scheduled`` / ``pass_1_clean_devices`` are the EX-4.0
    instrumentation fields added to that event; older logs without them
    leave the optionals ``None`` and the metric layer falls back to
    ``len(pass_1_clean_devices)`` / ``n_devices``. Where the trace records
    what the mule's merge kept (``pass_1_merged_*``), the metric layer counts
    that instead of the CLEAN collections.
    """

    mission_round: int
    pass_1_contacts: int
    pass_2_contacts: int
    pass_1_updates: Optional[int]
    pass_1_scheduled: Optional[int]
    pass_1_clean_devices: Tuple[str, ...]
    delivered: Optional[int]
    undelivered: Optional[int]
    duration_s: Optional[float]
    # Trace-scorer additions (Phase 0). The time window comes from the
    # envelope timestamps of this mission's ``mission_started`` and
    # ``mission_completed`` events; hand-built rows without ``ts`` leave it
    # None. The plan and outcomes are None on traces recorded before the mule
    # emitted them — distinct from an empty tuple, which is a recorded plan
    # (or outcome list) with nothing in it.
    mule_id: Optional[str] = None
    started_ts: Optional[float] = None
    completed_ts: Optional[float] = None
    #: ``(device, deadline_ts)`` for every device admitted to Pass 1.
    pass_1_deadlines: Optional[Tuple[Tuple[str, float], ...]] = None
    #: ``(device, outcome, contact_ts)`` for every Pass-1 session recorded.
    pass_1_outcomes: Optional[Tuple[Tuple[str, str, float], ...]] = None
    # FeRRy audit #3. An age-aware merge can exclude a CLEAN update that is
    # past its age cutoff, so "collected" and "merged" differ: these are the
    # CLEAN devices minus the excluded ones, and their count. Traces recorded
    # between Phase 1 and these fields carry the exclusions only in
    # ``pass_1_merge.excluded``, from which the same subtraction recovers
    # them. None on traces older than Phase 1, where every CLEAN update was
    # merged; an empty tuple is a recorded merge that kept nothing.
    pass_1_merged_devices: Optional[Tuple[str, ...]] = None
    pass_1_merged_updates: Optional[int] = None
    #: Where ``pass_1_deadlines`` came from: ``"device"`` when the plan carried
    #: each member's own Deadline(j), ``"contact"`` when every member was held
    #: to its contact's (tightest-member) deadline, as traces recorded before
    #: ``device_deadlines`` existed are. None without a plan, or with a plan
    #: that admitted nobody.
    deadline_basis: Optional[str] = None

    def contains(self, ts: Optional[float]) -> bool:
        """Whether ``ts`` falls inside this mission's time window."""
        return (
            ts is not None
            and self.started_ts is not None
            and self.completed_ts is not None
            and self.started_ts <= ts <= self.completed_ts
        )


@dataclass(frozen=True)
class ModelEvalPoint:
    """One held-out convergence point (EX-4.1 ``model_eval`` event).

    ``cluster_round`` 0 is the seeded init θ baseline; 1..R are the
    aggregated models after each cross-mule FedAvg.
    """

    cluster_round: int
    accuracy: float
    auc: float
    loss: float
    n_test: int
    ts: Optional[float] = None


@dataclass
class Exp4Observation:
    """Everything the metric layer needs from one finished trial."""

    n_devices: int
    cluster_rounds_closed: int
    up_bundles_ingested: int
    missions: List[MissionRecord] = field(default_factory=list)
    mission_failures: int = 0
    missions_empty: int = 0
    # EX-4.2 — mission_rounds whose mule->BS backhaul upload was dropped
    # (round did not close; recoverable). Used to mark those rounds as
    # not-closed in round_close_rate, so H1's jittery penalty is visible.
    backhaul_lost_rounds: Set[int] = field(default_factory=set)
    backhaul_losses: int = 0
    # FeRRy audit #4 — the cluster's merge ledger under an age-aware rule
    # (empty under agg:plain, which emits none of these events). agg:fedbuff
    # buffers an upload without changing θ (``cluster_merge_deferred``) until
    # a later upload fills the buffer and flushes it, so a deferred mission's
    # updates reach the model at the flushing mission, not their own; an
    # upload that is never flushed never reaches it. ``flush_of`` maps each
    # flushed deferred round to the mission round whose applied merge flushed
    # it. An expired round (``cluster_merge_expired``) merged nothing at all.
    # Neither a deferred nor an expired round closes.
    deferred_rounds: Set[int] = field(default_factory=set)
    expired_rounds: Set[int] = field(default_factory=set)
    flush_of: Dict[int, int] = field(default_factory=dict)
    # EX-4.1 real-model convergence trace (empty on the stub path).
    model_evals: List[ModelEvalPoint] = field(default_factory=list)
    # Per-device Pass-1+Pass-2 serve counts, padded to every device that
    # announced itself (``device_ready``) so zero-serve devices still
    # count toward fairness / entropy denominators.
    per_device_serves: Dict[str, int] = field(default_factory=dict)
    device_serve_failures: int = 0
    # Sanity flags harvested from the streams, surfaced for debugging.
    cluster_ready: bool = False
    mule_ready: bool = False
    dock_bootstrapped: bool = False
    # FeRRy Phase 2 — several mules. ``mule_ids`` are the mules the trial ran
    # (every id on a mule row, and every configured mule), and ``mule_slices``
    # the devices each mule's config assigned it, where the config lists them.
    # The ``*_keys`` fields are the per-mission ledger above keyed by
    # :data:`MissionKey`; the metrics read these, because at K > 1 the
    # round-keyed sets collide (a lost round 2 on one mule would mark every
    # mule's round 2). The round-keyed fields are their projection onto
    # rounds, kept for the callers that read them: exact with one mule, and
    # ambiguous across mules with several. ``__post_init__`` fills whichever
    # half a caller leaves out, so an observation built from the round-keyed
    # fields alone, as before the keys existed, scores as it did then.
    mule_ids: Tuple[str, ...] = ()
    mule_slices: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    backhaul_lost_keys: Set[MissionKey] = field(default_factory=set)
    deferred_keys: Set[MissionKey] = field(default_factory=set)
    expired_keys: Set[MissionKey] = field(default_factory=set)
    flush_of_keys: Dict[MissionKey, MissionKey] = field(default_factory=dict)
    #: Cluster round → the mule whose upload closed it, which is the mule whose
    #: inter-pass dock the round's evaluation ran in.
    closed_by_mule: Dict[int, str] = field(default_factory=dict)
    #: Missions whose upload the cluster logged as ingested but never folded
    #: into θ: a second upload from a mule whose partial was still in the open
    #: round (refused as a duplicate), or a partial still waiting for its
    #: quorum when the trial ended. Only several mules produce them; with one,
    #: every ingest is answered at once and this stays empty.
    unmerged_keys: Set[MissionKey] = field(default_factory=set)

    def __post_init__(self) -> None:
        # Whichever half of the ledger the caller gave, the other follows, so
        # the metrics (which read the keyed half) never ignore a set that was
        # given. A round-keyed set is lifted onto the trial's one mule, where a
        # round names exactly one mission; with several mules it names one of
        # each, so it is refused rather than guessed at. Given both halves, both
        # are kept as given. Fields assigned after construction are not synced.
        lifted = _mule_key(self.mule_ids, None)
        for keyed, by_round in (
            ("backhaul_lost_keys", "backhaul_lost_rounds"),
            ("deferred_keys", "deferred_rounds"),
            ("expired_keys", "expired_rounds"),
        ):
            keys, rounds = getattr(self, keyed), getattr(self, by_round)
            if rounds and not keys:
                self._refuse_round_keyed(by_round)
                setattr(self, keyed, {(lifted, r) for r in rounds})
            elif keys and not rounds:
                setattr(self, by_round, {r for _, r in keys})
        if self.flush_of and not self.flush_of_keys:
            self._refuse_round_keyed("flush_of")
            self.flush_of_keys = {
                (lifted, p): (lifted, at) for p, at in self.flush_of.items()
            }
        elif self.flush_of_keys and not self.flush_of:
            self.flush_of = {p[1]: at[1] for p, at in self.flush_of_keys.items()}

    def _refuse_round_keyed(self, name: str) -> None:
        if self.n_mules > 1:
            raise ValueError(
                f"{name} is keyed by mission round alone, which names one mission "
                f"per mule with {self.n_mules} mules; give its (mule_id, "
                f"mission_round)-keyed counterpart instead"
            )

    @property
    def missions_completed(self) -> int:
        return len(self.missions)

    @property
    def n_mules(self) -> int:
        return len(self.mule_ids)

    def mule_key(self, mule_id: Optional[str]) -> Optional[str]:
        """The mule half of a :data:`MissionKey` for a row that names ``mule_id``."""
        return _mule_key(self.mule_ids, mule_id)

    def mission_key(self, mission: MissionRecord) -> MissionKey:
        return (self.mule_key(mission.mule_id), mission.mission_round)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def _events(rows: Sequence[dict], name: str) -> List[dict]:
    return [r for r in rows if r.get("event") == name]


def observation_from_rows(
    *,
    cluster_rows: Sequence[dict],
    mule_rows: Sequence[dict],
    device_rows: Sequence[dict],
    n_devices: int,
    mule_slices: Optional[Mapping[str, Sequence[str]]] = None,
) -> Exp4Observation:
    """Fold three role event streams into one :class:`Exp4Observation`.

    ``cluster_rows`` / ``mule_rows`` / ``device_rows`` are the parsed
    JSONL envelopes for, respectively, all cluster / mule / device
    processes in the run (already concatenated if there were several of
    a role). ``mule_slices`` maps each configured mule to the devices its
    config assigns it (empty where the config names none); without it the
    trial's mules are the ones its mule rows name.
    """
    # ------------------------------- mule -------------------------------- #
    # Parsed first so the cluster section can place its events in a
    # mission's time window. Rows are walked in order per mule: each
    # ``mission_started`` opens the window its ``mission_completed`` closes.
    missions: List[MissionRecord] = []
    started_at: Dict[Optional[str], Optional[float]] = {}
    for r in mule_rows:
        event = r.get("event")
        mule_id = _opt_str(r.get("id"))
        if event == "mission_started":
            started_at[mule_id] = _opt_float(r.get("ts"))
            continue
        if event != "mission_completed":
            continue
        clean = r.get("pass_1_clean_devices")
        clean_tuple: Tuple[str, ...] = (
            tuple(str(d) for d in clean) if isinstance(clean, (list, tuple)) else ()
        )
        merged = r.get("pass_1_merged_devices")
        merged_n = _opt_int(r.get("pass_1_merged_updates"))
        if not isinstance(merged, (list, tuple)):
            merged = _merged_from_merge_record(clean_tuple, r.get("pass_1_merge"))
            if merged is not None:
                merged_n = len(merged)
        deadlines, deadline_basis = _plan_deadlines(r.get("pass_1_plan"))
        missions.append(
            MissionRecord(
                mission_round=int(r.get("mission_round", 0)),
                pass_1_contacts=int(r.get("pass_1_contacts", 0) or 0),
                pass_2_contacts=int(r.get("pass_2_contacts", 0) or 0),
                pass_1_updates=_opt_int(r.get("pass_1_updates")),
                pass_1_scheduled=_opt_int(r.get("pass_1_scheduled")),
                pass_1_clean_devices=clean_tuple,
                delivered=_opt_int(r.get("delivered")),
                undelivered=_opt_int(r.get("undelivered")),
                duration_s=_opt_float(r.get("duration_s")),
                mule_id=mule_id,
                started_ts=started_at.pop(mule_id, None),
                completed_ts=_opt_float(r.get("ts")),
                pass_1_deadlines=deadlines,
                pass_1_outcomes=_session_outcomes(r.get("pass_1_outcomes")),
                pass_1_merged_devices=(
                    tuple(str(d) for d in merged)
                    if isinstance(merged, (list, tuple)) else None
                ),
                pass_1_merged_updates=merged_n,
                deadline_basis=deadline_basis,
            )
        )

    # The trial's mules: every id a mule row carries, and every configured
    # mule (one may have flown nothing).
    mule_ids = tuple(sorted(
        {str(r["id"]) for r in mule_rows if r.get("id") is not None}
        | {str(m) for m in (mule_slices or {})}
    ))
    place = _mission_placer(missions, mule_ids)

    # ------------------------------ cluster ------------------------------ #
    cluster_rounds_closed = len(_events(cluster_rows, "cluster_round_closed"))
    # A refused partial (Phase 2, several mules) is still logged as ingested,
    # for its reports; it is not an accepted upload.
    up_bundles_ingested = sum(
        1 for r in _events(cluster_rows, "up_bundle_ingested")
        if not r.get("partial_refused")
    )
    cluster_ready = bool(_events(cluster_rows, "cluster_ready"))

    # Traces recorded before Freeze Amendment 5 carry no mission round on
    # ``backhaul_upload_lost`` (the cluster read it from the wrong object).
    # The loss happens at the inter-pass dock, inside the mission's window,
    # so the round is recovered from the mule's timestamps instead.
    backhaul_lost_keys: Set[MissionKey] = set()
    for r in _events(cluster_rows, "backhaul_upload_lost"):
        key = place(
            _opt_str(r.get("mule_id")), _opt_int(r.get("mission_round")),
            _opt_float(r.get("ts")),
        )
        if key is not None:
            backhaul_lost_keys.add(key)
    backhaul_losses = len(_events(cluster_rows, "backhaul_upload_lost"))
    deferred_keys, expired_keys, flush_of_keys = _merge_ledger(
        cluster_rows, place, mule_ids,
    )
    unmerged_keys = _unmerged_uploads(cluster_rows, place, mule_ids)
    # A mule with dock_on_empty (Phase 2) docks after a mission with nothing
    # to upload, and its empty partial goes through the cluster's fold like
    # any other: an age-aware fold cuts it to zero weight and lists it with the
    # expired partials, FedBuff reports it deferred without buffering it, and
    # it holds its mule's place in the open round. These ledgers count updates
    # that failed to reach θ, and it carried none, so it is left out of them;
    # ``missions_empty`` counts it. No trace without dock_on_empty has one.
    empty_uploads = _empty_uploads(mule_rows, mule_ids)
    # Likewise the empty partial a cluster holds in a mule's place when its
    # upload was lost under a quorum above 1 (``awaits_quorum``): the fold
    # lists it with the expired partials, but its mission is counted where it
    # belongs, as a backhaul loss, and carried no update to lose again.
    no_update = empty_uploads | _held_places(cluster_rows, place)
    deferred_keys -= no_update
    expired_keys -= no_update
    unmerged_keys -= no_update
    flush_of_keys = {p: at for p, at in flush_of_keys.items() if p not in no_update}

    model_evals: List[ModelEvalPoint] = []
    for r in _events(cluster_rows, "model_eval"):
        model_evals.append(
            ModelEvalPoint(
                cluster_round=int(r.get("cluster_round", 0) or 0),
                accuracy=float(r.get("accuracy", 0.0) or 0.0),
                auc=float(r.get("auc", 0.0) or 0.0),
                loss=float(r.get("loss", 0.0) or 0.0),
                n_test=int(r.get("n_test", 0) or 0),
                ts=_opt_float(r.get("ts")),
            )
        )
    model_evals.sort(key=lambda p: p.cluster_round)

    mission_failures = len(_events(mule_rows, "mission_failed"))
    missions_empty = len(_events(mule_rows, "mission_empty"))
    mule_ready = bool(_events(mule_rows, "mule_ready"))
    dock_bootstrapped = bool(_events(mule_rows, "dock_bootstrapped"))

    # ------------------------------ device ------------------------------- #
    # Seed the visit map with every device that announced itself so a
    # device that never served still occupies a (zero) slot — otherwise
    # Jain's index / entropy would be computed over the served subset only
    # and overstate fairness.
    per_device_serves: Dict[str, int] = {}
    for r in _events(device_rows, "device_ready"):
        did = r.get("id")
        if did is not None:
            per_device_serves.setdefault(str(did), 0)
    for r in _events(device_rows, "device_served"):
        did = r.get("id")
        if did is None:
            continue
        did = str(did)
        per_device_serves[did] = per_device_serves.get(did, 0) + 1
    device_serve_failures = len(_events(device_rows, "device_serve_failed"))

    return Exp4Observation(
        n_devices=int(n_devices),
        cluster_rounds_closed=cluster_rounds_closed,
        up_bundles_ingested=up_bundles_ingested,
        missions=missions,
        mission_failures=mission_failures,
        missions_empty=missions_empty,
        # The round-keyed fields are left to __post_init__, which projects
        # the keyed ledger below onto rounds.
        backhaul_losses=backhaul_losses,
        model_evals=model_evals,
        per_device_serves=per_device_serves,
        device_serve_failures=device_serve_failures,
        cluster_ready=cluster_ready,
        mule_ready=mule_ready,
        dock_bootstrapped=dock_bootstrapped,
        mule_ids=mule_ids,
        mule_slices={
            str(m): tuple(str(d) for d in devices)
            for m, devices in (mule_slices or {}).items() if devices
        },
        backhaul_lost_keys=backhaul_lost_keys,
        deferred_keys=deferred_keys,
        expired_keys=expired_keys,
        flush_of_keys=flush_of_keys,
        closed_by_mule=_round_closers(cluster_rows),
        unmerged_keys=unmerged_keys,
    )


def consume_run_dir(run_dir, *, n_devices: int) -> Exp4Observation:
    """Read every ``{cluster,mule,device}-*.jsonl`` under ``run_dir``.

    Globs by role prefix so it is agnostic to the exact node ids (and
    tolerant of multi-mule / multi-device topologies). Missing files are
    treated as empty streams — a trial where the mule never started still
    produces a (zeroed) observation rather than raising. Each mule's config
    (``mule-<id>.json``, written by the orchestrator) supplies its slice.
    """
    run_dir = Path(run_dir)
    cluster_rows = _read_role(run_dir, "cluster")
    mule_rows = _read_role(run_dir, "mule")
    device_rows = _read_role(run_dir, "device")
    return observation_from_rows(
        cluster_rows=cluster_rows,
        mule_rows=mule_rows,
        device_rows=device_rows,
        n_devices=n_devices,
        mule_slices=_read_mule_slices(run_dir),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _read_role(run_dir: Path, prefix: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(run_dir.glob(f"{prefix}-*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _read_mule_slices(run_dir: Path) -> Dict[str, Tuple[str, ...]]:
    """Each ``mule-<id>.json``'s ``expected_devices``, by mule id.

    The id is the config's ``mule_id``, else the file name's. A config that
    lists no devices maps to an empty tuple: the mule exists, but its slice
    was assigned elsewhere (round-robin in ``TopologyConfig.validate``).
    """
    slices: Dict[str, Tuple[str, ...]] = {}
    for path in sorted(run_dir.glob("mule-*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(cfg, dict):
            continue
        mule_id = str(cfg.get("mule_id") or path.stem[len("mule-"):])
        devices = cfg.get("expected_devices")
        slices[mule_id] = (
            tuple(str(d) for d in devices) if isinstance(devices, (list, tuple)) else ()
        )
    return slices


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A crash mid-write can leave a torn final line; the
                    # completed lines above it are still valid.
                    continue
    except OSError:
        return rows
    return rows


def _opt_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _opt_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_str(v) -> Optional[str]:
    return None if v is None else str(v)


def _mule_key(mule_ids: Sequence[str], mule_id: Optional[str]) -> Optional[str]:
    """The mule half of a :data:`MissionKey` (see :meth:`Exp4Observation.mule_key`).

    With one mule, or none named, every mission is that mule's, whatever a row
    calls it (hand-built rows often leave the id out), so every key takes the
    trial's one id: the keyed ledger is then the round-keyed one exactly. With
    several, each row's own id.
    """
    if len(mule_ids) > 1:
        return mule_id
    return mule_ids[0] if mule_ids else None


def _mission_round_at(
    missions: Sequence[MissionRecord],
    mule_id: Optional[str],
    ts: Optional[float],
) -> Optional[int]:
    """Round of the mission (flown by ``mule_id``) whose window contains ``ts``."""
    for m in missions:
        same_mule = mule_id is None or m.mule_id is None or m.mule_id == mule_id
        if same_mule and m.contains(ts):
            return m.mission_round
    return None


def _mission_placer(
    missions: Sequence[MissionRecord], mule_ids: Sequence[str],
) -> Callable[..., Optional[MissionKey]]:
    """``place(mule_id, mission_round, ts, hint=None)`` → the event's mission key.

    With one mule an event is placed exactly as before multi-mule keys
    existed: by its recorded round, else by the mission window its timestamp
    falls in. With several, the mule matters. An event that names no mule
    takes ``hint`` (the uploader, when the caller knows it), then the one mule
    that flew that round, then the one whose window at that round contains
    the event; an event without a round is placed by the window of its mule's
    missions, and by any mule's only while those windows name a single mule.
    None when the event cannot be placed.
    """
    multi = len(mule_ids) > 1

    def place(
        mule_id: Optional[str], rnd: Optional[int], ts: Optional[float],
        hint: Optional[str] = None,
    ) -> Optional[MissionKey]:
        if not multi:
            if rnd is None:
                rnd = _mission_round_at(missions, mule_id, ts)
            return None if rnd is None else (_mule_key(mule_ids, mule_id), rnd)
        mule_id = mule_id if mule_id is not None else hint
        if rnd is None:
            return _only_key(
                m for m in missions
                if (mule_id is None or m.mule_id == mule_id) and m.contains(ts)
            )
        if mule_id is None:
            same_round = [m for m in missions if m.mission_round == rnd]
            key = _only_key(same_round)
            if key is None:
                key = _only_key(m for m in same_round if m.contains(ts))
            return key
        return (mule_id, rnd)

    return place


def _only_key(missions) -> Optional[MissionKey]:
    """The key of the first mission, if every one given is the same mule's."""
    missions = list(missions)
    if not missions or len({m.mule_id for m in missions}) != 1:
        return None
    return (missions[0].mule_id, missions[0].mission_round)


def _merge_ledger(
    cluster_rows: Sequence[dict],
    place: Callable[..., Optional[MissionKey]],
    mule_ids: Sequence[str],
) -> Tuple[Set[MissionKey], Set[MissionKey], Dict[MissionKey, MissionKey]]:
    """Deferred missions, expired missions, and which mission flushed each deferral.

    Walks the cluster's merge events in order. Each is placed (see
    :func:`_mission_placer`) by its recorded ``mule_id`` and ``mission_round``
    (the uploading mission) or, on traces from before the cluster recorded
    the round, by the mission window its timestamp falls in — the merge runs
    at the inter-pass dock, as a backhaul loss does. An applied
    ``cluster_merge`` names no mule; its uploader is the mule of the
    ``up_bundle_ingested`` just before it, whose ingest ran the fold. An
    applied merge that lists the ``partials`` it flushed is taken at its
    word; one that does not flushes every mission its mule has had deferred
    since the previous flush, which is what a FedBuff buffer does. Only
    missions seen deferred enter the flush map. A merge whose own mission
    cannot be placed still empties the buffer, but its flushed missions get
    no flush mission and so are never credited.
    """
    deferred: Set[MissionKey] = set()
    expired: Set[MissionKey] = set()
    flush_of: Dict[MissionKey, MissionKey] = {}
    # Deferred missions awaiting a flush, per mule as each event names it,
    # for the fallback above.
    pending: Dict[Optional[str], List[MissionKey]] = {}
    last_up: Tuple[Optional[str], Optional[int]] = (None, None)
    for r in cluster_rows:
        event = r.get("event")
        if _runs_a_fold(r):
            last_up = (_opt_str(r.get("mule_id")), _opt_int(r.get("mission_round")))
            continue
        if event not in ("cluster_merge_deferred", "cluster_merge_expired", "cluster_merge"):
            continue
        mule_id = _opt_str(r.get("mule_id"))
        rnd = _opt_int(r.get("mission_round"))
        partials = _partial_keys(r.get("partials"), mule_ids)
        key = place(
            mule_id, rnd, _opt_float(r.get("ts")),
            hint=_uploader(last_up, rnd, partials),
        )
        if event == "cluster_merge_deferred":
            if key is not None:
                deferred.add(key)
                pending.setdefault(mule_id, []).append(key)
            continue
        if event == "cluster_merge_expired":
            # An expired fold discards every pending partial, not only the
            # uploader's; its ``partials`` names them all.
            if key is not None:
                expired.add(key)
            expired.update(partials or ())
            continue
        if not r.get("applied", True):
            continue
        # A partial cut to zero weight inside an applied fold never reached θ.
        expired.update(_partial_keys(r.get("expired_partials"), mule_ids) or ())
        if partials is None:
            if mule_id is None:
                flushed = [p for keys in pending.values() for p in keys]
                pending.clear()
            else:
                flushed = pending.pop(mule_id, [])
        else:
            flushed = [p for p in partials if p != key]
            for keys in pending.values():
                keys[:] = [p for p in keys if p not in flushed]
        if key is not None:
            for p in flushed:
                if p in deferred:
                    flush_of[p] = key
    return deferred, expired, flush_of


def _unmerged_uploads(
    cluster_rows: Sequence[dict],
    place: Callable[..., Optional[MissionKey]],
    mule_ids: Sequence[str],
) -> Set[MissionKey]:
    """Missions the cluster logged as ingested but never folded into θ.

    Replays the cluster's open round from its event log, which one service
    thread writes in order. ``HFLHostCluster.ingest_up_bundle`` keeps one
    partial per mule per open round and refuses a second as a duplicate, yet
    the service logs ``up_bundle_ingested`` for it all the same: a mule that
    stopped waiting for its quorum's DOWN (``down_wait_s``) flies on and
    uploads again while its first partial still waits. So an ingest from a
    mule that already holds a partial in the open round never reached θ,
    and nor did a partial still open when the log ends. The round empties at
    ``cluster_round_closed``, at a deferral (FedBuff moves every pending
    partial into its buffer) and at an expiry; a backhaul loss never reached
    the round and leaves it as it was.

    Where the cluster says so, the replay follows it: an ingest marked
    ``partial_refused`` is a refused duplicate whatever the replay holds, and
    a ``backhaul_upload_lost`` marked ``awaits_quorum`` put an empty partial
    in the lost mule's place (unless the round already held one of its),
    which holds that place and carries no update, so it is never itself
    counted here. An applied fold that lists its ``partials`` has the last
    word: an open partial it neither merged nor cut (``expired_partials``)
    did not reach θ, and any mission it merged did, whatever the replay said
    — the fold is what the cluster did, the replay only a model of it.

    Only with several mules: with one, each ingest is answered by a merge,
    deferral or expiry before the next, so no upload is refused, and the
    ledger stays exactly what it was before this existed.
    """
    if len(mule_ids) <= 1:
        return set()
    unmerged: Set[MissionKey] = set()
    folded: Set[MissionKey] = set()
    # The mule of each partial in the open round → its mission (None when it
    # cannot be placed, which still holds the mule's place).
    open_round: Dict[str, Optional[MissionKey]] = {}
    for r in cluster_rows:
        event = r.get("event")
        if event == "up_bundle_ingested":
            mule_id = _opt_str(r.get("mule_id"))
            if mule_id is None:
                continue          # whose upload it was is unknown
            rnd = _opt_int(r.get("mission_round"))
            key = place(mule_id, rnd, _opt_float(r.get("ts")))
            if r.get("partial_refused"):
                # The round kept the partial it already held for this mule;
                # a resend of that same mission changes nothing.
                held = _opt_int(r.get("held_mission_round"))
                if key is not None and (held is None or held != rnd):
                    unmerged.add(key)
            elif mule_id in open_round:
                if key is not None:
                    unmerged.add(key)
            else:
                open_round[mule_id] = key
        elif event == "backhaul_upload_lost":
            mule_id = _opt_str(r.get("mule_id"))
            if r.get("awaits_quorum") and mule_id is not None:
                open_round.setdefault(mule_id, None)
        elif event == "cluster_merge":
            partials = _partial_keys(r.get("partials"), mule_ids)
            if not r.get("applied", True) or partials is None:
                continue
            folded.update(partials)
            named = set(partials) | set(_partial_keys(r.get("expired_partials"), mule_ids) or ())
            unmerged.update(
                k for k in open_round.values() if k is not None and k not in named
            )
            open_round.clear()
        elif event in ("cluster_round_closed", "cluster_merge_deferred", "cluster_merge_expired"):
            open_round.clear()
    unmerged.update(k for k in open_round.values() if k is not None)
    return unmerged - folded


def _runs_a_fold(r: dict) -> bool:
    """Whether a cluster row is an upload whose partial joined the open round
    and ran the fold: an accepted ingest, or the empty partial held for a
    lost upload under a quorum (a refused ingest runs none)."""
    event = r.get("event")
    if event == "up_bundle_ingested":
        return not r.get("partial_refused")
    return event == "backhaul_upload_lost" and bool(r.get("awaits_quorum"))


def _held_places(
    cluster_rows: Sequence[dict], place: Callable[..., Optional[MissionKey]],
) -> Set[MissionKey]:
    """Missions whose upload was lost under a quorum above 1, where the cluster
    held an empty partial in the mule's place (``awaits_quorum``)."""
    keys: Set[MissionKey] = set()
    for r in _events(cluster_rows, "backhaul_upload_lost"):
        if not r.get("awaits_quorum"):
            continue
        key = place(
            _opt_str(r.get("mule_id")), _opt_int(r.get("mission_round")),
            _opt_float(r.get("ts")),
        )
        if key is not None:
            keys.add(key)
    return keys


def _empty_uploads(mule_rows: Sequence[dict], mule_ids: Sequence[str]) -> Set[MissionKey]:
    """Missions that docked with nothing to upload: the ``mission_empty`` rows
    marked ``docked``, which only a mule with ``dock_on_empty`` writes."""
    keys: Set[MissionKey] = set()
    for r in _events(mule_rows, "mission_empty"):
        rnd = _opt_int(r.get("mission_round"))
        if r.get("docked") and rnd is not None:
            keys.add((_mule_key(mule_ids, _opt_str(r.get("id"))), rnd))
    return keys


def _uploader(
    last_up: Tuple[Optional[str], Optional[int]],
    rnd: Optional[int],
    partials: Optional[Sequence[MissionKey]],
) -> Optional[str]:
    """The mule whose upload ran a merge event that names none.

    The cluster emits ``up_bundle_ingested`` and then any merge event its
    fold produced, so the last ingested upload is the uploader when the rounds
    agree. Without one, the only partial at the merge's round is the
    uploader's own.
    """
    up_mule, up_round = last_up
    if up_mule is not None and (rnd is None or up_round is None or up_round == rnd):
        return up_mule
    if rnd is not None and partials:
        own = {p[0] for p in partials if p[1] == rnd}
        if len(own) == 1:
            return own.pop()
    return None


def _round_closers(cluster_rows: Sequence[dict]) -> Dict[int, str]:
    """Cluster round → the mule whose upload closed it.

    ``cluster_round_closed`` follows the ingest of the upload that completed
    the merge (and any merge event between them), so the round belongs to the
    last ingested upload's mule unless the event names its own.
    """
    closers: Dict[int, str] = {}
    last_mule: Optional[str] = None
    for r in cluster_rows:
        event = r.get("event")
        if _runs_a_fold(r):
            last_mule = _opt_str(r.get("mule_id"))
        elif event == "cluster_round_closed":
            rnd = _opt_int(r.get("cluster_round"))
            mule = _opt_str(r.get("mule_id")) or last_mule
            if rnd is not None and mule is not None:
                closers[rnd] = mule
    return closers


def _merged_from_merge_record(
    clean: Tuple[str, ...], merge,
) -> Optional[Tuple[str, ...]]:
    """CLEAN devices minus ``pass_1_merge.excluded``, for traces recorded
    after Phase 1 added the merge record but before the mule emitted
    ``pass_1_merged_devices``. None when there is no merge record."""
    if not isinstance(merge, dict) or not isinstance(merge.get("excluded"), (list, tuple)):
        return None
    excluded = {str(d) for d in merge["excluded"]}
    return tuple(d for d in clean if d not in excluded)


def _partial_keys(raw, mule_ids: Sequence[str]) -> Optional[List[MissionKey]]:
    """Mission keys of a merge's ``partials`` (``[[mule_id, round], ...]``)."""
    if not isinstance(raw, (list, tuple)):
        return None
    keys: List[MissionKey] = []
    for p in raw:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            rnd = _opt_int(p[1])
            if rnd is not None:
                keys.append((_mule_key(mule_ids, _opt_str(p[0])), rnd))
    return keys


def _plan_deadlines(
    raw,
) -> Tuple[Optional[Tuple[Tuple[str, float], ...]], Optional[str]]:
    """``pass_1_plan`` → ``(device, deadline_ts)`` pairs and their basis.

    Each member is held to its own Deadline(j) when its contact records one
    (``device_deadlines``). A contact also carries its tightest member's
    deadline (S3a), which is all a trace recorded before ``device_deadlines``
    has, so there every member falls back to it — and a later member that
    finished after the contact's deadline but before its own counts as late.
    The basis says which was used: ``"device"`` when every member had its own,
    ``"contact"`` when any fell back. Both are None when the plan is absent;
    the basis is also None for a plan that admitted nobody.
    """
    if not isinstance(raw, list):
        return None, None
    pairs: List[Tuple[str, float]] = []
    fell_back = False
    for contact in raw:
        if not isinstance(contact, dict):
            continue
        contact_deadline = _opt_float(contact.get("deadline_ts"))
        own = contact.get("device_deadlines")
        if not isinstance(own, dict):
            own = {}
        for device in contact.get("devices") or ():
            deadline = _opt_float(own.get(str(device)))
            if deadline is None:
                if contact_deadline is None:
                    continue
                deadline = contact_deadline
                fell_back = True
            pairs.append((str(device), deadline))
    if not pairs:
        return (), None
    return tuple(pairs), ("contact" if fell_back else "device")


def _session_outcomes(raw) -> Optional[Tuple[Tuple[str, str, float], ...]]:
    """``pass_1_outcomes`` → ``(device, outcome, contact_ts)``; None when absent."""
    if not isinstance(raw, list):
        return None
    sessions: List[Tuple[str, str, float]] = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        contact_ts = _opt_float(s.get("contact_ts"))
        if s.get("device") is None or s.get("outcome") is None or contact_ts is None:
            continue
        sessions.append((str(s["device"]), str(s["outcome"]), contact_ts))
    return tuple(sessions)
