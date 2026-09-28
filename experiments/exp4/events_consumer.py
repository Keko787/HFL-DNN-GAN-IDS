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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


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
    ``len(pass_1_clean_devices)`` / ``n_devices``.
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

    @property
    def missions_completed(self) -> int:
        return len(self.missions)


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
) -> Exp4Observation:
    """Fold three role event streams into one :class:`Exp4Observation`.

    ``cluster_rows`` / ``mule_rows`` / ``device_rows`` are the parsed
    JSONL envelopes for, respectively, all cluster / mule / device
    processes in the run (already concatenated if there were several of
    a role).
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
                pass_1_deadlines=_plan_deadlines(r.get("pass_1_plan")),
                pass_1_outcomes=_session_outcomes(r.get("pass_1_outcomes")),
            )
        )

    # ------------------------------ cluster ------------------------------ #
    cluster_rounds_closed = len(_events(cluster_rows, "cluster_round_closed"))
    up_bundles_ingested = len(_events(cluster_rows, "up_bundle_ingested"))
    cluster_ready = bool(_events(cluster_rows, "cluster_ready"))

    # Traces recorded before Freeze Amendment 5 carry no mission round on
    # ``backhaul_upload_lost`` (the cluster read it from the wrong object).
    # The loss happens at the inter-pass dock, inside the mission's window,
    # so the round is recovered from the mule's timestamps instead.
    backhaul_lost_rounds: Set[int] = set()
    for r in _events(cluster_rows, "backhaul_upload_lost"):
        mr = _opt_int(r.get("mission_round"))
        if mr is None:
            mr = _mission_round_at(
                missions, _opt_str(r.get("mule_id")), _opt_float(r.get("ts")),
            )
        if mr is not None:
            backhaul_lost_rounds.add(mr)
    backhaul_losses = len(_events(cluster_rows, "backhaul_upload_lost"))

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
        backhaul_lost_rounds=backhaul_lost_rounds,
        backhaul_losses=backhaul_losses,
        model_evals=model_evals,
        per_device_serves=per_device_serves,
        device_serve_failures=device_serve_failures,
        cluster_ready=cluster_ready,
        mule_ready=mule_ready,
        dock_bootstrapped=dock_bootstrapped,
    )


def consume_run_dir(run_dir, *, n_devices: int) -> Exp4Observation:
    """Read every ``{cluster,mule,device}-*.jsonl`` under ``run_dir``.

    Globs by role prefix so it is agnostic to the exact node ids (and
    tolerant of multi-mule / multi-device topologies). Missing files are
    treated as empty streams — a trial where the mule never started still
    produces a (zeroed) observation rather than raising.
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
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _read_role(run_dir: Path, prefix: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(run_dir.glob(f"{prefix}-*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


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


def _plan_deadlines(raw) -> Optional[Tuple[Tuple[str, float], ...]]:
    """``pass_1_plan`` → ``(device, deadline_ts)`` pairs; None when absent.

    Each planned contact carries its tightest member's deadline (S3a), and
    every member is held to it: that is the deadline the plan committed to.
    """
    if not isinstance(raw, list):
        return None
    pairs: List[Tuple[str, float]] = []
    for contact in raw:
        if not isinstance(contact, dict):
            continue
        deadline = _opt_float(contact.get("deadline_ts"))
        if deadline is None:
            continue
        for device in contact.get("devices") or ():
            pairs.append((str(device), deadline))
    return tuple(pairs)


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
