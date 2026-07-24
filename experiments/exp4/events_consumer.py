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
from typing import Dict, List, Optional, Sequence, Tuple


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


@dataclass
class Exp4Observation:
    """Everything the metric layer needs from one finished trial."""

    n_devices: int
    cluster_rounds_closed: int
    up_bundles_ingested: int
    missions: List[MissionRecord] = field(default_factory=list)
    mission_failures: int = 0
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
    # ------------------------------ cluster ------------------------------ #
    cluster_rounds_closed = len(_events(cluster_rows, "cluster_round_closed"))
    up_bundles_ingested = len(_events(cluster_rows, "up_bundle_ingested"))
    cluster_ready = bool(_events(cluster_rows, "cluster_ready"))

    # ------------------------------- mule -------------------------------- #
    missions: List[MissionRecord] = []
    for r in _events(mule_rows, "mission_completed"):
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
            )
        )
    mission_failures = len(_events(mule_rows, "mission_failed"))
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
