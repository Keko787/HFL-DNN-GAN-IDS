"""Exp 5 addendum, Study 5.11: a trial's processes and peak memory (the footprint).

A real-process trial runs 1 + K + N processes (the cluster, K mules and N
devices), each importing TensorFlow under ``--real-model`` at about 1.6 GB
apiece (the build plan's addendum, 5.11's caveats), so memory, not CPU, sets
how large an N the stack can run. The footprint probe measures it per trial:

* **Sampled.** A daemon thread reads the resident set size (RSS) of every
  process of the trial, each with its children, every ``interval_s`` while the
  trial runs. ``peak_rss_bytes_total`` is the largest sum over one sample: the
  concurrent peak, which is what has to fit in the host's memory.
* **Recorded by the OS.** Each process's own high-water mark (``VmHWM`` from
  ``/proc/<pid>/status`` on Linux, the peak working set on Windows), which no
  sampling interval can miss, read at every sample and at the end, so a
  process that exits during the trial (a mule, once its missions are flown)
  keeps the last mark read while it ran; on a platform that records none,
  the largest sample of that process instead (``peak_source``). Per role,
  the largest of these (``peak_rss_bytes_by_role``), and their sum
  (``peak_rss_bytes_sum``), an upper bound on the concurrent peak, since the
  processes need not peak together.

The probe reads only: it never signals, waits on or changes a process, and a
process that has exited or cannot be read is skipped for that sample. Wall
time, sampled on the wall clock, so nothing it writes enters a determinism
comparison. It needs ``psutil``, imported only when a probe is built, so no
other path loads it.

The driver (``Exp4Driver(footprint_probe=True)``, ``--footprint-probe``)
starts one before the orchestrator starts any process, follows each process
from the instant it is launched (the orchestrator's ``on_spawn`` hook,
:meth:`FootprintProbe.on_spawn`), stops it before shutdown, and writes
:data:`FOOTPRINT_FILE` beside the kept trace; the scorer reads it into its
cost columns (``traces_scorer --cost-columns``).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_INTERVAL_S",
    "FOOTPRINT_FILE",
    "FOOTPRINT_SCHEMA",
    "ROLES",
    "Footprint",
    "FootprintProbe",
    "probe_available",
    "read_footprint",
]

#: The footprint's file beside a kept trace (next to ``trial_status.json``).
FOOTPRINT_FILE = "footprint.json"
#: The file's schema version.
FOOTPRINT_SCHEMA = 1
#: The sampling interval: fine enough for a training burst of a few seconds,
#: and about 100 reads a sample at N = 96.
DEFAULT_INTERVAL_S = 0.5
#: The roles of a trial's processes, in the orchestrator's start order.
ROLES: Tuple[str, ...] = ("cluster", "mule", "device")

#: ``peak_source`` values: every process's peak came from the OS, from the
#: samples, or some of each.
PEAK_OS = "os"
PEAK_SAMPLED = "sampled"
PEAK_MIXED = "mixed"


def probe_available() -> bool:
    """Whether ``psutil`` can be imported (the probe's one dependency)."""
    try:
        import psutil  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class Footprint:
    """One trial's processes and peak memory (:data:`FOOTPRINT_FILE`)."""

    #: How many processes of each role the trial ran (its children not counted).
    processes_by_role: Mapping[str, int]
    #: The largest summed RSS over one sample (the concurrent peak), bytes;
    #: None with no sample.
    peak_rss_bytes_total: Optional[int]
    #: The largest per-process peak of each role, bytes (None for a role whose
    #: processes could not be read).
    peak_rss_bytes_by_role: Mapping[str, Optional[int]]
    #: Every process's own peak, summed: an upper bound on the concurrent peak.
    peak_rss_bytes_sum: Optional[int]
    #: Where the per-process peaks came from: ``os``, ``sampled`` or ``mixed``.
    peak_source: str
    samples: int
    interval_s: float
    #: The probe's library and the platform, for the record.
    probe: str = ""
    schema: int = FOOTPRINT_SCHEMA

    @property
    def processes(self) -> int:
        return int(sum(self.processes_by_role.values()))

    def to_json(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "processes": self.processes,
            "processes_by_role": {r: int(self.processes_by_role.get(r, 0)) for r in ROLES},
            "peak_rss_bytes_total": self.peak_rss_bytes_total,
            "peak_rss_bytes_by_role": {r: self.peak_rss_bytes_by_role.get(r) for r in ROLES},
            "peak_rss_bytes_sum": self.peak_rss_bytes_sum,
            "peak_source": self.peak_source,
            "samples": int(self.samples),
            "interval_s": float(self.interval_s),
            "probe": self.probe,
        }

    def write(self, directory) -> Path:
        path = Path(directory) / FOOTPRINT_FILE
        path.write_text(json.dumps(self.to_json(), sort_keys=True), encoding="utf-8")
        return path


def read_footprint(trace_dir) -> Optional[Dict[str, object]]:
    """The :data:`FOOTPRINT_FILE` of a kept trace as written; None when it is
    absent or not a JSON object (a trial run without the probe)."""
    path = Path(trace_dir) / FOOTPRINT_FILE
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _os_peak_bytes(proc) -> Optional[int]:
    """The process's own high-water mark of resident memory, from the OS.

    Windows records the peak working set (``memory_info().peak_wset``) and
    Linux ``VmHWM`` in ``/proc/<pid>/status``; anywhere else, or when the
    process cannot be read, None.
    """
    try:
        peak = getattr(proc.memory_info(), "peak_wset", None)
        if peak is not None:
            return int(peak)
        if sys.platform.startswith("linux"):
            with open(f"/proc/{proc.pid}/status", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001 - an exited or unreadable process has none
        return None
    return None


class FootprintProbe:
    """Samples a trial's processes from a daemon thread; :meth:`stop` returns the :class:`Footprint`.

    ``pids`` maps each role (:data:`ROLES`) to the pids of its processes;
    more can join at any time (:meth:`add`, :meth:`on_spawn`). Each process
    is read with its children (``children(recursive=True)``), so a process
    that forks a helper is counted whole; the children's own peaks are not in
    the OS figure, which is the parent's.
    """

    def __init__(self, pids: Optional[Mapping[str, Sequence[int]]] = None, *,
                 interval_s: float = DEFAULT_INTERVAL_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        import psutil

        if not interval_s > 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s!r}")
        pids = dict(pids or {})
        unknown = set(pids) - set(ROLES)
        if unknown:
            raise ValueError(f"unknown role(s) {sorted(unknown)}; the roles are {ROLES}")
        self._psutil = psutil
        self.interval_s = float(interval_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._roles: Dict[str, List[int]] = {r: [] for r in ROLES}
        self._procs: Dict[int, object] = {}
        self._sampled_peak: Dict[int, int] = {}
        self._os_peak: Dict[int, int] = {}
        self._total_peak: Optional[int] = None
        self._samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        for role in ROLES:
            for pid in pids.get(role, ()):
                self.add(role, pid)

    def add(self, role: str, pid: int) -> None:
        """Follow process ``pid`` of ``role`` from now on (thread-safe)."""
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; the roles are {ROLES}")
        try:
            proc = self._psutil.Process(int(pid))
        except self._psutil.Error:
            proc = None
        with self._lock:
            self._roles[role].append(int(pid))
            if proc is not None:
                self._procs[int(pid)] = proc

    def on_spawn(self, name: str, pid: int) -> None:
        """The orchestrator's ``on_spawn`` hook: ``cluster``, ``mule-<id>`` or
        ``device-<id>`` launched as ``pid``."""
        role = "cluster" if name == "cluster" else name.split("-", 1)[0]
        self.add(role, pid)

    @classmethod
    def of_orchestrator(cls, orch, **kw) -> "FootprintProbe":
        """A probe over every process an orchestrator started (its handles)."""
        cluster = orch.cluster_handle
        return cls({
            "cluster": [] if cluster is None else [cluster.proc.pid],
            "mule": [h.proc.pid for h in orch.mule_handles.values()],
            "device": [h.proc.pid for h in orch.device_handles.values()],
        }, **kw)

    def _rss(self, proc) -> Optional[int]:
        psutil = self._psutil
        try:
            total = int(proc.memory_info().rss)
        except psutil.Error:
            return None
        try:
            children = proc.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                total += int(child.memory_info().rss)
            except psutil.Error:
                continue
        return total

    def sample(self) -> None:
        """Read every process once (the thread's step; public for tests)."""
        with self._lock:
            procs = list(self._procs.items())
        total, read = 0, False
        for pid, proc in procs:
            rss = self._rss(proc)
            if rss is None:
                continue
            read = True
            total += rss
            peak = _os_peak_bytes(proc)
            with self._lock:
                self._sampled_peak[pid] = max(self._sampled_peak.get(pid, 0), rss)
                if peak is not None:
                    self._os_peak[pid] = max(self._os_peak.get(pid, 0), peak)
        with self._lock:
            self._samples += 1
            if read:
                self._total_peak = total if self._total_peak is None else max(
                    self._total_peak, total)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = self._clock()
            self.sample()
            self._stop.wait(max(0.0, self.interval_s - (self._clock() - started)))

    def start(self) -> "FootprintProbe":
        if self._thread is not None:
            raise RuntimeError("the probe is already started")
        self._thread = threading.Thread(target=self._run, name="footprint-probe", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Footprint:
        """Stop sampling (one last sample first) and return the footprint."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=10.0 * self.interval_s + 5.0)
        self.sample()
        peaks: Dict[int, Optional[int]] = {}
        sources = set()
        for pid in self._procs:
            # The OS's mark as last read (at the last sample the process was
            # alive for), else the largest sample.
            peak = self._os_peak.get(pid)
            if peak is not None:
                sources.add(PEAK_OS)
            else:
                peak = self._sampled_peak.get(pid)
                if peak is not None:
                    sources.add(PEAK_SAMPLED)
            peaks[pid] = peak
        by_role: Dict[str, Optional[int]] = {}
        for role, pids in self._roles.items():
            values = [peaks[p] for p in pids if peaks.get(p) is not None]
            by_role[role] = max(values) if values else None
        known = [v for v in peaks.values() if v is not None]
        source = (PEAK_MIXED if len(sources) > 1
                  else next(iter(sources)) if sources else PEAK_SAMPLED)
        return Footprint(
            processes_by_role={r: len(p) for r, p in self._roles.items()},
            peak_rss_bytes_total=self._total_peak,
            peak_rss_bytes_by_role=by_role,
            peak_rss_bytes_sum=sum(known) if known else None,
            peak_source=source,
            samples=self._samples,
            interval_s=self.interval_s,
            probe=f"psutil {self._psutil.__version__} on {sys.platform}",
        )
