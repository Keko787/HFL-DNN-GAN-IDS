"""Sprint 2 — multi-process orchestrator.

Brings up an AVN-shaped topology by launching one Python subprocess
per role (cluster + N mules + M devices) using the entry-point modules
in this package.

Lifecycle:

1. :meth:`MultiProcessOrchestrator.start_cluster` writes a
   :class:`ClusterConfig` JSON to a temp dir, spawns the cluster
   subprocess, and waits for the cluster to bind its dock port (read
   back via the ``--port-out`` file).
2. :meth:`start_mules` spawns one mule subprocess per :class:`MuleConfig`,
   each pointed at the cluster's now-known dock port. Waits for each
   mule's RF port via the same ``--port-out`` mechanism.
3. :meth:`start_devices` spawns one device subprocess per
   :class:`DeviceConfig`, each pointed at the right mule's now-known
   RF port.
4. :meth:`shutdown_all` sends SIGTERM to every subprocess and waits
   (with timeout) for clean exit; falls back to SIGKILL.

Used by chunk N's end-to-end happy-path test and chunk O's fault tests.

Cross-platform notes:

* On Windows, ``Popen.send_signal(SIGTERM)`` translates to
  ``TerminateProcess`` which is more like ``SIGKILL`` — we don't get a
  graceful-shutdown window. The processes' service loops handle that
  fine because socket teardown is in ``finally`` blocks.
* Subprocess stdout/stderr are inherited (passes through to the parent
  test's terminal). For programmatic capture, set ``capture_output=True``.
"""

from __future__ import annotations

import collections
import logging
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

from .config import (
    ClusterConfig,
    DeviceConfig,
    MuleConfig,
    TopologyConfig,
    cluster_config_to_json,
    device_config_to_json,
    mule_config_to_json,
)

log = logging.getLogger("hermes.processes.orchestrator")


# L-H3 / L-L8 — keep the most recent N stderr lines in a ring buffer
# so an :class:`OrchestratorError` can show *what the process said*
# instead of forcing the caller to grep through ``proc.stderr.read()``.
# Lines are drained off the pipe in a background thread to prevent the
# OS pipe buffer from filling and deadlocking the child.
_STDERR_TAIL_LINES = 200


class _StderrDrainer:
    """Background thread that reads ``proc.stderr`` line-by-line.

    Each line is forwarded to the parent's stderr (so test logs still
    show the child's output) AND retained in a fixed-size ring buffer
    so :meth:`tail` can return the last N lines on demand. Without this
    drain, a child producing >64 KiB of output would block on its
    next ``write(stderr)`` call — the classic ``stdout=PIPE`` deadlock
    documented in CPython's subprocess docs.
    """

    def __init__(self, name: str, stream, *, mirror_to_parent: bool = True):
        self.name = name
        self._stream = stream
        self._mirror = mirror_to_parent
        self._buf: Deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, name=f"stderr-drain-{name}", daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        try:
            for raw in iter(self._stream.readline, b""):
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                except Exception:
                    line = repr(raw)
                with self._lock:
                    self._buf.append(line)
                if self._mirror:
                    sys.stderr.write(f"[{self.name}] {line}\n")
                    sys.stderr.flush()
        except Exception:  # pragma: no cover — drainer should never crash
            pass

    def tail(self, n: int = _STDERR_TAIL_LINES) -> str:
        with self._lock:
            return "\n".join(list(self._buf)[-n:])


@dataclass
class ProcessHandle:
    """Tracks one subprocess we launched."""

    name: str
    proc: subprocess.Popen
    config_path: Path
    port_path: Optional[Path] = None
    actual_port: Optional[int] = None
    stderr_drainer: Optional[_StderrDrainer] = None

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def returncode(self) -> Optional[int]:
        return self.proc.poll()

    def stderr_tail(self, n: int = _STDERR_TAIL_LINES) -> str:
        """L-L8: most recent ``n`` lines of stderr from this subprocess."""
        if self.stderr_drainer is None:
            return ""
        return self.stderr_drainer.tail(n)


class OrchestratorError(RuntimeError):
    """Raised on startup / shutdown failures.

    L-L8: when raised because a subprocess failed, the offending
    process's last 200 stderr lines are appended to the message so the
    caller doesn't have to dig.
    """


class MultiProcessOrchestrator:
    """Launches and manages a HERMES topology of subprocesses.

    Typical usage::

        topo = TopologyConfig(...)
        orch = MultiProcessOrchestrator(topo)
        orch.start_all(timeout=30.0)
        try:
            ...  # exercise the topology
        finally:
            orch.shutdown_all()
    """

    def __init__(
        self,
        topology: TopologyConfig,
        *,
        python_executable: Optional[str] = None,
        capture_output: bool = False,
    ) -> None:
        # L-H4 / L-M4: validate up front. ``validate()`` populates
        # ``topology.device_to_mule`` so later steps read assignment
        # from a single source of truth instead of mutating MuleConfig
        # mid-flight.
        topology.validate()
        self.topology = topology
        self.python = python_executable or sys.executable
        self.capture_output = capture_output

        self._tmpdir = Path(tempfile.mkdtemp(prefix="hermes_orch_"))
        self._cluster: Optional[ProcessHandle] = None
        self._mules: Dict[str, ProcessHandle] = {}
        self._devices: Dict[str, ProcessHandle] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def tmpdir(self) -> Path:
        return self._tmpdir

    @property
    def cluster_handle(self) -> Optional[ProcessHandle]:
        return self._cluster

    @property
    def mule_handles(self) -> Dict[str, ProcessHandle]:
        return dict(self._mules)

    @property
    def device_handles(self) -> Dict[str, ProcessHandle]:
        return dict(self._devices)

    def start_all(self, *, timeout: float = 30.0) -> None:
        """Convenience: start cluster → mules → devices in dependency order."""
        deadline = time.time() + timeout
        self.start_cluster(timeout=max(0.5, deadline - time.time()))
        self.start_mules(timeout=max(0.5, deadline - time.time()))
        self.start_devices(timeout=max(0.5, deadline - time.time()))

    def start_cluster(self, *, timeout: float = 10.0) -> int:
        """Spawn the cluster subprocess; return the bound dock port."""
        if self._cluster is not None:
            raise OrchestratorError("cluster already started")

        cfg = self.topology.cluster
        # Build seed_devices from the validated assignment map.
        seed_devices = [
            {
                "device_id": d.device_id,
                "position": list(d.position),
                "assigned_mule": self.topology.mule_for(d.device_id),
            }
            for d in self.topology.devices
        ]
        cfg = ClusterConfig(
            **{
                **asdict(cfg),
                "expected_mules": (
                    cfg.expected_mules or [m.mule_id for m in self.topology.mules]
                ),
                "seed_devices": seed_devices,
            }
        )

        config_path = self._tmpdir / "cluster.json"
        config_path.write_text(cluster_config_to_json(cfg), encoding="utf-8")
        port_path = self._tmpdir / "cluster.port"

        proc, drainer = self._spawn(
            name="cluster",
            module="hermes.processes.cluster",
            config_path=config_path,
            port_path=port_path,
        )
        handle = ProcessHandle(
            name="cluster", proc=proc,
            config_path=config_path, port_path=port_path,
            stderr_drainer=drainer,
        )
        self._cluster = handle

        port = self._wait_for_port(handle, timeout=timeout)
        handle.actual_port = port
        log.info("orchestrator: cluster up on port %d", port)
        return port

    def start_mules(self, *, timeout: float = 10.0) -> Dict[str, int]:
        """Spawn each mule subprocess; return mule_id → bound RF port."""
        if self._cluster is None or self._cluster.actual_port is None:
            raise OrchestratorError("start_cluster must succeed first")

        ports: Dict[str, int] = {}
        for mcfg in self.topology.mules:
            if mcfg.mule_id in self._mules:
                continue
            # Auto-fill the cluster's actual port + expected_devices
            # from the validated assignment.
            expected_devs = self.topology.devices_of(mcfg.mule_id)
            populated = MuleConfig(
                **{
                    **asdict(mcfg),
                    "dock_port": self._cluster.actual_port,
                    "expected_devices": expected_devs,
                }
            )
            cfg_path = self._tmpdir / f"mule-{mcfg.mule_id}.json"
            cfg_path.write_text(mule_config_to_json(populated), encoding="utf-8")
            port_path = self._tmpdir / f"mule-{mcfg.mule_id}.port"

            proc, drainer = self._spawn(
                name=f"mule-{mcfg.mule_id}",
                module="hermes.processes.mule",
                config_path=cfg_path,
                port_path=port_path,
            )
            handle = ProcessHandle(
                name=f"mule-{mcfg.mule_id}", proc=proc,
                config_path=cfg_path, port_path=port_path,
                stderr_drainer=drainer,
            )
            self._mules[mcfg.mule_id] = handle

            port = self._wait_for_port(handle, timeout=timeout)
            handle.actual_port = port
            ports[mcfg.mule_id] = port
            log.info("orchestrator: mule %s up on RF port %d", mcfg.mule_id, port)
        return ports

    def start_devices(self, *, timeout: float = 10.0) -> None:
        """Spawn each device subprocess pointed at its mule's RF port."""
        if not self._mules:
            raise OrchestratorError("start_mules must succeed first")

        for dcfg in self.topology.devices:
            if dcfg.device_id in self._devices:
                continue
            mid = self.topology.mule_for(dcfg.device_id)
            mule_handle = self._mules.get(mid)
            if mule_handle is None or mule_handle.actual_port is None:
                raise OrchestratorError(
                    f"device {dcfg.device_id} maps to unknown mule {mid}"
                )
            populated = DeviceConfig(
                **{
                    **asdict(dcfg),
                    "mule_rf_port": mule_handle.actual_port,
                    "position": tuple(dcfg.position),
                }
            )
            cfg_path = self._tmpdir / f"device-{dcfg.device_id}.json"
            cfg_path.write_text(device_config_to_json(populated), encoding="utf-8")
            proc, drainer = self._spawn(
                name=f"device-{dcfg.device_id}",
                module="hermes.processes.device",
                config_path=cfg_path,
                port_path=None,
            )
            handle = ProcessHandle(
                name=f"device-{dcfg.device_id}", proc=proc,
                config_path=cfg_path, port_path=None,
                stderr_drainer=drainer,
            )
            self._devices[dcfg.device_id] = handle
            log.info("orchestrator: device %s spawned (PID %d)",
                     dcfg.device_id, proc.pid)

        # No port-back-channel for devices — give them a moment to
        # establish their TCP client connections.
        time.sleep(0.3)

    def shutdown_all(
        self,
        *,
        timeout: float = 15.0,
        cleanup_tmpdir: bool = True,
    ) -> None:
        """Send SIGTERM to every subprocess; SIGKILL anything that doesn't die.

        L-M5: default timeout is generous enough to swallow a mule that
        is mid-mission when the signal arrives — the supervisor loop
        only checks the stop event between missions, and a Pass-2
        delivery in flight can take several seconds. Tests that want a
        faster shutdown can pass an explicit ``timeout`` argument.

        Chunk M: pass ``cleanup_tmpdir=False`` if you want to inspect
        the per-process JSONL event logs after every subprocess has
        exited but before the tmpdir is removed. Call :meth:`cleanup`
        afterwards to release the directory.
        """
        all_handles: List[ProcessHandle] = (
            list(self._devices.values())
            + list(self._mules.values())
            + ([self._cluster] if self._cluster else [])
        )
        # Devices first (they need their mules / cluster up to ack a
        # final round), mules next, cluster last — reverse-startup order.
        for h in all_handles:
            if h.is_alive():
                try:
                    h.proc.terminate()
                except Exception:
                    pass

        deadline = time.time() + timeout
        for h in all_handles:
            remaining = max(0.0, deadline - time.time())
            try:
                h.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log.warning("orchestrator: %s did not exit cleanly; killing", h.name)
                try:
                    h.proc.kill()
                except Exception:
                    pass
                try:
                    h.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    pass

        if not cleanup_tmpdir:
            return

        # Cleanup tmp config + port + jsonl files.
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the orchestrator's tmpdir (configs, port files, JSONL logs).

        Idempotent. Called by :meth:`shutdown_all` by default; tests that
        passed ``cleanup_tmpdir=False`` to read JSONL events before
        teardown should call this after their assertions.
        """
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Context-manager + signal-handler conveniences (L-L4)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "MultiProcessOrchestrator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown_all()

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers that trigger shutdown_all.

        L-L4: handy for scripts that drive the orchestrator from the
        main thread (e.g., ``python -m hermes.scripts.run_demo``). Tests
        usually don't need this since pytest's own teardown unwinds the
        ``finally`` block. Best-effort — silently skips on platforms
        where the signal isn't installable from the current thread.
        """
        def _handle(_signum, _frame):
            log.info("orchestrator received shutdown signal")
            self.shutdown_all()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle)
            except (ValueError, OSError):  # pragma: no cover
                pass

    def all_alive(self) -> bool:
        if self._cluster is None or not self._cluster.is_alive():
            return False
        if not all(h.is_alive() for h in self._mules.values()):
            return False
        if not all(h.is_alive() for h in self._devices.values()):
            return False
        return True

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _spawn(
        self,
        *,
        name: str,
        module: str,
        config_path: Path,
        port_path: Optional[Path],
    ) -> tuple:
        """Launch a subprocess, return (Popen, stderr_drainer-or-None).

        L-H3: when ``capture_output`` is True we always drain stderr in
        a background thread (otherwise the child blocks once the OS
        pipe buffer fills, then ``shutdown_all`` hangs on ``proc.wait``).
        ``stdout`` is sent to ``DEVNULL`` since nothing currently uses
        it — chunk M emits structured observability events to per-process
        JSONL files under :attr:`tmpdir`.
        """
        argv = [self.python, "-m", module, "--config", str(config_path)]
        if port_path is not None:
            argv += ["--port-out", str(port_path)]
        # Chunk M: every spawned process gets the orchestrator's tmpdir
        # as its run-dir. Per-process JSONL filenames are role-prefixed
        # (cluster-<id>.jsonl, mule-<id>.jsonl, device-<id>.jsonl) so a
        # single tmpdir holds the full run.
        argv += ["--run-dir", str(self._tmpdir)]

        kwargs = {}
        drainer = None
        if self.capture_output:
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.PIPE
        # else inherit: subprocess output flows to the parent's stdio.

        log.info("orchestrator: launching %s: %s", name, " ".join(argv))
        proc = subprocess.Popen(argv, **kwargs)
        if self.capture_output and proc.stderr is not None:
            drainer = _StderrDrainer(name, proc.stderr, mirror_to_parent=False)
        return proc, drainer

    def _wait_for_port(self, handle: ProcessHandle, *, timeout: float) -> int:
        """Poll the port-out file until it appears, or the process dies.

        L-L8: when a startup error fires, the offending subprocess's
        recent stderr is appended to the message so the caller doesn't
        have to dig through pipes themselves.
        """
        if handle.port_path is None:
            raise OrchestratorError(f"{handle.name} has no port-out file")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not handle.is_alive():
                rc = handle.returncode()
                raise OrchestratorError(
                    f"{handle.name} exited with code {rc} before binding a "
                    f"port\n{self._format_stderr_tail(handle)}"
                )
            if handle.port_path.exists():
                try:
                    text = handle.port_path.read_text(encoding="utf-8").strip()
                    if text:
                        return int(text)
                except (OSError, ValueError):
                    pass
            time.sleep(0.05)
        raise OrchestratorError(
            f"{handle.name} did not write port file within {timeout}s\n"
            f"{self._format_stderr_tail(handle)}"
        )

    @staticmethod
    def _format_stderr_tail(handle: ProcessHandle, n: int = 60) -> str:
        tail = handle.stderr_tail(n)
        if not tail:
            return f"(no stderr captured for {handle.name}; "\
                   "pass capture_output=True to MultiProcessOrchestrator to enable)"
        return f"--- last stderr from {handle.name} ---\n{tail}\n--- end stderr ---"

