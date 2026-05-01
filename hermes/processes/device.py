"""Sprint 2 — device-process entry point + service loop.

Run with::

    python -m hermes.processes.device --config /path/to/device.json

The device process:

1. Reads :class:`DeviceConfig` from JSON.
2. Stands up :class:`TCPRFLinkClient` connecting outbound to the
   mule's RF port.
3. Builds :class:`ClientMission` with a stub local-train callback
   (Sprint 2 doesn't run a real model on the device; chunk N's e2e
   test asserts the wiring, not the training).
4. Sets state to ``FL_OPEN``.
5. Loops :meth:`ClientMission.serve_once` until ``n_serves`` calls
   land or shutdown signal arrives.

Sprint 1.5 H6 wired ``train_offline`` to fire automatically after a
Pass-2 delivery push, so the device has a prepared Δθ ready for the
next Pass-1 visit without any external scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from hermes.mission import ClientMission, LocalTrainResult
from hermes.observability import (
    JsonEventEmitter,
    MetricsRegistry,
    NullEventEmitter,
)
from hermes.transport import TCPRFLinkClient
from hermes.types import DeviceID, FLState

from .config import DeviceConfig, device_config_from_json

log = logging.getLogger("hermes.processes.device")


def _stub_train_factory(seed: int = 0):
    """Sprint 2 stub: deterministic noisy delta on top of the pushed θ.

    Real model training lives in the AC-GAN code path; chunk N just
    needs the wire-level handshake to round-trip. Sprint 6 swaps this
    for the real training callback.
    """
    rng = np.random.default_rng(seed)

    def _train(theta, synth):
        delta = [
            w + rng.normal(0.0, 0.01, size=w.shape).astype(w.dtype)
            for w in theta
        ]
        return LocalTrainResult(
            delta_theta=delta,
            num_examples=int(rng.integers(4, 16)),
            accuracy=float(rng.uniform(0.7, 0.9)),
            auc=float(rng.uniform(0.7, 0.9)),
            loss=float(rng.uniform(0.1, 0.3)),
            theta_after=delta,
        )
    return _train


class DeviceService:
    """Lifecycle holder for a device-process service loop."""

    def __init__(
        self,
        cfg: DeviceConfig,
        *,
        events: Optional[JsonEventEmitter] = None,
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        self.cfg = cfg
        self._stop_event = threading.Event()

        self.events = events or NullEventEmitter(role="device", node_id=cfg.device_id)
        self.metrics = metrics or MetricsRegistry()

        # L-M3: Python's built-in ``hash()`` is randomized per process
        # (PYTHONHASHSEED) so two subprocess runs of the same device_id
        # would diverge — breaking the reproducibility this seed is
        # supposed to provide. SHA-256 → first 4 bytes is stable across
        # processes / interpreter versions / platforms. L-L5: 31-bit
        # truncation gives ~2 billion seed buckets; collisions across
        # devices in a topology are vanishingly unlikely (birthday-bound
        # ~46 K device IDs before 50% collision risk), well above any
        # realistic deployment.
        digest = hashlib.sha256(cfg.device_id.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big") % (2**31)

        self.rf = TCPRFLinkClient(
            device_id=DeviceID(cfg.device_id),
            host=cfg.mule_rf_host,
            port=cfg.mule_rf_port,
        )
        self.client = ClientMission(
            device_id=DeviceID(cfg.device_id),
            rf=self.rf,
            local_train=_stub_train_factory(seed),
            solicit_timeout_s=2.0,
            disc_push_timeout_s=10.0,
        )
        self.client.set_state(FLState.FL_OPEN)

        self.events.emit(
            "device_ready",
            mule_rf_host=self.cfg.mule_rf_host,
            mule_rf_port=self.cfg.mule_rf_port,
            position=list(self.cfg.position),
            n_serves=self.cfg.n_serves,
        )

    def request_stop(self) -> None:
        self._stop_event.set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        """Service loop — runs until ``request_stop`` or n_serves reached."""
        log.info(
            "device %s ready: RF client to %s:%d",
            self.cfg.device_id, self.cfg.mule_rf_host, self.cfg.mule_rf_port,
        )

        n_serves = self.cfg.n_serves
        served = 0
        while not self._stop_event.is_set():
            if n_serves is not None and served >= n_serves:
                log.info(
                    "device %s: served %d times, exiting",
                    self.cfg.device_id, served,
                )
                break
            try:
                outcome = self.client.serve_once()
                if outcome is not None:
                    served += 1
                    log.info(
                        "device %s: served, outcome=%s",
                        self.cfg.device_id, outcome.value,
                    )
                    self.events.emit("device_served", outcome=outcome.value)
                    self.metrics.increment("serves_completed")
                    self.metrics.increment(f"serves_outcome_{outcome.value}")
            except Exception as e:
                log.exception("device %s: serve_once raised", self.cfg.device_id)
                self.events.emit("device_serve_failed", reason=repr(e))
                self.metrics.increment("serves_failed")

        log.info("device %s service loop exiting", self.cfg.device_id)

    def shutdown(self) -> None:
        self.request_stop()
        try:
            self.rf.close()
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

def _install_signal_handlers(svc: DeviceService) -> None:
    def _handle(_signum, _frame):
        log.info("device received shutdown signal")
        svc.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):  # pragma: no cover
            pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes.processes.device")
    parser.add_argument("--config", required=True, type=Path)
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

    cfg = device_config_from_json(args.config.read_text(encoding="utf-8"))

    events: Optional[JsonEventEmitter] = None
    if args.run_dir is not None:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        events = JsonEventEmitter(
            args.run_dir / f"device-{cfg.device_id}.jsonl",
            role="device",
            node_id=cfg.device_id,
        )

    svc = DeviceService(cfg, events=events)
    _install_signal_handlers(svc)

    try:
        svc.run()
    finally:
        svc.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
