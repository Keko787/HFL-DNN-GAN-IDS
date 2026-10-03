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
   land or shutdown signal arrives. If the RF link drops, the loop
   re-dials the mule with backoff instead of spinning (Amendment 10).

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
import time
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


def _build_local_train(cfg: DeviceConfig, seed: int):
    """Select the device's local-training callback.

    EX-4.1: when ``cfg.train_shard_path`` is set, build a *real*
    ``local_train`` that fits the canonical DNN-IDS on this device's
    CICIOT shard (lazy import so the default stub path never pulls in
    TensorFlow / the experiments package). Otherwise fall back to the
    Sprint-2 noise stub — the multi-process integration tests rely on it.
    """
    if getattr(cfg, "train_shard_path", None):
        # Layer note: hermes is the core library; this reaches up into the
        # experiments package only on the opt-in real-model path, and only
        # in a spawned device subprocess whose CWD is the repo root.
        from experiments.exp4.model_task import load_xy, make_local_train_fn

        X, y = load_xy(cfg.train_shard_path)
        log.info(
            "device %s: real DNN-IDS local_train over shard %s "
            "(rows=%d, input_dim=%s, epochs=%d)",
            cfg.device_id, cfg.train_shard_path, len(y),
            cfg.input_dim, cfg.local_epochs,
        )
        arch = getattr(cfg, "model_arch", None)
        return make_local_train_fn(
            X, y,
            input_dim=cfg.input_dim,
            epochs=cfg.local_epochs,
            batch_size=cfg.local_batch_size,
            seed=seed,
            fedprox_rho=float(getattr(cfg, "fedprox_rho", 0.0) or 0.0),
            # Exp 5 addendum (Study 5.12): the architecture, only when set.
            **({} if arch is None else {"arch": arch}),
        )
    return _stub_train_factory(seed)


class DeviceService:
    """Lifecycle holder for a device-process service loop."""

    # Amendment 10 (P-02): the wait before each re-dial of a lost RF link.
    # It starts at the initial step and doubles after every failed dial, up
    # to the cap, so a dead link costs one wake-up per step instead of a busy
    # loop. A link that a re-dial brought back but that is lost again within
    # the hold time did not hold, and the next re-dial goes on from twice the
    # last wait instead of starting over. Two devices registering under one
    # id on one mule (a stale device of another trial that reached a mule
    # without a link token, and that mule's own device) evict each other at
    # every re-dial; this keeps that from happening every few seconds for as
    # long as both run. The hold is three times the cap, so that even at the
    # cap one round of such evictions (the other side's wait plus the up to
    # 2 s a solicit poll takes to notice) counts as not holding.
    # Class attributes so a test can shorten them on one instance.
    _RECONNECT_INITIAL_S: float = 0.5
    _RECONNECT_MAX_S: float = 10.0
    _RECONNECT_HOLD_S: float = 30.0

    def __init__(
        self,
        cfg: DeviceConfig,
        *,
        events: Optional[JsonEventEmitter] = None,
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        self.cfg = cfg
        self._stop_event = threading.Event()
        # Amendment 10: when the last successful re-dial happened (monotonic
        # seconds; None before any) and the wait that preceded it.
        self._relinked_at: Optional[float] = None
        self._relink_wait_s: float = 0.0

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

        # EX-4.1: build the training callback FIRST. On the real-model path
        # this imports TensorFlow and constructs the Keras model, which can
        # take several seconds; doing it before the RF client connects means
        # the device only registers on the mule's RF link once it is truly
        # ready to serve a contact, so the mule's ``wait_for_devices``
        # barrier doubles as a serve-readiness barrier and the first solicit
        # can't race an unbuilt model.
        local_train = _build_local_train(cfg, seed)
        self.rf = TCPRFLinkClient(
            device_id=DeviceID(cfg.device_id),
            host=cfg.mule_rf_host,
            port=cfg.mule_rf_port,
            # FeRRy Phase 3 (critic B1): off, solicits are answered in arrival
            # order, as in every recorded run; the ferry topology turns it on.
            newest_solicit_only=bool(getattr(cfg, "newest_solicit_only", False)),
            # Amendment 10: None, as in every recorded run, sends no token.
            link_token=getattr(cfg, "rf_link_token", None),
        )
        self.client = ClientMission(
            device_id=DeviceID(cfg.device_id),
            rf=self.rf,
            local_train=local_train,
            solicit_timeout_s=2.0,
            disc_push_timeout_s=10.0,
            contact_reliability=getattr(cfg, "contact_reliability", None),
            # Distinct from the training seed so contact luck != data shuffle.
            contact_rng_seed=(seed ^ 0x9E3779B9),
        )
        self.client.set_state(FLState.FL_OPEN)

        self.events.emit(
            "device_ready",
            mule_rf_host=self.cfg.mule_rf_host,
            mule_rf_port=self.cfg.mule_rf_port,
            position=list(self.cfg.position),
            n_serves=self.cfg.n_serves,
            # The proximal coefficient this process received (audit #15), so
            # a trace shows it reached the device. The stub trainer ignores it.
            fedprox_rho=float(getattr(self.cfg, "fedprox_rho", 0.0) or 0.0),
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
            # Amendment 10 (P-02): with its link down, serve_once returns None
            # at once, so this loop used to spin a core until the process was
            # killed, and the device never came back. Re-dial instead.
            if not self._link_up():
                if self._reconnect_with_backoff():
                    continue
                break
            try:
                outcome = self.client.serve_once()
                if outcome is not None:
                    served += 1
                    log.info(
                        "device %s: served, outcome=%s",
                        self.cfg.device_id, outcome.value,
                    )
                    # The pass and round let a trace tell a Pass-1 collect from
                    # a Pass-2 delivery; both land here. None when no push came.
                    self.events.emit(
                        "device_served",
                        outcome=outcome.value,
                        mission_round=getattr(self.client, "last_push_round", None),
                        pass_kind=getattr(self.client, "last_push_pass", None),
                    )
                    self.metrics.increment("serves_completed")
                    self.metrics.increment(f"serves_outcome_{outcome.value}")
            except Exception as e:
                log.exception("device %s: serve_once raised", self.cfg.device_id)
                self.events.emit("device_serve_failed", reason=repr(e))
                self.metrics.increment("serves_failed")

        log.info("device %s service loop exiting", self.cfg.device_id)

    def _link_up(self) -> bool:
        # A link that has no notion of being down (a test double) counts as up.
        return bool(getattr(self.rf, "connected", True))

    def _reconnect_with_backoff(self) -> bool:
        """Re-dial the mule until the link is back (True) or a stop is requested (False).

        Every dial waits its backoff step first, so even a link that drops
        again right after each dial cannot spin the loop, and a link that a
        re-dial brought back but that did not hold for ``_RECONNECT_HOLD_S``
        resumes the backoff where it stopped. A re-dial succeeds only once
        the mule acknowledges the registration (``TCPRFLinkClient.reconnect``),
        so another process that took the port after the mule exited cannot
        pass for it. Only a successful re-dial is recorded
        (``device_reconnected`` with the dials and the wall seconds the link
        was down, counter ``rf_reconnects``): the link also drops at the end
        of every trial, when the mule exits, and that must leave the
        device's trace as it was.
        """
        log.warning(
            "device %s: RF link to %s:%d lost; re-dialling with backoff",
            self.cfg.device_id, self.cfg.mule_rf_host, self.cfg.mule_rf_port,
        )
        lost_at = time.monotonic()
        cap = float(self._RECONNECT_MAX_S)
        delay = float(self._RECONNECT_INITIAL_S)
        if (
            self._relinked_at is not None
            and lost_at - self._relinked_at < float(self._RECONNECT_HOLD_S)
        ):
            delay = min(max(delay, self._relink_wait_s * 2.0), cap)
        attempts = 0
        while not self._stop_event.wait(delay):
            attempts += 1
            try:
                self.rf.reconnect()
            except Exception as e:
                delay = min(delay * 2.0, cap)
                log.info(
                    "device %s: re-dial %d failed (%s); next in %.1fs",
                    self.cfg.device_id, attempts, e, delay,
                )
                continue
            self._relinked_at = time.monotonic()
            self._relink_wait_s = delay
            down_s = self._relinked_at - lost_at
            log.info(
                "device %s: RF link re-established after %d dial(s), %.1fs down",
                self.cfg.device_id, attempts, down_s,
            )
            self.events.emit(
                "device_reconnected", attempts=attempts, down_s=round(down_s, 3),
            )
            self.metrics.increment("rf_reconnects")
            return True
        return False

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
