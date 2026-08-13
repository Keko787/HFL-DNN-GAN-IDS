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
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from hermes.mule import MuleSupervisor, MuleSupervisorError
from hermes.observability import (
    JsonEventEmitter,
    MetricsRegistry,
    NullEventEmitter,
)
from hermes.transport import TCPDockLinkClient, TCPRFLinkServer
from hermes.types import DeviceID, MuleID

from .config import MuleConfig, mule_config_from_json

log = logging.getLogger("hermes.processes.mule")


def _build_target_selector(cfg: MuleConfig):
    """EX-4.2 — build the S3.5 RL target selector when configured (arm H2).

    Returns ``None`` (deterministic distance ranking = arm H1) unless
    ``cfg.use_rl_selector`` is set. With ``selector_weights_path`` it loads a
    trained DDQN (.npz); otherwise it builds a random-init selector — a
    plumbing smoke, NOT paper-grade (train weights via
    ``experiments.exp3.train_a4`` for a real H2-vs-H1 comparison).
    """
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

        self.events = events or NullEventEmitter(role="mule", node_id=cfg.mule_id)
        self.metrics = metrics or MetricsRegistry()

        # 1. Mule's RF server — devices connect here.
        self.rf = TCPRFLinkServer(host=cfg.rf_host, port=cfg.rf_port)
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
        self.supervisor = MuleSupervisor(
            mule_id=MuleID(cfg.mule_id),
            rf=self.rf,
            dock=self.dock,
            session_ttl_s=cfg.session_ttl_s,
            rf_range_m=cfg.rf_range_m,
            target_selector=_build_target_selector(cfg),
            **sup_kwargs,
        )

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
        )

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
        ok = self._wait_with_stop(
            self.supervisor.wait_for_initial_dock,
            total_timeout=30.0,
        )
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
            self.events.emit("mission_started", mission_index=completed)
            try:
                result = self.supervisor.run_one_mission()
                completed += 1
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
                    self.events.emit(
                        "mission_empty", mission_round=result.mission_round,
                    )
                    self.metrics.increment("missions_empty")
                # EX-4.0 instrumentation — surface the Pass-1 aggregation
                # ledger so an integrated-experiment consumer can compute
                # update-yield / round-close-rate from the real event
                # stream. ``report`` is the MissionRoundCloseReport from
                # ``close_round`` (populated by both the single-pass and
                # two-pass paths); ``pass_1_queue`` carries the scheduled
                # contact membership. All three fields are additive and
                # optional per the observability schema policy, so adding
                # them does not break existing consumers.
                report = getattr(result, "report", None)
                if report is not None:
                    pass_1_updates = report.counts()[0]  # CLEAN collections
                    pass_1_clean_devices = [
                        str(line.device_id)
                        for line in report.lines
                        if line.outcome.is_on_time()
                    ]
                else:
                    pass_1_updates = None
                    pass_1_clean_devices = None
                pass_1_scheduled = (
                    sum(len(c.devices) for c in result.pass_1_queue) or None
                )
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
                )
            except MuleSupervisorError as e:
                log.error("mule %s: supervisor error: %s", self.cfg.mule_id, e)
                self.events.emit("mission_failed", reason=str(e), kind="supervisor")
                self.metrics.increment("mission_failures")
                break
            except Exception as e:
                log.exception("mule %s: unexpected mission failure", self.cfg.mule_id)
                self.events.emit("mission_failed", reason=repr(e), kind="unexpected")
                self.metrics.increment("mission_failures")
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
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
