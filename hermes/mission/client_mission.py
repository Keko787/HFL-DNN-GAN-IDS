"""``ClientMission`` — edge-device FL client.

Responsibilities per design §5.3:

* ``FLState`` state machine (BUSY / UNAVAILABLE / FL_OPEN).
* Post-round utility computation using ``performance_score`` +
  ``diversity_adjusted`` + ``utility`` formulas.
* FL_READY_ADV payload building.
* RF beacon emitter (stubbed via ``beacon_fn`` — a shim in Phase 2, real
  radio in Phase 6).
* Wrap an existing local training step (the AC-GAN discriminator loop
  lives outside this module; we call back into it via ``local_train``).

``local_train`` is a Protocol-shaped callable: it is given θ_disc plus a
synth batch and returns a ``LocalTrainResult`` with Δθ, num_examples,
and evaluation metrics the utility uses.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

import numpy as np

from hermes.transport import RFLink, RFLinkError
from hermes.types import (
    DeviceID,
    DiscPush,
    FLOpenSolicit,
    FLReadyAdv,
    FLState,
    GradientSubmission,
    MissionOutcome,
    Weights,
)

from .utility import diversity_adjusted, performance_score, utility

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Local training contract
# --------------------------------------------------------------------------- #

@dataclass
class LocalTrainResult:
    """Return value of one local training step on the device.

    ``delta_theta`` is the Δ actually shipped back to the mule. The
    device may send raw weights or gradients — the convention is fixed
    by the AC-GAN adapter in Phase 6; from this module's perspective it
    is just a list of numpy arrays.
    """

    delta_theta: Weights
    num_examples: int
    accuracy: float = 0.0
    auc: float = 0.0
    loss: float = 0.0
    # post-step local weights used for cosine-to-global diversity term
    theta_after: Weights = field(default_factory=list)


class LocalTrainFn(Protocol):
    """Signature of the callable the device plugs in."""

    def __call__(
        self, theta_disc: Weights, synth_batch: List[np.ndarray]
    ) -> LocalTrainResult: ...


# --------------------------------------------------------------------------- #
# Beacon emitter shim
# --------------------------------------------------------------------------- #

BeaconFn = Callable[[FLReadyAdv], None]


def _null_beacon(_adv: FLReadyAdv) -> None:
    """Default beacon sink — drops the advertisement.

    Replaced in Phase 6 by a radio shim. In Phase 2 tests a list-append
    closure is used so the test can assert the beacon fired.
    """


# --------------------------------------------------------------------------- #
# ClientMission
# --------------------------------------------------------------------------- #

class ClientMission:
    """Edge-device FL client. One instance per device.

    Threading model: one worker loop calls ``serve_once`` in a cycle;
    ``set_state`` and ``last_utility`` are safe to call from other threads
    (e.g. a local sensor process flipping the state to BUSY).
    """

    def __init__(
        self,
        *,
        device_id: DeviceID,
        rf: RFLink,
        local_train: LocalTrainFn,
        beacon_fn: BeaconFn = _null_beacon,
        w1: float = 0.7,
        w2: float = 0.3,
        fl_threshold: float = 0.0,
        solicit_timeout_s: float = 30.0,
        disc_push_timeout_s: float = 30.0,
    ) -> None:
        self.device_id = device_id
        self.rf = rf
        self.local_train = local_train
        self.beacon_fn = beacon_fn
        self.w1 = w1
        self.w2 = w2
        self.fl_threshold = fl_threshold
        self.solicit_timeout_s = solicit_timeout_s
        self.disc_push_timeout_s = disc_push_timeout_s

        self._lock = threading.RLock()
        self._state: FLState = FLState.UNAVAILABLE

        # post-round snapshot used for the *next* advertisement
        self._last_performance: float = 0.0
        self._last_diversity: float = 0.0
        self._last_utility: float = 0.0
        self._last_outcome: Optional[MissionOutcome] = None

        # register self with the loopback (real RF would just listen on addr)
        if hasattr(rf, "register_device"):
            rf.register_device(device_id)  # type: ignore[attr-defined]

    # ---------------------------------------------- public state API

    def set_state(self, new_state: FLState) -> None:
        with self._lock:
            log.debug(
                "device=%s state %s -> %s",
                self.device_id, self._state.value, new_state.value,
            )
            self._state = new_state

    @property
    def state(self) -> FLState:
        with self._lock:
            return self._state

    @property
    def last_utility(self) -> float:
        with self._lock:
            return self._last_utility

    # ---------------------------------------------- advertisement

    def build_ready_adv(self) -> FLReadyAdv:
        """Build the next ``FLReadyAdv`` from the most recent round's metrics."""
        with self._lock:
            return FLReadyAdv(
                device_id=self.device_id,
                state=self._state,
                performance_score=self._last_performance,
                diversity_adjusted=self._last_diversity,
                utility=self._last_utility,
                last_round_outcome=self._last_outcome,
                issued_at=time.time(),
            )

    def emit_beacon(self) -> FLReadyAdv:
        """Fire one beacon burst — used by the opportunistic path."""
        adv = self.build_ready_adv()
        try:
            self.beacon_fn(adv)
        except Exception:  # pragma: no cover
            log.exception("beacon_fn raised; continuing")
        return adv

    # ---------------------------------------------- session driver

    def serve_once(self) -> Optional[MissionOutcome]:
        """Wait for a solicit, run one round if FL_OPEN, update utility.

        Returns the outcome tag, or ``None`` if:
          * the solicit timed out, or
          * the device was not FL_OPEN (we still *replied* with the
            current state so the mule can record a contact).
        """
        # Wait for a mule to ping us
        try:
            solicit = self.rf.recv_open_solicit(
                self.device_id, timeout=self.solicit_timeout_s
            )
        except RFLinkError:
            return None

        # Always reply so the mule sees our current state
        adv = self.build_ready_adv()
        self.rf.send_ready_adv(adv)

        if not adv.is_eligible() or adv.utility < self.fl_threshold:
            log.info(
                "device=%s refused solicit from mule=%s state=%s util=%.3f",
                self.device_id, solicit.mule_id, adv.state.value, adv.utility,
            )
            return None

        # Await θ_disc push
        try:
            push: DiscPush = self.rf.recv_disc_push(
                self.device_id, timeout=self.disc_push_timeout_s
            )
        except RFLinkError:
            log.warning(
                "device=%s: disc push timeout mule=%s", self.device_id, solicit.mule_id
            )
            return MissionOutcome.TIMEOUT

        # Local training step
        try:
            result = self.local_train(push.theta_disc, push.synth_batch)
        except Exception:
            log.exception(
                "device=%s: local_train raised; reporting PARTIAL", self.device_id
            )
            return MissionOutcome.PARTIAL

        # Ship the gradient back
        grad = GradientSubmission(
            device_id=self.device_id,
            mule_id=push.mule_id,
            mission_round=push.mission_round,
            delta_theta=result.delta_theta,
            num_examples=result.num_examples,
            submitted_at=time.time(),
        )
        self.rf.send_gradient(grad)

        # Post-round utility update (used by the *next* adv)
        self._update_utility(result, theta_global=push.theta_disc)
        self._set_last_outcome(MissionOutcome.CLEAN)
        return MissionOutcome.CLEAN

    # ---------------------------------------------- internal

    def _update_utility(
        self, result: LocalTrainResult, *, theta_global: Weights
    ) -> None:
        perf = performance_score(
            accuracy=result.accuracy,
            auc=result.auc,
            loss=result.loss,
        )
        # ``perf_discount`` is the same perf score (design doc: "cosine * perf")
        div = diversity_adjusted(
            theta_local=result.theta_after or result.delta_theta,
            theta_global=theta_global,
            perf_discount=perf,
        )
        u = utility(performance=perf, diversity=div, w1=self.w1, w2=self.w2)
        with self._lock:
            self._last_performance = perf
            self._last_diversity = div
            self._last_utility = u
        log.debug(
            "device=%s utility update perf=%.3f div=%.3f util=%.3f",
            self.device_id, perf, div, u,
        )

    def _set_last_outcome(self, outcome: MissionOutcome) -> None:
        with self._lock:
            self._last_outcome = outcome
