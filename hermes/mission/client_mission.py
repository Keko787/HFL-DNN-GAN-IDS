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
    DeliveryAck,
    DeviceID,
    DiscPush,
    FLOpenSolicit,
    FLReadyAdv,
    FLState,
    GradientSubmission,
    MissionOutcome,
    MissionPass,
    Weights,
)
from hermes.types.fl_messages import UPDATE_FORM_DELTA, UPDATE_FORM_WEIGHTS

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
        contact_reliability: Optional[float] = None,
        contact_rng_seed: Optional[int] = None,
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
        # EX-4.2 — short-range contact reliability (device<->mule uplink).
        # None => every collect completes (the default). Otherwise a
        # per-collect Bernoulli models Exp 3's ``reliability x rf_factor``
        # completion, so some contacts fail even under clean links.
        self._contact_reliability = contact_reliability
        self._contact_rng = np.random.default_rng(contact_rng_seed)

        self._lock = threading.RLock()
        self._state: FLState = FLState.UNAVAILABLE

        # post-round snapshot used for the *next* advertisement
        self._last_performance: float = 0.0
        self._last_diversity: float = 0.0
        self._last_utility: float = 0.0
        self._last_outcome: Optional[MissionOutcome] = None
        # Oort baseline inputs (arm B2). The RAW loss and sample count, kept
        # because `_last_utility` collapses accuracy/AUC/loss into one score and
        # Oort's |B_i|*sqrt(mean Loss^2) cannot be recovered from it.
        self._last_loss: Optional[float] = None
        self._last_num_examples: int = 0
        # Which push the most recent ``serve_once`` answered — its mission
        # round and pass ("collect" / "deliver") — or None when no push
        # arrived. Read by the device process to label ``device_served``.
        self.last_push_round: Optional[int] = None
        self.last_push_pass: Optional[str] = None

        # Sprint 1.5 — offline-training state. ``train_offline()`` runs
        # local training against ``_theta_basis`` (the θ most recently
        # delivered by a mule) and stashes the result in
        # ``_prepared_delta``. ``serve_once`` ships ``_prepared_delta``
        # back at the next mule visit, so no fitting happens during the
        # contact (design §7 principle 14).
        #
        # Backward compat: if ``train_offline()`` was never called and
        # ``_prepared_delta`` is None when ``serve_once`` runs, the
        # client falls back to calling ``local_train`` inline (legacy
        # path). This keeps Phase-3 demos and Sprint-1A tests working.
        self._theta_basis: Optional[Weights] = None
        self._last_synth_batch: List[np.ndarray] = []
        self._prepared_delta: Optional[LocalTrainResult] = None
        # FeRRy Phase 1 — the version of ``_theta_basis`` (from the push that
        # delivered it), and the basis a prepared result was trained from, with
        # its version. A prepared update ships later, possibly after a newer
        # basis arrived, so it must carry the basis it was actually trained on.
        self._theta_basis_version: Optional[int] = None
        self._prepared_basis: Optional[Weights] = None
        self._prepared_basis_version: Optional[int] = None
        # One local fit at a time: a train-ahead runs on a background thread
        # and must not share the local model with a delivery-triggered fit.
        self._train_lock = threading.Lock()
        self._train_thread: Optional[threading.Thread] = None

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

    def build_ready_adv(self, *, in_reply_to: int = 0) -> FLReadyAdv:
        """Build the next ``FLReadyAdv`` from the most recent round's metrics.

        ``in_reply_to`` is the ``solicit_id`` of the solicit being answered
        (FeRRy Phase 3): the ferry mule only takes adverts that carry its own
        contact's number. 0, the default, is an unnumbered solicit or a beacon.
        """
        with self._lock:
            return FLReadyAdv(
                device_id=self.device_id,
                state=self._state,
                performance_score=self._last_performance,
                diversity_adjusted=self._last_diversity,
                utility=self._last_utility,
                last_round_outcome=self._last_outcome,
                issued_at=time.time(),
                local_loss=self._last_loss,
                num_examples=self._last_num_examples,
                in_reply_to=in_reply_to,
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
        """Wait for a solicit, exchange one FL contact, update utility.

        Branches on the solicit's ``pass_kind``:

        * **Pass 1 (COLLECT)** — exchange-only: receive θ, return the
          ``_prepared_delta`` from offline training. If no delta is
          prepared (cold start, or the device hasn't called
          ``train_offline`` yet), falls back to running ``local_train``
          inline so existing Phase-3 / Sprint-1A demos keep working.
        * **Pass 2 (DELIVER)** — push-only: receive θ' as the new basis
          for offline training, ack with ``DeliveryAck``, no Δθ sent.

        Returns the outcome tag, or ``None`` if the solicit timed out
        or the device was not FL_OPEN.
        """
        self.last_push_round = None
        self.last_push_pass = None

        # Wait for a mule to ping us
        try:
            solicit = self.rf.recv_open_solicit(
                self.device_id, timeout=self.solicit_timeout_s
            )
        except RFLinkError:
            return None
        # FeRRy Phase 3: echo the solicit's number in the advert and in the
        # reply to the push that follows (0 for the legacy broadcast).
        solicit_id = int(getattr(solicit, "solicit_id", 0) or 0)

        # Always reply so the mule sees our current state
        adv = self.build_ready_adv(in_reply_to=solicit_id)
        self.rf.send_ready_adv(adv)

        if self.device_id in getattr(solicit, "not_ready", ()):
            # Exp 5 addendum (Study 5.12): the mule's fit clock says this
            # device's local fit has not finished at this contact. No push
            # follows: the device keeps its basis and its fit, and sends nothing.
            log.info(
                "device=%s: no update ready for mule=%s (its fit runs on)",
                self.device_id, solicit.mule_id,
            )
            return None

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

        self.last_push_round = push.mission_round
        self.last_push_pass = push.pass_kind.value

        if push.pass_kind is MissionPass.DELIVER:
            return self._handle_delivery_push(push, in_reply_to=solicit_id)

        return self._handle_collect_push(push, in_reply_to=solicit_id)

    def serve_delivery(self) -> Optional[MissionOutcome]:
        """Pass-2 explicit handler — same as :meth:`serve_once` but only
        accepts a Pass-2 solicit. Returns ``None`` if the solicit was
        Pass-1 (the device drops it; the mule will retry).

        Used by tests / supervisors that want to tightly bind a worker
        thread to one Pass.
        """
        try:
            solicit = self.rf.recv_open_solicit(
                self.device_id, timeout=self.solicit_timeout_s
            )
        except RFLinkError:
            return None

        if solicit.pass_kind is not MissionPass.DELIVER:
            log.debug(
                "device=%s serve_delivery: dropping non-DELIVER solicit %s",
                self.device_id, solicit.pass_kind.value,
            )
            return None
        solicit_id = int(getattr(solicit, "solicit_id", 0) or 0)

        adv = self.build_ready_adv(in_reply_to=solicit_id)
        self.rf.send_ready_adv(adv)

        try:
            push: DiscPush = self.rf.recv_disc_push(
                self.device_id, timeout=self.disc_push_timeout_s
            )
        except RFLinkError:
            return MissionOutcome.TIMEOUT

        if push.pass_kind is not MissionPass.DELIVER:
            log.warning(
                "device=%s serve_delivery: solicit was DELIVER but push is %s",
                self.device_id, push.pass_kind.value,
            )
            return MissionOutcome.PARTIAL

        return self._handle_delivery_push(push, in_reply_to=solicit_id)

    def train_offline(
        self, *, synth_batch: Optional[List[np.ndarray]] = None
    ) -> Optional[LocalTrainResult]:
        """Run local training between mule visits — Sprint 1.5 principle 14.

        Trains against the most recently received θ_basis. Stashes the
        result in ``_prepared_delta`` so the next ``serve_once`` ships
        it without doing any in-session compute. No-op if no θ has
        been received yet.

        Args:
            synth_batch: synthetic samples to train against. Defaults
                to the synth batch from the last DiscPush.

        Returns:
            The training result, or ``None`` if no θ basis is available.
        """
        with self._train_lock:
            with self._lock:
                theta = self._theta_basis
                theta_version = self._theta_basis_version
                stashed_synth = list(self._last_synth_batch)
            if theta is None:
                log.debug(
                    "device=%s train_offline: no θ basis yet, skipping",
                    self.device_id,
                )
                return None

            synth = list(synth_batch) if synth_batch is not None else stashed_synth
            try:
                result = self.local_train(theta, synth)
            except Exception:
                log.exception("device=%s train_offline raised", self.device_id)
                return None

            with self._lock:
                if self._theta_basis is not theta:
                    # A newer basis arrived while this fit ran (a background
                    # train-ahead overtaken by a delivery): the result is stale.
                    log.debug(
                        "device=%s train_offline: basis replaced mid-fit; "
                        "discarding", self.device_id,
                    )
                    return None
                self._prepared_delta = result
                self._prepared_basis = theta
                self._prepared_basis_version = theta_version
        self._update_utility(result, theta_global=theta)
        return result

    def _start_train_ahead(self) -> None:
        """Train on the current basis on a background thread (FeRRy Phase 1).

        Used when the mule's Pass 2 is budgeted and may not return: the device
        prepares its next update on the basis it just adopted, between visits,
        so a device collected in Pass 1 and skipped in Pass 2 later ships an
        update with that basis's age instead of fitting in session. A delivery
        that arrives meanwhile replaces the basis, and the stale result is
        discarded.
        """
        t = threading.Thread(
            target=self.train_offline, name=f"train-ahead-{self.device_id}",
            daemon=True,
        )
        self._train_thread = t
        t.start()

    # ---------------------------------------------- pass-specific helpers

    def _handle_collect_push(
        self, push: DiscPush, *, in_reply_to: int = 0,
    ) -> MissionOutcome:
        """Pass-1 path — ship the prepared Δθ (or fall back to inline training).

        ``in_reply_to`` is the number of the solicit this contact answered; the
        update carries it back so the mule can tell it from a late update of an
        earlier contact (FeRRy Phase 3, critic B2). 0 when unnumbered.
        """
        # FeRRy Phase 3 — the mule's availability draw failed this contact's
        # uplink (``DiscPush.uplink_drop``). As under EX-4.2 below, the device
        # still received θ: it adopts the basis, trains ahead if asked, and
        # keeps its prepared update for the next contact; nothing is sent. No
        # local draw is made, so the device's own stream is not consumed.
        if push.uplink_drop:
            self._set_theta_basis(
                push.theta_disc, push.synth_batch, version=push.basis_version,
            )
            self._set_last_outcome(MissionOutcome.TIMEOUT)
            log.info(
                "device=%s: Pass-1 contact uplink dropped by the mule's "
                "availability draw", self.device_id,
            )
            if push.train_ahead:
                self._start_train_ahead()
            return MissionOutcome.TIMEOUT
        # EX-4.2 — short-range contact reliability. With prob
        # (1 - contact_reliability) this device's Δθ does not reach the mule
        # this contact (a failed short-range uplink, modelling Exp 3's
        # ``reliability x rf_factor``). The device still received θ, so it
        # adopts the basis and retrains for the next mission; the mule sees a
        # non-CLEAN outcome and this device does not count toward the round.
        if (
            self._contact_reliability is not None
            and float(self._contact_rng.random()) >= self._contact_reliability
        ):
            self._set_theta_basis(
                push.theta_disc, push.synth_batch, version=push.basis_version,
            )
            self._set_last_outcome(MissionOutcome.TIMEOUT)
            log.info(
                "device=%s: Pass-1 contact uplink dropped (reliability=%.3f)",
                self.device_id, self._contact_reliability,
            )
            if push.train_ahead:
                self._start_train_ahead()
            return MissionOutcome.TIMEOUT
        # H2 — atomic take-and-clear: read AND null out the prepared
        # slot under a single lock acquisition so a concurrent
        # train_offline() call can't lose its result between read and
        # clear. Without this, train_offline could race in between
        # `prepared = ...` and `self._prepared_delta = None` and have
        # its fresh delta clobbered to None silently.
        # Wait out a train-ahead still running, so its result is taken here.
        with self._train_lock, self._lock:
            prepared = self._prepared_delta
            prepared_basis = self._prepared_basis
            prepared_version = self._prepared_basis_version
            self._prepared_delta = None
            self._prepared_basis = None
            self._prepared_basis_version = None

        if prepared is None:
            # Fallback path — the device hasn't run train_offline yet.
            # M1 — log a warning so a misconfigured production device
            # can be detected (this path violates principle 14: "no
            # fitting in session"). Existing Phase-3 / Sprint-1A tests
            # still pass; they just emit a one-line warning each session.
            log.warning(
                "device=%s _handle_collect_push: no _prepared_delta — "
                "falling back to in-session training. This violates "
                "design principle 14 ('FL sessions are exchange-only'). "
                "Call train_offline() between mule visits to suppress.",
                self.device_id,
            )
            try:
                with self._train_lock:
                    result = self.local_train(push.theta_disc, push.synth_batch)
            except Exception:
                log.exception(
                    "device=%s local_train raised in fallback path",
                    self.device_id,
                )
                return MissionOutcome.PARTIAL
            basis, basis_version = push.theta_disc, push.basis_version
        else:
            result = prepared
            basis, basis_version = prepared_basis, prepared_version

        # The mule names the form it merges in: full weights (agg:plain, every
        # recorded run) or the update against the basis it was trained from.
        form = push.update_form
        if form == UPDATE_FORM_DELTA:
            if basis is None:
                log.warning(
                    "device=%s: delta requested but the training basis is "
                    "unknown; sending nothing this contact", self.device_id,
                )
                return MissionOutcome.PARTIAL
            payload = [
                np.asarray(w) - np.asarray(b)
                for w, b in zip(result.delta_theta, basis)
            ]
        else:
            form = UPDATE_FORM_WEIGHTS
            payload = result.delta_theta

        grad = GradientSubmission(
            device_id=self.device_id,
            mule_id=push.mule_id,
            mission_round=push.mission_round,
            delta_theta=payload,
            num_examples=result.num_examples,
            submitted_at=time.time(),
            local_loss=float(result.loss),
            basis_version=basis_version,
            update_form=form,
            in_reply_to=in_reply_to,
        )
        self.rf.send_gradient(grad)

        # Update post-round utility + adopt the pushed θ as the new basis
        # for the *next* round of offline training. (In steady state
        # under two-pass missions, the Pass-1 push and Pass-2 push carry
        # the same θ — Pass 1 is informational, Pass 2 is authoritative.)
        self._update_utility(result, theta_global=push.theta_disc)
        self._set_theta_basis(
            push.theta_disc, push.synth_batch, version=push.basis_version,
        )
        self._set_last_outcome(MissionOutcome.CLEAN)
        if push.train_ahead:
            self._start_train_ahead()
        return MissionOutcome.CLEAN

    def _handle_delivery_push(
        self, push: DiscPush, *, in_reply_to: int = 0,
    ) -> MissionOutcome:
        """Pass-2 path — store θ' as new basis, ack receipt, train ahead.

        ``in_reply_to`` (FeRRy Phase 3) is the number of the solicit this
        delivery answered, echoed on the ack; 0 when unnumbered.

        H6 — design §7 principle 13 says the device "starts fresh local
        training immediately" after Pass-2 receipt. We honour that
        synchronously here: ack first (fast, the mule needs to move on),
        then run ``train_offline()`` so a fresh ``_prepared_delta`` is
        ready when the next mule visit's Pass-1 contact arrives.

        ``train_offline`` failures don't propagate — the device just
        stays without a prepared delta and the next Pass-1 visit will
        log an M1 fallback warning.
        """
        self._set_theta_basis(
            push.theta_disc, push.synth_batch, version=push.basis_version,
        )
        ack = DeliveryAck(
            device_id=self.device_id,
            mule_id=push.mule_id,
            mission_round=push.mission_round,
            weights_sig=push.weights_sig,
            received_at=time.time(),
            in_reply_to=in_reply_to,
        )
        self.rf.send_delivery_ack(ack)

        # Train ahead for the next mission's Pass-1 visit. Best-effort —
        # if training raises, the device stays without a prepared delta
        # but the delivery itself was already acked.
        try:
            self.train_offline()
        except Exception:  # pragma: no cover — defensive belt + suspenders
            log.exception(
                "device=%s _handle_delivery_push: train_offline raised",
                self.device_id,
            )

        # Pass-2 isn't a Pass-1 outcome category; we report CLEAN as the
        # closest analogue (delivery succeeded). Callers (the supervisor)
        # use the dedicated MissionDeliveryReport from HFLHostMission for
        # Pass-2-specific accounting.
        self._set_last_outcome(MissionOutcome.CLEAN)
        return MissionOutcome.CLEAN

    def _set_theta_basis(
        self,
        theta: Weights,
        synth: List[np.ndarray],
        *,
        version: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._theta_basis = [w.copy() for w in theta]
            self._theta_basis_version = version
            self._last_synth_batch = list(synth)

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
            self._last_loss = float(result.loss)
            self._last_num_examples = int(result.num_examples)
        log.debug(
            "device=%s utility update perf=%.3f div=%.3f util=%.3f",
            self.device_id, perf, div, u,
        )

    def _set_last_outcome(self, outcome: MissionOutcome) -> None:
        with self._lock:
            self._last_outcome = outcome
