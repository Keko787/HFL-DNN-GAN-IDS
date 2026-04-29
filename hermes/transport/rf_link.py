"""RF link — Mule (HFLHostMission) <-> Edge Device (ClientMission).

Carries the four mission-scope FL messages:

    mule -> device : FLOpenSolicit, DiscPush
    device -> mule : FLReadyAdv,    GradientSubmission

Phase 2 ships an in-process loopback (``LoopbackRFLink``) so the mission
server and mission client can be exercised without a radio. Phase 6
swaps in a real Flower-over-RF transport behind the same ABC.

Design refs:
* HERMES_FL_Scheduler_Design.md §6.9 (interface contracts)
* HERMES_FL_Scheduler_Implementation_Plan.md §3 Phase 2
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional

from hermes.types import (
    DeviceID,
    DiscPush,
    FLOpenSolicit,
    FLReadyAdv,
    GradientSubmission,
    MuleID,
)


class RFLinkError(RuntimeError):
    """Raised when an RF-link operation fails (timeout, drop, etc.)."""


class RFLink(ABC):
    """Symmetric ABC covering both mule-side and device-side ops.

    The implementation is responsible for routing per-device, so a single
    mule can hold sessions with N devices concurrently. The loopback does
    this with per-device queues; a real radio would do it with addressing.
    """

    # ---- mule-side (server) ----

    @abstractmethod
    def broadcast_open_solicit(self, msg: FLOpenSolicit) -> None:
        """Mule -> all devices on the channel."""

    @abstractmethod
    def recv_ready_adv(self, timeout: Optional[float] = None) -> FLReadyAdv:
        """Block until any device answers with FLReadyAdv (or time out)."""

    @abstractmethod
    def push_disc(self, device_id: DeviceID, msg: DiscPush) -> None:
        """Unicast θ_disc + synth batch to one device."""

    @abstractmethod
    def recv_gradient(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> GradientSubmission:
        """Block for one device's gradient submission."""

    # ---- device-side (client) ----

    @abstractmethod
    def recv_open_solicit(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> FLOpenSolicit:
        """Device-side: wait for the next solicitation from any mule."""

    @abstractmethod
    def send_ready_adv(self, msg: FLReadyAdv) -> None:
        """Device -> mule: reply with current state + utility."""

    @abstractmethod
    def recv_disc_push(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> DiscPush:
        """Device-side: wait for a θ_disc push."""

    @abstractmethod
    def send_gradient(self, msg: GradientSubmission) -> None:
        """Device -> mule: submit Δθ_disc + meta."""

    @abstractmethod
    def close(self) -> None: ...


class LoopbackRFLink(RFLink):
    """Thread-safe in-process loopback.

    One instance is shared by the mule server and one-or-more device
    clients. Routing:
      * ``open_solicit``     fanned out to every registered device queue
      * ``ready_adv``        single mule-side queue (FIFO across devices)
      * ``disc_push``        per-device queue (keyed by device_id)
      * ``gradient``         per-device queue on the mule side
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._closed = False

        # mule -> device(s)
        self._solicit_per_device: dict[DeviceID, "queue.Queue[FLOpenSolicit]"] = {}
        self._disc_per_device: dict[DeviceID, "queue.Queue[DiscPush]"] = {}

        # device -> mule
        self._ready_q: "queue.Queue[FLReadyAdv]" = queue.Queue()
        self._gradient_per_device: dict[
            DeviceID, "queue.Queue[GradientSubmission]"
        ] = {}

    # ---- device registration --------------------------------------------------

    def register_device(self, device_id: DeviceID) -> None:
        """Ensure queues exist for a device before it starts receiving."""
        with self._lock:
            self._solicit_per_device.setdefault(device_id, queue.Queue())
            self._disc_per_device.setdefault(device_id, queue.Queue())
            self._gradient_per_device.setdefault(device_id, queue.Queue())

    def known_devices(self) -> list[DeviceID]:
        with self._lock:
            return list(self._solicit_per_device.keys())

    # ---- mule-side ------------------------------------------------------------

    def broadcast_open_solicit(self, msg: FLOpenSolicit) -> None:
        self._raise_if_closed()
        with self._lock:
            targets = list(self._solicit_per_device.values())
        for q in targets:
            q.put(msg)

    def recv_ready_adv(self, timeout: Optional[float] = None) -> FLReadyAdv:
        self._raise_if_closed()
        try:
            return self._ready_q.get(timeout=timeout)
        except queue.Empty as e:
            raise RFLinkError(f"recv_ready_adv timed out after {timeout}s") from e

    def push_disc(self, device_id: DeviceID, msg: DiscPush) -> None:
        self._raise_if_closed()
        q = self._ensure_device_queue(self._disc_per_device, device_id)
        q.put(msg)

    def recv_gradient(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> GradientSubmission:
        self._raise_if_closed()
        q = self._ensure_device_queue(self._gradient_per_device, device_id)
        try:
            return q.get(timeout=timeout)
        except queue.Empty as e:
            raise RFLinkError(
                f"recv_gradient for {device_id!r} timed out after {timeout}s"
            ) from e

    # ---- device-side ----------------------------------------------------------

    def recv_open_solicit(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> FLOpenSolicit:
        self._raise_if_closed()
        q = self._ensure_device_queue(self._solicit_per_device, device_id)
        try:
            return q.get(timeout=timeout)
        except queue.Empty as e:
            raise RFLinkError(
                f"recv_open_solicit for {device_id!r} timed out after {timeout}s"
            ) from e

    def send_ready_adv(self, msg: FLReadyAdv) -> None:
        self._raise_if_closed()
        self._ready_q.put(msg)

    def recv_disc_push(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> DiscPush:
        self._raise_if_closed()
        q = self._ensure_device_queue(self._disc_per_device, device_id)
        try:
            return q.get(timeout=timeout)
        except queue.Empty as e:
            raise RFLinkError(
                f"recv_disc_push for {device_id!r} timed out after {timeout}s"
            ) from e

    def send_gradient(self, msg: GradientSubmission) -> None:
        self._raise_if_closed()
        q = self._ensure_device_queue(
            self._gradient_per_device, msg.device_id
        )
        q.put(msg)

    # ---- shared ---------------------------------------------------------------

    def close(self) -> None:
        self._closed = True

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RFLinkError("rf link closed")

    def _ensure_device_queue(self, store: dict, device_id: DeviceID):
        with self._lock:
            q = store.get(device_id)
            if q is None:
                q = queue.Queue()
                store[device_id] = q
            return q
