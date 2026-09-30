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

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Tuple

from hermes.types import (
    DeliveryAck,
    DeviceID,
    DiscPush,
    FLOpenSolicit,
    FLReadyAdv,
    GradientSubmission,
    MuleID,
)

log = logging.getLogger(__name__)


class RFLinkError(RuntimeError):
    """Raised when an RF-link operation fails (timeout, drop, etc.)."""


def _target_ids(device_ids: Iterable[DeviceID]) -> List[DeviceID]:
    """The ids a targeted solicit addresses: in the given order, no repeats.

    A bare string is refused: iterating it would address one "device" per
    character, and a solicit silently reaching nobody is hard to spot.
    """
    if isinstance(device_ids, (str, bytes)):
        raise TypeError(
            f"solicit() takes a collection of device ids, not one id "
            f"({device_ids!r}); wrap it in a list"
        )
    return list(dict.fromkeys(device_ids))


def _take_newest(q: "queue.Queue[Any]", timeout: Optional[float]) -> Tuple[Any, int]:
    """Block like ``q.get(timeout=...)``, then drain ``q`` and keep the newest item.

    Returns ``(newest, n_dropped)``; raises ``queue.Empty`` when nothing
    arrives in time. Used by the newest-solicit option (FeRRy Phase 3,
    critic B1): a device that missed a gather must answer the solicit the
    mule is gathering for now, not the stale one queued before it.
    """
    item = q.get(timeout=timeout)
    dropped = 0
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return item, dropped
        dropped += 1


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

    @abstractmethod
    def recv_delivery_ack(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> DeliveryAck:
        """Pass-2 only: wait for one device's DeliveryAck."""

    def solicit(
        self, msg: FLOpenSolicit, device_ids: Iterable[DeviceID]
    ) -> List[DeviceID]:
        """Mule -> the listed devices only (targeted solicit, FeRRy Phase 3).

        A ferry contact solicits the members of its stop instead of every
        device on the channel. Ids the link does not know (never registered,
        or dropped) are skipped, not raised: no advert comes back from them
        and the mule records them as unreachable. Returns the ids the solicit
        went out to, in the given order without repeats.

        Not abstract, so every existing ``RFLink`` (test fakes included)
        stays constructible; a link that cannot address single devices
        raises ``NotImplementedError``. :meth:`broadcast_open_solicit` is
        unchanged and stays the legacy path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support targeted solicits"
        )

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
    def send_delivery_ack(self, msg: DeliveryAck) -> None:
        """Pass-2 only: device acks θ' receipt to mule."""

    @abstractmethod
    def close(self) -> None: ...


class LoopbackRFLink(RFLink):
    """Thread-safe in-process loopback. **Tests + demos only.**

    Phase 7 retirement: this class is allowed in ``tests/`` and the
    ``hermes.{cluster,mule,mission}.__main__`` pedagogical demos, but
    must **not** be imported from any production runtime path
    (``hermes.processes.*`` and the supervised cluster / mule / device
    services). Production uses :class:`TCPRFLinkServer` +
    :class:`TCPRFLinkClient`. The ``test_loopback_retirement.py``
    import-graph test pins this invariant.

    One instance is shared by the mule server and one-or-more device
    clients. Routing:
      * ``open_solicit``     fanned out to every registered device queue
                             (``solicit``: only the listed ones)
      * ``ready_adv``        single mule-side queue (FIFO across devices)
      * ``disc_push``        per-device queue (keyed by device_id)
      * ``gradient``         per-device queue on the mule side

    ``newest_solicit_only`` (default off, FIFO as in every recorded run)
    makes ``recv_open_solicit`` return the newest queued solicit and drop
    the older ones, for every device on this link; it mirrors
    ``TCPRFLinkClient(newest_solicit_only=...)`` for in-process tests.
    """

    def __init__(self, *, newest_solicit_only: bool = False) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._newest_solicit_only = bool(newest_solicit_only)

        # mule -> device(s)
        self._solicit_per_device: dict[DeviceID, "queue.Queue[FLOpenSolicit]"] = {}
        self._disc_per_device: dict[DeviceID, "queue.Queue[DiscPush]"] = {}

        # device -> mule
        self._ready_q: "queue.Queue[FLReadyAdv]" = queue.Queue()
        self._gradient_per_device: dict[
            DeviceID, "queue.Queue[GradientSubmission]"
        ] = {}
        self._delivery_ack_per_device: dict[
            DeviceID, "queue.Queue[DeliveryAck]"
        ] = {}

    # ---- device registration --------------------------------------------------

    def register_device(self, device_id: DeviceID) -> None:
        """Ensure queues exist for a device before it starts receiving."""
        with self._lock:
            self._solicit_per_device.setdefault(device_id, queue.Queue())
            self._disc_per_device.setdefault(device_id, queue.Queue())
            self._gradient_per_device.setdefault(device_id, queue.Queue())
            self._delivery_ack_per_device.setdefault(device_id, queue.Queue())

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

    def solicit(
        self, msg: FLOpenSolicit, device_ids: Iterable[DeviceID]
    ) -> List[DeviceID]:
        self._raise_if_closed()
        wanted = _target_ids(device_ids)
        # Only registered devices: creating a queue for an unknown id would
        # hand this stale solicit to a device that registers later.
        with self._lock:
            targets = [
                (did, self._solicit_per_device[did])
                for did in wanted if did in self._solicit_per_device
            ]
        for _did, q in targets:
            q.put(msg)
        return [did for did, _q in targets]

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

    def recv_delivery_ack(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> DeliveryAck:
        self._raise_if_closed()
        q = self._ensure_device_queue(self._delivery_ack_per_device, device_id)
        try:
            return q.get(timeout=timeout)
        except queue.Empty as e:
            raise RFLinkError(
                f"recv_delivery_ack for {device_id!r} timed out after {timeout}s"
            ) from e

    # ---- device-side ----------------------------------------------------------

    def recv_open_solicit(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> FLOpenSolicit:
        self._raise_if_closed()
        q = self._ensure_device_queue(self._solicit_per_device, device_id)
        try:
            if not self._newest_solicit_only:
                return q.get(timeout=timeout)
            msg, dropped = _take_newest(q, timeout)
        except queue.Empty as e:
            raise RFLinkError(
                f"recv_open_solicit for {device_id!r} timed out after {timeout}s"
            ) from e
        if dropped:
            log.debug(
                "LoopbackRFLink %s: answering the newest solicit, dropped %d older",
                device_id, dropped,
            )
        return msg

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

    def send_delivery_ack(self, msg: DeliveryAck) -> None:
        self._raise_if_closed()
        q = self._ensure_device_queue(
            self._delivery_ack_per_device, msg.device_id
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
