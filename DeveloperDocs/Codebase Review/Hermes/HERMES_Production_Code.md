# HERMES — Production-Grade Code Proposals

**Status:** Proposal. **Not applied.** Requires approval per
[`../01_Refactoring_Strategy.md` §5](../01_Refactoring_Strategy.md#5-approval-checklist).
**Constraint:** every change below is **behaviour-preserving** in the currently-tested
regime. Where a change alters behaviour outside that regime (the >30 s idle case, which
is broken today), that is stated explicitly.

Each section states the defect, the current code, the replacement, and how to verify the
replacement did not change anything.

---

## Contents

1. [Platform-correct socket options](#1-platform-correct-socket-options) — H-01
2. [RF link socket lifetime, heartbeat, reconnect](#2-rf-link-socket-lifetime) — H-01
3. [Unified contact exchange](#3-unified-contact-exchange) — H-02, H-03, Q-01
4. [Cancellable dock wait](#4-cancellable-dock-wait) — H-04
5. [Observability that cannot crash its caller](#5-observability-that-cannot-crash-its-caller) — H-07
6. [Inverting the `experiments` dependency](#6-inverting-the-experiments-dependency) — H-08
7. [Registry persistence and indexing](#7-registry-persistence-and-indexing) — H-06, H-05, H-09

---

## 1. Platform-correct socket options

**Defect (H-01, secondary).** `tcp_dock_link.py:290-299` packs `SO_SNDTIMEO` as a Linux
`struct timeval` and claims cross-platform compatibility. Windows expects a 4-byte DWORD
of milliseconds, so `setsockopt` raises `OSError`, the bare handler swallows it, and the
dock send timeout is silently absent on the development platform.

**New file — `hermes/transport/_sockopt.py`:**

```python
"""Platform-correct socket-option helpers shared by the TCP transports.

``SO_SNDTIMEO`` / ``SO_RCVTIMEO`` take different value shapes per platform:

* POSIX  — ``struct timeval`` (two native longs: seconds, microseconds)
* Windows — a single ``DWORD`` of milliseconds

The previous inline implementation in ``tcp_dock_link.py`` packed a
``timeval`` unconditionally, so on Windows ``setsockopt`` raised ``OSError``
and the timeout was silently not applied. This module gets it right on both
and reports whether the option actually took effect, so callers can log a
degraded transport instead of assuming a bound that isn't there.
"""

from __future__ import annotations

import logging
import socket
import struct
import sys

log = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform.startswith("win")


def _encode_timeout(seconds: float) -> bytes:
    if _IS_WINDOWS:
        # DWORD milliseconds, little-endian native.
        return struct.pack("@I", max(1, int(seconds * 1000)))
    whole = int(seconds)
    micros = int((seconds - whole) * 1_000_000)
    return struct.pack("@ll", whole, micros)


def set_send_timeout(sock: socket.socket, seconds: float) -> bool:
    """Bound blocking ``send``/``sendall`` without touching the read timeout.

    Returns True iff the option was accepted. A False return is not fatal —
    the caller falls back to relying on the peer to drain — but it should be
    logged, because it means a stuck peer can block a sender for as long as
    the OS allows.
    """
    return _set(sock, socket.SO_SNDTIMEO, seconds, "SO_SNDTIMEO")


def set_recv_timeout(sock: socket.socket, seconds: float) -> bool:
    """Bound blocking ``recv`` without touching the send timeout."""
    return _set(sock, socket.SO_RCVTIMEO, seconds, "SO_RCVTIMEO")


def _set(sock: socket.socket, option: int, seconds: float, label: str) -> bool:
    if seconds <= 0:
        raise ValueError(f"{label} requires a positive timeout, got {seconds!r}")
    try:
        sock.setsockopt(socket.SOL_SOCKET, option, _encode_timeout(seconds))
        return True
    except OSError as e:
        log.debug("%s not honoured on this platform: %s", label, e)
        return False


def enable_keepalive(
    sock: socket.socket,
    *,
    idle_s: int = 30,
    interval_s: int = 10,
    probes: int = 3,
) -> bool:
    """Turn on TCP keepalive so a vanished peer is detected without traffic.

    The per-connection tuning knobs are optional and platform-dependent:
    Linux exposes TCP_KEEPIDLE/INTVL/CNT, macOS uses TCP_KEEPALIVE, and
    Windows needs ``SIO_KEEPALIVE_VALS`` via ioctl. We set what is available
    and fall back to the OS defaults for the rest — the important part is
    that ``SO_KEEPALIVE`` itself is on.
    """
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError as e:
        log.debug("SO_KEEPALIVE unavailable: %s", e)
        return False

    if _IS_WINDOWS:
        try:
            sock.ioctl(socket.SIO_KEEPALIVE_VALS,
                       (1, idle_s * 1000, interval_s * 1000))
        except (AttributeError, OSError) as e:
            log.debug("SIO_KEEPALIVE_VALS unavailable: %s", e)
        return True

    for name, value in (
        ("TCP_KEEPIDLE", idle_s),
        ("TCP_KEEPINTVL", interval_s),
        ("TCP_KEEPCNT", probes),
        ("TCP_KEEPALIVE", idle_s),   # macOS spelling
    ):
        option = getattr(socket, name, None)
        if option is None:
            continue
        try:
            sock.setsockopt(socket.IPPROTO_TCP, option, value)
        except OSError:
            pass
    return True
```

**Then in `tcp_dock_link.py`,** replace the inline `struct.pack("ll", …)` block with:

```python
from ._sockopt import enable_keepalive, set_send_timeout

conn.settimeout(None)                     # reader blocks forever — unchanged
if not set_send_timeout(conn, self._send_timeout_s):
    log.warning(
        "dock: send timeout not enforced for mule %s on this platform; "
        "a stuck peer can block send_down", mid,
    )
enable_keepalive(conn)
```

**Verification.** Behaviour on Linux is identical (same option, same value shape). On
Windows the option now actually applies, which is a strict improvement over silently not
applying. `tests/integration/test_tcp_dock_link.py` unchanged.

---

## 2. RF link socket lifetime

**Defect (H-01, primary).** `tcp_rf_link.py:375` — `conn.settimeout(self._send_timeout_s)`
governs reads as well as writes, so a device silent for 30 s is disconnected.

### 2a. Server side

```python
# hermes/transport/tcp_rf_link.py  —  _spawn_reader, replacing line 375

        did = reg.device_id

        # H-01 FIX (mirrors tcp_dock_link.py:283-299).
        #
        # A device is legitimately silent for the entire flight leg between
        # two contact events — minutes, not seconds. The previous
        # ``conn.settimeout(self._send_timeout_s)`` applied that 30 s bound
        # to *reads* as well as writes, so ``_reader_loop``'s next
        # ``recv_message`` raised socket.timeout -> OSError -> WireError and
        # the device was silently dropped mid-mission. No test caught it
        # because every integration test finishes inside 30 s.
        #
        # Correct shape: block forever on read (peer-vanish surfaces via
        # WireError on the next frame, or via TCP keepalive), and bound the
        # SEND side only, so a stuck recipient still cannot pin the mule.
        conn.settimeout(None)
        if not set_send_timeout(conn, self._send_timeout_s):
            log.warning(
                "rf: send timeout not enforced for device %s on this "
                "platform; a stuck peer can block push_disc", did,
            )
        enable_keepalive(conn, idle_s=self._keepalive_idle_s)
```

Add to `__init__`:

```python
        keepalive_idle_s: int = 30,
        ...
        self._keepalive_idle_s = int(keepalive_idle_s)
```

### 2b. Client side — bounded reconnect

Replace the fixed `self._sock.settimeout(60.0)` at `:497` with the same
`settimeout(None)` + `set_send_timeout` + `enable_keepalive` treatment, and give the
client a reconnect path so a transient mule restart is survivable:

```python
class TCPRFLinkClient(RFLink):

    def __init__(
        self,
        device_id: DeviceID,
        host: str,
        port: int,
        *,
        emulator: Optional[ChannelEmulator] = None,
        connect_timeout_s: float = 5.0,
        send_timeout_s: float = 30.0,
        reconnect_backoff_s: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0),
    ) -> None:
        self._device_id = device_id
        self._host, self._port = host, port
        self._emulator = emulator or no_op_emulator()
        self._connect_timeout_s = connect_timeout_s
        self._send_timeout_s = send_timeout_s
        self._backoff = tuple(reconnect_backoff_s)

        self._closed = threading.Event()      # set only by close(), not by a drop
        self._link_up = threading.Event()
        # H-10: this lock existed but was never acquired. Two threads calling
        # send_ready_adv / send_gradient concurrently would interleave halves
        # of two frames on one socket and corrupt the stream irrecoverably.
        # Every sendall now happens under it.
        self._send_lock = threading.Lock()

        self._solicit_q: "queue.Queue[FLOpenSolicit]" = queue.Queue()
        self._disc_q: "queue.Queue[DiscPush]" = queue.Queue()

        self._sock: Optional[socket.socket] = None
        self._connect_locked()                # raises if the first connect fails

        self._reader = threading.Thread(
            target=self._reader_supervisor,
            name=f"TCPRFLinkClient-{device_id}",
            daemon=True,
        )
        self._reader.start()

    # ------------------------------------------------------------------ #

    def _connect_locked(self) -> None:
        """Open a socket, register, and mark the link up. Raises on failure."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout_s)
        sock.connect((self._host, self._port))
        sock.settimeout(None)                              # H-01: reads block
        set_send_timeout(sock, self._send_timeout_s)       # writes bounded
        enable_keepalive(sock)
        send_message(sock, _DeviceRegistrationMessage(device_id=self._device_id))
        self._sock = sock
        self._link_up.set()

    def _reader_supervisor(self) -> None:
        """Pump frames; on peer loss, reconnect with bounded backoff.

        The device has no way to make progress without the mule, so an
        indefinite retry with a capped interval is the right policy — but it
        must be a *sleep*, not a spin. Previously a dropped link left
        ``ClientMission.serve_once`` returning None immediately forever and
        ``DeviceService.run`` burning a core.
        """
        attempt = 0
        while not self._closed.is_set():
            try:
                self._reader_loop()            # returns on peer close
            except Exception:                  # pragma: no cover — defensive
                log.exception("rf client %s reader loop raised", self._device_id)

            if self._closed.is_set():
                break

            self._link_up.clear()
            self._teardown_socket()
            delay = self._backoff[min(attempt, len(self._backoff) - 1)]
            attempt += 1
            log.warning(
                "rf client %s: link down, reconnecting in %.1fs (attempt %d)",
                self._device_id, delay, attempt,
            )
            if self._closed.wait(delay):
                break
            try:
                self._connect_locked()
                log.info("rf client %s: link restored", self._device_id)
                attempt = 0
            except OSError as e:
                log.debug("rf client %s: reconnect failed: %s", self._device_id, e)

    def _send_with_emulator(self, msg, *, label: str) -> None:
        drop, delay = self._emulator.apply()
        if drop:
            log.debug("TCPRFLinkClient %s: dropped outbound %s", self._device_id, label)
            return
        if delay > 0.0:
            time.sleep(delay)
        with self._send_lock:                  # H-10: framing integrity
            sock = self._sock
            if sock is None or not self._link_up.is_set():
                raise RFLinkError(
                    f"send {label} from {self._device_id!r}: link is down"
                )
            try:
                send_message(sock, msg)
            except WireError as e:
                self._link_up.clear()          # NOT _closed — supervisor retries
                raise RFLinkError(
                    f"send {label} from {self._device_id!r} failed: {e}"
                ) from e
```

**Note the `_closed` / `_link_up` split.** Previously a single `_closed` event meant both
"the user called `close()`" and "the peer went away", which is why a transient drop was
terminal. Separating them is what makes reconnect expressible.

### 2c. Device service — no hot spin

```python
# hermes/processes/device.py  —  DeviceService.run

    # Back-off applied when serve_once returns None. A None means "no solicit
    # arrived" (normal, the mule is in transit) or "the link is down"
    # (transient, the client is reconnecting). Previously neither case slept,
    # so a dropped link spun a core on the edge device indefinitely.
    _IDLE_BACKOFF_S: float = 0.25

    def run(self) -> None:
        log.info(
            "device %s ready: RF client to %s:%d",
            self.cfg.device_id, self.cfg.mule_rf_host, self.cfg.mule_rf_port,
        )
        n_serves = self.cfg.n_serves
        served = 0
        while not self._stop_event.is_set():
            if n_serves is not None and served >= n_serves:
                log.info("device %s: served %d times, exiting",
                         self.cfg.device_id, served)
                break
            try:
                outcome = self.client.serve_once()
            except Exception as e:
                log.exception("device %s: serve_once raised", self.cfg.device_id)
                self.events.emit("device_serve_failed", reason=repr(e))
                self.metrics.increment("serves_failed")
                self._stop_event.wait(self._IDLE_BACKOFF_S)
                continue

            if outcome is None:
                # Interruptible sleep — SIGTERM still exits within one tick.
                self._stop_event.wait(self._IDLE_BACKOFF_S)
                continue

            served += 1
            log.info("device %s: served, outcome=%s",
                     self.cfg.device_id, outcome.value)
            self.events.emit("device_served", outcome=outcome.value)
            self.metrics.increment("serves_completed")
            self.metrics.increment(f"serves_outcome_{outcome.value}")

        log.info("device %s service loop exiting", self.cfg.device_id)
```

**Behaviour change, stated plainly.** Inside 30 s of continuous activity — every case the
current test suite covers — behaviour is identical. Beyond 30 s of idle, the link now
survives where it previously died. That is a fix, not a regression, but it *is* a change
and should be validated by the new test below.

**New regression test:**

```python
# tests/integration/test_rf_link_idle_survival.py
@pytest.mark.slow
def test_device_survives_long_inter_contact_gap(rf_server, rf_client):
    """H-01: a device idle far longer than send_timeout_s stays registered.

    Reproduces the field failure: the mule flies a leg longer than the
    socket timeout, and every device in the next cluster has been dropped
    by the time it broadcasts. Fails on the pre-fix implementation.
    """
    assert rf_server.wait_for_devices([DEVICE_ID], timeout=5.0)

    idle = rf_server._send_timeout_s * 1.5          # 45 s at the default
    time.sleep(idle)

    rf_server.broadcast_open_solicit(_solicit())
    adv = rf_server.recv_ready_adv(timeout=5.0)     # pre-fix: RFLinkError
    assert adv.device_id == DEVICE_ID
```

---

## 3. Unified contact exchange

**Defect (H-02, H-03, Q-01).** `run_contact` (187 lines) and `deliver_contact` (140
lines) share ~90 % of their body; both spawn unbounded threads and join per-thread; both
can have abandoned workers write into a closed round's ledger.

### 3a. Round epoch — stop stale workers writing

```python
# hermes/mission/host_mission.py

class HFLHostMission:

    # Bound on concurrent per-device workers inside one contact event. A
    # contact is I/O-bound (one push + one recv per device), so a modest
    # pool saturates the link; the cap exists so a large S3a cluster at
    # rf_range_m=120 cannot spawn an unbounded number of OS threads.
    MAX_CONTACT_WORKERS: int = 16

    def __init__(self, ...):
        ...
        # H-02(c): monotonic epoch bumped by open_round / open_pass_2.
        # Every worker captures the epoch it was launched under; late
        # writes from an abandoned worker are dropped *loudly* instead of
        # silently landing in the next round's ledger.
        self._epoch: int = 0
```

`open_round` and `open_pass_2` each do `self._epoch += 1` inside the existing `with
self._lock` block. `_record_outcome`, `_record_contact` and `_record_delivery_line` gain
an `epoch` parameter:

```python
    def _record_outcome(
        self,
        *,
        epoch: int,
        device_id: DeviceID,
        outcome: MissionOutcome,
        contact_ts: float,
        utility: float,
        bytes_received: int,
        bytes_sent: int,
    ) -> None:
        with self._lock:
            if epoch != self._epoch:
                log.warning(
                    "dropping late outcome from epoch=%d (current=%d) "
                    "device=%s outcome=%s — worker outlived its round",
                    epoch, self._epoch, device_id, outcome.value,
                )
                self.metrics_late_outcomes = getattr(self, "metrics_late_outcomes", 0) + 1
                return
            if self._report is None:
                return
            self._report.append(MissionRoundCloseLine(...))
            mission_round = self._mission_round
        ...
```

and the accept path in the collect worker becomes epoch-checked too:

```python
            if outcome is MissionOutcome.CLEAN:
                with self._lock:
                    if epoch == self._epoch:
                        self._accepted.append(grad)
                    else:
                        log.warning(
                            "dropping late gradient from epoch=%d device=%s",
                            epoch, adv.device_id,
                        )
```

### 3b. The shared exchange engine

```python
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar
#                                          ^^^^^^^^ Q-01: was used, never imported

_R = TypeVar("_R")


    def _exchange_contact(
        self,
        devices: Sequence[DeviceID],
        *,
        pass_kind: MissionPass,
        mission_round: int,
        epoch: int,
        on_present: Callable[[DeviceID, FLReadyAdv, int], _R],
        on_absent: Callable[[DeviceID, int], _R],
        on_broadcast_failure: _R,
    ) -> Dict[DeviceID, _R]:
        """One contact event: broadcast, gather adverts, serve in parallel.

        Shared by Pass-1 ``run_contact`` and Pass-2 ``deliver_contact``.
        Everything except the per-device body is identical between the two —
        keeping one copy means the H-02 thread-management fixes and the H1
        misrouted-advert discipline exist exactly once.

        Timing contract: the whole call is bounded by ``2 * session_ttl_s``
        wall-clock — one TTL to gather adverts, one to serve them. The
        previous per-thread ``join(timeout=2*ttl)`` gave a worst case of
        ``N * 2 * ttl``, which at N=20 / ttl=30 s was twenty minutes.
        """
        devices = tuple(devices)
        if not devices:
            raise ValueError(
                f"{pass_kind.value} contact requires at least one device"
            )

        solicit = FLOpenSolicit(
            mule_id=self.mule_id,
            mission_round=mission_round,
            issued_at=time.time(),
            pass_kind=pass_kind,
        )
        try:
            self.rf.broadcast_open_solicit(solicit)
        except RFLinkError as e:
            log.warning("%s contact: broadcast failed: %s", pass_kind.value, e)
            return {did: on_broadcast_failure for did in devices}

        advs = self._gather_ready_advs(devices, deadline=time.monotonic() + self.session_ttl_s)

        outcomes: Dict[DeviceID, _R] = {}
        outcomes_lock = threading.Lock()

        def _serve(did: DeviceID) -> None:
            adv = advs.get(did)
            result = (
                on_absent(did, epoch) if adv is None
                else on_present(did, adv, epoch)
            )
            with outcomes_lock:
                outcomes[did] = result

        pool = ThreadPoolExecutor(
            max_workers=min(len(devices), self.MAX_CONTACT_WORKERS),
            thread_name_prefix=f"contact-{self.mule_id}-r{mission_round}",
        )
        try:
            futures: Dict[Future, DeviceID] = {
                pool.submit(_serve, did): did for did in devices
            }
            serve_deadline = time.monotonic() + 2.0 * self.session_ttl_s
            _done, pending = wait(
                futures, timeout=max(0.0, serve_deadline - time.monotonic()),
            )
            for fut in pending:
                did = futures[fut]
                fut.cancel()
                log.warning(
                    "%s contact: device %s did not finish within the "
                    "contact deadline; recording %s",
                    pass_kind.value, did, on_broadcast_failure,
                )
                with outcomes_lock:
                    outcomes.setdefault(did, on_broadcast_failure)
            # Surface a worker exception rather than losing it in the future.
            for fut in _done:
                exc = fut.exception()
                if exc is not None:
                    log.error(
                        "%s contact: worker for %s raised: %r",
                        pass_kind.value, futures[fut], exc,
                    )
                    with outcomes_lock:
                        outcomes.setdefault(futures[fut], on_broadcast_failure)
        finally:
            # Do NOT wait — that would restore the unbounded-latency
            # behaviour the deadline above exists to remove. Cancelled
            # futures that never started are dropped; ones already running
            # are epoch-guarded (§3a) so their late writes are rejected.
            pool.shutdown(wait=False, cancel_futures=True)

        return outcomes

    def _gather_ready_advs(
        self, devices: Sequence[DeviceID], *, deadline: float,
    ) -> Dict[DeviceID, FLReadyAdv]:
        """Collect one FLReadyAdv per expected device before ``deadline``.

        H1 discipline preserved verbatim: adverts that arrive from devices
        outside this contact's expected set go into ``_misrouted_advs`` for
        the next contact's drain, rather than being re-sent on the link (the
        mule-side TCP server does not implement the device->mule direction).
        """
        expected: Dict[DeviceID, FLReadyAdv] = {}
        expected_set = set(devices)

        leftover: List[FLReadyAdv] = []
        with self._lock:
            for adv in self._misrouted_advs:
                if adv.device_id in expected_set and adv.device_id not in expected:
                    expected[adv.device_id] = adv
                else:
                    leftover.append(adv)
            self._misrouted_advs = leftover

        while expected_set - expected.keys():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                adv = self.rf.recv_ready_adv(timeout=remaining)
            except RFLinkError:
                break
            if adv.device_id in expected_set and adv.device_id not in expected:
                expected[adv.device_id] = adv
            else:
                with self._lock:
                    self._misrouted_advs.append(adv)
        return expected
```

### 3c. The two public methods become thin

```python
    def run_contact(
        self,
        contact_devices: Sequence[DeviceID],
        synth_batch,
        *,
        min_utility: float = 0.0,
    ) -> Dict[DeviceID, MissionOutcome]:
        """Pass-1 parallel exchange-only sessions for one contact event.

        Unchanged contract: per-device outcome map; individual failures
        (TTL, bad receipt, FL_READY=False) never affect the other devices in
        the same contact.
        """
        self._require_pass(MissionPass.COLLECT, "run_contact")
        with self._lock:
            self._require_open_round()
            theta = [w.copy() for w in self._current_theta]   # type: ignore[arg-type]
            mission_round = self._mission_round
            epoch = self._epoch

        return self._exchange_contact(
            contact_devices,
            pass_kind=MissionPass.COLLECT,
            mission_round=mission_round,
            epoch=epoch,
            on_present=lambda did, adv, ep: self._collect_from_device(
                did, adv, theta=theta, mission_round=mission_round,
                synth_batch=synth_batch, min_utility=min_utility, epoch=ep,
            ),
            on_absent=lambda did, ep: self._record_absent_collect(did, epoch=ep),
            on_broadcast_failure=MissionOutcome.TIMEOUT,
        )

    def deliver_contact(
        self,
        contact_devices: Sequence[DeviceID],
        synth_batch,
    ) -> Dict[DeviceID, DeliveryOutcome]:
        """Pass-2 push-only delivery for one contact event."""
        self._require_pass(MissionPass.DELIVER, "deliver_contact")
        with self._lock:
            if self._current_theta is None or self._mission_round <= 0:
                raise MissionSessionError(
                    "deliver_contact called without open_pass_2 staging θ'"
                )
            theta = [w.copy() for w in self._current_theta]
            mission_round = self._mission_round
            epoch = self._epoch

        return self._exchange_contact(
            contact_devices,
            pass_kind=MissionPass.DELIVER,
            mission_round=mission_round,
            epoch=epoch,
            on_present=lambda did, adv, ep: self._deliver_to_device(
                did, adv, theta=theta, mission_round=mission_round,
                synth_batch=synth_batch, epoch=ep,
            ),
            on_absent=lambda did, ep: self._record_absent_delivery(did, epoch=ep),
            on_broadcast_failure=DeliveryOutcome.UNDELIVERED,
        )

    def _require_pass(self, expected: MissionPass, fn_name: str) -> None:
        current = self.current_pass
        if current is not expected:
            raise MissionSessionError(
                f"{fn_name} called in pass={current.value}; "
                f"expected {expected.value}"
            )
```

The two `_collect_from_device` / `_deliver_to_device` helpers are the existing
`_device_worker` bodies lifted out verbatim, with `epoch` threaded into their `_record_*`
calls. No logic inside them changes.

**Net effect:** `host_mission.py` drops roughly 150 duplicated lines; `run_contact`'s
complexity falls from 35 to under 10; contact wall-clock is bounded at `2 × ttl`
regardless of N; thread count is capped at 16; late writes are rejected loudly.

**Verification.** `tests/unit/test_host_mission.py`,
`tests/integration/test_two_pass_contact.py`, `test_sprint_1_5_fixes.py`,
`test_e2e_faults.py` all pass unchanged — they assert on the returned outcome maps and
the report contents, both of which are preserved. Add:

```python
def test_contact_wall_clock_is_bounded_independent_of_device_count(...):
    """H-02(b): 32 unresponsive devices must not cost 32 x 2 x ttl."""
    mission = HFLHostMission(mule_id="m", rf=SilentRFLink(), session_ttl_s=0.2)
    mission.open_round(theta)
    t0 = time.monotonic()
    outcomes = mission.run_contact([f"d{i}" for i in range(32)], synth)
    assert time.monotonic() - t0 < 2.0          # ~2*ttl, not 32*2*ttl
    assert all(o is MissionOutcome.TIMEOUT for o in outcomes.values())

def test_late_worker_cannot_write_into_the_next_round(...):
    """H-02(c): a worker that outlives close_round is rejected, not folded in."""
```

---

## 4. Cancellable dock wait

**Defect (H-04).** `wait_for_dock(timeout=None)` is an uncancellable `while True` with a
`time.sleep`, called from the supervisor at `mule_main.py:306` and `:425`.

```python
# hermes/mule/client_cluster.py

    def wait_for_dock(
        self,
        *,
        timeout: Optional[float] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """Poll ``dock.is_available()`` until True, timeout, or stop.

        ``stop_event`` makes the wait cancellable: SIGTERM during an
        inter-pass dock wait now exits within one poll interval instead of
        being absorbed until the orchestrator's SIGKILL — which previously
        discarded the Pass-1 aggregate and the delivery-report carryover
        that were about to be uploaded.

        Sleeping on ``Event.wait`` rather than ``time.sleep`` is what makes
        the cancellation prompt; a plain sleep would still hold the thread
        for a full interval after the flag is set.
        """
        with self._lock:
            self._set_state(ClientClusterState.AWAIT_DOCK)

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if stop_event is not None and stop_event.is_set():
                log.info("wait_for_dock: cancelled by stop event")
                return False
            if self.dock.is_available():
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            remaining = self.dock_poll_interval_s
            if deadline is not None:
                remaining = min(remaining, max(0.0, deadline - time.monotonic()))
            if stop_event is not None:
                if stop_event.wait(remaining):
                    return False
            else:
                time.sleep(remaining)
```

`MuleSupervisor` gains `stop_event` and `dock_wait_timeout_s` constructor parameters, and
the two call sites become:

```python
        # A dead cluster must surface as an error, not an infinite wait.
        # 300 s is generous relative to a mission (tens of seconds) while
        # still bounded.
        if not self.client_cluster.wait_for_dock(
            timeout=self.dock_wait_timeout_s, stop_event=self._stop_event,
        ):
            raise MuleSupervisorError(
                f"dock did not become available within "
                f"{self.dock_wait_timeout_s}s between passes"
            )
```

`MuleService` passes its existing `_stop_event` into the supervisor.

**Verification.** Existing tests construct the supervisor without a stop event and get
`stop_event=None` plus a generous default timeout — the current
`wait_for_dock(timeout=None)` semantics are preserved for every path that has a live
cluster. New test: SIGTERM during an inter-pass dock wait exits within one poll interval.

---

## 5. Observability that cannot crash its caller

**Defect (H-07).** `events.py:108` serializes outside the guard, contradicting the module
docstring and `_coerce`'s own comment.

```python
# hermes/observability/events.py

def _coerce(v: Any) -> Any:
    """JSON-friendly coercion of the value types that reach event lines.

    numpy scalars and 0-d arrays are the common accident: ``metrics.observe``
    and per-round evaluation both produce ``np.float32`` / ``np.float64``,
    and ``json.dumps`` rejects them. Coercing here (rather than relying on
    every call site to remember ``float(...)``) is what makes the
    "observability never crashes the caller" contract actually hold.
    """
    if isinstance(v, tuple):
        return [_coerce(x) for x in v]
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _coerce(x) for k, x in v.items()}
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, Enum):
        return v.value
    # numpy scalars / arrays — imported lazily so observability keeps its
    # stdlib-only dependency footprint when numpy is absent.
    item = getattr(v, "item", None)
    if callable(item) and getattr(v, "shape", None) == ():
        return item()
    tolist = getattr(v, "tolist", None)
    if callable(tolist) and hasattr(v, "shape"):
        return tolist()
    return v


    def emit(self, event: str, **fields: Any) -> None:
        """Append one event line. Never raises."""
        record = {
            "ts": float(self._clock()),
            "schema_version": SCHEMA_VERSION,
            "role": self._role,
            "id": self._id,
            "event": event,
        }
        for k, v in fields.items():
            if k in record:
                continue
            record[k] = _coerce(v)

        with self._lock:
            fp = self._fp
            if fp is None:
                return
            try:
                # H-07: serialization moved INSIDE the guard. It was outside,
                # so a non-serializable field raised TypeError straight into
                # the instrumented code path — the exact failure the module
                # docstring promises cannot happen. ``default=repr`` means an
                # exotic value degrades to a readable string instead of
                # dropping the whole event.
                line = json.dumps(record, separators=(",", ":"), default=repr)
                fp.write(line + "\n")
            except Exception:
                log.debug("event emit failed (event=%s)", event, exc_info=True)
```

**Verification.** Existing `test_observability.py` unchanged. Add:

```python
def test_emit_never_raises_on_numpy_or_exotic_values(tmp_path):
    em = JsonEventEmitter(tmp_path / "e.jsonl", role="test", node_id="t")
    em.emit("m", auc=np.float32(0.87), shape=np.zeros(3), obj=object())  # must not raise
    em.close()
    rec = json.loads((tmp_path / "e.jsonl").read_text().strip())
    assert rec["auc"] == pytest.approx(0.87, abs=1e-6)
    assert rec["shape"] == [0.0, 0.0, 0.0]
```

---

## 6. Inverting the `experiments` dependency

**Defect (H-08).** `hermes/processes/cluster.py:113,140,461` and `device.py:89` import
`experiments.exp4.model_task`, so the core library cannot ship without the paper harness.

`hermes/` already defines the two Protocols this needs. The fix is to resolve the
*implementation* from configuration instead of naming it:

```python
# hermes/processes/providers.py  (new)
"""Runtime resolution of pluggable model providers.

``hermes`` defines two Protocols the outside world implements:

* :class:`hermes.cluster.host_cluster.GeneratorHost`  — θ_gen + synth batch
* :class:`hermes.mission.client_mission.LocalTrainFn` — device-side training

Before this module, ``hermes.processes.{cluster,device}`` imported concrete
implementations from ``experiments.exp4.model_task`` directly, inverting the
layering: the reusable library depended on the throwaway harness, and the
real-model path only worked when CWD happened to be the repo root.

Now the config carries an entry-point string ("pkg.module:Factory") and this
module resolves it at runtime. ``hermes`` names no concrete provider, and
callers can supply their own without touching the library.
"""

from __future__ import annotations

import importlib
from typing import Any


class ProviderResolutionError(RuntimeError):
    """Raised when a configured provider cannot be imported or constructed."""


def load_provider(spec: str, **kwargs: Any) -> Any:
    """Resolve ``"package.module:attribute"`` and call it with ``kwargs``.

    Raises :class:`ProviderResolutionError` with the offending spec on any
    failure — an unimportable provider is a configuration error and must be
    loud, not a silent fall-back to the stub.
    """
    if ":" not in spec:
        raise ProviderResolutionError(
            f"provider spec {spec!r} must be 'package.module:attribute'"
        )
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ProviderResolutionError(
            f"cannot import provider module {module_name!r} from spec {spec!r}: {e}"
        ) from e
    try:
        factory = getattr(module, attr)
    except AttributeError as e:
        raise ProviderResolutionError(
            f"module {module_name!r} has no attribute {attr!r} (spec {spec!r})"
        ) from e
    try:
        return factory(**kwargs)
    except Exception as e:
        raise ProviderResolutionError(
            f"provider {spec!r} raised during construction: {e!r}"
        ) from e
```

`ClusterConfig` / `DeviceConfig` gain optional `generator_provider` /
`local_train_provider` strings, and the entry points become:

```python
# hermes/processes/cluster.py
        if cfg.generator_provider:
            self.generator = load_provider(
                cfg.generator_provider,
                init_theta_path=cfg.init_theta_path,
                input_dim=cfg.input_dim,
            )
        else:
            self.generator = StubGeneratorHost(disc_weights=_stub_disc_weights())

# hermes/processes/device.py
def _build_local_train(cfg: DeviceConfig, seed: int):
    if cfg.local_train_provider:
        return load_provider(
            cfg.local_train_provider,
            shard_path=cfg.train_shard_path,
            input_dim=cfg.input_dim,
            epochs=cfg.local_epochs,
            batch_size=cfg.local_batch_size,
            seed=seed,
        )
    return _stub_train_factory(seed)
```

Existing Exp-4 topologies keep working by setting
`"local_train_provider": "experiments.exp4.model_task:make_local_train_provider"` in the
generated config — the same code runs, but `hermes/` no longer names it.

**Verification.** `grep -rn "^from experiments\|^import experiments" hermes/` returns
nothing. `pip install hermes` with `experiments/` absent imports cleanly. Exp-4
integration tests pass with the provider string set. Add an import-graph test alongside
the existing `test_loopback_retirement.py`, which already pins a similar invariant.

---

## 7. Registry persistence and indexing

**Defect (H-06, H-05, H-09).** `save()`/`load()` are no-ops; `slice_for` is O(N) per
call; `rebalance` is round-robin over device IDs and unstable under `is_new` flips.

```python
# hermes/cluster/registry_store.py  (new)
"""Pluggable persistence for :class:`DeviceRegistry`.

The registry is the cluster's authoritative state — on-time history, missed
history, delivery priority, per-device position, and mule assignment. Before
this module its ``save``/``load`` were documented no-ops ("Phase-6 hook"),
so a cluster restart produced empty MissionSlices for every mule, empty
contact queues, and permanent "no submissions to aggregate" failures with
no operator-visible cause.

``InMemoryStore`` is the default and reproduces the previous behaviour
exactly, so nothing changes unless a store is configured.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Protocol

from hermes.types import DeviceID, DeviceRecord, MuleID, SpectrumSig


class RegistryStore(Protocol):
    def save(self, records: Iterable[DeviceRecord]) -> None: ...
    def load(self) -> List[DeviceRecord]: ...


class InMemoryStore:
    """No-op store — preserves the pre-existing (non-persistent) behaviour."""

    def save(self, records: Iterable[DeviceRecord]) -> None:
        return None

    def load(self) -> List[DeviceRecord]:
        return []


class SQLiteRegistryStore:
    """Durable store. One row per device, whole-registry upsert per snapshot.

    SQLite rather than a JSON dump because the write must be atomic against
    a mid-write crash — a torn JSON file is indistinguishable from an empty
    registry, which is the exact failure mode this is meant to eliminate.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS device_records (
        device_id        TEXT PRIMARY KEY,
        assigned_mule    TEXT,
        pos_x            REAL NOT NULL,
        pos_y            REAL NOT NULL,
        pos_z            REAL NOT NULL,
        is_new           INTEGER NOT NULL,
        on_time_history  INTEGER NOT NULL,
        missed_history   INTEGER NOT NULL,
        delivery_priority INTEGER NOT NULL,
        spectrum_sig     TEXT NOT NULL
    );
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, isolation_level="IMMEDIATE")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def save(self, records: Iterable[DeviceRecord]) -> None:
        rows = [
            (
                str(r.device_id),
                str(r.assigned_mule) if r.assigned_mule else None,
                float(r.last_known_position[0]),
                float(r.last_known_position[1]),
                float(r.last_known_position[2]),
                int(r.is_new),
                int(r.on_time_history),
                int(r.missed_history),
                int(r.delivery_priority),
                json.dumps({
                    "bands": list(r.spectrum_sig.bands),
                    "last_good_snr_per_band": list(r.spectrum_sig.last_good_snr_per_band),
                }),
            )
            for r in records
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO device_records VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(device_id) DO UPDATE SET "
                "  assigned_mule=excluded.assigned_mule,"
                "  pos_x=excluded.pos_x, pos_y=excluded.pos_y, pos_z=excluded.pos_z,"
                "  is_new=excluded.is_new,"
                "  on_time_history=excluded.on_time_history,"
                "  missed_history=excluded.missed_history,"
                "  delivery_priority=excluded.delivery_priority,"
                "  spectrum_sig=excluded.spectrum_sig",
                rows,
            )

    def load(self) -> List[DeviceRecord]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM device_records")
            out: List[DeviceRecord] = []
            for row in cur.fetchall():
                sig = json.loads(row[9])
                out.append(DeviceRecord(
                    device_id=DeviceID(row[0]),
                    assigned_mule=MuleID(row[1]) if row[1] else None,
                    last_known_position=(row[2], row[3], row[4]),
                    is_new=bool(row[5]),
                    on_time_history=row[6],
                    missed_history=row[7],
                    delivery_priority=row[8],
                    spectrum_sig=SpectrumSig(
                        bands=tuple(sig["bands"]),
                        last_good_snr_per_band=tuple(sig["last_good_snr_per_band"]),
                    ),
                ))
            return out
```

`DeviceRegistry` gains the store plus an assignment index (H-05: `slice_for` is called
once per mule per dock, each time scanning every device):

```python
class DeviceRegistry:

    def __init__(self, store: Optional[RegistryStore] = None) -> None:
        self._records: Dict[DeviceID, DeviceRecord] = {}
        # H-05: maintained alongside _records so slice_for is O(slice) rather
        # than O(all devices). At 1 K devices x M mules x every dock the old
        # full scan dominated dock latency.
        self._by_mule: Dict[MuleID, Set[DeviceID]] = {}
        self._lock = threading.RLock()
        self._round_counter = 0
        self._store: RegistryStore = store or InMemoryStore()

    def slice_for(self, mule_id: MuleID) -> Tuple[DeviceID, ...]:
        with self._lock:
            return tuple(sorted(self._by_mule.get(mule_id, ())))

    def snapshot_to_store(self) -> None:
        """Persist current state. Called on every cluster-round close."""
        with self._lock:
            records = list(self._records.values())
        self._store.save(records)

    @classmethod
    def restore(cls, store: RegistryStore) -> "DeviceRegistry":
        reg = cls(store=store)
        for rec in store.load():
            reg._records[rec.device_id] = rec
            if rec.assigned_mule:
                reg._by_mule.setdefault(rec.assigned_mule, set()).add(rec.device_id)
        return reg
```

And the H-09 assignment fix, defaulting to today's behaviour:

```python
    def rebalance(
        self,
        mules: Iterable[MuleID],
        *,
        round_counter: Optional[int] = None,
        now: Optional[float] = None,
        strategy: str = "round_robin",       # "round_robin" | "spatial"
    ) -> Dict[MuleID, MissionSlice]:
        """Disjoint slice over all devices.

        ``strategy="round_robin"`` is the historical behaviour and stays the
        default so nothing changes without an explicit opt-in.

        ``strategy="spatial"`` addresses two defects in the round-robin path:

        * It ignores ``last_known_position`` entirely, so two devices 500 m
          apart with adjacent IDs go to different mules while two devices at
          the same spot also go to different mules — S3a can never cluster
          them into one contact.
        * Its sort key ``(not is_new, device_id)`` makes assignment unstable:
          when one device's ``is_new`` flips to False, every device after it
          shifts a slot and roughly half the fleet is reassigned, discarding
          each moved device's cached scheduler state and delivery-priority
          carryover on the mule.

        The spatial strategy partitions by position and breaks ties with
        rendezvous hashing on ``(device_id, mule_id)``, so adding or removing
        a mule moves only the devices that must move.
        """
```

**Verification.** Every existing registry test passes untouched against the
`InMemoryStore` + `round_robin` defaults. New tests: kill-and-restart preserves history;
`slice_for` is O(slice); a spatial rebalance after an `is_new` flip moves ≤1 device.

---

## 8. Summary of expected impact

| Change | LOC delta | Behaviour in tested regime | Fixes |
|---|---|---|---|
| `_sockopt.py` + RF socket lifetime | +130 | identical | H-01 |
| Client reconnect + device backoff | +70 | identical | H-01 |
| Unified contact exchange | **−150** | identical | H-02, H-03, Q-01 |
| Cancellable dock wait | +25 | identical | H-04 |
| Observability guard | +20 | identical | H-07 |
| Provider indirection | +60 | identical | H-08 |
| Registry store + index + strategy | +220 | identical (defaults) | H-06, H-05, H-09 |

Net ≈ **+375 LOC** for seven defect classes, of which one (H-01) is the difference
between a system that works in a 30-second smoke test and one that works in the field.
No public API in `hermes/` changes signature except by adding optional keyword arguments
with backward-compatible defaults.
