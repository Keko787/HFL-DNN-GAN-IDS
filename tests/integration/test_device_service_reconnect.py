"""Amendment 10 (finding P-02), device side: ``DeviceService`` over real TCP.

Before the amendment a device whose RF link dropped never got it back, and
``DeviceService.run`` spun on the dead link: ``serve_once`` returned None at
once, forever (``probe_p02.py`` counts about 350,000 calls in 0.2 s at
afa9526 on the development host). Now the loop re-dials its mule with
backoff, and records only a successful re-dial: one the mule acknowledged,
so another listener that took the port after the mule exited, or a mule of
another trial with a different link token, does not count.

The service runs in-process on a thread, with its backoff, solicit poll and
connect timeout shortened on the instance (a refused dial takes about 2 s
on Windows otherwise). The backoff's schedule is checked exactly by standing
a recording event in for the service's stop event, and on the wall clock
with lower bounds only (a busy host makes waits longer, never shorter).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from hermes.processes.config import DeviceConfig
from hermes.processes.device import DeviceService
from hermes.transport import RFLinkError, TCPDockLinkServer, TCPRFLinkClient, TCPRFLinkServer
from hermes.types import DeviceID, DiscPush, FLOpenSolicit, MissionPass, MuleID

MULE = MuleID("mule-dev")
D0 = DeviceID("d0")


class _Recorder:
    """Stands in for the JSONL emitter; keeps (event, fields) in order."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, Dict[str, Any]]] = []

    def emit(self, event: str, **fields: Any) -> None:
        self.rows.append((event, fields))

    def close(self) -> None:
        pass

    def names(self) -> List[str]:
        return [e for e, _ in self.rows]

    def metrics(self) -> Dict[str, Any]:
        """The metrics the service emitted at shutdown."""
        return dict(self.rows)["metrics_snapshot"]["metrics"]


class _RecordingStop(threading.Event):
    """Stands in for ``DeviceService._stop_event`` in the backoff checks.

    Records the wait the backoff asks for each time and returns at once;
    after ``stop_after`` waits it is set, as if a stop had come then.
    """

    def __init__(self, stop_after: int) -> None:
        super().__init__()
        self.waits: List[Optional[float]] = []
        self._stop_after = stop_after

    def wait(self, timeout: Optional[float] = None) -> bool:
        self.waits.append(timeout)
        if len(self.waits) >= self._stop_after:
            self.set()
        return self.is_set()


def _server(port: int = 0, **kwargs) -> TCPRFLinkServer:
    s = TCPRFLinkServer(host="127.0.0.1", port=port, **kwargs)
    s.start()
    return s


def _service(port: int, **cfg_kwargs) -> Tuple[DeviceService, _Recorder]:
    rec = _Recorder()
    cfg = DeviceConfig(
        device_id=str(D0), mule_rf_host="127.0.0.1", mule_rf_port=port, **cfg_kwargs,
    )
    svc = DeviceService(cfg, events=rec)
    svc._RECONNECT_INITIAL_S = 0.05
    svc._RECONNECT_MAX_S = 0.2
    svc.client.solicit_timeout_s = 0.2
    svc.rf._connect_timeout_s = 0.3
    return svc, rec


def _start(svc: DeviceService) -> threading.Thread:
    t = threading.Thread(target=svc.run, name="device-service-under-test", daemon=True)
    t.start()
    return t


def _stop(svc: DeviceService, t: threading.Thread) -> None:
    svc.request_stop()
    t.join(timeout=5.0)
    assert not t.is_alive(), "service loop did not honour request_stop"
    svc.shutdown()


def _wait_until(pred, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _collect_contact(server: TCPRFLinkServer, mission_round: int) -> None:
    """One Pass-1 contact end to end: solicit, advert, push, gradient."""
    server.broadcast_open_solicit(FLOpenSolicit(
        mule_id=MULE, mission_round=mission_round, issued_at=time.time(),
        pass_kind=MissionPass.COLLECT,
    ))
    assert server.recv_ready_adv(timeout=5.0).device_id == D0
    server.push_disc(D0, DiscPush(
        mule_id=MULE, mission_round=mission_round,
        theta_disc=[np.ones((4,), dtype=np.float32)], synth_batch=[],
    ))
    assert server.recv_gradient(D0, timeout=5.0).mission_round == mission_round


def _count_calls(fn, log: List[float]):
    """Wrap ``fn`` so each call appends its monotonic start time to ``log``."""

    def _counted(*args, **kwargs):
        log.append(time.monotonic())
        return fn(*args, **kwargs)

    return _counted


def _refuse() -> None:
    """A re-dial that fails at once (stands in for ``rf.reconnect``)."""
    raise RFLinkError("refused (test)")


@pytest.fixture(scope="module")
def never_dropped() -> _Recorder:
    """What a device records over an idle stretch on a link that never drops.

    The trace a device leaves when its link goes and does not come back
    (every trial's end, when the mule exits) must be exactly this one.
    """
    server = _server()
    svc, rec = _service(server.port)
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        t = _start(svc)
        time.sleep(1.0)
        _stop(svc, t)
    finally:
        server.close()
    return rec


# --------------------------------------------------------------------------- #

def test_a_dead_link_does_not_spin_the_service_loop(never_dropped):
    server = _server()
    svc, rec = _service(server.port)
    counts = {"serve_once": 0, "reconnect": 0}
    serve_once, reconnect = svc.client.serve_once, svc.rf.reconnect

    def _counted_serve():
        counts["serve_once"] += 1
        return serve_once()

    def _counted_reconnect():
        counts["reconnect"] += 1
        return reconnect()

    svc.client.serve_once = _counted_serve
    svc.rf.reconnect = _counted_reconnect
    assert server.wait_for_devices([D0], timeout=2.0)
    server.close()  # the mule goes away for good
    assert _wait_until(lambda: not svc.rf.connected)

    t = _start(svc)
    time.sleep(1.5)
    _stop(svc, t)

    # Before the fix: hundreds of thousands of serve_once calls here.
    assert counts["serve_once"] <= 2, counts
    # Re-dials follow the backoff (0.05, 0.1, then 0.2 s plus each failed
    # dial); the exact schedule is checked below.
    assert 1 <= counts["reconnect"] <= 12, counts
    # A link that never came back (every trial's end, when the mule exits)
    # leaves the device's trace as a link that never dropped does: the same
    # events and the same metrics, whatever they are named.
    assert rec.names() == never_dropped.names()
    assert rec.metrics() == never_dropped.metrics()


def test_redial_waits_double_up_to_the_cap():
    server = _server()
    svc, rec = _service(server.port)  # backoff 0.05 s doubling to 0.2 s
    try:
        dials: List[float] = []
        svc.rf.reconnect = _count_calls(_refuse, dials)
        svc._stop_event = _RecordingStop(stop_after=6)
        assert svc._reconnect_with_backoff() is False  # the stop came
        assert svc._stop_event.waits == pytest.approx([0.05, 0.1, 0.2, 0.2, 0.2, 0.2])
        assert len(dials) == 5  # one dial after each wait but the last
        # Failed dials record nothing.
        assert svc.metrics.snapshot() == {}
        assert rec.names() == ["device_ready"]
    finally:
        svc.shutdown()
        server.close()


def test_a_relinked_link_that_drops_again_keeps_backing_off():
    # Two devices under one id on one mule evict each other at every re-dial:
    # each re-dial succeeds (the mule acknowledges it) and the link is lost
    # again soon after. The waits must keep growing, not start over.
    server = _server()
    svc, rec = _service(server.port)
    try:
        svc.rf.reconnect = lambda: None  # every re-dial succeeds at once
        first_waits = []
        for _ in range(4):
            svc._stop_event = _RecordingStop(stop_after=99)
            assert svc._reconnect_with_backoff() is True
            first_waits.append(svc._stop_event.waits)
        assert first_waits == [pytest.approx([w]) for w in (0.05, 0.1, 0.2, 0.2)]

        # A link that held for the hold time starts over at the initial step.
        svc._RECONNECT_HOLD_S = 0.0
        svc._stop_event = _RecordingStop(stop_after=99)
        assert svc._reconnect_with_backoff() is True
        assert svc._stop_event.waits == pytest.approx([0.05])

        # Every one of them was acknowledged, so every one is recorded.
        assert svc.metrics.counter_value("rf_reconnects") == 5
        assert rec.names().count("device_reconnected") == 5
    finally:
        svc.shutdown()
        server.close()


def test_the_service_loop_is_paced_by_the_backoff():
    # Every re-dial is refused at once, so only the backoff spaces the dials
    # (a real refused dial takes its own time on Windows, which would hide a
    # missing wait). Lower bounds only: a busy host lengthens waits.
    server = _server()
    svc, rec = _service(server.port)
    svc._RECONNECT_INITIAL_S = 0.1
    svc._RECONNECT_MAX_S = 0.4
    dials: List[float] = []
    svc.rf.reconnect = _count_calls(_refuse, dials)
    assert server.wait_for_devices([D0], timeout=2.0)
    server.close()
    assert _wait_until(lambda: not svc.rf.connected)

    started = time.monotonic()
    t = _start(svc)
    try:
        assert _wait_until(lambda: len(dials) >= 4, timeout=5.0)
    finally:
        _stop(svc, t)
    gaps = [b - a for a, b in zip(dials, dials[1:])]
    # Waits of 0.1, 0.2, 0.4 and 0.4 s before the four dials.
    assert dials[0] - started >= 0.08, dials[0] - started
    assert gaps[0] >= 0.16 and gaps[1] >= 0.32 and gaps[2] >= 0.32, gaps


def test_a_listener_that_is_not_the_mule_gives_no_reconnect(never_dropped):
    # The device's mule exits and a cluster's dock server (another trial's,
    # say) binds the same port. It rejects the device's registration and
    # closes. Before re-dials waited for the mule's acknowledgement, every
    # one of them counted as a reconnect (seven in six seconds, with the
    # default backoff).
    server = _server()
    port = server.port
    svc, rec = _service(port)
    dials: List[float] = []
    svc.rf.reconnect = _count_calls(svc.rf.reconnect, dials)
    dock = None
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        server.close()
        assert _wait_until(lambda: not svc.rf.connected)
        dock = TCPDockLinkServer(host="127.0.0.1", port=port)
        dock.start()
        t = _start(svc)
        try:
            assert _wait_until(lambda: len(dials) >= 3, timeout=5.0)
        finally:
            _stop(svc, t)
        assert dock.registered_mules() == []
    finally:
        server.close()
        if dock is not None:
            dock.close()
    assert not svc.rf.connected
    assert rec.names() == never_dropped.names()
    assert rec.metrics() == never_dropped.metrics()


def test_a_mule_of_another_trial_does_not_take_the_device():
    # With link tokens, a device whose mule exited cannot register with a
    # mule of another trial that binds the same port, and so cannot evict
    # that mule's own device of the same id.
    server = _server(link_token="trial-A")
    port = server.port
    svc, rec = _service(port, rf_link_token="trial-A")
    dials: List[float] = []
    svc.rf.reconnect = _count_calls(svc.rf.reconnect, dials)
    other = own = None
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        server.close()
        assert _wait_until(lambda: not svc.rf.connected)
        other = _server(port, link_token="trial-B")
        own = TCPRFLinkClient(D0, "127.0.0.1", port, link_token="trial-B")
        assert other.wait_for_devices([D0], timeout=2.0)
        own_sock = other._sockets[D0]
        t = _start(svc)
        try:
            assert _wait_until(lambda: len(dials) >= 3, timeout=5.0)
        finally:
            _stop(svc, t)
        assert other._sockets[D0] is own_sock
        assert own.connected
    finally:
        server.close()
        if own is not None:
            own.close()
        if other is not None:
            other.close()
    assert "device_reconnected" not in rec.names()
    assert svc.metrics.counter_value("rf_reconnects") == 0


def test_device_re_registers_after_the_mule_drops_its_socket():
    server = _server()
    svc, rec = _service(server.port)
    t = _start(svc)
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        # What the old 30 s read timeout did to a silent device (test poke).
        server._drop_device(D0)
        assert _wait_until(lambda: svc.metrics.counter_value("rf_reconnects") == 1)
        assert server.wait_for_devices([D0], timeout=2.0)
        _collect_contact(server, 1)
    finally:
        _stop(svc, t)
        server.close()
    reconnects = [f for e, f in rec.rows if e == "device_reconnected"]
    assert len(reconnects) == 1 and reconnects[0]["attempts"] >= 1
    assert reconnects[0]["down_s"] >= 0.0


def test_device_re_dials_a_restarted_mule_with_backoff():
    server = _server()
    port = server.port
    svc, rec = _service(port)
    t = _start(svc)
    server2 = None
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        server.close()
        time.sleep(0.6)  # the mule is down for a while: some dials fail
        assert svc.metrics.counter_value("rf_reconnects") == 0
        server2 = _server(port)
        assert server2.wait_for_devices([D0], timeout=5.0)
        _collect_contact(server2, 1)
    finally:
        _stop(svc, t)
        server.close()
        if server2 is not None:
            server2.close()
    reconnects = [f for e, f in rec.rows if e == "device_reconnected"]
    assert len(reconnects) == 1 and reconnects[0]["attempts"] >= 2
    # Down about as long as the mule was away (0.6 s, less the up to 0.2 s the
    # loop takes to notice), and reported in seconds.
    assert 0.3 <= reconnects[0]["down_s"] < 30.0
    assert svc.metrics.counter_value("rf_reconnects") == 1


def test_a_healthy_link_serves_as_before():
    server = _server()
    svc, rec = _service(server.port, n_serves=1)
    t = _start(svc)
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        _collect_contact(server, 1)
        t.join(timeout=5.0)
        assert not t.is_alive()  # n_serves reached: the loop exits by itself
    finally:
        _stop(svc, t)
        server.close()
    assert rec.names() == ["device_ready", "device_served", "metrics_snapshot", "service_stopped"]
    assert svc.metrics.counter_value("rf_reconnects") == 0


@pytest.mark.parametrize("flag", [False, True])
def test_newest_solicit_only_reaches_the_rf_client(flag):
    server = _server()
    try:
        svc, _rec = _service(server.port, newest_solicit_only=flag)
        try:
            assert svc.rf._newest_solicit_only is flag
        finally:
            svc.shutdown()
    finally:
        server.close()


@pytest.mark.parametrize("token", [None, "trial-7"])
def test_rf_link_token_reaches_the_rf_client(token):
    server = _server()  # no token of its own: registers any device
    try:
        svc, _rec = _service(server.port, rf_link_token=token)
        try:
            assert svc.rf._link_token == token
            assert server.wait_for_devices([D0], timeout=2.0)
        finally:
            svc.shutdown()
    finally:
        server.close()
