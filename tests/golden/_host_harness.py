"""Deterministic harness for ``HFLHostMission``'s contact routines at afa9526.

A port of the contact map's ``probe_equivalence.py`` (28 scripted scenarios
covering every branch of ``run_contact`` and ``deliver_contact``: refusals,
silent devices, failed pushes, bad receipts, the misrouted-advert stash, wrong
pass, missing round), plus ``run_session``'s branches (``S ...``), recording
what legacy mode must keep byte-identical (design section 5.2; unit U5 merges
the two contact routines behind a sink and leaves ``run_session`` alone):

* the returned outcome map (with order) or the exception;
* every ``RoundCloseDelta`` on the scheduler bus;
* the report, contact and delivery ledgers, the accepted submissions, the
  stash and the busy flags;
* every RF call (messages as their afa9526 field sets, not wire hashes:
  critic A3), and every log record of ``hermes.mission``.

Differences from the probe, and why:

* The clock (``time.time``) advances only when the fake link is called, by
  1 ms per call, never on a read. A stamp therefore says where in the RF
  exchange it was taken, and an extra clock read added by a later change does
  not shift every stamp after it.
* Each scenario runs with synchronous threads (exact comparison) and with real
  threads. With real threads a multi-device contact's workers interleave in no
  fixed order (critic R6), so those runs are compared order-free with the
  stamps masked; single-device contacts stay exact, since the only worker runs
  while the caller is blocked in ``join``.
* The late-writer scenario (critic A2) is added: a gradient that arrives after
  ``close_round`` and ``open_round`` lands in round 2's accepted list.
* The sequential joins (P-01 defect 2) are added, for both passes, with real
  threads: each worker is joined in turn for up to 2 x TTL, so a reply that
  comes after the first join gave up, inside the second, is still in the map
  the routine returns.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from hermes.mission.aggregation_rules import AGG_CUTOFF, AggregationSpec
from hermes.mission.host_mission import HFLHostMission, MissionSessionError
from hermes.transport import RFLink, RFLinkError
from hermes.types import (
    DeliveryAck,
    DeviceID,
    FLReadyAdv,
    FLState,
    GradientSubmission,
    MissionPass,
    MuleID,
)
from hermes.types.fl_messages import UPDATE_FORM_WEIGHTS

from tests.golden._canon import canon, mask_floats

MULE = MuleID("m")
T0 = 1_000_000.0
#: Stamps within this band come from the harness clock or the scripted
#: messages; they are masked in the order-free comparisons.
TS_BAND = (T0 - 2000.0, T0 + 2000.0)
_real_time = time.time
_real_thread = threading.Thread


class StepClock:
    """``time.time`` for the harness: moved by the fake link, never by a read."""

    def __init__(self, t0: float = T0) -> None:
        self.t = float(t0)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 0.001) -> None:
        with self._lock:
            self.t += dt


class SyncThread:
    """``threading.Thread`` while a harness runs: HERMES targets run on ``start()``.

    A target defined in ``hermes`` (the mission server's per-device workers, a
    device's train-ahead fit) runs synchronously, so their order is fixed and
    ``join()`` returns at once. Any other target (a library's worker loop, say
    a thread pool's) still gets a real thread, so the harness cannot hang on it.
    """

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}
        module = getattr(target, "__module__", None) or ""
        self._real = None if module.startswith("hermes") else _real_thread(
            group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

    def start(self) -> None:
        if self._real is not None:
            self._real.start()
        elif self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout=None) -> None:
        if self._real is not None:
            self._real.join(timeout)

    def is_alive(self) -> bool:
        return self._real is not None and self._real.is_alive()


class FakeRF(RFLink):
    """Scripted, never-blocking link that logs every mule-side call."""

    def __init__(self, clock: StepClock, *, advs=(), grads=None, acks=None,
                 fail_broadcast=False, fail_push=()):
        self.clock = clock
        self.advs = list(advs)
        self.grads = dict(grads or {})
        self.acks = dict(acks or {})
        self.fail_broadcast = fail_broadcast
        self.fail_push = set(fail_push)
        self.calls: List[Any] = []
        self._lock = threading.Lock()

    def _log(self, *entry) -> None:
        with self._lock:
            self.clock.tick()
            self.calls.append(list(entry))

    # mule side
    def broadcast_open_solicit(self, msg):
        self._log("solicit", canon(msg))
        if self.fail_broadcast:
            raise RFLinkError("link closed")

    def recv_ready_adv(self, timeout=None):
        self._log("recv_adv", canon(timeout))
        with self._lock:
            if not self.advs:
                raise RFLinkError("recv_ready_adv timed out")
            return self.advs.pop(0)

    def push_disc(self, device_id, msg):
        self._log("push", str(device_id), canon(msg))
        if device_id in self.fail_push:
            raise RFLinkError(f"push to {device_id} failed")

    def recv_gradient(self, device_id, timeout=None):
        self._log("recv_grad", str(device_id), canon(timeout))
        g = self.grads.get(device_id)
        if g is None:
            raise RFLinkError("recv_gradient timed out")
        return g

    def recv_delivery_ack(self, device_id, timeout=None):
        self._log("recv_ack", str(device_id), canon(timeout))
        a = self.acks.get(device_id)
        if a is None:
            raise RFLinkError("recv_delivery_ack timed out")
        return a

    # device side: never used by the mule
    def recv_open_solicit(self, device_id, timeout=None):
        raise NotImplementedError

    def send_ready_adv(self, msg):
        raise NotImplementedError

    def recv_disc_push(self, device_id, timeout=None):
        raise NotImplementedError

    def send_gradient(self, msg):
        raise NotImplementedError

    def send_delivery_ack(self, msg):
        raise NotImplementedError

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# Scripted messages (the probe's)
# --------------------------------------------------------------------------- #

def theta(v: float = 0.0):
    return [np.full((4,), v, dtype=np.float32), np.full((3, 3), v, dtype=np.float32)]


def adv(did, *, state=FLState.FL_OPEN, utility=0.7, ts=T0 - 5.0):
    return FLReadyAdv(device_id=DeviceID(did), state=state, performance_score=utility,
                      diversity_adjusted=0.3, utility=utility, issued_at=ts,
                      local_loss=0.2, num_examples=11)


def grad(did, rnd, *, ts=T0 + 0.5, form=UPDATE_FORM_WEIGHTS, corrupt=False, basis=5):
    g = GradientSubmission(device_id=DeviceID(did), mule_id=MULE, mission_round=rnd,
                           delta_theta=theta(1.0), num_examples=17, submitted_at=ts,
                           local_loss=0.3, basis_version=basis, update_form=form)
    if corrupt:
        g.checksum = "00" * 32
    return g


def ack(did, *, ts=T0 + 0.7):
    return DeliveryAck(device_id=DeviceID(did), mule_id=MULE, mission_round=1,
                       weights_sig="x", received_at=ts)


D = [DeviceID(f"d{i}") for i in range(5)]
SYNTH = [np.zeros((8,), np.float32)]


def _p1(rf_kw, calls, **host_kw):
    def script(host):
        host.open_round(theta(0.5), theta_version=5)
        return [host.run_contact(devs, SYNTH, **kw) for devs, kw in calls]
    return host_kw, rf_kw, script


def _p2(rf_kw, calls, pre=None, **host_kw):
    def script(host):
        host.open_round(theta(0.5), theta_version=5)
        if pre:
            pre(host)
        host.open_pass_2(theta(2.0), theta_version=6)
        return [host.deliver_contact(devs, SYNTH) for devs in calls]
    return host_kw, rf_kw, script


def scenarios() -> Dict[str, Tuple[dict, dict, Callable]]:
    """The probe's 28 scenarios: name -> (host kwargs, link kwargs, script)."""
    s: Dict[str, Tuple[dict, dict, Callable]] = {}
    s["P1 clean"] = _p1(dict(advs=[adv("d0")], grads={D[0]: grad("d0", 1)}), [([D[0]], {})])
    s["P1 refused state"] = _p1(dict(advs=[adv("d0", state=FLState.UNAVAILABLE)]), [([D[0]], {})])
    s["P1 refused min_utility"] = _p1(dict(advs=[adv("d0", utility=0.1)]),
                                      [([D[0]], {"min_utility": 0.5})])
    s["P1 silent"] = _p1(dict(), [([D[0]], {})])
    s["P1 push fails"] = _p1(dict(advs=[adv("d0")], fail_push={D[0]}), [([D[0]], {})])
    s["P1 grad timeout"] = _p1(dict(advs=[adv("d0")]), [([D[0]], {})])
    s["P1 bad checksum"] = _p1(dict(advs=[adv("d0")], grads={D[0]: grad("d0", 1, corrupt=True)}),
                               [([D[0]], {})])
    s["P1 wrong round"] = _p1(dict(advs=[adv("d0")], grads={D[0]: grad("d0", 99)}), [([D[0]], {})])
    s["P1 ttl expired"] = _p1(dict(advs=[adv("d0")], grads={D[0]: grad("d0", 1, ts=T0 - 1000)}),
                              [([D[0]], {})])
    s["P1 delta rule, wrong form, train_ahead"] = _p1(
        dict(advs=[adv("d0")], grads={D[0]: grad("d0", 1)}), [([D[0]], {})],
        aggregation=AggregationSpec(rule=AGG_CUTOFF), train_ahead=True)
    s["P1 broadcast fails"] = _p1(dict(fail_broadcast=True), [([D[0], D[1]], {})])
    s["P1 misrouted stash then drain"] = _p1(
        dict(advs=[adv("d1"), adv("d0")], grads={D[0]: grad("d0", 1), D[1]: grad("d1", 1)}),
        [([D[0]], {}), ([D[1]], {})])
    s["P1 multi mixed"] = _p1(
        dict(advs=[adv("d2"), adv("d4"), adv("d0"), adv("d1")],
             grads={D[0]: grad("d0", 1), D[2]: grad("d2", 1, corrupt=True)}),
        [([D[0], D[1], D[2], D[3]], {})])

    def wrong_pass(host):
        host.open_round(theta(), theta_version=1)
        host.open_pass_2(theta(), theta_version=2)
        return host.run_contact([D[0]], [])
    s["P1 wrong pass"] = ({}, {}, wrong_pass)
    s["P1 empty"] = ({}, {}, lambda host: (host.open_round(theta()), host.run_contact([], []))[1])
    s["P1 no open round"] = ({}, {}, lambda host: host.run_contact([D[0]], []))

    def after_close(host):
        host.open_round(theta(), theta_version=1)
        try:
            host.close_round()
        except MissionSessionError:          # nothing accepted
            pass
        return host.run_contact([D[0]], [])
    s["P1 after close_round"] = ({}, {}, after_close)

    s["P2 delivered"] = _p2(dict(advs=[adv("d0")], acks={D[0]: ack("d0")}), [[D[0]]])
    s["P2 silent"] = _p2(dict(), [[D[0]]])
    s["P2 push fails"] = _p2(dict(advs=[adv("d0")], fail_push={D[0]}), [[D[0]]])
    s["P2 ack timeout"] = _p2(dict(advs=[adv("d0")]), [[D[0]]])
    s["P2 broadcast fails"] = _p2(dict(fail_broadcast=True), [[D[0], D[1]]])
    s["P2 ineligible still pushed"] = _p2(
        dict(advs=[adv("d0", state=FLState.UNAVAILABLE)], acks={D[0]: ack("d0")}), [[D[0]]])
    s["P2 multi mixed"] = _p2(dict(advs=[adv("d1"), adv("d0"), adv("d4")], acks={D[0]: ack("d0")}),
                              [[D[0], D[1], D[2]]])

    def pre_stash(host):
        host.run_contact([D[0]], [])         # link advs: d1 then d0 -> d1 stashed
    s["P2 drains a Pass-1 stash"] = _p2(
        dict(advs=[adv("d1"), adv("d0")], grads={D[0]: grad("d0", 1)}, acks={D[1]: ack("d1")}),
        [[D[1]]], pre=pre_stash)

    def p2_wrong_pass(host):
        host.open_round(theta(), theta_version=1)
        return host.deliver_contact([D[0]], [])
    s["P2 wrong pass"] = ({}, {}, p2_wrong_pass)

    def p2_empty(host):
        host.open_round(theta(), theta_version=1)
        host.open_pass_2(theta(), theta_version=2)
        return host.deliver_contact([], [])
    s["P2 empty"] = ({}, {}, p2_empty)

    def p2_unstaged(host):
        host._current_pass = MissionPass.DELIVER   # forced, no θ' staged
        return host.deliver_contact([D[0]], [])
    s["P2 unstaged theta"] = ({}, {}, p2_unstaged)
    s.update(session_scenarios())
    return s


def _s(rf_kw, n=1, **host_kw):
    """``run_session`` (the single-pass driver, untouched by design) ``n`` times."""
    def script(host):
        host.open_round(theta(0.5), theta_version=5)
        return [{"session": host.run_session(SYNTH)} for _ in range(n)]
    return host_kw, rf_kw, script


def session_scenarios() -> Dict[str, Tuple[dict, dict, Callable]]:
    """``run_session``'s branches: the design leaves it untouched, U5 must too."""
    s: Dict[str, Tuple[dict, dict, Callable]] = {}
    s["S clean"] = _s(dict(advs=[adv("d0")], grads={D[0]: grad("d0", 1)}))
    s["S silent"] = _s(dict())
    s["S refused state"] = _s(dict(advs=[adv("d0", state=FLState.UNAVAILABLE)]))
    s["S push fails"] = _s(dict(advs=[adv("d0")], fail_push={D[0]}))
    s["S grad timeout"] = _s(dict(advs=[adv("d0")]))
    s["S bad checksum"] = _s(dict(advs=[adv("d0")], grads={D[0]: grad("d0", 1, corrupt=True)}))
    s["S wrong round"] = _s(dict(advs=[adv("d0")], grads={D[0]: grad("d0", 99)}))
    s["S ttl expired"] = _s(dict(advs=[adv("d0")], grads={D[0]: grad("d0", 1, ts=T0 - 1000)}))
    s["S delta rule, wrong form, train_ahead"] = _s(
        dict(advs=[adv("d0")], grads={D[0]: grad("d0", 1)}),
        aggregation=AggregationSpec(rule=AGG_CUTOFF), train_ahead=True)
    s["S first come first served"] = _s(
        dict(advs=[adv("d2"), adv("d0")], grads={D[0]: grad("d0", 1), D[2]: grad("d2", 1)}), n=3)

    def refused_min_utility(host):
        host.open_round(theta(0.5), theta_version=5)
        return [{"session": host.run_session(SYNTH, min_utility=0.5)}]
    s["S refused min_utility"] = ({}, dict(advs=[adv("d0", utility=0.1)]), refused_min_utility)
    s["S no open round"] = ({}, {}, lambda host: [{"session": host.run_session(SYNTH)}])
    return s


def is_multi(name: str) -> bool:
    """Scenarios whose contact runs several workers at once."""
    return "multi" in name


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #

class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: List[list] = []
        self._lock_rec = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock_rec:
            self.records.append([record.levelname, record.name, str(record.msg),
                                 record.getMessage()])


class _Patched:
    """``time.time`` (and optionally ``threading.Thread``) swapped for a run.

    The ``hermes.mission`` logger is captured at DEBUG without propagating, so
    the records are the routine's own, whatever the test runner configured.
    """

    def __init__(self, clock: StepClock, *, sync: bool):
        self.clock, self.sync = clock, sync
        self.handler = ListHandler()

    def __enter__(self):
        self.logger = logging.getLogger("hermes.mission")
        self._old = (self.logger.level, self.logger.propagate)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        time.time = self.clock
        if self.sync:
            threading.Thread = SyncThread
        return self

    def __exit__(self, *exc):
        time.time = _real_time
        threading.Thread = _real_thread
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self._old[0])
        self.logger.propagate = self._old[1]
        return False


def _result(ret) -> Any:
    def one(r):
        return [[str(k), canon(v)] for k, v in r.items()]
    if isinstance(ret, list):
        return ["ok", [one(r) for r in ret]]
    return ["ok", one(ret)]


def snapshot(host: HFLHostMission, rf: FakeRF, bus: List[Any], handler: ListHandler,
             result: Any) -> Dict[str, Any]:
    """Everything the routine wrote, as afa9526 field sets."""
    rep = host._report
    con = host._contacts
    dlv = host._delivery_report
    return {
        "result": result,
        "bus": [canon(d) for d in bus],
        "report": None if rep is None else canon(rep),
        "contacts": None if con is None else canon(con),
        "delivery": None if dlv is None else canon(dlv),
        "accepted": [canon(g) for g in host._accepted],
        "stash": [canon(a) for a in host._misrouted_advs],
        "busy": [[str(k), canon(v.until_ts)] for k, v in sorted(host._busy.items())],
        "mission_round": host.mission_round,
        "current_pass": canon(host.current_pass),
        # Copies: the lists keep growing after the snapshot.
        "rf": [list(c) for c in rf.calls],
        "logs": [list(r) for r in handler.records],
    }


def run_scenario(name: str, *, sync: bool) -> Dict[str, Any]:
    host_kw, rf_kw, script = scenarios()[name]
    clock = StepClock()
    rf = FakeRF(clock, **rf_kw)
    bus: List[Any] = []
    with _Patched(clock, sync=sync) as p:
        host = HFLHostMission(mule_id=MULE, rf=rf, scheduler_bus=bus.append,
                              session_ttl_s=3.0, busy_ttl_s=45.0, **host_kw)
        try:
            result = _result(script(host))
        except Exception as e:  # noqa: BLE001 - the exception is the result
            result = ["raised", type(e).__name__, str(e)]
        return snapshot(host, rf, bus, p.handler, result)


#: Trace parts real threads may reorder in a multi-device contact: top-level
#: lists, and the line lists inside the ledgers.
UNORDERED_KEYS = ("bus", "rf", "logs", "accepted")
UNORDERED_LEDGERS = (("report", "lines"), ("contacts", "records"), ("delivery", "lines"))


def split_unordered(trace: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """(the ordered rest, the unordered lists) of a masked copy of ``trace``.

    Stamps from the shared clock are masked first. The caller compares the
    rest exactly and each unordered list as a multiset.
    """
    t = mask_floats(trace, *TS_BAND)
    rest = dict(t)
    bags: Dict[str, List[Any]] = {}
    for key in UNORDERED_KEYS:
        bags[key] = rest.pop(key)
    for key, fld in UNORDERED_LEDGERS:
        if t[key] is not None:
            ledger = dict(t[key])
            bags[f"{key}.{fld}"] = ledger.pop(fld)
            rest[key] = ledger
    if t["result"][0] == "ok":
        # The scripts return one outcome map per contact; the pairs of every
        # map are compared as one multiset (one contact per multi scenario).
        rest["result"] = ["ok", len(t["result"][1])]
        bags["result"] = [pair for contact in t["result"][1] for pair in contact]
    return rest, bags


# --------------------------------------------------------------------------- #
# The late writer (critic A2), with real threads
# --------------------------------------------------------------------------- #

class SlowRF(RFLink):
    """One advert; the gradient is held until the test releases it."""

    def __init__(self, clock: StepClock):
        self.clock = clock
        self.released = threading.Event()
        self._gave = False
        self.calls: List[Any] = []

    def _log(self, *entry):
        self.clock.tick()
        self.calls.append(list(entry))

    def broadcast_open_solicit(self, msg):
        self._log("solicit", canon(msg))

    def recv_ready_adv(self, timeout=None):
        self._log("recv_adv")
        if self._gave:
            raise RFLinkError("none")
        self._gave = True
        return FLReadyAdv(device_id=D[0], state=FLState.FL_OPEN, performance_score=1.0,
                          diversity_adjusted=1.0, utility=1.0, issued_at=0.0)

    def push_disc(self, device_id, msg):
        self._log("push", str(device_id), canon(msg))

    def recv_gradient(self, device_id, timeout=None):
        self._log("recv_grad", str(device_id))
        self.released.wait(10.0)               # outlive the join
        return GradientSubmission(device_id=D[0], mule_id=MULE, mission_round=1,
                                  delta_theta=[np.full((2,), 2.0, np.float32)],
                                  num_examples=3, submitted_at=time.time())

    def recv_delivery_ack(self, *a, **k):
        raise RFLinkError("x")

    def recv_open_solicit(self, *a, **k):
        raise NotImplementedError

    def send_ready_adv(self, *a, **k):
        raise NotImplementedError

    def recv_disc_push(self, *a, **k):
        raise NotImplementedError

    def send_gradient(self, *a, **k):
        raise NotImplementedError

    def send_delivery_ack(self, *a, **k):
        raise NotImplementedError

    def close(self):
        pass


def run_late_writer() -> Dict[str, Any]:
    """Round 1's worker outlives its join; round 2 is open when it writes.

    The original closure reads ``self._accepted`` (and the ledgers) when it
    writes, so the round-1 gradient lands in round 2: its accepted list has
    length 1, its report and contact ledgers get the line, the delta carries
    mission round 2, and the caller's outcome map is filled in after the fact.
    Round 2's merge then refuses the mixed round. The merged routine must keep
    all of this (critic A2).
    """
    clock = StepClock()
    rf = SlowRF(clock)
    bus: List[Any] = []
    with _Patched(clock, sync=False) as p:
        host = HFLHostMission(mule_id=MULE, rf=rf, scheduler_bus=bus.append,
                              session_ttl_s=0.05)
        host.open_round([np.full((2,), 1.0, np.float32)], theta_version=1)
        out1 = host.run_contact([D[0]], [])          # join gives up after 0.1 s
        returned_at_close = [[str(k), canon(v)] for k, v in out1.items()]
        try:
            host.close_round()
            closed = ["ok"]
        except MissionSessionError as e:
            closed = ["raised", type(e).__name__, str(e)]
        unmerged = None if host.last_unmerged is None else canon(list(host.last_unmerged))
        host.open_round([np.full((2,), 1.0, np.float32)], theta_version=2)
        rf.released.set()
        # Wait for the late worker's last write: its entry in the outcome map
        # it returned, set after the accepted list, the ledgers, the delta and
        # the busy flag.
        deadline = _real_time() + 10.0
        while _real_time() < deadline and D[0] not in out1:
            time.sleep(0.005)
        round2 = snapshot(host, rf, bus, p.handler, None)
        round2["accepted_len"] = len(host._accepted)
        round2["returned_map_after"] = [[str(k), canon(v)] for k, v in out1.items()]
        # Round 2's merge then sees a round-1 submission among its own and
        # refuses the whole round (P-01 defect 3, as recorded at afa9526).
        try:
            agg, report, contacts = host.close_round()
            merge2 = {"ok": [canon(agg), canon(report), canon(contacts)]}
        except MissionSessionError as e:
            merge2 = {"raised": [type(e).__name__, str(e)],
                      "last_unmerged": None if host.last_unmerged is None
                      else canon(list(host.last_unmerged))}
        trace = {
            "returned_at_close": returned_at_close,
            "close_round_1": closed,
            "last_unmerged_1": unmerged,
            "round_2": round2,
            "round_2_merge": merge2,
        }
    # The late write runs on its own thread after the release, so its stamps
    # say when the thread got to run, not what the routine did: masked.
    return mask_floats(trace, *TS_BAND)


# --------------------------------------------------------------------------- #
# Sequential joins (P-01 defect 2), with real threads
# --------------------------------------------------------------------------- #

#: Session TTL of the sequential-joins runs. Legacy joins each worker in turn
#: for up to 2 x TTL; the timing margins below are one TTL (0.3 s) each way.
JOIN_TTL_S = 0.3
#: Device A's reply is released this long after both workers began waiting.
RELEASE_A_AFTER_S = 3.0 * JOIN_TTL_S
#: Order-free parts of a sequential-joins trace: the two workers' RF calls
#: (and any log records) interleave in no fixed order.
JOINS_UNORDERED = ("rf", "logs")


class GatedRF(RFLink):
    """Adverts from every device at once; each device's reply (its gradient in
    Pass 1, its ack in Pass 2) is held until the harness opens its gate."""

    def __init__(self, clock: StepClock, devices, *, gate_wait_s: float = 10.0):
        self.clock = clock
        self.advs = [adv(str(d)) for d in devices]
        self.gates = {d: threading.Event() for d in devices}
        self.waiting = {d: threading.Event() for d in devices}
        self.gate_wait_s = gate_wait_s
        self.calls: List[Any] = []
        self._lock = threading.Lock()

    def _log(self, *entry) -> None:
        with self._lock:
            self.clock.tick()
            self.calls.append(list(entry))

    def _await_gate(self, device_id) -> None:
        self.waiting[device_id].set()
        if not self.gates[device_id].wait(self.gate_wait_s):
            raise RFLinkError(f"the gate of {device_id} never opened")

    def broadcast_open_solicit(self, msg):
        self._log("solicit", canon(msg))

    def recv_ready_adv(self, timeout=None):
        self._log("recv_adv", canon(timeout))
        with self._lock:
            if not self.advs:
                raise RFLinkError("recv_ready_adv timed out")
            return self.advs.pop(0)

    def push_disc(self, device_id, msg):
        self._log("push", str(device_id), canon(msg))

    def recv_gradient(self, device_id, timeout=None):
        self._log("recv_grad", str(device_id), canon(timeout))
        self._await_gate(device_id)
        return grad(str(device_id), 1)

    def recv_delivery_ack(self, device_id, timeout=None):
        self._log("recv_ack", str(device_id), canon(timeout))
        self._await_gate(device_id)
        return ack(str(device_id))

    def recv_open_solicit(self, *a, **k):
        raise NotImplementedError

    def send_ready_adv(self, *a, **k):
        raise NotImplementedError

    def recv_disc_push(self, *a, **k):
        raise NotImplementedError

    def send_gradient(self, *a, **k):
        raise NotImplementedError

    def send_delivery_ack(self, *a, **k):
        raise NotImplementedError

    def close(self):
        pass


def _ledger_ids(ledger, attr: str) -> Any:
    if ledger is None:
        return None
    return [[str(x.device_id), canon(x.outcome)] for x in list(getattr(ledger, attr))]


def run_sequential_joins(pass_kind: str) -> Dict[str, Any]:
    """Two workers outlive their joins (P-01 defect 2, kept in legacy mode).

    ``run_contact`` and ``deliver_contact`` join their workers one after the
    other, each for up to 2 x TTL, so a slow contact can hold the mule for
    2 x TTL per device. Devices A and B answer the solicit; neither replies.
    A's reply is released 3 x TTL after both workers began waiting (after A's
    own join gave up, inside B's), so the routine, which returns after about
    4 x TTL, returns with A's outcome in its map and B still out. B's reply is
    released only after the return and lands afterwards, into the map the
    caller already holds and into the ledgers (P-01 defect 3). A single
    2 x TTL deadline for all joins (the ferry path, design section 4.3 step 5)
    would return before A's release. Stamps are masked (real threads).
    """
    if pass_kind not in ("collect", "deliver"):
        raise ValueError(pass_kind)
    a, b = D[0], D[1]
    clock = StepClock()
    rf = GatedRF(clock, [a, b])
    bus: List[Any] = []
    released_a = threading.Event()

    def release_a() -> None:
        if rf.waiting[a].wait(10.0) and rf.waiting[b].wait(10.0):
            time.sleep(RELEASE_A_AFTER_S)
        rf.gates[a].set()
        released_a.set()

    with _Patched(clock, sync=False) as p:
        host = HFLHostMission(mule_id=MULE, rf=rf, scheduler_bus=bus.append,
                              session_ttl_s=JOIN_TTL_S)
        host.open_round(theta(0.5), theta_version=5)
        if pass_kind == "deliver":
            host.open_pass_2(theta(2.0), theta_version=6)
        releaser = _real_thread(target=release_a, daemon=True)
        releaser.start()
        if pass_kind == "collect":
            out = host.run_contact([a, b], SYNTH)
        else:
            out = host.deliver_contact([a, b], SYNTH)
        with host._lock:
            at_return = {
                "map": [[str(k), canon(v)] for k, v in list(out.items())],
                "a_released_before_return": released_a.is_set(),
                "accepted": [str(g.device_id) for g in host._accepted],
                "bus": [[str(d.device_id), canon(d.outcome)] for d in list(bus)],
                "report": _ledger_ids(host._report, "lines"),
                "delivery": _ledger_ids(host._delivery_report, "lines"),
                "busy": sorted(str(k) for k in host._busy),
            }
        rf.gates[b].set()
        # B's worker writes its outcome into the returned map last.
        deadline = _real_time() + 10.0
        while _real_time() < deadline and b not in out:
            time.sleep(0.005)
        releaser.join(10.0)
        after = snapshot(host, rf, bus, p.handler, None)
        after["returned_map_after"] = [[str(k), canon(v)] for k, v in out.items()]
    trace = mask_floats({"at_return": at_return, "after": after}, *TS_BAND)
    for key in JOINS_UNORDERED:
        trace["after"][key] = sorted(
            trace["after"][key], key=lambda v: json.dumps(v, sort_keys=True))
    return trace


def build_cases() -> Dict[str, Any]:
    """Each scenario with synchronous threads (the exact reference), the late
    writer and the sequential joins. Real-thread runs of the scenarios are
    compared against the same reference."""
    cases: Dict[str, Any] = {}
    for name in scenarios():
        cases[f"scenario:{name}"] = run_scenario(name, sync=True)
    cases["late_writer"] = run_late_writer()
    for pass_kind in ("collect", "deliver"):
        cases[f"sequential_joins:{pass_kind}"] = run_sequential_joins(pass_kind)
    return cases
