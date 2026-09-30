"""FeRRy Phase 3, unit U5: the host's ferry path (design section 4.3).

``run_contact(plan=...)`` / ``deliver_contact(plan=...)`` on the mission clock:
a numbered solicit to the plan's targets only, stale adverts, gradients and
acks rejected (critics B1, B2), one join deadline, and a commit that writes
the ledger in device order stamped at arrival + cumulative dwell (+ the listen
window), with the band and SNR on the lines, and charges the clock once for
the dwell and once for the listen, so ``clock() == max(contact_ts)``.

The link is a scripted fake whose devices answer at once from a script, so
outcomes do not depend on how the worker threads interleave; like a real link,
a receive on an empty queue blocks until its timeout, so a wait the host
should not make shows up. The few tests that need a slow device say so. The
clock is U1's real ``MissionClock`` (epoch 1e6 s) behind a recorder.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import socket
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np
import pytest

from hermes.mission import HFLHostMission, MissionSessionError
from hermes.mission.contact_plan import ContactPlan, planar_distance_m
from hermes.mission.host_mission import push_byte_count
from hermes.transport import RFLink, RFLinkError
from hermes.types import (
    DeliveryAck,
    DeliveryOutcome,
    DeviceID,
    DiscPush,
    FLOpenSolicit,
    FLReadyAdv,
    FLState,
    GradientSubmission,
    MissionDeliveryLine,
    MissionOutcome,
    MissionPass,
    MissionRoundCloseLine,
    MuleID,
)

mission_clock = pytest.importorskip("hermes.l1.mission_clock")

MULE = MuleID("m")
D = [DeviceID(f"d{i}") for i in range(9)]
FLOOR = -6.7
SIM_CEILING = 1.0e9


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

class RecClock:
    """U1's MissionClock, recording every charge."""

    def __init__(self) -> None:
        self.clock = mission_clock.MissionClock()
        self.charges: List[tuple] = []

    def __call__(self) -> float:
        return self.clock()

    def advance(self, dt: float, kind: str) -> float:
        self.charges.append((dt, kind))
        return self.clock.advance(dt, kind)


def theta(v: float = 0.5):
    return [np.full((4,), v, dtype=np.float32), np.full((3, 3), v, dtype=np.float32)]


SYNTH = [np.zeros((2, 4), np.float32)]


def make_grad(did, rnd, *, in_reply_to, v=1.0, corrupt=False,
              age_s=0.0) -> GradientSubmission:
    g = GradientSubmission(device_id=did, mule_id=MULE, mission_round=rnd,
                           delta_theta=theta(v), num_examples=7,
                           submitted_at=time.time() - age_s,
                           local_loss=0.2, basis_version=5, in_reply_to=in_reply_to)
    if corrupt:
        g.checksum = "00" * 32
    return g


class FerryRF(RFLink):
    """Mule-side fake whose devices answer from a per-device script.

    ``script[did]`` keys (all optional):

    * ``adv``: False = silent; "wrong_id" = answers with another solicit's
      number; default True.
    * ``state``: the advert's FLState (default FL_OPEN); ``utility``.
    * ``push_fails``: push_disc raises.
    * ``reply``: False = no gradient/ack after the push; default True.
    * ``stale_first``: a stale gradient/ack is queued before the fresh one.
    * ``corrupt``: the update's checksum is wrong.
    * ``age_s``: the update's ``submitted_at`` is this many wall seconds old.
    * ``push_delay``: seconds push_disc sleeps (real wall time).
    * ``push_gate`` / ``recv_gate``: a threading.Event the call waits on.

    Like a real link, a receive on an empty queue waits until something
    arrives or its timeout runs out (a zero timeout polls, as the host's drain
    does). ``adv_timeouts`` records the timeout of every advert receive, and
    ``on_recv(kind, device_id, timeout)``, if set, runs as each receive
    starts. Adverts, gradients and acks carry wall-clock stamps, as a real
    device's do.
    """

    #: The longest any receive waits, so a broken test cannot hang the suite.
    WAIT_CAP_S = 10.0

    def __init__(self, known, script=None) -> None:
        self.known = list(known)
        self.script: Dict[DeviceID, dict] = dict(script or {})
        self.calls: List[tuple] = []
        self.ready = deque()
        self.grads: Dict[DeviceID, deque] = defaultdict(deque)
        self.acks: Dict[DeviceID, deque] = defaultdict(deque)
        self.answered: Dict[DeviceID, int] = {}
        self.lock = threading.RLock()
        self.arrived = threading.Condition(self.lock)
        self.solicit_raises = False
        self.after_push = None          # optional hook(device_id, msg)
        self.on_recv = None             # optional hook(kind, device_id, timeout)
        self.adv_timeouts: List[float] = []

    def _log(self, *entry) -> None:
        with self.lock:
            self.calls.append(entry)

    def _take(self, q: deque, timeout, what: str):
        """The oldest queued item, waiting up to ``timeout`` for one."""
        with self.arrived:
            if not q and (timeout is None or timeout > 0):
                cap = self.WAIT_CAP_S if timeout is None else min(timeout, self.WAIT_CAP_S)
                self.arrived.wait_for(lambda: bool(q), timeout=cap)
            if not q:
                raise RFLinkError(f"{what}: nothing arrived within {timeout}s")
            return q.popleft()

    def _answer(self, msg: FLOpenSolicit, ids) -> List[DeviceID]:
        """Every registered device in ``ids`` answers per its script."""
        reached = [d for d in ids if d in self.known]
        with self.arrived:
            for did in reached:
                s = self.script.get(did, {})
                if s.get("adv", True) is False:
                    continue
                sid = msg.solicit_id + 100 if s.get("adv") == "wrong_id" else msg.solicit_id
                self.answered[did] = sid
                self.ready.append(FLReadyAdv(
                    device_id=did, state=s.get("state", FLState.FL_OPEN),
                    performance_score=0.5, diversity_adjusted=0.3,
                    utility=s.get("utility", 0.7), issued_at=time.time(),
                    local_loss=0.25, num_examples=9, in_reply_to=sid,
                ))
            self.arrived.notify_all()
        return reached

    # ---- mule side ------------------------------------------------------ #
    def broadcast_open_solicit(self, msg) -> None:
        self._log("broadcast", msg.solicit_id)
        raise AssertionError("the ferry path never broadcasts")

    def solicit(self, msg: FLOpenSolicit, device_ids) -> List[DeviceID]:
        ids = list(device_ids)
        self._log("solicit", msg.solicit_id, tuple(ids), msg.issued_at, msg.pass_kind)
        if self.solicit_raises:
            raise RFLinkError("link closed")
        return self._answer(msg, ids)

    def recv_ready_adv(self, timeout=None) -> FLReadyAdv:
        with self.lock:
            self.adv_timeouts.append(timeout)
        if self.on_recv is not None:
            self.on_recv("adv", None, timeout)
        return self._take(self.ready, timeout, "recv_ready_adv")

    def push_disc(self, device_id, msg: DiscPush) -> None:
        s = self.script.get(device_id, {})
        self._log("push", device_id, msg.pass_kind, msg.uplink_drop, msg.mission_round)
        if s.get("push_gate") is not None:
            s["push_gate"].wait(10.0)
        if s.get("push_delay"):
            time.sleep(s["push_delay"])
        if s.get("push_fails"):
            raise RFLinkError(f"push to {device_id} failed")
        if not msg.uplink_drop and s.get("reply", True) is not False:
            sid = self.answered.get(device_id, 0)
            with self.arrived:
                if msg.pass_kind is MissionPass.DELIVER:
                    if s.get("stale_first"):
                        self.acks[device_id].append(DeliveryAck(
                            device_id=device_id, mule_id=MULE,
                            mission_round=msg.mission_round,
                            weights_sig="stale-signature", received_at=time.time(),
                            in_reply_to=sid))
                    self.acks[device_id].append(DeliveryAck(
                        device_id=device_id, mule_id=MULE, mission_round=msg.mission_round,
                        weights_sig=msg.weights_sig, received_at=time.time(),
                        in_reply_to=sid))
                else:
                    if s.get("stale_first"):
                        self.grads[device_id].append(make_grad(
                            device_id, msg.mission_round, in_reply_to=sid - 1, v=9.0))
                    self.grads[device_id].append(make_grad(
                        device_id, msg.mission_round, in_reply_to=sid,
                        corrupt=s.get("corrupt", False), age_s=s.get("age_s", 0.0)))
                self.arrived.notify_all()
        if self.after_push is not None:
            self.after_push(device_id, msg)

    def recv_gradient(self, device_id, timeout=None) -> GradientSubmission:
        s = self.script.get(device_id, {})
        if self.on_recv is not None:
            self.on_recv("gradient", device_id, timeout)
        # A gated receive ignores its timeout (a misbehaving link), but a
        # zero-timeout poll, like the host's drain, never blocks.
        if s.get("recv_gate") is not None and (timeout is None or timeout > 0):
            s["recv_gate"].wait(10.0)
        with self.lock:
            q = self.grads[device_id]
        return self._take(q, timeout, "recv_gradient")

    def recv_delivery_ack(self, device_id, timeout=None) -> DeliveryAck:
        if self.on_recv is not None:
            self.on_recv("ack", device_id, timeout)
        with self.lock:
            q = self.acks[device_id]
        return self._take(q, timeout, "recv_delivery_ack")

    # ---- device side: unused ------------------------------------------- #
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

    def close(self) -> None:
        pass

    def pushes(self):
        return [c for c in self.calls if c[0] == "push"]

    def solicits(self):
        return [c for c in self.calls if c[0] == "solicit"]


class DualRF(FerryRF):
    """``FerryRF`` that also serves the legacy broadcast: every registered
    device answers it from the same script, echoing its number (0), so one
    script can run through both paths."""

    def broadcast_open_solicit(self, msg) -> None:
        self._log("broadcast", msg.solicit_id)
        self._answer(msg, self.known)


def dwell_fn(nbytes, snr):
    """Toy airtime: 1 Mb/s per dB above the floor (+1); None below it."""
    if snr < FLOOR:
        return None
    return 8.0 * nbytes / (1e6 * (snr - FLOOR + 1.0))


def make_host(rf, clock, *, ttl=0.2, bus=None, **kw) -> HFLHostMission:
    return HFLHostMission(mule_id=MULE, rf=rf,
                          scheduler_bus=(bus.append if bus is not None else None),
                          session_ttl_s=ttl, now_fn=clock, **kw)


def banded_plan(clock, members, *, snr=None, positions=None, drop=(), payload=None,
                listen=1.0, snr_fn=None):
    snr = snr or {}
    positions = positions or {d: (1.0, 0.0, 0.0) for d in members}
    fn = snr_fn or (lambda j, d, t: snr.get(j, 10.0))
    return ContactPlan.at_arrival(
        members, clock=clock, advance=clock.advance, band="wide", band_index=0,
        stop=(0.0, 0.0, 0.0), positions=positions, range_planar_m=60.0,
        snr_fn=fn, snr_floor_db=FLOOR, dwell_fn=dwell_fn, drop_uplink=drop,
        listen_s=listen, payload_bytes=payload,
    )


def lines_by_id(report):
    return {l.device_id: l for l in report.lines}


def push_bytes(pass_kind=MissionPass.COLLECT, v=0.5) -> int:
    return push_byte_count(DiscPush(mule_id=MULE, mission_round=1, theta_disc=theta(v),
                                    synth_batch=SYNTH, pass_kind=pass_kind))


def _join_new_threads(before) -> None:
    """Wait for the workers a contact started (every thread not in ``before``).

    A worker that waits out its TTL can still be running when the contact
    returns on a loaded host. Legacy workers write when they finish, even
    after their join gave up, so waiting keeps a legacy ledger whole; a ferry
    worker's late writes are dropped, but its busy flag is released only then.
    """
    for t in threading.enumerate():
        if t not in before:
            t.join(10.0)


# --------------------------------------------------------------------------- #
# The commit: order, stamps, one charge, the invariant
# --------------------------------------------------------------------------- #

def test_commit_stamps_arrival_plus_cumulative_dwell_in_device_order():
    rf = FerryRF(D[:3])
    clock = RecClock()
    bus: list = []
    host = make_host(rf, clock, bus=bus)
    host.open_round(theta(), theta_version=5)
    snr = {D[0]: 10.0, D[1]: 3.0, D[2]: 20.0}
    plan = banded_plan(clock, D[:3], snr=snr)
    t0 = clock()
    out = host.run_contact(D[:3], SYNTH, plan=plan)

    assert list(out) == D[:3]                       # the map is in device order
    assert set(out.values()) == {MissionOutcome.CLEAN}
    # Each session: push + update bytes (measured), at its SNR.
    session = push_bytes() + make_grad(D[0], 1, in_reply_to=1).byte_count
    t = t0
    expected = {}
    for did in D[:3]:
        t = t + dwell_fn(session, snr[did])
        expected[did] = t
    rep = host._report
    assert [l.device_id for l in rep.lines] == D[:3]
    assert {l.device_id: l.contact_ts for l in rep.lines} == expected
    assert [d.contact_ts for d in bus] == [expected[d] for d in D[:3]]
    assert [r.contact_ts for r in host._contacts.records] == [expected[d] for d in D[:3]]
    # One dwell charge, no listen (nothing missing); the invariant.
    assert clock.charges == [(expected[D[2]] - t0, "dwell")]
    assert clock() == max(l.contact_ts for l in rep.lines) == expected[D[2]]
    # Band and SNR ride the lines and the contact records.
    assert all(l.band == 0 for l in rep.lines)
    assert {l.device_id: l.snr_db for l in rep.lines} == snr
    assert {r.device_id: r.snr_at_contact for r in host._contacts.records} == snr
    # Accepted in device order; the advert's loss and example count on the delta.
    assert [g.device_id for g in host._accepted] == D[:3]
    assert {(d.local_loss, d.num_examples) for d in bus} == {(0.25, 9)}
    commit = host.last_contact
    assert commit.arrival_ts == t0 and commit.end_ts == clock()
    assert commit.dwell_s == expected[D[2]] - t0 and commit.listen_s == 0.0
    assert commit.solicited == commit.answered == tuple(D[:3])


def test_device_order_holds_when_workers_finish_in_reverse():
    rf = FerryRF(D[:3], {D[0]: {"push_delay": 0.15}, D[1]: {"push_delay": 0.05}})
    clock = RecClock()
    host = make_host(rf, clock, ttl=0.5)
    host.open_round(theta(), theta_version=5)
    host.run_contact(D[:3], SYNTH, plan=banded_plan(clock, D[:3]))
    assert [l.device_id for l in host._report.lines] == D[:3]
    assert [g.device_id for g in host._accepted] == D[:3]
    stamps = [l.contact_ts for l in host._report.lines]
    assert stamps == sorted(stamps)


def test_each_session_is_priced_at_its_own_start_snr_c2():
    """A long dwell: the second target's SNR is read where its session starts."""
    rf = FerryRF(D[:2])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    t_arr = clock()
    base = {D[0]: 12.0, D[1]: 8.0}

    def snr_fn(j, d, t):          # every link loses 0.5 dB per simulated second
        return base[j] - 0.5 * (t - t_arr)

    plan = banded_plan(clock, D[:2], snr_fn=snr_fn, payload=2_000_000)
    host.run_contact(D[:2], SYNTH, plan=plan)
    d0 = dwell_fn(4_000_000, 12.0)                      # A starts at arrival
    t1 = t_arr + d0
    d1 = dwell_fn(4_000_000, snr_fn(D[1], 0.0, t1))     # B starts after A's dwell
    lines = lines_by_id(host._report)
    assert d0 > 1.0                                     # long enough to matter
    assert lines[D[0]].snr_db == 12.0
    assert lines[D[1]].snr_db == snr_fn(D[1], 0.0, t1) != base[D[1]]
    assert lines[D[1]].contact_ts == t1 + d1
    assert host.last_contact.session_dwell_s == {D[0]: d0, D[1]: d1}
    assert clock() == t1 + d1


def test_every_member_is_read_at_its_own_distance_and_session_start_c2():
    """Critic C2 on a channel that depends on distance AND time, for every kind
    of member: the SNR is read at the member's own distance, at arrival if it
    was never solicited or never answered, otherwise where its session starts
    (arrival plus the dwells of the targets before it, in device order), and
    each dwell is priced at that SNR."""
    script = {D[1]: {"reply": False}, D[2]: {"state": FLState.UNAVAILABLE},
              D[3]: {"push_fails": True}, D[5]: {"adv": False}}
    members = D[:7]
    rf = FerryRF(members, script)
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    t_arr = clock()
    positions = {D[0]: (5.0, 0.0, 0.0), D[1]: (0.0, 40.0, 0.0), D[2]: (-9.0, 12.0, 0.0),
                 D[3]: (30.0, 0.0, 0.0), D[4]: (0.0, -55.0, 0.0), D[5]: (6.0, 8.0, 0.0),
                 D[6]: (70.0, 0.0, 0.0)}                   # D6: out of range

    def snr_fn(j, d, t):          # 0.4 dB lost per metre and 0.3 dB per simulated second
        return 30.0 - 0.4 * d - 0.3 * (t - t_arr)

    plan = banded_plan(clock, members, snr_fn=snr_fn, positions=positions,
                       payload=2_000_000)
    out = host.run_contact(members, SYNTH, plan=plan)
    assert plan.unreachable == (D[6],)
    assert out == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.TIMEOUT,
                   D[2]: MissionOutcome.PARTIAL, D[3]: MissionOutcome.TIMEOUT,
                   D[4]: MissionOutcome.CLEAN, D[5]: MissionOutcome.TIMEOUT,
                   D[6]: MissionOutcome.TIMEOUT}
    dist = {j: planar_distance_m((0.0, 0.0, 0.0), positions[j]) for j in members}
    assert len(set(dist.values())) == len(members)       # nobody shares a distance

    s0 = snr_fn(D[0], dist[D[0]], t_arr)                  # CLEAN, first in line
    w0 = dwell_fn(4_000_000, s0)
    t1 = t_arr + w0
    s1 = snr_fn(D[1], dist[D[1]], t1)                     # its push, then no reply
    w1 = dwell_fn(2_000_000, s1)
    t2 = t1 + w1
    s2 = snr_fn(D[2], dist[D[2]], t2)                     # refused: no airtime
    s3 = snr_fn(D[3], dist[D[3]], t2)                     # push failed: no airtime
    s4 = snr_fn(D[4], dist[D[4]], t2)                     # CLEAN, after D0 and D1
    w4 = dwell_fn(4_000_000, s4)
    t3 = t2 + w4
    s5 = snr_fn(D[5], dist[D[5]], t_arr)                  # silent: arrival reading
    s6 = snr_fn(D[6], dist[D[6]], t_arr)                  # unreachable: arrival reading
    t_end = t3 + 1.0                                      # D1 and D5 are missing

    want_snr = {D[0]: s0, D[1]: s1, D[2]: s2, D[3]: s3, D[4]: s4, D[5]: s5, D[6]: s6}
    lines = lines_by_id(host._report)
    assert {j: lines[j].snr_db for j in members} == want_snr
    assert dict(host.last_contact.snr_db) == want_snr
    assert host.last_contact.session_dwell_s == {D[0]: w0, D[1]: w1, D[4]: w4}
    assert {j: lines[j].contact_ts for j in members} == {
        D[0]: t1, D[1]: t_end, D[2]: t2, D[3]: t2, D[4]: t3, D[5]: t_end, D[6]: t_arr}
    records = {r.device_id: r.snr_at_contact for r in host._contacts.records}
    assert records == {D[0]: s0, D[1]: s1, D[2]: s2, D[4]: s4}
    assert clock() == t_end


def test_a_session_whose_snr_falls_below_the_floor_is_priced_at_the_floor():
    rf = FerryRF(D[:2])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    t_arr = clock()

    def snr_fn(j, d, t):          # D1 passes the gate, then drops below the floor
        return 10.0 if j == D[0] else (0.0 if t == t_arr else -30.0)

    plan = banded_plan(clock, D[:2], snr_fn=snr_fn, payload=1_000_000)
    assert plan.targets == (D[0], D[1])
    host.run_contact(D[:2], SYNTH, plan=plan)
    d1 = host.last_contact.session_dwell_s[D[1]]
    assert d1 == dwell_fn(2_000_000, FLOOR)             # finite: the floor rate
    assert lines_by_id(host._report)[D[1]].snr_db == -30.0


def test_declared_payload_prices_both_directions_in_pass_1_and_one_in_pass_2():
    rf = FerryRF(D[:1])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1], payload=1_000_000))
    assert host.last_contact.session_dwell_s[D[0]] == dwell_fn(2_000_000, 10.0)
    host.close_round()
    host.open_pass_2(theta(2.0), theta_version=6)
    host.deliver_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1], payload=1_000_000))
    assert host.last_contact.session_dwell_s[D[0]] == dwell_fn(1_000_000, 10.0)


def test_several_contacts_each_charge_once_and_keep_the_invariant():
    rf = FerryRF(D[:4], {D[3]: {"adv": False}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    for members in ([D[0], D[1]], [D[2]], [D[3]]):
        n_before = len(clock.charges)
        plan = banded_plan(clock, members)
        host.run_contact(members, SYNTH, plan=plan)
        charges = clock.charges[n_before:]
        assert [k for _dt, k in charges] == (["dwell", "listen"] if D[3] in members
                                             else ["dwell"])
        stamps = [host.last_contact.contact_ts[d] for d in members]
        assert clock() == max(stamps) == host.last_contact.end_ts
    ledger = clock.clock.ledger()
    assert ledger["listen"] == 1.0
    assert ledger["dwell"] > 0.0


def test_float_rounding_in_the_charge_is_absorbed_not_refused():
    """Far past the epoch, sessions longer than the clock's own reading can make
    the charged clock land an ulp off the summed stamps (this case does). That
    is rounding, not a clock the charge missed: the stamps that ended the
    contact move onto the clock, so clock() == max(contact_ts) stays exact."""
    rf = FerryRF(D[:3])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    clock.advance(52462045.13, "transit")
    t_arr = clock()
    dwell = {1.0: 5069.768686, 2.0: 71951359.660011, 3.0: 13.542289}
    snr = {D[0]: 1.0, D[1]: 2.0, D[2]: 3.0}
    plan = ContactPlan.at_arrival(
        D[:3], clock=clock, advance=clock.advance, band="wide", band_index=0,
        stop=(0.0, 0.0, 0.0), positions={d: (1.0, 0.0, 0.0) for d in D[:3]},
        range_planar_m=60.0, snr_fn=lambda j, d, t: snr[j], snr_floor_db=FLOOR,
        dwell_fn=lambda n, s: dwell[s])
    host.run_contact(D[:3], SYNTH, plan=plan)
    t1 = t_arr + dwell[1.0]
    t2 = t1 + dwell[2.0]
    summed = t2 + dwell[3.0]
    assert clock() != summed and abs(clock() - summed) <= math.ulp(summed)
    assert {l.device_id: l.contact_ts for l in host._report.lines} == {
        D[0]: t1, D[1]: t2, D[2]: clock()}
    assert clock() == host.last_contact.end_ts == max(host.last_contact.contact_ts.values())


# --------------------------------------------------------------------------- #
# Range gating
# --------------------------------------------------------------------------- #

def test_unreachable_members_are_not_solicited_and_cost_nothing():
    rf = FerryRF(D[:3])
    clock = RecClock()
    bus: list = []
    host = make_host(rf, clock, bus=bus)
    host.open_round(theta(), theta_version=5)
    positions = {D[0]: (10.0, 0.0, 0.0), D[1]: (61.0, 0.0, 0.0), D[2]: (5.0, 0.0, 0.0)}
    snr = {D[0]: 10.0, D[1]: 10.0, D[2]: -9.0}          # D1 out of range, D2 below floor
    plan = banded_plan(clock, D[:3], snr=snr, positions=positions)
    assert plan.targets == (D[0],) and plan.unreachable == (D[1], D[2])
    t_arr = clock()
    out = host.run_contact(D[:3], SYNTH, plan=plan)

    assert [c[2] for c in rf.solicits()] == [(D[0],)]   # only the target is solicited
    assert {c[1] for c in rf.pushes()} == {D[0]}
    assert out == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.TIMEOUT,
                   D[2]: MissionOutcome.TIMEOUT}
    lines = lines_by_id(host._report)
    for did in (D[1], D[2]):
        assert lines[did].contact_ts == t_arr           # stamped at arrival
        assert lines[did].bytes_sent == lines[did].bytes_received == 0
        assert lines[did].snr_db == snr[did] and lines[did].band == 0
    deltas = {d.device_id: d for d in bus}
    assert deltas[D[1]].answered is False and deltas[D[2]].answered is False
    assert deltas[D[0]].answered is True
    # No dwell for them and no listen: they were never expected to answer.
    assert set(host.last_contact.session_dwell_s) == {D[0]}
    assert [k for _dt, k in clock.charges] == ["dwell"]
    assert host.last_contact.unreachable == (D[1], D[2])
    assert {r.device_id for r in host._contacts.records} == {D[0]}


def test_a_contact_with_no_target_makes_no_rf_call_and_charges_zero():
    rf = FerryRF(D[:2])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    plan = banded_plan(clock, D[:2], snr={D[0]: -20.0, D[1]: -20.0})
    t_arr = clock()
    out = host.run_contact(D[:2], SYNTH, plan=plan)
    assert rf.calls == []
    assert set(out.values()) == {MissionOutcome.TIMEOUT}
    assert clock.charges == [(0.0, "dwell")] and clock() == t_arr
    assert all(l.contact_ts == t_arr for l in host._report.lines)


# --------------------------------------------------------------------------- #
# Missing replies and the listen window (critic C7)
# --------------------------------------------------------------------------- #

def test_silent_targets_are_stamped_after_one_listen_window():
    rf = FerryRF(D[:3], {D[1]: {"adv": False}, D[2]: {"adv": False}})
    clock = RecClock()
    bus: list = []
    host = make_host(rf, clock, bus=bus)
    host.open_round(theta(), theta_version=5)
    plan = banded_plan(clock, D[:3], listen=1.5)
    t_arr = clock()
    out = host.run_contact(D[:3], SYNTH, plan=plan)
    assert out[D[1]] is out[D[2]] is MissionOutcome.TIMEOUT
    t_dwell = host.last_contact.contact_ts[D[0]]
    lines = lines_by_id(host._report)
    assert lines[D[1]].contact_ts == lines[D[2]].contact_ts == t_dwell + 1.5
    assert [k for _dt, k in clock.charges] == ["dwell", "listen"]
    assert clock.charges[1][0] == 1.5                    # listen charged once
    assert clock() == t_dwell + 1.5 == max(l.contact_ts for l in host._report.lines)
    assert {d.device_id: d.answered for d in bus} == {D[0]: True, D[1]: False,
                                                      D[2]: False}
    assert host.last_contact.missing == (D[1], D[2])
    assert t_dwell > t_arr


def test_a_timeout_after_the_push_costs_its_airtime_and_the_listen():
    rf = FerryRF(D[:2], {D[1]: {"reply": False}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    plan = banded_plan(clock, D[:2], payload=500_000)
    t_arr = clock()
    out = host.run_contact(D[:2], SYNTH, plan=plan)
    assert out[D[1]] is MissionOutcome.TIMEOUT
    d0 = dwell_fn(1_000_000, 10.0)
    d1 = dwell_fn(500_000, 10.0)                       # the push only
    line = lines_by_id(host._report)[D[1]]
    assert line.contact_ts == t_arr + d0 + d1 + 1.0
    assert line.bytes_sent == push_bytes() and line.bytes_received == 0
    assert [r.in_session for r in host._contacts.records if r.device_id == D[1]] == [True]


def test_uplink_drop_is_not_waited_for_is_answered_and_charges_the_listen():
    rf = FerryRF(D[:2])
    clock = RecClock()
    bus: list = []
    host = make_host(rf, clock, ttl=5.0, bus=bus)         # a TTL wait would show
    host.open_round(theta(), theta_version=5)
    plan = banded_plan(clock, D[:2], drop=[D[1]])
    wall = time.monotonic()
    out = host.run_contact(D[:2], SYNTH, plan=plan)
    assert time.monotonic() - wall < 2.0                 # no 5 s TTL wait
    drops = {c[1]: c[3] for c in rf.pushes()}
    assert drops == {D[0]: False, D[1]: True}            # the push is marked
    assert out[D[1]] is MissionOutcome.TIMEOUT
    line = lines_by_id(host._report)[D[1]]
    delta = {d.device_id: d for d in bus}[D[1]]
    assert delta.answered is True                         # its advert did arrive
    assert line.bytes_sent == push_bytes() and line.bytes_received == 0
    assert [k for _dt, k in clock.charges] == ["dwell", "listen"]
    assert line.contact_ts == clock() == host.last_contact.end_ts
    assert D[1] in host.last_contact.missing
    assert host.last_contact.uplink_dropped == (D[1],)
    assert D[1] not in [g.device_id for g in host._accepted]
    assert rf.grads[D[1]] == deque()                      # nothing was awaited


def test_refused_and_push_failed_sessions_cost_nothing_and_need_no_listen():
    rf = FerryRF(D[:3], {D[0]: {"state": FLState.UNAVAILABLE}, D[1]: {"push_fails": True}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    t_arr = clock()
    out = host.run_contact(D[:3], SYNTH, plan=banded_plan(clock, D[:3]))
    assert out == {D[0]: MissionOutcome.PARTIAL, D[1]: MissionOutcome.TIMEOUT,
                   D[2]: MissionOutcome.CLEAN}
    lines = lines_by_id(host._report)
    assert lines[D[0]].contact_ts == lines[D[1]].contact_ts == t_arr
    assert lines[D[0]].bytes_sent == lines[D[1]].bytes_sent == 0
    records = {r.device_id: r.in_session for r in host._contacts.records}
    assert records == {D[0]: False, D[2]: True}          # push failure: no record
    assert [k for _dt, k in clock.charges] == ["dwell"]  # nothing missing
    assert set(host.last_contact.session_dwell_s) == {D[2]}


def test_min_utility_refuses_like_legacy():
    rf = FerryRF(D[:1], {D[0]: {"utility": 0.1}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]), min_utility=0.5)
    assert out == {D[0]: MissionOutcome.PARTIAL}
    assert rf.pushes() == []


def test_a_corrupt_update_is_partial_as_in_legacy():
    rf = FerryRF(D[:1], {D[0]: {"corrupt": True}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert out == {D[0]: MissionOutcome.PARTIAL}
    assert host._accepted == []
    line = host._report.lines[0]
    assert line.bytes_received > 0 and line.contact_ts == clock()


def test_a_failed_targeted_solicit_leaves_every_target_silent_but_still_commits():
    rf = FerryRF(D[:2])
    rf.solicit_raises = True
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    t_arr = clock()
    out = host.run_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:2]))
    assert set(out.values()) == {MissionOutcome.TIMEOUT}
    assert all(l.contact_ts == t_arr + 1.0 for l in host._report.lines)
    assert clock() == t_arr + 1.0
    assert host.last_contact.solicited == ()


def test_an_unregistered_target_is_not_reached_and_not_waited_for():
    """The fake's receives block like a real link's, so waiting for a target the
    solicit never reached would take the whole 5 s TTL."""
    rf = FerryRF([D[0]])                                  # D1 never registered
    clock = RecClock()
    host = make_host(rf, clock, ttl=5.0)
    host.open_round(theta(), theta_version=5)
    wall = time.monotonic()
    out = host.run_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:2]))
    assert time.monotonic() - wall < 2.0
    assert out == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.TIMEOUT}
    assert host.last_contact.solicited == (D[0],)
    assert host.last_contact.missing == (D[1],)
    # One gather receive, which took D0's advert; none was left waiting for D1.
    assert len([t for t in rf.adv_timeouts if t > 0]) == 1


def test_the_gather_waits_one_ttl_for_a_reached_target_that_stays_silent():
    ttl = 0.3
    rf = FerryRF(D[:2], {D[1]: {"adv": False}})           # reached, never answers
    clock = RecClock()
    host = make_host(rf, clock, ttl=ttl)
    host.open_round(theta(), theta_version=5)
    wall = time.monotonic()
    out = host.run_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:2]))
    elapsed = time.monotonic() - wall
    assert out == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.TIMEOUT}
    assert host.last_contact.solicited == tuple(D[:2])
    assert host.last_contact.missing == (D[1],)
    assert elapsed >= ttl - 0.02                          # it did wait for D1
    # ...and never asked the link to wait past one TTL (the drain polls with 0).
    waits = [t for t in rf.adv_timeouts if t > 0]
    assert waits and max(waits) <= ttl


def test_adverts_already_queued_when_the_ttl_runs_out_still_count():
    """Starting each worker inside the gather can use up a short TTL on a loaded
    host (a stall here). Adverts that had arrived by then are still taken, by
    polling; nothing is waited for past the deadline."""
    ttl = 0.2
    rf = FerryRF(D[:3])                                   # all three answer at once
    clock = RecClock()
    host = make_host(rf, clock, ttl=ttl)
    host.open_round(theta(), theta_version=5)
    stalled: list = []

    def on_recv(kind, did, timeout):
        if kind == "adv" and timeout > 0 and not stalled:
            stalled.append(timeout)
            time.sleep(1.5 * ttl)                         # the gather's first step overruns

    rf.on_recv = on_recv
    out = host.run_contact(D[:3], SYNTH, plan=banded_plan(clock, D[:3]))
    assert out == {d: MissionOutcome.CLEAN for d in D[:3]}
    assert host.last_contact.answered == tuple(D[:3]) and host.last_contact.missing == ()
    # One wait, then polls: the adverts behind the stalled one were taken
    # after the deadline without waiting (the drain before the solicit polls too).
    assert [t > 0 for t in rf.adv_timeouts] == [False, True, False, False]


class _FloodRF(FerryRF):
    """A link whose advert queue never runs dry: when nothing real is queued, a
    stale advert from a device outside the contact is always waiting."""

    def recv_ready_adv(self, timeout=None) -> FLReadyAdv:
        with self.lock:
            self.adv_timeouts.append(timeout)
            if self.ready:
                return self.ready.popleft()
        return FLReadyAdv(device_id=DeviceID("elsewhere"), state=FLState.FL_OPEN,
                          performance_score=0.1, diversity_adjusted=0.1, utility=0.5)


def test_a_link_that_never_runs_dry_cannot_hold_the_gather():
    """Past the deadline the gather only polls, and at most ``_DRAIN_LIMIT``
    times, so stale adverts cannot keep the contact open."""
    from hermes.mission.host_mission import _DRAIN_LIMIT

    rf = _FloodRF(D[:2], {D[1]: {"adv": False}})
    clock = RecClock()
    host = make_host(rf, clock, ttl=0.1)
    host.open_round(theta(), theta_version=5)
    result: dict = {}
    worker = threading.Thread(daemon=True, target=lambda: result.setdefault(
        "out", host.run_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:2]))))
    worker.start()
    worker.join(30.0)
    assert not worker.is_alive(), "the gather never stopped"
    assert result["out"] == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.TIMEOUT}
    first_wait = next(i for i, t in enumerate(rf.adv_timeouts) if t > 0)
    assert rf.adv_timeouts[:first_wait] == [0.0] * _DRAIN_LIMIT      # the drain's cap
    late = [t for t in rf.adv_timeouts[first_wait:] if t == 0.0]
    assert len(late) == _DRAIN_LIMIT                                  # the gather's cap


class _StaleGradientFloodRF(FerryRF):
    """A gradient queue that never runs dry: an update answering another
    solicit is always waiting, and the device's own never comes."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.grad_timeouts: List[float] = []

    def recv_gradient(self, device_id, timeout=None) -> GradientSubmission:
        with self.lock:
            self.grad_timeouts.append(timeout)
        return make_grad(device_id, 1, in_reply_to=999)


def test_stale_replies_that_never_run_dry_cannot_hold_a_worker():
    """Past its deadline the reply wait only polls, at most ``_DRAIN_LIMIT``
    times, so the worker gives up and the session is a missing reply."""
    from hermes.mission.host_mission import _DRAIN_LIMIT

    rf = _StaleGradientFloodRF(D[:1], {D[0]: {"reply": False}})
    clock = RecClock()
    host = make_host(rf, clock, ttl=0.1)
    host.open_round(theta(), theta_version=5)
    result: dict = {}
    before = set(threading.enumerate())
    runner = threading.Thread(daemon=True, target=lambda: result.setdefault(
        "out", host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))))
    runner.start()
    runner.join(30.0)
    assert not runner.is_alive(), "the contact never ended"
    _join_new_threads(before)
    assert result["out"] == {D[0]: MissionOutcome.TIMEOUT}
    assert host.last_contact.missing == (D[0],)
    first_wait = next(i for i, t in enumerate(rf.grad_timeouts) if t > 0)
    assert rf.grad_timeouts[:first_wait] == [0.0] * _DRAIN_LIMIT     # the drain's cap
    late = [t for t in rf.grad_timeouts[first_wait:] if t == 0.0]
    assert len(late) == _DRAIN_LIMIT                                 # the reply wait's cap


# --------------------------------------------------------------------------- #
# Stale messages (critics B1, B2)
# --------------------------------------------------------------------------- #

def test_an_advert_answering_another_solicit_is_discarded_not_stashed():
    rf = FerryRF(D[:2], {D[1]: {"adv": "wrong_id"}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    out = host.run_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:2]))
    assert out == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.TIMEOUT}
    assert {c[1] for c in rf.pushes()} == {D[0]}          # never pushed to
    assert host._misrouted_advs == []
    assert host.last_contact.stale_discarded["adv"] == 1
    assert host.last_contact.answered == (D[0],)


def test_stale_adverts_and_the_legacy_stash_are_flushed_before_the_solicit():
    rf = FerryRF(D[:2])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    old = FLReadyAdv(device_id=D[0], state=FLState.FL_OPEN, performance_score=0.1,
                     diversity_adjusted=0.1, utility=0.9, issued_at=1.0, in_reply_to=0)
    rf.ready.append(old)                                  # left over on the link
    host._misrouted_advs.append(old)                      # and in the stash
    out = host.run_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:2]))
    assert set(out.values()) == {MissionOutcome.CLEAN}
    assert host._misrouted_advs == []
    assert host.last_contact.stale_discarded["adv"] == 1


def test_numbered_solicits_carry_the_arrival_time_and_increase():
    rf = FerryRF(D[:2])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    seen = []
    for did in D[:2]:
        plan = banded_plan(clock, [did])
        host.run_contact([did], SYNTH, plan=plan)
        _kind, sid, _ids, issued_at, pass_kind = rf.solicits()[-1]
        seen.append((sid, issued_at, plan.arrival_ts, pass_kind))
    assert [s[0] for s in seen] == [1, 2]
    assert all(issued == arrival for _sid, issued, arrival, _p in seen)
    assert {p for *_x, p in seen} == {MissionPass.COLLECT}
    assert host.last_contact.solicit_id == 2


def test_a_stale_gradient_queued_before_the_contact_is_drained_not_taken_as_partial():
    rf = FerryRF(D[:1])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    # An update and an ack of an earlier round, left after their contacts gave up.
    rf.grads[D[0]].append(make_grad(D[0], 0, in_reply_to=0, v=3.0))
    rf.acks[D[0]].append(DeliveryAck(device_id=D[0], mule_id=MULE, mission_round=0,
                                     weights_sig="x", received_at=1.0))
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert out == {D[0]: MissionOutcome.CLEAN}
    assert host.last_contact.stale_discarded == {"adv": 0, "gradient": 1, "ack": 1}
    assert [g.in_reply_to for g in host._accepted] == [1]
    assert rf.grads[D[0]] == deque() and rf.acks[D[0]] == deque()


def test_the_drain_empties_a_targets_queues_even_when_no_reply_is_awaited():
    """A refused target is never waited on, so only the drain before the solicit
    can clear its stale update and ack: nothing is left for its next contact."""
    rf = FerryRF(D[:1], {D[0]: {"state": FLState.UNAVAILABLE}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    rf.grads[D[0]].append(make_grad(D[0], 0, in_reply_to=0, v=3.0))
    rf.acks[D[0]].append(DeliveryAck(device_id=D[0], mule_id=MULE, mission_round=0,
                                     weights_sig="x", received_at=1.0))
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert out == {D[0]: MissionOutcome.PARTIAL}
    assert rf.grads[D[0]] == deque() and rf.acks[D[0]] == deque()
    assert host.last_contact.stale_discarded == {"adv": 0, "gradient": 1, "ack": 1}


def test_stale_adverts_are_drained_even_when_no_target_is_reached():
    rf = FerryRF([])                                      # nobody registered
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    rf.ready.append(FLReadyAdv(device_id=D[0], state=FLState.FL_OPEN,
                               performance_score=0.1, diversity_adjusted=0.1,
                               utility=0.9, issued_at=1.0, in_reply_to=0))
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert out == {D[0]: MissionOutcome.TIMEOUT}
    assert rf.ready == deque()
    assert host.last_contact.stale_discarded["adv"] == 1


def test_a_gradient_for_another_mule_is_not_taken():
    rf = FerryRF(D[:1], {D[0]: {"reply": False}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)

    def other_mule(device_id, msg):
        g = make_grad(device_id, msg.mission_round, in_reply_to=rf.answered[device_id])
        rf.grads[device_id].append(GradientSubmission(
            device_id=device_id, mule_id=MuleID("other"), mission_round=g.mission_round,
            delta_theta=g.delta_theta, num_examples=g.num_examples,
            submitted_at=g.submitted_at, in_reply_to=g.in_reply_to))

    rf.after_push = other_mule
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert out == {D[0]: MissionOutcome.TIMEOUT} and host._accepted == []


def test_a_gradient_naming_another_device_is_not_taken():
    """TCP routes by the socket's registered id, whatever the message says."""
    rf = FerryRF(D[:1], {D[0]: {"reply": False}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    rf.after_push = lambda did, msg: rf.grads[did].append(
        make_grad(D[4], msg.mission_round, in_reply_to=rf.answered[did]))
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert out == {D[0]: MissionOutcome.TIMEOUT} and host._accepted == []
    assert host.last_contact.stale_discarded["gradient"] == 1


def test_a_stale_gradient_arriving_during_the_wait_is_discarded():
    rf = FerryRF(D[:1], {D[0]: {"stale_first": True}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert out == {D[0]: MissionOutcome.CLEAN}
    assert host.last_contact.stale_discarded["gradient"] == 1
    np.testing.assert_array_equal(host._accepted[0].delta_theta[0], theta(1.0)[0])


def test_a_matching_reply_queued_behind_a_stale_one_counts_past_the_ttl():
    """The worker's wait can overrun the TTL (a stall here) while it discards a
    stale update; the matching one already queued behind it is still read, by
    polling, within the contact's join deadline."""
    ttl = 0.5
    rf = FerryRF(D[:1], {D[0]: {"stale_first": True}})
    clock = RecClock()
    host = make_host(rf, clock, ttl=ttl)
    host.open_round(theta(), theta_version=5)
    stalled: list = []

    def on_recv(kind, did, timeout):
        if kind == "gradient" and timeout > 0 and not stalled:
            stalled.append(timeout)
            time.sleep(1.2 * ttl)                         # past the TTL, inside 2 x TTL

    rf.on_recv = on_recv
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert stalled and out == {D[0]: MissionOutcome.CLEAN}
    assert host.last_contact.stale_discarded["gradient"] == 1
    assert [g.in_reply_to for g in host._accepted] == [host.last_contact.solicit_id]


def test_a_wrong_round_gradient_never_becomes_partial():
    rf = FerryRF(D[:1], {D[0]: {"reply": False}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)

    def late_old_round(device_id, msg):   # arrives after the drain, answering this solicit
        rf.grads[device_id].append(make_grad(device_id, msg.mission_round + 7,
                                             in_reply_to=rf.answered[device_id]))

    rf.after_push = late_old_round
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert out == {D[0]: MissionOutcome.TIMEOUT}         # not PARTIAL
    assert host._accepted == []
    assert host.last_contact.stale_discarded["gradient"] == 1


def test_a_stale_ack_is_never_counted_as_delivered():
    rf = FerryRF(D[:2], {D[0]: {"reply": False}, D[1]: {"stale_first": True}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    host.open_pass_2(theta(2.0), theta_version=6)
    # D0: only a stale ack (another round and signature) is waiting.
    rf.acks[D[0]].append(DeliveryAck(device_id=D[0], mule_id=MULE, mission_round=0,
                                     weights_sig="old", received_at=1.0))
    out = host.deliver_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:2]))
    assert out == {D[0]: DeliveryOutcome.UNDELIVERED, D[1]: DeliveryOutcome.DELIVERED}
    # D0's was drained before the solicit; D1's (a wrong signature, ahead of
    # its fresh ack) was discarded while waiting.
    assert host.last_contact.stale_discarded["ack"] == 2
    assert host.last_contact.missing == (D[0],)


@pytest.mark.parametrize("which", ["round", "solicit"])
def test_an_ack_for_the_same_theta_from_another_round_or_solicit_is_not_delivered(which):
    """When no round closes the cluster sends the same θ' again, so a stale ack can
    carry the right signature: its round or its solicit number gives it away."""
    rf = FerryRF(D[:1], {D[0]: {"reply": False}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    host.open_pass_2(theta(2.0), theta_version=6)

    def same_theta_old_ack(device_id, msg):
        sid = rf.answered[device_id]
        rf.acks[device_id].append(DeliveryAck(
            device_id=device_id, mule_id=MULE,
            mission_round=msg.mission_round - (1 if which == "round" else 0),
            weights_sig=msg.weights_sig, received_at=time.time(),
            in_reply_to=sid if which == "round" else sid - 1))

    rf.after_push = same_theta_old_ack
    out = host.deliver_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert out == {D[0]: DeliveryOutcome.UNDELIVERED}
    assert host.last_contact.stale_discarded["ack"] == 1


def test_an_ack_addressed_to_another_mule_is_not_delivered():
    """Right device, round, signature and solicit number, but another mule's."""
    rf = FerryRF(D[:1], {D[0]: {"reply": False}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    host.open_pass_2(theta(2.0), theta_version=6)

    def other_mules_ack(device_id, msg):
        rf.acks[device_id].append(DeliveryAck(
            device_id=device_id, mule_id=MuleID("other"), mission_round=msg.mission_round,
            weights_sig=msg.weights_sig, received_at=time.time(),
            in_reply_to=rf.answered[device_id]))

    rf.after_push = other_mules_ack
    out = host.deliver_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert out == {D[0]: DeliveryOutcome.UNDELIVERED}
    assert host.last_contact.stale_discarded["ack"] == 1
    assert host.last_contact.missing == (D[0],)


# --------------------------------------------------------------------------- #
# The ferry ledger is the legacy ledger, restamped
# --------------------------------------------------------------------------- #

#: One script for both paths: every Pass-1 kind of session the ledger records.
MIXED_PASS_1 = {
    D[1]: {"state": FLState.UNAVAILABLE},    # refused by S2B: PARTIAL, no push
    D[2]: {"push_fails": True},              # TIMEOUT, answered, nothing sent
    D[3]: {"reply": False},                  # TIMEOUT after its push
    D[4]: {"adv": False},                    # silent: TIMEOUT, not answered
    D[5]: {"corrupt": True},                 # a bad receipt: PARTIAL, bytes received
    D[6]: {"utility": 0.1},                  # under min_utility: PARTIAL, no push
    D[7]: {"age_s": 60.0},                   # past the receipt TTL: PARTIAL
}                                            # D0 and D8: CLEAN
MIXED_PASS_2 = {D[2]: {"push_fails": True}, D[3]: {"reply": False}, D[4]: {"adv": False}}


def _run_pass_1(*, ferry: bool):
    rf = DualRF(D, MIXED_PASS_1)
    bus: list = []
    clock = RecClock() if ferry else None
    host = HFLHostMission(mule_id=MULE, rf=rf, scheduler_bus=bus.append,
                          session_ttl_s=0.2, now_fn=clock)
    host.open_round(theta(), theta_version=5)
    plan = banded_plan(clock, D, snr={d: 10.0 for d in D}) if ferry else None
    before = set(threading.enumerate())
    out = host.run_contact(D, SYNTH, min_utility=0.5, plan=plan)
    _join_new_threads(before)
    accepted = sorted(g.device_id for g in host._accepted)
    agg, report, contacts = host.close_round()
    return out, report, contacts, bus, accepted, agg, rf


def _run_pass_2(*, ferry: bool):
    rf = DualRF(D, MIXED_PASS_2)
    clock = RecClock() if ferry else None
    host = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=0.2, now_fn=clock)
    host.open_round(theta(), theta_version=5)
    host.open_pass_2(theta(2.0), theta_version=6)
    plan = banded_plan(clock, D) if ferry else None
    before = set(threading.enumerate())
    out = host.deliver_contact(D, SYNTH, plan=plan)
    _join_new_threads(before)
    return out, host.close_pass_2(), rf


def _masked(obj, *fields):
    """A ledger entry as a dict, without the fields the two paths may differ in."""
    out = dataclasses.asdict(obj)
    for name in fields:
        out.pop(name)
    return out


def test_the_ferry_ledger_is_the_legacy_ledger_but_for_stamps_band_and_snr():
    """The commit writes every line, delta and contact record itself; each must
    carry the legacy values for the same session, field for field. Only the
    stamps (simulated), the band and the SNR (new fields) may differ."""
    leg_out, leg_rep, leg_con, leg_bus, leg_acc, leg_agg, leg_rf = _run_pass_1(ferry=False)
    fer_out, fer_rep, fer_con, fer_bus, fer_acc, fer_agg, fer_rf = _run_pass_1(ferry=True)
    assert [c[0] for c in leg_rf.calls[:1]] == ["broadcast"]      # the two paths ran
    assert [c[0] for c in fer_rf.calls[:1]] == ["solicit"]

    assert fer_out == leg_out == {
        D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.PARTIAL,
        D[2]: MissionOutcome.TIMEOUT, D[3]: MissionOutcome.TIMEOUT,
        D[4]: MissionOutcome.TIMEOUT, D[5]: MissionOutcome.PARTIAL,
        D[6]: MissionOutcome.PARTIAL, D[7]: MissionOutcome.PARTIAL,
        D[8]: MissionOutcome.CLEAN}
    for rep, bus in ((leg_rep, leg_bus), (fer_rep, fer_bus)):
        assert sorted(l.device_id for l in rep.lines) == sorted(D)   # one line each
        assert sorted(d.device_id for d in bus) == sorted(D)         # one delta each

    def lines(rep):
        return {l.device_id: _masked(l, "contact_ts", "band", "snr_db") for l in rep.lines}

    def deltas(bus):
        return {d.device_id: _masked(d, "contact_ts") for d in bus}

    def records(con):
        return {r.device_id: _masked(r, "contact_ts", "snr_at_contact")
                for r in con.records}

    assert lines(fer_rep) == lines(leg_rep)
    assert deltas(fer_bus) == deltas(leg_bus)
    assert records(fer_con) == records(leg_con)
    # A record per device heard from, except D2 (its push failed); none for D4.
    assert len(fer_con.records) == len(leg_con.records) == 7
    # The fields that are the point of the comparison, spelled out once.
    clean = lines(fer_rep)[D[0]]
    assert (clean["basis_version"], clean["age"], clean["num_examples"]) == (5, 0, 7)
    assert clean["bytes_sent"] == push_bytes() and clean["bytes_received"] > 0
    fer_deltas = deltas(fer_bus)
    assert fer_deltas[D[0]]["utility"] == 0.7
    assert fer_deltas[D[2]]["answered"] is True and fer_deltas[D[4]]["answered"] is False
    # Only the new fields tell the paths apart.
    assert {(l.band, l.snr_db) for l in leg_rep.lines} == {(None, None)}
    assert {(l.band, l.snr_db) for l in fer_rep.lines} == {(0, 10.0)}
    # The same updates merged into the same partial.
    assert fer_acc == leg_acc == [D[0], D[8]]
    assert sorted(fer_agg.contributing_devices) == sorted(leg_agg.contributing_devices)
    assert fer_agg.num_examples == leg_agg.num_examples
    for a, b in zip(fer_agg.weights, leg_agg.weights):
        np.testing.assert_array_equal(a, b)


def test_the_ferry_delivery_ledger_is_the_legacy_one_but_for_stamps_band_and_bytes():
    leg_out, leg_rep, leg_rf = _run_pass_2(ferry=False)
    fer_out, fer_rep, fer_rf = _run_pass_2(ferry=True)
    assert [c[0] for c in leg_rf.calls[:1]] == ["broadcast"]
    assert [c[0] for c in fer_rf.calls[:1]] == ["solicit"]
    undelivered = {D[2], D[3], D[4]}
    assert fer_out == leg_out == {
        d: (DeliveryOutcome.UNDELIVERED if d in undelivered else DeliveryOutcome.DELIVERED)
        for d in D}
    assert sorted(l.device_id for l in fer_rep.lines) == sorted(D)
    assert ({l.device_id: _masked(l, "contact_ts", "band", "bytes_sent") for l in fer_rep.lines}
            == {l.device_id: _masked(l, "contact_ts", "band", "bytes_sent")
                for l in leg_rep.lines})
    assert {(l.band, l.bytes_sent) for l in leg_rep.lines} == {(None, 0)}


# --------------------------------------------------------------------------- #
# Pass 2
# --------------------------------------------------------------------------- #

def test_pass_2_ferry_delivery_lines_carry_band_bytes_and_sim_stamps():
    rf = FerryRF(D[:3], {D[1]: {"reply": False}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    host.open_pass_2(theta(2.0), theta_version=6)
    positions = {D[0]: (1.0, 0.0, 0.0), D[1]: (2.0, 0.0, 0.0), D[2]: (70.0, 0.0, 0.0)}
    plan = banded_plan(clock, D[:3], positions=positions)
    t_arr = clock()
    out = host.deliver_contact(D[:3], SYNTH, plan=plan)
    assert out == {D[0]: DeliveryOutcome.DELIVERED, D[1]: DeliveryOutcome.UNDELIVERED,
                   D[2]: DeliveryOutcome.UNDELIVERED}
    lines = {l.device_id: l for l in host._delivery_report.lines}
    push_b = push_bytes(MissionPass.DELIVER, 2.0)
    assert lines[D[0]].bytes_sent == lines[D[1]].bytes_sent == push_b
    assert lines[D[2]].bytes_sent == 0 and lines[D[2]].contact_ts == t_arr
    assert all(l.band == 0 for l in lines.values())
    d = dwell_fn(push_b, 10.0)
    assert lines[D[0]].contact_ts == t_arr + d
    assert lines[D[1]].contact_ts == t_arr + d + d + 1.0 == clock()
    assert [c[2] for c in rf.solicits()] == [(D[0], D[1])]
    assert {c[4] for c in rf.solicits()} == {MissionPass.DELIVER}


def test_pass_2_refuses_an_uplink_drop():
    rf = FerryRF(D[:1])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    host.open_pass_2(theta(2.0), theta_version=6)
    with pytest.raises(ValueError, match="Pass-1 availability draw"):
        host.deliver_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1], drop=[D[0]]))
    assert rf.calls == [] and clock.charges == []


# --------------------------------------------------------------------------- #
# The clock without a band (critic A1) and no wall stamps (critic B3)
# --------------------------------------------------------------------------- #

def test_clock_without_a_band_charges_one_session_per_contact():
    rf = FerryRF(D[:3], {D[2]: {"adv": False}})
    clock = RecClock()
    bus: list = []
    host = make_host(rf, clock, bus=bus)
    host.open_round(theta(), theta_version=5)
    t_arr = clock()
    plan = ContactPlan.at_arrival(D[:2], clock=clock, advance=clock.advance)
    host.run_contact(D[:2], SYNTH, plan=plan)
    assert clock.charges == [(1.0, "dwell")]             # 1 s for the whole contact
    assert [l.contact_ts for l in host._report.lines] == [t_arr + 1.0] * 2
    assert all(l.band is None and l.snr_db is None for l in host._report.lines)
    assert [r.snr_at_contact for r in host._contacts.records] == [0.0, 0.0]
    assert host.last_contact.session_dwell_s == {}
    # A second contact with a silent member: 1 s, then the listen window.
    plan = ContactPlan.at_arrival([D[2]], clock=clock, advance=clock.advance)
    host.run_contact([D[2]], SYNTH, plan=plan)
    assert clock.charges[1:] == [(1.0, "dwell"), (1.0, "listen")]
    assert host._report.lines[-1].contact_ts == t_arr + 3.0 == clock()
    assert [d.contact_ts for d in bus] == [t_arr + 1.0, t_arr + 1.0, t_arr + 3.0]


def test_without_a_band_every_session_heard_from_ends_with_the_one_charge():
    """Critic A1: the contact's one ``session_time_s`` is the session of every
    target heard from, a refusal and a failed push included; a reply that
    never came ends after the listen window."""
    rf = FerryRF(D[:4], {D[1]: {"state": FLState.UNAVAILABLE}, D[2]: {"push_fails": True},
                         D[3]: {"reply": False}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    t_arr = clock()
    out = host.run_contact(D[:4], SYNTH, plan=ContactPlan.at_arrival(
        D[:4], clock=clock, advance=clock.advance))
    assert out == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.PARTIAL,
                   D[2]: MissionOutcome.TIMEOUT, D[3]: MissionOutcome.TIMEOUT}
    assert {l.device_id: l.contact_ts for l in host._report.lines} == {
        D[0]: t_arr + 1.0, D[1]: t_arr + 1.0, D[2]: t_arr + 1.0, D[3]: t_arr + 2.0}
    assert clock.charges == [(1.0, "dwell"), (1.0, "listen")]
    assert clock() == t_arr + 2.0


def test_no_wall_stamp_reaches_a_line_delta_record_or_report_b3():
    """Adverts, gradients and acks carry wall stamps (~1.7e9 s); none may leak."""
    rf = FerryRF(D[:4], {D[1]: {"state": FLState.UNAVAILABLE}, D[2]: {"reply": False},
                         D[3]: {"adv": False}})
    clock = RecClock()
    bus: list = []
    host = make_host(rf, clock, bus=bus)
    host.open_round(theta(), theta_version=5)
    host.run_contact(D[:4], SYNTH, plan=banded_plan(clock, D[:4]))
    host.run_contact(D[:1], SYNTH, plan=ContactPlan.at_arrival(
        D[:1], clock=clock, advance=clock.advance))
    _agg, report, contacts = host.close_round()
    host.open_pass_2(theta(2.0), theta_version=6)
    host.deliver_contact(D[:4], SYNTH, plan=banded_plan(clock, D[:4]))
    host.record_skipped_delivery([D[5]])
    delivery = host.close_pass_2()
    stamps = ([l.contact_ts for l in report.lines] + [d.contact_ts for d in bus]
              + [r.contact_ts for r in contacts.records]
              + [l.contact_ts for l in delivery.lines]
              + [report.started_at, report.finished_at,
                 delivery.started_at, delivery.finished_at])
    assert len(stamps) > 20
    assert all(mission_clock.SIM_EPOCH_S <= s < SIM_CEILING for s in stamps), stamps
    assert max(stamps) == clock()


def test_the_mission_clock_stamps_the_reports_and_skipped_deliveries():
    rf = FerryRF(D[:1])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    assert host._report.started_at == mission_clock.SIM_EPOCH_S
    clock.advance(4.0, "transit")
    host.open_pass_2(theta(), theta_version=6)
    host.record_skipped_delivery([D[0]])
    clock.advance(2.0, "transit")
    rep = host.close_pass_2()
    assert rep.started_at == rep.lines[0].contact_ts == mission_clock.SIM_EPOCH_S + 4.0
    assert rep.lines[0].outcome is DeliveryOutcome.SKIPPED
    assert rep.finished_at == mission_clock.SIM_EPOCH_S + 6.0


# --------------------------------------------------------------------------- #
# Late workers (P-01 defect 3 closed) and one join deadline (defect 2)
# --------------------------------------------------------------------------- #

def _join_worker(name: str) -> None:
    for t in threading.enumerate():
        if t.name == name:
            t.join(10.0)


def test_a_worker_past_the_join_deadline_is_committed_and_its_late_writes_dropped(caplog):
    gate = threading.Event()
    rf = FerryRF(D[:2], {D[1]: {"push_gate": gate}})
    clock = RecClock()
    bus: list = []
    host = make_host(rf, clock, ttl=0.05, bus=bus)
    host.open_round(theta(), theta_version=5)
    with caplog.at_level(logging.WARNING, logger="hermes.mission.host_mission"):
        out = host.run_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:2]))
        snapshot = (list(host._report.lines), list(host._accepted), dict(out), clock(),
                    list(bus), list(host._contacts.records))
        gate.set()                                        # the push goes out now
        _join_worker(f"ferry-collect-{D[1]}")
    # Committed as a push the mule gave up on: TIMEOUT, no bytes, no record,
    # stamped when D0's session ended; not a missing reply, so no listen.
    assert out == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.TIMEOUT}
    line = lines_by_id(host._report)[D[1]]
    assert line.bytes_sent == 0 and line.contact_ts == host.last_contact.contact_ts[D[0]]
    assert host.last_contact.missing == () and host.last_contact.listen_s == 0.0
    assert [k for _dt, k in clock.charges] == ["dwell"]
    assert {d.device_id: d.answered for d in bus}[D[1]] is True
    # Nothing it did afterwards reached the ledgers, the accepted list, the
    # returned map, the bus or the clock.
    assert (list(host._report.lines), list(host._accepted), dict(out), clock(),
            list(bus), list(host._contacts.records)) == snapshot
    assert [g.device_id for g in host._accepted] == [D[0]]
    late = [r.getMessage() for r in caplog.records if "late" in r.getMessage()]
    assert len(late) == 2 and all("dropped" in m for m in late)   # its push, its session


def test_a_pushed_worker_past_the_join_deadline_is_a_missing_reply():
    gate = threading.Event()
    rf = FerryRF(D[:1], {D[0]: {"recv_gate": gate}})     # a link ignoring its timeout
    clock = RecClock()
    host = make_host(rf, clock, ttl=0.05)
    host.open_round(theta(), theta_version=5)
    t_arr = clock()
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1], payload=100_000))
    gate.set()
    _join_worker(f"ferry-collect-{D[0]}")
    assert out == {D[0]: MissionOutcome.TIMEOUT}
    line = host._report.lines[0]
    assert line.bytes_sent == push_bytes()                # the push had gone out
    assert line.contact_ts == t_arr + dwell_fn(100_000, 10.0) + 1.0 == clock()
    assert host._accepted == []


def test_one_join_deadline_bounds_the_whole_contact():
    """Two workers stuck in their pushes: the contact returns after one 2 x TTL,
    not one per worker (P-01 defect 2 closed on the ferry path)."""
    g0, g1 = threading.Event(), threading.Event()
    rf = FerryRF(D[:2], {D[0]: {"push_gate": g0}, D[1]: {"push_gate": g1}})
    clock = RecClock()
    ttl = 0.5
    host = make_host(rf, clock, ttl=ttl)
    host.open_round(theta(), theta_version=5)
    wall = time.monotonic()
    host.run_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:2]))
    elapsed = time.monotonic() - wall
    g0.set()
    g1.set()
    # One deadline: about 2 x TTL. Joined in turn (legacy) it would be 4 x TTL.
    assert 2 * ttl - 0.05 <= elapsed < 3.5 * ttl


# --------------------------------------------------------------------------- #
# Wiring refusals
# --------------------------------------------------------------------------- #

def test_a_mission_clock_host_refuses_a_contact_without_a_plan():
    rf = FerryRF(D[:1])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    with pytest.raises(ValueError, match="needs a ContactPlan"):
        host.run_contact(D[:1], SYNTH)
    host.open_pass_2(theta(), theta_version=6)
    with pytest.raises(ValueError, match="needs a ContactPlan"):
        host.deliver_contact(D[:1], SYNTH)
    assert rf.calls == []


def test_a_plan_needs_a_host_on_the_mission_clock():
    rf = FerryRF(D[:1])
    clock = RecClock()
    host = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=0.2)
    host.open_round(theta(), theta_version=5)
    with pytest.raises(ValueError, match="now_fn"):
        host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))


def test_the_plan_must_describe_this_contact_at_this_time():
    rf = FerryRF(D[:3])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    with pytest.raises(ValueError, match="members"):
        host.run_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:3]))
    with pytest.raises(ValueError, match="repeat"):
        host.run_contact([D[0], D[0]], SYNTH, plan=banded_plan(clock, D[:1]))
    stale_plan = banded_plan(clock, D[:1])
    clock.advance(1.0, "transit")
    with pytest.raises(ValueError, match="build the plan at arrival"):
        host.run_contact(D[:1], SYNTH, plan=stale_plan)
    with pytest.raises(TypeError):
        host.run_contact(D[:1], SYNTH, plan=object())
    assert rf.calls == []


def test_a_clock_charged_by_someone_else_during_the_contact_is_refused_loudly():
    """Only the commit charges the clock; anything else is a wiring bug, raised as
    a RuntimeError (the supervisor swallows MissionSessionError) before any charge
    or write."""
    rf = FerryRF(D[:1])
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    rf.after_push = lambda _did, _msg: clock.clock.advance(3.0, "transit")
    with pytest.raises(RuntimeError, match="moved during the contact") as info:
        host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert not isinstance(info.value, MissionSessionError)
    assert clock.charges == [] and host._report.lines == [] and host._accepted == []


def test_a_plan_on_another_clock_than_the_hosts_is_refused_when_charged():
    """Two fresh clocks both read the epoch, so the arrival check cannot tell
    them apart; once the commit has charged the plan's clock the host's must
    read the same, or its reports would end before its own sessions."""
    host_clock, plan_clock = RecClock(), RecClock()
    rf = FerryRF(D[:2])
    bus: list = []
    host = make_host(rf, host_clock, bus=bus)
    host.open_round(theta(), theta_version=5)
    plan = banded_plan(plan_clock, D[:2], payload=1_000_000)
    with pytest.raises(RuntimeError, match="host's clock") as info:
        host.run_contact(D[:2], SYNTH, plan=plan)
    assert not isinstance(info.value, MissionSessionError)
    assert host._report.lines == [] and host._accepted == [] and bus == []
    assert host._contacts.records == [] and host.last_contact is None
    assert host_clock.charges == [] and [k for _dt, k in plan_clock.charges] == ["dwell"]


def test_a_plan_whose_advance_charges_another_clock_is_refused():
    clock, other = RecClock(), RecClock()
    rf = FerryRF(D[:2], {D[1]: {"adv": False}})
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    plan = ContactPlan.at_arrival(
        D[:2], clock=clock, advance=other.advance, band="wide", band_index=0,
        stop=(0.0, 0.0, 0.0), positions={d: (1.0, 0.0, 0.0) for d in D[:2]},
        range_planar_m=60.0, snr_fn=lambda j, d, t: 10.0, snr_floor_db=FLOOR,
        dwell_fn=dwell_fn)
    with pytest.raises(RuntimeError, match="plan.advance must charge plan.clock"):
        host.run_contact(D[:2], SYNTH, plan=plan)
    assert host._report.lines == [] and host._accepted == []
    assert clock.charges == [] and [k for _dt, k in other.charges] == ["dwell", "listen"]


def test_one_clock_behind_two_wrappers_is_accepted():
    """The check compares readings, not objects: a host and a plan may reach the
    same clock through different callables."""
    clock = RecClock()
    rf = FerryRF(D[:1])
    host = make_host(rf, lambda: clock())
    host.open_round(theta(), theta_version=5)
    out = host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    assert out == {D[0]: MissionOutcome.CLEAN}
    assert host._report.lines[0].contact_ts == clock() > mission_clock.SIM_EPOCH_S


def test_legacy_guards_still_come_first():
    rf = FerryRF(D[:1])
    clock = RecClock()
    host = make_host(rf, clock)
    with pytest.raises(MissionSessionError, match="no mission round is open"):
        host.run_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))
    host.open_round(theta(), theta_version=5)
    with pytest.raises(ValueError, match="requires at least one device"):
        host.run_contact([], SYNTH, plan=banded_plan(clock, D[:1]))
    with pytest.raises(MissionSessionError, match="deliver_contact called in pass=collect"):
        host.deliver_contact(D[:1], SYNTH, plan=banded_plan(clock, D[:1]))


# --------------------------------------------------------------------------- #
# What stays on the wall clock in ferry mode (design section 2.4)
# --------------------------------------------------------------------------- #

def test_the_receipt_ttl_stays_on_the_wall_clock():
    """The receipt check compares the mule's wall time with the device's
    ``submitted_at``; read from the mission clock (1e6 s against 1.7e9 s) it
    could never fire."""
    rf = FerryRF(D[:2], {D[1]: {"age_s": 10.0}})         # 10 s > 2 x TTL (0.4 s)
    clock = RecClock()
    bus: list = []
    host = make_host(rf, clock, bus=bus)
    host.open_round(theta(), theta_version=5)
    out = host.run_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:2]))
    assert out == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.PARTIAL}
    assert [g.device_id for g in host._accepted] == [D[0]]
    line = lines_by_id(host._report)[D[1]]
    assert line.bytes_received > 0                       # the update did arrive
    # Its session still took airtime and a simulated stamp, like any reply.
    assert D[1] in host.last_contact.session_dwell_s
    assert line.contact_ts == clock() < SIM_CEILING


def test_busy_flags_stay_on_the_wall_clock_while_a_worker_waits():
    """``is_busy`` compares wall time with the flag, so the flag must be claimed
    on the wall clock too: on the mission clock it would never be live."""
    rf = FerryRF(D[:2], {D[1]: {"reply": False}})
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    seen = []

    def on_recv(kind, did, timeout):
        if kind == "gradient":
            seen.append((did, timeout > 0, host.is_busy(did)))

    rf.on_recv = on_recv
    before = set(threading.enumerate())
    host.run_contact(D[:2], SYNTH, plan=banded_plan(clock, D[:2]))
    _join_new_threads(before)                            # D1 gives up after one TTL
    waiting = [(did, busy) for did, blocking, busy in seen if blocking]
    assert sorted(did for did, _busy in waiting) == D[:2]
    assert all(busy for _did, busy in waiting)           # live while it waits
    drained = [busy for _did, blocking, busy in seen if not blocking]
    assert drained and not any(drained)                  # not claimed before the push
    assert not host.is_busy(D[0]) and not host.is_busy(D[1])   # released after


# --------------------------------------------------------------------------- #
# Message defaults (critic A3: additive, default-valued fields)
# --------------------------------------------------------------------------- #

def test_new_message_and_line_fields_default_to_the_legacy_values():
    assert FLOpenSolicit(mule_id=MULE, mission_round=1, issued_at=0.0).solicit_id == 0
    adv = FLReadyAdv(device_id=D[0], state=FLState.FL_OPEN, performance_score=0.0,
                     diversity_adjusted=0.0, utility=0.0)
    assert adv.in_reply_to == 0
    push = DiscPush(mule_id=MULE, mission_round=1, theta_disc=theta(), synth_batch=[])
    assert push.uplink_drop is False
    assert GradientSubmission(device_id=D[0], mule_id=MULE, mission_round=1,
                              delta_theta=theta(), num_examples=1,
                              submitted_at=0.0).in_reply_to == 0
    assert DeliveryAck(device_id=D[0], mule_id=MULE, mission_round=1, weights_sig="",
                       received_at=0.0).in_reply_to == 0
    line = MissionRoundCloseLine(device_id=D[0], outcome=MissionOutcome.CLEAN,
                                 contact_ts=0.0)
    assert line.band is None and line.snr_db is None
    dline = MissionDeliveryLine(device_id=D[0], outcome=DeliveryOutcome.DELIVERED,
                                contact_ts=0.0)
    assert dline.band is None and dline.bytes_sent == 0


def test_the_new_fields_cross_the_wire():
    from hermes.transport.wire import recv_message, send_message

    msgs = [
        FLOpenSolicit(mule_id=MULE, mission_round=3, issued_at=1e6, solicit_id=7),
        FLReadyAdv(device_id=D[0], state=FLState.FL_OPEN, performance_score=0.0,
                   diversity_adjusted=0.0, utility=0.0, in_reply_to=7),
        DiscPush(mule_id=MULE, mission_round=3, theta_disc=theta(), synth_batch=[],
                 uplink_drop=True),
        make_grad(D[0], 3, in_reply_to=7),
        DeliveryAck(device_id=D[0], mule_id=MULE, mission_round=3, weights_sig="s",
                    received_at=0.0, in_reply_to=7),
    ]
    a, b = socket.socketpair()
    try:
        for msg in msgs:
            send_message(a, msg)
            back = recv_message(b, timeout=5.0)
            for attr in ("solicit_id", "in_reply_to", "uplink_drop"):
                if hasattr(msg, attr):
                    assert getattr(back, attr) == getattr(msg, attr)
    finally:
        a.close()
        b.close()
