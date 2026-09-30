"""FeRRy Phase 3, unit U5: ferry contacts end to end with real devices.

Real ``ClientMission`` devices serve in their own threads over the loopback
link and over real TCP sockets (``TCPRFLinkServer``/``TCPRFLinkClient``, U0's
targeted solicit and newest-solicit option), while the mule's
``HFLHostMission`` runs its ferry path on U1's ``MissionClock`` with contact
plans built from U3's ``ContactLink`` and U2's ``ContactChannel``, as the
supervisor will build them (unit U6).

What is pinned: range gating keeps a far device from ever being solicited;
the uplink drop is honoured by the device; each member's SNR, airtime and
stamp are the channel's and the link's for its own distance at its own session
start (critic C2); every stamp in every ledger, delta and record is simulated
time (critic B3); the clock ends each contact on the latest stamp; a late
update from one contact is drained before the next and never becomes its
PARTIAL (critic B2); the gather waits for reached targets only; the round
merges.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List

import numpy as np
import pytest

from hermes.mission import ClientMission, HFLHostMission, LocalTrainResult
from hermes.mission.contact_plan import ContactPlan, planar_distance_m
from hermes.transport import LoopbackRFLink, TCPRFLinkClient, TCPRFLinkServer
from hermes.types import (
    DeliveryAck,
    DeliveryOutcome,
    DeviceID,
    FLState,
    MissionOutcome,
    MuleID,
)

mission_clock = pytest.importorskip("hermes.l1.mission_clock")
contact_link = pytest.importorskip("hermes.l1.contact_link")
channel_model = pytest.importorskip("hermes.l1.channel_model")

MULE = MuleID("mule-ferry")
SIM_CEILING = 1.0e9
IDS = [DeviceID(f"dev-{i:02d}") for i in range(4)]
# dev-02 sits 75 m from the stop: outside R_planar(wide) = 60 m.
POSITIONS = {IDS[0]: (10.0, 5.0, 0.0), IDS[1]: (-20.0, 0.0, 0.0),
             IDS[2]: (75.0, 0.0, 0.0), IDS[3]: (0.0, 15.0, 0.0)}
STOP = (0.0, 0.0, 0.0)


def _trainer(i: int, *, slow_first_s: float = 0.0):
    calls = {"n": 0}

    def _train(theta, synth):
        calls["n"] += 1
        if calls["n"] == 1 and slow_first_s:
            time.sleep(slow_first_s)
        after = [np.asarray(w, dtype=np.float32) + np.float32(0.01 * (i + 1)) for w in theta]
        return LocalTrainResult(delta_theta=after, num_examples=4 + i, accuracy=0.7,
                                auc=0.7, loss=0.3 - 0.01 * i, theta_after=after)
    return _train


class Devices:
    """Device threads calling ``serve_once`` until stopped."""

    def __init__(self, clients: List[ClientMission]) -> None:
        self.clients = clients
        self._stop = threading.Event()
        self.threads = [threading.Thread(target=self._loop, args=(cm,), daemon=True)
                        for cm in clients]

    def _loop(self, cm: ClientMission) -> None:
        while not self._stop.is_set():
            cm.serve_once()

    def __enter__(self):
        for t in self.threads:
            t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        for t in self.threads:
            t.join(5.0)
        return False


def _theta(v=0.0):
    return [np.full((4,), v, dtype=np.float32), np.full((3, 3), 0.01, dtype=np.float32)]


SYNTH = [np.zeros((2, 4), dtype=np.float32)]


class Physics:
    """The link models the supervisor wires into each plan (wide band)."""

    def __init__(self, seed: int = 7) -> None:
        self.link = contact_link.ContactLink(anchor_planar_m=60.0)
        self.channel = channel_model.ContactChannel.from_link(
            self.link, salt=channel_model.ferry_salt(seed, channel_model.SALT_CONTACT))

    def plan(self, clock, members, *, drop=(), payload=None) -> ContactPlan:
        return ContactPlan.at_arrival(
            members, clock=clock, advance=clock.advance, band="wide",
            band_index=self.link.index("wide"), stop=STOP,
            positions={d: POSITIONS[d] for d in members},
            range_planar_m=self.link.range_planar_m("wide"),
            snr_fn=lambda j, d, t: self.channel.snr_db(t, "wide", d, link_key=j),
            snr_floor_db=self.link.snr_floor_db,
            dwell_fn=lambda n, s: self.link.dwell_s(n, "wide", s),
            drop_uplink=drop, listen_s=1.0, payload_bytes=payload,
        )


def _priced(phys: Physics, plan: ContactPlan, *, replied: Dict[DeviceID, int],
            no_reply: Dict[DeviceID, int]):
    """What the commit must stamp and price, recomputed from U2's channel and
    U3's link: in device order from the arrival, an unreachable member at
    arrival with its arrival SNR; every other one at the channel's SNR for ITS
    distance where its session starts (critic C2), priced by the link for its
    bytes; a push that got no reply is stamped after the one listen window.
    ``replied`` / ``no_reply`` map each target to the bytes its airtime is for.
    """
    link, chan = phys.link, phys.channel
    t = plan.arrival_ts
    snr, dwell, stamps, missing = {}, {}, {}, []
    for j in IDS:
        d = planar_distance_m(STOP, POSITIONS[j])
        if j in plan.unreachable:
            snr[j] = chan.snr_db(plan.arrival_ts, "wide", d, link_key=j)
            stamps[j] = plan.arrival_ts
            continue
        s = chan.snr_db(t, "wide", d, link_key=j)
        nbytes = replied[j] if j in replied else no_reply[j]
        w = link.dwell_s(nbytes, "wide", s)
        if w is None:                       # fell below the floor since arrival
            w = link.dwell_s(nbytes, "wide", link.snr_floor_db)
        snr[j], dwell[j] = s, w
        t = t + w
        if j in no_reply:
            missing.append(j)
        else:
            stamps[j] = t
    t_end = t + 1.0 if missing else t
    stamps.update({j: t_end for j in missing})
    return snr, dwell, stamps


def _all_stamps(report, contacts, delivery, bus) -> List[float]:
    out = [l.contact_ts for l in report.lines] + [r.contact_ts for r in contacts.records]
    out += [d.contact_ts for d in bus] + [l.contact_ts for l in delivery.lines]
    out += [report.started_at, report.finished_at, delivery.started_at,
            delivery.finished_at]
    return out


def _run_mission(rf, devices: List[ClientMission], *, ttl: float = 5.0):
    """One round on the mission clock: Pass-1 contact, merge, Pass-2 contact."""
    clock = mission_clock.MissionClock()
    phys = Physics()
    bus: list = []
    host = HFLHostMission(mule_id=MULE, rf=rf, scheduler_bus=bus.append,
                          session_ttl_s=ttl, now_fn=clock)
    host.open_round(_theta(), theta_version=1)
    plan1 = phys.plan(clock, IDS, drop=[IDS[3]], payload=1_000_000)
    out1 = host.run_contact(IDS, SYNTH, plan=plan1)
    commit1 = host.last_contact
    assert clock() == max(commit1.contact_ts.values()) == commit1.end_ts
    agg, report, contacts = host.close_round()
    clock.advance(30.0, "turnaround")
    host.open_pass_2(_theta(0.5), theta_version=2)
    plan2 = phys.plan(clock, IDS)
    out2 = host.deliver_contact(IDS, SYNTH, plan=plan2)
    commit2 = host.last_contact
    assert clock() == max(commit2.contact_ts.values()) == commit2.end_ts
    delivery = host.close_pass_2()
    return dict(clock=clock, host=host, bus=bus, out1=out1, out2=out2, plan1=plan1,
                plan2=plan2, agg=agg, report=report, contacts=contacts,
                delivery=delivery, commit1=commit1, commit2=commit2, phys=phys)


def _check_mission(r, devices: Dict[DeviceID, ClientMission]) -> None:
    far, dropped = IDS[2], IDS[3]
    assert r["plan1"].unreachable == (far,)                 # 75 m: out of range
    assert r["out1"] == {IDS[0]: MissionOutcome.CLEAN, IDS[1]: MissionOutcome.CLEAN,
                         far: MissionOutcome.TIMEOUT, dropped: MissionOutcome.TIMEOUT}
    assert r["out2"] == {IDS[0]: DeliveryOutcome.DELIVERED,
                         IDS[1]: DeliveryOutcome.DELIVERED,
                         far: DeliveryOutcome.UNDELIVERED,
                         dropped: DeliveryOutcome.DELIVERED}
    # The far device was never solicited or pushed to.
    assert devices[far].last_push_round is None
    assert devices[far]._theta_basis is None
    # The uplink-dropped device adopted Pass 1's basis, then Pass 2's.
    np.testing.assert_array_equal(devices[dropped]._theta_basis[0], _theta(0.5)[0])
    lines = {l.device_id: l for l in r["report"].lines}
    deltas = {d.device_id: d for d in r["bus"]}
    assert deltas[far].answered is False and deltas[dropped].answered is True
    assert lines[dropped].bytes_sent > 0 and lines[dropped].bytes_received == 0
    assert lines[far].contact_ts == r["plan1"].arrival_ts
    assert r["commit1"].missing == (dropped,) and r["commit1"].listen_s == 1.0
    # Band and SNR on every line; the SNR is the channel's for the member's
    # own distance at its own session start, and the airtime the link's at
    # that SNR (critic C2): 1 MB each way declared in Pass 1 (the push alone
    # for the dropped uplink), the measured push in Pass 2.
    assert {l.band for l in r["report"].lines} == {0}
    assert {l.band for l in r["delivery"].lines} == {0}
    assert len({planar_distance_m(STOP, POSITIONS[j]) for j in IDS}) == len(IDS)
    snr1, dwell1, stamps1 = _priced(
        r["phys"], r["plan1"], replied={IDS[0]: 2_000_000, IDS[1]: 2_000_000},
        no_reply={dropped: 1_000_000})
    assert {l.device_id: l.snr_db for l in r["report"].lines} == snr1
    assert dict(r["commit1"].snr_db) == snr1
    assert r["commit1"].session_dwell_s == dwell1
    assert {l.device_id: l.contact_ts for l in r["report"].lines} == stamps1
    delivered = {l.device_id: l.bytes_sent for l in r["delivery"].lines
                 if l.outcome is DeliveryOutcome.DELIVERED}
    assert set(delivered) == {IDS[0], IDS[1], dropped} and min(delivered.values()) > 0
    snr2, dwell2, stamps2 = _priced(r["phys"], r["plan2"], replied=delivered, no_reply={})
    assert dict(r["commit2"].snr_db) == snr2
    assert r["commit2"].session_dwell_s == dwell2
    assert {l.device_id: l.contact_ts for l in r["delivery"].lines} == stamps2
    # The merge used the two CLEAN updates, in device order.
    assert list(r["agg"].contributing_devices) == [IDS[0], IDS[1]]
    # No wall stamp anywhere (critic B3): devices stamp adverts, updates and
    # acks with their wall clock, ~1.7e9 s.
    stamps = _all_stamps(r["report"], r["contacts"], r["delivery"], r["bus"])
    assert all(mission_clock.SIM_EPOCH_S <= s < SIM_CEILING for s in stamps), stamps
    # The ledger accounts for every simulated second.
    ledger = r["clock"].ledger()
    assert ledger["dwell"] == pytest.approx(r["commit1"].dwell_s + r["commit2"].dwell_s)
    assert ledger["listen"] == 1.0 and ledger["turnaround"] == 30.0
    assert r["clock"]() - mission_clock.SIM_EPOCH_S == pytest.approx(sum(ledger.values()))
    # 1 MB each way on wide takes a visible, simulated time (not wall time).
    assert r["commit1"].dwell_s > 0.5


def test_a_ferry_round_over_the_loopback_link():
    rf = LoopbackRFLink(newest_solicit_only=True)
    devices = {}
    for i, did in enumerate(IDS):
        rf.register_device(did)
        cm = ClientMission(device_id=did, rf=rf, local_train=_trainer(i),
                           solicit_timeout_s=0.5, disc_push_timeout_s=2.0)
        cm.set_state(FLState.FL_OPEN)
        devices[did] = cm
    with Devices(list(devices.values())):
        wall = time.monotonic()
        r = _run_mission(rf, list(devices.values()))
        elapsed = time.monotonic() - wall
    _check_mission(r, devices)
    # Nothing waited out the 5 s TTL: the uplink drop is not waited for, the
    # far device is not solicited, and no reply is missing in Pass 2.
    assert elapsed < 4.0


def test_a_ferry_round_over_real_tcp_sockets():
    server = TCPRFLinkServer(host="127.0.0.1", port=0)
    server.start()
    clients: List[TCPRFLinkClient] = []
    devices = {}
    try:
        for i, did in enumerate(IDS):
            link = TCPRFLinkClient(device_id=did, host=server.host, port=server.port,
                                   newest_solicit_only=True)
            clients.append(link)
            cm = ClientMission(device_id=did, rf=link, local_train=_trainer(i),
                               solicit_timeout_s=0.5, disc_push_timeout_s=2.0)
            cm.set_state(FLState.FL_OPEN)
            devices[did] = cm
        assert server.wait_for_devices(IDS, timeout=5.0)
        with Devices(list(devices.values())):
            r = _run_mission(server, list(devices.values()))
        _check_mission(r, devices)
    finally:
        for c in clients:
            c.close()
        server.close()


def test_a_late_update_is_drained_and_never_becomes_the_next_contacts_partial():
    """Critic B2 on the real loopback queues (``probe_stale_queue.py``'s case).

    The device's first fit outlasts the session TTL, so contact 1 gives up and
    its update lands in the mule's queue afterwards. At afa9526 the device's
    next contact would read that update as its reply (a PARTIAL, round
    mismatch) and leave the fresh one queued. On the ferry path contact 2
    drains it first and takes the fresh update: CLEAN.
    """
    rf = LoopbackRFLink(newest_solicit_only=True)
    did = IDS[0]
    rf.register_device(did)
    cm = ClientMission(device_id=did, rf=rf, local_train=_trainer(0, slow_first_s=0.8),
                       solicit_timeout_s=0.3, disc_push_timeout_s=2.0)
    cm.set_state(FLState.FL_OPEN)
    clock = mission_clock.MissionClock()
    phys = Physics()
    host = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=0.3, now_fn=clock)
    with Devices([cm]):
        host.open_round(_theta(), theta_version=1)
        out1 = host.run_contact([did], SYNTH, plan=phys.plan(clock, [did]))
        assert out1 == {did: MissionOutcome.TIMEOUT}
        assert host.last_contact.missing == (did,)
        with pytest.raises(Exception):
            host.close_round()                             # nothing was accepted
        # Let the slow fit finish: its update is now queued on the mule.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and rf._gradient_per_device[did].qsize() == 0:
            time.sleep(0.02)
        assert rf._gradient_per_device[did].qsize() == 1
        host.open_round(_theta(0.1), theta_version=2)
        clock.advance(10.0, "transit")
        out2 = host.run_contact([did], SYNTH, plan=phys.plan(clock, [did]))
    assert out2 == {did: MissionOutcome.CLEAN}
    assert host.last_contact.stale_discarded["gradient"] == 1
    assert [g.mission_round for g in host._accepted] == [2]
    assert rf._gradient_per_device[did].qsize() == 0


def test_a_stale_ack_on_the_real_queue_is_drained_before_a_delivery():
    rf = LoopbackRFLink(newest_solicit_only=True)
    did = IDS[1]
    rf.register_device(did)
    cm = ClientMission(device_id=did, rf=rf, local_train=_trainer(1),
                       solicit_timeout_s=0.3, disc_push_timeout_s=2.0)
    cm.set_state(FLState.FL_OPEN)
    clock = mission_clock.MissionClock()
    host = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=1.0, now_fn=clock)
    host.open_round(_theta(), theta_version=1)
    host.open_pass_2(_theta(0.5), theta_version=2)
    rf.send_delivery_ack(DeliveryAck(device_id=did, mule_id=MULE, mission_round=1,
                                     weights_sig="an-older-theta", received_at=time.time()))
    with Devices([cm]):
        out = host.deliver_contact([did], SYNTH, plan=Physics().plan(clock, [did]))
    assert out == {did: DeliveryOutcome.DELIVERED}
    assert host.last_contact.stale_discarded["ack"] == 1
    line = host._delivery_report.lines[0]
    assert line.contact_ts < SIM_CEILING and line.bytes_sent > 0


def test_the_gather_waits_for_reached_targets_only():
    """On the real loopback queues a receive blocks, so the gather's waits show
    in wall time: a target the solicit never reached (not registered) is not
    waited for, and a reached target that stays silent is waited for one TTL."""
    rf = LoopbackRFLink(newest_solicit_only=True)
    served, silent, unregistered = IDS[0], IDS[1], IDS[3]
    rf.register_device(served)
    rf.register_device(silent)                             # no device serves it
    cm = ClientMission(device_id=served, rf=rf, local_train=_trainer(0),
                       solicit_timeout_s=0.3, disc_push_timeout_s=2.0)
    cm.set_state(FLState.FL_OPEN)
    clock = mission_clock.MissionClock()
    phys = Physics()
    host = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=5.0, now_fn=clock)
    with Devices([cm]):
        host.open_round(_theta(), theta_version=1)
        members = [served, unregistered]
        wall = time.monotonic()
        out1 = host.run_contact(members, SYNTH, plan=phys.plan(clock, members))
        not_waited = time.monotonic() - wall
        commit1 = host.last_contact
        host.session_ttl_s = 0.4
        members = [served, silent]
        wall = time.monotonic()
        out2 = host.run_contact(members, SYNTH, plan=phys.plan(clock, members))
        waited = time.monotonic() - wall
        commit2 = host.last_contact
    assert out1 == {served: MissionOutcome.CLEAN, unregistered: MissionOutcome.TIMEOUT}
    assert commit1.solicited == (served,) and commit1.missing == (unregistered,)
    assert not_waited < 2.5                                # not the 5 s TTL
    assert out2 == {served: MissionOutcome.CLEAN, silent: MissionOutcome.TIMEOUT}
    assert commit2.solicited == (served, silent) and commit2.missing == (silent,)
    assert waited >= 0.4 - 0.02                            # one TTL for the silent one
