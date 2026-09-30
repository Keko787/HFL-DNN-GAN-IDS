"""Deterministic in-process mule worlds for the mission-clock tests (unit U6).

Built on the golden supervisor harness (``tests/golden/_mule_harness.py``): the
same real ``ClientMission`` devices with pure trainers, the same real
``HFLHostCluster`` run inside each upload with ``ClusterService``'s fold and
dispatch policy, the same synchronous worker threads. Two things differ:

* **The RF link** answers both the recorded broadcast solicit and the ferry
  path's targeted, numbered solicit (``RFLink.solicit``). A targeted device
  answers with an advert that echoes the solicit's number, and its reply to
  the push carries it too, as ``ClientMission.serve_once`` does. A device can
  be made silent (it never answers a solicit).
* **The wall clock** (``time.time`` while a world runs) starts at a realistic
  epoch, 1.7e9 s, so a wall stamp that leaks into a ledger, a delta or the
  scheduler's state is caught by ``< 1e9`` (the mission clock's ceiling). It
  moves only on the harness's own events (+2 s per solicit, +0.5 s per push,
  +3 s per upload), never on a read.

A world runs the same scenario on the wall clock (``now_fn``, the recorded
path) or on a ``MissionClock`` with a ``FerrySpec``, so the two can be
compared device by device.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.l1.mission_clock import MissionClock
from hermes.mission import ClientMission
from hermes.mission.aggregation_rules import AggregationSpec
from hermes.mule import MuleSupervisor
from hermes.transport import RFLink, RFLinkError
from hermes.types import DeviceID, FLState, MissionPass, MuleID, SpectrumSig

from tests.golden import _mule_harness as GH
from tests.golden._host_harness import SyncThread

WALL_T0 = 1_700_000_000.0
SIM_CEILING = 1.0e9
DOCK = (0.0, 0.0, 0.0)
_real_time = time.time
_real_thread = threading.Thread


class WallClock:
    """The patched ``time.time``: moved only by the harness."""

    def __init__(self, t0: float = WALL_T0) -> None:
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class Patched:
    """``time.time`` -> the wall clock, ``threading.Thread`` -> synchronous."""

    def __init__(self, clock: WallClock) -> None:
        self.clock = clock

    def __enter__(self):
        time.time = self.clock
        threading.Thread = SyncThread
        return self

    def __exit__(self, *exc):
        time.time = _real_time
        threading.Thread = _real_thread
        return False


class FerrySyncRF(RFLink):
    """One mule's RF link; its devices answer synchronously.

    ``broadcast_open_solicit`` reaches every device (the recorded path);
    ``solicit`` only the listed, registered ones, each answering with the
    solicit's number. ``silent`` devices never answer a solicit. Every
    mule-side call is logged in ``calls``.
    """

    def __init__(self, clock: WallClock, *, solicit_dt: float = GH.SOLICIT_DT,
                 push_dt: float = GH.PUSH_DT) -> None:
        self.clock = clock
        self.solicit_dt = float(solicit_dt)
        self.push_dt = float(push_dt)
        self.order: List[DeviceID] = []
        self.devices: Dict[DeviceID, ClientMission] = {}
        self.silent: Set[DeviceID] = set()
        self._ready: Deque = deque()
        self._grads: Dict[DeviceID, Deque] = defaultdict(deque)
        self._acks: Dict[DeviceID, Deque] = defaultdict(deque)
        self._listening: Set[DeviceID] = set()
        self._answered_solicit: Dict[DeviceID, int] = {}
        self.calls: List[list] = []

    def register_device(self, device_id: DeviceID) -> None:
        if device_id not in self.order:
            self.order.append(device_id)

    def attach(self, cm: ClientMission) -> None:
        self.devices[cm.device_id] = cm

    def _answer(self, did: DeviceID, solicit_id: int) -> None:
        cm = self.devices[did]
        adv = cm.build_ready_adv(in_reply_to=solicit_id)
        self._ready.append(adv)
        self._answered_solicit[did] = solicit_id
        if adv.is_eligible() and adv.utility >= cm.fl_threshold:
            self._listening.add(did)
        else:
            self._listening.discard(did)

    # ---- mule side ------------------------------------------------------- #

    def broadcast_open_solicit(self, msg) -> None:
        self.clock.advance(self.solicit_dt)
        self.calls.append(["broadcast", msg.pass_kind.value, msg.mission_round])
        for did in self.order:
            if did not in self.silent:
                self._answer(did, 0)

    def solicit(self, msg, device_ids) -> List[DeviceID]:
        self.clock.advance(self.solicit_dt)
        reached: List[DeviceID] = []
        for did in device_ids:
            if did in self.devices and did not in reached:
                reached.append(did)
        self.calls.append(["solicit", msg.pass_kind.value, msg.mission_round,
                           [str(d) for d in reached], int(msg.solicit_id)])
        for did in reached:
            if did not in self.silent:
                self._answer(did, int(msg.solicit_id))
        return reached

    def recv_ready_adv(self, timeout=None):
        if not self._ready:
            raise RFLinkError("recv_ready_adv: nothing queued")
        return self._ready.popleft()

    def push_disc(self, device_id, msg) -> None:
        self.clock.advance(self.push_dt)
        self.calls.append(["push", str(device_id), msg.pass_kind.value, msg.mission_round,
                           bool(getattr(msg, "uplink_drop", False))])
        if device_id not in self._listening:
            return
        self._listening.discard(device_id)
        cm = self.devices[device_id]
        cm.last_push_round = msg.mission_round
        cm.last_push_pass = msg.pass_kind.value
        sid = self._answered_solicit.get(device_id, 0)
        if msg.pass_kind is MissionPass.DELIVER:
            cm._handle_delivery_push(msg, in_reply_to=sid)
        else:
            cm._handle_collect_push(msg, in_reply_to=sid)

    def recv_gradient(self, device_id, timeout=None):
        q = self._grads.get(device_id)
        if not q:
            raise RFLinkError(f"recv_gradient for {device_id!r}: nothing queued")
        return q.popleft()

    def recv_delivery_ack(self, device_id, timeout=None):
        q = self._acks.get(device_id)
        if not q:
            raise RFLinkError(f"recv_delivery_ack for {device_id!r}: nothing queued")
        return q.popleft()

    # ---- device side ----------------------------------------------------- #

    def send_gradient(self, msg) -> None:
        self._grads[msg.device_id].append(msg)

    def send_delivery_ack(self, msg) -> None:
        self._acks[msg.device_id].append(msg)

    def send_ready_adv(self, msg) -> None:
        self._ready.append(msg)

    def recv_open_solicit(self, device_id, timeout=None):
        raise NotImplementedError("the harness drives devices directly")

    def recv_disc_push(self, device_id, timeout=None):
        raise NotImplementedError("the harness drives devices directly")

    def close(self) -> None:
        pass


class SimDockServer(GH.DockServer):
    """The golden dock server; optionally echoes simulated time on DOWNs.

    With ``echo_sim_ts`` every DOWN carries ``cluster_sim_ts`` = the latest
    ``sim_upload_ts`` ingested so far (what unit U7's cluster will send).
    """

    def __init__(self, cluster, clock, *, echo_sim_ts: bool = False) -> None:
        super().__init__(cluster, clock)
        self.echo_sim_ts = bool(echo_sim_ts)
        self.max_sim_ts: Optional[float] = None
        self.ups: List[Any] = []
        self.downs: List[Any] = []

    def on_up(self, up) -> None:
        self.ups.append(up)
        ts = getattr(up, "sim_upload_ts", None)
        if ts is not None:
            self.max_sim_ts = ts if self.max_sim_ts is None else max(self.max_sim_ts, ts)
        super().on_up(up)

    def send_down(self, mule_id, *, why: str) -> None:
        bundle = self.cluster.dispatch_down_bundle(mule_id)
        if self.echo_sim_ts and self.max_sim_ts is not None:
            bundle.cluster_sim_ts = self.max_sim_ts
        self.downs.append((why, bundle))
        self.log.append(["down", why, str(mule_id)])
        assert self.dock is not None
        self.dock.send_down(bundle)


class World:
    """Devices, one RF link per mule, a cluster, and supervisors."""

    def __init__(self, *, mule_ids: Sequence[str] = ("mule-g",),
                 layout=GH.LAYOUT, flaky=GH.FLAKY,
                 assignment: Optional[Dict[str, str]] = None,
                 aggregation: Optional[AggregationSpec] = None,
                 min_participation: int = 1, echo_sim_ts: bool = False,
                 push_dt: float = GH.PUSH_DT,
                 extra_devices: Optional[Dict[str, Tuple[float, float, float]]] = None) -> None:
        """``extra_devices`` are registered with the cluster but assigned to a
        mule that never flies, and listen on the first mule's link: devices
        that mule can reach but was never given in a slice (beacon offers)."""
        self.clock = WallClock()
        self.mule_ids = [MuleID(m) for m in mule_ids]
        self.aggregation = aggregation or AggregationSpec()
        self.layout = tuple(layout)
        registry = DeviceRegistry()
        for did, pos in self.layout:
            registry.register(
                device_id=DeviceID(did), position=pos,
                spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
            )
        registry.rebalance(self.mule_ids, round_counter=0)
        if assignment is not None:
            for did, mid in assignment.items():
                registry.get(DeviceID(did)).assigned_mule = MuleID(mid)
        self.cluster = HFLHostCluster(
            registry=registry,
            generator=StubGeneratorHost(disc_weights=[
                np.zeros((4,), dtype=np.float32),
                np.ones((3, 3), dtype=np.float32) * 0.01,
            ]),
            dock=None,  # type: ignore[arg-type]
            synth_batch_size=2, min_participation=min_participation,
            aggregation=self.aggregation,
        )
        self.server = SimDockServer(self.cluster, self.clock, echo_sim_ts=echo_sim_ts)
        self.dock = GH.SyncDock(self.server)
        self.cluster.dock = self.dock
        self.rfs: Dict[MuleID, FerrySyncRF] = {}
        self.devices: Dict[DeviceID, ClientMission] = {}
        for idx, (did, _pos) in enumerate(self.layout):
            mid = registry.get(DeviceID(did)).assigned_mule
            if mid not in self.rfs:
                self.rfs[mid] = FerrySyncRF(self.clock, push_dt=push_dt)
            rf = self.rfs[mid]
            rel, seed = flaky.get(did, (None, None))
            cm = ClientMission(
                device_id=DeviceID(did), rf=rf, local_train=GH.make_trainer(idx),
                solicit_timeout_s=1.0, disc_push_timeout_s=1.0,
                contact_reliability=rel, contact_rng_seed=seed,
            )
            cm.set_state(FLState.FL_OPEN)
            rf.attach(cm)
            self.devices[DeviceID(did)] = cm
        for k, (did, pos) in enumerate(sorted((extra_devices or {}).items())):
            registry.register(
                device_id=DeviceID(did), position=pos,
                spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
            )
            registry.get(DeviceID(did)).assigned_mule = MuleID("mule-never-flies")
            rf = self.rfs[self.mule_ids[0]]
            cm = ClientMission(
                device_id=DeviceID(did), rf=rf,
                local_train=GH.make_trainer(len(self.layout) + k),
                solicit_timeout_s=1.0, disc_push_timeout_s=1.0,
            )
            cm.set_state(FLState.FL_OPEN)
            rf.attach(cm)
            self.devices[DeviceID(did)] = cm
        self.sups: Dict[MuleID, MuleSupervisor] = {}
        self.deltas: Dict[MuleID, List[Tuple[str, Any]]] = defaultdict(list)

    def supervisor(self, mule_id: str, *, sim: bool, ferry=None, **kwargs) -> MuleSupervisor:
        mid = MuleID(mule_id)
        kwargs.setdefault("session_ttl_s", 3.0)
        kwargs.setdefault("rf_range_m", 60.0)
        if sim:
            kwargs["mission_clock"] = kwargs.get("mission_clock") or MissionClock()
            if ferry is not None:
                kwargs["ferry"] = ferry
        else:
            kwargs["now_fn"] = self.clock
        sup = MuleSupervisor(mule_id=mid, rf=self.rfs[mid], dock=self.dock,
                             aggregation=self.aggregation, **kwargs)
        ingest = sup.scheduler.ingest_round_close_delta
        sink = self.deltas[mid]

        def _bus(delta, _ingest=ingest, _sink=sink):
            _sink.append(("session", delta))
            return _ingest(delta)

        def _direct(delta, _ingest=ingest, _sink=sink):
            _sink.append(("direct", delta))
            return _ingest(delta)

        sup.mission.scheduler_bus = _bus
        sup.scheduler.ingest_round_close_delta = _direct
        self.sups[mid] = sup
        return sup

    def bootstrap(self) -> None:
        for mid in self.mule_ids:
            self.server.bootstrap(mid)
        for mid, sup in self.sups.items():
            if not sup.wait_for_initial_dock(timeout=2.0):
                raise AssertionError(f"{mid} did not bootstrap")

    def set_state(self, device_ids: Sequence[str], state: FLState) -> None:
        for did in device_ids:
            self.devices[DeviceID(did)].set_state(state)

    def take_deltas(self, mule_id) -> List[Tuple[str, Any]]:
        out = list(self.deltas[MuleID(mule_id)])
        self.deltas[MuleID(mule_id)].clear()
        return out


# --------------------------------------------------------------------------- #
# Reading results
# --------------------------------------------------------------------------- #

def pass_1_outcomes(result) -> Dict[str, str]:
    """Pass-1 outcome per device (merged or not), empty for no ledger."""
    report = result.report or result.unmerged_report
    if report is None:
        return {}
    return {str(l.device_id): l.outcome.value for l in report.lines}


def delivery_outcomes(result) -> Dict[str, str]:
    report = result.delivery_report
    if report is None:
        return {}
    return {str(l.device_id): l.outcome.value for l in report.lines}


def queue_sets(queue) -> List[frozenset]:
    return sorted((frozenset(str(d) for d in wp.devices) for wp in queue), key=sorted)


def state_stamps(states) -> Dict[str, float]:
    """Every time stamp a device state holds (the 0.0 'never' included)."""
    out: Dict[str, float] = {}
    for did, st in states.items():
        for name in ("last_contact_ts", "idle_time_ref_ts", "last_clean_ts",
                     "last_beacon_ts", "deadline_override_ts"):
            value = getattr(st, name, None)
            if value is not None:
                out[f"{did}.{name}"] = float(value)
    return out


def ledger_stamps(result) -> List[float]:
    """Every stamp in a mission's ledgers (report lines and times, delivery lines)."""
    out: List[float] = []
    for report in (result.report, result.unmerged_report):
        if report is not None:
            out += [l.contact_ts for l in report.lines]
            out += [report.started_at, report.finished_at]
    if result.contacts is not None:
        out += [r.contact_ts for r in result.contacts.records]
    if result.delivery_report is not None:
        out += [l.contact_ts for l in result.delivery_report.lines]
        out += [result.delivery_report.started_at, result.delivery_report.finished_at]
    out += list(result.pass_1_device_deadlines.values())
    out += [wp.deadline_ts for wp in result.pass_1_queue]
    return out
