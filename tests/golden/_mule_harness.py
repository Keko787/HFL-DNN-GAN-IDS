"""Deterministic end-to-end ``MuleSupervisor`` runs at afa9526.

The supervisor, its scheduler, its mission server, real ``ClientMission``
devices and a real ``HFLHostCluster`` run in one thread, so a run is a pure
function of the scenario. Everything that is nondeterministic in the loopback
integration tests is replaced:

* **Clock.** ``time.time`` and the supervisor's ``now_fn`` read a virtual
  clock that only the harness moves: +2 s per solicit broadcast, +0.5 s per
  push, +3 s per dock upload and +20 s between missions. A read never moves it,
  so a later change that reads the clock more often shifts nothing; flight
  legs take no clock time, as in legacy mode (the pose jumps).
* **Threads.** ``threading.Thread`` runs its target on ``start()``: the
  mission server's per-device workers and the devices' train-ahead fits run in
  a fixed order (critic R6).
* **Devices.** :class:`SyncDeviceRF` answers every solicit at once, in
  registration order, with each device's own ``build_ready_adv()``, and runs
  a push through the device's own Pass-1 or Pass-2 handler on the spot. A
  device whose advert was not eligible is not listening for a push, as in
  ``ClientMission.serve_once``. Trainers are pure functions of the basis.
* **Cluster.** :class:`SyncDock` runs the cluster side inside the upload, with
  ``ClusterService``'s dispatch policy: ingest, fold, and on a merge close the
  round and answer every mule waiting at the dock; on an expiry or a FedBuff
  deferral answer the uploader (and, after an expiry, everyone waiting).
* **Several mules (critic B6).** A mule that waits for a quorum runs the other
  mule's next mission inside its own blocking DOWN wait
  (:meth:`DockServer.run_waiting_task`), so the order is fixed; a wait with no
  such task times out after ``down_wait_s`` of real time, which is survived.

What is recorded per mission: the whole ``MissionRunResult``, every scheduler
device state, every delta the scheduler ingested (from sessions and from
widening), the mule's RF calls, the stash, the pose, the S3b result and plan
deadlines, the FedEx diagnostics; per dock: the UP and DOWN bundles and the
fold's outcome.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import OUTCOME_DEFERRED, OUTCOME_EXPIRED, StubGeneratorHost
from hermes.mission import ClientMission, LocalTrainResult
from hermes.mission.aggregation_rules import AGG_CUTOFF, AggregationSpec
from hermes.mule import MuleSupervisor
from hermes.scheduler.policies import FedExCarpPolicy, MaxAoIPolicy
from hermes.scheduler.selector import TargetSelectorRL
from hermes.scheduler.stages.s3_deadline import DeadlineLaw
from hermes.scheduler.stages.s3c_mission_window import MissionWindowAdapter
from hermes.transport import DockLink, RFLink, RFLinkError
from hermes.transport.dock_link import DockLinkError, DockLinkTimeout
from hermes.types import DeviceID, FLState, MissionPass, MuleID, SpectrumSig

from tests.golden._canon import array_digest, canon, record
from tests.golden._host_harness import SyncThread

T0 = 1_000_000.0
SOLICIT_DT = 2.0
PUSH_DT = 0.5
UP_DT = 3.0
BETWEEN_MISSIONS_DT = 20.0
DOCK_POSE = (0.0, 0.0, 0.0)
_real_time = time.time
_real_thread = threading.Thread


class VirtualClock:
    def __init__(self, t0: float = T0) -> None:
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class Patched:
    """``time.time`` -> the virtual clock, ``threading.Thread`` -> synchronous."""

    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock

    def __enter__(self):
        time.time = self.clock
        threading.Thread = SyncThread
        return self

    def __exit__(self, *exc):
        time.time = _real_time
        threading.Thread = _real_thread
        return False


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #

def make_trainer(idx: int):
    """A pure function of the basis, so call order cannot change a result."""
    def _train(theta, synth):
        after = [np.asarray(w, dtype=np.float32) + np.float32(0.01 * (idx + 1)) for w in theta]
        return LocalTrainResult(
            delta_theta=after, num_examples=4 + idx, accuracy=0.60 + 0.03 * idx,
            auc=0.70 + 0.02 * idx, loss=0.40 - 0.03 * idx, theta_after=after,
        )
    return _train


class SyncDeviceRF(RFLink):
    """One mule's RF link with its devices answering synchronously."""

    def __init__(self, clock: VirtualClock, *, solicit_dt: float = SOLICIT_DT,
                 push_dt: float = PUSH_DT) -> None:
        self.clock = clock
        self.solicit_dt = float(solicit_dt)
        self.push_dt = float(push_dt)
        self.order: List[DeviceID] = []
        self.devices: Dict[DeviceID, ClientMission] = {}
        self._ready: Deque = deque()
        self._grads: Dict[DeviceID, Deque] = defaultdict(deque)
        self._acks: Dict[DeviceID, Deque] = defaultdict(deque)
        self._listening_for_push: set = set()
        self.calls: List[list] = []

    # ClientMission registers itself on construction.
    def register_device(self, device_id: DeviceID) -> None:
        if device_id not in self.order:
            self.order.append(device_id)

    def attach(self, cm: ClientMission) -> None:
        self.devices[cm.device_id] = cm

    # ---- mule side ------------------------------------------------------- #

    def broadcast_open_solicit(self, msg) -> None:
        self.clock.advance(self.solicit_dt)
        self.calls.append(["solicit", canon(msg.pass_kind), msg.mission_round])
        for did in self.order:
            cm = self.devices[did]
            adv = cm.build_ready_adv()                 # ClientMission.serve_once
            self._ready.append(adv)
            if adv.is_eligible() and adv.utility >= cm.fl_threshold:
                self._listening_for_push.add(did)
            else:
                self._listening_for_push.discard(did)

    def recv_ready_adv(self, timeout=None):
        if not self._ready:
            raise RFLinkError("recv_ready_adv: nothing queued")
        return self._ready.popleft()

    def push_disc(self, device_id, msg) -> None:
        self.clock.advance(self.push_dt)
        self.calls.append([
            "push", str(device_id), canon(msg.pass_kind), msg.mission_round,
            msg.basis_version, msg.update_form, bool(msg.train_ahead), msg.weights_sig[:16],
        ])
        if device_id not in self._listening_for_push:
            return                                     # nobody is listening
        self._listening_for_push.discard(device_id)
        cm = self.devices[device_id]
        cm.last_push_round = msg.mission_round
        cm.last_push_pass = msg.pass_kind.value
        if msg.pass_kind is MissionPass.DELIVER:
            cm._handle_delivery_push(msg)
        else:
            cm._handle_collect_push(msg)

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

    # ---- device side (called by the ClientMission handlers) --------------- #

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


# --------------------------------------------------------------------------- #
# Cluster
# --------------------------------------------------------------------------- #

class DockServer:
    """``ClusterService``'s fold and dispatch policy, run inside the upload."""

    def __init__(self, cluster: HFLHostCluster, clock: VirtualClock) -> None:
        self.cluster = cluster
        self.clock = clock
        self.dock: Optional["SyncDock"] = None
        self.awaiting: List[MuleID] = []
        self.tasks: Dict[MuleID, Deque[Callable[[], None]]] = defaultdict(deque)
        self.log: List[Any] = []

    def theta_digest(self) -> Any:
        return [array_digest(w) for w in self.cluster.generator.get_global_disc_weights()]

    def bootstrap(self, mule_id: MuleID) -> None:
        self.send_down(mule_id, why="bootstrap")

    def send_down(self, mule_id: MuleID, *, why: str) -> None:
        bundle = self.cluster.dispatch_down_bundle(mule_id)
        self.log.append(["down", why, str(mule_id), canon(bundle)])
        assert self.dock is not None
        self.dock.send_down(bundle)

    def release(self, why: str) -> None:
        waiting, self.awaiting = list(self.awaiting), []
        for mid in waiting:
            self.send_down(mid, why=why)

    @staticmethod
    def up_summary(up) -> Dict[str, Any]:
        """The UP bundle, short: its report and ledgers are the mission result's."""
        pa = up.partial_aggregate
        prev = up.prev_mission_delivery_report
        return record("UpSummary", {
            "mule_id": up.mule_id,
            "bundle_sig": up.bundle_sig,
            "mission_round": pa.mission_round,
            "num_examples": pa.num_examples,
            "contributing_devices": pa.contributing_devices,
            "weights": pa.weights,
            "rule": pa.rule,
            "base_version": pa.base_version,
            "report": [[str(l.device_id), l.outcome] for l in up.round_close_report.lines],
            "contacts": [[str(r.device_id), r.in_session] for r in up.contact_history.records],
            "prev_delivery": None if prev is None else [
                prev.mission_round, [[str(l.device_id), l.outcome] for l in prev.lines]],
        })

    def on_up(self, up) -> None:
        self.clock.advance(UP_DT)
        accepted = self.cluster.ingest_up_bundle(up)
        if up.mule_id not in self.awaiting:
            self.awaiting.append(up.mule_id)
        self.log.append(["up", str(up.mule_id), bool(accepted), self.up_summary(up)])
        if not accepted:
            return
        merged = self.cluster.aggregate_pending()
        outcome = self.cluster.last_outcome
        # ``last_merge`` is a diagnostic dict the cluster's trace reports from;
        # compared on its afa9526 keys, so a key added later is not a change.
        last_merge = self.cluster.last_merge
        self.log.append([
            "fold", str(up.mule_id), outcome, merged is not None,
            self.cluster.cluster_round,
            None if last_merge is None else record("LastMerge", last_merge),
        ])
        if merged is None and outcome in (OUTCOME_DEFERRED, OUTCOME_EXPIRED):
            if up.mule_id in self.awaiting:
                self.awaiting.remove(up.mule_id)
            self.send_down(up.mule_id, why=f"post-{outcome}")
            if outcome == OUTCOME_EXPIRED:
                self.release("post-expiry")
        if merged is not None:
            self.cluster.close_cluster_round()
            self.log.append(["closed", self.cluster.cluster_round, self.theta_digest()])
            self.release("post-aggregation")

    def run_waiting_task(self, mule_id: MuleID) -> None:
        q = self.tasks.get(mule_id)
        if q:
            q.popleft()()


class SyncDock(DockLink):
    def __init__(self, server: DockServer) -> None:
        self.server = server
        server.dock = self
        self._down: Dict[MuleID, Deque] = defaultdict(deque)

    def recv_up(self, timeout=None):
        raise DockLinkError("the harness cluster runs inside the upload")

    def send_down(self, bundle) -> None:
        self._down[bundle.mule_id].append(bundle)

    def client_send_up(self, bundle) -> None:
        self.server.on_up(bundle)

    def client_recv_down(self, mule_id, timeout=None):
        if not self._down[mule_id]:
            self.server.run_waiting_task(mule_id)
        if self._down[mule_id]:
            return self._down[mule_id].popleft()
        # Nothing will come in this thread: let the caller's real-time wait
        # run out without spinning (only survivable waits get here).
        time.sleep(min(float(timeout or 0.0), 0.005))
        raise DockLinkTimeout(f"client_recv_down for {mule_id!r} timed out after {timeout}s")

    def client_drain_down(self, mule_id):
        q = self._down[mule_id]
        out = list(q)
        q.clear()
        return out

    def is_available(self) -> bool:
        return True

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# The world
# --------------------------------------------------------------------------- #

#: Single-mule layout at a 60 m RF range: one contact at the dock, a far pair,
#: and three isolated devices.
LAYOUT: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("dev-00", (0.0, 0.0, 0.0)),
    ("dev-01", (20.0, 10.0, 0.0)),
    ("dev-02", (150.0, 0.0, 0.0)),
    ("dev-03", (170.0, 25.0, 0.0)),
    ("dev-04", (-90.0, 80.0, 0.0)),
    ("dev-05", (40.0, -130.0, 0.0)),
    ("dev-06", (-210.0, -150.0, 0.0)),
)
#: A device with a flaky uplink (ClientMission's seeded Bernoulli).
FLAKY = {"dev-04": (0.6, 11)}

#: Six single-device contacts on a 70 m ring around the dock. S3b's EDF walk
#: admits the first three and drops the rest as overdue; with slow exchanges
#: (``push_dt``) the mule then falls behind its plan in flight.
RING: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("dev-00", (70.0, 0.0, 0.0)),
    ("dev-01", (35.0, 60.62, 0.0)),
    ("dev-02", (-35.0, 60.62, 0.0)),
    ("dev-03", (-70.0, 0.0, 0.0)),
    ("dev-04", (-35.0, -60.62, 0.0)),
    ("dev-05", (35.0, -60.62, 0.0)),
)


class World:
    def __init__(self, clock: VirtualClock, *, mule_ids: Sequence[str],
                 assignment: Optional[Dict[str, str]] = None,
                 layout=LAYOUT, flaky=FLAKY, aggregation: Optional[AggregationSpec] = None,
                 min_participation: int = 1, solicit_dt: float = SOLICIT_DT,
                 push_dt: float = PUSH_DT) -> None:
        self.clock = clock
        self.mule_ids = [MuleID(m) for m in mule_ids]
        self.aggregation = aggregation or AggregationSpec()
        registry = DeviceRegistry()
        for did, pos in layout:
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
            dock=None,  # type: ignore[arg-type]  (the harness dock is set below)
            synth_batch_size=2, min_participation=min_participation,
            aggregation=self.aggregation,
        )
        self.server = DockServer(self.cluster, clock)
        self.dock = SyncDock(self.server)
        self.cluster.dock = self.dock
        self.rfs: Dict[MuleID, SyncDeviceRF] = {}
        self.devices: Dict[DeviceID, ClientMission] = {}
        for idx, (did, _pos) in enumerate(layout):
            mid = registry.get(DeviceID(did)).assigned_mule
            if mid not in self.rfs:
                self.rfs[mid] = SyncDeviceRF(clock, solicit_dt=solicit_dt, push_dt=push_dt)
            rf = self.rfs[mid]
            rel, seed = flaky.get(did, (None, None))
            cm = ClientMission(
                device_id=DeviceID(did), rf=rf, local_train=make_trainer(idx),
                solicit_timeout_s=1.0, disc_push_timeout_s=1.0,
                contact_reliability=rel, contact_rng_seed=seed,
            )
            cm.set_state(FLState.FL_OPEN)
            rf.attach(cm)
            self.devices[DeviceID(did)] = cm
        self.sups: Dict[MuleID, MuleSupervisor] = {}
        self.deltas: Dict[MuleID, List[Any]] = defaultdict(list)

    def supervisor(self, mule_id: str, **kwargs) -> MuleSupervisor:
        mid = MuleID(mule_id)
        kwargs.setdefault("session_ttl_s", 3.0)
        kwargs.setdefault("rf_range_m", 60.0)
        sup = MuleSupervisor(
            mule_id=mid, rf=self.rfs[mid], dock=self.dock, now_fn=self.clock,
            aggregation=self.aggregation, **kwargs,
        )
        # Record every delta the scheduler ingests: from sessions (the mission
        # server's bus, bound at construction) and from widening (direct).
        ingest = sup.scheduler.ingest_round_close_delta
        sink = self.deltas[mid]

        def _bus(delta, _ingest=ingest, _sink=sink):
            _sink.append(["session", canon(delta)])
            return _ingest(delta)

        def _direct(delta, _ingest=ingest, _sink=sink):
            _sink.append(["direct", canon(delta)])
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

    # ---- records ---------------------------------------------------------- #

    def mission_record(self, mule_id: MuleID, result) -> Dict[str, Any]:
        sup = self.sups[mule_id]
        sch = sup.scheduler
        rf = self.rfs[mule_id]
        policy = sch.target_selector
        out: Dict[str, Any] = {
            "mule": str(mule_id),
            "result": canon(result),
            "states": canon(sch.device_states),
            "deltas": list(self.deltas[mule_id]),
            "rf": list(rf.calls),
            "stash": canon(list(sup.mission._misrouted_advs)),
            # Adverts still queued on the link: the next gathers read them first.
            "link_backlog": [str(a.device_id) for a in rf._ready],
            "pose": canon(sup.mule_pose),
            "next_theta_version": sup._next_theta_version,
            "next_theta": None if sup._next_theta is None else canon(sup._next_theta),
            "last_feasibility": canon(sch.last_feasibility),
            "last_plan_deadlines": canon(sch.last_plan_deadlines),
            "window_scale": canon(sch.window_scale),
            "mission_start_ts": canon(sch.mission_start_ts),
            "stale_downs_dropped": sup.client_cluster.stale_downs_dropped,
            "retry_queue_depth": sup.client_cluster.retry_queue_depth(),
            # The ledger the next UP carries: this mission's, an older one, or
            # none (its lines are in the result or in the UP summary).
            "pending_delivery_report": (
                None if sup._pending_delivery_report is None
                else "this_mission" if result is not None
                and sup._pending_delivery_report is getattr(result, "delivery_report", None)
                else sup._pending_delivery_report.mission_round
            ),
            "clock": canon(self.clock()),
        }
        if isinstance(policy, FedExCarpPolicy):
            out["fedex"] = record("FedExDiagnostics", {
                k: getattr(policy, k) for k in (
                    "last_tour_cost_s", "last_tour_fits", "last_tour_overrun_s",
                    "last_return_leg_s", "last_fits_without_return",
                    "last_overrun_without_return_s",
                )
            })
        adapter = getattr(sch, "_window_adapter", None)
        if adapter is not None:
            # S3c's own record: (served, planned) per mission, as the mule
            # reported them (``mission_served_devices`` after an abort).
            out["s3c"] = record("S3cAdapter", {
                "history": [list(h) for h in getattr(adapter, "_history", ())],
                "success_rate": adapter.success_rate,
                "scale": adapter.scale,
            })
        self.deltas[mule_id].clear()
        rf.calls.clear()
        return out

    def device_record(self) -> Dict[str, Any]:
        return {
            str(did): record("DeviceSide", {
                "state": cm.state,
                "last_utility": cm.last_utility,
                "last_push_round": cm.last_push_round,
                "last_push_pass": cm.last_push_pass,
                "theta_basis_version": cm._theta_basis_version,
                "prepared_basis_version": cm._prepared_basis_version,
                "has_prepared": cm._prepared_delta is not None,
            })
            for did, cm in self.devices.items()
        }

    def cluster_record(self) -> Dict[str, Any]:
        reg = self.cluster.registry
        return {
            "cluster_round": self.cluster.cluster_round,
            "theta": self.server.theta_digest(),
            "registry": canon({str(r.device_id): r for r in reg.all()}),
            "log": list(self.server.log),
        }


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

def _single(name: str, *, missions: int = 3, sup_kwargs: Optional[dict] = None,
            aggregation: Optional[AggregationSpec] = None,
            before: Optional[Callable[[World, int], None]] = None,
            **world_kwargs) -> Dict[str, Any]:
    clock = VirtualClock()
    with Patched(clock):
        world = World(clock, mule_ids=["mule-g"], aggregation=aggregation, **world_kwargs)
        sup = world.supervisor("mule-g", **(sup_kwargs or {}))
        world.bootstrap()
        trace: Dict[str, Any] = {"bootstrap": world.mission_record(MuleID("mule-g"), None),
                                 "missions": []}
        for m in range(missions):
            if before is not None:
                before(world, m)
            result = sup.run_one_mission()
            trace["missions"].append(world.mission_record(MuleID("mule-g"), result))
            clock.advance(BETWEEN_MISSIONS_DT)
        trace["devices"] = world.device_record()
        trace["cluster"] = world.cluster_record()
    return trace


def _refuse_dev05_in_mission_2(world: World, m: int) -> None:
    """dev-05 refuses mission 2 (not FL_OPEN), then comes back."""
    if m == 1:
        world.set_state(["dev-05"], FLState.UNAVAILABLE)
    elif m == 2:
        world.set_state(["dev-05"], FLState.FL_OPEN)


def scenario_h1_no_budget() -> Dict[str, Any]:
    return _single("h1_no_budget", before=_refuse_dev05_in_mission_2)


def scenario_h1_budget() -> Dict[str, Any]:
    """Pre-flight S3b drops and their widening (EDF ties break on position)."""
    return _single("h1_budget", sup_kwargs=dict(mission_budget_s=70.0),
                   before=_refuse_dev05_in_mission_2)


def scenario_h1_budget_abort() -> Dict[str, Any]:
    """In flight, slow exchanges put the mule behind its plan: the next
    contact is overdue, Pass 1 aborts and the abandoned tail is widened."""
    return _single("h1_budget_abort", sup_kwargs=dict(mission_budget_s=200.0),
                   layout=RING, flaky={}, push_dt=25.0)


def scenario_h1_s3c_budget_abort() -> Dict[str, Any]:
    """The in-flight abort with S3c on: S3c is told the devices of the
    contacts flown, not of the abandoned tail (``mission_served_devices``),
    and the planned count includes the pre-flight drops."""
    return _single("h1_s3c_budget_abort", sup_kwargs=dict(
        mission_budget_s=200.0,
        mission_window_adapter=MissionWindowAdapter(enabled=True, window=2),
    ), layout=RING, flaky={}, push_dt=25.0)


def scenario_d1_max_aoi_budget_abort() -> Dict[str, Any]:
    """D1 in flight: only the budget is re-checked (Amendment 8), and it runs out."""
    return _single("d1_max_aoi_budget_abort", sup_kwargs=dict(
        target_selector=MaxAoIPolicy(), mission_budget_s=100.0),
        layout=RING, flaky={}, push_dt=25.0)


def scenario_h2_selector() -> Dict[str, Any]:
    return _single("h2_selector", sup_kwargs=dict(
        target_selector=TargetSelectorRL(epsilon=0.0, rng_seed=0), rf_prior_snr_db=12.5))


def scenario_d1_max_aoi_budget() -> Dict[str, Any]:
    return _single("d1_max_aoi_budget", sup_kwargs=dict(
        target_selector=MaxAoIPolicy(), mission_budget_s=70.0))


def scenario_d4_fedex() -> Dict[str, Any]:
    return _single("d4_fedex", sup_kwargs=dict(
        target_selector=FedExCarpPolicy(depot=DOCK_POSE), mission_budget_s=70.0))


def scenario_pass_2_budget() -> Dict[str, Any]:
    return _single("pass_2_budget", aggregation=AggregationSpec(rule=AGG_CUTOFF),
                   sup_kwargs=dict(mission_budget_s=70.0, pass_2_budget=True))


def scenario_h1_law_s3c() -> Dict[str, Any]:
    return _single("h1_law_s3c", missions=4, sup_kwargs=dict(
        mission_budget_s=45.0,
        deadline_law=DeadlineLaw(form="multiplicative"),
        miss_priority=True,
        mission_window_adapter=MissionWindowAdapter(enabled=True, window=2),
    ), before=_refuse_dev05_in_mission_2)


#: K = 2: two slices of the same layout plus a device of B's that no contact
#: of A's can reach.
K2_ASSIGNMENT = {
    "dev-00": "mule-a", "dev-01": "mule-a", "dev-04": "mule-a", "dev-06": "mule-a",
    "dev-02": "mule-b", "dev-03": "mule-b", "dev-05": "mule-b",
}


def _k2(name: str, *, missions: int, down_wait_s: float, dock_on_empty: bool,
        nest: Callable[[int], bool],
        before: Optional[Callable[[World, int], None]] = None) -> Dict[str, Any]:
    """Two mules on one quorum-2 cluster.

    In mission ``m`` mule B flies inside mule A's DOWN wait when ``nest(m)``;
    otherwise A flies alone (its wait runs out) and B flies afterwards.
    """
    clock = VirtualClock()
    ma, mb = MuleID("mule-a"), MuleID("mule-b")
    with Patched(clock):
        world = World(clock, mule_ids=[ma, mb], assignment=K2_ASSIGNMENT, min_participation=2)
        sup_a = world.supervisor(ma, down_wait_s=down_wait_s, dock_on_empty=dock_on_empty)
        sup_b = world.supervisor(mb, down_wait_s=down_wait_s, dock_on_empty=dock_on_empty)
        world.bootstrap()
        trace: Dict[str, Any] = {
            "bootstrap": [world.mission_record(ma, None), world.mission_record(mb, None)],
            "missions": [],
        }
        for m in range(missions):
            if before is not None:
                before(world, m)
            done: List[Tuple[str, Any]] = []

            def _fly_b():
                res = sup_b.run_one_mission()
                done.append(("b", world.mission_record(mb, res)))

            if nest(m):
                world.server.tasks[ma].append(_fly_b)
            res_a = sup_a.run_one_mission()
            done.append(("a", world.mission_record(ma, res_a)))
            if world.server.tasks[ma]:           # A never waited: B flies now
                world.server.tasks[ma].popleft()()
            elif not nest(m):
                _fly_b()
            trace["missions"].append([[who, rec] for who, rec in done])
            clock.advance(BETWEEN_MISSIONS_DT)
        trace["devices"] = world.device_record()
        trace["cluster"] = world.cluster_record()
    return trace


def scenario_k2_quorum_dock_on_empty() -> Dict[str, Any]:
    def before(world: World, m: int) -> None:
        if m == 1:                     # B collects nothing and docks empty
            world.set_state(["dev-02", "dev-03", "dev-05"], FLState.UNAVAILABLE)
        elif m == 2:                   # both empty: an all-empty fold
            world.set_state(["dev-00", "dev-01", "dev-04", "dev-06"], FLState.UNAVAILABLE)
    return _k2("k2_quorum_dock_on_empty", missions=3, down_wait_s=30.0,
               dock_on_empty=True, nest=lambda m: True, before=before)


def scenario_k2_down_timeout() -> Dict[str, Any]:
    # Mission 1: A's quorum wait runs out (survived: Pass 2 skipped, θ
    # restaged); B then closes the round and A's answer goes stale in its
    # queue. Missions 2-3: A drops the stale DOWN before uploading and B flies
    # inside A's wait.
    return _k2("k2_down_timeout", missions=3, down_wait_s=0.3,
               dock_on_empty=True, nest=lambda m: m > 0)


SCENARIOS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "h1_no_budget": scenario_h1_no_budget,
    "h1_budget": scenario_h1_budget,
    "h1_budget_abort": scenario_h1_budget_abort,
    "h1_s3c_budget_abort": scenario_h1_s3c_budget_abort,
    "h2_selector": scenario_h2_selector,
    "d1_max_aoi_budget": scenario_d1_max_aoi_budget,
    "d1_max_aoi_budget_abort": scenario_d1_max_aoi_budget_abort,
    "d4_fedex": scenario_d4_fedex,
    "pass_2_budget": scenario_pass_2_budget,
    "h1_law_s3c": scenario_h1_law_s3c,
    "k2_quorum_dock_on_empty": scenario_k2_quorum_dock_on_empty,
    "k2_down_timeout": scenario_k2_down_timeout,
}


def build_cases() -> Dict[str, Any]:
    return {name: fn() for name, fn in SCENARIOS.items()}
