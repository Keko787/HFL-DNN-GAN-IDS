"""Capture of Phase 3's simulated-clock pipeline at 6e6f92d (FeRRy Phase 4, unit UG4).

Freeze Rule 1 has two legacy faces in Phase 4: the wall-clock pipeline, pinned
by the afa9526 goldens beside this module, and Phase 3's simulated-clock
pipeline with ``plan_mode=legacy``, which only determinism and property tests
covered (``tests/integration/test_p3_ferry_trials.py``). This module records
the second as oracles, captured on the untouched tree at 6e6f92d: eight stub
trials, each run end to end through the code a real trial runs, and every
later Phase 4 unit must reproduce them with its switches at their defaults.

What runs, and what is stood in for:

* **Driver.** ``Exp4Driver.run_trial`` on a stub cell, unchanged: the arm
  table, T_nom, the clock settings, the topology builder, the provenance
  columns and the row, which it folds from the trial's JSONL as always
  (``consume_run_dir``).
* **Orchestrator.** The real ``MultiProcessOrchestrator.start_all`` writes the
  per-role JSON to its run directory (``_build_topology._StubbedOrchestrator``:
  placeholder ports, nothing spawned). Then, instead of spawning, the three
  roles run in this process from those files.
* **Roles.** The real ``ClusterService``, ``MuleService`` and ``DeviceService``
  are built from their JSON, each writing its own JSONL to the run directory.
  ``MuleService.run`` is the mule process's own service loop, so
  ``mule_ready``, ``mission_started``, ``mission_completed`` and the rest are
  emitted by the process code itself. The cluster's loop is driven by the
  harness: every UP goes through ``ClusterService._process_up`` (the loop's
  step for one UP, the loss draw, the fold, the events and the DOWNs) and the
  bootstrap through ``_dispatch_to_new_mules``. A device is the
  ``ClientMission`` its ``DeviceService`` builds (the stub trainer, its seeds,
  its contact reliability); the harness answers solicits and pushes through it
  at once, as ``serve_once`` would, instead of running the service loop, so
  the devices' traces hold only ``device_ready`` and the row's per-device
  serve counts are zero.
* **Links.** The TCP RF server and client and the TCP dock server and client
  are replaced by synchronous in-process links, with the Amendment 10 token
  check kept. ``time.time`` reads a wall clock only the harness moves (+2 s
  per solicit, +0.5 s per push, +3 s per upload, from 1.7e9 s, so a wall stamp
  leaking into simulated time is visible), and HERMES threads run their
  target on ``start()`` (``_host_harness.SyncThread``). Simulated time is the
  mule's own ``MissionClock``, as in the process.

A trial is therefore a pure function of its cell and settings (the same bytes
under any ``PYTHONHASHSEED``, whatever ran before it in the process), and
nothing is launched. What differs from a real-process trial is what the wall
clock, the OS and the device service loops decide there: the envelope ``ts``,
``duration_s``, the ports, the devices' serve events and the row columns built
from those. With no ``device_served`` event the row's ``coverage`` and
``participation_entropy`` are 0 and its ``jains_fairness`` is 1.0 (the index of
no serves), and ``mission_duration_s_mean`` is the harness clock's: harness
artifacts, not Phase 3's values. Every trial was run once through the real
orchestrator at 6e6f92d as well (real processes and TCP): every mule and
cluster event, the per-role JSON and every other row column agreed.

The trials (``TRIALS``): first the ones the unit spec names, H1 with the
in-flight re-plan and the trim fallback on wide, D1, D3, D4 route-only
(``agg:cutoff``), and the empty narrow mission (the Phase 3 cliff, trial T2 of
the Phase 3 final check). None of those flies a contact off wide, so three
more (added in the unit's review) pin the band classes' in-flight physics that
Phase 4 reworks (``FerryRuntime`` gains a mutable band, U6; the mule reads the
contact plan's band, U7): the cliff's other side, T2's field-wide narrow stop
admitted and flown at 99.5 s; H1 on medium with its pre-flight drops, an
in-flight re-plan and a member below the SNR floor at arrival; and H1 on
narrow at the measured payload, the driver's default payload mode, which the
declared 1 MB of every other trial never exercises.

Comparison (critic A3's rule, as in ``_canon``): every mapping in a row, a
per-role JSON or an event is stored with the keys it had at 6e6f92d and
compared on those keys only, at any depth, so a field added with a default
passes and a removed or renamed one fails. A mapping keyed by device id
(``deadline_state``, a stop's ``snr_db``) is data, not a record: it must keep
exactly its keys. Lists, scalars and each mule's sequence of event names must
match exactly; floats compare bit for bit.

    py -3.11 tests/golden/_build_p3_sim.py            # compare, write nothing
    py -3.11 tests/golden/_build_p3_sim.py --write    # only at 6e6f92d, clean tree
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Mapping, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.exp4 import driver as driver_mod  # noqa: E402
from experiments.exp4.driver import Exp4Driver  # noqa: E402
from experiments.runner import Cell  # noqa: E402
from hermes.observability import JsonEventEmitter  # noqa: E402
from hermes.processes import cluster as cluster_proc  # noqa: E402
from hermes.processes import device as device_proc  # noqa: E402
from hermes.processes import mule as mule_proc  # noqa: E402
from hermes.processes import orchestrator as orchestrator_mod  # noqa: E402
from hermes.processes.config import (  # noqa: E402
    cluster_config_from_json,
    device_config_from_json,
    mule_config_from_json,
)
from hermes.transport import DockLink, RFLink, RFLinkError  # noqa: E402
from hermes.transport.dock_link import DockLinkError, DockLinkTimeout  # noqa: E402
from hermes.types import DeviceID, MissionPass, MuleID  # noqa: E402

from tests.golden import _canon  # noqa: E402
from tests.golden._build_topology import (  # noqa: E402
    CLUSTER_PORT,
    _NoSubprocess,
    _StubbedOrchestrator,
    _TimeWithoutSleep,
)
from tests.golden._host_harness import SyncThread  # noqa: E402

#: The commit whose simulated-clock behaviour the fixture pins (main, the
#: Phase 3 docs follow-up), captured before any Phase 4 change.
BASE_COMMIT = "6e6f92da038227147489515d876cc3f353584283"
FIXTURE = "p3_sim"

#: The harness's wall clock: a realistic epoch, so a wall stamp that leaks into
#: a simulated field is caught by the mission clock's ceiling (1e9 s), moved
#: only by the harness's own events (the golden supervisor harness's steps).
WALL_T0 = 1_700_000_000.0
SOLICIT_DT = 2.0
PUSH_DT = 0.5
UP_DT = 3.0

_real_time = time.time
_real_thread = threading.Thread
TYPE_KEY = _canon.TYPE_KEY


# --------------------------------------------------------------------------- #
# The trials
# --------------------------------------------------------------------------- #

def _cell(arm: str, seed: int, **params: Any) -> Cell:
    p = {"N": 6, "rrf": 60.0, "n_missions": 4, "regime": "jittery"}
    p.update(params)
    return Cell(cell_id="|".join(f"{k}={v}" for k, v in sorted(p.items())), arm=arm,
                trial_index=0, seed=seed, params=p)


#: The Phase 4 pilots' common driver settings (Phase 4 spec, Decision 7: the
#: simulated clock, wide, the T_nom deadline unit, the re-plan with the trim
#: fallback, agg:cutoff, the channel reliability source), with a declared
#: 1 MB payload and the plan's 60 s knee budget (build plan L1009). At N = 6
#: and 1 MB an unbudgeted wide Pass 1 takes about 90 s at the median (Phase 4
#: planning probe), so the gates, the in-flight re-plan and the trim all bind.
PILOT: Dict[str, Any] = dict(
    mission_clock="sim", realism=True, contact_band="wide", deadline_time_scale="t_nom",
    in_flight_response="replan", replan_fallback="trim", aggregation="agg:cutoff",
    contact_reliability_source="channel", payload_bytes=1_000_000, mission_budget_s=60.0,
)

#: Trial T2 of the Phase 3 final check (the cliff of finding E2E1-01): the one
#: field-wide narrow stop of ``device_positions(8, 777, 100.0)`` needs 99.1 s
#: at 1 MB, so under a 60 s budget S3b admits nothing and every mission flies
#: empty (tests/unit/test_p3_final_fixes_mule.py pins it at the planner).
CLIFF: Dict[str, Any] = dict(
    mission_clock="sim", realism=True, contact_band="narrow", backhaul_model="seconds",
    payload_bytes=1_000_000, contact_reliability_source="origin", mission_budget_s=60.0,
    in_flight_response="abort", deadline_time_scale="t_nom",
)

#: One layout for the four H and D trials, as in one CSV. Seed 38 was chosen
#: (a probe over seeds 1-80) because every mechanism the unit spec names fires
#: on it: H1 re-plans in flight twice, once with the trim (order
#: ``arm_trimmed``), after S3b's pre-flight budget drops; D1 and D3 leave
#: stops out before takeoff and re-plan once each; D4's tour overruns the
#: budget on every mission; and the seconds model loses one upload each of
#: H1's and D3's.
SEED = 38

#: The cliff's other side: T2 at a 99.5 s budget, where S3b admits the one
#: field-wide narrow stop (99.1 s predicted; tests/unit/test_p3_final_fixes_mule.py
#: pins "all 8 at 99.5 s" at the planner). Each mission flies it with all eight
#: members in both passes (78 to 120 s of Pass-1 dwell at 1 MB), and one
#: overruns the budget in flight.
CLIFF_FLOWN_BUDGET_S = 99.5

#: The medium trial's layout. On seed 38 medium flies every stop with no drop
#: and no re-plan; seed 26 (a probe over seeds 1-80 at these settings) has
#: S3b's pre-flight budget drops priced on medium, an in-flight re-plan, a
#: Pass-1 member below the SNR floor at arrival (unreachable), backhaul
#: carrier 2 and a lost upload.
MEDIUM_SEED = 26

#: The pilots' flags at the measured payload (``payload_bytes`` unset, the
#: driver's default): sessions are priced on the θ and synthetic batch the mule
#: measures before each pass (``FerryRuntime.observe_payload``), which a
#: declared payload replaces. The stub's are small, so the airtime is short.
MEASURED: Dict[str, Any] = {k: v for k, v in PILOT.items() if k != "payload_bytes"}

#: name -> (driver settings, cell). H1 and D3 also run the seconds-axis
#: backhaul (its pricing, the keyed loss draw and the cluster's simulated
#: fields); D1 and D4 keep the pilots' recorded ``mission`` model. The last
#: three fly contacts on narrow and medium (see the module docstring).
TRIALS: Dict[str, Tuple[Dict[str, Any], Cell]] = {
    "h1_replan_trim_wide": (dict(PILOT, backhaul_model="seconds"), _cell("H1", SEED)),
    "d1_max_aoi": (dict(PILOT), _cell("D1", SEED)),
    "d3_whittle": (dict(PILOT, backhaul_model="seconds"), _cell("D3", SEED)),
    "d4_route_only": (dict(PILOT), _cell("D4", SEED)),
    "h1_narrow_cliff_empty": (dict(CLIFF), _cell("H1", 777, N=8, n_missions=3, regime="clean")),
    "h1_narrow_cliff_flown": (dict(CLIFF, mission_budget_s=CLIFF_FLOWN_BUDGET_S),
                              _cell("H1", 777, N=8, n_missions=3, regime="clean")),
    "h1_medium_replan": (dict(PILOT, contact_band="medium", backhaul_model="seconds"),
                         _cell("H1", MEDIUM_SEED)),
    "h1_narrow_measured": (dict(MEASURED, contact_band="narrow", backhaul_model="seconds"),
                           _cell("H1", SEED)),
}
TRIAL_NAMES = tuple(TRIALS)

#: The parts of a case, each compared on its own (one test each).
PARTS = ("row", "configs", "mule_events", "mule_ready", "mission_started",
         "mission_completed", "cluster_events", "device_events")


# --------------------------------------------------------------------------- #
# In-process links
# --------------------------------------------------------------------------- #

class WallClock:
    """``time.time`` while a trial runs: moved by the harness, never by a read."""

    def __init__(self, t0: float = WALL_T0) -> None:
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class MuleRF(RFLink):
    """A mule's RF server; its devices answer synchronously (stands in for
    ``TCPRFLinkServer``).

    A solicited device answers at once with its own ``build_ready_adv``, and a
    push runs through its own Pass-1 or Pass-2 handler, as
    ``ClientMission.serve_once`` does; a device whose advert was not eligible
    is not listening for a push. The server refuses a device whose link token
    differs from its own (Amendment 10), as the TCP server does.
    """

    def __init__(self, clock: WallClock, *, port: int, link_token: Optional[str]) -> None:
        self.clock = clock
        self.port = int(port)
        self.link_token = link_token
        self.order: List[DeviceID] = []
        self.devices: Dict[DeviceID, Any] = {}
        self._ready: Deque = deque()
        self._grads: Dict[DeviceID, Deque] = defaultdict(deque)
        self._acks: Dict[DeviceID, Deque] = defaultdict(deque)
        self._listening: Dict[DeviceID, int] = {}
        self.closed = False

    # ---- what MuleService calls on its server ------------------------------ #

    def start(self) -> None:
        return None

    def wait_for_devices(self, device_ids, timeout: Optional[float] = None) -> bool:
        return all(DeviceID(d) in self.devices for d in device_ids)

    def close(self) -> None:
        self.closed = True

    # ---- device registration (the device link's constructor) -------------- #

    def register(self, device_id: DeviceID, link_token: Optional[str]) -> None:
        if self.link_token is not None and link_token != self.link_token:
            raise ConnectionError(
                f"device {device_id!r} carries link token {link_token!r}, the mule "
                f"{self.link_token!r} (Amendment 10: the TCP server refuses it)")
        if device_id not in self.order:
            self.order.append(device_id)

    def attach(self, client_mission) -> None:
        self.devices[client_mission.device_id] = client_mission

    def _answer(self, did: DeviceID, solicit_id: int) -> None:
        cm = self.devices[did]
        adv = cm.build_ready_adv(in_reply_to=solicit_id)
        self._ready.append(adv)
        if adv.is_eligible() and adv.utility >= cm.fl_threshold:
            self._listening[did] = solicit_id
        else:
            self._listening.pop(did, None)

    # ---- mule side --------------------------------------------------------- #

    def broadcast_open_solicit(self, msg) -> None:
        self.clock.advance(SOLICIT_DT)
        for did in self.order:
            if did in self.devices:
                self._answer(did, 0)

    def solicit(self, msg, device_ids) -> List[DeviceID]:
        self.clock.advance(SOLICIT_DT)
        reached: List[DeviceID] = []
        for did in device_ids:
            if did in self.devices and did not in reached:
                reached.append(did)
        for did in reached:
            self._answer(did, int(msg.solicit_id))
        return reached

    def recv_ready_adv(self, timeout: Optional[float] = None):
        if not self._ready:
            raise RFLinkError("recv_ready_adv: nothing queued")
        return self._ready.popleft()

    def push_disc(self, device_id, msg) -> None:
        self.clock.advance(PUSH_DT)
        if device_id not in self.devices:
            raise RFLinkError(f"push_disc to {device_id!r}: no such device on the link")
        if device_id not in self._listening:
            return                                      # nobody is listening
        solicit_id = self._listening.pop(device_id)
        cm = self.devices[device_id]
        cm.last_push_round = msg.mission_round
        cm.last_push_pass = msg.pass_kind.value
        if msg.pass_kind is MissionPass.DELIVER:
            cm._handle_delivery_push(msg, in_reply_to=solicit_id)
        else:
            cm._handle_collect_push(msg, in_reply_to=solicit_id)

    def recv_gradient(self, device_id, timeout: Optional[float] = None):
        q = self._grads.get(device_id)
        if not q:
            raise RFLinkError(f"recv_gradient for {device_id!r}: nothing queued")
        return q.popleft()

    def recv_delivery_ack(self, device_id, timeout: Optional[float] = None):
        q = self._acks.get(device_id)
        if not q:
            raise RFLinkError(f"recv_delivery_ack for {device_id!r}: nothing queued")
        return q.popleft()

    # ---- what the devices send -------------------------------------------- #

    def send_gradient(self, msg) -> None:
        self._grads[msg.device_id].append(msg)

    def send_delivery_ack(self, msg) -> None:
        self._acks[msg.device_id].append(msg)

    def send_ready_adv(self, msg) -> None:
        self._ready.append(msg)

    def recv_open_solicit(self, device_id, timeout: Optional[float] = None):
        raise NotImplementedError("the harness drives the devices directly")

    def recv_disc_push(self, device_id, timeout: Optional[float] = None):
        raise NotImplementedError("the harness drives the devices directly")


class DeviceRF(RFLink):
    """A device's RF client (stands in for ``TCPRFLinkClient``).

    Registers with its mule's server on construction, as the TCP client's dial
    does, and forwards what the device's ``ClientMission`` sends.
    """

    connected = True

    def __init__(self, server: MuleRF, device_id: DeviceID, link_token: Optional[str]) -> None:
        self.server = server
        self.device_id = device_id
        server.register(device_id, link_token)

    def send_gradient(self, msg) -> None:
        self.server.send_gradient(msg)

    def send_delivery_ack(self, msg) -> None:
        self.server.send_delivery_ack(msg)

    def send_ready_adv(self, msg) -> None:
        self.server.send_ready_adv(msg)

    def recv_open_solicit(self, device_id, timeout: Optional[float] = None):
        raise NotImplementedError("the harness drives the devices directly")

    def recv_disc_push(self, device_id, timeout: Optional[float] = None):
        raise NotImplementedError("the harness drives the devices directly")

    def broadcast_open_solicit(self, msg) -> None:
        raise NotImplementedError("a device link has no mule side")

    def recv_ready_adv(self, timeout: Optional[float] = None):
        raise NotImplementedError("a device link has no mule side")

    def push_disc(self, device_id, msg) -> None:
        raise NotImplementedError("a device link has no mule side")

    def recv_gradient(self, device_id, timeout: Optional[float] = None):
        raise NotImplementedError("a device link has no mule side")

    def recv_delivery_ack(self, device_id, timeout: Optional[float] = None):
        raise NotImplementedError("a device link has no mule side")

    def close(self) -> None:
        return None


class Dock:
    """The cluster's dock server (stands in for ``TCPDockLinkServer``).

    One mule's UP is folded inside its upload: ``ClusterService`` first picks
    up newly docked mules and then processes the UP, the two things its
    service loop does for each UP. The bootstrap DOWN is dispatched when the
    mule first waits for a DOWN, as the loop would on seeing it register.
    """

    def __init__(self, world: "World") -> None:
        self.world = world
        self.registered: List[MuleID] = []
        self.downs: Dict[MuleID, Deque] = defaultdict(deque)

    @property
    def port(self) -> int:
        return CLUSTER_PORT

    def start(self) -> None:
        return None

    def registered_mules(self) -> List[MuleID]:
        return list(self.registered)

    def wait_for_mules(self, mule_ids, timeout: Optional[float] = None) -> bool:
        return set(MuleID(m) for m in mule_ids) <= set(self.registered)

    def send_down(self, bundle) -> None:
        self.downs[bundle.mule_id].append(bundle)

    def recv_up(self, timeout: Optional[float] = None):
        raise DockLinkTimeout("the harness folds each UP inside its upload")

    def close(self) -> None:
        return None


class MuleDock(DockLink):
    """One mule's dock client (stands in for ``TCPDockLinkClient``)."""

    def __init__(self, dock: Dock, mule_id: MuleID) -> None:
        self.dock = dock
        self.mule_id = MuleID(mule_id)
        if self.mule_id not in dock.registered:
            dock.registered.append(self.mule_id)

    def client_send_up(self, bundle) -> None:
        self.dock.world.on_up(bundle)

    def client_recv_down(self, mule_id, timeout: Optional[float] = None):
        q = self.dock.downs[MuleID(mule_id)]
        if not q:
            self.dock.world.dispatch_new_mules()
        if q:
            return q.popleft()
        # Nothing will come in this thread: let the caller's wait run out.
        time.sleep(min(float(timeout or 0.0), 0.005))
        raise DockLinkTimeout(f"client_recv_down for {mule_id!r} timed out after {timeout}s")

    def client_drain_down(self, mule_id):
        q = self.dock.downs[MuleID(mule_id)]
        out = list(q)
        q.clear()
        return out

    def is_available(self) -> bool:
        return True

    def recv_up(self, timeout: Optional[float] = None):
        raise DockLinkError("a mule's dock client has no server side")

    def send_down(self, bundle) -> None:
        raise DockLinkError("a mule's dock client has no server side")

    def close(self) -> None:
        return None


class World:
    """The in-process trial: clock, links and the three roles' services."""

    def __init__(self) -> None:
        self.clock = WallClock()
        self.dock = Dock(self)
        self.rfs: Dict[int, MuleRF] = {}
        self.cluster: Optional[cluster_proc.ClusterService] = None
        self.bootstrapped: set = set()
        self._next_rf_port: Optional[int] = None

    # ---- the constructors the process modules call ------------------------ #

    def dock_server(self, host: str, port: int, **_kw: Any) -> Dock:
        return self.dock

    def dock_client(self, mule_id, host: str, port: int) -> MuleDock:
        if int(port) != CLUSTER_PORT:
            raise AssertionError(f"mule {mule_id} dials dock port {port}, the cluster's is "
                                 f"{CLUSTER_PORT}")
        return MuleDock(self.dock, mule_id)

    def rf_server(self, host: str, port: int, link_token: Optional[str] = None) -> MuleRF:
        if self._next_rf_port is None:
            raise AssertionError("an RF server was built outside World.mule_service")
        rf = MuleRF(self.clock, port=self._next_rf_port, link_token=link_token)
        self.rfs[rf.port] = rf
        return rf

    def rf_client(self, device_id, host: str, port: int, newest_solicit_only: bool = False,
                  link_token: Optional[str] = None) -> DeviceRF:
        server = self.rfs.get(int(port))
        if server is None:
            raise AssertionError(f"device {device_id} dials RF port {port}: no mule there")
        return DeviceRF(server, DeviceID(device_id), link_token)

    # ---- roles ------------------------------------------------------------- #

    def mule_service(self, cfg, *, rf_port: int, events) -> mule_proc.MuleService:
        self._next_rf_port = int(rf_port)
        try:
            return mule_proc.MuleService(cfg, events=events)
        finally:
            self._next_rf_port = None

    # ---- the cluster's service loop, one step at a time ------------------- #

    def dispatch_new_mules(self) -> None:
        assert self.cluster is not None
        self.cluster._dispatch_to_new_mules(self.bootstrapped)

    def on_up(self, up) -> None:
        assert self.cluster is not None
        self.clock.advance(UP_DT)
        self.cluster._dispatch_to_new_mules(self.bootstrapped)
        self.cluster._process_up(up, cluster_proc._up_mission_round(up))


#: The link classes the process modules build, by module and name.
_LINK_SEAMS = (
    (cluster_proc, "TCPDockLinkServer", "dock_server"),
    (mule_proc, "TCPRFLinkServer", "rf_server"),
    (mule_proc, "TCPDockLinkClient", "dock_client"),
    (device_proc, "TCPRFLinkClient", "rf_client"),
)


@contextlib.contextmanager
def patched(world: World) -> Iterator[None]:
    """``time.time``, ``threading.Thread`` and the process modules' TCP links.

    A seam that no longer exists fails the capture instead of letting a real
    socket open (a renamed link class would otherwise be built for real, and
    its devices would find no mule: :meth:`World.rf_client`).
    """
    gone = [f"{m.__name__}.{n}" for m, n, _f in _LINK_SEAMS if not hasattr(m, n)]
    if gone:
        raise AssertionError(f"{gone} are gone: update the golden harness's link seams "
                             f"(tests/golden/_build_p3_sim.py)")
    saved = [(m, n, getattr(m, n)) for m, n, _f in _LINK_SEAMS]
    try:
        for module, name, factory in _LINK_SEAMS:
            setattr(module, name, getattr(world, factory))
        time.time = world.clock
        threading.Thread = SyncThread
        yield
    finally:
        time.time = _real_time
        threading.Thread = _real_thread
        for module, name, value in saved:
            setattr(module, name, value)


# --------------------------------------------------------------------------- #
# The orchestrator: real per-role JSON, roles run in this process
# --------------------------------------------------------------------------- #

class _InProcess:
    """What ``_spawn`` returns: a role that runs in this process.

    Alive until the trial has run; then ``poll`` and ``wait`` give its exit
    status (a mule's is ``MuleService.exit_code``).
    """

    pid = 0

    def __init__(self, name: str) -> None:
        self.name = name
        self.rc: Optional[int] = None

    def poll(self) -> Optional[int]:
        return self.rc

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        return self.rc

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


class InProcessOrchestrator(_StubbedOrchestrator):
    """``MultiProcessOrchestrator`` whose roles run in this process.

    ``start_all`` is the real one (per-role JSON in the run directory, the
    placeholder ports of ``_StubbedOrchestrator``); it then runs the whole
    trial (:func:`run_roles`) before returning, so the driver finds every mule
    exited. ``cleanup`` keeps the run directory's files in ``files`` before
    removing it.
    """

    last: Optional["InProcessOrchestrator"] = None

    def __init__(self, topology, *, python_executable=None, capture_output: bool = False):
        super().__init__(topology)
        self.files: Dict[str, str] = {}
        self.exit_codes: Dict[str, Optional[int]] = {}
        InProcessOrchestrator.last = self

    def _spawn(self, *, name, module, config_path, port_path):  # noqa: D401
        return _InProcess(name), None

    def start_all(self, *, timeout: float = 30.0) -> None:
        real = (orchestrator_mod.time, orchestrator_mod.subprocess)
        orchestrator_mod.time, orchestrator_mod.subprocess = _TimeWithoutSleep(), _NoSubprocess()
        try:
            super().start_all(timeout=timeout)
        finally:
            orchestrator_mod.time, orchestrator_mod.subprocess = real
        self.exit_codes = run_roles(self)

    def cleanup(self) -> None:
        for path in sorted(self.tmpdir.glob("*.json*")):
            self.files[path.name] = path.read_text(encoding="utf-8")
        super().cleanup()


def run_roles(orch: InProcessOrchestrator) -> Dict[str, Optional[int]]:
    """Run the trial's cluster, mules and devices in this process.

    Built in the orchestrator's start order from the JSON it wrote, each role
    with the JSONL emitter its process's ``main`` would open. Each mule runs
    its service loop and then its shutdown, as its process does. The cluster
    and the devices are stopped by the orchestrator in a real trial; on
    Windows ``TerminateProcess`` skips their ``finally`` blocks, so their
    emitters are closed here without the end-of-run events.
    """
    run_dir = orch.tmpdir
    topo = orch.topology
    if len(topo.mules) != 1:
        raise AssertionError("the in-process trial runs one mule (no simulated-order gate)")
    world = World()
    handles = {h.name: h for h in [orch.cluster_handle, *orch.mule_handles.values(),
                                   *orch.device_handles.values()]}

    def read(name: str) -> str:
        return (run_dir / name).read_text(encoding="utf-8")

    def emitter(role: str, node_id: str) -> JsonEventEmitter:
        return JsonEventEmitter(run_dir / f"{role}-{node_id}.jsonl", role=role, node_id=node_id,
                                clock=world.clock)

    codes: Dict[str, Optional[int]] = {}
    stopped_by_orchestrator: List[Any] = []          # the cluster and the devices
    with patched(world):
        try:
            ccfg = cluster_config_from_json(read("cluster.json"))
            world.cluster = cluster_proc.ClusterService(
                ccfg, events=emitter("cluster", ccfg.cluster_id))
            stopped_by_orchestrator.append(world.cluster)
            mules = []
            for mcfg0 in topo.mules:
                mcfg = mule_config_from_json(read(f"mule-{mcfg0.mule_id}.json"))
                port = orch.mule_handles[mcfg0.mule_id].actual_port
                mules.append(world.mule_service(mcfg, rf_port=port,
                                                events=emitter("mule", mcfg.mule_id)))
            for dcfg0 in topo.devices:
                dcfg = device_config_from_json(read(f"device-{dcfg0.device_id}.json"))
                svc = device_proc.DeviceService(dcfg, events=emitter("device", dcfg.device_id))
                stopped_by_orchestrator.append(svc)
                world.rfs[int(dcfg.mule_rf_port)].attach(svc.client)
            for svc in mules:
                try:
                    svc.run()
                finally:
                    svc.shutdown()
                codes[f"mule-{svc.cfg.mule_id}"] = svc.exit_code
        finally:
            for svc in stopped_by_orchestrator:
                svc.events.close()
    for name, handle in handles.items():
        handle.proc.rc = codes.get(name, 0)
    return codes


@contextlib.contextmanager
def in_process_orchestrator() -> Iterator[None]:
    """The driver's orchestrator is :class:`InProcessOrchestrator`.

    The orchestrator module's ``subprocess`` refuses ``Popen`` meanwhile, so a
    driver that reached a real orchestrator some other way fails at once
    instead of launching a trial's processes.
    """
    real = driver_mod.MultiProcessOrchestrator, orchestrator_mod.subprocess
    driver_mod.MultiProcessOrchestrator = InProcessOrchestrator
    orchestrator_mod.subprocess = _NoSubprocess()
    try:
        yield
    finally:
        driver_mod.MultiProcessOrchestrator, orchestrator_mod.subprocess = real


# --------------------------------------------------------------------------- #
# Canonical form of a trial
# --------------------------------------------------------------------------- #

def payload(value: Any, kind: str, device_ids: frozenset) -> Any:
    """``value`` in canonical form, every non-empty mapping a record (see the
    module docstring) unless its keys are all device ids."""
    if isinstance(value, Mapping):
        if not value:
            return {}
        if all(k in device_ids for k in value):
            return {str(k): payload(v, kind + "[dev]", device_ids) for k, v in value.items()}
        out: Dict[str, Any] = {TYPE_KEY: kind}
        for k, v in value.items():
            out[str(k)] = payload(v, f"{kind}.{k}", device_ids)
        return out
    if isinstance(value, (list, tuple)):
        return [payload(v, kind + "[]", device_ids) for v in value]
    return _canon.canon(value)


def _jsonl(text: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def inputs_of(settings: Mapping[str, Any], cell: Cell) -> Dict[str, Any]:
    """A trial's driver settings and cell, as the fixture stores them."""
    return {"settings": _canon.canon(dict(settings)),
            "cell": {"cell_id": cell.cell_id, "arm": cell.arm, "seed": cell.seed,
                     "trial_index": cell.trial_index, "params": _canon.canon(dict(cell.params))}}


def case_of(settings: Mapping[str, Any], cell: Cell, row: Mapping[str, Any],
            orch: InProcessOrchestrator) -> Dict[str, Any]:
    """One trial's canonical record, split into the parts the tests compare."""
    ids = frozenset(d.device_id for d in orch.topology.devices)
    files = orch.files

    def events(role_file: str) -> List[Dict[str, Any]]:
        return _jsonl(files.get(role_file, ""))

    mule_files = sorted(f for f in files if f.startswith("mule-") and f.endswith(".jsonl"))
    by_mule = {f[len("mule-"):-len(".jsonl")]: events(f) for f in mule_files}

    def of(name: str) -> Dict[str, Any]:
        return {m: [payload(e, f"mule.{name}", ids) for e in evs if e["event"] == name]
                for m, evs in by_mule.items()}

    named = ("mule_ready", "mission_started", "mission_completed")
    cluster = [e for f in sorted(files) if f.startswith("cluster-") and f.endswith(".jsonl")
               for e in events(f)]
    return {
        "inputs": inputs_of(settings, cell),
        "row": payload(dict(row), "Exp4Row", ids),
        "configs": {
            name: payload(json.loads(text), "config." + name.split("-")[0].split(".")[0], ids)
            for name, text in sorted(files.items()) if name.endswith(".json")
        },
        "mule_events": {
            "exit_codes": dict(orch.exit_codes),
            "names": {m: [e["event"] for e in evs] for m, evs in by_mule.items()},
            "other": {m: [payload(e, f"mule.{e['event']}", ids) for e in evs
                          if e["event"] not in named] for m, evs in by_mule.items()},
        },
        "mule_ready": of("mule_ready"),
        "mission_started": of("mission_started"),
        "mission_completed": of("mission_completed"),
        "cluster_events": [payload(e, f"cluster.{e['event']}", ids) for e in cluster],
        "device_events": {
            f[len("device-"):-len(".jsonl")]: [payload(e, f"device.{e['event']}", ids)
                                                for e in events(f)]
            for f in sorted(files) if f.startswith("device-") and f.endswith(".jsonl")
        },
    }


@functools.lru_cache(maxsize=None)
def capture(name: str) -> Dict[str, Any]:
    """Run trial ``name`` in this process; its canonical record (cached per process)."""
    settings, cell = TRIALS[name]
    driver = Exp4Driver(**settings)
    with in_process_orchestrator():
        InProcessOrchestrator.last = None
        row = dict(driver.run_trial(cell))
        orch = InProcessOrchestrator.last
    if orch is None:
        raise AssertionError(f"trial {name}: the driver never built the orchestrator")
    return case_of(settings, cell, row, orch)


def build_cases() -> Dict[str, Any]:
    return {name: capture(name) for name in TRIAL_NAMES}


# --------------------------------------------------------------------------- #
# Comparison helpers (also used by the tests)
# --------------------------------------------------------------------------- #

def added_keys(golden: Any, current: Any, path: str = "$") -> List[str]:
    """Keys a current record has that its golden record lacks (additive fields)."""
    out: List[str] = []
    if isinstance(golden, dict) and isinstance(current, dict):
        if TYPE_KEY in golden:
            out += [f"{path}.{k}" for k in current if k not in golden]
        for k in golden:
            if k in current and k != TYPE_KEY:
                out += added_keys(golden[k], current[k], f"{path}.{k}")
    elif isinstance(golden, list) and isinstance(current, list):
        for i, (g, c) in enumerate(zip(golden, current)):
            out += added_keys(g, c, f"{path}[{i}]")
    return out


def load_golden() -> Dict[str, Any]:
    return _canon.load(FIXTURE)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help=f"rewrite data/{FIXTURE}.json")
    ap.add_argument("--force", action="store_true",
                    help="allow --write away from a clean 6e6f92d tree (a golden that was wrong)")
    args = ap.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    logging.basicConfig(level=logging.ERROR, format="%(name)s: %(message)s")
    if args.write:
        head = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--", "hermes", "experiments")
        if (head != BASE_COMMIT or dirty) and not args.force:
            print(f"refusing to write: the fixture pins {BASE_COMMIT[:7]}, but HEAD is "
                  f"{head[:7]}" + (f" with changes in hermes/ or experiments/:\n{dirty}"
                                   if dirty else "")
                  + "\n(--force overrides; only for a golden that was itself wrong)",
                  file=sys.stderr)
            return 2
    t0 = time.time()
    cases = build_cases()
    took = time.time() - t0
    if args.write:
        meta = {"base_commit": BASE_COMMIT, "unit": "UG4",
                "what": "Phase 3 simulated-clock stub trials run in process: rows, per-role "
                        "JSON and every role's events"}
        path = _canon.dump(FIXTURE, meta, cases)
        print(f"wrote {path.relative_to(REPO)}: {len(cases)} trials, "
              f"{path.stat().st_size / 1024:.0f} KiB ({took:.1f} s)")
        return 0
    golden = load_golden()["cases"]
    status = 0
    for name in TRIAL_NAMES:
        for part in ("inputs",) + PARTS:
            g = golden.get(name, {}).get(part, "<missing>")
            c = cases[name][part]
            problems = _canon.diff(g, c)
            added = added_keys(g, c)
            if problems:
                status = 1
            if problems or added:
                print(f"{name}.{part}: {len(problems)} mismatch(es), {len(added)} added key(s)")
                for p in problems[:8]:
                    print(f"    {p}")
                for a in added[:8]:
                    print(f"    added: {a}")
    print(f"{len(TRIAL_NAMES)} trials x {len(PARTS)} parts compared "
          f"({'differ' if status else 'same'}; {took:.1f} s)")
    return status


if __name__ == "__main__":
    sys.exit(main())
