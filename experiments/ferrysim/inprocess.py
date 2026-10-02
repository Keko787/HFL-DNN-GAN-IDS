"""FerrySim's in-process trial: the stack's own trial, run in one process.

FeRRy Phase 5, unit U8a (the Phase 5 spec, other choices 8; the user's
decision 2 (a)). FerrySim practises the pair score on the real system: the
Exp 4 driver's own trial (``Exp4Driver.run_trial``), its per-role JSON, and the
real cluster, mule and device services, run in this process on synchronous
links and a virtual wall clock. Beyond the links and the clock, two things are
stood in for: the devices' local training (:data:`DEVICE_MODELS`), and their
service loops, which do not run, so their traces hold only ``device_ready``
(the device-serve columns, below). This module is a copy of UG4's in-process
orchestrator (``tests/golden/_build_p3_sim.py``, the 6e6f92d golden harness),
with the helpers it takes from ``tests`` copied too (critic C7):
``experiments`` never imports ``tests``, and the golden harness is not edited.
A parity test runs a trial through this copy and compares it, part for part,
with UG5's oracle of the same trial (``tests/golden/data/p4_plan.json``), so
drift in the copy shows; another runs a trial through the real orchestrator
(real processes and TCP) and compares the row, the configs and every mule and
cluster event (critic B8), the row bar its wall and device-serve columns.

What runs, and what is stood in for (as in UG4's harness):

* **Driver.** ``Exp4Driver.run_trial``, unchanged: the arm table, T_nom, the
  clock settings, the topology builder, the provenance columns and the row.
* **Orchestrator.** The real ``MultiProcessOrchestrator.start_all`` writes the
  per-role JSON to its run directory (:class:`_StubbedOrchestrator`:
  placeholder ports, nothing spawned); the roles then run in this process from
  those files (:func:`run_roles`).
* **Roles.** The real ``ClusterService``, ``MuleService`` and
  ``DeviceService``, each writing its own JSONL. ``MuleService.run`` is the
  mule process's own loop. The cluster's loop is driven one UP at a time
  (``ClusterService._process_up``), and each device is the ``ClientMission``
  its ``DeviceService`` builds, answering solicits and pushes at once, as
  ``serve_once`` would. The device process's service loop
  (``DeviceService.run``) does not run, so a device logs ``device_ready``
  (when its service is built) and never ``device_served``, which only that
  loop logs.
* **Links.** Synchronous in-process links stand in for TCP, with the
  Amendment 10 token check kept; ``time.time`` reads a wall clock only the
  harness moves (+2 s per solicit, +0.5 s per push, +3 s per upload), and
  HERMES threads run their target on ``start()`` (:class:`SyncThread`).
  Simulated time is the mule's own ``MissionClock``.

**The device-serve columns** (the orchestrator's resolution R25; the final
check's FS-1). What differs from a real-process trial is what the wall
clock, the OS and the devices' service loops decide there: the envelope
``ts``, ``duration_s``, the ports, the planner's wall time
(:data:`WALL_TOKEN`), the devices' serve events and the row columns built
from those. With no ``device_served`` event the driver's row has
``coverage`` 0.0, ``participation_entropy`` 0.0 and ``jains_fairness`` 1.0
(the index of no serves), where a real trial's come from its devices' serve
counts, and its ``mission_duration_s_mean`` is the harness clock's: harness
artifacts, as UG4 declared for its harness, in every FerrySim trial whatever
flies it and under either device model. The trace scorer's row of a kept
trace (:mod:`experiments.ferrysim.episode`) holds the same three values, as
it reads the same device events. No study reads these columns: FerrySim's
reward, training, validation, evaluation, report and headroom read the
flight from the mule's own records (:mod:`experiments.ferrysim.episode`),
never the row, and the evaluation keeps no traces. The parity tests compare
the row with these four columns masked, and pin the three serve values.

**What FerrySim adds** (:class:`RoleHooks`), each off by default so that a
trial with no hooks is UG4's trial exactly:

* ``on_mule``: callables given each ``MuleService`` once every role is built
  and before its loop runs. FerrySim installs an episode's pair slot there
  (``MuleSupervisor.install_flight_slot``) and observes the missions.
* ``device_model``: ``stub`` keeps the stack's own stub trainer
  (``hermes/processes/device.py``, n_i drawn afresh in [4, 15] at every
  training call), which the parity tests need; ``equal`` replaces each
  device's local training with an equal shard (:func:`equal_shard_trainer`:
  the same noisy update, a constant example count and constant scores), the
  training cells' device model, so every update weighs the same in the merge
  (the spec, other choices 7; critic C5).

A trial is a pure function of its cell, its settings and its hooks: the same
bytes under any ``PYTHONHASHSEED``, whatever ran before it in the process.
The patches (``time.time``, ``threading.Thread`` and the process modules' TCP
link classes) are process-wide while a trial runs, so a process runs one
trial at a time; FerrySim's parallel runs use worker processes (risk R10).
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import hashlib
import json
import math
import subprocess
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from experiments.exp4 import driver as driver_mod
from experiments.ferrysim.reward import EQUAL_SHARD_EXAMPLES
from experiments.runner import Cell
from hermes.mission import LocalTrainResult
from hermes.observability import JsonEventEmitter
from hermes.processes import cluster as cluster_proc
from hermes.processes import device as device_proc
from hermes.processes import mule as mule_proc
from hermes.processes import orchestrator as orchestrator_mod
from hermes.processes.config import (
    cluster_config_from_json,
    device_config_from_json,
    mule_config_from_json,
)
from hermes.processes.orchestrator import MultiProcessOrchestrator
from hermes.transport import DockLink, RFLink, RFLinkError
from hermes.transport.dock_link import DockLinkError, DockLinkTimeout
from hermes.types import DeviceID, MissionPass, MuleID

#: The harness's wall clock (UG4's): a realistic epoch, so a wall stamp that
#: leaks into a simulated field is caught by the mission clock's ceiling
#: (1e9 s), moved only by the harness's own events.
WALL_T0 = 1_700_000_000.0
SOLICIT_DT = 2.0
PUSH_DT = 0.5
UP_DT = 3.0

#: Placeholder ports, standing in for the ones the processes would bind
#: (``tests/golden/_build_topology.py``).
CLUSTER_PORT = 50000
MULE_PORT_BASE = 51000

#: The canonical form's record marker (``tests/golden/_canon.py``).
TYPE_KEY = "__type__"

#: The device models a trial can run (:class:`RoleHooks`). ``stub`` is the
#: stack's own stub trainer; ``equal`` the training cells' equal shards.
DEVICE_MODEL_STUB = "stub"
DEVICE_MODEL_EQUAL = "equal"
DEVICE_MODELS: Tuple[str, ...] = (DEVICE_MODEL_STUB, DEVICE_MODEL_EQUAL)

#: The equal device model's constant training scores: the midpoints of the
#: stub's draws (accuracy and AUC uniform on [0.7, 0.9], loss on [0.1, 0.3]),
#: so a device's utility, and with it its advert, is the stub's on average.
EQUAL_SHARD_ACCURACY = 0.8
EQUAL_SHARD_AUC = 0.8
EQUAL_SHARD_LOSS = 0.2

#: ``mission_completed.plan_wall_s``, the planner's wall time, as a case
#: stores it (UG5's ``WALL_TOKEN``): the one wall time in a plan-mode trace.
WALL_TOKEN = "wall-seconds"

_real_time = time.time
_real_thread = threading.Thread


# --------------------------------------------------------------------------- #
# Helpers copied from tests/golden (critic C7: experiments never imports tests)
# --------------------------------------------------------------------------- #

class SyncThread:
    """``threading.Thread`` while a trial runs: HERMES targets run on ``start()``.

    Copied from ``tests/golden/_host_harness.py``. A target defined in
    ``hermes`` (the mission server's per-device workers, a device's train-ahead
    fit) runs synchronously, so their order is fixed and ``join()`` returns at
    once. Any other target still gets a real thread, so a trial cannot hang on
    it.
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


class _NotLaunched:
    """What the stubbed ``_spawn`` returns instead of a ``Popen``: alive, no process."""

    pid = 0

    def poll(self) -> None:
        return None


class _TimeWithoutSleep:
    """The ``time`` module as the orchestrator sees it here, minus ``sleep``.

    Copied from ``tests/golden/_build_topology.py``: ``start_devices`` sleeps
    0.3 s so the devices can connect, and nothing is launched here.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)

    @staticmethod
    def sleep(_seconds: float) -> None:
        return None


class _NoSubprocess:
    """The ``subprocess`` module as the orchestrator sees it here: no ``Popen``.

    Copied from ``tests/golden/_build_topology.py``. A guard: if the stubs
    below stop applying (say ``_spawn`` is renamed), the trial fails instead
    of launching a process tree.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(subprocess, name)

    @staticmethod
    def Popen(*_args: Any, **_kwargs: Any) -> None:  # noqa: N802 (the module's name)
        raise AssertionError(
            "FerrySim reached subprocess.Popen: the stubs of "
            "MultiProcessOrchestrator._spawn/_wait_for_port no longer apply"
        )


class _StubbedOrchestrator(MultiProcessOrchestrator):
    """The real orchestrator with its process launch and port read-back stubbed.

    Copied from ``tests/golden/_build_topology.py``. Everything ``start_all``
    does to the configs (``expected_mules``, ``seed_devices``,
    ``expected_devices``, the dock and RF ports) and the JSON it writes to its
    run directory is the production code's.
    """

    def __init__(self, topology) -> None:
        for name in ("_spawn", "_wait_for_port"):
            if not callable(getattr(MultiProcessOrchestrator, name, None)):
                raise AssertionError(
                    f"MultiProcessOrchestrator.{name} is gone: update FerrySim's stubs "
                    f"(experiments/ferrysim/inprocess.py)")
        super().__init__(topology)
        self._mule_ports = iter(range(MULE_PORT_BASE, MULE_PORT_BASE + 10_000))

    def _spawn(self, *, name, module, config_path, port_path):  # noqa: D401
        return _NotLaunched(), None

    def _wait_for_port(self, handle, *, timeout):
        if handle.name == "cluster":
            return CLUSTER_PORT
        return next(self._mule_ports)


def _f(x: float) -> str:
    """A float as its exact repr, tagged so it never collides with a string."""
    return "f:" + repr(float(x))


def _array_digest(a: np.ndarray) -> Dict[str, Any]:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("utf-8"))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(a.tobytes())
    return {"__nd__": list(a.shape), "dtype": str(a.dtype), "sha": h.hexdigest()[:20]}


def _canon_key(k: Any) -> str:
    if isinstance(k, enum.Enum):
        return f"e:{type(k).__name__}.{k.name}"
    if isinstance(k, float):
        return _f(k)
    return str(k)


def _sort_key(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def canon(value: Any) -> Any:
    """JSON-able canonical form of ``value`` (copied from ``tests/golden/_canon.py``).

    Floats are stored as ``"f:" + repr(x)`` so a comparison is bit for bit,
    and a dataclass as a record of its fields, so a field added later passes
    the golden comparison and a removed one fails it.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, enum.Enum):
        return f"e:{type(value).__name__}.{value.name}"
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _f(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, np.ndarray):
        return _array_digest(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out: Dict[str, Any] = {TYPE_KEY: type(value).__name__}
        for fld in dataclasses.fields(value):
            out[fld.name] = canon(getattr(value, fld.name))
        return out
    if isinstance(value, Mapping):
        return {str(_canon_key(k)): canon(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canon(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canon(v) for v in value), key=_sort_key)
    raise TypeError(f"no canonical form for {type(value).__name__}: {value!r}")


# --------------------------------------------------------------------------- #
# The device models
# --------------------------------------------------------------------------- #

def device_seed(device_id: str) -> int:
    """A device's training seed, as ``DeviceService`` derives it.

    ``hermes/processes/device.py``: SHA-256 of the device id, first 4 bytes,
    modulo 2**31, so the equal device model's stream starts from the seed the
    stub's would (:func:`equal_shard_trainer` says why the draws then part).
    """
    digest = hashlib.sha256(str(device_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def equal_shard_trainer(seed: int, examples: int = EQUAL_SHARD_EXAMPLES):
    """The equal device model's local training: every update weighs the same.

    The stack's stub (``processes/device.py``, ``_stub_train_factory``) adds the
    same N(0, 0.01) noise to θ, but draws n_i in [4, 15] and the scores afresh
    at every call, so per-device merge weights are noise (critic B4, C5). Here
    n_i is ``examples`` at every call and the scores are constants
    (:data:`EQUAL_SHARD_ACCURACY`, ...): at the training cells' merge
    (``agg:cutoff``, ``value=uniform``, every update of age 0 at K = 1) each
    collected update's raw L3 weight is ``examples``, and a stop's gain is its
    collected count over N. The noise is the stub's law, N(0, 0.01), drawn
    from a stream seeded with the device's own seed (:func:`device_seed`), but
    it is not the stub's sequence: the stub also draws n_i and the three
    scores from that stream after the noise at every call, so from the second
    training call on the two draw different noise. Nothing in a flight reads
    θ (nor, in plan mode, n_i or the scores), so the two models fly alike (a
    test checks it).
    """
    if isinstance(examples, bool) or not isinstance(examples, int) or examples < 1:
        raise ValueError(f"examples must be an int >= 1, got {examples!r}")
    rng = np.random.default_rng(seed)

    def _train(theta, synth):
        delta = [
            w + rng.normal(0.0, 0.01, size=w.shape).astype(w.dtype)
            for w in theta
        ]
        return LocalTrainResult(
            delta_theta=delta,
            num_examples=int(examples),
            accuracy=EQUAL_SHARD_ACCURACY,
            auc=EQUAL_SHARD_AUC,
            loss=EQUAL_SHARD_LOSS,
            theta_after=delta,
        )
    return _train


@dataclasses.dataclass(frozen=True)
class RoleHooks:
    """What FerrySim adds to UG4's in-process trial; the defaults add nothing.

    ``on_mule`` holds callables, each given every ``MuleService`` once all the
    roles are built and before its loop runs, in order. ``device_model`` is
    one of :data:`DEVICE_MODELS`; ``equal_examples`` is the equal model's n_i.
    """

    on_mule: Tuple[Callable[[Any], None], ...] = ()
    device_model: str = DEVICE_MODEL_STUB
    equal_examples: int = EQUAL_SHARD_EXAMPLES

    def __post_init__(self) -> None:
        if self.device_model not in DEVICE_MODELS:
            raise ValueError(f"device_model must be one of {DEVICE_MODELS}, got "
                             f"{self.device_model!r}")
        hooks = tuple(self.on_mule)
        for hook in hooks:
            if not callable(hook):
                raise TypeError(f"on_mule holds callables, got {hook!r}")
        object.__setattr__(self, "on_mule", hooks)


# --------------------------------------------------------------------------- #
# In-process links (UG4's, copied)
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

    A seam that no longer exists fails the trial instead of letting a real
    socket open (a renamed link class would otherwise be built for real, and
    its devices would find no mule: :meth:`World.rf_client`).
    """
    gone = [f"{m.__name__}.{n}" for m, n, _f in _LINK_SEAMS if not hasattr(m, n)]
    if gone:
        raise AssertionError(f"{gone} are gone: update FerrySim's link seams "
                             f"(experiments/ferrysim/inprocess.py)")
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
    placeholder ports of :class:`_StubbedOrchestrator`); it then runs the whole
    trial (:func:`run_roles`) with this trial's ``hooks`` before returning, so
    the driver finds every mule exited. ``cleanup`` keeps the run directory's
    files in ``files`` before removing it.
    """

    def __init__(self, topology, *, python_executable=None, capture_output: bool = False,
                 hooks: Optional[RoleHooks] = None):
        super().__init__(topology)
        self.files: Dict[str, str] = {}
        self.exit_codes: Dict[str, Optional[int]] = {}
        self.hooks = hooks if hooks is not None else RoleHooks()

    def _spawn(self, *, name, module, config_path, port_path):  # noqa: D401
        return _InProcess(name), None

    def start_all(self, *, timeout: float = 30.0) -> None:
        real = (orchestrator_mod.time, orchestrator_mod.subprocess)
        orchestrator_mod.time, orchestrator_mod.subprocess = _TimeWithoutSleep(), _NoSubprocess()
        try:
            super().start_all(timeout=timeout)
        finally:
            orchestrator_mod.time, orchestrator_mod.subprocess = real
        self.exit_codes = run_roles(self, self.hooks)

    def cleanup(self) -> None:
        for path in sorted(self.tmpdir.glob("*.json*")):
            self.files[path.name] = path.read_text(encoding="utf-8")
        super().cleanup()


def run_roles(orch: InProcessOrchestrator,
              hooks: Optional[RoleHooks] = None) -> Dict[str, Optional[int]]:
    """Run the trial's cluster, mules and devices in this process.

    UG4's ``run_roles``, with FerrySim's :class:`RoleHooks`: built in the
    orchestrator's start order from the JSON it wrote, each role with the JSONL
    emitter its process's ``main`` would open; under the ``equal`` device model
    each device's local training is replaced right after its service is built
    (before any training call); once every role is built, each ``on_mule`` hook
    is given each mule's service. Each mule then runs its service loop and its
    shutdown, as its process does. The cluster and the devices are stopped by
    the orchestrator in a real trial; on Windows ``TerminateProcess`` skips
    their ``finally`` blocks, so their emitters are closed here without the
    end-of-run events.
    """
    hooks = hooks if hooks is not None else RoleHooks()
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
                if hooks.device_model == DEVICE_MODEL_EQUAL:
                    svc.client.local_train = equal_shard_trainer(
                        device_seed(dcfg.device_id), hooks.equal_examples)
                world.rfs[int(dcfg.mule_rf_port)].attach(svc.client)
            for svc in mules:
                for hook in hooks.on_mule:
                    hook(svc)
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
def in_process_orchestrator(hooks: Optional[RoleHooks] = None,
                            built: Optional[List[InProcessOrchestrator]] = None
                            ) -> Iterator[List[InProcessOrchestrator]]:
    """The driver's orchestrator is :class:`InProcessOrchestrator` with ``hooks``.

    Yields the list each orchestrator the driver builds is appended to
    (``built`` when given). The orchestrator module's ``subprocess`` refuses
    ``Popen`` meanwhile, so a driver that reached a real orchestrator some
    other way fails at once instead of launching a trial's processes.
    """
    out: List[InProcessOrchestrator] = [] if built is None else built

    def factory(topology, **kwargs: Any) -> InProcessOrchestrator:
        orch = InProcessOrchestrator(topology, hooks=hooks, **kwargs)
        out.append(orch)
        return orch

    real = driver_mod.MultiProcessOrchestrator, orchestrator_mod.subprocess
    driver_mod.MultiProcessOrchestrator = factory
    orchestrator_mod.subprocess = _NoSubprocess()
    try:
        yield out
    finally:
        driver_mod.MultiProcessOrchestrator, orchestrator_mod.subprocess = real


@dataclasses.dataclass
class TrialRun:
    """One in-process trial: the driver's row and the orchestrator's record."""

    row: Dict[str, Any]
    orch: InProcessOrchestrator

    @property
    def files(self) -> Dict[str, str]:
        """The run directory's per-role JSON and JSONL, by file name."""
        return self.orch.files

    @property
    def exit_codes(self) -> Dict[str, Optional[int]]:
        return self.orch.exit_codes

    def events(self, role_file_prefix: str = "mule-") -> List[Dict[str, Any]]:
        """Every event of the files whose names start with the prefix, in file order."""
        out: List[Dict[str, Any]] = []
        for name in sorted(self.files):
            if name.startswith(role_file_prefix) and name.endswith(".jsonl"):
                out += _jsonl(self.files[name])
        return out


class TrialFailure(RuntimeError):
    """An in-process trial whose mule failed, with the reasons its trace gives.

    The mule's service loop turns an exception in a mission into a
    ``mission_failed`` event and a non-zero exit, which the driver raises as
    ``Exp4MuleFailure``; this carries the events' reasons, since the trial's
    files are gone once the driver has cleaned up.
    """

    def __init__(self, message: str, reasons: List[str]) -> None:
        super().__init__(message)
        self.reasons = reasons


def run_trial(driver: "driver_mod.Exp4Driver", cell: Cell, *,
              hooks: Optional[RoleHooks] = None) -> TrialRun:
    """Run one driver trial in this process; its row and its files.

    ``driver.run_trial(cell)`` with the orchestrator replaced for the call
    (:func:`in_process_orchestrator`), so everything the driver does, the row
    included, is the production code's. Raises what the driver raises, but a
    mule that failed (``Exp4MuleFailure``) as :class:`TrialFailure` with the
    reasons of its ``mission_failed`` events.
    """
    with in_process_orchestrator(hooks) as built:
        try:
            row = dict(driver.run_trial(cell))
        except driver_mod.Exp4MuleFailure as exc:
            reasons = [str(e.get("reason")) for orch in built
                       for name, text in sorted(orch.files.items())
                       if name.startswith("mule-") and name.endswith(".jsonl")
                       for e in _jsonl(text) if e.get("event") == "mission_failed"]
            raise TrialFailure(f"{exc}; mission_failed: {reasons}", reasons) from exc
    if len(built) != 1:
        raise AssertionError(f"the driver built {len(built)} orchestrators for one trial")
    return TrialRun(row=row, orch=built[0])


# --------------------------------------------------------------------------- #
# Canonical form of a trial (UG4's, copied; used by FerrySim's determinism
# checks, and comparable with the goldens)
# --------------------------------------------------------------------------- #

def payload(value: Any, kind: str, device_ids: frozenset) -> Any:
    """``value`` in canonical form, every non-empty mapping a record unless its
    keys are all device ids."""
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
    return canon(value)


def _jsonl(text: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def inputs_of(settings: Mapping[str, Any], cell: Cell) -> Dict[str, Any]:
    """A trial's driver settings and cell, as the goldens store them."""
    return {"settings": canon(dict(settings)),
            "cell": {"cell_id": cell.cell_id, "arm": cell.arm, "seed": cell.seed,
                     "trial_index": cell.trial_index, "params": canon(dict(cell.params))}}


def case_of(settings: Mapping[str, Any], cell: Cell, row: Mapping[str, Any],
            orch: InProcessOrchestrator) -> Dict[str, Any]:
    """One trial's canonical record, split into the parts the goldens compare."""
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


def masked_wall(value: Any) -> Any:
    """``plan_wall_s`` as a case stores it (:data:`WALL_TOKEN`), UG5's rule."""
    if isinstance(value, str) and value.startswith("f:"):
        try:
            x = float(value[2:])
        except ValueError:
            return value
        if x == x and x != math.inf and x >= 0.0:
            return WALL_TOKEN
    return value


#: ``mission_completed``'s per-decision wall times (Exp 5 addendum, Study 5.11
#: (a)): lists of ``{"decide_s", "mask_s"}``, one per pair decision or E3 call,
#: each value masked as ``plan_wall_s`` is.
DECISION_WALL_FIELDS = ("pass_1_pairs_wall", "pass_1_e3_wall")


def mask_wall_times(case: Dict[str, Any]) -> Dict[str, Any]:
    """``case`` with every wall time of ``mission_completed`` masked, in place.

    The planner's wall time (``plan_wall_s``) and the flight clock's
    per-decision wall times (:data:`DECISION_WALL_FIELDS`) are the only wall
    times in a plan-mode trial run in process (the envelope stamps and
    durations are the harness clock's), so a case masked this way is the same
    for the same trial. The keys stay, so a field that goes missing still
    differs.
    """
    for events in case["mission_completed"].values():
        for e in events:
            if "plan_wall_s" in e:
                e["plan_wall_s"] = masked_wall(e["plan_wall_s"])
            for name in DECISION_WALL_FIELDS:
                entries = e.get(name)
                if isinstance(entries, list):
                    e[name] = [
                        {k: masked_wall(v) for k, v in entry.items()}
                        if isinstance(entry, dict) else entry
                        for entry in entries
                    ]
    return case


def trial_case(cell: Cell, run: TrialRun,
               settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """``run``'s canonical case with the planner's wall time masked.

    ``settings`` are the driver settings the case records as its inputs
    (default: none, ``{}``); the goldens record the settings the trial was
    built from.
    """
    return mask_wall_times(case_of(dict(settings or {}), cell, run.row, run.orch))


__all__ = [
    "CLUSTER_PORT",
    "DECISION_WALL_FIELDS",
    "DEVICE_MODELS",
    "DEVICE_MODEL_EQUAL",
    "DEVICE_MODEL_STUB",
    "EQUAL_SHARD_EXAMPLES",
    "InProcessOrchestrator",
    "RoleHooks",
    "SyncThread",
    "TrialFailure",
    "TrialRun",
    "WALL_TOKEN",
    "World",
    "canon",
    "case_of",
    "device_seed",
    "equal_shard_trainer",
    "in_process_orchestrator",
    "inputs_of",
    "mask_wall_times",
    "masked_wall",
    "patched",
    "payload",
    "run_roles",
    "run_trial",
    "trial_case",
]
