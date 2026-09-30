"""Capture of the Exp 4 topology, the per-role configs and the driver's rows.

Phase 3 (unit U7) edits the process configs, the topology builder and the
driver; in legacy mode the configs may only gain default-valued keys (design
section 5.2). Pinned here:

* ``build_exp4_topology`` over a grid of cells (device count, RF range,
  realism, one to three mules, the CARP slice assignment of arm D4): every
  config, and the JSON each role is started with. That JSON is written by a
  real ``MultiProcessOrchestrator.start_all`` (``expected_mules``,
  ``seed_devices``, ``expected_devices`` and the ports filled in) and read
  back from its run directory; only the spawn and the port read-back are
  stubbed, so the ports are fixed placeholders instead of the ones the
  processes would bind, and nothing is launched.
* ``Exp4Driver.run_trial`` on stub cells, with the orchestrator replaced by a
  fake that starts nothing and an empty run directory: the topology the
  driver hands the orchestrator and the row it returns, provenance columns
  included, and the per-role JSON the real orchestrator writes for that
  topology (as above). Nothing is spawned.
* The kept traces in ``results/``: their recorded per-role configs (written
  by the orchestrator when they ran) are re-derived from their seeds by the
  driver and today's orchestrator (positions, reliabilities, seeds, loss
  schedules, RF priors), and old per-role JSON still loads. There the
  orchestrator's writes are taken in memory, for speed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from experiments.exp4 import driver as driver_mod
from experiments.exp4.driver import Exp4Driver, d4_slice_assignment
from experiments.exp4.model_task import device_reliabilities
from experiments.exp4.topology_builder import angular_slices, build_exp4_topology
from experiments.runner import Cell
from hermes.processes import orchestrator as orchestrator_mod
from hermes.processes.config import (
    ClusterConfig,
    DeviceConfig,
    MuleConfig,
    TopologyConfig,
    cluster_config_from_json,
    device_config_from_json,
    mule_config_from_json,
)
from hermes.processes.orchestrator import MultiProcessOrchestrator

from tests.golden._build_channel import parse_trace_dir
from tests.golden._canon import record

REPO = Path(__file__).resolve().parents[2]
#: Placeholder ports, standing in for the ones the processes would bind.
CLUSTER_PORT = 50000
MULE_PORT_BASE = 51000


# --------------------------------------------------------------------------- #
# Per-role JSON, as the orchestrator writes it
# --------------------------------------------------------------------------- #

class _NotLaunched:
    """What the stubbed ``_spawn`` returns instead of a ``Popen``: alive, no process."""

    pid = 0

    def poll(self) -> None:
        return None


class _TimeWithoutSleep:
    """The ``time`` module as the orchestrator sees it here, minus ``sleep``.

    ``start_devices`` sleeps 0.3 s so the devices can connect; nothing is
    launched here, and the kept-trace check starts about 500 topologies.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)

    @staticmethod
    def sleep(_seconds: float) -> None:
        return None


class _NoSubprocess:
    """The ``subprocess`` module as the orchestrator sees it here: no ``Popen``.

    A guard: if the stubs below stop applying (say ``_spawn`` is renamed),
    the harness fails instead of launching a process tree.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(subprocess, name)

    @staticmethod
    def Popen(*_args: Any, **_kwargs: Any) -> None:  # noqa: N802 (the module's name)
        raise AssertionError(
            "the golden harness reached subprocess.Popen: the stubs of "
            "MultiProcessOrchestrator._spawn/_wait_for_port no longer apply"
        )


class _StubbedOrchestrator(MultiProcessOrchestrator):
    """The real orchestrator with its process launch and port read-back stubbed.

    Everything ``start_all`` does to the configs (``expected_mules``,
    ``seed_devices``, ``expected_devices``, the dock and RF ports) and the JSON
    it writes to its run directory is the production code's.
    """

    def __init__(self, topology: TopologyConfig) -> None:
        for name in ("_spawn", "_wait_for_port"):
            if not callable(getattr(MultiProcessOrchestrator, name, None)):
                raise AssertionError(
                    f"MultiProcessOrchestrator.{name} is gone: update the golden "
                    f"harness's stubs (tests/golden/_build_topology.py)")
        super().__init__(topology)
        self._mule_ports = iter(range(MULE_PORT_BASE, MULE_PORT_BASE + 10_000))

    def _spawn(self, *, name, module, config_path, port_path):  # noqa: D401
        return _NotLaunched(), None

    def _wait_for_port(self, handle, *, timeout):
        if handle.name == "cluster":
            return CLUSTER_PORT
        return next(self._mule_ports)


@contextlib.contextmanager
def _writes_kept_in_memory(run_dir: Path, captured: Dict[str, str]) -> Iterator[None]:
    """``Path.write_text`` into ``run_dir`` lands in ``captured``, not on disk.

    For the kept-trace check, which starts about 500 topologies: a small file
    write costs a few milliseconds here. The text captured is the text the
    orchestrator writes; a write by any other means still goes to disk, and
    :func:`role_configs` reads it from there.
    """
    real = Path.write_text

    def write_text(self, data, encoding=None, errors=None, newline=None):
        if self.parent == run_dir:
            captured[self.name] = data
            return len(data)
        return real(self, data, encoding=encoding, errors=errors, newline=newline)

    Path.write_text = write_text  # type: ignore[method-assign]
    try:
        yield
    finally:
        Path.write_text = real  # type: ignore[method-assign]


def role_configs(topo: TopologyConfig, *, on_disk: bool = True) -> Dict[str, Any]:
    """The JSON each process is started with, ports replaced by placeholders.

    ``MultiProcessOrchestrator.start_all`` writes ``cluster.json``,
    ``mule-<id>.json`` and ``device-<id>.json`` to its run directory; they
    are read back from there (``on_disk``), or taken as written
    (``on_disk=False``, see :func:`_writes_kept_in_memory`). The cluster's port
    is ``CLUSTER_PORT`` and the mules' are ``MULE_PORT_BASE + k`` in start
    order.
    """
    orch = _StubbedOrchestrator(topo)
    run_dir = orch.tmpdir
    captured: Dict[str, str] = {}
    real = (orchestrator_mod.time, orchestrator_mod.subprocess)
    orchestrator_mod.time, orchestrator_mod.subprocess = _TimeWithoutSleep(), _NoSubprocess()
    try:
        with (contextlib.nullcontext() if on_disk
              else _writes_kept_in_memory(run_dir, captured)):
            orch.start_all()

        def read(name: str) -> Dict[str, Any]:
            text = captured.get(name)
            if text is None:
                text = (run_dir / name).read_text(encoding="utf-8")
            return json.loads(text)

        return {
            "cluster": read("cluster.json"),
            "mules": {m.mule_id: read(f"mule-{m.mule_id}.json") for m in topo.mules},
            "devices": {d.device_id: read(f"device-{d.device_id}.json") for d in topo.devices},
        }
    finally:
        orchestrator_mod.time, orchestrator_mod.subprocess = real
        orch.cleanup()


def topology_record(topo: TopologyConfig) -> Dict[str, Any]:
    """The topology's own configs, and the per-role JSON built from them."""
    roles = role_configs(topo)
    return {
        "device_to_mule": dict(topo.device_to_mule),
        "cluster": record("ClusterConfig", roles["cluster"]),
        "mules": {k: record("MuleConfig", v) for k, v in roles["mules"].items()},
        "devices": {k: record("DeviceConfig", v) for k, v in roles["devices"].items()},
        # The topology JSON round-trips (the orchestrator's input form).
        "round_trip": TopologyConfig.from_json(topo.to_json()).to_json() == topo.to_json(),
    }


# --------------------------------------------------------------------------- #
# The builder grid
# --------------------------------------------------------------------------- #

def _realism(seed: int, n: int, regime: str) -> Dict[str, Any]:
    """The driver's realism kwargs (``Exp4Driver.run_trial``), for the builder."""
    return dict(
        device_reliability=True,
        reliabilities=device_reliabilities(seed, n),
        world_radius_m=100.0,
        field_radius_m=100.0,
        backhaul_loss_pct=2.0 if regime == "jittery" else 0.0,
        backhaul_rng_seed=seed ^ 0x0BACC0DE,
    )


#: (seed, N, RF range, realism, mule counts) of the grid. K = 1 is the
#: recorded single-mule topology; K > 1 is built with the angular split and
#: with D4's CARP split, at quorum K (agg:plain), and once at quorum 1
#: (agg:cutoff, asynchronous).
GRID = (
    (7, 1, 60.0, False, (1,)),
    (7, 1, 60.0, True, (1,)),
    (7, 2, 30.0, True, (1, 2)),
    (7, 6, 60.0, False, (1,)),
    (7, 6, 60.0, True, (1, 2, 3)),
    (7, 6, 30.0, True, (1,)),
    (7, 13, 60.0, True, (1, 3)),
    (2191267877, 6, 60.0, False, (1,)),
    (2191267877, 6, 60.0, True, (1, 2)),
    (2191267877, 13, 60.0, True, (1, 3)),
)


def builder_cells() -> Iterator[Tuple[str, Dict[str, Any], Optional[str]]]:
    """(name, build_exp4_topology kwargs, slice rule) for every grid cell."""
    for seed, n, rrf, realism, mule_counts in GRID:
        base = dict(n_devices=n, rf_range_m=rrf, n_missions=4, seed=seed)
        if realism:
            base.update(_realism(seed, n, "jittery"))
        tag = f"seed={seed}|N={n}|rrf={rrf}|realism={int(realism)}"
        for k in mule_counts:
            if k == 1:
                yield f"{tag}|K=1", dict(base), None
                continue
            plain = dict(base, n_mules=k, min_participation=k, dock_on_empty=True,
                         down_wait_s=120.0)
            yield f"{tag}|K={k}|plain|angular", plain, None
            yield f"{tag}|K={k}|plain|carp", plain, "carp"
            if k == 3:
                cutoff = dict(base, n_mules=3, aggregation="agg:cutoff",
                              aggregation_params={"a_max": 2})
                yield f"{tag}|K=3|cutoff|angular", cutoff, None
    # The builder's own defaults (spread, TTL, synth batch), and its options.
    yield "options", dict(
        n_devices=4, rf_range_m=45.0, n_missions=2, seed=3, spread_m=12.5,
        session_ttl_s=7.0, synth_batch_size=3, mission_budget_s=33.0,
        contact_policy="fedcs", fedcs_value="devices", mission_window_adaptation=True,
        mission_window_history=3, mission_window_target=0.6, mission_window_gain=1.5,
        mission_window_max_scale=2.5, fedprox_rho=0.01, pass_2_budget=True,
        deadline_law="multiplicative", deadline_params={"beta_on": 0.9},
        miss_priority=True, use_rl_selector=False, rf_prior_snr_db=11.25,
        backhaul_loss_schedule=[0.1, 0.25], cluster_id="c-x", mule_id="m-x",
        whittle_variant="literal", whittle_weights="uniform",
    ), None


def build_builder_cases() -> Dict[str, Any]:
    cases: Dict[str, Any] = {}
    for name, kwargs, rule in builder_cells():
        topo = build_exp4_topology(**kwargs)
        extra: Dict[str, Any] = {}
        if rule == "carp":
            assignment = d4_slice_assignment(topo.devices, kwargs["n_mules"], kwargs["seed"])
            extra["slice_assignment"] = {str(i): k for i, k in sorted(assignment.items())}
            topo = build_exp4_topology(**kwargs, slice_assignment=assignment)
        elif kwargs.get("n_mules", 1) > 1:
            extra["angular_slices"] = {
                str(i): k for i, k in sorted(angular_slices(
                    [(d.position[0], d.position[1]) for d in topo.devices],
                    kwargs["n_mules"]).items())
            }
        cases[f"builder:{name}"] = {**topology_record(topo), **extra}
    return cases


# --------------------------------------------------------------------------- #
# The driver on stub cells
# --------------------------------------------------------------------------- #

class FakeOrchestrator:
    """Stands in for ``MultiProcessOrchestrator``: starts nothing.

    Every mule has exited at once, the run directory is empty, so the driver
    folds a zero observation into its row. The topology it was given is kept.
    """

    last: Optional["FakeOrchestrator"] = None

    def __init__(self, topology, *, capture_output: bool = False, python_executable=None):
        topology.validate()
        self.topology = topology
        self._tmpdir = Path(tempfile.mkdtemp(prefix="golden_orch_"))
        FakeOrchestrator.last = self

    @property
    def tmpdir(self) -> Path:
        return self._tmpdir

    @property
    def mule_handles(self) -> Dict[str, Any]:
        return {}

    def start_all(self, *, timeout: float = 30.0) -> None:
        return None

    def shutdown_all(self, *, timeout: float = 10.0, cleanup_tmpdir: bool = True) -> None:
        return None

    def cleanup(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)


@contextlib.contextmanager
def fake_orchestrator() -> Iterator[None]:
    real = driver_mod.MultiProcessOrchestrator
    driver_mod.MultiProcessOrchestrator = FakeOrchestrator
    try:
        yield
    finally:
        driver_mod.MultiProcessOrchestrator = real


def run_stub_trial(driver: Exp4Driver, cell: Cell) -> Tuple[Dict[str, Any], TopologyConfig]:
    """(row, topology) of one trial, with nothing spawned."""
    with fake_orchestrator():
        FakeOrchestrator.last = None
        row = dict(driver.run_trial(cell))
        topo = FakeOrchestrator.last.topology
    return row, topo


def _cell(arm: str, seed: int = 7, **params) -> Cell:
    p = {"N": 6, "rrf": 60.0, "n_missions": 4, "regime": "jittery"}
    p.update(params)
    cell_id = "|".join(f"{k}={v}" for k, v in sorted(p.items()))
    return Cell(cell_id=cell_id, arm=arm, trial_index=0, seed=seed, params=p)


DRIVER_CASES: Tuple[Tuple[str, Dict[str, Any], Cell], ...] = (
    ("H1-default-clean", dict(), _cell("H1", regime="clean")),
    ("H1-realism-jittery", dict(realism=True), _cell("H1")),
    ("H2-realism", dict(realism=True), _cell("H2")),
    ("H3-realism-l1", dict(realism=True, l1_channel=True), _cell("H3", seed=2191267877)),
    ("H2-realism-l1", dict(realism=True, l1_channel=True), _cell("H2", seed=2191267877)),
    ("D1-budget", dict(realism=True, mission_budget_s=60.0), _cell("D1")),
    ("D3-whittle", dict(realism=True, whittle_variant="literal"), _cell("D3")),
    ("D4-one-mule", dict(realism=True, mission_budget_s=60.0), _cell("D4")),
    ("D5-fedcs-devices", dict(realism=True, mission_budget_s=45.0, fedcs_value="devices"),
     _cell("D5", regime="clean")),
    ("H1-phase1-options", dict(
        realism=True, mission_budget_s=90.0, mission_window_adaptation=True,
        pass_2_budget=True, aggregation="agg:cutoff", aggregation_params={"a_max": 3},
        deadline_law="multiplicative", deadline_params={"beta_timeout": 1.4},
        miss_priority=True, fedprox_rho=0.05), _cell("H1", n_missions=3)),
    ("D4-two-mules-quorum-2", dict(realism=True, n_mules=2, min_participation=2),
     _cell("D4", N=7)),
    ("H1-three-mules-async", dict(n_mules=3, aggregation="agg:cutoff"), _cell("H1", N=9)),
)


def build_driver_cases() -> Dict[str, Any]:
    cases: Dict[str, Any] = {}
    for name, kwargs, cell in DRIVER_CASES:
        driver = Exp4Driver(**kwargs)
        row, topo = run_stub_trial(driver, cell)
        cases[f"driver:{name}"] = {
            "row": record("Exp4Row", row),
            "topology": topology_record(topo),
        }
    return cases


def build_cases() -> Dict[str, Any]:
    cases = build_builder_cases()
    cases.update(build_driver_cases())
    return cases


# --------------------------------------------------------------------------- #
# Kept traces: re-derive their recorded per-role configs
# --------------------------------------------------------------------------- #

RECORDED_SETS = (
    "results/exp4_matrix/A_clean_traces",
    "results/exp4_matrix/A_jittery_traces",
    "results/exp4_matrix/C_traces",
    "results/exp4_matrix/C2_traces",
    "results/exp4_s3c/off_traces",
    "results/exp4_s3c/on_traces",
    "results/exp4_sota/b60_traces",
    "results/exp4_sota/pilot_traces",
)
#: Driver settings a study passed that its recorded configs do not show. The
#: S3c pilot scattered devices over 150 m (``run_s3c_pilot.sh``:
#: ``--h1-field-radius-m 150``); the others used the driver defaults.
STUDY_OVERRIDES = {
    "results/exp4_s3c/off_traces": {"h1_field_radius_m": 150.0},
    "results/exp4_s3c/on_traces": {"h1_field_radius_m": 150.0},
}
#: Recorded keys that are per-process or per-trial and not derived from the seed.
VOLATILE = {
    "cluster": {"dock_port", "init_theta_path", "eval_test_path", "input_dim"},
    "mule": {"dock_port", "rf_port"},
    "device": {"mule_rf_port", "train_shard_path", "input_dim"},
}


#: Arms the stub driver refuses: D2 (Oort) needs the real model, so its kept
#: trials cannot be re-derived without loading CICIoT. Its topology is D1's
#: with ``contact_policy="oort"``.
STUB_REFUSED_ARMS = ("D2",)


def recorded_trace_dirs(per_cell: Optional[int] = None) -> List[Path]:
    out: List[Path] = []
    for rel in RECORDED_SETS:
        root = REPO / rel
        if not root.is_dir():
            continue
        seen: Dict[Tuple[str, str], int] = {}
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            info = parse_trace_dir(d.name)
            if info is None or info["arm"] in STUB_REFUSED_ARMS:
                continue
            key = (info["cell"], info["arm"])
            seen[key] = seen.get(key, 0) + 1
            if per_cell is None or seen[key] <= per_cell:
                out.append(d)
    return out


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default(cls, name: str) -> Any:
    fld = {f.name: f for f in dataclasses.fields(cls)}[name]
    if fld.default is not dataclasses.MISSING:
        return fld.default
    if fld.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return fld.default_factory()  # type: ignore[misc]
    raise KeyError(name)


def _means_default(cls, name: str, value: Any, config: Dict[str, Any]) -> bool:
    """Whether an added key holds (the meaning of) its dataclass default.

    The parameter dicts are compared by what they configure: at afa9526 the
    driver writes ``aggregation_params`` as the rule's full default parameters
    (``AggregationSpec.to_params()``), not ``{}``, which builds the same spec.
    """
    if name == "aggregation_params":
        from hermes.mission.aggregation_rules import AggregationSpec
        rule = config.get("aggregation")
        return (AggregationSpec.from_config(rule, value)
                == AggregationSpec.from_config(rule, _default(cls, name)))
    if name == "deadline_params":
        from hermes.scheduler.stages.s3_deadline import DeadlineLaw
        law = config.get("deadline_law")
        return (DeadlineLaw.from_config(law, value)
                == DeadlineLaw.from_config(law, _default(cls, name)))
    return value == json.loads(json.dumps(_default(cls, name)))


def driver_for_trace(trace_dir: Path) -> Tuple[Exp4Driver, Cell]:
    """The stub driver and cell that re-create a kept trial's topology.

    The study's settings are read back from the recorded configs: realism from
    the devices' contact reliability, the L1 channel from the cluster's loss
    schedule, and the budget, policy, selector and window adaptation from the
    mule. The recorded runs used the real model; that only changes the
    per-trial paths and ``input_dim``, which are not compared.
    """
    info = parse_trace_dir(trace_dir.name)
    cluster = _load(trace_dir / "cluster.json")
    mule = _load(sorted(trace_dir.glob("mule-*.json"))[0])
    dev0 = _load(sorted(trace_dir.glob("device-*.json"))[0])
    p = info["params"]
    regime = p["regime"]
    kwargs: Dict[str, Any] = dict(
        realism=dev0.get("contact_reliability") is not None,
        l1_channel=cluster.get("backhaul_loss_schedule") is not None,
        mission_budget_s=mule.get("mission_budget_s"),
        mission_window_adaptation=bool(mule.get("mission_window_adaptation", False)),
        selector_weights_path=mule.get("selector_weights_path"),
    )
    pct = float(cluster.get("backhaul_loss_pct", 0.0))
    if regime == "jittery":
        kwargs["jittery_backhaul_loss_pct"] = pct
    else:
        kwargs["clean_backhaul_loss_pct"] = pct
    for k in ("mission_window_history", "mission_window_target", "mission_window_gain",
              "mission_window_max_scale"):
        if k in mule:
            kwargs[k] = mule[k]
    study = trace_dir.parent.relative_to(REPO).as_posix()
    kwargs.update(STUDY_OVERRIDES.get(study, {}))
    params = {
        "N": int(p["N"]), "rrf": float(p["rrf"]), "n_missions": int(p["n_missions"]),
        "regime": regime, "dead_zone": float(p["dead_zone"]),
        "link_quality": float(p["link_quality"]),
    }
    cell = Cell(cell_id=info["cell"], arm=info["arm"], trial_index=info["trial"],
                seed=info["seed"], params=params)
    return Exp4Driver(**kwargs), cell


def compare_recorded(trace_dir: Path) -> List[str]:
    """Mismatches between a kept trial's recorded configs and today's.

    Every recorded key must match (volatile ones aside); every key added since
    must hold its dataclass default, so old JSON and new JSON mean the same.
    Old per-role JSON must also still load.
    """
    problems: List[str] = []
    driver, cell = driver_for_trace(trace_dir)
    _row, topo = run_stub_trial(driver, cell)
    roles = role_configs(topo, on_disk=False)

    def check(kind: str, cls, recorded: Dict[str, Any], current: Dict[str, Any], where: str):
        for k, v in recorded.items():
            if k in VOLATILE[kind]:
                continue
            if k not in current:
                problems.append(f"{where}: recorded key {k!r} is gone")
            elif current[k] != v:
                problems.append(f"{where}.{k}: recorded {v!r} != derived {current[k]!r}")
        for k in sorted(set(current) - set(recorded)):
            if not _means_default(cls, k, current[k], current):
                problems.append(f"{where}.{k}: added key is not its default ({current[k]!r})")

    rec_cluster = _load(trace_dir / "cluster.json")
    check("cluster", ClusterConfig, rec_cluster, roles["cluster"], "cluster")
    cluster_config_from_json(json.dumps(rec_cluster))
    for path in sorted(trace_dir.glob("mule-*.json")):
        rec = _load(path)
        cur = roles["mules"].get(rec["mule_id"])
        if cur is None:
            problems.append(f"{path.name}: no mule {rec['mule_id']!r} in the derived topology")
            continue
        check("mule", MuleConfig, rec, cur, path.name)
        mule_config_from_json(json.dumps(rec))
    for path in sorted(trace_dir.glob("device-*.json")):
        rec = _load(path)
        cur = roles["devices"].get(rec["device_id"])
        if cur is None:
            problems.append(f"{path.name}: no device {rec['device_id']!r} in the derived topology")
            continue
        check("device", DeviceConfig, rec, cur, path.name)
        device_config_from_json(json.dumps(rec))
    return problems
