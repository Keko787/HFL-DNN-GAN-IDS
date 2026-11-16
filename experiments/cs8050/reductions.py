"""CMP_SC 8050 project, Phase 2: the two conversions of the hardness proof T1.

Each conversion turns an instance of a known NP-complete problem into a
planning instance (:class:`~experiments.cs8050.problem.Instance`) and a target
W, and comes with the map that turns a certificate of the known problem into a
valid plan worth W. The proofs are in ``DeveloperDocs/cs8050/report/t1_hardness.tex``:

* :func:`from_grid_graph` is Lemma 1 (routing). A grid graph has a Hamiltonian
  cycle exactly when the instance has a valid plan serving every device.
  :func:`plan_from_cycle` is the yes-to-yes map.
* :func:`from_knapsack` is Lemma 2 (packing). A knapsack instance has a subset
  of size at most K and value at least V exactly when the instance has a valid
  plan worth at least V. :func:`plan_from_items` is the yes-to-yes map.

Both conversions give the same instance for exact and for rounded travel
times (decision 1.6), up to the ``travel_quantum_s`` field, and both proofs
hold for the two. The no-to-no direction is checked on small cases by the
exact solver in step 3.7; the tests here check the yes-to-yes maps with
:func:`~experiments.cs8050.problem.check_plan`.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

from experiments.cs8050.problem import Device, Instance, Plan, Setting, Stop

__all__ = [
    "GridPoint",
    "grid_device_id", "from_grid_graph", "plan_from_cycle", "is_grid_cycle",
    "item_device_id", "from_knapsack", "plan_from_items",
]

GridPoint = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Lemma 1: Hamiltonian cycle in a grid graph
# --------------------------------------------------------------------------- #

def grid_device_id(v: GridPoint) -> str:
    """The id of the device placed on grid point ``v``."""
    return f"g{v[0]}_{v[1]}"


def _grid_points(vertices: Iterable[GridPoint]) -> Tuple[GridPoint, ...]:
    out = []
    for v in vertices:
        x, y = v
        if not (isinstance(x, int) and isinstance(y, int)) or isinstance(x, bool) or isinstance(y, bool):
            raise TypeError(f"grid points must be integer pairs, got {v!r}")
        out.append((x, y))
    if len(set(out)) != len(out):
        raise ValueError("grid points must be distinct")
    if not out:
        raise ValueError("a grid graph needs at least one point")
    return tuple(out)


def from_grid_graph(
    vertices: Sequence[GridPoint],
    *,
    quantum_s: Optional[float] = None,
    size_bytes: float = 0.0,
    settings: Optional[Sequence[Setting]] = None,
    spacing_m: float = 1.0,
) -> Tuple[Instance, float]:
    """Lemma 1's instance for the grid graph on ``vertices``, and its target W = n.

    One device of importance 1 on each grid point, ``spacing_m`` metres apart,
    each sending ``size_bytes`` with no deadline. The sites are the device
    positions and the dock is the first point. Every setting's reach must be
    below the spacing, so a device can only be served at its own point. One
    grid step takes ``leg`` seconds: the quantum when travel is rounded, else
    1 s. The time limit is n legs plus the n sends at the fastest setting's
    top rate, and there is no upload.

    The defaults are the proof's instance: unit spacing, no data, one setting
    of reach 1/2. ``size_bytes`` and ``settings`` show the hardness holds for
    equal data sizes and for FeRRy's own three settings (spacing above 232 m).
    """
    pts = _grid_points(vertices)
    spacing = float(spacing_m)
    if not spacing > 0:
        raise ValueError(f"spacing_m must be > 0, got {spacing_m!r}")
    if settings is None:
        settings = (Setting.constant("unit", spacing / 2, 1.0),)
    settings = tuple(settings)
    for s in settings:
        if s.reach_m >= spacing:
            raise ValueError(
                f"setting {s.name!r} reaches {s.reach_m!r} m, not below the {spacing!r} m spacing"
            )
    leg = 1.0 if quantum_s is None else float(quantum_s)
    top_rate = max(s.rate_steps[0][1] for s in settings)
    n = len(pts)
    devices = tuple(
        Device(grid_device_id(v), (spacing * v[0], spacing * v[1]), weight=1.0,
               size_bytes=size_bytes)
        for v in pts
    )
    instance = Instance(
        devices=devices,
        settings=settings,
        dock=devices[0].position,
        speed_m_s=spacing / leg,
        time_limit_s=n * leg + n * size_bytes / top_rate,
        upload_s=0.0,
        sites=tuple(d.position for d in devices),
        travel_quantum_s=quantum_s,
    )
    return instance, float(n)


def is_grid_cycle(vertices: Sequence[GridPoint], cycle: Sequence[GridPoint]) -> bool:
    """True when ``cycle`` lists every grid point once and each step is 1 apart."""
    pts = set(_grid_points(vertices))
    cyc = tuple(cycle)
    if len(cyc) < 3 or len(cyc) != len(pts) or set(cyc) != pts:
        return False
    return all(
        abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        for a, b in zip(cyc, cyc[1:] + cyc[:1])
    )


def plan_from_cycle(instance: Instance, cycle: Sequence[GridPoint], spacing_m: float = 1.0) -> Plan:
    """The yes-to-yes map of Lemma 1: a Hamiltonian cycle becomes a plan worth n.

    The cycle is turned to start at the dock's point. The first stop serves
    the dock's own device without flying, then the drone follows the cycle,
    serving each point's device there, and flies the last step home. It uses
    the setting with the fastest top rate, the one the time limit is set by.
    """
    cyc = tuple(cycle)
    spacing = float(spacing_m)
    dock = (round(instance.dock[0] / spacing), round(instance.dock[1] / spacing))
    if dock not in cyc:
        raise ValueError("the cycle does not pass through the dock's point")
    i = cyc.index(dock)
    cyc = cyc[i:] + cyc[:i]
    setting = max(instance.settings, key=lambda s: s.rate_steps[0][1])
    stops = tuple(
        Stop((spacing * v[0], spacing * v[1]), (grid_device_id(v),)) for v in cyc
    )
    return Plan(setting.name, stops)


# --------------------------------------------------------------------------- #
# Lemma 2: knapsack
# --------------------------------------------------------------------------- #

def item_device_id(i: int) -> str:
    """The id of the device standing for knapsack item ``i``."""
    return f"item{i}"


def from_knapsack(
    sizes: Sequence[float],
    values: Sequence[float],
    capacity: float,
    target: float,
    *,
    quantum_s: Optional[float] = None,
) -> Tuple[Instance, float]:
    """Lemma 2's instance for a knapsack instance, and its target W = V.

    Item i becomes a device of importance c_i that sends a_i bytes at one byte
    per second, so its send time is its size. The devices stand on a short
    line from the dock, all within reach of the one site, the dock itself, so
    the drone never flies. No deadlines, no upload, time limit K.
    """
    a = tuple(float(x) for x in sizes)
    c = tuple(float(x) for x in values)
    n = len(a)
    if n == 0 or len(c) != n:
        raise ValueError("sizes and values must be non-empty and of equal length")
    if any(x < 0 or not math.isfinite(x) for x in a):
        raise ValueError(f"sizes must be finite and >= 0, got {sizes!r}")
    if any(x <= 0 or not math.isfinite(x) for x in c):
        raise ValueError(f"values must be finite and > 0, got {values!r}")
    reach = 1.0
    devices = tuple(
        Device(item_device_id(i), (reach * i / (2 * n), 0.0), weight=c[i], size_bytes=a[i])
        for i in range(n)
    )
    instance = Instance(
        devices=devices,
        settings=(Setting.constant("one", reach, 1.0),),
        dock=(0.0, 0.0),
        speed_m_s=1.0,
        time_limit_s=float(capacity),
        upload_s=0.0,
        sites=((0.0, 0.0),),
        travel_quantum_s=quantum_s,
    )
    return instance, float(target)


def plan_from_items(instance: Instance, items: Iterable[int]) -> Plan:
    """The yes-to-yes map of Lemma 2: a set of items becomes one stop at the dock."""
    chosen = tuple(sorted(set(items)))
    setting = instance.settings[0].name
    if not chosen:
        return Plan(setting)
    return Plan(setting, (Stop(instance.dock, tuple(item_device_id(i) for i in chosen)),))
