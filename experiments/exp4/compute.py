"""Device compute on the simulated clock (Exp 5 addendum, Study 5.12).

Every recorded trial charged a device's local fit nothing on the simulated
clock: an update was always ready when the mule came. Study 5.12 sweeps the
local training time per device: none (today), a seeded spread, and a share of
stragglers at a multiple. This module draws each device's fit time ``T_j`` in
simulated seconds from four settings (``train_time_params``):

* ``median_s``: the median fit time (>= 0; 0 is the sweep's "none" level,
  which flies as a recorded run and records the fits and uplinks for the
  device energy);
* ``sigma``: the log-normal spread, ``T_j = median_s * exp(sigma * z_j)`` with
  ``z_j`` a standard normal (0, the default: every device the median);
* ``straggler_share`` and ``straggler_factor``: exactly
  ``round(share * N)`` devices (half up) are stragglers, whose time is
  multiplied by the factor (>= 1). The plan's "20 % stragglers at 5x" is
  ``0.2`` and ``5.0``; the defaults ``0`` and ``1`` are none.

Each device's ``z_j`` and its straggler rank come from its own keyed stream
(the trial seed and the device id), so every arm of a trial, which shares the
seed, holds the same times (paired by construction), and a device's time does
not depend on the other devices'. The mule times the fits with them
(``hermes/mule/fit_clock.py``). numpy only.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "TRAIN_TIME_DEFAULTS",
    "check_train_time_params",
    "device_train_times",
    "stragglers",
]

#: The settings and their defaults; ``median_s`` has none (it turns the model on).
TRAIN_TIME_DEFAULTS: Dict[str, float] = {
    "sigma": 0.0, "straggler_share": 0.0, "straggler_factor": 1.0,
}


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"train time {name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"train time {name} must be finite, got {value!r}")
    return value


def check_train_time_params(params: Optional[Mapping[str, float]]) -> Optional[Dict[str, float]]:
    """The settings validated and completed with their defaults; None stays None.

    Refused: an unknown key, no ``median_s``, ``median_s < 0``, ``sigma < 0``,
    a share outside [0, 1] and a factor below 1.
    """
    if params is None:
        return None
    if not isinstance(params, Mapping):
        raise TypeError(f"train_time_params must be a mapping, got {type(params).__name__}")
    unknown = sorted(set(params) - {"median_s", *TRAIN_TIME_DEFAULTS})
    if unknown:
        raise ValueError(f"train_time_params: unknown keys {unknown} (known: median_s, "
                         f"{', '.join(TRAIN_TIME_DEFAULTS)})")
    if "median_s" not in params:
        raise ValueError("train_time_params needs median_s, the median fit time in s")
    out = {"median_s": _number(params["median_s"], "median_s")}
    for name, default in TRAIN_TIME_DEFAULTS.items():
        out[name] = _number(params.get(name, default), name)
    if out["median_s"] < 0.0:
        raise ValueError(f"train time median_s must be >= 0 s, got {out['median_s']!r}")
    if out["sigma"] < 0.0:
        raise ValueError(f"train time sigma must be >= 0, got {out['sigma']!r}")
    if not 0.0 <= out["straggler_share"] <= 1.0:
        raise ValueError(f"straggler_share must lie in [0, 1], got {out['straggler_share']!r}")
    if out["straggler_factor"] < 1.0:
        raise ValueError(f"straggler_factor must be >= 1, got {out['straggler_factor']!r}")
    return out


def _stream(seed: int, what: str, device_id: str) -> np.random.Generator:
    payload = f"{int(seed)}|train_time|{what}|{device_id}"
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],
                                                "big"))


def stragglers(device_ids: Sequence[str], *, seed: int, share: float) -> frozenset:
    """The ``round(share * N)`` (half up) devices with the smallest keyed draws."""
    ids = [str(d) for d in device_ids]
    k = int(math.floor(float(share) * len(ids) + 0.5))
    if k <= 0:
        return frozenset()
    ranked = sorted(ids, key=lambda d: (float(_stream(seed, "straggler", d).random()), d))
    return frozenset(ranked[:k])


def device_train_times(
    device_ids: Sequence[str],
    *,
    seed: int,
    params: Mapping[str, float],
) -> Dict[str, float]:
    """Each device's fit time ``T_j`` (simulated s), in ``device_ids`` order."""
    p = check_train_time_params(params)
    ids = [str(d) for d in device_ids]
    if len(set(ids)) != len(ids):
        raise ValueError(f"device ids repeat: {ids!r}")
    slow = stragglers(ids, seed=seed, share=p["straggler_share"])
    out: Dict[str, float] = {}
    for did in ids:
        t = p["median_s"]
        if p["sigma"] > 0.0:
            t *= math.exp(p["sigma"] * float(_stream(seed, "spread", did).standard_normal()))
        if did in slow:
            t *= p["straggler_factor"]
        out[did] = float(t)
    return out
