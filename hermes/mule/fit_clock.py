"""The devices' local fits on the simulated clock (Exp 5 addendum, Study 5.12).

On the wall clock a device trains between visits in real time, and on the
simulated mission clock every recorded run charged its training nothing: an
update was always ready when the mule came. Study 5.12 asks whether FeRRy
holds up when devices train at different speeds, so a device's local fit takes
simulated time, and a Pass-1 contact can find no update ready.

**The model.** It lives on the mule, as the availability draw does (the draw
and its outcome are the mule's; the device only obeys the mark):

* Device j's fit takes ``T_j`` simulated seconds (``MuleConfig.
  device_train_time_s``, drawn per trial by the driver,
  ``experiments/exp4/compute.py``). A device without one is always ready, as
  in every recorded run, and so is one whose ``T_j`` is 0: the sweep's "none"
  level, which flies exactly as a recorded run but records its fits and
  uplinks for the device energy.
* A device starts a fit each time a model reaches it: a Pass-1 push (collected,
  uplink dropped or unanswered) or a Pass-2 delivery, at that session's stamp
  (``ContactCommit.pushed`` and ``contact_ts``). Its first fit starts at the
  trial's start, the mule's first takeoff: it was deployed with the seed model.
* A Pass-1 contact at time ``t`` finds an update ready iff
  ``t >= start_j + T_j``. Otherwise the target is *not ready*: the solicit
  names it, it answers with its advert, and nothing is pushed, so its fit runs
  on and its start does not move (``ContactPlan.not_ready``).

The device code trains after a delivery and, when the mule asks it to train
ahead, after a Pass-1 push, so this model times its fits exactly there. A
device that adopts a Pass-1 basis without training ahead (an unbudgeted Pass 2)
and then misses its delivery fits at its next contact, on that contact's model;
the model times that fit from the basis it last received, the one place where
it is optimistic. Pure bookkeeping: no wall clock and no randomness.
"""

from __future__ import annotations

import math
import numbers
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

__all__ = ["FitClock", "check_train_times"]


def check_train_times(train_time_s: Mapping[str, float]) -> Dict[str, float]:
    """``{device_id: T_j}`` validated: string ids, finite times >= 0."""
    if not isinstance(train_time_s, Mapping):
        raise TypeError(f"train times must be a mapping of device id to seconds, "
                        f"got {type(train_time_s).__name__}")
    out: Dict[str, float] = {}
    for did, t in train_time_s.items():
        if not isinstance(did, str):
            raise TypeError(f"train-time keys are device ids (str), got {did!r}")
        if isinstance(t, bool) or not isinstance(t, numbers.Real):
            raise TypeError(f"train time of {did!r} must be a number, got {t!r}")
        t = float(t)
        if not (math.isfinite(t) and t >= 0.0):
            raise ValueError(f"train time of {did!r} must be finite and >= 0 s, got {t!r}")
        out[did] = t
    return out


class FitClock:
    """Each device's current fit on the mission clock (module docstring)."""

    def __init__(self, train_time_s: Mapping[str, float]) -> None:
        self._train_s = check_train_times(train_time_s)
        if not self._train_s:
            raise ValueError("a FitClock needs at least one device with a train time")
        self._start: Dict[str, float] = {}
        self._origin: Optional[float] = None

    @property
    def n_devices(self) -> int:
        """How many devices the clock times."""
        return len(self._train_s)

    @property
    def origin(self) -> Optional[float]:
        """The trial's start (the first takeoff), None before it."""
        return self._origin

    def train_time_s(self, device_id: str) -> Optional[float]:
        return self._train_s.get(str(device_id))

    def take_off(self, t_s: float) -> List[Tuple[str, float]]:
        """Note a takeoff at ``t_s``. The first one starts every device's first
        fit there; returns those fits ``(device, start)`` in id order, else []."""
        if self._origin is not None:
            return []
        self._origin = float(t_s)
        return [(did, self._origin) for did in sorted(self._train_s)]

    def ready_at(self, device_id: str) -> Optional[float]:
        """When the device's current fit ends; None without a train time."""
        t = self._train_s.get(str(device_id))
        if t is None:
            return None
        if self._origin is None:
            raise RuntimeError("the fit clock has no origin yet: call take_off first")
        return self._start.get(str(device_id), self._origin) + t

    def not_ready(self, members: Iterable[str], t_s: float) -> FrozenSet[str]:
        """The members whose current fit has not finished at ``t_s``."""
        out = set()
        for did in members:
            end = self.ready_at(did)
            if end is not None and float(t_s) < end:
                out.add(did)
        return frozenset(out)

    def received(self, devices: Iterable[str],
                 stamps: Mapping[str, float]) -> List[Tuple[str, float]]:
        """A model reached each of ``devices`` at its stamp: each starts a fit
        there. Returns the fits started ``(device, start)``, in the given order
        (devices without a train time are left out)."""
        out: List[Tuple[str, float]] = []
        for did in devices:
            if str(did) not in self._train_s:
                continue
            t = float(stamps[did])
            self._start[str(did)] = t
            out.append((str(did), t))
        return out
