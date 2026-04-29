"""``DeviceRegistry`` — single source of truth for cluster-scope devices.

Two responsibilities:

1. **CRUD** on ``DeviceRecord`` rows (register, get, update, list).
2. **Slicing** — partition unassigned devices into disjoint per-mule
   ``MissionSlice`` values, refreshed on every dock.

The slicing rule (Design §7 principle 8): every active device belongs to
exactly one mule's slice during a single round. New mules joining mid-
round get an empty slice until the next ``rebalance``.

Persistence is left as a hook (``save`` / ``load``) — Phase 1 keeps the
registry in memory; Phase 6 will plug a real store behind the same API.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from hermes.types import (
    DeviceID,
    DeviceRecord,
    MissionSlice,
    MuleID,
    SpectrumSig,
)


@dataclass(frozen=True)
class RegistrySnapshot:
    """Read-only view of registry state for tests / debug / metrics."""

    total: int
    assigned: int
    new_count: int
    by_mule: Dict[MuleID, Tuple[DeviceID, ...]]


class DeviceRegistry:
    """In-memory registry with a thread-safe public surface."""

    def __init__(self) -> None:
        self._records: Dict[DeviceID, DeviceRecord] = {}
        self._lock = threading.RLock()
        self._round_counter: int = 0

    # ------------------------------------------------------------------ CRUD

    def register(
        self,
        device_id: DeviceID,
        position: Tuple[float, float, float],
        spectrum_sig: SpectrumSig,
    ) -> DeviceRecord:
        """Add a brand-new device. No-op (returns existing) if already known."""
        with self._lock:
            existing = self._records.get(device_id)
            if existing is not None:
                return existing
            rec = DeviceRecord(
                device_id=device_id,
                last_known_position=position,
                spectrum_sig=spectrum_sig,
            )
            self._records[device_id] = rec
            return rec

    def get(self, device_id: DeviceID) -> Optional[DeviceRecord]:
        with self._lock:
            return self._records.get(device_id)

    def all(self) -> List[DeviceRecord]:
        with self._lock:
            return list(self._records.values())

    def update_after_round(
        self,
        device_id: DeviceID,
        on_time: bool,
        new_position: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Bump on-time / missed counters and (optionally) refresh position."""
        with self._lock:
            rec = self._records.get(device_id)
            if rec is None:
                # Unknown device — silently drop. The caller (cluster
                # round-close ingest) is the place to log this; the
                # registry keeps no opinion.
                return
            if on_time:
                rec.on_time_history += 1
            else:
                rec.missed_history += 1
            rec.is_new = False
            if new_position is not None:
                rec.last_known_position = new_position

    # ------------------------------------------------------------------ slice

    def rebalance(
        self,
        mules: Iterable[MuleID],
        *,
        round_counter: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Dict[MuleID, MissionSlice]:
        """Disjoint round-robin slice over all devices.

        Strategy: clear the current ``assigned_mule`` field on every record,
        then walk devices in insertion order assigning them to mules in a
        round-robin pattern. New (``is_new=True``) devices are placed first
        so they fill the front of each slice — Rank-1 priority bucket later.

        ``round_counter`` lets the caller step it; defaults to the registry's
        own monotonic counter (incremented on each rebalance).
        """
        mule_list = list(mules)
        if not mule_list:
            raise ValueError("rebalance requires at least one MuleID")
        if len(set(mule_list)) != len(mule_list):
            raise ValueError("duplicate MuleID in rebalance call")

        ts = now if now is not None else time.time()
        with self._lock:
            self._round_counter = (
                round_counter if round_counter is not None else self._round_counter + 1
            )

            # clear prior assignments so a previously-assigned device that is
            # no longer in `mules` becomes unassigned this round.
            for rec in self._records.values():
                rec.assigned_mule = None

            # new devices first, then known devices, both stable-sorted by ID
            # so the slicing is deterministic for tests.
            ordered = sorted(self._records.values(), key=lambda r: (not r.is_new, r.device_id))

            buckets: Dict[MuleID, List[DeviceID]] = {m: [] for m in mule_list}
            for i, rec in enumerate(ordered):
                target = mule_list[i % len(mule_list)]
                buckets[target].append(rec.device_id)
                rec.assigned_mule = target

            return {
                mule: MissionSlice(
                    mule_id=mule,
                    device_ids=tuple(ids),
                    issued_round=self._round_counter,
                    issued_at=ts,
                )
                for mule, ids in buckets.items()
            }

    def slice_for(self, mule_id: MuleID) -> Tuple[DeviceID, ...]:
        """Read current assignment for one mule (no rebalance)."""
        with self._lock:
            return tuple(
                d for d, r in self._records.items() if r.assigned_mule == mule_id
            )

    # ------------------------------------------------------------------ misc

    def snapshot(self) -> RegistrySnapshot:
        with self._lock:
            by_mule: Dict[MuleID, List[DeviceID]] = {}
            assigned = 0
            new_count = 0
            for rec in self._records.values():
                if rec.is_new:
                    new_count += 1
                if rec.assigned_mule is not None:
                    assigned += 1
                    by_mule.setdefault(rec.assigned_mule, []).append(rec.device_id)
            return RegistrySnapshot(
                total=len(self._records),
                assigned=assigned,
                new_count=new_count,
                by_mule={k: tuple(v) for k, v in by_mule.items()},
            )

    @property
    def round_counter(self) -> int:
        with self._lock:
            return self._round_counter

    # --- persistence hooks (no-op now, real impl in a later phase) ---
    def save(self, path: str) -> None:
        """Phase-6 hook. Intentionally a no-op until then."""
        # Keeping the signature stable means downstream code can call save()
        # today without conditional branches.
        return None

    @classmethod
    def load(cls, path: str) -> "DeviceRegistry":
        """Phase-6 hook. Returns an empty registry until real persistence lands."""
        return cls()
