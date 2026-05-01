"""TargetSelectorRL — the Phase 5 intra-bucket selector wrapper.

Binds three pieces into one object the scheduler can plug into S3.5:

1. :class:`DDQN` actor.
2. :func:`extract_features_batch` for feature shaping.
3. :func:`assert_candidates_admitted` for the design §7 principle 12
   scope guard.

Two inference modes (design §2.7 "dual-purpose"):

* :meth:`select_target`  — per-round, per-bucket device ordering.
* :meth:`select_server`  — end-of-mission, picks which edge server to
  dock at. Uses the same feature vocabulary so the actor's knowledge
  transfers directly (distance + energy + rf_prior_snr dominate).

Epsilon-greedy exploration is exposed for training only — production
inference uses ``epsilon=0.0``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionPass,
    ServerID,
)

from .ddqn import DDQN
from .features import (
    FEATURE_DIM,
    SelectorEnv,
    _DISTANCE_SCALE,
    _POS_SCALE,
    extract_features_batch,
    extract_features_contact_batch,
)
from .scope_guard import SelectorScopeViolation, assert_candidates_admitted


MulePose = Tuple[float, float, float]


def _enforce_collect_pass(pass_kind: MissionPass, fn_name: str) -> None:
    """Sprint 1.5 design §7 principle 13 — selector is Pass-1-only.

    Pass 2 walks every contact greedily for universal delivery; the
    selector has no role there. Calling the selector in Pass 2 is a
    wiring bug, not a behavioural choice — so we raise a structured
    violation rather than silently doing nothing.
    """
    if pass_kind is not MissionPass.COLLECT:
        raise SelectorScopeViolation(
            f"{fn_name} called with pass_kind={pass_kind.value!r}; "
            f"the selector only runs in Pass 1 (COLLECT). Pass 2 "
            f"(DELIVER) walks every contact greedily — bypass the "
            f"selector entirely."
        )


class TargetSelectorRL:
    """S3.5 selector — orders devices within one bucket.

    The ``TargetSelectorRL`` wraps a :class:`DDQN` and does the three
    jobs S3.5 callers expect:

    * Feature extraction from scheduler state.
    * Scope-guard check (principle #12).
    * Argmax / ranking via the actor.

    Training lives in :mod:`selector_train` — this class only exposes
    the ``last_chosen_features`` hook so the trainer can pair a chosen
    action with an observed reward in the replay buffer.
    """

    def __init__(
        self,
        *,
        ddqn: Optional[DDQN] = None,
        epsilon: float = 0.0,
        rng_seed: Optional[int] = None,
    ):
        self._ddqn = ddqn if ddqn is not None else DDQN(feature_dim=FEATURE_DIM)
        if self._ddqn.feature_dim != FEATURE_DIM:
            raise ValueError(
                f"DDQN feature_dim={self._ddqn.feature_dim} != "
                f"FEATURE_DIM={FEATURE_DIM}"
            )
        self._epsilon = float(epsilon)
        self._rng = np.random.default_rng(rng_seed)
        self._last_chosen_features: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Introspection — used by the trainer
    # ------------------------------------------------------------------ #

    @property
    def ddqn(self) -> DDQN:
        return self._ddqn

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def set_epsilon(self, epsilon: float) -> None:
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError(f"epsilon must be in [0,1], got {epsilon}")
        self._epsilon = float(epsilon)

    @property
    def last_chosen_features(self) -> Optional[np.ndarray]:
        """Feature row of the most recent argmax (for replay-buffer pairing)."""
        return self._last_chosen_features

    # ------------------------------------------------------------------ #
    # Device selection — per-bucket within one round
    # ------------------------------------------------------------------ #

    def select_target(
        self,
        candidates: Sequence[DeviceID],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        bucket: Bucket,
        env: SelectorEnv,
        admitted: Optional[Sequence[DeviceID]] = None,
        *,
        pass_kind: MissionPass = MissionPass.COLLECT,
    ) -> Optional[DeviceID]:
        """Pick a single device (argmax Q) or ``None`` for empty input.

        Sprint 1.5: raises :class:`SelectorScopeViolation` when called
        with ``pass_kind=DELIVER`` — Pass 2 has no selector role
        (universal delivery via greedy-nearest-first), so any caller
        invoking the selector in DELIVER mode is a wiring bug.

        ``admitted`` defaults to ``candidates`` itself — override when
        the caller wants a stricter guard (e.g. the full S3-admitted set
        minus this bucket's members, to assert we aren't being fed a
        gated-out device via a caller bug).
        """
        _enforce_collect_pass(pass_kind, "select_target")
        if not candidates:
            self._last_chosen_features = None
            return None

        assert_candidates_admitted(
            candidates, admitted if admitted is not None else candidates
        )

        feats = extract_features_batch(
            candidates, device_states, bucket=bucket, env=env
        )

        if self._epsilon > 0.0 and self._rng.random() < self._epsilon:
            idx = int(self._rng.integers(0, feats.shape[0]))
        else:
            idx = self._ddqn.argmax(feats)

        self._last_chosen_features = feats[idx].copy()
        return candidates[idx]

    def rank(
        self,
        candidates: Sequence[DeviceID],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        bucket: Bucket,
        env: SelectorEnv,
        admitted: Optional[Sequence[DeviceID]] = None,
        *,
        pass_kind: MissionPass = MissionPass.COLLECT,
    ) -> List[DeviceID]:
        """Full bucket ordering, highest Q first.

        Sprint 1.5: raises :class:`SelectorScopeViolation` when called
        with ``pass_kind=DELIVER`` — see :meth:`select_target`.

        Ties break on ``device_id`` for determinism. Epsilon is ignored
        here — ranking is only called from inference paths.
        """
        _enforce_collect_pass(pass_kind, "rank")
        if not candidates:
            return []
        assert_candidates_admitted(
            candidates, admitted if admitted is not None else candidates
        )
        feats = extract_features_batch(
            candidates, device_states, bucket=bucket, env=env
        )
        q = self._ddqn.predict(feats)
        # argsort descending by Q; stable sort on device_id as tiebreaker.
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-float(q[i]), str(candidates[i])),
        )
        return [candidates[i] for i in order]

    # ------------------------------------------------------------------ #
    # Sprint 1.5 — per-contact selection
    # ------------------------------------------------------------------ #

    def select_contact(
        self,
        candidates: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        pass_kind: MissionPass = MissionPass.COLLECT,
        admitted: Optional[Sequence[DeviceID]] = None,
    ) -> Optional[ContactWaypoint]:
        """Pick the next contact event (argmax Q) or ``None`` for empty input.

        Sprint 1.5 per-contact API. Operates on :class:`ContactWaypoint`s
        rather than individual devices — the candidate set is one bucket's
        worth of contact positions, and the selector ranks them using the
        per-contact aggregate features (mean on_time_rate, member count,
        etc.).

        ``admitted`` (Sprint 1.5 M2 fix) is the upstream-admitted device
        set from S1/S2/S3 — i.e. every device the deterministic pipeline
        approved for this round. The scope guard verifies every contact
        member came from that set; without ``admitted``, the guard
        degrades to a self-check (still flags duplicate / empty cases
        but cannot catch a bucket leaking gated-out devices).

        Raises :class:`SelectorScopeViolation` if called in Pass 2 *or*
        if any candidate's members are not in ``admitted``.
        """
        _enforce_collect_pass(pass_kind, "select_contact")
        if not candidates:
            self._last_chosen_features = None
            return None

        members: List[DeviceID] = []
        for wp in candidates:
            members.extend(wp.devices)
        assert_candidates_admitted(
            members, admitted if admitted is not None else members,
        )

        feats = extract_features_contact_batch(candidates, device_states, env)

        if self._epsilon > 0.0 and self._rng.random() < self._epsilon:
            idx = int(self._rng.integers(0, feats.shape[0]))
        else:
            idx = self._ddqn.argmax(feats)

        self._last_chosen_features = feats[idx].copy()
        return candidates[idx]

    def rank_contacts(
        self,
        candidates: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        pass_kind: MissionPass = MissionPass.COLLECT,
        admitted: Optional[Sequence[DeviceID]] = None,
    ) -> List[ContactWaypoint]:
        """Full contact-event ordering for one bucket, highest Q first.

        Tie-break is on the tuple ``(position, devices)`` so ties are
        deterministic across re-runs. Epsilon is ignored — ranking is
        only called from inference paths.

        ``admitted`` mirrors :meth:`select_contact`'s scope-guard arg
        — see that method's docstring for the M2 rationale.

        Raises :class:`SelectorScopeViolation` if called in Pass 2 *or*
        if any candidate's members are not in ``admitted``.
        """
        _enforce_collect_pass(pass_kind, "rank_contacts")
        if not candidates:
            return []
        members: List[DeviceID] = []
        for wp in candidates:
            members.extend(wp.devices)
        assert_candidates_admitted(
            members, admitted if admitted is not None else members,
        )

        feats = extract_features_contact_batch(candidates, device_states, env)
        q = self._ddqn.predict(feats)
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-float(q[i]), candidates[i].position, candidates[i].devices),
        )
        return [candidates[i] for i in order]

    # ------------------------------------------------------------------ #
    # Server selection — end-of-mission dock target
    # ------------------------------------------------------------------ #

    def select_server(
        self,
        reachable_servers: Sequence[Tuple[ServerID, MulePose]],
        *,
        mule_pose: MulePose,
        mule_energy: float,
        rf_prior_snr_db: float = 20.0,
    ) -> Optional[ServerID]:
        """Argmax over reachable docking servers.

        Reuses the same feature vocabulary: each server is scored with
        its distance + energy budget + RF prior. No bucket concept
        applies — we one-hot the "new" slot to keep the input shape
        stable; the trainer can choose to freeze this bit.
        """
        if not reachable_servers:
            return None

        rows: List[np.ndarray] = []
        for _, pos in reachable_servers:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(mule_pose, pos)))
            row = np.asarray(
                [
                    dist / _DISTANCE_SCALE,
                    float(pos[0]) / _POS_SCALE,
                    float(pos[1]) / _POS_SCALE,
                    float(pos[2]) / _POS_SCALE,
                    0.5,            # on_time_rate neutral
                    0.0,            # beacon_fresh n/a
                    1.0, 0.0, 0.0,  # bucket one-hot pinned to slot 0
                    float(mule_energy),
                    float(rf_prior_snr_db) / 30.0,
                ],
                dtype=np.float32,
            )
            rows.append(row)

        feats = np.stack(rows, axis=0)
        idx = self._ddqn.argmax(feats)
        self._last_chosen_features = feats[idx].copy()
        return reachable_servers[idx][0]
