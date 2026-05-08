"""Experiment-3 simulator — extends ContactSim with the paper's knobs.

The Sprint-1.5 :class:`~hermes.scheduler.selector.sim_env.ContactSim`
gives us most of what Experiment 3 needs (planar field, S3a clustering,
per-device reliability, parallel sessions per contact). The paper
additionally specifies:

* **3 base stations** along the field's far edge (so propulsion energy
  is the round-trip cost from the contact to the *nearest* BS, not a
  return-to-origin abstraction).
* **Time-varying per-channel upload rates** (the cluster-edge link
  bandwidth fluctuates during a trial — captures shadowing / contention
  at AERPAW radio scale).
* **Fixed-speed mule** (the legacy ``time = SESSION_TIME +
  TIME_PER_DIST · dist`` model implicitly assumed a teleporting mule;
  fixed-speed gives propulsion energy the right shape at long path
  lengths via Eq. 5).
* **Configurable N / β / deadline-heterogeneity / rrf** (the trial
  grid sweeps over all four).

Per the implementation plan §4.2 trade-off note: rather than rewrite
ContactSim, we *compose* it — Exp3Sim wraps a ContactSim instance,
overrides the cost model, and swaps in the experiment's calibrated
reward shape.

The selector / policy interface is unchanged: ``Exp3SelectorEnv``
subclasses :class:`SelectorEnv` (which is frozen) by way of a sibling
dataclass with the additional fields the EDF policy reads via getattr
(``mission_deadline_s``, ``upload_rate_bps_at``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionOutcome,
)
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.selector.sim_env import (
    COMPLETION_BONUS,
    ENERGY_W,
    SESSION_TIME,
    _SimDevice,
)
from hermes.scheduler.stages.s3a_cluster import cluster_by_rf_range


Position = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# Env subclass — adds the EDF policy's hooks
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Exp3SelectorEnv(SelectorEnv):
    """SelectorEnv with the experiment-3 hooks the EDF policy reads.

    These fields are optional from the selector's perspective (it
    ignores them); the EDF policy looks them up via ``getattr`` so a
    plain :class:`SelectorEnv` still works in unit tests.
    """

    mission_deadline_s: float = float("inf")
    # Callable signature: ``f(position) -> bits-per-second``. Wrapped in
    # a default-factory tuple so dataclass equality stays well-defined.
    upload_rate_bps_at: Optional[Callable[[Position], float]] = None


# --------------------------------------------------------------------------- #
# Configuration knobs
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Exp3SimConfig:
    """All knobs the trial grid sweeps over.

    The four primary axes:

    * ``n_devices`` (paper's *N*) — bucket size per round.
    * ``beta`` — deadline tightness multiplier; lower ⇒ tighter, less
      slack to chase distant or flaky devices.
    * ``deadline_heterogeneity`` — when True, half the devices get
      ``deadline_fulfilment_s · 0.5`` and half get ``· 1.5`` so the
      EDF policy actually has work to do; when False every device
      shares the median deadline.
    * ``rf_range_m`` (paper's *rrf*) — S3a clustering radius.

    Other fields are physical constants the calibration TOML may
    override at the experiment-driver level.
    """

    n_devices: int = 10
    beta: float = 1.0
    deadline_heterogeneity: bool = False
    rf_range_m: float = 60.0

    world_radius: float = 100.0
    cruise_speed_m_s: float = 5.0
    session_time_s: float = SESSION_TIME

    # Total mission budget in seconds. The trial-grid scales it via
    # ``beta`` so a tighter deadline cell exposes scheduling pressure.
    mission_budget_s: float = 600.0

    # Upload model. ``upload_bytes_per_contact`` defaults to 0 so an
    # un-calibrated test run doesn't burn budget on the upload term.
    upload_bytes_per_contact: float = 0.0
    nominal_upload_bps: float = 1.0e7
    upload_jitter_pct: float = 0.20  # ±20% multiplicative noise per evaluation

    # Three base stations on the far edge of the world. Defaults match
    # the paper's "evenly-spaced along the +y boundary" layout.
    base_station_positions: Tuple[Position, ...] = (
        (-80.0, 100.0, 0.0),
        (0.0, 100.0, 0.0),
        (80.0, 100.0, 0.0),
    )

    # ---------------------------------------------------------------- #
    # Jittery-mode parameters — match Experiment 1's ``--jittery`` cell
    # (tc/netem ±30% latency jitter + 2% packet loss) so the
    # scheduling-layer ablation can be evaluated under the same
    # network-link impairments. Default 0 leaves the simulator in its
    # clean (non-jittery) state.
    # ---------------------------------------------------------------- #
    # Per-device completion is multiplied by an additional Bernoulli
    # `(1 - packet_loss_pct/100)` on top of the existing
    # reliability×rf_factor product. Set to 2.0 to mirror Exp.\ 1
    # jittery.
    packet_loss_pct: float = 0.0
    # transit_s, upload_s, and collect_s are each multiplied by
    # `Normal(1, latency_jitter_pct/100)` per call, clamped to
    # ≥ 0.05× the deterministic value to prevent negative or
    # implausibly small times. Set to 30.0 to mirror Exp.\ 1 jittery.
    latency_jitter_pct: float = 0.0

    # ---------------------------------------------------------------- #
    # Two-pass mission knobs — make Exp3Sim a faithful walk of the
    # PRE-MISSION-load → Pass-1-collect → DOCK → Pass-2-deliver flow
    # described in HERMES_FL_Scheduler_Design.md §V.
    # ---------------------------------------------------------------- #
    # Inter-pass dock event: between Pass 1 and Pass 2 the mule flies
    # back to the nearest BS / dock, ships the mission_aggregate UP,
    # and receives fresh θ_disc' DOWN. Modeled as a fixed dock_time_s
    # charge in addition to the transit. Default 30 s mirrors the
    # paper's intra-NUC handoff window.
    dock_time_s: float = 30.0
    # Pass-2 delivery session per contact — push-only downlink (no Δθ
    # pull), so much shorter than Pass-1's collect_s. Default 5 s vs
    # the 30 s collect default.
    delivery_session_s: float = 5.0

    energy_weight: float = 1.0
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
# Step result
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Exp3StepResult:
    """One contact event's outcome under the Exp3 cost model."""

    contact: ContactWaypoint
    reward: float
    transit_s: float
    collect_s: float
    return_s: float
    upload_s: float
    energy_used: float
    transit_distance_m: float
    return_distance_m: float
    upload_bytes: float
    nearest_bs: Position
    per_device_completed: List[bool]

    @property
    def time_to_complete(self) -> float:
        return self.transit_s + self.collect_s + self.upload_s

    @property
    def completed_count(self) -> int:
        return sum(self.per_device_completed)

    @property
    def member_count(self) -> int:
        return len(self.per_device_completed)


@dataclass(frozen=True)
class Exp3DockResult:
    """Outcome of an inter-pass dock event.

    The mule flies from its post-Pass-1 position back to the nearest
    BS, swaps the mission_aggregate UP and θ_disc' DOWN bundles, and
    is now positioned at the BS for the start of Pass 2.
    """

    transit_s: float
    transit_distance_m: float
    dock_time_s: float
    bs_position: Position
    success: bool


@dataclass(frozen=True)
class Exp3DeliveryStepResult:
    """One Pass-2 contact's outcome.

    Pass 2 is push-only (deliver fresh θ to every device in range);
    no Δθ is pulled back, so this carries no upload accounting.
    """

    contact: ContactWaypoint
    transit_s: float
    transit_distance_m: float
    delivery_s: float
    per_device_delivered: List[bool]

    @property
    def delivered_count(self) -> int:
        return sum(self.per_device_delivered)

    @property
    def member_count(self) -> int:
        return len(self.per_device_delivered)


# --------------------------------------------------------------------------- #
# Per-episode roll-up
# --------------------------------------------------------------------------- #

@dataclass
class Exp3EpisodeMetrics:
    """All experiment-3 reportables that derive from per-step state."""

    contacts_visited: int = 0
    contacts_total_pass1: int = 0  # ρ_contact denominator
    devices_visited: int = 0
    devices_completed: int = 0
    energy_total: float = 0.0
    transit_distance_m: float = 0.0
    # Legacy: per-step contact→nearest-BS distance summed across Pass 1.
    # Originally a placeholder for end-of-mission return cost; with the
    # explicit ``dock_at_bs`` event now wired up, this field is retained
    # for backward compat but is NO LONGER added to ``path_length_m``
    # (the dock event records the real flight). Future work may delete it.
    return_distance_m: float = 0.0
    upload_bytes: float = 0.0
    transit_time_s: float = 0.0
    upload_time_s: float = 0.0
    collect_time_s: float = 0.0
    # Per-device service counters (fairness inputs).
    per_device_visits: Dict[DeviceID, int] = field(default_factory=dict)
    per_device_completions: Dict[DeviceID, int] = field(default_factory=dict)
    # Per-round close-rate inputs.
    rounds_completed: int = 0
    rounds_attempted: int = 0
    # Pass-2 deliverability — count of devices that received fresh θ.
    pass2_devices_reached: int = 0
    # Inter-pass dock accounting (one event per mission, between Pass 1
    # and Pass 2). dock_attempted is True iff arm_mule called dock_at_bs;
    # dock_success captures whether budget allowed it.
    dock_transit_distance_m: float = 0.0
    dock_transit_time_s: float = 0.0
    dock_time_s: float = 0.0
    dock_attempted: bool = False
    dock_success: bool = False
    # Pass-2 walk accounting (cumulative across the greedy walk).
    pass2_transit_distance_m: float = 0.0
    pass2_transit_time_s: float = 0.0
    pass2_delivery_time_s: float = 0.0
    pass2_contacts_visited: int = 0
    # Devices whose Δθ slot was empty when the mule arrived (mule was
    # late vs ``deadline_fulfilment_s``). Diagnostic for the deadline-
    # gating mechanism — surfaces *why* yield drops in budget-tight cells.
    pass1_devices_deadline_missed: int = 0

    @property
    def path_length_m(self) -> float:
        # Real flight path: Pass 1 transits + dock leg + Pass 2 walk.
        # The legacy ``return_distance_m`` is intentionally excluded —
        # it was a per-step placeholder for an end-of-mission return
        # that the explicit dock event now models honestly.
        return (
            self.transit_distance_m
            + self.dock_transit_distance_m
            + self.pass2_transit_distance_m
        )

    @property
    def time_total_s(self) -> float:
        return (
            self.transit_time_s + self.upload_time_s + self.collect_time_s
            + self.dock_transit_time_s + self.dock_time_s
            + self.pass2_transit_time_s + self.pass2_delivery_time_s
        )


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #

class Exp3Sim:
    """Experiment-3 simulator with paper-specified physics.

    Compared to :class:`ContactSim`:

    * ``transit_s = dist / cruise_speed_m_s`` (fixed-speed mule).
    * ``return_s = dist(contact, nearest_bs) / cruise_speed_m_s``
      (used for cost accounting + in propulsion energy via Eq. 5).
    * ``upload_s = upload_bytes_per_contact * 8 / rate_at(position)``
      where the rate is a noisy function of the contact's distance to
      the nearest BS — captures the time-varying-channel knob.
    * ``deadline_heterogeneity`` populates per-device
      ``deadline_fulfilment_s`` so EDF actually has work to do.
    """

    def __init__(self, cfg: Exp3SimConfig) -> None:
        if cfg.n_devices < 1:
            raise ValueError(f"n_devices must be >= 1, got {cfg.n_devices}")
        if cfg.rf_range_m <= 0.0:
            raise ValueError(f"rf_range_m must be > 0, got {cfg.rf_range_m}")
        if cfg.cruise_speed_m_s <= 0.0:
            raise ValueError(
                f"cruise_speed_m_s must be > 0, got {cfg.cruise_speed_m_s}"
            )
        if cfg.beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {cfg.beta}")
        if cfg.mission_budget_s <= 0.0:
            raise ValueError(
                f"mission_budget_s must be > 0, got {cfg.mission_budget_s}"
            )
        if not cfg.base_station_positions:
            raise ValueError("base_station_positions must contain at least one BS")

        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        # Mule starts at the BS-infrastructure centroid — the
        # PRE-MISSION dock load (HERMES_FL_Scheduler_Design.md §V) puts
        # θ_disc on the mule at the cluster's edge server, not at the
        # field origin. Using the centroid of the BS positions
        # represents an abstract dock co-located with the BS array;
        # picking any single BS would bias the first-leg geometry.
        self._mule_pose: Position = _bs_centroid(cfg.base_station_positions)
        self._mule_energy: float = 1.0
        self._mission_budget: float = cfg.mission_budget_s * cfg.beta
        self._budget_remaining: float = self._mission_budget
        self._now: float = 0.0
        self._devices: List[_SimDevice] = []
        self._states: Dict[DeviceID, DeviceSchedulerState] = {}
        self._deadlines: Dict[DeviceID, float] = {}
        self._contacts: List[ContactWaypoint] = []
        # Snapshot of every contact produced by S3a clustering at reset.
        # Pass 1 depletes ``_contacts``; ``_initial_contacts`` is what
        # the Pass-2 greedy walk iterates over (every admitted-device
        # cluster, regardless of Pass-1 service status).
        self._initial_contacts: List[ContactWaypoint] = []
        self._episode_metrics: Exp3EpisodeMetrics = Exp3EpisodeMetrics()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        cfg = self._cfg
        self._mule_pose = _bs_centroid(cfg.base_station_positions)
        self._mule_energy = 1.0
        self._mission_budget = cfg.mission_budget_s * cfg.beta
        self._budget_remaining = self._mission_budget
        self._now = 0.0
        self._devices = []
        self._states = {}
        self._deadlines = {}
        self._contacts = []
        self._initial_contacts = []
        self._episode_metrics = Exp3EpisodeMetrics()

        _PRE_ROUNDS = 11
        for k in range(cfg.n_devices):
            did = DeviceID(f"sim-{k:02d}")
            x = float(self._rng.uniform(-cfg.world_radius, cfg.world_radius))
            y = float(self._rng.uniform(-cfg.world_radius, cfg.world_radius))
            reliability = float(self._rng.uniform(0.15, 1.0))
            self._devices.append(
                _SimDevice(device_id=did, pos=(x, y, 0.0), reliability=reliability)
            )
            hits = sum(
                1 for _ in range(_PRE_ROUNDS) if self._rng.random() < reliability
            )
            last_outcome = (
                MissionOutcome.CLEAN
                if hits * 2 >= _PRE_ROUNDS
                else MissionOutcome.TIMEOUT
            )

            # Per-device prepared-Δθ deadline. Scales with the mission
            # budget so the gating produces real selection pressure for
            # EDF without nuking every contact past the first:
            #
            # * uniform mode (deadline_heterogeneity=False): every
            #   device gets ~50% of the (β-scaled) mission budget.
            #   With mission_budget_s=600 and β=1, that's 300 s — the
            #   mule can reach 4-5 contacts before the boundary bites.
            # * heterogeneous mode: half tight (25% of budget), half
            #   loose (75%). Gives EDF a clear ranking signal — tight-
            #   deadline contacts must be visited early.
            #
            # The legacy 60 s constant was harmless when deadlines only
            # fed EDF's heuristic; now that ``step`` actually gates
            # completion on them, the value must be in scale with the
            # mission. cfg.beta is *not* applied a second time here —
            # mission_budget_s already includes the β scaling further
            # down (mission_budget = mission_budget_s * beta).
            base_deadline = cfg.mission_budget_s * 0.5
            if cfg.deadline_heterogeneity:
                deadline_window = base_deadline * (0.5 if k % 2 == 0 else 1.5)
            else:
                deadline_window = base_deadline
            deadline_window *= cfg.beta

            st = DeviceSchedulerState(
                device_id=did,
                is_in_slice=True,
                is_new=False,
                last_known_position=(x, y, 0.0),
                last_outcome=last_outcome,
                on_time_count=hits,
                missed_count=_PRE_ROUNDS - hits,
                deadline_fulfilment_s=deadline_window,
            )
            st.bucket = Bucket.SCHEDULED_THIS_ROUND
            self._states[did] = st
            # Deadline timestamp = now + per-device window. The EDF
            # policy uses this directly when ranking contacts.
            self._deadlines[did] = self._now + deadline_window

            self._episode_metrics.per_device_visits[did] = 0
            self._episode_metrics.per_device_completions[did] = 0

        # Cluster into contacts via the production S3a algorithm.
        self._contacts = cluster_by_rf_range(
            eligible_device_ids=list(self._states.keys()),
            device_states=self._states,
            deadlines=self._deadlines,
            rf_range_m=cfg.rf_range_m,
        )
        self._initial_contacts = list(self._contacts)
        self._episode_metrics.contacts_total_pass1 = len(self._contacts)

    # ------------------------------------------------------------------ #
    # Snapshot accessors — what arms read each step
    # ------------------------------------------------------------------ #

    @property
    def cfg(self) -> Exp3SimConfig:
        return self._cfg

    @property
    def mule_pose(self) -> Position:
        return self._mule_pose

    @property
    def mule_energy(self) -> float:
        return self._mule_energy

    @property
    def now(self) -> float:
        return self._now

    @property
    def budget_remaining(self) -> float:
        return self._budget_remaining

    @property
    def mission_budget(self) -> float:
        return self._mission_budget

    @property
    def done(self) -> bool:
        if not self._contacts:
            return True
        # Cheapest possible visit = transit-time-zero + collect; if even
        # that doesn't fit, the episode is over.
        return self._budget_remaining < self._cfg.session_time_s

    def candidates(self) -> List[ContactWaypoint]:
        return list(self._contacts)

    def device_states(self) -> Dict[DeviceID, DeviceSchedulerState]:
        return dict(self._states)

    def deadlines(self) -> Dict[DeviceID, float]:
        return dict(self._deadlines)

    def selector_env(self) -> Exp3SelectorEnv:
        """Env handed to the policy / selector at this step."""
        return Exp3SelectorEnv(
            mule_pose=self._mule_pose,
            mule_energy=self._mule_energy,
            rf_prior_snr_db=20.0,
            now=self._now,
            mission_deadline_s=self._mission_budget,
            upload_rate_bps_at=self._upload_rate_at,
        )

    @property
    def episode_metrics(self) -> Exp3EpisodeMetrics:
        return self._episode_metrics

    @property
    def base_station_positions(self) -> Tuple[Position, ...]:
        return self._cfg.base_station_positions

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, contact: ContactWaypoint) -> Exp3StepResult:
        """Visit one contact (selected by the arm-specific policy).

        Unlike :class:`ContactSim.step`, this takes a
        :class:`ContactWaypoint` directly rather than an index — A2 / A3
        return *re-ordered* candidate lists and the driver chooses by
        ContactWaypoint identity, so accepting the contact is friendlier.
        """
        if self.done:
            raise RuntimeError("step called on terminated episode")
        try:
            idx = next(
                i for i, c in enumerate(self._contacts) if c is contact
            )
        except StopIteration:
            # Fall back to value equality if the caller copied the
            # contact (A2 returns ``list(candidates)`` so identity holds,
            # but A3 sorts so identity is preserved too — value match is
            # only needed for tests passing reconstructed waypoints).
            try:
                idx = self._contacts.index(contact)
            except ValueError as e:
                raise ValueError(
                    f"step called with contact not in candidates: {contact}"
                ) from e

        cfg = self._cfg

        transit_dist = _euclid(self._mule_pose, contact.position)
        transit_s = transit_dist / cfg.cruise_speed_m_s
        collect_s = cfg.session_time_s

        nearest_bs = self._nearest_bs(contact.position)
        return_dist = _euclid(contact.position, nearest_bs)
        return_s = return_dist / cfg.cruise_speed_m_s

        upload_rate = self._upload_rate_at(contact.position)
        upload_bytes = cfg.upload_bytes_per_contact
        upload_s = upload_bytes * 8.0 / max(upload_rate, 1e-6)

        # Network-jitter physics: only the long-range mule→BS upload
        # link is exposed to jitter. Mule transit is a physical flight
        # operation (not network-bound); the device↔mule collect is a
        # short-range RF session with much higher reliability than the
        # long-range BS link. Applying jitter to transit_s or collect_s
        # would conflate physical time with network time.
        if cfg.latency_jitter_pct > 0.0:
            sigma = cfg.latency_jitter_pct / 100.0
            upload_s *= max(0.05, float(self._rng.normal(1.0, sigma)))

        # Long-range packet loss applied at the *contact* level: the
        # mule→BS upload either succeeds for the whole batch or fails
        # for the whole batch (one Bernoulli draw per contact). This
        # is the right model because a single upload over the long-
        # range link carries the aggregated payload from every device
        # in the contact; if that upload drops, every device loses
        # its update for this round, but the device→mule short hop
        # itself never fails.
        contact_uplink_kept = (
            cfg.packet_loss_pct <= 0.0
            or self._rng.random() < (1.0 - cfg.packet_loss_pct / 100.0)
        )

        per_device_completed: List[bool] = []
        # Anticipate the post-step time so deadline gating reflects
        # *when the contact actually wraps up*, not when the mule
        # arrived. A device whose deadline elapses mid-session has an
        # empty Δθ slot by the time the mule asks for it.
        contact_finish_time = self._now + transit_s + collect_s
        for did in contact.devices:
            dev = next((d for d in self._devices if d.device_id == did), None)
            if dev is None:
                per_device_completed.append(False)
                continue
            d_dist = _euclid(contact.position, dev.pos)
            rf_factor = max(0.4, 1.0 - d_dist / (3.0 * cfg.world_radius))
            # Short-range device↔mule completion (no packet loss here).
            p_complete = max(0.0, min(1.0, dev.reliability * rf_factor))
            # Deadline gating: a device's prepared Δθ has a window
            # (``deadline_fulfilment_s``). The design doc has the
            # device stash its Δθ in a delivery slot — if the mule
            # arrives after the absolute deadline, the slot is empty
            # because the device has either moved on to the next
            # local-train cycle or the prepared Δθ has been recycled.
            # This is what gives EDF (A3) genuine signal vs FIFO (A2):
            # A3 prioritizes tighter-deadline contacts, which under
            # this gating are the ones with real participation risk.
            deadline_passed = (
                contact_finish_time > self._deadlines.get(did, float("inf"))
            )
            if deadline_passed:
                completed = False
                self._episode_metrics.pass1_devices_deadline_missed += 1
            else:
                completed = bool(
                    contact_uplink_kept
                    and self._rng.random() < p_complete
                )
            per_device_completed.append(completed)
            # Per-device service counters for fairness metrics.
            self._episode_metrics.per_device_visits[did] = (
                self._episode_metrics.per_device_visits.get(did, 0) + 1
            )
            if completed:
                self._episode_metrics.per_device_completions[did] = (
                    self._episode_metrics.per_device_completions.get(did, 0) + 1
                )

        n_completed = sum(per_device_completed)
        # Energy proxy used for the *training* reward only; the paper's
        # Eq. 5 value is computed post-hoc by metrics.exp3_propulsion_energy
        # off the per-step accounting.
        energy_used = transit_dist * ENERGY_W

        # Reward shape matches ContactSim's so a TargetSelectorRL trained
        # against the legacy A/B sim transfers without re-tuning.
        reward = (
            COMPLETION_BONUS * n_completed
            - (transit_s + collect_s)
            - cfg.energy_weight * energy_used
        )

        # Round close: a "round" here = one contact attempt; the round
        # closes successfully iff at least one device completed.
        self._episode_metrics.rounds_attempted += 1
        if n_completed > 0:
            self._episode_metrics.rounds_completed += 1

        # Advance state — note we do NOT add return_s to the budget,
        # because the mule only flies back to the BS at end-of-mission.
        # return_s is recorded for Eq. 5 propulsion accounting only.
        self._mule_pose = contact.position
        self._mule_energy = max(0.0, self._mule_energy - energy_used)
        consumed = transit_s + collect_s + upload_s
        self._budget_remaining = max(0.0, self._budget_remaining - consumed)
        self._now += consumed

        self._episode_metrics.contacts_visited += 1
        self._episode_metrics.devices_visited += len(contact.devices)
        self._episode_metrics.devices_completed += n_completed
        self._episode_metrics.energy_total += energy_used
        self._episode_metrics.transit_distance_m += transit_dist
        self._episode_metrics.return_distance_m += return_dist
        self._episode_metrics.upload_bytes += upload_bytes
        self._episode_metrics.transit_time_s += transit_s
        self._episode_metrics.upload_time_s += upload_s
        self._episode_metrics.collect_time_s += collect_s

        # Drop the contact from the Pass-1 unvisited set. Device
        # state (``_states`` / ``_deadlines`` / ``_devices``) is
        # intentionally retained so Pass 2's greedy walk can look up
        # positions and short-range reliabilities. Each device only
        # appears in one contact (``cluster_by_rf_range`` partitions
        # by RF range), so there's no risk of double-scheduling within
        # Pass 1 from leaving the entries in place.
        self._contacts.pop(idx)

        return Exp3StepResult(
            contact=contact,
            reward=reward,
            transit_s=transit_s,
            collect_s=collect_s,
            return_s=return_s,
            upload_s=upload_s,
            energy_used=energy_used,
            transit_distance_m=transit_dist,
            return_distance_m=return_dist,
            upload_bytes=upload_bytes,
            nearest_bs=nearest_bs,
            per_device_completed=per_device_completed,
        )

    def record_pass2_deliveries(self, devices_reached: int) -> None:
        """Hook for the arm driver to record Pass-2 delivery coverage.

        Pass 2 walks every remaining contact greedily (handled by the
        arm driver, not the policy); this is where it stamps the count
        of devices that received fresh weights for the metric module.

        Note: the production-faithful path now uses :meth:`step_deliver`,
        which updates ``pass2_devices_reached`` directly. This setter
        is retained for tests and external callers that drive the
        Pass-2 metric without calling :meth:`step_deliver`.
        """
        if devices_reached < 0:
            raise ValueError("devices_reached must be non-negative")
        self._episode_metrics.pass2_devices_reached = devices_reached

    # ------------------------------------------------------------------ #
    # Inter-pass dock event
    # ------------------------------------------------------------------ #

    def dock_at_bs(self) -> Exp3DockResult:
        """Fly to the nearest BS and dock with the cluster.

        Models the inter-pass dock from
        HERMES_FL_Scheduler_Design.md §V (lines 305-319): mule ships
        Pass-1's mission_aggregate UP, the cluster runs cross-mule
        FedAvg, and the mule receives fresh ``θ_disc'`` DOWN for
        Pass-2 delivery.

        Charges ``transit_s + cfg.dock_time_s`` against the budget
        and updates ``mule_pose`` to the BS position. If the residual
        budget can't cover the round-trip, returns ``success=False``
        without advancing state — the caller should skip Pass 2.
        """
        cfg = self._cfg
        bs = self._nearest_bs(self._mule_pose)
        transit_dist = _euclid(self._mule_pose, bs)
        transit_s = transit_dist / cfg.cruise_speed_m_s
        cost = transit_s + cfg.dock_time_s
        self._episode_metrics.dock_attempted = True
        if cost > self._budget_remaining:
            return Exp3DockResult(
                transit_s=transit_s,
                transit_distance_m=transit_dist,
                dock_time_s=cfg.dock_time_s,
                bs_position=bs,
                success=False,
            )
        self._mule_pose = bs
        self._budget_remaining = max(0.0, self._budget_remaining - cost)
        self._now += cost
        # Energy proxy: the dock leg is real flight, so charge it the
        # same per-meter cost the Pass-1 transit uses.
        self._mule_energy = max(0.0, self._mule_energy - transit_dist * ENERGY_W)
        self._episode_metrics.dock_transit_distance_m = transit_dist
        self._episode_metrics.dock_transit_time_s = transit_s
        self._episode_metrics.dock_time_s = cfg.dock_time_s
        self._episode_metrics.dock_success = True
        self._episode_metrics.energy_total += transit_dist * ENERGY_W
        return Exp3DockResult(
            transit_s=transit_s,
            transit_distance_m=transit_dist,
            dock_time_s=cfg.dock_time_s,
            bs_position=bs,
            success=True,
        )

    # ------------------------------------------------------------------ #
    # Pass-2 walk
    # ------------------------------------------------------------------ #

    def pass2_candidates(self) -> List[ContactWaypoint]:
        """Snapshot of every contact produced by S3a clustering.

        Pass 2 walks all of these (selector bypassed — the goal is
        universal delivery), greedy nearest-first from the post-dock
        mule position. Caller pops as it visits.
        """
        return list(self._initial_contacts)

    def step_deliver(self, contact: ContactWaypoint) -> Exp3DeliveryStepResult:
        """Visit one contact in Pass 2 to push fresh ``θ_disc'``.

        Costs ``transit_s + cfg.delivery_session_s``; no upload (Pass
        2 is push-only — the long-range link was exercised once at
        the dock for the DOWN bundle, not per-contact).

        Per-device delivery succeeds iff a Bernoulli on the device's
        short-range ``reliability × rf_factor`` passes. Pass 2 does
        NOT inherit the long-range packet-loss check (the mule already
        holds ``θ_disc'`` after docking; the per-contact link is the
        reliable short-range hop).

        If residual budget can't cover the cost, returns a result
        with all-False delivery flags and does not advance state.
        """
        if self.done and self._budget_remaining <= 0.0:
            # Defensive: empty result if mission is fully exhausted.
            return Exp3DeliveryStepResult(
                contact=contact,
                transit_s=0.0,
                transit_distance_m=0.0,
                delivery_s=0.0,
                per_device_delivered=[False] * len(contact.devices),
            )
        cfg = self._cfg
        transit_dist = _euclid(self._mule_pose, contact.position)
        transit_s = transit_dist / cfg.cruise_speed_m_s
        delivery_s = cfg.delivery_session_s
        cost = transit_s + delivery_s
        if cost > self._budget_remaining:
            return Exp3DeliveryStepResult(
                contact=contact,
                transit_s=transit_s,
                transit_distance_m=transit_dist,
                delivery_s=delivery_s,
                per_device_delivered=[False] * len(contact.devices),
            )

        per_device_delivered: List[bool] = []
        for did in contact.devices:
            dev = next((d for d in self._devices if d.device_id == did), None)
            if dev is None:
                per_device_delivered.append(False)
                continue
            d_dist = _euclid(contact.position, dev.pos)
            rf_factor = max(0.4, 1.0 - d_dist / (3.0 * cfg.world_radius))
            p_deliver = max(0.0, min(1.0, dev.reliability * rf_factor))
            per_device_delivered.append(self._rng.random() < p_deliver)

        self._mule_pose = contact.position
        self._budget_remaining = max(0.0, self._budget_remaining - cost)
        self._now += cost
        self._mule_energy = max(0.0, self._mule_energy - transit_dist * ENERGY_W)
        self._episode_metrics.pass2_transit_distance_m += transit_dist
        self._episode_metrics.pass2_transit_time_s += transit_s
        self._episode_metrics.pass2_delivery_time_s += delivery_s
        self._episode_metrics.pass2_contacts_visited += 1
        self._episode_metrics.pass2_devices_reached += sum(per_device_delivered)
        self._episode_metrics.energy_total += transit_dist * ENERGY_W

        return Exp3DeliveryStepResult(
            contact=contact,
            transit_s=transit_s,
            transit_distance_m=transit_dist,
            delivery_s=delivery_s,
            per_device_delivered=per_device_delivered,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _nearest_bs(self, position: Position) -> Position:
        return min(
            self._cfg.base_station_positions,
            key=lambda b: _euclid(position, b),
        )

    def _upload_rate_at(self, position: Position) -> float:
        """Time-varying per-channel upload rate at ``position``.

        Model: nominal rate scaled by an inverse-distance falloff to the
        nearest BS, then perturbed by ±``upload_jitter_pct`` Gaussian
        noise. The falloff prevents a contact deep in the field from
        getting the same throughput as one parked next to a BS — that's
        what makes upload_s a meaningful pressure on the schedule.
        """
        cfg = self._cfg
        nearest = self._nearest_bs(position)
        d = _euclid(position, nearest)
        # 1.0 at the BS, 0.4 at world-diameter distance — keeps the
        # rate inside [0.4·nominal, nominal].
        falloff = max(0.4, 1.0 - d / (3.0 * cfg.world_radius))
        jitter = float(self._rng.normal(1.0, cfg.upload_jitter_pct))
        jitter = max(0.1, jitter)  # never quite zero
        return cfg.nominal_upload_bps * falloff * jitter


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _euclid(a: Position, b: Position) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _bs_centroid(positions: Tuple[Position, ...]) -> Position:
    """Geometric centroid of the configured BS array.

    The mule docks at the cluster's edge server, which the simulator
    represents as the centroid of the BS positions — symmetric across
    all BSes so the first-leg geometry doesn't bias toward any one.
    """
    if not positions:
        raise ValueError("base_station_positions must not be empty")
    n = len(positions)
    return (
        sum(p[0] for p in positions) / n,
        sum(p[1] for p in positions) / n,
        sum(p[2] for p in positions) / n,
    )
