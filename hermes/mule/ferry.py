"""The mule's ferry mode: the mission clock and the contact link, wired for one mule.

FeRRy Phase 3 (design sections 2.2-2.3, 3.3-3.7, 4.2 and 4.6). With
``MuleConfig.mission_clock == "sim"`` the mule runs every mission on a
simulated :class:`~hermes.l1.mission_clock.MissionClock` instead of the wall
clock: flight legs, contact airtime, missed replies, the backhaul upload and
the dock turnaround are charged to it, and every mission-time stamp (plans,
deadlines, the S3b budget, contact outcomes) comes from it. This module holds
the glue so that ``mule_main.py`` stays small and the physics stays out of the
scheduler and the mission server:

* :class:`FerrySpec`: the configuration, immutable and built from plain values
  (:meth:`FerrySpec.from_config`, which unit U7 calls from ``MuleConfig``):
  the contact band, the contact link (unit U3) and channel (unit U2), the
  seconds-axis backhaul and its carrier policy, the flight and energy model
  (unit U1), the payload model, the response to a remainder that stops
  fitting, the contact-reliability source and the ground-truth availability
  map, and what ``Deadline(j)`` bounds.
* :class:`PayloadModel`: measured or declared bytes per pass (decision D3).
* :class:`FerryRuntime`: one per supervisor. It builds the scheduler's
  :class:`~hermes.scheduler.stages.s3b_feasibility.FerryPhysics` (the
  predicted dwell, return leg, upload and energy), annotates planned
  waypoints, observes the channel at arrival, builds each stop's
  :class:`~hermes.mission.contact_plan.ContactPlan`, draws the keyed
  availability, builds the L1 state, and charges the backhaul upload.

**Why the physics is wired here.** The scheduler must not import ``hermes.l1``
(it receives floats and callables, design section 3.1) and neither may the
mission package (``contact_plan``'s docstring). The mule is the one place that
knows both the radio model and the plan, so it binds the link's numbers into
callables for each of them.

**The channel-free control (critic A1).** ``FerrySpec()`` with no band is the
mission clock without a contact link: every contact costs the cost model's
``session_time_s`` once (plus the listen window when a reply is missing), the
planner prices it the same way, and no SNR, rate or availability draw exists.
It is what re-runs a study "on the simulated clock, channel-free".

**Ground truth stays out of decision code (critic B16).** The availability map
``{device_id: rel_i}`` is used only for the keyed availability draw at a
Pass-1 contact (design section 4.8, spec Q8). It is kept out of ``repr``,
equality and :meth:`FerrySpec.describe`, and nothing here hands it to the
scheduler, a policy or the L1 state: the scheduler sees only the TIMEOUT that
a failed draw produces, as it would for any failed uplink.

**Below the SNR floor (critic B12).** A rate of 0 is never charged as an
infinite time. A contact member below the floor at arrival is unreachable and
costs no airtime (the contact plan's gate). A backhaul upload whose SNR is
below the floor is a lost upload, and the mule charges the time the same
upload would take at the floor rate (CQI 1), the longest a successful upload
can take, before it gives up (:meth:`FerryRuntime.upload_cap_s`). The planner
prices a carrier whose mean is below the floor the same way.

The module imports nothing from ``experiments/`` (finding A-01).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import numbers
import statistics
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hermes.l1.channel_ddqn import L1_STATE_DIM
from hermes.l1.channel_model import (
    SALT_AVAILABILITY,
    SALT_BACKHAUL,
    SALT_CONTACT,
    BackhaulChannel,
    ContactChannel,
    backhaul_period_s,
    ferry_salt,
    keyed_uniform,
    loss_from_snr,
)
from hermes.l1.contact_link import SNR_FLOOR_DB, ContactLink
from hermes.l1.mission_clock import EnergyModel, FlightModel
from hermes.l1.rf_prior import RFPriorProducer
from hermes.mission.contact_plan import ContactPlan, planar_distance_m
from hermes.scheduler.routing.replan import FALLBACK_REORDER, FALLBACKS
from hermes.scheduler.stages.s3b_feasibility import (
    DEADLINE_BOUNDS,
    DEADLINE_BOUNDS_COLLECTION,
    FeasibilityModel,
    FerryPhysics,
)
from hermes.types import ContactWaypoint, DeviceID, MissionPass, weights_byte_count
from hermes.types.bundles import BackhaulUpload

log = logging.getLogger(__name__)

Pose = Tuple[float, float, float]

#: ``MuleConfig.in_flight_response``: what the mule does when the rest of a
#: pass stops fitting. ``abort`` is Freeze Amendment 8's rule (the next stop is
#: re-checked and the tail abandoned when it fails), priced on the mission
#: clock; ``replan`` checks the whole remainder at every departure and repairs
#: it through ``FLScheduler.replan_remainder`` (design section 3.4).
RESPONSE_ABORT = "abort"
RESPONSE_REPLAN = "replan"
RESPONSES: Tuple[str, ...] = (RESPONSE_ABORT, RESPONSE_REPLAN)

#: ``MuleConfig.contact_reliability_source``. ``origin``: the device's own
#: seeded draw of ``rel x rf_factor`` (the recorded model, EX-4.2), so the
#: devices keep their ``DeviceConfig.contact_reliability``. ``channel``: the SNR
#: gate at the stop plus the availability ``rel_i`` drawn on the mule, keyed by
#: (seed, device, round) (design section 4.8, spec Q8); only then are the
#: devices built with ``contact_reliability=None``, so the failure is not drawn
#: twice. Nulling it under ``origin`` would remove every contact failure.
RELIABILITY_ORIGIN = "origin"
RELIABILITY_CHANNEL = "channel"
RELIABILITY_SOURCES: Tuple[str, ...] = (RELIABILITY_ORIGIN, RELIABILITY_CHANNEL)

#: The backhaul carrier policy. ``fixed``: ``argmax_c g_c``, the noise-free
#: long-run best carrier (every arm but H3). ``adaptive``: H3's
#: ``AdaptiveChannelController`` at every upload, holding its carrier across
#: missions (design section 1 D2).
BACKHAUL_FIXED = "fixed"
BACKHAUL_ADAPTIVE = "adaptive"
BACKHAUL_POLICIES: Tuple[str, ...] = (BACKHAUL_FIXED, BACKHAUL_ADAPTIVE)

#: ``MuleConfig.backhaul_model``: ``mission`` is the recorded schedule by
#: mission round (the cluster's, Amendment 5), with no simulated upload time;
#: ``seconds`` is the seconds-axis :class:`BackhaulChannel` at the dock.
BACKHAUL_MODEL_MISSION = "mission"
BACKHAUL_MODEL_SECONDS = "seconds"
BACKHAUL_MODELS: Tuple[str, ...] = (BACKHAUL_MODEL_MISSION, BACKHAUL_MODEL_SECONDS)


def _int_or_none(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int or None, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return int(value)


def _choice(value: Any, allowed: Sequence[str], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(allowed)}, got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# The payload model (decision D3)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PayloadModel:
    """What a contact's airtime and the backhaul upload are priced for.

    ``payload_bytes`` None (the default) prices the bytes the mule measures:
    a Pass-1 session is the push (θ plus the synthetic batch) and the update
    (θ-sized), a Pass-2 session is the push, the upload is the partial
    aggregate. With the canonical 21-input model that is 37,576 B, 18,820 B
    and about 18,756 B (design section 0, finding 1). An integer declares a
    payload per direction instead (D3's 1 MB and 10 MB): a Pass-1 session is
    priced for two directions, a Pass-2 session and the upload for one. The
    real θ still crosses the link; only the charge changes. These are the
    host's own rules (``ContactPlan.session_bytes``), so the planner predicts
    exactly what the commit charges when the channel meets its mean.

    Stub runs carry a 52 B θ, so without a declared payload their airtime is
    negligible (design section 1 D3).
    """

    payload_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload_bytes", _int_or_none(self.payload_bytes, "payload_bytes"))

    @property
    def declared(self) -> bool:
        return self.payload_bytes is not None

    def session_bytes(self, pass_kind: MissionPass, *, push_bytes: int, update_bytes: int = 0) -> int:
        """A session's priced bytes: Pass 1 push + update, Pass 2 the push."""
        if MissionPass(pass_kind) is MissionPass.COLLECT:
            if self.declared:
                return 2 * int(self.payload_bytes)  # type: ignore[arg-type]
            return int(push_bytes) + int(update_bytes)
        if self.declared:
            return int(self.payload_bytes)  # type: ignore[arg-type]
        return int(push_bytes)

    def upload_bytes(self, measured: int) -> int:
        """The UP's priced bytes: the partial's own, or the declared payload.

        An empty partial (``dock_on_empty``) carries no model, so it is 0
        bytes either way: a declared payload replaces the size of a model that
        is sent, it does not invent one.
        """
        measured = int(measured)
        if self.declared and measured > 0:
            return int(self.payload_bytes)  # type: ignore[arg-type]
        return measured

    def describe(self) -> Dict[str, Any]:
        return {
            "payload_bytes": self.payload_bytes,
            "mode": "declared" if self.declared else "measured",
        }


# --------------------------------------------------------------------------- #
# The configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class FerrySpec:
    """One mule's ferry-mode configuration (immutable; equal only to itself,
    since the channels it holds have no value equality).

    * ``band``: the contact band class every stop flies (``"wide"``,
      ``"medium"``, ``"narrow"``; Phase 3 re-baselines fly ``wide``), or None
      for the channel-free control (critic A1). A band needs ``link`` and
      ``contact_channel``.
    * ``link``: unit U3's :class:`ContactLink`; its anchor must be the run's
      ``rf_range_m`` (``R_planar(wide) == rf_range_m``). Also prices the
      backhaul upload, on the anchor class's rate table (design section 2.3:
      ``8 * UP bytes / rate(wide, SNR_bh(t))``).
    * ``contact_channel``: unit U2's :class:`ContactChannel` on the link.
    * ``backhaul``: unit U2's seconds-axis :class:`BackhaulChannel`, or None
      for the recorded ``mission`` model, whose losses are the cluster's
      schedule by mission round and which has no SNR over time.
    * ``mission_upload_snr_db``: under the ``mission`` model only (no
      ``backhaul``), the SNR the upload is TIMED at: the upload is charged
      ``8 * UP bytes / rate(anchor class, this SNR)`` on the clock and the
      planner prices the same, but the mule records no backhaul outcome, since
      the loss is still the cluster's recorded schedule. :meth:`from_config`
      sets it to the fixed carrier's noise-free mean, ``base + max_c g_c`` of
      the seconds-axis model for the trial's seed and regime (decision taken
      on unit U6's open question 4). None (the default, and a hand-built
      spec) charges no upload time, as before.
    * ``backhaul_policy``: ``fixed`` or ``adaptive`` (H3).
    * ``flight``: cruise speed, dock, turnaround, listen window and energy.
    * ``payload``: measured or declared bytes (:class:`PayloadModel`).
    * ``in_flight_response``: ``abort`` or ``replan``; ``replan_fallback``:
      ``reorder`` or ``trim`` (unit U4; chosen at the pilot, critic B5).
    * ``reliability_source``: ``origin`` or ``channel``; ``channel`` needs a
      band (critic B16) and ``availability_salt``.
    * ``availability``: the ground-truth ``{device_id: rel_i}``, for the
      keyed draw only (see the module docstring).
    * ``deadline_bounds``: what Deadline(j) bounds (spec Q2; see
      :data:`~hermes.scheduler.stages.s3b_feasibility.DEADLINE_BOUNDS`):
      ``collection`` (the default: arrival + dwell), ``delivery_per_stop``
      (the plan's single-contact predicate: each stop's own return and
      upload must meet its deadline; ``delivery`` at ef1faa1, renamed) or
      ``delivery`` (route-level: every update collected must reach the
      cluster, the route's landing plus the upload, by its own deadline).
      ``delivery`` changed meaning after ef1faa1 with no shim, as no
      recorded run or committed trace used it.
    """

    band: Optional[str] = None
    link: Optional[ContactLink] = None
    contact_channel: Optional[ContactChannel] = None
    backhaul: Optional[BackhaulChannel] = None
    backhaul_policy: str = BACKHAUL_FIXED
    flight: FlightModel = field(default_factory=FlightModel)
    payload: PayloadModel = field(default_factory=PayloadModel)
    in_flight_response: str = RESPONSE_ABORT
    replan_fallback: str = FALLBACK_REORDER
    reliability_source: str = RELIABILITY_ORIGIN
    availability: Mapping[str, float] = field(default_factory=dict, repr=False)
    availability_salt: Optional[int] = field(default=None, repr=False)
    deadline_bounds: str = DEADLINE_BOUNDS_COLLECTION
    mission_upload_snr_db: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mission_upload_snr_db is not None:
            snr = self.mission_upload_snr_db
            if isinstance(snr, bool) or not isinstance(snr, numbers.Real) or not math.isfinite(snr):
                raise ValueError(f"mission_upload_snr_db must be a finite number, got {snr!r}")
            if self.backhaul is not None:
                raise ValueError(
                    "mission_upload_snr_db times the upload under the recorded 'mission' "
                    "backhaul; with a seconds-axis backhaul the channel prices it"
                )
            if self.link is None:
                raise ValueError(
                    "mission_upload_snr_db needs the ContactLink: the upload is priced "
                    "on the anchor class's rate table"
                )
            object.__setattr__(self, "mission_upload_snr_db", float(snr))
        if not isinstance(self.flight, FlightModel):
            raise TypeError(f"flight must be a FlightModel, got {type(self.flight).__name__}")
        if not isinstance(self.payload, PayloadModel):
            raise TypeError(f"payload must be a PayloadModel, got {type(self.payload).__name__}")
        _choice(self.in_flight_response, RESPONSES, "in_flight_response")
        _choice(self.replan_fallback, FALLBACKS, "replan_fallback")
        _choice(self.reliability_source, RELIABILITY_SOURCES, "reliability_source")
        _choice(self.backhaul_policy, BACKHAUL_POLICIES, "backhaul_policy")
        _choice(self.deadline_bounds, DEADLINE_BOUNDS, "deadline_bounds")
        if self.link is not None and not isinstance(self.link, ContactLink):
            raise TypeError(f"link must be a ContactLink, got {type(self.link).__name__}")

        if self.band is None:
            if self.contact_channel is not None:
                raise ValueError(
                    "a contact channel needs a contact band; the channel-free "
                    "control (band None) has no SNR"
                )
        else:
            if self.link is None:
                raise ValueError(f"contact band {self.band!r} needs a ContactLink")
            self.link.index(self.band)                      # raises on an unknown class
            if self.contact_channel is None:
                raise ValueError(f"contact band {self.band!r} needs a ContactChannel")
            self.contact_channel.band_index(self.band)      # the channel knows the class
            if tuple(self.contact_channel.bands) != tuple(self.link.names):
                raise ValueError(
                    f"the channel's classes {tuple(self.contact_channel.bands)} are not "
                    f"the link's {tuple(self.link.names)}"
                )
        if self.backhaul is not None and self.link is None:
            raise ValueError(
                "a seconds-axis backhaul needs the ContactLink: the upload is priced "
                "on the anchor class's rate table"
            )

        if self.reliability_source == RELIABILITY_CHANNEL:
            if self.band is None:
                raise ValueError(
                    "contact_reliability_source='channel' needs a contact band: the "
                    "reliability then comes from the SNR gate (critic B16)"
                )
            if self.availability_salt is None:
                raise ValueError("contact_reliability_source='channel' needs availability_salt")
        if self.availability_salt is not None:
            if isinstance(self.availability_salt, bool) or not isinstance(
                self.availability_salt, numbers.Integral
            ):
                raise TypeError(f"availability_salt must be an int, got {self.availability_salt!r}")
        avail: Dict[str, float] = {}
        for did, rel in dict(self.availability or {}).items():
            if not isinstance(did, str):
                raise TypeError(f"availability keys are device ids (str), got {did!r}")
            if isinstance(rel, bool) or not isinstance(rel, numbers.Real):
                raise TypeError(f"availability[{did!r}] must be a number, got {rel!r}")
            r = float(rel)
            if not (0.0 <= r <= 1.0):
                raise ValueError(f"availability[{did!r}] must lie in [0, 1], got {rel!r}")
            avail[did] = r
        object.__setattr__(self, "availability", MappingProxyType(avail))

    # ------------------------------------------------------------------ #
    # Construction from plain values (unit U7 calls this from MuleConfig)
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        *,
        rf_range_m: float,
        seed: int,
        contact_band: Optional[str] = None,
        in_flight_response: str = RESPONSE_ABORT,
        replan_fallback: str = FALLBACK_REORDER,
        backhaul_model: str = BACKHAUL_MODEL_MISSION,
        backhaul_policy: str = BACKHAUL_FIXED,
        backhaul_regime: str = "clean",
        backhaul_period: Optional[float] = None,
        n_missions: Optional[int] = None,
        t_nom_s: Optional[float] = None,
        contact_reliability_source: str = RELIABILITY_ORIGIN,
        device_availability: Optional[Mapping[str, float]] = None,
        payload_bytes: Optional[int] = None,
        snr_floor_db: float = SNR_FLOOR_DB,
        altitude_m: float = 25.0,
        n_pl: float = 2.2,
        shadow_sigma_db: float = 4.0,
        margin_quantile: float = 0.9,
        band_classes: Optional[Sequence[Any]] = None,
        contact_regime: str = "clean",
        interference_period_s: float = 60.0,
        noise_bin_s: float = 1.0,
        shadow_corr_s: float = 7.4,
        shadow_keying: str = "time",
        cruise_speed_m_s: float = 5.0,
        turnaround_s: float = 30.0,
        listen_s: float = 1.0,
        energy_capacity_j: Optional[float] = None,
        p_move_w: Optional[float] = None,
        p_hover_w: Optional[float] = None,
        deadline_bounds: str = DEADLINE_BOUNDS_COLLECTION,
    ) -> "FerrySpec":
        """A spec from plain values; every default is the design's.

        ``seed`` is the trial seed: the contact channel, the backhaul and the
        availability draw take their salts from it (``ferry_salt``), so two
        arms of one trial see the same channel (paired by construction).
        ``backhaul_model="seconds"`` needs the backhaul period ``P_bh``, given
        directly (``backhaul_period``) or as ``n_missions * t_nom_s``. Under
        the recorded ``backhaul_model="mission"`` the upload is still charged
        on the clock (timing only): at ``mission_upload_snr_db``, the fixed
        carrier's noise-free mean ``base + max_c g_c`` of the seconds-axis
        model for this seed and ``backhaul_regime``, the same for every arm (H3
        included); the losses stay the cluster's recorded schedule.
        ``p_move_w`` / ``p_hover_w`` override the Zeng model's powers at the
        cruise speed; ``energy_capacity_j`` switches the simulated energy
        clause on (it binds only with a budget).

        ``contact_reliability_source="channel"`` keeps ``device_availability``
        for the mule's keyed draw, and the cell's devices must then be built
        with ``contact_reliability=None``. Under ``origin`` (the default, and
        the channel-free control) the devices keep their own draw, and the
        availability map is dropped here.

        **T_nom before P_bh.** A seconds-axis cell needs ``T_nom`` for its
        period, and ``T_nom`` is computed from a spec (:meth:`feasibility_model`
        with ``fl_scheduler.median_nominal_mission_period_s``). Build that
        first spec with any placeholder ``backhaul_period`` and the cell's own
        seed and ``backhaul_regime``: the planner prices the upload on the
        carrier means ``base + g_c`` alone (``BackhaulChannel.pred_snr_db``),
        which do not depend on the period. Then build the cell's spec with
        ``n_missions`` and ``t_nom_s``.
        """
        _choice(backhaul_model, BACKHAUL_MODELS, "backhaul_model")
        classes = {} if band_classes is None else {"classes": tuple(band_classes)}
        link = ContactLink(
            anchor_planar_m=float(rf_range_m), snr_floor_db=snr_floor_db, altitude_m=altitude_m,
            n_pl=n_pl, shadow_sigma_db=shadow_sigma_db, margin_quantile=margin_quantile,
            **classes,
        )
        channel = None
        if contact_band is not None:
            channel = ContactChannel.from_link(
                link, salt=ferry_salt(seed, SALT_CONTACT), regime=contact_regime,
                interference_period_s=interference_period_s, noise_bin_s=noise_bin_s,
                shadow_corr_s=shadow_corr_s, shadow_keying=shadow_keying,
            )
        backhaul = None
        mission_upload_snr_db = None
        if backhaul_model == BACKHAUL_MODEL_SECONDS:
            if backhaul_period is None:
                if n_missions is None or t_nom_s is None:
                    raise ValueError(
                        "backhaul_model='seconds' needs the backhaul period P_bh: give "
                        "backhaul_period or both n_missions and t_nom_s"
                    )
                backhaul_period = backhaul_period_s(n_missions, t_nom_s)
            backhaul = BackhaulChannel(
                salt=ferry_salt(seed, SALT_BACKHAUL), period_s=backhaul_period,
                regime=backhaul_regime, noise_bin_s=noise_bin_s,
            )
        else:
            # The recorded model has no SNR over time; the upload is timed at
            # the fixed carrier's noise-free mean. ``pred_snr_db`` does not
            # read the period, so any positive placeholder gives the same value.
            timing = BackhaulChannel(
                salt=ferry_salt(seed, SALT_BACKHAUL), period_s=1.0,
                regime=backhaul_regime, noise_bin_s=noise_bin_s,
            )
            mission_upload_snr_db = timing.pred_snr_db(timing.fixed_band())
        energy = EnergyModel.at_speed(cruise_speed_m_s, capacity_j=energy_capacity_j)
        overrides = {k: v for k, v in (("p_move_w", p_move_w), ("p_hover_w", p_hover_w))
                     if v is not None}
        if overrides:
            energy = dataclasses.replace(energy, **overrides)
        flight = FlightModel(
            cruise_speed_m_s=cruise_speed_m_s, turnaround_s=turnaround_s, listen_s=listen_s,
            energy=energy,
        )
        channel_reliability = contact_reliability_source == RELIABILITY_CHANNEL
        return cls(
            band=contact_band,
            link=link,
            contact_channel=channel,
            backhaul=backhaul,
            backhaul_policy=backhaul_policy,
            flight=flight,
            payload=PayloadModel(payload_bytes),
            in_flight_response=in_flight_response,
            replan_fallback=replan_fallback,
            reliability_source=contact_reliability_source,
            availability=dict(device_availability or {}) if channel_reliability else {},
            availability_salt=ferry_salt(seed, SALT_AVAILABILITY) if channel_reliability else None,
            deadline_bounds=deadline_bounds,
            mission_upload_snr_db=mission_upload_snr_db,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def banded(self) -> bool:
        """True with a contact band; False for the channel-free control."""
        return self.band is not None

    def feasibility_model(
        self,
        *,
        rf_range_m: float,
        theta_bytes: int,
        synth_bytes: int = 0,
        base: Optional[FeasibilityModel] = None,
    ) -> FeasibilityModel:
        """A planning model with this spec's physics and a fixed payload.

        For callers without a mule, such as the driver's T_nom helper
        (``fl_scheduler.median_nominal_mission_period_s``, spec Q1): the
        measured payload is that of a θ of ``theta_bytes`` pushed with a
        ``synth_bytes`` synthetic batch, and the held backhaul carrier is the
        fixed one. The mule itself plans with :meth:`FerryRuntime.feasibility_model`,
        which follows the θ it carries.

        The model never reads the backhaul period (the predicted upload uses
        the carrier means only), so for a seconds-axis cell a spec built with a
        placeholder ``backhaul_period`` gives the same T_nom as the cell's own
        (see :meth:`from_config`).
        """
        rt = FerryRuntime(self, None, rf_range_m=rf_range_m,
                          session_time_s=(base or FeasibilityModel()).session_time_s)
        rt.set_payload(theta_bytes=theta_bytes, synth_bytes=synth_bytes)
        return rt.feasibility_model(base)

    def describe(self) -> Dict[str, Any]:
        """The settings as a JSON-ready dict, for ``mule_ready`` (design section 2.5).

        The availability map is left out on purpose (critic B16); only how
        many devices it covers is reported.
        """
        energy = self.flight.energy
        return {
            "contact_band": self.band,
            "band_classes": None if self.link is None else self.link.describe(),
            "channel_params": {
                "contact": None if self.contact_channel is None else self.contact_channel.describe(),
                "backhaul": None if self.backhaul is None else self.backhaul.describe(),
            },
            "backhaul_model": (BACKHAUL_MODEL_SECONDS if self.backhaul is not None
                               else BACKHAUL_MODEL_MISSION),
            "backhaul_policy": self.backhaul_policy if self.backhaul is not None else None,
            # Under the recorded mission model: the SNR the upload is timed at
            # (None: the upload is not charged). Absent a seconds backhaul only.
            "backhaul_timing_snr_db": self.mission_upload_snr_db,
            "in_flight_response": self.in_flight_response,
            "replan_fallback": self.replan_fallback,
            "contact_reliability_source": self.reliability_source,
            "device_availability_n": len(self.availability),
            "payload": self.payload.describe(),
            "energy_params": {
                "p_move_w": energy.p_move_w,
                "p_hover_w": energy.p_hover_w,
                "capacity_j": energy.capacity_j,
                "speed_m_s": energy.speed_m_s,
                "status": energy.status,
            },
            "cruise_speed_m_s": self.flight.cruise_speed_m_s,
            "dock": list(self.flight.dock),
            "dock_turnaround_s": self.flight.turnaround_s,
            "listen_s": self.flight.listen_s,
            "deadline_bounds": self.deadline_bounds,
        }


# --------------------------------------------------------------------------- #
# What the mule sees at a stop
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StopObservation:
    """The channel at one stop when the mule arrives (design section 4.6).

    ``distances_m`` are the members' planar distances to the stop (S3a's
    metric), in member order. With a band, ``snr_db`` is each member's
    realized SNR on the contact band at ``t_s`` (the same values the contact
    plan gates on), ``class_snr_db`` the median realized SNR over the members
    on every class of the link, in class order, and ``max_slant_m`` the
    largest member slant distance. All three are None without a band.
    """

    t_s: float
    devices: Tuple[DeviceID, ...]
    distances_m: Tuple[float, ...]
    snr_db: Optional[Tuple[float, ...]] = None
    class_snr_db: Optional[Tuple[float, ...]] = None
    max_slant_m: Optional[float] = None


# --------------------------------------------------------------------------- #
# The runtime
# --------------------------------------------------------------------------- #

class FerryRuntime:
    """One mule's ferry state and the builders bound to it.

    ``clock`` is the mule's :class:`~hermes.l1.mission_clock.MissionClock`
    (None only for planning-only use, :meth:`FerrySpec.feasibility_model`).
    The runtime holds what changes during a trial: the payload the mule
    carries (the θ of the current pass), the backhaul carrier H3 holds across
    missions, and the causal RF prior fed by each upload (critic B4).

    Only the supervisor thread uses it.
    """

    def __init__(
        self,
        spec: FerrySpec,
        clock,
        *,
        rf_range_m: float,
        session_time_s: float = 1.0,
    ) -> None:
        if not isinstance(spec, FerrySpec):
            raise TypeError(f"spec must be a FerrySpec, got {type(spec).__name__}")
        rng = float(rf_range_m)
        if not (math.isfinite(rng) and rng > 0.0):
            raise ValueError(f"rf_range_m must be finite and > 0, got {rf_range_m!r}")
        link = spec.link
        if link is not None and link.anchor_planar_m != rng:
            raise ValueError(
                f"the ContactLink is anchored at {link.anchor_planar_m!r} m but the run's "
                f"rf_range_m is {rng!r}: R_planar(wide) must be rf_range_m"
            )
        self.spec = spec
        self.clock = clock
        self.rf_range_m = rng
        self.session_time_s = float(session_time_s)
        #: S3a's radius and the contact gate's range: R_planar(band), exactly
        #: ``rf_range_m`` at wide, and ``rf_range_m`` itself without a band.
        self.range_planar_m: float = rng if spec.band is None else link.range_planar_m(spec.band)  # type: ignore[union-attr]
        self.band_index: Optional[int] = None if spec.band is None else link.index(spec.band)  # type: ignore[union-attr]
        self._theta_bytes = 0
        self._synth_bytes = 0
        #: The backhaul carrier held since the last upload (-1 before the first).
        self.carrier: int = -1
        self.rf_prior: Optional[RFPriorProducer] = (
            RFPriorProducer() if spec.backhaul is not None else None
        )
        self._warned_missing: set = set()

    # ------------------------------------------------------------------ #
    # Payload
    # ------------------------------------------------------------------ #

    @property
    def banded(self) -> bool:
        return self.spec.band is not None

    def set_payload(self, *, theta_bytes: int, synth_bytes: int = 0) -> None:
        self._theta_bytes = int(theta_bytes)
        self._synth_bytes = int(synth_bytes)

    def observe_payload(self, theta, synth_batch) -> None:
        """Measure the θ (and synthetic batch) the coming pass pushes."""
        synth = sum(int(np.asarray(a).nbytes) for a in (synth_batch or ()))
        self.set_payload(theta_bytes=weights_byte_count(theta or []), synth_bytes=synth)

    def session_bytes(self, pass_kind: MissionPass) -> int:
        """The predicted bytes of one member's session in ``pass_kind``.

        The host's own count: push = θ + synthetic batch, update = θ-sized.
        """
        push = self._theta_bytes + self._synth_bytes
        return self.spec.payload.session_bytes(
            pass_kind, push_bytes=push, update_bytes=self._theta_bytes,
        )

    def predicted_upload_bytes(self) -> int:
        """The UP's predicted bytes: a θ-sized partial, or the declared payload."""
        return self.spec.payload.upload_bytes(self._theta_bytes)

    # ------------------------------------------------------------------ #
    # The scheduler's physics
    # ------------------------------------------------------------------ #

    def member_dwell_s(
        self, d_planar: float, pass_kind: MissionPass, snr_offset_db: float,
    ) -> Optional[float]:
        """One member's predicted airtime at the mean SNR (plus δ_obs); None below the floor."""
        link, band = self.spec.link, self.spec.band
        snr = link.mean_snr_db(band, d_planar) + float(snr_offset_db)  # type: ignore[union-attr]
        return link.dwell_s(self.session_bytes(pass_kind), band, snr)  # type: ignore[union-attr]

    def held_carrier(self) -> Optional[int]:
        """The carrier the planner prices the upload on (None without a backhaul)."""
        bh = self.spec.backhaul
        if bh is None:
            return None
        if self.spec.backhaul_policy == BACKHAUL_ADAPTIVE and self.carrier >= 0:
            return self.carrier
        return bh.fixed_band()

    def upload_cap_s(self, nbytes: int) -> float:
        """What a lost below-floor upload costs: the floor-rate time (critic B12).

        The link carries nothing below the floor, so the upload fails; the
        mule is charged what the same upload takes at the floor rate (CQI 1),
        the longest a successful upload can take, before it gives up. Finite
        for every payload, and 0 for an empty partial.
        """
        link = self.spec.link
        cap = link.dwell_s(int(nbytes), link.anchor_class, link.snr_floor_db)  # type: ignore[union-attr]
        assert cap is not None  # at the floor the rate is CQI 1's, never 0
        return cap

    def _upload_time_at(self, snr_db: float, nbytes: int) -> Tuple[float, bool]:
        link = self.spec.link
        rate = link.rate_bps(link.anchor_class, snr_db)  # type: ignore[union-attr]
        if rate > 0.0:
            return 8.0 * int(nbytes) / rate, False
        return self.upload_cap_s(nbytes), True

    def predicted_upload_s(self) -> float:
        """The planner's upload: bytes at the held carrier's mean SNR, no noise.

        Causal: the mean ``base + g_c`` is known before any upload. Under the
        recorded ``mission`` model the bytes are priced at the spec's
        ``mission_upload_snr_db`` (the same timing the clock is charged), and
        at 0 without one.
        """
        bh = self.spec.backhaul
        if bh is None:
            snr = self.spec.mission_upload_snr_db
            if snr is None:
                return 0.0
            dt, _ = self._upload_time_at(snr, self.predicted_upload_bytes())
            return dt
        carrier = self.held_carrier()
        dt, _ = self._upload_time_at(bh.pred_snr_db(carrier), self.predicted_upload_bytes())
        return dt

    def physics(self) -> FerryPhysics:
        """The :class:`FerryPhysics` the scheduler prices with (design section 3.1).

        Member positions are looked up by the scheduler's device states
        (``FLScheduler`` binds its own map, critic B11). Without a band the
        physics has no member dwell and a contact costs ``session_time_s``
        (critic A1).
        """
        energy = self.spec.flight.energy
        return FerryPhysics(
            dock=self.spec.flight.dock,
            member_dwell_s=self.member_dwell_s if self.banded else None,
            upload_s=self.predicted_upload_s,
            p_move_w=energy.p_move_w,
            p_hover_w=energy.p_hover_w,
            energy_capacity_j=energy.capacity_j,
            deadline_bounds=self.spec.deadline_bounds,
            range_m=self.range_planar_m if self.banded else None,
        )

    def feasibility_model(self, base: Optional[FeasibilityModel] = None) -> FeasibilityModel:
        """``base`` (cruise speed and session time) with this runtime's physics.

        The clock charges legs with the flight model's speed, so the planner
        must use the same one; a base that already carries physics is refused.
        """
        base = base if base is not None else FeasibilityModel(
            cruise_speed_m_s=self.spec.flight.cruise_speed_m_s,
            session_time_s=self.session_time_s,
        )
        if getattr(base, "ferry", None) is not None:
            raise ValueError("the base feasibility model already carries ferry physics")
        if float(base.cruise_speed_m_s) != self.spec.flight.cruise_speed_m_s:
            raise ValueError(
                f"the feasibility model plans at {base.cruise_speed_m_s!r} m/s but the "
                f"flight model flies at {self.spec.flight.cruise_speed_m_s!r} m/s: the "
                "clock would charge other legs than the planner predicts"
            )
        return dataclasses.replace(base, ferry=self.physics())

    # ------------------------------------------------------------------ #
    # Stops
    # ------------------------------------------------------------------ #

    def annotate(
        self, queue: Sequence[ContactWaypoint], positions: Mapping[DeviceID, Sequence[float]],
    ) -> List[ContactWaypoint]:
        """The planned waypoints with ``band``, ``range_m`` and ``pred_snr_db`` (design section 4.5).

        Filled right after planning with ``dataclasses.replace``: the three
        fields are ``compare=False``, so an annotated waypoint equals and
        hashes like the plan's. ``pred_snr_db`` is each member's mean SNR on
        the band at its planar distance, in member order (None without a
        band). The mule flies the annotated objects from here on.
        """
        out: List[ContactWaypoint] = []
        for wp in queue:
            pred: Optional[Tuple[float, ...]] = None
            if self.banded:
                link, band = self.spec.link, self.spec.band
                pred = tuple(
                    link.mean_snr_db(band, planar_distance_m(wp.position, positions[d]))  # type: ignore[union-attr]
                    for d in wp.devices
                )
            out.append(dataclasses.replace(
                wp, band=self.spec.band, range_m=self.range_planar_m, pred_snr_db=pred,
            ))
        return out

    def observe(
        self, wp: ContactWaypoint, positions: Mapping[DeviceID, Sequence[float]], t_s: float,
    ) -> StopObservation:
        """The channel at ``wp`` at simulated time ``t_s`` (the arrival)."""
        stop = tuple(wp.position)
        devices = tuple(wp.devices)
        dist = tuple(planar_distance_m(stop, positions[d]) for d in devices)
        if not self.banded:
            return StopObservation(t_s=float(t_s), devices=devices, distances_m=dist)
        link, chan, band = self.spec.link, self.spec.contact_channel, self.spec.band
        snr = tuple(
            chan.snr_db(t_s, band, d, link_key=j, stop_pos=stop)  # type: ignore[union-attr]
            for j, d in zip(devices, dist)
        )
        per_class = tuple(
            float(statistics.median(
                chan.snr_db(t_s, name, d, link_key=j, stop_pos=stop)  # type: ignore[union-attr]
                for j, d in zip(devices, dist)
            ))
            for name in link.names  # type: ignore[union-attr]
        )
        slant = max(link.slant_m(d) for d in dist)  # type: ignore[union-attr]
        return StopObservation(
            t_s=float(t_s), devices=devices, distances_m=dist, snr_db=snr,
            class_snr_db=per_class, max_slant_m=slant,
        )

    def uplink_drops(self, members: Sequence[DeviceID], mission_round: int) -> FrozenSet[DeviceID]:
        """The members whose keyed availability draw fails this Pass-1 contact.

        ``u(salt_avail, j, mission_round) >= rel_j`` (design section 4.2, spec
        Q8), only with ``reliability_source == "channel"``. The draw is a pure
        function of (seed, device, round), so every arm of a trial draws the
        same outcome for the same device in the same mission. A member without
        an availability entry is always available (a warning is logged once).
        """
        spec = self.spec
        if spec.reliability_source != RELIABILITY_CHANNEL:
            return frozenset()
        out = set()
        for did in members:
            rel = spec.availability.get(str(did))
            if rel is None:
                if did not in self._warned_missing:
                    self._warned_missing.add(did)
                    log.warning("ferry: no availability for device %s; treated as available", did)
                continue
            if keyed_uniform(spec.availability_salt, str(did), int(mission_round)) >= rel:
                out.add(did)
        return frozenset(out)

    def contact_plan(
        self,
        wp: ContactWaypoint,
        positions: Mapping[DeviceID, Sequence[float]],
        *,
        pass_kind: MissionPass,
        mission_round: int,
    ) -> ContactPlan:
        """The stop's :class:`ContactPlan`, built at arrival (design section 4.2).

        ``arrival_ts`` is the clock now, so nothing may charge the clock
        between this call and the contact. With a band: the gate by
        R_planar(b) (S3a's metric) and the SNR floor at arrival, each target
        priced at its own session-start SNR (critic C2), the band's airtime,
        and in Pass 1 the uplink drops of the availability draw. Without a
        band: every member a target, ``session_time_s`` per contact.
        """
        if self.clock is None:
            raise ValueError("a planning-only FerryRuntime has no clock to build contact plans on")
        members = list(wp.devices)
        drops = (self.uplink_drops(members, mission_round)
                 if MissionPass(pass_kind) is MissionPass.COLLECT else frozenset())
        common = dict(
            clock=self.clock, advance=self.clock.advance, stop=tuple(wp.position),
            drop_uplink=drops, listen_s=self.spec.flight.listen_s,
            session_time_s=self.session_time_s,
            payload_bytes=self.spec.payload.payload_bytes,
        )
        if not self.banded:
            return ContactPlan.at_arrival(members, **common)
        link, chan, band = self.spec.link, self.spec.contact_channel, self.spec.band
        stop = tuple(wp.position)
        return ContactPlan.at_arrival(
            members,
            band=band,
            band_index=self.band_index,
            positions={d: tuple(positions[d]) for d in members},
            range_planar_m=self.range_planar_m,
            snr_fn=lambda j, d, t: chan.snr_db(t, band, d, link_key=j, stop_pos=stop),  # type: ignore[union-attr]
            snr_floor_db=link.snr_floor_db,  # type: ignore[union-attr]
            dwell_fn=lambda n, s: link.dwell_s(n, band, s),  # type: ignore[union-attr]
            **common,
        )

    def rates_bps(self, snr_db: Sequence[float]) -> Tuple[float, ...]:
        """The band's rate at each SNR (0 below the floor)."""
        link, band = self.spec.link, self.spec.band
        return tuple(link.rate_bps(band, s) for s in snr_db)  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # L1 state (design section 4.6)
    # ------------------------------------------------------------------ #

    def l1_state(
        self,
        obs: StopObservation,
        *,
        pose: Sequence[float],
        energy_j: float,
        budget_s: Optional[float],
    ) -> np.ndarray:
        """ChannelDDQN's state at a stop, from what the mule observed there.

        Slots 0-2: the median realized SNR over the members on wide, medium
        and narrow at arrival, / 30; slot 3: the largest member slant
        distance, / 100; slots 4-6: the pose, / 100; slot 7: ``1 - E/E_ref``
        with ``E`` the simulated energy spent this sortie (Pass 2 restarts
        at its takeoff, as the energy clause does; the caller passes it) and
        ``E_ref`` the capacity if one is set, else ``P_hover * budget`` if a
        budget is set, else the slot stays 1.0. Both references are per
        sortie, so the slot is the battery the energy clause sees. The
        choice is recorded, not acted on (ChannelDDQN's actions are backhaul
        carriers, and it has no trainer yet; that is Phase 5). Needs a band.
        """
        if obs.class_snr_db is None or obs.max_slant_m is None:
            raise ValueError("the L1 state needs the contact channel (a band)")
        state = np.zeros(L1_STATE_DIM, dtype=np.float32)
        for i, snr in enumerate(obs.class_snr_db[:3]):
            state[i] = float(snr) / 30.0
        state[3] = float(obs.max_slant_m) / 100.0
        state[4] = float(pose[0]) / 100.0
        state[5] = float(pose[1]) / 100.0
        state[6] = float(pose[2]) / 100.0
        energy = self.spec.flight.energy
        if energy.capacity_j is not None:
            e_ref: Optional[float] = energy.capacity_j
        elif budget_s is not None:
            e_ref = energy.p_hover_w * float(budget_s)
        else:
            e_ref = None
        state[7] = 1.0 if not e_ref else 1.0 - float(energy_j) / e_ref
        return state

    # ------------------------------------------------------------------ #
    # The dock
    # ------------------------------------------------------------------ #

    def charge_upload(self, measured_bytes: int) -> Optional[BackhaulUpload]:
        """Price the UP at the dock and charge it to the clock (design section 2.3).

        The carrier is the fixed one or H3's controller's pick at the upload
        time; the SNR is the backhaul's at that time; the charge is ``8 *
        bytes / rate(anchor class, SNR)``, or, below the floor, the capped
        lost upload (:meth:`upload_cap_s`), recorded with ``p_loss = 1`` since
        the link carried nothing. Otherwise ``p_loss = loss_from_snr(SNR)``,
        the model the cluster's keyed draw uses. The observed SNR feeds the
        causal RF prior.

        Under the recorded ``mission`` model (no seconds-axis backhaul) the
        upload is timed at the spec's ``mission_upload_snr_db`` and charged
        to the clock, but None is returned: the mule has no backhaul outcome
        to report, since the loss is the cluster's recorded schedule (the
        charge shows in the ledger's ``upload``). Nothing is charged without a
        timing SNR either (a hand-built spec).
        """
        bh = self.spec.backhaul
        if bh is None:
            snr = self.spec.mission_upload_snr_db
            if snr is not None:
                nbytes = self.spec.payload.upload_bytes(measured_bytes)
                dt, _ = self._upload_time_at(snr, nbytes)
                self.clock.advance(dt, "upload")
            return None
        clock = self.clock
        t0 = clock()
        carrier = bh.select_carrier(
            t0, adaptive=self.spec.backhaul_policy == BACKHAUL_ADAPTIVE, current=self.carrier,
        )
        self.carrier = int(carrier)
        snr = bh.snr_db(t0, carrier)
        nbytes = self.spec.payload.upload_bytes(measured_bytes)
        dt, below = self._upload_time_at(snr, nbytes)
        clock.advance(dt, "upload")
        if self.rf_prior is not None:
            self.rf_prior.observe_upload(carrier, snr, t0)
        return BackhaulUpload(
            carrier=int(carrier),
            snr_db=float(snr),
            p_loss=1.0 if below else float(loss_from_snr(snr)),
            t_start_s=float(t0),
            upload_s=float(dt),
            nbytes=int(nbytes),
            below_floor=bool(below),
        )

    def energy_j(self) -> float:
        """Simulated energy spent since takeoff, from the clock's ledger (SIMULATED)."""
        return self.spec.flight.energy.energy_j(self.clock.ledger())


def backhaul_record(up: Optional[BackhaulUpload]) -> Optional[Dict[str, Any]]:
    """``MissionRunResult.backhaul``: the upload as a JSON-ready dict (None without one)."""
    if up is None:
        return None
    return {
        "carrier": up.carrier,
        "snr_db": up.snr_db,
        "p_loss": up.p_loss,
        "t_upload_s": up.t_start_s + up.upload_s,
        "t_start_s": up.t_start_s,
        "upload_s": up.upload_s,
        "bytes": up.nbytes,
        "below_floor": up.below_floor,
    }


__all__ = [
    "BACKHAUL_ADAPTIVE",
    "BACKHAUL_FIXED",
    "BACKHAUL_MODELS",
    "BACKHAUL_MODEL_MISSION",
    "BACKHAUL_MODEL_SECONDS",
    "BACKHAUL_POLICIES",
    "FerryRuntime",
    "FerrySpec",
    "PayloadModel",
    "RELIABILITY_CHANNEL",
    "RELIABILITY_ORIGIN",
    "RELIABILITY_SOURCES",
    "RESPONSES",
    "RESPONSE_ABORT",
    "RESPONSE_REPLAN",
    "StopObservation",
    "backhaul_record",
]
