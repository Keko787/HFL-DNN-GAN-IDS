"""Sprint 2 — multi-process topology configuration.

One :class:`TopologyConfig` describes the AVN-shaped layout the
orchestrator brings up: 1 cluster, N mules, M devices, with all the
host:port pairs that wire them together. The orchestrator (chunk L)
serializes per-role configs to JSON files; each entry-point script
(``hermes.processes.{cluster,mule,device}``) reads its config from a
``--config`` arg and runs its service loop.

Maps onto AERPAW's AVN model 1:1:

* Cluster config → one fixed AVN running ``HFLHostCluster``.
* Mule config → one mobile AVN per mule, running ``MuleSupervisor``.
* Device config → one fixed or mobile AVN per device, running
  ``ClientMission``.

When AERPAW returns, the only thing that changes is the host strings
(localhost → AVN routable IPs); the rest of the wiring stays.

Schema is plain dataclasses with JSON helpers — no extra deps.

FeRRy Phase 3 — the mission clock and the contact link (design sections 1,
2.2 and 5.1). ``MuleConfig.mission_clock`` / ``ClusterConfig.mission_clock``
"wall" (the default) is every recorded run. "sim" flies every mission on a
simulated clock configured by the ferry fields of :class:`MuleConfig`, which
the mule turns into a ``hermes.mule.ferry.FerrySpec``
(:meth:`MuleConfig.ferry_spec_kwargs`), and puts the cluster's simulated-time
bookkeeping on (``ClusterConfig``). Every new field defaults to the recorded
behaviour, so old per-role JSON loads unchanged. Combinations that cannot run,
or would silently measure something else, are refused by
:func:`mule_config_errors`, :func:`cluster_config_errors` and
:meth:`TopologyConfig.validate` (critic B16). Several mules on the simulated
clock below a full quorum are served in simulated-time order by the cluster
(critic B9, unit U9: ``hermes.processes.cluster.SimOrderGate``).

FeRRy Phase 4 — the plan clock (the Phase 4 spec, other choices 10). The
``MuleConfig`` plan fields (:data:`PLAN_MULE_FIELDS`) configure it:
``plan_mode`` "legacy" (the default) is every recorded run, "ferry" commits
each mission to a band class and a route as one decision; the others are the
plan's options (``hermes.scheduler.plan.PlanOptions.from_config``'s keywords)
and ``member_admission``, which the H and D arms may set too (the user's
decision 4 (b)). They mean something only on the simulated clock, and they
are not ferry-spec fields, so Phase 3's ``ferry_params`` keep their strings.
:func:`mule_config_errors` refuses what plan mode cannot fly.

FeRRy Phase 5 — the learned fillings (the Phase 5 spec, other choices 6).
``flight_slot`` gains ``pair_q``, the learned (band, next stop) score of the
FQ arms, and ``contact_policy`` gains ``chen_dqn``, arm E3. Each flies one
checkpoint named by three fields (:data:`CHECKPOINT_MULE_FIELDS`: its path,
the sha256 of its arrays and its tag), all three with the switch and none
without it. They are simulated-clock only and neither ferry-spec nor plan
fields, so Phase 3's and Phase 4's ``ferry_params`` keep their strings, and
an old per-role JSON loads with them at None. :func:`mule_config_errors`
refuses what the learned fillings cannot fly.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


Position = Tuple[float, float, float]

#: ``mission_clock`` values: "wall" (every recorded run) and "sim".
CLOCK_WALL = "wall"
CLOCK_SIM = "sim"
MISSION_CLOCKS: Tuple[str, ...] = (CLOCK_WALL, CLOCK_SIM)

#: ``backhaul_model`` values (hermes/mule/ferry.py): "mission" is the recorded
#: per-mission loss schedule or flat percentage the cluster draws from a
#: stream; "seconds" is the seconds-axis channel with a keyed draw.
BACKHAUL_MISSION = "mission"
BACKHAUL_SECONDS = "seconds"
BACKHAUL_MODELS: Tuple[str, ...] = (BACKHAUL_MISSION, BACKHAUL_SECONDS)

#: ``deadline_bounds`` values: the scheduler's
#: ``hermes.scheduler.stages.s3b_feasibility.DEADLINE_BOUNDS``, restated so
#: this module stays import-free (a unit test keeps the two equal).
DEADLINE_BOUNDS: Tuple[str, ...] = ("collection", "delivery_per_stop", "delivery")

#: FeRRy Phase 4 — the plan clock's switch values, each with its recorded value
#: first: ``plan_mode``, ``member_admission``, ``flight_slot`` and the
#: band-class policy (``search`` or ``fixed:<class>``). Restated from
#: ``hermes.scheduler.plan.types`` and ``hermes.types.scheduler`` (and
#: ``MEMBER_ADMISSIONS`` from ``hermes.scheduler.stages.s3b_feasibility``) so
#: that this module stays import-free and a recorded mule never loads the plan
#: package; a unit test keeps each equal to its source. FeRRy Phase 5 adds the
#: flight slot's ``pair_q``, the learned (band, next stop) score (the FQ arms).
PLAN_MODE_LEGACY = "legacy"
PLAN_MODE_FERRY = "ferry"
PLAN_MODES: Tuple[str, ...] = (PLAN_MODE_LEGACY, PLAN_MODE_FERRY)
MEMBER_ADMISSION_WHOLE = "whole"
MEMBER_ADMISSION_SUBSET = "subset"
MEMBER_ADMISSIONS: Tuple[str, ...] = (MEMBER_ADMISSION_WHOLE, MEMBER_ADMISSION_SUBSET)
FLIGHT_SLOT_COMMITTED = "committed"
FLIGHT_SLOT_CROSS_HEURISTIC = "cross_heuristic"
FLIGHT_SLOT_PAIR_Q = "pair_q"
FLIGHT_SLOTS: Tuple[str, ...] = (
    FLIGHT_SLOT_COMMITTED, FLIGHT_SLOT_CROSS_HEURISTIC, FLIGHT_SLOT_PAIR_Q,
)
BAND_POLICY_SEARCH = "search"
BAND_POLICY_FIXED_PREFIX = "fixed:"

#: FeRRy Phase 5 — ``contact_policy`` value of arm E3, the per-departure DQN
#: after Chen et al. (GLOBECOM Workshops 2023; build plan L1026), which flies
#: legacy mode on the simulated clock with a contact band. Restated from the
#: policy's own ``name`` (``hermes.scheduler.policies.chen_dqn``, unit U6) so
#: that this module stays import-free; the two must stay equal.
CONTACT_POLICY_CHEN_DQN = "chen_dqn"

#: A checkpoint's sha256 as ``hashlib.sha256(...).hexdigest()`` writes it: 64
#: lowercase hexadecimal digits, so the mule's check compares exact strings.
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

#: A checkpoint tag (``main``, ``hand``, ``dwell``, ``cov``, ``g0`` ... ``g99``,
#: ``e3``; the Phase 5 spec, other choices 5): ASCII letters, digits, ``_`` and
#: ``-``, a letter or digit first, at most 28 characters, the bound the arm
#: labels keep. A tag names a directory of the checkpoint layout
#: (``results/exp5/checkpoints/<study>/<tag>/``), the runner's
#: ``--pair-checkpoint TAG=PATH`` and the provenance, so it holds no path
#: separator, dot, space or ``=``.
_CHECKPOINT_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,27}")


def mission_schedule_index(mission_round, length: int) -> int:
    """The entry of a per-mission schedule that mission ``mission_round`` reads.

    ``mission_round - 1``, clamped to the schedule: a missing or zero round
    reads the first entry, a round past the end the last. This is the rule
    the cluster's backhaul loss draw has used since EX-4.3
    (``ClusterService._backhaul_dropped``), kept in one place so the draw, the
    loss probability a sim-clock event reports and the mule's causal RF prior
    (``MuleConfig.rf_prior_schedule_db``) always read the same entry.
    """
    idx = (int(mission_round) - 1) if mission_round else 0
    return min(max(idx, 0), int(length) - 1)

#: The merge rule whose buffer, not the quorum, decides when θ moves.
_AGG_FEDBUFF = "agg:fedbuff"


@dataclass
class _SeedDevice:
    """Lightweight device registration row for cluster pre-seeding."""

    device_id: str
    position: Position = (0.0, 0.0, 0.0)
    assigned_mule: Optional[str] = None


@dataclass
class ClusterConfig:
    """Settings for the single edge-server (cluster) process."""

    cluster_id: str
    # TCP host/port the dock listens on. Mules connect here.
    dock_host: str = "127.0.0.1"
    dock_port: int = 0  # 0 = ephemeral; orchestrator reads it back
    # Mules expected to register before the cluster dispatches the
    # first DOWN bundle. The orchestrator populates this from the
    # topology — the cluster service waits until all show up.
    expected_mules: List[str] = field(default_factory=list)
    # Devices to pre-seed in the registry before any mule docks. Each
    # entry carries (device_id, position, assigned_mule). The cluster
    # registers them and rebalances onto the listed mules so the very
    # first DOWN bundle dispatches a populated MissionSlice.
    seed_devices: List[dict] = field(default_factory=list)
    # Cluster-controlled tunables.
    synth_batch_size: int = 4
    # L-L2: minimum number of mules that must contribute an UP bundle
    # before the cluster aggregates and closes a round. Set to the
    # number of mules in the topology for full-FedAvg semantics; set to
    # 1 for partial-FedAvg (cluster aggregates as soon as any mule
    # reports, accepting staleness from absent mules). Defaults to 1
    # because in Sprint 2 demos we want forward progress with a single
    # mule; production deployments typically pin this to len(mules).
    min_participation: int = 1
    # Optional Tier-3 endpoint (cloud link). When set, the cluster
    # service polls / posts on its own cadence. None = no cloud link.
    tier3_url: Optional[str] = None
    # EX-4.1 — real DNN-IDS. When ``init_theta_path`` is set, the global
    # model is seeded from those weights (real create_CICIOT_Model) instead
    # of the 13-param stub, so the whole pipeline carries the real shapes.
    # When ``eval_test_path`` + ``input_dim`` are set, the cluster evaluates
    # the aggregated θ on the held-out test set after each round and emits a
    # ``model_eval`` event (accuracy/auc/loss). All None -> the stub path.
    init_theta_path: Optional[str] = None
    eval_test_path: Optional[str] = None
    input_dim: Optional[int] = None
    # EX-4.2 — long-range mule->base-station backhaul loss (%). Under the
    # jittery regime the mule's aggregate upload drops with this probability
    # per dock; the round does not close but θ' still flows for Pass 2, so
    # the update is carried, not lost (recoverable — unlike H0's permanent
    # dead-zone). 0.0 -> reliable backhaul.
    backhaul_loss_pct: float = 0.0
    # Seed for the backhaul-loss RNG so the loss pattern is deterministic and
    # varies per trial. Set by the driver from the paired trial seed.
    backhaul_rng_seed: Optional[int] = None
    # EX-4.3 arm H3 — per-mission backhaul-loss probabilities from the L1
    # channel model (index = mission_round-1). When set, it overrides the
    # flat ``backhaul_loss_pct``: adaptive channel selection (H3) yields a
    # lower-loss schedule than the fixed channel (H1/H2), so L1's effect is
    # a real, seed-consistent reduction in dropped rounds.
    backhaul_loss_schedule: Optional[List[float]] = None
    # FeRRy Phase 1 — the L3 merge rule (hermes/mission/aggregation_rules.py)
    # and its parameters (AggregationSpec fields other than ``rule``).
    # "agg:plain" is the num_examples mean every recorded run used. The mule
    # must run the same rule; the driver sets both from one flag.
    aggregation: str = "agg:plain"
    aggregation_params: dict = field(default_factory=dict)
    # FeRRy Phase 3 — the cluster on the simulated mission clock. The cluster
    # keeps no clock: "sim" makes it track the latest ``UpBundle.sim_upload_ts``
    # it has ingested and echo it on every DOWN (``cluster_sim_ts``, the
    # mules' Lamport sync), add simulated-time fields to its events, refuse
    # wall-clock deadline overrides, and forward each device's contact SNR per
    # band class (SpectrumSig, design section 4.7). "wall" is every recorded
    # run. Must match the mules' ``mission_clock``.
    mission_clock: str = CLOCK_WALL
    # How a backhaul upload is lost. "mission" (the recorded model): the flat
    # ``backhaul_loss_pct`` or the per-mission ``backhaul_loss_schedule``,
    # drawn from a stream. "seconds" (sim clock only, critic B16): the UP's
    # own loss probability (``UpBundle.backhaul.p_loss``, the seconds-axis
    # SNR at the mule's upload), drawn keyed by (trial seed, mule, mission
    # round), so a mission's outcome is paired across arms. Must match the
    # mules'.
    backhaul_model: str = BACKHAUL_MISSION
    # The trial seed the keyed draws are salted with (``ferry_salt``). Needed
    # for the seconds model; the mules carry the same value.
    trial_seed: Optional[int] = None
    # Names of the contact link's band classes in index order, to read the
    # ``band`` index on a report line; None is the D1 default (wide, medium,
    # narrow). Must match the mules'.
    contact_band_classes: Optional[List[str]] = None
    # Exp 5 addendum (Study 5.12): the IDS model's architecture
    # (``experiments.exp4.model_task.MODEL_ARCHS``) the cluster evaluates;
    # None is the canonical CICIoT model, every recorded run's. Written to the
    # per-role JSON only when set (:data:`CONFIG_FIELDS_OMITTED_AT_NONE`).
    model_arch: Optional[str] = None


@dataclass
class MuleConfig:
    """Settings for one mule process."""

    mule_id: str
    # Mule's RF link is a TCP server — devices connect inbound to here.
    rf_host: str = "127.0.0.1"
    rf_port: int = 0
    # Cluster's dock to connect outbound to.
    dock_host: str = "127.0.0.1"
    dock_port: int = 0
    # Devices expected to register on the RF link before the mule
    # starts running missions (otherwise contacts would broadcast to
    # an empty room). Populated by the orchestrator from the topology.
    expected_devices: List[str] = field(default_factory=list)
    # Two-pass / clustering tunables. ``rf_range_m=None`` keeps the
    # legacy single-pass path (Sprint 1A); set it to enable Sprint 1.5.
    rf_range_m: Optional[float] = 60.0
    session_ttl_s: float = 5.0
    # Number of mission cycles to run before the service exits. None =
    # run until shutdown signal.
    n_missions: Optional[int] = None
    # EX-4.2 arm H2 — RL target selector (S3.5 tie-break). When
    # ``use_rl_selector`` is set, the mule builds a ``TargetSelectorRL`` and
    # passes it to its supervisor (deterministic distance ranking otherwise,
    # = arm H1). ``selector_weights_path`` loads a trained DDQN (.npz from
    # experiments.exp3.train_a4); omit it for a random-init selector (smoke
    # only — not paper-grade).
    use_rl_selector: bool = False
    selector_weights_path: Optional[str] = None
    # EX-4.3 arm H3 — the L1->L2 edge. When set, the mule's scheduler feeds
    # this (the mean effective SNR of the L1-chosen channel) to the target
    # selector's rf_prior feature, instead of the hardcoded 20.0 default. This
    # is the real RF prior the SEC26 audit found was never wired at runtime.
    rf_prior_snr_db: Optional[float] = None
    # S3b — per-mission time budget (seconds). When set, the scheduler's
    # deadline feasibility gate is ACTIVE: contacts that cannot be reached
    # before their own deadline, or that would overrun this budget, are
    # dropped before ordering. ``None`` keeps the historical behaviour in
    # which Deadline(j) is only a sort key.
    mission_budget_s: Optional[float] = None
    # SOTA baseline arm. ``None`` = our scheduler. ``"max_aoi"`` swaps the
    # ranking for the Age-of-Information greedy comparator
    # (hermes/scheduler/policies/max_aoi.py); ``"oort"``, ``"whittle"``,
    # ``"fedex"`` and ``"fedcs"`` are arms D2-D5. Mutually exclusive with
    # ``use_rl_selector`` — both occupy the single target-selector slot, so
    # setting both is a configuration error rather than a blend.
    contact_policy: Optional[str] = None
    # S3c — mission-level deadline-window adaptation. Off by default, which is
    # exactly how every recorded sweep ran: the window scale stays 1.0 and the
    # per-device rule alone sets Deadline(j). Turned on, the mule tracks how
    # much of each mission it actually served and widens ALL windows together
    # when it is systematically falling short — the systemic signal the
    # per-device rule is blind to. Kept as scalars rather than an adapter
    # object because this config crosses a process boundary; the mule process
    # builds the adapter from these.
    mission_window_adaptation: bool = False
    mission_window_history: int = 5
    mission_window_target: float = 0.8
    mission_window_gain: float = 2.0
    mission_window_max_scale: float = 4.0
    # FeRRy Phase 1 — the merge rule, which must match the cluster's (it
    # decides how the mule merges and which form devices answer in), and
    # whether Pass 2 is walked against ``mission_budget_s``. Off, Pass 2
    # delivers to the whole slice, as in every recorded run.
    aggregation: str = "agg:plain"
    aggregation_params: dict = field(default_factory=dict)
    pass_2_budget: bool = False
    # FeRRy Phase 1 — the deadline law (DeadlineLaw in
    # hermes/scheduler/stages/s3_deadline.py). "additive" is the recorded
    # −5 s / +10 s law; "multiplicative" scales and clamps the window, splits
    # PARTIAL from TIMEOUT and makes cluster overrides one-shot. Parameters
    # are DeadlineLaw fields. ``miss_priority`` makes S3b admit contacts by
    # their members' miss streak before their deadline.
    deadline_law: str = "additive"
    deadline_params: dict = field(default_factory=dict)
    miss_priority: bool = False
    # FeRRy Phase 2 — several mules on one cluster. ``down_wait_s`` bounds the
    # inter-pass dock's wait for its DOWN and makes running out survivable:
    # the mule emits ``dock_down_timeout``, skips Pass 2 and flies the next
    # mission on this one's θ. None is the recorded single-mule dock: one
    # 10 s wait whose expiry ends the mission loop. ``dock_on_empty`` makes a
    # mission that collected nothing dock anyway with an empty partial, which
    # counts toward the cluster's quorum and merges nothing; off, an empty
    # mission skips the dock as every recorded run did.
    down_wait_s: Optional[float] = None
    dock_on_empty: bool = False
    # FeRRy Phase 2 — options of the whole-scheduler baselines, read only by
    # the policy they belong to (``contact_policy``). Defaults are the
    # policies' own. D3 ``whittle`` (hermes/scheduler/policies/whittle.py):
    # ``whittle_variant`` "expected" | "literal", ``whittle_weights``
    # "uniform" | "oort" (Oort's utility, which needs a real model). D5
    # ``fedcs`` (policies/fedcs_degraded.py): ``fedcs_value`` "unit" |
    # "devices".
    whittle_variant: str = "expected"
    whittle_weights: str = "uniform"
    fedcs_value: str = "unit"

    # ---------------- FeRRy Phase 3: the mission clock and contact link ------
    # ``mission_clock`` "wall" is every recorded run: ``time.time`` stamps
    # every mission-time read. "sim" flies every mission on one simulated
    # MissionClock per mule process (hermes/l1/mission_clock.py), priced by a
    # FerrySpec built from the fields below (``ferry_spec_kwargs``;
    # hermes/mule/ferry.py documents each). On the wall clock every field of
    # this block except the deadline time unit must keep its default
    # (``mule_config_errors``). Defaults are the design's (Phase 3 design
    # section 1, decisions D1-D3 as accepted on 2026-09-29).
    mission_clock: str = CLOCK_WALL
    # The trial seed: the contact channel, the backhaul and the availability
    # draw are salted with it, so every arm of a trial sees the same channel.
    trial_seed: Optional[int] = None
    # D1 — the band class every stop flies ("wide", "medium", "narrow"), or
    # None: the mission clock without a contact link (the channel-free
    # control, critic A1). ``contact_band_classes`` names the link's classes
    # (None: wide, medium, narrow; the 10 MHz option adds "medium_wide").
    contact_band: Optional[str] = None
    contact_band_classes: Optional[List[str]] = None
    # What the mule does when the rest of a pass stops fitting: "abort"
    # (Amendment 8's rule, the recorded one) or "replan"; and the re-plan's
    # fallback for our arms, "reorder" or "trim" (unit U4). Chosen at the
    # pilot (critic B5).
    in_flight_response: str = "abort"
    replan_fallback: str = "reorder"
    # D2 — the backhaul. "mission": the cluster's recorded loss schedule (the
    # upload is still timed on the clock at the fixed carrier's mean SNR);
    # "seconds": the seconds-axis channel, with ``backhaul_policy`` "fixed"
    # (argmax g_c) or "adaptive" (arm H3's controller at every upload),
    # ``backhaul_regime`` its "clean" / "jittery" constants, and its period
    # P_bh given directly or as ``n_missions * t_nom_s``.
    backhaul_model: str = BACKHAUL_MISSION
    backhaul_policy: str = "fixed"
    backhaul_regime: str = "clean"
    backhaul_period_s: Optional[float] = None
    t_nom_s: Optional[float] = None
    # Spec Q8 — "origin": the device's own draw of rel x rf_factor (the
    # recorded model); "channel": the SNR gate at the stop plus the
    # availability rel_i drawn on the mule, keyed by (seed, device, round).
    # ``device_availability`` is that ground truth {device_id: rel_i}, used
    # only by the mule's keyed draw and never shown to the scheduler, the
    # policies or the L1 state (critic B16); empty unless "channel".
    contact_reliability_source: str = "origin"
    device_availability: Dict[str, float] = field(default_factory=dict)
    # D3 — the bytes each direction is priced for (None: measured) and what
    # Deadline(j) bounds (spec Q2, ``DEADLINE_BOUNDS``): "collection" (the
    # default, arrival + dwell), "delivery_per_stop" (per stop, that stop's
    # own return plus the upload, not the route's actual delivery; what
    # "delivery" meant at ef1faa1) or "delivery" (route-level: the route's
    # landing plus the upload meets the deadline of every update collected).
    payload_bytes: Optional[int] = None
    deadline_bounds: str = "collection"
    # D1 — the contact link: SNR floor (CQI 1), altitude, path-loss exponent,
    # shadowing sigma and the edge-availability quantile of the margin.
    snr_floor_db: float = -6.7
    altitude_m: float = 25.0
    n_pl: float = 2.2
    shadow_sigma_db: float = 4.0
    margin_quantile: float = 0.9
    # D2 — the contact channel: interference regime (clean by default, for
    # every cell; "jittery" is test (c)), its period P_c, the noise bin, and
    # the shadowing correlation time and keying ("time" or "position").
    contact_regime: str = "clean"
    interference_period_s: float = 60.0
    noise_bin_s: float = 1.0
    shadow_corr_s: float = 7.4
    shadow_keying: str = "time"
    # Exp 5 addendum, Study 5.15: the contact regime's interference amplitude
    # A and noise sigma_I (dB), overridden; None keeps the regime's own (the
    # recorded run), and leaves each out of ``ferry_params``.
    interference_amp_db: Optional[float] = None
    interference_sigma_db: Optional[float] = None
    # D3 — flight and SIMULATED energy (Zeng-Xu-Zhang 2019 at the cruise
    # speed unless the powers are given; capacity None = no energy clause).
    cruise_speed_m_s: float = 5.0
    turnaround_s: float = 30.0
    listen_s: float = 1.0
    energy_capacity_j: Optional[float] = None
    p_move_w: Optional[float] = None
    p_hover_w: Optional[float] = None
    # Spec Q1 — the deadline law's time unit (valid on either clock): the
    # law's constants are multiplied by ``deadline_time_scale``, and
    # ``initial_window_s`` sets Φ₀ in the law's recorded unit (None: 60 s).
    # 1.0 and None are the recorded law.
    deadline_time_scale: float = 1.0
    initial_window_s: Optional[float] = None
    # The model's input width, for ``mule_ready`` on the mission clock (the
    # payload's provenance, design R8). None on the stub.
    input_dim: Optional[int] = None
    # Freeze Amendment 10 — the token this mule's RF server requires of every
    # registration (``TCPRFLinkServer(link_token=...)``). None (every recorded
    # run) accepts any; one value per trial across its mules and devices.
    rf_link_token: Optional[str] = None
    # Critic B4 — the causal RF prior under the recorded ``mission`` backhaul
    # model with the L1 channel (``--l1-channel``). Entry r - 1 is the SNR
    # (dB) the L1 trace gives the carrier chosen for mission round r's upload
    # (``mission_schedule_index``): the trace the cluster's
    # ``backhaul_loss_schedule`` is ``loss_from_snr`` of, entry for entry.
    # After each mission that uploaded, the mule process sets the planner's
    # ``rf_prior_snr_db`` to that mission's entry, so a Pass-1 plan only ever
    # sees uploads already made, as the seconds model's producer does
    # (hermes/l1/rf_prior.py): the non-causal mean over the whole trial is not
    # handed to a ferry cell. None, every recorded run: the prior stays
    # ``rf_prior_snr_db`` (20 dB by default). Simulated clock only, and not
    # with the seconds model, which observes its own channel.
    rf_prior_schedule_db: Optional[List[float]] = None

    # ---------------- FeRRy Phase 5: the learned fillings' checkpoints ---------
    # Declared before the Phase 4 block because a test pins the plan fields as
    # this class's last (test_p4_config_driver.py). A learned filling flies one
    # checkpoint, named by three fields set together or not at all
    # (``mule_config_errors``): its path (repo-relative when the driver finds
    # it inside the repo, else absolute; read under the repo root when
    # relative), the sha256 of its arrays, which the mule verifies before it
    # flies and refuses to run on a mismatch, and its tag (``main``, ``hand``,
    # ``g0`` ... ``g99``, ``e3``), which with the sha is the run's provenance.
    # The ``pair_checkpoint*`` triple goes with ``flight_slot="pair_q"`` (plan
    # mode; the FQ arms), the ``policy_checkpoint*`` triple with
    # ``contact_policy="chen_dqn"`` (arm E3: legacy mode on the simulated
    # clock, with a contact band); None elsewhere, every recorded run included.
    # Simulated-clock only, and neither ferry-spec nor plan fields, so Phase
    # 3's and Phase 4's ``ferry_params`` keep their strings (the Phase 5 spec,
    # other choices 6).
    pair_checkpoint: Optional[str] = None
    pair_checkpoint_sha256: Optional[str] = None
    pair_checkpoint_tag: Optional[str] = None
    policy_checkpoint: Optional[str] = None
    policy_checkpoint_sha256: Optional[str] = None
    policy_checkpoint_tag: Optional[str] = None

    # ---------------- FeRRy Phase 4: the plan clock -----------------------------
    # ``plan_mode`` "legacy" is every recorded run: the scheduler plans each
    # mission with ``build_contact_queue``. "ferry" commits each mission at the
    # dock to one band class and a Pass-1 route as one decision
    # (``FLScheduler.build_ferry_plan``), flies that class in both passes and
    # closes the plan once the merge is known (hermes/mule/mule_main.py). It
    # needs the simulated clock, a ``contact_band`` (under ``search`` only the
    # reference class) and ``t_nom_s`` (T in the plan score, the user's
    # decision 2 (b)); ``mule_config_errors`` lists what else it refuses. The
    # fields after it are ``hermes.scheduler.plan.PlanOptions.from_config``'s
    # keywords and keep its names (:data:`PLAN_OPTION_FIELDS`):
    # ``band_class_policy`` "search" (every class of the link, arm F) or
    # "fixed:<class>" (the run's ``contact_band`` only, arm FB+<class>);
    # ``flight_slot`` "committed" (arm F), "cross_heuristic" (arm FX, decision
    # 5) or, FeRRy Phase 5, "pair_q" (the FQ arms, with the pair checkpoint
    # above); the age cap S in the device's own missions (None: off, arm
    # F-cap) and its lookahead L (decision 1); the plan score's and the
    # search's settings
    # (``PlanScoreParams`` and ``PlanSearchParams`` field names; {} keeps their
    # defaults, and an unknown key is refused).
    # ``member_admission`` (decision 4 (b)): "whole", the recorded rule, admits a
    # stop with all its members or not at all; "subset" may admit the members
    # that still fit. It is valid on the simulated clock in plan mode (the F
    # arms fly "subset" unless a run asks otherwise) and for the H and D arms
    # (S3b, the D1-D3 and D5 walks; never D4, whose tour has no gate).
    plan_mode: str = PLAN_MODE_LEGACY
    band_class_policy: str = BAND_POLICY_SEARCH
    member_admission: str = MEMBER_ADMISSION_WHOLE
    flight_slot: str = FLIGHT_SLOT_COMMITTED
    age_cap_missions: Optional[int] = None
    age_cap_lookahead: int = 0
    plan_score_params: dict = field(default_factory=dict)
    plan_search_params: dict = field(default_factory=dict)

    def ferry_spec_kwargs(self) -> Dict[str, Any]:
        """The keyword arguments of ``FerrySpec.from_config`` this config gives.

        One mapping, used by the mule process and by the driver (which prices
        T_nom and the D4 CARP split with the same physics). Pure data: this
        module imports nothing from the mule.
        """
        out: Dict[str, Any] = {
            "rf_range_m": self.rf_range_m,
            "seed": self.trial_seed,
            "n_missions": self.n_missions,
        }
        for name, kwarg in FERRY_SPEC_FIELDS.items():
            value = getattr(self, name)
            if name == "device_availability":
                value = dict(value or {})
            elif name == "contact_band_classes":
                value = None if value is None else list(value)
            out[kwarg] = value
        return out


#: ``MuleConfig`` fields that configure the ferry mode, and the keyword of
#: ``hermes.mule.ferry.FerrySpec.from_config`` each one feeds. Their defaults
#: are ``from_config``'s (a unit test keeps them equal).
FERRY_SPEC_FIELDS: Dict[str, str] = {
    "contact_band": "contact_band",
    "contact_band_classes": "band_classes",
    "in_flight_response": "in_flight_response",
    "replan_fallback": "replan_fallback",
    "backhaul_model": "backhaul_model",
    "backhaul_policy": "backhaul_policy",
    "backhaul_regime": "backhaul_regime",
    "backhaul_period_s": "backhaul_period",
    "t_nom_s": "t_nom_s",
    "contact_reliability_source": "contact_reliability_source",
    "device_availability": "device_availability",
    "payload_bytes": "payload_bytes",
    "deadline_bounds": "deadline_bounds",
    "snr_floor_db": "snr_floor_db",
    "altitude_m": "altitude_m",
    "n_pl": "n_pl",
    "shadow_sigma_db": "shadow_sigma_db",
    "margin_quantile": "margin_quantile",
    "contact_regime": "contact_regime",
    "interference_period_s": "interference_period_s",
    "noise_bin_s": "noise_bin_s",
    "shadow_corr_s": "shadow_corr_s",
    "shadow_keying": "shadow_keying",
    "interference_amp_db": "interference_amp_db",
    "interference_sigma_db": "interference_sigma_db",
    "cruise_speed_m_s": "cruise_speed_m_s",
    "turnaround_s": "turnaround_s",
    "listen_s": "listen_s",
    "energy_capacity_j": "energy_capacity_j",
    "p_move_w": "p_move_w",
    "p_hover_w": "p_hover_w",
}

#: Every ``MuleConfig`` field the Exp 5 addendum (Studies 5.11-5.15) adds, each
#: at a default that is the recorded run, so a recorded trial's per-role JSON
#: gains exactly these keys at their defaults; the tests that pin what it gains
#: (Freeze Rule 1, additive fields only) name them beside Phase 4's plan fields
#: and Phase 5's checkpoint fields.
ADDENDUM_MULE_FIELDS: Tuple[str, ...] = ("interference_amp_db", "interference_sigma_db")

#: Ferry fields a trial's ``ferry_params`` leaves out while they are None (the
#: Exp 5 addendum): their None is the recorded run (the contact regime's own
#: interference amplitude and noise, Study 5.15), so every recorded row keeps
#: its string, and a run that sets one shows it.
FERRY_PARAMS_OMITTED_AT_NONE: Tuple[str, ...] = ("interference_amp_db", "interference_sigma_db")

#: ``MuleConfig``'s FeRRy Phase 4 plan fields: ``plan_mode`` and the plan
#: options (:data:`PLAN_OPTION_FIELDS`). Not ferry-spec fields (the spec prices
#: flight and the channel; the plan decides what to fly), so a Phase 3 trial's
#: ``ferry_params`` is unchanged; each is simulated-clock only
#: (:data:`SIM_ONLY_MULE_FIELDS`).
PLAN_MULE_FIELDS: Tuple[str, ...] = (
    "plan_mode", "band_class_policy", "member_admission", "flight_slot",
    "age_cap_missions", "age_cap_lookahead", "plan_score_params", "plan_search_params",
)

#: The plan fields ``hermes.scheduler.plan.PlanOptions.from_config`` takes, under
#: the same names (a unit test keeps them equal to its keywords).
PLAN_OPTION_FIELDS: Tuple[str, ...] = PLAN_MULE_FIELDS[1:]

#: ``MuleConfig``'s FeRRy Phase 5 checkpoint fields, (path, sha256, tag) per
#: learned filling: the pair score's, with ``flight_slot="pair_q"``, and arm
#: E3's, with ``contact_policy="chen_dqn"``. Declared in this order, right
#: before ``plan_mode``. Neither ferry-spec nor plan fields, so no Phase 3 or
#: Phase 4 ``ferry_params`` string changes (the Phase 5 spec, other choices 6);
#: each is simulated-clock only (:data:`SIM_ONLY_MULE_FIELDS`).
PAIR_CHECKPOINT_FIELDS: Tuple[str, ...] = (
    "pair_checkpoint", "pair_checkpoint_sha256", "pair_checkpoint_tag",
)
POLICY_CHECKPOINT_FIELDS: Tuple[str, ...] = (
    "policy_checkpoint", "policy_checkpoint_sha256", "policy_checkpoint_tag",
)
CHECKPOINT_MULE_FIELDS: Tuple[str, ...] = PAIR_CHECKPOINT_FIELDS + POLICY_CHECKPOINT_FIELDS

#: ``MuleConfig`` fields that only mean something on the simulated clock:
#: the ferry fields, the trial seed, the input width, the causal RF prior
#: schedule, the checkpoint fields and the plan fields. On the wall clock each
#: must keep its default. The deadline time unit is not among them: it is a
#: law parameter on either clock.
SIM_ONLY_MULE_FIELDS: Tuple[str, ...] = tuple(FERRY_SPEC_FIELDS) + (
    "trial_seed", "input_dim", "rf_prior_schedule_db",
) + CHECKPOINT_MULE_FIELDS + PLAN_MULE_FIELDS


def _field_default(cls, name: str) -> Any:
    fld = {f.name: f for f in fields(cls)}[name]
    if fld.default_factory is not MISSING:  # type: ignore[misc]
        return fld.default_factory()  # type: ignore[misc]
    return fld.default


def _positive_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) > 0.0)


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def mule_config_errors(cfg: "MuleConfig") -> List[str]:
    """What is wrong with ``cfg``'s clock settings; empty when it can run.

    Cheap checks, import-free but for plan mode's options (the mule builds its
    ``FerrySpec`` at start, which validates every value in full):

    * ``mission_clock`` is "wall" or "sim";
    * on the wall clock every sim-only field is at its default: a contact
      band, ``replan``, the seconds-axis backhaul (critic B16), the channel
      reliability source, a declared payload... need the mission clock;
    * on the sim clock: the two-pass path (``rf_range_m``), a trial seed,
      a known backhaul model and ``deadline_bounds`` (:data:`DEADLINE_BOUNDS`),
      the channel reliability source only with a band (critic B16), the
      ground-truth availability only under that source, a backhaul period
      for the seconds model, and the causal RF prior schedule only under the
      ``mission`` model, as finite SNRs (critic B4);
    * on the sim clock, the FeRRy Phase 4 plan fields
      (:func:`_plan_config_errors`);
    * the FeRRy Phase 5 learned fillings: arm E3's ``contact_policy='chen_dqn'``
      only on the sim clock, and there their settings and checkpoint fields
      (:func:`_learned_config_errors`).
    """
    errors: List[str] = []
    clock = getattr(cfg, "mission_clock", CLOCK_WALL)
    if clock not in MISSION_CLOCKS:
        return [f"mission_clock must be one of {MISSION_CLOCKS}, got {clock!r}"]
    if clock == CLOCK_WALL:
        changed = [
            name for name in SIM_ONLY_MULE_FIELDS
            if getattr(cfg, name, _field_default(MuleConfig, name))
            != _field_default(MuleConfig, name)
        ]
        if changed:
            errors.append(
                f"{', '.join(changed)}: only on the simulated mission clock; set "
                f"mission_clock='sim' or leave the default"
            )
        if getattr(cfg, "contact_policy", None) == CONTACT_POLICY_CHEN_DQN:
            errors.append(
                f"contact_policy={CONTACT_POLICY_CHEN_DQN!r} (arm E3) chooses each next stop "
                f"in flight on the simulated mission clock: set mission_clock='sim'"
            )
        return errors
    if cfg.rf_range_m is None:
        errors.append("mission_clock='sim' runs the two-pass contact path: set rf_range_m")
    if cfg.trial_seed is None:
        errors.append("mission_clock='sim' needs trial_seed (the channel's salts)")
    if cfg.backhaul_model not in BACKHAUL_MODELS:
        errors.append(f"backhaul_model must be one of {BACKHAUL_MODELS}, got {cfg.backhaul_model!r}")
    if cfg.deadline_bounds not in DEADLINE_BOUNDS:
        errors.append(
            f"deadline_bounds must be one of {DEADLINE_BOUNDS}, got {cfg.deadline_bounds!r}"
        )
    if cfg.contact_reliability_source == "channel" and cfg.contact_band is None:
        errors.append(
            "contact_reliability_source='channel' needs a contact_band: the reliability "
            "is then the SNR gate at the stop (critic B16)"
        )
    if cfg.device_availability and cfg.contact_reliability_source != "channel":
        errors.append(
            "device_availability is the ground truth of contact_reliability_source="
            "'channel'; under 'origin' the devices draw their own reliability"
        )
    if (cfg.backhaul_model == BACKHAUL_SECONDS and cfg.backhaul_period_s is None
            and (cfg.t_nom_s is None or cfg.n_missions is None)):
        errors.append(
            "backhaul_model='seconds' needs its period: backhaul_period_s, or t_nom_s "
            "with n_missions (P_bh = n_missions * T_nom)"
        )
    for name in ("t_nom_s", "backhaul_period_s"):
        value = getattr(cfg, name)
        if value is not None and not _positive_number(value):
            errors.append(f"{name} must be finite and > 0, got {value!r}")
    schedule = getattr(cfg, "rf_prior_schedule_db", None)
    if schedule is not None:
        if cfg.backhaul_model == BACKHAUL_SECONDS:
            errors.append(
                "rf_prior_schedule_db feeds the RF prior under the recorded 'mission' "
                "backhaul model; the seconds model feeds it from its own channel (critic B4)"
            )
        if (not isinstance(schedule, (list, tuple)) or not schedule
                or not all(_finite_number(v) for v in schedule)):
            errors.append(
                f"rf_prior_schedule_db must be a non-empty list of finite SNRs (dB), "
                f"got {schedule!r}"
            )
    errors += _plan_config_errors(cfg)
    errors += _learned_config_errors(cfg)
    return errors


def _plan_config_errors(cfg: "MuleConfig") -> List[str]:
    """What is wrong with a simulated-clock config's plan fields (FeRRy Phase 4).

    The Phase 4 spec, other choices 10, with the critic's corrections. Every
    check fires only on a value other than the recorded one, so a Phase 3
    config passes as it always did.

    * ``plan_mode`` and ``member_admission`` are known values.
    * Outside plan mode, ``member_admission`` is the one plan field an arm may
      set (the H and D arms, decision 4 (b)), and not with D4's
      ``contact_policy='fedex'``, whose tour has no gate to admit part of a
      stop (unit_U3b.md section 1.4). Every other plan field, the age cap
      included, configures plan mode and is refused, not silently ignored.
    * In plan mode: a ``contact_band`` (the reference class; the channel-free
      control has no classes to choose among) and ``t_nom_s`` (T in the score,
      decision 2 (b)); a band-class policy of ``search`` or ``fixed:<class>``
      with the class the run's ``contact_band`` (other choices 2); the
      ``trim`` fallback, whatever the in-flight response, since the scheduler
      refuses ``reorder`` in plan mode: there the Pass-1 re-plan is a trim of
      the committed plan (members under ``subset``, whole stops only under
      ``whole``, R5), priority stops first when the rest does not fit as
      flown, and re-ordering belongs to the flight slot (other choices 9,
      critic B11); no ``pass_2_budget``, whose walk admits whole stops, so a
      narrow Pass 2 at 1 MB (45 s at the median) would deliver nothing under a
      short budget (critic B8); a cap S that is an int >= 1 and a lookahead
      that is an int >= 0; no ``contact_policy`` and no ``use_rl_selector``,
      since the plan owns admission and order; and no ``abort`` together with
      a cap, since abort gives up the whole tail, capped stops that would fit
      alone included (critic A10). FeRRy Phase 5: the pair slot
      (``flight_slot='pair_q'``) flies ``replan`` only, since its mask admits a
      pair when the whole rest of the flight still fits, the fold that only the
      re-plan's departure check runs next (the orchestrator's resolution R3);
      with a cap, A10's reason is the one given.
    * When all of that holds, the options are built as the mule builds them
      (``hermes.scheduler.plan.PlanOptions.from_config``, imported only here,
      in plan mode), so every check of the plan's own types runs before a
      trial starts rather than when its mule does: an unknown flight slot, the
      cross-heuristic slot with a pinned band (FB+c flies only class c while
      FX switches class on arrival, unit U0), and so the pair slot with one
      (FeRRy Phase 5: ``pair_q`` chooses the class on arrival too), an
      unknown score or search setting, or a value those settings refuse.
    """
    mode = getattr(cfg, "plan_mode", PLAN_MODE_LEGACY)
    if mode not in PLAN_MODES:
        return [f"plan_mode must be one of {PLAN_MODES}, got {mode!r}"]
    errors: List[str] = []
    admission = getattr(cfg, "member_admission", MEMBER_ADMISSION_WHOLE)
    if admission not in MEMBER_ADMISSIONS:
        errors.append(f"member_admission must be one of {MEMBER_ADMISSIONS}, got {admission!r}")
    if mode == PLAN_MODE_LEGACY:
        if admission == MEMBER_ADMISSION_SUBSET and cfg.contact_policy == "fedex":
            errors.append(
                "member_admission='subset' re-issues a stop that fails whole with the "
                "members that still fit; contact_policy='fedex' (arm D4) visits every "
                "contact with no gate, so it runs 'whole'"
            )
        changed = [
            name for name in PLAN_OPTION_FIELDS
            if name != "member_admission"
            and getattr(cfg, name, _field_default(MuleConfig, name))
            != _field_default(MuleConfig, name)
        ]
        if changed:
            errors.append(
                f"{', '.join(changed)}: plan mode only; set plan_mode='ferry' or leave "
                f"the default"
            )
        return errors
    if cfg.contact_band is None:
        errors.append(
            "plan_mode='ferry' chooses the band class each mission: it needs a "
            "contact_band (the reference class; the channel-free control has none)"
        )
    if cfg.t_nom_s is None:
        errors.append(
            "plan_mode='ferry' scores the whole mission against T_nom: set t_nom_s"
        )
    policy = getattr(cfg, "band_class_policy", BAND_POLICY_SEARCH)
    if isinstance(policy, str) and policy.startswith(BAND_POLICY_FIXED_PREFIX):
        if policy[len(BAND_POLICY_FIXED_PREFIX):] != cfg.contact_band:
            errors.append(
                f"band_class_policy={policy!r} must pin the run's contact_band "
                f"({cfg.contact_band!r}): FB+<class> flies that class only"
            )
    elif policy != BAND_POLICY_SEARCH:
        errors.append(
            f"band_class_policy must be {BAND_POLICY_SEARCH!r} or "
            f"'{BAND_POLICY_FIXED_PREFIX}<class>', got {policy!r}"
        )
    if cfg.replan_fallback != "trim":
        errors.append(
            f"plan_mode='ferry' re-plans Pass 1 by trimming the committed plan (members "
            f"under member_admission='subset', whole stops only under 'whole'), priority "
            f"stops first, and leaves re-ordering to the flight slot: replan_fallback must "
            f"be 'trim', got {cfg.replan_fallback!r} (the scheduler refuses 'reorder' in "
            f"plan mode, critic B11)"
        )
    if cfg.pass_2_budget:
        errors.append(
            "plan_mode='ferry' refuses pass_2_budget: the budgeted Pass-2 walk admits "
            "whole stops, so a narrow Pass 2 would deliver nothing under a short "
            "budget (critic B8)"
        )
    cap = getattr(cfg, "age_cap_missions", None)
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        errors.append(f"age_cap_missions must be None or an int >= 1, got {cap!r}")
    lookahead = getattr(cfg, "age_cap_lookahead", 0)
    if isinstance(lookahead, bool) or not isinstance(lookahead, int) or lookahead < 0:
        errors.append(f"age_cap_lookahead must be an int >= 0, got {lookahead!r}")
    if cfg.contact_policy is not None:
        errors.append(
            f"plan_mode='ferry' plans admission and order itself: it takes no "
            f"contact_policy (got {cfg.contact_policy!r})"
        )
    if cfg.use_rl_selector:
        errors.append(
            "plan_mode='ferry' plans admission and order itself: it takes no RL "
            "selector (use_rl_selector)"
        )
    if cap is not None and cfg.in_flight_response == "abort":
        errors.append(
            "in_flight_response='abort' gives up the whole tail, capped stops that "
            "would fit alone included: it is refused with an age cap (critic A10); "
            "use 'replan'"
        )
    elif (getattr(cfg, "flight_slot", FLIGHT_SLOT_COMMITTED) == FLIGHT_SLOT_PAIR_Q
          and cfg.in_flight_response != "replan"):
        # FeRRy Phase 5 (the orchestrator's resolution R3): the pair slot's mask
        # folds the whole rest of the flight after the stop, the fold only the
        # re-plan's departure check runs next; under 'abort' that check folds
        # the next stop alone. With a cap set, A10's reason above says it.
        errors.append(
            f"flight_slot='pair_q' needs in_flight_response='replan', got "
            f"{cfg.in_flight_response!r}: the pair slot admits a (band, next stop) pair only "
            f"when the whole rest of the flight still fits after the stop, which is the "
            f"fold the re-plan's departure check runs next (resolution R3)"
        )
    if errors:
        return errors
    from hermes.scheduler.plan.types import PlanOptions  # plan mode only

    try:
        PlanOptions.from_config(**{name: getattr(cfg, name) for name in PLAN_OPTION_FIELDS})
    except (TypeError, ValueError) as e:
        errors.append(f"plan options: {e}")
    return errors


def _checkpoint_field_errors(name: str, value: Any) -> List[str]:
    """What is malformed about one checkpoint field's value (empty when it is not)."""
    if name.endswith("_sha256"):
        if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
            return [f"{name} must be the sha256 of the checkpoint's arrays, 64 lowercase hex "
                    f"digits as hashlib writes them, got {value!r}"]
    elif name.endswith("_tag"):
        if not isinstance(value, str) or not _CHECKPOINT_TAG.fullmatch(value):
            return [f"{name} must be a tag of ASCII letters, digits, '_' and '-', a letter or "
                    f"digit first, at most 28 characters, got {value!r}"]
    elif not isinstance(value, str) or not value.strip():
        return [f"{name} must be the checkpoint's path, a non-empty string, got {value!r}"]
    return []


def _learned_config_errors(cfg: "MuleConfig") -> List[str]:
    """What is wrong with a simulated-clock config's learned fillings (FeRRy Phase 5).

    The Phase 5 spec's switches table and other choices 6 and 11. A learned
    filling flies only from a verified checkpoint, never a random one (Freeze
    L491: no random-init arm), and a checkpoint field set where nothing reads
    it would be silently ignored, so the fields come with their switch, all
    three together, and never without it:

    * ``flight_slot='pair_q'`` in plan mode needs ``pair_checkpoint``,
      ``pair_checkpoint_sha256`` and ``pair_checkpoint_tag``; any other slot
      takes none of them. The slot is a plan option, so plan mode is the plan
      guard's to require, and a pinned band ``PlanOptions``' to refuse
      (:func:`_plan_config_errors`);
    * ``contact_policy='chen_dqn'`` (arm E3) in legacy mode needs a
      ``contact_band``, the one class it flies (build plan L1026), whole
      stops (``member_admission`` 'whole': it visits a stop for all its
      members, as D4 does, and the scheduler refuses member subsets to a
      policy that does not take them), and ``policy_checkpoint``,
      ``policy_checkpoint_sha256`` and ``policy_checkpoint_tag``; any other
      policy takes none of them. Plan mode refuses it with every
      ``contact_policy`` (the plan guard);
    * a field set beside its switch is well formed: a path, a sha256 as
      ``hashlib`` writes it (:data:`_SHA256_HEX`), a tag
      (:data:`_CHECKPOINT_TAG`).

    Where a switch cannot run at all (``pair_q`` outside plan mode,
    ``chen_dqn`` in it), the plan guard refuses the switch itself and its
    requirements are not added on top, so each mistake reads as one reason.
    Every check fires only on a value other than the recorded one (the six
    fields None, no ``pair_q`` and no ``chen_dqn``), so a Phase 3 or Phase 4
    config passes as it always did. The mule verifies the file itself, its
    sha and its schema, when it loads it; this checks the config.
    """
    errors: List[str] = []
    mode = getattr(cfg, "plan_mode", PLAN_MODE_LEGACY)
    policy = getattr(cfg, "contact_policy", None)
    if policy == CONTACT_POLICY_CHEN_DQN and mode == PLAN_MODE_LEGACY:
        if cfg.contact_band is None:
            errors.append(
                f"contact_policy={CONTACT_POLICY_CHEN_DQN!r} (arm E3) flies the cell's one "
                f"contact band: set contact_band"
            )
        if getattr(cfg, "member_admission", MEMBER_ADMISSION_WHOLE) != MEMBER_ADMISSION_WHOLE:
            errors.append(
                f"member_admission={cfg.member_admission!r}: contact_policy="
                f"{CONTACT_POLICY_CHEN_DQN!r} (arm E3) visits a stop for all its members, so "
                f"it runs 'whole'"
            )
    for names, switch, value, wanted, runs in (
        (PAIR_CHECKPOINT_FIELDS, "flight_slot",
         getattr(cfg, "flight_slot", FLIGHT_SLOT_COMMITTED), FLIGHT_SLOT_PAIR_Q,
         mode == PLAN_MODE_FERRY),
        (POLICY_CHECKPOINT_FIELDS, "contact_policy", policy, CONTACT_POLICY_CHEN_DQN,
         mode == PLAN_MODE_LEGACY),
    ):
        given = [name for name in names if getattr(cfg, name, None) is not None]
        if value != wanted:
            if given:
                errors.append(
                    f"{', '.join(given)}: only with {switch}={wanted!r}; leave them unset (None)"
                )
            continue
        if not runs:
            continue
        missing = [name for name in names if name not in given]
        if missing:
            errors.append(
                f"{switch}={wanted!r} flies a verified checkpoint, never a random one: set "
                f"{', '.join(names)} (missing: {', '.join(missing)})"
            )
        for name in given:
            errors += _checkpoint_field_errors(name, getattr(cfg, name))
    return errors


def cluster_config_errors(cfg: "ClusterConfig") -> List[str]:
    """What is wrong with the cluster's clock settings; empty when it can run."""
    errors: List[str] = []
    clock = getattr(cfg, "mission_clock", CLOCK_WALL)
    if clock not in MISSION_CLOCKS:
        return [f"mission_clock must be one of {MISSION_CLOCKS}, got {clock!r}"]
    model = getattr(cfg, "backhaul_model", BACKHAUL_MISSION)
    if model not in BACKHAUL_MODELS:
        errors.append(f"backhaul_model must be one of {BACKHAUL_MODELS}, got {model!r}")
    if clock == CLOCK_WALL:
        if model == BACKHAUL_SECONDS:
            errors.append(
                "backhaul_model='seconds' prices the upload at its simulated time: it "
                "needs mission_clock='sim' (critic B16)"
            )
        if getattr(cfg, "contact_band_classes", None) is not None:
            errors.append("contact_band_classes configures the simulated mission clock")
        return errors
    if model == BACKHAUL_SECONDS and getattr(cfg, "trial_seed", None) is None:
        errors.append("backhaul_model='seconds' needs trial_seed (the keyed loss draw's salt)")
    return errors


@dataclass
class DeviceConfig:
    """Settings for one edge-device process."""

    device_id: str
    # Mule whose RF this device connects to.
    mule_rf_host: str = "127.0.0.1"
    mule_rf_port: int = 0
    position: Position = (0.0, 0.0, 0.0)
    # Number of solicits to serve before exiting. None = run forever.
    n_serves: Optional[int] = None
    # EX-4.1 — real DNN-IDS training. When ``train_shard_path`` points at a
    # serialized ``(X, y)`` CICIOT shard, the device builds a real
    # ``local_train`` over it (experiments.exp4.model_task) instead of the
    # noise stub. ``input_dim`` must match the cluster's seeded model.
    # Left None -> the Sprint-2 stub trainer (backward compatible).
    train_shard_path: Optional[str] = None
    input_dim: Optional[int] = None
    local_epochs: int = 1
    local_batch_size: int = 64
    # EX-4.2 — per-device short-range contact reliability (device<->mule).
    # p that a Pass-1 collect delivers this device's Δθ, modelling Exp 3's
    # ``reliability x rf_factor`` completion. None -> always completes
    # (the EX-4.0/4.1 behaviour). Set by the driver from a seeded
    # Uniform(0.15, 1.0) reliability x distance falloff.
    contact_reliability: Optional[float] = None
    # FeRRy Phase 1 — FedProx proximal weight ρ: local training minimises
    # loss + (ρ/2)·‖θ − θ_received‖². 0 keeps the plain Keras fit.
    fedprox_rho: float = 0.0
    # FeRRy Phase 3 (critic B1) — answer only the newest queued solicit and
    # drop older ones (TCPRFLinkClient ``newest_solicit_only``). Off, solicits
    # are answered in arrival order, as in every recorded run. The ferry
    # topology turns it on: its mule matches adverts to the solicit it is
    # gathering for, so a stale one would be discarded and cost the device
    # its push wait through the next gather too.
    newest_solicit_only: bool = False
    # Freeze Amendment 10 — the token this device's RF registrations carry
    # (TCPRFLinkClient ``link_token``). A mule whose RF server was started
    # with a token (``TCPRFLinkServer(link_token=...)``) refuses a
    # registration carrying any other, so a device whose mule has exited
    # cannot re-dial into a mule of another trial that later binds the same
    # port and evict its device of the same id. All mules and devices of one
    # trial share one value. None (every recorded run) sends no token; a mule
    # without one accepts any.
    rf_link_token: Optional[str] = None
    # Exp 5 addendum (Study 5.12): the IDS model's architecture the device
    # trains (``model_task.MODEL_ARCHS``); None is the canonical CICIoT model.
    # Must match the cluster's. Written to the JSON only when set.
    model_arch: Optional[str] = None


#: ``ClusterConfig`` and ``DeviceConfig`` fields the Exp 5 addendum adds whose
#: None is every recorded run: their per-role JSON leaves each out while it is
#: None, so a recorded trial's JSON keeps its keys, and a reader takes the
#: default for an absent key.
CONFIG_FIELDS_OMITTED_AT_NONE: Tuple[str, ...] = ("model_arch",)


def _set_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in raw.items()
            if not (k in CONFIG_FIELDS_OMITTED_AT_NONE and v is None)}


class TopologyValidationError(ValueError):
    """Raised by :meth:`TopologyConfig.validate` on a malformed deployment."""


@dataclass
class TopologyConfig:
    """One AVN-shaped deployment description."""

    cluster: ClusterConfig
    mules: List[MuleConfig] = field(default_factory=list)
    devices: List[DeviceConfig] = field(default_factory=list)
    # L-M4: per-device → mule assignment. Populated by ``validate()`` from
    # MuleConfig.expected_devices, or round-robin if not specified. The
    # orchestrator reads this map (NOT MuleConfig.expected_devices
    # directly) to avoid the L-H1 bug where assignment depends on
    # config fields that haven't been populated yet.
    device_to_mule: Dict[str, str] = field(default_factory=dict)

    # ------------------------- Validation ---------------------------- #

    def validate(self) -> None:
        """Catch malformed topologies before subprocesses launch.

        Sprint 2 L-M4: empty mules with non-empty devices, dangling
        ``assigned_mule`` references, duplicate IDs, conflicting ports.
        Also populates :attr:`device_to_mule` so later steps don't have
        to re-derive assignment.
        """
        # Duplicate ID check.
        mule_ids = [m.mule_id for m in self.mules]
        if len(set(mule_ids)) != len(mule_ids):
            raise TopologyValidationError(
                f"duplicate mule_id in topology: {mule_ids}"
            )
        device_ids = [d.device_id for d in self.devices]
        if len(set(device_ids)) != len(device_ids):
            raise TopologyValidationError(
                f"duplicate device_id in topology: {device_ids}"
            )

        # Mule with devices but no mule.
        if self.devices and not self.mules:
            raise TopologyValidationError(
                f"{len(self.devices)} devices configured but no mules to serve them"
            )

        # Conflicting non-zero ports across mules.
        nonzero_rf = [m.rf_port for m in self.mules if m.rf_port != 0]
        if len(set(nonzero_rf)) != len(nonzero_rf):
            raise TopologyValidationError(
                f"conflicting non-zero rf_port across mules: {nonzero_rf}"
            )

        # Build / validate the device → mule assignment.
        # Strategy:
        #   1. If MuleConfig.expected_devices is populated, honour it.
        #   2. Otherwise, round-robin distribute devices across mules
        #      in declaration order (deterministic).
        # An assigned_mule that doesn't reference a real mule is rejected.
        explicit: Dict[str, str] = {}
        for m in self.mules:
            for did in m.expected_devices:
                if did in explicit:
                    raise TopologyValidationError(
                        f"device {did!r} is in expected_devices of multiple mules"
                    )
                if did not in device_ids:
                    raise TopologyValidationError(
                        f"mule {m.mule_id!r} expected_devices references "
                        f"unknown device {did!r}"
                    )
                explicit[did] = m.mule_id

        # Round-robin everything not explicitly claimed.
        assignment: Dict[str, str] = dict(explicit)
        unclaimed = [d.device_id for d in self.devices if d.device_id not in explicit]
        for i, did in enumerate(unclaimed):
            if not self.mules:
                # Empty-devices case already raised above; defensive.
                break
            assignment[did] = self.mules[i % len(self.mules)].mule_id

        self.device_to_mule = assignment
        self._validate_clock(assignment)

    def _validate_clock(self, assignment: Dict[str, str]) -> None:
        """FeRRy Phase 3: refuse clock settings that cannot run or would mis-measure.

        Every check fires only on a non-default setting, so a recorded
        topology validates as it always did.

        * Each role's own settings (:func:`mule_config_errors`,
          :func:`cluster_config_errors`).
        * The cluster and every mule on one clock, one backhaul model, one
          trial seed and one set of band classes.
        * Several mules on the simulated clock below a quorum of every mule,
          or under ``agg:fedbuff``, run (critic B9's refusal is lifted: the
          cluster folds their uploads in simulated-time order, unit U9,
          ``hermes.processes.cluster.SimOrderGate``), but only with a
          ``down_wait_s`` on every mule: the cluster may hold an upload until
          the other mules pass it in simulated time, and the recorded 10 s
          DOWN wait's expiry would end the mule's run.
        * The channel reliability source with devices that still draw their
          own reliability: the failure would be drawn twice.
        * A mule's causal RF prior schedule without the cluster's per-mission
          loss schedule of the same length: the prior and the losses must
          come from one L1 trace (critic B4).
        * One RF link token per mule and its devices (Amendment 10): a mule
          with a token refuses a device registering without it.
        """
        errors: List[str] = []
        for m in self.mules:
            errors += [f"mule {m.mule_id!r}: {e}" for e in mule_config_errors(m)]
        errors += [f"cluster: {e}" for e in cluster_config_errors(self.cluster)]
        if errors:
            raise TopologyValidationError("; ".join(errors))

        c = self.cluster
        clock = getattr(c, "mission_clock", CLOCK_WALL)
        for m in self.mules:
            if m.mission_clock != clock:
                raise TopologyValidationError(
                    f"mule {m.mule_id!r} runs mission_clock={m.mission_clock!r} but the "
                    f"cluster runs {clock!r}: one trial runs on one clock"
                )
            if m.backhaul_model != getattr(c, "backhaul_model", BACKHAUL_MISSION):
                raise TopologyValidationError(
                    f"mule {m.mule_id!r} prices backhaul_model={m.backhaul_model!r} but the "
                    f"cluster draws losses under {c.backhaul_model!r}"
                )
        if clock == CLOCK_SIM:
            if c.backhaul_model == BACKHAUL_SECONDS and any(
                m.trial_seed != c.trial_seed for m in self.mules
            ):
                raise TopologyValidationError(
                    "the mules and the cluster must share one trial_seed: the keyed "
                    "backhaul draw and the channel are salted with it"
                )
            if any(m.contact_band_classes != c.contact_band_classes for m in self.mules):
                raise TopologyValidationError(
                    "the mules and the cluster must name the same contact_band_classes: "
                    "the cluster reads each report line's band index with them"
                )
            k = len(self.mules)
            ordered = k > 1 and (
                int(c.min_participation) < k or c.aggregation == _AGG_FEDBUFF
            )
            waitless = [m.mule_id for m in self.mules if ordered and m.down_wait_s is None]
            if waitless:
                raise TopologyValidationError(
                    f"mules {waitless} have no down_wait_s: with {k} mules on the "
                    f"simulated clock below a full quorum (min_participation="
                    f"{c.min_participation}, {c.aggregation}) the cluster holds each upload "
                    f"until the other mules pass it in simulated time (unit U9), and the "
                    f"recorded 10 s DOWN wait's expiry would end the mule's run"
                )
            channel_mules = {
                m.mule_id for m in self.mules if m.contact_reliability_source == "channel"
            }
            doubled = [
                d.device_id for d in self.devices
                if assignment.get(d.device_id) in channel_mules
                and d.contact_reliability is not None
            ]
            if doubled:
                raise TopologyValidationError(
                    f"devices {doubled} draw their own contact_reliability, but their "
                    f"mule's contact_reliability_source is 'channel': the failure would "
                    f"be drawn twice; build them with contact_reliability=None"
                )
            n_losses = len(c.backhaul_loss_schedule or ())
            unmatched = [
                m.mule_id for m in self.mules
                if m.rf_prior_schedule_db is not None
                and len(m.rf_prior_schedule_db) != n_losses
            ]
            if unmatched:
                raise TopologyValidationError(
                    f"mules {unmatched} carry an rf_prior_schedule_db but the cluster's "
                    f"backhaul_loss_schedule has {n_losses} entries: the RF prior and the "
                    f"losses must come from one L1 trace (critic B4)"
                )
        tokens = {m.mule_id: m.rf_link_token for m in self.mules}
        mismatched = [
            d.device_id for d in self.devices
            if getattr(d, "rf_link_token", None) != tokens.get(assignment.get(d.device_id))
        ]
        if mismatched:
            raise TopologyValidationError(
                f"devices {mismatched} carry an rf_link_token other than their mule's: "
                f"the mule would refuse their registrations (Amendment 10)"
            )

    def mule_for(self, device_id: str) -> str:
        """Return the mule assigned to ``device_id`` post-:meth:`validate`."""
        if not self.device_to_mule:
            raise TopologyValidationError(
                "topology not validated yet — call validate() first"
            )
        try:
            return self.device_to_mule[device_id]
        except KeyError:
            raise TopologyValidationError(
                f"no mule assignment for device {device_id!r}"
            )

    def devices_of(self, mule_id: str) -> List[str]:
        """Return device ids assigned to ``mule_id`` post-:meth:`validate`."""
        if not self.device_to_mule:
            raise TopologyValidationError(
                "topology not validated yet — call validate() first"
            )
        return [d for d, m in self.device_to_mule.items() if m == mule_id]

    # ------------------------- JSON helpers -------------------------- #

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "TopologyConfig":
        raw = json.loads(payload)
        return cls(
            cluster=ClusterConfig(**raw["cluster"]),
            mules=[MuleConfig(**m) for m in raw["mules"]],
            devices=[DeviceConfig(**d) for d in raw["devices"]],
            device_to_mule=dict(raw.get("device_to_mule", {})),
        )

    @classmethod
    def from_file(cls, path: Path) -> "TopologyConfig":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# Per-role config helpers — entry points read JSON of just one of these
# rather than the whole topology, so a single mule process doesn't see
# device positions it has no need for.

def cluster_config_to_json(cfg: ClusterConfig) -> str:
    return json.dumps(_set_fields(asdict(cfg)), indent=2)


def cluster_config_from_json(payload: str) -> ClusterConfig:
    return ClusterConfig(**json.loads(payload))


def mule_config_to_json(cfg: MuleConfig) -> str:
    return json.dumps(asdict(cfg), indent=2)


def mule_config_from_json(payload: str) -> MuleConfig:
    return MuleConfig(**json.loads(payload))


def device_config_to_json(cfg: DeviceConfig) -> str:
    raw = _set_fields(asdict(cfg))
    # asdict converts the position tuple to a list — preserve the
    # tuple-shape on the inverse via a custom decoder below.
    return json.dumps(raw, indent=2)


def device_config_from_json(payload: str) -> DeviceConfig:
    raw = json.loads(payload)
    pos = raw.get("position", (0.0, 0.0, 0.0))
    if isinstance(pos, list):
        pos = tuple(pos)
    raw["position"] = pos
    return DeviceConfig(**raw)
