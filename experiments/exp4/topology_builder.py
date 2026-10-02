"""Build a finite, natural-exit topology for one Experiment-4 trial.

EX-4.0 arm **H1**: 1 cluster + 1 mule + N devices, the mule capped at
``n_missions`` so the whole process tree exits on its own once the
missions are done (the driver then reads the JSONL logs). Devices are
placed in a tight cluster near the origin so the Four-Stage Gated
Scheduler's S3a range-clustering reliably forms at least one contact
event within ``rf_range_m`` — the point of EX-4.0 is to measure the real
two-pass path, not to stress contact formation (that is the sweep's job
once the plumbing is proven).

Positions are seeded off the trial's paired seed so the same
``(cell, trial_index)`` lays devices out identically across arms — the
paired-seed property the analysis relies on.

FeRRy Phase 2 — ``n_mules`` > 1 splits the same seeded devices between K
mules. The default split is spatial (:func:`angular_slices`): K contiguous
sectors by angle around the dock at the origin, so each mule tours its own
part of the field; every mule starts at the dock. ``slice_assignment``
overrides it (arm D4 passes its CARP assignment). ``n_mules`` = 1 is the
recorded single-mule topology, built by the same code as before.

FeRRy Phase 3 — ``mission_clock="sim"`` builds a ferry cell (design section
4.8): the mules and the cluster run on the simulated mission clock with the
trial seed as their channel salt and ``ferry_settings`` (``MuleConfig`` ferry
fields: band, response, backhaul, payload, D1-D3 physics); devices answer only
the newest solicit (``newest_solicit_only``, critic B1); and under
``contact_reliability_source="channel"`` the devices stop drawing their own
reliability (``contact_reliability=None``) while each mule receives its
slice's ground-truth availability ``{device_id: rel_i}`` from
``device_reliabilities(seed, N)`` for its keyed draw. The positions, the
legacy reliability formula and the device-id stream are unchanged, and with
the defaults every config is the recorded one.

FeRRy Phase 4 — the plan fields (``hermes.processes.config.PLAN_MULE_FIELDS``:
``plan_mode``, the band-class policy, member admission, the flight slot, the
age cap S and its lookahead, and the plan score's and search's settings) are
explicit parameters, copied to every mule; they are not ``ferry_settings``,
which hold only ferry-spec fields (critic A8). Their defaults are the recorded
mule, and off the simulated clock the topology's validation accepts only the
defaults.

FeRRy Phase 5 — a learned arm's checkpoint (``MuleConfig``'s six checkpoint
fields: the path, sha256 and tag of the pair score's checkpoint, which goes
with ``flight_slot="pair_q"``, or of arm E3's, which goes with
``contact_policy="chen_dqn"``) is an explicit parameter too, copied to every
mule. None, the default, builds the recorded mule; the topology's validation
refuses the fields off the simulated clock and beside any other slot or policy.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hermes.mission.aggregation_rules import AGG_FEDBUFF, AggregationSpec
from hermes.processes import (
    ClusterConfig,
    DeviceConfig,
    MuleConfig,
    TopologyConfig,
)
from hermes.processes.config import (
    BAND_POLICY_SEARCH,
    CLOCK_SIM,
    CLOCK_WALL,
    FERRY_SPEC_FIELDS,
    FLIGHT_SLOT_COMMITTED,
    MEMBER_ADMISSION_WHOLE,
    MISSION_CLOCKS,
    PLAN_MODE_LEGACY,
)

#: The cluster's synthetic batch size in every Exp 4 topology (the builder's
#: default); the driver prices the pushed batch with it (FeRRy Phase 3).
SYNTH_BATCH_SIZE = 2
#: The mule's wall-clock session TTL in every Exp 4 topology unless the
#: driver sets one (the builder's default); the driver re-costs its wall
#: budget from it (FeRRy Phase 3, critic B14).
SESSION_TTL_S = 3.0


def device_spread_m(
    rf_range_m: float, *, spread_m: Optional[float] = None,
    field_radius_m: Optional[float] = None,
) -> float:
    """Half-width of the square the devices are scattered in.

    ``spread_m`` when given; else the realism field radius (EX-4.2, devices
    spread so S3a forms several contacts); else the tight EX-4.0/4.1 cluster,
    a fraction of the RF range capped at 25 m.
    """
    if spread_m is not None:
        return spread_m
    return field_radius_m if field_radius_m is not None else min(rf_range_m * 0.4, 25.0)


def device_positions(n_devices: int, seed: int, spread_m: float) -> List[Tuple[float, float]]:
    """The devices' (x, y), drawn exactly as every recorded trial drew them.

    ``random.Random(seed)``, then x and y per device in index order, each
    uniform on [-spread_m, spread_m]. The Exp 4 driver draws its T_nom
    reference layouts with it (FeRRy Phase 3).
    """
    rng = random.Random(seed)
    out: List[Tuple[float, float]] = []
    for _ in range(n_devices):
        x = rng.uniform(-spread_m, spread_m)
        y = rng.uniform(-spread_m, spread_m)
        out.append((float(x), float(y)))
    return out


def angular_slices(
    positions: Sequence[Tuple[float, float]], n_mules: int,
) -> Dict[int, int]:
    """Device index -> mule index: K contiguous sectors by angle around the origin.

    Devices are ordered by their angle around the dock, starting just after
    the widest empty arc, so the one sector boundary that is not between two
    slices' neighbours falls where there are no devices. The order is cut into
    K runs whose sizes differ by at most one, so every mule gets a slice and
    none gets more than its share. Deterministic: ties in angle go by index.
    """
    n = len(positions)
    if n_mules < 1:
        raise ValueError(f"n_mules must be >= 1, got {n_mules}")
    if n < n_mules:
        raise ValueError(
            f"{n} devices cannot fill {n_mules} disjoint slices; each mule needs "
            f"at least one device"
        )
    angles = [math.atan2(float(y), float(x)) for x, y in positions]
    order = sorted(range(n), key=lambda i: (angles[i], i))
    if n > 1:
        # Gap after order[j], wrapping from the last device back to the first.
        gaps = [
            (angles[order[(j + 1) % n]] - angles[order[j]]) % (2.0 * math.pi)
            for j in range(n)
        ]
        widest = max(range(n), key=lambda j: (gaps[j], -j))
        start = (widest + 1) % n
        order = order[start:] + order[:start]
    base, extra = divmod(n, n_mules)
    out: Dict[int, int] = {}
    cursor = 0
    for k in range(n_mules):
        size = base + (1 if k < extra else 0)
        for i in order[cursor:cursor + size]:
            out[i] = k
        cursor += size
    return out


def _check_multi_mule(
    *,
    n_mules: int,
    n_devices: int,
    min_participation: int,
    aggregation: str,
    aggregation_params: Optional[dict],
    dock_on_empty: bool,
    slice_assignment: Optional[Mapping[int, int]],
) -> None:
    """Refuse multi-mule settings that would silently mis-measure or stall."""
    if not 1 <= int(min_participation) <= n_mules:
        raise ValueError(
            f"min_participation must be in 1..n_mules={n_mules}, got {min_participation}"
        )
    spec = AggregationSpec.from_config(aggregation, aggregation_params)
    if spec.is_plain and int(min_participation) != n_mules:
        # agg:plain overwrites θ with the mean of the partials in the merge.
        # With fewer than every mule in it, each merge replaces the other
        # mules' work with one mule's models: last writer wins, not FedAvg.
        raise ValueError(
            f"agg:plain with {n_mules} mules needs min_participation={n_mules} "
            f"(got {min_participation}): a smaller quorum makes each merge "
            f"overwrite θ with one mule's models. Use an age-aware rule for "
            f"asynchronous merges."
        )
    if 1 < int(min_participation) < n_mules and spec.rule != AGG_FEDBUFF:
        # Each merge takes the first quorum of mules to upload, so the mules'
        # partials pair up in no fixed order, and near the end of the run the
        # last mule's final partial can wait for mules that have finished.
        raise ValueError(
            f"min_participation={min_participation} with {n_mules} mules: use 1 "
            f"(asynchronous merges) or {n_mules} (every mule in each merge); a "
            f"quorum between them can leave the last partial of the run waiting "
            f"for mules that have already finished"
        )
    if int(min_participation) > 1 and spec.rule != AGG_FEDBUFF and not dock_on_empty:
        raise ValueError(
            f"min_participation={min_participation} needs dock_on_empty: a mule "
            f"whose mission collects nothing would never dock, and the quorum "
            f"could never close"
        )
    if slice_assignment is not None:
        keys = sorted(int(i) for i in slice_assignment)
        if keys != list(range(n_devices)):
            raise ValueError(
                f"slice_assignment must map every device index 0..{n_devices - 1} "
                f"exactly once, got indices {keys}"
            )
        bad = {int(i): int(k) for i, k in slice_assignment.items()
               if not 0 <= int(k) < n_mules}
        if bad:
            raise ValueError(
                f"slice_assignment sends devices outside mules 0..{n_mules - 1}: {bad}"
            )


def build_exp4_topology(
    *,
    n_devices: int,
    rf_range_m: float,
    n_missions: int,
    seed: int,
    spread_m: Optional[float] = None,
    session_ttl_s: float = SESSION_TTL_S,
    synth_batch_size: int = SYNTH_BATCH_SIZE,
    min_participation: int = 1,
    cluster_id: str = "exp4-cluster",
    mule_id: str = "exp4-mule",
    # EX-4.1 real-model wiring (all optional; omitted -> EX-4.0 stub path).
    train_shard_paths: Optional[List[str]] = None,
    input_dim: Optional[int] = None,
    local_epochs: int = 1,
    local_batch_size: int = 64,
    init_theta_path: Optional[str] = None,
    eval_test_path: Optional[str] = None,
    # EX-4.2 realism wiring (all optional; omitted -> ideal links).
    device_reliability: bool = False,
    reliabilities: Optional[List[float]] = None,
    world_radius_m: float = 100.0,
    field_radius_m: Optional[float] = None,
    backhaul_loss_pct: float = 0.0,
    backhaul_rng_seed: Optional[int] = None,
    # EX-4.2 arm H2 — RL target selector on the mule.
    use_rl_selector: bool = False,
    selector_weights_path: Optional[str] = None,
    # EX-4.3 arm H3 — L1 channel model: per-mission backhaul-loss schedule
    # (cluster) + the chosen channel's mean SNR as the selector's RF prior (mule).
    backhaul_loss_schedule: Optional[List[float]] = None,
    rf_prior_snr_db: Optional[float] = None,
    # S3b — per-mission time budget; when set, the deadline is ENFORCED.
    mission_budget_s: Optional[float] = None,
    # SOTA baseline arm: None = our scheduler, "max_aoi" = the AoI comparator.
    contact_policy: Optional[str] = None,
    # S3c — mission-level window adaptation; off reproduces recorded sweeps.
    mission_window_adaptation: bool = False,
    mission_window_history: int = 5,
    mission_window_target: float = 0.8,
    mission_window_gain: float = 2.0,
    mission_window_max_scale: float = 4.0,
    # FeRRy Phase 1 — the L3 merge rule (set on cluster AND mule so they
    # agree), FedProx on the devices, and the budgeted Pass 2.
    aggregation: str = "agg:plain",
    aggregation_params: Optional[dict] = None,
    fedprox_rho: float = 0.0,
    pass_2_budget: bool = False,
    # FeRRy Phase 1 — the mule scheduler's deadline law and priority key.
    deadline_law: str = "additive",
    deadline_params: Optional[dict] = None,
    miss_priority: bool = False,
    # FeRRy Phase 2 — several mules. ``n_mules`` = 1 is the recorded topology.
    n_mules: int = 1,
    slice_assignment: Optional[Mapping[int, int]] = None,
    down_wait_s: Optional[float] = None,
    dock_on_empty: bool = False,
    # FeRRy Phase 2 — options of the D3/D5 policies (their defaults).
    whittle_variant: str = "expected",
    whittle_weights: str = "uniform",
    fedcs_value: str = "unit",
    # FeRRy Phase 3 — the mission clock and the contact link. The defaults
    # build the recorded wall-clock topology.
    mission_clock: str = CLOCK_WALL,
    ferry_settings: Optional[Mapping[str, Any]] = None,
    deadline_time_scale: float = 1.0,
    initial_window_s: Optional[float] = None,
    rf_link_token: Optional[str] = None,
    newest_solicit_only: Optional[bool] = None,
    rf_prior_schedule_db: Optional[Sequence[float]] = None,
    # FeRRy Phase 4 — the plan clock (MuleConfig's plan fields). The defaults
    # build the recorded mule.
    plan_mode: str = PLAN_MODE_LEGACY,
    band_class_policy: str = BAND_POLICY_SEARCH,
    member_admission: str = MEMBER_ADMISSION_WHOLE,
    flight_slot: str = FLIGHT_SLOT_COMMITTED,
    age_cap_missions: Optional[int] = None,
    age_cap_lookahead: int = 0,
    plan_score_params: Optional[Mapping[str, Any]] = None,
    plan_search_params: Optional[Mapping[str, Any]] = None,
    # FeRRy Phase 5 — a learned arm's checkpoint (MuleConfig's checkpoint
    # fields). The defaults build the recorded mule.
    pair_checkpoint: Optional[str] = None,
    pair_checkpoint_sha256: Optional[str] = None,
    pair_checkpoint_tag: Optional[str] = None,
    policy_checkpoint: Optional[str] = None,
    policy_checkpoint_sha256: Optional[str] = None,
    policy_checkpoint_tag: Optional[str] = None,
) -> TopologyConfig:
    """Return a validated :class:`TopologyConfig` for one H1 trial.

    ``spread_m`` bounds the square the devices are scattered in; it
    defaults to a fraction of ``rf_range_m`` (capped) so the cluster
    stays inside one contact radius.

    When ``train_shard_paths`` is given (EX-4.1 real-model path), device
    ``i`` is pointed at ``train_shard_paths[i]`` and the cluster is seeded
    from ``init_theta_path`` + scored on ``eval_test_path``.

    ``n_mules`` > 1 (FeRRy Phase 2) builds mules ``<mule_id>-0`` ..
    ``<mule_id>-<K-1>`` over the same devices, drawn from the same seed. Each
    mule gets the devices ``slice_assignment`` (device index -> mule index)
    gives it, or by default a contiguous angular sector
    (:func:`angular_slices`), as its explicit ``expected_devices``; the
    cluster is seeded with the same assignment. Refused: ``agg:plain`` with a
    quorum smaller than every mule, a quorum strictly between 1 and every
    mule, and a quorum above 1 without ``dock_on_empty`` (FedBuff, whose K is
    its own quorum, is exempt from the last two). ``down_wait_s`` and ``dock_on_empty`` go to every mule
    as given, whatever ``n_mules``.

    FeRRy Phase 3. ``mission_clock="sim"`` builds a ferry cell (module
    docstring): ``ferry_settings`` maps ``MuleConfig`` ferry fields
    (``hermes.processes.config.FERRY_SPEC_FIELDS``, the ground-truth
    ``device_availability`` excepted: the builder derives it) to their
    values; the trial seed becomes every role's ``trial_seed``; the mules
    carry ``input_dim``. ``deadline_time_scale`` and ``initial_window_s``
    restate the deadline law's time unit on either clock (1.0 and None are
    the recorded law). ``rf_link_token`` goes to every mule and device
    (Amendment 10). ``newest_solicit_only`` None turns it on exactly on the
    simulated clock. ``rf_prior_schedule_db`` (simulated clock only) is the
    L1 trace's SNR at each mission's upload, the causal RF prior every mule
    adopts mission by mission under the recorded ``mission`` backhaul model
    (critic B4; ``MuleConfig.rf_prior_schedule_db``). The topology's own
    validation refuses what cannot run (``TopologyConfig.validate``: critic
    B4 and B16, and an ordered simulated-clock topology without
    ``down_wait_s``; critic B9's refusal was lifted by unit U9).

    FeRRy Phase 4. ``plan_mode``, ``band_class_policy``, ``member_admission``,
    ``flight_slot``, ``age_cap_missions``, ``age_cap_lookahead``,
    ``plan_score_params`` and ``plan_search_params`` are the ``MuleConfig``
    plan fields every mule gets (a plan arm's, or an H or D arm's member
    admission). The topology's validation refuses any but their defaults off
    the simulated clock, and on it what plan mode cannot fly
    (``hermes.processes.config.mule_config_errors``).

    FeRRy Phase 5. ``pair_checkpoint``, ``pair_checkpoint_sha256`` and
    ``pair_checkpoint_tag`` (an FQ arm's, with ``flight_slot="pair_q"``), and
    ``policy_checkpoint``, ``policy_checkpoint_sha256`` and
    ``policy_checkpoint_tag`` (arm E3's, with ``contact_policy="chen_dqn"``),
    are the checkpoint fields every mule gets as given; each mule verifies its
    file when it starts. The topology's validation refuses them off the
    simulated clock, beside any other slot or policy, and a switch without
    all three (``mule_config_errors``: no random-init arm).
    """
    if mission_clock not in MISSION_CLOCKS:
        raise ValueError(f"mission_clock must be one of {MISSION_CLOCKS}, got {mission_clock!r}")
    sim = mission_clock == CLOCK_SIM
    # FeRRy Phase 4: the plan fields go to every mule as given; off the
    # simulated clock the topology's validation refuses any but the defaults,
    # and on it what plan mode cannot fly (``mule_config_errors``).
    plan = dict(
        plan_mode=plan_mode, band_class_policy=band_class_policy,
        member_admission=member_admission, flight_slot=flight_slot,
        age_cap_missions=age_cap_missions, age_cap_lookahead=age_cap_lookahead,
        plan_score_params=dict(plan_score_params or {}),
        plan_search_params=dict(plan_search_params or {}),
    )
    # FeRRy Phase 5: a learned arm's checkpoint fields go to every mule as
    # given (None, the recorded mule's); the validation refuses misplaced ones.
    checkpoints = dict(
        pair_checkpoint=pair_checkpoint, pair_checkpoint_sha256=pair_checkpoint_sha256,
        pair_checkpoint_tag=pair_checkpoint_tag, policy_checkpoint=policy_checkpoint,
        policy_checkpoint_sha256=policy_checkpoint_sha256,
        policy_checkpoint_tag=policy_checkpoint_tag,
    )
    ferry = dict(ferry_settings or {})
    unknown = sorted(set(ferry) - set(FERRY_SPEC_FIELDS))
    if unknown:
        raise ValueError(
            f"ferry_settings keys {unknown} are not MuleConfig ferry fields "
            f"({sorted(FERRY_SPEC_FIELDS)})"
        )
    if "device_availability" in ferry:
        raise ValueError(
            "ferry_settings must not carry device_availability: the builder derives "
            "it from the trial's reliability draw"
        )
    if ferry and not sim:
        raise ValueError(
            f"ferry_settings {sorted(ferry)}: only on the simulated mission clock; "
            f"pass mission_clock='sim'"
        )
    if rf_prior_schedule_db is not None and not sim:
        raise ValueError(
            "rf_prior_schedule_db is the causal RF prior of the simulated mission "
            "clock; pass mission_clock='sim' (the wall clock takes rf_prior_snr_db)"
        )
    channel_source = sim and ferry.get("contact_reliability_source", "origin") == "channel"
    if n_devices < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_devices}")
    if n_missions < 1:
        raise ValueError(f"n_missions must be >= 1, got {n_missions}")
    if n_mules < 1:
        raise ValueError(f"n_mules must be >= 1, got {n_mules}")
    if train_shard_paths is not None and len(train_shard_paths) != n_devices:
        raise ValueError(
            f"train_shard_paths has {len(train_shard_paths)} entries, "
            f"expected n_devices={n_devices}"
        )
    if n_mules > 1:
        _check_multi_mule(
            n_mules=n_mules, n_devices=n_devices,
            min_participation=min_participation,
            aggregation=aggregation, aggregation_params=aggregation_params,
            dock_on_empty=dock_on_empty, slice_assignment=slice_assignment,
        )
    elif slice_assignment is not None and any(int(k) != 0 for k in slice_assignment.values()):
        raise ValueError("slice_assignment names a mule other than 0, but n_mules=1")

    # field_radius_m (EX-4.2) spreads devices across the field so S3a forms
    # multiple contacts; otherwise the tight EX-4.0/4.1 cluster.
    spread_m = device_spread_m(rf_range_m, spread_m=spread_m, field_radius_m=field_radius_m)
    # Shared per-device reliability draw (same values H0 uses) — set by the
    # driver for a paired comparison; fall back to the canonical draw so the
    # builder is usable standalone.
    if device_reliability and reliabilities is None:
        from .model_task import device_reliabilities as _dr
        reliabilities = _dr(seed, n_devices)
    # FeRRy Phase 3 (design section 4.8): under the channel reliability
    # source the same draw is each device's ground-truth availability rel_i,
    # drawn on its mule; the devices then draw nothing themselves.
    availability_rel: Optional[List[float]] = None
    if channel_source:
        if reliabilities is None:
            from .model_task import device_reliabilities as _dr
            reliabilities = _dr(seed, n_devices)
        availability_rel = [float(r) for r in reliabilities]
    if newest_solicit_only is None:
        newest_solicit_only = sim
    device_extra: Dict[str, Any] = {}
    if newest_solicit_only:
        device_extra["newest_solicit_only"] = True
    if rf_link_token is not None:
        device_extra["rf_link_token"] = str(rf_link_token)

    devices: List[DeviceConfig] = []
    for i, (x, y) in enumerate(device_positions(n_devices, seed, spread_m)):
        contact_reliability: Optional[float] = None
        if device_reliability and not channel_source:
            # Short-range device<->mule completion: p = reliability x rf_factor
            # (Exp 3's model). ``reliability`` is the shared per-device draw;
            # rf_factor = max(0.4, 1 - d_eff/(3*world_radius)) with d_eff the
            # device's distance to the mule's contact stop, bounded by rf_range
            # (the mule flies to within rf_range). This is REGIME-INDEPENDENT:
            # jitter degrades long-range links, not this short hop — the whole
            # point of routing collection through the mule. The jittery cost
            # falls only on the mule's one long-range backhaul upload.
            rel_i = float(reliabilities[i]) if reliabilities else 0.575
            d = (float(x) ** 2 + float(y) ** 2) ** 0.5
            d_eff = min(d, rf_range_m)
            rf_factor = max(0.4, 1.0 - d_eff / (3.0 * world_radius_m))
            contact_reliability = max(0.0, min(1.0, rel_i * rf_factor))
        devices.append(
            DeviceConfig(
                device_id=f"exp4-dev-{i:03d}",
                position=(float(x), float(y), 0.0),
                train_shard_path=(
                    train_shard_paths[i] if train_shard_paths else None
                ),
                input_dim=input_dim,
                local_epochs=local_epochs,
                local_batch_size=local_batch_size,
                contact_reliability=contact_reliability,
                fedprox_rho=float(fedprox_rho),
                **device_extra,
            )
        )

    classes = ferry.get("contact_band_classes")
    if classes is not None:
        classes = [str(c) for c in classes]
        ferry["contact_band_classes"] = classes
    cluster_extra: Dict[str, Any] = {}
    if sim:
        cluster_extra = dict(
            mission_clock=CLOCK_SIM,
            backhaul_model=str(ferry.get("backhaul_model", "mission")),
            trial_seed=int(seed),
            contact_band_classes=None if classes is None else list(classes),
        )
    cluster = ClusterConfig(
        cluster_id=cluster_id,
        dock_host="127.0.0.1",
        dock_port=0,
        synth_batch_size=synth_batch_size,
        min_participation=min_participation,
        init_theta_path=init_theta_path,
        eval_test_path=eval_test_path,
        input_dim=input_dim,
        backhaul_loss_pct=backhaul_loss_pct,
        backhaul_rng_seed=backhaul_rng_seed,
        backhaul_loss_schedule=backhaul_loss_schedule,
        aggregation=str(aggregation),
        aggregation_params=dict(aggregation_params or {}),
        **cluster_extra,
    )
    mule_extra: Dict[str, Any] = {}
    if sim:
        mule_extra.update(ferry)
        mule_extra.update(mission_clock=CLOCK_SIM, trial_seed=int(seed), input_dim=input_dim)
        if availability_rel is not None:
            mule_extra["device_availability"] = {
                dev.device_id: rel for dev, rel in zip(devices, availability_rel)
            }
        if rf_prior_schedule_db is not None:
            mule_extra["rf_prior_schedule_db"] = [float(v) for v in rf_prior_schedule_db]
    if isinstance(deadline_time_scale, bool) or deadline_time_scale != 1.0:
        mule_extra["deadline_time_scale"] = float(deadline_time_scale)
    if initial_window_s is not None:
        mule_extra["initial_window_s"] = float(initial_window_s)
    if rf_link_token is not None:
        mule_extra["rf_link_token"] = str(rf_link_token)
    mule = MuleConfig(
        mule_id=mule_id,
        rf_host="127.0.0.1",
        rf_port=0,
        rf_range_m=float(rf_range_m),
        session_ttl_s=session_ttl_s,
        n_missions=int(n_missions),
        use_rl_selector=use_rl_selector,
        selector_weights_path=selector_weights_path,
        rf_prior_snr_db=rf_prior_snr_db,
        mission_budget_s=mission_budget_s,
        contact_policy=contact_policy,
        mission_window_adaptation=bool(mission_window_adaptation),
        mission_window_history=int(mission_window_history),
        mission_window_target=float(mission_window_target),
        mission_window_gain=float(mission_window_gain),
        mission_window_max_scale=float(mission_window_max_scale),
        aggregation=str(aggregation),
        aggregation_params=dict(aggregation_params or {}),
        pass_2_budget=bool(pass_2_budget),
        deadline_law=str(deadline_law),
        deadline_params=dict(deadline_params or {}),
        miss_priority=bool(miss_priority),
        down_wait_s=(None if down_wait_s is None else float(down_wait_s)),
        dock_on_empty=bool(dock_on_empty),
        whittle_variant=str(whittle_variant),
        whittle_weights=str(whittle_weights),
        fedcs_value=str(fedcs_value),
        **plan,
        **checkpoints,
        **mule_extra,
    )
    if n_mules == 1:
        topo = TopologyConfig(cluster=cluster, mules=[mule], devices=devices)
    else:
        topo = _split_between_mules(
            cluster, mule, devices, n_mules=n_mules, slice_assignment=slice_assignment,
        )
    topo.validate()
    return topo


def _split_between_mules(
    cluster: ClusterConfig,
    mule: MuleConfig,
    devices: List[DeviceConfig],
    *,
    n_mules: int,
    slice_assignment: Optional[Mapping[int, int]],
) -> TopologyConfig:
    """K copies of ``mule`` over disjoint slices of ``devices``, and a cluster
    seeded with the same assignment.

    Each copy differs only in its id and ``expected_devices``, so every mule
    runs the same scheduler, rule and budget. The orchestrator wires each
    device to its mule's RF port from ``expected_devices`` and seeds the
    cluster's registry from the same map; ``seed_devices`` and
    ``expected_mules`` are filled here too so the cluster config says so on
    its own.
    """
    if slice_assignment is None:
        assignment = angular_slices(
            [(d.position[0], d.position[1]) for d in devices], n_mules,
        )
    else:
        assignment = {int(i): int(k) for i, k in slice_assignment.items()}
    mule_ids = [f"{mule.mule_id}-{k}" for k in range(n_mules)]
    slices: List[List[str]] = [[] for _ in range(n_mules)]
    for i, dev in enumerate(devices):
        slices[assignment[i]].append(dev.device_id)
    mules = [
        replace(
            mule, mule_id=mid, expected_devices=list(slices[k]),
            # Own copies, so no two mule configs share a mutable field.
            aggregation_params=dict(mule.aggregation_params),
            deadline_params=dict(mule.deadline_params),
            # FeRRy Phase 3: each mule holds only its own slice's ground truth.
            device_availability={
                d: a for d, a in mule.device_availability.items() if d in slices[k]
            },
            contact_band_classes=(
                None if mule.contact_band_classes is None else list(mule.contact_band_classes)
            ),
            # ... and its own copy of the per-mission RF prior (critic B4).
            rf_prior_schedule_db=(
                None if mule.rf_prior_schedule_db is None else list(mule.rf_prior_schedule_db)
            ),
            # FeRRy Phase 4: every mule plans with the same settings, each
            # from its own copy (at K > 1 plan mode runs per slice, critic B10).
            plan_score_params=dict(mule.plan_score_params),
            plan_search_params=dict(mule.plan_search_params),
        )
        for k, mid in enumerate(mule_ids)
    ]
    cluster = replace(
        cluster,
        expected_mules=list(mule_ids),
        seed_devices=[
            {
                "device_id": dev.device_id,
                "position": list(dev.position),
                "assigned_mule": mule_ids[assignment[i]],
            }
            for i, dev in enumerate(devices)
        ],
    )
    return TopologyConfig(cluster=cluster, mules=mules, devices=devices)
