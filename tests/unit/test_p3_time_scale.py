"""FeRRy Phase 3 (unit U4): the deadline law's time unit, Φ₀, overrides, spectrum.

Spec Q1: ``deadline_time_scale`` (recorded 1.0) moves every time constant of
the deadline law together — the additive steps and floor, the multiplicative
clamps — and ``initial_window_s`` sets Φ₀ (None: the recorded 60 s in the
law's unit). At the recorded values everything is exactly as before; the
supervisor goldens replayed with the switches set explicitly are in
``test_p3_legacy_equivalence.py``. Also: the "Φ₀ in missions" helpers (critic
A7), the refusal of cluster deadline overrides on the simulated clock (critic
B3), the spectrum fold (design §4.7), and the waypoint annotations that must
not change equality or hashing (design §4.5).
"""

from __future__ import annotations

import dataclasses
import math
import random
from types import SimpleNamespace

import pytest

from hermes.scheduler import FLScheduler, FLSchedulerError
from hermes.scheduler.fl_scheduler import (
    median_nominal_mission_period_s,
    nominal_mission_period_s,
)
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel, FerryPhysics
from hermes.scheduler.stages.s3_deadline import (
    FAST_PHASE_MISSED_WIDEN_S,
    FAST_PHASE_ON_TIME_SHRINK_S,
    LAW_ADDITIVE,
    LAW_MULTIPLICATIVE,
    LEGACY_MISSION_PERIOD_S,
    MIN_DEADLINE_FULFILMENT_S,
    DeadlineLaw,
    DeadlineLawError,
    DeadlineOverrideRefused,
    compute_deadline,
    effective_window,
    fold_cluster_amendment,
    fold_round_close_delta,
    initial_window_for_missions,
    time_scale_for_period,
    window_in_missions,
)
from hermes.types import (
    BeaconObservation,
    Bucket,
    ClusterAmendment,
    ContactWaypoint,
    DeviceID,
    DeviceRecord,
    DeviceSchedulerState,
    MissionOutcome,
    MissionPass,
    MissionSlice,
    MuleID,
    RoundCloseDelta,
    SpectrumSig,
)

D = DeviceID("d")
CLEAN, PARTIAL, TIMEOUT = MissionOutcome.CLEAN, MissionOutcome.PARTIAL, MissionOutcome.TIMEOUT
#: The keys ``DeadlineLaw.to_params()`` had at afa9526.
AFA9526_PARAMS = {"beta_on", "beta_partial", "beta_timeout", "phi_min", "phi_max",
                  "expire_overrides"}


def _delta(outcome, *, ts=1000.0, did=D, rnd=1, answered=False):
    return RoundCloseDelta(device_id=did, mule_id=MuleID("m"), mission_round=rnd,
                           outcome=outcome, utility=0.0, contact_ts=ts, answered=answered)


def _state(phi=60.0, did=D):
    return DeviceSchedulerState(device_id=did, deadline_fulfilment_s=phi)


# --------------------------------------------------------------------------- #
# The recorded unit is untouched
# --------------------------------------------------------------------------- #

def test_the_recorded_law_keeps_its_params_and_identity():
    law = DeadlineLaw()
    assert law.time_scale == 1.0 and law.is_recorded
    assert set(law.to_params()) == AFA9526_PARAMS
    assert set(DeadlineLaw(form=LAW_MULTIPLICATIVE).to_params()) == AFA9526_PARAMS
    assert DeadlineLaw(time_scale=1.0) == law
    assert DeadlineLaw.from_config(None, {"time_scale": 1}) == law
    assert (law.floor_s, law.on_time_shrink_s, law.missed_widen_s) == (
        MIN_DEADLINE_FULFILMENT_S, FAST_PHASE_ON_TIME_SHRINK_S, FAST_PHASE_MISSED_WIDEN_S)
    mult = DeadlineLaw(form=LAW_MULTIPLICATIVE)
    assert mult.phi_bounds == (mult.phi_min, mult.phi_max)


@pytest.mark.parametrize("law", [DeadlineLaw(), DeadlineLaw(time_scale=1.0),
                                 DeadlineLaw(form=LAW_ADDITIVE, time_scale=1)])
def test_scale_one_reproduces_the_recorded_fold_exactly(law):
    rng = random.Random(17)
    legacy, explicit = _state(), _state()
    for step in range(400):
        outcome = rng.choice((CLEAN, PARTIAL, TIMEOUT))
        d = _delta(outcome, ts=1000.0 + step, rnd=step, answered=rng.random() < 0.5)
        fold_round_close_delta(legacy, d)
        fold_round_close_delta(explicit, d, law=law)
        assert legacy == explicit
        now = 1000.0 + step + 0.25
        assert compute_deadline(legacy, now=now) == compute_deadline(explicit, now=now, law=law)
        assert effective_window(legacy) == effective_window(explicit, law=law)
    for val in (1.0, 4.999, 5.0, 77.5, 10_000.0):
        a, b = {D: _state()}, {D: _state()}
        amend = ClusterAmendment(cluster_round=1,
                                 registry_deltas={D: {"deadline_fulfilment_s": val}})
        fold_cluster_amendment(a, amend)
        fold_cluster_amendment(b, amend, law=law)
        assert a == b


def test_multiplicative_at_scale_one_is_the_multiplicative_law():
    rng = random.Random(5)
    base, scaled = DeadlineLaw(form=LAW_MULTIPLICATIVE), DeadlineLaw(
        form=LAW_MULTIPLICATIVE, time_scale=1.0)
    a, b = _state(), _state()
    for step in range(300):
        d = _delta(rng.choice((CLEAN, PARTIAL, TIMEOUT)), rnd=step)
        fold_round_close_delta(a, d, law=base)
        fold_round_close_delta(b, d, law=scaled)
        assert a == b


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan"), True, "2"])
def test_invalid_time_scales_are_refused(bad):
    with pytest.raises(DeadlineLawError):
        DeadlineLaw(time_scale=bad)


# --------------------------------------------------------------------------- #
# Another unit
# --------------------------------------------------------------------------- #

def test_scale_moves_every_time_constant_together():
    add = DeadlineLaw(time_scale=2.0)
    assert (add.floor_s, add.on_time_shrink_s, add.missed_widen_s) == (10.0, 10.0, 20.0)
    assert add.clamp(3.0) == 10.0 and add.clamp(12.0) == 12.0     # the scaled floor
    assert DeadlineLaw().clamp(3.0) == MIN_DEADLINE_FULFILMENT_S
    assert add.next_window(60.0, CLEAN) == 50.0
    assert add.next_window(12.0, CLEAN) == 10.0
    assert add.next_window(60.0, TIMEOUT) == 80.0
    assert effective_window(_state(3.0), law=add) == 10.0
    st = {D: _state()}
    fold_cluster_amendment(st, ClusterAmendment(cluster_round=1, registry_deltas={
        D: {"deadline_fulfilment_s": 1.0}}), law=add)
    assert st[D].deadline_fulfilment_s == 10.0
    mult = DeadlineLaw(form=LAW_MULTIPLICATIVE, time_scale=2.0)
    assert mult.phi_bounds == (10.0, 600.0)
    assert mult.clamp(1000.0) == 600.0 and mult.clamp(1.0) == 10.0
    assert mult.next_window(60.0, TIMEOUT) == 90.0          # β is a ratio: unscaled
    # The break-even rates are unit-free.
    assert add.break_even_on_time_rate() == DeadlineLaw().break_even_on_time_rate()


def test_a_scaled_trajectory_is_the_recorded_one_in_the_new_unit():
    """Φ under scale s, divided by s, follows the recorded law from Φ₀."""
    s = 21.9
    law = DeadlineLaw(time_scale=s)
    rng = random.Random(23)
    rec, scl = _state(60.0), _state(60.0 * s)
    for step in range(200):
        d = _delta(rng.choice((CLEAN, CLEAN, TIMEOUT)), rnd=step)
        fold_round_close_delta(rec, d)
        fold_round_close_delta(scl, d, law=law)
        assert scl.deadline_fulfilment_s / s == pytest.approx(rec.deadline_fulfilment_s, rel=1e-12)


def test_to_params_records_a_scale_other_than_one():
    law = DeadlineLaw(form=LAW_MULTIPLICATIVE, beta_on=0.7, time_scale=12.5)
    params = law.to_params()
    assert params["time_scale"] == 12.5
    assert DeadlineLaw.from_config(law.form, params) == law
    assert not DeadlineLaw(time_scale=2.0).is_recorded
    assert DeadlineLaw(time_scale=2).time_scale == 2.0 and isinstance(
        DeadlineLaw(time_scale=2).time_scale, float)


# --------------------------------------------------------------------------- #
# The scheduler's switches
# --------------------------------------------------------------------------- #

def _ingest(sch):
    """Three ways a device row is created: slice member, registry record, beacon."""
    sig = SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,))
    sch.ingest_slice(
        MissionSlice(mule_id=MuleID("m"), device_ids=(DeviceID("s"), DeviceID("r")),
                     issued_round=0, issued_at=0.0),
        registry_records=[DeviceRecord(device_id=DeviceID("r"),
                                       last_known_position=(1.0, 2.0, 0.0),
                                       spectrum_sig=sig, delivery_priority=2)],
    )
    sch.ingest_beacon(BeaconObservation(device_id=DeviceID("b"), observed_at=5.0))
    return sch.device_states


@pytest.mark.parametrize("kwargs", [
    {}, dict(deadline_time_scale=1.0), dict(initial_window_s=60.0),
    dict(deadline_time_scale=1.0, initial_window_s=None),
])
def test_the_scheduler_at_the_recorded_values_builds_the_same_rows(kwargs):
    legacy = _ingest(FLScheduler(now_fn=lambda: 1000.0))
    other = FLScheduler(now_fn=lambda: 1000.0, **kwargs)
    assert _ingest(other) == legacy
    assert other.deadline_law is None and other.deadline_time_scale == 1.0
    assert other.initial_window_s == 60.0


def test_a_scaled_scheduler_scales_the_law_and_phi0():
    sch = FLScheduler(now_fn=lambda: 1000.0, deadline_time_scale=2.0)
    assert sch.deadline_law == DeadlineLaw(time_scale=2.0) and sch.deadline_time_scale == 2.0
    states = _ingest(sch)
    assert {str(d): st.deadline_fulfilment_s for d, st in states.items()} == {
        "s": 120.0, "r": 120.0, "b": 120.0}
    st = states[DeviceID("s")]
    assert compute_deadline(st, now=1000.0, law=sch.deadline_law) == 1120.0
    sch.ingest_round_close_delta(_delta(TIMEOUT, did=DeviceID("s")))
    assert st.deadline_fulfilment_s == 140.0
    # The mule's merge cutoff reads the same scaled Φ (its floor included).
    st.deadline_fulfilment_s = 3.0
    assert effective_window(st, law=sch.deadline_law) == 10.0
    mult = FLScheduler(deadline_law=DeadlineLaw(form=LAW_MULTIPLICATIVE),
                       deadline_time_scale=2.0).deadline_law
    assert mult.form == LAW_MULTIPLICATIVE and mult.phi_bounds == (10.0, 600.0)


@pytest.mark.parametrize("scale", [1.0, 2.0, 21.9])
def test_an_explicit_phi0_is_in_the_recorded_unit_and_scales_with_the_law(scale):
    """Spec Q1: the time scale applies to Φ₀ like to every other constant of
    the law, so the recorded 60 s is the same Φ₀ whether it is spelled out or
    left at its default (the Freeze row's legacy value "60 s")."""
    default = FLScheduler(deadline_time_scale=scale)
    spelled = FLScheduler(deadline_time_scale=scale, initial_window_s=60.0)
    assert _ingest(spelled) == _ingest(default)
    for sch in (default, spelled):
        assert sch.initial_window_s == 60.0
        assert sch.effective_initial_window_s == (60.0 if scale == 1.0 else 60.0 * scale)
    other = FLScheduler(deadline_time_scale=scale, initial_window_s=90.0)
    assert other.initial_window_s == 90.0
    expected = 90.0 if scale == 1.0 else 90.0 * scale
    assert other.effective_initial_window_s == expected
    assert {st.deadline_fulfilment_s for st in _ingest(other).values()} == {expected}


def test_conflicting_or_invalid_scales_are_refused():
    with pytest.raises(FLSchedulerError, match="conflicts"):
        FLScheduler(deadline_law=DeadlineLaw(time_scale=3.0), deadline_time_scale=2.0)
    assert FLScheduler(deadline_law=DeadlineLaw(time_scale=2.0),
                       deadline_time_scale=2.0).deadline_time_scale == 2.0
    assert FLScheduler(deadline_law=DeadlineLaw(time_scale=3.0)).deadline_time_scale == 3.0
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(FLSchedulerError):
            FLScheduler(deadline_time_scale=bad)
        with pytest.raises(FLSchedulerError):
            FLScheduler(initial_window_s=bad)


def test_windows_in_missions():
    t_nom = 219.0                         # design §0: the median --realism mission
    scale = time_scale_for_period(t_nom)
    assert scale == t_nom / LEGACY_MISSION_PERIOD_S
    assert window_in_missions(1314.0, t_nom) == 6.0
    # Scaling by T_nom / 10 s makes the recorded 60 s window six missions long.
    sch = FLScheduler(deadline_time_scale=scale)
    assert window_in_missions(sch.effective_initial_window_s, t_nom) == pytest.approx(6.0)
    # Φ₀ in missions (critic A7): the value to hand the scheduler divides its
    # time scale back out, so the window it builds is exactly m missions.
    assert initial_window_for_missions(6, t_nom, time_scale=1.0) == 1314.0
    for missions in (3, 6, 12.5):
        phi0 = initial_window_for_missions(missions, t_nom, time_scale=scale)
        assert phi0 == pytest.approx(missions * LEGACY_MISSION_PERIOD_S)
        swept = FLScheduler(deadline_time_scale=scale, initial_window_s=phi0)
        assert window_in_missions(swept.effective_initial_window_s, t_nom) == pytest.approx(missions)
        row = _ingest(swept)[DeviceID("s")]
        assert row.deadline_fulfilment_s == pytest.approx(missions * t_nom)
    # The scale has no default: a window in clock seconds handed over
    # unscaled would be stretched twice.
    with pytest.raises(TypeError):
        initial_window_for_missions(6, t_nom)          # type: ignore[call-arg]
    for bad in ((0.0,), (-1.0,), (float("nan"),)):
        with pytest.raises(ValueError):
            time_scale_for_period(*bad)
        with pytest.raises(ValueError):
            initial_window_for_missions(1.0, *bad, time_scale=1.0)
        with pytest.raises(ValueError):
            initial_window_for_missions(1.0, t_nom, time_scale=bad[0])
        with pytest.raises(ValueError):
            window_in_missions(1.0, *bad)
    with pytest.raises(ValueError):
        initial_window_for_missions(0.0, t_nom, time_scale=1.0)


# --------------------------------------------------------------------------- #
# Cluster overrides on the simulated clock (critic B3)
# --------------------------------------------------------------------------- #

def _slice():
    return MissionSlice(mule_id=MuleID("m"), device_ids=(D,), issued_round=0, issued_at=0.0)


def test_the_simulated_clock_refuses_deadline_overrides_before_ingesting_anything():
    sch = FLScheduler(refuse_deadline_overrides=True)
    with pytest.raises(FLSchedulerError, match="refused"):
        sch.ingest_slice(_slice(), ClusterAmendment(cluster_round=1,
                                                    deadline_overrides={D: 123.0}))
    assert sch.current_slice is None and not sch.device_states
    # An amendment without overrides is fine.
    sch.ingest_slice(_slice(), ClusterAmendment(cluster_round=1, registry_deltas={
        D: {"deadline_fulfilment_s": 42.0}}))
    assert sch.device_states[D].deadline_fulfilment_s == 42.0
    # The legacy scheduler folds overrides as it always did.
    legacy = FLScheduler()
    legacy.ingest_slice(_slice(), ClusterAmendment(cluster_round=1,
                                                   deadline_overrides={D: 123.0}))
    assert legacy.device_states[D].deadline_override_ts == 123.0


def test_the_fold_itself_can_refuse_overrides():
    states = {D: _state()}
    with pytest.raises(DeadlineOverrideRefused):
        fold_cluster_amendment(states, ClusterAmendment(
            cluster_round=1, deadline_overrides={D: 5.0},
            registry_deltas={D: {"deadline_fulfilment_s": 9.0}}), refuse_overrides=True)
    assert states[D] == _state()                      # nothing folded


# --------------------------------------------------------------------------- #
# The spectrum fold (design §4.7) and the waypoint annotations (design §4.5)
# --------------------------------------------------------------------------- #

def test_the_spectrum_fold():
    states = {D: _state(), DeviceID("e"): _state(did=DeviceID("e"))}
    assert states[D].spectrum_snr_db is None                         # never reported
    fold_cluster_amendment(states, ClusterAmendment(cluster_round=1, registry_deltas={
        D: {"spectrum_sig": SimpleNamespace(contact_class_snr_db={"wide": 12.5, "narrow": 20})},
        DeviceID("e"): {"spectrum_sig": {"medium": 3.25, "bogus": "x", "flag": True}},
        DeviceID("unknown"): {"spectrum_sig": {"wide": 1.0}},
    }))
    assert states[D].spectrum_snr_db == {"wide": 12.5, "narrow": 20.0}
    assert states[DeviceID("e")].spectrum_snr_db == {"medium": 3.25}
    # A SpectrumSig with no contact-class reading, or an empty one, observes
    # nothing and changes nothing.
    legacy_sig = SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,))
    fold_cluster_amendment(states, ClusterAmendment(cluster_round=2, registry_deltas={
        D: {"spectrum_sig": legacy_sig, "delivery_priority": 1},
        DeviceID("e"): {"spectrum_sig": {}}}))
    assert states[D].spectrum_snr_db == {"wide": 12.5, "narrow": 20.0}
    assert states[DeviceID("e")].spectrum_snr_db == {"medium": 3.25}


def test_the_spectrum_fold_keeps_the_latest_reading_per_class():
    """A report updates the classes it carries; the others keep their last
    value. Non-finite readings are not observations."""
    states = {D: _state()}

    def fold(sig):
        fold_cluster_amendment(states, ClusterAmendment(
            cluster_round=1, registry_deltas={D: {"spectrum_sig": sig}}))

    fold({"wide": 12.5, "narrow": 20.0})
    fold(SimpleNamespace(contact_class_snr_db={"medium": 3.0, "wide": 11.0}))
    assert states[D].spectrum_snr_db == {"wide": 11.0, "narrow": 20.0, "medium": 3.0}
    fold({"wide": float("nan"), "medium": float("inf"), "narrow": 1.5})
    assert states[D].spectrum_snr_db == {"wide": 11.0, "narrow": 1.5, "medium": 3.0}
    before = dict(states[D].spectrum_snr_db)
    fold({"wide": float("-inf")})                    # nothing finite: nothing observed
    assert states[D].spectrum_snr_db == before


def test_a_boolean_time_scale_is_refused():
    with pytest.raises(FLSchedulerError):
        FLScheduler(deadline_time_scale=True)


# --------------------------------------------------------------------------- #
# T_nom, the nominal mission period (spec Q1)
# --------------------------------------------------------------------------- #

DOCK = (0.0, 0.0, 0.0)


def _member_dwell(d, pass_kind, offset):
    """2 s + d per member in Pass 1, 1 s + d in Pass 2."""
    return (2.0 if pass_kind is MissionPass.COLLECT else 1.0) + d


def _ferry(dwell=_member_dwell, upload=3.0, device_states=None):
    return FeasibilityModel(ferry=FerryPhysics(
        dock=DOCK, member_dwell_s=dwell, upload_s=lambda: upload,
        p_move_w=143.6, p_hover_w=168.5, device_states=device_states))


#: a and b (10 m apart) share the stop (30, 40, 0), 50 m out, 5 m from each;
#: c is alone at (-36, -48, 0), 60 m out and 110 m from the first stop.
LAYOUT = {DeviceID("a"): (27.0, 36.0, 0.0), DeviceID("b"): (33.0, 44.0, 0.0),
          DeviceID("c"): (-36.0, -48.0, 0.0)}


def test_the_nominal_period_by_hand():
    """At 5 m/s: Pass 1 = 10 s out + 14 s dwell (7 + 7) + 22 s on + 2 s
    dwell + 12 s home + 3 s upload = 63 s; Pass 2 = 10 + 12 + 22 + 1 + 12 =
    57 s; plus a 30 s turnaround: 150 s."""
    assert nominal_mission_period_s(LAYOUT, rf_range_m=10.0, feasibility_model=_ferry(),
                                     turnaround_s=30.0) == 150.0
    # Each layout's members are looked up in the layout itself, whatever map
    # the model was bound to.
    elsewhere = {d: (1e4, 1e4, 0.0) for d in LAYOUT}
    assert nominal_mission_period_s(
        LAYOUT, rf_range_m=10.0, feasibility_model=_ferry(device_states=elsewhere),
        turnaround_s=30.0) == 150.0
    # A range that separates a and b: three stops (45, 55 and 60 m out), no
    # member off its stop; 45 + 10 + 115 + 60 m is 46 s of flight per pass.
    # Pass 1 = 46 + 3 * 2 + 3 = 55 s; Pass 2 = 46 + 3 * 1 = 49 s.
    assert nominal_mission_period_s(LAYOUT, rf_range_m=5.0, feasibility_model=_ferry(),
                                    turnaround_s=30.0) == 55.0 + 30.0 + 49.0
    # No devices: the turnaround alone.
    assert nominal_mission_period_s({}, rf_range_m=10.0, feasibility_model=_ferry(),
                                    turnaround_s=30.0) == 30.0


def test_t_nom_is_the_median_over_the_layouts():
    shifted = {d: (x * 2.0, y * 2.0, z) for d, (x, y, z) in LAYOUT.items()}
    far = {d: (x * 3.0, y * 3.0, z) for d, (x, y, z) in LAYOUT.items()}
    kw = dict(rf_range_m=10.0, feasibility_model=_ferry(), turnaround_s=30.0)
    periods = [nominal_mission_period_s(lay, **kw) for lay in (LAYOUT, shifted, far)]
    assert median_nominal_mission_period_s([far, LAYOUT, shifted], **kw) == sorted(periods)[1]
    with pytest.raises(FLSchedulerError):
        median_nominal_mission_period_s([], **kw)


def test_the_nominal_period_needs_the_ferry_physics_and_sane_inputs():
    with pytest.raises(FLSchedulerError, match="ferry"):
        nominal_mission_period_s(LAYOUT, rf_range_m=10.0, feasibility_model=FeasibilityModel(),
                                 turnaround_s=30.0)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(FLSchedulerError):
            nominal_mission_period_s(LAYOUT, rf_range_m=10.0, feasibility_model=_ferry(),
                                     turnaround_s=bad)
    with pytest.raises(FLSchedulerError, match=r"\(x, y, z\)"):
        nominal_mission_period_s({DeviceID("a"): (1.0, 2.0)}, rf_range_m=10.0,
                                 feasibility_model=_ferry(), turnaround_s=30.0)


def test_the_nominal_period_reproduces_the_design_probe():
    """Design §0 finding 2 (``probe_ferry_period.py``): N = 6, 60 m, 5 m/s,
    0.03 s per device, a 30 s turnaround and both return legs, 20 seeds.
    The median period is 36 s at the default spread and 219 s under
    ``--realism`` (field radius 100 m). Checked per layout against the
    probe's own walk, which prices each stop and the return independently."""
    from experiments.exp4.topology_builder import build_exp4_topology

    model = _ferry(dwell=lambda d, p, o: 0.03, upload=0.0)

    def probe_walk(stops):
        t, pose = 0.0, DOCK
        for w in stops:
            t += math.dist(pose, w.position) / 5.0
            for _ in w.devices:
                t += 0.03
            pose = w.position
        return t + math.dist(pose, DOCK) / 5.0

    for field, expected in ((None, 36), (100.0, 219)):
        layouts = []
        for seed in range(20):
            topo = build_exp4_topology(n_devices=6, rf_range_m=60.0, n_missions=4, seed=seed,
                                       field_radius_m=field)
            layout = {DeviceID(x.device_id): tuple(x.position) for x in topo.devices}
            layouts.append(layout)
            planner = FLScheduler(now_fn=lambda: 1e6)
            planner.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=tuple(layout),
                                              issued_round=1, issued_at=1e6))
            for did, pos in layout.items():
                planner.device_states[did].last_known_position = pos
            probe = (probe_walk(planner.build_contact_queue(rf_range_m=60.0, mule_pose=DOCK))
                     + 30.0
                     + probe_walk(planner.build_pass_2_queue(rf_range_m=60.0, mule_pose=DOCK)))
            mine = nominal_mission_period_s(layout, rf_range_m=60.0, feasibility_model=model,
                                            turnaround_s=30.0)
            assert mine == pytest.approx(probe, rel=1e-12, abs=1e-9), (field, seed)
        t_nom = median_nominal_mission_period_s(layouts, rf_range_m=60.0,
                                                feasibility_model=model, turnaround_s=30.0)
        assert round(t_nom) == expected, (field, t_nom)


def test_waypoint_annotations_leave_equality_and_hashing_unchanged():
    wp = ContactWaypoint(position=(1.0, 2.0, 0.0), devices=(D, DeviceID("e")),
                         bucket=Bucket.NEW, deadline_ts=10.0)
    annotated = dataclasses.replace(wp, band="wide", range_m=60.0, pred_snr_db=(12.5, 3.0))
    assert (wp.band, wp.range_m, wp.pred_snr_db) == (None, None, None)
    assert annotated.band == "wide" and annotated.pred_snr_db == (12.5, 3.0)
    assert annotated == wp and hash(annotated) == hash(wp)
    assert {wp: "plan"}[annotated] == "plan" and annotated in {wp}
    other_band = dataclasses.replace(wp, band="narrow", range_m=232.2)
    assert other_band == annotated and hash(other_band) == hash(annotated)
    # Identity fields still count.
    assert dataclasses.replace(wp, deadline_ts=11.0) != wp
    with pytest.raises(dataclasses.FrozenInstanceError):
        annotated.band = "medium"
    # The positional form every existing caller uses still works.
    assert ContactWaypoint((1.0, 2.0, 0.0), (D, DeviceID("e")), Bucket.NEW, 10.0) == wp
