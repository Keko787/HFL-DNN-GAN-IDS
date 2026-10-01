"""FeRRy Phase 4 (unit U2): the plan score V and its coverage weights.

Pins V = −[c₁(Δ/T)² + c₂U + c₃L] − c₄E/(P_hover·T) as the Phase 4 spec (other
choices 7) and the user's decisions 2 and 3 of 2026-09-30 fix it. Δ is the
whole mission on the chosen band (Pass 1, the turnaround, Pass 2) against
T = T_nom, and the empty plan flies neither pass but still pays the
turnaround; E covers both passes; U and L weigh each device by age × (1 +
miss streak) (arm F), by age alone (F-prio) or uniformly, and never by 0; L is
the mean-SNR outage with σ_eff = √(σ_sh² + σ_I² + A²/2), and the mule
runtime's ``PlanClass.outage``, a second copy of that formula (U6), agrees
with this module's. The plan's own test, "V falls as coverage falls" (L842),
is pinned with Δ and E held, c₂ > 0 and c₃ ≤ c₂, as the design restates it
(§0.4 item 16, §4); at c₂ = c₃ = 0 (F-cov) V does not see coverage. Also
pinned: V of the empty plan, uniform weights reproducing 1 − served/N, the
constants and the pilot's sweep, F-cov's cap-only service, the dwell_in_delta
switch (arm F-dwell), the predicted mission, which stays the whole mission
under F-dwell (design D-M), the hand-off to the commit, order-independent
sums, the refusals, and the layering: numpy-free, nothing from hermes.l1 or
experiments, and never loaded by the legacy pipeline (Freeze Rule 1).

The rank (the orchestrator's resolution R11): ``coverage_rank`` is a score
setting, ``lexicographic`` by default, refused when misspelt, carried by the
options and never read by V; the served weight share is Σ_served w / Σ_demand
w (1 for an empty demand); the rank applied is the setting except under F-cov
(κ = 0), which ranks by V alone either way; and the plan key puts the share
between the cap key and V under ``lexicographic`` and is U0's
``Candidate.key`` under ``weighted``, so on U7's probe numbers the share
prefers serving a device that V alone would leave for the empty plan.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import pytest

from hermes.l1.channel_model import CONTACT_REGIMES, SHADOW_SIGMA_DB, ContactChannel
from hermes.l1.contact_link import CLASSES_WITH_10MHZ, ContactLink
from hermes.scheduler.plan import (
    BAND_POLICY_SEARCH,
    COVERAGE_WEIGHTS,
    COVERAGE_WEIGHTS_AGE,
    COVERAGE_WEIGHTS_UNIFORM,
    PLAN_SCORE_KEYS,
    AgeCapSpec,
    Candidate,
    CapState,
    MemberFold,
    PlanClass,
    PlanCommit,
    PlanOptions,
    PlanScoreParams,
    ScoreTerms,
)
from hermes.scheduler.plan import types as plan_types
from hermes.scheduler.plan.plan_score import (
    COVERAGE_RANK_LEXICOGRAPHIC,
    COVERAGE_RANK_WEIGHTED,
    COVERAGE_RANKS,
    MISSION_SCORE_KEY,
    PILOT_C_ENERGIES,
    PILOT_KAPPAS,
    applied_rank,
    coverage_weight,
    demand_weights,
    mean_snr_outage,
    outage_by_distance,
    plan_key,
    predicted_mission_s,
    score,
    served_share,
    sigma_eff_db,
)
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel, FerryPhysics, FlightState
from hermes.types import Bucket, ContactWaypoint, DeviceID
from hermes.types.scheduler import DeviceSchedulerState

REPO = Path(__file__).resolve().parents[2]
A, B, C, D = (DeviceID(x) for x in ("a", "b", "c", "d"))
T_NOM = 220.0        # T, the cell's nominal mission period (s)
TURN = 30.0          # the dock turnaround (FerrySpec's turnaround_s default)
P_HOVER = 168.5      # hover power, Zeng-Xu-Zhang 2019 (build plan L925)
PHI = NormalDist()

# A candidate's primitives, held fixed wherever a test varies only the served set.
HELD = dict(pass_1_s=70.0, pass_1_dwell_s=12.0, pass_1_energy_j=8000.0, turnaround_s=TURN,
            pass_2_s=60.0, pass_2_dwell_s=9.0, pass_2_energy_j=9000.0,
            t_ref_s=T_NOM, p_hover_w=P_HOVER)
# The empty plan at the dock: no Pass 1; Pass 2's numbers are the class's, not flown.
EMPTY = dict(HELD, pass_1_s=0.0, pass_1_dwell_s=0.0, pass_1_energy_j=0.0)


def _score(params=None, *, weights, served, **overrides):
    kwargs = dict(HELD)
    kwargs.update(overrides)
    return score(params or PlanScoreParams(), weights=weights, served_outage=served, **kwargs)


def _state(did, streak=0):
    return DeviceSchedulerState(device_id=did, miss_streak=streak)


def _devices(n):
    return [DeviceID(f"d{i}") for i in range(n)]


# --------------------------------------------------------------------------- #
# Layering (Freeze Rule 1; critic B13; the U3b import rule)
# --------------------------------------------------------------------------- #

_STDLIB = {"__future__", "collections", "math", "numbers", "typing"}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield "." * node.level + (node.module or "")


def test_the_score_imports_only_the_standard_library_and_the_plan_types():
    """Numpy-free, and nothing from hermes.l1, the mule, experiments, the
    policies or the scheduler: physics reaches V as floats and callables."""
    names = list(_imports(REPO / "hermes/scheduler/plan/plan_score.py"))
    assert names
    for name in names:
        assert (name.split(".")[0] in _STDLIB or name.startswith("hermes.types.")
                or name == ".types"), name


def test_the_legacy_pipeline_never_loads_the_score():
    """Legacy never reaches the plan path, so importing its entry points (and
    the plan package, which re-exports types only) must not load plan_score."""
    code = (
        "import sys\n"
        "import hermes.scheduler.plan, hermes.scheduler.fl_scheduler\n"
        "import hermes.scheduler.stages.s3b_feasibility, hermes.mule.mule_main, hermes.mule.ferry\n"
        "import hermes.processes.mule, hermes.processes.config\n"
        "import experiments.exp4.driver, experiments.exp4.runner_main\n"
        "import experiments.exp4.topology_builder, experiments.exp4.events_consumer\n"
        "import experiments.analysis.traces_scorer\n"
        "print('hermes.scheduler.plan.plan_score' in sys.modules)\n"
    )
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


# --------------------------------------------------------------------------- #
# The link term: mean-SNR outage (design link option (ii))
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("regime", sorted(CONTACT_REGIMES))
def test_sigma_eff_adds_the_variances_of_shadowing_noise_and_wave(regime):
    amp, noise = CONTACT_REGIMES[regime]
    expected = math.sqrt(SHADOW_SIGMA_DB ** 2 + noise ** 2 + amp ** 2 / 2.0)
    assert sigma_eff_db(SHADOW_SIGMA_DB, noise, amp) == pytest.approx(expected, rel=1e-15)
    assert sigma_eff_db(4.0, 0.0, 0.0) == 4.0        # the shadowing alone
    assert sigma_eff_db(0.0, 0.0, 2.0) == pytest.approx(math.sqrt(2.0))   # a sine: A/√2


def test_the_outage_is_the_normal_cdf_of_the_gap_to_the_floor():
    floor, sigma = -6.7, 4.0
    assert mean_snr_outage(floor, snr_floor_db=floor, sigma_db=sigma) == 0.5
    for gap in (-3.0, -1.0, -0.25, 0.0, 0.5, 1.0, 2.0, 3.0):
        got = mean_snr_outage(floor + gap * sigma, snr_floor_db=floor, sigma_db=sigma)
        assert got == pytest.approx(PHI.cdf(-gap), rel=1e-12)
    # erfc keeps the tail: Φ(−10), which 0.5·(1 + erf) would round to 0.
    far = mean_snr_outage(floor + 10.0 * sigma, snr_floor_db=floor, sigma_db=sigma)
    assert far > 0.0
    assert far == pytest.approx(7.619853024160526e-24, rel=1e-12, abs=0.0)
    assert mean_snr_outage(floor - 40.0 * sigma, snr_floor_db=floor, sigma_db=sigma) == 1.0


def test_without_spread_the_outage_is_a_step_with_an_inclusive_floor():
    assert mean_snr_outage(-6.8, snr_floor_db=-6.7, sigma_db=0.0) == 1.0
    assert mean_snr_outage(-6.7, snr_floor_db=-6.7, sigma_db=0.0) == 0.0   # at the floor: reached
    assert mean_snr_outage(3.0, snr_floor_db=-6.7, sigma_db=0.0) == 0.0


def test_at_each_class_edge_the_shadowing_alone_gives_the_links_edge_outage():
    """R(b) is where the mean SNR sits Φ⁻¹(q)·σ_sh above the floor
    (``ContactLink``, q = 0.9): with σ_eff = σ_sh the outage there is 1 − q."""
    link = ContactLink(anchor_planar_m=60.0)
    for band in link.names:
        outage = outage_by_distance(lambda d, b=band: link.mean_snr_db(b, d),
                                    snr_floor_db=link.snr_floor_db, sigma_db=link.shadow_sigma_db)
        assert outage(link.range_planar_m(band)) == pytest.approx(1.0 - link.margin_quantile,
                                                                  rel=1e-9)


def test_beyond_the_gates_range_a_member_is_never_solicited():
    """With ``range_m`` = R_planar(b), a member farther away is certain to be
    missed (the contact gate's range, inclusive); without it, the formula."""
    link = ContactLink(anchor_planar_m=60.0)
    for band in link.names:
        reach = link.range_planar_m(band)
        kwargs = dict(snr_floor_db=link.snr_floor_db, sigma_db=link.shadow_sigma_db)
        gated = outage_by_distance(lambda d, b=band: link.mean_snr_db(b, d), range_m=reach,
                                   **kwargs)
        free = outage_by_distance(lambda d, b=band: link.mean_snr_db(b, d), **kwargs)
        assert gated(reach) == free(reach) == pytest.approx(0.1, rel=1e-9)
        assert gated(math.nextafter(reach, math.inf)) == 1.0
        assert gated(reach + 50.0) == 1.0 and free(reach + 50.0) < 1.0
        assert all(gated(float(d)) == free(float(d)) for d in range(0, int(reach) + 1))


@pytest.mark.parametrize("regime", sorted(CONTACT_REGIMES))
def test_the_interference_raises_the_edge_outage_and_it_grows_with_distance(regime):
    link = ContactLink(anchor_planar_m=60.0)
    amp, noise = CONTACT_REGIMES[regime]
    sigma = sigma_eff_db(link.shadow_sigma_db, noise, amp)
    margin = PHI.inv_cdf(link.margin_quantile) * link.shadow_sigma_db
    for band in link.names:
        outage = outage_by_distance(lambda d, b=band: link.mean_snr_db(b, d),
                                    snr_floor_db=link.snr_floor_db, sigma_db=sigma)
        edge = outage(link.range_planar_m(band))
        assert edge == pytest.approx(PHI.cdf(-margin / sigma), rel=1e-9)
        assert edge > 1.0 - link.margin_quantile
        values = [outage(float(d)) for d in range(0, 401, 5)]
        assert all(0.0 <= p <= 1.0 for p in values)
        assert values == sorted(values)                  # farther never reaches better


def _mule_runtime(channel, classes):
    """The mule's runtime (U6) on a link of ``classes`` (None: the default
    three), in a contact regime or, for ``"noise-free"``, on critic B4's
    channel: it draws no noise while the link keeps its σ in the margin, and a
    0.2 edge quantile puts both sides of the step inside each class's range.
    The mule is imported here, by the one test that needs it."""
    from hermes.mule.ferry import FerryRuntime, FerrySpec

    rf = 60.0
    extra = {} if classes is None else {"band_classes": classes}
    if channel != "noise-free":
        spec = FerrySpec.from_config(rf_range_m=rf, seed=11, contact_band="wide",
                                     contact_regime=channel, **extra)
        return FerryRuntime(spec, None, rf_range_m=rf)
    link = ContactLink(anchor_planar_m=rf, margin_quantile=0.2,
                       **({} if classes is None else {"classes": classes}))
    chan = ContactChannel(link.mean_snr_db, salt=1, bands=link.names, interference_amp_db=0.0,
                          interference_sigma_db=0.0, shadow_sigma_db=0.0)
    spec = FerrySpec(band="wide", link=link, contact_channel=chan)
    return FerryRuntime(spec, None, rf_range_m=rf)


@pytest.mark.parametrize("classes", [None, CLASSES_WITH_10MHZ])
@pytest.mark.parametrize("channel", sorted(CONTACT_REGIMES) + ["noise-free"])
def test_the_mule_runtimes_plan_class_outage_is_this_formula(channel, classes):
    """The ``PlanClass.outage`` the planner calls is the runtime's own copy of
    the link term (``FerryRuntime.outage_probability``, through NormalDist);
    ``outage_by_distance`` bound to the same link and channel must give the
    same outage for every class, from the dock out past every class's range,
    across the inclusive gate and on the step. So a change to either copy
    fails here. They differ in the last bits only: erfc against erf, which
    rounds Φ to 0 below about −8.3σ."""
    rt = _mule_runtime(channel, classes)
    link, chan = rt.spec.link, rt.spec.contact_channel
    sigma = sigma_eff_db(chan.shadow_sigma_db, chan.interference_sigma_db,
                         chan.interference_amp_db)
    planned = rt.plan_classes()
    assert [c.name for c in planned] == list(link.names)
    for c in planned:
        ours = outage_by_distance(lambda d, b=c.name: link.mean_snr_db(b, d),
                                  snr_floor_db=link.snr_floor_db, sigma_db=sigma,
                                  range_m=link.range_planar_m(c.name))
        edge = c.radius_m
        beyond = math.nextafter(edge, math.inf)
        for d in [k * 0.5 for k in range(1001)] + [edge, beyond]:
            assert c.outage(d) == pytest.approx(ours(d), rel=0.0, abs=1e-15), (c.name, d)
        assert c.outage(beyond) == ours(beyond) == 1.0


@pytest.mark.parametrize("call, error", [
    (lambda: sigma_eff_db(-1.0, 0.0, 0.0), ValueError),
    (lambda: sigma_eff_db(4.0, math.nan, 0.0), ValueError),
    (lambda: sigma_eff_db(4.0, 0.0, True), TypeError),
    (lambda: mean_snr_outage(math.inf, snr_floor_db=-6.7, sigma_db=4.0), ValueError),
    (lambda: mean_snr_outage(1.0, snr_floor_db=None, sigma_db=4.0), TypeError),
    (lambda: mean_snr_outage(1.0, snr_floor_db=-6.7, sigma_db=-0.1), ValueError),
    (lambda: outage_by_distance(3.0, snr_floor_db=-6.7, sigma_db=4.0), TypeError),
    (lambda: outage_by_distance(lambda d: 0.0, snr_floor_db=-6.7, sigma_db=-4.0), ValueError),
    (lambda: outage_by_distance(lambda d: 0.0, snr_floor_db=-6.7, sigma_db=4.0)(-1.0), ValueError),
    (lambda: outage_by_distance(lambda d: math.nan, snr_floor_db=-6.7, sigma_db=4.0)(1.0),
     ValueError),
    (lambda: outage_by_distance(lambda d: 0.0, snr_floor_db=-6.7, sigma_db=4.0, range_m=0.0),
     ValueError),
    (lambda: outage_by_distance(lambda d: 0.0, snr_floor_db=-6.7, sigma_db=4.0, range_m=-5.0),
     ValueError),
    (lambda: outage_by_distance(lambda d: 0.0, snr_floor_db=-6.7, sigma_db=4.0, range_m=True),
     TypeError),
])
def test_the_outage_refuses_bad_values(call, error):
    with pytest.raises(error):
        call()


# --------------------------------------------------------------------------- #
# Coverage weights (decision 3)
# --------------------------------------------------------------------------- #

def test_f_weighs_age_times_one_plus_the_miss_streak():
    assert coverage_weight(3, 2, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE) == 9.0
    assert coverage_weight(5, 0, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE) == 5.0


def test_f_prio_weighs_age_alone():
    """miss_priority off: the streak is not read, which is the one-step
    Network-AoU objective (critic A5)."""
    assert coverage_weight(3, 2, miss_priority=False, mode=COVERAGE_WEIGHTS_AGE) == 3.0
    assert coverage_weight(3, 7, miss_priority=False, mode=COVERAGE_WEIGHTS_AGE) == 3.0


def test_uniform_weighs_one_or_one_plus_the_streak_and_reads_no_age():
    assert coverage_weight(None, 4, miss_priority=False, mode=COVERAGE_WEIGHTS_UNIFORM) == 1.0
    assert coverage_weight(None, 4, miss_priority=True, mode=COVERAGE_WEIGHTS_UNIFORM) == 5.0
    assert coverage_weight(9, 0, miss_priority=True, mode=COVERAGE_WEIGHTS_UNIFORM) == 1.0


def test_f_weighs_about_age_squared_when_every_left_out_device_is_a_miss():
    """Under a plan, each device left out is widened as a miss and a clean
    contact clears the streak, so m = a − 1 and w = a·(1 + m) = a² (critic A5):
    F's objective is declared quadratic in age."""
    for age in range(1, 9):
        w = coverage_weight(age, age - 1, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE)
        assert w == age * age


def test_every_demanded_device_weighs_at_least_one():
    """U1's age is m − (last_merged_round or 0), unclamped; the mule plans from
    mission 1 and records merges after Pass 1, so ages are >= 1 on its path.
    Age 0 (a merge in the mission being planned, or round 0 off that path)
    weighs as the freshest device: the weight floors, the age does not."""
    assert coverage_weight(0, 0, miss_priority=False, mode=COVERAGE_WEIGHTS_AGE) == 1.0
    assert coverage_weight(0, 3, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE) == 4.0
    assert coverage_weight(1, 3, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE) == 4.0
    for age, streak, prio, mode in itertools.product(
            range(0, 6), range(0, 6), (False, True), COVERAGE_WEIGHTS):
        assert coverage_weight(age, streak, miss_priority=prio, mode=mode) >= 1.0


def test_demand_weights_read_the_device_states_in_demand_order():
    states = {A: _state(A, 0), B: _state(B, 2), C: _state(C, 1), D: _state(D, 5)}
    ages = {A: 1, B: 3, C: 2, D: 0, DeviceID("x"): 9}          # extra ages are not read
    demand = (C, A, B)
    f = demand_weights(demand, states, ages=ages, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE)
    assert list(f) == [C, A, B] and f == {C: 4.0, A: 1.0, B: 9.0}
    prio = demand_weights(demand, states, ages=ages, miss_priority=False,
                          mode=COVERAGE_WEIGHTS_AGE)
    assert prio == {C: 2.0, A: 1.0, B: 3.0}
    uniform = demand_weights(demand, states, ages=None, miss_priority=True,
                             mode=COVERAGE_WEIGHTS_UNIFORM)
    assert uniform == {C: 2.0, A: 1.0, B: 3.0}
    for partial in ({}, {A: 1}):                  # uniform never looks an age up
        assert demand_weights(demand, states, ages=partial, miss_priority=True,
                              mode=COVERAGE_WEIGHTS_UNIFORM) == uniform
    assert demand_weights((), states, ages=None, miss_priority=True,
                          mode=COVERAGE_WEIGHTS_AGE) == {}
    assert demand_weights([D], states, ages=ages, miss_priority=True,
                          mode=COVERAGE_WEIGHTS_AGE) == {D: 6.0}   # age 0 floors at 1


def test_demand_weights_take_the_cap_states_ages_with_the_cap_on_or_off():
    """U5 hands over U1's ``CapState.ages`` (read-only); the ages are kept with
    the cap off too, because the age weights read them."""
    states = {A: _state(A, 1), B: _state(B, 0)}
    for spec in (AgeCapSpec(), AgeCapSpec(s_missions=2)):
        cap = CapState(spec=spec, ages={A: 2, B: 0})
        assert demand_weights((A, B), states, ages=cap.ages, miss_priority=True,
                              mode=COVERAGE_WEIGHTS_AGE) == {A: 4.0, B: 1.0}


@pytest.mark.parametrize("call, error", [
    (lambda: coverage_weight(1, 0, miss_priority=True, mode="age2"), ValueError),
    (lambda: coverage_weight(None, 0, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE), ValueError),
    (lambda: coverage_weight(-1, 0, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE), ValueError),
    (lambda: coverage_weight(2.0, 0, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE), TypeError),
    (lambda: coverage_weight(True, 0, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE), TypeError),
    (lambda: coverage_weight(1, -1, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE), ValueError),
    (lambda: coverage_weight(1, None, miss_priority=True, mode=COVERAGE_WEIGHTS_AGE), TypeError),
    (lambda: coverage_weight(1, 0, miss_priority=1, mode=COVERAGE_WEIGHTS_AGE), TypeError),
])
def test_a_coverage_weight_refuses_bad_values(call, error):
    with pytest.raises(error):
        call()


_STATES = {A: _state(A), B: _state(B)}


@pytest.mark.parametrize("kwargs, error", [
    (dict(demand="ab"), TypeError),                                    # one string, not ids
    (dict(demand=(A, A)), ValueError),                                 # listed twice
    (dict(demand=(A, 3)), TypeError),
    (dict(demand=(A, DeviceID("zz")), ages={A: 1, DeviceID("zz"): 2}), ValueError),  # no state
    (dict(device_states=[_state(A)]), TypeError),
    (dict(device_states={A: object(), B: _state(B)}), TypeError),      # no miss streak
    (dict(ages=None), ValueError),                                     # age mode needs ages
    (dict(ages={A: 1}), ValueError),                                   # b has none
    (dict(ages=[1, 2]), TypeError),
    (dict(ages={A: 1, B: -2}), ValueError),
    (dict(mode="Age"), ValueError),
    (dict(miss_priority="yes"), TypeError),
])
def test_demand_weights_refuse_bad_values(kwargs, error):
    values = dict(demand=(A, B), device_states=_STATES, ages={A: 1, B: 2}, miss_priority=True,
                  mode=COVERAGE_WEIGHTS_AGE)
    values.update(kwargs)
    with pytest.raises(error):
        demand_weights(values.pop("demand"), values.pop("device_states"), **values)


# --------------------------------------------------------------------------- #
# The score
# --------------------------------------------------------------------------- #

def test_a_hand_worked_score():
    """Four devices weighted 1-4, b and c served with outage 0.1 and 0.5."""
    weights = {A: 1.0, B: 2.0, C: 3.0, D: 4.0}
    terms = _score(weights=weights, served={B: 0.1, C: 0.5}, pass_1_s=60.0, pass_1_dwell_s=10.0,
                   pass_1_energy_j=9000.0, pass_2_s=90.0, pass_2_dwell_s=12.0,
                   pass_2_energy_j=14000.0, t_ref_s=200.0)
    assert isinstance(terms, ScoreTerms) and tuple(terms.as_dict()) == PLAN_SCORE_KEYS
    assert terms.delta_s == 180.0                              # 60 + 30 + 90
    assert terms.time == pytest.approx(0.81, rel=1e-15)        # (180 / 200)²
    assert terms.served_weight == 5.0 and terms.demand_weight == 10.0
    assert terms.coverage == 0.5
    assert terms.link == pytest.approx(0.17, rel=1e-15)        # (2·0.1 + 3·0.5) / 10
    assert terms.energy_j == 23000.0
    assert terms.energy == pytest.approx(23000.0 / (P_HOVER * 200.0), rel=1e-15)
    # (c1, c2, c3, c4) = (1, 4, 4, 0.1).
    expected = -(0.81 + 4 * 0.5 + 4 * 0.17) - 0.1 * 23000.0 / (P_HOVER * 200.0)
    assert terms.v == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("seed", range(20))
def test_v_is_its_four_terms_weighted_by_the_resolved_constants(seed):
    rng = random.Random(seed)
    devices = _devices(rng.randint(1, 7))
    weights = {d: float(rng.randint(1, 30)) for d in devices}
    served = {d: rng.random() for d in rng.sample(devices, rng.randint(0, len(devices)))}
    params = PlanScoreParams(c_time=rng.choice((0.5, 1.0, 2.0)),
                             c_cov_per_device=rng.choice(PILOT_KAPPAS),
                             c_link=rng.choice((None, 0.0, 1.5)),
                             c_energy=rng.choice(PILOT_C_ENERGIES),
                             dwell_in_delta=rng.random() < 0.5)
    terms = _score(params, weights=weights, served=served)
    c1, c2, c3, c4 = params.constants(len(devices))
    assert terms.v == pytest.approx(
        -(c1 * terms.time + c2 * terms.coverage + c3 * terms.link) - c4 * terms.energy,
        rel=1e-14, abs=1e-15)
    assert terms.time == pytest.approx((terms.delta_s / T_NOM) ** 2, rel=1e-15)
    assert terms.energy == pytest.approx(terms.energy_j / (P_HOVER * T_NOM), rel=1e-15)
    assert terms.coverage == 1.0 - terms.served_weight / terms.demand_weight
    assert 0.0 <= terms.link <= terms.served_weight / terms.demand_weight + 1e-15


def test_delta_is_the_whole_mission_on_the_band_and_e_covers_both_passes():
    """Decision 2 (b): Pass 1 + turnaround + Pass 2, both at b̄, against T_nom."""
    terms = _score(weights={A: 1.0, B: 1.0}, served={A: 0.0}, pass_1_s=50.0, pass_1_energy_j=7000.0,
                   pass_2_s=80.0, pass_2_energy_j=12000.0)
    assert terms.delta_s == 50.0 + TURN + 80.0
    assert terms.energy_j == 7000.0 + 12000.0
    assert terms.time == pytest.approx(((50.0 + TURN + 80.0) / T_NOM) ** 2, rel=1e-15)
    # A slower band's Pass 2 costs the plan, even with the same Pass 1.
    slower = _score(weights={A: 1.0, B: 1.0}, served={A: 0.0}, pass_1_s=50.0,
                    pass_1_energy_j=7000.0, pass_2_s=95.0, pass_2_energy_j=12000.0)
    assert slower.v < terms.v


def test_t_is_the_reference_period():
    weights = {A: 1.0, B: 2.0}
    base = _score(weights=weights, served={A: 0.2})
    doubled = _score(weights=weights, served={A: 0.2}, t_ref_s=2 * T_NOM)
    assert doubled.time == pytest.approx(base.time / 4.0, rel=1e-15)
    assert doubled.energy == pytest.approx(base.energy / 2.0, rel=1e-15)
    assert (doubled.delta_s, doubled.coverage, doubled.link) == (base.delta_s, base.coverage,
                                                                 base.link)


@pytest.mark.parametrize("seed", range(60))
def test_v_falls_as_coverage_falls_with_delta_and_energy_held(seed):
    """The plan's test (L842) as the design restates it (§4): with Δ and E
    held, c₂ > 0 and c₃ ≤ c₂, leaving out any one more device strictly lowers
    V (here κ > 0 and every outage is below 1)."""
    rng = random.Random(seed)
    devices = _devices(rng.randint(2, 8))
    states = {d: _state(d, rng.randint(0, 5)) for d in devices}
    ages = {d: rng.randint(0, 6) for d in devices}
    weights = demand_weights(devices, states, ages=ages, miss_priority=rng.random() < 0.5,
                             mode=rng.choice(COVERAGE_WEIGHTS))
    kappa = rng.choice(PILOT_KAPPAS + (2.0,))
    c2 = kappa * len(devices)
    params = PlanScoreParams(c_cov_per_device=kappa,
                             c_link=rng.choice((None, 0.0, c2 * rng.random(), c2)),
                             c_energy=rng.choice(PILOT_C_ENERGIES),
                             dwell_in_delta=rng.random() < 0.5)
    assert params.constants(len(devices))[2] <= c2
    outage = {d: rng.random() for d in devices}
    served = rng.sample(devices, rng.randint(2, len(devices)))
    held = dict(pass_1_s=rng.uniform(20.0, 120.0), pass_2_s=rng.uniform(20.0, 120.0))
    base = _score(params, weights=weights, served={d: outage[d] for d in served}, **held)
    for left_out in served:
        fewer = _score(params, weights=weights,
                       served={d: outage[d] for d in served if d != left_out}, **held)
        assert (fewer.delta_s, fewer.energy_j) == (base.delta_s, base.energy_j)
        assert fewer.coverage > base.coverage
        assert fewer.v < base.v


def test_a_device_certain_to_be_in_outage_is_worth_nothing_only_when_c3_equals_c2():
    """The gain from serving device j is (w_j/Σw)(c₂ − c₃·p_out,j): zero at
    c₃ = c₂ and p_out = 1, positive below, and negative once c₃ > c₂, which is
    why the property needs c₃ ≤ c₂."""
    weights = {A: 2.0, B: 3.0, C: 1.0}
    served, lost = {A: 0.3, B: 1.0}, {A: 0.3}
    equal = PlanScoreParams()                                   # c3 = c2 = 3
    assert _score(equal, weights=weights, served=served).v == pytest.approx(
        _score(equal, weights=weights, served=lost).v, abs=1e-12)
    below = PlanScoreParams(c_link=2.0)
    assert _score(below, weights=weights, served=served).v > _score(
        below, weights=weights, served=lost).v
    above = PlanScoreParams(c_link=4.0)
    assert _score(above, weights=weights, served=served).v < _score(
        above, weights=weights, served=lost).v


@pytest.mark.parametrize("p_out", [0.0, 0.3, 0.999, 1.0])
def test_at_c2_zero_serving_more_at_the_same_delta_and_energy_gains_nothing(p_out):
    """The property needs c₂ > 0 (design §4): with c₃ ≤ c₂ = 0, which is F-cov,
    the gain (w_j/Σw)(c₂ − c₃·p_out,j) is 0 for every device whatever its
    outage, so V does not see coverage and only the cap makes F-cov serve."""
    f_cov = PlanScoreParams.from_mapping({"c_cov_per_device": 0, "c_link": 0})
    assert f_cov.constants(3)[1:3] == (0.0, 0.0)
    weights = {A: 2.0, B: 3.0, C: 1.0}
    one = _score(f_cov, weights=weights, served={A: 0.3})
    two = _score(f_cov, weights=weights, served={A: 0.3, B: p_out})
    assert (two.delta_s, two.energy_j) == (one.delta_s, one.energy_j)
    assert two.coverage < one.coverage
    assert two.v == one.v


def test_v_of_the_empty_plan_is_the_turnaround_and_the_whole_demand_forgone():
    """Decision 2 (b): the empty plan flies neither pass (no Pass 2 without an
    update) and still pays the turnaround; V = −c₁(t_turn/T)² − c₂."""
    weights = {A: 1.0, B: 4.0, C: 9.0}
    empty = _score(weights=weights, served={}, **EMPTY)
    c1, c2, _, _ = PlanScoreParams().constants(3)
    assert empty.delta_s == TURN
    assert (empty.energy_j, empty.energy) == (0.0, 0.0)          # Pass 2's energy not flown
    assert (empty.coverage, empty.link) == (1.0, 0.0)
    assert (empty.served_weight, empty.demand_weight) == (0.0, 14.0)
    assert empty.v == pytest.approx(-c1 * (TURN / T_NOM) ** 2 - c2, rel=1e-15)
    # Without a turnaround it is the design's original V(empty) = −c₂.
    assert _score(weights=weights, served={}, **dict(EMPTY, turnaround_s=0.0)).v == -c2
    # The class's Pass 2 does not enter the empty plan: any class scores it alike.
    other = _score(weights=weights, served={}, **dict(EMPTY, pass_2_s=5.0, pass_2_dwell_s=1.0,
                                                       pass_2_energy_j=1.0))
    assert other == empty


def test_an_empty_demand_still_scores_finite_terms():
    """Nothing demanded, nothing forgone: U = L = 0 and c₂ = c₃ = 0."""
    empty = _score(weights={}, served={}, **EMPTY)
    assert (empty.coverage, empty.link, empty.demand_weight) == (0.0, 0.0, 0.0)
    assert empty.v == pytest.approx(-(TURN / T_NOM) ** 2, rel=1e-15)
    assert all(math.isfinite(x) for x in empty.as_dict().values())
    nothing = _score(weights={}, served={}, **dict(EMPTY, turnaround_s=0.0))
    assert nothing.v == 0.0 and math.copysign(1.0, nothing.v) == 1.0     # 0.0, never -0.0


@pytest.mark.parametrize("n", range(1, 9))
def test_uniform_weights_reproduce_one_minus_served_over_n(n):
    """The plan's letter (L830): uniform weights without the streak factor give
    U = 1 − served/N exactly, and at κ = 1 each device left out costs 1."""
    devices = _devices(n)
    states = {d: _state(d, i) for i, d in enumerate(devices)}      # streaks not read
    weights = demand_weights(devices, states, ages=None, miss_priority=False,
                             mode=COVERAGE_WEIGHTS_UNIFORM)
    assert set(weights.values()) == {1.0}
    c2 = PlanScoreParams().constants(n)[1]
    for s in range(n + 1):
        terms = _score(weights=weights, served={d: 0.0 for d in devices[:s]})
        assert terms.coverage == 1 - s / n
        assert (terms.served_weight, terms.demand_weight) == (s, n)
        assert c2 * terms.coverage == pytest.approx(n - s, rel=1e-12, abs=1e-12)
    # Equal ages without the streak reduce to the letter too (design D-B).
    same_age = demand_weights(devices, states, ages={d: 3 for d in devices},
                              miss_priority=False, mode=COVERAGE_WEIGHTS_AGE)
    for s in range(n + 1):
        assert _score(weights=same_age, served={d: 0.0 for d in devices[:s]}).coverage == 1 - s / n


def test_with_the_streak_factor_uniform_weights_favour_the_devices_missed_longest():
    devices = _devices(3)
    states = {d: _state(d, i) for i, d in enumerate(devices)}      # streaks 0, 1, 2
    weights = demand_weights(devices, states, ages=None, miss_priority=True,
                             mode=COVERAGE_WEIGHTS_UNIFORM)
    fresh = _score(weights=weights, served={devices[0]: 0.0})
    missed = _score(weights=weights, served={devices[2]: 0.0})
    assert fresh.coverage == pytest.approx(1 - 1 / 6) and missed.coverage == pytest.approx(0.5)
    assert missed.v > fresh.v


def test_f_prio_coverage_is_the_share_of_network_aou_the_mission_leaves():
    """Age weights without the streak: 1 − U is the share of the Network AoU
    that serving the plan removes, with the scorer's age after the mission
    (0 for a merged device, a_j otherwise; design D-A)."""
    ages = {A: 1, B: 2, C: 3}
    weights = demand_weights(tuple(ages), {d: _state(d) for d in ages}, ages=ages,
                             miss_priority=False, mode=COVERAGE_WEIGHTS_AGE)
    for served in ({C}, {A, B}, {B}, {A, B, C}):
        terms = _score(weights=weights, served={d: 0.0 for d in served})
        aou_if_none = sum(ages.values()) / len(ages)
        aou_after = sum(a for d, a in ages.items() if d not in served) / len(ages)
        assert 1.0 - terms.coverage == pytest.approx((aou_if_none - aou_after) / aou_if_none)


# -- the constants ----------------------------------------------------------- #

@pytest.mark.parametrize("n", [1, 2, 6, 24])
def test_the_default_constants_reach_v(n):
    """c₁ = 1, c₂ = κ·N_demand with κ = 1, c₃ = c₂, c₄ = 0.1 (spec, other choices 7)."""
    assert PlanScoreParams().constants(n) == (1.0, float(n), float(n), 0.1)
    devices = _devices(n)
    weights = {d: float(i + 1) for i, d in enumerate(devices)}
    terms = _score(weights=weights, served={d: 0.25 for d in devices[: n // 2 + 1]})
    assert terms.v == pytest.approx(
        -(terms.time + n * terms.coverage + n * terms.link) - 0.1 * terms.energy, rel=1e-14)


def test_c3_follows_c2_unless_set():
    weights, served = {A: 1.0, B: 1.0}, {A: 0.5}
    tied = _score(PlanScoreParams(c_cov_per_device=0.25), weights=weights, served=served)
    apart = _score(PlanScoreParams(c_cov_per_device=0.25, c_link=0.1), weights=weights,
                   served=served)
    assert tied.v - apart.v == pytest.approx(-(0.5 - 0.1) * tied.link, rel=1e-12)


def test_kappa_one_prices_one_average_device_as_one_full_t_squared():
    """Decision 2: one average device is worth κ·T² of time; with κ = 1,
    leaving one of N uniform devices out costs what a Δ of T costs."""
    devices = _devices(4)
    weights = {d: 1.0 for d in devices}
    everyone = _score(weights=weights, served={d: 0.0 for d in devices})
    one_out = _score(weights=weights, served={d: 0.0 for d in devices[:3]})
    assert everyone.v - one_out.v == pytest.approx(1.0 * (T_NOM / T_NOM) ** 2, rel=1e-12)
    quarter = PlanScoreParams(c_cov_per_device=0.25)
    assert (_score(quarter, weights=weights, served={d: 0.0 for d in devices}).v
            - _score(quarter, weights=weights, served={d: 0.0 for d in devices[:3]}).v
            ) == pytest.approx(0.25, rel=1e-12)


def test_the_pilot_sweep_holds_the_defaults():
    """Decision 2: the pilot sweeps κ in {0.15, 0.25, 1} and c₄ in {0, 0.1}."""
    assert PILOT_KAPPAS == (0.15, 0.25, 1.0)
    assert PILOT_C_ENERGIES == (0.0, 0.1)
    default = PlanScoreParams()
    assert default.c_cov_per_device in PILOT_KAPPAS and default.c_energy in PILOT_C_ENERGIES
    for kappa, c4 in itertools.product(PILOT_KAPPAS, PILOT_C_ENERGIES):
        params = PlanScoreParams.from_mapping({"c_cov_per_device": kappa, "c_energy": c4})
        terms = _score(params, weights={A: 1.0, B: 2.0}, served={B: 0.1})
        assert math.isfinite(terms.v) and terms.v < 0.0


def test_f_cov_is_cap_only_service():
    """c₂ = c₃ = 0 (F-cov): V no longer pays for coverage, so the empty plan,
    which only pays the turnaround, beats every plan that flies (decision 3);
    only the cap's key, ahead of V, can make F-cov serve anyone."""
    f_cov = PlanScoreParams.from_mapping({"c_cov_per_device": 0, "c_link": 0})
    weights = {d: float(i + 1) for i, d in enumerate(_devices(4))}
    empty = _score(f_cov, weights=weights, served={}, **EMPTY)
    assert empty.v == pytest.approx(-(TURN / T_NOM) ** 2, rel=1e-15)
    for k in range(1, 5):
        served = {d: 0.0 for d in list(weights)[:k]}
        flown = _score(f_cov, weights=weights, served=served, pass_1_s=5.0 * k,
                       pass_1_dwell_s=1.0 * k, pass_1_energy_j=500.0 * k)
        assert flown.v < empty.v


# -- dwell_in_delta (arm F-dwell, Study 5.7) --------------------------------- #

def test_f_dwell_takes_both_passes_dwell_out_of_delta_only():
    weights, served = {A: 2.0, B: 1.0}, {A: 0.1}
    kwargs = dict(pass_1_s=50.0, pass_1_dwell_s=5.0, pass_2_s=80.0, pass_2_dwell_s=6.0)
    on = _score(PlanScoreParams(), weights=weights, served=served, **kwargs)
    off = _score(PlanScoreParams(dwell_in_delta=False), weights=weights, served=served, **kwargs)
    assert on.delta_s == 50.0 + TURN + 80.0
    assert off.delta_s == 50.0 + TURN + 80.0 - 5.0 - 6.0
    # delta_s is the Δ that V prices; the predicted mission is still the whole one.
    assert predicted_mission_s(serves_any=True, pass_1_s=50.0, turnaround_s=TURN,
                               pass_2_s=80.0) == on.delta_s
    assert off.time == pytest.approx((off.delta_s / T_NOM) ** 2, rel=1e-15)
    # E keeps the hovering; the coverage and link terms are untouched.
    assert (off.energy_j, off.energy) == (on.energy_j, on.energy)
    assert (off.coverage, off.link, off.served_weight) == (on.coverage, on.link, on.served_weight)
    assert off.v > on.v


def test_f_dwell_prices_plans_that_differ_only_in_dwell_alike():
    weights, served = {A: 1.0}, {A: 0.0}
    short = dict(pass_1_s=40.0, pass_1_dwell_s=2.0)
    long_ = dict(pass_1_s=58.0, pass_1_dwell_s=20.0)          # the same legs, 18 s more dwell
    on = PlanScoreParams()
    off = PlanScoreParams(dwell_in_delta=False)
    assert _score(on, weights=weights, served=served, **long_).v < _score(
        on, weights=weights, served=served, **short).v
    assert _score(off, weights=weights, served=served, **long_).v == _score(
        off, weights=weights, served=served, **short).v


def test_f_dwell_leaves_the_empty_plan_alone():
    weights = {A: 1.0, B: 1.0}
    on = _score(PlanScoreParams(), weights=weights, served={}, **EMPTY)
    off = _score(PlanScoreParams(dwell_in_delta=False), weights=weights, served={}, **EMPTY)
    assert on == off and off.delta_s == TURN                    # Pass 2's dwell is not flown


def test_f_dwell_never_prices_a_negative_delta():
    """A dwell summed apart from its pass may exceed it by rounding; Δ stays >= 0."""
    terms = _score(PlanScoreParams(dwell_in_delta=False), weights={A: 1.0}, served={A: 0.0},
                   pass_1_s=0.3, pass_1_dwell_s=0.3 + 1e-12, turnaround_s=0.0,
                   pass_2_s=0.5, pass_2_dwell_s=0.5)
    assert terms.delta_s == 0.0 and terms.time == 0.0


# -- the predicted mission (design D-M) --------------------------------------- #

@pytest.mark.parametrize("seed", range(20))
def test_the_predicted_mission_is_delta_with_the_dwell_in_for_every_arm(seed):
    """Design D-M sets the predicted mission against the realized ledger, so it
    must not move with the arm: it is F's Δ bit for bit, and F-dwell's Δ plus
    the dwell of the passes flown, while delta_s is the Δ each arm prices."""
    rng = random.Random(seed)
    devices = _devices(rng.randint(1, 5))
    weights = {d: float(rng.randint(1, 9)) for d in devices}
    served = {d: rng.random() for d in rng.sample(devices, rng.randint(0, len(devices)))}
    p1 = rng.uniform(5.0, 150.0) if served else 0.0
    p2 = rng.uniform(5.0, 120.0)
    d1, d2 = rng.uniform(0.0, p1), rng.uniform(0.0, p2)
    turn = rng.choice((0.0, TURN))
    times = dict(pass_1_s=p1, pass_1_dwell_s=d1, turnaround_s=turn, pass_2_s=p2, pass_2_dwell_s=d2)
    mission = predicted_mission_s(serves_any=bool(served), pass_1_s=p1, turnaround_s=turn,
                                  pass_2_s=p2)
    f = _score(PlanScoreParams(), weights=weights, served=served, **times)
    f_dwell = _score(PlanScoreParams(dwell_in_delta=False), weights=weights, served=served, **times)
    assert mission == f.delta_s == p1 + turn + (p2 if served else 0.0)
    assert f_dwell.delta_s == max(0.0, mission - (d1 + (d2 if served else 0.0)))


def test_pass_2_is_flown_exactly_when_the_plan_serves_someone():
    """Pass 2 follows the served set, not Pass 1's time (decision 2 (b); the
    supervisor skips Pass 2 after a mission that collects nothing), also at
    the edges the planner does not reach: a served plan whose Pass 1 took no
    time still flies Pass 2, and an empty plan charged Pass-1 time does not."""
    weights = {A: 1.0, B: 1.0}
    instant = _score(weights=weights, served={A: 0.0}, pass_1_s=0.0, pass_1_dwell_s=0.0,
                     pass_1_energy_j=0.0)
    assert instant.delta_s == TURN + HELD["pass_2_s"]
    assert instant.energy_j == HELD["pass_2_energy_j"]
    idle = _score(weights=weights, served={}, pass_1_s=12.0, pass_1_dwell_s=0.0,
                  pass_1_energy_j=500.0)
    assert (idle.delta_s, idle.energy_j) == (12.0 + TURN, 500.0)
    for serves_any, p1, expected in ((True, 0.0, instant.delta_s), (False, 12.0, idle.delta_s)):
        assert predicted_mission_s(serves_any=serves_any, pass_1_s=p1, turnaround_s=TURN,
                                   pass_2_s=HELD["pass_2_s"]) == expected


@pytest.mark.parametrize("kwargs, error", [
    (dict(serves_any=1), TypeError),                       # a bool, not a count
    (dict(serves_any=frozenset({A})), TypeError),          # nor the served set itself
    (dict(serves_any=None), TypeError),
    (dict(pass_1_s=-1.0), ValueError),
    (dict(turnaround_s=math.nan), ValueError),
    (dict(pass_2_s=math.inf), ValueError),
    (dict(pass_2_s="80"), TypeError),
])
def test_the_predicted_mission_refuses_bad_values(kwargs, error):
    values = dict(serves_any=True, pass_1_s=50.0, turnaround_s=TURN, pass_2_s=80.0)
    values.update(kwargs)
    with pytest.raises(error):
        predicted_mission_s(**values)


# -- the rank (the orchestrator's resolution R11) ------------------------------ #

def test_the_coverage_rank_is_a_score_setting_lexicographic_by_default():
    """``lexicographic`` is the default and ``weighted`` the pilot's κ sweep.
    The setting is appended to the score settings (so every earlier one keeps
    its place), reaches ``mule_ready`` through the options and comes back from
    it; the three names are defined once, beside the other switch values."""
    assert COVERAGE_RANKS == ("lexicographic", "weighted")
    assert (COVERAGE_RANK_LEXICOGRAPHIC, COVERAGE_RANK_WEIGHTED) == COVERAGE_RANKS
    assert COVERAGE_RANKS is plan_types.COVERAGE_RANKS
    default = PlanScoreParams()
    assert default.coverage_rank == COVERAGE_RANK_LEXICOGRAPHIC
    assert list(default.as_dict()) == ["c_time", "c_cov_per_device", "c_link", "c_energy",
                                       "coverage_weights", "dwell_in_delta", "coverage_rank"]
    swept = PlanScoreParams.from_mapping({"c_cov_per_device": 0.15, "coverage_rank": "weighted"})
    assert swept.coverage_rank == COVERAGE_RANK_WEIGHTED and swept.c_cov_per_device == 0.15
    assert PlanScoreParams.from_mapping(swept.as_dict()) == swept
    options = PlanOptions.from_config(plan_score_params={"coverage_rank": "weighted"})
    assert options.describe()["plan_score_params"]["coverage_rank"] == "weighted"
    assert PlanOptions.from_config(**options.describe()) == options
    assert PlanOptions().describe()["plan_score_params"]["coverage_rank"] == "lexicographic"
    with pytest.raises(ValueError, match="coverage_ranking"):
        PlanScoreParams.from_mapping({"coverage_ranking": "weighted"})   # misspelt: never ignored


@pytest.mark.parametrize("bad", ["Lexicographic", "weighted ", "", "share", "age", None, 1, True,
                                 ("lexicographic",)])
def test_the_coverage_rank_refuses_anything_else(bad):
    with pytest.raises(ValueError, match="coverage_rank"):
        PlanScoreParams(coverage_rank=bad)
    with pytest.raises(ValueError, match="coverage_rank"):
        PlanScoreParams.from_mapping({"coverage_rank": bad})


@pytest.mark.parametrize("seed", range(20))
def test_the_rank_never_changes_v(seed):
    """The rank changes how candidates are ordered, not V or its terms: the
    same primitives score bit for bit alike under either setting."""
    rng = random.Random(seed)
    devices = _devices(rng.randint(1, 7))
    weights = {d: float(rng.randint(1, 30)) for d in devices}
    served = {d: rng.random() for d in rng.sample(devices, rng.randint(0, len(devices)))}
    common = dict(c_cov_per_device=rng.choice(PILOT_KAPPAS + (0.0,)),
                  c_energy=rng.choice(PILOT_C_ENERGIES), dwell_in_delta=rng.random() < 0.5)
    times = dict(pass_1_s=rng.uniform(20.0, 120.0), pass_2_s=rng.uniform(20.0, 120.0))
    lex = _score(PlanScoreParams(**common), weights=weights, served=served, **times)
    weighted = _score(PlanScoreParams(coverage_rank="weighted", **common), weights=weights,
                      served=served, **times)
    assert lex == weighted


def test_the_served_share_is_the_served_weight_over_the_demand():
    weights = {A: 1.0, B: 2.0, C: 3.0, D: 4.0}
    terms = _score(weights=weights, served={B: 0.1, C: 0.5})
    assert served_share(terms) == 0.5 == 1.0 - terms.coverage
    assert served_share(_score(weights=weights, served={}, **EMPTY)) == 0.0
    assert served_share(_score(weights=weights, served={d: 0.0 for d in weights})) == 1.0
    # An empty demand forgoes nothing (U = 0), so its share is 1.
    nothing = _score(weights={}, served={}, **EMPTY)
    assert nothing.coverage == 0.0 and served_share(nothing) == 1.0
    # Uniform weights: the plan's letter, served / N.
    for n in range(1, 7):
        devices = _devices(n)
        for s in range(n + 1):
            uniform = _score(weights={d: 1.0 for d in devices},
                             served={d: 0.0 for d in devices[:s]})
            assert served_share(uniform) == s / n
    with pytest.raises(TypeError):
        served_share({"served_weight": 1.0, "demand_weight": 2.0})


@pytest.mark.parametrize("seed", range(20))
def test_the_served_share_is_one_less_the_coverage_shortfall(seed):
    rng = random.Random(seed)
    devices = _devices(rng.randint(1, 8))
    weights = {d: rng.uniform(1.0, 40.0) for d in devices}
    served = {d: rng.random() for d in rng.sample(devices, rng.randint(0, len(devices)))}
    terms = _score(weights=weights, served=served)
    share = served_share(terms)
    assert share == terms.served_weight / terms.demand_weight
    assert share == pytest.approx(1.0 - terms.coverage, rel=0.0, abs=1e-15)
    assert 0.0 <= share <= 1.0
    assert (share == 1.0) is (len(served) == len(devices))
    assert (share == 0.0) is (not served)


def test_the_applied_rank_is_the_setting_unless_the_coverage_term_is_off():
    """F-cov (κ = 0) ranks by V alone under either setting: a share-first rank
    would serve everyone the budget allows, and F-cov is cap-only service
    (decision 3). Any κ above 0, however small, keeps the setting."""
    assert applied_rank(PlanScoreParams()) == COVERAGE_RANK_LEXICOGRAPHIC
    assert applied_rank(PlanScoreParams(coverage_rank="weighted")) == COVERAGE_RANK_WEIGHTED
    for rank in COVERAGE_RANKS:
        for kappa in PILOT_KAPPAS + (1e-9, 2.0):
            assert applied_rank(PlanScoreParams(c_cov_per_device=kappa, coverage_rank=rank)) == rank
        f_cov = PlanScoreParams.from_mapping({"c_cov_per_device": 0, "c_link": 0,
                                              "coverage_rank": rank})
        assert applied_rank(f_cov) == COVERAGE_RANK_WEIGHTED
        no_coverage = PlanScoreParams(c_cov_per_device=0.0, c_link=1.0, coverage_rank=rank)
        assert applied_rank(no_coverage) == COVERAGE_RANK_WEIGHTED
    with pytest.raises(TypeError):
        applied_rank({"coverage_rank": "weighted"})


def _plan_class(index=0):
    """A class for hand-built candidates: the plan key reads only its index."""
    physics = FerryPhysics(dock=(0.0, 0.0, 0.0), member_dwell_s=lambda d, p, o: 1.0,
                           upload_s=lambda: 0.0, p_move_w=143.6, p_hover_w=P_HOVER,
                           range_m=60.0)
    model = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=1.0, ferry=physics)
    return PlanClass(name=f"class{index}", index=index, radius_m=60.0, model=model,
                     outage=lambda d: 0.0)


def _stop(device, x):
    return ContactWaypoint(position=(x, 0.0, 0.0), devices=(device,),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=1e6)


def _candidate(terms, *, route=(), cap_key=(), index=0):
    fold = MemberFold(route=tuple(route), dropped=(), state=FlightState((0.0, 0.0, 0.0), 0.0),
                      home=0.0, feasible=True, energy_j=0.0)
    return Candidate(cls=_plan_class(index), fold=fold, terms=terms, cap_key=cap_key)


#: U7's empty-plan probe (Phase 4 build): u and v 60 m either side of
#: the dock fit alone, not together, and z 400 m out fits nowhere; FB+wide at
#: 1 MB, T = 200 s, the cap off, every weight 1. Serving v is priced over a
#: 264.4 s mission (Pass 2 reaches out to z) and 33.65 kJ; the empty plan pays
#: its 30 s turnaround only.
PROBE_T = dict(t_ref_s=200.0, turnaround_s=TURN)


def test_on_u7s_probe_numbers_the_share_serves_the_device_v_alone_would_leave():
    """V alone prefers flying empty (−3.02 against −3.85): the plan that serves
    pays a whole Pass 2. The lexicographic key serves the device; the
    weighted key, U0's ``Candidate.key``, flies empty."""
    u, v, z = (DeviceID(x) for x in "uvz")
    weights = {u: 1.0, v: 1.0, z: 1.0}
    serve_v = _candidate(_score(weights=weights, served={v: 0.0}, pass_1_s=25.198,
                                pass_1_dwell_s=0.8, pass_1_energy_j=3700.0, pass_2_s=209.1972,
                                pass_2_dwell_s=2.4, pass_2_energy_j=29951.5204, **PROBE_T),
                         route=[_stop(v, -60.0)])
    empty = _candidate(_score(weights=weights, served={}, **dict(EMPTY, **PROBE_T)))
    assert serve_v.terms.delta_s == pytest.approx(264.3952, abs=1e-9)
    assert round(serve_v.terms.v, 4) == -3.8475 and empty.terms.v == pytest.approx(-3.0225)
    assert serve_v.terms.v < empty.terms.v
    lex, weighted = PlanScoreParams(), PlanScoreParams(coverage_rank="weighted")
    assert (served_share(serve_v.terms), served_share(empty.terms)) == (1 / 3, 0.0)
    assert plan_key(lex, serve_v) < plan_key(lex, empty)
    assert plan_key(weighted, empty) < plan_key(weighted, serve_v)
    for cand in (serve_v, empty):
        assert plan_key(weighted, cand) == cand.key
        assert plan_key(lex, cand) == (cand.cap_key, -round(served_share(cand.terms), 9)) + (
            cand.key[1:])


def test_the_lexicographic_key_ranks_the_cap_key_the_share_v_the_class_then_the_stops():
    lex, weighted = PlanScoreParams(), PlanScoreParams(coverage_rank="weighted")
    weights = {A: 1.0, B: 2.0, C: 3.0}

    def cand(served, *, cap_key=(), index=0, pass_1_s=40.0, x=10.0):
        terms = _score(weights=weights, served={d: 0.0 for d in served}, pass_1_s=pass_1_s)
        route = [_stop(d, x + 10.0 * i) for i, d in enumerate(served)]
        return _candidate(terms, route=route, cap_key=cap_key, index=index)

    # The cap key first: keeping an older capped device beats a larger share.
    keeps, shares_more = cand([A], cap_key=(2,)), cand([B, C], cap_key=(3,))
    assert served_share(keeps.terms) < served_share(shares_more.terms)
    assert plan_key(lex, keeps) < plan_key(lex, shares_more)
    # Then the share, whatever V: a slow plan serving more weight beats a fast one.
    slow, fast = cand([B, C], pass_1_s=400.0), cand([C], pass_1_s=15.0)
    assert slow.terms.v < fast.terms.v
    assert plan_key(lex, slow) < plan_key(lex, fast)
    assert plan_key(weighted, fast) < plan_key(weighted, slow)
    # Equal shares: V decides, under either rank.
    quick, late = cand([C], pass_1_s=20.0), cand([C], pass_1_s=50.0)
    for params in (lex, weighted):
        assert plan_key(params, quick) < plan_key(params, late)
    # Equal share and V: the class index, then the stops.
    assert plan_key(lex, cand([C], index=0)) < plan_key(lex, cand([C], index=1))
    assert plan_key(lex, cand([C], x=5.0)) < plan_key(lex, cand([C], x=7.0))
    # Shares that differ by rounding alone tie, and V decides.
    base = quick.terms
    nudged = _candidate(dataclasses.replace(late.terms, served_weight=base.served_weight + 1e-12),
                        route=late.fold.route)
    assert served_share(nudged.terms) > served_share(base)
    assert plan_key(lex, quick)[1] == plan_key(lex, nudged)[1]
    assert plan_key(lex, quick) < plan_key(lex, nudged)
    # F-cov ranks by U0's key under either setting.
    for rank in COVERAGE_RANKS:
        f_cov = PlanScoreParams(c_cov_per_device=0.0, c_link=0.0, coverage_rank=rank)
        for c in (keeps, shares_more, slow, fast):
            assert plan_key(f_cov, c) == c.key
    with pytest.raises(TypeError):
        plan_key(lex, quick.terms)


@pytest.mark.parametrize("seed", range(30))
def test_the_plan_key_rises_as_coverage_falls_under_either_rank(seed):
    """The rank's side of "V falls as coverage falls" (L842): with Δ and E held,
    the cap key the same, κ > 0 and c₃ ≤ c₂, leaving out any one more device
    raises the plan key under either rank (the share falls, and so does V)."""
    rng = random.Random(seed)
    devices = _devices(rng.randint(2, 8))
    weights = {d: float(rng.randint(1, 30)) for d in devices}
    kappa = rng.choice(PILOT_KAPPAS + (2.0,))
    c2 = kappa * len(devices)
    common = dict(c_cov_per_device=kappa, c_link=rng.choice((None, 0.0, c2 * rng.random())),
                  c_energy=rng.choice(PILOT_C_ENERGIES))
    outage = {d: 0.99 * rng.random() for d in devices}
    served = rng.sample(devices, rng.randint(2, len(devices)))
    held = dict(pass_1_s=rng.uniform(20.0, 120.0), pass_2_s=rng.uniform(20.0, 120.0))

    def cand(members):
        terms = _score(PlanScoreParams(**common), weights=weights,
                       served={d: outage[d] for d in members}, **held)
        return _candidate(terms, route=[_stop(d, 5.0 + i) for i, d in enumerate(members)])

    base = cand(served)
    for rank in COVERAGE_RANKS:
        params = PlanScoreParams(coverage_rank=rank, **common)
        for left_out in served:
            fewer = cand([d for d in served if d != left_out])
            assert plan_key(params, base) < plan_key(params, fewer)


# -- hand-off, determinism and refusals --------------------------------------- #

def test_the_commit_records_the_weights_and_the_score():
    """U5 builds the commit from these: ``weights`` from demand_weights and
    ``score`` from the terms, with the constants resolved for the demand and
    the predicted mission as an extra term, so the trace's plan carries it."""
    demand = (A, B, C)
    states = {A: _state(A, 1), B: _state(B, 0), C: _state(C, 3)}
    ages = {A: 2, B: 1, C: 4}
    params = PlanScoreParams(dwell_in_delta=False)
    weights = demand_weights(demand, states, ages=ages, miss_priority=True,
                             mode=params.coverage_weights)
    stop = ContactWaypoint(position=(10.0, 0.0, 0.0), devices=(A, C),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=100.0)
    terms = _score(params, weights=weights, served={A: 0.05, C: 0.2})
    mission = predicted_mission_s(serves_any=True, pass_1_s=HELD["pass_1_s"], turnaround_s=TURN,
                                  pass_2_s=HELD["pass_2_s"])
    commit = PlanCommit(
        mission_round=5, band="medium", band_index=1, band_class_policy=BAND_POLICY_SEARCH,
        queue=(stop,), demand=demand, weights=weights, budget_end=60.0, t_ref_s=T_NOM,
        score=dict(terms.as_dict(), **{MISSION_SCORE_KEY: mission}),
        constants=params.constants(len(demand)), search_mode="exact",
        n_candidates=9, per_class=({"band": "medium", "v": terms.v},), ages=ages,
    )
    assert commit.served == frozenset({A, C})
    described = commit.describe()
    assert described["weights"] == {"a": 4.0, "b": 1.0, "c": 16.0}
    assert described["score"]["v"] == terms.v and described["score"]["c"] == [1.0, 3.0, 3.0, 0.1]
    assert terms.served_weight == 20.0 and terms.demand_weight == 21.0
    # Under F-dwell the priced Δ leaves both passes' dwell out; the prediction does not.
    assert described["score"]["delta_s"] == 70.0 + TURN + 60.0 - 12.0 - 9.0
    assert described["score"][MISSION_SCORE_KEY] == 70.0 + TURN + 60.0
    assert MISSION_SCORE_KEY == "mission_s"      # the trace key, which the scorer restates


def test_the_sums_do_not_depend_on_the_order_of_the_mappings():
    """Exactly rounded sums: the same plan priced along another order ties bit
    for bit, as the plan key (V to 9 decimals) and the brute force expect."""
    rng = random.Random(7)
    devices = _devices(8)
    weights = {d: rng.uniform(1.0, 50.0) for d in devices}
    served = {d: rng.random() for d in devices[:6]}
    reference = _score(weights=weights, served=served)
    for _ in range(20):
        w_order, s_order = list(weights), list(served)
        rng.shuffle(w_order)
        rng.shuffle(s_order)
        again = _score(weights={d: weights[d] for d in w_order},
                       served={d: served[d] for d in s_order})
        assert again == reference


def test_without_a_hover_power_the_energy_term_is_off():
    """PlanSetup allows P_hover = 0 only with c₄ = 0; E is still reported."""
    terms = _score(PlanScoreParams(c_energy=0.0), weights={A: 1.0}, served={A: 0.0}, p_hover_w=0.0)
    assert terms.energy == 0.0
    assert terms.energy_j == HELD["pass_1_energy_j"] + HELD["pass_2_energy_j"]
    with pytest.raises(ValueError):
        _score(PlanScoreParams(), weights={A: 1.0}, served={A: 0.0}, p_hover_w=0.0)


@pytest.mark.parametrize("overrides, error", [
    (dict(params={"c_time": 1.0}), TypeError),
    (dict(weights=[(A, 1.0)]), TypeError),
    (dict(weights={A: 0.0, B: 1.0}), ValueError),              # every device counts
    (dict(weights={A: -1.0, B: 1.0}), ValueError),
    (dict(weights={A: math.nan, B: 1.0}), ValueError),
    (dict(weights={A: math.inf, B: 1.0}), ValueError),
    (dict(weights={A: True, B: 1.0}), TypeError),
    (dict(weights={A: "1", B: 1.0}), TypeError),
    (dict(weights={3: 1.0, B: 1.0}), TypeError),
    (dict(weights={"": 1.0, B: 1.0}), TypeError),
    (dict(served=[(A, 0.1)]), TypeError),
    (dict(served={C: 0.1}), ValueError),                       # not in the demand
    (dict(served={A: -0.01}), ValueError),
    (dict(served={A: 1.01}), ValueError),
    (dict(served={A: math.nan}), ValueError),
    (dict(served={A: False}), TypeError),
    (dict(pass_1_s=-1.0), ValueError),
    (dict(pass_1_s=math.inf), ValueError),
    (dict(pass_1_s=None), TypeError),
    (dict(pass_1_dwell_s=-0.5), ValueError),
    (dict(pass_1_dwell_s=70.5), ValueError),                   # more than the pass
    (dict(pass_1_energy_j=-1.0), ValueError),
    (dict(turnaround_s=-30.0), ValueError),
    (dict(pass_2_s=math.nan), ValueError),
    (dict(pass_2_dwell_s=60.5), ValueError),
    (dict(pass_2_energy_j=True), TypeError),
    (dict(t_ref_s=0.0), ValueError),
    (dict(t_ref_s=-220.0), ValueError),
    (dict(t_ref_s=math.inf), ValueError),
    (dict(p_hover_w=-168.5), ValueError),
])
def test_the_score_refuses_bad_values(overrides, error):
    values = dict(params=PlanScoreParams(), weights={A: 1.0, B: 2.0}, served={A: 0.1})
    values.update(overrides)
    with pytest.raises(error):
        _score(values.pop("params"), weights=values.pop("weights"), served=values.pop("served"),
               **values)
