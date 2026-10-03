"""FeRRy Phase 4 (unit U8): the per-role config, the mule process, the builder, the
driver and the runner.

Pinned (the Phase 4 spec, other choices 10-12; unit_U3b.md section 5.4; the
user's decisions 4 (b) and 6; critic A8, A10, B8, B11, B12, D2):

* **Config.** The plan fields (``PLAN_MULE_FIELDS``) are simulated-clock only
  and not ferry-spec fields, so Phase 3's ``ferry_params`` are unchanged; the
  switch values restated in ``hermes.processes.config`` equal their sources;
  every guard of ``mule_config_errors`` fires for its own reason, and the
  configurations the arms fly pass. Old per-role JSON loads with the plan
  fields at their defaults.
* **The mule process.** Plan mode hands the supervisor the options its config
  describes, T_nom and the member admission; ``mule_ready`` states the plan
  settings the scheduler runs; ``mission_completed`` carries ``plan``,
  ``plan_wall_s`` and ``pass_1_policy_drops`` only when a mission has them. A
  recorded mule's supervisor gets no new argument and loads no plan module.
* **The builder.** The plan fields are explicit parameters, copied to every
  mule (each with its own dicts), refused off the simulated clock.
* **The driver.** The eight plan arms and their labels; the nine Phase 3 arms
  stay the runner's default; each arm's plan fields, band (FB+<class>'s in its
  ferry settings), trim fallback, member admission (D4 always whole) and
  ``miss_priority`` (in the row too); plan arms refused on the wall clock and
  wherever they cannot fly, before anything is spawned; T_nom forced for them;
  the plan keys in ``ferry_params`` in plan mode only, ``contact_band`` reading
  ``search`` for the search arms; the trial CSV header unchanged.
* **Rule 1 at the defaults** (UG4's hand-off: the goldens let an added key
  pass, so absence is pinned here). Re-running the eight Phase 3 simulated-
  clock oracles (``tests/golden/_build_p3_sim.py``), no event gains a key but
  a D arm's ``pass_1_policy_drops`` where it left something out, the rows'
  strings equal the recorded ones, and the per-role JSON gains exactly the
  plan fields at their defaults.
* **Every plan arm in process**, end to end through the real mule, cluster and
  device services (UG4's in-process harness), and H1, D1 and D5 with member
  subsets on the Phase 3 cliff layout.
* **The runner.** The plan flags (the score and search settings as JSON, so a
  pilot sweeps kappa and the coverage rank), ``--member-admission`` and
  ``--base-seed`` reach the driver and the grid; a plan arm the driver cannot
  run is a usage error.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.analysis.traces_scorer import parse_trial_dir
from experiments.exp4.driver import (
    ARMS,
    DEFAULT_ARMS,
    F_COV_SCORE,
    PHASE_5_ARMS, PLAN_ARMS,
    PROVENANCE_COLUMNS,
    Exp4Driver,
    contact_band_column,
    plan_ferry_params,
    trace_dir_name,
)
from experiments.exp4.metrics import Exp4MetricSummary
from experiments.exp4.topology_builder import build_exp4_topology
from experiments.runner import Cell
from hermes.mule import mule_main
from hermes.mule.mule_main import MissionRunResult
from hermes.processes import config as C
from hermes.processes import mule as mule_process
from hermes.processes.config import (
    FERRY_SPEC_FIELDS,
    PLAN_MULE_FIELDS,
    PLAN_OPTION_FIELDS,
    SIM_ONLY_MULE_FIELDS,
    MuleConfig,
    TopologyValidationError,
    mule_config_errors,
    mule_config_from_json,
    mule_config_to_json,
)
from hermes.scheduler.plan import PlanOptions
from hermes.scheduler.plan import types as PT
from hermes.scheduler.policies.cross_heuristic import CommittedSlot, CrossHeuristic
from hermes.scheduler.stages import s3b_feasibility as S3B
from hermes.transport import TCPDockLinkServer
from hermes.types import scheduler as TS

from tests.golden import _build_p3_sim as P3
from tests.golden import _build_topology as T

REPO = Path(__file__).resolve().parents[2]

#: The plan fields' recorded values (``MuleConfig``'s defaults).
PLAN_DEFAULTS = {
    "plan_mode": "legacy", "band_class_policy": "search", "member_admission": "whole",
    "flight_slot": "committed", "age_cap_missions": None, "age_cap_lookahead": 0,
    "plan_score_params": {}, "plan_search_params": {},
}
#: The trial CSV's provenance columns at 6e6f92d (Phase 3); Phase 4 adds none.
PROVENANCE_6E6F92D = (
    "mission_budget_s", "mission_window_adaptation", "aggregation", "aggregation_params",
    "fedprox_rho", "pass_2_budget", "deadline_law", "deadline_params", "miss_priority",
    "n_mules", "min_participation", "dock_params", "policy_params", "mission_clock",
    "contact_band", "in_flight_response", "backhaul_model", "contact_reliability_source",
    "deadline_time_scale", "initial_window_s", "t_nom_s", "session_ttl_s", "ferry_params",
    "l1_channel", "realism", "input_dim",
)
#: The pilots' flags (Phase 4 spec, decision 7; UG4's ``PILOT``: sim clock, wide,
#: the T_nom deadline unit, re-plan with trim, agg:cutoff, the channel
#: reliability source, 1 MB, 60 s), with the cap at 2 and T_nom over 5 layouts.
PLAN_PILOT = dict(P3.PILOT, backhaul_model="seconds", age_cap_missions=2, t_nom_layouts=5)


def _cell(arm="H1", seed=7, trial=0, **params) -> Cell:
    p = {"N": 6, "rrf": 60.0, "n_missions": 4, "regime": "jittery"}
    p.update(params)
    return Cell(cell_id="|".join(f"{k}={v}" for k, v in sorted(p.items())), arm=arm,
                trial_index=trial, seed=seed, params=p)


def _sim_mule(**kw) -> MuleConfig:
    kw.setdefault("mule_id", "m")
    kw.setdefault("mission_clock", "sim")
    kw.setdefault("trial_seed", 7)
    kw.setdefault("n_missions", 4)
    kw.setdefault("rf_range_m", 60.0)
    return MuleConfig(**kw)


def _plan_mule(**kw) -> MuleConfig:
    """Arm F's mule config: plan mode on the pilots' settings."""
    base = dict(plan_mode="ferry", contact_band="wide", t_nom_s=200.0,
                in_flight_response="replan", replan_fallback="trim",
                member_admission="subset", age_cap_missions=2)
    base.update(kw)
    return _sim_mule(**base)


def _driver(**kw) -> Exp4Driver:
    return Exp4Driver(**dict(PLAN_PILOT, **kw))


# --------------------------------------------------------------------------- #
# Config: the fields, the restated values, JSON
# --------------------------------------------------------------------------- #

def test_the_restated_switch_values_equal_their_sources():
    assert C.PLAN_MODES == PT.PLAN_MODES == mule_main._PLAN_MODES == ("legacy", "ferry")
    assert C.MEMBER_ADMISSIONS == PT.MEMBER_ADMISSIONS == S3B.MEMBER_ADMISSIONS == (
        "whole", "subset")
    assert C.FLIGHT_SLOTS == PT.FLIGHT_SLOTS == ("committed", "cross_heuristic", "pair_q")
    assert (C.BAND_POLICY_SEARCH, C.BAND_POLICY_FIXED_PREFIX) == (
        TS.BAND_POLICY_SEARCH, TS.BAND_POLICY_FIXED_PREFIX) == ("search", "fixed:")
    assert (C.PLAN_MODE_LEGACY, C.PLAN_MODE_FERRY) == (PT.PLAN_MODE_LEGACY, PT.PLAN_MODE_FERRY)
    assert (C.MEMBER_ADMISSION_WHOLE, C.MEMBER_ADMISSION_SUBSET) == (
        S3B.MEMBER_ADMISSION_WHOLE, S3B.MEMBER_ADMISSION_SUBSET)
    assert (C.FLIGHT_SLOT_COMMITTED, C.FLIGHT_SLOT_CROSS_HEURISTIC) == (
        PT.FLIGHT_SLOT_COMMITTED, PT.FLIGHT_SLOT_CROSS_HEURISTIC)


def test_the_plan_fields_are_sim_only_options_and_not_ferry_spec_fields():
    """Outside ``FERRY_SPEC_FIELDS``, so Phase 3's ``ferry_params`` keep their
    strings; inside ``SIM_ONLY_MULE_FIELDS``; the options keep
    ``PlanOptions.from_config``'s names, in its order, with its defaults but
    the member admission, whose recorded value (``whole``) is the config's
    default while the F family's ``subset`` is the options' (decision 4 (b))."""
    assert set(PLAN_MULE_FIELDS) <= set(SIM_ONLY_MULE_FIELDS)
    assert not set(PLAN_MULE_FIELDS) & set(FERRY_SPEC_FIELDS)
    assert PLAN_MULE_FIELDS == ("plan_mode",) + PLAN_OPTION_FIELDS
    params = inspect.signature(PlanOptions.from_config).parameters
    assert PLAN_OPTION_FIELDS == tuple(params)
    defaults = MuleConfig(mule_id="m")
    assert {f: getattr(defaults, f) for f in PLAN_MULE_FIELDS} == PLAN_DEFAULTS
    for name in PLAN_OPTION_FIELDS:
        theirs = params[name].default
        mine = getattr(defaults, name)
        if name == "member_admission":
            assert (mine, theirs) == ("whole", "subset")
        else:
            assert mine == theirs or (mine == {} and theirs is None), name
    # Every MuleConfig field is either recorded at 6e6f92d or a plan field.
    names = [f.name for f in dataclasses.fields(MuleConfig)]
    assert names[-len(PLAN_MULE_FIELDS):] == list(PLAN_MULE_FIELDS)


def test_an_old_mule_json_loads_with_the_plan_fields_at_their_defaults():
    full = json.loads(mule_config_to_json(_sim_mule(contact_band="wide")))
    old = {k: v for k, v in full.items() if k not in PLAN_MULE_FIELDS}
    cfg = mule_config_from_json(json.dumps(old))
    assert cfg == _sim_mule(contact_band="wide") and mule_config_errors(cfg) == []
    assert {f: getattr(cfg, f) for f in PLAN_MULE_FIELDS} == PLAN_DEFAULTS


def test_a_plan_config_round_trips_through_json():
    cfg = _plan_mule(band_class_policy="search", flight_slot="cross_heuristic",
                     age_cap_lookahead=1, plan_score_params={"c_cov_per_device": 0.25},
                     plan_search_params={"heuristic_max_evaluations": 500})
    assert mule_config_errors(cfg) == []
    back = mule_config_from_json(mule_config_to_json(cfg))
    assert back == cfg and back.plan_score_params == {"c_cov_per_device": 0.25}


# --------------------------------------------------------------------------- #
# Config: every guard, for its own reason
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,value", [
    ("plan_mode", "ferry"),
    ("band_class_policy", "fixed:wide"),
    ("member_admission", "subset"),
    ("flight_slot", "cross_heuristic"),
    ("age_cap_missions", 3),
    ("age_cap_lookahead", 1),
    ("plan_score_params", {"c_energy": 0.0}),
    ("plan_search_params", {"exact_max_devices": 4}),
])
def test_every_plan_field_is_refused_on_the_wall_clock(field, value):
    """Plan mode (and member subsets) need the simulated clock."""
    errors = mule_config_errors(MuleConfig(mule_id="m", **{field: value}))
    assert len(errors) == 1 and field in errors[0] and "mission_clock='sim'" in errors[0]


@pytest.mark.parametrize("field,value", [
    ("band_class_policy", "fixed:wide"),
    ("flight_slot", "cross_heuristic"),
    ("age_cap_missions", 3),                          # the cap is plan mode only
    ("age_cap_lookahead", 1),
    ("plan_score_params", {"c_energy": 0.0}),
    ("plan_search_params", {"exact_max_devices": 4}),
])
def test_plan_options_are_refused_outside_plan_mode(field, value):
    """A legacy mule would ignore them; refused rather than silently dropped."""
    errors = mule_config_errors(_sim_mule(contact_band="wide", **{field: value}))
    assert errors == [f"{field}: plan mode only; set plan_mode='ferry' or leave the default"]


@pytest.mark.parametrize("kw", [
    dict(),                                   # H1, H3
    dict(use_rl_selector=True),               # H2
    dict(contact_policy="max_aoi"), dict(contact_policy="oort"),
    dict(contact_policy="whittle"), dict(contact_policy="fedcs"),
], ids=["H1", "H2", "D1", "D2", "D3", "D5"])
def test_member_subsets_run_on_the_h_and_d_arms_but_d4(service, kw):
    """Decision 4 (b): H1-H3, D1-D3 and D5 take ``subset``, and their mule's
    scheduler runs it; D4 (``fedex``) has no gate and is refused
    (unit_U3b.md section 1.4)."""
    cfg = _sim_mule(contact_band="narrow", member_admission="subset", **kw)
    assert mule_config_errors(cfg) == []
    svc, _events, kwargs = service(cfg)
    assert kwargs["member_admission"] == "subset"
    assert svc.supervisor.scheduler.member_admission == "subset"
    errors = mule_config_errors(_sim_mule(contact_band="narrow", contact_policy="fedex",
                                          member_admission="subset"))
    assert len(errors) == 1 and "contact_policy='fedex'" in errors[0]
    assert mule_config_errors(_sim_mule(contact_band="narrow", contact_policy="fedex",
                                        member_admission="whole")) == []


@pytest.mark.parametrize("kw,match", [
    (dict(plan_mode="plan"), "plan_mode must be one of"),
    (dict(member_admission="some"), "member_admission must be one of"),
    (dict(plan_mode="ferry", member_admission="some", contact_band="wide", t_nom_s=200.0,
          replan_fallback="trim"), "member_admission must be one of"),
])
def test_unknown_switch_values_are_refused(kw, match):
    errors = mule_config_errors(_sim_mule(**kw))
    assert any(match in e for e in errors), errors


@pytest.mark.parametrize("kw,match", [
    (dict(contact_band=None), "needs a contact_band"),
    (dict(t_nom_s=None), "set t_nom_s"),
    (dict(band_class_policy="fixed:narrow"), "must pin the run's contact_band"),
    (dict(band_class_policy="fixed:"), "must pin the run's contact_band"),
    (dict(band_class_policy="narrow"), "band_class_policy must be"),
    (dict(replan_fallback="reorder"), "replan_fallback must be 'trim'"),
    (dict(in_flight_response="abort", age_cap_missions=None, replan_fallback="reorder"),
     "replan_fallback must be 'trim'"),                                   # critic B11
    (dict(pass_2_budget=True, mission_budget_s=60.0), "critic B8"),
    (dict(age_cap_missions=0), "age_cap_missions must be None or an int >= 1"),
    (dict(age_cap_missions=-2), "age_cap_missions must be None or an int >= 1"),
    (dict(age_cap_missions=2.0), "age_cap_missions must be None or an int >= 1"),
    (dict(age_cap_missions=True), "age_cap_missions must be None or an int >= 1"),
    (dict(age_cap_missions="3"), "age_cap_missions must be None or an int >= 1"),
    (dict(age_cap_lookahead=-1), "age_cap_lookahead must be an int >= 0"),
    (dict(age_cap_lookahead=True), "age_cap_lookahead must be an int >= 0"),
    (dict(contact_policy="max_aoi"), "no contact_policy"),
    (dict(contact_policy="fedex"), "no contact_policy"),
    (dict(use_rl_selector=True), "no RL selector"),
    (dict(in_flight_response="abort"), "critic A10"),
    (dict(flight_slot="nearest"), "plan options: flight_slot must be one of"),
    (dict(band_class_policy="fixed:wide", flight_slot="cross_heuristic"),
     "plan options: band_class_policy 'fixed:wide' pins one class"),     # unit U0
    (dict(plan_score_params={"kappa": 1.0}), "plan options: plan_score_params: unknown"),
    (dict(plan_search_params={"passes": 3}), "plan options: plan_search_params: unknown"),
    (dict(plan_score_params={"coverage_weights": "squared"}), "coverage_weights"),
    (dict(plan_score_params=[("c_time", 1.0)]), "plan_score_params must be a mapping"),
    (dict(plan_search_params={"heuristic_max_evaluations": 0}), "heuristic_max_evaluations"),
])
def test_plan_mode_refuses_what_it_cannot_fly(kw, match):
    """Other choices 10: the clock, a band and T_nom; ``fixed:<c>`` equal to
    the band; the trim fallback whatever the response (the scheduler refuses
    ``reorder`` in plan mode, critic B11); no budgeted Pass 2 (B8); a cap that
    is an int >= 1; no policy or selector; no ``abort`` with a cap (A10); no
    pinned band with FX (unit U0); and the options' own types (unknown score
    or search keys). Each case fires its own reason and no other."""
    errors = mule_config_errors(_plan_mule(**kw))
    assert len(errors) == 1 and match in errors[0], errors


#: The fallback refusal, word for word. Plan mode's Pass-1 re-plan is U5's
#: ``FLScheduler._trim_plan``: U3's member trim under ``subset`` and
#: ``_trim_whole`` (whole stops only, R5) under ``whole``, the priority stops
#: first in both when the rest does not fit as flown (pinned in
#: tests/unit/test_p4_fl_scheduler_plan.py, the ``test_under_whole_*`` tests).
#: So it is not "trimming members in the flight order", as the reason read
#: before the review of U8.
TRIM_REFUSAL = (
    "plan_mode='ferry' re-plans Pass 1 by trimming the committed plan (members under "
    "member_admission='subset', whole stops only under 'whole'), priority stops first, "
    "and leaves re-ordering to the flight slot: replan_fallback must be 'trim', got "
    "'reorder' (the scheduler refuses 'reorder' in plan mode, critic B11)"
)


@pytest.mark.parametrize("response", ["replan", "abort"])
@pytest.mark.parametrize("admission", ["subset", "whole"])
def test_the_fallback_refusal_names_the_trim_of_either_admission(admission, response):
    """One reason for both admissions and both responses, since the
    scheduler refuses ``reorder`` in plan mode whatever they are (critic
    B11). It states what the trim does under each: a ``whole`` plan arm (the
    cliff comparison, design D-D) keeps its stops whole in flight too (R5)."""
    cap = 2 if response == "replan" else None           # abort flies without a cap (A10)
    cfg = _plan_mule(member_admission=admission, in_flight_response=response,
                     age_cap_missions=cap, replan_fallback="reorder")
    assert mule_config_errors(cfg) == [TRIM_REFUSAL]


@pytest.mark.parametrize("kw", [
    dict(),                                                               # F
    dict(flight_slot="cross_heuristic"),                                  # FX
    dict(band_class_policy="fixed:wide"),                                 # FB+wide
    dict(contact_band="narrow", band_class_policy="fixed:narrow"),        # FB+narrow
    dict(plan_score_params=dict(F_COV_SCORE)),                            # F-cov
    dict(age_cap_missions=None),                                          # F-cap
    dict(age_cap_missions=None, in_flight_response="abort"),              # abort, no cap
    dict(member_admission="whole"),                                       # the cliff side
    dict(age_cap_missions=3, age_cap_lookahead=1, mission_budget_s=45.0),
    dict(plan_score_params={"c_cov_per_device": 0.15, "c_energy": 0.0,
                            "coverage_rank": "weighted"}),                # the pilot's sweep
])
def test_the_configs_the_plan_arms_fly_pass(kw):
    assert mule_config_errors(_plan_mule(**kw)) == []


def test_a_topology_whose_mule_cannot_fly_the_plan_is_refused():
    # No T_nom and the recorded 'reorder' fallback: both reported by validate().
    with pytest.raises(TopologyValidationError, match="set t_nom_s.*replan_fallback"):
        build_exp4_topology(n_devices=3, rf_range_m=60.0, n_missions=2, seed=5,
                            mission_clock="sim", ferry_settings={"contact_band": "wide"},
                            plan_mode="ferry", member_admission="subset")


# --------------------------------------------------------------------------- #
# The mule process
# --------------------------------------------------------------------------- #

class _Events:
    def __init__(self):
        self.lines = []

    def emit(self, event, **fields):
        json.dumps(fields)                   # as the JSONL emitter serialises
        self.lines.append((event, fields))

    def close(self):
        return

    def named(self, event):
        return [f for e, f in self.lines if e == event]


@pytest.fixture
def service(monkeypatch):
    """A ``MuleService`` factory (a real dock server, the bootstrap skipped) that
    records the keywords each supervisor was built with."""
    server = TCPDockLinkServer(host="127.0.0.1", port=0)
    server.start()
    made, built = [], []
    real = mule_process.MuleSupervisor

    class _Recorded(real):
        def __init__(self, **kwargs):
            built.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(mule_process, "MuleSupervisor", _Recorded)

    def _make(cfg: MuleConfig):
        events = _Events()
        cfg = dataclasses.replace(cfg, dock_port=server.port)
        svc = mule_process.MuleService(cfg, events=events)
        svc.supervisor.wait_for_initial_dock = lambda timeout=None: True
        made.append(svc)
        return svc, events, built[-1]

    yield _make
    for svc in made:
        svc.shutdown()
    server.close()


NEW_SUPERVISOR_KWARGS = {"member_admission", "plan_mode", "plan_options", "t_nom_s"}


@pytest.mark.parametrize("cfg", [
    MuleConfig(mule_id="m-wall", rf_range_m=60.0),
    _sim_mule(mule_id="m-h1", contact_band="wide"),
    _sim_mule(mule_id="m-d4", contact_band="wide", contact_policy="fedex"),
    _sim_mule(mule_id="m-h1-whole", contact_band="narrow", member_admission="whole"),
], ids=["wall", "sim-H1", "sim-D4", "sim-H1-whole"])
def test_a_recorded_mule_is_built_with_the_recorded_arguments(service, cfg):
    """No Phase 4 keyword reaches its supervisor, and ``mule_ready`` gains no key."""
    svc, events, kwargs = service(cfg)
    assert not NEW_SUPERVISOR_KWARGS & set(kwargs)
    assert svc.supervisor.scheduler.plan_mode == "legacy"
    assert getattr(svc.supervisor, "_flight_slot", None) is None
    (ready,) = events.named("mule_ready")
    assert not set(PLAN_MULE_FIELDS) & set(ready)


def test_an_h_or_d_arms_subsets_reach_its_scheduler_and_its_mule_ready(service):
    svc, events, kwargs = service(_sim_mule(contact_band="narrow", contact_policy="max_aoi",
                                            member_admission="subset"))
    assert kwargs["member_admission"] == "subset"
    assert not {"plan_mode", "plan_options", "t_nom_s"} & set(kwargs)
    assert svc.supervisor.scheduler.member_admission == "subset"
    (ready,) = events.named("mule_ready")
    assert ready["member_admission"] == "subset" and "plan_mode" not in ready


@pytest.mark.parametrize("kw,slot", [
    (dict(), CommittedSlot),
    (dict(flight_slot="cross_heuristic", age_cap_lookahead=1), CrossHeuristic),
    (dict(band_class_policy="fixed:wide", member_admission="whole",
          plan_score_params={"c_cov_per_device": 0.0, "c_link": 0.0},
          plan_search_params={"exact_max_devices": 4}), CommittedSlot),
])
def test_a_plan_mule_hands_its_supervisor_the_configs_options(service, kw, slot):
    cfg = _plan_mule(**kw)
    svc, events, kwargs = service(cfg)
    options = PlanOptions.from_config(**{f: getattr(cfg, f) for f in PLAN_OPTION_FIELDS})
    assert kwargs["plan_mode"] == "ferry" and kwargs["plan_options"] == options
    assert kwargs["t_nom_s"] == 200.0 and kwargs["member_admission"] == cfg.member_admission
    sched = svc.supervisor.scheduler
    assert sched.plan_mode == "ferry" and sched.plan_setup.options == options
    assert sched.member_admission == cfg.member_admission
    assert sched.plan_setup.t_ref_s == 200.0 and sched.plan_setup.reference == "wide"
    assert isinstance(svc.supervisor._flight_slot, slot)
    # mule_ready states what the scheduler runs, the settings resolved.
    (ready,) = events.named("mule_ready")
    shown = {k: ready[k] for k in PLAN_MULE_FIELDS}
    assert shown == {"plan_mode": "ferry",
                     **json.loads(json.dumps(sched.plan_setup.options.describe())),
                     "member_admission": sched.member_admission}
    assert shown["plan_score_params"] == json.loads(json.dumps(options.score.as_dict()))
    assert ready["miss_priority"] is False and ready["t_nom_s"] == 200.0


def test_a_plan_mule_ready_adds_exactly_the_plan_fields(service):
    _svc, h1_events, _ = service(_sim_mule(contact_band="wide", t_nom_s=200.0,
                                           in_flight_response="replan",
                                           replan_fallback="trim"))
    _svc, f_events, _ = service(_plan_mule())
    (h1,), (f,) = h1_events.named("mule_ready"), f_events.named("mule_ready")
    assert set(f) == set(h1) | set(PLAN_MULE_FIELDS)


def _result(clock, rnd, **kw) -> MissionRunResult:
    start = clock()
    return MissionRunResult(
        mission_round=rnd, empty=True, sim_start_s=start, sim_end_s=start + 30.0,
        sim_ledger={"turnaround": 30.0}, pass_1_flown=[], pass_2_flown=[], replans=[],
        aborts=[], inserts=[], offers_refused=[], energy_j=100.0, band="narrow",
        pass_1_preflight_drops=[], **kw)


def test_mission_completed_carries_the_plan_fields_only_when_a_mission_has_them(service):
    """Critic D2 and B12: ``plan`` and ``plan_wall_s`` in plan mode, a
    baseline's ``pass_1_policy_drops`` only when it left something out (an
    empty list is left out too); each after the Phase 3 fields and before
    ``energy_status``; a mission without them keeps the Phase 3 key set."""
    svc, events = service(_sim_mule(contact_band="wide", n_missions=4))[:2]
    clock = svc.supervisor.mission_clock
    plan = {"band": "narrow", "score": {"v": np.float64(-1.5)}, "served": ("a", "b")}
    drops = [{"position": [1.0, 2.0, 0.0], "devices": ["c"], "deadline_ts": 1.0e6 + 5.0,
              "reason": "budget", "widened": False}]
    results = iter([
        _result(clock, 1),
        _result(clock, 2, plan=plan, plan_wall_s=np.float64(0.012)),
        _result(clock, 3, pass_1_policy_drops=drops),
        _result(clock, 4, pass_1_policy_drops=[]),
    ])
    svc.supervisor.run_one_mission = lambda: next(results)
    svc.run()
    assert svc.exit_code == 0
    plain, planned, dropped, empty_drops = events.named("mission_completed")
    assert set(empty_drops) == set(plain)
    assert not {"plan", "plan_wall_s", "pass_1_policy_drops"} & set(plain)
    assert set(planned) == set(plain) | {"plan", "plan_wall_s"}
    assert list(planned)[-3:] == ["plan", "plan_wall_s", "energy_status"]
    assert planned["plan"] == {"band": "narrow", "score": {"v": -1.5}, "served": ["a", "b"]}
    assert type(planned["plan"]["score"]["v"]) is float and type(planned["plan_wall_s"]) is float
    assert set(dropped) == set(plain) | {"pass_1_policy_drops"}
    assert dropped["pass_1_policy_drops"] == drops


# --------------------------------------------------------------------------- #
# The topology builder
# --------------------------------------------------------------------------- #

PLAN_KW = dict(plan_mode="ferry", band_class_policy="fixed:wide", member_admission="subset",
               flight_slot="committed", age_cap_missions=3, age_cap_lookahead=1,
               plan_score_params={"c_energy": 0.0}, plan_search_params={"exact_max_devices": 5})
#: One value other than the recorded one for every plan field.
NOT_RECORDED = dict(plan_mode="ferry", band_class_policy="fixed:wide", member_admission="subset",
                    flight_slot="cross_heuristic", age_cap_missions=3, age_cap_lookahead=1,
                    plan_score_params={"c_energy": 0.0},
                    plan_search_params={"exact_max_devices": 5})


@pytest.mark.parametrize("k", [1, 2])
def test_the_builder_gives_every_mule_the_plan_fields_each_with_its_own_dicts(k):
    plan = copy.deepcopy(PLAN_KW)
    kw = dict(n_devices=6, rf_range_m=60.0, n_missions=2, seed=5, mission_clock="sim",
              ferry_settings={"contact_band": "wide", "t_nom_s": 200.0,
                              "in_flight_response": "replan", "replan_fallback": "trim"},
              **plan)
    if k > 1:
        kw.update(n_mules=2, min_participation=2, dock_on_empty=True)
    topo = build_exp4_topology(**kw)
    assert len(topo.mules) == k
    for mule in topo.mules:
        assert {f: getattr(mule, f) for f in PLAN_MULE_FIELDS} == PLAN_KW
        assert mule.plan_score_params is not plan["plan_score_params"]
        assert mule.plan_search_params is not plan["plan_search_params"]
    if k > 1:
        a, b = topo.mules
        assert a.plan_score_params is not b.plan_score_params
        assert a.plan_search_params is not b.plan_search_params
    plan["plan_score_params"]["c_energy"] = 9.0       # the caller's dict is not shared
    assert all(m.plan_score_params == {"c_energy": 0.0} for m in topo.mules)


def test_the_builders_defaults_are_the_recorded_mule():
    for clock in ("wall", "sim"):
        topo = build_exp4_topology(n_devices=3, rf_range_m=60.0, n_missions=1, seed=2,
                                   mission_clock=clock)
        assert {f: getattr(topo.mules[0], f) for f in PLAN_MULE_FIELDS} == PLAN_DEFAULTS


@pytest.mark.parametrize("field", PLAN_MULE_FIELDS)
def test_the_builder_refuses_plan_fields_off_the_simulated_clock(field):
    """Through the topology's own validation (``SIM_ONLY_MULE_FIELDS``)."""
    assert NOT_RECORDED[field] != PLAN_DEFAULTS[field]
    with pytest.raises(TopologyValidationError, match=f"{field}: only on the simulated"):
        build_exp4_topology(n_devices=2, rf_range_m=60.0, n_missions=1, seed=1,
                            **{field: copy.deepcopy(NOT_RECORDED[field])})


# --------------------------------------------------------------------------- #
# The driver: arms and labels
# --------------------------------------------------------------------------- #

def test_the_arm_lists():
    assert DEFAULT_ARMS == ("H0", "H1", "H2", "H3", "D1", "D2", "D3", "D4", "D5")
    assert PLAN_ARMS == ("F", "FX", "FB+wide", "FB+medium", "FB+narrow", "F-cov", "F-cap",
                         "F-prio")
    # and the Exp 5 addendum's F+L1 (tests/unit/test_exp5_f_l1.py)
    assert ARMS == DEFAULT_ARMS + PLAN_ARMS + PHASE_5_ARMS + ("F+L1",)
    assert len(set(ARMS)) == len(ARMS)


@pytest.mark.parametrize("arm", PLAN_ARMS)
def test_a_plan_arms_label_survives_its_trace_directory(arm):
    """ASCII, no "__" and none of <>:"/\\|?*: a kept trace's directory holds the
    label whole and the scorer parses it back (other choices 11)."""
    assert arm.isascii() and arm == arm.strip(" .")
    assert "__" not in arm and not set('<>:"/\\|?*') & set(arm)
    cell = _cell(arm, seed=2191267877, trial=3)
    name = trace_dir_name(cell)
    assert f"__{arm}__" in name
    key = parse_trial_dir(name)
    assert (key.arm, key.trial_index, key.seed) == (arm, 3, 2191267877)


# --------------------------------------------------------------------------- #
# The driver: each arm's settings
# --------------------------------------------------------------------------- #

def _f(**kw):
    out = dict(plan_mode="ferry", band_class_policy="search", member_admission="subset",
               flight_slot="committed", age_cap_missions=3, age_cap_lookahead=1,
               plan_score_params={"c_energy": 0.0},
               plan_search_params={"heuristic_max_evaluations": 500})
    out.update(kw)
    return out


EXPECTED_PLAN_SETTINGS = {
    "F": _f(),
    "FX": _f(flight_slot="cross_heuristic"),
    "FB+wide": _f(band_class_policy="fixed:wide"),
    "FB+medium": _f(band_class_policy="fixed:medium"),
    "FB+narrow": _f(band_class_policy="fixed:narrow"),
    "F-cov": _f(plan_score_params={"c_energy": 0.0, "c_cov_per_device": 0.0, "c_link": 0.0}),
    "F-cap": _f(age_cap_missions=None, age_cap_lookahead=0),
    "F-prio": _f(),
}


@pytest.mark.parametrize("arm", PLAN_ARMS)
def test_each_plan_arm_gets_fs_fields_with_its_own_change(arm):
    drv = _driver(age_cap_missions=3, age_cap_lookahead=1, plan_score_params={"c_energy": 0.0},
                  plan_search_params={"heuristic_max_evaluations": 500})
    assert drv.plan_settings(arm) == EXPECTED_PLAN_SETTINGS[arm]
    # The driver's own settings are not changed by an arm's (F-cov's merge).
    assert drv.plan_score_params == {"c_energy": 0.0}


@pytest.mark.parametrize("setting,expected", [
    (None, {"plan": "subset", "H1": "whole", "H2": "whole", "H3": "whole", "D1": "whole",
            "D2": "whole", "D3": "whole", "D4": "whole", "D5": "whole"}),
    ("subset", {"plan": "subset", "H1": "subset", "H2": "subset", "H3": "subset",
                "D1": "subset", "D2": "subset", "D3": "subset", "D4": "whole", "D5": "subset"}),
    ("whole", {"plan": "whole", "H1": "whole", "H2": "whole", "H3": "whole", "D1": "whole",
               "D2": "whole", "D3": "whole", "D4": "whole", "D5": "whole"}),
])
def test_each_arms_member_admission(setting, expected):
    """unit_U3b.md section 5.4: plan arms the setting else ``subset``; H1-H3,
    D1-D3 and D5 the setting else ``whole``; D4 always ``whole``. An H or D
    arm's topology gets the field only when it is ``subset``."""
    drv = _driver(member_admission=setting)
    for arm in PLAN_ARMS:
        assert drv.effective_member_admission(arm) == expected["plan"]
        assert drv.plan_settings(arm)["member_admission"] == expected["plan"]
    for arm in DEFAULT_ARMS[1:]:
        assert drv.effective_member_admission(arm) == expected[arm], arm
        assert drv.plan_settings(arm) == (
            {} if expected[arm] == "whole" else {"member_admission": "subset"})


@pytest.mark.parametrize("configured", [False, True])
def test_each_arm_runs_and_records_its_own_miss_priority(configured):
    """Decision 3: every plan arm weighs by (1 + miss streak) but F-prio; every
    other arm runs the configured value, as recorded. The row records what
    the mule ran (driver.py wrote the driver-global value before)."""
    drv = _driver(miss_priority=configured)
    for arm, expected in [("F", True), ("FX", True), ("FB+narrow", True), ("F-cov", True),
                          ("F-cap", True), ("F-prio", False), ("H1", configured),
                          ("D1", configured), ("D4", configured)]:
        assert drv.effective_miss_priority(arm) is expected, arm
        row, topo = T.run_stub_trial(drv, _cell(arm))
        assert row["miss_priority"] == int(expected)
        assert topo.mules[0].miss_priority is expected


@pytest.mark.parametrize("arm,band", [("F", "wide"), ("FX", "wide"), ("F-cov", "wide"),
                                      ("F-cap", "wide"), ("F-prio", "wide"),
                                      ("FB+wide", "wide"), ("FB+medium", "medium"),
                                      ("FB+narrow", "narrow")])
def test_a_plan_arms_band_and_fallback_travel_in_its_ferry_settings(arm, band):
    """Critic A8: FB+<class>'s band reaches the builder in the ferry settings
    (``build_exp4_topology`` has no band parameter); every plan arm re-plans
    with the trim (critic B11) whatever the configured fallback, while the H
    arms keep the configured one."""
    drv = _driver(replan_fallback="reorder")
    settings = drv.ferry_settings(arm=arm, regime="jittery")
    assert (settings["contact_band"], settings["replan_fallback"]) == (band, "trim")
    assert drv.arm_contact_band(arm) == band
    assert drv.ferry_settings(arm="H1", regime="jittery")["replan_fallback"] == "reorder"
    row, topo = T.run_stub_trial(drv, _cell(arm))
    (mule,) = topo.mules
    assert (mule.contact_band, mule.replan_fallback) == (band, "trim")
    assert json.loads(row["ferry_params"])["replan_fallback"] == "trim"


# --------------------------------------------------------------------------- #
# The driver: refusals, before anything is spawned
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", PLAN_ARMS)
def test_plan_arms_are_refused_on_the_wall_clock(arm):
    drv = Exp4Driver(realism=True)
    with pytest.raises(ValueError, match="simulated mission clock"):
        drv.check_arm(arm)
    T.FakeOrchestrator.last = None
    with pytest.raises(ValueError, match="simulated mission clock"):
        T.run_stub_trial(drv, _cell(arm))
    assert T.FakeOrchestrator.last is None               # nothing was built


@pytest.mark.parametrize("kw,arms,match", [
    (dict(pass_2_budget=True), PLAN_ARMS, "critic B8"),
    (dict(contact_band=None, contact_reliability_source="origin"),
     ("F", "FX", "F-cov", "F-cap", "F-prio"), "needs a contact_band"),
    (dict(in_flight_response="abort"),
     ("F", "FX", "FB+wide", "FB+medium", "FB+narrow", "F-cov", "F-prio"), "critic A10"),
    (dict(contact_band_classes=["wide", "narrow"]), ("FB+medium",), "medium"),
])
def test_plan_arms_are_refused_where_they_cannot_fly(kw, arms, match):
    drv = _driver(**kw)
    for arm in arms:
        with pytest.raises(ValueError, match=match):
            drv.check_arm(arm)
        T.FakeOrchestrator.last = None
        with pytest.raises(ValueError, match=match):
            T.run_stub_trial(drv, _cell(arm))
        assert T.FakeOrchestrator.last is None


def test_what_a_plan_arm_can_fly_without_passes():
    # FB+<class> brings its own band; F-cap has no cap, so abort is allowed.
    _driver(contact_band=None, contact_reliability_source="origin").check_arm("FB+narrow")
    _driver(in_flight_response="abort").check_arm("F-cap")
    _driver(contact_band_classes=["wide", "narrow"]).check_arm("FB+narrow")
    for arm in DEFAULT_ARMS:
        Exp4Driver().check_arm(arm)                       # nothing to check
    with pytest.raises(ValueError, match="unknown arm"):
        Exp4Driver().check_arm("F2")


@pytest.mark.parametrize("kw,match", [
    (dict(member_admission="subset"), "mission_clock='sim'"),
    (dict(age_cap_missions=2), "mission_clock='sim'"),
    (dict(age_cap_lookahead=1), "mission_clock='sim'"),
    (dict(plan_score_params={"c_energy": 0.0}), "mission_clock='sim'"),
    (dict(plan_search_params={"exact_max_devices": 4}), "mission_clock='sim'"),
    (dict(mission_clock="sim", member_admission="most"), "member_admission must be one of"),
    (dict(mission_clock="sim", age_cap_missions=0), "age_cap_missions must be None or an int"),
    (dict(mission_clock="sim", age_cap_missions=True), "age_cap_missions must be None or an int"),
    (dict(mission_clock="sim", age_cap_lookahead=-1), "age_cap_lookahead must be an int"),
    (dict(mission_clock="sim", plan_score_params={"kappa": 1}), "plan settings: .*unknown"),
    (dict(mission_clock="sim", plan_search_params={"passes": 1}), "plan settings: .*unknown"),
    (dict(mission_clock="sim", plan_score_params=[1]), "must be a mapping"),
])
def test_the_driver_refuses_plan_settings_that_no_arm_could_run(kw, match):
    with pytest.raises(ValueError, match=match):
        Exp4Driver(**kw)


def test_member_admission_whole_is_the_recorded_value_on_the_wall_clock():
    assert Exp4Driver(member_admission="whole").effective_member_admission("H1") == "whole"


# --------------------------------------------------------------------------- #
# The driver: T_nom
# --------------------------------------------------------------------------- #

def test_plan_arms_force_the_t_nom_computation(monkeypatch):
    """A plan arm always has T_nom (T in its score; decision 2 (b)): computed
    per cell when not given, even where no H arm setting needs one; the same
    value for every arm of the cell (priced on wide, critic C4); a given
    T_nom is used as given."""
    kw = dict(mission_clock="sim", realism=True, contact_band="wide", t_nom_layouts=4,
              in_flight_response="replan", replan_fallback="trim")
    drv = Exp4Driver(**kw)
    assert not drv._needs_t_nom("H1") and all(drv._needs_t_nom(a) for a in PLAN_ARMS)
    row, topo = T.run_stub_trial(drv, _cell("H1"))
    assert row["t_nom_s"] == "" and topo.mules[0].t_nom_s is None
    values = set()
    for arm in PLAN_ARMS:
        row, topo = T.run_stub_trial(drv, _cell(arm))
        values.add(topo.mules[0].t_nom_s)
        assert row["t_nom_s"] == topo.mules[0].t_nom_s
        assert json.loads(row["ferry_params"])["t_nom_computed"] is True
    (t_nom,) = values
    assert t_nom is not None and t_nom > 0.0
    # The H arms' own T_nom, where a setting needs one, is the same value.
    h1 = Exp4Driver(**dict(kw, deadline_time_scale="t_nom"))
    _row, topo = T.run_stub_trial(h1, _cell("H1"))
    assert topo.mules[0].t_nom_s == t_nom
    monkeypatch.setattr(Exp4Driver, "nominal_period_s",
                        lambda self, **k: pytest.fail("T_nom was given"))
    row, topo = T.run_stub_trial(Exp4Driver(**dict(kw, t_nom_s=180.0)), _cell("F"))
    assert topo.mules[0].t_nom_s == 180.0 == row["t_nom_s"]
    assert json.loads(row["ferry_params"])["t_nom_computed"] is False


# --------------------------------------------------------------------------- #
# The driver: provenance
# --------------------------------------------------------------------------- #

def test_the_trial_csv_header_is_unchanged():
    assert PROVENANCE_COLUMNS == PROVENANCE_6E6F92D
    columns = set(Exp4MetricSummary.csv_columns()) | set(PROVENANCE_COLUMNS)
    drv = _driver()
    for arm in PLAN_ARMS + ("H1", "D4"):
        row, _topo = T.run_stub_trial(drv, _cell(arm))
        assert set(row) == columns, arm


@pytest.mark.parametrize("arm", PLAN_ARMS)
def test_a_plan_rows_provenance(arm):
    """``contact_band`` reads ``search`` for a search arm and the pinned class
    for FB+<class>; ``ferry_params`` shows every plan field as the mule runs
    it, and nothing else changes format."""
    drv = _driver(plan_score_params={"c_cov_per_device": 0.25})
    row, topo = T.run_stub_trial(drv, _cell(arm))
    (mule,) = topo.mules
    expected_band = arm[len("FB+"):] if arm.startswith("FB+") else "search"
    assert row["contact_band"] == expected_band
    params = json.loads(row["ferry_params"])
    assert {f: params[f] for f in PLAN_MULE_FIELDS} == {f: getattr(mule, f) for f in
                                                        PLAN_MULE_FIELDS}
    assert params["plan_mode"] == "ferry"
    if arm == "F-cov":
        assert params["plan_score_params"] == {"c_cov_per_device": 0.0, "c_link": 0.0}
    else:
        assert params["plan_score_params"] == {"c_cov_per_device": 0.25}
    assert row["mission_clock"] == "sim" and row["in_flight_response"] == "replan"
    assert "contact_band" not in params and "device_availability" not in params


def test_every_mule_of_a_plan_trial_plans_its_own_slice_with_the_arms_settings():
    """At K > 1 plan mode runs per slice (critic B10): every mule carries the
    arm's plan fields, each with its own dicts, and T_nom is the slowest
    slice's, as for every other arm of the cell."""
    drv = _driver(n_mules=2, min_participation=2, plan_score_params={"c_energy": 0.0})
    row, topo = T.run_stub_trial(drv, _cell("FX", N=8))
    a, b = topo.mules
    for mule in (a, b):
        assert {f: getattr(mule, f) for f in PLAN_MULE_FIELDS} == dict(
            drv.plan_settings("FX"))
        assert mule.t_nom_s == row["t_nom_s"]
    assert a.plan_score_params == b.plan_score_params == {"c_energy": 0.0}
    assert a.plan_score_params is not b.plan_score_params
    _row, h1 = T.run_stub_trial(drv, _cell("H1", N=8))
    assert h1.mules[0].t_nom_s == row["t_nom_s"]


@pytest.mark.parametrize("arm", ["H1", "H3", "D1", "D3", "D4", "D5"])
def test_an_h_or_d_row_shows_member_admission_only_when_it_runs_subsets(arm):
    """Phase 3's strings at ``whole``; ``member_admission`` in ``ferry_params``
    under ``subset`` (unit_U3b.md 5.4), never for D4; ``contact_band`` the
    class the mule flies."""
    whole, _t = T.run_stub_trial(_driver(), _cell(arm))
    subset, topo = T.run_stub_trial(_driver(member_admission="subset"), _cell(arm))
    base = json.loads(whole["ferry_params"])
    assert not set(PLAN_MULE_FIELDS) & set(base)
    assert whole["contact_band"] == subset["contact_band"] == "wide"
    shown = json.loads(subset["ferry_params"])
    if arm == "D4":
        assert subset["ferry_params"] == whole["ferry_params"]
        assert topo.mules[0].member_admission == "whole"
    else:
        assert shown == {**base, "member_admission": "subset"}
        assert topo.mules[0].member_admission == "subset"


def test_the_shared_provenance_helpers_read_a_config_or_a_kept_trace():
    """``plan_ferry_params`` and ``contact_band_column`` take a mule config as a
    mapping (the scorer's JSON, possibly from before Phase 4): a missing key
    is the recorded default."""
    old = {k: v for k, v in dataclasses.asdict(_sim_mule(contact_band="medium")).items()
           if k not in PLAN_MULE_FIELDS}
    assert plan_ferry_params(old) == {} and contact_band_column(old) == "medium"
    assert contact_band_column({}) == "" and plan_ferry_params({}) == {}
    assert plan_ferry_params({"member_admission": "subset"}) == {"member_admission": "subset"}
    f = dataclasses.asdict(_plan_mule(plan_score_params={"c_time": 2.0}))
    assert plan_ferry_params(f) == {k: f[k] for k in PLAN_MULE_FIELDS}
    assert contact_band_column(f) == "search"
    fb = dataclasses.asdict(_plan_mule(contact_band="narrow", band_class_policy="fixed:narrow"))
    assert contact_band_column(fb) == "narrow"
    # A plan config missing the options (hand-edited) still reads its defaults.
    assert plan_ferry_params({"plan_mode": "ferry"}) == dict(PLAN_DEFAULTS, plan_mode="ferry")
    assert contact_band_column({"plan_mode": "ferry", "contact_band": "wide"}) == "search"


# --------------------------------------------------------------------------- #
# Rule 1 at the defaults: the Phase 3 simulated-clock oracles
# --------------------------------------------------------------------------- #

EVENT_PARTS = ("row", "mule_events", "mule_ready", "mission_started", "cluster_events",
               "device_events")


@pytest.mark.parametrize("name", P3.TRIAL_NAMES)
def test_at_the_defaults_no_trace_gains_a_plan_field(name):
    """UG4's hand-off: the goldens let an added key pass, so absence is pinned
    here, on all eight 6e6f92d oracles (H1 and D4 among them): no event and
    no row gains a key, but a D arm's ``pass_1_policy_drops`` (decision 6's
    one field at the defaults) in the missions where it left something out;
    the per-role JSON gains exactly the plan fields, at their defaults, and
    Phase 5's six checkpoint keys; the rows' Phase 3 strings are the recorded ones."""
    golden = P3.load_golden()["cases"][name]
    case = P3.capture(name)
    for part in EVENT_PARTS:
        assert P3.added_keys(golden[part], case[part]) == [], (name, part)
    for field in ("ferry_params", "contact_band", "miss_priority", "t_nom_s"):
        assert case["row"][field] == golden["row"][field], field
    added = P3.added_keys(golden["mission_completed"], case["mission_completed"])
    missions = case["mission_completed"]["exp4-mule"]
    with_drops = [i for i, m in enumerate(missions) if "pass_1_policy_drops" in m]
    assert added == [f"$.exp4-mule[{i}].pass_1_policy_drops" for i in with_drops]
    if name in ("d1_max_aoi", "d3_whittle"):
        assert with_drops                           # the fixture's D arms drop stops
    else:
        assert not with_drops
    for i in with_drops:
        drops = missions[i]["pass_1_policy_drops"]
        assert drops and all(d["widened"] is False for d in drops)
    configs = P3.added_keys(golden["configs"], case["configs"])
    # and Phase 5's six, and the Exp 5 addendum's fields
    new_keys = PLAN_MULE_FIELDS + C.CHECKPOINT_MULE_FIELDS + C.ADDENDUM_MULE_FIELDS
    assert sorted(configs) == sorted(f"$.mule-exp4-mule.json.{f}" for f in new_keys)
    mule_json = json.loads(json.dumps(case["configs"]["mule-exp4-mule.json"], default=str))
    assert {f: mule_json[f] for f in PLAN_MULE_FIELDS} == PLAN_DEFAULTS
    defaults = MuleConfig(mule_id="m")
    assert {f: mule_json[f] for f in C.ADDENDUM_MULE_FIELDS} == {
        f: getattr(defaults, f) for f in C.ADDENDUM_MULE_FIELDS}


def test_a_recorded_path_loads_no_plan_module():
    """A fresh interpreter builds wall and simulated H1 and D4 trials through
    the driver (nothing spawned) and their mules' processes, and ends with no
    ``hermes.scheduler.plan`` module loaded."""
    code = r"""
import sys
from experiments.exp4 import runner_main  # noqa: F401
from experiments.exp4.driver import Exp4Driver
from experiments.runner import Cell
from hermes.processes import mule as mule_process
from hermes.processes.config import MuleConfig
from hermes.transport import TCPDockLinkServer
from tests.golden import _build_topology as T

p = {"N": 4, "rrf": 60.0, "n_missions": 2, "regime": "clean"}
cid = "|".join(f"{k}={v}" for k, v in sorted(p.items()))
for kw in (dict(), dict(mission_clock="sim", contact_band="wide", realism=True,
                        in_flight_response="replan", deadline_time_scale="t_nom",
                        t_nom_layouts=2)):
    for arm in ("H1", "D4"):
        row, topo = T.run_stub_trial(Exp4Driver(**kw), Cell(cid, arm, 0, 7, p))
server = TCPDockLinkServer(host="127.0.0.1", port=0)
server.start()
for kw in (dict(), dict(mission_clock="sim", trial_seed=1, n_missions=1, contact_band="wide"),
           dict(mission_clock="sim", trial_seed=1, n_missions=1, contact_band="wide",
                contact_policy="fedex")):
    svc = mule_process.MuleService(MuleConfig(mule_id="m", rf_range_m=60.0,
                                              dock_port=server.port, **kw))
    svc.shutdown()
server.close()
print(sorted(m for m in sys.modules if m.startswith("hermes.scheduler.plan")
             or m.endswith("s3d_age_cap") or m.endswith("cross_heuristic")))
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                         text=True, timeout=240)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"


# --------------------------------------------------------------------------- #
# Every plan arm in process, end to end
# --------------------------------------------------------------------------- #

def _keys(value):
    """Every mapping key in ``value``, at any depth."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from _keys(v)


def _in_process(settings, cell):
    """(row, events by name, orchestrator) of one trial run through the real
    services in this process (UG4's harness, ``tests/golden/_build_p3_sim.py``)."""
    driver = Exp4Driver(**settings)
    with P3.in_process_orchestrator():
        P3.InProcessOrchestrator.last = None
        row = dict(driver.run_trial(cell))
        orch = P3.InProcessOrchestrator.last
    assert orch.exit_codes == {"mule-exp4-mule": 0}
    events = [json.loads(line) for f, text in sorted(orch.files.items())
              if f.startswith("mule-") and f.endswith(".jsonl")
              for line in text.splitlines() if line.strip()]
    by_name = {}
    for e in events:
        by_name.setdefault(e["event"], []).append(e)
    return row, by_name, orch


@pytest.fixture(scope="module")
def h1_pilot_events():
    """H1 on the same settings and cell: the key sets a plan arm adds to."""
    return _in_process(PLAN_PILOT, P3._cell("H1", P3.SEED, n_missions=3))[1]


@pytest.mark.parametrize("arm", PLAN_ARMS)
def test_every_plan_arm_flies_end_to_end(arm, h1_pilot_events):
    """Each plan arm through the driver and the real mule, cluster and device
    services: the trial ends ok; ``mule_ready`` adds exactly the plan fields
    and ``mission_completed`` exactly ``plan`` and ``plan_wall_s`` to H1's
    key sets on the same cell; each mission's plan is its closed commit on
    the arm's policy (FB+<class> flies only its class, in both passes); the
    row's provenance says what the mule ran."""
    row, events, orch = _in_process(PLAN_PILOT, P3._cell(arm, P3.SEED, n_missions=3))
    assert (row["missions_completed"], row["mission_failures"]) == (3, 0)
    (ready,) = events["mule_ready"]
    (h1_ready,) = h1_pilot_events["mule_ready"]
    assert set(ready) == set(h1_ready) | set(PLAN_MULE_FIELDS)
    mule = json.loads(orch.files["mule-exp4-mule.json"])
    assert {f: ready[f] for f in ("plan_mode", "band_class_policy", "flight_slot",
                                  "member_admission", "age_cap_missions",
                                  "age_cap_lookahead")} == {
        f: mule[f] for f in ("plan_mode", "band_class_policy", "flight_slot",
                             "member_admission", "age_cap_missions", "age_cap_lookahead")}
    assert ready["miss_priority"] is (arm != "F-prio")
    done = events["mission_completed"]
    h1_keys = set(h1_pilot_events["mission_completed"][0]) - {"pass_1_policy_drops"}
    policy = mule["band_class_policy"]
    for e in done:
        assert set(e) == h1_keys | {"plan", "plan_wall_s"}
        plan = e["plan"]
        assert plan["band_class_policy"] == policy and plan["band"] == e["band"]
        assert plan["visited"] is not None                     # closed
        assert isinstance(e["plan_wall_s"], float) and e["plan_wall_s"] >= 0.0
        assert not [k for k in _keys(plan) if "wall" in k]      # critic B12
        if arm.startswith("FB+"):
            c = arm[len("FB+"):]
            assert e["band"] == c
            assert {s["band"] for s in e["pass_1_flown"] + e["pass_2_flown"]} <= {c}
        if arm == "F-cov":
            # Decision 3, "cap-only service": with the coverage term off only
            # the cap brings a device into the plan.
            assert set(plan["served"]) <= set(plan["cap"]["capped"])
        if arm == "F-cap":
            assert (plan["cap"]["s"], plan["cap"]["capped"]) == (None, [])
        # Each stop records the class it was flown on: b-bar, but for FX's
        # Pass-1 switches on arrival (decision 5); Pass 2 always flies b-bar.
        assert {s["band"] for s in e["pass_2_flown"]} <= {e["band"]}
        pass_1 = {s["band"] for s in e["pass_1_flown"]}
        if arm == "FX":
            assert pass_1 <= {"wide", "medium", "narrow"}
        else:
            assert pass_1 <= {e["band"]}
    assert any(e["pass_1_flown"] for e in done)
    assert row["contact_band"] == (arm[3:] if arm.startswith("FB+") else "search")
    assert row["miss_priority"] == int(arm != "F-prio")
    assert {k: v for k, v in json.loads(row["ferry_params"]).items()
            if k in PLAN_MULE_FIELDS} == {f: mule[f] for f in PLAN_MULE_FIELDS}


def test_a_repeated_plan_trial_gives_the_same_trace_bar_the_plans_wall_time():
    """Critic B12: ``plan_wall_s`` is the only wall time a plan adds, and it is
    kept out of ``plan``, so a repeated trial's plan fields are equal."""
    cell = P3._cell("FX", P3.SEED, n_missions=3)

    def plans():
        _row, events, _orch = _in_process(PLAN_PILOT, cell)
        return [(e["plan"], e["pass_1_flown"], e["band"]) for e in events["mission_completed"]]

    assert plans() == plans()


CLIFF_SUBSET = dict(P3.CLIFF, member_admission="subset")


@pytest.mark.parametrize("arm", ["H1", "D1", "D5"])
def test_member_subsets_fly_the_cliff_layout_for_the_h_and_d_arms(arm):
    """unit_U3b.md 7.2 (U8): the Phase 3 cliff (``device_positions(8, 777,
    100.0)``, narrow, 1 MB, 60 s), which flies empty at ``whole``, flies part
    of the field-wide stop every mission under ``subset``. H1's complement is
    a pre-flight drop, widened; D1's and D5's are reported and never widened
    (decision 6)."""
    cell = P3._cell(arm, 777, N=8, n_missions=3, regime="clean")
    row, events, orch = _in_process(CLIFF_SUBSET, cell)
    assert (row["missions_completed"], row["mission_failures"]) == (3, 0)
    assert json.loads(row["ferry_params"])["member_admission"] == "subset"
    (ready,) = events["mule_ready"]
    assert ready["member_admission"] == "subset" and "plan_mode" not in ready
    for e in events["mission_completed"]:
        flown = [d for s in e["pass_1_flown"] for d in s["devices"]]
        assert 0 < len(flown) < 8
        if arm == "H1":
            assert "pass_1_policy_drops" not in e
            assert [d["reason"] for d in e["pass_1_preflight_drops"]] == ["budget"]
        else:
            drops = e["pass_1_policy_drops"]
            assert drops and all(d["widened"] is False for d in drops)
            assert sorted(flown + [d for x in drops for d in x["devices"]]) == [
                f"exp4-dev-{i:03d}" for i in range(8)]
    whole, whole_events, _o = _in_process(P3.CLIFF, cell)
    assert all(not e["pass_1_flown"] for e in whole_events["mission_completed"])


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #

def _runner(monkeypatch, argv):
    from experiments.exp4 import runner_main

    captured, grids = {}, []

    class _Driver(Exp4Driver):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    class _Runner:
        def __init__(self, *args, **kwargs):
            return

        def run(self, run_trial):
            return 0

    real = runner_main._build_grid
    monkeypatch.setattr(runner_main, "Exp4Driver", _Driver)
    monkeypatch.setattr(runner_main, "TrialRunner", _Runner)
    monkeypatch.setattr(runner_main, "_build_grid", lambda **kw: grids.append(kw) or real(**kw))
    assert runner_main.main(argv) == 0
    return captured, grids[0]


SIM_FLAGS = ["--mission-clock", "sim", "--contact-band", "wide", "--in-flight-response",
             "replan", "--replan-fallback", "trim"]


def test_the_runner_defaults_are_the_recorded_run(monkeypatch, tmp_path):
    kwargs, grid = _runner(monkeypatch, ["--csv", str(tmp_path / "t.csv")])
    assert (kwargs["member_admission"], kwargs["age_cap_missions"],
            kwargs["age_cap_lookahead"], kwargs["plan_score_params"],
            kwargs["plan_search_params"]) == (None, None, 0, {}, {})
    # The default arm list is the nine Phase 3 arms (H0 dropped on the stub).
    assert grid["arms"] == [a for a in DEFAULT_ARMS if a != "H0"]
    assert grid["base_seed"] == 42


def test_the_runner_passes_the_plan_flags(monkeypatch, tmp_path):
    kwargs, grid = _runner(monkeypatch, [
        "--csv", str(tmp_path / "t.csv"), *SIM_FLAGS, "--arms", "F", "FX", "FB+narrow",
        "F-cov", "H1", "D1", "--member-admission", "subset", "--age-cap-missions", "3",
        "--age-cap-lookahead", "1",
        "--plan-score-params", '{"c_cov_per_device": 0.15, "c_energy": 0, '
                               '"coverage_rank": "weighted"}',
        "--plan-search-params", '{"heuristic_max_evaluations": 500}',
        "--base-seed", "2027",
    ])
    assert kwargs["member_admission"] == "subset"
    assert (kwargs["age_cap_missions"], kwargs["age_cap_lookahead"]) == (3, 1)
    assert kwargs["plan_score_params"] == {"c_cov_per_device": 0.15, "c_energy": 0,
                                           "coverage_rank": "weighted"}
    assert kwargs["plan_search_params"] == {"heuristic_max_evaluations": 500}
    assert grid["arms"] == ["F", "FX", "FB+narrow", "F-cov", "H1", "D1"]
    assert grid["base_seed"] == 2027


@pytest.mark.parametrize("extra,why", [
    (["--arms", "F"], "the wall clock"),
    (["--arms", "FB+medium", *SIM_FLAGS, "--contact-band-classes", "wide", "narrow"],
     "a class the link lacks"),
    (["--arms", "F", *SIM_FLAGS, "--mission-budget-s", "60", "--pass-2-budget"], "B8"),
    (["--arms", "F", "--mission-clock", "sim", "--contact-band", "wide",
      "--age-cap-missions", "2"], "abort with a cap (A10)"),
    (["--plan-score-params", "{c_time: 1}", *SIM_FLAGS], "not JSON"),
    (["--plan-score-params", "[1, 2]", *SIM_FLAGS], "not an object"),
    (["--plan-search-params", '{"passes": 3}', *SIM_FLAGS], "an unknown key"),
    (["--member-admission", "subset"], "subsets on the wall clock"),
    (["--member-admission", "most", *SIM_FLAGS], "an unknown admission"),
])
def test_the_runner_refuses_a_plan_setting_it_cannot_run(monkeypatch, tmp_path, extra, why):
    with pytest.raises(SystemExit):
        _runner(monkeypatch, ["--csv", str(tmp_path / "t.csv"), *extra])


def test_the_base_seed_moves_every_seed_and_keeps_the_arms_paired():
    """Decision 7: a pilot takes its own ``--base-seed``, so its trials are not
    the headline's first ones; the arms of one trial still share its seed."""
    from experiments.exp4 import runner_main

    def seeds(base):
        grid = runner_main._build_grid(arms=["H1", "F", "FB+narrow"], Ns=[6], rrfs=[60.0],
                                       n_missions_values=[4], regimes=["jittery"],
                                       dead_zones=[0.6], link_qualities=[0.4], n_trials=3,
                                       base_seed=base)
        out = {}
        for cell in grid.cells():
            out.setdefault(cell.trial_index, set()).add(cell.seed)
        return out

    headline, pilot = seeds(42), seeds(2027)
    assert all(len(s) == 1 for s in headline.values()) and all(len(s) == 1 for s in
                                                               pilot.values())
    assert not set().union(*headline.values()) & set().union(*pilot.values())
