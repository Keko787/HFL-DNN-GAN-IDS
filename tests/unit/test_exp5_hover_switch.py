"""Exp 5 addendum, Study 5.14: the hover-stop switch ("hover stops off").

The hover rule (``hermes/scheduler/plan/hover.py``; the user's decision of
2026-09-30) was unconditional in plan mode. ``PlanSearchParams.hover_stops``
switches it, through ``--plan-search-params '{"hover_stops": false}'``.
Pinned:

* **At its default (True) nothing changes:** ``as_dict`` leaves it out, so a
  plan's ``mule_ready`` and every pinned dict keep their keys, and the row's
  ``ferry_params`` records the search settings as given (``{}``).
* **Off,** the search runs over S3a's stops alone: on UG5's f_45s trial, where
  the rule moves capped devices to the dock side, no Pass-1 stop is a hover
  stop; ``mule_ready`` and ``ferry_params`` show the switch.
* It takes a bool only.
"""

from __future__ import annotations

import json

import pytest

from experiments.exp4.driver import Exp4Driver
from hermes.scheduler.plan.types import PlanOptions, PlanSearchParams

from tests.golden import _build_p3_sim as UG4
from tests.golden import _build_p4_plan as UG5
from tests.golden.test_golden_p4_plan import on_the_dock_side, plain

OFF = {"hover_stops": False}


def test_the_switch_is_left_out_at_its_default():
    assert PlanSearchParams().hover_stops is True
    assert PlanSearchParams().as_dict() == {
        "exact_max_devices": 6, "exhaustive_max_stops": 6, "heuristic_max_passes": 50,
        "heuristic_max_evaluations": 2000}
    off = PlanSearchParams.from_mapping(OFF)
    assert off.hover_stops is False and off.as_dict() == dict(PlanSearchParams().as_dict(),
                                                              hover_stops=False)
    options = PlanOptions.from_config(plan_search_params=OFF)
    assert PlanOptions.from_config(**options.describe()) == options
    for bad in (0, 1, "false", None):
        with pytest.raises(TypeError, match="hover_stops"):
            PlanSearchParams(hover_stops=bad)


def _capture(settings, cell):
    driver = Exp4Driver(**settings)
    with UG4.in_process_orchestrator(), UG5.flight_slot_spy() as calls:
        UG4.InProcessOrchestrator.last = None
        row = dict(driver.run_trial(cell))
        orch = UG4.InProcessOrchestrator.last
    return plain(UG5.case_of(settings, cell, row, orch, calls))


def _hover_stops(case):
    positions = {cfg["device_id"]: cfg["position"] for name, cfg in case["configs"].items()
                 if name.startswith("device-")}
    (ready,) = case["mule_ready"].values()
    dk = ready[0]["dock"]
    (missions,) = case["mission_completed"].values()
    return [s for m in missions for s in m["pass_1_flown"]
            if len(s["devices"]) == 1
            and on_the_dock_side(s["position"], positions[s["devices"][0]], dk)]


@pytest.fixture(scope="module")
def f_45s():
    settings, cell = UG5.TRIALS["f_45s"]
    return (_capture(dict(settings), cell),
            _capture(dict(settings, plan_search_params=OFF), cell))


def test_off_the_search_flies_s3as_stops_alone(f_45s):
    on, off = f_45s
    assert _hover_stops(on)                      # the rule moves capped devices there
    assert _hover_stops(off) == []


def test_the_switch_shows_only_when_off(f_45s):
    on, off = f_45s
    (ready_on,), (ready_off,) = (list(c["mule_ready"].values())[0] for c in (on, off))
    assert "hover_stops" not in ready_on["plan_search_params"]
    assert ready_off["plan_search_params"]["hover_stops"] is False
    assert json.loads(on["row"]["ferry_params"])["plan_search_params"] == {}
    assert json.loads(off["row"]["ferry_params"])["plan_search_params"] == OFF
