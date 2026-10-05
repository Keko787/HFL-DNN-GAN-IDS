"""Exp 5 addendum, Study 5.15: the contact channel's interference strength as settings.

``ContactChannel`` took the interference amplitude A and noise sigma_I, but
``FerrySpec.from_config`` never passed them, so only the two regimes' pairs
(clean 1 and 0.4 dB, jittery 5 and 1.5 dB) could fly. Pinned:

* **The spec** builds its contact channel with the given A and sigma_I, and
  None keeps the regime's own; the planner's link term (the outage) prices
  what flies; a negative value is refused.
* **The config** carries them (``MuleConfig.interference_amp_db``,
  ``interference_sigma_db``, None by default) into the spec, as ferry-spec
  fields that are simulated-clock only; the runner's ``--interference-amp-db``
  and ``--interference-sigma-db`` reach the driver's ``ferry_physics`` only
  when given.
* **The row**: ``ferry_params`` leaves each out at None, so every recorded
  string is unchanged, and shows it when set; the scorer's provenance agrees
  with the driver's either way.
"""

from __future__ import annotations

import json
import logging

import pytest

from experiments.analysis.traces_scorer import score_trial
from experiments.exp4 import runner_main
from experiments.exp4.driver import FERRY_PHYSICS_FIELDS, Exp4Driver
from experiments.ferrysim import cells as C
from experiments.ferrysim.episode import Policy, run_episode
from hermes.l1.channel_model import CONTACT_REGIMES
from hermes.mule.ferry import FerrySpec
from hermes.processes.config import (
    ADDENDUM_MULE_FIELDS,
    FERRY_PARAMS_OMITTED_AT_NONE,
    FERRY_SPEC_FIELDS,
    SIM_ONLY_MULE_FIELDS,
    MuleConfig,
    mule_config_errors,
)

FIELDS = ("interference_amp_db", "interference_sigma_db")


def _spec(**kw):
    return FerrySpec.from_config(rf_range_m=60.0, seed=7, contact_band="wide", **kw)


@pytest.mark.parametrize("regime", sorted(CONTACT_REGIMES))
def test_none_keeps_the_regimes_own_and_a_value_overrides_it(regime):
    amp, sigma = CONTACT_REGIMES[regime]
    chan = _spec(contact_regime=regime).contact_channel
    assert (chan.interference_amp_db, chan.interference_sigma_db) == (amp, sigma)
    chan = _spec(contact_regime=regime, interference_amp_db=8.0).contact_channel
    assert (chan.interference_amp_db, chan.interference_sigma_db) == (8.0, sigma)
    chan = _spec(contact_regime=regime, interference_sigma_db=0.0).contact_channel
    assert (chan.interference_amp_db, chan.interference_sigma_db) == (amp, 0.0)
    described = _spec(contact_regime=regime, interference_amp_db=8.0,
                      interference_sigma_db=3.0).contact_channel.describe()
    assert (described["interference_amp_db"], described["interference_sigma_db"]) == (8.0, 3.0)


def test_the_planner_prices_the_interference_that_flies():
    """The outage's sigma is sqrt(sigma_sh^2 + sigma_I^2 + A^2 / 2): a harsher
    channel raises the planned outage at the same distance."""
    from hermes.mule.ferry import FerryRuntime

    base = FerryRuntime(_spec(contact_regime="jittery"), None, rf_range_m=60.0)
    harsh = FerryRuntime(_spec(contact_regime="jittery", interference_amp_db=10.0,
                               interference_sigma_db=3.0), None, rf_range_m=60.0)
    d = 0.9 * base.spec.link.range_planar_m("wide")
    assert 0.0 < base.outage_probability(d) < harsh.outage_probability(d) < 1.0


def test_a_negative_strength_is_refused():
    for field in FIELDS:
        with pytest.raises(ValueError, match=field):
            _spec(contact_regime="jittery", **{field: -0.5})


def test_the_config_carries_them_as_sim_only_ferry_fields():
    # Unit U10's narrow_range_ratio follows them in the omitted-at-None list.
    assert ADDENDUM_MULE_FIELDS[:2] == FIELDS == FERRY_PARAMS_OMITTED_AT_NONE[:2]
    defaults = MuleConfig(mule_id="m")
    assert {f: getattr(defaults, f) for f in FIELDS} == {f: None for f in FIELDS}
    for f in FIELDS:
        assert FERRY_SPEC_FIELDS[f] == f and f in SIM_ONLY_MULE_FIELDS and f in FERRY_PHYSICS_FIELDS
    sim = MuleConfig(mule_id="m", mission_clock="sim", trial_seed=7, contact_band="wide",
                     contact_regime="jittery", interference_amp_db=8.0)
    kw = sim.ferry_spec_kwargs()
    assert (kw["interference_amp_db"], kw["interference_sigma_db"]) == (8.0, None)
    assert FerrySpec.from_config(**kw).contact_channel.interference_amp_db == 8.0
    wall = mule_config_errors(MuleConfig(mule_id="m", interference_amp_db=8.0))
    assert len(wall) == 1 and "interference_amp_db" in wall[0]


def _driver_kwargs(monkeypatch, argv):
    seen = {}

    class _Stop(Exception):
        pass

    def fake(**kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(runner_main, "Exp4Driver", fake)
    with pytest.raises(_Stop):
        runner_main.main(argv)
    return seen


def test_the_runner_passes_them_only_when_given(tmp_path, monkeypatch):
    base = ["--csv", str(tmp_path / "t.csv"), "--arms", "H1", "--mission-clock", "sim",
            "--contact-band", "wide", "--contact-regime", "jittery"]
    plain = _driver_kwargs(monkeypatch, base)
    assert plain["ferry_physics"] == {"contact_regime": "jittery"}
    set_ = _driver_kwargs(monkeypatch, base + ["--interference-amp-db", "8",
                                               "--interference-sigma-db", "2.5"])
    assert set_["ferry_physics"] == {"contact_regime": "jittery", "interference_amp_db": 8.0,
                                     "interference_sigma_db": 2.5}


def _episode(tmp_path, physics):
    cell = C.cell_named("jit-n6-90")
    settings = dict(cell.driver_settings()["ferry_physics"], **physics)
    logging.disable(logging.WARNING)
    try:
        return run_episode(cell, C.stream_seeds(C.VAL_STREAM, cell.name, 1)[0],
                           Policy.of_arm("FX"),
                           driver_overrides={"ferry_physics": settings,
                                             "trace_root": str(tmp_path)})
    finally:
        logging.disable(logging.NOTSET)


def test_the_row_shows_them_only_when_set_and_the_scorer_agrees(tmp_path):
    plain = _episode(tmp_path / "plain", {})
    params = json.loads(plain.row["ferry_params"])
    assert not set(FIELDS) & set(params) and params["contact_regime"] == "jittery"
    harsh = _episode(tmp_path / "harsh", {"interference_amp_db": 8.0})
    shown = json.loads(harsh.row["ferry_params"])
    assert shown == dict(params, interference_amp_db=8.0)
    for root, result, amp in ((tmp_path / "plain", plain, 5.0),
                              (tmp_path / "harsh", harsh, 8.0)):
        (trace,) = [d for d in root.rglob("*__FX__*") if d.is_dir()]
        assert score_trial(trace).to_row()["ferry_params"] == result.row["ferry_params"]
        # what the mule flies, as it announces it
        (ready,) = [json.loads(line) for line in (trace / "mule-exp4-mule.jsonl").read_text(
            encoding="utf-8").splitlines() if '"mule_ready"' in line]
        contact = ready["channel_params"]["contact"]
        assert (contact["interference_amp_db"], contact["interference_sigma_db"]) == (amp, 1.5)
