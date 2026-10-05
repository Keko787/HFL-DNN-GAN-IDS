"""Exp 5 addendum, unit U10 (Study 5.4's sweep knobs).

Pinned:

* **The narrow class's reach** (``ContactLink.narrow_range_ratio``): None is the
  D1 derivation, every range and SNR as before; a ratio sets the narrow
  class's planar reach to that multiple of the anchor's, leaves wide and
  medium alone, and keeps the edge's definition (the mean SNR at the reach is
  the floor plus the margin, as at every class's edge). A ratio needs a narrow
  class that is not the anchor and must be positive.
* **Its plumbing**: ``FerrySpec.from_config`` forwards it; ``MuleConfig``
  carries it as a sim-only ferry field, None by default, left out of
  ``ferry_params`` at None and shown when set; the runner passes
  ``--narrow-range-ratio`` only when given.
* **The far share** (``device_positions(..., far_share=)``): None is the
  recorded draw, number for number; a share places exactly
  ``far_count(N, share)`` devices beyond ``far_radius_m`` of the dock (the trace
  scorer's ``far_devices``) and the rest within, the same for one seed; shares
  outside [0, 1] and a field with no far point are refused.
* **Its plumbing**: the driver needs realism, builds the trial's devices and
  T_nom's reference layouts with it, and keys T_nom by it; the S* tool's
  layouts take it; the runner passes ``--far-share`` only when given.
"""

from __future__ import annotations

import json
import logging
import math
from statistics import NormalDist

import pytest

from experiments.analysis.age_cap_s_star import reference_layouts
from experiments.analysis.traces_scorer import far_devices
from experiments.exp4 import runner_main
from experiments.exp4.driver import FERRY_PHYSICS_FIELDS, Exp4Driver
from experiments.exp4.topology_builder import (
    build_exp4_topology, device_positions, far_count,
)
from experiments.ferrysim import cells as C
from experiments.ferrysim.episode import Policy, run_episode
from hermes.l1.contact_link import ContactLink
from hermes.mule.ferry import FerrySpec
from hermes.processes.config import (
    ADDENDUM_MULE_FIELDS,
    FERRY_PARAMS_OMITTED_AT_NONE,
    FERRY_SPEC_FIELDS,
    SIM_ONLY_MULE_FIELDS,
    MuleConfig,
)


# --------------------------------------------------------------------------- #
# The narrow class's reach
# --------------------------------------------------------------------------- #

def test_none_is_the_derivation():
    link = ContactLink(anchor_planar_m=60.0)
    assert ContactLink(anchor_planar_m=60.0, narrow_range_ratio=None) == link
    # D1's ranges (configuration reference 17.1), unchanged.
    assert [round(link.range_planar_m(c), 1) for c in ("wide", "medium", "narrow")] == [
        60.0, 119.5, 232.2]


@pytest.mark.parametrize("ratio", [1.5, 2.0, 3.0, 3.87])
def test_a_ratio_sets_narrows_reach_and_keeps_its_edge(ratio):
    base = ContactLink(anchor_planar_m=60.0)
    link = ContactLink(anchor_planar_m=60.0, narrow_range_ratio=ratio)
    assert link.range_planar_m("narrow") == pytest.approx(ratio * 60.0, abs=1e-9)
    for other in ("wide", "medium"):
        assert link.range_planar_m(other) == base.range_planar_m(other)
        assert link.mean_snr_db(other, 50.0) == base.mean_snr_db(other, 50.0)
    edge = link.snr_floor_db + NormalDist().inv_cdf(link.margin_quantile) * link.shadow_sigma_db
    for cls in ("wide", "medium", "narrow"):
        assert link.mean_snr_db(cls, link.range_planar_m(cls)) == pytest.approx(edge, abs=1e-9)


def test_a_bad_ratio_is_refused():
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="narrow_range_ratio"):
            ContactLink(anchor_planar_m=60.0, narrow_range_ratio=bad)
    with pytest.raises(ValueError, match="narrow"):
        ContactLink(anchor_planar_m=60.0, classes=("wide", "medium"), narrow_range_ratio=2.0)


def test_the_spec_and_the_config_carry_it():
    kw = dict(rf_range_m=60.0, seed=7, contact_band="wide")
    assert FerrySpec.from_config(**kw).link == ContactLink(anchor_planar_m=60.0)
    assert FerrySpec.from_config(**kw, narrow_range_ratio=2.0).link.range_planar_m(
        "narrow") == pytest.approx(120.0)
    f = "narrow_range_ratio"
    assert MuleConfig(mule_id="m").narrow_range_ratio is None
    assert FERRY_SPEC_FIELDS[f] == f and f in SIM_ONLY_MULE_FIELDS and f in FERRY_PHYSICS_FIELDS
    assert f in ADDENDUM_MULE_FIELDS and f in FERRY_PARAMS_OMITTED_AT_NONE
    sim = MuleConfig(mule_id="m", mission_clock="sim", trial_seed=7, contact_band="wide",
                     narrow_range_ratio=2.0)
    assert FerrySpec.from_config(**sim.ferry_spec_kwargs()).link.range_planar_m(
        "narrow") == pytest.approx(120.0)


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


def test_the_row_shows_the_ratio_only_when_set(tmp_path):
    plain = json.loads(_episode(tmp_path / "plain", {}).row["ferry_params"])
    assert "narrow_range_ratio" not in plain
    set_ = json.loads(_episode(tmp_path / "set", {"narrow_range_ratio": 2.0}).row["ferry_params"])
    assert set_["narrow_range_ratio"] == 2.0


# --------------------------------------------------------------------------- #
# The far share
# --------------------------------------------------------------------------- #

def test_none_is_the_recorded_draw():
    for seed in (1, 7, 2026):
        assert device_positions(6, seed, 100.0) == device_positions(
            6, seed, 100.0, far_share=None, far_radius_m=60.0)


def test_far_count_rounds_half_up():
    assert [far_count(6, s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)] == [0, 2, 3, 5, 6]
    assert far_count(12, 0.25) == 3 and far_count(2, 0.25) == 1


@pytest.mark.parametrize("share", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("seed", [3, 11, 404])
def test_a_share_places_exactly_that_many_far(share, seed):
    xy = device_positions(6, seed, 100.0, far_share=share, far_radius_m=60.0)
    assert xy == device_positions(6, seed, 100.0, far_share=share, far_radius_m=60.0)
    assert all(abs(x) <= 100.0 and abs(y) <= 100.0 for x, y in xy)
    positions = {f"d{i}": (x, y, 0.0) for i, (x, y) in enumerate(xy)}
    assert len(far_devices(positions, 60.0)) == far_count(6, share)


def test_a_bad_share_or_field_is_refused():
    for bad in (-0.1, 1.5, math.nan):
        with pytest.raises(ValueError, match="far_share"):
            device_positions(6, 1, 100.0, far_share=bad, far_radius_m=60.0)
    with pytest.raises(ValueError, match="far_radius_m"):
        device_positions(6, 1, 100.0, far_share=0.5)
    with pytest.raises(ValueError, match="cannot be placed"):
        device_positions(6, 1, 40.0, far_share=0.5, far_radius_m=60.0)
    with pytest.raises(ValueError, match="far_share"):
        Exp4Driver(far_share=0.5)                       # no realism
    with pytest.raises(ValueError, match="far_share"):
        Exp4Driver(realism=True, far_share=2.0)


def test_the_trials_devices_and_t_nom_take_it():
    topo = build_exp4_topology(n_devices=6, rf_range_m=60.0, n_missions=1, seed=9,
                               field_radius_m=100.0, far_share=0.75)
    positions = {d.device_id: d.position for d in topo.devices}
    assert len(far_devices(positions, 60.0)) == far_count(6, 0.75)
    plain = build_exp4_topology(n_devices=6, rf_range_m=60.0, n_missions=1, seed=9,
                                field_radius_m=100.0)
    assert [d.position for d in plain.devices] == [
        (x, y, 0.0) for x, y in device_positions(6, 9, 100.0)]

    def t_nom(driver):
        settings = driver.ferry_settings(arm="F", regime="jittery")
        theta, synth = driver._payload_bytes(None)
        return driver.nominal_period_s(n_devices=6, rf_range_m=60.0, regime="jittery",
                                       settings=settings, theta_bytes=theta, synth_bytes=synth)

    kw = dict(mission_clock="sim", realism=True, contact_band="wide", t_nom_layouts=4)
    near, far = t_nom(Exp4Driver(**kw, far_share=0.0)), t_nom(Exp4Driver(**kw, far_share=1.0))
    assert far > near            # every reference device beyond wide's reach costs more flight


def test_the_s_star_layouts_take_it():
    plain = reference_layouts(6, count=3, spread_m=100.0)
    assert plain == reference_layouts(6, count=3, spread_m=100.0, far_share=None)
    shared = reference_layouts(6, count=3, spread_m=100.0, far_share=0.5, far_radius_m=60.0)
    for layout in shared:
        assert len(far_devices(dict(layout.positions), 60.0)) == far_count(6, 0.5)


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #

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
            "--contact-band", "wide", "--realism"]
    plain = _driver_kwargs(monkeypatch, base)
    assert "far_share" not in plain
    assert "narrow_range_ratio" not in (plain.get("ferry_physics") or {})
    set_ = _driver_kwargs(monkeypatch, base + ["--far-share", "0.5",
                                               "--narrow-range-ratio", "2"])
    assert set_["far_share"] == 0.5
    assert set_["ferry_physics"]["narrow_range_ratio"] == 2.0
