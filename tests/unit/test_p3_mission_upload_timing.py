"""FeRRy Phase 3, unit U7 — the upload is timed under the recorded backhaul model.

Decision on unit U6's open question 4: on the mission clock with the recorded
``backhaul_model="mission"`` the upload is still charged to the clock, at the
fixed carrier's noise-free mean backhaul SNR (``base + max_c g_c`` of the
seconds-axis model for the trial's seed and regime), timing only: the mule
reports no backhaul outcome, the cluster's recorded schedule still decides
the loss, and the planner prices exactly what the clock charges. A hand-built
spec without that SNR charges nothing, as before; the seconds model is
unchanged.
"""

from __future__ import annotations

import math

import pytest

from hermes.l1.channel_model import SALT_BACKHAUL, BackhaulChannel, ferry_salt
from hermes.l1.contact_link import ContactLink
from hermes.l1.mission_clock import SIM_EPOCH_S, MissionClock
from hermes.mule.ferry import FerryRuntime, FerrySpec
from hermes.types import DeviceID

from tests.golden import _mule_harness as GH
from tests.integration import _ferry_harness as H

RF = 60.0


def _timing_snr(seed: int, regime: str) -> float:
    ch = BackhaulChannel(salt=ferry_salt(seed, SALT_BACKHAUL), period_s=500.0, regime=regime)
    return ch.pred_snr_db(ch.fixed_band())


@pytest.mark.parametrize("regime", ["clean", "jittery"])
def test_from_config_times_the_mission_model_at_the_fixed_carriers_mean(regime):
    spec = FerrySpec.from_config(rf_range_m=RF, seed=13, backhaul_regime=regime)
    assert spec.backhaul is None
    assert spec.mission_upload_snr_db == _timing_snr(13, regime)
    base = 12.0 if regime == "clean" else 6.0
    assert base <= spec.mission_upload_snr_db <= base + 3.0
    assert spec.describe()["backhaul_timing_snr_db"] == spec.mission_upload_snr_db
    # The seconds model prices the upload itself: no timing SNR there.
    seconds = FerrySpec.from_config(rf_range_m=RF, seed=13, backhaul_model="seconds",
                                    backhaul_period=800.0)
    assert seconds.mission_upload_snr_db is None
    assert seconds.describe()["backhaul_timing_snr_db"] is None


@pytest.mark.parametrize("payload", [None, 1_000_000])
def test_the_charge_is_the_rate_at_that_snr_and_the_planner_prices_the_same(payload):
    spec = FerrySpec.from_config(rf_range_m=RF, seed=13, payload_bytes=payload)
    clock = MissionClock()
    rt = FerryRuntime(spec, clock, rf_range_m=RF)
    rt.set_payload(theta_bytes=18_756, synth_bytes=64)
    nbytes = 18_756 if payload is None else payload
    expected = 8.0 * nbytes / spec.link.rate_bps("wide", spec.mission_upload_snr_db)
    assert rt.predicted_upload_s() == expected
    assert rt.physics().upload_time_s() == expected
    assert rt.charge_upload(18_756) is None                 # no backhaul outcome to report
    assert clock() == SIM_EPOCH_S + expected
    assert clock.ledger()["upload"] == expected
    assert rt.charge_upload(0) is None and clock() == SIM_EPOCH_S + expected   # empty partial
    assert rt.rf_prior is None


def test_a_hand_built_spec_still_charges_nothing():
    clock = MissionClock()
    rt = FerryRuntime(FerrySpec(), clock, rf_range_m=RF)
    assert rt.charge_upload(10_000) is None and clock() == SIM_EPOCH_S
    assert rt.predicted_upload_s() == 0.0


@pytest.mark.parametrize("kw,match", [
    (dict(mission_upload_snr_db=float("nan")), "finite"),
    (dict(mission_upload_snr_db=True), "finite"),
    (dict(mission_upload_snr_db=10.0), "ContactLink"),
])
def test_the_timing_snr_is_validated(kw, match):
    with pytest.raises(ValueError, match=match):
        FerrySpec(**kw)


def test_the_timing_snr_is_refused_beside_a_seconds_backhaul():
    link = ContactLink(anchor_planar_m=RF)
    bh = BackhaulChannel(salt=1, period_s=100.0)
    with pytest.raises(ValueError, match="seconds-axis backhaul"):
        FerrySpec(link=link, backhaul=bh, mission_upload_snr_db=10.0)


def test_a_mission_on_the_clock_charges_the_upload_under_the_mission_model():
    layout = (("dev-a", (10.0, 0.0, 0.0)), ("dev-b", (0.0, 20.0, 0.0)))
    w = H.World(layout=layout, flaky={})
    spec = FerrySpec.from_config(rf_range_m=RF, seed=7)
    sup = w.supervisor("mule-g", sim=True, ferry=spec)
    with H.Patched(w.clock):
        w.bootstrap()
        result = sup.run_one_mission()
    assert not result.empty and result.backhaul is None
    theta_bytes = sum(int(a.nbytes) for a in result.aggregate.weights)
    expected = 8.0 * theta_bytes / spec.link.rate_bps("wide", spec.mission_upload_snr_db)
    assert result.sim_ledger["upload"] == pytest.approx(expected, rel=0, abs=1e-12)
    assert math.isclose(sum(result.sim_ledger.values()), result.sim_end_s - result.sim_start_s,
                        abs_tol=1e-6)
    (up,) = w.server.ups
    assert up.backhaul is None
    # The UP's time is the upload's completion; the 30 s turnaround follows,
    # and Pass 2 takes off from there (no DOWN echo here, so no sync).
    assert up.sim_upload_ts + 30.0 == pytest.approx(result.sim_pass_2_start_s, abs=1e-9)
    landing = result.pass_1_flown[-1]["end_s"]
    assert up.sim_upload_ts > landing + expected - 1e-9
    assert DeviceID("dev-a") in {line.device_id for line in result.report.lines}
