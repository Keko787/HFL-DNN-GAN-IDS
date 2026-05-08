"""EX-3.4 — exp3 metrics tests."""

from __future__ import annotations

import math

import pytest

from experiments.calibration import Exp3Calibration
from experiments.exp3.metrics import (
    Exp3MetricSummary,
    Exp3RoundLog,
    aggregate_round_logs,
    completion_fairness,
    coverage,
    jains_fairness,
    mission_completion_rate,
    participation_entropy,
    propulsion_energy,
    rho_contact,
    round_close_rate,
    summarise_trial,
    update_yield,
)
from experiments.exp3.sim_env import Exp3EpisodeMetrics


# --------------------------------------------------------------------------- #
# Federation-side metrics
# --------------------------------------------------------------------------- #

def test_update_yield_mean_of_n_updates():
    rounds = [
        Exp3RoundLog(0, 3, 5, True),
        Exp3RoundLog(1, 5, 5, True),
        Exp3RoundLog(2, 1, 5, True),
    ]
    assert update_yield(rounds) == pytest.approx(3.0)


def test_aggregate_round_logs_emits_quorum_threshold():
    """``aggregate_round_logs`` must include kmin=2 (the FL quorum
    threshold) alongside the legacy 1, N/2, N thresholds.
    """
    rounds = [
        Exp3RoundLog(0, 0, 10, True),  # below quorum
        Exp3RoundLog(1, 1, 10, True),  # below quorum
        Exp3RoundLog(2, 2, 10, True),  # AT quorum
        Exp3RoundLog(3, 5, 10, True),  # above quorum, at majority
        Exp3RoundLog(4, 10, 10, True),  # full slice
    ]
    _, close_rates = aggregate_round_logs(rounds)
    # 3 of 5 rounds had n_updates >= 2 (rounds 2, 3, 4) → 0.6
    assert close_rates[2] == pytest.approx(3 / 5)
    # legacy thresholds still computed
    assert 1 in close_rates and 5 in close_rates and 10 in close_rates


def test_summarise_trial_populates_kmin2():
    rounds = [
        Exp3RoundLog(0, 3, 10, True),
        Exp3RoundLog(1, 1, 10, True),
        Exp3RoundLog(2, 0, 10, True),
    ]
    s = summarise_trial(
        rounds=rounds, metrics=None, cal=None,
        n_devices=10, is_mule_arm=False,
    )
    # Only round 0 (n_updates=3) hits the FL quorum (≥2).
    assert s.round_close_rate_kmin2 == pytest.approx(1 / 3)


def test_round_close_rate_kmin_threshold():
    rounds = [
        Exp3RoundLog(0, 3, 5, True),  # ≥1, ≥2, not ≥5
        Exp3RoundLog(1, 5, 5, True),  # all
        Exp3RoundLog(2, 0, 5, True),  # none
    ]
    assert round_close_rate(rounds, kmin=1) == pytest.approx(2 / 3)
    assert round_close_rate(rounds, kmin=2) == pytest.approx(2 / 3)
    assert round_close_rate(rounds, kmin=5) == pytest.approx(1 / 3)


def test_round_close_rate_respects_deadline_met():
    rounds = [
        Exp3RoundLog(0, 5, 5, False),  # missed deadline ⇒ not counted
        Exp3RoundLog(1, 5, 5, True),
    ]
    assert round_close_rate(rounds, kmin=1) == pytest.approx(0.5)


def test_round_close_rate_invalid_kmin():
    with pytest.raises(ValueError):
        round_close_rate([], kmin=0)


def test_aggregate_round_logs_picks_n_target_max():
    rounds = [
        Exp3RoundLog(0, 1, 4, True),
        Exp3RoundLog(1, 2, 6, True),  # n_target=6 ⇒ kmin=⌈6/2⌉=3, full=6
    ]
    yld, by_k = aggregate_round_logs(rounds)
    assert yld == pytest.approx(1.5)
    assert 1 in by_k and 3 in by_k and 6 in by_k


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #

def test_coverage_counts_visited_devices():
    visits = {"a": 1, "b": 0, "c": 2, "d": 0}
    assert coverage(visits) == pytest.approx(0.5)


def test_coverage_uses_explicit_scheduled_count():
    visits = {"a": 1}
    assert coverage(visits, scheduled_count=10) == pytest.approx(0.1)


def test_coverage_empty_returns_zero():
    assert coverage({}) == 0.0


# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #

def test_jains_fairness_perfectly_equal_is_one():
    visits = {f"d{i}": 5 for i in range(4)}
    assert jains_fairness(visits) == pytest.approx(1.0)


def test_jains_fairness_one_hog_is_one_over_n():
    # One device has all the visits; J = (k)² / (4 · k²) = 1/4.
    visits = {"a": 10, "b": 0, "c": 0, "d": 0}
    assert jains_fairness(visits) == pytest.approx(0.25)


def test_jains_fairness_handles_empty_input():
    assert jains_fairness({}) == 1.0


def test_jains_fairness_all_zero_returns_one():
    assert jains_fairness({"a": 0, "b": 0}) == 1.0


def test_participation_entropy_uniform_equals_log2_n():
    visits = {f"d{i}": 1 for i in range(4)}
    assert participation_entropy(visits) == pytest.approx(2.0)


def test_participation_entropy_one_hog_is_zero():
    visits = {"a": 5, "b": 0, "c": 0}
    assert participation_entropy(visits) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# ρ_contact
# --------------------------------------------------------------------------- #

def test_rho_contact_is_devices_per_visited_contact():
    m = Exp3EpisodeMetrics(contacts_visited=3, devices_visited=9)
    assert rho_contact(m) == pytest.approx(3.0)


def test_rho_contact_zero_when_nothing_visited():
    m = Exp3EpisodeMetrics(contacts_visited=0, devices_visited=0)
    assert rho_contact(m) == 0.0


# --------------------------------------------------------------------------- #
# Propulsion energy via Eq. 5
# --------------------------------------------------------------------------- #

def _fake_cal() -> Exp3Calibration:
    return Exp3Calibration(
        P_idle_W=2.0,
        epsilon_bit_J_per_bit=1.0e-9,
        epsilon_prop_J_per_m=10.0,
        mule_cruise_speed_m_s=5.0,
    )


def test_propulsion_energy_combines_three_components():
    # Real flight path = Pass-1 transits (10m) + dock leg (3m) + Pass-2
    # walk (2m) = 15m. Mission time = Pass-1 (2+1+10) + dock (0.5+1) +
    # Pass-2 (0.4+0.6) = 15.5s. The legacy ``return_distance_m`` is
    # intentionally not part of ``path_length_m`` anymore — see
    # docstring on Exp3EpisodeMetrics.
    m = Exp3EpisodeMetrics(
        transit_distance_m=10.0,
        return_distance_m=5.0,           # legacy field, no longer counted
        upload_bytes=1_000_000.0,
        transit_time_s=2.0,
        upload_time_s=1.0,
        collect_time_s=10.0,
        dock_transit_distance_m=3.0,
        dock_transit_time_s=0.5,
        dock_time_s=1.0,
        pass2_transit_distance_m=2.0,
        pass2_transit_time_s=0.4,
        pass2_delivery_time_s=0.6,
    )
    cal = _fake_cal()
    e = propulsion_energy(m, cal)
    # idle: 15.5s × 2W = 31 J
    assert e.idle_J == pytest.approx(31.0)
    # tx: 1e6 bytes × 8 × 1e-9 J/bit = 0.008 J
    assert e.tx_J == pytest.approx(0.008)
    # prop: 15m × 10 J/m = 150 J  (10 + 3 + 2, NOT counting return_distance_m)
    assert e.prop_J == pytest.approx(150.0)
    assert e.total_J == pytest.approx(181.008)


# --------------------------------------------------------------------------- #
# summarise_trial — A1 (no mule) vs mule arms
# --------------------------------------------------------------------------- #

def test_summarise_trial_a1_emits_none_for_mule_fields():
    rounds = [Exp3RoundLog(0, 3, 5, True)]
    s = summarise_trial(
        rounds=rounds, metrics=None, cal=None,
        n_devices=5, is_mule_arm=False,
    )
    assert s.rho_contact is None
    assert s.pass2_coverage is None
    assert s.propulsion_energy_J is None
    assert s.update_yield == pytest.approx(3.0)


def test_summarise_trial_mule_requires_metrics():
    with pytest.raises(ValueError):
        summarise_trial(
            rounds=[], metrics=None, cal=None,
            n_devices=5, is_mule_arm=True,
        )


def test_summarise_trial_mule_emits_propulsion_when_cal_supplied():
    m = Exp3EpisodeMetrics(
        contacts_visited=2, devices_visited=4, devices_completed=3,
        transit_distance_m=10.0, return_distance_m=5.0,
        upload_bytes=1000.0, transit_time_s=1.0,
        upload_time_s=0.5, collect_time_s=2.0,
        per_device_visits={f"d{i}": 1 for i in range(4)},
    )
    rounds = [
        Exp3RoundLog(0, 2, 4, True),
        Exp3RoundLog(1, 1, 4, True),
    ]
    s = summarise_trial(
        rounds=rounds, metrics=m, cal=_fake_cal(),
        n_devices=4, is_mule_arm=True,
    )
    assert s.rho_contact == pytest.approx(2.0)
    assert s.propulsion_energy_J is not None
    assert s.propulsion_energy_J > 0.0
    assert s.coverage == pytest.approx(1.0)


def test_to_row_writes_blank_for_none_values():
    s = Exp3MetricSummary(
        update_yield=1.0, coverage=1.0, jains_fairness=1.0,
        participation_entropy=1.0,
        round_close_rate_kmin1=1.0,
        round_close_rate_kmin2=1.0,
        round_close_rate_kminhalf=1.0,
        round_close_rate_kminN=1.0,
        mission_completion_rate=1.0,
        completion_fairness=1.0,
        rho_contact=None, pass2_coverage=None,
        propulsion_energy_J=None, propulsion_idle_J=None,
        propulsion_tx_J=None, propulsion_prop_J=None,
        mission_completion_s=None, path_length_m=None,
    )
    row = s.to_row()
    assert row["rho_contact"] == ""
    assert row["update_yield"] == 1.0


# --------------------------------------------------------------------------- #
# mission_completion_rate — round-count-invariant yield
# --------------------------------------------------------------------------- #

def test_mission_completion_rate_fraction_with_at_least_one_completion():
    completions = {"d0": 2, "d1": 0, "d2": 1, "d3": 0}
    # 2 of 4 devices had ≥1 completion → 0.5
    assert mission_completion_rate(
        completions, n_devices=4,
    ) == pytest.approx(0.5)


def test_mission_completion_rate_clamps_when_sparse_map():
    # Only 3 entries in the map but 5 admitted devices — the missing
    # ones count as zero completions, so the denominator stays 5.
    completions = {"d0": 1, "d1": 1, "d2": 1}
    assert mission_completion_rate(
        completions, n_devices=5,
    ) == pytest.approx(3 / 5)


def test_mission_completion_rate_zero_devices():
    assert mission_completion_rate({}, n_devices=0) == 0.0


def test_mission_completion_rate_all_zero_completions():
    assert mission_completion_rate(
        {"d0": 0, "d1": 0}, n_devices=2,
    ) == 0.0


def test_mission_completion_rate_in_summarise_trial_for_mule_arm():
    """``summarise_trial`` must populate ``mission_completion_rate`` for
    mule arms from the episode-metrics ``per_device_completions`` map.
    """
    metrics = Exp3EpisodeMetrics(
        contacts_visited=2,
        devices_visited=4,
        devices_completed=2,
        per_device_visits={"d0": 1, "d1": 1, "d2": 1, "d3": 1},
        per_device_completions={"d0": 1, "d1": 1, "d2": 0, "d3": 0},
    )
    rounds = [Exp3RoundLog(0, 1, 2, True), Exp3RoundLog(1, 1, 2, True)]
    s = summarise_trial(
        rounds=rounds, metrics=metrics, cal=None,
        n_devices=4, is_mule_arm=True,
    )
    # 2 of 4 admitted devices completed → 0.5
    assert s.mission_completion_rate == pytest.approx(0.5)


def test_completion_fairness_perfect_when_all_clients_complete_equally():
    completions = {f"d{i}": 5 for i in range(10)}
    assert completion_fairness(
        completions, n_devices=10,
    ) == pytest.approx(1.0)


def test_completion_fairness_drops_with_dead_zones():
    """Half of clients with zero completions, half with 20 each. The
    fairness must collapse — that's the whole reason this metric
    exists, to expose the hidden inequality A1's universal-sampling
    visit-based fairness ignores.
    """
    # 5 silent clients, 5 active clients each with 20 completions.
    completions = {f"silent-{i}": 0 for i in range(5)}
    completions.update({f"active-{i}": 20 for i in range(5)})
    cf = completion_fairness(completions, n_devices=10)
    # J = (5*20)^2 / (10 * 5*400) = 10000 / 20000 = 0.5
    assert cf == pytest.approx(0.5)


def test_completion_fairness_uses_n_devices_for_padding():
    """A sparse completion map (only 3 entries for n_devices=10) must
    be padded to 10 zeros before computing J — otherwise a trial where
    only 3 clients ever completed would report misleadingly high
    fairness on the 3-entry distribution.
    """
    completions = {"d0": 5, "d1": 5, "d2": 5}
    cf = completion_fairness(completions, n_devices=10)
    # J = (15)^2 / (10 * (3*25)) = 225 / 750 = 0.3
    assert cf == pytest.approx(0.3)


def test_completion_fairness_zero_devices_returns_neutral():
    assert completion_fairness({}, n_devices=0) == 1.0


def test_completion_fairness_in_summarise_trial_mule_arm():
    metrics = Exp3EpisodeMetrics(
        contacts_visited=2, devices_visited=4, devices_completed=2,
        per_device_visits={"d0": 1, "d1": 1, "d2": 1, "d3": 1},
        per_device_completions={"d0": 1, "d1": 1, "d2": 0, "d3": 0},
    )
    rounds = [Exp3RoundLog(0, 1, 2, True), Exp3RoundLog(1, 1, 2, True)]
    s = summarise_trial(
        rounds=rounds, metrics=metrics, cal=None,
        n_devices=4, is_mule_arm=True,
    )
    # 2 of 4 clients completed once each → distribution [1,1,0,0]
    # J = (2)^2 / (4 * 2) = 0.5
    assert s.completion_fairness == pytest.approx(0.5)


def test_mission_completion_rate_unaffected_by_round_count():
    """Two trials with the same completions but different round counts
    must report the same ``mission_completion_rate`` — that's the whole
    reason this metric exists.
    """
    completions = {"d0": 1, "d1": 1, "d2": 0, "d3": 0}
    # Two-round trial with same completion set:
    metrics_short = Exp3EpisodeMetrics(
        contacts_visited=2, devices_visited=4, devices_completed=2,
        per_device_visits={"d0": 1, "d1": 1, "d2": 1, "d3": 1},
        per_device_completions=dict(completions),
    )
    # Eight-round trial with same completion set (e.g. multiple
    # revisits each contributing zero):
    metrics_long = Exp3EpisodeMetrics(
        contacts_visited=8, devices_visited=10, devices_completed=2,
        per_device_visits={"d0": 4, "d1": 4, "d2": 1, "d3": 1},
        per_device_completions=dict(completions),
    )
    short_rounds = [Exp3RoundLog(i, 1, 2, True) for i in range(2)]
    long_rounds = [Exp3RoundLog(i, 0, 2, True) for i in range(8)]
    s_short = summarise_trial(
        rounds=short_rounds, metrics=metrics_short, cal=None,
        n_devices=4, is_mule_arm=True,
    )
    s_long = summarise_trial(
        rounds=long_rounds, metrics=metrics_long, cal=None,
        n_devices=4, is_mule_arm=True,
    )
    # update_yield differs (per-round mean shifts) — that's the bias.
    assert s_short.update_yield != s_long.update_yield
    # mission_completion_rate is identical.
    assert s_short.mission_completion_rate == pytest.approx(
        s_long.mission_completion_rate
    )
