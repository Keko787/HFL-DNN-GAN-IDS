"""Legacy pin (Freeze Rule 1): the exp4 channel at afa9526, bit for bit.

Unit U2 moves ``ChannelModel``, ``loss_from_snr``, ``BackhaulPlan`` and
``backhaul_plan`` verbatim to ``hermes/l1/channel_model.py``; the old import
path becomes a re-export shim. Both paths must reproduce the recorded SNR
traces, loss schedules, chosen bands and chosen-band mean SNR (the mule's
``rf_prior_snr_db``) exactly, and so must the 120 recorded C1/C2 L1-channel
trials. Fixtures: ``data/channel.json`` (``make_goldens.py``).
"""

from __future__ import annotations

import importlib

import pytest

from tests.golden import _build_channel as B
from tests.golden._canon import BASE_COMMIT, diff, load

GOLDEN = load("channel")


@pytest.fixture(scope="module")
def current():
    return B.build_cases()


def _mismatches(golden_cases, current_cases, prefix=""):
    out = []
    for key, value in golden_cases.items():
        if not key.startswith(prefix):
            continue
        if key not in current_cases:
            out.append(f"{key}: no longer built")
            continue
        problems = diff(value, current_cases[key])
        if problems:
            out.append(f"{key}: " + "; ".join(problems[:4]))
    return out


def _report(bad):
    return (f"{len(bad)} channel case(s) changed since afa9526:\n  "
            + "\n  ".join(bad[:15]))


def test_fixture_was_captured_at_the_base_commit():
    assert GOLDEN["_meta"]["base_commit"] == BASE_COMMIT


@pytest.mark.parametrize("prefix", ["cell:", "controller:", "loss_from_snr:", "edge:"])
def test_channel_is_bit_identical(current, prefix):
    bad = _mismatches(GOLDEN["cases"], current, prefix)
    assert not bad, _report(bad)


def test_the_hermes_l1_home_matches_once_it_exists():
    """After U2 the code lives in ``hermes.l1.channel_model``: same numbers."""
    try:
        module = importlib.import_module("hermes.l1.channel_model")
    except ModuleNotFoundError:
        pytest.skip("hermes.l1.channel_model does not exist yet (unit U2 creates it)")
    bad = _mismatches(GOLDEN["cases"], B.build_cases(module))
    assert not bad, _report(bad)


def test_recorded_l1_channel_trials_rederive_exactly():
    """The kept C1/C2 trials (results/exp4_matrix/C*_traces): the cluster's loss
    schedule and backhaul seed and the mule's RF prior, from today's code."""
    trials = B.recorded_trials()
    if not trials:
        pytest.skip("no recorded C1/C2 traces under results/exp4_matrix")
    assert len(trials) == 120, f"expected the 120 recorded C1/C2 arm-trials, found {len(trials)}"
    bad = []
    for d in trials:
        recorded, derived = B.rederive_recorded(d)
        if recorded != derived:
            bad.append(f"{d.parent.name}/{d.name}: recorded {recorded} derived {derived}")
    assert not bad, f"{len(bad)} recorded trial(s) no longer re-derive:\n  " + "\n  ".join(bad[:5])
