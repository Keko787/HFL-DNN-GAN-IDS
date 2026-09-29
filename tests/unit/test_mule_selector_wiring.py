"""EX-4.2 arm H2 — mule RL-selector wiring (fast, no subprocess/TF).

Pins the config -> selector construction: no selector by default (H1),
a random-init TargetSelectorRL when use_rl_selector is set (H2 smoke), and
(FeRRy Phase 2) the D3-D5 whole-scheduler policies with their options.
"""

from __future__ import annotations

import pytest

from hermes.processes.config import MuleConfig
from hermes.processes.mule import DOCK_POSE, _build_target_selector
from hermes.scheduler.policies import (
    FedCSDegradedPolicy,
    FedExCarpPolicy,
    WhittlePolicy,
)
from hermes.scheduler.selector import TargetSelectorRL


def test_no_selector_by_default():
    assert _build_target_selector(MuleConfig(mule_id="m")) is None


def test_random_init_selector_when_enabled():
    sel = _build_target_selector(MuleConfig(mule_id="m", use_rl_selector=True))
    assert isinstance(sel, TargetSelectorRL)
    assert sel.epsilon == 0.0        # greedy


def test_missing_weights_file_falls_back_only_when_path_none():
    # A configured-but-empty path uses random init; a real .npz load is
    # covered by the exp3 path. Here we only assert the None path is safe.
    sel = _build_target_selector(
        MuleConfig(mule_id="m", use_rl_selector=True, selector_weights_path=None)
    )
    assert isinstance(sel, TargetSelectorRL)


# --------------------------------------------------------------------------- #
# FeRRy Phase 2 — arms D3-D5 through the same slot
# --------------------------------------------------------------------------- #

def test_whittle_builds_with_the_module_defaults():
    sel = _build_target_selector(MuleConfig(mule_id="m", contact_policy="whittle"))
    assert isinstance(sel, WhittlePolicy)
    ref = WhittlePolicy()
    assert (sel.variant, sel.weights, sel.rho_min) == (ref.variant, ref.weights, ref.rho_min)


def test_whittle_takes_its_variant_and_weights_from_the_config():
    sel = _build_target_selector(MuleConfig(
        mule_id="m", contact_policy="whittle",
        whittle_variant="literal", whittle_weights="oort",
    ))
    assert (sel.variant, sel.weights) == ("literal", "oort")


def test_fedex_tours_home_to_the_dock():
    sel = _build_target_selector(MuleConfig(mule_id="m", contact_policy="fedex"))
    assert isinstance(sel, FedExCarpPolicy)
    assert sel.depot == DOCK_POSE == (0.0, 0.0, 0.0)


def test_fedcs_takes_its_value_from_the_config():
    sel = _build_target_selector(MuleConfig(mule_id="m", contact_policy="fedcs"))
    assert isinstance(sel, FedCSDegradedPolicy) and sel.value == "unit"
    sel = _build_target_selector(
        MuleConfig(mule_id="m", contact_policy="fedcs", fedcs_value="devices")
    )
    assert sel.value == "devices"


def test_a_bad_policy_option_is_refused_when_the_mule_builds_it():
    with pytest.raises(ValueError):
        _build_target_selector(MuleConfig(
            mule_id="m", contact_policy="whittle", whittle_variant="optimal",
        ))


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="unknown contact_policy"):
        _build_target_selector(MuleConfig(mule_id="m", contact_policy="nope"))
