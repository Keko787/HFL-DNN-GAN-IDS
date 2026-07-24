"""EX-4.2 arm H2 — mule RL-selector wiring (fast, no subprocess/TF).

Pins the config -> selector construction: no selector by default (H1),
a random-init TargetSelectorRL when use_rl_selector is set (H2 smoke).
"""

from __future__ import annotations

from hermes.processes.config import MuleConfig
from hermes.processes.mule import _build_target_selector
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
