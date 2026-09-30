"""Legacy pin (Freeze Rule 1): ``MuleSupervisor`` end to end at afa9526.

Unit U6 threads the mission clock, the ferry contact and the re-plan through
``mule_main.py`` and the K > 1 dock code (critic B6). With every Phase 3
switch at its default, these deterministic loopback runs must reproduce the
recorded ``MissionRunResult``s, scheduler device states, deltas (session and
widening), RF calls, queues, S3b results, FedEx diagnostics and S3c's record,
and the cluster's UP/DOWN traffic: H1 with no budget; H1 with a budget
(pre-flight drops, and an in-flight abort with widening, also with S3c on);
H2; D1 with a budget (and its budget-only abort); D4; a budgeted Pass 2; the
multiplicative law with the miss priority and S3c; and two mules at quorum 2,
once with empty missions docking (``dock_on_empty``) and once with a survived
DOWN timeout. Dataclasses are compared on their afa9526 field sets (critic
A3). The harness is described in ``_mule_harness.py``. Fixture:
``data/supervisor.json``.
"""

from __future__ import annotations

import pytest

from tests.golden import _mule_harness as M
from tests.golden._canon import BASE_COMMIT, assert_same, load

GOLDEN = load("supervisor")


def test_fixture_was_captured_at_the_base_commit():
    assert GOLDEN["_meta"]["base_commit"] == BASE_COMMIT
    assert sorted(GOLDEN["cases"]) == sorted(M.SCENARIOS)


@pytest.mark.parametrize("name", sorted(M.SCENARIOS))
def test_supervisor_scenario(name):
    assert_same(GOLDEN["cases"][name], M.SCENARIOS[name](), name)


def _missions(case):
    for m in case["missions"]:
        if isinstance(m, list):
            for _who, rec in m:
                yield rec
        else:
            yield m


def _queue_devices(rec):
    return [d for wp in rec["result"]["pass_1_queue"] for d in wp["devices"]]


def _abandoned_in_flight(rec):
    """Devices of the mission's own Pass-1 queue that were widened without a
    contact: the tail an in-flight abort gave up on.

    Widening (``direct`` deltas) also covers the contacts S3b dropped before
    take-off, but those are not in the queue.
    """
    report = rec["result"]["report"] or {}
    contacted = {line["device_id"] for line in report.get("lines", [])}
    widened = {d["device_id"] for src, d in rec["deltas"] if src == "direct"}
    return (set(_queue_devices(rec)) & widened) - contacted


def test_the_scenarios_reach_the_paths_they_pin():
    """Guards the harness itself: each scenario still exercises its path."""
    cases = GOLDEN["cases"]
    direct = lambda name: sum(  # noqa: E731
        1 for rec in _missions(cases[name]) for src, _d in rec["deltas"] if src == "direct")
    assert direct("h1_budget") > 0                 # pre-flight drops widened
    # An in-flight abort, and its widening, not only pre-flight drops.
    for name in ("h1_budget_abort", "h1_s3c_budget_abort", "d1_max_aoi_budget_abort"):
        assert any(_abandoned_in_flight(rec) for rec in _missions(cases[name])), name
    # S3c after an abort: ``served`` counts the contacts flown, not the queue.
    aborted = [rec for rec in _missions(cases["h1_s3c_budget_abort"]) if _abandoned_in_flight(rec)]
    assert aborted
    for rec in aborted:
        served, planned = rec["s3c"]["history"][-1]
        assert served == len(_queue_devices(rec)) - len(_abandoned_in_flight(rec))
        assert planned > len(_queue_devices(rec))          # pre-flight drops count
    assert any(rec.get("fedex") for rec in _missions(cases["d4_fedex"]))
    skipped = [
        line for rec in _missions(cases["pass_2_budget"])
        for line in (rec["result"]["delivery_report"] or {}).get("lines", [])
        if line["outcome"] == "e:DeliveryOutcome.SKIPPED"
    ]
    assert skipped
    k2 = list(_missions(cases["k2_quorum_dock_on_empty"]))
    assert any(rec["result"]["docked_empty"] for rec in k2)
    k2t = list(_missions(cases["k2_down_timeout"]))
    assert any(rec["result"]["down_timeout"] for rec in k2t)
    assert any(rec["stale_downs_dropped"] for rec in k2t)
