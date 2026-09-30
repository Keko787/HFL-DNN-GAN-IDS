"""Legacy pin (Freeze Rule 1): Exp 4 topologies, per-role configs, driver rows.

Unit U7 edits the process configs, the topology builder and the driver. In
legacy mode positions, reliabilities, seeds and every config value must stay
as recorded at afa9526; configs and rows may only gain keys (compared on the
afa9526 key sets, critic A3), and the kept traces in ``results/`` must still
re-derive from their seeds, with any added key at its default and old per-role
JSON still loading (design section 5.2). Fixture: ``data/topology.json``.
"""

from __future__ import annotations

import logging

import pytest

from tests.golden import _build_topology as T
from tests.golden._canon import BASE_COMMIT, assert_same, load

GOLDEN = load("topology")
BUILDER = [k for k in GOLDEN["cases"] if k.startswith("builder:")]
DRIVER = [k for k in GOLDEN["cases"] if k.startswith("driver:")]


def test_fixture_was_captured_at_the_base_commit():
    assert GOLDEN["_meta"]["base_commit"] == BASE_COMMIT


def test_builder_grid():
    current = T.build_builder_cases()
    assert sorted(current) == sorted(BUILDER)
    for key in BUILDER:
        assert_same(GOLDEN["cases"][key], current[key], key)


def test_role_json_taken_in_memory_matches_the_files():
    """The kept-trace check takes the orchestrator's writes in memory; the
    pins read the files it wrote. Both give the same JSON."""
    name, kwargs, _rule = next(c for c in T.builder_cells() if "K=3|cutoff" in c[0])
    on_disk = T.role_configs(T.build_exp4_topology(**kwargs))
    in_memory = T.role_configs(T.build_exp4_topology(**kwargs), on_disk=False)
    assert len(on_disk["mules"]) == 3, name
    assert in_memory == on_disk


@pytest.mark.parametrize("key", DRIVER)
def test_driver_stub_trial(key, caplog):
    """The topology the driver builds and the row it returns (nothing spawned)."""
    caplog.set_level(logging.ERROR, logger="experiments.exp4.driver")
    name = key.split(":", 1)[1]
    kwargs, cell = {n: (k, c) for n, k, c in T.DRIVER_CASES}[name]
    row, topo = T.run_stub_trial(T.Exp4Driver(**kwargs), cell)
    current = {"row": T.record("Exp4Row", row), "topology": T.topology_record(topo)}
    assert_same(GOLDEN["cases"][key], current, key)


def test_recorded_trials_rederive_from_their_seeds(caplog):
    caplog.set_level(logging.ERROR, logger="experiments.exp4.driver")
    dirs = T.recorded_trace_dirs()
    if not dirs:
        pytest.skip("no kept traces under results/")
    bad = []
    for d in dirs:
        problems = T.compare_recorded(d)
        if problems:
            bad.append(f"{d.parent.name}/{d.name}: " + "; ".join(problems[:3]))
    assert not bad, (
        f"{len(bad)} of {len(dirs)} kept trial(s) no longer re-derive:\n  " + "\n  ".join(bad[:6])
    )
    assert len(dirs) >= 500
