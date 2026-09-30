"""Legacy pin (Freeze Rule 1): ``HFLHostMission``'s contact routines at afa9526.

Unit U5 merges ``run_contact`` and ``deliver_contact`` behind a sink and adds a
ferry path, and leaves ``run_session`` alone. With no contact plan the code
must reproduce, for the contact map's 28 scenarios and for ``run_session``'s
branches (``S ...``), the outcome maps, deltas, ledgers, accepted
submissions, stash, busy flags, RF calls and log records recorded here, run
with synchronous threads (exact) and with real threads (exact for one device;
order-free, stamps masked, for several: critic R6). Messages and lines are
compared on their afa9526 field sets only, so the fields Phase 3 adds with
defaults do not break the pins (critic A3). The late-writer scenario pins
P-01 defect 3 (critic A2), and the sequential-joins scenarios P-01 defect 2,
both kept in legacy mode. Fixture: ``data/host_mission.json``.
"""

from __future__ import annotations

import pytest

from tests.golden import _host_harness as H
from tests.golden._canon import BASE_COMMIT, assert_same, diff, load, order_free

GOLDEN = load("host_mission")
NAMES = list(H.scenarios())


def test_fixture_was_captured_at_the_base_commit():
    assert GOLDEN["_meta"]["base_commit"] == BASE_COMMIT
    assert sorted(f"scenario:{n}" for n in NAMES) == sorted(
        k for k in GOLDEN["cases"] if k.startswith("scenario:"))
    assert {"late_writer", "sequential_joins:collect", "sequential_joins:deliver"} <= set(
        GOLDEN["cases"])
    # The contact map's 28 scenarios, and run_session's own.
    assert sum(1 for n in NAMES if not n.startswith("S ")) == 28
    assert sum(1 for n in NAMES if n.startswith("S ")) >= 10


@pytest.mark.parametrize("name", NAMES)
def test_scenario_synchronous_threads(name):
    assert_same(GOLDEN["cases"][f"scenario:{name}"], H.run_scenario(name, sync=True), name)


@pytest.mark.parametrize("name", NAMES)
def test_scenario_real_threads(name):
    golden = GOLDEN["cases"][f"scenario:{name}"]
    current = H.run_scenario(name, sync=False)
    if not H.is_multi(name):
        # One worker, run while the caller is blocked in join: exact.
        assert_same(golden, current, f"{name} (real threads)")
        return
    g_rest, g_bags = H.split_unordered(golden)
    c_rest, c_bags = H.split_unordered(current)
    problems = diff(g_rest, c_rest)
    for key, items in g_bags.items():
        want, got = order_free(items, c_bags.get(key, []))
        if want != got:
            problems.append(f"{key}: as a multiset\n    golden  {want}\n    current {got}")
    assert not problems, f"{name} (real threads) changed:\n  " + "\n  ".join(problems[:8])


@pytest.mark.parametrize("pass_kind", ["collect", "deliver"])
def test_sequential_joins_wait_for_each_worker_in_turn(pass_kind):
    """P-01 defect 2, kept in legacy mode (design section 4.1).

    Each worker is joined in turn for up to 2 x TTL. Device A's reply comes
    after A's join gave up, inside B's: it is in the map the routine returns.
    B's reply comes after the return and lands afterwards. Real threads; the
    margins are one TTL each way; stamps masked, RF calls order-free.
    """
    current = H.run_sequential_joins(pass_kind)
    at_return = current["at_return"]
    assert [k for k, _v in at_return["map"]] == ["d0"], at_return
    assert at_return["a_released_before_return"] is True
    assert [k for k, _v in current["after"]["returned_map_after"]] == ["d0", "d1"]
    golden = dict(GOLDEN["cases"][f"sequential_joins:{pass_kind}"])
    g_after, c_after = dict(golden.pop("after")), dict(current.pop("after"))
    problems = []
    for key in H.JOINS_UNORDERED:
        want, got = order_free(g_after.pop(key), c_after.pop(key))
        if want != got:
            problems.append(f"after.{key}: as a multiset\n    golden  {want}\n    current {got}")
    problems += diff(g_after, c_after, "$.after") + diff(golden, current)
    assert not problems, f"sequential joins ({pass_kind}) changed:\n  " + "\n  ".join(problems[:8])


def test_late_writer_lands_in_the_next_round():
    """A Pass-1 worker outlives its join; round 1 closes empty and round 2 opens.

    Its gradient then lands in round 2: the accepted list has length 1, the
    report and contact ledgers get its line, the delta carries round 2, the
    caller's outcome map is filled in after the fact, and round 2's merge
    refuses the round-1 submission. Real threads, so the stamps are masked.
    """
    current = H.run_late_writer()
    assert current["round_2"]["accepted_len"] == 1
    assert current["round_2"]["bus"][0]["mission_round"] == 2
    assert_same(GOLDEN["cases"]["late_writer"], current, "late writer")
