"""Phase 3's simulated-clock pipeline, pinned at 6e6f92d (FeRRy Phase 4, unit UG4).

Freeze Rule 1 in Phase 4: every mechanism sits behind a switch whose default is
the recorded pipeline, and "recorded" now includes Phase 3's simulated clock
with ``plan_mode=legacy``. These tests run the eight stub trials of
``_build_p3_sim`` in this process (the driver, the per-role JSON the real
orchestrator writes, the mule, cluster and device services, and the mule
process's own service loop) and compare each part of each trial with
``data/p3_sim.json``, captured on the untouched tree at 6e6f92d. A field added
with a default passes; a removed or renamed field, a changed value, a new
event, or a device appearing in or leaving a device-keyed map fails (critic
A3's rule, see ``_build_p3_sim``). A few tests read the fixture itself, so
that it keeps exercising what the unit spec names: H1's in-flight re-plan
with the trim, D1's and D3's pre-flight drops and re-plans, D4's route-only
tour, and the narrow cliff's empty missions; and what the unit's review found
missing: contacts flown on narrow and on medium (the cliff's other side, and
medium's drops, re-plan and a member below the floor at arrival), and the
measured payload.

The second half checks the unit's full-suite tooling, ``make_baseline.py``:
the flaky allow-list, the second baseline, recorded at 6e6f92d, and that a
missing baseline fails ``compare`` instead of being skipped.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, List

import pytest

from tests.golden import _build_p3_sim as B
from tests.golden import _canon
from tests.golden import make_baseline as MB

GOLDEN = B.load_golden()
CASES: Dict[str, Any] = GOLDEN["cases"]


def plain(value: Any) -> Any:
    """A canonical value back as plain JSON: records untyped, floats as floats."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if k != _canon.TYPE_KEY}
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, str) and value.startswith("f:"):
        return float(value[2:])
    return value


def missions(name: str) -> List[Dict[str, Any]]:
    (mule,) = CASES[name]["mission_completed"].values()
    return plain(mule)


def row(name: str) -> Dict[str, Any]:
    return plain(CASES[name]["row"])


# --------------------------------------------------------------------------- #
# The fixture
# --------------------------------------------------------------------------- #

def test_the_fixture_is_the_6e6f92d_capture_of_the_named_trials():
    meta = GOLDEN["_meta"]
    assert meta["base_commit"] == B.BASE_COMMIT == "6e6f92da038227147489515d876cc3f353584283"
    assert meta["unit"] == "UG4"
    assert tuple(CASES) == B.TRIAL_NAMES
    for name in B.TRIAL_NAMES:
        assert set(CASES[name]) == {"inputs", *B.PARTS}
        # The trial table is the one the fixture was captured with.
        assert CASES[name]["inputs"] == B.inputs_of(*B.TRIALS[name]), name


# --------------------------------------------------------------------------- #
# The pins
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("part", B.PARTS)
@pytest.mark.parametrize("name", B.TRIAL_NAMES)
def test_the_simulated_clock_pipeline_at_its_defaults_is_the_6e6f92d_one(name, part):
    golden = CASES[name][part]
    current = B.capture(name)[part]
    problems = _canon.diff(golden, current)
    assert not problems, (
        f"{name}.{part}: Phase 3's simulated-clock behaviour pinned at 6e6f92d changed "
        f"({len(problems)} mismatch(es) shown; a key added with a default would pass):\n  "
        + "\n  ".join(problems)
    )


# --------------------------------------------------------------------------- #
# What the fixture exercises
# --------------------------------------------------------------------------- #

def test_h1_replans_in_flight_with_the_trim_after_pre_flight_drops():
    ms = missions("h1_replan_trim_wide")
    orders = [r["order_used"] for m in ms for r in m["replans"]]
    assert "arm_trimmed" in orders and "arm" in orders, orders
    dropped = [d for m in ms for r in m["replans"] for d in r["dropped"]]
    assert dropped and all(d["reason"] == "budget" and d["devices"] for d in dropped)
    drops = [d["reason"] for m in ms for d in m["pass_1_preflight_drops"]]
    assert drops and set(drops) == {"budget"}
    r = row("h1_replan_trim_wide")
    assert (r["mission_clock"], r["contact_band"], r["in_flight_response"]) == ("sim", "wide", "replan")
    assert json.loads(r["ferry_params"])["replan_fallback"] == "trim"
    assert r["aggregation"] == "agg:cutoff" and r["backhaul_model"] == "seconds"
    events = [e["event"] for e in plain(CASES["h1_replan_trim_wide"]["cluster_events"])]
    assert "backhaul_upload_lost" in events and events.count("cluster_round_closed") == 3


@pytest.mark.parametrize("name", ["d1_max_aoi", "d3_whittle"])
def test_d1_and_d3_leave_stops_out_before_takeoff_and_replan(name):
    """Their walks drop stops before takeoff, which they do not report at
    6e6f92d (``pass_1_preflight_drops`` is empty; Phase 4's Decision 6 adds a
    ``pass_1_policy_drops`` field, which passes here as an added key)."""
    ms = missions(name)
    n = CASES[name]["inputs"]["cell"]["params"]["N"]
    planned = [sum(len(p["devices"]) for p in m["pass_1_plan"]) for m in ms]
    assert min(planned) < n, planned
    assert all(m["pass_1_preflight_drops"] == [] for m in ms)
    assert all("pass_1_policy_drops" not in m for m in ms)
    assert [r["order_used"] for m in ms for r in m["replans"]] == ["arm"]


def test_d4_flies_its_whole_tour_route_only_and_overruns():
    ms = missions("d4_route_only")
    n = CASES["d4_route_only"]["inputs"]["cell"]["params"]["N"]
    assert [sum(len(p["devices"]) for p in m["pass_1_plan"]) for m in ms] == [n] * len(ms)
    assert all(m["budget_overrun_s"] > 0.0 and m["replans"] == [] for m in ms)
    assert row("d4_route_only")["aggregation"] == "agg:cutoff"


def test_the_narrow_cliff_flies_every_mission_empty():
    ms = missions("h1_narrow_cliff_empty")
    everyone = sorted(CASES["h1_narrow_cliff_empty"]["device_events"])
    for m in ms:
        assert m["pass_1_contacts"] == 0 and m["pass_1_flown"] == []
        (drop,) = m["pass_1_preflight_drops"]
        assert drop["reason"] == "budget" and sorted(drop["devices"]) == everyone
        assert m["sim_ledger"]["turnaround"] == m["sim_end_s"] - m["sim_start_s"] == 30.0
    (names,) = CASES["h1_narrow_cliff_empty"]["mule_events"]["names"].values()
    assert names.count("mission_empty") == len(ms) == 3
    events = [e["event"] for e in plain(CASES["h1_narrow_cliff_empty"]["cluster_events"])]
    assert events == ["cluster_ready", "mule_bootstrapped"]
    r = row("h1_narrow_cliff_empty")
    assert (r["contact_band"], r["mission_budget_s"]) == ("narrow", 60.0)


def stops_flown(name: str) -> List[Dict[str, Any]]:
    return [s for m in missions(name) for s in m["pass_1_flown"] + m["pass_2_flown"]]


def test_the_cliffs_other_side_flies_the_field_wide_narrow_stop():
    """The same trial as the empty one but for the budget: at 99.5 s S3b
    admits the stop it drops at 60 s, and every mission flies it, all eight
    members targeted in both passes, with narrow's airtime at 1 MB."""
    empty = CASES["h1_narrow_cliff_empty"]["inputs"]
    flown = CASES["h1_narrow_cliff_flown"]["inputs"]
    assert flown["cell"] == empty["cell"]
    budget = "mission_budget_s"
    assert ({k: v for k, v in flown["settings"].items() if k != budget}
            == {k: v for k, v in empty["settings"].items() if k != budget})
    assert (plain(empty["settings"][budget]), plain(flown["settings"][budget])) == (60.0, 99.5)
    everyone = sorted(CASES["h1_narrow_cliff_flown"]["device_events"])
    ms = missions("h1_narrow_cliff_flown")
    for m in ms:
        assert m["pass_1_preflight_drops"] == []
        for kind in ("pass_1_flown", "pass_2_flown"):
            (stop,) = m[kind]
            assert stop["band"] == "narrow" and stop["unreachable"] == []
            assert sorted(stop["targets"]) == everyone and stop["dwell_s"] > 30.0
    assert sum(m["budget_overrun_s"] > 0.0 for m in ms) == 1
    events = [e["event"] for e in plain(CASES["h1_narrow_cliff_flown"]["cluster_events"])]
    assert events.count("cluster_round_closed") == len(ms) == 3


def test_medium_prices_drops_replans_and_gates_on_its_own_physics():
    ms = missions("h1_medium_replan")
    assert {s["band"] for s in stops_flown("h1_medium_replan")} == {"medium"}
    assert {d["reason"] for m in ms for d in m["pass_1_preflight_drops"]} == {"budget"}
    assert [r["order_used"] for m in ms for r in m["replans"]] == ["arm"]
    # A Pass-1 member below the SNR floor at arrival is not a target.
    floor = json.loads(row("h1_medium_replan")["ferry_params"])["snr_floor_db"]
    below = [(d, s) for m in ms for s in m["pass_1_flown"] for d in s["unreachable"]]
    assert below
    for d, s in below:
        assert s["snr_db"][d] < floor and d not in s["targets"] and s["rate_bps"][d] == 0.0
    assert {m["backhaul"]["carrier"] for m in ms} == {2}
    events = [e["event"] for e in plain(CASES["h1_medium_replan"]["cluster_events"])]
    assert "backhaul_upload_lost" in events
    r = row("h1_medium_replan")
    assert (r["contact_band"], r["in_flight_response"], r["mission_budget_s"]) == (
        "medium", "replan", 60.0)


def test_the_measured_payload_prices_what_the_mule_measures():
    settings, _cell = B.TRIALS["h1_narrow_measured"]
    assert "payload_bytes" not in settings
    (ready,) = plain(CASES["h1_narrow_measured"]["mule_ready"]).values()
    assert [e["payload"] for e in ready] == [{"payload_bytes": None, "mode": "measured"}]
    assert json.loads(row("h1_narrow_measured")["ferry_params"])["payload_bytes"] is None
    stops = stops_flown("h1_narrow_measured")
    assert {s["band"] for s in stops} == {"narrow"}
    # The stub's θ and synthetic batch: milliseconds of airtime, not zero.
    assert all(0.0 < s["dwell_s"] < 0.05 for s in stops)


def test_every_band_class_and_both_payload_modes_fly_contacts():
    """The review's gap: no trial flew a contact off wide, so the band
    classes' in-flight physics (arrival SNR, gate, airtime, rates) were not
    pinned on narrow and medium, nor the measured payload anywhere."""
    bands = {s["band"] for name in B.TRIAL_NAMES for s in stops_flown(name)}
    assert bands == {"wide", "medium", "narrow"}

    def payload_modes(name: str) -> set:
        (ready,) = plain(CASES[name]["mule_ready"]).values()
        return {e["payload"]["mode"] for e in ready}

    assert set().union(*map(payload_modes, B.TRIAL_NAMES)) == {"declared", "measured"}


def test_the_rows_device_serve_columns_are_harness_values():
    """In process no device service loop runs, so no ``device_served`` event
    is written: every row has coverage 0, participation entropy 0 and Jain's
    index 1.0 (no serves). Harness artifacts, not Phase 3's values (the
    README says so); the consumer's fold of real serves is tested in
    tests/unit/test_exp4_metrics.py."""
    for name in B.TRIAL_NAMES:
        r = row(name)
        assert (r["coverage"], r["jains_fairness"], r["participation_entropy"]) == (
            0.0, 1.0, 0.0), name
        for evs in plain(CASES[name]["device_events"]).values():
            assert [e["event"] for e in evs] == ["device_ready"], name


# --------------------------------------------------------------------------- #
# The comparison rule
# --------------------------------------------------------------------------- #

def test_an_added_key_passes_and_everything_else_is_caught():
    """Critic A3's rule as the fixture applies it, on the H1 trial's second mission."""
    ids = frozenset(CASES["h1_replan_trim_wide"]["device_events"])
    golden = CASES["h1_replan_trim_wide"]["mission_completed"]
    raw = plain(golden)
    (mule,) = raw

    def diff_after(change) -> List[str]:
        current = json.loads(json.dumps(raw))
        change(current[mule][1])
        return _canon.diff(golden, {m: [B.payload(e, "mule.mission_completed", ids) for e in evs]
                                    for m, evs in current.items()})

    assert diff_after(lambda m: None) == []
    # Added fields pass, at the top and inside a record.
    assert diff_after(lambda m: m.update(pass_1_policy_drops=[{"devices": ["x"]}])) == []
    assert diff_after(lambda m: m["backhaul"].update(band="wide")) == []
    # A removed or renamed field, a changed value, a device in a device-keyed map: caught.
    assert diff_after(lambda m: m.update(re_plans=m.pop("replans")))
    assert diff_after(lambda m: m["backhaul"].pop("p_loss"))
    assert diff_after(lambda m: m.update(energy_j=m["energy_j"] + 1e-9))
    assert diff_after(lambda m: m["deadline_state"].pop(sorted(m["deadline_state"])[0]))
    assert diff_after(lambda m: m["pass_1_flown"].pop())


def test_the_harness_links_keep_the_rf_link_token_check():
    server = B.MuleRF(B.WallClock(), port=1, link_token="tok-a")
    B.DeviceRF(server, "dev-a", "tok-a")
    with pytest.raises(ConnectionError, match="Amendment 10"):
        B.DeviceRF(server, "dev-b", "tok-b")
    assert server.order == ["dev-a"]


# --------------------------------------------------------------------------- #
# make_baseline.py: the flaky allow-list and the second baseline
# --------------------------------------------------------------------------- #

FLAKY_NID = "tests/x.py::test_flaky"
SIG = "tests/x.py:52 | AssertionError | assert 0 >= 1"


def _cmp(base_result, run_result, flaky=((FLAKY_NID, (SIG,)),)):
    base = {FLAKY_NID: base_result, "tests/x.py::test_ok": MB.Result("passed")}
    run = {FLAKY_NID: run_result, "tests/x.py::test_ok": MB.Result("passed")}
    return MB.compare(base, run, None if flaky is None else dict(flaky))


def test_a_flaky_test_may_pass_or_fail_its_known_way():
    failed, passed = MB.Result("failed", SIG), MB.Result("passed")
    c = _cmp(failed, passed)
    assert not c.differs and c.flaky == [(FLAKY_NID, "failed", "passed")]
    assert _cmp(failed, passed, flaky=None).outcome_changed == [(FLAKY_NID, "failed", "passed")]
    assert not _cmp(passed, failed).differs                     # fails its known way
    c = _cmp(failed, failed)
    assert not c.differs and c.flaky == []                      # no change at all
    # Failing another way, erroring or not running is still a difference.
    other = MB.Result("failed", "tests/x.py:60 | KeyError | 'rounds_closed'")
    assert _cmp(failed, other).signature_changed and _cmp(passed, other).outcome_changed
    assert _cmp(passed, MB.Result("error", SIG)).outcome_changed
    assert MB.compare({FLAKY_NID: passed}, {}, {FLAKY_NID: (SIG,)}).missing == [FLAKY_NID]
    # Declared flaky with no known signature: any failure is allowed.
    assert not _cmp(passed, other, flaky=((FLAKY_NID, ()),)).differs


def test_the_smoke_test_is_allow_listed_with_its_afa9526_failure():
    (nid,) = MB.FLAKY
    recorded = MB.read_baseline()[nid]
    assert recorded.outcome == "failed" and recorded.signature in MB.FLAKY[nid]
    c = MB.compare({nid: recorded}, {nid: MB.Result("passed")}, MB.FLAKY)
    text = MB.report(c, n_baseline=1, n_run=1)
    assert "flaky tests that changed (allowed): 1" in text
    assert text.endswith("same as the afa9526 baseline")


def test_the_afa9526_baseline_renders_as_recorded():
    """The header learnt to name other bases; afa9526's is still the recorded one."""
    recorded = MB.BASELINE.read_text(encoding="utf-8").splitlines()
    rendered = MB.render_baseline(MB.read_baseline(), suite_time=358.8,
                                  head=MB.BASE_COMMIT).splitlines()
    env = 2                                  # the environment line names the Python version
    assert rendered[:env] + rendered[env + 1:] == recorded[:env] + recorded[env + 1:]


def _junit(path, cases):
    """A JUnit XML of ``(name, kind, message, text)`` cases of this module."""
    suite = ET.Element("testsuite", time="2.5")
    for name, kind, message, text in cases:
        tc = ET.SubElement(suite, "testcase", classname="tests.golden.test_golden_p3_sim",
                           name=name)
        if kind == "failure":
            ET.SubElement(tc, kind, message=message).text = text
    root = ET.Element("testsuites")
    root.append(suite)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_write_records_the_base_it_is_asked_for(tmp_path, monkeypatch, capsys):
    xml = _junit(tmp_path / "run.xml", [
        ("test_a", "passed", "", ""),
        ("test_b", "failure", "AssertionError: x\nassert 1 == 2",
         "tests\\golden\\test_golden_p3_sim.py:7: AssertionError"),
    ])
    e6f = MB.BASES["6e6f92d"]
    head = {"sha": e6f.commit}
    monkeypatch.setattr(MB, "_git", lambda *a: head["sha"] if a[0] == "rev-parse" else "")
    out = tmp_path / "b.txt"
    assert MB.main(["write", str(xml), "--base", "6e6f92d", "--baseline", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# FeRRy Phase 4 unit UG4 - pytest baseline at 6e6f92d (main), "
                           "before any Phase 4 change.\n")
    assert MB.baseline_label(out) == "6e6f92d"
    assert MB.read_baseline(out) == MB.results_from_junit(xml)[0]
    # Each base is written only at its own commit (--force aside).
    assert MB.main(["write", str(xml), "--baseline", str(out)]) == 2          # afa9526's
    head["sha"] = MB.BASE_COMMIT
    assert MB.main(["write", str(xml), "--base", "6e6f92d", "--baseline", str(out)]) == 2
    assert "refusing to write: the baseline is 6e6f92d's" in capsys.readouterr().err


def test_compare_checks_every_recorded_baseline(tmp_path, monkeypatch, capsys):
    ok = _junit(tmp_path / "ok.xml", [("test_a", "passed", "", "")])
    bad = _junit(tmp_path / "bad.xml", [("test_a", "failure", "AssertionError: x\nassert 0",
                                         "tests\\golden\\test_golden_p3_sim.py:9: AssertionError")])
    results, _ = MB.results_from_junit(ok)
    bases = {}
    for key, base in MB.BASES.items():
        path = tmp_path / base.path.name
        path.write_text(MB.render_baseline(results, suite_time=1.0, head=base.commit),
                        encoding="utf-8")
        bases[key] = base._replace(path=path)
    monkeypatch.setattr(MB, "BASES", bases)
    assert MB.main(["compare", str(ok)]) == 0
    text = capsys.readouterr().out
    assert "== the afa9526 baseline" in text and "== the 6e6f92d baseline" in text
    assert "same as the afa9526 baseline" in text and "same as the 6e6f92d baseline" in text
    assert MB.main(["compare", str(bad), "--base", "6e6f92d"]) == 1
    text = capsys.readouterr().out
    assert "DIFFERS from the 6e6f92d baseline" in text and "afa9526" not in text


def test_compare_fails_on_a_missing_baseline_instead_of_skipping_it(tmp_path, monkeypatch,
                                                                     capsys):
    """A recorded baseline whose file is absent (say, left out of a commit)
    used to be skipped in silence, switching its gate off; now ``compare``
    says so and exits 2, having still compared the baselines that exist."""
    ok = _junit(tmp_path / "ok.xml", [("test_a", "passed", "", "")])
    results, _ = MB.results_from_junit(ok)
    bases = {}
    for key, base in MB.BASES.items():
        path = tmp_path / base.path.name
        if key == "afa9526":                   # 6e6f92d's file is never written
            path.write_text(MB.render_baseline(results, suite_time=1.0, head=base.commit),
                            encoding="utf-8")
        bases[key] = base._replace(path=path)
    monkeypatch.setattr(MB, "BASES", bases)
    assert MB.main(["compare", str(ok)]) == MB.MISSING_STATUS == 2
    text = capsys.readouterr().out
    assert "same as the afa9526 baseline" in text
    assert "== the 6e6f92d baseline (pytest_baseline_6e6f92d.txt)\nMISSING: " in text
    assert MB.main(["compare", str(ok), "--base", "6e6f92d"]) == 2
    assert "nothing was compared with the 6e6f92d baseline" in capsys.readouterr().out
    assert MB.main(["compare", str(ok), "--baseline", str(tmp_path / "nowhere.txt")]) == 2
    assert MB.main(["compare", str(ok), "--base", "afa9526"]) == 0


def test_the_6e6f92d_baseline_is_recorded_and_gates_phase_3():
    """The second baseline is on disk (a checkout without it fails here, and
    ``compare`` exits 2), says it was recorded at 6e6f92d, agrees with its own
    header, keeps every afa9526 test, holds every Phase 3 test file, and fails
    exactly the afa9526 baseline's deterministic failures, with their
    signatures (the flaky smoke test passes or fails its known way)."""
    path = MB.BASES["6e6f92d"].path
    assert path.is_file(), f"{path} is missing: it is recorded at 6e6f92d and must be committed"
    assert MB.baseline_label(path) == "6e6f92d"
    e6f, afa = MB.read_baseline(path), MB.read_baseline(MB.BASELINE)
    totals = re.search(r"^# Suite time: .* Totals: (.*), total=(\d+)$",
                       path.read_text(encoding="utf-8"), re.M)
    assert totals and int(totals.group(2)) == len(e6f)
    assert {k: int(v) for k, v in (kv.split("=") for kv in totals.group(1).split(", "))} == (
        Counter(r.outcome for r in e6f.values()))
    assert set(afa) <= set(e6f)
    phase3 = {p.relative_to(MB.REPO).as_posix()
              for p in (MB.REPO / "tests").rglob("test_p3_*.py")}
    assert phase3 and phase3 <= {nid.split("::")[0] for nid in e6f}

    def deterministic_failures(results):
        return {nid: r.signature for nid, r in results.items()
                if r.outcome in ("failed", "error") and nid not in MB.FLAKY}

    assert deterministic_failures(e6f) == deterministic_failures(afa)
    assert len(deterministic_failures(e6f)) == 5
    for nid, known in MB.FLAKY.items():
        assert e6f[nid].outcome == "passed" or (
            e6f[nid].outcome == "failed" and e6f[nid].signature in known), nid
