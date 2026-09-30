"""The full-suite pass/fail baseline and its comparer (``make_baseline.py``).

"The full suite passes" for Phase 3 is judged against ``pytest_baseline.txt``:
same outcome per node id, and the same failure signature for each failure
recorded at afa9526, so a known failure that starts failing for another
reason is reported rather than matched.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from tests.golden import make_baseline as MB

#: A test module that exists, so JUnit class names map to node ids.
CLS = "tests.golden.test_make_baseline"
NID = "tests/golden/test_make_baseline.py::"


def test_the_baseline_signs_every_known_failure():
    base = MB.read_baseline()
    assert len(base) >= 1500
    failed = {nid: r for nid, r in base.items() if r.outcome != "passed"}
    assert len(failed) == 6, sorted(failed)          # the afa9526 run (see its header)
    assert all(r.signature and " | " in r.signature for r in failed.values())
    assert all(r.signature is None for r in base.values() if r.outcome == "passed")


def _junit(path, cases):
    """Write a JUnit XML of ``(name, kind, message, text)`` test cases."""
    suite = ET.Element("testsuite", time="1.5")
    for name, kind, message, text in cases:
        tc = ET.SubElement(suite, "testcase", classname=CLS, name=name)
        if kind in ("failure", "error"):
            ET.SubElement(tc, kind, message=message).text = text
        elif kind == "skipped":
            ET.SubElement(tc, "skipped")
    root = ET.Element("testsuites")
    root.append(suite)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_signature_reads_where_and_why():
    quoted = ("AssertionError: \nassert 'MODE=hermes; ClientMission ready' in ''\n"
              " +  where '' = CompletedProcess(args=['python', 'x.py'], stderr='import y"
              + "\\n" + "ModuleNotFoundError: No module named \\'Config\\'" + "\\n" + "').stdout")
    text = ("    def test_m5():\n>       assert 'MODE=hermes; ClientMission ready' in proc.stdout\n"
            "E       AssertionError: \n\ntests\\unit\\test_mode_switch.py:146: AssertionError")
    assert MB.failure_signature(quoted, text) == (
        "tests/unit/test_mode_switch.py:146 | AssertionError | "
        "assert 'MODE=hermes; ClientMission ready' in '' | "
        "ModuleNotFoundError: No module named 'Config'")
    # No assert line: the message's first line says why.
    assert MB.failure_signature("KeyError: 'rounds_closed'", "tests\\x.py:9: KeyError") == (
        "tests/x.py:9 | KeyError | 'rounds_closed'")


def test_compare_reports_outcome_and_signature_changes(tmp_path):
    before = _junit(tmp_path / "before.xml", [
        ("test_a", "passed", "", ""),
        ("test_b", "failure", "AssertionError: {'row': 1}\nassert 0 >= 1",
         "E       assert 0 >= 1\n\ntests\\golden\\test_make_baseline.py:12: AssertionError"),
        ("test_c", "passed", "", ""),
        ("test_s", "skipped", "", ""),
    ])
    base, _ = MB.results_from_junit(before)
    assert base[NID + "test_b"] == MB.Result(
        "failed", "tests/golden/test_make_baseline.py:12 | AssertionError | assert 0 >= 1")
    assert base[NID + "test_s"] == MB.Result("skipped")
    # The file format round-trips.
    path = tmp_path / "baseline.txt"
    path.write_text(MB.render_baseline(base, suite_time=1.5, head=MB.BASE_COMMIT),
                    encoding="utf-8")
    assert MB.read_baseline(path) == base
    assert not MB.compare(base, base).differs

    # Later: a fails, b still fails but for another reason, c did not run, d is new.
    after = _junit(tmp_path / "after.xml", [
        ("test_a", "failure", "AssertionError: x\nassert 1 == 2",
         "tests\\golden\\test_make_baseline.py:5: AssertionError"),
        ("test_b", "failure", "KeyError: 'rounds_closed'",
         "tests\\golden\\test_make_baseline.py:11: KeyError"),
        ("test_s", "skipped", "", ""),
        ("test_d", "passed", "", ""),
    ])
    run, _ = MB.results_from_junit(after)
    c = MB.compare(base, run)
    assert c.outcome_changed == [(NID + "test_a", "passed", "failed")]
    assert [nid for nid, _was, _now in c.signature_changed] == [NID + "test_b"]
    assert c.missing == [NID + "test_c"]
    assert c.new == [(NID + "test_d", "passed")]
    assert c.differs
    text = MB.report(c, n_baseline=len(base), n_run=len(run))
    assert "DIFFERS" in text and "KeyError" in text
