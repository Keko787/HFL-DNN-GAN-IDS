"""Record, or compare against, the full-suite pass/fail baselines (units UG, UG4).

A baseline records the suite as it ran at one commit on this host: one
``<outcome> <node id>`` line per test and, under each failing test, an
indented ``signature:`` line saying where and why it failed (the failing
line, the exception type, the first ``assert`` line of the message or else
its first line, and any exception the message quotes, such as a subprocess's
``ModuleNotFoundError``). "The full suite passes" means: the same outcome per
node id, the same signature for each known failure, new tests allowed. There
are two (``BASES``):

* ``pytest_baseline.txt``, afa9526, before any Phase 3 change (unit UG);
* ``pytest_baseline_6e6f92d.txt``, 6e6f92d, before any Phase 4 change (unit
  UG4), so that Phase 3's own tests, new against afa9526 and so never gated
  by it, are gated too.

    py -3.11 -m pytest tests -p no:cacheprovider -q -rfE --junitxml=<run.xml>
    py -3.11 tests/golden/make_baseline.py compare <run.xml>     # every baseline
    py -3.11 tests/golden/make_baseline.py compare <run.xml> --base 6e6f92d
    py -3.11 tests/golden/make_baseline.py write <run.xml> --base 6e6f92d  # at 6e6f92d only

``compare`` lists, per baseline, the tests whose outcome changed, the known
failures whose signature changed, the baseline tests that did not run, the
allow-listed flaky tests that changed (not differences) and the new tests (not
differences, but the failing ones are listed); it exits 1 if anything differs
from any baseline compared, and 2 if a baseline it was to compare with is
missing (every recorded one by default: a baseline file left out of a commit
must not switch its gate off in silence). Run the suite with the flags above: pytest
shortens assertion messages by verbosity, and the signatures are read from
those messages. ``write`` records the baseline of ``--base`` (afa9526 by
default) and refuses unless HEAD is that commit and ``hermes/`` and
``experiments/`` are unchanged (``--force`` overrides).

Flaky tests (``FLAKY``, extended with ``--flaky``, switched off with
``--strict``) may pass or fail from run to run; when one fails it must fail
its known way (a signature listed for it, or the one its baseline recorded),
so a new failure of a flaky test is still a difference. Only
``test_exp4_real_model_synthetic_converges`` is listed: it fails under load
(``rounds_closed == 0``), so it flipped the afa9526 comparison to DIFFERS on
its own (FeRRy Phase 4 harness map, ``compare_full_run2.txt``); the user
signed that baseline off with it on 2026-09-29, and its fix waits for the
session-TTL pilot.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = HERE / "pytest_baseline.txt"
BASE_COMMIT = "afa952682c1e8a30160a390397f7f369a898b584"
COMMAND = "py -3.11 -m pytest tests -p no:cacheprovider -q -rfE --junitxml=<run.xml>"
SIGNATURE_PREFIX = "    signature: "
#: ``compare``'s exit status when a baseline it was to compare with is missing
#: (1 means the run differs; write's refusal is 2 as well).
MISSING_STATUS = 2

#: A test that may pass or fail from run to run -> the signatures it is known
#: to fail with. The real-model smoke test fails under load with no round
#: closed (its afa9526 signature, recorded in ``pytest_baseline.txt``).
FLAKY: Dict[str, Tuple[str, ...]] = {
    "tests/integration/test_exp4_realmodel_smoke.py::test_exp4_real_model_synthetic_converges": (
        "tests/integration/test_exp4_realmodel_smoke.py:52 | AssertionError | assert 0 >= 1",
    ),
}

#: What was learnt about the failures of the afa9526 run (kept in the header).
AFA9526_NOTES = (
    "test_contact_selector_ab.py DoD cell [60.0]: the selector's decision-dense cell misses its DoD.",
    "four test_mode_switch.py subprocess tests: the spawned legacy scripts import 'Config', which is",
    "  not on the path (ModuleNotFoundError), so the banner never prints.",
    "test_exp4_realmodel_smoke.py::test_exp4_real_model_synthetic_converges: rounds_closed == 0;",
    "  it failed in the full run and again when re-run alone, both at afa9526.",
    "  The unit spec expected five failures; the user signed this sixth off on 2026-09-29: the",
    "  test has since been judged flaky under load, and its fix waits for the session-TTL pilot.",
)

#: What is known about the failures of the 6e6f92d run (kept in the header).
E6F92D_NOTES = (
    "the five deterministic afa9526 failures, with their afa9526 signatures: the selector's",
    "  decision-dense DoD cell [60.0] (test_contact_selector_ab.py), and the four",
    "  test_mode_switch.py subprocess tests (the legacy scripts import 'Config', not on the path).",
    "test_exp4_realmodel_smoke.py::test_exp4_real_model_synthetic_converges is flaky under load",
    "  (FLAKY in make_baseline.py): compare lets it pass or fail its known way (see below).",
)


class Base(NamedTuple):
    """One recorded baseline: the commit it ran at and how its file is headed."""

    commit: str                        # the full SHA HEAD must be at to write it
    path: Path
    unit: str                          # the unit that recorded it, for the header
    phase: str                         # the phase it was recorded before
    notes: Tuple[str, ...]


BASES: Dict[str, Base] = {
    "afa9526": Base(BASE_COMMIT, BASELINE, "FeRRy Phase 3 unit UG", "Phase 3", AFA9526_NOTES),
    "6e6f92d": Base("6e6f92da038227147489515d876cc3f353584283",
                    HERE / "pytest_baseline_6e6f92d.txt", "FeRRy Phase 4 unit UG4", "Phase 4",
                    E6F92D_NOTES),
}


def base_of(head: str) -> Optional[Base]:
    """The recorded base whose commit is ``head`` (a full SHA or a prefix of one)."""
    head = head.strip().lower()
    for base in BASES.values():
        if head and base.commit.startswith(head):
            return base
    return None


class Result(NamedTuple):
    outcome: str                       # passed | failed | error | skipped
    signature: Optional[str] = None    # failures and errors only


# --------------------------------------------------------------------------- #
# JUnit XML -> results
# --------------------------------------------------------------------------- #

_LOCATION = re.compile(r"^(?P<path>\S.*?\.py):(?P<line>\d+): (?P<exc>[A-Za-z_][\w.]*)\s*$")
_QUOTED_EXC = re.compile(r"([A-Z]\w*(?:Error|Exception)): (.*?)(?=\\n|\n|$)")


def node_id(classname: str, name: str, repo: Path = REPO) -> str:
    """``tests.unit.test_x.TestC`` + ``test_y`` -> ``tests/unit/test_x.py::TestC::test_y``."""
    parts = classname.split(".")
    for i in range(len(parts), 0, -1):
        if (repo.joinpath(*parts[:i]).with_suffix(".py")).is_file():
            rest = "".join("::" + r for r in parts[i:])
            return "/".join(parts[:i]) + ".py" + rest + "::" + name
    raise ValueError(f"cannot map {classname}::{name} to a test file under {repo}")


def failure_signature(message: str, text: str) -> str:
    """Where and why a test failed, stable across runs of the same failure."""
    message = message or ""
    lines = [ln.strip() for ln in message.splitlines() if ln.strip()]
    head = lines[0] if lines else ""
    exc, _, rest = head.partition(":")
    if not re.fullmatch(r"[A-Za-z_][\w.]*", exc.strip()):
        exc, rest = "?", head
    exc = exc.strip()
    why = next((ln for ln in lines if ln.startswith("assert ")), rest.strip() or head)
    where = "?"
    for ln in reversed((text or "").splitlines()):
        m = _LOCATION.match(ln.strip())
        if m:
            where = f"{m.group('path').replace(os.sep, '/').replace(chr(92), '/')}:{m.group('line')}"
            break
    quoted = []
    for name, detail in _QUOTED_EXC.findall(message):
        item = f"{name}: {detail.replace(chr(92) + chr(39), chr(39)).strip()}"
        if name != exc and item not in quoted:
            quoted.append(item)
    sig = " | ".join([where, exc, why] + quoted)
    return sig if len(sig) <= 400 else sig[:397] + "..."


def results_from_junit(path: Path, repo: Path = REPO) -> Tuple[Dict[str, Result], float]:
    tree = ET.parse(str(path))
    out: Dict[str, Result] = {}
    for tc in tree.iter("testcase"):
        nid = node_id(tc.get("classname", ""), tc.get("name", ""), repo)
        result = Result("passed")
        for child in tc:
            if child.tag in ("failure", "error"):
                result = Result("failed" if child.tag == "failure" else "error",
                                failure_signature(child.get("message", ""), child.text or ""))
                break
            if child.tag == "skipped":
                result = Result("skipped")
        out[nid] = result
    suite = next(tree.iter("testsuite"), None)
    return out, float(suite.get("time", "0")) if suite is not None else 0.0


# --------------------------------------------------------------------------- #
# The baseline file
# --------------------------------------------------------------------------- #

def read_baseline(path: Path = BASELINE) -> Dict[str, Result]:
    out: Dict[str, Result] = {}
    last: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith(SIGNATURE_PREFIX):
            if last is None:
                raise ValueError(f"{path}: a signature line before any test")
            out[last] = out[last]._replace(signature=raw[len(SIGNATURE_PREFIX):])
            continue
        if not raw.strip() or raw.startswith("#"):
            continue
        outcome, _, nid = raw.partition(" ")
        out[nid] = Result(outcome)
        last = nid
    return out


def render_baseline(results: Dict[str, Result], *, suite_time: float, head: str) -> str:
    """The baseline file for ``results``, headed for the base ``head`` is.

    The afa9526 header is the recorded one, line for line. A later base also
    lists the allow-listed flaky tests with their outcome in this run.
    """
    base = base_of(head) or Base(head, BASELINE, "FeRRy", "this", ())
    counts = Counter(r.outcome for r in results.values())
    lines = [
        f"# {base.unit} - pytest baseline at {head[:7]} (main), before any {base.phase} change.",
        f"# Command: {COMMAND}",
        f"#   (PYTHONIOENCODING=utf-8, Windows 11, Python {platform.python_version()}); written by "
        "tests/golden/make_baseline.py.",
        f"# Suite time: {suite_time:.1f} s. Totals: "
        + ", ".join(f"{k}={counts[k]}" for k in sorted(counts)) + f", total={len(results)}",
        "# Format: '<outcome> <node id>', sorted by node id; under a failure, its signature:",
        "#   <file:line> | <exception> | <first assert line, or first message line> | <quoted exceptions>",
        f"# 'The full suite passes' for {base.phase}: same outcome per node id and same signature per",
        "# known failure (tests/golden/make_baseline.py compare <run.xml>); new tests allowed.",
        f"# Known failures at {head[:7]} on this host:",
    ]
    lines += [f"#   {note}" for note in base.notes]
    if base.commit != BASE_COMMIT:
        lines.append("# Flaky (make_baseline.py FLAKY; compare lets each pass or fail its known way),")
        lines.append("#   with the outcome of this run:")
        lines += [f"#   {results[nid].outcome} {nid}" for nid in sorted(FLAKY) if nid in results]
    for nid in sorted(results):
        r = results[nid]
        lines.append(f"{r.outcome} {nid}")
        if r.signature is not None:
            lines.append(SIGNATURE_PREFIX + r.signature)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

class Comparison(NamedTuple):
    outcome_changed: List[Tuple[str, str, str]]
    signature_changed: List[Tuple[str, str, str]]
    missing: List[str]
    new: List[Tuple[str, str]]
    #: Allow-listed flaky tests whose result changed in their known way:
    #: ``(node id, baseline outcome, run outcome)``. Not a difference.
    flaky: Sequence[Tuple[str, str, str]] = ()

    @property
    def differs(self) -> bool:
        return bool(self.outcome_changed or self.signature_changed or self.missing)


def _flaky_ok(want: Result, got: Result, known: Sequence[str]) -> bool:
    """Whether a flaky test's result is one it may have: a pass, or a failure
    with a signature known for it (``known``, or the one its baseline
    recorded). A test declared flaky with no known signature may fail any way."""
    if got.outcome == "passed":
        return True
    if got.outcome != "failed":
        return False
    allowed = set(known) | ({want.signature} if want.signature else set())
    return not allowed or got.signature in allowed


def compare(baseline: Dict[str, Result], run: Dict[str, Result],
            flaky: Optional[Mapping[str, Sequence[str]]] = None) -> Comparison:
    """``run`` against ``baseline``; ``flaky`` maps each allow-listed test to
    its known failure signatures (none by default, the recorded rule)."""
    flaky = {} if flaky is None else flaky
    outcome_changed, signature_changed, missing, flipped = [], [], [], []
    for nid, want in sorted(baseline.items()):
        got = run.get(nid)
        if got is None:
            missing.append(nid)
        elif nid in flaky and _flaky_ok(want, got, flaky[nid]):
            if got != want:
                flipped.append((nid, want.outcome, got.outcome))
        elif got.outcome != want.outcome:
            outcome_changed.append((nid, want.outcome, got.outcome))
        elif want.signature is not None and got.signature != want.signature:
            signature_changed.append((nid, want.signature, got.signature or "(none)"))
    new = [(nid, r.outcome) for nid, r in sorted(run.items()) if nid not in baseline]
    return Comparison(outcome_changed, signature_changed, missing, new, flipped)


def report(c: Comparison, *, n_baseline: int, n_run: int, label: str = "afa9526") -> str:
    out = [f"baseline: {n_baseline} tests; this run: {n_run} tests"]
    out.append(f"outcome changed: {len(c.outcome_changed)}")
    out += [f"  {was} -> {now}  {nid}" for nid, was, now in c.outcome_changed]
    out.append(f"known failure, another signature: {len(c.signature_changed)}")
    for nid, was, now in c.signature_changed:
        out += [f"  {nid}", f"    was: {was}", f"    now: {now}"]
    out.append(f"baseline tests that did not run: {len(c.missing)}")
    out += [f"  {nid}" for nid in c.missing]
    out.append(f"flaky tests that changed (allowed): {len(c.flaky)}")
    out += [f"  {was} -> {now}  {nid}" for nid, was, now in c.flaky]
    new_bad = [(nid, o) for nid, o in c.new if o in ("failed", "error")]
    out.append(f"new tests (allowed): {len(c.new)}, of which failing: {len(new_bad)}")
    out += [f"  {o}  {nid}" for nid, o in new_bad]
    out.append(f"DIFFERS from the {label} baseline" if c.differs
               else f"same as the {label} baseline")
    return "\n".join(out)


def baseline_label(path: Path) -> str:
    """The commit a baseline file says it was recorded at (its first line), else its name."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            first = fh.readline()
    except OSError:
        return path.name
    m = re.search(r"pytest baseline at ([0-9a-f]{7,40})", first)
    return m.group(1)[:7] if m else path.name


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("compare", "write"))
    ap.add_argument("junit_xml", type=Path)
    ap.add_argument("--base", choices=sorted(BASES), default=None,
                    help="the baseline to compare with (default: every recorded one) or to "
                         "write (default: afa9526)")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="a baseline file to use instead of --base's")
    ap.add_argument("--flaky", action="append", default=[], metavar="NODE_ID",
                    help="compare: also let this test pass or fail (repeatable; FLAKY is "
                         "always included unless --strict)")
    ap.add_argument("--strict", action="store_true",
                    help="compare: no flaky allow-list; every outcome change differs")
    ap.add_argument("--force", action="store_true",
                    help="write away from a clean tree at the base commit (only for a baseline "
                         "that was wrong)")
    args = ap.parse_args(argv)
    run, suite_time = results_from_junit(args.junit_xml)
    if args.action == "compare":
        # (the BASES key, when the baseline is a recorded one; its file)
        if args.baseline is not None:
            chosen: List[Tuple[Optional[str], Path]] = [(None, args.baseline)]
        elif args.base is not None:
            chosen = [(args.base, BASES[args.base].path)]
        else:
            chosen = [(key, b.path) for key, b in BASES.items()]
        flaky: Dict[str, Tuple[str, ...]] = {} if args.strict else dict(FLAKY)
        for nid in args.flaky:
            flaky.setdefault(nid, ())
        status = 0
        for i, (key, path) in enumerate(chosen):
            present = path.is_file()
            label = baseline_label(path) if present else (key or path.name)
            if len(chosen) > 1:
                print(("" if i == 0 else "\n") + f"== the {label} baseline ({path.name})")
            if not present:
                print(f"MISSING: {path} does not exist, so nothing was compared with the "
                      f"{label} baseline")
                status = max(status, MISSING_STATUS)
                continue
            base = read_baseline(path)
            c = compare(base, run, flaky)
            print(report(c, n_baseline=len(base), n_run=len(run), label=label))
            status = max(status, 1 if c.differs else 0)
        return status
    base = BASES[args.base or "afa9526"]
    out_path = args.baseline or base.path
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", "hermes", "experiments")
    if (head != base.commit or dirty) and not args.force:
        print(f"refusing to write: the baseline is {base.commit[:7]}'s, but HEAD is {head[:7]}"
              + (" with changes in hermes/ or experiments/" if dirty else ""), file=sys.stderr)
        return 2
    out_path.write_text(render_baseline(run, suite_time=suite_time, head=head),
                        encoding="utf-8", newline="\n")
    counts = Counter(r.outcome for r in run.values())
    print(f"wrote {out_path}: {len(run)} tests, {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
