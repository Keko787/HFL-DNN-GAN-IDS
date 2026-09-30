"""Record, or compare against, the full-suite pass/fail baseline (unit UG).

``pytest_baseline.txt`` records the suite as it ran at afa9526 on this host,
before any Phase 3 change: one ``<outcome> <node id>`` line per test and,
under each failing test, an indented ``signature:`` line saying where and why
it failed (the failing line, the exception type, the first ``assert`` line of
the message or else its first line, and any exception the message quotes,
such as a subprocess's ``ModuleNotFoundError``). "The full suite passes" for
Phase 3 means: the same outcome per node id, the same signature for each known
failure, new tests allowed.

    py -3.11 -m pytest tests -p no:cacheprovider -q --junitxml=<run.xml>
    py -3.11 tests/golden/make_baseline.py compare <run.xml>
    py -3.11 tests/golden/make_baseline.py write <run.xml>    # at afa9526 only

``compare`` lists the tests whose outcome changed, the known failures whose
signature changed, the baseline tests that did not run, and the new tests (not
differences); it exits 1 if anything differs. Run the suite with the flags
above: pytest shortens assertion messages by verbosity, and the signatures
are read from those messages. ``write`` refuses unless HEAD is the base commit
and ``hermes/`` and ``experiments/`` are unchanged (``--force`` overrides).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "pytest_baseline.txt"
BASE_COMMIT = "afa952682c1e8a30160a390397f7f369a898b584"
COMMAND = "py -3.11 -m pytest tests -p no:cacheprovider -q -rfE --junitxml=<run.xml>"
SIGNATURE_PREFIX = "    signature: "

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
    counts = Counter(r.outcome for r in results.values())
    lines = [
        f"# FeRRy Phase 3 unit UG - pytest baseline at {head[:7]} (main), before any Phase 3 change.",
        f"# Command: {COMMAND}",
        "#   (PYTHONIOENCODING=utf-8, Windows 11, Python 3.11.9); written by tests/golden/make_baseline.py.",
        f"# Suite time: {suite_time:.1f} s. Totals: "
        + ", ".join(f"{k}={counts[k]}" for k in sorted(counts)) + f", total={len(results)}",
        "# Format: '<outcome> <node id>', sorted by node id; under a failure, its signature:",
        "#   <file:line> | <exception> | <first assert line, or first message line> | <quoted exceptions>",
        "# 'The full suite passes' for Phase 3: same outcome per node id and same signature per",
        "# known failure (tests/golden/make_baseline.py compare <run.xml>); new tests allowed.",
        "# Known failures at afa9526 on this host:",
    ]
    lines += [f"#   {note}" for note in AFA9526_NOTES]
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

    @property
    def differs(self) -> bool:
        return bool(self.outcome_changed or self.signature_changed or self.missing)


def compare(baseline: Dict[str, Result], run: Dict[str, Result]) -> Comparison:
    outcome_changed, signature_changed, missing = [], [], []
    for nid, want in sorted(baseline.items()):
        got = run.get(nid)
        if got is None:
            missing.append(nid)
        elif got.outcome != want.outcome:
            outcome_changed.append((nid, want.outcome, got.outcome))
        elif want.signature is not None and got.signature != want.signature:
            signature_changed.append((nid, want.signature, got.signature or "(none)"))
    new = [(nid, r.outcome) for nid, r in sorted(run.items()) if nid not in baseline]
    return Comparison(outcome_changed, signature_changed, missing, new)


def report(c: Comparison, *, n_baseline: int, n_run: int) -> str:
    out = [f"baseline: {n_baseline} tests; this run: {n_run} tests"]
    out.append(f"outcome changed: {len(c.outcome_changed)}")
    out += [f"  {was} -> {now}  {nid}" for nid, was, now in c.outcome_changed]
    out.append(f"known failure, another signature: {len(c.signature_changed)}")
    for nid, was, now in c.signature_changed:
        out += [f"  {nid}", f"    was: {was}", f"    now: {now}"]
    out.append(f"baseline tests that did not run: {len(c.missing)}")
    out += [f"  {nid}" for nid in c.missing]
    new_bad = [(nid, o) for nid, o in c.new if o in ("failed", "error")]
    out.append(f"new tests (allowed): {len(c.new)}, of which failing: {len(new_bad)}")
    out += [f"  {o}  {nid}" for nid, o in new_bad]
    out.append("DIFFERS from the afa9526 baseline" if c.differs else "same as the afa9526 baseline")
    return "\n".join(out)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("compare", "write"))
    ap.add_argument("junit_xml", type=Path)
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--force", action="store_true",
                    help="write away from a clean afa9526 tree (only for a baseline that was wrong)")
    args = ap.parse_args(argv)
    run, suite_time = results_from_junit(args.junit_xml)
    if args.action == "compare":
        base = read_baseline(args.baseline)
        c = compare(base, run)
        print(report(c, n_baseline=len(base), n_run=len(run)))
        return 1 if c.differs else 0
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", "hermes", "experiments")
    if (head != BASE_COMMIT or dirty) and not args.force:
        print(f"refusing to write: the baseline is afa9526's, but HEAD is {head[:7]}"
              + (" with changes in hermes/ or experiments/" if dirty else ""), file=sys.stderr)
        return 2
    args.baseline.write_text(render_baseline(run, suite_time=suite_time, head=head),
                             encoding="utf-8", newline="\n")
    counts = Counter(r.outcome for r in run.values())
    print(f"wrote {args.baseline}: {len(run)} tests, {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
