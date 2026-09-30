"""Capture, or check, the FeRRy Phase 3 legacy goldens (unit UG).

The fixtures in ``tests/golden/data`` pin the behaviour of main at afa9526,
recorded before any Phase 3 change (Freeze Rule 1). They are regenerated only
on purpose, from that commit:

    py -3.11 tests/golden/make_goldens.py            # compare, write nothing
    py -3.11 tests/golden/make_goldens.py --write    # rewrite (afa9526, clean tree)
    py -3.11 tests/golden/make_goldens.py --only supervisor host_mission

``--write`` refuses unless HEAD is the base commit and ``hermes/`` and
``experiments/`` have no changes; ``--force`` overrides that, and is only right
when a golden itself was wrong (say so in the change that does it). The tests
never write a fixture.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.golden import (  # noqa: E402
    _build_channel,
    _build_feasibility,
    _build_topology,
    _canon,
    _host_harness,
    _mule_harness,
)

FIXTURES = {
    "channel": (_build_channel.build_cases,
                "exp4 channel: SNR traces, loss schedules, chosen bands, rf prior"),
    "feasibility": (_build_feasibility.build_cases,
                    "S3b, greedy walks, FedCS, FedEx diagnostics, in-flight check, Pass-2 walk"),
    "host_mission": (_host_harness.build_cases,
                     "HFLHostMission run_contact / deliver_contact scenarios, the late writer "
                     "and the sequential joins"),
    "supervisor": (_mule_harness.build_cases,
                   "MuleSupervisor end to end on loopback: H1, H2, D1, D4, Pass-2 budget, "
                   "S3c with an abort, K=2"),
    "topology": (_build_topology.build_cases,
                 "Exp 4 topologies, per-role JSON, driver rows on stub cells"),
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _tree_is_base() -> tuple:
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", "hermes", "experiments")
    return head == _canon.BASE_COMMIT and not dirty, head, dirty


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="rewrite the fixtures")
    ap.add_argument("--force", action="store_true",
                    help="allow --write away from a clean afa9526 tree")
    ap.add_argument("--only", nargs="*", choices=sorted(FIXTURES), default=None)
    args = ap.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Quiet the scenarios' own warnings. The host harness sets the level of the
    # logger it records (``hermes.mission``) itself, so this changes no case.
    logging.basicConfig(level=logging.ERROR, format="%(name)s: %(message)s")

    if args.write:
        ok, head, dirty = _tree_is_base()
        if not ok and not args.force:
            print(
                f"refusing to write: the goldens pin {_canon.BASE_COMMIT[:7]}, but HEAD is "
                f"{head[:7]}" + (f" with changes in hermes/ or experiments/:\n{dirty}" if dirty else "")
                + "\n(--force overrides; only for a golden that was itself wrong)",
                file=sys.stderr,
            )
            return 2

    status = 0
    for name in args.only or sorted(FIXTURES):
        build, what = FIXTURES[name]
        t0 = time.time()
        cases = build()
        took = time.time() - t0
        if args.write:
            path = _canon.dump(name, _canon.meta(name, what=what), cases)
            print(f"wrote {path.relative_to(REPO)}: {len(cases)} cases, "
                  f"{path.stat().st_size / 1024:.0f} KiB ({took:.1f} s)")
            continue
        try:
            golden = _canon.load(name)["cases"]
        except FileNotFoundError as e:
            print(f"{name}: {e}")
            status = 1
            continue
        bad = []
        for key, value in golden.items():
            problems = _canon.diff(value, cases.get(key, None) if key in cases else "<missing>")
            if problems:
                bad.append((key, problems))
        extra = sorted(set(cases) - set(golden))
        print(f"{name}: {len(golden)} golden cases, {len(bad)} differ, "
              f"{len(extra)} new ({took:.1f} s)")
        for key, problems in bad[:10]:
            print(f"  {key}:")
            for p in problems[:6]:
                print(f"    {p}")
        if bad:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
