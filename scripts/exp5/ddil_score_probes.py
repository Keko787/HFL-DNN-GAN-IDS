"""Score the DDIL probes' kept traces in parallel, with launch.py's own job runner.

``loss_probes.py <probe> --score`` runs the trace scorer one file at a time and writes no
argv sidecar, so a later ``--score`` scores the same files again. This builds each flight's
jobs and score plan as that does (with the flight's --set changes, so the DDIL probes score
with ``--ddil-columns`` into ``_scored_ddil.csv``; DeveloperDocs/FeRRy_DDIL_Unified_Plan.md,
build D0), keeps the scored files the flight's own entries read that launch.py's ``done()``
does not find current, and runs them with launch.py's ``cmd_run``: in parallel, an argv
sidecar beside each scored file, and a manifest under the out-root's ``scores/``. A scored
file two flights share is scored once. It writes no study comparisons, so no out-root's
recorded score files move. Write the probe reports after it with ``loss_probes.py <probe>``
(without ``--score``).

Usage::

    py -3.11 scripts/exp5/ddil_score_probes.py --plan DDIL1 DDILFQ1 DDIL2   # what it would score
    py -3.11 scripts/exp5/ddil_score_probes.py DDIL1 DDILFQ1 DDIL2 [--allow-dirty] [--max-jobs N]
    py -3.11 scripts/exp5/loss_probes.py DDIL1 DDILFQ1 DDIL2

Added 17 Nov 2026, when step 1, FQ-s1 and step 2 were scored with it (190 files).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import loss_probes as LP  # noqa: E402


@dataclass
class FlightScoring:
    """One flight's scoring: its entries, the scored files they read, and the runs to make."""

    probe: str
    flight: Any                # loss_probes.Flight
    settings: Any              # launch.py Settings, with the flight's --set changes
    params_text: str
    entries: int
    scored_files: int
    todo: List[Any] = field(default_factory=list)     # launch.py score Jobs


def plan(names: Sequence[str], L=None) -> List[FlightScoring]:
    """Each flight of each probe in ``names``, in order, with the score jobs it needs; a
    scored file an earlier flight already scores is left to that flight."""
    L = L or LP._launcher()
    seen: Set[str] = set()
    out: List[FlightScoring] = []
    for name in names:
        for fl in LP.PROBES[name].flights:
            s, text = L.load_settings(L.PARAMS)
            ns = argparse.Namespace(trials=fl.trials, set=list(fl.sets), dataset=None,
                                    **{o: None for o, _, _ in L.LEVERS})
            with contextlib.redirect_stdout(io.StringIO()):
                L.apply_levers(s, ns)
                jobs = L.build(fl.stage, s, None, fl.out_root)
                runs, entries, _, _ = L.score_plan(fl.stage, s, jobs)
            mine = [e for e in entries.get(fl.study, []) if fl.arms is None or e.arm in fl.arms]
            wanted = {str(e.scored) for e in mine}
            todo = [j for j in runs if str(L.REPO / j.out) in wanted and j.out not in seen
                    and not L.done(j)]
            seen.update(j.out for j in todo)
            out.append(FlightScoring(name, fl, s, text, len(mine), len(wanted), todo))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probes", nargs="+", help=f"probes to score ({', '.join(LP.PROBES)})")
    ap.add_argument("--plan", action="store_true", help="say what it would score; run nothing")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="score with uncommitted changes under hermes/ or experiments/ "
                         "(each manifest lists them)")
    ap.add_argument("--max-jobs", type=int, default=None,
                    help="at most this many scorer runs at once (default: the launcher's)")
    a = ap.parse_args(argv)
    unknown = [p for p in a.probes if p not in LP.PROBES]
    if unknown:
        ap.error(f"not a probe: {', '.join(unknown)}")
    L = LP._launcher()
    rc = 0
    for f in plan(a.probes, L):
        print(f"=== {f.probe} / {f.flight.label or '(main)'}: {f.entries} entries, "
              f"{f.scored_files} scored files, {len(f.todo)} to score", flush=True)
        if a.plan or not f.todo:
            continue
        rc = L.cmd_run(f"score-{f.flight.stage}", f.settings, f.params_text, f.todo,
                       allow_dirty=a.allow_dirty, skip_blocked=True, yes=True,
                       max_jobs=a.max_jobs, out_root=f.flight.out_root) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
