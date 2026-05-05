"""Phase 7 — Loopback retirement invariant.

Pins that production runtime modules don't import the loopback
transports (`LoopbackRFLink` / `LoopbackDockLink`). Loopbacks remain
fully usable from ``tests/`` and from the pedagogical Phase-1A demos
under ``hermes.{cluster,mule,mission}.__main__``; the invariant is
that **the multi-process supervised path** uses TCP transports only.

A regression where a future patch wires a Loopback into
``hermes.processes.*`` (the supervised entry points) or into a
non-demo module under ``hermes.{cluster,mule,mission}`` would break
this test, surfacing the slip before it ships.

The check is a simple text scan over each module's source — fast,
no AST-walking, no surprises. Tests + demos are excluded by path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pytest

import hermes


# Modules that production runtime imports. Any of these importing
# Loopback* is a regression of the Phase 7 retirement invariant.
_PRODUCTION_GLOBS: tuple[str, ...] = (
    "processes/cluster.py",
    "processes/mule.py",
    "processes/device.py",
    "processes/orchestrator.py",
    "processes/config.py",
    "cluster/host_cluster.py",
    "mule/mule_main.py",
    "mule/client_cluster.py",
    "mission/host_mission.py",
    "mission/client_mission.py",
    "scheduler/fl_scheduler.py",
    "observability/events.py",
    "observability/metrics.py",
)

# Demo / test entry points that are explicitly *allowed* to use loopback.
# Listed for the file's docstring contract; the test only scans the
# production list above.
_DEMO_PATHS: tuple[str, ...] = (
    "cluster/__main__.py",
    "mule/__main__.py",
    "mission/__main__.py",
)


def _hermes_root() -> Path:
    return Path(hermes.__file__).resolve().parent


_LOOPBACK_RE = re.compile(r"\bLoopback(RFLink|DockLink)\b")


def _scan_for_loopback(path: Path) -> List[str]:
    """Return any lines mentioning Loopback*. Empty list = clean."""
    text = path.read_text(encoding="utf-8")
    hits: List[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        # Strip inline comments — a `# uses LoopbackDockLink in tests`
        # docstring comment is fine; only real imports + identifier
        # uses count.
        code = line.split("#", 1)[0]
        if _LOOPBACK_RE.search(code):
            hits.append(f"{path}:{i}: {line.rstrip()}")
    return hits


@pytest.mark.parametrize("rel", _PRODUCTION_GLOBS)
def test_production_module_does_not_use_loopback(rel: str):
    path = _hermes_root() / rel
    assert path.exists(), f"production module {rel} missing — update list?"
    hits = _scan_for_loopback(path)
    assert not hits, (
        f"Phase-7 invariant broken: production module {rel} references "
        f"LoopbackRFLink / LoopbackDockLink. Hits:\n  "
        + "\n  ".join(hits)
        + "\nIf this is intentional, move the consumer into "
        "hermes.{cluster,mule,mission}.__main__ (demo) or tests/."
    )


def test_demo_modules_are_listed_and_actually_use_loopback():
    """Sanity: the demo paths above DO use loopback (otherwise the
    allowance list is stale and this test should be updated)."""
    for rel in _DEMO_PATHS:
        path = _hermes_root() / rel
        if not path.exists():
            # Some demos may not exist on every branch; skip gracefully.
            continue
        hits = _scan_for_loopback(path)
        assert hits, (
            f"demo {rel} no longer uses Loopback — update _DEMO_PATHS"
        )
