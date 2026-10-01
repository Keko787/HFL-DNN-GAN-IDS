"""FeRRy Phase 4 (unit U3b): the recorded walk modules, loaded from git objects.

Freeze Rule 1 asks that ``member_admission="whole"``, the default, behave
exactly as the recorded pipeline. The live modules now carry the ``subset``
branch, so the reference is the code as recorded at :data:`COMMIT`: ``git show
6e6f92d:<path>`` of S3b, the budget walk, FedCS, MAX-AoI, Oort and Whittle,
each executed as a module of its own under a private name beside its live
package (``hermes.scheduler.policies._p4ref_6e6f92d_oort``, so its relative
imports resolve as the live module's do).

While a blob executes, ``sys.modules`` maps the live names it imports to the
reference modules already loaded: the reference budget walk prices with the
reference predicate, the reference policies walk with the reference budget
walk, and the reference Whittle reads the reference Oort's utility. The live
entries are restored as soon as the blob has run, so nothing leaks under a
live name. What else a blob imports is live: ``hermes.types`` (unchanged at
6e6f92d but for U0's additive plan types) and the selector's environment and
scope guard.

Used by ``tests/unit/test_p4_member_subset_hd.py``. :func:`load` skips the
calling test only when git or the commit is not available (a source archive
without history); a commit that is present but lacks one of the files is an
error, not a skip.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Dict, Mapping, Optional, Tuple

import pytest

#: The recorded pipeline: HEAD when Phase 4 began (Phase 4 spec, "Base"), the
#: commit the UG4 oracles were captured at.
COMMIT = "6e6f92d"
REPO = Path(__file__).resolve().parents[2]

#: ``(short name, module, the short names of the references it imports in
#: place of the live modules)``, in load order.
MODULES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("s3b", "hermes.scheduler.stages.s3b_feasibility", ()),
    ("budget_walk", "hermes.scheduler.policies.budget_walk", ("s3b",)),
    ("fedcs", "hermes.scheduler.policies.fedcs_degraded", ("s3b", "budget_walk")),
    ("max_aoi", "hermes.scheduler.policies.max_aoi", ("s3b", "budget_walk")),
    ("oort", "hermes.scheduler.policies.oort", ("s3b", "budget_walk")),
    ("whittle", "hermes.scheduler.policies.whittle", ("s3b", "budget_walk", "oort")),
)
LIVE_NAMES: Dict[str, str] = {short: dotted for short, dotted, _ in MODULES}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, timeout=120)


def unavailable() -> Optional[str]:
    """Why the reference cannot be loaded here, or None when it can."""
    if shutil.which("git") is None:
        return "git is not on PATH"
    if _git("cat-file", "-e", f"{COMMIT}^{{commit}}").returncode != 0:
        return f"commit {COMMIT} is not in this checkout"
    return None


def private_name(dotted: str) -> str:
    """The reference module's name: beside the live one, never equal to it."""
    package, _, leaf = dotted.rpartition(".")
    return f"{package}._p4ref_{COMMIT}_{leaf}"


def _execute(name: str, path: Path, overrides: Mapping[str, ModuleType]) -> ModuleType:
    """Run ``path`` as module ``name`` with ``overrides`` in ``sys.modules``.

    The module stays registered under its private name (dataclasses resolve
    their annotations through it); the overrides hold only while it runs.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    saved = {live: sys.modules.get(live) for live in overrides}
    sys.modules[name] = module
    sys.modules.update(overrides)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        for live, before in saved.items():
            if before is None:
                sys.modules.pop(live, None)
            else:
                sys.modules[live] = before
    return module


def load(directory: Path) -> SimpleNamespace:
    """The six modules at :data:`COMMIT`, their blobs written to ``directory``.

    Returns a namespace of modules keyed by short name (``s3b``,
    ``budget_walk``, ``fedcs``, ``max_aoi``, ``oort``, ``whittle``). Skips the
    calling test when :func:`unavailable` says why.
    """
    why = unavailable()
    if why is not None:
        pytest.skip(f"the {COMMIT} reference cannot be loaded: {why}")
    loaded: Dict[str, ModuleType] = {}
    try:
        for short, dotted, deps in MODULES:
            blob = _git("show", f"{COMMIT}:{dotted.replace('.', '/')}.py")
            if blob.returncode != 0:
                raise RuntimeError(
                    f"git show {COMMIT}:{dotted} failed: {blob.stderr.decode(errors='replace')}"
                )
            target = Path(directory) / f"{short}.py"
            target.write_bytes(blob.stdout)
            loaded[short] = _execute(private_name(dotted), target,
                                     {LIVE_NAMES[d]: loaded[d] for d in deps})
    except BaseException:
        unload(SimpleNamespace(**loaded))
        raise
    return SimpleNamespace(**loaded)


def unload(ref: SimpleNamespace) -> None:
    """Forget the reference modules' private names."""
    for module in vars(ref).values():
        if sys.modules.get(module.__name__) is module:
            del sys.modules[module.__name__]
