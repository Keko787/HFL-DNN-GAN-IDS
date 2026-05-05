"""Phase 6 / Sprint 1B — ``--mode {legacy,hermes}`` test bundle.

Implementation Plan §6.5 — the mode switch is a load-bearing rollback
path. These tests pin the contract so it can't quietly drift:

    M1: parse_HFL_Host_args([])            → args.mode == "legacy"
    M2: parse_HFL_Host_args(["--mode", "bogus"]) → SystemExit
    M3: subprocess HFLHost.py <smoke-args>             → "running Flower server"
    M4: subprocess HFLHost.py --mode hermes <smoke-args> → "HFLHostCluster ready"
    M5: same M3+M4 for TrainingClient.py
    M6: repo-wide grep — `import hermes` in HFLHost.py / TrainingClient.py
        appears ONLY inside the `if args.mode == "hermes":` branch.
    M7: CI matrix — both modes run on every PR. Documented for CI;
        not implemented as a unit test.

M3-M5 require flwr to actually launch the legacy script. We auto-skip
them when flwr is unavailable in the test env (the typical local case)
and rely on CI to exercise them. M1, M2, and M6 are fast file-level
checks that always run.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HFL_HOST_PATH = REPO_ROOT / "App" / "TrainingApp" / "HFLHost" / "HFLHost.py"
TRAINING_CLIENT_PATH = (
    REPO_ROOT / "App" / "TrainingApp" / "Client" / "TrainingClient.py"
)


def _flwr_available() -> bool:
    return importlib.util.find_spec("flwr") is not None


# Make sure the repo root is importable so we can pull in
# Config.SessionConfig.ArgumentConfigLoad without subprocessing.
sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# M1 — default mode is legacy
# --------------------------------------------------------------------------- #

def test_M1_HFL_host_default_mode_is_legacy(monkeypatch):
    """parse_HFL_Host_args() with no flags must set mode=='legacy'."""
    from Config.SessionConfig.ArgumentConfigLoad import parse_HFL_Host_args

    monkeypatch.setattr(sys, "argv", ["HFLHost.py"])
    args = parse_HFL_Host_args()
    assert args.mode == "legacy", (
        "Default mode silently changed — this is a load-bearing "
        "rollback flag and must default to 'legacy'."
    )


def test_M1_training_client_default_mode_is_legacy(monkeypatch):
    """parse_training_client_args() with no flags must set mode=='legacy'."""
    from Config.SessionConfig.ArgumentConfigLoad import parse_training_client_args

    monkeypatch.setattr(sys, "argv", ["TrainingClient.py"])
    args = parse_training_client_args()
    assert args.mode == "legacy", (
        "Default mode silently changed — this is a load-bearing "
        "rollback flag and must default to 'legacy'."
    )


# --------------------------------------------------------------------------- #
# M2 — invalid mode raises SystemExit (argparse choices enforcement)
# --------------------------------------------------------------------------- #

def test_M2_HFL_host_rejects_unknown_mode(monkeypatch):
    """argparse choices=['legacy', 'hermes'] must reject anything else."""
    from Config.SessionConfig.ArgumentConfigLoad import parse_HFL_Host_args

    monkeypatch.setattr(sys, "argv", ["HFLHost.py", "--mode", "bogus"])
    with pytest.raises(SystemExit):
        parse_HFL_Host_args()


def test_M2_training_client_rejects_unknown_mode(monkeypatch):
    from Config.SessionConfig.ArgumentConfigLoad import parse_training_client_args

    monkeypatch.setattr(sys, "argv", ["TrainingClient.py", "--mode", "bogus"])
    with pytest.raises(SystemExit):
        parse_training_client_args()


def test_M2_HFL_host_accepts_both_modes_explicitly(monkeypatch):
    from Config.SessionConfig.ArgumentConfigLoad import parse_HFL_Host_args

    monkeypatch.setattr(sys, "argv", ["HFLHost.py", "--mode", "legacy"])
    args = parse_HFL_Host_args()
    assert args.mode == "legacy"

    monkeypatch.setattr(sys, "argv", ["HFLHost.py", "--mode", "hermes"])
    args = parse_HFL_Host_args()
    assert args.mode == "hermes"


# --------------------------------------------------------------------------- #
# M3 / M4 — subprocess HFLHost.py in both modes (skipped when flwr missing)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.skipif(not _flwr_available(), reason="flwr not installed in test env")
def test_M3_HFL_host_legacy_subprocess_logs_flower_banner():
    """python HFLHost.py (no --mode) launches the legacy Flower server."""
    proc = _spawn_smoke(HFL_HOST_PATH, extra_args=[])
    assert "MODE=legacy; running Flower server" in proc.stdout, proc.stdout


@pytest.mark.slow
@pytest.mark.skipif(not _flwr_available(), reason="flwr not installed in test env")
def test_M4_HFL_host_hermes_subprocess_logs_cluster_banner():
    """python HFLHost.py --mode hermes launches HFLHostCluster path."""
    proc = _spawn_smoke(HFL_HOST_PATH, extra_args=["--mode", "hermes"])
    assert "MODE=hermes; HFLHostCluster ready on dock" in proc.stdout, proc.stdout


# --------------------------------------------------------------------------- #
# M5 — same M3+M4 for TrainingClient.py
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.skipif(not _flwr_available(), reason="flwr not installed in test env")
def test_M5_training_client_legacy_subprocess_logs_flower_banner():
    proc = _spawn_smoke(TRAINING_CLIENT_PATH, extra_args=[])
    assert "MODE=legacy; running Flower client" in proc.stdout, proc.stdout


@pytest.mark.slow
@pytest.mark.skipif(not _flwr_available(), reason="flwr not installed in test env")
def test_M5_training_client_hermes_subprocess_logs_mission_banner():
    proc = _spawn_smoke(TRAINING_CLIENT_PATH, extra_args=["--mode", "hermes"])
    assert "MODE=hermes; ClientMission ready" in proc.stdout, proc.stdout


def _spawn_smoke(script_path: Path, extra_args):
    """Run a script with minimal smoke args; return CompletedProcess.

    Uses a short timeout so a misbehaving long-running mode banner test
    doesn't wedge CI. Real e2e is in Sprint 2's integration suite.
    """
    cmd = [sys.executable, str(script_path), *extra_args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=15.0,
        cwd=str(REPO_ROOT),
    )


# --------------------------------------------------------------------------- #
# M6 — repo-wide grep: `import hermes` only inside the guarded branch
# --------------------------------------------------------------------------- #

# Match `import hermes` and `from hermes.x import y` styles.
_HERMES_IMPORT_RE = re.compile(r"^\s*(import\s+hermes|from\s+hermes\b)", re.MULTILINE)
_GUARD_RE = re.compile(r"if\s+args\.mode\s*==\s*[\"']hermes[\"']\s*:")


def _hermes_imports_outside_guard(file_path: Path) -> list[str]:
    """Return import lines that appear *outside* an `if args.mode == "hermes":` block.

    Heuristic: an import is "guarded" if it lives inside a function whose
    body is reached only via the hermes branch — i.e., the function is
    called from inside a `_run_hermes_main`-style helper, OR the import
    line is itself indented inside a guard. We approximate by checking
    the guard occurrence and the function-scope of the import.

    Implementation: find every `import hermes`/`from hermes` line, then
    walk back through the file scanning for the nearest enclosing
    `def _run_hermes_main` block. If the import is inside such a block,
    it's guarded. Otherwise it's an unguarded leak.
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    leaks: list[str] = []
    for m in _HERMES_IMPORT_RE.finditer(text):
        # Line number (1-based) of this import.
        line_no = text.count("\n", 0, m.start()) + 1
        # Walk backwards looking for a `def _run_hermes_main(` declaration
        # before any other `def ...:` at column 0.
        guarded = False
        for i in range(line_no - 1, -1, -1):
            stripped = lines[i].lstrip()
            indent = len(lines[i]) - len(stripped)
            if indent == 0 and stripped.startswith("def "):
                if stripped.startswith("def _run_hermes_main"):
                    guarded = True
                break
        if not guarded:
            leaks.append(f"{file_path.name}:{line_no}: {lines[line_no - 1].strip()}")
    return leaks


def test_M6_HFL_host_hermes_imports_are_guarded():
    """No `import hermes` outside the _run_hermes_main function in HFLHost.py."""
    leaks = _hermes_imports_outside_guard(HFL_HOST_PATH)
    assert not leaks, (
        "HFLHost.py imports hermes.* outside the mode-hermes guard — "
        "this breaks the legacy rollback path:\n  " + "\n  ".join(leaks)
    )


def test_M6_training_client_hermes_imports_are_guarded():
    """No `import hermes` outside _run_hermes_main in TrainingClient.py."""
    leaks = _hermes_imports_outside_guard(TRAINING_CLIENT_PATH)
    assert not leaks, (
        "TrainingClient.py imports hermes.* outside the mode-hermes "
        "guard — this breaks the legacy rollback path:\n  "
        + "\n  ".join(leaks)
    )


def test_M6_grep_helper_catches_unguarded_imports(tmp_path):
    """Sanity check: the guard-detector flags a deliberately-unguarded import."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import hermes.cluster\n"
        "def main():\n"
        "    pass\n",
        encoding="utf-8",
    )
    leaks = _hermes_imports_outside_guard(bad)
    assert len(leaks) == 1
    assert "import hermes.cluster" in leaks[0]


def test_M6_grep_helper_passes_guarded_imports(tmp_path):
    """Sanity check: the guard-detector accepts an import inside _run_hermes_main."""
    good = tmp_path / "good.py"
    good.write_text(
        "def main():\n"
        "    if args.mode == 'hermes':\n"
        "        _run_hermes_main(args)\n"
        "\n"
        "def _run_hermes_main(args):\n"
        "    from hermes.cluster import HFLHostCluster\n"
        "    HFLHostCluster()\n",
        encoding="utf-8",
    )
    leaks = _hermes_imports_outside_guard(good)
    assert leaks == []
