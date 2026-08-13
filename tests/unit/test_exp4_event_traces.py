"""Event-trace retention (`--keep-event-traces`).

Why this exists: the per-contact record lives in the orchestrator's run dir,
`consume_run_dir` folds it into aggregates, and teardown deletes it. So a
finished sweep cannot be re-scored against a new scheduling baseline — there is
nothing left to replay, and answering "how would policy X have done?" costs a
full re-run. Retention is the cheap insurance against that.

The naming is where the real failure modes are. Cell ids are built from the grid
axes and contain `|` and `=` (`N=6|dead_zone=0.0|regime=jittery`), so a directory
named after one is unopenable on Windows; and two trials sharing a directory
would interleave their events and quietly corrupt both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.exp4.driver import Exp4Driver, trace_dir_name


class _Cell:
    def __init__(self, cell_id, arm="H1", trial_index=0, seed=42):
        self.cell_id = cell_id
        self.arm = arm
        self.trial_index = trial_index
        self.seed = seed


REAL_CELL_ID = "N=6|dead_zone=0.0|link_quality=0.3|n_missions=4|regime=jittery|rrf=60.0"


# --------------------------------------------------------------------------- #
# 1. The name must be usable as a path
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ch", list('<>:"/\\|?*'))
def test_windows_illegal_characters_are_removed(ch):
    name = trace_dir_name(_Cell(f"a{ch}b"))
    assert ch not in name


def test_a_real_cell_id_survives_sanitisation():
    """The pipe characters in a genuine cell id are the whole problem."""
    name = trace_dir_name(_Cell(REAL_CELL_ID))
    assert "|" not in name
    assert "N=6" in name and "regime=jittery" in name


def test_the_directory_can_actually_be_created(tmp_path):
    """The real test of a filename is whether the filesystem accepts it."""
    d = tmp_path / trace_dir_name(_Cell(REAL_CELL_ID))
    d.mkdir(parents=True)
    assert d.is_dir()


def test_no_trailing_dot_or_space():
    """Windows silently rejects both."""
    name = trace_dir_name(_Cell("trailing. "))
    assert not name.endswith((".", " "))


# --------------------------------------------------------------------------- #
# 2. The name must identify the row it came from, uniquely
# --------------------------------------------------------------------------- #

def test_name_encodes_arm_trial_and_seed():
    """A trace nobody can match back to a CSV row is worthless."""
    name = trace_dir_name(_Cell("c", arm="H3", trial_index=7, seed=999))
    assert "H3" in name and "t7" in name and "s999" in name


@pytest.mark.parametrize("a,b", [
    (_Cell("c", arm="H1"), _Cell("c", arm="H2")),
    (_Cell("c", trial_index=0), _Cell("c", trial_index=1)),
    (_Cell("c", seed=1), _Cell("c", seed=2)),
    (_Cell("c1"), _Cell("c2")),
])
def test_trials_that_differ_get_different_directories(a, b):
    """Sharing a directory would interleave two trials' events."""
    assert trace_dir_name(a) != trace_dir_name(b)


def test_long_names_are_truncated_but_stay_unique():
    """Truncation must not turn 'too long' into 'silently collides'."""
    base = "axis=value|" * 40
    a = trace_dir_name(_Cell(base + "one"))
    b = trace_dir_name(_Cell(base + "two"))
    assert len(a) <= 120 and len(b) <= 120
    assert a != b


def test_the_same_cell_always_yields_the_same_name():
    """Resuming a sweep must land traces in the same place."""
    assert trace_dir_name(_Cell(REAL_CELL_ID)) == trace_dir_name(_Cell(REAL_CELL_ID))


# --------------------------------------------------------------------------- #
# 3. Capture behaviour
# --------------------------------------------------------------------------- #

def _run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "mule-m1.jsonl").write_text('{"ts": 1, "event": "device_served"}\n')
    (d / "device-d0.jsonl").write_text('{"ts": 2, "event": "device_ready"}\n')
    (d / "device-d0.json").write_text('{"position": [1, 2, 0]}')
    (d / "cluster.port").write_text("5000")
    return d


def test_disabled_by_default_nothing_is_written(tmp_path):
    """Off is the default; every committed sweep ran that way."""
    driver = Exp4Driver()
    assert driver.trace_root is None
    driver._capture_traces(_run_dir(tmp_path), _Cell("c"))
    assert not (tmp_path / "traces").exists()


def test_events_and_configs_are_both_kept(tmp_path):
    """Configs carry device positions; the events do not.

    Without positions, no spatial policy (MAX-AoI's nearest-predecessor
    pathing, any travel-cost rule) can be scored from the trace.
    """
    root = tmp_path / "traces"
    Exp4Driver(trace_root=root)._capture_traces(_run_dir(tmp_path), _Cell("c"))

    kept = {p.name for p in (root / trace_dir_name(_Cell("c"))).iterdir()}
    assert "mule-m1.jsonl" in kept, "per-contact events are the point"
    assert "device-d0.json" in kept, "positions live only in the configs"


def test_trace_content_is_preserved_verbatim(tmp_path):
    root = tmp_path / "traces"
    Exp4Driver(trace_root=root)._capture_traces(_run_dir(tmp_path), _Cell("c"))
    got = (root / trace_dir_name(_Cell("c")) / "mule-m1.jsonl").read_text()
    assert got == '{"ts": 1, "event": "device_served"}\n'


def test_two_trials_do_not_share_a_directory(tmp_path):
    root = tmp_path / "traces"
    driver = Exp4Driver(trace_root=root)
    src = _run_dir(tmp_path)
    driver._capture_traces(src, _Cell(REAL_CELL_ID, trial_index=0))
    driver._capture_traces(src, _Cell(REAL_CELL_ID, trial_index=1))
    assert len(list(root.iterdir())) == 2


def test_a_capture_failure_never_kills_the_trial(tmp_path):
    """Bookkeeping is not worth losing a 70-second real-model trial over."""
    driver = Exp4Driver(trace_root=tmp_path / "traces")
    driver._capture_traces(tmp_path / "does-not-exist", _Cell("c"))  # no raise


def test_capture_is_idempotent_for_a_rerun_of_the_same_cell(tmp_path):
    """Re-running a cell overwrites its trace rather than failing on mkdir."""
    root = tmp_path / "traces"
    driver = Exp4Driver(trace_root=root)
    src = _run_dir(tmp_path)
    driver._capture_traces(src, _Cell("c"))
    driver._capture_traces(src, _Cell("c"))  # no raise
    assert (root / trace_dir_name(_Cell("c")) / "mule-m1.jsonl").exists()
