"""Exp 5 addendum, Study 5.11: the footprint probe (a trial's processes and peak memory).

Pinned:

* **The probe** reads the resident memory of the processes it is given, each
  with its children, from a daemon thread: the concurrent peak is the largest
  sum over one sample, and each process's own peak comes from the OS where it
  records one (``VmHWM`` on Linux), else from the samples. It only reads: a
  process that exits is skipped, and the processes keep running.
* **The file** (``footprint.json``) round-trips, and a trace without one, or
  with one that is not a JSON object, has none.
* **The driver** refuses the probe without a trace root, with an interval
  that is not > 0 and without psutil, before any trial; off by default, and
  the runner passes nothing unless ``--footprint-probe`` is given, which needs
  ``--keep-event-traces``.
* **The scorer** reads the file into the cost columns, in MiB, blank without
  it.
* **End to end** (slow): a stub trial through the real orchestrator writes a
  footprint of 1 + K + N processes beside its kept trace, and the scorer
  reads it back.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

psutil = pytest.importorskip("psutil")

from experiments.analysis.traces_scorer import COST_COLUMNS, cost_report, score_trial
from experiments.exp4 import footprint as FP
from experiments.exp4 import runner_main
from experiments.exp4.driver import Exp4Driver, trace_dir_name
from experiments.exp4.events_consumer import observation_from_rows
from experiments.runner.grid import Cell

MIB = float(1 << 20)


@pytest.fixture
def sleepers():
    """Two child processes that hold about 30 MiB and 1 MiB until killed."""
    hold = "import time; b = bytearray({n}); b[::4096] = b'x' * len(b[::4096]); time.sleep(60)"
    procs = [subprocess.Popen([sys.executable, "-c", hold.format(n=30 << 20)]),
             subprocess.Popen([sys.executable, "-c", hold.format(n=1 << 20)])]
    time.sleep(1.0)
    yield procs
    for p in procs:
        p.kill()
        p.wait()


def test_the_probe_reads_each_role_and_the_concurrent_peak(sleepers):
    big, small = sleepers
    probe = FP.FootprintProbe({"cluster": [small.pid], "device": [big.pid]}, interval_s=0.05)
    probe.start()
    time.sleep(0.4)
    fp = probe.stop()
    assert fp.processes == 2 and dict(fp.processes_by_role) == {
        "cluster": 1, "mule": 0, "device": 1}
    assert fp.samples >= 3
    assert fp.peak_rss_bytes_by_role["device"] >= 30 * MIB
    assert fp.peak_rss_bytes_by_role["cluster"] < fp.peak_rss_bytes_by_role["device"]
    assert fp.peak_rss_bytes_by_role["mule"] is None
    assert fp.peak_rss_bytes_total >= 30 * MIB
    # Each process's own peak is at least what the samples saw of it, so the
    # sum bounds the concurrent peak from above.
    assert fp.peak_rss_bytes_sum >= fp.peak_rss_bytes_total
    if sys.platform.startswith("linux"):
        assert fp.peak_source == FP.PEAK_OS
    assert fp.probe.startswith("psutil ")
    # It only reads: both are still running.
    assert big.poll() is None and small.poll() is None


def test_a_process_that_exits_is_skipped(sleepers):
    big, small = sleepers
    probe = FP.FootprintProbe({"device": [big.pid, small.pid]}, interval_s=0.05)
    probe.sample()
    small.kill()
    small.wait()
    probe.sample()
    fp = probe.stop()
    assert fp.processes_by_role["device"] == 2 and fp.samples == 3
    assert fp.peak_rss_bytes_by_role["device"] >= 30 * MIB


def test_a_process_with_no_os_peak_takes_its_largest_sample(sleepers, monkeypatch):
    big, _ = sleepers
    monkeypatch.setattr(FP, "_os_peak_bytes", lambda proc: None)
    probe = FP.FootprintProbe({"mule": [big.pid]}, interval_s=0.05)
    probe.sample()
    fp = probe.stop()
    assert fp.peak_source == FP.PEAK_SAMPLED
    assert fp.peak_rss_bytes_by_role["mule"] == fp.peak_rss_bytes_total


def test_the_probe_refuses_what_it_cannot_read():
    with pytest.raises(ValueError, match="interval_s"):
        FP.FootprintProbe({}, interval_s=0.0)
    with pytest.raises(ValueError, match="unknown role"):
        FP.FootprintProbe({"edge": [1]})
    probe = FP.FootprintProbe({}).start()
    with pytest.raises(RuntimeError, match="already started"):
        probe.start()
    fp = probe.stop()
    assert (fp.processes, fp.peak_rss_bytes_total, fp.peak_rss_bytes_sum) == (0, None, None)


def test_the_probe_follows_each_process_the_orchestrator_launches(sleepers):
    big, small = sleepers
    probe = FP.FootprintProbe(interval_s=0.05)
    probe.on_spawn("cluster", small.pid)
    probe.on_spawn("mule-m1", big.pid)
    probe.on_spawn("device-exp4-dev-000", small.pid)
    with pytest.raises(ValueError, match="unknown role"):
        probe.on_spawn("edge-1", small.pid)
    probe.sample()
    fp = probe.stop()
    assert dict(fp.processes_by_role) == {"cluster": 1, "mule": 1, "device": 1}
    assert fp.peak_rss_bytes_by_role["mule"] >= 30 * MIB


def test_a_process_that_exits_keeps_the_last_os_mark_read_while_it_ran(sleepers):
    big, _ = sleepers
    probe = FP.FootprintProbe({"mule": [big.pid]}, interval_s=0.05)
    probe.sample()
    big.kill()
    big.wait()
    fp = probe.stop()
    if sys.platform.startswith("linux"):
        assert fp.peak_source == FP.PEAK_OS
    assert fp.peak_rss_bytes_by_role["mule"] >= 30 * MIB


def test_the_file_round_trips_and_a_trace_without_one_has_none(tmp_path):
    fp = FP.Footprint(processes_by_role={"cluster": 1, "mule": 2, "device": 12},
                      peak_rss_bytes_total=5 << 30, peak_rss_bytes_by_role={
                          "cluster": 1 << 30, "mule": 1 << 29, "device": 3 << 29},
                      peak_rss_bytes_sum=6 << 30, peak_source=FP.PEAK_OS, samples=40,
                      interval_s=0.5, probe="psutil 7 on linux")
    assert FP.read_footprint(tmp_path) is None
    path = fp.write(tmp_path)
    assert path.name == FP.FOOTPRINT_FILE == "footprint.json"
    raw = FP.read_footprint(tmp_path)
    assert raw == fp.to_json() and raw["processes"] == 15 and raw["schema"] == 1
    path.write_text("[1, 2]", encoding="utf-8")
    assert FP.read_footprint(tmp_path) is None
    path.write_text("{not json", encoding="utf-8")
    assert FP.read_footprint(tmp_path) is None


# --------------------------------------------------------------------------- #
# The driver and the runner
# --------------------------------------------------------------------------- #

def test_the_driver_refuses_a_probe_it_cannot_run(tmp_path, monkeypatch):
    assert Exp4Driver().footprint_probe is False
    with pytest.raises(ValueError, match="trace_root"):
        Exp4Driver(footprint_probe=True)
    with pytest.raises(ValueError, match="footprint_interval_s"):
        Exp4Driver(footprint_probe=True, trace_root=tmp_path, footprint_interval_s=0.0)
    monkeypatch.setattr(FP, "probe_available", lambda: False)
    with pytest.raises(ValueError, match="psutil"):
        Exp4Driver(footprint_probe=True, trace_root=tmp_path)


def _driver_kwargs(monkeypatch, argv):
    seen = {}

    class _Stop(Exception):
        pass

    def fake(**kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(runner_main, "Exp4Driver", fake)
    with pytest.raises(_Stop):
        runner_main.main(argv)
    return seen


def test_the_runner_passes_the_probe_only_when_asked(tmp_path, monkeypatch):
    csv = str(tmp_path / "t.csv")
    plain = _driver_kwargs(monkeypatch, ["--csv", csv, "--arms", "H1"])
    assert not {"footprint_probe", "footprint_interval_s"} & set(plain)
    probed = _driver_kwargs(monkeypatch, ["--csv", csv, "--arms", "H1", "--keep-event-traces",
                                          "--footprint-probe", "--footprint-interval-s", "0.25"])
    assert (probed["footprint_probe"], probed["footprint_interval_s"]) == (True, 0.25)
    assert {k: v for k, v in probed.items()
            if k not in ("footprint_probe", "footprint_interval_s", "trace_root")} == {
        k: v for k, v in plain.items() if k != "trace_root"}


def test_the_runner_refuses_the_probe_without_kept_traces(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        runner_main.main(["--csv", str(tmp_path / "t.csv"), "--arms", "H1",
                          "--footprint-probe"])
    assert e.value.code == 2
    assert "--keep-event-traces" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The scorer
# --------------------------------------------------------------------------- #

def test_the_scorer_reads_the_footprint_in_mib():
    obs = observation_from_rows(cluster_rows=[], mule_rows=[], device_rows=[], n_devices=1)
    raw = {"processes": 8, "peak_rss_bytes_total": 3 << 30,
           "peak_rss_bytes_by_role": {"cluster": 512 << 20, "mule": 256 << 20,
                                      "device": 1 << 30}}
    got = cost_report(obs, mule_cfg={}, footprint=raw)
    assert (got.trial_processes, got.peak_rss_mib_total) == (8, 3072.0)
    assert (got.peak_rss_mib_cluster, got.peak_rss_mib_mule, got.peak_rss_mib_device) == (
        512.0, 256.0, 1024.0)
    bad = cost_report(obs, mule_cfg={}, footprint={
        "processes": True, "peak_rss_bytes_total": -1,
        "peak_rss_bytes_by_role": {"cluster": 1.5, "device": "1"}})
    assert (bad.trial_processes, bad.peak_rss_mib_total, bad.peak_rss_mib_cluster,
            bad.peak_rss_mib_device, bad.peak_rss_mib_mule) == (None,) * 5
    none = cost_report(obs, mule_cfg={}, footprint=None)
    assert (none.trial_processes, none.peak_rss_mib_total) == (None, None)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_a_stub_trial_through_the_real_orchestrator_writes_its_footprint(tmp_path):
    driver = Exp4Driver(default_n_devices=3, default_rf_range_m=60.0, default_n_missions=1,
                        trial_budget_s=90.0, startup_timeout_s=30.0, trace_root=tmp_path,
                        footprint_probe=True, footprint_interval_s=0.1)
    cell = Cell(cell_id="N=3-rrf=60.0", arm="H1", trial_index=0, seed=12345,
                params={"N": 3, "rrf": 60.0, "n_missions": 1})
    row = dict(driver.run_trial(cell))
    assert row["missions_completed"] >= 1
    trace = tmp_path / trace_dir_name(cell)
    raw = FP.read_footprint(trace)
    assert raw["processes"] == 5 and raw["processes_by_role"] == {
        "cluster": 1, "mule": 1, "device": 3}
    assert raw["samples"] >= 1 and raw["peak_rss_bytes_total"] > 0
    assert all(raw["peak_rss_bytes_by_role"][r] > 0 for r in FP.ROLES)
    scored = score_trial(trace, cost_columns=True).to_row()
    assert scored["trial_processes"] == 5
    assert scored["peak_rss_mib_total"] == pytest.approx(raw["peak_rss_bytes_total"] / MIB)
