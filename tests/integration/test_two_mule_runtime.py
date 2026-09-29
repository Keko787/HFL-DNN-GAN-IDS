"""FeRRy Phase 2 — two mules on one cluster, through the real process tree.

Every earlier process-level test ran one mule, so the cross-mule merge had
only ever been an identity. These run 2 mules x 3 devices on the stub model,
3 missions each, over real subprocesses and TCP (``build_exp4_topology`` with
``n_mules=2``), and check the multi-mule runtime:

* both mules finish every mission, with no ``mission_failed``;
* no stale DOWN: each mule's mission n+1 flies the θ the cluster answered its
  mission-n upload with (derived from the single-threaded cluster log);
* under ``agg:plain`` with a quorum of 2, every closed round merged a partial
  from each mule;
* under ``agg:cutoff`` with ``a_max=1``, no merge fails and every upload is
  answered by exactly one merge or expiry;
* a mule whose devices stop early still docks (``dock_on_empty``), so the
  other mule's quorum closes; a mule that outlives its DOWN wait
  (``down_wait_s``) flies on instead of failing, and the partial the cluster
  refuses from it is traced as refused;
* with backhaul loss under a quorum of 2, a lost upload holds its mule's place
  in the round, so the mules stay in step to the last mission even when one
  loses more uploads than the other;
* arm D4 with two mules runs end to end through ``Exp4Driver``.

Stub missions take milliseconds, so one device per mule never delivers
(``contact_reliability=0``): its session runs to ``session_ttl_s``, which
holds every mission open long enough for the two mules' missions to overlap.

Marked ``slow``: each run spawns a 9-process tree (about 4 s).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from experiments.exp4.topology_builder import build_exp4_topology
from hermes.processes import MultiProcessOrchestrator, TopologyConfig

N_MISSIONS = 3
MULES = ("exp4-mule-0", "exp4-mule-1")


# --------------------------------------------------------------------------- #
# Running a topology and reading its traces
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class _Run:
    return_codes: Dict[str, Optional[int]]
    cluster: List[dict]
    mules: Dict[str, List[dict]]
    cluster_stderr: str
    mule_stderr: Dict[str, str]

    def named(self, mule: str, event: str) -> List[dict]:
        return [r for r in self.mules[mule] if r["event"] == event]


def _topology(
    *, min_participation: int, rule: str, params: Optional[dict] = None,
    dock_on_empty: bool = False, down_wait_s: Optional[float] = None,
    session_ttl_s: float = 0.8, **extra,
) -> TopologyConfig:
    topo = build_exp4_topology(
        n_devices=6, rf_range_m=60.0, n_missions=N_MISSIONS, seed=11, n_mules=2,
        min_participation=min_participation, aggregation=rule,
        aggregation_params=dict(params or {}), session_ttl_s=session_ttl_s,
        dock_on_empty=dock_on_empty, down_wait_s=down_wait_s, **extra,
    )
    slow = {m.expected_devices[0] for m in topo.mules}
    for dev in topo.devices:
        if dev.device_id in slow:
            dev.contact_reliability = 0.0
    return topo


def _run(topo: TopologyConfig, *, budget_s: float = 90.0) -> _Run:
    orch = MultiProcessOrchestrator(topo, capture_output=True)
    try:
        orch.start_all(timeout=30.0)
        for mule_id, handle in orch.mule_handles.items():
            try:
                handle.proc.wait(timeout=budget_s)
            except Exception as e:  # pragma: no cover — reported below
                pytest.fail(f"mule {mule_id} did not finish: {e}\n{handle.stderr_tail(80)}")
        codes = {m: h.returncode() for m, h in orch.mule_handles.items()}
        mule_stderr = {m: h.stderr_tail(200) for m, h in orch.mule_handles.items()}
        cluster_stderr = orch.cluster_handle.stderr_tail(200)
    finally:
        orch.shutdown_all(timeout=10.0, cleanup_tmpdir=False)
    try:
        (cluster_log,) = sorted(orch.tmpdir.glob("cluster-*.jsonl"))
        return _Run(
            return_codes=codes,
            cluster=_read_jsonl(cluster_log),
            mules={m: _read_jsonl(orch.tmpdir / f"mule-{m}.jsonl") for m in codes},
            cluster_stderr=cluster_stderr,
            mule_stderr=mule_stderr,
        )
    finally:
        orch.cleanup()


# --------------------------------------------------------------------------- #
# What the cluster log implies
# --------------------------------------------------------------------------- #

def _answered_versions(cluster_rows: List[dict]) -> Dict[Tuple[str, int], int]:
    """(mule, mission n) -> version of the θ the cluster answered n's upload with.

    The service is single-threaded and logs in order, and a DOWN carries the
    cluster round current when it is sent. An ingested UP waits until a round
    closes (every waiting mule gets the new round), a fold expires (every
    waiting mule gets the current one) or FedBuff defers it (its mule gets the
    current one). A lost upload is answered at once with the current one,
    unless it ``awaits_quorum``: then it waits like an ingested one.
    """
    version = 0
    waiting: Dict[str, int] = {}
    answered: Dict[Tuple[str, int], int] = {}
    for row in cluster_rows:
        event = row["event"]
        if event == "up_bundle_ingested" or (
            event == "backhaul_upload_lost" and row.get("awaits_quorum")
        ):
            waiting.setdefault(row["mule_id"], row["mission_round"])
        elif event == "cluster_round_closed":
            version = row["cluster_round"]
            answered.update({(m, n): version for m, n in waiting.items()})
            waiting.clear()
        elif event == "cluster_merge_expired":
            answered.update({(m, n): version for m, n in waiting.items()})
            waiting.clear()
        elif event == "cluster_merge_deferred":
            answered[(row["mule_id"], waiting.pop(row["mule_id"]))] = version
        elif event == "backhaul_upload_lost":
            answered[(row["mule_id"], row["mission_round"])] = version
    return answered


def _assert_every_mission_finished(run: _Run) -> None:
    assert run.return_codes == {m: 0 for m in MULES}, run.mule_stderr
    for mule in MULES:
        assert run.named(mule, "mission_failed") == [], run.mule_stderr[mule]
        done = run.named(mule, "mission_completed")
        assert [r["mission_round"] for r in done] == list(range(1, N_MISSIONS + 1))


def _assert_no_stale_down(run: _Run) -> None:
    answered = _answered_versions(run.cluster)
    for mule in MULES:
        bases = {
            r["mission_round"]: r["pass_1_merge"]["base_version"]
            for r in run.named(mule, "mission_completed")
            if r.get("pass_1_merge") is not None
        }
        assert bases.get(1) == 0, (mule, bases)      # the bootstrap θ
        for n in range(1, N_MISSIONS):
            if n + 1 in bases:
                assert bases[n + 1] == answered[(mule, n)], (mule, n, bases, answered)


def _assert_missions_overlap(run: _Run) -> None:
    def spans(mule):
        starts = [r["ts"] for r in run.named(mule, "mission_started")]
        ends = [r["ts"] for r in run.named(mule, "mission_completed")]
        return list(zip(starts, ends))

    a, b = spans(MULES[0]), spans(MULES[1])
    assert any(s1 < e2 and s2 < e1 for s1, e1 in a for s2, e2 in b), (a, b)


def _uploads_between_closes(cluster_rows: List[dict]) -> List[set]:
    out, current = [], set()
    for row in cluster_rows:
        if row["event"] == "up_bundle_ingested":
            current.add(row["mule_id"])
        elif row["event"] == "cluster_round_closed":
            out.append(current)
            current = set()
    return out


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.parametrize(
    "min_participation, rule, params, dock_on_empty",
    [
        (1, "agg:cutoff", {"a_max": 1}, False),
        (2, "agg:plain", {}, True),
        (1, "agg:fedex", {}, False),
    ],
    ids=["cutoff-a_max1-quorum1", "plain-quorum2", "fedex-quorum1"],
)
def test_two_mules_merge_across_the_cluster_without_stale_downs(
    min_participation, rule, params, dock_on_empty,
):
    run = _run(_topology(
        min_participation=min_participation, rule=rule, params=params,
        dock_on_empty=dock_on_empty,
    ))
    _assert_every_mission_finished(run)
    _assert_no_stale_down(run)
    _assert_missions_overlap(run)

    uploads = [r for r in run.cluster if r["event"] == "up_bundle_ingested"]
    assert len(uploads) == 2 * N_MISSIONS
    closes = _uploads_between_closes(run.cluster)
    if min_participation == 2:
        # A real cross-mule merge: every round folded one partial per mule.
        assert closes == [set(MULES)] * N_MISSIONS
    else:
        # Each upload is its own merge, or (cutoff only) an expiry.
        answers = [
            r["event"] for r in run.cluster
            if r["event"] in ("up_bundle_ingested", "cluster_merge", "cluster_merge_expired")
        ]
        assert answers[0::2] == ["up_bundle_ingested"] * len(uploads), answers
        assert len(answers[1::2]) == len(uploads), answers
        assert set(answers[1::2]) <= {"cluster_merge", "cluster_merge_expired"}, answers
    if rule == "agg:fedex":
        merges = [r for r in run.cluster if r["event"] == "cluster_merge"]
        assert len(merges) == 2 * N_MISSIONS and all(m["n_clients"] == 6 for m in merges)
    assert "FedAvgError" not in run.cluster_stderr
    assert "Traceback" not in run.cluster_stderr


@pytest.mark.slow
def test_a_mule_whose_devices_stop_early_does_not_stall_the_other():
    """Mule 1's devices exit after mission 1 (one collect, one delivery). Its
    later missions are empty but still dock, so the quorum of 2 keeps closing
    and mule 0 finishes; before Phase 2 mule 0 waited 10 s and died."""
    topo = _topology(min_participation=2, rule="agg:plain", dock_on_empty=True,
                     down_wait_s=20.0)
    stops = set(topo.mules[1].expected_devices)
    for dev in topo.devices:
        if dev.device_id in stops:
            dev.n_serves = 2
    run = _run(topo)
    _assert_every_mission_finished(run)
    _assert_no_stale_down(run)
    empty = run.named(MULES[1], "mission_empty")
    assert [(r["mission_round"], r["docked"]) for r in empty] == [(2, True), (3, True)]
    assert _uploads_between_closes(run.cluster) == [set(MULES)] * N_MISSIONS
    assert run.named(MULES[0], "dock_down_timeout") == []


@pytest.mark.slow
def test_a_mule_that_outwaits_its_quorum_flies_on():
    """Mule 0's missions are eight times shorter than mule 1's, and the quorum
    is 2: mule 0's DOWN waits run out. With ``down_wait_s`` it skips Pass 2
    and flies on; without it, it used to end its run with ``mission_failed``."""
    topo = _topology(min_participation=2, rule="agg:plain", dock_on_empty=True,
                     down_wait_s=0.5)
    topo.mules[0].session_ttl_s = 0.3
    topo.mules[1].session_ttl_s = 2.5
    run = _run(topo)
    _assert_every_mission_finished(run)
    timeouts = run.named(MULES[0], "dock_down_timeout")
    assert timeouts and all(t["down_wait_s"] == 0.5 for t in timeouts)
    skipped = {t["mission_round"] for t in timeouts}
    for done in run.named(MULES[0], "mission_completed"):
        if done["mission_round"] in skipped:
            assert done["pass_2_contacts"] == 0 and done["delivered"] is None
    # Mule 0 uploaded mission 2 while its mission-1 partial still waited for
    # mule 1: the round kept mission 1's, and the trace says mission 2's was
    # refused. (Mule 1, left alone once mule 0 finishes, does the same later.)
    refused = [r for r in run.cluster
               if r["event"] == "up_bundle_ingested" and r.get("partial_refused")]
    assert (MULES[0], 2, 1) in {
        (r["mule_id"], r["mission_round"], r["held_mission_round"]) for r in refused
    }
    assert all(r["held_mission_round"] < r["mission_round"] for r in refused)


def _quorum_groups(cluster_rows: List[dict]) -> List[set]:
    """The (mule, mission) partials each fold took: ingested uploads and lost
    ones holding a place, between one round close (or expiry) and the next."""
    out, current = [], set()
    for row in cluster_rows:
        event = row["event"]
        if event == "up_bundle_ingested" and not row.get("partial_refused"):
            current.add((row["mule_id"], row["mission_round"]))
        elif event == "backhaul_upload_lost" and row.get("awaits_quorum"):
            current.add((row["mule_id"], row["mission_round"]))
        elif event in ("cluster_round_closed", "cluster_merge_expired"):
            out.append(current)
            current = set()
    return out + ([current] if current else [])


@pytest.mark.slow
def test_lost_uploads_keep_a_full_quorum_in_step():
    """Quorum 2 under agg:plain with 50% backhaul loss. Mule 0 loses missions
    1 and 2, mule 1 only mission 1 (each mule's own seeded stream). A lost
    upload used to be answered at once with nothing in the round, so mule 0
    ran ahead with one partial in the cluster to mule 1's two, and mule 1's
    last one waited for a partner that had finished — for the whole trial
    budget under the driver. Every mission now holds its mule's place."""
    topo = _topology(min_participation=2, rule="agg:plain", dock_on_empty=True,
                     down_wait_s=10.0, backhaul_loss_pct=50.0, backhaul_rng_seed=7)
    run = _run(topo)
    _assert_every_mission_finished(run)
    lost = {(r["mule_id"], r["mission_round"]) for r in run.cluster
            if r["event"] == "backhaul_upload_lost"}
    assert lost == {(MULES[0], 1), (MULES[0], 2), (MULES[1], 1)}
    for mule in MULES:
        assert run.named(mule, "dock_down_timeout") == [], run.mule_stderr[mule]
    assert all(r.get("awaits_quorum") for r in run.cluster
               if r["event"] == "backhaul_upload_lost")
    # Every fold took mission n from each mule: round 1 (both lost) expired,
    # rounds 2 and 3 merged across the mules.
    assert _quorum_groups(run.cluster) == [
        {(m, n) for m in MULES} for n in range(1, N_MISSIONS + 1)
    ]
    assert len([r for r in run.cluster if r["event"] == "cluster_round_closed"]) == 2
    _assert_no_stale_down(run)
    assert "Traceback" not in run.cluster_stderr


@pytest.mark.slow
def test_arm_d4_runs_two_mules_end_to_end_through_the_driver(tmp_path):
    from experiments.exp4.driver import Exp4Driver, d4_slice_assignment, trace_dir_name
    from experiments.runner import Cell

    cell = Cell(
        cell_id="d4-two-mules", arm="D4", trial_index=0, seed=2024,
        params={"N": 6, "rrf": 60.0, "n_missions": N_MISSIONS},
    )
    driver = Exp4Driver(
        n_mules=2, aggregation="agg:fedex", trial_budget_s=90.0,
        startup_timeout_s=30.0, trace_root=tmp_path / "traces",
    )
    row = dict(driver.run_trial(cell))
    assert row["n_mules"] == 2 and row["min_participation"] == 1
    assert row["aggregation"] == "agg:fedex"
    assert row["mission_failures"] == 0

    trace = tmp_path / "traces" / trace_dir_name(cell)
    reference = build_exp4_topology(n_devices=6, rf_range_m=60.0, n_missions=N_MISSIONS,
                                    seed=2024)
    split = d4_slice_assignment(reference.devices, 2, 2024)
    for k, mule in enumerate(MULES):
        cfg = json.loads((trace / f"mule-{mule}.json").read_text(encoding="utf-8"))
        assert cfg["contact_policy"] == "fedex" and cfg["dock_on_empty"] is True
        assert sorted(cfg["expected_devices"]) == sorted(
            d.device_id for i, d in enumerate(reference.devices) if split[i] == k
        )
        events = _read_jsonl(trace / f"mule-{mule}.jsonl")
        assert [e for e in events if e["event"] == "mission_failed"] == []
        assert len([e for e in events if e["event"] == "mission_completed"]) == N_MISSIONS
    merges = [e for e in _read_jsonl(next(trace.glob("cluster-*.jsonl")))
              if e["event"] == "cluster_merge"]
    assert merges and all(m["rule"] == "agg:fedex" for m in merges)
