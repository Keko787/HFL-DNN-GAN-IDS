"""Design section 5.2(b): the integration scenarios, re-run with the clock injected.

Each scenario runs twice on the same deterministic world: once as recorded
(``now_fn``, the wall clock) and once with ``mission_clock=MissionClock()``,
no contact band (the channel-free control, critic A1), ``abort`` and no
budget. Asserted for the clock:

* the same per-device outcomes, mission by mission: Pass-1 outcomes,
  deliveries, the merged devices, empty / docked / DOWN-timeout missions,
  and the same contact sets in both passes;
* every stamp in the scheduler's state, the deltas and the ledgers is below
  1e9 (simulated; the wall clock here reads 1.7e9);
* a monotone clock: each mission takes off at the previous landing, and the
  ledger accounts for every second in between;
* the pose is the dock at each takeoff and each landing.

The differences the clock is allowed, each pinned below rather than masked:

1. **S3.5 order.** The recorded mule plans Pass 1 from wherever its last
   mission ended; on the clock every mission takes off from the dock, so the
   distance order inside a bucket, and H2's learned order, can differ. The
   contact sets do not.
2. **Pass-2 order.** Recorded: nearest-first from the last Pass-1 stop; on
   the clock: from the dock.
3. **Targeted solicits** (unit U5; U5's open question 1). The recorded
   broadcast reaches every device and stashes the adverts of non-members, and
   a later contact can use a stale eligible advert of a device that has since
   become unavailable: it pushes, nobody listens, TIMEOUT. On the clock every
   contact solicits its own members and reads their fresh adverts: the same
   device refuses, PARTIAL. Device-side solicit and push-wait counts differ as
   well (fewer solicits, no push waits); those are not per-device outcomes.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.l1.mission_clock import MissionClock, SIM_EPOCH_S
from hermes.mission import ClientMission, LocalTrainResult
from hermes.mule import MuleSupervisor
from hermes.scheduler.selector import TargetSelectorRL
from hermes.transport import LoopbackDockLink, LoopbackRFLink
from hermes.types import DeviceID, FLState, MuleID, SpectrumSig

from tests.golden import _mule_harness as GH
from tests.integration import _ferry_harness as H

logging.getLogger("hermes.mission.client_mission").setLevel(logging.ERROR)

DOCK = (0.0, 0.0, 0.0)


def _summary(result) -> Dict:
    agg = result.aggregate
    return {
        "pass_1": H.pass_1_outcomes(result),
        "pass_2": H.delivery_outcomes(result),
        "merged": sorted(str(d) for d in (agg.contributing_devices if agg else ())),
        "empty": result.empty,
        "down_timeout": result.down_timeout,
        "docked_empty": result.docked_empty,
        "pass_1_contacts": H.queue_sets(result.pass_1_queue),
        "pass_2_contacts": H.queue_sets(result.pass_2_queue),
    }


def _check_sim_mission(rec, prev_end) -> float:
    r = rec["result"]
    assert rec["pose_before"] == DOCK and rec["pose_after"] == DOCK
    # Monotone: the first mission takes off at the epoch, every later one at
    # the previous landing.
    assert r.sim_start_s == (SIM_EPOCH_S if prev_end is None else prev_end)
    assert r.sim_end_s > r.sim_start_s
    assert sum(r.sim_ledger.values()) == pytest.approx(r.sim_end_s - r.sim_start_s, abs=1e-6)
    assert r.sim_ledger["turnaround"] == 30.0
    assert all(d.contact_ts < H.SIM_CEILING for _, d in rec["deltas"]), rec["deltas"]
    assert all(s < H.SIM_CEILING for s in H.ledger_stamps(r))
    assert all(v < H.SIM_CEILING for v in rec["state_stamps"].values())
    assert rec["mission_start_ts"] < H.SIM_CEILING
    return r.sim_end_s


def _record(w, sup, mid, result, pose_before):
    return {
        "result": result,
        "pose_before": pose_before,
        "pose_after": tuple(sup.mule_pose),
        "deltas": w.take_deltas(mid),
        "state_stamps": H.state_stamps(sup.scheduler.device_states),
        "mission_start_ts": sup.scheduler.mission_start_ts,
    }


# --------------------------------------------------------------------------- #
# One mule (the golden supervisor scenarios without a budget)
# --------------------------------------------------------------------------- #

def _run_single(sim: bool, *, missions=3, before=None, sup_kwargs=None) -> List[Dict]:
    w = H.World()
    mid = w.mule_ids[0]
    out = []
    with H.Patched(w.clock):
        sup = w.supervisor(mid, sim=sim, **dict(sup_kwargs() if sup_kwargs else {}))
        w.bootstrap()
        for m in range(missions):
            if before is not None:
                before(w, m)
            pose = tuple(sup.mule_pose)
            out.append(_record(w, sup, mid, sup.run_one_mission(), pose))
            w.clock.advance(GH.BETWEEN_MISSIONS_DT)
    return out


def _refuse_dev05_in_mission_2(world, m):
    if m == 1:
        world.set_state(["dev-05"], FLState.UNAVAILABLE)
    elif m == 2:
        world.set_state(["dev-05"], FLState.FL_OPEN)


def _compare(legacy: List[Dict], sim: List[Dict], *, expected_differences=None):
    """Same per-device outcomes, but for the differences named (mission, key, device)."""
    expected_differences = dict(expected_differences or {})
    prev_end = None
    for k, (a, b) in enumerate(zip(legacy, sim)):
        sa, sb = _summary(a["result"]), _summary(b["result"])
        for key in ("pass_1", "pass_2"):
            devices = set(sa[key]) | set(sb[key])
            for did in devices:
                pair = (sa[key].get(did), sb[key].get(did))
                if (k, key, did) in expected_differences:
                    assert pair == expected_differences.pop((k, key, did)), (k, key, did, pair)
                else:
                    assert pair[0] == pair[1], (k, key, did, pair)
        for key in ("merged", "empty", "down_timeout", "docked_empty",
                    "pass_1_contacts", "pass_2_contacts"):
            assert sa[key] == sb[key], (k, key, sa[key], sb[key])
        prev_end = _check_sim_mission(b, prev_end)
    assert not expected_differences, expected_differences


def test_h1_no_budget_on_the_clock():
    legacy = _run_single(False, before=_refuse_dev05_in_mission_2)
    sim = _run_single(True, before=_refuse_dev05_in_mission_2)
    # Difference 3: in mission 2 the recorded mule pushed dev-05 on the stale
    # eligible advert it stashed in mission 1 (TIMEOUT); the clock's targeted
    # solicit reads its fresh advert and S2B refuses it (PARTIAL).
    _compare(legacy, sim, expected_differences={
        (1, "pass_1", "dev-05"): ("timeout", "partial"),
    })
    # Difference 1: the recorded pose carries over; the clock's is the dock.
    assert legacy[0]["pose_after"] != DOCK
    assert all(rec["pose_before"] == DOCK for rec in sim)
    # Difference 2: Pass 2 is walked from the dock on the clock.
    from hermes.scheduler.stages import order_pass_2_greedy
    for rec in sim:
        q2 = rec["result"].pass_2_queue
        assert list(q2) == order_pass_2_greedy(list(q2), mule_pose=DOCK)


def test_h2_selector_on_the_clock():
    kwargs = lambda: dict(target_selector=TargetSelectorRL(epsilon=0.0, rng_seed=0),  # noqa: E731
                          rf_prior_snr_db=12.5)
    legacy = _run_single(False, sup_kwargs=kwargs)
    sim = _run_single(True, sup_kwargs=kwargs)
    _compare(legacy, sim)


# --------------------------------------------------------------------------- #
# Two mules at quorum 2 (the golden K = 2 scenarios)
# --------------------------------------------------------------------------- #

def _run_k2(sim: bool, *, missions, down_wait_s, nest, before=None):
    w = H.World(mule_ids=["mule-a", "mule-b"], assignment=GH.K2_ASSIGNMENT,
                min_participation=2)
    ma, mb = MuleID("mule-a"), MuleID("mule-b")
    out: Dict[str, List[Dict]] = {"a": [], "b": []}
    with H.Patched(w.clock):
        sa = w.supervisor(ma, sim=sim, down_wait_s=down_wait_s, dock_on_empty=True)
        sb = w.supervisor(mb, sim=sim, down_wait_s=down_wait_s, dock_on_empty=True)
        w.bootstrap()
        for m in range(missions):
            if before is not None:
                before(w, m)

            def fly_b():
                pose = tuple(sb.mule_pose)
                out["b"].append(_record(w, sb, mb, sb.run_one_mission(), pose))

            if nest(m):
                w.server.tasks[ma].append(fly_b)
            pose = tuple(sa.mule_pose)
            out["a"].append(_record(w, sa, ma, sa.run_one_mission(), pose))
            if w.server.tasks[ma]:
                w.server.tasks[ma].popleft()()
            elif not nest(m):
                fly_b()
            w.clock.advance(GH.BETWEEN_MISSIONS_DT)
    return out


def _empties(world, m):
    if m == 1:
        world.set_state(["dev-02", "dev-03", "dev-05"], FLState.UNAVAILABLE)
    elif m == 2:
        world.set_state(["dev-00", "dev-01", "dev-04", "dev-06"], FLState.UNAVAILABLE)


@pytest.mark.parametrize("scenario", ["k2_quorum_dock_on_empty", "k2_down_timeout"])
def test_two_mules_on_the_clock(scenario):
    kw = (dict(missions=3, down_wait_s=30.0, nest=lambda m: True, before=_empties)
          if scenario == "k2_quorum_dock_on_empty"
          else dict(missions=3, down_wait_s=0.3, nest=lambda m: m > 0))
    legacy = _run_k2(False, **kw)
    sim = _run_k2(True, **kw)
    for mule in ("a", "b"):
        _compare(legacy[mule], sim[mule])
    flags = [(rec["result"].empty, rec["result"].docked_empty, rec["result"].down_timeout)
             for mule in ("a", "b") for rec in sim[mule]]
    if scenario == "k2_quorum_dock_on_empty":
        assert (True, True, False) in flags                  # an empty mission docked
    else:
        assert (False, False, True) in flags                 # a survived DOWN timeout
        timed_out = [rec["result"] for rec in sim["a"] if rec["result"].down_timeout]
        for r in timed_out:
            assert r.pass_2_flown == [] and r.sim_pass_2_start_s is None
            assert r.sim_ledger["turnaround"] == 30.0


# --------------------------------------------------------------------------- #
# Real threads: the two-pass loopback scenario
# --------------------------------------------------------------------------- #

MULE = MuleID("mule-test")
DEVICE_IDS = [DeviceID(f"dev-{i:02d}") for i in range(4)]


def _train_factory(seed):
    rng = np.random.default_rng(seed)

    def _train(theta, synth):
        delta = [w + rng.normal(0.0, 0.01, size=w.shape).astype(w.dtype) for w in theta]
        return LocalTrainResult(delta_theta=delta, num_examples=int(rng.integers(4, 16)),
                                accuracy=0.8, auc=0.8, loss=float(rng.uniform(0.1, 0.3)),
                                theta_after=delta)
    return _train


def _real_thread_mission(sim: bool, positions):
    """``test_mule_supervisor_two_pass``'s scenario: real device threads that
    serve twice each (Pass 1 and Pass 2) and a cluster thread serving one
    inter-pass dock."""
    rf = LoopbackRFLink(newest_solicit_only=True) if sim else LoopbackRFLink()
    dock = LoopbackDockLink()
    devices = []
    for i, did in enumerate(DEVICE_IDS):
        rf.register_device(did)
        cm = ClientMission(device_id=did, rf=rf, local_train=_train_factory(200 + i),
                           solicit_timeout_s=2.0, disc_push_timeout_s=2.0)
        cm.set_state(FLState.FL_OPEN)
        devices.append(cm)
    registry = DeviceRegistry()
    for did, pos in zip(DEVICE_IDS, positions):
        registry.register(device_id=did, position=pos,
                          spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)))
    registry.rebalance([MULE], round_counter=0)
    cluster = HFLHostCluster(
        registry=registry,
        generator=StubGeneratorHost(disc_weights=[np.zeros((4,), dtype=np.float32),
                                                  np.ones((3, 3), dtype=np.float32) * 0.01]),
        dock=dock, synth_batch_size=4,
    )
    kw = {"mission_clock": MissionClock()} if sim else {}
    sup = MuleSupervisor(mule_id=MULE, rf=rf, dock=dock, session_ttl_s=2.0,
                         rf_range_m=60.0, **kw)
    deltas = []
    ingest = sup.scheduler.ingest_round_close_delta
    sup.mission.scheduler_bus = lambda d: (deltas.append(d), ingest(d))[1]
    dock.send_down(cluster.dispatch_down_bundle(MULE))
    assert sup.wait_for_initial_dock(timeout=2.0)

    def serve_dock():
        up = dock.recv_up(timeout=5.0)
        cluster.ingest_up_bundle(up)
        cluster.aggregate_pending()
        cluster.close_cluster_round()
        dock.send_down(cluster.dispatch_down_bundle(MULE))

    cluster_t = threading.Thread(target=serve_dock, daemon=True)
    cluster_t.start()
    workers = []
    for cm in devices:
        t = threading.Thread(target=lambda c=cm: [c.serve_once() for _ in range(2)], daemon=True)
        t.start()
        workers.append(t)
    result = sup.run_one_mission()
    cluster_t.join(timeout=5.0)
    for t in workers:
        t.join(timeout=5.0)
    return result, deltas, sup


@pytest.mark.parametrize("positions", [
    [(0.0, 0.0, 0.0), (10.0, 5.0, 0.0), (20.0, 0.0, 0.0), (15.0, 15.0, 0.0)],
    [(0.0, 0.0, 0.0), (10.0, 5.0, 0.0), (200.0, 0.0, 0.0), (210.0, 15.0, 0.0)],
], ids=["one-contact", "two-contacts"])
def test_the_real_thread_two_pass_mission_on_the_clock(positions):
    legacy, _, _ = _real_thread_mission(False, positions)
    sim, deltas, sup = _real_thread_mission(True, positions)
    for r in (legacy, sim):
        assert set(H.pass_1_outcomes(r).values()) == {"clean"}
        assert set(H.delivery_outcomes(r).values()) == {"delivered"}
        assert sorted(str(d) for d in r.aggregate.contributing_devices) == \
            sorted(str(d) for d in DEVICE_IDS)
    assert H.queue_sets(legacy.pass_1_queue) == H.queue_sets(sim.pass_1_queue)
    assert H.queue_sets(legacy.pass_2_queue) == H.queue_sets(sim.pass_2_queue)
    # The real wall clock (1.7e9 s) never reaches simulated state.
    assert all(d.contact_ts < H.SIM_CEILING for d in deltas)
    assert all(s < H.SIM_CEILING for s in H.ledger_stamps(sim))
    assert all(v < H.SIM_CEILING for v in H.state_stamps(sup.scheduler.device_states).values())
    assert sup.mule_pose == DOCK
    assert sum(sim.sim_ledger.values()) == pytest.approx(sim.sim_end_s - sim.sim_start_s)
