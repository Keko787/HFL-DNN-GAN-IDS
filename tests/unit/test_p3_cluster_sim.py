"""FeRRy Phase 3, unit U7 — the cluster on the simulated mission clock.

* ``HFLHostCluster(sim_clock=True)`` tracks the latest simulated upload it
  ingested (accepted, refused-but-folded, the lost-upload stand-in; not an
  ignored duplicate) and echoes it on every DOWN as ``cluster_sim_ts``; the
  recorded cluster never does, whatever the UPs carry.
* The seconds backhaul model's loss draw is keyed by (trial seed, mule,
  mission round): the same mission is decided by the same uniform whatever
  came before or in which order (paired by mission), at the UP's own
  ``p_loss`` (1.0 below the floor); an unpriced UP is never lost. The
  recorded model's stream is untouched on the sim clock.
* Sim-mode events gain ``sim_upload_ts``/``carrier``/``snr_db``/``p_loss``
  and ``sim_ts``; wall-mode events are the recorded ones.
* The lost-upload stand-in keeps ``sim_upload_ts`` and ``backhaul`` (critic
  B8); a quorum-2 DOWN after a held loss carries the lost upload's time.
* SpectrumSig forwarding (design section 4.7), with the band classes the
  service is configured with, and the refusal of wall-clock deadline
  overrides (critic B3).
* On the clock under the recorded ``mission`` model, each event's ``p_loss``
  is the schedule entry the draw read (one index rule,
  ``mission_schedule_index``); a refused partial's event carries its own
  upload's simulated fields.
"""

from __future__ import annotations

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.l1.channel_model import SALT_BACKHAUL_LOSS, ferry_salt, keyed_uniform
from hermes.mission.aggregation_rules import AGG_CUTOFF, AggregationSpec
from hermes.processes.cluster import ClusterService, _lost_upload_stand_in, stub_disc_weights
from hermes.processes.config import ClusterConfig
from hermes.scheduler.stages.s3_deadline import fold_cluster_amendment
from hermes.types import (
    ClusterAmendment,
    ContactHistory,
    DeviceID,
    DeviceSchedulerState,
    MissionOutcome,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MuleID,
    PartialAggregate,
    SpectrumSig,
    UpBundle,
)
from hermes.types.bundles import BackhaulUpload

T0 = 1.0e6
SEED = 4242


def _up(mule: str, mission_round: int, *, spec=None, sim_ts=None, p_loss=None,
        lines=(), empty=False, base_version=0) -> UpBundle:
    spec = spec or AggregationSpec()
    m = MuleID(mule)
    theta = stub_disc_weights()
    partial = PartialAggregate(
        mule_id=m, mission_round=mission_round,
        weights=[] if empty else [w + 0.1 for w in theta] if spec.is_plain
        else [np.full(w.shape, 0.1, dtype=np.float32) for w in theta],
        num_examples=0 if empty else 5,
        contributing_devices=() if empty else (DeviceID(f"{mule}-d0"),),
        rule=spec.rule, update_form=spec.update_form, base_version=base_version,
        weight_mass=0.0 if empty else 5.0, n_updates=0 if empty else 1,
    )
    backhaul = None
    if p_loss is not None:
        backhaul = BackhaulUpload(carrier=2, snr_db=9.5, p_loss=float(p_loss),
                                  t_start_s=(sim_ts or T0) - 0.25, upload_s=0.25, nbytes=52)
    return UpBundle(
        mule_id=m, partial_aggregate=partial,
        round_close_report=MissionRoundCloseReport(
            mule_id=m, mission_round=mission_round, started_at=T0, finished_at=T0,
            lines=list(lines),
        ),
        contact_history=ContactHistory(mule_id=m, mission_round=mission_round),
        sim_upload_ts=sim_ts, backhaul=backhaul,
    )


def _cluster(*, sim: bool, min_participation: int = 1, spec=None) -> HFLHostCluster:
    registry = DeviceRegistry()
    for i in range(3):
        registry.register(DeviceID(f"d{i}"), (float(10 * i), 0.0, 0.0),
                          SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)))
    registry.rebalance([MuleID("m1"), MuleID("m2")], round_counter=0)
    extra = dict(sim_clock=True) if sim else {}
    return HFLHostCluster(
        registry=registry, generator=StubGeneratorHost(disc_weights=stub_disc_weights()),
        dock=None, synth_batch_size=1, min_participation=min_participation,
        aggregation=spec or AggregationSpec(), **extra,
    )


# --------------------------------------------------------------------------- #
# HFLHostCluster: simulated time as data
# --------------------------------------------------------------------------- #

def test_the_cluster_echoes_the_latest_ingested_upload_time():
    c = _cluster(sim=True, min_participation=2)
    assert c.sim_ts is None
    assert c.dispatch_down_bundle(MuleID("m1")).cluster_sim_ts is None       # bootstrap
    assert c.ingest_up_bundle(_up("m1", 1, sim_ts=T0 + 90.0))
    assert c.ingest_up_bundle(_up("m2", 1, sim_ts=T0 + 40.0))
    assert c.sim_ts == T0 + 90.0                                             # max, not last
    down = c.dispatch_down_bundle(MuleID("m2"))
    assert down.cluster_sim_ts == T0 + 90.0


def test_refused_partials_count_and_ignored_duplicates_do_not():
    c = _cluster(sim=True, min_participation=2)
    c.ingest_up_bundle(_up("m1", 1, sim_ts=T0 + 10.0))
    assert not c.ingest_up_bundle(_up("m1", 1, sim_ts=T0 + 99.0))   # a resend: ignored whole
    assert c.sim_ts == T0 + 10.0
    assert not c.ingest_up_bundle(_up("m1", 2, sim_ts=T0 + 50.0))   # refused, reports folded
    assert c.sim_ts == T0 + 50.0


def test_a_wall_stamp_is_never_taken_as_simulated_time():
    c = _cluster(sim=True)
    c.ingest_up_bundle(_up("m1", 1, sim_ts=T0 + 10.0))
    c.ingest_up_bundle(_up("m1", 2, sim_ts=1.7e9))            # refused partial, wall stamp
    assert c.sim_ts == T0 + 10.0
    assert c.dispatch_down_bundle(MuleID("m1")).cluster_sim_ts == T0 + 10.0


def test_the_recorded_cluster_never_echoes_simulated_time():
    c = _cluster(sim=False)
    c.ingest_up_bundle(_up("m1", 1, sim_ts=T0 + 90.0))
    assert c.sim_ts is None
    assert c.dispatch_down_bundle(MuleID("m1")).cluster_sim_ts is None


def test_the_echo_is_not_signed_so_legacy_signatures_are_unchanged():
    from hermes.types.signatures import verify_down_bundle

    sim, wall = _cluster(sim=True), _cluster(sim=False)
    for c in (sim, wall):
        c.ingest_up_bundle(_up("m1", 1, sim_ts=T0 + 5.0))
    a, b = sim.dispatch_down_bundle(MuleID("m1")), wall.dispatch_down_bundle(MuleID("m1"))
    assert a.cluster_sim_ts == T0 + 5.0 and b.cluster_sim_ts is None
    assert a.bundle_sig == b.bundle_sig and verify_down_bundle(a)


# --------------------------------------------------------------------------- #
# SpectrumSig forwarding (design section 4.7) and deadline overrides (B3)
# --------------------------------------------------------------------------- #

def _line(did, band, snr, outcome=MissionOutcome.CLEAN):
    return MissionRoundCloseLine(device_id=DeviceID(did), outcome=outcome,
                                 contact_ts=T0 + 1.0, band=band, snr_db=snr)


def test_contact_snr_per_band_class_reaches_the_mules_scheduler():
    c = _cluster(sim=True)
    c.ingest_up_bundle(_up("m1", 1, sim_ts=T0 + 1.0, lines=[
        _line("d0", 0, 12.5), _line("d2", 2, -3.25), _line("d1", None, None),
    ]))
    c.ingest_up_bundle(_up("m1", 2, sim_ts=T0 + 2.0, lines=[_line("d0", 1, 7.0)]))
    assert c.registry.get(DeviceID("d0")).spectrum_sig.contact_class_snr_db == {
        "wide": 12.5, "medium": 7.0}
    deltas = {}
    for mule in ("m1", "m2"):
        deltas.update(c.dispatch_down_bundle(MuleID(mule)).cluster_amendments.registry_deltas)
    assert deltas[DeviceID("d2")]["spectrum_sig"].contact_class_snr_db == {"narrow": -3.25}
    assert "spectrum_sig" not in deltas[DeviceID("d1")]          # nothing observed
    # The mule's scheduler folds it (unit U4's fold).
    states = {DeviceID("d0"): DeviceSchedulerState(device_id=DeviceID("d0"))}
    fold_cluster_amendment(states, ClusterAmendment(
        cluster_round=1, registry_deltas={DeviceID("d0"): deltas[DeviceID("d0")]}))
    assert states[DeviceID("d0")].spectrum_snr_db == {"wide": 12.5, "medium": 7.0}


def test_a_banded_mission_forwards_its_contact_snr_to_the_mules_scheduler():
    """End to end in process: a banded mission's Pass-1 lines reach the cluster,
    whose inter-pass DOWN carries each device's SNR to the mule's scheduler."""
    from hermes.l1.contact_link import CLASSES_WITH_10MHZ
    from hermes.mule.ferry import FerrySpec
    from tests.integration import _ferry_harness as H

    layout = (("dev-a", (10.0, 0.0, 0.0)), ("dev-b", (0.0, 20.0, 0.0)))
    w = H.World(layout=layout, flaky={})
    w.cluster._sim_clock = True                     # the process builds it with sim_clock=True
    w.cluster._band_names = tuple(CLASSES_WITH_10MHZ)
    spec = FerrySpec.from_config(rf_range_m=60.0, seed=3, contact_band="wide")
    sup = w.supervisor("mule-g", sim=True, ferry=spec)
    with H.Patched(w.clock):
        w.bootstrap()
        result = sup.run_one_mission()
    lines = {str(l.device_id): l for l in result.report.lines}
    assert set(lines) == {"dev-a", "dev-b"}
    for did, line in lines.items():
        assert line.band == 0 and line.snr_db is not None
        state = sup.scheduler.device_states[DeviceID(did)]
        assert state.spectrum_snr_db == {"wide": line.snr_db}
    # The DOWN also carried the cluster's simulated time: the upload's.
    (up,) = w.server.ups
    assert w.server.downs[-1][1].cluster_sim_ts == up.sim_upload_ts


def test_the_recorded_cluster_forwards_no_spectrum():
    c = _cluster(sim=False)
    c.ingest_up_bundle(_up("m1", 1, lines=[_line("d0", 0, 12.5)]))
    assert c.registry.get(DeviceID("d0")).spectrum_sig.contact_class_snr_db is None
    for mule in ("m1", "m2"):
        for patch in c.dispatch_down_bundle(MuleID(mule)).cluster_amendments.registry_deltas.values():
            assert set(patch) == {"last_known_position", "delivery_priority"}


def test_spectrum_sig_equality_and_hash_ignore_the_contact_readings():
    a = SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,))
    b = a.with_contact_class_snr({"wide": 3.0, "medium": float("nan"), "narrow": True})
    assert b.contact_class_snr_db == {"wide": 3.0} and a.contact_class_snr_db is None
    assert a == b and hash(a) == hash(b)
    assert a.with_contact_class_snr({}) is a


def test_deadline_overrides_are_refused_on_the_simulated_clock():
    c = _cluster(sim=True)
    with pytest.raises(ValueError, match="wall-clock stamps"):
        c.close_cluster_round(deadline_overrides={DeviceID("d0"): 1.7e9})
    assert c.cluster_round == 0
    with pytest.raises(ValueError, match="wall-clock stamps"):
        c.dispatch_down_bundle(MuleID("m1"), amendment=ClusterAmendment(
            cluster_round=0, deadline_overrides={DeviceID("d0"): 1.7e9}))
    wall = _cluster(sim=False)
    wall.close_cluster_round(deadline_overrides={DeviceID("d0"): 1.7e9})   # recorded: allowed
    assert wall.dispatch_down_bundle(MuleID("m1")).cluster_amendments.deadline_overrides


# --------------------------------------------------------------------------- #
# The lost-upload stand-in (critic B8)
# --------------------------------------------------------------------------- #

def test_the_stand_in_keeps_the_lost_uploads_simulated_time_and_pricing():
    spec = AggregationSpec(rule=AGG_CUTOFF)
    up = _up("m1", 3, spec=spec, sim_ts=T0 + 321.5, p_loss=0.4, base_version=2)
    held = _lost_upload_stand_in(up, 3, spec)
    assert held.sim_upload_ts == T0 + 321.5 and held.backhaul is up.backhaul
    assert held.partial_aggregate.is_empty() and held.partial_aggregate.base_version == 2


def test_a_recorded_stand_in_is_the_afa9526_one():
    spec = AggregationSpec()
    held = _lost_upload_stand_in(_up("m1", 2, spec=spec), 2, spec)
    assert held == UpBundle(
        mule_id=MuleID("m1"),
        partial_aggregate=PartialAggregate(
            mule_id=MuleID("m1"), mission_round=2, weights=[], num_examples=0,
            rule=spec.rule, update_form=spec.update_form, base_version=0,
        ),
        round_close_report=MissionRoundCloseReport(
            mule_id=MuleID("m1"), mission_round=2, started_at=0.0, finished_at=0.0,
        ),
        contact_history=ContactHistory(mule_id=MuleID("m1"), mission_round=2),
    )


# --------------------------------------------------------------------------- #
# ClusterService: the seconds model's keyed draw and the sim-mode events
# --------------------------------------------------------------------------- #

class _Dock:
    """A stand-in dock: hands out the scripted UPs, records every DOWN bundle.

    A list in the script sets which mules are registered from then on (that
    call times out), as ``tests/unit/test_multi_mule_dock.py``'s dock does.
    """

    def __init__(self, svc, script, registered):
        self._svc, self._script = svc, list(script)
        self.registered = list(registered)
        self.downs = []

    def registered_mules(self):
        return list(self.registered)

    def wait_for_mules(self, mules, timeout=None):
        return set(mules) <= set(self.registered)

    def recv_up(self, timeout=None):
        if not self._script:
            self._svc.request_stop()
            raise TimeoutError("script exhausted")
        step = self._script.pop(0)
        if isinstance(step, list):
            self.registered = list(step)
            raise TimeoutError("no UP this tick")
        return step

    def send_down(self, bundle):
        self.downs.append(bundle)

    def close(self):
        return


def _service(*, clock="sim", model="seconds", mules=("m1",), min_participation=1,
             spec=None, **kw) -> ClusterService:
    spec = spec or AggregationSpec()
    extra = dict(mission_clock=clock)
    if clock == "sim":
        extra.update(backhaul_model=model, trial_seed=SEED)
    extra.update(kw)
    cfg = ClusterConfig(
        cluster_id="c-sim", dock_port=0, expected_mules=list(mules),
        seed_devices=[{"device_id": f"d{i}", "position": [float(i), 0.0, 0.0],
                       "assigned_mule": mules[i % len(mules)]} for i in range(3)],
        synth_batch_size=1, min_participation=min_participation,
        aggregation=spec.rule, aggregation_params=spec.to_params(), **extra,
    )
    svc = ClusterService(cfg)
    svc.dock.close()
    return svc


def _run(svc, script, registered=("m1",)):
    events = []
    svc.events.emit = lambda event, **fields: events.append((event, fields))
    dock = _Dock(svc, script, registered)
    svc.dock = dock
    try:
        svc.run()
    finally:
        svc.shutdown()
    return dock, events


def _named(events, name):
    return [f for e, f in events if e == name]


def test_the_seconds_draw_is_keyed_by_seed_mule_and_round():
    salt = ferry_salt(SEED, SALT_BACKHAUL_LOSS)
    svc = _service()
    try:
        for rnd in range(1, 40):
            u = keyed_uniform(salt, "m1", rnd)
            for p in (0.0, 0.25, 0.5, 0.9, 1.0):
                assert svc._upload_lost(_up("m1", rnd, sim_ts=T0, p_loss=p), rnd) == (u < p)
    finally:
        svc.shutdown()


def test_the_seconds_draw_is_paired_by_mission_not_by_upload_count():
    """Two arms that docked on different missions face the same uniform for the
    same mission (the recorded stream paired by upload count instead)."""
    a, b = _service(), _service()
    try:
        seq_a = [a._upload_lost(_up("m1", r, sim_ts=T0, p_loss=0.5), r) for r in (1, 2, 3, 4)]
        seq_b = [b._upload_lost(_up("m1", r, sim_ts=T0, p_loss=0.5), r) for r in (4, 3, 1)]
        assert [seq_a[3], seq_a[2], seq_a[0]] == seq_b
        # Another mule's uploads and the arrival order move nothing.
        b._upload_lost(_up("m2", 2, sim_ts=T0, p_loss=0.5), 2)
        assert b._upload_lost(_up("m1", 2, sim_ts=T0, p_loss=0.5), 2) == seq_a[1]
    finally:
        a.shutdown()
        b.shutdown()


def test_below_the_floor_is_always_lost_and_an_unpriced_upload_never_is():
    svc = _service()
    try:
        assert all(svc._upload_lost(_up("m1", r, sim_ts=T0, p_loss=1.0), r) for r in range(1, 50))
        assert not svc._upload_lost(_up("m1", 1, sim_ts=T0), 1)
        assert svc.metrics.snapshot()["counter.backhaul_unpriced_uploads"] == 1
    finally:
        svc.shutdown()


def test_an_empty_partials_upload_is_drawn_too():
    svc = _service()
    try:
        assert svc._upload_lost(_up("m1", 1, sim_ts=T0, p_loss=1.0, empty=True), 1)
    finally:
        svc.shutdown()


def test_the_mission_model_on_the_sim_clock_keeps_the_recorded_stream():
    sim = _service(model="mission", backhaul_loss_pct=40.0, backhaul_rng_seed=11)
    wall = _service(clock="wall", backhaul_loss_pct=40.0, backhaul_rng_seed=11)
    try:
        ups = [_up("m1", r, sim_ts=T0 + r, p_loss=1.0) for r in range(1, 30)]
        assert [sim._upload_lost(u, r) for r, u in enumerate(ups, 1)] == \
            [wall._upload_lost(u, r) for r, u in enumerate(ups, 1)]
    finally:
        sim.shutdown()
        wall.shutdown()


def test_the_seconds_model_is_refused_off_the_clock():
    with pytest.raises(ValueError, match="critic B16"):
        ClusterService(ClusterConfig(cluster_id="c", backhaul_model="seconds"))


def test_sim_events_carry_the_simulated_fields_and_the_downs_echo_the_time():
    svc = _service()
    lost = next(r for r in range(1, 200)
                if keyed_uniform(ferry_salt(SEED, SALT_BACKHAUL_LOSS), "m1", r) < 0.5)
    kept = next(r for r in range(1, 200)
                if keyed_uniform(ferry_salt(SEED, SALT_BACKHAUL_LOSS), "m1", r) >= 0.5)
    dock, events = _run(svc, [
        _up("m1", kept, sim_ts=T0 + 100.0, p_loss=0.5),
        _up("m1", lost, sim_ts=T0 + 300.0, p_loss=0.5),
    ])
    (ingested,) = _named(events, "up_bundle_ingested")
    assert ingested == {"mule_id": "m1", "mission_round": kept, "sim_upload_ts": T0 + 100.0,
                        "carrier": 2, "snr_db": 9.5, "p_loss": 0.5}
    (lost_ev,) = _named(events, "backhaul_upload_lost")
    assert lost_ev["sim_upload_ts"] == T0 + 300.0 and lost_ev["p_loss"] == 0.5
    (closed,) = _named(events, "cluster_round_closed")
    assert closed == {"cluster_round": 1, "sim_ts": T0 + 100.0}
    # bootstrap, the merge's answer, the lost upload's answer: the latest ingested time
    assert [d.cluster_sim_ts for d in dock.downs] == [None, T0 + 100.0, T0 + 100.0]


class _Recorder:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))

    def close(self):
        return


@pytest.mark.parametrize("clock", ["sim", "wall"])
def test_cluster_ready_names_the_clock_only_on_the_simulated_one(clock):
    rec = _Recorder()
    cfg = ClusterConfig(cluster_id="c", dock_port=0, **(
        dict(mission_clock="sim", backhaul_model="seconds", trial_seed=1) if clock == "sim" else {}))
    svc = ClusterService(cfg, events=rec)
    svc.shutdown()
    (ready,) = _named(rec.events, "cluster_ready")
    extra = set(ready) - {"dock_host", "dock_port", "expected_mules", "seed_devices",
                          "synth_batch_size", "min_participation", "tier3_wired"}
    if clock == "sim":
        assert extra == {"mission_clock", "backhaul_model"}
        assert (ready["mission_clock"], ready["backhaul_model"]) == ("sim", "seconds")
    else:
        assert extra == set()


def test_a_restarted_mules_bootstrap_carries_the_clusters_simulated_time():
    """Critic B8: a mule restarted with a fresh clock at the epoch adopts the
    cluster's simulated time from its bootstrap DOWN (``advance_to``)."""
    svc = _service()
    dock, events = _run(svc, [
        _up("m1", 1, sim_ts=T0 + 250.0, p_loss=0.0),
        [],                       # m1's process is gone
        ["m1"],                   # ... and back, with a fresh clock
    ])
    boots = _named(events, "mule_bootstrapped")
    assert len(boots) == 2
    assert dock.downs[0].cluster_sim_ts is None           # the first bootstrap
    assert dock.downs[-1].cluster_sim_ts == T0 + 250.0    # the restarted mule's


def test_a_quorum_down_after_a_held_loss_carries_the_lost_uploads_time():
    """Critic B8: at quorum 2 the lost UP's stand-in holds its place and its
    time, so the DOWN that answers both mules is not behind either upload."""
    svc = _service(mules=("m1", "m2"), min_participation=2)
    svc._upload_lost = lambda up, rnd: str(up.mule_id) == "m2"
    dock, events = _run(svc, [
        _up("m1", 1, sim_ts=T0 + 50.0, p_loss=0.1),
        _up("m2", 1, sim_ts=T0 + 80.0, p_loss=0.9),
    ], registered=("m1", "m2"))
    (held,) = _named(events, "backhaul_upload_lost")
    assert held["awaits_quorum"] is True and held["sim_upload_ts"] == T0 + 80.0
    answers = [d for d in dock.downs[2:]]
    assert sorted(str(d.mule_id) for d in answers) == ["m1", "m2"]
    assert {d.cluster_sim_ts for d in answers} == {T0 + 80.0}
    (closed,) = _named(events, "cluster_round_closed")
    assert closed["sim_ts"] == T0 + 80.0


def test_wall_clock_events_are_the_recorded_ones():
    svc = _service(clock="wall", backhaul_loss_pct=0.0)
    dock, events = _run(svc, [_up("m1", 1, sim_ts=T0 + 100.0, p_loss=0.5)])
    assert _named(events, "up_bundle_ingested") == [{"mule_id": "m1", "mission_round": 1}]
    assert _named(events, "cluster_round_closed") == [{"cluster_round": 1}]
    assert all(d.cluster_sim_ts is None for d in dock.downs)


def test_model_eval_carries_the_simulated_time_on_the_clock(monkeypatch):
    import experiments.exp4.model_task as model_task

    monkeypatch.setattr(model_task, "evaluate_theta",
                        lambda theta, X, y, input_dim: {"accuracy": 0.5, "auc": 0.5, "loss": 1.0})
    for clock, expected in (("sim", [None, T0 + 7.0]), ("wall", None)):
        svc = _service(clock=clock, **({} if clock == "sim" else {"backhaul_loss_pct": 0.0}))
        svc._eval_X, svc._eval_y, svc._eval_input_dim = np.zeros((2, 3)), np.zeros(2), 3
        _dock, events = _run(svc, [_up("m1", 1, sim_ts=T0 + 7.0, p_loss=0.0)])
        evals = _named(events, "model_eval")
        assert len(evals) == 2
        if expected is None:
            assert all("sim_ts" not in e for e in evals)
        else:
            assert [e["sim_ts"] for e in evals] == expected


# --------------------------------------------------------------------------- #
# The recorded model's probability on sim events; refused partials; classes
# --------------------------------------------------------------------------- #

def test_sim_events_report_the_schedule_entry_the_recorded_draw_used():
    """The mission model on the clock: each event's ``p_loss`` is the entry
    the draw read (``mission_schedule_index``: round - 1, clamped past the
    end), so the event and the outcome can never disagree. The mule prices
    no carrier or SNR under this model."""
    svc = _service(model="mission", backhaul_loss_schedule=[0.0, 1.0], backhaul_rng_seed=11)
    _dock, events = _run(svc, [
        _up("m1", 1, sim_ts=T0 + 10.0),      # p = 0.0: kept
        _up("m1", 2, sim_ts=T0 + 20.0),      # p = 1.0: lost
        _up("m1", 3, sim_ts=T0 + 30.0),      # past the end: the last entry, lost
    ])
    assert _named(events, "up_bundle_ingested") == [{
        "mule_id": "m1", "mission_round": 1, "sim_upload_ts": T0 + 10.0,
        "carrier": None, "snr_db": None, "p_loss": 0.0,
    }]
    lost = _named(events, "backhaul_upload_lost")
    assert [(e["mission_round"], e["p_loss"], e["sim_upload_ts"]) for e in lost] == [
        (2, 1.0, T0 + 20.0), (3, 1.0, T0 + 30.0)]


def test_a_refused_partials_event_carries_its_simulated_fields():
    """A mule that flew on while its partial was held uploads again; the
    refused bundle is still logged as ingested, with its own upload's time
    and pricing (and its time counts toward the echo)."""
    svc = _service(mules=("m1", "m2"), min_participation=2)
    _dock, events = _run(svc, [
        _up("m1", 1, sim_ts=T0 + 10.0, p_loss=0.0),
        _up("m1", 2, sim_ts=T0 + 70.0, p_loss=0.0),
    ], registered=("m1", "m2"))
    kept, refused = _named(events, "up_bundle_ingested")
    assert "partial_refused" not in kept
    assert refused == {
        "mule_id": "m1", "mission_round": 2, "partial_refused": True, "held_mission_round": 1,
        "sim_upload_ts": T0 + 70.0, "carrier": 2, "snr_db": 9.5, "p_loss": 0.0,
    }
    assert svc.cluster.sim_ts == T0 + 70.0


@pytest.mark.parametrize("classes,name", [(None, "wide"), (["narrow", "wide"], "narrow")])
def test_the_service_reads_band_indices_with_its_configured_classes(classes, name):
    """``ClusterConfig.contact_band_classes`` reaches the cluster: a line's band
    index 0 is the first configured class (the D1 set when None)."""
    svc = _service(contact_band_classes=classes)
    try:
        svc.cluster.ingest_up_bundle(_up("m1", 1, sim_ts=T0 + 1.0, lines=[_line("d0", 0, 6.5)]))
        sig = svc.cluster.registry.get(DeviceID("d0")).spectrum_sig
        assert sig.contact_class_snr_db == {name: 6.5}
    finally:
        svc.shutdown()
