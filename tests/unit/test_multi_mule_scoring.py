"""FeRRy Phase 2 — scoring trials flown by several mules.

Every mule numbers its own missions from 1, so with two mules "mission round
2" names two missions. The consumer, the metrics and the trace scorer key
every per-mission fact by ``(mule_id, mission_round)``; each two-mule trial
below makes the rounds collide, and each test notes what the round-keyed
logic said instead, so it fails if the keying is reverted. Also pinned:

* Pass-2 coverage is a share of the mission's own mule's slice (from the
  mule configs), not of every device;
* a device's age counts its own mule's missions, and Network AoU is sampled
  after every mission of any mule;
* time to τ counts the reaching mule's missions;
* an upload the cluster logged but never folded (a duplicate it refused, or
  a partial still waiting for quorum at the end) merges nothing;
* an empty upload (``dock_on_empty``) is neither deferred nor expired;
* events that name no mule are placed by round, window or partials.

The last section pins that one mule changes nothing: the keyed ledger is the
round-keyed one exactly, an observation built from the round-keyed fields
scores as it did, and the recorded L1 cell scores column for column as it did
before multi-mule scoring existed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.analysis.traces_scorer import (
    age_profile,
    merged_devices,
    score_trial,
    tau_reach,
)
from experiments.exp4.driver import PROVENANCE_COLUMNS
from experiments.exp4.events_consumer import (
    Exp4Observation,
    MissionRecord,
    consume_run_dir,
    observation_from_rows,
)
from experiments.exp4.metrics import summarise_observation

A, B = "exp4-mule-0", "exp4-mule-1"
SLICES = {A: ("a1", "a2"), B: ("b1", "b2")}
DEVICES = ["a1", "a2", "b1", "b2"]


# --------------------------------------------------------------------------- #
# Building two-mule trials
# --------------------------------------------------------------------------- #

def _flight(mule, rnd, start, clean, *, delivered=2, scheduled=2, duration=10.0, **fields):
    """One mission's two mule rows: started at ``start``, completed ``duration`` later."""
    return [
        {"ts": start, "event": "mission_started", "role": "mule", "id": mule},
        {"ts": start + duration, "event": "mission_completed", "role": "mule", "id": mule,
         "mission_round": rnd, "pass_1_contacts": 1, "pass_2_contacts": 1,
         "pass_1_updates": len(clean), "pass_1_scheduled": scheduled,
         "pass_1_clean_devices": list(clean), "delivered": delivered, **fields},
    ]


def _cev(ts, event, **kw):
    return {"ts": ts, "event": event, "role": "cluster", "id": "exp4-cluster", **kw}


def _up(ts, mule, rnd):
    return _cev(ts, "up_bundle_ingested", mule_id=mule, mission_round=rnd)


def _obs(mule_rows, cluster_rows, *, slices=SLICES, n_devices=4):
    return observation_from_rows(
        cluster_rows=cluster_rows, mule_rows=mule_rows,
        device_rows=[{"event": "device_ready", "role": "device", "id": d} for d in DEVICES],
        n_devices=n_devices, mule_slices=slices,
    )


def _summary(obs, n_devices=4):
    return summarise_observation(obs, n_devices=n_devices, rf_range_m=60.0, n_missions_target=3)


def _mission(obs, mule, rnd):
    (m,) = [m for m in obs.missions if (m.mule_id, m.mission_round) == (mule, rnd)]
    return m


# --------------------------------------------------------------------------- #
# A plain trial: A's round-2 upload is lost; B's round 2 is not
# --------------------------------------------------------------------------- #
#
#   mission  window      Pass-1 CLEAN  delivered  cluster
#   A1       1000–1010   a1, a2        2 / 2      ingested @1005, closes r1 (acc .50)
#   B1       1002–1012   b1            2 / 2      ingested @1007, closes r2 (acc .60)
#   A2       1010–1020   a1            2 / 2      LOST @1015
#   B2       1012–1022   b1, b2        1 / 2      ingested @1017, closes r3 (acc .83)
#   A3       1020–1030   a2            1 / 2      ingested @1025, closes r4 (acc .86)
#   B3       1022–1032   b2            2 / 2      ingested @1027, closes r5 (acc .90)
#
# Completion order A1 B1 A2 B2 A3 B3. Ages (a1, a2, b1, b2), each counted in
# the device's own mule's missions, after each:
#   A1 (0,0,0,0)  B1 (0,0,0,1)  A2 (1,1,0,1)  B2 (1,1,0,0)  A3 (2,0,0,0)  B3 (2,0,1,0)
# Network AoU 0, 1/4, 3/4, 1/2, 1/2, 3/4 -> mean 2.75/6, final 3/4.

def _plain_mules():
    return [
        *_flight(A, 1, 1000.0, ["a1", "a2"]),
        *_flight(B, 1, 1002.0, ["b1"]),
        *_flight(A, 2, 1010.0, ["a1"]),
        *_flight(B, 2, 1012.0, ["b1", "b2"], delivered=1),
        *_flight(A, 3, 1020.0, ["a2"], delivered=1),
        *_flight(B, 3, 1022.0, ["b2"]),
    ]


def _plain_cluster(*, lost_round_recorded=True):
    lost = _cev(1015.0, "backhaul_upload_lost", mule_id=A, mission_round=2)
    if not lost_round_recorded:
        # Before Freeze Amendment 5 the loss carried no round: it is placed by
        # the window of the named mule's missions (A2), not B2's, which also
        # contains 1015.
        lost["mission_round"] = None
    return [
        _cev(999.0, "model_eval", cluster_round=0, accuracy=0.30, auc=0.3, loss=0.7, n_test=10),
        _up(1005.0, A, 1),
        _cev(1005.0, "cluster_round_closed", cluster_round=1),
        _cev(1006.0, "model_eval", cluster_round=1, accuracy=0.50, auc=0.5, loss=0.6, n_test=10),
        _up(1007.0, B, 1),
        _cev(1007.0, "cluster_round_closed", cluster_round=2),
        _cev(1008.0, "model_eval", cluster_round=2, accuracy=0.60, auc=0.6, loss=0.5, n_test=10),
        lost,
        _up(1017.0, B, 2),
        _cev(1017.0, "cluster_round_closed", cluster_round=3),
        _cev(1018.0, "model_eval", cluster_round=3, accuracy=0.83, auc=0.8, loss=0.4, n_test=10),
        _up(1025.0, A, 3),
        _cev(1025.0, "cluster_round_closed", cluster_round=4),
        _cev(1026.0, "model_eval", cluster_round=4, accuracy=0.86, auc=0.9, loss=0.3, n_test=10),
        _up(1027.0, B, 3),
        _cev(1027.0, "cluster_round_closed", cluster_round=5),
        _cev(1028.0, "model_eval", cluster_round=5, accuracy=0.90, auc=0.9, loss=0.2, n_test=10),
    ]


@pytest.mark.parametrize("recorded", [True, False], ids=["recorded", "placed-by-time"])
def test_a_lost_upload_marks_only_its_own_mules_round(recorded):
    obs = _obs(_plain_mules(), _plain_cluster(lost_round_recorded=recorded))
    assert obs.mule_ids == (A, B) and obs.n_mules == 2
    assert obs.backhaul_lost_keys == {(A, 2)}
    assert obs.backhaul_lost_rounds == {2}          # the projection, ambiguous here
    assert merged_devices(obs, _mission(obs, A, 2)) == ()
    assert merged_devices(obs, _mission(obs, B, 2)) == ("b1", "b2")
    # Five of six missions close; keyed by round alone B2 fell with A2 (4/6).
    assert _summary(obs).round_close_rate_kmin1 == pytest.approx(5 / 6)


def test_pass2_coverage_is_a_share_of_the_missions_own_slice():
    obs = _obs(_plain_mules(), _plain_cluster())
    # Deliveries 2, 2, 2, 1, 1, 2 out of 2 each; out of all 4 devices the
    # mean would be 2.5/6, however complete each mule's Pass 2 was.
    assert _summary(obs).pass2_coverage == pytest.approx(5 / 6)


def test_slice_sizes_come_from_the_configs_else_an_even_share():
    mules = [
        *_flight(A, 1, 1000.0, ["a1", "a2", "a3"], delivered=3, scheduled=3),
        *_flight(B, 1, 1002.0, ["b1"], delivered=1, scheduled=1),
    ]
    configured = _obs(mules, [], slices={A: ("a1", "a2", "a3"), B: ("b1",)})
    assert _summary(configured).pass2_coverage == pytest.approx(1.0)
    # No configs: each mule is taken to serve 4 / 2 devices, so A's 3 cap at
    # 1 and B's 1 is a half.
    unconfigured = _obs(mules, [], slices=None)
    assert unconfigured.mule_ids == (A, B) and unconfigured.mule_slices == {}
    assert _summary(unconfigured).pass2_coverage == pytest.approx(0.75)


def test_an_empty_plan_is_measured_against_its_own_slice():
    """A mission that scheduled nobody targets its mule's slice, not every
    device: here the full-quorum threshold is 2, which A's merge meets. With
    the population as the target it was 4, which no one mule's merge can."""
    mules = [
        *_flight(A, 1, 1000.0, ["a1", "a2"]),
        *_flight(B, 1, 1002.0, [], delivered=0, scheduled=0),
    ]
    obs = _obs(mules, [_up(1005.0, A, 1), _cev(1005.0, "cluster_round_closed", cluster_round=1)])
    assert _summary(obs).round_close_rate_kminN == pytest.approx(0.5)


def test_each_device_ages_in_its_own_mules_missions():
    ages = age_profile(_obs(_plain_mules(), _plain_cluster()), DEVICES)
    assert ages.n_missions == 6
    assert ages.network_aou_mean == pytest.approx(2.75 / 6)
    assert ages.network_aou_final == pytest.approx(0.75)
    assert ages.age_max == 2                      # a1, unmerged since A1
    assert ages.merged_updates == {"a1": 1, "a2": 2, "b1": 2, "b2": 2}


def test_a_device_no_config_places_is_placed_by_its_missions():
    """Without configs, a device belongs to the mule whose missions name it."""
    with_configs = age_profile(_obs(_plain_mules(), _plain_cluster()), DEVICES)
    without = age_profile(_obs(_plain_mules(), _plain_cluster(), slices=None), DEVICES)
    assert without == with_configs


def test_a_configured_device_no_mission_names_ages_in_its_own_mules_missions():
    """a3 is in A's slice, and no mission names it. Its age after each mission
    is A's mission count, 1 1 2 2 3 3, so Network AoU over five devices has
    sums 1 2 5 4 5 6 (the table above plus a3's)."""
    devices = [*DEVICES, "a3"]
    slices = {A: ("a1", "a2", "a3"), B: ("b1", "b2")}
    ages = age_profile(_obs(_plain_mules(), _plain_cluster(), slices=slices), devices)
    assert ages.age_max == 3
    assert ages.network_aou_final == pytest.approx(6 / 5)
    assert ages.network_aou_mean == pytest.approx(23 / 30)
    # A device no config and no mission places ages in every mission of the
    # fleet: six by the end.
    assert age_profile(_obs(_plain_mules(), _plain_cluster()), devices).age_max == 6


def test_time_to_tau_counts_the_reaching_mules_missions():
    obs = _obs(_plain_mules(), _plain_cluster())
    # r3 (acc .83) was B2's merge. A2's window also contains its evaluation at
    # 1018; in completion order across both mules that is mission 3.
    b = tau_reach(obs, 0.82)
    assert (b.mission, b.mule_id, b.cluster_round, b.wall_s) == (2, B, 3, 18.0)
    # r4 (acc .86) was A3's merge, evaluated at 1026 inside B3's window too.
    a = tau_reach(obs, 0.85)
    assert (a.mission, a.mule_id, a.cluster_round, a.wall_s) == (3, A, 4, 26.0)
    assert not tau_reach(obs, 0.95).reached


def test_without_the_closing_upload_the_reaching_mission_still_counts_in_its_mule():
    """No ingest rows: the evaluation is placed by the first window of any
    mule that holds it (A3's, before B3's), then counted among A's missions,
    not all six (5)."""
    cluster = [r for r in _plain_cluster() if r["event"] != "up_bundle_ingested"]
    reach = tau_reach(_obs(_plain_mules(), cluster), 0.85)
    assert (reach.mission, reach.mule_id) == (3, A)


def test_wall_clock_time_runs_from_the_fleets_first_start():
    """B took off first but landed second; the clock starts at its take-off."""
    mules = [*_flight(A, 1, 1000.0, ["a1"]), *_flight(B, 1, 996.0, ["b1"], duration=16.0)]
    cluster = [
        _up(1005.0, A, 1),
        _cev(1005.0, "cluster_round_closed", cluster_round=1),
        _cev(1006.0, "model_eval", cluster_round=1, accuracy=0.9, auc=0.9, loss=0.2, n_test=10),
    ]
    reach = tau_reach(_obs(mules, cluster), 0.82)
    assert (reach.mission, reach.mule_id, reach.wall_s) == (1, A, 10.0)


# --------------------------------------------------------------------------- #
# agg:fedbuff, K = 2 across mules: each B upload flushes A's deferral
# --------------------------------------------------------------------------- #
#
#   mission  CLEAN  cluster
#   A1       a1     deferred (1/2)
#   B1       b1     applied, partials (A,1) (B,1): flushes A1 — both round 1
#   A2       a2     deferred (1/2)
#   B2       b2     applied, partials (A,2) (B,2): flushes A2
#
# Ages (a1, a2, b1, b2): A1 (1,1,0,0)  B1 (0,1,0,1)  A2 (1,2,0,1)  B2 (1,0,1,0)
# -> Network AoU 1/2, 1/2, 1, 1/2; mean 5/8.

def _fedbuff_mules():
    return [
        *_flight(A, 1, 1000.0, ["a1"]),
        *_flight(B, 1, 1002.0, ["b1"]),
        *_flight(A, 2, 1010.0, ["a2"]),
        *_flight(B, 2, 1012.0, ["b2"]),
    ]


def _fedbuff_cluster(first, second):
    """``first``'s uploads are deferred, ``second``'s flush them. The applied
    merge names no mule: its uploader is the upload ingested just before it."""
    rows = []
    for rnd, t in ((1, 1005.0), (2, 1015.0)):
        rows += [
            _up(t, first, rnd),
            _cev(t, "cluster_merge_deferred", mule_id=first, mission_round=rnd,
                 rule="agg:fedbuff", applied=False, buffered=1, k=2,
                 partials=[[first, rnd]]),
            _up(t + 2, second, rnd),
            _cev(t + 2, "cluster_merge", mission_round=rnd, rule="agg:fedbuff", applied=True,
                 buffered=2, k=2, partials=[[first, rnd], [second, rnd]]),
            _cev(t + 2, "cluster_round_closed", cluster_round=rnd),
        ]
    return rows


def test_a_fedbuff_flush_credits_the_other_mules_deferral():
    obs = _obs(_fedbuff_mules(), _fedbuff_cluster(A, B))
    assert obs.deferred_keys == {(A, 1), (A, 2)}
    # Keyed by round alone the flushing merge's partials all equalled its own
    # round, so it flushed nothing and B1's round 1 counted as deferred.
    assert obs.flush_of_keys == {(A, 1): (B, 1), (A, 2): (B, 2)}
    assert merged_devices(obs, _mission(obs, A, 1)) == ()
    assert merged_devices(obs, _mission(obs, B, 1)) == ("b1", "a1")
    assert merged_devices(obs, _mission(obs, B, 2)) == ("b2", "a2")

    s = _summary(obs)
    assert s.round_close_rate_kmin1 == pytest.approx(0.5)     # B1 and B2
    assert s.round_close_rate_kmin2 == pytest.approx(0.5)     # each releases 2 updates

    ages = age_profile(obs, DEVICES)
    assert ages.merged_updates == {"a1": 1, "a2": 1, "b1": 1, "b2": 1}
    assert ages.network_aou_mean == pytest.approx(5 / 8)
    assert ages.network_aou_final == pytest.approx(0.5)
    assert ages.age_max == 2


def test_the_flushing_mission_is_the_uploader_of_the_merge():
    obs = _obs(_fedbuff_mules(), _fedbuff_cluster(B, A))
    assert obs.flush_of_keys == {(B, 1): (A, 1), (B, 2): (A, 2)}
    assert merged_devices(obs, _mission(obs, A, 1)) == ("a1", "b1")
    assert merged_devices(obs, _mission(obs, B, 1)) == ()


# --------------------------------------------------------------------------- #
# Age-aware folds that cut one mule's partial
# --------------------------------------------------------------------------- #

def test_a_partial_cut_inside_an_applied_fold_expires_only_that_mission():
    """agg:cutoff at quorum 2: A2 waits, B2's upload folds both and cuts A2."""
    cluster = [
        _up(1005.0, A, 1),
        _cev(1005.0, "cluster_merge", mission_round=1, rule="agg:cutoff", applied=True,
             partials=[[A, 1]], expired_partials=[]),
        _up(1007.0, B, 1),
        _cev(1007.0, "cluster_merge", mission_round=1, rule="agg:cutoff", applied=True,
             partials=[[B, 1]], expired_partials=[]),
        _up(1015.0, A, 2),
        _up(1017.0, B, 2),
        _cev(1017.0, "cluster_merge", mission_round=2, rule="agg:cutoff", applied=True,
             partials=[[B, 2]], expired_partials=[[A, 2]]),
    ]
    obs = _obs(_fedbuff_mules(), cluster)
    assert obs.expired_keys == {(A, 2)}
    assert merged_devices(obs, _mission(obs, A, 2)) == ()
    assert merged_devices(obs, _mission(obs, B, 2)) == ("b2",)
    # Keyed by round, B2 expired with A2 (2/4).
    assert _summary(obs).round_close_rate_kmin1 == pytest.approx(3 / 4)


def test_an_expired_fold_expires_every_partial_it_names_and_no_other():
    """A's round-1 partial waits for quorum; B lost its round 1, and B2's
    upload expires the fold of (A,1) and (B,2). A2 later merges with B3."""
    mules = [
        *_flight(A, 1, 1000.0, ["a1"]),
        *_flight(B, 1, 1002.0, ["b1"]),
        *_flight(A, 2, 1010.0, ["a2"]),
        *_flight(B, 2, 1012.0, ["b2"]),
        *_flight(B, 3, 1022.0, ["b1"]),
    ]
    cluster = [
        _up(1005.0, A, 1),
        _cev(1007.0, "backhaul_upload_lost", mule_id=B, mission_round=1),
        _up(1017.0, B, 2),
        _cev(1017.0, "cluster_merge_expired", mule_id=B, mission_round=2, rule="agg:cutoff",
             applied=False, outcome="expired", partials=[[A, 1], [B, 2]]),
        _up(1019.0, A, 2),
        _up(1027.0, B, 3),
        _cev(1027.0, "cluster_merge", mission_round=3, rule="agg:cutoff", applied=True,
             partials=[[A, 2], [B, 3]], expired_partials=[]),
        _cev(1027.0, "cluster_round_closed", cluster_round=1),
    ]
    obs = _obs(mules, cluster)
    assert obs.expired_keys == {(A, 1), (B, 2)}
    assert obs.backhaul_lost_keys == {(B, 1)}
    assert merged_devices(obs, _mission(obs, A, 2)) == ("a2",)
    # A2 and B3 close; keyed by round, A2 expired with B2 (1/5).
    assert _summary(obs).round_close_rate_kmin1 == pytest.approx(2 / 5)


# --------------------------------------------------------------------------- #
# Uploads the cluster logs but never folds
# --------------------------------------------------------------------------- #
#
# Quorum 2, and A stops waiting for its DOWN (down_wait_s), so it flies on
# while its first partial waits for B's. The cluster keeps one partial per
# mule in the open round and refuses A's next two uploads as duplicates, but
# logs their ingest all the same. B's slow first mission completes the quorum;
# its second then waits alone until the trial ends, and its third is refused
# too. As in a real two-mule run of this shape, only A1 and B1 reach θ.
#
#   mission  window      cluster
#   A1       1000–1010   ingested @1003, waits for quorum
#   A2       1010–1020   ingested @1013: refused, A1 still waiting
#   A3       1020–1030   ingested @1023: refused
#   B1       1000–1032   ingested @1031: folds A1 and B1, closes r1
#   B2       1032–1045   ingested @1040, waits; no quorum before the end
#   B3       1045–1060   ingested @1055: refused, B2 still waiting

def _quorum_mules():
    return [
        *_flight(A, 1, 1000.0, ["a1", "a2"]),
        *_flight(B, 1, 1000.0, ["b1", "b2"], duration=32.0),
        *_flight(A, 2, 1010.0, ["a1", "a2"]),
        *_flight(A, 3, 1020.0, ["a1", "a2"]),
        *_flight(B, 2, 1032.0, ["b1", "b2"], duration=13.0),
        *_flight(B, 3, 1045.0, ["b1", "b2"], duration=15.0),
    ]


def _quorum_cluster(rule):
    """agg:plain logs no fold, only the close; an age-aware rule logs both."""
    fold = [] if rule == "agg:plain" else [
        _cev(1031.0, "cluster_merge", mission_round=1, rule=rule, applied=True,
             partials=[[A, 1], [B, 1]], expired_partials=[]),
    ]
    return [
        _up(1003.0, A, 1), _up(1013.0, A, 2), _up(1023.0, A, 3),
        _up(1031.0, B, 1), *fold,
        _cev(1031.0, "cluster_round_closed", cluster_round=1),
        _up(1040.0, B, 2), _up(1055.0, B, 3),
    ]


@pytest.mark.parametrize("rule", ["agg:plain", "agg:cutoff"])
def test_an_upload_the_cluster_never_folded_is_not_merged(rule):
    obs = _obs(_quorum_mules(), _quorum_cluster(rule))
    assert obs.unmerged_keys == {(A, 2), (A, 3), (B, 2), (B, 3)}
    assert (obs.backhaul_lost_keys, obs.expired_keys, obs.deferred_keys) == (set(), set(), set())
    for m in obs.missions:
        want = m.pass_1_clean_devices if m.mission_round == 1 else ()
        assert (m.mule_id, m.mission_round, merged_devices(obs, m)) == (
            m.mule_id, m.mission_round, want,
        )
    # A1 and B1 close. Crediting every completed mission, all six did, and
    # each device was merged three times.
    assert _summary(obs).round_close_rate_kmin1 == pytest.approx(2 / 6)
    assert age_profile(obs, DEVICES).merged_updates == {"a1": 1, "a2": 1, "b1": 1, "b2": 1}


def test_a_fold_that_lists_its_partials_has_the_last_word():
    """A failed merge resets the open round, logging only to stderr, so A2's
    upload is accepted though the replay takes it for a duplicate of A1's.
    B1's fold names A2 and B1: A2 reached θ, and A1, open but unnamed, did not."""
    mules = [
        *_flight(A, 1, 1000.0, ["a1"]),
        *_flight(A, 2, 1010.0, ["a2"]),
        *_flight(B, 1, 1000.0, ["b1"], duration=22.0),
    ]
    cluster = [
        _up(1003.0, A, 1),
        _up(1013.0, A, 2),
        _up(1021.0, B, 1),
        _cev(1021.0, "cluster_merge", mission_round=1, rule="agg:cutoff", applied=True,
             partials=[[A, 2], [B, 1]], expired_partials=[]),
        _cev(1021.0, "cluster_round_closed", cluster_round=1),
    ]
    obs = _obs(mules, cluster)
    assert obs.unmerged_keys == {(A, 1)}
    assert merged_devices(obs, _mission(obs, A, 1)) == ()
    assert merged_devices(obs, _mission(obs, A, 2)) == ("a2",)


def test_a_deferral_empties_the_open_round():
    """FedBuff moves each upload into its buffer, so A2 was no duplicate of A1:
    both are deferred (never flushed here), neither unmerged."""
    cluster = [
        _up(1005.0, A, 1),
        _cev(1005.0, "cluster_merge_deferred", mule_id=A, mission_round=1, rule="agg:fedbuff",
             applied=False, buffered=1, k=3, partials=[[A, 1]]),
        _up(1015.0, A, 2),
        _cev(1015.0, "cluster_merge_deferred", mule_id=A, mission_round=2, rule="agg:fedbuff",
             applied=False, buffered=2, k=3, partials=[[A, 1], [A, 2]]),
    ]
    obs = _obs([*_flight(A, 1, 1000.0, ["a1"]), *_flight(A, 2, 1010.0, ["a2"])], cluster)
    assert obs.deferred_keys == {(A, 1), (A, 2)}
    assert obs.unmerged_keys == set()


# --------------------------------------------------------------------------- #
# Empty uploads (dock_on_empty)
# --------------------------------------------------------------------------- #

def _empty(mule, rnd, ts):
    """The mule's report of a mission with nothing to upload that docked anyway."""
    return {"ts": ts, "event": "mission_empty", "role": "mule", "id": mule,
            "mission_round": rnd, "docked": True}


def test_an_all_empty_round_under_agg_plain_is_neither_expired_nor_left_open():
    """Both round-1 missions docked empty; agg:plain has nothing to average and
    expires the round, which empties it, so the round-2 uploads are accepted
    and close it. The empty uploads count as empty missions, not expired ones."""
    mules = [
        *_flight(A, 1, 1000.0, [], delivered=0), _empty(A, 1, 1010.0),
        *_flight(B, 1, 1002.0, [], delivered=0), _empty(B, 1, 1012.0),
        *_flight(A, 2, 1010.0, ["a1"]),
        *_flight(B, 2, 1012.0, ["b1"]),
    ]
    cluster = [
        _up(1005.0, A, 1), _up(1007.0, B, 1),
        _cev(1007.0, "cluster_merge_expired", mule_id=B, mission_round=1, rule="agg:plain",
             applied=False, outcome="expired", partials=[[A, 1], [B, 1]]),
        _up(1015.0, A, 2), _up(1017.0, B, 2),
        _cev(1017.0, "cluster_round_closed", cluster_round=1),
    ]
    obs = _obs(mules, cluster)
    assert (obs.expired_keys, obs.unmerged_keys) == (set(), set())
    assert obs.missions_empty == 2
    assert merged_devices(obs, _mission(obs, A, 2)) == ("a1",)
    assert _summary(obs).round_close_rate_kmin1 == pytest.approx(2 / 4)


def test_an_empty_upload_left_waiting_for_quorum_is_not_unmerged():
    """B2 docked empty and waited for a quorum that never came: it held B's
    place in the open round, but carried no update that failed to reach θ."""
    mules = [
        *_flight(A, 1, 1000.0, ["a1"]),
        *_flight(B, 1, 1002.0, ["b1"]),
        *_flight(B, 2, 1012.0, [], delivered=0), _empty(B, 2, 1022.0),
    ]
    cluster = [
        _up(1005.0, A, 1), _up(1007.0, B, 1),
        _cev(1007.0, "cluster_round_closed", cluster_round=1),
        _up(1017.0, B, 2),
    ]
    obs = _obs(mules, cluster)
    assert obs.unmerged_keys == set()
    assert obs.missions_empty == 1


def test_a_fold_cutting_an_empty_upload_does_not_expire_it():
    """agg:cutoff lists B2's empty partial with the expired ones (it has no
    weight) beside A3's genuinely stale one; only A3's expired."""
    mules = [
        *_flight(A, 1, 1000.0, ["a1"]), *_flight(B, 1, 1002.0, ["b1"]),
        *_flight(A, 2, 1010.0, ["a2"]),
        *_flight(B, 2, 1012.0, [], delivered=0), _empty(B, 2, 1022.0),
        *_flight(A, 3, 1020.0, ["a1"]), *_flight(B, 3, 1022.0, ["b2"]),
    ]

    def fold(t, rnd, live, cut):
        return _cev(t, "cluster_merge", mission_round=rnd, rule="agg:cutoff", applied=True,
                    partials=live, expired_partials=cut)

    cluster = [
        _up(1005.0, A, 1), _up(1007.0, B, 1), fold(1007.0, 1, [[A, 1], [B, 1]], []),
        _cev(1007.0, "cluster_round_closed", cluster_round=1),
        _up(1015.0, A, 2), _up(1017.0, B, 2), fold(1017.0, 2, [[A, 2]], [[B, 2]]),
        _cev(1017.0, "cluster_round_closed", cluster_round=2),
        _up(1025.0, A, 3), _up(1027.0, B, 3), fold(1027.0, 3, [[B, 3]], [[A, 3]]),
        _cev(1027.0, "cluster_round_closed", cluster_round=3),
    ]
    obs = _obs(mules, cluster)
    assert obs.expired_keys == {(A, 3)}
    assert obs.unmerged_keys == set()
    assert obs.missions_empty == 1


# --------------------------------------------------------------------------- #
# A lost upload under a quorum: the cluster's own flags
# --------------------------------------------------------------------------- #
#
# Quorum 2. B1's upload is lost, so the cluster holds an empty partial in B's
# place (``awaits_quorum``). B stops waiting (``down_wait_s``) and uploads B2,
# which the round refuses: it already holds B's place (``partial_refused``).
# A1 then completes the quorum and closes r1. Only A1 reached θ.

def _held_place_mules():
    return [
        *_flight(A, 1, 1000.0, ["a1"], duration=30.0),
        *_flight(B, 1, 1000.0, ["b1"], duration=8.0),
        *_flight(B, 2, 1010.0, ["b2"], duration=8.0),
    ]


def _lost(ts, mule, rnd, **kw):
    return _cev(ts, "backhaul_upload_lost", mule_id=mule, mission_round=rnd, **kw)


def test_a_refused_upload_behind_a_held_place_is_not_merged():
    cluster = [
        _lost(1005.0, B, 1, awaits_quorum=True),
        _cev(1015.0, "up_bundle_ingested", mule_id=B, mission_round=2,
             partial_refused=True, held_mission_round=1),
        _up(1025.0, A, 1),
        _cev(1025.0, "cluster_round_closed", cluster_round=1),
    ]
    obs = _obs(_held_place_mules(), cluster)
    # Without the flags the replay took B2 for B's open partial and cleared it
    # at the close, crediting it as merged (close rate 2/3).
    assert obs.unmerged_keys == {(B, 2)}
    assert obs.backhaul_lost_keys == {(B, 1)}
    assert obs.expired_keys == set()
    assert merged_devices(obs, _mission(obs, B, 2)) == ()
    assert merged_devices(obs, _mission(obs, A, 1)) == ("a1",)
    assert _summary(obs).round_close_rate_kmin1 == pytest.approx(1 / 3)
    assert age_profile(obs, DEVICES).merged_updates == {"a1": 1, "a2": 0, "b1": 0, "b2": 0}
    assert obs.up_bundles_ingested == 1                  # the refused one is not


def test_a_refused_upload_behind_a_held_place_is_not_merged_when_the_fold_expires():
    """The same, with the fold expiring (every partial empty or past its
    cutoff): the refused B2 was never in it, and nothing reached θ."""
    mules = [
        *_flight(A, 1, 1000.0, [], delivered=0, duration=30.0), _empty(A, 1, 1030.0),
        *_flight(B, 1, 1000.0, ["b1"], duration=8.0),
        *_flight(B, 2, 1010.0, ["b2"], duration=8.0),
    ]
    cluster = [
        _lost(1005.0, B, 1, awaits_quorum=True),
        _cev(1015.0, "up_bundle_ingested", mule_id=B, mission_round=2,
             partial_refused=True, held_mission_round=1),
        _up(1025.0, A, 1),
        _cev(1025.0, "cluster_merge_expired", mule_id=A, mission_round=1, rule="agg:plain",
             applied=False, outcome="expired", partials=[[B, 1], [A, 1]]),
    ]
    obs = _obs(mules, cluster)
    assert obs.unmerged_keys == {(B, 2)}
    # B1 is a loss and A1 an empty upload: neither carried an update to expire.
    assert obs.expired_keys == set()
    assert _summary(obs).round_close_rate_kmin1 == pytest.approx(0.0)
    assert age_profile(obs, DEVICES).merged_updates == {"a1": 0, "a2": 0, "b1": 0, "b2": 0}


def test_the_held_place_alone_marks_the_later_upload_a_duplicate():
    """Without the refusal flag, the replay still knows the round holds B's
    place from the loss, so B2 is a duplicate it never folded."""
    cluster = [
        _lost(1005.0, B, 1, awaits_quorum=True),
        _up(1015.0, B, 2),
        _up(1025.0, A, 1),
        _cev(1025.0, "cluster_round_closed", cluster_round=1),
    ]
    obs = _obs(_held_place_mules(), cluster)
    assert obs.unmerged_keys == {(B, 2)}
    # A loss without a held place (quorum 1) leaves the round as it was.
    cluster[0] = _lost(1005.0, B, 1)
    assert _obs(_held_place_mules(), cluster).unmerged_keys == set()


def test_a_resend_of_the_held_mission_is_not_marked_twice():
    """A refused upload of the very mission the round holds changes nothing."""
    mules = [*_flight(A, 1, 1000.0, ["a1"], duration=30.0), *_flight(B, 1, 1000.0, ["b1"])]
    cluster = [
        _up(1005.0, B, 1),
        _cev(1006.0, "up_bundle_ingested", mule_id=B, mission_round=1,
             partial_refused=True, held_mission_round=1),
        _up(1025.0, A, 1),
        _cev(1025.0, "cluster_round_closed", cluster_round=1),
    ]
    obs = _obs(mules, cluster)
    assert obs.unmerged_keys == set()
    assert merged_devices(obs, _mission(obs, B, 1)) == ("b1",)


@pytest.mark.parametrize("rule", ["agg:cutoff", "agg:fedex"])
def test_a_held_place_listed_with_the_expired_partials_counts_only_as_lost(rule):
    """The fold lists B1's held place apart from what it merged (both rules
    do). B1 is a backhaul loss, and only that: it carried no update to expire,
    so ``expired_missions`` is the same whichever rule ran the fold."""
    mules = [*_flight(A, 1, 1000.0, ["a1"], duration=30.0), *_flight(B, 1, 1000.0, ["b1"])]
    cluster = [
        _lost(1005.0, B, 1, awaits_quorum=True),
        _up(1025.0, A, 1),
        _cev(1025.0, "cluster_merge", mission_round=1, rule=rule, applied=True,
             partials=[[A, 1]], expired_partials=[[B, 1]]),
        _cev(1025.0, "cluster_round_closed", cluster_round=1),
    ]
    obs = _obs(mules, cluster)
    assert obs.backhaul_lost_keys == {(B, 1)}
    assert obs.expired_keys == set()                     # was {(B, 1)}: counted twice
    assert obs.unmerged_keys == set()


def test_a_fold_run_by_a_held_place_belongs_to_its_mule():
    """A2's upload is lost after B2's ingest; the held place completes the
    quorum and runs the fold. The merge names no mule: it is A2's, not B2's
    (the last ingested upload)."""
    mules = [
        *_flight(A, 1, 1000.0, ["a1"]), *_flight(B, 1, 1000.0, ["b1"]),
        *_flight(A, 2, 1010.0, ["a2"], duration=12.0),
        *_flight(B, 2, 1010.0, ["b2"]),
    ]
    cluster = [
        _up(1005.0, A, 1), _up(1006.0, B, 1),
        _cev(1006.0, "cluster_round_closed", cluster_round=1),
        _up(1015.0, B, 2),
        _lost(1021.0, A, 2, awaits_quorum=True),
        _cev(1021.0, "cluster_round_closed", cluster_round=2, mule_id=A),
    ]
    from experiments.exp4.events_consumer import _round_closers
    assert _round_closers(cluster) == {1: B, 2: A}
    obs = _obs(mules, cluster)
    assert obs.backhaul_lost_keys == {(A, 2)}
    assert obs.unmerged_keys == set()
    assert merged_devices(obs, _mission(obs, B, 2)) == ("b2",)


def test_an_empty_upload_fedbuff_never_buffered_is_not_deferred():
    """B1 docked empty; FedBuff buffers nothing for it but still reports the
    upload deferred. A2's upload flushes A1 alone."""
    mules = [
        *_flight(A, 1, 1000.0, ["a1"]),
        *_flight(B, 1, 1002.0, [], delivered=0), _empty(B, 1, 1012.0),
        *_flight(A, 2, 1010.0, ["a2"]),
    ]
    cluster = [
        _up(1005.0, A, 1),
        _cev(1005.0, "cluster_merge_deferred", mule_id=A, mission_round=1, rule="agg:fedbuff",
             applied=False, buffered=1, k=2, partials=[[A, 1]]),
        _up(1007.0, B, 1),
        _cev(1007.0, "cluster_merge_deferred", mule_id=B, mission_round=1, rule="agg:fedbuff",
             applied=False, buffered=1, k=2, partials=[[A, 1]]),
        _up(1015.0, A, 2),
        _cev(1015.0, "cluster_merge", mission_round=2, rule="agg:fedbuff", applied=True,
             buffered=2, k=2, partials=[[A, 1], [A, 2]]),
        _cev(1015.0, "cluster_round_closed", cluster_round=1),
    ]
    obs = _obs(mules, cluster)
    assert obs.deferred_keys == {(A, 1)}
    assert obs.flush_of_keys == {(A, 1): (A, 2)}


# --------------------------------------------------------------------------- #
# Placing events that name no mule
# --------------------------------------------------------------------------- #

def test_a_merge_with_no_ingest_before_it_is_placed_by_its_own_partial():
    """No ingest rows, and the flush names no mule; at 1015 both A2 and B2
    were in flight. Of its partials only B2's is at the merge's round, so B2's
    upload ran it, and A1 is the deferral it flushed."""
    cluster = [
        _cev(1005.0, "cluster_merge_deferred", mule_id=A, mission_round=1, rule="agg:fedbuff",
             applied=False, partials=[[A, 1]]),
        _cev(1015.0, "cluster_merge", mission_round=2, rule="agg:fedbuff", applied=True,
             partials=[[A, 1], [B, 2]]),
    ]
    obs = _obs(_fedbuff_mules(), cluster)
    assert obs.flush_of_keys == {(A, 1): (B, 2)}
    assert merged_devices(obs, _mission(obs, B, 2)) == ("b2", "a1")


def test_an_event_naming_no_mule_goes_to_the_only_mission_of_its_round():
    """B flew once, so round 2 is A's alone, even for a loss logged just after
    A2's window closed."""
    mules = [
        *_flight(A, 1, 1000.0, ["a1"]),
        *_flight(B, 1, 1002.0, ["b1"]),
        *_flight(A, 2, 1010.0, ["a2"]),
    ]
    obs = _obs(mules, [_cev(1020.5, "backhaul_upload_lost", mission_round=2)])
    assert obs.backhaul_lost_keys == {(A, 2)}


def test_an_event_naming_no_mule_is_placed_by_the_windows_of_its_rounds_missions():
    """Both mules flew round 2, and at 1011 only A2 had taken off (B1's window
    also holds 1011, but B1 is not a round-2 mission)."""
    obs = _obs(_plain_mules(), [_cev(1011.0, "backhaul_upload_lost", mission_round=2)])
    assert obs.backhaul_lost_keys == {(A, 2)}


def test_a_flush_listed_by_its_partials_is_not_flushed_again():
    """B1's merge lists the A1 deferral it flushed. A later merge of A's that
    lists nothing (the form before merges recorded partials) flushes only what
    A deferred since: A2, not A1 a second time."""
    cluster = [
        _up(1005.0, A, 1),
        _cev(1005.0, "cluster_merge_deferred", mule_id=A, mission_round=1, applied=False,
             partials=[[A, 1]]),
        _up(1007.0, B, 1),
        _cev(1007.0, "cluster_merge", mission_round=1, rule="agg:fedbuff", applied=True,
             partials=[[A, 1], [B, 1]]),
        _cev(1007.0, "cluster_round_closed", cluster_round=1),
        _up(1015.0, A, 2),
        _cev(1015.0, "cluster_merge_deferred", mule_id=A, mission_round=2, applied=False),
        _up(1025.0, A, 3),
        _cev(1025.0, "cluster_merge", mule_id=A, mission_round=3, rule="agg:fedbuff",
             applied=True),
        _cev(1025.0, "cluster_round_closed", cluster_round=2),
    ]
    obs = _obs([*_fedbuff_mules(), *_flight(A, 3, 1020.0, ["a1"])], cluster)
    assert obs.flush_of_keys == {(A, 1): (B, 1), (A, 2): (A, 3)}


# --------------------------------------------------------------------------- #
# End to end on a trace directory
# --------------------------------------------------------------------------- #

TRIAL = "N=4-n_mules=2-regime=clean-rrf=60.0__H1__t0__s7"


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _write_two_mule_trial(root: Path, name=TRIAL, mule_rows=None, cluster_rows=None) -> Path:
    """A trace directory as the orchestrator leaves it; the plain trial by default."""
    d = root / name
    d.mkdir(parents=True)
    _write_jsonl(d / "cluster-exp4-cluster.jsonl",
                 _plain_cluster() if cluster_rows is None else cluster_rows)
    rows = _plain_mules() if mule_rows is None else mule_rows
    for mule, members in SLICES.items():
        _write_jsonl(d / f"mule-{mule}.jsonl", [r for r in rows if r["id"] == mule])
        # As the orchestrator writes them: the slice is the mule's expected_devices.
        (d / f"mule-{mule}.json").write_text(json.dumps(
            {"mule_id": mule, "expected_devices": list(members),
             "rf_range_m": 60.0, "n_missions": 3}
        ))
    for dev in DEVICES:
        _write_jsonl(d / f"device-{dev}.jsonl",
                     [{"ts": 998.0, "event": "device_ready", "role": "device", "id": dev}])
    (d / "cluster.json").write_text(json.dumps(
        {"seed_devices": [{"device_id": dev, "position": [0, 0, 0]} for dev in DEVICES]}
    ))
    return d


def test_the_run_dir_supplies_each_mules_slice(tmp_path):
    obs = consume_run_dir(_write_two_mule_trial(tmp_path), n_devices=4)
    assert obs.mule_ids == (A, B)
    assert obs.mule_slices == {A: ("a1", "a2"), B: ("b1", "b2")}


def test_a_two_mule_trace_scores_end_to_end(tmp_path):
    score = score_trial(_write_two_mule_trial(tmp_path), taus=(0.82,))
    row = score.to_row()
    assert row["n_mules"] == 2
    assert row["backhaul_lost_missions"] == 1
    assert row["unmerged_missions"] == 0
    assert row["pass2_coverage"] == pytest.approx(5 / 6)
    assert row["round_close_rate_kmin1"] == pytest.approx(5 / 6)
    assert row["network_aou_mean"] == pytest.approx(2.75 / 6)
    assert (row["missions_to_tau0.82"], row["rounds_to_tau0.82"]) == (2, 3)
    assert row["wall_s_to_tau0.82"] == pytest.approx(18.0)


def test_the_ledger_columns_count_missions_not_rounds(tmp_path):
    """Both mules lose round 2, both expire round 1 (agg:cutoff, quorum 2), and
    under agg:fedbuff three uploads of two rounds are deferred. Counted by
    round each column is 1 or 2, not 2 or 3."""
    cutoff = [
        _up(1005.0, A, 1), _up(1007.0, B, 1),
        _cev(1007.0, "cluster_merge_expired", mule_id=B, mission_round=1, rule="agg:cutoff",
             applied=False, outcome="expired", partials=[[A, 1], [B, 1]]),
        _cev(1015.0, "backhaul_upload_lost", mule_id=A, mission_round=2),
        _cev(1017.0, "backhaul_upload_lost", mule_id=B, mission_round=2),
        _up(1025.0, A, 3), _up(1027.0, B, 3),
        _cev(1027.0, "cluster_merge", mission_round=3, rule="agg:cutoff", applied=True,
             partials=[[A, 3], [B, 3]], expired_partials=[]),
        _cev(1027.0, "cluster_round_closed", cluster_round=1),
    ]
    row = score_trial(_write_two_mule_trial(tmp_path, cluster_rows=cutoff)).to_row()
    assert (row["backhaul_lost_missions"], row["expired_missions"]) == (2, 2)
    assert (row["deferred_missions"], row["unmerged_missions"]) == (0, 0)

    def deferred(t, mule, rnd, buffered):
        return _cev(t, "cluster_merge_deferred", mule_id=mule, mission_round=rnd,
                    rule="agg:fedbuff", applied=False, k=3, partials=buffered)

    fedbuff = [
        _up(1005.0, A, 1), deferred(1005.0, A, 1, [[A, 1]]),
        _up(1007.0, B, 1), deferred(1007.0, B, 1, [[A, 1], [B, 1]]),
        _up(1015.0, A, 2),
        _cev(1015.0, "cluster_merge", mission_round=2, rule="agg:fedbuff", applied=True,
             k=3, partials=[[A, 1], [B, 1], [A, 2]]),
        _cev(1015.0, "cluster_round_closed", cluster_round=1),
        _up(1017.0, B, 2), deferred(1017.0, B, 2, [[B, 2]]),
    ]
    other = "N=4-n_mules=2-regime=clean-rrf=60.0__H1__t1__s8"
    row = score_trial(_write_two_mule_trial(tmp_path, other, cluster_rows=fedbuff)).to_row()
    assert (row["deferred_missions"], row["expired_missions"]) == (3, 0)


def test_the_unmerged_column_counts_the_uploads_the_cluster_never_folded(tmp_path):
    d = _write_two_mule_trial(
        tmp_path, mule_rows=_quorum_mules(), cluster_rows=_quorum_cluster("agg:plain"),
    )
    row = score_trial(d).to_row()
    assert row["unmerged_missions"] == 4
    assert row["round_close_rate_kmin1"] == pytest.approx(2 / 6)
    assert row["merged_total"] == 4


# --------------------------------------------------------------------------- #
# One mule: nothing moves
# --------------------------------------------------------------------------- #

def test_with_one_mule_the_keyed_ledger_is_the_round_keyed_one():
    """Every row maps to the one mule, named or not, so the keys are the rounds."""
    mule = "exp4-mule"
    mules = [
        *_flight(mule, 1, 1000.0, ["a1"]),
        *_flight(mule, 2, 1010.0, ["a2"]),
        *_flight(mule, 3, 1020.0, ["b1"]),
        *_flight(mule, 4, 1030.0, ["b2"]),
        # As every recorded trace writes it: no ``docked``, so no upload, and
        # the expiry of round 4 below stands.
        {"ts": 1040.0, "event": "mission_empty", "role": "mule", "id": mule, "mission_round": 4},
    ]
    cluster = [
        _cev(1005.0, "cluster_merge_deferred", mule_id=mule, mission_round=1, applied=False),
        # No mule and no partials (Phase 1): flushes every deferral so far.
        _cev(1015.0, "cluster_merge", mission_round=2, rule="agg:fedbuff", applied=True),
        # No mule and no round (before Amendment 5): placed by time in mission 3.
        _cev(1025.0, "backhaul_upload_lost"),
        _cev(1035.0, "cluster_merge_expired", mission_round=4, partials=[["other", 4]]),
    ]
    obs = _obs(mules, cluster, slices={mule: ()})
    assert obs.mule_ids == (mule,) and obs.mule_slices == {}
    assert obs.backhaul_lost_rounds == {3} and obs.backhaul_lost_keys == {(mule, 3)}
    assert obs.deferred_rounds == {1} and obs.deferred_keys == {(mule, 1)}
    assert obs.flush_of == {1: 2} and obs.flush_of_keys == {(mule, 1): (mule, 2)}
    assert obs.expired_rounds == {4} and obs.expired_keys == {(mule, 4)}
    assert obs.unmerged_keys == set()
    assert [obs.mission_key(m) for m in obs.missions] == [(mule, r) for r in (1, 2, 3, 4)]
    # The slice is every device, whatever the config says.
    assert _summary(obs).pass2_coverage == pytest.approx(0.5)


def test_with_one_mule_an_unanswered_ingest_is_credited_as_before():
    """A trial cut off between an ingest and its close: with one mule the
    ledger does not replay the open round, so the mission counts as it always
    did (with several, it would be unmerged)."""
    mule = "exp4-mule"
    obs = _obs(
        [*_flight(mule, 1, 1000.0, ["a1"]), *_flight(mule, 2, 1010.0, ["a2"])],
        [_up(1005.0, mule, 1), _cev(1005.0, "cluster_round_closed", cluster_round=1),
         _up(1015.0, mule, 2), _up(1016.0, mule, 2)],
        slices=None,
    )
    assert obs.mule_ids == (mule,) and obs.unmerged_keys == set()
    assert _summary(obs).round_close_rate_kmin1 == pytest.approx(1.0)


def _record(rnd, device):
    return MissionRecord(
        mission_round=rnd, pass_1_contacts=1, pass_2_contacts=1, pass_1_updates=1,
        pass_1_scheduled=1, pass_1_clean_devices=(device,), delivered=1, undelivered=0,
        duration_s=1.0, mule_id="exp4-mule",
    )


def test_an_observation_built_from_the_round_keyed_fields_scores_as_before():
    """Before the keyed ledger, callers built observations from the round-keyed
    sets; with one mule they are lifted onto it. Round 1 expired, 2 was lost,
    3 was deferred and flushed by 4, which alone closes, with both updates."""
    missions = [_record(r, f"d{r}") for r in (1, 2, 3, 4)]
    obs = Exp4Observation(
        n_devices=4, cluster_rounds_closed=1, up_bundles_ingested=3, missions=missions,
        backhaul_lost_rounds={2}, deferred_rounds={3}, expired_rounds={1}, flush_of={3: 4},
    )
    assert obs.backhaul_lost_keys == {(None, 2)} and obs.flush_of_keys == {(None, 3): (None, 4)}
    assert [merged_devices(obs, m) for m in missions] == [(), (), (), ("d4", "d3")]
    s = summarise_observation(obs, n_devices=4, rf_range_m=60.0, n_missions_target=4)
    assert (s.round_close_rate_kmin1, s.round_close_rate_kmin2) == (0.25, 0.25)


def test_an_observation_built_from_the_keyed_fields_projects_them_onto_rounds():
    obs = Exp4Observation(
        n_devices=4, cluster_rounds_closed=0, up_bundles_ingested=0, mule_ids=(A, B),
        backhaul_lost_keys={(A, 2)}, deferred_keys={(A, 1)}, expired_keys={(B, 3)},
        flush_of_keys={(A, 1): (B, 1)},
    )
    assert (obs.backhaul_lost_rounds, obs.deferred_rounds, obs.expired_rounds) == ({2}, {1}, {3})
    assert obs.flush_of == {1: 1}


@pytest.mark.parametrize("field_name, value", [
    ("backhaul_lost_rounds", {2}), ("deferred_rounds", {2}),
    ("expired_rounds", {2}), ("flush_of", {1: 2}),
])
def test_with_several_mules_a_round_keyed_field_alone_is_refused(field_name, value):
    """Round 2 names a mission of each mule, so it cannot be lifted."""
    with pytest.raises(ValueError, match=field_name):
        Exp4Observation(n_devices=4, cluster_rounds_closed=0, up_bundles_ingested=0,
                        mule_ids=(A, B), **{field_name: value})


REPO = Path(__file__).resolve().parents[2]
RECORDED = REPO / "results" / "exp4_matrix" / "C_traces"
RECORDED_TAUS = (0.82, 0.75, 0.85, 0.9)

#: Every column (bar ``trace_root``, a machine path) of three recorded trials,
#: as the scorer produced them before multi-mule keys. H2 t18 lost two backhaul
#: uploads and H3 t7 one, and both reached τ = 0.82 in their fourth mission;
#: H2 t2 lost three and never reached it.
RECORDED_ROWS = {
    'N=6-dead_zone=0.6-link_quality=0.4-n_missions=4-regime=jittery-rrf=60.0__H2__t18__s1687592285': {
        'cell_id': 'N=6-dead_zone=0.6-link_quality=0.4-n_missions=4-regime=jittery-rrf=60.0',
        'arm': 'H2', 'trial_index': 18, 'seed': 1687592285, 'mission_budget_s': 120.0,
        'mission_window_adaptation': 0, 'aggregation': 'agg:plain', 'aggregation_params': '',
        'fedprox_rho': 0.0, 'pass_2_budget': 0, 'deadline_law': 'additive', 'deadline_params': '',
        'miss_priority': 0, 'status': 'ok', 'update_yield': 1.75, 'round_close_rate_kmin1': 0.5,
        'round_close_rate_kmin2': 0.25, 'round_close_rate_kminhalf': 0.25,
        'round_close_rate_kminN': 0.0, 'coverage': 1.0, 'jains_fairness': 0.9704301075268817,
        'participation_entropy': 2.560671912866221, 'mission_completion_rate': 1.0,
        'completion_fairness': 0.9074074074074074, 'pass2_coverage': 1.0,
        'rho_contact': 1.6666666666666667, 'rounds_closed': 2, 'missions_completed': 4,
        'mission_failures': 0, 'pass1_contacts_mean': 3.0, 'pass2_contacts_mean': 4.0,
        'mission_duration_s_mean': 7.769028604030609, 'n_devices': 6, 'rf_range_m': 60.0,
        'n_missions_target': 4, 'init_auc': 0.27618662499999996, 'init_accuracy': 0.3555,
        'init_loss': 0.695851743221283, 'final_auc': 0.875157,
        'final_accuracy': 0.8476666666666667, 'final_loss': 0.6707206964492798,
        'best_auc': 0.875157, 'delta_auc': 0.5989703749999999, 'rounds_evaluated': 3,
        't_at_tau_round': 2, 'tau': 0.82, 'missions_empty': 0, 'backhaul_lost_missions': 2,
        'deferred_missions': 0, 'expired_missions': 0, 'network_aou_mean': 1.6666666666666665,
        'network_aou_final': 1.1666666666666665, 'age_max': 4, 'age_p95': 3.0, 'merged_total': 5,
        'jain_merged': 0.8333333333333334, 'deadline_admitted': 0, 'deadline_missed': 0,
        'deadline_miss_rate': '', 'deadline_basis': '', 'reached_tau0.82': 1,
        'missions_to_tau0.82': 4, 'rounds_to_tau0.82': 2, 'wall_s_to_tau0.82': 31.06959056854248,
        'reached_tau0.75': 1, 'missions_to_tau0.75': 4, 'rounds_to_tau0.75': 2,
        'wall_s_to_tau0.75': 31.06959056854248, 'reached_tau0.85': 0, 'missions_to_tau0.85': '',
        'rounds_to_tau0.85': '', 'wall_s_to_tau0.85': '', 'reached_tau0.9': 0,
        'missions_to_tau0.9': '', 'rounds_to_tau0.9': '', 'wall_s_to_tau0.9': '',
    },
    'N=6-dead_zone=0.6-link_quality=0.4-n_missions=4-regime=jittery-rrf=60.0__H3__t7__s1893876901': {
        'cell_id': 'N=6-dead_zone=0.6-link_quality=0.4-n_missions=4-regime=jittery-rrf=60.0',
        'arm': 'H3', 'trial_index': 7, 'seed': 1893876901, 'mission_budget_s': 120.0,
        'mission_window_adaptation': 0, 'aggregation': 'agg:plain', 'aggregation_params': '',
        'fedprox_rho': 0.0, 'pass_2_budget': 0, 'deadline_law': 'additive', 'deadline_params': '',
        'miss_priority': 0, 'status': 'ok', 'update_yield': 2.0, 'round_close_rate_kmin1': 0.75,
        'round_close_rate_kmin2': 0.5, 'round_close_rate_kminhalf': 0.25,
        'round_close_rate_kminN': 0.0, 'coverage': 1.0, 'jains_fairness': 1.0,
        'participation_entropy': 2.584962500721156, 'mission_completion_rate': 0.8333333333333334,
        'completion_fairness': 0.7619047619047619, 'pass2_coverage': 1.0, 'rho_contact': 2.4,
        'rounds_closed': 3, 'missions_completed': 4, 'mission_failures': 0,
        'pass1_contacts_mean': 2.5, 'pass2_contacts_mean': 2.5,
        'mission_duration_s_mean': 7.788571000099182, 'n_devices': 6, 'rf_range_m': 60.0,
        'n_missions_target': 4, 'init_auc': 0.2574081875, 'init_accuracy': 0.3581666666666667,
        'init_loss': 0.6959713697433472, 'final_auc': 0.921106875, 'final_accuracy': 0.8265,
        'final_loss': 0.640066385269165, 'best_auc': 0.921106875, 'delta_auc': 0.6636986874999999,
        'rounds_evaluated': 4, 't_at_tau_round': 3, 'tau': 0.82, 'missions_empty': 0,
        'backhaul_lost_missions': 1, 'deferred_missions': 0, 'expired_missions': 0,
        'network_aou_mean': 1.375, 'network_aou_final': 1.1666666666666665, 'age_max': 4,
        'age_p95': 3.0, 'merged_total': 6, 'jain_merged': 0.75, 'deadline_admitted': 0,
        'deadline_missed': 0, 'deadline_miss_rate': '', 'deadline_basis': '', 'reached_tau0.82': 1,
        'missions_to_tau0.82': 4, 'rounds_to_tau0.82': 3, 'wall_s_to_tau0.82': 31.14928412437439,
        'reached_tau0.75': 1, 'missions_to_tau0.75': 4, 'rounds_to_tau0.75': 3,
        'wall_s_to_tau0.75': 31.14928412437439, 'reached_tau0.85': 0, 'missions_to_tau0.85': '',
        'rounds_to_tau0.85': '', 'wall_s_to_tau0.85': '', 'reached_tau0.9': 0,
        'missions_to_tau0.9': '', 'rounds_to_tau0.9': '', 'wall_s_to_tau0.9': '',
    },
    'N=6-dead_zone=0.6-link_quality=0.4-n_missions=4-regime=jittery-rrf=60.0__H2__t2__s970667198': {
        'cell_id': 'N=6-dead_zone=0.6-link_quality=0.4-n_missions=4-regime=jittery-rrf=60.0',
        'arm': 'H2', 'trial_index': 2, 'seed': 970667198, 'mission_budget_s': 120.0,
        'mission_window_adaptation': 0, 'aggregation': 'agg:plain', 'aggregation_params': '',
        'fedprox_rho': 0.0, 'pass_2_budget': 0, 'deadline_law': 'additive', 'deadline_params': '',
        'miss_priority': 0, 'status': 'ok', 'update_yield': 3.25, 'round_close_rate_kmin1': 0.25,
        'round_close_rate_kmin2': 0.25, 'round_close_rate_kminhalf': 0.25,
        'round_close_rate_kminN': 0.0, 'coverage': 1.0, 'jains_fairness': 1.0,
        'participation_entropy': 2.584962500721156, 'mission_completion_rate': 0.8333333333333334,
        'completion_fairness': 0.7612612612612613, 'pass2_coverage': 1.0, 'rho_contact': 4.8,
        'rounds_closed': 1, 'missions_completed': 4, 'mission_failures': 0,
        'pass1_contacts_mean': 1.25, 'pass2_contacts_mean': 1.75,
        'mission_duration_s_mean': 3.886547327041626, 'n_devices': 6, 'rf_range_m': 60.0,
        'n_missions_target': 4, 'init_auc': 0.27346493749999995,
        'init_accuracy': 0.3566666666666667, 'init_loss': 0.6958889961242676,
        'final_auc': 0.6952256875, 'final_accuracy': 0.7066666666666667,
        'final_loss': 0.6832300424575806, 'best_auc': 0.6952256875,
        'delta_auc': 0.4217607500000001, 'rounds_evaluated': 2, 't_at_tau_round': '', 'tau': 0.82,
        'missions_empty': 0, 'backhaul_lost_missions': 3, 'deferred_missions': 0,
        'expired_missions': 0, 'network_aou_mean': 1.8333333333333333,
        'network_aou_final': 1.3333333333333333, 'age_max': 4, 'age_p95': 3.849999999999998,
        'merged_total': 4, 'jain_merged': 0.6666666666666666, 'deadline_admitted': 0,
        'deadline_missed': 0, 'deadline_miss_rate': '', 'deadline_basis': '', 'reached_tau0.82': 0,
        'missions_to_tau0.82': '', 'rounds_to_tau0.82': '', 'wall_s_to_tau0.82': '',
        'reached_tau0.75': 0, 'missions_to_tau0.75': '', 'rounds_to_tau0.75': '',
        'wall_s_to_tau0.75': '', 'reached_tau0.85': 0, 'missions_to_tau0.85': '',
        'rounds_to_tau0.85': '', 'wall_s_to_tau0.85': '', 'reached_tau0.9': 0,
        'missions_to_tau0.9': '', 'rounds_to_tau0.9': '', 'wall_s_to_tau0.9': '',
    },
}


@pytest.mark.skipif(not RECORDED.is_dir(), reason="recorded traces not present")
@pytest.mark.parametrize("trial", sorted(RECORDED_ROWS))
def test_the_recorded_cell_scores_as_it_did_with_one_mule(trial):
    score = score_trial(RECORDED / trial, taus=RECORDED_TAUS)
    row = score.to_row()
    want = RECORDED_ROWS[trial]
    assert score.n_mules == 1
    # Columns added since: the Phase 2 fleet provenance (one mule, the rest
    # blank at their recorded values) and the count of uploads the cluster
    # never folded, which one mule never has.
    added = set(row) - set(want) - {"trace_root"}
    assert "n_mules" in added and added <= set(PROVENANCE_COLUMNS) | {"unmerged_missions"}
    assert row["unmerged_missions"] == 0
    assert {c: row[c] for c in added - {"unmerged_missions"}} == {
        c: 1 if c == "n_mules" else "" for c in added - {"unmerged_missions"}
    }
    for col, value in want.items():
        if isinstance(value, float):
            assert (col, row[col]) == (col, pytest.approx(value, rel=1e-12))
        else:
            assert (col, row[col]) == (col, value)
