"""DDIL plan, step 0: re-price P1's kept traces under other contact physics (read-only).

The unified plan (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md, "Step 0, extended") asks how
much of F's extra hover at 17 dB is the recorded model's fixed-rate charge, what step 1's
session model does to it, and what a per-transfer rule could save on top. This script
answers from the traces P1 kept, without flying anything:

* It rebuilds each trial's contact link and channel with the simulator's own classes
  (``ContactLink``, ``ContactChannel.from_link``) from the mule's recorded settings and
  channel salt, and checks that the rebuilt channel reproduces every recorded arrival SNR,
  every recorded uplink drop, and, under the recorded pricing (a), every stop's dwell.
* It then re-walks every stop's sessions in contact order, each starting when the one
  before ends, under each pricing below. Stops keep their recorded arrival times, routes,
  classes and gates: only the time inside a stop moves (a longer stop does not delay the
  next one). Missions keep their recorded starts.

Pricings (the plan's table):

  a     as recorded: the rate at the SNR at the session's start, the floor rate below it
  b     the rate follows the channel in 1 s steps, nothing delivered below the floor
  b0    b with the interference clipped at 0 from above (no gain from the sinusoid's crest)
  c     b, on the fastest class that reaches the device from the stop at the session start
  d     ``fastest_defer`` (guard on, theta = 2, D_max = 1) on pricing a
  s     step 1's session model: b, but a session fails once the SNR has stayed below the
        floor for ``t_out`` s in a row, costs the airtime it used, and is tried once more
        at the end of the queue (not resumable unless said)
  e     ``fastest_defer`` on pricing s
  p     s, plus one re-poll at the stop's end of each member that was in range but below
        the floor at arrival (Pass 1)
  o     hindsight on s: ``o_lb`` is a lower bound on any policy's airtime (each session at
        its best class and start time, alone); ``o_gr`` an achievable foresight greedy
        (always serve the session, class and wait, up to 30 s, that ends soonest)

Each pricing runs on the per-class interference model as recorded and, with suffix ``_sh``,
on shared interference: every class takes the stop class's own interference term, so the
stop class's SNR is unchanged and the other classes trough with it (nested bands, the
plan's primary model since 16 Nov 2026).

Usage::

    py -3.11 scripts/exp5/ddil_replay.py --out results/exp5/scores/ddil
"""

from __future__ import annotations

import argparse
import csv
import functools
import glob
import json
import math
import os
import random
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hermes.l1.channel_model import (  # noqa: E402
    SALT_AVAILABILITY, ContactChannel, ferry_salt, keyed_uniform,
)
from hermes.l1.contact_link import ContactLink  # noqa: E402
from hermes.mission.contact_plan import planar_distance_m  # noqa: E402

STEP_S = 1.0          # NOISE_BIN_S: the walk's step
WALK_CAP_S = 600.0    # a walk that has not delivered by then is cut and counted
HINDSIGHT_WAIT_S = 30  # the foresight policies' longest wait
HINDSIGHT_GRID_S = 0.5  # and the grid their waits are on
FLOORS = ("narrow",)  # the guard: narrow is a candidate only at a narrow stop

MODES = ("per_class", "shared")


# --------------------------------------------------------------------------- the trial
@dataclass
class Session:
    did: str
    nbytes: int          # what the session is charged for (push, plus the update if any)
    plan_bytes: int      # what the mule expects to move (the planner's session bytes)
    update: bool         # a Pass-1 session that carries an update when it completes
    d: float             # planar distance to the stop
    tries: int = 0
    defers: int = 0
    delivered_bits: float = 0.0


@dataclass
class Stop:
    mission: int
    pass_kind: int       # 1 or 2
    arrival: float
    end: float
    band: str
    position: Tuple[float, float, float]
    sessions: List[Session]
    gated_in_range: List[Tuple[str, float]]   # Pass 1: in range, below the floor at arrival
    rec_dwell: float


@dataclass(eq=False)  # hashed by identity, so the SNR cache can key on it
class Trial:
    path: str
    seed: int
    salt: int
    link: ContactLink
    chan: ContactChannel
    floor: float
    payload: int
    availability: Dict[str, float]
    stops: List[Stop]
    missions: List[int]
    rec_dwell: Dict[int, float]                 # mission -> ledger dwell
    rec_updates: Dict[int, int]                 # mission -> Pass-1 outcomes not timeout
    checks: Dict[str, float] = field(default_factory=dict)

    # ---- the channel ---------------------------------------------------------------
    @functools.lru_cache(maxsize=1 << 21)
    def snr(self, mode: str, did: str, c: str, d: float, stop: Tuple[float, ...],
            stop_band: str, t: float) -> float:
        chan = self.chan
        if mode == "per_class":
            return chan.snr_db(t, c, d, link_key=did, stop_pos=stop)
        if mode == "shared":
            if c == stop_band:
                return chan.snr_db(t, c, d, link_key=did, stop_pos=stop)
            return (chan.pred_snr_db(c, d) + chan.shadow_db(t, did, stop_pos=stop)
                    + chan.interference_db(t, stop_band))
        if mode == "clip":
            return (chan.pred_snr_db(c, d) + chan.shadow_db(t, did, stop_pos=stop)
                    + min(0.0, chan.interference_db(t, c)))
        raise ValueError(mode)

    def dropped(self, did: str, mission_round: int) -> bool:
        rel = self.availability.get(did)
        if rel is None:
            return False
        salt = ferry_salt(self.seed, SALT_AVAILABILITY)
        return keyed_uniform(salt, did, int(mission_round)) >= rel


def _events(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_trial(tdir: str) -> Trial:
    with open(os.path.join(tdir, "mule-exp4-mule.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    evs = _events(os.path.join(tdir, "mule-exp4-mule.jsonl"))
    ready = next(e for e in evs if e.get("event") == "mule_ready")
    cp = ready["channel_params"]["contact"]
    pos: Dict[str, Tuple[float, ...]] = {}
    for f in glob.glob(os.path.join(tdir, "device-*.json")):
        with open(f, encoding="utf-8") as fh:
            dj = json.load(fh)
        pos[dj["device_id"]] = tuple(dj["position"])

    link = ContactLink(
        anchor_planar_m=float(cfg["rf_range_m"]), snr_floor_db=float(cfg["snr_floor_db"]),
        altitude_m=float(cfg["altitude_m"]), n_pl=float(cfg["n_pl"]),
        shadow_sigma_db=float(cfg["shadow_sigma_db"]),
        margin_quantile=float(cfg["margin_quantile"]),
        narrow_range_ratio=cfg.get("narrow_range_ratio"),
    )
    chan = ContactChannel.from_link(
        link, salt=int(cp["salt"]), regime=cp["regime"],
        interference_period_s=float(cp["interference_period_s"]),
        noise_bin_s=float(cp["noise_bin_s"]), shadow_corr_s=float(cp["shadow_corr_s"]),
        shadow_keying=cp["shadow_keying"],
        interference_amp_db=float(cp["interference_amp_db"]),
        interference_sigma_db=float(cp["interference_sigma_db"]),
    )
    payload = int(cfg["payload_bytes"])
    trial = Trial(path=tdir, seed=int(cfg["trial_seed"]), salt=int(cp["salt"]), link=link,
                  chan=chan, floor=float(cfg["snr_floor_db"]), payload=payload,
                  availability=dict(cfg.get("device_availability") or {}), stops=[],
                  missions=[], rec_dwell={}, rec_updates={})
    desc = chan.describe()
    trial.checks["channel_params_match"] = float(all(
        desc[k] == cp[k] for k in cp if k in desc))

    snr_err = 0.0
    drop_mismatch = 0
    for m in (e for e in evs if e.get("event") == "mission_completed"):
        mr = int(m["mission_round"])
        trial.missions.append(mr)
        trial.rec_dwell[mr] = float((m.get("sim_ledger") or {}).get("dwell", 0.0))
        trial.rec_updates[mr] = sum(1 for o in (m.get("pass_1_outcomes") or [])
                                    if o.get("outcome") != "timeout")
        for pk, key in ((1, "pass_1_flown"), (2, "pass_2_flown")):
            for s in m.get(key) or []:
                band = s["band"]
                stop = tuple(float(x) for x in s["position"])
                targets = set(s.get("targets") or [])
                dropped = set(s.get("uplink_dropped") or [])
                missing = set(s.get("missing") or [])
                rng = link.range_planar_m(band)
                sessions: List[Session] = []
                gated: List[Tuple[str, float]] = []
                for did in s["devices"]:
                    d = planar_distance_m(stop, pos[did])
                    rec = (s.get("snr_db") or {}).get(did)
                    if rec is not None:
                        got = chan.snr_db(float(s["arrival_s"]), band, d, link_key=did,
                                          stop_pos=stop)
                        snr_err = max(snr_err, abs(got - float(rec)))
                    if did not in targets:
                        if pk == 1 and d <= rng:
                            gated.append((did, d))
                        continue
                    if pk == 1:
                        if trial.availability and (trial.dropped(did, mr) != (did in dropped)):
                            drop_mismatch += 1
                        if did in dropped or did in missing:
                            sessions.append(Session(did, payload, 2 * payload, False, d))
                        else:
                            sessions.append(Session(did, 2 * payload, 2 * payload, True, d))
                    else:
                        sessions.append(Session(did, payload, payload, False, d))
                trial.stops.append(Stop(mr, pk, float(s["arrival_s"]), float(s["end_s"]),
                                        band, stop, sessions, gated, float(s["dwell_s"])))
    trial.checks["arrival_snr_max_err_db"] = snr_err
    trial.checks["uplink_drop_mismatches"] = float(drop_mismatch)
    return trial


# --------------------------------------------------------------------------- the walks
@dataclass
class Walk:
    dur: float
    done: bool
    bits: float          # delivered by the end
    capped: bool = False


def walk_fixed(tr: Trial, mode: str, st: Stop, s: Session, c: str, t0: float) -> Walk:
    """Pricing a: the whole session at the rate the SNR at its start gives (floor if below)."""
    snr = tr.snr(mode, s.did, c, s.d, st.position, st.band, t0)
    dw = tr.link.dwell_s(s.nbytes, c, snr)
    if dw is None:
        dw = tr.link.dwell_s(s.nbytes, c, tr.floor)
    return Walk(float(dw), True, 8.0 * s.nbytes)


def walk_follow(tr: Trial, mode: str, st: Stop, s: Session, c: str, t0: float,
                t_out: Optional[float], start_bits: float = 0.0) -> Walk:
    """Pricings b and s: the rate follows the SNR in 1 s steps; nothing moves below the
    floor. With ``t_out``, the session fails once the SNR has been below the floor for
    ``t_out`` s in a row (read at each step's start), and costs the time it used."""
    need = 8.0 * s.nbytes - start_bits
    got = 0.0
    t = t0
    below_since: Optional[float] = None
    while True:
        if t - t0 >= WALK_CAP_S:
            return Walk(t - t0, False, start_bits + got, capped=True)
        rate = tr.link.rate_bps(c, tr.snr(mode, s.did, c, s.d, st.position, st.band, t))
        if rate <= 0.0:
            if below_since is None:
                below_since = t
            if t_out is not None and t + STEP_S - below_since >= t_out:
                t_fail = below_since + t_out
                return Walk(t_fail - t0, False, start_bits + got)
            t += STEP_S
            continue
        below_since = None
        if got + rate * STEP_S >= need:
            return Walk(t - t0 + (need - got) / rate, True, start_bits + need)
        got += rate * STEP_S
        t += STEP_S


# --------------------------------------------------------------------------- one stop
@dataclass
class StopResult:
    hover: float = 0.0
    updates: int = 0
    sessions: int = 0
    started_below: int = 0
    failed: int = 0
    retry_failed: int = 0
    lost_updates: int = 0
    defers: int = 0
    switched: int = 0
    wide_to_medium: int = 0
    forced: int = 0
    capped: int = 0
    repolled: int = 0
    repoll_updates: int = 0


@dataclass(frozen=True)
class Pricing:
    name: str
    physics: str                 # "fixed" | "follow"
    t_out: Optional[float] = None
    resumable: bool = False
    mode: str = "per_class"      # "per_class" | "shared" | "clip"
    switch: bool = False         # pricing c: the fastest class, no guard, no deferral
    rule: bool = False           # fastest_defer
    theta: float = 2.0
    d_max: int = 1
    guard: str = "narrow"        # "narrow" | "off"
    repoll: bool = False


def _classes_in_reach(tr: Trial, st: Stop, s: Session) -> List[str]:
    return [c for c in tr.link.names if s.d <= tr.link.range_planar_m(c)]


def _choose(tr: Trial, pr: Pricing, st: Stop, s: Session, t: float,
            queue_left: int) -> Tuple[Optional[str], bool, bool]:
    """(class, defer?, forced?) for the session about to start at ``t``. ``forced``: the
    best price missed the bar but no deferral was left (B8's ``tx_forced``)."""
    link = tr.link
    if not (pr.switch or pr.rule):
        return st.band, False, False
    only_stop = pr.name.startswith("e_dfr")   # FX-txd: the stop's class is the only candidate
    cands = []
    for c in _classes_in_reach(tr, st, s):
        if only_stop and c != st.band:
            continue
        if pr.rule and pr.guard == "narrow" and c in FLOORS and st.band not in FLOORS:
            continue
        snr = tr.snr(pr.mode, s.did, c, s.d, st.position, st.band, t)
        if snr < tr.floor:
            continue
        price = link.dwell_s(s.plan_bytes, c, snr)
        if price is None:
            continue
        cands.append((price, 0 if c == st.band else 1, link.index(c), c))
    can_defer = pr.rule and s.defers < pr.d_max and queue_left > 0
    if not cands:
        if can_defer:
            return None, True, False
        return st.band, False, pr.rule
    cands.sort()
    price, _, _, best = cands[0]
    if pr.switch:
        return best, False, False
    r_j = link.dwell_s(s.plan_bytes, st.band, link.mean_snr_db(st.band, s.d))
    if r_j is not None and price <= pr.theta * r_j:
        return best, False, False
    if can_defer:
        return None, True, False
    return best, False, True


def run_stop(tr: Trial, pr: Pricing, st: Stop) -> StopResult:
    res = StopResult()
    queue = deque(Session(s.did, s.nbytes, s.plan_bytes, s.update, s.d) for s in st.sessions)
    t = st.arrival
    while queue:
        s = queue.popleft()
        c, defer, forced = _choose(tr, pr, st, s, t, len(queue))
        if defer:
            s.defers += 1
            res.defers += 1
            queue.append(s)
            continue
        res.forced += int(forced)
        if c != st.band:
            res.switched += 1
            if st.band == "wide" and c == "medium":
                res.wide_to_medium += 1
        snr0 = tr.snr(pr.mode, s.did, c, s.d, st.position, st.band, t)
        if snr0 < tr.floor:
            res.started_below += 1
        res.sessions += 1
        if pr.physics == "fixed":
            w = walk_fixed(tr, pr.mode, st, s, c, t)
        else:
            w = walk_follow(tr, pr.mode, st, s, c, t, pr.t_out,
                            s.delivered_bits if pr.resumable else 0.0)
        t += w.dur
        res.hover += w.dur
        res.capped += int(w.capped)
        if w.done:
            res.updates += int(s.update)
            continue
        res.failed += 1
        if s.tries == 0:
            s.tries = 1
            s.delivered_bits = w.bits if pr.resumable else 0.0
            queue.append(s)
        else:
            res.retry_failed += 1
            res.lost_updates += int(s.update)
    if pr.repoll and st.pass_kind == 1:
        for did, d in st.gated_in_range:
            if tr.snr(pr.mode, did, st.band, d, st.position, st.band, t) < tr.floor:
                continue
            res.repolled += 1
            upd = not tr.dropped(did, st.mission)
            nbytes = 2 * tr.payload if upd else tr.payload
            s = Session(did, nbytes, 2 * tr.payload, upd, d)
            w = walk_follow(tr, pr.mode, st, s, st.band, t, pr.t_out)
            t += w.dur
            res.hover += w.dur
            res.sessions += 1
            if w.done and upd:
                res.updates += 1
                res.repoll_updates += 1
    return res


# --------------------------------------------------------------------------- hindsight
def _best_walk(tr: Trial, mode: str, st: Stop, s: Session, t_from: float, horizon: float,
               t_out: float) -> Optional[Tuple[float, float, str]]:
    """The session's shortest completing walk, over classes in reach and start times on a
    1 s grid from ``t_from`` to ``t_from + horizon``: (duration, start, class)."""
    best = None
    k = 0
    while k <= horizon:
        t0 = t_from + k
        for c in _classes_in_reach(tr, st, s):
            if tr.snr(mode, s.did, c, s.d, st.position, st.band, t0) < tr.floor:
                continue
            w = walk_follow(tr, mode, st, s, c, t0, t_out)
            if w.done and (best is None or w.dur < best[0]):
                best = (w.dur, t0, c)
        k += 1
    return best


def _earliest_end(tr: Trial, mode: str, st: Stop, s: Session, t_from: float,
                  t_out: float) -> Optional[Tuple[float, str]]:
    """The earliest a completing session of ``s`` can end, starting at ``t_from`` or up to
    HINDSIGHT_WAIT_S later on a 1 s grid, on any class in reach: (end, class)."""
    best: Optional[Tuple[float, str]] = None
    for k in range(int(round(HINDSIGHT_WAIT_S / HINDSIGHT_GRID_S)) + 1):
        t0 = t_from + k * HINDSIGHT_GRID_S
        if best is not None and t0 >= best[0]:
            break
        for c in _classes_in_reach(tr, st, s):
            if tr.snr(mode, s.did, c, s.d, st.position, st.band, t0) < tr.floor:
                continue
            w = walk_follow(tr, mode, st, s, c, t0, t_out)
            if w.done and (best is None or t0 + w.dur < best[0]):
                best = (t0 + w.dur, c)
    return best


def foresight_opt(tr: Trial, mode: str, st: Stop, t_out: float) -> StopResult:
    """The best order, class and wait (up to HINDSIGHT_WAIT_S) for the stop's sessions with
    the channel known in advance. With waiting allowed, ending a set of sessions earlier
    never hurts what follows, so the earliest end of each subset is a sufficient state: a
    DP over the subsets is exact (on the 1 s wait grid)."""
    n = len(st.sessions)
    res = StopResult()
    if n == 0:
        return res
    f: Dict[int, float] = {0: st.arrival}
    for mask in sorted(range(1 << n), key=lambda m: bin(m).count("1")):
        if mask not in f:
            continue
        T = f[mask]
        for j in range(n):
            if mask >> j & 1:
                continue
            e = _earliest_end(tr, mode, st, st.sessions[j], T, t_out)
            if e is None:
                continue
            new = mask | (1 << j)
            if e[0] < f.get(new, math.inf):
                f[new] = e[0]
    # the most updates served, then the least hover
    def key(m: int) -> Tuple[int, int, float]:
        ups = sum(1 for j in range(n) if m >> j & 1 and st.sessions[j].update)
        return (-ups, -bin(m).count("1"), f[m])
    best = min(f, key=key)
    res.hover = f[best] - st.arrival
    res.sessions = bin(best).count("1")
    res.updates = sum(1 for j in range(n) if best >> j & 1 and st.sessions[j].update)
    res.lost_updates = sum(1 for j in range(n) if not best >> j & 1 and st.sessions[j].update)
    return res


def hindsight(tr: Trial, mode: str, st: Stop, t_out: float) -> Tuple[StopResult, StopResult]:
    lb = StopResult()
    horizon = max(0.0, st.end - st.arrival) + 60.0
    for s in st.sessions:
        b = _best_walk(tr, mode, st, s, st.arrival, horizon, t_out)
        if b is None:
            lb.lost_updates += int(s.update)
            continue
        lb.hover += b[0]
        lb.sessions += 1
        lb.updates += int(s.update)
    gr = StopResult()
    left = list(st.sessions)
    t = st.arrival
    while left:
        pick = None
        for i, s in enumerate(left):
            for k in range(int(round(HINDSIGHT_WAIT_S / HINDSIGHT_GRID_S)) + 1):
                t0 = t + k * HINDSIGHT_GRID_S
                for c in _classes_in_reach(tr, st, s):
                    if tr.snr(mode, s.did, c, s.d, st.position, st.band, t0) < tr.floor:
                        continue
                    w = walk_follow(tr, mode, st, s, c, t0, t_out)
                    if not w.done:
                        continue
                    key = (t0 + w.dur, w.dur)
                    if pick is None or key < pick[0]:
                        pick = (key, i, c)
                if pick is not None and pick[0][0] <= t0:
                    break
        if pick is None:
            for s in left:
                gr.lost_updates += int(s.update)
            break
        (end, _), i, c = pick
        s = left.pop(i)
        gr.hover += end - t
        gr.sessions += 1
        gr.updates += int(s.update)
        gr.switched += int(c != st.band)
        t = end
    return lb, gr


# --------------------------------------------------------------------------- below-floor runs
def below_floor_runs(tr: Trial, st: Stop) -> List[int]:
    """Lengths (s, 1 s grid) of the stop class's below-floor runs for the stop's Pass-1
    targets, over the stop and 60 s either side, so runs at the window's edge are whole."""
    runs: List[int] = []
    if st.pass_kind != 1:
        return runs
    t_lo, t_hi = st.arrival - 60.0, st.end + 60.0
    for s in st.sessions:
        n = 0
        inside = False
        t = t_lo
        while t <= t_hi:
            below = tr.snr("per_class", s.did, st.band, s.d, st.position, st.band, t) < tr.floor
            if below:
                n += 1
                inside = inside or (st.arrival <= t <= st.end)
            else:
                if n and inside:
                    runs.append(n)
                n, inside = 0, False
            t += STEP_S
        if n and inside:
            runs.append(n)
    return runs


# --------------------------------------------------------------------------- the grid
def pricings_for(arm: str) -> List[Pricing]:
    base = [
        Pricing("a", "fixed"),
        Pricing("s", "follow", t_out=10.0),
        Pricing("s_sh", "follow", t_out=10.0, mode="shared"),
        Pricing("p", "follow", t_out=10.0, repoll=True),
        Pricing("p_sh", "follow", t_out=10.0, mode="shared", repoll=True),
    ]
    if arm not in ("F", "FX"):
        return base
    out = base + [
        Pricing("b", "follow"),
        Pricing("b0", "follow", mode="clip"),
        Pricing("b_sh", "follow", mode="shared"),
        Pricing("c", "follow", switch=True),
        Pricing("c_sh", "follow", switch=True, mode="shared"),
        Pricing("d", "fixed", rule=True),
        Pricing("s2", "follow", t_out=2.0),
        Pricing("s_res", "follow", t_out=10.0, resumable=True),
        Pricing("e", "follow", t_out=10.0, rule=True),
        Pricing("e_sh", "follow", t_out=10.0, rule=True, mode="shared"),
        Pricing("e_ng", "follow", t_out=10.0, rule=True, guard="off"),
        Pricing("e_ng_sh", "follow", t_out=10.0, rule=True, guard="off", mode="shared"),
    ]
    for th in (1.5, 2.0, 3.0):
        for dm in (1, 2):
            for mode, sfx in (("per_class", ""), ("shared", "_sh")):
                out.append(Pricing(f"e_t{th:g}_d{dm}{sfx}", "follow", t_out=10.0, rule=True,
                                   theta=th, d_max=dm, mode=mode))
    # FX-txd's twin: deferral only (the stop's class is the only candidate; see _choose)
    out.append(Pricing("e_dfr", "follow", t_out=10.0, rule=True))
    out.append(Pricing("e_dfr_sh", "follow", t_out=10.0, rule=True, mode="shared"))
    return out


CELLS = [
    # (backhaul, cell, arms, hindsight?)
    ("mission", "results/exp5_p1mbh/b3/s515", "base", ("F", "FX", "H1", "D4"), True),
    ("mission", "results/exp5_p1mbh/b3/s515", "ampharsh", ("F", "FX", "H1", "D4"), True),
    ("seconds", "results/exp5/b3/s515", "ampharsh", ("F", "FX"), False),
    ("seconds", "results/exp5/b3/s515", "base", ("FX",), False),
]


def run_trial(tr: Trial, arm: str, do_hind: bool) -> Tuple[Dict[str, Dict[str, float]], List[int]]:
    """Per pricing: per-mission means over the trial's missions."""
    out: Dict[str, Dict[str, float]] = {}
    for pr in pricings_for(arm):
        tot: Dict[str, float] = {k: 0.0 for k in StopResult.__dataclass_fields__}
        for st in tr.stops:
            res = run_stop(tr, pr, st)
            for k in tot:
                tot[k] += getattr(res, k)
        n = max(len(tr.missions), 1)
        out[pr.name] = {k: v / n for k, v in tot.items()}
    if do_hind and arm in ("F", "FX"):
        for mode, sfx in (("per_class", ""), ("shared", "_sh")):
            lb_t = {k: 0.0 for k in StopResult.__dataclass_fields__}
            gr_t = {k: 0.0 for k in StopResult.__dataclass_fields__}
            op_t = {k: 0.0 for k in StopResult.__dataclass_fields__}
            for st in tr.stops:
                lb, gr = hindsight(tr, mode, st, 10.0)
                op = foresight_opt(tr, mode, st, 10.0)
                for k in lb_t:
                    lb_t[k] += getattr(lb, k)
                    gr_t[k] += getattr(gr, k)
                    op_t[k] += getattr(op, k)
            n = max(len(tr.missions), 1)
            out["o_lb" + sfx] = {k: v / n for k, v in lb_t.items()}
            out["o_gr" + sfx] = {k: v / n for k, v in gr_t.items()}
            out["o_opt" + sfx] = {k: v / n for k, v in op_t.items()}
    runs: List[int] = []
    for st in tr.stops:
        runs.extend(below_floor_runs(tr, st))
    return out, runs


# --------------------------------------------------------------------------- stats
def boot_ci(xs: Sequence[float], n: int = 10000, seed: int = 20261116) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, float("nan"), float("nan")
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return m, means[int(0.025 * n)], means[int(0.975 * n) - 1]


# --------------------------------------------------------------------------- report
LABELS = {
    "a": "(a) as recorded", "b": "(b) rate follows the channel",
    "b0": "(b) with the interference clipped at 0", "b_sh": "(b), shared interference",
    "c": "(c) fastest class, no deferral", "c_sh": "(c), shared",
    "d": "(d) the rule on (a)", "s": "(s) step 1's model, t_out 10 s",
    "s2": "(s) at t_out 2 s", "s_res": "(s), resumable", "s_sh": "(s′) shared",
    "e": "(e) the rule on (s)", "e_sh": "(e′) the rule on (s′)",
    "e_ng": "(e) without the guard", "e_ng_sh": "(e′) without the guard",
    "e_dfr": "(e) deferral only", "e_dfr_sh": "(e′) deferral only",
    "p": "(p) (s) + re-poll", "p_sh": "(p) (s′) + re-poll",
    "o_lb": "loose bound on (s): each session alone", "o_gr": "foresight greedy on (s)",
    "o_lb_sh": "loose bound on (s′)", "o_gr_sh": "foresight greedy on (s′)",
    "o_opt": "(o) foresight optimum on (s)", "o_opt_sh": "(o) foresight optimum on (s′)",
}


def _load_rows(out_dir: str) -> List[dict]:
    with open(os.path.join(out_dir, "step0_replay_trials.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k not in ("backhaul", "cell", "arm", "pricing"):
                r[k] = float(v)
        r["completed"] = r["sessions"] - r["failed"]
    return rows


def _by(rows, backhaul, cell, arm, pricing) -> Dict[int, dict]:
    return {int(r["trial"]): r for r in rows if r["backhaul"] == backhaul and r["cell"] == cell
            and r["arm"] == arm and r["pricing"] == pricing}


def _mean(rows, backhaul, cell, arm, pricing, col) -> Optional[float]:
    xs = [r[col] for r in _by(rows, backhaul, cell, arm, pricing).values()]
    return statistics.fmean(xs) if xs else None


def _paired(rows, bh, cell, arm, p_ref, p_new, col) -> Tuple[float, float, float, int]:
    """Mean over trials of (ref - new) per mission, with a 95% bootstrap CI over trials."""
    a, b = _by(rows, bh, cell, arm, p_ref), _by(rows, bh, cell, arm, p_new)
    xs = [a[t][col] - b[t][col] for t in sorted(set(a) & set(b))]
    if not xs:
        return float("nan"), float("nan"), float("nan"), 0
    m, lo, hi = boot_ci(xs)
    return m, lo, hi, len(xs)


def _fmt(x: Optional[float], nd: int = 1) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_report(out_dir: str) -> str:
    rows = _load_rows(out_dir)
    with open(os.path.join(out_dir, "step0_replay_checks.csv"), encoding="utf-8") as fh:
        checks = list(csv.DictReader(fh))
    with open(os.path.join(out_dir, "step0_replay_runs.json"), encoding="utf-8") as fh:
        runs = json.load(fh)
    L: List[str] = []
    w = L.append
    w("# DDIL step 0: the trace replay")
    w("")
    w("*Read-only, from P1's kept traces; nothing flown. Script: `scripts/exp5/ddil_replay.py`; "
      "plan: `DeveloperDocs/FeRRy_DDIL_Unified_Plan.md`, \"Step 0, extended\". Per-trial rows "
      "are in `step0_replay_trials.csv`, the rebuild checks in `step0_replay_checks.csv`.*")
    w("")
    w("All values are per mission, averaged over a trial's 4 missions and then over the 20 "
      "trials. Hover is dwell only (both passes), as the ledger counts it. Updates are Pass-1 "
      "sessions that carry an update and complete. Stops keep their recorded arrival times, "
      "classes and gates; only time inside a stop moves.")
    w("")

    # ---- checks
    n = len(checks)
    snr = max(float(c["arrival_snr_max_err_db"]) for c in checks)
    drops = sum(float(c["uplink_drop_mismatches"]) for c in checks)
    dwell = max(float(c["dwell_a_max_err_s"]) for c in checks)
    par = all(float(c["channel_params_match"]) == 1.0 for c in checks)
    a_err = 0.0
    for c in checks:
        r = _by(rows, c["backhaul"], c["cell"], c["arm"], "a").get(int(float(c["trial"])))
        if r is not None:
            a_err = max(a_err, abs(r["hover"] - float(c["ledger_dwell_mean"])))
    w("## The rebuild")
    w("")
    w(f"- {n} trials rebuilt. Channel parameters match the recorded ones in every trial: "
      f"{'yes' if par else 'NO'}.")
    w(f"- Recorded arrival SNRs reproduced to within {snr:.1e} dB; uplink drops recomputed "
      f"from the trial seed: {int(drops)} mismatches.")
    w(f"- Pricing (a) reproduces every stop's recorded dwell to within {dwell:.1e} s, and "
      f"each trial's ledger dwell per mission to within {a_err:.1e} s.")
    w("")

    # ---- main table
    order = ["a", "b", "b0", "b_sh", "c", "c_sh", "d", "s", "s2", "s_res", "s_sh",
             "e", "e_sh", "e_ng", "e_ng_sh", "e_dfr", "e_dfr_sh", "p", "p_sh",
             "o_opt", "o_opt_sh", "o_gr", "o_gr_sh", "o_lb", "o_lb_sh"]
    for arm in ("F", "FX"):
        w(f"## {arm}, mission backhaul")
        w("")
        w("| Pricing | Hover, 5 dB | Hover, 17 dB | Updates, 5 dB | Updates, 17 dB "
          "| Failed sessions, 17 dB | Updates lost, 17 dB |")
        w("|---|---|---|---|---|---|---|")
        for p in order:
            h5 = _mean(rows, "mission", "base", arm, p, "hover")
            h17 = _mean(rows, "mission", "ampharsh", arm, p, "hover")
            if h5 is None and h17 is None:
                continue
            w(f"| {LABELS.get(p, p)} | {_fmt(h5)} | {_fmt(h17)} "
              f"| {_fmt(_mean(rows, 'mission', 'base', arm, p, 'updates'), 2)} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, p, 'updates'), 2)} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, p, 'failed'), 2)} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, p, 'lost_updates'), 2)} |")
        w("")

    # ---- decomposition
    w("## How much of F's 17 dB hover rise is the fixed-rate charge")
    w("")
    w("| | F | FX |")
    w("|---|---|---|")
    for lab, p in (("Rise under (a), 17 dB − 5 dB", "a"), ("Rise under (b)", "b"),
                   ("Rise under (b), interference clipped at 0", "b0"),
                   ("Rise under (s)", "s"), ("Rise under (s′)", "s_sh")):
        cells = []
        for arm in ("F", "FX"):
            h17 = _mean(rows, "mission", "ampharsh", arm, p, "hover")
            h5 = _mean(rows, "mission", "base", arm, p, "hover")
            cells.append(_fmt(None if h17 is None or h5 is None else h17 - h5))
        w(f"| {lab} | {cells[0]} | {cells[1]} |")
    for lab, (p1, p2) in (("17 dB hover, (a) − (b): the fixed-rate charge", ("a", "b")),
                          ("17 dB hover, (a) − (s)", ("a", "s")),
                          ("17 dB hover, (a) − (s′)", ("a", "s_sh"))):
        cells = []
        for arm in ("F", "FX"):
            m, lo, hi, k = _paired(rows, "mission", "ampharsh", arm, p1, p2, "hover")
            cells.append(f"{_fmt(m)} [{_fmt(lo)}, {_fmt(hi)}]")
        w(f"| {lab} | {cells[0]} | {cells[1]} |")
    w("")

    # ---- go / no-go
    w("## Go/no-go for step 3 (the per-transfer rule)")
    w("")
    w("The bar (decided 16 Nov, before this was read): the rule saves at least 15 s of hover a "
      "mission at 17 dB on the mission backhaul, at `t_out` = 10 s, under shared interference "
      "((e′) against (s′)), with no fewer sessions completed. The per-class number is the "
      "sensitivity. Differences are (without the rule) − (with it), so a saving is positive; "
      "95% bootstrap CIs over the 20 trials.")
    w("")
    w("| Arm | Comparison | Hover saved | Sessions completed, change | Updates, change | Meets the bar |")
    w("|---|---|---|---|---|---|")
    verdict = {}
    for arm in ("F", "FX"):
        for lab, (p1, p2), primary in (("(e′) vs (s′), shared (primary)", ("s_sh", "e_sh"), True),
                                       ("(e) vs (s), per class", ("s", "e"), False),
                                       ("(e′) without the guard vs (s′)", ("s_sh", "e_ng_sh"), False),
                                       ("(e′) deferral only vs (s′)", ("s_sh", "e_dfr_sh"), False)):
            m, lo, hi, k = _paired(rows, "mission", "ampharsh", arm, p1, p2, "hover")
            mc, loc, hic, _ = _paired(rows, "mission", "ampharsh", arm, p2, p1, "completed")
            mu, lou, hiu, _ = _paired(rows, "mission", "ampharsh", arm, p2, p1, "updates")
            ok = m >= 15.0 and mc >= 0.0
            if primary:
                verdict[arm] = ok
            w(f"| {arm} | {lab} | {_fmt(m)} [{_fmt(lo)}, {_fmt(hi)}] | {_fmt(mc, 2)} "
              f"[{_fmt(loc, 2)}, {_fmt(hic, 2)}] | {_fmt(mu, 2)} [{_fmt(lou, 2)}, {_fmt(hiu, 2)}] "
              f"| {'yes' if ok else 'no'} |")
    w("")
    w(f"**Verdict on the primary model:** F {'meets' if verdict.get('F') else 'does not meet'} "
      f"the bar; FX {'meets' if verdict.get('FX') else 'does not meet'} it.")
    w("")

    # ---- where the saving comes from
    w("## Where the rule's saving comes from")
    w("")
    w("Per mission, mission backhaul. \"Moved\" counts sessions the rule served on a class other "
      "than the stop's; \"saved\" is hover without the rule minus hover with it, 95% CI over trials.")
    w("")
    w("| Arm | Channel | Model | Sessions | Moved to another class | Deferrals | Hover saved |")
    w("|---|---|---|---|---|---|---|")
    for arm in ("F", "FX"):
        for cell, lab in (("base", "5 dB"), ("ampharsh", "17 dB")):
            for base, e, mlab in (("s", "e", "per class"), ("s_sh", "e_sh", "shared")):
                ss = _mean(rows, "mission", cell, arm, e, "sessions") or 0.0
                sw = _mean(rows, "mission", cell, arm, e, "switched") or 0.0
                m, lo, hi, _ = _paired(rows, "mission", cell, arm, base, e, "hover")
                w(f"| {arm} | {lab} | {mlab} | {_fmt(ss, 2)} | {_fmt(sw, 2)} "
                  f"({100.0 * sw / ss if ss else 0.0:.0f}%) "
                  f"| {_fmt(_mean(rows, 'mission', cell, arm, e, 'defers'), 2)} "
                  f"| {_fmt(m)} [{_fmt(lo)}, {_fmt(hi)}] |")
    w("")

    # ---- the grid
    w("## The rule's settings (reported, not used for the decision)")
    w("")
    w("Hover saved a mission at 17 dB, mission backhaul: (s) − (e) per class, and (s′) − (e′) shared.")
    w("")
    w("| θ | D_max | F, per class | F, shared | FX, per class | FX, shared |")
    w("|---|---|---|---|---|---|")
    for th in (1.5, 2.0, 3.0):
        for dm in (1, 2):
            cells = []
            for arm in ("F", "FX"):
                for base, sfx in (("s", ""), ("s_sh", "_sh")):
                    m, lo, hi, k = _paired(rows, "mission", "ampharsh", arm, base,
                                           f"e_t{th:g}_d{dm}{sfx}", "hover")
                    cells.append(_fmt(m))
            w(f"| {th:g} | {dm} | " + " | ".join(cells) + " |")
    w("")

    # ---- d - e
    w("## What step 1's model removes from the rule's recorded-physics gain")
    w("")
    w("| Arm | Rule's saving on (a): (a) − (d) | Rule's saving on (s): (s) − (e) | (d) − (e) |")
    w("|---|---|---|---|")
    for arm in ("F", "FX"):
        ad, _, _, _ = _paired(rows, "mission", "ampharsh", arm, "a", "d", "hover")
        se, _, _, _ = _paired(rows, "mission", "ampharsh", arm, "s", "e", "hover")
        de, lo, hi, _ = _paired(rows, "mission", "ampharsh", arm, "d", "e", "hover")
        w(f"| {arm} | {_fmt(ad)} | {_fmt(se)} | {_fmt(de)} [{_fmt(lo)}, {_fmt(hi)}] |")
    w("")

    # ---- failures
    w("## Step 1's session model: failures")
    w("")
    w("| Arm | Pricing | Sessions failed | Retries failed | Updates lost | Updates |")
    w("|---|---|---|---|---|---|")
    for arm in ("F", "FX"):
        for p in ("s", "s2", "s_res", "s_sh"):
            w(f"| {arm} | {LABELS[p]} | {_fmt(_mean(rows, 'mission', 'ampharsh', arm, p, 'failed'), 2)} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, p, 'retry_failed'), 2)} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, p, 'lost_updates'), 2)} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, p, 'updates'), 2)} |")
    w("")

    # ---- repoll
    w("## The re-poll, (p), at 17 dB on the mission backhaul")
    w("")
    w("| Arm | Model | Members re-polled above the floor | Updates recovered | Updates, (s) | Updates, (p) | Hover added |")
    w("|---|---|---|---|---|---|---|")
    for arm in ("F", "FX", "H1", "D4"):
        for base, p, lab in (("s", "p", "per class"), ("s_sh", "p_sh", "shared")):
            add, lo, hi, _ = _paired(rows, "mission", "ampharsh", arm, p, base, "hover")
            w(f"| {arm} | {lab} | {_fmt(_mean(rows, 'mission', 'ampharsh', arm, p, 'repolled'), 2)} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, p, 'repoll_updates'), 2)} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, base, 'updates'), 2)} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, p, 'updates'), 2)} | {_fmt(add)} |")
    w("")

    # ---- hindsight
    w("## The hindsight bound, (o), and step 3c's gate")
    w("")
    w("Step 3c (a learned per-transfer decision) is built only if (o) is at least 10 s of hover a "
      "mission better than the rule at 17 dB, under shared interference ((o) against (e′)). "
      "(o) is the foresight optimum: the best order, class and wait (up to 30 s, on a 0.5 s "
      "grid) for each stop's sessions with the channel known in advance, found by a DP over the "
      "stop's subsets of sessions, which is exact up to that grid. No policy that reads only the "
      "present can beat it, learned or not. The loose bound (each session alone at its best class and start, ignoring that "
      "sessions run one after another) and the foresight greedy are shown for scale.")
    w("")
    w("| Arm | Model | (e) hover | (o) hover | (e) − (o) | Greedy hover | Loose bound | Updates (e) / (o) | Gate |")
    w("|---|---|---|---|---|---|---|---|---|")
    for arm in ("F", "FX"):
        for e, op, gr, lb, lab in (("e", "o_opt", "o_gr", "o_lb", "per class"),
                                   ("e_sh", "o_opt_sh", "o_gr_sh", "o_lb_sh", "shared (primary)")):
            m1, lo1, hi1, _ = _paired(rows, "mission", "ampharsh", arm, e, op, "hover")
            gate = "" if math.isnan(m1) else ("passes" if m1 >= 10.0 else "fails")
            w(f"| {arm} | {lab} | {_fmt(_mean(rows, 'mission', 'ampharsh', arm, e, 'hover'))} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, op, 'hover'))} "
              f"| {_fmt(m1)} [{_fmt(lo1)}, {_fmt(hi1)}] "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, gr, 'hover'))} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, lb, 'hover'))} "
              f"| {_fmt(_mean(rows, 'mission', 'ampharsh', arm, e, 'updates'), 2)} / "
              f"{_fmt(_mean(rows, 'mission', 'ampharsh', arm, op, 'updates'), 2)} | {gate} |")
    w("")

    # ---- below-floor runs
    w("## Below-floor stretches at 17 dB (for decision 11's burst length)")
    w("")
    w("Runs of consecutive 1 s steps below the floor on the stop's class, for every Pass-1 "
      "target, over each stop and 60 s either side (runs that touch the stop).")
    w("")
    w("| Backhaul, arm | Runs | Mean (s) | Median (s) | 90th percentile (s) | Longest (s) |")
    w("|---|---|---|---|---|---|")
    pooled: List[int] = []
    for key, v in sorted(runs.items()):
        bh, cell, arm = key.split("|")
        if cell != "ampharsh" or not v:
            continue
        if bh == "mission" and arm in ("F", "FX"):
            pooled.extend(v)
        vs = sorted(v)
        w(f"| {bh}, {arm} | {len(v)} | {statistics.fmean(v):.1f} | {statistics.median(v):.0f} "
          f"| {vs[int(0.9 * (len(vs) - 1))]:.0f} | {vs[-1]:.0f} |")
    if pooled:
        ps = sorted(pooled)
        w(f"| **mission, F and FX pooled** | {len(ps)} | **{statistics.fmean(ps):.1f}** "
          f"| {statistics.median(ps):.0f} | {ps[int(0.9 * (len(ps) - 1))]:.0f} | {ps[-1]:.0f} |")
    w("")

    # ---- seconds backhaul
    w("## The seconds backhaul (read, not a gate)")
    w("")
    w("`results/exp5/b3/s515` keeps no default-channel F traces, so F is shown at 17 dB only.")
    w("")
    w("| Arm, cell | (a) | (b) | (s) | (s′) | (e) | (e′) |")
    w("|---|---|---|---|---|---|---|")
    for arm, cell in (("F", "ampharsh"), ("FX", "ampharsh"), ("FX", "base")):
        vals = [_fmt(_mean(rows, "seconds", cell, arm, p, "hover"))
                for p in ("a", "b", "s", "s_sh", "e", "e_sh")]
        w(f"| {arm}, {'17 dB' if cell == 'ampharsh' else '5 dB'} | " + " | ".join(vals) + " |")
    w("")
    path = os.path.join(out_dir, "step0_replay.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/exp5/scores/ddil")
    ap.add_argument("--limit", type=int, default=None, help="trials per arm-cell (a quick look)")
    ap.add_argument("--no-hindsight", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="rewrite step0_replay.md from the CSVs already written")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    if a.report_only:
        print(write_report(os.path.join(ROOT, a.out)))
        return 0
    rows: List[dict] = []
    checks: List[dict] = []
    runs_by: Dict[Tuple[str, str, str], List[int]] = {}
    t_start = time.time()
    for backhaul, root, cell, arms, hind in CELLS:
        for arm in arms:
            tdirs = sorted(glob.glob(os.path.join(ROOT, root, f"n6k1_knee_{cell}__{arm}_traces", "*")))
            tdirs = [d for d in tdirs if os.path.isfile(os.path.join(d, "mule-exp4-mule.jsonl"))]
            if a.limit:
                tdirs = tdirs[:a.limit]
            if not tdirs:
                print(f"[skip] {backhaul} {cell} {arm}: no traces", flush=True)
                continue
            for tdir in tdirs:
                tr = load_trial(tdir)
                trial_idx = int(os.path.basename(tdir).split("__t")[1].split("__")[0])
                # pricing a against the recorded dwell, stop by stop and per mission
                worst_stop = 0.0
                for st in tr.stops:
                    r = run_stop(tr, Pricing("a", "fixed"), st)
                    worst_stop = max(worst_stop, abs(r.hover - st.rec_dwell))
                res, runs = run_trial(tr, arm, hind and not a.no_hindsight)
                checks.append(dict(backhaul=backhaul, cell=cell, arm=arm, trial=trial_idx,
                                   seed=tr.seed, **tr.checks,
                                   dwell_a_max_err_s=worst_stop,
                                   ledger_dwell_mean=statistics.fmean(tr.rec_dwell.values()),
                                   updates_recorded_mean=statistics.fmean(tr.rec_updates.values())))
                runs_by.setdefault((backhaul, cell, arm), []).extend(runs)
                for name, vals in res.items():
                    rows.append(dict(backhaul=backhaul, cell=cell, arm=arm, trial=trial_idx,
                                     seed=tr.seed, pricing=name, missions=len(tr.missions), **vals))
                tr.snr.cache_clear()
            print(f"[done] {backhaul} {cell} {arm}: {len(tdirs)} trials, "
                  f"{time.time() - t_start:.0f} s", flush=True)

    out_dir = os.path.join(ROOT, a.out)
    with open(os.path.join(out_dir, "step0_replay_trials.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "step0_replay_checks.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(checks[0]))
        w.writeheader()
        w.writerows(checks)
    with open(os.path.join(out_dir, "step0_replay_runs.json"), "w", encoding="utf-8") as fh:
        json.dump({"|".join(k): v for k, v in runs_by.items()}, fh)
    print(f"wrote {len(rows)} rows to {out_dir}", flush=True)
    print(write_report(out_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
