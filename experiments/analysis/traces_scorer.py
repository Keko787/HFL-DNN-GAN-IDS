"""Score retained Exp 4 traces after the fact (Phase 0 of the FeRRy build plan).

``--keep-event-traces`` keeps each trial's JSONL streams under
``<csv>_traces/<cell_id>__<arm>__t<i>__s<seed>/``. Everything below is a
function of those streams, so a new metric needs neither a re-run nor a
change to the trial CSV's schema. Per trial it reports:

* **The standard Exp 4 summary**, from :func:`summarise_observation`, with
  round closure corrected. Traces recorded before Freeze Amendment 5 carry no
  mission round on ``backhaul_upload_lost``, so the recorded CSVs counted
  backhaul-dropped rounds as closed; the consumer now places each loss in the
  mission whose time window contains it.
* **Time to τ** for any list of thresholds: the mission, the cluster round
  and the wall-clock seconds at which accuracy first reached τ. Wall-clock
  time is dominated by local training, not flight, until the Phase 3 mission
  clock exists; missions are the fairer unit for comparing arms.
* **Update age.** After each mission, a device's age is the number of
  missions since its last update was merged into the global model — Age of
  Updates, after Cui et al. (TMC 2024). Network AoU is the weighted mean age
  (uniform weights unless given), reported as its mean over the trial and at
  the end, with the worst and 95th-percentile per-device age.
* **Deadline misses**: the share of devices admitted to Pass 1 whose update
  was not collected by the deadline they were admitted under. Only traces
  that record the Pass-1 plan and outcomes (from 2026-09-28) carry this;
  older ones report it blank.

Single-mule traces only: every recorded Exp 4 trial used one mule, and the
multi-mule runtime is Phase 2 work.

Usage::

    python -m experiments.analysis.traces_scorer \\
        --traces results/exp4_matrix/C_traces --tau 0.82 0.85 --csv scored.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from experiments.exp3.metrics import jains_fairness
from experiments.exp4.events_consumer import (
    Exp4Observation,
    MissionRecord,
    consume_run_dir,
)
from experiments.exp4.metrics import Exp4MetricSummary, summarise_observation


# --------------------------------------------------------------------------- #
# Trial identity
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrialKey:
    """What a trace directory's name says about its trial."""

    cell_id: str
    arm: str
    trial_index: int
    seed: int


def parse_trial_dir(name: str) -> TrialKey:
    """``<cell_id>__<arm>__t<i>__s<seed>``, as ``--keep-event-traces`` writes it."""
    parts = name.split("__")
    if (
        len(parts) < 4
        or not parts[-2].startswith("t") or not parts[-2][1:].isdigit()
        or not parts[-1].startswith("s") or not parts[-1][1:].isdigit()
    ):
        raise ValueError(f"not a trace directory name: {name!r}")
    return TrialKey(
        cell_id="__".join(parts[:-3]),
        arm=parts[-3],
        trial_index=int(parts[-2][1:]),
        seed=int(parts[-1][1:]),
    )


# --------------------------------------------------------------------------- #
# Time to τ
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TauReach:
    """When a trial's global model first reached accuracy τ, if it did."""

    tau: float
    reached: bool
    mission: Optional[int] = None        # 1-based, in completion order
    cluster_round: Optional[int] = None
    wall_s: Optional[float] = None       # from the first mission's start


def tau_reach(obs: Exp4Observation, tau: float) -> TauReach:
    """First aggregated model with accuracy ≥ τ.

    The seeded model (cluster round 0) does not count, matching
    ``t_at_tau_round`` in :mod:`experiments.exp4.metrics`: reaching τ takes at
    least one aggregation. The evaluation runs at the inter-pass dock, inside
    the window of the mission that produced it; if it lands just after that
    window closes, it belongs to the latest mission that had started.
    """
    first = next(
        (e for e in obs.model_evals if e.cluster_round > 0 and e.accuracy >= tau),
        None,
    )
    if first is None:
        return TauReach(tau=float(tau), reached=False)
    missions = _ordered(obs.missions)
    mission = next(
        (i for i, m in enumerate(missions, start=1) if m.contains(first.ts)), None,
    )
    if mission is None and first.ts is not None:
        started = [
            i for i, m in enumerate(missions, start=1)
            if m.started_ts is not None and m.started_ts <= first.ts
        ]
        mission = started[-1] if started else None
    start = missions[0].started_ts if missions else None
    wall_s = (first.ts - start) if (first.ts is not None and start is not None) else None
    return TauReach(
        tau=float(tau), reached=True, mission=mission,
        cluster_round=first.cluster_round, wall_s=wall_s,
    )


# --------------------------------------------------------------------------- #
# Update age and Network AoU
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AgeProfile:
    """Per-device update ages after each mission, rolled up."""

    n_missions: int
    network_aou_mean: float      # weighted mean age, averaged over missions
    network_aou_final: float     # weighted mean age after the last mission
    age_max: int                 # worst age any device reached
    age_p95: float               # 95th percentile over every (device, mission)
    merged_updates: Dict[str, int]
    jain_merged: float           # Jain's index over merged-update counts


def merged_devices(obs: Exp4Observation, mission: MissionRecord) -> Tuple[str, ...]:
    """Devices whose update this mission merged into the global model.

    A Pass-1 CLEAN reaches the model only if the mission's upload to the
    cluster was not lost on the backhaul.
    """
    if mission.mission_round in obs.backhaul_lost_rounds:
        return ()
    return mission.pass_1_clean_devices


def age_profile(
    obs: Exp4Observation,
    devices: Sequence[str],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> AgeProfile:
    """Age of Updates for every device, after every mission.

    ``age_i(m) = m − U_i(m)``, where ``U_i(m)`` is the last mission up to
    ``m`` that merged device ``i``'s update, and 0 if none has, so a device
    never merged ages from the start of the trial. Empty missions count: a
    mission that merged nothing still makes every device one mission older.
    Network AoU is ``Σ_i ω_i · age_i(m)`` with the weights normalised to sum
    to 1; uniform weights make it the mean age.
    """
    devices = list(dict.fromkeys(str(d) for d in devices))
    if not devices:
        raise ValueError("age_profile needs at least one device")
    w = _normalised_weights(devices, weights)

    last_merged = {d: 0 for d in devices}
    merged_counts = {d: 0 for d in devices}
    network_aou: List[float] = []
    ages_seen: List[int] = []
    missions = _ordered(obs.missions)
    for m_index, mission in enumerate(missions, start=1):
        for d in merged_devices(obs, mission):
            if d in last_merged:
                last_merged[d] = m_index
                merged_counts[d] += 1
        ages = {d: m_index - last_merged[d] for d in devices}
        network_aou.append(sum(w[d] * ages[d] for d in devices))
        ages_seen.extend(ages.values())

    if not missions:
        return AgeProfile(
            n_missions=0, network_aou_mean=0.0, network_aou_final=0.0,
            age_max=0, age_p95=0.0, merged_updates=merged_counts,
            jain_merged=jains_fairness(merged_counts),
        )
    return AgeProfile(
        n_missions=len(missions),
        network_aou_mean=float(np.mean(network_aou)),
        network_aou_final=float(network_aou[-1]),
        age_max=int(max(ages_seen)),
        age_p95=float(np.percentile(ages_seen, 95)),
        merged_updates=merged_counts,
        jain_merged=jains_fairness(merged_counts),
    )


# --------------------------------------------------------------------------- #
# Deadline misses
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DeadlineMisses:
    """Admitted devices whose update was not collected by their deadline."""

    missions_with_plan: int
    admitted: int
    missed: int

    @property
    def rate(self) -> Optional[float]:
        return self.missed / self.admitted if self.admitted else None


def deadline_misses(obs: Exp4Observation) -> DeadlineMisses:
    """Score every mission whose trace records its Pass-1 plan.

    A device admitted to the plan is on time only if the mule recorded a
    CLEAN session with it no later than the deadline it was admitted under.
    Everything else is a miss: a late CLEAN, a failed session, a device the
    mule aborted before reaching, and every device of an empty mission.
    """
    with_plan = admitted = missed = 0
    for mission in obs.missions:
        if mission.pass_1_deadlines is None:
            continue
        with_plan += 1
        deadline = dict(mission.pass_1_deadlines)
        on_time = {
            device
            for device, outcome, contact_ts in (mission.pass_1_outcomes or ())
            if outcome == "clean" and device in deadline and contact_ts <= deadline[device]
        }
        admitted += len(deadline)
        missed += sum(1 for device in deadline if device not in on_time)
    return DeadlineMisses(missions_with_plan=with_plan, admitted=admitted, missed=missed)


# --------------------------------------------------------------------------- #
# One trial, and a whole trace root
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrialScore:
    key: TrialKey
    n_devices: int
    summary: Exp4MetricSummary
    missions_empty: int
    backhaul_lost_missions: int
    tau: Tuple[TauReach, ...]
    ages: AgeProfile
    deadlines: DeadlineMisses

    def to_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "cell_id": self.key.cell_id,
            "arm": self.key.arm,
            "trial_index": self.key.trial_index,
            "seed": self.key.seed,
        }
        row.update(self.summary.to_row())
        row.update({
            "missions_empty": self.missions_empty,
            "backhaul_lost_missions": self.backhaul_lost_missions,
            "network_aou_mean": self.ages.network_aou_mean,
            "network_aou_final": self.ages.network_aou_final,
            "age_max": self.ages.age_max,
            "age_p95": self.ages.age_p95,
            "jain_merged": self.ages.jain_merged,
            "deadline_admitted": self.deadlines.admitted,
            "deadline_missed": self.deadlines.missed,
            "deadline_miss_rate": _blank(self.deadlines.rate),
        })
        for reach in self.tau:
            tag = _tau_tag(reach.tau)
            row[f"reached_{tag}"] = int(reach.reached)
            row[f"missions_to_{tag}"] = _blank(reach.mission)
            row[f"rounds_to_{tag}"] = _blank(reach.cluster_round)
            row[f"wall_s_to_{tag}"] = _blank(reach.wall_s)
        return row


def score_trial(
    trace_dir,
    *,
    taus: Sequence[float] = (0.82,),
    weights: Optional[Mapping[str, float]] = None,
) -> TrialScore:
    """Score one retained trial directory."""
    trace_dir = Path(trace_dir)
    if not taus:
        raise ValueError("score_trial needs at least one tau")
    key = parse_trial_dir(trace_dir.name)
    devices = _device_ids(trace_dir)
    obs = consume_run_dir(trace_dir, n_devices=len(devices))
    if not devices:
        devices = sorted(obs.per_device_serves)
        obs.n_devices = len(devices)

    mules = {m.mule_id for m in obs.missions if m.mule_id is not None}
    if len(mules) > 1:
        raise NotImplementedError(
            f"{trace_dir.name}: {len(mules)} mules — the scorer handles "
            "single-mule traces only (the multi-mule runtime is Phase 2 work)"
        )

    mule_cfg = _first_json(trace_dir, "mule-*.json")
    summary = summarise_observation(
        obs,
        n_devices=len(devices),
        rf_range_m=float(mule_cfg.get("rf_range_m") or 0.0),
        n_missions_target=int(mule_cfg.get("n_missions") or len(obs.missions)),
        tau=float(taus[0]),
    )
    return TrialScore(
        key=key,
        n_devices=len(devices),
        summary=summary,
        missions_empty=obs.missions_empty,
        backhaul_lost_missions=len(obs.backhaul_lost_rounds),
        tau=tuple(tau_reach(obs, t) for t in taus),
        ages=age_profile(obs, devices, weights=weights),
        deadlines=deadline_misses(obs),
    )


def score_traces(
    trace_root,
    *,
    taus: Sequence[float] = (0.82,),
    arms: Optional[Iterable[str]] = None,
) -> List[TrialScore]:
    """Score every trial directory under ``trace_root``, in name order.

    Directories whose names are not trial names are skipped.
    """
    wanted = set(arms) if arms is not None else None
    scores: List[TrialScore] = []
    for d in sorted(Path(trace_root).iterdir()):
        if not d.is_dir():
            continue
        try:
            key = parse_trial_dir(d.name)
        except ValueError:
            continue
        if wanted is not None and key.arm not in wanted:
            continue
        scores.append(score_trial(d, taus=taus))
    return scores


def write_scores_csv(scores: Sequence[TrialScore], path) -> None:
    """One row per trial. The column set is fixed by the first score's τ list."""
    if not scores:
        raise ValueError("no scores to write")
    rows = [s.to_row() for s in scores]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="traces_scorer", description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", required=True, nargs="+",
                    help="One or more trace roots, e.g. results/exp4_matrix/C_traces")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.82],
                    help="Accuracy thresholds for time to tau.")
    ap.add_argument("--arms", nargs="+", default=None, help="Only these arms.")
    ap.add_argument("--csv", default=None, help="Write one row per trial here.")
    args = ap.parse_args(argv)

    scores: List[TrialScore] = []
    for root in args.traces:
        scores.extend(score_traces(root, taus=args.tau, arms=args.arms))
    if not scores:
        print("no trial directories found")
        return 1
    if args.csv:
        write_scores_csv(scores, args.csv)
        print(f"wrote {len(scores)} rows to {args.csv}")

    by_arm: Dict[str, List[TrialScore]] = {}
    for s in scores:
        by_arm.setdefault(s.key.arm, []).append(s)
    reach_hdr = "".join(f"{'reach@' + _tau_tag(t):>16}" for t in args.tau)
    print(f"\n{'arm':<6}{'trials':>7}{reach_hdr}{'rcr_k1':>9}{'lost':>6}"
          f"{'NAoU':>7}{'age_max':>9}{'miss':>7}")
    for arm in sorted(by_arm):
        group = by_arm[arm]
        reach = "".join(
            f"{sum(s.tau[i].reached for s in group):>12}/{len(group):<3}"
            for i in range(len(args.tau))
        )
        misses = [s.deadlines.rate for s in group if s.deadlines.rate is not None]
        miss = f"{np.mean(misses):>7.3f}" if misses else f"{'—':>7}"
        print(
            f"{arm:<6}{len(group):>7}{reach}"
            f"{np.mean([s.summary.round_close_rate_kmin1 for s in group]):>9.3f}"
            f"{sum(s.backhaul_lost_missions for s in group):>6}"
            f"{np.mean([s.ages.network_aou_mean for s in group]):>7.2f}"
            f"{np.mean([s.ages.age_max for s in group]):>9.2f}{miss}"
        )
    print("\nrcr_k1: round closure with backhaul-lost rounds excluded. lost: "
          "missions whose upload was lost. NAoU: mean age of the devices' last "
          "merged updates, in missions. miss: deadline-miss rate (blank on "
          "traces without the Pass-1 plan).")
    return 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ordered(missions: Sequence[MissionRecord]) -> List[MissionRecord]:
    """Missions in completion order; the recorded order when timestamps are absent."""
    if all(m.completed_ts is not None for m in missions):
        return sorted(missions, key=lambda m: m.completed_ts)
    return list(missions)


def _normalised_weights(
    devices: Sequence[str], weights: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    if weights is None:
        return {d: 1.0 / len(devices) for d in devices}
    missing = [d for d in devices if d not in weights]
    if missing:
        raise ValueError(f"weights missing for devices {missing}")
    raw = {d: float(weights[d]) for d in devices}
    if any(v < 0 for v in raw.values()) or sum(raw.values()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    total = sum(raw.values())
    return {d: v / total for d, v in raw.items()}


def _device_ids(trace_dir: Path) -> List[str]:
    """The trial's devices, from the cluster's seed list or the device configs."""
    cluster_cfg = _first_json(trace_dir, "cluster*.json")
    seeded = [
        str(s["device_id"]) for s in cluster_cfg.get("seed_devices") or ()
        if isinstance(s, dict) and "device_id" in s
    ]
    if seeded:
        return seeded
    return sorted(
        str(_read_json(p).get("device_id") or p.stem[len("device-"):])
        for p in trace_dir.glob("device-*.json")
    )


def _first_json(trace_dir: Path, pattern: str) -> dict:
    for p in sorted(trace_dir.glob(pattern)):
        return _read_json(p)
    return {}


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _tau_tag(tau: float) -> str:
    return f"tau{tau:g}"


def _blank(v):
    return "" if v is None else v


if __name__ == "__main__":
    raise SystemExit(main())
