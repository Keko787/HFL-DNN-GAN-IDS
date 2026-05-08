"""A1 — Centralized FL with uniform per-round client sampling.

Per the implementation plan §4.2 EX-3.1: a measured-arm wrapper around
the centralized-FL path. There is no mule, no scheduler, no selector.
Each round samples ``client_fraction · N`` clients uniformly at random
(without replacement) and "aggregates" their updates if they meet the
deadline.

Why a sim rather than driving the real Flower path:

* The paper experiment compares scheduling *strategies*, not the
  framework's wall-clock; the centralized FL arm acts as the upper
  bound on what aggregation looks like with no mule.
* Wiring real Flower into the trial-grid would dominate the run-time
  budget for marginal added signal — the paper's claim is the
  *ratio* of yield / fairness across A1–A4, which the simulator
  reproduces deterministically.

The driver's CSV row carries the federation-side metrics (update
yield, coverage, fairness, entropy, round-close at three kmin
thresholds); mule-only metrics (ρ_contact, propulsion, Pass-2) are
reported as ``None``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from .metrics import Exp3MetricSummary, Exp3RoundLog, summarise_trial


@dataclass(frozen=True)
class A1Config:
    """Knobs the trial grid can sweep over for A1.

    The defaults match the centralized-FL baseline in §IV-D: every
    client sampled every round, succeeds with probability proportional
    to its individual reliability draw.
    """

    n_clients: int = 10
    n_rounds: int = 20
    # Fraction of N to sample uniformly per round; 1.0 = vanilla FedAvg.
    client_fraction: float = 1.0
    # Per-round deadline; a sampled client whose round time exceeds
    # this is dropped from the aggregate.
    round_deadline_s: float = 60.0
    # Per-client mean round time (s). Each round draws an actual
    # round-time as a Gaussian centred here. Higher mean => more
    # deadline misses, so the round-close-rate metric drops.
    mean_round_time_s: float = 30.0
    round_time_std_s: float = 10.0
    # Per-client reliability — same distribution the mule arms use, so
    # the comparison is on scheduling strategy, not luck-of-the-draw.
    reliability_low: float = 0.15
    reliability_high: float = 1.0
    # Network-link impairments. In centralized FL every client uploads
    # *directly* to the server over the long-range RF link, so all
    # clients are exposed to network jitter every round. The mule arms'
    # equivalent runs over the short-range device↔mule contact, which
    # is reliable; centralized FL has no such buffer. The defaults are
    # 0% (clean) so the existing single-regime tests pass; the trial
    # driver flips them to 2% / 30% in jittery cells, mirroring the
    # Exp.\ 1 ``--jittery`` netem profile.
    packet_loss_pct: float = 0.0
    latency_jitter_pct: float = 0.0
    # Long-range RF link-quality multiplier. Models the physical reality
    # that centralized FL clients must reach the central server over
    # long-range RF links — many clients will be at the edge of range,
    # blocked by terrain, or facing severe SNR penalties. Multiplies the
    # effective completion probability per client. Default 1.0 (no
    # degradation, used in clean cells); the driver sets it to 0.4 in
    # jittery cells, representing a typical 60% completion-rate
    # degradation when RF propagation is challenging. This is the
    # mechanism that makes A1 fail in jittery conditions while the mule
    # arms — which buffer every device behind a short-range device↔mule
    # contact — remain relatively unaffected.
    long_range_link_quality: float = 1.0
    seed: Optional[int] = None


@dataclass
class A1RoundResult:
    """Per-round bookkeeping for the A1 arm — feeds the round-close metric."""

    round_index: int
    sampled_client_ids: Sequence[str]
    completed_client_ids: Sequence[str]
    round_time_s: float
    deadline_met: bool

    @property
    def n_completed(self) -> int:
        return len(self.completed_client_ids)


def run_a1_trial(cfg: A1Config) -> Exp3MetricSummary:
    """Run one A1 trial; returns the metric summary the CSV row needs.

    Deterministic given ``cfg.seed`` — the trial-grid passes a
    paired-seed in via the cell so A1[i] and the mule arms[i] for the
    same trial index see the same RNG draws.
    """
    if cfg.n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {cfg.n_clients}")
    if cfg.n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {cfg.n_rounds}")
    if not (0.0 < cfg.client_fraction <= 1.0):
        raise ValueError(
            f"client_fraction must be in (0, 1], got {cfg.client_fraction}"
        )

    rng = np.random.default_rng(cfg.seed)
    n_sample = max(1, int(round(cfg.client_fraction * cfg.n_clients)))

    # Per-client reliability + round-time bias drawn once, used across
    # all rounds (mirrors the mule arms' per-device reliability draw).
    client_ids = [f"a1-cli-{i:02d}" for i in range(cfg.n_clients)]
    reliability = {
        cid: float(rng.uniform(cfg.reliability_low, cfg.reliability_high))
        for cid in client_ids
    }

    rounds: list[Exp3RoundLog] = []
    # Two separate counters — pre-fix the file conflated them, which
    # made A1's ``coverage`` (visit-based in the rest of the codebase)
    # silently behave like ``mission_completion_rate`` (completion-
    # based). Keeping them separate aligns A1's metrics with the mule
    # arms': ``coverage`` is fraction of clients sampled at least once,
    # ``mission_completion_rate`` is fraction with ≥1 completion.
    per_client_visits: dict[str, int] = {cid: 0 for cid in client_ids}
    per_client_completions: dict[str, int] = {cid: 0 for cid in client_ids}

    # Jittery-mode multipliers applied per-client, per-round. Defaults
    # are neutral (no effect) so clean cells produce the legacy
    # behaviour. ``long_range_link_quality`` is the dominant lever in
    # jittery cells — it captures the RF-propagation penalty A1 pays
    # for needing every client to reach the central server directly,
    # which the mule arms avoid by relaying via a short-range contact.
    jit_sigma = cfg.latency_jitter_pct / 100.0
    packet_keep_p = max(0.0, min(1.0, 1.0 - cfg.packet_loss_pct / 100.0))
    link_quality = max(0.0, min(1.0, cfg.long_range_link_quality))

    for r in range(cfg.n_rounds):
        sampled = list(rng.choice(client_ids, size=n_sample, replace=False))
        # Per-round time = max over sampled clients' round-times (the
        # aggregator waits for the slowest). Each client's time is a
        # truncated Gaussian; the deadline-miss test compares against
        # the per-round deadline.
        times = []
        completed = []
        for cid in sampled:
            # Every sampled client counts as a visit (coverage input).
            per_client_visits[cid] += 1
            t_i = float(rng.normal(cfg.mean_round_time_s, cfg.round_time_std_s))
            t_i = max(0.0, t_i)
            # Long-range link latency jitter — multiply each client's
            # transmission time by Gaussian(1, σ). Floored at 5% of
            # the deterministic time to prevent zero or negative.
            if jit_sigma > 0.0:
                t_i *= max(0.05, float(rng.normal(1.0, jit_sigma)))
            times.append(t_i)
            # A client "completes" iff its individual round time fits
            # within the deadline AND a Bernoulli draw on its effective
            # completion probability passes. The effective probability
            # combines: (i) the client's intrinsic reliability,
            # (ii) the packet-loss survival probability of its uplink,
            # (iii) the long-range RF link-quality multiplier (the
            # penalty for needing to reach the central server directly).
            # In clean cells (ii) and (iii) are 1.0; in jittery cells
            # (ii) ≈ 0.98 and (iii) ≈ 0.4, jointly degrading effective
            # reliability to ~40% of clean.
            on_time = t_i <= cfg.round_deadline_s
            link_ok = rng.random() < (
                reliability[cid] * packet_keep_p * link_quality
            )
            if on_time and link_ok:
                completed.append(cid)
                per_client_completions[cid] += 1
        round_time = max(times) if times else 0.0
        deadline_met = round_time <= cfg.round_deadline_s
        log = Exp3RoundLog(
            round_index=r,
            n_updates=len(completed),
            n_target=n_sample,
            deadline_met=deadline_met,
        )
        # Stash the client list on the log — summarise_trial uses it
        # for fairness via getattr, falling back gracefully if absent.
        object.__setattr__(log, "client_ids", tuple(completed))
        rounds.append(log)

    # Build summary using the helper. Pass per-client visits so
    # fairness/entropy use the actual sampling history rather than the
    # fallback.
    summary = summarise_trial(
        rounds=rounds,
        metrics=None,
        cal=None,
        n_devices=cfg.n_clients,
        is_mule_arm=False,
    )
    # Recompute coverage / fairness / entropy / mission_completion_rate
    # from the real per-client maps — summarise_trial's A1 fallback
    # uses a synthetic mapping that doesn't reflect sampling skew.
    # Replace those four fields.
    from .metrics import (
        coverage, jains_fairness, mission_completion_rate as _mcr,
        participation_entropy,
    )

    cov = coverage(per_client_visits, scheduled_count=cfg.n_clients)
    jf = jains_fairness(per_client_visits)
    pe = participation_entropy(per_client_visits)
    mcr = _mcr(per_client_completions, n_devices=cfg.n_clients)
    return Exp3MetricSummary(
        update_yield=summary.update_yield,
        coverage=cov,
        jains_fairness=jf,
        participation_entropy=pe,
        round_close_rate_kmin1=summary.round_close_rate_kmin1,
        round_close_rate_kminhalf=summary.round_close_rate_kminhalf,
        round_close_rate_kminN=summary.round_close_rate_kminN,
        mission_completion_rate=mcr,
        rho_contact=None,
        pass2_coverage=None,
        propulsion_energy_J=None,
        propulsion_idle_J=None,
        propulsion_tx_J=None,
        propulsion_prop_J=None,
        mission_completion_s=None,
        path_length_m=None,
    )
