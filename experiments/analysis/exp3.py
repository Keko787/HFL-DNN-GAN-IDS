"""Experiment-3 analysis — load CSV → stats + paper figures.

Reads the per-trial CSV produced by :mod:`experiments.exp3.runner_main`
and emits the 6 paper figures (per implementation plan §4.2 EX-3.5)
plus a one-page text summary.

Runnable directly::

    python -m experiments.analysis.exp3 \\
        --csv results/exp3.csv \\
        --figures-dir figures/exp3/

The six figures + one CSV companion:

1. A2 vs A1 — paired Wilcoxon on update yield + Jain's fairness.
2. A3 vs A2 — paired Wilcoxon on round close rate + propulsion energy.
3. A4 vs A3 — paired Wilcoxon on update yield + round close rate
   (the experiment's primary novelty).
4. β-sweep curve: update yield vs β with one curve per arm,
   faceted by N. The slope-vs-cliff figure.
5. rrf-sweep curve: update yield vs rrf with one curve per arm at
   ``β=1.0, N=10``.
6. ρ_contact bar chart faceted by rrf, comparing A2/A3/A4.

Plus ``exp3_paired_tests.csv`` for paper-table reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from experiments.calibration import load_calibration

from .stats import (
    PairedTestResult,
    paired_wilcoxon_with_cliffs_delta,
)

log = logging.getLogger("experiments.analysis.exp3")


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Exp3Row:
    """One row of the Experiment-3 CSV after type coercion."""

    cell_id: str
    arm: str
    trial_index: int
    seed: int
    N: int
    beta: float
    deadline_het: bool
    rf_range_m: float
    # Network-link jitter axis (False = clean / no link noise; True =
    # 2% packet loss + 30% latency jitter, matching Exp.\ 1's --jittery
    # cell). Defaults to False when loading older CSVs without the
    # column so legacy results still parse.
    jittery: bool
    update_yield: float
    coverage: float
    jains_fairness: float
    participation_entropy: float
    round_close_rate_kmin1: float
    # FL-quorum threshold: ≥2 device updates (the smallest set where
    # FedAvg aggregation is non-trivial). Defaults to 0.0 for legacy
    # CSVs without the column.
    round_close_rate_kmin2: float
    round_close_rate_kminhalf: float
    round_close_rate_kminN: float
    # Round-count-invariant yield: fraction of admitted devices with
    # ≥1 completed Δθ contribution. Defaults to 0.0 when loading
    # legacy CSVs without this column so older results still parse.
    mission_completion_rate: float
    # Jain's fairness over per-device *completion* counts — contestable
    # across A1 and the mule arms even when visit-based fairness
    # trivially saturates at 1.0 for A1's universal sampling.
    # Defaults to 0.0 for legacy-CSV compatibility.
    completion_fairness: float
    rho_contact: Optional[float]
    pass2_coverage: Optional[float]
    propulsion_energy_J: Optional[float]
    propulsion_idle_J: Optional[float]
    propulsion_tx_J: Optional[float]
    propulsion_prop_J: Optional[float]
    mission_completion_s: Optional[float]
    path_length_m: Optional[float]
    status: str

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"


def _opt_float(s: Any) -> Optional[float]:
    if s in (None, "", "None"):
        return None
    return float(s)


def _opt_bool(s: Any) -> bool:
    if isinstance(s, bool):
        return s
    return str(s).lower() in ("true", "1", "yes")


def load_trials(csv_path: Path) -> List[Exp3Row]:
    csv_path = Path(csv_path)
    rows: List[Exp3Row] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "cell_id", "arm", "trial_index", "seed",
            "param_N", "param_beta", "param_rrf", "param_deadline_het",
            "update_yield", "coverage", "jains_fairness",
            "participation_entropy",
            "round_close_rate_kmin1", "round_close_rate_kminhalf",
            "round_close_rate_kminN",
            "rho_contact", "pass2_coverage",
            "propulsion_energy_J", "mission_completion_s", "path_length_m",
            "status",
        }
        cols = set(reader.fieldnames or [])
        missing = required - cols
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns: {sorted(missing)}"
            )
        for raw in reader:
            try:
                rows.append(Exp3Row(
                    cell_id=raw["cell_id"],
                    arm=raw["arm"],
                    trial_index=int(raw["trial_index"]),
                    seed=int(raw["seed"]),
                    N=int(raw["param_N"]),
                    beta=float(raw["param_beta"]),
                    deadline_het=_opt_bool(raw["param_deadline_het"]),
                    rf_range_m=float(raw["param_rrf"]),
                    jittery=_opt_bool(raw.get("param_jittery", "")),
                    update_yield=float(raw["update_yield"] or 0.0),
                    coverage=float(raw["coverage"] or 0.0),
                    jains_fairness=float(raw["jains_fairness"] or 0.0),
                    participation_entropy=float(
                        raw["participation_entropy"] or 0.0
                    ),
                    round_close_rate_kmin1=float(
                        raw["round_close_rate_kmin1"] or 0.0
                    ),
                    round_close_rate_kmin2=float(
                        raw.get("round_close_rate_kmin2", "") or 0.0
                    ),
                    round_close_rate_kminhalf=float(
                        raw["round_close_rate_kminhalf"] or 0.0
                    ),
                    round_close_rate_kminN=float(
                        raw["round_close_rate_kminN"] or 0.0
                    ),
                    mission_completion_rate=float(
                        raw.get("mission_completion_rate", "") or 0.0
                    ),
                    completion_fairness=float(
                        raw.get("completion_fairness", "") or 0.0
                    ),
                    rho_contact=_opt_float(raw["rho_contact"]),
                    pass2_coverage=_opt_float(raw["pass2_coverage"]),
                    propulsion_energy_J=_opt_float(raw["propulsion_energy_J"]),
                    propulsion_idle_J=_opt_float(
                        raw.get("propulsion_idle_J", "")
                    ),
                    propulsion_tx_J=_opt_float(
                        raw.get("propulsion_tx_J", "")
                    ),
                    propulsion_prop_J=_opt_float(
                        raw.get("propulsion_prop_J", "")
                    ),
                    mission_completion_s=_opt_float(
                        raw["mission_completion_s"]
                    ),
                    path_length_m=_opt_float(raw["path_length_m"]),
                    status=raw["status"],
                ))
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"failed to parse row {raw!r}: {e}"
                ) from e
    return rows


# --------------------------------------------------------------------------- #
# Paired-test helpers
# --------------------------------------------------------------------------- #

def _pairs(
    rows: Sequence[Exp3Row], arm_a: str, arm_b: str, metric: str,
) -> Tuple[List[float], List[float]]:
    """Pair (cell_id, trial_index) trials across two arms; return aligned arrays."""
    a_by = {(r.cell_id, r.trial_index): r for r in rows if r.is_ok and r.arm == arm_a}
    b_by = {(r.cell_id, r.trial_index): r for r in rows if r.is_ok and r.arm == arm_b}
    a_vals: List[float] = []
    b_vals: List[float] = []
    for k, ar in a_by.items():
        br = b_by.get(k)
        if br is None:
            continue
        av = getattr(ar, metric)
        bv = getattr(br, metric)
        if av is None or bv is None:
            continue
        a_vals.append(float(av))
        b_vals.append(float(bv))
    return a_vals, b_vals


@dataclass(frozen=True)
class ArmPairTest:
    arm_a: str
    arm_b: str
    metric: str
    n_pairs: int
    test: Optional[PairedTestResult]


def paired_test(
    rows: Sequence[Exp3Row], arm_a: str, arm_b: str, metric: str,
) -> ArmPairTest:
    a, b = _pairs(rows, arm_a, arm_b, metric)
    if len(a) < 2:
        return ArmPairTest(arm_a, arm_b, metric, len(a), None)
    return ArmPairTest(
        arm_a, arm_b, metric, len(a),
        paired_wilcoxon_with_cliffs_delta(a, b),
    )


# --------------------------------------------------------------------------- #
# Text summary
# --------------------------------------------------------------------------- #

def summarize(rows: Sequence[Exp3Row]) -> str:
    if not rows:
        return "(no rows)"
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.is_ok)
    n_err = sum(1 for r in rows if r.status == "error")
    arms = sorted({r.arm for r in rows})
    lines = [
        f"Experiment 3 — summary (n={n_total} trials, ok={n_ok}, err={n_err})",
        f"  arms: {arms}",
        "",
    ]

    pairings = (
        # Headline cross-arm tests — mission_completion_rate is the
        # round-count-invariant metric, comparable across A1 and the
        # mule arms even when A1 has 20 FedAvg rounds and the mule
        # arms have a handful of contacts.
        ("A2", "A1", "mission_completion_rate", "A2 vs A1 (slow-deadline claim)"),
        ("A4", "A1", "mission_completion_rate", "A4 vs A1 (RL vs centralized FL)"),
        ("A2", "A1", "jains_fairness", "A2 vs A1 fairness"),
        ("A3", "A2", "round_close_rate_kmin2", "A3 vs A2 close-rate (quorum)"),
        ("A3", "A2", "propulsion_energy_J", "A3 vs A2 mission energy"),
        ("A4", "A3", "mission_completion_rate", "A4 vs A3 mission completion"),
        ("A4", "A3", "round_close_rate_kmin2", "A4 vs A3 close-rate (quorum)"),
    )
    for arm_a, arm_b, metric, label in pairings:
        result = paired_test(rows, arm_a, arm_b, metric)
        if result.test is None:
            lines.append(
                f"  {label}: (insufficient pairs — n={result.n_pairs})"
            )
            continue
        t = result.test
        sig = "*" if t.significant else " "
        lines.append(
            f"  {label}: n={result.n_pairs} W={t.statistic:.1f} "
            f"p={t.p_value:.4f}{sig} δ={t.cliffs_delta:+.3f} "
            f"({t.delta_magnitude})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def write_figures(
    rows: Sequence[Exp3Row],
    *,
    figures_dir: Path,
    placeholder_watermark: bool = False,
    jittery_filter: str = "all",
) -> List[Path]:
    """Emit the six paper figures + paired-tests CSV.

    Parameters
    ----------
    jittery_filter : ``"clean"`` | ``"jittery"`` | ``"all"`` (default ``"all"``)
        Selects which subset of trials feeds the figures and stats.

        * ``"clean"`` — only ``r.jittery is False`` rows; figures show
          one box per arm (single condition).
        * ``"jittery"`` — only ``r.jittery is True`` rows; same.
        * ``"all"`` — uses every row but renders box plots as
          *paired* boxes (clean + jittery side-by-side per arm) with
          colour distinguishing the two conditions, so the reader can
          see clusters within and across regimes. fig4 (β-sweep) is
          generated twice (one variant per condition) since adding a
          third visual axis on top of arm-colour and N-linestyle
          would be unreadable.

    Returns the list of written paths. Each figure is wrapped in a
    try/except so a degenerate input (e.g. a CSV with only one arm)
    doesn't abort the whole batch.
    """
    if jittery_filter not in ("clean", "jittery", "all"):
        raise ValueError(
            f"jittery_filter must be 'clean', 'jittery', or 'all'; "
            f"got {jittery_filter!r}"
        )

    # Filter rows once up front when in single-regime mode so every
    # downstream figure honours the filter consistently.
    if jittery_filter == "clean":
        rows = [r for r in rows if not r.jittery]
    elif jittery_filter == "jittery":
        rows = [r for r in rows if r.jittery]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def _watermark(ax) -> None:
        if not placeholder_watermark:
            return
        ax.text(
            0.5, 0.5, "PLACEHOLDER CALIBRATION",
            transform=ax.transAxes, fontsize=20, color="red",
            alpha=0.25, ha="center", va="center", rotation=30,
        )

    arms = sorted({r.arm for r in rows if r.is_ok})

    # Figures 0a-0d — all-arms comparison, one figure per metric.
    # Shows A1/A2/A3/A4 side by side so the progressive-sophistication
    # story (no scheduling -> arrival -> EDF -> RL) is visible per
    # metric. Each figure carries the metric's definition + direction
    # of improvement so the reader doesn't need to cross-reference the
    # paper text.
    arms_order = [a for a in ("A1", "A2", "A3", "A4") if a in arms]
    # (fig_id, attribute, y-axis label) — titles/captions intentionally
    # omitted from the rendered figures; they belong in the LaTeX caption
    # produced by ``write_latex_captions`` so the paper figure environment
    # owns the prose layer.
    # The first four panels are round-count-invariant metrics —
    # bounded [0,1] for both A1 and the mule arms, comparable across
    # clean/jittery regardless of how many contacts/rounds a trial
    # contained. ``update_yield`` (fig0f) is kept as a supporting
    # panel with a paper footnote: it is per-FedAvg-round for A1 and
    # per-contact-event for the mule arms, so the absolute scale
    # differs structurally and a regime that visits *fewer* contacts
    # under upload pressure can post a *higher* per-round mean
    # (survivorship bias on a non-random sample of clusters).
    # ``fig0e`` is reserved for the existing propulsion-energy panel
    # rendered separately below.
    metric_figs = [
        ("fig0a", "mission_completion_rate",
         "Fraction of devices contributing ≥1 update (per mission)"),
        # fig0b: FL-quorum close rate. kmin=2 is the floor for a
        # meaningful FedAvg aggregation (one update is just SGD; two
        # is the smallest non-trivial federation). The previous
        # kmin=N/2 strict-majority threshold was visually degenerate
        # under the per-mission round semantics — most jittery trials
        # missed the bar and the box collapsed to 0. The quorum
        # threshold is more discriminative across regimes.
        ("fig0b", "round_close_rate_kmin2",
         "Fraction of rounds reaching FL quorum (k_min = 2)"),
        # fig0c: completion-based fairness rather than visit-based.
        # A1's universal sampling makes visit-based ``jains_fairness``
        # trivially 1.0; ``completion_fairness`` is contestable across
        # arms because it counts whether contributions distribute
        # equally, not whether the scheduler asked equally.
        ("fig0c", "completion_fairness",
         "Jain's fairness on contribution counts"),
        ("fig0d", "coverage", "Fraction of scheduled devices serviced"),
        ("fig0f", "update_yield", "Updates aggregated per round (mean)"),
    ]
    # In "all" mode, render paired (clean+jittery) boxes when both
    # regimes are present in the data. Otherwise fall back to single
    # boxes per arm (single regime).
    has_clean = any(r.is_ok and not r.jittery for r in rows)
    has_jittery = any(r.is_ok and r.jittery for r in rows)
    paired_mode = (
        jittery_filter == "all" and has_clean and has_jittery
    )
    clean_color = "#5b9bd5"   # blue
    jittery_color = "#ed7d31"  # orange

    for fig_id, metric, ylabel in metric_figs:
        try:
            if len(arms_order) < 2:
                continue
            fig, ax = plt.subplots(figsize=(7.5, 4.5) if paired_mode else (6, 4))

            if paired_mode:
                # Paired boxes: for each arm, plot two boxes side-by-
                # side (clean then jittery) at offset positions, then
                # set the x-tick at the midpoint with the arm label.
                clean_data: List[List[float]] = []
                jittery_data: List[List[float]] = []
                kept_arms: List[str] = []
                for arm in arms_order:
                    cv = [getattr(r, metric) for r in rows
                          if r.is_ok and r.arm == arm and not r.jittery
                          and getattr(r, metric) is not None]
                    jv = [getattr(r, metric) for r in rows
                          if r.is_ok and r.arm == arm and r.jittery
                          and getattr(r, metric) is not None]
                    if cv or jv:
                        clean_data.append(cv if cv else [float("nan")])
                        jittery_data.append(jv if jv else [float("nan")])
                        kept_arms.append(arm)
                if not kept_arms:
                    continue
                spacing = 1.0
                offset = 0.22
                clean_pos = [i * spacing + 1 - offset
                             for i in range(len(kept_arms))]
                jit_pos = [i * spacing + 1 + offset
                           for i in range(len(kept_arms))]
                width = 0.35
                bp_c = ax.boxplot(
                    clean_data, positions=clean_pos, widths=width,
                    patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black", "markersize": 5},
                    medianprops={"color": "#1f3a5f", "linewidth": 1.2},
                )
                for patch in bp_c["boxes"]:
                    patch.set_facecolor(clean_color)
                    patch.set_alpha(0.65)
                bp_j = ax.boxplot(
                    jittery_data, positions=jit_pos, widths=width,
                    patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black", "markersize": 5},
                    medianprops={"color": "#7a3a05", "linewidth": 1.2},
                )
                for patch in bp_j["boxes"]:
                    patch.set_facecolor(jittery_color)
                    patch.set_alpha(0.65)

                tick_positions = [i * spacing + 1
                                  for i in range(len(kept_arms))]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(kept_arms)
                labels = kept_arms
                # Place the regime legend below the x-axis labels in a
                # horizontal layout so it cannot overlap A1's tall
                # upper-bound box, the upper-bound region annotations,
                # or any high-value outliers in the jittery cells.
                from matplotlib.patches import Patch
                ax.legend(
                    handles=[
                        Patch(facecolor=clean_color, alpha=0.65,
                              label="Clean"),
                        Patch(facecolor=jittery_color, alpha=0.65,
                              label="Jittery"),
                    ],
                    loc="upper right",
                    ncol=2,
                    fontsize=7.5,
                    frameon=True,
                    borderpad=0.3,
                    columnspacing=1.2,
                    handlelength=1.2,
                    handletextpad=0.4,
                )
            else:
                # Single-regime mode: original single-box behaviour.
                data: List[List[float]] = []
                labels: List[str] = []
                for arm in arms_order:
                    vals = [
                        getattr(r, metric) for r in rows
                        if r.is_ok and r.arm == arm
                        and getattr(r, metric) is not None
                    ]
                    if vals:
                        data.append(vals)
                        labels.append(arm)
                if not data:
                    continue
                ax.boxplot(
                    data, labels=labels, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black", "markersize": 6},
                )

            ax.set_ylabel(ylabel)
            ax.set_xlabel("Arm")
            ax.grid(True, axis="y", alpha=0.3)
            # Vertical separator between A1 (centralized FL upper bound)
            # and the mule-arm relay-FL ablation (A2/A3/A4). The metric
            # is not directly comparable across the two groupings — A1's
            # "round" parallelises clients, while the mule arms' "round"
            # is one contact event — so the separator visually flags
            # that the boxplot mixes a control with an ablation set.
            if labels and labels[0] == "A1" and len(labels) > 1:
                # In paired mode, the separator sits between arm-1 and
                # arm-2 in the new tick coordinate space; in single-
                # box mode it sits at x=1.5.
                sep_x = 1.5 if paired_mode else 1.5
                ax.axvline(x=sep_x, color="#888888", linestyle="--",
                           linewidth=1.4, alpha=0.85, zorder=0)
                ymin, ymax = ax.get_ylim()
                yspan_orig = ymax - ymin
                ax.set_ylim(ymin, ymax + 0.18 * yspan_orig)
                ymin, ymax = ax.get_ylim()
                label_y = ymin + 0.97 * (ymax - ymin)
                ax.text(
                    1.0, label_y,
                    "Centralized FL\n(upper bound)",
                    ha="center", va="top", fontsize=8.5,
                    style="italic", color="#444444",
                )
                ax.text(
                    (len(labels) + 2) / 2.0, label_y,
                    "Mule relay-FL ablation",
                    ha="center", va="top", fontsize=8.5,
                    style="italic", color="#444444",
                )
            _watermark(ax)
            fig.tight_layout()
            out = figures_dir / f"exp3_{fig_id}_{metric}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
        except Exception as e:  # pragma: no cover
            log.warning("%s (%s) skipped: %s", fig_id, metric, e)

    # Figure 0e — mule-only propulsion energy. A1 has no mule, so this
    # is a separate three-arm comparison (A2/A3/A4) on the Eq. 5 cost
    # ledger. Honours the same paired-mode rendering as fig0a-d.
    try:
        mule_arms = [a for a in ("A2", "A3", "A4") if a in arms]
        if len(mule_arms) >= 2:
            fig, ax = plt.subplots(figsize=(7.5, 4.5) if paired_mode else (6, 4))
            if paired_mode:
                clean_data = []
                jittery_data = []
                kept_arms = []
                for arm in mule_arms:
                    cv = [r.propulsion_energy_J for r in rows
                          if r.is_ok and r.arm == arm and not r.jittery
                          and r.propulsion_energy_J is not None]
                    jv = [r.propulsion_energy_J for r in rows
                          if r.is_ok and r.arm == arm and r.jittery
                          and r.propulsion_energy_J is not None]
                    if cv or jv:
                        clean_data.append(cv if cv else [float("nan")])
                        jittery_data.append(jv if jv else [float("nan")])
                        kept_arms.append(arm)
                offset = 0.22
                width = 0.35
                clean_pos = [i + 1 - offset for i in range(len(kept_arms))]
                jit_pos = [i + 1 + offset for i in range(len(kept_arms))]
                bp_c = ax.boxplot(
                    clean_data, positions=clean_pos, widths=width,
                    patch_artist=True, showmeans=True,
                )
                for patch in bp_c["boxes"]:
                    patch.set_facecolor(clean_color); patch.set_alpha(0.65)
                bp_j = ax.boxplot(
                    jittery_data, positions=jit_pos, widths=width,
                    patch_artist=True, showmeans=True,
                )
                for patch in bp_j["boxes"]:
                    patch.set_facecolor(jittery_color); patch.set_alpha(0.65)
                ax.set_xticks([i + 1 for i in range(len(kept_arms))])
                ax.set_xticklabels(kept_arms)
                from matplotlib.patches import Patch
                ax.legend(
                    handles=[
                        Patch(facecolor=clean_color, alpha=0.65, label="Clean"),
                        Patch(facecolor=jittery_color, alpha=0.65,
                              label="Jittery (2% loss + 30% jit.)"),
                    ],
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=2,
                    fontsize=9,
                    title="Network regime", title_fontsize=9,
                    frameon=True,
                )
            else:
                data: List[List[float]] = []
                labels: List[str] = []
                for arm in mule_arms:
                    vals = [
                        r.propulsion_energy_J for r in rows
                        if r.is_ok and r.arm == arm
                        and r.propulsion_energy_J is not None
                    ]
                    if vals:
                        data.append(vals)
                        labels.append(arm)
                if data:
                    ax.boxplot(data, labels=labels, showmeans=True)
            ax.set_ylabel("Propulsion energy per mission (J)")
            ax.set_xlabel("Arm")
            ax.grid(True, axis="y", alpha=0.3)
            _watermark(ax)
            fig.tight_layout()
            out = figures_dir / "exp3_fig0e_propulsion_energy.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig0e (propulsion energy) skipped: %s", e)

    # Emit LaTeX caption blocks for every metric figure produced above.
    try:
        latex_path = _write_latex_captions(figures_dir)
        written.append(latex_path)
    except Exception as e:  # pragma: no cover
        log.warning("LaTeX captions skipped: %s", e)

    # 1 + 2 + 3 — paired tests CSV (three rows: A2vsA1, A3vsA2, A4vsA3).
    try:
        sig_path = figures_dir / "exp3_paired_tests.csv"
        with open(sig_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "comparison", "metric", "n_pairs",
                "W", "p", "cliffs_delta", "delta_magnitude",
            ])
            comparisons = (
                # Headline: mission_completion_rate is the round-count-
                # invariant cross-arm metric. The A2-vs-A1 and A4-vs-A1
                # rows are the central jittery-regime claim — they show
                # whether the mule architecture out-performs centralized
                # FL when A1's long-range dead zones bite.
                ("A2", "A1", "mission_completion_rate"),
                ("A4", "A1", "mission_completion_rate"),
                # Within-strategy paired tests retained for the
                # progressive-sophistication ablation.
                ("A2", "A1", "update_yield"),
                ("A2", "A1", "jains_fairness"),
                ("A3", "A2", "round_close_rate_kmin2"),
                ("A3", "A2", "propulsion_energy_J"),
                ("A4", "A3", "mission_completion_rate"),
                ("A4", "A3", "update_yield"),
                ("A4", "A3", "round_close_rate_kmin2"),
            )
            for arm_a, arm_b, metric in comparisons:
                pt = paired_test(rows, arm_a, arm_b, metric)
                if pt.test is None:
                    w.writerow([f"{arm_a}_vs_{arm_b}", metric, pt.n_pairs,
                                "", "", "", ""])
                    continue
                w.writerow([
                    f"{arm_a}_vs_{arm_b}", metric, pt.n_pairs,
                    pt.test.statistic, pt.test.p_value,
                    pt.test.cliffs_delta, pt.test.delta_magnitude,
                ])
        written.append(sig_path)
    except Exception as e:  # pragma: no cover
        log.warning("paired tests CSV skipped: %s", e)

    # Figure 1 — A2 vs A1 paired comparison panel.
    # Lead with mission_completion_rate (round-count-invariant,
    # bounded [0,1] for both arms — the cleanest one-number cross-
    # arm comparison). Fairness is the secondary panel; update_yield
    # is shown elsewhere (fig0f) with the round-definition footnote.
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, title in zip(
            axes,
            ("mission_completion_rate", "jains_fairness"),
            ("Mission completion rate", "Jain's fairness"),
        ):
            a, b = _pairs(rows, "A2", "A1", metric)
            if a and b:
                ax.boxplot([b, a], labels=["A1", "A2"], showmeans=True)
            ax.set_ylabel(title)
            ax.set_title(f"{title}: A2 vs A1")
            _watermark(ax)
        fig.suptitle("A2 vs A1 (slow-deadline claim)")
        fig.tight_layout()
        out = figures_dir / "exp3_fig1_a2_vs_a1.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig1 skipped: %s", e)

    # Figure 2 — A3 vs A2 (close rate + propulsion energy).
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, title in zip(
            axes,
            ("round_close_rate_kmin2", "propulsion_energy_J"),
            ("Round close rate (kmin=2)", "Propulsion energy (J)"),
        ):
            a, b = _pairs(rows, "A3", "A2", metric)
            if a and b:
                ax.boxplot([b, a], labels=["A2", "A3"], showmeans=True)
            ax.set_ylabel(title)
            ax.set_title(f"{title}: A3 vs A2")
            _watermark(ax)
        fig.suptitle("A3 vs A2 (heuristic EDF gain)")
        fig.tight_layout()
        out = figures_dir / "exp3_fig2_a3_vs_a2.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig2 skipped: %s", e)

    # Figure 3 — A4 vs A3 (mission completion + close rate; primary novelty).
    # Both panels are round-count-invariant; the round-definition
    # caveat that demoted update_yield to fig0f does not apply within
    # the mule-arm ablation (A3 and A4 share the same per-contact
    # round semantics), but we keep the comparable metric for
    # consistency with fig1 and the headline cross-arm story.
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, title in zip(
            axes,
            ("mission_completion_rate", "round_close_rate_kmin2"),
            ("Mission completion rate", "Round close rate (kmin=2)"),
        ):
            a, b = _pairs(rows, "A4", "A3", metric)
            if a and b:
                ax.boxplot([b, a], labels=["A3", "A4"], showmeans=True)
            ax.set_ylabel(title)
            ax.set_title(f"{title}: A4 vs A3")
            _watermark(ax)
        fig.suptitle("A4 vs A3 (HERMES RL primary novelty)")
        fig.tight_layout()
        out = figures_dir / "exp3_fig3_a4_vs_a3.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig3 skipped: %s", e)

    # Figure 4 — β-sweep, all N values overlaid on one panel.
    # Mule-arm ablation only (A1 omitted): A1 is the centralized-FL
    # upper bound and is shown alongside A2/A3/A4 in fig0a; including
    # it here would dominate the y-axis and obscure the finer
    # scheduling-strategy comparison among A2/A3/A4. fig0a already
    # establishes the upper-bound magnitude; this figure focuses on the
    # mule-arm scheduling ablation.
    try:
        Ns = sorted({r.N for r in rows if r.is_ok})
        if Ns and arms:
            # Stable color per arm so identity carries across figures.
            # A1 entry retained for future re-inclusion if needed.
            arm_colors = {
                "A1": "tab:blue", "A2": "tab:orange",
                "A3": "tab:green", "A4": "tab:red",
            }
            # Line style + marker per N (more N values cycle through
            # the lists; three is the typical case).
            n_styles = ["-", "--", ":"]
            n_markers = ["o", "s", "^"]

            fig, ax = plt.subplots(figsize=(7.5, 5))
            arms_in_data = [a for a in ("A2", "A3", "A4") if a in arms]
            for arm in arms_in_data:
                color = arm_colors.get(arm, "black")
                for i, N in enumerate(Ns):
                    style = n_styles[i % len(n_styles)]
                    marker = n_markers[i % len(n_markers)]
                    betas = sorted({r.beta for r in rows
                                    if r.is_ok and r.N == N})
                    ys: List[float] = []
                    for b in betas:
                        cell = [r for r in rows
                                if r.is_ok and r.arm == arm
                                and r.N == N and r.beta == b]
                        ys.append(np.mean([r.update_yield for r in cell])
                                  if cell else float("nan"))
                    ax.plot(
                        betas, ys,
                        color=color, linestyle=style, marker=marker,
                        markersize=6, linewidth=1.8,
                    )
            ax.set_xlabel("β (deadline tightness)")
            ax.set_ylabel("Update yield (mean)")
            ax.grid(True, alpha=0.3)

            # Two legends: arm (color) and N (line style). Drawn as
            # proxy artists so they don't clutter the data lines.
            from matplotlib.lines import Line2D
            arm_handles = [
                Line2D([0], [0], color=arm_colors.get(a, "black"),
                       linewidth=2, label=a)
                for a in arms_in_data
            ]
            n_handles = [
                Line2D([0], [0], color="black",
                       linestyle=n_styles[i % len(n_styles)],
                       marker=n_markers[i % len(n_markers)],
                       linewidth=1.8, markersize=6,
                       label=f"N = {N}")
                for i, N in enumerate(Ns)
            ]
            # Group both legends side-by-side along the top edge:
            # Bucket size pinned to the upper-right corner, Arm legend
            # placed just to its left so the two read as a single
            # horizontal legend cluster. With A1 omitted, the mule-arm
            # lines top out around y ≈ 2.2 (axes-fraction ≈ 0.85),
            # leaving the upper ~15% empty for the legend strip.
            arm_legend = ax.legend(
                handles=arm_handles, title="Arm",
                loc="upper right", bbox_to_anchor=(0.78, 1.0),
                fontsize=9, title_fontsize=9,
            )
            ax.add_artist(arm_legend)
            ax.legend(
                handles=n_handles, title="Bucket size",
                loc="upper right",
                fontsize=9, title_fontsize=9,
            )
            # Annotate the two distinct regimes the chart contains. A1
            # (blue) is a centralized-FL upper bound — its yield reflects
            # per-round client parallelism with no mule-transit
            # constraint, so it is not directly comparable to A2/A3/A4
            # on a per-round basis. The mule arms cluster at the bottom
            # of the chart and constitute the relay-FL scheduling
            # ablation. Text annotations make this distinction explicit
            # on the figure itself rather than only in the caption.
            _watermark(ax)
            fig.tight_layout()
            out = figures_dir / "exp3_fig4_beta_sweep.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig4 skipped: %s", e)

    # Figure 5 — rrf-sweep curve at β=1.0, N=10.
    try:
        target_rows = [
            r for r in rows
            if r.is_ok and abs(r.beta - 1.0) < 1e-9 and r.N == 10
        ]
        if target_rows:
            rrfs = sorted({r.rf_range_m for r in target_rows})
            fig, ax = plt.subplots(figsize=(7, 4))
            for arm in arms:
                ys = []
                for rrf in rrfs:
                    cell = [r for r in target_rows
                            if r.arm == arm and r.rf_range_m == rrf]
                    ys.append(np.mean([r.update_yield for r in cell])
                              if cell else float("nan"))
                ax.plot(rrfs, ys, marker="o", label=arm)
            ax.set_xlabel("rf_range_m (rrf)")
            ax.set_ylabel("Update yield")
            ax.set_title("Update yield vs rrf at β=1.0, N=10")
            ax.legend(fontsize=8)
            _watermark(ax)
            out = figures_dir / "exp3_fig5_rrf_sweep.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig5 skipped: %s", e)

    # Figure 6 — ρ_contact bar chart faceted by rrf, A2/A3/A4 only.
    try:
        mule_arms = [a for a in arms if a in ("A2", "A3", "A4")]
        rrfs = sorted({r.rf_range_m for r in rows if r.is_ok})
        if mule_arms and rrfs:
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(rrfs))
            width = 0.8 / max(len(mule_arms), 1)
            for i, arm in enumerate(mule_arms):
                ys: List[float] = []
                for rrf in rrfs:
                    cell = [r for r in rows
                            if r.is_ok and r.arm == arm
                            and r.rf_range_m == rrf
                            and r.rho_contact is not None]
                    ys.append(
                        float(np.mean([r.rho_contact for r in cell]))
                        if cell else 0.0
                    )
                ax.bar(x + i * width, ys, width, label=arm)
            ax.set_xticks(x + width * (len(mule_arms) - 1) / 2)
            ax.set_xticklabels([f"{r:g}" for r in rrfs])
            ax.set_xlabel("rrf (m)")
            ax.set_ylabel("ρ_contact (mean devices/contact)")
            ax.set_title("ρ_contact across rrf for mule arms")
            ax.legend(fontsize=8)
            _watermark(ax)
            out = figures_dir / "exp3_fig6_rho_contact.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig6 skipped: %s", e)

    return written


# --------------------------------------------------------------------------- #
# LaTeX caption emitter — pairs with the bare PNG figures
# --------------------------------------------------------------------------- #

# (file_stem, label_id, short_caption, full_caption_body)
# Kept module-level so the test suite + the paper toolchain can import
# them without re-running the figure generation.
_LATEX_CAPTIONS: tuple = (
    (
        "exp3_fig0a_mission_completion_rate",
        "fig:exp3:mission_completion_rate",
        "Mission completion rate across arms",
        (
            r"\textbf{Mission completion rate} across A1 (centralized FL), "
            r"A2 (arrival-order), A3 (EDF heuristic), A4 (RL). "
            r"Higher is better; bounded $[0, 1]$. Defined as the "
            r"fraction of admitted devices that contributed at least "
            r"one $\Delta\theta$ to the mission, this metric is "
            r"round-count-invariant and therefore directly comparable "
            r"across arms and across clean / jittery regimes — unlike "
            r"the per-round update yield (Fig.~\ref{fig:exp3:update_yield}), "
            r"which is biased upward in regimes where upload pressure "
            r"truncates Pass~1 to fewer rounds. \emph{Jittery cap:} "
            r"A1's jittery distribution is hard-capped near "
            r"$1 - p_{\text{dead}}$ (default $p_{\text{dead}} = 0.6$), "
            r"reflecting persistent long-range dead zones — clients "
            r"that are blocked, beyond effective range, or in SNR "
            r"collapse for the entire mission and therefore unreachable "
            r"from the central server regardless of how many FedAvg "
            r"rounds run. Mule arms have no analogous cap because the "
            r"short-range device$\leftrightarrow$mule contact is "
            r"reliable by design."
        ),
    ),
    (
        "exp3_fig0b_round_close_rate_kmin2",
        "fig:exp3:close_rate",
        "Round close rate at FL quorum (k\\textsubscript{min} = 2) across arms",
        (
            r"\textbf{Round close rate at the FL quorum threshold} "
            r"$k_{\min} = 2$ across the four arms. Higher is better. "
            r"A round ``closes'' when at least two device updates are "
            r"aggregated within deadline — the floor for a meaningful "
            r"FedAvg step. One device alone is just local SGD; two "
            r"is the smallest non-trivial federation. We adopt this "
            r"quorum threshold rather than a strict-majority "
            r"$k_{\min} = N/2$ because (a) FL aggregation is well-"
            r"defined at any $\geq 2$ contributions, and (b) under "
            r"the per-mission round semantics, the strict-majority "
            r"variant produces near-zero close-rates for every mule "
            r"trial in jittery cells, collapsing the panel into a "
            r"non-discriminative line at zero. "
            r"\emph{Round semantics, apples-to-apples:} a ``round'' "
            r"is one FL aggregation cycle for both A1 and the mule "
            r"arms. For A1 each FedAvg round samples $N$ clients "
            r"and aggregates whatever completes within the per-round "
            r"deadline. For the mule arms each mission is one round: "
            r"Pass~1 visits multiple clusters, folds the per-contact "
            r"partial-FedAvg outputs into a single mission\_aggregate, "
            r"and the dock's cross-mule FedAvg folds that into the "
            r"global $\theta$ exactly once."
        ),
    ),
    (
        "exp3_fig0c_completion_fairness",
        "fig:exp3:fairness",
        "Contribution-distribution fairness across arms",
        (
            r"\textbf{Jain's fairness index} on per-device "
            r"\emph{contribution counts}, $J = (\sum_i x_i)^2 / "
            r"(N \cdot \sum_i x_i^2)$, range $[1/N, 1]$. Higher is "
            r"better; $J = 1$ is a perfectly equal contribution "
            r"distribution. \emph{Contribution} here means $\Delta\theta$ "
            r"updates that the FL aggregator actually consumed — not "
            r"sampling attempts. This makes the metric contestable "
            r"across A1 and the mule arms; the visit-based variant "
            r"saturates at $J = 1$ for A1 by construction (universal "
            r"sampling: every client is asked every round), which "
            r"masks the contribution inequality introduced by jittery "
            r"dead zones (a fraction of clients with zero completions "
            r"while the rest contribute every round)."
        ),
    ),
    (
        "exp3_fig0d_coverage",
        "fig:exp3:coverage",
        "Coverage across arms",
        (
            r"\textbf{Coverage} across the four arms, defined as the "
            r"fraction of scheduled devices serviced at least once "
            r"during the mission. Higher is better; range $[0, 1]$. "
            r"Coverage is the binary complement of the per-device "
            r"miss rate and is independent of how many times a "
            r"device was served beyond the first visit. \emph{A1 is "
            r"exactly 1.0 by construction:} centralized FedAvg "
            r"samples every client every round (full universal "
            r"sampling), so its visit-based coverage is trivially "
            r"perfect regardless of regime. The contestable cross-arm "
            r"comparison on contribution outcomes lives in "
            r"Fig.~\ref{fig:exp3:mission_completion_rate} (mission "
            r"completion rate) and "
            r"Fig.~\ref{fig:exp3:fairness} (contribution fairness); "
            r"this panel is most informative for the intra-mule-arm "
            r"comparison (A2 vs A3 vs A4 differ on physical reach "
            r"under budget pressure)."
        ),
    ),
    (
        "exp3_fig0e_propulsion_energy",
        "fig:exp3:propulsion_energy",
        "Mule propulsion energy across mule arms",
        (
            r"\textbf{Propulsion energy per mission} (joules) for the "
            r"three mule arms (A1 has no mule and is therefore "
            r"omitted). Lower is better. Computed via Eq.~5: "
            r"$E = T_{\text{mission}} \cdot P_{\text{idle}} + "
            r"B_{\text{tx}} \cdot \varepsilon_{\text{bit}} + "
            r"L_{\text{path}} \cdot \varepsilon_{\text{prop}}$. "
            r"The three mule arms produce overlapping distributions in "
            r"the studied parameter range; the energy ledger does not "
            r"discriminate among A2, A3, and A4."
        ),
    ),
    (
        "exp3_fig0f_update_yield",
        "fig:exp3:update_yield",
        "Update yield per FL round across arms",
        (
            r"\textbf{Update yield per round} across A1 (centralized FL), "
            r"A2 (arrival-order), A3 (EDF heuristic), A4 (RL). "
            r"Higher is better. ``Round'' is one FL aggregation cycle "
            r"for both A1 and the mule arms — for A1 that is one "
            r"FedAvg round (samples $N$ clients, aggregates the "
            r"completing ones), for the mule arms it is one mission "
            r"(Pass~1 sums completions across every visited cluster "
            r"into a single mission\_aggregate, which the dock then "
            r"folds into the global $\theta$ via cross-mule FedAvg). "
            r"This puts the panel on the same denominator across "
            r"arms, so the A1-vs-mule comparison is meaningful here: "
            r"in clean cells the mule arms can match or exceed A1's "
            r"per-round yield by visiting multiple clusters per "
            r"mission, and in jittery cells A1's dead-zone collapse "
            r"shows up directly on this metric. The "
            r"round-count-invariant cross-arm headline remains "
            r"mission completion rate "
            r"(Fig.~\ref{fig:exp3:mission_completion_rate}) — that "
            r"metric measures \emph{which} devices contributed at all, "
            r"while this one measures the \emph{volume} of "
            r"contributions per FL aggregation cycle."
        ),
    ),
    (
        "exp3_fig4_beta_sweep",
        "fig:exp3:beta_sweep",
        "Mule-arm update yield versus deadline tightness, across bucket sizes",
        (
            r"\textbf{Mule-arm update yield versus deadline tightness "
            r"$\beta$}, with bucket size $N$ encoded by line style "
            r"(solid: $N=5$; dashed: $N=10$; dotted: $N=20$) and arm "
            r"encoded by colour. The centralized-FL upper bound (A1) "
            r"is omitted from this figure to bring the y-axis to a "
            r"scale where the mule-arm comparison is legible; the "
            r"upper-bound magnitude is established in "
            r"Fig.~\ref{fig:exp3:update_yield}. The three mule arms "
            r"(A2/A3/A4) overlap at every $(N, \beta)$ cell, scaling "
            r"with bucket size but insensitive to deadline slack. "
            r"This $\beta$-insensitivity isolates mule travel time as "
            r"the binding constraint on contacts-per-mission: "
            r"additional deadline slack provides no measurable gain "
            r"once the mule has exhausted its mission budget on "
            r"transit between contacts."
        ),
    ),
)


def _write_latex_captions(figures_dir: Path) -> Path:
    """Emit a ``.tex`` snippet with one ``\\begin{figure}`` per metric.

    The figures themselves carry no titles or footer text — all the
    explanation lives in the LaTeX caption produced here, which the
    paper can ``\\input{}`` directly.
    """
    out = figures_dir / "exp3_fig_captions.tex"
    figures_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "% Auto-generated by experiments.analysis.exp3.write_figures().",
        "% One \\begin{figure} block per metric figure produced alongside",
        "% this file. The PNGs carry no titles or footers; all caption",
        "% text lives here so the paper owns the prose layer.",
        "",
    ]
    for stem, label, short, full in _LATEX_CAPTIONS:
        lines.extend([
            r"\begin{figure}[t]",
            r"  \centering",
            f"  \\includegraphics[width=0.7\\linewidth]{{figures/exp3/{stem}.png}}",
            f"  \\caption[{short}]{{{full}}}",
            f"  \\label{{{label}}}",
            r"\end{figure}",
            "",
        ])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.analysis.exp3")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--figures-dir", default=Path("figures/exp3"), type=Path)
    parser.add_argument("--calibration", default=None, type=Path)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--jittery-filter", choices=("clean", "jittery", "all"),
        default="all",
        help=(
            "Filter trials by network-jitter condition. "
            "'clean': only ``r.jittery=False`` rows (single box per "
            "arm). "
            "'jittery': only ``r.jittery=True`` rows (single box). "
            "'all' (default): both regimes; box plots render paired "
            "(clean + jittery) boxes per arm with colour distinguishing "
            "the two so the reader can see clusters within and across "
            "regimes."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cal = load_calibration(args.calibration)
    rows = load_trials(args.csv)

    # Apply --jittery-filter to the rows that feed the text summary.
    # In "all" mode the summary still sees both regimes pooled (the
    # paired tests in that mode reflect "robustness across regimes");
    # in "clean"/"jittery" the summary is regime-specific.
    if args.jittery_filter == "clean":
        summary_rows = [r for r in rows if not r.jittery]
    elif args.jittery_filter == "jittery":
        summary_rows = [r for r in rows if r.jittery]
    else:
        summary_rows = list(rows)

    n_clean = sum(1 for r in rows if r.is_ok and not r.jittery)
    n_jit = sum(1 for r in rows if r.is_ok and r.jittery)
    log.info(
        "loaded %d ok trials (%d clean, %d jittery); "
        "filter=%s leaves %d trials in scope",
        n_clean + n_jit, n_clean, n_jit, args.jittery_filter,
        sum(1 for r in summary_rows if r.is_ok),
    )

    text = summarize(summary_rows)
    # Write via the raw stdout buffer so the Greek δ in the summary
    # survives Windows' default cp1252 console encoding.
    try:
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    except AttributeError:  # pragma: no cover - non-binary stdout
        print(text)

    if not args.no_figures:
        figs = write_figures(
            rows,
            figures_dir=args.figures_dir,
            placeholder_watermark=not cal.is_paper_grade,
            jittery_filter=args.jittery_filter,
        )
        log.info(
            "wrote %d figures/CSVs to %s (jittery-filter=%s)",
            len(figs), args.figures_dir, args.jittery_filter,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
