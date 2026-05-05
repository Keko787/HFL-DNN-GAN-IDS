"""Experiment-1 analysis — load CSV → stats + paper figures.

Reads the per-trial CSV produced by :mod:`experiments.exp1.server`,
applies the calibration TOML, computes the paper's stats (paired
Wilcoxon + Cliff's δ on Bpw / Ttx, bootstrap CI on R*), and emits
the five paper figures + a one-page text summary.

Runnable directly::

    python -m experiments.analysis.exp1 \\
        --csv results/exp1.csv \\
        --figures-dir figures/exp1/

Or imported into a notebook for interactive exploration::

    from experiments.analysis.exp1 import load_trials, summarize
    df = load_trials("results/exp1.csv")
    print(summarize(df))
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from experiments.calibration import (
    Exp1Calibration,
    exp1_energy_proxy,
    load_calibration,
)

from .stats import (
    CrossoverEstimate,
    PairedTestResult,
    bootstrap_R_star_ci,
    paired_wilcoxon_with_cliffs_delta,
    solve_crossover_round,
)

log = logging.getLogger("experiments.analysis.exp1")


# --------------------------------------------------------------------------- #
# CSV loading — stdlib csv (no pandas dependency)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrialRow:
    """One row of the Experiment-1 CSV after type coercion."""

    cell_id: str
    arm: str
    trial_index: int
    seed: int
    Dpd: str
    alpha: float
    R: int
    Tproc_s: float
    Ttx_s: float
    Bpw_bytes: float
    eta: float
    deadline_s: float
    Pcomplete: int
    n_rounds: int
    n_clients: int
    status: str

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"


def load_trials(csv_path: Path) -> List[TrialRow]:
    """Read the experiment CSV; skip non-ok rows by default.

    Raises if the CSV doesn't have the expected columns — surfaces a
    schema drift between the experiment server and the analysis sooner
    rather than later.
    """
    csv_path = Path(csv_path)
    rows: List[TrialRow] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "cell_id", "arm", "trial_index", "seed",
            "param_Dpd", "param_alpha", "param_R",
            "Tproc_s", "Ttx_s", "Bpw_bytes", "eta",
            "deadline_s", "Pcomplete", "n_rounds", "n_clients",
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
                row = TrialRow(
                    cell_id=raw["cell_id"],
                    arm=raw["arm"],
                    trial_index=int(raw["trial_index"]),
                    seed=int(raw["seed"]),
                    Dpd=raw["param_Dpd"],
                    alpha=float(raw["param_alpha"]),
                    R=int(raw["param_R"]),
                    Tproc_s=float(raw["Tproc_s"]) if raw["Tproc_s"] else 0.0,
                    Ttx_s=float(raw["Ttx_s"]) if raw["Ttx_s"] else 0.0,
                    Bpw_bytes=float(raw["Bpw_bytes"]) if raw["Bpw_bytes"] else 0.0,
                    eta=float(raw["eta"]) if raw["eta"] else 0.0,
                    deadline_s=float(raw["deadline_s"]) if raw["deadline_s"] else 0.0,
                    Pcomplete=int(raw["Pcomplete"]) if raw["Pcomplete"] else 0,
                    n_rounds=int(raw["n_rounds"]) if raw["n_rounds"] else 0,
                    n_clients=int(raw["n_clients"]) if raw["n_clients"] else 0,
                    status=raw["status"],
                )
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"failed to parse row {raw!r}: {e}"
                ) from e
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Per-cell paired tests
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CellPairedTests:
    """Paired Wilcoxon + Cliff's δ for one ``(Dpd, alpha, R)`` cell."""

    Dpd: str
    alpha: float
    R: int
    n_pairs: int
    Bpw_test: PairedTestResult
    Ttx_test: PairedTestResult


def _arm_filter(rows: Sequence[TrialRow], arm: str) -> List[TrialRow]:
    return [r for r in rows if r.is_ok and r.arm == arm]


def _key(r: TrialRow) -> tuple:
    return (r.Dpd, r.alpha, r.R, r.trial_index)


def per_cell_paired_tests(rows: Sequence[TrialRow]) -> List[CellPairedTests]:
    """One :class:`CellPairedTests` per unique ``(Dpd, alpha, R)`` cell.

    Pair FL[i] with Centralized[i] within each cell on
    ``trial_index``; that's the paired-seed contract.
    """
    fl_by_key = {_key(r): r for r in _arm_filter(rows, "FL")}
    cent_by_key = {_key(r): r for r in _arm_filter(rows, "Centralized")}

    results: List[CellPairedTests] = []
    cell_groups: Dict[tuple, List[tuple]] = {}
    for k in fl_by_key:
        cell = (k[0], k[1], k[2])
        cell_groups.setdefault(cell, []).append(k)

    for (Dpd, alpha, R), keys in sorted(cell_groups.items()):
        paired = [(fl_by_key[k], cent_by_key[k]) for k in keys if k in cent_by_key]
        if len(paired) < 2:
            continue  # not enough pairs for a Wilcoxon

        bpw_fl = [p[0].Bpw_bytes for p in paired]
        bpw_cent = [p[1].Bpw_bytes for p in paired]
        ttx_fl = [p[0].Ttx_s for p in paired]
        ttx_cent = [p[1].Ttx_s for p in paired]

        results.append(CellPairedTests(
            Dpd=Dpd, alpha=alpha, R=R,
            n_pairs=len(paired),
            Bpw_test=paired_wilcoxon_with_cliffs_delta(bpw_fl, bpw_cent),
            Ttx_test=paired_wilcoxon_with_cliffs_delta(ttx_fl, ttx_cent),
        ))
    return results


# --------------------------------------------------------------------------- #
# R* recovery — one estimate per (Dpd, alpha) over the R sweep
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CellCrossover:
    Dpd: str
    alpha: float
    R_star: Optional[float]
    R_star_lo: Optional[float]
    R_star_hi: Optional[float]
    cent_baseline_s: float


def per_cell_crossovers(
    rows: Sequence[TrialRow],
    *,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> List[CellCrossover]:
    """One CrossoverEstimate per (Dpd, alpha) over the R sweep."""
    fl = _arm_filter(rows, "FL")
    cent = _arm_filter(rows, "Centralized")

    out: List[CellCrossover] = []
    da_keys = sorted({(r.Dpd, r.alpha) for r in fl})
    for Dpd, alpha in da_keys:
        fl_cell = [r for r in fl if r.Dpd == Dpd and r.alpha == alpha]
        cent_cell = [r for r in cent if r.Dpd == Dpd and r.alpha == alpha]
        if not fl_cell or not cent_cell:
            continue
        R_vals = [r.R for r in fl_cell]
        T_vals = [r.Tproc_s for r in fl_cell]
        cent_T = [r.Tproc_s for r in cent_cell]
        point, lo, hi = bootstrap_R_star_ci(
            R_vals, T_vals, cent_T,
            n_bootstraps=n_bootstraps, seed=seed,
        )
        out.append(CellCrossover(
            Dpd=Dpd, alpha=alpha,
            R_star=point, R_star_lo=lo, R_star_hi=hi,
            cent_baseline_s=float(np.mean(cent_T)),
        ))
    return out


# --------------------------------------------------------------------------- #
# Energy decomposition (per row)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EnergyRow:
    cell_id: str
    arm: str
    trial_index: int
    Dpd: str
    alpha: float
    R: int
    idle_J: float
    tx_J: float
    total_J: float


def per_row_energy(
    rows: Sequence[TrialRow], cal: Exp1Calibration,
) -> List[EnergyRow]:
    out: List[EnergyRow] = []
    for r in rows:
        if not r.is_ok:
            continue
        e = exp1_energy_proxy(
            T_proc_s=r.Tproc_s, B_pw_bytes=r.Bpw_bytes, cal=cal,
        )
        out.append(EnergyRow(
            cell_id=r.cell_id, arm=r.arm, trial_index=r.trial_index,
            Dpd=r.Dpd, alpha=r.alpha, R=r.R,
            idle_J=e.idle_J, tx_J=e.tx_J, total_J=e.total_J,
        ))
    return out


# --------------------------------------------------------------------------- #
# Text summary
# --------------------------------------------------------------------------- #

def summarize(rows: Sequence[TrialRow]) -> str:
    """Human-readable one-page summary; what the paper figure-caption draft pulls from."""
    if not rows:
        return "(no rows)"

    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.is_ok)
    n_err = sum(1 for r in rows if r.status == "error")
    n_to = sum(1 for r in rows if r.status == "timeout")
    cells = {(r.Dpd, r.alpha, r.R) for r in rows}

    lines = [
        f"Experiment 1 — summary (n={n_total} trials)",
        f"  ok={n_ok}  error={n_err}  timeout={n_to}  cells={len(cells)}",
        "",
    ]

    cross = per_cell_crossovers(rows)
    if cross:
        lines.append("R* (cumulative crossover round count, paired bootstrap):")
        for c in cross:
            if c.R_star is None:
                lines.append(
                    f"  Dpd={c.Dpd:>6}  alpha={c.alpha:.2f}  "
                    f"R* = (degenerate; FL dominates or insufficient R sweep)"
                )
                continue
            ci = (
                f"[{c.R_star_lo:.1f}, {c.R_star_hi:.1f}]"
                if c.R_star_lo is not None and c.R_star_hi is not None
                else "(no CI)"
            )
            lines.append(
                f"  Dpd={c.Dpd:>6}  alpha={c.alpha:.2f}  "
                f"R* = {c.R_star:.1f}  95% CI {ci}"
            )

    paired = per_cell_paired_tests(rows)
    if paired:
        lines.append("")
        lines.append("Bpw effect size (Cliff's δ, FL vs Centralized):")
        seen_dpd: set = set()
        for cell in paired:
            if cell.Dpd in seen_dpd:
                continue
            seen_dpd.add(cell.Dpd)
            lines.append(
                f"  Dpd={cell.Dpd:>6}  δ = {cell.Bpw_test.cliffs_delta:+.3f}  "
                f"({cell.Bpw_test.delta_magnitude})"
            )

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figure generation (matplotlib) — kept optional
# --------------------------------------------------------------------------- #

def write_figures(
    rows: Sequence[TrialRow],
    cal: Exp1Calibration,
    *,
    figures_dir: Path,
    placeholder_watermark: bool = False,
) -> List[Path]:
    """Emit the five paper figures to ``figures_dir``.

    Returns the list of file paths produced. Skips any figure for
    which the input data is degenerate (e.g., R* heatmap with only
    one R cell) and logs a warning rather than erroring.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
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

    # 1. η heatmap over (|D|pd, R) with the η · Bpw = |D|pd contour.
    try:
        fl = [r for r in rows if r.is_ok and r.arm == "FL"]
        Dpd_order = sorted({r.Dpd for r in fl}, key=lambda s: _parse_size(s))
        R_order = sorted({r.R for r in fl})
        if len(Dpd_order) >= 2 and len(R_order) >= 2:
            mat = np.full((len(Dpd_order), len(R_order)), np.nan)
            for i, d in enumerate(Dpd_order):
                for j, R in enumerate(R_order):
                    cell_rows = [r for r in fl if r.Dpd == d and r.R == R]
                    if cell_rows:
                        mat[i, j] = np.mean([r.eta for r in cell_rows])
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(R_order)))
            ax.set_xticklabels([str(R) for R in R_order])
            ax.set_yticks(range(len(Dpd_order)))
            ax.set_yticklabels(Dpd_order)
            ax.set_xlabel("R (number of FL rounds)")
            ax.set_ylabel("|D|_pd")
            ax.set_title("Throughput efficiency η (FL arm)")
            fig.colorbar(im, ax=ax)
            _watermark(ax)
            out = figures_dir / "exp1_eta_heatmap.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("eta heatmap skipped: %s", e)

    # 2. R* regression — one curve per (Dpd, alpha) cell.
    try:
        fl = [r for r in rows if r.is_ok and r.arm == "FL"]
        cent = [r for r in rows if r.is_ok and r.arm == "Centralized"]
        cells = sorted({(r.Dpd, r.alpha) for r in fl})
        if cells:
            fig, ax = plt.subplots(figsize=(7, 4))
            for Dpd, alpha in cells:
                fl_cell = [r for r in fl if r.Dpd == Dpd and r.alpha == alpha]
                if not fl_cell:
                    continue
                R_vals = sorted({r.R for r in fl_cell})
                means = [
                    np.mean([r.Tproc_s for r in fl_cell if r.R == R])
                    for R in R_vals
                ]
                ax.plot(R_vals, means, marker="o",
                        label=f"Dpd={Dpd}, α={alpha}")
            cent_mean = np.mean([r.Tproc_s for r in cent]) if cent else np.nan
            if not np.isnan(cent_mean):
                ax.axhline(cent_mean, linestyle="--", color="black",
                           label=f"Centralized baseline ({cent_mean:.2f}s)")
            ax.set_xlabel("R (FL rounds)")
            ax.set_ylabel("T_proc (s)")
            ax.set_title("R* regression — FL T_proc vs R")
            ax.legend(fontsize=8)
            _watermark(ax)
            out = figures_dir / "exp1_Rstar_regression.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("R* regression plot skipped: %s", e)

    # 3. Energy stacked bar (idle vs tx) at α=1.0.
    try:
        e_rows = per_row_energy([r for r in rows if r.alpha == 1.0], cal)
        if e_rows:
            arms = sorted({e.arm for e in e_rows})
            Dpd_order = sorted({e.Dpd for e in e_rows}, key=lambda s: _parse_size(s))
            idle = np.zeros((len(arms), len(Dpd_order)))
            tx = np.zeros_like(idle)
            for i, arm in enumerate(arms):
                for j, d in enumerate(Dpd_order):
                    cell = [e for e in e_rows if e.arm == arm and e.Dpd == d]
                    if cell:
                        idle[i, j] = np.mean([c.idle_J for c in cell])
                        tx[i, j] = np.mean([c.tx_J for c in cell])
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(Dpd_order))
            width = 0.35
            for i, arm in enumerate(arms):
                offset = (i - 0.5) * width
                ax.bar(x + offset, idle[i], width, label=f"{arm} idle")
                ax.bar(x + offset, tx[i], width, bottom=idle[i],
                       label=f"{arm} tx")
            ax.set_xticks(x)
            ax.set_xticklabels(Dpd_order)
            ax.set_ylabel("Energy (J)")
            ax.set_title("Energy proxy at α=1.0 (stacked: idle / tx)")
            ax.legend(fontsize=8)
            _watermark(ax)
            out = figures_dir / "exp1_energy_stacked_bar.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("energy bar skipped: %s", e)

    # 4. Pcomplete two-panel heatmap (FL + Centralized) over (alpha × Dpd).
    try:
        Dpd_order = sorted({r.Dpd for r in rows if r.is_ok}, key=lambda s: _parse_size(s))
        alpha_order = sorted({r.alpha for r in rows if r.is_ok})
        if len(Dpd_order) >= 1 and len(alpha_order) >= 1:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
            for ax, arm in zip(axes, ("FL", "Centralized")):
                arm_rows = [r for r in rows if r.is_ok and r.arm == arm]
                mat = np.full((len(alpha_order), len(Dpd_order)), np.nan)
                for i, a in enumerate(alpha_order):
                    for j, d in enumerate(Dpd_order):
                        cell = [r for r in arm_rows if r.alpha == a and r.Dpd == d]
                        if cell:
                            mat[i, j] = np.mean([r.Pcomplete for r in cell])
                im = ax.imshow(mat, aspect="auto", origin="lower",
                               cmap="RdYlGn", vmin=0.0, vmax=1.0)
                ax.set_xticks(range(len(Dpd_order)))
                ax.set_xticklabels(Dpd_order)
                ax.set_yticks(range(len(alpha_order)))
                ax.set_yticklabels([f"{a:.1f}" for a in alpha_order])
                ax.set_xlabel("|D|_pd")
                ax.set_title(arm)
                _watermark(ax)
            axes[0].set_ylabel("α (deadline multiplier)")
            fig.suptitle("P_complete heatmap")
            fig.colorbar(im, ax=axes, fraction=0.04)
            out = figures_dir / "exp1_Pcomplete_heatmap.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("Pcomplete heatmap skipped: %s", e)

    # 5. Bpw / Ttx significance table — saved as a CSV alongside figures.
    try:
        paired = per_cell_paired_tests(rows)
        if paired:
            sig_path = figures_dir / "exp1_paired_tests.csv"
            with open(sig_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "Dpd", "alpha", "R", "n_pairs",
                    "Bpw_W", "Bpw_p", "Bpw_cliffs_delta", "Bpw_delta_mag",
                    "Ttx_W", "Ttx_p", "Ttx_cliffs_delta", "Ttx_delta_mag",
                ])
                for c in paired:
                    w.writerow([
                        c.Dpd, c.alpha, c.R, c.n_pairs,
                        c.Bpw_test.statistic, c.Bpw_test.p_value,
                        c.Bpw_test.cliffs_delta, c.Bpw_test.delta_magnitude,
                        c.Ttx_test.statistic, c.Ttx_test.p_value,
                        c.Ttx_test.cliffs_delta, c.Ttx_test.delta_magnitude,
                    ])
            written.append(sig_path)
    except Exception as e:  # pragma: no cover
        log.warning("paired-tests CSV skipped: %s", e)

    return written


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse_size(s: str) -> int:
    s = s.strip().upper()
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    for suf in ("GB", "MB", "KB", "B"):
        if s.endswith(suf):
            return int(float(s[: -len(suf)]) * units[suf])
    return int(s)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.analysis.exp1")
    parser.add_argument("--csv", required=True, type=Path,
                        help="Per-trial CSV from experiments.exp1.server.")
    parser.add_argument("--figures-dir", default=Path("figures/exp1"), type=Path,
                        help="Output directory for paper figures.")
    parser.add_argument("--calibration", default=None, type=Path,
                        help="Override path to calibration.toml.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation; print summary only.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cal = load_calibration(args.calibration)
    rows = load_trials(args.csv)
    print(summarize(rows))

    if not args.no_figures:
        figs = write_figures(
            rows, cal.exp1,
            figures_dir=args.figures_dir,
            placeholder_watermark=not cal.is_paper_grade,
        )
        log.info("wrote %d figures/CSVs to %s", len(figs), args.figures_dir)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
