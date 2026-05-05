"""Trial-grid generator with paired-seed logic.

Design constraints (from the experiments plan, Chunk EX-0):

* **Cartesian product** over independent variables, expanded by arms ×
  trial indices.
* **Paired seeds across arms.** For a given (cell_id, trial_index), every
  arm sees the *same* seed. This is what lets the analysis use paired
  Wilcoxon — a per-trial difference (FL vs Centralized at trial 7) is
  meaningful only if both arms saw the same network trace + dataset
  shard.
* **Deterministic cell_id.** Stable across reorderings of the
  independent-variable dict so a partial CSV from yesterday matches
  today's grid.

Shapes:

  experiments_grid = TrialGrid(
      independent_vars={
          "Dpd":   ["10MB", "100MB", "1GB"],
          "alpha": [0.5, 1.0, 2.0],
          "R":     [5, 20, 50],
      },
      arms=["FL", "Centralized"],
      n_trials=20,
      base_seed=42,
  )
  for cell in experiments_grid:
      run_trial(cell)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class Cell:
    """One trial slot — one (independent-var assignment, arm, trial_index).

    ``cell_id`` is the stable hash of ``params``; ``arm`` and
    ``trial_index`` make the full unique key. ``seed`` is derived from
    ``(cell_id, trial_index)`` so it's identical across arms within a
    cell — paired-seed semantics.
    """

    cell_id: str
    arm: str
    trial_index: int
    seed: int
    params: Mapping[str, Any]

    @property
    def unique_key(self) -> Tuple[str, str, int]:
        """The triple that the CSV log uses to skip-if-done."""
        return (self.cell_id, self.arm, self.trial_index)

    def to_row_prefix(self) -> Dict[str, Any]:
        """Columns the runner stamps onto every CSV row.

        Drivers return their per-experiment metric columns; the runner
        merges those with this prefix so every row is self-describing.
        """
        out: Dict[str, Any] = {
            "cell_id": self.cell_id,
            "arm": self.arm,
            "trial_index": self.trial_index,
            "seed": self.seed,
        }
        # Independent-variable columns inline so a row can be filtered
        # by Dpd / alpha / R / etc. directly without having to re-parse
        # cell_id.
        for k, v in self.params.items():
            out[f"param_{k}"] = v
        return out


@dataclass
class TrialGrid:
    """Cartesian product of independent variables × arms × trials."""

    independent_vars: Mapping[str, Sequence[Any]]
    arms: Sequence[str]
    n_trials: int = 20
    base_seed: int = 42
    # If set, only emit cells whose params dict matches this filter.
    # Useful for a "rerun only the {alpha=0.5} cells" workflow.
    filter_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be ≥ 1, got {self.n_trials}")
        if not self.arms:
            raise ValueError("arms must contain at least one entry")
        if not self.independent_vars:
            # Permitted — degenerate "single cell" grid useful for tests
            # that only sweep over arms × trials.
            pass
        for k, vs in self.independent_vars.items():
            if not vs:
                raise ValueError(
                    f"independent_var {k!r} has empty value list"
                )

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[Cell]:
        return self.cells()

    def cells(self) -> Iterator[Cell]:
        """Yield every Cell in the grid in deterministic order.

        Order: independent-vars cartesian product (in dict-insertion
        order), then arms, then trial_index. Stable across runs so a
        Ctrl-C resume picks up where it left off.
        """
        for params in self._param_combinations():
            cid = self._cell_id(params)
            for trial_index in range(self.n_trials):
                trial_seed = self._derive_seed(cid, trial_index)
                for arm in self.arms:
                    yield Cell(
                        cell_id=cid,
                        arm=arm,
                        trial_index=trial_index,
                        seed=trial_seed,
                        params=dict(params),
                    )

    def total(self) -> int:
        """Total number of cells the iteration will emit."""
        n_combos = 1
        for vs in self.independent_vars.values():
            n_combos *= len(vs)
        return n_combos * self.n_trials * len(self.arms)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _param_combinations(self) -> Iterator[Dict[str, Any]]:
        keys = list(self.independent_vars.keys())
        if not keys:
            yield {}
            return
        for combo in product(*(self.independent_vars[k] for k in keys)):
            params = dict(zip(keys, combo))
            if not self._matches_filter(params):
                continue
            yield params

    def _matches_filter(self, params: Mapping[str, Any]) -> bool:
        for k, v in self.filter_params.items():
            if k not in params or params[k] != v:
                return False
        return True

    def _cell_id(self, params: Mapping[str, Any]) -> str:
        """Stable, human-readable, hash-friendly cell ID.

        Uses a `key=value|key=value` form sorted by key so the same
        params dict always produces the same ID regardless of insertion
        order. Short enough for log lines, unique enough for the CSV
        key.
        """
        if not params:
            return "_singleton_"
        return "|".join(f"{k}={params[k]}" for k in sorted(params.keys()))

    def _derive_seed(self, cell_id: str, trial_index: int) -> int:
        """Per-(cell, trial) seed shared across arms."""
        digest = hashlib.sha256(
            f"{self.base_seed}|{cell_id}|{trial_index}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:4], "big")
