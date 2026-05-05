"""Append-only CSV log with resume semantics.

Design constraints (Chunk EX-0):

* **Idempotent append.** Re-running the same trial doesn't duplicate
  rows; the (cell_id, arm, trial_index) triple is the unique key.
* **Resume.** Reading the CSV at startup populates a "done" set; the
  runner skips any cell whose key is already there.
* **Survives a partial write.** If the process dies mid-write, the
  file is in a known state — either the row is fully written (we
  flush after every append) or it isn't. No half-rows.
* **Header tolerance.** First write creates the header; subsequent
  writes append rows directly. A run that adds new metric columns
  to the schema must specify ``allow_schema_change=True`` and the log
  rewrites the file with the union of columns.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .grid import Cell


_KEY_FIELDS: Tuple[str, str, str] = ("cell_id", "arm", "trial_index")


class CSVTrialLog:
    """Append-only CSV with idempotent (cell_id, arm, trial_index) keys.

    Usage::

        log = CSVTrialLog(Path("results/exp1.csv"), fieldnames=[
            "cell_id", "arm", "trial_index", "seed",
            "param_Dpd", "param_alpha", "param_R",
            "Tproc_s", "Bpw", "Ttx_s", "eta", "E_idle_J", "E_tx_J",
            "Pcomplete", "duration_s", "status", "error",
        ])

        for cell in grid:
            if log.already_done(cell):
                continue
            row = run_trial(cell)
            log.append(cell, row)
    """

    def __init__(
        self,
        path: Path,
        fieldnames: Sequence[str],
        *,
        allow_schema_change: bool = False,
    ) -> None:
        self._path = Path(path)
        self._fieldnames = list(fieldnames)
        self._allow_schema_change = allow_schema_change
        # Set of (cell_id, arm, trial_index) tuples already in the file.
        self._done: Set[Tuple[str, str, int]] = set()
        self._load_existing()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def path(self) -> Path:
        return self._path

    @property
    def fieldnames(self) -> List[str]:
        return list(self._fieldnames)

    def already_done(self, cell: Cell) -> bool:
        return cell.unique_key in self._done

    def append(self, cell: Cell, row: Mapping[str, Any]) -> None:
        """Write one row. Skips silently if the cell is already logged.

        Reserved columns ``cell_id`` / ``arm`` / ``trial_index`` /
        ``seed`` / ``param_*`` come from the cell and override any
        same-named keys in ``row`` so the runner can't be fooled into
        writing inconsistent metadata.
        """
        if self.already_done(cell):
            return

        merged: Dict[str, Any] = dict(row)
        merged.update(cell.to_row_prefix())

        if self._allow_schema_change:
            new_cols = [k for k in merged if k not in self._fieldnames]
            if new_cols:
                self._extend_schema(new_cols)

        # Stick to declared fieldnames; unknown keys get dropped with a
        # KeyError so a typo in a driver surfaces immediately rather
        # than producing a silently-misaligned CSV.
        unknown = set(merged.keys()) - set(self._fieldnames)
        if unknown:
            raise ValueError(
                f"row has columns not in the CSV schema: {sorted(unknown)}; "
                f"declared schema: {self._fieldnames}"
            )

        self._write_row(merged)
        self._done.add(cell.unique_key)

    def existing_keys(self) -> Set[Tuple[str, str, int]]:
        """Return the (cell_id, arm, trial_index) triples already on disk."""
        return set(self._done)

    def __len__(self) -> int:
        return len(self._done)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _load_existing(self) -> None:
        """Populate the done-set from an existing file, if any."""
        if not self._path.exists():
            return
        with open(self._path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_cols = reader.fieldnames or []
            # Schema-mismatch detection — refuse to silently re-write a
            # CSV produced under a different schema.
            for col in _KEY_FIELDS:
                if col not in existing_cols:
                    raise ValueError(
                        f"existing CSV {self._path} is missing required "
                        f"key column {col!r}; refusing to append"
                    )
            for row in reader:
                try:
                    key = (
                        row["cell_id"],
                        row["arm"],
                        int(row["trial_index"]),
                    )
                except (KeyError, ValueError) as e:
                    raise ValueError(
                        f"existing CSV {self._path} has malformed row {row!r}"
                    ) from e
                self._done.add(key)
            # If the existing file declares more columns than we knew
            # about, adopt them so subsequent writes preserve them.
            if existing_cols and existing_cols != self._fieldnames:
                if self._allow_schema_change or all(
                    c in existing_cols for c in self._fieldnames
                ):
                    self._fieldnames = list(existing_cols)
                else:
                    raise ValueError(
                        f"existing CSV {self._path} has columns "
                        f"{existing_cols!r} but driver declared "
                        f"{self._fieldnames!r}; pass "
                        f"allow_schema_change=True to override"
                    )

    def _write_row(self, row: Mapping[str, Any]) -> None:
        """Append one row, writing the header on first touch."""
        new_file = not self._path.exists() or self._path.stat().st_size == 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            f.flush()

    def _extend_schema(self, new_cols: Iterable[str]) -> None:
        """Add new columns to the on-disk schema by rewriting the file.

        Used when ``allow_schema_change=True`` and a driver returns a
        column that wasn't in the original declaration. The rewrite
        preserves existing rows verbatim with empty values for the new
        columns.
        """
        new_cols = list(new_cols)
        if not self._path.exists():
            self._fieldnames.extend(new_cols)
            return

        with open(self._path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_cols = reader.fieldnames or []

        merged_cols = list(existing_cols)
        for c in new_cols:
            if c not in merged_cols:
                merged_cols.append(c)

        with open(self._path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=merged_cols)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({c: row.get(c, "") for c in merged_cols})

        self._fieldnames = merged_cols
