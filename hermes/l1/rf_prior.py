"""Read-only RF-prior store.

Design §2.1: the scheduler's S3.5 selector may read L1's env state as a
read-only feature — it must never write. :class:`RFPriorStore` exposes
``snapshot()`` / ``read(band)`` only; writes come from the L1 module
itself via a package-private ``_record`` method.

The payload is tiny on purpose — just a per-band "last-good SNR" buffer
with a timestamp, averaged into a single scalar when the selector asks.
That matches the design §6.4 ``rf_prior_snr`` feature.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class RFPrior:
    """One per-band snapshot."""

    band: int
    last_good_snr_db: float
    observed_at: float


class RFPriorStore:
    """Thread-safe store of the most recent per-band SNR observations.

    The scheduler only calls :meth:`snapshot` and :meth:`read`.
    Producers inside the L1 package call :meth:`_record` to post a new
    observation — there's no public setter because design §7 principle 5
    forbids the scheduler writing to L1's env state.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._priors: Dict[int, RFPrior] = {}

    # ------------------------------------------------------------------ #
    # Reads (scheduler-facing)
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Tuple[RFPrior, ...]:
        """All known priors in insertion order — safe to hand to the selector."""
        with self._lock:
            return tuple(self._priors.values())

    def read(self, band: int) -> Optional[RFPrior]:
        with self._lock:
            return self._priors.get(band)

    def mean_snr_db(self) -> float:
        """Average of known priors — handy default for ``rf_prior_snr_db``."""
        with self._lock:
            if not self._priors:
                return 0.0
            return sum(p.last_good_snr_db for p in self._priors.values()) / len(
                self._priors
            )

    # ------------------------------------------------------------------ #
    # Writes (L1-internal — do not call from outside this package)
    # ------------------------------------------------------------------ #

    def _record(self, prior: RFPrior) -> None:
        with self._lock:
            self._priors[prior.band] = prior
