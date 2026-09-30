"""Read-only RF-prior store.

Design §2.1: the scheduler's S3.5 selector may read L1's env state as a
read-only feature — it must never write. :class:`RFPriorStore` exposes
``snapshot()`` / ``read(band)`` only; writes come from the L1 module
itself via a package-private ``_record`` method.

The payload is tiny on purpose — just a per-band "last-good SNR" buffer
with a timestamp, averaged into a single scalar when the selector asks.
That matches the design §6.4 ``rf_prior_snr`` feature.

FeRRy Phase 3 (critic B4) adds the store's first production writer,
:class:`RFPriorProducer`. It sits in this module because ``_record`` is
L1-internal. In ferry mode it feeds the store with the SNR observed at each
backhaul upload, which gives a causal replacement for the driver's
``rf_prior_snr_db``: under ``--l1-channel`` that value is the chosen band's
mean SNR over the whole trial, so it uses the future. The store's read API is
unchanged.
"""

from __future__ import annotations

import math
import operator
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


#: The RF prior before any observation: ``MuleSupervisor``'s default
#: ``rf_prior_snr_db`` (``hermes/mule/mule_main.py``), which it uses when
#: ``MuleConfig.rf_prior_snr_db`` is None.
DEFAULT_RF_PRIOR_SNR_DB = 20.0


class RFPriorProducer:
    """The causal producer that feeds an :class:`RFPriorStore` (FeRRy Phase 3, critic B4).

    It replaces the non-causal trial constant. Under ``--l1-channel`` the
    driver gives the mule ``rf_prior_snr_db = backhaul_plan(...).mean_chosen_snr_db``,
    the chosen band's mean SNR over every mission of the trial, the later ones
    included. In ferry mode the prior is built only from what the mule has
    already seen:

    * The mule calls :meth:`observe_upload` after each backhaul upload, with the
      carrier used, the SNR observed on it and the simulated upload time.
    * :meth:`prior_snr_db` returns 20 dB (:data:`DEFAULT_RF_PRIOR_SNR_DB`)
      before the first upload. Afterwards it returns the store's
      ``mean_snr_db()``, or one carrier's last observation when asked for it.

    Every upload is recorded, lost or not. The mule measures the SNR before
    the cluster decides whether the upload is lost, so ``last_good_snr_db``
    (the store's field name) holds the last *observed* SNR of a carrier.

    This class writes through the store's L1-internal ``_record``. The
    scheduler side keeps only the read API (``snapshot``, ``read`` and
    ``mean_snr_db``), unchanged, so the design's rule that the scheduler never
    writes L1 state (§7 principle 5) still holds. The producer is meant for
    the mule's own thread; the store is thread-safe for readers.
    """

    def __init__(self, store: Optional[RFPriorStore] = None, *,
                 default_snr_db: float = DEFAULT_RF_PRIOR_SNR_DB) -> None:
        self._store = store if store is not None else RFPriorStore()
        self._default = float(default_snr_db)
        if not math.isfinite(self._default):
            raise ValueError(f"default_snr_db must be finite, got {default_snr_db!r}")

    @property
    def store(self) -> RFPriorStore:
        """The store; hand this (read-only API) to the scheduler side."""
        return self._store

    @property
    def default_snr_db(self) -> float:
        return self._default

    def observe_upload(self, carrier: int, snr_db: float, observed_at: float) -> RFPrior:
        """Record the SNR observed on ``carrier`` at an upload, and return the record."""
        if isinstance(carrier, bool):
            raise TypeError(f"carrier must be an integer, got {carrier!r}")
        band = operator.index(carrier)
        if band < 0:
            raise ValueError(f"carrier must be >= 0, got {carrier!r}")
        snr = float(snr_db)
        ts = float(observed_at)
        if not (math.isfinite(snr) and math.isfinite(ts)):
            raise ValueError(
                f"snr_db and observed_at must be finite, got {snr_db!r}, {observed_at!r}")
        prior = RFPrior(band=band, last_good_snr_db=snr, observed_at=ts)
        self._store._record(prior)
        return prior

    def prior_snr_db(self, carrier: Optional[int] = None) -> float:
        """The causal RF prior in dB.

        With ``carrier=None`` this is the mean over the carriers observed so
        far (``RFPriorStore.mean_snr_db``). With a carrier it is that carrier's
        last observation. Either way it is ``default_snr_db`` (20 dB) until
        the first matching upload.
        """
        if carrier is None:
            if not self._store.snapshot():
                return self._default
            return self._store.mean_snr_db()
        prior = self._store.read(operator.index(carrier))
        return self._default if prior is None else prior.last_good_snr_db
