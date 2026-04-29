"""Phase 5 — RFPriorStore tests.

Enforces design §7 principle 5 — the scheduler may read L1 env state
but must never write. These tests codify the public read surface and
verify the writer is package-private (not in the public API).
"""

from __future__ import annotations

import threading

import pytest

from hermes.l1 import RFPrior, RFPriorStore
import hermes.l1 as l1_pkg


# --------------------------------------------------------------------------- #
# Read surface
# --------------------------------------------------------------------------- #

def test_snapshot_empty_when_no_priors():
    store = RFPriorStore()
    assert store.snapshot() == ()


def test_read_returns_none_when_missing():
    store = RFPriorStore()
    assert store.read(0) is None


def test_mean_snr_empty_is_zero():
    store = RFPriorStore()
    assert store.mean_snr_db() == 0.0


def test_record_then_read():
    store = RFPriorStore()
    p = RFPrior(band=1, last_good_snr_db=22.5, observed_at=100.0)
    store._record(p)  # package-private producer
    assert store.read(1) == p
    assert store.read(0) is None


def test_snapshot_returns_all_records():
    store = RFPriorStore()
    store._record(RFPrior(band=0, last_good_snr_db=10.0, observed_at=0.0))
    store._record(RFPrior(band=1, last_good_snr_db=20.0, observed_at=1.0))
    snap = store.snapshot()
    assert len(snap) == 2
    assert {p.band for p in snap} == {0, 1}


def test_mean_snr_averages_priors():
    store = RFPriorStore()
    store._record(RFPrior(band=0, last_good_snr_db=10.0, observed_at=0.0))
    store._record(RFPrior(band=1, last_good_snr_db=30.0, observed_at=1.0))
    assert store.mean_snr_db() == pytest.approx(20.0)


def test_record_overwrites_same_band():
    store = RFPriorStore()
    store._record(RFPrior(band=0, last_good_snr_db=5.0, observed_at=0.0))
    store._record(RFPrior(band=0, last_good_snr_db=25.0, observed_at=1.0))
    assert store.read(0).last_good_snr_db == 25.0
    assert len(store.snapshot()) == 1


# --------------------------------------------------------------------------- #
# Design §7 principle 5 — no public writer surface
# --------------------------------------------------------------------------- #

def test_no_public_mutator_on_store():
    """The scheduler side must have no way to *write* — only _record exists."""
    pub_methods = {
        name for name in dir(RFPriorStore)
        if not name.startswith("_") and callable(getattr(RFPriorStore, name))
    }
    assert "snapshot" in pub_methods
    assert "read" in pub_methods
    assert "mean_snr_db" in pub_methods
    # Any method whose name suggests a write must be package-private.
    for forbidden in ("record", "write", "set", "update", "push"):
        assert forbidden not in pub_methods, (
            f"Public writer '{forbidden}' breaks principle #5"
        )


def test_l1_public_api_does_not_expose_record():
    """hermes.l1 package must not re-export the private writer."""
    assert "RFPriorStore" in dir(l1_pkg)
    assert "RFPrior" in dir(l1_pkg)
    # _record is an instance method, not a package symbol — but confirm
    # no convenience writer leaks at the package level either.
    assert not hasattr(l1_pkg, "record_rf_prior")
    assert not hasattr(l1_pkg, "write_rf_prior")


# --------------------------------------------------------------------------- #
# Thread-safety smoke
# --------------------------------------------------------------------------- #

def test_concurrent_record_and_read_do_not_race():
    store = RFPriorStore()
    errors = []

    def writer(band: int):
        try:
            for t in range(200):
                store._record(
                    RFPrior(band=band, last_good_snr_db=float(t), observed_at=float(t))
                )
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(200):
                store.snapshot()
                store.mean_snr_db()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(0,)),
        threading.Thread(target=writer, args=(1,)),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    # Both bands should have landed.
    assert store.read(0) is not None
    assert store.read(1) is not None
