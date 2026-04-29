"""Bundle integrity signatures.

Phase 3 verifier uses a deterministic SHA-256 over the load-bearing
fields of a dock bundle. The ``bundle_sig`` field on ``UpBundle`` and
``DownBundle`` is populated on send (``sign_up_bundle`` /
``sign_down_bundle``) and checked on receive (``verify_bundle``).

The signature is not cryptographic authentication — it only catches
corruption / mismatch. Real HMAC with a shared dock secret lands with
the real transport in Phase 6.

Design ref: HERMES_FL_Scheduler_Design.md §5.4 (VERIFY stage).
"""

from __future__ import annotations

import hashlib

from .bundles import DownBundle, UpBundle
from .fl_messages import weights_signature


# --------------------------------------------------------------------------- #
# Signers
# --------------------------------------------------------------------------- #

def sign_up_bundle(bundle: UpBundle) -> str:
    """Compute and stamp ``bundle.bundle_sig``; also returns the signature."""
    sig = _compute_up_sig(bundle)
    bundle.bundle_sig = sig
    return sig


def sign_down_bundle(bundle: DownBundle) -> str:
    """Compute and stamp ``bundle.bundle_sig``; also returns the signature."""
    sig = _compute_down_sig(bundle)
    bundle.bundle_sig = sig
    return sig


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #

def verify_up_bundle(bundle: UpBundle) -> bool:
    """Return True iff ``bundle.bundle_sig`` matches the payload."""
    return bool(bundle.bundle_sig) and bundle.bundle_sig == _compute_up_sig(bundle)


def verify_down_bundle(bundle: DownBundle) -> bool:
    """Return True iff ``bundle.bundle_sig`` matches the payload."""
    return bool(bundle.bundle_sig) and bundle.bundle_sig == _compute_down_sig(bundle)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _compute_up_sig(b: UpBundle) -> str:
    h = hashlib.sha256()
    h.update(b.mule_id.encode("utf-8"))
    h.update(str(b.partial_aggregate.mission_round).encode("utf-8"))
    h.update(str(b.partial_aggregate.num_examples).encode("utf-8"))
    h.update(weights_signature(b.partial_aggregate.weights).encode("utf-8"))
    h.update(",".join(b.partial_aggregate.contributing_devices).encode("utf-8"))
    for line in b.round_close_report.lines:
        h.update(line.device_id.encode("utf-8"))
        h.update(line.outcome.value.encode("utf-8"))
    for rec in b.contact_history.records:
        h.update(rec.device_id.encode("utf-8"))
        h.update(str(rec.in_session).encode("utf-8"))
    return h.hexdigest()


def _compute_down_sig(b: DownBundle) -> str:
    h = hashlib.sha256()
    h.update(b.mule_id.encode("utf-8"))
    h.update(str(b.mission_slice.issued_round).encode("utf-8"))
    h.update(",".join(b.mission_slice.device_ids).encode("utf-8"))
    h.update(weights_signature(b.theta_disc).encode("utf-8"))
    h.update(weights_signature(b.synth_batch).encode("utf-8"))
    h.update(str(b.cluster_amendments.cluster_round).encode("utf-8"))
    for dev, ts in sorted(b.cluster_amendments.deadline_overrides.items()):
        h.update(f"{dev}={ts}".encode("utf-8"))
    return h.hexdigest()
