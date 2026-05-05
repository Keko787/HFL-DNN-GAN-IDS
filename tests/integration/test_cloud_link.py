"""Sprint 2 — outbound cloud link end-to-end (real HTTP loopback).

Spins up :class:`MockTier3Server` and runs :class:`HTTPCloudLink` against
it over real HTTP on localhost. Verifies:

* POST /partials accepts a pickled :class:`ClusterPartialUpload`.
* GET /refinement/{cluster_id} returns 204 when nothing's queued.
* GET /refinement/{cluster_id} returns the pickled refinement when one's queued.
* Errors propagate as :class:`CloudLinkError`.

The AERPAW no-inbound rule is structural: the cluster only initiates;
Tier-3 only responds. Both methods on :class:`CloudLink` are outbound.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import pytest

from hermes.transport import (
    CloudLinkError,
    HTTPCloudLink,
    MockTier3Server,
)
from hermes.types import ClusterPartialUpload, GeneratorRefinement


def _partial(cluster_id: str = "cluster-A", round_: int = 1) -> ClusterPartialUpload:
    return ClusterPartialUpload(
        cluster_id=cluster_id,
        cluster_round=round_,
        theta_disc=[np.zeros((4,), dtype=np.float32)],
        n_devices_contributing=3,
    )


def _refinement(round_: int = 1) -> GeneratorRefinement:
    return GeneratorRefinement(
        refinement_round=round_,
        theta_gen=[np.full((8,), 7.0, dtype=np.float32)],
        notes="cross-cluster sync",
    )


# --------------------------------------------------------------------------- #
# Mock server lifecycle
# --------------------------------------------------------------------------- #

def test_mock_tier3_server_start_stop():
    srv = MockTier3Server()
    srv.start()
    assert srv.port > 0
    assert srv.url.startswith("http://127.0.0.1:")
    srv.stop()


# --------------------------------------------------------------------------- #
# send_partial round-trip
# --------------------------------------------------------------------------- #

def test_send_partial_lands_in_received_partials():
    srv = MockTier3Server()
    srv.start()
    try:
        link = HTTPCloudLink(base_url=srv.url)
        link.send_partial(_partial())
        # MockTier3Server records the partial.
        rcv = srv.received_partials
        assert len(rcv) == 1
        assert rcv[0].cluster_id == "cluster-A"
        assert rcv[0].cluster_round == 1
        np.testing.assert_array_equal(
            rcv[0].theta_disc[0], np.zeros((4,), dtype=np.float32)
        )
    finally:
        srv.stop()


def test_send_partial_multiple_appends():
    srv = MockTier3Server()
    srv.start()
    try:
        link = HTTPCloudLink(base_url=srv.url)
        link.send_partial(_partial(round_=1))
        link.send_partial(_partial(round_=2))
        link.send_partial(_partial(round_=3))
        rounds = [p.cluster_round for p in srv.received_partials]
        assert rounds == [1, 2, 3]
    finally:
        srv.stop()


# --------------------------------------------------------------------------- #
# poll_refinement
# --------------------------------------------------------------------------- #

def test_poll_refinement_returns_none_when_nothing_queued():
    srv = MockTier3Server()
    srv.start()
    try:
        link = HTTPCloudLink(base_url=srv.url)
        out = link.poll_refinement(cluster_id="cluster-A")
        assert out is None
    finally:
        srv.stop()


def test_poll_refinement_returns_queued_then_clears():
    srv = MockTier3Server()
    srv.start()
    try:
        link = HTTPCloudLink(base_url=srv.url)
        srv.queue_refinement(_refinement(round_=42))
        out1 = link.poll_refinement(cluster_id="cluster-A")
        assert out1 is not None
        assert out1.refinement_round == 42
        np.testing.assert_array_equal(
            out1.theta_gen[0], np.full((8,), 7.0, dtype=np.float32)
        )
        # Second poll: slot is empty → 204.
        out2 = link.poll_refinement(cluster_id="cluster-A")
        assert out2 is None
    finally:
        srv.stop()


def test_poll_refinement_per_cluster_url_path():
    """The cluster_id portion of the URL is preserved (mock ignores it,
    but the Tier-3 production server uses it for routing)."""
    srv = MockTier3Server()
    srv.start()
    try:
        link = HTTPCloudLink(base_url=srv.url)
        # No queued refinement for any cluster_id — both should 204 cleanly.
        assert link.poll_refinement(cluster_id="A") is None
        assert link.poll_refinement(cluster_id="B") is None
    finally:
        srv.stop()


def test_s2_m5_queue_refinement_fifo_drain():
    """S2-M5: multiple queue_refinement calls stack and drain in order."""
    srv = MockTier3Server()
    srv.start()
    try:
        link = HTTPCloudLink(base_url=srv.url)
        srv.queue_refinement(_refinement(round_=1))
        srv.queue_refinement(_refinement(round_=2))
        srv.queue_refinement(_refinement(round_=3))
        assert srv.pending_refinements == 3

        rounds = []
        for _ in range(3):
            r = link.poll_refinement(cluster_id="A")
            assert r is not None
            rounds.append(r.refinement_round)
        # FIFO: oldest-staged returned first.
        assert rounds == [1, 2, 3]

        # Drained: subsequent poll → 204.
        assert link.poll_refinement(cluster_id="A") is None
        assert srv.pending_refinements == 0
    finally:
        srv.stop()


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #

def test_send_partial_against_dead_server_raises():
    """Server isn't running → CloudLinkError instead of urllib bubble-up."""
    link = HTTPCloudLink(
        base_url="http://127.0.0.1:1",  # almost certainly not listening
        request_timeout_s=0.5,
    )
    with pytest.raises(CloudLinkError):
        link.send_partial(_partial())


def test_poll_refinement_against_dead_server_raises():
    link = HTTPCloudLink(
        base_url="http://127.0.0.1:1",
        request_timeout_s=0.5,
    )
    with pytest.raises(CloudLinkError):
        link.poll_refinement(cluster_id="A", timeout_s=0.5)


def test_close_then_use_raises():
    srv = MockTier3Server()
    srv.start()
    try:
        link = HTTPCloudLink(base_url=srv.url)
        link.close()
        with pytest.raises(CloudLinkError, match="closed"):
            link.send_partial(_partial())
        with pytest.raises(CloudLinkError, match="closed"):
            link.poll_refinement(cluster_id="A")
    finally:
        srv.stop()


# --------------------------------------------------------------------------- #
# Concurrent use — multiple clusters POSTing in parallel
# --------------------------------------------------------------------------- #

def test_concurrent_send_partial_from_two_clusters():
    """Sprint 2 multi-cluster scenario — two HFLHostClusters report."""
    srv = MockTier3Server()
    srv.start()
    try:
        link_a = HTTPCloudLink(base_url=srv.url)
        link_b = HTTPCloudLink(base_url=srv.url)

        def _fire(link: HTTPCloudLink, cluster_id: str) -> None:
            for r in range(3):
                link.send_partial(
                    ClusterPartialUpload(
                        cluster_id=cluster_id,
                        cluster_round=r,
                        theta_disc=[np.zeros((2,), dtype=np.float32)],
                    )
                )

        ta = threading.Thread(target=_fire, args=(link_a, "A"), daemon=True)
        tb = threading.Thread(target=_fire, args=(link_b, "B"), daemon=True)
        ta.start(); tb.start()
        ta.join(timeout=5.0); tb.join(timeout=5.0)

        seen = srv.received_partials
        assert len(seen) == 6
        assert sum(1 for p in seen if p.cluster_id == "A") == 3
        assert sum(1 for p in seen if p.cluster_id == "B") == 3
    finally:
        srv.stop()
