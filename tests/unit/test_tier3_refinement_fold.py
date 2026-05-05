"""Phase 7 — Tier-3 refinement-fold unit tests.

Covers the wiring added in chunk P2:

* ``GeneratorHost.apply_tier3_gen_refinement`` is part of the protocol.
* ``StubGeneratorHost`` implements the fold + the out-of-order guard.
* The cluster service's poll path actually calls the fold when
  ``HTTPCloudLink.poll_refinement`` returns a non-None refinement
  (verified with a fake cloud link that doesn't need a real socket).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from hermes.cluster import DeviceRegistry
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.processes.cluster import ClusterService
from hermes.processes.config import ClusterConfig
from hermes.types import GeneratorRefinement


# --------------------------------------------------------------------------- #
# StubGeneratorHost.apply_tier3_gen_refinement
# --------------------------------------------------------------------------- #

def test_stub_generator_applies_first_refinement():
    gen = StubGeneratorHost(disc_weights=[np.zeros(4, dtype=np.float32)])
    fresh = [np.ones(4, dtype=np.float32) * 0.5]
    gen.apply_tier3_gen_refinement(fresh, refinement_round=3)
    assert gen.last_refinement_round == 3
    assert len(gen.gen_weights) == 1
    np.testing.assert_array_equal(gen.gen_weights[0], fresh[0])


def test_stub_generator_ignores_out_of_order_older_refinement():
    """An older refinement_round must not clobber a newer one already applied."""
    gen = StubGeneratorHost(disc_weights=[np.zeros(4, dtype=np.float32)])
    newer = [np.ones(4, dtype=np.float32) * 0.7]
    older = [np.ones(4, dtype=np.float32) * 0.1]

    gen.apply_tier3_gen_refinement(newer, refinement_round=10)
    gen.apply_tier3_gen_refinement(older, refinement_round=5)

    assert gen.last_refinement_round == 10, "older refinement clobbered newer"
    np.testing.assert_array_equal(gen.gen_weights[0], newer[0])


def test_stub_generator_copies_input_to_avoid_aliasing():
    """The stub must not retain a reference into the caller's array."""
    gen = StubGeneratorHost(disc_weights=[np.zeros(4, dtype=np.float32)])
    src = [np.ones(4, dtype=np.float32) * 0.5]
    gen.apply_tier3_gen_refinement(src, refinement_round=1)

    # Mutate the source array — the stub's stored copy must not change.
    src[0][:] = 9.0
    assert not np.array_equal(gen.gen_weights[0], src[0]), \
        "stub stored a reference; caller mutation leaked in"


# --------------------------------------------------------------------------- #
# ClusterService._poll_tier3_if_wired calls the fold
# --------------------------------------------------------------------------- #

class _FakeCloudLink:
    """Tier-3 stub used to drive the cluster's poll path without a socket."""

    def __init__(self, scripted_refinements: list):
        # Sequence of `Optional[GeneratorRefinement]` — pop one per poll.
        self._scripted = list(scripted_refinements)
        self.poll_calls = 0

    def poll_refinement(
        self, cluster_id: str, *, timeout_s: float = 5.0
    ) -> Optional[GeneratorRefinement]:
        self.poll_calls += 1
        if not self._scripted:
            return None
        return self._scripted.pop(0)

    def close(self) -> None:
        return


def _build_cluster_service_with_fake_cloud(
    cloud: _FakeCloudLink,
) -> ClusterService:
    cfg = ClusterConfig(
        cluster_id="cluster-tier3-test",
        dock_host="127.0.0.1",
        dock_port=0,
        synth_batch_size=2,
        min_participation=1,
    )
    svc = ClusterService(cfg)
    svc.cloud = cloud  # bypass the URL-based default
    return svc


def test_cluster_service_applies_refinement_when_poll_returns_one():
    refined = GeneratorRefinement(
        refinement_round=4,
        theta_gen=[np.ones(4, dtype=np.float32) * 0.42],
        notes="phase-7-test",
    )
    cloud = _FakeCloudLink(scripted_refinements=[refined])
    svc = _build_cluster_service_with_fake_cloud(cloud)
    try:
        # Sanity: generator hasn't seen any refinement yet.
        assert svc.generator.last_refinement_round == -1

        svc._poll_tier3_if_wired()

        assert cloud.poll_calls == 1
        assert svc.generator.last_refinement_round == 4
        np.testing.assert_array_equal(
            svc.generator.gen_weights[0], refined.theta_gen[0],
        )
        # Counter bumped by one fold.
        assert svc.metrics.counter_value("tier3_refinements_applied") == 1
    finally:
        svc.shutdown()


def test_cluster_service_no_op_when_tier3_returns_none():
    cloud = _FakeCloudLink(scripted_refinements=[None])
    svc = _build_cluster_service_with_fake_cloud(cloud)
    try:
        svc._poll_tier3_if_wired()
        assert cloud.poll_calls == 1
        # No refinement applied.
        assert svc.generator.last_refinement_round == -1
        assert svc.metrics.counter_value("tier3_refinements_applied") == 0
    finally:
        svc.shutdown()


def test_cluster_service_records_poll_failure_without_crashing():
    class _BoomCloud:
        def __init__(self):
            self.poll_calls = 0

        def poll_refinement(self, cluster_id, *, timeout_s=5.0):
            self.poll_calls += 1
            raise RuntimeError("simulated transient")

        def close(self):
            return

    cloud = _BoomCloud()
    svc = _build_cluster_service_with_fake_cloud(cloud)
    try:
        # Must not raise — the poll path is best-effort.
        svc._poll_tier3_if_wired()
        assert cloud.poll_calls == 1
        assert svc.metrics.counter_value("tier3_poll_failures") == 1
    finally:
        svc.shutdown()
